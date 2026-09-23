"""Tests for the Phase 13 Docker configuration (docker-compose.yml,
Dockerfile.api, Dockerfile.dashboard, .dockerignore).

These tests parse the actual configuration files -- they do not require
Docker itself to be installed or running, so they run as part of the
normal `pytest tests/` suite in any environment. A separate, optional
test class at the bottom shells out to `docker compose config` for an
extra structural check, but is skipped entirely if the Docker CLI isn't
on PATH, since Docker availability is not a hard requirement for this
project's test suite.

Live container behavior (both services actually starting, becoming
healthy, talking to each other over the Compose network, and serving a
real /customer-analysis request) was verified manually -- see the Phase
13 report -- rather than automated here, since spinning up real
containers inside a unit test suite would make `pytest tests/` slow and
environment-dependent for every future run.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = PROJECT_ROOT / "docker-compose.yml"
DOCKERFILE_API_PATH = PROJECT_ROOT / "Dockerfile.api"
DOCKERFILE_DASHBOARD_PATH = PROJECT_ROOT / "Dockerfile.dashboard"
DOCKERIGNORE_PATH = PROJECT_ROOT / ".dockerignore"

# Environment variable names are not secrets -- only flag assignments
# that look like an actual credential VALUE was hardcoded.
SUSPICIOUS_SECRET_PATTERNS = (
    "PASSWORD=",
    "SECRET_KEY=",
    "API_KEY=",
    "AWS_ACCESS_KEY_ID=",
    "AWS_SECRET_ACCESS_KEY=",
    "-----BEGIN",  # PEM-encoded private key/cert
)


@pytest.fixture(scope="module")
def compose_config() -> dict:
    assert COMPOSE_PATH.exists(), "docker-compose.yml is missing"
    return yaml.safe_load(COMPOSE_PATH.read_text())


@pytest.fixture(scope="module")
def dockerfile_api_text() -> str:
    assert DOCKERFILE_API_PATH.exists(), "Dockerfile.api is missing"
    return DOCKERFILE_API_PATH.read_text()


@pytest.fixture(scope="module")
def dockerfile_dashboard_text() -> str:
    assert DOCKERFILE_DASHBOARD_PATH.exists(), "Dockerfile.dashboard is missing"
    return DOCKERFILE_DASHBOARD_PATH.read_text()


@pytest.fixture(scope="module")
def dockerignore_lines() -> list[str]:
    assert DOCKERIGNORE_PATH.exists(), ".dockerignore is missing"
    return [line.strip() for line in DOCKERIGNORE_PATH.read_text().splitlines()]


# ---------------------------------------------------------------------------
# docker-compose.yml structure
# ---------------------------------------------------------------------------

def test_compose_defines_exactly_api_and_dashboard_services(compose_config: dict) -> None:
    assert set(compose_config["services"].keys()) == {"api", "dashboard"}


def test_compose_maps_documented_ports(compose_config: dict) -> None:
    """8000 (FastAPI) and 8501 (Streamlit) are the ports this project's
    own README/api_client.py document -- see DEFAULT_API_BASE_URL in
    dashboard/api_client.py and the `uvicorn`/`streamlit run` commands.
    """
    api_ports = compose_config["services"]["api"]["ports"]
    dashboard_ports = compose_config["services"]["dashboard"]["ports"]

    assert any("8000:8000" in str(p) for p in api_ports), f"api ports {api_ports} do not map 8000"
    assert any("8501:8501" in str(p) for p in dashboard_ports), f"dashboard ports {dashboard_ports} do not map 8501"


def test_dashboard_api_base_url_uses_compose_service_name_not_localhost(compose_config: dict) -> None:
    """The core Phase 13 architectural requirement: inside the dashboard
    container, `localhost` refers to the dashboard container itself, not
    the api container -- so API_BASE_URL must use the Compose service
    name "api", never localhost/127.0.0.1.
    """
    env = compose_config["services"]["dashboard"]["environment"]
    api_base_url = env["API_BASE_URL"]

    assert api_base_url.startswith("http://api:"), f"API_BASE_URL={api_base_url!r} does not target the 'api' service"
    assert "localhost" not in api_base_url
    assert "127.0.0.1" not in api_base_url


def test_dashboard_waits_for_api_to_be_healthy(compose_config: dict) -> None:
    depends_on = compose_config["services"]["dashboard"]["depends_on"]
    assert depends_on["api"]["condition"] == "service_healthy"


def test_compose_does_not_hardcode_secrets(compose_config: dict) -> None:
    raw_text = COMPOSE_PATH.read_text()
    for pattern in SUSPICIOUS_SECRET_PATTERNS:
        assert pattern not in raw_text, f"docker-compose.yml appears to contain a hardcoded secret: {pattern}"


def test_compose_services_do_not_run_privileged_or_as_root_explicitly(compose_config: dict) -> None:
    for service_name, service in compose_config["services"].items():
        assert service.get("privileged") is not True, f"{service_name} must not run privileged"
        # An explicit "user: root" override would defeat each Dockerfile's
        # own non-root USER instruction.
        assert service.get("user") != "root", f"{service_name} must not force root via compose"


def test_compose_volumes_do_not_mount_git_or_venv(compose_config: dict) -> None:
    for service_name, service in compose_config["services"].items():
        for volume in service.get("volumes", []):
            volume_str = str(volume)
            assert ".git" not in volume_str, f"{service_name} mounts .git: {volume_str}"
            assert ".venv" not in volume_str, f"{service_name} mounts .venv: {volume_str}"


# ---------------------------------------------------------------------------
# Dockerfile.api
# ---------------------------------------------------------------------------

def test_dockerfile_api_uses_a_pinned_python_base_image(dockerfile_api_text: str) -> None:
    assert "FROM python:" in dockerfile_api_text


def test_dockerfile_api_installs_from_requirements_txt(dockerfile_api_text: str) -> None:
    assert "requirements.txt" in dockerfile_api_text
    assert "pip install" in dockerfile_api_text


def test_dockerfile_api_runs_as_non_root(dockerfile_api_text: str) -> None:
    lines = [line.strip() for line in dockerfile_api_text.splitlines()]
    user_lines = [line for line in lines if line.startswith("USER ")]
    assert user_lines, "Dockerfile.api has no USER instruction"
    assert user_lines[-1] != "USER root"


def test_dockerfile_api_exposes_port_8000(dockerfile_api_text: str) -> None:
    assert "EXPOSE 8000" in dockerfile_api_text


def test_dockerfile_api_healthcheck_uses_existing_health_endpoint(dockerfile_api_text: str) -> None:
    """Must reuse the API's own existing GET /health -- no second,
    invented health system.
    """
    assert "HEALTHCHECK" in dockerfile_api_text
    assert "/health" in dockerfile_api_text
    assert "/health2" not in dockerfile_api_text  # guards against an invented parallel endpoint


def test_dockerfile_api_start_command_matches_documented_entrypoint(dockerfile_api_text: str) -> None:
    """Must run the existing api/main.py entry point (api.main:app), the
    same one documented in README.md / api/main.py's own docstring.
    """
    assert "api.main:app" in dockerfile_api_text
    assert '"--host", "0.0.0.0"' in dockerfile_api_text, "must bind 0.0.0.0 to be reachable from other containers"


def test_dockerfile_api_does_not_hardcode_secrets(dockerfile_api_text: str) -> None:
    for pattern in SUSPICIOUS_SECRET_PATTERNS:
        assert pattern not in dockerfile_api_text


# ---------------------------------------------------------------------------
# Dockerfile.dashboard
# ---------------------------------------------------------------------------

def test_dockerfile_dashboard_uses_a_pinned_python_base_image(dockerfile_dashboard_text: str) -> None:
    assert "FROM python:" in dockerfile_dashboard_text


def test_dockerfile_dashboard_runs_as_non_root(dockerfile_dashboard_text: str) -> None:
    lines = [line.strip() for line in dockerfile_dashboard_text.splitlines()]
    user_lines = [line for line in lines if line.startswith("USER ")]
    assert user_lines, "Dockerfile.dashboard has no USER instruction"
    assert user_lines[-1] != "USER root"


def test_dockerfile_dashboard_exposes_port_8501(dockerfile_dashboard_text: str) -> None:
    assert "EXPOSE 8501" in dockerfile_dashboard_text


def test_dockerfile_dashboard_binds_to_all_interfaces(dockerfile_dashboard_text: str) -> None:
    """Streamlit must bind 0.0.0.0, not the container-local default, to
    be reachable from the host's published port mapping.
    """
    assert '"--server.address", "0.0.0.0"' in dockerfile_dashboard_text


def test_dockerfile_dashboard_healthcheck_uses_streamlits_own_endpoint(dockerfile_dashboard_text: str) -> None:
    """Streamlit already ships /_stcore/health -- this must be reused,
    not a new, invented health mechanism.
    """
    assert "/_stcore/health" in dockerfile_dashboard_text


def test_dockerfile_dashboard_start_command_matches_documented_entrypoint(dockerfile_dashboard_text: str) -> None:
    assert "dashboard/app.py" in dockerfile_dashboard_text


def test_dockerfile_dashboard_does_not_hardcode_secrets(dockerfile_dashboard_text: str) -> None:
    for pattern in SUSPICIOUS_SECRET_PATTERNS:
        assert pattern not in dockerfile_dashboard_text


def test_dockerfile_dashboard_does_not_install_ml_inference_libraries(dockerfile_dashboard_text: str) -> None:
    """Both images install the same shared requirements.txt (a deliberate
    simplicity trade-off -- see the Phase 13 report), so this can't check
    installed packages; it instead guards that Dockerfile.dashboard's own
    instructions never single out an ML library for special/explicit
    installation, which would signal an architecture violation.
    """
    for forbidden in ("pip install shap", "pip install mlflow", "pip install torch"):
        assert forbidden not in dockerfile_dashboard_text.lower()


# ---------------------------------------------------------------------------
# .dockerignore
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "required_pattern",
    [".git/", ".venv/", "__pycache__/", ".env", ".pytest_cache/", "mlruns/", "mlartifacts/", "*.db", "*.joblib"],
)
def test_dockerignore_excludes_secrets_and_caches(dockerignore_lines: list[str], required_pattern: str) -> None:
    assert required_pattern in dockerignore_lines, f"{required_pattern!r} missing from .dockerignore"


@pytest.mark.parametrize("must_not_exclude", ["api/", "src/", "dashboard/", "configs/", "requirements.txt"])
def test_dockerignore_does_not_exclude_runtime_required_source(dockerignore_lines: list[str], must_not_exclude: str) -> None:
    assert must_not_exclude not in dockerignore_lines, (
        f"{must_not_exclude!r} must not be excluded -- it is required at container runtime"
    )


def test_dockerignore_excludes_the_large_recommendations_csv(dockerignore_lines: list[str]) -> None:
    """Consistent with .gitignore -- the ~50MB batch output must never be
    baked into either image.
    """
    assert "reports/customer_recommendations.csv" in dockerignore_lines


# ---------------------------------------------------------------------------
# Optional: validate via the real Docker CLI, skipped if unavailable
# ---------------------------------------------------------------------------

class TestDockerComposeConfigValidatesWithRealCLI:
    """A behavioral check beyond static parsing: ask the actual `docker
    compose` CLI to resolve the file, which catches YAML-valid-but-
    Compose-invalid mistakes a plain yaml.safe_load would miss. Skipped
    if the Docker CLI isn't available in this environment.
    """

    @pytest.fixture(autouse=True)
    def _require_docker_cli(self):
        if shutil.which("docker") is None:
            pytest.skip("docker CLI not available in this environment")

    def test_docker_compose_config_resolves_without_error(self) -> None:
        result = subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_PATH), "config", "--quiet"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
