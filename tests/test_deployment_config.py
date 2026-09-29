"""Tests for Phase 15's deployment configuration: render.yaml, the
self-contained Docker images, pinned dependencies, and the absence of any
developer-machine path from the production code path. Static checks only
-- nothing here calls Render or any cloud service.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RENDER_PATH = PROJECT_ROOT / "render.yaml"
REPO_URL = "https://github.com/mandaldhananjay248-beep/customer-retention-intelligence"
API_SERVICE = "customer-retention-api"
DASHBOARD_SERVICE = "customer-retention-dashboard"

# Everything that ends up in, or configures, the running services.
PRODUCTION_PATHS = [
    "api",
    "dashboard",
    "src",
    "configs",
    "Dockerfile.api",
    "Dockerfile.dashboard",
    "docker-compose.yml",
    "render.yaml",
    ".github",
    "requirements.txt",
    ".env.example",
]


@pytest.fixture(scope="module")
def render() -> dict:
    assert RENDER_PATH.exists(), "render.yaml is missing"
    return yaml.safe_load(RENDER_PATH.read_text())


@pytest.fixture(scope="module")
def services(render: dict) -> dict[str, dict]:
    return {service["name"]: service for service in render["services"]}


def _env(service: dict) -> dict[str, dict]:
    return {var["key"]: var for var in service.get("envVars", [])}


def _exposed_port(dockerfile: str) -> str:
    return re.search(r"^EXPOSE (\d+)$", (PROJECT_ROOT / dockerfile).read_text(), re.MULTILINE).group(1)


# ---------------------------------------------------------------------------
# render.yaml
# ---------------------------------------------------------------------------

def test_render_defines_exactly_the_api_and_dashboard_web_services(render: dict, services: dict) -> None:
    assert set(render) == {"services"}
    assert set(services) == {API_SERVICE, DASHBOARD_SERVICE}
    for service in services.values():
        assert service["type"] == "web"
        assert service["runtime"] == "docker"


def test_render_services_deploy_from_main_of_this_repo_after_ci_passes(services: dict) -> None:
    for service in services.values():
        assert service["repo"] == REPO_URL
        assert service["branch"] == "main"
        assert service["autoDeployTrigger"] == "checksPass"


@pytest.mark.parametrize(("name", "dockerfile"), [(API_SERVICE, "Dockerfile.api"), (DASHBOARD_SERVICE, "Dockerfile.dashboard")])
def test_render_services_build_the_existing_dockerfiles(services: dict, name: str, dockerfile: str) -> None:
    service = services[name]
    assert service["dockerfilePath"] == f"./{dockerfile}"
    assert service["dockerContext"] == "."
    assert (PROJECT_ROOT / dockerfile).exists()


@pytest.mark.parametrize(("name", "dockerfile"), [(API_SERVICE, "Dockerfile.api"), (DASHBOARD_SERVICE, "Dockerfile.dashboard")])
def test_render_port_matches_the_port_the_image_serves(services: dict, name: str, dockerfile: str) -> None:
    assert _env(services[name])["PORT"]["value"] == _exposed_port(dockerfile)


def test_render_health_checks_use_the_existing_endpoints(services: dict) -> None:
    assert services[API_SERVICE]["healthCheckPath"] == "/health"
    assert services[DASHBOARD_SERVICE]["healthCheckPath"] == "/_stcore/health"


def test_render_plans_match_measured_memory_needs(services: dict) -> None:
    """The API peaks at ~860 MB while loading (above 512 MB instances); the
    dashboard stays on the Free plan.
    """
    assert services[API_SERVICE]["plan"] == "1c-2g"
    assert services[DASHBOARD_SERVICE]["plan"] == "free"


def test_render_services_share_one_region_for_private_networking(services: dict) -> None:
    assert services[API_SERVICE]["region"] == services[DASHBOARD_SERVICE]["region"]


def test_render_dashboard_reaches_the_api_over_the_private_network(services: dict) -> None:
    env = _env(services[DASHBOARD_SERVICE])
    assert env["API_HOSTPORT"]["fromService"] == {"type": "web", "name": API_SERVICE, "property": "hostport"}
    assert "API_BASE_URL" not in env


def test_render_yaml_sets_only_non_secret_configuration(services: dict) -> None:
    """The app needs no secrets: the only variables are the port pins and
    the API's private address -- nothing with an inline credential value.
    """
    assert set(_env(services[API_SERVICE])) == {"PORT"}
    assert set(_env(services[DASHBOARD_SERVICE])) == {"PORT", "API_HOSTPORT"}
    text = RENDER_PATH.read_text()
    assert "-----BEGIN" not in text
    assert not re.search(r"AKIA[0-9A-Z]{16}|rnd_[A-Za-z0-9]{20,}|gh[opsu]_[A-Za-z0-9]{20,}", text)


def test_render_yaml_never_points_a_service_at_localhost(render: dict) -> None:
    values = yaml.safe_dump(render)  # parsed configuration only, not comments
    assert "localhost" not in values and "127.0.0.1" not in values


# ---------------------------------------------------------------------------
# Self-contained, portable images
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dockerfile", ["Dockerfile.api", "Dockerfile.dashboard"])
def test_images_regenerate_the_feature_dataset_at_build_time(dockerfile: str) -> None:
    text = (PROJECT_ROOT / dockerfile).read_text()
    regenerate = text.index("RUN python -m src.data.regenerate_feature_dataset --prune")
    assert text.index("COPY . .") < regenerate < text.index("USER app")


def test_compose_uses_no_host_bind_mounts() -> None:
    compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text())
    for name, service in compose["services"].items():
        assert "volumes" not in service, f"{name} must not depend on host files"


def test_packaged_model_is_in_the_build_context_but_local_registry_is_not() -> None:
    lines = [line.strip() for line in (PROJECT_ROOT / ".dockerignore").read_text().splitlines()]
    for excluded in ("mlruns/", "mlartifacts/", "*.db", "**/*.db", "**/*.pkl", "**/*.joblib", "data/processed/"):
        assert excluded in lines
    assert "!models/registered/*/*/model.pkl" in lines
    assert lines.index("!models/registered/*/*/model.pkl") > lines.index("**/*.pkl")


def test_git_tracks_only_the_packaged_model_and_no_data_or_registry() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    pickles = [p for p in tracked if p.endswith((".pkl", ".joblib"))]
    assert pickles == ["models/registered/customer-churn-logistic-regression/1/model.pkl"]
    for path in tracked:
        assert not path.endswith((".db", ".parquet")), path
        assert not path.startswith(("mlartifacts/", "mlruns/")), path
        if path.startswith("data/"):
            assert path.endswith(".gitkeep"), path


def test_production_code_path_has_no_developer_machine_paths() -> None:
    offenders = []
    for entry in PRODUCTION_PATHS:
        root = PROJECT_ROOT / entry
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
        for path in files:
            if re.search(r"/Users/[^/\s]+/|/home/[^/\s]+/|[A-Z]:\\\\Users", path.read_text(errors="ignore")):
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def _requirements(path: str) -> list[str]:
    lines = (PROJECT_ROOT / path).read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def test_runtime_requirements_are_all_pinned() -> None:
    for requirement in _requirements("requirements.txt"):
        assert re.fullmatch(r"[A-Za-z0-9_.\-]+(\[[a-z,]+\])?==[\w.]+", requirement), requirement


def test_runtime_requirements_exclude_dev_tools() -> None:
    names = {re.split(r"[=\[]", r)[0].lower() for r in _requirements("requirements.txt")}
    assert not names & {"pytest", "jupyter", "ipykernel"}


def test_dev_requirements_extend_runtime_requirements() -> None:
    dev = _requirements("requirements-dev.txt")
    assert dev[0] == "-r requirements.txt"
    assert any(r.startswith("pytest==") for r in dev)


def test_env_example_holds_no_values_that_look_like_secrets() -> None:
    for line in _requirements(".env.example"):
        key, _, value = line.partition("=")
        if any(word in key.upper() for word in ("KEY", "SECRET", "TOKEN", "PASSWORD")):
            assert value == "", f"{key} must be an empty placeholder"
