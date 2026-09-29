"""Tests for Phase 15's deployment configuration: Streamlit Community
Cloud readiness, the self-contained Docker images, pinned dependencies, and
the absence of any developer-machine path from the production code path.
Static checks only -- nothing here calls any cloud service.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Everything that ends up in, or configures, the running services.
PRODUCTION_PATHS = [
    "api",
    "dashboard",
    "src",
    "configs",
    "Dockerfile.api",
    "Dockerfile.dashboard",
    "docker-compose.yml",
    "streamlit_app.py",
    ".github",
    "requirements.txt",
    ".env.example",
]


# ---------------------------------------------------------------------------
# Streamlit Community Cloud
# ---------------------------------------------------------------------------

def test_streamlit_cloud_entrypoint_is_at_the_repo_root() -> None:
    """Community Cloud's default entrypoint name; it looks for the dependency
    file next to the entrypoint first, then at the repo root.
    """
    assert (PROJECT_ROOT / "streamlit_app.py").is_file()
    assert (PROJECT_ROOT / "requirements.txt").is_file()


def test_no_competing_dependency_file_overrides_requirements_txt() -> None:
    """Community Cloud uses the first of uv.lock, Pipfile, environment.yml,
    requirements.txt, pyproject.toml it finds -- only requirements.txt may exist.
    """
    for name in ("uv.lock", "Pipfile", "environment.yml", "environment.yaml", "pyproject.toml"):
        assert not (PROJECT_ROOT / name).exists(), name


def test_streamlit_cloud_needs_no_secrets_or_system_packages() -> None:
    assert not (PROJECT_ROOT / ".streamlit" / "secrets.toml").exists()
    assert not (PROJECT_ROOT / "packages.txt").exists()


def test_no_render_deployment_configuration_remains() -> None:
    assert not (PROJECT_ROOT / "render.yaml").exists()
    tracked = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    for path in tracked:
        if path.endswith((".py", ".yml", ".yaml", ".toml", ".txt", ".example")) and path != "tests/test_deployment_config.py":
            text = (PROJECT_ROOT / path).read_text(errors="ignore")
            assert "onrender.com" not in text and "API_HOSTPORT" not in text, path


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
