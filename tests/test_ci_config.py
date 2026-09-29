"""Tests for Phase 15's GitHub Actions CI (.github/workflows/ci.yml) and
the suite-wide skip policy (tests/conftest.py). Static checks only --
nothing here calls GitHub or any cloud service.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def workflow_text() -> str:
    assert CI_WORKFLOW_PATH.exists(), ".github/workflows/ci.yml is missing"
    return CI_WORKFLOW_PATH.read_text()


@pytest.fixture(scope="module")
def workflow(workflow_text: str) -> dict:
    return yaml.safe_load(workflow_text)


def _triggers(workflow: dict) -> dict:
    # PyYAML (YAML 1.1) parses the bare key `on` as boolean True.
    return workflow.get("on", workflow.get(True))


def _steps(workflow: dict, job: str) -> list[dict]:
    return workflow["jobs"][job]["steps"]


def _run_commands(workflow: dict, job: str) -> list[str]:
    return [step.get("run", "") for step in _steps(workflow, job)]


def test_ci_runs_on_push_to_main_and_on_pull_requests(workflow: dict) -> None:
    triggers = _triggers(workflow)
    assert triggers["push"]["branches"] == ["main"]
    assert "pull_request" in triggers


def test_ci_token_is_read_only_and_requests_no_cloud_identity(workflow: dict) -> None:
    assert workflow["permissions"] == {"contents": "read"}
    for job in workflow["jobs"].values():
        assert "permissions" not in job


def test_ci_uses_a_pinned_ubuntu_runner(workflow: dict) -> None:
    for job in workflow["jobs"].values():
        assert re.fullmatch(r"ubuntu-\d+\.\d+", job["runs-on"])


def test_ci_python_version_matches_the_docker_images(workflow: dict) -> None:
    setup = next(s for s in _steps(workflow, "tests") if s.get("uses", "").startswith("actions/setup-python"))
    python_version = setup["with"]["python-version"]
    for dockerfile in ("Dockerfile.api", "Dockerfile.dashboard"):
        assert f"FROM python:{python_version}-slim" in (PROJECT_ROOT / dockerfile).read_text()


def test_ci_installs_dev_requirements(workflow: dict) -> None:
    assert "pip install -r requirements-dev.txt" in _run_commands(workflow, "tests")


def test_ci_provisions_artifacts_before_running_the_suite(workflow: dict) -> None:
    commands = _run_commands(workflow, "tests")
    regenerate = commands.index("python -m src.data.regenerate_feature_dataset")
    provision = commands.index("python -m src.models.provision_tuned_model")
    pytest_step = next(i for i, c in enumerate(commands) if c.startswith("python -m pytest"))
    assert regenerate < pytest_step and provision < pytest_step


def test_ci_runs_the_full_suite_with_skips_failing_the_run(workflow: dict) -> None:
    step = next(s for s in _steps(workflow, "tests") if s.get("run", "").startswith("python -m pytest"))
    assert step["run"].startswith("python -m pytest -q")
    assert "-k " not in step["run"] and "--deselect" not in step["run"] and "--ignore" not in step["run"]
    assert step["env"]["FAIL_ON_SKIPPED_TESTS"] == "1"


def test_ci_builds_both_images_and_smoke_tests_them(workflow: dict) -> None:
    commands = "\n".join(_run_commands(workflow, "docker"))
    assert "docker compose build" in commands
    assert "docker compose up -d --wait" in commands
    assert "scripts/smoke_test.py --api-url http://localhost:8000 --dashboard-url http://localhost:8501" in commands
    assert "docker compose down" in commands


def test_ci_does_not_push_images_or_deploy(workflow_text: str) -> None:
    for forbidden in ("docker push", "docker login", "render.com/deploy", "api.render.com"):
        assert forbidden not in workflow_text


def test_ci_uses_no_secrets_or_cloud_credentials(workflow_text: str) -> None:
    assert "secrets." not in workflow_text
    for forbidden in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "RENDER_API_KEY", "id-token"):
        assert forbidden not in workflow_text
    assert not re.search(r"AKIA[0-9A-Z]{16}", workflow_text)


@pytest.mark.parametrize(
    ("env_value", "expected_exit_code"),
    [("1", pytest.ExitCode.TESTS_FAILED), (None, pytest.ExitCode.OK)],
)
def test_skip_policy_fails_a_run_with_skips_only_when_enabled(
    tmp_path: Path, env_value: str | None, expected_exit_code: int
) -> None:
    """Runs a throwaway suite (one pass, one skip) under the project's
    conftest hook in a subprocess.
    """
    (tmp_path / "conftest.py").write_text(
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('project_conftest', {str(PROJECT_ROOT / 'tests' / 'conftest.py')!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "pytest_sessionfinish = module.pytest_sessionfinish\n"
    )
    (tmp_path / "test_sample.py").write_text(
        "import pytest\n\n"
        "def test_passes():\n    assert True\n\n"
        "def test_skipped():\n    pytest.skip('artifact missing')\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "FAIL_ON_SKIPPED_TESTS"}
    if env_value is not None:
        env["FAIL_ON_SKIPPED_TESTS"] = env_value
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == expected_exit_code, result.stdout + result.stderr
