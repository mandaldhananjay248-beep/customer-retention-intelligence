"""Suite-wide pytest configuration.

Phase 15 skip policy: several modules skip themselves when a generated
artifact (the Phase 7 model, the Phase 4 feature dataset) is missing, which
is convenient locally but would let CI "pass" while silently running only
part of the suite. When FAIL_ON_SKIPPED_TESTS=1 (set in
.github/workflows/ci.yml), any skipped test fails the session.
"""

from __future__ import annotations

import os

import pytest

FAIL_ON_SKIPPED_ENV_VAR = "FAIL_ON_SKIPPED_TESTS"


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if os.environ.get(FAIL_ON_SKIPPED_ENV_VAR) != "1":
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter is not None else []
    if skipped and exitstatus == pytest.ExitCode.OK:
        reporter.write_line(
            f"{FAIL_ON_SKIPPED_ENV_VAR}=1: {len(skipped)} test(s) skipped -- failing the run. "
            "Provision the missing artifacts instead of skipping.",
            red=True,
            bold=True,
        )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
