"""Phase 15: headless end-to-end check of the Streamlit Community Cloud entry
point (streamlit_app.py), the way Community Cloud will run it.

    python scripts/check_streamlit_app.py

Runs the real app script with Streamlit's AppTest (part of streamlit itself,
so only the runtime requirements.txt is needed): the dataset is regenerated
first if missing, all four tabs must render without errors, and analyzing
C000001's latest snapshot must return the verified 63.3% churn score and a
recommendation. Exits non-zero on the first failure.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

ENTRYPOINT_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"
EXPECTED_TABS = ["Overview", "Customer Analysis", "Model Information", "Monitoring"]


def check(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL  {message}")
        sys.exit(1)
    print(f"ok    {message}")


def main() -> int:
    started = time.monotonic()
    at = AppTest.from_file(str(ENTRYPOINT_PATH), default_timeout=600).run()
    check(not at.exception, f"first run without exceptions ({time.monotonic() - started:.1f}s) {list(at.exception)}")
    check([tab.label for tab in at.tabs] == EXPECTED_TABS, f"tabs: {[tab.label for tab in at.tabs]}")
    check(not at.error, f"no error messages {[e.value for e in at.error]}")
    customer = at.sidebar.selectbox[0].value
    snapshot = at.sidebar.selectbox[1].value
    check((customer, str(snapshot)) == ("C000001", "2026-05-31"), f"default selection {customer} / {snapshot}")

    at = next(b for b in at.button if b.label == "Analyze Customer").click().run()
    check(not at.exception, f"analysis without exceptions {list(at.exception)}")
    check(not at.error, f"no error messages {[e.value for e in at.error]}")
    metrics = {m.label: m.value for m in at.metric}
    check(metrics.get("Model Churn Score") == "63.3%", f"C000001 churn score {metrics.get('Model Churn Score')}")
    check(metrics.get("Model Version") == "1", f"model version {metrics.get('Model Version')}")
    actions = [str(md.value) for md in at.markdown if "Recommended Action:" in str(md.value)]
    check(bool(actions), f"recommendation shown: {actions}")
    check(metrics.get("Drift Monitoring") == "Active", f"monitoring tab: drift monitoring {metrics.get('Drift Monitoring')}")
    print("Streamlit app check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
