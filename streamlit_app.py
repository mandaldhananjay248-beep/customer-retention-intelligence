"""Streamlit Community Cloud entry point (Phase 15).

Runs the existing dashboard (dashboard/app.py) as a single process: the
dashboard's API client sends its usual requests to the existing FastAPI app
in-process (DASHBOARD_API_MODE=inprocess, see api/inprocess.py) instead of
to a separately hosted API. Same routes, same packaged model (SHA-256
verified), same SHAP, recommendation and monitoring code.

Community Cloud has no build step, so on first start this regenerates
data/processed/customer_features.parquet from the committed seeded pipeline
(~30 s, fingerprint verified) if it isn't there yet. Nothing is retrained.

Local Docker / development keeps using `streamlit run dashboard/app.py`
against the FastAPI service over HTTP.
"""

from __future__ import annotations

import os

import streamlit as st

os.environ.setdefault("DASHBOARD_API_MODE", "inprocess")

from dashboard import app as dashboard_app  # noqa: E402
from src.data.regenerate_feature_dataset import ensure_feature_dataset  # noqa: E402


# No spinner: dashboard_app.main() must issue the page's first Streamlit
# command (st.set_page_config).
@st.cache_resource(show_spinner=False)
def _prepare_feature_dataset() -> str:
    return str(ensure_feature_dataset())


_prepare_feature_dataset()
dashboard_app.main()
