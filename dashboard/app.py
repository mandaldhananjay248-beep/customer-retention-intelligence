"""Customer Retention Intelligence -- Streamlit business dashboard (Phase 12).

Architecture:

    Business User -> Streamlit Dashboard -> HTTP -> FastAPI (Phase 11)
        -> MLflow registered model (Phase 10) -> SHAP (Phase 8) / Next-Best-Action (Phase 9)

This dashboard NEVER loads the .joblib or MLflow model directly, computes
a churn probability itself, calls SHAP directly, or re-implements the
Phase 9 recommendation rules. Every prediction, explanation, and
recommendation shown here comes from a FastAPI response
(`dashboard/api_client.py`). The dashboard only reads
`data/processed/customer_features.parquet` to let the user pick a real
customer/snapshot and to build the API request payload from that row's
actual values -- it never fabricates a feature value.

Run with:
    uvicorn api.main:app --reload      # in one terminal
    streamlit run dashboard/app.py     # in another

Configuration:
    API_BASE_URL   Base URL of the FastAPI service (default http://localhost:8000)
    DASHBOARD_API_MODE  "inprocess" to call the FastAPI app in this process
                        (set by streamlit_app.py for Streamlit Community Cloud)

Note on structure: Streamlit executes this file as `__main__` when run via
`streamlit run`. All actual page rendering lives inside main(), called
only under `if __name__ == "__main__"`, so this module can be imported
safely (e.g. by tests/test_dashboard.py) without launching a page render
or making a network call at import time -- only the small, pure/cached
data-loading functions below are module-level.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard import api_client
from dashboard.components import (
    render_action_distribution_chart,
    render_api_status,
    render_disclaimer,
    render_error_banner,
    render_drift_section,
    render_model_info_sidebar,
    render_monitoring_health_summary,
    render_prediction_metrics_section,
    render_prediction_summary,
    render_recommendation_section,
    render_request_metrics_section,
    render_risk_distribution_chart,
    render_shap_section,
    render_supporting_features_table,
)
from dashboard.styles import CHURN_SCORE_NOTE, DISCLAIMER_TEXT, PAGE_SUBTITLE, PAGE_TITLE, inject_base_styles

FEATURE_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "customer_features.parquet"
RECOMMENDATION_SUMMARY_PATH = PROJECT_ROOT / "reports" / "recommendation_summary.csv"
TUNING_RESULTS_PATH = PROJECT_ROOT / "reports" / "tuning_results.csv"

HEALTH_CACHE_TTL_SECONDS = 15
MONITORING_CACHE_TTL_SECONDS = 5
ANALYSIS_TOP_K = 5


# ---------------------------------------------------------------------------
# Pure / cached data access -- module level, safe to import in tests.
# Loaded once and reused across reruns; never the 55MB
# reports/customer_recommendations.csv.
# ---------------------------------------------------------------------------

def _load_customer_features_uncached() -> pd.DataFrame:
    if not FEATURE_DATASET_PATH.exists():
        raise FileNotFoundError(f"Feature dataset not found: {FEATURE_DATASET_PATH}")
    # Same frame pd.read_parquet returns, but self_destruct frees each Arrow
    # column as it is converted, cutting the load's peak memory by ~100MB
    # (Phase 15, measured under a 512MB memory limit).
    return pq.read_table(FEATURE_DATASET_PATH).to_pandas(self_destruct=True, split_blocks=True)


# cache_resource, not cache_data: cache_data keeps a pickled copy of this
# ~150MB frame and hands every rerun a fresh unpickled copy, which peaks
# far above a 512MB memory limit (Phase 15, measured). The app
# only ever filters this frame (each filter builds a new frame), never
# mutates it, so one shared read-only instance is safe.
@st.cache_resource(show_spinner="Loading customer feature dataset...")
def load_customer_features() -> pd.DataFrame:
    return _load_customer_features_uncached()


def _load_recommendation_summary_uncached() -> pd.DataFrame | None:
    if not RECOMMENDATION_SUMMARY_PATH.exists():
        return None
    return pd.read_csv(RECOMMENDATION_SUMMARY_PATH)


@st.cache_data(show_spinner=False)
def load_recommendation_summary() -> pd.DataFrame | None:
    return _load_recommendation_summary_uncached()


def _load_tuning_report_row_uncached() -> dict | None:
    if not TUNING_RESULTS_PATH.exists():
        return None
    tuning_results = pd.read_csv(TUNING_RESULTS_PATH)
    selected = tuning_results[tuning_results["selected"] == True]  # noqa: E712
    if selected.empty:
        return None
    return selected.iloc[0].to_dict()


@st.cache_data(show_spinner=False)
def load_tuning_report_row() -> dict | None:
    return _load_tuning_report_row_uncached()


@st.cache_data(ttl=HEALTH_CACHE_TTL_SECONDS, show_spinner=False)
def cached_health() -> api_client.APIResult:
    return api_client.get_health()


@st.cache_data(ttl=HEALTH_CACHE_TTL_SECONDS, show_spinner=False)
def cached_model_info() -> api_client.APIResult:
    return api_client.get_model_info()


@st.cache_data(ttl=MONITORING_CACHE_TTL_SECONDS, show_spinner=False)
def cached_monitoring_metrics() -> api_client.APIResult:
    return api_client.get_monitoring_metrics()


@st.cache_data(ttl=MONITORING_CACHE_TTL_SECONDS, show_spinner=False)
def cached_monitoring_health() -> api_client.APIResult:
    return api_client.get_monitoring_health()


@st.cache_data(ttl=MONITORING_CACHE_TTL_SECONDS, show_spinner=False)
def cached_monitoring_drift() -> api_client.APIResult:
    return api_client.get_monitoring_drift()


def get_customer_snapshot_row(features_df: pd.DataFrame, customer_id: str, snapshot_date) -> pd.Series:
    """The one real row for (customer_id, snapshot_date). Raises
    ValueError if no such row exists -- never fabricates one.
    """
    customer_rows = features_df[features_df["customer_id"] == customer_id]
    matches = customer_rows[pd.to_datetime(customer_rows["snapshot_date"]).dt.date == snapshot_date]
    if matches.empty:
        raise ValueError(f"No feature row found for customer_id={customer_id!r}, snapshot_date={snapshot_date!r}")
    return matches.iloc[0]


# ---------------------------------------------------------------------------
# Page rendering -- only executed when run via `streamlit run dashboard/app.py`
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="📊", layout="wide")
    inject_base_styles()

    st.title(PAGE_TITLE)
    st.caption(PAGE_SUBTITLE)

    # -- Sidebar: API status, model info, customer selection --------------
    st.sidebar.header("Service Status")
    health_result = cached_health()
    render_api_status(health_result)

    model_info_result = cached_model_info()
    st.sidebar.divider()
    st.sidebar.subheader("Model")
    render_model_info_sidebar(model_info_result)

    if st.sidebar.button("Refresh status", help="Re-check API health and model info now."):
        cached_health.clear()
        cached_model_info.clear()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.header("Customer Selection")

    data_load_error: str | None = None
    try:
        features_df = load_customer_features()
    except FileNotFoundError as exc:
        features_df = pd.DataFrame()
        data_load_error = str(exc)

    selected_row = None
    if data_load_error:
        st.sidebar.error(data_load_error)
    else:
        customer_ids = sorted(features_df["customer_id"].unique().tolist())

        if "selected_customer_id" not in st.session_state:
            st.session_state["selected_customer_id"] = customer_ids[0]

        if st.sidebar.button("🔀 Pick a random customer/snapshot"):
            random_row = features_df.sample(n=1).iloc[0]
            st.session_state["selected_customer_id"] = random_row["customer_id"]
            st.session_state["selected_snapshot_date"] = pd.Timestamp(random_row["snapshot_date"]).date()

        selected_customer_id = st.sidebar.selectbox("Customer ID", customer_ids, key="selected_customer_id")

        customer_rows = features_df[features_df["customer_id"] == selected_customer_id]
        snapshot_dates = sorted(pd.to_datetime(customer_rows["snapshot_date"]).dt.date.unique().tolist())

        if st.session_state.get("selected_snapshot_date") not in snapshot_dates:
            st.session_state["selected_snapshot_date"] = snapshot_dates[-1]

        selected_snapshot_date = st.sidebar.selectbox("Snapshot Date", snapshot_dates, key="selected_snapshot_date")
        selected_row = get_customer_snapshot_row(features_df, selected_customer_id, selected_snapshot_date)

    st.sidebar.divider()
    render_disclaimer(DISCLAIMER_TEXT)

    # -- Main content -------------------------------------------------------
    overview_tab, customer_analysis_tab, model_info_tab, monitoring_tab = st.tabs(
        ["Overview", "Customer Analysis", "Model Information", "Monitoring"]
    )

    with overview_tab:
        _render_overview_tab(health_result)

    with customer_analysis_tab:
        _render_customer_analysis_tab(data_load_error, selected_row)

    with model_info_tab:
        _render_model_info_tab(model_info_result)

    with monitoring_tab:
        _render_monitoring_tab()


def _render_overview_tab(health_result: api_client.APIResult) -> None:
    st.subheader("Business Overview")
    st.info(CHURN_SCORE_NOTE)

    summary_df = load_recommendation_summary()

    if summary_df is None:
        st.warning(
            "reports/recommendation_summary.csv was not found. Run the Phase 9 "
            "next-best-action batch (src/recommendations/next_best_action.py) to generate it."
        )
        return

    total_snapshots = int(summary_df["customer_count"].sum())
    risk_totals = summary_df.groupby("risk_level")["customer_count"].sum()
    action_totals = summary_df.groupby("primary_action")["customer_count"].sum()
    high_critical = int(risk_totals.get("high", 0) + risk_totals.get("critical", 0))
    no_intervention = int(action_totals.get("No intervention", 0))
    actionable = total_snapshots - no_intervention

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Customer Snapshots", f"{total_snapshots:,}")
    kpi_cols[1].metric(
        "High + Critical Risk",
        f"{high_critical:,}",
        help="Customer/snapshot rows classified as High or Critical risk (Phase 9 business thresholds).",
    )
    kpi_cols[2].metric(
        "Recommended Interventions",
        f"{actionable:,}",
        help="Snapshots where the recommendation engine suggested an action other than 'No intervention'.",
    )
    api_healthy = health_result.ok and isinstance(health_result.data, dict) and health_result.data.get("status") == "healthy"
    kpi_cols[3].metric("API Status", "Connected" if api_healthy else "Unavailable")

    st.divider()
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("**Risk Level Distribution**")
        render_risk_distribution_chart(risk_totals.to_dict())
    with chart_col2:
        st.markdown("**Recommended Action Distribution**")
        render_action_distribution_chart(action_totals)

    st.caption(
        "Source: reports/recommendation_summary.csv, generated by the Phase 9 batch "
        "next-best-action run over the full feature dataset."
    )


def _render_customer_analysis_tab(data_load_error: str | None, selected_row: pd.Series | None) -> None:
    st.subheader("Customer Analysis")

    if data_load_error or selected_row is None:
        st.error("Customer data is unavailable, so individual analysis cannot be performed.")
        return

    st.markdown(f"**Customer ID:** {selected_row['customer_id']}")
    st.markdown(f"**Snapshot Date:** {pd.Timestamp(selected_row['snapshot_date']).date().isoformat()}")

    run_analysis = st.button("Analyze Customer", type="primary")

    analysis_key = (selected_row["customer_id"], str(selected_row["snapshot_date"]))
    if run_analysis:
        payload = api_client.build_customer_payload(selected_row)
        with st.spinner("Requesting prediction, explanation, and recommendation from the API..."):
            result = api_client.customer_analysis(payload, top_k=ANALYSIS_TOP_K)
        st.session_state["last_analysis_key"] = analysis_key
        st.session_state["last_analysis_result"] = result

    cached_result = st.session_state.get("last_analysis_result")
    cached_key = st.session_state.get("last_analysis_key")

    if cached_result is None:
        st.caption("Select a customer/snapshot in the sidebar, then click **Analyze Customer**.")
    elif cached_key != analysis_key:
        st.caption("Selection changed -- click **Analyze Customer** to refresh the analysis for this customer.")
    elif not cached_result.ok:
        render_error_banner(cached_result, "Customer analysis failed")
    else:
        analysis = cached_result.data
        st.divider()

        st.markdown("### Prediction")
        render_prediction_summary(analysis["prediction"])

        st.divider()
        st.markdown("### Recommendation")
        render_recommendation_section(analysis["recommendation"])

        st.divider()
        st.markdown("### Churn Drivers (SHAP)")
        render_shap_section(analysis["explanation"])

        st.divider()
        st.markdown("### Supporting Customer Signals")
        render_supporting_features_table(selected_row)


def _render_model_info_tab(model_info_result: api_client.APIResult) -> None:
    st.subheader("Model Information")
    st.caption(
        "The dashboard sends customer features to the FastAPI inference service. "
        "The API loads the registered MLflow model and returns the prediction."
    )

    st.markdown("#### Live Model (from `GET /model-info`)")
    if model_info_result.ok and isinstance(model_info_result.data, dict):
        info = model_info_result.data
        info_cols = st.columns(len(info))
        for col, (key, value) in zip(info_cols, info.items()):
            col.metric(key.replace("_", " ").title(), value)
    else:
        render_error_banner(model_info_result, "Model information unavailable")

    tuning_row = load_tuning_report_row()
    if tuning_row is not None:
        st.markdown("#### Training & Tuning Report")
        st.caption(
            "From reports/tuning_results.csv (Phase 7 hyperparameter tuning) -- a static report "
            "file, not a live API response."
        )
        metric_cols = st.columns(4)
        metric_cols[0].metric("Validation PR-AUC (selection metric)", f"{tuning_row['val_pr_auc']:.4f}")
        metric_cols[1].metric("Validation ROC-AUC", f"{tuning_row['val_roc_auc']:.4f}")
        metric_cols[2].metric("Precision", f"{tuning_row['precision']:.4f}")
        metric_cols[3].metric("Recall", f"{tuning_row['recall']:.4f}")
        st.caption(f"Hyperparameters: {tuning_row['hyperparameters']}")
    else:
        st.caption("No Phase 7 tuning report found at reports/tuning_results.csv.")

    render_disclaimer(DISCLAIMER_TEXT)


def _render_monitoring_tab() -> None:
    """Phase 14: everything here is read directly from GET /monitoring/*
    responses -- the dashboard never computes a metric or a drift value
    itself.
    """
    st.subheader("Monitoring & Observability")
    st.caption(
        "In-process metrics collected by the FastAPI service since it last started (reset on "
        "restart) -- request traffic, model inference outcomes, and data/prediction drift "
        "versus the Phase 7 training baseline."
    )

    monitoring_health_result = cached_monitoring_health()
    monitoring_metrics_result = cached_monitoring_metrics()
    monitoring_drift_result = cached_monitoring_drift()

    st.markdown("#### API & Model Health")
    render_monitoring_health_summary(monitoring_health_result)

    st.divider()
    st.markdown("#### Request Metrics")
    render_request_metrics_section(monitoring_metrics_result)

    st.divider()
    st.markdown("#### Prediction Metrics")
    render_prediction_metrics_section(monitoring_metrics_result)

    st.divider()
    st.markdown("#### Drift Monitoring")
    render_drift_section(monitoring_drift_result)

    render_disclaimer(DISCLAIMER_TEXT)


if __name__ == "__main__":
    main()
