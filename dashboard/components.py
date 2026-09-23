"""Reusable Streamlit rendering components.

Every function here only renders data already computed elsewhere -- by
the FastAPI backend (predictions, SHAP explanations, recommendations) or
read directly from an existing local report file (reports/*.csv). None of
them predict, explain, or recommend anything themselves.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.api_client import APIResult
from dashboard.styles import PRIORITY_COLORS, PRIORITY_ORDER, RISK_LEVEL_COLORS, RISK_LEVEL_ORDER, badge_html

# A curated, real subset of customer_features.parquet columns -- spanning
# engagement, transactions, and support -- shown by default so the
# Customer Analysis tab doesn't dump all 131+ raw features on screen.
# Every name here is verified against the actual dataset; any that turn
# out to be absent from a given row are simply skipped (see
# render_supporting_features_table), never fabricated.
DEFAULT_COMPACT_FEATURES: tuple[str, ...] = (
    "tenure_days",
    "active_days_90d",
    "avg_active_minutes_90d",
    "avg_pages_viewed_90d",
    "avg_login_count_90d",
    "txn_count_90d",
    "txn_amount_sum_90d",
    "txn_failed_count_90d",
    "ticket_count_90d",
    "pending_ticket_count_90d",
    "avg_satisfaction_90d",
    "days_since_last_transaction",
)


def render_api_status(health: APIResult) -> None:
    """`health` is the APIResult from GET /health."""
    is_healthy = health.ok and isinstance(health.data, dict) and health.data.get("status") == "healthy"
    if is_healthy:
        st.sidebar.markdown("🟢 **API Connected**")
    else:
        st.sidebar.markdown("🔴 **API Unavailable**")
        detail = health.data.get("detail") if isinstance(health.data, dict) else None
        st.sidebar.caption(detail or health.error or "The inference service could not be reached.")


def render_model_info_sidebar(model_info: APIResult) -> None:
    """`model_info` is the APIResult from GET /model-info. Only fields
    actually present in the response are shown -- nothing is invented.
    """
    if not (model_info.ok and isinstance(model_info.data, dict)):
        st.sidebar.caption("Model information unavailable.")
        return

    data = model_info.data
    if "model_name" in data:
        st.sidebar.markdown(f"**Model:** {data['model_name']}")
    if "model_version" in data:
        st.sidebar.markdown(f"**Version:** {data['model_version']}")
    if "model_type" in data:
        st.sidebar.markdown(f"**Type:** {data['model_type']}")
    if "C" in data and "class_weight" in data:
        st.sidebar.caption(f'C={data["C"]}, class_weight="{data["class_weight"]}"')


def render_error_banner(result: APIResult, context: str) -> None:
    """A user-friendly error message for a failed API call -- never a
    Python traceback or internal implementation detail.
    """
    if result.status_code == 503:
        st.error(f"{context}: the inference service is unavailable. " + (result.error or ""))
    elif result.status_code == 422:
        st.warning(f"{context}: the request was rejected as invalid. {result.error or ''}")
    elif result.status_code is not None and result.status_code >= 500:
        st.error(f"{context}: the inference service returned an error. Please check the API logs.")
    else:
        st.error(f"{context}: {result.error or 'an unknown error occurred.'}")


def render_disclaimer(text: str) -> None:
    st.markdown(f'<div class="disclaimer-box">{text}</div>', unsafe_allow_html=True)


def render_risk_badge(risk_level: str) -> None:
    st.markdown(badge_html(risk_level, RISK_LEVEL_COLORS, "risk-badge"), unsafe_allow_html=True)


def render_priority_badge(priority: str) -> None:
    st.markdown(badge_html(priority, PRIORITY_COLORS, "priority-badge"), unsafe_allow_html=True)


def render_risk_distribution_chart(counts_by_level: dict[str, int]) -> None:
    """`counts_by_level` keys must already be the exact Phase 9 risk-level
    strings (low/medium/high/critical) -- categories are never reordered
    or renamed here.
    """
    labels = [level.title() for level in RISK_LEVEL_ORDER]
    values = [int(counts_by_level.get(level, 0)) for level in RISK_LEVEL_ORDER]
    colors = [RISK_LEVEL_COLORS[level] for level in RISK_LEVEL_ORDER]

    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=values, textposition="outside"))
    fig.update_layout(
        xaxis_title="Risk Level",
        yaxis_title="Customer Snapshots",
        margin=dict(l=10, r=10, t=20, b=10),
        height=340,
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


def render_action_distribution_chart(counts_by_action: pd.Series) -> None:
    """`counts_by_action` is a pandas Series indexed by the exact
    `primary_action` label strings from reports/recommendation_summary.csv
    (Phase 9), summed across risk levels and sorted descending.
    """
    ordered = counts_by_action.sort_values(ascending=True)
    fig = go.Figure(
        go.Bar(x=ordered.values, y=ordered.index, orientation="h", marker_color="#1565C0", text=ordered.values, textposition="outside")
    )
    fig.update_layout(
        xaxis_title="Customer Snapshots",
        yaxis_title="",
        margin=dict(l=10, r=10, t=20, b=10),
        height=max(300, 45 * len(ordered)),
    )
    st.plotly_chart(fig, width="stretch")


def render_prediction_summary(prediction: dict[str, Any]) -> None:
    """`prediction` is the `prediction` section of a /customer-analysis
    response (or a /predict response directly) -- rendered as-is, never
    recalculated.
    """
    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric("Model Churn Score", f"{prediction['churn_probability']:.1%}")
    with col2:
        st.markdown("**Risk Level**")
        render_risk_badge(prediction["risk_level"])


def render_recommendation_section(recommendation: dict[str, Any]) -> None:
    """`recommendation` is the `recommendation` section of a
    /customer-analysis response (or a /recommend response directly).
    """
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Recommended Action:** {recommendation['primary_action']}")
    with col2:
        st.markdown("**Priority**")
        render_priority_badge(recommendation["priority"])

    st.markdown(f"**Reason:** {recommendation['action_reason']}")

    supporting = recommendation.get("supporting_features") or []
    if supporting:
        st.markdown("**Supporting Signals**")
        for signal in supporting:
            st.markdown(f"- {signal}")
    else:
        st.caption("No specific triggering signal -- this action was determined by risk level alone.")

    if recommendation.get("disclaimer"):
        st.caption(recommendation["disclaimer"])


def render_shap_section(explanation: dict[str, Any]) -> None:
    """`explanation` is the `explanation` section of a /customer-analysis
    response (or a /predict/explain response directly). Every SHAP value
    plotted here comes directly from that response -- SHAP is never
    invoked from within the dashboard.
    """
    positive = explanation.get("top_positive_drivers") or []
    negative = explanation.get("top_negative_drivers") or []

    if not positive and not negative:
        st.info("No SHAP drivers were returned for this customer.")
        return

    rows = [{"feature": d["feature"], "shap_value": d["shap_value"], "direction": "Increases score"} for d in positive]
    rows += [{"feature": d["feature"], "shap_value": d["shap_value"], "direction": "Decreases score"} for d in negative]
    driver_df = pd.DataFrame(rows).sort_values("shap_value")

    fig = go.Figure(
        go.Bar(
            x=driver_df["shap_value"],
            y=driver_df["feature"],
            orientation="h",
            marker_color=["#C62828" if v > 0 else "#1565C0" for v in driver_df["shap_value"]],
        )
    )
    fig.update_layout(
        xaxis_title="SHAP value (impact on model score)",
        yaxis_title="",
        margin=dict(l=10, r=10, t=20, b=10),
        height=max(280, 38 * len(driver_df)),
    )
    st.plotly_chart(fig, width="stretch")

    legend_col1, legend_col2 = st.columns(2)
    legend_col1.markdown('🔴 **Drivers increasing model score**')
    legend_col2.markdown('🔵 **Drivers decreasing model score**')

    st.caption(
        "SHAP values explain how model features contributed to this prediction. "
        "They do not establish causal relationships."
    )


def render_supporting_features_table(row: pd.Series, compact_features: tuple[str, ...] = DEFAULT_COMPACT_FEATURES) -> None:
    """Renders a compact table of real customer signals from the selected
    feature row, plus an expander with every available feature. Only
    columns that actually exist on `row` are shown.
    """
    available_compact = [f for f in compact_features if f in row.index]
    if available_compact:
        compact_df = pd.DataFrame({"Feature": available_compact, "Value": [row[f] for f in available_compact]})
        st.dataframe(compact_df, hide_index=True, width="stretch")
    else:
        st.caption("None of the default compact features are present in this dataset.")

    with st.expander("Show all available features"):
        # Values are stringified here because this row legitimately mixes
        # types (str customer_id, a Timestamp, floats, bools) in one
        # column -- Arrow (which st.dataframe serializes through) cannot
        # represent that mix natively. The compact table above stays
        # numeric/unconverted since its features are homogeneous.
        all_df = pd.DataFrame({"Feature": row.index, "Value": [str(v) for v in row.values]})
        st.dataframe(all_df, hide_index=True, width="stretch", height=400)


# ---------------------------------------------------------------------------
# Phase 14: monitoring tab -- every number here is read directly from a
# GET /monitoring/* response. Nothing is computed in the dashboard.
# ---------------------------------------------------------------------------

DRIFT_STATUS_COLORS = {
    "no_significant_shift": "#2E7D32",
    "moderate_shift": "#F9A825",
    "significant_shift": "#C62828",
    "insufficient_data": "#9E9E9E",
    "no_baseline": "#9E9E9E",
}


def render_monitoring_health_summary(monitoring_health: APIResult) -> None:
    if not (monitoring_health.ok and isinstance(monitoring_health.data, dict)):
        st.error("Monitoring health is unavailable.")
        return

    data = monitoring_health.data
    cols = st.columns(4)
    cols[0].metric("API Status", data.get("api_status", "unknown").title())
    cols[1].metric("Model Status", data.get("model_status", "unknown").title())
    cols[2].metric("Drift Monitoring", data.get("drift_monitoring_status", "unknown").title())
    model_label = data.get("model_name") or "n/a"
    version_label = data.get("model_version") or "n/a"
    cols[3].metric("Model", f"{model_label} v{version_label}" if model_label != "n/a" else "n/a")


def render_request_metrics_section(metrics: APIResult) -> None:
    if not (metrics.ok and isinstance(metrics.data, dict)):
        st.error("Request metrics are unavailable.")
        return

    data = metrics.data
    cols = st.columns(3)
    cols[0].metric("Total Requests", f"{data['total_requests']:,}")
    cols[1].metric("Successful", f"{data['successful_requests']:,}")
    cols[2].metric("Failed", f"{data['failed_requests']:,}")

    latency = data.get("latency_ms", {})
    st.markdown("**Latency (ms)**")
    latency_cols = st.columns(4)
    latency_cols[0].metric("Average", f"{latency['avg']:.1f}" if latency.get("avg") is not None else "n/a")
    latency_cols[1].metric("p50", f"{latency['p50']:.1f}" if latency.get("p50") is not None else "n/a")
    latency_cols[2].metric("p95", f"{latency['p95']:.1f}" if latency.get("p95") is not None else "n/a")
    latency_cols[3].metric("Max", f"{latency['max']:.1f}" if latency.get("max") is not None else "n/a")

    if data.get("requests_by_endpoint"):
        st.markdown("**Requests by Endpoint**")
        endpoint_df = pd.DataFrame(
            {"Endpoint": list(data["requests_by_endpoint"].keys()), "Count": list(data["requests_by_endpoint"].values())}
        ).sort_values("Count", ascending=False)
        st.dataframe(endpoint_df, hide_index=True, width="stretch")


def render_prediction_metrics_section(metrics: APIResult) -> None:
    if not (metrics.ok and isinstance(metrics.data, dict)):
        st.error("Prediction metrics are unavailable.")
        return

    predictions = metrics.data.get("predictions", {})
    cols = st.columns(4)
    cols[0].metric("Total Predictions", f"{predictions.get('total_predictions', 0):,}")
    cols[1].metric("Positive", f"{predictions.get('positive_predictions', 0):,}")
    cols[2].metric("Negative", f"{predictions.get('negative_predictions', 0):,}")
    cols[3].metric("Failed", f"{predictions.get('failed_predictions', 0):,}")

    avg_prob = predictions.get("average_predicted_probability")
    avg_inference_ms = predictions.get("average_inference_time_ms")
    caption_parts = []
    if avg_prob is not None:
        caption_parts.append(f"Average predicted probability (recent window): {avg_prob:.1%}")
    if avg_inference_ms is not None:
        caption_parts.append(f"Average inference time: {avg_inference_ms:.1f} ms")
    if caption_parts:
        st.caption(" · ".join(caption_parts))


def render_drift_section(drift: APIResult) -> None:
    if drift.status_code == 503:
        st.info("Drift monitoring is unavailable (no baseline was loaded at API startup).")
        return
    if not (drift.ok and isinstance(drift.data, dict)):
        st.error("Drift monitoring is unavailable.")
        return

    data = drift.data
    cols = st.columns(3)
    cols[0].metric("Drifted Features", data.get("drifted_feature_count", 0))
    prediction_drift = data.get("prediction_drift", {})
    cols[1].metric("Prediction Drift", prediction_drift.get("status", "unknown").replace("_", " ").title())
    psi_value = prediction_drift.get("psi")
    cols[2].metric("Prediction PSI", f"{psi_value:.3f}" if psi_value is not None else "n/a")

    feature_drift = data.get("feature_drift", [])
    if feature_drift:
        st.markdown("**Feature Drift (PSI vs. training baseline)**")
        drift_df = pd.DataFrame(feature_drift)
        st.dataframe(drift_df, hide_index=True, width="stretch")

    st.caption(data.get("note", ""))
    st.caption(f"Baseline: {data.get('baseline_source', 'n/a')}")
