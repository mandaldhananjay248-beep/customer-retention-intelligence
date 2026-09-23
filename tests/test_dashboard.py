"""Tests for dashboard/api_client.py and dashboard/app.py's pure data
helpers.

All HTTP calls are mocked (requests.request) -- no live FastAPI server is
required for these tests. Data-loading tests use the real
data/processed/customer_features.parquet (skipped if it doesn't exist),
mirroring the pattern used in test_explainability.py / test_api.py: a
synthetic 131-feature row would just re-derive the same schema these
tests are trying to verify against.
"""

from __future__ import annotations

import ast
import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from dashboard import api_client
import dashboard.app as dashboard_app


def _mock_response(status_code: int = 200, json_data=None, json_error: bool = False) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    if json_error:
        response.json.side_effect = ValueError("invalid json")
    else:
        response.json.return_value = json_data
    return response


# ---------------------------------------------------------------------------
# API client: base URL configuration
# ---------------------------------------------------------------------------

def test_get_api_base_url_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_BASE_URL", raising=False)
    assert api_client.get_api_base_url() == api_client.DEFAULT_API_BASE_URL


def test_get_api_base_url_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_BASE_URL", "http://example.com:9999/")
    assert api_client.get_api_base_url() == "http://example.com:9999"


# ---------------------------------------------------------------------------
# API client: successful responses
# ---------------------------------------------------------------------------

def test_get_health_success_parsing() -> None:
    body = {"status": "healthy", "model": "customer-churn-logistic-regression", "model_version": "1", "detail": None}
    with patch("dashboard.api_client.requests.request", return_value=_mock_response(200, body)):
        result = api_client.get_health()

    assert result.ok
    assert result.status_code == 200
    assert result.data["status"] == "healthy"
    assert result.data["model_version"] == "1"


def test_get_model_info_success_parsing() -> None:
    body = {
        "model_name": "customer-churn-logistic-regression",
        "model_version": "1",
        "model_type": "LogisticRegression",
        "C": 0.01,
        "class_weight": "balanced",
    }
    with patch("dashboard.api_client.requests.request", return_value=_mock_response(200, body)):
        result = api_client.get_model_info()

    assert result.ok
    assert result.data == body


def test_get_monitoring_metrics_success_parsing() -> None:
    body = {"total_requests": 10, "successful_requests": 9, "failed_requests": 1, "predictions": {"total_predictions": 5}}
    with patch("dashboard.api_client.requests.request", return_value=_mock_response(200, body)):
        result = api_client.get_monitoring_metrics()

    assert result.ok
    assert result.data["total_requests"] == 10


def test_get_monitoring_health_success_parsing() -> None:
    body = {"api_status": "healthy", "model_status": "available", "drift_monitoring_status": "active"}
    with patch("dashboard.api_client.requests.request", return_value=_mock_response(200, body)):
        result = api_client.get_monitoring_health()

    assert result.ok
    assert result.data["api_status"] == "healthy"


def test_get_monitoring_drift_success_parsing() -> None:
    body = {"feature_drift": [], "drifted_feature_count": 0, "prediction_drift": {"status": "no_significant_shift"}}
    with patch("dashboard.api_client.requests.request", return_value=_mock_response(200, body)):
        result = api_client.get_monitoring_drift()

    assert result.ok
    assert result.data["drifted_feature_count"] == 0


def test_get_monitoring_drift_unavailable_returns_503_result() -> None:
    with patch(
        "dashboard.api_client.requests.request",
        return_value=_mock_response(503, {"detail": "Model service is unavailable."}),
    ):
        result = api_client.get_monitoring_drift()

    assert not result.ok
    assert result.status_code == 503


def test_predict_success() -> None:
    body = {
        "customer_id": "C0001", "snapshot_date": "2025-06-30", "churn_probability": 0.42,
        "risk_level": "medium", "model_name": "customer-churn-logistic-regression", "model_version": "1",
    }
    with patch("dashboard.api_client.requests.request", return_value=_mock_response(200, body)):
        result = api_client.predict({"customer_id": "C0001"})

    assert result.ok
    assert result.data["churn_probability"] == 0.42


def test_customer_analysis_success_and_top_k_forwarded() -> None:
    body = {"customer": {}, "prediction": {}, "explanation": {}, "recommendation": {}}
    with patch("dashboard.api_client.requests.request", return_value=_mock_response(200, body)) as mock_request:
        result = api_client.customer_analysis({"customer_id": "C0001"}, top_k=3)

    assert result.ok
    sent_json = mock_request.call_args.kwargs["json"]
    assert sent_json["top_k"] == 3
    assert sent_json["customer_id"] == "C0001"


def test_explain_prediction_and_recommend_success() -> None:
    with patch("dashboard.api_client.requests.request", return_value=_mock_response(200, {"ok": True})):
        explain_result = api_client.explain_prediction({"customer_id": "C1"}, top_k=5)
        recommend_result = api_client.recommend({"customer_id": "C1"})

    assert explain_result.ok
    assert recommend_result.ok


# ---------------------------------------------------------------------------
# API client: connection / timeout failures
# ---------------------------------------------------------------------------

def test_connection_error_gives_actionable_message() -> None:
    with patch("dashboard.api_client.requests.request", side_effect=requests.exceptions.ConnectionError()):
        result = api_client.get_health()

    assert not result.ok
    assert "uvicorn" in result.error.lower()


def test_timeout_error_handling() -> None:
    with patch("dashboard.api_client.requests.request", side_effect=requests.exceptions.Timeout()):
        result = api_client.get_health()

    assert not result.ok
    assert "took too long" in result.error.lower()


def test_connect_timeout_handling() -> None:
    with patch("dashboard.api_client.requests.request", side_effect=requests.exceptions.ConnectTimeout()):
        result = api_client.get_health()

    assert not result.ok
    assert "timed out" in result.error.lower()


def test_generic_request_exception_handling() -> None:
    with patch("dashboard.api_client.requests.request", side_effect=requests.exceptions.RequestException("boom")):
        result = api_client.get_health()

    assert not result.ok
    assert "boom" not in (result.error or "")  # never leak the raw exception text either


# ---------------------------------------------------------------------------
# API client: HTTP error responses
# ---------------------------------------------------------------------------

def test_http_503_with_string_detail() -> None:
    with patch(
        "dashboard.api_client.requests.request",
        return_value=_mock_response(503, {"detail": "Model service is unavailable."}),
    ):
        result = api_client.get_health()

    assert not result.ok
    assert result.status_code == 503
    assert "unavailable" in result.error.lower()


def test_http_422_with_validation_error_list_detail() -> None:
    detail = [{"loc": ["body", "age"], "msg": "Field required", "type": "missing"}]
    with patch("dashboard.api_client.requests.request", return_value=_mock_response(422, {"detail": detail})):
        result = api_client.predict({"customer_id": "C1"})

    assert not result.ok
    assert result.status_code == 422
    assert "age" in result.error


def test_http_500_error_handling() -> None:
    with patch("dashboard.api_client.requests.request", return_value=_mock_response(500, {"detail": "Internal server error."})):
        result = api_client.predict({"customer_id": "C1"})

    assert not result.ok
    assert result.status_code == 500


def test_malformed_json_response_handling() -> None:
    with patch("dashboard.api_client.requests.request", return_value=_mock_response(200, json_error=True)):
        result = api_client.get_health()

    assert not result.ok
    assert "could not be parsed" in result.error.lower()


def test_unexpected_response_structure_handling() -> None:
    """A 200 response whose body is a list, not the expected dict."""
    with patch("dashboard.api_client.requests.request", return_value=_mock_response(200, [1, 2, 3])):
        result = api_client.get_health()

    assert not result.ok
    assert "unexpected" in result.error.lower()


# ---------------------------------------------------------------------------
# Dashboard data: customer selection, row lookup, payload construction
# ---------------------------------------------------------------------------

if not dashboard_app.FEATURE_DATASET_PATH.exists():
    pytest.skip(
        f"{dashboard_app.FEATURE_DATASET_PATH} not found -- run Phase 4 feature engineering first",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def features_df() -> pd.DataFrame:
    return dashboard_app._load_customer_features_uncached()


def test_load_customer_features_has_expected_columns(features_df: pd.DataFrame) -> None:
    assert "customer_id" in features_df.columns
    assert "snapshot_date" in features_df.columns
    assert len(features_df) > 0


def test_get_customer_snapshot_row_valid_lookup(features_df: pd.DataFrame) -> None:
    sample = features_df.iloc[0]
    row = dashboard_app.get_customer_snapshot_row(
        features_df, sample["customer_id"], pd.Timestamp(sample["snapshot_date"]).date()
    )
    assert row["customer_id"] == sample["customer_id"]
    assert pd.Timestamp(row["snapshot_date"]) == pd.Timestamp(sample["snapshot_date"])


def test_get_customer_snapshot_row_invalid_raises(features_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        dashboard_app.get_customer_snapshot_row(features_df, "NOT_A_REAL_CUSTOMER_ID", date(2000, 1, 1))


def test_build_customer_payload_excludes_target_column(features_df: pd.DataFrame) -> None:
    payload = api_client.build_customer_payload(features_df.iloc[0])
    assert "churn_next_30_days" not in payload


def test_build_customer_payload_has_no_fabricated_columns(features_df: pd.DataFrame) -> None:
    """The payload's keys must be exactly the real row's columns minus the
    target -- nothing invented, nothing silently dropped.
    """
    row = features_df.iloc[0]
    payload = api_client.build_customer_payload(row)
    expected_keys = set(row.index) - {"churn_next_30_days"}
    assert set(payload.keys()) == expected_keys


def test_build_customer_payload_is_json_serializable(features_df: pd.DataFrame) -> None:
    payload = api_client.build_customer_payload(features_df.iloc[0])
    json.dumps(payload)  # must not raise


def test_build_customer_payload_preserves_null_values(features_df: pd.DataFrame) -> None:
    """A legitimately missing feature (e.g. no support-ticket history)
    must come through as JSON null, never a fabricated default.
    """
    nan_rows = features_df[features_df["avg_satisfaction_90d"].isna()]
    assert len(nan_rows) > 0, "expected at least one row with a legitimately missing avg_satisfaction_90d"
    payload = api_client.build_customer_payload(nan_rows.iloc[0])
    assert payload["avg_satisfaction_90d"] is None


def test_build_customer_payload_snapshot_date_is_iso_string(features_df: pd.DataFrame) -> None:
    payload = api_client.build_customer_payload(features_df.iloc[0])
    date.fromisoformat(payload["snapshot_date"])  # must not raise


def test_build_customer_payload_preserves_real_values(features_df: pd.DataFrame) -> None:
    row = features_df.iloc[0]
    payload = api_client.build_customer_payload(row)
    assert payload["customer_id"] == row["customer_id"]
    assert payload["age"] == row["age"]


def test_build_customer_payload_converts_numpy_types_to_native(features_df: pd.DataFrame) -> None:
    payload = api_client.build_customer_payload(features_df.iloc[0])
    for key, value in payload.items():
        assert type(value).__module__ != "numpy", f"{key} is still a numpy type: {type(value)}"


# ---------------------------------------------------------------------------
# Architectural guardrail: the dashboard must never import ML libraries
# directly (no duplicated model/SHAP/preprocessing logic).
# ---------------------------------------------------------------------------

def test_dashboard_modules_do_not_import_ml_libraries() -> None:
    forbidden_top_level_modules = {"shap", "mlflow", "sklearn", "joblib", "xgboost", "lightgbm", "catboost"}
    dashboard_dir = Path(dashboard_app.__file__).parent

    for py_file in dashboard_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
        imported_top_level_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_top_level_modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_top_level_modules.add(node.module.split(".")[0])

        overlap = forbidden_top_level_modules & imported_top_level_modules
        assert not overlap, f"{py_file.name} imports a forbidden ML library directly: {overlap}"


def test_dashboard_does_not_reference_joblib_model_path() -> None:
    dashboard_dir = Path(dashboard_app.__file__).parent
    for py_file in dashboard_dir.glob("*.py"):
        source = py_file.read_text()
        assert "tuned_logistic_regression.joblib" not in source, f"{py_file.name} references the raw model artifact directly"


def test_monitoring_tab_is_wired_into_the_dashboard() -> None:
    """Phase 14: the Monitoring tab must actually be registered as one of
    st.tabs(...)'s labels and rendered via a dedicated function -- not
    just mentioned in a comment somewhere.
    """
    source = Path(dashboard_app.__file__).read_text()
    assert '"Monitoring"' in source
    assert "_render_monitoring_tab" in source


def test_monitoring_render_functions_only_read_api_results() -> None:
    """The Phase 14 monitoring render functions must accept an APIResult
    (or plain dict already sourced from one) and never import shap/mlflow/
    sklearn -- covered generically by test_dashboard_modules_do_not_import_ml_libraries,
    this additionally checks the specific functions exist and are exported
    from components.py, so a broken import wouldn't silently pass.
    """
    from dashboard import components

    for function_name in (
        "render_monitoring_health_summary",
        "render_request_metrics_section",
        "render_prediction_metrics_section",
        "render_drift_section",
    ):
        assert hasattr(components, function_name), f"dashboard.components.{function_name} is missing"
