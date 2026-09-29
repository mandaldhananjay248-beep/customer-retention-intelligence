"""Tests for Phase 15's Streamlit Community Cloud path: streamlit_app.py
running the dashboard with DASHBOARD_API_MODE=inprocess, which sends the
dashboard's normal API requests to the existing FastAPI app in-process
(api/inprocess.py).

The parity tests compare that path against an independently loaded
ModelService and the packaged model itself, for C000001 and customers
spread across the dataset, so the Cloud deployment can't silently serve
different predictions, explanations or recommendations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from api.services import ModelService
from dashboard import api_client
from dashboard import app as dashboard_app
from src.data import regenerate_feature_dataset as regen
from src.models import mlflow_tracking as mt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT_PATH = PROJECT_ROOT / "streamlit_app.py"
EXPECTED_MODEL_INFO = {
    "model_name": "customer-churn-logistic-regression",
    "model_version": "1",
    "model_type": "LogisticRegression",
    "C": 0.01,
    "class_weight": "balanced",
}
# Row positions spread across the 136,701-row feature dataset.
REPRESENTATIVE_ROW_POSITIONS = [0, 1, 25_000, 60_000, 99_999, 136_700]


@pytest.fixture
def inprocess_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(api_client.API_MODE_ENV_VAR, api_client.INPROCESS_API_MODE)
    monkeypatch.delenv("API_BASE_URL", raising=False)


@pytest.fixture(scope="module")
def features_df() -> pd.DataFrame:
    return dashboard_app._load_customer_features_uncached()


@pytest.fixture(scope="module")
def reference_service() -> ModelService:
    """An independently loaded model service -- not the in-process app's."""
    service = ModelService()
    service.load()
    return service


def _representative_rows(features_df: pd.DataFrame) -> list[pd.Series]:
    c000001 = features_df[features_df["customer_id"] == "C000001"].sort_values("snapshot_date").iloc[-1]
    return [c000001] + [features_df.iloc[i] for i in REPRESENTATIVE_ROW_POSITIONS]


def _feature_payload(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in ("customer_id", "snapshot_date")}


# ---------------------------------------------------------------------------
# Mode switch
# ---------------------------------------------------------------------------

def test_inprocess_mode_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(api_client.API_MODE_ENV_VAR, raising=False)
    assert not api_client.is_inprocess_mode()


def test_inprocess_mode_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(api_client.API_MODE_ENV_VAR, "InProcess")
    assert api_client.is_inprocess_mode()
    monkeypatch.setenv(api_client.API_MODE_ENV_VAR, "http")
    assert not api_client.is_inprocess_mode()


def test_http_mode_never_touches_the_inprocess_app(monkeypatch: pytest.MonkeyPatch) -> None:
    from api import inprocess

    monkeypatch.delenv(api_client.API_MODE_ENV_VAR, raising=False)
    monkeypatch.setattr(inprocess, "request", lambda *a, **k: pytest.fail("in-process app used in HTTP mode"))

    def fake_http(method, url, json=None, timeout=None):
        raise api_client.requests.exceptions.ConnectionError("no server")

    monkeypatch.setattr(api_client.requests, "request", fake_http)
    result = api_client.get_health()
    assert not result.ok and "Could not connect" in result.error


def test_inprocess_failure_is_reported_not_raised(inprocess_mode, monkeypatch: pytest.MonkeyPatch) -> None:
    from api import inprocess

    def boom(*args, **kwargs):
        raise RuntimeError("secret internals")

    monkeypatch.setattr(inprocess, "request", boom)
    result = api_client.get_health()
    assert not result.ok
    assert "RuntimeError" in result.error and "secret internals" not in result.error


# ---------------------------------------------------------------------------
# The in-process API: same endpoints, same answers
# ---------------------------------------------------------------------------

def test_inprocess_health_and_model_info(inprocess_mode) -> None:
    health = api_client.get_health()
    assert health.ok and health.data["status"] == "healthy"
    info = api_client.get_model_info()
    assert info.ok and info.data == EXPECTED_MODEL_INFO


def test_inprocess_api_serves_the_sha256_verified_packaged_model(inprocess_mode) -> None:
    from api import inprocess

    api_client.get_health()  # ensure the app has started
    service = inprocess.get_client().app.state.model_service
    config = mt.load_config()
    assert service.model_version == config["packaged_model"]["version"]
    reference = mt.load_packaged_model(config)
    assert (service.pipeline.named_steps["model"].coef_ == reference.named_steps["model"].coef_).all()


def test_customer_analysis_parity_with_model_service_and_packaged_model(
    inprocess_mode, features_df: pd.DataFrame, reference_service: ModelService
) -> None:
    packaged_model = mt.load_packaged_model()
    feature_columns = reference_service.feature_columns

    for row in _representative_rows(features_df):
        payload = api_client.build_customer_payload(row)
        result = api_client.customer_analysis(payload, top_k=5)
        assert result.ok, (row["customer_id"], result.error)
        analysis = result.data
        features = _feature_payload(payload)

        # Prediction: identical to the model service and to the packaged
        # model applied directly to the dataset row (same float64 code path,
        # so exact equality is expected -- no tolerance).
        probability = analysis["prediction"]["churn_probability"]
        assert probability == reference_service.predict_probability(features)
        assert probability == float(packaged_model.predict_proba(row[feature_columns].to_frame().T.infer_objects())[0, 1])
        assert analysis["prediction"]["risk_level"] == reference_service.predict(features)["risk_level"]
        assert analysis["prediction"]["model_version"] == "1"

        # SHAP explanation: same drivers and values as the model service.
        expected_explanation = reference_service.explain(features, top_k=5)
        for key in ("top_positive_drivers", "top_negative_drivers"):
            assert [d["feature"] for d in analysis["explanation"][key]] == [d["feature"] for d in expected_explanation[key]]
            assert [d["shap_value"] for d in analysis["explanation"][key]] == [d["shap_value"] for d in expected_explanation[key]]

        # Recommendation: same action, priority and reasoning.
        expected_recommendation = reference_service.recommend(
            features, payload["customer_id"], pd.Timestamp(payload["snapshot_date"]).date(), probability=probability
        )
        for key in ("primary_action", "priority", "risk_level", "action_reason"):
            assert analysis["recommendation"][key] == expected_recommendation[key], key


def test_c000001_latest_snapshot_matches_the_verified_dashboard_score(inprocess_mode, features_df: pd.DataFrame) -> None:
    """63.3% is the score the Docker (HTTP) dashboard showed for C000001."""
    row = _representative_rows(features_df)[0]
    assert str(pd.Timestamp(row["snapshot_date"]).date()) == "2026-05-31"
    result = api_client.customer_analysis(api_client.build_customer_payload(row))
    assert result.ok
    assert f"{result.data['prediction']['churn_probability']:.1%}" == "63.3%"


def test_inprocess_requests_are_recorded_by_phase14_monitoring(inprocess_mode, features_df: pd.DataFrame) -> None:
    before = api_client.get_monitoring_metrics().data
    api_client.customer_analysis(api_client.build_customer_payload(features_df.iloc[0]))
    after = api_client.get_monitoring_metrics().data
    assert after["total_requests"] > before["total_requests"]
    assert after["predictions"]["total_predictions"] == before["predictions"]["total_predictions"] + 1
    assert api_client.get_monitoring_health().data["drift_monitoring_status"] == "active"
    assert api_client.get_monitoring_drift().ok


def test_inprocess_validation_errors_match_the_http_api(inprocess_mode) -> None:
    result = api_client.predict({"customer_id": "X"})
    assert not result.ok and result.status_code == 422


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def test_streamlit_app_renders_all_tabs_and_analyzes_c000001(inprocess_mode) -> None:
    at = AppTest.from_file(str(ENTRYPOINT_PATH), default_timeout=300).run()
    assert not at.exception, at.exception
    assert [tab.label for tab in at.tabs] == ["Overview", "Customer Analysis", "Model Information", "Monitoring"]
    assert not at.error, [e.value for e in at.error]
    assert at.sidebar.selectbox[0].value == "C000001"

    at = next(b for b in at.button if b.label == "Analyze Customer").click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Model Churn Score"] == "63.3%"
    assert metrics["Model Version"] == "1"
    assert any("Recommended Action:" in str(md.value) for md in at.markdown)


def test_streamlit_app_turns_on_inprocess_mode_without_overriding_an_explicit_choice() -> None:
    source = ENTRYPOINT_PATH.read_text()
    assert 'os.environ.setdefault("DASHBOARD_API_MODE", "inprocess")' in source
    assert source.index("os.environ.setdefault") < source.index("from dashboard import app")
    assert "dashboard_app.main()" in source


def test_ensure_feature_dataset_keeps_an_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = tmp_path / "customer_features.parquet"
    existing.write_bytes(b"placeholder")
    monkeypatch.setattr(regen, "regenerate", lambda **kwargs: pytest.fail("must not regenerate an existing dataset"))
    assert regen.ensure_feature_dataset(existing) == existing


def test_ensure_feature_dataset_regenerates_a_missing_file_in_a_child_process_without_pruning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(regen.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    target = tmp_path / "missing.parquet"
    assert regen.ensure_feature_dataset(target) == target
    (command,), kwargs = calls[0]
    assert command[1:] == ["-m", "src.data.regenerate_feature_dataset"]  # no --prune
    assert kwargs == {"cwd": regen.PROJECT_ROOT, "check": True}
