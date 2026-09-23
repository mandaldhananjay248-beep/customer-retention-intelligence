"""Tests for src/monitoring/ (metrics, logging_config, drift) and the
Phase 14 /monitoring/* API endpoints + dashboard Monitoring tab.

Pure unit tests for RequestMetrics/PredictionMetrics/DriftMonitor use
small synthetic data (fast, isolated). API/dashboard integration tests
reuse the real registered Phase 10 model (skipped if it isn't available
yet), mirroring the pattern already established in test_api.py and
test_dashboard.py.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import DriftMonitor, classify_psi, compute_psi, select_top_numeric_features_by_shap
from src.monitoring.logging_config import ACCESS_LOGGER_NAME, log_request
from src.monitoring.metrics import PredictionMetrics, RequestMetrics

# ---------------------------------------------------------------------------
# RequestMetrics
# ---------------------------------------------------------------------------

def test_request_metrics_counts_total_and_by_dimension() -> None:
    metrics = RequestMetrics()
    metrics.record_request("GET", "/health", 200, 5.0)
    metrics.record_request("POST", "/predict", 200, 10.0)
    metrics.record_request("POST", "/predict", 422, 3.0)

    snapshot = metrics.snapshot()
    assert snapshot["total_requests"] == 3
    assert snapshot["requests_by_endpoint"] == {"/health": 1, "/predict": 2}
    assert snapshot["requests_by_method"] == {"GET": 1, "POST": 2}
    assert snapshot["requests_by_status"] == {"200": 2, "422": 1}


def test_request_metrics_success_vs_failure_classification() -> None:
    metrics = RequestMetrics()
    for status_code in (200, 201, 304, 400, 422, 500, 503):
        metrics.record_request("GET", "/x", status_code, 1.0)

    snapshot = metrics.snapshot()
    assert snapshot["successful_requests"] == 3  # 200, 201, 304
    assert snapshot["failed_requests"] == 4  # 400, 422, 500, 503


def test_request_metrics_latency_statistics() -> None:
    metrics = RequestMetrics()
    for latency in [10.0, 20.0, 30.0, 40.0, 50.0]:
        metrics.record_request("GET", "/x", 200, latency)

    latency_stats = metrics.snapshot()["latency_ms"]
    assert latency_stats["avg"] == pytest.approx(30.0)
    assert latency_stats["min"] == 10.0
    assert latency_stats["max"] == 50.0
    assert latency_stats["p50"] == 30.0
    assert latency_stats["sample_size"] == 5


def test_request_metrics_rolling_window_bounds_memory() -> None:
    metrics = RequestMetrics(rolling_window=10)
    for i in range(100):
        metrics.record_request("GET", "/x", 200, float(i))

    snapshot = metrics.snapshot()
    assert snapshot["total_requests"] == 100  # counts are exact, unbounded
    assert snapshot["latency_ms"]["sample_size"] == 10  # percentile window is bounded
    assert snapshot["latency_ms"]["max"] == 99.0  # min/max are exact running stats, not windowed


def test_request_metrics_thread_safety_under_concurrency() -> None:
    import threading

    metrics = RequestMetrics()

    def worker() -> None:
        for _ in range(200):
            metrics.record_request("GET", "/x", 200, 1.0)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert metrics.snapshot()["total_requests"] == 1600


# ---------------------------------------------------------------------------
# PredictionMetrics
# ---------------------------------------------------------------------------

def test_prediction_metrics_success_and_failure_counts() -> None:
    metrics = PredictionMetrics(classification_threshold=0.5)
    metrics.record_success(0.8, 12.0)
    metrics.record_success(0.2, 8.0)
    metrics.record_failure()

    snapshot = metrics.snapshot()
    assert snapshot["total_predictions"] == 3
    assert snapshot["successful_predictions"] == 2
    assert snapshot["failed_predictions"] == 1
    assert snapshot["positive_predictions"] == 1
    assert snapshot["negative_predictions"] == 1


def test_prediction_metrics_average_probability_and_inference_time() -> None:
    metrics = PredictionMetrics(classification_threshold=0.5)
    metrics.record_success(0.4, 10.0)
    metrics.record_success(0.6, 20.0)

    snapshot = metrics.snapshot()
    assert snapshot["average_predicted_probability"] == pytest.approx(0.5)
    assert snapshot["average_inference_time_ms"] == pytest.approx(15.0)
    assert snapshot["last_successful_inference_at"] is not None


def test_prediction_metrics_null_field_rate() -> None:
    metrics = PredictionMetrics(classification_threshold=0.5)
    metrics.record_success(0.5, 10.0, null_field_count=13, total_field_count=131)
    metrics.record_success(0.5, 10.0, null_field_count=0, total_field_count=131)

    snapshot = metrics.snapshot()
    assert snapshot["average_null_field_rate"] == pytest.approx(13 / 262)


def test_prediction_metrics_no_data_returns_none_not_error() -> None:
    metrics = PredictionMetrics(classification_threshold=0.5)
    snapshot = metrics.snapshot()
    assert snapshot["total_predictions"] == 0
    assert snapshot["average_predicted_probability"] is None
    assert snapshot["average_null_field_rate"] is None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def test_log_request_emits_structured_line(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=ACCESS_LOGGER_NAME):
        log_request("POST", "/customer-analysis", 200, 42.5, "req-abc-123")

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "method=POST" in message
    assert "path=/customer-analysis" in message
    assert "status=200" in message
    assert "duration_ms=42.5" in message
    assert "request_id=req-abc-123" in message


def test_log_request_never_includes_arbitrary_payload_data(caplog: pytest.LogCaptureFixture) -> None:
    """log_request's signature structurally cannot accept a payload -- this
    guards that no future edit sneaks one into the formatted message.
    """
    with caplog.at_level(logging.INFO, logger=ACCESS_LOGGER_NAME):
        log_request("POST", "/predict", 200, 10.0, "req-1")

    message = caplog.records[0].getMessage()
    for forbidden_token in ("customer_id", "avg_pages_viewed", "churn_probability", "password", "token", "secret"):
        assert forbidden_token not in message


# ---------------------------------------------------------------------------
# Drift: compute_psi / classify_psi
# ---------------------------------------------------------------------------

def test_compute_psi_is_zero_for_identical_distributions() -> None:
    counts = np.array([10.0, 20.0, 30.0, 40.0])
    assert compute_psi(counts, counts) == pytest.approx(0.0, abs=1e-9)


def test_compute_psi_is_large_for_very_different_distributions() -> None:
    baseline = np.array([100.0, 0.0, 0.0, 0.0])
    shifted = np.array([0.0, 0.0, 0.0, 100.0])
    assert compute_psi(baseline, shifted) > 1.0


@pytest.mark.parametrize(
    "psi_value,expected_status",
    [(0.01, "no_significant_shift"), (0.15, "moderate_shift"), (0.5, "significant_shift")],
)
def test_classify_psi_thresholds(psi_value: float, expected_status: str) -> None:
    assert classify_psi(psi_value) == expected_status


def test_select_top_numeric_features_by_shap_excludes_categorical_dummies() -> None:
    shap_importance = pd.DataFrame({
        "feature": ["plan_Premium", "avg_pages_viewed_90d", "country_India", "active_days_90d"],
        "mean_abs_shap": [0.9, 0.8, 0.7, 0.6],
    })
    numeric_columns = ["avg_pages_viewed_90d", "active_days_90d"]

    selected = select_top_numeric_features_by_shap(shap_importance, numeric_columns, top_n=5)
    assert selected == ["avg_pages_viewed_90d", "active_days_90d"]
    assert "plan_Premium" not in selected


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_train_features() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        "numeric_feature": rng.normal(loc=10.0, scale=2.0, size=n),
        "category_feature": rng.choice(["A", "B", "C"], size=n, p=[0.6, 0.3, 0.1]),
    })


@pytest.fixture
def drift_monitor(synthetic_train_features: pd.DataFrame) -> DriftMonitor:
    return DriftMonitor(
        synthetic_train_features, numeric_features=["numeric_feature"], categorical_features=["category_feature"],
        rolling_window=200,
    )


def test_drift_monitor_reports_insufficient_data_before_any_observation(drift_monitor: DriftMonitor) -> None:
    snapshot = drift_monitor.snapshot()
    assert all(r["status"] == "insufficient_data" for r in snapshot["feature_drift"])
    assert snapshot["prediction_drift"]["status"] == "no_baseline"
    assert snapshot["drifted_feature_count"] == 0


def test_drift_monitor_no_drift_case(drift_monitor: DriftMonitor) -> None:
    """Uses 400 comparison samples, not the MIN_SAMPLES_FOR_DRIFT minimum
    of 30: with 10 baseline bins, a small sample (e.g. 100) genuinely
    produces a noisy PSI even when drawn from the identical distribution
    -- roughly 10 samples/bin is right at the edge of statistical
    reliability for this kind of binned comparison. That sensitivity is a
    real, documented property of PSI (see notebooks/14_monitoring_observability.ipynb's
    Limitations section), not a bug -- this test uses enough samples to
    exercise the "no drift" case reliably rather than flakily.
    """
    rng = np.random.default_rng(1)
    for _ in range(400):
        payload = {
            "numeric_feature": float(rng.normal(loc=10.0, scale=2.0)),
            "category_feature": str(rng.choice(["A", "B", "C"], p=[0.6, 0.3, 0.1])),
        }
        drift_monitor.record_observation(payload, probability=None)

    snapshot = drift_monitor.snapshot()
    numeric_result = next(r for r in snapshot["feature_drift"] if r["feature"] == "numeric_feature")
    assert numeric_result["status"] == "no_significant_shift"
    assert numeric_result["psi"] < 0.1


def test_drift_monitor_detects_significant_numeric_shift(drift_monitor: DriftMonitor) -> None:
    rng = np.random.default_rng(2)
    for _ in range(100):
        # Shifted far from the baseline's mean=10, std=2.
        payload = {"numeric_feature": float(rng.normal(loc=30.0, scale=2.0)), "category_feature": "A"}
        drift_monitor.record_observation(payload, probability=None)

    snapshot = drift_monitor.snapshot()
    numeric_result = next(r for r in snapshot["feature_drift"] if r["feature"] == "numeric_feature")
    assert numeric_result["status"] == "significant_shift"
    assert numeric_result["psi"] >= 0.25
    assert "numeric_feature" in snapshot["drifted_features"]


def test_drift_monitor_detects_categorical_shift(drift_monitor: DriftMonitor) -> None:
    for _ in range(100):
        # Baseline is mostly "A"; feed entirely "C" (the rarest baseline category).
        drift_monitor.record_observation({"numeric_feature": 10.0, "category_feature": "C"}, probability=None)

    snapshot = drift_monitor.snapshot()
    categorical_result = next(r for r in snapshot["feature_drift"] if r["feature"] == "category_feature")
    assert categorical_result["status"] in ("moderate_shift", "significant_shift")


def test_drift_monitor_prediction_drift_no_shift(drift_monitor: DriftMonitor) -> None:
    rng = np.random.default_rng(3)
    baseline_probs = rng.uniform(0.0, 1.0, size=500)
    drift_monitor.set_prediction_baseline(baseline_probs)

    for _ in range(100):
        drift_monitor.record_observation({}, probability=float(rng.uniform(0.0, 1.0)))

    prediction_drift = drift_monitor.snapshot()["prediction_drift"]
    assert prediction_drift["status"] == "no_significant_shift"


def test_drift_monitor_prediction_drift_significant_shift(drift_monitor: DriftMonitor) -> None:
    rng = np.random.default_rng(4)
    baseline_probs = rng.uniform(0.0, 0.3, size=500)
    drift_monitor.set_prediction_baseline(baseline_probs)

    for _ in range(100):
        drift_monitor.record_observation({}, probability=float(rng.uniform(0.7, 1.0)))

    prediction_drift = drift_monitor.snapshot()["prediction_drift"]
    assert prediction_drift["status"] == "significant_shift"


def test_drift_monitor_handles_malformed_observation_values_safely(drift_monitor: DriftMonitor) -> None:
    """A non-numeric value for a numeric feature must be skipped, not
    crash the monitor -- observation recording must never be able to take
    down a request that already succeeded.
    """
    drift_monitor.record_observation({"numeric_feature": "not-a-number", "category_feature": "A"}, probability=None)
    drift_monitor.record_observation({"numeric_feature": None, "category_feature": None}, probability=None)
    snapshot = drift_monitor.snapshot()  # must not raise
    assert snapshot["feature_drift"][0]["sample_size"] == 0


def test_drift_monitor_handles_degenerate_baseline_feature_safely() -> None:
    """A constant-valued (zero-variance) training feature can't produce a
    meaningful PSI baseline -- it must be silently skipped, not crash
    construction.
    """
    constant_df = pd.DataFrame({
        "constant_feature": [5.0] * 100,
        "category_feature": ["A"] * 100,
    })
    monitor = DriftMonitor(constant_df, numeric_features=["constant_feature"], categorical_features=["category_feature"])
    assert "constant_feature" not in monitor.monitored_numeric_features
    snapshot = monitor.snapshot()  # must not raise
    assert snapshot["monitored_numeric_features"] == []


def test_drift_monitor_handles_empty_training_data_safely() -> None:
    empty_df = pd.DataFrame({"numeric_feature": [], "category_feature": []})
    monitor = DriftMonitor(empty_df, numeric_features=["numeric_feature"], categorical_features=["category_feature"])
    snapshot = monitor.snapshot()  # must not raise
    assert snapshot["monitored_numeric_features"] == []


def test_drift_monitor_rolling_window_drops_oldest(drift_monitor: DriftMonitor) -> None:
    small_monitor = DriftMonitor(
        pd.DataFrame({"numeric_feature": np.random.default_rng(0).normal(10, 2, 500), "category_feature": ["A"] * 500}),
        numeric_features=["numeric_feature"], categorical_features=["category_feature"], rolling_window=50,
    )
    for i in range(200):
        small_monitor.record_observation({"numeric_feature": float(i), "category_feature": "A"}, probability=None)

    result = next(r for r in small_monitor.snapshot()["feature_drift"] if r["feature"] == "numeric_feature")
    assert result["sample_size"] == 50


# ---------------------------------------------------------------------------
# API integration -- reuses the real registered Phase 10 model, mirroring
# test_api.py's skip-if-unavailable pattern.
# ---------------------------------------------------------------------------

from src.models.explainability import TUNED_MODEL_PATH  # noqa: E402

if not TUNED_MODEL_PATH.exists():
    pytest.skip(f"{TUNED_MODEL_PATH} not found -- run Phase 7 (and Phase 10 registration) first", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from api.services import feature_row_to_payload  # noqa: E402
from src.models import baseline as bl  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_payload() -> dict:
    features = bl.load_feature_dataset()
    row = features.sample(n=1, random_state=42).iloc[0]
    return feature_row_to_payload(row)


def test_request_id_generated_when_missing(client: TestClient) -> None:
    response = client.get("/health")
    assert "x-request-id" in response.headers
    assert len(response.headers["x-request-id"]) > 0


def test_request_id_preserved_when_provided(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "my-custom-id-42"})
    assert response.headers["x-request-id"] == "my-custom-id-42"


def test_monitoring_metrics_endpoint_returns_expected_shape(client: TestClient, sample_payload: dict) -> None:
    client.post("/predict", json=sample_payload)
    response = client.get("/monitoring/metrics")
    assert response.status_code == 200

    body = response.json()
    for key in ("total_requests", "successful_requests", "failed_requests", "requests_by_endpoint", "latency_ms", "predictions"):
        assert key in body
    assert body["total_requests"] >= 1
    assert body["predictions"]["total_predictions"] >= 1


def test_monitoring_metrics_reflects_a_failed_request(client: TestClient, sample_payload: dict) -> None:
    incomplete_payload = dict(sample_payload)
    del incomplete_payload["age"]

    before = client.get("/monitoring/metrics").json()["failed_requests"]
    client.post("/predict", json=incomplete_payload)
    after = client.get("/monitoring/metrics").json()["failed_requests"]

    assert after > before


def test_monitoring_health_endpoint_reports_healthy(client: TestClient) -> None:
    response = client.get("/monitoring/health")
    assert response.status_code == 200
    body = response.json()
    assert body["api_status"] == "healthy"
    assert body["model_status"] == "available"
    assert body["drift_monitoring_status"] in ("active", "unavailable")


def test_monitoring_health_endpoint_reports_unavailable_model(client: TestClient) -> None:
    original_service = app.state.model_service
    try:
        app.state.model_service = None
        response = client.get("/monitoring/health")
        assert response.status_code == 200
        body = response.json()
        assert body["api_status"] == "unhealthy"
        assert body["model_status"] == "unavailable"
        assert body["model_name"] is None
    finally:
        app.state.model_service = original_service


def test_monitoring_drift_endpoint_returns_expected_shape(client: TestClient) -> None:
    response = client.get("/monitoring/drift")
    assert response.status_code == 200
    body = response.json()
    for key in ("baseline_source", "monitored_numeric_features", "feature_drift", "prediction_drift", "psi_thresholds", "note"):
        assert key in body
    assert "causal" not in body["note"].lower() or "not proof" in body["note"].lower()


def test_monitoring_drift_endpoint_unavailable_when_model_unavailable(client: TestClient) -> None:
    original_service = app.state.model_service
    try:
        app.state.model_service = None
        response = client.get("/monitoring/drift")
        assert response.status_code == 503
    finally:
        app.state.model_service = original_service


def test_prediction_endpoints_update_prediction_metrics(client: TestClient, sample_payload: dict) -> None:
    before = client.get("/monitoring/metrics").json()["predictions"]["successful_predictions"]
    client.post("/customer-analysis", json=sample_payload)
    after = client.get("/monitoring/metrics").json()["predictions"]["successful_predictions"]
    assert after == before + 1


def test_monitoring_endpoints_never_expose_filesystem_paths(client: TestClient) -> None:
    for path in ("/monitoring/metrics", "/monitoring/health", "/monitoring/drift"):
        response = client.get(path)
        body_text = response.text
        assert "/Users/" not in body_text
        assert "sqlite:" not in body_text
