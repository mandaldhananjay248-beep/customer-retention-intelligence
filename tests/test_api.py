"""Tests for the Phase 11 FastAPI inference service (api/main.py).

Uses a single module-scoped TestClient so the (real) registered Phase 10
model, SHAP context, and recommendation config are loaded exactly once
for the whole file -- mirroring how the real app boots, and avoiding a
~1s reload per test. Sample request payloads come from real rows in
data/processed/customer_features.parquet (via api.services.feature_row_to_payload),
never invented feature values.

Skipped entirely if Phase 10's registered model isn't available yet
(mirrors the "skip if the artifact doesn't exist" pattern used in
test_explainability.py / test_next_best_action.py / test_mlflow_tracking.py).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.models import baseline as bl
from src.models.explainability import TUNED_MODEL_PATH
from src.recommendations.next_best_action import PRIORITY_LEVELS, RISK_LEVELS

if not TUNED_MODEL_PATH.exists():
    pytest.skip(
        f"{TUNED_MODEL_PATH} not found -- run Phase 7 (and Phase 10 registration) first",
        allow_module_level=True,
    )

from api.main import app  # noqa: E402  (import after the skip check above)
from api.services import feature_row_to_payload  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_payload() -> dict:
    features = bl.load_feature_dataset()
    row = features.sample(n=1, random_state=42).iloc[0]
    return feature_row_to_payload(row)


@pytest.fixture(scope="module")
def high_risk_payload(client: TestClient) -> dict:
    """A real row whose predicted probability lands in "high"/"critical",
    so explanation/recommendation tests exercise a non-trivial action
    (not always "no_intervention").
    """
    features = bl.load_feature_dataset()
    sample = features.sample(n=300, random_state=7)

    best_payload = None
    best_probability = -1.0
    for _, row in sample.iterrows():
        payload = feature_row_to_payload(row)
        response = client.post("/predict", json=payload)
        probability = response.json()["churn_probability"]
        if probability > best_probability:
            best_probability = probability
            best_payload = payload

    return best_payload


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_reports_model_metadata(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["model"] == "customer-churn-logistic-regression"
    assert body["model_version"] is not None


def test_health_never_exposes_filesystem_paths(client: TestClient) -> None:
    body = client.get("/health").json()
    serialized = json.dumps(body)
    assert "/Users/" not in serialized
    assert "sqlite:" not in serialized
    assert ".db" not in serialized


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

def test_model_info_returns_expected_phase7_hyperparameters(client: TestClient) -> None:
    body = client.get("/model-info").json()
    assert body["model_name"] == "customer-churn-logistic-regression"
    assert body["model_type"] == "LogisticRegression"
    assert body["C"] == 0.01
    assert body["class_weight"] == "balanced"
    assert "model_version" in body


def test_model_info_does_not_expose_mlflow_internals(client: TestClient) -> None:
    body = client.get("/model-info").json()
    assert set(body.keys()) == {"model_name", "model_version", "model_type", "C", "class_weight"}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_predict_valid_request_returns_200(client: TestClient, sample_payload: dict) -> None:
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200


def test_predict_probability_in_valid_range(client: TestClient, sample_payload: dict) -> None:
    body = client.post("/predict", json=sample_payload).json()
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_returns_correct_model_and_version(client: TestClient, sample_payload: dict) -> None:
    body = client.post("/predict", json=sample_payload).json()
    assert body["model_name"] == "customer-churn-logistic-regression"
    assert body["model_version"] is not None


def test_predict_risk_level_is_valid(client: TestClient, sample_payload: dict) -> None:
    body = client.post("/predict", json=sample_payload).json()
    assert body["risk_level"] in RISK_LEVELS


def test_predict_echoes_customer_identity(client: TestClient, sample_payload: dict) -> None:
    body = client.post("/predict", json=sample_payload).json()
    assert body["customer_id"] == sample_payload["customer_id"]
    assert body["snapshot_date"] == sample_payload["snapshot_date"]


def test_predict_is_deterministic(client: TestClient, sample_payload: dict) -> None:
    first = client.post("/predict", json=sample_payload).json()
    second = client.post("/predict", json=sample_payload).json()
    assert first["churn_probability"] == second["churn_probability"]
    assert first["risk_level"] == second["risk_level"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_predict_missing_required_feature_returns_422(client: TestClient, sample_payload: dict) -> None:
    incomplete = dict(sample_payload)
    del incomplete["age"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_missing_customer_id_returns_422(client: TestClient, sample_payload: dict) -> None:
    incomplete = dict(sample_payload)
    del incomplete["customer_id"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_invalid_numeric_string_returns_422(client: TestClient, sample_payload: dict) -> None:
    invalid = dict(sample_payload)
    invalid["age"] = "not-a-number"
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_nan_value_returns_422_not_500(client: TestClient, sample_payload: dict) -> None:
    """NaN can't be sent through httpx's json= convenience (it refuses to
    encode it client-side), so the raw body is built and sent manually --
    this is also what proves the server-side sanitization of the echoed
    invalid value (see api/main.py's validation_exception_handler) works,
    since Starlette's default strict-JSON renderer would otherwise crash
    trying to echo the NaN back, turning this into an unhandled 500.
    """
    invalid = dict(sample_payload)
    invalid["age"] = float("nan")
    raw_body = json.dumps(invalid)  # Python's json.dumps allows a bare NaN token
    response = client.post("/predict", content=raw_body, headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_predict_infinite_value_returns_422(client: TestClient, sample_payload: dict) -> None:
    invalid = dict(sample_payload)
    invalid["age"] = float("inf")
    raw_body = json.dumps(invalid)
    response = client.post("/predict", content=raw_body, headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_predict_malformed_date_returns_422(client: TestClient, sample_payload: dict) -> None:
    invalid = dict(sample_payload)
    invalid["snapshot_date"] = "not-a-date"
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_unexpected_field_returns_422(client: TestClient, sample_payload: dict) -> None:
    """No invented feature names are accepted -- the schema forbids extras."""
    invalid = dict(sample_payload)
    invalid["a_feature_that_does_not_exist"] = 1.0
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_null_value_for_legitimately_missing_feature_is_accepted(client: TestClient, sample_payload: dict) -> None:
    """A present-but-null feature (e.g. a customer with no ticket history)
    must be accepted and imputed downstream, not rejected -- this is
    different from an entirely missing key.
    """
    payload = dict(sample_payload)
    payload["avg_satisfaction_90d"] = None
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def test_explain_returns_200_with_shap_explanation(client: TestClient, sample_payload: dict) -> None:
    response = client.post("/predict/explain", json={**sample_payload, "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert "top_positive_drivers" in body
    assert "top_negative_drivers" in body
    assert "shap_base_value" in body


def test_explain_drivers_reference_valid_features(client: TestClient, high_risk_payload: dict) -> None:
    features = bl.load_feature_dataset()
    numeric_columns, categorical_columns = bl.get_feature_columns(features)
    valid_display_prefixes = set(numeric_columns) | {c for c in categorical_columns}

    body = client.post("/predict/explain", json={**high_risk_payload, "top_k": 5}).json()
    all_drivers = body["top_positive_drivers"] + body["top_negative_drivers"]
    assert len(all_drivers) > 0

    for driver in all_drivers:
        feature_name = driver["feature"]
        is_valid = feature_name in numeric_columns or any(
            feature_name.startswith(f"{col}_") for col in categorical_columns
        )
        assert is_valid, f"SHAP driver references unknown feature: {feature_name}"


def test_explain_no_nan_or_inf_shap_values(client: TestClient, sample_payload: dict) -> None:
    body = client.post("/predict/explain", json={**sample_payload, "top_k": 10}).json()
    all_drivers = body["top_positive_drivers"] + body["top_negative_drivers"]
    for driver in all_drivers:
        value = driver["shap_value"]
        assert not np.isnan(value)
        assert not np.isinf(value)
    assert not np.isnan(body["shap_base_value"])
    assert not np.isinf(body["shap_base_value"])


def test_explain_includes_disclaimer(client: TestClient, sample_payload: dict) -> None:
    body = client.post("/predict/explain", json=sample_payload).json()
    assert "model_signal" in body["disclaimer"]
    assert "causal" in body["disclaimer"].lower() or "not evidence" in body["disclaimer"].lower()


def test_explain_probability_matches_predict(client: TestClient, sample_payload: dict) -> None:
    predict_body = client.post("/predict", json=sample_payload).json()
    explain_body = client.post("/predict/explain", json=sample_payload).json()
    assert predict_body["churn_probability"] == pytest.approx(explain_body["churn_probability"])
    assert predict_body["risk_level"] == explain_body["risk_level"]


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def test_recommend_returns_valid_action(client: TestClient, sample_payload: dict) -> None:
    body = client.post("/recommend", json=sample_payload).json()
    assert body["primary_action"]
    assert body["action_reason"]


def test_recommend_priority_is_valid(client: TestClient, sample_payload: dict) -> None:
    body = client.post("/recommend", json=sample_payload).json()
    assert body["priority"] in PRIORITY_LEVELS


def test_recommend_includes_supporting_signals_for_triggered_action(client: TestClient, high_risk_payload: dict) -> None:
    body = client.post("/recommend", json=high_risk_payload).json()
    if body["primary_action"] != "No intervention":
        assert len(body["supporting_features"]) > 0


def test_recommend_includes_decision_support_disclaimer(client: TestClient, sample_payload: dict) -> None:
    body = client.post("/recommend", json=sample_payload).json()
    assert "decision support" in body["disclaimer"].lower()
    assert "not" in body["disclaimer"].lower()


def test_recommend_matches_predict_risk_level(client: TestClient, sample_payload: dict) -> None:
    predict_body = client.post("/predict", json=sample_payload).json()
    recommend_body = client.post("/recommend", json=sample_payload).json()
    assert predict_body["risk_level"] == recommend_body["risk_level"]
    assert predict_body["churn_probability"] == pytest.approx(recommend_body["churn_probability"])


# ---------------------------------------------------------------------------
# Combined analysis
# ---------------------------------------------------------------------------

def test_customer_analysis_returns_all_sections(client: TestClient, sample_payload: dict) -> None:
    response = client.post("/customer-analysis", json={**sample_payload, "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"customer", "prediction", "explanation", "recommendation"}


def test_customer_analysis_is_internally_consistent(client: TestClient, sample_payload: dict) -> None:
    body = client.post("/customer-analysis", json=sample_payload).json()
    probability = body["prediction"]["churn_probability"]

    assert body["explanation"]["churn_probability"] == pytest.approx(probability)
    assert body["recommendation"]["churn_probability"] == pytest.approx(probability)
    assert body["prediction"]["risk_level"] == body["explanation"]["risk_level"] == body["recommendation"]["risk_level"]
    assert body["customer"]["customer_id"] == sample_payload["customer_id"]


def test_customer_analysis_matches_individual_endpoints(client: TestClient, high_risk_payload: dict) -> None:
    combined = client.post("/customer-analysis", json={**high_risk_payload, "top_k": 5}).json()
    individual_predict = client.post("/predict", json=high_risk_payload).json()
    individual_recommend = client.post("/recommend", json=high_risk_payload).json()

    assert combined["prediction"]["churn_probability"] == pytest.approx(individual_predict["churn_probability"])
    assert combined["recommendation"]["primary_action"] == individual_recommend["primary_action"]


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

def test_model_unavailable_returns_503(client: TestClient, sample_payload: dict) -> None:
    """Simulates the model service becoming unavailable, without ever
    creating a second TestClient/lifespan cycle against the shared `app`
    singleton (which would clobber the module-scoped client's state) --
    the original service is saved and restored so later tests are
    unaffected regardless of execution order.
    """
    original_service = app.state.model_service
    original_error = getattr(app.state, "model_load_error", None)
    try:
        app.state.model_service = None
        app.state.model_load_error = "Simulated unavailability for testing."

        assert client.get("/health").status_code == 503
        assert client.get("/model-info").status_code == 503
        assert client.post("/predict", json=sample_payload).status_code == 503
        assert client.post("/predict/explain", json=sample_payload).status_code == 503
        assert client.post("/recommend", json=sample_payload).status_code == 503
        assert client.post("/customer-analysis", json=sample_payload).status_code == 503
    finally:
        app.state.model_service = original_service
        app.state.model_load_error = original_error


def test_service_restored_after_simulated_outage(client: TestClient, sample_payload: dict) -> None:
    """Guards the fixture-hygiene of the test above: normal operation must
    resume immediately after the simulated outage.
    """
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200


def test_unexpected_internal_error_returns_500_without_leaking_details(client: TestClient, sample_payload: dict) -> None:
    from api.services import ModelService

    with patch.object(ModelService, "predict", side_effect=RuntimeError("secret internal path /etc/passwd leaked")):
        response = client.post("/predict", json=sample_payload)

    assert response.status_code == 500
    body = response.json()
    assert body == {"detail": "Internal server error."}
    assert "secret" not in json.dumps(body)
    assert "/etc/passwd" not in json.dumps(body)


def test_request_too_large_returns_413(client: TestClient, sample_payload: dict) -> None:
    oversized = dict(sample_payload)
    oversized["country"] = "x" * 2_000_000  # far past MAX_REQUEST_BODY_BYTES
    response = client.post("/predict", json=oversized)
    assert response.status_code == 413


# ---------------------------------------------------------------------------
# API documentation
# ---------------------------------------------------------------------------

def test_docs_endpoints_are_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_openapi_schema_documents_all_endpoints(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    paths = set(schema["paths"].keys())
    assert {"/health", "/model-info", "/predict", "/predict/explain", "/recommend", "/customer-analysis"} <= paths
