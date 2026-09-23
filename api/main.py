"""Phase 11 FastAPI inference service.

    API Request -> Pydantic validation -> feature preparation
        -> MLflow registered model (Phase 10) -> churn probability
        -> risk level (Phase 9 config) -> SHAP explanation (Phase 8)
        -> next-best action (Phase 9) -> JSON response

This service NEVER retrains, retunes, or modifies the Phase 7 model. It
loads the already-registered MLflow model (Phase 10) exactly once at
startup and reuses Phase 8's SHAP logic and Phase 9's recommendation
rules completely unmodified -- see api/services.py.

Run locally with:
    uvicorn api.main:app --reload

Configuration: this service reads the same configs/mlflow_config.yaml
and configs/recommendation_config.yaml as every other phase. No
machine-specific paths, credentials, or secrets are hard-coded here or
returned by any endpoint. MLflow's tracking URI comes entirely from
configs/mlflow_config.yaml (Phase 10) -- nothing here overrides it with
an environment variable or a hard-coded absolute path.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.dependencies import get_model_service
from api.schemas import (
    CustomerAnalysisRequest,
    CustomerAnalysisResponse,
    ExplanationRequest,
    ExplanationResponse,
    FeatureDriftResult,
    HealthResponse,
    LatencyStats,
    ModelInfoResponse,
    MonitoringDriftResponse,
    MonitoringHealthResponse,
    MonitoringMetricsResponse,
    PredictionDriftResult,
    PredictionMetricsSummary,
    PredictionRequest,
    PredictionResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from api.services import ModelService, ModelUnavailableError
from src.models.baseline import CLASSIFICATION_THRESHOLD
from src.monitoring.logging_config import configure_logging, log_request
from src.monitoring.metrics import PredictionMetrics, RequestMetrics
from src.recommendations.next_best_action import assign_risk_level

configure_logging()
logger = logging.getLogger("api.main")

MAX_REQUEST_BODY_BYTES = 1_000_000  # 1 MB -- generous for a ~130-feature JSON row

# Phase 14: process-lifetime, in-memory only (see src/monitoring/metrics.py's
# docstring for why) -- created once at import time since neither depends on
# the model being loaded. The model-dependent drift monitor instead lives on
# ModelService itself (api/services.py), built once at startup alongside the
# SHAP context it mirrors.
request_metrics = RequestMetrics()
prediction_metrics = PredictionMetrics(classification_threshold=CLASSIFICATION_THRESHOLD)


class MaxBodySizeMiddleware:
    """Rejects oversized request bodies before they reach any route
    handler or Pydantic validation -- a minimal, dependency-free request
    size guard.

    Implemented as a plain ASGI middleware rather than a
    starlette.middleware.base.BaseHTTPMiddleware subclass: that base class
    is known to interfere with app-level `@app.exception_handler(Exception)`
    handlers (it can re-raise an exception the handler already converted
    into a response), which would defeat this API's "never leak a stack
    trace" guarantee. Plain ASGI middleware has no such interaction.
    """

    def __init__(self, app: FastAPI, max_bytes: int = MAX_REQUEST_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        content_length = headers.get(b"content-length")
        if content_length is not None and int(content_length) > self.max_bytes:
            response = JSONResponse(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE, content={"detail": "Request body too large."}
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


class RequestMonitoringMiddleware:
    """Phase 14: records request-level metrics (request_metrics), emits one
    structured access-log line per request (src.monitoring.logging_config),
    and attaches an X-Request-ID to every response -- preserving a
    client-supplied one if present, generating a UUID4 otherwise.

    Plain ASGI, not BaseHTTPMiddleware, for the exact same reason
    MaxBodySizeMiddleware is -- see its docstring. Wrapping the downstream
    call in try/finally means a request is recorded exactly once even if
    something raises before a response is ever sent.
    """

    def __init__(self, app: FastAPI, metrics: RequestMetrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        request_id = headers.get(b"x-request-id", b"").decode() or str(uuid.uuid4())
        method = scope.get("method", "?")
        path = scope.get("path", "?")
        status_holder = {"code": 500}  # fallback if the app never sends a response start

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                status_holder["code"] = message["status"]
                response_headers = list(message.get("headers", []))
                response_headers.append((b"x-request-id", request_id.encode()))
                message = {**message, "headers": response_headers}
            await send(message)

        start = time.perf_counter()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            status_code = status_holder["code"]
            self.metrics.record_request(method, path, status_code, duration_ms)
            log_request(method, path, status_code, duration_ms, request_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the registered model (and everything it needs: SHAP context,
    recommendation config) exactly once at startup -- never per request.

    If loading fails, the app still starts rather than crashing outright,
    but every model-dependent route (via get_model_service) and /health
    both report the failure clearly with HTTP 503. This never silently
    falls back to a different model -- either the configured one loads,
    or the service reports itself unavailable.
    """
    service = ModelService()
    try:
        service.load()
        app.state.model_service = service
        app.state.model_load_error = None
    except ModelUnavailableError as exc:
        logger.error("Startup model load failed: %s", exc)
        app.state.model_service = None
        app.state.model_load_error = str(exc)

    yield

    app.state.model_service = None


app = FastAPI(
    title="Customer Retention Intelligence API",
    description=(
        'Serves the frozen Phase 7 Logistic Regression (`C=0.01`, `class_weight="balanced"`) '
        "via its Phase 10 MLflow Model Registry entry, with Phase 8 SHAP explanations and "
        "Phase 9 next-best-action recommendations layered on top. No model is trained, tuned, "
        "or modified by this service.\n\n"
        "**Read this before interpreting any response:**\n\n"
        "- `churn_probability` is a **model output** (a prediction), not a certainty.\n"
        "- `risk_level` and `priority` come from **business-decision thresholds** "
        "(`configs/recommendation_config.yaml`), not statistically optimal cutoffs.\n"
        "- `top_positive_drivers` / `top_negative_drivers` are `model_signal`s -- SHAP-based "
        "associations the model found for this customer -- **never a causal claim**.\n"
        "- `/recommend` and `/customer-analysis` return **decision-support recommendations**, "
        "not causal treatment effects: taking the recommended action is not guaranteed to "
        "reduce churn."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(MaxBodySizeMiddleware)
# Added last so it wraps outermost (Starlette applies middleware in
# reverse registration order): every request/response passes through it,
# including one MaxBodySizeMiddleware itself rejects with a 413.
app.add_middleware(RequestMonitoringMiddleware, metrics=request_metrics)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: log the real error server-side, return a generic body.
    Never leaks a stack trace, file path, or internal exception message
    to the client.
    """
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, type(exc).__name__)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": "Internal server error."}
    )


def _sanitize_non_json_floats(value: Any) -> Any:
    """FastAPI's default validation-error handler echoes the rejected raw
    value back in the error detail (e.g. "input": nan). Starlette's
    strict-JSON response renderer then refuses to serialize that same NaN/
    Infinity, turning a validation error into a crash. Replacing those
    specific values with their string form keeps the response strict-JSON
    while still telling the client exactly what it sent.
    """
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    if isinstance(value, dict):
        return {key: _sanitize_non_json_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_non_json_floats(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """A NaN/Infinity/malformed field must produce a clean HTTP 422, never
    the HTTP 500 that Starlette's default strict-JSON rendering would
    otherwise raise when echoing a rejected NaN/Infinity value back.
    """
    sanitized_errors = [_sanitize_non_json_floats(dict(error)) for error in exc.errors()]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": jsonable_encoder(sanitized_errors)},
    )


def _safe_call(operation_name: str, func, *args, **kwargs) -> tuple[Any, float]:
    """Run one service-layer call, translating failures into the right HTTP
    status: ValueError (a problem the request schema couldn't catch on its
    own, e.g. a downstream consistency check) -> 422; anything else
    unexpected -> a logged, detail-free 500. Centralizing this here is
    what makes /predict, /predict/explain, /recommend, and
    /customer-analysis all fail the exact same safe way, and is this
    version's actual enforcement point for "no stack trace leaks" -- a
    bare `@app.exception_handler(Exception)` does not intercept exceptions
    raised inside route handlers in this FastAPI/Starlette version.

    Phase 14: also times the call and records the outcome as a prediction
    metric (success timing returned to the caller; failure recorded here).
    Every current caller is one of the four inference endpoints, so that
    classification is accurate -- if a non-prediction caller is ever added,
    split this responsibility out rather than assuming otherwise here.
    Returns (result, duration_ms) on success.
    """
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
    except ValueError as exc:
        prediction_metrics.record_failure()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except HTTPException:
        prediction_metrics.record_failure()
        raise
    except Exception as exc:
        prediction_metrics.record_failure()
        logger.error("Unexpected error during %s: %s", operation_name, type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error."
        ) from exc
    return result, (time.perf_counter() - start) * 1000


def _split_customer_payload(request_model: Any) -> tuple[dict[str, Any], str, Any]:
    """A validated request model -> (raw feature payload dict, customer_id,
    snapshot_date). Never logs the full payload -- only the customer_id is
    ever written to the application log.
    """
    payload = request_model.model_dump()
    customer_id = payload.pop("customer_id")
    snapshot_date = payload.pop("snapshot_date")
    payload.pop("top_k", None)
    return payload, customer_id, snapshot_date


def _record_prediction_observation(
    service: ModelService, payload: dict[str, Any], probability: float, duration_ms: float
) -> None:
    """Phase 14: observe an already-computed prediction -- never
    recomputes anything, and never alters what the endpoint returns.
    Updates prediction_metrics and, if available, the model's drift
    monitor with only its small monitored feature subset (never the full
    ~130-feature payload, never a customer identifier).
    """
    null_field_count = sum(1 for value in payload.values() if value is None)
    prediction_metrics.record_success(
        probability, duration_ms, null_field_count=null_field_count, total_field_count=len(payload)
    )
    if service.drift_monitor is not None:
        service.drift_monitor.record_observation(payload, probability)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["operations"],
    summary="Service health check",
    description="Verifies the configured registered model is loaded and usable. Never exposes filesystem paths or credentials.",
)
async def health(request: Request):
    service: ModelService | None = getattr(request.app.state, "model_service", None)
    if service is not None and service.is_ready:
        return HealthResponse(status="healthy", model=service.model_name, model_version=service.model_version)

    detail = getattr(request.app.state, "model_load_error", None) or "Model not loaded."
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=HealthResponse(status="unhealthy", detail=detail).model_dump(),
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["operations"],
    summary="Registered model metadata",
    description="Safe, whitelisted model metadata only -- never raw MLflow internals, tracking URIs, or environment variables.",
)
async def model_info(service: ModelService = Depends(get_model_service)) -> ModelInfoResponse:
    return ModelInfoResponse(**service.model_metadata())


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    summary="Predict churn probability and risk level",
    description=(
        "Scores one customer/snapshot's full raw feature row with the frozen Phase 7 model. "
        "`risk_level` is assigned using Phase 9's business-decision thresholds, never "
        "re-implemented here."
    ),
)
async def predict(request: PredictionRequest, service: ModelService = Depends(get_model_service)) -> PredictionResponse:
    payload, customer_id, snapshot_date = _split_customer_payload(request)
    logger.info("POST /predict customer_id=%s", customer_id)

    result, inference_ms = _safe_call("predict", service.predict, payload)
    _record_prediction_observation(service, payload, result["churn_probability"], inference_ms)

    return PredictionResponse(
        customer_id=customer_id,
        snapshot_date=snapshot_date,
        churn_probability=result["churn_probability"],
        risk_level=result["risk_level"],
        model_name=service.model_name,
        model_version=service.model_version,
    )


@app.post(
    "/predict/explain",
    response_model=ExplanationResponse,
    tags=["inference"],
    summary="Predict churn probability with a SHAP explanation",
    description=(
        "Same scoring as /predict, plus the Phase 8 SHAP explanation (reused unmodified): "
        "the top positive (risk-raising) and negative (risk-lowering) model_signals for this "
        "specific customer, with their own raw feature values attached."
    ),
)
async def predict_explain(
    request: ExplanationRequest, service: ModelService = Depends(get_model_service)
) -> ExplanationResponse:
    payload, customer_id, snapshot_date = _split_customer_payload(request)
    logger.info("POST /predict/explain customer_id=%s", customer_id)

    explanation, inference_ms = _safe_call("predict_explain", service.explain, payload, top_k=request.top_k)

    probability = explanation["predicted_probability"]
    _record_prediction_observation(service, payload, probability, inference_ms)
    risk_level = assign_risk_level(probability, service.recommendation_config["risk_thresholds"])

    return ExplanationResponse(
        customer_id=customer_id,
        snapshot_date=snapshot_date,
        churn_probability=probability,
        risk_level=risk_level,
        top_positive_drivers=explanation["top_positive_drivers"],
        top_negative_drivers=explanation["top_negative_drivers"],
        shap_base_value=explanation["base_value"],
        model_name=service.model_name,
        model_version=service.model_version,
    )


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
    tags=["decision-support"],
    summary="Next-best-action recommendation",
    description=(
        "Reuses Phase 9's business-rule recommendation engine unmodified: churn probability, "
        "risk level, priority, a recommended action, its reason, and the exact supporting "
        "feature values that triggered it. Decision support only -- not a causal estimate."
    ),
)
async def recommend(
    request: RecommendationRequest, service: ModelService = Depends(get_model_service)
) -> RecommendationResponse:
    payload, customer_id, snapshot_date = _split_customer_payload(request)
    logger.info("POST /recommend customer_id=%s", customer_id)

    recommendation, inference_ms = _safe_call("recommend", service.recommend, payload, customer_id, snapshot_date)
    _record_prediction_observation(service, payload, recommendation["churn_probability"], inference_ms)

    return RecommendationResponse(
        customer_id=customer_id,
        snapshot_date=snapshot_date,
        churn_probability=recommendation["churn_probability"],
        risk_level=recommendation["risk_level"],
        priority=recommendation["priority"],
        primary_action=recommendation["primary_action"],
        action_reason=recommendation["action_reason"],
        supporting_features=recommendation["supporting_features"],
        model_name=service.model_name,
        model_version=service.model_version,
    )


@app.post(
    "/customer-analysis",
    response_model=CustomerAnalysisResponse,
    tags=["decision-support"],
    summary="Combined prediction + explanation + recommendation",
    description=(
        "One response combining /predict, /predict/explain, and /recommend for the same "
        "customer/snapshot -- built for the Streamlit dashboard. `churn_probability` is "
        "computed once and shared across all three sections, so they are always consistent "
        "with each other for the same request."
    ),
)
async def customer_analysis(
    request: CustomerAnalysisRequest, service: ModelService = Depends(get_model_service)
) -> CustomerAnalysisResponse:
    payload, customer_id, snapshot_date = _split_customer_payload(request)
    logger.info("POST /customer-analysis customer_id=%s", customer_id)

    def _run_full_analysis():
        probability = service.predict_probability(payload)
        risk_level = assign_risk_level(probability, service.recommendation_config["risk_thresholds"])
        explanation = service.explain(payload, top_k=request.top_k)
        recommendation = service.recommend(payload, customer_id, snapshot_date, probability=probability)
        return probability, risk_level, explanation, recommendation

    (probability, risk_level, explanation, recommendation), inference_ms = _safe_call(
        "customer_analysis", _run_full_analysis
    )
    _record_prediction_observation(service, payload, probability, inference_ms)

    prediction_response = PredictionResponse(
        customer_id=customer_id,
        snapshot_date=snapshot_date,
        churn_probability=probability,
        risk_level=risk_level,
        model_name=service.model_name,
        model_version=service.model_version,
    )
    explanation_response = ExplanationResponse(
        customer_id=customer_id,
        snapshot_date=snapshot_date,
        churn_probability=explanation["predicted_probability"],
        risk_level=risk_level,
        top_positive_drivers=explanation["top_positive_drivers"],
        top_negative_drivers=explanation["top_negative_drivers"],
        shap_base_value=explanation["base_value"],
        model_name=service.model_name,
        model_version=service.model_version,
    )
    recommendation_response = RecommendationResponse(
        customer_id=customer_id,
        snapshot_date=snapshot_date,
        churn_probability=recommendation["churn_probability"],
        risk_level=recommendation["risk_level"],
        priority=recommendation["priority"],
        primary_action=recommendation["primary_action"],
        action_reason=recommendation["action_reason"],
        supporting_features=recommendation["supporting_features"],
        model_name=service.model_name,
        model_version=service.model_version,
    )

    return CustomerAnalysisResponse(
        customer={"customer_id": customer_id, "snapshot_date": snapshot_date},
        prediction=prediction_response,
        explanation=explanation_response,
        recommendation=recommendation_response,
    )


# ---------------------------------------------------------------------------
# Phase 14: monitoring / observability endpoints.
#
# These are read-only views over request_metrics, prediction_metrics, and
# (per-request) service.drift_monitor -- all of which are only ever
# updated as a side effect of real traffic through the endpoints above.
# Nothing here performs inference, SHAP, or recommendation logic, and
# nothing here returns a customer identifier, a raw feature value, a
# filesystem path, or a credential.
# ---------------------------------------------------------------------------

@app.get(
    "/monitoring/metrics",
    response_model=MonitoringMetricsResponse,
    tags=["monitoring"],
    summary="Request and prediction metrics",
    description=(
        "In-process, in-memory metrics collected since this process started (reset on restart): "
        "HTTP request counts/latency (every endpoint) and model-inference outcome counts "
        "(the four prediction endpoints only). Never exposes a customer identifier or raw "
        "feature value."
    ),
)
async def monitoring_metrics() -> MonitoringMetricsResponse:
    request_snapshot = request_metrics.snapshot()
    prediction_snapshot = prediction_metrics.snapshot()
    return MonitoringMetricsResponse(
        total_requests=request_snapshot["total_requests"],
        successful_requests=request_snapshot["successful_requests"],
        failed_requests=request_snapshot["failed_requests"],
        requests_by_endpoint=request_snapshot["requests_by_endpoint"],
        requests_by_method=request_snapshot["requests_by_method"],
        requests_by_status=request_snapshot["requests_by_status"],
        latency_ms=LatencyStats(**request_snapshot["latency_ms"]),
        predictions=PredictionMetricsSummary(**prediction_snapshot),
        generated_at=datetime.now(timezone.utc),
    )


@app.get(
    "/monitoring/health",
    response_model=MonitoringHealthResponse,
    tags=["monitoring"],
    summary="Operational health of the API, model, and monitoring subsystem",
    description=(
        "A richer, monitoring-oriented view than GET /health: also reports whether drift "
        "monitoring has a usable baseline. Never exposes MLflow internals, tracking URIs, "
        "or environment variables -- see GET /model-info for the same discipline."
    ),
)
async def monitoring_health(request: Request) -> MonitoringHealthResponse:
    service: ModelService | None = getattr(request.app.state, "model_service", None)
    model_ready = service is not None and service.is_ready

    return MonitoringHealthResponse(
        api_status="healthy" if model_ready else "unhealthy",
        model_status="available" if model_ready else "unavailable",
        model_name=service.model_name if model_ready else None,
        model_version=service.model_version if model_ready else None,
        drift_monitoring_status="active" if (model_ready and service.drift_monitor is not None) else "unavailable",
        generated_at=datetime.now(timezone.utc),
    )


@app.get(
    "/monitoring/drift",
    response_model=MonitoringDriftResponse,
    tags=["monitoring"],
    summary="Data and prediction drift signals",
    description=(
        "Population Stability Index (PSI) for a small set of SHAP-important numeric features, "
        "the categorical features, and the model's own prediction distribution, each compared "
        "against a Phase 7 training-data baseline. PSI is a monitoring SIGNAL for investigation, "
        "never proof that model performance has degraded -- see the `note` field."
    ),
)
async def monitoring_drift(service: ModelService = Depends(get_model_service)) -> MonitoringDriftResponse:
    if service.drift_monitor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Drift monitoring is unavailable (its baseline report could not be loaded at startup).",
        )

    snapshot = service.drift_monitor.snapshot()
    return MonitoringDriftResponse(
        baseline_source=snapshot["baseline_source"],
        monitored_numeric_features=snapshot["monitored_numeric_features"],
        monitored_categorical_features=snapshot["monitored_categorical_features"],
        feature_drift=[FeatureDriftResult(**item) for item in snapshot["feature_drift"]],
        drifted_feature_count=snapshot["drifted_feature_count"],
        drifted_features=snapshot["drifted_features"],
        prediction_drift=PredictionDriftResult(**snapshot["prediction_drift"]),
        psi_thresholds=snapshot["psi_thresholds"],
        note=snapshot["note"],
        generated_at=datetime.now(timezone.utc),
    )
