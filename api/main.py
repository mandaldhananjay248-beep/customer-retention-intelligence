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
from contextlib import asynccontextmanager
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
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from api.services import ModelService, ModelUnavailableError
from src.recommendations.next_best_action import assign_risk_level

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("api.main")

MAX_REQUEST_BODY_BYTES = 1_000_000  # 1 MB -- generous for a ~130-feature JSON row


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


def _safe_call(operation_name: str, func, *args, **kwargs) -> Any:
    """Run one service-layer call, translating failures into the right HTTP
    status: ValueError (a problem the request schema couldn't catch on its
    own, e.g. a downstream consistency check) -> 422; anything else
    unexpected -> a logged, detail-free 500. Centralizing this here is
    what makes /predict, /predict/explain, /recommend, and
    /customer-analysis all fail the exact same safe way, and is this
    version's actual enforcement point for "no stack trace leaks" -- a
    bare `@app.exception_handler(Exception)` does not intercept exceptions
    raised inside route handlers in this FastAPI/Starlette version.
    """
    try:
        return func(*args, **kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected error during %s: %s", operation_name, type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error."
        ) from exc


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

    result = _safe_call("predict", service.predict, payload)

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

    explanation = _safe_call("predict_explain", service.explain, payload, top_k=request.top_k)

    probability = explanation["predicted_probability"]
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

    recommendation = _safe_call("recommend", service.recommend, payload, customer_id, snapshot_date)

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

    probability, risk_level, explanation, recommendation = _safe_call("customer_analysis", _run_full_analysis)

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
