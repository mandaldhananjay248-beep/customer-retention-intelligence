"""FastAPI dependency wiring for the Phase 11 inference service.

The single shared ModelService instance lives on `app.state`, built once
during the app's lifespan startup (see api/main.py) -- these dependencies
just retrieve it, never construct or reload it per request.
"""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from api.services import ModelService


def get_model_service(request: Request) -> ModelService:
    """The shared ModelService for routes that need the model, SHAP
    context, or recommendation config. Raises HTTP 503 -- never a bare
    500 -- if the model failed to load at startup, so a client sees a
    clear "service unavailable" rather than an unexplained error.
    """
    service: ModelService | None = getattr(request.app.state, "model_service", None)
    if service is None or not service.is_ready:
        detail = "Model service is unavailable: the configured registered model could not be loaded."
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
    return service
