"""HTTP client for the Phase 11 FastAPI inference service, plus the
pure, non-ML helpers needed to build a request payload from a real
feature row.

This module never predicts, explains, or recommends anything itself --
it only sends requests to FastAPI and returns whatever FastAPI decides
(or a clearly-labeled connection/timeout/parsing failure if the backend
couldn't be reached at all). FastAPI remains the single source of truth
for inference, explanations, and recommendations.

The API base URL comes from the `API_BASE_URL` environment variable
(default `http://localhost:8000`) -- never hard-coded to a production
address.

With DASHBOARD_API_MODE=inprocess (set by the Streamlit Community Cloud
entry point, streamlit_app.py), the same requests are sent to the same
FastAPI app running inside this process (api/inprocess.py) instead of over
HTTP -- still FastAPI deciding every response, just without a network hop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import requests

DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 10.0
API_MODE_ENV_VAR = "DASHBOARD_API_MODE"
INPROCESS_API_MODE = "inprocess"

# The one column in customer_features.parquet that is a label, not a
# model feature (src.models.baseline.TARGET_COLUMN). Every other column
# is either an identifier the API schema accepts (customer_id,
# snapshot_date) or a raw feature it requires -- see
# src.models.baseline.get_feature_columns, which api/schemas.py itself is
# generated from, so this assumption holds by construction.
TARGET_COLUMN = "churn_next_30_days"


def get_api_base_url() -> str:
    """Read fresh on every call (not cached at import time) so tests and
    a running app can both override it via the environment.
    """
    return os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def is_inprocess_mode() -> bool:
    """True when requests go to the in-process FastAPI app (see module
    docstring). Read fresh on every call, like get_api_base_url.
    """
    return os.environ.get(API_MODE_ENV_VAR, "").strip().lower() == INPROCESS_API_MODE


@dataclass
class APIResult:
    """A uniform outcome for every API call, so dashboard code never has
    to branch on exception types or guess at a response's shape.
    """

    ok: bool
    data: Any = None
    status_code: int | None = None
    error: str | None = None
    raw: dict = field(default_factory=dict)


def _format_error_detail(detail: Any) -> str | None:
    """FastAPI's `detail` is either a plain string (HTTPException) or a
    list of {"loc", "msg", ...} dicts (a Pydantic validation error). Turn
    either into one readable sentence -- never echo a raw stack trace.
    """
    if detail is None:
        return None
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        messages = []
        for item in detail:
            if isinstance(item, dict) and "msg" in item:
                location = ".".join(str(part) for part in item.get("loc", []) if part != "body")
                messages.append(f"{location}: {item['msg']}" if location else str(item["msg"]))
        if messages:
            return "; ".join(messages)
    return None


def _request(
    method: str, path: str, json_body: dict | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> APIResult:
    if is_inprocess_mode():
        # Imported lazily: only this mode loads the model into this process.
        from api import inprocess

        try:
            response = inprocess.request(method, path, json_body)
        except Exception as exc:  # noqa: BLE001 -- surfaced as a labeled failure, like a network error
            return APIResult(ok=False, error=f"The in-process API failed to handle the request ({type(exc).__name__}).")
        return _parse_response(response)

    url = f"{get_api_base_url()}{path}"

    try:
        response = requests.request(method, url, json=json_body, timeout=timeout)
    except requests.exceptions.ConnectTimeout:
        return APIResult(ok=False, error=f"Connection to the API timed out ({url}). Is FastAPI running?")
    except requests.exceptions.ConnectionError:
        return APIResult(
            ok=False,
            error=(
                f"Could not connect to the API at {get_api_base_url()}. "
                "Start it with: uvicorn api.main:app --reload"
            ),
        )
    except requests.exceptions.Timeout:
        return APIResult(ok=False, error="The API took too long to respond.")
    except requests.exceptions.RequestException:
        return APIResult(ok=False, error="An unexpected network error occurred while contacting the API.")

    return _parse_response(response)


def _parse_response(response) -> APIResult:
    """Turn an HTTP response (a `requests` response, or the in-process
    client's equivalent) into an APIResult.
    """
    try:
        body = response.json()
    except ValueError:
        return APIResult(
            ok=False,
            status_code=response.status_code,
            error="The API returned a response that could not be parsed.",
        )

    if response.status_code >= 400:
        detail = body.get("detail") if isinstance(body, dict) else None
        message = _format_error_detail(detail) or f"The API returned an error (HTTP {response.status_code})."
        return APIResult(ok=False, status_code=response.status_code, data=body, error=message)

    if not isinstance(body, dict):
        return APIResult(ok=False, status_code=response.status_code, error="The API returned an unexpected response structure.")

    return APIResult(ok=True, status_code=response.status_code, data=body)


def get_health(timeout: float = DEFAULT_TIMEOUT_SECONDS) -> APIResult:
    return _request("GET", "/health", timeout=timeout)


def get_model_info(timeout: float = DEFAULT_TIMEOUT_SECONDS) -> APIResult:
    return _request("GET", "/model-info", timeout=timeout)


def get_monitoring_metrics(timeout: float = DEFAULT_TIMEOUT_SECONDS) -> APIResult:
    """Phase 14: request/prediction metrics from GET /monitoring/metrics."""
    return _request("GET", "/monitoring/metrics", timeout=timeout)


def get_monitoring_health(timeout: float = DEFAULT_TIMEOUT_SECONDS) -> APIResult:
    """Phase 14: operational health from GET /monitoring/health."""
    return _request("GET", "/monitoring/health", timeout=timeout)


def get_monitoring_drift(timeout: float = DEFAULT_TIMEOUT_SECONDS) -> APIResult:
    """Phase 14: data/prediction drift signals from GET /monitoring/drift."""
    return _request("GET", "/monitoring/drift", timeout=timeout)


def predict(payload: dict, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> APIResult:
    return _request("POST", "/predict", json_body=payload, timeout=timeout)


def explain_prediction(payload: dict, top_k: int = 5, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> APIResult:
    return _request("POST", "/predict/explain", json_body={**payload, "top_k": top_k}, timeout=timeout)


def recommend(payload: dict, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> APIResult:
    return _request("POST", "/recommend", json_body=payload, timeout=timeout)


def customer_analysis(payload: dict, top_k: int = 5, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> APIResult:
    return _request("POST", "/customer-analysis", json_body={**payload, "top_k": top_k}, timeout=timeout)


# ---------------------------------------------------------------------------
# Payload construction from a real feature row -- pure data formatting,
# not business logic: no prediction, scoring, imputation, or preprocessing
# happens here. A legitimately missing value (NaN in the source parquet)
# is preserved as JSON null so the API's own imputers (already fit in
# Phase 5) handle it, exactly as the API expects.
# ---------------------------------------------------------------------------

def build_customer_payload(row: pd.Series) -> dict[str, Any]:
    """Convert one real row from customer_features.parquet into a
    JSON-safe request payload for /predict, /predict/explain, /recommend,
    or /customer-analysis: drop the label column, convert NaN -> None,
    numpy/pandas scalars -> native Python types, timestamps -> ISO date
    strings. Every other column is passed through unchanged -- no feature
    name is invented and no value is fabricated.
    """
    payload: dict[str, Any] = {}
    for column, value in row.items():
        if column == TARGET_COLUMN:
            continue
        payload[column] = _to_json_safe(value)
    return payload


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
