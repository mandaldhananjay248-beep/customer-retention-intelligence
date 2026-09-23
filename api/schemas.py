"""Pydantic request/response schemas for the Phase 11 inference API.

`CustomerFeaturePayload` is generated once, directly from the same
`data/processed/customer_features.parquet` columns Phases 5-10 already
use (via `src.models.baseline.get_feature_columns`) -- so this schema can
never drift from the model's actual feature requirements, and no feature
name is invented here. `PredictionRequest`, `ExplanationRequest`,
`RecommendationRequest`, and `CustomerAnalysisRequest` all subclass it
rather than each redeclaring the ~130 feature fields.

Every feature is a REQUIRED key (the client must send it), but its VALUE
may be null: the trained pipeline's own imputers (Phase 5) already handle
a legitimately missing value, e.g. `avg_satisfaction_90d` for a customer
with no support-ticket history. An explicit NaN/Infinity float, by
contrast, is rejected outright (`allow_inf_nan=False`) -- that is never a
legitimate business value, only a malformed request.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.models import baseline as bl


def _feature_field_definitions() -> dict[str, tuple[type, Any]]:
    """One (type, FieldInfo) pair per raw model feature column, typed from
    the real feature dataset's dtypes (bool columns stay bool; everything
    else numeric becomes float, categoricals become str).
    """
    features = bl.load_feature_dataset()
    numeric_columns, categorical_columns = bl.get_feature_columns(features)

    fields: dict[str, tuple[type, Any]] = {}
    for column in numeric_columns:
        if features[column].dtype == bool:
            fields[column] = (Optional[bool], Field(..., description=f"Raw feature '{column}'."))
        else:
            fields[column] = (
                Optional[float],
                Field(..., allow_inf_nan=False, description=f"Raw numeric feature '{column}'."),
            )
    for column in categorical_columns:
        fields[column] = (Optional[str], Field(..., description=f"Raw categorical feature '{column}'."))

    return fields


def _build_customer_feature_payload() -> type[BaseModel]:
    identifier_fields: dict[str, tuple[type, Any]] = {
        "customer_id": (str, Field(..., min_length=1, description="Customer identifier, passed through unchanged.")),
        "snapshot_date": (date, Field(..., description="The feature snapshot date (YYYY-MM-DD).")),
    }
    return create_model(
        "CustomerFeaturePayload",
        __doc__=(
            "One customer/snapshot's full raw feature row -- the exact columns the frozen "
            "Phase 7 pipeline expects, read live from data/processed/customer_features.parquet "
            "via src.models.baseline.get_feature_columns. No feature name here is invented."
        ),
        __config__=ConfigDict(extra="forbid"),
        **identifier_fields,
        **_feature_field_definitions(),
    )


CustomerFeaturePayload: type[BaseModel] = _build_customer_feature_payload()


class PredictionRequest(CustomerFeaturePayload):
    """Request body for POST /predict -- one customer/snapshot's full raw feature row."""


class ExplanationRequest(CustomerFeaturePayload):
    """Request body for POST /predict/explain."""

    top_k: int = Field(default=5, ge=1, le=20, description="Number of top positive/negative SHAP drivers to return.")


class RecommendationRequest(CustomerFeaturePayload):
    """Request body for POST /recommend."""


class CustomerAnalysisRequest(CustomerFeaturePayload):
    """Request body for POST /customer-analysis."""

    top_k: int = Field(default=5, ge=1, le=20, description="Number of top positive/negative SHAP drivers to return.")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., description='"healthy" or "unhealthy".')
    model: Optional[str] = Field(None, description="Registered model name.")
    model_version: Optional[str] = Field(None, description="Registered model version.")
    detail: Optional[str] = Field(None, description="Present only when unhealthy; never a stack trace or secret.")


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    C: float
    class_weight: str


class PredictionResponse(BaseModel):
    customer_id: str
    snapshot_date: date
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Model output -- the frozen Phase 7 Logistic Regression's predicted probability.")
    risk_level: str = Field(..., description="Business risk band (Phase 9 configs/recommendation_config.yaml), not a statistically optimal cutoff.")
    model_name: str
    model_version: str


class DriverSignal(BaseModel):
    feature: str
    shap_value: float
    customer_value: Any = Field(..., description="This customer's own raw value for `feature`; never fabricated.")


class ExplanationResponse(BaseModel):
    customer_id: str
    snapshot_date: date
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    top_positive_drivers: list[DriverSignal] = Field(..., description="Top model_signals pushing predicted risk up.")
    top_negative_drivers: list[DriverSignal] = Field(..., description="Top model_signals pushing predicted risk down.")
    shap_base_value: float
    model_name: str
    model_version: str
    disclaimer: str = Field(
        default=(
            "Each driver is a model_signal: an association this model's SHAP explanation assigns to "
            "this feature for this customer. It is not evidence that the feature causes churn."
        )
    )


class RecommendationResponse(BaseModel):
    customer_id: str
    snapshot_date: date
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    priority: str
    primary_action: str
    action_reason: str
    supporting_features: list[str]
    model_name: str
    model_version: str
    disclaimer: str = Field(
        default=(
            "This recommendation maps observed model/feature signals to a predefined business action. "
            "It is decision support, not a causal treatment-effect estimate -- it does not claim that "
            "taking this action will reduce churn."
        )
    )


class CustomerAnalysisResponse(BaseModel):
    customer: dict[str, Any] = Field(..., description="customer_id and snapshot_date, passed through from the request.")
    prediction: PredictionResponse
    explanation: ExplanationResponse
    recommendation: RecommendationResponse
