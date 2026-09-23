"""Phase 11 service layer -- thin orchestration over Phases 7-10.

No business logic is duplicated here:

- Churn scoring uses the registered MLflow pipeline (Phase 10) directly.
- SHAP explanations reuse src.models.explainability's pure functions
  (Phase 8) completely unmodified.
- Recommendations reuse src.recommendations.next_best_action (Phase 9)
  completely unmodified.

This module only wires those together for the API and caches the
expensive-to-build pieces (the model, its SHAP background/explainer, the
business config) as a single shared ModelService instance, constructed
once at API startup (see api/main.py's lifespan) -- never per request.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models import baseline as bl
from src.models import explainability as ex
from src.models import mlflow_tracking as mt
from src.monitoring.drift import DriftMonitor, select_top_numeric_features_by_shap
from src.recommendations import next_best_action as nba

logger = logging.getLogger("api.services")

SHAP_IMPORTANCE_REPORT_PATH = bl.REPORTS_DIR / "shap_feature_importance.csv"


class ModelUnavailableError(RuntimeError):
    """Raised when the configured registered model (or anything it needs:
    SHAP context, recommendation config) cannot be loaded. Never caught
    silently -- callers turn this into a clear HTTP 503.
    """


def feature_row_to_payload(row: pd.Series) -> dict[str, Any]:
    """Convert one real row from customer_features.parquet into a
    JSON-serializable dict suitable as an API request body: NaN -> None,
    numpy/pandas scalars -> native Python types, timestamps -> ISO date
    strings. Used by tests and the Phase 11 integration notebook so
    example requests are built from real, not invented, feature values.
    """
    payload: dict[str, Any] = {}
    for key, value in row.items():
        if key == bl.TARGET_COLUMN:
            continue
        if isinstance(value, pd.Timestamp):
            payload[key] = value.date().isoformat()
        elif value is None:
            payload[key] = None
        elif isinstance(value, float) and np.isnan(value):
            payload[key] = None
        elif pd.isna(value):
            payload[key] = None
        elif isinstance(value, np.bool_):
            payload[key] = bool(value)
        elif isinstance(value, np.integer):
            payload[key] = int(value)
        elif isinstance(value, np.floating):
            payload[key] = float(value)
        else:
            payload[key] = value
    return payload


class ModelService:
    """The one shared, cached instance of everything expensive to build:
    the registered Phase 10 model, its expected feature columns, the
    Phase 8 SHAP context, and the Phase 9 recommendation config.
    """

    def __init__(self) -> None:
        self.pipeline = None
        self.model_version: str | None = None
        self.feature_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.shap_context: dict | None = None
        self.recommendation_config: dict | None = None
        self.mlflow_config: dict | None = None
        self.drift_monitor: DriftMonitor | None = None

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None

    @property
    def model_name(self) -> str:
        if self.mlflow_config is None:
            return "unknown"
        return self.mlflow_config["registry"]["model_name"]

    def load(self) -> None:
        """Load the registered MLflow model (Phase 10) and build the SHAP
        context (Phase 8) / recommendation config (Phase 9) it needs.
        Never fits, retrains, or tunes anything -- every artifact loaded
        here is already frozen from an earlier phase. Raises
        ModelUnavailableError (never a bare, unlabeled exception) on any
        failure, so a failed startup is unambiguous.
        """
        try:
            self.mlflow_config = mt.load_config()
            self.recommendation_config = nba.load_config()

            features = bl.load_feature_dataset()
            splits = bl.split_by_snapshot_date(features)
            self.numeric_columns, self.categorical_columns = bl.get_feature_columns(features)
            self.feature_columns = self.numeric_columns + self.categorical_columns

            # load_registered_model (Phase 10) validates model type, C,
            # and class_weight, raising ValueError on any mismatch --
            # caught below and re-raised as ModelUnavailableError.
            self.pipeline = mt.load_registered_model(config=self.mlflow_config)

            # Additionally validate "expected feature structure": the raw
            # column names/order the registered model's preprocessor was
            # actually fit on must match what the current feature dataset
            # produces. This catches dataset drift (a renamed/added/removed
            # feature column since the model was registered) that a bare
            # transformed-coefficient-count check would miss.
            preprocessor = self.pipeline.named_steps["preprocessor"]
            expected_numeric = list(preprocessor.transformers_[0][2])
            expected_categorical = list(preprocessor.transformers_[1][2])
            if expected_numeric != self.numeric_columns or expected_categorical != self.categorical_columns:
                raise ModelUnavailableError(
                    "Registered model's expected feature columns do not match the current "
                    "feature dataset's columns -- refusing to serve a schema-mismatched model."
                )

            client = mt.MlflowClient()
            model_name = self.mlflow_config["registry"]["model_name"]
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise ModelUnavailableError(f"No registered versions found for model '{model_name}'")
            self.model_version = str(max(versions, key=lambda v: int(v.version)).version)

            self.shap_context = nba.build_shap_context(
                self.pipeline, splits["train"][self.feature_columns], self.categorical_columns
            )

            # Phase 14 drift baseline. Never a second copy of the model or
            # its preprocessing: reuses the already-loaded self.pipeline
            # (via nba.predict_risk, the same scoring path predict_probability
            # uses) purely to score the VALIDATION split once at startup --
            # an out-of-sample reference for the prediction-drift baseline,
            # never used to select or retrain anything.
            try:
                shap_importance = pd.read_csv(SHAP_IMPORTANCE_REPORT_PATH)
                top_numeric_features = select_top_numeric_features_by_shap(shap_importance, self.numeric_columns)
                self.drift_monitor = DriftMonitor(splits["train"], top_numeric_features, self.categorical_columns)
                validation_probabilities = nba.predict_risk(self.pipeline, splits["validation"][self.feature_columns])
                self.drift_monitor.set_prediction_baseline(validation_probabilities)
            except FileNotFoundError:
                logger.error(
                    "Drift monitoring baseline unavailable: %s not found. "
                    "Run Phase 8 (src/models/explainability.py) to generate it.",
                    SHAP_IMPORTANCE_REPORT_PATH,
                )
                self.drift_monitor = None

            logger.info(
                "Model service ready: model=%s version=%s features=%d drift_monitor=%s",
                model_name, self.model_version, len(self.feature_columns), self.drift_monitor is not None,
            )
        except ModelUnavailableError:
            self.pipeline = None
            raise
        except Exception as exc:
            self.pipeline = None
            logger.error("Failed to load model service: %s", exc)
            raise ModelUnavailableError(str(exc)) from exc

    def model_metadata(self) -> dict[str, Any]:
        model = self.pipeline.named_steps["model"]
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_type": type(model).__name__,
            "C": model.C,
            "class_weight": model.class_weight,
        }

    def _payload_to_row(self, payload: dict[str, Any]) -> pd.DataFrame:
        """One customer's raw feature row as a 1-row DataFrame, in exactly
        the column set the pipeline expects. A None value is left as NaN
        so the pipeline's own imputers (fit in Phase 7) handle it, exactly
        as every other phase already does -- this function never invents
        or defaults a feature value itself.
        """
        missing = [column for column in self.feature_columns if column not in payload]
        if missing:
            raise ValueError(f"Missing required feature(s): {missing}")

        row = {column: payload[column] for column in self.feature_columns}
        return pd.DataFrame([row])

    def predict_probability(self, payload: dict[str, Any]) -> float:
        X = self._payload_to_row(payload)
        return float(nba.predict_risk(self.pipeline, X)[0])

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Phase 9's assign_risk_level applied to a freshly predicted
        probability -- the same risk-band logic every other phase uses,
        never re-implemented here.
        """
        probability = self.predict_probability(payload)
        risk_level = nba.assign_risk_level(probability, self.recommendation_config["risk_thresholds"])
        return {"churn_probability": probability, "risk_level": risk_level}

    def explain(self, payload: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
        """Phase 8's explain_customer, unmodified, on this customer's row."""
        X = self._payload_to_row(payload)
        feature_metadata = self.shap_context["feature_metadata"]
        return ex.explain_customer(
            self.pipeline, self.shap_context["explainer"], X, feature_metadata, top_k=top_k
        )

    def recommend(
        self, payload: dict[str, Any], customer_id: str, snapshot_date: Any, probability: float | None = None
    ) -> dict[str, Any]:
        """Phase 9's generate_customer_action, unmodified. `probability`
        may be passed in (e.g. from a combined analysis that already
        computed it) to avoid a redundant predict_proba call -- recomputing
        it here would give the identical value regardless, since scoring
        is a deterministic function of the model and the row.
        """
        X = self._payload_to_row(payload)
        if probability is None:
            probability = float(nba.predict_risk(self.pipeline, X)[0])

        row_for_rules = dict(payload)
        row_for_rules["customer_id"] = customer_id
        row_for_rules["snapshot_date"] = snapshot_date

        model_signals = None
        if self.shap_context is not None:
            model_signals = nba.compute_top_positive_shap_drivers(self.shap_context, X, top_k=3)[0]

        return nba.generate_customer_action(
            row_for_rules, probability, self.recommendation_config, model_signals=model_signals
        )
