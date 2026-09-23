"""Phase 8 model explainability: SHAP explanations for the Phase 7 tuned
Logistic Regression model (C=0.01, class_weight="balanced").

This module never retrains, retunes, or otherwise modifies the Phase 7
model -- it only loads the already-fitted pipeline from
models/tuned_logistic_regression.joblib and explains its predictions.

All explanations are computed on the VALIDATION split, matching the
validation-only philosophy already established in Phase 7 -- the test
split is not read anywhere in this module, since explaining an
already-frozen model has no need to touch it.

SHAP method: shap.LinearExplainer, the exact (non-sampling) explainer for
linear models. For a Logistic Regression's raw decision function (a linear
combination of the transformed features), SHAP values are additive and
exact: base_value + sum(shap_values) == the model's linear output for that
row, which reconstructs predict_proba via a sigmoid. A background dataset
(a fixed-seed sample of TRAIN, transformed the same way as at prediction
time) supplies the reference point that SHAP values are computed relative
to -- background statistics are taken from TRAIN only, never
validation/test, so this stays leakage-safe the same way Phase 5's
preprocessing is.

Feature-name mapping: the pipeline's ColumnTransformer prefixes every
output column with its transformer's name ("numeric__..." /
"categorical__..."). Feature metadata below strips that prefix for display
and, for one-hot categorical columns, recovers the original column and
category (e.g. "categorical__plan_Premium" -> origin_column="plan",
category="Premium") so a customer's actual raw value can be reported next
to each SHAP value instead of an opaque transformed column name.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.models.baseline import (
    CLASSIFICATION_THRESHOLD,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET_COLUMN,
    get_feature_columns,
    load_feature_dataset,
    split_by_snapshot_date,
)

TUNED_MODEL_PATH = MODELS_DIR / "tuned_logistic_regression.joblib"
FIGURES_DIR = REPORTS_DIR / "figures"

# The exact Phase 7 selection -- explainability.py explains this model and
# no other; a mismatch here means the wrong artifact was loaded.
EXPECTED_C = 0.01
EXPECTED_CLASS_WEIGHT = "balanced"

SHAP_BACKGROUND_SIZE = 100
TOP_N_GLOBAL_FEATURES = 20
TOP_K_LOCAL_DRIVERS = 5


def load_tuned_pipeline(path: Path = TUNED_MODEL_PATH) -> Pipeline:
    """Load the frozen Phase 7 pipeline. Raises loudly if it's missing or
    if it's not actually the expected tuned Logistic Regression -- this
    module must never silently explain the wrong model.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Tuned model not found: {path}. Run src/models/hyperparameter_tuning.py "
            "(Phase 7) first -- this module explains that model, it does not train one."
        )

    pipeline = joblib.load(path)
    model = pipeline.named_steps["model"]

    if not hasattr(model, "coef_") and not hasattr(model, "C"):
        raise ValueError(f"Loaded model at {path} does not look like a fitted LogisticRegression: {model!r}")
    if model.C != EXPECTED_C or model.class_weight != EXPECTED_CLASS_WEIGHT:
        raise ValueError(
            f"Loaded model hyperparameters (C={model.C}, class_weight={model.class_weight}) do not "
            f"match the expected Phase 7 selection (C={EXPECTED_C}, class_weight={EXPECTED_CLASS_WEIGHT}). "
            "Refusing to explain a different model without an explicit decision to do so."
        )

    return pipeline


def build_feature_metadata(preprocessor, categorical_columns: list[str]) -> pd.DataFrame:
    """One row per transformed feature: transformed_name (as ColumnTransformer
    produced it), display_name (prefix stripped), kind, origin_column (the
    raw feature it came from), and category (the one-hot level, for
    categorical features only).
    """
    rows = []
    for raw_name in preprocessor.get_feature_names_out():
        prefix, display_name = raw_name.split("__", 1)

        if prefix == "numeric":
            rows.append({
                "transformed_name": raw_name,
                "display_name": display_name,
                "kind": "numeric",
                "origin_column": display_name,
                "category": None,
            })
        else:
            matched_column = next(
                (col for col in sorted(categorical_columns, key=len, reverse=True)
                 if display_name == col or display_name.startswith(col + "_")),
                None,
            )
            category = display_name[len(matched_column) + 1:] if matched_column else None
            rows.append({
                "transformed_name": raw_name,
                "display_name": display_name,
                "kind": "categorical",
                "origin_column": matched_column or display_name,
                "category": category,
            })

    return pd.DataFrame(rows)


def build_background(preprocessor, X_train: pd.DataFrame, sample_size: int = SHAP_BACKGROUND_SIZE) -> np.ndarray:
    """A fixed-seed sample of TRAIN, transformed with the already-fitted
    preprocessor. This is SHAP's reference point, not a refit -- the
    preprocessor itself was fit on TRAIN back in Phase 7 and is not
    touched here.
    """
    sample = X_train.sample(n=min(sample_size, len(X_train)), random_state=RANDOM_SEED)
    return preprocessor.transform(sample)


def build_explainer(pipeline: Pipeline, background: np.ndarray) -> shap.LinearExplainer:
    """shap.LinearExplainer on the pipeline's LogisticRegression, with the
    background sized to exactly match what the masker will use so there is
    no additional internal subsampling step (which would otherwise be a
    second, separate source of randomness beyond our own fixed-seed sample).
    """
    model = pipeline.named_steps["model"]
    return shap.LinearExplainer(model, background)


def compute_shap_values(
    explainer: shap.LinearExplainer, preprocessor, X: pd.DataFrame, display_names: list[str]
) -> shap.Explanation:
    transformed = preprocessor.transform(X)
    explanation = explainer(transformed)
    explanation.feature_names = display_names
    return explanation


def global_feature_importance(shap_values: shap.Explanation) -> pd.DataFrame:
    """Mean absolute / mean signed SHAP value per feature, ranked by mean
    absolute SHAP value (overall importance, direction-agnostic).
    """
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    mean_shap = shap_values.values.mean(axis=0)

    importance = pd.DataFrame({
        "feature": shap_values.feature_names,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap": mean_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance["rank"] = importance.index + 1

    return importance


def save_feature_importance(importance: pd.DataFrame, filename: str = "shap_feature_importance.csv") -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / filename
    importance.to_csv(output_path, index=False)
    return output_path


def plot_global_bar(shap_values: shap.Explanation, top_n: int = TOP_N_GLOBAL_FEATURES, save_path: Path | None = None) -> Path | None:
    shap.plots.bar(shap_values, max_display=top_n, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    if save_path is not None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_global_beeswarm(shap_values: shap.Explanation, top_n: int = TOP_N_GLOBAL_FEATURES, save_path: Path | None = None) -> Path | None:
    """The SHAP summary (beeswarm) plot -- shows importance AND direction:
    each dot is one customer, colored by that customer's raw feature value,
    positioned by that feature's SHAP value for them.
    """
    shap.plots.beeswarm(shap_values, max_display=top_n, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    if save_path is not None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_local_waterfall(shap_values_row: shap.Explanation, save_path: Path | None = None, max_display: int = 15) -> Path | None:
    shap.plots.waterfall(shap_values_row, max_display=max_display, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    if save_path is not None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def compute_risk_tier_bounds(probabilities: np.ndarray) -> tuple[float, float]:
    """Data-driven tercile cutoffs (33rd/66th percentile) used ONLY to label
    illustrative example customers as High/Medium/Low risk for this report.
    This is a communication convenience, distinct from CLASSIFICATION_THRESHOLD
    (0.5), which remains the model's actual decision boundary from Phases 5-7.
    """
    low_cutoff, high_cutoff = np.quantile(probabilities, [1 / 3, 2 / 3])
    return float(low_cutoff), float(high_cutoff)


def assign_risk_tier(probability: float, low_cutoff: float, high_cutoff: float) -> str:
    if probability >= high_cutoff:
        return "High"
    if probability <= low_cutoff:
        return "Low"
    return "Medium"


def select_representative_customers(pipeline: Pipeline, X: pd.DataFrame) -> dict[str, int]:
    """Positional indices (into X) of the highest-, median-, and
    lowest-predicted-probability customers -- deterministic given a fixed
    model and data, since predict_proba has no randomness for a fitted
    Logistic Regression.
    """
    probabilities = pipeline.predict_proba(X)[:, 1]
    high_idx = int(np.argmax(probabilities))
    low_idx = int(np.argmin(probabilities))
    median_idx = int(np.argsort(probabilities)[len(probabilities) // 2])
    return {"high": high_idx, "medium": median_idx, "low": low_idx}


def _to_native_scalar(value):
    """A raw feature value straight from a pandas row can be a numpy
    scalar (np.bool_, np.integer, np.floating) rather than a native Python
    type -- harmless for in-process use, but Pydantic/JSON serialization
    (e.g. the Phase 11 API's DriverSignal.customer_value: Any) does not
    know how to serialize a bare numpy scalar. Converting here, once, at
    the one place a raw value is attached to an explanation, keeps every
    downstream consumer (API, dashboard, notebooks) safe without each of
    them having to guard against it separately.
    """
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def explain_customer(
    pipeline: Pipeline,
    explainer: shap.LinearExplainer,
    customer_row: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    risk_tier_bounds: tuple[float, float] | None = None,
    threshold: float = CLASSIFICATION_THRESHOLD,
    top_k: int = TOP_K_LOCAL_DRIVERS,
) -> dict:
    """Explain a single customer row (a 1-row DataFrame with the same raw
    feature columns used at training time). Returns predicted probability,
    predicted class, an optional risk tier, the top positive (risk-raising)
    and negative (risk-lowering) SHAP drivers with the customer's own raw
    values attached, the full per-feature SHAP value dict, and the SHAP
    base value. Every number here comes directly from the fitted pipeline
    and its SHAP explanation -- nothing is inferred or invented.
    """
    if len(customer_row) != 1:
        raise ValueError(f"explain_customer expects exactly one row, got {len(customer_row)}")

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    transformed = preprocessor.transform(customer_row)
    probability = float(model.predict_proba(transformed)[0, 1])
    predicted_class = int(probability >= threshold)

    explanation = explainer(transformed)
    values = explanation.values[0]
    base_value = float(np.ravel(explanation.base_values)[0])

    raw_row = customer_row.iloc[0]
    drivers = []
    for i, shap_value in enumerate(values):
        meta = feature_metadata.iloc[i]
        if meta["kind"] == "numeric":
            customer_value = _to_native_scalar(raw_row[meta["origin_column"]])
        else:
            customer_value = bool(raw_row[meta["origin_column"]] == meta["category"])
        drivers.append({
            "feature": meta["display_name"],
            "shap_value": float(shap_value),
            "customer_value": customer_value,
        })

    positive_drivers = sorted((d for d in drivers if d["shap_value"] > 0), key=lambda d: d["shap_value"], reverse=True)
    negative_drivers = sorted((d for d in drivers if d["shap_value"] < 0), key=lambda d: d["shap_value"])

    result = {
        "predicted_probability": probability,
        "predicted_class": predicted_class,
        "base_value": base_value,
        "top_positive_drivers": positive_drivers[:top_k],
        "top_negative_drivers": negative_drivers[:top_k],
        "shap_values": {d["feature"]: d["shap_value"] for d in drivers},
    }

    if risk_tier_bounds is not None:
        low_cutoff, high_cutoff = risk_tier_bounds
        result["risk_tier"] = assign_risk_tier(probability, low_cutoff, high_cutoff)

    return result


def run_explainability(pipeline: Pipeline | None = None, features: pd.DataFrame | None = None) -> dict:
    """End-to-end Phase 8 run: load the frozen Phase 7 model, build the SHAP
    explainer from a TRAIN background, compute global SHAP values on the
    full VALIDATION split, and prepare high/medium/low representative
    customer explanations. Does not save anything -- callers decide when
    to persist results/figures.
    """
    if pipeline is None:
        pipeline = load_tuned_pipeline()
    if features is None:
        features = load_feature_dataset()

    splits = split_by_snapshot_date(features)
    numeric_columns, categorical_columns = get_feature_columns(features)
    feature_columns = numeric_columns + categorical_columns

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_metadata = build_feature_metadata(preprocessor, categorical_columns)
    display_names = feature_metadata["display_name"].tolist()

    X_train = splits["train"][feature_columns]
    X_val = splits["validation"][feature_columns]

    background = build_background(preprocessor, X_train)
    explainer = build_explainer(pipeline, background)

    shap_values = compute_shap_values(explainer, preprocessor, X_val, display_names)
    importance = global_feature_importance(shap_values)

    validation_probabilities = pipeline.predict_proba(X_val)[:, 1]
    risk_tier_bounds = compute_risk_tier_bounds(validation_probabilities)
    representative_positions = select_representative_customers(pipeline, X_val)

    representative_explanations = {}
    for tier, position in representative_positions.items():
        customer_row = X_val.iloc[[position]]
        representative_explanations[tier] = {
            "customer_id": splits["validation"].iloc[position]["customer_id"],
            "snapshot_date": splits["validation"].iloc[position]["snapshot_date"],
            "actual_label": int(splits["validation"].iloc[position][TARGET_COLUMN]),
            **explain_customer(pipeline, explainer, customer_row, feature_metadata, risk_tier_bounds=risk_tier_bounds),
        }

    return {
        "pipeline": pipeline,
        "splits": splits,
        "feature_columns": feature_columns,
        "feature_metadata": feature_metadata,
        "display_names": display_names,
        "explainer": explainer,
        "shap_values": shap_values,
        "importance": importance,
        "risk_tier_bounds": risk_tier_bounds,
        "representative_positions": representative_positions,
        "representative_explanations": representative_explanations,
    }


if __name__ == "__main__":
    result = run_explainability()

    importance_path = save_feature_importance(result["importance"])
    print(f"Saved global feature importance -> {importance_path}")
    print(result["importance"].head(20).to_string(index=False))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    bar_path = plot_global_bar(result["shap_values"], save_path=FIGURES_DIR / "shap_global_bar.png")
    print(f"Saved global bar plot -> {bar_path}")
    beeswarm_path = plot_global_beeswarm(result["shap_values"], save_path=FIGURES_DIR / "shap_global_beeswarm.png")
    print(f"Saved global beeswarm plot -> {beeswarm_path}")

    for tier, explanation in result["representative_explanations"].items():
        print(f"\n{tier.upper()}-risk customer {explanation['customer_id']}: "
              f"probability={explanation['predicted_probability']:.4f}, "
              f"risk_tier={explanation['risk_tier']}, actual_label={explanation['actual_label']}")
        print("  Top risk-raising drivers:", explanation["top_positive_drivers"][:3])
        print("  Top risk-lowering drivers:", explanation["top_negative_drivers"][:3])
