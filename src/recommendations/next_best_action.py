"""Phase 9 Next-Best-Action (NBA) layer.

Turns the frozen Phase 7 Logistic Regression's churn probability, plus
Phase 8's SHAP explanations, into a business-rule-based retention action:

    customer features -> churn probability -> SHAP drivers (optional)
        -> risk segment -> recommended action + priority

This module never retrains, retunes, or otherwise touches the Phase 7
model or the Phase 8 explainability implementation -- it only calls
predict_proba on the already-fitted pipeline and (optionally) reuses
src.models.explainability's SHAP building blocks unmodified.

Every threshold used below comes from configs/recommendation_config.yaml,
never hard-coded here, so the business/retention team can revise them
without touching code. All of it are BUSINESS DECISION THRESHOLDS, not
statistically optimized cutoffs -- see the config file's comments.

IMPORTANT: recommendations map observed model/feature signals to
predefined actions. They do not claim, and must never be read as
claiming, that taking a given action will cause a reduction in churn.
Establishing that would require a controlled experiment or a causal
inference framework, not this rule layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from src.models.baseline import PROJECT_ROOT, REPORTS_DIR
from src.recommendations.rules import REQUIRED_SIGNAL_FEATURES, evaluate_signals

CONFIG_PATH = PROJECT_ROOT / "configs" / "recommendation_config.yaml"

RISK_LEVELS: tuple[str, ...] = ("low", "medium", "high", "critical")
PRIORITY_LEVELS: tuple[str, ...] = ("low", "medium", "high", "urgent")

ID_COLUMNS: tuple[str, ...] = ("customer_id", "snapshot_date")

REQUIRED_OUTPUT_COLUMNS: tuple[str, ...] = (
    "customer_id",
    "snapshot_date",
    "churn_probability",
    "risk_level",
    "primary_action",
    "action_reason",
    "supporting_features",
    "priority",
)


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load the Phase 9 business-rule configuration. Raises loudly if it's
    missing -- this module has no hard-coded fallback thresholds.
    """
    if not path.exists():
        raise FileNotFoundError(f"Recommendation config not found: {path}")
    with open(path) as config_file:
        config = yaml.safe_load(config_file)

    required_sections = {"risk_thresholds", "priority_thresholds", "signal_thresholds", "actions"}
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"Recommendation config at {path} is missing required section(s): {missing}")

    return config


def _validate_probability(probability: float) -> None:
    if probability is None or (isinstance(probability, float) and np.isnan(probability)):
        raise ValueError(f"Invalid churn probability: {probability!r}")
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"Churn probability must be within [0, 1], got {probability!r}")


def predict_risk(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Churn probabilities from the frozen Phase 7 pipeline. A thin
    predict_proba wrapper -- never fits or otherwise modifies `pipeline`
    -- kept here so this module, a future FastAPI service, and a future
    Streamlit dashboard all share one code path for scoring.
    """
    return pipeline.predict_proba(X)[:, 1]


def assign_risk_level(probability: float, risk_thresholds: dict) -> str:
    """probability < low -> "low"; low <= probability < medium -> "medium";
    medium <= probability < high -> "high"; probability >= high -> "critical".
    """
    _validate_probability(probability)
    if probability < risk_thresholds["low"]:
        return "low"
    if probability < risk_thresholds["medium"]:
        return "medium"
    if probability < risk_thresholds["high"]:
        return "high"
    return "critical"


def assign_priority_from_probability(probability: float, priority_thresholds: dict) -> str:
    """probability < medium -> "low"; medium <= probability < high -> "medium";
    high <= probability < urgent -> "high"; probability >= urgent -> "urgent".
    """
    _validate_probability(probability)
    if probability < priority_thresholds["medium"]:
        return "low"
    if probability < priority_thresholds["high"]:
        return "medium"
    if probability < priority_thresholds["urgent"]:
        return "high"
    return "urgent"


def bump_priority(priority: str, steps: int = 1) -> str:
    """One level up PRIORITY_LEVELS, capped at "urgent"."""
    index = PRIORITY_LEVELS.index(priority)
    return PRIORITY_LEVELS[min(index + steps, len(PRIORITY_LEVELS) - 1)]


def compute_priority(probability: float, signals: dict[str, tuple[bool, list[str]]], config: dict) -> str:
    """Base priority from probability alone, bumped one level (capped at
    "urgent") when 2+ signal categories are simultaneously triggered --
    priority reflects both how likely a customer is to churn and how many
    distinct problems were actually observed for them.
    """
    priority_thresholds = config["priority_thresholds"]
    base_priority = assign_priority_from_probability(probability, priority_thresholds)

    triggered_count = sum(1 for is_triggered, _ in signals.values() if is_triggered)
    if priority_thresholds.get("multi_signal_bump", True) and triggered_count >= 2:
        return bump_priority(base_priority)
    return base_priority


def determine_action(
    risk_level: str, signals: dict[str, tuple[bool, list[str]]], probability: float, config: dict
) -> tuple[str, str, list[str]]:
    """The core rule table. Returns (action_key, human_reason, supporting_features).

    Precedence (most specific / most severe first):
      1. critical/high risk + 2 or more distinct triggered signal
         categories -> priority_retention_outreach (a genuinely compound
         problem, not just one weak signal riding on a high probability).
      2. payment_problems -> billing_payment_assistance
      3. support_problems -> customer_support_follow_up
      4. reduced_usage     -> re_engagement_campaign
      5. low_engagement    -> engagement_campaign
      6. nothing triggered -> no_intervention (risk is low, or risk is
         non-trivial but no configured business signal explains it).

    This intentionally does not map every high-risk customer to the same
    action: which of 2-5 fires depends entirely on which real feature
    thresholds that specific customer's row actually crossed.
    """
    triggered_categories = [name for name, (is_triggered, _) in signals.items() if is_triggered]

    if risk_level in ("critical", "high") and len(triggered_categories) >= 2:
        reason = (
            f"{risk_level.capitalize()} churn risk (probability={probability:.2f}) is accompanied by "
            f"multiple negative signals: {', '.join(triggered_categories)}."
        )
        supporting = [feature for name in triggered_categories for feature in signals[name][1]]
        return "priority_retention_outreach", reason, supporting

    if signals["payment_problems"][0]:
        return (
            "billing_payment_assistance",
            "Elevated payment/transaction problems detected in the last 90 days.",
            signals["payment_problems"][1],
        )
    if signals["support_problems"][0]:
        return (
            "customer_support_follow_up",
            "Unresolved, high-severity, or low-satisfaction support activity detected in the last 90 days.",
            signals["support_problems"][1],
        )
    if signals["reduced_usage"][0]:
        return (
            "re_engagement_campaign",
            "Recent engagement has declined meaningfully versus the prior 30-day period.",
            signals["reduced_usage"][1],
        )
    if signals["low_engagement"][0]:
        return (
            "engagement_campaign",
            "Overall activity/engagement over the last 90 days is low.",
            signals["low_engagement"][1],
        )

    if risk_level == "low":
        reason = f"Predicted churn risk is low (probability={probability:.2f})."
    else:
        reason = (
            f"Predicted churn risk is {risk_level} (probability={probability:.2f}), but no configured business "
            "signal (engagement, usage trend, support, or payment) met its threshold -- insufficient evidence "
            "for a targeted intervention."
        )
    return "no_intervention", reason, []


def generate_customer_action(row, probability: float, config: dict, model_signals: list[str] | None = None) -> dict:
    """The full per-customer decision: risk level, action, reason, priority,
    and every raw feature/value that triggered it. `row` must expose the
    id columns and the features rules.py reads (a pandas Series or a plain
    dict both work). `model_signals`, if given, is a list of Phase 8 SHAP
    top-positive-driver feature names for this customer -- included in
    supporting_features prefixed with "model_signal:" so it's always
    distinguishable from a rule-triggered feature.
    """
    risk_level = assign_risk_level(probability, config["risk_thresholds"])
    signals = evaluate_signals(row, config)
    action_key, reason, supporting_features = determine_action(risk_level, signals, probability, config)
    priority = compute_priority(probability, signals, config)

    all_supporting_features = list(supporting_features)
    if model_signals:
        all_supporting_features += [f"model_signal:{feature}" for feature in model_signals]

    get = row.get if hasattr(row, "get") else row.__getitem__

    return {
        "customer_id": get("customer_id"),
        "snapshot_date": get("snapshot_date"),
        "churn_probability": probability,
        "risk_level": risk_level,
        "primary_action": config["actions"][action_key],
        "action_key": action_key,
        "action_reason": reason,
        "supporting_features": all_supporting_features,
        "priority": priority,
    }


def generate_recommendation(
    pipeline: Pipeline,
    customer_row: pd.DataFrame,
    feature_columns: list[str],
    config: dict,
    model_signals: list[str] | None = None,
) -> dict:
    """Single-customer convenience wrapper: score with the frozen Phase 7
    pipeline, then generate_customer_action on the result. `customer_row`
    must be a 1-row DataFrame with the same raw feature columns used at
    training time.
    """
    if len(customer_row) != 1:
        raise ValueError(f"generate_recommendation expects exactly one row, got {len(customer_row)}")

    probability = float(predict_risk(pipeline, customer_row[feature_columns])[0])
    return generate_customer_action(customer_row.iloc[0], probability, config, model_signals=model_signals)


def build_shap_context(pipeline: Pipeline, X_train: pd.DataFrame, categorical_columns: list[str]) -> dict:
    """Reuses Phase 8's explainability building blocks completely
    unmodified: a TRAIN-only background and a LinearExplainer on the
    frozen model. Import is local to avoid a hard dependency on shap for
    callers of this module that never request SHAP-based model_signals.
    """
    from src.models import explainability as ex

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_metadata = ex.build_feature_metadata(preprocessor, categorical_columns)
    display_names = feature_metadata["display_name"].tolist()
    background = ex.build_background(preprocessor, X_train)
    explainer = ex.build_explainer(pipeline, background)

    return {
        "preprocessor": preprocessor,
        "feature_metadata": feature_metadata,
        "display_names": display_names,
        "explainer": explainer,
    }


def compute_top_positive_shap_drivers(shap_context: dict, X: pd.DataFrame, top_k: int = 3) -> list[list[str]]:
    """Vectorized top-k positive (risk-raising) SHAP driver names per row
    in X, using the already-built `shap_context`. A negative or zero SHAP
    value is never reported as a "driver" here -- this feeds
    model_signals, which is specifically the model's own top risk-raising
    evidence for that customer.
    """
    from src.models import explainability as ex

    shap_values = ex.compute_shap_values(
        shap_context["explainer"], shap_context["preprocessor"], X, shap_context["display_names"]
    )
    values = shap_values.values
    order = np.argsort(-values, axis=1)[:, :top_k]

    display_names = shap_context["display_names"]
    return [
        [display_names[j] for j in order[i] if values[i, j] > 0]
        for i in range(values.shape[0])
    ]


def generate_batch_recommendations(
    pipeline: Pipeline,
    features: pd.DataFrame,
    feature_columns: list[str],
    config: dict,
    shap_context: dict | None = None,
    top_k_shap: int = 3,
) -> pd.DataFrame:
    """Score every row in `features` and generate its recommendation.
    `features` must contain the id columns, every column in feature_columns
    (for scoring), and whichever of rules.REQUIRED_SIGNAL_FEATURES are
    available (a missing one simply never triggers its condition).
    """
    probabilities = predict_risk(pipeline, features[feature_columns])

    model_signals_per_row: list[list[str] | None] = [None] * len(features)
    if shap_context is not None:
        model_signals_per_row = compute_top_positive_shap_drivers(shap_context, features[feature_columns], top_k=top_k_shap)

    signal_columns = [c for c in REQUIRED_SIGNAL_FEATURES if c in features.columns]
    records = features[list(ID_COLUMNS) + signal_columns].to_dict("records")

    rows = [
        generate_customer_action(record, float(probabilities[i]), config, model_signals=model_signals_per_row[i])
        for i, record in enumerate(records)
    ]
    return pd.DataFrame(rows)


def save_recommendations(recommendations: pd.DataFrame, filename: str = "customer_recommendations.csv") -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / filename

    to_save = recommendations.copy()
    to_save["supporting_features"] = to_save["supporting_features"].apply(json.dumps)
    to_save.to_csv(output_path, index=False)
    return output_path


def build_recommendation_summary(recommendations: pd.DataFrame) -> pd.DataFrame:
    """risk_level x primary_action -> customer_count, percentage_of_customers."""
    total = len(recommendations)

    summary = (
        recommendations.groupby(["risk_level", "primary_action"], observed=True)
        .size()
        .reset_index(name="customer_count")
    )
    summary["percentage_of_customers"] = (summary["customer_count"] / total * 100).round(3)

    summary["risk_level"] = pd.Categorical(summary["risk_level"], categories=RISK_LEVELS, ordered=True)
    summary = summary.sort_values(["risk_level", "customer_count"], ascending=[True, False]).reset_index(drop=True)
    summary["risk_level"] = summary["risk_level"].astype(str)

    return summary


def save_recommendation_summary(summary: pd.DataFrame, filename: str = "recommendation_summary.csv") -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / filename
    summary.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    from src.models import baseline as bl
    from src.models.explainability import load_tuned_pipeline

    config = load_config()
    pipeline = load_tuned_pipeline()

    features = bl.load_feature_dataset()
    numeric_columns, categorical_columns = bl.get_feature_columns(features)
    feature_columns = numeric_columns + categorical_columns

    splits = bl.split_by_snapshot_date(features)
    shap_context = build_shap_context(pipeline, splits["train"][feature_columns], categorical_columns)

    recommendations = generate_batch_recommendations(
        pipeline, features, feature_columns, config, shap_context=shap_context
    )

    recommendations_path = save_recommendations(recommendations)
    print(f"Saved {len(recommendations)} recommendations -> {recommendations_path}")

    summary = build_recommendation_summary(recommendations)
    summary_path = save_recommendation_summary(summary)
    print(f"Saved recommendation summary -> {summary_path}")
    print(summary.to_string(index=False))
