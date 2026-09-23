"""Phase 6 model comparison: the Phase 5 baselines (Logistic Regression,
Random Forest) vs. XGBoost, LightGBM, and CatBoost -- using the exact same
chronological split, feature set, target, and leakage-safe preprocessing
established in src/models/baseline.py.

All hyperparameters below are simple, untuned baseline/default
configurations. Hyperparameter tuning is Phase 7 scope, not this phase.

Imbalance strategy per model (documented explicitly; positive rate ~1-2%):
  - Logistic Regression : class_weight="balanced"       (Phase 5, reused as-is)
  - Random Forest        : class_weight="balanced"       (Phase 5, reused as-is)
  - XGBoost               : scale_pos_weight = negative_count / positive_count,
                            computed from the TRAINING split only. This is
                            XGBoost's own documented mechanism for weighting
                            a rare positive class -- there is no "balanced"
                            string option in its API, so the equivalent
                            ratio is computed explicitly instead.
  - LightGBM              : class_weight="balanced"       (native LGBMClassifier option)
  - CatBoost              : auto_class_weights="Balanced"  (native CatBoostClassifier option)

Every model is the same idea -- inversely weight the loss by class
frequency -- expressed through whichever option each library provides.
"""

from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.models.baseline import (
    CLASSIFICATION_THRESHOLD,
    RANDOM_SEED,
    TARGET_COLUMN,
    build_logistic_regression_pipeline,
    build_preprocessor,
    build_random_forest_pipeline,
    evaluate_model,
    get_feature_columns,
    load_feature_dataset,
    split_by_snapshot_date,
    summarize_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

# Shared, simple gradient-boosting settings across all three boosting
# libraries, so differences in results reflect the algorithms themselves
# rather than mismatched capacity. None of this is tuned.
GRADIENT_BOOSTING_N_ESTIMATORS = 200
GRADIENT_BOOSTING_MAX_DEPTH = 4
GRADIENT_BOOSTING_LEARNING_RATE = 0.1


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """negative_count / positive_count from the TRAINING labels only -- the
    ratio XGBoost's scale_pos_weight expects. Never computed from
    validation/test, which would leak their class balance into training.
    """
    positive_count = int(y_train.sum())
    negative_count = len(y_train) - positive_count
    return negative_count / positive_count if positive_count else 1.0


def build_xgboost_pipeline(preprocessor, scale_pos_weight: float) -> Pipeline:
    """XGBoost baseline. See module docstring for the imbalance strategy."""
    model = XGBClassifier(
        n_estimators=GRADIENT_BOOSTING_N_ESTIMATORS,
        max_depth=GRADIENT_BOOSTING_MAX_DEPTH,
        learning_rate=GRADIENT_BOOSTING_LEARNING_RATE,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_lightgbm_pipeline(preprocessor) -> Pipeline:
    """LightGBM baseline. See module docstring for the imbalance strategy."""
    model = LGBMClassifier(
        n_estimators=GRADIENT_BOOSTING_N_ESTIMATORS,
        max_depth=GRADIENT_BOOSTING_MAX_DEPTH,
        learning_rate=GRADIENT_BOOSTING_LEARNING_RATE,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_catboost_pipeline(preprocessor) -> Pipeline:
    """CatBoost baseline. See module docstring for the imbalance strategy."""
    model = CatBoostClassifier(
        iterations=GRADIENT_BOOSTING_N_ESTIMATORS,
        depth=GRADIENT_BOOSTING_MAX_DEPTH,
        learning_rate=GRADIENT_BOOSTING_LEARNING_RATE,
        auto_class_weights="Balanced",
        random_state=RANDOM_SEED,
        verbose=False,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_all_pipelines(
    numeric_columns: list[str], categorical_columns: list[str], y_train: pd.Series
) -> dict[str, Pipeline]:
    """Construct all five candidate pipelines. Each gets its own fresh
    ColumnTransformer instance built from the identical recipe in
    src/models/baseline.build_preprocessor, so every model sees exactly the
    same preprocessing rules while remaining fully independent objects.
    """
    scale_pos_weight = compute_scale_pos_weight(y_train)

    return {
        "logistic_regression": build_logistic_regression_pipeline(
            build_preprocessor(numeric_columns, categorical_columns)
        ),
        "random_forest": build_random_forest_pipeline(
            build_preprocessor(numeric_columns, categorical_columns)
        ),
        "xgboost": build_xgboost_pipeline(
            build_preprocessor(numeric_columns, categorical_columns), scale_pos_weight
        ),
        "lightgbm": build_lightgbm_pipeline(
            build_preprocessor(numeric_columns, categorical_columns)
        ),
        "catboost": build_catboost_pipeline(
            build_preprocessor(numeric_columns, categorical_columns)
        ),
    }


def run_comparison_experiment(features: pd.DataFrame | None = None) -> dict:
    """End-to-end comparison: the same chronological split, feature columns,
    target, and 0.5 threshold as Phase 5, extended to all five models.
    """
    if features is None:
        features = load_feature_dataset()

    splits = split_by_snapshot_date(features)
    numeric_columns, categorical_columns = get_feature_columns(features)
    feature_columns = numeric_columns + categorical_columns

    X_train = splits["train"][feature_columns]
    y_train = splits["train"][TARGET_COLUMN]

    pipelines = build_all_pipelines(numeric_columns, categorical_columns, y_train)

    fitted_pipelines = {}
    results_rows = []

    for model_name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)  # preprocessing + model fit on TRAIN only
        fitted_pipelines[model_name] = pipeline

        for split_name in ("validation", "test"):
            split_df = splits[split_name]
            X_split = split_df[feature_columns]
            y_split = split_df[TARGET_COLUMN]

            metrics = evaluate_model(pipeline, X_split, y_split, threshold=CLASSIFICATION_THRESHOLD)
            results_rows.append({"model": model_name, "split": split_name, **metrics})

    return {
        "splits": splits,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "feature_columns": feature_columns,
        "pipelines": fitted_pipelines,
        "results": pd.DataFrame(results_rows),
    }


def save_comparison_results(
    results: pd.DataFrame, filename: str = "model_comparison_results.csv"
) -> Path:
    """Save the comparison results table to reports/, with the required
    columns (model, split, pr_auc, roc_auc, precision, recall, f1) first.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / filename

    required_columns = ["model", "split", "pr_auc", "roc_auc", "precision", "recall", "f1"]
    remaining_columns = [c for c in results.columns if c not in required_columns]
    ordered_results = results[required_columns + remaining_columns]

    ordered_results.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    experiment = run_comparison_experiment()

    for split_name, split_df in experiment["splits"].items():
        print(summarize_split(split_name, split_df))

    print(experiment["results"].to_string(index=False))

    results_path = save_comparison_results(experiment["results"])
    print(f"Saved results -> {results_path}")
