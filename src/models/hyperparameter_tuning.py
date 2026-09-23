"""Phase 7 hyperparameter tuning for the five Phase 6 candidate models
(Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost).

Reuses the exact chronological split, feature columns, target, and
leakage-safe preprocessing established in src/models/baseline.py --
nothing about Phases 1-6 is changed here.

Critical rule: the TEST split is never touched by this module's tuning or
model-selection logic. Every trial is trained on the TRAIN split only and
scored on the VALIDATION split only, using PR-AUC (average precision) as
the selection metric -- appropriate for the ~1-2% positive rate, where
ROC-AUC and accuracy are both far too optimistic. The test split is
returned as part of the experiment splits purely so a caller (the
notebook) can run a single, final, non-selecting evaluation of whichever
model this module already chose.

Search spaces are small, hand-curated lists of hyperparameter
combinations (not a full Cartesian grid) so the whole sweep finishes in a
few minutes on a laptop.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.models.baseline import (
    CLASSIFICATION_THRESHOLD,
    RANDOM_SEED,
    TARGET_COLUMN,
    build_logistic_regression_pipeline,
    build_preprocessor,
    evaluate_model,
    get_feature_columns,
    load_feature_dataset,
    split_by_snapshot_date,
)
from src.models.model_comparison import compute_scale_pos_weight

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

BASELINE_MODEL_LABEL = "logistic_regression_baseline_phase5"

# ---------------------------------------------------------------------------
# Small, hand-curated search spaces (not a full factorial grid) -- kept
# short deliberately so the whole sweep is computationally reasonable on a
# laptop. Every combination fixes the same imbalance handling and
# random_state used in Phases 5/6, and only varies the parameters listed
# in the Phase 7 spec.
# ---------------------------------------------------------------------------

LOGISTIC_REGRESSION_GRID: list[dict] = [
    {"C": 0.01},
    {"C": 0.1},
    {"C": 1.0},
    {"C": 10.0},
]

RANDOM_FOREST_GRID: list[dict] = [
    {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 5, "max_features": "sqrt"},
    {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 5, "max_features": "log2"},
    {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 1, "max_features": "log2"},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5, "max_features": "sqrt"},
]

XGBOOST_GRID: list[dict] = [
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 0.8},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 1.0},
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03, "subsample": 1.0, "colsample_bytree": 1.0},
]

LIGHTGBM_GRID: list[dict] = [
    {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.1, "max_depth": -1},
    {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05, "max_depth": 6},
    {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.1, "max_depth": -1},
    {"n_estimators": 300, "num_leaves": 63, "learning_rate": 0.05, "max_depth": 8},
    {"n_estimators": 400, "num_leaves": 15, "learning_rate": 0.03, "max_depth": 4},
]

CATBOOST_GRID: list[dict] = [
    {"iterations": 200, "depth": 4, "learning_rate": 0.1, "l2_leaf_reg": 3.0},
    {"iterations": 200, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5.0},
    {"iterations": 300, "depth": 4, "learning_rate": 0.1, "l2_leaf_reg": 5.0},
    {"iterations": 300, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 7.0},
    {"iterations": 400, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
]

DEFAULT_PARAM_GRIDS: dict[str, list[dict]] = {
    "logistic_regression": LOGISTIC_REGRESSION_GRID,
    "random_forest": RANDOM_FOREST_GRID,
    "xgboost": XGBOOST_GRID,
    "lightgbm": LIGHTGBM_GRID,
    "catboost": CATBOOST_GRID,
}


# ---------------------------------------------------------------------------
# Pipeline builders -- one trial's worth of hyperparameters in, a fresh
# (unfitted) pipeline out. Imbalance handling mirrors Phase 6 exactly for
# each library; only the tunable parameters below change per trial.
# ---------------------------------------------------------------------------

def build_logistic_regression_trial(preprocessor, params: dict) -> Pipeline:
    model = LogisticRegression(
        C=params["C"], class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_random_forest_trial(preprocessor, params: dict) -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_xgboost_trial(preprocessor, params: dict, scale_pos_weight: float) -> Pipeline:
    model = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_lightgbm_trial(preprocessor, params: dict) -> Pipeline:
    model = LGBMClassifier(
        n_estimators=params["n_estimators"],
        num_leaves=params["num_leaves"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_catboost_trial(preprocessor, params: dict) -> Pipeline:
    model = CatBoostClassifier(
        iterations=params["iterations"],
        depth=params["depth"],
        learning_rate=params["learning_rate"],
        l2_leaf_reg=params["l2_leaf_reg"],
        auto_class_weights="Balanced",
        random_state=RANDOM_SEED,
        verbose=False,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _build_trial_pipeline(
    model_name: str,
    params: dict,
    numeric_columns: list[str],
    categorical_columns: list[str],
    scale_pos_weight: float,
) -> Pipeline:
    """Build one trial's pipeline with a brand-new, unfitted preprocessor
    (never shared across trials/models) so every fit starts clean.
    """
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)

    if model_name == "logistic_regression":
        return build_logistic_regression_trial(preprocessor, params)
    if model_name == "random_forest":
        return build_random_forest_trial(preprocessor, params)
    if model_name == "xgboost":
        return build_xgboost_trial(preprocessor, params, scale_pos_weight)
    if model_name == "lightgbm":
        return build_lightgbm_trial(preprocessor, params)
    if model_name == "catboost":
        return build_catboost_trial(preprocessor, params)
    raise ValueError(f"Unknown model_name: {model_name}")


def _trial_row(model_name: str, params: dict, metrics: dict, training_time_seconds: float) -> dict:
    return {
        "model": model_name,
        "hyperparameters": json.dumps(params, sort_keys=True),
        "val_pr_auc": metrics["pr_auc"],
        "val_roc_auc": metrics["roc_auc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "training_time_seconds": training_time_seconds,
    }


def run_hyperparameter_tuning(
    features: pd.DataFrame | None = None,
    param_grids: dict[str, list[dict]] | None = None,
) -> dict:
    """Tune all five models on TRAIN, select using VALIDATION PR-AUC only.

    The TEST split is produced by split_by_snapshot_date (so callers get it
    back for a later, separate final evaluation) but is never referenced
    anywhere in this function's fitting, scoring, or selection logic.

    Returns a dict with: splits, feature_columns, results (tidy trial-level
    DataFrame incl. the Phase 5 baseline Logistic Regression as a labeled
    row, sorted by val_pr_auc descending, with exactly one 'selected' row),
    best_per_model (best trial per tuned model, with its fitted pipeline),
    baseline_metrics, final_model_name, final_pipeline, final_hyperparameters,
    final_metrics, improved_over_baseline, decision_reason.
    """
    if features is None:
        features = load_feature_dataset()
    if param_grids is None:
        param_grids = DEFAULT_PARAM_GRIDS

    splits = split_by_snapshot_date(features)
    numeric_columns, categorical_columns = get_feature_columns(features)
    feature_columns = numeric_columns + categorical_columns

    X_train = splits["train"][feature_columns]
    y_train = splits["train"][TARGET_COLUMN]
    X_val = splits["validation"][feature_columns]
    y_val = splits["validation"][TARGET_COLUMN]

    scale_pos_weight = compute_scale_pos_weight(y_train)

    trial_rows: list[dict] = []
    best_per_model: dict[str, dict] = {}

    for model_name, grid in param_grids.items():
        for params in grid:
            pipeline = _build_trial_pipeline(
                model_name, params, numeric_columns, categorical_columns, scale_pos_weight
            )

            start = time.perf_counter()
            pipeline.fit(X_train, y_train)  # preprocessing + model fit on TRAIN only
            elapsed = time.perf_counter() - start

            metrics = evaluate_model(pipeline, X_val, y_val, threshold=CLASSIFICATION_THRESHOLD)
            trial_rows.append(_trial_row(model_name, params, metrics, elapsed))

            current_best = best_per_model.get(model_name)
            if current_best is None or metrics["pr_auc"] > current_best["metrics"]["pr_auc"]:
                best_per_model[model_name] = {
                    "pipeline": pipeline,
                    "params": params,
                    "metrics": metrics,
                    "training_time_seconds": elapsed,
                }

    # Phase 5/6 baseline Logistic Regression, refit fresh on this exact
    # train/validation split so the comparison is apples-to-apples and does
    # not depend on a possibly stale reports/baseline_results.csv.
    baseline_preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    baseline_pipeline = build_logistic_regression_pipeline(baseline_preprocessor)
    baseline_start = time.perf_counter()
    baseline_pipeline.fit(X_train, y_train)
    baseline_elapsed = time.perf_counter() - baseline_start
    baseline_metrics = evaluate_model(baseline_pipeline, X_val, y_val, threshold=CLASSIFICATION_THRESHOLD)
    baseline_params = {"C": 1.0, "class_weight": "balanced"}
    trial_rows.append(_trial_row(BASELINE_MODEL_LABEL, baseline_params, baseline_metrics, baseline_elapsed))

    results_df = pd.DataFrame(trial_rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)

    best_tuned_model_name = max(best_per_model, key=lambda m: best_per_model[m]["metrics"]["pr_auc"])
    best_tuned = best_per_model[best_tuned_model_name]
    improved = best_tuned["metrics"]["pr_auc"] > baseline_metrics["pr_auc"]

    if improved:
        final_model_name = best_tuned_model_name
        final_pipeline = best_tuned["pipeline"]
        final_hyperparameters = best_tuned["params"]
        final_metrics = best_tuned["metrics"]
        match_label, match_params = best_tuned_model_name, best_tuned["params"]
        decision_reason = (
            f"Tuned {best_tuned_model_name} reached validation PR-AUC "
            f"{final_metrics['pr_auc']:.4f}, above the Phase 5/6 baseline Logistic "
            f"Regression's {baseline_metrics['pr_auc']:.4f}."
        )
    else:
        final_model_name = "logistic_regression"
        final_pipeline = baseline_pipeline
        final_hyperparameters = baseline_params
        final_metrics = baseline_metrics
        match_label, match_params = BASELINE_MODEL_LABEL, baseline_params
        decision_reason = (
            f"No tuned model exceeded the baseline Logistic Regression's validation "
            f"PR-AUC ({baseline_metrics['pr_auc']:.4f}); the best tuned candidate was "
            f"{best_tuned_model_name} at {best_tuned['metrics']['pr_auc']:.4f}. Keeping "
            "the simpler, already-strong baseline model rather than a more complex one "
            "that does not actually rank churners better on held-out validation data."
        )

    match_hyperparameters = json.dumps(match_params, sort_keys=True)
    results_df["selected"] = (results_df["model"] == match_label) & (
        results_df["hyperparameters"] == match_hyperparameters
    )

    return {
        "splits": splits,
        "feature_columns": feature_columns,
        "results": results_df,
        "best_per_model": best_per_model,
        "baseline_pipeline": baseline_pipeline,
        "baseline_metrics": baseline_metrics,
        "final_model_name": final_model_name,
        "final_pipeline": final_pipeline,
        "final_hyperparameters": final_hyperparameters,
        "final_metrics": final_metrics,
        "improved_over_baseline": improved,
        "decision_reason": decision_reason,
    }


def evaluate_final_candidate_on_test(pipeline: Pipeline, splits: dict, feature_columns: list[str]) -> dict:
    """One final, non-selecting evaluation of the already-chosen pipeline on
    the TEST split. Must only ever be called after run_hyperparameter_tuning
    has already picked final_pipeline using validation PR-AUC alone.
    """
    X_test = splits["test"][feature_columns]
    y_test = splits["test"][TARGET_COLUMN]
    return evaluate_model(pipeline, X_test, y_test, threshold=CLASSIFICATION_THRESHOLD)


def save_tuning_results(results: pd.DataFrame, filename: str = "tuning_results.csv") -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / filename
    results.to_csv(output_path, index=False)
    return output_path


def save_tuned_model(pipeline: Pipeline, model_name: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / f"tuned_{model_name}.joblib"
    joblib.dump(pipeline, output_path)
    return output_path


if __name__ == "__main__":
    tuning = run_hyperparameter_tuning()

    print(tuning["results"].to_string(index=False))
    print(f"\nDecision: {tuning['decision_reason']}")

    results_path = save_tuning_results(tuning["results"])
    print(f"Saved tuning results -> {results_path}")

    model_path = save_tuned_model(tuning["final_pipeline"], tuning["final_model_name"])
    print(f"Saved final model ({tuning['final_model_name']}) -> {model_path}")

    test_metrics = evaluate_final_candidate_on_test(
        tuning["final_pipeline"], tuning["splits"], tuning["feature_columns"]
    )
    print(f"\nFinal candidate ({tuning['final_model_name']}) TEST metrics (reporting only): {test_metrics}")
