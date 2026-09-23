"""Baseline churn_next_30_days models: time-based split, leakage-safe
preprocessing, and simple Logistic Regression / Random Forest baselines.

Phase 5 scope only: baseline models with simple, untuned hyperparameters.
No hyperparameter tuning, threshold optimization, SHAP, or MLflow -- those
belong to later phases.

Splitting convention: chronological on snapshot_date, never random. Fitting
convention: every preprocessing step (imputation, scaling, encoding) is fit
on the training split only; validation and test are only ever transformed.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FEATURE_DATASET_PATH = PROCESSED_DATA_DIR / "customer_features.parquet"

RANDOM_SEED = 42
CLASSIFICATION_THRESHOLD = 0.5
TARGET_COLUMN = "churn_next_30_days"

# Columns that must never be used as model features: identifiers, the
# snapshot key (used only for splitting), and anything that would reveal
# the future churn event itself. churn_date/churn_reason are not actually
# present in customer_features.parquet, but are excluded defensively in
# case this module is ever pointed at an earlier-stage frame that has them.
NON_FEATURE_COLUMNS = ["customer_id", "snapshot_date", "churn_date", "churn_reason", TARGET_COLUMN]

# Chronological split boundaries (inclusive), aligned to the 17 monthly
# snapshot dates produced by src/features/feature_engineering.py:
# 11 train snapshots, 3 validation snapshots, 3 test snapshots, with no
# overlap between windows.
TRAIN_START = "2025-01-31"
TRAIN_END = "2025-11-30"
VALIDATION_START = "2025-12-31"
VALIDATION_END = "2026-02-28"
TEST_START = "2026-03-31"
TEST_END = "2026-05-31"


def load_feature_dataset(path: Path = FEATURE_DATASET_PATH) -> pd.DataFrame:
    """Load the Phase 4 feature dataset, failing loudly if it's missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {path}. "
            "Run notebooks/04_feature_engineering.ipynb (or src/features/feature_engineering.py) first."
        )
    return pd.read_parquet(path)


def split_by_snapshot_date(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Chronological train/validation/test split on snapshot_date -- never a random split."""
    train_start, train_end = pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)
    val_start, val_end = pd.Timestamp(VALIDATION_START), pd.Timestamp(VALIDATION_END)
    test_start, test_end = pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)

    if not (train_end < val_start and val_end < test_start):
        raise ValueError("Split windows are not strictly chronological / overlap.")

    train = features.loc[features["snapshot_date"].between(train_start, train_end)].copy()
    validation = features.loc[features["snapshot_date"].between(val_start, val_end)].copy()
    test = features.loc[features["snapshot_date"].between(test_start, test_end)].copy()

    return {"train": train, "validation": validation, "test": test}


def summarize_split(name: str, split_df: pd.DataFrame) -> dict:
    """Row/customer/class-balance summary for one split (used for printing and tests)."""
    total = len(split_df)
    positive = int(split_df[TARGET_COLUMN].sum())
    negative = total - positive

    return {
        "split": name,
        "rows": total,
        "unique_customers": split_df["customer_id"].nunique(),
        "positive_count": positive,
        "negative_count": negative,
        "positive_pct": round(positive / total * 100, 3) if total else 0.0,
        "min_snapshot_date": split_df["snapshot_date"].min(),
        "max_snapshot_date": split_df["snapshot_date"].max(),
    }


def get_feature_columns(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns): every column except
    identifiers, snapshot_date, churn_date/churn_reason, and the target.
    """
    candidate_columns = [c for c in features.columns if c not in NON_FEATURE_COLUMNS]
    categorical_columns = [c for c in candidate_columns if pd.api.types.is_string_dtype(features[c])]
    numeric_columns = [c for c in candidate_columns if c not in categorical_columns]

    return numeric_columns, categorical_columns


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    """Impute + scale numeric features, impute + one-hot-encode categorical
    features. This must be fit on the TRAINING split only -- run_baseline_experiment()
    enforces that by calling pipeline.fit() exclusively on X_train, so
    validation/test are only ever transformed with training-derived statistics.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])


def build_logistic_regression_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Baseline Logistic Regression. class_weight='balanced' re-weights the
    loss inversely to class frequency, countering the ~1.2% positive rate
    without altering the training data itself. Hyperparameters are kept
    deliberately simple -- this is a baseline, not a tuned model.
    """
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_random_forest_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Baseline Random Forest. class_weight='balanced' re-weights each tree's
    split criterion inversely to class frequency. max_depth is capped at 10
    mainly to keep the baseline fast and to avoid trivially memorizing the
    rare positive class; it is not a tuned value.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate_model(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = CLASSIFICATION_THRESHOLD
) -> dict:
    """Evaluate a fitted pipeline on (X, y) with imbalance-appropriate metrics.

    Accuracy is deliberately not reported: with a ~1.2% positive rate, a
    model that predicts "no churn" for every row scores ~98.8% accuracy
    while catching zero churners -- a useless baseline that accuracy alone
    would hide. ROC-AUC and PR-AUC (average precision) are reported instead
    because they evaluate ranking quality across all thresholds; PR-AUC in
    particular is far more informative under heavy class imbalance since it
    is driven entirely by how well the model separates the rare positive
    class, without the large true-negative count inflating the score the
    way ROC-AUC can. Recall matters most operationally: a false negative
    here means an about-to-churn customer is missed entirely and never
    reaches a retention action, which is the core failure mode this whole
    project exists to prevent.
    """
    predicted_probabilities = pipeline.predict_proba(X)[:, 1]
    predicted_labels = (predicted_probabilities >= threshold).astype(int)

    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        y, predicted_labels, labels=[0, 1]
    ).ravel()

    return {
        "roc_auc": roc_auc_score(y, predicted_probabilities),
        "pr_auc": average_precision_score(y, predicted_probabilities),
        "precision": precision_score(y, predicted_labels, zero_division=0),
        "recall": recall_score(y, predicted_labels, zero_division=0),
        "f1": f1_score(y, predicted_labels, zero_division=0),
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_positives": int(true_positives),
        "positive_prediction_rate": float(predicted_labels.mean()),
        "threshold": threshold,
    }


def save_pipeline(pipeline: Pipeline, filename: str) -> Path:
    """Serialize a fitted pipeline to models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / filename
    joblib.dump(pipeline, output_path)
    return output_path


def save_results(results: pd.DataFrame, filename: str = "baseline_results.csv") -> Path:
    """Save the tidy (model, split, metrics) results table to reports/."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / filename
    results.to_csv(output_path, index=False)
    return output_path


def run_baseline_experiment(features: pd.DataFrame | None = None) -> dict:
    """End-to-end baseline experiment: split, preprocess, train, evaluate.

    Returns a dict with the splits, feature column lists, fitted pipelines,
    and a tidy (model, split, metric...) results DataFrame. Does not save
    anything to disk -- callers (the notebook, __main__ below) decide when
    to persist results.
    """
    if features is None:
        features = load_feature_dataset()

    splits = split_by_snapshot_date(features)
    numeric_columns, categorical_columns = get_feature_columns(features)
    feature_columns = numeric_columns + categorical_columns

    X_train = splits["train"][feature_columns]
    y_train = splits["train"][TARGET_COLUMN]

    pipeline_builders = {
        "logistic_regression": build_logistic_regression_pipeline,
        "random_forest": build_random_forest_pipeline,
    }

    fitted_pipelines = {}
    results_rows = []

    for model_name, build_pipeline in pipeline_builders.items():
        pipeline = build_pipeline(build_preprocessor(numeric_columns, categorical_columns))
        pipeline.fit(X_train, y_train)  # preprocessing + model fit on TRAIN only
        fitted_pipelines[model_name] = pipeline

        for split_name in ("validation", "test"):
            split_df = splits[split_name]
            X_split = split_df[feature_columns]
            y_split = split_df[TARGET_COLUMN]

            metrics = evaluate_model(pipeline, X_split, y_split)
            results_rows.append({"model": model_name, "split": split_name, **metrics})

    return {
        "splits": splits,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "feature_columns": feature_columns,
        "pipelines": fitted_pipelines,
        "results": pd.DataFrame(results_rows),
    }


if __name__ == "__main__":
    experiment = run_baseline_experiment()

    for split_name, split_df in experiment["splits"].items():
        print(summarize_split(split_name, split_df))

    print(experiment["results"].to_string(index=False))

    for model_name, pipeline in experiment["pipelines"].items():
        path = save_pipeline(pipeline, f"{model_name}_baseline.joblib")
        print(f"Saved {model_name} -> {path}")

    results_path = save_results(experiment["results"])
    print(f"Saved results -> {results_path}")
