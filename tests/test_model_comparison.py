"""Tests for src/models/model_comparison.py.

Uses a small, fully deterministic synthetic feature frame (fast, isolated)
for the "does everything actually train and produce sane metrics" checks,
mirroring the approach in tests/test_baseline_model.py. This keeps the
suite fast while still exercising all five real model implementations
(not mocks).
"""

import numpy as np
import pandas as pd
import pytest

from src.models import baseline as bl
from src.models import model_comparison as mc

EXPECTED_MODEL_NAMES = {"logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"}


@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_per_split = 60

    def make_split(snapshot_date: str) -> pd.DataFrame:
        return pd.DataFrame({
            "customer_id": [f"C{i:04d}" for i in range(n_per_split)],
            "snapshot_date": pd.Timestamp(snapshot_date),
            "age": rng.integers(18, 70, n_per_split),
            "country": rng.choice(["India", "Nepal"], n_per_split),
            "plan": rng.choice(["Basic", "Premium"], n_per_split),
            "contract_type": rng.choice(["Monthly", "Yearly"], n_per_split),
            "acquisition_channel": rng.choice(["Organic", "Referral"], n_per_split),
            "tenure_days": rng.integers(30, 900, n_per_split),
            "some_metric": rng.normal(loc=0.0, scale=1.0, size=n_per_split),
            "another_metric": rng.normal(loc=5.0, scale=2.0, size=n_per_split),
            # Deterministic ~10% positive rate -- avoids a flaky all-one-class draw.
            "churn_next_30_days": [1 if i % 10 == 0 else 0 for i in range(n_per_split)],
        })

    train = make_split(bl.TRAIN_START)
    validation = make_split(bl.VALIDATION_START)
    test = make_split(bl.TEST_START)

    return pd.concat([train, validation, test], ignore_index=True)


def test_all_five_models_can_be_constructed(synthetic_features: pd.DataFrame) -> None:
    numeric_columns, categorical_columns = bl.get_feature_columns(synthetic_features)
    y_train = synthetic_features["churn_next_30_days"]

    pipelines = mc.build_all_pipelines(numeric_columns, categorical_columns, y_train)

    assert set(pipelines.keys()) == EXPECTED_MODEL_NAMES
    for pipeline in pipelines.values():
        assert pipeline.named_steps["preprocessor"] is not None
        assert pipeline.named_steps["model"] is not None


def test_all_five_models_train_on_small_dataset(synthetic_features: pd.DataFrame) -> None:
    experiment = mc.run_comparison_experiment(synthetic_features)

    assert set(experiment["pipelines"].keys()) == EXPECTED_MODEL_NAMES
    for model_name, pipeline in experiment["pipelines"].items():
        predictions = pipeline.predict_proba(experiment["splits"]["test"][experiment["feature_columns"]])
        assert predictions.shape[1] == 2, f"{model_name} did not produce 2-class probabilities"


def test_chronological_split_is_preserved(synthetic_features: pd.DataFrame) -> None:
    experiment = mc.run_comparison_experiment(synthetic_features)
    splits = experiment["splits"]

    assert splits["train"]["snapshot_date"].max() < splits["validation"]["snapshot_date"].min()
    assert splits["validation"]["snapshot_date"].max() < splits["test"]["snapshot_date"].min()
    assert sum(len(df) for df in splits.values()) == len(synthetic_features)


def test_target_not_included_as_feature(synthetic_features: pd.DataFrame) -> None:
    experiment = mc.run_comparison_experiment(synthetic_features)

    assert bl.TARGET_COLUMN not in experiment["feature_columns"]
    assert "customer_id" not in experiment["feature_columns"]
    assert "snapshot_date" not in experiment["feature_columns"]


def test_metrics_are_generated_for_all_models(synthetic_features: pd.DataFrame) -> None:
    experiment = mc.run_comparison_experiment(synthetic_features)
    results = experiment["results"]

    assert len(results) == len(EXPECTED_MODEL_NAMES) * 2  # x {validation, test}
    assert set(results["model"]) == EXPECTED_MODEL_NAMES
    assert set(results["split"]) == {"validation", "test"}

    required_columns = {"pr_auc", "roc_auc", "precision", "recall", "f1"}
    assert required_columns <= set(results.columns)


def test_no_nan_or_infinite_metrics(synthetic_features: pd.DataFrame) -> None:
    experiment = mc.run_comparison_experiment(synthetic_features)
    results = experiment["results"]

    metric_columns = ["pr_auc", "roc_auc", "precision", "recall", "f1"]
    numeric_values = results[metric_columns].to_numpy(dtype=float)

    assert not np.isnan(numeric_values).any(), "Found NaN evaluation metrics"
    assert not np.isinf(numeric_values).any(), "Found infinite evaluation metrics"
    for column in metric_columns:
        assert results[column].between(0, 1).all(), f"{column} outside [0, 1]"


def test_results_contain_all_five_models(synthetic_features: pd.DataFrame) -> None:
    experiment = mc.run_comparison_experiment(synthetic_features)
    models_in_results = set(experiment["results"]["model"].unique())

    assert models_in_results == EXPECTED_MODEL_NAMES


def test_scale_pos_weight_uses_training_labels_only() -> None:
    y_train = pd.Series([0] * 90 + [1] * 10)
    ratio = mc.compute_scale_pos_weight(y_train)
    assert ratio == pytest.approx(9.0)


def test_save_comparison_results_has_required_columns(synthetic_features: pd.DataFrame, tmp_path) -> None:
    experiment = mc.run_comparison_experiment(synthetic_features)

    original_reports_dir = mc.REPORTS_DIR
    mc.REPORTS_DIR = tmp_path
    try:
        output_path = mc.save_comparison_results(experiment["results"], filename="test_results.csv")
    finally:
        mc.REPORTS_DIR = original_reports_dir

    saved = pd.read_csv(output_path)
    required_columns = ["model", "split", "pr_auc", "roc_auc", "precision", "recall", "f1"]
    assert list(saved.columns[: len(required_columns)]) == required_columns
