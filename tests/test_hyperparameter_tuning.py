"""Tests for src/models/hyperparameter_tuning.py.

Uses a small, fully deterministic synthetic feature frame (mirrors the
fixtures in test_baseline_model.py / test_model_comparison.py) together
with a reduced, single-combination-per-model param grid so the full
tuning sweep -- five real model fits, not mocks -- stays fast in CI.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models import baseline as bl
from src.models import hyperparameter_tuning as ht

EXPECTED_MODEL_NAMES = {"logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"}

# A tiny grid (1-2 combinations per model) so tests exercise the real
# tuning loop end-to-end without the full Phase 7 search space.
FAST_TEST_GRIDS = {
    "logistic_regression": [{"C": 0.1}, {"C": 1.0}],
    "random_forest": [{"n_estimators": 50, "max_depth": 4, "min_samples_leaf": 1, "max_features": "sqrt"}],
    "xgboost": [
        {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1, "subsample": 1.0, "colsample_bytree": 1.0}
    ],
    "lightgbm": [{"n_estimators": 50, "num_leaves": 7, "learning_rate": 0.1, "max_depth": -1}],
    "catboost": [{"iterations": 50, "depth": 3, "learning_rate": 0.1, "l2_leaf_reg": 3.0}],
}


@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_per_split = 80

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


def test_chronological_split_is_preserved(synthetic_features: pd.DataFrame) -> None:
    tuning = ht.run_hyperparameter_tuning(synthetic_features, param_grids=FAST_TEST_GRIDS)
    splits = tuning["splits"]

    assert splits["train"]["snapshot_date"].max() < splits["validation"]["snapshot_date"].min()
    assert splits["validation"]["snapshot_date"].max() < splits["test"]["snapshot_date"].min()
    assert sum(len(df) for df in splits.values()) == len(synthetic_features)


def test_test_set_never_used_for_tuning_or_selection(synthetic_features: pd.DataFrame) -> None:
    """Corrupt the TEST-window rows (NaN features, an invalid label) before
    tuning. If tuning or selection ever touched the test split, this would
    either raise or produce NaN/out-of-range metrics. It must not.
    """
    corrupted = synthetic_features.copy()
    test_mask = corrupted["snapshot_date"] == pd.Timestamp(bl.TEST_START)
    corrupted.loc[test_mask, "some_metric"] = np.nan
    corrupted.loc[test_mask, "another_metric"] = np.inf
    corrupted.loc[test_mask, "churn_next_30_days"] = -999

    tuning = ht.run_hyperparameter_tuning(corrupted, param_grids=FAST_TEST_GRIDS)
    results = tuning["results"]

    assert not results["val_pr_auc"].isna().any()
    assert results["val_pr_auc"].between(0, 1).all()
    assert results["val_roc_auc"].between(0, 1).all()
    assert tuning["final_metrics"]["pr_auc"] == pytest.approx(
        results.loc[results["selected"], "val_pr_auc"].iloc[0]
    )


def test_preprocessing_fit_only_on_train(synthetic_features: pd.DataFrame) -> None:
    numeric_columns, categorical_columns = bl.get_feature_columns(synthetic_features)
    splits = bl.split_by_snapshot_date(synthetic_features)
    feature_columns = numeric_columns + categorical_columns

    pipeline = ht._build_trial_pipeline(
        "logistic_regression", {"C": 1.0}, numeric_columns, categorical_columns, scale_pos_weight=1.0
    )
    pipeline.fit(splits["train"][feature_columns], splits["train"][bl.TARGET_COLUMN])

    fitted_imputer = pipeline.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["imputer"]
    metric_idx = numeric_columns.index("some_metric")
    train_only_median = splits["train"][feature_columns]["some_metric"].median()

    assert fitted_imputer.statistics_[metric_idx] == pytest.approx(train_only_median)


def test_no_validation_leakage_into_training_statistics(synthetic_features: pd.DataFrame) -> None:
    """Same idea as test_baseline_model.py's leakage check: validation gets a
    different distribution than train, so a preprocessor that (incorrectly)
    saw validation during fit would show a shifted statistic.
    """
    shifted = synthetic_features.copy()
    val_mask = shifted["snapshot_date"] == pd.Timestamp(bl.VALIDATION_START)
    shifted.loc[val_mask, "some_metric"] = shifted.loc[val_mask, "some_metric"] + 500.0

    numeric_columns, categorical_columns = bl.get_feature_columns(shifted)
    splits = bl.split_by_snapshot_date(shifted)
    feature_columns = numeric_columns + categorical_columns

    pipeline = ht._build_trial_pipeline(
        "logistic_regression", {"C": 1.0}, numeric_columns, categorical_columns, scale_pos_weight=1.0
    )
    pipeline.fit(splits["train"][feature_columns], splits["train"][bl.TARGET_COLUMN])

    fitted_imputer = pipeline.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["imputer"]
    metric_idx = numeric_columns.index("some_metric")
    train_only_median = splits["train"][feature_columns]["some_metric"].median()

    assert fitted_imputer.statistics_[metric_idx] == pytest.approx(train_only_median)
    assert fitted_imputer.statistics_[metric_idx] < 100  # would be ~500+ if validation leaked in


def test_tuning_produces_validation_pr_auc_for_all_models(synthetic_features: pd.DataFrame) -> None:
    tuning = ht.run_hyperparameter_tuning(synthetic_features, param_grids=FAST_TEST_GRIDS)
    results = tuning["results"]

    assert set(FAST_TEST_GRIDS) <= set(results["model"])
    assert ht.BASELINE_MODEL_LABEL in set(results["model"])

    required_columns = {
        "model", "hyperparameters", "val_pr_auc", "val_roc_auc",
        "precision", "recall", "f1", "training_time_seconds",
    }
    assert required_columns <= set(results.columns)
    assert results["val_pr_auc"].between(0, 1).all()
    assert (results["training_time_seconds"] > 0).all()

    expected_trials = sum(len(grid) for grid in FAST_TEST_GRIDS.values()) + 1  # +1 baseline row
    assert len(results) == expected_trials


def test_exactly_one_row_selected(synthetic_features: pd.DataFrame) -> None:
    tuning = ht.run_hyperparameter_tuning(synthetic_features, param_grids=FAST_TEST_GRIDS)
    assert tuning["results"]["selected"].sum() == 1


def test_best_model_can_be_saved_and_loaded(synthetic_features: pd.DataFrame, tmp_path) -> None:
    tuning = ht.run_hyperparameter_tuning(synthetic_features, param_grids=FAST_TEST_GRIDS)

    original_models_dir = ht.MODELS_DIR
    ht.MODELS_DIR = tmp_path
    try:
        saved_path = ht.save_tuned_model(tuning["final_pipeline"], tuning["final_model_name"])
    finally:
        ht.MODELS_DIR = original_models_dir

    assert saved_path.exists()
    loaded_pipeline = joblib.load(saved_path)

    X_val = tuning["splits"]["validation"][tuning["feature_columns"]]
    original_predictions = tuning["final_pipeline"].predict_proba(X_val)[:, 1]
    loaded_predictions = loaded_pipeline.predict_proba(X_val)[:, 1]
    np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)


def test_save_tuning_results_round_trips(synthetic_features: pd.DataFrame, tmp_path) -> None:
    tuning = ht.run_hyperparameter_tuning(synthetic_features, param_grids=FAST_TEST_GRIDS)

    original_reports_dir = ht.REPORTS_DIR
    ht.REPORTS_DIR = tmp_path
    try:
        output_path = ht.save_tuning_results(tuning["results"], filename="test_tuning_results.csv")
    finally:
        ht.REPORTS_DIR = original_reports_dir

    saved = pd.read_csv(output_path)
    required_columns = {
        "model", "hyperparameters", "val_pr_auc", "val_roc_auc",
        "precision", "recall", "f1", "training_time_seconds", "selected",
    }
    assert required_columns <= set(saved.columns)
    assert len(saved) == len(tuning["results"])


def test_deterministic_results_with_fixed_seed(synthetic_features: pd.DataFrame) -> None:
    first = ht.run_hyperparameter_tuning(synthetic_features, param_grids=FAST_TEST_GRIDS)
    second = ht.run_hyperparameter_tuning(synthetic_features, param_grids=FAST_TEST_GRIDS)

    pd.testing.assert_series_equal(
        first["results"]["val_pr_auc"].sort_index(),
        second["results"]["val_pr_auc"].sort_index(),
    )
    assert first["final_model_name"] == second["final_model_name"]
    assert first["final_metrics"]["pr_auc"] == pytest.approx(second["final_metrics"]["pr_auc"])


def test_final_test_evaluation_is_separate_from_selection(synthetic_features: pd.DataFrame) -> None:
    """evaluate_final_candidate_on_test must be an explicit, separate call --
    never invoked inside run_hyperparameter_tuning -- and must only score
    the already-chosen pipeline, not influence which one was chosen.
    """
    tuning = ht.run_hyperparameter_tuning(synthetic_features, param_grids=FAST_TEST_GRIDS)
    test_metrics = ht.evaluate_final_candidate_on_test(
        tuning["final_pipeline"], tuning["splits"], tuning["feature_columns"]
    )

    assert set(test_metrics) >= {"roc_auc", "pr_auc", "precision", "recall", "f1"}
    assert 0 <= test_metrics["pr_auc"] <= 1


def test_decision_documents_baseline_comparison(synthetic_features: pd.DataFrame) -> None:
    tuning = ht.run_hyperparameter_tuning(synthetic_features, param_grids=FAST_TEST_GRIDS)

    assert isinstance(tuning["improved_over_baseline"], (bool, np.bool_))
    assert isinstance(tuning["decision_reason"], str) and len(tuning["decision_reason"]) > 0
    if not tuning["improved_over_baseline"]:
        assert tuning["final_model_name"] == "logistic_regression"
