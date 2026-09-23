"""Tests for src/models/baseline.py.

Most tests use a small, fully deterministic synthetic feature frame (fast,
isolated) rather than the full ~137k-row real dataset. The synthetic
validation/test splits are given a deliberately different numeric
distribution than train specifically so that a "fit preprocessing on
everything" bug would be caught by test_preprocessing_fit_only_on_train.
"""

import numpy as np
import pandas as pd
import pytest

from src.models import baseline as bl


# ---------------------------------------------------------------------------
# Small deterministic synthetic dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_per_split = 40

    def make_split(snapshot_date: str, metric_mean: float) -> pd.DataFrame:
        return pd.DataFrame({
            "customer_id": [f"C{i:04d}" for i in range(n_per_split)],
            "snapshot_date": pd.Timestamp(snapshot_date),
            "age": rng.integers(18, 70, n_per_split),
            "country": rng.choice(["India", "Nepal"], n_per_split),
            "plan": rng.choice(["Basic", "Premium"], n_per_split),
            "contract_type": rng.choice(["Monthly", "Yearly"], n_per_split),
            "acquisition_channel": rng.choice(["Organic", "Referral"], n_per_split),
            "tenure_days": rng.integers(30, 900, n_per_split),
            # Deliberately different mean per split -- exposes a preprocessor
            # that was (incorrectly) fit on the full dataset.
            "some_metric": rng.normal(loc=metric_mean, scale=1.0, size=n_per_split),
            # Deterministic ~10% positive rate (4 of 40) -- avoids a flaky
            # all-one-class draw that would break model fitting.
            "churn_next_30_days": [1 if i % 10 == 0 else 0 for i in range(n_per_split)],
        })

    # validation and test are shifted the SAME direction (not +100/-100,
    # which a median -- unlike a mean -- would cancel out for two equal-sized
    # groups on opposite sides of train): this way, if a bug fit the
    # preprocessor on the full dataset, the combined median would clearly
    # move toward 100 instead of staying near train's median of ~0.
    train = make_split(bl.TRAIN_START, metric_mean=0.0)
    validation = make_split(bl.VALIDATION_START, metric_mean=100.0)
    test = make_split(bl.TEST_START, metric_mean=100.0)

    return pd.concat([train, validation, test], ignore_index=True)


def test_chronological_split_correctness(synthetic_features: pd.DataFrame) -> None:
    splits = bl.split_by_snapshot_date(synthetic_features)

    assert len(splits["train"]) == 40
    assert len(splits["validation"]) == 40
    assert len(splits["test"]) == 40
    assert splits["train"]["snapshot_date"].max() < splits["validation"]["snapshot_date"].min()
    assert splits["validation"]["snapshot_date"].max() < splits["test"]["snapshot_date"].min()


def test_no_train_test_date_overlap(synthetic_features: pd.DataFrame) -> None:
    splits = bl.split_by_snapshot_date(synthetic_features)
    train_dates = set(splits["train"]["snapshot_date"])
    validation_dates = set(splits["validation"]["snapshot_date"])
    test_dates = set(splits["test"]["snapshot_date"])

    assert train_dates.isdisjoint(validation_dates)
    assert train_dates.isdisjoint(test_dates)
    assert validation_dates.isdisjoint(test_dates)


def test_target_and_identifiers_excluded_from_features(synthetic_features: pd.DataFrame) -> None:
    numeric_columns, categorical_columns = bl.get_feature_columns(synthetic_features)
    feature_columns = numeric_columns + categorical_columns

    assert bl.TARGET_COLUMN not in feature_columns
    assert "customer_id" not in feature_columns
    assert "snapshot_date" not in feature_columns
    assert "churn_date" not in feature_columns
    assert "churn_reason" not in feature_columns


def test_categorical_and_numeric_columns_correctly_typed(synthetic_features: pd.DataFrame) -> None:
    numeric_columns, categorical_columns = bl.get_feature_columns(synthetic_features)

    assert set(categorical_columns) == {"country", "plan", "contract_type", "acquisition_channel"}
    assert "age" in numeric_columns
    assert "tenure_days" in numeric_columns
    assert "some_metric" in numeric_columns


def test_preprocessing_fit_only_on_train(synthetic_features: pd.DataFrame) -> None:
    """The fitted imputer's statistics must match TRAIN-only medians, not
    medians computed across train+validation+test -- proving validation/test
    data never influenced what the preprocessor learned.
    """
    splits = bl.split_by_snapshot_date(synthetic_features)
    numeric_columns, categorical_columns = bl.get_feature_columns(synthetic_features)
    feature_columns = numeric_columns + categorical_columns

    X_train = splits["train"][feature_columns]
    preprocessor = bl.build_preprocessor(numeric_columns, categorical_columns)
    preprocessor.fit(X_train)

    fitted_imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    metric_idx = numeric_columns.index("some_metric")

    train_only_median = X_train["some_metric"].median()
    full_data_median = synthetic_features["some_metric"].median()

    assert fitted_imputer.statistics_[metric_idx] == pytest.approx(train_only_median)
    # validation/test were built with means of +100 / -100, so a preprocessor
    # that leaked them in would land far from the train-only median.
    assert fitted_imputer.statistics_[metric_idx] != pytest.approx(full_data_median, abs=1.0)


def test_transform_on_val_test_does_not_change_fitted_statistics(synthetic_features: pd.DataFrame) -> None:
    """Transforming validation/test must not refit (and therefore must not
    shift) the preprocessor's learned statistics.
    """
    splits = bl.split_by_snapshot_date(synthetic_features)
    numeric_columns, categorical_columns = bl.get_feature_columns(synthetic_features)
    feature_columns = numeric_columns + categorical_columns

    preprocessor = bl.build_preprocessor(numeric_columns, categorical_columns)
    preprocessor.fit(splits["train"][feature_columns])
    statistics_before = preprocessor.named_transformers_["numeric"].named_steps["imputer"].statistics_.copy()

    preprocessor.transform(splits["validation"][feature_columns])
    preprocessor.transform(splits["test"][feature_columns])

    statistics_after = preprocessor.named_transformers_["numeric"].named_steps["imputer"].statistics_
    np.testing.assert_array_equal(statistics_before, statistics_after)


def test_models_train_successfully(synthetic_features: pd.DataFrame) -> None:
    experiment = bl.run_baseline_experiment(synthetic_features)

    lr_model = experiment["pipelines"]["logistic_regression"].named_steps["model"]
    rf_model = experiment["pipelines"]["random_forest"].named_steps["model"]

    assert hasattr(lr_model, "coef_")
    assert hasattr(rf_model, "feature_importances_")


def test_metrics_are_generated(synthetic_features: pd.DataFrame) -> None:
    experiment = bl.run_baseline_experiment(synthetic_features)
    results = experiment["results"]

    assert len(results) == 4  # 2 models x {validation, test}
    assert set(results["model"]) == {"logistic_regression", "random_forest"}
    assert set(results["split"]) == {"validation", "test"}

    expected_columns = {"roc_auc", "pr_auc", "precision", "recall", "f1", "positive_prediction_rate"}
    assert expected_columns <= set(results.columns)

    assert results["roc_auc"].between(0, 1).all()
    assert results["pr_auc"].between(0, 1).all()
    assert results["precision"].between(0, 1).all()
    assert results["recall"].between(0, 1).all()
    assert results["f1"].between(0, 1).all()


# ---------------------------------------------------------------------------
# Split checks against the real generated feature dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_features() -> pd.DataFrame:
    return bl.load_feature_dataset()


def test_real_split_rows_sum_to_total(real_features: pd.DataFrame) -> None:
    splits = bl.split_by_snapshot_date(real_features)
    assert sum(len(df) for df in splits.values()) == len(real_features)


def test_real_split_is_chronological_with_no_overlap(real_features: pd.DataFrame) -> None:
    splits = bl.split_by_snapshot_date(real_features)
    assert splits["train"]["snapshot_date"].max() < splits["validation"]["snapshot_date"].min()
    assert splits["validation"]["snapshot_date"].max() < splits["test"]["snapshot_date"].min()
