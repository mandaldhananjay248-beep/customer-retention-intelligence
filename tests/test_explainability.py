"""Tests for src/models/explainability.py.

Most tests fit a small, fully deterministic synthetic Logistic Regression
pipeline (same C/class_weight as the real Phase 7 selection) rather than
loading the ~20KB real model file, mirroring the synthetic-fixture
approach used throughout tests/test_baseline_model.py and
tests/test_model_comparison.py. A separate section at the bottom exercises
load_tuned_pipeline() against the real Phase 7 artifact.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.models import baseline as bl
from src.models import explainability as ex


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
            # A bool-dtype feature -- classified as "numeric" by
            # get_feature_columns (it isn't string dtype), exactly like
            # the real dataset's has_any_support_history etc. Exercises
            # explain_customer's numeric branch with a raw numpy.bool_
            # value (see test_explain_customer_never_returns_numpy_scalars).
            "has_history": rng.choice([True, False], n_per_split),
            "churn_next_30_days": [1 if i % 10 == 0 else 0 for i in range(n_per_split)],
        })

    train = make_split(bl.TRAIN_START)
    validation = make_split(bl.VALIDATION_START)
    test = make_split(bl.TEST_START)

    return pd.concat([train, validation, test], ignore_index=True)


@pytest.fixture
def fitted_tuned_style_pipeline(synthetic_features: pd.DataFrame):
    """A pipeline built exactly like the real Phase 7 selection
    (C=0.01, class_weight="balanced") but fit on the small synthetic
    dataset, so explainability logic can be tested fast and in isolation
    from the real ~137k-row dataset / real model file.
    """
    numeric_columns, categorical_columns = bl.get_feature_columns(synthetic_features)
    feature_columns = numeric_columns + categorical_columns
    splits = bl.split_by_snapshot_date(synthetic_features)

    preprocessor = bl.build_preprocessor(numeric_columns, categorical_columns)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    model = LogisticRegression(
        C=ex.EXPECTED_C, class_weight=ex.EXPECTED_CLASS_WEIGHT, max_iter=1000, random_state=bl.RANDOM_SEED
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train = splits["train"][feature_columns]
    y_train = splits["train"][bl.TARGET_COLUMN]
    pipeline.fit(X_train, y_train)

    return {
        "pipeline": pipeline,
        "splits": splits,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
    }


@pytest.fixture
def explainability_bundle(fitted_tuned_style_pipeline):
    pipeline = fitted_tuned_style_pipeline["pipeline"]
    splits = fitted_tuned_style_pipeline["splits"]
    feature_columns = fitted_tuned_style_pipeline["feature_columns"]
    categorical_columns = fitted_tuned_style_pipeline["categorical_columns"]

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_metadata = ex.build_feature_metadata(preprocessor, categorical_columns)
    display_names = feature_metadata["display_name"].tolist()

    X_train = splits["train"][feature_columns]
    X_val = splits["validation"][feature_columns]

    background = ex.build_background(preprocessor, X_train, sample_size=30)
    explainer = ex.build_explainer(pipeline, background)
    shap_values = ex.compute_shap_values(explainer, preprocessor, X_val, display_names)

    return {
        "pipeline": pipeline,
        "splits": splits,
        "feature_columns": feature_columns,
        "feature_metadata": feature_metadata,
        "display_names": display_names,
        "explainer": explainer,
        "shap_values": shap_values,
        "X_val": X_val,
    }


def test_feature_metadata_dimensions_match_transformed_features(explainability_bundle) -> None:
    pipeline = explainability_bundle["pipeline"]
    feature_metadata = explainability_bundle["feature_metadata"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    n_transformed = preprocessor.transform(explainability_bundle["X_val"]).shape[1]
    assert len(feature_metadata) == n_transformed
    assert len(feature_metadata) == model.coef_.shape[1]


def test_feature_metadata_recovers_categorical_origin(explainability_bundle) -> None:
    feature_metadata = explainability_bundle["feature_metadata"]
    categorical_rows = feature_metadata[feature_metadata["kind"] == "categorical"]

    assert set(categorical_rows["origin_column"]) == {
        "country", "plan", "contract_type", "acquisition_channel",
    }
    assert categorical_rows["category"].notna().all()


def test_shap_values_shape_matches_input_and_features(explainability_bundle) -> None:
    shap_values = explainability_bundle["shap_values"]
    X_val = explainability_bundle["X_val"]
    feature_metadata = explainability_bundle["feature_metadata"]

    assert shap_values.values.shape == (len(X_val), len(feature_metadata))


def test_feature_names_match_shap_values(explainability_bundle) -> None:
    shap_values = explainability_bundle["shap_values"]
    display_names = explainability_bundle["display_names"]

    assert list(shap_values.feature_names) == display_names
    assert len(shap_values.feature_names) == shap_values.values.shape[1]


def test_no_nan_or_inf_shap_values(explainability_bundle) -> None:
    values = explainability_bundle["shap_values"].values
    assert not np.isnan(values).any()
    assert not np.isinf(values).any()


def test_shap_values_reconstruct_predicted_probability(explainability_bundle) -> None:
    """SHAP values for a linear model must be additive-exact: base_value +
    sum(shap_values) equals the model's linear output, whose sigmoid is
    predict_proba. This is the core correctness check for LinearExplainer.
    """
    import scipy.special

    pipeline = explainability_bundle["pipeline"]
    shap_values = explainability_bundle["shap_values"]
    X_val = explainability_bundle["X_val"]

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(X_val)
    actual_probabilities = model.predict_proba(transformed)[:, 1]

    reconstructed_logits = shap_values.values.sum(axis=1) + np.ravel(shap_values.base_values)
    reconstructed_probabilities = scipy.special.expit(reconstructed_logits)

    np.testing.assert_allclose(reconstructed_probabilities, actual_probabilities, atol=1e-8)


def test_global_feature_importance_has_expected_columns_and_ranks(explainability_bundle) -> None:
    importance = ex.global_feature_importance(explainability_bundle["shap_values"])

    assert {"feature", "mean_abs_shap", "mean_shap", "rank"} <= set(importance.columns)
    assert list(importance["rank"]) == list(range(1, len(importance) + 1))
    assert importance["mean_abs_shap"].is_monotonic_decreasing
    assert set(importance["feature"]) == set(explainability_bundle["display_names"])


def test_explain_customer_returns_expected_keys_and_valid_probability(explainability_bundle) -> None:
    pipeline = explainability_bundle["pipeline"]
    explainer = explainability_bundle["explainer"]
    feature_metadata = explainability_bundle["feature_metadata"]
    X_val = explainability_bundle["X_val"]

    customer_row = X_val.iloc[[0]]
    explanation = ex.explain_customer(pipeline, explainer, customer_row, feature_metadata, risk_tier_bounds=(0.3, 0.7))

    required_keys = {
        "predicted_probability", "predicted_class", "base_value",
        "top_positive_drivers", "top_negative_drivers", "shap_values", "risk_tier",
    }
    assert required_keys <= set(explanation.keys())
    assert 0.0 <= explanation["predicted_probability"] <= 1.0
    assert explanation["predicted_class"] in (0, 1)
    assert explanation["risk_tier"] in {"High", "Medium", "Low"}
    assert len(explanation["shap_values"]) == len(feature_metadata)

    for driver in explanation["top_positive_drivers"]:
        assert driver["shap_value"] > 0
        assert "customer_value" in driver
    for driver in explanation["top_negative_drivers"]:
        assert driver["shap_value"] < 0


def test_explain_customer_never_returns_numpy_scalars(explainability_bundle) -> None:
    """Regression test: a bool-dtype feature (like the real dataset's
    has_any_support_history) is classified as "numeric" by
    get_feature_columns, so explain_customer's numeric branch used to
    attach a raw numpy.bool_ as customer_value. That value is JSON/Pydantic
    -serialized downstream by the Phase 11 API's DriverSignal.customer_value
    field, which fails on a bare numpy scalar (discovered via Phase 12's
    end-to-end dashboard verification). Every driver's customer_value must
    be a plain Python type.
    """
    pipeline = explainability_bundle["pipeline"]
    explainer = explainability_bundle["explainer"]
    feature_metadata = explainability_bundle["feature_metadata"]
    X_val = explainability_bundle["X_val"]

    # Ask for every feature as a "top" driver so has_history is guaranteed
    # to be included regardless of its actual SHAP rank.
    n_features = len(feature_metadata)
    customer_row = X_val.iloc[[0]]
    explanation = ex.explain_customer(
        pipeline, explainer, customer_row, feature_metadata, top_k=n_features
    )

    all_drivers = explanation["top_positive_drivers"] + explanation["top_negative_drivers"]
    has_history_drivers = [d for d in all_drivers if d["feature"] == "has_history"]
    assert has_history_drivers, "expected 'has_history' among the drivers for this top_k"

    for driver in all_drivers:
        value = driver["customer_value"]
        assert type(value).__module__ != "numpy", (
            f"{driver['feature']}: customer_value is a numpy type ({type(value)}), not JSON-serializable"
        )
        json.dumps(driver)  # must not raise


def test_explain_customer_rejects_multi_row_input(explainability_bundle) -> None:
    pipeline = explainability_bundle["pipeline"]
    explainer = explainability_bundle["explainer"]
    feature_metadata = explainability_bundle["feature_metadata"]
    X_val = explainability_bundle["X_val"]

    with pytest.raises(ValueError):
        ex.explain_customer(pipeline, explainer, X_val.iloc[:2], feature_metadata)


def test_representative_customers_have_valid_probabilities(explainability_bundle) -> None:
    pipeline = explainability_bundle["pipeline"]
    X_val = explainability_bundle["X_val"]

    positions = ex.select_representative_customers(pipeline, X_val)
    assert set(positions) == {"high", "medium", "low"}

    probabilities = pipeline.predict_proba(X_val)[:, 1]
    high_p = probabilities[positions["high"]]
    medium_p = probabilities[positions["medium"]]
    low_p = probabilities[positions["low"]]

    for p in (high_p, medium_p, low_p):
        assert 0.0 <= p <= 1.0
    assert high_p >= medium_p >= low_p


def test_risk_tier_bounds_and_assignment() -> None:
    probabilities = np.array([0.01, 0.02, 0.5, 0.5, 0.98, 0.99])
    low_cutoff, high_cutoff = ex.compute_risk_tier_bounds(probabilities)

    assert ex.assign_risk_tier(0.99, low_cutoff, high_cutoff) == "High"
    assert ex.assign_risk_tier(0.01, low_cutoff, high_cutoff) == "Low"
    assert ex.assign_risk_tier((low_cutoff + high_cutoff) / 2, low_cutoff, high_cutoff) == "Medium"


def test_background_built_only_from_train(fitted_tuned_style_pipeline) -> None:
    """The SHAP background must come from TRAIN only -- sampling from the
    full dataset (or validation) would leak validation-only information
    into the reference point every SHAP value is computed relative to.
    """
    splits = fitted_tuned_style_pipeline["splits"]
    feature_columns = fitted_tuned_style_pipeline["feature_columns"]
    pipeline = fitted_tuned_style_pipeline["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]

    X_train = splits["train"][feature_columns]
    background = ex.build_background(preprocessor, X_train, sample_size=1000)

    assert background.shape[0] == len(X_train)  # capped at available TRAIN rows, not padded from elsewhere


def test_deterministic_shap_values_with_fixed_seed(fitted_tuned_style_pipeline) -> None:
    pipeline = fitted_tuned_style_pipeline["pipeline"]
    splits = fitted_tuned_style_pipeline["splits"]
    feature_columns = fitted_tuned_style_pipeline["feature_columns"]
    categorical_columns = fitted_tuned_style_pipeline["categorical_columns"]

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_metadata = ex.build_feature_metadata(preprocessor, categorical_columns)
    display_names = feature_metadata["display_name"].tolist()
    X_train = splits["train"][feature_columns]
    X_val = splits["validation"][feature_columns]

    def run_once():
        background = ex.build_background(preprocessor, X_train, sample_size=30)
        explainer = ex.build_explainer(pipeline, background)
        return ex.compute_shap_values(explainer, preprocessor, X_val, display_names)

    first = run_once()
    second = run_once()

    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(np.ravel(first.base_values), np.ravel(second.base_values))


def test_save_feature_importance_round_trips(explainability_bundle, tmp_path) -> None:
    importance = ex.global_feature_importance(explainability_bundle["shap_values"])

    original_reports_dir = ex.REPORTS_DIR
    ex.REPORTS_DIR = tmp_path
    try:
        output_path = ex.save_feature_importance(importance, filename="test_shap_importance.csv")
    finally:
        ex.REPORTS_DIR = original_reports_dir

    saved = pd.read_csv(output_path)
    assert {"feature", "mean_abs_shap", "mean_shap", "rank"} <= set(saved.columns)
    assert len(saved) == len(importance)


# ---------------------------------------------------------------------------
# Real Phase 7 artifact checks (mirrors the real-dataset section of
# test_baseline_model.py) -- skipped if Phase 7 hasn't been run yet.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_tuned_pipeline():
    if not ex.TUNED_MODEL_PATH.exists():
        pytest.skip(f"{ex.TUNED_MODEL_PATH} not found -- run Phase 7 (src/models/hyperparameter_tuning.py) first")
    return ex.load_tuned_pipeline()


def test_real_tuned_pipeline_loads_and_matches_phase7_hyperparameters(real_tuned_pipeline) -> None:
    model = real_tuned_pipeline.named_steps["model"]
    assert model.C == ex.EXPECTED_C
    assert model.class_weight == ex.EXPECTED_CLASS_WEIGHT
    assert hasattr(model, "coef_")  # already fitted, never refit by this module


def test_load_tuned_pipeline_rejects_mismatched_hyperparameters(tmp_path) -> None:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    wrong_pipeline = Pipeline(steps=[
        ("preprocessor", bl.build_preprocessor(["age"], [])),
        ("model", LogisticRegression(C=1.0, class_weight="balanced")),
    ])
    wrong_pipeline.fit(pd.DataFrame({"age": [1, 2, 3, 4]}), pd.Series([0, 1, 0, 1]))

    bad_path = tmp_path / "wrong_model.joblib"
    joblib.dump(wrong_pipeline, bad_path)

    with pytest.raises(ValueError):
        ex.load_tuned_pipeline(path=bad_path)
