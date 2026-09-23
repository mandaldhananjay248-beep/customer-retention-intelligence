"""Tests for src/models/mlflow_tracking.py.

Every test that exercises real MLflow tracking/registry calls uses a
temporary, per-test SQLite backend and artifact store (via tmp_path) plus
a uniquely-named experiment/registered-model -- never the project's real
mlruns.db/mlartifacts. This keeps the suite from polluting the developer's
actual tracking store and lets tests run in any order/parallelism without
colliding on experiment or registered-model names.

A separate, real-artifact section at the bottom exercises the full
run_phase10_tracking() against the actual Phase 7 model file, skipped if
that file hasn't been produced yet (mirrors the pattern used in
test_explainability.py and test_next_best_action.py).
"""

import uuid

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models import baseline as bl
from src.models import mlflow_tracking as mt


# ---------------------------------------------------------------------------
# Isolated per-test MLflow config + a small, fast synthetic pipeline
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_config(tmp_path) -> dict:
    unique = uuid.uuid4().hex[:8]
    return {
        "tracking": {
            "backend_store_uri": f"sqlite:///{tmp_path / 'mlruns_test.db'}",
            "artifact_location": str(tmp_path / "mlartifacts_test"),
        },
        "experiment": {"name": f"test-experiment-{unique}"},
        "registry": {"model_name": f"test-registry-model-{unique}"},
        "model": {"serialization_format": "pickle"},
        "expected_model": {"model_type": "LogisticRegression", "C": 0.01, "class_weight": "balanced"},
        "random_seed": 42,
    }


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
            "tenure_days": rng.integers(30, 900, n_per_split),
            "some_metric": rng.normal(loc=0.0, scale=1.0, size=n_per_split),
            "churn_next_30_days": [1 if i % 10 == 0 else 0 for i in range(n_per_split)],
        })

    train = make_split(bl.TRAIN_START)
    validation = make_split(bl.VALIDATION_START)
    test = make_split(bl.TEST_START)
    return pd.concat([train, validation, test], ignore_index=True)


@pytest.fixture
def tuned_style_pipeline(synthetic_features: pd.DataFrame):
    """A pipeline built exactly like the real Phase 7 selection
    (C=0.01, class_weight="balanced") but fit on a small synthetic
    dataset -- fast, isolated from the real ~137k-row dataset/model file.
    """
    numeric_columns, categorical_columns = bl.get_feature_columns(synthetic_features)
    feature_columns = numeric_columns + categorical_columns
    splits = bl.split_by_snapshot_date(synthetic_features)

    preprocessor = bl.build_preprocessor(numeric_columns, categorical_columns)
    model = LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000, random_state=bl.RANDOM_SEED)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(splits["train"][feature_columns], splits["train"][bl.TARGET_COLUMN])

    return {"pipeline": pipeline, "splits": splits, "feature_columns": feature_columns}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def test_load_config_real_file_has_required_sections() -> None:
    config = mt.load_config()
    assert {"tracking", "experiment", "registry", "model", "expected_model"} <= set(config.keys())
    assert config["expected_model"] == {"model_type": "LogisticRegression", "C": 0.01, "class_weight": "balanced"}


def test_load_config_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        mt.load_config(path=tmp_path / "missing.yaml")


def test_load_config_rejects_incomplete_config(tmp_path) -> None:
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text("tracking:\n  backend_store_uri: sqlite:///x.db\n")
    with pytest.raises(ValueError):
        mt.load_config(path=incomplete)


# ---------------------------------------------------------------------------
# Experiment creation / loading (idempotent)
# ---------------------------------------------------------------------------

def test_create_mlflow_experiment_creates_new(temp_config: dict) -> None:
    experiment_id = mt.create_mlflow_experiment(config=temp_config)
    assert experiment_id is not None

    import mlflow
    experiment = mlflow.get_experiment(experiment_id)
    assert experiment.name == temp_config["experiment"]["name"]


def test_create_mlflow_experiment_is_idempotent(temp_config: dict) -> None:
    first_id = mt.create_mlflow_experiment(config=temp_config)
    second_id = mt.create_mlflow_experiment(config=temp_config)
    assert first_id == second_id


# ---------------------------------------------------------------------------
# Parameter / metric building blocks
# ---------------------------------------------------------------------------

def test_build_model_parameters_reports_expected_fields(tuned_style_pipeline) -> None:
    parameters = mt.build_model_parameters(tuned_style_pipeline["pipeline"], tuned_style_pipeline["feature_columns"])

    assert parameters["model_type"] == "LogisticRegression"
    assert parameters["C"] == 0.01
    assert parameters["class_weight"] == "balanced"
    assert parameters["solver"] == tuned_style_pipeline["pipeline"].named_steps["model"].solver
    assert parameters["random_seed"] == bl.RANDOM_SEED
    assert parameters["raw_feature_count"] == len(tuned_style_pipeline["feature_columns"])
    assert parameters["transformed_feature_count"] > 0
    assert parameters["train_start"] == str(bl.TRAIN_START)
    assert parameters["validation_start"] == str(bl.VALIDATION_START)


def test_build_validation_and_training_metrics_are_prefixed(tuned_style_pipeline) -> None:
    splits = tuned_style_pipeline["splits"]
    feature_columns = tuned_style_pipeline["feature_columns"]
    pipeline = tuned_style_pipeline["pipeline"]

    train_metrics = mt.build_training_metrics(
        pipeline, splits["train"][feature_columns], splits["train"][bl.TARGET_COLUMN]
    )
    val_metrics = mt.build_validation_metrics(
        pipeline, splits["validation"][feature_columns], splits["validation"][bl.TARGET_COLUMN]
    )

    assert all(key.startswith("train_") for key in train_metrics)
    assert all(key.startswith("val_") for key in val_metrics)
    assert "train_pr_auc" in train_metrics
    assert "val_pr_auc" in val_metrics
    assert 0.0 <= val_metrics["val_pr_auc"] <= 1.0


def test_build_holdout_test_metrics_are_prefixed_test(tuned_style_pipeline) -> None:
    splits = tuned_style_pipeline["splits"]
    feature_columns = tuned_style_pipeline["feature_columns"]
    holdout_metrics = mt.build_holdout_test_metrics(
        tuned_style_pipeline["pipeline"], splits["test"][feature_columns], splits["test"][bl.TARGET_COLUMN]
    )
    assert all(key.startswith("test_") for key in holdout_metrics)
    assert "test_pr_auc" in holdout_metrics


def test_reproducibility_tags_mark_holdout_as_final_only(tuned_style_pipeline) -> None:
    tags = mt.build_reproducibility_tags(tuned_style_pipeline["feature_columns"])
    assert tags["test_metrics_are_final_holdout_only"] == "true"
    assert "never" in tags["test_metrics_note"].lower()
    assert "sklearn_version" in tags
    assert "mlflow_version" in tags


# ---------------------------------------------------------------------------
# Logging a full run: parameters, metrics, artifacts, model
# ---------------------------------------------------------------------------

def test_log_model_run_logs_params_metrics_and_model(temp_config: dict, tuned_style_pipeline, tmp_path) -> None:
    mt.create_mlflow_experiment(config=temp_config)
    pipeline = tuned_style_pipeline["pipeline"]
    feature_columns = tuned_style_pipeline["feature_columns"]
    X_sample = tuned_style_pipeline["splits"]["validation"][feature_columns].iloc[:5]

    parameters = mt.build_model_parameters(pipeline, feature_columns)
    metrics = {"val_pr_auc": 0.5, "train_pr_auc": 0.6}

    result = mt.log_model_run(
        pipeline=pipeline,
        run_name="test-run",
        parameters=parameters,
        metrics=metrics,
        tags={"custom_tag": "yes"},
        artifact_paths=[],
        input_example=X_sample,
        config=temp_config,
    )

    assert result["run_id"]
    assert result["model_uri"]

    import mlflow
    run = mlflow.get_run(result["run_id"])
    assert run.data.params["C"] == "0.01"
    assert run.data.metrics["val_pr_auc"] == pytest.approx(0.5)
    assert run.data.tags["custom_tag"] == "yes"


def test_log_model_run_logs_only_existing_artifact_paths(temp_config: dict, tuned_style_pipeline, tmp_path) -> None:
    mt.create_mlflow_experiment(config=temp_config)
    pipeline = tuned_style_pipeline["pipeline"]

    real_file = tmp_path / "real_report.csv"
    real_file.write_text("a,b\n1,2\n")
    missing_file = tmp_path / "does_not_exist.csv"

    result = mt.log_model_run(
        pipeline=pipeline,
        run_name="artifact-test-run",
        parameters={"C": 0.01},
        metrics={"val_pr_auc": 0.5},
        artifact_paths=[real_file, missing_file],
        config=temp_config,
    )

    assert result["logged_artifact_paths"] == [real_file]

    import mlflow
    client = mlflow.MlflowClient()
    artifacts = client.list_artifacts(result["run_id"], path="reports")
    logged_names = {a.path.split("/")[-1] for a in artifacts}
    assert "real_report.csv" in logged_names
    assert "does_not_exist.csv" not in logged_names


def test_log_artifacts_never_includes_large_recommendation_csv() -> None:
    """Guards the "don't duplicate huge datasets into MLflow" rule: the
    default artifact list must stick to small, aggregated report files.
    """
    default_names = {path.name for path in mt.DEFAULT_ARTIFACT_PATHS}
    assert "customer_recommendations.csv" not in default_names
    assert "recommendation_summary.csv" in default_names


# ---------------------------------------------------------------------------
# Model registration and loading
# ---------------------------------------------------------------------------

def test_register_model_creates_version_linked_to_run(temp_config: dict, tuned_style_pipeline) -> None:
    mt.create_mlflow_experiment(config=temp_config)
    pipeline = tuned_style_pipeline["pipeline"]

    result = mt.log_model_run(
        pipeline=pipeline,
        run_name="register-test-run",
        parameters={"C": 0.01},
        metrics={"val_pr_auc": 0.5},
        artifact_paths=[],
        config=temp_config,
    )
    model_version = mt.register_model(result["model_uri"], config=temp_config)

    assert model_version.name == temp_config["registry"]["model_name"]
    assert model_version.run_id == result["run_id"]
    assert int(model_version.version) == 1


def test_load_registered_model_defaults_to_latest_version(temp_config: dict, tuned_style_pipeline) -> None:
    mt.create_mlflow_experiment(config=temp_config)
    pipeline = tuned_style_pipeline["pipeline"]

    for _ in range(2):
        result = mt.log_model_run(
            pipeline=pipeline, run_name="v-run", parameters={"C": 0.01}, metrics={"val_pr_auc": 0.5},
            artifact_paths=[], config=temp_config,
        )
        mt.register_model(result["model_uri"], config=temp_config)

    loaded = mt.load_registered_model(config=temp_config)
    assert loaded is not None


def test_prediction_equivalence_between_original_and_loaded_model(temp_config: dict, tuned_style_pipeline) -> None:
    mt.create_mlflow_experiment(config=temp_config)
    pipeline = tuned_style_pipeline["pipeline"]
    feature_columns = tuned_style_pipeline["feature_columns"]
    X_val = tuned_style_pipeline["splits"]["validation"][feature_columns]

    result = mt.log_model_run(
        pipeline=pipeline, run_name="predict-test-run", parameters={"C": 0.01}, metrics={"val_pr_auc": 0.5},
        artifact_paths=[], config=temp_config,
    )
    mt.register_model(result["model_uri"], config=temp_config)

    loaded = mt.load_registered_model(config=temp_config)
    original_predictions = pipeline.predict_proba(X_val)[:, 1]
    loaded_predictions = loaded.predict_proba(X_val)[:, 1]

    np.testing.assert_allclose(original_predictions, loaded_predictions)


def test_load_registered_model_by_explicit_version(temp_config: dict, tuned_style_pipeline) -> None:
    mt.create_mlflow_experiment(config=temp_config)
    pipeline = tuned_style_pipeline["pipeline"]
    result = mt.log_model_run(
        pipeline=pipeline, run_name="version-test-run", parameters={"C": 0.01}, metrics={"val_pr_auc": 0.5},
        artifact_paths=[], config=temp_config,
    )
    model_version = mt.register_model(result["model_uri"], config=temp_config)

    loaded = mt.load_registered_model(config=temp_config, version=model_version.version)
    assert loaded is not None


def test_load_registered_model_raises_when_no_versions_exist(temp_config: dict) -> None:
    mt.create_mlflow_experiment(config=temp_config)
    with pytest.raises(Exception):
        mt.load_registered_model(config=temp_config)


# ---------------------------------------------------------------------------
# Model configuration validation (compatibility check)
# ---------------------------------------------------------------------------

def test_validate_model_compatibility_accepts_matching_model(tuned_style_pipeline) -> None:
    expected = {"model_type": "LogisticRegression", "C": 0.01, "class_weight": "balanced"}
    mt.validate_model_compatibility(tuned_style_pipeline["pipeline"], expected)  # must not raise


@pytest.mark.parametrize(
    "bad_expected",
    [
        {"model_type": "RandomForestClassifier", "C": 0.01, "class_weight": "balanced"},
        {"model_type": "LogisticRegression", "C": 1.0, "class_weight": "balanced"},
        {"model_type": "LogisticRegression", "C": 0.01, "class_weight": None},
    ],
)
def test_validate_model_compatibility_rejects_mismatch(tuned_style_pipeline, bad_expected: dict) -> None:
    with pytest.raises(ValueError):
        mt.validate_model_compatibility(tuned_style_pipeline["pipeline"], bad_expected)


def test_validate_model_compatibility_rejects_wrong_feature_count(tuned_style_pipeline) -> None:
    expected = {"model_type": "LogisticRegression", "C": 0.01, "class_weight": "balanced"}
    actual_count = tuned_style_pipeline["pipeline"].named_steps["model"].coef_.shape[1]
    with pytest.raises(ValueError):
        mt.validate_model_compatibility(tuned_style_pipeline["pipeline"], expected, expected_feature_count=actual_count + 1)


def test_validate_model_compatibility_rejects_non_pipeline() -> None:
    with pytest.raises(ValueError):
        mt.validate_model_compatibility(LogisticRegression(), {"model_type": "LogisticRegression", "C": 0.01, "class_weight": "balanced"})


def test_load_registered_model_rejects_mismatched_model(temp_config: dict, synthetic_features: pd.DataFrame) -> None:
    """A model registered with the WRONG hyperparameters must be rejected
    by load_registered_model's compatibility check, not silently returned.
    """
    mt.create_mlflow_experiment(config=temp_config)

    numeric_columns, categorical_columns = bl.get_feature_columns(synthetic_features)
    feature_columns = numeric_columns + categorical_columns
    splits = bl.split_by_snapshot_date(synthetic_features)

    wrong_pipeline = Pipeline(steps=[
        ("preprocessor", bl.build_preprocessor(numeric_columns, categorical_columns)),
        ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=bl.RANDOM_SEED)),
    ])
    wrong_pipeline.fit(splits["train"][feature_columns], splits["train"][bl.TARGET_COLUMN])

    result = mt.log_model_run(
        pipeline=wrong_pipeline, run_name="wrong-model-run", parameters={"C": 1.0}, metrics={"val_pr_auc": 0.1},
        artifact_paths=[], config=temp_config,
    )
    mt.register_model(result["model_uri"], config=temp_config)

    with pytest.raises(ValueError):
        mt.load_registered_model(config=temp_config)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_build_model_parameters_deterministic_across_calls(tuned_style_pipeline) -> None:
    first = mt.build_model_parameters(tuned_style_pipeline["pipeline"], tuned_style_pipeline["feature_columns"])
    second = mt.build_model_parameters(tuned_style_pipeline["pipeline"], tuned_style_pipeline["feature_columns"])

    non_timestamp_keys = [k for k in first if k != "training_logged_at"]
    for key in non_timestamp_keys:
        assert first[key] == second[key]


def test_build_validation_metrics_deterministic_across_calls(tuned_style_pipeline) -> None:
    splits = tuned_style_pipeline["splits"]
    feature_columns = tuned_style_pipeline["feature_columns"]
    pipeline = tuned_style_pipeline["pipeline"]

    first = mt.build_validation_metrics(pipeline, splits["validation"][feature_columns], splits["validation"][bl.TARGET_COLUMN])
    second = mt.build_validation_metrics(pipeline, splits["validation"][feature_columns], splits["validation"][bl.TARGET_COLUMN])
    assert first == second


def test_get_mlflow_ui_command_is_deterministic_and_uses_configured_backend(temp_config: dict) -> None:
    command = mt.get_mlflow_ui_command(config=temp_config)
    assert command.startswith("mlflow ui --backend-store-uri sqlite:///")
    assert command == mt.get_mlflow_ui_command(config=temp_config)


# ---------------------------------------------------------------------------
# Real Phase 7 artifact integration (skipped if Phase 7 hasn't run yet)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_config(tmp_path_factory) -> dict:
    tmp_path = tmp_path_factory.mktemp("mlflow_real")
    unique = uuid.uuid4().hex[:8]
    base_config = mt.load_config()
    return {
        **base_config,
        "tracking": {
            "backend_store_uri": f"sqlite:///{tmp_path / 'mlruns_real.db'}",
            "artifact_location": str(tmp_path / "mlartifacts_real"),
        },
        "experiment": {"name": f"real-test-experiment-{unique}"},
        "registry": {"model_name": f"real-test-model-{unique}"},
    }


@pytest.fixture(scope="module")
def real_run_result(real_config: dict):
    from src.models.explainability import TUNED_MODEL_PATH

    if not TUNED_MODEL_PATH.exists():
        pytest.skip(f"{TUNED_MODEL_PATH} not found -- run Phase 7 first")

    return mt.run_phase10_tracking(config=real_config)


def test_real_run_logs_expected_phase7_hyperparameters(real_run_result: dict) -> None:
    assert real_run_result["parameters"]["C"] == 0.01
    assert real_run_result["parameters"]["class_weight"] == "balanced"
    assert real_run_result["parameters"]["model_type"] == "LogisticRegression"


def test_real_run_validation_metrics_present(real_run_result: dict) -> None:
    assert "val_pr_auc" in real_run_result["validation_metrics"]
    assert 0.0 <= real_run_result["validation_metrics"]["val_pr_auc"] <= 1.0


def test_real_run_holdout_metrics_are_separate_from_validation(real_run_result: dict) -> None:
    assert set(real_run_result["holdout_metrics"]) & set(real_run_result["validation_metrics"]) == set()
    assert all(k.startswith("test_") for k in real_run_result["holdout_metrics"])


def test_real_model_registered_and_loadable(real_run_result: dict, real_config: dict) -> None:
    expected_feature_count = real_run_result["pipeline"].named_steps["model"].coef_.shape[1]
    loaded = mt.load_registered_model(config=real_config, expected_feature_count=expected_feature_count)
    assert loaded is not None


def test_real_loaded_model_predictions_match_original(real_run_result: dict, real_config: dict) -> None:
    pipeline = real_run_result["pipeline"]
    feature_columns = real_run_result["feature_columns"]
    X_val = real_run_result["splits"]["validation"][feature_columns].iloc[:100]

    loaded = mt.load_registered_model(config=real_config)
    np.testing.assert_allclose(pipeline.predict_proba(X_val)[:, 1], loaded.predict_proba(X_val)[:, 1])


def test_real_phase7_artifact_not_modified(real_run_result: dict, tmp_path_factory) -> None:
    """The frozen Phase 7 model file on disk must be byte-for-byte
    unchanged after the entire Phase 10 tracking/registration flow.
    `real_run_result` (module-scoped) has already exercised the full flow
    once; this re-runs it a second time, against its own fresh temp
    tracking store, and diffs the artifact bytes around both runs.
    """
    from src.models.explainability import TUNED_MODEL_PATH

    before_bytes = TUNED_MODEL_PATH.read_bytes()

    tmp_path = tmp_path_factory.mktemp("mlflow_mutation_check")
    unique = uuid.uuid4().hex[:8]
    second_config = {
        **mt.load_config(),
        "tracking": {
            "backend_store_uri": f"sqlite:///{tmp_path / 'mlruns.db'}",
            "artifact_location": str(tmp_path / "mlartifacts"),
        },
        "experiment": {"name": f"mutation-check-{unique}"},
        "registry": {"model_name": f"mutation-check-model-{unique}"},
    }
    mt.run_phase10_tracking(config=second_config)

    after_bytes = TUNED_MODEL_PATH.read_bytes()
    assert before_bytes == after_bytes
