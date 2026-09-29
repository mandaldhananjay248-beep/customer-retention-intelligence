"""Phase 10 MLflow experiment tracking and model registry for the frozen
Phase 7 Logistic Regression (C=0.01, class_weight="balanced").

This module never fits, retunes, or otherwise modifies that model -- it
loads the already-fitted pipeline exactly the way Phase 8/9 do (via
src.models.explainability.load_tuned_pipeline, which already refuses to
load anything whose hyperparameters don't match the Phase 7 selection)
and only logs/registers/reloads it:

    data/features -> model config -> training (Phase 7, already done)
        -> metrics -> model artifact -> MLflow tracking
        -> MLflow Model Registry -> versioned, reloadable model

Tracking backend: a local SQLite database (configs/mlflow_config.yaml's
tracking.backend_store_uri) -- the Model Registry requires a database
backend, and SQLite needs no server or cloud infrastructure. Both the
database file and the artifact store it points to are generated locally,
excluded from version control, and fully reproducible from this module
plus the already-tracked Phase 7 artifact and reports.

Validation metrics remain the ones Phase 7 already used to select this
model. Any test-split metrics logged here are a separate, clearly-labeled
FINAL HOLDOUT evaluation (tagged accordingly) -- they are logged for the
record, never used to choose or tune anything.
"""

from __future__ import annotations

import hashlib
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import yaml
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from src.models.baseline import (
    PROJECT_ROOT,
    REPORTS_DIR,
    TARGET_COLUMN,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    CLASSIFICATION_THRESHOLD,
    MODELS_DIR,
    evaluate_model,
    get_feature_columns,
    load_feature_dataset,
    split_by_snapshot_date,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "mlflow_config.yaml"

# Where packaged registered-model versions live unless MODEL_ARTIFACT_ROOT
# overrides it (see configs/mlflow_config.yaml's packaged_model section).
DEFAULT_MODEL_ARTIFACT_ROOT = MODELS_DIR / "registered"

# The report artifacts that are actually useful for reproducing/auditing
# this run -- small, aggregated tables. Deliberately excludes
# reports/customer_recommendations.csv (Phase 9's full-population,
# multi-tens-of-MB inference output) and the SHAP figures: neither is
# needed to reproduce or audit the *model*, and MLflow's tracking store
# should not become a second copy of large generated datasets.
DEFAULT_ARTIFACT_PATHS: tuple[Path, ...] = (
    REPORTS_DIR / "baseline_results.csv",
    REPORTS_DIR / "model_comparison_results.csv",
    REPORTS_DIR / "tuning_results.csv",
    REPORTS_DIR / "shap_feature_importance.csv",
    REPORTS_DIR / "recommendation_summary.csv",
)


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load the Phase 10 MLflow configuration. Raises loudly if missing or
    incomplete -- this module has no hard-coded fallback tracking URI,
    experiment name, or expected-model spec.
    """
    if not path.exists():
        raise FileNotFoundError(f"MLflow config not found: {path}")
    with open(path) as config_file:
        config = yaml.safe_load(config_file)

    required_sections = {"tracking", "experiment", "registry", "model", "expected_model"}
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"MLflow config at {path} is missing required section(s): {missing}")

    return config


def _resolve_sqlite_uri(uri: str) -> str:
    """Resolve a `sqlite:///relative/path` URI against PROJECT_ROOT so
    tracking behaves identically regardless of the caller's working
    directory. An already-absolute path (as tests pass via tmp_path)
    passes through unchanged -- joining an absolute path onto PROJECT_ROOT
    with pathlib simply replaces it.
    """
    if not uri.startswith("sqlite:///"):
        return uri
    relative_or_absolute_path = uri[len("sqlite:///"):]
    resolved_path = (PROJECT_ROOT / relative_or_absolute_path).resolve()
    return f"sqlite:///{resolved_path}"


def configure_tracking(config: dict | None = None) -> str:
    """Point MLflow at the local SQLite-backed tracking store. Returns the
    resolved tracking URI actually set.
    """
    config = config or load_config()
    tracking_uri = _resolve_sqlite_uri(config["tracking"]["backend_store_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def create_mlflow_experiment(name: str | None = None, config: dict | None = None) -> str:
    """Create the experiment if it doesn't exist yet, or load (reuse) it if
    it does -- idempotent, so re-running this notebook/module never
    duplicates the experiment. Returns the experiment_id and leaves it set
    as the active experiment for subsequent runs.
    """
    config = config or load_config()
    configure_tracking(config)
    name = name or config["experiment"]["name"]

    existing = mlflow.get_experiment_by_name(name)
    if existing is not None and existing.lifecycle_stage == "active":
        experiment_id = existing.experiment_id
    else:
        artifact_location = str((PROJECT_ROOT / config["tracking"]["artifact_location"]).resolve())
        experiment_id = mlflow.create_experiment(name, artifact_location=artifact_location)

    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def log_model_parameters(parameters: dict) -> None:
    """Thin wrapper around mlflow.log_params -- kept as its own function so
    callers (this module, tests, a future retraining script) share one
    logging code path instead of calling the mlflow API directly.
    """
    mlflow.log_params(parameters)


def log_model_metrics(metrics: dict) -> None:
    """Thin wrapper around mlflow.log_metrics. Metric dict keys are
    expected to already carry a split prefix (train_/val_/test_) from
    build_training_metrics/build_validation_metrics/build_holdout_test_metrics
    so nothing here needs to guess which split a bare "pr_auc" belongs to.
    """
    mlflow.log_metrics(metrics)


def log_artifacts(artifact_paths: list[Path] = DEFAULT_ARTIFACT_PATHS, artifact_subdir: str = "reports") -> list[Path]:
    """Log each existing path in artifact_paths under `artifact_subdir` in
    the run's artifact store. Silently skips any path that doesn't exist
    (e.g. a report from a phase that hasn't been run yet) rather than
    failing the whole run over an optional file. Returns the paths that
    were actually logged.
    """
    logged = []
    for path in artifact_paths:
        path = Path(path)
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path=artifact_subdir)
            logged.append(path)
    return logged


def build_model_parameters(pipeline: Pipeline, feature_columns: list[str]) -> dict:
    """Everything Phase 10's spec asks to track under "Parameters"."""
    model = pipeline.named_steps["model"]

    return {
        "model_type": type(model).__name__,
        "C": model.C,
        "class_weight": model.class_weight,
        "solver": model.solver,
        "random_seed": model.random_state,
        "raw_feature_count": len(feature_columns),
        "transformed_feature_count": int(model.coef_.shape[1]),
        "train_start": str(TRAIN_START),
        "train_end": str(TRAIN_END),
        "validation_start": str(VALIDATION_START),
        "validation_end": str(VALIDATION_END),
        "training_logged_at": datetime.now(timezone.utc).isoformat(),
    }


def build_training_metrics(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    metrics = evaluate_model(pipeline, X_train, y_train, threshold=CLASSIFICATION_THRESHOLD)
    return {f"train_{key}": value for key, value in metrics.items()}


def build_validation_metrics(pipeline: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """The metrics that actually mattered for Phase 7's model selection
    (validation PR-AUC above all). Logged here purely for the record --
    Phase 7's selection already happened; nothing is re-decided.
    """
    metrics = evaluate_model(pipeline, X_val, y_val, threshold=CLASSIFICATION_THRESHOLD)
    return {f"val_{key}": value for key, value in metrics.items()}


def build_holdout_test_metrics(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """A FINAL HOLDOUT evaluation only. Every key is prefixed "test_" and
    run_phase10_tracking additionally tags the run to make this explicit --
    these numbers must never be used to choose or tune a model; Phase 7's
    selection was already made using validation PR-AUC alone.
    """
    metrics = evaluate_model(pipeline, X_test, y_test, threshold=CLASSIFICATION_THRESHOLD)
    return {f"test_{key}": value for key, value in metrics.items()}


def build_reproducibility_tags(feature_columns: list[str]) -> dict:
    """Everything needed to reproduce this model besides the parameters
    already logged by build_model_parameters: package/environment info and
    provenance back to the exact Phase 7 artifact this run wraps.
    """
    return {
        "source_model_path": "models/tuned_logistic_regression.joblib",
        "source_phase": "phase7_hyperparameter_tuning",
        "test_metrics_are_final_holdout_only": "true",
        "test_metrics_note": (
            "Never used for model selection -- Phase 7 selected this model using "
            "validation PR-AUC only, before these test metrics were ever computed here."
        ),
        "n_raw_features": str(len(feature_columns)),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "mlflow_version": mlflow.__version__,
    }


def log_model_run(
    pipeline: Pipeline,
    run_name: str,
    parameters: dict,
    metrics: dict,
    tags: dict | None = None,
    artifact_paths: list[Path] = DEFAULT_ARTIFACT_PATHS,
    signature=None,
    input_example: pd.DataFrame | None = None,
    config: dict | None = None,
) -> dict:
    """Log one complete run: parameters, metrics, reproducibility tags,
    report artifacts, and the model itself (sklearn flavor). Requires an
    experiment to already be active (create_mlflow_experiment). Returns
    {run_id, model_uri, artifact_uri, logged_artifact_paths}.
    """
    config = config or load_config()

    with mlflow.start_run(run_name=run_name) as run:
        log_model_parameters(parameters)
        log_model_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)

        logged_artifact_paths = log_artifacts(artifact_paths)

        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
            serialization_format=config["model"]["serialization_format"],
        )
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri

    return {
        "run_id": run_id,
        "model_uri": model_info.model_uri,
        "artifact_uri": artifact_uri,
        "logged_artifact_paths": logged_artifact_paths,
    }


def register_model(model_uri: str, model_name: str | None = None, config: dict | None = None):
    """Register a logged model (identified by `model_uri`, as returned in
    log_model_run's result dict) in the MLflow Model Registry. Returns the
    ModelVersion, which carries both the new registry version number and
    the original run_id the model came from.

    `model_uri` should be the URI mlflow.sklearn.log_model itself returned
    (e.g. "models:/m-<id>"), not a manually reconstructed "runs:/<run_id>/model"
    -- MLflow 3.x logs models as their own first-class entities rather than
    plain run artifacts, so reconstructing the older runs:/ form works but
    triggers a fallback lookup warning.
    """
    config = config or load_config()
    model_name = model_name or config["registry"]["model_name"]
    return mlflow.register_model(model_uri, model_name)


def validate_model_compatibility(pipeline, expected_config: dict, expected_feature_count: int | None = None) -> None:
    """Refuse to treat `pipeline` as the expected project model unless its
    type and Phase 7 hyperparameters match exactly. Raises ValueError with
    a specific reason rather than silently returning a mismatched model --
    this is the same defensive pattern Phase 8's load_tuned_pipeline uses.
    """
    if not hasattr(pipeline, "named_steps") or "model" not in pipeline.named_steps:
        raise ValueError(f"Loaded object is not a scikit-learn Pipeline with a 'model' step: {pipeline!r}")

    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise ValueError(f"Loaded model is not a fitted estimator: {model!r}")

    expected_type = expected_config["model_type"]
    if type(model).__name__ != expected_type:
        raise ValueError(f"Loaded model type {type(model).__name__!r} does not match expected {expected_type!r}")

    if model.C != expected_config["C"]:
        raise ValueError(f"Loaded model C={model.C!r} does not match expected C={expected_config['C']!r}")

    if model.class_weight != expected_config["class_weight"]:
        raise ValueError(
            f"Loaded model class_weight={model.class_weight!r} does not match "
            f"expected class_weight={expected_config['class_weight']!r}"
        )

    if expected_feature_count is not None:
        actual_feature_count = model.coef_.shape[1]
        if actual_feature_count != expected_feature_count:
            raise ValueError(
                f"Loaded model has {actual_feature_count} transformed features, "
                f"expected {expected_feature_count}."
            )


def load_registered_model(
    model_name: str | None = None,
    version: int | str | None = None,
    alias: str | None = None,
    config: dict | None = None,
    expected_feature_count: int | None = None,
) -> Pipeline:
    """Load a model from the MLflow Model Registry for future use (e.g. a
    FastAPI service). Defaults to the latest registered version if neither
    `version` nor `alias` is given. Validates the loaded pipeline against
    configs/mlflow_config.yaml's expected_model section (and, if given,
    expected_feature_count) before returning it -- fails clearly rather
    than silently handing back an incompatible model.
    """
    config = config or load_config()
    configure_tracking(config)
    model_name = model_name or config["registry"]["model_name"]

    if alias is not None:
        model_uri = f"models:/{model_name}@{alias}"
    elif version is not None:
        model_uri = f"models:/{model_name}/{version}"
    else:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No registered versions found for model '{model_name}'")
        latest = max(versions, key=lambda v: int(v.version))
        model_uri = f"models:/{model_name}/{latest.version}"

    pipeline = mlflow.sklearn.load_model(model_uri)
    validate_model_compatibility(pipeline, config["expected_model"], expected_feature_count=expected_feature_count)
    return pipeline


def resolve_packaged_model_dir(config: dict | None = None) -> Path:
    """Directory of the packaged registered model (configs/mlflow_config.yaml's
    packaged_model.path), resolved against MODEL_ARTIFACT_ROOT if set,
    otherwise against the project's models/registered/ -- never an absolute
    developer-machine path.
    """
    config = config or load_config()
    artifact_root = Path(os.environ.get("MODEL_ARTIFACT_ROOT") or DEFAULT_MODEL_ARTIFACT_ROOT)
    return (artifact_root / config["packaged_model"]["path"]).resolve()


def load_packaged_model(config: dict | None = None, expected_feature_count: int | None = None) -> Pipeline:
    """Load the packaged copy of the registered model version (see
    configs/mlflow_config.yaml's packaged_model section) with the same MLflow
    sklearn loader load_registered_model uses, after verifying model.pkl's
    SHA-256, and with the same compatibility validation. Portable: needs no
    tracking database and no machine-specific artifact location.
    """
    config = config or load_config()
    model_dir = resolve_packaged_model_dir(config)
    model_file = model_dir / "model.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"Packaged model not found: {model_file}")

    expected_sha256 = config["packaged_model"]["model_sha256"]
    actual_sha256 = hashlib.sha256(model_file.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Packaged model {model_file} SHA-256 {actual_sha256} does not match expected {expected_sha256} "
            "-- refusing to load it."
        )

    pipeline = mlflow.sklearn.load_model(str(model_dir))
    validate_model_compatibility(pipeline, config["expected_model"], expected_feature_count=expected_feature_count)
    return pipeline


def get_mlflow_ui_command(config: dict | None = None) -> str:
    """The exact local command to inspect this project's experiments/runs
    -- no cloud server required.
    """
    config = config or load_config()
    tracking_uri = _resolve_sqlite_uri(config["tracking"]["backend_store_uri"])
    return f"mlflow ui --backend-store-uri {tracking_uri}"


def run_phase10_tracking(config: dict | None = None) -> dict:
    """End-to-end Phase 10: load the frozen Phase 7 model (never refit),
    compute train/validation metrics plus a clearly-labeled final holdout
    test evaluation, log everything to MLflow, and register the model.

    Returns a dict with experiment_id, run_id, model_version, parameters,
    metrics, holdout_metrics, tags, feature_columns, pipeline, and splits
    -- everything the notebook needs to demonstrate the rest of Phase 10
    (loading the registered model back and checking prediction equality)
    without recomputing anything.
    """
    from src.models.explainability import load_tuned_pipeline

    config = config or load_config()
    experiment_id = create_mlflow_experiment(config=config)

    pipeline = load_tuned_pipeline()  # frozen Phase 7 artifact; never fit here

    features = load_feature_dataset()
    splits = split_by_snapshot_date(features)
    numeric_columns, categorical_columns = get_feature_columns(features)
    feature_columns = numeric_columns + categorical_columns

    X_train, y_train = splits["train"][feature_columns], splits["train"][TARGET_COLUMN]
    X_val, y_val = splits["validation"][feature_columns], splits["validation"][TARGET_COLUMN]
    X_test, y_test = splits["test"][feature_columns], splits["test"][TARGET_COLUMN]

    parameters = build_model_parameters(pipeline, feature_columns)
    training_metrics = build_training_metrics(pipeline, X_train, y_train)
    validation_metrics = build_validation_metrics(pipeline, X_val, y_val)
    holdout_metrics = build_holdout_test_metrics(pipeline, X_test, y_test)
    tags = build_reproducibility_tags(feature_columns)

    input_example = X_val.iloc[:5]
    signature = infer_signature(input_example, pipeline.predict_proba(input_example)[:, 1])

    run_result = log_model_run(
        pipeline=pipeline,
        run_name="phase7-logistic-regression-frozen",
        parameters=parameters,
        metrics={**training_metrics, **validation_metrics, **holdout_metrics},
        tags=tags,
        signature=signature,
        input_example=input_example,
        config=config,
    )

    model_version = register_model(run_result["model_uri"], config=config)

    return {
        "experiment_id": experiment_id,
        "run_id": run_result["run_id"],
        "model_uri": run_result["model_uri"],
        "logged_artifact_paths": run_result["logged_artifact_paths"],
        "model_version": model_version,
        "parameters": parameters,
        "training_metrics": training_metrics,
        "validation_metrics": validation_metrics,
        "holdout_metrics": holdout_metrics,
        "tags": tags,
        "feature_columns": feature_columns,
        "pipeline": pipeline,
        "splits": splits,
    }


if __name__ == "__main__":
    result = run_phase10_tracking()

    print(f"Experiment ID: {result['experiment_id']}")
    print(f"Run ID: {result['run_id']}")
    print(f"Model URI: {result['model_uri']}")
    print(f"Registered model: {result['model_version'].name} v{result['model_version'].version}")
    print(f"\nValidation metrics: {result['validation_metrics']}")
    print(f"\nFinal holdout (test) metrics: {result['holdout_metrics']}")
    print(f"\nInspect with: {get_mlflow_ui_command()}")
