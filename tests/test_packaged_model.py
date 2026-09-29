"""Tests for Phase 15's portable packaged registered model
(models/registered/, configs/mlflow_config.yaml's packaged_model section)
and the deterministic feature-dataset regeneration it is deployed with.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn
import yaml

from src.data import regenerate_feature_dataset as regen
from src.models import baseline as bl
from src.models import mlflow_tracking as mt
from src.models.explainability import TUNED_MODEL_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict:
    return mt.load_config()


@pytest.fixture(scope="module")
def feature_dataset() -> pd.DataFrame:
    return bl.load_feature_dataset()


def test_packaged_model_config_section_is_complete(config: dict) -> None:
    packaged = config["packaged_model"]
    assert set(packaged) >= {"path", "version", "model_id", "run_id", "model_sha256"}
    assert not Path(packaged["path"]).is_absolute()


def test_default_packaged_model_dir_is_inside_the_project(config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_ARTIFACT_ROOT", raising=False)
    model_dir = mt.resolve_packaged_model_dir(config)
    assert model_dir == (PROJECT_ROOT / "models" / "registered" / config["packaged_model"]["path"]).resolve()
    assert (model_dir / "model.pkl").exists()
    assert (model_dir / "MLmodel").exists()


def test_committed_model_pkl_matches_configured_sha256(config: dict) -> None:
    model_file = mt.resolve_packaged_model_dir(config) / "model.pkl"
    assert hashlib.sha256(model_file.read_bytes()).hexdigest() == config["packaged_model"]["model_sha256"]


def test_mlmodel_metadata_matches_config_and_pinned_sklearn(config: dict) -> None:
    mlmodel = yaml.safe_load((mt.resolve_packaged_model_dir(config) / "MLmodel").read_text())
    assert mlmodel["model_id"] == config["packaged_model"]["model_id"]
    assert mlmodel["run_id"] == config["packaged_model"]["run_id"]
    # Unpickling a scikit-learn model with a different version is unsupported.
    assert mlmodel["flavors"]["sklearn"]["sklearn_version"] == sklearn.__version__


def test_load_packaged_model_returns_the_validated_phase7_pipeline(config: dict, feature_dataset: pd.DataFrame) -> None:
    numeric, categorical = bl.get_feature_columns(feature_dataset)
    pipeline = mt.load_packaged_model(config)
    model = pipeline.named_steps["model"]
    assert type(model).__name__ == "LogisticRegression"
    assert model.C == 0.01
    assert model.class_weight == "balanced"
    assert list(pipeline.named_steps["preprocessor"].transformers_[0][2]) == numeric
    assert list(pipeline.named_steps["preprocessor"].transformers_[1][2]) == categorical


def test_packaged_model_predictions_equal_the_phase7_model(config: dict, feature_dataset: pd.DataFrame) -> None:
    """Same fitted pipeline as the Phase 7 artifact -- identical predictions."""
    import joblib

    numeric, categorical = bl.get_feature_columns(feature_dataset)
    X = feature_dataset[numeric + categorical]
    packaged = mt.load_packaged_model(config).predict_proba(X)[:, 1]
    phase7 = joblib.load(TUNED_MODEL_PATH).predict_proba(X)[:, 1]
    assert np.array_equal(packaged, phase7)


def test_model_artifact_root_env_var_relocates_the_model(config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = mt.resolve_packaged_model_dir(config)
    relocated = tmp_path / "artifacts" / config["packaged_model"]["path"]
    shutil.copytree(source, relocated)
    monkeypatch.setenv("MODEL_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    assert mt.resolve_packaged_model_dir(config) == relocated.resolve()
    assert mt.load_packaged_model(config).named_steps["model"].C == 0.01


def test_load_packaged_model_refuses_a_hash_mismatch(config: dict) -> None:
    tampered = {**config, "packaged_model": {**config["packaged_model"], "model_sha256": "0" * 64}}
    with pytest.raises(ValueError, match="SHA-256"):
        mt.load_packaged_model(tampered)


def test_load_packaged_model_fails_clearly_when_missing(config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_ARTIFACT_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Packaged model not found"):
        mt.load_packaged_model(config)


def test_api_model_service_needs_no_mlflow_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The API must start without the machine-specific local registry
    (mlruns.db / mlartifacts): any registry access here would raise.
    """
    from api.services import ModelService

    def _no_registry(*args, **kwargs):
        raise AssertionError("the API must not touch the MLflow registry at startup")

    monkeypatch.setattr(mt, "MlflowClient", _no_registry)
    monkeypatch.setattr(mt, "load_registered_model", _no_registry)
    service = ModelService()
    service.load()
    assert service.is_ready
    assert service.model_version == mt.load_config()["packaged_model"]["version"]


def test_feature_dataset_matches_the_registered_fingerprint(feature_dataset: pd.DataFrame) -> None:
    """The dataset in use (regenerated in CI/Docker) is exactly the one the
    model was registered alongside.
    """
    assert feature_dataset.shape == regen.FEATURE_DATASET_SHAPE
    assert regen.fingerprint(feature_dataset) == regen.FEATURE_DATASET_FINGERPRINT


def test_fingerprint_detects_a_changed_value(feature_dataset: pd.DataFrame) -> None:
    changed = feature_dataset.head(100).copy()
    original = regen.fingerprint(changed)
    changed.iloc[0, changed.columns.get_loc("age")] += 1
    assert regen.fingerprint(changed) != original
