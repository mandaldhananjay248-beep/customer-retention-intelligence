"""Phase 15: provision models/tuned_logistic_regression.joblib (the Phase 7
artifact path Phases 8-11's tests load) on a machine that doesn't have it,
such as a CI runner, from the committed packaged registered model.

This is not retraining: the packaged model.pkl is the Phase 7 pipeline
exactly as MLflow registered it (verified SHA-256), so the joblib written
here holds the same fitted pipeline -- the same coefficients and
predictions. An existing local Phase 7 artifact is never overwritten.

    python -m src.models.provision_tuned_model
"""

from __future__ import annotations

import sys

import joblib

from src.models import mlflow_tracking as mt
from src.models.explainability import TUNED_MODEL_PATH


def provision() -> bool:
    """Write TUNED_MODEL_PATH if it's missing. Returns True if written."""
    if TUNED_MODEL_PATH.exists():
        print(f"{TUNED_MODEL_PATH.name} already exists -- left untouched.")
        return False
    pipeline = mt.load_packaged_model()
    TUNED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, TUNED_MODEL_PATH)
    print(f"Wrote {TUNED_MODEL_PATH.name} from the packaged registered model.")
    return True


if __name__ == "__main__":
    provision()
    sys.exit(0)
