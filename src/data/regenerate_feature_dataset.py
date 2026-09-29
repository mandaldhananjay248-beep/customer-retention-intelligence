"""Phase 15: rebuild data/processed/customer_features.parquet from the
committed, seeded pipeline -- no dataset is committed or downloaded.

Runs exactly the existing Phase 1-4 code, in order:

    src.data.generate_data (seeded synthetic raw CSVs)
        -> src.data.load_data (processed Parquet tables)
        -> src.features.feature_engineering.build_feature_dataset

and then verifies the result against FEATURE_DATASET_FINGERPRINT, the
content hash of the dataset the packaged Phase 10 model was registered
alongside. Only data is regenerated here -- nothing is fitted, tuned, or
retrained. Used by both Dockerfiles at image build time, by CI, and by the
Streamlit Community Cloud entry point (streamlit_app.py, via
ensure_feature_dataset) on first start.

    python -m src.data.regenerate_feature_dataset [--prune]

--prune deletes the intermediate raw CSVs and processed tables afterwards
(the API and dashboard containers only read customer_features.parquet).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "customer_features.parquet"

# SHA-256 over pandas' per-row content hashes (index excluded) of the
# 136,701 x 134 customer_features.parquet produced by Phases 1-4 with
# configs/data_config.yaml's random_seed.
FEATURE_DATASET_FINGERPRINT = "cfaf6e2cd2a3c58a3980cd59a910c00156172f7d5e705e609bb2d57c2df34a5c"
FEATURE_DATASET_SHAPE = (136701, 134)


def fingerprint(features: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(features, index=False).values.tobytes()).hexdigest()


def regenerate(prune: bool = False) -> Path:
    # generate_data resolves its config and output directory relative to
    # the working directory at import time.
    os.chdir(PROJECT_ROOT)

    from src.data import generate_data
    from src.data.load_data import PROCESSED_DATA_DIR, RAW_DATA_DIR, load_all_datasets, save_processed_datasets
    from src.features import feature_engineering as fe

    generate_data.main()
    datasets = load_all_datasets()
    save_processed_datasets(datasets)
    features = fe.build_feature_dataset(
        datasets["customers"],
        datasets["transactions"],
        datasets["usage"],
        datasets["support_tickets"],
        datasets["churn_events"],
    )
    output_path = fe.save_feature_dataset(features)

    actual = fingerprint(pd.read_parquet(output_path))
    if features.shape != FEATURE_DATASET_SHAPE or actual != FEATURE_DATASET_FINGERPRINT:
        raise RuntimeError(
            f"Regenerated {output_path.name} does not match the dataset the model was registered with "
            f"(shape {features.shape}, fingerprint {actual})."
        )

    if prune:
        for path in list(RAW_DATA_DIR.glob("*.csv")) + list(PROCESSED_DATA_DIR.glob("*.parquet")):
            if path != output_path:
                path.unlink()

    print(f"Regenerated {output_path.relative_to(PROJECT_ROOT)}: shape {features.shape}, fingerprint verified.")
    return output_path


def ensure_feature_dataset(path: Path = FEATURE_DATASET_PATH) -> Path:
    """Regenerate (fingerprint verified) only if the dataset is missing --
    e.g. on a fresh Streamlit Community Cloud checkout, which has no build
    step. An existing file is used as-is, exactly as the API and dashboard
    already do. Never prunes, so a developer's local data/raw files are
    never deleted.

    Runs in a child process: generation peaks at ~1.2 GB, and memory freed
    inside a long-lived process isn't reliably returned to the OS, so running
    it in-process would stack on top of the app's own ~1.1 GB (measured).
    """
    if path.exists():
        return path
    subprocess.run([sys.executable, "-m", "src.data.regenerate_feature_dataset"], cwd=PROJECT_ROOT, check=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prune", action="store_true", help="delete intermediate raw/processed files afterwards")
    args = parser.parse_args(argv)
    regenerate(prune=args.prune)
    return 0


if __name__ == "__main__":
    sys.exit(main())
