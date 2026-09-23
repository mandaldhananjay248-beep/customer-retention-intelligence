"""Reusable loaders for the Customer Retention Intelligence raw datasets.

All functions return pandas DataFrames with date columns already parsed as
datetimes. Paths are resolved relative to the project root (not the current
working directory), so these functions work the same way whether they are
called from a notebook, a script, or a test.
"""

import importlib.util
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_csv(filename: str, date_columns: list[str] | None = None) -> pd.DataFrame:
    """Load a CSV file from data/raw, parsing any given date columns.

    Raises FileNotFoundError if the file does not exist, and ValueError if
    it is empty, instead of silently returning an empty DataFrame.
    """
    file_path = RAW_DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Expected raw data file not found: {file_path}. "
            "Run src/data/generate_data.py to generate the raw datasets first."
        )

    dataframe = pd.read_csv(file_path, parse_dates=date_columns)

    if dataframe.empty:
        raise ValueError(f"Loaded file is unexpectedly empty: {file_path}")

    return dataframe


def load_customers() -> pd.DataFrame:
    """Load customers.csv with signup_date parsed as a date."""
    return load_csv("customers.csv", date_columns=["signup_date"])


def load_transactions() -> pd.DataFrame:
    """Load transactions.csv with transaction_date parsed as a date."""
    return load_csv("transactions.csv", date_columns=["transaction_date"])


def load_usage() -> pd.DataFrame:
    """Load usage.csv with usage_date parsed as a date."""
    return load_csv("usage.csv", date_columns=["usage_date"])


def load_support_tickets() -> pd.DataFrame:
    """Load support_tickets.csv with created_date parsed as a date."""
    return load_csv("support_tickets.csv", date_columns=["created_date"])


def load_churn_events() -> pd.DataFrame:
    """Load churn_events.csv with churn_date parsed as a date."""
    return load_csv("churn_events.csv", date_columns=["churn_date"])


def load_all_datasets() -> dict[str, pd.DataFrame]:
    """Load all five raw datasets once, keyed by dataset name."""
    return {
        "customers": load_customers(),
        "transactions": load_transactions(),
        "usage": load_usage(),
        "support_tickets": load_support_tickets(),
        "churn_events": load_churn_events(),
    }


def save_processed_datasets(datasets: dict[str, pd.DataFrame]) -> dict[str, Path]:
    """Save cleaned datasets to data/processed/ as Parquet files.

    Raw CSV files in data/raw/ are never touched by this function. Requires
    a Parquet engine (pyarrow or fastparquet) to already be installed; it
    will not install one silently.
    """
    has_parquet_engine = (
        importlib.util.find_spec("pyarrow") is not None
        or importlib.util.find_spec("fastparquet") is not None
    )

    if not has_parquet_engine:
        raise ImportError(
            "No Parquet engine is installed (checked for pyarrow and "
            "fastparquet). Install one of them to save processed data as "
            "Parquet -- this function will not install packages silently."
        )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    for name, dataframe in datasets.items():
        output_path = PROCESSED_DATA_DIR / f"{name}.parquet"
        dataframe.to_parquet(output_path, index=False)
        saved_paths[name] = output_path

    return saved_paths
