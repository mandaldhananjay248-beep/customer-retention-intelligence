"""Tests for src/features/feature_engineering.py.

Two kinds of tests here:

1. A small, fully deterministic hand-built dataset (`hand_built_case`) where
   every transaction/usage/ticket/churn date is chosen so we know the exact
   expected feature values by hand -- including some events placed AFTER
   the snapshot date, which must never influence any feature. This is the
   real leakage proof: if the date-boundary filtering were broken, these
   exact-value assertions would fail.

2. Checks against the real generated data (built once per test session via
   the `real_features` fixture) for scale-level properties: snapshot-range,
   duplicates, target bounds, non-negativity, and a sampled statistical
   leakage audit.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import (
    load_churn_events,
    load_customers,
    load_support_tickets,
    load_transactions,
    load_usage,
)
from src.features import feature_engineering as fe


# ---------------------------------------------------------------------------
# Hand-built deterministic case
# ---------------------------------------------------------------------------

SNAPSHOT_DATE = pd.Timestamp("2025-01-31")


@pytest.fixture(scope="module")
def hand_built_case() -> dict:
    """Two customers, one snapshot date, with events deliberately placed both
    before and after the snapshot so leakage would be obvious if it occurred.
    """
    customers = pd.DataFrame({
        "customer_id": ["C000001", "C000002"],
        "signup_date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
        "age": [30, 40],
        "country": ["India", "Nepal"],
        "plan": ["Basic", "Premium"],
        "contract_type": ["Monthly", "Yearly"],
        "acquisition_channel": ["Organic", "Referral"],
    })

    # C000001 has: one txn well inside the 180d window, one in the 30d
    # window, and one in the FUTURE (after snapshot) that must be ignored.
    transactions = pd.DataFrame({
        "customer_id": ["C000001", "C000001", "C000001"],
        "transaction_date": pd.to_datetime(["2024-10-01", "2025-01-15", "2025-02-15"]),
        "transaction_type": ["Renewal", "Purchase", "Upgrade"],
        "amount": [50.0, 100.0, 999.0],
        "payment_method": ["Credit Card", "Credit Card", "Credit Card"],
        "payment_status": ["Success", "Success", "Failed"],
    })

    usage = pd.DataFrame({
        "customer_id": ["C000001", "C000001"],
        "usage_date": pd.to_datetime(["2025-01-20", "2025-02-05"]),  # second is future
        "login_count": [5, 99],
        "session_minutes": [30.0, 999.0],
        "feature_usage_count": [2, 99],
        "pages_viewed": [4, 99],
        "active_minutes": [20.0, 999.0],
    })

    support_tickets = pd.DataFrame({
        "ticket_id": ["T000001", "T000002"],
        "customer_id": ["C000001", "C000001"],
        "created_date": pd.to_datetime(["2025-01-10", "2025-02-10"]),  # second is future
        "category": ["Billing", "Technical"],
        "priority": ["Low", "Critical"],
        "resolution_time_hours": [10.0, 500.0],
        "satisfaction_score": [4, 1],
        "ticket_status": ["Resolved", "Pending"],
    })

    # C000001 churns 20 days after the snapshot (inside the 30-day horizon).
    # C000002 never churns.
    churn_events = pd.DataFrame({
        "customer_id": ["C000001"],
        "churn_date": pd.to_datetime(["2025-02-20"]),
        "churn_reason": ["Poor Experience"],
    })

    spine = fe.create_prediction_snapshots(customers, churn_events)
    spine = spine.loc[spine["snapshot_date"] == SNAPSHOT_DATE].reset_index(drop=True)

    return {
        "customers": customers,
        "transactions": transactions,
        "usage": usage,
        "support_tickets": support_tickets,
        "churn_events": churn_events,
        "spine": spine,
    }


def test_hand_built_eligible_customers(hand_built_case: dict) -> None:
    spine = hand_built_case["spine"]
    assert set(spine["customer_id"]) == {"C000001", "C000002"}


def test_hand_built_target_construction(hand_built_case: dict) -> None:
    spine = hand_built_case["spine"]
    target = fe.build_target(spine).to_numpy()
    lookup = dict(zip(spine["customer_id"], target))

    assert lookup["C000001"] == 1  # churns 20 days later, inside the 30-day horizon
    assert lookup["C000002"] == 0  # never churns


def test_hand_built_transaction_features_ignore_future_transaction(hand_built_case: dict) -> None:
    spine = hand_built_case["spine"]
    result = fe.build_transaction_features(spine, hand_built_case["transactions"])
    row = result.loc[result["customer_id"] == "C000001"].iloc[0]

    # Only the 2025-01-15 transaction falls in the 30-day window; the
    # 2025-02-15 transaction is in the future and must be excluded.
    assert row["txn_count_30d"] == 1
    assert row["txn_amount_sum_30d"] == 100.0

    # Both 2024-10-01 and 2025-01-15 fall in the 180-day window (the future
    # transaction still must not count).
    assert row["txn_count_180d"] == 2
    assert row["txn_amount_sum_180d"] == 150.0

    # 2024-10-01 is 122 days before the snapshot, outside the 90-day window.
    assert row["txn_count_90d"] == 1

    # Recency must reflect the past transaction (16 days), never the future
    # one -- which would even make this go negative if it leaked in.
    assert row["days_since_last_transaction"] == 16
    assert row["days_since_last_transaction"] >= 0


def test_hand_built_usage_features_ignore_future_usage(hand_built_case: dict) -> None:
    spine = hand_built_case["spine"]
    result = fe.build_usage_features(spine, hand_built_case["usage"])
    row = result.loc[result["customer_id"] == "C000001"].iloc[0]

    assert row["active_days_recent30d"] == 1
    assert row["avg_login_count_recent30d"] == 5
    assert row["days_since_last_usage"] == 11
    assert row["days_since_last_usage"] >= 0


def test_hand_built_support_features_ignore_future_ticket(hand_built_case: dict) -> None:
    spine = hand_built_case["spine"]
    result = fe.build_support_features(spine, hand_built_case["support_tickets"])
    row = result.loc[result["customer_id"] == "C000001"].iloc[0]

    assert row["ticket_count_30d"] == 1
    assert row["avg_satisfaction_30d"] == 4.0
    assert row["days_since_last_ticket"] == 21
    assert row["days_since_last_ticket"] >= 0


def test_hand_built_full_dataset_matches_manual_values(hand_built_case: dict) -> None:
    features = fe.build_feature_dataset(
        hand_built_case["customers"],
        hand_built_case["transactions"],
        hand_built_case["usage"],
        hand_built_case["support_tickets"],
        hand_built_case["churn_events"],
    )
    row = features.loc[
        (features["customer_id"] == "C000001") & (features["snapshot_date"] == SNAPSHOT_DATE)
    ].iloc[0]

    assert row["tenure_days"] == (SNAPSHOT_DATE - pd.Timestamp("2024-01-01")).days
    assert row["txn_count_30d"] == 1
    assert row["avg_login_count_recent30d"] == 5
    assert row["ticket_count_30d"] == 1
    assert row["churn_next_30_days"] == 1
    assert "churn_date" not in features.columns
    assert "churn_reason" not in features.columns


# ---------------------------------------------------------------------------
# Checks against the real generated data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_datasets() -> dict[str, pd.DataFrame]:
    return {
        "customers": load_customers(),
        "transactions": load_transactions(),
        "usage": load_usage(),
        "support_tickets": load_support_tickets(),
        "churn_events": load_churn_events(),
    }


@pytest.fixture(scope="module")
def real_features(real_datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return fe.build_feature_dataset(
        real_datasets["customers"],
        real_datasets["transactions"],
        real_datasets["usage"],
        real_datasets["support_tickets"],
        real_datasets["churn_events"],
    )


def test_snapshot_dates_within_allowed_range(real_features: pd.DataFrame) -> None:
    assert real_features["snapshot_date"].min() >= pd.Timestamp(fe.SNAPSHOT_START)
    assert real_features["snapshot_date"].max() <= pd.Timestamp(fe.SNAPSHOT_END)


def test_no_duplicate_customer_snapshot_pairs(real_features: pd.DataFrame) -> None:
    assert not real_features.duplicated(subset=["customer_id", "snapshot_date"]).any()


def test_no_post_churn_snapshots(real_features: pd.DataFrame, real_datasets: dict) -> None:
    validation = fe.validate_feature_dataset(
        real_features, real_datasets["customers"], real_datasets["churn_events"]
    )
    assert validation["post_churn_snapshots"] == 0


def test_target_construction_matches_business_rule(real_features: pd.DataFrame, real_datasets: dict) -> None:
    validation = fe.validate_feature_dataset(
        real_features, real_datasets["customers"], real_datasets["churn_events"]
    )
    assert validation["target_mismatches"] == 0


def test_target_is_binary(real_features: pd.DataFrame) -> None:
    assert set(real_features["churn_next_30_days"].unique()) <= {0, 1}


def test_tenure_is_non_negative(real_features: pd.DataFrame) -> None:
    assert (real_features["tenure_days"] >= 0).all()


def test_key_numeric_features_are_non_negative(real_features: pd.DataFrame) -> None:
    non_negative_columns = [
        "txn_count_30d", "txn_amount_sum_30d", "txn_failed_rate_30d",
        "active_days_recent30d", "ticket_count_30d",
    ]
    for column in non_negative_columns:
        values = real_features[column].dropna()
        assert (values >= 0).all(), f"{column} has negative values"


def test_satisfaction_scores_within_bounds(real_features: pd.DataFrame) -> None:
    for column in ["avg_satisfaction_30d", "min_satisfaction_30d"]:
        values = real_features[column].dropna()
        assert values.between(1, 5).all()


def test_no_infinite_values(real_features: pd.DataFrame) -> None:
    numeric_columns = real_features.select_dtypes(include=[np.number]).columns
    assert not np.isinf(real_features[numeric_columns].to_numpy(dtype=float)).any()


def test_leakage_audit_finds_no_future_data_used(real_features: pd.DataFrame, real_datasets: dict) -> None:
    """Statistical leakage proof on the real dataset: sampled recency features
    match ground truth computed directly from raw data cut at snapshot_date,
    and -- critically -- differ from what they'd be if future events leaked in.
    """
    audit = fe.run_leakage_audit(
        real_features,
        real_datasets["transactions"],
        real_datasets["usage"],
        real_datasets["support_tickets"],
        sample_size=150,
    )
    for table_name, result in audit.items():
        assert result["mismatches"] == 0, f"{table_name}: {result['mismatches']} leakage mismatches found"


def test_feature_dataset_round_trip_parquet(real_features: pd.DataFrame, tmp_path) -> None:
    output_path = tmp_path / "customer_features.parquet"
    real_features.to_parquet(output_path, index=False)
    reloaded = pd.read_parquet(output_path)

    assert reloaded.shape == real_features.shape
    assert list(reloaded.columns) == list(real_features.columns)
    assert reloaded["snapshot_date"].dtype == real_features["snapshot_date"].dtype
