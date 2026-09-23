"""Leakage-safe feature engineering for churn_next_30_days prediction.

Core convention (also documented in notebooks/04_feature_engineering.ipynb):

    FEATURES: every feature uses only  event_date <= snapshot_date
    TARGET:   churn_next_30_days uses  snapshot_date < churn_date <= snapshot_date + 30 days

The target is the ONLY place future information (churn_date) is
intentionally used -- it is the supervised label being predicted, never a
feature. Every function below that touches transactions/usage/support
tickets filters its source table to "<date column> <= snapshot_date"
*before* aggregating anything.

Snapshot strategy: monthly, month-end snapshots from 2025-01-31 through
2026-05-31. The final snapshot (2026-05-31) plus the 30-day horizon lands
exactly on 2026-06-30, the last date any generated event can have -- so no
snapshot's target window ever needs data beyond what exists.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

SNAPSHOT_START = "2025-01-01"
SNAPSHOT_END = "2026-05-31"
PREDICTION_HORIZON_DAYS = 30

TRANSACTION_WINDOWS_DAYS = [30, 60, 90, 180]
USAGE_RECENT_WINDOW_DAYS = 30
USAGE_LONG_WINDOW_DAYS = 90
SUPPORT_WINDOWS_DAYS = [30, 90, 180]

USAGE_METRIC_COLUMNS = [
    "login_count", "session_minutes",
    "feature_usage_count", "pages_viewed", "active_minutes",
]
USAGE_TREND_METRICS = [
    "login_count", "session_minutes", "feature_usage_count", "active_minutes",
]

CUSTOMER_STATIC_COLUMNS = [
    "customer_id", "age", "country", "plan", "contract_type", "acquisition_channel",
]

# Column-name prefixes that represent a genuine "zero activity" count/sum/rate
# and are therefore safe to fill with 0 when a customer had no matching
# events in that window. Anything NOT matching one of these (e.g. an average,
# a maximum, or a recency/"days since" value) is left as NaN when undefined,
# since 0 would misrepresent "no observation" as a real measured zero.
# Matching is done with str.startswith (not "contains") specifically so that
# columns like "login_count_change_30d" are never mistaken for a plain count.
ZERO_FILLABLE_PREFIXES = (
    "txn_count_", "txn_amount_sum_", "txn_success_count_", "txn_failed_count_",
    "txn_purchase_count_", "txn_renewal_count_", "txn_upgrade_count_", "txn_failed_rate_",
    "active_days_",
    "ticket_count_", "pending_ticket_count_", "resolved_ticket_count_",
    "closed_ticket_count_", "high_critical_ticket_count_", "category_",
)


def _slugify(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Elementwise division that returns 0.0 (never inf/NaN) when the denominator is 0."""
    safe_denominator = denominator.replace(0, np.nan)
    return (numerator / safe_denominator).fillna(0.0)


def _safe_pct_change(recent: pd.Series, previous: pd.Series) -> pd.Series:
    """Percentage change from `previous` to `recent`. NaN (not inf) when previous is 0 or NaN."""
    safe_previous = previous.abs().replace(0, np.nan)
    return (recent - previous) / safe_previous * 100


# ---------------------------------------------------------------------------
# 1. Snapshot spine + target
# ---------------------------------------------------------------------------

def create_prediction_snapshots(customers: pd.DataFrame, churn_events: pd.DataFrame) -> pd.DataFrame:
    """Build the eligible (customer_id, snapshot_date) spine.

    A customer is eligible for a given snapshot only if they had already
    signed up by that date AND had not already churned on or before that
    date (no post-churn prediction records). churn_date is carried on the
    returned frame only so build_target() can compute the label -- it must
    never be merged into the final feature dataset.
    """
    snapshot_dates = pd.date_range(start=SNAPSHOT_START, end=SNAPSHOT_END, freq="MS") + pd.offsets.MonthEnd(0)

    customers_idx = customers[["customer_id", "signup_date"]].assign(_key=1)
    snapshots_idx = pd.DataFrame({"snapshot_date": snapshot_dates}).assign(_key=1)

    spine = customers_idx.merge(snapshots_idx, on="_key").drop(columns="_key")

    # Customer must have already signed up by the snapshot date.
    spine = spine.loc[spine["signup_date"] <= spine["snapshot_date"]]

    spine = spine.merge(
        churn_events[["customer_id", "churn_date"]], on="customer_id", how="left"
    )

    already_churned = spine["churn_date"].notna() & (spine["churn_date"] <= spine["snapshot_date"])
    spine = spine.loc[~already_churned].reset_index(drop=True)

    return spine[["customer_id", "snapshot_date", "signup_date", "churn_date"]]


def build_target(spine: pd.DataFrame) -> pd.Series:
    """churn_next_30_days = 1 iff snapshot_date < churn_date <= snapshot_date + 30 days.

    This is the one field in the whole pipeline allowed to use future
    information, because it is the supervised label, not a feature.
    """
    horizon_end = spine["snapshot_date"] + pd.Timedelta(days=PREDICTION_HORIZON_DAYS)
    within_horizon = (
        spine["churn_date"].notna()
        & (spine["snapshot_date"] < spine["churn_date"])
        & (spine["churn_date"] <= horizon_end)
    )
    return within_horizon.astype(int).rename("churn_next_30_days")


# ---------------------------------------------------------------------------
# 2. Customer / static features
# ---------------------------------------------------------------------------

def build_customer_features(spine: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Static customer attributes + tenure. No windowing needed: these are
    either constant (plan, country, ...) or a direct function of
    snapshot_date and signup_date, both of which are already known at the
    snapshot -- never future information.
    """
    merged = spine[["customer_id", "snapshot_date", "signup_date"]].merge(
        customers[CUSTOMER_STATIC_COLUMNS], on="customer_id", how="left"
    )

    merged["tenure_days"] = (merged["snapshot_date"] - merged["signup_date"]).dt.days
    merged["tenure_months"] = (merged["tenure_days"] / 30.44).round(2)

    return merged.drop(columns=["signup_date"])


# ---------------------------------------------------------------------------
# Generic per-snapshot windowing helper (shared by transactions/usage/support)
# ---------------------------------------------------------------------------

def _per_snapshot_features(spine: pd.DataFrame, events: pd.DataFrame, date_column: str, compute_fn) -> pd.DataFrame:
    """Run compute_fn once per distinct snapshot_date in spine, each time
    using only events with `date_column <= snapshot_date`, then stack the
    results into one (customer_id, snapshot_date, ...features) frame ready
    to merge back onto the spine.

    Iterating over the handful of distinct snapshot dates (17, for the
    default monthly range) -- rather than every (customer, snapshot) pair --
    is what keeps this fast on a 1.2M-row usage table: each iteration is a
    single vectorized boolean filter plus groupby, not a per-row scan.
    """
    snapshot_dates = spine["snapshot_date"].unique()
    pieces = []

    for snapshot_date in snapshot_dates:
        snapshot_ts = pd.Timestamp(snapshot_date)
        events_upto = events.loc[events[date_column] <= snapshot_ts]

        per_customer = compute_fn(events_upto, snapshot_ts)
        per_customer = per_customer.reset_index()
        per_customer["snapshot_date"] = snapshot_ts
        pieces.append(per_customer)

    return pd.concat(pieces, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Transaction / RFM features
# ---------------------------------------------------------------------------

def _transaction_snapshot_features(txn_upto: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Transaction features for one snapshot date, given transactions already
    filtered to transaction_date <= snapshot_date.
    """
    customers_with_history = txn_upto["customer_id"].unique()
    combined = pd.DataFrame(index=pd.Index(customers_with_history, name="customer_id"))

    for window in TRANSACTION_WINDOWS_DAYS:
        window_start = snapshot_date - pd.Timedelta(days=window)
        txn_window = txn_upto.loc[txn_upto["transaction_date"] > window_start]

        # Aggregating precomputed boolean columns with the built-in "sum" is
        # far faster than a per-group Python lambda (e.g. lambda s: (s ==
        # "Success").sum()), which pandas cannot vectorize.
        agg = txn_window.groupby("customer_id").agg(**{
            f"txn_count_{window}d": ("amount", "size"),
            f"txn_amount_sum_{window}d": ("amount", "sum"),
            f"txn_amount_mean_{window}d": ("amount", "mean"),
            f"txn_amount_max_{window}d": ("amount", "max"),
            f"txn_success_count_{window}d": ("_is_success", "sum"),
            f"txn_failed_count_{window}d": ("_is_failed", "sum"),
            f"txn_purchase_count_{window}d": ("_is_purchase", "sum"),
            f"txn_renewal_count_{window}d": ("_is_renewal", "sum"),
            f"txn_upgrade_count_{window}d": ("_is_upgrade", "sum"),
        })
        agg[f"txn_failed_rate_{window}d"] = _safe_divide(
            agg[f"txn_failed_count_{window}d"], agg[f"txn_count_{window}d"]
        )

        combined = combined.join(agg, how="left")

    last_txn = txn_upto.groupby("customer_id")["transaction_date"].max()
    last_success_txn = (
        txn_upto.loc[txn_upto["payment_status"] == "Success"]
        .groupby("customer_id")["transaction_date"].max()
    )
    combined["days_since_last_transaction"] = (snapshot_date - last_txn).dt.days
    combined["days_since_last_successful_transaction"] = (snapshot_date - last_success_txn).dt.days

    combined["has_any_transaction_history"] = True

    return combined


def build_transaction_features(spine: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """RFM-style transaction features, using only transaction_date <= snapshot_date."""
    # Precompute boolean flags once (vectorized) instead of re-deriving them
    # with a per-group lambda inside every window x snapshot aggregation.
    transactions = transactions.assign(
        _is_success=transactions["payment_status"] == "Success",
        _is_failed=transactions["payment_status"] == "Failed",
        _is_purchase=transactions["transaction_type"] == "Purchase",
        _is_renewal=transactions["transaction_type"] == "Renewal",
        _is_upgrade=transactions["transaction_type"] == "Upgrade",
    )
    return _per_snapshot_features(spine, transactions, "transaction_date", _transaction_snapshot_features)


# ---------------------------------------------------------------------------
# 4. Usage features
# ---------------------------------------------------------------------------

def _usage_snapshot_features(usage_upto: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Usage features for one snapshot date, given usage already filtered to
    usage_date <= snapshot_date.
    """
    customers_with_history = usage_upto["customer_id"].unique()
    combined = pd.DataFrame(index=pd.Index(customers_with_history, name="customer_id"))

    recent_start = snapshot_date - pd.Timedelta(days=USAGE_RECENT_WINDOW_DAYS)
    previous_start = snapshot_date - pd.Timedelta(days=2 * USAGE_RECENT_WINDOW_DAYS)
    long_start = snapshot_date - pd.Timedelta(days=USAGE_LONG_WINDOW_DAYS)

    windows = {
        "recent30d": usage_upto.loc[usage_upto["usage_date"] > recent_start],
        "previous30d": usage_upto.loc[
            (usage_upto["usage_date"] > previous_start) & (usage_upto["usage_date"] <= recent_start)
        ],
        "90d": usage_upto.loc[usage_upto["usage_date"] > long_start],
    }

    for label, window_df in windows.items():
        agg = window_df.groupby("customer_id").agg(**{
            f"active_days_{label}": ("usage_date", "size"),
            **{f"avg_{col}_{label}": (col, "mean") for col in USAGE_METRIC_COLUMNS},
        })

        # Missing-measurement rate: average, across this window's rows, of the
        # fraction of the 5 metric columns that were null in that row. Left
        # as NaN (not 0) when the window has no rows -- there is nothing to
        # measure a missing-rate from.
        per_row_missing_fraction = window_df[USAGE_METRIC_COLUMNS].isna().mean(axis=1)
        agg[f"usage_missing_rate_{label}"] = (
            window_df.assign(_missing_fraction=per_row_missing_fraction)
            .groupby("customer_id")["_missing_fraction"].mean()
        )

        combined = combined.join(agg, how="left")

    for metric in USAGE_TREND_METRICS:
        recent_col = f"avg_{metric}_recent30d"
        previous_col = f"avg_{metric}_previous30d"
        combined[f"{metric}_change_30d"] = combined[recent_col] - combined[previous_col]
        combined[f"{metric}_pct_change_30d"] = _safe_pct_change(combined[recent_col], combined[previous_col])

    last_usage = usage_upto.groupby("customer_id")["usage_date"].max()
    combined["days_since_last_usage"] = (snapshot_date - last_usage).dt.days

    combined["has_any_usage_history"] = True

    return combined


def build_usage_features(spine: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    """Engagement features, using only usage_date <= snapshot_date."""
    return _per_snapshot_features(spine, usage, "usage_date", _usage_snapshot_features)


# ---------------------------------------------------------------------------
# 5. Support-ticket features
# ---------------------------------------------------------------------------

def _support_snapshot_features(support_upto: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Support-ticket features for one snapshot date, given tickets already
    filtered to created_date <= snapshot_date.
    """
    customers_with_history = support_upto["customer_id"].unique()
    combined = pd.DataFrame(index=pd.Index(customers_with_history, name="customer_id"))

    # Categories are discovered from the data itself, never hard-coded, so
    # this stays correct even if the raw category set ever changes.
    categories = sorted(support_upto["category"].dropna().unique())

    for window in SUPPORT_WINDOWS_DAYS:
        window_start = snapshot_date - pd.Timedelta(days=window)
        window_df = support_upto.loc[support_upto["created_date"] > window_start]

        # Precomputed boolean columns + built-in "sum" instead of a per-group
        # lambda -- see the identical rationale in _transaction_snapshot_features.
        agg = window_df.groupby("customer_id").agg(**{
            f"ticket_count_{window}d": ("ticket_id", "size"),
            f"pending_ticket_count_{window}d": ("_is_pending", "sum"),
            f"resolved_ticket_count_{window}d": ("_is_resolved", "sum"),
            f"closed_ticket_count_{window}d": ("_is_closed", "sum"),
            f"high_critical_ticket_count_{window}d": ("_is_high_critical", "sum"),
            f"avg_satisfaction_{window}d": ("satisfaction_score", "mean"),
            f"min_satisfaction_{window}d": ("satisfaction_score", "min"),
            f"avg_resolution_hours_{window}d": ("resolution_time_hours", "mean"),
            f"max_resolution_hours_{window}d": ("resolution_time_hours", "max"),
        })

        if categories and not window_df.empty:
            category_counts = window_df.pivot_table(
                index="customer_id", columns="category", values="ticket_id",
                aggfunc="size", fill_value=0,
            )
            category_counts.columns = [
                f"category_{_slugify(category)}_count_{window}d" for category in category_counts.columns
            ]
            agg = agg.join(category_counts, how="left")

        combined = combined.join(agg, how="left")

    last_ticket = support_upto.groupby("customer_id")["created_date"].max()
    combined["days_since_last_ticket"] = (snapshot_date - last_ticket).dt.days

    combined["has_any_support_history"] = True

    return combined


def build_support_features(spine: pd.DataFrame, support_tickets: pd.DataFrame) -> pd.DataFrame:
    """Support-ticket features, using only created_date <= snapshot_date."""
    support_tickets = support_tickets.assign(
        _is_pending=support_tickets["ticket_status"] == "Pending",
        _is_resolved=support_tickets["ticket_status"] == "Resolved",
        _is_closed=support_tickets["ticket_status"] == "Closed",
        _is_high_critical=support_tickets["priority"].isin(["High", "Critical"]),
    )
    return _per_snapshot_features(spine, support_tickets, "created_date", _support_snapshot_features)


# ---------------------------------------------------------------------------
# 6. Assemble the final feature dataset
# ---------------------------------------------------------------------------

def _finalize_missing_values(features: pd.DataFrame) -> pd.DataFrame:
    """Fill genuine zero-activity facts with 0; leave undefined statistics
    (an average/max/recency with zero underlying observations) as NaN so
    they are never mistaken for a real measured zero.
    """
    history_flag_columns = [c for c in features.columns if c.startswith("has_any_")]
    features[history_flag_columns] = features[history_flag_columns].fillna(False).astype(bool)

    zero_fillable_columns = [c for c in features.columns if c.startswith(ZERO_FILLABLE_PREFIXES)]
    features[zero_fillable_columns] = features[zero_fillable_columns].fillna(0)

    return features


def build_feature_dataset(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    usage: pd.DataFrame,
    support_tickets: pd.DataFrame,
    churn_events: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full leakage-safe customer-snapshot feature dataset.

    churn_date and churn_reason are never merged into the returned frame --
    only churn_next_30_days, the constructed target, is.
    """
    spine = create_prediction_snapshots(customers, churn_events)
    target = build_target(spine)

    features = spine[["customer_id", "snapshot_date"]].copy()
    features = features.merge(build_customer_features(spine, customers), on=["customer_id", "snapshot_date"], how="left")
    features = features.merge(build_transaction_features(spine, transactions), on=["customer_id", "snapshot_date"], how="left")
    features = features.merge(build_usage_features(spine, usage), on=["customer_id", "snapshot_date"], how="left")
    features = features.merge(build_support_features(spine, support_tickets), on=["customer_id", "snapshot_date"], how="left")

    features = _finalize_missing_values(features)

    # De-fragment after the many sequential column joins/merges above before
    # adding the final column.
    features = features.copy()
    features["churn_next_30_days"] = target.to_numpy()

    if features.duplicated(subset=["customer_id", "snapshot_date"]).any():
        raise ValueError("build_feature_dataset produced duplicate (customer_id, snapshot_date) rows")

    return features


def save_feature_dataset(features: pd.DataFrame, filename: str = "customer_features.parquet") -> Path:
    """Save the feature dataset to data/processed/ as Parquet. Never touches data/raw/."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / filename
    features.to_parquet(output_path, index=False)
    return output_path


# ---------------------------------------------------------------------------
# 7. Validation / leakage audit
# ---------------------------------------------------------------------------

def validate_feature_dataset(
    features: pd.DataFrame,
    customers: pd.DataFrame,
    churn_events: pd.DataFrame,
) -> dict:
    """Structural + target-correctness checks on an already-built feature dataset.

    Mirrors the reporting style of src/data/validate_data.py: nothing here
    modifies the data, it only counts and reports problems.
    """
    issues = []

    merged_signup = features[["customer_id", "snapshot_date"]].merge(
        customers[["customer_id", "signup_date"]], on="customer_id", how="left"
    )
    snapshot_before_signup = int((merged_signup["snapshot_date"] < merged_signup["signup_date"]).sum())
    if snapshot_before_signup:
        issues.append(f"{snapshot_before_signup} snapshots occur before the customer's signup_date")

    merged_churn = features[["customer_id", "snapshot_date", "churn_next_30_days"]].merge(
        churn_events[["customer_id", "churn_date"]], on="customer_id", how="left"
    )

    post_churn = int(
        (merged_churn["churn_date"].notna() & (merged_churn["churn_date"] <= merged_churn["snapshot_date"])).sum()
    )
    if post_churn:
        issues.append(f"{post_churn} post-churn snapshots found (customer had already churned)")

    horizon_end = merged_churn["snapshot_date"] + pd.Timedelta(days=PREDICTION_HORIZON_DAYS)
    expected_target = (
        merged_churn["churn_date"].notna()
        & (merged_churn["snapshot_date"] < merged_churn["churn_date"])
        & (merged_churn["churn_date"] <= horizon_end)
    ).astype(int)
    target_mismatches = int((expected_target.to_numpy() != merged_churn["churn_next_30_days"].to_numpy()).sum())
    if target_mismatches:
        issues.append(f"{target_mismatches} rows have an incorrect churn_next_30_days value")

    duplicate_pairs = int(features.duplicated(subset=["customer_id", "snapshot_date"]).sum())
    if duplicate_pairs:
        issues.append(f"{duplicate_pairs} duplicate (customer_id, snapshot_date) pairs")

    numeric_columns = features.select_dtypes(include=[np.number]).columns
    infinite_count = int(np.isinf(features[numeric_columns].to_numpy(dtype=float)).sum())
    if infinite_count:
        issues.append(f"{infinite_count} infinite values found across numeric features")

    target_values = set(features["churn_next_30_days"].unique())
    if not target_values <= {0, 1}:
        issues.append(f"churn_next_30_days contains values outside {{0, 1}}: {target_values}")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "snapshot_before_signup": snapshot_before_signup,
        "post_churn_snapshots": post_churn,
        "target_mismatches": target_mismatches,
        "duplicate_customer_snapshot_pairs": duplicate_pairs,
        "infinite_value_count": infinite_count,
    }


def _recency_leakage_check(
    features: pd.DataFrame,
    events: pd.DataFrame,
    date_column: str,
    recency_column: str,
    feature_builder,
    sample_size: int = 200,
    random_state: int = 42,
) -> dict:
    """Non-tautological proof that a recency feature ("days since last X")
    does not use future events.

    For a random sample of (customer_id, snapshot_date) pairs, this
    recomputes the recency value directly from the raw table (filtered to
    <= snapshot_date) and compares it to feature_builder's own output. It
    also checks, for customers who have events both before AND after their
    snapshot_date, what the recency value WOULD have been if future events
    were (incorrectly) included -- and confirms the actual feature does not
    match that leaky value. That second check is what makes this a real
    test of the date boundary rather than a tautological restatement of it.
    """
    sample_spine = features[["customer_id", "snapshot_date"]].drop_duplicates().sample(
        n=min(sample_size, len(features)), random_state=random_state
    ).reset_index(drop=True)

    computed = feature_builder(sample_spine, events)
    computed = sample_spine.merge(computed, on=["customer_id", "snapshot_date"], how="left")

    mismatches = 0
    leakage_proof_found = False

    for _, row in computed.iterrows():
        customer_dates = events.loc[events["customer_id"] == row["customer_id"], date_column]

        past = customer_dates[customer_dates <= row["snapshot_date"]]
        future = customer_dates[customer_dates > row["snapshot_date"]]

        expected = (row["snapshot_date"] - past.max()).days if not past.empty else np.nan
        actual = row[recency_column]

        if not (pd.isna(expected) and pd.isna(actual)) and expected != actual:
            mismatches += 1

        if not past.empty and not future.empty:
            leaky_value = (row["snapshot_date"] - customer_dates.max()).days
            if leaky_value != expected:
                leakage_proof_found = True

    return {"checked": len(computed), "mismatches": mismatches, "leakage_proof_found": leakage_proof_found}


def run_leakage_audit(
    features: pd.DataFrame,
    transactions: pd.DataFrame,
    usage: pd.DataFrame,
    support_tickets: pd.DataFrame,
    sample_size: int = 200,
) -> dict:
    """Run the non-tautological recency leakage proof across all three event tables."""
    return {
        "transactions": _recency_leakage_check(
            features, transactions, "transaction_date",
            "days_since_last_transaction", build_transaction_features, sample_size,
        ),
        "usage": _recency_leakage_check(
            features, usage, "usage_date",
            "days_since_last_usage", build_usage_features, sample_size,
        ),
        "support_tickets": _recency_leakage_check(
            features, support_tickets, "created_date",
            "days_since_last_ticket", build_support_features, sample_size,
        ),
    }
