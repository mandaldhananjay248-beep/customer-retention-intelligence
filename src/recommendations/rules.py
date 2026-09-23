"""Phase 9 business-rule signal detectors.

Each function inspects a handful of real feature columns (present in
data/processed/customer_features.parquet -- see notebooks/09_next_best_action.ipynb
section 1 for the percentile analysis behind the thresholds in
configs/recommendation_config.yaml) and reports whether that category of
problem is present for one customer/snapshot row, together with the exact
feature/value pairs that triggered it.

Every check accepts anything with a dict-like `.get(key)` (a pandas Series
or a plain dict both work). A missing or NaN feature simply fails to
trigger that specific condition -- it never raises -- since a customer can
legitimately have no support-ticket or transaction history at all.

No function here fabricates a value: every "supporting feature" string is
built directly from the row's own data.
"""

from __future__ import annotations

import pandas as pd


def _get_numeric(row, feature: str) -> float | None:
    """The row's value for `feature`, or None if the column is absent or
    the value is missing/NaN. Never raises on an unexpected row shape.
    """
    value = row.get(feature) if hasattr(row, "get") else None
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)


def check_low_engagement(row, config: dict) -> tuple[bool, list[str]]:
    """Sustained low activity: both active days and active minutes over
    the last 90 days are at/below their configured (bottom-quartile-ish)
    ceilings. Requires both, not just one, so a customer who is merely
    quiet on one dimension isn't flagged as broadly disengaged.
    """
    thresholds = config["signal_thresholds"]["low_engagement"]
    active_days = _get_numeric(row, "active_days_90d")
    active_minutes = _get_numeric(row, "avg_active_minutes_90d")

    triggered = (
        active_days is not None and active_days <= thresholds["active_days_90d_max"]
        and active_minutes is not None and active_minutes <= thresholds["avg_active_minutes_90d_max"]
    )
    if not triggered:
        return False, []

    return True, [
        f"active_days_90d={active_days:g} (<= {thresholds['active_days_90d_max']:g})",
        f"avg_active_minutes_90d={active_minutes:g} (<= {thresholds['avg_active_minutes_90d_max']:g})",
    ]


def check_reduced_usage(row, config: dict) -> tuple[bool, list[str]]:
    """A meaningful recent decline vs. the prior 30-day period, on any one
    of active minutes, login count, or feature-usage count. Any single
    qualifying drop is enough -- these three overlap but aren't identical.
    """
    thresholds = config["signal_thresholds"]["reduced_usage"]
    checks = [
        ("active_minutes_pct_change_30d", thresholds["active_minutes_pct_change_30d_max"]),
        ("login_count_pct_change_30d", thresholds["login_count_pct_change_30d_max"]),
        ("feature_usage_count_pct_change_30d", thresholds["feature_usage_count_pct_change_30d_max"]),
    ]

    matched = []
    for feature, max_threshold in checks:
        value = _get_numeric(row, feature)
        if value is not None and value <= max_threshold:
            matched.append(f"{feature}={value:g}% (<= {max_threshold:g}%)")

    return bool(matched), matched


def check_support_problems(row, config: dict) -> tuple[bool, list[str]]:
    """Unresolved, high-severity, or low-satisfaction support activity in
    the last 90 days. Any single qualifying condition is enough.
    """
    thresholds = config["signal_thresholds"]["support_problems"]
    matched = []

    pending = _get_numeric(row, "pending_ticket_count_90d")
    if pending is not None and pending >= thresholds["pending_ticket_count_90d_min"]:
        matched.append(f"pending_ticket_count_90d={pending:g} (>= {thresholds['pending_ticket_count_90d_min']:g})")

    critical = _get_numeric(row, "high_critical_ticket_count_90d")
    if critical is not None and critical >= thresholds["high_critical_ticket_count_90d_min"]:
        matched.append(
            f"high_critical_ticket_count_90d={critical:g} (>= {thresholds['high_critical_ticket_count_90d_min']:g})"
        )

    satisfaction = _get_numeric(row, "avg_satisfaction_90d")
    if satisfaction is not None and satisfaction <= thresholds["avg_satisfaction_90d_max"]:
        matched.append(f"avg_satisfaction_90d={satisfaction:g} (<= {thresholds['avg_satisfaction_90d_max']:g})")

    return bool(matched), matched


def check_payment_problems(row, config: dict) -> tuple[bool, list[str]]:
    """Payment friction: repeated recent failures, a high failure rate, or
    a long gap since the last successful transaction. Any single
    qualifying condition is enough.
    """
    thresholds = config["signal_thresholds"]["payment_problems"]
    matched = []

    failed_count = _get_numeric(row, "txn_failed_count_90d")
    if failed_count is not None and failed_count >= thresholds["txn_failed_count_90d_min"]:
        matched.append(f"txn_failed_count_90d={failed_count:g} (>= {thresholds['txn_failed_count_90d_min']:g})")

    failed_rate = _get_numeric(row, "txn_failed_rate_90d")
    if failed_rate is not None and failed_rate >= thresholds["txn_failed_rate_90d_min"]:
        matched.append(f"txn_failed_rate_90d={failed_rate:g} (>= {thresholds['txn_failed_rate_90d_min']:g})")

    days_since_success = _get_numeric(row, "days_since_last_successful_transaction")
    if days_since_success is not None and days_since_success >= thresholds["days_since_last_successful_transaction_min"]:
        matched.append(
            f"days_since_last_successful_transaction={days_since_success:g} "
            f"(>= {thresholds['days_since_last_successful_transaction_min']:g})"
        )

    return bool(matched), matched


# Ordered so callers that need a fixed iteration order (e.g. deciding
# which signal "wins" when several are checked in sequence) get one
# without re-declaring the list elsewhere.
SIGNAL_CHECKS: dict[str, callable] = {
    "payment_problems": check_payment_problems,
    "support_problems": check_support_problems,
    "reduced_usage": check_reduced_usage,
    "low_engagement": check_low_engagement,
}

# Every raw feature column any rule above reads -- used to subset large
# batches before turning them into per-row dicts, and to assert real
# features (never fabricated ones) back these recommendations.
REQUIRED_SIGNAL_FEATURES: tuple[str, ...] = (
    "active_days_90d",
    "avg_active_minutes_90d",
    "active_minutes_pct_change_30d",
    "login_count_pct_change_30d",
    "feature_usage_count_pct_change_30d",
    "pending_ticket_count_90d",
    "high_critical_ticket_count_90d",
    "avg_satisfaction_90d",
    "txn_failed_count_90d",
    "txn_failed_rate_90d",
    "days_since_last_successful_transaction",
)


def evaluate_signals(row, config: dict) -> dict[str, tuple[bool, list[str]]]:
    """Run every signal category against one row. Returns
    {category_name: (triggered, supporting_feature_strings)}.
    """
    return {name: check(row, config) for name, check in SIGNAL_CHECKS.items()}
