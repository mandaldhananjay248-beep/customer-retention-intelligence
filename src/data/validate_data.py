"""Validation module for the Customer Retention Intelligence raw datasets.

This module checks schema, datatypes, data quality, foreign-key
relationships, date consistency, and business rules across the five raw
datasets (customers, transactions, usage, support_tickets, churn_events).

Nothing here deletes or modifies records -- every check counts and reports
problems so a human can decide what to do next. Call validate_all() to run
every check at once and get a structured summary.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data_config.yaml"

with open(CONFIG_PATH, "r") as _config_file:
    _config = yaml.safe_load(_config_file)

# The latest date any generated record is allowed to have.
MAX_DATE = pd.Timestamp(_config["end_date"])

# A generously wide bound used only as a sanity check, not tied to the
# specific synthetic generation parameters.
MIN_AGE, MAX_AGE = 18, 100

CUSTOMER_ID_PATTERN = r"^C\d{6}$"

CUSTOMERS_REQUIRED_COLUMNS = [
    "customer_id", "signup_date", "age", "country",
    "plan", "contract_type", "acquisition_channel",
]
TRANSACTIONS_REQUIRED_COLUMNS = [
    "customer_id", "transaction_date", "transaction_type",
    "amount", "payment_method", "payment_status",
]
USAGE_REQUIRED_COLUMNS = [
    "customer_id", "usage_date", "login_count", "session_minutes",
    "feature_usage_count", "pages_viewed", "active_minutes",
]
SUPPORT_TICKETS_REQUIRED_COLUMNS = [
    "ticket_id", "customer_id", "created_date", "category", "priority",
    "resolution_time_hours", "satisfaction_score", "ticket_status",
]
CHURN_EVENTS_REQUIRED_COLUMNS = ["customer_id", "churn_date", "churn_reason"]

USAGE_METRIC_COLUMNS = [
    "login_count", "session_minutes",
    "feature_usage_count", "pages_viewed", "active_minutes",
]

# These mirror the categories used to generate the synthetic data in
# src/data/generate_data.py.
EXPECTED_PLANS = {"Basic", "Standard", "Premium"}
EXPECTED_CONTRACT_TYPES = {"Monthly", "Yearly"}
EXPECTED_ACQUISITION_CHANNELS = {
    "Organic", "Paid Ads", "Referral", "Social Media", "Partner",
}
EXPECTED_TRANSACTION_TYPES = {"Purchase", "Renewal", "Upgrade"}
EXPECTED_PAYMENT_METHODS = {"Credit Card", "Debit Card", "UPI", "Bank Transfer"}
EXPECTED_PAYMENT_STATUSES = {"Success", "Failed"}
EXPECTED_TICKET_CATEGORIES = {
    "Billing", "Technical", "Account",
    "Feature Request", "Login", "Performance", "Other",
}
EXPECTED_TICKET_PRIORITIES = {"Low", "Medium", "High", "Critical"}
EXPECTED_TICKET_STATUSES = {"Resolved", "Closed", "Pending"}
EXPECTED_CHURN_REASONS = {
    "Low Engagement", "Payment Issues", "Poor Experience",
    "Technical Problems", "Customer Choice", "Other",
}


@dataclass
class ValidationResult:
    """Structured validation outcome for a single dataset."""

    dataset: str
    rows: int
    columns: int
    missing_values: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    invalid_ids: int = 0
    invalid_dates: int = 0
    invalid_values: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "FAIL" if self.errors else "PASS"

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


# ---------------------------------------------------------------------------
# Shared, reusable check helpers (used by more than one validate_* function)
# ---------------------------------------------------------------------------

def _check_required_columns(
    dataframe: pd.DataFrame, required_columns: list[str], result: ValidationResult
) -> bool:
    """Record an error and return False if any required column is missing."""
    missing = [c for c in required_columns if c not in dataframe.columns]
    if missing:
        result.add_error(f"Missing required columns: {missing}")
        return False
    return True


def _check_unexpected_categories(
    dataframe: pd.DataFrame, column: str, expected: set, result: ValidationResult
) -> None:
    unexpected = set(dataframe[column].dropna().unique()) - expected
    if unexpected:
        result.add_error(f"Unexpected {column} categories: {sorted(unexpected)}")


def _check_foreign_key(
    dataframe: pd.DataFrame, column: str, valid_ids: set, result: ValidationResult
) -> int:
    invalid_count = int((~dataframe[column].isin(valid_ids)).sum())
    if invalid_count:
        result.add_error(f"{invalid_count} rows reference unknown {column}")
    return invalid_count


def _check_non_negative(dataframe: pd.DataFrame, column: str, result: ValidationResult) -> int:
    invalid_count = int((dataframe[column] < 0).sum())
    if invalid_count:
        result.add_error(f"{invalid_count} rows have negative {column}")
    return invalid_count


def _check_value_range(
    dataframe: pd.DataFrame,
    column: str,
    min_value: float,
    max_value: float,
    result: ValidationResult,
    severity: str = "error",
) -> int:
    series = dataframe[column].dropna()
    invalid_count = int((~series.between(min_value, max_value)).sum())
    if invalid_count:
        message = f"{invalid_count} rows have {column} outside [{min_value}, {max_value}]"
        if severity == "error":
            result.add_error(message)
        else:
            result.add_warning(message)
    return invalid_count


def _check_date_after_signup(
    dataframe: pd.DataFrame,
    date_column: str,
    customers: pd.DataFrame,
    result: ValidationResult,
    strict: bool = False,
) -> int:
    """Check a child table's date against each customer's signup_date.

    strict=False allows the date to equal signup_date (used for
    transactions/usage/support tickets, which may start on signup day).
    strict=True requires the date to be strictly after signup_date (used
    for churn_date).
    """
    merged = dataframe[["customer_id", date_column]].merge(
        customers[["customer_id", "signup_date"]], on="customer_id", how="left"
    )

    if strict:
        violation_mask = merged[date_column] <= merged["signup_date"]
        description = "on or before"
    else:
        violation_mask = merged[date_column] < merged["signup_date"]
        description = "before"

    violations = int(violation_mask.sum())
    if violations:
        result.add_error(f"{violations} rows have {date_column} {description} signup_date")
    return violations


def _check_date_not_after_max(
    dataframe: pd.DataFrame, date_column: str, max_date: pd.Timestamp, result: ValidationResult
) -> int:
    violations = int((dataframe[date_column] > max_date).sum())
    if violations:
        result.add_error(f"{violations} rows have {date_column} after {max_date.date()}")
    return violations


# ---------------------------------------------------------------------------
# Per-dataset validation
# ---------------------------------------------------------------------------

def validate_customers(customers: pd.DataFrame) -> ValidationResult:
    """Validate the customers dataset."""
    result = ValidationResult("customers", len(customers), len(customers.columns))

    if not _check_required_columns(customers, CUSTOMERS_REQUIRED_COLUMNS, result):
        return result

    result.missing_values = customers[CUSTOMERS_REQUIRED_COLUMNS].isna().sum().to_dict()
    result.duplicate_rows = int(customers.duplicated().sum())
    if result.duplicate_rows:
        result.add_warning(f"{result.duplicate_rows} fully duplicate rows found")

    null_ids = int(customers["customer_id"].isna().sum())
    if null_ids:
        result.add_error(f"{null_ids} null customer_id values")

    duplicate_ids = int(customers["customer_id"].duplicated().sum())
    if duplicate_ids:
        result.add_error(f"{duplicate_ids} duplicate customer_id values")

    bad_format = int(
        (~customers["customer_id"].astype(str).str.match(CUSTOMER_ID_PATTERN)).sum()
    )
    if bad_format:
        result.add_error(f"{bad_format} customer_id values do not match format C000001")

    result.invalid_ids = duplicate_ids + bad_format

    if not pd.api.types.is_datetime64_any_dtype(customers["signup_date"]):
        result.add_error("signup_date is not a valid datetime column")

    null_signup = int(customers["signup_date"].isna().sum())
    if null_signup:
        result.add_error(f"{null_signup} null signup_date values")

    result.invalid_values = _check_value_range(
        customers, "age", MIN_AGE, MAX_AGE, result, severity="warning"
    )

    null_country = int(customers["country"].isna().sum())
    if null_country:
        result.add_error(f"{null_country} null country values")

    _check_unexpected_categories(customers, "plan", EXPECTED_PLANS, result)
    _check_unexpected_categories(customers, "contract_type", EXPECTED_CONTRACT_TYPES, result)
    _check_unexpected_categories(
        customers, "acquisition_channel", EXPECTED_ACQUISITION_CHANNELS, result
    )

    return result


def validate_transactions(transactions: pd.DataFrame, customers: pd.DataFrame) -> ValidationResult:
    """Validate the transactions dataset, including customer_id foreign keys."""
    result = ValidationResult("transactions", len(transactions), len(transactions.columns))

    if not _check_required_columns(transactions, TRANSACTIONS_REQUIRED_COLUMNS, result):
        return result

    result.missing_values = transactions[TRANSACTIONS_REQUIRED_COLUMNS].isna().sum().to_dict()
    result.duplicate_rows = int(transactions.duplicated().sum())
    if result.duplicate_rows:
        result.add_warning(f"{result.duplicate_rows} fully duplicate rows found")

    null_ids = int(transactions["customer_id"].isna().sum())
    if null_ids:
        result.add_error(f"{null_ids} null customer_id values")

    result.invalid_ids = _check_foreign_key(
        transactions, "customer_id", set(customers["customer_id"]), result
    )

    if not pd.api.types.is_datetime64_any_dtype(transactions["transaction_date"]):
        result.add_error("transaction_date is not a valid datetime column")

    null_dates = int(transactions["transaction_date"].isna().sum())
    if null_dates:
        result.add_error(f"{null_dates} null transaction_date values")

    before_signup = _check_date_after_signup(transactions, "transaction_date", customers, result)
    after_max = _check_date_not_after_max(transactions, "transaction_date", MAX_DATE, result)
    result.invalid_dates = before_signup + after_max

    result.invalid_values = _check_non_negative(transactions, "amount", result)

    _check_unexpected_categories(transactions, "transaction_type", EXPECTED_TRANSACTION_TYPES, result)
    _check_unexpected_categories(transactions, "payment_method", EXPECTED_PAYMENT_METHODS, result)
    _check_unexpected_categories(transactions, "payment_status", EXPECTED_PAYMENT_STATUSES, result)

    return result


def validate_usage(usage: pd.DataFrame, customers: pd.DataFrame) -> ValidationResult:
    """Validate the usage dataset. ~3% missing values in metric columns are expected."""
    result = ValidationResult("usage", len(usage), len(usage.columns))

    if not _check_required_columns(usage, USAGE_REQUIRED_COLUMNS, result):
        return result

    result.missing_values = usage[USAGE_REQUIRED_COLUMNS].isna().sum().to_dict()
    result.duplicate_rows = int(usage.duplicated().sum())
    if result.duplicate_rows:
        result.add_warning(f"{result.duplicate_rows} fully duplicate rows found")

    null_ids = int(usage["customer_id"].isna().sum())
    if null_ids:
        result.add_error(f"{null_ids} null customer_id values")

    result.invalid_ids = _check_foreign_key(
        usage, "customer_id", set(customers["customer_id"]), result
    )

    if not pd.api.types.is_datetime64_any_dtype(usage["usage_date"]):
        result.add_error("usage_date is not a valid datetime column")

    null_dates = int(usage["usage_date"].isna().sum())
    if null_dates:
        result.add_error(f"{null_dates} null usage_date values")

    before_signup = _check_date_after_signup(usage, "usage_date", customers, result)
    after_max = _check_date_not_after_max(usage, "usage_date", MAX_DATE, result)
    result.invalid_dates = before_signup + after_max

    # Non-negative checks ignore NaN (intentional missingness is not an error).
    invalid_values = 0
    for column in USAGE_METRIC_COLUMNS:
        invalid_values += _check_non_negative(usage, column, result)
    result.invalid_values = invalid_values

    return result


def validate_support_tickets(
    support_tickets: pd.DataFrame, customers: pd.DataFrame
) -> ValidationResult:
    """Validate the support_tickets dataset. ~3% missing resolution/satisfaction is expected."""
    result = ValidationResult(
        "support_tickets", len(support_tickets), len(support_tickets.columns)
    )

    if not _check_required_columns(support_tickets, SUPPORT_TICKETS_REQUIRED_COLUMNS, result):
        return result

    result.missing_values = (
        support_tickets[SUPPORT_TICKETS_REQUIRED_COLUMNS].isna().sum().to_dict()
    )
    result.duplicate_rows = int(support_tickets.duplicated().sum())
    if result.duplicate_rows:
        result.add_warning(f"{result.duplicate_rows} fully duplicate rows found")

    null_ticket_ids = int(support_tickets["ticket_id"].isna().sum())
    if null_ticket_ids:
        result.add_error(f"{null_ticket_ids} null ticket_id values")

    duplicate_ticket_ids = int(support_tickets["ticket_id"].duplicated().sum())
    if duplicate_ticket_ids:
        result.add_error(f"{duplicate_ticket_ids} duplicate ticket_id values")

    null_ids = int(support_tickets["customer_id"].isna().sum())
    if null_ids:
        result.add_error(f"{null_ids} null customer_id values")

    fk_invalid = _check_foreign_key(
        support_tickets, "customer_id", set(customers["customer_id"]), result
    )
    result.invalid_ids = duplicate_ticket_ids + fk_invalid

    if not pd.api.types.is_datetime64_any_dtype(support_tickets["created_date"]):
        result.add_error("created_date is not a valid datetime column")

    null_dates = int(support_tickets["created_date"].isna().sum())
    if null_dates:
        result.add_error(f"{null_dates} null created_date values")

    before_signup = _check_date_after_signup(support_tickets, "created_date", customers, result)
    after_max = _check_date_not_after_max(support_tickets, "created_date", MAX_DATE, result)
    result.invalid_dates = before_signup + after_max

    invalid_resolution = _check_non_negative(support_tickets, "resolution_time_hours", result)
    invalid_satisfaction = _check_value_range(
        support_tickets, "satisfaction_score", 1, 5, result
    )
    result.invalid_values = invalid_resolution + invalid_satisfaction

    _check_unexpected_categories(support_tickets, "category", EXPECTED_TICKET_CATEGORIES, result)
    _check_unexpected_categories(support_tickets, "priority", EXPECTED_TICKET_PRIORITIES, result)
    _check_unexpected_categories(support_tickets, "ticket_status", EXPECTED_TICKET_STATUSES, result)

    return result


def validate_churn_events(churn_events: pd.DataFrame, customers: pd.DataFrame) -> ValidationResult:
    """Validate the churn_events dataset (at most one event per customer)."""
    result = ValidationResult("churn_events", len(churn_events), len(churn_events.columns))

    if not _check_required_columns(churn_events, CHURN_EVENTS_REQUIRED_COLUMNS, result):
        return result

    result.missing_values = churn_events[CHURN_EVENTS_REQUIRED_COLUMNS].isna().sum().to_dict()
    result.duplicate_rows = int(churn_events.duplicated().sum())
    if result.duplicate_rows:
        result.add_warning(f"{result.duplicate_rows} fully duplicate rows found")

    null_ids = int(churn_events["customer_id"].isna().sum())
    if null_ids:
        result.add_error(f"{null_ids} null customer_id values")

    duplicate_ids = int(churn_events["customer_id"].duplicated().sum())
    if duplicate_ids:
        result.add_error(f"{duplicate_ids} duplicate customer_id values (more than one churn event)")

    fk_invalid = _check_foreign_key(
        churn_events, "customer_id", set(customers["customer_id"]), result
    )
    result.invalid_ids = duplicate_ids + fk_invalid

    if not pd.api.types.is_datetime64_any_dtype(churn_events["churn_date"]):
        result.add_error("churn_date is not a valid datetime column")

    null_dates = int(churn_events["churn_date"].isna().sum())
    if null_dates:
        result.add_error(f"{null_dates} null churn_date values")

    not_after_signup = _check_date_after_signup(
        churn_events, "churn_date", customers, result, strict=True
    )
    after_max = _check_date_not_after_max(churn_events, "churn_date", MAX_DATE, result)
    result.invalid_dates = not_after_signup + after_max

    _check_unexpected_categories(churn_events, "churn_reason", EXPECTED_CHURN_REASONS, result)

    return result


# ---------------------------------------------------------------------------
# Cross-table checks
# ---------------------------------------------------------------------------

def validate_customer_relationships(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    usage: pd.DataFrame,
    support_tickets: pd.DataFrame,
    churn_events: pd.DataFrame,
) -> dict[str, int]:
    """Count orphan records: rows in child tables whose customer_id doesn't exist in customers.csv."""
    valid_ids = set(customers["customer_id"])

    orphan_transactions = int((~transactions["customer_id"].isin(valid_ids)).sum())
    orphan_usage = int((~usage["customer_id"].isin(valid_ids)).sum())
    orphan_support_tickets = int((~support_tickets["customer_id"].isin(valid_ids)).sum())
    orphan_churn_events = int((~churn_events["customer_id"].isin(valid_ids)).sum())

    return {
        "orphan_transactions": orphan_transactions,
        "orphan_usage_records": orphan_usage,
        "orphan_support_tickets": orphan_support_tickets,
        "orphan_churn_events": orphan_churn_events,
        "total_orphans": (
            orphan_transactions + orphan_usage + orphan_support_tickets + orphan_churn_events
        ),
    }


def validate_date_consistency(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    usage: pd.DataFrame,
    support_tickets: pd.DataFrame,
    churn_events: pd.DataFrame,
) -> dict[str, int]:
    """Cross-table date checks: every event date vs. its customer's signup_date and the project end date."""
    scratch = ValidationResult(dataset="date_consistency", rows=0, columns=0)

    return {
        "transaction_date_before_signup": _check_date_after_signup(
            transactions, "transaction_date", customers, scratch
        ),
        "usage_date_before_signup": _check_date_after_signup(
            usage, "usage_date", customers, scratch
        ),
        "created_date_before_signup": _check_date_after_signup(
            support_tickets, "created_date", customers, scratch
        ),
        "churn_date_not_after_signup": _check_date_after_signup(
            churn_events, "churn_date", customers, scratch, strict=True
        ),
        "transaction_date_after_max": _check_date_not_after_max(
            transactions, "transaction_date", MAX_DATE, scratch
        ),
        "usage_date_after_max": _check_date_not_after_max(
            usage, "usage_date", MAX_DATE, scratch
        ),
        "created_date_after_max": _check_date_not_after_max(
            support_tickets, "created_date", MAX_DATE, scratch
        ),
        "churn_date_after_max": _check_date_not_after_max(
            churn_events, "churn_date", MAX_DATE, scratch
        ),
    }


def business_rule_summary(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    usage: pd.DataFrame,
    support_tickets: pd.DataFrame,
    churn_events: pd.DataFrame,
) -> dict:
    """Business-level statistics. These are informational, not pass/fail checks."""
    all_customer_ids = set(customers["customer_id"])
    customers_with_transactions = set(transactions["customer_id"])
    customers_with_usage = set(usage["customer_id"])
    customers_with_tickets = set(support_tickets["customer_id"])
    customers_with_churn = set(churn_events["customer_id"])

    negative_usage_values = int(
        sum((usage[column] < 0).sum() for column in USAGE_METRIC_COLUMNS)
    )

    return {
        "total_customers": len(customers),
        "total_transactions": len(transactions),
        "negative_transaction_amounts": int((transactions["amount"] < 0).sum()),
        "failed_payment_rate_pct": round(
            float((transactions["payment_status"] == "Failed").mean()) * 100, 2
        ),
        "negative_usage_metric_values": negative_usage_values,
        "satisfaction_score_out_of_bounds": int(
            (~support_tickets["satisfaction_score"].dropna().between(1, 5)).sum()
        ),
        "negative_resolution_time_hours": int(
            (support_tickets["resolution_time_hours"].dropna() < 0).sum()
        ),
        "customers_with_multiple_churn_events": int(
            (churn_events["customer_id"].value_counts() > 1).sum()
        ),
        "churn_rate_pct": round(len(customers_with_churn) / len(customers) * 100, 2),
        "customers_with_zero_transactions": len(all_customer_ids - customers_with_transactions),
        "customers_with_zero_usage_records": len(all_customer_ids - customers_with_usage),
        "customers_with_zero_support_tickets": len(all_customer_ids - customers_with_tickets),
        "customers_with_churn_event": len(customers_with_churn),
    }


def build_validation_report(results: list[ValidationResult]) -> pd.DataFrame:
    """Turn a list of ValidationResults into the structured summary table."""
    rows = []
    for result in results:
        rows.append({
            "Dataset": result.dataset,
            "Rows": result.rows,
            "Columns": result.columns,
            "Missing Values": sum(result.missing_values.values()),
            "Duplicate Rows": result.duplicate_rows,
            "Invalid IDs": result.invalid_ids,
            "Invalid Dates": result.invalid_dates,
            "Invalid Values": result.invalid_values,
            "Errors": len(result.errors),
            "Warnings": len(result.warnings),
            "Status": result.status,
        })
    return pd.DataFrame(rows)


def validate_all(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    usage: pd.DataFrame,
    support_tickets: pd.DataFrame,
    churn_events: pd.DataFrame,
) -> dict:
    """Run every dataset, relationship, date-consistency, and business-rule check.

    Returns a dict with:
      results          -- list[ValidationResult], one per dataset
      report            -- pd.DataFrame summary table (see build_validation_report)
      relationships     -- dict of orphan-record counts
      date_consistency  -- dict of cross-table date violation counts
      business_rules    -- dict of business-level statistics
      passed            -- bool, True only if every dataset PASSED and there are 0 orphans
    """
    results = [
        validate_customers(customers),
        validate_transactions(transactions, customers),
        validate_usage(usage, customers),
        validate_support_tickets(support_tickets, customers),
        validate_churn_events(churn_events, customers),
    ]

    relationships = validate_customer_relationships(
        customers, transactions, usage, support_tickets, churn_events
    )
    date_consistency = validate_date_consistency(
        customers, transactions, usage, support_tickets, churn_events
    )
    business_rules = business_rule_summary(
        customers, transactions, usage, support_tickets, churn_events
    )

    passed = all(r.status == "PASS" for r in results) and relationships["total_orphans"] == 0

    return {
        "results": results,
        "report": build_validation_report(results),
        "relationships": relationships,
        "date_consistency": date_consistency,
        "business_rules": business_rules,
        "passed": passed,
    }


def print_validation_summary(validation_output: dict) -> None:
    """Print a human-readable summary of validate_all()'s output."""
    print(validation_output["report"].to_string(index=False))
    print()

    for result in validation_output["results"]:
        if result.errors or result.warnings:
            print(f"[{result.dataset}]")
            for message in result.errors:
                print(f"  ERROR: {message}")
            for message in result.warnings:
                print(f"  WARNING: {message}")

    print()
    print("Relationship checks:", validation_output["relationships"])
    print("Date consistency:", validation_output["date_consistency"])
    print("Business rules:", validation_output["business_rules"])
    print()
    print("Overall status:", "PASS" if validation_output["passed"] else "FAIL")
