"""Tests that the generated raw datasets satisfy basic identity and range rules.

These run against the actual files in data/raw/, so they assume
src/data/generate_data.py has already been run at least once.
"""

import pandas as pd
import pytest

from src.data.load_data import (
    load_customers,
    load_support_tickets,
    load_transactions,
    load_usage,
)

MAX_DATE = pd.Timestamp("2026-06-30")


@pytest.fixture(scope="module")
def customers() -> pd.DataFrame:
    return load_customers()


@pytest.fixture(scope="module")
def transactions() -> pd.DataFrame:
    return load_transactions()


@pytest.fixture(scope="module")
def usage() -> pd.DataFrame:
    return load_usage()


@pytest.fixture(scope="module")
def support_tickets() -> pd.DataFrame:
    return load_support_tickets()


def test_customer_id_is_unique(customers: pd.DataFrame) -> None:
    assert customers["customer_id"].is_unique


def test_transaction_customer_ids_exist_in_customers(
    transactions: pd.DataFrame, customers: pd.DataFrame
) -> None:
    valid_ids = set(customers["customer_id"])
    assert transactions["customer_id"].isin(valid_ids).all()


def test_usage_customer_ids_exist_in_customers(
    usage: pd.DataFrame, customers: pd.DataFrame
) -> None:
    valid_ids = set(customers["customer_id"])
    assert usage["customer_id"].isin(valid_ids).all()


def test_support_ticket_customer_ids_exist_in_customers(
    support_tickets: pd.DataFrame, customers: pd.DataFrame
) -> None:
    valid_ids = set(customers["customer_id"])
    assert support_tickets["customer_id"].isin(valid_ids).all()


def test_transaction_dates_not_before_signup(
    transactions: pd.DataFrame, customers: pd.DataFrame
) -> None:
    merged = transactions.merge(
        customers[["customer_id", "signup_date"]], on="customer_id", how="left"
    )
    assert (merged["transaction_date"] >= merged["signup_date"]).all()


def test_usage_dates_not_before_signup(usage: pd.DataFrame, customers: pd.DataFrame) -> None:
    merged = usage.merge(
        customers[["customer_id", "signup_date"]], on="customer_id", how="left"
    )
    assert (merged["usage_date"] >= merged["signup_date"]).all()


def test_transaction_amounts_are_non_negative(transactions: pd.DataFrame) -> None:
    assert (transactions["amount"] >= 0).all()


def test_usage_metrics_are_non_negative(usage: pd.DataFrame) -> None:
    metric_columns = [
        "login_count", "session_minutes",
        "feature_usage_count", "pages_viewed", "active_minutes",
    ]
    for column in metric_columns:
        assert (usage[column].dropna() >= 0).all()


def test_satisfaction_score_within_bounds(support_tickets: pd.DataFrame) -> None:
    scores = support_tickets["satisfaction_score"].dropna()
    assert scores.between(1, 5).all()


def test_no_dates_after_project_end(transactions: pd.DataFrame, usage: pd.DataFrame) -> None:
    assert (transactions["transaction_date"] <= MAX_DATE).all()
    assert (usage["usage_date"] <= MAX_DATE).all()
