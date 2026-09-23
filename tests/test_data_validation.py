"""Tests for src/data/validate_data.py itself.

These check both the "happy path" (the real generated data passes) and that
the validators correctly catch injected problems in small synthetic frames,
so the checks are known to actually work rather than always passing by luck.
"""

import pandas as pd
import pytest

from src.data.load_data import (
    load_churn_events,
    load_customers,
    load_support_tickets,
    load_transactions,
    load_usage,
)
from src.data.validate_data import (
    validate_all,
    validate_churn_events,
    validate_customers,
)


@pytest.fixture(scope="module")
def real_datasets() -> dict[str, pd.DataFrame]:
    return {
        "customers": load_customers(),
        "transactions": load_transactions(),
        "usage": load_usage(),
        "support_tickets": load_support_tickets(),
        "churn_events": load_churn_events(),
    }


def test_real_data_passes_validate_all(real_datasets: dict[str, pd.DataFrame]) -> None:
    output = validate_all(
        real_datasets["customers"],
        real_datasets["transactions"],
        real_datasets["usage"],
        real_datasets["support_tickets"],
        real_datasets["churn_events"],
    )
    assert output["passed"] is True
    assert output["relationships"]["total_orphans"] == 0


def test_real_churn_customer_ids_are_unique(real_datasets: dict[str, pd.DataFrame]) -> None:
    result = validate_churn_events(real_datasets["churn_events"], real_datasets["customers"])
    assert result.status == "PASS"


def test_churn_date_after_signup_violation_is_caught() -> None:
    customers = pd.DataFrame({
        "customer_id": ["C000001"],
        "signup_date": pd.to_datetime(["2024-01-10"]),
        "age": [30],
        "country": ["India"],
        "plan": ["Basic"],
        "contract_type": ["Monthly"],
        "acquisition_channel": ["Organic"],
    })
    # churn_date is BEFORE signup_date -- this must be flagged as an error.
    churn_events = pd.DataFrame({
        "customer_id": ["C000001"],
        "churn_date": pd.to_datetime(["2024-01-01"]),
        "churn_reason": ["Low Engagement"],
    })

    result = validate_churn_events(churn_events, customers)

    assert result.status == "FAIL"
    assert result.invalid_dates == 1


def test_duplicate_customer_id_is_caught() -> None:
    customers = pd.DataFrame({
        "customer_id": ["C000001", "C000001"],
        "signup_date": pd.to_datetime(["2024-01-10", "2024-02-01"]),
        "age": [30, 40],
        "country": ["India", "Nepal"],
        "plan": ["Basic", "Premium"],
        "contract_type": ["Monthly", "Yearly"],
        "acquisition_channel": ["Organic", "Referral"],
    })

    result = validate_customers(customers)

    assert result.status == "FAIL"
    assert result.invalid_ids == 1
