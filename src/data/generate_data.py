from pathlib import Path
import numpy as np
import pandas as pd
import yaml


# Load configuration
CONFIG_PATH = Path("configs/data_config.yaml")

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)


# Configuration values
RANDOM_SEED = config["random_seed"]
NUM_CUSTOMERS = config["num_customers"]
OUTPUT_DIRECTORY = Path(config["output_directory"])
MISSING_VALUE_RATE = config["missing_value_rate"]


# Reproducibility
np.random.seed(RANDOM_SEED)


def generate_customers(num_customers: int) -> pd.DataFrame:
    """Generate synthetic customer profile data."""

    customer_ids = [
        f"C{i:06d}"
        for i in range(1, num_customers + 1)
    ]

    signup_dates = pd.to_datetime(
        np.random.choice(
            pd.date_range("2023-01-01", "2025-12-31"),
            size=num_customers
        )
    )

    ages = np.random.randint(18, 70, size=num_customers)

    countries = np.random.choice(
        ["Nepal", "India", "United States", "United Kingdom", "Australia"],
        size=num_customers,
        p=[0.20, 0.40, 0.20, 0.10, 0.10]
    )

    plans = np.random.choice(
        ["Basic", "Standard", "Premium"],
        size=num_customers,
        p=[0.50, 0.30, 0.20]
    )

    contract_types = np.random.choice(
        ["Monthly", "Yearly"],
        size=num_customers,
        p=[0.70, 0.30]
    )

    acquisition_channels = np.random.choice(
        ["Organic", "Paid Ads", "Referral", "Social Media", "Partner"],
        size=num_customers,
        p=[0.30, 0.25, 0.20, 0.15, 0.10]
    )

    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "signup_date": signup_dates,
        "age": ages,
        "country": countries,
        "plan": plans,
        "contract_type": contract_types,
        "acquisition_channel": acquisition_channels
    })

    return customers


def generate_transactions(customers: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic transaction data."""

    transaction_records = []

    payment_methods = [
        "Credit Card",
        "Debit Card",
        "UPI",
        "Bank Transfer"
    ]

    transaction_types = [
        "Purchase",
        "Renewal",
        "Upgrade"
    ]

    for _, customer in customers.iterrows():

        num_transactions = np.random.poisson(30)

        if num_transactions == 0:
            num_transactions = 1

        signup_date = customer["signup_date"]

        available_dates = pd.date_range(
            signup_date,
            "2026-06-30"
        )

        if len(available_dates) == 0:
            continue

        dates = np.random.choice(
            available_dates,
            size=num_transactions
        )

        for date in dates:

            transaction_type = np.random.choice(
                transaction_types,
                p=[0.65, 0.25, 0.10]
            )

            if customer["plan"] == "Basic":
                amount = np.random.normal(30, 8)

            elif customer["plan"] == "Standard":
                amount = np.random.normal(60, 15)

            else:
                amount = np.random.normal(120, 25)

            amount = max(5, round(amount, 2))

            payment_status = np.random.choice(
                ["Success", "Failed"],
                p=[0.95, 0.05]
            )

            payment_method = np.random.choice(
                payment_methods,
                p=[0.30, 0.25, 0.35, 0.10]
            )

            transaction_records.append({
                "customer_id": customer["customer_id"],
                "transaction_date": date,
                "transaction_type": transaction_type,
                "amount": amount,
                "payment_method": payment_method,
                "payment_status": payment_status
            })

    return pd.DataFrame(transaction_records)


def generate_usage(customers: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic longitudinal customer usage/engagement data."""

    end_date = pd.Timestamp("2026-06-30")

    # Base engagement level per plan tier
    plan_engagement_base = {
        "Basic": 2.5,
        "Standard": 4.5,
        "Premium": 7.0,
    }

    usage_frames = []

    for _, customer in customers.iterrows():

        signup_date = customer["signup_date"]

        available_dates = pd.date_range(signup_date, end_date)

        if len(available_dates) == 0:
            continue

        num_observations = np.random.randint(100, 151)
        num_observations = min(num_observations, len(available_dates))

        dates = np.random.choice(
            available_dates,
            size=num_observations,
            replace=False
        )
        dates = np.sort(dates)

        # Per-customer engagement multiplier: right-skewed so most customers
        # cluster around "typical" engagement, with some naturally very low
        # and some naturally very high.
        engagement_factor = np.random.gamma(shape=2.0, scale=0.6)

        base_rate = plan_engagement_base[customer["plan"]] * engagement_factor

        login_count = np.random.poisson(
            lam=max(base_rate, 0.1),
            size=num_observations
        )
        login_count = np.clip(login_count, 0, 15)

        session_minutes = (
            login_count * np.random.uniform(4, 18, size=num_observations)
            + np.random.normal(0, 5, size=num_observations)
        )
        session_minutes = np.clip(session_minutes, 0, None)

        feature_lambda = np.clip(
            0.5 * login_count + 0.05 * session_minutes + 0.2,
            0.05,
            None
        )
        feature_usage_count = np.random.poisson(lam=feature_lambda)

        pages_lambda = np.clip(0.3 * session_minutes + 0.5, 0.05, None)
        pages_viewed = np.random.poisson(lam=pages_lambda)

        active_minutes = session_minutes * np.random.uniform(
            0.4, 0.9, size=num_observations
        )
        active_minutes = np.clip(active_minutes, 0, None)

        customer_usage = pd.DataFrame({
            "customer_id": customer["customer_id"],
            "usage_date": dates,
            "login_count": login_count,
            "session_minutes": np.round(session_minutes, 2),
            "feature_usage_count": feature_usage_count,
            "pages_viewed": pages_viewed,
            "active_minutes": np.round(active_minutes, 2),
        })

        usage_frames.append(customer_usage)

    usage = pd.concat(usage_frames, ignore_index=True)

    usage = usage.sort_values(
        ["customer_id", "usage_date"]
    ).reset_index(drop=True)

    # Introduce missing values in usage metric columns (not customer_id)
    missing_columns = [
        "login_count",
        "session_minutes",
        "feature_usage_count",
        "pages_viewed",
        "active_minutes",
    ]

    for column in missing_columns:
        missing_mask = np.random.rand(len(usage)) < MISSING_VALUE_RATE
        usage.loc[missing_mask, column] = np.nan

    return usage


def generate_support_tickets(customers: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic longitudinal customer support ticket data."""

    end_date = pd.Timestamp("2026-06-30")

    categories = [
        "Billing",
        "Technical",
        "Account",
        "Feature Request",
        "Login",
        "Performance",
        "Other",
    ]
    category_weights = [0.22, 0.22, 0.13, 0.10, 0.13, 0.12, 0.08]

    priorities = ["Low", "Medium", "High", "Critical"]
    priority_weights = [0.55, 0.30, 0.11, 0.04]

    # Typical resolution time (hours) per priority — higher priority tickets
    # get more urgent attention and resolve faster, on average.
    priority_base_hours = {
        "Low": 60.0,
        "Medium": 30.0,
        "High": 12.0,
        "Critical": 5.0,
    }

    statuses = ["Resolved", "Closed", "Pending"]
    status_weights = [0.60, 0.25, 0.15]

    # Pending tickets tend to reflect unresolved frustration; resolved
    # tickets skew slightly happier. This is a soft nudge, not a rule.
    status_satisfaction_adjustment = {
        "Resolved": 0.3,
        "Closed": 0.0,
        "Pending": -0.8,
    }

    # Baseline support-contact volume per plan tier (Premium customers use
    # more features and therefore file somewhat more tickets on average).
    plan_ticket_base = {
        "Basic": 7.0,
        "Standard": 9.0,
        "Premium": 12.0,
    }

    ticket_frames = []
    ticket_counter = 1

    for _, customer in customers.iterrows():

        signup_date = customer["signup_date"]

        available_dates = pd.date_range(signup_date, end_date)

        if len(available_dates) == 0:
            continue

        # A meaningful share of customers essentially never contact support.
        if np.random.rand() < 0.22:
            continue

        # Per-customer support-proneness multiplier: right-skewed so most
        # customers file a handful of tickets, a few file many.
        ticket_proneness = np.random.gamma(shape=1.8, scale=0.7)

        lam = plan_ticket_base[customer["plan"]] * ticket_proneness

        num_tickets = np.random.poisson(lam=max(lam, 0.05))

        if num_tickets == 0:
            continue

        dates = np.sort(
            np.random.choice(available_dates, size=num_tickets, replace=True)
        )

        category_arr = np.random.choice(
            categories, size=num_tickets, p=category_weights
        )

        priority_arr = np.random.choice(
            priorities, size=num_tickets, p=priority_weights
        )

        status_arr = np.random.choice(
            statuses, size=num_tickets, p=status_weights
        )

        base_hours_arr = pd.Series(priority_arr).map(priority_base_hours).values

        resolution_time_hours = np.random.lognormal(
            mean=np.log(base_hours_arr), sigma=0.6
        )

        status_adjustment_arr = pd.Series(status_arr).map(
            status_satisfaction_adjustment
        ).values

        satisfaction_raw = (
            5
            - np.sqrt(resolution_time_hours) / 4
            + status_adjustment_arr
            + np.random.normal(0, 0.7, size=num_tickets)
        )
        satisfaction_score = np.clip(np.round(satisfaction_raw), 1, 5)

        ticket_ids = [
            f"T{i:06d}"
            for i in range(ticket_counter, ticket_counter + num_tickets)
        ]
        ticket_counter += num_tickets

        customer_tickets = pd.DataFrame({
            "ticket_id": ticket_ids,
            "customer_id": customer["customer_id"],
            "created_date": dates,
            "category": category_arr,
            "priority": priority_arr,
            "resolution_time_hours": np.round(resolution_time_hours, 2),
            "satisfaction_score": satisfaction_score,
            "ticket_status": status_arr,
        })

        ticket_frames.append(customer_tickets)

    support_tickets = pd.concat(ticket_frames, ignore_index=True)

    support_tickets = support_tickets.sort_values(
        ["customer_id", "created_date"]
    ).reset_index(drop=True)

    # Introduce missing values in the appropriate metric columns only.
    # Identifiers and created_date must never be missing.
    missing_columns = ["resolution_time_hours", "satisfaction_score"]

    for column in missing_columns:
        missing_mask = np.random.rand(len(support_tickets)) < MISSING_VALUE_RATE
        support_tickets.loc[missing_mask, column] = np.nan

    return support_tickets


def generate_churn_events(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    usage: pd.DataFrame,
    support_tickets: pd.DataFrame,
) -> pd.DataFrame:
    """Generate realistic churn events driven by aggregated historical behavior.

    Behavioral signals (transaction activity, usage/engagement, support
    experience) are aggregated per customer with vectorized groupby
    operations, combined into a continuous, noisy risk score, and converted
    into a churn probability. Whether a customer churns is then sampled
    probabilistically rather than via a deterministic threshold rule.

    Each customer's churn_date is sampled independently within their own
    lifecycle window (signup_date, 2026-06-30], skewed toward the later
    part of that window. The event table only ever stores a single date per
    churned customer and carries no future information about that customer
    — downstream feature engineering remains responsible for only using
    data on/before a chosen prediction_date when building features.
    """

    end_date = pd.Timestamp("2026-06-30")

    # ---- Aggregate historical signals per customer (vectorized) ----

    txn_agg = transactions.groupby("customer_id").agg(
        txn_count=("amount", "size"),
        failed_rate=("payment_status", lambda s: (s == "Failed").mean()),
    )

    usage_agg = usage.groupby("customer_id").agg(
        avg_login=("login_count", "mean"),
        avg_session=("session_minutes", "mean"),
        avg_feature_usage=("feature_usage_count", "mean"),
    )

    support_agg = support_tickets.groupby("customer_id").agg(
        ticket_count=("ticket_id", "size"),
        avg_satisfaction=("satisfaction_score", "mean"),
        avg_resolution=("resolution_time_hours", "mean"),
        pending_count=("ticket_status", lambda s: (s == "Pending").sum()),
    )

    tech_ratio = (
        support_tickets
        .assign(is_tech=support_tickets["category"].isin(["Technical", "Performance"]))
        .groupby("customer_id")["is_tech"]
        .mean()
        .rename("tech_ticket_ratio")
    )

    profile = customers.set_index("customer_id").join(
        [txn_agg, usage_agg, support_agg, tech_ratio]
    )

    # Customers with no support tickets have no support burden at all;
    # fill with neutral/zero defaults rather than dropping them.
    zero_fill_columns = [
        "txn_count", "failed_rate",
        "avg_login", "avg_session", "avg_feature_usage",
        "ticket_count", "avg_resolution", "pending_count", "tech_ticket_ratio",
    ]
    for column in zero_fill_columns:
        profile[column] = profile[column].fillna(0)

    profile["avg_satisfaction"] = profile["avg_satisfaction"].fillna(
        profile["avg_satisfaction"].mean()
    )

    profile["tenure_days"] = (end_date - profile["signup_date"]).dt.days

    pending_ratio = profile["pending_count"] / profile["ticket_count"].replace(0, np.nan)
    profile["pending_ratio"] = pending_ratio.fillna(0)

    engagement_index = (
        profile["avg_login"].rank(pct=True)
        + profile["avg_session"].rank(pct=True)
        + profile["avg_feature_usage"].rank(pct=True)
    ) / 3
    profile["engagement_index"] = engagement_index

    # ---- Continuous, noisy risk score (not a deterministic threshold) ----

    risk_score = (
        0.22 * (1 - engagement_index)
        + 0.13 * (1 - profile["txn_count"].rank(pct=True))
        + 0.10 * profile["failed_rate"].rank(pct=True)
        + 0.13 * profile["ticket_count"].rank(pct=True)
        + 0.14 * (1 - profile["avg_satisfaction"].rank(pct=True))
        + 0.08 * profile["avg_resolution"].rank(pct=True)
        + 0.08 * profile["pending_ratio"].rank(pct=True)
        + 0.07 * (1 - profile["tenure_days"].rank(pct=True))
    )

    plan_adjustment = profile["plan"].map(
        {"Basic": 0.03, "Standard": 0.0, "Premium": -0.03}
    )
    contract_adjustment = profile["contract_type"].map(
        {"Monthly": 0.03, "Yearly": -0.03}
    )

    risk_score = risk_score + plan_adjustment + contract_adjustment
    risk_score = risk_score + np.random.normal(0, 0.08, size=len(profile))

    centered_risk = risk_score - risk_score.mean()

    # Calibrate the logistic intercept so overall churn probability lands
    # near 20% (within the 15-25% target band), then sample probabilistically.
    slope = 3.0
    target_mean_probability = 0.20

    intercept_lo, intercept_hi = -6.0, 2.0
    for _ in range(40):
        intercept_mid = (intercept_lo + intercept_hi) / 2
        mean_probability = (
            1 / (1 + np.exp(-(intercept_mid + slope * centered_risk)))
        ).mean()

        if mean_probability < target_mean_probability:
            intercept_lo = intercept_mid
        else:
            intercept_hi = intercept_mid

    intercept = (intercept_lo + intercept_hi) / 2
    profile["churn_probability"] = 1 / (1 + np.exp(-(intercept + slope * centered_risk)))

    churn_flag = np.random.rand(len(profile)) < profile["churn_probability"]
    churned_profile = profile.loc[churn_flag].copy()

    # ---- Churn dates: skewed toward the later part of each customer's ----
    # ---- lifecycle, always after signup_date and on/before 2026-06-30. ----

    lifecycle_fraction = np.random.beta(2.5, 1.5, size=len(churned_profile))
    days_offset = np.maximum(
        1,
        np.round(lifecycle_fraction * churned_profile["tenure_days"]).astype(int)
    )

    churned_profile["churn_date"] = (
        churned_profile["signup_date"] + pd.to_timedelta(days_offset, unit="D")
    )
    churned_profile["churn_date"] = churned_profile["churn_date"].clip(upper=end_date)

    # ---- Churn reason: weighted, probabilistic, multi-signal assignment ----

    reasons = [
        "Low Engagement",
        "Payment Issues",
        "Poor Experience",
        "Technical Problems",
        "Customer Choice",
        "Other",
    ]

    churn_reasons = []

    for _, row in churned_profile.iterrows():

        weights = np.array([
            1 - row["engagement_index"],          # Low Engagement
            row["failed_rate"],                     # Payment Issues
            1 - (row["avg_satisfaction"] / 5),       # Poor Experience
            row["tech_ticket_ratio"],                # Technical Problems
            0.5,                                     # Customer Choice (baseline)
            0.3,                                     # Other (baseline)
        ])
        weights = np.clip(weights, 0.01, None)
        probabilities = weights / weights.sum()

        churn_reasons.append(np.random.choice(reasons, p=probabilities))

    churned_profile["churn_reason"] = churn_reasons

    churn_events = churned_profile.reset_index()[
        ["customer_id", "churn_date", "churn_reason"]
    ]

    churn_events = churn_events.sort_values("customer_id").reset_index(drop=True)

    return churn_events


def main():
    """Generate and save synthetic datasets."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    # Generate customers
    customers = generate_customers(NUM_CUSTOMERS)

    customers_path = OUTPUT_DIRECTORY / "customers.csv"

    customers.to_csv(
        customers_path,
        index=False
    )

    print(f"Generated {len(customers):,} customers.")
    print(f"Saved to: {customers_path}")

    # Generate transactions
    transactions = generate_transactions(customers)

    transactions_path = OUTPUT_DIRECTORY / "transactions.csv"

    transactions.to_csv(
        transactions_path,
        index=False
    )

    print(f"Generated {len(transactions):,} transactions.")
    print(f"Saved to: {transactions_path}")

    # Generate usage
    usage = generate_usage(customers)

    usage_path = OUTPUT_DIRECTORY / "usage.csv"

    usage.to_csv(
        usage_path,
        index=False
    )

    print(f"Generated {len(usage):,} usage records.")
    print(f"Saved to: {usage_path}")

    # Generate support tickets
    support_tickets = generate_support_tickets(customers)

    support_tickets_path = OUTPUT_DIRECTORY / "support_tickets.csv"

    support_tickets.to_csv(
        support_tickets_path,
        index=False
    )

    print(f"Generated {len(support_tickets):,} support tickets.")
    print(f"Saved to: {support_tickets_path}")

    # Generate churn events
    churn_events = generate_churn_events(
        customers,
        transactions,
        usage,
        support_tickets
    )

    churn_events_path = OUTPUT_DIRECTORY / "churn_events.csv"

    churn_events.to_csv(
        churn_events_path,
        index=False
    )

    print(f"Generated {len(churn_events):,} churn events.")
    print(f"Saved to: {churn_events_path}")


if __name__ == "__main__":
    main()