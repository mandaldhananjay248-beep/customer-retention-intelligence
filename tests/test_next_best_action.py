"""Tests for src/recommendations/next_best_action.py and rules.py.

Uses the real configs/recommendation_config.yaml (this module has no
hard-coded fallback thresholds, so testing against a fake config would
not actually exercise the real business rules) together with small,
hand-built synthetic customer rows -- fast, deterministic, and each one
built to sit clearly on one side of a specific threshold.
"""

import numpy as np
import pandas as pd
import pytest

from src.recommendations import next_best_action as nba
from src.recommendations import rules


@pytest.fixture(scope="module")
def config() -> dict:
    return nba.load_config()


def make_row(**overrides) -> dict:
    """A customer row with every rule feature at a clearly-non-triggering
    value, so a test can override just the one or two features it cares
    about and know nothing else will accidentally also trigger.
    """
    base = {
        "customer_id": "C_TEST",
        "snapshot_date": pd.Timestamp("2026-01-31"),
        "active_days_90d": 50.0,
        "avg_active_minutes_90d": 60.0,
        "active_minutes_pct_change_30d": 10.0,
        "login_count_pct_change_30d": 10.0,
        "feature_usage_count_pct_change_30d": 10.0,
        "pending_ticket_count_90d": 0.0,
        "high_critical_ticket_count_90d": 0.0,
        "avg_satisfaction_90d": 4.5,
        "txn_failed_count_90d": 0.0,
        "txn_failed_rate_90d": 0.0,
        "days_since_last_successful_transaction": 3.0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def test_load_config_has_required_sections(config: dict) -> None:
    assert {"risk_thresholds", "priority_thresholds", "signal_thresholds", "actions"} <= set(config.keys())
    assert config["risk_thresholds"]["low"] < config["risk_thresholds"]["medium"] < config["risk_thresholds"]["high"]


def test_load_config_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        nba.load_config(path=tmp_path / "does_not_exist.yaml")


def test_load_config_rejects_incomplete_config(tmp_path) -> None:
    bad_config_path = tmp_path / "bad_config.yaml"
    bad_config_path.write_text("risk_thresholds:\n  low: 0.2\n")
    with pytest.raises(ValueError):
        nba.load_config(path=bad_config_path)


# ---------------------------------------------------------------------------
# Risk level thresholds and boundary conditions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "probability,expected",
    [
        (0.0, "low"),
        (0.19, "low"),
        (0.20, "medium"),  # boundary: low threshold is exclusive on the low side
        (0.49, "medium"),
        (0.50, "high"),
        (0.79, "high"),
        (0.80, "critical"),
        (1.0, "critical"),
    ],
)
def test_assign_risk_level_boundaries(probability: float, expected: str, config: dict) -> None:
    assert nba.assign_risk_level(probability, config["risk_thresholds"]) == expected


@pytest.mark.parametrize("bad_probability", [-0.01, 1.01, float("nan"), None])
def test_assign_risk_level_rejects_invalid_probability(bad_probability, config: dict) -> None:
    with pytest.raises(ValueError):
        nba.assign_risk_level(bad_probability, config["risk_thresholds"])


@pytest.mark.parametrize("bad_probability", [-0.5, 2.0, float("nan")])
def test_assign_priority_rejects_invalid_probability(bad_probability, config: dict) -> None:
    with pytest.raises(ValueError):
        nba.assign_priority_from_probability(bad_probability, config["priority_thresholds"])


def test_bump_priority_caps_at_urgent() -> None:
    assert nba.bump_priority("urgent") == "urgent"
    assert nba.bump_priority("high") == "urgent"
    assert nba.bump_priority("low", steps=10) == "urgent"


# ---------------------------------------------------------------------------
# Signal rules (rules.py) -- every rule must reference only real features
# ---------------------------------------------------------------------------

def test_low_engagement_requires_both_conditions(config: dict) -> None:
    thresholds = config["signal_thresholds"]["low_engagement"]
    only_one_low = make_row(active_days_90d=thresholds["active_days_90d_max"])  # minutes still high
    triggered, _ = rules.check_low_engagement(only_one_low, config)
    assert triggered is False

    both_low = make_row(
        active_days_90d=thresholds["active_days_90d_max"],
        avg_active_minutes_90d=thresholds["avg_active_minutes_90d_max"],
    )
    triggered, matched = rules.check_low_engagement(both_low, config)
    assert triggered is True
    assert any("active_days_90d" in m for m in matched)
    assert any("avg_active_minutes_90d" in m for m in matched)


def test_reduced_usage_triggers_on_any_single_decline(config: dict) -> None:
    threshold = config["signal_thresholds"]["reduced_usage"]["active_minutes_pct_change_30d_max"]
    row = make_row(active_minutes_pct_change_30d=threshold)
    triggered, matched = rules.check_reduced_usage(row, config)
    assert triggered is True
    assert len(matched) == 1
    assert "active_minutes_pct_change_30d" in matched[0]


def test_support_problems_triggers_on_pending_tickets(config: dict) -> None:
    threshold = config["signal_thresholds"]["support_problems"]["pending_ticket_count_90d_min"]
    row = make_row(pending_ticket_count_90d=threshold)
    triggered, matched = rules.check_support_problems(row, config)
    assert triggered is True
    assert any("pending_ticket_count_90d" in m for m in matched)


def test_payment_problems_triggers_on_failed_transactions(config: dict) -> None:
    threshold = config["signal_thresholds"]["payment_problems"]["txn_failed_count_90d_min"]
    row = make_row(txn_failed_count_90d=threshold)
    triggered, matched = rules.check_payment_problems(row, config)
    assert triggered is True
    assert any("txn_failed_count_90d" in m for m in matched)


def test_no_signal_triggers_on_healthy_row(config: dict) -> None:
    row = make_row()  # every feature deliberately non-triggering
    signals = rules.evaluate_signals(row, config)
    assert all(not is_triggered for is_triggered, _ in signals.values())


def test_missing_feature_does_not_trigger_or_raise(config: dict) -> None:
    """A customer with no support-ticket history at all (a real, common
    case in this dataset -- ~30% of customers per Phase 1 EDA) must not
    crash or spuriously trigger a support-problems flag.
    """
    row = make_row()
    del row["pending_ticket_count_90d"]
    del row["avg_satisfaction_90d"]

    triggered, matched = rules.check_support_problems(row, config)
    assert triggered is False
    assert matched == []


def test_nan_feature_does_not_trigger(config: dict) -> None:
    row = make_row(avg_satisfaction_90d=np.nan, pending_ticket_count_90d=np.nan, high_critical_ticket_count_90d=np.nan)
    triggered, matched = rules.check_support_problems(row, config)
    assert triggered is False
    assert matched == []


def test_every_rule_references_only_real_dataset_features() -> None:
    """rules.REQUIRED_SIGNAL_FEATURES must exactly match the columns the
    Phase 4 feature dataset actually provides -- proof that nothing here
    is checking a feature that was invented rather than engineered.
    """
    from src.models import baseline as bl

    features = bl.load_feature_dataset()
    missing = set(rules.REQUIRED_SIGNAL_FEATURES) - set(features.columns)
    assert not missing, f"Rules reference non-existent features: {missing}"


# ---------------------------------------------------------------------------
# determine_action / generate_customer_action -- reasons must match the
# actual triggered rule, and no action is forced onto every high-risk row
# ---------------------------------------------------------------------------

def test_healthy_low_risk_customer_gets_no_intervention(config: dict) -> None:
    row = make_row()
    result = nba.generate_customer_action(row, probability=0.05, config=config)

    assert result["risk_level"] == "low"
    assert result["action_key"] == "no_intervention"
    assert result["primary_action"] == config["actions"]["no_intervention"]
    assert result["supporting_features"] == []
    assert result["priority"] == "low"


def test_medium_risk_no_signal_gets_no_intervention_with_explanatory_reason(config: dict) -> None:
    row = make_row()
    result = nba.generate_customer_action(row, probability=0.35, config=config)

    assert result["risk_level"] == "medium"
    assert result["action_key"] == "no_intervention"
    assert "insufficient evidence" in result["action_reason"]


def test_single_signal_maps_to_its_specific_action_not_generic(config: dict) -> None:
    payment_threshold = config["signal_thresholds"]["payment_problems"]["txn_failed_count_90d_min"]
    row = make_row(txn_failed_count_90d=payment_threshold)
    result = nba.generate_customer_action(row, probability=0.4, config=config)

    assert result["action_key"] == "billing_payment_assistance"
    assert "payment" in result["action_reason"].lower()
    assert any("txn_failed_count_90d" in f for f in result["supporting_features"])


def test_different_single_signals_yield_different_actions(config: dict) -> None:
    """Guards against the "every high-risk customer gets the same action"
    failure mode explicitly called out in the Phase 9 spec.
    """
    support_threshold = config["signal_thresholds"]["support_problems"]["pending_ticket_count_90d_min"]
    usage_threshold = config["signal_thresholds"]["reduced_usage"]["active_minutes_pct_change_30d_max"]

    support_row = make_row(pending_ticket_count_90d=support_threshold)
    usage_row = make_row(active_minutes_pct_change_30d=usage_threshold)

    support_result = nba.generate_customer_action(support_row, probability=0.6, config=config)
    usage_result = nba.generate_customer_action(usage_row, probability=0.6, config=config)

    assert support_result["action_key"] == "customer_support_follow_up"
    assert usage_result["action_key"] == "re_engagement_campaign"
    assert support_result["action_key"] != usage_result["action_key"]


def test_high_risk_with_multiple_signals_gets_priority_retention_outreach(config: dict) -> None:
    row = make_row(
        pending_ticket_count_90d=config["signal_thresholds"]["support_problems"]["pending_ticket_count_90d_min"],
        txn_failed_count_90d=config["signal_thresholds"]["payment_problems"]["txn_failed_count_90d_min"],
    )
    result = nba.generate_customer_action(row, probability=0.85, config=config)

    assert result["risk_level"] == "critical"
    assert result["action_key"] == "priority_retention_outreach"
    assert result["priority"] == "urgent"
    assert "support_problems" in result["action_reason"]
    assert "payment_problems" in result["action_reason"]


def test_high_risk_single_signal_does_not_get_priority_retention_outreach(config: dict) -> None:
    """One triggered category alone -- even at critical risk -- should not
    escalate to the compound "priority_retention_outreach" action.
    """
    row = make_row(txn_failed_count_90d=config["signal_thresholds"]["payment_problems"]["txn_failed_count_90d_min"])
    result = nba.generate_customer_action(row, probability=0.9, config=config)

    assert result["risk_level"] == "critical"
    assert result["action_key"] == "billing_payment_assistance"


def test_priority_bumped_when_multiple_signals_triggered_at_lower_probability(config: dict) -> None:
    row = make_row(
        pending_ticket_count_90d=config["signal_thresholds"]["support_problems"]["pending_ticket_count_90d_min"],
        txn_failed_count_90d=config["signal_thresholds"]["payment_problems"]["txn_failed_count_90d_min"],
    )
    probability = 0.40  # base priority alone would be "low" (< priority_thresholds.medium=0.35 is false, so "medium")
    base_priority = nba.assign_priority_from_probability(probability, config["priority_thresholds"])
    result = nba.generate_customer_action(row, probability=probability, config=config)

    assert nba.PRIORITY_LEVELS.index(result["priority"]) == nba.PRIORITY_LEVELS.index(base_priority) + 1


def test_action_reason_mentions_the_actual_probability(config: dict) -> None:
    row = make_row()
    result = nba.generate_customer_action(row, probability=0.6789, config=config)
    assert "0.68" in result["action_reason"]


def test_model_signals_are_prefixed_and_included_in_supporting_features(config: dict) -> None:
    row = make_row(txn_failed_count_90d=config["signal_thresholds"]["payment_problems"]["txn_failed_count_90d_min"])
    result = nba.generate_customer_action(row, probability=0.5, config=config, model_signals=["avg_pages_viewed_90d"])
    assert "model_signal:avg_pages_viewed_90d" in result["supporting_features"]


def test_no_fabricated_customer_data(config: dict) -> None:
    """Every supporting-feature string must be traceable to a real value
    on the row it was generated from -- nothing invented.
    """
    row = make_row(pending_ticket_count_90d=3.0)
    result = nba.generate_customer_action(row, probability=0.6, config=config)
    for feature_string in result["supporting_features"]:
        if feature_string.startswith("model_signal:"):
            continue
        feature_name = feature_string.split("=")[0]
        assert feature_name in row, f"Reported feature '{feature_name}' is not on the source row"


def test_deterministic_recommendation_for_identical_input(config: dict) -> None:
    row = make_row(pending_ticket_count_90d=2.0, txn_failed_count_90d=2.0)
    first = nba.generate_customer_action(row, probability=0.77, config=config)
    second = nba.generate_customer_action(row, probability=0.77, config=config)
    assert first == second


# ---------------------------------------------------------------------------
# Batch generation and output schema
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_customers() -> pd.DataFrame:
    rows = [
        make_row(customer_id="C0001", snapshot_date=pd.Timestamp("2026-01-31")),  # healthy
        make_row(
            customer_id="C0002", snapshot_date=pd.Timestamp("2026-01-31"),
            pending_ticket_count_90d=2.0,
        ),  # support problem only
        make_row(
            customer_id="C0003", snapshot_date=pd.Timestamp("2026-01-31"),
            txn_failed_count_90d=3.0, pending_ticket_count_90d=2.0,
        ),  # compound
    ]
    return pd.DataFrame(rows)


class _FakePipeline:
    """A minimal stand-in exposing only predict_proba, so batch generation
    can be tested without loading the real ~20KB model file or its
    preprocessor -- this test suite cares about the recommendation logic,
    not re-testing Phase 7's own fitted model.
    """

    def __init__(self, probabilities: np.ndarray):
        self._probabilities = probabilities

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([1 - self._probabilities[:n], self._probabilities[:n]])


def test_generate_batch_recommendations_schema(synthetic_customers: pd.DataFrame, config: dict) -> None:
    pipeline = _FakePipeline(np.array([0.05, 0.55, 0.90]))
    feature_columns = ["active_days_90d"]  # _FakePipeline ignores contents, only needs row count

    recommendations = nba.generate_batch_recommendations(pipeline, synthetic_customers, feature_columns, config)

    assert len(recommendations) == 3
    assert set(nba.REQUIRED_OUTPUT_COLUMNS) <= set(recommendations.columns)
    assert recommendations["churn_probability"].between(0, 1).all()
    assert set(recommendations["risk_level"]) <= set(nba.RISK_LEVELS)
    assert set(recommendations["priority"]) <= set(nba.PRIORITY_LEVELS)


def test_generate_batch_recommendations_differentiates_customers(synthetic_customers: pd.DataFrame, config: dict) -> None:
    pipeline = _FakePipeline(np.array([0.05, 0.55, 0.90]))
    feature_columns = ["active_days_90d"]

    recommendations = nba.generate_batch_recommendations(pipeline, synthetic_customers, feature_columns, config)
    by_id = recommendations.set_index("customer_id")

    assert by_id.loc["C0001", "action_key"] == "no_intervention"
    assert by_id.loc["C0002", "action_key"] == "customer_support_follow_up"
    assert by_id.loc["C0003", "action_key"] == "priority_retention_outreach"


def test_save_recommendations_round_trips(synthetic_customers: pd.DataFrame, config: dict, tmp_path) -> None:
    pipeline = _FakePipeline(np.array([0.05, 0.55, 0.90]))
    feature_columns = ["active_days_90d"]
    recommendations = nba.generate_batch_recommendations(pipeline, synthetic_customers, feature_columns, config)

    original_reports_dir = nba.REPORTS_DIR
    nba.REPORTS_DIR = tmp_path
    try:
        output_path = nba.save_recommendations(recommendations, filename="test_recommendations.csv")
    finally:
        nba.REPORTS_DIR = original_reports_dir

    saved = pd.read_csv(output_path)
    assert set(nba.REQUIRED_OUTPUT_COLUMNS) <= set(saved.columns)
    assert len(saved) == 3
    import json
    assert isinstance(json.loads(saved.loc[1, "supporting_features"]), list)


def test_recommendation_summary_percentages_sum_to_100(synthetic_customers: pd.DataFrame, config: dict) -> None:
    pipeline = _FakePipeline(np.array([0.05, 0.55, 0.90]))
    feature_columns = ["active_days_90d"]
    recommendations = nba.generate_batch_recommendations(pipeline, synthetic_customers, feature_columns, config)

    summary = nba.build_recommendation_summary(recommendations)
    assert {"risk_level", "primary_action", "customer_count", "percentage_of_customers"} <= set(summary.columns)
    assert summary["customer_count"].sum() == len(recommendations)
    assert summary["percentage_of_customers"].sum() == pytest.approx(100.0, abs=0.01)


def test_generate_recommendation_single_row_matches_batch(synthetic_customers: pd.DataFrame, config: dict) -> None:
    pipeline = _FakePipeline(np.array([0.55]))
    feature_columns = ["active_days_90d"]
    single_row = synthetic_customers.iloc[[1]]

    result = nba.generate_recommendation(pipeline, single_row, feature_columns, config)
    assert result["customer_id"] == "C0002"
    assert result["action_key"] == "customer_support_follow_up"


def test_generate_recommendation_rejects_multi_row_input(synthetic_customers: pd.DataFrame, config: dict) -> None:
    pipeline = _FakePipeline(np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        nba.generate_recommendation(pipeline, synthetic_customers.iloc[:2], ["active_days_90d"], config)


# ---------------------------------------------------------------------------
# Real Phase 7 model integration (skipped if Phase 7 hasn't been run yet)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_pipeline_and_data():
    from src.models import baseline as bl
    from src.models.explainability import TUNED_MODEL_PATH, load_tuned_pipeline

    if not TUNED_MODEL_PATH.exists():
        pytest.skip(f"{TUNED_MODEL_PATH} not found -- run Phase 7 first")

    pipeline = load_tuned_pipeline()
    features = bl.load_feature_dataset()
    numeric_columns, categorical_columns = bl.get_feature_columns(features)
    feature_columns = numeric_columns + categorical_columns
    return pipeline, features, feature_columns


def test_real_model_batch_recommendations_have_valid_probabilities(real_pipeline_and_data, config: dict) -> None:
    pipeline, features, feature_columns = real_pipeline_and_data
    sample = features.sample(n=200, random_state=42)

    recommendations = nba.generate_batch_recommendations(pipeline, sample, feature_columns, config)

    assert recommendations["churn_probability"].between(0, 1).all()
    assert recommendations["customer_id"].isin(sample["customer_id"]).all()


def test_real_model_does_not_mutate_pipeline(real_pipeline_and_data, config: dict) -> None:
    """The Phase 7 model must never be refit or altered by this layer."""
    import copy

    pipeline, features, feature_columns = real_pipeline_and_data
    coef_before = copy.deepcopy(pipeline.named_steps["model"].coef_)

    sample = features.sample(n=50, random_state=42)
    nba.generate_batch_recommendations(pipeline, sample, feature_columns, config)

    np.testing.assert_array_equal(coef_before, pipeline.named_steps["model"].coef_)


def test_high_probability_real_customers_get_appropriate_priority(real_pipeline_and_data, config: dict) -> None:
    pipeline, features, feature_columns = real_pipeline_and_data
    sample = features.sample(n=500, random_state=42)

    recommendations = nba.generate_batch_recommendations(pipeline, sample, feature_columns, config)
    high_probability = recommendations[recommendations["churn_probability"] >= config["priority_thresholds"]["urgent"]]

    if len(high_probability) > 0:
        assert set(high_probability["priority"]) <= {"urgent"}
