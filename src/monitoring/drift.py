"""Phase 14: lightweight drift monitoring.

**Baseline.** The reference distribution is the Phase 7 TRAIN split of
data/processed/customer_features.parquet -- the same data the registered
model was fit on, and the same split Phase 8's SHAP background comes from
(src.models.explainability.build_background). Comparing live inference
requests against this exact baseline answers the practically relevant
question: "does live traffic still look like what the model was trained
on?" This baseline choice, and the feature-selection choice below, were
made after inspecting the project's actual reports (see
notebooks/14_monitoring_observability.ipynb) rather than picked
arbitrarily.

**Which features to monitor.** Monitoring all ~130 raw features would be
noisy and expensive for an interview-scale project, so only a handful of
numeric features are tracked -- chosen as the top-N by Phase 8's own SHAP
importance ranking (reports/shap_feature_importance.csv). This is a
defensible, non-arbitrary criterion: the features that matter most to the
model's predictions are the ones most worth watching for drift. All
one-hot-encoded categorical columns (country, plan, contract_type,
acquisition_channel) are monitored directly on their raw category
frequencies.

**Metric: Population Stability Index (PSI)**, a standard, well-understood
metric in churn/credit-risk-style monitoring, applied uniformly to (a)
the monitored numeric features (binned into baseline deciles), (b) the
monitored categorical features (binned by category), and (c) the model's
own predicted-probability distribution. Using one metric family for all
three keeps the implementation small and easy to explain rather than
mixing several different statistical tests.

PSI thresholds used below are the standard industry convention (not
invented here):
    PSI < 0.10            : no significant shift
    0.10 <= PSI < 0.25     : moderate shift, worth investigating
    PSI >= 0.25            : significant shift

**What PSI does NOT mean.** A PSI-flagged shift is a signal that the
input population or the model's own output distribution has changed --
it is NOT proof that the model's predictive performance has degraded.
Confirming actual performance degradation requires ground-truth labels
(churn_next_30_days), which are not available at inference time. Treat
drift as "worth investigating," never as a verdict.

Storage is in-process memory only: a small bounded rolling window of the
monitored feature values and predictions (never a full feature row, never
a customer identifier), reset on process restart.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque

import numpy as np
import pandas as pd

PSI_MODERATE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25
DEFAULT_TOP_N_NUMERIC_FEATURES = 8
DEFAULT_ROLLING_WINDOW = 200
MIN_SAMPLES_FOR_DRIFT = 30
_NUMBER_OF_BASELINE_BINS = 10


def classify_psi(psi_value: float) -> str:
    if psi_value >= PSI_SIGNIFICANT_THRESHOLD:
        return "significant_shift"
    if psi_value >= PSI_MODERATE_THRESHOLD:
        return "moderate_shift"
    return "no_significant_shift"


def compute_psi(baseline_counts: np.ndarray, comparison_counts: np.ndarray) -> float:
    """Population Stability Index between two distributions given as raw
    counts over the SAME bins/categories (same array length and order).
    A small epsilon avoids log(0)/division-by-zero for empty bins.
    """
    baseline_props = np.clip(baseline_counts / baseline_counts.sum(), 1e-6, None)
    comparison_props = np.clip(comparison_counts / comparison_counts.sum(), 1e-6, None)
    return float(np.sum((comparison_props - baseline_props) * np.log(comparison_props / baseline_props)))


def select_top_numeric_features_by_shap(
    shap_importance: pd.DataFrame, numeric_columns: list[str], top_n: int = DEFAULT_TOP_N_NUMERIC_FEATURES
) -> list[str]:
    """reports/shap_feature_importance.csv ranks TRANSFORMED features,
    which includes one-hot categorical dummy names (e.g. "plan_Premium")
    alongside raw numeric names -- only the numeric ones are usable here,
    so this filters to `numeric_columns` before taking the top N.
    """
    ranked = shap_importance.sort_values("mean_abs_shap", ascending=False)["feature"].tolist()
    numeric_set = set(numeric_columns)
    return [feature for feature in ranked if feature in numeric_set][:top_n]


class _NumericBaseline:
    __slots__ = ("bin_edges", "counts")

    def __init__(self, bin_edges: np.ndarray, counts: np.ndarray) -> None:
        self.bin_edges = bin_edges
        self.counts = counts


class DriftMonitor:
    """Holds the one-time-computed baseline (from TRAIN) plus a small
    in-memory rolling window of recent live requests' monitored feature
    values and predictions.
    """

    def __init__(
        self,
        train_features: pd.DataFrame,
        numeric_features: list[str],
        categorical_features: list[str],
        rolling_window: int = DEFAULT_ROLLING_WINDOW,
    ) -> None:
        self._lock = threading.Lock()
        self.rolling_window = rolling_window
        self.categorical_features = list(categorical_features)

        self._numeric_baselines: dict[str, _NumericBaseline] = {}
        for feature in numeric_features:
            baseline = self._build_numeric_baseline(train_features[feature])
            if baseline is not None:
                self._numeric_baselines[feature] = baseline

        self._categorical_baselines: dict[str, pd.Series] = {
            feature: train_features[feature].value_counts() for feature in self.categorical_features
        }

        self._recent_numeric: dict[str, Deque[float]] = {
            feature: deque(maxlen=rolling_window) for feature in self._numeric_baselines
        }
        self._recent_categorical: dict[str, Deque[str]] = {
            feature: deque(maxlen=rolling_window) for feature in self.categorical_features
        }
        self._recent_predictions: Deque[float] = deque(maxlen=rolling_window)

        self._prediction_baseline: _NumericBaseline | None = None

    @staticmethod
    def _build_numeric_baseline(values: pd.Series) -> _NumericBaseline | None:
        clean_values = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
        if len(clean_values) < 10:
            return None
        bin_edges = np.unique(np.quantile(clean_values, np.linspace(0, 1, _NUMBER_OF_BASELINE_BINS + 1)))
        if len(bin_edges) < 3:
            return None  # degenerate/near-constant feature -- PSI isn't meaningful

        # np.histogram silently DROPS any value outside [edges[0], edges[-1]].
        # Using the exact sample min/max as the outer edges (from the
        # quantile call above) means a later comparison sample with a new
        # extreme -- entirely plausible sampling variation, not drift -- would
        # be silently excluded, corrupting the comparison's proportions (and,
        # if a badly shifted sample falls entirely outside the range, divides
        # by zero into NaN). Extending the outer edges to +/-inf guarantees
        # every value, however extreme, lands in the first/last bin instead.
        open_ended_edges = bin_edges.copy()
        open_ended_edges[0] = -np.inf
        open_ended_edges[-1] = np.inf

        counts, _ = np.histogram(clean_values, bins=open_ended_edges)
        return _NumericBaseline(open_ended_edges, counts.astype(float))

    def set_prediction_baseline(self, baseline_probabilities: np.ndarray) -> None:
        """Call once at startup with predictions on a held-out split
        (validation, not train, so the baseline reflects out-of-sample
        behavior) -- never per request.
        """
        baseline = self._build_numeric_baseline(pd.Series(baseline_probabilities))
        with self._lock:
            self._prediction_baseline = baseline

    @property
    def monitored_numeric_features(self) -> list[str]:
        return list(self._numeric_baselines.keys())

    def record_observation(self, feature_payload: dict, probability: float | None) -> None:
        """Record ONLY the small monitored subset of one request's feature
        values (never the full raw feature row, never a customer
        identifier) plus its predicted probability, if available.
        """
        with self._lock:
            for feature, window in self._recent_numeric.items():
                value = feature_payload.get(feature)
                if value is None:
                    continue
                try:
                    window.append(float(value))
                except (TypeError, ValueError):
                    continue
            for feature, window in self._recent_categorical.items():
                value = feature_payload.get(feature)
                if value is not None:
                    window.append(str(value))
            if probability is not None:
                self._recent_predictions.append(float(probability))

    def _numeric_feature_drift_locked(self) -> list[dict]:
        results = []
        for feature, baseline in self._numeric_baselines.items():
            recent = list(self._recent_numeric[feature])
            if len(recent) < MIN_SAMPLES_FOR_DRIFT:
                results.append({"feature": feature, "psi": None, "status": "insufficient_data", "sample_size": len(recent)})
                continue
            counts, _ = np.histogram(np.array(recent), bins=baseline.bin_edges)
            psi_value = compute_psi(baseline.counts, counts.astype(float))
            results.append(
                {"feature": feature, "psi": round(psi_value, 4), "status": classify_psi(psi_value), "sample_size": len(recent)}
            )
        return results

    def _categorical_feature_drift_locked(self) -> list[dict]:
        results = []
        for feature, baseline_value_counts in self._categorical_baselines.items():
            recent = list(self._recent_categorical[feature])
            if len(recent) < MIN_SAMPLES_FOR_DRIFT:
                results.append({"feature": feature, "psi": None, "status": "insufficient_data", "sample_size": len(recent)})
                continue
            categories = sorted(set(baseline_value_counts.index) | set(recent))
            baseline_counts = np.array([baseline_value_counts.get(category, 0) for category in categories], dtype=float)
            recent_counter = pd.Series(recent).value_counts()
            recent_counts = np.array([recent_counter.get(category, 0) for category in categories], dtype=float)
            psi_value = compute_psi(baseline_counts, recent_counts)
            results.append(
                {"feature": feature, "psi": round(psi_value, 4), "status": classify_psi(psi_value), "sample_size": len(recent)}
            )
        return results

    def _prediction_drift_locked(self) -> dict:
        recent = list(self._recent_predictions)
        if self._prediction_baseline is None:
            return {"psi": None, "status": "no_baseline", "sample_size": len(recent)}
        if len(recent) < MIN_SAMPLES_FOR_DRIFT:
            return {"psi": None, "status": "insufficient_data", "sample_size": len(recent)}
        counts, _ = np.histogram(np.array(recent), bins=self._prediction_baseline.bin_edges)
        psi_value = compute_psi(self._prediction_baseline.counts, counts.astype(float))
        return {"psi": round(psi_value, 4), "status": classify_psi(psi_value), "sample_size": len(recent)}

    def snapshot(self) -> dict:
        with self._lock:
            feature_drift = self._numeric_feature_drift_locked() + self._categorical_feature_drift_locked()
            prediction_drift = self._prediction_drift_locked()
            monitored_numeric = self.monitored_numeric_features

        drifted_features = [r["feature"] for r in feature_drift if r["status"] in ("moderate_shift", "significant_shift")]

        return {
            "baseline_source": "Phase 7 TRAIN split of data/processed/customer_features.parquet",
            "monitored_numeric_features": monitored_numeric,
            "monitored_categorical_features": self.categorical_features,
            "feature_drift": feature_drift,
            "drifted_feature_count": len(drifted_features),
            "drifted_features": drifted_features,
            "prediction_drift": prediction_drift,
            "psi_thresholds": {"moderate": PSI_MODERATE_THRESHOLD, "significant": PSI_SIGNIFICANT_THRESHOLD},
            "note": (
                "PSI measures a distributional shift versus the Phase 7 training baseline. "
                "It is a monitoring signal for investigation, not proof that model performance "
                "has degraded -- confirming that requires ground-truth outcomes."
            ),
        }
