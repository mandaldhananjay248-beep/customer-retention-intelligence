"""Phase 14: thread-safe in-memory metrics collectors.

Two independent collectors, matching the distinction the Phase 14 spec
draws between generic HTTP traffic and model inference:

- `RequestMetrics` -- every HTTP request/response (all endpoints,
  including /health, /model-info, and the /monitoring/* endpoints
  themselves), recorded by api/main.py's RequestMonitoringMiddleware.
- `PredictionMetrics` -- only the four inference endpoints (/predict,
  /predict/explain, /recommend, /customer-analysis), recorded explicitly
  by their route handlers after api.services.ModelService already
  produced the real result. Nothing here computes a prediction; it only
  observes one that already happened.

Storage is in-process memory only, on purpose: this is a lightweight,
interview-appropriate monitor, not a production metrics database. Counts
are exact and unbounded; latency/probability samples are kept in a
bounded rolling window (oldest dropped first) so memory can't grow
without limit, which means percentiles reflect the most recent window,
while totals/averages-since-start reflect the process's entire lifetime.
Everything resets on process restart -- see README.md's Monitoring
section for this trade-off.

Nothing here stores a customer identifier, a raw feature row, or any
value that could not be republished safely in a monitoring endpoint.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from typing import Deque

DEFAULT_ROLLING_WINDOW = 1000


def _percentile(sorted_values: list[float], p: float) -> float | None:
    """Nearest-rank percentile over an already-sorted list."""
    n = len(sorted_values)
    if n == 0:
        return None
    index = min(n - 1, max(0, round(p / 100 * (n - 1))))
    return sorted_values[index]


class RequestMetrics:
    """Counters and latency stats for every HTTP request the API serves."""

    def __init__(self, rolling_window: int = DEFAULT_ROLLING_WINDOW) -> None:
        self._lock = threading.Lock()
        self._rolling_window = rolling_window
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._by_endpoint: Counter[str] = Counter()
        self._by_method: Counter[str] = Counter()
        self._by_status: Counter[int] = Counter()
        self._latency_sum_ms = 0.0
        self._latency_min_ms: float | None = None
        self._latency_max_ms: float | None = None
        self._recent_latencies_ms: Deque[float] = deque(maxlen=rolling_window)

    def record_request(self, method: str, path: str, status_code: int, duration_ms: float) -> None:
        """Called once per completed HTTP request. A 2xx/3xx status counts
        as successful; anything else (4xx/5xx) counts as failed.
        """
        with self._lock:
            self._total_requests += 1
            if 200 <= status_code < 400:
                self._successful_requests += 1
            else:
                self._failed_requests += 1
            self._by_endpoint[path] += 1
            self._by_method[method] += 1
            self._by_status[status_code] += 1
            self._latency_sum_ms += duration_ms
            self._latency_min_ms = duration_ms if self._latency_min_ms is None else min(self._latency_min_ms, duration_ms)
            self._latency_max_ms = duration_ms if self._latency_max_ms is None else max(self._latency_max_ms, duration_ms)
            self._recent_latencies_ms.append(duration_ms)

    def snapshot(self) -> dict:
        with self._lock:
            total = self._total_requests
            avg_ms = (self._latency_sum_ms / total) if total else None
            sorted_recent = sorted(self._recent_latencies_ms)

            return {
                "total_requests": total,
                "successful_requests": self._successful_requests,
                "failed_requests": self._failed_requests,
                "requests_by_endpoint": dict(self._by_endpoint),
                "requests_by_method": dict(self._by_method),
                "requests_by_status": {str(code): count for code, count in self._by_status.items()},
                "latency_ms": {
                    "avg": avg_ms,
                    "min": self._latency_min_ms,
                    "max": self._latency_max_ms,
                    "p50": _percentile(sorted_recent, 50),
                    "p95": _percentile(sorted_recent, 95),
                    "p99": _percentile(sorted_recent, 99),
                    "sample_size": len(sorted_recent),
                    "window_size": self._rolling_window,
                },
            }


class PredictionMetrics:
    """Counters and rolling stats for model inference outcomes --
    separate from RequestMetrics since not every HTTP request scores a
    customer (e.g. /health, /model-info, /monitoring/*).
    """

    def __init__(self, classification_threshold: float, rolling_window: int = DEFAULT_ROLLING_WINDOW) -> None:
        self._lock = threading.Lock()
        self._threshold = classification_threshold
        self._rolling_window = rolling_window
        self._total_predictions = 0
        self._successful_predictions = 0
        self._failed_predictions = 0
        self._positive_predictions = 0
        self._negative_predictions = 0
        self._recent_probabilities: Deque[float] = deque(maxlen=rolling_window)
        self._recent_inference_ms: Deque[float] = deque(maxlen=rolling_window)
        self._last_successful_inference_at: float | None = None
        # Data-quality signal: how many of an accepted request's features
        # were null (legitimately missing, per Phase 11's schema design)
        # -- NOT a re-validation of what Pydantic already enforces (missing
        # keys, wrong types, NaN/Infinity are all already rejected with a
        # 422 before a request ever reaches here; see api/schemas.py).
        self._null_fields_seen = 0
        self._total_fields_seen = 0

    def record_success(
        self, probability: float, duration_ms: float, null_field_count: int = 0, total_field_count: int = 0
    ) -> None:
        with self._lock:
            self._total_predictions += 1
            self._successful_predictions += 1
            if probability >= self._threshold:
                self._positive_predictions += 1
            else:
                self._negative_predictions += 1
            self._recent_probabilities.append(probability)
            self._recent_inference_ms.append(duration_ms)
            self._last_successful_inference_at = time.time()
            self._null_fields_seen += null_field_count
            self._total_fields_seen += total_field_count

    def record_failure(self) -> None:
        with self._lock:
            self._total_predictions += 1
            self._failed_predictions += 1

    def recent_probabilities(self) -> list[float]:
        with self._lock:
            return list(self._recent_probabilities)

    def snapshot(self) -> dict:
        with self._lock:
            probabilities = list(self._recent_probabilities)
            inference_times = list(self._recent_inference_ms)
            avg_probability = sum(probabilities) / len(probabilities) if probabilities else None
            avg_inference_ms = sum(inference_times) / len(inference_times) if inference_times else None
            null_rate = (self._null_fields_seen / self._total_fields_seen) if self._total_fields_seen else None

            return {
                "total_predictions": self._total_predictions,
                "successful_predictions": self._successful_predictions,
                "failed_predictions": self._failed_predictions,
                "positive_predictions": self._positive_predictions,
                "negative_predictions": self._negative_predictions,
                "classification_threshold": self._threshold,
                "average_predicted_probability": avg_probability,
                "average_inference_time_ms": avg_inference_ms,
                "last_successful_inference_at": self._last_successful_inference_at,
                "sample_size": len(probabilities),
                "average_null_field_rate": null_rate,
            }
