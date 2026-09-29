"""Phase 15: end-to-end smoke test for a running API/dashboard deployment
-- Docker Compose locally or the CI Docker job.

    python scripts/smoke_test.py --api-url http://localhost:8000 \
        [--dashboard-url http://localhost:8501] [--wait 300]

Standard library only, so it runs anywhere without the project's
dependencies. The customer-analysis request uses the first row of the
packaged model's own MLflow input example (synthetic data, a placeholder
customer_id) and checks the returned churn probability against the value
the packaged model produces for it -- so a pass also confirms the deployed
model is the exact registered model. Exits non-zero on the first failure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_EXAMPLE_PATH = (
    PROJECT_ROOT / "models" / "registered" / "customer-churn-logistic-regression" / "1" / "input_example.json"
)
EXPECTED_MODEL = {
    "model_name": "customer-churn-logistic-regression",
    "model_version": "1",
    "model_type": "LogisticRegression",
    "C": 0.01,
    "class_weight": "balanced",
}
# The packaged model's predict_proba for INPUT_EXAMPLE_PATH's first row.
EXPECTED_CHURN_PROBABILITY = 0.45027374394397185


def request(method: str, url: str, body: dict | None = None, timeout: float = 60.0) -> tuple[int, dict | str, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status, raw, headers = response.status, response.read().decode(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        status, raw, headers = exc.code, exc.read().decode(), dict(exc.headers)
    try:
        return status, json.loads(raw), headers
    except json.JSONDecodeError:
        return status, raw, headers


def check(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL  {message}")
        sys.exit(1)
    print(f"ok    {message}")


def wait_until_healthy(url: str, wait_seconds: float) -> None:
    """Cold containers take a while to load the model -- poll first."""
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            status, _, _ = request("GET", url, timeout=30)
            if status == 200:
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            status = None
        if time.monotonic() >= deadline:
            check(False, f"{url} healthy within {wait_seconds:.0f}s (last status {status})")
        time.sleep(5)


def analysis_payload() -> dict:
    example = json.loads(INPUT_EXAMPLE_PATH.read_text())
    row = dict(zip(example["columns"], example["data"][0]))
    features = {k: None if isinstance(v, float) and math.isnan(v) else v for k, v in row.items()}
    return {"customer_id": "SMOKE-TEST-0001", "snapshot_date": "2026-01-01", **features}


def smoke_api(api_url: str) -> None:
    status, body, _ = request("GET", f"{api_url}/health")
    check(status == 200 and body.get("status") == "healthy", f"GET /health -> {status} {body}")

    status, body, _ = request("GET", f"{api_url}/model-info")
    check(status == 200 and body == EXPECTED_MODEL, f"GET /model-info -> {status} {body}")

    status, body, _ = request("GET", f"{api_url}/monitoring/health")
    check(
        status == 200 and body.get("model_status") == "available" and body.get("drift_monitoring_status") == "active",
        f"GET /monitoring/health -> {status} model_status={body.get('model_status')} "
        f"drift_monitoring_status={body.get('drift_monitoring_status')}",
    )

    _, before, _ = request("GET", f"{api_url}/monitoring/metrics")

    status, body, headers = request("POST", f"{api_url}/customer-analysis", analysis_payload())
    check(status == 200, f"POST /customer-analysis -> {status}")
    probability = body["prediction"]["churn_probability"]
    check(
        abs(probability - EXPECTED_CHURN_PROBABILITY) < 1e-9,
        f"churn_probability {probability} matches the packaged model ({EXPECTED_CHURN_PROBABILITY})",
    )
    explanation = body["explanation"]
    check(
        bool(explanation["top_positive_drivers"] or explanation["top_negative_drivers"]),
        f"SHAP explanation returned {len(explanation['top_positive_drivers'])} positive / "
        f"{len(explanation['top_negative_drivers'])} negative drivers",
    )
    check(bool(body["recommendation"].get("primary_action")), f"recommendation: {body['recommendation']['primary_action']!r}")
    request_id = {k.lower(): v for k, v in headers.items()}.get("x-request-id")
    check(bool(request_id), f"X-Request-ID response header present ({request_id})")

    status, _, _ = request("POST", f"{api_url}/predict", {"customer_id": "SMOKE-TEST-INVALID"})
    check(status == 422, f"invalid POST /predict rejected with 422 (got {status})")

    status, after, _ = request("GET", f"{api_url}/monitoring/metrics")
    check(status == 200, f"GET /monitoring/metrics -> {status}")
    check(after["total_requests"] > before["total_requests"], f"total_requests grew {before['total_requests']} -> {after['total_requests']}")
    check(
        after["predictions"]["total_predictions"] > before["predictions"]["total_predictions"],
        f"total_predictions grew {before['predictions']['total_predictions']} -> {after['predictions']['total_predictions']}",
    )
    check(after["failed_requests"] > before["failed_requests"], f"failed_requests recorded the 422 ({after['failed_requests']})")
    check(after["latency_ms"]["sample_size"] > 0, f"latency p95 {after['latency_ms']['p95']:.1f} ms")

    status, body, _ = request("GET", f"{api_url}/monitoring/drift")
    check(status == 200 and "feature_drift" in body, f"GET /monitoring/drift -> {status}, {body.get('drifted_feature_count')} drifted features")


def smoke_dashboard(dashboard_url: str) -> None:
    status, body, _ = request("GET", f"{dashboard_url}/_stcore/health")
    check(status == 200 and body == "ok", f"GET dashboard /_stcore/health -> {status} {body!r}")
    status, body, _ = request("GET", dashboard_url)
    check(status == 200 and "<html" in str(body).lower(), f"GET dashboard / -> {status}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a running API (and optionally dashboard).")
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--dashboard-url")
    parser.add_argument("--wait", type=float, default=0, help="seconds to wait for services to become healthy")
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")
    wait_until_healthy(f"{api_url}/health", args.wait)
    smoke_api(api_url)
    if args.dashboard_url:
        dashboard_url = args.dashboard_url.rstrip("/")
        wait_until_healthy(f"{dashboard_url}/_stcore/health", args.wait)
        smoke_dashboard(dashboard_url)
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
