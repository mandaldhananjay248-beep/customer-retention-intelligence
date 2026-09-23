"""Phase 14: centralized logging configuration.

Replaces the ad-hoc `logging.basicConfig(...)` previously inline in
api/main.py with one place to configure format/level for the whole
service, plus a dedicated "api.access" logger for the one-line-per-request
summary api/main.py's RequestMonitoringMiddleware emits.

What gets logged, deliberately: HTTP method, path, status code, duration,
and a request id. What never gets logged: request/response bodies, query
parameters, or headers -- which is exactly where a customer's raw feature
values, a customer_id-bearing payload, or a credential could end up. This
matches the logging discipline Phase 11 already established (its
per-endpoint `logger.info(...)` calls only ever include `customer_id`,
never the full payload) -- Phase 14 extends that discipline uniformly to
every request via one middleware, rather than each endpoint repeating it.
"""

from __future__ import annotations

import logging

ACCESS_LOGGER_NAME = "api.access"
REQUEST_ID_HEADER = "X-Request-ID"


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent: safe to call more than once (e.g. once at module import
    and again in a test) -- logging.basicConfig no-ops if handlers already
    exist.
    """
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def get_access_logger() -> logging.Logger:
    return logging.getLogger(ACCESS_LOGGER_NAME)


def log_request(method: str, path: str, status_code: int, duration_ms: float, request_id: str) -> None:
    """One structured line per HTTP request. Never includes the request or
    response body -- see this module's docstring.
    """
    get_access_logger().info(
        "request method=%s path=%s status=%s duration_ms=%.1f request_id=%s",
        method, path, status_code, duration_ms, request_id,
    )
