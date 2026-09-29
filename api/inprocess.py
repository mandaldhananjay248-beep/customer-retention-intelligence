"""Phase 15: serve this FastAPI app in-process, without a network hop, for
single-process deployments such as Streamlit Community Cloud.

The dashboard's API client (dashboard/api_client.py, in
DASHBOARD_API_MODE=inprocess) sends the same requests it would send over
HTTP to the very same `api.main.app` through Starlette's in-memory ASGI
client. So every response comes from the existing routes, schemas, model
service, SHAP and recommendation code, and the Phase 14 monitoring
middleware -- nothing is reimplemented here. The app's lifespan (the one
model load) runs once per process, on first use.
"""

from __future__ import annotations

import threading
from typing import Any

from fastapi.testclient import TestClient

from api.main import app

_lock = threading.Lock()
_client: TestClient | None = None


def get_client() -> TestClient:
    """The process-wide client, started (lifespan run, model loaded) on
    first use. raise_server_exceptions=False makes an unhandled error come
    back as the same HTTP 500 response a real server would send.
    """
    global _client
    with _lock:
        if _client is None:
            client = TestClient(app, raise_server_exceptions=False)
            client.__enter__()  # runs api.main's lifespan: loads the model service once
            _client = client
        return _client


def request(method: str, path: str, json_body: dict[str, Any] | None = None):
    """Send one request to the in-process app and return its response
    (status_code / json(), like a `requests` response). Requests are
    serialized, as they would be by a single uvicorn worker process.
    """
    client = get_client()
    with _lock:
        return client.request(method, path, json=json_body)
