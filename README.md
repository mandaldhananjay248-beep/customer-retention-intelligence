# Customer Retention Intelligence Platform

## Overview
To be implemented.

## Business Problem
To be implemented.

## Project Objectives
To be implemented.

## Solution Architecture
To be implemented.

## Data Sources
To be implemented.

## Data Pipeline
To be implemented.

## Exploratory Data Analysis
To be implemented.

## Feature Engineering
To be implemented.

## Machine Learning
To be implemented.

## Model Evaluation
Results will be added after model training.

## Explainability
To be implemented.

## Next Best Action
To be implemented.

## MLflow Experiment Tracking
To be implemented.

## FastAPI
To be implemented.

## Streamlit Dashboard

A business-facing dashboard (`dashboard/`) for churn prediction, explainability, and
retention recommendations, built on top of the Phase 11 FastAPI service.

**Architecture:**

```
Streamlit Dashboard -> HTTP -> FastAPI -> MLflow registered model -> Prediction / SHAP / Recommendation
```

The dashboard never loads the model (`.joblib` or MLflow) directly, computes a churn
probability itself, calls SHAP directly, or re-implements the next-best-action rules.
FastAPI remains the single source of truth for inference, explanations, and
recommendations. The dashboard reads `data/processed/customer_features.parquet` only
to let a user pick a real customer/snapshot and to build the API request payload from
that row's actual values.

**Running it locally:**

1. Start FastAPI:

   ```bash
   uvicorn api.main:app --reload
   ```

2. In a separate terminal, start Streamlit:

   ```bash
   streamlit run dashboard/app.py
   ```

3. (Optional) Point the dashboard at a different API host/port:

   ```bash
   export API_BASE_URL=http://localhost:8000
   ```

**Sections:** Overview (portfolio-level KPIs and risk/action distributions from
`reports/recommendation_summary.csv`), Customer Analysis (prediction, SHAP drivers, and
recommendation for a selected real customer/snapshot, via `POST /customer-analysis`),
and Model Information (`GET /model-info` plus the Phase 7 tuning report).

## Monitoring & Observability

A lightweight, in-process monitoring layer (`src/monitoring/`) built on top of the
existing FastAPI service -- no new infrastructure (no Prometheus/Grafana, no database,
no external monitoring platform). Storage is in-memory only and resets when the API
process restarts; see "Limitations" below.

### What is monitored

- **API requests** -- every HTTP request/response, via a middleware that wraps the
  whole app.
- **Model inference** -- only the four prediction endpoints (`/predict`,
  `/predict/explain`, `/recommend`, `/customer-analysis`), tracked separately from
  generic HTTP traffic.
- **Model health** -- whether the registered model is loaded and usable, reusing the
  existing `GET /model-info` metadata (never a second model load).
- **Data quality** -- the rate of legitimately-null features in *accepted* requests.
  This is additive, not a re-validation: missing fields, wrong types, and NaN/Infinity
  values are already rejected with HTTP 422 by the existing Pydantic schema (Phase 11)
  before a request ever reaches this layer.
- **Data drift and prediction drift** -- see below.

### Monitoring endpoints

```bash
curl http://localhost:8000/monitoring/metrics
curl http://localhost:8000/monitoring/health
curl http://localhost:8000/monitoring/drift
```

- `GET /monitoring/metrics` -- total/successful/failed request counts, counts by
  endpoint/method/status code, latency (average since start, p50/p95/p99 over a
  rolling window of the most recent 1000 requests), and prediction counts
  (successful/failed/positive/negative, average predicted probability, average
  inference time, average null-field rate).
- `GET /monitoring/health` -- API status, model status/name/version, and whether drift
  monitoring has a usable baseline. Always returns HTTP 200 (even when unhealthy), so
  it can be polled to report *that* something is down.
- `GET /monitoring/drift` -- see below. Returns HTTP 503 if the registered model itself
  isn't available (drift monitoring is built alongside the model at startup).

None of these expose a customer identifier, a raw feature value, a filesystem path, or
a credential.

### Request IDs

Every response includes an `X-Request-ID` header: a client-supplied `X-Request-ID`
request header is preserved, otherwise a UUID4 is generated. The same ID appears in
the corresponding access-log line for correlating client- and server-side logs.

### Logging

A single structured access-log line per request (`api.access` logger):

```text
INFO request method=POST path=/customer-analysis status=200 duration_ms=123.4 request_id=...
```

This never includes the request or response body, query parameters, or headers --
exactly where a raw feature value, a customer identifier beyond what Phase 11 already
logged, or a credential could end up.

### Data drift vs. prediction drift vs. model performance degradation

These are three different things, and only the first two are actually monitored here:

- **Data drift**: the distribution of *input features* in live traffic has shifted
  compared to the data the model was trained on.
- **Prediction drift**: the distribution of the model's *own predicted probabilities*
  has shifted compared to its behavior on a held-out reference set.
- **Model performance degradation**: the model is actually making worse predictions.
  This requires ground-truth outcomes (`churn_next_30_days`), which are never
  available at inference time -- **this project does not and cannot measure
  performance degradation directly**. Drift is a signal that something changed and is
  worth investigating; it is never presented as proof that the model is performing
  worse.

**Method:** Population Stability Index (PSI), a standard metric in churn/credit-risk
monitoring, applied uniformly to (a) the top-8 numeric features by Phase 8's own SHAP
importance ranking (`reports/shap_feature_importance.csv`), (b) the four categorical
features, and (c) the model's predicted-probability distribution. The baseline is the
Phase 7 **TRAIN** split for features and the **VALIDATION** split (out-of-sample) for
predictions. Standard thresholds: PSI < 0.10 no significant shift, 0.10-0.25 moderate
shift, >= 0.25 significant shift. At least 30 recent observations are required before
a PSI is reported at all.

### Streamlit dashboard

A fourth **Monitoring** tab (alongside Overview, Customer Analysis, and Model
Information) displays API/model health, request metrics, latency, prediction counts,
and drift status -- every number read directly from a `GET /monitoring/*` response;
the dashboard never computes a metric itself.

### Docker

No Docker configuration changes were needed: the monitoring endpoints are served by
the existing `api` container on its existing port (8000), and the dashboard's existing
`API_BASE_URL=http://api:8000` already routes the Monitoring tab's requests correctly.

### Limitations

- **In-memory only.** All counters and rolling windows reset when the API process
  restarts (including a `docker compose restart api`). There is no persistent metrics
  history or long-term trend storage -- a deliberate simplicity trade-off for this
  project's scope, not an oversight.
- **PSI is sensitive to sample size.** With 10 histogram bins, a comparison window of
  only ~30-100 observations can produce a noisy PSI even with no real distributional
  change; the reported status should be read alongside `sample_size`, not in isolation.
- **Drift baselines are fixed at startup.** They are not periodically refreshed;
  restarting the API recomputes them from whatever `data/processed/customer_features.parquet`
  and the registered model currently are.
- **No performance-degradation detection.** As explained above, this is a hard
  limitation of monitoring without ground-truth labels, not something a smarter metric
  would fix.

## Docker

Runs the existing FastAPI service and Streamlit dashboard as two containers on a shared
Compose network, without changing how either application behaves locally.

```
Browser -> Streamlit container (host port 8501)
              | HTTP, using the Compose service name "api" -- never localhost
              v
          FastAPI container (host port 8000)
              |
              v
    existing MLflow-registered model / SHAP / recommendation code (unchanged)
```

### 1. Prerequisites

- Docker Desktop (or an equivalent Docker Engine + Compose v2) installed and running.
- Phases 1-10 already run **locally at least once**, so the following git-ignored,
  developer-generated artifacts exist on disk (see "Model artifact / runtime
  requirements" below for why these can't be baked into the images):
  - `data/processed/customer_features.parquet` (Phase 4)
  - `mlruns.db` and `mlartifacts/` (Phase 10 MLflow registration)

### 2. Environment setup

Copy the example environment file if you want to override any defaults:

```bash
cp .env.example .env
```

No real secrets are needed for local use -- `.env.example` only documents
`API_BASE_URL`, and Compose already sets the correct in-network value for the
dashboard container (see below).

### 3. Building images

```bash
docker compose build
```

Both `Dockerfile.api` and `Dockerfile.dashboard` install the project's single
`requirements.txt` and copy the application source (respecting `.dockerignore`).

### 4. Starting the application

```bash
docker compose up -d
```

The dashboard container waits for the API container's health check to pass before
starting (`depends_on: condition: service_healthy`).

- **FastAPI:** http://localhost:8000 (docs at `/docs`, health at `/health`)
- **Streamlit:** http://localhost:8501

### 5. Stopping containers

```bash
docker compose down
```

This removes the containers and the Compose network but leaves the built images and
your local `mlruns.db` / `mlartifacts/` / `data/` untouched.

### 6. Viewing logs

```bash
docker compose logs -f api
docker compose logs -f dashboard
```

### 7. Rebuilding after changes

```bash
docker compose up -d --build
```

### 8. Troubleshooting API connectivity

- `GET http://localhost:8000/health` returning `{"status": "unhealthy", ...}` almost
  always means `data/processed/customer_features.parquet`, `mlruns.db`, or
  `mlartifacts/` isn't present on the host -- run Phases 1-10 locally first.
- The dashboard's sidebar shows "API Unavailable" if it can't reach the `api`
  service -- check `docker compose logs api` for the actual startup error.
- Inside the `dashboard` container, `http://localhost:8000` will **not** reach the API
  (that's the dashboard container itself); it must use `http://api:8000`, which is
  exactly what `docker-compose.yml` sets `API_BASE_URL` to.

### 9. Model artifact / runtime requirements -- important limitation

`data/processed/customer_features.parquet`, `mlruns.db`, and `mlartifacts/` are
git-ignored (see `.gitignore`) and are **not** copied into either image. Instead they
are mounted read-only (mostly) into the `api`/`dashboard` containers at runtime from
the host, so a missing artifact fails clearly at container startup (via the existing
`/health` check) rather than silently baking a stale copy into an image layer.

**A machine-specific quirk discovered while building this phase:** MLflow's local
SQLite backend (`mlruns.db`) records each registered model's artifact location as an
**absolute host path** at the moment it was registered (Phase 10) --
`/Users/dhananjaymandal/customer-retention-intelligence/mlartifacts/...` on this
machine. `docker-compose.yml` therefore mounts `mlartifacts/` a second time at that
exact absolute path inside the `api` container (in addition to the normal `/app`
mount), and mounts it **read-write** because `mlflow.sklearn.load_model` writes a
small internal bookkeeping file (`registered_model_meta`) when loading an
already-registered model -- this is MLflow's own housekeeping, not retraining or a
model behavior change.

Practical effect: this `docker-compose.yml`, as committed, is tied to running on this
project's current absolute path. Running it from a different clone location or a
different machine requires either re-registering the model there (`python -m
src.models.mlflow_tracking`, after Phases 1-10 have produced fresh local artifacts) or
adjusting the second `mlartifacts` mount target in `docker-compose.yml` to match that
machine's actual project path.

## Deployment
To be implemented.

## Project Structure
To be implemented.

## Installation
To be implemented.

## How to Run
To be implemented.

## Results
Results will be added after model training.

## Business Impact
To be implemented.

## Future Improvements
To be implemented.

## Interview Discussion Points
To be implemented.
