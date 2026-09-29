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
    packaged MLflow registered model / SHAP / recommendation code (unchanged)
```

Since Phase 15 both images are self-contained: no host files are mounted, and the same
images are what CI smoke-tests and Render runs (see [Deployment](#deployment)).

### 1. Prerequisites

- Docker Desktop (or an equivalent Docker Engine + Compose v2) installed and running.
- Nothing else: no local data, model, or MLflow state is needed (see "Model artifact /
  runtime requirements" below).

### 2. Environment setup

Copy the example environment file if you want to override any defaults:

```bash
cp .env.example .env
```

No secrets are needed anywhere -- `.env.example` only documents optional,
non-sensitive settings (`API_BASE_URL`, `API_HOSTPORT`, `MODEL_ARTIFACT_ROOT`), and
Compose already sets the correct in-network value for the dashboard container.

### 3. Building images

```bash
docker compose build
```

Both `Dockerfile.api` and `Dockerfile.dashboard` install the pinned runtime
`requirements.txt`, copy the application source (respecting `.dockerignore`), and
regenerate `data/processed/customer_features.parquet` from the seeded pipeline
(about 30 seconds; see below).

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
your local `mlruns.db` / `mlartifacts/` / `data/` untouched (the containers never
used them).

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

- `GET http://localhost:8000/health` returning `{"status": "unhealthy", ...}` means
  the model service failed to load -- `docker compose logs api` shows why (for
  example a packaged-model SHA-256 mismatch).
- The dashboard's sidebar shows "API Unavailable" if it can't reach the `api`
  service -- check `docker compose logs api` for the actual startup error.
- Inside the `dashboard` container, `http://localhost:8000` will **not** reach the API
  (that's the dashboard container itself); it must use `http://api:8000`, which is
  exactly what `docker-compose.yml` sets `API_BASE_URL` to.

### 9. Model artifact / runtime requirements

Phase 13 bind-mounted `mlruns.db`, `mlartifacts/` and `data/processed/` from the host,
and had to mount `mlartifacts/` a second time at the developer machine's absolute path,
because MLflow's SQLite registry records each model's artifact location as an absolute
host path. Phase 15 removed that dependency entirely:

- **Model:** the API loads `models/registered/customer-churn-logistic-regression/1/`, a
  byte-for-byte copy of registry version 1's MLflow model directory (~55 KB), committed
  to the repo. `src.models.mlflow_tracking.load_packaged_model` loads it with MLflow's
  own sklearn loader after checking `model.pkl`'s SHA-256 against
  `configs/mlflow_config.yaml`, and applies the same model-type / `C` / `class_weight`
  validation as before. `MODEL_ARTIFACT_ROOT` can point it elsewhere. Nothing was
  retrained: predictions are identical to the Phase 7 artifact on all 136,701 rows.
- **Data:** `customer_features.parquet` is never committed or downloaded. The image
  build runs `python -m src.data.regenerate_feature_dataset --prune`, which re-runs the
  committed seeded Phase 1-4 pipeline and fails the build unless the result matches the
  content fingerprint of the dataset the model was registered alongside.
- **Local MLflow registry:** `mlruns.db` / `mlartifacts/` remain a local-only tool for
  `mlflow ui` and Phase 10 experiments; nothing in the running services reads them.

## Deployment

Phase 15: GitHub, CI/CD and a public deployment on [Render](https://render.com). No AWS
or other cloud account is involved.

### Production architecture

```
Browser
   | HTTPS (public *.onrender.com URL)
   v
customer-retention-dashboard   Render web service, Docker, Free instance (512 MB)
   | HTTP over Render's private network (API_HOSTPORT)
   v
customer-retention-api         Render web service, Docker, 1c-2g instance (2 GB) -- paid
   |                           also public, for /health, /docs and the monitoring endpoints
   v
packaged MLflow registered model (v1) -> SHAP explanations -> next-best-action
   |
   v
in-memory request metrics + drift monitoring (Phase 14)
```

Both services are defined in [`render.yaml`](render.yaml), a Render Blueprint that
builds the same `Dockerfile.api` / `Dockerfile.dashboard` used locally and in CI.

**Why two services, not one.** The API and dashboard are already separate containers
with separate health checks. Running both in one container would need a process
supervisor, would put Streamlit on the paid 2 GB instance, and would mean one crash
restarts both. Keeping two services preserves the tested Phase 13 architecture. Render
lets a Free service *send* private-network traffic to a paid one, so the dashboard
calls the API privately without going through its public URL.

**Why these instance types (measured, Linux containers):**

| Service | Measured memory | Instance type |
|---|---|---|
| API | ~730 MiB steady, ~860 MB peak while loading the model, SHAP background and drift baselines | `1c-2g` (1 CPU, 2 GB; paid) -- 512 MB instances would be killed |
| Dashboard | ~470-500 MB peak for a full session under a hard 512 MB limit | `free` (512 MB) -- tight, see limitations |

Check current prices on Render's [pricing page](https://render.com/pricing) before
creating the Blueprint. This repo does not create any Render resource itself.

### GitHub and branch strategy

- Repository: https://github.com/mandaldhananjay248-beep/customer-retention-intelligence
- `main` is the deployed branch. Work on a feature branch and open a pull request; CI
  runs on the PR, and merging to `main` triggers the deployment once CI passes.

### CI/CD

```
git push / pull request
   |
   v
GitHub Actions (.github/workflows/ci.yml)
   |-- tests:  Python 3.14, pip install -r requirements-dev.txt,
   |           regenerate dataset, provision Phase 7 model path from the packaged model,
   |           FAIL_ON_SKIPPED_TESTS=1 python -m pytest -q -rs
   |-- docker: docker compose build, docker compose up --wait,
   |           scripts/smoke_test.py against both containers
   v
all checks green on main
   |
   v
Render auto-deploys both services (render.yaml: autoDeployTrigger: checksPass)
```

- **No skipped tests in CI.** Several test modules skip themselves when a generated
  artifact is missing. CI provisions those artifacts from committed sources instead
  (the seeded data pipeline and the packaged model -- nothing is retrained), and
  `tests/conftest.py` fails the run if *any* test is skipped while
  `FAIL_ON_SKIPPED_TESTS=1`. The job summary reports collected / passed / failed /
  errors / skipped.
- **No credentials.** CI's token is `contents: read`; it never pushes images or calls
  Render. Render builds from GitHub through its own GitHub integration.

### Local development

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt        # runtime deps + pytest + Jupyter

# First time only: generate the dataset (seeded, ~30 s) and provision the Phase 7
# model path from the packaged model if you haven't run Phases 1-10 locally.
python -m src.data.regenerate_feature_dataset
python -m src.models.provision_tuned_model

uvicorn api.main:app --reload              # http://localhost:8000/docs
streamlit run dashboard/app.py             # http://localhost:8501
```

Or run both containers with `docker compose up -d --build` (see [Docker](#docker)).

### Testing

```bash
python -m pytest -q                              # full suite
FAIL_ON_SKIPPED_TESTS=1 python -m pytest -q -rs  # exactly what CI runs

# End-to-end smoke test of any running deployment (standard library only):
python scripts/smoke_test.py --api-url http://localhost:8000 --dashboard-url http://localhost:8501
```

The smoke test checks `/health`, `/model-info`, `/monitoring/health`, a real
`/customer-analysis` request (its churn probability must equal the packaged model's
output for the model's own MLflow input example), SHAP drivers, the recommendation,
the `X-Request-ID` header, that an invalid request is rejected and counted as a failed
request, that `/monitoring/metrics` counters grow, `/monitoring/drift`, and the
dashboard's `/_stcore/health`.

### Deploying on Render (first time)

1. Push `main` to GitHub and confirm the CI workflow is green.
2. Review `render.yaml`: set `region` on **both** services if Render's default
   (`oregon`) isn't the one you want -- region can't be changed later, and both must
   match for private networking.
3. In the Render Dashboard choose **New > Blueprint**, connect the GitHub repository,
   and select `main`. Render lists the two services and their instance types; the API's
   `1c-2g` instance is billed.
4. Wait for both deploys to go live, then run:
   `python scripts/smoke_test.py --api-url https://<api-service>.onrender.com --dashboard-url https://<dashboard-service>.onrender.com --wait 300`

### Environment variables

| Variable | Service | Value | Secret? |
|---|---|---|---|
| `PORT` | API / dashboard | `8000` / `8501`, set in `render.yaml` to match each image's port | No |
| `API_HOSTPORT` | dashboard | the API's private `host:port`, injected by `render.yaml` (`fromService ... hostport`) | No |
| `API_BASE_URL` | dashboard | full API URL; overrides `API_HOSTPORT` (Compose sets `http://api:8000`) | No |
| `MODEL_ARTIFACT_ROOT` | API | optional override for `models/registered/` | No |

The application needs **no secrets**. None are defined in the repo, `render.yaml`,
Dockerfiles or CI.

### Health checks and endpoints

| Check | Where |
|---|---|
| API health (Render health check) | `GET /health` -- 200 only when the model service is loaded |
| Dashboard health (Render health check) | `GET /_stcore/health` |
| Model metadata | `GET /model-info` |
| Monitoring | `GET /monitoring/health`, `GET /monitoring/metrics`, `GET /monitoring/drift` |
| Interactive API docs | `GET /docs` |

Public URLs: `https://<service-name>.onrender.com`, assigned by Render when the
Blueprint is created. They will be recorded here once the services exist and have been
verified.

### Redeploy and rollback

- **Redeploy:** push or merge to `main` (auto-deploys after CI passes), or use
  **Manual Deploy > Deploy latest commit** on a service in the Render Dashboard.
- **Rollback:** on a service's **Deploys** page, pick an earlier successful deploy and
  choose **Rollback > Rollback to this deploy** (only builds Render still retains are
  available). A Dashboard rollback disables that service's automatic deploys, so revert
  the bad commit on `main` (`git revert <sha>`, never a force-push) before turning
  auto-deploy back on. Roll back both services if the bad change touched both.
- **Stop costs:** suspend or delete `customer-retention-api` in the Render Dashboard.

### Production limitations

- **Cost:** the API needs the paid `1c-2g` instance; nothing about it scales to zero.
- **Free dashboard spin-down:** the Free dashboard spins down after 15 minutes without
  traffic and takes about a minute to start again. It is not "always on".
- **Tight dashboard memory:** a full dashboard session peaked at ~470-500 MB under a
  hard 512 MB limit in local tests. Render's own overhead or heavy concurrent use could
  still push it over and restart the service; the fix is a larger instance type.
- **Ephemeral filesystem:** nothing written at runtime survives a redeploy, restart or
  spin-down. The app writes nothing it needs, but see the next point.
- **In-memory monitoring:** `/monitoring/*` counters, latency windows and drift
  observations live in process memory. They reset on every deploy and restart, and
  would not be aggregated across multiple instances.
- **Model/data provenance:** the served model is the committed packaged copy of
  registry version 1. The dataset is regenerated at build time and fingerprint-checked,
  so any change to the data pipeline code fails the build until the fingerprint (and,
  if intended, the model) is deliberately updated.
- **Public model artifact:** the repository is public, so the packaged model (trained
  on synthetic data) and its five-row MLflow input example are public too.
- **No custom domain, auth or rate limiting:** both services are open on
  `*.onrender.com` with Render-managed HTTPS.

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
