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

## Monitoring
To be implemented.

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
