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
To be implemented.

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
