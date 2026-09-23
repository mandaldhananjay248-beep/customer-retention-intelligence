# Customer Retention Intelligence Platform
## Project Definition

### 1. Business Problem
Predict customers who are likely to churn within the next 30 days.

### 2. Business Objective
Identify high-risk customers early and recommend appropriate retention actions.

### 3. ML Objective
Predict:
`churn_next_30_days`

Where:
- 1 = customer churns within the next 30 days
- 0 = customer does not churn within the next 30 days

### 4. Prediction Window
30 days.

### 5. Data Sources
- Customers
- Transactions
- Usage
- Support Tickets
- Churn Events

### 6. Important Feature Categories
- Customer demographics
- Engagement
- Financial behavior
- Support activity
- Payment behavior
- Behavioral changes

### 7. Data Leakage Rule
Only information available before or on the prediction date can be used as model features.

Future information must never be used as a feature.

### 8. Expected Model Output
- Customer ID
- Churn probability
- Risk level
- Top churn reasons
- Recommended retention action

### 9. Business Impact
Help the company identify high-risk customers and take proactive retention actions.

### 10. Initial Dataset Plan
Start with realistic synthetic longitudinal data and later scale the dataset.

### 11. Planned ML Models
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

### 12. Planned Production Components
- MLflow
- FastAPI
- Streamlit
- Docker
- Monitoring
- Cloud deployment