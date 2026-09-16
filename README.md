# MerchantGuard

MerchantGuard is an end-to-end MLOps project for merchant churn prediction. It covers the full lifecycle from synthetic data generation and model training to model registry, API serving, Kubernetes deployment, monitoring, drift detection, automated retraining, and CI/CD.

## What the project demonstrates

MerchantGuard simulates a SaaS retention use case where merchants at risk of churn are identified early enough for Customer Success teams to intervene.

The project includes:

- Synthetic merchant activity data generation
- Logistic Regression churn model
- MLflow experiment tracking and model registry
- Candidate and champion model aliases
- Validation gates before promotion
- FastAPI prediction service
- Docker containerization
- Kubernetes deployment
- Prometheus metrics
- Grafana monitoring dashboard
- Evidently data drift detection
- Automatic retraining when drift exceeds the configured threshold
- GitHub Actions CI/CD
- Automated API tests and container health checks

## Architecture

```text
Synthetic Merchant Data
        |
        v
Model Training
        |
        v
MLflow Experiment Tracking
        |
        v
Candidate Model
        |
        v
Validation Gate
        |
        v
Champion Model
        |
        v
FastAPI Prediction Service
        |
        v
Docker
        |
        v
Kubernetes
        |
        +--------------------+
        |                    |
        v                    v
   Prometheus            Prediction API
        |
        v
     Grafana

New / Shifted Data
        |
        v
    Evidently
        |
        v
Data Drift Detected
        |
        v
Automatic Retraining
        |
        v
New MLflow Candidate Version
Tech stack
Python
pandas
scikit-learn
MLflow
FastAPI
Docker
Kubernetes
Prometheus
Grafana
Evidently
GitHub Actions
pytest
Model training

The churn model uses a Logistic Regression pipeline with preprocessing for numeric and categorical features.

Key features include:

annual contract value
weekly logins
feature adoption
support tickets
support sentiment
product release events
plan tier

The model predicts:

churn_within_3_months

The pipeline tracks:

PR AUC
ROC AUC
Precision
Recall
Top 10% churn capture
MLflow model lifecycle

MLflow is used for:

experiment tracking
parameter logging
metric logging
model artifacts
model registration
candidate / champion aliases

The registry keeps multiple model versions while allowing the application to use a stable champion model.

FastAPI serving

MerchantGuard exposes the approved model through a FastAPI service.

The application includes:

prediction endpoint
health endpoint
Prometheus metrics endpoint
browser-based retention dashboard

Kubernetes deployment

The API is containerized with Docker and deployed to Kubernetes.

The Kubernetes configuration includes:

Deployment
Service
health checks
application port configuration

The application can be accessed locally through port forwarding.

kubectl port-forward service/merchantguard-service 18000:80
Monitoring with Prometheus and Grafana

Prometheus scrapes application metrics from the FastAPI service.

Grafana visualizes operational metrics including:

API error rate
total API requests
request rate
P95 latency

Data drift monitoring

Evidently compares reference data with current production-like data.

The monitoring workflow calculates column-level drift and determines the overall share of drifted features.

In the drift simulation used for this project:

12 of 13 columns drifted
Drift share: ~0.923

Because the configured dataset drift threshold is:

0.50

the workflow identifies significant drift.

Automated retraining

When the drift share exceeds the configured threshold, MerchantGuard automatically triggers:

src/retrain_model.py

The retraining workflow:

loads the drifted dataset
retrains the model
evaluates model performance
logs the run to MLflow
creates a new registered model version
assigns the new version the candidate alias

Example retraining result:

pr_auc: 0.6729
roc_auc: 0.4918
precision: 0.6773
recall: 1.0000
top_10_percent_capture: 0.0990

The low ROC AUC in this drift simulation is intentional evidence that a newly retrained model should still be validated before being promoted to champion.

CI/CD

GitHub Actions runs the complete validation pipeline on pushes and pull requests.

The workflow:

installs dependencies
generates synthetic data
generates a drifted dataset
runs drift monitoring
triggers automatic retraining
trains a candidate model
runs the validation gate
exports the approved champion
runs automated tests
builds the Docker image
pushes the image to GitHub Container Registry on main
launches the container
checks the /health endpoint

Automated tests

The current test suite validates:

dashboard loading
health endpoint
high-risk prediction
low-risk prediction
invalid request handling

Current result:

5 passed
Monitoring stack

Start Prometheus and Grafana:

docker compose -f monitoring/docker-compose.yml up -d

Prometheus:

http://localhost:9090

Grafana:

http://localhost:3000

The Grafana datasource and dashboard are provisioned from the repository so the monitoring environment can be recreated automatically.

Run the project locally

Create and activate a virtual environment:

python -m venv .venv

Windows:

.venv\Scripts\Activate.ps1

Install dependencies:

python -m pip install -r requirements-dev.txt

Generate data:

python src/generate_data.py

Train the model:

python src/train_model.py

Run validation:

python src/validate_model.py

Export the champion:

python src/export_champion.py

Run tests:

pytest -v
Run drift monitoring

Generate drifted data:

python monitoring/evidently/generate_drifted_data.py

Run the monitoring workflow:

python monitoring/evidently/drift_report.py

If drift exceeds the configured threshold, retraining is triggered automatically.

Project structure
merchantguard-mlops/
|
├── .github/
│   └── workflows/
│       └── ci.yml
|
├── data/
|
├── k8s/
│   └── merchantguard.yaml
|
├── monitoring/
│   ├── docker-compose.yml
│   ├── prometheus.yml
│   ├── evidently/
│   │   ├── drift_report.py
│   │   └── generate_drifted_data.py
│   └── grafana/
│       ├── dashboards/
│       │   └── merchantguard-dashboard.json
│       └── provisioning/
|
├── reports/
│   └── screenshots/
|
├── src/
│   ├── api.py
│   ├── export_champion.py
│   ├── generate_data.py
│   ├── retrain_model.py
│   ├── train_model.py
│   └── validate_model.py
|
├── tests/
|
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
Why this project matters

MerchantGuard is designed to show more than model training.

The focus is the operational side of machine learning:

reproducibility
model versioning
controlled promotion
containerized serving
Kubernetes deployment
observability
drift detection
retraining
CI/CD
testable production workflows

The goal is to demonstrate how a machine learning model can move from experimentation into a monitored, repeatable MLOps lifecycle.
