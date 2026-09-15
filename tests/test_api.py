from pathlib import Path
import os

import pytest
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.environ["MODEL_URI"] = str(
    PROJECT_ROOT
    / "artifacts"
    / "champion_model"
)

from src.api import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_dashboard_loads(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Retention workspace" in response.text


def test_health_endpoint(client):
    response = client.get("/health")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["model_alias"] == "champion"


def test_high_risk_prediction(client):
    response = client.post(
        "/predict",
        json={
            "merchant_id": "MER-TEST-HIGH",
            "plan_tier": "Growth",
            "annual_contract_value": 12000,
            "logins_per_week": 2,
            "feature_adoption_pct": 25,
            "support_tickets": 7,
            "support_sentiment": -0.7,
            "product_release_event": 0,
        },
    )

    body = response.json()

    assert response.status_code == 200
    assert body["at_risk"] is True
    assert body["risk_level"] == "high"
    assert body["churn_probability"] >= 0.70


def test_low_risk_prediction(client):
    response = client.post(
        "/predict",
        json={
            "merchant_id": "MER-TEST-LOW",
            "plan_tier": "Enterprise",
            "annual_contract_value": 45000,
            "logins_per_week": 18,
            "feature_adoption_pct": 85,
            "support_tickets": 0,
            "support_sentiment": 0.8,
            "product_release_event": 0,
        },
    )

    body = response.json()

    assert response.status_code == 200
    assert body["at_risk"] is False
    assert body["risk_level"] == "low"
    assert body["churn_probability"] < 0.30


def test_invalid_request_is_rejected(client):
    response = client.post(
        "/predict",
        json={
            "merchant_id": "MER-INVALID",
            "plan_tier": "Unknown",
            "annual_contract_value": 5000,
            "logins_per_week": -1,
            "feature_adoption_pct": 120,
            "support_tickets": -2,
            "support_sentiment": 2,
            "product_release_event": 3,
        },
    )

    assert response.status_code == 422