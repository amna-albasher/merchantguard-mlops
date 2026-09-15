from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import json
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from mlflow import MlflowClient
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DATABASE = PROJECT_ROOT / "mlflow.db"
UI_FILE = PROJECT_ROOT / "ui" / "index.html"

MODEL_NAME = "merchant-churn-model"
MODEL_ALIAS = "champion"
DECISION_THRESHOLD = 0.30

DEFAULT_MODEL_URI = (
    f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
)

MODEL_URI = os.getenv(
    "MODEL_URI",
    DEFAULT_MODEL_URI,
)

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_DATABASE.as_posix()}",
)


class PredictionRequest(BaseModel):
    merchant_id: str = Field(
        min_length=1,
        examples=["MER-2001"],
    )
    plan_tier: Literal[
        "Growth",
        "Professional",
        "Enterprise",
    ]
    annual_contract_value: float = Field(
        ge=10_000,
        le=50_000,
    )
    logins_per_week: float = Field(
        ge=0,
        le=50,
    )
    feature_adoption_pct: float = Field(
        ge=0,
        le=100,
    )
    support_tickets: int = Field(ge=0)
    support_sentiment: float = Field(
        ge=-1,
        le=1,
    )
    product_release_event: int = Field(
        ge=0,
        le=1,
    )


class PredictionResponse(BaseModel):
    merchant_id: str
    churn_probability: float
    at_risk: bool
    risk_level: str
    model_name: str
    model_alias: str
    model_version: str


def determine_risk_level(
    probability: float,
) -> str:
    if probability >= 0.70:
        return "high"

    if probability >= DECISION_THRESHOLD:
        return "medium"

    return "low"


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    if MODEL_URI.startswith("models:/"):
        client = MlflowClient()

        model_version = (
            client.get_model_version_by_alias(
                MODEL_NAME,
                MODEL_ALIAS,
            )
        )

        app.state.model_version = str(
            model_version.version
        )

    else:
        metadata_file = (
            Path(MODEL_URI)
            / "champion_metadata.json"
        )

        metadata = json.loads(
            metadata_file.read_text(
                encoding="utf-8"
            )
        )

        app.state.model_version = metadata[
            "model_version"
        ]

    app.state.model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    yield


app = FastAPI(
    title="MerchantGuard Churn Prediction API",
    description=(
        "Predicts whether a merchant is at risk "
        "of churning within the next three months."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(UI_FILE)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": hasattr(
            app.state,
            "model",
        ),
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": (
            app.state.model_version
        ),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
):
    try:
        model_input = pd.DataFrame(
            [
                {
                    "annual_contract_value": (
                        request.annual_contract_value
                    ),
                    "logins_per_week": (
                        request.logins_per_week
                    ),
                    "feature_adoption_pct": (
                        request.feature_adoption_pct
                    ),
                    "support_tickets": (
                        request.support_tickets
                    ),
                    "support_sentiment": (
                        request.support_sentiment
                    ),
                    "product_release_event": (
                        request.product_release_event
                    ),
                    "plan_tier": request.plan_tier,
                }
            ]
        )

        probability = float(
            app.state.model.predict_proba(
                model_input
            )[0, 1]
        )

        return PredictionResponse(
            merchant_id=request.merchant_id,
            churn_probability=round(
                probability,
                4,
            ),
            at_risk=(
                probability
                >= DECISION_THRESHOLD
            ),
            risk_level=determine_risk_level(
                probability
            ),
            model_name=MODEL_NAME,
            model_alias=MODEL_ALIAS,
            model_version=(
                app.state.model_version
            ),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: {error}"
            ),
        ) from error