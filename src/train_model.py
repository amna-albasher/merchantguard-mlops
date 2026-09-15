from pathlib import Path
import math

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "merchant_activity.csv"
MLFLOW_DATABASE = PROJECT_ROOT / "mlflow.db"

EXPERIMENT_NAME = "merchant-churn-training"
MODEL_NAME = "merchant-churn-model"
TRAIN_END_MONTH = 11

NUMERIC_FEATURES = [
    "annual_contract_value",
    "logins_per_week",
    "feature_adoption_pct",
    "support_tickets",
    "support_sentiment",
    "product_release_event",
]

CATEGORICAL_FEATURES = ["plan_tier"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn_within_3_months"


def calculate_top_10_capture(
    actual: pd.Series,
    probabilities,
) -> float:
    results = pd.DataFrame(
        {
            "actual": actual.to_numpy(),
            "probability": probabilities,
        }
    ).sort_values("probability", ascending=False)

    number_to_contact = max(
        1,
        math.ceil(len(results) * 0.10),
    )

    top_accounts = results.head(number_to_contact)
    total_churners = results["actual"].sum()

    if total_churners == 0:
        return 0.0

    return float(
        top_accounts["actual"].sum() / total_churners
    )


def main() -> None:
    dataset = pd.read_csv(DATA_FILE)

    training_data = dataset[
        dataset["month_index"] <= TRAIN_END_MONTH
    ].copy()

    validation_data = dataset[
        dataset["month_index"] > TRAIN_END_MONTH
    ].copy()

    x_train = training_data[FEATURES]
    y_train = training_data[TARGET]

    x_validation = validation_data[FEATURES]
    y_validation = validation_data[TARGET]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_validation)[:, 1]
    predictions = (probabilities >= 0.30).astype(int)

    metrics = {
        "pr_auc": average_precision_score(
            y_validation,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_validation,
            probabilities,
        ),
        "precision": precision_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "top_10_percent_capture": calculate_top_10_capture(
            y_validation,
            probabilities,
        ),
    }

    tracking_uri = (
        f"sqlite:///{MLFLOW_DATABASE.as_posix()}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    signature = infer_signature(
        x_train.head(10),
        model.predict(x_train.head(10)),
    )

    with mlflow.start_run(run_name="logistic-regression-baseline"):
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "class_weight": "balanced",
                "decision_threshold": 0.30,
                "train_end_month_index": TRAIN_END_MONTH,
                "training_rows": len(training_data),
                "validation_rows": len(validation_data),
            }
        )

        mlflow.log_metrics(metrics)

        mlflow.set_tags(
            {
                "project": "MerchantGuard",
                "model_status": "candidate",
                "data_type": "synthetic",
            }
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=x_train.head(5),
        )

    client = MlflowClient()
    model_versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )
    newest_version = max(
        model_versions,
        key=lambda version: int(version.version),
    )

    client.set_registered_model_alias(
        MODEL_NAME,
        "candidate",
        newest_version.version,
    )

    print(f"Training rows: {len(training_data):,}")
    print(f"Validation rows: {len(validation_data):,}")
    print(f"Training target rate: {y_train.mean():.2%}")
    print(
        f"Validation target rate: "
        f"{y_validation.mean():.2%}"
    )
    print()
    print("Validation metrics")
    print("------------------")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print()
    print(f"Registered model: {MODEL_NAME}")
    print(f"Candidate version: {newest_version.version}")
    print("Candidate alias assigned successfully")


if __name__ == "__main__":
    main()