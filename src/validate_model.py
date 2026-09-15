from pathlib import Path
import math

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "merchant_activity.csv"
MLFLOW_DATABASE = PROJECT_ROOT / "mlflow.db"

MODEL_NAME = "merchant-churn-model"
VALIDATION_START_MONTH = 12
DECISION_THRESHOLD = 0.30

MINIMUM_PR_AUC = 0.65
MINIMUM_PRECISION = 0.50
MINIMUM_RECALL = 0.60
MINIMUM_TOP_10_CAPTURE = 0.55
MINIMUM_PR_AUC_IMPROVEMENT = 0.005

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

    total_churners = results["actual"].sum()

    if total_churners == 0:
        return 0.0

    return float(
        results.head(number_to_contact)["actual"].sum()
        / total_churners
    )


def evaluate_model(model, features, target) -> dict:
    probabilities = model.predict_proba(features)[:, 1]

    predictions = (
        probabilities >= DECISION_THRESHOLD
    ).astype(int)

    return {
        "pr_auc": average_precision_score(
            target,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            target,
            probabilities,
        ),
        "precision": precision_score(
            target,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            target,
            predictions,
            zero_division=0,
        ),
        "top_10_percent_capture": (
            calculate_top_10_capture(
                target,
                probabilities,
            )
        ),
    }


def passes_minimum_requirements(metrics: dict) -> bool:
    return (
        metrics["pr_auc"] >= MINIMUM_PR_AUC
        and metrics["precision"] >= MINIMUM_PRECISION
        and metrics["recall"] >= MINIMUM_RECALL
        and metrics["top_10_percent_capture"]
        >= MINIMUM_TOP_10_CAPTURE
    )


def print_metrics(title: str, metrics: dict) -> None:
    print(title)
    print("-" * len(title))

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print()


def main() -> None:
    tracking_uri = (
        f"sqlite:///{MLFLOW_DATABASE.as_posix()}"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment("merchant-churn-validation")

    client = MlflowClient()

    dataset = pd.read_csv(DATA_FILE)
    validation_data = dataset[
        dataset["month_index"] >= VALIDATION_START_MONTH
    ].copy()

    features = validation_data[FEATURES]
    target = validation_data[TARGET]

    candidate_info = client.get_model_version_by_alias(
        MODEL_NAME,
        "candidate",
    )

    candidate_model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@candidate"
    )

    candidate_metrics = evaluate_model(
        candidate_model,
        features,
        target,
    )

    print(
        f"Evaluating candidate version "
        f"{candidate_info.version}"
    )
    print_metrics("Candidate metrics", candidate_metrics)

    champion_exists = True

    try:
        champion_info = (
            client.get_model_version_by_alias(
                MODEL_NAME,
                "champion",
            )
        )

        champion_model = mlflow.sklearn.load_model(
            f"models:/{MODEL_NAME}@champion"
        )

        champion_metrics = evaluate_model(
            champion_model,
            features,
            target,
        )

        print(
            f"Current champion version: "
            f"{champion_info.version}"
        )
        print_metrics(
            "Champion metrics",
            champion_metrics,
        )

    except MlflowException:
        champion_exists = False
        champion_info = None
        champion_metrics = None

        print("No champion model currently exists.")
        print()

    minimums_passed = passes_minimum_requirements(
        candidate_metrics
    )

    if champion_exists:
        beats_champion = (
            candidate_metrics["pr_auc"]
            >= champion_metrics["pr_auc"]
            + MINIMUM_PR_AUC_IMPROVEMENT
            and candidate_metrics[
                "top_10_percent_capture"
            ]
            >= champion_metrics[
                "top_10_percent_capture"
            ]
        )
    else:
        beats_champion = True

    promotion_approved = (
        minimums_passed and beats_champion
    )

    with mlflow.start_run(
        run_name=(
            f"validation-gate-v"
            f"{candidate_info.version}"
        )
    ):
        mlflow.log_params(
            {
                "candidate_version": (
                    candidate_info.version
                ),
                "champion_version": (
                    champion_info.version
                    if champion_info
                    else "none"
                ),
                "decision_threshold": (
                    DECISION_THRESHOLD
                ),
                "minimum_pr_auc": MINIMUM_PR_AUC,
                "minimum_precision": (
                    MINIMUM_PRECISION
                ),
                "minimum_recall": MINIMUM_RECALL,
                "minimum_top_10_capture": (
                    MINIMUM_TOP_10_CAPTURE
                ),
                "minimum_pr_auc_improvement": (
                    MINIMUM_PR_AUC_IMPROVEMENT
                ),
            }
        )

        mlflow.log_metrics(
            {
                f"candidate_{name}": value
                for name, value
                in candidate_metrics.items()
            }
        )

        mlflow.set_tags(
            {
                "gate_result": (
                    "passed"
                    if promotion_approved
                    else "failed"
                ),
                "minimums_passed": str(
                    minimums_passed
                ).lower(),
                "beats_champion": str(
                    beats_champion
                ).lower(),
            }
        )

    if promotion_approved:
        client.set_registered_model_alias(
            MODEL_NAME,
            "champion",
            candidate_info.version,
        )

        client.set_model_version_tag(
            MODEL_NAME,
            candidate_info.version,
            "validation_status",
            "passed",
        )

        print("VALIDATION PASSED")
        print(
            f"Version {candidate_info.version} "
            f"promoted to champion."
        )
    else:
        client.set_model_version_tag(
            MODEL_NAME,
            candidate_info.version,
            "validation_status",
            "failed",
        )

        print("VALIDATION FAILED")

        if not minimums_passed:
            print(
                "Candidate did not meet the "
                "minimum quality thresholds."
            )

        if champion_exists and not beats_champion:
            print(
                "Candidate did not outperform "
                "the current champion."
            )


if __name__ == "__main__":
    main()