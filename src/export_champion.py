from pathlib import Path
import json

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DATABASE = PROJECT_ROOT / "mlflow.db"

MODEL_NAME = "merchant-churn-model"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


def main() -> None:
    tracking_uri = (
        f"sqlite:///{MLFLOW_DATABASE.as_posix()}"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    client = MlflowClient()
    champion_info = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS,
    )

    export_directory = (
    PROJECT_ROOT / "artifacts" / "champion_model"
)

    if export_directory.exists():
        raise FileExistsError(
            f"{export_directory} already exists."
        )

    export_directory.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    champion_model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    mlflow.sklearn.save_model(
        sk_model=champion_model,
        path=str(export_directory),
    )

    metadata = {
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": str(
            champion_info.version
        ),
    }

    metadata_file = (
        export_directory / "champion_metadata.json"
    )
    metadata_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(
        f"Champion version: "
        f"{champion_info.version}"
    )
    print(
        f"Exported model to: "
        f"{export_directory}"
    )


if __name__ == "__main__":
    main()