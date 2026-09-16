from pathlib import Path
import subprocess
import sys

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "merchant_activity.csv"
)

CURRENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "merchant_activity_drifted.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "drift_report.html"
)

RETRAIN_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "retrain_model.py"
)

DATASET_DRIFT_THRESHOLD = 0.50


def main():
    reference_data = pd.read_csv(
        REFERENCE_PATH
    )

    current_data = pd.read_csv(
        CURRENT_PATH
    )

    print(
        f"Reference dataset: "
        f"{reference_data.shape[0]} rows, "
        f"{reference_data.shape[1]} columns"
    )

    print(
        f"Current dataset: "
        f"{current_data.shape[0]} rows, "
        f"{current_data.shape[1]} columns"
    )

    report = Report([
        DataDriftPreset()
    ])

    result = report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    result.save_html(
        str(OUTPUT_PATH)
    )

    print(
        f"Drift report saved to: "
        f"{OUTPUT_PATH}"
    )

    report_data = result.dump_dict()

    drifted_columns = 0
    total_columns = 0

    for metric in report_data[
        "metric_results"
    ].values():

        display_name = metric.get(
            "display_name",
            "",
        )

        if not display_name.startswith(
            "Value drift for "
        ):
            continue

        total_columns += 1

        drift_score = metric.get(
            "value"
        )

        params = (
            metric
            .get(
                "metric_value_location",
                {},
            )
            .get(
                "metric",
                {},
            )
            .get(
                "params",
                {},
            )
        )

        threshold = params.get(
            "threshold"
        )

        if (
            drift_score is not None
            and threshold is not None
            and drift_score > threshold
        ):
            drifted_columns += 1

    drift_share = (
        drifted_columns / total_columns
        if total_columns
        else 0.0
    )

    print()
    print("Drift summary")
    print("-------------")

    print(
        f"Drifted columns: "
        f"{drifted_columns}/{total_columns}"
    )

    print(
        f"Drift share: "
        f"{drift_share:.3f}"
    )

    if (
        drift_share
        >= DATASET_DRIFT_THRESHOLD
    ):
        print()
        print(
            "Dataset drift detected."
        )

        print(
            "Triggering model retraining..."
        )

        subprocess.run(
            [
                sys.executable,
                str(RETRAIN_SCRIPT),
            ],
            check=True,
        )

    else:
        print()
        print(
            "Dataset drift not detected."
        )

        print(
            "Retraining not required."
        )


if __name__ == "__main__":
    main()