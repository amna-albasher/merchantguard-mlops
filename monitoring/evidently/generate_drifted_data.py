
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "merchant_activity.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "merchant_activity_drifted.csv"

RANDOM_SEED = 42


def main():
    np.random.seed(RANDOM_SEED)

    df = pd.read_csv(INPUT_PATH)
    drifted = df.copy()

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    for column in drifted.columns:

        # Keep identifier unchanged
        if column == "merchant_id":
            continue

        # Numeric features
        if pd.api.types.is_numeric_dtype(drifted[column]):

            unique_values = set(drifted[column].dropna().unique())

            # Flip binary features
            if unique_values.issubset({0, 1}):
                mask = np.random.rand(len(drifted)) < 0.70
                drifted.loc[mask, column] = 1 - drifted.loc[mask, column]

            else:
                std = drifted[column].std()

                if pd.notna(std) and std > 0:
                    drifted[column] = (
                        drifted[column] * 1.35
                        + std * 1.5
                    )

        # Categorical features
        else:
            mask = np.random.rand(len(drifted)) < 0.70
            drifted.loc[mask, column] = "DRIFTED_" + drifted.loc[
                mask, column
            ].astype(str)

    drifted.to_csv(OUTPUT_PATH, index=False)

    print(f"Drifted dataset created: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()