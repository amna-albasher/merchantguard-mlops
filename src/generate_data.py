from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUMBER_OF_MERCHANTS = 2_000
NUMBER_OF_MONTHS = 18
START_DATE = "2025-01-01"
PRODUCT_RELEASE_MONTH_INDEX = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "merchant_activity.csv"
)
CHART_FILE = (
    PROJECT_ROOT
    / "reports"
    / "dataset_overview.png"
)

rng = np.random.default_rng(RANDOM_SEED)


def generate_merchant_profiles() -> pd.DataFrame:
    merchant_ids = [
        f"MER-{number:04d}"
        for number in range(
            1,
            NUMBER_OF_MERCHANTS + 1,
        )
    ]

    plan_tiers = rng.choice(
        [
            "Growth",
            "Professional",
            "Enterprise",
        ],
        size=NUMBER_OF_MERCHANTS,
        p=[0.50, 0.35, 0.15],
    )

    contract_ranges = {
        "Growth": (10_000, 20_000),
        "Professional": (20_000, 35_000),
        "Enterprise": (35_000, 50_000),
    }

    annual_contract_values = [
        round(
            rng.uniform(
                *contract_ranges[plan]
            ),
            2,
        )
        for plan in plan_tiers
    ]

    return pd.DataFrame(
        {
            "merchant_id": merchant_ids,
            "plan_tier": plan_tiers,
            "annual_contract_value": (
                annual_contract_values
            ),
            "baseline_logins_per_week": np.clip(
                rng.normal(
                    12,
                    4,
                    NUMBER_OF_MERCHANTS,
                ),
                2,
                30,
            ),
            "baseline_feature_adoption_pct": (
                np.clip(
                    rng.normal(
                        65,
                        15,
                        NUMBER_OF_MERCHANTS,
                    ),
                    15,
                    100,
                )
            ),
        }
    )


def assign_churn_outcomes(
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    profiles = profiles.copy()

    login_risk = 1 - np.clip(
        profiles["baseline_logins_per_week"]
        / 20,
        0,
        1,
    )

    adoption_risk = (
        1
        - profiles[
            "baseline_feature_adoption_pct"
        ]
        / 100
    )

    growth_plan_risk = (
        profiles["plan_tier"] == "Growth"
    ).astype(int) * 0.20

    risk_score = (
        -2.3
        + login_risk
        + (1.4 * adoption_risk)
        + growth_plan_risk
    )

    churn_probability = (
        1 / (1 + np.exp(-risk_score))
    )

    profiles["will_churn"] = (
        rng.random(NUMBER_OF_MERCHANTS)
        < churn_probability
    )

    profiles["churn_month_index"] = np.where(
        profiles["will_churn"],
        rng.integers(
            12,
            NUMBER_OF_MONTHS,
            NUMBER_OF_MERCHANTS,
        ),
        -1,
    )

    profiles["shows_decline"] = (
        rng.random(NUMBER_OF_MERCHANTS)
        < 0.92
    )

    profiles["decline_duration"] = (
        rng.integers(
            2,
            7,
            NUMBER_OF_MERCHANTS,
        )
    )

    profiles["decline_intensity"] = (
        rng.uniform(
            0.55,
            0.90,
            NUMBER_OF_MERCHANTS,
        )
    )

    profiles["has_temporary_slump"] = (
        (~profiles["will_churn"])
        & (
            rng.random(
                NUMBER_OF_MERCHANTS
            )
            < 0.06
        )
    )

    profiles["slump_start"] = rng.integers(
        8,
        15,
        NUMBER_OF_MERCHANTS,
    )

    profiles["slump_duration"] = rng.integers(
        2,
        5,
        NUMBER_OF_MERCHANTS,
    )

    profiles["slump_intensity"] = rng.uniform(
        0.20,
        0.55,
        NUMBER_OF_MERCHANTS,
    )

    profiles["renewal_offset_days"] = (
        (
            np.arange(NUMBER_OF_MERCHANTS)
            * 47
        )
        % 335
    ) + 30

    return profiles


def generate_monthly_activity(
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    months = pd.date_range(
        START_DATE,
        periods=NUMBER_OF_MONTHS,
        freq="MS",
    )

    records = []

    for merchant in profiles.itertuples(
        index=False
    ):
        for month_index, month in enumerate(
            months
        ):
            churn_month = int(
                merchant.churn_month_index
            )

            if (
                merchant.will_churn
                and month_index > churn_month
            ):
                continue

            months_to_churn = (
                churn_month - month_index
                if merchant.will_churn
                else 999
            )

            decline = 0.0

            if (
                merchant.will_churn
                and merchant.shows_decline
                and 0 <= months_to_churn
                < merchant.decline_duration
            ):
                decline = (
                    merchant.decline_intensity
                    * (
                        merchant.decline_duration
                        - months_to_churn
                    )
                    / merchant.decline_duration
                )

            if (
                merchant.has_temporary_slump
                and merchant.slump_start
                <= month_index
                < (
                    merchant.slump_start
                    + merchant.slump_duration
                )
            ):
                slump_position = (
                    month_index
                    - merchant.slump_start
                )

                midpoint = (
                    merchant.slump_duration
                    - 1
                ) / 2

                slump = (
                    merchant.slump_intensity
                    * (
                        1
                        - abs(
                            slump_position
                            - midpoint
                        )
                        / max(
                            1,
                            merchant.slump_duration
                            / 2,
                        )
                    )
                )

                decline = max(
                    decline,
                    slump,
                )

            release_strength = 0.0

            if (
                month_index
                == PRODUCT_RELEASE_MONTH_INDEX
            ):
                release_strength = 1.0

            elif (
                month_index
                == PRODUCT_RELEASE_MONTH_INDEX
                + 1
            ):
                release_strength = 0.5

            logins = (
                merchant.baseline_logins_per_week
                * (1 - 0.70 * decline)
                * (
                    1
                    - 0.18
                    * release_strength
                )
                + rng.normal(0, 1.8)
            )

            feature_adoption = (
                merchant.baseline_feature_adoption_pct
                - (30 * decline)
                - (6 * release_strength)
                + rng.normal(0, 5)
            )

            expected_tickets = max(
                0.1,
                1.4
                + (3 * decline)
                + (
                    1.2
                    * release_strength
                )
                + rng.normal(0, 0.4),
            )

            support_tickets = rng.poisson(
                expected_tickets
            )

            support_sentiment = (
                0.45
                - (0.90 * decline)
                - (
                    0.25
                    * release_strength
                )
                + rng.normal(0, 0.22)
            )

            days_to_renewal = int(
                (
                    merchant.renewal_offset_days
                    - month_index * 30
                )
                % 365
            )

            records.append(
                {
                    "merchant_id": (
                        merchant.merchant_id
                    ),
                    "month": month,
                    "month_index": month_index,
                    "plan_tier": (
                        merchant.plan_tier
                    ),
                    "annual_contract_value": (
                        round(
                            merchant
                            .annual_contract_value,
                            2,
                        )
                    ),
                    "logins_per_week": round(
                        max(0, logins),
                        2,
                    ),
                    "feature_adoption_pct": (
                        round(
                            np.clip(
                                feature_adoption,
                                0,
                                100,
                            ),
                            2,
                        )
                    ),
                    "support_tickets": (
                        support_tickets
                    ),
                    "support_sentiment": round(
                        np.clip(
                            support_sentiment,
                            -1,
                            1,
                        ),
                        3,
                    ),
                    "days_to_renewal": (
                        days_to_renewal
                    ),
                    "product_release_event": int(
                        release_strength > 0
                    ),
                    "churn_within_3_months": int(
                        merchant.will_churn
                        and 0
                        <= months_to_churn
                        <= 3
                    ),
                    "churned_this_month": int(
                        merchant.will_churn
                        and months_to_churn == 0
                    ),
                }
            )

    return pd.DataFrame(records)


def create_overview_chart(
    dataset: pd.DataFrame,
) -> None:
    monthly = (
        dataset.groupby("month")
        .agg(
            average_logins=(
                "logins_per_week",
                "mean",
            ),
            average_adoption=(
                "feature_adoption_pct",
                "mean",
            ),
        )
        .reset_index()
    )

    release_date = (
        pd.Timestamp(START_DATE)
        + pd.DateOffset(
            months=(
                PRODUCT_RELEASE_MONTH_INDEX
            )
        )
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
    )

    axes[0].plot(
        monthly["month"],
        monthly["average_logins"],
        marker="o",
    )

    axes[0].axvline(
        release_date,
        color="red",
        linestyle="--",
        label="Product release",
    )

    axes[0].set_title(
        "Average Weekly Logins"
    )

    axes[0].legend()

    axes[1].plot(
        monthly["month"],
        monthly["average_adoption"],
        marker="o",
        color="green",
    )

    axes[1].axvline(
        release_date,
        color="red",
        linestyle="--",
        label="Product release",
    )

    axes[1].set_title(
        "Average Feature Adoption"
    )

    axes[1].legend()

    figure.tight_layout()

    figure.savefig(
        CHART_FILE,
        dpi=150,
    )

    plt.close(figure)


def main() -> None:
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHART_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profiles = generate_merchant_profiles()

    profiles = assign_churn_outcomes(
        profiles
    )

    dataset = generate_monthly_activity(
        profiles
    )

    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    create_overview_chart(dataset)

    churned_merchants = dataset.loc[
        dataset["churned_this_month"] == 1,
        "merchant_id",
    ].nunique()

    print(
        f"Rows generated: "
        f"{len(dataset):,}"
    )

    print(
        f"Unique merchants: "
        f"{dataset['merchant_id'].nunique():,}"
    )

    print(
        f"Merchants that churned: "
        f"{churned_merchants:,}"
    )

    print(
        f"Prediction target rate: "
        f"{dataset['churn_within_3_months'].mean():.2%}"
    )

    print(
        f"Dataset saved to: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Chart saved to: "
        f"{CHART_FILE}"
    )


if __name__ == "__main__":
    main()