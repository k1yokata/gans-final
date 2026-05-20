from pathlib import Path
import random
import pandas as pd
import numpy as np

from src.config import PROCESSED_DATA_DIR


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


def generate_procurement_data(rows: int = 5000) -> pd.DataFrame:
    """
    Generates a synthetic public procurement dataset.

    The dataset imitates tender records and includes risk-related features:
    tender value, number of bidders, procedure type, supplier frequency,
    buyer frequency, price reduction, and risk score.
    """

    procedure_types = [
        "open_tender",
        "limited_tender",
        "direct_award",
        "negotiation",
    ]

    regions = [
        "Almaty",
        "Astana",
        "Shymkent",
        "Karaganda",
        "Aktobe",
        "Atyrau",
        "Kostanay",
        "Pavlodar",
    ]

    data = []

    for i in range(1, rows + 1):
        tender_id = f"T-{i:06d}"

        buyer_id = f"B-{np.random.randint(1, 301):04d}"
        supplier_id = f"S-{np.random.randint(1, 501):04d}"

        procedure_type = np.random.choice(
            procedure_types,
            p=[0.58, 0.18, 0.16, 0.08],
        )

        region = np.random.choice(regions)

        expected_value = float(
            np.random.lognormal(mean=12.5, sigma=0.9)
        )

        if procedure_type == "open_tender":
            number_of_bidders = np.random.randint(2, 9)
        elif procedure_type == "limited_tender":
            number_of_bidders = np.random.randint(1, 4)
        elif procedure_type == "direct_award":
            number_of_bidders = 1
        else:
            number_of_bidders = np.random.randint(1, 3)

        price_reduction_percent = float(
            np.clip(
                np.random.normal(loc=8, scale=6),
                0,
                35,
            )
        )

        if number_of_bidders == 1:
            price_reduction_percent = float(
                np.clip(
                    np.random.normal(loc=1.5, scale=1.2),
                    0,
                    5,
                )
            )

        final_value = expected_value * (1 - price_reduction_percent / 100)

        supplier_previous_wins = int(
            np.random.poisson(lam=8)
        )

        buyer_previous_tenders = int(
            np.random.poisson(lam=20)
        )

        contract_days = int(
            np.clip(
                np.random.normal(loc=90, scale=35),
                10,
                365,
            )
        )

        risk_score = 0

        if number_of_bidders == 1:
            risk_score += 30

        if procedure_type in ["direct_award", "negotiation"]:
            risk_score += 25

        if price_reduction_percent < 2:
            risk_score += 15

        if supplier_previous_wins > 15:
            risk_score += 15

        if expected_value > 1_000_000:
            risk_score += 10

        if contract_days < 20:
            risk_score += 5

        risk_score = min(risk_score, 100)

        if risk_score >= 60:
            risk_label = 1
        else:
            risk_label = 0

        data.append(
            {
                "tender_id": tender_id,
                "buyer_id": buyer_id,
                "supplier_id": supplier_id,
                "region": region,
                "procedure_type": procedure_type,
                "expected_value": round(expected_value, 2),
                "final_value": round(final_value, 2),
                "price_reduction_percent": round(price_reduction_percent, 2),
                "number_of_bidders": number_of_bidders,
                "supplier_previous_wins": supplier_previous_wins,
                "buyer_previous_tenders": buyer_previous_tenders,
                "contract_days": contract_days,
                "risk_score": risk_score,
                "risk_label": risk_label,
            }
        )

    return pd.DataFrame(data)


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_procurement_data(rows=5000)

    output_path = PROCESSED_DATA_DIR / "test_procurement_data.csv"
    df.to_csv(output_path, index=False)

    print("Test procurement dataset created successfully.")
    print(f"Path: {output_path}")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()
