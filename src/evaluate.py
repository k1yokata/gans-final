import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
SYNTHETIC_DIR = OUTPUT_DIR / "synthetic"
PREPROCESSOR_DIR = OUTPUT_DIR / "preprocessors"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"
IMAGE_DIR = OUTPUT_DIR / "images"

EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def find_real_dataset() -> Path:

    possible_files = [
        PROCESSED_DIR / "procurement_data.csv",
        PROCESSED_DIR / "test_procurement_data.csv",
        PROCESSED_DIR / "processed_procurement_data.csv",
    ]

    for file_path in possible_files:
        if file_path.exists():
            return file_path

    csv_files = list(PROCESSED_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No processed CSV files found in {PROCESSED_DIR}. "
            f"Run stage 4 first: python3 -m src.dataset"
        )

    return csv_files[0]


def load_metadata() -> dict:


    metadata_path = PREPROCESSOR_DIR / "gan_training_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}. "
            f"Run stage 5 first: python3 -m src.train"
        )

    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_real_data(metadata: dict) -> pd.DataFrame:

    real_dataset_path = find_real_dataset()

    print(f"Loading real data: {real_dataset_path}")

    df = pd.read_csv(real_dataset_path)

    id_columns = [
        "tender_id",
        "buyer_id",
        "supplier_id",
    ]

    df = df.drop(columns=[col for col in id_columns if col in df.columns])

    df_encoded = pd.get_dummies(df, drop_first=False)

    numeric_df = df_encoded.select_dtypes(include=["number", "bool"]).copy()
    numeric_df = numeric_df.fillna(0)

    for col in numeric_df.columns:
        numeric_df[col] = numeric_df[col].astype(float)

    expected_columns = metadata["columns"]

    for col in expected_columns:
        if col not in numeric_df.columns:
            numeric_df[col] = 0.0

    numeric_df = numeric_df[expected_columns]

    return numeric_df


def load_synthetic_data(metadata: dict) -> pd.DataFrame:

    synthetic_path = SYNTHETIC_DIR / "synthetic_procurement_data.csv"

    if not synthetic_path.exists():
        raise FileNotFoundError(
            f"Synthetic data file not found: {synthetic_path}. "
            f"Run stage 5 first: python3 -m src.train"
        )

    print(f"Loading synthetic data: {synthetic_path}")

    synthetic_df = pd.read_csv(synthetic_path)

    expected_columns = metadata["columns"]

    for col in expected_columns:
        if col not in synthetic_df.columns:
            synthetic_df[col] = 0.0

    synthetic_df = synthetic_df[expected_columns]
    synthetic_df = synthetic_df.fillna(0)

    for col in synthetic_df.columns:
        synthetic_df[col] = synthetic_df[col].astype(float)

    return synthetic_df


def calculate_column_metrics(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Считает статистики по каждой колонке.
    """

    rows = []

    for column in real_df.columns:
        real_col = real_df[column]
        synthetic_col = synthetic_df[column]

        real_mean = real_col.mean()
        synthetic_mean = synthetic_col.mean()

        real_std = real_col.std()
        synthetic_std = synthetic_col.std()

        mean_diff = abs(real_mean - synthetic_mean)
        std_diff = abs(real_std - synthetic_std)

        real_min = real_col.min()
        synthetic_min = synthetic_col.min()

        real_max = real_col.max()
        synthetic_max = synthetic_col.max()

        rows.append(
            {
                "column": column,
                "real_mean": real_mean,
                "synthetic_mean": synthetic_mean,
                "mean_abs_diff": mean_diff,
                "real_std": real_std,
                "synthetic_std": synthetic_std,
                "std_abs_diff": std_diff,
                "real_min": real_min,
                "synthetic_min": synthetic_min,
                "real_max": real_max,
                "synthetic_max": synthetic_max,
            }
        )

    metrics_df = pd.DataFrame(rows)

    return metrics_df


def calculate_correlation_difference(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> float:
    """
    Сравнение корреляционной матрицы реальных и синтетических данных.
    Чем меньше значение, тем лучше.
    """

    real_corr = real_df.corr().fillna(0)
    synthetic_corr = synthetic_df.corr().fillna(0)

    difference = (real_corr - synthetic_corr).abs().mean().mean()

    real_corr_path = EVALUATION_DIR / "real_correlation_matrix.csv"
    synthetic_corr_path = EVALUATION_DIR / "synthetic_correlation_matrix.csv"

    real_corr.to_csv(real_corr_path)
    synthetic_corr.to_csv(synthetic_corr_path)

    return float(difference)


def calculate_overall_score(metrics_df: pd.DataFrame, correlation_diff: float) -> dict:

    avg_mean_diff = float(metrics_df["mean_abs_diff"].mean())
    avg_std_diff = float(metrics_df["std_abs_diff"].mean())

    report = {
        "average_mean_absolute_difference": avg_mean_diff,
        "average_std_absolute_difference": avg_std_diff,
        "average_correlation_difference": correlation_diff,
        "interpretation": {
            "mean_difference": "Lower value means synthetic data has closer average values to real data.",
            "std_difference": "Lower value means synthetic data has closer variability to real data.",
            "correlation_difference": "Lower value means synthetic data preserves relationships between features better.",
        },
    }

    return report


def plot_distribution_comparison(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    column: str,
) -> None:


    plt.figure(figsize=(10, 6))

    plt.hist(
        real_df[column],
        bins=30,
        alpha=0.6,
        label="Real data",
        density=True,
    )

    plt.hist(
        synthetic_df[column],
        bins=30,
        alpha=0.6,
        label="Synthetic data",
        density=True,
    )

    plt.title(f"Distribution comparison: {column}")
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    safe_column_name = column.replace("/", "_").replace(" ", "_")
    output_path = IMAGE_DIR / f"distribution_{safe_column_name}.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Distribution plot saved: {output_path}")


def plot_top_metric_differences(metrics_df: pd.DataFrame) -> None:
    top_metrics = metrics_df.sort_values(
        by="mean_abs_diff",
        ascending=False,
    ).head(10)

    plt.figure(figsize=(12, 6))

    plt.bar(
        top_metrics["column"],
        top_metrics["mean_abs_diff"],
    )

    plt.title("Top 10 columns by mean absolute difference")
    plt.xlabel("Column")
    plt.ylabel("Mean absolute difference")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")

    output_path = IMAGE_DIR / "top_mean_differences.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Top differences plot saved: {output_path}")


def run_evaluation() -> None:
    print("Synthetic data evaluation started")

    metadata = load_metadata()

    real_df = load_real_data(metadata)
    synthetic_df = load_synthetic_data(metadata)

    print(f"Real data shape: {real_df.shape}")
    print(f"Synthetic data shape: {synthetic_df.shape}")

    metrics_df = calculate_column_metrics(
        real_df=real_df,
        synthetic_df=synthetic_df,
    )

    metrics_path = EVALUATION_DIR / "column_quality_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Column metrics saved to: {metrics_path}")

    correlation_diff = calculate_correlation_difference(
        real_df=real_df,
        synthetic_df=synthetic_df,
    )

    report = calculate_overall_score(
        metrics_df=metrics_df,
        correlation_diff=correlation_diff,
    )

    report_path = EVALUATION_DIR / "synthetic_data_quality_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print(f"Quality report saved to: {report_path}")

    important_columns = [
        "expected_value",
        "final_value",
        "price_reduction_percent",
        "number_of_bidders",
        "supplier_previous_wins",
        "buyer_previous_tenders",
        "contract_days",
        "risk_score",
        "risk_label",
    ]

    for column in important_columns:
        if column in real_df.columns and column in synthetic_df.columns:
            plot_distribution_comparison(
                real_df=real_df,
                synthetic_df=synthetic_df,
                column=column,
            )

    plot_top_metric_differences(metrics_df)

    print()
    print("Evaluation summary")
    print("------------------")
    print(f"Average mean absolute difference: {report['average_mean_absolute_difference']:.4f}")
    print(f"Average std absolute difference: {report['average_std_absolute_difference']:.4f}")
    print(f"Average correlation difference: {report['average_correlation_difference']:.4f}")
    print()
    print("Synthetic data evaluation finished successfully")


if __name__ == "__main__":
    run_evaluation()