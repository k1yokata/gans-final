from pathlib import Path
import json
import joblib
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PREPROCESSOR_DIR = OUTPUT_DIR / "preprocessors"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSOR_DIR.mkdir(parents=True, exist_ok=True)


DATASET_PATH = PROCESSED_DIR / "test_procurement_data.csv"
PREPROCESSOR_PATH = PREPROCESSOR_DIR / "procurement_preprocessor.joblib"
METADATA_PATH = PREPROCESSOR_DIR / "procurement_metadata.json"

BATCH_SIZE = 64


DROP_COLUMNS = [
    "tender_id",
    "buyer_id",
    "supplier_id",
]

TARGET_COLUMNS = [
    "risk_score",
    "risk_label",
]


def load_procurement_data(path: Path = DATASET_PATH) -> pd.DataFrame:
    """
    Loads procurement dataset from CSV.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Run first: python3 -m src.generate_test_data"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df


def split_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes IDs and target columns from the dataset.
    GAN should learn feature distribution, not direct labels.
    """
    columns_to_drop = []

    for column in DROP_COLUMNS + TARGET_COLUMNS:
        if column in df.columns:
            columns_to_drop.append(column)

    features = df.drop(columns=columns_to_drop)

    return features


def detect_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Detects numerical and categorical columns.
    """
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_columns = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    return numerical_columns, categorical_columns


def build_preprocessor(
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """
    Builds preprocessing pipeline:
    - numerical columns -> StandardScaler
    - categorical columns -> OneHotEncoder
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_dataset(df: pd.DataFrame):
    """
    Prepares dataset for GAN training.
    Returns:
    - tensor dataset
    - input dimension
    - processed numpy data
    - metadata
    """
    features = split_features(df)

    numerical_columns, categorical_columns = detect_columns(features)

    if not numerical_columns and not categorical_columns:
        raise ValueError("No usable columns found for training.")

    preprocessor = build_preprocessor(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
    )

    processed_data = preprocessor.fit_transform(features)

    processed_tensor = torch.tensor(processed_data, dtype=torch.float32)

    dataset = TensorDataset(processed_tensor)

    metadata = {
        "original_columns": df.columns.tolist(),
        "used_columns": features.columns.tolist(),
        "dropped_columns": [col for col in DROP_COLUMNS + TARGET_COLUMNS if col in df.columns],
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "input_dim": processed_data.shape[1],
        "rows": processed_data.shape[0],
    }

    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    return dataset, processed_data.shape[1], processed_data, metadata


def create_dataloader(batch_size: int = BATCH_SIZE, shuffle: bool = True) -> tuple[DataLoader, int, dict]:
    """
    Creates PyTorch DataLoader for GAN training.
    """
    df = load_procurement_data()
    dataset, input_dim, _, metadata = prepare_dataset(df)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )

    return dataloader, input_dim, metadata


def main():
    df = load_procurement_data()
    dataset, input_dim, processed_data, metadata = prepare_dataset(df)

    print("Procurement dataset prepared successfully.")
    print(f"Original dataset path: {DATASET_PATH}")
    print(f"Original shape: {df.shape}")
    print(f"Processed shape: {processed_data.shape}")
    print(f"Input dimension for GAN: {input_dim}")
    print()
    print("Numerical columns:")
    print(metadata["numerical_columns"])
    print()
    print("Categorical columns:")
    print(metadata["categorical_columns"])
    print()
    print(f"Preprocessor saved to: {PREPROCESSOR_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()