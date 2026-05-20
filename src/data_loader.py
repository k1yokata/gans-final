from pathlib import Path
import pandas as pd


def read_csv_sample(
    file_path: str | Path,
    nrows: int = 10000,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Reads a limited number of rows from a CSV file.

    This function is used to avoid high memory usage on a local machine.
    Large datasets should be processed in small samples or in Google Colab.

    Args:
        file_path: Path to the CSV file.
        nrows: Number of rows to read.
        usecols: Optional list of columns to read.

    Returns:
        Pandas DataFrame with selected rows and columns.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(
        file_path,
        nrows=nrows,
        usecols=usecols,
        low_memory=False,
    )

    return df


def show_dataset_info(df: pd.DataFrame) -> None:
    """
    Prints basic dataset information.
    """
    print("Dataset shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nPreview:")
    print(df.head())
