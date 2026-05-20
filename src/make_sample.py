from pathlib import Path
import pandas as pd

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data_loader import read_csv_sample, show_dataset_info


DEFAULT_SAMPLE_SIZE = 10000


def find_csv_files(raw_data_dir: Path) -> list[Path]:
    """
    Finds all CSV files in the raw data directory.
    """
    return sorted(raw_data_dir.glob("*.csv"))


def create_sample(
    input_file: Path,
    output_file: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> None:
    """
    Creates a small sample from a large CSV file.
    """
    print(f"Reading file: {input_file}")
    print(f"Sample size: {sample_size}")

    df = read_csv_sample(
        file_path=input_file,
        nrows=sample_size,
    )

    show_dataset_info(df)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"\nSample saved to: {output_file}")


def main() -> None:
    csv_files = find_csv_files(RAW_DATA_DIR)

    if not csv_files:
        print("No CSV files found in data/raw/")
        print("Please place raw procurement CSV files into data/raw/")
        return

    print("Available CSV files:")

    for index, file_path in enumerate(csv_files, start=1):
        print(f"{index}. {file_path.name}")

    selected_file = csv_files[0]

    output_file = PROCESSED_DATA_DIR / "procurement_sample.csv"

    create_sample(
        input_file=selected_file,
        output_file=output_file,
        sample_size=DEFAULT_SAMPLE_SIZE,
    )


if __name__ == "__main__":
    main()
