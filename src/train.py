import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import Generator, Discriminator


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
IMAGE_DIR = OUTPUT_DIR / "images"
SYNTHETIC_DIR = OUTPUT_DIR / "synthetic"
PREPROCESSOR_DIR = OUTPUT_DIR / "preprocessors"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

LATENT_DIM = 100
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0002
BETA1 = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_processed_dataset() -> Path:
    """
    Находит подготовленный CSV-файл после этапа 4.
    """

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


def load_dataset() -> tuple[torch.Tensor, list[str], pd.DataFrame]:
    """
    Загружает подготовленный датасет и оставляет только числовые признаки.
    """

    dataset_path = find_processed_dataset()

    print(f"Loading dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)

    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    id_columns = [
        "tender_id",
        "buyer_id",
        "supplier_id",
    ]

    df_for_training = df.drop(columns=[col for col in id_columns if col in df.columns])

    df_encoded = pd.get_dummies(df_for_training, drop_first=False)

    numeric_df = df_encoded.select_dtypes(include=["number", "bool"]).copy()

    numeric_df = numeric_df.fillna(0)

    for col in numeric_df.columns:
        numeric_df[col] = numeric_df[col].astype(float)

    min_values = numeric_df.min()
    max_values = numeric_df.max()

    normalized_df = 2 * (numeric_df - min_values) / (max_values - min_values + 1e-8) - 1

    tensor_data = torch.tensor(normalized_df.values, dtype=torch.float32)

    metadata = {
        "dataset_path": str(dataset_path),
        "original_shape": list(df.shape),
        "training_shape": list(normalized_df.shape),
        "columns": list(normalized_df.columns),
        "min_values": min_values.to_dict(),
        "max_values": max_values.to_dict(),
    }

    metadata_path = PREPROCESSOR_DIR / "gan_training_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print(f"Training shape: {normalized_df.shape}")
    print(f"Metadata saved to: {metadata_path}")

    return tensor_data, list(normalized_df.columns), normalized_df


def denormalize_data(
    generated_data: torch.Tensor,
    columns: list[str],
    metadata_path: Path,
) -> pd.DataFrame:
    """
    Возвращает сгенерированные данные из диапазона [-1, 1]
    обратно в исходный числовой диапазон.
    """

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    min_values = pd.Series(metadata["min_values"])
    max_values = pd.Series(metadata["max_values"])

    generated_df = pd.DataFrame(
        generated_data.detach().cpu().numpy(),
        columns=columns,
    )

    generated_df = (generated_df + 1) / 2
    generated_df = generated_df * (max_values - min_values + 1e-8) + min_values

    return generated_df


def save_loss_plot(generator_losses: list[float], discriminator_losses: list[float]) -> None:
    """
    Сохраняет график обучения GAN.
    """

    plt.figure(figsize=(10, 6))
    plt.plot(generator_losses, label="Generator loss")
    plt.plot(discriminator_losses, label="Discriminator loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss")
    plt.legend()
    plt.grid(True)

    output_path = IMAGE_DIR / "gan_training_loss.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Loss plot saved to: {output_path}")


def train() -> None:
    print("GAN training started")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {DEVICE}")

    data_tensor, columns, _ = load_dataset()

    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    data_dim = data_tensor.shape[1]

    generator = Generator(
        latent_dim=LATENT_DIM,
        output_dim=data_dim,
    ).to(DEVICE)

    discriminator = Discriminator(
        input_dim=data_dim,
    ).to(DEVICE)

    criterion = nn.BCELoss()

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, 0.999),
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, 0.999),
    )

    generator_losses = []
    discriminator_losses = []

    for epoch in range(1, EPOCHS + 1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for real_batch, in dataloader:
            real_batch = real_batch.to(DEVICE)
            batch_size = real_batch.size(0)

            real_labels = torch.ones(batch_size, 1).to(DEVICE)
            fake_labels = torch.zeros(batch_size, 1).to(DEVICE)

            # =========================
            # Train Discriminator
            # =========================
            optimizer_d.zero_grad()

            real_outputs = discriminator(real_batch)
            real_loss = criterion(real_outputs, real_labels)

            noise = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            fake_data = generator(noise)

            fake_outputs = discriminator(fake_data.detach())
            fake_loss = criterion(fake_outputs, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # =========================
            # Train Generator
            # =========================
            optimizer_g.zero_grad()

            noise = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            fake_data = generator(noise)

            fake_outputs = discriminator(fake_data)
            g_loss = criterion(fake_outputs, real_labels)

            g_loss.backward()
            optimizer_g.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_g_loss = epoch_g_loss / len(dataloader)

        discriminator_losses.append(avg_d_loss)
        generator_losses.append(avg_g_loss)

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"D Loss: {avg_d_loss:.4f} | "
            f"G Loss: {avg_g_loss:.4f}"
        )

    generator_path = MODEL_DIR / "generator.pth"
    discriminator_path = MODEL_DIR / "discriminator.pth"

    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)

    print(f"Generator saved to: {generator_path}")
    print(f"Discriminator saved to: {discriminator_path}")

    save_loss_plot(generator_losses, discriminator_losses)

    # =========================
    # Generate synthetic data
    # =========================
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(1000, LATENT_DIM).to(DEVICE)
        synthetic_tensor = generator(noise)

    metadata_path = PREPROCESSOR_DIR / "gan_training_metadata.json"

    synthetic_df = denormalize_data(
        generated_data=synthetic_tensor,
        columns=columns,
        metadata_path=metadata_path,
    )

    synthetic_path = SYNTHETIC_DIR / "synthetic_procurement_data.csv"
    synthetic_df.to_csv(synthetic_path, index=False)

    print(f"Synthetic data saved to: {synthetic_path}")
    print("GAN training finished successfully")


if __name__ == "__main__":
    train()