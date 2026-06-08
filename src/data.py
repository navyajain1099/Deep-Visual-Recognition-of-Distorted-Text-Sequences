from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from src.charset import Charset


IMAGE_COLS = ("image", "filename", "file", "id", "path")
LABEL_COLS = ("label", "text", "prediction", "target")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_label_csv(data_dir: Path) -> Path:
    candidates = [
        data_dir / "train_labels.csv",
        data_dir / "labels.csv",
        data_dir / "train.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    csvs = sorted(data_dir.glob("*.csv"))
    if csvs:
        return csvs[0]
    raise FileNotFoundError(f"No label CSV found in {data_dir}")


def find_image_dir(data_dir: Path, split: str) -> Path:
    names = {
        "train": ("train", "train_images", "images/train"),
        "test": ("test", "test_images", "images/test"),
    }[split]
    for name in names:
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {split} image directory found in {data_dir}")


def read_labels(data_dir: Path) -> pd.DataFrame:
    csv_path = find_label_csv(data_dir)
    df = pd.read_csv(csv_path)
    lower = {col.lower(): col for col in df.columns}

    image_col = next((lower[col] for col in IMAGE_COLS if col in lower), None)
    label_col = next((lower[col] for col in LABEL_COLS if col in lower), None)
    if image_col is None or label_col is None:
        raise ValueError(
            f"{csv_path} must have image and label columns. Found: {list(df.columns)}"
        )

    out = df[[image_col, label_col]].copy()
    out.columns = ["image", "label"]
    out["image"] = out["image"].astype(str)
    out["label"] = out["label"].astype(str)
    return out


def list_test_images(data_dir: Path) -> list[Path]:
    image_dir = find_image_dir(data_dir, "test")
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)


class MildDistortion:
    def __call__(self, image: Image.Image) -> Image.Image:
        if torch.rand(1).item() < 0.25:
            image = image.filter(ImageFilter.GaussianBlur(radius=float(torch.rand(1).item() * 0.8)))
        return image


def image_transform(img_height: int, img_width: int, augment: bool) -> transforms.Compose:
    ops = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_height, img_width)),
    ]
    if augment:
        ops.extend(
            [
                MildDistortion(),
                transforms.RandomApply([transforms.RandomAffine(degrees=3, translate=(0.03, 0.08), shear=4)], p=0.45),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.35)], p=0.45),
            ]
        )
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    return transforms.Compose(ops)


class SequenceDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        image_dir: Path,
        charset: Charset,
        img_height: int,
        img_width: int,
        augment: bool = False,
    ):
        self.records = records.reset_index(drop=True)
        self.image_dir = image_dir
        self.charset = charset
        self.transform = image_transform(img_height, img_width, augment)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records.iloc[idx]
        image_path = self.image_dir / row.image
        image = Image.open(image_path).convert("L")
        label = row.label
        encoded = torch.tensor(self.charset.encode(label), dtype=torch.long)
        return {
            "image": self.transform(image),
            "target": encoded,
            "target_length": torch.tensor(len(encoded), dtype=torch.long),
            "label": label,
            "filename": row.image,
        }


class TestSequenceDataset(Dataset):
    def __init__(self, image_paths: list[Path], img_height: int, img_width: int):
        self.image_paths = image_paths
        self.transform = image_transform(img_height, img_width, augment=False)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("L")
        return {"image": self.transform(image), "filename": image_path.name}


def collate_train(batch: list[dict]) -> dict:
    images = torch.stack([item["image"] for item in batch])
    targets = torch.cat([item["target"] for item in batch])
    target_lengths = torch.stack([item["target_length"] for item in batch])
    return {
        "image": images,
        "target": targets,
        "target_length": target_lengths,
        "label": [item["label"] for item in batch],
        "filename": [item["filename"] for item in batch],
    }


def split_records(records: pd.DataFrame, val_split: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    lengths = records["label"].str.len()
    stratify = lengths if lengths.value_counts().min() >= 2 else None
    train_df, val_df = train_test_split(
        records,
        test_size=val_split,
        random_state=seed,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

