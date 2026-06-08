from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.charset import Charset
from src.data import TestSequenceDataset, list_test_images
from src.model import CRNN
from src.train import decode_batch
from src.utils import device_name, ensure_dir


def predict(data_dir: Path, checkpoint: Path, out: Path, img_height: int = 64, img_width: int = 256, batch_size: int = 64) -> Path:
    device = torch.device(device_name())
    payload = torch.load(checkpoint, map_location=device)
    charset = Charset(payload["alphabet"])

    model = CRNN(num_classes=charset.num_classes).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    image_paths = list_test_images(data_dir)
    dataset = TestSequenceDataset(image_paths, img_height=img_height, img_width=img_width)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device)
            logits = model(images)
            predictions = decode_batch(logits, charset)
            rows.extend({"image": name, "prediction": pred} for name, pred in zip(batch["filename"], predictions))

    ensure_dir(out.parent)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} predictions to {out}")
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/best.pt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--img-height", type=int, default=64)
    parser.add_argument("--img-width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.data_dir, args.checkpoint, args.out, args.img_height, args.img_width, args.batch_size)

