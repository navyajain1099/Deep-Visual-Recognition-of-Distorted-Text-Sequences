from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.charset import Charset
from src.config import TrainConfig
from src.data import SequenceDataset, collate_train, find_image_dir, read_labels, split_records
from src.metrics import cer
from src.model import CRNN
from src.utils import device_name, ensure_dir, seed_everything


def decode_batch(logits: torch.Tensor, charset: Charset) -> list[str]:
    token_ids = logits.softmax(dim=-1).argmax(dim=-1).transpose(0, 1).cpu().tolist()
    return [charset.decode_ctc(tokens) for tokens in token_ids]


def run_epoch(model, loader, criterion, optimizer, device, charset, train: bool) -> tuple[float, float]:
    model.train(train)
    losses = []
    predictions = []
    targets = []
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device)
            target = batch["target"].to(device)
            target_length = batch["target_length"].to(device)

            logits = model(images)
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=logits.size(0),
                dtype=torch.long,
                device=device,
            )
            loss = criterion(logits.log_softmax(2), target, input_lengths, target_length)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            losses.append(loss.item())
            predictions.extend(decode_batch(logits.detach(), charset))
            targets.extend(batch["label"])

    return sum(losses) / max(1, len(losses)), cer(predictions, targets)


def train(config: TrainConfig) -> Path:
    seed_everything(config.seed)
    device = torch.device(device_name())
    checkpoint_dir = ensure_dir(config.output_dir / "checkpoints")

    records = read_labels(config.data_dir)
    observed_chars = "".join(sorted(set("".join(records["label"].astype(str)))))
    alphabet = "".join(dict.fromkeys(config.alphabet + observed_chars))
    charset = Charset(alphabet)
    train_df, val_df = split_records(records, config.val_split, config.seed)
    train_image_dir = find_image_dir(config.data_dir, "train")

    train_ds = SequenceDataset(train_df, train_image_dir, charset, config.img_height, config.img_width, augment=True)
    val_ds = SequenceDataset(val_df, train_image_dir, charset, config.img_height, config.img_width, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_train,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_train,
        pin_memory=torch.cuda.is_available(),
    )

    model = CRNN(num_classes=charset.num_classes).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    best_cer = float("inf")
    stale_epochs = 0
    best_path = checkpoint_dir / "best.pt"

    for epoch in range(1, config.epochs + 1):
        train_loss, train_cer = run_epoch(model, train_loader, criterion, optimizer, device, charset, train=True)
        val_loss, val_cer = run_epoch(model, val_loader, criterion, optimizer, device, charset, train=False)
        scheduler.step(val_cer)

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} train_cer={train_cer:.4f} "
            f"val_loss={val_loss:.4f} val_cer={val_cer:.4f}"
        )

        if val_cer < best_cer:
            best_cer = val_cer
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "alphabet": charset.alphabet,
                    "config": config.__dict__,
                    "val_cer": best_cer,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"Early stopping after {config.patience} stale epochs.")
                break

    print(f"Best validation CER: {best_cer:.4f}")
    print(f"Saved checkpoint: {best_path}")
    return best_path


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--img-height", type=int, default=64)
    parser.add_argument("--img-width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.12)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
