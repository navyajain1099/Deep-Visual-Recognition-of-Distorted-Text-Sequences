from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    img_height: int = 64
    img_width: int = 256
    batch_size: int = 64
    epochs: int = 35
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_split: float = 0.12
    seed: int = 42
    num_workers: int = 2
    patience: int = 8
    alphabet: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


BLANK_TOKEN = 0

