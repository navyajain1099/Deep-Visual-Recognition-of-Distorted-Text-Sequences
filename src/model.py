from __future__ import annotations

import torch
from torch import nn


class CRNN(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256, dropout: float = 0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            self._block(1, 64),
            nn.MaxPool2d(2, 2),
            self._block(64, 128),
            nn.MaxPool2d(2, 2),
            self._block(128, 256),
            self._block(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            self._block(256, 512),
            nn.Dropout2d(dropout),
            self._block(512, 512),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        self.encoder = nn.LSTM(
            input_size=512 * 4,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        batch, channels, height, width = features.shape
        features = features.permute(0, 3, 1, 2).contiguous().view(batch, width, channels * height)
        encoded, _ = self.encoder(features)
        logits = self.classifier(encoded)
        return logits.permute(1, 0, 2)

