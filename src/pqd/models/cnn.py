"""The 1-D CNN. It reads a raw window of voltage samples and names the disturbance.

Four convolution blocks learn their own features (no hand-made feature table).
Global average pooling keeps the overall shape; global max pooling keeps short
bursts such as transients. Both are joined before the final layers.
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import CLASSES


def _block(c_in: int, c_out: int, kernel: int, pool: bool) -> nn.Sequential:
    layers = [
        nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool1d(2))
    return nn.Sequential(*layers)


class PQDNet(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            _block(1, 16, 15, pool=True),
            _block(16, 32, 9, pool=True),
            _block(32, 64, 5, pool=True),
            _block(64, 64, 3, pool=False),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, samples)
        f = self.features(x)
        pooled = torch.cat([f.mean(dim=2), f.amax(dim=2)], dim=1)
        return self.head(pooled)


def build_model(n_classes: int = len(CLASSES)) -> PQDNet:
    return PQDNet(n_classes)
