"""Baseline convolutional network for CADICA severity classification."""

from __future__ import annotations

import torch
import torch.nn as nn

from cadica.config import NUM_CLASSES


class CADICACNN(nn.Module):
    """Small VGG-style CNN with a global average pooling head.

    Parameters
    ----------
    num_classes:
        Number of output classes. Defaults to the three CADICA severity
        categories defined in :mod:`cadica.config`.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a ``(N, 3, H, W)`` image batch to ``(N, num_classes)`` logits."""
        return self.classifier(self.features(x))
