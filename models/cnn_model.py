"""Simple 2-conv + 2-fc CNN for CIFAR-10."""

import torch
import torch.nn as nn

from dpfl.models.base_model import BaseModel
from dpfl.registry import register, MODELS


@register(MODELS, "cnn")
class CNN(BaseModel):
    """Conv(3->32) -> ReLU -> MaxPool(2) -> Conv(32->64) -> ReLU -> MaxPool(2)
    -> FC(4096->hidden) -> ReLU -> FC(hidden->classes)"""

    def __init__(self, input_channels: int = 3, hidden_size: int = 128,
                 num_classes: int = 10, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # After 2x MaxPool(2) on 32x32 input: 64 x 8 x 8 = 4096
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
