"""MLP model for MNIST classification."""

import torch
import torch.nn as nn

from dpfl.models.base_model import BaseModel
from dpfl.registry import register, MODELS


@register(MODELS, "mlp")
class MLP(BaseModel):
    def __init__(self, input_dim: int = 784, hidden_size: int = 100, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.fc2(self.relu(self.fc1(x)))
