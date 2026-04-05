"""Abstract base model with flat param get/set for DFL exchange."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_flat_params(self) -> torch.Tensor:
        """Flatten all params into 1D vector."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat: torch.Tensor):
        """Set params from 1D vector."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset:offset + numel].view(p.shape))
            offset += numel
