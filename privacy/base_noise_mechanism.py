"""ABC for DP noise injection strategy (clip + noise)."""

import torch
from abc import ABC, abstractmethod


class BaseNoiseMechanism(ABC):

    @abstractmethod
    def clip(self, per_sample_grads: torch.Tensor, clip_bound: float) -> torch.Tensor:
        """Clip per-sample grads (B, D) -> (B, D)."""
        ...

    @abstractmethod
    def add_noise(self, avg_grad: torch.Tensor, clip_bound: float,
                  noise_mult: float, batch_size: int) -> torch.Tensor:
        """Add noise to averaged gradient (D,) -> (D,)."""
        ...

    def clip_and_noise(self, per_sample_grads: torch.Tensor, clip_bound: float,
                       noise_mult: float, batch_size: int) -> torch.Tensor:
        """Template method: clip -> average -> noise."""
        clipped = self.clip(per_sample_grads, clip_bound)
        avg = clipped.mean(dim=0)
        return self.add_noise(avg, clip_bound, noise_mult, batch_size)
