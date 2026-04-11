"""ABC for model poisoning attacks."""

import torch
from abc import ABC, abstractmethod


class BaseAttack(ABC):
    @abstractmethod
    def perturb(self, honest_update: torch.Tensor) -> torch.Tensor:
        """Transform honest update into malicious update."""
        ...
