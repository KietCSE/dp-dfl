"""ABC for model poisoning attacks."""

import torch
from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseAttack(ABC):
    @abstractmethod
    def perturb(self, honest_update: torch.Tensor,
                context: Optional[Dict] = None) -> torch.Tensor:
        """Transform honest update into malicious update.

        Args:
            honest_update: The attacker's own honest gradient.
            context: Optional dict with neighbor info for adaptive attacks.
                     Keys: 'neighbor_updates' -> Dict[int, Tensor]
        """
        ...
