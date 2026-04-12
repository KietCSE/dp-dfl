"""ALIE attack: A Little Is Enough. Baruch et al., NeurIPS 2019.

Crafts subtle poisoning within statistical bounds of neighbor gradients:
    g_attack = mean(neighbors) - z_max * std(neighbors)
"""

import torch
from typing import Dict, Optional

from dpfl.core.base_attack import BaseAttack
from dpfl.registry import register, ATTACKS


@register(ATTACKS, "alie")
class ALIEAttack(BaseAttack):
    """A Little Is Enough: subtle poisoning within statistical bounds."""

    def __init__(self, z_max: float = 1.0, **kwargs):
        super().__init__()
        self.z_max = z_max

    def perturb(self, honest_update: torch.Tensor,
                context: Optional[Dict] = None) -> torch.Tensor:
        if context is None or "neighbor_updates" not in context:
            # Fallback: no context available, slight negation
            return honest_update * -0.5

        neighbor_updates = context["neighbor_updates"]
        if not neighbor_updates:
            return honest_update * -0.5

        stacked = torch.stack(list(neighbor_updates.values()))  # (N, D)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0).clamp(min=1e-8)

        return mean - self.z_max * std
