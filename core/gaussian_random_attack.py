"""Gaussian random attack: replace gradient with random noise matching honest norm."""

import torch
from typing import Dict, Optional

from dpfl.core.base_attack import BaseAttack
from dpfl.registry import register, ATTACKS


@register(ATTACKS, "gaussian_random")
class GaussianRandomAttack(BaseAttack):
    """Replace gradient with random Gaussian noise matching honest norm."""

    def perturb(self, honest_update: torch.Tensor,
                context: Optional[Dict] = None) -> torch.Tensor:
        noise = torch.randn_like(honest_update)
        target_norm = honest_update.norm()
        noise_norm = noise.norm()
        if noise_norm > 0:
            noise = noise * (target_norm / noise_norm)
        return noise
