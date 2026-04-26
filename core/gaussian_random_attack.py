"""Gaussian random attack: replace gradient with random noise matching honest norm."""

import torch
from typing import Dict, Optional

from dpfl.core.base_attack import BaseAttack
from dpfl.registry import register, ATTACKS


@register(ATTACKS, "gaussian_random")
class GaussianRandomAttack(BaseAttack):
    """Replace gradient with random Gaussian noise matching honest norm."""

    def __init__(self):
        super().__init__()
        self._gen: "torch.Generator | None" = None

    def set_generator(self, gen: "torch.Generator"):
        """Set isolated RNG for noise sampling."""
        self._gen = gen

    def perturb(self, honest_update: torch.Tensor,
                context: Optional[Dict] = None) -> torch.Tensor:
        if self._gen is not None:
            noise = torch.randn(honest_update.shape, generator=self._gen,
                                device=honest_update.device,
                                dtype=honest_update.dtype)
        else:
            noise = torch.randn_like(honest_update)
        target_norm = honest_update.norm()
        noise_norm = noise.norm()
        if noise_norm > 0:
            noise = noise * (target_norm / noise_norm)
        return noise
