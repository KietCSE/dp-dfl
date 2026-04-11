"""Scale attack: multiply honest update by a factor."""

import torch

from dpfl.core.base_attack import BaseAttack
from dpfl.registry import register, ATTACKS


@register(ATTACKS, "scale")
class ScaleAttack(BaseAttack):
    def __init__(self, scale_factor: float = 3.0):
        self.scale_factor = scale_factor

    def perturb(self, honest_update: torch.Tensor) -> torch.Tensor:
        return honest_update * self.scale_factor
