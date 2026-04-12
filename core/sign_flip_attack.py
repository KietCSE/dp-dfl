"""Sign-flip attack: negate honest gradient."""

import torch
from typing import Dict, Optional

from dpfl.core.base_attack import BaseAttack
from dpfl.registry import register, ATTACKS


@register(ATTACKS, "sign_flip")
class SignFlipAttack(BaseAttack):
    """Negate honest gradient: g -> -g. Simple but effective."""

    def perturb(self, honest_update: torch.Tensor,
                context: Optional[Dict] = None) -> torch.Tensor:
        return -honest_update
