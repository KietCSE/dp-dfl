"""FL node: wraps model, data, and role (honest/attacker)."""

import torch
from torch.utils.data import Subset
from typing import Optional, Set, Tuple

from dpfl.models.base_model import BaseModel
from dpfl.core.dpsgd_trainer import DPSGDTrainer
from dpfl.core.base_noise_mechanism import BaseNoiseMechanism
from dpfl.core.base_attack import BaseAttack


class Node:
    """One FL participant."""

    def __init__(self, node_id: int, model: BaseModel, data: Subset, is_attacker: bool):
        self.id = node_id
        self.model = model
        self.data = data
        self.is_attacker = is_attacker
        self.neighbors: Set[int] = set()
        self.n_samples = len(data)

    def compute_update(
        self,
        trainer: DPSGDTrainer,
        noise_mechanism: BaseNoiseMechanism,
        attack: Optional[BaseAttack] = None,
        apply_noise: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute local update.
        Honest: train with apply_noise (default True for DP-SGD).
        Attacker: train without noise then attack.perturb().
        """
        if self.is_attacker and attack is not None:
            update, n_steps = trainer.train(
                self.model, self.data, noise_mechanism, apply_noise=False
            )
            return attack.perturb(update), n_steps
        return trainer.train(
            self.model, self.data, noise_mechanism, apply_noise=apply_noise
        )
