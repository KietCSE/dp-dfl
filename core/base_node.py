"""FL node: wraps model, data, and role (honest/attacker)."""

import logging

import numpy as np
import torch
from torch.utils.data import Subset
from typing import Optional, Set, Tuple

from dpfl.models.base_model import BaseModel
from dpfl.core.dpsgd_trainer import DPSGDTrainer
from dpfl.core.base_noise_mechanism import BaseNoiseMechanism
from dpfl.core.base_attack import BaseAttack

logger = logging.getLogger(__name__)


class Node:
    """One FL participant."""

    def __init__(self, node_id: int, model: BaseModel, data: Subset, is_attacker: bool):
        self.id = node_id
        self.model = model
        self.data = data
        self.is_attacker = is_attacker
        self.neighbors: Set[int] = set()
        self.n_samples = len(data)
        self.root_data: Optional[Subset] = None

    def compute_update(
        self,
        trainer: DPSGDTrainer,
        noise_mechanism: BaseNoiseMechanism,
        attack: Optional[BaseAttack] = None,
        apply_noise: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        """Compute local update.

        Honest: train with apply_noise (default True for DP-SGD).
        Data poisoning (LabelFlip): train normally with noise (data already poisoned).
        Model poisoning: train without noise then attack.perturb().
        """
        if self.is_attacker and attack is not None:
            if hasattr(attack, 'wrap_dataset'):
                logger.debug("Node %d: data poisoning (%s)", self.id, type(attack).__name__)
                return trainer.train(
                    self.model, self.data, noise_mechanism, apply_noise=apply_noise
                )
            logger.debug("Node %d: model poisoning (%s)", self.id, type(attack).__name__)
            update, n_steps = trainer.train(
                self.model, self.data, noise_mechanism, skip_dp=True
            )
            return attack.perturb(update, context=None), n_steps
        return trainer.train(
            self.model, self.data, noise_mechanism, apply_noise=apply_noise
        )

    def apply_data_attack(self, attack: BaseAttack):
        """Apply data poisoning attack (e.g., LabelFlip) to local data."""
        if hasattr(attack, 'wrap_dataset') and self.is_attacker:
            self.data = attack.wrap_dataset(self.data)

    def split_root_data(self, ratio: float = 0.1):
        """Split a portion of local data as root/validation set for FLTrust."""
        n = len(self.data)
        root_size = max(1, int(n * ratio))
        indices = list(range(n))
        np.random.shuffle(indices)
        root_indices = indices[:root_size]
        train_indices = indices[root_size:]

        original_indices = self.data.indices
        self.root_data = Subset(
            self.data.dataset,
            [original_indices[i] for i in root_indices]
        )
        self.data = Subset(
            self.data.dataset,
            [original_indices[i] for i in train_indices]
        )
        self.n_samples = len(self.data)
