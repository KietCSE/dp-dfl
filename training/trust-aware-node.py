"""Trust-aware FL node: extends Node with per-neighbor trust state and buffers."""

import torch
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from dpfl.training.node import Node
from dpfl.training.dpsgd_trainer import DPSGDTrainer
from dpfl.privacy.base_noise_mechanism import BaseNoiseMechanism
from dpfl.attacks.base_attack import BaseAttack


class TrustAwareNode(Node):
    """Node with trust scores, clip history, and temporal buffers for D2B-DP."""

    def __init__(self, node_id, model, data, is_attacker, trust_config):
        super().__init__(node_id, model, data, is_attacker)
        self.trust_scores: Dict[int, float] = {}
        self.clip_history: deque = deque(maxlen=trust_config.clip_window)
        self.z_score_buffer: deque = deque(maxlen=trust_config.temporal_window)
        self.trust_buffer: deque = deque(maxlen=trust_config.temporal_window)
        self.per_neighbor_rdp: Dict[int, List[float]] = {}

    def init_trust(self, neighbors: Set[int], trust_init: float,
                   alpha_list: List[float]):
        """Initialize per-neighbor trust scores and RDP arrays."""
        for j in neighbors:
            self.trust_scores[j] = trust_init
            self.per_neighbor_rdp[j] = [0.0] * len(alpha_list)

    def compute_update(
        self,
        trainer: DPSGDTrainer,
        noise_mechanism: BaseNoiseMechanism,
        attack: Optional[BaseAttack] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute local update WITHOUT noise — noise added per-edge in Phase 2.
        Attacker: train without noise, then apply attack.
        Honest: train without noise (per-neighbor noise comes later).
        """
        if self.is_attacker and attack is not None:
            update, n_steps = trainer.train(
                self.model, self.data, noise_mechanism, apply_noise=False,
            )
            return attack.perturb(update), n_steps
        # Honest: also no noise here
        return trainer.train(
            self.model, self.data, noise_mechanism, apply_noise=False,
        )
