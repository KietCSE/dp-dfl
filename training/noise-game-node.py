"""Noise-game node: extends Node with gradient history, momentum, EMA,
SCAFFOLD control variates, and two-track model state."""

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from typing import Optional, Tuple

from dpfl.models.base_model import BaseModel
from dpfl.training.dpsgd_trainer import DPSGDTrainer
from dpfl.privacy.base_noise_mechanism import BaseNoiseMechanism
from dpfl.attacks.base_attack import BaseAttack
from dpfl.training.node import Node
from dpfl.noise_game_config import NoiseGameConfig


class NoiseGameNode(Node):
    """FL node with strategic noise game state.

    Extends base Node with buffers for:
      - Previous gradient (attack direction estimation)
      - Momentum (optimization stabilization)
      - EMA gradient (denoising)
      - SCAFFOLD control variate (variance reduction)
      - Two-track model params (clean + robust)
    """

    def __init__(self, node_id: int, model: BaseModel, data: Subset,
                 is_attacker: bool, ng_config: NoiseGameConfig):
        super().__init__(node_id, model, data, is_attacker)
        self.ng = ng_config

        # State buffers — initialized via init_buffers() after model is set
        self.prev_gradient: Optional[torch.Tensor] = None
        self.momentum: Optional[torch.Tensor] = None
        self.ema_gradient: Optional[torch.Tensor] = None
        self.control_variate: Optional[torch.Tensor] = None
        self.clean_params: Optional[torch.Tensor] = None
        self.robust_params: Optional[torch.Tensor] = None

    def init_buffers(self, param_dim: int):
        """Zero-initialize all state buffers. Call after model is created."""
        device = next(self.model.parameters()).device
        self.prev_gradient = torch.zeros(param_dim, device=device)
        self.momentum = torch.zeros(param_dim, device=device)
        self.ema_gradient = torch.zeros(param_dim, device=device)
        self.control_variate = torch.zeros(param_dim, device=device)
        p = self.model.get_flat_params()
        self.clean_params = p.clone()
        self.robust_params = p.clone()

    def compute_update(
        self,
        trainer: DPSGDTrainer,
        noise_mechanism: BaseNoiseMechanism,
        attack: Optional[BaseAttack] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Train WITHOUT noise injection. Noise-game pipeline runs in simulator.

        Attackers: train without noise, then apply attack.perturb().
        Honest: train without noise (noise-game handles it externally).
        """
        if self.is_attacker and attack is not None:
            update, n_steps = trainer.train(
                self.model, self.data, noise_mechanism, apply_noise=False)
            return attack.perturb(update), n_steps
        return trainer.train(
            self.model, self.data, noise_mechanism, apply_noise=False)

    # -- SCAFFOLD variance reduction --

    def apply_scaffold(self, gradient: torch.Tensor,
                       global_control: torch.Tensor) -> torch.Tensor:
        """SCAFFOLD correction: g' = g - c_i + c."""
        return gradient - self.control_variate + global_control

    def update_control_variate(self, gradient: torch.Tensor):
        """Simplified SCAFFOLD: c_i = gradient (current local gradient)."""
        self.control_variate = gradient.clone()

    # -- Momentum --

    def update_momentum(self, noised_grad: torch.Tensor) -> torch.Tensor:
        """Momentum update: m = beta * m + (1-beta) * g_hat."""
        beta = self.ng.momentum_beta
        self.momentum = beta * self.momentum + (1.0 - beta) * noised_grad
        return self.momentum

    # -- EMA denoising --

    def update_ema(self, noised_grad: torch.Tensor) -> torch.Tensor:
        """EMA gradient denoising: g_tilde = gamma * g_tilde + (1-gamma) * g_hat."""
        gamma = self.ng.ema_gamma
        self.ema_gradient = gamma * self.ema_gradient + (1.0 - gamma) * noised_grad
        return self.ema_gradient

    # -- Gradient alignment filtering --

    def check_alignment(self, noised_grad: torch.Tensor) -> bool:
        """Check if noised gradient aligns with EMA trend.

        Returns True if cos(g_hat, ema_prev) >= tau.
        Always True on first round (no EMA history).
        """
        if self.ema_gradient is None or self.ema_gradient.norm() < 1e-12:
            return True
        cos = F.cosine_similarity(
            noised_grad.unsqueeze(0), self.ema_gradient.unsqueeze(0))
        return cos.item() >= self.ng.align_tau

    # -- Two-track model --

    def update_two_track(self, raw_update: torch.Tensor,
                         robust_update: torch.Tensor):
        """Update clean and robust params separately, then combine.

        clean_params  += raw_update (pre-noise, pre-aggregation)
        robust_params += robust_update (post-aggregation)
        model = lambda * clean + (1-lambda) * robust
        """
        self.clean_params = self.clean_params + raw_update
        self.robust_params = self.robust_params + robust_update
        lam = self.ng.two_track_lambda
        combined = lam * self.clean_params + (1.0 - lam) * self.robust_params
        self.model.set_flat_params(combined)

    # -- State management --

    def store_gradient(self, gradient: torch.Tensor):
        """Store current gradient as previous for next round."""
        self.prev_gradient = gradient.clone()
