"""Renyi DP accountant for DP-SGD with subsampled Gaussian mechanism."""

import math
from typing import List

from dpfl.core.base_accountant import BaseAccountant
from dpfl.registry import register, ACCOUNTANTS


@register(ACCOUNTANTS, "renyi_dpsgd")
class RenyiAccountant(BaseAccountant):
    """
    Tracks cumulative RDP cost across DP-SGD steps.
    eps_step(alpha) = q^2 * alpha / (2 * z^2)  per step.
    """

    def __init__(self, alpha_list: List[float], delta: float):
        self.alpha_list = alpha_list
        self.delta = delta
        self.eps_rdp = {a: 0.0 for a in alpha_list}
        self.total_steps = 0

    def step(self, n_steps: int, sampling_rate: float, noise_mult: float):
        """Accumulate n_steps of DP-SGD RDP cost.
        Uses first-order approximation valid for small q (~0.01)."""
        for a in self.alpha_list:
            cost = n_steps * sampling_rate ** 2 * a / (2.0 * noise_mult ** 2)
            self.eps_rdp[a] += cost
        self.total_steps += n_steps

    def get_epsilon(self) -> float:
        """Convert RDP -> (eps, delta)-DP: min over alpha."""
        return min(
            self.eps_rdp[a] + math.log(1.0 / self.delta) / (a - 1.0)
            for a in self.alpha_list
        )

    def get_best_alpha(self) -> float:
        """Return alpha that gives tightest epsilon bound."""
        return min(
            self.alpha_list,
            key=lambda a: self.eps_rdp[a] + math.log(1.0 / self.delta) / (a - 1.0)
        )
