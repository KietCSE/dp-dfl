"""Renyi DP accountant for DP-SGD with subsampled Gaussian mechanism.

Backed by Opacus (Meta AI) implementation of the Sampled Gaussian Mechanism
(Mironov 2019) — tight RDP bounds for subsampled Gaussian, used by all major
PyTorch DP libraries. Replaces the previous loose approximation
`RDP(α) ≈ q²·α/(2z²)` which is not a valid upper bound at α ≥ 3 and can
under-report privacy cost.
"""

from typing import List

from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent

from dpfl.core.base_accountant import BaseAccountant
from dpfl.registry import register, ACCOUNTANTS


@register(ACCOUNTANTS, "renyi_dpsgd")
class RenyiAccountant(BaseAccountant):
    """Cumulative RDP tracking for DP-SGD via Opacus SGM bounds."""

    def __init__(self, alpha_list: List[float], delta: float):
        self.alpha_list = [float(a) for a in alpha_list]
        self.delta = delta
        # Per-alpha accumulated RDP cost, aligned with self.alpha_list.
        self.eps_rdp = {a: 0.0 for a in self.alpha_list}
        self.total_steps = 0

    def step(self, n_steps: int, sampling_rate: float, noise_mult: float):
        """Accumulate n_steps of DP-SGD RDP cost using Opacus SGM.

        noise_mult here is the unit-sensitivity noise multiplier z = σ/C,
        matching Opacus's `noise_multiplier` convention directly.
        """
        if n_steps <= 0:
            return
        if sampling_rate <= 0.0:
            self.total_steps += n_steps
            return
        # Guard near-zero noise to avoid NaN/Inf propagation.
        nm = max(float(noise_mult), 0.01)
        rdp_vec = compute_rdp(
            q=float(sampling_rate),
            noise_multiplier=nm,
            steps=int(n_steps),
            orders=self.alpha_list,
        )
        for a, cost in zip(self.alpha_list, rdp_vec):
            self.eps_rdp[a] += float(cost)
        self.total_steps += n_steps

    def _rdp_vector(self) -> List[float]:
        """Current accumulated RDP vector aligned with self.alpha_list."""
        return [self.eps_rdp[a] for a in self.alpha_list]

    def get_epsilon(self) -> float:
        """Convert accumulated RDP -> (eps, delta)-DP via Opacus optimizer."""
        eps, _ = get_privacy_spent(
            orders=self.alpha_list,
            rdp=self._rdp_vector(),
            delta=self.delta,
        )
        return float(eps)

    def get_best_alpha(self) -> float:
        """Return alpha giving tightest epsilon bound (per Opacus)."""
        _, best_alpha = get_privacy_spent(
            orders=self.alpha_list,
            rdp=self._rdp_vector(),
            delta=self.delta,
        )
        return float(best_alpha)
