"""Bounded Gaussian mechanism: per-neighbor noise with clamping for D2B-DP."""

import math
import torch


class BoundedGaussianMechanism:
    """Computes per-neighbor noise variance, bounds, and injects bounded noise."""

    def __init__(self, eta: float = 0.1):
        self.eta = eta

    def compute_noise_variance(self, clip_bound: float, dataset_size: int,
                               budget: float) -> float:
        """sigma^2_ij = 2 * C^2 / (|D_i|^2 * rho_ij). Guard: budget <= 0 -> large variance."""
        if budget <= 1e-12:
            return 1e6
        return 2.0 * clip_bound ** 2 / (budget)

    def compute_bound(self, epsilon: float) -> float:
        """b = (e^eps - eta) / (e^eps + eta). Clamp to [0, inf)."""
        exp_eps = math.exp(min(epsilon, 50.0))  # prevent overflow
        b = (exp_eps - self.eta) / (exp_eps + self.eta + 1e-12)
        return max(b, 0.0)

    def add_bounded_noise(self, gradient: torch.Tensor, sigma_sq: float,
                          epsilon: float) -> torch.Tensor:
        """Add bounded Gaussian noise: N(0, sigma^2) clamped to [-b, b]."""
        noise = torch.randn_like(gradient) * math.sqrt(max(sigma_sq, 1e-12))
        b = self.compute_bound(epsilon)
        bounded = torch.clamp(noise, -b, b)
        return gradient + bounded
