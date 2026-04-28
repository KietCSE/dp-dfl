"""Layer-wise bounded Gaussian mechanism for Trust-Aware D2B-DP.

Implements Step 3 of docs/d2b.md:
    ρ^(t)      = min((1 + βt)·ρ_min, ρ_max)
    σ²_l,t     = 2 · Clip_l² / ρ^(t)
    ̃W_{i,l}   = ΔW'_l + B(x_l),  x_l ~ N(0, σ²_l,t)

The bound function B is deterministic and based on single-step RDP epsilon:
    b_l = (e^ε - η) / (e^ε + η)
    B(x_l) = max(-b_l, min(x_l, b_l))
"""

import math
from typing import Optional

import torch


class LayerwiseBoundedGaussian:
    """Per-layer Gaussian noise injector with DP-budget-aware clamp bounds."""

    def __init__(self, bound_eta: float = 0.5):
        self.bound_eta = bound_eta
        self._gen: Optional[torch.Generator] = None

    def set_generator(self, gen: torch.Generator):
        """Wire isolated RNG so noise sampling is reproducible across runs."""
        self._gen = gen

    @staticmethod
    def compute_noise_variance(clip_l: float, rho: float) -> float:
        """σ²_l = 2 · Clip_l² / ρ^(t). Guards near-zero ρ to avoid Inf."""
        return 2.0 * float(clip_l) ** 2 / max(float(rho), 1e-12)

    def add_bounded_noise(self, layer: torch.Tensor,
                          sigma_sq: float, epsilon_l: float) -> torch.Tensor:
        """Add N(0, σ²) noise to a layer, clamp to ±b_l per element based on ε."""
        sigma = math.sqrt(max(sigma_sq, 1e-12))
        if self._gen is not None:
            raw = torch.randn(layer.shape, generator=self._gen,
                              device=layer.device, dtype=layer.dtype)
        else:
            raw = torch.randn_like(layer)
        noise = raw * sigma
        
        # Calculate dynamic absolute bound
        e_eps = math.exp(epsilon_l)
        b_l = (e_eps - self.bound_eta) / (e_eps + self.bound_eta)
        
        return layer + torch.clamp(noise, -b_l, b_l)

