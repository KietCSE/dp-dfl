"""Layer-wise Gaussian noise mechanism for Trust-Aware D2B-DP.

Implements Step 2.2 of docs/Trust-Aware-D2B-DP.md (Outbound Processing):
    ρ^(t)      = min((1 + βt)·ρ_min, ρ_max)
    σ²_l,t     = 2 · Clip_l² / ρ^(t)
    x_l        ~ N(0, σ²_l,t)
    W̃_{i,l}    = ΔW'_l + x_l

Pure Gaussian — no bounded clamping. Per-layer σ² is computed from the
layer's own clipping threshold so the L2-sensitivity contribution of every
layer is normalized identically (Rényi DP at order α: ε^(t)(α) = α·ρ_t/4).
"""

import math
from typing import Optional

import torch


class LayerwiseGaussianNoise:
    """Per-layer Gaussian noise injector keyed off the per-layer clip threshold."""

    def __init__(self):
        self._gen: Optional[torch.Generator] = None

    def set_generator(self, gen: torch.Generator):
        """Wire isolated RNG so noise sampling is reproducible across runs."""
        self._gen = gen

    @staticmethod
    def compute_noise_variance(clip_l: float, rho: float) -> float:
        """σ²_l = 2 · Clip_l² / ρ^(t). Guards near-zero ρ to avoid Inf."""
        return 2.0 * float(clip_l) ** 2 / max(float(rho), 1e-12)

    def add_noise(self, layer: torch.Tensor, sigma_sq: float) -> torch.Tensor:
        """Add N(0, σ²) noise to a layer (Step 2.2; no bounded clamping)."""
        sigma = math.sqrt(max(sigma_sq, 1e-12))
        if self._gen is not None:
            raw = torch.randn(layer.shape, generator=self._gen,
                              device=layer.device, dtype=layer.dtype)
        else:
            raw = torch.randn_like(layer)
        return layer + raw * sigma
