"""Layer-wise bounded Gaussian mechanism for Trust-Aware D2B-DP.

Implements Step 3 of docs/Trust-Aware-D2B-DP.md:
    ρ^(t)      = min((1 + βt)·ρ_min, ρ_max)
    σ²_l,t     = 2 · Clip_l² / ρ^(t)
    ̃W_{i,l}   = ΔW'_l + B(x_l),  x_l ~ N(0, σ²_l,t)

The doc leaves B (the noise-bound function) abstract; we use a symmetric
±k·σ clamp (3-sigma rule by default) — keeps tails bounded without coupling
to a per-step ε derivation.
"""

import math
from typing import Optional

import torch


class LayerwiseBoundedGaussian:
    """Per-layer Gaussian noise injector with k·σ-clamp for outlier control."""

    def __init__(self, bound_k: float = 3.0):
        self.bound_k = bound_k
        self._gen: Optional[torch.Generator] = None

    def set_generator(self, gen: torch.Generator):
        """Wire isolated RNG so noise sampling is reproducible across runs."""
        self._gen = gen

    @staticmethod
    def compute_noise_variance(clip_l: float, rho: float) -> float:
        """σ²_l = 2 · Clip_l² / ρ^(t). Guards near-zero ρ to avoid Inf."""
        return 2.0 * float(clip_l) ** 2 / max(float(rho), 1e-12)

    def add_bounded_noise(self, layer: torch.Tensor,
                          sigma_sq: float) -> torch.Tensor:
        """Add N(0, σ²) noise to a layer, clamp to ±bound_k·σ per element."""
        sigma = math.sqrt(max(sigma_sq, 1e-12))
        if self._gen is not None:
            raw = torch.randn(layer.shape, generator=self._gen,
                              device=layer.device, dtype=layer.dtype)
        else:
            raw = torch.randn_like(layer)
        noise = raw * sigma
        bound = self.bound_k * sigma
        return layer + torch.clamp(noise, -bound, bound)
