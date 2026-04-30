"""Gaussian mechanism: L2-norm clipping + Gaussian noise for DP-SGD."""

import torch

from dpfl.core.base_noise_mechanism import BaseNoiseMechanism
from dpfl.registry import register, NOISE_MECHANISMS


@register(NOISE_MECHANISMS, "gaussian")
class GaussianMechanism(BaseNoiseMechanism):

    def __init__(self):
        self._gen: "torch.Generator | None" = None

    def set_generator(self, gen: "torch.Generator"):
        """Set isolated RNG for noise sampling (called by BaseSimulator)."""
        self._gen = gen

    def clip(self, per_sample_grads: torch.Tensor, clip_bound: float) -> torch.Tensor:
        """L2-norm clip: g' = g * min(1, C / ||g||_2)"""
        norms = per_sample_grads.norm(2, dim=1, keepdim=True)
        clip_factors = torch.clamp(clip_bound / (norms + 1e-12), max=1.0)
        return per_sample_grads * clip_factors

    def add_noise(self, avg_grad: torch.Tensor, clip_bound: float,
                  noise_mult: float, batch_size: int) -> torch.Tensor:
        """Gaussian noise: sigma = z * C / B per dimension."""
        noise_std = noise_mult * clip_bound / batch_size
        if self._gen is not None:
            noise = torch.randn(avg_grad.shape, generator=self._gen,
                                device=avg_grad.device, dtype=avg_grad.dtype)
        else:
            noise = torch.randn_like(avg_grad)
        return avg_grad + noise * noise_std

    def clip_and_noise_batched(self, per_sample_grads_NBD: torch.Tensor,
                                clip_bound: float, noise_mult: float,
                                batch_size: int) -> torch.Tensor:
        """Batched clip + per-client mean + per-client Gaussian noise.

        Args:
          per_sample_grads_NBD: (N, B, D) per-sample grads, one slice per client
          clip_bound: L2 clip threshold
          noise_mult: z, noise multiplier
          batch_size: B (denominator for noise sigma — matches single-client formula)

        Returns: (N, D) noised mean grad per client. Each client's noise is
        independently drawn so privacy guarantees compose per the standard
        Gaussian mechanism sensitivity argument.
        """
        # L2-norm clip on the D axis for every (N, B) slot
        norms = per_sample_grads_NBD.norm(2, dim=2, keepdim=True)
        clip_factors = torch.clamp(clip_bound / (norms + 1e-12), max=1.0)
        clipped = per_sample_grads_NBD * clip_factors            # (N, B, D)
        avg = clipped.mean(dim=1)                                # (N, D)

        noise_std = noise_mult * clip_bound / batch_size
        if self._gen is not None:
            noise = torch.randn(avg.shape, generator=self._gen,
                                device=avg.device, dtype=avg.dtype)
        else:
            noise = torch.randn_like(avg)
        return avg + noise * noise_std
