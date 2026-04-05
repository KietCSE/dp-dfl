"""Gaussian mechanism: L2-norm clipping + Gaussian noise for DP-SGD."""

import torch

from dpfl.privacy.base_noise_mechanism import BaseNoiseMechanism
from dpfl.registry import register, NOISE_MECHANISMS


@register(NOISE_MECHANISMS, "gaussian")
class GaussianMechanism(BaseNoiseMechanism):

    def clip(self, per_sample_grads: torch.Tensor, clip_bound: float) -> torch.Tensor:
        """L2-norm clip: g' = g * min(1, C / ||g||_2)"""
        norms = per_sample_grads.norm(2, dim=1, keepdim=True)
        clip_factors = torch.clamp(clip_bound / (norms + 1e-12), max=1.0)
        return per_sample_grads * clip_factors

    def add_noise(self, avg_grad: torch.Tensor, clip_bound: float,
                  noise_mult: float, batch_size: int) -> torch.Tensor:
        """Gaussian noise: sigma = z * C / B per dimension."""
        noise_std = noise_mult * clip_bound / batch_size
        return avg_grad + torch.randn_like(avg_grad) * noise_std
