"""Strategic noise mechanism: directional + orthogonal + spectrum-aware noise."""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


class NoiseGameMechanism:
    """Computes structured noise for the Strategic Noise Game algorithm.

    Three noise components:
      1. Directional noise — cancels estimated attack direction
      2. Orthogonal noise — DP randomness preserving gradient direction
      3. Spectrum-aware noise — SVD-based, less noise on important directions
    """

    def __init__(self, alpha_attack: float, sigma_0: float,
                 anneal_kappa: float, svd_rank: int, svd_reshape_k: int):
        self.alpha_attack = alpha_attack
        self.sigma_0 = sigma_0
        self.anneal_kappa = anneal_kappa
        self.svd_rank = svd_rank
        self.svd_reshape_k = svd_reshape_k

    def compute_annealed_sigma(self, round_t: int) -> float:
        """Noise annealing: sigma_t = sigma_0 * exp(-kappa * t)."""
        return self.sigma_0 * math.exp(-self.anneal_kappa * round_t)

    def compute_trust(self, g_curr: torch.Tensor, g_prev: torch.Tensor) -> float:
        """Trust score: cosine similarity between current and previous gradient.

        Returns value in [-1, 1]:
          ~1  = stable training
          ~0  = drift
          <0  = adversarial flip
        """
        if g_prev.norm() < 1e-12 or g_curr.norm() < 1e-12:
            return 1.0  # No info -> assume stable
        cos = F.cosine_similarity(g_curr.unsqueeze(0), g_prev.unsqueeze(0))
        return cos.item()

    def compute_attack_direction(self, g_curr: torch.Tensor,
                                  g_prev: torch.Tensor) -> torch.Tensor:
        """Attack direction estimation: v_i(t) = g_i(t) - g_i(t-1)."""
        return g_curr - g_prev

    def directional_noise(self, attack_dir: torch.Tensor,
                          trust: float) -> torch.Tensor:
        """Directional noise: n_attack = alpha_i * v_i(t).

        alpha_i = alpha * (1 - trust): high noise when trust is low.
        """
        alpha_i = self.alpha_attack * (1.0 - trust)
        return alpha_i * attack_dir

    def orthogonal_noise(self, gradient: torch.Tensor,
                         sigma: float) -> torch.Tensor:
        """Orthogonal noise: z projected perpendicular to gradient.

        n_orth = z - (z . g / ||g||^2) * g
        Preserves optimization direction while providing DP randomness.
        """
        z = torch.randn_like(gradient) * sigma
        g_norm_sq = gradient.dot(gradient)
        if g_norm_sq < 1e-12:
            return z  # Zero gradient -> return raw noise
        proj = (z.dot(gradient) / g_norm_sq) * gradient
        return z - proj

    def spectrum_noise(self, gradient: torch.Tensor,
                       sigma: float) -> torch.Tensor:
        """Spectrum-aware noise via truncated SVD.

        Reshape gradient to matrix, compute lowrank SVD, inject noise
        inversely proportional to singular values: less noise on
        important directions, more on less critical ones.
        """
        D = gradient.numel()
        k = min(self.svd_reshape_k, D)

        # Pad gradient so D is divisible by k
        cols = math.ceil(D / k)
        padded_len = k * cols
        if padded_len > D:
            padded = torch.zeros(padded_len, device=gradient.device)
            padded[:D] = gradient
        else:
            padded = gradient

        matrix = padded.reshape(k, cols)

        # Truncated SVD via randomized algorithm
        rank = min(self.svd_rank, k, cols)
        try:
            U, S, V = torch.svd_lowrank(matrix, q=rank)
        except RuntimeError:
            # Fallback to raw Gaussian noise if SVD fails
            return torch.randn_like(gradient) * sigma

        # Inverse-weighted noise: n_spec = U * diag(1/(lambda+eps)) * r
        eps = 1e-8
        inv_weights = 1.0 / (S + eps)
        r = torch.randn(rank, device=gradient.device) * sigma
        # (k, rank) @ diag(rank) @ (rank,) -> (k,)
        n_spec_rows = U @ (inv_weights * r)
        # Expand back to full matrix shape via V
        n_spec_matrix = n_spec_rows.unsqueeze(1) * V[:rank, :].sum(dim=0, keepdim=True)

        # Simplified: construct noise in original shape
        # n_spec = U @ diag(inv_weights * r) @ V^T -> flatten
        n_spec_flat = (U @ torch.diag(inv_weights * r) @ V.T).flatten()[:D]
        return n_spec_flat

    def compute_total_noise(
        self, gradient: torch.Tensor, prev_gradient: torch.Tensor,
        round_t: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Full noise pipeline: trust -> 3 components -> sum.

        Returns (noise_vector, metrics_dict).
        """
        sigma_t = self.compute_annealed_sigma(round_t)
        trust = self.compute_trust(gradient, prev_gradient)

        # Component 1: Directional noise (cancel attack direction)
        attack_dir = self.compute_attack_direction(gradient, prev_gradient)
        n_attack = self.directional_noise(attack_dir, trust)

        # Component 2: Orthogonal noise (DP randomness)
        n_orth = self.orthogonal_noise(gradient, sigma_t)

        # Component 3: Spectrum-aware noise (SVD-based)
        n_spec = self.spectrum_noise(gradient, sigma_t)

        # Total strategic noise
        total_noise = n_attack + n_orth + n_spec

        metrics = {
            "trust": trust,
            "sigma_t": sigma_t,
            "n_attack_norm": float(n_attack.norm().item()),
            "n_orth_norm": float(n_orth.norm().item()),
            "n_spec_norm": float(n_spec.norm().item()),
            "total_noise_norm": float(total_noise.norm().item()),
        }
        return total_noise, metrics
