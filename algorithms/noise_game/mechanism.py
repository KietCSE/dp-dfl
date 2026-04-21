"""Strategic noise mechanism: 4-layer architecture.

Layer 1 — DP Gaussian noise (privacy guarantee)
Layer 2 — RDP-based adaptive sigma scheduling
Layer 3 — Strategic noise: directional + orthogonal + spectrum-aware
Budget constraint: ||n_DP||^2 + ||n_strategic||^2 <= sigma_total^2
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


class NoiseGameMechanism:
    """4-layer strategic noise for the Noise Game algorithm.

    Layer 1: DP Gaussian noise (n_DP) for privacy
    Layer 2: Adaptive sigma control via RDP budget / trust / attack signal
    Layer 3: Structured strategic noise (directional + orthogonal + spectrum)
    """

    def __init__(self, alpha_attack: float, sigma_0: float,
                 anneal_kappa: float, svd_rank: int, svd_reshape_k: int,
                 clip_bound: float, delta: float, epsilon_max: float,
                 beta_strat: float, sigma_total: float, alpha_rd: float = 2.0):
        self.alpha_attack = alpha_attack
        self.sigma_0 = sigma_0
        self.anneal_kappa = anneal_kappa
        self.svd_rank = svd_rank
        self.svd_reshape_k = svd_reshape_k
        # Layer 2 params
        self.clip_bound = clip_bound
        self.delta = delta
        self.epsilon_max = epsilon_max
        self.beta_strat = beta_strat
        self.sigma_total = sigma_total
        self.alpha_rd = alpha_rd
        # Internal RDP budget tracker (heuristic for adaptive scheduler)
        self.epsilon_spent = 0.0
        # DP sigma floor: minimum sigma for (eps_max, delta)-DP
        self._sigma_floor = clip_bound * math.sqrt(
            2.0 * math.log(1.25 / delta)) / epsilon_max
        self._gen: "torch.Generator | None" = None

    def set_generator(self, gen: "torch.Generator"):
        """Set isolated RNG for all noise sampling."""
        self._gen = gen

    def _randn_like(self, x: torch.Tensor) -> torch.Tensor:
        """Draw N(0,1) matching shape/device/dtype of x, using isolated gen."""
        if self._gen is not None:
            return torch.randn(x.shape, generator=self._gen,
                               device=x.device, dtype=x.dtype)
        return torch.randn_like(x)

    # -- Layer 2: Adaptive sigma scheduling --

    def compute_sigma_dp(self, round_t: int,
                         attack_signal: float) -> float:
        """Adaptive DP noise scale: f(budget, attack_signal).

        Combines annealing baseline with budget-awareness and threat response.
        Returns sigma_DP >= sigma_floor (DP guarantee).
        """
        base = self.sigma_0 * math.exp(-self.anneal_kappa * round_t)
        eps_remain = max(0.0, self.epsilon_max - self.epsilon_spent)
        budget_factor = max(0.05, eps_remain / self.epsilon_max)
        threat_factor = 1.0 + attack_signal
        return max(base * budget_factor * threat_factor, self._sigma_floor)

    # -- Layer 1: DP Gaussian noise --

    def compute_dp_noise(self, gradient: torch.Tensor,
                         sigma_dp: float) -> torch.Tensor:
        """Standard Gaussian DP noise: n_DP ~ N(0, sigma_dp^2 I)."""
        return self._randn_like(gradient) * sigma_dp

    # -- Layer 3: Strategic noise components --

    def compute_trust(self, g_curr: torch.Tensor,
                      g_prev: torch.Tensor) -> float:
        """Trust score: cosine similarity between current and previous gradient."""
        if g_prev.norm() < 1e-12 or g_curr.norm() < 1e-12:
            return 1.0
        cos = F.cosine_similarity(g_curr.unsqueeze(0), g_prev.unsqueeze(0))
        return cos.item()

    def compute_attack_direction(self, g_curr: torch.Tensor,
                                  g_prev: torch.Tensor) -> torch.Tensor:
        """Attack direction estimation: v_t = g_curr - g_prev."""
        return g_curr - g_prev

    def directional_noise(self, attack_dir: torch.Tensor,
                          trust: float) -> torch.Tensor:
        """Directional noise: alpha * (1-trust) * attack_dir."""
        alpha_i = self.alpha_attack * (1.0 - trust)
        return alpha_i * attack_dir

    def orthogonal_noise(self, gradient: torch.Tensor,
                         sigma: float) -> torch.Tensor:
        """Orthogonal noise: z projected perpendicular to gradient."""
        z = self._randn_like(gradient) * sigma
        g_norm_sq = gradient.dot(gradient)
        if g_norm_sq < 1e-12:
            return z
        proj = (z.dot(gradient) / g_norm_sq) * gradient
        return z - proj

    def spectrum_noise(self, gradient: torch.Tensor,
                       sigma: float) -> torch.Tensor:
        """Spectrum-aware noise via truncated SVD."""
        D = gradient.numel()
        k = min(self.svd_reshape_k, D)
        cols = math.ceil(D / k)
        padded_len = k * cols
        if padded_len > D:
            padded = torch.zeros(padded_len, device=gradient.device)
            padded[:D] = gradient
        else:
            padded = gradient
        matrix = padded.reshape(k, cols)
        rank = min(self.svd_rank, k, cols)
        try:
            U, S, V = torch.svd_lowrank(matrix, q=rank)
        except RuntimeError:
            return self._randn_like(gradient) * sigma
        eps = 1e-8
        inv_weights = 1.0 / (S + eps)
        if self._gen is not None:
            r = torch.randn((rank,), generator=self._gen,
                            device=gradient.device, dtype=gradient.dtype) * sigma
        else:
            r = torch.randn(rank, device=gradient.device) * sigma
        n_spec_flat = (U @ torch.diag(inv_weights * r) @ V.T).flatten()[:D]
        return n_spec_flat

    # -- Budget constraint --

    def _enforce_budget(self, n_dp: torch.Tensor,
                        n_strat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scale both noises proportionally if total energy exceeds cap."""
        energy = n_dp.norm() ** 2 + n_strat.norm() ** 2
        cap = self.sigma_total ** 2
        if energy > cap and energy > 1e-12:
            factor = math.sqrt(cap / energy.item())
            return n_dp * factor, n_strat * factor
        return n_dp, n_strat

    # -- Full pipeline --

    def compute_total_noise(
        self, gradient: torch.Tensor, prev_gradient: torch.Tensor,
        round_t: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Full 4-layer noise pipeline.

        Returns (total_noise, metrics_dict).
        """
        # Trust + attack signal
        trust = self.compute_trust(gradient, prev_gradient)
        attack_signal = 1.0 - trust

        # Layer 2: Adaptive sigma
        sigma_dp = self.compute_sigma_dp(round_t, attack_signal)
        sigma_strat = self.beta_strat * sigma_dp

        # Layer 1: DP Gaussian noise
        n_dp = self.compute_dp_noise(gradient, sigma_dp)

        # Layer 3: Strategic noise (direction only, then normalize + scale)
        attack_dir = self.compute_attack_direction(gradient, prev_gradient)
        n_attack = self.directional_noise(attack_dir, trust)
        n_orth = self.orthogonal_noise(gradient, 1.0)
        n_spec = self.spectrum_noise(gradient, 1.0)

        combined = n_attack + n_orth + n_spec
        combined_norm = combined.norm()
        if combined_norm > 1e-12:
            hat_n = combined / combined_norm
        else:
            hat_n = combined
        n_strategic = sigma_strat * hat_n

        # Budget constraint
        n_dp, n_strategic = self._enforce_budget(n_dp, n_strategic)
        total_noise = n_dp + n_strategic

        # NSR monitoring
        g_norm = gradient.norm().item()
        nsr = total_noise.norm().item() / (g_norm + 1e-12)

        # Track approximate RDP cost using fixed Renyi order and clipping bound.
        # After clipping, sensitivity is C = clip_bound, so
        # epsilon_t(alpha) = alpha * C^2 / (2 * sigma_dp^2).
        if sigma_dp > 1e-12:
            self.epsilon_spent += self.alpha_rd * self.clip_bound ** 2 / (
                2.0 * sigma_dp ** 2)

        metrics = {
            "trust": trust,
            "sigma_dp": sigma_dp,
            "sigma_strat": sigma_strat,
            "n_dp_norm": float(n_dp.norm().item()),
            "n_strategic_norm": float(n_strategic.norm().item()),
            "total_noise_norm": float(total_noise.norm().item()),
            "nsr": nsr,
            "epsilon_spent": self.epsilon_spent,
        }
        return total_noise, metrics
