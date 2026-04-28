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
                 beta_strat: float, sigma_total: float, param_dim: int,
                 alpha_rd: float = 2.0):
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
        # sigma_total is interpreted as per-dimension std cap; energy budget
        # then scales as (sigma_total)^2 * D so the cap aligns with sigma_0
        # (also per-dim). See REPORT.md "Lựa chọn đã cân nhắc" #1.
        self.param_dim = max(int(param_dim), 1)
        self.alpha_rd = alpha_rd
        # Internal RDP budget tracker (heuristic for adaptive scheduler).
        # NOTE (Bug #2 fix): this is RDP_α, NOT (ε, δ)-DP epsilon. Use
        # compute_eps_dp() to convert before comparing against epsilon_max.
        self.rdp_spent = 0.0
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

    def compute_eps_dp(self) -> float:
        """Convert accumulated RDP_α → (ε, δ)-DP via Mironov 2017 Theorem 8.

        ε(α, δ) = RDP_total(α) + log(1/δ) / (α − 1)

        Returns ∞ if α ≤ 1 (the conversion is undefined). Used by the adaptive
        sigma scheduler to compare consumed privacy against `epsilon_max`.
        """
        if self.alpha_rd <= 1.0:
            return float("inf")
        return self.rdp_spent + math.log(1.0 / self.delta) / (self.alpha_rd - 1.0)

    def compute_sigma_dp(self, round_t: int,
                         attack_signal: float) -> float:
        """Adaptive DP noise scale: f(budget, attack_signal).

        Combines annealing baseline with budget-awareness and threat response.
        Returns sigma_DP >= sigma_floor (DP guarantee).
        """
        base = self.sigma_0 * math.exp(-self.anneal_kappa * round_t)
        # Bug #2 fix: convert RDP → (ε, δ)-DP before comparing to epsilon_max.
        eps_consumed = self.compute_eps_dp()
        eps_remain = max(0.0, self.epsilon_max - eps_consumed)
        budget_factor = max(0.05, eps_remain / max(self.epsilon_max, 1e-12))
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

    def commit_round_rdp(self, sigma_dp_round: float) -> None:
        """Accumulate one round's worth of RDP cost into the internal tracker.

        Called ONCE per round by the simulator (not per honest node), to match
        the outer accountant's per-round cadence under node-level DP. Updating
        per-call would multiply the cost by num_active_honest_nodes, conflate
        node-level DP with group privacy, and crash the adaptive scheduler's
        budget prematurely (see Bug #4 in plans/code-trong-th-m-c-...md).
        """
        if sigma_dp_round > 1e-12:
            self.rdp_spent += self.alpha_rd * self.clip_bound ** 2 / (
                2.0 * sigma_dp_round ** 2)

    def _enforce_budget(self, n_dp: torch.Tensor,
                        n_strat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scale both noises proportionally if total energy exceeds cap.

        Cap = (sigma_total)^2 * D — interprets sigma_total as per-dimension
        std cap (same unit as sigma_0). Without the D factor the cap is an
        absolute L2-norm threshold, which forces sigma_eff_post ≈ sigma_total/√D
        and inflates the privacy cost by ~D× when D is large.
        """
        energy = n_dp.norm() ** 2 + n_strat.norm() ** 2
        cap = (self.sigma_total ** 2) * self.param_dim
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

        # NOTE: rdp_spent is committed ONCE per round by simulator via
        # commit_round_rdp(), not here. Updating per-call would over-count
        # by num_honest_active_nodes (see commit_round_rdp docstring).

        metrics = {
            "trust": trust,
            "sigma_dp": sigma_dp,
            "sigma_strat": sigma_strat,
            "n_dp_norm": float(n_dp.norm().item()),
            "n_strategic_norm": float(n_strategic.norm().item()),
            "total_noise_norm": float(total_noise.norm().item()),
            "nsr": nsr,
            "rdp_spent": self.rdp_spent,
            "eps_dp_internal": self.compute_eps_dp(),
        }
        return total_noise, metrics
