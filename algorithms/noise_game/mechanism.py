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
from scipy.optimize import brentq
from scipy.stats import norm


def analytic_gaussian_sigma(epsilon: float, delta: float,
                            sensitivity: float) -> float:
    """Balle-Wang 2018 analytic Gaussian Mechanism — tight σ for any ε > 0.

    Solves the (ε, δ)-DP boundary condition (Balle-Wang 2018 Theorem 9):

        Φ(Δ/(2σ) - εσ/Δ) - e^ε · Φ(-Δ/(2σ) - εσ/Δ) = δ

    where Φ is the standard normal CDF and Δ is the L2 sensitivity.

    Unlike the classical Dwork-Roth bound σ ≥ C·√(2·ln(1.25/δ))/ε which is
    only valid for ε ∈ (0, 1) (proof uses Taylor truncation requiring |ε| ≤ 1),
    this formula gives the tight σ for any ε > 0.

    Returns the smallest σ such that the Gaussian Mechanism is (ε, δ)-DP.
    Solved numerically via bisection (scipy.optimize.brentq).
    """
    if epsilon <= 0 or delta <= 0 or delta >= 1 or sensitivity <= 0:
        raise ValueError(f"Need ε > 0, 0 < δ < 1, sensitivity > 0; "
                         f"got ε={epsilon}, δ={delta}, Δ={sensitivity}")
    # Clamp ε to avoid math.exp overflow at extreme values (Inf · 0 → NaN).
    # Practical DP regime never needs ε > 100; this guard protects numerical
    # stability without affecting realistic configs.
    eps = min(float(epsilon), 700.0)

    def boundary(sigma: float) -> float:
        # f(σ) = δ - [Φ(a) - e^ε · Φ(b)]
        # σ tiny → f < 0 (privacy violated, RHS large)
        # σ large → f → δ > 0 (privacy slack, RHS → 0)
        # Find smallest σ where f(σ) = 0.
        a = sensitivity / (2.0 * sigma) - eps * sigma / sensitivity
        b = -sensitivity / (2.0 * sigma) - eps * sigma / sensitivity
        return delta - (norm.cdf(a) - math.exp(eps) * norm.cdf(b))

    lo, hi = sensitivity * 1e-6, sensitivity * 1e3
    # Expand upper bracket if needed (very small ε requires very large σ)
    while boundary(hi) < 0 and hi < 1e12:
        hi *= 10.0
    return float(brentq(boundary, lo, hi, xtol=1e-9))


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
        # DP sigma floor: minimum σ s.t. single-round Gaussian Mechanism is
        # (ε_max, δ)-DP. Use Balle-Wang 2018 analytic Gaussian (tight for any
        # ε > 0). Replaces the Dwork-Roth bound σ ≥ C·√(2·ln(1.25/δ))/ε which
        # is only valid for ε ∈ (0, 1) and under-estimates required σ at the
        # default epsilon_max=50 regime (Bug #8 fix).
        self._sigma_floor = analytic_gaussian_sigma(
            epsilon=float(epsilon_max),
            delta=float(delta),
            sensitivity=float(clip_bound),
        )
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
        """Reshape-decomposed structured noise (a.k.a. "spectrum-aware").

        Pipeline:
          1. Pad gradient to length k * cols, reshape to matrix M ∈ R^{k×cols}
             where k = svd_reshape_k, cols = ceil(D/k).
          2. Truncated low-rank SVD: M ≈ U·Σ·V^T, rank = min(svd_rank, k, cols).
          3. Regularized inverse weights: w = 1/(Σ + 1e-8) (guard blow-up when
             singular value → 0).
          4. Random projection: ρ ~ N(0, σ²·I_rank).
          5. Reconstruct + flatten: n_spec = (U·diag(w⊙ρ)·V^T).flatten()[:D].

        Caveat — semantics of "spectrum" here:
          U, Σ, V are singular decomposition of the ARTIFICIAL 2D reshape of the
          flat gradient vector — NOT the spectrum of model weights/layers.
          Reshape mapping is sequential (g[i·cols + j] → M[i,j]) and does not
          align with layer boundaries; params from multiple layers (weights,
          biases, BN) get mixed into rows/cols without semantic structure.
          Honest naming would be "low-rank-shaped structured noise".

        Edge case: torch.svd_lowrank can raise RuntimeError on degenerate
        matrices → fallback to plain Gaussian noise of the same magnitude.
        """
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
        """Scale ONLY n_strat to fit remaining budget after n_DP (Bug #7 fix).

        Cap = (sigma_total)^2 * D — interprets sigma_total as per-dimension
        std cap (same unit as sigma_0). Without the D factor the cap is an
        absolute L2-norm threshold, which forces sigma_eff_post ≈ sigma_total/√D
        and inflates the privacy cost by ~D× when D is large.

        Bug #7 fix (Option A): n_DP is left UNTOUCHED to preserve its Gaussian
        distribution N(0, σ_DP²·I). Rescaling n_DP by a factor that depends on
        ‖n_DP‖² + ‖n_strat‖² destroys the Gaussian property — the post-cap n_DP
        becomes a complex mixture, breaking the Gaussian Mechanism privacy
        guarantee (Mironov 2017 RDP_α = α·C²/(2σ²) requires fixed-σ Gaussian).

        New cap logic:
          - budget_remain = cap - ‖n_DP‖²
          - if budget_remain ≤ 0 (n_DP alone fills budget) → n_strat = 0
          - else → scale n_strat to ‖n_strat‖² ≤ budget_remain (no-op if already fits)
          - n_DP always returned as-is

        Trade-off: occasionally ‖n_DP‖² > cap (rare with Gaussian concentration
        when σ_DP ≤ σ_total) → total noise exceeds cap. Acceptable — privacy
        guarantee from n_DP alone is exact; n_strat for robustness only.
        """
        cap = (self.sigma_total ** 2) * self.param_dim
        n_dp_energy = n_dp.norm().item() ** 2
        budget_remain = cap - n_dp_energy
        if budget_remain <= 0.0:
            return n_dp, torch.zeros_like(n_strat)
        n_strat_energy = n_strat.norm().item() ** 2
        if n_strat_energy <= budget_remain or n_strat_energy < 1e-12:
            return n_dp, n_strat
        factor_strat = math.sqrt(budget_remain / n_strat_energy)
        return n_dp, n_strat * factor_strat

    # -- Full pipeline --

    def compute_total_noise(
        self, gradient: torch.Tensor, prev_gradient: torch.Tensor,
        round_t: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Full 4-layer noise pipeline.

        Returns (total_noise, metrics_dict).

        Privacy note (Bug #9 fix — Lipschitz inflation auto-applied):
        n_strat depends on raw `gradient` (via directional, orthogonal, spectrum
        noise components) → output Y = g + n_DP + n_strat(g) is NOT a pure
        Gaussian Mechanism with sensitivity C; effective sensitivity is

            C_total = C · (1 + L_strat),  L_strat = 2·β_strat·z,  z = σ_DP/C

        The simulator's accountant.step() automatically passes effective_mult
        = σ_DP/C_total (instead of σ_DP/C) so Opacus reports rigorous ε
        already including the (1 + L_strat)² inflation. No manual correction
        needed at publish time. See REPORT.md §4.1 + Bug #9.
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
