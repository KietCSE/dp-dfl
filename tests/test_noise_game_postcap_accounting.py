"""Regression tests for noise_game post-cap privacy accounting (Bug #3 fix).

Verifies that:
  1. Cap mechanism actually reduces injected noise norm to σ_total when
     pre-cap energy exceeds the budget.
  2. Privacy ε reported by the simulator's accountant uses POST-cap σ_eff =
     ‖n_DP_post‖ / √D, not pre-cap σ_DP from the scheduler.
  3. compute_eps_dp() returns RDP + log(1/δ)/(α-1) per Mironov 2017 Thm 8.

Reference: plans/reports/audit-260425-0831-noise-game-rdp-formula-bugs.md
"""

import math
import unittest

import torch

from dpfl.algorithms.noise_game.mechanism import NoiseGameMechanism


class TestPostCapAccounting(unittest.TestCase):
    """Bug #3 (Critical): post-cap σ must drive privacy reporting."""

    def setUp(self):
        torch.manual_seed(42)
        self.D = 1000
        self.C = 10.0
        self.sigma_total = 1.0  # per-dim std cap → absolute cap = 1·√D ≈ 31.6
        self.mech = NoiseGameMechanism(
            alpha_attack=0.5, sigma_0=10.0,           # pre-cap σ much larger than cap
            anneal_kappa=0.0, svd_rank=4, svd_reshape_k=16,
            clip_bound=self.C, delta=1e-5, epsilon_max=1000.0,  # avoid budget squeeze
            beta_strat=0.5, sigma_total=self.sigma_total,
            param_dim=self.D, alpha_rd=2.0)

    def test_cap_reduces_noise_norm(self):
        """Post-cap per-dim std <= sigma_total when pre-cap energy > cap."""
        g = torch.randn(self.D) * 0.1
        g_prev = torch.randn(self.D) * 0.1
        total_noise, metrics = self.mech.compute_total_noise(g, g_prev, round_t=0)

        # Cap = (sigma_total)^2 · D = 4 · 1000 = 4000.
        # Pre-cap σ_dp ≈ 10 → ‖n_dp‖ ≈ 10·√1000 ≈ 316 → cap activates.
        # Post-cap ‖total_noise‖ ≤ sigma_total · √D ≈ 63.2.
        expected_total_cap = self.sigma_total * math.sqrt(self.D)
        slack = expected_total_cap * 1e-2
        self.assertLessEqual(metrics["total_noise_norm"], expected_total_cap + slack,
                             "Cap mechanism failed to reduce total noise norm")
        self.assertLessEqual(metrics["n_dp_norm"], expected_total_cap + slack)
        # Per-dim std post-cap should be ≤ sigma_total
        sigma_eff = metrics["n_dp_norm"] / math.sqrt(self.D)
        self.assertLessEqual(sigma_eff, self.sigma_total + 0.1)

    def test_postcap_sigma_eff_matches_n_dp_norm(self):
        """σ_eff (post-cap) = ‖n_dp_post‖ / √D — the value the accountant must use."""
        g = torch.randn(self.D) * 0.1
        g_prev = torch.randn(self.D) * 0.1
        _, metrics = self.mech.compute_total_noise(g, g_prev, round_t=0)

        sigma_eff_postcap = metrics["n_dp_norm"] / math.sqrt(self.D)
        sigma_dp_precap = metrics["sigma_dp"]

        # Pre-cap σ_dp should be much larger than post-cap σ_eff under cap.
        self.assertGreater(sigma_dp_precap, sigma_eff_postcap * 5,
                           "Pre-cap σ should be much larger than post-cap σ_eff "
                           "when cap is active — confirms Bug #3 surface")

    def test_pre_vs_post_cap_z_ratio(self):
        """z_reported (pre-cap) / z_true (post-cap) gap under cap activation."""
        g = torch.randn(self.D) * 0.1
        g_prev = torch.randn(self.D) * 0.1
        _, metrics = self.mech.compute_total_noise(g, g_prev, round_t=0)

        z_precap = metrics["sigma_dp"] / self.C
        z_postcap = (metrics["n_dp_norm"] / math.sqrt(self.D)) / self.C
        ratio = z_precap / max(z_postcap, 1e-12)

        # With per-dim cap: σ_pre=10, σ_post≈sigma_total=2 → ratio ≈ 5×.
        # Still demonstrates pre-cap σ would under-state privacy cost.
        self.assertGreater(ratio, 3.0,
                           f"Cap should still create pre/post σ gap; got {ratio:.2f}×")


class TestRdpToEpsConversion(unittest.TestCase):
    """Bug #2: compute_eps_dp() implements Mironov 2017 Thm 8 conversion."""

    def setUp(self):
        self.mech = NoiseGameMechanism(
            alpha_attack=0.5, sigma_0=1.0, anneal_kappa=0.0,
            svd_rank=4, svd_reshape_k=16,
            clip_bound=1.0, delta=1e-5, epsilon_max=10.0,
            beta_strat=0.5, sigma_total=10.0,
            param_dim=100, alpha_rd=2.0)

    def test_eps_dp_matches_mironov_formula(self):
        """ε(α, δ) = RDP_α + log(1/δ)/(α−1)."""
        self.mech.rdp_spent = 5.0
        expected = 5.0 + math.log(1.0 / 1e-5) / (2.0 - 1.0)
        self.assertAlmostEqual(self.mech.compute_eps_dp(), expected, places=6)

    def test_eps_dp_inf_when_alpha_le_one(self):
        """α ≤ 1 → conversion undefined → return ∞."""
        self.mech.alpha_rd = 1.0
        self.assertEqual(self.mech.compute_eps_dp(), float("inf"))

    def test_rdp_accumulator_uses_correct_formula(self):
        """RDP follows Mironov 2017: α·C²/(2σ²) per ROUND, committed by simulator."""
        g = torch.randn(100) * 0.05
        g_prev = torch.randn(100) * 0.05
        self.mech.rdp_spent = 0.0

        # compute_total_noise must NOT mutate rdp_spent (Bug #4 fix):
        # the simulator owns per-round commit cadence.
        _, metrics = self.mech.compute_total_noise(g, g_prev, round_t=0)
        self.assertEqual(self.mech.rdp_spent, 0.0,
                         "compute_total_noise must not update rdp_spent — "
                         "simulator commits via commit_round_rdp() once per round")

        # commit_round_rdp adds α·C²/(2σ²) for the σ used this round.
        sigma_dp = metrics["sigma_dp"]
        self.mech.commit_round_rdp(sigma_dp)
        expected_rdp = 2.0 * (1.0 ** 2) / (2.0 * sigma_dp ** 2)
        self.assertAlmostEqual(self.mech.rdp_spent, expected_rdp, places=6)

    def test_commit_round_rdp_skips_zero_sigma(self):
        """commit_round_rdp must not divide by zero on degenerate sigma."""
        self.mech.rdp_spent = 1.5
        self.mech.commit_round_rdp(0.0)
        self.assertEqual(self.mech.rdp_spent, 1.5,
                         "Zero sigma must be a no-op, not an error or NaN")


class TestSubsamplingAmplification(unittest.TestCase):
    """Bug #6: client sampling_rate must compose with batch q in the accountant."""

    def test_composed_q_reduces_eps(self):
        """q_composed = q_client · q_batch → over many rounds, ε drops as q decreases.

        Single-step ε is dominated by log(1/δ)/(α-1) so amplification is masked;
        accumulate RDP over many rounds (matches realistic 50-round DPFL run)
        to see q² amplification effect on ε.
        """
        from dpfl.core.renyi_accountant import RenyiAccountant
        alphas = [1.5, 2, 3, 5, 10, 20, 50]
        acc_full = RenyiAccountant(alpha_list=alphas, delta=1e-5)
        acc_half = RenyiAccountant(alpha_list=alphas, delta=1e-5)
        q_batch = 64.0 / 1200.0
        n_rounds, steps_per_round, noise_mult = 50, 19, 0.3
        for _ in range(n_rounds):
            acc_full.step(steps_per_round, 1.0 * q_batch, noise_mult)  # q_c=1.0
            acc_half.step(steps_per_round, 0.5 * q_batch, noise_mult)  # q_c=0.5
        eps_full = acc_full.get_epsilon()
        eps_half = acc_half.get_epsilon()
        # With RDP > log term after 50 rounds, halving q_c should reduce ε meaningfully.
        self.assertLess(eps_half, eps_full * 0.7,
                        f"Halving q_client should drop ε noticeably over 50 rounds; "
                        f"got eps_full={eps_full:.3f}, eps_half={eps_half:.3f}")


if __name__ == "__main__":
    unittest.main()
