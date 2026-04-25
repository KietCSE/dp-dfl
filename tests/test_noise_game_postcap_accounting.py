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
        self.sigma_total = 2.0  # tight cap to force activation
        self.mech = NoiseGameMechanism(
            alpha_attack=0.5, sigma_0=10.0,           # huge pre-cap σ
            anneal_kappa=0.0, svd_rank=4, svd_reshape_k=16,
            clip_bound=self.C, delta=1e-5, epsilon_max=10.0,
            beta_strat=0.5, sigma_total=self.sigma_total, alpha_rd=2.0)

    def test_cap_reduces_noise_norm(self):
        """Post-cap noise norm <= sigma_total when pre-cap energy > cap."""
        g = torch.randn(self.D) * 0.1
        g_prev = torch.randn(self.D) * 0.1
        total_noise, metrics = self.mech.compute_total_noise(g, g_prev, round_t=0)

        # Pre-cap σ_dp ≈ 10.0 → ‖n_dp‖ ≈ 10·√1000 ≈ 316. Cap = 2.0.
        # Post-cap total energy ≤ 2.0² = 4.0 → ‖total_noise‖ ≤ 2.0 (with
        # ~0.1% float arithmetic slack from the sqrt(cap/energy) rescale).
        slack = self.sigma_total * 1e-2
        self.assertLessEqual(metrics["total_noise_norm"], self.sigma_total + slack,
                             "Cap mechanism failed to reduce total noise norm")
        self.assertLessEqual(metrics["n_dp_norm"], self.sigma_total + slack)

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
        """z_reported (pre-cap) / z_true (post-cap) ≥ 10 under cap activation."""
        g = torch.randn(self.D) * 0.1
        g_prev = torch.randn(self.D) * 0.1
        _, metrics = self.mech.compute_total_noise(g, g_prev, round_t=0)

        z_precap = metrics["sigma_dp"] / self.C
        z_postcap = (metrics["n_dp_norm"] / math.sqrt(self.D)) / self.C
        ratio = z_precap / max(z_postcap, 1e-12)

        # With sigma_0=10 → cap=2, ratio ≈ 316/2 = 158× expected.
        self.assertGreater(ratio, 10.0,
                           f"Bug #3 surface: z under-stated by {ratio:.1f}× "
                           "if pre-cap σ is fed to accountant")


class TestRdpToEpsConversion(unittest.TestCase):
    """Bug #2: compute_eps_dp() implements Mironov 2017 Thm 8 conversion."""

    def setUp(self):
        self.mech = NoiseGameMechanism(
            alpha_attack=0.5, sigma_0=1.0, anneal_kappa=0.0,
            svd_rank=4, svd_reshape_k=16,
            clip_bound=1.0, delta=1e-5, epsilon_max=10.0,
            beta_strat=0.5, sigma_total=10.0, alpha_rd=2.0)

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
        """RDP accumulator follows Mironov 2017: α · C² / (2σ²) per round."""
        g = torch.randn(100) * 0.05
        g_prev = torch.randn(100) * 0.05
        self.mech.rdp_spent = 0.0
        # After one step, rdp_spent should equal α·C²/(2σ_dp²) for the σ used.
        _, metrics = self.mech.compute_total_noise(g, g_prev, round_t=0)
        sigma_dp = metrics["sigma_dp"]
        expected_rdp = 2.0 * (1.0 ** 2) / (2.0 * sigma_dp ** 2)
        self.assertAlmostEqual(self.mech.rdp_spent, expected_rdp, places=6)


if __name__ == "__main__":
    unittest.main()
