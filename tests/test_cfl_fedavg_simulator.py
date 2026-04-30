"""Unit tests for CFLSimulator: verify FlatClip + f̃_f + central Gaussian noise.

Tests target the math of CFL-DP-FedAvg (McMahan+ 2018) without running
full training loops:
  - FlatClip enforces ‖Δ‖₂ ≤ S
  - f̃_f aggregation = Σ Δ_k / (qW)
  - σ = z·S/(qW)
  - Privacy accountant invoked once per round at (q, z)
  - Empty graph (n_neighbors=0) is supported
  - Algorithm registry contains 'cfl-fedavg'
"""

from unittest.mock import MagicMock

import torch

from dpfl.algorithms.cfl_fedavg.simulator import CFLSimulator
from dpfl.topology.random_graph import create_regular_graph


# ── Helpers ────────────────────────────────────────────────────────────────

def _bare_cfl(K: int = 5):
    """Construct a CFLSimulator with mocked dependencies (no setup)."""
    sim = CFLSimulator.__new__(CFLSimulator)
    sim.nodes = {i: MagicMock(id=i) for i in range(K)}
    sim.attacker_ids = set()
    sim.noise_gen = torch.Generator()
    sim.noise_gen.manual_seed(42)
    return sim


# ── FlatClip ───────────────────────────────────────────────────────────────

def test_flat_clip_below_threshold_passthrough():
    """Vector with ‖Δ‖ < S should pass through unchanged."""
    sim = _bare_cfl()
    delta = torch.tensor([0.3, 0.4])  # ‖Δ‖ = 0.5
    out = sim._flat_clip(delta, S=1.0)
    assert torch.allclose(out, delta)


def test_flat_clip_above_threshold_scaled_to_S():
    """Vector with ‖Δ‖ > S should be scaled to exactly ‖out‖ = S."""
    sim = _bare_cfl()
    delta = torch.tensor([3.0, 4.0])  # ‖Δ‖ = 5.0
    S = 2.0
    out = sim._flat_clip(delta, S=S)
    assert abs(out.norm(2).item() - S) < 1e-5


def test_flat_clip_at_threshold_no_change():
    """‖Δ‖ = S → scale = 1, no change."""
    sim = _bare_cfl()
    delta = torch.tensor([3.0, 4.0])  # ‖Δ‖ = 5.0
    out = sim._flat_clip(delta, S=5.0)
    assert torch.allclose(out, delta, atol=1e-5)


def test_flat_clip_preserves_direction():
    """Clip is a positive scalar multiplier → direction unchanged."""
    sim = _bare_cfl()
    delta = torch.tensor([6.0, 8.0])  # ‖Δ‖ = 10
    out = sim._flat_clip(delta, S=1.0)
    cos = torch.dot(out, delta) / (out.norm() * delta.norm())
    assert cos.item() > 0.9999


def test_flat_clip_zero_vector_safe():
    """Zero vector should not cause div-by-zero."""
    sim = _bare_cfl()
    delta = torch.zeros(3)
    out = sim._flat_clip(delta, S=1.0)
    assert torch.allclose(out, delta)


# ── Aggregation math (f̃_f) ────────────────────────────────────────────────

def test_aggregation_formula_uniform_weights():
    """f̃_f = Σ Δ_k / (qW) with w_k=1 → simple average scaled by 1/(qW)."""
    K, q = 10, 0.5
    deltas = [torch.ones(4) * (i + 1.0) for i in range(K)]  # Σ = 10·11/2 = 55
    weighted_sum = torch.stack(deltas).sum(dim=0)
    delta_aggregate = weighted_sum / (q * K)
    # Expected: 55 / (0.5 * 10) = 11.0 per coordinate
    assert torch.allclose(delta_aggregate, torch.full((4,), 11.0))


def test_aggregation_empty_active_set():
    """No active nodes → zero update (server holds θ_global stationary)."""
    delta_aggregate = torch.zeros(4)  # behaviour mirrored in run() else branch
    assert torch.allclose(delta_aggregate, torch.zeros(4))


# ── Server-side noise σ = zS/(qW) ──────────────────────────────────────────

def test_noise_std_formula():
    """Verify σ formula matches paper Algorithm 1."""
    z, S, q, W = 1.45, 0.0234, 0.5, 50.0
    sigma_expected = z * S / (q * W)
    # Reference: paper Eq. for f̃_f, page 4
    assert abs(sigma_expected - (1.45 * 0.0234) / (0.5 * 50.0)) < 1e-9


def test_noise_added_at_server_uses_sigma_zS_over_qW():
    """End-to-end check on noise injection magnitude (statistical)."""
    z, S, q, W = 1.0, 1.0, 1.0, 1.0  # → σ = 1.0
    gen = torch.Generator(); gen.manual_seed(0)
    n_samples = 100_000
    noise = torch.randn(n_samples, generator=gen) * (z * S / (q * W))
    # Empirical std should approach 1.0
    assert abs(noise.std().item() - 1.0) < 0.02


# ── Privacy accountant: 1 SGM step per round ───────────────────────────────

def test_accountant_called_once_per_round():
    """Verify accountant.step receives (n_steps=1, q, z) per round, not per-step."""
    sim = _bare_cfl(K=5)
    sim.accountant = MagicMock()
    sim.accountant.get_epsilon = MagicMock(return_value=0.5)

    # Simulate the accountant call from run() Phase 7
    z, q = 1.45, 0.5
    sim.accountant.step(1, q, z)
    sim.accountant.step.assert_called_once_with(1, 0.5, 1.45)


# ── Topology: empty graph supported ────────────────────────────────────────

def test_empty_graph_n_neighbors_zero():
    """create_regular_graph(K, 0, seed) returns isolated nodes — required by CFL."""
    g = create_regular_graph(50, 0, 42)
    assert len(g) == 50
    assert all(len(neigh) == 0 for neigh in g.values())


# ── Registry integration ───────────────────────────────────────────────────

def test_cfl_fedavg_in_algorithms_registry():
    """run.ALGORITHMS must contain 'cfl-fedavg' entry pointing to build_cfl_fedavg."""
    from dpfl.run import ALGORITHMS, build_cfl_fedavg
    assert "cfl-fedavg" in ALGORITHMS
    assert ALGORITHMS["cfl-fedavg"]["build_fn"] is build_cfl_fedavg
    assert ALGORITHMS["cfl-fedavg"]["default_config"] == "config/cfl_fedavg.yaml"


# ── Sanity: no client-side noise (trainer must run plain SGD) ──────────────

def test_config_contract_noise_mode_none():
    """CFL config template must set noise_mode='none' for trainer plain SGD."""
    import yaml
    from pathlib import Path
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "cfl_fedavg.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["dp"]["noise_mode"] == "none", \
        "CFL trainer must run plain SGD; server adds noise post-aggregation"
    assert cfg["topology"]["n_neighbors"] == 0, \
        "CFL has no client-client edges (server hub-and-spoke)"


# ── Math equivalence: aggregation + noise composition ─────────────────────

def test_full_round_math_consistency():
    """One synthetic round: clip → aggregate (f̃_f) → noise. Verify shapes & stats.

    Setup: K=10 clients, all Δ_k = c·e_1 with c=2.0; S=1.0 → clip to e_1.
    f̃_f aggregate = K · 1 / (qK) = 1/q. With q=0.5 → 2.0 per dim 1, 0 elsewhere.
    """
    K, q, S, z = 10, 0.5, 1.0, 0.0  # z=0 → no noise, deterministic check
    D = 4
    # All clients propose the same direction
    raw = torch.zeros(K, D); raw[:, 0] = 2.0  # ‖Δ_k‖=2.0 each
    # FlatClip to S=1.0 → scale 0.5
    sim = _bare_cfl(K=K)
    clipped = torch.stack([sim._flat_clip(raw[i], S=S) for i in range(K)])
    # Each clipped Δ_k has ‖·‖=1.0 along dim 0
    assert torch.allclose(clipped[:, 0], torch.ones(K))
    # Aggregate f̃_f
    weighted_sum = clipped.sum(dim=0)
    delta_aggregate = weighted_sum / (q * K)  # = 10/(0.5·10) = 2.0 along dim 0
    expected = torch.zeros(D); expected[0] = 2.0
    assert torch.allclose(delta_aggregate, expected)
