"""Unit tests for FLTrustAggregator: verify true decentralized FLTrust semantics.

Own_update is included in trust-weighted aggregation (one "client" among peers).
All updates normalized to ||root_gradient|| per Cao et al., NDSS 2021.
"""

import torch

from dpfl.algorithms.fltrust.fltrust_aggregator import FLTrustAggregator


def test_fltrust_own_included_in_weighted_sum():
    """When all updates point the same direction, weighted_sum scales to ||root||.

    Setup:
      root = [1, 0]    (||root|| = 1)
      own  = [2, 0]    (same direction, ||own|| = 2)
      neighbor 1 = [1, 0]  (||nbr|| = 1)

    Each trust = 1.0 (cosine with root); normalized = 0.5 each.
    weighted_sum = 0.5 * (1/2) * own + 0.5 * (1/1) * nbr
                 = [0.5, 0] + [0.5, 0] = [1.0, 0]
    new_params = initial + weighted_sum = [0, 0] + [1.0, 0] = [1.0, 0]
    """
    root = torch.tensor([1.0, 0.0])
    own = torch.tensor([2.0, 0.0])
    neighbors = {1: torch.tensor([1.0, 0.0])}
    initial = torch.zeros(2)
    own_params = initial + own

    agg = FLTrustAggregator(trust_threshold=0.01)
    result = agg.aggregate(own, own_params, neighbors, root_gradient=root)

    assert torch.allclose(result.new_params, torch.tensor([1.0, 0.0]), atol=1e-6)
    assert 1 in result.clean_ids


def test_fltrust_flags_opposite_direction():
    """Neighbor opposite to root gradient → cos < 0 → ReLU clamps to 0 → flagged."""
    root = torch.tensor([1.0, 0.0])
    own = torch.tensor([1.0, 0.0])
    neighbors = {1: torch.tensor([-1.0, 0.0])}
    own_params = own.clone()

    agg = FLTrustAggregator(trust_threshold=0.01)
    result = agg.aggregate(own, own_params, neighbors, root_gradient=root)

    assert 1 in result.flagged_ids
    assert 1 not in result.clean_ids


def test_fltrust_relu_clamps_orthogonal_neighbor():
    """Orthogonal neighbor (cos=0) → trust=0 → below threshold → flagged."""
    root = torch.tensor([1.0, 0.0])
    own = torch.tensor([1.0, 0.0])
    neighbors = {
        1: torch.tensor([0.0, 1.0]),  # orthogonal -> cos=0
        2: torch.tensor([1.0, 0.0]),  # aligned -> cos=1
    }
    own_params = own.clone()

    agg = FLTrustAggregator(trust_threshold=0.01)
    result = agg.aggregate(own, own_params, neighbors, root_gradient=root)

    assert 1 in result.flagged_ids
    assert 2 in result.clean_ids


def test_fltrust_self_never_in_flagged_ids():
    """Own_update is internally trust-scored but must not appear in neighbor-level
    detection output (flagged_ids / clean_ids). Only real neighbor IDs reported."""
    root = torch.tensor([1.0, 0.0])
    own = torch.tensor([-1.0, 0.0])  # adversarial own (hypothetical)
    neighbors = {1: torch.tensor([1.0, 0.0])}
    own_params = torch.zeros(2) + own

    agg = FLTrustAggregator(trust_threshold=0.01)
    result = agg.aggregate(own, own_params, neighbors, root_gradient=root)

    # Sentinel key "__self__" must never leak into clean/flagged ids.
    assert "__self__" not in result.flagged_ids
    assert "__self__" not in result.clean_ids
    # Only neighbor 1 is reported.
    assert set(result.clean_ids) | set(result.flagged_ids) == {1}


def test_fltrust_no_root_gradient_returns_own_params():
    """Degenerate case: root_gradient=None → return own_params unchanged."""
    own = torch.tensor([1.0, 2.0])
    own_params = torch.tensor([5.0, 6.0])
    neighbors = {1: torch.tensor([1.0, 1.0])}

    agg = FLTrustAggregator()
    result = agg.aggregate(own, own_params, neighbors, root_gradient=None)

    assert torch.allclose(result.new_params, own_params)
