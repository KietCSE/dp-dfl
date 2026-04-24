"""Krum aggregator: Byzantine-resilient aggregation via distance-based selection."""

import torch
from typing import Dict

from dpfl.core.base_aggregator import BaseAggregator, AggregationResult
from dpfl.registry import register, AGGREGATORS


@register(AGGREGATORS, "krum")
class KrumAggregator(BaseAggregator):
    """Multi-Krum: select top-m updates with smallest sum of distances to
    nearest (n - f - 2) neighbors, then average them.

    Args:
        n_attackers: expected number of Byzantine nodes (f)
        multi_k: number of updates to select (m). Default 1 = single Krum.
    """

    def __init__(self, n_attackers: int = 4, multi_k: int = 1, **kwargs):
        self.f = n_attackers
        self.m = multi_k

    def aggregate(self, own_update: torch.Tensor, own_params: torch.Tensor,
                  neighbor_updates: Dict[int, torch.Tensor]) -> AggregationResult:
        # Combine own + neighbor updates
        all_ids = [-1] + list(neighbor_updates.keys())  # -1 = own
        all_updates = [own_update] + [neighbor_updates[j] for j in neighbor_updates]
        n = len(all_updates)

        if n <= 1:
            return AggregationResult(
                new_params=own_params,  # own_params already post-training
                clean_ids=list(neighbor_updates.keys()), flagged_ids=[])

        # Compute pairwise distances
        stacked = torch.stack(all_updates)  # (n, D)
        dists = torch.cdist(stacked.unsqueeze(0), stacked.unsqueeze(0)).squeeze(0)  # (n, n)

        # For each update, sum SQUARED distances to nearest (n - f - 2) neighbors
        # per Blanchard et al. NeurIPS 2017: s(i) = Σ ||V_i - V_j||_2^2
        k = max(n - self.f - 2, 1)
        scores = []
        for i in range(n):
            sorted_dists, _ = dists[i].sort()
            # Skip self (distance 0), take next k nearest, square-sum
            scores.append(sorted_dists[1:k + 1].pow(2).sum().item())

        # Select top-m updates with smallest scores
        sorted_indices = sorted(range(n), key=lambda i: scores[i])
        selected = sorted_indices[:self.m]

        # Average selected updates
        selected_updates = torch.stack([all_updates[i] for i in selected])
        avg_update = selected_updates.mean(dim=0)

        # Determine which neighbor IDs were selected vs flagged
        selected_ids = set()
        for i in selected:
            if all_ids[i] != -1:
                selected_ids.add(all_ids[i])

        neighbor_ids = list(neighbor_updates.keys())
        clean_ids = [j for j in neighbor_ids if j in selected_ids]
        flagged_ids = [j for j in neighbor_ids if j not in selected_ids]

        # own_params is post-training (= initial + own_update), use initial as anchor
        initial_params = own_params - own_update
        return AggregationResult(
            new_params=initial_params + avg_update,
            clean_ids=clean_ids,
            flagged_ids=flagged_ids,
            node_metrics={"krum_score": min(scores)},
        )
