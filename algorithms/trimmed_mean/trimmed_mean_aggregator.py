"""Coordinate-wise trimmed mean aggregator. Yin et al., ICML 2018.

Per-dimension: sort N values, trim top-k and bottom-k, average remaining.
Detection: nodes whose values are trimmed most often -> flagged.
"""

import logging

import torch
from typing import Dict

from dpfl.core.base_aggregator import BaseAggregator, AggregationResult
from dpfl.registry import register, AGGREGATORS

logger = logging.getLogger(__name__)


@register(AGGREGATORS, "trimmed_mean")
class TrimmedMeanAggregator(BaseAggregator):
    """Coordinate-wise trimmed mean: per-dimension sort, trim top/bottom k,
    average remaining. Yin et al., ICML 2018."""

    def __init__(self, trim_ratio: float = 0.2, **kwargs):
        self.trim_ratio = trim_ratio

    def aggregate(
        self,
        own_update: torch.Tensor,
        own_params: torch.Tensor,
        neighbor_updates: Dict[int, torch.Tensor],
    ) -> AggregationResult:
        if not neighbor_updates:
            return AggregationResult(
                new_params=own_params + own_update,
                clean_ids=[], flagged_ids=[],
            )

        # Stack: own (-1) + neighbors
        all_ids = [-1] + list(neighbor_updates.keys())
        all_updates = [own_update] + [neighbor_updates[nid] for nid in neighbor_updates]
        stacked = torch.stack(all_updates)  # (N, D)
        n = stacked.shape[0]
        k = max(1, int(self.trim_ratio * n))

        # Per-dimension trimmed mean
        sorted_vals, sorted_idx = torch.sort(stacked, dim=0)
        trimmed = sorted_vals[k: n - k]  # (N-2k, D)
        mean_update = trimmed.mean(dim=0)

        # Detection: count how often each node's values fall in trimmed extremes
        # Vectorized: flatten trimmed indices and scatter_add for O(1) GPU ops
        trim_count = torch.zeros(n, device=stacked.device)
        bottom_flat = sorted_idx[:k, :].reshape(-1)
        top_flat = sorted_idx[n - k:, :].reshape(-1)
        all_trimmed = torch.cat([bottom_flat, top_flat])
        trim_count.scatter_add_(0, all_trimmed,
                                torch.ones_like(all_trimmed, dtype=torch.float))

        # Flag nodes trimmed > 50% of dimensions
        threshold = 0.5 * stacked.shape[1]
        flagged_ids = []
        clean_ids = []
        for i, nid in enumerate(all_ids):
            if nid == -1:
                continue
            if trim_count[i] > threshold:
                flagged_ids.append(nid)
            else:
                clean_ids.append(nid)

        logger.debug("TrimmedMean: n=%d, k=%d, flagged=%s", n, k, flagged_ids)

        return AggregationResult(
            new_params=own_params + mean_update,
            clean_ids=clean_ids,
            flagged_ids=flagged_ids,
            metrics={"trim_k": k, "n_updates": n},
        )
