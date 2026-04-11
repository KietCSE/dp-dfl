"""Simple averaging aggregator: no filtering, just mean of all neighbor updates."""

import torch
from typing import Dict

from dpfl.core.base_aggregator import BaseAggregator, AggregationResult
from dpfl.registry import register, AGGREGATORS


@register(AGGREGATORS, "simple_avg")
class SimpleAvgAggregator(BaseAggregator):
    """Average own update with all neighbor updates. No defense filtering."""

    def __init__(self, **kwargs):
        pass  # Accept and ignore any extra kwargs for registry compatibility

    def aggregate(self, own_update: torch.Tensor, own_params: torch.Tensor,
                  neighbor_updates: Dict[int, torch.Tensor]) -> AggregationResult:
        neighbor_ids = list(neighbor_updates.keys())
        all_updates = [own_update] + [neighbor_updates[j] for j in neighbor_ids]
        avg_update = torch.stack(all_updates).mean(dim=0)

        return AggregationResult(
            new_params=own_params + avg_update,
            clean_ids=neighbor_ids,
            flagged_ids=[],
        )
