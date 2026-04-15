"""FedAvg aggregator: weighted average of updates by dataset proportion.

McMahan et al. 2017: w_k = n_k / Σ n_j for all participants (own + neighbors).
"""

import torch
from typing import Dict, Optional

from dpfl.core.base_aggregator import BaseAggregator, AggregationResult
from dpfl.registry import register, AGGREGATORS


@register(AGGREGATORS, "fedavg")
class FedAvgAggregator(BaseAggregator):
    """Dataset-proportional weighted average (FedAvg).

    Each update is weighted by n_k / Σ n_j where n_k = dataset size of node k.
    Requires set_data_sizes() after simulator setup.
    """

    def __init__(self, **kwargs):
        # node_id -> n_samples, populated by simulator after setup()
        self.data_sizes: Dict[int, int] = {}

    def set_data_sizes(self, data_sizes: Dict[int, int]):
        """Store dataset sizes for weighted aggregation."""
        self.data_sizes = data_sizes

    def aggregate(self, own_update: torch.Tensor, own_params: torch.Tensor,
                  neighbor_updates: Dict[int, torch.Tensor],
                  own_node_id: Optional[int] = None) -> AggregationResult:
        neighbor_ids = list(neighbor_updates.keys())
        participant_ids = [own_node_id] + neighbor_ids

        # Weighted average: w_k = n_k / Σ n_j
        weights = torch.tensor(
            [self.data_sizes.get(nid, 1) for nid in participant_ids],
            dtype=own_update.dtype, device=own_update.device)
        weights = weights / weights.sum()

        all_updates = torch.stack(
            [own_update] + [neighbor_updates[j] for j in neighbor_ids])
        avg_update = (all_updates * weights.unsqueeze(1)).sum(dim=0)

        # own_params is post-training (= initial + own_update), subtract to get initial anchor
        initial_params = own_params - own_update
        return AggregationResult(
            new_params=initial_params + avg_update,
            clean_ids=neighbor_ids,
            flagged_ids=[],
        )
