"""BALANCE aggregator: distance-based filter with full-model norm threshold.

Filter rule (per honest node `i`, neighbor `j`):
    keep j iff  ||w_i - w_j||_2  <=  A · exp(-K · (t+1) / T) · ||w_i||_2

Aggregation:
    new_w_i = a · w_i + (1 - a) · (1 / S) · Σ_{j ∈ kept} w_j

In dpfl's delta-space DFL (each node has its own pre-state but converges to
near-consensus via aggregation), neighbors send Δ_j = w_j - w_pre. Treating
neighbors as "would-be" post-states from my pre-state (own_params - own_update),
the model distance simplifies:
    ||w_i - w_j||  ≈  ||own_update - neighbor_update||
The threshold scale uses ||own_params|| (post-training, full-model norm),
matching the Nebula reference implementation.

Round info: simulator must call `set_round(t, T)` once per round before the
per-node aggregation loop. `dpsgd_kurtosis` simulator does this via duck-typed
`hasattr(aggregator, 'set_round')` check, so this aggregator integrates without
changing the BaseAggregator signature.
"""

import logging
import math
from typing import Dict

import torch

from dpfl.core.base_aggregator import BaseAggregator, AggregationResult
from dpfl.registry import register, AGGREGATORS

logger = logging.getLogger(__name__)


@register(AGGREGATORS, "balance")
class BalanceAggregator(BaseAggregator):
    """BALANCE Byzantine-resilient aggregator (Nebula reference).

    Args:
        A:  threshold scale factor (default 2.0)
        K:  decay rate over rounds  (default 1.0)
        a:  own-model weight in convex combination (default 0.4)
    """

    def __init__(self, A: float = 2.0, K: float = 1.0, a: float = 0.4, **kwargs):
        self.A = float(A)
        self.K = float(K)
        self.a = float(a)
        self._round_t: int = 0
        self._total_rounds: int = 1

    def set_round(self, round_t: int, total_rounds: int) -> None:
        """Simulator hook: set current round (0-indexed) and total rounds."""
        self._round_t = int(round_t)
        self._total_rounds = max(int(total_rounds), 1)

    def aggregate(
        self,
        own_update: torch.Tensor,
        own_params: torch.Tensor,
        neighbor_updates: Dict[int, torch.Tensor],
    ) -> AggregationResult:
        if not neighbor_updates:
            return AggregationResult(
                new_params=own_params,
                clean_ids=[], flagged_ids=[],
                node_metrics={
                    "balance_threshold": 0.0,
                    "balance_kept": 0.0,
                    "balance_local_norm": float(own_params.norm(2).item()),
                },
            )

        # Threshold: A · exp(-K · (t+1)/T) · ||w_i||_2
        # (t+1 mirrors Nebula's `current_round = self.engine.round + 1`)
        local_norm = float(own_params.norm(2).item())
        decay = math.exp(-self.K * (self._round_t + 1) / self._total_rounds)
        threshold = self.A * decay * local_norm

        # Filter neighbors via delta-space distance
        # ||w_i - w_j|| ≈ ||own_update - neighbor_update|| under near-consensus
        filtered_ids: list = []
        flagged_ids: list = []
        for nid, n_update in neighbor_updates.items():
            distance = float((own_update - n_update).norm(2).item())
            if distance <= threshold:
                filtered_ids.append(nid)
            else:
                flagged_ids.append(nid)

        # Aggregate: new_delta = a · own_update + (1 - a) / S · Σ filtered_deltas
        if not filtered_ids:
            new_update = own_update.clone()
        else:
            S = len(filtered_ids)
            sum_filtered = torch.zeros_like(own_update)
            for nid in filtered_ids:
                sum_filtered = sum_filtered + neighbor_updates[nid]
            new_update = self.a * own_update + (1.0 - self.a) * sum_filtered / S

        # own_params is post-training (= initial + own_update); reconstruct
        initial_params = own_params - own_update
        new_params = initial_params + new_update

        return AggregationResult(
            new_params=new_params,
            clean_ids=filtered_ids,
            flagged_ids=flagged_ids,
            node_metrics={
                "balance_threshold": threshold,
                "balance_kept": float(len(filtered_ids)),
                "balance_local_norm": local_norm,
            },
        )
