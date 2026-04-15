"""Kurtosis-based defense + simple averaging aggregation."""

import torch
from typing import Dict

from dpfl.core.base_aggregator import BaseAggregator, AggregationResult
from dpfl.registry import register, AGGREGATORS


@register(AGGREGATORS, "kurtosis_avg")
class KurtosisAvgAggregator(BaseAggregator):
    """
    Filter neighbors by excess kurtosis, then simple-average clean updates.
    Threshold: T_k = confidence * sqrt(24 / param_dim)
    """

    def __init__(self, param_dim: int, centered: bool = False, confidence: float = 1.96):
        self.threshold = confidence * (24.0 / param_dim) ** 0.5
        self.centered = centered

    def aggregate(self, own_update: torch.Tensor, own_params: torch.Tensor,
                  neighbor_updates: Dict[int, torch.Tensor]) -> AggregationResult:
        clean_ids, flagged_ids = [], []
        kurtosis_values = {}

        for j, update in neighbor_updates.items():
            k = self._excess_kurtosis(update)
            kurtosis_values[j] = k
            if abs(k) > self.threshold:
                flagged_ids.append(j)
            else:
                clean_ids.append(j)

        # own_params already contains own_update (post-training), add clean neighbor mean
        new_params = own_params
        if clean_ids:
            clean_stack = torch.stack([neighbor_updates[j] for j in clean_ids])
            new_params = new_params + clean_stack.mean(dim=0)

        own_kurtosis = self._excess_kurtosis(own_update)

        return AggregationResult(
            new_params=new_params,
            clean_ids=clean_ids,
            flagged_ids=flagged_ids,
            metrics={"kurtosis": kurtosis_values, "threshold": self.threshold},
            node_metrics={"kurtosis": float(own_kurtosis)},
        )

    def _excess_kurtosis(self, update: torch.Tensor) -> float:
        """Uncentered (RMS) or centered excess kurtosis."""
        if self.centered:
            normalized = (update - update.mean()) / (update.std() + 1e-12)
        else:
            rms = update.pow(2).mean().sqrt()
            normalized = update / (rms + 1e-12)
        return (normalized.pow(4).mean() - 3.0).item()
