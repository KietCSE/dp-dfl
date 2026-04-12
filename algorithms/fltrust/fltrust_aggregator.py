"""FLTrust aggregator: trust bootstrapping via root dataset.
Cao et al., NDSS 2021. Adapted for decentralized FL."""

import logging

import torch
import torch.nn.functional as F
from typing import Dict, Optional

from dpfl.core.base_aggregator import BaseAggregator, AggregationResult
from dpfl.registry import register, AGGREGATORS

logger = logging.getLogger(__name__)


@register(AGGREGATORS, "fltrust")
class FLTrustAggregator(BaseAggregator):
    """FLTrust: ReLU cosine trust scoring + norm-scaled weighted aggregation."""

    def __init__(self, trust_threshold: float = 0.1, **kwargs):
        self.trust_threshold = trust_threshold

    def aggregate(
        self,
        own_update: torch.Tensor,
        own_params: torch.Tensor,
        neighbor_updates: Dict[int, torch.Tensor],
        root_gradient: Optional[torch.Tensor] = None,
    ) -> AggregationResult:
        if not neighbor_updates or root_gradient is None:
            return AggregationResult(new_params=own_params + own_update)

        root_norm = root_gradient.norm()
        if root_norm < 1e-10:
            # Root gradient too small, fallback to simple average
            mean_update = torch.stack(list(neighbor_updates.values())).mean(dim=0)
            return AggregationResult(
                new_params=own_params + mean_update,
                clean_ids=list(neighbor_updates.keys()),
            )

        # Compute ReLU cosine trust scores
        trust_scores = {}
        for nid, g_i in neighbor_updates.items():
            cos_sim = F.cosine_similarity(
                root_gradient.unsqueeze(0), g_i.unsqueeze(0)
            ).item()
            trust_scores[nid] = max(0.0, cos_sim)

        # Normalize
        total_ts = sum(trust_scores.values())
        if total_ts < 1e-10:
            logger.warning("FLTrust: all trust scores zero, flagging all neighbors")
            return AggregationResult(
                new_params=own_params + own_update,
                flagged_ids=list(neighbor_updates.keys()),
            )
        norm_ts = {nid: ts / total_ts for nid, ts in trust_scores.items()}

        # Weighted + norm-scaled aggregation
        weighted_sum = torch.zeros_like(own_update)
        for nid, g_i in neighbor_updates.items():
            g_i_norm = g_i.norm()
            if g_i_norm > 1e-10:
                scaled = norm_ts[nid] * (root_norm / g_i_norm) * g_i
            else:
                scaled = torch.zeros_like(g_i)
            weighted_sum += scaled

        # Detection: low trust score -> flagged
        clean_ids = [nid for nid, ts in trust_scores.items()
                     if ts > self.trust_threshold]
        flagged_ids = [nid for nid, ts in trust_scores.items()
                       if ts <= self.trust_threshold]

        logger.debug("FLTrust: trust_scores=%s, flagged=%s",
                      {k: f"{v:.3f}" for k, v in trust_scores.items()}, flagged_ids)

        return AggregationResult(
            new_params=own_params + weighted_sum,
            clean_ids=clean_ids,
            flagged_ids=flagged_ids,
            metrics={"mean_trust": sum(trust_scores.values()) / len(trust_scores)},
            node_metrics={f"trust_{nid}": ts for nid, ts in trust_scores.items()},
        )
