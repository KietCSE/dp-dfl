"""FLTrust aggregator: trust bootstrapping via root dataset.

Cao et al., NDSS 2021. True decentralized FLTrust: own_update included in
trust-weighted aggregation (treated as one client among neighbors). Node acts
as its own "server" using local root gradient as trust anchor.
"""

import logging

import torch
import torch.nn.functional as F
from typing import Dict, Optional

from dpfl.core.base_aggregator import BaseAggregator, AggregationResult
from dpfl.registry import register, AGGREGATORS

logger = logging.getLogger(__name__)

# Sentinel key used to include own_update in trust-weighted aggregation.
_SELF_KEY = "__self__"


@register(AGGREGATORS, "fltrust")
class FLTrustAggregator(BaseAggregator):
    """FLTrust: ReLU cosine trust scoring + norm-scaled weighted aggregation.

    In decentralized adaptation, own_update is treated as one "client" with its
    own trust score (typically ≈ 1 since cos(root, own) is high). All updates
    are normalized to ||root_gradient|| and weighted by normalized trust.
    """

    def __init__(self, trust_threshold: float = 0.1, **kwargs):
        self.trust_threshold = trust_threshold

    def aggregate(
        self,
        own_update: torch.Tensor,
        own_params: torch.Tensor,
        neighbor_updates: Dict[int, torch.Tensor],
        root_gradient: Optional[torch.Tensor] = None,
    ) -> AggregationResult:
        # own_params is post-training (initial + own_update); revert to initial anchor.
        initial_params = own_params - own_update

        if root_gradient is None:
            return AggregationResult(new_params=own_params)

        root_norm = root_gradient.norm()
        if root_norm < 1e-10:
            # Root gradient too small: fallback to simple average of own + neighbors.
            all_updates = [own_update] + list(neighbor_updates.values())
            mean_update = torch.stack(all_updates).mean(dim=0)
            return AggregationResult(
                new_params=initial_params + mean_update,
                clean_ids=list(neighbor_updates.keys()),
            )

        # Trust-score own_update + neighbors uniformly.
        all_updates = {_SELF_KEY: own_update, **neighbor_updates}
        trust_scores: Dict = {}
        for key, g_i in all_updates.items():
            cos_sim = F.cosine_similarity(
                root_gradient.unsqueeze(0), g_i.unsqueeze(0)
            ).item()
            trust_scores[key] = max(0.0, cos_sim)

        total_ts = sum(trust_scores.values())
        if total_ts < 1e-10:
            logger.warning("FLTrust: all trust scores zero (incl. self), reverting to initial")
            return AggregationResult(
                new_params=initial_params,
                flagged_ids=list(neighbor_updates.keys()),
            )
        norm_ts = {k: ts / total_ts for k, ts in trust_scores.items()}

        # Norm-scaled weighted aggregation: each update scaled to ||root||.
        weighted_sum = torch.zeros_like(own_update)
        for key, g_i in all_updates.items():
            g_i_norm = g_i.norm()
            if g_i_norm > 1e-10:
                weighted_sum += norm_ts[key] * (root_norm / g_i_norm) * g_i

        # Detection: only report neighbors (self is always "trusted" by design).
        clean_ids = [nid for nid, ts in trust_scores.items()
                     if nid != _SELF_KEY and ts > self.trust_threshold]
        flagged_ids = [nid for nid, ts in trust_scores.items()
                       if nid != _SELF_KEY and ts <= self.trust_threshold]

        logger.debug("FLTrust: trust_scores=%s, flagged=%s",
                     {k: f"{v:.3f}" for k, v in trust_scores.items()}, flagged_ids)

        neighbor_trust = {k: v for k, v in trust_scores.items() if k != _SELF_KEY}
        return AggregationResult(
            new_params=initial_params + weighted_sum,
            clean_ids=clean_ids,
            flagged_ids=flagged_ids,
            metrics={"mean_trust": (sum(neighbor_trust.values()) / len(neighbor_trust))
                     if neighbor_trust else 0.0},
            node_metrics={f"trust_{nid}": ts for nid, ts in neighbor_trust.items()},
        )
