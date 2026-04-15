"""FLAME aggregator: clustering + adaptive clipping + DP noise.
Nguyen et al., USENIX Security 2022. Adapted for decentralized FL."""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict

from dpfl.core.base_aggregator import BaseAggregator, AggregationResult
from dpfl.registry import register, AGGREGATORS

logger = logging.getLogger(__name__)


@register(AGGREGATORS, "flame")
class FLAMEAggregator(BaseAggregator):
    """FLAME: cosine-distance clustering + adaptive norm clipping + DP noise."""

    def __init__(self, noise_mult: float = 0.01, delta: float = 1e-5,
                 min_cluster_size: int = 2, **kwargs):
        self.noise_mult = noise_mult
        self.delta = delta
        self.min_cluster_size = min_cluster_size

    def aggregate(
        self,
        own_update: torch.Tensor,
        own_params: torch.Tensor,
        neighbor_updates: Dict[int, torch.Tensor],
    ) -> AggregationResult:
        if not neighbor_updates:
            return AggregationResult(new_params=own_params)  # own_params already post-training

        all_ids = [-1] + list(neighbor_updates.keys())
        all_updates = [own_update] + [neighbor_updates[nid] for nid in neighbor_updates]
        stacked = torch.stack(all_updates)  # (N, D)
        n = stacked.shape[0]

        # Step 1: Cosine-distance clustering
        clean_mask, cluster_labels = self._cluster_updates(stacked, n)

        clean_indices = [i for i in range(n) if clean_mask[i]]
        flagged_indices = [i for i in range(n) if not clean_mask[i]]
        clean_ids = [all_ids[i] for i in clean_indices if all_ids[i] != -1]
        flagged_ids = [all_ids[i] for i in flagged_indices if all_ids[i] != -1]

        if not clean_indices:
            logger.warning("FLAME: all nodes flagged, falling back to own update")
            return AggregationResult(
                new_params=own_params,  # own_params already post-training
                flagged_ids=[nid for nid in neighbor_updates],
            )

        clean_updates = stacked[clean_indices]

        # Step 2: Adaptive norm clipping (median of clean cluster)
        norms = clean_updates.norm(dim=1)
        median_norm = norms.median().item()
        clip_bound = max(median_norm, 1e-6)
        clipped = self._clip_updates(clean_updates, clip_bound)

        # Step 3: Average + DP noise
        avg_update = clipped.mean(dim=0)
        noised_update = self._add_dp_noise(avg_update, clip_bound, len(clean_indices))

        logger.debug("FLAME: clean=%d, flagged=%d, clip_bound=%.4f, median_norm=%.4f",
                      len(clean_indices), len(flagged_indices), clip_bound, median_norm)

        # own_params is post-training (= initial + own_update), use initial as anchor
        initial_params = own_params - own_update
        return AggregationResult(
            new_params=initial_params + noised_update,
            clean_ids=clean_ids,
            flagged_ids=flagged_ids,
            metrics={
                "n_clean": len(clean_indices),
                "n_flagged": len(flagged_indices),
                "clip_bound": clip_bound,
                "median_norm": median_norm,
            },
        )

    def _cluster_updates(self, stacked: torch.Tensor, n: int):
        """Cluster via cosine distance. Returns (clean_mask, labels)."""
        if n <= 3:
            return [True] * n, [0] * n

        # Cosine distance matrix
        normed = F.normalize(stacked, dim=1)
        cos_sim = normed @ normed.T
        cos_dist = (1.0 - cos_sim).clamp(min=0).cpu().numpy()

        try:
            from sklearn.cluster import HDBSCAN
            clusterer = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                metric="precomputed",
                allow_single_cluster=True,
            )
            labels = clusterer.fit_predict(cos_dist)
        except ImportError:
            labels = self._simple_outlier_detection(cos_dist, n)

        # Largest cluster = honest, rest = flagged
        unique_labels = set(labels)
        unique_labels.discard(-1)
        if not unique_labels:
            return [True] * n, labels.tolist() if hasattr(labels, 'tolist') else labels

        largest_label = max(unique_labels, key=lambda l: (labels == l).sum())
        clean_mask = [(labels[i] == largest_label) for i in range(n)]
        lbl_list = labels.tolist() if hasattr(labels, 'tolist') else labels
        return clean_mask, lbl_list

    def _simple_outlier_detection(self, cos_dist: np.ndarray, n: int):
        """Fallback when sklearn HDBSCAN unavailable."""
        avg_dist = cos_dist.sum(axis=1) / (n - 1)
        median_dist = np.median(avg_dist)
        mad = np.median(np.abs(avg_dist - median_dist))
        threshold = median_dist + 3.0 * max(mad, 1e-6)
        return np.where(avg_dist <= threshold, 0, -1)

    def _clip_updates(self, updates: torch.Tensor, clip_bound: float):
        """L2-norm clip each update."""
        norms = updates.norm(dim=1, keepdim=True)
        scale = torch.clamp(clip_bound / norms, max=1.0)
        return updates * scale

    def _add_dp_noise(self, avg_update: torch.Tensor, clip_bound: float,
                      n_clean: int):
        """Add Gaussian noise for (eps,delta)-DP."""
        sigma = self.noise_mult * clip_bound / max(n_clean, 1)
        noise = torch.randn_like(avg_update) * sigma
        return avg_update + noise
