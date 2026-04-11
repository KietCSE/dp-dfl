"""Trust-Aware D2B aggregator: Z-Score + Cosine + MAD + Trust + Penalty."""

import torch
import torch.nn.functional as F
from collections import deque
from typing import Dict

from dpfl.core.base_aggregator import BaseAggregator, AggregationResult
from dpfl.registry import register, AGGREGATORS


@register(AGGREGATORS, "trust_aware_d2b")
class TrustAwareD2BAggregator(BaseAggregator):
    """Full D2B-DP defense: Z-Score -> Cosine -> Harmonic -> EMA -> MAD -> Filter."""

    def __init__(self, param_dim: int, ema_lambda: float = 0.8,
                 gamma_z: float = 3.0, sigma_floor_z: float = 1e-4,
                 alpha_drop: float = 2.0, sigma_floor_drop: float = 1e-3,
                 gamma_penalty: float = 0.5):
        self.ema_lambda = ema_lambda
        self.gamma_z = gamma_z
        self.sigma_floor_z = sigma_floor_z
        self.alpha_drop = alpha_drop
        self.sigma_floor_drop = sigma_floor_drop
        self.gamma_penalty = gamma_penalty

    # -- Step 6: Z-Score --

    @staticmethod
    def _compute_zscore(gradient: torch.Tensor) -> float:
        """Z(S_j) = |mean(S_j)| — detects magnitude anomaly."""
        return abs(gradient.mean().item())

    def _compute_z_threshold(self, z_score_buffer: deque) -> float:
        """MAD-based robust threshold from temporal Z-score buffer."""
        if len(z_score_buffer) < 2:
            return float('inf')  # Don't filter early rounds
        values = torch.tensor(list(z_score_buffer))
        med = values.median()
        mad = max((values - med).abs().median().item(), self.sigma_floor_z)
        return med.item() + self.gamma_z * mad

    # -- Step 7: Cosine + Harmonic --

    @staticmethod
    def _compute_similarity(own_update: torch.Tensor,
                            neighbor_update: torch.Tensor) -> float:
        """Cosine similarity normalized to [0, 1]."""
        cos = F.cosine_similarity(
            own_update.unsqueeze(0), neighbor_update.unsqueeze(0),
        )
        return 0.5 * (cos.item() + 1.0)

    @staticmethod
    def _behavior_score(p_safe: float, p_sim: float) -> float:
        """Harmonic mean — both P_safe and P_sim must be high."""
        return 2.0 * p_safe * p_sim / (p_safe + p_sim + 1e-12)

    # -- Step 8: EMA trust update --

    def _update_trust(self, current: float, behavior: float) -> float:
        return self.ema_lambda * current + (1.0 - self.ema_lambda) * behavior

    # -- Step 9: Dynamic tau_drop --

    def _compute_drop_threshold(self, trust_buffer: deque) -> float:
        """MAD-based robust threshold for trust filtering."""
        if len(trust_buffer) < 2:
            return 0.0  # Don't filter early rounds
        values = torch.tensor(list(trust_buffer))
        med = values.median()
        mad = max((values - med).abs().median().item(), self.sigma_floor_drop)
        return med.item() - self.alpha_drop * mad

    # -- Main aggregate (Steps 6-11) --

    def aggregate(self, own_update, own_params, neighbor_updates,
                  node_trust_scores=None, z_score_buffer=None,
                  trust_buffer=None):
        """
        Full D2B-DP pipeline. Extended signature with trust context.
        Falls back to simple mean if no trust context (ABC compat).
        """
        # Fallback: no trust context -> simple mean
        if node_trust_scores is None:
            stack = torch.stack(list(neighbor_updates.values()))
            return AggregationResult(
                new_params=own_params + own_update + stack.mean(0),
                clean_ids=list(neighbor_updates.keys()), flagged_ids=[])

        # Phase 3: Inbound evaluation
        z_threshold = self._compute_z_threshold(z_score_buffer)
        z_values = []
        per_neighbor = {}

        for j, s_j in neighbor_updates.items():
            z_j = self._compute_zscore(s_j)
            z_values.append(z_j)
            p_safe = max(0.0, 1.0 - z_j / (z_threshold + 1e-12))
            p_sim = self._compute_similarity(own_update, s_j)
            r_ij = self._behavior_score(p_safe, p_sim)
            node_trust_scores[j] = self._update_trust(node_trust_scores[j], r_ij)
            per_neighbor[j] = {"z": z_j, "p_safe": p_safe, "p_sim": p_sim,
                               "behavior": r_ij, "trust": node_trust_scores[j]}

        z_score_buffer.extend(z_values)

        # Phase 4: Trust buffer -> MAD filtering -> penalty
        trust_buffer.extend(node_trust_scores.values())
        tau_drop = self._compute_drop_threshold(trust_buffer)

        clean_ids, flagged_ids = [], []
        for j in neighbor_updates:
            if node_trust_scores[j] < tau_drop:
                flagged_ids.append(j)
                node_trust_scores[j] *= self.gamma_penalty  # Step 10: Penalty
            else:
                clean_ids.append(j)

        # Step 11: Aggregate from clean set only
        new_params = own_params + own_update
        if clean_ids:
            new_params = new_params + torch.stack(
                [neighbor_updates[j] for j in clean_ids]).mean(0)

        mean_trust = sum(node_trust_scores.values()) / max(len(node_trust_scores), 1)
        return AggregationResult(
            new_params=new_params,
            clean_ids=clean_ids,
            flagged_ids=flagged_ids,
            metrics={"z_threshold": z_threshold, "tau_drop": tau_drop,
                     "neighbor_detail": per_neighbor},
            node_metrics={"mean_trust": mean_trust, "tau_drop": tau_drop,
                          "n_rejected": float(len(flagged_ids))},
        )
