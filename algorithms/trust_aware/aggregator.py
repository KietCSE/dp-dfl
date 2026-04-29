"""Trust-Aware D2B-DP aggregator (RMSE distance + softmax + momentum).

Implements Phase 3 (Inbound Evaluation) and Phase 4 (History & Aggregation)
of docs/Trust-Aware-D2B-DP.md:
    Step 3.1 — Cosine similarity  C_{i,j}^(t)
    Step 3.2 — RMSE distance      D_{i,j}^(t) = ||ΔW'_i - S_j||₂ / √D_total
    Step 3.3 — D_threshold (decay vs. C_DP floor) is computed by the simulator
               (it owns the per-layer σ²_l,t needed for C_DP^(t))
    Step 4.1 — Q_{i,j}^(t) = p_dist · p_cos
                with p_dist = exp(-D / D_threshold), p_cos = max(0, cos)
    Step 4.2 — T_{i,j}^(t) = α_T · T_{i,j}^(t-1) + (1 - α_T) · Q_{i,j}^(t)
    Step 4.3 — V = {j | T_{i,j}^(t) ≥ T_min}
                w_{i,j} = softmax(β_soft · T)
                S_agg   = Σ w_{i,j} · S_j
                V_agg^(t) = β_m · V_agg^(t-1) + (1 - β_m) · S_agg^(t)
                W_i^(t)   = W_i^(t-1) + V_agg^(t)

Per-round state lives on the node (V_agg, trust_scores).
"""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from dpfl.core.base_aggregator import AggregationResult, BaseAggregator
from dpfl.registry import AGGREGATORS, register


@register(AGGREGATORS, "trust_aware_d2b")
class TrustAwareD2BAggregator(BaseAggregator):
    """Soft-trust aggregator with momentum for Trust-Aware D2B-DP."""

    def __init__(self, theta: float = 1.2, gamma: float = 3.0, kappa: float = 0.2,
                 alpha_T: float = 0.85, T_min: float = 0.4, beta_soft: float = 8.0,
                 beta_m: float = 0.9, **_kwargs):
        # theta/gamma/kappa are consumed by the simulator (D_threshold compute);
        # parked on the aggregator so a single config block drives everything.
        self.theta = theta
        self.gamma = gamma
        self.kappa = kappa
        self.alpha_T = alpha_T
        self.T_min = T_min
        self.beta_soft = beta_soft
        self.beta_m = beta_m

    def aggregate(self, own_update: torch.Tensor, own_params: torch.Tensor,
                  neighbor_updates: Dict[int, torch.Tensor],
                  W_old: Optional[torch.Tensor] = None,
                  V_agg_prev: Optional[torch.Tensor] = None,
                  D_threshold: float = 1.0,
                  trust_scores: Optional[Dict[int, float]] = None,
                  D_total: int = 0) -> AggregationResult:
        """Run trust update + softmax aggregation + momentum step.

        own_update     : ΔW'_i (clipped, no noise) — flat tensor.
        own_params     : current W_trained — fallback if W_old is missing.
        neighbor_updates: {j -> S_j} flat noisy packets from active neighbors.
        W_old          : start-of-round weights; required for the global step.
        V_agg_prev     : node's persistent momentum buffer; None ⇒ zeros.
        D_threshold    : precomputed RMS-distance threshold (Step 3.3).
        trust_scores   : node's persistent T_{i,·} dict; mutated in place.
        D_total        : total parameter count (for RMS scaling of distance).
        """
        if W_old is None:
            W_old = own_params - own_update
        if trust_scores is None:
            trust_scores = {}

        sqrt_D = math.sqrt(max(D_total, 1))
        per_neighbor: Dict[int, dict] = {}
        n_total = len(neighbor_updates)
        clean_ids: list = []
        flagged_ids: list = []

        if n_total == 0:
            S_agg = torch.zeros_like(own_update)
        else:
            # Steps 3.1, 3.2, 4.1, 4.2 — cosine, RMSE, instant Q, trust EMA
            for j, s_j in neighbor_updates.items():
                d_ij = (own_update - s_j).norm(2).item() / sqrt_D
                cos_ij = F.cosine_similarity(
                    own_update.unsqueeze(0), s_j.unsqueeze(0)).item()
                p_dist = math.exp(-d_ij / max(D_threshold, 1e-12))
                p_cos = max(0.0, cos_ij)
                q_ij = p_dist * p_cos
                prev_t = trust_scores.get(j, 1.0)
                t_ij = self.alpha_T * prev_t + (1.0 - self.alpha_T) * q_ij
                trust_scores[j] = t_ij
                per_neighbor[j] = {
                    "d_rms": d_ij, "cos": cos_ij, "p_dist": p_dist,
                    "p_cos": p_cos, "q": q_ij, "trust": t_ij,
                }

            # Step 4.3 — safe set V + softmax weights
            safe_ids = [j for j in neighbor_updates
                        if trust_scores.get(j, 0.0) >= self.T_min]
            if safe_ids:
                logits = torch.tensor(
                    [self.beta_soft * trust_scores[j] for j in safe_ids],
                    dtype=own_update.dtype)
                weights = F.softmax(logits, dim=0).tolist()
                S_agg = torch.zeros_like(own_update)
                for w, j in zip(weights, safe_ids):
                    S_agg = S_agg + w * neighbor_updates[j]
                    per_neighbor[j]["softmax_w"] = w
                clean_ids = list(safe_ids)
                flagged_ids = [j for j in neighbor_updates if j not in safe_ids]
            else:
                S_agg = torch.zeros_like(own_update)
                flagged_ids = list(neighbor_updates.keys())

        # Step 4.3 — momentum buffer + additive global step
        if V_agg_prev is None:
            V_agg_prev = torch.zeros_like(own_update)
        V_agg = self.beta_m * V_agg_prev + (1.0 - self.beta_m) * S_agg
        new_params = W_old + V_agg

        return AggregationResult(
            new_params=new_params,
            clean_ids=clean_ids,
            flagged_ids=flagged_ids,
            metrics={
                "D_threshold": D_threshold,
                "n_safe": len(clean_ids),
                "n_total": n_total,
                "neighbor_detail": per_neighbor,
                "V_agg": V_agg,
            },
            node_metrics={
                "D_threshold": float(D_threshold),
                "n_rejected": float(n_total - len(clean_ids)),
                "V_agg_norm": float(V_agg.norm().item()),
                "S_agg_norm": float(S_agg.norm().item()),
            },
        )
