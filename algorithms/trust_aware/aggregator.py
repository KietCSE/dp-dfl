"""Trust-Aware D2B-DP aggregator (RMS distance + softmax + momentum).

Implements Steps 4-7 of docs/Trust-Aware-D2B-DP.md:
    Step 4 — RMS distance D_{ij} and cosine similarity C_{ij}
    Step 5 — D_threshold (decay term vs. C_DP floor) is computed by simulator
             (it needs per-layer σ² — kept out of the aggregator state)
    Step 6 — Continuous trust update: Q = p_dist · p_cos, T ← α_T·T + (1-α_T)·Q
    Step 7 — Softmax aggregation over safe set V = {j | T_{ij} ≥ T_min},
             then momentum:  V_agg = β_m·V_agg + (1-β_m)·S_agg
                              W    = W_old + η_global·V_agg

Sign note (doc convention): Sec 7 of the spec writes
    W_i^(t) = W_i^(t-1) - η_global · V_agg
but ΔW = W_trained - W_old is positive forward-progress, so subtracting drives
the model away from the consensus direction. We use `+` to match every other
delta-aggregation algorithm in this codebase (FedAvg / Krum / Trimmed Mean) and
to converge toward neighbors' learning signal — interpreting the doc's `-` as a
notational convention error. State (V_agg, trust_scores) lives on the node.
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

    def __init__(self, theta: float = 1.1, gamma: float = 2.5, kappa: float = 5.0,
                 alpha_T: float = 0.85, T_min: float = 0.3, beta_soft: float = 3.0,
                 beta_m: float = 0.9, eta_global: float = 0.01, **_kwargs):
        # theta/gamma/kappa are consumed by the simulator (D_threshold compute)
        # but kept on the aggregator so a single config block drives everything.
        self.theta = theta
        self.gamma = gamma
        self.kappa = kappa
        self.alpha_T = alpha_T
        self.T_min = T_min
        self.beta_soft = beta_soft
        self.beta_m = beta_m
        self.eta_global = eta_global

    def aggregate(self, own_update: torch.Tensor, own_params: torch.Tensor,
                  neighbor_updates: Dict[int, torch.Tensor],
                  W_old: Optional[torch.Tensor] = None,
                  V_agg_prev: Optional[torch.Tensor] = None,
                  D_threshold: float = 1.0,
                  trust_scores: Optional[Dict[int, float]] = None,
                  D_total: int = 0) -> AggregationResult:
        """Run trust update + softmax aggregation + momentum step.

        own_update     : ΔW'_i (clipped, NO noise) — flat tensor.
        own_params     : current W_trained (post-train) — used as fallback if
                         W_old is not provided (W_old = own_params - own_update).
        neighbor_updates: {j -> S_j} flat noisy packets from active neighbors.
        W_old          : start-of-round weights; required for the global step.
        V_agg_prev     : node's persistent momentum buffer; None ⇒ zeros.
        D_threshold    : precomputed RMS-distance threshold (Step 5).
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
            # Step 4-6: distance, cosine, trust EMA per neighbor
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

            # Step 7: safe set V + softmax weights
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

        # Momentum buffer + global step
        if V_agg_prev is None:
            V_agg_prev = torch.zeros_like(own_update)
        V_agg = self.beta_m * V_agg_prev + (1.0 - self.beta_m) * S_agg
        new_params = W_old - self.eta_global * V_agg

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
