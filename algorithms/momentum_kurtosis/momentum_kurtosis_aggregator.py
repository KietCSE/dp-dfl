"""Momentum Cosine + Kurtosis two-layer defense aggregator.

Layer 1 — Momentum Cosine (Karimireddy, He, Jaggi, ICML 2021)
    Per-caller state: EMA buffer m_j for each (caller, neighbor) pair and
    also for the caller itself (reference). Over rounds, DP Gaussian noise
    averages toward zero while a persistent attack direction (e.g. label
    flipping) remains, boosting SNR. Cosine similarity between the
    neighbor momentum and the caller momentum separates honest from
    direction-poisoning attackers. MAD-adaptive threshold handles non-IID
    and DP noise variance without a fixed cutoff. Disabled in the first
    `warmup_rounds` while momentum buffers converge.

Layer 2 — Sample Excess Kurtosis
    Same shape as KurtosisAvgAggregator. Flags neighbors whose update
    distribution deviates from Gaussian (scale, ALIE, gaussian-random).

Composition: AND-of-clean. A neighbor is accepted only if neither layer
flags it. Aggregation is simple averaging over the accepted set, added to
the caller's post-training params.

Notes:
    - State (momentum buffers, round counters) is keyed by caller node_id.
      The simulator passes `node_id` as a keyword argument so a single
      aggregator instance can serve all nodes.
    - Neighbors missing in a given round (e.g. Poisson subsampled out)
      skip their momentum update that round — their buffer simply carries
      the previous value.
    - own_params is post-training per the codebase convention; we add the
      mean of clean-neighbor updates to it.
"""

from typing import Dict, Optional, Set

import torch

from dpfl.core.base_aggregator import BaseAggregator, AggregationResult
from dpfl.registry import AGGREGATORS, register


@register(AGGREGATORS, "momentum_kurtosis_avg")
class MomentumKurtosisAggregator(BaseAggregator):
    """Two-layer defense: Momentum Cosine (L1) AND Kurtosis (L2) -> simple avg."""

    def __init__(
        self,
        param_dim: int,
        beta_m: float = 0.9,
        gamma_mad: float = 2.0,
        warmup_rounds: int = 5,
        kurtosis_confidence: float = 1.96,
        kurtosis_centered: bool = False,
    ):
        self.beta_m = float(beta_m)
        self.gamma_mad = float(gamma_mad)
        self.warmup = int(warmup_rounds)
        self.kurt_threshold = float(kurtosis_confidence) * (24.0 / float(param_dim)) ** 0.5
        self.centered = bool(kurtosis_centered)
        # Per-caller state. Lazily initialized on first call.
        self._momentum: Dict[int, Dict[int, torch.Tensor]] = {}
        self._round: Dict[int, int] = {}

    # -- main entry --

    def aggregate(
        self,
        own_update: torch.Tensor,
        own_params: torch.Tensor,
        neighbor_updates: Dict[int, torch.Tensor],
        node_id: Optional[int] = None,
        **_,
    ) -> AggregationResult:
        # Without caller id we cannot keep per-caller momentum — fall back
        # to kurtosis-only so the aggregator is still safe to use.
        if node_id is None:
            return self._kurtosis_only(own_update, own_params, neighbor_updates)

        t = self._advance_round(node_id)
        buf = self._momentum[node_id]
        self._update_momentum(buf, node_id, own_update)
        for j, upd in neighbor_updates.items():
            self._update_momentum(buf, j, upd)

        cos_scores = self._cosine_scores(buf, node_id, neighbor_updates)
        f1_flags, tau_cos = self._layer1_flags(cos_scores, round_t=t)
        kurt_scores, f2_flags = self._layer2_flags(neighbor_updates)

        malicious: Set[int] = f1_flags | f2_flags
        clean_ids = [j for j in neighbor_updates if j not in malicious]
        flagged_ids = [j for j in neighbor_updates if j in malicious]

        new_params = own_params
        if clean_ids:
            stack = torch.stack([neighbor_updates[j] for j in clean_ids])
            new_params = new_params + stack.mean(dim=0)

        own_kurt = self._excess_kurtosis(own_update)
        return AggregationResult(
            new_params=new_params,
            clean_ids=clean_ids,
            flagged_ids=flagged_ids,
            metrics={
                "cosine": cos_scores,
                "kurtosis": kurt_scores,
                "tau_cos": tau_cos,
                "tau_kurt": self.kurt_threshold,
                "f1_ids": sorted(f1_flags),
                "f2_ids": sorted(f2_flags),
                "round_t": t,
            },
            node_metrics={
                "own_kurtosis": float(own_kurt),
                "n_f1_flagged": float(len(f1_flags)),
                "n_f2_flagged": float(len(f2_flags)),
                "tau_cos": float(tau_cos),
            },
        )

    # -- state helpers --

    def _advance_round(self, node_id: int) -> int:
        if node_id not in self._momentum:
            self._momentum[node_id] = {}
            self._round[node_id] = 0
        self._round[node_id] += 1
        return self._round[node_id]

    def _update_momentum(
        self,
        buf: Dict[int, torch.Tensor],
        key: int,
        update: torch.Tensor,
    ) -> None:
        """m_key <- beta * m_key + (1-beta) * update, zero-init on first sight."""
        prev = buf.get(key)
        if prev is None:
            buf[key] = (1.0 - self.beta_m) * update.detach().clone()
        else:
            buf[key] = self.beta_m * prev + (1.0 - self.beta_m) * update.detach()

    # -- Layer 1: momentum cosine --

    @staticmethod
    def _cosine_scores(
        buf: Dict[int, torch.Tensor],
        owner: int,
        neighbor_updates: Dict[int, torch.Tensor],
    ) -> Dict[int, float]:
        m_own = buf[owner]
        own_norm = m_own.norm().item()
        scores: Dict[int, float] = {}
        for j in neighbor_updates:
            m_j = buf[j]
            denom = own_norm * m_j.norm().item() + 1e-12
            scores[j] = float((m_own @ m_j).item() / denom)
        return scores

    def _layer1_flags(
        self,
        cos_scores: Dict[int, float],
        round_t: int,
    ) -> (Set[int], float):
        # Warmup: momentum buffers not yet converged -> no L1 flagging.
        if round_t <= self.warmup or len(cos_scores) < 2:
            return set(), float("-inf")
        vals = torch.tensor(list(cos_scores.values()))
        med = vals.median().item()
        mad = (vals - med).abs().median().item() * 1.4826
        tau = med - self.gamma_mad * max(mad, 1e-12)
        flags = {j for j, s in cos_scores.items() if s < tau}
        return flags, float(tau)

    # -- Layer 2: kurtosis --

    def _layer2_flags(
        self,
        neighbor_updates: Dict[int, torch.Tensor],
    ) -> (Dict[int, float], Set[int]):
        scores: Dict[int, float] = {}
        flags: Set[int] = set()
        for j, upd in neighbor_updates.items():
            k = self._excess_kurtosis(upd)
            scores[j] = k
            if abs(k) > self.kurt_threshold:
                flags.add(j)
        return scores, flags

    def _excess_kurtosis(self, update: torch.Tensor) -> float:
        if self.centered:
            normalized = (update - update.mean()) / (update.std() + 1e-12)
        else:
            rms = update.pow(2).mean().sqrt()
            normalized = update / (rms + 1e-12)
        return float((normalized.pow(4).mean() - 3.0).item())

    # -- fallback (no caller id) --

    def _kurtosis_only(
        self,
        own_update: torch.Tensor,
        own_params: torch.Tensor,
        neighbor_updates: Dict[int, torch.Tensor],
    ) -> AggregationResult:
        kurt_scores, flags = self._layer2_flags(neighbor_updates)
        clean_ids = [j for j in neighbor_updates if j not in flags]
        flagged_ids = sorted(flags)
        new_params = own_params
        if clean_ids:
            stack = torch.stack([neighbor_updates[j] for j in clean_ids])
            new_params = new_params + stack.mean(dim=0)
        return AggregationResult(
            new_params=new_params,
            clean_ids=clean_ids,
            flagged_ids=flagged_ids,
            metrics={"kurtosis": kurt_scores, "tau_kurt": self.kurt_threshold,
                     "round_t": 0, "fallback": True},
            node_metrics={"own_kurtosis": self._excess_kurtosis(own_update)},
        )
