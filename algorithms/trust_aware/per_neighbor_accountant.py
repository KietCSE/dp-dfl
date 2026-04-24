"""Per-neighbor RDP accountant: tracks per-edge privacy budget for D2B-DP.

Backed by Opacus (Meta AI) SGM implementation (Mironov 2019) — tight RDP
bounds for subsampled Gaussian, covering both integer and fractional α.
Per-edge state lives in `node.per_neighbor_rdp[neighbor_id]`. Sensitivity
(clip bound C) and noise variance σ² vary per-edge per-round → Opacus called
with `steps=1` each call and results accumulated additively.
"""

import math
from typing import List

from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent


class PerNeighborRDPAccountant:
    """Tracks per-edge RDP cost. State lives in TrustAwareNode.per_neighbor_rdp."""

    def __init__(self, alpha_list: List[float], delta: float, epsilon_max: float):
        self.alpha_list = [float(a) for a in alpha_list]
        self.delta = delta
        self.epsilon_max = epsilon_max

    # -- Accumulation --

    def step(self, node, neighbor_id: int, clip_bound: float, sigma_sq: float,
             sampling_rate: float = 1.0):
        """Accumulate per-alpha RDP cost for one edge via Opacus SGM.

        Under Poisson subsampling, call this for every honest edge every round
        — the amplified cost reflects adversary uncertainty about the sampling
        coin flip, not conditional on actual participation.
        """
        if sampling_rate <= 0.0:
            return
        nm = self._noise_multiplier(clip_bound, sigma_sq)
        rdp_vec = compute_rdp(
            q=float(sampling_rate),
            noise_multiplier=nm,
            steps=1,
            orders=self.alpha_list,
        )
        for i, cost in enumerate(rdp_vec):
            node.per_neighbor_rdp[neighbor_id][i] += float(cost)

    # -- Query --

    def can_send(self, node, neighbor_id: int) -> bool:
        """Check if accumulated budget allows sending."""
        return self.get_epsilon(node, neighbor_id) <= self.epsilon_max

    def get_epsilon(self, node, neighbor_id: int) -> float:
        """Get accumulated (eps, delta)-DP for one edge."""
        eps, _ = get_privacy_spent(
            orders=self.alpha_list,
            rdp=node.per_neighbor_rdp[neighbor_id],
            delta=self.delta,
        )
        return float(eps)

    def get_step_epsilon(self, clip_bound: float, sigma_sq: float,
                         sampling_rate: float = 1.0) -> float:
        """Per-step epsilon for the bounded noise formula (NOT accumulated).

        Default sampling_rate=1.0 keeps the D2B-DP bounded-noise bound
        computation unchanged (per-step DP level used for clamp b). Pass
        sampling_rate=q to get amplified per-step ε if needed.
        """
        if sampling_rate <= 0.0:
            return 0.0
        nm = self._noise_multiplier(clip_bound, sigma_sq)
        rdp_vec = compute_rdp(
            q=float(sampling_rate),
            noise_multiplier=nm,
            steps=1,
            orders=self.alpha_list,
        )
        eps, _ = get_privacy_spent(
            orders=self.alpha_list,
            rdp=list(rdp_vec),
            delta=self.delta,
        )
        return float(eps)

    # -- Helpers --

    @staticmethod
    def _noise_multiplier(clip_bound: float, sigma_sq: float) -> float:
        """Convert (C, σ²) to Opacus unit-sensitivity noise_multiplier = σ/C.

        Guards near-zero values to prevent NaN/Inf propagation.
        """
        sigma = math.sqrt(max(float(sigma_sq), 1e-12))
        nm = sigma / max(float(clip_bound), 1e-12)
        return max(nm, 0.01)
