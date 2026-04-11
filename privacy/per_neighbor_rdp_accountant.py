"""Per-neighbor RDP accountant: tracks per-edge privacy budget for D2B-DP."""

import math
from typing import List


class PerNeighborRDPAccountant:
    """Tracks per-edge RDP cost. State lives in TrustAwareNode.per_neighbor_rdp."""

    def __init__(self, alpha_list: List[float], delta: float, epsilon_max: float):
        self.alpha_list = alpha_list
        self.delta = delta
        self.epsilon_max = epsilon_max

    def compute_step_cost(self, alpha: float, clip_bound: float,
                          sigma_sq: float) -> float:
        """Per-step RDP cost: alpha * C^2 / (2 * sigma^2)."""
        return alpha * clip_bound ** 2 / (2.0 * sigma_sq + 1e-12)

    def step(self, node, neighbor_id: int, clip_bound: float, sigma_sq: float):
        """Accumulate per-alpha RDP cost for one edge."""
        for i, alpha in enumerate(self.alpha_list):
            cost = self.compute_step_cost(alpha, clip_bound, sigma_sq)
            node.per_neighbor_rdp[neighbor_id][i] += cost

    def _rdp_to_eps(self, rdp_array: List[float]) -> float:
        """Convert per-alpha RDP to (eps, delta)-DP: min_a{rdp[a] + log(1/d)/(a-1)}."""
        return min(
            rdp_array[i] + math.log(1.0 / self.delta) / (alpha - 1.0)
            for i, alpha in enumerate(self.alpha_list)
        )

    def can_send(self, node, neighbor_id: int) -> bool:
        """Check if accumulated budget allows sending."""
        eps = self._rdp_to_eps(node.per_neighbor_rdp[neighbor_id])
        return eps <= self.epsilon_max

    def get_epsilon(self, node, neighbor_id: int) -> float:
        """Get accumulated (eps, delta)-DP for one edge."""
        return self._rdp_to_eps(node.per_neighbor_rdp[neighbor_id])

    def get_step_epsilon(self, clip_bound: float, sigma_sq: float) -> float:
        """Per-step epsilon for bounded noise formula (NOT accumulated)."""
        return min(
            self.compute_step_cost(a, clip_bound, sigma_sq)
            + math.log(1.0 / self.delta) / (a - 1.0)
            for a in self.alpha_list
        )
