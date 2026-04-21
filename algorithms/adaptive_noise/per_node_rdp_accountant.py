"""Per-node RDP accountant for Adaptive Noise DP DFL.

State lives on each node (node.rdp_per_alpha). Sensitivity is user-level C
(same for all nodes), noise std sigma is adaptive per round per node.

Per-step RDP cost (Gaussian mechanism, user-level):
    eps_step(alpha) = alpha * C^2 / (2 * sigma^2)

Composition over rounds is additive. Convert to (eps, delta)-DP via:
    eps_n = min_{alpha > 1} { rdp_n(alpha) + ln(1/delta) / (alpha - 1) }
"""

import math
from typing import List


class PerNodeRDPAccountant:
    """Tracks per-node RDP cost. Shared instance across nodes; state on node."""

    def __init__(self, alpha_list: List[float], delta: float, epsilon_max: float):
        self.alpha_list = list(alpha_list)
        self.delta = delta
        self.epsilon_max = epsilon_max

    def init_node_state(self, node):
        """Attach zero-initialized RDP accumulator to a node."""
        node.rdp_per_alpha = [0.0] * len(self.alpha_list)

    def compute_step_cost(self, alpha: float, clip_bound: float,
                          sigma: float) -> float:
        """Per-step RDP cost at order alpha: alpha * C^2 / (2 * sigma^2)."""
        return alpha * clip_bound ** 2 / (2.0 * sigma ** 2 + 1e-12)

    def step(self, node, clip_bound: float, sigma: float,
             sampling_rate: float = 1.0):
        """Accumulate one round of RDP cost into node.rdp_per_alpha.

        sampling_rate q: Poisson subsampling rate (1.0 = no subsampling).
        For q < 1.0 applies approximation RDP_sub ≈ q² · RDP_full (valid for
        small q, moderate α; Mironov 2019). Every round, every honest node
        accumulates amplified cost regardless of its actual sampled-or-not
        outcome — the amplification comes from the adversary's uncertainty
        about which clients were sampled, not from conditional skipping.
        """
        amp = sampling_rate ** 2 if sampling_rate < 1.0 else 1.0
        for i, alpha in enumerate(self.alpha_list):
            full_cost = self.compute_step_cost(alpha, clip_bound, sigma)
            node.rdp_per_alpha[i] += amp * full_cost

    def _rdp_to_eps(self, rdp_array: List[float]) -> float:
        """Convert per-alpha RDP to (eps, delta)-DP."""
        log_term = math.log(1.0 / self.delta)
        return min(
            rdp_array[i] + log_term / (alpha - 1.0)
            for i, alpha in enumerate(self.alpha_list)
        )

    def get_epsilon(self, node) -> float:
        """Current (eps, delta)-DP cost for this node."""
        return self._rdp_to_eps(node.rdp_per_alpha)

    def get_best_alpha(self, node) -> float:
        """Alpha giving the tightest eps bound for this node."""
        log_term = math.log(1.0 / self.delta)
        return min(
            self.alpha_list,
            key=lambda a: (
                node.rdp_per_alpha[self.alpha_list.index(a)]
                + log_term / (a - 1.0)
            ),
        )

    def exceeds_budget(self, node) -> bool:
        """True if cumulative epsilon for node exceeds epsilon_max."""
        return self.get_epsilon(node) > self.epsilon_max
