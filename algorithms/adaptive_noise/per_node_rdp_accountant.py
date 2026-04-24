"""Per-node RDP accountant for Adaptive Noise DP DFL.

Backed by Opacus (Meta AI) SGM implementation (Mironov 2019) — tight RDP
bounds for subsampled Gaussian. State lives on each node (`node.rdp_per_alpha`).
Sensitivity is user-level C (same for all nodes), noise std sigma is adaptive
per round per node → Opacus called with `steps=1` each round and results are
accumulated additively.
"""

from typing import List

from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent


class PerNodeRDPAccountant:
    """Tracks per-node RDP cost. Shared instance across nodes; state on node."""

    def __init__(self, alpha_list: List[float], delta: float, epsilon_max: float):
        self.alpha_list = [float(a) for a in alpha_list]
        self.delta = delta
        self.epsilon_max = epsilon_max

    def init_node_state(self, node):
        """Attach zero-initialized RDP accumulator to a node."""
        node.rdp_per_alpha = [0.0] * len(self.alpha_list)

    def step(self, node, clip_bound: float, sigma: float,
             sampling_rate: float = 1.0):
        """Accumulate one round of RDP cost into node.rdp_per_alpha via Opacus.

        sampling_rate q: Poisson subsampling rate (1.0 = no subsampling). Every
        round, every honest node accumulates amplified cost regardless of its
        actual sampled-or-not outcome — the amplification comes from adversary
        uncertainty about the sampling coin flip.
        """
        if sampling_rate <= 0.0:
            return  # no participation, no cost
        # noise_multiplier = σ / C (unit sensitivity), guard near-zero values.
        nm = max(float(sigma) / max(float(clip_bound), 1e-12), 0.01)
        rdp_vec = compute_rdp(
            q=float(sampling_rate),
            noise_multiplier=nm,
            steps=1,
            orders=self.alpha_list,
        )
        for i, cost in enumerate(rdp_vec):
            node.rdp_per_alpha[i] += float(cost)

    def get_epsilon(self, node) -> float:
        """Current (eps, delta)-DP cost for this node."""
        eps, _ = get_privacy_spent(
            orders=self.alpha_list,
            rdp=node.rdp_per_alpha,
            delta=self.delta,
        )
        return float(eps)

    def get_best_alpha(self, node) -> float:
        """Alpha giving the tightest eps bound for this node."""
        _, best_alpha = get_privacy_spent(
            orders=self.alpha_list,
            rdp=node.rdp_per_alpha,
            delta=self.delta,
        )
        return float(best_alpha)

    def exceeds_budget(self, node) -> bool:
        """True if cumulative epsilon for node exceeds epsilon_max."""
        return self.get_epsilon(node) > self.epsilon_max
