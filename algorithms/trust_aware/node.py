"""Trust-aware FL node: extends Node with per-neighbor trust state and buffers."""

from collections import deque
from typing import Dict, List, Set

from dpfl.core.base_node import Node


class TrustAwareNode(Node):
    """Node with trust scores, clip history, and temporal buffers for D2B-DP.

    compute_update() inherited from Node — call with apply_noise=False
    since per-edge noise is added in Phase 2 of TrustAwareDFLSimulator.
    """

    def __init__(self, node_id, model, data, is_attacker, trust_config):
        super().__init__(node_id, model, data, is_attacker)
        self.trust_scores: Dict[int, float] = {}
        self.clip_history: deque = deque(maxlen=trust_config.clip_window)
        self.z_score_buffer: deque = deque(maxlen=trust_config.temporal_window)
        self.trust_buffer: deque = deque(maxlen=trust_config.temporal_window)
        self.per_neighbor_rdp: Dict[int, List[float]] = {}

    def init_trust(self, neighbors: Set[int], trust_init: float,
                   alpha_list: List[float]):
        """Initialize per-neighbor trust scores and RDP arrays."""
        for j in neighbors:
            self.trust_scores[j] = trust_init
            self.per_neighbor_rdp[j] = [0.0] * len(alpha_list)
