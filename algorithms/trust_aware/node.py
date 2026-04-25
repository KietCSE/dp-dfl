"""Trust-Aware D2B-DP node: extends Node with per-layer + per-neighbor state.

State carried across rounds:
  clip_history    : per-layer FIFO of L2 norms (Step 2 — adaptive Clip_l)
  trust_scores    : per-neighbor T_{i,j} EMA buffer (Step 6)
  V_agg           : flat momentum buffer for the global step (Step 7)

compute_update() inherits from Node — call with apply_noise=False since the
D2B layer-wise bounded Gaussian noise is injected by the simulator (Step 3).
"""

from collections import deque
from typing import Dict, List, Sequence

import torch

from dpfl.core.base_node import Node


class TrustAwareNode(Node):
    """Node with per-layer clip history, per-neighbor trust, and momentum buffer."""

    def __init__(self, node_id, model, data, is_attacker, trust_config):
        super().__init__(node_id, model, data, is_attacker)
        self.tc = trust_config
        # Filled lazily in init_d2b_state once param shapes are known.
        self.clip_history: List[deque] = []
        self.trust_scores: Dict[int, float] = {}
        self.V_agg: torch.Tensor = torch.empty(0)

    def init_d2b_state(self, layer_sizes: Sequence[int], param_dim: int,
                       k: int, trust_init: float,
                       device: torch.device = None):
        """Initialize per-layer FIFOs, per-neighbor trust, and momentum buffer.

        Called from the simulator's setup() once the model layout is known.
        """
        self.clip_history = [deque(maxlen=k) for _ in layer_sizes]
        self.trust_scores = {j: float(trust_init) for j in self.neighbors}
        dev = device if device is not None else self.model.get_flat_params().device
        self.V_agg = torch.zeros(param_dim, device=dev)
