"""AdaptiveNoiseNode: per-node state for adaptive sigma + RDP accountant.

Extends core Node with:
  - sigma_n:         current noise std (adaptive per round)
  - loss_ema_prev:   EMA of local loss from previous round (None at round 0)
  - last_loss:       most recent local loss value (for logging/printing)
  - rdp_per_alpha:   RDP accumulator per alpha (set by PerNodeRDPAccountant)
  - frozen:          True once cumulative epsilon exceeds epsilon_max
  - val_data:        held-out validation Subset for adaptive-ratio loss eval
                     (Issue 2 fix — populated by simulator.setup())
"""

from typing import Optional

from torch.utils.data import Subset

from dpfl.core.base_node import Node
from dpfl.models.base_model import BaseModel


class AdaptiveNoiseNode(Node):
    """Node variant carrying Loss-based adaptive-noise DP state."""

    def __init__(self, node_id: int, model: BaseModel, data: Subset,
                 is_attacker: bool, sigma_0: float):
        super().__init__(node_id, model, data, is_attacker)
        self.sigma_n: float = sigma_0
        self.loss_ema_prev: Optional[float] = None
        self.last_loss: float = 0.0
        self.rdp_per_alpha: list = []   # initialized by accountant.init_node_state()
        self.frozen: bool = False
        self.val_data: Optional[Subset] = None   # set by simulator.setup() for honest nodes
