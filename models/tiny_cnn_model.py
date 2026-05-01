"""TinyCNN: compact 2-conv + GAP + FC for DP-FL on small-image datasets.

Designed for low-D regimes where DP noise variance ∝ σ²·D is the bottleneck.
GroupNorm is DP-safe (no per-batch running stats vs BatchNorm).
AdaptiveAvgPool2d(1) makes the model spatial-size agnostic — auto-fits 28×28
(MNIST / FEMNIST) and 32×32 (CIFAR) without code changes.

Param count (FEMNIST, 62 classes, hidden_size=32):
  Conv1 (1→16, 3×3):     160
  GroupNorm(4,16):        32
  Conv2 (16→32, 3×3):  4,640
  GroupNorm(8,32):        64
  FC1 (32→32):         1,056
  FC2 (32→62):         2,046
  Total                ≈ 8k params  (≈ 21× smaller than MLP h=200)
"""

import torch
import torch.nn as nn

from dpfl.models.base_model import BaseModel
from dpfl.registry import register, MODELS


def _infer_channels(input_dim: int) -> int:
    """Recover input_channels from input_dim assuming square spatial dims.

    Tries common spatial sizes in order; returns 1 as fallback.
    """
    for spatial in (28, 32, 64, 96, 224):
        if input_dim % (spatial * spatial) == 0:
            return input_dim // (spatial * spatial)
    return 1


@register(MODELS, "tiny_cnn")
class TinyCNN(BaseModel):
    """Compact CNN: 2 conv (16→32 ch) + GroupNorm + GAP + 2-layer FC head.

    Args:
        input_dim: Flattened input size (e.g. 784 for 1×28×28). Used to infer
            input_channels via `_infer_channels`. Spatial size is irrelevant
            because of AdaptiveAvgPool2d(1).
        hidden_size: FC bottleneck after GAP. Keep small (≤64) for DP safety.
        num_classes: Output classes.
    """

    def __init__(self, input_dim: int = 784, hidden_size: int = 32,
                 num_classes: int = 10, **kwargs):
        super().__init__()
        c = _infer_channels(input_dim)
        self.features = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
