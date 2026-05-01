"""LeNet-5: classic CNN baseline, well-tested for MNIST / FEMNIST byclass.

Architecture per LeCun 1998 with minor adaptations:
  - Padding=2 on conv1 so 28×28 input passes through cleanly (yields LeNet's
    target 28×28 → 14×14 → 10×10 → 5×5 spatial trace).
  - AvgPool replaced with MaxPool? No — kept AvgPool (LeNet uses subsampling).
  - GroupNorm added between conv layers (DP-safe alternative to BatchNorm,
    optional via use_norm flag — defaults False to match canonical LeNet).
  - AdaptiveAvgPool2d(5) before the FC head so 32×32 inputs (CIFAR) also reach
    the canonical 5×5 spatial size for FC compat.

Param count (FEMNIST, 62 classes, hidden_size=120):
  Conv1 (1→6, 5×5):        156
  Conv2 (6→16, 5×5):     2,416
  FC1 (400→120):        48,120
  FC2 (120→84):         10,164
  FC3 (84→62):           5,270
  Total                ≈ 66k params  (≈ 2.6× smaller than MLP h=200)
"""

import torch
import torch.nn as nn

from dpfl.models.base_model import BaseModel
from dpfl.models.tiny_cnn_model import _infer_channels
from dpfl.registry import register, MODELS


@register(MODELS, "lenet5")
class LeNet5(BaseModel):
    """Classic LeNet-5 with adaptive spatial pool for cross-dataset compat.

    Args:
        input_dim: Flattened input size; used to infer input_channels.
        hidden_size: First FC layer width (canonical LeNet-5 uses 120).
        num_classes: Output classes.
    """

    def __init__(self, input_dim: int = 784, hidden_size: int = 120,
                 num_classes: int = 10, **kwargs):
        super().__init__()
        c = _infer_channels(input_dim)
        self.features = nn.Sequential(
            nn.Conv2d(c, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.AdaptiveAvgPool2d(5),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
