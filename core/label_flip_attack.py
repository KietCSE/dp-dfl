"""Label-flip attack: data poisoning by rotating training labels.

Unlike model poisoning, this wraps the attacker's dataset so training
produces a poisoned gradient naturally. perturb() is a no-op.
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, Dict

from dpfl.core.base_attack import BaseAttack
from dpfl.registry import register, ATTACKS


class LabelFlipDataset(Dataset):
    """Wrapper that flips labels on-the-fly."""

    def __init__(self, dataset: Dataset, num_classes: int = 10,
                 flip_mode: str = "rotate"):
        self.dataset = dataset
        self.num_classes = num_classes
        self.flip_mode = flip_mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.flip_mode == "rotate":
            y = (y + 1) % self.num_classes
        elif self.flip_mode == "random":
            y = torch.randint(0, self.num_classes, (1,)).item()
        elif self.flip_mode == "negate":
            y = (self.num_classes - 1 - y) % self.num_classes
        return x, y


@register(ATTACKS, "label_flip")
class LabelFlipAttack(BaseAttack):
    """Data poisoning: flip training labels before local training.

    perturb() is a no-op — poisoning happens via wrapped dataset.
    """

    def __init__(self, num_classes: int = 10,
                 flip_mode: str = "rotate", **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.flip_mode = flip_mode

    def wrap_dataset(self, dataset: Dataset) -> Dataset:
        """Wrap dataset with label flipping. Called during node setup."""
        return LabelFlipDataset(dataset, self.num_classes, self.flip_mode)

    def perturb(self, honest_update: torch.Tensor,
                context: Optional[Dict] = None) -> torch.Tensor:
        # No-op: poisoning already happened via data wrapper
        return honest_update
