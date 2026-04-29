"""FEMNIST (EMNIST byclass) dataset with IID and Dirichlet non-IID split.

EMNIST byclass: 62 classes (10 digits + 26 lowercase + 26 uppercase letters),
28x28 grayscale. Total ~697k train + ~116k test samples.

Note: torchvision EMNIST byclass images are stored rotated 90° + horizontally
flipped vs natural orientation. The Lambda transform corrects this before
ToTensor conversion.
"""

import logging

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from typing import Dict, Tuple

from dpfl.data.base_dataset import BaseDataset
from dpfl.registry import register, DATASETS

logger = logging.getLogger(__name__)


@register(DATASETS, "femnist")
class FEMNISTDataset(BaseDataset):
    """EMNIST byclass: 62 classes, 28x28 grayscale."""

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (1, 28, 28)

    @property
    def num_classes(self) -> int:
        return 62

    def load(self) -> Tuple[Dataset, Dataset]:
        # Correct EMNIST byclass orientation: flip horizontally then rotate 90°.
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90)),
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3317,)),
        ])
        logger.info("Loading FEMNIST (EMNIST byclass) dataset...")
        train = datasets.EMNIST(
            ".cache/femnist", split="byclass", train=True,
            download=True, transform=transform,
        )
        test = datasets.EMNIST(
            ".cache/femnist", split="byclass", train=False,
            download=True, transform=transform,
        )
        logger.info("FEMNIST loaded: %d train, %d test samples",
                    len(train), len(test))
        return train, test

    def split(self, dataset: Dataset, n_nodes: int,
              mode: str = "iid", alpha: float = 0.5,
              samples_per_node: int = None) -> Dict[int, Subset]:
        if mode == "iid":
            return self._split_iid(dataset, n_nodes, samples_per_node)
        elif mode == "dirichlet":
            return self._split_dirichlet(dataset, n_nodes, alpha)
        else:
            raise ValueError(f"Unknown split mode: {mode}")

    def _split_iid(self, dataset: Dataset, n_nodes: int,
                   samples_per_node: int = None) -> Dict[int, Subset]:
        """IID split. Disjoint (default) or overlap-sampled if samples_per_node set."""
        n_total = len(dataset)
        if samples_per_node is None:
            indices = np.random.permutation(n_total)
            chunks = np.array_split(indices, n_nodes)
            return {i: Subset(dataset, chunk.tolist()) for i, chunk in enumerate(chunks)}
        k = min(samples_per_node, n_total)
        return {
            i: Subset(dataset, np.random.permutation(n_total)[:k].tolist())
            for i in range(n_nodes)
        }

    def _split_dirichlet(self, dataset: Dataset, n_nodes: int,
                         alpha: float) -> Dict[int, Subset]:
        """Dirichlet(alpha) non-IID partition. Some classes may be empty in
        train data — guard with len-check. Warn if many nodes end up starved."""
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        n_classes = self.num_classes
        node_indices = {i: [] for i in range(n_nodes)}

        for c in range(n_classes):
            class_idx = np.where(targets == c)[0]
            if len(class_idx) == 0:
                continue
            np.random.shuffle(class_idx)
            proportions = np.random.dirichlet([alpha] * n_nodes)
            splits = (np.cumsum(proportions) * len(class_idx)).astype(int)
            splits = np.clip(splits, 0, len(class_idx))
            parts = np.split(class_idx, splits[:-1])
            for i, part in enumerate(parts):
                node_indices[i].extend(part.tolist())

        # Sanity: warn on starved nodes (frequent at large n_nodes / low alpha).
        starved = [i for i, idx in node_indices.items() if len(idx) < 10]
        if starved:
            logger.warning(
                "FEMNIST Dirichlet: %d/%d nodes have <10 samples "
                "(n_nodes=%d, alpha=%.2f). Consider lower n_nodes or higher alpha.",
                len(starved), n_nodes, n_nodes, alpha)

        return {i: Subset(dataset, idx) for i, idx in node_indices.items()}
