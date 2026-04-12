"""CIFAR-10 dataset with IID and Dirichlet non-IID split."""

import logging

import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from typing import Dict, Tuple

from dpfl.data.base_dataset import BaseDataset
from dpfl.registry import register, DATASETS

logger = logging.getLogger(__name__)


@register(DATASETS, "cifar10")
class CIFAR10Dataset(BaseDataset):

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (3, 32, 32)

    @property
    def num_classes(self) -> int:
        return 10

    def load(self) -> Tuple[Dataset, Dataset]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ])
        logger.info("Loading CIFAR-10 dataset...")
        train = datasets.CIFAR10(
            ".cache/cifar10", train=True, download=True, transform=transform
        )
        test = datasets.CIFAR10(
            ".cache/cifar10", train=False, download=True, transform=transform
        )
        logger.info("CIFAR-10 loaded: %d train, %d test samples", len(train), len(test))
        return train, test

    def split(self, dataset: Dataset, n_nodes: int,
              mode: str = "iid", alpha: float = 0.5) -> Dict[int, Subset]:
        if mode == "iid":
            return self._split_iid(dataset, n_nodes)
        elif mode == "dirichlet":
            return self._split_dirichlet(dataset, n_nodes, alpha)
        else:
            raise ValueError(f"Unknown split mode: {mode}")

    def _split_iid(self, dataset: Dataset, n_nodes: int) -> Dict[int, Subset]:
        indices = np.random.permutation(len(dataset))
        chunks = np.array_split(indices, n_nodes)
        return {i: Subset(dataset, chunk.tolist()) for i, chunk in enumerate(chunks)}

    def _split_dirichlet(self, dataset: Dataset, n_nodes: int,
                          alpha: float) -> Dict[int, Subset]:
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        n_classes = self.num_classes
        node_indices = {i: [] for i in range(n_nodes)}

        for c in range(n_classes):
            class_idx = np.where(targets == c)[0]
            np.random.shuffle(class_idx)
            proportions = np.random.dirichlet([alpha] * n_nodes)
            splits = (np.cumsum(proportions) * len(class_idx)).astype(int)
            splits = np.clip(splits, 0, len(class_idx))
            parts = np.split(class_idx, splits[:-1])
            for i, part in enumerate(parts):
                node_indices[i].extend(part.tolist())

        return {i: Subset(dataset, idx) for i, idx in node_indices.items()}
