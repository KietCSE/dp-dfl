"""MNIST dataset with IID and Dirichlet non-IID split."""

import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from typing import Dict, Tuple

from dpfl.data.base_dataset import BaseDataset
from dpfl.registry import register, DATASETS


@register(DATASETS, "mnist")
class MNISTDataset(BaseDataset):

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (1, 28, 28)

    @property
    def num_classes(self) -> int:
        return 10

    def load(self) -> Tuple[Dataset, Dataset]:
        """Download MNIST, return (train, test) datasets."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(".cache/mnist", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(".cache/mnist", train=False, download=True, transform=transform)
        return train_ds, test_ds

    def split(self, dataset: Dataset, n_nodes: int, mode: str = "iid",
              alpha: float = 0.5) -> Dict[int, Subset]:
        if mode == "iid":
            return self._split_iid(dataset, n_nodes)
        elif mode == "dirichlet":
            return self._split_dirichlet(dataset, n_nodes, alpha)
        else:
            raise ValueError(f"Unknown split mode: {mode}")

    def _split_iid(self, dataset: Dataset, n_nodes: int) -> Dict[int, Subset]:
        """Random shuffle then equal-size partitions."""
        indices = torch.randperm(len(dataset)).tolist()
        chunk_size = len(dataset) // n_nodes
        return {
            i: Subset(dataset, indices[i * chunk_size:(i + 1) * chunk_size])
            for i in range(n_nodes)
        }

    def _split_dirichlet(self, dataset: Dataset, n_nodes: int,
                          alpha: float) -> Dict[int, Subset]:
        """Dirichlet(alpha) non-IID split: group by label, sample proportions."""
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        n_classes = len(np.unique(targets))
        node_indices = {i: [] for i in range(n_nodes)}

        for c in range(n_classes):
            class_idx = np.where(targets == c)[0]
            np.random.shuffle(class_idx)
            # Sample proportions from Dirichlet
            proportions = np.random.dirichlet([alpha] * n_nodes)
            # Convert proportions to cumulative split points
            splits = (np.cumsum(proportions) * len(class_idx)).astype(int)
            splits = np.clip(splits, 0, len(class_idx))
            parts = np.split(class_idx, splits[:-1])
            for i, part in enumerate(parts):
                node_indices[i].extend(part.tolist())

        return {i: Subset(dataset, idxs) for i, idxs in node_indices.items()}
