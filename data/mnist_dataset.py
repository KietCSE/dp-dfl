"""MNIST dataset with IID and Dirichlet non-IID split."""

import logging

import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from typing import Dict, Tuple

from dpfl.data.base_dataset import BaseDataset
from dpfl.registry import register, DATASETS

logger = logging.getLogger(__name__)


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
        logger.info("Loading MNIST dataset...")
        train_ds = datasets.MNIST(".cache/mnist", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(".cache/mnist", train=False, download=True, transform=transform)
        logger.info("MNIST loaded: %d train, %d test samples", len(train_ds), len(test_ds))
        return train_ds, test_ds

    def split(self, dataset: Dataset, n_nodes: int, mode: str = "iid",
              alpha: float = 0.5, samples_per_node: int = None
              ) -> Dict[int, Subset]:
        if mode == "iid":
            return self._split_iid(dataset, n_nodes, samples_per_node)
        elif mode == "dirichlet":
            return self._split_dirichlet(dataset, n_nodes, alpha)
        else:
            raise ValueError(f"Unknown split mode: {mode}")

    def _split_iid(self, dataset: Dataset, n_nodes: int,
                    samples_per_node: int = None) -> Dict[int, Subset]:
        """IID split across nodes.

        samples_per_node=None → disjoint partition (size len(dataset)//n_nodes).
        samples_per_node=int  → each node randomly draws this many samples from
        the full pool (overlap allowed; each node still IID to dataset).
        """
        n_total = len(dataset)
        if samples_per_node is None:
            # Disjoint partition (default, backward compatible).
            indices = torch.randperm(n_total).tolist()
            chunk_size = n_total // n_nodes
            return {
                i: Subset(dataset, indices[i * chunk_size:(i + 1) * chunk_size])
                for i in range(n_nodes)
            }
        # Overlap mode: each node draws its own random subset from full pool.
        k = min(samples_per_node, n_total)
        return {
            i: Subset(dataset, torch.randperm(n_total)[:k].tolist())
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
