"""Abstract base class for datasets in DP-DFL."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
from torch.utils.data import Dataset, Subset


class BaseDataset(ABC):
    @abstractmethod
    def load(self) -> Tuple[Dataset, Dataset]:
        """Return (train_dataset, test_dataset)."""
        ...

    @abstractmethod
    def split(self, dataset: Dataset, n_nodes: int, mode: str = "iid",
              alpha: float = 0.5) -> Dict[int, Subset]:
        """Split train dataset across n_nodes. Returns {node_id: Subset}."""
        ...

    @property
    @abstractmethod
    def input_shape(self) -> Tuple[int, ...]:
        """e.g. (1, 28, 28) for MNIST."""
        ...

    @property
    @abstractmethod
    def num_classes(self) -> int:
        ...
