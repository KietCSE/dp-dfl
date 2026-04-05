"""ABC for aggregation with built-in defense (filter + aggregate)."""

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class AggregationResult:
    new_params: torch.Tensor
    clean_ids: List[int] = field(default_factory=list)
    flagged_ids: List[int] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class BaseAggregator(ABC):
    @abstractmethod
    def aggregate(
        self,
        own_update: torch.Tensor,
        own_params: torch.Tensor,
        neighbor_updates: Dict[int, torch.Tensor],
    ) -> AggregationResult:
        """Filter + aggregate neighbor updates in one step."""
        ...
