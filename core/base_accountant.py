"""ABC for privacy accountants (DP budget tracking)."""

from abc import ABC, abstractmethod
from typing import Optional


class BaseAccountant(ABC):
    @abstractmethod
    def step(self, n_steps: int, sampling_rate: float, noise_mult: float):
        """Accumulate privacy cost for n_steps of DP-SGD."""
        ...

    @abstractmethod
    def get_epsilon(self) -> float:
        """Return current cumulative epsilon."""
        ...

    def get_best_alpha(self) -> Optional[float]:
        """Return best Renyi order (RDP-specific). None for non-RDP accountants."""
        return None
