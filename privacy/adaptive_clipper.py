"""Adaptive L2-norm clipper with FIFO history for dynamic threshold."""

import torch
from collections import deque
from typing import Optional


class AdaptiveClipper:
    """Stateless clipper — state (clip_history deque) lives in TrustAwareNode."""

    def __init__(self, clip_window: int = 3):
        self.clip_window = clip_window

    @staticmethod
    def get_threshold(history: deque) -> Optional[float]:
        """Mean of raw L2 norms in history. None if empty (cold start)."""
        if not history:
            return None
        return sum(history) / len(history)

    @staticmethod
    def clip(gradient: torch.Tensor, threshold: float) -> torch.Tensor:
        """L2-norm clip: g' = g * min(1, threshold / ||g||_2)."""
        norm = gradient.norm(2)
        factor = min(1.0, threshold / (norm.item() + 1e-12))
        return gradient * factor
