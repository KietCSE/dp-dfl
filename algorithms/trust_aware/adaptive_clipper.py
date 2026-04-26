"""Layer-wise adaptive L2-norm clipper for Trust-Aware D2B-DP.

Implements Step 2 of docs/Trust-Aware-D2B-DP.md:
    Clip_l = (1/k) · Σ H_l[m]            (mean of last-k L2 norms per layer)
    ΔW'_l = ΔW_l / max(1, ||ΔW_l||_2 / Clip_l)

State (per-layer FIFO deques) lives on TrustAwareNode; this class is stateless.
"""

from collections import deque
from typing import List, Optional, Sequence

import torch


class LayerwiseAdaptiveClipper:
    """Per-layer L2-norm clipping with FIFO history for dynamic threshold."""

    def __init__(self, k: int = 5):
        self.k = k

    @staticmethod
    def get_thresholds(history: Sequence[deque]) -> List[Optional[float]]:
        """Return per-layer Clip_l = mean(H_l). None when no history yet."""
        out: List[Optional[float]] = []
        for h in history:
            out.append(sum(h) / len(h) if h else None)
        return out

    @staticmethod
    def clip(layers: Sequence[torch.Tensor],
             thresholds: Sequence[Optional[float]]) -> List[torch.Tensor]:
        """Clip each layer to its per-layer threshold; pass through if None."""
        clipped: List[torch.Tensor] = []
        for layer, thr in zip(layers, thresholds):
            if thr is None:
                clipped.append(layer.clone())
                continue
            norm = layer.norm(2).item()
            factor = min(1.0, thr / (norm + 1e-12))
            clipped.append(layer * factor)
        return clipped
