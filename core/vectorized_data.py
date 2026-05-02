"""GPU-resident batched data pipeline for vectorized FL training/eval.

Loads the full underlying dataset to device once at setup, then per round
yields batched slices of shape (N, B, *input_shape) plus a (N, B) bool mask
that flags real (vs padded) samples for variable per-client sizes.

Padding strategy: clients with n_i < max_n_samples pad with their own sample
indices (no out-of-range), but mask is False for positions >= n_i so caller
zeros out their loss / gradient contribution.

Determinism contract: per-client shuffles drawn sequentially from data_gen
(client 0's permutation drawn first, then client 1, ...). Same seed yields
same sequence of permutations.
"""

from typing import Callable, Dict, Iterator, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset


class VectorizedDataPipeline:
    """Pre-stacked client data on device. One instance per (train, test) split.

    Optional data-poisoning support: if `attacker_mask` and `flip_fn` are set,
    `iter_train_batches` applies `flip_fn` to attacker-row labels per batch,
    mirroring `LabelFlipDataset.__getitem__` semantics for the vectorized
    path. Set `attack_active=False` to skip flipping (e.g., before
    `attack.start_round`).
    """

    def __init__(self, dataset: Dataset, node_data: Dict[int, Subset],
                 device: torch.device, data_gen: torch.Generator,
                 attacker_mask: Optional[torch.Tensor] = None,
                 flip_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.device = device
        self.data_gen = data_gen

        underlying = self._resolve_underlying(node_data)
        self.X_full, self.Y_full = self._materialize(underlying, device)
        self.input_shape = self.X_full.shape[1:]

        self.node_ids = sorted(node_data.keys())
        self.client_indices = [
            torch.tensor(node_data[nid].indices, dtype=torch.long, device=device)
            for nid in self.node_ids
        ]
        self.client_sizes = torch.tensor(
            [len(idx) for idx in self.client_indices],
            dtype=torch.long, device=device,
        )
        self.max_n_samples = int(self.client_sizes.max().item())
        self.N = len(self.node_ids)

        # Data-poisoning hooks (None = no poisoning, fast path unchanged).
        # attacker_mask: (N,) bool — True at attacker rows. flip_fn: callable
        # taking a 1-D tensor of labels and returning flipped labels (same
        # shape, same device, same dtype). attack_active toggled per round
        # by simulator to honor attack.start_round.
        self.attacker_mask = attacker_mask
        self.flip_fn = flip_fn
        self.attack_active = True

    @staticmethod
    def _resolve_underlying(node_data: Dict[int, Subset]) -> Dataset:
        """All Subsets share the same underlying Dataset; grab the first."""
        first = next(iter(node_data.values()))
        return first.dataset

    @staticmethod
    def _materialize(dataset: Dataset, device: torch.device
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Iterate dataset once via DataLoader, stack into (M, *) and (M,) on device."""
        loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
        xs, ys = [], []
        for x, y in loader:
            xs.append(x)
            ys.append(y)
        X = torch.cat(xs, dim=0).to(device)
        Y = torch.cat(ys, dim=0).to(torch.long).to(device)
        return X, Y

    def iter_train_batches(self, batch_size: int
                           ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield (x_batch, y_batch, mask) per batch.

        x_batch: (N, B, *input_shape) — gathered from X_full
        y_batch: (N, B)
        mask:    (N, B) bool — True where real, False where padded

        n_batches per epoch = max_n_samples // batch_size. Clients with smaller
        n_i contribute fewer real samples (later batches mostly masked out).
        """
        if self.max_n_samples < batch_size:
            return

        shuffled_idx = self._build_shuffled_indices()
        n_batches = self.max_n_samples // batch_size

        for b in range(n_batches):
            start, end = b * batch_size, (b + 1) * batch_size
            batch_idx = shuffled_idx[:, start:end]

            x_batch = self.X_full[batch_idx]
            y_batch = self.Y_full[batch_idx]

            # Data-poisoning: flip labels on attacker rows in-place per batch.
            # Mirrors LabelFlipDataset.__getitem__ for the vectorized path.
            # Inactive when flip_fn/mask unset (non-label_flip configs) or
            # when simulator toggled attack_active=False (pre-start_round).
            if (self.attack_active and self.flip_fn is not None
                    and self.attacker_mask is not None):
                y_batch = y_batch.clone()
                y_batch[self.attacker_mask] = self.flip_fn(
                    y_batch[self.attacker_mask])

            positions = torch.arange(start, end, device=self.device).unsqueeze(0)
            mask = positions < self.client_sizes.unsqueeze(1)

            yield x_batch, y_batch, mask

    def _build_shuffled_indices(self) -> torch.Tensor:
        """Per-client randperm into client_indices, padded with first index.

        data_gen is on CPU (DataLoader-style); we draw on CPU then move.
        """
        gen_device = self.data_gen.device
        shuffled = torch.zeros(
            (self.N, self.max_n_samples), dtype=torch.long, device=self.device,
        )
        for i in range(self.N):
            n_i = int(self.client_sizes[i].item())
            perm = torch.randperm(n_i, generator=self.data_gen, device=gen_device)
            if perm.device != self.device:
                perm = perm.to(self.device)
            shuffled[i, :n_i] = self.client_indices[i][perm]
            if n_i < self.max_n_samples:
                shuffled[i, n_i:] = self.client_indices[i][0]
        return shuffled

    def iter_test_batches(self, batch_size: int = 256
                          ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Plain (x, y) batches over the FULL materialized dataset (no per-client).

        For test pipeline: caller pre-stacks test_ds and uses this iterator
        for vectorized eval (single test set, applied to all N client models).
        """
        n = self.X_full.shape[0]
        for s in range(0, n, batch_size):
            yield self.X_full[s:s + batch_size], self.Y_full[s:s + batch_size]
