"""Unit tests for EMNISTDigitsDataset: split logic, properties, registration.

Tests use a mock in-memory dataset to avoid downloading EMNIST (~535MB).
The real .load() path is exercised indirectly only via integration runs.
"""

import logging

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from dpfl.data.emnist_digits_dataset import EMNISTDigitsDataset
from dpfl.registry import DATASETS


# ── Mock dataset ───────────────────────────────────────────────────────────

class MockImageDataset(Dataset):
    """In-memory (image, label) pairs. Mimics torchvision Dataset interface."""

    def __init__(self, n_samples: int, n_classes: int = 10, balanced: bool = True,
                 missing_classes: list = None):
        self.n_samples = n_samples
        self.n_classes = n_classes
        if balanced:
            base = n_samples // n_classes
            remainder = n_samples - base * n_classes
            counts = [base + (1 if i < remainder else 0) for i in range(n_classes)]
            self.targets = []
            for c, cnt in enumerate(counts):
                self.targets.extend([c] * cnt)
        else:
            rng = np.random.RandomState(0)
            self.targets = rng.randint(0, n_classes, size=n_samples).tolist()
        if missing_classes:
            self.targets = [
                t if t not in missing_classes else (t + 1) % n_classes
                for t in self.targets
            ]
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(self.targets))
        self.targets = [self.targets[i] for i in perm]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.zeros(1, 28, 28), self.targets[idx]


# ── Properties + registration ──────────────────────────────────────────────

def test_emnist_digits_registered():
    assert "emnist_digits" in DATASETS
    assert DATASETS["emnist_digits"] is EMNISTDigitsDataset


def test_emnist_digits_input_shape():
    ds = EMNISTDigitsDataset()
    assert ds.input_shape == (1, 28, 28)


def test_emnist_digits_num_classes():
    ds = EMNISTDigitsDataset()
    assert ds.num_classes == 10


def test_emnist_digits_input_dim_product():
    """Simulator computes input_dim = prod(input_shape) for MLP."""
    ds = EMNISTDigitsDataset()
    from math import prod
    assert prod(ds.input_shape) == 784


# ── _split_iid ─────────────────────────────────────────────────────────────

def test_split_iid_disjoint_total_preserved():
    """np.array_split spreads all samples; union = full dataset (no loss)."""
    ds = EMNISTDigitsDataset()
    mock = MockImageDataset(1000)
    parts = ds._split_iid(mock, n_nodes=10)
    all_idx = sorted(idx for sub in parts.values() for idx in sub.indices)
    assert len(all_idx) == 1000
    assert len(set(all_idx)) == 1000


def test_split_iid_disjoint_balanced_sizes():
    """np.array_split spreads remainder across first chunks (sizes differ ≤1)."""
    ds = EMNISTDigitsDataset()
    mock = MockImageDataset(1003)
    parts = ds._split_iid(mock, n_nodes=10)
    sizes = [len(sub) for sub in parts.values()]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == 1003


def test_split_iid_overlap_mode_size():
    """samples_per_node=k: each node gets exactly k samples."""
    ds = EMNISTDigitsDataset()
    mock = MockImageDataset(1000)
    parts = ds._split_iid(mock, n_nodes=5, samples_per_node=200)
    for sub in parts.values():
        assert len(sub) == 200


def test_split_iid_overlap_capped_at_total():
    """samples_per_node > n_total caps at n_total."""
    ds = EMNISTDigitsDataset()
    mock = MockImageDataset(100)
    parts = ds._split_iid(mock, n_nodes=3, samples_per_node=500)
    for sub in parts.values():
        assert len(sub) == 100


def test_split_iid_n_nodes_count():
    """Returns exactly n_nodes subsets keyed 0..n_nodes-1."""
    ds = EMNISTDigitsDataset()
    mock = MockImageDataset(500)
    parts = ds._split_iid(mock, n_nodes=20)
    assert set(parts.keys()) == set(range(20))


# ── _split_dirichlet ───────────────────────────────────────────────────────

def test_split_dirichlet_returns_n_nodes():
    ds = EMNISTDigitsDataset()
    mock = MockImageDataset(2000)
    parts = ds._split_dirichlet(mock, n_nodes=10, alpha=0.5)
    assert set(parts.keys()) == set(range(10))


def test_split_dirichlet_total_preserved():
    """All samples assigned exactly once (per-class disjoint partition)."""
    ds = EMNISTDigitsDataset()
    mock = MockImageDataset(2000)
    parts = ds._split_dirichlet(mock, n_nodes=10, alpha=0.5)
    all_idx = [idx for sub in parts.values() for idx in sub.indices]
    assert sum(len(sub) for sub in parts.values()) == 2000
    assert len(set(all_idx)) == len(all_idx)


def test_split_dirichlet_handles_empty_class():
    """If a class has 0 samples, the loop's `continue` guard prevents crash."""
    ds = EMNISTDigitsDataset()
    mock = MockImageDataset(1000, missing_classes=[3, 7])
    parts = ds._split_dirichlet(mock, n_nodes=8, alpha=0.5)
    assert len(parts) == 8


def test_split_dirichlet_alpha_concentration():
    """High alpha (≥10) → near-uniform partition. Low alpha (≤0.1) → skewed."""
    ds = EMNISTDigitsDataset()
    mock = MockImageDataset(5000)
    np.random.seed(42)
    parts_uniform = ds._split_dirichlet(mock, n_nodes=10, alpha=10.0)
    np.random.seed(42)
    parts_skewed = ds._split_dirichlet(mock, n_nodes=10, alpha=0.05)

    sizes_uniform = sorted(len(sub) for sub in parts_uniform.values())
    sizes_skewed = sorted(len(sub) for sub in parts_skewed.values())

    cv_uniform = np.std(sizes_uniform) / max(np.mean(sizes_uniform), 1)
    cv_skewed = np.std(sizes_skewed) / max(np.mean(sizes_skewed), 1)
    assert cv_skewed > cv_uniform


def test_split_dirichlet_warns_on_starved_nodes(caplog):
    """When n_nodes >> n_classes and alpha is low, many nodes get <10 samples
    → a warning must be logged."""
    ds = EMNISTDigitsDataset()
    mock = MockImageDataset(500)
    with caplog.at_level(logging.WARNING, logger="dpfl.data.emnist_digits_dataset"):
        ds._split_dirichlet(mock, n_nodes=200, alpha=0.05)
    assert any("<10 samples" in rec.message for rec in caplog.records)


# ── public split() dispatcher ──────────────────────────────────────────────

def test_split_dispatcher_iid():
    ds = EMNISTDigitsDataset()
    mock = MockImageDataset(100)
    parts = ds.split(mock, n_nodes=5, mode="iid")
    assert len(parts) == 5


def test_split_dispatcher_dirichlet():
    ds = EMNISTDigitsDataset()
    mock = MockImageDataset(500)
    parts = ds.split(mock, n_nodes=5, mode="dirichlet", alpha=1.0)
    assert len(parts) == 5


def test_split_dispatcher_unknown_mode_raises():
    ds = EMNISTDigitsDataset()
    mock = MockImageDataset(100)
    with pytest.raises(ValueError, match="Unknown split mode"):
        ds.split(mock, n_nodes=5, mode="bogus")


# ── Pattern parity with MNIST/FEMNIST ──────────────────────────────────────

def test_split_signature_matches_base():
    """Confirm split signature matches BaseDataset abstract contract."""
    import inspect
    from dpfl.data.base_dataset import BaseDataset
    base_sig = inspect.signature(BaseDataset.split)
    impl_sig = inspect.signature(EMNISTDigitsDataset.split)
    assert list(base_sig.parameters.keys()) == list(impl_sig.parameters.keys())


def test_load_signature_matches_base():
    import inspect
    from dpfl.data.base_dataset import BaseDataset
    base_sig = inspect.signature(BaseDataset.load)
    impl_sig = inspect.signature(EMNISTDigitsDataset.load)
    assert list(base_sig.parameters.keys()) == list(impl_sig.parameters.keys())
