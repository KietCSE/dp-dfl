"""Smoke tests for Phase 0 vectorized infrastructure: pack/unpack + data pipeline."""

import torch
import torch.nn as nn

from dpfl.core.vectorized_state import (
    ParamShapeSpec, pack_node_params, unpack_to_nodes,
)
from dpfl.core.vectorized_data import VectorizedDataPipeline
from dpfl.models.mlp_model import MLP


# ── ParamShapeSpec ─────────────────────────────────────────────────────────

def test_param_shape_spec_records_all_params_in_order():
    model = MLP(input_dim=784, hidden_size=16, num_classes=10)
    spec = ParamShapeSpec(model)

    expected_names = [n for n, _ in model.named_parameters()]
    actual_names = [n for n, _, _ in spec.specs]
    assert actual_names == expected_names

    expected_total = sum(p.numel() for p in model.parameters())
    assert spec.D == expected_total


def test_to_dict_batched_round_trip():
    model = MLP(input_dim=784, hidden_size=16, num_classes=10)
    spec = ParamShapeSpec(model)
    N = 5

    stack = torch.randn(N, spec.D)
    pdict = spec.to_dict_batched(stack)
    recovered = spec.from_dict_batched(pdict)
    assert torch.allclose(stack, recovered)


def test_to_dict_round_trip_single():
    model = MLP(input_dim=784, hidden_size=16, num_classes=10)
    spec = ParamShapeSpec(model)

    flat = torch.randn(spec.D)
    pdict = spec.to_dict(flat)
    recovered = spec.from_dict(pdict)
    assert torch.allclose(flat, recovered)


def test_to_dict_batched_shapes_match_named_params():
    """Each entry in batched dict has shape (N, *param.shape)."""
    model = MLP(input_dim=784, hidden_size=16, num_classes=10)
    spec = ParamShapeSpec(model)
    N = 7
    pdict = spec.to_dict_batched(torch.randn(N, spec.D))

    for name, p in model.named_parameters():
        assert pdict[name].shape == (N, *p.shape), (
            f"{name}: expected {(N, *p.shape)}, got {pdict[name].shape}")


# ── pack/unpack ────────────────────────────────────────────────────────────

class _Node:
    def __init__(self, model):
        self.model = model


def test_pack_unpack_round_trip():
    """pack -> unpack restores per-node params bit-exact."""
    torch.manual_seed(42)
    nodes = [_Node(MLP(784, 16, 10)) for _ in range(4)]
    originals = [n.model.get_flat_params().clone() for n in nodes]

    stack = pack_node_params(nodes)
    assert stack.shape == (4, originals[0].numel())

    # Mutate models, then restore via unpack
    for n in nodes:
        n.model.set_flat_params(torch.zeros_like(n.model.get_flat_params()))
    unpack_to_nodes(stack, nodes)

    for orig, node in zip(originals, nodes):
        assert torch.allclose(orig, node.model.get_flat_params())


def test_pack_preserves_get_flat_params_layout():
    """A packed row equals the corresponding node's get_flat_params()."""
    torch.manual_seed(7)
    nodes = [_Node(MLP(784, 16, 10)) for _ in range(3)]
    stack = pack_node_params(nodes)

    for i, node in enumerate(nodes):
        assert torch.allclose(stack[i], node.model.get_flat_params())


# ── VectorizedDataPipeline ─────────────────────────────────────────────────

def _make_synth_dataset(n_total=100, dim=8):
    """Tiny in-memory dataset, behaves like torchvision (idx -> (tensor, label))."""

    class _DS:
        def __init__(self):
            self.x = torch.randn(n_total, dim)
            self.y = torch.randint(0, 4, (n_total,))

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, idx):
            return self.x[idx], int(self.y[idx])

    return _DS()


def test_pipeline_materializes_on_device():
    from torch.utils.data import Subset
    ds = _make_synth_dataset(n_total=60)
    node_data = {0: Subset(ds, list(range(20))), 1: Subset(ds, list(range(20, 60)))}
    gen = torch.Generator(); gen.manual_seed(1)

    pipe = VectorizedDataPipeline(ds, node_data, torch.device("cpu"), gen)

    assert pipe.X_full.shape == (60, 8)
    assert pipe.Y_full.shape == (60,)
    assert pipe.N == 2
    assert pipe.client_sizes.tolist() == [20, 40]
    assert pipe.max_n_samples == 40


def test_pipeline_iter_train_batches_shape_and_mask():
    """Batches have shape (N, B, *input_shape); mask is True for real samples only."""
    from torch.utils.data import Subset
    ds = _make_synth_dataset(n_total=60)
    node_data = {0: Subset(ds, list(range(10))), 1: Subset(ds, list(range(10, 60)))}
    gen = torch.Generator(); gen.manual_seed(1)
    pipe = VectorizedDataPipeline(ds, node_data, torch.device("cpu"), gen)

    batches = list(pipe.iter_train_batches(batch_size=5))
    # max_n_samples=50, batch_size=5 -> 10 batches per epoch
    assert len(batches) == 10

    for x, y, mask in batches:
        assert x.shape == (2, 5, 8)
        assert y.shape == (2, 5)
        assert mask.shape == (2, 5)
        assert mask.dtype == torch.bool

    # Client 0 has only 10 samples: only first 2 batches should be all-real
    # (positions 0..9), batch 2+ should be all-padded (positions 10..)
    real_counts = [m[0].sum().item() for _, _, m in batches]
    assert real_counts[:2] == [5, 5]
    assert all(c == 0 for c in real_counts[2:])

    # Client 1 has 50 samples: all 10 batches all-real
    assert all(m[1].sum().item() == 5 for _, _, m in batches)


def test_pipeline_deterministic_with_same_seed():
    """Same data_gen seed -> same shuffled batches."""
    from torch.utils.data import Subset
    ds = _make_synth_dataset(n_total=40)
    node_data = {0: Subset(ds, list(range(20))), 1: Subset(ds, list(range(20, 40)))}

    def collect():
        gen = torch.Generator(); gen.manual_seed(777)
        pipe = VectorizedDataPipeline(ds, node_data, torch.device("cpu"), gen)
        return [b[0].clone() for b in pipe.iter_train_batches(batch_size=4)]

    a = collect()
    b = collect()
    assert len(a) == len(b)
    for xa, xb in zip(a, b):
        assert torch.allclose(xa, xb)
