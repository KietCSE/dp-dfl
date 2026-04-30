"""Phase 4: chunking equivalence — chunked vmap output must match unchunked.

For both standard SGD and DP-SGD per-step (with z=0 to remove RNG order
sensitivity), running with chunk_size=k must produce the same updates as
chunk_size=0 to within float32 reduction noise.
"""

import copy

import torch

from dpfl.core.gaussian_mechanism import GaussianMechanism
from dpfl.core.vectorized_state import ParamShapeSpec, pack_node_params
from dpfl.core.vectorized_trainer import (
    train_all_standard, train_all_dpsgd_per_step,
)
from dpfl.models.mlp_model import MLP


class _Node:
    def __init__(self, model, idx):
        self.model = model
        self.id = idx


class _Pipe:
    def __init__(self, x, y, node_ids):
        self.x, self.y = x, y
        self.node_ids = list(node_ids)
        self.N = x.shape[0]
        self.max_n_samples = x.shape[1]

    def iter_train_batches(self, B):
        n = self.max_n_samples // B
        for b in range(n):
            s, e = b * B, (b + 1) * B
            yield (
                self.x[:, s:e], self.y[:, s:e],
                torch.ones(self.N, B, dtype=torch.bool, device=self.x.device),
            )


def test_standard_sgd_chunked_matches_unchunked():
    torch.manual_seed(101)
    N, n_per, B, lr, epochs = 8, 16, 4, 0.05, 2
    base = MLP(784, 8, 10)
    spec = ParamShapeSpec(base)
    models = [copy.deepcopy(base) for _ in range(N)]
    for k, m in enumerate(models):
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.01 * k)

    nodes = [_Node(m, k) for k, m in enumerate(models)]
    pstack = pack_node_params(nodes)
    X = torch.randn(N, n_per, 1, 28, 28)
    Y = torch.randint(0, 10, (N, n_per))
    pipe = _Pipe(X, Y, list(range(N)))

    upd_full, steps_full = train_all_standard(
        base, pstack.clone(), spec, pipe, B, epochs, lr, chunk_size=0)
    upd_chunk, steps_chunk = train_all_standard(
        base, pstack.clone(), spec, pipe, B, epochs, lr, chunk_size=3)

    assert steps_full == steps_chunk
    diff = (upd_full - upd_chunk).abs().max().item()
    assert diff < 1e-4, f"chunked vs full diff: {diff:.2e}"


def test_dpsgd_chunked_zero_noise_matches_unchunked():
    """With z=0, no RNG draws — chunked must match unchunked bit-near-equal."""
    torch.manual_seed(202)
    N, n_per, B, lr, epochs = 8, 12, 4, 0.05, 1
    clip = 1.0
    base = MLP(784, 8, 10)
    spec = ParamShapeSpec(base)
    models = [copy.deepcopy(base) for _ in range(N)]
    for k, m in enumerate(models):
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.005 * k)

    nodes = [_Node(m, k) for k, m in enumerate(models)]
    pstack = pack_node_params(nodes)
    X = torch.randn(N, n_per, 1, 28, 28)
    Y = torch.randint(0, 10, (N, n_per))
    pipe = _Pipe(X, Y, list(range(N)))

    mech = GaussianMechanism()
    mech.set_generator(torch.Generator().manual_seed(0))

    upd_full, _ = train_all_dpsgd_per_step(
        base, pstack.clone(), spec, pipe, mech,
        batch_size=B, epochs=epochs, lr=lr,
        clip_bound=clip, noise_mult=0.0, chunk_size=0,
    )
    upd_chunk, _ = train_all_dpsgd_per_step(
        base, pstack.clone(), spec, pipe, mech,
        batch_size=B, epochs=epochs, lr=lr,
        clip_bound=clip, noise_mult=0.0, chunk_size=3,
    )

    diff = (upd_full - upd_chunk).abs().max().item()
    assert diff < 1e-4, f"DP-SGD chunked vs full diff: {diff:.2e}"
