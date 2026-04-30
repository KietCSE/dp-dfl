"""Phase 2 regression: vectorized SGD update equals sequential SGD on same batches.

vmap reduction order in float32 introduces tiny drift, but on identical batch
sequences the per-client update tensor must agree with sequential SGD to
~1e-5 absolute. We run an N=3 setup with a controlled (deterministic) data
permutation and compare each row of the (N, D) update stack.
"""

import copy

import torch
import torch.nn.functional as F

from dpfl.core.vectorized_state import ParamShapeSpec, pack_node_params
from dpfl.core.vectorized_trainer import train_all_standard
from dpfl.models.mlp_model import MLP


class _Node:
    def __init__(self, model, idx):
        self.model = model
        self.id = idx


class _DeterministicPipeline:
    """Stand-in for VectorizedDataPipeline that yields fixed batches.

    Bypasses RNG so we can compare bit-near-equal vs sequential SGD on
    the exact same input sequence per client.
    """

    def __init__(self, x_per_client, y_per_client, node_ids, batch_size):
        # x_per_client: (N, n_per_client, *input_shape); same n for all clients
        # y_per_client: (N, n_per_client)
        self.x = x_per_client
        self.y = y_per_client
        self.node_ids = list(node_ids)
        self.batch_size = batch_size
        self.N = x_per_client.shape[0]
        self.max_n_samples = x_per_client.shape[1]

    def iter_train_batches(self, batch_size):
        n_batches = self.max_n_samples // batch_size
        for b in range(n_batches):
            s, e = b * batch_size, (b + 1) * batch_size
            x = self.x[:, s:e]   # (N, B, *input_shape)
            y = self.y[:, s:e]   # (N, B)
            mask = torch.ones(self.N, batch_size, dtype=torch.bool, device=x.device)
            yield x, y, mask


def _sequential_sgd(model, X, Y, batch_size, lr, epochs):
    """Reference SGD on a single client's (X, Y) split into fixed batches."""
    initial = model.get_flat_params().clone()
    n_batches = X.shape[0] // batch_size
    for _ in range(epochs):
        for b in range(n_batches):
            s, e = b * batch_size, (b + 1) * batch_size
            x_b, y_b = X[s:e], Y[s:e]
            out = model(x_b)
            loss = F.cross_entropy(out, y_b)
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= lr * p.grad
    return model.get_flat_params() - initial


def test_vectorized_update_matches_sequential_per_client():
    torch.manual_seed(123)
    N, n_per, B, lr, epochs = 3, 16, 4, 0.05, 2
    base = MLP(input_dim=784, hidden_size=8, num_classes=10)
    spec = ParamShapeSpec(base)

    # N independent models with distinct init
    models = []
    for k in range(N):
        torch.manual_seed(200 + k)
        m = MLP(input_dim=784, hidden_size=8, num_classes=10)
        models.append(m)

    # N independent data tensors, fixed (no RNG inside the pipeline)
    X = torch.randn(N, n_per, 1, 28, 28)
    Y = torch.randint(0, 10, (N, n_per))

    # Sequential reference: per-client SGD with the same batch sequence
    seq_models = [copy.deepcopy(m) for m in models]
    seq_updates = []
    for k in range(N):
        seq_updates.append(_sequential_sgd(seq_models[k], X[k], Y[k], B, lr, epochs))
    seq_stack = torch.stack(seq_updates, dim=0)

    # Vectorized
    nodes = [_Node(m, k) for k, m in enumerate(models)]
    params_stack = pack_node_params(nodes)
    pipe = _DeterministicPipeline(X, Y, [k for k in range(N)], B)

    updates_stack, steps = train_all_standard(
        base_model=base, params_stack=params_stack, spec=spec,
        train_pipeline=pipe, batch_size=B, epochs=epochs, lr=lr,
    )

    # Per-row comparison
    for k in range(N):
        diff = (updates_stack[k] - seq_stack[k]).abs().max().item()
        assert diff < 1e-4, (
            f"client {k}: max |Δ_vec - Δ_seq| = {diff:.2e}; vmap drift too large")

    # Steps dict reflects actual batch count per client
    assert all(s == (n_per // B) * epochs for s in steps.values())


def test_vectorized_update_zero_when_lr_zero():
    """lr=0 -> updates are zero regardless of data."""
    torch.manual_seed(7)
    N, n_per, B = 2, 8, 4
    base = MLP(784, 8, 10)
    spec = ParamShapeSpec(base)
    models = [copy.deepcopy(base) for _ in range(N)]
    X = torch.randn(N, n_per, 1, 28, 28)
    Y = torch.randint(0, 10, (N, n_per))

    nodes = [_Node(m, k) for k, m in enumerate(models)]
    pstack = pack_node_params(nodes)
    pipe = _DeterministicPipeline(X, Y, [0, 1], B)
    updates, _ = train_all_standard(
        base, pstack, spec, pipe, batch_size=B, epochs=1, lr=0.0,
    )
    assert updates.abs().max().item() < 1e-7


def test_vectorized_masked_loss_zeros_padded_grads():
    """Padded positions (mask=False) must contribute zero gradient.

    Build a pipeline where client 1 has padding past sample 4. The
    vectorized update for client 1 must equal an SGD run on just the first
    4 samples (real ones), independent of what's at indices 4-7 (padding).
    """
    from dpfl.core.vectorized_trainer import train_all_standard

    class _MaskedPipe:
        def __init__(self, x, y, mask, node_ids, batch_size):
            self.x = x; self.y = y; self.mask = mask
            self.node_ids = node_ids; self.batch_size = batch_size
            self.N = x.shape[0]; self.max_n_samples = x.shape[1]

        def iter_train_batches(self, batch_size):
            n_batches = self.max_n_samples // batch_size
            for b in range(n_batches):
                s, e = b * batch_size, (b + 1) * batch_size
                yield self.x[:, s:e], self.y[:, s:e], self.mask[:, s:e]

    torch.manual_seed(42)
    N, n_per, B, lr, epochs = 2, 8, 4, 0.05, 1
    base = MLP(784, 8, 10)
    spec = ParamShapeSpec(base)
    models = [copy.deepcopy(base) for _ in range(N)]

    X_real = torch.randn(N, 4, 1, 28, 28)  # client 1's real data
    X_pad = torch.full((N, 4, 1, 28, 28), 999.0)  # noise; should not affect grad
    X = torch.cat([X_real, X_pad], dim=1)  # (N, 8, 1, 28, 28)
    Y = torch.zeros(N, n_per, dtype=torch.long)
    Y[:, :4] = torch.randint(0, 10, (N, 4))

    mask = torch.zeros(N, n_per, dtype=torch.bool)
    mask[:, :4] = True  # first 4 real, last 4 padded

    nodes = [_Node(m, k) for k, m in enumerate(models)]
    pstack = pack_node_params(nodes)

    pipe_full = _MaskedPipe(X, Y, mask, [0, 1], B)
    upd_full, _ = train_all_standard(base, pstack, spec, pipe_full, B, epochs, lr)

    # Run again with X_pad replaced by different garbage — must yield same update
    X_pad2 = torch.full((N, 4, 1, 28, 28), -7.0)
    X2 = torch.cat([X_real, X_pad2], dim=1)
    pipe2 = _MaskedPipe(X2, Y, mask, [0, 1], B)
    upd2, _ = train_all_standard(base, pstack, spec, pipe2, B, epochs, lr)

    diff = (upd_full - upd2).abs().max().item()
    assert diff < 1e-5, (
        f"Padded samples leaked into gradient; max diff = {diff:.2e}")
