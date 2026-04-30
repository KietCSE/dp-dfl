"""Quick benchmark: vectorized SGD training vs sequential SGD (legacy-style).

Approximates the production setup: N clients, MNIST-shaped MLP, batch_size=64,
1 local epoch over n_per_client samples.

Run: python -m dpfl.tests.bench_vectorized_train
"""

import copy
import time

import torch
import torch.nn.functional as F

from dpfl.core.vectorized_state import ParamShapeSpec, pack_node_params
from dpfl.core.vectorized_trainer import train_all_standard
from dpfl.models.mlp_model import MLP


class _Pipe:
    """Pre-batched (no-RNG) pipeline for the bench."""
    def __init__(self, x, y, node_ids):
        self.x, self.y = x, y
        self.node_ids = node_ids
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


def _seq_train(models, X, Y, B, lr, epochs):
    """Sequential per-client SGD (mirrors current ThreadPool behavior)."""
    out = []
    for k, m in enumerate(models):
        initial = m.get_flat_params().clone()
        n = X.shape[1] // B
        for _ in range(epochs):
            for b in range(n):
                s, e = b * B, (b + 1) * B
                logits = m(X[k, s:e])
                loss = F.cross_entropy(logits, Y[k, s:e])
                m.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for p in m.parameters():
                        p -= lr * p.grad
        out.append(m.get_flat_params() - initial)
    return torch.stack(out, dim=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for N in [50, 200, 500]:
        torch.manual_seed(0)
        models = [MLP(784, 100, 10).to(device) for _ in range(N)]
        for k, m in enumerate(models):
            with torch.no_grad():
                for p in m.parameters():
                    p.add_(0.01 * k)  # distinct init

        base = copy.deepcopy(models[0])
        spec = ParamShapeSpec(base)

        class _N:
            def __init__(self, m, i): self.model, self.id = m, i
        nodes = [_N(m, i) for i, m in enumerate(models)]
        params_stack = pack_node_params(nodes, device=device)

        n_per, B, lr, epochs = 200, 64, 0.05, 1
        X = torch.randn(N, n_per, 1, 28, 28, device=device)
        Y = torch.randint(0, 10, (N, n_per), device=device)

        # Warmup vectorized
        _ = train_all_standard(base, params_stack.clone(), spec,
                                _Pipe(X, Y, list(range(N))), B, epochs, lr)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Bench vectorized
        t0 = time.perf_counter()
        upd_vec, _ = train_all_standard(
            base, params_stack.clone(), spec,
            _Pipe(X, Y, list(range(N))), B, epochs, lr,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_vec = time.perf_counter() - t0

        # Bench sequential (clone models so state is fresh)
        seq_models = [copy.deepcopy(m) for m in models]
        t0 = time.perf_counter()
        upd_seq = _seq_train(seq_models, X, Y, B, lr, epochs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_seq = time.perf_counter() - t0

        max_diff = (upd_vec - upd_seq).abs().max().item()

        print(
            f"N={N:4d}  n_per={n_per}  B={B}  "
            f"seq={t_seq:7.3f}s  vec={t_vec:7.3f}s  "
            f"speedup={t_seq / t_vec:5.2f}x  |Δupd|max={max_diff:.2e}"
        )


if __name__ == "__main__":
    main()
