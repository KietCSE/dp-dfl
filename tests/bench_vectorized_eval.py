"""Quick benchmark: vectorized eval vs sequential eval on a real-sized setup.

Run: python -m dpfl.tests.bench_vectorized_eval
"""

import copy
import time

import torch
import torch.nn.functional as F

from dpfl.core.vectorized_eval import vectorized_evaluate
from dpfl.core.vectorized_state import ParamShapeSpec, pack_node_params
from dpfl.models.mlp_model import MLP


def _seq_eval(models, X, Y, bs=256):
    M = X.shape[0]
    out = {}
    for i, m in enumerate(models):
        m.eval()
        c, ls = 0, 0.0
        with torch.no_grad():
            for s in range(0, M, bs):
                lg = m(X[s:s + bs])
                ls += F.cross_entropy(lg, Y[s:s + bs], reduction="sum").item()
                c += (lg.argmax(-1) == Y[s:s + bs]).sum().item()
        m.train()
        out[i] = (c / M, ls / M)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for N in [50, 200, 500]:
        torch.manual_seed(0)
        models = [MLP(784, 100, 10).to(device) for _ in range(N)]
        for k, m in enumerate(models):
            with torch.no_grad():
                for p in m.parameters():
                    p.add_(0.01 * k)  # Make models distinct

        base = copy.deepcopy(models[0])
        spec = ParamShapeSpec(base)

        class _N:
            def __init__(self, m, i): self.model, self.id = m, i

        nodes = [_N(m, i) for i, m in enumerate(models)]
        params_stack = pack_node_params(nodes, device=device)

        M = 10000
        X = torch.randn(M, 1, 28, 28, device=device)
        Y = torch.randint(0, 10, (M,), device=device)

        # Warm up
        _ = vectorized_evaluate(base, params_stack, spec, list(range(N)), X, Y, 256)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        out_vec = vectorized_evaluate(base, params_stack, spec, list(range(N)),
                                      X, Y, 256)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_vec = time.perf_counter() - t0

        t0 = time.perf_counter()
        out_seq = _seq_eval(models, X, Y, 256)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_seq = time.perf_counter() - t0

        # Sanity check
        max_acc_diff = max(abs(out_vec[i]["accuracy"] - out_seq[i][0]) for i in range(N))
        max_loss_diff = max(abs(out_vec[i]["test_loss"] - out_seq[i][1]) for i in range(N))

        print(
            f"N={N:4d}  M={M}  seq={t_seq:7.3f}s  vec={t_vec:7.3f}s  "
            f"speedup={t_seq / t_vec:5.1f}x  "
            f"|Δacc|={max_acc_diff:.2e}  |Δloss|={max_loss_diff:.2e}"
        )


if __name__ == "__main__":
    main()
