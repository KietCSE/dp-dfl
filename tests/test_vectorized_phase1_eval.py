"""Phase 1 regression: vmapped evaluation matches sequential evaluation.

vectorized_evaluate(N models, X_test) must produce the same per-client
{accuracy, test_loss} as a sequential loop calling each model on X_test.
Tolerance is loose because vmap reduction order may differ in float32, but
output should agree to 4-5 decimal places on small test sets.
"""

import copy

import torch
import torch.nn.functional as F

from dpfl.core.vectorized_eval import vectorized_evaluate
from dpfl.core.vectorized_state import ParamShapeSpec, pack_node_params
from dpfl.models.mlp_model import MLP


class _Node:
    def __init__(self, model, node_id):
        self.model = model
        self.id = node_id


def _sequential_eval(models, X_test, Y_test, batch_size=256):
    """Legacy-style per-node sequential eval; ground truth for the vectorized path."""
    out = {}
    M = X_test.shape[0]
    for i, m in enumerate(models):
        m.eval()
        correct, loss_sum = 0.0, 0.0
        with torch.no_grad():
            for s in range(0, M, batch_size):
                x = X_test[s:s + batch_size]
                y = Y_test[s:s + batch_size]
                logits = m(x)
                loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
                correct += (logits.argmax(-1) == y).sum().item()
        m.train()
        out[i] = {"accuracy": correct / M, "test_loss": loss_sum / M}
    return out


def _build_distinct_models(N, hidden=16):
    """N MLPs each initialized to different random weights."""
    models = []
    for k in range(N):
        torch.manual_seed(100 + k)
        m = MLP(input_dim=784, hidden_size=hidden, num_classes=10)
        models.append(m)
    return models


# ── Correctness vs sequential ──────────────────────────────────────────────

def test_vectorized_eval_matches_sequential_5_clients():
    torch.manual_seed(0)
    N = 5
    models = _build_distinct_models(N)
    base_model = copy.deepcopy(models[0])  # template (params irrelevant)
    spec = ParamShapeSpec(base_model)

    nodes = [_Node(m, i) for i, m in enumerate(models)]
    params_stack = pack_node_params(nodes)

    M = 200
    X_test = torch.randn(M, 1, 28, 28)
    Y_test = torch.randint(0, 10, (M,))

    seq = _sequential_eval(models, X_test, Y_test, batch_size=64)
    vec = vectorized_evaluate(
        base_model=base_model,
        params_stack=params_stack,
        spec=spec,
        node_ids=[n.id for n in nodes],
        X_test=X_test,
        Y_test=Y_test,
        batch_size=64,
    )

    for i in range(N):
        assert abs(seq[i]["accuracy"] - vec[i]["accuracy"]) < 1e-5, (
            f"client {i} acc: seq={seq[i]['accuracy']} vec={vec[i]['accuracy']}")
        assert abs(seq[i]["test_loss"] - vec[i]["test_loss"]) < 1e-4, (
            f"client {i} loss: seq={seq[i]['test_loss']} vec={vec[i]['test_loss']}")


def test_vectorized_eval_chunked_matches_unchunked():
    """chunk_size=2 should give identical output to chunk_size=0 (no chunking)."""
    torch.manual_seed(1)
    N = 6
    models = _build_distinct_models(N)
    base_model = copy.deepcopy(models[0])
    spec = ParamShapeSpec(base_model)
    nodes = [_Node(m, i) for i, m in enumerate(models)]
    params_stack = pack_node_params(nodes)

    M = 150
    X_test = torch.randn(M, 1, 28, 28)
    Y_test = torch.randint(0, 10, (M,))
    common = dict(
        base_model=base_model, params_stack=params_stack, spec=spec,
        node_ids=[n.id for n in nodes], X_test=X_test, Y_test=Y_test,
        batch_size=32,
    )

    full = vectorized_evaluate(chunk_size=0, **common)
    chunked = vectorized_evaluate(chunk_size=2, **common)

    for i in range(N):
        assert abs(full[i]["accuracy"] - chunked[i]["accuracy"]) < 1e-6
        assert abs(full[i]["test_loss"] - chunked[i]["test_loss"]) < 1e-5


def test_vectorized_eval_large_batch_does_not_oom():
    """Smoke test: many clients, larger model — should run without errors."""
    torch.manual_seed(2)
    N = 20
    models = _build_distinct_models(N, hidden=64)
    base_model = copy.deepcopy(models[0])
    spec = ParamShapeSpec(base_model)
    nodes = [_Node(m, i) for i, m in enumerate(models)]
    params_stack = pack_node_params(nodes)

    M = 100
    X_test = torch.randn(M, 1, 28, 28)
    Y_test = torch.randint(0, 10, (M,))

    out = vectorized_evaluate(
        base_model=base_model,
        params_stack=params_stack,
        spec=spec,
        node_ids=[n.id for n in nodes],
        X_test=X_test, Y_test=Y_test,
        batch_size=32,
    )
    assert len(out) == N
    for i in range(N):
        assert 0.0 <= out[i]["accuracy"] <= 1.0
        assert out[i]["test_loss"] > 0  # untrained MLP -> nonzero loss
