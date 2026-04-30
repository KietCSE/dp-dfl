"""Vectorized evaluation: torch.func.vmap over N client models on shared test set.

Replaces the sequential loop in BaseSimulator._evaluate_nodes (which runs
N × n_test_batches forward passes) with a single vmapped pass over all clients
per test batch. Big win when N is large.

Output dict shape matches the legacy _evaluate_nodes output exactly so
downstream metric tracking is untouched.
"""

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap

from dpfl.core.vectorized_state import ParamShapeSpec
from dpfl.models.base_model import BaseModel


@torch.no_grad()
def vectorized_evaluate(
    base_model: BaseModel,
    params_stack: torch.Tensor,
    spec: ParamShapeSpec,
    node_ids: List[int],
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    batch_size: int = 256,
    chunk_size: int = 0,
) -> Dict[int, Dict[str, float]]:
    """vmap forward over N client models on the shared (X_test, Y_test).

    Args:
      base_model: stateless template; functional_call uses its forward
      params_stack: (N, D) packed client params, on same device as X_test
      spec: ParamShapeSpec for the base_model
      node_ids: ordered list — row i of params_stack belongs to node_ids[i]
      X_test, Y_test: device tensors, shapes (M, *input_shape) and (M,)
      batch_size: test batch size
      chunk_size: 0 = process all N at once; >0 = chunk to bound VRAM

    Returns:
      {node_id: {"accuracy": float, "test_loss": float}}
    """
    N = params_stack.shape[0]
    M = X_test.shape[0]
    device = X_test.device

    if chunk_size <= 0 or chunk_size >= N:
        return _eval_chunk(base_model, params_stack, spec, node_ids,
                           X_test, Y_test, batch_size)

    # Chunked path: split params_stack into chunks, evaluate each
    correct = torch.zeros(N, device=device)
    loss_sum = torch.zeros(N, device=device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_ids = node_ids[start:end]
        partial = _eval_chunk(
            base_model, params_stack[start:end], spec, chunk_ids,
            X_test, Y_test, batch_size,
        )
        for i, nid in enumerate(chunk_ids):
            correct[start + i] = partial[nid]["_correct"]
            loss_sum[start + i] = partial[nid]["_loss_sum"]

    return {
        nid: {"accuracy": float(correct[i] / M), "test_loss": float(loss_sum[i] / M)}
        for i, nid in enumerate(node_ids)
    }


def _eval_chunk(
    base_model: BaseModel,
    params_stack: torch.Tensor,
    spec: ParamShapeSpec,
    node_ids: List[int],
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    batch_size: int,
) -> Dict[int, Dict[str, float]]:
    """Evaluate a single chunk of clients on the full test set."""
    N = params_stack.shape[0]
    M = X_test.shape[0]
    device = X_test.device
    pdict = spec.to_dict_batched(params_stack)

    def fmodel(params, x):
        return functional_call(base_model, params, (x,))

    batched_fmodel = vmap(fmodel, in_dims=(0, None))

    correct = torch.zeros(N, device=device)
    loss_sum = torch.zeros(N, device=device)

    for s in range(0, M, batch_size):
        x_batch = X_test[s:s + batch_size]
        y_batch = Y_test[s:s + batch_size]
        # logits: (N, B, C)
        logits = batched_fmodel(pdict, x_batch)
        preds = logits.argmax(dim=-1)
        # Broadcast y over N
        correct += (preds == y_batch.unsqueeze(0)).sum(dim=-1)
        # Per-client loss sum: cross_entropy reduction='none' on flattened
        N_, B, C = logits.shape
        flat_logits = logits.reshape(N_ * B, C)
        flat_y = y_batch.unsqueeze(0).expand(N_, -1).reshape(N_ * B)
        per = F.cross_entropy(flat_logits, flat_y, reduction="none").reshape(N_, B)
        loss_sum += per.sum(dim=-1)

    return {
        nid: {
            "accuracy": float(correct[i] / M),
            "test_loss": float(loss_sum[i] / M),
            # Hidden keys for chunked aggregation:
            "_correct": correct[i],
            "_loss_sum": loss_sum[i],
        }
        for i, nid in enumerate(node_ids)
    }
