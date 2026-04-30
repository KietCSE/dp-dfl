"""Vectorized client training: torch.func.vmap over N clients in one forward+backward.

Replaces the ThreadPoolExecutor-driven per-client loop in BaseSimulator with
a single vmapped step that updates all N client param vectors in parallel on
device. Targets ~20-40x speedup on GPU vs the GIL-bound thread pool.

Public entries:
  - train_all_standard: plain SGD, single vmap over clients
  - train_all_dpsgd_per_step: DP-SGD per-step, NESTED vmap (clients × samples)

Returns (updates, steps_dict) matching the legacy _train_all_nodes contract:
caller can unstack updates by node id.

Variable per-client data sizes (Dirichlet split) handled via padding + bool
mask: padded sample positions contribute zero gradient (loss masked to 0
before reduction). Steps dict reports each client's actual batch count
(ceil(n_i / B) * epochs) so the privacy accountant tallies correctly.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from dpfl.core.gaussian_mechanism import GaussianMechanism
from dpfl.core.vectorized_data import VectorizedDataPipeline
from dpfl.core.vectorized_state import ParamShapeSpec
from dpfl.models.base_model import BaseModel


def train_all_standard(
    base_model: BaseModel,
    params_stack: torch.Tensor,
    spec: ParamShapeSpec,
    train_pipeline: VectorizedDataPipeline,
    batch_size: int,
    epochs: int,
    lr: float,
    chunk_size: int = 0,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """Vectorized SGD: train N clients in parallel for `epochs` over their data.

    Args:
      base_model: stateless template for functional_call
      params_stack: (N, D) initial params on device — modified IN PLACE? No,
                    a new stack is built and returned via 'updates' (delta).
      spec: ParamShapeSpec
      train_pipeline: GPU-resident data pipeline (yields N-batched mini-batches)
      batch_size, epochs, lr: SGD hyperparams
      chunk_size: 0 = process all N clients per vmap call; >0 = chunk to bound VRAM

    Returns:
      updates: (N, D) — final_params - initial_params, ordered to match
               train_pipeline.node_ids
      steps_dict: {node_id: actual_batch_count} accounting for variable n_i
    """
    return _train_chunk(
        base_model, params_stack, spec, train_pipeline,
        batch_size, epochs, lr, chunk_size,
    )


def _train_chunk(
    base_model: BaseModel,
    params_stack: torch.Tensor,
    spec: ParamShapeSpec,
    pipeline: VectorizedDataPipeline,
    batch_size: int,
    epochs: int,
    lr: float,
    chunk_size: int = 0,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """Train N clients with optional client-axis chunking to bound VRAM.

    chunk_size=0 (or >= N): full vmap over all clients per batch (max throughput).
    chunk_size=k (k<N): split each per-batch (N, B, *) tensor into k-sized
    sub-batches, vmap each independently, concat. Same RNG draw order on data
    (pipeline yields full N batches once per batch position).
    """
    initial = params_stack.clone()
    cur = params_stack.clone()
    N = cur.shape[0]
    eff_chunk = N if (chunk_size <= 0 or chunk_size >= N) else chunk_size

    def per_client_loss(params_dict, x, y):
        out = functional_call(base_model, params_dict, (x,))
        return F.cross_entropy(out, y)

    def per_client_loss_masked(params_dict, x, y, mask):
        out = functional_call(base_model, params_dict, (x,))
        per_sample = F.cross_entropy(out, y, reduction="none")
        masked = per_sample * mask.to(per_sample.dtype)
        denom = mask.sum().clamp(min=1).to(per_sample.dtype)
        return masked.sum() / denom

    grad_fn_unmasked = grad(per_client_loss)
    grad_fn_masked = grad(per_client_loss_masked)
    vmap_grad_unmasked = vmap(grad_fn_unmasked, in_dims=(0, 0, 0))
    vmap_grad_masked = vmap(grad_fn_masked, in_dims=(0, 0, 0, 0))

    steps_per_client = torch.zeros(N, dtype=torch.long, device=cur.device)

    for _ in range(epochs):
        for x_batch, y_batch, mask in pipeline.iter_train_batches(batch_size):
            had_real = mask.any(dim=1).to(torch.long)
            steps_per_client = steps_per_client + had_real

            for s in range(0, N, eff_chunk):
                e = min(s + eff_chunk, N)
                cur_chunk = cur[s:e]
                x_chunk = x_batch[s:e]
                y_chunk = y_batch[s:e]
                mask_chunk = mask[s:e]
                pdict = spec.to_dict_batched(cur_chunk)

                if mask_chunk.all():
                    g_dict = vmap_grad_unmasked(pdict, x_chunk, y_chunk)
                else:
                    g_dict = vmap_grad_masked(pdict, x_chunk, y_chunk, mask_chunk)

                g_stack = spec.from_dict_batched(g_dict)
                cur[s:e] = cur_chunk - lr * g_stack

    updates = cur - initial
    steps_dict = {
        nid: int(steps_per_client[i].item())
        for i, nid in enumerate(pipeline.node_ids)
    }
    return updates, steps_dict


def train_all_dpsgd_per_step(
    base_model: BaseModel,
    params_stack: torch.Tensor,
    spec: ParamShapeSpec,
    train_pipeline: VectorizedDataPipeline,
    mechanism: GaussianMechanism,
    batch_size: int,
    epochs: int,
    lr: float,
    clip_bound: float,
    noise_mult: float,
    chunk_size: int = 0,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """Vectorized per-step DP-SGD: nested vmap (clients × samples) -> clip+noise.

    For each batch:
      1. Inner vmap (over B samples): per-sample gradients per client
      2. Outer vmap (over N clients): replicates inner across clients
         -> grads shape (N, B, D)
      3. mechanism.clip_and_noise_batched: per-client L2 clip + per-client
         independent Gaussian noise -> (N, D)
      4. SGD step: params -= lr * (N, D)

    Phase 3 currently requires uniform per-client batch sizes (no padding).
    Caller (BaseSimulator._can_use_vectorized_training) gates on this.
    """
    initial = params_stack.clone()
    cur = params_stack.clone()
    N = cur.shape[0]
    eff_chunk = N if (chunk_size <= 0 or chunk_size >= N) else chunk_size

    def per_sample_loss(params, x, y):
        out = functional_call(base_model, params, (x.unsqueeze(0),))
        return F.cross_entropy(out, y.unsqueeze(0))

    inner = vmap(grad(per_sample_loss), in_dims=(None, 0, 0))
    nested = vmap(inner, in_dims=(0, 0, 0))

    steps_per_client = torch.zeros(N, dtype=torch.long, device=cur.device)

    for _ in range(epochs):
        for x_batch, y_batch, mask in train_pipeline.iter_train_batches(batch_size):
            if not mask.all():
                raise RuntimeError(
                    "DP-SGD per-step needs uniform per-client batches; "
                    "got padding mask — set use_vectorized=False or use IID split")

            for s in range(0, N, eff_chunk):
                e = min(s + eff_chunk, N)
                cur_chunk = cur[s:e]
                x_chunk = x_batch[s:e]
                y_chunk = y_batch[s:e]
                pdict = spec.to_dict_batched(cur_chunk)

                grads_dict = nested(pdict, x_chunk, y_chunk)
                per_sample_NBD = torch.cat(
                    [grads_dict[name].reshape(e - s, batch_size, -1)
                     for name, _, _ in spec.specs],
                    dim=2,
                )

                noised_avg = mechanism.clip_and_noise_batched(
                    per_sample_NBD, clip_bound, noise_mult, batch_size,
                )
                cur[s:e] = cur_chunk - lr * noised_avg

            steps_per_client = steps_per_client + 1

    updates = cur - initial
    steps_dict = {
        nid: int(steps_per_client[i].item())
        for i, nid in enumerate(train_pipeline.node_ids)
    }
    return updates, steps_dict
