"""Phase 3 regression: nested-vmap DP-SGD per-step matches per-client DP-SGD.

For each client we:
  1. Compute per-sample grads via the same vmap path used in legacy
     DPSGDTrainer._compute_per_sample_grads
  2. Apply L2 clip + Gaussian noise via GaussianMechanism.clip_and_noise
  3. Apply SGD step
The vectorized nested-vmap path must produce the same final params row-by-row
under a deterministic noise generator (same RNG state for both paths is the
hard part — we use generators seeded identically on the same call order).
"""

import copy

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from dpfl.core.gaussian_mechanism import GaussianMechanism
from dpfl.core.vectorized_state import ParamShapeSpec, pack_node_params
from dpfl.core.vectorized_trainer import train_all_dpsgd_per_step
from dpfl.models.mlp_model import MLP


class _Node:
    def __init__(self, model, idx):
        self.model = model
        self.id = idx


class _Pipe:
    """Fixed-batch pipeline (no internal RNG) for deterministic comparison."""
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


def _per_client_dpsgd(model, X, Y, B, lr, epochs, clip, z, gen):
    """Reference: legacy-style DP-SGD on a single client; uses the same gen
    as the vectorized batched path so noise sequence matches per-step.
    """
    initial = model.get_flat_params().clone()
    n = X.shape[0] // B

    def per_sample_loss(params, x, y):
        out = functional_call(model, params, (x.unsqueeze(0),))
        return F.cross_entropy(out, y.unsqueeze(0))

    inner = vmap(grad(per_sample_loss), in_dims=(None, 0, 0))

    for _ in range(epochs):
        for b in range(n):
            s, e = b * B, (b + 1) * B
            x_b, y_b = X[s:e], Y[s:e]
            params = dict(model.named_parameters())
            grads_dict = inner(params, x_b, y_b)
            # Flatten to (B, D) using same iteration order as get_flat_params
            per_sample_BD = torch.cat(
                [grads_dict[name].reshape(B, -1) for name, _ in model.named_parameters()],
                dim=1,
            )
            # Clip per-row
            norms = per_sample_BD.norm(2, dim=1, keepdim=True)
            cf = torch.clamp(clip / (norms + 1e-12), max=1.0)
            clipped = per_sample_BD * cf
            avg = clipped.mean(dim=0)
            noise = torch.randn(avg.shape, generator=gen,
                                device=avg.device, dtype=avg.dtype)
            avg = avg + noise * (z * clip / B)
            # Apply gradient
            offset = 0
            with torch.no_grad():
                for p in model.parameters():
                    numel = p.numel()
                    p -= lr * avg[offset:offset + numel].view(p.shape)
                    offset += numel

    return model.get_flat_params() - initial


# ── Tests ──────────────────────────────────────────────────────────────────

def test_dpsgd_per_step_zero_noise_matches_per_client():
    """With noise_mult=0, vectorized DP-SGD = per-client DP-SGD = clipped SGD."""
    torch.manual_seed(11)
    N, n_per, B, lr, epochs = 3, 16, 4, 0.05, 1
    clip, z = 1.0, 0.0
    base = MLP(784, 8, 10)
    spec = ParamShapeSpec(base)

    models = []
    for k in range(N):
        torch.manual_seed(300 + k)
        models.append(MLP(784, 8, 10))

    X = torch.randn(N, n_per, 1, 28, 28)
    Y = torch.randint(0, 10, (N, n_per))

    # Reference: per-client DP-SGD with z=0 (noise has no effect; same gen seed)
    ref_models = [copy.deepcopy(m) for m in models]
    ref_updates = []
    for k in range(N):
        gen = torch.Generator(); gen.manual_seed(0)  # unused since z=0
        ref_updates.append(_per_client_dpsgd(
            ref_models[k], X[k], Y[k], B, lr, epochs, clip, z, gen))
    ref_stack = torch.stack(ref_updates, dim=0)

    # Vectorized
    nodes = [_Node(m, k) for k, m in enumerate(models)]
    params_stack = pack_node_params(nodes)
    pipe = _Pipe(X, Y, list(range(N)))
    mech = GaussianMechanism()
    gen = torch.Generator(); gen.manual_seed(0)
    mech.set_generator(gen)

    upd_vec, steps = train_all_dpsgd_per_step(
        base_model=base, params_stack=params_stack, spec=spec,
        train_pipeline=pipe, mechanism=mech,
        batch_size=B, epochs=epochs, lr=lr,
        clip_bound=clip, noise_mult=z,
    )

    for k in range(N):
        diff = (upd_vec[k] - ref_stack[k]).abs().max().item()
        assert diff < 1e-4, f"client {k}: |Δ| = {diff:.2e}"

    assert all(s == n_per // B * epochs for s in steps.values())


def test_dpsgd_per_step_clip_bounds_update_norm():
    """With small clip and z=0, per-row update norm ≤ lr * clip * n_steps (loose bound)."""
    torch.manual_seed(22)
    N, n_per, B, lr, epochs = 4, 8, 4, 0.1, 1
    clip, z = 0.5, 0.0
    base = MLP(784, 8, 10)
    spec = ParamShapeSpec(base)

    models = [copy.deepcopy(base) for _ in range(N)]
    # Push into chaotic loss landscape so unclipped grads would be huge
    for m in models:
        with torch.no_grad():
            for p in m.parameters():
                p.mul_(50.0)

    X = torch.randn(N, n_per, 1, 28, 28)
    Y = torch.randint(0, 10, (N, n_per))

    nodes = [_Node(m, k) for k, m in enumerate(models)]
    pstack = pack_node_params(nodes)
    pipe = _Pipe(X, Y, list(range(N)))
    mech = GaussianMechanism()
    mech.set_generator(torch.Generator())

    upd_vec, _ = train_all_dpsgd_per_step(
        base_model=base, params_stack=pstack, spec=spec,
        train_pipeline=pipe, mechanism=mech,
        batch_size=B, epochs=epochs, lr=lr,
        clip_bound=clip, noise_mult=z,
    )

    n_steps = (n_per // B) * epochs
    bound = lr * clip * n_steps + 1e-3
    for k in range(N):
        u_norm = upd_vec[k].norm(2).item()
        assert u_norm <= bound, (
            f"client {k} update norm {u_norm:.3f} exceeded clip bound {bound:.3f}")


def test_dpsgd_per_step_independent_noise_across_clients():
    """Per-client noise must be independent: with z>0, two identical client
    setups should produce DIFFERENT updates (independent noise draws).
    """
    torch.manual_seed(33)
    N, n_per, B, lr, epochs = 2, 8, 4, 0.05, 1
    clip, z = 1.0, 0.5
    base = MLP(784, 8, 10)
    spec = ParamShapeSpec(base)

    # Both clients identical: same params, same data
    m0 = copy.deepcopy(base)
    m1 = copy.deepcopy(base)
    X0 = torch.randn(n_per, 1, 28, 28)
    Y0 = torch.randint(0, 10, (n_per,))
    X = torch.stack([X0, X0], dim=0)
    Y = torch.stack([Y0, Y0], dim=0)

    nodes = [_Node(m0, 0), _Node(m1, 1)]
    pstack = pack_node_params(nodes)
    pipe = _Pipe(X, Y, [0, 1])
    mech = GaussianMechanism()
    mech.set_generator(torch.Generator().manual_seed(99))

    upd, _ = train_all_dpsgd_per_step(
        base_model=base, params_stack=pstack, spec=spec,
        train_pipeline=pipe, mechanism=mech,
        batch_size=B, epochs=epochs, lr=lr,
        clip_bound=clip, noise_mult=z,
    )

    # Same clean grads + INDEPENDENT noise -> updates should differ
    diff = (upd[0] - upd[1]).abs().max().item()
    assert diff > 1e-3, (
        f"clients with identical input got identical noised updates "
        f"({diff:.2e}) — RNG draws may be coupled across the N axis")
