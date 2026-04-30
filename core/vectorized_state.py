"""Pack/unpack utilities for vectorized FL: (N, D) tensor <-> per-client param dicts.

The (N, D) layout matches BaseModel.get_flat_params() iteration order so a row
can be set back via BaseModel.set_flat_params() without surprises.

Used by vectorized_trainer to feed torch.func.vmap + functional_call:
- to_dict_batched(stack) -> {name: Tensor[N, *shape]} for vmap input
- from_dict_batched(dict) -> (N, D) for stacking results back
"""

from typing import Dict, List, Tuple

import torch

from dpfl.models.base_model import BaseModel


class ParamShapeSpec:
    """Captures parameter layout for vectorized ops.

    Records (name, shape, slice) per param in BaseModel.get_flat_params() order.
    Built once from a base_model; reused for all clients (they share architecture).
    """

    def __init__(self, base_model: BaseModel):
        self.specs: List[Tuple[str, torch.Size, slice]] = []
        offset = 0
        for name, p in base_model.named_parameters():
            numel = p.numel()
            self.specs.append((name, p.shape, slice(offset, offset + numel)))
            offset += numel
        self.total = offset

    @property
    def D(self) -> int:
        return self.total

    def to_dict(self, flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """(D,) flat -> {name: Tensor[*shape]} for single-client functional_call."""
        return {name: flat[s].view(shape) for name, shape, s in self.specs}

    def to_dict_batched(self, stack: torch.Tensor) -> Dict[str, torch.Tensor]:
        """(N, D) stack -> {name: Tensor[N, *shape]} for vmap functional_call."""
        N = stack.shape[0]
        return {
            name: stack[:, s].reshape(N, *shape)
            for name, shape, s in self.specs
        }

    def from_dict(self, params_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """{name: Tensor[*shape]} -> (D,) flat. Order matches get_flat_params."""
        return torch.cat([
            params_dict[name].reshape(-1) for name, _, _ in self.specs
        ])

    def from_dict_batched(self, params_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """{name: Tensor[N, *shape]} -> (N, D). Order matches get_flat_params."""
        first_name = self.specs[0][0]
        N = params_dict[first_name].shape[0]
        return torch.cat([
            params_dict[name].reshape(N, -1) for name, _, _ in self.specs
        ], dim=1)


def pack_node_params(nodes, device: torch.device = None) -> torch.Tensor:
    """Stack each node's flat params into (N, D). Caller controls iteration order."""
    flats = [node.model.get_flat_params() for node in nodes]
    stack = torch.stack(flats, dim=0)
    if device is not None and stack.device != device:
        stack = stack.to(device)
    return stack


def unpack_to_nodes(stack: torch.Tensor, nodes) -> None:
    """Write each row of (N, D) back to the matching node.model via set_flat_params.

    Iteration order MUST match the order used in pack_node_params.
    """
    if stack.shape[0] != len(nodes):
        raise ValueError(
            f"Stack rows ({stack.shape[0]}) != n_nodes ({len(nodes)})")
    for i, node in enumerate(nodes):
        node.model.set_flat_params(stack[i])
