"""DP-SGD local trainer: per-sample gradient via torch.func.vmap + clip + noise."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from typing import Tuple

from dpfl.config import TrainingConfig, DPConfig
from dpfl.models.base_model import BaseModel
from dpfl.privacy.base_noise_mechanism import BaseNoiseMechanism


class DPSGDTrainer:
    """Local DP-SGD training for a single node."""

    def __init__(self, config: TrainingConfig, dp_config: DPConfig, device=None):
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.epochs = config.local_epochs
        self.clip_bound = dp_config.clip_bound
        self.noise_mult = dp_config.noise_mult
        self.device = device or torch.device("cpu")

    def train(self, model: BaseModel, dataset: Subset,
              noise_mechanism: BaseNoiseMechanism,
              apply_noise: bool = True) -> Tuple[torch.Tensor, int]:
        """
        DP-SGD local training.
        Returns: (model_update_flat, n_steps)
        apply_noise=False -> clip only, no noise (for attackers).
        """
        initial_params = model.get_flat_params().clone()
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=True, drop_last=True)
        n_steps = 0

        for _ in range(self.epochs):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                per_sample_grads = self._compute_per_sample_grads(model, x_batch, y_batch)

                if apply_noise:
                    noised_avg = noise_mechanism.clip_and_noise(
                        per_sample_grads, self.clip_bound, self.noise_mult, self.batch_size
                    )
                else:
                    clipped = noise_mechanism.clip(per_sample_grads, self.clip_bound)
                    noised_avg = clipped.mean(dim=0)

                self._apply_gradient(model, noised_avg)
                n_steps += 1

        update = model.get_flat_params() - initial_params
        return update, n_steps

    @staticmethod
    def _compute_per_sample_grads(model: BaseModel, x_batch: torch.Tensor,
                                   y_batch: torch.Tensor) -> torch.Tensor:
        """Per-sample gradients via torch.func.vmap + grad. Returns (B, D)."""
        params = dict(model.named_parameters())

        def compute_loss(params, x, y):
            out = torch.func.functional_call(model, params, (x.unsqueeze(0),))
            return F.cross_entropy(out, y.unsqueeze(0))

        grads_dict = torch.func.vmap(
            torch.func.grad(compute_loss), in_dims=(None, 0, 0)
        )(params, x_batch, y_batch)

        # Flatten dict of (B, *shape) -> (B, D)
        return torch.cat([g.flatten(1) for g in grads_dict.values()], dim=1)

    def _apply_gradient(self, model: BaseModel, flat_grad: torch.Tensor):
        """Apply flat gradient to model params via SGD."""
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.data -= self.lr * flat_grad[offset:offset + numel].view(p.shape)
            offset += numel
