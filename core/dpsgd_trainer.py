"""DP-SGD local trainer: supports per-step DP, post-training DP, and plain SGD."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from typing import Tuple

from dpfl.config import TrainingConfig, DPConfig
from dpfl.models.base_model import BaseModel
from dpfl.core.base_noise_mechanism import BaseNoiseMechanism


class DPSGDTrainer:
    """Local training for a single node. Noise injection timing controlled by noise_mode."""

    def __init__(self, config: TrainingConfig, dp_config: DPConfig, device=None,
                 data_gen: "torch.Generator" = None):
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.epochs = config.local_epochs
        self.clip_bound = dp_config.clip_bound
        self.noise_mult = dp_config.noise_mult
        self.noise_mode = dp_config.noise_mode
        self.device = device or torch.device("cpu")
        # Isolated RNG for DataLoader shuffle. If None, falls back to global
        # torch RNG (legacy behavior).
        self.data_gen = data_gen

    def train(self, model: BaseModel, dataset: Subset,
              noise_mechanism: BaseNoiseMechanism,
              apply_noise: bool = True,
              skip_dp: bool = False) -> Tuple[torch.Tensor, int]:
        """
        Local training. Returns: (model_update_flat, n_steps).
        noise_mode controls when DP is applied:
          - per_step: clip+noise every batch (standard DP-SGD)
          - post_training: standard SGD, then clip+noise on final update
          - none: plain SGD, no DP
        skip_dp=True -> plain SGD, bypass all DP (for model poisoning attackers).
        """
        if skip_dp:
            return self._train_standard_sgd(model, dataset)

        if self.noise_mode == "per_step":
            return self._train_dpsgd_per_step(model, dataset, noise_mechanism, apply_noise)

        update, n_steps = self._train_standard_sgd(model, dataset)

        if self.noise_mode == "post_training":
            update = self._apply_post_training_dp(update, noise_mechanism, apply_noise)

        return update, n_steps

    # -- per-step DP-SGD (vmap per-sample grads) --

    def _train_dpsgd_per_step(self, model: BaseModel, dataset: Subset,
                              noise_mechanism: BaseNoiseMechanism,
                              apply_noise: bool) -> Tuple[torch.Tensor, int]:
        """Per-batch: per-sample grads -> clip -> noise -> SGD update."""
        initial_params = model.get_flat_params().clone()
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=True, drop_last=True,
                            generator=self.data_gen)
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

    # -- standard SGD (no per-sample grads, no DP) --

    def _train_standard_sgd(self, model: BaseModel,
                            dataset: Subset) -> Tuple[torch.Tensor, int]:
        """Standard mini-batch SGD without per-sample clipping."""
        initial_params = model.get_flat_params().clone()
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=True, drop_last=True,
                            generator=self.data_gen)
        n_steps = 0

        for _ in range(self.epochs):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                out = model(x_batch)
                loss = F.cross_entropy(out, y_batch)
                model.zero_grad()
                loss.backward()

                with torch.no_grad():
                    for p in model.parameters():
                        p -= self.lr * p.grad

                n_steps += 1

        update = model.get_flat_params() - initial_params
        return update, n_steps

    # -- post-training DP: clip + noise on final update --

    def _apply_post_training_dp(self, update: torch.Tensor,
                                noise_mechanism: BaseNoiseMechanism,
                                apply_noise: bool) -> torch.Tensor:
        """Clip the final update; add noise if apply_noise=True."""
        # clip() expects (B, D) -> unsqueeze to (1, D)
        clipped = noise_mechanism.clip(update.unsqueeze(0), self.clip_bound).squeeze(0)
        if apply_noise:
            # batch_size=1: sensitivity = C for one clipped update, sigma = z*C
            return noise_mechanism.add_noise(clipped, self.clip_bound,
                                             self.noise_mult, batch_size=1)
        return clipped

    # -- shared utilities --

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

        return torch.cat([g.flatten(1) for g in grads_dict.values()], dim=1)

    def _apply_gradient(self, model: BaseModel, flat_grad: torch.Tensor):
        """Apply flat gradient to model params via SGD."""
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.data -= self.lr * flat_grad[offset:offset + numel].view(p.shape)
            offset += numel
