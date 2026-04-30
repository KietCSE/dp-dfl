"""CFL-DP-FedAvg simulator: trusted server, central Gaussian noise.

Algorithm (paper-faithful subset of McMahan+ 2018):
  1. Server broadcasts θ_global to active clients (Poisson sample, rate q)
  2. Each client trains local SGD on D_k → returns Δ_k = θ_local - θ_global
  3. Server FlatClips each Δ_k: ‖Δ_k‖₂ ≤ S
  4. Server aggregates (estimator f̃_f, uniform w_k=1):
       Δ = Σ_{k∈C} Δ_k / (qW)         where W = K (n_nodes)
  5. Server adds Gaussian noise:
       σ = z · S / (qW)
       Δ̃ = Δ + N(0, σ² I)
  6. Server updates: θ_global ← θ_global + Δ̃
  7. Privacy accounting: 1 SGM step per round at (q, z)

Trainer config: noise_mode='none' (plain SGD). Server alone owns DP.
"""

import logging
from typing import Optional

import torch

from dpfl.core.base_simulator import BaseSimulator
from dpfl.core.base_node import Node

logger = logging.getLogger(__name__)


class CFLSimulator(BaseSimulator):
    """Centralized FedAvg with user-level Central DP (McMahan+ 2018)."""

    def _create_node(self, node_id, model, data, is_attacker) -> Node:
        return Node(node_id, model, data, is_attacker)

    def _flat_clip(self, delta: torch.Tensor, S: float) -> torch.Tensor:
        """π(Δ, S) = Δ · min(1, S/‖Δ‖) — paper Eq. (1)."""
        norm = delta.norm(2).item()
        scale = min(1.0, S / max(norm, 1e-12))
        return delta * scale

    def run(self):
        """CFL-DP-FedAvg main loop."""
        # Server's global model: snapshot from any node post-setup.
        # All nodes share identical init params via copy.deepcopy in setup().
        theta_global = next(iter(self.nodes.values())) \
            .model.get_flat_params().clone()

        S = float(self.config.dp.clip_bound)
        z = float(self.config.dp.noise_mult)
        q = float(self.config.dp.sampling_rate)
        W = float(len(self.nodes))   # uniform w_k=1, W = K
        apply_dp = (z > 0.0) and (self.accountant is not None)

        logger.info("CFL-DP-FedAvg: K=%d, q=%.3f, S=%.4f, z=%.4f, "
                    "σ=zS/(qW)=%.4f, DP=%s",
                    int(W), q, S, z, z * S / max(q * W, 1e-12), apply_dp)

        for t in range(self.config.training.n_rounds):
            attack_active = t >= self.config.attack.start_round
            active_ids = self._sample_active_nodes(t)

            # Phase 1: Broadcast θ_global to all nodes (incl. inactive — keeps
            # eval consistent across runs; only active will train this round).
            for node in self.nodes.values():
                node.model.set_flat_params(theta_global)

            # Phase 2: Local SGD. apply_noise=False because noise_mode='none' →
            # trainer returns plain SGD update (no client-side clip/noise).
            # Attacker model-poisoning is applied inside compute_update() when
            # round_t >= start_round (handled by _train_all_nodes).
            updates, _ = self._train_all_nodes(apply_noise=False, round_t=t)

            # Phase 3: Server-side FlatClip each Δ_k for k ∈ C^t.
            # Inactive nodes' updates exist but are ignored (not in active_ids).
            clipped: list = []
            for nid in active_ids:
                if nid not in updates:
                    continue
                clipped.append(self._flat_clip(updates[nid], S))

            # Phase 4: Aggregate f̃_f = Σ Δ_k / (qW). Empty batch → no update.
            if clipped:
                weighted_sum = torch.stack(clipped).sum(dim=0)
                delta_aggregate = weighted_sum / max(q * W, 1e-12)
            else:
                delta_aggregate = torch.zeros_like(theta_global)

            # Phase 5: Server adds Gaussian noise σ = zS/(qW). Skip if no DP.
            if apply_dp:
                sigma = z * S / max(q * W, 1e-12)
                noise = torch.randn(
                    delta_aggregate.shape, generator=self.noise_gen,
                    device=delta_aggregate.device, dtype=delta_aggregate.dtype)
                delta_noisy = delta_aggregate + noise * sigma
            else:
                delta_noisy = delta_aggregate

            # Phase 6: Server updates θ_global, then re-broadcasts so
            # downstream eval (_log_round) sees the post-update model on
            # every node — matches CFL semantics of single global model.
            theta_global = theta_global + delta_noisy
            for node in self.nodes.values():
                node.model.set_flat_params(theta_global)

            # Phase 7: Privacy accounting — 1 SGM step at (q, z) per round.
            epsilon = 0.0
            if apply_dp:
                self.accountant.step(1, q, z)
                epsilon = self.accountant.get_epsilon()

            # Phase 8: Logging. CFL has no defense filtering → all detection
            # metrics zero (placeholder); _log_round will still compute
            # accuracy / loss / update_norms across nodes.
            per_node_detection = {nid: (0, 0, 0, 0) for nid in self.nodes}
            node_agg_metrics = {nid: {} for nid in self.nodes}
            self._log_round(
                t, epsilon, updates, per_node_detection, node_agg_metrics,
                0, 0, 0, 0)

            if apply_dp and epsilon > self.config.dp.epsilon_max:
                logger.warning(
                    "Round %d/%d | Budget exceeded (eps=%.2f > %.2f)",
                    t + 1, self.config.training.n_rounds,
                    epsilon, self.config.dp.epsilon_max)
                break
