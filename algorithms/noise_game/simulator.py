"""Noise-game DFL simulator: orchestrates the full strategic noise algorithm."""

import logging

import numpy as np
import torch

from dpfl.core.base_simulator import BaseSimulator
from dpfl.algorithms.noise_game.node import NoiseGameNode

logger = logging.getLogger(__name__)


class NoiseGameDFLSimulator(BaseSimulator):
    """Strategic Noise Game DFL.

    Algorithm per round (Section 5 of noise-game.md):
      Phase 1: Local training + L2-norm clip
      Phase 2: Noise-game pipeline (SCAFFOLD, strategic noise, alignment, EMA, momentum)
      Phase 3: Aggregation
      Phase 4: Privacy accounting
      Phase 5: Evaluate + log
    """

    def __init__(self, config, ng_config, dataset_cls, model_cls,
                 noise_mechanism, game_mechanism, aggregator, attack,
                 accountant=None, tracker=None, device=None):
        super().__init__(config, dataset_cls, model_cls,
                         noise_mechanism, aggregator, attack,
                         accountant=accountant, tracker=tracker, device=device)
        self.ng = ng_config
        self.game_mechanism = game_mechanism
        # Wire isolated RNG into the strategic noise mechanism.
        if hasattr(self.game_mechanism, "set_generator"):
            self.game_mechanism.set_generator(self.noise_gen)

    def _create_node(self, node_id, model, data, is_attacker):
        node = NoiseGameNode(node_id, model, data, is_attacker, self.ng)
        node.init_buffers(self.param_dim)
        return node

    def run(self):
        """Main loop: T rounds of noise-game DFL."""
        for t in range(self.config.training.n_rounds):
            # Step 0: Deterministic Poisson client subsampling.
            active_ids = self._sample_active_nodes(t)

            # Phase 1: Local training (no noise — game handles it)
            raw_updates, all_steps = self._train_all_nodes(apply_noise=False, round_t=t)

            # L2-norm clip each update
            C = self.config.dp.clip_bound
            clipped = {}
            for nid, upd in raw_updates.items():
                norm = upd.norm()
                factor = min(1.0, C / (norm + 1e-12))
                clipped[nid] = upd * factor

            # Phase 2: Noise-game pipeline (active honest nodes only).
            # Inactive honest nodes don't contribute updates this round.
            # Attackers still contribute their clipped updates (always active).
            final_updates = {}
            extra_node = {}
            sigma_dps = {}
            for nid, node in self.nodes.items():
                g = clipped[nid]
                if node.is_attacker:
                    final_updates[nid] = g
                    continue
                if nid not in active_ids:
                    continue  # inactive honest: no update contributed

                # SCAFFOLD variance reduction
                if self.ng.scaffold:
                    global_c = self._neighbor_avg_control(node)
                    g = node.apply_scaffold(g, global_c)

                # 4-layer noise injection (DP + strategic)
                noise, metrics = self.game_mechanism.compute_total_noise(
                    g, node.prev_gradient, round_t=t)
                g_hat = g + noise

                # NSR warning
                if metrics.get("nsr", 0) > self.ng.nsr_warn:
                    logger.warning("Node %d round %d: NSR=%.2f exceeds %.1f",
                                   nid, t, metrics["nsr"], self.ng.nsr_warn)

                # Gradient alignment filtering
                if not node.check_alignment(g_hat):
                    g_hat = node.ema_gradient.clone()

                # EMA denoising + Momentum
                node.update_ema(g_hat)
                m = node.update_momentum(node.ema_gradient)

                # Trust-aware LR scaling
                trust = metrics["trust"]
                final_updates[nid] = m * max(trust, 0.0)

                # Store state for next round
                node.store_gradient(clipped[nid])
                if self.ng.scaffold:
                    node.update_control_variate(clipped[nid])

                sigma_dps[nid] = metrics["sigma_dp"]
                extra_node[nid] = {
                    "trust": metrics["trust"],
                    "noise_norm": metrics["total_noise_norm"],
                    "nsr": metrics["nsr"],
                    "sigma_dp": metrics["sigma_dp"],
                }

            # Phase 3: Aggregation (active nodes only)
            attack_active = t >= self.config.attack.start_round
            total_tp = total_fp = total_fn = total_tn = 0
            per_node_detection = {nid: (0, 0, 0, 0) for nid in self.nodes}
            node_agg_metrics = {nid: {} for nid in self.nodes}

            for node in self.nodes.values():
                if node.id not in active_ids:
                    continue  # inactive: skip aggregation
                if node.id not in final_updates:
                    continue  # safety: no final update produced (shouldn't happen)
                neighbor_upds = {
                    j: final_updates[j] for j in node.neighbors
                    if j in final_updates and j in active_ids}
                result = self.aggregator.aggregate(
                    final_updates[node.id], node.model.get_flat_params(), neighbor_upds)
                node_agg_metrics[node.id] = result.node_metrics

                # Two-track model update
                if self.ng.two_track and not node.is_attacker:
                    agg_update = result.new_params - node.model.get_flat_params()
                    node.update_two_track(clipped[node.id], agg_update)
                else:
                    node.model.set_flat_params(result.new_params)

                tp, fp, fn, tn = self._compute_detection(
                    result.flagged_ids, result.clean_ids, node.neighbors,
                    attack_active=attack_active)
                per_node_detection[node.id] = (tp, fp, fn, tn)
                total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn

            # Phase 4: Privacy accounting
            epsilon = 0.0
            if self.accountant is not None:
                honest_steps = next(
                    s for nid, s in all_steps.items() if nid not in self.attacker_ids)
                q_batch = self.config.training.batch_size / self.nodes[
                    self.config.topology.n_attackers].n_samples
                avg_sigma_dp = (np.mean(list(sigma_dps.values()))
                                if sigma_dps else self.game_mechanism.sigma_0)
                effective_mult = max(avg_sigma_dp / (C + 1e-12), 0.01)
                self.accountant.step(honest_steps, q_batch, effective_mult)
                epsilon = self.accountant.get_epsilon()

            # Phase 5: Evaluate + log
            honest_trust = [m["trust"] for m in extra_node.values()]
            honest_nsr = [m["nsr"] for m in extra_node.values()]
            self._log_round(
                t, epsilon, final_updates, per_node_detection, node_agg_metrics,
                total_tp, total_fp, total_fn, total_tn,
                extra_node_data=extra_node,
                extra_round_metrics={
                    "avg_trust": float(np.mean(honest_trust)) if honest_trust else 0.0,
                    "avg_sigma_dp": float(avg_sigma_dp) if sigma_dps else 0.0,
                    "avg_nsr": float(np.mean(honest_nsr)) if honest_nsr else 0.0,
                })

            if self.accountant is not None and epsilon > self.config.dp.epsilon_max:
                logger.warning("Round %3d/%d | Budget exceeded (eps=%.2f)",
                               t + 1, self.config.training.n_rounds, epsilon)
                break

    def _neighbor_avg_control(self, node: NoiseGameNode) -> torch.Tensor:
        """D-SCAFFOLD: approximate global control as average of neighbors' c_i."""
        controls = [self.nodes[j].control_variate for j in node.neighbors
                    if self.nodes[j].control_variate is not None]
        if not controls:
            return torch.zeros_like(node.control_variate)
        return torch.stack(controls).mean(dim=0)
