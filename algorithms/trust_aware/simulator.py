"""Trust-Aware DFL simulator: 4-phase per-round loop for D2B-DP algorithm."""

import time
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dpfl.core.base_simulator import BaseSimulator
from dpfl.algorithms.trust_aware.node import TrustAwareNode

logger = logging.getLogger(__name__)


class TrustAwareDFLSimulator(BaseSimulator):
    """Orchestrator for Trust-Aware D2B-DP: train -> noise -> evaluate -> filter."""

    def __init__(self, config, trust_config, dataset_cls, model_cls,
                 noise_mechanism, aggregator, attack,
                 adaptive_clipper, bounded_noise, per_neighbor_accountant,
                 tracker=None, device=None):
        super().__init__(config, dataset_cls, model_cls,
                         noise_mechanism, aggregator, attack,
                         accountant=None, tracker=tracker, device=device)
        self.tc = trust_config
        self.clipper = adaptive_clipper
        self.bounded_noise = bounded_noise
        self.rdp = per_neighbor_accountant

    def _create_node(self, node_id, model, data, is_attacker):
        node = TrustAwareNode(node_id, model, data, is_attacker, self.tc)
        return node

    def setup(self):
        """Standard setup + initialize per-neighbor trust scores."""
        super().setup()
        alpha_list = self.config.dp.accountant_params.get(
            "alpha_list", [1.25, 1.5, 2, 3, 5, 10, 20, 50, 100])
        for node in self.nodes.values():
            node.init_trust(node.neighbors, self.tc.trust_init, alpha_list)

    def run(self):
        """Main loop: 4-phase D2B-DP per round."""
        n_workers = self.config.training.n_workers
        logger.info("Training mode: %s (n_workers=%d)",
                     "PARALLEL" if n_workers > 1 else "SEQUENTIAL", n_workers)
        for t in range(self.config.training.n_rounds):
            # Phase 1: Train (parallel if n_workers>1) + Adaptive Clip
            t0 = time.time()
            attack_active = t >= self.config.attack.start_round

            def _train_node(node):
                atk = self.attack if (node.is_attacker and attack_active) else None
                update, _ = node.compute_update(
                    self.trainer, self.noise_mechanism, atk, apply_noise=False)
                return node.id, update

            raw_updates = {}
            if n_workers > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    for nid, upd in pool.map(_train_node, self.nodes.values()):
                        raw_updates[nid] = upd
            else:
                for node in self.nodes.values():
                    nid, upd = _train_node(node)
                    raw_updates[nid] = upd

            # Adaptive clip (sequential — cheap, updates per-node state)
            updates = {}
            for node in self.nodes.values():
                update = raw_updates[node.id]
                raw_norm = update.norm(2).item()
                node.clip_history.append(raw_norm)
                thresh = self.clipper.get_threshold(node.clip_history)
                if thresh is not None:
                    update = self.clipper.clip(update, thresh)
                updates[node.id] = update
            logger.debug("Round %d Phase 1 (train+clip): %.2fs, %d nodes, workers=%d",
                         t, time.time() - t0, len(self.nodes), n_workers)

            # Phase 2: Outbound — per-edge noise injection
            edge_updates = {}
            for node in self.nodes.values():
                edge_updates[node.id] = {}
                clip_e = (self.clipper.get_threshold(node.clip_history)
                          or updates[node.id].norm(2).item())
                for j in node.neighbors:
                    rho_base = min(
                        (1 + self.tc.beta * t) * self.tc.rho_min, self.tc.rho_max)
                    rho_ij = rho_base  # trust only affects aggregation, not noise
                    sigma_sq = self.bounded_noise.compute_noise_variance(
                        clip_e, node.n_samples, rho_ij)
                    self.rdp.step(node, j, clip_e, sigma_sq)
                    if not self.rdp.can_send(node, j):
                        edge_updates[node.id][j] = None
                        continue
                    eps_step = self.rdp.get_step_epsilon(clip_e, sigma_sq)
                    edge_updates[node.id][j] = self.bounded_noise.add_bounded_noise(
                        updates[node.id], sigma_sq, eps_step)

            # Phase 3+4: Inbound eval + filter + aggregate
            attack_active = t >= self.config.attack.start_round
            tp_all = fp_all = fn_all = tn_all = 0
            per_node_detection = {}
            node_agg_metrics = {}
            for node in self.nodes.values():
                received = {
                    j: edge_updates[j][node.id]
                    for j in node.neighbors
                    if edge_updates.get(j, {}).get(node.id) is not None}
                result = self.aggregator.aggregate(
                    updates[node.id], node.model.get_flat_params(), received,
                    node.trust_scores, node.z_score_buffer, node.trust_buffer)
                node.model.set_flat_params(result.new_params)
                node_agg_metrics[node.id] = result.node_metrics
                tp, fp, fn, tn = self._compute_detection(
                    result.flagged_ids, result.clean_ids, node.neighbors,
                    attack_active=attack_active)
                per_node_detection[node.id] = (tp, fp, fn, tn)
                tp_all += tp; fp_all += fp; fn_all += fn; tn_all += tn

            # Privacy accounting: max epsilon across all honest edges
            eps_vals = [
                self.rdp.get_epsilon(node, j)
                for node in self.nodes.values()
                if node.id not in self.attacker_ids
                for j in node.neighbors]
            max_eps = max(eps_vals) if eps_vals else 0.0

            # Trust diagnostics for extra metrics
            t_h, t_a = [], []
            for n in self.nodes.values():
                if n.id in self.attacker_ids:
                    continue
                for j, tv in n.trust_scores.items():
                    (t_a if j in self.attacker_ids else t_h).append(tv)

            self._log_round(
                t, max_eps, updates, per_node_detection, node_agg_metrics,
                tp_all, fp_all, fn_all, tn_all,
                extra_round_metrics={
                    "trust_toward_honest": float(np.mean(t_h)) if t_h else 0.0,
                    "trust_toward_attacker": float(np.mean(t_a)) if t_a else 0.0,
                })
