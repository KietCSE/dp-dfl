"""Standard DFL simulator: DP-SGD + Kurtosis defense."""

import logging

from dpfl.core.base_simulator import BaseSimulator
from dpfl.core.base_node import Node

logger = logging.getLogger(__name__)


class DFLSimulator(BaseSimulator):
    """Standard DP-SGD DFL with kurtosis-based attack detection."""

    def _create_node(self, node_id, model, data, is_attacker) -> Node:
        return Node(node_id, model, data, is_attacker)

    def run(self):
        """Main loop: train (with DP noise) → aggregate → account → log."""
        for t in range(self.config.training.n_rounds):
            attack_active = t >= self.config.attack.start_round
            # Step 0: Deterministic Poisson client subsampling (q = dp.sampling_rate).
            # q=1.0 => all nodes active (backward-compat). Same seed+t yields
            # identical active_ids across algorithms for fair comparison.
            active_ids = self._sample_active_nodes(t)

            # Step 1: All nodes train with DP-SGD noise (apply_noise=True)
            updates, all_steps = self._train_all_nodes(apply_noise=True, round_t=t)

            # Step 2: Aggregate with defense filtering (active nodes only).
            # Init detection/metrics for ALL nodes so _log_round iterates without KeyError.
            total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
            per_node_detection = {nid: (0, 0, 0, 0) for nid in self.nodes}
            node_agg_metrics = {nid: {} for nid in self.nodes}

            for node in self.nodes.values():
                if node.id not in active_ids:
                    continue  # inactive this round: skip aggregation
                neighbor_updates = {
                    j: updates[j] for j in node.neighbors
                    if j in updates and j in active_ids}
                result = self.aggregator.aggregate(
                    updates[node.id], node.model.get_flat_params(), neighbor_updates)
                node.model.set_flat_params(result.new_params)
                node_agg_metrics[node.id] = result.node_metrics

                tp, fp, fn, tn = self._compute_detection(
                    result.flagged_ids, result.clean_ids, node.neighbors,
                    attack_active=attack_active)
                per_node_detection[node.id] = (tp, fp, fn, tn)
                total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn

            # Step 3: Privacy accounting
            # NOTE: FLAME aggregator adds its own DP noise internally; using
            # noise_mode=post_training with FLAME would double-count noise.
            # Out-of-scope fix here — keep FLAME on per_step / none.
            epsilon = 0.0
            if self.accountant is not None:
                # Client-level Poisson subsampling rate (q_client). Compose
                # with batch subsampling for per_step mode per RDP standard.
                q_client = max(min(float(self.config.dp.sampling_rate), 1.0), 0.0)
                if self.config.dp.noise_mode == "post_training":
                    # User-level DP: 1 noise application per round at q_client.
                    self.accountant.step(1, q_client, self.config.dp.noise_mult)
                else:
                    # per_step DP-SGD: K steps with composed q = q_client * q_batch
                    honest_steps = next(
                        s for nid, s in all_steps.items() if nid not in self.attacker_ids)
                    q_batch = self.config.training.batch_size / self.nodes[
                        self.config.topology.n_attackers].n_samples
                    q_composed = q_client * q_batch
                    self.accountant.step(honest_steps, q_composed, self.config.dp.noise_mult)
                epsilon = self.accountant.get_epsilon()

            # Step 4: Evaluate + log
            self._log_round(t, epsilon, updates, per_node_detection,
                            node_agg_metrics, total_tp, total_fp, total_fn, total_tn)

            if self.accountant is not None and epsilon > self.config.dp.epsilon_max:
                logger.warning("Round %3d/%d | Budget exceeded (eps=%.2f)",
                               t + 1, self.config.training.n_rounds, epsilon)
                break
