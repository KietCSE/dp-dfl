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
            # Step 1: All nodes train with DP-SGD noise (apply_noise=True)
            updates, all_steps = self._train_all_nodes(apply_noise=True, round_t=t)

            # Step 2: Aggregate with defense filtering
            total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
            per_node_detection = {}
            node_agg_metrics = {}

            for node in self.nodes.values():
                neighbor_updates = {j: updates[j] for j in node.neighbors}
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
            epsilon = 0.0
            if self.accountant is not None:
                honest_steps = next(
                    s for nid, s in all_steps.items() if nid not in self.attacker_ids)
                q_batch = self.config.training.batch_size / self.nodes[
                    self.config.topology.n_attackers].n_samples
                self.accountant.step(honest_steps, q_batch, self.config.dp.noise_mult)
                epsilon = self.accountant.get_epsilon()

            # Step 4: Evaluate + log
            self._log_round(t, epsilon, updates, per_node_detection,
                            node_agg_metrics, total_tp, total_fp, total_fn, total_tn)

            if self.accountant is not None and epsilon > self.config.dp.epsilon_max:
                logger.warning("Round %3d/%d | Budget exceeded (eps=%.2f)",
                               t + 1, self.config.training.n_rounds, epsilon)
                break
