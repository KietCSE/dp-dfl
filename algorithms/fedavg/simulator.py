"""FedAvg simulator: dataset-proportional weighted aggregation."""

import logging

from dpfl.core.base_simulator import BaseSimulator
from dpfl.core.base_node import Node

logger = logging.getLogger(__name__)


class FedAvgSimulator(BaseSimulator):
    """FedAvg with dataset-proportional weighted aggregation.

    Differences from DFLSimulator:
      - setup() passes data_sizes to aggregator for weighted averaging
      - aggregate() passes own_node_id so aggregator knows each node's weight
    """

    def _create_node(self, node_id, model, data, is_attacker) -> Node:
        return Node(node_id, model, data, is_attacker)

    def setup(self):
        super().setup()
        # Pass dataset sizes to aggregator for FedAvg weighting
        if hasattr(self.aggregator, 'set_data_sizes'):
            sizes = {nid: node.n_samples for nid, node in self.nodes.items()}
            self.aggregator.set_data_sizes(sizes)
            logger.info("FedAvg data_sizes set: min=%d, max=%d, total=%d",
                        min(sizes.values()), max(sizes.values()), sum(sizes.values()))

    def run(self):
        """Main loop: train -> weighted aggregate -> account -> log."""
        for t in range(self.config.training.n_rounds):
            attack_active = t >= self.config.attack.start_round
            # Step 0: Deterministic Poisson client subsampling.
            active_ids = self._sample_active_nodes(t)

            updates, all_steps = self._train_all_nodes(apply_noise=True, round_t=t)

            total_tp = total_fp = total_fn = total_tn = 0
            per_node_detection = {nid: (0, 0, 0, 0) for nid in self.nodes}
            node_agg_metrics = {nid: {} for nid in self.nodes}

            for node in self.nodes.values():
                if node.id not in active_ids:
                    continue  # inactive this round
                neighbor_updates = {
                    j: updates[j] for j in node.neighbors
                    if j in updates and j in active_ids}
                result = self.aggregator.aggregate(
                    updates[node.id], node.model.get_flat_params(),
                    neighbor_updates, own_node_id=node.id)
                node.model.set_flat_params(result.new_params)
                node_agg_metrics[node.id] = result.node_metrics

                tp, fp, fn, tn = self._compute_detection(
                    result.flagged_ids, result.clean_ids, node.neighbors,
                    attack_active=attack_active)
                per_node_detection[node.id] = (tp, fp, fn, tn)
                total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn

            # Privacy accounting
            epsilon = 0.0
            if self.accountant is not None:
                if self.config.dp.noise_mode == "post_training":
                    # User-level: 1 noise application per round, no subsampling
                    # RDP cost: α / (2z²) per round
                    self.accountant.step(1, 1.0, self.config.dp.noise_mult)
                else:
                    # per_step: K steps with subsampling q = B/n
                    honest_steps = next(
                        s for nid, s in all_steps.items()
                        if nid not in self.attacker_ids)
                    q_batch = self.config.training.batch_size / self.nodes[
                        self.config.topology.n_attackers].n_samples
                    self.accountant.step(
                        honest_steps, q_batch, self.config.dp.noise_mult)
                epsilon = self.accountant.get_epsilon()

            self._log_round(t, epsilon, updates, per_node_detection,
                            node_agg_metrics,
                            total_tp, total_fp, total_fn, total_tn)

            if self.accountant is not None and epsilon > self.config.dp.epsilon_max:
                logger.warning("Round %3d/%d | Budget exceeded (eps=%.2f)",
                               t + 1, self.config.training.n_rounds, epsilon)
                break
