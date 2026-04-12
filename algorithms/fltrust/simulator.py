"""FLTrust DFL simulator: computes root gradient before aggregation."""

import logging

from dpfl.algorithms.dpsgd_kurtosis.simulator import DFLSimulator
from dpfl.core.base_node import Node

logger = logging.getLogger(__name__)


class FLTrustSimulator(DFLSimulator):
    """DFL simulator with FLTrust root gradient computation."""

    def __init__(self, *args, root_data_ratio: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_data_ratio = root_data_ratio

    def setup(self):
        super().setup()
        # Split root data for each node
        for node in self.nodes.values():
            node.split_root_data(self.root_data_ratio)

    def run(self):
        for t in range(self.config.training.n_rounds):
            # Phase 1: Train all nodes
            updates, steps = self._train_all_nodes(
                apply_noise=self.config.dp.noise_mode != "none",
                round_t=t,
            )

            # Phase 2: Compute root gradients + aggregate per node
            total_tp = total_fp = total_fn = total_tn = 0
            per_node_det = {}
            node_agg_metrics = {}

            for node in self.nodes.values():
                if node.id not in updates:
                    continue

                root_grad = self._compute_root_gradient(node)

                nbr_updates = {
                    nid: updates[nid]
                    for nid in node.neighbors
                    if nid in updates
                }

                result = self.aggregator.aggregate(
                    own_update=updates[node.id],
                    own_params=node.model.get_flat_params(),
                    neighbor_updates=nbr_updates,
                    root_gradient=root_grad,
                )

                node.model.set_flat_params(result.new_params)
                node_agg_metrics[node.id] = result.node_metrics

                tp, fp, fn, tn = self._compute_detection(
                    result.flagged_ids, result.clean_ids, node.neighbors
                )
                per_node_det[node.id] = (tp, fp, fn, tn)
                total_tp += tp; total_fp += fp
                total_fn += fn; total_tn += tn

            # Phase 3: Privacy accounting
            epsilon = 0.0
            if self.accountant and self.config.dp.noise_mode != "none":
                honest_steps = [
                    steps[nid] for nid in steps
                    if not self.nodes[nid].is_attacker
                ]
                if honest_steps:
                    avg_steps = sum(honest_steps) / len(honest_steps)
                    q = min(self.config.training.batch_size /
                            min(n.n_samples for n in self.nodes.values()
                                if not n.is_attacker), 1.0)
                    self.accountant.step(
                        int(avg_steps), q, self.config.dp.noise_mult
                    )
                    epsilon = self.accountant.get_epsilon()
                    if epsilon > self.config.dp.epsilon_max:
                        logger.warning("Round %3d/%d | Budget exceeded (eps=%.2f)",
                                       t + 1, self.config.training.n_rounds, epsilon)
                        break

            # Phase 4: Log
            self._log_round(
                t, epsilon, updates, per_node_det, node_agg_metrics,
                total_tp, total_fp, total_fn, total_tn,
            )

    def _compute_root_gradient(self, node: Node):
        """Train on node's root data to get root gradient."""
        if node.root_data is None or len(node.root_data) == 0:
            logger.debug("Node %d: no root data, returning zero gradient", node.id)
            return node.model.get_flat_params() * 0

        old_params = node.model.get_flat_params().clone()
        update, _ = self.trainer.train(
            node.model, node.root_data, self.noise_mechanism,
            apply_noise=False,
        )
        node.model.set_flat_params(old_params)
        logger.debug("Node %d: root gradient norm=%.4f", node.id, update.norm().item())
        return update
