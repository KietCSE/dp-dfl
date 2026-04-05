"""Main DFL orchestrator: setup nodes, run rounds, track metrics."""

import copy
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn.functional as F
import numpy as np
from math import prod
from typing import Dict, Set, Type

from dpfl.config import ExperimentConfig
from dpfl.data.base_dataset import BaseDataset
from dpfl.models.base_model import BaseModel
from dpfl.privacy.base_noise_mechanism import BaseNoiseMechanism
from dpfl.privacy.renyi_accountant import RenyiAccountant
from dpfl.aggregation.base_aggregator import BaseAggregator
from dpfl.attacks.base_attack import BaseAttack
from dpfl.topology.random_graph import create_regular_graph
from dpfl.training.dpsgd_trainer import DPSGDTrainer
from dpfl.training.node import Node


class DFLSimulator:
    def __init__(self, config: ExperimentConfig,
                 dataset_cls: Type[BaseDataset], model_cls: Type[BaseModel],
                 noise_mechanism: BaseNoiseMechanism,
                 aggregator: BaseAggregator, attack: BaseAttack,
                 tracker=None, device=None):
        self.config = config
        self.dataset_cls = dataset_cls
        self.model_cls = model_cls
        self.noise_mechanism = noise_mechanism
        self.aggregator = aggregator
        self.attack = attack
        self.tracker = tracker
        self.device = device or torch.device("cpu")
        self.nodes: Dict[int, Node] = {}
        self.topology: Dict[int, Set[int]] = {}
        self.accountant = None
        self.trainer = None
        self.test_dataset = None
        self.test_loader = None
        self.attacker_ids: set = set()

    def setup(self):
        """Initialize data, models, topology, nodes."""
        ds = self.dataset_cls()
        train_ds, self.test_dataset = ds.load()
        node_data = ds.split(
            train_ds, self.config.topology.n_nodes,
            self.config.dataset.split.mode, self.config.dataset.split.alpha,
        )

        self.topology = create_regular_graph(
            self.config.topology.n_nodes, self.config.topology.n_neighbors,
            self.config.topology.seed,
        )

        self.attacker_ids = set(range(self.config.topology.n_attackers))

        # Create base model and deep-copy for each node
        input_dim = prod(ds.input_shape)
        base_model = self.model_cls(
            input_dim=input_dim,
            hidden_size=self.config.model.hidden_size,
            num_classes=ds.num_classes,
        )

        for i in range(self.config.topology.n_nodes):
            model_copy = copy.deepcopy(base_model).to(self.device)
            node = Node(i, model_copy, node_data[i], i in self.attacker_ids)
            node.neighbors = self.topology[i]
            self.nodes[i] = node

        self.accountant = RenyiAccountant(self.config.dp.alpha_list, self.config.dp.delta)
        self.trainer = DPSGDTrainer(self.config.training, self.config.dp, self.device)

        # Cache test loader
        from torch.utils.data import DataLoader
        self.test_loader = DataLoader(self.test_dataset, batch_size=256, shuffle=False)

    def run(self):
        """Main loop: T rounds of DFL with DP-SGD + kurtosis detection."""
        for t in range(self.config.training.n_rounds):
            # Step 1: All nodes compute updates (parallel via thread pool)
            updates, all_steps = {}, {}
            n_workers = self.config.training.n_workers

            def _train_node(node):
                attack = self.attack if node.is_attacker else None
                update, n_steps = node.compute_update(
                    self.trainer, self.noise_mechanism, attack,
                )
                return node.id, update, n_steps

            if n_workers > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    for nid, update, n_steps in pool.map(_train_node, self.nodes.values()):
                        updates[nid] = update
                        all_steps[nid] = n_steps
            else:
                for node in self.nodes.values():
                    nid, update, n_steps = _train_node(node)
                    updates[nid] = update
                    all_steps[nid] = n_steps

            # Compute update norms per node
            update_norms = {nid: float(torch.norm(upd).item()) for nid, upd in updates.items()}

            # Step 2-4: Aggregate (includes kurtosis defense)
            total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
            per_node_detection = {}
            kurtosis_per_node = {}

            for node in self.nodes.values():
                neighbor_updates = {j: updates[j] for j in node.neighbors}
                result = self.aggregator.aggregate(
                    updates[node.id], node.model.get_flat_params(), neighbor_updates,
                )
                node.model.set_flat_params(result.new_params)

                # Per-node detection metrics
                tp, fp, fn, tn = self._compute_detection(
                    result.flagged_ids, result.clean_ids, node.neighbors,
                )
                per_node_detection[node.id] = (tp, fp, fn, tn)
                total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn

            # Log own-update kurtosis per node
            kurtosis_honest, kurtosis_attacker = [], []
            for nid, upd in updates.items():
                k_val = float(self.aggregator._excess_kurtosis(upd))
                kurtosis_per_node[nid] = k_val
                if nid in self.attacker_ids:
                    kurtosis_attacker.append(k_val)
                else:
                    kurtosis_honest.append(k_val)

            # Step 5: Privacy accounting
            honest_steps = next(
                s for nid, s in all_steps.items() if nid not in self.attacker_ids
            )
            q_batch = self.config.training.batch_size / self.nodes[
                self.config.topology.n_attackers
            ].n_samples
            self.accountant.step(honest_steps, q_batch, self.config.dp.noise_mult)
            epsilon = self.accountant.get_epsilon()

            if epsilon > self.config.dp.epsilon_max:
                print(f"Round {t + 1}: Budget exceeded (eps={epsilon:.2f})")
                if self.tracker:
                    self.tracker.log_round(
                        round_num=t, epsilon=epsilon, accuracy=0.0,
                        best_alpha=self.accountant.get_best_alpha(),
                        stopped_early=True,
                    )
                break

            # Step 6: Evaluate all nodes (accuracy + test loss)
            eval_results = self._evaluate_nodes()

            # Compute aggregated metrics (honest nodes only)
            honest_evals = {nid: v for nid, v in eval_results.items() if nid not in self.attacker_ids}
            accuracy = np.mean([v["accuracy"] for v in honest_evals.values()])
            test_loss = np.mean([v["test_loss"] for v in honest_evals.values()])

            # Aggregated detection metrics
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            # Update norm aggregates
            honest_norms = [update_norms[nid] for nid in update_norms if nid not in self.attacker_ids]
            attacker_norms = [update_norms[nid] for nid in update_norms if nid in self.attacker_ids]

            if self.tracker:
                # Aggregated round log
                self.tracker.log_round(
                    round_num=t, epsilon=epsilon,
                    accuracy=float(accuracy),
                    test_loss=float(test_loss),
                    f1_score=f1,
                    mean_update_norm_honest=float(np.mean(honest_norms)) if honest_norms else 0.0,
                    mean_update_norm_attacker=float(np.mean(attacker_norms)) if attacker_norms else 0.0,
                    best_alpha=self.accountant.get_best_alpha(),
                    precision=precision, recall=recall,
                    kurtosis_honest=float(np.mean(kurtosis_honest)) if kurtosis_honest else 0.0,
                    kurtosis_attacker=float(np.mean(kurtosis_attacker)) if kurtosis_attacker else 0.0,
                )

                # Per-node round log
                nodes_data = {}
                for nid in self.nodes:
                    tp, fp, fn, tn = per_node_detection[nid]
                    p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    node_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                    nodes_data[nid] = {
                        "accuracy": eval_results[nid]["accuracy"],
                        "test_loss": eval_results[nid]["test_loss"],
                        "precision": p,
                        "recall": r,
                        "f1_score": node_f1,
                        "update_norm": update_norms[nid],
                        "kurtosis": kurtosis_per_node[nid],
                        "is_attacker": nid in self.attacker_ids,
                    }
                self.tracker.log_node_round(t, nodes_data)

            if (t + 1) % 10 == 0:
                print(
                    f"Round {t + 1:3d}/{self.config.training.n_rounds} | "
                    f"Acc: {accuracy:.4f} | Loss: {test_loss:.4f} | "
                    f"eps: {epsilon:.2f} | P: {precision:.2f} R: {recall:.2f} F1: {f1:.2f}"
                )

    def _evaluate_nodes(self) -> Dict[int, Dict[str, float]]:
        """Evaluate accuracy and test loss for all nodes on global test set."""
        results = {}
        for nid, node in self.nodes.items():
            correct, total = 0, 0
            total_loss = 0.0
            node.model.eval()
            with torch.no_grad():
                for x, y in self.test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = node.model(x)
                    total_loss += F.cross_entropy(out, y, reduction="sum").item()
                    correct += (out.argmax(1) == y).sum().item()
                    total += len(y)
            node.model.train()
            results[nid] = {
                "accuracy": correct / total if total else 0.0,
                "test_loss": total_loss / total if total else 0.0,
            }
        return results

    def _compute_detection(self, flagged_ids, clean_ids, neighbors):
        """Compute TP/FP/FN/TN from flagged vs actual attackers."""
        tp = sum(1 for j in flagged_ids if j in self.attacker_ids)
        fp = sum(1 for j in flagged_ids if j not in self.attacker_ids)
        fn = sum(1 for j in clean_ids if j in self.attacker_ids)
        tn = sum(1 for j in clean_ids if j not in self.attacker_ids)
        return tp, fp, fn, tn
