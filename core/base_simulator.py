"""Base DFL simulator: shared setup, evaluation, detection, logging."""

import copy
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from math import prod
from typing import Dict, Set, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dpfl.config import BaseExperimentConfig
from dpfl.data.base_dataset import BaseDataset
from dpfl.models.base_model import BaseModel
from dpfl.core.base_noise_mechanism import BaseNoiseMechanism
from dpfl.core.base_accountant import BaseAccountant
from dpfl.core.base_aggregator import BaseAggregator
from dpfl.core.base_attack import BaseAttack
from dpfl.topology.random_graph import create_regular_graph
from dpfl.core.dpsgd_trainer import DPSGDTrainer
from dpfl.core.base_node import Node


class BaseSimulator(ABC):
    """Shared infrastructure for all DFL algorithm variants.

    Subclasses implement:
      - _create_node(): factory for variant-specific node types
      - run(): the algorithm's main loop
    """

    def __init__(self, config: BaseExperimentConfig,
                 dataset_cls: Type[BaseDataset], model_cls: Type[BaseModel],
                 noise_mechanism: BaseNoiseMechanism,
                 aggregator: BaseAggregator, attack: BaseAttack,
                 accountant: BaseAccountant = None,
                 tracker=None, device=None):
        self.config = config
        self.dataset_cls = dataset_cls
        self.model_cls = model_cls
        self.noise_mechanism = noise_mechanism
        self.aggregator = aggregator
        self.attack = attack
        self.accountant = accountant
        self.tracker = tracker
        self.device = device or torch.device("cpu")
        self.nodes: Dict[int, Node] = {}
        self.topology: Dict[int, Set[int]] = {}
        self.trainer = None
        self.test_loader = None
        self.attacker_ids: set = set()
        self.param_dim: int = 0

    def setup(self):
        """Initialize data, models, topology, nodes. Shared across all variants."""
        ds = self.dataset_cls()
        train_ds, test_ds = ds.load()
        node_data = ds.split(
            train_ds, self.config.topology.n_nodes,
            self.config.dataset.split.mode, self.config.dataset.split.alpha)

        self.topology = create_regular_graph(
            self.config.topology.n_nodes, self.config.topology.n_neighbors,
            self.config.topology.seed)

        self.attacker_ids = set(range(self.config.topology.n_attackers))

        input_dim = prod(ds.input_shape)
        base_model = self.model_cls(
            input_dim=input_dim, hidden_size=self.config.model.hidden_size,
            num_classes=ds.num_classes)
        self.param_dim = base_model.count_params()

        for i in range(self.config.topology.n_nodes):
            model_copy = copy.deepcopy(base_model).to(self.device)
            node = self._create_node(i, model_copy, node_data[i], i in self.attacker_ids)
            node.neighbors = self.topology[i]
            self.nodes[i] = node

        self.trainer = DPSGDTrainer(self.config.training, self.config.dp, self.device)
        self.test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    @abstractmethod
    def _create_node(self, node_id, model, data, is_attacker) -> Node:
        """Factory method: create variant-specific node."""
        ...

    @abstractmethod
    def run(self):
        """Main algorithm loop. Implement per variant."""
        ...

    def _train_all_nodes(self, apply_noise: bool = True, round_t: int = 0):
        """Train all nodes in parallel. Returns (updates_dict, steps_dict).
        Attackers only attack when round_t >= config.attack.start_round.
        """
        updates, steps = {}, {}
        n_workers = self.config.training.n_workers
        attack_active = round_t >= self.config.attack.start_round

        def _train(node):
            atk = self.attack if (node.is_attacker and attack_active) else None
            upd, n = node.compute_update(
                self.trainer, self.noise_mechanism, atk, apply_noise=apply_noise)
            return node.id, upd, n

        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for nid, upd, n in pool.map(_train, self.nodes.values()):
                    updates[nid] = upd; steps[nid] = n
        else:
            for node in self.nodes.values():
                nid, upd, n = _train(node)
                updates[nid] = upd; steps[nid] = n
        return updates, steps

    def _evaluate_nodes(self) -> Dict[int, Dict[str, float]]:
        """Evaluate accuracy and test loss for all nodes on global test set."""
        results = {}
        for nid, node in self.nodes.items():
            correct, total, total_loss = 0, 0, 0.0
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

    def _log_round(self, t, epsilon, updates, per_node_detection,
                   node_agg_metrics, total_tp, total_fp, total_fn, total_tn,
                   extra_node_data=None, extra_round_metrics=None):
        """Shared logging: evaluate, compute metrics, log to tracker."""
        eval_results = self._evaluate_nodes()
        honest_evals = {n: v for n, v in eval_results.items() if n not in self.attacker_ids}
        accuracy = np.mean([v["accuracy"] for v in honest_evals.values()])
        test_loss = np.mean([v["test_loss"] for v in honest_evals.values()])

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        update_norms = {nid: float(u.norm().item()) for nid, u in updates.items()}
        honest_norms = [update_norms[n] for n in update_norms if n not in self.attacker_ids]
        attacker_norms = [update_norms[n] for n in update_norms if n in self.attacker_ids]

        if self.tracker:
            nodes_data = {}
            _known = {"accuracy", "test_loss", "precision", "recall",
                      "f1_score", "update_norm", "is_attacker"}
            for nid in self.nodes:
                tp, fp, fn, tn = per_node_detection[nid]
                p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                nf1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                nd = {
                    "accuracy": eval_results[nid]["accuracy"],
                    "test_loss": eval_results[nid]["test_loss"],
                    "precision": p, "recall": r, "f1_score": nf1,
                    "update_norm": update_norms.get(nid, 0.0),
                    "is_attacker": nid in self.attacker_ids,
                    **node_agg_metrics.get(nid, {}),
                }
                if extra_node_data and nid in extra_node_data:
                    nd.update(extra_node_data[nid])
                nodes_data[nid] = nd

            # Auto-discover defense metrics and aggregate by role
            defense_keys = set()
            for nd in nodes_data.values():
                for k, v in nd.items():
                    if k not in _known and isinstance(v, (int, float)):
                        defense_keys.add(k)
            defense_round = {}
            for key in defense_keys:
                h = [nd[key] for nd in nodes_data.values()
                     if not nd["is_attacker"] and key in nd]
                a = [nd[key] for nd in nodes_data.values()
                     if nd["is_attacker"] and key in nd]
                defense_round[f"{key}_honest"] = float(np.mean(h)) if h else 0.0
                defense_round[f"{key}_attacker"] = float(np.mean(a)) if a else 0.0

            round_metrics = dict(
                round_num=t, epsilon=epsilon, accuracy=float(accuracy),
                test_loss=float(test_loss), f1_score=f1,
                mean_update_norm_honest=float(np.mean(honest_norms)) if honest_norms else 0.0,
                mean_update_norm_attacker=float(np.mean(attacker_norms)) if attacker_norms else 0.0,
                best_alpha=self.accountant.get_best_alpha() if self.accountant else None,
                precision=precision, recall=recall,
                **defense_round)
            if extra_round_metrics:
                round_metrics.update(extra_round_metrics)
            self.tracker.log_round(**round_metrics)
            self.tracker.log_node_round(t, nodes_data)

        print(
            f"Round {t + 1:3d}/{self.config.training.n_rounds} | "
            f"Acc: {accuracy:.4f} | Loss: {test_loss:.4f} | "
            f"eps: {epsilon:.2f} | P: {precision:.2f} R: {recall:.2f} F1: {f1:.2f}")
