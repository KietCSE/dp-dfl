"""Trust-Aware DFL simulator: 4-phase per-round loop for D2B-DP algorithm."""

import copy
import importlib
import torch
import torch.nn.functional as F
import numpy as np
from math import prod
from typing import Dict, Set

from dpfl.data.base_dataset import BaseDataset
from dpfl.models.base_model import BaseModel
from dpfl.privacy.base_noise_mechanism import BaseNoiseMechanism
from dpfl.attacks.base_attack import BaseAttack
from dpfl.topology.random_graph import create_regular_graph
from dpfl.training.dpsgd_trainer import DPSGDTrainer
from dpfl.privacy.adaptive_clipper import AdaptiveClipper
from dpfl.privacy.bounded_gaussian_mechanism import BoundedGaussianMechanism
from dpfl.privacy.per_neighbor_rdp_accountant import PerNeighborRDPAccountant

# Kebab-case module import
_node_mod = importlib.import_module("dpfl.training.trust-aware-node")
TrustAwareNode = _node_mod.TrustAwareNode


class TrustAwareDFLSimulator:
    """Orchestrator for Trust-Aware D2B-DP: train -> noise -> evaluate -> filter."""

    def __init__(self, config, trust_config, dataset_cls, model_cls,
                 noise_mechanism, aggregator, attack,
                 adaptive_clipper, bounded_noise, per_neighbor_accountant,
                 tracker=None, device=None):
        self.config = config
        self.tc = trust_config
        self.dataset_cls = dataset_cls
        self.model_cls = model_cls
        self.noise_mechanism = noise_mechanism
        self.aggregator = aggregator
        self.attack = attack
        self.clipper = adaptive_clipper
        self.bounded_noise = bounded_noise
        self.rdp = per_neighbor_accountant
        self.tracker = tracker
        self.device = device or torch.device("cpu")
        self.nodes: Dict[int, TrustAwareNode] = {}
        self.attacker_ids: set = set()
        self.trainer = None
        self.test_loader = None

    def setup(self):
        """Initialize data, models, topology, trust-aware nodes."""
        ds = self.dataset_cls()
        train_ds, test_ds = ds.load()
        node_data = ds.split(train_ds, self.config.topology.n_nodes,
                             self.config.dataset.split.mode,
                             self.config.dataset.split.alpha)
        topology = create_regular_graph(self.config.topology.n_nodes,
                                        self.config.topology.n_neighbors,
                                        self.config.topology.seed)
        self.attacker_ids = set(range(self.config.topology.n_attackers))
        input_dim = prod(ds.input_shape)
        base_model = self.model_cls(input_dim=input_dim,
                                    hidden_size=self.config.model.hidden_size,
                                    num_classes=ds.num_classes)
        alpha_list = self.config.dp.accountant_params.get(
            "alpha_list", [1.25, 1.5, 2, 3, 5, 10, 20, 50, 100])
        for i in range(self.config.topology.n_nodes):
            model_copy = copy.deepcopy(base_model).to(self.device)
            node = TrustAwareNode(i, model_copy, node_data[i],
                                  i in self.attacker_ids, self.tc)
            node.neighbors = topology[i]
            node.init_trust(node.neighbors, self.tc.trust_init, alpha_list)
            self.nodes[i] = node
        self.trainer = DPSGDTrainer(self.config.training, self.config.dp,
                                    self.device)
        from torch.utils.data import DataLoader
        self.test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    def run(self):
        """Main loop: 4-phase D2B-DP per round."""
        for t in range(self.config.training.n_rounds):
            # Phase 1: Train + Adaptive Clip
            updates = {}
            for node in self.nodes.values():
                atk = self.attack if node.is_attacker else None
                update, _ = node.compute_update(
                    self.trainer, self.noise_mechanism, atk)
                raw_norm = update.norm(2).item()
                node.clip_history.append(raw_norm)
                thresh = self.clipper.get_threshold(node.clip_history)
                if thresh is not None:
                    update = self.clipper.clip(update, thresh)
                updates[node.id] = update

            # Phase 2: Outbound — per-edge noise injection
            edge_updates = {}
            for node in self.nodes.values():
                edge_updates[node.id] = {}
                clip_e = (self.clipper.get_threshold(node.clip_history)
                          or updates[node.id].norm(2).item())
                for j in node.neighbors:
                    rho_base = min(
                        (1 + self.tc.beta * t) * self.tc.rho_min, self.tc.rho_max)
                    rho_ij = rho_base * node.trust_scores[j]
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
            tp_all = fp_all = fn_all = tn_all = 0
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
                for j in result.flagged_ids:
                    if j in self.attacker_ids: tp_all += 1
                    else: fp_all += 1
                for j in result.clean_ids:
                    if j in self.attacker_ids: fn_all += 1
                    else: tn_all += 1

            # Privacy accounting: max epsilon across all honest edges
            eps_vals = [
                self.rdp.get_epsilon(node, j)
                for node in self.nodes.values()
                if node.id not in self.attacker_ids
                for j in node.neighbors
            ]
            max_eps = max(eps_vals) if eps_vals else 0.0

            self._log_round(t, updates, tp_all, fp_all, fn_all, tn_all,
                            node_agg_metrics, max_eps)

    def _log_round(self, t, updates, tp, fp, fn, tn, node_agg_metrics, epsilon):
        """Evaluate, compute metrics, log to tracker + console."""
        evals = self._evaluate_nodes()
        honest = {nid: v for nid, v in evals.items()
                  if nid not in self.attacker_ids}
        acc = np.mean([v["accuracy"] for v in honest.values()])
        loss = np.mean([v["test_loss"] for v in honest.values()])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        # Trust diagnostics: honest nodes' view of honest vs attacker neighbors
        t_h, t_a = [], []
        for n in self.nodes.values():
            if n.id in self.attacker_ids:
                continue
            for j, tv in n.trust_scores.items():
                (t_a if j in self.attacker_ids else t_h).append(tv)
        trust_h = float(np.mean(t_h)) if t_h else 0.0
        trust_a = float(np.mean(t_a)) if t_a else 0.0

        if self.tracker:
            norms = {nid: float(u.norm().item()) for nid, u in updates.items()}
            nodes_data = {}
            _known = {"accuracy", "test_loss", "update_norm", "is_attacker"}
            for nid in self.nodes:
                nodes_data[nid] = {
                    "accuracy": evals[nid]["accuracy"],
                    "test_loss": evals[nid]["test_loss"],
                    "update_norm": norms[nid],
                    "is_attacker": nid in self.attacker_ids,
                    **node_agg_metrics.get(nid, {})}
            # Auto-discover defense metrics by role
            defense = {}
            dkeys = {k for nd in nodes_data.values() for k, v in nd.items()
                     if k not in _known and isinstance(v, (int, float))}
            for key in dkeys:
                h = [nd[key] for nd in nodes_data.values()
                     if not nd["is_attacker"] and key in nd]
                a = [nd[key] for nd in nodes_data.values()
                     if nd["is_attacker"] and key in nd]
                defense[f"{key}_honest"] = float(np.mean(h)) if h else 0.0
                defense[f"{key}_attacker"] = float(np.mean(a)) if a else 0.0
            self.tracker.log_round(
                round_num=t, accuracy=float(acc), test_loss=float(loss),
                precision=prec, recall=rec, f1_score=f1,
                epsilon=epsilon,
                trust_toward_honest=trust_h, trust_toward_attacker=trust_a,
                **defense)
            self.tracker.log_node_round(t, nodes_data)

        print(f"Round {t+1:3d}/{self.config.training.n_rounds} | "
              f"Acc: {acc:.4f} | eps: {epsilon:.4f} | "
              f"Trust H->H: {trust_h:.3f} H->A: {trust_a:.3f}"
              f" | P: {prec:.2f} R: {rec:.2f} F1: {f1:.2f}")

    def _evaluate_nodes(self):
        """Evaluate accuracy + test loss for all nodes on global test set."""
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
            results[nid] = {"accuracy": correct / total if total else 0.0,
                            "test_loss": total_loss / total if total else 0.0}
        return results
