"""Noise-game DFL simulator: orchestrates the full strategic noise algorithm."""

import copy
from concurrent.futures import ThreadPoolExecutor
import importlib
import math
import torch
import torch.nn.functional as F
import numpy as np
from math import prod
from typing import Dict, Set, Type

from dpfl.noise_game_config import NoiseGameConfig, NoiseGameExperimentConfig
from dpfl.data.base_dataset import BaseDataset
from dpfl.models.base_model import BaseModel
from dpfl.privacy.base_noise_mechanism import BaseNoiseMechanism
from dpfl.privacy.base_accountant import BaseAccountant
from dpfl.aggregation.base_aggregator import BaseAggregator
from dpfl.attacks.base_attack import BaseAttack
from dpfl.topology.random_graph import create_regular_graph
from dpfl.training.dpsgd_trainer import DPSGDTrainer

# Kebab-case module imports
_node_mod = importlib.import_module("dpfl.training.noise-game-node")
NoiseGameNode = _node_mod.NoiseGameNode

_mech_mod = importlib.import_module("dpfl.privacy.noise-game-mechanism")
NoiseGameMechanism = _mech_mod.NoiseGameMechanism


class NoiseGameDFLSimulator:
    """Orchestrator for Strategic Noise Game DFL.

    Algorithm per round (Section 5 of noise-game.md):
      Phase 1: Local training + L2-norm clip
      Phase 2: Noise-game pipeline (SCAFFOLD, strategic noise, alignment, EMA, momentum)
      Phase 3: Aggregation (reuse existing aggregator)
      Phase 4: Privacy accounting
      Phase 5: Evaluate + log
    """

    def __init__(self, config: NoiseGameExperimentConfig,
                 ng_config: NoiseGameConfig,
                 dataset_cls: Type[BaseDataset], model_cls: Type[BaseModel],
                 noise_mechanism: BaseNoiseMechanism,
                 game_mechanism: NoiseGameMechanism,
                 aggregator: BaseAggregator, attack: BaseAttack,
                 accountant: BaseAccountant = None,
                 tracker=None, device=None):
        self.config = config
        self.ng = ng_config
        self.dataset_cls = dataset_cls
        self.model_cls = model_cls
        self.noise_mechanism = noise_mechanism
        self.game_mechanism = game_mechanism
        self.aggregator = aggregator
        self.attack = attack
        self.accountant = accountant
        self.tracker = tracker
        self.device = device or torch.device("cpu")
        self.nodes: Dict[int, NoiseGameNode] = {}
        self.topology: Dict[int, Set[int]] = {}
        self.trainer = None
        self.test_loader = None
        self.attacker_ids: set = set()

    def setup(self):
        """Initialize data, models, topology, and noise-game nodes."""
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
        param_dim = base_model.count_params()

        for i in range(self.config.topology.n_nodes):
            model_copy = copy.deepcopy(base_model).to(self.device)
            node = NoiseGameNode(i, model_copy, node_data[i],
                                 i in self.attacker_ids, self.ng)
            node.neighbors = self.topology[i]
            node.init_buffers(param_dim)
            self.nodes[i] = node

        self.trainer = DPSGDTrainer(self.config.training, self.config.dp, self.device)

        from torch.utils.data import DataLoader
        self.test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    def run(self):
        """Main loop: T rounds of noise-game DFL."""
        for t in range(self.config.training.n_rounds):
            # Phase 1: Local training (parallel)
            raw_updates, all_steps = self._train_all_nodes()

            # L2-norm clip each update
            C = self.config.dp.clip_bound
            clipped = {}
            for nid, upd in raw_updates.items():
                norm = upd.norm()
                factor = min(1.0, C / (norm + 1e-12))
                clipped[nid] = upd * factor

            # Phase 2: Noise-game pipeline (honest nodes only)
            final_updates = {}
            node_noise_metrics = {}
            for nid, node in self.nodes.items():
                g = clipped[nid]
                if node.is_attacker:
                    final_updates[nid] = g
                    continue

                # SCAFFOLD variance reduction
                if self.ng.scaffold:
                    global_c = self._neighbor_avg_control(node)
                    g = node.apply_scaffold(g, global_c)

                # Strategic noise injection
                noise, metrics = self.game_mechanism.compute_total_noise(
                    g, node.prev_gradient, round_t=t)
                g_hat = g + noise

                # Gradient alignment filtering
                if not node.check_alignment(g_hat):
                    g_hat = node.ema_gradient.clone()

                # EMA denoising
                node.update_ema(g_hat)

                # Momentum
                m = node.update_momentum(g_hat)

                # Trust-aware LR scaling
                trust = metrics["trust"]
                final_updates[nid] = m * max(trust, 0.0)

                # Store state for next round
                node.store_gradient(clipped[nid])
                if self.ng.scaffold:
                    node.update_control_variate(clipped[nid])

                node_noise_metrics[nid] = metrics

            # Phase 3: Aggregation
            update_norms = {nid: float(u.norm().item()) for nid, u in final_updates.items()}
            total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
            per_node_detection = {}
            node_agg_metrics = {}

            for node in self.nodes.values():
                neighbor_upds = {j: final_updates[j] for j in node.neighbors}
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
                    result.flagged_ids, result.clean_ids, node.neighbors)
                per_node_detection[node.id] = (tp, fp, fn, tn)
                total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn

            # Phase 4: Privacy accounting (treat annealed sigma as Gaussian noise_mult)
            epsilon = 0.0
            if self.accountant is not None:
                honest_steps = next(
                    s for nid, s in all_steps.items() if nid not in self.attacker_ids)
                q_batch = self.config.training.batch_size / self.nodes[
                    self.config.topology.n_attackers].n_samples
                sigma_t = self.game_mechanism.compute_annealed_sigma(t)
                effective_noise_mult = max(sigma_t / (C + 1e-12), 0.01)
                self.accountant.step(honest_steps, q_batch, effective_noise_mult)
                epsilon = self.accountant.get_epsilon()

                if epsilon > self.config.dp.epsilon_max:
                    print(f"Round {t + 1}: Budget exceeded (eps={epsilon:.2f})")
                    break

            # Phase 5: Evaluate + log
            self._log_round(t, epsilon, update_norms, per_node_detection,
                            node_agg_metrics, node_noise_metrics,
                            total_tp, total_fp, total_fn, total_tn)

    # -- Helpers --

    def _train_all_nodes(self):
        """Phase 1: all nodes compute local updates (parallel if n_workers > 1)."""
        updates, steps = {}, {}
        n_workers = self.config.training.n_workers

        def _train(node):
            atk = self.attack if node.is_attacker else None
            upd, n = node.compute_update(self.trainer, self.noise_mechanism, atk)
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

    def _neighbor_avg_control(self, node: NoiseGameNode) -> torch.Tensor:
        """D-SCAFFOLD: approximate global control as average of neighbors' c_i."""
        controls = [self.nodes[j].control_variate for j in node.neighbors
                    if self.nodes[j].control_variate is not None]
        if not controls:
            return torch.zeros_like(node.control_variate)
        return torch.stack(controls).mean(dim=0)

    def _evaluate_nodes(self) -> Dict[int, Dict[str, float]]:
        """Evaluate accuracy and test loss for all nodes."""
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
        tp = sum(1 for j in flagged_ids if j in self.attacker_ids)
        fp = sum(1 for j in flagged_ids if j not in self.attacker_ids)
        fn = sum(1 for j in clean_ids if j in self.attacker_ids)
        tn = sum(1 for j in clean_ids if j not in self.attacker_ids)
        return tp, fp, fn, tn

    def _log_round(self, t, epsilon, update_norms, per_node_detection,
                   node_agg_metrics, node_noise_metrics,
                   total_tp, total_fp, total_fn, total_tn):
        """Evaluate all nodes, compute metrics, log to tracker."""
        eval_results = self._evaluate_nodes()
        honest_evals = {n: v for n, v in eval_results.items() if n not in self.attacker_ids}
        accuracy = np.mean([v["accuracy"] for v in honest_evals.values()])
        test_loss = np.mean([v["test_loss"] for v in honest_evals.values()])

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        honest_norms = [update_norms[n] for n in update_norms if n not in self.attacker_ids]
        attacker_norms = [update_norms[n] for n in update_norms if n in self.attacker_ids]

        if self.tracker:
            # Collect noise-game specific metrics
            honest_trust = [m["trust"] for m in node_noise_metrics.values()]
            avg_trust = float(np.mean(honest_trust)) if honest_trust else 0.0
            avg_sigma = float(np.mean([m["sigma_t"] for m in node_noise_metrics.values()])) \
                if node_noise_metrics else 0.0

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
                    "update_norm": update_norms[nid],
                    "is_attacker": nid in self.attacker_ids,
                    **node_agg_metrics.get(nid, {}),
                }
                if nid in node_noise_metrics:
                    nd["trust"] = node_noise_metrics[nid]["trust"]
                    nd["noise_norm"] = node_noise_metrics[nid]["total_noise_norm"]
                nodes_data[nid] = nd

            # Auto-discover defense metrics
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

            self.tracker.log_round(
                round_num=t, epsilon=epsilon, accuracy=float(accuracy),
                test_loss=float(test_loss), f1_score=f1,
                mean_update_norm_honest=float(np.mean(honest_norms)) if honest_norms else 0.0,
                mean_update_norm_attacker=float(np.mean(attacker_norms)) if attacker_norms else 0.0,
                best_alpha=self.accountant.get_best_alpha() if self.accountant else None,
                precision=precision, recall=recall,
                avg_trust=avg_trust, avg_sigma_t=avg_sigma,
                **defense_round)
            self.tracker.log_node_round(t, nodes_data)

        print(
            f"Round {t + 1:3d}/{self.config.training.n_rounds} | "
            f"Acc: {accuracy:.4f} | Loss: {test_loss:.4f} | "
            f"eps: {epsilon:.2f} | P: {precision:.2f} R: {recall:.2f} F1: {f1:.2f}")
