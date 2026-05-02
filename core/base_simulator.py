"""Base DFL simulator: shared setup, evaluation, detection, logging."""

import copy
import logging
import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from math import prod
from typing import Dict, Set, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

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
from dpfl.core.vectorized_state import (
    ParamShapeSpec, pack_node_params, unpack_to_nodes,
)
from dpfl.core.vectorized_eval import vectorized_evaluate
from dpfl.core.vectorized_data import VectorizedDataPipeline
from dpfl.core.vectorized_trainer import (
    train_all_standard, train_all_dpsgd_per_step,
)


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

        # Vectorized state (populated in setup() if config.training.use_vectorized).
        # base_model_template: stateless module for functional_call (no per-node copy).
        # param_spec: pack/unpack between (N, D) and per-client param dicts.
        # X_test, Y_test: pre-stacked test set on device for vmapped eval.
        # train_pipeline: GPU-resident per-client training data with padding/mask.
        self.base_model_template: BaseModel = None
        self.param_spec: ParamShapeSpec = None
        self.X_test: torch.Tensor = None
        self.Y_test: torch.Tensor = None
        self.train_pipeline: VectorizedDataPipeline = None

        # Isolated torch RNGs per purpose to prevent cross-algorithm state
        # drift. Same config.seed yields identical sequences for each purpose,
        # regardless of how many torch.randn calls other algorithms make.
        # data_gen MUST be CPU (DataLoader RandomSampler runs on CPU).
        # noise_gen / attack_gen match self.device (noise is created on device).
        self.data_gen = self._make_isolated_gen(1_000_007, force_cpu=True)
        self.noise_gen = self._make_isolated_gen(1_000_013)
        self.attack_gen = self._make_isolated_gen(1_000_019)

        # Distribute noise_gen / attack_gen to mechanism and attack if they
        # support set_generator. Algorithm-specific mechanisms (e.g.,
        # BoundedGaussian, NoiseGameMechanism) are wired by their subclass
        # simulators in their own __init__.
        if hasattr(self.noise_mechanism, "set_generator") \
                and self.noise_mechanism is not None:
            self.noise_mechanism.set_generator(self.noise_gen)
        if self.attack is not None and hasattr(self.attack, "set_generator"):
            self.attack.set_generator(self.attack_gen)
        if hasattr(self.aggregator, "set_generator"):
            # FLAME aggregator adds internal DP noise — uses noise_gen.
            self.aggregator.set_generator(self.noise_gen)

    def _make_isolated_gen(self, seed_multiplier: int,
                           force_cpu: bool = False) -> torch.Generator:
        """Create a torch.Generator seeded deterministically from config.seed.

        Device-aware: matches self.device so downstream torch.randn calls can
        use this generator without device-mismatch errors. For CPU-bound
        consumers (DataLoader RandomSampler) set force_cpu=True. Seeded with
        `config.seed * seed_multiplier` (distinct primes per purpose to
        guarantee independent streams).
        """
        gen_device = torch.device("cpu") if force_cpu else self.device
        gen = torch.Generator(device=gen_device)
        gen.manual_seed(int(self.config.seed) * int(seed_multiplier))
        return gen

    def setup(self):
        """Initialize data, models, topology, nodes. Shared across all variants."""
        ds = self.dataset_cls()
        train_ds, test_ds = ds.load()
        node_data = ds.split(
            train_ds, self.config.topology.n_nodes,
            self.config.dataset.split.mode, self.config.dataset.split.alpha,
            samples_per_node=self.config.dataset.split.samples_per_node)

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

        # Apply data poisoning (LabelFlip) to attacker nodes
        if self.attack and hasattr(self.attack, 'wrap_dataset'):
            for node in self.nodes.values():
                if node.is_attacker:
                    node.apply_data_attack(self.attack)

        self.trainer = DPSGDTrainer(
            self.config.training, self.config.dp, self.device,
            data_gen=self.data_gen)
        self.test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

        if getattr(self.config.training, "use_vectorized", False):
            self._setup_vectorized(base_model, test_ds, node_data)

        logger.info("Setup complete: %d nodes (%d attackers), param_dim=%d, split=%s",
                     self.config.topology.n_nodes, len(self.attacker_ids),
                     self.param_dim, self.config.dataset.split.mode)
        logger.debug("Attacker IDs: %s", self.attacker_ids)
        if self.attack and hasattr(self.attack, 'wrap_dataset'):
            logger.info("Data poisoning active: %s", type(self.attack).__name__)

    def _setup_vectorized(self, base_model: BaseModel, test_ds,
                          node_data: Dict[int, "torch.utils.data.Subset"]) -> None:
        """Init vectorized state: stateless template, ParamShapeSpec, test tensors,
        per-client training pipeline.

        We keep per-node `node.model` deepcopies for backward compat with
        algorithm-specific simulators that read/write `node.model` directly
        (e.g., aggregators, attacks). Vectorized methods repack from nodes
        on demand so legacy paths that mutate node.model stay consistent.

        node_data is the pre-wrap split (LabelFlip wrap happens after this in
        setup()). Vectorized training falls back to legacy when label_flip
        is active, so the unwrapped data here is correct for honest clients.
        """
        # Stateless template — keep on device, params will come from params_stack
        self.base_model_template = copy.deepcopy(base_model).to(self.device)
        self.param_spec = ParamShapeSpec(self.base_model_template)

        # Pre-stack test data on device once (vectorized eval iterates this)
        xs, ys = [], []
        for x, y in DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0):
            xs.append(x)
            ys.append(y)
        self.X_test = torch.cat(xs, dim=0).to(self.device)
        self.Y_test = torch.cat(ys, dim=0).to(torch.long).to(self.device)

        # Data-poisoning hook: build attacker_mask (aligned with sorted node
        # ids the pipeline uses internally) and bind attack.flip_y so the
        # pipeline can apply label flipping per batch in lieu of Python-level
        # Dataset wrapping. None for non-data-poisoning attacks.
        attacker_mask = None
        flip_fn = None
        if self.attack is not None and hasattr(self.attack, "flip_y"):
            sorted_node_ids = sorted(node_data.keys())
            attacker_mask = torch.tensor(
                [nid in self.attacker_ids for nid in sorted_node_ids],
                dtype=torch.bool, device=self.device,
            )
            flip_fn = self.attack.flip_y

        # Per-client training pipeline (GPU-resident). Uses data_gen for
        # deterministic per-client shuffles in the same RNG order as legacy.
        self.train_pipeline = VectorizedDataPipeline(
            dataset=next(iter(node_data.values())).dataset,
            node_data=node_data, device=self.device, data_gen=self.data_gen,
            attacker_mask=attacker_mask, flip_fn=flip_fn,
        )

        logger.info(
            "Vectorized state ready: param_spec.D=%d, X_test=%s, "
            "train clients N=%d, max_n_samples=%d",
            self.param_spec.D, tuple(self.X_test.shape),
            self.train_pipeline.N, self.train_pipeline.max_n_samples,
        )

    @abstractmethod
    def _create_node(self, node_id, model, data, is_attacker) -> Node:
        """Factory method: create variant-specific node."""
        ...

    @abstractmethod
    def run(self):
        """Main algorithm loop. Implement per variant."""
        ...

    def _sample_active_nodes(self, round_t: int) -> Set[int]:
        """Deterministic Poisson client subsampling per config.dp.sampling_rate.

        Same config.seed + round_t yields the same active set across all
        algorithms — enables fair cross-algorithm comparison (subsampling
        schedule is shared; algorithms differ only in noise/aggregation/defense).

        Rules:
          - Seeded RNG: random.Random(seed * 1_000_003 + round_t).
          - Iterate nodes in sorted(id) order so coin-flip sequence is stable.
          - Draw coin for every honest node regardless of frozen state
            (keeps the RNG stream aligned across algorithms); frozen filter
            applied after the draw.
          - Attacker nodes always active (no coin flip consumed).
          - q >= 1.0 => all non-frozen nodes active (backward-compatible).
        """
        q = float(self.config.dp.sampling_rate)
        rng = random.Random(self.config.seed * 1_000_003 + round_t)
        active: Set[int] = set()
        for nid in sorted(self.nodes.keys()):
            node = self.nodes[nid]
            if node.is_attacker:
                if not getattr(node, "frozen", False):
                    active.add(nid)
                continue
            coin = True if q >= 1.0 else (rng.random() < q)
            if getattr(node, "frozen", False):
                continue
            if coin:
                active.add(nid)
        return active

    def _train_all_nodes(self, apply_noise: bool = True, round_t: int = 0):
        """Train all nodes. Returns (updates_dict, steps_dict).

        Dispatches to vectorized vmap path when use_vectorized=True AND the
        config is supported by Phase 2 (non-DP, non-data-poisoning); falls
        back to the legacy ThreadPool/sequential path otherwise.
        """
        if self._can_use_vectorized_training():
            return self._train_all_nodes_vectorized(apply_noise, round_t)
        return self._train_all_nodes_legacy(apply_noise, round_t)

    def _can_use_vectorized_training(self) -> bool:
        """Vectorized path supports:
        - Phase 2: noise_mode='none' (FedAvg, Krum, Trimmed Mean, FLTrust, FLAME,
                   Trust-Aware/NoiseGame Phase-1 SGD)
        - Phase 3: noise_mode='per_step' (DP-FedAvg, DP-SGD+Kurtosis) — requires
                   uniform per-client batch sizes (no Dirichlet padding)
        - Phase 3: noise_mode='post_training' — clip+noise applied on (N, D)
                   update after vectorized SGD
        - Data poisoning attacks that expose vectorized flip_y(): pipeline
          applies flip on attacker rows per batch (replaces Python-level
          Dataset wrapping for the GPU-resident path).
        Falls back to legacy for: data poisoning attacks WITHOUT a vectorized
        flip_y() (future backdoor/etc.), or per_step DP with non-uniform
        client sizes.
        """
        if not getattr(self.config.training, "use_vectorized", False):
            return False
        if self.train_pipeline is None or self.base_model_template is None:
            return False
        # Data poisoning: only supported when attack provides vectorized flip_y
        if self.attack is not None and hasattr(self.attack, "wrap_dataset"):
            if not hasattr(self.attack, "flip_y"):
                return False  # unsupported data poisoning -> legacy fallback
        nm = self.config.dp.noise_mode
        if nm == "per_step":
            # Phase 3 requires uniform batch sizes (no padding mid-batch)
            sizes = self.train_pipeline.client_sizes
            if int(sizes.min().item()) != int(sizes.max().item()):
                return False
        elif nm not in ("none", "post_training"):
            return False
        return True

    def _train_all_nodes_legacy(self, apply_noise: bool = True, round_t: int = 0):
        """Legacy path: ThreadPool over per-client compute_update (kept for fallback)."""
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

        # ALIE post-pass: re-perturb attacker updates with neighbor context
        if attack_active and self.attack is not None:
            from dpfl.core.alie_attack import ALIEAttack
            if isinstance(self.attack, ALIEAttack):
                for node in self.nodes.values():
                    if node.is_attacker and node.id in updates:
                        nbr_updates = {
                            nid: updates[nid] for nid in node.neighbors
                            if nid in updates and nid not in self.attacker_ids
                        }
                        context = {"neighbor_updates": nbr_updates}
                        updates[node.id] = self.attack.perturb(
                            updates[node.id], context=context)

        return updates, steps

    def _train_all_nodes_vectorized(self, apply_noise: bool, round_t: int):
        """Vectorized SGD/DP-SGD via vmap; attacker post-perturbation kept sequential.

        Branches on config.dp.noise_mode:
          - 'none'         -> plain SGD (Phase 2)
          - 'per_step'     -> nested-vmap DP-SGD (Phase 3)
          - 'post_training'-> plain SGD then clip+noise on (N, D) update
        For DP modes, attacker rows skip clip+noise (model-poisoning bypasses DP).
        """
        ordered_nodes = [self.nodes[nid] for nid in sorted(self.nodes.keys())]
        params_stack = pack_node_params(ordered_nodes, device=self.device)
        chunk = int(getattr(self.config.training, "vmap_chunk", 0) or 0)
        nm = self.config.dp.noise_mode
        attack_active = round_t >= self.config.attack.start_round

        # NOTE: data-poisoning (label_flip) ignores start_round to match
        # legacy semantics — `apply_data_attack` wraps attacker datasets ONCE
        # at setup (line 144-147) regardless of round. Pipeline.attack_active
        # stays True; only model-poisoning attacks gate on round_t below.

        # Mask of rows that are honest (or data-poisoning attackers — DP still applies)
        honest_mask = torch.ones(len(ordered_nodes), dtype=torch.bool,
                                  device=self.device)
        is_data_poisoning = self.attack is not None \
            and hasattr(self.attack, "wrap_dataset")
        if attack_active and self.attack is not None and not is_data_poisoning:
            for i, node in enumerate(ordered_nodes):
                if node.is_attacker:
                    honest_mask[i] = False  # skip DP for model-poisoning rows

        if nm == "per_step" and apply_noise:
            updates_stack, steps_dict = train_all_dpsgd_per_step(
                base_model=self.base_model_template,
                params_stack=params_stack,
                spec=self.param_spec,
                train_pipeline=self.train_pipeline,
                mechanism=self.noise_mechanism,
                batch_size=self.config.training.batch_size,
                epochs=self.config.training.local_epochs,
                lr=self.config.training.lr,
                clip_bound=self.config.dp.clip_bound,
                noise_mult=self.config.dp.noise_mult,
                chunk_size=chunk,
            )
            # Attacker rows: re-train with plain SGD (skip DP) so they can craft
            # an arbitrary update before perturb is applied.
            if (~honest_mask).any():
                plain_updates, _ = train_all_standard(
                    base_model=self.base_model_template,
                    params_stack=params_stack, spec=self.param_spec,
                    train_pipeline=self.train_pipeline,
                    batch_size=self.config.training.batch_size,
                    epochs=self.config.training.local_epochs,
                    lr=self.config.training.lr, chunk_size=chunk,
                )
                # Replace attacker rows with plain SGD updates
                updates_stack = torch.where(
                    honest_mask.unsqueeze(1), updates_stack, plain_updates,
                )
        else:
            updates_stack, steps_dict = train_all_standard(
                base_model=self.base_model_template,
                params_stack=params_stack,
                spec=self.param_spec,
                train_pipeline=self.train_pipeline,
                batch_size=self.config.training.batch_size,
                epochs=self.config.training.local_epochs,
                lr=self.config.training.lr,
                chunk_size=chunk,
            )
            if nm == "post_training" and apply_noise:
                # Each (D,) update is treated as one "sample" (B=1). Sensitivity
                # = C, sigma = z*C; matches the legacy single-client formula.
                # Attacker (model-poisoning) rows bypass DP so they can craft
                # an arbitrary update — same as legacy skip_dp behavior.
                clip = self.config.dp.clip_bound
                noised = self.noise_mechanism.clip_and_noise_batched(
                    updates_stack.unsqueeze(1), clip,
                    self.config.dp.noise_mult, batch_size=1,
                )
                updates_stack = torch.where(
                    honest_mask.unsqueeze(1), noised, updates_stack,
                )

        # Sync post-training params back to nodes so aggregators that read
        # node.model.get_flat_params() see the trained state.
        post_params = params_stack + updates_stack
        unpack_to_nodes(post_params, ordered_nodes)

        updates = {
            node.id: updates_stack[i] for i, node in enumerate(ordered_nodes)
        }

        # Model-poisoning attackers: perturb update + resync node.model.
        attack_active = round_t >= self.config.attack.start_round
        if attack_active and self.attack is not None \
                and not hasattr(self.attack, "wrap_dataset"):
            for i, node in enumerate(ordered_nodes):
                if node.is_attacker:
                    perturbed = self.attack.perturb(updates[node.id], context=None)
                    updates[node.id] = perturbed
                    node.model.set_flat_params(params_stack[i] + perturbed)

        # ALIE post-pass identical to legacy (operates on updates dict).
        if attack_active and self.attack is not None:
            from dpfl.core.alie_attack import ALIEAttack
            if isinstance(self.attack, ALIEAttack):
                # Map node_id -> stack index for resync
                idx_of = {n.id: i for i, n in enumerate(ordered_nodes)}
                for node in ordered_nodes:
                    if node.is_attacker and node.id in updates:
                        nbr_updates = {
                            nid: updates[nid] for nid in node.neighbors
                            if nid in updates and nid not in self.attacker_ids
                        }
                        context = {"neighbor_updates": nbr_updates}
                        new_upd = self.attack.perturb(updates[node.id], context=context)
                        updates[node.id] = new_upd
                        node.model.set_flat_params(
                            params_stack[idx_of[node.id]] + new_upd)

        return updates, steps_dict

    def _evaluate_nodes(self) -> Dict[int, Dict[str, float]]:
        """Evaluate accuracy and test loss for all nodes on global test set.

        Dispatches to vmapped path if config.training.use_vectorized, else the
        legacy sequential loop. Output dict shape is identical either way.
        """
        if getattr(self.config.training, "use_vectorized", False) \
                and self.base_model_template is not None:
            return self._evaluate_nodes_vectorized()

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

    def _evaluate_nodes_vectorized(self) -> Dict[int, Dict[str, float]]:
        """vmap forward over all client models on shared test set in one pass.

        Repacks params_stack from nodes (cheap) so it picks up any post-aggregation
        updates the algorithm-specific simulator made to node.model.
        """
        ordered_nodes = [self.nodes[nid] for nid in sorted(self.nodes.keys())]
        node_ids = [n.id for n in ordered_nodes]
        params_stack = pack_node_params(ordered_nodes, device=self.device)
        chunk = int(getattr(self.config.training, "vmap_chunk", 0) or 0)
        return vectorized_evaluate(
            base_model=self.base_model_template,
            params_stack=params_stack,
            spec=self.param_spec,
            node_ids=node_ids,
            X_test=self.X_test,
            Y_test=self.Y_test,
            batch_size=256,
            chunk_size=chunk,
        )

    def _compute_detection(self, flagged_ids, clean_ids, neighbors,
                           attack_active: bool = True):
        """Compute TP/FP/FN/TN from flagged vs actual attackers.

        When attack_active=False, no node is a true attacker this round —
        flagging an attacker-ID node = FP (not TP), leaving it = TN (not FN).
        """
        active_ids = self.attacker_ids if attack_active else set()
        tp = sum(1 for j in flagged_ids if j in active_ids)
        fp = sum(1 for j in flagged_ids if j not in active_ids)
        fn = sum(1 for j in clean_ids if j in active_ids)
        tn = sum(1 for j in clean_ids if j not in active_ids)
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

        # When extra_round_metrics carries eps_avg (per-node heterogeneous DP),
        # show both max and avg in the console line so users see worst-case +
        # mean-case at a glance.
        eps_avg_metric = (extra_round_metrics or {}).get("eps_avg")
        if eps_avg_metric is not None:
            eps_str = f"eps_max: {epsilon:.2f} eps_avg: {float(eps_avg_metric):.2f}"
        else:
            eps_str = f"eps: {epsilon:.2f}"
        round_msg = (
            f"Round {t + 1:3d}/{self.config.training.n_rounds} | "
            f"Acc: {accuracy:.4f} | Loss: {test_loss:.4f} | "
            f"{eps_str} | P: {precision:.2f} R: {recall:.2f} F1: {f1:.2f}")
        logger.info(round_msg)

        # Debug: per-node norms
        logger.debug("Update norms — honest: %s, attacker: %s",
                      [f"{n:.2f}" for n in honest_norms[:5]],
                      [f"{n:.2f}" for n in attacker_norms[:5]])
