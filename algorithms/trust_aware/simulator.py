"""Trust-Aware D2B-DP simulator.

Maps to docs/Trust-Aware-D2B-DP.md, the four phases per round t:

  Phase 1 — Local Training:        ΔW^(t) = W_trained − W_i^(t-1)
  Phase 2 — Outbound Processing:   per-layer adaptive Clip_l (Step 2.1)
                                   + Gaussian noise σ²_l = 2·Clip_l²/ρ^(t)
                                   under ρ^(t) = min((1+βt)·ρ_min, ρ_max)
                                   (Steps 2.2–2.3)
  Phase 3 — Inbound Evaluation:    cosine + RMSE distance vs. own ΔW'_i,
                                   D_threshold = max(γ·exp(-κ·t)·rms_self,
                                                     C_DP^(t))
                                   with C_DP^(t) = θ·√((1/D_total)·Σ d_l·σ²_l)
                                   (Steps 3.1–3.3)
  Phase 4 — History & Aggregation: Q = p_dist·p_cos, T ← α_T·T + (1-α_T)·Q,
                                   safe set V = {j | T ≥ T_min},
                                   softmax (β_soft) over V,
                                   V_agg = β_m·V_agg + (1-β_m)·S_agg,
                                   W ← W_old + V_agg          (Step 4.1–4.3)

Privacy ε reporting uses Opacus SGM with z = √(2/ρ_t). For sampling_rate=1.0
this matches the spec's Step 2.3 bound: ε^(t)(α) = α·ρ_t/4 (since
(Clip_l/σ_l)² = ρ_t/2 for every layer under our σ² = 2·Clip²/ρ schedule).
"""

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
import torch

from dpfl.algorithms.trust_aware.node import TrustAwareNode
from dpfl.core.base_simulator import BaseSimulator

logger = logging.getLogger(__name__)


class TrustAwareDFLSimulator(BaseSimulator):
    """Orchestrator for Trust-Aware D2B-DP."""

    def __init__(self, config, trust_config, dataset_cls, model_cls,
                 noise_mechanism, aggregator, attack,
                 adaptive_clipper, gaussian_noise,
                 accountant=None, tracker=None, device=None):
        super().__init__(config, dataset_cls, model_cls,
                         noise_mechanism, aggregator, attack,
                         accountant=accountant, tracker=tracker, device=device)
        self.tc = trust_config
        self.clipper = adaptive_clipper
        self.gaussian_noise = gaussian_noise
        if hasattr(self.gaussian_noise, "set_generator"):
            self.gaussian_noise.set_generator(self.noise_gen)
        self._layer_sizes: List[int] = []
        # Per-node RDP accountants — required for honest reporting under
        # heterogeneous per-layer-per-node clip & noise. Even though the
        # noise multiplier z_eff = √(2/ρ_t) is nominally uniform, we track
        # per-node so future per-node z variants drop in cleanly. Per-layer
        # composition is applied via n_steps=L (number of model layers).
        self._per_node_acc = {}

    def _create_node(self, node_id, model, data, is_attacker):
        return TrustAwareNode(node_id, model, data, is_attacker, self.tc)

    def setup(self):
        super().setup()
        sample_model = next(iter(self.nodes.values())).model
        self._layer_sizes = [p.numel() for p in sample_model.parameters()]
        for node in self.nodes.values():
            node.init_d2b_state(self._layer_sizes, self.param_dim,
                                self.tc.k, self.tc.trust_init,
                                device=self.device)

    # ── Phase 2: per-layer clip + Gaussian noise ────────────────────────────

    def _build_packet(self, node: TrustAwareNode, raw_update: torch.Tensor,
                      rho_t: float, attack_active: bool):
        """Per-layer clip → Gaussian noise. Returns (own_clipped_flat,
        packet_flat, sigma_sq_per_layer). Attackers bypass clip + noise
        entirely (model-poisoning by spec) — raw_update propagates as-is to
        both own_clipped and packet so D_threshold and aggregator see the
        full perturbed update."""
        layers = list(torch.split(raw_update, self._layer_sizes))
        if node.is_attacker:
            return raw_update, raw_update.clone(), [0.0] * len(layers)

        for li, layer in enumerate(layers):
            node.clip_history[li].append(layer.norm(2).item())
        thresholds = self.clipper.get_thresholds(node.clip_history)
        clipped_layers = self.clipper.clip(layers, thresholds)
        own_clipped = torch.cat([cl.reshape(-1) for cl in clipped_layers])

        sigma_sqs: List[float] = []
        noisy_layers: List[torch.Tensor] = []
        for cl_layer, thr in zip(clipped_layers, thresholds):
            clip_l = thr if thr is not None else cl_layer.norm(2).item()
            sigma_sq = self.gaussian_noise.compute_noise_variance(clip_l, rho_t)
            sigma_sqs.append(sigma_sq)
            noisy_layers.append(self.gaussian_noise.add_noise(cl_layer, sigma_sq))
        packet = torch.cat([nl.reshape(-1) for nl in noisy_layers])
        return own_clipped, packet, sigma_sqs

    # ── Phase 3 helper: D_threshold (Step 3.3) ──────────────────────────────

    def _compute_d_threshold(self, own_clipped: torch.Tensor,
                             own_sigma_sqs: List[float], t: int) -> float:
        # C_DP^(t) = θ · √( (1/D_total) · Σ d_l · σ²_l )
        weighted_var = sum(d * s for d, s in
                           zip(self._layer_sizes, own_sigma_sqs))
        weighted_var /= max(self.param_dim, 1)
        c_dp = self.tc.theta * math.sqrt(max(weighted_var, 0.0))
        # Decay term: γ · exp(-κ · t) · ||ΔW'_i||₂ / √D_total
        rms_self = own_clipped.norm(2).item() / math.sqrt(max(self.param_dim, 1))
        decay = self.tc.gamma * math.exp(-self.tc.kappa * (t / self.config.training.n_rounds)) * rms_self
        return max(decay, c_dp)

    # ── Main loop ───────────────────────────────────────────────────────────

    def run(self):
        n_workers = self.config.training.n_workers
        sampling_rate = float(self.config.dp.sampling_rate)
        logger.info("Trust-Aware D2B-DP: rounds=%d, n_workers=%d, layers=%d, ρ∈[%.3f,%.3f]",
                    self.config.training.n_rounds, n_workers,
                    len(self._layer_sizes), self.tc.rho_min, self.tc.rho_max)

        for t in range(self.config.training.n_rounds):
            active_ids = self._sample_active_nodes(t)
            attack_active = t >= self.config.attack.start_round
            rho_t = min((1.0 + self.tc.beta * t) * self.tc.rho_min,
                        self.tc.rho_max)

            # Phase 1: snapshot W_old, train, build per-layer noisy packets.
            W_old_map: Dict[int, torch.Tensor] = {}
            own_clipped: Dict[int, torch.Tensor] = {}
            packets: Dict[int, torch.Tensor] = {}
            sigma_sq_map: Dict[int, List[float]] = {}
            updates: Dict[int, torch.Tensor] = {}

            t0 = time.time()
            for node in self.nodes.values():
                W_old_map[node.id] = node.model.get_flat_params().clone()

            raw_updates: Dict[int, torch.Tensor] = {}
            if self._can_use_vectorized_training():
                # Phase 5a: vectorized vmap'd SGD over all clients in one pass.
                # noise_mode='none' for Trust-Aware (self-managed DP via Phase 2),
                # so apply_noise is a no-op here. Attack perturbation is applied
                # inside _train_all_nodes_vectorized for model-poisoning rows.
                # We still keep only active_ids — train cost is amortized but
                # we filter to match the legacy raw_updates contract.
                all_updates, _ = self._train_all_nodes_vectorized(
                    apply_noise=False, round_t=t)
                raw_updates = {
                    nid: u for nid, u in all_updates.items() if nid in active_ids
                }
            else:
                def _train(node):
                    if node.id not in active_ids:
                        return node.id, None
                    atk = self.attack if (node.is_attacker and attack_active) else None
                    upd, _ = node.compute_update(
                        self.trainer, self.noise_mechanism, atk, apply_noise=False)
                    # Spec sends actual delta ΔW = W_trained - W_old.
                    return node.id, upd

                if n_workers > 1:
                    with ThreadPoolExecutor(max_workers=n_workers) as pool:
                        for nid, upd in pool.map(_train, self.nodes.values()):
                            if upd is not None:
                                raw_updates[nid] = upd
                else:
                    for node in self.nodes.values():
                        nid, upd = _train(node)
                        if upd is not None:
                            raw_updates[nid] = upd

            # Phase 2: per-layer clip + Gaussian noise (sequential — mutates
            # per-node clip_history FIFO).
            for node in self.nodes.values():
                if node.id not in raw_updates:
                    continue
                upd = raw_updates[node.id]
                updates[node.id] = upd
                oc, pkt, ssqs = self._build_packet(node, upd, rho_t, attack_active)
                own_clipped[node.id] = oc
                packets[node.id] = pkt
                sigma_sq_map[node.id] = ssqs
            logger.debug("Round %d Phase 1+2: %.2fs", t, time.time() - t0)

            # Phase 3+4: inbound eval + soft trust aggregation.
            tp_all = fp_all = fn_all = tn_all = 0
            per_node_det = {nid: (0, 0, 0, 0) for nid in self.nodes}
            node_agg_metrics: Dict[int, dict] = {nid: {} for nid in self.nodes}

            for node in self.nodes.values():
                if node.id not in active_ids or node.id not in own_clipped:
                    continue
                d_thr = self._compute_d_threshold(
                    own_clipped[node.id], sigma_sq_map[node.id], t)
                received = {j: packets[j] for j in node.neighbors
                            if j in active_ids and j in packets}
                result = self.aggregator.aggregate(
                    own_update=own_clipped[node.id],
                    own_params=node.model.get_flat_params(),
                    neighbor_updates=received,
                    W_old=W_old_map[node.id],
                    V_agg_prev=node.V_agg,
                    D_threshold=d_thr,
                    trust_scores=node.trust_scores,
                    D_total=self.param_dim,
                )
                v_agg_new = result.metrics.get("V_agg")
                if v_agg_new is not None:
                    node.V_agg = v_agg_new.detach()
                node.model.set_flat_params(result.new_params)
                node_agg_metrics[node.id] = result.node_metrics
                tp, fp, fn, tn = self._compute_detection(
                    result.flagged_ids, result.clean_ids, node.neighbors,
                    attack_active=attack_active)
                if node.is_attacker:
                    tp, fp, fn, tn = 0, 0, 0, 0
                per_node_det[node.id] = (tp, fp, fn, tn)
                tp_all += tp; fp_all += fp; fn_all += fn; tn_all += tn

            # Privacy ε reporting (Step 2.3): per-layer Gaussian → L
            # sequential SGM applications per round (RDP composes additively).
            # Per-node accountant tracks each honest node's cumulative cost;
            # ε reported as max(ε_k) for guarantee + avg(ε_k) for diagnostic.
            epsilon = 0.0
            eps_avg = 0.0
            per_node_eps: Dict[int, float] = {}
            if self.accountant is not None and rho_t > 0:
                from dpfl.core.renyi_accountant import RenyiAccountant
                z_eff = math.sqrt(2.0 / max(rho_t, 1e-12))
                L = max(len(self._layer_sizes), 1)  # # layers per Gaussian round
                for nid, node in self.nodes.items():
                    if node.is_attacker:
                        continue
                    acc = self._per_node_acc.get(nid)
                    if acc is None:
                        acc = RenyiAccountant(
                            alpha_list=self.accountant.alpha_list,
                            delta=self.accountant.delta)
                        self._per_node_acc[nid] = acc
                    # Per-layer composition: L Gaussian mechanisms per round
                    acc.step(n_steps=L, sampling_rate=sampling_rate,
                             noise_mult=z_eff)
                    per_node_eps[nid] = acc.get_epsilon()

                if per_node_eps:
                    epsilon = max(per_node_eps.values())
                    eps_avg = float(np.mean(list(per_node_eps.values())))

            # Trust diagnostics — split by recipient role (honest vs attacker).
            t_h, t_a = [], []
            for n in self.nodes.values():
                if n.id in self.attacker_ids:
                    continue
                for j, tv in n.trust_scores.items():
                    (t_a if j in self.attacker_ids else t_h).append(tv)

            # Per-node ε for node_data export
            extra_node_data = {nid: {"eps_n": e}
                               for nid, e in per_node_eps.items()}

            eps_std = (float(np.std(list(per_node_eps.values())))
                       if per_node_eps else 0.0)

            self._log_round(
                t, epsilon, updates, per_node_det, node_agg_metrics,
                tp_all, fp_all, fn_all, tn_all,
                extra_node_data=extra_node_data,
                extra_round_metrics={
                    "eps_avg": eps_avg,
                    "eps_std": eps_std,
                    "rho_t": float(rho_t),
                    "trust_toward_honest": float(np.mean(t_h)) if t_h else 0.0,
                    "trust_toward_attacker": float(np.mean(t_a)) if t_a else 0.0,
                })

            if (self.accountant is not None
                    and epsilon > self.config.dp.epsilon_max):
                logger.warning(
                    "Round %3d/%d | Budget exceeded (eps_max=%.2f > %.2f)",
                    t + 1, self.config.training.n_rounds,
                    epsilon, self.config.dp.epsilon_max)
                break
