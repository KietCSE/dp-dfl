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
                      rho_t: float):
        """Per-layer clip → Gaussian noise. Returns (own_clipped_flat,
        packet_flat, sigma_sq_per_layer). Attackers skip noise but still get
        clipped (their own clipping history is also tracked)."""
        layers = list(torch.split(raw_update, self._layer_sizes))
        for li, layer in enumerate(layers):
            node.clip_history[li].append(layer.norm(2).item())
        thresholds = self.clipper.get_thresholds(node.clip_history)
        clipped_layers = self.clipper.clip(layers, thresholds)
        own_clipped = torch.cat([cl.reshape(-1) for cl in clipped_layers])

        if node.is_attacker:
            return own_clipped, own_clipped.clone(), [0.0] * len(layers)

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
        decay = self.tc.gamma * math.exp(-self.tc.kappa * t) * rms_self
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

            def _train(node):
                if node.id not in active_ids:
                    return node.id, None
                atk = self.attack if (node.is_attacker and attack_active) else None
                upd, _ = node.compute_update(
                    self.trainer, self.noise_mechanism, atk, apply_noise=False)
                # Spec sends actual delta ΔW = W_trained - W_old, not gradient.
                return node.id, upd

            raw_updates: Dict[int, torch.Tensor] = {}
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
                oc, pkt, ssqs = self._build_packet(node, upd, rho_t)
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
                per_node_det[node.id] = (tp, fp, fn, tn)
                tp_all += tp; fp_all += fp; fn_all += fn; tn_all += tn

            # Privacy ε reporting (Step 2.3): one Gaussian round at
            # z = √(2/ρ_t). Under sampling_rate=1.0 this reproduces
            # ε^(t)(α) = α·ρ_t/4. Sub-sampling tightens via Opacus SGM.
            epsilon = 0.0
            if self.accountant is not None and rho_t > 0:
                z_eff = math.sqrt(2.0 / max(rho_t, 1e-12))
                self.accountant.step(
                    n_steps=1, sampling_rate=sampling_rate, noise_mult=z_eff)
                epsilon = self.accountant.get_epsilon()

            # Trust diagnostics — split by recipient role (honest vs attacker).
            t_h, t_a = [], []
            for n in self.nodes.values():
                if n.id in self.attacker_ids:
                    continue
                for j, tv in n.trust_scores.items():
                    (t_a if j in self.attacker_ids else t_h).append(tv)

            self._log_round(
                t, epsilon, updates, per_node_det, node_agg_metrics,
                tp_all, fp_all, fn_all, tn_all,
                extra_round_metrics={
                    "rho_t": float(rho_t),
                    "trust_toward_honest": float(np.mean(t_h)) if t_h else 0.0,
                    "trust_toward_attacker": float(np.mean(t_a)) if t_a else 0.0,
                })
