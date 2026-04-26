"""Trust-Aware D2B-DP simulator: layer-wise clip+noise → RMS+softmax+momentum.

Maps to docs/Trust-Aware-D2B-DP.md Section 2 ("Quy trình thuật toán tại Vòng t"):

  Phase 1  — Local training:        Δ = W_trained − W_old   (Step 1)
  Phase 2  — Outbound processing:   per-layer Clip_l + bounded Gaussian noise
                                    using ρ^(t) schedule    (Steps 2-3)
  Phase 3  — Inbound evaluation:    RMS distance + cosine vs own ΔW'_i,
                                    D_threshold = max(γ·exp(-κt/T_max)·rms_self,
                                                      θ·sqrt(weighted σ²))
                                                            (Steps 4-5)
  Phase 4  — Trust + aggregation:   trust EMA, softmax over T ≥ T_min,
                                    momentum, global step   (Steps 6-7)

Privacy ε reporting (heuristic): use Opacus SGM with z = sqrt(2/ρ_t) per round.
This is a cross-algo comparison aid — the algorithm itself does not track a
hard privacy budget (ρ is treated as a noise budget per Step 3's formula).
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
    """Orchestrator for Trust-Aware D2B-DP (RMS+Softmax+Momentum version)."""

    def __init__(self, config, trust_config, dataset_cls, model_cls,
                 noise_mechanism, aggregator, attack,
                 adaptive_clipper, bounded_noise,
                 accountant=None, tracker=None, device=None):
        super().__init__(config, dataset_cls, model_cls,
                         noise_mechanism, aggregator, attack,
                         accountant=accountant, tracker=tracker, device=device)
        self.tc = trust_config
        self.clipper = adaptive_clipper
        self.bounded_noise = bounded_noise
        if hasattr(self.bounded_noise, "set_generator"):
            self.bounded_noise.set_generator(self.noise_gen)
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

    # ── Phase 1+2: train + per-layer clip + per-layer bounded noise ──────────

    def _build_packet(self, node: TrustAwareNode, raw_update: torch.Tensor,
                      rho_t: float):
        """Per-layer clip → noise. Returns (own_clipped_flat, packet_flat,
        sigma_sq_per_layer). Attackers skip noise but still get clipped."""
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
            sigma_sq = self.bounded_noise.compute_noise_variance(clip_l, rho_t)
            sigma_sqs.append(sigma_sq)
            noisy_layers.append(
                self.bounded_noise.add_bounded_noise(cl_layer, sigma_sq))
        packet = torch.cat([nl.reshape(-1) for nl in noisy_layers])
        return own_clipped, packet, sigma_sqs

    # ── Phase 3 helper: D_threshold from receiver-side σ² and own RMS norm ──

    def _compute_d_threshold(self, own_clipped: torch.Tensor,
                             own_sigma_sqs: List[float], t: int,
                             T_max: int) -> float:
        weighted_var = sum(d * s for d, s in
                           zip(self._layer_sizes, own_sigma_sqs))
        weighted_var /= max(self.param_dim, 1)
        c_dp = self.tc.theta * math.sqrt(max(weighted_var, 0.0))
        rms_self = own_clipped.norm(2).item() / math.sqrt(max(self.param_dim, 1))
        decay = self.tc.gamma * math.exp(
            -self.tc.kappa * t / max(T_max, 1)) * rms_self
        return max(decay, c_dp)

    # ── Main loop ───────────────────────────────────────────────────────────

    def run(self):
        T_max = max(self.config.training.n_rounds, 1)
        n_workers = self.config.training.n_workers
        sampling_rate = float(self.config.dp.sampling_rate)
        logger.info("Trust-Aware D2B-DP: T_max=%d, n_workers=%d, layers=%d, ρ∈[%.3f,%.3f]",
                    T_max, n_workers, len(self._layer_sizes),
                    self.tc.rho_min, self.tc.rho_max)

        for t in range(self.config.training.n_rounds):
            active_ids = self._sample_active_nodes(t)
            attack_active = t >= self.config.attack.start_round
            rho_t = min((1.0 + self.tc.beta * t) * self.tc.rho_min,
                        self.tc.rho_max)

            # Phase 1: Capture W_old, train, build per-layer noisy packets.
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

            # Phase 2: per-layer clip + bounded Gaussian noise (sequential —
            # mutates per-node clip_history).
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

            # Phase 3+4: inbound eval + soft trust aggregation (active receivers).
            tp_all = fp_all = fn_all = tn_all = 0
            per_node_det = {nid: (0, 0, 0, 0) for nid in self.nodes}
            node_agg_metrics: Dict[int, dict] = {nid: {} for nid in self.nodes}

            for node in self.nodes.values():
                if node.id not in active_ids or node.id not in own_clipped:
                    continue
                d_thr = self._compute_d_threshold(
                    own_clipped[node.id], sigma_sq_map[node.id], t, T_max)
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

            # Privacy ε reporting (heuristic): treat each round as one Gaussian
            # mechanism step with unit-sensitivity z = sqrt(2/ρ_t).
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
