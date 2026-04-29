"""Adaptive Noise DP DFL simulator — Loss-based adaptive ratio.

Implements the 9-step round loop from docs/adaptive-noise.md:
  1. Local training (no DP noise at trainer).
  2. Compute local Loss_n^t on the just-trained model (w_temp).
  3. User-level L2 clip + per-node adaptive Gaussian noise (sigma_n).
  4. P2P send/receive (frozen nodes broadcast zeros).
  5. Simple-average aggregation over self + active neighbors.
  6. Loss-based adaptive ratio + EMA:
       r_n^t = min(1, Loss_n^t / (Loss̄_n^{t-1} + eps_num))
       Loss̄_n^t = gamma * Loss̄_n^{t-1} + (1 - gamma) * Loss_n^t
     Round 1 (loss_ema_prev is None): force r_n^1 = 1 (keep sigma_0).
  7. Compute next-round sigma_{n,t+1} = max(sigma_min,
       sigma_n * (beta_min + (1 - beta_min) * r)).
  8. RDP accounting with CURRENT-round sigma_n.
  9. Freeze node if epsilon_n > epsilon_max.

Prints adaptive values (sigma, loss, ratio, decay) after every round to
verify how they evolve across training.
"""

import logging
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dpfl.core.base_simulator import BaseSimulator
from dpfl.algorithms.adaptive_noise.node import AdaptiveNoiseNode

logger = logging.getLogger(__name__)

# Issue 2 fix: hold out 10% local data as D_n^val for adaptive-ratio loss eval.
# Spec docs/adaptive-noise.md §7 recommends val (not train) so r_n^t reflects
# generalization signal, not raw SGD optimization progress. Hardcoded — not
# exposed via config (KISS; default chosen per DL convention).
_VAL_RATIO = 0.1


class AdaptiveNoiseSimulator(BaseSimulator):
    """Adaptive Noise DP Decentralized FedAvg with per-node RDP accountant.

    Adaptive signal: local Loss ratio (EMA-smoothed). Each node adapts
    its own sigma using only its local loss — no dependency on neighbors,
    robust under Non-IID, and costs 0 privacy budget (loss on local model
    is post-processing).
    """

    def __init__(self, config, an_config, dataset_cls, model_cls,
                 noise_mechanism, aggregator, attack,
                 rdp_accountant=None, tracker=None, device=None):
        # PerNodeRDPAccountant is not a BaseAccountant — pass None to super
        # and keep it as self.rdp (same pattern as TrustAware).
        super().__init__(config, dataset_cls, model_cls,
                         noise_mechanism, aggregator, attack,
                         accountant=None, tracker=tracker, device=device)
        self.an = an_config
        self.rdp = rdp_accountant
        self._prev_sigma_avg = None

    def _create_node(self, node_id, model, data, is_attacker):
        node = AdaptiveNoiseNode(node_id, model, data, is_attacker,
                                 sigma_0=self.an.sigma_0)
        if self.rdp is not None:
            self.rdp.init_node_state(node)
        return node

    def setup(self):
        """Initialize via base, then split honest nodes' data into train+val.

        Issue 2 fix: hold out _VAL_RATIO of D_n as D_n^val so the adaptive
        ratio's Loss_n^t reflects generalization (val), not optimization
        progress (train). SGD trainer mutates `node.data` for batches → we
        replace it with the train subset so training auto-uses train-only.
        Attackers don't compute loss → skipped (val_data stays None).

        Split is deterministic per (config.seed, node.id) using a prime
        multiplier (same pattern as base_simulator._make_isolated_gen) so
        identical configs yield identical splits across runs.
        """
        super().setup()
        for node in self.nodes.values():
            if node.is_attacker:
                continue
            base = node.data
            n = len(base)
            val_size = max(1, int(round(_VAL_RATIO * n)))
            rng = np.random.RandomState(int(self.config.seed) * 1_000_039 + node.id)
            indices = np.arange(n)
            rng.shuffle(indices)
            val_idx = indices[:val_size].tolist()
            train_idx = indices[val_size:].tolist()
            base_indices = base.indices
            node.val_data = Subset(base.dataset,
                                   [base_indices[i] for i in val_idx])
            node.data = Subset(base.dataset,
                               [base_indices[i] for i in train_idx])
        n_split = sum(1 for n in self.nodes.values()
                      if n.val_data is not None)
        logger.info("Adaptive-noise val split: %d honest nodes, ratio=%.2f",
                    n_split, _VAL_RATIO)

    def _clip_and_noise(self, update: torch.Tensor, sigma: float,
                        clip_bound: float) -> torch.Tensor:
        """User-level L2 clip to norm C, then add Gaussian noise with std sigma.

        Uses isolated self.noise_gen (from BaseSimulator) so noise sampling
        doesn't drift torch global RNG state across algorithms.
        """
        norm = update.norm()
        factor = min(1.0, clip_bound / (norm.item() + 1e-12))
        clipped = update * factor
        if self.noise_gen is not None:
            raw = torch.randn(clipped.shape, generator=self.noise_gen,
                              device=clipped.device, dtype=clipped.dtype)
        else:
            raw = torch.randn_like(clipped)
        return clipped + raw * sigma

    def _compute_local_loss(self, node) -> float:
        """Mean cross-entropy on D_n^val using current node.model (w_temp).

        Called right after local training (BƯỚC 2). Model is in eval mode
        for a single pass, then restored to train mode. Cost: one forward
        pass over D_n^val per round (no backward). Post-processing of the
        local model → 0 privacy budget.

        Issue 2 fix: evaluate on held-out validation subset (not train data)
        so Loss_n^t reflects generalization → adaptive ratio r correctly
        signals "still learning" vs "saturated/overfit". Falls back to
        node.data if val_data is None (safety guard; honest nodes get
        val_data via setup()).
        """
        node.model.eval()
        total_loss = 0.0
        total_n = 0
        eval_data = node.val_data if node.val_data is not None else node.data
        loader = DataLoader(eval_data, batch_size=256, shuffle=False)
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = node.model(x)
                total_loss += F.cross_entropy(out, y, reduction="sum").item()
                total_n += len(y)
        node.model.train()
        return total_loss / total_n if total_n else 0.0

    def run(self):
        C = self.config.dp.clip_bound
        T = self.config.training.n_rounds
        q = float(self.config.dp.sampling_rate)

        for t in range(T):
            # BƯỚC 0: Deterministic Poisson client subsampling via shared helper.
            # - q=1.0 → all non-frozen nodes active (backward compatible).
            # - q<1.0 → honest non-frozen nodes active w.p. q (seeded coin flip).
            #   Seeded by (config.seed, round_t) → identical schedule across
            #   algorithms for fair comparison.
            # - Attackers always active (within attack window).
            active_ids = self._sample_active_nodes(t)

            # Snapshot W_old for honest active nodes BEFORE training. Needed
            # to reset model state to (W_old + Δw̃) after clip+noise so the
            # aggregator contract `own_params - own_update == W_old` holds.
            # Attackers don't need a snapshot — their noisy_update is the raw
            # poisoned delta, so own_params - own_update == W_old already.
            W_old_map: Dict[int, torch.Tensor] = {}
            for nid in active_ids:
                node = self.nodes[nid]
                if not node.is_attacker:
                    W_old_map[nid] = node.model.get_flat_params().clone()

            # BƯỚC 1: local training — train ALL nodes (simulation cost), then
            # filter to active only. For attacker/frozen logic downstream.
            raw_updates, _ = self._train_all_nodes(apply_noise=False, round_t=t)

            # BƯỚC 2: compute local loss on w_temp for active honest nodes only.
            local_losses: Dict[int, float] = {}
            for nid in active_ids:
                node = self.nodes[nid]
                if node.is_attacker:
                    continue
                local_losses[nid] = self._compute_local_loss(node)
                node.last_loss = local_losses[nid]

            # BƯỚC 3: user-level clip + adaptive Gaussian noise for active honest.
            # Inactive nodes contribute nothing this round (skipped in aggregation).
            noisy_updates: Dict[int, torch.Tensor] = {}
            for nid in active_ids:
                node = self.nodes[nid]
                if node.is_attacker:
                    noisy_updates[nid] = raw_updates[nid]
                    continue
                noisy_updates[nid] = self._clip_and_noise(
                    raw_updates[nid], node.sigma_n, C)

            # Bug 3 fix: reset honest active models to (W_old + Δw̃) so the
            # aggregator's contract `own_params - own_update == W_old` holds.
            # Without this, models stay at (W_old + Δw_raw) → aggregator
            # back-out yields a spurious (Δw_raw - Δw̃) term per round.
            for nid, noisy in noisy_updates.items():
                node = self.nodes[nid]
                if node.is_attacker or nid not in W_old_map:
                    continue
                node.model.set_flat_params(W_old_map[nid] + noisy)

            # BƯỚC 4+5: aggregate — only active nodes update their model,
            # only from active neighbors (simple_avg over received).
            # Initialize detection/metrics for ALL nodes so _log_round can
            # iterate without KeyError. Inactive nodes get zero contributions.
            per_node_detection = {nid: (0, 0, 0, 0) for nid in self.nodes}
            node_agg_metrics = {nid: {} for nid in self.nodes}
            extra_node = {}
            attack_active = t >= self.config.attack.start_round
            total_tp = total_fp = total_fn = total_tn = 0

            for node in self.nodes.values():
                if node.id not in active_ids:
                    # Inactive: skip entirely, model unchanged.
                    continue
                nbr_ids = [j for j in node.neighbors if j in noisy_updates]
                nbr_upds = {j: noisy_updates[j] for j in nbr_ids}

                own_upd = noisy_updates[node.id]
                # Pass node_id so stateful aggregators (e.g. momentum-cosine
                # defense) can key their per-caller buffers. Stateless
                # aggregators ignore it via **_ in their signature.
                result = self.aggregator.aggregate(
                    own_upd, node.model.get_flat_params(), nbr_upds,
                    node_id=node.id)
                node.model.set_flat_params(result.new_params)
                node_agg_metrics[node.id] = result.node_metrics

                tp, fp, fn, tn = self._compute_detection(
                    result.flagged_ids, result.clean_ids, node.neighbors,
                    attack_active=attack_active)
                per_node_detection[node.id] = (tp, fp, fn, tn)
                total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn

                # Skip adaptation for attackers.
                if node.is_attacker:
                    continue

                # BƯỚC 6: Loss-based adaptive ratio + EMA (active honest only).
                loss_t = local_losses[node.id]
                if node.loss_ema_prev is None:
                    ratio = 1.0
                    new_ema = loss_t
                else:
                    ratio = min(
                        1.0,
                        loss_t / (node.loss_ema_prev + self.an.epsilon_num),
                    )
                    new_ema = (self.an.gamma * node.loss_ema_prev
                               + (1 - self.an.gamma) * loss_t)

                # BƯỚC 8: RDP step here only for bookkeeping reference — the
                # full accounting pass for ALL honest non-frozen nodes (active
                # OR inactive) runs after this loop. Amplification q² applied
                # inside accountant.step() represents the hidden-coin-flip bound.
                eps_n = 0.0

                # BƯỚC 7: update sigma for NEXT round.
                decay = self.an.beta_min + (1 - self.an.beta_min) * ratio
                sigma_before = node.sigma_n
                next_sigma = max(self.an.sigma_min, sigma_before * decay)

                extra_node[node.id] = {
                    "eps_n": eps_n,
                    "sigma_n": sigma_before,
                    "sigma_next": next_sigma,
                    "decay_factor": decay,
                    "frozen": node.frozen,
                    "loss": loss_t,
                    "loss_ema_prev": (node.loss_ema_prev
                                      if node.loss_ema_prev is not None
                                      else loss_t),
                    "adapt_ratio": ratio,
                }

                node.loss_ema_prev = new_ema
                node.sigma_n = next_sigma

            # BƯỚC 8 (post-loop): RDP accounting for ALL honest non-frozen
            # nodes using their CURRENT-round sigma_n. Under Poisson subsampling
            # at rate q, every round contributes q²·full_cost regardless of
            # whether this specific node was sampled — the amplification
            # factor comes from adversary uncertainty about the coin flip.
            #
            # Bug 1 fix: active honest nodes have already mutated `node.sigma_n`
            # to next-round σ inside the inner loop. Use sigma_before captured
            # in extra_node[nid]["sigma_n"] to charge cost against THIS round's
            # σ (per spec line 549: "DÙNG σ_{n,t} CỦA ROUND HIỆN TẠI").
            # Inactive honest nodes never entered the inner loop → their
            # node.sigma_n is still this round's value, so fall back to it.
            if self.rdp is not None:
                for nid, node in self.nodes.items():
                    if node.is_attacker or node.frozen:
                        continue
                    sigma_for_step = extra_node.get(nid, {}).get(
                        "sigma_n", node.sigma_n)
                    self.rdp.step(node, C, sigma_for_step, sampling_rate=q)
                    eps_n = self.rdp.get_epsilon(node)
                    if eps_n > self.rdp.epsilon_max:
                        node.frozen = True
                    if nid in extra_node:
                        extra_node[nid]["eps_n"] = eps_n

            # -- Aggregate stats across honest nodes for round-level logging --
            honest_eps = [extra_node[nid]["eps_n"] for nid in extra_node]
            honest_sigma = [extra_node[nid]["sigma_n"] for nid in extra_node]
            honest_sigma_next = [extra_node[nid]["sigma_next"] for nid in extra_node]
            honest_ratio = [extra_node[nid]["adapt_ratio"] for nid in extra_node]
            honest_loss = [extra_node[nid]["loss"] for nid in extra_node]
            honest_decay = [extra_node[nid]["decay_factor"] for nid in extra_node]
            n_frozen = sum(1 for nid in extra_node if extra_node[nid]["frozen"])
            eps_system = max(honest_eps) if honest_eps else 0.0
            eps_avg = float(np.mean(honest_eps)) if honest_eps else 0.0
            eps_std = float(np.std(honest_eps)) if honest_eps else 0.0

            self._log_round(
                t, eps_system, noisy_updates, per_node_detection, node_agg_metrics,
                total_tp, total_fp, total_fn, total_tn,
                extra_node_data=extra_node,
                extra_round_metrics={
                    "eps_avg": eps_avg,
                    "eps_std": eps_std,
                    "avg_sigma": float(np.mean(honest_sigma)) if honest_sigma else 0.0,
                    "avg_ratio": float(np.mean(honest_ratio)) if honest_ratio else 0.0,
                    "avg_loss_local": float(np.mean(honest_loss)) if honest_loss else 0.0,
                    "avg_decay_factor": float(np.mean(honest_decay)) if honest_decay else 0.0,
                    "n_frozen": n_frozen,
                })

            # -- Print adaptive-noise trajectory after every round --
            if honest_sigma:
                sigma_avg = float(np.mean(honest_sigma))
                sigma_next_avg = float(np.mean(honest_sigma_next))
                loss_avg = float(np.mean(honest_loss))
                ratio_avg = float(np.mean(honest_ratio))
                ratio_min = float(min(honest_ratio))
                ratio_max = float(max(honest_ratio))
                decay_avg = float(np.mean(honest_decay))

                # Within-round decay: sigma after decay vs before.
                if sigma_avg > 0:
                    delta_pct = (sigma_next_avg - sigma_avg) / sigma_avg * 100
                    delta_str = f"Δσ={delta_pct:+.3f}%"
                else:
                    delta_str = "Δσ= ——   "

                active_honest = len([n for n in active_ids
                                     if not self.nodes[n].is_attacker])
                total_honest = sum(1 for n in self.nodes.values()
                                   if not n.is_attacker and not n.frozen)
                logger.info(
                    "  [adapt] σ: %.4f→%.4f (%s) | decay=%.4f | "
                    "loss_avg=%.4f | r: avg=%.3f min=%.3f max=%.3f | "
                    "eps_avg=%.2f eps_max=%.2f | active=%d/%d | frozen=%d/%d",
                    sigma_avg, sigma_next_avg, delta_str, decay_avg,
                    loss_avg, ratio_avg, ratio_min, ratio_max,
                    eps_avg, eps_system,
                    active_honest, total_honest,
                    n_frozen, len(honest_sigma),
                )
                self._prev_sigma_avg = sigma_next_avg

            # Stop if ALL honest nodes are frozen.
            if honest_eps and n_frozen == len(honest_eps):
                logger.warning("Round %3d/%d | All honest nodes frozen — stop.",
                               t + 1, T)
                break
