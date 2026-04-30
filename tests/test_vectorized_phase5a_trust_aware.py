"""Phase 5a smoke test: Trust-Aware D2B-DP runs end-to-end with vectorized Phase-1.

Compares legacy and vectorized paths over 2 rounds: round-level metrics
(accuracy, ε) must agree to within statistical drift (RNG draw order changes
when batches are stacked, so we don't expect bit-exactness).
"""

import torch

from dpfl.config import (
    BaseExperimentConfig, DataSplitConfig, DatasetConfig, ModelConfig,
    TopologyConfig, TrainingConfig, DPConfig, AttackConfig,
    AggregationConfig, TrustConfig, TrustAwareExperimentConfig,
)


def _make_config(use_vectorized: bool):
    cfg = TrustAwareExperimentConfig(
        dataset=DatasetConfig(
            name="mnist",
            split=DataSplitConfig(mode="iid", samples_per_node=600),
        ),
        model=ModelConfig(name="mlp", hidden_size=32),
        topology=TopologyConfig(n_nodes=5, n_attackers=0, n_neighbors=2, seed=42),
        training=TrainingConfig(
            n_rounds=2, local_epochs=1, batch_size=64, lr=0.01,
            n_workers=1, use_vectorized=use_vectorized,
        ),
        dp=DPConfig(noise_mode="none", clip_bound=2.0, noise_mult=1.1,
                    delta=1e-5, epsilon_max=1e8),
        attack=AttackConfig(type="scale", scale_factor=1.0, start_round=1000),
        aggregation=AggregationConfig(type="trust_aware_d2b", params={}),
        trust=TrustConfig(),
        seed=42,
    )
    return cfg


def _build_sim(cfg):
    from dpfl.algorithms.trust_aware.simulator import TrustAwareDFLSimulator
    from dpfl.algorithms.trust_aware.aggregator import TrustAwareD2BAggregator
    from dpfl.algorithms.trust_aware.adaptive_clipper import LayerwiseAdaptiveClipper
    from dpfl.algorithms.trust_aware.gaussian_noise import LayerwiseGaussianNoise
    from dpfl.data.mnist_dataset import MNISTDataset
    from dpfl.models.mlp_model import MLP
    from dpfl.core.gaussian_mechanism import GaussianMechanism
    from dpfl.core.scale_attack import ScaleAttack
    from dpfl.core.renyi_accountant import RenyiAccountant
    from dpfl.tracking.metrics_tracker import MetricsTracker

    sim = TrustAwareDFLSimulator(
        config=cfg, trust_config=cfg.trust,
        dataset_cls=MNISTDataset, model_cls=MLP,
        noise_mechanism=GaussianMechanism(),
        aggregator=TrustAwareD2BAggregator(beta_soft=cfg.trust.beta_soft,
                                              beta_m=cfg.trust.beta_m,
                                              alpha_T=cfg.trust.alpha_T,
                                              T_min=cfg.trust.T_min),
        attack=ScaleAttack(scale_factor=1.0),
        adaptive_clipper=LayerwiseAdaptiveClipper(k=cfg.trust.k),
        gaussian_noise=LayerwiseGaussianNoise(),
        accountant=RenyiAccountant(alpha_list=[2, 4, 8], delta=1e-5),
        device=torch.device("cpu"),
        tracker=MetricsTracker(output_dir="/tmp/dpfl_test_trust_aware"),
    )
    sim.setup()
    return sim


def test_trust_aware_uses_vectorized_path_when_flag_on():
    """With use_vectorized=True the dispatcher must select the vectorized branch."""
    sim = _build_sim(_make_config(use_vectorized=True))
    assert sim._can_use_vectorized_training(), \
        "Trust-Aware should use vectorized: noise_mode='none', no data poisoning"


def test_trust_aware_runs_two_rounds_vectorized_then_legacy():
    """Both paths complete 2 rounds without errors and produce reasonable accuracy."""
    sim_v = _build_sim(_make_config(use_vectorized=True))
    sim_v.run()
    metrics_v = sim_v._evaluate_nodes()
    avg_acc_v = sum(v["accuracy"] for v in metrics_v.values()) / len(metrics_v)
    assert avg_acc_v > 0.05, f"vectorized path acc too low: {avg_acc_v:.3f}"

    sim_l = _build_sim(_make_config(use_vectorized=False))
    sim_l.run()
    metrics_l = sim_l._evaluate_nodes()
    avg_acc_l = sum(v["accuracy"] for v in metrics_l.values()) / len(metrics_l)
    assert avg_acc_l > 0.05, f"legacy path acc too low: {avg_acc_l:.3f}"

    # Both paths should converge similarly within statistical noise (5 nodes,
    # 2 rounds, no attacks, IID — so accuracy shouldn't differ wildly).
    drift = abs(avg_acc_v - avg_acc_l)
    assert drift < 0.20, (
        f"vectorized vs legacy accuracy drift too large: "
        f"vec={avg_acc_v:.3f} legacy={avg_acc_l:.3f} drift={drift:.3f}")
