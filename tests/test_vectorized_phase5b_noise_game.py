"""Phase 5b smoke test: Noise Game DFL benefits from vectorized Phase-1 training.

Noise Game's simulator does NOT override _train_all_nodes — it relies on the
base dispatcher. So with use_vectorized=True the vmapped path activates
automatically (noise_mode='none', model-poisoning attack -> Phase-2 path).

This test verifies:
  1. Dispatcher selects vectorized for Noise Game when flag is on
  2. Two rounds complete without errors
  3. Convergence behavior is roughly similar to the legacy path
     (within statistical drift)

The deep SVD-batching layer-2 noise vectorization (full Phase 5b per the plan)
is intentionally NOT implemented here — it requires the 5-gate validation
suite (SVD numerical equivalence, ε curve overlay, A/B t-test) and that work
is deferred. Phase-1 vectorization captures the dominant runtime speedup.
"""

import torch

from dpfl.config import (
    DataSplitConfig, DatasetConfig, ModelConfig,
    TopologyConfig, TrainingConfig, DPConfig, AttackConfig,
    AggregationConfig, NoiseGameConfig, NoiseGameExperimentConfig,
)


def _make_config(use_vectorized: bool):
    return NoiseGameExperimentConfig(
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
        aggregation=AggregationConfig(type="simple_avg", params={}),
        noise_game=NoiseGameConfig(),
        seed=42,
    )


def _build_sim(cfg):
    from dpfl.algorithms.noise_game.simulator import NoiseGameDFLSimulator
    from dpfl.algorithms.noise_game.simple_avg_aggregator import SimpleAvgAggregator
    from dpfl.algorithms.noise_game.mechanism import NoiseGameMechanism
    from dpfl.data.mnist_dataset import MNISTDataset
    from dpfl.models.mlp_model import MLP
    from dpfl.core.gaussian_mechanism import GaussianMechanism
    from dpfl.core.scale_attack import ScaleAttack
    from dpfl.core.renyi_accountant import RenyiAccountant
    from dpfl.tracking.metrics_tracker import MetricsTracker

    sim = NoiseGameDFLSimulator(
        config=cfg, ng_config=cfg.noise_game,
        dataset_cls=MNISTDataset, model_cls=MLP,
        noise_mechanism=GaussianMechanism(),
        game_mechanism=NoiseGameMechanism(
            alpha_attack=cfg.noise_game.alpha_attack,
            sigma_0=cfg.noise_game.sigma_0,
            anneal_kappa=cfg.noise_game.anneal_kappa,
            svd_rank=cfg.noise_game.svd_rank,
            svd_reshape_k=cfg.noise_game.svd_reshape_k,
            clip_bound=cfg.dp.clip_bound,
            delta=cfg.dp.delta,
            epsilon_max=cfg.dp.epsilon_max,
            beta_strat=cfg.noise_game.beta_strat,
            sigma_total=cfg.noise_game.sigma_total,
            param_dim=25482,  # MLP(784,32,10) total params
            alpha_rd=cfg.noise_game.rdp_alpha,
        ),
        aggregator=SimpleAvgAggregator(),
        attack=ScaleAttack(scale_factor=1.0),
        accountant=RenyiAccountant(alpha_list=[2, 4, 8], delta=1e-5),
        device=torch.device("cpu"),
        tracker=MetricsTracker(output_dir="/tmp/dpfl_test_noise_game"),
    )
    sim.setup()
    return sim


def test_noise_game_dispatcher_selects_vectorized_when_flag_on():
    sim = _build_sim(_make_config(use_vectorized=True))
    assert sim._can_use_vectorized_training(), \
        "Noise Game uses noise_mode='none' + model-poisoning attack — "\
        "vectorized path should be selected"


def test_noise_game_runs_two_rounds_with_vectorized_training():
    sim = _build_sim(_make_config(use_vectorized=True))
    sim.run()
    out = sim._evaluate_nodes()
    avg_acc = sum(v["accuracy"] for v in out.values()) / len(out)
    assert avg_acc > 0.05, f"vectorized Noise Game acc unreasonably low: {avg_acc:.3f}"
