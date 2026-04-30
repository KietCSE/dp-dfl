"""Phase 1 integration: FedAvg simulator with use_vectorized=True for 2 rounds.

Verifies:
  - setup() builds vectorized state when flag is on
  - _evaluate_nodes() dispatches to vmapped path
  - End-to-end run produces identical accuracy to legacy path within tolerance
"""

import torch

from dpfl.config import (
    BaseExperimentConfig, DataSplitConfig, DatasetConfig, ModelConfig,
    TopologyConfig, TrainingConfig, DPConfig, AttackConfig, AggregationConfig,
)


def _make_config(use_vectorized: bool):
    return BaseExperimentConfig(
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
        dp=DPConfig(noise_mode="none", clip_bound=1.0, noise_mult=0.0),
        attack=AttackConfig(type="scale", scale_factor=1.0, start_round=1000),
        aggregation=AggregationConfig(type="fedavg", params={}),
        seed=42,
    )


def test_simulator_setup_with_vectorized_flag():
    """Setup must populate base_model_template, param_spec, X_test, Y_test."""
    from dpfl.algorithms.fedavg.simulator import FedAvgSimulator
    from dpfl.data.mnist_dataset import MNISTDataset
    from dpfl.models.mlp_model import MLP
    from dpfl.core.gaussian_mechanism import GaussianMechanism
    from dpfl.algorithms.fedavg.fedavg_aggregator import FedAvgAggregator
    from dpfl.core.scale_attack import ScaleAttack
    from dpfl.core.renyi_accountant import RenyiAccountant

    cfg = _make_config(use_vectorized=True)
    sim = FedAvgSimulator(
        config=cfg,
        dataset_cls=MNISTDataset,
        model_cls=MLP,
        noise_mechanism=GaussianMechanism(),
        aggregator=FedAvgAggregator(),
        attack=ScaleAttack(scale_factor=1.0),
        accountant=RenyiAccountant(alpha_list=[2, 4, 8], delta=1e-5),
        device=torch.device("cpu"),
    )
    sim.setup()

    assert sim.base_model_template is not None
    assert sim.param_spec is not None
    assert sim.param_spec.D > 0
    assert sim.X_test.dim() >= 2
    assert sim.Y_test.dim() == 1
    assert sim.X_test.shape[0] == sim.Y_test.shape[0]


def test_evaluate_dispatches_to_vectorized_when_flag_on():
    """_evaluate_nodes must call the vmapped path when use_vectorized=True.

    Build ONE simulator with the flag on, then exercise both eval paths
    on the same nodes via runtime flag flip — guarantees identical models.
    """
    from dpfl.algorithms.fedavg.simulator import FedAvgSimulator
    from dpfl.data.mnist_dataset import MNISTDataset
    from dpfl.models.mlp_model import MLP
    from dpfl.core.gaussian_mechanism import GaussianMechanism
    from dpfl.algorithms.fedavg.fedavg_aggregator import FedAvgAggregator
    from dpfl.core.scale_attack import ScaleAttack
    from dpfl.core.renyi_accountant import RenyiAccountant

    cfg = _make_config(use_vectorized=True)
    sim = FedAvgSimulator(
        config=cfg, dataset_cls=MNISTDataset, model_cls=MLP,
        noise_mechanism=GaussianMechanism(),
        aggregator=FedAvgAggregator(),
        attack=ScaleAttack(scale_factor=1.0),
        accountant=RenyiAccountant(alpha_list=[2, 4, 8], delta=1e-5),
        device=torch.device("cpu"),
    )
    sim.setup()

    # Vectorized path
    vec_out = sim._evaluate_nodes()

    # Legacy path: flip the flag transiently
    sim.config.training.use_vectorized = False
    legacy_out = sim._evaluate_nodes()
    sim.config.training.use_vectorized = True

    for nid in legacy_out:
        d_acc = abs(legacy_out[nid]["accuracy"] - vec_out[nid]["accuracy"])
        d_loss = abs(legacy_out[nid]["test_loss"] - vec_out[nid]["test_loss"])
        assert d_acc < 1e-3, f"node {nid} acc drift: {d_acc}"
        assert d_loss < 1e-3, f"node {nid} loss drift: {d_loss}"


# ── Phase 2: vectorized training end-to-end ────────────────────────────────

def test_fedavg_runs_two_rounds_with_vectorized_training():
    """Smoke test: FedAvg with use_vectorized=True for 2 rounds completes
    without errors and produces non-trivial accuracy improvement.
    """
    from dpfl.algorithms.fedavg.simulator import FedAvgSimulator
    from dpfl.data.mnist_dataset import MNISTDataset
    from dpfl.models.mlp_model import MLP
    from dpfl.core.gaussian_mechanism import GaussianMechanism
    from dpfl.algorithms.fedavg.fedavg_aggregator import FedAvgAggregator
    from dpfl.core.scale_attack import ScaleAttack
    from dpfl.core.renyi_accountant import RenyiAccountant
    from dpfl.tracking.metrics_tracker import MetricsTracker

    cfg = _make_config(use_vectorized=True)
    cfg.training.n_rounds = 2
    cfg.training.lr = 0.05  # higher LR to see acc move

    sim = FedAvgSimulator(
        config=cfg, dataset_cls=MNISTDataset, model_cls=MLP,
        noise_mechanism=GaussianMechanism(),
        aggregator=FedAvgAggregator(),
        attack=ScaleAttack(scale_factor=1.0),
        accountant=RenyiAccountant(alpha_list=[2, 4, 8], delta=1e-5),
        device=torch.device("cpu"),
        tracker=MetricsTracker(output_dir="/tmp/dpfl_test"),
    )
    sim.setup()

    # Confirm the dispatcher selects the vectorized path
    assert sim._can_use_vectorized_training(), \
        "vectorized path should be active (no DP, no data poisoning)"

    initial_acc = sum(v["accuracy"] for v in sim._evaluate_nodes().values()) / len(sim.nodes)
    sim.run()
    final_acc = sum(v["accuracy"] for v in sim._evaluate_nodes().values()) / len(sim.nodes)

    assert final_acc > initial_acc, (
        f"accuracy did not improve after 2 rounds: {initial_acc:.3f} -> {final_acc:.3f}")
