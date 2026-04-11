"""Entry point for Trust-Aware D2B-DP: python -m dpfl.trust-aware-main"""

import sys
import random
import shutil
import importlib
from datetime import datetime
from math import prod
from pathlib import Path

import torch
import numpy as np

from dpfl.trust_aware_config import TrustAwareExperimentConfig
from dpfl.registry import DATASETS, MODELS, NOISE_MECHANISMS, ATTACKS, AGGREGATORS

# Force registry population — standard imports
import dpfl.data.mnist_dataset  # noqa: F401
import dpfl.models.mlp_model  # noqa: F401
import dpfl.privacy.gaussian_mechanism  # noqa: F401
import dpfl.attacks.scale_attack  # noqa: F401

# Kebab-case module: triggers @register decorator for trust_aware_d2b aggregator
importlib.import_module("dpfl.aggregation.trust-aware-d2b-aggregator")

from dpfl.tracking.metrics_tracker import MetricsTracker
from dpfl.privacy.adaptive_clipper import AdaptiveClipper
from dpfl.privacy.bounded_gaussian_mechanism import BoundedGaussianMechanism
from dpfl.privacy.per_neighbor_rdp_accountant import PerNeighborRDPAccountant

# Kebab-case module import for simulator
_sim_mod = importlib.import_module("dpfl.training.trust-aware-dfl-simulator")
TrustAwareDFLSimulator = _sim_mod.TrustAwareDFLSimulator


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parent / "trust-aware-config.yaml")
    config = TrustAwareExperimentConfig.from_yaml(config_path)

    device = torch.device(
        "cuda" if config.device == "auto" and torch.cuda.is_available()
        else "cpu" if config.device == "auto" else config.device)
    print(f"Using device: {device}")

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Resolve components from registries
    dataset_cls = DATASETS[config.dataset.name]
    model_cls = MODELS[config.model.name]
    noise_mechanism = NOISE_MECHANISMS["gaussian"]()
    attack = ATTACKS[config.attack.type](scale_factor=config.attack.scale_factor)

    # Compute param_dim for aggregator
    ds_meta = dataset_cls()
    input_dim = prod(ds_meta.input_shape)
    temp_model = model_cls(input_dim=input_dim,
                           hidden_size=config.model.hidden_size,
                           num_classes=ds_meta.num_classes)
    param_dim = temp_model.count_params()
    del temp_model

    # Aggregator with trust params
    aggregator = AGGREGATORS["trust_aware_d2b"](
        param_dim=param_dim,
        ema_lambda=config.trust.ema_lambda,
        gamma_z=config.trust.gamma_z,
        sigma_floor_z=config.trust.sigma_floor_z,
        alpha_drop=config.trust.alpha_drop,
        sigma_floor_drop=config.trust.sigma_floor_drop,
        gamma_penalty=config.trust.gamma_penalty)

    adaptive_clipper = AdaptiveClipper(config.trust.clip_window)
    bounded_noise = BoundedGaussianMechanism(eta=config.trust.eta)

    alpha_list = config.dp.accountant_params.get(
        "alpha_list", [1.25, 1.5, 2, 3, 5, 10, 20, 50, 100])
    per_neighbor_accountant = PerNeighborRDPAccountant(
        alpha_list=alpha_list, delta=config.dp.delta,
        epsilon_max=config.dp.epsilon_max)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / f"trust_d2b_{timestamp}"
    tracker = MetricsTracker(str(run_dir))
    shutil.copy2(config_path, run_dir / "config.yaml")

    # Create and run simulator
    simulator = TrustAwareDFLSimulator(
        config, config.trust, dataset_cls, model_cls,
        noise_mechanism, aggregator, attack,
        adaptive_clipper, bounded_noise, per_neighbor_accountant,
        tracker=tracker, device=device)
    simulator.setup()
    simulator.run()

    # Export results
    tracker.to_csv()
    tracker.to_json()
    tracker.save_node_data()
    tracker.plot_accuracy()
    tracker.plot_accuracy_spread()
    tracker.plot_detection()
    report = tracker.save_report(timestamp)
    print(report)


if __name__ == "__main__":
    main()
