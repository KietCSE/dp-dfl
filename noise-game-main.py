"""Entry point for Strategic Noise Game DFL: python -m dpfl.noise-game-main"""

import sys
import random
import shutil
import importlib
from datetime import datetime
from math import prod
from pathlib import Path

import torch
import numpy as np

from dpfl.noise_game_config import NoiseGameExperimentConfig
from dpfl.registry import DATASETS, MODELS, ATTACKS, AGGREGATORS, ACCOUNTANTS

# Force registry population
import dpfl.data.mnist_dataset  # noqa: F401
import dpfl.models.mlp_model  # noqa: F401
import dpfl.privacy.gaussian_mechanism  # noqa: F401
import dpfl.attacks.scale_attack  # noqa: F401
import dpfl.aggregation.kurtosis_avg_aggregator  # noqa: F401
importlib.import_module("dpfl.aggregation.simple-avg-aggregator")  # noqa: F401
import dpfl.privacy.renyi_dpsgd  # noqa: F401

from dpfl.tracking.metrics_tracker import MetricsTracker
from dpfl.privacy.gaussian_mechanism import GaussianMechanism

# Kebab-case module imports
_mech_mod = importlib.import_module("dpfl.privacy.noise-game-mechanism")
NoiseGameMechanism = _mech_mod.NoiseGameMechanism

_sim_mod = importlib.import_module("dpfl.training.noise-game-dfl-simulator")
NoiseGameDFLSimulator = _sim_mod.NoiseGameDFLSimulator


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parent / "noise-game-config.yaml")
    config = NoiseGameExperimentConfig.from_yaml(config_path)

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
    noise_mechanism = GaussianMechanism()  # For L2-norm clipping only
    attack = ATTACKS[config.attack.type](scale_factor=config.attack.scale_factor)

    # Compute param_dim for aggregator
    ds_meta = dataset_cls()
    input_dim = prod(ds_meta.input_shape)
    temp_model = model_cls(input_dim=input_dim,
                           hidden_size=config.model.hidden_size,
                           num_classes=ds_meta.num_classes)
    param_dim = temp_model.count_params()
    del temp_model

    # Aggregator
    aggregator = AGGREGATORS[config.aggregation.type](
        param_dim=param_dim, **config.aggregation.params)

    # Noise-game mechanism
    ng = config.noise_game
    game_mechanism = NoiseGameMechanism(
        alpha_attack=ng.alpha_attack, sigma_0=ng.sigma_0,
        anneal_kappa=ng.anneal_kappa, svd_rank=ng.svd_rank,
        svd_reshape_k=ng.svd_reshape_k)

    # Privacy accountant
    alpha_list = config.dp.accountant_params.get(
        "alpha_list", [1.25, 1.5, 2, 3, 5, 10, 20, 50, 100])
    accountant = ACCOUNTANTS[config.dp.accountant](
        delta=config.dp.delta, alpha_list=alpha_list)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / f"noise_game_{timestamp}"
    tracker = MetricsTracker(str(run_dir))
    shutil.copy2(config_path, run_dir / "config.yaml")

    # Create and run simulator
    simulator = NoiseGameDFLSimulator(
        config, config.noise_game, dataset_cls, model_cls,
        noise_mechanism, game_mechanism, aggregator, attack,
        accountant=accountant, tracker=tracker, device=device)
    simulator.setup()
    simulator.run()

    # Export results
    tracker.to_csv()
    tracker.to_json()
    tracker.save_node_data()
    tracker.plot_accuracy()
    tracker.plot_accuracy_spread()
    tracker.plot_privacy_budget()
    tracker.plot_detection()
    report = tracker.save_report(timestamp)
    print(report)


if __name__ == "__main__":
    main()
