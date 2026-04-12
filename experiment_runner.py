"""Shared entry point template for all DFL algorithm variants."""

import logging
import sys
import random
import shutil
import time
from datetime import datetime
from math import prod
from pathlib import Path

import torch
import numpy as np

from dpfl.registry import DATASETS, MODELS
from dpfl.tracking.metrics_tracker import MetricsTracker
from dpfl.tracking.logger_setup import setup_experiment_logger


def run_experiment(config_cls, build_fn, prefix, default_config_name, algo_name=None):
    """Generic experiment runner.

    Args:
        config_cls: Config dataclass (ExperimentConfig, TrustAwareExperimentConfig, etc.)
        build_fn: Callable(config, dataset_cls, model_cls, param_dim, device) -> simulator
        prefix: Output directory prefix (e.g. "dfl", "trust_d2b", "noise_game")
        default_config_name: Default YAML config filename in dpfl/ directory
        algo_name: Algorithm key name (e.g. "trust-aware", "dpsgd-kurtosis")
    """
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parent / default_config_name)
    config = config_cls.from_yaml(config_path)

    device = torch.device(
        "cuda" if config.device == "auto" and torch.cuda.is_available()
        else "cpu" if config.device == "auto" else config.device)

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Resolve base components
    dataset_cls = DATASETS[config.dataset.name]
    model_cls = MODELS[config.model.name]
    ds_meta = dataset_cls()
    input_dim = prod(ds_meta.input_shape)
    temp_model = model_cls(input_dim=input_dim,
                           hidden_size=config.model.hidden_size,
                           num_classes=ds_meta.num_classes)
    param_dim = temp_model.count_params()
    del temp_model

    # Output directory + tracker + logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / f"{prefix}_{timestamp}"
    metadata = {
        "algorithm": algo_name or prefix,
        "attack_type": config.attack.type,
        "dataset": config.dataset.name,
        "seed": config.seed,
        "n_nodes": config.topology.n_nodes,
        "n_attackers": config.topology.n_attackers,
        "noise_mult": config.dp.noise_mult,
        "clip_bound": config.dp.clip_bound,
        "split_mode": config.dataset.split.mode,
        "dirichlet_alpha": config.dataset.split.alpha,
    }
    tracker = MetricsTracker(str(run_dir), metadata=metadata)
    shutil.copy2(config_path, run_dir / "config.yaml")

    logger = setup_experiment_logger(str(run_dir))
    logger.info("Experiment start: algorithm=%s, dataset=%s, model=%s, device=%s",
                prefix, config.dataset.name, config.model.name, device)
    logger.info("Config: nodes=%d, attackers=%d, rounds=%d, noise_mode=%s, attack=%s",
                config.topology.n_nodes, config.topology.n_attackers,
                config.training.n_rounds, config.dp.noise_mode, config.attack.type)

    # Build and run variant-specific simulator
    t_start = time.time()
    simulator = build_fn(config, dataset_cls, model_cls, param_dim, tracker, device)
    simulator.setup()
    simulator.run()
    elapsed = time.time() - t_start

    # Export results
    tracker.to_csv()
    tracker.to_json()
    tracker.save_node_data()
    tracker.plot_accuracy()
    tracker.plot_accuracy_spread()
    tracker.plot_privacy_budget()
    tracker.plot_detection()
    report = tracker.save_report(timestamp)
    logger.info("Experiment complete in %.1fs\n%s", elapsed, report)
