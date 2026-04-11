"""Shared entry point template for all DFL algorithm variants."""

import sys
import random
import shutil
from datetime import datetime
from math import prod
from pathlib import Path

import torch
import numpy as np

from dpfl.registry import DATASETS, MODELS
from dpfl.tracking.metrics_tracker import MetricsTracker


def run_experiment(config_cls, build_fn, prefix, default_config_name):
    """Generic experiment runner.

    Args:
        config_cls: Config dataclass (ExperimentConfig, TrustAwareExperimentConfig, etc.)
        build_fn: Callable(config, dataset_cls, model_cls, param_dim, device) -> simulator
        prefix: Output directory prefix (e.g. "dfl", "trust_d2b", "noise_game")
        default_config_name: Default YAML config filename in dpfl/ directory
    """
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parent / default_config_name)
    config = config_cls.from_yaml(config_path)

    device = torch.device(
        "cuda" if config.device == "auto" and torch.cuda.is_available()
        else "cpu" if config.device == "auto" else config.device)
    print(f"Using device: {device}")

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

    # Output directory + tracker
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / f"{prefix}_{timestamp}"
    tracker = MetricsTracker(str(run_dir))
    shutil.copy2(config_path, run_dir / "config.yaml")

    # Build and run variant-specific simulator
    simulator = build_fn(config, dataset_cls, model_cls, param_dim, tracker, device)
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
