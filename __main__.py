"""Entry point: python -m dpfl config.yaml"""

import sys
import random
import shutil
from datetime import datetime
from math import prod
from pathlib import Path

import torch
import numpy as np

from dpfl.config import ExperimentConfig
from dpfl.registry import DATASETS, MODELS, NOISE_MECHANISMS, AGGREGATORS, ATTACKS

# Force registry population by importing concrete implementations
import dpfl.data.mnist_dataset       # noqa: F401
import dpfl.models.mlp_model         # noqa: F401
import dpfl.privacy.gaussian_mechanism  # noqa: F401
import dpfl.attacks.scale_attack     # noqa: F401
import dpfl.aggregation.kurtosis_avg_aggregator  # noqa: F401

from dpfl.tracking.metrics_tracker import MetricsTracker
from dpfl.training.dfl_simulator import DFLSimulator


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "config.yaml")
    config = ExperimentConfig.from_yaml(config_path)

    # Resolve device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    print(f"Using device: {device}")

    # Seed everything
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Resolve components from registry
    dataset_cls = DATASETS[config.dataset.name]
    model_cls = MODELS[config.model.name]
    noise_mechanism = NOISE_MECHANISMS["gaussian"]()
    attack = ATTACKS[config.attack.type](scale_factor=config.attack.scale_factor)

    # Compute param_dim for aggregator (lightweight — no data download)
    ds_meta = dataset_cls()
    input_dim = prod(ds_meta.input_shape)
    temp_model = model_cls(input_dim=input_dim, hidden_size=config.model.hidden_size,
                           num_classes=ds_meta.num_classes)
    param_dim = temp_model.count_params()
    del temp_model

    aggregator = AGGREGATORS[config.aggregation.type](
        param_dim=param_dim,
        centered=config.aggregation.centered,
        confidence=config.aggregation.confidence,
    )

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / timestamp

    tracker = MetricsTracker(str(run_dir))

    # Copy config into run directory
    shutil.copy2(config_path, run_dir / "config.yaml")

    # Create and run simulator
    simulator = DFLSimulator(
        config, dataset_cls, model_cls,
        noise_mechanism, aggregator, attack, tracker,
        device=device,
    )
    simulator.setup()
    simulator.run()

    # Export results
    tracker.to_csv()
    tracker.to_json()
    tracker.save_node_data()
    tracker.plot_accuracy()
    tracker.plot_accuracy_spread()
    tracker.plot_privacy_budget()
    tracker.plot_kurtosis()
    tracker.plot_detection()
    report = tracker.save_report(timestamp)
    print(report)


if __name__ == "__main__":
    main()
