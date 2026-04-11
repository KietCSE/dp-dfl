"""Nested dataclass config with YAML loader."""

from dataclasses import dataclass, field, fields
from typing import List, get_type_hints
import yaml


def _from_dict(cls, data: dict):
    """Recursively construct a dataclass from a nested dict."""
    if not isinstance(data, dict):
        return data
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        ftype = hints.get(f.name, f.type)
        if isinstance(val, dict) and hasattr(ftype, '__dataclass_fields__'):
            kwargs[f.name] = _from_dict(ftype, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


@dataclass
class TopologyConfig:
    n_nodes: int = 20
    n_attackers: int = 4
    n_neighbors: int = 10
    seed: int = 42


@dataclass
class TrainingConfig:
    n_rounds: int = 100
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.1
    n_workers: int = 1  # thread pool size for parallel node training


@dataclass
class ModelConfig:
    name: str = "mlp"
    hidden_size: int = 100


@dataclass
class DPConfig:
    clip_bound: float = 2.0
    noise_mult: float = 1.1
    delta: float = 1e-5
    alpha_list: List[float] = field(
        default_factory=lambda: [1.25, 1.5, 2, 3, 5, 10, 20, 50, 100]
    )
    epsilon_max: float = 10.0


@dataclass
class AttackConfig:
    type: str = "scale"
    scale_factor: float = 3.0


@dataclass
class AggregationConfig:
    type: str = "kurtosis_avg"
    params: dict = field(default_factory=lambda: {
        "centered": False,
        "confidence": 1.96,
    })


@dataclass
class DataSplitConfig:
    mode: str = "iid"
    alpha: float = 0.5


@dataclass
class DatasetConfig:
    name: str = "mnist"
    split: DataSplitConfig = field(default_factory=DataSplitConfig)


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    output_dir: str = "dpfl/results"
    device: str = "auto"  # "cpu", "cuda", or "auto" (auto-detect GPU)
    seed: int = 42

    @staticmethod
    def from_yaml(path: str) -> "ExperimentConfig":
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return _from_dict(ExperimentConfig, data or {})
