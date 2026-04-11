"""Trust-Aware D2B-DP config: TrustConfig + TrustAwareExperimentConfig."""

from dataclasses import dataclass, field

import yaml

from dpfl.config import (
    _from_dict, DatasetConfig, ModelConfig, TopologyConfig,
    TrainingConfig, DPConfig, AttackConfig, AggregationConfig,
)


@dataclass
class TrustConfig:
    """Hyperparameters for Trust-Aware D2B-DP algorithm."""
    trust_init: float = 0.5        # T[i,j] initial value
    ema_lambda: float = 0.8        # EMA memory factor
    rho_min: float = 0.1           # Min privacy budget
    rho_max: float = 5.0           # Max privacy budget
    beta: float = 0.01             # Budget time ramp coefficient
    gamma_penalty: float = 0.5     # Trust penalty multiplier
    gamma_z: float = 3.0           # Z-Score threshold multiplier
    sigma_floor_z: float = 1e-4    # Z-Score sensitivity floor
    alpha_drop: float = 2.0        # MAD multiplier for tau_drop
    sigma_floor_drop: float = 1e-3 # Trust sensitivity floor
    clip_window: int = 3           # Adaptive clip FIFO queue size
    temporal_window: int = 5       # Temporal buffer size (W)
    eta: float = 0.1               # Bounded noise eta parameter


@dataclass
class TrustAwareExperimentConfig:
    """Extends ExperimentConfig with trust section for D2B-DP."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    trust: TrustConfig = field(default_factory=TrustConfig)
    output_dir: str = "results"
    device: str = "auto"
    seed: int = 42

    @staticmethod
    def from_yaml(path: str) -> "TrustAwareExperimentConfig":
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return _from_dict(TrustAwareExperimentConfig, data or {})
