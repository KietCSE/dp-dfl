"""Noise-Game config: NoiseGameConfig + NoiseGameExperimentConfig."""

from dataclasses import dataclass, field

import yaml

from dpfl.config import (
    _from_dict, DatasetConfig, ModelConfig, TopologyConfig,
    TrainingConfig, DPConfig, AttackConfig, AggregationConfig,
)


@dataclass
class NoiseGameConfig:
    """Hyperparameters for Strategic Noise Game algorithm."""
    # Section 3: Noise components
    alpha_attack: float = 0.5       # Base directional noise weight
    sigma_0: float = 1.0            # Initial orthogonal/spectral noise scale
    svd_rank: int = 16              # Truncated SVD rank for spectrum noise
    svd_reshape_k: int = 64         # Reshape gradient (D,) -> (k, D/k) for SVD

    # Section 4: Accuracy enhancement
    momentum_beta: float = 0.9      # Momentum decay factor
    ema_gamma: float = 0.9          # Gradient denoising EMA factor
    anneal_kappa: float = 0.02      # Noise annealing decay: sigma_t = sigma_0 * exp(-kappa*t)
    align_tau: float = 0.1          # Gradient alignment cosine threshold
    scaffold: bool = True           # Enable SCAFFOLD variance reduction
    two_track: bool = True          # Enable two-track model (clean + robust)
    two_track_lambda: float = 0.7   # Clean model weight in combination


@dataclass
class NoiseGameExperimentConfig:
    """Extends ExperimentConfig with noise_game section."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    noise_game: NoiseGameConfig = field(default_factory=NoiseGameConfig)
    output_dir: str = "results"
    device: str = "auto"
    seed: int = 42

    @staticmethod
    def from_yaml(path: str) -> "NoiseGameExperimentConfig":
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return _from_dict(NoiseGameExperimentConfig, data or {})
