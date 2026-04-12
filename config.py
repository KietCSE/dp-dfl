"""All experiment config dataclasses: base, DFL, Trust-Aware, Noise-Game."""

from dataclasses import dataclass, field, fields
from typing import get_type_hints
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


# -- Shared sub-configs --

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
    n_workers: int = 1

@dataclass
class ModelConfig:
    name: str = "mlp"
    hidden_size: int = 100

@dataclass
class DPConfig:
    noise_mode: str = "per_step"
    clip_bound: float = 2.0
    noise_mult: float = 1.1
    delta: float = 1e-5
    epsilon_max: float = 10.0
    accountant: str = "renyi_dpsgd"
    accountant_params: dict = field(default_factory=lambda: {
        "alpha_list": [1.25, 1.5, 2, 3, 5, 10, 20, 50, 100],
    })

@dataclass
class AttackConfig:
    type: str = "scale"
    scale_factor: float = 3.0
    start_round: int = 0  # round to start attacking (0 = always attack)
    z_max: float = 1.0           # ALIE z_max parameter
    flip_mode: str = "rotate"    # LabelFlip mode: rotate|random|negate

@dataclass
class AggregationConfig:
    type: str = "kurtosis_avg"
    params: dict = field(default_factory=lambda: {
        "centered": False, "confidence": 1.96,
    })

@dataclass
class DataSplitConfig:
    mode: str = "iid"
    alpha: float = 0.5

@dataclass
class DatasetConfig:
    name: str = "mnist"
    split: DataSplitConfig = field(default_factory=DataSplitConfig)


# -- Base experiment config --

@dataclass
class BaseExperimentConfig:
    """Shared config fields for all algorithm variants."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    output_dir: str = "dpfl/results"
    device: str = "auto"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return _from_dict(cls, data or {})


# -- Algorithm-specific configs --

@dataclass
class ExperimentConfig(BaseExperimentConfig):
    """Standard DP-SGD DFL config."""
    pass


@dataclass
class TrustConfig:
    """Hyperparameters for Trust-Aware D2B-DP algorithm."""
    trust_init: float = 0.5
    ema_lambda: float = 0.8
    rho_min: float = 0.1
    rho_max: float = 5.0
    beta: float = 0.01
    gamma_penalty: float = 0.5
    gamma_z: float = 3.0
    sigma_floor_z: float = 1e-4
    alpha_drop: float = 2.0
    sigma_floor_drop: float = 1e-3
    clip_window: int = 3
    temporal_window: int = 5
    eta: float = 0.1

@dataclass
class TrustAwareExperimentConfig(BaseExperimentConfig):
    """Extends base with trust section for D2B-DP."""
    trust: TrustConfig = field(default_factory=TrustConfig)


@dataclass
class FLTrustConfig:
    """Hyperparameters for FLTrust algorithm."""
    root_data_ratio: float = 0.1
    trust_threshold: float = 0.1

@dataclass
class FLTrustExperimentConfig(BaseExperimentConfig):
    """Extends base with FLTrust section."""
    fltrust: FLTrustConfig = field(default_factory=FLTrustConfig)


@dataclass
class NoiseGameConfig:
    """Hyperparameters for Strategic Noise Game algorithm."""
    alpha_attack: float = 0.5
    sigma_0: float = 1.0
    svd_rank: int = 16
    svd_reshape_k: int = 64
    momentum_beta: float = 0.9
    ema_gamma: float = 0.9
    anneal_kappa: float = 0.02
    align_tau: float = 0.1
    scaffold: bool = True
    two_track: bool = True
    two_track_lambda: float = 0.7
    beta_strat: float = 0.5      # sigma_strat = beta * sigma_DP coupling
    sigma_total: float = 3.0     # total noise energy cap
    rdp_alpha: float = 2.0       # Renyi order for internal budget tracking
    nsr_warn: float = 5.0        # NSR warning threshold

@dataclass
class NoiseGameExperimentConfig(BaseExperimentConfig):
    """Extends base with noise_game section."""
    noise_game: NoiseGameConfig = field(default_factory=NoiseGameConfig)
