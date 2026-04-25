"""Unified runner for all DFL algorithms.

Usage:
    python run.py -a dpsgd-kurtosis    [config.yaml]
    python run.py -a trust-aware       [config.yaml]
    python run.py -a noise-game        [config.yaml]
"""

import sys
import argparse
from pathlib import Path

# Add parent dir to sys.path so 'import dpfl' works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force registry population — datasets, models, attacks, aggregators
import dpfl.data.mnist_dataset  # noqa: F401
import dpfl.data.cifar10_dataset  # noqa: F401
import dpfl.models.mlp_model  # noqa: F401
import dpfl.models.cnn_model  # noqa: F401
import dpfl.core.gaussian_mechanism  # noqa: F401
import dpfl.core.scale_attack  # noqa: F401
import dpfl.core.sign_flip_attack  # noqa: F401
import dpfl.core.gaussian_random_attack  # noqa: F401
import dpfl.core.alie_attack  # noqa: F401
import dpfl.core.label_flip_attack  # noqa: F401
import dpfl.core.renyi_accountant  # noqa: F401
import dpfl.algorithms.dpsgd_kurtosis.kurtosis_aggregator  # noqa: F401
import dpfl.algorithms.momentum_kurtosis.momentum_kurtosis_aggregator  # noqa: F401
import dpfl.algorithms.noise_game.simple_avg_aggregator  # noqa: F401
import dpfl.algorithms.fedavg.fedavg_aggregator  # noqa: F401
import dpfl.algorithms.trust_aware.aggregator  # noqa: F401
import dpfl.algorithms.krum.krum_aggregator  # noqa: F401
import dpfl.algorithms.trimmed_mean.trimmed_mean_aggregator  # noqa: F401
import dpfl.algorithms.fltrust.fltrust_aggregator  # noqa: F401
import dpfl.algorithms.flame.flame_aggregator  # noqa: F401

from dpfl.registry import DATASETS, NOISE_MECHANISMS, AGGREGATORS, ATTACKS, ACCOUNTANTS
from dpfl.experiment_runner import run_experiment


# -- Shared helpers --

def _build_attack(config):
    """Build attack instance from config, handling type-specific params."""
    attack_cls = ATTACKS.get(config.attack.type)
    if attack_cls is None:
        raise ValueError(f"Unknown attack type: {config.attack.type}")
    atype = config.attack.type
    if atype == "scale":
        return attack_cls(scale_factor=config.attack.scale_factor)
    elif atype == "alie":
        return attack_cls(z_max=config.attack.z_max)
    elif atype == "label_flip":
        ds_cls = DATASETS[config.dataset.name]
        num_classes = ds_cls().num_classes
        return attack_cls(num_classes=num_classes, flip_mode=config.attack.flip_mode)
    else:
        # sign_flip, gaussian_random — no extra params
        return attack_cls()


def _build_accountant(config):
    """Build privacy accountant from config."""
    return ACCOUNTANTS[config.dp.accountant](
        delta=config.dp.delta, **config.dp.accountant_params)


# -- Build functions per algorithm --

def build_dpsgd_kurtosis(config, dataset_cls, model_cls, param_dim, tracker, device):
    from dpfl.algorithms.dpsgd_kurtosis.simulator import DFLSimulator
    noise_mechanism = NOISE_MECHANISMS["gaussian"]()
    attack = _build_attack(config)
    aggregator = AGGREGATORS[config.aggregation.type](
        param_dim=param_dim, **config.aggregation.params)
    accountant = None
    if config.dp.noise_mode != "none":
        accountant = _build_accountant(config)
    return DFLSimulator(
        config, dataset_cls, model_cls,
        noise_mechanism, aggregator, attack,
        accountant=accountant, tracker=tracker, device=device)


def build_trust_aware(config, dataset_cls, model_cls, param_dim, tracker, device):
    from dpfl.algorithms.trust_aware.simulator import TrustAwareDFLSimulator
    from dpfl.algorithms.trust_aware.adaptive_clipper import LayerwiseAdaptiveClipper
    from dpfl.algorithms.trust_aware.bounded_gaussian import LayerwiseBoundedGaussian
    tc = config.trust
    noise_mechanism = NOISE_MECHANISMS["gaussian"]()
    attack = _build_attack(config)
    aggregator = AGGREGATORS["trust_aware_d2b"](
        theta=tc.theta, gamma=tc.gamma, kappa=tc.kappa,
        alpha_T=tc.alpha_T, T_min=tc.T_min, beta_soft=tc.beta_soft,
        beta_m=tc.beta_m, eta_global=tc.eta_global)
    # Heuristic ε reporting only — D2B-DP itself uses ρ-based noise schedule.
    accountant = _build_accountant(config)
    return TrustAwareDFLSimulator(
        config, tc, dataset_cls, model_cls,
        noise_mechanism, aggregator, attack,
        LayerwiseAdaptiveClipper(k=tc.k),
        LayerwiseBoundedGaussian(bound_eta=tc.bound_eta),
        accountant=accountant, tracker=tracker, device=device)


def build_noise_game(config, dataset_cls, model_cls, param_dim, tracker, device):
    from dpfl.algorithms.noise_game.simulator import NoiseGameDFLSimulator
    from dpfl.algorithms.noise_game.mechanism import NoiseGameMechanism
    from dpfl.core.gaussian_mechanism import GaussianMechanism
    noise_mechanism = GaussianMechanism()
    attack = _build_attack(config)
    aggregator = AGGREGATORS[config.aggregation.type](
        param_dim=param_dim, **config.aggregation.params)
    ng = config.noise_game
    game_mechanism = NoiseGameMechanism(
        alpha_attack=ng.alpha_attack, sigma_0=ng.sigma_0,
        anneal_kappa=ng.anneal_kappa, svd_rank=ng.svd_rank,
        svd_reshape_k=ng.svd_reshape_k,
        clip_bound=config.dp.clip_bound, delta=config.dp.delta,
        epsilon_max=config.dp.epsilon_max,
        beta_strat=ng.beta_strat, sigma_total=ng.sigma_total,
        param_dim=param_dim,
        alpha_rd=ng.rdp_alpha)
    alpha_list = config.dp.accountant_params.get(
        "alpha_list", [1.25, 1.5, 2, 3, 5, 10, 20, 50, 100])
    accountant = ACCOUNTANTS[config.dp.accountant](
        delta=config.dp.delta, alpha_list=alpha_list)
    return NoiseGameDFLSimulator(
        config, config.noise_game, dataset_cls, model_cls,
        noise_mechanism, game_mechanism, aggregator, attack,
        accountant=accountant, tracker=tracker, device=device)


def build_fedavg(config, dataset_cls, model_cls, param_dim, tracker, device):
    from dpfl.algorithms.fedavg.simulator import FedAvgSimulator
    noise_mechanism = NOISE_MECHANISMS["gaussian"]()
    attack = _build_attack(config)
    aggregator = AGGREGATORS["fedavg"]()
    accountant = None
    if config.dp.noise_mode != "none":
        accountant = _build_accountant(config)
    return FedAvgSimulator(
        config, dataset_cls, model_cls,
        noise_mechanism, aggregator, attack,
        accountant=accountant, tracker=tracker, device=device)


# DP-FedAvg reuses FedAvgSimulator (weighted avg) with DP-SGD noise
build_dp_fedavg = build_fedavg

# Krum, Trimmed Mean, FLAME reuse DFLSimulator
build_krum = build_dpsgd_kurtosis
build_trimmed_mean = build_dpsgd_kurtosis
build_flame = build_dpsgd_kurtosis


def build_adaptive_noise(config, dataset_cls, model_cls, param_dim, tracker, device):
    import inspect
    from dpfl.algorithms.adaptive_noise.simulator import AdaptiveNoiseSimulator
    from dpfl.algorithms.adaptive_noise.per_node_rdp_accountant import PerNodeRDPAccountant
    noise_mechanism = NOISE_MECHANISMS["gaussian"]()
    attack = _build_attack(config)

    # Build aggregator from config (simple_avg, kurtosis_avg,
    # momentum_kurtosis_avg, ...). Pass param_dim only if the constructor
    # accepts it so dimension-free aggregators (simple_avg) still work.
    agg_type = config.aggregation.type
    agg_params = dict(config.aggregation.params or {})
    agg_cls = AGGREGATORS[agg_type]
    sig = inspect.signature(agg_cls.__init__)
    if "param_dim" in sig.parameters:
        aggregator = agg_cls(param_dim=param_dim, **agg_params)
    else:
        aggregator = agg_cls(**agg_params)

    alpha_list = config.dp.accountant_params.get(
        "alpha_list", [1.25, 1.5, 2, 3, 5, 10, 20, 50, 100])
    rdp_accountant = PerNodeRDPAccountant(
        alpha_list=alpha_list, delta=config.dp.delta,
        epsilon_max=config.dp.epsilon_max)
    return AdaptiveNoiseSimulator(
        config, config.adaptive_noise, dataset_cls, model_cls,
        noise_mechanism, aggregator, attack,
        rdp_accountant=rdp_accountant, tracker=tracker, device=device)


def build_fltrust(config, dataset_cls, model_cls, param_dim, tracker, device):
    from dpfl.algorithms.fltrust.simulator import FLTrustSimulator
    noise_mechanism = NOISE_MECHANISMS["gaussian"]() if config.dp.noise_mode != "none" else None
    attack = _build_attack(config)
    aggregator = AGGREGATORS["fltrust"](**config.aggregation.params)
    accountant = _build_accountant(config) if config.dp.noise_mode != "none" else None
    return FLTrustSimulator(
        config, dataset_cls, model_cls,
        noise_mechanism, aggregator, attack,
        accountant=accountant, tracker=tracker, device=device,
        root_data_ratio=config.fltrust.root_data_ratio,
    )


# -- Algorithm registry --

ALGORITHMS = {
    "dpsgd-kurtosis": {
        "config_cls": "dpfl.config.ExperimentConfig",
        "build_fn": build_dpsgd_kurtosis,
        "prefix": "dpsgd_kurtosis",
        "default_config": "config/dpsgd_kurtosis.yaml",
    },
    "fedavg": {
        "config_cls": "dpfl.config.ExperimentConfig",
        "build_fn": build_fedavg,
        "prefix": "fedavg",
        "default_config": "config/fedavg.yaml",
    },
    "krum": {
        "config_cls": "dpfl.config.ExperimentConfig",
        "build_fn": build_krum,
        "prefix": "krum",
        "default_config": "config/krum.yaml",
    },
    "trust-aware": {
        "config_cls": "dpfl.config.TrustAwareExperimentConfig",
        "build_fn": build_trust_aware,
        "prefix": "trust_d2b",
        "default_config": "config/trust_aware.yaml",
    },
    "noise-game": {
        "config_cls": "dpfl.config.NoiseGameExperimentConfig",
        "build_fn": build_noise_game,
        "prefix": "noise_game",
        "default_config": "config/noise_game.yaml",
    },
    "dp-fedavg": {
        "config_cls": "dpfl.config.ExperimentConfig",
        "build_fn": build_dp_fedavg,
        "prefix": "dp_fedavg",
        "default_config": "config/dp_fedavg.yaml",
    },
    "trimmed-mean": {
        "config_cls": "dpfl.config.ExperimentConfig",
        "build_fn": build_trimmed_mean,
        "prefix": "trimmed_mean",
        "default_config": "config/trimmed_mean.yaml",
    },
    "fltrust": {
        "config_cls": "dpfl.config.FLTrustExperimentConfig",
        "build_fn": build_fltrust,
        "prefix": "fltrust",
        "default_config": "config/fltrust.yaml",
    },
    "flame": {
        "config_cls": "dpfl.config.ExperimentConfig",
        "build_fn": build_flame,
        "prefix": "flame",
        "default_config": "config/flame.yaml",
    },
    "adaptive-noise": {
        "config_cls": "dpfl.config.AdaptiveNoiseExperimentConfig",
        "build_fn": build_adaptive_noise,
        "prefix": "adaptive_noise",
        "default_config": "config/adaptive_noise.yaml",
    },
}


def main():
    parser = argparse.ArgumentParser(description="DPFL — Run DFL experiment")
    parser.add_argument("--algorithm", "-a", choices=ALGORITHMS.keys(),
                        default="dpsgd-kurtosis",
                        help="Algorithm variant (default: dpsgd-kurtosis)")
    parser.add_argument("config", nargs="?", help="Path to config YAML")
    args = parser.parse_args()

    algo = ALGORITHMS[args.algorithm]

    # Lazy import config class
    import importlib
    mod_path, cls_name = algo["config_cls"].rsplit(".", 1)
    config_cls = getattr(importlib.import_module(mod_path), cls_name)

    # Override sys.argv for experiment_runner
    config_path = args.config or str(Path(__file__).parent / algo["default_config"])
    sys.argv = [sys.argv[0], config_path]

    run_experiment(config_cls, algo["build_fn"], algo["prefix"], algo["default_config"],
                   algo_name=args.algorithm)


if __name__ == "__main__":
    main()
