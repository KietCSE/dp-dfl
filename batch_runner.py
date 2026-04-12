"""Batch experiment runner for DPFL paper experiments.

Usage:
    python batch_runner.py --experiment EXP1          # Run EXP-1 (main table)
    python batch_runner.py --experiment EXP3 --dry-run # Preview configs only
    python batch_runner.py --experiment ALL            # Run all experiments
    python batch_runner.py --list                      # List experiments
"""

import argparse
import itertools
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml

from dpfl.tracking.logger_setup import setup_batch_logger

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────

ALGORITHMS = [
    "fedavg", "dp-fedavg", "krum", "trimmed-mean",
    "fltrust", "flame", "dpsgd-kurtosis", "trust-aware", "noise-game",
]

ATTACKS = ["none", "scale", "sign_flip", "alie", "gaussian_random", "label_flip"]

DATASETS = {
    "mnist": {"model": "mlp", "hidden_size": 100, "rounds": 100},
    "cifar10": {"model": "cnn", "hidden_size": 128, "rounds": 200},
}

SEEDS = [42, 123, 456]

# ─── Experiment Definitions ──────────────────────────────────────

EXPERIMENTS = {
    "EXP1": {
        "name": "Main Table — Cross-Algorithm Comparison",
        "algorithms": ALGORITHMS,
        "attacks": ATTACKS,
        "datasets": list(DATASETS.keys()),
        "seeds": SEEDS,
        "vary": {},
    },
    "EXP2": {
        "name": "Ablation Study",
        "algorithms": ["trust-aware", "noise-game"],
        "attacks": ["scale"],
        "datasets": list(DATASETS.keys()),
        "seeds": SEEDS,
        "vary": {},
    },
    "EXP3": {
        "name": "Accuracy vs Attacker Fraction",
        "algorithms": ALGORITHMS,
        "attacks": ["scale", "alie"],
        "datasets": list(DATASETS.keys()),
        "seeds": SEEDS,
        "vary": {"n_attackers": [0, 2, 4, 6, 8, 10]},
    },
    "EXP4": {
        "name": "Privacy-Utility Pareto",
        "algorithms": ["dp-fedavg", "flame", "dpsgd-kurtosis", "trust-aware", "noise-game"],
        "attacks": ["scale"],
        "datasets": list(DATASETS.keys()),
        "seeds": SEEDS,
        "vary": {"noise_mult": [0.5, 0.8, 1.1, 1.5, 2.0, 3.0]},
    },
    "EXP5": {
        "name": "Non-IID Impact",
        "algorithms": ALGORITHMS,
        "attacks": ["scale"],
        "datasets": list(DATASETS.keys()),
        "seeds": SEEDS,
        "vary": {"alpha": [0.1, 0.3, 0.5, 1.0, 5.0, 100.0]},
    },
    "EXP6": {
        "name": "Detection F1 Across Attacks",
        "algorithms": ["krum", "trimmed-mean", "fltrust", "flame", "dpsgd-kurtosis", "trust-aware"],
        "attacks": ATTACKS,
        "datasets": ["mnist"],
        "seeds": SEEDS,
        "vary": {},
    },
    "EXP7": {
        "name": "Convergence Curves",
        "algorithms": ALGORITHMS,
        "attacks": ["scale"],
        "datasets": list(DATASETS.keys()),
        "seeds": SEEDS,
        "vary": {},
    },
    "EXP8": {
        "name": "Epsilon Accumulation",
        "algorithms": ["dp-fedavg", "flame", "dpsgd-kurtosis", "trust-aware", "noise-game"],
        "attacks": ["scale"],
        "datasets": ["mnist"],
        "seeds": SEEDS,
        "vary": {},
    },
}


def generate_configs(experiment_id: str) -> List[Dict]:
    """Generate list of run configs for an experiment."""
    exp = EXPERIMENTS[experiment_id]
    configs = []

    vary = exp.get("vary", {})
    vary_keys = list(vary.keys())
    vary_values = [vary[k] for k in vary_keys] if vary_keys else [[None]]

    for algo in exp["algorithms"]:
        for attack in exp["attacks"]:
            for ds_name in exp["datasets"]:
                for seed in exp["seeds"]:
                    for combo in itertools.product(*vary_values):
                        config = {
                            "experiment": experiment_id,
                            "algorithm": algo,
                            "attack": attack,
                            "dataset": ds_name,
                            "seed": seed,
                        }
                        for i, key in enumerate(vary_keys):
                            config[key] = combo[i] if combo[0] is not None else None
                        configs.append(config)

    return configs


def build_yaml_config(run_config: Dict, base_config_path: str) -> Dict:
    """Build YAML config dict from run config + base config."""
    with open(base_config_path) as f:
        cfg = yaml.safe_load(f)

    ds_info = DATASETS[run_config["dataset"]]
    cfg["dataset"]["name"] = run_config["dataset"]
    cfg["model"]["name"] = ds_info["model"]
    cfg["model"]["hidden_size"] = ds_info["hidden_size"]
    cfg["training"]["n_rounds"] = ds_info["rounds"]
    cfg["seed"] = run_config["seed"]

    # Attack config
    attack = run_config["attack"]
    if attack == "none":
        cfg["attack"]["type"] = "scale"
        cfg["attack"]["scale_factor"] = 1.0  # No-op
        cfg["topology"]["n_attackers"] = 0
    else:
        cfg["attack"]["type"] = attack

    # Parameter sweeps
    if run_config.get("n_attackers") is not None:
        cfg["topology"]["n_attackers"] = run_config["n_attackers"]
    if run_config.get("noise_mult") is not None:
        cfg["dp"]["noise_mult"] = run_config["noise_mult"]
    if run_config.get("alpha") is not None:
        alpha_val = run_config["alpha"]
        cfg["dataset"]["split"]["mode"] = "iid" if alpha_val >= 100 else "dirichlet"
        cfg["dataset"]["split"]["alpha"] = alpha_val

    return cfg


def get_output_dir(run_config: Dict) -> str:
    """Generate deterministic output directory name."""
    parts = [
        run_config["experiment"],
        run_config["algorithm"],
        run_config["attack"],
        run_config["dataset"],
        f"seed{run_config['seed']}",
    ]
    for key in ["n_attackers", "noise_mult", "alpha"]:
        if run_config.get(key) is not None:
            parts.append(f"{key}={run_config[key]}")
    return os.path.join("results", "experiments", "_".join(parts))


def get_base_config_path(algo: str) -> str:
    """Resolve base config path for an algorithm."""
    name = algo.replace("-", "_")
    path = f"config/{name}.yaml"
    if os.path.exists(path):
        return path
    # Fallback: dpsgd_kurtosis as default base
    return "config/dpsgd_kurtosis.yaml"


def run_single(run_config: Dict, dry_run: bool = False) -> bool:
    """Run a single experiment config. Returns True if successful."""
    output_dir = get_output_dir(run_config)

    # Skip if already completed
    if os.path.exists(os.path.join(output_dir, "metrics.csv")):
        logger.info("  [SKIP] %s", output_dir)
        return True

    algo = run_config["algorithm"]
    base_config = get_base_config_path(algo)
    cfg = build_yaml_config(run_config, base_config)
    cfg["output_dir"] = output_dir

    if dry_run:
        logger.info("  [DRY] %s", output_dir)
        return True

    # Write temp config and execute
    os.makedirs(output_dir, exist_ok=True)
    temp_cfg_path = os.path.join(output_dir, "run_config.yaml")
    with open(temp_cfg_path, "w") as f:
        yaml.dump(cfg, f)

    cmd = [sys.executable, "run.py", "-a", algo, temp_cfg_path]
    logger.info("  [RUN] %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.warning("  [FAIL] %s: %s", output_dir, result.stderr[:200])
        return False

    return True


def aggregate_results(experiment_id: str) -> None:
    """Aggregate results across seeds into summary CSV."""
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas required for aggregation: pip install pandas")
        return

    configs = generate_configs(experiment_id)
    results = []

    for cfg in configs:
        output_dir = get_output_dir(cfg)
        metrics_path = os.path.join(output_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path) as f:
            data = json.load(f)
        if data:
            last_round = data[-1]
            results.append({**cfg, **last_round})

    if not results:
        logger.warning("No results found for %s", experiment_id)
        return

    df = pd.DataFrame(results)
    group_cols = ["algorithm", "attack", "dataset"]
    for key in ["n_attackers", "noise_mult", "alpha"]:
        if key in df.columns and df[key].notna().any():
            group_cols.append(key)

    agg_dict = {"accuracy": ["mean", "std"]}
    if "epsilon" in df.columns:
        agg_dict["epsilon"] = "mean"
    if "f1_score" in df.columns:
        agg_dict["f1_score"] = "mean"

    summary = df.groupby(group_cols).agg(agg_dict).reset_index()
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]

    out_path = f"results/experiments/{experiment_id}_summary.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary.to_csv(out_path, index=False)
    logger.info("Summary saved: %s (%d rows)", out_path, len(summary))


def main():
    parser = argparse.ArgumentParser(description="DPFL Batch Experiment Runner")
    parser.add_argument("--experiment", "-e", type=str,
                        help="Experiment ID (EXP1-EXP8 or ALL)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview runs without executing")
    parser.add_argument("--list", action="store_true",
                        help="List all experiments")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate results only")
    args = parser.parse_args()

    if args.list:
        for eid, exp in EXPERIMENTS.items():
            configs = generate_configs(eid)
            logger.info("  %s: %s (%d runs)", eid, exp["name"], len(configs))
        return

    if not args.experiment:
        parser.print_help()
        return

    # Setup batch logger to file
    setup_batch_logger("results/experiments")

    exp_ids = list(EXPERIMENTS.keys()) if args.experiment == "ALL" else [args.experiment]

    for eid in exp_ids:
        if eid not in EXPERIMENTS:
            logger.error("Unknown experiment: %s", eid)
            continue

        exp = EXPERIMENTS[eid]
        configs = generate_configs(eid)
        logger.info("\n%s", "=" * 60)
        logger.info("%s: %s — %d runs", eid, exp["name"], len(configs))
        logger.info("%s", "=" * 60)

        if args.aggregate:
            aggregate_results(eid)
            continue

        success = 0
        for i, cfg in enumerate(configs):
            logger.info("[%d/%d] %s / %s / %s",
                        i + 1, len(configs), cfg["algorithm"], cfg["attack"], cfg["dataset"])
            if run_single(cfg, dry_run=args.dry_run):
                success += 1

        logger.info("Completed: %d/%d", success, len(configs))

        if not args.dry_run:
            aggregate_results(eid)


if __name__ == "__main__":
    main()
