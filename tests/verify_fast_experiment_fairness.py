"""Verify all fast-experiment MNIST configs share identical shared fields.

Exits with code 1 if any divergence detected on shared keys. Fields that are
algorithm-specific (aggregation, dp.noise_mode, adaptive_noise, trust, fltrust,
noise_game) and known-exempted fields (n_rounds, n_workers) are excluded.
"""
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "config" / "fast-experiment" / "mnist"

SHARED_KEYS = [
    ("model", "hidden_size"),
    ("topology", "n_nodes"),
    ("topology", "n_attackers"),
    ("topology", "n_neighbors"),
    ("topology", "seed"),
    ("training", "local_epochs"),
    ("training", "batch_size"),
    ("training", "lr"),
    ("dataset", "split", "mode"),
    ("dataset", "split", "samples_per_node"),
    ("attack", "type"),
    ("attack", "scale_factor"),
    ("attack", "start_round"),
    ("dp", "sampling_rate"),
]

# Known exempt keys (intentionally allowed to differ)
EXEMPT_NOTES = {
    "training.n_rounds": "fedavg may converge faster",
    "training.n_workers": "some algos require single-threaded",
    "dataset.split.alpha": "ignored in iid mode",
    "dp.noise_mode": "per-algorithm DP strategy",
    "dp.clip_bound": "only relevant when noise_mode != none",
    "aggregation.type": "algorithm selector",
    "aggregation.params": "algorithm-specific",
}


def _get(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main():
    configs = {p.name: yaml.safe_load(p.read_text())
               for p in sorted(ROOT.glob("*.yaml"))}
    if not configs:
        print(f"No configs found in {ROOT}")
        return 1

    print(f"Found {len(configs)} configs:")
    for name in configs:
        print(f"  - {name}")
    print()

    divergences = []
    for keys in SHARED_KEYS:
        vals = {name: _get(c, keys) for name, c in configs.items()}
        uniq = {v for v in vals.values() if v is not None}
        if len(uniq) > 1:
            divergences.append((keys, vals))

    if divergences:
        print("DIVERGENCES DETECTED:")
        for keys, vals in divergences:
            print(f"\n  {'.'.join(keys)}:")
            for name, v in sorted(vals.items()):
                print(f"    {name}: {v}")
        print(f"\n{len(divergences)} shared field(s) diverge across configs.")
        return 1

    print(f"OK — {len(configs)} configs share {len(SHARED_KEYS)} fields identically")
    print("\nExempt fields (per-algorithm, not checked):")
    for k, note in EXEMPT_NOTES.items():
        print(f"  {k}: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
