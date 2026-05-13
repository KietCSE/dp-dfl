"""Plot accuracy-vs-round line charts for Fashion-MNIST experiments.

Supports two scenario groups under `results/fashsion-mnist/`:
  - Scale-attack intensity:   scale_5 / scale_10 / scale_20
  - Malicious-client ratio:   20% / 40% / 60%

For each subfolder, group records by `algorithm`, take the first 150 rounds, round
accuracy to 2 decimals, and render one comparison chart (5 algorithms per chart).
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent / "results" / "fashsion-mnist"
MAX_ROUNDS = 150
LINE_WIDTH = 2.0

# Map raw algorithm names (from JSON) to display names shown in legend.
# Algorithms not listed keep their original name.
ALGO_DISPLAY_NAMES = {
    "cfl-fedavg": "LDP-DFL-FedAvg",
    "noise-game": "AEGIS",
}

# Each scenario group renders one chart per subdir. `title_fmt` receives `n` (the
# integer parsed from the subdir name); `output_slug_fmt` builds the PNG basename.
SCENARIO_GROUPS = [
    {
        "subdirs": ["scale_5", "scale_10", "scale_20"],
        "title_fmt": "DFL Training Accuracy — Scale Attack ×{n}",
        "output_slug_fmt": "accuracy-vs-round-scale_{n}",
    },
    {
        "subdirs": ["20%", "40%", "60%"],
        "title_fmt": "DFL Training Accuracy — Malicious Ratio {n}%",
        "output_slug_fmt": "accuracy-vs-round-malicious_{n}pct",
    },
]


def load_algorithm_curves(scenario_dir: Path) -> dict[str, tuple[list[int], list[float]]]:
    """Group records in all JSON files of `scenario_dir` by `algorithm`.

    Returns mapping {algorithm: (rounds_sorted, accuracies_rounded_2dp)} restricted
    to round < MAX_ROUNDS. Accuracy values are rounded to 2 decimal places.
    """
    by_algo: dict[str, list[tuple[int, float]]] = defaultdict(list)

    for fn in sorted(os.listdir(scenario_dir)):
        if not fn.endswith(".json"):
            continue
        with open(scenario_dir / fn) as f:
            records = json.load(f)
        if not isinstance(records, list):
            continue
        for rec in records:
            if not isinstance(rec, dict):
                continue
            algo = rec.get("algorithm")
            rnd = rec.get("round")
            acc = rec.get("accuracy")
            if algo is None or rnd is None or acc is None:
                continue
            if rnd >= MAX_ROUNDS:
                continue
            by_algo[algo].append((int(rnd), round(float(acc), 2)))

    curves: dict[str, tuple[list[int], list[float]]] = {}
    for algo, points in by_algo.items():
        points.sort(key=lambda p: p[0])
        rounds = [p[0] for p in points]
        accs = [p[1] for p in points]
        curves[algo] = (rounds, accs)
    return curves


def plot_scenario(title: str, curves: dict, output_path: Path) -> None:
    """Render one comparison line chart for a scenario."""
    plt.figure(figsize=(10, 6))
    display = {algo: ALGO_DISPLAY_NAMES.get(algo, algo) for algo in curves}
    # Sort by display name (case-insensitive) so color order stays consistent
    # across charts within and across scenario groups.
    for algo in sorted(curves.keys(), key=lambda a: display[a].lower()):
        rounds, accs = curves[algo]
        plt.plot(rounds, accs, label=display[algo], linewidth=LINE_WIDTH)
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _parse_n(subdir_name: str) -> str:
    """Extract first integer substring from subdir name (e.g. 'scale_10' -> '10', '40%' -> '40')."""
    m = re.search(r"\d+", subdir_name)
    return m.group(0) if m else subdir_name


def main() -> None:
    for group in SCENARIO_GROUPS:
        for subdir_name in group["subdirs"]:
            scenario_dir = BASE_DIR / subdir_name
            if not scenario_dir.is_dir():
                print(f"Skip (missing): {scenario_dir}")
                continue
            curves = load_algorithm_curves(scenario_dir)
            if not curves:
                print(f"Skip (no data): {scenario_dir}")
                continue
            n = _parse_n(subdir_name)
            title = group["title_fmt"].format(n=n)
            output_path = scenario_dir / f"{group['output_slug_fmt'].format(n=n)}.png"
            plot_scenario(title, curves, output_path)
            rel = output_path.relative_to(Path(__file__).parent)
            print(f"Saved: {rel} (algorithms: {sorted(curves.keys())})")


if __name__ == "__main__":
    main()
