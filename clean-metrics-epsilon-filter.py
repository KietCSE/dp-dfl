"""Clean metrics.json: keep algorithm, round, accuracy, epsilon.
Only include rounds where epsilon <= 15. Save as metrics-clean-2.json."""

import json
from pathlib import Path

EPSILON_MAX = 15.0


def clean_metrics_epsilon(results_dir: str = "./results"):
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Directory not found: {results_dir}")
        return

    folders = sorted(d for d in results_path.iterdir() if d.is_dir())
    print(f"Found {len(folders)} result folders\n")

    for folder in folders:
        src = folder / "metrics.json"
        dst = folder / "metrics-clean-2.json"

        if not src.exists():
            print(f"  SKIP {folder.name} — no metrics.json")
            continue

        with open(src) as f:
            data = json.load(f)

        total = len(data)
        cleaned = [
            {
                "algorithm": row.get("algorithm", ""),
                "round": row.get("round", 0),
                "accuracy": row.get("accuracy", 0.0),
                "epsilon": row.get("epsilon", 0.0),
            }
            for row in data
            if row.get("epsilon", 0.0) <= EPSILON_MAX
        ]

        with open(dst, "w") as f:
            json.dump(cleaned, f, indent=2)

        print(f"  OK {folder.name} — {len(cleaned)}/{total} rounds (eps <= {EPSILON_MAX})")

    print("\nDone.")


if __name__ == "__main__":
    clean_metrics_epsilon()
