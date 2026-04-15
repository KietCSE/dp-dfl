"""Clean metrics.json in all result folders.
Keep only: algorithm, round, accuracy. Save as metrics-clean.json."""

import json
from pathlib import Path


def clean_metrics(results_dir: str = "./results"):
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Directory not found: {results_dir}")
        return

    folders = sorted(d for d in results_path.iterdir() if d.is_dir())
    print(f"Found {len(folders)} result folders\n")

    for folder in folders:
        src = folder / "metrics.json"
        dst = folder / "metrics-clean.json"

        if not src.exists():
            print(f"  SKIP {folder.name} — no metrics.json")
            continue

        with open(src) as f:
            data = json.load(f)

        cleaned = [
            {
                "algorithm": row.get("algorithm", ""),
                "round": row.get("round", 0),
                "accuracy": row.get("accuracy", 0.0),
            }
            for row in data
        ]

        with open(dst, "w") as f:
            json.dump(cleaned, f, indent=2)

        print(f"  OK {folder.name} — {len(cleaned)} rounds")

    print("\nDone.")


if __name__ == "__main__":
    clean_metrics()
