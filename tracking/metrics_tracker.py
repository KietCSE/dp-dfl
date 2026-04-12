"""CSV/JSON metrics export + matplotlib plots for DFL experiments."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class MetricsTracker:
    def __init__(self, output_dir: str, metadata: Dict[str, Any] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata or {}
        self.rounds: List[Dict[str, Any]] = []
        self.node_rounds: List[Dict[str, Any]] = []

    def log_round(self, round_num: int, **metrics):
        self.rounds.append({**self.metadata, "round": round_num, **metrics})

    def log_node_round(self, round_num: int, nodes_data: Dict[int, Dict[str, Any]]):
        """Log per-node metrics for a single round.

        nodes_data: {node_id: {accuracy, test_loss, precision, recall, f1_score,
                                update_norm, kurtosis, is_attacker}}
        """
        # Convert int keys to str for JSON serialization
        self.node_rounds.append({
            "round": round_num,
            "nodes": {str(k): v for k, v in nodes_data.items()},
        })

    # ── Export ────────────────────────────────────────────────────────

    def to_csv(self, filename: str = "metrics.csv"):
        if not self.rounds:
            return
        path = self.output_dir / filename
        keys = list(dict.fromkeys(k for r in self.rounds for k in r.keys()))
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rounds)

    def to_json(self, filename: str = "metrics.json"):
        with open(self.output_dir / filename, "w") as f:
            json.dump(self.rounds, f, indent=2)

    def save_node_data(self):
        """Write per-node per-round JSON files into node_data/ subfolder."""
        if not self.node_rounds:
            return
        node_dir = self.output_dir / "node_data"
        node_dir.mkdir(parents=True, exist_ok=True)
        for entry in self.node_rounds:
            rnd = entry["round"]
            path = node_dir / f"round_{rnd:03d}.json"
            with open(path, "w") as f:
                json.dump(entry, f, indent=2)

    def save_report(self, timestamp: str = ""):
        """Write a human-readable report.txt summarizing the experiment."""
        if not self.rounds:
            return
        last = self.rounds[-1]
        n = len(self.rounds)

        def avg(key):
            vals = [r.get(key, 0) for r in self.rounds]
            return sum(vals) / n if n else 0

        lines = [
            "=" * 60,
            "REPORT: DP-SGD Decentralized Federated Learning",
            "=" * 60,
        ]
        if timestamp:
            lines.append(f"Timestamp:                  {timestamp}")
        lines += [
            f"Rounds completed:           {n}",
            f"Final accuracy:             {last.get('accuracy', 0):.4f}",
            f"Final test loss:            {last.get('test_loss', 0):.4f}",
            f"Final epsilon:              {last.get('epsilon', 0):.2f}",
            f"Best alpha:                 {last.get('best_alpha', 'N/A')}",
            f"Avg precision:              {avg('precision'):.4f}",
            f"Avg recall:                 {avg('recall'):.4f}",
            f"Avg F1 score:               {avg('f1_score'):.4f}",
            f"Avg update norm (honest):   {avg('mean_update_norm_honest'):.4f}",
            f"Avg update norm (attacker): {avg('mean_update_norm_attacker'):.4f}",
        ]
        if any("kurtosis_honest" in r for r in self.rounds):
            lines += [
                f"Avg kurtosis (honest):      {avg('kurtosis_honest'):.6f}",
                f"Avg kurtosis (attacker):    {avg('kurtosis_attacker'):.2f}",
            ]
        lines.append("=" * 60)
        report = "\n".join(lines)
        with open(self.output_dir / "report.txt", "w") as f:
            f.write(report + "\n")
        return report

    # ── Plots ─────────────────────────────────────────────────────────

    def plot_accuracy(self, filename: str = "accuracy.png"):
        rounds = [r["round"] for r in self.rounds if "accuracy" in r]
        acc = [r["accuracy"] for r in self.rounds if "accuracy" in r]
        if not rounds:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, acc, color="steelblue")
        plt.xlabel("Round"); plt.ylabel("Accuracy")
        plt.title("DFL Training Accuracy")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_accuracy_spread(self, filename: str = "accuracy_spread.png"):
        """Plot per-node accuracy spread for honest nodes across rounds."""
        if not self.node_rounds:
            return

        # Collect honest node IDs from first round
        first = self.node_rounds[0]["nodes"]
        honest_ids = [nid for nid, d in first.items() if not d.get("is_attacker", False)]
        if not honest_ids:
            return

        rounds_x = [entry["round"] for entry in self.node_rounds]
        # Per-node accuracy traces
        node_accs = {nid: [] for nid in honest_ids}
        mean_accs, min_accs, max_accs = [], [], []

        for entry in self.node_rounds:
            accs = []
            for nid in honest_ids:
                a = entry["nodes"].get(nid, {}).get("accuracy", 0.0)
                node_accs[nid].append(a)
                accs.append(a)
            mean_accs.append(sum(accs) / len(accs))
            min_accs.append(min(accs))
            max_accs.append(max(accs))

        plt.figure(figsize=(10, 6))
        # Thin lines per node
        for nid in honest_ids:
            plt.plot(rounds_x, node_accs[nid], alpha=0.15, linewidth=0.8, color="steelblue")
        # Min-max shading
        plt.fill_between(rounds_x, min_accs, max_accs, alpha=0.15, color="steelblue")
        # Bold mean line
        plt.plot(rounds_x, mean_accs, color="steelblue", linewidth=2.5, label="Mean")
        plt.xlabel("Round"); plt.ylabel("Accuracy")
        plt.title("Accuracy Spread (Honest Nodes)")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_privacy_budget(self, filename: str = "epsilon.png"):
        rounds = [r["round"] for r in self.rounds if "epsilon" in r]
        eps = [r["epsilon"] for r in self.rounds if "epsilon" in r]
        if not rounds:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, eps, color="purple")
        plt.xlabel("Round"); plt.ylabel("Epsilon")
        plt.title("Privacy Budget Over Rounds")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_kurtosis(self, filename: str = "kurtosis.png"):
        rounds = [r["round"] for r in self.rounds if "kurtosis_honest" in r]
        k_h = [r["kurtosis_honest"] for r in self.rounds if "kurtosis_honest" in r]
        k_a = [r["kurtosis_attacker"] for r in self.rounds if "kurtosis_attacker" in r]
        if not rounds:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, k_h, label="Honest", color="steelblue")
        plt.plot(rounds, k_a, label="Attacker", color="tomato")
        plt.xlabel("Round"); plt.ylabel("Mean Excess Kurtosis")
        plt.title("Kurtosis: Honest vs Attacker")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_detection(self, filename: str = "detection.png"):
        rounds = [r["round"] for r in self.rounds if "precision" in r]
        prec = [r["precision"] for r in self.rounds if "precision" in r]
        rec = [r["recall"] for r in self.rounds if "recall" in r]
        if not rounds:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, prec, label="Precision", color="steelblue")
        plt.plot(rounds, rec, label="Recall", color="tomato")
        plt.xlabel("Round"); plt.ylabel("Score"); plt.ylim(-0.05, 1.05)
        plt.title("Detection Performance")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    # ── Summary ───────────────────────────────────────────────────────

    def summary(self) -> str:
        if not self.rounds:
            return "No data recorded."
        last = self.rounds[-1]
        n = len(self.rounds)

        def avg(key):
            vals = [r.get(key, 0) for r in self.rounds]
            return sum(vals) / n if n else 0

        lines = [
            "=" * 60,
            "SUMMARY: DP-SGD Decentralized Federated Learning",
            "=" * 60,
            f"Rounds completed:           {n}",
            f"Final accuracy:             {last.get('accuracy', 0):.4f}",
            f"Final test loss:            {last.get('test_loss', 0):.4f}",
            f"Final epsilon:              {last.get('epsilon', 0):.2f}",
            f"Best alpha:                 {last.get('best_alpha', 'N/A')}",
            f"Avg precision:              {avg('precision'):.4f}",
            f"Avg recall:                 {avg('recall'):.4f}",
            f"Avg F1 score:               {avg('f1_score'):.4f}",
            f"Avg update norm (honest):   {avg('mean_update_norm_honest'):.4f}",
            f"Avg update norm (attacker): {avg('mean_update_norm_attacker'):.4f}",
        ]
        return "\n".join(lines)
