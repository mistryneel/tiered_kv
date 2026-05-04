from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


COLORS = {
    "FixedWindow": "#1f77b4",
    "Ratio": "#ff7f0e",
    "Hybrid": "#2ca02c",
    None: "#777777",
}


def latest_run(root: Path) -> Path:
    runs = [p for p in root.glob("local_*") if (p / "progress.jsonl").exists()]
    if not runs:
        raise FileNotFoundError(f"no tuning runs found under {root}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def load_rows(run_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in (run_dir / "progress.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no trials found in {run_dir / 'progress.jsonl'}")
    return rows


def best_so_far(values: List[float]) -> List[float]:
    out: List[float] = []
    best = float("-inf")
    for value in values:
        best = max(best, value)
        out.append(best)
    return out


def plot(run_dir: Path, output: Path) -> None:
    rows = load_rows(run_dir)
    trials = [int(row["trial_idx"]) for row in rows]
    summaries = [row["summary"] for row in rows]
    scores = [float(s["score"]) for s in summaries]
    best_scores = best_so_far(scores)
    best_strategy = [s.get("best_quantized_strategy") for s in summaries]
    colors = [COLORS.get(name, COLORS[None]) for name in best_strategy]
    memory = [s.get("best_memory_ratio") for s in summaries]
    recall = [float(s.get("best_quantized_recall_mean", 0.0)) for s in summaries]
    ready = [bool(s.get("h100_candidate_ready")) for s in summaries]

    ppl_mean = []
    for summary in summaries:
        best = summary.get("best_strategy") or {}
        ppl_mean.append(best.get("ppl_ratio_mean"))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    ax = axes[0][0]
    ax.scatter(trials, scores, c=colors, s=18, alpha=0.75, label="trial score")
    ax.plot(trials, best_scores, color="black", linewidth=2.0, label="best so far")
    ax.set_ylabel("score")
    ax.set_title("Policy Tuning Score")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[0][1]
    ax.scatter(trials, memory, c=colors, s=18, alpha=0.75)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("KV memory / FP16")
    ax.set_title("Best Strategy Memory Ratio")
    ax.grid(True, alpha=0.25)

    ax = axes[1][0]
    ax.scatter(trials, recall, c=colors, s=18, alpha=0.75)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("trial")
    ax.set_ylabel("mean recall")
    ax.set_title("Best Strategy Recall")
    ax.grid(True, alpha=0.25)

    ax = axes[1][1]
    ax.scatter(trials, ppl_mean, c=colors, s=18, alpha=0.75)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("trial")
    ax.set_ylabel("PPL ratio")
    ax.set_title("Best Strategy Perplexity Ratio")
    ax.grid(True, alpha=0.25)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=name)
        for name, color in COLORS.items()
        if name is not None
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    fig.suptitle(
        f"Policy tuning: {len(rows)} trials, best trial {trials[best_idx]} "
        f"({best_strategy[best_idx]}), score={scores[best_idx]:.4f}, "
        f"passing={sum(ready)}/{len(ready)}",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot policy tuning progress.")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--root", default="autoresearch/tuning")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run(Path(args.root))
    output = Path(args.output) if args.output else run_dir / "progress.png"
    plot(run_dir, output)
    print(output)


if __name__ == "__main__":
    main()
