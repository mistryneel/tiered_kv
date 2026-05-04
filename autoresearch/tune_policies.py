from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoresearch.run_trial import quicken
from autoresearch.scoring import summarize
from experiments.kv_benchmark import (
    BenchmarkProfile,
    evaluate_loaded_model,
    get_profile,
    load_model_and_tokenizer,
)


TIER_BITS_CHOICES = [
    [16, 8, 4, 2],
    [16, 8, 4, 4],
    [16, 8, 8, 4],
    [16, 8, 8, 8],
]


def _seed_configs() -> List[Dict[str, Any]]:
    return [
        {
            "name": "default_local",
            "tier_bits": [16, 8, 4, 2],
            "group_size": 32,
            "fixed": {"fp16": 128, "int8": 512, "int4": 2048, "n_sink": 4},
            "ratio": {"fp16_pct": 0.25, "int8_pct": 0.35, "int4_pct": 0.40, "n_sink": 4, "min_fp16": 128},
            "hybrid": {"n_sink": 4, "fp16": 128, "geometric": True, "growth": 4.0},
        },
        {
            "name": "less_int2_pressure",
            "tier_bits": [16, 8, 4, 4],
            "group_size": 32,
            "fixed": {"fp16": 192, "int8": 1024, "int4": 3072, "n_sink": 4},
            "ratio": {"fp16_pct": 0.20, "int8_pct": 0.35, "int4_pct": 0.45, "n_sink": 4, "min_fp16": 128},
            "hybrid": {"n_sink": 4, "fp16": 192, "geometric": True, "growth": 4.0},
        },
        {
            "name": "int8_floor",
            "tier_bits": [16, 8, 8, 8],
            "group_size": 32,
            "fixed": {"fp16": 128, "int8": 1024, "int4": 2048, "n_sink": 4},
            "ratio": {"fp16_pct": 0.15, "int8_pct": 0.35, "int4_pct": 0.50, "n_sink": 4, "min_fp16": 128},
            "hybrid": {"n_sink": 4, "fp16": 192, "geometric": True, "growth": 3.0},
        },
        {
            "name": "wide_recent",
            "tier_bits": [16, 8, 4, 2],
            "group_size": 64,
            "fixed": {"fp16": 256, "int8": 1024, "int4": 3072, "n_sink": 8},
            "ratio": {"fp16_pct": 0.30, "int8_pct": 0.35, "int4_pct": 0.35, "n_sink": 8, "min_fp16": 192},
            "hybrid": {"n_sink": 8, "fp16": 256, "geometric": True, "growth": 3.0},
        },
    ]


def _random_ratio(rng: random.Random) -> Dict[str, Any]:
    fp16_pct = rng.choice([0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30])
    int8_pct = rng.choice([0.18, 0.22, 0.25, 0.30, 0.35, 0.40, 0.45])
    int4_pct = rng.choice([0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
    total = fp16_pct + int8_pct + int4_pct
    if total > 1.0:
        scale = 1.0 / total
        fp16_pct *= scale
        int8_pct *= scale
        int4_pct *= scale
    return {
        "fp16_pct": round(fp16_pct, 4),
        "int8_pct": round(int8_pct, 4),
        "int4_pct": round(int4_pct, 4),
        "n_sink": rng.choice([0, 4, 8]),
        "min_fp16": rng.choice([64, 96, 128, 192, 256]),
    }


def _random_config(rng: random.Random, trial_idx: int) -> Dict[str, Any]:
    fixed_fp16 = rng.choice([64, 96, 128, 192, 256, 384, 512])
    fixed_int8 = rng.choice([256, 512, 768, 1024, 1536, 2048])
    fixed_int4 = rng.choice([1024, 1536, 2048, 3072, 4096, 6144])
    hybrid_geometric = rng.random() < 0.85
    hybrid_fp16 = rng.choice([64, 96, 128, 192, 256, 384])
    cfg: Dict[str, Any] = {
        "name": f"random_{trial_idx:04d}",
        "tier_bits": rng.choice(TIER_BITS_CHOICES),
        "group_size": rng.choice([16, 32, 64]),
        "fixed": {
            "fp16": fixed_fp16,
            "int8": fixed_int8,
            "int4": fixed_int4,
            "n_sink": rng.choice([0, 4, 8]),
        },
        "ratio": _random_ratio(rng),
        "hybrid": {
            "n_sink": rng.choice([0, 4, 8]),
            "fp16": hybrid_fp16,
            "geometric": hybrid_geometric,
            "growth": rng.choice([2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]),
            "int8": rng.choice([256, 512, 768, 1024, 1536, 2048]),
            "int4": rng.choice([1024, 1536, 2048, 3072, 4096, 6144]),
        },
    }
    return cfg


def candidate_configs(seed: int) -> Iterator[Dict[str, Any]]:
    rng = random.Random(seed)
    for cfg in _seed_configs():
        yield cfg
    idx = 0
    while True:
        yield _random_config(rng, idx)
        idx += 1


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def update_leaderboard(
    leaderboard: List[Dict[str, Any]],
    *,
    trial_idx: int,
    trial_dir: Path,
    config: Dict[str, Any],
    summary: Dict[str, Any],
    elapsed_seconds: float,
) -> List[Dict[str, Any]]:
    row = {
        "trial_idx": trial_idx,
        "trial_dir": str(trial_dir),
        "config_name": config.get("name"),
        "elapsed_seconds": elapsed_seconds,
        "score": summary["score"],
        "h100_candidate_ready": summary["h100_candidate_ready"],
        "ready_strategy_count": summary["ready_strategy_count"],
        "best_quantized_strategy": summary["best_quantized_strategy"],
        "best_memory_ratio": summary["best_memory_ratio"],
        "best_quantized_recall_mean": summary["best_quantized_recall_mean"],
    }
    leaderboard.append(row)
    return sorted(leaderboard, key=lambda item: item["score"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-running local policy tuner for tiered KV.")
    parser.add_argument("--profile", choices=("local", "runpod-h100"), default="local")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--run-niah", action="store_true")
    parser.add_argument("--output-dir", default="autoresearch/tuning")
    parser.add_argument("--resume-dir", default=None)
    args = parser.parse_args()

    profile: BenchmarkProfile = get_profile(args.profile, model_id=args.model_id)
    if args.quick:
        profile = quicken(profile)

    started = time.time()
    deadline = started + args.hours * 3600
    run_dir = Path(args.resume_dir) if args.resume_dir else Path(args.output_dir) / f"{profile.name}_{int(started)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"
    leaderboard_path = run_dir / "leaderboard.json"
    best_path = run_dir / "best.json"
    completed_trials = 0
    if args.resume_dir and progress_path.exists():
        completed_trials = len(progress_path.read_text().splitlines())

    print(f"run_dir={run_dir}", flush=True)
    print(f"profile={profile}", flush=True)
    print(f"device={args.device}", flush=True)
    if completed_trials:
        print(f"resuming after {completed_trials} completed trials", flush=True)

    model, tokenizer, dtype = load_model_and_tokenizer(profile, args.device)
    leaderboard: List[Dict[str, Any]] = json.loads(leaderboard_path.read_text()) if leaderboard_path.exists() else []
    best_payload: Optional[Dict[str, Any]] = json.loads(best_path.read_text()) if best_path.exists() else None

    for trial_idx, config in enumerate(candidate_configs(args.seed)):
        if trial_idx < completed_trials:
            continue
        if args.max_trials is not None and trial_idx >= args.max_trials:
            break
        if time.time() >= deadline:
            break

        trial_start = time.time()
        trial_dir = run_dir / f"trial_{trial_idx:04d}_{config.get('name', 'config')}"
        print(f"\n=== trial {trial_idx} {config.get('name')} ===", flush=True)
        print(json.dumps(config, sort_keys=True), flush=True)

        try:
            result = evaluate_loaded_model(
                model,
                tokenizer,
                dtype,
                profile,
                device=args.device,
                policy_config=config,
                run_recall=True,
                run_perplexity=True,
                run_niah=args.run_niah,
                run_memory=True,
                output_dir=trial_dir,
                started=trial_start,
            )
            summary = summarize(result)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            summary = {
                "correctness_passed": False,
                "h100_candidate_ready": False,
                "ready_strategy_count": 0,
                "score": -100.0,
                "error": repr(exc),
                "per_strategy": [],
            }
            result = {"profile": profile.__dict__, "policy_config": config, "error": repr(exc)}

        elapsed = time.time() - trial_start
        write_json(trial_dir / "config.json", config)
        write_json(trial_dir / "summary.json", summary)
        leaderboard = update_leaderboard(
            leaderboard,
            trial_idx=trial_idx,
            trial_dir=trial_dir,
            config=config,
            summary=summary,
            elapsed_seconds=elapsed,
        )
        write_json(leaderboard_path, leaderboard)

        row = {
            "trial_idx": trial_idx,
            "elapsed_seconds": elapsed,
            "config": config,
            "summary": summary,
        }
        with progress_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        if best_payload is None or summary["score"] > best_payload["summary"]["score"]:
            best_payload = {"trial_idx": trial_idx, "config": config, "summary": summary, "trial_dir": str(trial_dir)}
            write_json(best_path, best_payload)

        print(
            "score={score:.4f} ready={ready} best={best} recall={recall:.3f} mem={mem}".format(
                score=summary["score"],
                ready=summary["h100_candidate_ready"],
                best=summary["best_quantized_strategy"],
                recall=summary["best_quantized_recall_mean"],
                mem=summary["best_memory_ratio"],
            ),
            flush=True,
        )
        print(f"current_best={leaderboard[0]}", flush=True)

    print(f"\nfinished run_dir={run_dir}", flush=True)
    if best_payload:
        print(json.dumps(best_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
