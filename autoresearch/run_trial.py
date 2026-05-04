from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kv_benchmark import BenchmarkProfile, get_profile, run_benchmark
from autoresearch.scoring import summarize


def quicken(profile: BenchmarkProfile) -> BenchmarkProfile:
    return replace(
        profile,
        recall_lengths=profile.recall_lengths[:2],
        recall_samples=min(2, profile.recall_samples),
        perplexity_lengths=profile.perplexity_lengths[:1],
        max_scored_tokens=min(512, profile.max_scored_tokens),
        memory_prompt_tokens=min(1024, profile.memory_prompt_tokens),
        throughput_new_tokens=min(8, profile.throughput_new_tokens),
        throughput_iters=1,
        niah_lengths=profile.niah_lengths[:1],
        niah_depths=profile.niah_depths[:1],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoresearch trial wrapper for tiered KV.")
    parser.add_argument("--profile", choices=("local", "runpod-h100"), default="local")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--run-niah", action="store_true")
    parser.add_argument("--output-dir", default="autoresearch/runs")
    parser.add_argument("--policy-config-file", default=None)
    parser.add_argument("--policy-config-json", default=None)
    args = parser.parse_args()

    policy_config = None
    if args.policy_config_file and args.policy_config_json:
        raise SystemExit("use only one of --policy-config-file or --policy-config-json")
    if args.policy_config_file:
        policy_config = json.loads(Path(args.policy_config_file).read_text())
    if args.policy_config_json:
        policy_config = json.loads(args.policy_config_json)

    profile = get_profile(args.profile, model_id=args.model_id)
    if args.quick:
        profile = quicken(profile)

    run_dir = Path(args.output_dir) / f"{profile.name}_{int(time.time())}"
    result = run_benchmark(
        profile,
        device=args.device,
        policy_config=policy_config,
        run_recall=True,
        run_perplexity=True,
        run_niah=args.run_niah,
        run_memory=True,
        output_dir=run_dir,
    )
    summary = summarize(result)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
