from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
for path in (ROOT, PARENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from tiered_kv import (
    FixedWindowPolicy,
    HybridPolicy,
    RatioPolicy,
    TierConfig,
    TieredKVCache,
    quantization as Q,
)
from tiered_kv import evaluation as E


@dataclass
class BenchmarkProfile:
    name: str
    model_id: str
    dtype: str
    recall_lengths: Sequence[int]
    recall_samples: int
    recall_pairs: int
    perplexity_lengths: Sequence[int]
    max_scored_tokens: int
    memory_prompt_tokens: int
    throughput_new_tokens: int
    throughput_iters: int
    niah_lengths: Sequence[int]
    niah_depths: Sequence[int]


def get_profile(name: str, model_id: Optional[str] = None) -> BenchmarkProfile:
    if name == "h100-safe":
        return BenchmarkProfile(
            name="h100-safe",
            model_id=model_id or "Qwen/Qwen2.5-7B-Instruct",
            dtype="bfloat16",
            recall_lengths=(1024, 2048, 4096, 8192, 16384),
            recall_samples=5,
            recall_pairs=20,
            perplexity_lengths=(2048, 8192, 16384),
            max_scored_tokens=8000,
            memory_prompt_tokens=16000,
            throughput_new_tokens=64,
            throughput_iters=3,
            niah_lengths=(2000, 8000, 16000, 32000),
            niah_depths=(10, 30, 50, 70, 90),
        )
    if name == "h100-full":
        return BenchmarkProfile(
            name="h100-full",
            model_id=model_id or "Qwen/Qwen2.5-7B-Instruct",
            dtype="bfloat16",
            recall_lengths=(1024, 2048, 4096, 8192, 16384),
            recall_samples=5,
            recall_pairs=20,
            perplexity_lengths=(2048, 8192, 32768),
            max_scored_tokens=8000,
            memory_prompt_tokens=16000,
            throughput_new_tokens=64,
            throughput_iters=3,
            niah_lengths=(2000, 8000, 16000, 32000),
            niah_depths=(10, 30, 50, 70, 90),
        )
    raise ValueError(f"unknown profile {name!r}; expected h100-safe or h100-full")


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype {name!r}")


def default_policy_config() -> Dict[str, Any]:
    return {
        "tier_bits": [16, 8, 4, 2],
        "group_size": 32,
        "fixed": {"fp16": 128, "int8": 1024, "int4": 8192, "n_sink": 4},
        "ratio": {"fp16_pct": 0.05, "int8_pct": 0.15, "int4_pct": 0.30, "n_sink": 4, "min_fp16": 32},
        "hybrid": {"n_sink": 4, "fp16": 64, "geometric": True, "growth": 8.0},
    }


def deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    if not override:
        return out
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_policy_config(policy_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return deep_merge(default_policy_config(), policy_config)


def make_strategies(compute_dtype: torch.dtype, policy_config: Dict[str, Any]) -> Dict[str, Callable[[], Any]]:
    def cfg() -> TierConfig:
        return TierConfig(
            tier_bits=list(policy_config["tier_bits"]),
            group_size=int(policy_config["group_size"]),
            compute_dtype=compute_dtype,
        )

    fixed = policy_config["fixed"]
    ratio = policy_config["ratio"]
    hybrid = policy_config["hybrid"]

    return {
        "FP16 baseline": lambda: DynamicCache(),
        "FixedWindow": lambda: TieredKVCache(
            policy=FixedWindowPolicy(
                fp16=int(fixed["fp16"]),
                int8=int(fixed["int8"]),
                int4=int(fixed["int4"]),
                n_sink=int(fixed.get("n_sink", 4)),
            ),
            config=cfg(),
        ),
        "Ratio": lambda: TieredKVCache(
            policy=RatioPolicy(
                fp16_pct=float(ratio["fp16_pct"]),
                int8_pct=float(ratio["int8_pct"]),
                int4_pct=float(ratio["int4_pct"]),
                n_sink=int(ratio.get("n_sink", 4)),
                min_fp16=int(ratio.get("min_fp16", 32)),
            ),
            config=cfg(),
        ),
        "Hybrid": lambda: TieredKVCache(
            policy=HybridPolicy(
                n_sink=int(hybrid.get("n_sink", 4)),
                fp16=int(hybrid["fp16"]),
                geometric=bool(hybrid.get("geometric", True)),
                growth=float(hybrid.get("growth", 4.0)),
                int8=int(hybrid.get("int8", 512)),
                int4=int(hybrid.get("int4", 4096)),
            ),
            config=cfg(),
        ),
    }


def load_model_and_tokenizer(profile: BenchmarkProfile, device: str):
    dtype = resolve_dtype(profile.dtype, "cpu" if device == "cpu" else "cuda")
    tokenizer = AutoTokenizer.from_pretrained(profile.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        profile.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model, tokenizer, dtype


def quantizer_sanity() -> Dict[str, float]:
    x = torch.linspace(10.0, 11.0, steps=128, dtype=torch.float16).reshape(1, 1, 128, 1)
    out: Dict[str, float] = {}
    for bits in (8, 4, 2):
        qt = Q.quantize(x, bits=bits, group_axis=-2, group_size=32)
        x_hat = Q.dequantize(qt)
        out[f"int{bits}_max_abs_error"] = float((x_hat - x).abs().max())
    return out


@torch.no_grad()
def cache_passthrough_sanity(model, tokenizer, device: str, compute_dtype: torch.dtype) -> Dict[str, float]:
    ids = tokenizer("The cache sanity check asks for the next token:", return_tensors="pt").to(device)
    baseline = DynamicCache()
    tiered = TieredKVCache(
        policy=FixedWindowPolicy(fp16=8, int8=16, int4=32, n_sink=2),
        config=TierConfig(tier_bits=[16, 16, 16, 16], group_size=32, compute_dtype=compute_dtype),
    )
    base_logits = model(**ids, past_key_values=baseline, use_cache=True).logits[:, -1, :]
    tier_logits = model(**ids, past_key_values=tiered, use_cache=True).logits[:, -1, :]
    diff = (base_logits.float() - tier_logits.float()).abs()
    return {
        "max_abs_logit_diff": float(diff.max()),
        "mean_abs_logit_diff": float(diff.mean()),
        "tokens": int(ids.input_ids.shape[1]),
    }


def make_prompt_ids(tokenizer, target_tokens: int) -> torch.Tensor:
    unit = "The quick brown fox jumps over the lazy dog. KV cache measurements use deterministic text. "
    text = unit
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += unit
    return tokenizer(text, return_tensors="pt").input_ids[:, :target_tokens]


def run_benchmark(
    profile: BenchmarkProfile,
    *,
    device: str,
    policy_config: Dict[str, Any],
    run_recall: bool = True,
    run_perplexity: bool = True,
    run_niah: bool = False,
    run_memory: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    started = time.time()
    model, tokenizer, dtype = load_model_and_tokenizer(profile, device)
    strategies = make_strategies(dtype, policy_config)
    cfg = model.config
    result: Dict[str, Any] = {
        "profile": asdict(profile),
        "policy_config": policy_config,
        "device": device,
        "model": {
            "num_hidden_layers": int(cfg.num_hidden_layers),
            "num_attention_heads": int(cfg.num_attention_heads),
            "num_key_value_heads": int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)),
            "hidden_size": int(cfg.hidden_size),
            "dtype": str(next(model.parameters()).dtype),
        },
        "sanity": {
            "quantizer": quantizer_sanity(),
            "cache_passthrough": cache_passthrough_sanity(model, tokenizer, device, dtype),
        },
        "recall": [],
        "perplexity": [],
        "niah": [],
        "memory": [],
        "throughput": [],
    }

    if run_recall:
        for name, factory in strategies.items():
            rows = E.synthetic_kv_recall(
                model,
                tokenizer,
                factory,
                context_lengths=profile.recall_lengths,
                num_pairs=profile.recall_pairs,
                n_samples_per_length=profile.recall_samples,
                max_new_tokens=12,
                seed=0,
                device=device,
                show_progress=False,
            )
            result["recall"].append(
                {
                    "strategy": name,
                    "lengths": [r.context_len for r in rows],
                    "accuracy": [r.accuracy for r in rows],
                    "n_correct": [r.n_correct for r in rows],
                    "n_total": [r.n_total for r in rows],
                    "sample_outputs": [r.sample_outputs for r in rows],
                }
            )

    if run_perplexity:
        text = E.get_pg19_book(min_chars=500_000)
        result["perplexity_text_chars"] = len(text)
        for name, factory in strategies.items():
            rows = []
            for length in profile.perplexity_lengths:
                ppl = E.perplexity_sliding(
                    model,
                    tokenizer,
                    factory,
                    text,
                    context_len=length,
                    stride=max(1, length // 2),
                    max_scored_tokens=profile.max_scored_tokens,
                    device=device,
                    show_progress=False,
                )
                rows.append(asdict(ppl))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            result["perplexity"].append({"strategy": name, "rows": rows})

    if run_niah:
        haystack = E.get_haystack_text(min_chars=500_000)
        for name, factory in strategies.items():
            heatmap = E.niah_heatmap(
                model,
                tokenizer,
                factory,
                haystack,
                lengths=profile.niah_lengths,
                depths=profile.niah_depths,
                max_new_tokens=64,
                device=device,
                show_progress=False,
            )
            result["niah"].append(
                {
                    "strategy": name,
                    "lengths": heatmap.lengths,
                    "depths": heatmap.depths,
                    "accuracy": heatmap.accuracy.tolist(),
                    "sample_outputs": heatmap.sample_outputs,
                }
            )

    if run_memory:
        prompt_ids = make_prompt_ids(tokenizer, profile.memory_prompt_tokens)
        for name, factory in strategies.items():
            cache = factory()
            mem = E.measure_memory(model, cache, prompt_ids, label=name, device=device)
            result["memory"].append({"strategy": name, **asdict(mem)})
            thru = E.measure_throughput(
                model,
                factory,
                prompt_ids,
                n_new=profile.throughput_new_tokens,
                n_warmup=1,
                n_iter=profile.throughput_iters,
                device=device,
            )
            result["throughput"].append({"strategy": name, **thru})

    result["elapsed_seconds"] = time.time() - started
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{profile.name}_{int(started)}.json"
        result["output_path"] = str(path)
        path.write_text(json.dumps(result, indent=2))
    return result


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def summarize(result: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"recall": {}, "perplexity_ratio": {}, "memory": {}, "throughput": {}, "niah": {}}
    recall = {g["strategy"]: g for g in result.get("recall", [])}
    baseline_recall = mean([float(x) for x in recall.get("FP16 baseline", {}).get("accuracy", [])])
    for name, group in recall.items():
        vals = [float(x) for x in group["accuracy"]]
        out["recall"][name] = {"mean": mean(vals), "retention": mean(vals) / max(baseline_recall, 1e-9)}

    ppl = {g["strategy"]: g for g in result.get("perplexity", [])}
    if "FP16 baseline" in ppl:
        base = {int(row["context_len"]): float(row["perplexity"]) for row in ppl["FP16 baseline"]["rows"]}
        for name, group in ppl.items():
            ratios = [float(row["perplexity"]) / base[int(row["context_len"])] for row in group["rows"]]
            out["perplexity_ratio"][name] = {"mean": mean(ratios), "max": max(ratios)}

    mem = {m["strategy"]: m for m in result.get("memory", [])}
    base_mem = float(mem.get("FP16 baseline", {}).get("cache_bytes_empirical", 0.0))
    for name, row in mem.items():
        out["memory"][name] = {
            "cache_MB": float(row["cache_bytes_empirical"]) / 1e6,
            "ratio": float(row["cache_bytes_empirical"]) / max(base_mem, 1.0),
        }

    thru = {t["strategy"]: t for t in result.get("throughput", [])}
    base_tps = float(thru.get("FP16 baseline", {}).get("tokens_per_second", 0.0))
    for name, row in thru.items():
        out["throughput"][name] = {
            "tokens_per_second": float(row["tokens_per_second"]),
            "ratio": float(row["tokens_per_second"]) / max(base_tps, 1e-9),
        }

    for group in result.get("niah", []):
        vals = [float(v) for row in group["accuracy"] for v in row]
        out["niah"][group["strategy"]] = {"accuracy": mean(vals), "correct": int(sum(vals)), "total": len(vals)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the H100 tiered-KV benchmark drop-in.")
    parser.add_argument("--profile", choices=("h100-safe", "h100-full"), default="h100-safe")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--policy-config-file", default=str(ROOT / "best_policy_config.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "runpod_h100"))
    parser.add_argument("--run-niah", action="store_true")
    parser.add_argument("--skip-recall", action="store_true")
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    args = parser.parse_args()

    policy_path = Path(args.policy_config_file)
    if not policy_path.exists():
        raise FileNotFoundError(f"missing policy config: {policy_path}")
    policy_config = resolve_policy_config(json.loads(policy_path.read_text()))

    profile = get_profile(args.profile, model_id=args.model_id)
    print("profile", profile)
    print("device", args.device)
    print("policy_config_file", policy_path)
    print(json.dumps(policy_config, indent=2))

    result = run_benchmark(
        profile,
        device=args.device,
        policy_config=policy_config,
        run_recall=not args.skip_recall,
        run_perplexity=not args.skip_perplexity,
        run_niah=args.run_niah,
        run_memory=not args.skip_memory,
        output_dir=Path(args.output_dir),
    )
    summary = summarize(result)
    out_path = Path(result["output_path"])
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print("saved", out_path)
    print("summary", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
