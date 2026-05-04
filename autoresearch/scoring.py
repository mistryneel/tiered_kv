from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _by_strategy(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {item["strategy"]: item for item in items}


def _ppl_ratios(result: Dict[str, Any], strategy: str) -> Dict[str, float]:
    groups = _by_strategy(result.get("perplexity", []))
    baseline = groups.get("FP16 baseline", {}).get("rows", [])
    current = groups.get(strategy, {}).get("rows", [])
    base_by_len = {int(row["context_len"]): float(row["perplexity"]) for row in baseline}
    ratios = []
    for row in current:
        base = base_by_len.get(int(row["context_len"]))
        if base and base > 0:
            ratios.append(float(row["perplexity"]) / base)
    return {
        "ppl_ratio_mean": _mean(ratios) if ratios else math.inf,
        "ppl_ratio_max": max(ratios) if ratios else math.inf,
        "ppl_finite": all(math.isfinite(x) for x in ratios) and bool(ratios),
    }


def summarize(result: Dict[str, Any]) -> Dict[str, Any]:
    recall_groups = _by_strategy(result.get("recall", []))
    memory_groups = _by_strategy(result.get("memory", []))
    throughput_groups = _by_strategy(result.get("throughput", []))

    baseline_recall_values = [
        float(x) for x in recall_groups.get("FP16 baseline", {}).get("accuracy", [])
    ]
    baseline_recall_mean = _mean(baseline_recall_values)
    baseline_recall_min = min(baseline_recall_values) if baseline_recall_values else 0.0
    baseline_bytes = float(memory_groups.get("FP16 baseline", {}).get("cache_bytes_empirical", 0.0))
    baseline_tps = float(throughput_groups.get("FP16 baseline", {}).get("tokens_per_second", 0.0))

    per_strategy = []
    for strategy, recall_group in recall_groups.items():
        if strategy == "FP16 baseline":
            continue

        recall_values = [float(x) for x in recall_group.get("accuracy", [])]
        recall_mean = _mean(recall_values)
        recall_min = min(recall_values) if recall_values else 0.0
        recall_retention = recall_mean / max(baseline_recall_mean, 1e-9)
        recall_min_retention = recall_min / max(baseline_recall_min, 1e-9)
        recall_score = min(recall_retention, 1.0)
        recall_min_score = min(recall_min_retention, 1.0)

        cache_bytes = float(memory_groups.get(strategy, {}).get("cache_bytes_empirical", 0.0))
        memory_ratio = cache_bytes / baseline_bytes if baseline_bytes > 0 and cache_bytes > 0 else math.inf
        memory_gain = 1.0 - min(memory_ratio, 1.0) if math.isfinite(memory_ratio) else 0.0

        tps = float(throughput_groups.get(strategy, {}).get("tokens_per_second", 0.0))
        throughput_ratio = tps / baseline_tps if baseline_tps > 0 and tps > 0 else 0.0

        ppl = _ppl_ratios(result, strategy)
        ppl_excess = max(0.0, ppl["ppl_ratio_mean"] - 1.0) if math.isfinite(ppl["ppl_ratio_mean"]) else 10.0
        collapse_penalty = max(0.0, 0.75 - recall_min_score)

        quality_passed = (
            recall_retention >= 0.90
            and recall_min_retention >= 0.75
            and ppl["ppl_finite"]
            and ppl["ppl_ratio_max"] <= 1.25
            and memory_ratio < 1.0
        )

        score = (
            1.50 * recall_score
            + 0.70 * memory_gain
            + 0.20 * throughput_ratio
            - 0.90 * ppl_excess
            - 0.80 * collapse_penalty
        )
        if not math.isfinite(memory_ratio) or not ppl["ppl_finite"] or recall_mean <= 0:
            score -= 10.0

        per_strategy.append(
            {
                "strategy": strategy,
                "score": float(score),
                "quality_passed": quality_passed,
                "recall_mean": float(recall_mean),
                "recall_min": float(recall_min),
                "recall_retention": float(recall_retention),
                "recall_min_retention": float(recall_min_retention),
                "memory_ratio": float(memory_ratio) if math.isfinite(memory_ratio) else None,
                "throughput_ratio": float(throughput_ratio),
                **ppl,
            }
        )

    per_strategy.sort(key=lambda row: row["score"], reverse=True)
    best: Optional[Dict[str, Any]] = per_strategy[0] if per_strategy else None
    ready = [row for row in per_strategy if row["quality_passed"]]
    correctness_passed = (
        baseline_recall_mean > 0
        and any(row["recall_mean"] > 0 for row in per_strategy)
        and any((row["memory_ratio"] or math.inf) < 1.0 for row in per_strategy)
        and any(row["ppl_finite"] for row in per_strategy)
    )

    return {
        "correctness_passed": correctness_passed,
        "h100_candidate_ready": bool(ready),
        "ready_strategy_count": len(ready),
        "score": float(best["score"]) if best else -10.0,
        "baseline_recall_mean": float(baseline_recall_mean),
        "baseline_recall_min": float(baseline_recall_min),
        "best_quantized_strategy": best["strategy"] if best else None,
        "best_quantized_recall_mean": float(best["recall_mean"]) if best else 0.0,
        "best_memory_ratio": best["memory_ratio"] if best else None,
        "best_strategy": best,
        "per_strategy": per_strategy,
    }
