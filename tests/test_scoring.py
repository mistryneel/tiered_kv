from autoresearch.scoring import summarize


def test_summary_scores_each_strategy_coherently():
    result = {
        "recall": [
            {"strategy": "FP16 baseline", "accuracy": [1.0, 1.0]},
            {"strategy": "A_high_quality", "accuracy": [1.0, 1.0]},
            {"strategy": "B_low_memory_bad_quality", "accuracy": [0.0, 1.0]},
        ],
        "perplexity": [
            {
                "strategy": "FP16 baseline",
                "rows": [
                    {"context_len": 1024, "perplexity": 10.0},
                    {"context_len": 2048, "perplexity": 10.0},
                ],
            },
            {
                "strategy": "A_high_quality",
                "rows": [
                    {"context_len": 1024, "perplexity": 10.1},
                    {"context_len": 2048, "perplexity": 10.2},
                ],
            },
            {
                "strategy": "B_low_memory_bad_quality",
                "rows": [
                    {"context_len": 1024, "perplexity": 10.1},
                    {"context_len": 2048, "perplexity": 40.0},
                ],
            },
        ],
        "memory": [
            {"strategy": "FP16 baseline", "cache_bytes_empirical": 1000},
            {"strategy": "A_high_quality", "cache_bytes_empirical": 600},
            {"strategy": "B_low_memory_bad_quality", "cache_bytes_empirical": 250},
        ],
        "throughput": [
            {"strategy": "FP16 baseline", "tokens_per_second": 20.0},
            {"strategy": "A_high_quality", "tokens_per_second": 15.0},
            {"strategy": "B_low_memory_bad_quality", "tokens_per_second": 12.0},
        ],
    }

    summary = summarize(result)

    assert summary["best_quantized_strategy"] == "A_high_quality"
    assert summary["best_memory_ratio"] == 0.6
    assert summary["h100_candidate_ready"] is True
