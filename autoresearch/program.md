# Program: Tiered KV Cache Autoresearch

You are improving this repo's dynamic KV-cache quantization experiment.

## Objective

Compare the FP16 `DynamicCache` baseline against three dynamic age-based cache policies:

- `FixedWindow`
- `Ratio`
- `Hybrid`

Correctness comes first. A trial is not useful unless:

- the baseline has non-trivial recall,
- at least one quantized policy has non-zero recall,
- perplexity is finite,
- measured cache bytes for at least one quantized policy are below the baseline.

After those gates pass, improve the best tradeoff between recall retention, perplexity, memory ratio, and decode throughput.

## Fixed Command

Use this local single-trial command while checking a specific config:

```bash
python autoresearch/run_trial.py --profile local --quick
```

Use this multi-trial tuner for several hours:

```bash
CUDA_VISIBLE_DEVICES=0 python autoresearch/tune_policies.py --profile local --hours 3
```

Use the full local single-trial profile before proposing a RunPod run:

```bash
python autoresearch/run_trial.py --profile local
```

Use the H100 profile only after local results are sane:

```bash
python autoresearch/run_trial.py --profile runpod-h100
```

## Allowed Changes

You may edit:

- `tiered_kv/*.py`
- `experiments/*.py`
- `autoresearch/*.py`
- notebooks
- requirements and benchmark configs

You may add kernels or helper modules if they are covered by the same benchmark command.

## Do Not Do

- Do not change the OS or require a permanent machine-level setup.
- Do not make a result look good by weakening the baseline.
- Do not remove any of the three required policies from the comparison.
- Do not accept a change based only on memory if all quantized quality scores collapse to zero.

## Trial Review

Single trials write JSON under `autoresearch/runs/`. Multi-trial tuning runs write
under `autoresearch/tuning/`. Inspect:

- `correctness_passed`
- `h100_candidate_ready`
- `score`
- `baseline_recall_mean`
- `best_quantized_recall_mean`
- `best_memory_ratio`
- `best_quantized_strategy`
- `per_strategy`

Keep a change only if correctness still passes and score improves. If correctness fails, repair that before optimizing speed.
