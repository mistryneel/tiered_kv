# Tiered KV Autoresearch

This folder adapts the Karpathy `autoresearch` loop to this repo:

1. Keep the benchmark command fixed.
2. Let an agent modify cache code, policies, configs, or kernels.
3. Run a short trial.
4. Keep only changes that improve the measured score without breaking correctness.

Run one reproducible trial with:

```bash
python autoresearch/run_trial.py --profile local --quick
```

Run a multi-hour policy search with:

```bash
CUDA_VISIBLE_DEVICES=0 python autoresearch/tune_policies.py --profile local --hours 3
```

For overnight:

```bash
CUDA_VISIBLE_DEVICES=0 python autoresearch/tune_policies.py --profile local --hours 8
```

The tuner writes `progress.jsonl`, `leaderboard.json`, and `best.json` under
`autoresearch/tuning/<run-id>/`.

If a terminal or SSH session dies, resume the same deterministic search from the
next unfinished trial:

```bash
CUDA_VISIBLE_DEVICES=0 python autoresearch/tune_policies.py --profile local --hours 8 --resume-dir autoresearch/tuning/<run-id>
```

Once local tuning is stable, use:

```bash
python autoresearch/run_trial.py --profile runpod-h100
```

The agent-facing instructions live in `program.md`.
