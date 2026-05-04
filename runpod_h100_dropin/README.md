# RunPod H100 Drop-In

Copy this folder and the `tiered_kv/` helper package to the same parent folder on RunPod:

```text
/workspace/tiered_kv_run/
  runpod_h100_dropin/
  tiered_kv/
```

The drop-in folder contains the H100 notebook, CLI runner, tuned policy config, and RunPod-specific requirements. It does not include `tiered_kv/`.

## Setup

From the parent folder:

```bash
cd runpod_h100_dropin
bash setup_runpod.sh
```

The setup intentionally does not install PyTorch. The RunPod image already provides CUDA-enabled PyTorch.

## Notebook

Launch JupyterLab from the parent folder or this folder:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

Open:

```text
runpod_h100_dropin/tiered_kv_h100_runpod.ipynb
```

Run the safe profile first. It tests up to 16K perplexity and avoids the 32K full-logit OOM seen in the previous run.

## CLI

Safe H100 run:

```bash
CUDA_VISIBLE_DEVICES=0 python run_h100_benchmark.py --profile h100-safe
```

Full H100 run with 32K perplexity:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_h100_benchmark.py --profile h100-full
```

Optional NIAH run:

```bash
CUDA_VISIBLE_DEVICES=0 python run_h100_benchmark.py --profile h100-safe --run-niah \
  --skip-recall --skip-perplexity --skip-memory \
  --output-dir results/runpod_h100_niah
```

Summarize latest result:

```bash
python summarize_results.py
```

## Included Tuned Config

`best_policy_config.json` is the best local autoresearch config from `random_0061`.

It made all three policies pass locally. H100 may still need additional tuning, but this prevents accidentally running the untuned default policy again.
