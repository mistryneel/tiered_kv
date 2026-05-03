# Tiered KV Cache — Age-Based Dynamic Quantization

A self-contained, educational implementation of an **age-tiered KV cache** for transformer LLM
inference: the *newest* tokens are kept at FP16, *old* tokens are stored at INT8, *very old*
at INT4, and *very very old* at INT2. The intuition is that recent tokens dominate attention
output, so older tokens can carry quantization noise without much quality loss — and the
memory savings let you push context length far beyond what a vanilla FP16 cache allows.

This project is intended for readers who are **beginner-to-intermediate** with LLM internals
and inference. The notebook walks through the math, the design space, and an honest ablation
study on a single H100 GPU.

## What's in here

```
tiered_kv/
├── tiered_kv_cache.ipynb       # the notebook — start here
├── tiered_kv/                 # helper package, imported from the notebook
│   ├── quantization.py        # asymmetric uniform quant + bit-packing
│   ├── policies.py            # FixedWindow / Ratio / Hybrid tier policies
│   ├── cache.py               # TieredKVCache (DynamicCache subclass)
│   ├── evaluation.py          # synthetic recall, perplexity, NIAH harness
│   └── viz.py                 # plotting helpers
├── requirements.txt
└── README.md
```

## What it produces

Two headline plots:
1. **Quality vs context length** — accuracy / perplexity of vanilla FP16 vs each tiered policy
   as the context grows. Shows when the recency assumption breaks.
2. **KV cache bytes vs context length** — the actual storage cost (with bit-packed INT2/INT4)
   compared to the FP16 baseline. Shows how far you can push context on the same GPU.

Plus ablations across tier policies, group sizes, and the sink-token trick.

## Quickstart on RunPod (one H100)

1. **Spin up a pod** with the `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
   image (or any CUDA 12.x + Python 3.10/3.11 image with PyTorch 2.3+). 80 GB VRAM is
   enough for everything in this notebook.

2. **Clone and install:**
   ```bash
   git clone <your-fork-or-zip> tiered_kv && cd tiered_kv
   pip install -r requirements.txt
   ```

3. **Set your HuggingFace token** (only needed if you want to use a gated model like
   Llama-3.1-8B; the default Qwen2.5-7B-Instruct is ungated):
   ```bash
   export HF_TOKEN=hf_xxx
   huggingface-cli login --token $HF_TOKEN
   ```

4. **Launch JupyterLab:**
   ```bash
   jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
   ```
   Open `tiered_kv_cache.ipynb` and run top to bottom.

## Time budget on one H100

| Section | Time |
| --- | --- |
| Quantization primitives + cache unit tests | <1 min (CPU-fine) |
| Model load (Qwen2.5-7B-Instruct, bf16) | 1–2 min |
| Synthetic key→value recall sweep | 3–5 min |
| PG-19 perplexity at 4 context lengths × 3 caches | 10–15 min |
| Needle-in-a-Haystack lite heatmap | 10–12 min |
| Memory + throughput measurement | 2–3 min |
| **Total** | **~30–40 min** |

## Model choice

The notebook defaults to **`Qwen/Qwen2.5-7B-Instruct`** because it's ungated and uses
extreme GQA (4 KV heads), which makes every byte of KV-cache savings meaningful. Set
`MODEL_ID` at the top of the notebook to swap to:

- `meta-llama/Llama-3.1-8B-Instruct` — most-cited reference model in KV-quant literature
  (gated, requires `HF_TOKEN` and Meta access approval).
- `meta-llama/Llama-3.2-3B-Instruct` — fast iteration, smaller weights (gated).
- `Qwen/Qwen2.5-3B-Instruct` — fastest ungated option for ablations.

## How this relates to prior work

This is **not** a new research paper. It's an educational reproduction and extension of
several published ideas:

- **KIVI** (Liu et al., ICML 2024) — binary scheme: newest *R* tokens at FP16, everything
  else at INT2 with per-channel-K / per-token-V grouping. Our INT2 tier follows KIVI's
  grouping convention exactly.
- **MiKV** (Yang et al., ICML 2024) — mixed precision driven by attention-score importance.
  We replace their importance signal with token age (a free, prefix-deterministic proxy).
- **StreamingLLM** (Xiao et al., ICLR 2024) — first *K* "sink" tokens carry disproportionate
  attention; keeping them at FP16 prevents collapse. Our Hybrid policy adopts this trick.
- **TTKV** (2026) — proposes 2-tier age-based KV. We extend to N tiers and compare policies.
- **KVQuant** (Hooper et al., NeurIPS 2024) — pre-RoPE quantization and dense-and-sparse
  decomposition. We discuss but do not implement; noted as future work.

We cite all of these in the notebook's references section.

## Caveats and honest threats

- **Post-RoPE storage**: HuggingFace's `Cache.update()` is called *after* RoPE has been
  applied to keys, so we quantize post-RoPE. This is the simpler path but loses some
  quality vs pre-RoPE quantization (KVQuant's approach).
- **No CUDA kernels**: quantize/dequantize is implemented in plain PyTorch for clarity.
  Throughput numbers are therefore not deployment-grade. Memory numbers, however, are
  honest because we genuinely bit-pack INT2/INT4 into `uint8`.
- **Single-GPU, single-batch**: the notebook focuses on quality and memory; concurrency and
  batched serving are out of scope.
- **The claim "old tokens don't matter" has limits**: the ablation will show the breaking
  point where the INT2 tier hurts retrieval. That's the *point* of the study.

## License

MIT. Use it, fork it, write a paper extending it.
# tiered_kv
