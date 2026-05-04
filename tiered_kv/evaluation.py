"""Evaluation harnesses for the tiered KV cache notebook.

All harnesses take a `model`, `tokenizer`, and a `cache_factory: Callable[[], Cache]`
that returns a fresh cache instance. The factory pattern lets us A/B compare a
DynamicCache baseline against various TieredKVCache configurations using the
exact same evaluation loop.

Three eval harnesses are provided:

    1. synthetic_kv_recall(...)     — key→value retrieval test, fast (~1 min/sweep)
    2. perplexity_pg19(...)         — sliding-window NLL on a PG-19 book
    3. niah_heatmap(...)            — needle-in-a-haystack at multiple depths/lengths

Plus helpers:

    measure_memory(model, cache, prompt_ids)
    measure_throughput(model, cache_factory, prompt_ids, n_new)
    kv_bytes_theoretical(num_layers, num_kv_heads, head_dim, seq_len, bits, group_size)

These are kept GPU-free where possible (the recall harness can run on CPU for
unit testing) and use `torch.cuda.synchronize()` + `torch.cuda.Event` where GPU
timing is needed.

Reference recipes synthesized from:
- HF perplexity docs: https://huggingface.co/docs/transformers/en/perplexity
- NVIDIA RULER: https://github.com/NVIDIA/RULER
- Greg Kamradt NIAH: https://github.com/gkamradt/LLMTest_NeedleInAHaystack
"""
from __future__ import annotations

import random
import string
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# All HF imports are deferred until the harnesses are actually called so this
# module loads on a torch-only environment for unit tests.


def _is_cuda_device(device: str) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()


# ----------------------------------------------------------------------------
# 1. Synthetic key→value recall.
# ----------------------------------------------------------------------------

@dataclass
class RecallResult:
    context_len: int
    n_pairs: int
    accuracy: float
    n_correct: int
    n_total: int
    sample_outputs: List[Tuple[str, str, str]] = field(default_factory=list)
    # (target_key, target_value, model_output) for first few examples


def _build_kv_prompt(
    num_pairs: int,
    target_padding_tokens: int,
    tokenizer,
    rng: random.Random,
) -> Tuple[str, str, str]:
    """Construct (prompt, target_key, target_value) for one recall example."""
    keys = [
        "".join(rng.choices(string.ascii_lowercase, k=8)) for _ in range(num_pairs)
    ]
    vals = [
        "".join(rng.choices(string.digits, k=6)) for _ in range(num_pairs)
    ]
    pairs_text = "".join(f"key {k} -> value {v}\n" for k, v in zip(keys, vals))

    target_idx = num_pairs // 2
    target_k, target_v = keys[target_idx], vals[target_idx]

    # Neutral filler — varied enough that tokenizer doesn't hit aggressive merges
    filler_unit = (
        "The quick brown fox jumps over the lazy dog. "
        "Numbers and letters arrange themselves in unexpected sequences. "
        "Sometimes the best path forward is the one least taken. "
    )
    body_tokens = len(tokenizer.encode(pairs_text))
    needed = max(0, target_padding_tokens - body_tokens)
    # Rough 4-chars-per-token estimate; we don't need exact length.
    pad_text = filler_unit * (needed // 12 + 1)

    prompt = (
        "Below are key-value pairs followed by neutral text. "
        "Then I will ask you for one value.\n\n"
        f"{pairs_text}\n{pad_text}\n\n"
        f"Question: what is the value for key {target_k}?\n"
        "Answer with only the value: "
    )
    return prompt, target_k, target_v


@torch.no_grad()
def synthetic_kv_recall(
    model,
    tokenizer,
    cache_factory: Callable[[], Any],
    *,
    context_lengths: Sequence[int] = (1024, 4096, 8192, 16384),
    num_pairs: int = 20,
    n_samples_per_length: int = 5,
    max_new_tokens: int = 12,
    seed: int = 0,
    device: str = "cuda",
    show_progress: bool = True,
) -> List[RecallResult]:
    """Sweep recall accuracy vs context length.

    For each `L` in `context_lengths`, inserts `num_pairs` (key, value) pairs into
    a context of approximately `L` tokens, then asks the model to retrieve the
    middle pair's value. Score = exact substring match.

    Returns a list of RecallResult, one per context length.
    """
    rng = random.Random(seed)
    results: List[RecallResult] = []

    for L in context_lengths:
        n_correct = 0
        sample_outputs: List[Tuple[str, str, str]] = []
        for s in range(n_samples_per_length):
            prompt, tk, tv = _build_kv_prompt(num_pairs, L, tokenizer, rng)
            ids = tokenizer(prompt, return_tensors="pt").to(device)
            cache = cache_factory()
            out = model.generate(
                **ids,
                past_key_values=cache,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen = tokenizer.decode(
                out[0, ids.input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()
            ok = tv in gen
            n_correct += int(ok)
            if len(sample_outputs) < 3:
                sample_outputs.append((tk, tv, gen))
            if show_progress:
                print(
                    f"  [L={L:>6d}] sample {s+1}/{n_samples_per_length}: "
                    f"target={tv!r}  got={gen[:40]!r}  ok={ok}"
                )

        results.append(RecallResult(
            context_len=L,
            n_pairs=num_pairs,
            accuracy=n_correct / n_samples_per_length,
            n_correct=n_correct,
            n_total=n_samples_per_length,
            sample_outputs=sample_outputs,
        ))
    return results


# ----------------------------------------------------------------------------
# 2. Perplexity (sliding-window NLL) on a long passage.
# ----------------------------------------------------------------------------

@dataclass
class PerplexityResult:
    context_len: int
    perplexity: float
    n_tokens_scored: int


@torch.no_grad()
def perplexity_sliding(
    model,
    tokenizer,
    cache_factory: Callable[[], Any],
    text: str,
    *,
    context_len: int,
    stride: Optional[int] = None,
    max_scored_tokens: Optional[int] = 50_000,
    device: str = "cuda",
    show_progress: bool = True,
) -> PerplexityResult:
    """Sliding-window NLL — exact recipe from the HF perplexity docs.

    Args:
        text: a single long string (e.g. a PG-19 book).
        context_len: window size L (the model sees this many tokens at once).
        stride: how far to advance per step. Default L // 2 (50% overlap).
        max_scored_tokens: cap on tokens scored, so long books don't take forever.

    Returns a PerplexityResult.
    """
    if stride is None:
        stride = context_len // 2

    enc = tokenizer(text, return_tensors="pt")
    seq_len = enc.input_ids.size(1)
    if max_scored_tokens is not None:
        seq_len = min(seq_len, max_scored_tokens + context_len)

    nll_sum = 0.0
    n_tokens_scored = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + context_len, seq_len)
        trg_len = end - prev_end
        input_ids = enc.input_ids[:, begin:end].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        # Each window starts fresh with a new cache instance.
        cache = cache_factory()
        out = model(input_ids, labels=target_ids, past_key_values=cache, use_cache=True)
        n_valid = (target_ids != -100).sum().item() - 1  # internal label shift
        if n_valid > 0:
            nll_sum += float(out.loss) * n_valid
            n_tokens_scored += n_valid
        prev_end = end
        if show_progress:
            ppl_so_far = float(np.exp(nll_sum / max(1, n_tokens_scored)))
            print(f"  window [{begin:>6d}:{end:>6d}]  ppl_so_far={ppl_so_far:.3f}")
        if end == seq_len:
            break

    return PerplexityResult(
        context_len=context_len,
        perplexity=float(np.exp(nll_sum / max(1, n_tokens_scored))),
        n_tokens_scored=n_tokens_scored,
    )


# ----------------------------------------------------------------------------
# 3. Needle-in-a-haystack heatmap.
# ----------------------------------------------------------------------------

@dataclass
class NIAHResult:
    lengths: List[int]
    depths: List[int]
    accuracy: np.ndarray  # shape (len(depths), len(lengths))
    sample_outputs: List[Dict[str, Any]]


_DEFAULT_NEEDLE = (
    "The best thing to do in San Francisco is to eat a sandwich and "
    "sit in Dolores Park on a sunny day."
)
_DEFAULT_NEEDLE_QUESTION = "What is the best thing to do in San Francisco?"
_DEFAULT_NEEDLE_KEYWORDS = ("sandwich", "Dolores Park")


@torch.no_grad()
def niah_heatmap(
    model,
    tokenizer,
    cache_factory: Callable[[], Any],
    haystack_text: str,
    *,
    lengths: Sequence[int] = (2_000, 8_000, 16_000, 32_000),
    depths: Sequence[int] = (10, 30, 50, 70, 90),
    needle: str = _DEFAULT_NEEDLE,
    question: str = _DEFAULT_NEEDLE_QUESTION,
    score_keywords: Sequence[str] = _DEFAULT_NEEDLE_KEYWORDS,
    max_new_tokens: int = 64,
    device: str = "cuda",
    show_progress: bool = True,
) -> NIAHResult:
    """Build a depth × length heatmap of needle retrieval accuracy.

    For each (length, depth_pct) pair, the haystack is truncated to roughly
    `length` tokens, the needle is inserted at `depth_pct%` of the way through,
    and the model is asked the question. Score = all keywords found in output.

    Returns an NIAHResult with shape (len(depths), len(lengths)).
    """
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    haystack_ids = tokenizer.encode(haystack_text)

    # Pre-compute haystack pool extension so we can pull any length we need.
    while len(haystack_ids) < max(lengths) + len(needle_ids) + 200:
        haystack_ids = haystack_ids + haystack_ids  # cheap repeat

    acc = np.zeros((len(depths), len(lengths)), dtype=float)
    sample_outputs: List[Dict[str, Any]] = []

    for li, L in enumerate(lengths):
        truncated = haystack_ids[:L]
        for di, depth_pct in enumerate(depths):
            insert_at = int(L * depth_pct / 100)
            ctx_ids = truncated[:insert_at] + needle_ids + truncated[insert_at:]
            ctx_text = tokenizer.decode(ctx_ids, skip_special_tokens=True)
            prompt = (
                f"{ctx_text}\n\n"
                f"Question: {question}\n"
                f"Answer:"
            )
            ids = tokenizer(prompt, return_tensors="pt").to(device)
            cache = cache_factory()
            out = model.generate(
                **ids,
                past_key_values=cache,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen = tokenizer.decode(
                out[0, ids.input_ids.shape[1]:],
                skip_special_tokens=True,
            ).lower()
            ok = all(kw.lower() in gen for kw in score_keywords)
            acc[di, li] = float(ok)
            if show_progress:
                print(f"  [L={L:>6d}, d={depth_pct:>3d}%] ok={ok}  out={gen[:60]!r}")
            if len(sample_outputs) < 5:
                sample_outputs.append(dict(L=L, depth=depth_pct, ok=bool(ok), output=gen[:200]))

    return NIAHResult(
        lengths=list(lengths),
        depths=list(depths),
        accuracy=acc,
        sample_outputs=sample_outputs,
    )


# ----------------------------------------------------------------------------
# 4. Memory + throughput.
# ----------------------------------------------------------------------------

@dataclass
class MemoryReport:
    cache_bytes_empirical: int
    cache_bytes_theoretical: int
    peak_alloc_bytes: int
    peak_reserved_bytes: int
    seq_len: int
    label: str = ""


def kv_bytes_theoretical(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    bits: float = 16,
    group_size: int = 32,
) -> int:
    """Theoretical KV cache size in bytes assuming uniform `bits` per element.

    For mixed precision, sum this call across the per-tier (n_tokens, bits)
    splits.
    """
    if bits == 16:
        bytes_per_elem = 2.0
    else:
        # codes + scale + zero (fp16 each, one per group)
        bytes_per_elem = bits / 8.0 + 2.0 * 2.0 / group_size
    return int(2 * num_layers * num_kv_heads * head_dim * seq_len * bytes_per_elem)


def _tensor_storage_bytes(x: Any) -> int:
    if torch.is_tensor(x):
        return x.element_size() * x.numel()
    return 0


def cache_storage_bytes(cache: Any) -> Optional[int]:
    """Best-effort actual KV-cache storage bytes for HF and TieredKV caches."""
    if hasattr(cache, "total_bytes"):
        return int(cache.total_bytes())

    total = 0
    found = False
    for attr in ("key_cache", "value_cache"):
        tensors = getattr(cache, attr, None)
        if tensors is None:
            continue
        for t in tensors:
            if torch.is_tensor(t):
                total += _tensor_storage_bytes(t)
                found = True
    if found:
        return total

    if hasattr(cache, "to_legacy_cache"):
        try:
            legacy = cache.to_legacy_cache()
            for layer in legacy:
                if isinstance(layer, (tuple, list)):
                    for t in layer:
                        total += _tensor_storage_bytes(t)
                        found = found or torch.is_tensor(t)
            if found:
                return total
        except Exception:
            return None

    return None


@torch.no_grad()
def measure_memory(
    model,
    cache,
    prompt_ids: torch.Tensor,
    *,
    label: str = "",
    device: str = "cuda",
) -> MemoryReport:
    """Run one prefill, return empirical + theoretical KV cache sizes."""
    if _is_cuda_device(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
    else:
        baseline = 0

    input_ids = prompt_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    _ = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
    )

    if _is_cuda_device(device):
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
    else:
        peak_alloc = 0
        peak_reserved = 0

    seq_len = prompt_ids.shape[1]
    cache_bytes_empirical = cache_storage_bytes(cache)
    if cache_bytes_empirical is None:
        cache_bytes_empirical = peak_alloc - baseline
    cfg = model.config
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    cache_bytes_theoretical = kv_bytes_theoretical(
        cfg.num_hidden_layers, num_kv_heads, head_dim, seq_len, bits=16,
    )
    return MemoryReport(
        cache_bytes_empirical=cache_bytes_empirical,
        cache_bytes_theoretical=cache_bytes_theoretical,
        peak_alloc_bytes=peak_alloc,
        peak_reserved_bytes=peak_reserved,
        seq_len=seq_len,
        label=label,
    )


@torch.no_grad()
def measure_throughput(
    model,
    cache_factory: Callable[[], Any],
    prompt_ids: torch.Tensor,
    *,
    n_new: int = 64,
    n_warmup: int = 1,
    n_iter: int = 3,
    device: str = "cuda",
) -> Dict[str, float]:
    """Median tok/s over `n_iter` warmed-up runs."""
    times = []

    # Warmup
    for _ in range(n_warmup):
        cache = cache_factory()
        input_ids = prompt_ids.to(device)
        attention_mask = torch.ones_like(input_ids, device=device)
        _ = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, past_key_values=cache,
            max_new_tokens=n_new, do_sample=False, use_cache=True,
            pad_token_id=getattr(model.generation_config, "pad_token_id", 0),
        )
    if _is_cuda_device(device):
        torch.cuda.synchronize()

    for _ in range(n_iter):
        cache = cache_factory()
        if _is_cuda_device(device):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        else:
            t0 = time.perf_counter()
        input_ids = prompt_ids.to(device)
        attention_mask = torch.ones_like(input_ids, device=device)
        out = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, past_key_values=cache,
            max_new_tokens=n_new, do_sample=False, use_cache=True,
            pad_token_id=getattr(model.generation_config, "pad_token_id", 0),
        )
        if _is_cuda_device(device):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        produced = out.shape[1] - prompt_ids.shape[1]
        times.append(t1 - t0)

    median = float(np.median(times))
    return {
        "tokens_per_second": produced / median,
        "median_seconds": median,
        "n_new_tokens": produced,
    }


# ----------------------------------------------------------------------------
# 5. Helpers for fetching haystack text without external dependencies.
# ----------------------------------------------------------------------------

_FALLBACK_HAYSTACK = (
    "Paul Graham wrote essays on startups. He observed that the best programmers "
    "tended to be the ones who liked programming for its own sake. Programming, "
    "in his view, is a kind of writing — it is a way to think out loud, to make "
    "your ideas concrete enough that a machine can follow them.\n\n"
    "He noted that good investors are often contrarian: they invest in companies "
    "that look strange to most people. The trick is being right about strangeness. "
    "Many strange companies fail. The ones that succeed redefine what was strange. "
    "This is the asymmetric upside that drives venture capital economics.\n\n"
    "On cities, Graham argued that great cities concentrate ambition. Each city "
    "specializes: New York rewards money, Silicon Valley rewards engineering, Paris "
    "rewards style. The city you live in shapes what you take seriously, because "
    "you absorb what the people around you take seriously. Graham wrote this in 2008 "
    "from a kitchen in Cambridge, Massachusetts, while drinking weak tea.\n\n"
    "Time and again, the lesson of his essays is that what matters is the long arc: "
    "ten years of compounding small advantages, ten years of focusing on what you "
    "actually find interesting, ten years of resisting the temptation to optimize "
    "for what looks impressive in the short run.\n\n"
)


def get_haystack_text(min_chars: int = 200_000) -> str:
    """Return ~min_chars of essay text for NIAH and perplexity tests.

    Tries to load `sgoel9/paul_graham_essays` from the HF datasets hub.
    Falls back to repeating a tiny embedded sample for offline testing.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("sgoel9/paul_graham_essays", split="train")
        out = []
        for row in ds:
            out.append(row["text"])
            if sum(len(s) for s in out) >= min_chars:
                break
        return "\n\n".join(out)[: max(min_chars, 1)]
    except Exception:
        pass
    # Fallback for offline / no-dataset environments
    text = _FALLBACK_HAYSTACK
    while len(text) < min_chars:
        text = text + "\n\n" + _FALLBACK_HAYSTACK
    return text


def get_pg19_book(min_chars: int = 200_000) -> str:
    """Return ~min_chars of PG-19 text for perplexity tests."""
    try:
        from datasets import load_dataset
        ds = load_dataset("deepmind/pg19", split="test", streaming=True)
        for row in ds:
            t = row["text"]
            if len(t) >= min_chars:
                return t
        # Concatenate if no single book is long enough
        out = ""
        ds = load_dataset("deepmind/pg19", split="test", streaming=True)
        for row in ds:
            out += "\n\n" + row["text"]
            if len(out) >= min_chars:
                return out
    except Exception:
        pass
    return get_haystack_text(min_chars)
