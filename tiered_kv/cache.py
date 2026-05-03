"""TieredKVCache — a `DynamicCache` subclass with age-based mixed-precision storage.

Storage layout (per layer, K and V independent):

    [ FP16 sinks ] [ INT2 oldest body ] [ INT4 ] [ INT8 ] [ FP16 newest body ]
       (front)                                                  (back, recent)

The four "body" tiers cascade as the cache fills:
    1. New tokens arrive into FP16 newest_body.
    2. When newest_body exceeds its policy capacity, the OLDEST tokens overflow
       to INT8 (head of the INT8 buffer).
    3. INT8 overflow → INT4. INT4 overflow → INT2 (the unbounded oldest tier).

Sinks are a separate FP16 buffer of fixed size (StreamingLLM trick) at the
absolute FRONT of the sequence, never demoted.

API contract (matches HF transformers DynamicCache subclass pattern):
    update(key_states, value_states, layer_idx, cache_kwargs) -> (K_full, V_full)
    get_seq_length(layer_idx=0) -> int

The returned K_full and V_full are dense tensors in compute_dtype, ordered as
[sinks, tier3, tier2, tier1, tier0_fp16]. They are the inputs to attention.

Threats acknowledged in the notebook:
- We quantize POST-RoPE (HF's update is called after rotary embedding has been
  applied to keys). KVQuant shows pre-RoPE is better for long-context retrieval.
- The cascade re-quantizes tokens as they age (INT8→INT4→INT2), each step
  losing more information. During prefill, we partition directly from FP16 to
  the target tier, so the cascade only matters during long decode.
- No CUDA kernel — quantize/dequantize is plain PyTorch. Throughput numbers
  are not deployment-grade. Memory accounting IS honest because we bit-pack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

# Import lazily so this module is usable without transformers installed
# for unit tests.
try:
    from transformers.cache_utils import DynamicCache
    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover
    DynamicCache = object  # type: ignore[misc,assignment]
    _HAS_TRANSFORMERS = False

from tiered_kv import quantization as Q
from tiered_kv.policies import BasePolicy, PolicyDecision, DEFAULT_TIER_BITS


# ----------------------------------------------------------------------------

@dataclass
class TierConfig:
    """Configuration shared across all layers and the K/V tensors.

    Attributes:
        tier_bits: bits per tier (newest first).  Default [16, 8, 4, 2].
        group_size: per-tier group size for asymmetric quantization.
        axis_key: GROUPING axis for K. -2 = group along token-dim, which means
            *each channel keeps its own (per-token-window) scale*. This is the
            correct "per-channel-K" KIVI convention — outlier channels don't
            poison normal channels.
        axis_value: GROUPING axis for V. -1 = group along channel-dim, which
            means *each token keeps its own (per-channel-window) scale*. This
            is the correct "per-token-V" KIVI convention — error in one token
            does not bleed into other tokens through the attention-weighted sum.
        compute_dtype: dtype attention reads (matches model dtype).

    Note on terminology: the AXIS we GROUP ALONG is the axis that gets chunked
    into groups of `group_size`. Scales then vary along the OTHER axis. So
    "group along token" → scales vary per channel ("per-channel"). Easy to
    flip — that exact mistake produces ~40% noise on normal channels at INT4.
    """
    tier_bits: List[int] = field(default_factory=lambda: list(DEFAULT_TIER_BITS))
    group_size: int = 32
    axis_key: int = -2     # group along token → scales per channel (per-channel-K)
    axis_value: int = -1   # group along channel → scales per token (per-token-V)
    compute_dtype: torch.dtype = torch.float16

    @property
    def num_tiers(self) -> int:
        return len(self.tier_bits)


# ----------------------------------------------------------------------------

class TieredKVCache(DynamicCache):
    """KV cache that quantizes older tokens to lower bit-widths.

    Usage:
        cache = TieredKVCache(policy=HybridPolicy(), config=TierConfig())
        out = model.generate(**inputs, past_key_values=cache, max_new_tokens=200)
    """

    # transformers>=4.50 checks `cache.is_compileable` in generate() to decide
    # whether to torch.compile the forward. We resize buffers and create new
    # QuantTensors on every update — torch.compile would re-trace constantly,
    # so we explicitly opt out. (Inherited DynamicCache attribute may be missing
    # depending on version, so we set it explicitly here.)
    is_compileable: bool = False

    @property
    def is_sliding(self) -> List[bool]:
        """transformers>=4.50 mask code does `False in cache.is_sliding` and
        `cache.is_sliding.index(False)` — i.e. it expects a per-layer list, not
        a bool. Every layer of our cache does full attention (no sliding window),
        so we return [False] per known layer (or [False] as fallback when no
        layer has been touched yet)."""
        n = max(1, len(self._lengths))
        return [False] * n

    def __init__(self, policy: BasePolicy, config: Optional[TierConfig] = None):
        if _HAS_TRANSFORMERS:
            super().__init__()
        self.policy = policy
        self.config = config or TierConfig()

        # Per-layer storage. We grow lists lazily as new layer_idx arrive.
        # newest_k[i]:    fp16 tensor [B, H, T_fp16, D]      (the recent buffer)
        # tier_k[i][t]:   QuantTensor for tier t (1=int8, 2=int4, 3=int2)
        # sink_k[i]:      fp16 tensor [B, H, T_sink, D]
        self._newest_k: List[Optional[torch.Tensor]] = []
        self._newest_v: List[Optional[torch.Tensor]] = []
        self._tier_k: List[List[Optional[Q.QuantTensor]]] = []
        self._tier_v: List[List[Optional[Q.QuantTensor]]] = []
        self._sink_k: List[Optional[torch.Tensor]] = []
        self._sink_v: List[Optional[torch.Tensor]] = []
        self._lengths: List[int] = []  # total tokens stored per layer

        self._seen_tokens: int = 0     # for compatibility with HF generate()

    # ---------------- HF Cache interface ----------------

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._lengths):
            return 0
        return self._lengths[layer_idx]

    def get_max_cache_shape(self) -> Optional[int]:
        return None  # dynamic, unbounded

    def get_max_length(self) -> Optional[int]:  # legacy alias
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        # Beam search not supported — fail loudly rather than silently corrupt.
        raise NotImplementedError(
            "TieredKVCache does not support beam search reordering. "
            "Use greedy or sampling decoding."
        )

    # ---------------- core update / read ----------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append (key_states, value_states), then return the FULL K, V for attention.

        Shapes:
            key_states / value_states: [B, num_kv_heads, T_new, D]
            returns: ([B, num_kv_heads, T_total, D], same)
        """
        # Lazy-grow per-layer lists.
        while len(self._newest_k) <= layer_idx:
            self._newest_k.append(None)
            self._newest_v.append(None)
            self._tier_k.append([None] * self.config.num_tiers)
            self._tier_v.append([None] * self.config.num_tiers)
            self._sink_k.append(None)
            self._sink_v.append(None)
            self._lengths.append(0)

        # Bookkeeping for HF generate(): only count new tokens once (on layer 0).
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        self._lengths[layer_idx] += key_states.shape[-2]

        is_first_call = self._sink_k[layer_idx] is None and self._newest_k[layer_idx] is None

        if is_first_call:
            # PREFILL: assign each token directly to its target tier in one shot.
            # This avoids the lossy cascade re-quantization.
            self._prefill_partition(key_states, value_states, layer_idx)
        else:
            # DECODE: append to the FP16 newest buffer, then cascade.
            self._newest_k[layer_idx] = self._cat_seq(self._newest_k[layer_idx], key_states)
            self._newest_v[layer_idx] = self._cat_seq(self._newest_v[layer_idx], value_states)
            self._cascade(layer_idx)

        return self._read_full(layer_idx)

    # ---------------- internal helpers ----------------

    @staticmethod
    def _cat_seq(buf: Optional[torch.Tensor], new: torch.Tensor) -> torch.Tensor:
        if buf is None:
            return new
        return torch.cat([buf, new], dim=-2)

    @staticmethod
    def _seq_len(t: Optional[torch.Tensor]) -> int:
        return 0 if t is None else t.shape[-2]

    @staticmethod
    def _seq_len_qt(qt: Optional[Q.QuantTensor]) -> int:
        if qt is None:
            return 0
        # The token dimension in K/V is always axis -2 in original shape.
        return qt.orig_shape[-2]

    def _prefill_partition(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """Direct, lossless-by-tier partition during prefill.

        Splits the full prompt's K/V into [sinks, oldest_body→tier3, …, newest_body→tier0]
        and quantizes each piece at its target bitwidth in one shot.
        """
        seq_len = K.shape[-2]
        decision = self.policy.decide(seq_len)

        n_sink = decision.n_sink
        body = seq_len - n_sink

        # 1. sinks (fp16, kept as-is)
        if n_sink > 0:
            self._sink_k[layer_idx] = K[..., :n_sink, :].contiguous()
            self._sink_v[layer_idx] = V[..., :n_sink, :].contiguous()

        # 2. body: oldest first into highest tier (int2), capacities fill from oldest end.
        #    Layout in source: positions [n_sink, seq_len)
        #    Newest tier (0) takes the LAST cap[0] body tokens.
        #    Then tier 1 takes the cap[1] before that, etc.
        #    Whatever remains at the very oldest end goes to tier N-1.
        cap = decision.capacities  # newest-first; cap[-1] is unbounded placeholder
        cur_end = seq_len  # exclusive
        # Fill newest tiers first by walking cur_end backward.
        for tier_id in range(len(cap) - 1):
            n = min(cap[tier_id], cur_end - n_sink)
            if n <= 0:
                continue
            slice_start = cur_end - n
            K_slice = K[..., slice_start:cur_end, :]
            V_slice = V[..., slice_start:cur_end, :]
            self._store_tier(layer_idx, tier_id, K_slice, V_slice)
            cur_end = slice_start

        # Whatever remains [n_sink, cur_end) goes to the oldest tier.
        if cur_end > n_sink:
            K_slice = K[..., n_sink:cur_end, :]
            V_slice = V[..., n_sink:cur_end, :]
            self._store_tier(layer_idx, len(cap) - 1, K_slice, V_slice)

    def _store_tier(
        self,
        layer_idx: int,
        tier_id: int,
        K_slice: torch.Tensor,
        V_slice: torch.Tensor,
    ) -> None:
        """Quantize a contiguous K/V slice and store into tier `tier_id` (replacing existing)."""
        bits = self.config.tier_bits[tier_id]
        if tier_id == 0:
            # tier 0 = FP16 newest body
            self._newest_k[layer_idx] = K_slice.contiguous()
            self._newest_v[layer_idx] = V_slice.contiguous()
            return

        K_q = Q.quantize(K_slice, bits=bits, group_axis=self.config.axis_key,
                         group_size=self.config.group_size)
        V_q = Q.quantize(V_slice, bits=bits, group_axis=self.config.axis_value,
                         group_size=self.config.group_size)
        self._tier_k[layer_idx][tier_id] = K_q
        self._tier_v[layer_idx][tier_id] = V_q

    def _cascade(self, layer_idx: int) -> None:
        """Push overflow from each tier into the next-coarser tier.

        Walks tier 0 → 1 → 2 → 3. At each step, if tier_i has more tokens than
        its policy capacity, take the OLDEST overflow from tier_i, dequantize,
        re-quantize at tier_{i+1}'s bitwidth, and PREPEND to tier_{i+1}.
        The final (oldest) tier is unbounded.
        """
        seq_len = self._lengths[layer_idx]
        decision = self.policy.decide(seq_len)
        cap = decision.capacities

        # Tier 0 → 1: overflow from FP16 newest_body to INT8.
        # Tier 0 storage is fp16 tensor in self._newest_k.
        for src in range(len(cap) - 1):
            cap_src = cap[src]
            cur_n_src = self._tier_seq_len(layer_idx, src)
            if cur_n_src <= cap_src:
                continue
            n_overflow = cur_n_src - cap_src
            K_evict, V_evict = self._evict_oldest(layer_idx, src, n_overflow)
            self._prepend_to_tier(layer_idx, src + 1, K_evict, V_evict)

    def _tier_seq_len(self, layer_idx: int, tier_id: int) -> int:
        if tier_id == 0:
            return self._seq_len(self._newest_k[layer_idx])
        return self._seq_len_qt(self._tier_k[layer_idx][tier_id])

    def _evict_oldest(
        self,
        layer_idx: int,
        tier_id: int,
        n_evict: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remove the oldest n_evict tokens from tier_id; return them as fp16 K, V tensors."""
        if tier_id == 0:
            buf_k = self._newest_k[layer_idx]
            buf_v = self._newest_v[layer_idx]
            K_evict = buf_k[..., :n_evict, :].contiguous()
            V_evict = buf_v[..., :n_evict, :].contiguous()
            self._newest_k[layer_idx] = buf_k[..., n_evict:, :].contiguous()
            self._newest_v[layer_idx] = buf_v[..., n_evict:, :].contiguous()
            return K_evict, V_evict

        # tier > 0: dequant the entire tier, slice off oldest, re-quantize remainder.
        K_full = Q.dequantize(self._tier_k[layer_idx][tier_id])
        V_full = Q.dequantize(self._tier_v[layer_idx][tier_id])
        K_evict = K_full[..., :n_evict, :].contiguous()
        V_evict = V_full[..., :n_evict, :].contiguous()
        K_keep = K_full[..., n_evict:, :].contiguous()
        V_keep = V_full[..., n_evict:, :].contiguous()
        if K_keep.shape[-2] == 0:
            self._tier_k[layer_idx][tier_id] = None
            self._tier_v[layer_idx][tier_id] = None
        else:
            bits = self.config.tier_bits[tier_id]
            self._tier_k[layer_idx][tier_id] = Q.quantize(
                K_keep, bits=bits, group_axis=self.config.axis_key,
                group_size=self.config.group_size,
            )
            self._tier_v[layer_idx][tier_id] = Q.quantize(
                V_keep, bits=bits, group_axis=self.config.axis_value,
                group_size=self.config.group_size,
            )
        return K_evict, V_evict

    def _prepend_to_tier(
        self,
        layer_idx: int,
        tier_id: int,
        K_new: torch.Tensor,
        V_new: torch.Tensor,
    ) -> None:
        """Prepend (older) K_new, V_new to tier_id's storage, re-quantizing as needed."""
        bits = self.config.tier_bits[tier_id]
        if tier_id == 0:
            # FP16 prepend
            existing_k = self._newest_k[layer_idx]
            existing_v = self._newest_v[layer_idx]
            self._newest_k[layer_idx] = (
                K_new if existing_k is None else torch.cat([K_new, existing_k], dim=-2)
            )
            self._newest_v[layer_idx] = (
                V_new if existing_v is None else torch.cat([V_new, existing_v], dim=-2)
            )
            return

        # Quantize K_new/V_new at tier's bitwidth, then merge with existing.
        # Easy correct path: dequant existing, concat fp16, re-quant the whole tier.
        existing_k_q = self._tier_k[layer_idx][tier_id]
        existing_v_q = self._tier_v[layer_idx][tier_id]
        K_full = K_new if existing_k_q is None else torch.cat(
            [K_new, Q.dequantize(existing_k_q)], dim=-2,
        )
        V_full = V_new if existing_v_q is None else torch.cat(
            [V_new, Q.dequantize(existing_v_q)], dim=-2,
        )
        self._tier_k[layer_idx][tier_id] = Q.quantize(
            K_full, bits=bits, group_axis=self.config.axis_key,
            group_size=self.config.group_size,
        )
        self._tier_v[layer_idx][tier_id] = Q.quantize(
            V_full, bits=bits, group_axis=self.config.axis_value,
            group_size=self.config.group_size,
        )

    def _read_full(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the full K, V tensor for attention by concatenating all tiers in
        oldest→newest order: [sinks, tier3, tier2, tier1, tier0_fp16]."""
        parts_k: List[torch.Tensor] = []
        parts_v: List[torch.Tensor] = []
        if self._sink_k[layer_idx] is not None:
            parts_k.append(self._sink_k[layer_idx])
            parts_v.append(self._sink_v[layer_idx])
        # tiers oldest first
        for tier_id in reversed(range(1, self.config.num_tiers)):
            qk = self._tier_k[layer_idx][tier_id]
            qv = self._tier_v[layer_idx][tier_id]
            if qk is not None:
                parts_k.append(Q.dequantize(qk))
                parts_v.append(Q.dequantize(qv))
        # tier 0 newest fp16
        if self._newest_k[layer_idx] is not None:
            parts_k.append(self._newest_k[layer_idx])
            parts_v.append(self._newest_v[layer_idx])

        if not parts_k:
            # Empty layer (shouldn't happen post-update). Return zero-length tensors
            # with the right hidden dims, derived from any other layer if needed.
            raise RuntimeError(f"TieredKVCache layer {layer_idx} has no data after update")

        K_full = torch.cat(parts_k, dim=-2).to(self.config.compute_dtype)
        V_full = torch.cat(parts_v, dim=-2).to(self.config.compute_dtype)
        return K_full, V_full

    # ---------------- memory accounting ----------------

    def total_bytes(self, layer_idx: Optional[int] = None) -> int:
        """Total bytes occupied by the cache (codes + scales + zeros + fp16 buffers).

        If layer_idx is None, sum across all layers.
        """
        layer_iter = range(len(self._lengths)) if layer_idx is None else [layer_idx]
        total = 0
        for li in layer_iter:
            for buf in (self._sink_k[li], self._sink_v[li],
                        self._newest_k[li], self._newest_v[li]):
                if buf is not None:
                    total += buf.element_size() * buf.numel()
            for qt in self._tier_k[li] + self._tier_v[li]:
                if qt is not None:
                    total += qt.nbytes_packed()
        return total

    def avg_bits_per_token(self, layer_idx: int = 0) -> float:
        """Empirical average effective bits per token in this layer."""
        n_tokens = self.get_seq_length(layer_idx)
        if n_tokens == 0:
            return 0.0
        bytes_used = self.total_bytes(layer_idx)
        # Subtract: cache holds K and V each of shape [B, H, T, D]
        # so we divide bytes by (2 * B * H * D) to get bytes per token, then *8.
        # We need shape info from any non-empty buffer.
        sample = (
            self._sink_k[layer_idx] if self._sink_k[layer_idx] is not None
            else self._newest_k[layer_idx]
        )
        if sample is None:
            return 0.0
        B, H, _, D = sample.shape
        bytes_per_token = bytes_used / (2 * B * H * D * n_tokens)
        return bytes_per_token * 8

    # ---------------- HF-compat extras ----------------

    @property
    def seen_tokens(self) -> int:  # legacy attr accessed by some HF code paths
        return self._seen_tokens

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """Materialize as legacy-format ((K, V), ...) tuple. Used for compatibility."""
        out = []
        for li in range(len(self._lengths)):
            K, V = self._read_full(li)
            out.append((K, V))
        return tuple(out)

    @classmethod
    def from_legacy_cache(cls, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(
            "from_legacy_cache is not supported for TieredKVCache. "
            "Construct an empty cache and run the prompt through `model.generate()`."
        )

    def __len__(self) -> int:
        return len(self._lengths)


# ----------------------------------------------------------------------------
# Convenience builder for ablation sweeps in the notebook.
# ----------------------------------------------------------------------------

def build_cache(
    policy: BasePolicy,
    *,
    tier_bits: Optional[List[int]] = None,
    group_size: int = 32,
    compute_dtype: torch.dtype = torch.float16,
) -> TieredKVCache:
    """Construct a fresh TieredKVCache with the given policy.

    Pass `tier_bits=[16, 16, 16, 16]` to force FP16 baseline (useful for A/B
    comparisons that share the cascade path).
    """
    cfg = TierConfig(
        tier_bits=tier_bits or list(DEFAULT_TIER_BITS),
        group_size=group_size,
        compute_dtype=compute_dtype,
    )
    return TieredKVCache(policy=policy, config=cfg)
