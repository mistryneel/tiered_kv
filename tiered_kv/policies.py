"""Tier-assignment policies.

A *policy* tells the cache, given a current sequence length, how to partition the
positions [0, seq_len) into N tiers ordered from oldest (highest index in the
returned list) to newest (index 0). Newer tokens get higher precision.

By convention we use 4 tiers in this notebook:
    tier 0 = FP16 (newest)
    tier 1 = INT8
    tier 2 = INT4
    tier 3 = INT2 (oldest)

A policy returns a list of *capacities* `[c0, c1, c2, c3]` such that
    - the newest c0 tokens are tier 0,
    - the next c1 tokens (older) are tier 1,
    - the next c2 tokens are tier 2,
    - everything older fills tier 3 (capacity unbounded).

If `c0 + c1 + c2 < seq_len`, the remainder spills into tier 3.
If `c0 + c1 + c2 + c3 > seq_len`, tiers fill greedily from the newest end.

This makes the cache implementation simple: it only ever evicts oldest from
tier i into the head of tier i+1 when tier i exceeds its capacity.

We also support a 'sink' prefix: the first N_sink tokens never move out of FP16,
even when they become very old. This is the StreamingLLM trick — sink tokens
absorb a disproportionate share of attention mass and dropping them collapses
the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Default bits assigned to each tier.  Index 0 = newest, -1 = oldest.
DEFAULT_TIER_BITS: List[int] = [16, 8, 4, 2]


# ----------------------------------------------------------------------------

@dataclass
class PolicyDecision:
    """Output of a policy for one decoding step.

    Attributes:
        capacities: per-tier capacity for the *non-sink* portion.
            len(capacities) == num_tiers. The last tier is conventionally
            unbounded; we still return its capacity for accounting, but
            the cache treats it as a sink for spillover.
        n_sink: number of sink tokens (always FP16) at the *front* of the seq.
        seq_len: total tokens currently in the cache (for sanity checking).
    """
    capacities: List[int]
    n_sink: int
    seq_len: int

    def total_kept(self) -> int:
        return self.n_sink + sum(self.capacities)


class BasePolicy:
    """Abstract base.  A policy is just a callable: seq_len -> PolicyDecision."""

    num_tiers: int = 4

    def decide(self, seq_len: int) -> PolicyDecision:
        raise NotImplementedError

    # Convenience for tests / plotting: per-position tier id (0..N-1).
    def tier_per_position(self, seq_len: int) -> List[int]:
        d = self.decide(seq_len)
        tiers = [0] * seq_len
        # sinks: first n_sink positions are tier 0
        for i in range(min(d.n_sink, seq_len)):
            tiers[i] = 0
        # remaining positions: oldest first → walk from oldest end
        # capacities are listed newest-first (cap[0] = newest tier),
        # so to fill positions oldest-first we pour into tiers high→low.
        rem_start = d.n_sink
        rem = seq_len - rem_start
        if rem <= 0:
            return tiers
        # build a list of (tier_id, capacity) starting from oldest
        ordered_tiers_oldest_first: List[tuple[int, int]] = []
        for tier_id in reversed(range(len(d.capacities))):
            ordered_tiers_oldest_first.append((tier_id, d.capacities[tier_id]))
        # last tier = sink-bucket → unbounded; assign all remaining there if needed
        # walk from the OLDEST non-sink position forward
        cursor = rem_start
        oldest_to_newest_positions = list(range(rem_start, seq_len))
        # but since capacities are stated newest-first, we start with the OLDEST
        # tier and walk the OLDEST positions first.
        # However the oldest tier is "unbounded" → we treat it as taking the leftover.
        leftover = rem - sum(d.capacities[:-1])  # all but oldest tier
        leftover = max(0, leftover)
        # Fill oldest tier (last in the list) for `leftover` oldest positions
        oldest_tier_id = len(d.capacities) - 1
        for k in range(leftover):
            tiers[rem_start + k] = oldest_tier_id
        cursor = rem_start + leftover
        # Now fill the inner tiers from oldest-1 → tier 0 (newest), each with capacities[i]
        for tier_id in reversed(range(len(d.capacities) - 1)):
            cap = d.capacities[tier_id]
            for k in range(cap):
                if cursor >= seq_len:
                    break
                tiers[cursor] = tier_id
                cursor += 1
            if cursor >= seq_len:
                break
        return tiers


# ----------------------------------------------------------------------------
# 1. FixedWindow — absolute token-count thresholds.
# ----------------------------------------------------------------------------

@dataclass
class FixedWindowPolicy(BasePolicy):
    """Hard absolute thresholds for each tier.

    Example: capacities=(128, 1024, 8192) means
        - newest 128 tokens   -> tier 0 (FP16)
        - next 1024 tokens    -> tier 1 (INT8)
        - next 8192 tokens    -> tier 2 (INT4)
        - everything older    -> tier 3 (INT2)

    Strengths: deterministic memory profile per layer; easy to reason about.
    Weakness: short contexts under-use the FP16 budget; very long contexts
    spend an unbounded share at INT2.
    """
    fp16: int = 128
    int8: int = 1024
    int4: int = 8192
    n_sink: int = 4

    @property
    def num_tiers(self) -> int:
        return 4

    def decide(self, seq_len: int) -> PolicyDecision:
        # capacities listed newest-first; last tier ('int2') is unbounded → 0 placeholder.
        caps = [self.fp16, self.int8, self.int4, 0]
        # adjust for sinks
        return PolicyDecision(capacities=caps, n_sink=min(self.n_sink, seq_len), seq_len=seq_len)


# ----------------------------------------------------------------------------
# 2. Ratio — percentages of current seq_len.
# ----------------------------------------------------------------------------

@dataclass
class RatioPolicy(BasePolicy):
    """Each tier is a fixed fraction of the current sequence length.

    Example: ratios=(0.05, 0.15, 0.30) leaves 50% for tier 3 (INT2).

    Strengths: scales gracefully with context length; preserves the *recency
    fraction* of FP16 even at 100k tokens.
    Weakness: at short contexts the FP16 window can be smaller than is
    healthy (e.g. at 100 tokens, 5% = 5 FP16 — too few for sinks + recents).
    Mitigated by `min_fp16` clamp.
    """
    fp16_pct: float = 0.05
    int8_pct: float = 0.15
    int4_pct: float = 0.30
    n_sink: int = 4
    min_fp16: int = 32   # clamp so very short contexts still have a healthy FP16 window

    @property
    def num_tiers(self) -> int:
        return 4

    def decide(self, seq_len: int) -> PolicyDecision:
        sink = min(self.n_sink, seq_len)
        body = max(0, seq_len - sink)
        c0 = max(self.min_fp16, int(round(self.fp16_pct * seq_len)))
        c1 = int(round(self.int8_pct * seq_len))
        c2 = int(round(self.int4_pct * seq_len))
        # clamp so the three named tiers don't overflow body
        total_named = c0 + c1 + c2
        if total_named > body:
            scale = body / max(1, total_named)
            c0 = int(c0 * scale)
            c1 = int(c1 * scale)
            c2 = int(c2 * scale)
        caps = [c0, c1, c2, 0]
        return PolicyDecision(capacities=caps, n_sink=sink, seq_len=seq_len)


# ----------------------------------------------------------------------------
# 3. Hybrid — sinks + recent FP16 + tiered middle.
# ----------------------------------------------------------------------------

@dataclass
class HybridPolicy(BasePolicy):
    """Sink + recent-window-FP16 + (geometric or fixed) tiered middle.

    Two free parameters control the tier widths in the middle:
      - `geometric` (bool): if True, tier widths grow geometrically with `growth`
        factor (e.g. INT8 is 8x the FP16 window, INT4 is 8x INT8). This expresses
        the intuition that older = exponentially more compressible.
      - `growth` (float): geometric growth factor. Ignored if geometric=False.
        If geometric=False, falls back to FixedWindow-style absolute thresholds.

    Example (default): n_sink=4, fp16=64, growth=8.0 →
        - tier 0:    sinks (4) + newest 64 = FP16
        - tier 1:    next 64*8  = 512 INT8
        - tier 2:    next 512*8 = 4096 INT4
        - tier 3:    everything older → INT2
    """
    n_sink: int = 4
    fp16: int = 64
    geometric: bool = True
    growth: float = 8.0
    int8: int = 512   # used only if geometric=False
    int4: int = 4096  # used only if geometric=False

    @property
    def num_tiers(self) -> int:
        return 4

    def decide(self, seq_len: int) -> PolicyDecision:
        sink = min(self.n_sink, seq_len)
        if self.geometric:
            c0 = self.fp16
            c1 = int(round(c0 * self.growth))
            c2 = int(round(c1 * self.growth))
        else:
            c0 = self.fp16
            c1 = self.int8
            c2 = self.int4
        caps = [c0, c1, c2, 0]
        return PolicyDecision(capacities=caps, n_sink=sink, seq_len=seq_len)


# ----------------------------------------------------------------------------
# Diagnostic: average bits per token under a policy at given seq_len.
# ----------------------------------------------------------------------------

def avg_bits_per_token(policy: BasePolicy, seq_len: int,
                        tier_bits: List[int] = None) -> float:
    """Effective average bits per token across the whole cache."""
    tier_bits = tier_bits or DEFAULT_TIER_BITS
    d = policy.decide(seq_len)
    # sinks always FP16
    total_bits = d.n_sink * tier_bits[0]
    body = seq_len - d.n_sink
    # capacities are newest-first; oldest-tier holds spillover
    used = 0
    for tier_id in range(len(d.capacities) - 1):
        n = min(d.capacities[tier_id], max(0, body - used))
        total_bits += n * tier_bits[tier_id]
        used += n
    leftover = max(0, body - used)
    total_bits += leftover * tier_bits[-1]
    return total_bits / max(1, seq_len)
