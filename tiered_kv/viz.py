"""Plotting helpers for the tiered KV cache notebook.

All functions return the matplotlib `Figure` so the notebook can decide whether
to display, save, or both.

Two headline plots requested by the user:

    plot_quality_vs_length(...)   — accuracy / perplexity per cache strategy
    plot_kv_bytes_vs_length(...)  — KV cache bytes per cache strategy

Plus diagnostic plots:

    plot_tier_assignment(policy, seq_len)  — bar chart of which tier each position is in
    plot_niah_heatmap(niah_result)         — depth × length heatmap
    plot_throughput_bars(records)          — tok/s comparison
    plot_round_trip_error(...)             — quantization noise per tier (didactic)
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Matplotlib import is deferred to make this module importable in test environments
# where matplotlib isn't installed.
try:
    import matplotlib.pyplot as plt  # noqa: F401
    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False


# A consistent palette used across all plots so the same strategy has the same
# colour everywhere.
STRATEGY_COLORS = {
    "FP16 baseline": "#222222",
    "FixedWindow":   "#1f77b4",
    "Ratio":         "#ff7f0e",
    "Hybrid":        "#2ca02c",
    "Hybrid+Sinks":  "#9467bd",
    "INT8 uniform":  "#8c564b",
    "INT4 uniform":  "#e377c2",
    "INT2 uniform":  "#d62728",
}


def _need_mpl():
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for plotting; pip install matplotlib")
    import matplotlib.pyplot as plt
    return plt


# ----------------------------------------------------------------------------
# Quality vs context length.
# ----------------------------------------------------------------------------

def plot_quality_vs_length(
    records: Sequence[Dict[str, Any]],
    *,
    metric_label: str = "Accuracy",
    title: str = "Quality vs Context Length",
    log_x: bool = True,
    figsize: Tuple[float, float] = (8, 5),
):
    """Plot quality metric per strategy over a sweep of context lengths.

    Each `records[i]` should be a dict with:
        strategy: str   — name of the cache strategy
        lengths:  list of int
        scores:   list of float
    """
    plt = _need_mpl()
    fig, ax = plt.subplots(figsize=figsize)
    for r in records:
        color = STRATEGY_COLORS.get(r["strategy"])
        ax.plot(
            r["lengths"], r["scores"],
            marker="o", linewidth=2, color=color, label=r["strategy"],
        )
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# KV cache bytes vs context length.
# ----------------------------------------------------------------------------

def plot_kv_bytes_vs_length(
    records: Sequence[Dict[str, Any]],
    *,
    title: str = "KV Cache Bytes vs Context Length",
    log_x: bool = True,
    log_y: bool = True,
    figsize: Tuple[float, float] = (8, 5),
    show_ratio: bool = True,
):
    """Plot KV bytes per strategy on a single axes (or two stacked axes if show_ratio).

    Each `records[i]` should be a dict with:
        strategy: str
        lengths:  list of int
        bytes:    list of int
    The first record is treated as the FP16 baseline for ratio computation.
    """
    plt = _need_mpl()
    if show_ratio:
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 1.3),
                                        sharex=True)
    else:
        fig, ax0 = plt.subplots(figsize=figsize)
        ax1 = None

    baseline = None
    for r in records:
        color = STRATEGY_COLORS.get(r["strategy"])
        bytes_arr = np.array(r["bytes"], dtype=float)
        ax0.plot(
            r["lengths"], bytes_arr / 1e6,
            marker="o", linewidth=2, color=color, label=r["strategy"],
        )
        if baseline is None:
            baseline = (np.array(r["lengths"]), bytes_arr)
        if ax1 is not None and not np.array_equal(bytes_arr, baseline[1]):
            base_at = np.interp(r["lengths"], baseline[0], baseline[1])
            ax1.plot(
                r["lengths"], bytes_arr / base_at,
                marker="o", linewidth=2, color=color, label=r["strategy"],
            )

    ax0.set_ylabel("KV bytes (MB)")
    ax0.set_title(title)
    if log_x: ax0.set_xscale("log")
    if log_y: ax0.set_yscale("log")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best", frameon=True)

    if ax1 is not None:
        ax1.set_xlabel("Context length (tokens)")
        ax1.set_ylabel("Ratio to FP16 baseline")
        ax1.axhline(1.0, color="#222", linewidth=0.8, linestyle="--", alpha=0.5)
        if log_x: ax1.set_xscale("log")
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
    else:
        ax0.set_xlabel("Context length (tokens)")

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Tier-assignment bar — pedagogical figure for the design section.
# ----------------------------------------------------------------------------

def plot_tier_assignment(
    policies: Dict[str, Any],
    seq_len: int = 4096,
    *,
    figsize: Tuple[float, float] = (10, 1.0),
):
    """Stacked-bar visualization of which positions are in which tier.

    `policies` maps strategy name -> BasePolicy instance.
    Each row is one strategy. Position 0 is leftmost (oldest), position seq_len-1 rightmost (newest).
    """
    plt = _need_mpl()
    n = len(policies)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], max(figsize[1] * n, 2)))
    if n == 1:
        axes = [axes]
    tier_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]  # 0..3 (FP16 -> INT2)
    tier_labels = ["FP16", "INT8", "INT4", "INT2"]

    for ax, (name, policy) in zip(axes, policies.items()):
        tiers = policy.tier_per_position(seq_len)
        for t in range(4):
            mask = np.array(tiers) == t
            xs = np.where(mask)[0]
            if len(xs):
                ax.scatter(xs, np.zeros_like(xs), c=tier_colors[t], marker="|", s=50,
                           label=tier_labels[t] if t not in {tier_labels.index(x) for x in [] } else None)
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
        ax.set_xlim(0, seq_len)
        ax.set_xticks([])
        for spine in ("top", "right", "left", "bottom"):
            ax.spines[spine].set_visible(False)

    axes[0].set_title(f"Tier assignment per position (oldest → newest), seq_len={seq_len}")
    axes[-1].set_xticks([0, seq_len // 4, seq_len // 2, 3 * seq_len // 4, seq_len - 1])
    axes[-1].set_xlabel("position (oldest → newest)")
    # Custom legend
    handles = [
        plt.Line2D([0], [0], marker="|", linestyle="", color=tier_colors[t],
                   markersize=10, label=tier_labels[t])
        for t in range(4)
    ]
    fig.legend(handles=handles, loc="upper right", ncol=4, bbox_to_anchor=(1.0, 1.05))
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# NIAH heatmap.
# ----------------------------------------------------------------------------

def plot_niah_heatmap(niah_result, *, title: str = "Needle-in-a-Haystack",
                      figsize: Tuple[float, float] = (8, 5)):
    plt = _need_mpl()
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(niah_result.accuracy, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(niah_result.lengths)))
    ax.set_xticklabels([f"{L:,}" for L in niah_result.lengths])
    ax.set_yticks(range(len(niah_result.depths)))
    ax.set_yticklabels([f"{d}%" for d in niah_result.depths])
    ax.set_xlabel("Context length")
    ax.set_ylabel("Needle depth (% from start)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")
    # annotate cells
    for di in range(niah_result.accuracy.shape[0]):
        for li in range(niah_result.accuracy.shape[1]):
            v = niah_result.accuracy[di, li]
            ax.text(li, di, f"{v:.0%}", ha="center", va="center",
                    fontsize=8, color="black" if v > 0.5 else "white")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Round-trip error per tier — didactic plot for the math section.
# ----------------------------------------------------------------------------

def plot_round_trip_error(samples_per_bit: Dict[int, np.ndarray],
                          *, title: str = "Quantization noise vs bitwidth",
                          figsize: Tuple[float, float] = (8, 4)):
    """Histogram of (x_hat - x) per bit-width.

    `samples_per_bit` maps bits (16/8/4/2) to a 1D numpy array of element-wise errors.
    """
    plt = _need_mpl()
    fig, ax = plt.subplots(figsize=figsize)
    for bits, arr in samples_per_bit.items():
        ax.hist(arr, bins=80, alpha=0.55, label=f"{bits}-bit", density=True)
    ax.set_xlabel("Round-trip error (x̂ − x)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_throughput_bars(records: Sequence[Dict[str, Any]],
                          *, title: str = "Decode throughput (tok/s)",
                          figsize: Tuple[float, float] = (8, 4)):
    plt = _need_mpl()
    fig, ax = plt.subplots(figsize=figsize)
    names = [r["strategy"] for r in records]
    vals = [r["tokens_per_second"] for r in records]
    colors = [STRATEGY_COLORS.get(n, "#888") for n in names]
    ax.bar(names, vals, color=colors)
    ax.set_ylabel("tokens / sec")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    return fig
