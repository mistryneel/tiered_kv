"""Asymmetric uniform quantization with bit-packing.

Math (asymmetric, group-wise):

    For a group of values x = (x_1, ..., x_G):
        x_min  = min(x)
        x_max  = max(x)
        scale   = (x_max - x_min) / (2^b - 1)       # b = bits {2,4,8}
        offset  = x_min
        q_i     = clip(round((x_i - offset) / scale), 0, 2^b - 1)
        x_hat_i = q_i * scale + offset

The 'group' is the slice that shares one (scale, zero) pair. The axis we
GROUP ALONG is the axis that gets chunked; scales then vary along the OTHER
axis. So:
    - per-channel-K  -> group along the TOKEN dim (axis=-2): each channel keeps
                        its own scale within each token-window. Outlier
                        channels stay isolated.
    - per-token-V    -> group along the CHANNEL dim (axis=-1): each token keeps
                        its own scale within each channel-window. Quantization
                        error in one token doesn't bleed into others.

(Easy to flip the wrong way: grouping along the channel for K mashes outlier
channels into the same scale as normal channels, crushing the normals.)

For real memory savings we pack 2-bit and 4-bit codes into uint8 (8 / b values per byte).

Why asymmetric?  Cached K has channel-wise outliers and is *not* zero-centered, so a
symmetric range wastes codes at low bit-width. Asymmetric maps the actual group range
exactly into [0, 2^b - 1], which is especially important at INT2.

Why group-wise?  One scale per channel/token preserves resolution where the distribution
varies. Group size 32 is the canonical sweet-spot from KIVI.

References:
- KIVI (Liu et al., 2024): https://arxiv.org/abs/2402.02750
- HF Cache API (axis_key=1, axis_value=0): https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/cache_utils.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


# ----------------------------------------------------------------------------
# Quantize / dequantize at a chosen bit-width.
# ----------------------------------------------------------------------------

@dataclass
class QuantTensor:
    """A quantized tensor + the metadata needed to dequantize it.

    `codes` holds the integer codes in either uint8 (INT8) or *packed* uint8
    (INT4 = 2 vals/byte, INT2 = 4 vals/byte). `scale` and `zero` are stored
    in fp16 for memory efficiency. The `zero` field is the per-group floating
    offset (the group minimum), kept under its historical name for API
    compatibility.

    Shapes:
      - For per-channel grouping along axis `group_axis` with group size G,
        the original tensor x had shape [..., dim_along_group_axis, ...].
        We reshape to [..., n_groups, G] and store `codes` accordingly,
        `scale`/`zero` with shape [..., n_groups, 1].
    """
    codes: torch.Tensor          # uint8 (packed for INT4/INT2)
    scale: torch.Tensor          # fp16, shape (..., n_groups, 1) along group_axis
    zero: torch.Tensor           # fp16 offset/min, same shape as scale
    bits: int                    # 2, 4, or 8
    group_size: int              # number of original elements per (scale, zero) group
    orig_shape: torch.Size       # exact original shape, needed to undo padding/reshape
    group_axis: int              # which axis was grouped (channel for K, token for V)
    compute_dtype: torch.dtype   # dtype to dequantize back into

    def nbytes_packed(self) -> int:
        """Estimated total bytes occupied (packed codes + scales + zeros)."""
        return (
            self.codes.element_size() * self.codes.numel()
            + self.scale.element_size() * self.scale.numel()
            + self.zero.element_size() * self.zero.numel()
        )


def _maybe_pad_for_group(x: torch.Tensor, axis: int, group_size: int) -> Tuple[torch.Tensor, int]:
    """Pad axis with zeros to a multiple of group_size.  Returns (padded, n_pad)."""
    n = x.shape[axis]
    rem = n % group_size
    if rem == 0:
        return x, 0
    n_pad = group_size - rem
    pad_shape = list(x.shape)
    pad_shape[axis] = n_pad
    pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=axis), n_pad


def quantize(
    x: torch.Tensor,
    bits: int,
    group_axis: int,
    group_size: int = 32,
) -> QuantTensor:
    """Asymmetric uniform group-wise quantization.

    Args:
        x: input tensor in fp16 / bf16 / fp32.
        bits: 2, 4, or 8.
        group_axis: axis whose elements are grouped together (one (scale,zero) per group).
                    Scales then vary along the OTHER axes.
                    For per-channel-K with shape [B, H, T, D], pass group_axis=-2 (the
                    token dim) so each channel keeps its own scale.
                    For per-token-V with shape [B, H, T, D], pass group_axis=-1 (the
                    channel dim) so each token keeps its own scale.
        group_size: number of consecutive elements along group_axis sharing one scale.

    Returns:
        A QuantTensor.  Use `dequantize(qt)` to invert.
    """
    if bits not in (2, 4, 8, 16):
        raise ValueError(f"bits must be 2, 4, 8, or 16 — got {bits}")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"x must be float — got {x.dtype}")

    if bits == 16:
        # No-op passthrough. We still wrap in a QuantTensor so the cache code
        # path is uniform. scale/zero are unused.
        dummy = torch.zeros(1, dtype=torch.float16, device=x.device)
        return QuantTensor(
            codes=x.contiguous(),
            scale=dummy,
            zero=dummy,
            bits=16,
            group_size=group_size,
            orig_shape=x.shape,
            group_axis=group_axis if group_axis >= 0 else x.dim() + group_axis,
            compute_dtype=x.dtype,
        )

    orig_shape = x.shape
    compute_dtype = x.dtype
    if group_axis < 0:
        group_axis = x.dim() + group_axis

    # Pad along group axis to a multiple of group_size.
    x_padded, n_pad = _maybe_pad_for_group(x, group_axis, group_size)

    # Do the range math in fp32 even when model activations are bf16/fp16.
    # The stored scale/offset remain fp16 for the intended cache footprint.
    x_work = x_padded.float()

    # Reshape so the group axis becomes (n_groups, group_size).
    new_shape = list(x_padded.shape)
    new_shape[group_axis : group_axis + 1] = [-1, group_size]
    x_grp = x_work.reshape(new_shape)

    # Compute per-group min and max along the new group dimension (group_axis + 1).
    grp_dim = group_axis + 1
    x_min = x_grp.amin(dim=grp_dim, keepdim=True)
    x_max = x_grp.amax(dim=grp_dim, keepdim=True)

    qmax = (1 << bits) - 1
    # Avoid div-by-zero for constant groups, and keep the value representable
    # after fp16 scale storage.
    scale = ((x_max - x_min) / qmax).clamp_min(torch.finfo(torch.float16).tiny)
    offset = x_min

    # Quantize.
    q = torch.round((x_grp - offset) / scale).clamp(0, qmax).to(torch.uint8)

    # Flatten group dims back, then pack.
    q_flat_shape = list(x_padded.shape)
    q_flat = q.reshape(q_flat_shape)
    if bits == 8:
        codes = q_flat
    elif bits == 4:
        codes = _pack_int4(q_flat, axis=group_axis)
    else:  # bits == 2
        codes = _pack_int2(q_flat, axis=group_axis)

    # Squeeze the (n_groups, group_size) shape down for scale/zero so they keep one
    # entry per group.
    s_shape = list(x_grp.shape)
    s_shape[grp_dim] = 1
    return QuantTensor(
        codes=codes,
        scale=scale.reshape(s_shape).to(torch.float16),
        zero=offset.reshape(s_shape).to(torch.float16),
        bits=bits,
        group_size=group_size,
        orig_shape=orig_shape,
        group_axis=group_axis,
        compute_dtype=compute_dtype,
    )


def dequantize(qt: QuantTensor) -> torch.Tensor:
    """Inverse of `quantize` — returns a tensor in qt.compute_dtype with shape qt.orig_shape."""
    if qt.bits == 16:
        return qt.codes.to(qt.compute_dtype)
    if qt.bits == 8:
        q = qt.codes
    elif qt.bits == 4:
        q = _unpack_int4(qt.codes, axis=qt.group_axis)
    else:  # 2
        q = _unpack_int2(qt.codes, axis=qt.group_axis)

    # Re-introduce the (n_groups, group_size) split.
    new_shape = list(q.shape)
    new_shape[qt.group_axis : qt.group_axis + 1] = [-1, qt.group_size]
    q_grp = q.reshape(new_shape).to(qt.compute_dtype)

    scale = qt.scale.to(qt.compute_dtype)
    offset = qt.zero.to(qt.compute_dtype)
    x_grp = q_grp * scale + offset

    x = x_grp.reshape(*q.shape)

    # Strip the padding we added in quantize().
    n_orig = qt.orig_shape[qt.group_axis]
    if x.shape[qt.group_axis] != n_orig:
        idx = [slice(None)] * x.dim()
        idx[qt.group_axis] = slice(0, n_orig)
        x = x[tuple(idx)]
    return x


# ----------------------------------------------------------------------------
# Bit-packing for INT4 (2 vals/byte) and INT2 (4 vals/byte).
#
# Packing along an arbitrary axis is a little fiddly because we need to combine
# adjacent values along that axis into a single byte. We do it with reshape-pair-
# bitshift which is fully vectorised.
# ----------------------------------------------------------------------------

def _norm_axis(x: torch.Tensor, axis: int) -> int:
    return axis if axis >= 0 else x.dim() + axis


def _pack_int4(q_uint8: torch.Tensor, axis: int) -> torch.Tensor:
    """Pack values in [0, 15] along `axis` two-per-byte. axis length must be even."""
    axis = _norm_axis(q_uint8, axis)
    n = q_uint8.shape[axis]
    if n % 2 != 0:
        raise AssertionError(f"INT4 pack axis must be even; got {n}")
    new_shape = list(q_uint8.shape)
    new_shape[axis : axis + 1] = [n // 2, 2]
    pair = q_uint8.reshape(new_shape)
    # low nibble = first of pair, high nibble = second of pair
    low_idx = [slice(None)] * pair.dim()
    high_idx = [slice(None)] * pair.dim()
    low_idx[axis + 1] = 0
    high_idx[axis + 1] = 1
    low = pair[tuple(low_idx)]
    high = pair[tuple(high_idx)]
    return (low | (high << 4)).contiguous()


def _unpack_int4(packed: torch.Tensor, axis: int) -> torch.Tensor:
    """Inverse of `_pack_int4`. Returns uint8 values in [0, 15] along `axis`."""
    axis = _norm_axis(packed, axis)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    # Stack along a new dim immediately after `axis`, then merge that with `axis`.
    stacked = torch.stack([low, high], dim=axis + 1)
    new_shape = list(packed.shape)
    new_shape[axis] = packed.shape[axis] * 2
    return stacked.reshape(new_shape).contiguous()


def _pack_int2(q_uint8: torch.Tensor, axis: int) -> torch.Tensor:
    """Pack values in [0, 3] along `axis` four-per-byte. axis length must be % 4."""
    axis = _norm_axis(q_uint8, axis)
    n = q_uint8.shape[axis]
    if n % 4 != 0:
        raise AssertionError(f"INT2 pack axis must be a multiple of 4; got {n}")
    new_shape = list(q_uint8.shape)
    new_shape[axis : axis + 1] = [n // 4, 4]
    quad = q_uint8.reshape(new_shape)
    idx = [[slice(None)] * quad.dim() for _ in range(4)]
    for k in range(4):
        idx[k][axis + 1] = k
    a, b, c, d = (quad[tuple(idx[k])] for k in range(4))
    return (a | (b << 2) | (c << 4) | (d << 6)).contiguous()


def _unpack_int2(packed: torch.Tensor, axis: int) -> torch.Tensor:
    """Inverse of `_pack_int2`. Returns uint8 in [0, 3] along `axis`."""
    axis = _norm_axis(packed, axis)
    a = packed & 0x03
    b = (packed >> 2) & 0x03
    c = (packed >> 4) & 0x03
    d = (packed >> 6) & 0x03
    stacked = torch.stack([a, b, c, d], dim=axis + 1)
    new_shape = list(packed.shape)
    new_shape[axis] = packed.shape[axis] * 4
    return stacked.reshape(new_shape).contiguous()


# ----------------------------------------------------------------------------
# Theoretical bytes-per-element accounting (for the memory plot).
#
# Per element in the original tensor, the storage cost is:
#   codes:     bits / 8   bytes
#   scale:     2 / G      bytes  (fp16, one per group)
#   zero:      2 / G      bytes
#
# For FP16 baseline: 2 bytes per element, no scales.
# ----------------------------------------------------------------------------

def bytes_per_element(bits: int, group_size: int) -> float:
    """Bytes per ORIGINAL element including amortised scale + zero overhead."""
    if bits == 16:  # FP16 baseline
        return 2.0
    if bits not in (2, 4, 8):
        raise ValueError(f"bits must be 2, 4, 8, or 16 — got {bits}")
    return bits / 8.0 + 2.0 * 2.0 / group_size  # codes + (scale+zero)*fp16


# ----------------------------------------------------------------------------
# Quality metrics for round-trip error.
# ----------------------------------------------------------------------------

@torch.no_grad()
def round_trip_mse(x: torch.Tensor, bits: int, group_axis: int, group_size: int = 32) -> float:
    """Mean-squared error of (x - dequantize(quantize(x))). Useful for sanity tests."""
    qt = quantize(x, bits, group_axis, group_size)
    x_hat = dequantize(qt)
    return ((x - x_hat).float() ** 2).mean().item()
