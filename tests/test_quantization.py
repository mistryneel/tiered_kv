import torch

from tiered_kv import quantization as Q


def test_int8_positive_range_does_not_saturate():
    x = torch.linspace(10.0, 11.0, steps=128, dtype=torch.float16).reshape(1, 1, 128, 1)

    qt = Q.quantize(x, bits=8, group_axis=-2, group_size=32)
    x_hat = Q.dequantize(qt)

    assert x_hat.shape == x.shape
    assert (x_hat - x).abs().max().item() < 0.01


def test_int4_pack_round_trip_preserves_order():
    q = (torch.arange(32, dtype=torch.uint8).reshape(1, 1, 16, 2) % 16).contiguous()

    packed = Q._pack_int4(q, axis=-2)
    unpacked = Q._unpack_int4(packed, axis=-2)

    assert torch.equal(unpacked, q)


def test_int2_pack_round_trip_preserves_order():
    q = (torch.arange(64, dtype=torch.uint8).reshape(1, 1, 32, 2) % 4).contiguous()

    packed = Q._pack_int2(q, axis=-2)
    unpacked = Q._unpack_int2(packed, axis=-2)

    assert torch.equal(unpacked, q)
