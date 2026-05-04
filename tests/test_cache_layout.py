import torch

from tiered_kv import FixedWindowPolicy, TierConfig, TieredKVCache


def test_prefill_partition_preserves_sequence_order_with_int8_tiers():
    k = torch.linspace(10.0, 11.0, steps=4096, dtype=torch.float16).reshape(1, 1, 128, 32)
    v = torch.linspace(-3.0, 2.0, steps=4096, dtype=torch.float16).reshape(1, 1, 128, 32)
    cache = TieredKVCache(
        policy=FixedWindowPolicy(fp16=16, int8=32, int4=64, n_sink=4),
        config=TierConfig(tier_bits=[16, 8, 8, 8], group_size=8, compute_dtype=torch.float16),
    )

    k_full, v_full = cache.update(k, v, layer_idx=0)

    assert k_full.shape == k.shape
    assert v_full.shape == v.shape
    assert torch.allclose(k_full, k, atol=0.02, rtol=0)
    assert torch.allclose(v_full, v, atol=0.02, rtol=0)
    assert cache.get_seq_length(0) == 128
    assert cache.total_bytes() < 2 * k.element_size() * k.numel()
