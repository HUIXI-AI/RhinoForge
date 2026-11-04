"""Cache storage follows requested capacity and allocator failures, not 8K."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.api.cache import RPUCache
from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache


@pytest.mark.parametrize("capacity", [8193, 16384, 32769])
def test_cache_factory_allocates_complete_large_capacity(capacity):
    model = SimpleNamespace(config=SimpleNamespace(
        num_hidden_layers=1, hidden_size=128, num_attention_heads=8,
        num_key_value_heads=1, head_dim=16))
    cache = RPUCache.from_model(model, max_seq_len=capacity, device="cpu")
    physical_rows = (capacity + 15) // 16 * 16
    assert cache.max_seq_len == capacity
    for tensor in (cache.k_caches[0], cache.v_caches[0]):
        assert tensor.numel() == physical_rows * 16
        assert not tensor.count_nonzero()
        tensor[:, -1].fill_(3)
        assert torch.all(tensor[:, -1] == 3)
    cache.update_position(capacity)
    with pytest.raises(ValueError, match="overflow"):
        cache.update_position(1)
    assert cache.position == capacity


@pytest.mark.parametrize("capacity", [16384, 32768])
def test_hybrid_cache_keeps_padding_and_recurrent_storage_separate(capacity):
    cache = Qwen3_5Cache(
        num_layers=2, batch_size=1, max_seq_len=capacity,
        num_kv_heads=8, head_dim=16,
        layer_types=["linear_attention", "full_attention"],
        num_v_heads=8, key_head_dim=1, value_head_dim=1,
        conv_dim=16, conv_kernel_dim=4, device="cpu")
    assert cache.max_seq_len == capacity
    assert cache.allocated_max_seq_len == capacity + 128
    assert cache.k_caches[0].numel() == cache.v_caches[0].numel() == 0
    assert cache.k_caches[1].numel() == (capacity + 128) * 8 * 16
    assert cache.recurrent_states[0].numel() == 8
    assert cache.conv_states[0].numel() == 64
    cache.update_position(capacity)
    with pytest.raises(ValueError, match="overflow"):
        cache.update_position(1)


def test_cache_propagates_allocator_failure_without_truncating(monkeypatch):
    error = torch.OutOfMemoryError("requested cache does not fit")
    calls = []

    def allocate(shape, **kwargs):
        calls.append(shape)
        raise error

    monkeypatch.setattr(torch, "zeros", allocate)
    with pytest.raises(torch.OutOfMemoryError) as caught:
        RPUCache(1, 1, 32768, 1, 16, device="cpu")
    assert caught.value is error
    assert len(calls) == 1
    assert calls[0][1] * 16 == 32768
