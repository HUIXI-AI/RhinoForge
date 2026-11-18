"""Native cache contracts at the real Transformers generation boundary."""
import inspect
from types import SimpleNamespace

import pytest
import torch
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from rpu_backend.api.cache import RPUCache
from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache
from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache
from rpu_backend.adapters.gemma4.cache import Gemma4KVCache
from rpu_backend.adapters.gemma4.geometry import layer_geometry


def _cache(kind="plain"):
    if kind == "gemma4":
        geometry = layer_geometry(SimpleNamespace(
            num_hidden_layers=2, num_kv_shared_layers=0, head_dim=64,
            global_head_dim=64, num_attention_heads=8, num_key_value_heads=2,
            layer_types=["sliding_attention", "full_attention"], sliding_window=16))
        return Gemma4KVCache(geometry, max_seq_len=16, device="cpu")
    kwargs = dict(num_layers=2, batch_size=1, max_seq_len=16,
                  num_kv_heads=2, head_dim=16, device="cpu")
    if kind == "plain":
        return RPUCache(**kwargs)
    cls = Qwen3_5Cache if kind == "dense" else Qwen3_5MoeCache
    return cls(**kwargs, layer_types=["linear_attention", "full_attention"],
               num_v_heads=2, key_head_dim=4, value_head_dim=4,
               conv_dim=16, conv_kernel_dim=3)


@pytest.mark.parametrize("kind", ["plain", "dense", "moe", "gemma4"])
def test_native_cache_metadata_and_mask_interfaces(kind):
    cache = _cache(kind)
    assert cache.batch_size == cache.max_batch_size == 1
    assert cache.get_max_length() == cache.get_max_length(0) == 16
    assert cache.get_max_cache_shape(1) == cache.max_cache_len == 16
    assert cache.is_initialized
    assert not cache.is_compileable and not cache.is_croppable
    assert cache.is_sliding == [False, False]
    assert cache.is_linear == ([True, False] if kind in ("dense", "moe") else [False, False])
    assert cache.get_mask_sizes(3, 1) == (3, 0)
    cache.update_position(3)
    assert cache.get_query_offset() == cache.get_seq_length() == 3
    assert cache.get_mask_sizes(torch.tensor([3, 4]), 1) == (5, 0)
    assert cache.get_mask_sizes(cache_position=torch.tensor([3, 4])) == (5, 0)
    assert cache.get_mask_sizes(query_length=2, layer_idx=1) == (5, 0)
    with pytest.raises(ValueError, match="capacity"):
        cache.get_mask_sizes(14)
    with pytest.raises(ValueError, match="not both"):
        cache.get_mask_sizes(2, cache_position=torch.tensor([3, 4]))


@pytest.mark.parametrize("kind", ["plain", "dense", "moe", "gemma4"])
def test_real_hf_mask_includes_current_query_without_advancing_cache(kind):
    from transformers.masking_utils import create_causal_mask

    cache = _cache(kind)
    cache.update_position(3)
    config = PretrainedConfig()
    config._attn_implementation = "eager"
    kwargs = dict(config=config, inputs_embeds=torch.zeros((1, 2, 16)),
                  attention_mask=torch.ones((1, 5), dtype=torch.long),
                  past_key_values=cache)
    if "cache_position" in inspect.signature(create_causal_mask).parameters:
        kwargs["cache_position"] = torch.tensor([3, 4])
    mask = create_causal_mask(**kwargs)
    assert mask.shape == (1, 1, 2, 5)
    assert (mask[0, 0] == 0).tolist() == [
        [True, True, True, True, False], [True, True, True, True, True]]
    assert cache.position == 3


def test_uninstalled_hf_attention_cannot_silently_drop_cache_history():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(vocab_size=11, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=16)
    model = Qwen3ForCausalLM(config).eval()
    cache = RPUCache(1, 1, 16, 1, 16, device="cpu")
    with torch.inference_mode(), pytest.raises(NotImplementedError, match="update"):
        model(input_ids=torch.tensor([[1, 2]]), past_key_values=cache, use_cache=True)
    assert cache.position == 0
    assert not cache.k_caches[0].any() and not cache.v_caches[0].any()


@pytest.mark.parametrize("kind", ["plain", "dense", "moe", "gemma4"])
@pytest.mark.parametrize("operation,args,kwargs", [
    ("update", (torch.ones(1), torch.ones(1), 0), {"cache_position": torch.tensor([3])}),
    ("update_conv_state", (torch.ones(1), 0), {}),
    ("update_recurrent_state", (torch.ones(1), 0), {}),
    ("has_previous_state", (), {"layer_idx": 0}),
    ("reorder_cache", (torch.tensor([0]),), {}),
    ("batch_select_indices", (torch.tensor([0]),), {}),
    ("batch_repeat_interleave", (2,), {}),
    ("crop", (-1,), {}),
    ("crop", (), {"max_length": 0}),
    ("activate_past_recording", (), {}),
    ("early_initialization", (1, 2, 16, torch.float16, torch.device("cpu")), {}),
    ("prefetch", (0,), {}),
    ("offload", (0,), {}),
])
def test_hf_mutations_fail_before_changing_native_buffers(kind, operation, args, kwargs):
    cache = _cache(kind)
    cache.update_position(3)
    tensors = cache.k_caches + cache.v_caches
    tensors += getattr(cache, "recurrent_states", []) + getattr(cache, "conv_states", [])
    for tensor in tensors:
        tensor.fill_(1)
    pointers = [tensor.data_ptr() for tensor in tensors]
    with pytest.raises(NotImplementedError, match=operation):
        getattr(cache, operation)(*args, **kwargs)
    assert cache.position == cache.seen_tokens == 3
    assert [tensor.data_ptr() for tensor in tensors] == pointers
    assert all(bool(torch.all(tensor == 1)) for tensor in tensors)
    cache.reset()
    assert cache.position == cache.seen_tokens == 0
    assert [tensor.data_ptr() for tensor in tensors] == pointers
    assert all(not bool(tensor.any()) for tensor in tensors)


class _TinyFusedCaller(PreTrainedModel, GenerationMixin):
    """Use real HF generation with a tiny CPU stand-in for the native forward."""
    config_class = PretrainedConfig
    main_input_name = "input_ids"

    def __init__(self):
        config = PretrainedConfig(vocab_size=11, num_hidden_layers=2,
                                  is_decoder=True, is_encoder_decoder=False,
                                  pad_token_id=0, eos_token_id=None)
        super().__init__(config)
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.requests = []

    def forward(self, input_ids, past_key_values=None, attention_mask=None,
                position_ids=None, cache_position=None, use_cache=True,
                return_dict=True, **kwargs):
        assert isinstance(past_key_values, RPUCache) and use_cache
        assert input_ids.shape[0] == past_key_values.batch_size
        start, count = past_key_values.position, input_ids.shape[1]
        self.requests.append((start, count))
        # Fused callers own native writes and commit the logical position once.
        past_key_values.k_caches[-1].fill_(start + count)
        past_key_values.update_position(count)
        logits = torch.zeros((*input_ids.shape, self.config.vocab_size))
        logits.scatter_(-1, ((input_ids + 1) % self.config.vocab_size).unsqueeze(-1), 1)
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)


@pytest.mark.parametrize("kind", ["plain", "dense", "moe", "gemma4"])
def test_real_greedy_generate_reuses_native_cache_after_reset(kind):
    model = _TinyFusedCaller().eval()
    cache = _cache(kind)
    original = [tensor.data_ptr() for tensor in cache.k_caches + cache.v_caches]
    for _ in range(2):
        cache.reset()
        model.requests.clear()
        result = model.generate(input_ids=torch.tensor([[1, 2, 3]]),
                                attention_mask=torch.ones((1, 3), dtype=torch.long),
                                past_key_values=cache, max_new_tokens=3,
                                do_sample=False, return_dict_in_generate=True)
        assert result.sequences.tolist() == [[1, 2, 3, 4, 5, 6]]
        assert result.past_key_values is cache
        assert model.requests == [(0, 3), (3, 1), (4, 1)]
        assert cache.position == 5
        assert [tensor.data_ptr() for tensor in cache.k_caches + cache.v_caches] == original
