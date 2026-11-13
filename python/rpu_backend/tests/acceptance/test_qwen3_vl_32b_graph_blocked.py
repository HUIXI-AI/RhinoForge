from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from rpu_backend.adapters import qwen3_vl
from rpu_backend.adapters.qwen3_vl import text as qwen3vl_text
from rpu_backend.adapters.qwen3_vl import vision
from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError

def _config_32b():
    return SimpleNamespace(architectures=['Qwen3VLForConditionalGeneration'], model_type='qwen3_vl', tie_word_embeddings=False, quant_config={'method': 'w8a16', 'mode': 'per_channel_symmetric', 'qaxis': 0, 'skip_modules': ['lm_head'], 'quantized_lm_head': False, 'quantized_embed_tokens': False, 'lm_head_untied': False, 'embed_tokens_untied': False}, text_config=SimpleNamespace(model_type='qwen3_vl_text', hidden_size=5120, intermediate_size=25600, num_hidden_layers=64, num_attention_heads=64, num_key_value_heads=8, head_dim=128, vocab_size=151936, hidden_act='silu', rms_norm_eps=1e-06, attention_bias=False, use_cache=True, rope_scaling={'mrope_interleaved': True, 'mrope_section': [24, 20, 20], 'rope_type': 'default', 'rope_theta': 5000000}), vision_config=SimpleNamespace(model_type='qwen3_vl', in_channels=3, num_position_embeddings=2304, hidden_size=1152, intermediate_size=4304, depth=27, num_heads=16, patch_size=16, temporal_patch_size=2, spatial_merge_size=2, deepstack_visual_indexes=[8, 16, 24], out_hidden_size=5120, hidden_act='gelu_pytorch_tanh'))

def _config_8b():
    config = _config_32b()
    config.quant_config = None
    config.text_config.hidden_size = 4096
    config.text_config.intermediate_size = 12288
    config.text_config.num_hidden_layers = 36
    config.text_config.num_attention_heads = 32
    config.vision_config.out_hidden_size = 4096
    return config

@pytest.mark.parametrize('value', [None, '0', 'true', '01', ' 1', '1 '])
def test_32b_preflight_requires_exact_graph_blocked_opt_in(monkeypatch, value):
    if value is None:
        monkeypatch.delenv('QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED', raising=False)
    else:
        monkeypatch.setenv('QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED', value)
    with pytest.raises(UnsupportedModelError, match='graph-blocked'):
        qwen3_vl.Qwen3VLAdapter.preflight(_config_32b())

def test_32b_exact_opt_in_admits_only_exact_logical_config(monkeypatch):
    monkeypatch.setenv('QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED', '1')
    config = _config_32b()
    qwen3_vl.Qwen3VLAdapter.preflight(config)
    assert config.vision_config.intermediate_size == 4304
    config.vision_config.intermediate_size = 4352
    with pytest.raises(UnsupportedModelError, match='only the exact'):
        qwen3_vl.Qwen3VLAdapter.preflight(config)

def test_exact_8b_is_admitted_but_nearby_profiles_fail_closed():
    config = _config_8b()
    qwen3_vl.Qwen3VLAdapter.preflight(config)
    config.vision_config.depth = 26
    with pytest.raises(UnsupportedModelError, match='unsupported vision profile'):
        qwen3_vl.Qwen3VLAdapter.preflight(config)
    config = _config_8b()
    config.quant_config = {'method': 'w8a16'}
    with pytest.raises(UnsupportedModelError, match='requires an FP16 checkpoint or a registered runtime quantization profile'):
        qwen3_vl.Qwen3VLAdapter.preflight(config)
    config = _config_8b()
    config.quantization_config = {'quant_method': 'bitsandbytes'}
    with pytest.raises(UnsupportedModelError, match='requires an FP16 checkpoint or a registered runtime quantization profile'):
        qwen3_vl.Qwen3VLAdapter.preflight(config)
    config = _config_8b()
    config.text_config.hidden_act = 'gelu'
    with pytest.raises(UnsupportedModelError, match='unsupported.*text semantics'):
        qwen3_vl.Qwen3VLAdapter.preflight(config)
    config = _config_8b()
    config.vision_config.in_channels = 1
    with pytest.raises(UnsupportedModelError, match='unsupported.*vision semantics'):
        qwen3_vl.Qwen3VLAdapter.preflight(config)

def test_32b_loaded_inventory_is_checked_before_rpu_mutation(monkeypatch):
    monkeypatch.setenv('QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED', '1')
    model = SimpleNamespace(config=_config_32b(), model=SimpleNamespace(language_model=SimpleNamespace(layers=[], embed_tokens=SimpleNamespace(weight=torch.empty(1, 1)))), lm_head=SimpleNamespace(weight=torch.empty(1, 1)))
    with pytest.raises(RPUBackendError, match='layers=0, expected 64'):
        qwen3_vl._validate_qwen3_vl_32b_w8a16_model(model)

def test_direct_32b_vision_install_cannot_bypass_top_level_gate(monkeypatch):
    config = _config_32b().vision_config
    model = SimpleNamespace(config=config)
    monkeypatch.setenv('QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED', '1')
    with pytest.raises(ValueError, match='gated top-level adapter'):
        vision.install_qwen3_vl_vision_for_rpu(model)
    monkeypatch.setenv('QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED', 'true')
    with pytest.raises(ValueError, match="exact opt-in value '1'"):
        vision.install_qwen3_vl_vision_for_rpu(model, _allow_graph_blocked_32b=True)

def test_direct_8b_vision_install_requires_the_top_level_padding_gate(monkeypatch):
    model = SimpleNamespace(config=_config_8b().vision_config)
    calls = []
    monkeypatch.setenv('RPU_QWEN3VL_VISION_HOST_FP32_PATCH', 'true')
    monkeypatch.setenv('RPU_QWEN3VL_VISION_BATCH', 'on')
    monkeypatch.setenv('RPU_QWEN3VL_VISION_BATCH_CAP', '4')
    monkeypatch.setattr(vision, '_install_qwen3_vl_vision_for_rpu_impl', lambda *args, **kwargs: calls.append(kwargs) or 17)
    with pytest.raises(ValueError, match='gated top-level adapter'):
        vision.install_qwen3_vl_vision_for_rpu(model)
    assert calls == []
    assert vision.install_qwen3_vl_vision_for_rpu(model, _allow_padded_8b=True) == 17
    assert calls[0]['_allow_padded_8b'] is True

def test_8b_long_text_envelope_retains_its_chunk_ceiling():
    envelope = qwen3vl_text.lookup_causal_decoder('qwen3_vl_text', 36, 4096)
    assert (envelope.max_kv_len, envelope.chunk) == (0, 128)

def _tensor_module(weight, bias=None):
    module = SimpleNamespace(weight=torch.nn.Parameter(weight, requires_grad=False))
    if bias is not None:
        module.bias = torch.nn.Parameter(bias, requires_grad=False)
    return module

def test_8b_32b_vision_padding_is_exactly_zero_at_conversion_boundary(monkeypatch):
    hidden = 1152
    logical_intermediate = 4304
    block = SimpleNamespace(attn=SimpleNamespace(qkv=_tensor_module(torch.ones(3 * hidden, hidden, dtype=torch.float16), torch.ones(3 * hidden, dtype=torch.float16)), proj=_tensor_module(torch.ones(hidden, hidden, dtype=torch.float16), torch.ones(hidden, dtype=torch.float16))), mlp=SimpleNamespace(linear_fc1=_tensor_module(torch.ones(logical_intermediate, hidden, dtype=torch.float16), torch.ones(logical_intermediate, dtype=torch.float16)), linear_fc2=_tensor_module(torch.ones(hidden, logical_intermediate, dtype=torch.float16), torch.ones(hidden, dtype=torch.float16))))
    monkeypatch.setattr(vision, 'tp_col_swizzle_mc_weight', lambda weight, *a, **k: weight)
    monkeypatch.setattr(vision, 'tp_row_swizzle_mc_weight', lambda weight, *a, **k: weight)
    vision._convert_vision_block_weights_for_rpu(block, num_heads=16, hidden_size=hidden, physical_head_dim=80, physical_intermediate_size=4352)
    q = block._rpu_q_w.view(16, 80, hidden)
    o = block.attn.proj.weight.view(hidden, 16, 80)
    assert torch.count_nonzero(q[:, :72, :] != 1) == 0
    assert torch.count_nonzero(q[:, 72:, :]) == 0
    assert torch.count_nonzero(o[:, :, :72] != 1) == 0
    assert torch.count_nonzero(o[:, :, 72:]) == 0
    assert torch.count_nonzero(block._rpu_q_b.view(16, 80)[:, 72:]) == 0
    assert tuple(block.mlp.linear_fc1.weight.shape) == (4352, hidden)
    assert tuple(block.mlp.linear_fc2.weight.shape) == (hidden, 4352)
    assert torch.count_nonzero(block.mlp.linear_fc1.weight[4304:, :]) == 0
    assert torch.count_nonzero(block._rpu_fc1_b[4304:]) == 0
    assert torch.count_nonzero(block.mlp.linear_fc2.weight[:, 4304:]) == 0
    vision._convert_vision_block_weights_for_rpu(block, num_heads=16, hidden_size=hidden, physical_head_dim=80, physical_intermediate_size=4352)

def test_generic_vision_padding_is_rejected_before_swizzle():
    block = SimpleNamespace(attn=SimpleNamespace(qkv=torch.nn.Linear(16, 48, bias=True), proj=torch.nn.Linear(16, 16, bias=True)), mlp=SimpleNamespace(linear_fc1=torch.nn.Linear(16, 32, bias=True), linear_fc2=torch.nn.Linear(32, 16, bias=True)))
    with pytest.raises(ValueError, match='private to the exact 8B/32B'):
        vision._convert_vision_block_weights_for_rpu(block, num_heads=1, hidden_size=16, physical_head_dim=32, physical_intermediate_size=128)

def test_native_32b_layout_keeps_logical_rope_and_scale_contracts():
    source = (Path(__file__).resolve().parents[4] / 'src/fused/rpu_qwen3vl_vision_model.cpp').read_text(encoding='utf-8')
    assert 'logical_head_dim == 72 && head_dim == 80' in source
    assert 'intermediate_size == 4352' in source
    assert 'orig_head_dim_ = logical_head_dim' in source
    assert 'local_heads, orig_head_dim_, head_dim_pad' in source
    assert 'std::sqrt(static_cast<double>(orig_head_dim_))' in source
    assert 'max_hw_ * (orig_head_dim_ / 4)' in source
    assert 'head_dim() / 4' not in source
    assert 'GRAPH_CLEAR_EVERY' not in source
