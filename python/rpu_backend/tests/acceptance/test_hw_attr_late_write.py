import ast
import copy
import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
ROOT = Path(__file__).resolve().parents[4]

def test_public_hw_attrs_frozen():
    from rpu_backend.runtime.hw_attrs import PUBLIC_HW_ATTRS
    assert isinstance(PUBLIC_HW_ATTRS, frozenset)
    assert PUBLIC_HW_ATTRS == frozenset({'_rpu_chunk_size', '_rpu_execution', '_rpu_spm_mode', '_rpu_warmup', '_rpu_debug_export'})

def test_cold_only_attrs_frozen():
    from rpu_backend.runtime.hw_attrs import COLD_ONLY_ATTRS, PUBLIC_HW_ATTRS
    assert isinstance(COLD_ONLY_ATTRS, frozenset)
    assert COLD_ONLY_ATTRS == frozenset({'_rpu_execution', '_rpu_spm_mode', '_rpu_warmup'})
    assert COLD_ONLY_ATTRS.issubset(PUBLIC_HW_ATTRS), 'COLD_ONLY_ATTRS must be a subset of PUBLIC_HW_ATTRS'

def test_internal_transitional_frozen():
    from rpu_backend.runtime.hw_attrs import INTERNAL_HW_ATTRS_TRANSITIONAL
    assert isinstance(INTERNAL_HW_ATTRS_TRANSITIONAL, frozenset)
    assert INTERNAL_HW_ATTRS_TRANSITIONAL == frozenset({'_rpu_decoder_topology', '_rpu_vision_num_cores', '_rpu_qwen3vl_vision_num_cores', '_rpu_swizzled', '_rpu_swizzle_started', '_rpu_attn_tp', '_rpu_effective_num_kv_heads', '_rpu_max_seq_len', '_rpu_requested_max_seq_len', '_rpu_cache', '_rpu_weights_converted', '_rpu_gemm_weight', '_rpu_pos_emb_fused', '_rpu_linear_partition', '_rpu_linear_num_cores', '_rpu_decoder_handle', '_rpu_decoder_handle_finalizer', '_rpu_vlm_decoder_handle', '_rpu_vlm_decoder_handle_finalizer', '_rpu_action_handle', '_rpu_action_handle_finalizer', '_rpu_vision_handle', '_rpu_vision_handle_finalizer', '_rpu_gemma2_handle', '_rpu_kh', '_rpu_kw', '_rpu_cin_orig', '_rpu_cin_padded', '_rpu_cout', '_rpu_stride', '_rpu_lm_head_w_keepalive', '_rpu_lm_head_scale_keepalive', '_rpu_lm_head_int8_cpu_ref', '_rpu_lm_head_scale_cpu_ref', '_rpu_lm_head_fp16_cpu_ref', '_rpu_embed_tokens_int8_cpu_ref', '_rpu_embed_tokens_scale_cpu_ref', '_rpu_siglip_weights_converted', '_rpu_layer_idx', '_rpu_num_layers', '_rpu_num_heads', '_rpu_head_dim', '_rpu_intermediate_size', '_rpu_noise_seed', '_rpu_legacy_prefix_pad16_request', '_rpu_legacy_kvinsert_pad16_request', '_rpu_lazy_init_checked', '_rpu_dynamo_safe', '_rpu_required_attrs', '_rpu_deepstack_lang_layers', '_rpu_batch_decode_enabled', '_rpu_decoder_graph_cache', '_rpu_decoder_retirement_state', '_rpu_vlm_decoder_retirement_state', '_rpu_action_retirement_state', '_rpu_vision_retirement_state', '_rpu_decoder_num_layers', '_rpu_decoder_hidden_size', '_rpu_decoder_deepstack_hash', '_rpu_prefill_execution_alignment', '_rpu_kv_cache_layer_bank_size', '_rpu_prefill_kv_only', '_rpu_decode_stage_descriptor', '_rpu_execution_generation', '_rpu_adarms_graph_cache', '_rpu_siglip_graph_cache', '_rpu_gemma_graph_cache', '_rpu_gemma2_graph_cache', '_rpu_planned_prefill_plan', '_rpu_fused_denoise_handle', '_rpu_fused_denoise_handle_finalizer', '_rpu_fused_denoise_retirement_state', '_rpu_fuse_validated', '_rpu_fused_denoise_graph_cache', '_rpu_runtime_device', '_rpu_adarms_cond_cache', '_rpu_denoise_mask_cache', '_rpu_siglip_batch_n', '_rpu_qwen3_5_text_install_started', '_rpu_qwen3_5', '_rpu_qwen3_5_moe_text_install_started', '_rpu_qwen3_5_moe', '_rpu_qwen3_5_vision_conversion_started', '_rpu_qwen3_5_vision_weights_converted', '_rpu_vision_w8a16', '_rpu_vision_math_precision', '_rpu_vision_k1_b3_available', '_rpu_vision_k1_camera_group_available', '_rpu_g05_quant_inventory', '_rpu_q_ws', '_rpu_k_ws', '_rpu_v_ws', '_rpu_o_ws', '_rpu_fc1_ws', '_rpu_fc2_ws', '_rpu_qwen3_5_had_instance_forward', '_rpu_qwen3_5_original_forward', '_rpu_q_w', '_rpu_k_w', '_rpu_v_w', '_rpu_q_b', '_rpu_k_b', '_rpu_v_b', '_rpu_o_b', '_rpu_fc1_b', '_rpu_fc2_b', '_rpu_vision_installing', '_rpu_vision_installing_handle', '_rpu_vision_installing_finalizer', '_rpu_vision_freq_cos', '_rpu_vision_freq_sin', '_rpu_vision_position_idx_keepalive', '_rpu_vision_position_idx_cpu', '_rpu_vision_position_idx_grid', '_rpu_vision_position_idx_cache_key', '_rpu_vision_step0', '_rpu_vision_patch_embed_w', '_rpu_vision_patch_embed_b', '_rpu_vision_pe_w_swz', '_rpu_vision_pe_b_rpu', '_rpu_vision_step0_pos', '_rpu_vision_step0_pos_grid', '_rpu_vision_step0_pos_temporal_num_frames', '_rpu_vision_step0_pos_camera_batch_count', '_rpu_vision_merger_refs', '_rpu_vision_has_merger', '_rpu_vision_kv_cache', '_rpu_vision_graph_disable', '_rpu_vision_rope_disable', '_rpu_vision_linear_acc32', '_rpu_vision_gelu_erf_mode', '_rpu_vision_w8a16', '_rpu_vision_math_precision', '_rpu_vision_k1_b3_available', '_rpu_vision_k1_camera_group_available', '_rpu_g05_quant_inventory', '_rpu_vision_graph_cache', '_rpu_vision_graph_key', '_rpu_vision_graph_sig', '_rpu_vision_debug_graph', '_rpu_vision_spatial_merge_size', '_rpu_vision_num_layers', '_rpu_vision_hidden_size', '_rpu_vision_chunk_size_cap', '_rpu_vision_exact_chunk_size', '_rpu_vision_padding_rows', '_rpu_vision_padding_budget', '_rpu_vision_control_snapshot', '_rpu_vision_last_a6_plan', '_rpu_vision_a6_plans', '_rpu_vision_has_dispatched'})

def test_validate_preinstall_rejects_transitional():
    from rpu_backend.runtime.hw_attrs import validate_preinstall
    from rpu_backend.api.errors import RPUConfigError
    m = nn.Linear(16, 16)
    m._rpu_swizzled = True
    with pytest.raises(RPUConfigError, match='_rpu_swizzled'):
        validate_preinstall(m)

def test_validate_postinstall_accepts_transitional():
    from rpu_backend.runtime.hw_attrs import validate_postinstall
    m = nn.Linear(16, 16)
    m._rpu_swizzled = True
    m._rpu_chunk_size = 160
    validate_postinstall(m)

def test_validate_postinstall_accepts_prefill_execution_alignment():
    from rpu_backend.runtime.hw_attrs import validate_postinstall
    m = nn.Linear(16, 16)
    m._rpu_prefill_execution_alignment = 16
    validate_postinstall(m)

def test_native_decode_descriptor_is_internal_install_state_not_public_input():
    from rpu_backend.runtime.hw_attrs import validate_preinstall, validate_postinstall, install_hw_attr_validator
    from rpu_backend.api.errors import RPUConfigError
    model = nn.Linear(16, 16)
    model._rpu_decode_stage_descriptor = (3, 1)
    with pytest.raises(RPUConfigError, match='_rpu_decode_stage_descriptor'):
        validate_preinstall(model)
    validate_postinstall(model)
    install_hw_attr_validator(model)
    model._rpu_decode_stage_descriptor = (3, 2)
    assert model._rpu_decode_stage_descriptor == (3, 2)

def test_unknown_rpu_attr_raises():
    from rpu_backend.runtime.hw_attrs import validate_existing_hw_attrs
    from rpu_backend.api.errors import RPUConfigError
    m = nn.Linear(16, 16)
    m._rpu_unexpected_name = 'bad'
    with pytest.raises(RPUConfigError, match='_rpu_unexpected_name'):
        validate_existing_hw_attrs(m)

def test_cold_set_on_public_attr_succeeds():
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator
    m = nn.Linear(16, 16)
    m._rpu_chunk_size = 160
    install_hw_attr_validator(m)

def test_post_install_public_attr_write_succeeds_pre_first_forward():
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator
    m = nn.Linear(16, 16)
    install_hw_attr_validator(m)
    m._rpu_chunk_size = 128
    m._rpu_chunk_size = 64
    assert m._rpu_chunk_size == 64

@pytest.mark.parametrize('attr_name', ['_rpu_execution', '_rpu_spm_mode', '_rpu_warmup'])
def test_cold_only_attr_post_first_forward_raises(attr_name):
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator, mark_first_forward_done
    from rpu_backend.api.errors import RPUConfigError
    m = nn.Linear(16, 16)
    install_hw_attr_validator(m)
    setattr(m, attr_name, False)
    mark_first_forward_done(m)
    with pytest.raises(RPUConfigError, match=attr_name):
        setattr(m, attr_name, True)

@pytest.mark.parametrize('attr_name', ['_rpu_chunk_size', '_rpu_debug_export'])
def test_hot_settable_attr_post_first_forward_succeeds(attr_name):
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator, mark_first_forward_done
    m = nn.Linear(16, 16)
    install_hw_attr_validator(m)
    mark_first_forward_done(m)
    setattr(m, attr_name, 128 if attr_name == '_rpu_chunk_size' else True)
    setattr(m, attr_name, 64 if attr_name == '_rpu_chunk_size' else False)

def test_pi05_planned_prefill_handoff_is_admitted_runtime_state():
    from rpu_backend.runtime.hw_attrs import INTERNAL_HW_ATTRS_TRANSITIONAL, install_hw_attr_validator, mark_first_forward_done
    name = '_rpu_planned_prefill_plan'
    assert name in INTERNAL_HW_ATTRS_TRANSITIONAL
    m = nn.Linear(16, 16)
    install_hw_attr_validator(m)
    mark_first_forward_done(m)
    setattr(m, name, 64)
    setattr(m, name, 128)

@pytest.mark.parametrize('operation', ('set', 'delete'))
def test_attn_tp_internal_install_input_is_immutable_after_publication(operation):
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator, validate_postinstall
    from rpu_backend.api.errors import RPUConfigError
    model = nn.Linear(16, 16)
    model._rpu_attn_tp = 4
    validate_postinstall(model)
    install_hw_attr_validator(model)
    with pytest.raises(RPUConfigError, match='_rpu_attn_tp'):
        if operation == 'set':
            model._rpu_attn_tp = 2
        else:
            del model._rpu_attn_tp
    assert model._rpu_attn_tp == 4

def test_monotonic_internal_attr_post_install_raises():
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator
    from rpu_backend.api.errors import RPUConfigError
    m = nn.Linear(16, 16)
    m._rpu_swizzled = True
    install_hw_attr_validator(m)
    with pytest.raises(RPUConfigError, match='_rpu_swizzled'):
        m._rpu_swizzled = False

def test_unknown_post_install_write_raises():
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator
    from rpu_backend.api.errors import RPUConfigError
    m = nn.Linear(16, 16)
    install_hw_attr_validator(m)
    with pytest.raises(RPUConfigError, match='_rpu_unknown'):
        m._rpu_unknown = 'bad'

@pytest.mark.parametrize('name,value', [('_rpu_linear_partition', 1), ('_rpu_linear_num_cores', 8)])
@pytest.mark.parametrize('operation', ['assign', 'delete'])
@pytest.mark.parametrize('hot', [False, True])
def test_fixed_linear_layout_cannot_reauthorize_skipped_weight(name, value, operation, hot):
    from types import SimpleNamespace
    from rpu_backend.api.errors import RPUConfigError
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator, mark_first_forward_done
    from rpu_backend.runtime.weights import _record_linear_layout, SKIPPED_PARTITION
    layer = nn.Linear(16, 16, bias=False).half()
    cpu_input = torch.ones(1, 16, dtype=torch.float16)
    expected = layer(cpu_input)
    _record_linear_layout(layer, SKIPPED_PARTITION, None)
    install_hw_attr_validator(layer)
    if hot:
        mark_first_forward_done(layer)
    with pytest.raises(RPUConfigError, match=name):
        if operation == 'assign':
            setattr(layer, name, value)
        else:
            delattr(layer, name)
    assert torch.equal(layer(cpu_input), expected)
    with pytest.raises(RuntimeError, match='SKIP_LINEAR_NAMES'):
        layer(SimpleNamespace(device=SimpleNamespace(type='rpu')))

def test_qwen3_5_lazy_vision_state_is_admitted_but_typos_are_not():
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator, mark_first_forward_done
    from rpu_backend.api.errors import RPUConfigError
    root = nn.Module()
    root.visual = nn.Module()
    install_hw_attr_validator(root)
    mark_first_forward_done(root)
    root._rpu_qwen3_5 = object()
    root.visual._rpu_qwen3_5_vision_conversion_started = True
    root.visual._rpu_vision_graph_cache = object()
    root.visual._rpu_vision_step0_pos = None
    root.visual._rpu_vision_step0_pos_temporal_num_frames = None
    root.visual._rpu_vision_step0_pos_camera_batch_count = None
    root.visual._rpu_vision_chunk_size_cap = 0
    root.visual._rpu_vision_exact_chunk_size = None
    root.visual._rpu_vision_padding_rows = 'auto'
    root.visual._rpu_vision_padding_budget = 0
    root.visual._rpu_vision_control_snapshot = {}
    root.visual._rpu_vision_last_a6_plan = None
    root.visual._rpu_vision_a6_plans = {}
    root.visual._rpu_vision_linear_acc32 = True
    root.visual._rpu_vision_gelu_erf_mode = 'ultra'
    root.visual._rpu_vision_w8a16 = False
    root.visual._rpu_vision_math_precision = 'high'
    root.visual._rpu_vision_k1_b3_available = False
    root.visual._rpu_vision_k1_camera_group_available = False
    root.visual._rpu_g05_quant_inventory = []
    del root.visual._rpu_vision_math_precision
    with pytest.raises(RPUConfigError, match='_rpu_vision_graph_cahce'):
        root.visual._rpu_vision_graph_cahce = object()

def test_qwen3_5_lazy_vision_state_inventory_is_admitted():
    from rpu_backend.runtime.hw_attrs import INTERNAL_HW_ATTRS_TRANSITIONAL, PUBLIC_HW_ATTRS
    path = ROOT / 'python/rpu_backend/adapters/qwen3_5/vision.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    writes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.attr.startswith('_rpu_')}
    assert writes <= INTERNAL_HW_ATTRS_TRANSITIONAL | PUBLIC_HW_ATTRS

def test_qwen3_5_adapter_uses_the_full_a9_lifecycle():
    source = (ROOT / 'python' / 'rpu_backend' / 'adapters' / 'qwen3_5' / '__init__.py').read_text()
    install = source[source.index('def to_rpu(self, max_seq_len=None):'):source.index('def _rpu_qwen3_5_forward')]
    forward = source[source.index('def _rpu_qwen3_5_forward'):]
    assert install.index('validate_preinstall(model)') < install.index('_claim_live_instance(model)')
    assert install.index('validate_postinstall(model)') < install.index('install_hw_attr_validator(model)') < install.index('model._rpu_swizzled = True')
    assert 'mark_first_forward_done(self)' in forward

def test_rpu_cache_is_mutable_runtime():
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator
    m = nn.Linear(16, 16)
    install_hw_attr_validator(m)
    m._rpu_cache = 'first'
    m._rpu_cache = 'second'
    assert m._rpu_cache == 'second'

def test_child_module_sees_root_state_after_mark():
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator, mark_first_forward_done, _MODULE_STATE
    from rpu_backend.api.errors import RPUConfigError
    root = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16))
    install_hw_attr_validator(root)
    child = root[0]
    assert child in _MODULE_STATE, 'H4 regression: child not in _MODULE_STATE'
    assert _MODULE_STATE[child] is _MODULE_STATE[root], 'H4 regression: child has distinct state object (iter-1 bug)'
    mark_first_forward_done(root)
    with pytest.raises(RPUConfigError, match='_rpu_spm_mode'):
        child._rpu_spm_mode = False

def test_slotted_module_install_raises():
    from unittest.mock import patch
    from rpu_backend.runtime import hw_attrs as _hw
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator
    from rpu_backend.api.errors import RPUConfigError

    def _synthetic_typeerror(module, shared_state):
        _hw._MODULE_STATE[module] = shared_state
        try:
            raise TypeError('__class__ assignment: synthetic layout differs from Validated_nnLinear (C-backed or slotted source)')
        except TypeError as exc:
            _hw._raise_config_error(f'Cannot install A9 validator on {type(module).__name__} (module={module!r}): {exc}. This module type does not accept __class__ reassignment (likely __slots__-bearing, C-backed, or frozen).')
    m = nn.Linear(16, 16)
    with patch.object(_hw, '_install_on_one', _synthetic_typeerror):
        with pytest.raises(RPUConfigError, match='__class__|layout|slots|C-backed'):
            install_hw_attr_validator(m)
    m2 = nn.Linear(16, 16)
    install_hw_attr_validator(m2)

def test_validated_class_survives_deepcopy():
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator
    m = nn.Linear(16, 16)
    install_hw_attr_validator(m)
    assert type(m).__rpu_hw_attr_validated__ is True
    m2 = copy.deepcopy(m)
    assert getattr(type(m2), '__rpu_hw_attr_validated__', False) is True, 'M3: Validated_* class did not survive deepcopy'

def test_validated_class_state_dict_roundtrip():
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator
    m1 = nn.Linear(16, 16)
    install_hw_attr_validator(m1)
    sd = m1.state_dict()
    m2 = nn.Linear(16, 16)
    m2.load_state_dict(sd)
    assert torch.allclose(m1.weight, m2.weight)

def main() -> int:
    test_public_hw_attrs_frozen()
    test_cold_only_attrs_frozen()
    test_internal_transitional_frozen()
    test_validate_preinstall_rejects_transitional()
    test_validate_postinstall_accepts_transitional()
    test_unknown_rpu_attr_raises()
    test_cold_set_on_public_attr_succeeds()
    test_post_install_public_attr_write_succeeds_pre_first_forward()
    for attr in ('_rpu_spm_mode', '_rpu_warmup'):
        test_cold_only_attr_post_first_forward_raises(attr)
    for attr in ('_rpu_chunk_size', '_rpu_debug_export'):
        test_hot_settable_attr_post_first_forward_succeeds(attr)
    test_monotonic_internal_attr_post_install_raises()
    test_unknown_post_install_write_raises()
    test_rpu_cache_is_mutable_runtime()
    test_child_module_sees_root_state_after_mark()
    test_slotted_module_install_raises()
    test_validated_class_survives_deepcopy()
    test_validated_class_state_dict_roundtrip()
    print('PASS: A9 hw-attr validator contract (15+ scenarios; H4/H5/L2/M3 reviews iter-2)')
    return 0
if __name__ == '__main__':
    sys.exit(main())
