"""Public configs must select the MLP-only asset and reject the old uniform W4 asset."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from examples.common import config as common

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'examples/configs/qwen3_5/text/2b'


def metadata(profile='w4_mlp_only'):
    value = json.loads((ROOT / 'tests/fixtures/qwen35_quanttrio_awq/2b.json').read_text())
    del value['quantization_config']
    value['quant_config'] = {
        'format_version': 2, 'architecture': 'qwen3_5', 'profile': profile,
        'default': {'method': 'fp16' if profile == 'w4_mlp_only' else 'w8a16'},
        'projection_overrides': {
            f'mlp.{role}': {'method': 'w4a16_pgrp', 'group_size': 32}
            for role in ('gate_proj', 'up_proj', 'down_proj')},
    }
    return value


def test_public_configs_select_mlp_only_asset_and_real_prefix(monkeypatch, tmp_path):
    alias = 'qwen3_5-2b-w4-mlp-only-g32'
    spec = importlib.util.spec_from_file_location('registry_mlp_only', ROOT / 'python/rpu_backend/model_registry.py')
    registry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registry)
    monkeypatch.setenv('RPU_MODEL_CACHE', str(tmp_path))
    assert registry.model_path(alias) == tmp_path / 'qwen3_5-2b-w4a16-g32-mlp-only'
    assert registry.model_path('qwen3_5-2b-w4a16') == tmp_path / 'qwen3_5_quant_ckpts/qwen3_5-2b-w4a16'
    (tmp_path / 'config.json').write_text(json.dumps(metadata()))
    base = common.load_config(CONFIGS / 'w4a16_mlp_rtn_g32.toml')
    common.validate_checkpoint_profile(base, str(tmp_path))
    assert base['example']['registry_alias'] == alias
    assert base['input']['prompt'] and base['input']['decode_steps'] == 4
    assert base['run']['warmup'] == 2 and base['run']['runs'] == 5
    assert base['rpu_execution']['prefill']['linear_acc32'] is False


@pytest.mark.parametrize('change', ['uniform', 'g128', 'first_layer_fp16', 'wrong_size', 'wrong_architecture', 'wrong_profile'])
def test_mlp_only_profile_rejects_wrong_scope_before_weight_load(tmp_path, change):
    config = common.load_config(CONFIGS / 'w4a16_mlp_rtn_g32.toml')
    value = metadata()
    quant = value['quant_config']
    if change == 'uniform':
        quant.update(default={'method': 'w4a16_pgrp', 'group_size': 32}, projection_overrides={})
    elif change == 'g128':
        quant['projection_overrides']['mlp.down_proj']['group_size'] = 128
    elif change == 'first_layer_fp16':
        quant['layer_overrides'] = {'0': {'default': {'method': 'fp16'}}}
    elif change == 'wrong_size':
        value['text_config']['hidden_size'] = 2560
    elif change == 'wrong_architecture':
        value['architectures'] = ['Qwen3ForCausalLM']
    else:
        quant['profile'] = 'other'
    (tmp_path / 'config.json').write_text(json.dumps(value))
    with pytest.raises(common.ConfigError, match='MLP-only G32'):
        common.validate_checkpoint_profile(config, str(tmp_path))


def test_public_mixed_configs_bind_recipe_and_same_real_prefix(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location('registry_mixed', ROOT / 'python/rpu_backend/model_registry.py')
    registry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registry)
    alias = 'qwen3_5-2b-w4-mlp-w8-attn-gdn-g32'
    monkeypatch.setenv('RPU_MODEL_CACHE', str(tmp_path))
    assert registry.model_path(alias) == tmp_path / 'qwen3_5-2b-w4a16-g32-mlp-w8-attn-gdn'
    assert registry.model_path(alias) != registry.model_path('qwen3_5-2b-w4-mlp-only-g32')
    (tmp_path / 'config.json').write_text(json.dumps(metadata('w4_mlp_w8_attn_gdn')))
    config = common.load_config(CONFIGS / 'w4a16_mlp_rtn_g32_attn_gdn_w8a16.toml')
    baseline = common.load_config(CONFIGS / 'w4a16_mlp_rtn_g32.toml')
    common.validate_checkpoint_profile(config, str(tmp_path))
    assert config['example']['registry_alias'] == alias
    assert config['example']['dtype'] == 'w4a16'
    assert config['example']['opt_in'] == {}
    assert config['input'] == baseline['input']
    assert config['run']['warmup'] == 2 and config['run']['runs'] == 5
    assert config['run']['timing_mode'] == 'per_call'
    assert config['cpu']['threads'] == 4
    assert config['rpu_execution']['model']['num_cores'] == 8
    assert config['rpu_execution']['prefill']['linear_acc32'] is False
    # The MLP-only entry must not silently accept this new mixed checkpoint.
    with pytest.raises(common.ConfigError, match='MLP-only G32'):
        common.validate_checkpoint_profile(baseline, str(tmp_path))


@pytest.mark.parametrize('change', [
    'mlp_only', 'uniform', 'uniform_w8', 'g128', 'first_layer_fp16',
    'gdn_a_fp16', 'gdn_b_fp16', 'full_attn_fp16', 'wrong_size',
    'wrong_schedule', 'wrong_architecture', 'wrong_profile', 'nested_conflict',
    'wrong_storage', 'wrong_scale_dtype',
])
def test_mixed_profile_rejects_wrong_scope_before_weight_load(tmp_path, change):
    config = common.load_config(CONFIGS / 'w4a16_mlp_rtn_g32_attn_gdn_w8a16.toml')
    value = metadata('w4_mlp_w8_attn_gdn')
    quant = value['quant_config']
    if change == 'mlp_only':
        quant['default'] = {'method': 'fp16'}
    elif change == 'uniform':
        quant.update(default={'method': 'w4a16_pgrp', 'group_size': 32}, projection_overrides={})
    elif change == 'uniform_w8':
        quant['projection_overrides'] = {}
    elif change == 'g128':
        quant['projection_overrides']['mlp.gate_proj']['group_size'] = 128
    elif change == 'first_layer_fp16':
        quant['layer_overrides'] = {'0': {'default': {'method': 'fp16'}}}
    elif change == 'gdn_a_fp16':
        quant['projection_overrides']['linear_attn.in_proj_a'] = {'method': 'fp16'}
    elif change == 'gdn_b_fp16':
        quant['projection_overrides']['linear_attn.in_proj_b'] = {'method': 'fp16'}
    elif change == 'full_attn_fp16':
        quant['projection_overrides']['self_attn.q_proj'] = {'method': 'fp16'}
    elif change == 'wrong_size':
        value['text_config']['hidden_size'] = 2560
    elif change == 'wrong_schedule':
        value['text_config']['layer_types'][3] = 'linear_attention'
    elif change == 'wrong_architecture':
        value['architectures'] = ['Qwen3ForCausalLM']
    elif change == 'wrong_profile':
        quant['profile'] = 'w4_mlp_only'
    elif change == 'nested_conflict':
        value['text_config']['quant_config'] = dict(quant)
    elif change == 'wrong_storage':
        quant['projection_overrides']['linear_attn.in_proj_qkv'] = {
            'method': 'w8a16', 'storage_dtype': 'uint8'}
    else:
        quant['default']['scale_dtype'] = 'float32'
    (tmp_path / 'config.json').write_text(json.dumps(value))
    with pytest.raises(common.ConfigError, match=r'MLP W4/G32 \+ Attention/GDN W8'):
        common.validate_checkpoint_profile(config, str(tmp_path))
