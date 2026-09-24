"""Public quantized configs must select the declared recipe before model loading."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def common(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'examples'))
    return load('_common', 'examples/_common.py', monkeypatch)


def config_path(size, bits, mode):
    return ROOT / f'examples/configs/qwen3_vl/{mode}/{size}b/w{bits}a16.toml'


@pytest.mark.parametrize('size', [2, 4, 8, 32])
@pytest.mark.parametrize('bits', [8, 4])
@pytest.mark.parametrize('mode', ['text', 'vl'])
def test_runtime_quant_config_binds_asset_recipe_and_cold_accumulation(common, size, bits, mode, tmp_path):
    config = common.load_config(config_path(size, bits, mode))
    profile = common.example_profiles()[config['example']['profile_id']]
    expected = profile['input_envelope']['limits']['checkpoint_quant_profile']
    suffix = '_awq' if bits == 4 and size != 32 else ''
    assert expected == f'qwen3_vl_{size}b_w{bits}a16{suffix}_runtime_align_v1'
    assert config['input']['decode_steps'] == 31  # 32 generated tokens including prefill
    assert config['rpu_execution']['prefill']['fast_replay'] is True
    assert config['rpu_execution']['prefill']['linear_acc32'] is False
    assert config['rpu_execution']['vision']['linear_acc32'] is False
    (tmp_path / 'config.json').write_text(json.dumps({
        'quant_config': {'profile': expected, 'method': f'w{bits}a16'}}))
    common.validate_checkpoint_profile(config, str(tmp_path))
    # Both component overrides are admitted cold, without changing weight precision.
    request = copy.deepcopy(config['rpu_execution'])
    request['components'] = {
        'language_model': {'prefill': {'linear_acc32': True}},
        'vision_encoder': {'vision': {'linear_acc32': True}},
    }
    common._validate_execution(request, profile=profile)
    request['model']['num_cores'] = 4
    with pytest.raises((common.ConfigError, ValueError), match='4/6|eight|8|FP16'):
        common._validate_execution(request, profile=profile)


@pytest.mark.parametrize('quant', [
    None,
    {'method': 'w4a16', 'profile': 'qwen3_vl_4b_w4a16_runtime_align_v1'},
    {'method': 'w4a16', 'profile': 'qwen3_vl_2b_w4a16_awq_runtime_align_v1'},
    {'method': 'w8a16', 'profile': 'qwen3_vl_4b_w4a16_awq_runtime_align_v1'},
])
def test_wrong_asset_recipe_fails_before_processor_or_weight_load(common, monkeypatch, tmp_path, quant):
    runner = load('quantized_example_runner', 'examples/_text_vision_runners.py', monkeypatch)
    config = common.load_config(config_path(4, 4, 'vl'))
    (tmp_path / 'config.json').write_text(json.dumps({'quant_config': quant}))
    loader, processor = install_stubs(monkeypatch, tmp_path)
    with pytest.raises(common.ConfigError, match='checkpoint quant_config.profile'):
        runner._prepare_text_vision(config, SimpleNamespace(owner=None))
    loader.assert_not_called()
    processor.assert_not_called()


def install_stubs(monkeypatch, checkpoint):
    from transformers import AutoConfig

    model = SimpleNamespace(model=SimpleNamespace(language_model=object()))
    loader = Mock(return_value=model)
    processor = Mock()
    for name, values in {
        'transformers': {'AutoConfig': AutoConfig,
                         'AutoTokenizer': SimpleNamespace(from_pretrained=Mock()),
                         'AutoProcessor': SimpleNamespace(from_pretrained=processor)},
        'rpu_backend.model_registry': {'model_path': lambda _: checkpoint},
        'rpu_backend.api': {
            'RPUCache': SimpleNamespace(from_model=Mock(return_value=object())),
            'RPUModelForConditionalGeneration': SimpleNamespace(from_pretrained=loader)},
    }.items():
        module = ModuleType(name)
        vars(module).update(values)
        monkeypatch.setitem(sys.modules, name, module)
    return loader, processor


def original_quant_config(size, bits):
    from rpu_backend.quant.load import _QWEN3_VL_2B_W8A16_QUANT_CONFIG
    from test_qwen3_vl_awq_packing import exact_config

    cfg = exact_config(size)
    if bits == 8:
        del cfg.quantization_config
        cfg.quant_config = copy.deepcopy(_QWEN3_VL_2B_W8A16_QUANT_CONFIG)
    return cfg


@pytest.mark.parametrize('size,bits', [(2, 8), (2, 4), (4, 4)])
@pytest.mark.parametrize('mode', ['text', 'vl'])
def test_original_text_quant_configs_validate_actual_metadata(common, tmp_path, size, bits, mode):
    variant = 'w8a16_text' if bits == 8 else 'w4a16_text_awq'
    config = common.load_config(ROOT / f'examples/configs/qwen3_vl/{mode}/{size}b/{variant}.toml')
    cfg = original_quant_config(size, bits)
    cfg.to_json_file(tmp_path / 'config.json')
    common.validate_checkpoint_profile(config, str(tmp_path))
    # A valid but different recipe/size cannot inherit this example's label.
    wrong = original_quant_config(4 if size == 2 else 2, 4)
    wrong.to_json_file(tmp_path / 'config.json')
    with pytest.raises(common.ConfigError, match='checkpoint format='):
        common.validate_checkpoint_profile(config, str(tmp_path))


def test_original_awq_wrong_recipe_fails_before_processor_or_weights(common, monkeypatch, tmp_path):
    runner = load('original_quantized_runner', 'examples/_text_vision_runners.py', monkeypatch)
    config = common.load_config(ROOT / 'examples/configs/qwen3_vl/vl/4b/w4a16_text_awq.toml')
    cfg = original_quant_config(4, 4)
    cfg.quantization_config['config_groups']['group_0']['weights']['group_size'] = 64
    cfg.to_json_file(tmp_path / 'config.json')
    loader, processor = install_stubs(monkeypatch, tmp_path)
    with pytest.raises(common.ConfigError, match='checkpoint format='):
        runner._prepare_text_vision(config, SimpleNamespace(owner=None))
    loader.assert_not_called()
    processor.assert_not_called()


@pytest.mark.parametrize('mode', ['text', 'vl'])
@pytest.mark.parametrize('acc32', [False, True])
def test_public_quantized_prepare_uses_existing_loader_and_preserves_config(common, monkeypatch, tmp_path, mode, acc32):
    runner = load('quantized_prepare_runner', 'examples/_text_vision_runners.py', monkeypatch)
    config = common.load_config(config_path(8, 4, mode))
    for stage in ('prefill', 'vision'):
        config['rpu_execution'][stage]['linear_acc32'] = acc32
    expected = common.example_profiles()[config['example']['profile_id']]['input_envelope']['limits']['checkpoint_quant_profile']
    (tmp_path / 'config.json').write_text(json.dumps({
        'quant_config': {'profile': expected, 'method': 'w4a16'}}))
    loader, processor = install_stubs(monkeypatch, tmp_path)
    ids = torch.ones((1, 16), dtype=torch.long)
    processor.return_value.apply_chat_template.return_value = {'input_ids': ids}
    monkeypatch.setattr(runner, '_text_input_ids', lambda *_: ids)
    monkeypatch.setattr(runner, '_image_inputs', lambda *_: [object()])
    monkeypatch.setattr(runner, '_extend_multimodal_input', lambda encoded, *_: encoded)
    generate = Mock(return_value={'token_ids': ids})
    monkeypatch.setattr(runner, '_generate', generate)
    state = SimpleNamespace(owner=None)
    infer, owner = runner._prepare_text_vision(config, state)
    loader.assert_called_once_with(str(tmp_path), dtype=torch.float16, device='rpu',
        rpu_execution=config['rpu_execution'], local_files_only=True, trust_remote_code=False)
    assert owner is state.owner
    infer()
    assert generate.call_args.args[3] == 31
    assert generate.call_args.kwargs['stop_on_eos'] is False


@pytest.mark.parametrize('omit_dtype', [False, True])
@pytest.mark.parametrize('recipe', ['w4a16', 'w4a16_lm_head'])
def test_qwen3_w4_public_runner_selects_cold_quantization_without_changing_asset(common, monkeypatch, omit_dtype, recipe):
    runner = load('plain_qwen_w4_example_runner', 'examples/_text_vision_runners.py', monkeypatch)
    config = common.load_config(ROOT / f'examples/configs/qwen3/text/0_6b/{recipe}.toml')
    if omit_dtype:
        del config['example']['dtype']
    checkpoint = '/cache/Qwen3-0.6B'
    model = object()
    load_model = Mock(return_value=model)
    tokenizer = object()
    limits = common.example_profiles()[config['example']['profile_id']]['input_envelope']['limits']
    for name, values in {
        'transformers': {
            'AutoConfig': SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(**limits))),
            'AutoTokenizer': SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))},
        'rpu_backend': {
            'RPUCache': SimpleNamespace(from_model=Mock(return_value=object())),
            'RPUModelForCausalLM': SimpleNamespace(from_pretrained=load_model)},
        'rpu_backend.model_registry': {'model_path': Mock(return_value=checkpoint)},
    }.items():
        module = ModuleType(name)
        vars(module).update(values)
        monkeypatch.setitem(sys.modules, name, module)
    ids = torch.ones((1, 128), dtype=torch.long)
    monkeypatch.setattr(runner, '_text_input_ids', lambda *_: ids)
    generate = Mock()
    monkeypatch.setattr(runner, '_generate', generate)
    infer, owner = runner._prepare_text_vision(config, SimpleNamespace(owner=None))
    assert owner is model
    load_model.assert_called_once_with(checkpoint, dtype=torch.float16, device='rpu',
        rpu_execution=config['rpu_execution'], local_files_only=True, trust_remote_code=False,
        quantization=recipe)
    infer()
    assert generate.call_args.args[3] == 7


@pytest.mark.parametrize('declared,quant', [
    ('fp16', None), ('w8a16', {'method': 'w8a16'}),
    ('w4a16', {'method': 'w4a16_pgrp', 'group_size': 32}),
])
@pytest.mark.parametrize('omit_dtype', [False, True])
@pytest.mark.parametrize('component', ['text', 'vl'])
def test_qwen35_public_precision_check_uses_actual_quant_scope(common, monkeypatch, declared, quant, omit_dtype, component):
    config = common.load_config(ROOT / f'examples/configs/qwen3_5/{component}/0_8b/{declared}.toml')
    if omit_dtype:
        del config['example']['dtype']
    metadata = SimpleNamespace(architectures=['Qwen3_5ForConditionalGeneration'],
        text_config=SimpleNamespace(layer_types=['linear_attention', 'full_attention']), quant_config=quant)
    constructor = Mock(return_value=metadata)
    transformers = ModuleType('transformers')
    transformers.AutoConfig = SimpleNamespace(from_pretrained=constructor)
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    assert common.validate_qwen35_checkpoint(config, '/caller/model') is metadata
    constructor.assert_called_once_with('/caller/model', local_files_only=True, trust_remote_code=False)


@pytest.mark.parametrize('declared,quant,architecture,match', [
    ('fp16', {'method': 'w8a16'}, 'Qwen3_5ForConditionalGeneration', 'precision'),
    ('w8a16', None, 'Qwen3_5ForConditionalGeneration', 'precision'),
    ('w8a16', {'method': 'w4a16_pgrp', 'group_size': 32}, 'Qwen3_5ForConditionalGeneration', 'precision'),
    ('w4a16', {'method': 'w8a16'}, 'Qwen3_5ForConditionalGeneration', 'precision'),
    ('fp16', None, 'Qwen3_5MoeForConditionalGeneration', 'architecture'),
])
@pytest.mark.parametrize('component', ['text', 'vl'])
def test_qwen35_label_mismatch_fails_before_processor_or_weights(
        common, monkeypatch, declared, quant, architecture, match, component):
    config = common.load_config(ROOT / f'examples/configs/qwen3_5/{component}/0_8b/{declared}.toml')
    metadata = SimpleNamespace(architectures=[architecture],
        text_config=SimpleNamespace(layer_types=['linear_attention', 'full_attention']), quant_config=quant)
    tokenizer, loader = Mock(), Mock()
    for name, values in {
        'transformers': {'AutoConfig': SimpleNamespace(from_pretrained=Mock(return_value=metadata)),
                         'AutoProcessor': SimpleNamespace(from_pretrained=tokenizer),
                         'AutoTokenizer': SimpleNamespace(from_pretrained=tokenizer)},
        'rpu_backend.model_registry': {'model_path': lambda alias: Path('/caller/checkpoint')},
        'rpu_backend.api': {'RPUModelForConditionalGeneration': SimpleNamespace(from_pretrained=loader)},
    }.items():
        module = ModuleType(name)
        vars(module).update(values)
        monkeypatch.setitem(sys.modules, name, module)
    runner = load('qwen35_declared_precision_runner',
                  'examples/_qwen35_vision.py' if component == 'vl' else 'examples/_text_vision_runners.py',
                  monkeypatch)
    with pytest.raises(common.ConfigError, match=match):
        if component == 'vl':
            runner._qwen35_vision(SimpleNamespace(source_config=config, owner=None))
        else:
            runner._prepare_text_vision(config, SimpleNamespace(owner=None))
    tokenizer.assert_not_called()
    loader.assert_not_called()


@pytest.mark.parametrize('precision,quant', [
    ('w8a16', {'method': 'w8a16'}),
    ('w4a16', {'method': 'w4a16_pgrp', 'group_size': 32}),
])
def test_qwen35_vl_quantized_prepare_keeps_public_loader_and_image_request(
        common, monkeypatch, precision, quant):
    monkeypatch.syspath_prepend(str(ROOT / 'examples'))
    runner = load('qwen35_quantized_vision_runner', 'examples/_qwen35_vision.py', monkeypatch)
    import _text_vision_runners

    config = common.load_config(ROOT / f'examples/configs/qwen3_5/vl/2b/{precision}.toml')
    checkpoint = '/caller/quantized-checkpoint'
    metadata = SimpleNamespace(architectures=['Qwen3_5ForConditionalGeneration'],
        text_config=SimpleNamespace(layer_types=['linear_attention', 'full_attention']), quant_config=quant)
    model, cache, device_pixels = object(), object(), object()
    loader = Mock(return_value=model)
    pixel_values = SimpleNamespace(to=Mock(return_value=device_pixels))
    ids = torch.ones((1, 16), dtype=torch.long)
    grid = torch.tensor([[1, 2, 2]])
    mm_types = torch.zeros_like(ids)
    processor = Mock(return_value={'input_ids': ids, 'pixel_values': pixel_values,
                                  'image_grid_thw': grid, 'mm_token_type_ids': mm_types})
    processor.apply_chat_template.return_value = 'image prompt'
    for name, values in {
        'transformers': {'AutoConfig': SimpleNamespace(from_pretrained=Mock(return_value=metadata)),
                         'AutoProcessor': SimpleNamespace(from_pretrained=Mock(return_value=processor))},
        'rpu_backend.api': {'RPUModelForConditionalGeneration': SimpleNamespace(from_pretrained=loader)},
        'rpu_backend.api.qwen3_5_cache': {'Qwen3_5Cache': SimpleNamespace(from_model=Mock(return_value=cache))},
    }.items():
        module = ModuleType(name)
        vars(module).update(values)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runner, '_checkpoint', lambda _: checkpoint)
    monkeypatch.setattr(runner, '_images', lambda _: [object()])
    monkeypatch.setattr(_text_vision_runners, '_extend_multimodal_input', lambda encoded, *_: encoded)
    generate = Mock(return_value={'token_ids': ids})
    monkeypatch.setattr(_text_vision_runners, '_generate', generate)
    state = SimpleNamespace(source_config=config, owner=None)
    infer = runner._qwen35_vision(state)
    loader.assert_called_once_with(checkpoint, dtype=torch.float16, device='rpu',
        local_files_only=True, rpu_execution=config['rpu_execution'])
    assert state.owner is model
    pixel_values.to.assert_called_once_with('rpu')
    infer()
    args = generate.call_args.args
    assert args[:2] == (model, cache)
    assert args[2] is ids and args[3] == 32
    assert args[4]['pixel_values'] is device_pixels
    assert args[4]['image_grid_thw'] is grid and args[4]['mm_token_type_ids'] is mm_types
    assert generate.call_args.kwargs['stop_on_eos'] is False
