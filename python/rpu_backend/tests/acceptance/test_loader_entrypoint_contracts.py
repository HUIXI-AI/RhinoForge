from __future__ import annotations
from types import SimpleNamespace
import pytest
import torch
from rpu_backend.api import causal_lm, conditional_generation
from rpu_backend.api._loading import resolve_config_architecture, resolve_loader_device
from rpu_backend.api.errors import UnsupportedModelError
from rpu_backend.quant import load as w8a16_load
_LOADERS = ((causal_lm, causal_lm.RPUModelForCausalLM, 'RPUModelForCausalLM.from_pretrained'), (conditional_generation, conditional_generation.RPUModelForConditionalGeneration, 'RPUModelForConditionalGeneration.from_pretrained'))

def _qwen3_vl_32b_w8_config():
    return SimpleNamespace(architectures=['Qwen3VLForConditionalGeneration'], model_type='qwen3_vl', tie_word_embeddings=False, quant_config={'method': 'w8a16', 'mode': 'per_channel_symmetric', 'qaxis': 0, 'skip_modules': ['lm_head'], 'quantized_lm_head': False, 'quantized_embed_tokens': False, 'lm_head_untied': False, 'embed_tokens_untied': False}, text_config=SimpleNamespace(model_type='qwen3_vl_text', hidden_size=5120, intermediate_size=25600, num_hidden_layers=64, num_attention_heads=64, num_key_value_heads=8, head_dim=128, vocab_size=151936, hidden_act='silu', rms_norm_eps=1e-06), vision_config=SimpleNamespace(model_type='qwen3_vl', hidden_size=1152, intermediate_size=4304, depth=27, num_heads=16, patch_size=16, temporal_patch_size=2, spatial_merge_size=2, deepstack_visual_indexes=[8, 16, 24], out_hidden_size=5120, hidden_act='gelu_pytorch_tanh'))

@pytest.mark.parametrize('module,loader,entry_point', _LOADERS)
@pytest.mark.parametrize('device', ['cuda', 'rpu:1', 'rpu-ish', 0])
def test_invalid_device_fails_before_config_load(monkeypatch, module, loader, entry_point, device):

    def unexpected_config_load(*args, **kwargs):
        raise AssertionError('invalid device reached AutoConfig')
    monkeypatch.setattr(module.AutoConfig, 'from_pretrained', unexpected_config_load)
    with pytest.raises(ValueError, match=entry_point):
        loader.from_pretrained('unused/path', device=device)

@pytest.mark.parametrize('device,move_to_rpu', [(None, False), ('cpu', False), ('cpu:0', False), (torch.device('cpu'), False), ('rpu', True), ('rpu:0', True), (torch.device('rpu'), True)])
def test_valid_loader_devices(device, move_to_rpu):
    assert resolve_loader_device(device, entry_point='test_loader') is move_to_rpu

@pytest.mark.parametrize('architectures', [None, [], (), 'Qwen3ForCausalLM', [''], [None], ['Qwen3ForCausalLM', '']])
def test_malformed_architectures_raise_backend_error(architectures):
    config = SimpleNamespace(architectures=architectures)
    with pytest.raises(UnsupportedModelError, match='non-empty list'):
        resolve_config_architecture(config, source='unused/path', entry_point='RPUModelForCausalLM.from_pretrained', supported_builtins={'Qwen3ForCausalLM', 'LlamaForCausalLM'})

@pytest.mark.parametrize('module,loader,architecture,expected_route', [(causal_lm, causal_lm.RPUModelForCausalLM, 'Qwen3VLForConditionalGeneration', 'RPUModelForConditionalGeneration'), (causal_lm, causal_lm.RPUModelForCausalLM, 'Gemma4ForConditionalGeneration', 'Gemma4Adapter'), (causal_lm, causal_lm.RPUModelForCausalLM, 'Qwen3_5ForConditionalGeneration', 'RPUModelForConditionalGeneration'), (conditional_generation, conditional_generation.RPUModelForConditionalGeneration, 'Qwen3ForCausalLM', 'RPUModelForCausalLM')])
def test_wrong_builtin_entry_point_fails_before_adapter_import_and_weight_load(monkeypatch, module, loader, architecture, expected_route):
    config = SimpleNamespace(architectures=[architecture])
    monkeypatch.setattr(module.AutoConfig, 'from_pretrained', lambda *a, **k: config)

    def unexpected_adapter_lookup(*args, **kwargs):
        raise AssertionError('wrong built-in entry point reached adapter lookup')

    def unexpected_model_load(*args, **kwargs):
        raise AssertionError('wrong built-in entry point reached model load')
    monkeypatch.setattr(module, 'get_adapter', unexpected_adapter_lookup)
    model_loader = module.AutoModelForCausalLM if module is causal_lm else module.AutoModelForImageTextToText
    monkeypatch.setattr(model_loader, 'from_pretrained', unexpected_model_load)
    with pytest.raises(UnsupportedModelError, match=expected_route):
        loader.from_pretrained('unused/path')

def test_unknown_architecture_remains_available_to_external_plugins():
    architecture = resolve_config_architecture(SimpleNamespace(architectures=['VendorModelForCausalLM']), source='vendor/model', entry_point='RPUModelForCausalLM.from_pretrained', supported_builtins={'Qwen3ForCausalLM', 'LlamaForCausalLM'})
    assert architecture == 'VendorModelForCausalLM'

def test_supported_builtin_architectures_keep_their_public_routes():
    assert resolve_config_architecture(SimpleNamespace(architectures=['Qwen3ForCausalLM']), source='qwen3', entry_point='RPUModelForCausalLM.from_pretrained', supported_builtins=causal_lm.RPUModelForCausalLM._SUPPORTED_BUILTIN_ARCHITECTURES) == 'Qwen3ForCausalLM'
    assert resolve_config_architecture(SimpleNamespace(architectures=['Qwen3VLForConditionalGeneration']), source='qwen3-vl', entry_point='RPUModelForConditionalGeneration.from_pretrained', supported_builtins=conditional_generation.RPUModelForConditionalGeneration._SUPPORTED_BUILTIN_ARCHITECTURES) == 'Qwen3VLForConditionalGeneration'
    assert resolve_config_architecture(SimpleNamespace(architectures=['Qwen3_5ForConditionalGeneration']), source='qwen3.5', entry_point='RPUModelForConditionalGeneration.from_pretrained', supported_builtins=conditional_generation.RPUModelForConditionalGeneration._SUPPORTED_BUILTIN_ARCHITECTURES) == 'Qwen3_5ForConditionalGeneration'

def test_conditional_preflight_blocks_before_w8_loader_or_rpu(monkeypatch):
    events = []
    config = _qwen3_vl_32b_w8_config()

    class BlockedAdapter:

        @classmethod
        def preflight(cls, seen_config):
            assert seen_config is config
            events.append('preflight')
            raise UnsupportedModelError('graph-blocked')

        def __init__(self, _model):
            raise AssertionError('blocked profile constructed adapter')
    monkeypatch.setattr(conditional_generation.AutoConfig, 'from_pretrained', lambda *_args, **_kwargs: events.append('config') or config)
    monkeypatch.setattr(conditional_generation, 'get_adapter', lambda _arch: events.append('adapter-lookup') or BlockedAdapter)
    monkeypatch.setattr(w8a16_load, 'load_w8a16_imagetext', lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('blocked profile opened checkpoint')))
    with pytest.raises(UnsupportedModelError, match='graph-blocked'):
        conditional_generation.RPUModelForConditionalGeneration.from_pretrained('unused/path', device='rpu')
    assert events == ['config', 'adapter-lookup', 'preflight']

def test_conditional_w8_loader_rejects_mismatched_profile(monkeypatch):
    config = _qwen3_vl_32b_w8_config()
    config.text_config.hidden_size = 2048

    class Adapter:

        @classmethod
        def preflight(cls, _config):
            return None
    monkeypatch.setattr(conditional_generation.AutoConfig, 'from_pretrained', lambda *_a, **_k: config)
    monkeypatch.setattr(conditional_generation, 'get_adapter', lambda _arch: Adapter)
    monkeypatch.setattr(w8a16_load, 'load_w8a16_imagetext', lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('non-exact profile reached W8 loader')))
    monkeypatch.setattr(conditional_generation.AutoModelForImageTextToText, 'from_pretrained', lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('W8 config fell through to generic HF loader')))
    with pytest.raises(UnsupportedModelError, match='W8A16 image-text loading requires'):
        conditional_generation.RPUModelForConditionalGeneration.from_pretrained('unused/path')

@pytest.mark.parametrize('loader_kwarg', [{'variant': 'fp16'}, {'output_loading_info': True}, {'use_safetensors': True}])
def test_conditional_exact_32b_w8_rejects_unsupported_loader_kwargs(monkeypatch, loader_kwarg):
    config = _qwen3_vl_32b_w8_config()

    class Adapter:

        @classmethod
        def preflight(cls, _config):
            return None
    monkeypatch.setattr(conditional_generation.AutoConfig, 'from_pretrained', lambda *_a, **_k: config)
    monkeypatch.setattr(conditional_generation, 'get_adapter', lambda _arch: Adapter)
    monkeypatch.setattr(w8a16_load, 'load_w8a16_imagetext', lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('unsupported kwarg reached W8 loader')))
    key = next(iter(loader_kwarg))
    with pytest.raises(ValueError, match=key):
        conditional_generation.RPUModelForConditionalGeneration.from_pretrained('unused/path', **loader_kwarg)

@pytest.mark.parametrize('capacity', [None, 16384, 32768])
def test_conditional_exact_32b_w8_stages_metadata_before_adapter_owns_rpu(monkeypatch, capacity):
    events = []
    config = _qwen3_vl_32b_w8_config()

    class Model:

        def eval(self):
            events.append('eval')
            return self
    model = Model()

    class Adapter:

        @classmethod
        def preflight(cls, seen_config):
            assert seen_config is config
            events.append('preflight')

        def __init__(self, seen_model):
            assert seen_model is model
            assert getattr(model, '_rpu_requested_max_seq_len', None) == capacity
            events.append('adapter-init')

        def to_rpu(self):
            events.append('adapter-to-rpu')
            return 'rpu-model'

    def staged_loader(seen_config, path, **kwargs):
        assert seen_config is config
        assert path == 'unused/path'
        assert kwargs == {'dtype': torch.float16, 'revision': 'weights-rev', 'subfolder': 'weights', 'local_files_only': True}
        events.append('metadata-stage')
        return model
    monkeypatch.setattr(conditional_generation.AutoConfig, 'from_pretrained', lambda *_args, **_kwargs: events.append('config') or config)
    monkeypatch.setattr(conditional_generation, 'get_adapter', lambda _arch: events.append('adapter-lookup') or Adapter)
    monkeypatch.setattr(w8a16_load, 'stage_w8a16_imagetext_for_rpu', staged_loader)
    result = conditional_generation.RPUModelForConditionalGeneration.from_pretrained('unused/path', device='rpu', revision='weights-rev', subfolder='weights', max_seq_len=capacity, local_files_only=True)
    assert result == 'rpu-model'
    assert events == ['config', 'adapter-lookup', 'preflight', 'metadata-stage', 'eval', 'adapter-init', 'adapter-to-rpu']

@pytest.mark.parametrize('device', [None, 'rpu'])
@pytest.mark.parametrize('extra', [{}, {'output_loading_info': True}])
def test_conditional_awq_uses_exact_loader_before_adapter_and_never_hf_quantizer(monkeypatch, device, extra):
    config = SimpleNamespace(architectures=['Qwen3VLForConditionalGeneration'])
    events = []
    model = SimpleNamespace(eval=lambda : events.append('eval'))

    class Adapter:

        @classmethod
        def preflight(cls, value):
            assert value is config
            events.append('preflight')

        def __init__(self, value):
            assert value is model
            events.append('adapter')

        def to_rpu(self):
            events.append('to-rpu')
            return model

    def load(value, path, **kwargs):
        assert value is config and path == 'awq/path'
        assert kwargs == {'dtype': torch.float16, 'local_files_only': True}
        events.append('awq-load')
        return model
    monkeypatch.setattr(conditional_generation.AutoConfig, 'from_pretrained', lambda *a, **k: config)
    monkeypatch.setattr(conditional_generation, 'get_adapter', lambda arch: Adapter)
    monkeypatch.setattr(w8a16_load, 'is_qwen3_vl_awq_config', lambda value: value is config)
    monkeypatch.setattr(w8a16_load, 'load_qwen3_vl_awq_imagetext', load)
    monkeypatch.setattr(conditional_generation.AutoModelForImageTextToText, 'from_pretrained', lambda *a, **k: pytest.fail('exact AWQ reached HF quantizer'))
    if extra:
        with pytest.raises(ValueError, match='output_loading_info'):
            conditional_generation.RPUModelForConditionalGeneration.from_pretrained('awq/path', device=device, local_files_only=True, **extra)
        assert events == ['preflight']
    else:
        result = conditional_generation.RPUModelForConditionalGeneration.from_pretrained('awq/path', device=device, local_files_only=True)
        assert result is model
        assert events == ['preflight', 'awq-load', 'eval', 'adapter'] + (['to-rpu'] if device else [])
