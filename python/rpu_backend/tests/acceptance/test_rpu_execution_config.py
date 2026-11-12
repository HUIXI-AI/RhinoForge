from __future__ import annotations
import copy
import json
import pickle
import tomllib
import pytest
import torch
from rpu_backend.api._execution import bind_rpu_execution, normalize_rpu_execution, resolve_component_rpu_execution
from rpu_backend.api.policy import Pi05Policy
from rpu_backend.api.rhinovla import RhinoVLAPolicy
from rpu_backend.api.wall_oss import WallOssPolicy
REGISTERED_COMPONENTS = {'language_model': {'prefill': ('chunk_size', 'padding_rows', 'padding_budget')}, 'vision_encoder': {'vision': ('chunk_size',)}, 'action_expert': {'action': ('chunk_size',)}}

def test_execution_config_is_normalized_detached_and_deeply_read_only():
    source = {'prefill': {'chunk_size': 160, 'padding_rows': 'auto', 'padding_budget': 64}, 'action': {'chunk_size': 'auto', 'padding_rows': 0}}
    result = normalize_rpu_execution(source, entry_point='test')
    assert result == source
    assert result is not source
    assert result['prefill'] is not source['prefill']
    source['prefill']['chunk_size'] = 32
    assert result['prefill']['chunk_size'] == 160
    with pytest.raises(TypeError):
        result['prefill'] = {'chunk_size': 32}
    with pytest.raises(TypeError):
        result['prefill']['chunk_size'] = 32
    assert copy.deepcopy(result) is result
    restored = pickle.loads(pickle.dumps(result))
    assert restored == result
    with pytest.raises(TypeError):
        restored['prefill']['chunk_size'] = 32

def test_composite_config_is_canonical_deeply_frozen_and_stdlib_loadable():
    document = {'prefill': {'chunk_size': 'auto'}, 'components': {'vision_encoder': {'vision': {'chunk_size': 224}}}}
    json_document = json.loads(json.dumps({'rpu_execution': document}))
    toml_document = tomllib.loads('\n        [rpu_execution.prefill]\n        chunk_size = "auto"\n\n        [rpu_execution.components.vision_encoder.vision]\n        chunk_size = 224\n        ')
    result = normalize_rpu_execution(json_document['rpu_execution'], entry_point='test', supported_components=REGISTERED_COMPONENTS)
    assert result == normalize_rpu_execution(toml_document['rpu_execution'], entry_point='test', supported_components=REGISTERED_COMPONENTS)
    json_document['rpu_execution']['components']['vision_encoder']['vision']['chunk_size'] = 256
    assert result['components']['vision_encoder']['vision']['chunk_size'] == 224
    with pytest.raises(TypeError):
        result['components']['vision_encoder']['vision']['chunk_size'] = 256
    assert copy.deepcopy(result) is result
    assert pickle.loads(pickle.dumps(result)) == result

def test_component_resolution_precedence_covers_all_existing_stages():
    request = {'prefill': {'chunk_size': 160, 'padding_budget': 64}, 'action': {'chunk_size': 48}, 'components': {'language_model': {'prefill': {'padding_rows': 0}}, 'vision_encoder': {'vision': {'chunk_size': 224}}}}
    assert resolve_component_rpu_execution(request, 'language_model', entry_point='test', supported_components=REGISTERED_COMPONENTS, profile_auto={'prefill': {'chunk_size': 'auto', 'padding_budget': 32}}) == {'prefill': {'chunk_size': 160, 'padding_rows': 0}}
    assert resolve_component_rpu_execution(request, 'vision_encoder', entry_point='test', supported_components=REGISTERED_COMPONENTS, profile_auto={'vision': {'chunk_size': 'auto'}}) == {'vision': {'chunk_size': 224}}
    assert resolve_component_rpu_execution(request, 'action_expert', entry_point='test', supported_components=REGISTERED_COMPONENTS, profile_auto={'action': {'chunk_size': 'auto'}}) == {'action': {'chunk_size': 48}}

def test_component_capability_flows_through_cold_binding():
    from types import SimpleNamespace
    owner = SimpleNamespace()
    bound = bind_rpu_execution(owner, {'components': {'vision_encoder': {'vision': {'chunk_size': 224}}}}, entry_point='test', supported_components=REGISTERED_COMPONENTS)
    assert owner._rpu_execution is bound
    assert bound['components']['vision_encoder']['vision']['chunk_size'] == 224
    with pytest.raises(ValueError, match='unknown rpu_execution component'):
        resolve_component_rpu_execution(bound, 'missing', entry_point='test', supported_components=REGISTERED_COMPONENTS, profile_auto={})

@pytest.mark.parametrize(('value', 'match'), [({'components': {'missing': {}}}, 'unknown component'), ({'components': {'vision_encoder': []}}, 'must be a mapping'), ({'components': {'vision_encoder': {'decode': {}}}}, 'unknown stage'), ({'components': {'vision_encoder': {'prefill': {}}}}, 'not supported'), ({'components': {'vision_encoder': {'vision': {'mode': 'auto'}}}}, 'unknown field'), ({'components': {'vision_encoder': {'vision': {'padding_rows': 0}}}}, 'not supported'), ({'components': {'vision_encoder': {'vision': {'chunk_size': True}}}}, 'multiple of 16'), ({'action': {'padding_rows': 0}}, 'not supported')])
def test_composite_config_rejects_unknown_or_invalid_child_values(value, match):
    with pytest.raises((TypeError, ValueError), match=match):
        normalize_rpu_execution(value, entry_point='test', supported_components=REGISTERED_COMPONENTS)

def test_simple_model_rejects_components_without_a_registered_capability():
    assert normalize_rpu_execution({'prefill': {'chunk_size': 32}}, entry_point='test') == {'prefill': {'chunk_size': 32}}
    with pytest.raises(ValueError, match='unknown stage'):
        normalize_rpu_execution({'components': {'language_model': {}}}, entry_point='test')

@pytest.mark.parametrize(('value', 'match'), [({'decode': {}}, 'unknown stage'), ({'prefill': {'mode': 'auto'}}, 'unknown field'), ({'prefill': {'chunk_size': 17}}, 'multiple of 16'), ({'vision': {'padding_rows': -1}}, 'non-negative'), ({'action': {'padding_budget': True}}, 'non-negative'), ({'prefill': {'padding_rows': 1, 'padding_budget': 64}}, 'conflicts')])
def test_execution_config_rejects_invalid_values(value, match):
    with pytest.raises((TypeError, ValueError), match=match):
        normalize_rpu_execution(value, entry_point='test')

def test_legacy_vlm_chunk_alias_and_conflict():
    assert normalize_rpu_execution({'prefill': {'padding_rows': 'auto'}}, entry_point='test', vlm_chunk_size=0) == {'prefill': {'chunk_size': 'auto', 'padding_rows': 'auto'}}
    with pytest.raises(ValueError, match='conflicts'):
        normalize_rpu_execution({'prefill': {'chunk_size': 160}}, entry_point='test', vlm_chunk_size=160)

def test_direct_adapter_binding_reuses_equal_config_and_rejects_conflict():
    from types import SimpleNamespace
    existing = {'prefill': {'chunk_size': 32}}
    model = SimpleNamespace(_rpu_execution=existing)
    bound = bind_rpu_execution(model, {'prefill': {'chunk_size': 32}}, entry_point='test')
    assert bound is model._rpu_execution
    assert bound is not existing
    with pytest.raises(TypeError):
        model._rpu_execution['prefill']['chunk_size'] = 64
    assert bind_rpu_execution(model, {'prefill': {'chunk_size': 32}}, entry_point='test') is bound
    with pytest.raises(ValueError, match='conflicts'):
        bind_rpu_execution(model, {'prefill': {'chunk_size': 64}}, entry_point='test')

def test_pi05_legacy_alias_is_attached_to_policy_and_adapter(monkeypatch):
    from types import SimpleNamespace
    from rpu_backend.adapters.pi05 import loader
    policy = SimpleNamespace(_adapter=SimpleNamespace())
    monkeypatch.setattr(loader, 'load_and_construct_pi05_policy', lambda *args, **kwargs: policy)
    result = Pi05Policy.from_pretrained('unused', vlm_chunk_size=0)
    assert result._rpu_execution == {'prefill': {'chunk_size': 'auto'}}
    assert result._adapter._rpu_execution is result._rpu_execution
    assert result._vlm_chunk_size == 0

def test_pi05_direct_policy_consumes_prefill_chunk(monkeypatch):
    from types import SimpleNamespace
    import rpu_backend.adapters.pi05 as pi05

    class Adapter:

        def __init__(self, policy):
            self._rpu_is_ready = False
            self._vlm_chunk_size = 0
    monkeypatch.setattr(pi05, 'Pi05Adapter', Adapter)
    policy = Pi05Policy.from_lerobot_policy(SimpleNamespace(), rpu_execution={'prefill': {'chunk_size': 32}})
    assert policy._vlm_chunk_size == 32
    assert policy._adapter._vlm_chunk_size == 32

def test_pi05_component_override_precedes_top_level_default():
    from rpu_backend.adapters.pi05.runtime import PI05_TEXT_COMPONENT, resolve_pi05_execution_components
    (root, vision, text, action) = resolve_pi05_execution_components({'prefill': {'chunk_size': 64}, 'vision': {'chunk_size': 768}, 'action': {'chunk_size': 64}, 'components': {PI05_TEXT_COMPONENT: {'prefill': {'chunk_size': 160}}}}, entry_point='test')
    assert root['prefill']['chunk_size'] == 64
    assert text['prefill']['chunk_size'] == 160
    assert vision['vision']['chunk_size'] == 768
    assert action['action']['chunk_size'] == 64

def test_pi05_unknown_component_and_cross_component_stage_fail_early():
    from rpu_backend.adapters.pi05.runtime import PI05_TEXT_COMPONENT, resolve_pi05_execution_components
    with pytest.raises(ValueError, match='unknown component'):
        resolve_pi05_execution_components({'components': {'typo': {'prefill': {'chunk_size': 64}}}}, entry_point='test')
    with pytest.raises(ValueError, match='not supported'):
        resolve_pi05_execution_components({'components': {PI05_TEXT_COMPONENT: {'vision': {'chunk_size': 256}}}}, entry_point='test')

@pytest.mark.parametrize(('rpu_execution', 'expected'), [({'prefill': {'chunk_size': 32}}, 32), ({'prefill': {'chunk_size': 'auto'}}, 0), (None, 160)])
def test_pi05_public_prefill_chunk_overrides_legacy_only_when_configured(rpu_execution, expected):
    from types import SimpleNamespace
    import rpu_backend.adapters.pi05 as pi05
    vlm = SimpleNamespace(_rpu_chunk_size=160)
    pawe = SimpleNamespace(paligemma=SimpleNamespace(model=SimpleNamespace(language_model=vlm)), gemma_expert=SimpleNamespace(model=SimpleNamespace()))
    lerobot = SimpleNamespace(model=SimpleNamespace(paligemma_with_expert=pawe))
    policy = Pi05Policy.from_lerobot_policy(lerobot, rpu_execution=rpu_execution)
    pi05._apply_vlm_chunk_size(policy._adapter, vlm)
    assert vlm._rpu_chunk_size == expected

def test_pi05_internal_adapter_defaults_direct_calls_to_auto(monkeypatch):
    from types import SimpleNamespace
    import rpu_backend.adapters.pi05 as pi05
    monkeypatch.setattr(pi05, '_pi05_runtime_state', lambda _policy: None)
    adapter = pi05.Pi05Adapter(SimpleNamespace(model=SimpleNamespace()))
    assert adapter._rpu_execution == {}

def test_large_profile_chunk_ceiling_fails_before_weight_load():
    from types import SimpleNamespace
    from rpu_backend.adapters.llama import LlamaAdapter
    from rpu_backend.adapters.qwen3 import Qwen3Adapter
    from rpu_backend.adapters.qwen3_5 import Qwen3_5Adapter
    from rpu_backend.adapters.qwen3_vl import Qwen3VLAdapter
    from rpu_backend.api.errors import UnsupportedModelError
    qwen14 = SimpleNamespace(num_hidden_layers=40, hidden_size=5120)
    with pytest.raises(UnsupportedModelError, match='chunk ceiling 32'):
        Qwen3Adapter.preflight_execution(qwen14, {'prefill': {'chunk_size': 64}})
    qwen8 = SimpleNamespace(num_hidden_layers=36, hidden_size=4096)
    with pytest.raises(UnsupportedModelError, match='chunk ceiling 128'):
        Qwen3Adapter.preflight_execution(qwen8, {'prefill': {'chunk_size': 224}})
    with pytest.raises(UnsupportedModelError, match='not admissible'):
        Qwen3Adapter.preflight_execution(qwen8, {'prefill': {'chunk_size': 48}})
    q35_9b = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=32, hidden_size=4096))
    with pytest.raises(UnsupportedModelError, match='declared ceiling 256'):
        Qwen3_5Adapter.preflight_execution(q35_9b, {'prefill': {'chunk_size': 320}})
    with pytest.raises(ValueError, match='multiple of 64'):
        Qwen3_5Adapter.preflight_execution(q35_9b, {'prefill': {'chunk_size': 32}})
    with pytest.raises(ValueError, match='padding_budget.*maximum 64'):
        Qwen3_5Adapter.preflight_execution(q35_9b, {'prefill': {'padding_budget': 65}})
    with pytest.raises(ValueError, match='padding_rows.*maximum 127'):
        Qwen3_5Adapter.preflight_execution(q35_9b, {'prefill': {'padding_rows': 128}})
    with pytest.raises(ValueError, match='vision padding is unsupported'):
        Qwen3_5Adapter.preflight_execution(q35_9b, {'prefill': {'chunk_size': 'auto'}, 'vision': {'padding_budget': 1}})
    llama = SimpleNamespace(num_hidden_layers=16, hidden_size=2048)
    with pytest.raises(UnsupportedModelError, match='chunk ceiling 64'):
        LlamaAdapter.preflight_execution(llama, {'prefill': {'chunk_size': 128}})
    qvl = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=36, hidden_size=2560))
    with pytest.raises(UnsupportedModelError, match='planning ceiling 256'):
        Qwen3VLAdapter.preflight_execution(qvl, {'prefill': {'chunk_size': 512}})

def test_pi05_geometry_is_rejected_before_swizzle(monkeypatch):
    from types import SimpleNamespace
    from rpu_backend.adapters.pi05 import Pi05Adapter, _validate_execution_geometry
    model = SimpleNamespace(config=SimpleNamespace(chunk_size=50, image_features=('a', 'b', 'c')))
    adapter = Pi05Adapter(SimpleNamespace(model=model))
    adapter._rpu_execution = {'action': {'chunk_size': 32}}
    with pytest.raises(ValueError, match='single-chunk capacity 64'):
        _validate_execution_geometry(adapter)
    adapter._rpu_execution = {'action': {'chunk_size': 128}}
    with pytest.raises(ValueError, match='single-chunk capacity 64'):
        _validate_execution_geometry(adapter)
    adapter._rpu_execution = {'vision': {'chunk_size': 512}}
    monkeypatch.setenv('RPU_PI05_SIGLIP_BATCH', '1')
    with pytest.raises(ValueError, match='single-chunk capacity 768'):
        _validate_execution_geometry(adapter)
    adapter._rpu_execution = {'vision': {'chunk_size': 1024}}
    with pytest.raises(ValueError, match='single-chunk capacity 768'):
        _validate_execution_geometry(adapter)
    adapter._rpu_execution = {'vision': {'chunk_size': 768}, 'action': {'chunk_size': 64}}
    _validate_execution_geometry(adapter)
    monkeypatch.setenv('RPU_PI05_SIGLIP_BATCH', '0')
    adapter._rpu_execution = {'vision': {'chunk_size': 768}}
    with pytest.raises(ValueError, match='single-chunk capacity 256'):
        _validate_execution_geometry(adapter)
    monkeypatch.setenv('RPU_PI05_FUSED_DENOISE', '0')
    adapter._rpu_execution = {'action': {'chunk_size': 64}}
    _validate_execution_geometry(adapter)

def test_wall_oss_rejects_unwired_execution_config():
    with pytest.raises(ValueError, match='not supported'):
        WallOssPolicy.from_checkpoint('/unused', rpu_execution={'vision': {'padding_budget': 32}})

def test_rhinovla_rejects_bad_config_before_factory_and_attaches_good_config():
    calls = []

    class Session:

        def __init__(self, config):
            self.config = config

        def register_config_view(self, view):
            view._rpu_execution = self.config

    class Controller:

        def __init__(self, runtime, config):
            self.runtime = runtime
            self.session = Session(config)

    class Runtime:
        _rpu_execution_capabilities = {'prefill': ('chunk_size', 'padding_rows', 'padding_budget'), 'vision': ('chunk_size', 'padding_rows', 'padding_budget'), 'action': ('chunk_size', 'padding_rows', 'padding_budget')}

        def __init__(self, execution=None, *, bound=True):
            if execution is not None:
                self._rpu_execution = execution
            if bound:
                config = getattr(self, '_rpu_execution', {})
                self._rhinovla_execution_controller = Controller(self, config)

        def predict(self, **request):
            return request

    def factory(config, **kwargs):
        calls.append((config, kwargs))
        return Runtime(kwargs.get('rpu_execution'))
    factory._rpu_execution_capabilities = Runtime._rpu_execution_capabilities
    with pytest.raises(ValueError, match='multiple of 16'):
        RhinoVLAPolicy.from_factory(factory, {}, rpu_execution={'action': {'chunk_size': 17}})
    assert calls == []
    policy = RhinoVLAPolicy.from_factory(factory, {}, rpu_execution={'action': {'chunk_size': 32}})
    assert policy._rpu_execution == {'action': {'chunk_size': 32}}
    assert calls == [({}, {'rpu_execution': {'action': {'chunk_size': 32}}})]
    assert RhinoVLAPolicy.from_runtime(policy, rpu_execution={'action': {'chunk_size': 32}}) is policy
    with pytest.raises(ValueError, match='conflicts'):
        RhinoVLAPolicy.from_runtime(policy, rpu_execution={'action': {'chunk_size': 64}})
    with pytest.raises(ValueError, match='must call'):
        RhinoVLAPolicy.from_runtime(Runtime(bound=False), rpu_execution={'action': {'chunk_size': 32}})
    inherited_runtime = Runtime({'vision': {'chunk_size': 768}})
    inherited = RhinoVLAPolicy.from_runtime(inherited_runtime)
    assert inherited._rpu_execution == {'vision': {'chunk_size': 768}}

@pytest.mark.parametrize('module_name', ['causal_lm', 'conditional_generation'])
def test_generic_loader_attaches_one_canonical_config(monkeypatch, module_name):
    from types import SimpleNamespace
    from rpu_backend.api import causal_lm, conditional_generation
    module = causal_lm if module_name == 'causal_lm' else conditional_generation
    loader = module.RPUModelForCausalLM if module_name == 'causal_lm' else module.RPUModelForConditionalGeneration
    auto_model = module.AutoModelForCausalLM if module_name == 'causal_lm' else module.AutoModelForImageTextToText

    class Model:

        def eval(self):
            return self

    class Adapter:
        last = None

        @classmethod
        def preflight(cls, config):
            pass

        def __init__(self, model):
            type(self).last = self
    model = Model()
    monkeypatch.setattr(module.AutoConfig, 'from_pretrained', lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(module, 'resolve_config_architecture', lambda *args, **kwargs: 'TestArchitecture')
    monkeypatch.setattr(module, 'get_adapter', lambda arch: Adapter)
    monkeypatch.setattr(auto_model, 'from_pretrained', lambda *args, **kwargs: model)
    if module_name == 'causal_lm':
        from rpu_backend.quant import load
        monkeypatch.setattr(load, 'is_w8a16_config', lambda config: False)
    execution = {'prefill': {'chunk_size': 32}}
    if module_name == 'conditional_generation':
        execution['vision'] = {'chunk_size': 256}
    result = loader.from_pretrained('unused', dtype=torch.float16, rpu_execution=execution)
    assert result is model
    assert model._rpu_execution == execution
    assert Adapter.last._rpu_execution is model._rpu_execution
    with pytest.raises(TypeError):
        model._rpu_execution['prefill']['chunk_size'] = 64

@pytest.mark.parametrize('module_name,symbol', [('dinov3', 'DINOv3Adapter'), ('gemma4', 'Gemma4Adapter'), ('qwen3_5', 'Qwen3_5Adapter'), ('gr00t', 'build_gr00t_vla'), ('lingbot_vla', 'build_lingbot_vla'), ('wall_oss', 'build_wall_oss_vla')], ids=['dinov3', 'gemma4', 'qwen35', 'gr00t', 'lingbot_v1', 'wall'])
def test_model_specific_public_routes_accept_execution_config(module_name, symbol):
    import importlib
    import inspect
    module = importlib.import_module(f'rpu_backend.adapters.{module_name}')
    assert 'rpu_execution' in inspect.signature(getattr(module, symbol)).parameters

def test_lingbot_builder_rejects_unconsumed_vision_before_loading():
    from rpu_backend.adapters.lingbot_vla import build_lingbot_vla
    with pytest.raises(ValueError, match='not supported'):
        build_lingbot_vla('/unused', rpu_execution={'vision': {'chunk_size': 32}})

def test_public_loaders_and_direct_adapters_fail_before_unwired_routes_run(monkeypatch):
    from types import SimpleNamespace
    from rpu_backend.adapters.qwen3_5 import Qwen3_5Adapter
    from rpu_backend.api.causal_lm import RPUModelForCausalLM
    from rpu_backend.api.conditional_generation import RPUModelForConditionalGeneration
    from rpu_backend.api import conditional_generation
    monkeypatch.setattr(conditional_generation.AutoConfig, 'from_pretrained', lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(conditional_generation, 'resolve_config_architecture', lambda *args, **kwargs: 'UnwiredTestArchitecture')
    monkeypatch.setattr(conditional_generation, 'get_adapter', lambda arch: SimpleNamespace())
    monkeypatch.setattr(conditional_generation.AutoModelForImageTextToText, 'from_pretrained', lambda *args, **kwargs: pytest.fail('weights must not load'))
    calls = (lambda : RPUModelForCausalLM.from_pretrained('unused', rpu_execution={'action': {'chunk_size': 32}}), lambda : RPUModelForConditionalGeneration.from_pretrained('unused', rpu_execution={'vision': {'padding_rows': 0}}), lambda : Qwen3_5Adapter(SimpleNamespace(), rpu_execution={'action': {'chunk_size': 32}}))
    for call in calls:
        with pytest.raises(ValueError, match='not supported'):
            call()

def test_pi05_consumes_three_stage_chunks_but_rejects_non_prefill_padding(monkeypatch):
    from types import SimpleNamespace
    from rpu_backend.adapters.pi05 import loader
    policy = SimpleNamespace(_adapter=SimpleNamespace())
    monkeypatch.setattr(loader, 'load_and_construct_pi05_policy', lambda *args, **kwargs: policy)
    result = Pi05Policy.from_pretrained('unused', rpu_execution={'prefill': {'chunk_size': 'auto', 'padding_budget': 15}, 'vision': {'chunk_size': 768}, 'action': {'chunk_size': 64}})
    assert result._rpu_execution['vision']['chunk_size'] == 768
    assert result._rpu_execution['action']['chunk_size'] == 64
    for stage in ('vision', 'action'):
        with pytest.raises(ValueError, match='not supported'):
            Pi05Policy.from_pretrained('unused', rpu_execution={stage: {'padding_rows': 0}})

def test_qwen35_rejects_16_aligned_but_not_64_aligned_chunk_before_profile():
    from types import SimpleNamespace
    from rpu_backend.adapters.qwen3_5 import Qwen3_5Adapter
    with pytest.raises(ValueError, match='multiple of 64'):
        Qwen3_5Adapter(SimpleNamespace(), rpu_execution={'prefill': {'chunk_size': 32}})

def test_qwen35_direct_adapter_rejects_ceiling_before_runtime_mutation():
    from types import SimpleNamespace
    from rpu_backend.adapters.qwen3_5 import Qwen3_5Adapter
    from rpu_backend.api.errors import UnsupportedModelError
    events = []

    class Model:
        config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=4096, intermediate_size=12288, num_hidden_layers=32, num_key_value_heads=4, layer_types=['full_attention'] * 32))

        def to(self, *_args, **_kwargs):
            events.append('to')
            return self

        def half(self):
            events.append('half')
            return self
    model = Model()
    with pytest.raises(UnsupportedModelError, match='declared ceiling 256'):
        Qwen3_5Adapter(model, rpu_execution={'prefill': {'chunk_size': 320}})
    assert events == []
    assert not hasattr(model, '_rpu_swizzle_started')
    assert not hasattr(model, '_rpu_swizzled')
