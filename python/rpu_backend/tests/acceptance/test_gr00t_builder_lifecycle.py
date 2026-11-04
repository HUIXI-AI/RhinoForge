from __future__ import annotations
import inspect
import math
from types import SimpleNamespace
import pytest
import torch
from rpu_backend.adapters.gr00t import runtime

@pytest.fixture(autouse=True)
def isolated_retirement(monkeypatch):
    from rpu_backend.api import _execution, causal_lm
    monkeypatch.setattr(_execution, '_UNSAFE_PROCESS_REASON', None)
    monkeypatch.setattr(causal_lm, '_LIVE_REF', None)
    monkeypatch.setattr(causal_lm, '_LIVE_TERMINAL_REASON', None)
    monkeypatch.setattr(runtime, '_FAILED_RETIREMENTS', [])
    monkeypatch.delenv('RPU_GR00T_PRECOMPUTE_ADALN', raising=False)
    monkeypatch.delenv('RPU_GR00T_PRECOMPUTE_TIME_GLUE', raising=False)

class _Cache:

    def __init__(self, events, name):
        self._events = events
        self._name = name

    def clear(self):
        self._events.append(('clear', self._name))

    def cache_invariant_ok(self):
        return True

class _Finalizer:

    def __init__(self, events, name):
        self._events = events
        self._name = name
        self.alive = True

    def __call__(self):
        self._events.append(('finalize', self._name))
        self.alive = False

    def detach(self):
        self.alive = False

def _bind_close_session(owner, monkeypatch):
    events = owner._d_gc._events
    owner._bb = SimpleNamespace()
    (owner._d_handle, owner._a_handle) = (32, 31)
    for (child, attr, default) in ((owner._text, '_rpu_decoder_handle', 12), (owner._vision, '_rpu_vision_handle', 11)):
        if not hasattr(child, attr):
            setattr(child, attr, default)
    _raw_backbone_ops(monkeypatch, events)
    monkeypatch.setattr(runtime, '_destroy_gr00t_dit_handle', lambda h: events.append(('destroy', 'dit', h)))
    monkeypatch.setattr(runtime, '_destroy_gr00t_vl_encoder_handle', lambda h: events.append(('destroy', 'vl', h)))
    owner._rpu_execution = {}
    owner._execution_session = runtime.bind_execution_session(owner, owner._rpu_execution, entry_point='test_gr00t_close', supported_components=runtime._EXECUTION_COMPONENTS, graph_mode=runtime.GRAPH_COMPOSITE_CHILD)

def _raw_backbone_ops(monkeypatch, events):
    for (op, kind) in (('causal_decoder_destroy', 'decoder'), ('qwen3vl_vision_destroy', 'vision')):
        monkeypatch.setattr(torch.ops.rpu, op, lambda h, kind=kind: events.append(('destroy', kind, h)), raising=False)

@pytest.fixture
def native_ops(monkeypatch):
    state = {'next': 1, 'alive': {}, 'events': [], 'fail': None}

    def create(kind):
        handle = state['next']
        state['next'] += 1
        state['alive'][handle] = kind
        state['events'].append(('create', kind, handle))
        return handle

    def step(name, handle):
        state['events'].append((name, handle))
        if state['fail'] == name:
            raise RuntimeError(f'synthetic {name} failure')

    def destroy(kind, handle):
        state['events'].append(('destroy', kind, handle))
        assert state['alive'].pop(handle) == kind
    monkeypatch.setattr(torch.ops.rpu, 'gr00t_vl_encoder_create', lambda : create('vl'), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'gr00t_vl_encoder_set_weights', lambda handle, *_: step('vl_fp16', handle), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'gr00t_vl_encoder_set_weights_w8a16', lambda handle, *_: step('vl_w8', handle), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'gr00t_vl_encoder_destroy', lambda handle: destroy('vl', handle), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'gr00t_dit_create', lambda : create('dit'), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'gr00t_dit_set_weights', lambda handle, *_: step('dit_fp16', handle), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'gr00t_dit_set_weights_w8a16', lambda handle, *_: step('dit_w8', handle), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'gr00t_denoise_set_weights', lambda handle, *_: step('denoise', handle), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'gr00t_dit_prepare_adaln_table', lambda handle: step('adaln_prepare', handle), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'gr00t_dit_prepare_time_tables', lambda handle: step('time_prepare', handle), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'gr00t_dit_destroy', lambda handle: destroy('dit', handle), raising=False)
    return state

@pytest.mark.parametrize('w8a16,failure', [(False, 'vl_fp16'), (True, 'vl_w8')])
def test_vl_encoder_configuration_failure_destroys_handle(native_ops, w8a16, failure):
    native_ops['fail'] = failure
    with pytest.raises(RuntimeError, match=failure):
        runtime._configure_gr00t_vl_encoder_handle(weight_args=(), scale_args=(), w8a16=w8a16)
    assert native_ops['alive'] == {}
    assert native_ops['events'][-1][0:2] == ('destroy', 'vl')

@pytest.mark.parametrize('w8a16,failure', [(False, 'dit_fp16'), (True, 'dit_w8'), (False, 'denoise'), (True, 'denoise')])
def test_denoise_configuration_failure_destroys_handle(native_ops, w8a16, failure):
    native_ops['fail'] = failure
    with pytest.raises(RuntimeError, match=failure):
        runtime._configure_gr00t_denoise_handle(dit_args=(), dit_scale_args=(), denoise_args=(), w8a16=w8a16)
    assert native_ops['alive'] == {}
    assert native_ops['events'][-1][0:2] == ('destroy', 'dit')

@pytest.mark.parametrize('value', [None, '', '0', '1'])
@pytest.mark.parametrize('w8a16', [False, True])
def test_adaln_preparation_is_cold_after_both_weight_setters(monkeypatch, native_ops, value, w8a16):
    if value is not None:
        monkeypatch.setenv('RPU_GR00T_PRECOMPUTE_ADALN', value)
    handle = runtime._configure_gr00t_denoise_handle(dit_args=(), dit_scale_args=(), denoise_args=(), w8a16=w8a16)
    expected = [('create', 'dit', handle), ('dit_w8' if w8a16 else 'dit_fp16', handle), ('denoise', handle)]
    if value == '1':
        expected.append(('adaln_prepare', handle))
    assert native_ops['events'] == expected

def test_adaln_preparation_failure_retires_native_handle(monkeypatch, native_ops):
    monkeypatch.setenv('RPU_GR00T_PRECOMPUTE_ADALN', '1')
    native_ops['fail'] = 'adaln_prepare'
    with pytest.raises(RuntimeError, match='adaln_prepare'):
        runtime._configure_gr00t_denoise_handle(dit_args=(), dit_scale_args=(), denoise_args=(), w8a16=True)
    assert native_ops['alive'] == {}
    assert native_ops['events'][-1][:2] == ('destroy', 'dit')

@pytest.mark.parametrize('value', ['true', '2', ' 1'])
def test_invalid_adaln_policy_fails_before_native_creation(monkeypatch, native_ops, value):
    monkeypatch.setenv('RPU_GR00T_PRECOMPUTE_ADALN', value)
    with pytest.raises(ValueError, match='RPU_GR00T_PRECOMPUTE_ADALN'):
        runtime._configure_gr00t_denoise_handle(dit_args=(), dit_scale_args=(), denoise_args=(), w8a16=True)
    assert native_ops['events'] == []

@pytest.mark.parametrize('value', [None, '', '0', '1'])
@pytest.mark.parametrize('w8a16', [False, True])
def test_time_glue_preparation_is_cold_and_independent(monkeypatch, native_ops, value, w8a16):
    if value is not None:
        monkeypatch.setenv('RPU_GR00T_PRECOMPUTE_TIME_GLUE', value)
    handle = runtime._configure_gr00t_denoise_handle(dit_args=(), dit_scale_args=(), denoise_args=(), w8a16=w8a16)
    expected = [('create', 'dit', handle), ('dit_w8' if w8a16 else 'dit_fp16', handle), ('denoise', handle)]
    if value == '1':
        expected.append(('time_prepare', handle))
    assert native_ops['events'] == expected

def test_time_glue_preparation_composes_with_adaln(monkeypatch, native_ops):
    monkeypatch.setenv('RPU_GR00T_PRECOMPUTE_ADALN', '1')
    monkeypatch.setenv('RPU_GR00T_PRECOMPUTE_TIME_GLUE', '1')
    handle = runtime._configure_gr00t_denoise_handle(dit_args=(), dit_scale_args=(), denoise_args=(), w8a16=True)
    assert native_ops['events'][-2:] == [('adaln_prepare', handle), ('time_prepare', handle)]

def test_time_glue_preparation_failure_retires_handle(monkeypatch, native_ops):
    monkeypatch.setenv('RPU_GR00T_PRECOMPUTE_ADALN', '1')
    monkeypatch.setenv('RPU_GR00T_PRECOMPUTE_TIME_GLUE', '1')
    native_ops['fail'] = 'time_prepare'
    with pytest.raises(RuntimeError, match='time_prepare'):
        runtime._configure_gr00t_denoise_handle(dit_args=(), dit_scale_args=(), denoise_args=(), w8a16=True)
    assert native_ops['alive'] == {}
    assert native_ops['events'][-1][:2] == ('destroy', 'dit')

@pytest.mark.parametrize('value', ['true', '2', ' 1'])
def test_invalid_time_glue_policy_fails_before_creation(monkeypatch, native_ops, value):
    monkeypatch.setenv('RPU_GR00T_PRECOMPUTE_TIME_GLUE', value)
    with pytest.raises(ValueError, match='RPU_GR00T_PRECOMPUTE_TIME_GLUE'):
        runtime._configure_gr00t_denoise_handle(dit_args=(), dit_scale_args=(), denoise_args=(), w8a16=True)
    assert native_ops['events'] == []

def test_composite_adoption_detaches_both_backbone_raw_finalizers():
    events = []
    text = SimpleNamespace(_rpu_decoder_handle_finalizer=_Finalizer(events, 'text'))
    vision = SimpleNamespace(_rpu_vision_handle_finalizer=_Finalizer(events, 'vision'))
    owner = SimpleNamespace(_text=text, _vision=vision)
    runtime._take_gr00t_backbone_ownership(owner)
    assert events == []
    assert not text._rpu_decoder_handle_finalizer.alive
    assert not vision._rpu_vision_handle_finalizer.alive
    assert text._gr00t_retirement_owner is vision._gr00t_retirement_owner is owner

def test_composite_adoption_rejects_consumed_backbone_finalizer():
    events = []
    finalizer = _Finalizer(events, 'text')
    finalizer.alive = False
    owner = SimpleNamespace(_text=SimpleNamespace(_rpu_decoder_handle_finalizer=finalizer))
    with pytest.raises(RuntimeError, match='consumed'):
        runtime._take_gr00t_backbone_ownership(owner)
    assert events == []

def test_runtime_close_clears_graphs_before_all_handles(monkeypatch):
    events = []
    text = SimpleNamespace(_rpu_text_graph_cache=_Cache(events, 'text'), _rpu_decoder_graph_cache=_Cache(events, 'decoder'), _rpu_decoder_handle_finalizer=_Finalizer(events, 'decoder'))
    vision = SimpleNamespace(_rpu_vision_graph_cache=_Cache(events, 'vision'), _rpu_vision_handle_finalizer=_Finalizer(events, 'vision'))
    owner = runtime.Gr00tN1d7VLA.__new__(runtime.Gr00tN1d7VLA)
    owner._closed = False
    owner._d_gc = _Cache(events, 'dit')
    owner._a_gc = _Cache(events, 'vl')
    owner._d_handle_finalizer = _Finalizer(events, 'dit')
    owner._a_handle_finalizer = _Finalizer(events, 'vl')
    owner._text = text
    owner._vision = vision
    _bind_close_session(owner, monkeypatch)
    owner.close()
    owner.close()
    assert events == [('clear', 'dit'), ('clear', 'vl'), ('clear', 'text'), ('clear', 'decoder'), ('clear', 'vision'), ('destroy', 'dit', 32), ('destroy', 'vl', 31), ('destroy', 'decoder', 12), ('destroy', 'vision', 11)]

def test_runtime_close_retires_composite_graph_before_component_handles(monkeypatch):
    events = []
    monkeypatch.setattr(torch.ops.rpu, 'qwen3vl_pooler_spm_z1_unprepare', lambda vision, text, plan: events.append(('unprepare', vision, text, plan)), raising=False)
    text = SimpleNamespace(_rpu_decoder_handle=12, _rpu_text_graph_cache=_Cache(events, 'text'), _rpu_decoder_graph_cache=_Cache(events, 'decoder'), _rpu_decoder_handle_finalizer=_Finalizer(events, 'decoder'))
    vision = SimpleNamespace(_rpu_vision_handle=11, _rpu_vision_graph_cache=_Cache(events, 'vision'), _rpu_vision_handle_finalizer=_Finalizer(events, 'vision'))
    owner = runtime.Gr00tN1d7VLA.__new__(runtime.Gr00tN1d7VLA)
    owner._closed = False
    owner._qwen_z1_plan_hash = 91
    owner._qwen_z1_gc = _Cache(events, 'composite')
    owner._d_gc = _Cache(events, 'dit')
    owner._a_gc = _Cache(events, 'vl')
    owner._d_handle_finalizer = _Finalizer(events, 'dit')
    owner._a_handle_finalizer = _Finalizer(events, 'vl')
    owner._text = text
    owner._vision = vision
    _bind_close_session(owner, monkeypatch)
    owner.close()
    owner.close()
    assert events == [('clear', 'composite'), ('clear', 'dit'), ('clear', 'vl'), ('clear', 'text'), ('clear', 'decoder'), ('clear', 'vision'), ('unprepare', 11, 12, 91), ('destroy', 'dit', 32), ('destroy', 'vl', 31), ('destroy', 'decoder', 12), ('destroy', 'vision', 11)]

def test_partial_build_cleanup_retires_raw_handles_before_backbone(monkeypatch):
    events = []
    _raw_backbone_ops(monkeypatch, events)
    monkeypatch.setattr(runtime, '_destroy_gr00t_dit_handle', lambda handle: events.append(('destroy', 'dit', handle)))
    monkeypatch.setattr(runtime, '_destroy_gr00t_vl_encoder_handle', lambda handle: events.append(('destroy', 'vl', handle)))
    model = SimpleNamespace(_rpu_swizzled=True)
    text = SimpleNamespace(_rpu_text_graph_cache=_Cache(events, 'text'), _rpu_decoder_handle_finalizer=_Finalizer(events, 'decoder'), _rpu_decoder_handle=7, _rpu_deepstack_lang_layers=[0, 1, 2], forward=lambda *a, **k: None)
    vision = SimpleNamespace(_rpu_vision_graph_cache=_Cache(events, 'vision'), _rpu_vision_handle_finalizer=_Finalizer(events, 'vision'), _rpu_vision_handle=8, _rpu_lazy_init_checked=True, forward=lambda *a, **k: None)
    snapshots = tuple(((owner, False, None) for owner in (model, text, vision)))
    runtime._cleanup_gr00t_partial_build(a_handle=41, d_handle=42, model=model, text_model=text, vision_model=vision, forward_snapshots=snapshots)
    assert events == [('clear', 'text'), ('clear', 'vision'), ('destroy', 'dit', 42), ('destroy', 'vl', 41), ('destroy', 'decoder', 7), ('destroy', 'vision', 8), ('clear', 'text'), ('clear', 'vision')]
    assert not [n for n in vars(text) if n.startswith('_rpu_')], vars(text)
    assert not [n for n in vars(vision) if n.startswith('_rpu_')], vars(vision)
    assert 'forward' not in vars(text)
    assert 'forward' not in vars(vision)
    assert model._rpu_swizzled is False

def test_partial_build_retires_qwen_then_temporary_z2_authority(monkeypatch):
    events = []
    _raw_backbone_ops(monkeypatch, events)
    monkeypatch.setattr(torch.ops.rpu, 'qwen3vl_pooler_spm_z1_unprepare', lambda vision, text, plan: events.append(('unprepare', 'qwen', vision, text, plan)), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'gr00t_spm_z2_unprepare', lambda vl, denoise, plan: events.append(('unprepare', 'z2', vl, denoise, plan)), raising=False)
    monkeypatch.setattr(runtime, '_destroy_gr00t_dit_handle', lambda handle: events.append(('destroy', 'dit', handle)))
    monkeypatch.setattr(runtime, '_destroy_gr00t_vl_encoder_handle', lambda handle: events.append(('destroy', 'vl', handle)))
    text = SimpleNamespace(_rpu_decoder_handle=12, _rpu_text_graph_cache=_Cache(events, 'text'), _rpu_decoder_handle_finalizer=_Finalizer(events, 'decoder'))
    vision = SimpleNamespace(_rpu_vision_handle=11, _rpu_vision_graph_cache=_Cache(events, 'vision'), _rpu_vision_handle_finalizer=_Finalizer(events, 'vision'))
    model = SimpleNamespace(_rpu_swizzled=True)
    runtime._cleanup_gr00t_partial_build(a_handle=41, d_handle=42, model=model, text_model=text, vision_model=vision, qwen_z1_pipeline_cache=_Cache(events, 'composite'), qwen_z1_plan_hash=91, gr00t_z2_plan_hash=92)
    assert events[:7] == [('clear', 'composite'), ('clear', 'text'), ('clear', 'vision'), ('unprepare', 'qwen', 11, 12, 91), ('unprepare', 'z2', 41, 42, 92), ('destroy', 'dit', 42), ('destroy', 'vl', 41)]

def test_builder_claims_process_owner_before_irreversible_swizzle():
    source = inspect.getsource(runtime.build_gr00t_vla)
    assert source.index('_claim_live_instance(model)') < source.index('convert_linear_weights_inplace(text_model')
    assert source.index('model._rpu_swizzle_started = True') < source.index('convert_linear_weights_inplace(text_model')

def test_qwen_z1_builder_and_invocation_lifecycle_are_transactional():
    builder = inspect.getsource(runtime.build_gr00t_vla)
    assert builder.index('if _w8a16_on():') < builder.index('Qwen3VLConfig.from_pretrained')
    assert builder.index('os.environ["RPU_QWEN3VL_VISION_FUSED_MERGER"] = "1"') < builder.index('install_qwen3_vl_vision_for_rpu(')
    for (exact_env, value) in (('RPU_QWEN3VL_VISION_ROPE_SPM', '1'), ('RPU_KVINSERT_V16', '1'), ('RPU_GR00T_PARTIAL_MROPE', '0'), ('RPU_QWEN3VL_VISION_HOST_FP32_PATCH', '0'), ('RPU_QWEN3VL_VISION_BATCH', '0')):
        assert f'os.environ["{exact_env}"] = "{value}"' in builder
    prepare_ad = builder.index('torch.ops.rpu.gr00t_spm_z2_prepare(')
    prepare_qwen = builder.index('torch.ops.rpu.qwen3vl_pooler_spm_z1_prepare(')
    unprepare_ad = builder.index('_unprepare_gr00t_spm_z2(', prepare_qwen)
    transfer = builder.index('Gr00tN1d7VLA.__init__(runtime,')
    assert prepare_ad < prepare_qwen < unprepare_ad < transfer
    assert 'gr00t_z2_plan_hash=gr00t_z2_plan_hash' in builder
    assert 'qwen_z1_plan_hash=qwen_z1_plan_hash' in builder
    forward = inspect.getsource(runtime.Gr00tN1d7VLA._backbone_qwen_z1)
    begin = forward.index('qwen3vl_pooler_spm_z1_begin(')
    capture = forward.index('with self._qwen_z1_gc.capture(')
    dispatch = forward.index('qwen3vl_pooler_spm_z1_pipeline_forward(')
    cancel = forward.index('qwen3vl_pooler_spm_z1_cancel(')
    end = forward.index('qwen3vl_pooler_spm_z1_end(', cancel)
    assert begin < capture < dispatch < cancel < end
    assert 'invocation_consumed' not in forward
    assert 'qwen3vl_pooler_spm_z1_abort(' not in forward
    constructor = inspect.getsource(runtime.Gr00tN1d7VLA.__init__)
    assert 'GraphCache(max_entries=1).begin_warmup()' in constructor
    assert constructor.count('begin_warmup()') == 3
    assert 'GRAPH_NATIVE_COMPOSITE1' in constructor
    assert 'if qwen_z1_plan_hash is not None' in constructor
    assert forward.index('_preflight_ready_graph(') < begin
    action = inspect.getsource(runtime.Gr00tN1d7VLA.get_action)
    denoise = action.index('action = self._denoise(')
    freeze = action.index('cache.freeze()')
    returned = action.index('return action')
    assert denoise < freeze < returned

@pytest.fixture
def cold_builder(monkeypatch, native_ops):
    import transformers
    from rpu_backend.api import causal_lm
    state = native_ops
    state.update(envelopes={}, caches=[], z1=False, w8=False, model_reads=[], denoise_steps=[], cache_requests=[], honor_capacity=False)
    text_config = SimpleNamespace(num_hidden_layers=16, num_key_value_heads=4, head_dim=128)
    cfg = SimpleNamespace(text_config=text_config, vision_config=SimpleNamespace(deepstack_visual_indexes=[5, 11, 17]))

    class Module:

        def to(self, *args):
            return self

        def half(self):
            return self

        def eval(self):
            return self

        def forward(self, *args, **kwargs):
            state['events'].append(('forward',))
            raise AssertionError('cold builder must not run a forward')

        def load_state_dict(self, *args, **kwargs):
            return SimpleNamespace(missing_keys=[])
    (text, vision, model) = (Module(), Module(), Module())
    text.norm = SimpleNamespace(weight=torch.ones(16))
    model.model = SimpleNamespace(language_model=text, visual=vision)
    model.config = SimpleNamespace(image_token_id=1)
    state['model'] = model

    def read_config(path):
        state['model_reads'].append(path)
        return cfg
    monkeypatch.setattr(transformers, 'Qwen3VLConfig', SimpleNamespace(from_pretrained=read_config))
    monkeypatch.setattr(transformers, 'Qwen3VLForConditionalGeneration', lambda _: model)

    class Store:

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def keys(self):
            return ['backbone.model.weight']

        def get_tensor(self, key):
            return torch.ones(1)
    monkeypatch.setattr(runtime.glob, 'glob', lambda _: ['unused-cpu-fixture'])
    monkeypatch.setattr(runtime, 'safe_open', lambda *args, **kwargs: Store())
    monkeypatch.setattr(runtime, '_ckpt_reader', lambda _: None)
    monkeypatch.setattr(runtime, '_w8a16_on', lambda : state['w8'])
    monkeypatch.setattr(runtime, '_qwen3vl_spm_z1_on', lambda : state['z1'])
    for name in ('_quantize_backbone_inplace', '_backbone_scale_lists', 'convert_linear_weights_inplace', 'enable_qwen3_vl_vision_merger_on_device', 'register_qwen3vl_vision_fused_merger'):
        monkeypatch.setattr(runtime, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, '_extract_mrope_params', lambda _: (128, 10000, (16, 24, 24)))
    for name in ('RPU_QWEN3VL_VISION_FUSED_MERGER', 'RPU_QWEN3VL_VISION_ROPE_SPM', 'RPU_KVINSERT_V16', 'RPU_GR00T_PARTIAL_MROPE', 'RPU_QWEN3VL_VISION_HOST_FP32_PATCH', 'RPU_QWEN3VL_VISION_BATCH', 'RPU_WALL_OSS_FAST_REPLAY', 'RPU_FASTREPLAY_SKIP_SYNC', 'RPU_DEEP_FAST_REPLAY'):
        monkeypatch.setenv(name, '0')
        monkeypatch.delenv(name, raising=False)

    def install(owner, kind, handle):
        state['events'].append(('install', kind, handle))
        setattr(owner, '_rpu_' + kind + '_handle', handle)
        setattr(owner, '_rpu_' + kind + '_graph_cache', _Cache(state['events'], kind))
        setattr(owner, '_rpu_' + kind + '_handle_finalizer', _Finalizer(state['events'], kind))
    monkeypatch.setattr(runtime, 'install_qwen3_vl_vision_for_rpu', lambda owner, **kwargs: install(owner, 'vision', 91))
    monkeypatch.setattr(runtime, 'install_qwen3_vl_text_for_rpu', lambda owner, **kwargs: install(owner, 'decoder', 90))
    for prefix in ('qwen3vl_vision', 'causal_decoder'):
        monkeypatch.setattr(torch.ops.rpu, prefix + '_kvinsert_cost_catalog_sha256', lambda handle: '0' * 64, raising=False)
    _raw_backbone_ops(monkeypatch, state['events'])

    class Cache:

        def __init__(self, **kwargs):
            state['cache_requests'].append(dict(kwargs))
            self.max_seq_len = kwargs['max_seq_len'] if state['honor_capacity'] else {4: 144, 32: 384, 16: 512}[kwargs['num_layers']]
            state['caches'].append(self)
    monkeypatch.setattr(runtime, 'RPUCache', Cache)

    class Graph(_Cache):

        def __init__(self, **kwargs):
            super().__init__(state['events'], 'runtime')

        def begin_warmup(self):
            return self

        def capture(self, *args, **kwargs):
            state['events'].append(('capture',))
            raise AssertionError('cold builder must not capture a Graph')
    monkeypatch.setattr(runtime.rpu_backend.graph, 'GraphCache', Graph)
    original_init = runtime.Gr00tN1d7VLA.__init__

    def construct(owner, **kwargs):
        state['events'].append(('constructor',))
        return original_init(owner, **kwargs)
    monkeypatch.setattr(runtime.Gr00tN1d7VLA, '__init__', construct)

    def first_plan(*args, **kwargs):
        state['events'].append(('plan', kwargs['execution_component']))
        raise RuntimeError('first native plan boundary')
    monkeypatch.setattr(runtime, 'plan_bounded_prefill_execution', first_plan)

    def build_vl(*args):
        return (runtime._configure_gr00t_vl_encoder_handle(weight_args=(), scale_args=(), w8a16=state['w8']), ())

    def build_dit(*args, num_steps=runtime._NSTEPS):
        state['denoise_steps'].append(num_steps)
        return (runtime._configure_gr00t_denoise_handle(dit_args=(), dit_scale_args=(), denoise_args=(), w8a16=state['w8']), ())
    monkeypatch.setattr(runtime, '_build_vl_encoder', build_vl)
    monkeypatch.setattr(runtime, '_build_denoise', build_dit)

    def persistent(kind):

        def reserve(*args):
            assert ('install', 'vision', 91) in state['events']
            assert ('install', 'decoder', 90) in state['events']
            assert ('vl_w8' if state['w8'] else 'vl_fp16', 1) in state['events']
            assert ('dit_w8' if state['w8'] else 'dit_fp16', 2) in state['events']
            assert ('denoise', 2) in state['events']
            assert not any((e[0] in ('constructor', 'plan', 'forward', 'capture') for e in state['events']))
            state['events'].append(('persistent', kind, args))
            if state['fail'] == 'persistent_' + kind:
                raise RuntimeError('synthetic persistent_' + kind + ' failure')
        return reserve
    for (prefix, kind) in (('qwen3vl_vision', 'vision'), ('causal_decoder', 'text'), ('gr00t_vl_encoder', 'vl'), ('gr00t_dit', 'dit')):
        monkeypatch.setattr(torch.ops.rpu, prefix + '_prepare_persistent_spm', persistent(kind), raising=False)

    def envelope(kind):

        def bind(handle, capacity, chunk):
            state['events'].append((kind + '_envelope', handle))
            if state['fail'] == kind + '_envelope':
                raise RuntimeError('synthetic envelope failure')
            state['envelopes'][kind] = (handle, capacity, chunk)
        return bind
    for (prefix, kind) in (('gr00t_vl_encoder', 'vl'), ('gr00t_dit', 'dit')):
        monkeypatch.setattr(torch.ops.rpu, prefix + '_set_chunk_envelope', envelope(kind), raising=False)

    def prepare(*args):
        assert state['envelopes'] == {'vl': (1, 144, 0), 'dit': (2, 384, 0)}
        assert len(state['caches']) == 2
        state['events'].append(('prepare', args))
        if state['fail'] == 'prepare':
            raise RuntimeError('synthetic prepare failure')
        return 7
    for op in ('gr00t_spm_z2_prepare', 'qwen3vl_pooler_spm_z1_prepare'):
        monkeypatch.setattr(torch.ops.rpu, op, prepare, raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'qwen3vl_pooler_spm_z1_vision_stage_descriptor', lambda *args: [1], raising=False)
    for op in ('gr00t_spm_z2_unprepare', 'qwen3vl_pooler_spm_z1_unprepare'):
        monkeypatch.setattr(torch.ops.rpu, op, lambda *args: None, raising=False)
    yield state
    live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF else None
    if live is not None:
        causal_lm._release_live_instance(live)

@pytest.mark.parametrize('w8,z1', [(False, False), (True, False), (False, True)])
def test_actual_gr00t_cold_envelope_precedes_prepare_and_reuses_owner_caches(cold_builder, w8, z1):
    cold_builder.update(w8=w8, z1=z1)
    owner = runtime.build_gr00t_vla('unused', 'unused')
    try:
        assert cold_builder['envelopes'] == {'vl': (1, 144, 0), 'dit': (2, 384, 0)}
        assert len(cold_builder['caches']) == 3
        assert owner._max_prefix_len == 256
        assert [row['max_seq_len'] for row in cold_builder['cache_requests']] == [256, 256, 256]
        assert owner._a_cache is cold_builder['caches'][0]
        assert owner._d_cache is cold_builder['caches'][1]
        assert ('vl_w8' if w8 else 'vl_fp16', 1) in cold_builder['events']
        assert ('dit_w8' if w8 else 'dit_fp16', 2) in cold_builder['events']
        assert bool([e for e in cold_builder['events'] if e[0] == 'prepare']) is z1
        events = cold_builder['events']
        persistent = [event for event in events if event[0] == 'persistent']
        assert persistent == ([] if z1 else [('persistent', 'vision', (91, 256)), ('persistent', 'text', (90, 256, 0, True)), ('persistent', 'vl', (1, 256)), ('persistent', 'dit', (2, 41))])
        assert not any((e[0] in ('plan', 'forward', 'capture') for e in events))
        if z1:
            assert all((events.index(event) < events.index(('constructor',)) for event in events if event[0] == 'prepare'))
        else:
            assert events.index(persistent[-1]) < events.index(('constructor',))
    finally:
        owner.close()

@pytest.mark.parametrize('failure', ['vl_envelope', 'dit_envelope', 'prepare'])
def test_actual_gr00t_cold_envelope_or_prepare_failure_retires_all_pending_handles(cold_builder, failure):
    cold_builder.update(z1=True, fail=failure)
    with pytest.raises(RuntimeError, match='synthetic (envelope|prepare) failure'):
        runtime.build_gr00t_vla('unused', 'unused')
    assert cold_builder['alive'] == {}
    for (kind, handle) in (('dit', 2), ('vl', 1), ('decoder', 90), ('vision', 91)):
        assert cold_builder['events'].count(('destroy', kind, handle)) == 1
    assert not getattr(cold_builder['model'], '_rpu_swizzled', False)

@pytest.mark.parametrize('steps', [4, 10])
def test_actual_gr00t_persistent_barrier_precedes_first_plan(cold_builder, monkeypatch, steps):
    monkeypatch.setenv('RPU_GR00T_PRECOMPUTE_ADALN', '1')
    monkeypatch.setenv('RPU_GR00T_PRECOMPUTE_TIME_GLUE', '1')
    owner = runtime.build_gr00t_vla('unused', 'unused', num_steps=steps)
    try:
        events = cold_builder['events']
        first = events.index(('persistent', 'vision', (91, 256)))
        last = events.index(('persistent', 'dit', (2, 41)))
        assert events.index(('adaln_prepare', 2)) < first
        assert events.index(('time_prepare', 2)) < first
        assert last < events.index(('constructor',))
        for plan in (owner._vl_execution_plan, owner._action_execution_plan):
            with pytest.raises(RuntimeError, match='first native plan boundary'):
                plan(214)
            assert events[-1][0] == 'plan'
            assert events.index(('constructor',)) < len(events) - 1
        assert not any((e[0] in ('forward', 'capture') for e in events))
    finally:
        owner.close()

@pytest.mark.parametrize('missing', ['all', 'vision', 'vl', 'dit'])
def test_actual_gr00t_persistent_api_legacy_or_partial_is_all_or_none(monkeypatch, missing):
    calls = []
    names = {'vision': 'qwen3vl_vision_prepare_persistent_spm', 'vl': 'gr00t_vl_encoder_prepare_persistent_spm', 'dit': 'gr00t_dit_prepare_persistent_spm'}
    for (kind, name) in names.items():
        operation = None if missing in ('all', kind) else lambda *args, kind=kind: calls.append((kind, args))
        monkeypatch.setattr(torch.ops.rpu, name, operation, raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_prepare_persistent_spm', lambda *args: calls.append(('text', args)), raising=False)
    if missing == 'all':
        assert runtime._prepare_gr00t_persistent_spm(91, 90, 1, 2) is None
    else:
        with pytest.raises(RuntimeError, match='Persistent preparation API is incomplete'):
            runtime._prepare_gr00t_persistent_spm(91, 90, 1, 2)
    assert calls == []

@pytest.mark.parametrize('failure', ['vision', 'text', 'vl', 'dit'])
def test_actual_gr00t_persistent_prepare_failure_retires_all_pending_handles(cold_builder, failure):
    cold_builder['fail'] = 'persistent_' + failure
    with pytest.raises(RuntimeError, match='synthetic persistent_' + failure + ' failure'):
        runtime.build_gr00t_vla('unused', 'unused', num_steps=10)
    assert cold_builder['alive'] == {}
    events = cold_builder['events']
    for (kind, handle) in (('dit', 2), ('vl', 1), ('decoder', 90), ('vision', 91)):
        assert events.count(('destroy', kind, handle)) == 1
    prepared = [e[1] for e in events if e[0] == 'persistent']
    order = ['vision', 'text', 'vl', 'dit']
    assert prepared == order[:order.index(failure) + 1]
    assert not any((e[0] in ('constructor', 'plan', 'forward', 'capture') for e in events))
    assert not getattr(cold_builder['model'], '_rpu_swizzled', False)

@pytest.mark.parametrize('build_kwargs,steps', [({}, 4), ({'num_steps': 4}, 4), ({'num_steps': 10}, 10)])
def test_actual_gr00t_cold_schedule_and_matching_action_reuse(cold_builder, monkeypatch, build_kwargs, steps):
    owner = runtime.build_gr00t_vla('unused', 'unused', **build_kwargs)
    try:
        assert cold_builder['denoise_steps'] == [steps]
        assert owner._num_steps == steps
        calls = []
        (raw, vl, action) = (object(), object(), object())
        (vl_plan, action_plan) = (object(), object())
        inputs = {'input_ids': torch.zeros((1, 82), dtype=torch.int64)}
        monkeypatch.setattr(owner, '_validate_forward_execution', lambda value: calls.append(('validate', value)))
        monkeypatch.setattr(owner, '_vl_execution_plan', lambda seq: vl_plan)
        monkeypatch.setattr(owner, '_action_execution_plan', lambda seq: action_plan)
        monkeypatch.setattr(owner, '_backbone', lambda value: (raw, 82))

        def encode(value, seq, plan):
            assert value is raw and seq == 82 and (plan is vl_plan)
            calls.append(('vl', seq))
            return vl

        def denoise(value, actual_inputs, seq, seed, plan):
            assert value is vl and actual_inputs is inputs and (seq == 82) and (plan is action_plan)
            calls.append(('denoise', seed))
            return action
        monkeypatch.setattr(owner, '_vl_encode', encode)
        monkeypatch.setattr(owner, '_denoise', denoise)
        for kwargs in ({}, {'num_steps': steps}):
            calls.clear()
            assert owner.get_action(inputs, seed=17, **kwargs) is action
            assert calls == [('validate', inputs), ('vl', 82), ('denoise', 17)]
        assert cold_builder['denoise_steps'] == [steps]
        before = list(cold_builder['events'])
        for invalid in (10 if steps == 4 else 4, True, float(steps), str(steps)):
            calls.clear()
            with pytest.raises(ValueError, match='num_steps'):
                owner.get_action({}, num_steps=invalid)
            assert calls == [] and cold_builder['events'] == before
    finally:
        owner.close()

@pytest.mark.parametrize('capacity,logical', [(16, 16), (256, 238), (288, 270), (304, 289), (336, 334), (512, 512), (16384, 8200)])
def test_actual_gr00t_prefix_capacity_binds_caches_envelopes_and_reconfigure(cold_builder, monkeypatch, capacity, logical):
    from rpu_backend.tests.acceptance.test_gr00t_execution_config import _install_reconfigure_ops
    _install_reconfigure_ops(monkeypatch)
    cold_builder['honor_capacity'] = True
    exact = {'components': {'vl_encoder': {'prefill': {'chunk_size': capacity}}}}
    owner = runtime.build_gr00t_vla('unused', 'unused', max_prefix_len=capacity, rpu_execution=exact)
    try:
        assert owner._max_prefix_len == capacity and owner._num_steps == 4
        assert [row['max_seq_len'] for row in cold_builder['cache_requests']] == [capacity, capacity, capacity]
        assert cold_builder['envelopes'] == {'vl': (1, capacity, 0), 'dit': (2, capacity, 0)}
        assert owner._a_cache.max_seq_len == owner._d_cache.max_seq_len == capacity
        assert owner._bb_cache.max_seq_len == capacity
        assert owner._vl_execution['prefill']['chunk_size'] == capacity
        caches = tuple(cold_builder['caches'])
        owner._execution_session.reconfigure({'components': {'vl_encoder': {'prefill': {'chunk_size': 'auto'}}}})
        owner._execution_session.reconfigure(exact)
        assert tuple(cold_builder['caches']) == caches and owner._max_prefix_len == capacity
        before = list(cold_builder['events'])
        with pytest.raises(ValueError, match=f'{capacity}-row cache/envelope'):
            owner._execution_session.reconfigure({'components': {'vl_encoder': {'prefill': {'chunk_size': capacity + 16}}}})
        assert cold_builder['events'] == before
        monkeypatch.setattr(owner, '_validate_forward_execution', lambda inputs: None)
        inputs = {'input_ids': torch.zeros((1, logical), dtype=torch.int64)}
        with pytest.raises(RuntimeError, match='first native plan boundary'):
            owner.get_action(inputs)
        for cache in (owner._bb_cache, owner._a_cache, owner._d_cache):
            original = cache.max_seq_len
            cache.max_seq_len = logical - 1
            before = list(cold_builder['events'])
            with pytest.raises(ValueError, match='KV-cache capacity'):
                owner.get_action(inputs)
            assert cold_builder['events'] == before
            cache.max_seq_len = original
        with pytest.raises(ValueError, match='KV-cache capacity'):
            owner.get_action({'input_ids': torch.zeros((1, capacity + 1), dtype=torch.int64)})
        assert not any((event[0] in ('forward', 'capture') for event in cold_builder['events']))
    finally:
        owner.close()

@pytest.mark.parametrize('capacity', [None, True, False, 0, -16, 15, 257, 256.0, '256'])
def test_actual_gr00t_invalid_prefix_capacity_precedes_model_read(cold_builder, capacity):
    with pytest.raises(ValueError, match='max_prefix_len'):
        runtime.build_gr00t_vla('unused', 'unused', max_prefix_len=capacity)
    assert cold_builder['model_reads'] == cold_builder['cache_requests'] == cold_builder['events'] == []

@pytest.mark.parametrize('capacity', [16, 240, 272, 512])
def test_actual_gr00t_z1_nondefault_prefix_capacity_precedes_model_read(cold_builder, capacity):
    cold_builder['z1'] = True
    with pytest.raises(ValueError, match='Z1 requires max_prefix_len=256'):
        runtime.build_gr00t_vla('unused', 'unused', max_prefix_len=capacity)
    assert cold_builder['model_reads'] == cold_builder['cache_requests'] == cold_builder['events'] == []

def test_actual_gr00t_prefix_execution_limit_fails_before_model_read(cold_builder):
    with pytest.raises(ValueError, match='304-row cache/envelope'):
        runtime.build_gr00t_vla('unused', 'unused', max_prefix_len=304, rpu_execution={'components': {'vl_encoder': {'prefill': {'chunk_size': 320}}}})
    assert cold_builder['model_reads'] == cold_builder['cache_requests'] == cold_builder['events'] == []

@pytest.mark.parametrize('value', [None, True, False, 0, -1, 3, 5, 11, 4.0, 10.0, '10'])
def test_actual_gr00t_invalid_cold_schedule_precedes_model_read(cold_builder, value):
    with pytest.raises(ValueError, match='num_steps'):
        runtime.build_gr00t_vla('unused', 'unused', num_steps=value)
    assert cold_builder['model_reads'] == []
    assert cold_builder['denoise_steps'] == []
    assert cold_builder['events'] == [] and cold_builder['caches'] == []

def test_actual_gr00t_z1_ten_step_schedule_rejects_before_model_read(cold_builder):
    cold_builder['z1'] = True
    with pytest.raises(ValueError, match='num_steps|four|4'):
        runtime.build_gr00t_vla('unused', 'unused', num_steps=10)
    assert cold_builder['model_reads'] == []
    assert cold_builder['denoise_steps'] == []
    assert cold_builder['events'] == [] and cold_builder['caches'] == []

@pytest.mark.parametrize('steps', [4, 10])
def test_actual_build_denoise_bakes_complete_schedule_on_cpu(monkeypatch, steps):
    monkeypatch.setattr(runtime, '_NL', 0)
    monkeypatch.setattr(runtime, '_w8a16_on', lambda : False)
    monkeypatch.setattr(runtime, '_half_rpu', lambda tensor: tensor.half().contiguous())
    monkeypatch.setattr(runtime, 'tp_col_swizzle_mc_weight', lambda tensor, *args: tensor)
    monkeypatch.setattr(runtime, 'tp_row_swizzle_mc_weight', lambda tensor, *args: tensor)
    (hidden, horizon) = (1536, 40)
    w1 = torch.zeros(4, 256)
    (w1[0, 0], w1[1, 128], w1[2, 1], w1[3, 129]) = (1, 1, 1, 1)
    b1 = torch.tensor([0.1, -0.2, 0.3, -0.4])
    w2 = torch.linspace(-0.2, 0.2, hidden * 4).reshape(hidden, 4)
    b2 = torch.linspace(-0.1, 0.1, hidden)
    position = torch.arange(horizon * hidden, dtype=torch.float32).reshape(horizon, hidden) / 10000
    table_weights = {'model.timestep_encoder.timestep_embedder.linear_1.weight': w1, 'model.timestep_encoder.timestep_embedder.linear_1.bias': b1, 'model.timestep_encoder.timestep_embedder.linear_2.weight': w2, 'model.timestep_encoder.timestep_embedder.linear_2.bias': b2, 'position_embedding.weight': position}

    def weight(key):
        name = key.removeprefix(runtime._AP)
        if name in table_weights:
            return table_weights[name]
        if name == 'action_encoder.W2.W':
            return torch.zeros(1, hidden * 2, 1)
        if name.endswith('.W'):
            return torch.zeros(1, 2, 2)
        if name.endswith('.b'):
            return torch.zeros(1, 2)
        return torch.zeros(2, 2) if name.endswith('.weight') else torch.zeros(2)
    configured = {}

    def configure(**kwargs):
        configured.update(kwargs)
        return 71
    monkeypatch.setattr(runtime, '_configure_gr00t_denoise_handle', configure)
    (handle, keep) = runtime._build_denoise(weight, 0, num_steps=steps)
    args = configured['denoise_args']
    (cond, tau, pos) = args[19:22]
    assert handle == 71 and keep and (configured['w8a16'] is False)
    assert cond.shape == (steps, hidden) and tau.shape == (steps, horizon, hidden)
    assert pos.shape == (horizon, hidden)
    assert args[22:26] == (132, 144, steps, 1.0 / steps)
    assert all((tensor.device.type == 'cpu' and tensor.dtype == torch.float16 for tensor in (cond, tau, pos)))
    buckets = torch.tensor(range(0, 1000, 1000 // steps), dtype=torch.float32)
    if steps == 10:
        assert buckets.tolist() == list(range(0, 1000, 100)) and args[25] == 0.1
    cond_angles = buckets[:, None] * torch.exp(-math.log(10000.0) * torch.arange(128) / 127)[None, :]
    embeddings = torch.cat((cond_angles.cos(), cond_angles.sin()), dim=1)
    expected_cond = torch.cat([torch.nn.functional.silu(torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(row[None, :], w1, b1)), w2, b2)) for row in embeddings])
    tau_angles = buckets[:, None] * torch.exp(-torch.arange(hidden // 2) * (math.log(10000.0) / (hidden // 2)))[None, :]
    expected_tau = torch.cat((tau_angles.sin(), tau_angles.cos()), dim=1)[:, None, :].expand(-1, horizon, -1)
    torch.testing.assert_close(cond, expected_cond.half(), rtol=0, atol=0)
    torch.testing.assert_close(tau, expected_tau.half(), rtol=0, atol=0)
    torch.testing.assert_close(pos, position.half(), rtol=0, atol=0)
    assert not torch.equal(cond[0], cond[-1]) and (not torch.equal(tau[0], tau[-1]))
