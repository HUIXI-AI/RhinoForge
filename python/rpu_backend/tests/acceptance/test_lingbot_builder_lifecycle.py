from __future__ import annotations
from contextlib import nullcontext
import inspect
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from rpu_backend.adapters.lingbot_vla import runtime
from rpu_backend.api import causal_lm
from rpu_backend.api._execution import bind_execution_session, reconfigure_rpu_execution
from rpu_backend.runtime.execution_planner import StageTuple
_REPO_ROOT = Path(__file__).resolve().parents[4]

class _Cache:

    def __init__(self, events, name):
        self._events = events
        self._name = name

    def clear(self):
        self._events.append(('clear', self._name))

    def cache_invariant_ok(self):
        return True

class _ForwardCache:

    def __init__(self):
        self.events = []
        self.k_caches = []
        self.v_caches = []

    def reset_to_position(self, position):
        self.events.append(('reset', position))

    def update_position(self, position):
        self.events.append(('update', position))

class _CaptureCache:

    def __init__(self):
        self.signatures = []

    def capture(self, signature):
        self.signatures.append(signature)
        return nullcontext()

class _HotCache(_Cache):

    def __init__(self, events, name, *, fail_count=0):
        super().__init__(events, name)
        self._fail_count = fail_count

    def begin_warmup(self):
        self._events.append(('warm', self._name))

    def clear(self):
        super().clear()
        if self._fail_count:
            self._fail_count -= 1
            raise RuntimeError('synthetic graph cleanup failure')

    def cache_invariant_ok(self):
        return True

class _TensorSlice:

    def __init__(self, shape, dtype='F32'):
        self._shape = shape
        self._dtype = dtype

    def get_shape(self):
        return self._shape

    def get_dtype(self):
        return self._dtype

class _ShapeReader:

    def __init__(self, metadata):
        self._metadata = metadata

    def get_slice(self, key):
        (shape, dtype) = self._metadata[key]
        return _TensorSlice(shape, dtype)

class _KeyReader:

    def __init__(self, keys):
        self._keys = keys

    def keys(self):
        return self._keys

def _stage_plan(logical_len, chunk_size, *, position=0):
    box = {}
    runtime.plan_bounded_prefill_execution(logical_len, logical_len, 0, position=position, padding_rows=0, exact_chunk_size=chunk_size, resolve_stage_domain=lambda _length: (StageTuple(chunk_size, chunk_size, chunk_size, physical_descriptor=(1, logical_len, position, chunk_size)),), plan_result_sink=lambda result: box.__setitem__('result', result), graph_mode=runtime.GRAPH_COMPOSITE_CHILD)
    return box['result']

@pytest.fixture
def decoder_ops(monkeypatch):
    state = {'next': 1, 'alive': set(), 'events': [], 'error': None}

    def create(*_args):
        handle = state['next']
        state['next'] += 1
        state['alive'].add(handle)
        state['events'].append(('create', handle))
        return handle

    def set_weights(handle, *args):
        state['events'].append(('set', handle))
        if state['error'] is not None:
            raise state['error']

    def destroy(handle):
        state['events'].append(('destroy', handle))
        state['alive'].remove(handle)
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_create', create, raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_set_weights', set_weights, raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_set_explicit_mask_spm_residency', lambda handle, enabled: state['events'].append(('mask_residency', handle, enabled)), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_set_chunk_envelope', lambda handle, max_kv_len, chunk: state['events'].append(('envelope', handle, max_kv_len, chunk)), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_set_chunk_size_override', lambda handle, chunk: state['events'].append(('chunk', handle, chunk)), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_destroy', destroy, raising=False)
    monkeypatch.setattr(runtime, 'VLM_LAYERS', 1)
    monkeypatch.setattr(runtime, 'EXP_LAYERS', 1)
    monkeypatch.setattr(runtime, '_h', lambda value: value)
    monkeypatch.setattr(runtime, 'tp_col_swizzle_mc_weight', lambda value, cores: value)
    monkeypatch.setattr(runtime, 'tp_row_swizzle_mc_weight', lambda value, cores: value)
    monkeypatch.setattr(runtime, '_build_rope_tables', lambda *args: (torch.zeros(1), torch.zeros(1)))
    return state

@pytest.mark.parametrize('builder', ['_build_vlm_decoder', '_build_expert'])
def test_decoder_configuration_base_exception_destroys_raw_handle(decoder_ops, builder):

    class BuildInterrupted(BaseException):
        pass
    decoder_ops['error'] = BuildInterrupted(f'synthetic {builder} interruption')
    weight = torch.zeros(2, 2)
    with pytest.raises(BuildInterrupted, match=builder):
        getattr(runtime, builder)(lambda key: weight, 64)
    assert decoder_ops['alive'] == set()
    assert decoder_ops['events'][-1][0] == 'destroy'

def test_decoder_builders_apply_certified_exact_envelopes(decoder_ops):
    weight = torch.zeros(2, 2)
    (vlm, _) = runtime._build_vlm_decoder(lambda key: weight, 256, 80)
    (expert, _) = runtime._build_expert(lambda key: weight, 256, 64)
    assert decoder_ops['events'] == [('create', vlm), ('set', vlm), ('envelope', vlm, 256, 176), ('chunk', vlm, 80), ('create', expert), ('mask_residency', expert, True), ('set', expert), ('envelope', expert, 256, 64), ('chunk', expert, 64)]

def test_vlm_auto_keeps_certified_capability_for_later_exact_requests(decoder_ops):
    weight = torch.zeros(2, 2)
    (vlm, _) = runtime._build_vlm_decoder(lambda key: weight, 256, 0)
    assert decoder_ops['events'] == [('create', vlm), ('set', vlm), ('envelope', vlm, 256, 176), ('chunk', vlm, 0)]

def test_fused_denoise_configuration_base_exception_destroys_handle(monkeypatch):

    class BuildInterrupted(BaseException):
        pass
    events = []
    enc_w_comp = torch.zeros(2, 1)
    head_w = {'model.action_time_mlp_out.weight': torch.zeros(2, 2), 'model.action_time_mlp_out.bias': torch.zeros(2), 'model.action_out_proj.weight': torch.zeros(1, 2), 'model.action_out_proj.bias': torch.zeros(1)}
    time_parts = [torch.zeros(1, 2) for _ in range(runtime.NUM_STEPS)]
    monkeypatch.setattr(runtime, '_h', lambda value: value)
    monkeypatch.setattr(runtime, 'tp_col_swizzle_mc_weight', lambda value, cores: value)
    monkeypatch.setattr(runtime.torch, 'zeros', lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime.torch, 'empty', lambda *args, **kwargs: object())
    monkeypatch.setattr(torch.ops.rpu, 'lingbot_denoise_create', lambda : events.append(('create', 7)) or 7, raising=False)

    def fail_set(*args, **kwargs):
        events.append(('set', 7))
        raise BuildInterrupted('synthetic denoise interruption')
    monkeypatch.setattr(torch.ops.rpu, 'lingbot_denoise_set_weights', fail_set, raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'lingbot_denoise_destroy', lambda handle: events.append(('destroy', handle)), raising=False)
    owner = runtime.LingbotVlaPolicy.__new__(runtime.LingbotVlaPolicy)
    owner._du_handle = None
    owner._exp_keep = tuple((object() for _ in range(15)))
    owner._enc_W_comp = enc_w_comp
    owner._head_W = head_w
    owner._enc_time_parts = time_parts
    owner._fold_all = tuple((object() for _ in range(4)))
    with pytest.raises(BuildInterrupted, match='denoise'):
        owner._build_denoise_unroll()
    assert owner._du_handle is None
    assert events == [('create', 7), ('set', 7), ('destroy', 7)]

def test_fused_denoise_applies_action_chunk_before_publication(monkeypatch):
    events = []
    enc_w_comp = torch.zeros(2, 1)
    head_w = {'model.action_time_mlp_out.weight': torch.zeros(2, 2), 'model.action_time_mlp_out.bias': torch.zeros(2), 'model.action_out_proj.weight': torch.zeros(1, 2), 'model.action_out_proj.bias': torch.zeros(1)}
    time_parts = [torch.zeros(1, 2) for _ in range(runtime.NUM_STEPS)]
    monkeypatch.setattr(runtime, '_h', lambda value: value)
    monkeypatch.setattr(runtime, 'tp_col_swizzle_mc_weight', lambda value, cores: value)
    monkeypatch.setattr(runtime.torch, 'zeros', lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime.torch, 'empty', lambda *args, **kwargs: object())
    monkeypatch.setattr(torch.ops.rpu, 'lingbot_denoise_create', lambda : events.append(('create', 8)) or 8, raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'lingbot_denoise_set_weights', lambda *args: events.append(('set', args[0])), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'lingbot_denoise_set_chunk_size', lambda handle, chunk: events.append(('chunk', handle, chunk)), raising=False)
    owner = runtime.LingbotVlaPolicy.__new__(runtime.LingbotVlaPolicy)
    owner._du_handle = None
    owner._exp_keep = tuple((object() for _ in range(15)))
    owner._enc_W_comp = enc_w_comp
    owner._head_W = head_w
    owner._enc_time_parts = time_parts
    owner._fold_all = tuple((object() for _ in range(4)))
    owner._action_chunk_size = 64
    owner._build_denoise_unroll()
    assert owner._du_handle == 8
    assert events == [('create', 8), ('set', 8), ('chunk', 8, 64)]

def _checkpoint_shape_readers():
    metadata = {key: (shape, 'F32') for (key, shape) in runtime._expected_checkpoint_shapes().items()}
    reader = _ShapeReader(metadata)
    return (metadata, {key: reader for key in metadata})

def test_certified_checkpoint_profile_accepts_all_consumed_tensor_geometry():
    (metadata, readers) = _checkpoint_shape_readers()
    runtime._validate_checkpoint_profile(readers)
    assert len(metadata) == 1555

def test_checkpoint_reader_opens_shards_deterministically(monkeypatch):
    opened = []
    monkeypatch.setattr(runtime.glob, 'glob', lambda pattern: ['/ckpt/z.safetensors', '/ckpt/a.safetensors'])

    def open_reader(path, framework):
        opened.append((path, framework))
        return _KeyReader([path])
    monkeypatch.setattr(runtime, 'safe_open', open_reader)
    (_, readers) = runtime._ckpt_reader('/ckpt')
    assert opened == [('/ckpt/a.safetensors', 'pt'), ('/ckpt/z.safetensors', 'pt')]
    assert list(readers) == ['/ckpt/a.safetensors', '/ckpt/z.safetensors']

def test_checkpoint_reader_rejects_duplicate_tensor_keys(monkeypatch):
    monkeypatch.setattr(runtime.glob, 'glob', lambda pattern: ['/ckpt/a.safetensors', '/ckpt/b.safetensors'])
    monkeypatch.setattr(runtime, 'safe_open', lambda path, framework: _KeyReader(['duplicate.weight']))
    with pytest.raises(ValueError, match='duplicate tensor key'):
        runtime._ckpt_reader('/ckpt')

@pytest.mark.parametrize('defect', ['missing', 'shape', 'dtype', 'depth'])
def test_checkpoint_profile_rejects_defect_before_rpu_materialization(defect):
    (metadata, readers) = _checkpoint_shape_readers()
    key = runtime.VLM_P + 'layers.0.self_attn.q_proj.weight'
    if defect == 'missing':
        readers.pop(key)
    elif defect == 'shape':
        metadata[key] = ((1, 1), 'F32')
    elif defect == 'dtype':
        metadata[key] = (metadata[key][0], 'F16')
    else:
        extra = runtime.VLM_P + f'layers.{runtime.VLM_LAYERS}.extra.weight'
        metadata[extra] = ((1,), 'F32')
        readers[extra] = next(iter(readers.values()))
    with pytest.raises(ValueError, match='certified 4B fp32 profile'):
        runtime._validate_checkpoint_profile(readers)

def test_policy_close_clears_all_graphs_before_handles_and_releases_once(monkeypatch):
    events = []
    owner = runtime.LingbotVlaPolicy.__new__(runtime.LingbotVlaPolicy)
    owner._closed = False
    owner._live_slot_released = False
    owner._dgc = _Cache(events, 'denoise')
    owner._egc = _Cache(events, 'expert')
    owner._vgc = _Cache(events, 'vlm')
    owner._cache = object()
    owner._suffix_buf = object()
    owner._vlm = 11
    owner._exp = 12
    owner._du_handle = 13
    monkeypatch.setattr(runtime, '_destroy_denoise_handle', lambda handle: events.append(('destroy', handle)))
    monkeypatch.setattr(runtime, '_destroy_causal_decoder_handle', lambda handle: events.append(('destroy', handle)))
    monkeypatch.setattr(causal_lm, '_release_live_instance', lambda policy: events.append(('release', policy)))
    owner.close()
    owner.close()
    assert events == [('clear', 'denoise'), ('clear', 'expert'), ('clear', 'vlm'), ('destroy', 13), ('destroy', 12), ('destroy', 11), ('release', owner)]
    assert owner._cache is None
    assert owner._suffix_buf is None
    assert owner._vlm is None
    assert owner._exp is None
    assert owner._du_handle is None

def test_policy_close_keeps_process_slot_when_cleanup_fails(monkeypatch):
    events = []

    class FailingCache(_Cache):

        def clear(self):
            super().clear()
            raise RuntimeError('synthetic graph cleanup failure')
    owner = runtime.LingbotVlaPolicy.__new__(runtime.LingbotVlaPolicy)
    owner._closed = False
    owner._cleanup_ok = None
    owner._live_slot_released = False
    owner._dgc = FailingCache(events, 'denoise')
    owner._egc = _Cache(events, 'expert')
    owner._vgc = _Cache(events, 'vlm')
    monkeypatch.setattr(causal_lm, '_release_live_instance', lambda policy: pytest.fail('failed cleanup must not release owner'))
    with pytest.raises(RuntimeError, match='synthetic graph cleanup failure'):
        owner.close()
    with pytest.raises(RuntimeError, match='cleanup failed'):
        owner.close()
    assert events == [('clear', 'denoise')]
    assert owner._closed is False
    assert owner._cleanup_ok is False
    assert owner._live_slot_released is False

def test_partial_policy_cleanup_uses_reverse_handle_order(monkeypatch):
    events = []
    owner = runtime.LingbotVlaPolicy.__new__(runtime.LingbotVlaPolicy)
    owner._closed = False
    owner._dgc = _Cache(events, 'denoise')
    owner._egc = _Cache(events, 'expert')
    owner._vgc = _Cache(events, 'vlm')
    owner._vlm = 11
    owner._exp = 12
    owner._du_handle = 13
    monkeypatch.setattr(runtime, '_destroy_denoise_handle', lambda handle: events.append(('destroy', 'denoise', handle)) or True)
    monkeypatch.setattr(runtime, '_destroy_causal_decoder_handle', lambda handle: events.append(('destroy', 'decoder', handle)) or True)
    owner._retire_resources()
    owner._retire_resources()
    assert events == [('clear', 'denoise'), ('clear', 'expert'), ('clear', 'vlm'), ('destroy', 'denoise', 13), ('destroy', 'decoder', 12), ('destroy', 'decoder', 11)]

def _patch_builder_cpu_preflight(monkeypatch):
    monkeypatch.setattr(runtime, 'EXP_LAYERS', 0)
    monkeypatch.setattr(runtime, '_ckpt_reader', lambda ckpt: (lambda key: object(), {}))
    monkeypatch.setattr(runtime, '_validate_checkpoint_profile', lambda readers: None)
    monkeypatch.setattr(runtime.torch.rpu, 'set_ddr_flush', lambda enabled: None)

def test_execution_config_reaches_all_cold_handles(monkeypatch):
    _patch_builder_cpu_preflight(monkeypatch)
    events = []
    monkeypatch.setattr(causal_lm, '_claim_live_instance', lambda owner: events.append(('claim', owner)))
    monkeypatch.setattr(runtime, '_build_vlm_decoder', lambda gw, max_seq, chunk: events.append(('vlm', max_seq, chunk)) or (21, object()))
    monkeypatch.setattr(runtime, '_build_expert', lambda gw, max_seq, chunk: events.append(('expert', max_seq, chunk)) or (22, object()))

    def init_policy(self, **kwargs):
        events.append(('policy', kwargs['prefill_chunk_size'], kwargs['action_chunk_size']))
    monkeypatch.setattr(runtime.LingbotVlaPolicy, '__init__', init_policy)
    policy = runtime.build_lingbot_vla('unused', rpu_execution={'prefill': {'chunk_size': 80}, 'action': {'chunk_size': 64}})
    assert events[1:] == [('vlm', 256, 80), ('expert', 256, 64), ('policy', 80, 64)]
    assert dict(policy._rpu_execution['prefill']) == {'chunk_size': 80}
    assert dict(policy._rpu_execution['action']) == {'chunk_size': 64}

def test_execution_auto_reaches_both_native_planners(monkeypatch):
    _patch_builder_cpu_preflight(monkeypatch)
    chunks = []
    monkeypatch.setattr(causal_lm, '_claim_live_instance', lambda owner: None)
    monkeypatch.setattr(runtime, '_build_vlm_decoder', lambda gw, max_seq, chunk: chunks.append(('vlm', chunk)) or (31, object()))
    monkeypatch.setattr(runtime, '_build_expert', lambda gw, max_seq, chunk: chunks.append(('expert', chunk)) or (32, object()))
    monkeypatch.setattr(runtime.LingbotVlaPolicy, '__init__', lambda self, **kw: None)
    runtime.build_lingbot_vla('unused', rpu_execution={'prefill': {'chunk_size': 'auto'}, 'action': {'chunk_size': 'auto'}})
    assert chunks == [('vlm', 0), ('expert', 0)]

def test_execution_component_ids_resolve_defaults_and_child_overrides():
    (root, vlm, action) = runtime._resolve_execution_components({'prefill': {'chunk_size': 80}, 'action': {'chunk_size': 64}, 'components': {'language_model': {'prefill': {'chunk_size': 96}}, 'action_expert': {'action': {'chunk_size': 'auto'}}, 'vision_encoder': {}}}, entry_point='test')
    assert root['prefill']['chunk_size'] == 80
    assert vlm == {'prefill': {'chunk_size': 96}}
    assert action == {'action': {'chunk_size': 'auto'}}
    assert runtime.resolve_component_rpu_execution(root, 'vision_encoder', entry_point='test', supported_components=runtime._EXECUTION_COMPONENTS, profile_auto={}) == {}

def test_builder_uses_component_override_after_top_level_default(monkeypatch):
    _patch_builder_cpu_preflight(monkeypatch)
    chunks = []
    monkeypatch.setattr(causal_lm, '_claim_live_instance', lambda owner: None)
    monkeypatch.setattr(runtime, '_build_vlm_decoder', lambda gw, max_seq, chunk: chunks.append(('vlm', chunk)) or (31, object()))
    monkeypatch.setattr(runtime, '_build_expert', lambda gw, max_seq, chunk: chunks.append(('expert', chunk)) or (32, object()))
    monkeypatch.setattr(runtime.LingbotVlaPolicy, '__init__', lambda self, **kwargs: None)
    runtime.build_lingbot_vla('unused', rpu_execution={'prefill': {'chunk_size': 80}, 'action': {'chunk_size': 64}, 'components': {'language_model': {'prefill': {'chunk_size': 96}}, 'action_expert': {'action': {'chunk_size': 'auto'}}}})
    assert chunks == [('vlm', 96), ('expert', 0)]

@pytest.mark.parametrize(('execution', 'error'), [({'components': {'unknown': {}}}, 'unknown component'), ({'components': {'vision_encoder': {'vision': {'chunk_size': 'auto'}}}}, 'not supported'), ({'components': {'action_expert': {'prefill': {'chunk_size': 64}}}}, 'not supported')])
def test_execution_component_config_fails_before_claim(monkeypatch, execution, error):
    monkeypatch.setattr(causal_lm, '_claim_live_instance', lambda owner: pytest.fail('claim must not run'))
    with pytest.raises(ValueError, match=error):
        runtime.build_lingbot_vla('unused', rpu_execution=execution)

def _bind_hot_policy(monkeypatch, events, *, graph_fail_count=0, cold_execution=None):
    token = {'next': 90}

    def begin(_attempt):
        token['next'] += 1
        events.append(('txn', 'begin', token['next']))
        return token['next']
    monkeypatch.setattr(torch.ops.rpu, 'execution_reconfigure_begin', begin, raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'execution_reconfigure_commit', lambda value: events.append(('txn', 'commit', value)), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'execution_reconfigure_abort', lambda value: events.append(('txn', 'abort', value)), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'execution_reconfigure_abort_attempt', lambda attempt: events.append(('txn', 'abort_attempt', attempt)) or False, raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_stage_chunk_size_override', lambda handle, value, chunk: events.append(('stage', handle, value, chunk)), raising=False)
    owner = runtime.LingbotVlaPolicy.__new__(runtime.LingbotVlaPolicy)
    owner._closed = False
    owner._vlm = 11
    owner._exp = 22
    owner._dgc = _HotCache(events, 'denoise', fail_count=graph_fail_count)
    owner._egc = _HotCache(events, 'expert')
    owner._vgc = _HotCache(events, 'vlm')
    owner._mask_cache = {82: object()}
    (root, vlm, action) = runtime._resolve_execution_components(cold_execution, entry_point='test')
    owner._rpu_execution = root
    owner._vlm_execution = vlm
    owner._action_execution = action
    owner._prefill_chunk_size = 0 if vlm['prefill']['chunk_size'] == 'auto' else vlm['prefill']['chunk_size']
    owner._action_chunk_override = 0 if action['action']['chunk_size'] == 'auto' else action['action']['chunk_size']
    owner._action_chunk_size = 64
    owner._execution_generation = 0
    owner._execution_reconfigure_journal = None
    owner._rpu_last_execution_plan = {'old': {}}
    owner._execution_session = bind_execution_session(owner, root, entry_point='LingbotVlaPolicy', supported_components=runtime._EXECUTION_COMPONENTS, validate=owner.validate_execution_reconfigure, apply=lambda old, new, generation, *, force_rebuild=False: owner.apply_execution_reconfigure(new, generation, rollback_config=old), rollback=lambda old, _new, generation: owner.rollback_execution_reconfigure(old, generation), graph_mode=runtime.GRAPH_COMPOSITE_CHILD)
    return owner

def test_hot_reconfigure_stages_both_decoders_and_retires_exact_graph_owners(monkeypatch):
    events = []
    owner = _bind_hot_policy(monkeypatch, events)
    published = reconfigure_rpu_execution(owner, {'components': {'language_model': {'prefill': {'chunk_size': 80}}, 'action_expert': {'action': {'chunk_size': 64}}}})
    assert published is owner._rpu_execution
    assert owner._prefill_chunk_size == 80
    assert owner._action_chunk_override == 64
    assert owner._execution_generation == 1
    assert owner._mask_cache == {}
    staged = [event for event in events if event[0] == 'stage']
    assert [event[1:] for event in staged] == [(11, 91, 80), (22, 91, 64)]
    assert events.index(('txn', 'commit', 91)) < events.index(('warm', 'denoise'))
    assert [event for event in events if event[0] == 'clear'] == [('clear', 'denoise'), ('clear', 'expert'), ('clear', 'vlm')]

def test_same_config_calibration_retires_actual_lingbot_v1_graphs(monkeypatch):
    from rpu_backend.api._execution import record_planner_native
    from rpu_backend.runtime.execution_planner import PlannerCalibrationSelection, PlannerCostScope
    events = []
    owner = _bind_hot_policy(monkeypatch, events)
    session = owner._execution_session
    session._bind_planner_owner(owner, 'language_model')
    record_planner_native(owner, 'prefill', 'language_model', ('causal_decoder', 11))
    selection = PlannerCalibrationSelection(PlannerCostScope('a' * 64, 'b' * 64, 'c' * 64), 'd' * 64, 'e' * 64)
    old = session.config
    session._select_planner_calibration(owner, 'prefill', selection, component='language_model', native_prefix='causal_decoder')
    assert session.config == old and session.generation == owner._execution_generation == 1
    assert [event for event in events if event[0] == 'clear'] == [('clear', 'denoise'), ('clear', 'expert'), ('clear', 'vlm')]

def test_hot_reconfigure_cleanup_failure_rolls_back_both_native_controls(monkeypatch):
    events = []
    owner = _bind_hot_policy(monkeypatch, events, graph_fail_count=1)
    with pytest.raises(RuntimeError, match='synthetic graph cleanup failure'):
        reconfigure_rpu_execution(owner, {'prefill': {'chunk_size': 80}, 'action': {'chunk_size': 64}})
    staged = [event for event in events if event[0] == 'stage']
    assert [(event[1], event[3]) for event in staged] == [(11, 80), (22, 64), (11, 0), (22, 0)]
    assert owner._prefill_chunk_size == 0
    assert owner._action_chunk_override == 0
    assert owner._execution_session.generation == 0
    assert owner._execution_session.stats()['state'] == 'QUIESCENT'

def test_hot_reconfigure_begin_failure_does_not_start_rollback_transaction(monkeypatch):
    events = []
    owner = _bind_hot_policy(monkeypatch, events)
    original_config = owner._execution_session.config
    original_owner_config = owner._rpu_execution

    def fail_begin(attempt):
        events.append(('txn', 'begin_fail', attempt))
        raise RuntimeError('synthetic begin failure')
    monkeypatch.setattr(torch.ops.rpu, 'execution_reconfigure_begin', fail_begin, raising=False)
    with pytest.raises(RuntimeError, match='synthetic begin failure'):
        reconfigure_rpu_execution(owner, {'prefill': {'chunk_size': 80}})
    assert len([event for event in events if event[0:2] == ('txn', 'begin_fail')]) == 1
    assert not any((event[0] == 'stage' for event in events))
    assert owner._execution_reconfigure_journal is None
    assert owner._execution_session.config is original_config
    assert owner._rpu_execution is original_owner_config
    assert owner._prefill_chunk_size == 0
    assert owner._action_chunk_override == 0
    assert owner._execution_generation == 0
    assert owner._execution_session.stats()['state'] == 'QUIESCENT'

def test_hot_reconfigure_failed_rollback_poisons_session(monkeypatch):
    owner = _bind_hot_policy(monkeypatch, [], graph_fail_count=2)
    with pytest.raises(RuntimeError, match='synthetic graph cleanup failure'):
        reconfigure_rpu_execution(owner, {'prefill': {'chunk_size': 80}})
    assert owner._execution_session.stats()['state'] == 'POISONED'
    with pytest.raises(RuntimeError, match='poisoned'):
        reconfigure_rpu_execution(owner, {'prefill': {'chunk_size': 96}})

def test_get_action_and_close_use_common_execution_session(monkeypatch):
    source = inspect.getsource(runtime.LingbotVlaPolicy.get_action)
    assert '@execution_serialized' in source
    events = []
    owner = runtime.LingbotVlaPolicy.__new__(runtime.LingbotVlaPolicy)
    owner._execution_session = SimpleNamespace(shutdown=lambda retire: events.append('shutdown') or retire())

    def retire():
        events.append('retire')
        owner._closed = True
    owner._close_without_session = retire
    owner.close()
    assert events == ['shutdown', 'retire']

def test_prefill_dry_plan_rejects_uncertified_values_before_graph_mutation(monkeypatch):
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_planner_cache_identity', lambda handle: (handle, 1), raising=False)
    owner = runtime.LingbotVlaPolicy.__new__(runtime.LingbotVlaPolicy)
    owner._vlm_replay = False
    owner._vlm = 40
    owner._max_seq = 256
    owner._prefill_chunk_size = 80
    with pytest.raises(ValueError, match='requested=80.*ceil16.*64'):
        owner._plan_prefill_execution(64)
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_resolve_prefill_stage_domain', lambda handle, seq, position, is_causal, mask_kv_len, rope_mode, graph_lifecycle, logical_len: (StageTuple(96 if seq == 81 else 80, 96 if seq == 81 else 80, 96 if seq == 81 else 80, physical_descriptor=(1, seq, position)),), raising=False)
    assert owner._plan_prefill_execution(65).selected.stage_tuple.compute_chunk == 80
    assert owner._plan_prefill_execution(80).selected.stage_tuple.compute_chunk == 80
    assert owner._plan_prefill_execution(179).selected.stage_tuple.compute_chunk == 80
    with pytest.raises(ValueError):
        owner._plan_prefill_execution(206)
    with pytest.raises(ValueError, match='no certified native plan.*81'):
        owner._plan_prefill_execution(81)
    owner._prefill_chunk_size = 0
    assert owner._plan_prefill_execution(81).selected.stage_tuple.compute_chunk == 96
    source = inspect.getsource(runtime.LingbotVlaPolicy.get_action)
    assert source.index('_plan_prefill_execution') < source.index('self._vgc.clear()')

@pytest.mark.parametrize('requested', (0, 64))
def test_action_dry_plan_uses_fixed_c64_and_exact_mask_geometry(monkeypatch, requested):
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_planner_cache_identity', lambda handle: (handle, 1), raising=False)
    owner = runtime.LingbotVlaPolicy.__new__(runtime.LingbotVlaPolicy)
    owner._expert_replay = False
    owner._exp = 43
    owner._action_chunk_override = requested
    calls = []

    def resolve(handle, length, position, is_causal, mask_kv_len, rope_mode, graph_lifecycle, logical_len):
        calls.append((handle, length, position, is_causal, mask_kv_len, rope_mode, graph_lifecycle, logical_len))
        return (StageTuple(64, 64, 64, physical_descriptor=(1, length, position, mask_kv_len)),)
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_resolve_prefill_stage_domain', resolve, raising=False)
    plan = owner._plan_action_execution(82)
    assert plan.selected.stage_tuple.compute_chunk == 64
    assert plan.request.chunk_size == (requested or None)
    assert plan.request.mode == ('EXACT' if requested else 'AUTO')
    assert plan.optimality == ('EXTERNAL_EXACT' if requested else 'CAPABILITY_SELECTED')
    assert calls == [(43, runtime.N_ACTION + 1, 82, False, 133, 0, 4, runtime.N_ACTION + 1)]

def test_vlm_fill_publishes_detached_native_plan(monkeypatch):
    owner = runtime.LingbotVlaPolicy.__new__(runtime.LingbotVlaPolicy)
    owner._cache = _ForwardCache()
    owner._vgc = _CaptureCache()
    owner._vlm = 41
    owner._prefill_chunk_size = 80
    owner._planned_prefill_chunk_size = 80
    owner._rpu_last_execution_plan = {}
    plan = _stage_plan(82, 80)
    calls = []
    output = object()
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_forward', lambda *args: calls.append(args) or output, raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_get_resolved_chunk_size', lambda handle: 80, raising=False)
    owner._vlm_fill(object(), 82, object(), plan)
    assert owner._cache.events == [('reset', 0), ('update', 82)]
    assert owner.last_rpu_execution_plan['prefill'] == {'stage': 'prefill', 'component': 'language_model', 'logical_len': 82, 'execution_len': 82, 'chunk_size': 80, 'padding_rows': 0, 'position': 0, 'authority': 'NATIVE_A6_STAGE_DESCRIPTOR', 'selection_scope': plan.selection_scope, 'physical_plan_digest': plan.physical_plan_digest, 'plan_digest': plan.plan_digest, 'graph_mode': 'COMPOSITE_CHILD', 'physical_descriptor': plan.selected.stage_tuple.physical_descriptor}
    assert calls[0][-2] == 0
    assert calls[0][-1] == plan.selected.stage_tuple.physical_descriptor
    detached = owner.last_rpu_execution_plan
    detached['prefill']['chunk_size'] = 16
    assert owner.last_rpu_execution_plan['prefill']['chunk_size'] == 80

def test_vlm_fill_rejects_native_exact_drift(monkeypatch):
    owner = runtime.LingbotVlaPolicy.__new__(runtime.LingbotVlaPolicy)
    owner._cache = _ForwardCache()
    owner._vgc = _CaptureCache()
    owner._vlm = 42
    owner._prefill_chunk_size = 80
    owner._planned_prefill_chunk_size = 80
    owner._rpu_last_execution_plan = {}
    plan = _stage_plan(82, 80)
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_forward', lambda *args: object(), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_get_resolved_chunk_size', lambda handle: 64, raising=False)
    with pytest.raises(RuntimeError, match='dry=80, forward=64'):
        owner._vlm_fill(object(), 82, object(), plan)
    assert owner._rpu_last_execution_plan == {}
    assert owner._cache.events == [('reset', 0)]

def test_expert_forward_publishes_fixed_action_plan(monkeypatch):
    owner = runtime.LingbotVlaPolicy.__new__(runtime.LingbotVlaPolicy)
    owner._cache = _ForwardCache()
    owner._egc = _CaptureCache()
    owner._expert_replay = False
    owner._exp = 43
    owner._rpu_last_execution_plan = {}
    plan = _stage_plan(runtime.N_ACTION + 1, 64, position=82)
    calls = []
    output = object()
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_forward', lambda *args: calls.append(args) or output, raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_get_resolved_chunk_size', lambda handle: 64, raising=False)
    assert owner._exp_run(object(), object(), 82, object(), plan) is output
    assert owner.last_rpu_execution_plan['action'] == {'stage': 'action', 'component': 'action_expert', 'logical_len': runtime.N_ACTION + 1, 'execution_len': runtime.N_ACTION + 1, 'chunk_size': 64, 'padding_rows': 0, 'position': 82, 'authority': 'NATIVE_A6_STAGE_DESCRIPTOR', 'selection_scope': plan.selection_scope, 'physical_plan_digest': plan.physical_plan_digest, 'plan_digest': plan.plan_digest, 'graph_mode': 'COMPOSITE_CHILD', 'physical_descriptor': plan.selected.stage_tuple.physical_descriptor}
    assert calls[0][-2] == 0
    assert calls[0][-1] == plan.selected.stage_tuple.physical_descriptor

def test_lingbot_graph_signatures_include_chunk_policy():
    action_source = inspect.getsource(runtime.LingbotVlaPolicy.get_action)
    unroll_source = inspect.getsource(runtime.LingbotVlaPolicy._denoise_unroll_run)
    assert 'chunk_policy_key(self._vlm)' in action_source
    assert action_source.count('chunk_policy_key(self._exp)') == 2
    assert action_source.count('graph_key_words()') >= 3
    assert 'self._action_chunk_size' in unroll_source

def test_fused_denoise_chunk_getter_uses_direct_dispatch_registration():
    source = (_REPO_ROOT / 'src/core/rpu_dispatch_registrations.inc').read_text()
    assert 'm.def("lingbot_denoise_get_resolved_chunk_size",\n        TORCH_FN(rpu_lingbot_denoise_get_resolved_chunk_size));' in source
    assert 'm.impl("lingbot_denoise_get_resolved_chunk_size"' not in source

@pytest.mark.parametrize('execution,error', [({'prefill': {'chunk_size': 192}}, 'certified ceiling 176'), ({'action': {'chunk_size': 16}}, 'explicit action chunk_size must be 64'), ({'vision': {'chunk_size': 'auto'}}, 'stage.*vision.*not supported'), ({'prefill': {'padding_rows': 'auto'}}, 'padding_rows.*not supported'), ({'action': {'padding_budget': 16}}, 'padding_budget.*not supported')])
def test_execution_config_rejects_unconsumed_or_uncertified_values_before_claim(monkeypatch, execution, error):
    monkeypatch.setattr(causal_lm, '_claim_live_instance', lambda owner: pytest.fail('claim must not run'))
    with pytest.raises(ValueError, match=error):
        runtime.build_lingbot_vla('unused', rpu_execution=execution)

def test_cpu_only_builder_failure_releases_claim(monkeypatch):
    events = []
    monkeypatch.setattr(causal_lm, '_claim_live_instance', lambda owner: events.append(('claim', owner)))
    monkeypatch.setattr(causal_lm, '_release_live_instance', lambda owner: events.append(('release', owner)))
    monkeypatch.setattr(runtime, '_ckpt_reader', lambda ckpt: (_ for _ in ()).throw(ValueError('bad checkpoint')))
    with pytest.raises(ValueError, match='bad checkpoint'):
        runtime.build_lingbot_vla('unused')
    assert [event[0] for event in events] == ['claim', 'release']

@pytest.mark.parametrize('raw,expected', [('1', True), (' TRUE ', True), ('on', True), ('0', False), ('False', False), (' off ', False), ('', False)])
def test_runtime_boolean_parser_accepts_only_documented_values(monkeypatch, raw, expected):
    monkeypatch.setenv('RPU_LINGBOT_EXPERT_REPLAY', raw)
    assert runtime._runtime_options_from_env()['expert_replay'] is expected

def test_invalid_runtime_flag_fails_before_materialization_and_releases_claim(monkeypatch):
    _patch_builder_cpu_preflight(monkeypatch)
    events = []
    monkeypatch.setenv('RPU_LINGBOT_DENOISE_UNROLL', 'maybe')
    monkeypatch.setattr(causal_lm, '_claim_live_instance', lambda owner: events.append(('claim', owner)))
    monkeypatch.setattr(causal_lm, '_release_live_instance', lambda owner: events.append(('release', owner)))
    monkeypatch.setattr(runtime, '_build_vlm_decoder', lambda *args: pytest.fail('device materialization must not run'))
    with pytest.raises(ValueError, match='RPU_LINGBOT_DENOISE_UNROLL'):
        runtime.build_lingbot_vla('unused')
    assert [event[0] for event in events] == ['claim', 'release']

def test_second_decoder_failure_retires_first_without_releasing_poison(monkeypatch):

    class BuildInterrupted(BaseException):
        pass
    _patch_builder_cpu_preflight(monkeypatch)
    events = []
    monkeypatch.setattr(causal_lm, '_claim_live_instance', lambda owner: events.append(('claim', owner)))
    monkeypatch.setattr(causal_lm, '_release_live_instance', lambda owner: events.append(('release', owner)))
    monkeypatch.setattr(runtime, '_build_vlm_decoder', lambda *args: (21, object()))

    def fail_expert(*args):
        raise BuildInterrupted('synthetic expert interruption')
    monkeypatch.setattr(runtime, '_build_expert', fail_expert)
    monkeypatch.setattr(runtime, '_destroy_causal_decoder_handle', lambda handle: events.append(('destroy', handle)) if handle is not None else None)
    with pytest.raises(BuildInterrupted, match='expert'):
        runtime.build_lingbot_vla('unused')
    assert [event[0] for event in events] == ['claim', 'destroy']
    assert events[-1] == ('destroy', 21)

def test_constructor_failure_clears_graphs_then_retires_all_handles(monkeypatch):

    class BuildInterrupted(BaseException):
        pass
    _patch_builder_cpu_preflight(monkeypatch)
    events = []
    monkeypatch.setattr(causal_lm, '_claim_live_instance', lambda owner: events.append(('claim', owner)))
    monkeypatch.setattr(causal_lm, '_release_live_instance', lambda owner: events.append(('release', owner)))
    monkeypatch.setattr(runtime, '_build_vlm_decoder', lambda *args: (31, object()))
    monkeypatch.setattr(runtime, '_build_expert', lambda *args: (32, object()))
    monkeypatch.setattr(runtime, '_destroy_denoise_handle', lambda handle: events.append(('destroy', 'denoise', handle)) or True)
    monkeypatch.setattr(runtime, '_destroy_causal_decoder_handle', lambda handle: events.append(('destroy', 'decoder', handle)) or True)

    def fail_init(self, **kwargs):
        self._dgc = _Cache(events, 'denoise')
        self._egc = _Cache(events, 'expert')
        self._vgc = _Cache(events, 'vlm')
        self._du_handle = 33
        raise BuildInterrupted('synthetic policy interruption')
    monkeypatch.setattr(runtime.LingbotVlaPolicy, '__init__', fail_init)
    with pytest.raises(BuildInterrupted, match='policy'):
        runtime.build_lingbot_vla('unused')
    assert events[1:] == [('clear', 'denoise'), ('clear', 'expert'), ('clear', 'vlm'), ('destroy', 'denoise', 33), ('destroy', 'decoder', 32), ('destroy', 'decoder', 31)]
    assert all((event[0] != 'release' for event in events))

def test_closed_policy_rejects_action_before_reading_observation():
    owner = runtime.LingbotVlaPolicy.__new__(runtime.LingbotVlaPolicy)
    owner._closed = True
    with pytest.raises(RuntimeError, match='closed'):
        owner.get_action({})

@pytest.mark.parametrize('value', [True, 10.0, '10', None])
def test_action_rejects_non_integer_step_count_before_reading_observation(value):
    owner = runtime.LingbotVlaPolicy.__new__(runtime.LingbotVlaPolicy)
    owner._closed = False
    with pytest.raises(TypeError, match='num_steps'):
        owner.get_action({}, num_steps=value)

def _observation_policy():
    owner = runtime.LingbotVlaPolicy.__new__(runtime.LingbotVlaPolicy)
    owner._closed = False
    owner._vis_W = {runtime.VIS_P + 'patch_embed.proj.weight': torch.zeros(runtime.V_HID, 2)}
    owner._embed_w = torch.zeros(8, runtime.VLM_HID)
    owner._head_W = {'model.state_proj.weight': torch.zeros(2, 3), 'model.action_in_proj.weight': torch.zeros(2, 4)}
    owner._max_seq = 64
    return owner

def _valid_observation():
    return {'images': torch.zeros(1, 4, 2), 'img_masks': torch.ones(1, dtype=torch.bool), 'lang_tokens': torch.tensor([[1, 2]], dtype=torch.int64), 'lang_masks': torch.ones(1, 2, dtype=torch.int64), 'state': torch.zeros(1, 3), 'noise': torch.zeros(1, runtime.N_ACTION, 4)}

def test_observation_contract_accepts_exact_token_input_profile():
    owner = _observation_policy()
    (images, img_masks, lang_ids, state, noise) = owner._validate_observation(_valid_observation())
    assert tuple(images.shape) == (1, 4, 2)
    assert tuple(img_masks.shape) == (1,)
    assert lang_ids.tolist() == [[1, 2]]
    assert tuple(state.shape) == (1, 3)
    assert tuple(noise.shape) == (1, runtime.N_ACTION, 4)

@pytest.mark.parametrize('mutate,error', [(lambda obs: obs.pop('state'), 'missing required'), (lambda obs: obs.__setitem__('images', torch.zeros(1, 9, 2)), 'square patch grid'), (lambda obs: obs['img_masks'].zero_(), 'at least one valid image'), (lambda obs: obs.__setitem__('lang_tokens', torch.tensor([[1, 8]], dtype=torch.int64)), 'must be in'), (lambda obs: obs.__setitem__('noise', torch.zeros(1, runtime.N_ACTION - 1, 4)), 'noise.*shape'), (lambda obs: obs['state'].fill_(float('nan')), 'state.*finite')])
def test_observation_contract_rejects_invalid_input_before_dispatch(mutate, error):
    owner = _observation_policy()
    obs = _valid_observation()
    mutate(obs)
    with pytest.raises((TypeError, ValueError), match=error):
        owner._validate_observation(obs)

def test_invalid_observation_does_not_rotate_graph_owners():
    events = []
    owner = _observation_policy()
    owner._vlm_replay = False
    owner._expert_replay = False
    owner._vgc = _Cache(events, 'vlm')
    owner._egc = _Cache(events, 'expert')
    obs = _valid_observation()
    obs['noise'] = torch.zeros(1, runtime.N_ACTION - 1, 4)
    with pytest.raises(ValueError, match='noise.*shape'):
        owner.get_action(obs)
    assert events == []

@pytest.mark.parametrize('value', [True, 64.0, '64', None])
def test_builder_rejects_non_integer_max_seq_before_claim(monkeypatch, value):
    monkeypatch.setattr(causal_lm, '_claim_live_instance', lambda owner: pytest.fail('claim must not run'))
    with pytest.raises(TypeError, match='max_seq'):
        runtime.build_lingbot_vla('unused', max_seq=value)

def test_builder_rejects_capacity_without_minimum_multimodal_prefix(monkeypatch):
    monkeypatch.setattr(causal_lm, '_claim_live_instance', lambda owner: pytest.fail('claim must not run'))
    with pytest.raises(ValueError, match='non-empty image/text prefix'):
        runtime.build_lingbot_vla('unused', max_seq=runtime.N_ACTION + 2)
