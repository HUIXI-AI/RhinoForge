from __future__ import annotations
import inspect
from types import SimpleNamespace
import pytest
import torch
from rpu_backend.adapters.gr00t import runtime
from rpu_backend.adapters.qwen3_vl import text as qwen3_vl_text
from rpu_backend.runtime.execution_planner import PreparedExecutionPlans, StageTuple

@pytest.fixture(autouse=True)
def _native_planner_identities(monkeypatch):
    for prefix in ('causal_decoder', 'gr00t_vl_encoder', 'gr00t_dit'):
        monkeypatch.setattr(torch.ops.rpu, prefix + '_planner_cache_identity', lambda handle: (handle, 1), raising=False)

@pytest.mark.parametrize(('execution', 'match'), [({'vision': {'chunk_size': 160}}, 'certified native Vision'), ({'action': {'chunk_size': 64}}, 'exactly 48'), ({'prefill': {'chunk_size': 16}}, 'not certified'), ({'vision': {'padding_rows': 'auto'}}, 'not supported'), ({'action': {'padding_budget': 16}}, 'not supported')])
def test_invalid_fixed_stage_config_fails_before_checkpoint_access(execution, match):
    with pytest.raises(ValueError, match=match):
        runtime.build_gr00t_vla('/definitely/not/a/checkpoint', '/definitely/not/a/qwen-checkpoint', rpu_execution=execution)

def test_three_stage_config_accepts_only_real_fixed_geometries():
    normalized = runtime._normalize_gr00t_execution({'prefill': {'chunk_size': 'auto', 'padding_rows': 'auto', 'padding_budget': 16}, 'vision': {'chunk_size': 256}, 'action': {'chunk_size': 48}})
    assert normalized == {'prefill': {'chunk_size': 'auto', 'padding_rows': 'auto', 'padding_budget': 16}, 'vision': {'chunk_size': 256}, 'action': {'chunk_size': 48}}

def test_prefill_truncated_geometry_is_auto_only():
    auto = qwen3_vl_text._chunk_envelope_for_execution('qwen3_vl_text', 16, 2048, {'prefill': {'chunk_size': 'auto'}})
    assert tuple(auto) == (0, 512)

def test_prefill_uncertified_exact_envelope_fails_closed():
    with pytest.raises(ValueError, match='auto.*only'):
        qwen3_vl_text._chunk_envelope_for_execution('qwen3_vl_text', 16, 2048, {'prefill': {'chunk_size': 16}})

def _planner_owner(execution):
    owner = runtime.Gr00tN1d7VLA.__new__(runtime.Gr00tN1d7VLA)
    owner._vision_fused_merger = False
    (root, text, vision, vl, action) = runtime._resolve_gr00t_execution(execution, entry_point='test_gr00t')
    owner._rpu_execution = root
    owner._text_execution = text
    owner._vision_execution = vision
    owner._vl_execution = vl
    owner._action_execution = action
    owner._component_generations = {component: 0 for component in runtime._EXECUTION_COMPONENTS}
    owner._bb_cache = SimpleNamespace(max_seq_len=512)
    owner._text = SimpleNamespace(_rpu_decoder_handle=7, _rpu_text_graph_cache=_ExecutionCache('text', []))
    owner._a_gc = _ExecutionCache('vl', [])
    owner._d_gc = _ExecutionCache('action', [])
    return owner

def test_component_defaults_and_overrides_resolve_independently():
    (root, text, vision, vl, action) = runtime._resolve_gr00t_execution({'prefill': {'padding_rows': 0}, 'vision': {'chunk_size': 256}, 'components': {'vision_encoder': {'vision': {'chunk_size': 512}}, 'vl_encoder': {'prefill': {'chunk_size': 96}}, 'action_dit': {'action': {'chunk_size': 48}}}})
    assert root['vision']['chunk_size'] == 256
    assert text == {'prefill': {'chunk_size': 'auto', 'padding_rows': 0}}
    assert vision == {'vision': {'chunk_size': 512}}
    assert vl == {'prefill': {'chunk_size': 96}}
    assert action == {'action': {'chunk_size': 48}}

@pytest.mark.parametrize('execution', [{'components': {'not_a_child': {'prefill': {'chunk_size': 32}}}}, {'components': {'vl_encoder': {'vision': {'chunk_size': 32}}}}, {'components': {'action_dit': {'action': {'padding_rows': 0}}}}])
def test_unknown_or_unsupported_component_config_fails_before_checkpoint(execution):
    with pytest.raises(ValueError):
        runtime.build_gr00t_vla('/definitely/not/a/checkpoint', '/definitely/not/a/qwen-checkpoint', rpu_execution=execution)

def test_qwen_z1_accepts_only_physically_equivalent_backbone_controls():
    (_root, text, vision, _vl, _action) = runtime._resolve_gr00t_execution({'vision': {'chunk_size': 256}}, entry_point='test_gr00t_z1', qwen_z1=True)
    assert text == {'prefill': {'chunk_size': 'auto'}}
    assert vision == {'vision': {'chunk_size': 'auto'}}
    with pytest.raises(ValueError, match='frozen Qwen Z1 composite'):
        runtime._resolve_gr00t_execution({'prefill': {'padding_rows': 'auto'}}, entry_point='test_gr00t_z1', qwen_z1=True)

def test_prefill_default_preserves_off_grid_logical_length(monkeypatch):
    calls = []

    def resolve(handle, execution_len, position, causal, mask_kv, rope_mode, graph_lifecycle, logical_len):
        assert (causal, mask_kv, rope_mode, graph_lifecycle, logical_len) == (True, 0, 1, 4, 82)
        calls.append((handle, execution_len, position))
        return ((64, 64, 64),)
    monkeypatch.setattr(runtime.torch.ops.rpu, 'causal_decoder_resolve_prefill_stage_domain', resolve, raising=False)
    owner = _planner_owner({'prefill': {'chunk_size': 'auto'}})
    assert owner._prefill_execution_plan(82)[:2] == (82, 64)
    assert calls == [(7, 82, 0)]

def test_prefill_auto_padding_uses_shared_joint_planner(monkeypatch):

    def resolve(_handle, execution_len, _position, _causal, _mask_kv, _rope_mode, graph_lifecycle, logical_len):
        assert (graph_lifecycle, logical_len) == (4, 82)
        chunk = 96 if execution_len >= 96 else 64
        return ((chunk, chunk, chunk),)
    monkeypatch.setattr(runtime.torch.ops.rpu, 'causal_decoder_resolve_prefill_stage_domain', resolve, raising=False)
    owner = _planner_owner({'prefill': {'padding_rows': 'auto', 'padding_budget': 20}})
    assert owner._prefill_execution_plan(82)[:2] == (96, 96)

def test_prefill_exact_padding_is_not_rounded_to_sixteen(monkeypatch):

    def resolve(_handle, execution_len, _position, _causal, _mask_kv, _rope_mode, graph_lifecycle, logical_len):
        assert (execution_len, graph_lifecycle, logical_len) == (85, 4, 82)
        return ((32, 32, 32),)
    monkeypatch.setattr(runtime.torch.ops.rpu, 'causal_decoder_resolve_prefill_stage_domain', resolve, raising=False)
    owner = _planner_owner({'prefill': {'padding_rows': 3}})
    assert owner._prefill_execution_plan(82)[:2] == (85, 32)

def test_prefill_padding_updates_every_token_indexed_input_in_lockstep():
    inputs = torch.arange(12, dtype=torch.float16).reshape(1, 3, 4)
    attention = torch.ones(1, 3, dtype=torch.long)
    positions = torch.tensor([[[0, 1, 2]], [[10, 11, 12]], [[20, 21, 22]]], dtype=torch.long)
    deepstack = [torch.ones(3, 4), torch.full((3, 4), 2.0)]
    cos = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sin = -cos
    padded = runtime._pad_gr00t_prefill_inputs(inputs, attention, positions, deepstack, cos, sin, 5)
    (p_inputs, p_attention, p_positions, p_deepstack, p_cos, p_sin) = padded
    assert p_inputs.shape == (1, 5, 4)
    assert int(torch.count_nonzero(p_inputs[:, 3:])) == 0
    assert p_attention.tolist() == [[1, 1, 1, 0, 0]]
    assert torch.equal(p_positions[..., 3:], positions[..., -1:].expand(3, 1, 2))
    assert all((int(torch.count_nonzero(dense[3:])) == 0 for dense in p_deepstack))
    assert torch.equal(p_cos[3:], cos[-1:].expand(2, 2))
    assert torch.equal(p_sin[3:], sin[-1:].expand(2, 2))

def test_prefill_restore_slices_output_and_rewinds_cache():
    events = []
    cache = SimpleNamespace(reset_to_position=lambda position: events.append(position))
    raw = torch.arange(20).reshape(1, 5, 4)
    restored = runtime._restore_gr00t_logical_prefill(raw, cache, 3, 5)
    assert torch.equal(restored, raw[:, :3])
    assert events == [3]

def test_vision_exact_geometry_is_checked_before_backbone_dispatch():
    owner = _planner_owner({'vision': {'chunk_size': 256}})
    owner._validate_forward_execution({'image_grid_thw': torch.tensor([[1, 16, 16], [1, 8, 32]])})
    with pytest.raises(ValueError, match='got \\[128\\]'):
        owner._validate_forward_execution({'image_grid_thw': torch.tensor([[1, 8, 16]])})

def test_vision_exact_accepts_only_certified_dispatch_chunks():
    for chunk in (144, 256, 512, 768):
        normalized = runtime._normalize_gr00t_execution({'vision': {'chunk_size': chunk}})
        assert normalized['vision']['chunk_size'] == chunk

def test_vision_legacy_split_with_fused_merger_fails_before_checkpoint(monkeypatch):
    monkeypatch.setenv('RPU_QWEN3VL_VISION_FUSED_MERGER', '1')
    with pytest.raises(ValueError, match='before loading model weights'):
        runtime.build_gr00t_vla('/definitely/not/a/checkpoint', '/definitely/not/a/qwen-checkpoint', rpu_execution={'vision': {'chunk_size': 144}})

def test_vision_auto_keeps_dynamic_image_geometry():
    owner = _planner_owner({'vision': {'chunk_size': 'auto'}})
    owner._validate_forward_execution({'image_grid_thw': torch.tensor([[1, 8, 16]])})

@pytest.mark.parametrize(('chunk_size', 'grids'), [(512, [[1, 16, 16], [1, 16, 16]]), (768, [[1, 16, 16], [1, 16, 16], [1, 16, 16]])])
def test_vision_exact_accepts_realizable_identical_grid_groups(chunk_size, grids):
    owner = _planner_owner({'vision': {'chunk_size': chunk_size}})
    owner._validate_forward_execution({'image_grid_thw': torch.tensor(grids)})

@pytest.mark.parametrize(('chunk_size', 'grids', 'runs'), [(512, [[1, 16, 16]] * 3, '\\[3\\]'), (768, [[1, 16, 16]] * 2, '\\[2\\]'), (512, [[1, 16, 16], [1, 8, 32]], '\\[1, 1\\]')])
def test_vision_exact_rejects_unrealizable_groups_before_dispatch(chunk_size, grids, runs):
    owner = _planner_owner({'vision': {'chunk_size': chunk_size}})
    with pytest.raises(ValueError, match=runs):
        owner._validate_forward_execution({'image_grid_thw': torch.tensor(grids)})

def test_prefill_plan_is_resolved_after_vision_dispatch():
    source = inspect.getsource(runtime.Gr00tN1d7VLA._backbone)
    assert source.index('_prefill_execution_plan') > source.index('vision_model(')

def test_vl_and_action_use_native_descriptor_domains(monkeypatch):
    owner = _planner_owner({})
    owner._a_handle = 17
    owner._d_handle = 18
    monkeypatch.setattr(runtime.torch.ops.rpu, 'gr00t_vl_encoder_resolve_stage_domain', lambda handle, length: (StageTuple(96, 96, 96, physical_descriptor=(1, handle, length)),), raising=False)
    monkeypatch.setattr(runtime.torch.ops.rpu, 'gr00t_denoise_resolve_stage_domain', lambda handle, rows, cross: (StageTuple(48, 48, 48, physical_descriptor=(2, handle, rows, cross)),), raising=False)
    (vl_chunk, vl_plan) = owner._vl_execution_plan(82)
    (action_chunk, action_plan) = owner._action_execution_plan(82)
    assert vl_chunk == 96
    assert vl_plan.selected.stage_tuple.physical_descriptor == (1, 17, 82)
    assert action_chunk == 48
    assert action_plan.selected.stage_tuple.physical_descriptor == (2, 18, 41, 82)
    assert vl_plan.physical_plan_digest != action_plan.physical_plan_digest

def test_downstream_exact_plans_are_preflighted_before_backbone_dispatch():
    source = inspect.getsource(runtime.Gr00tN1d7VLA.get_action)
    assert source.index('_vl_execution_plan(_S_in)') < source.index('self._backbone(inputs)')
    assert source.index('_action_execution_plan(_S_in)') < source.index('self._backbone(inputs)')

class _ExecutionCache:

    def __init__(self, name, events, *, clear_failures=0):
        self.name = name
        self.events = events
        self.clear_failures = clear_failures
        self._plans = PreparedExecutionPlans()

    def is_frozen(self):
        return False

    def prepare_plan(self, key, factory):
        return self._plans.prepare(key, factory)

    def begin_warmup(self):
        self.events.append(('warmup', self.name))
        return self

    def clear(self):
        self.events.append(('clear', self.name))
        if self.clear_failures:
            self.clear_failures -= 1
            raise RuntimeError(f'synthetic {self.name} clear failure')

    def cache_invariant_ok(self):
        return True

def _live_execution_owner(*, vision_clear_failures=0):
    graph_events = []
    caches = {name: _ExecutionCache(name, graph_events, clear_failures=vision_clear_failures if name == 'vision' else 0) for name in ('text', 'decoder', 'vision', 'vl', 'action')}
    owner = runtime.Gr00tN1d7VLA.__new__(runtime.Gr00tN1d7VLA)
    owner._closed = False
    owner._vision_fused_merger = False
    owner._qwen_z1_plan_hash = None
    owner._qwen_z1_gc = None
    owner._a_gc = caches['vl']
    owner._d_gc = caches['action']
    owner._a_handle = 17
    owner._d_handle = 18
    owner._text = SimpleNamespace(_rpu_decoder_handle=19, _rpu_text_graph_cache=caches['text'], _rpu_decoder_graph_cache=caches['decoder'])
    owner._vision = SimpleNamespace(_rpu_vision_handle=20, _rpu_vision_graph_cache=caches['vision'])
    owner._component_generations = {component: 0 for component in runtime._EXECUTION_COMPONENTS}
    owner._execution_generation = 0
    owner._execution_reconfigure_journal = None
    owner._rpu_last_execution_plan = {component: {'old': True} for component in runtime._EXECUTION_COMPONENTS}
    resolved = runtime._resolve_gr00t_execution({}, entry_point='test_gr00t_live')
    owner._publish_execution_views(resolved, generation=0, component_generations=owner._component_generations)
    owner._execution_session = runtime.bind_execution_session(owner, owner._rpu_execution, entry_point='test_gr00t_live', supported_components=runtime._EXECUTION_COMPONENTS, validate=owner.validate_execution_reconfigure, apply=owner.apply_execution_reconfigure, rollback=owner.rollback_execution_reconfigure, graph_mode=runtime.GRAPH_COMPOSITE_CHILD)
    return (owner, caches, graph_events)

def _install_reconfigure_ops(monkeypatch, *, fail=None):
    state = {'events': [], 'fail': fail}

    def begin(attempt):
        state['events'].append(('begin', attempt))
        if state['fail'] == 'begin':
            raise RuntimeError('synthetic begin failure')
        return 701

    def stage(handle, token, chunk):
        state['events'].append(('stage', handle, token, chunk))

    def commit(token):
        state['events'].append(('commit', token))
        if state['fail'] == 'commit':
            raise RuntimeError('synthetic commit failure')
    monkeypatch.setattr(runtime.torch.ops.rpu, 'execution_reconfigure_begin', begin, raising=False)
    monkeypatch.setattr(runtime.torch.ops.rpu, 'execution_reconfigure_commit', commit, raising=False)
    monkeypatch.setattr(runtime.torch.ops.rpu, 'execution_reconfigure_abort', lambda token: state['events'].append(('abort', token)), raising=False)
    monkeypatch.setattr(runtime.torch.ops.rpu, 'execution_reconfigure_abort_attempt', lambda attempt: state['events'].append(('abort_attempt', attempt)), raising=False)
    monkeypatch.setattr(runtime.torch.ops.rpu, 'qwen3vl_vision_stage_chunk_size', stage, raising=False)
    return state

@pytest.mark.parametrize('force_rebuild', [False, True])
def test_hot_reconfigure_updates_only_changed_child_and_graph_owner(monkeypatch, force_rebuild):
    (owner, _caches, graph_events) = _live_execution_owner()
    native = _install_reconfigure_ops(monkeypatch)
    patch = {'components': {'vision_encoder': {'vision': {'chunk_size': 256}}}}
    if force_rebuild:
        owner._execution_session._reconfigure({}, force_rebuild=True)
    else:
        owner._execution_session.reconfigure(patch)
    assert [event[0] for event in native['events']] == ['begin', 'stage', 'commit']
    assert native['events'][1][1:] == (20, 701, 0 if force_rebuild else 256)
    if force_rebuild:
        assert {event[1] for event in graph_events if event[0] == 'clear'} == {'text', 'decoder', 'vision', 'vl', 'action'}
    else:
        assert graph_events == [('warmup', 'vision'), ('clear', 'vision')]
    assert owner._execution_session.stats()['generation'] == 1
    assert owner._component_generations == {'text_backbone': int(force_rebuild), 'vision_encoder': 1, 'vl_encoder': int(force_rebuild), 'action_dit': int(force_rebuild)}
    assert set(owner._rpu_last_execution_plan) == (set() if force_rebuild else {'text_backbone', 'vl_encoder', 'action_dit'})
    assert owner._vision._rpu_execution_graph_key_words == (1,)

def test_descriptor_only_child_reconfigure_uses_no_empty_native_transaction(monkeypatch):
    (owner, _caches, graph_events) = _live_execution_owner()
    native = _install_reconfigure_ops(monkeypatch)
    owner._execution_session.reconfigure({'components': {'action_dit': {'action': {'chunk_size': 48}}}})
    assert native['events'] == []
    assert graph_events == [('warmup', 'action'), ('clear', 'action')]
    assert owner._component_generations == {'text_backbone': 0, 'vision_encoder': 0, 'vl_encoder': 0, 'action_dit': 1}

def test_begin_failure_is_single_attempt_and_leaves_session_unpoisoned(monkeypatch):
    (owner, _caches, graph_events) = _live_execution_owner()
    native = _install_reconfigure_ops(monkeypatch, fail='begin')
    before_config = owner._execution_session.config
    before_generations = dict(owner._component_generations)
    with pytest.raises(RuntimeError, match='synthetic begin failure'):
        owner._execution_session.reconfigure({'components': {'vision_encoder': {'vision': {'chunk_size': 256}}}})
    assert [event[0] for event in native['events']] == ['begin', 'abort_attempt']
    assert graph_events == []
    assert owner._execution_session.config is before_config
    assert owner._component_generations == before_generations
    assert owner._execution_generation == 0
    assert owner._execution_reconfigure_journal is None
    stats = owner._execution_session.stats()
    assert stats['state'] == 'QUIESCENT'
    assert stats['failure_count'] == 1

def test_post_commit_failure_compensates_native_and_runtime_state(monkeypatch):
    (owner, _caches, graph_events) = _live_execution_owner(vision_clear_failures=1)
    native = _install_reconfigure_ops(monkeypatch)
    with pytest.raises(RuntimeError, match='synthetic vision clear failure'):
        owner._execution_session.reconfigure({'components': {'vision_encoder': {'vision': {'chunk_size': 256}}}})
    assert [event[0] for event in native['events']] == ['begin', 'stage', 'commit', 'begin', 'stage', 'commit']
    assert native['events'][1][-1] == 256
    assert native['events'][4][-1] == 0
    assert graph_events == [('warmup', 'vision'), ('clear', 'vision'), ('warmup', 'vision'), ('clear', 'vision')]
    assert owner._component_generations == {component: 0 for component in runtime._EXECUTION_COMPONENTS}
    assert owner._execution_generation == 0
    assert owner._execution_session.stats()['state'] == 'QUIESCENT'

def test_failed_compensation_poisons_the_shared_session(monkeypatch):
    (owner, _caches, _graph_events) = _live_execution_owner(vision_clear_failures=2)
    native = _install_reconfigure_ops(monkeypatch)
    with pytest.raises(RuntimeError, match='synthetic vision clear failure'):
        owner._execution_session.reconfigure({'components': {'vision_encoder': {'vision': {'chunk_size': 256}}}})
    assert [event[0] for event in native['events']] == ['begin', 'stage', 'commit', 'begin', 'stage', 'commit']
    assert owner._execution_session.stats()['state'] == 'POISONED'

def test_unknown_hot_component_fails_before_native_transaction(monkeypatch):
    (owner, _caches, _events) = _live_execution_owner()
    native = _install_reconfigure_ops(monkeypatch)
    with pytest.raises(ValueError, match='unknown component'):
        owner._execution_session.reconfigure({'components': {'not_a_child': {'vision': {'chunk_size': 256}}}})
    assert native['events'] == []
