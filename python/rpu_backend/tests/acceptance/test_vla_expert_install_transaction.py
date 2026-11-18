from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import pytest
import rpu_backend
from rpu_backend.adapters.qwenpi05 import convert as qwen_convert
from rpu_backend.adapters.qwenpi05 import fused as qwen_fused
from rpu_backend.adapters.rhinovla import convert as rhino_convert
from rpu_backend.adapters.rhinovla import fused as rhino_fused

class _FakeTensor:

    def __init__(self, *shape):
        self.shape = shape

class _Rotary:

    def cpu(self):
        return self

class _PublishInterrupted(BaseException):
    pass

class _Expert:

    def arm_assignment_failure(self, name):
        object.__setattr__(self, '_fail_assignment', name)

    def __setattr__(self, name, value):
        if self.__dict__.get('_fail_assignment') == name:
            object.__setattr__(self, '_fail_assignment', None)
            raise _PublishInterrupted(f'synthetic publication interruption for {name}')
        object.__setattr__(self, name, value)

def _expert():
    expert = _Expert()
    expert.layers = [SimpleNamespace()]
    expert.config = SimpleNamespace(num_attention_heads=8, num_key_value_heads=4, head_dim=16, width=128, mlp_dim=256, rms_norm_eps=1e-06)
    expert.norm = None
    expert.forward = lambda *args, **kwargs: 'original'
    return expert

@pytest.fixture
def native_install(monkeypatch):
    from rpu_backend.api import _execution, causal_lm
    from rpu_backend.runtime import _native_retirement
    monkeypatch.setattr(_execution, '_UNSAFE_PROCESS_REASON', None)
    monkeypatch.setattr(causal_lm, '_LIVE_REF', None)
    monkeypatch.setattr(_native_retirement, '_FAILED_RETIREMENTS', [])
    for name in _RHINO_NATIVE_COLD_ENV:
        monkeypatch.delenv(name, raising=False)
    state = {kind: {'next_handle': 1, 'alive': set(), 'events': [], 'lifecycle': [], 'precompute_calls': [], 'runtime_config_calls': [], 'fail_config': set(), 'fail_set': set(), 'fail_envelope': set(), 'envelopes': {}, 'fail_destroy': set()} for kind in ('rhino', 'qwen')}

    def install_native(kind, module, prefix):
        component = state[kind]

        def create():
            handle = component['next_handle']
            component['next_handle'] += 1
            component['alive'].add(handle)
            component['events'].append(('create', handle))
            component['lifecycle'].append(('create', handle))
            return handle

        def set_weights(handle, *args):
            component['events'].append(('set_weights', handle))
            component['lifecycle'].append(('set_weights', handle))
            if handle in component['fail_set']:
                raise RuntimeError(f'synthetic {kind} set_weights failure for {handle}')

        def destroy(handle):
            component['events'].append(('destroy', handle))
            component['lifecycle'].append(('destroy', handle))
            if handle in component['fail_destroy']:
                raise RuntimeError(f'synthetic {kind} destroy failure for {handle}')
            if handle not in component['alive']:
                raise RuntimeError(f'double destroy for {kind} handle {handle}')
            component['alive'].remove(handle)

        def set_envelope(handle, capacity, chunk):
            assert component['events'][-1] == ('set_weights', handle)
            if handle in component['fail_envelope']:
                raise RuntimeError('synthetic envelope failure')
            component['envelopes'][handle] = (capacity, chunk)
        monkeypatch.setattr(module.torch.ops.rpu, f'{prefix}_create', create, raising=False)
        monkeypatch.setattr(module.torch.ops.rpu, f'{prefix}_set_weights', set_weights, raising=False)
        monkeypatch.setattr(module.torch.ops.rpu, f'{prefix}_destroy', destroy, raising=False)
        monkeypatch.setattr(module.torch.ops.rpu, f'{prefix}_set_chunk_envelope', set_envelope, raising=False)
        if kind == 'rhino':

            def set_runtime_config(handle, *flags):
                component['runtime_config_calls'].append((handle, flags))
                component['lifecycle'].append(('set_runtime_config', handle))
                if handle in component['fail_config']:
                    raise RuntimeError(f'synthetic rhino runtime config failure for {handle}')
            monkeypatch.setattr(module.torch.ops.rpu, 'rhino_vla_set_runtime_config', set_runtime_config, raising=False)
    install_native('rhino', rhino_fused, 'rhino_vla')
    install_native('qwen', qwen_fused, 'qwenpi05')
    monkeypatch.setattr(rhino_fused.torch.ops.rpu, 'rhino_vla_set_rope_position', lambda *args: None, raising=False)

    class GraphCache:

        def begin_warmup(self):
            pass

        def clear(self):
            pass

        def cache_invariant_ok(self):
            return True

    class GraphSignature:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class RotaryWrapper:

        def __init__(self, rotary):
            self.rotary = rotary
    monkeypatch.setattr(rpu_backend.graph, 'GraphCache', GraphCache)
    monkeypatch.setattr(rpu_backend.graph, 'GraphSignature', GraphSignature)
    monkeypatch.setattr(rhino_convert, '_gather_expert_weights', lambda expert: tuple(['weights'] * 13))
    monkeypatch.setattr(rhino_convert, '_gather_expert_scales', lambda expert: ())

    def precompute_rhino(*args):
        state['rhino']['precompute_calls'].append(args[1:])
        return (_FakeTensor(31, 8), _FakeTensor(31, 8))
    monkeypatch.setattr(rhino_convert, '_precompute_mrope_cos_sin', precompute_rhino)
    monkeypatch.setattr(rhino_convert, '_create_kv_cache', lambda config, capacity, attn_tp=None: SimpleNamespace(max_seq_len=capacity))
    monkeypatch.setattr(qwen_fused, '_gather_qwenpi05_weights', lambda expert: tuple(['weights'] * 13))
    monkeypatch.setattr(qwen_fused, '_precompute_mrope_cos_sin', lambda *args: (_FakeTensor(31, 8), _FakeTensor(31, 8)))
    monkeypatch.setattr(qwen_fused, '_create_kv_cache', lambda config, capacity: SimpleNamespace(max_seq_len=capacity))
    monkeypatch.setattr(qwen_convert, '_RPURotaryEmbWrapper', RotaryWrapper)
    monkeypatch.setattr(qwen_fused, '_patch_adarms_norm_for_rpu', lambda : None)
    monkeypatch.setattr(qwen_fused, '_patch_action_io_decode_actions_for_rpu', lambda : None)
    return state
_COMPONENTS = {'rhino': {'module': rhino_fused, 'install': lambda expert: rhino_fused.patch_rhino_vla_for_rpu(expert, prefix_len=16, suffix_len=31, cpu_rotary_emb=_Rotary()), 'handle': '_rpu_handle', 'finalizer': '_rpu_handle_finalizer', 'cache': '_rpu_graph_cache'}, 'qwen': {'module': qwen_fused, 'install': lambda expert: qwen_fused.patch_qwenpi05_for_rpu_fused(expert, prefix_len=16, suffix_len=31, cpu_rotary_emb=_Rotary()), 'handle': '_rpu_qwenpi05_handle', 'finalizer': '_rpu_qwenpi05_handle_finalizer', 'cache': '_rpu_qwenpi05_graph_cache'}}
_RHINO_NATIVE_COLD_ENV = ('RPU_RHINOVLA_DENOISE_STATIC_CONTEXT_CACHE', 'RPU_RHINOVLA_FUSED_ADARMS_GEMV', 'RPU_RHINOVLA_SKIP_ADARMS_GEMV', 'RPU_RHINOVLA_PRECOMPUTE_ADARMS', 'RPU_RHINOVLA_PRECOMPUTE_TIME_PROJ', 'RPU_RHINOVLA_FOLD_ACTION_TIME_IN', 'RPU_RHINOVLA_GATED_NO_SUB', 'RPU_RHINOVLA_FUSED_SILU_MUL', 'RPU_RHINOVLA_EXPERT_FUSIONS', 'RPU_RHINOVLA_ADARMS_RESIDENT', 'RPU_RHINOVLA_PACKED_QKV', 'RPU_RHINOVLA_ALIGNED_KV')

@pytest.mark.parametrize('kind', ['rhino', 'qwen'])
@pytest.mark.parametrize('capacity', [256, 4096])
def test_envelope_uses_full_cache_not_suffix_rope(native_install, kind, capacity):
    expert = _expert()
    install = rhino_fused.patch_rhino_vla_for_rpu if kind == 'rhino' else qwen_fused.patch_qwenpi05_for_rpu_fused
    try:
        handle = install(expert, prefix_len=16, suffix_len=31, max_seq_len=capacity, cpu_rotary_emb=_Rotary())
        assert native_install[kind]['envelopes'] == {handle: (capacity, 0)}
        assert expert._rpu_kv_cache.max_seq_len == capacity
    finally:
        _finalize(expert, _COMPONENTS[kind])

@pytest.mark.parametrize('kind', ['rhino', 'qwen'])
@pytest.mark.parametrize('replacement', [False, True])
def test_envelope_failure_preserves_old_install_and_retires_pending(native_install, kind, replacement):
    expert = _expert()
    spec = _COMPONENTS[kind]
    if replacement:
        spec['install'](expert)
    before = dict(vars(expert))
    pending = 2 if replacement else 1
    native_install[kind]['fail_envelope'].add(pending)
    try:
        with pytest.raises(RuntimeError, match='envelope failure'):
            spec['install'](expert)
        assert vars(expert) == before
        assert native_install[kind]['alive'] == ({1} if replacement else set())
        assert native_install[kind]['events'].count(('destroy', pending)) == 1
    finally:
        _finalize(expert, spec)

def test_rhino_install_publishes_full_rope_table_contract(native_install):
    expert = _expert()
    assert _COMPONENTS['rhino']['install'](expert) == 1
    assert native_install['rhino']['precompute_calls'] == [(0, 2048)]
    assert native_install['rhino']['runtime_config_calls'] == [(1, (False,) * len(_RHINO_NATIVE_COLD_ENV))]
    assert native_install['rhino']['lifecycle'][:3] == [('create', 1), ('set_runtime_config', 1), ('set_weights', 1)]
    assert expert._rpu_rope_max_seq_len == 2048
    assert expert._rpu_suffix_len == 31
    assert expert._rpu_native_cold_config.fused_adarms is False
    _finalize(expert, _COMPONENTS['rhino'])

def test_rhino_runtime_config_is_frozen_and_bound_before_weights(native_install, monkeypatch):
    values = (True, False, False, True, False, False, True, False, True, True, True, True)
    for (name, value) in zip(_RHINO_NATIVE_COLD_ENV, values):
        monkeypatch.setenv(name, 'on' if value else 'off')
    expert = _expert()
    assert _COMPONENTS['rhino']['install'](expert) == 1
    assert native_install['rhino']['runtime_config_calls'] == [(1, values)]
    monkeypatch.setenv('RPU_RHINOVLA_FUSED_ADARMS_GEMV', 'on')
    monkeypatch.setenv('RPU_RHINOVLA_PACKED_QKV', 'off')
    monkeypatch.setenv('RPU_RHINOVLA_ALIGNED_KV', 'off')
    assert expert._rpu_native_cold_config.fused_adarms is False
    assert expert._rpu_native_cold_config.packed_qkv is True
    assert expert._rpu_native_cold_config.aligned_kv is True
    _finalize(expert, _COMPONENTS['rhino'])

def test_rhino_invalid_runtime_config_fails_before_create(native_install, monkeypatch):
    monkeypatch.setenv('RPU_RHINOVLA_FUSED_SILU_MUL', 'typo')
    with pytest.raises(ValueError, match='RPU_RHINOVLA_FUSED_SILU_MUL'):
        _COMPONENTS['rhino']['install'](_expert())
    assert native_install['rhino']['events'] == []

def test_rhino_runtime_config_failure_cleans_pending_handle(native_install):
    native_install['rhino']['fail_config'].add(1)
    with pytest.raises(RuntimeError, match='runtime config failure'):
        _COMPONENTS['rhino']['install'](_expert())
    assert native_install['rhino']['events'] == [('create', 1), ('destroy', 1)]
    assert native_install['rhino']['lifecycle'] == [('create', 1), ('set_runtime_config', 1), ('destroy', 1)]
    assert native_install['rhino']['alive'] == set()

def _finalize(expert, component):
    finalizer = getattr(expert, component['finalizer'], None)
    if finalizer is not None and finalizer.alive:
        finalizer()

@pytest.mark.parametrize('state', ['started', 'closed', 'poisoned'])
def test_qwenpi05_replacement_rejects_used_session_before_mutation(native_install, state):
    expert = _expert()
    component = _COMPONENTS['qwen']
    component['install'](expert)
    session = expert._execution_session
    if state == 'started':
        with pytest.raises(TypeError, match='suffix_embeds'):
            expert.forward()
    elif state == 'closed':
        expert.close_rpu_execution()
    else:
        session.poison()
    events = list(native_install['qwen']['events'])
    with pytest.raises(RuntimeError, match='cold session'):
        component['install'](expert)
    assert native_install['qwen']['events'] == events
    _finalize(expert, component)

def test_qwenpi05_session_fixed_reconfigure_and_idempotent_close(native_install):
    expert = _expert()
    _COMPONENTS['qwen']['install'](expert)
    session = expert._execution_session
    assert session.reconfigure({}) == {}
    with pytest.raises(ValueError, match='fixed'):
        session.reconfigure({'action': {'chunk_size': 32}})
    assert session.config == {}
    expert.close_rpu_execution()
    expert.close_rpu_execution()
    assert native_install['qwen']['events'].count(('destroy', 1)) == 1
    with pytest.raises(RuntimeError, match='closed'):
        expert.forward()

def test_qwenpi05_failed_install_retries_without_stale_session(native_install):
    expert = _expert()
    native_install['qwen']['fail_set'].add(1)
    with pytest.raises(RuntimeError, match='set_weights failure'):
        _COMPONENTS['qwen']['install'](expert)
    assert not hasattr(expert, '_execution_session')
    assert not hasattr(expert, '_planner_cost_session')
    assert _COMPONENTS['qwen']['install'](expert) == 2
    expert._execution_session.require_cold()
    expert.close_rpu_execution()

@pytest.mark.parametrize('kind', tuple(_COMPONENTS))
def test_missing_layers_fails_before_native_create(native_install, kind):
    component = _COMPONENTS[kind]
    expert = _expert()
    del expert.layers
    with pytest.raises(RuntimeError, match='no \\.layers'):
        component['install'](expert)
    assert native_install[kind]['events'] == []
    assert not hasattr(expert, component['handle'])

@pytest.mark.parametrize('kind', tuple(_COMPONENTS))
def test_prepare_failure_does_not_create_or_publish(native_install, monkeypatch, kind):
    component = _COMPONENTS[kind]
    expert = _expert()
    original_forward = expert.forward

    class FailingCache:

        def __init__(self):
            raise RuntimeError('synthetic cache preparation failure')
    monkeypatch.setattr(rpu_backend.graph, 'GraphCache', FailingCache)
    with pytest.raises(RuntimeError, match='cache preparation failure'):
        component['install'](expert)
    assert native_install[kind]['events'] == []
    assert not hasattr(expert, component['handle'])
    assert expert.forward is original_forward

@pytest.mark.parametrize('kind', tuple(_COMPONENTS))
def test_set_weights_failure_cleans_pending_handle(native_install, kind):
    component = _COMPONENTS[kind]
    expert = _expert()
    original_forward = expert.forward
    native_install[kind]['fail_set'].add(1)
    with pytest.raises(RuntimeError, match='set_weights failure'):
        component['install'](expert)
    assert native_install[kind]['events'] == [('create', 1), ('set_weights', 1), ('destroy', 1)]
    assert native_install[kind]['alive'] == set()
    assert not hasattr(expert, component['handle'])
    assert expert.forward is original_forward

@pytest.mark.parametrize('kind', tuple(_COMPONENTS))
def test_failed_replacement_preserves_published_install(native_install, kind):
    component = _COMPONENTS[kind]
    expert = _expert()
    assert component['install'](expert) == 1
    old_forward = expert.forward
    old_cache = getattr(expert, component['cache'])
    old_finalizer = getattr(expert, component['finalizer'])
    old_kv = expert._rpu_kv_cache
    native_install[kind]['events'].clear()
    native_install[kind]['fail_set'].add(2)
    with pytest.raises(RuntimeError, match='set_weights failure'):
        component['install'](expert)
    assert native_install[kind]['events'] == [('create', 2), ('set_weights', 2), ('destroy', 2)]
    assert native_install[kind]['alive'] == {1}
    assert getattr(expert, component['handle']) == 1
    assert getattr(expert, component['finalizer']) is old_finalizer
    assert old_finalizer.alive
    assert getattr(expert, component['cache']) is old_cache
    assert expert._rpu_kv_cache is old_kv
    assert expert.forward is old_forward
    _finalize(expert, component)

@pytest.mark.parametrize('kind', tuple(_COMPONENTS))
def test_old_destroy_failure_aborts_replacement(native_install, kind):
    component = _COMPONENTS[kind]
    expert = _expert()
    assert component['install'](expert) == 1
    old_forward = expert.forward
    old_finalizer = getattr(expert, component['finalizer'])
    native_install[kind]['events'].clear()
    native_install[kind]['fail_destroy'].add(1)
    with pytest.raises(RuntimeError, match='destroy failure'):
        component['install'](expert)
    assert native_install[kind]['events'] == [('create', 2), ('set_weights', 2), ('destroy', 1), *([] if kind == 'rhino' else [('destroy', 2)])]
    assert native_install[kind]['alive'] == ({1, 2} if kind == 'rhino' else {1})
    assert getattr(expert, component['handle']) == 1
    assert getattr(expert, component['finalizer']) is old_finalizer
    assert expert.forward is old_forward
    if kind == 'rhino':
        assert expert._rpu_retirement_state.failed is not None
    else:
        native_install[kind]['fail_destroy'].clear()
        _finalize(expert, component)

@pytest.mark.parametrize('kind', tuple(_COMPONENTS))
def test_first_install_publication_baseexception_rolls_back(native_install, kind):
    component = _COMPONENTS[kind]
    expert = _expert()
    original_forward = expert.forward
    expert.arm_assignment_failure(component['finalizer'])
    with pytest.raises(_PublishInterrupted, match='publication interruption'):
        component['install'](expert)
    assert native_install[kind]['events'] == [('create', 1), ('set_weights', 1), ('destroy', 1)]
    assert native_install[kind]['alive'] == set()
    assert not hasattr(expert, component['handle'])
    assert not hasattr(expert, component['cache'])
    assert not hasattr(expert, '_rpu_kv_cache')
    assert expert.forward is original_forward

@pytest.mark.parametrize('kind', tuple(_COMPONENTS))
def test_replacement_publication_baseexception_preserves_old_install(native_install, kind):
    component = _COMPONENTS[kind]
    expert = _expert()
    assert component['install'](expert) == 1
    old_forward = expert.forward
    old_receipt = {'generation': 7, 'physical_descriptor': (13, 17)}
    expert._rpu_last_execution_plan = old_receipt
    old_cache = getattr(expert, component['cache'])
    old_kv = expert._rpu_kv_cache
    old_cos = expert._rpu_cos_cached
    old_sin = expert._rpu_sin_cached
    old_finalizer = getattr(expert, component['finalizer'])
    native_install[kind]['events'].clear()
    expert.arm_assignment_failure('forward')
    with pytest.raises(_PublishInterrupted, match='publication interruption'):
        component['install'](expert)
    assert native_install[kind]['events'] == [('create', 2), ('set_weights', 2), ('destroy', 2)]
    assert native_install[kind]['alive'] == {1}
    assert getattr(expert, component['handle']) == 1
    assert getattr(expert, component['finalizer']) is old_finalizer
    assert old_finalizer.alive
    assert getattr(expert, component['cache']) is old_cache
    assert expert._rpu_kv_cache is old_kv
    assert expert._rpu_cos_cached is old_cos
    assert expert._rpu_sin_cached is old_sin
    assert expert.forward is old_forward
    assert expert._rpu_last_execution_plan is old_receipt
    _finalize(expert, component)

@pytest.mark.parametrize('kind', tuple(_COMPONENTS))
def test_successful_replacement_retires_old_finalizer(native_install, kind):
    component = _COMPONENTS[kind]
    expert = _expert()
    assert component['install'](expert) == 1
    old_finalizer = getattr(expert, component['finalizer'])
    old_cache = getattr(expert, component['cache'])
    old_session = getattr(expert, '_execution_session', None)
    if kind == 'qwen':
        callbacks = (old_session._validate, old_session._apply, old_session._rollback)
    native_install[kind]['events'].clear()
    assert component['install'](expert) == 2
    assert native_install[kind]['events'] == [('create', 2), ('set_weights', 2), ('destroy', 1)]
    assert native_install[kind]['alive'] == {2}
    assert getattr(expert, component['handle']) == 2
    assert getattr(expert, component['finalizer']).alive
    assert not old_finalizer.alive
    assert getattr(expert, component['cache']) is not old_cache
    if kind == 'qwen':
        assert expert._execution_session is old_session
        assert (old_session._validate, old_session._apply, old_session._rollback) == callbacks
        old_session._validate({})
        with pytest.raises(ValueError, match='fixed'):
            old_session._validate({'action': {'chunk_size': 32}})
    _finalize(expert, component)

@pytest.mark.parametrize('kind', tuple(_COMPONENTS))
def test_replacement_does_not_double_destroy_dead_finalizer(native_install, kind):
    component = _COMPONENTS[kind]
    expert = _expert()
    assert component['install'](expert) == 1
    getattr(expert, component['finalizer'])()
    native_install[kind]['events'].clear()
    assert component['install'](expert) == 2
    assert native_install[kind]['events'] == [('create', 2), ('set_weights', 2)]
    assert native_install[kind]['alive'] == {2}
    assert getattr(expert, component['handle']) == 2
    assert getattr(expert, component['finalizer']).alive
    _finalize(expert, component)

def test_qwenpi05_source_rejects_suffix_longer_than_rope_table():
    source = Path(qwen_fused.__file__).read_text(encoding='utf-8')
    guard = 'if actual_suffix_len > target_suffix_len:'
    dispatch = 'with self._rpu_qwenpi05_graph_cache.capture(sig):'
    assert guard in source
    assert source.index(guard) < source.index(dispatch)

@pytest.mark.parametrize('fail', (False, True))
def test_rhino_w8_install_uses_scale_api_and_preserves_old_owner(native_install, monkeypatch, fail):
    expert = _expert()
    install = _COMPONENTS['rhino']['install']
    install(expert)
    old_resource = expert._rpu_retirement_state
    scales = tuple(([object()] for _ in range(7)))
    monkeypatch.setattr(rhino_convert, '_gather_expert_scales', lambda expert: scales)
    calls = []
    original_set = rhino_fused.torch.ops.rpu.rhino_vla_set_weights

    def set_w8(handle, *args):
        calls.append(args)
        assert args[-7:] == scales
        if fail:
            raise RuntimeError('synthetic W8 scale rejection')
        original_set(handle, *args[:-7])
    monkeypatch.setattr(rhino_fused.torch.ops.rpu, 'rhino_vla_set_weights_w8a16', set_w8, raising=False)
    if fail:
        with pytest.raises(RuntimeError, match='W8 scale rejection'):
            install(expert)
        assert expert._rpu_retirement_state is old_resource
        assert expert._rpu_handle == 1
        assert native_install['rhino']['alive'] == {1}
    else:
        install(expert)
        assert expert._rpu_handle == 2
        assert native_install['rhino']['alive'] == {2}
        assert any((item is scales for item in expert._rpu_retirement_state.keepalive))
    assert len(calls) == 1
    expert._rpu_handle_finalizer()
