from __future__ import annotations

import contextlib
from pathlib import Path
from types import MethodType, SimpleNamespace
import pytest
import torch
import torch.nn as nn
from rpu_backend.adapters import llama, qwen3
from rpu_backend.adapters._template import minimal_causal_lm
from rpu_backend.api._execution import reconfigure_rpu_execution, rpu_execution_stats
from rpu_backend.api.errors import RPUBackendError
from rpu_backend.runtime import decoder, hw_attrs

class _InstallInterrupted(BaseException):
    pass

class _GraphCache:

    def __init__(self, events):
        self._events = events

    def clear(self):
        self._events.append('graph_clear')

    def begin_warmup(self):
        self._events.append('graph_warmup')
        return self

    def cache_invariant_ok(self):
        return True

class _Finalizer:

    def __init__(self, events, *, alive=True):
        self._events = events
        self.alive = alive

    def __call__(self):
        if self.alive:
            self._events.append('handle_finalize')
            self.alive = False
        return True

class _Inner(nn.Module):

    def __setattr__(self, name, value):
        if self.__dict__.get('_reject_runtime_cleanup', False):
            raise AssertionError('cleanup re-entered inner __setattr__')
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if self.__dict__.get('_reject_runtime_cleanup', False):
            raise AssertionError('cleanup re-entered inner __delattr__')
        super().__delattr__(name)

    def forward(self, *args, **kwargs):
        return 'original-forward'

class _Outer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self._events = []
        self._fail_ready_publish = False
        self.config = config
        self.model = _Inner()

    def __setattr__(self, name, value):
        events = self.__dict__.get('_events')
        if events is not None and name == '_rpu_swizzled':
            events.append('ready_publish')
            if self.__dict__.get('_fail_ready_publish', False):
                vars(self.model)['_reject_runtime_cleanup'] = True
                raise _InstallInterrupted('synthetic ready publish interruption')
        super().__setattr__(name, value)

    def to(self, target, *args, **kwargs):
        self._events.append(('move', str(target)))
        return self

def _qwen3_config():
    return SimpleNamespace(hidden_size=1024, intermediate_size=3072, num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8)

def _llama_config():
    return SimpleNamespace(hidden_size=2048, intermediate_size=8192, num_hidden_layers=16, num_attention_heads=32, num_key_value_heads=8)
_ADAPTER_CASES = (pytest.param(qwen3, qwen3.Qwen3Adapter, _qwen3_config, id='qwen3'), pytest.param(llama, llama.LlamaAdapter, _llama_config, id='llama'), pytest.param(minimal_causal_lm, minimal_causal_lm.MinimalCausalLMAdapter, _llama_config, id='template'))
_PLAIN_ADAPTER_CASES = _ADAPTER_CASES[:2]

class _NativeReconfigureOps:

    def __init__(self, events, *, interrupt_first_commit=False):
        self.events = events
        self.current_chunk = 0
        self._active = False
        self._staged_chunk = None
        self._next_token = 70
        self._interrupt_first_commit = interrupt_first_commit

    def execution_reconfigure_begin(self, _attempt):
        assert not self._active
        self._active = True
        self._staged_chunk = None
        self._next_token += 1
        self.events.append(('txn_begin', self._next_token))
        return self._next_token

    def execution_reconfigure_commit(self, token):
        assert self._active and token == self._next_token
        if self._staged_chunk is not None:
            self.current_chunk = self._staged_chunk
        self._active = False
        self.events.append(('txn_commit', token, self.current_chunk))
        if self._interrupt_first_commit:
            self._interrupt_first_commit = False
            raise KeyboardInterrupt('injected ambiguous native commit')

    def execution_reconfigure_abort(self, token):
        if not self._active:
            raise RuntimeError(f'stale token {token}')
        self._active = False
        self.events.append(('txn_abort', token))

    def execution_reconfigure_abort_attempt(self, _attempt):
        return False

    def causal_decoder_stage_chunk_size_override(self, handle, token, chunk):
        assert self._active and token == self._next_token and (handle == 41)
        self._staged_chunk = int(chunk)
        self.events.append(('stage_chunk', token, int(chunk)))

def _prepare_case(monkeypatch, module, model, *, postinstall_error=None):
    events = model._events
    if module in (qwen3, llama):
        monkeypatch.setattr(module, "require_native_backend", lambda: None)
    if module is minimal_causal_lm:
        monkeypatch.setattr(module, 'HF_ARCH', 'FixtureForCausalLM')
        monkeypatch.setattr(module, 'DECODER_ARCH', 'llama')
        monkeypatch.setattr(module, 'SUPPORTED_PROFILES', frozenset({(2048, 8192, 16, 32, 8)}))
        monkeypatch.setattr(module, 'RMSNORM_CLASS', object)
        monkeypatch.setattr(module, 'ROTARY_CLASS', object)
        monkeypatch.setattr(module, '_install_class_patches_once', lambda : events.append('class_patches'))
    monkeypatch.setattr(module, '_claim_live_instance', lambda owner: events.append('claim'))
    swizzle_name = 'convert_linear_weights_inplace' if module is qwen3 else 'swizzle_model_inplace'
    if module is qwen3 and (not hasattr(model, 'lm_head')):
        model.lm_head = nn.Linear(16, 32, bias=False, dtype=torch.float16)

    def swizzle(*args, **kwargs):
        events.append('swizzle')
        if module is qwen3:
            model.lm_head._rpu_linear_partition = 1
            model.lm_head._rpu_linear_num_cores = kwargs.get('lm_head_num_cores', 8)
    monkeypatch.setattr(module, swizzle_name, swizzle)

    def install(inner, **kwargs):
        events.append('decoder_install')
        if 'topology' in kwargs:
            inner._rpu_decoder_topology = kwargs['topology']

        def rpu_decoder_model_forward(self, *args, **kwargs):
            return 'rpu-forward'
        inner._rpu_deepstack_lang_layers = []
        inner._rpu_batch_decode_enabled = kwargs.get('arch') == 'qwen3'
        inner._rpu_decoder_graph_cache = _GraphCache(events)
        inner._rpu_decoder_num_layers = 1
        inner._rpu_decoder_hidden_size = 16
        inner._rpu_decoder_deepstack_hash = 0
        inner._rpu_decode_stage_descriptor = (3, 1, 1)
        inner._rpu_decoder_handle = 41
        resource = decoder._InstalledNativeResource(inner, 41, lambda handle: events.append('handle_finalize'), graphs=(inner._rpu_decoder_graph_cache,), keepalive=(), label='test decoder', handle_name='_rpu_decoder_handle')
        inner._rpu_decoder_retirement_state = resource
        inner._rpu_decoder_handle_finalizer = resource.finalizer
        inner.forward = MethodType(rpu_decoder_model_forward, inner)
        return 41
    monkeypatch.setattr(module, '_install_causal_decoder_forward', install)
    if module in (qwen3, llama):
        monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_enable_execution_reconfigure', lambda handle: events.append(('hot_guard', handle)), raising=False)
    monkeypatch.setattr(hw_attrs, 'validate_preinstall', lambda owner: events.append('preinstall'))

    def validate_postinstall(owner):
        events.append('postinstall')
        if postinstall_error is not None:
            raise postinstall_error
    monkeypatch.setattr(hw_attrs, 'validate_postinstall', validate_postinstall)
    monkeypatch.setattr(hw_attrs, 'install_hw_attr_validator', lambda owner: events.append('validator'))

@pytest.mark.parametrize('module,adapter_cls,config_factory', _ADAPTER_CASES)
def test_real_a9_accepts_the_actual_shared_session_install_state(monkeypatch, module, adapter_cls, config_factory):
    preinstall = hw_attrs.validate_preinstall
    postinstall = hw_attrs.validate_postinstall
    stamp = hw_attrs.install_hw_attr_validator
    model = _Outer(config_factory())
    _prepare_case(monkeypatch, module, model)
    monkeypatch.setattr(hw_attrs, 'validate_preinstall', preinstall)
    monkeypatch.setattr(hw_attrs, 'validate_postinstall', postinstall)
    monkeypatch.setattr(hw_attrs, 'install_hw_attr_validator', stamp)
    assert adapter_cls(model).to_rpu() is model
    assert model._rpu_swizzled is True
    postinstall(model)
    assert model.model._rpu_decode_stage_descriptor == (3, 1, 1)
    if module in (qwen3, llama):
        assert model._rpu_execution_generation == model._execution_session.generation
        assert model.model._rpu_execution_generation == model._execution_session.generation

@pytest.mark.parametrize('cores', [4, 6, 8])
@pytest.mark.parametrize('hidden, intermediate, layers, heads', [(1024, 3072, 28, 16), (2048, 6144, 28, 16), (2560, 9728, 36, 32), (4096, 12288, 36, 32)])
def test_model_topology_changes_fail_before_native_or_graph_mutation(monkeypatch, cores, hidden, intermediate, layers, heads):
    config = _qwen3_config()
    (config.hidden_size, config.intermediate_size) = (hidden, intermediate)
    (config.num_hidden_layers, config.num_attention_heads) = (layers, heads)
    (config.head_dim, config.vocab_size, config.model_type) = (128, 151936, 'qwen3')
    model = _Outer(config)
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from torch._subclasses.fake_tensor import FakeTensorMode
    # Exact8B and 6-core lm_head/MLP padding validate CPU residency before claiming the model.
    # Keep that validation real without allocating the checkpoint in this test.
    with (FakeTensorMode() if hidden == 4096 or cores == 6 else torch.device("meta")):
        actual = Qwen3ForCausalLM._from_config(Qwen3Config(**vars(config)), dtype=torch.float16)
    model.model = actual.model
    model.lm_head = actual.lm_head
    model._rpu_execution = {'model': {'num_cores': cores}}
    preinstall = hw_attrs.validate_preinstall
    postinstall = hw_attrs.validate_postinstall
    stamp = hw_attrs.install_hw_attr_validator
    _prepare_case(monkeypatch, qwen3, model)
    monkeypatch.setattr(hw_attrs, 'validate_preinstall', preinstall)
    monkeypatch.setattr(hw_attrs, 'validate_postinstall', postinstall)
    monkeypatch.setattr(hw_attrs, 'install_hw_attr_validator', stamp)
    adapter = qwen3.Qwen3Adapter(model)
    with (model.lm_head.weight.fake_mode if hidden == 4096 or cores == 6 else contextlib.nullcontext()):
        adapter.to_rpu()
    session = model._execution_session
    ops = _NativeReconfigureOps([])
    monkeypatch.setattr(decoder, 'torch', SimpleNamespace(ops=SimpleNamespace(rpu=ops)))
    (old, events) = (session.config, list(model._events))
    patches = [{'model': {'num_cores': 8 if cores != 8 else 6}}]
    if cores != 8:
        from rpu_backend.runtime import RPUConfigError
        patches.append({'model': {}})
        mlp_cores = cores
        assert model._rpu_decoder_topology.identity() == (2, cores, 4, mlp_cores, cores, 8)
        with pytest.raises(RPUConfigError, match="monotonic"):
            model._rpu_decoder_topology = model._rpu_decoder_topology
    for patch in patches:
        with pytest.raises(ValueError, match='cold-only'):
            reconfigure_rpu_execution(model, patch)
        assert session.config is old and session.generation == 0
        assert ops.events == [] and model._events == events
    reconfigure_rpu_execution(model, {'prefill': {'chunk_size': 64}})
    assert session.generation == 1 and ops.current_chunk == 64
    assert session.config['model'] == {'num_cores': cores}

@pytest.mark.parametrize('module,adapter_cls,config_factory', _ADAPTER_CASES)
def test_ready_is_published_last_and_reentry_is_idempotent(monkeypatch, module, adapter_cls, config_factory):
    model = _Outer(config_factory())
    _prepare_case(monkeypatch, module, model)
    adapter = adapter_cls(model)
    assert adapter.to_rpu() is model
    assert decoder._causal_lm_runtime_complete(model) is True
    assert model._events.index('validator') < model._events.index('ready_publish')
    first_events = list(model._events)
    second_adapter = adapter_cls(model)
    assert second_adapter.to_rpu() is model
    assert model._events == first_events

@pytest.mark.parametrize('module,adapter_cls,config_factory', _ADAPTER_CASES)
def test_postinstall_base_exception_cleans_runtime_and_poison_model(monkeypatch, module, adapter_cls, config_factory):
    model = _Outer(config_factory())
    _prepare_case(monkeypatch, module, model, postinstall_error=_InstallInterrupted('synthetic interruption'))
    adapter = adapter_cls(model)
    with pytest.raises(_InstallInterrupted, match='synthetic interruption'):
        adapter.to_rpu()
    assert 'graph_clear' in model._events
    assert 'handle_finalize' in model._events
    assert model._events.index('graph_clear') < model._events.index('handle_finalize')
    assert model._rpu_swizzle_started is True
    assert not hasattr(model, '_rpu_swizzled')
    assert decoder._causal_lm_runtime_complete(model) is False
    assert 'forward' not in vars(model.model)
    assert model.model.forward() == 'original-forward'
    for attr in decoder._CAUSAL_DECODER_INSTALL_ATTRS:
        assert not hasattr(model.model, attr)
    with pytest.raises(RPUBackendError, match='prior swizzle|prior.*failed'):
        adapter.to_rpu()
    with pytest.raises(RPUBackendError, match='swizzled|device'):
        model.to('cpu')

@pytest.mark.parametrize('module,adapter_cls,config_factory', _ADAPTER_CASES)
def test_incomplete_ready_marker_is_rejected_before_claim_or_swizzle(monkeypatch, module, adapter_cls, config_factory):
    model = _Outer(config_factory())
    model._rpu_swizzled = True
    model._events.clear()
    _prepare_case(monkeypatch, module, model)
    adapter = adapter_cls(model)
    with pytest.raises(RPUBackendError, match='runtime ownership is incomplete'):
        adapter.to_rpu()
    assert 'claim' not in model._events
    assert 'swizzle' not in model._events

@pytest.mark.parametrize('module,adapter_cls,config_factory', _ADAPTER_CASES)
def test_ready_publish_base_exception_rolls_back_adapter_and_runtime(monkeypatch, module, adapter_cls, config_factory):
    model = _Outer(config_factory())
    _prepare_case(monkeypatch, module, model)
    adapter = adapter_cls(model)
    model._fail_ready_publish = True
    with pytest.raises(_InstallInterrupted, match='ready publish interruption'):
        adapter.to_rpu()
    assert adapter._rpu_is_ready is False
    assert not hasattr(model, '_rpu_swizzled')
    assert model._rpu_swizzle_started is True
    assert model._events.index('validator') < model._events.index('ready_publish')
    assert model._events.index('graph_clear') < model._events.index('handle_finalize')
    assert 'forward' not in vars(model.model)
    for attr in decoder._CAUSAL_DECODER_INSTALL_ATTRS:
        assert not hasattr(model.model, attr)

def test_cleanup_does_not_double_destroy_a_dead_finalizer(monkeypatch):
    events = []
    direct_destroys = []
    inner = _Inner()
    inner._rpu_decoder_graph_cache = _GraphCache(events)
    inner._rpu_decoder_handle = 53
    resource = decoder._InstalledNativeResource(inner, 53, lambda handle: None, graphs=(inner._rpu_decoder_graph_cache,), keepalive=(), label='test decoder', handle_name='_rpu_decoder_handle')
    inner._rpu_decoder_retirement_state = resource
    inner._rpu_decoder_handle_finalizer = resource.finalizer
    resource.retire()
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_destroy', lambda handle: direct_destroys.append(handle), raising=False)
    decoder._cleanup_causal_decoder_install(inner, had_instance_forward=False, original_instance_forward=None)
    assert events == ['graph_clear']
    assert direct_destroys == []
    assert not hasattr(inner, '_rpu_decoder_handle')
    assert not hasattr(inner, '_rpu_decoder_handle_finalizer')

@pytest.mark.parametrize('module,adapter_cls,config_factory', _PLAIN_ADAPTER_CASES)
def test_plain_causal_lm_hot_exact_commits_one_generation_and_retires_graphs(monkeypatch, module, adapter_cls, config_factory):
    model = _Outer(config_factory())
    model._rpu_execution = {'prefill': {'chunk_size': 'auto', 'padding_budget': 32}}
    _prepare_case(monkeypatch, module, model)
    adapter = adapter_cls(model)
    assert adapter.to_rpu() is model
    session = model._execution_session
    inner = model.model
    assert adapter._execution_session is session
    assert inner._execution_session is session
    assert model._rpu_execution is session.config
    assert adapter._rpu_execution is session.config
    assert inner._rpu_execution is session.config
    inner._rpu_last_a6_plan = {'generation': 0}
    inner._rpu_last_execution_plan = {'generation': 0}
    events = []
    ops = _NativeReconfigureOps(events)
    monkeypatch.setattr(decoder, 'torch', SimpleNamespace(ops=SimpleNamespace(rpu=ops)))
    published = reconfigure_rpu_execution(model, {'prefill': {'chunk_size': 64, 'padding_budget': 32}})
    assert published['prefill'] == {'chunk_size': 64, 'padding_budget': 32}
    assert ops.current_chunk == 64
    assert events == [('txn_begin', 71), ('stage_chunk', 71, 64), ('txn_commit', 71, 64)]
    assert model._events[-2:] == ['graph_warmup', 'graph_clear']
    assert session.generation == 1
    assert inner._rpu_execution_generation == 1
    assert model._rpu_execution is published
    assert adapter._rpu_execution is published
    assert inner._rpu_execution is published
    assert not hasattr(inner, '_rpu_last_a6_plan')
    assert not hasattr(inner, '_rpu_last_execution_plan')
    assert rpu_execution_stats(model)['commit_count'] == 1

@pytest.mark.parametrize('module,adapter_cls,config_factory', _PLAIN_ADAPTER_CASES)
def test_same_config_calibration_retires_real_causal_owner(monkeypatch, module, adapter_cls, config_factory):
    from rpu_backend.api._execution import record_planner_native
    from rpu_backend.runtime.execution_planner import PlannerCalibrationSelection, PlannerCostScope
    model = _Outer(config_factory())
    _prepare_case(monkeypatch, module, model)
    adapter = adapter_cls(model)
    adapter.to_rpu()
    (session, inner) = (model._execution_session, model.model)
    model._events.clear()
    ops = _NativeReconfigureOps([])
    monkeypatch.setattr(decoder, 'torch', SimpleNamespace(ops=SimpleNamespace(rpu=ops)))
    record_planner_native(inner, 'prefill', '', ('causal_decoder', inner._rpu_decoder_handle))
    selection = PlannerCalibrationSelection(PlannerCostScope('a' * 64, 'b' * 64, 'c' * 64), 'd' * 64, 'e' * 64)
    old = session.config
    session._select_planner_calibration(inner, 'prefill', selection, native_prefix='causal_decoder')
    assert session.config == old and session.generation == 1
    assert inner._rpu_execution_generation == model._rpu_execution_generation == 1
    assert model._events == ['graph_warmup', 'graph_clear']
    assert ops.current_chunk == 0

@pytest.mark.parametrize('module,adapter_cls,config_factory', _PLAIN_ADAPTER_CASES)
def test_plain_causal_lm_ambiguous_commit_restores_auto_without_graph_loss(monkeypatch, module, adapter_cls, config_factory):
    model = _Outer(config_factory())
    model._rpu_execution = {'prefill': {'chunk_size': 'auto'}}
    _prepare_case(monkeypatch, module, model)
    adapter = adapter_cls(model)
    assert adapter.to_rpu() is model
    session = model._execution_session
    old = session.config
    inner = model.model
    inner._rpu_last_execution_plan = {'generation': 0}
    model._events.clear()
    events = []
    ops = _NativeReconfigureOps(events, interrupt_first_commit=True)
    monkeypatch.setattr(decoder, 'torch', SimpleNamespace(ops=SimpleNamespace(rpu=ops)))
    with pytest.raises(KeyboardInterrupt, match='ambiguous native commit'):
        reconfigure_rpu_execution(model, {'prefill': {'chunk_size': 64}})
    assert ops.current_chunk == 0
    assert [event for event in events if event[0] == 'stage_chunk'] == [('stage_chunk', 71, 64), ('stage_chunk', 72, 0)]
    assert model._events == []
    assert inner._rpu_last_execution_plan == {'generation': 0}
    assert session.config is old
    assert session.generation == 0
    assert model._rpu_execution is old
    assert adapter._rpu_execution is old
    assert inner._rpu_execution is old
    assert inner._rpu_execution_generation == 0
    assert rpu_execution_stats(model)['failure_count'] == 1
    assert rpu_execution_stats(model)['state'] == 'QUIESCENT'

def test_plain_causal_lm_forward_and_graph_keys_are_generation_guarded():
    root = Path(__file__).resolve().parents[4]
    decoder_source = (root / 'python/rpu_backend/runtime/decoder.py').read_text()
    qwen_source = (root / 'python/rpu_backend/adapters/qwen3.py').read_text()
    loader_source = (root / 'python/rpu_backend/api/causal_lm.py').read_text()
    assert 'rpu_decoder_model_forward = execution_serialized(' in decoder_source
    assert 'getattr(self, "_rpu_execution_generation", 0)' in decoder_source
    assert qwen_source.count('@execution_serialized') >= 2
    assert qwen_source.count('"_rpu_execution_generation", 0') >= 2
    assert 'adapter._rpu_execution = execution_config' not in loader_source
