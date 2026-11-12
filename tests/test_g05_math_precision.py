"""G05 HIGH is an immutable cold choice, independent of Linear precision."""
from types import MethodType, SimpleNamespace as NS

import pytest
import torch

from rpu_backend.adapters.g05 import action, quant
from rpu_backend.adapters.qwen3_5 import text, vision


def test_math_admission_and_missing_native_fail_before_mutation(monkeypatch):
    from test_g05_adapter_contracts import _load_adapter

    runtime = _load_adapter()
    for value in (None, True, 'HIGH', 'fp32'):
        with pytest.raises(ValueError, match='math_precision'):
            runtime.patch_g05_policy_for_rpu(object(), math_precision=value)
    for installer in (runtime.patch_g05_vlm_for_rpu, runtime.patch_g05_vision_for_rpu,
                      action.patch_g05_action_expert_for_rpu):
        with pytest.raises(ValueError, match='exact K1'):
            installer(object(), math_precision='high')
    monkeypatch.delattr(torch.ops.rpu, 'qwen3_5_set_high_precision', raising=False)
    monkeypatch.delattr(torch.ops.rpu, 'qwen3_5_vision_set_high_precision', raising=False)
    for installer, kwargs in (
            (runtime.patch_g05_vlm_for_rpu, {'_allow_certified_g05_k1': True}),
            (runtime.patch_g05_vision_for_rpu, {'_allow_certified_g05_k1': True}),
            (action.patch_g05_action_expert_for_rpu, {'_g05_exact_k1': True})):
        owner = NS()
        with pytest.raises(RuntimeError, match='HIGH math requires native'):
            installer(owner, math_precision='high', **kwargs)
        assert not vars(owner)
    with pytest.raises(ValueError, match='CPU-staged'):
        text.install_qwen3_5_text_for_rpu(object(), object(), _g05_math_precision='high')


@pytest.mark.parametrize('action_role', [False, True])
@pytest.mark.parametrize('changed_field', [None, 'num_key_value_heads', 'rms_norm_eps', 'rope_parameters'])
def test_shared_cold_admission_keeps_real_k1_geometry(action_role, changed_field):
    from copy import deepcopy
    from rpu_backend.adapters.qwen3_5 import _g05_cold
    from rpu_backend.api.errors import UnsupportedModelError
    from test_g05_adapter_contracts import _config, _load_adapter

    config = _config()
    if action_role:
        vars(config).update(hidden_size=1024, intermediate_size=4096,
                            adaptive_mode='adaLN', time_hidden_size=1024,
                            input_type='linear', input_dim=27,
                            output_type='linear', output_dim=27,
                            layer_types=['full_attention'] * 24)
    config.quant_config = quant.g05_w8_quant_config(action=action_role)
    if changed_field == 'num_key_value_heads':
        config.num_key_value_heads = 8
    elif changed_field == 'rms_norm_eps':
        config.rms_norm_eps = 1e-5
    elif changed_field == 'rope_parameters':
        config.rope_parameters['partial_rotary_factor'] = 0.5
    before = deepcopy(vars(config))
    legacy_check = (action._check_g05_action_profile if action_role else
                    _load_adapter()._check_g05_vlm_profile)
    checks = [legacy_check]
    for namespace in (_g05_cold, quant):
        checks.extend((
            lambda c, n=namespace: n.validate_g05_high_math_install(c, cpu_stage_weights=True),
            lambda c, n=namespace: n.validate_g05_w8_install(c, cpu_stage_weights=True)))
    for check in checks:
        if changed_field is None:
            check(config)
        else:
            with pytest.raises(UnsupportedModelError, match='canonical Qwen3.5'):
                check(config)
    assert vars(config) == before


@pytest.mark.parametrize('weight_dtype', ['fp16', 'w8a16'])
@pytest.mark.parametrize('accumulation', ['acc16', 'acc32'])
def test_high_vlm_install_commit_idempotence_and_mode_conflict(monkeypatch, request, weight_dtype, accumulation):
    from test_g05_adapter_contracts import _load_adapter, _g05_vlm_installer_fixture

    runtime = _load_adapter()
    events = []
    model, vlm = _g05_vlm_installer_fixture(monkeypatch, runtime, events, request,
                                          linear_acc32=accumulation == 'acc32')
    monkeypatch.setattr(runtime, 'quantize_g05_text_weights', lambda _: None)
    monkeypatch.setattr(torch.ops.rpu, 'qwen3_5_set_high_precision', lambda *_: None, raising=False)
    kwargs = dict(_allow_certified_g05_k1=True, weight_dtype=weight_dtype,
                  linear_accumulation=accumulation, math_precision='high')
    assert runtime.patch_g05_vlm_for_rpu(model, **kwargs) is model
    assert vlm._rpu_g05_math_precision == vlm._rpu_qwen3_5.g05_math_precision == 'high'
    assert vlm._rpu_qwen3_5.linear_acc32 is (accumulation == 'acc32')
    before = list(events)
    assert runtime.patch_g05_vlm_for_rpu(model, **kwargs) is model
    with pytest.raises(ValueError, match='math_precision conflicts'):
        runtime.patch_g05_vlm_for_rpu(model, **dict(kwargs, math_precision='base'))
    assert events == before
    vlm._rpu_qwen3_5.g05_math_precision = 'base'
    assert not runtime._g05_vlm_runtime_ready(model)


def test_policy_math_install_is_immutable_and_transaction_rolls_back_markers(monkeypatch):
    from test_g05_adapter_contracts import _load_adapter

    runtime = _load_adapter()
    ready = NS(_rpu_g05_policy_ready=True, _rpu_g05_math_precision='high')
    monkeypatch.setattr(runtime, '_g05_policy_runtime_ready', lambda _: True)
    assert runtime.patch_g05_policy_for_rpu(ready, math_precision='high') is ready
    with pytest.raises(ValueError, match='math_precision conflicts'):
        runtime.patch_g05_policy_for_rpu(ready)
    policy = NS(model=NS(vision_tower=NS(), vlm=NS(), action_expert=NS()))
    owners = [policy, policy.model, policy.model.vision_tower,
              policy.model.vlm, policy.model.action_expert]

    @runtime._transactional_g05_policy_install
    def fail(owner, **kwargs):
        owner._rpu_g05_policy_install_started = True
        for child in owners:
            child._rpu_g05_math_precision = 'high'
        raise RuntimeError('publish failed')

    with pytest.raises(RuntimeError, match='publish failed'):
        fail(policy, math_precision='high')
    assert all(not hasattr(child, '_rpu_g05_math_precision') for child in owners)
    assert policy._rpu_g05_policy_install_started


@pytest.mark.parametrize('accumulation', ['acc16', 'acc32'])
def test_action_high_reaches_shared_cold_installer_without_changing_acc(monkeypatch, accumulation):
    from test_g05_w8a16_adapter import _toy_backbone

    expert = _toy_backbone(action=True).eval()
    monkeypatch.setattr(action, '_check_g05_action_profile', lambda _: None)
    monkeypatch.setattr(torch.ops.rpu, 'qwen3_5_set_high_precision', lambda *_: None, raising=False)

    def install(owner, config, **kwargs):
        assert kwargs['_g05_math_precision'] == 'high'
        assert kwargs['execution_config'] == {'prefill': {'linear_acc32': accumulation == 'acc32'}}
        raise RuntimeError('reached shared cold install')

    monkeypatch.setattr(text, 'install_qwen3_5_text_for_rpu', install)
    with pytest.raises(RuntimeError, match='reached shared cold install'):
        action.patch_g05_action_expert_for_rpu(expert, _g05_exact_k1=True,
                                               linear_accumulation=accumulation, math_precision='high')
    assert not hasattr(expert, '_rpu_g05_math_precision')


@pytest.mark.parametrize('action_role', [False, True])
@pytest.mark.parametrize('linear_acc32', [False, True])
def test_fp16_shared_text_high_binds_before_set_weights(monkeypatch, action_role, linear_acc32):
    from rpu_backend.runtime import _native_retirement
    from test_g05_w8a16_adapter import _toy_backbone

    owner = _toy_backbone(action=action_role)
    original_to = torch.Tensor.to

    def host_to(value, *args, **kwargs):
        args = tuple('cpu' if isinstance(arg, str) and arg == 'rpu' else arg for arg in args)
        if kwargs.get('device') == 'rpu':
            kwargs['device'] = 'cpu'
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', host_to)
    monkeypatch.setattr(text, '_refresh_qwen35_text_retirement', lambda _: None)
    monkeypatch.setattr(_native_retirement, '_InstalledNativeResource',
                        lambda *a, **kw: NS(keepalive=kw['keepalive'], finalizer=None))
    monkeypatch.setattr(torch.ops.rpu, 'qwen3_5_create', lambda: 23, raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'qwen3_5_destroy', lambda *_: None, raising=False)
    calls = []
    monkeypatch.setattr(torch.ops.rpu, 'qwen3_5_set_linear_acc32',
                        lambda *args: calls.append(('acc', *args)), raising=False)
    def high(*args):
        calls.append(('high', *args))
        raise RuntimeError('reached HIGH cold setter')
    monkeypatch.setattr(torch.ops.rpu, 'qwen3_5_set_high_precision', high, raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'qwen3_5_set_weights',
                        lambda *a, **kw: pytest.fail('must stop before set_weights'), raising=False)
    with pytest.raises(RuntimeError, match='reached HIGH cold setter'):
        text._install_qwen3_5_text_for_rpu_impl(owner, owner.config, max_seq_len=64,
            chunk_size_cap=0, exact_chunk_size=0, padding_budget=0, padding_rows='auto',
            execution_config={'prefill': {'linear_acc32': linear_acc32}}, control_snapshot={},
            graph_cache=None, prefill_graph=None, cpu_stage_weights=True,
            g05_math_precision='high', pending_state=NS(handle=None))
    assert calls == [('acc', 23, linear_acc32), ('high', 23, True)]


@pytest.mark.parametrize('linear_acc32', [False, True])
def test_vision_high_binds_after_acc_before_any_weight_transform(monkeypatch, linear_acc32):
    from rpu_backend.runtime import _native_retirement
    from test_qwen3_5_vision_helpers import _runtime_capable_vision

    owner = _runtime_capable_vision()
    calls = []
    for name in ('qwen3_5_vision_set_chunk_size_cap', 'qwen3_5_vision_set_cache_capacity',
                 'qwen3_5_vision_resolve_prefill_domain',
                 'qwen3_5_vision_set_prefill_chunk_size', 'qwen3_5_vision_set_weight_scales',
                 'qwen3_5_vision_destroy'):
        monkeypatch.setattr(torch.ops.rpu, name, lambda *_: None, raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'qwen3_5_vision_create', lambda: 41, raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'qwen3_5_vision_set_linear_acc32',
                        lambda *args: calls.append(('acc', *args)), raising=False)
    monkeypatch.setattr(_native_retirement, '_InstalledNativeResource',
                        lambda *a, **kw: NS(finalizer=None, handle=None))
    monkeypatch.setattr(vision, '_convert_vision_block_weights_for_rpu',
                        lambda *a, **kw: pytest.fail('must stop before weight transformation'))

    def high(*args):
        calls.append(('high', *args))
        raise RuntimeError('reached HIGH cold setter')

    monkeypatch.setattr(torch.ops.rpu, 'qwen3_5_vision_set_high_precision', high, raising=False)
    with pytest.raises(RuntimeError, match='reached HIGH cold setter'):
        vision._install_qwen3_5_vision_for_rpu_impl(owner,
            execution_config={'vision': {'linear_acc32': linear_acc32}},
            _allow_certified_g05_k1=True, _g05_acc16_experiment=not linear_acc32,
            _g05_math_precision='high')
    assert calls == [('acc', 41, linear_acc32), ('high', 41, True)]


def test_shared_vision_high_scope_and_installed_mode_conflict(monkeypatch):
    from test_qwen3_5_vision_helpers import _runtime_capable_vision, _mark_complete_vision_runtime

    owner = _runtime_capable_vision()
    owner.config.depth = 24
    monkeypatch.setattr(torch.ops.rpu, 'qwen3_5_vision_set_high_precision', lambda *_: None, raising=False)
    with pytest.raises(ValueError, match='private to the exact G05'):
        vision.install_qwen3_5_vision_for_rpu(owner, _g05_math_precision='high')
    finalizer = _mark_complete_vision_runtime(owner)
    # The reusable helper also tests a separately loaded source module. This
    # regression calls the actual installer and binds its actual forward.
    owner.forward = MethodType(vision._rpu_vision_forward, owner)
    try:
        kwargs = dict(_allow_certified_g05_k1=True, _g05_acc16_experiment=True,
                      execution_config={'vision': {'linear_acc32': False}})
        with pytest.raises(RuntimeError, match='math precision is immutable'):
            vision.install_qwen3_5_vision_for_rpu(owner, _g05_math_precision='high', **kwargs)
        assert owner._rpu_vision_math_precision == 'base'
    finally:
        finalizer.detach()
