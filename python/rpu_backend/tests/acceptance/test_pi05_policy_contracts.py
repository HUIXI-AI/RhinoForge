from __future__ import annotations
import ast
import builtins
from collections import deque
from pathlib import Path
import pytest
import torch
from rpu_backend.api.errors import RPUBackendError
from rpu_backend.api.policy import Pi05Policy

class _FakeAdapter:

    def __init__(self):
        self.prepare_calls = []

    def prepare_graphs(self, batch, *, num_steps=None, precompute_adarms=False):
        self.prepare_calls.append((batch, num_steps, precompute_adarms))
        return {'prepared': True}

class _FakeLeRobotPolicy:

    def __init__(self):
        self.predict_calls = []
        self.select_result = torch.zeros(4)

    def predict_action_chunk(self, batch, **kwargs):
        self.predict_calls.append((batch, kwargs))
        return torch.zeros(1, 2, 4)

    def select_action(self, batch):
        return self.select_result

def _ready_policy() -> tuple[Pi05Policy, _FakeLeRobotPolicy, _FakeAdapter]:
    policy = object.__new__(Pi05Policy)
    runtime = _FakeLeRobotPolicy()
    adapter = _FakeAdapter()
    policy._lerobot_policy = runtime
    policy._adapter = adapter
    policy._rpu_ready = True
    return (policy, runtime, adapter)

def test_invalid_trust_remote_code_fails_before_adapter_import(monkeypatch):
    original_import = builtins.__import__
    imports = []

    def tracking_import(name, *args, **kwargs):
        if name == 'rpu_backend.adapters.pi05.loader':
            imports.append(name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', tracking_import)
    with pytest.raises(TypeError, match='trust_remote_code must be bool'):
        Pi05Policy.from_pretrained('unused', trust_remote_code='false')
    assert imports == []

@pytest.mark.parametrize('value', [-1, 1.5, 17, True, '400'])
def test_invalid_vlm_chunk_fails_before_adapter_import(monkeypatch, value):
    original_import = builtins.__import__
    imports = []

    def tracking_import(name, *args, **kwargs):
        if name == 'rpu_backend.adapters.pi05.loader':
            imports.append(name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', tracking_import)
    with pytest.raises(ValueError, match='vlm_chunk_size'):
        Pi05Policy.from_pretrained('unused', vlm_chunk_size=value)
    assert imports == []

def test_loader_defense_rejects_truthy_string_without_loading(monkeypatch):
    from rpu_backend.adapters.pi05 import loader
    called = False

    def fake_load(*args, **kwargs):
        nonlocal called
        called = True
    monkeypatch.setattr(loader, '_load_and_fp16_cast', fake_load)
    with pytest.raises(TypeError, match='trust_remote_code must be bool'):
        loader.load_and_construct_pi05_policy('unused', trust_remote_code='false')
    assert called is False

def test_loader_forwards_explicit_remote_code_opt_in_with_warning(monkeypatch):
    from rpu_backend.adapters.pi05 import loader
    runtime = object()
    observed = {}

    def fake_load(path, dtype, **kwargs):
        observed.update(path=path, dtype=dtype, kwargs=kwargs)
        return runtime
    monkeypatch.setattr(loader, '_load_and_fp16_cast', fake_load)
    monkeypatch.setattr(Pi05Policy, 'from_lerobot_policy', classmethod(lambda cls, policy: (cls, policy)))
    with pytest.warns(UserWarning, match='arbitrary code execution'):
        wrapped = loader.load_and_construct_pi05_policy('checkpoint', trust_remote_code=True)
    assert wrapped == (Pi05Policy, runtime)
    assert observed == {'path': 'checkpoint', 'dtype': torch.float16, 'kwargs': {'trust_remote_code': True}}

@pytest.mark.parametrize('num_steps', [0, -1, 1.5, True, '5'])
@pytest.mark.parametrize('method_name', ['prepare_graphs', 'predict_action_chunk'])
def test_num_steps_rejects_coercion_before_runtime(method_name, num_steps):
    (policy, runtime, adapter) = _ready_policy()
    method = getattr(policy, method_name)
    with pytest.raises(ValueError, match='num_steps'):
        method({}, num_steps=num_steps)
    assert runtime.predict_calls == []
    assert adapter.prepare_calls == []

@pytest.mark.parametrize('method_name', ['prepare_graphs', 'predict_action_chunk', 'select_action'])
def test_public_methods_require_dict_batch(method_name):
    (policy, runtime, adapter) = _ready_policy()
    with pytest.raises(TypeError, match='batch must be a dict'):
        getattr(policy, method_name)([])
    assert runtime.predict_calls == []
    assert adapter.prepare_calls == []

def test_valid_num_steps_is_forwarded_without_conversion():
    (policy, runtime, adapter) = _ready_policy()
    batch = {'observation.state': torch.zeros(1, 4)}
    assert policy.prepare_graphs(batch, num_steps=5) == {'prepared': True}
    assert adapter.prepare_calls == [(batch, 5, False)]
    chunk = policy.predict_action_chunk(batch, num_steps=7)
    assert tuple(chunk.shape) == (1, 2, 4)
    assert runtime.predict_calls == [(batch, {'num_steps': 7})]

def test_select_action_pins_supported_deque_without_swallowing_failures():
    (policy, runtime, _) = _ready_policy()
    runtime._action_queue = deque([torch.ones(4), torch.zeros(4)], maxlen=3)
    result = policy.select_action({})
    assert result.device.type == 'cpu'
    assert runtime._action_queue.maxlen == 3
    assert all((entry.device.type == 'cpu' for entry in runtime._action_queue))
    runtime._action_queue = []
    with pytest.raises(RPUBackendError, match='_action_queue.*deque'):
        policy.select_action({})

def test_select_action_reports_queue_transfer_and_result_contract_failures():
    (policy, runtime, _) = _ready_policy()
    runtime._action_queue = deque([torch.empty(1, device='meta')])
    with pytest.raises(RPUBackendError, match='move.*_action_queue.*CPU'):
        policy.select_action({})
    del runtime._action_queue
    runtime.select_result = 'not-a-tensor'
    with pytest.raises(RPUBackendError, match='must return a torch.Tensor'):
        policy.select_action({})

def test_internal_forward_requires_boolean_return_chunk():
    (policy, _, _) = _ready_policy()
    with pytest.raises(TypeError, match='return_chunk must be bool'):
        policy._forward_rpu({}, return_chunk='false')

def test_fused_parity_validator_reports_metrics_and_rejects_invalid_boundaries():
    from rpu_backend.adapters.pi05.runtime import _PI05_FUSED_OUTPUT_SHAPE, _validate_fused_parity
    reference = torch.ones(_PI05_FUSED_OUTPUT_SHAPE)
    metrics = _validate_fused_parity(reference + 0.5, reference)
    assert metrics['mse'] == 0.25 and metrics['max_abs'] == 0.5
    assert metrics['row_mse_max'] == 0.25
    assert 'relative_l2' in metrics and 'row_cosine_p01' in metrics
    zeros = torch.zeros_like(reference)
    assert _validate_fused_parity(zeros, zeros)['row_both_zero_count'] == 50
    tiny = zeros.clone()
    tiny[0, 0, 0] = 1e-07
    assert _validate_fused_parity(tiny, zeros)['row_single_zero_count'] == 1
    with pytest.raises(RuntimeError, match='shape mismatch'):
        _validate_fused_parity(torch.ones(1, 49, 32), reference)
    with pytest.raises(RuntimeError, match='expected shape'):
        _validate_fused_parity(torch.ones(2, 50, 32), torch.ones(2, 50, 32))
    with pytest.raises(RuntimeError, match='device mismatch'):
        _validate_fused_parity(torch.ones(_PI05_FUSED_OUTPUT_SHAPE, device='meta'), reference)
    for side in (0, 1):
        values = [reference, reference]
        values[side] = reference.half()
        with pytest.raises(RuntimeError, match='dtype mismatch'):
            _validate_fused_parity(*values)
        for invalid in (float('nan'), float('inf'), -float('inf')):
            values[side] = torch.full_like(reference, invalid)
            with pytest.raises(RuntimeError, match='non-finite'):
                _validate_fused_parity(*values)

def test_pi05_production_invariants_survive_python_optimized_mode():
    root = Path(__file__).resolve().parents[4]
    paths = (root / 'python/rpu_backend/adapters/pi05/__init__.py', root / 'python/rpu_backend/adapters/pi05/loader.py', root / 'python/rpu_backend/adapters/pi05/runtime.py', root / 'python/rpu_backend/adapters/pi05/weights.py')
    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)], path

def test_pi05_top_level_is_serialized_and_child_graphs_key_physical_plan():
    root = Path(__file__).resolve().parents[4]
    runtime_source = (root / 'python/rpu_backend/adapters/pi05/runtime.py').read_text(encoding='utf-8')
    assert runtime_source.count('torch.ops.rpu.pi05_denoise_step_get_resolved_chunk_size(handle)') == 2
    runtime_tree = ast.parse(runtime_source)
    patched = next((node for node in ast.walk(runtime_tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == '_patched_sample_actions'))
    decorator_names = {node.id for node in patched.decorator_list if isinstance(node, ast.Name)}
    assert 'execution_serialized' in decorator_names
    assert runtime_source.count('action_plan.graph_key_words()') == 2
    denoise_step_calls = [
        node for node in ast.walk(runtime_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'pi05_denoise_step_forward'
    ]
    assert len(denoise_step_calls) == 2
    assert all(
        ast.unparse(call.args[-1])
        == 'action_plan.selected.stage_tuple.physical_descriptor'
        for call in denoise_step_calls
    )
    siglip_source = (root / 'python/rpu_backend/adapters/siglip.py').read_text(encoding='utf-8')
    adarms_source = (root / 'python/rpu_backend/adapters/pi05/adarms.py').read_text(encoding='utf-8')
    assert '_component_plan.graph_key_words()' in siglip_source
    assert 'torch.ops.rpu.siglip_resolve_stage_domain(' in siglip_source
    assert 'planned_stage_descriptor' in siglip_source
    assert 'torch.ops.rpu.siglip_get_resolved_chunk_size(' not in siglip_source
    assert '_component_plan.graph_key_words()' in adarms_source
    siglip_cpp = (root / 'src/fused/rpu_siglip_model.cpp').read_text(encoding='utf-8')
    assert 'image_batch_count_ = packed_image_count;' in siglip_cpp
    assert siglip_cpp.count('configure_chunk_for_shape(') == 4


def test_pi05_denoise_graph_keeps_only_the_current_signature():
    from rpu_backend.adapters.pi05.runtime import _capture_bounded_denoise_graph

    events = []

    class _Scope:

        def __init__(self, sig):
            self.sig = sig

        def __enter__(self):
            events.append(('enter', self.sig))

        def __exit__(self, *_exc):
            events.append(('exit', self.sig))

    class _Cache:

        def capture(self, sig):
            events.append(('capture', sig))
            return _Scope(sig)

        def evict(self, sig):
            events.append(('evict', sig))

    cache = _Cache()
    sig_a = object()
    sig_b = object()
    with _capture_bounded_denoise_graph(cache, sig_a):
        pass
    with _capture_bounded_denoise_graph(cache, sig_a):
        pass
    with _capture_bounded_denoise_graph(cache, sig_b):
        pass

    assert [event for event in events if event[0] == 'evict'] == [
        ('evict', sig_a),
    ]
    assert vars(cache)['_pi05_denoise_graph_signature'] is sig_b

def test_pi05_regular_and_packed_siglip_share_component_plan_authority():
    root = Path(__file__).resolve().parents[4]
    source = (root / 'python/rpu_backend/adapters/siglip.py').read_text(encoding='utf-8')
    helper = source[source.index('def _siglip_component_execution_plan('):source.index('\n# GraphSignature branch discriminator', source.index('def _siglip_component_execution_plan('))]
    regular = source[source.index('    def rpu_siglip_forward('):source.index('\n    # P7.1g', source.index('    def rpu_siglip_forward('))]
    packed = source[source.index('def _siglip_forward_images('):]
    assert 'torch.ops.rpu.siglip_resolve_stage_domain(' in helper
    assert '("image_batch_count", int(image_batch_count))' in helper
    assert 'physical_descriptor' in helper
    assert 'queue_owner_id=int(handle)' in helper
    assert 'vars(vision_model)["_fmb_last_execution_plan"] = plan' in helper
    assert regular.count('_siglip_component_execution_plan(') == 1
    assert packed.count('_siglip_component_execution_plan(') == 1
    assert '*component_plan_words' in packed
