from __future__ import annotations
import gc
import math
from types import SimpleNamespace
import pytest
import torch
import rpu_backend
from rpu_backend.adapters import siglip

class _FakeTensor:

    def __init__(self, *shape):
        self.shape = shape
        self.dtype = torch.float16
        self.device = torch.device("cpu")

    def dim(self):
        return len(self.shape)

    def size(self, axis):
        return self.shape[axis]

    def numel(self):
        return math.prod(self.shape)

    def is_contiguous(self):
        return True

    def to(self, *args, **kwargs):
        return self

    def contiguous(self):
        return self

class _PublishInterrupted(BaseException):
    pass

class _Model:

    def arm_assignment_failure(self, name):
        object.__setattr__(self, '_fail_assignment', name)

    def __setattr__(self, name, value):
        if self.__dict__.get('_fail_assignment') == name:
            object.__setattr__(self, '_fail_assignment', None)
            raise _PublishInterrupted(f'synthetic publication failure: {name}')
        object.__setattr__(self, name, value)

def _linear(out_features, in_features):
    return SimpleNamespace(weight=_FakeTensor(out_features, in_features), bias=_FakeTensor(out_features), out_features=out_features)

def _model():
    hidden = 128
    intermediate = 256
    attention = SimpleNamespace(q_proj=_linear(hidden, hidden), k_proj=_linear(hidden, hidden), v_proj=_linear(hidden, hidden), out_proj=_linear(hidden, hidden))
    mlp = SimpleNamespace(fc1=_linear(intermediate, hidden), fc2=_linear(hidden, intermediate))
    layer = SimpleNamespace(self_attn=attention, mlp=mlp, layer_norm1=SimpleNamespace(weight=_FakeTensor(hidden), bias=_FakeTensor(hidden)), layer_norm2=SimpleNamespace(weight=_FakeTensor(hidden), bias=_FakeTensor(hidden)))
    embeddings = SimpleNamespace(patch_embedding=SimpleNamespace(kernel_size=(14, 14), stride=(14, 14)), _rpu_gemm_weight=_FakeTensor(hidden, hidden), _rpu_pos_emb_fused=_FakeTensor(256, hidden))
    model = _Model()
    model.encoder = SimpleNamespace(layers=[layer])
    model.config = SimpleNamespace(num_attention_heads=8, hidden_size=hidden, intermediate_size=intermediate, layer_norm_eps=1e-06)
    model.post_layernorm = SimpleNamespace(weight=_FakeTensor(hidden), bias=_FakeTensor(hidden))
    model.embeddings = embeddings
    model.forward = lambda *args, **kwargs: 'original'
    projector = _linear(hidden, hidden)
    return (model, projector)

def _pi05_model(*, converted=True):
    """Full logical Pi geometry using metadata only; no model-sized storage."""
    model, _ = _model()
    hidden, qkv_rows, intermediate = 1152, 1280 if converted else 1152, 4352 if converted else 4304
    layers = []
    for _ in range(27):
        attention = SimpleNamespace(
            **{name: _linear(qkv_rows, hidden) for name in ("q_proj", "k_proj", "v_proj")},
            out_proj=_linear(hidden, qkv_rows),
        )
        if converted:
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                projection = getattr(attention, name)
                projection.weight.dtype = torch.int8
                projection.weight_scale = _FakeTensor(projection.out_features)
        mlp = SimpleNamespace(fc1=_linear(intermediate, hidden), fc2=_linear(hidden, intermediate))
        if converted:
            for projection in (mlp.fc1, mlp.fc2):
                projection.weight.dtype = torch.int8
                projection.weight_scale = _FakeTensor(projection.out_features)
        layers.append(SimpleNamespace(
            self_attn=attention,
            mlp=mlp,
            layer_norm1=SimpleNamespace(weight=_FakeTensor(hidden), bias=_FakeTensor(hidden)),
            layer_norm2=SimpleNamespace(weight=_FakeTensor(hidden), bias=_FakeTensor(hidden)),
        ))
    model.encoder.layers = layers
    model.config = SimpleNamespace(
        num_attention_heads=16, hidden_size=hidden, intermediate_size=4304,
        layer_norm_eps=1e-6, num_hidden_layers=27,
    )
    model.post_layernorm = SimpleNamespace(weight=_FakeTensor(hidden), bias=_FakeTensor(hidden))
    model.embeddings._rpu_gemm_weight = _FakeTensor(hidden, 3136)
    model.embeddings._rpu_pos_emb_fused = _FakeTensor(1, 256, hidden)
    model._rpu_siglip_weights_converted = converted
    return model, _linear(2048, hidden)


@pytest.fixture
def native_install(monkeypatch):
    from rpu_backend.api import _execution, causal_lm
    from rpu_backend.runtime import _native_retirement
    gc.collect()
    monkeypatch.setattr(_execution, '_UNSAFE_PROCESS_REASON', None)
    monkeypatch.setattr(causal_lm, '_LIVE_REF', None)
    monkeypatch.setattr(causal_lm, '_LIVE_TERMINAL_REASON', None)
    monkeypatch.setattr(_native_retirement, '_FAILED_RETIREMENTS', [])
    state = {'next_handle': 1, 'alive': set(), 'events': [], 'fail_set': set(), 'fail_patch': set(), 'envelopes': {}, 'fail_envelope': set(), 'fail_destroy': set()}

    def create(linear_acc32=False, *cold_options):
        handle = state["next_handle"]
        state["next_handle"] += 1
        state["alive"].add(handle)
        state.setdefault("accumulation", {})[handle] = linear_acc32
        state.setdefault("cold_options", {})[handle] = cold_options
        state["events"].append(("create", handle))
        return handle

    def set_weights(handle, *args):
        state['events'].append(('set_weights', handle))
        if handle in state['fail_set']:
            raise RuntimeError(f'synthetic set_weights failure for {handle}')

    def set_patch(handle, *args):
        state['events'].append(('set_patch', handle))
        if handle in state['fail_patch']:
            raise RuntimeError(f'synthetic patch failure for {handle}')

    def set_envelope(handle, capacity, chunk):
        assert state['events'][-1] == ('set_weights', handle)
        if handle in state['fail_envelope']:
            raise RuntimeError(f'synthetic envelope failure for {handle}')
        state['envelopes'][handle] = (capacity, chunk)

    def destroy(handle):
        state['events'].append(('destroy', handle))
        if handle in state['fail_destroy']:
            raise RuntimeError(f'synthetic destroy failure for {handle}')
        if handle not in state['alive']:
            raise RuntimeError(f'double destroy for {handle}')
        state['alive'].remove(handle)

    class RPUCache:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class GraphCache:

        def __init__(self):
            self.clear_calls = 0

        def clear(self):
            self.clear_calls += 1

        def cache_invariant_ok(self):
            return True
    monkeypatch.setattr(siglip, '_convert_siglip_weights_for_rpu', lambda *a, **k: None)
    monkeypatch.setattr(siglip, '_convert_siglip_projector_for_rpu', lambda *a, **k: None)
    monkeypatch.setattr(siglip, '_detect_and_validate_siglip_encoder_w8a16', lambda *a: False)
    monkeypatch.setattr(siglip, '_siglip_weight_to_rpu', lambda module: module.weight)
    monkeypatch.setattr(siglip, 'RPUCache', RPUCache)
    monkeypatch.setattr(rpu_backend.graph, 'GraphCache', GraphCache)
    from rpu_backend.graph import lazy_init_guard
    monkeypatch.setattr(lazy_init_guard, '_verify_lazy_init', lambda model: None)
    monkeypatch.setattr(siglip.torch.ops.rpu, 'siglip_create', create, raising=False)
    monkeypatch.setattr(siglip.torch.ops.rpu, 'siglip_set_weights', set_weights, raising=False)
    monkeypatch.setattr(siglip.torch.ops.rpu, 'siglip_set_weights_w8a16', set_weights, raising=False)
    monkeypatch.setattr(siglip.torch.ops.rpu, 'siglip_set_chunk_envelope', set_envelope, raising=False)
    monkeypatch.setattr(siglip.torch.ops.rpu, 'siglip_model_set_patch_emb', set_patch, raising=False)
    monkeypatch.setattr(siglip.torch.ops.rpu, 'siglip_destroy', destroy, raising=False)
    return state

def _install(model, projector):
    return siglip.patch_siglip_model_for_rpu_all_layers_once(model, projector)

def _finalize(model):
    finalizer = getattr(model, '_rpu_vision_handle_finalizer', None)
    if finalizer is not None and finalizer.alive:
        finalizer()

@pytest.mark.parametrize('num_images', [1, 3])
@pytest.mark.parametrize('w8a16', [False, True])
def test_install_binds_actual_cache_envelope(native_install, monkeypatch, num_images, w8a16):
    (model, projector) = _model()
    monkeypatch.setattr(siglip, '_detect_and_validate_siglip_encoder_w8a16', lambda *args: w8a16)
    try:
        handle = siglip.patch_siglip_model_for_rpu_all_layers_once(model, projector, num_images=num_images)
        assert native_install['envelopes'] == {handle: (256 * num_images, 0)}
        assert model._rpu_cache.kwargs['max_seq_len'] == 256 * num_images
    finally:
        _finalize(model)

def test_envelope_failure_retires_pending_handle_without_publication(native_install):
    (model, projector) = _model()
    original_forward = model.forward
    native_install['fail_envelope'].add(1)
    with pytest.raises(RuntimeError, match='envelope failure'):
        _install(model, projector)
    assert native_install['events'] == [('create', 1), ('set_weights', 1), ('destroy', 1)]
    assert not native_install['alive'] and (not native_install['envelopes'])
    assert not hasattr(model, '_rpu_vision_handle')
    assert model.forward is original_forward

def test_prepare_failure_does_not_create_or_publish_handle(native_install, monkeypatch):
    (model, projector) = _model()
    original_forward = model.forward

    class FailingCache:

        def __init__(self, **kwargs):
            raise RuntimeError('synthetic cache preparation failure')
    monkeypatch.setattr(siglip, 'RPUCache', FailingCache)
    with pytest.raises(RuntimeError, match='cache preparation failure'):
        _install(model, projector)
    assert native_install['events'] == []
    assert not hasattr(model, '_rpu_vision_handle')
    assert model.forward is original_forward

def test_set_weights_failure_cleans_pending_handle_immediately(native_install):
    (model, projector) = _model()
    original_forward = model.forward
    native_install['fail_set'].add(1)
    with pytest.raises(RuntimeError, match='set_weights failure'):
        _install(model, projector)
    assert native_install['events'] == [('create', 1), ('set_weights', 1), ('destroy', 1)]
    assert native_install['alive'] == set()
    assert not hasattr(model, '_rpu_vision_handle')
    assert model.forward is original_forward

def test_failed_replacement_preserves_published_install(native_install):
    (model, projector) = _model()
    assert _install(model, projector) == 1
    old_forward = model.forward
    old_cache = model._rpu_cache
    old_finalizer = model._rpu_vision_handle_finalizer
    native_install['events'].clear()
    native_install['fail_patch'].add(2)
    with pytest.raises(RuntimeError, match='patch failure'):
        _install(model, projector)
    assert native_install['events'] == [('create', 2), ('set_weights', 2), ('set_patch', 2), ('destroy', 2)]
    assert native_install['alive'] == {1}
    assert model._rpu_vision_handle == 1
    assert model._rpu_vision_handle_finalizer is old_finalizer
    assert old_finalizer.alive
    assert model._rpu_cache is old_cache
    assert model.forward is old_forward
    _finalize(model)

def test_old_destroy_failure_aborts_replacement(native_install):
    from rpu_backend.api import _execution
    from rpu_backend.runtime import _native_retirement
    (model, projector) = _model()
    assert _install(model, projector) == 1
    old_finalizer = model._rpu_vision_handle_finalizer
    native_install['events'].clear()
    native_install['fail_destroy'].add(1)
    with pytest.raises(RuntimeError, match='destroy failure'):
        _install(model, projector)
    assert native_install['events'] == [('create', 2), ('set_weights', 2), ('set_patch', 2), ('destroy', 1)]
    assert native_install['alive'] == {1, 2}
    pending = model._rpu_vision_retirement_state
    assert model._rpu_vision_handle == pending.handle == 2
    assert pending.failed is not None and pending.keepalive and pending.graphs
    assert {resource.handle for resource in _native_retirement._FAILED_RETIREMENTS} == {1, 2}
    assert _execution._UNSAFE_PROCESS_REASON
    before = list(native_install['events'])
    native_install['fail_destroy'].clear()
    with pytest.raises(RuntimeError, match='unsafe|restart|Restart|retirement'):
        _install(model, projector)
    old_finalizer()
    _finalize(model)
    assert native_install['events'] == before

def test_first_install_publication_baseexception_rolls_back(native_install):
    (model, projector) = _model()
    original_forward = model.forward
    model.arm_assignment_failure('_rpu_vision_handle_finalizer')
    with pytest.raises(_PublishInterrupted, match='publication failure'):
        _install(model, projector)
    assert native_install['events'] == [('create', 1), ('set_weights', 1), ('set_patch', 1), ('destroy', 1)]
    assert native_install['alive'] == set()
    assert not hasattr(model, '_rpu_vision_handle')
    assert not hasattr(model, '_rpu_cache')
    assert model.forward is original_forward

def test_replacement_publication_baseexception_preserves_old_install(native_install):
    (model, projector) = _model()
    assert _install(model, projector) == 1
    old_forward = model.forward
    old_cache = model._rpu_cache
    old_graph_cache = model._rpu_siglip_graph_cache
    old_finalizer = model._rpu_vision_handle_finalizer
    native_install['events'].clear()
    model.arm_assignment_failure('forward')
    with pytest.raises(_PublishInterrupted, match='publication failure'):
        _install(model, projector)
    assert native_install['events'] == [('create', 2), ('set_weights', 2), ('set_patch', 2), ('destroy', 2)]
    assert native_install['alive'] == {1}
    assert model._rpu_vision_handle == 1
    assert model._rpu_vision_handle_finalizer is old_finalizer
    assert old_finalizer.alive
    assert model._rpu_cache is old_cache
    assert model._rpu_siglip_graph_cache is old_graph_cache
    assert model.forward is old_forward
    _finalize(model)

def test_freeze_baseexception_rolls_back_before_old_retirement(native_install, monkeypatch):
    (model, projector) = _model()
    assert _install(model, projector) == 1
    old_forward = model.forward
    old_cache = model._rpu_cache
    old_finalizer = model._rpu_vision_handle_finalizer
    native_install['events'].clear()
    from rpu_backend.graph import lazy_init_guard

    def fail_freeze(_model):
        raise _PublishInterrupted('synthetic freeze failure')
    monkeypatch.setattr(lazy_init_guard, '_verify_lazy_init', fail_freeze)
    with pytest.raises(_PublishInterrupted, match='freeze failure'):
        _install(model, projector)
    assert native_install['events'] == [('create', 2), ('set_weights', 2), ('set_patch', 2), ('destroy', 2)]
    assert native_install['alive'] == {1}
    assert model._rpu_vision_handle == 1
    assert model._rpu_vision_handle_finalizer is old_finalizer
    assert old_finalizer.alive
    assert model._rpu_cache is old_cache
    assert model.forward is old_forward
    _finalize(model)

def test_successful_replacement_retires_old_finalizer(native_install):
    (model, projector) = _model()
    assert _install(model, projector) == 1
    old_finalizer = model._rpu_vision_handle_finalizer
    old_cache = model._rpu_cache
    native_install['events'].clear()
    assert _install(model, projector) == 2
    assert native_install['events'] == [('create', 2), ('set_weights', 2), ('set_patch', 2), ('destroy', 1)]
    assert native_install['alive'] == {2}
    assert model._rpu_vision_handle == 2
    assert model._rpu_vision_handle_finalizer.alive
    assert not old_finalizer.alive
    assert model._rpu_cache is not old_cache
    _finalize(model)

def test_replacement_does_not_double_destroy_dead_finalizer(native_install):
    (model, projector) = _model()
    assert _install(model, projector) == 1
    model._rpu_vision_handle_finalizer()
    native_install['events'].clear()
    assert _install(model, projector) == 2
    assert native_install['events'] == [('create', 2), ('set_weights', 2), ('set_patch', 2)]
    assert native_install['alive'] == {2}
    assert model._rpu_vision_handle == 2
    assert model._rpu_vision_handle_finalizer.alive
    _finalize(model)

@pytest.mark.parametrize('acc32', (False, True))
def test_vision_accumulation_is_bound_before_weights(native_install, acc32):
    (model, projector) = _model()
    model._fmb_execution_component_config = {'vision': {'linear_acc32': acc32}}
    try:
        handle = siglip.patch_siglip_model_for_rpu_all_layers_once(model, projector, linear_acc32=acc32)
        assert native_install['accumulation'] == {handle: acc32}
        assert native_install['events'][:2] == [('create', handle), ('set_weights', handle)]
    finally:
        _finalize(model)

def test_standalone_vision_preserves_implicit_projector_accumulation(native_install):
    (model, projector) = _model()
    try:
        handle = _install(model, projector)
        assert native_install["accumulation"] == {handle: None}
        assert native_install["cold_options"] == {handle: ()}
    finally:
        _finalize(model)


@pytest.mark.parametrize("num_images", (2, 3))
@pytest.mark.parametrize("acc32", (None, False))
def test_pi_qkv_w8a8_is_bound_before_weights_and_retains_full_envelope(
    native_install, monkeypatch, num_images, acc32,
):
    model, projector = _pi05_model()
    monkeypatch.setattr(siglip, "_detect_and_validate_siglip_encoder_w8a16", lambda *args: True)
    try:
        handle = siglip.patch_siglip_model_for_rpu_all_layers_once(
            model, projector, num_images=num_images, linear_acc32=acc32, qkv_w8a8=True)
        assert native_install["accumulation"] == {handle: acc32}
        assert native_install["cold_options"] == {handle: (True,)}
        assert native_install["events"][:2] == [("create", handle), ("set_weights", handle)]
        assert native_install["envelopes"][handle] == (num_images * 256, 0)
        assert model._rpu_cache.kwargs["max_seq_len"] == num_images * 256
        assert model._siglip_qkv_w8a8 is True
    finally:
        _finalize(model)


def test_failed_w8a8_replacement_preserves_off_install(native_install, monkeypatch):
    model, projector = _pi05_model()
    monkeypatch.setattr(siglip, "_detect_and_validate_siglip_encoder_w8a16", lambda *args: True)
    try:
        assert siglip.patch_siglip_model_for_rpu_all_layers_once(model, projector, num_images=2) == 1
        old_forward, old_cache = model.forward, model._rpu_cache
        old_finalizer = model._rpu_vision_handle_finalizer
        native_install["fail_patch"].add(2)
        with pytest.raises(RuntimeError, match="patch failure"):
            siglip.patch_siglip_model_for_rpu_all_layers_once(
                model, projector, num_images=2, qkv_w8a8=True)
        assert native_install["cold_options"] == {1: (), 2: (True,)}
        assert native_install["alive"] == {1}
        assert model._rpu_vision_handle == 1
        assert model._rpu_vision_handle_finalizer is old_finalizer and old_finalizer.alive
        assert model.forward is old_forward and model._rpu_cache is old_cache
        assert model._siglip_qkv_w8a8 is False
    finally:
        _finalize(model)


@pytest.mark.parametrize("kwargs", (
    {"num_cores": 4}, {"num_cores": 6}, {"num_images": 1}, {"num_images": 4},
    {"linear_acc32": True}, {"qkv_w8a8": 1},
))
def test_pi_qkv_w8a8_invalid_execution_rejects_before_conversion_or_native(
    native_install, monkeypatch, kwargs,
):
    model, projector = _pi05_model()
    converted = []
    monkeypatch.setattr(siglip, "_convert_siglip_weights_for_rpu", lambda *a, **k: converted.append(True))
    options = {"num_images": 2, "qkv_w8a8": True, **kwargs}
    original_forward = model.forward
    with pytest.raises((ValueError, TypeError, RuntimeError), match="qkv_w8a8|QKV W8A8"):
        siglip.patch_siglip_model_for_rpu_all_layers_once(model, projector, **options)
    assert converted == [] and native_install["events"] == []
    assert not hasattr(model, "_rpu_vision_handle") and model.forward is original_forward


@pytest.mark.parametrize("scope,accepted", (("all", True), ("row", False), ("none", False)))
def test_pi_qkv_w8a8_raw_checkpoint_requires_existing_full_w8_conversion(monkeypatch, scope, accepted):
    model, projector = _pi05_model(converted=False)
    monkeypatch.setenv("RPU_PI05_SIGLIP_W8A16", "1")
    monkeypatch.setenv("RPU_PI05_SIGLIP_W8A16_SCOPE", scope)
    kwargs = dict(qkv_w8a8=True, num_images=2, w8a16_default=True)
    if accepted:
        siglip._validate_siglip_w8a8(model, projector, **kwargs)
        # Raw INT8's scale contract is not consumed by the current converter.
        model.encoder.layers[-1].self_attn.v_proj.weight.dtype = torch.int8
        with pytest.raises(ValueError, match="layer26.v_proj.weight"):
            siglip._validate_siglip_w8a8(model, projector, **kwargs)
    else:
        with pytest.raises(ValueError, match="every Q/K/V"):
            siglip._validate_siglip_w8a8(model, projector, **kwargs)


@pytest.mark.parametrize("corruption", (
    "missing_scale", "scale_dtype", "scale_shape", "weight_dtype",
    "missing_bias", "norm_dtype", "eps", "layers",
))
def test_pi_qkv_w8a8_checks_last_layer_before_any_conversion_or_native(
    native_install, monkeypatch, corruption,
):
    model, projector = _pi05_model()
    projection = model.encoder.layers[-1].self_attn.v_proj
    if corruption == "missing_scale":
        del projection.weight_scale
    elif corruption == "scale_dtype":
        projection.weight_scale.dtype = torch.float32
    elif corruption == "scale_shape":
        projection.weight_scale.shape = (1152,)
    elif corruption == "weight_dtype":
        projection.weight.dtype = torch.float16
    elif corruption == "missing_bias":
        projection.bias = None
    elif corruption == "norm_dtype":
        model.encoder.layers[-1].layer_norm1.weight.dtype = torch.float32
    elif corruption == "eps":
        model.config.layer_norm_eps = 1e-5
    else:
        model.encoder.layers.pop()
    converted = []
    monkeypatch.setattr(siglip, "_convert_siglip_weights_for_rpu", lambda *a, **k: converted.append(True))
    original_forward = model.forward
    with pytest.raises(ValueError, match="qkv_w8a8 requires"):
        siglip.patch_siglip_model_for_rpu_all_layers_once(
            model, projector, num_images=2, qkv_w8a8=True)
    assert converted == [] and native_install["events"] == []
    assert model.forward is original_forward and not hasattr(model, "_rpu_vision_handle")


@pytest.fixture
def mlp_profile(monkeypatch):
    monkeypatch.setenv("RPU_PI05_RING_XOR3", "1")
    monkeypatch.setenv("RPU_PI05_VISION_OWNER_LN", "1")
    monkeypatch.setenv("RPU_PI05_VISION_KV_SPM_ONLY", "1")
    monkeypatch.setenv("RPU_PI05_VISION_FC_WEIGHT_OUTER", "0")


@pytest.mark.parametrize("num_images", (2, 3))
@pytest.mark.parametrize("qkv_w8a8", (False, True))
@pytest.mark.parametrize("acc32", (None, False))
def test_pi_mlp_w8a8_installs_independently_with_complete_envelope(
    native_install, monkeypatch, mlp_profile, num_images, qkv_w8a8, acc32,
):
    model, projector = _pi05_model()
    monkeypatch.setattr(siglip, "_detect_and_validate_siglip_encoder_w8a16", lambda *args: True)
    try:
        handle = siglip.patch_siglip_model_for_rpu_all_layers_once(
            model, projector, num_images=num_images, linear_acc32=acc32,
            qkv_w8a8=qkv_w8a8, mlp_w8a8=True)
        assert native_install["cold_options"] == {handle: (qkv_w8a8, True)}
        assert native_install["accumulation"] == {handle: acc32}
        assert native_install["events"][:2] == [("create", handle), ("set_weights", handle)]
        assert native_install["envelopes"][handle] == (num_images * 256, 0)
        assert model._rpu_cache.kwargs["max_seq_len"] == num_images * 256
        assert model._siglip_qkv_w8a8 is qkv_w8a8
        assert model._siglip_mlp_w8a8 is True
    finally:
        _finalize(model)


@pytest.mark.parametrize("qkv_w8a8", (False, True))
@pytest.mark.parametrize("failure", ("patch", "publication"))
def test_failed_mlp_w8a8_replacement_restores_all_cold_identity(
    native_install, monkeypatch, mlp_profile, qkv_w8a8, failure,
):
    model, projector = _pi05_model()
    monkeypatch.setattr(siglip, "_detect_and_validate_siglip_encoder_w8a16", lambda *args: True)
    try:
        assert siglip.patch_siglip_model_for_rpu_all_layers_once(
            model, projector, num_images=2, qkv_w8a8=qkv_w8a8) == 1
        old_forward, old_cache = model.forward, model._rpu_cache
        old_finalizer = model._rpu_vision_handle_finalizer
        if failure == "patch":
            native_install["fail_patch"].add(2)
            error = RuntimeError
        else:
            model.arm_assignment_failure("_siglip_mlp_w8a8")
            error = _PublishInterrupted
        with pytest.raises(error):
            siglip.patch_siglip_model_for_rpu_all_layers_once(
                model, projector, num_images=2, qkv_w8a8=qkv_w8a8, mlp_w8a8=True)
        assert native_install["cold_options"] == {1: (True,) if qkv_w8a8 else (), 2: (qkv_w8a8, True)}
        assert native_install["alive"] == {1}
        assert model._rpu_vision_handle == 1
        assert model._rpu_vision_handle_finalizer is old_finalizer and old_finalizer.alive
        assert model.forward is old_forward and model._rpu_cache is old_cache
        assert model._siglip_qkv_w8a8 is qkv_w8a8
        assert model._siglip_mlp_w8a8 is False
    finally:
        _finalize(model)


@pytest.mark.parametrize("kwargs", (
    {"num_cores": 4}, {"num_cores": 6}, {"num_images": 1}, {"num_images": 4},
    {"linear_acc32": True}, {"mlp_w8a8": 1},
))
def test_pi_mlp_w8a8_invalid_execution_precedes_weight_mutation(
    native_install, monkeypatch, mlp_profile, kwargs,
):
    model, projector = _pi05_model()
    converted = []
    monkeypatch.setattr(siglip, "_convert_siglip_weights_for_rpu", lambda *a, **k: converted.append(True))
    options = {"num_images": 2, "mlp_w8a8": True, **kwargs}
    original_forward = model.forward
    with pytest.raises((ValueError, TypeError), match="mlp_w8a8"):
        siglip.patch_siglip_model_for_rpu_all_layers_once(model, projector, **options)
    assert converted == [] and native_install["events"] == []
    assert model.forward is original_forward and not hasattr(model, "_rpu_vision_handle")


@pytest.mark.parametrize("flag,value", (
    ("RPU_PI05_RING_XOR3", None),
    ("RPU_PI05_VISION_OWNER_LN", "0"),
    ("RPU_PI05_VISION_OWNER_LN", None),
    ("RPU_PI05_VISION_KV_SPM_ONLY", "0"),
    ("RPU_PI05_VISION_FC_WEIGHT_OUTER", "1"),
))
def test_pi_mlp_w8a8_requires_matching_native_cold_profile_before_conversion(
    native_install, monkeypatch, mlp_profile, flag, value,
):
    model, projector = _pi05_model()
    if value is None:
        monkeypatch.delenv(flag)
    else:
        monkeypatch.setenv(flag, value)
    converted = []
    monkeypatch.setattr(siglip, "_convert_siglip_weights_for_rpu", lambda *a, **k: converted.append(True))
    with pytest.raises(ValueError, match=flag):
        siglip.patch_siglip_model_for_rpu_all_layers_once(
            model, projector, num_images=2, mlp_w8a8=True)
    assert converted == [] and native_install["events"] == []


@pytest.mark.parametrize("scope", ("all", "row", "fc2", "none"))
def test_pi_mlp_w8a8_raw_checkpoint_admits_only_full_existing_conversion(
    monkeypatch, mlp_profile, scope,
):
    model, projector = _pi05_model(converted=False)
    monkeypatch.setenv("RPU_PI05_SIGLIP_W8A16", "1")
    monkeypatch.setenv("RPU_PI05_SIGLIP_W8A16_SCOPE", scope)
    kwargs = dict(mlp_w8a8=True, num_images=2, w8a16_default=True)
    if scope == "all":
        siglip._validate_siglip_w8a8(model, projector, **kwargs)
        model.encoder.layers[-1].mlp.fc2.weight.dtype = torch.int8
        with pytest.raises(ValueError, match="layer26.fc2.weight"):
            siglip._validate_siglip_w8a8(model, projector, **kwargs)
    else:
        with pytest.raises(ValueError, match="FC1/FC2"):
            siglip._validate_siglip_w8a8(model, projector, **kwargs)


@pytest.mark.parametrize("projection_name", ("fc1", "fc2"))
@pytest.mark.parametrize("corruption", (
    "missing_scale", "scale_dtype", "scale_shape", "weight_dtype", "weight_shape",
    "missing_bias", "bias_shape", "norm_dtype",
))
def test_pi_mlp_w8a8_checks_late_fc_payload_before_any_conversion(
    native_install, monkeypatch, mlp_profile, projection_name, corruption,
):
    model, projector = _pi05_model()
    projection = getattr(model.encoder.layers[-1].mlp, projection_name)
    if corruption == "missing_scale":
        del projection.weight_scale
    elif corruption == "scale_dtype":
        projection.weight_scale.dtype = torch.float32
    elif corruption == "scale_shape":
        projection.weight_scale.shape = (projection.out_features + 16,)
    elif corruption == "weight_dtype":
        projection.weight.dtype = torch.float16
    elif corruption == "weight_shape":
        projection.weight.shape = (projection.weight.shape[0], projection.weight.shape[1] + 16)
    elif corruption == "missing_bias":
        projection.bias = None
    elif corruption == "bias_shape":
        projection.bias.shape = (projection.out_features + 16,)
    else:
        model.encoder.layers[-1].layer_norm2.weight.dtype = torch.float32
    converted = []
    monkeypatch.setattr(siglip, "_convert_siglip_weights_for_rpu", lambda *a, **k: converted.append(True))
    with pytest.raises(ValueError, match="mlp_w8a8 requires"):
        siglip.patch_siglip_model_for_rpu_all_layers_once(
            model, projector, num_images=2, mlp_w8a8=True)
    assert converted == [] and native_install["events"] == []
    assert not hasattr(model, "_rpu_vision_handle")


@pytest.mark.parametrize("projection_name", ("q_proj", "out_proj"))
@pytest.mark.parametrize("corruption", ("weight_dtype", "missing_scale"))
def test_mlp_only_rejects_mixed_owner_attention_payload_before_conversion(
    native_install, monkeypatch, mlp_profile, projection_name, corruption,
):
    model, projector = _pi05_model()
    projection = getattr(model.encoder.layers[-1].self_attn, projection_name)
    if corruption == "weight_dtype":
        projection.weight.dtype = torch.float16
    else:
        del projection.weight_scale
    converted = []
    monkeypatch.setattr(siglip, "_convert_siglip_weights_for_rpu", lambda *a, **k: converted.append(True))
    with pytest.raises(ValueError, match=f"layer26.{projection_name}"):
        siglip.patch_siglip_model_for_rpu_all_layers_once(
            model, projector, num_images=2, mlp_w8a8=True)
    assert converted == [] and native_install["events"] == []
