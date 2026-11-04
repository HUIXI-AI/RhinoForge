"""Board-free transaction contracts for the generic decoder installer."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from rpu_backend.runtime.chunk_envelope import ChunkEnvelope, make_lookup

# This suite installs a SYNTHETIC 1-layer/128-hidden model. Deny-by-default is
# exactly what stops an unmeasured geometry reaching the SPM planner, so the
# fixture brings its own one-row table rather than widening a production one.
lookup_causal_decoder = make_lookup(
    {
        ("llama", 1, 128): ChunkEnvelope(128),
        ("qwen3", 1, 128): ChunkEnvelope(128),
        ("qwen3", 28, 1024): ChunkEnvelope(128),
        ("qwen3", 28, 2048): ChunkEnvelope(128),
        ("qwen3", 36, 2560): ChunkEnvelope(128),
        ("qwen3", 36, 4096): ChunkEnvelope(128),
    },
    "<test fixture>",
)

import rpu_backend
from rpu_backend.api._execution import normalize_rpu_execution
from rpu_backend.runtime import decoder


class _FakeTensor:
    def __init__(self, capacity=128):
        self.shape = (capacity, 8)

    def size(self, dim):
        return self.shape[dim]

    def to(self, *, dtype, device):
        if dtype is not torch.float16 or device != "rpu":
            raise AssertionError(f"unexpected fake tensor conversion: {dtype}/{device}")
        return self

    def contiguous(self):
        return self


class _Model:
    def __setattr__(self, name, value):
        if name == self.__dict__.get("_fail_publish_attr"):
            self.__dict__["_fail_publish_attr"] = None
            raise _PublishInterrupted(
                f"synthetic publication interruption for {name}"
            )
        super().__setattr__(name, value)


class _PublishInterrupted(BaseException):
    pass


class _FakeWeight(str):
    dtype = torch.float16


def _projection(*, weight, out_features):
    return SimpleNamespace(weight=_FakeWeight(weight), out_features=out_features)


def _model(*, model_type="llama"):
    attn = SimpleNamespace(
        head_dim=16,
        q_proj=_projection(weight="q", out_features=128),
        k_proj=_projection(weight="k", out_features=64),
        v_proj=_projection(weight="v", out_features=64),
        o_proj=_projection(weight="o", out_features=128),
    )
    if model_type == "qwen3":
        attn.q_norm = SimpleNamespace(weight="q_norm")
        attn.k_norm = SimpleNamespace(weight="k_norm")
    mlp = SimpleNamespace(
        gate_proj=_projection(weight="gate", out_features=256),
        up_proj=_projection(weight="up", out_features=256),
        down_proj=_projection(weight="down", out_features=128),
    )
    layer = SimpleNamespace(
        self_attn=attn,
        mlp=mlp,
        input_layernorm=SimpleNamespace(weight="input_norm", variance_epsilon=1e-6),
        post_attention_layernorm=SimpleNamespace(weight="post_norm"),
    )
    model = _Model()
    model.layers = [layer]
    model.norm = SimpleNamespace(weight="final_norm")
    model.config = SimpleNamespace(model_type=model_type, hidden_size=128)
    model.forward = lambda *args, **kwargs: "original"
    return model


@pytest.fixture
def native_install(monkeypatch):
    import gc
    gc.collect()
    from rpu_backend.api import _execution, causal_lm
    from rpu_backend.runtime import _native_retirement
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(_native_retirement, "_FAILED_RETIREMENTS", [])
    state = {
        "next_handle": 1,
        "alive": set(),
        "events": [],
        "create_options": [],
        "fail_set": set(),
        "fail_destroy": set(),
    }

    def create(*_cold_options, **kwargs):
        if kwargs:
            state.setdefault("create_topologies", []).append(kwargs)
        state["create_options"].append(_cold_options)
        handle = state["next_handle"]
        state["next_handle"] += 1
        state["alive"].add(handle)
        state["events"].append(("create", handle))
        return handle

    def set_weights(handle, *args):
        state["events"].append(("set_weights", handle))
        if handle in state["fail_set"]:
            raise RuntimeError(f"synthetic set_weights failure for {handle}")

    def destroy(handle):
        state["events"].append(("destroy", handle))
        if handle in state["fail_destroy"]:
            raise RuntimeError(f"synthetic destroy failure for {handle}")
        if handle not in state["alive"]:
            raise RuntimeError(f"double destroy for {handle}")
        state["alive"].remove(handle)

    class GraphCache:
        def __init__(self, *, runtime_policy=None):
            self.runtime_policy = runtime_policy

        def clear(self):
            pass

        def cache_invariant_ok(self):
            return True

    class GraphSignature:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        decoder.torch.ops.rpu, "causal_decoder_create", create, raising=False
    )
    monkeypatch.setattr(
        decoder.torch.ops.rpu,
        "causal_decoder_set_weights",
        set_weights,
        raising=False,
    )
    # MR-A: _install_causal_decoder_forward declares the certified chunk
    # envelope on the NEW handle before publication. Recorded so a future
    # regression that declares it on the OLD handle, or after publication,
    # is visible in the same event log the rollback assertions read.
    monkeypatch.setattr(
        decoder.torch.ops.rpu, "causal_decoder_set_chunk_envelope",
        lambda handle, *_: state.setdefault("envelope", []).append(handle),
        raising=False,
    )
    monkeypatch.setattr(
        decoder.torch.ops.rpu,
        "causal_decoder_set_chunk_size_override",
        lambda handle, chunk: state.setdefault("chunk_overrides", []).append(
            (handle, chunk)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        decoder.torch.ops.rpu, "causal_decoder_set_linear_acc32",
        lambda handle, enabled: state.setdefault("linear_acc32", []).append(
            (handle, enabled)), raising=False,
    )
    monkeypatch.setattr(
        decoder.torch.ops.rpu,
        "causal_decoder_resolve_decode_stage_descriptor",
        lambda handle: (3, handle, 1),
        raising=False,
    )
    monkeypatch.setattr(
        decoder.torch.ops.rpu, "causal_decoder_destroy", destroy, raising=False
    )
    monkeypatch.setattr(rpu_backend.graph, "GraphCache", GraphCache)
    monkeypatch.setattr(rpu_backend.graph, "GraphSignature", GraphSignature)
    return state


def _install(model, *, arch="llama", execution_config=None, topology=None):
    table = _FakeTensor()
    return decoder._install_causal_decoder_forward(
        model,
        arch=arch,
        cos_sin=(table, table),
        chunk_envelope_for=lookup_causal_decoder,
        execution_config=execution_config,
        **({"topology": topology} if topology is not None else {}),
    )


def _finalize(model):
    finalizer = getattr(model, "_rpu_decoder_handle_finalizer", None)
    if finalizer is not None and finalizer.alive:
        finalizer()


@pytest.mark.parametrize("fail_destroy", (False, True))
def test_outer_cleanup_retires_receipt_only_after_native_success(native_install, fail_destroy):
    model = _model()
    original_forward = model.forward
    handle = _install(model)
    descriptor = model._rpu_decode_stage_descriptor
    receipt = {"generation": 7, "descriptor": [13, 17]}
    model._rpu_last_execution_plan = receipt
    resource = model._rpu_decoder_retirement_state
    if fail_destroy:
        native_install["fail_destroy"].add(handle)
        with pytest.raises(RuntimeError, match="synthetic destroy failure"):
            decoder._cleanup_causal_decoder_install(model, had_instance_forward=True,
                original_instance_forward=original_forward)
        assert model._rpu_last_execution_plan is receipt
        assert model._rpu_decode_stage_descriptor is descriptor
        assert model._rpu_decoder_handle == handle
        assert model._rpu_decoder_retirement_state is resource
        assert resource.handle == handle
        assert native_install["alive"] == {handle}
        assert native_install["events"].count(("destroy", handle)) == 1
    else:
        decoder._cleanup_causal_decoder_install(model, had_instance_forward=True,
            original_instance_forward=original_forward)
        assert "_rpu_last_execution_plan" not in vars(model)
        assert "_rpu_decode_stage_descriptor" not in vars(model)
        assert "_rpu_decoder_handle" not in vars(model)
        assert model.forward is original_forward
        assert not native_install["alive"]
@pytest.mark.parametrize("cores", [4, 6, 8])
@pytest.mark.parametrize("hidden, intermediate, layers, heads", [
    (1024, 3072, 28, 16), (2048, 6144, 28, 16),
    (2560, 9728, 36, 32), (4096, 12288, 36, 32),
])
@pytest.mark.parametrize("fault", [None, "native", "weights"])
def test_fixed_core_install_binds_weight_layout_native_handle_and_published_topology(native_install, monkeypatch, fault, cores, hidden, intermediate, layers, heads):
    from rpu_backend.runtime.topology import resolve_decoder_topology
    model = _model(model_type="qwen3")
    # The real 0.6B vocabulary chooses TP4, unlike the native allocation
    # ceiling 200000 (TP5). A handle must use its cold model configuration.
    model.config = SimpleNamespace(model_type="qwen3", vocab_size=151936,
        hidden_size=hidden, intermediate_size=intermediate, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=8, head_dim=128)
    layer = model.layers[0]
    layer.self_attn.head_dim = 128
    for name, outputs in (("q_proj", heads * 128), ("k_proj", 1024), ("v_proj", 1024), ("o_proj", hidden)):
        getattr(layer.self_attn, name).out_features = outputs
    for name, outputs in (("gate_proj", intermediate), ("up_proj", intermediate), ("down_proj", hidden)):
        getattr(layer.mlp, name).out_features = outputs
    model.layers = [layer] * layers
    topology = resolve_decoder_topology(num_cores=cores, hidden_size=hidden,
        intermediate_size=intermediate, num_q_heads=heads, num_kv_heads=8, head_dim=128,
        vocab_size=model.config.vocab_size)
    if hidden == 2560 and cores == 6:
        for name, shape in (("gate_proj", (9792, hidden)), ("up_proj", (9792, hidden)),
                            ("down_proj", (hidden, 9792))):
            projection = getattr(layer.mlp, name)
            projection.out_features, projection.in_features = shape
            projection.weight = torch.empty(shape, dtype=torch.float16, device="meta")
    for owner, names, role_cores in (
        (model.layers[0].self_attn, ("q_proj", "k_proj", "v_proj", "o_proj"), topology.attn_tp),
        (model.layers[0].mlp, ("gate_proj", "up_proj", "down_proj"), topology.mlp_tp),
    ):
        for name in names:
            projection = getattr(owner, name)
            projection._rpu_linear_num_cores = role_cores
            projection._rpu_linear_partition = 0 if name in {"o_proj", "down_proj"} else 1
    if fault == "weights":
        model.layers[0].mlp.down_proj._rpu_linear_num_cores = 1
    def native_topology(_handle):
        cold = native_install["create_topologies"][-1]
        assert cold == {"num_cores": cores, "vocab_size": 151936}
        readback = list(topology.identity()[1:])
        if fault == "native":
            if hidden in (2560, 4096) and cores == 6:
                readback[2] = 4  # Reject the previously installed MLP4 template.
            else:
                readback[3] = 5  # A stale head layout must retire the tentative handle.
        return readback

    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_topology", native_topology, raising=False)
    if fault:
        with pytest.raises((ValueError, RuntimeError), match="layout differs|topology differs") as error:
            _install(model, arch="qwen3", execution_config={"model": {"num_cores": cores}}, topology=topology)
        assert not hasattr(model, "_rpu_decoder_handle") and not native_install["alive"]
        if fault == "weights":
            assert native_install["events"] == []
        else:
            assert native_install["events"][-1] == ("destroy", 1)
            assert f"expected={topology.identity()[1:]}" in str(error.value)
    else:
        assert _install(model, arch="qwen3", execution_config={"model": {"num_cores": cores}}, topology=topology) == 1
        assert native_install["create_topologies"] == [{"num_cores": cores, "vocab_size": 151936}]
        assert model._rpu_decoder_topology is topology
        assert model._rpu_batch_decode_enabled is (cores == 8)
        assert model._rpu_decoder_graph_cache.runtime_policy.execution_core_count == cores
        _finalize(model)


def test_rotary_table_capacity_uses_new_hf_qwen3_fields():
    rotary = SimpleNamespace(
        max_seq_len_cached=40960,
        config=SimpleNamespace(max_position_embeddings=40960),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=40960),
    )

    assert decoder._rotary_table_capacity(rotary, model) == 40960


def test_rotary_table_capacity_never_truncates_a_larger_config():
    rotary = SimpleNamespace(
        max_position_embeddings=4096,
        max_seq_len_cached=8192,
        config=SimpleNamespace(max_position_embeddings=16384),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=32768),
    )

    assert decoder._rotary_table_capacity(rotary, model) == 32768


def test_install_translates_legacy_padding_default_without_mutating_input(
    native_install,
):
    model = _model()
    execution = normalize_rpu_execution(None, entry_point="test")

    _install(model, execution_config=execution)

    assert execution == {}
    assert model._rpu_execution == {
        "prefill": {"padding_budget": 64}
    }
    assert model._rpu_prefill_execution_alignment == 1
    _finalize(model)


def test_batch_decode_capability_is_published_only_for_qwen3(native_install):
    llama = _model()
    assert _install(llama) == 1
    assert llama._rpu_batch_decode_enabled is False
    _finalize(llama)

    qwen3 = _model(model_type="qwen3")
    assert _install(qwen3, arch="qwen3") == 2
    assert qwen3._rpu_batch_decode_enabled is True
    _finalize(qwen3)


def test_prepare_failure_does_not_create_or_publish_handle(native_install):
    model = _model()
    del model.layers[0].mlp.down_proj
    original_forward = model.forward

    with pytest.raises(AttributeError, match="down_proj"):
        _install(model)

    assert native_install["events"] == []
    assert not hasattr(model, "_rpu_decoder_handle")
    assert model.forward is original_forward


def test_set_weights_failure_destroys_pending_handle_immediately(native_install):
    model = _model()
    original_forward = model.forward
    native_install["fail_set"].add(1)

    with pytest.raises(RuntimeError, match="set_weights failure"):
        _install(model)

    assert native_install["events"] == [
        ("create", 1),
        ("set_weights", 1),
        ("destroy", 1),
    ]
    assert native_install["alive"] == set()
    assert not hasattr(model, "_rpu_decoder_handle")
    assert not hasattr(model, "_rpu_decoder_graph_cache")
    assert model.forward is original_forward


def test_first_install_publication_base_exception_rolls_back(native_install):
    model = _model()
    original_forward = model.forward
    model._fail_publish_attr = "_rpu_decoder_handle_finalizer"

    with pytest.raises(_PublishInterrupted, match="publication interruption"):
        _install(model)

    assert native_install["events"] == [
        ("create", 1),
        ("set_weights", 1),
        ("destroy", 1),
    ]
    assert native_install["alive"] == set()
    for attr in decoder._CAUSAL_DECODER_INSTALL_ATTRS:
        assert not hasattr(model, attr)
    assert model.forward is original_forward


def test_failed_replacement_preserves_published_install(native_install):
    model = _model()
    assert _install(model) == 1
    old_forward = model.forward
    old_cache = model._rpu_decoder_graph_cache
    old_finalizer = model._rpu_decoder_handle_finalizer
    native_install["events"].clear()
    native_install["fail_set"].add(2)

    with pytest.raises(RuntimeError, match="set_weights failure"):
        _install(model)

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
        ("destroy", 2),
    ]
    assert native_install["alive"] == {1}
    assert model._rpu_decoder_handle == 1
    assert model._rpu_decoder_handle_finalizer is old_finalizer
    assert old_finalizer.alive
    assert model._rpu_decoder_graph_cache is old_cache
    assert model.forward is old_forward
    _finalize(model)


def test_replacement_publication_base_exception_preserves_old_install(
    native_install,
):
    model = _model()
    assert _install(model) == 1
    old_descriptor = model._rpu_decode_stage_descriptor
    assert old_descriptor == (3, 1, 1)
    old_forward = model.forward
    old_receipt = {"generation": 7, "physical_descriptor": (13, 17)}
    model._rpu_last_execution_plan = old_receipt
    old_cache = model._rpu_decoder_graph_cache
    old_finalizer = model._rpu_decoder_handle_finalizer
    native_install["events"].clear()
    model._fail_publish_attr = "_rpu_decoder_handle_finalizer"

    with pytest.raises(_PublishInterrupted, match="publication interruption"):
        _install(model)

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
        ("destroy", 2),
    ]
    assert native_install["alive"] == {1}
    assert model._rpu_decoder_handle == 1
    assert model._rpu_decoder_handle_finalizer is old_finalizer
    assert old_finalizer.alive
    assert model._rpu_decoder_graph_cache is old_cache
    assert model.forward is old_forward
    assert model._rpu_decode_stage_descriptor is old_descriptor
    assert model._rpu_last_execution_plan is old_receipt
    _finalize(model)


def test_old_destroy_failure_aborts_replacement_without_publishing_it(native_install):
    model = _model()
    assert _install(model) == 1
    old_forward = model.forward
    old_cache = model._rpu_decoder_graph_cache
    old_finalizer = model._rpu_decoder_handle_finalizer
    native_install["events"].clear()
    native_install["fail_destroy"].add(1)

    with pytest.raises(RuntimeError, match="destroy failure"):
        _install(model)

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
        ("destroy", 1),
    ]
    assert native_install["alive"] == {1, 2}
    assert model._rpu_decoder_handle == 1
    assert model._rpu_decoder_handle_finalizer is old_finalizer
    assert old_finalizer.alive
    assert model._rpu_decoder_graph_cache is old_cache
    assert model.forward is old_forward
    assert model._rpu_decoder_retirement_state.failed is not None


def test_successful_replacement_retires_handle_and_its_finalizer(native_install):
    model = _model()
    assert _install(model) == 1
    assert model._rpu_decode_stage_descriptor == (3, 1, 1)
    old_finalizer = model._rpu_decoder_handle_finalizer
    old_cache = model._rpu_decoder_graph_cache
    native_install["events"].clear()

    assert _install(model) == 2

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
        ("destroy", 1),
    ]
    assert native_install["alive"] == {2}
    assert model._rpu_decoder_handle == 2
    assert model._rpu_decode_stage_descriptor == (3, 2, 1)
    assert model._rpu_decoder_handle_finalizer.alive
    assert not old_finalizer.alive
    assert model._rpu_decoder_graph_cache is not old_cache
    _finalize(model)
    assert native_install["alive"] == set()


@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("failure", ["empty", "getter"])
def test_decode_descriptor_failure_preserves_original_install(native_install, monkeypatch, replacement, failure):
    model = _model()
    if replacement:
        assert _install(model) == 1
    absent = object()
    before = {name: getattr(model, name, absent)
              for name in decoder._CAUSAL_DECODER_INSTALL_ATTRS}
    old_forward = model.forward
    pending = 2 if replacement else 1
    native_install["events"].clear()
    def reject(handle):
        assert handle == pending
        # The old installation remains visible until the new getter succeeds.
        assert all(getattr(model, name, absent) is value for name, value in before.items())
        if failure == "getter":
            raise RuntimeError("native decode descriptor failure")
        return ()
    monkeypatch.setattr(decoder.torch.ops.rpu, "causal_decoder_resolve_decode_stage_descriptor", reject)
    with pytest.raises(RuntimeError, match="decode.*(empty descriptor|failure)"):
        _install(model)
    assert native_install["events"] == [
        ("create", pending), ("set_weights", pending), ("destroy", pending)]
    assert native_install["alive"] == ({1} if replacement else set())
    assert all(getattr(model, name, absent) is value for name, value in before.items())
    assert model.forward is old_forward
    _finalize(model)
    assert native_install["alive"] == set()


def test_replacement_does_not_double_destroy_dead_finalizer(native_install):
    model = _model()
    assert _install(model) == 1
    model._rpu_decoder_handle_finalizer()
    native_install["events"].clear()

    assert _install(model) == 2

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
    ]
    assert native_install["alive"] == {2}
    assert model._rpu_decoder_handle == 2
    assert model._rpu_decoder_handle_finalizer.alive
    _finalize(model)


def test_private_reduced_install_rejects_wrong_model_before_native_create(native_install):
    from rpu_backend.runtime.topology import resolve_decoder_topology
    topology = resolve_decoder_topology(num_cores=6, hidden_size=1024,
        intermediate_size=3072, num_q_heads=16, num_kv_heads=8, head_dim=128,
        vocab_size=151936)
    model = _model(model_type="qwen3")
    with pytest.raises(ValueError, match="plain FP16 Qwen3"):
        _install(model, arch="qwen3", execution_config={"model": {"num_cores": 6}}, topology=topology)
    assert native_install["events"] == []
    assert not hasattr(model, "_rpu_decoder_handle")


@pytest.mark.parametrize("hidden, intermediate", [(2560, 9728), (4096, 12288)])
def test_six_producer_reduced_install_rejects_legal_but_wrong_mlp_template_before_create(native_install, hidden, intermediate):
    from rpu_backend.runtime.topology import DecoderTopology
    model = _model(model_type="qwen3")
    model.config = SimpleNamespace(model_type="qwen3", vocab_size=151936,
        hidden_size=hidden, intermediate_size=intermediate, num_hidden_layers=36,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128)
    model.layers = model.layers * 36
    wrong = DecoderTopology(6, 4, 4, 4)
    with pytest.raises(ValueError, match="bound cold topology"):
        _install(model, arch="qwen3", execution_config={"model": {"num_cores": 6}}, topology=wrong)
    assert native_install["events"] == [] and not native_install["alive"]


@pytest.mark.parametrize("arch", ["llama", "qwen3"])
@pytest.mark.parametrize("capacity", [8193, 16384, 32768])
def test_install_binds_capacity_to_rotary_tables_not_profile_history(native_install, monkeypatch, arch, capacity):
    model = _model(model_type=arch)
    table = _FakeTensor(capacity)
    envelopes = []
    monkeypatch.setattr(decoder.torch.ops.rpu, "causal_decoder_set_chunk_envelope",
                        lambda handle, rows, chunk: envelopes.append((rows, chunk)), raising=False)
    decoder._install_causal_decoder_forward(
        model, arch=arch, cos_sin=(table, table),
        chunk_envelope_for=lookup_causal_decoder)
    assert envelopes == [(capacity, 0)]
    _finalize(model)


@pytest.mark.parametrize("capacity", [8193, 16384, 32768])
def test_explicit_rotary_capacity_overrides_checkpoint_default(capacity):
    rotary = SimpleNamespace(max_seq_len_cached=4096)
    model = SimpleNamespace(_rpu_max_seq_len=capacity,
                            config=SimpleNamespace(max_position_embeddings=131072))
    assert decoder._rotary_table_capacity(rotary, model) == capacity


@pytest.mark.parametrize("capacity", [None, 16384, 32768])
@pytest.mark.parametrize("position_horizon", [4096, 262144])
def test_mrope_capacity_is_independent_of_rotary_horizon(monkeypatch, capacity, position_horizon):
    """A long spatial table must not implicitly enlarge KV/DeepStack buffers."""
    model = SimpleNamespace(config=SimpleNamespace(
        rope_parameters={"mrope_section": [24, 20, 20]},
        max_position_embeddings=position_horizon,
        num_hidden_layers=28, hidden_size=2048))
    seen = {}
    monkeypatch.setattr(decoder, "_validate_mrope_decoder_topology", lambda *a, **k: None)
    monkeypatch.setattr(decoder, "_new_decoder_graph_cache", lambda *a, **k: object())
    def tables(config, **kwargs):
        seen["rotary_rows"] = kwargs["max_seq_len"]
        return object(), object()
    def install(owner, **kwargs):
        seen["kv_capacity"] = kwargs["_kv_capacity"]
        return 71
    monkeypatch.setattr(decoder, "build_mrope_cos_sin_tables", tables)
    monkeypatch.setattr(decoder, "_install_causal_decoder_forward", install)
    decoder.install_mrope_text_decoder_for_rpu(
        model, arch="qwen3_vl_text", chunk_envelope_for=lambda *a: (0, 256),
        max_seq_len=capacity)
    expected = 8192 if capacity is None else capacity
    assert seen == {"rotary_rows": max(position_horizon, expected), "kv_capacity": expected}
    assert model._rpu_mrope_capacity == expected


def test_capacity_is_bound_before_weights_allocate_native_keepalives(native_install, monkeypatch):
    events = []
    monkeypatch.setattr(decoder.torch.ops.rpu, "causal_decoder_set_chunk_envelope",
                        lambda handle, rows, chunk: events.append(("capacity", rows)), raising=False)
    monkeypatch.setattr(decoder.torch.ops.rpu, "causal_decoder_set_weights",
                        lambda *args: events.append(("weights",)), raising=False)
    model = _model()
    table = _FakeTensor(32768)
    decoder._install_causal_decoder_forward(
        model, arch="llama", cos_sin=(table, table), _kv_capacity=16384,
        chunk_envelope_for=lookup_causal_decoder)
    assert events == [("capacity", 16384), ("weights",)]
    model._rpu_decoder_handle_finalizer()
