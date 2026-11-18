"""Public cold construction uses real Sessions/builders; device work is doubled."""
import threading
import sys
import weakref
from types import SimpleNamespace

import pytest

from rpu_backend.adapters.lingbot_vla_v2 import runtime
from rpu_backend.api import _execution, lingbot2
from test_graph_cache_freeze import _cache
from test_lingbot2_strict_retirement import _owner, isolated_process


@pytest.mark.parametrize("theta", [10000.0, 20000.0])
def test_real_hf_derived_buffers_restore_the_original_configuration(theta):
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionConfig, Qwen3VLVisionRotaryEmbedding,
        Qwen3VLTextConfig, Qwen3VLTextRotaryEmbedding,
    )
    from rpu_backend.runtime.vision_position import canonical_vision_type

    vision_config = Qwen3VLVisionConfig(hidden_size=1024, num_heads=16,
        rope_parameters={"rope_type": "axial", "rope_theta": theta})
    text_config = Qwen3VLTextConfig(hidden_size=2560, num_attention_heads=32,
        num_key_value_heads=8, head_dim=128,
        rope_parameters={"rope_type": "default", "rope_theta": 5000000.0,
                         "mrope_section": [24, 20, 20]})

    def construct():
        root = torch.nn.Module()
        if hasattr(Qwen3VLVisionRotaryEmbedding, "compute_axial_rope_parameters"):
            root.visual = Qwen3VLVisionRotaryEmbedding(vision_config)
        else:
            root.visual = Qwen3VLVisionRotaryEmbedding(32, theta)
        root.text = Qwen3VLTextRotaryEmbedding(text_config)
        return root

    expected = construct()
    with torch.device("meta"):
        actual = construct()
        actual.weights = torch.nn.Linear(2, 2)
    original_names = set(dict(actual.named_buffers()))
    fixed = runtime._materialize_derived_buffers(actual)
    assert len(fixed) == len(original_names)
    assert set(dict(actual.named_buffers())) == original_names
    for name, value in actual.named_buffers():
        assert value.device.type == "cpu" and torch.isfinite(value).all()
        torch.testing.assert_close(value, expected.get_buffer(name), rtol=0, atol=0)
    assert all(parameter.is_meta for parameter in actual.parameters())
    assert actual.visual._non_persistent_buffers_set == expected.visual._non_persistent_buffers_set
    assert actual.text._non_persistent_buffers_set == expected.text._non_persistent_buffers_set
    # Restoring a config faithfully does not admit a new native RoPE profile.
    assert (canonical_vision_type(vision_config, "qwen3_vl") is not None) == (theta == 10000.0)


def test_derived_buffers_reject_unknown_meta_owners():
    import torch

    owner = torch.nn.Module()
    owner.register_buffer("unknown", torch.empty(2, device="meta"))
    with pytest.raises(RuntimeError, match="not a known config-derived module"):
        runtime._materialize_derived_buffers(owner)
    assert owner.unknown.is_meta


def test_checkpoint_sources_release_before_precision_and_device_conversion(monkeypatch, tmp_path):
    """The FP32 source dictionary must not pin a second backbone during upload."""
    import torch

    refs = []

    def getter(key):
        value = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        refs.append(weakref.ref(value))
        return value

    class StopBeforeMaterialization(Exception):
        pass

    def check_loaded(model):
        assert len(refs) == 1 and refs[0]() is None, "checkpoint source still retained"
        assert torch.equal(model.weight, torch.arange(4, dtype=torch.float32).reshape(2, 2))
        raise StopBeforeMaterialization

    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_vl.configuration_qwen3_vl",
                        SimpleNamespace(Qwen3VLConfig=SimpleNamespace(from_pretrained=lambda _: object())))
    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_vl.modeling_qwen3_vl",
                        SimpleNamespace(Qwen3VLModel=lambda _: torch.nn.Linear(2, 2, bias=False)))
    monkeypatch.setattr(runtime, "_validate_denoise_unroll", lambda: None)
    monkeypatch.setattr(runtime, "_validate_adarms_direct_schedule", lambda: None)
    monkeypatch.setattr(runtime, "resolve_lingbot2_execution_components", lambda *a, **k: ({}, {}, {}, {}))
    monkeypatch.setattr(torch.rpu, "set_ddr_flush", lambda _: None, raising=False)
    monkeypatch.setattr(runtime, "ckpt_reader", lambda _: (getter, {runtime.VLM_P + "weight": None}))
    monkeypatch.setattr(runtime, "_materialize_derived_buffers", check_loaded)
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(StopBeforeMaterialization):
        runtime._build_lingbot_vla_v2_impl(SimpleNamespace(), "unused", qwen3vl_base=str(tmp_path))


def _facade():
    policy = lingbot2.Lingbot2Policy.from_checkpoint("unused", qwen3vl_base="/models/processor")
    policy._preflight_processor_contract = lambda: None
    return policy


@pytest.mark.parametrize("entry", ["facade", "runtime"])
@pytest.mark.parametrize("opt_in", [None, "0", "true", "01"])
def test_generic_evaluation_guard_rejects_before_any_materialization(monkeypatch, entry, opt_in):
    """Public FP16/W8 templates are supported: no opt-in guard precedes
    materialization (name kept for the sensitive-attribute census)."""
    monkeypatch.delenv("RPU_LINGBOT2_MULTIVIEW_SPM_Z2", raising=False)
    monkeypatch.delenv("RPU_LINGBOT2_ALLOW_UNVALIDATED", raising=False)
    if opt_in is not None:
        monkeypatch.setenv("RPU_LINGBOT2_ALLOW_UNVALIDATED", opt_in)
    entered = []

    class Reached(Exception):
        pass

    def materialize(*args, **kwargs):
        entered.append("materialize")
        raise Reached()
    monkeypatch.setattr(runtime, "_build_lingbot_vla_v2_impl", materialize)
    policy = None
    if entry == "facade":
        policy = lingbot2.Lingbot2Policy.from_checkpoint("must-not-be-read", qwen3vl_base="/models/processor")
        policy._preflight_processor_contract = lambda: None
        invoke = lambda: policy.to("rpu")
    else:
        invoke = lambda: runtime.build_lingbot_vla_v2("must-not-be-read")
    with pytest.raises(Exception) as caught:
        invoke()
    assert entered == ["materialize"], caught.value
    if policy is not None:
        policy.close()


def _builder(monkeypatch, facade, *, fail=None, entered=None, proceed=None, retirement_fault=None):
    built = []

    def build(policy, *args, **kwargs):
        assert facade._execution_session._lock._is_owned()
        assert lingbot2._RPU_POLICY_RUNTIME_LOCK._is_owned()
        assert facade._rpu_build_lock._is_owned()
        facade._execution_session.require_cold()
        assert facade._execution_session.stats()["active"] == 0
        policy._rpu_materialization_started = True
        owner, events = _owner(monkeypatch, owner=policy, claim=False, fault=retirement_fault)
        owner._rpu_swizzled = True
        built.append((owner, events))
        if entered is not None:
            entered.set()
            assert proceed.wait(5)
        if fail is not None:
            raise fail
        return owner

    monkeypatch.setattr(runtime, "_build_lingbot_vla_v2_impl", build)
    return built


def test_public_build_keeps_original_child_binding_cold_and_ready_to_is_noop(monkeypatch):
    facade = _facade()
    built = _builder(monkeypatch, facade)
    assert facade.to("rpu") is facade
    session = facade._execution_session
    session.require_cold()
    before = session.stats()
    assert facade.to("rpu") is facade and session.stats() == before
    with session.execute():
        with pytest.raises(RuntimeError, match="quiescent"):
            facade.to("rpu")
    before = session.stats()
    assert facade.to("rpu") is facade and session.stats() == before
    assert len(built) == 1 and _execution._UNSAFE_PROCESS_REASON is None
    facade.close()


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_public_partial_build_keeps_primary_error_and_retires_without_active_poison(monkeypatch, error_type):
    facade = _facade()
    error = error_type("original partial build failure")
    built = _builder(monkeypatch, facade, fail=error)
    with pytest.raises(error_type) as caught:
        facade.to("rpu")
    assert caught.value is error and not getattr(error, "__notes__", ())
    owner, events = built[0]
    assert owner._closed and owner._cleanup_ok
    assert [handle for kind, handle in events if kind == "destroy"] == [13, 12, 11]
    assert facade._execution_session.stats()["state"] == "CLOSED"
    assert facade._execution_session.stats()["failure_count"] == 0
    assert _execution._UNSAFE_PROCESS_REASON is None
    before = list(events)
    facade.close()
    facade.close()
    assert facade._closed and events == before
    with pytest.raises(RuntimeError):
        facade.to("rpu")
    assert len(built) == 1


def test_partial_build_cleanup_failure_is_not_published_as_closed_facade(monkeypatch):
    facade = _facade()
    error = ValueError("original partial build failure")
    built = _builder(monkeypatch, facade, fail=error, retirement_fault=12)
    with pytest.raises(ValueError) as caught:
        facade.to("rpu")
    assert caught.value is error and getattr(error, "__notes__", ())
    owner, events = built[0]
    assert not facade._closed and not owner._closed and owner._cleanup_ok is False
    assert owner._text._rpu_decoder_handle == 12
    assert facade._execution_session.stats()["state"] == "POISONED"
    assert _execution._UNSAFE_PROCESS_REASON
    before = list(events)
    with pytest.raises(RuntimeError):
        facade.close()
    assert events == before and not facade._closed


def test_active_or_started_unbuilt_facade_cannot_begin_materialization(monkeypatch):
    facade = _facade()
    built = _builder(monkeypatch, facade)
    session = facade._execution_session
    with session.execute():
        with pytest.raises(RuntimeError, match="quiescent"):
            facade.to("rpu")
    with pytest.raises(RuntimeError, match="cold session"):
        facade.to("rpu")
    assert not built and not facade._rpu_build_started
    assert session.stats()["state"] == "QUIESCENT"
    assert _execution._UNSAFE_PROCESS_REASON is None
    facade.close()


def test_public_close_waits_for_cold_build_then_retires_once(monkeypatch):
    facade = _facade()
    entered, proceed, closing = (threading.Event() for _ in range(3))
    built = _builder(monkeypatch, facade, entered=entered, proceed=proceed)
    errors = []

    def build():
        try:
            facade.to("rpu")
        except BaseException as error:
            errors.append(error)

    def close():
        closing.set()
        try:
            facade.close()
        except BaseException as error:
            errors.append(error)

    builder = threading.Thread(target=build)
    closer = threading.Thread(target=close)
    builder.start()
    try:
        assert entered.wait(5)
        closer.start()
        assert closing.wait(5)
        assert not built[0][1]
    finally:
        proceed.set()
        builder.join(5)
        if closer.ident is not None:
            closer.join(5)
    assert not builder.is_alive() and not closer.is_alive() and not errors
    assert facade._closed and facade._execution_session.stats()["state"] == "CLOSED"
    assert [handle for kind, handle in built[0][1] if kind == "destroy"] == [13, 12, 11]


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_actual_cold_facade_compensation_controls_whether_build_is_still_allowed(monkeypatch, rollback_fails):
    facade = _facade()
    session = facade._execution_session
    old = session.config
    old_mirror = facade._rpu_execution
    old_components = dict(facade._execution_components)
    old_generations = dict(facade._component_generations)
    original_publish = facade._publish_execution_views
    error = KeyboardInterrupt("original cold publication failure")
    calls = []
    def publish(*args, **kwargs):
        original_publish(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            raise error
        if rollback_fails:
            raise ValueError("cold compensation failed")
    monkeypatch.setattr(facade, "_publish_execution_views", publish)
    with pytest.raises(KeyboardInterrupt) as caught:
        _execution.reconfigure_rpu_execution(facade, {"prefill": {"chunk_size": 128}})
    assert caught.value is error and len(calls) == 2
    assert session.config is old and facade._rpu_execution is old_mirror
    assert session.generation == 0 and session.stats()["commit_count"] == 0
    built = _builder(monkeypatch, facade)
    if rollback_fails:
        assert session.stats()["state"] == "POISONED"
        assert any("cold compensation failed" in note for note in error.__notes__)
        with pytest.raises(RuntimeError, match="poisoned"):
            facade.to("rpu")
        assert not built
    else:
        session.require_cold()
        assert facade._execution_components == old_components
        assert facade._component_generations == old_generations
        facade.to("rpu")
        assert len(built) == 1
    facade.close()


@pytest.mark.parametrize("preconfigure", [False, True])
def test_public_build_reaches_actual_constructor_and_binds_all_three_children(monkeypatch, preconfigure):
    facade = _facade()
    if preconfigure:
        _execution.reconfigure_rpu_execution(facade, {"prefill": {"chunk_size": 128}})
        facade._execution_session.require_cold()

    def graph(**kwargs):
        cache = _cache()
        cache._impl.cache_invariant_ok = lambda: True
        return cache

    monkeypatch.setattr(runtime, "RPUCache", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(runtime.rpu_backend.graph, "GraphCache", graph)
    monkeypatch.setattr(runtime, "compute_align_tokens", lambda *args: None)
    monkeypatch.setattr(runtime, "precompute_fold_all", lambda *args: ())
    monkeypatch.setattr(runtime, "precompute_enc_fused", lambda *args: (None, ()))
    monkeypatch.setattr(runtime, "_op_has_cos_sin_offset", lambda: True)
    monkeypatch.setattr(runtime, "_validate_denoise_unroll", lambda: True)
    monkeypatch.setattr(runtime.LingbotVlaV2Policy, "_build_denoise_unroll", lambda self: None)
    monkeypatch.setattr(runtime.torch.ops.rpu, "causal_decoder_enable_execution_reconfigure",
                        lambda handle: None, raising=False)

    def build(policy, *args, **kwargs):
        policy._rpu_materialization_started = True
        owner, _events = _owner(monkeypatch, owner=policy, claim=False)
        cfg = SimpleNamespace(vision_start_token_id=1, vision_end_token_id=2, image_token_id=3,
            text_config=SimpleNamespace(eos_token_id=4, head_dim=128,
                                        rope_parameters={"mrope_section": [24, 20, 20]}),
            vision_config=SimpleNamespace(spatial_merge_size=2))
        runtime.LingbotVlaV2Policy.__init__(owner, vlm=owner._vlm, cfg=cfg,
            head_W=None, norm_W=None, align_W=None, align_cfg=runtime.AlignConfig(),
            rope_theta=5_000_000, exp_handle=13, exp_keep=[None] * 20,
            rpu_execution=kwargs["rpu_execution"], execution_session=kwargs["execution_session"],
            execution_generation=kwargs["execution_generation"],
            component_generations=kwargs["component_generations"])
        owner._rpu_swizzled = True
        return owner

    monkeypatch.setattr(runtime, "_build_lingbot_vla_v2_impl", build)
    assert facade.to("rpu") is facade
    session, owner = facade._execution_session, facade._policy
    session.require_cold()
    assert session.generation == int(preconfigure)
    if preconfigure:
        assert owner._rpu_execution["prefill"]["chunk_size"] == 128
    for child, component in ((owner._text, runtime.LINGBOT2_TEXT_COMPONENT),
                             (owner._visual, runtime.LINGBOT2_VISION_COMPONENT),
                             (owner, runtime.LINGBOT2_ACTION_COMPONENT)):
        assert child._planner_cost_session[0]() is session
        assert child._planner_cost_session[1] == frozenset({component})
    facade.close()


@pytest.mark.parametrize("value", [True, "auto", object()])
def test_public_builder_rejects_invalid_graph_policy_before_assets(value, monkeypatch):
    monkeypatch.setattr(runtime, "_build_lingbot_vla_v2_impl",
                        lambda *a, **kw: pytest.fail("invalid policy reached materialization"))
    with pytest.raises(TypeError, match="graph_runtime_policy"):
        runtime.build_lingbot_vla_v2("/missing/checkpoint", graph_runtime_policy=value)


@pytest.mark.parametrize("mode", (None, "auto", "off"))
def test_real_builder_binds_vision_policy_before_install_and_handle(monkeypatch, tmp_path, mode):
    import torch

    requested = (runtime.rpu_backend.graph.GraphRuntimePolicy(batch_snapshot_mode=mode)
                 if mode is not None else None)
    env_policies = []
    original = runtime.rpu_backend.graph.GraphRuntimePolicy.from_environment

    def from_environment(**kwargs):
        policy = original(**kwargs)
        env_policies.append(policy)
        return policy

    monkeypatch.setattr(runtime.rpu_backend.graph.GraphRuntimePolicy,
                        "from_environment", from_environment)
    class Text:
        def half(self): return self
        def to(self, *args, **kwargs): return self
    class Vlm:
        language_model = Text()
        visual = SimpleNamespace()
        def load_state_dict(self, *args, **kwargs): return [], []
        def eval(self): return self
    vlm = Vlm()
    cfg = SimpleNamespace(vision_config=object())
    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_vl.configuration_qwen3_vl",
                        SimpleNamespace(Qwen3VLConfig=SimpleNamespace(from_pretrained=lambda _: cfg)))
    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_vl.modeling_qwen3_vl",
                        SimpleNamespace(Qwen3VLModel=lambda _: vlm))
    monkeypatch.setattr(runtime._Lingbot2MoeColdConfig, "from_env",
                        lambda: SimpleNamespace(runtime_graph=False))
    monkeypatch.setattr(runtime, "_validate_denoise_unroll", lambda: None)
    monkeypatch.setattr(runtime, "_validate_adarms_direct_schedule", lambda: None)
    monkeypatch.setattr(runtime, "resolve_lingbot2_execution_components",
                        lambda *a, **kw: ({}, {}, {}, {}))
    monkeypatch.setattr(runtime, "ckpt_reader", lambda _: (lambda _: None, {}))
    monkeypatch.setattr(runtime, "_materialize_derived_buffers", lambda _: None)
    monkeypatch.setattr(runtime, "_assert_no_meta", lambda *a: None)
    monkeypatch.setattr(runtime, "convert_linear_weights_inplace", lambda *a, **kw: None)
    monkeypatch.setattr(torch.rpu, "set_ddr_flush", lambda _: None, raising=False)
    for name in ("RPU_LINGBOT2_MULTIVIEW_SPM_Z2", "RPU_LINGBOT2_PREFILL_W8A16",
                 "RPU_LINGBOT2_PREFILL_W4A16"):
        monkeypatch.delenv(name, raising=False)
    observed = []
    class StopAfterVisionBinding(Exception): pass
    def install(visual, **kwargs):
        assert visual is vlm.visual
        # The actual installer also uses this argument to construct its retained
        # GraphCache, covered by the real installer regression in core46.
        observed.append(kwargs["graph_runtime_policy"])
        raise StopAfterVisionBinding
    monkeypatch.setattr(runtime, "install_qwen3_vl_vision_for_rpu", install)
    (tmp_path / "config.json").write_text("{}")
    owner = SimpleNamespace()
    with pytest.raises(StopAfterVisionBinding):
        runtime._build_lingbot_vla_v2_impl(owner, "unused", qwen3vl_base=str(tmp_path),
                                         graph_runtime_policy=requested)
    assert len(observed) == 1
    assert observed[0].batch_snapshot_mode == (mode or "auto")
    assert observed[0] is (requested if requested is not None else env_policies[0])
    assert len(env_policies) == (1 if requested is None else 0)
    assert owner._visual is vlm.visual and owner._rpu_materialization_started
