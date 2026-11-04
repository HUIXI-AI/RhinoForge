"""Exercise the real Pi05 constructor and preinstall boundary without devices."""
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from rpu_backend.adapters import pi05
from rpu_backend.adapters.pi05 import runtime
from rpu_backend.api import _execution, causal_lm
from rpu_backend.runtime import RPUConfigError, hw_attrs


class _StopBeforeWeights(Exception):
    pass


# These tests stop at the first patch/install boundary, before any LeRobot
# method or weight work. Keep that optional dependency outside collection.
patches = ModuleType("rpu_backend.adapters.pi05.patches")
patches.install_class_patches = lambda owner: None


def _policy():
    policy = torch.nn.Module()
    policy.model = torch.nn.Module()
    policy.model.config = SimpleNamespace(chunk_size=50, image_features=("cam0", "cam1", "cam2"))
    return policy


@pytest.fixture(autouse=True)
def isolated_process(monkeypatch):
    from rpu_backend.graph import GraphRuntimePolicy
    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas", lambda self: True)
    monkeypatch.setitem(sys.modules, patches.__name__, patches)
    monkeypatch.setattr(pi05, "patches", patches, raising=False)
    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(causal_lm, "_LIVE_TERMINAL_REASON", None)
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.delenv("RPU_PI05_KEEP_CPU", raising=False)


@pytest.mark.parametrize("prefix,kvinsert", [(False, False), (False, True), (True, False), (True, True)])
def test_constructor_keeps_cold_requests_off_model_until_actual_preinstall(monkeypatch, prefix, kvinsert):
    monkeypatch.setenv("RPU_PI05_PREFIX_PAD16", str(int(prefix)))
    monkeypatch.setenv("RPU_PI05_KVINSERT_PAD16", str(int(kvinsert)))
    policy = _policy()
    adapter = pi05.Pi05Adapter(policy)
    for name in ("_rpu_legacy_prefix_pad16_request", "_rpu_legacy_kvinsert_pad16_request"):
        assert not hasattr(policy.model, name)
        assert name not in hw_attrs.PUBLIC_HW_ATTRS
    hw_attrs.validate_preinstall(policy)

    # Even a newly constructed mirror must reuse the session's first snapshot,
    # without parsing the now-invalid ambient environment during installation.
    monkeypatch.setenv("RPU_PI05_PREFIX_PAD16", "changed-after-construction")
    monkeypatch.setenv("RPU_PI05_KVINSERT_PAD16", "changed-after-construction")
    mirror = pi05.Pi05Adapter(policy)
    assert mirror._execution_session is adapter._execution_session is policy.model._execution_session
    assert mirror._legacy_pad16_requests == adapter._legacy_pad16_requests == (prefix, kvinsert)
    reached_install = []
    monkeypatch.setattr(pi05, "_preflight_pi05_cold_model", lambda owner, *, attn_tp=8: None)
    monkeypatch.setattr(torch.rpu, "set_caching_allocator", lambda enabled: None)

    def before_weights(owner):
        reached_install.append(owner)
        assert runtime._cold_prefix_pad16_request(owner.model) is prefix
        assert runtime._cold_kvinsert_pad16_request(owner.model) is kvinsert
        assert owner.model._execution_session is adapter._execution_session
        hw_attrs.validate_postinstall(owner)
        hw_attrs.install_hw_attr_validator(owner)
        for name, value in (("_rpu_legacy_prefix_pad16_request", prefix),
                            ("_rpu_legacy_kvinsert_pad16_request", kvinsert)):
            assert name in hw_attrs.PLANNER_SENSITIVE_ATTRS
            with pytest.raises(RPUConfigError, match="monotonic"):
                setattr(owner.model, name, not value)
            assert getattr(owner.model, name) is value
        raise _StopBeforeWeights()

    monkeypatch.setattr(patches, "install_class_patches", before_weights)
    # Runs the production geometry check, lock, public-only preinstall, claim,
    # and cold publication. The first class-patch boundary stops all device work.
    with pytest.raises(_StopBeforeWeights):
        mirror.to_rpu()
    assert reached_install == [policy]
    assert policy._rpu_swizzle_started and not getattr(policy, "_rpu_swizzled", False)
    assert causal_lm._LIVE_REF is None


@pytest.mark.parametrize("name", ["_rpu_legacy_prefix_pad16_request", "_rpu_legacy_kvinsert_pad16_request"])
def test_user_injected_internal_cold_request_still_fails_preinstall(monkeypatch, name):
    policy = _policy()
    setattr(policy.model, name, True)
    adapter = pi05.Pi05Adapter(policy)
    monkeypatch.setattr(patches, "install_class_patches", lambda owner: pytest.fail("preinstall admitted injected state"))
    with pytest.raises(RPUConfigError, match="BEFORE"):
        adapter.to_rpu()
    assert not hasattr(policy, "_rpu_swizzle_started")
    assert causal_lm._LIVE_REF is None


def test_cold_publication_requires_the_same_session_snapshot(monkeypatch):
    policy = _policy()
    adapter = pi05.Pi05Adapter(policy)
    prefix, kvinsert = adapter._legacy_pad16_requests
    adapter._legacy_pad16_requests = (not prefix, kvinsert)
    monkeypatch.setattr(pi05, "_preflight_pi05_cold_model", lambda owner, *, attn_tp=8: None)
    monkeypatch.setattr(torch.rpu, "set_caching_allocator", lambda enabled: None)
    monkeypatch.setattr(patches, "install_class_patches", lambda owner: pytest.fail("lost snapshot owner reached install"))
    with pytest.raises(pi05.RPUBackendError, match="cold snapshot lost its execution owner"):
        adapter.to_rpu()
    assert not hasattr(policy.model, "_rpu_legacy_prefix_pad16_request")
    assert not hasattr(policy.model, "_rpu_legacy_kvinsert_pad16_request")


def test_dtype_and_model_config_fail_before_allocator_claim(monkeypatch):
    policy = _policy()
    policy.bad_weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
    adapter = pi05.Pi05Adapter(policy)
    calls = []
    monkeypatch.setattr(pi05, "_claim_live_instance", lambda owner: calls.append("claim"))
    monkeypatch.setattr(
        torch.rpu, "set_caching_allocator", lambda enabled: calls.append("allocator")
    )

    with pytest.raises(pi05.RPUUnsupportedDtypeError, match="torch.float16"):
        adapter.to_rpu()
    assert calls == []
    assert not hasattr(policy, "_rpu_swizzle_started")

    policy = _policy()
    adapter = pi05.Pi05Adapter(policy)
    with pytest.raises(pi05.RPUBackendError, match="incomplete or malformed"):
        adapter.to_rpu()
    assert calls == []
    assert not hasattr(policy, "_rpu_swizzle_started")


def test_allocator_failure_releases_pi_owner_before_marker(monkeypatch):
    policy = _policy()
    adapter = pi05.Pi05Adapter(policy)
    events = []
    monkeypatch.setattr(pi05, "_preflight_pi05_cold_model", lambda owner, *, attn_tp=8: None)
    monkeypatch.setattr(
        pi05, "_claim_live_instance", lambda owner: events.append("claim")
    )
    monkeypatch.setattr(
        pi05, "_release_live_instance", lambda owner: events.append("release")
    )

    def fail_allocator(enabled):
        events.append(("allocator", enabled))
        raise RuntimeError("allocator conflict")

    monkeypatch.setattr(torch.rpu, "set_caching_allocator", fail_allocator)
    with pytest.raises(RuntimeError, match="allocator conflict"):
        adapter.to_rpu()
    assert events == ["claim", ("allocator", True), "release"]
    assert not hasattr(policy, "_rpu_swizzle_started")


def test_pi_graph_policy_uses_generic_sync_default(monkeypatch):
    from rpu_backend.graph import GraphRuntimePolicy

    names = (
        "RPU_WALL_OSS_FAST_REPLAY",
        "RPU_FASTREPLAY_SKIP_SYNC",
        "RPU_DEEP_FAST_REPLAY",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)

    policy = pi05._pi05_graph_runtime_policy()
    assert policy.graph_arena_count == 5
    assert policy.fmb_fast_replay is True
    assert policy.fast_replay_skip_sync is True
    assert policy.fmb_deep_fast_replay is True
    assert all(name not in pi05.os.environ for name in names)

    unrelated = GraphRuntimePolicy.from_environment()
    assert unrelated.fmb_fast_replay is False
    assert unrelated.fast_replay_skip_sync is True
    assert unrelated.fmb_deep_fast_replay is False

    for name in names:
        monkeypatch.setenv(name, "0")
    disabled = pi05._pi05_graph_runtime_policy()
    assert disabled.fmb_fast_replay is False
    assert disabled.fast_replay_skip_sync is False
    assert disabled.fmb_deep_fast_replay is False


@pytest.mark.parametrize("mode", ["reject", "raise"])
def test_pi_arena_failure_releases_claim_before_any_weight_mutation(monkeypatch, mode):
    from rpu_backend.graph import GraphRuntimePolicy
    policy = _policy()
    adapter = pi05.Pi05Adapter(policy)
    events = []
    monkeypatch.setattr(pi05, "_preflight_pi05_cold_model", lambda *a, **k: None)
    monkeypatch.setattr(pi05, "_claim_live_instance", lambda owner: events.append("claim"))
    monkeypatch.setattr(pi05, "_release_live_instance", lambda owner: events.append("release"))
    monkeypatch.setattr(torch.rpu, "set_caching_allocator", lambda enabled: events.append("allocator"))
    def reserve(plan):
        assert plan.graph_arena_count == 5
        assert not hasattr(policy, "_rpu_swizzle_started")
        events.append("reserve")
        if mode == "raise":
            raise RuntimeError("Graph arena allocation failed")
        return False
    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas", reserve)
    with pytest.raises((RuntimeError, pi05.RPUBackendError), match="[Gg]raph arena"):
        adapter.to_rpu()
    assert events == ["claim", "allocator", "reserve", "release"]
    assert not hasattr(policy, "_rpu_swizzle_started")


@pytest.mark.parametrize("mode", ("auto", "off"))
def test_public_graph_runtime_policy_reaches_cold_adapter_without_loader_leak(monkeypatch, mode):
    from rpu_backend.api import Pi05Policy
    from rpu_backend.graph import GraphRuntimePolicy
    from rpu_backend.adapters.pi05 import loader

    requested = GraphRuntimePolicy(batch_snapshot_mode=mode, graph_arena_count=4)
    cold = _policy()
    calls = []
    def load(path, dtype, **kwargs):
        calls.append((path, dtype, kwargs))
        return cold
    monkeypatch.setattr(loader, "_load_and_fp16_cast", load)
    public = Pi05Policy.from_pretrained("/unused", graph_runtime_policy=requested)
    assert calls == [("/unused", torch.float16, {"trust_remote_code": False})]
    assert public._adapter._graph_runtime_policy_request is requested
    mirror = pi05.Pi05Adapter(cold)
    assert mirror._graph_runtime_policy_request is requested
    effective = pi05._pi05_graph_runtime_policy(requested)
    assert effective.batch_snapshot_mode == mode
    assert effective.graph_arena_count == 5
    assert effective.fmb_fast_replay and effective.fmb_deep_fast_replay
    assert requested.graph_arena_count == 4  # frozen public request is unchanged
    with pytest.raises(ValueError, match="cold-bound"):
        pi05.Pi05Adapter(cold, graph_runtime_policy=GraphRuntimePolicy(
            batch_snapshot_mode="off" if mode == "auto" else "auto"))


def test_public_cache_capacity_is_cold_bound_without_loader_leak(monkeypatch):
    from rpu_backend.api import Pi05Policy
    from rpu_backend.adapters.pi05 import loader

    cold = _policy()
    calls = []
    monkeypatch.setattr(loader, "_load_and_fp16_cast", lambda path, dtype, **kwargs:
                        calls.append(kwargs) or cold)
    public = Pi05Policy.from_pretrained("/unused", max_seq_len=16384)
    assert calls == [{"trust_remote_code": False}]
    assert public._adapter._max_seq_len == 16384
    assert pi05.Pi05Adapter(cold)._max_seq_len == 16384
    with pytest.raises(ValueError, match="cold-bound"):
        Pi05Policy.from_lerobot_policy(cold, max_seq_len=32768)
    for invalid in (True, 0, -1, 16384.0):
        with pytest.raises(ValueError, match="positive integer"):
            Pi05Policy.from_pretrained("/must-not-load", max_seq_len=invalid)
    assert calls == [{"trust_remote_code": False}]


@pytest.mark.parametrize("value", (True, "auto", {}, 1))
def test_graph_runtime_policy_type_rejected_before_pi_loading_or_owner_mutation(monkeypatch, value):
    from rpu_backend.api import Pi05Policy
    from rpu_backend.adapters.pi05 import loader
    monkeypatch.setattr(loader, "_load_and_fp16_cast", lambda *_a, **_k: pytest.fail("loaded assets"))
    for call in (
        lambda: Pi05Policy.from_pretrained("/must-not-load", graph_runtime_policy=value),
        lambda: Pi05Policy.from_lerobot_policy(object(), graph_runtime_policy=value),
        lambda: loader.load_and_construct_pi05_policy("/must-not-load", graph_runtime_policy=value),
        lambda: pi05.Pi05Adapter(object(), graph_runtime_policy=value),
    ):
        with pytest.raises(TypeError, match="graph_runtime_policy"):
            call()


@pytest.mark.parametrize("mode", ("auto", "off"))
@pytest.mark.parametrize("actual_cores,requested_cores", ((8, 4), (4, 8), (6, 8)))
def test_public_graph_policy_binds_actual_pi_cores_before_materialization(
        monkeypatch, mode, actual_cores, requested_cores):
    from rpu_backend.graph import GraphRuntimePolicy
    from rpu_backend.adapters.pi05 import cores as core_profiles

    policy = _policy()
    policy.model._rpu_execution = {"model": {"num_cores": actual_cores}}
    request = GraphRuntimePolicy(batch_snapshot_mode=mode,
                                 execution_core_count=requested_cores)
    adapter = pi05.Pi05Adapter(policy, graph_runtime_policy=request)
    monkeypatch.setattr(core_profiles, "validate_reduced_policy", lambda *a: None)
    monkeypatch.setattr(pi05, "_preflight_pi05_cold_model", lambda *a, **kw: None)
    monkeypatch.setattr(torch.rpu, "set_caching_allocator", lambda enabled: None)
    bound = []

    def reserve(actual):
        assert not hasattr(policy, "_rpu_swizzle_started")
        bound.append(actual)
        raise _StopBeforeWeights()

    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas", reserve)
    with pytest.raises(_StopBeforeWeights):
        adapter.to_rpu()
    assert len(bound) == 1
    assert bound[0].execution_core_count == actual_cores
    assert bound[0].batch_snapshot_mode == mode
    assert bound[0].graph_arena_count == 5
    assert bound[0].fmb_fast_replay and bound[0].fmb_deep_fast_replay
    assert request.execution_core_count == requested_cores
    assert causal_lm._LIVE_REF is None


def _denoise_install_fixture(monkeypatch, cores, mode, *, fail_weights=False):
    """CPU installed weights and native call doubles for the actual installer."""
    import rpu_backend
    from rpu_backend.adapters.pi05 import cores as core_profiles, weights, w8a16
    from rpu_backend.graph import GraphRuntimePolicy

    policy = _policy()
    adapter = pi05.Pi05Adapter(policy)
    model = policy.model
    model.config.max_action_dim = 32
    tensor = torch.ones(1, dtype=torch.float16)
    projection = SimpleNamespace(weight=tensor)
    modulation = SimpleNamespace(_rpu_dense_w_rp=tensor, _rpu_dense_b_rp=tensor)
    topology = core_profiles.core_topology({"model": {"num_cores": cores}})
    expert = SimpleNamespace(
        config=SimpleNamespace(num_key_value_heads=topology.attn_tp,
            num_attention_heads=8, head_dim=256, intermediate_size=4096,
            rms_norm_eps=1e-6),
        _rpu_decoder_topology=topology,
        _gemma_rope_cos=tensor, _gemma_rope_sin=tensor,
        norm=modulation,
        layers=[SimpleNamespace(
            self_attn=SimpleNamespace(q_proj=projection, k_proj=projection,
                                     v_proj=projection, o_proj=projection),
            mlp=SimpleNamespace(gate_proj=projection, up_proj=projection,
                                down_proj=projection),
            input_layernorm=modulation, post_attention_layernorm=modulation)])
    model.paligemma_with_expert = SimpleNamespace(
        gemma_expert=SimpleNamespace(model=expert),
        paligemma=SimpleNamespace(model=SimpleNamespace(
            language_model=SimpleNamespace(_rpu_cache=SimpleNamespace(max_seq_len=512)))))
    model.action_in_proj = SimpleNamespace(out_features=1024,
        _rpu_w_num_cores_1=tensor, _rpu_bias=tensor)
    model.action_out_proj = SimpleNamespace(_rpu_w_num_cores_1=tensor, _rpu_bias=tensor)
    monkeypatch.setattr(weights, "_prepare_final_norm_weights", lambda *a, **kw: None)
    monkeypatch.setattr(weights, "_swizzle_action_proj_weights", lambda *a: None)
    monkeypatch.setattr(w8a16, "pi05_expert_scale_lists", lambda *a, **kw: ([],) * 7)
    monkeypatch.setattr(w8a16, "pi05_nvfp4_tensor_scale_tables", lambda *a: [])
    monkeypatch.setenv("RPU_PI05_FUSED_DENOISE", "1")
    monkeypatch.setenv("RPU_PI05_ADARMS_DENSE_W8A16", "0")
    events, caches = [], []
    runtime_policy = GraphRuntimePolicy(batch_snapshot_mode=mode, execution_core_count=cores)

    def cache_factory(**kwargs):
        events.append(("cache", kwargs))
        cache = SimpleNamespace(clear=lambda: events.append(("clear",)),
                                cache_invariant_ok=lambda: True)
        caches.append(cache)
        return cache

    monkeypatch.setattr(rpu_backend.graph, "GraphCache", cache_factory)
    native_cores = [8]

    def create(acc32):
        events.append(("create", acc32))
        return 42

    def set_cores(handle, actual):
        assert handle == 42
        events.append(("cores", actual))
        native_cores[0] = actual

    error = RuntimeError("injected denoise weight bind failure")

    def set_weights(handle, *args):
        assert handle == 42 and args[19:27] == (1024, 32, 50, 8, topology.attn_tp, 256, 1, 1e-6)
        events.append(("weights", native_cores[0]))
        if fail_weights:
            raise error

    def get_topology(handle):
        assert handle == 42
        actual = core_profiles.core_topology({"model": {"num_cores": native_cores[0]}})
        return (1, actual.num_cores, actual.attn_tp, actual.mlp_tp, actual.attn_tp,
                8, 4096, core_profiles.physical_intermediate_size(4096, actual.mlp_tp, tensor.dtype))

    for name, callback in (
        ("create", create), ("set_execution_core_count", set_cores),
        ("set_weights", set_weights), ("get_execution_topology", get_topology),
        ("set_chunk_envelope", lambda handle, prefix, suffix: events.append(("envelope", handle, prefix, suffix))),
        ("set_chunk_size", lambda handle, chunk: events.append(("chunk", handle, chunk))),
        ("destroy", lambda handle: events.append(("destroy", handle))),
    ):
        monkeypatch.setattr(torch.ops.rpu, "pi05_denoise_step_" + name, callback, raising=False)
    return adapter, model, runtime_policy, events, caches, error


@pytest.mark.parametrize("mode", ("auto", "off"))
@pytest.mark.parametrize("cores", (8, 4, 6))
def test_actual_fused_denoise_install_binds_native_and_graph_policy(monkeypatch, cores, mode):
    adapter, model, policy, events, caches, _ = _denoise_install_fixture(monkeypatch, cores, mode)
    adapter._install_fused_denoise_handle(runtime_policy=policy)
    assert events[0] == ("cache", {"runtime_policy": policy})
    assert events[1] == ("create", False)
    assert [event for event in events if event[0] == "cores"] == ([] if cores == 8 else [("cores", cores)])
    assert ("weights", cores) in events
    assert events[-2:] == [("envelope", 42, 512, 0), ("chunk", 42, 0)]
    resource = model._rpu_fused_denoise_retirement_state
    assert model._rpu_fused_denoise_handle == resource.handle == 42
    assert model._rpu_fused_denoise_graph_cache is caches[0]
    assert resource.graphs == (caches[0],)
    assert adapter._finalizers == [resource.finalizer]
    resource.retire()
    assert events[-2:] == [("clear",), ("destroy", 42)]
    assert model._rpu_fused_denoise_handle is None and not resource.finalizer.alive


def test_actual_fused_denoise_install_failure_clears_before_destroy_and_restores(monkeypatch):
    adapter, model, policy, events, _, original = _denoise_install_fixture(
        monkeypatch, 4, "off", fail_weights=True)
    before = dict(vars(model))
    old_finalizers = adapter._finalizers
    with pytest.raises(RuntimeError, match="injected denoise weight bind failure") as caught:
        adapter._install_fused_denoise_handle(runtime_policy=policy)
    assert caught.value is original
    assert ("cores", 4) in events and ("weights", 4) in events
    assert events[-2:] == [("clear",), ("destroy", 42)]
    assert vars(model) == before
    assert adapter._finalizers is old_finalizers
    assert _execution._UNSAFE_PROCESS_REASON is None
