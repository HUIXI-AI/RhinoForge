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
