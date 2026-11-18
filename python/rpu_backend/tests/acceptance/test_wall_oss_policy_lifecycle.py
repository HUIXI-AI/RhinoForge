"""Board-free process-slot and build lifecycle contracts for WallOssPolicy."""
from __future__ import annotations

from pathlib import Path
import sys
import threading
import types

import pytest

from rpu_backend import adapters
from rpu_backend.api import WallOssPolicy
from rpu_backend.api.causal_lm import _release_live_instance
from rpu_backend.api.errors import RPUBackendError, RPUSingleHandleError


class _FakeVLA:
    action_dim = 26

    def __init__(self, *, complete=True):
        self._complete = complete
        self._closed = False
        self._rpu_execution = {}
        self.close_calls = 0

    def _runtime_complete(self):
        return self._complete and not self._closed

    def close(self):
        self.close_calls += 1
        self._closed = True


class _PublishInterrupted(BaseException):
    pass


class _FailingReadyPolicy(WallOssPolicy):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if (
            name == "_rpu_ready"
            and value is True
            and getattr(self, "_fail_ready_publish", False)
        ):
            raise _PublishInterrupted("synthetic ready publication interruption")


def _install_fake_builder(monkeypatch, builder):
    module = types.ModuleType("rpu_backend.adapters.wall_oss")
    # The build-failure path imports the real `._retirement` helper, so the
    # stub keeps the package search path instead of shadowing the package.
    module.__path__ = [str(Path(adapters.__file__).parent / "wall_oss")]
    module.build_wall_oss_vla = builder
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _new_policy(cls=WallOssPolicy) -> WallOssPolicy:
    return cls.from_checkpoint("unused/checkpoint")


def _new_rtc_policy(*, runtime_env=None) -> WallOssPolicy:
    return WallOssPolicy.from_checkpoint(
        "unused/checkpoint",
        dataset_key="ex_normal",
        camera_names=("face_view", "left_wrist_view", "right_wrist_view"),
        delta_action=True,
        state_bins=512,
        action_horizon=20,
        num_steps=6,
        max_seq_len=672,
        rtc=True,
        runtime_env=runtime_env,
    )


def test_rtc_policy_builds_without_opt_in(monkeypatch):
    monkeypatch.delenv("RPU_WALL_OSS_RTC_ALLOW_NUMERIC_BLOCKED", raising=False)
    builds = []
    _install_fake_builder(
        monkeypatch,
        lambda *args, **kwargs: builds.append((args, kwargs)) or _FakeVLA(),
    )
    policy = _new_rtc_policy()
    try:
        assert policy.to("rpu") is policy
        assert len(builds) == 1
        assert builds[0][1]["rtc"] is True
    finally:
        _release_live_instance(policy)


def test_successful_build_claims_process_wide_slot(monkeypatch):
    builds = []

    def build(*args, **kwargs):
        builds.append((args, kwargs))
        return _FakeVLA()

    _install_fake_builder(monkeypatch, build)
    first = _new_policy()
    second = _new_policy()
    try:
        assert first.to("rpu") is first
        assert first.to("rpu:0") is first
        assert len(builds) == 1
        with pytest.raises(RPUSingleHandleError, match="model/policy"):
            second.to("rpu")

        _release_live_instance(first)
        assert second.to("rpu") is second
        assert len(builds) == 2
    finally:
        _release_live_instance(first)
        _release_live_instance(second)


def test_failed_build_is_not_retryable_and_retains_slot(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic build failure")

    _install_fake_builder(monkeypatch, fail)
    broken = _new_policy()
    other = _new_policy()
    try:
        with pytest.raises(RuntimeError, match="synthetic build failure"):
            broken.to("rpu")

        _install_fake_builder(monkeypatch, lambda *a, **k: _FakeVLA())
        with pytest.raises(RPUBackendError, match="previously failed"):
            broken.to("rpu")
        with pytest.raises(RPUSingleHandleError):
            other.to("rpu")
    finally:
        _release_live_instance(broken)
        _release_live_instance(other)


@pytest.mark.parametrize("state", ["closed", "incomplete"])
def test_ready_marker_requires_complete_live_vla_before_noop(monkeypatch, state):
    builds = []
    _install_fake_builder(
        monkeypatch,
        lambda *args, **kwargs: builds.append((args, kwargs)) or _FakeVLA(),
    )
    policy = _new_policy()
    vla = _FakeVLA(complete=state != "incomplete")
    if state == "closed":
        vla.close()
    policy._vla = vla
    policy._rpu_build_started = True
    policy._rpu_ready = True

    with pytest.raises(RPUBackendError, match="incomplete or closed"):
        policy.to("rpu")

    assert builds == []


def test_incomplete_builder_result_is_closed_and_policy_is_poisoned(monkeypatch):
    vla = _FakeVLA(complete=False)
    _install_fake_builder(monkeypatch, lambda *args, **kwargs: vla)
    policy = _new_policy()
    try:
        with pytest.raises(RPUBackendError, match="without complete live"):
            policy.to("rpu")

        assert vla.close_calls == 1
        assert policy._vla is None
        assert policy._rpu_ready is False
        assert policy._rpu_build_started is True
        with pytest.raises(RPUBackendError, match="previously failed"):
            policy.to("rpu")
    finally:
        _release_live_instance(policy)


def test_ready_publication_base_exception_closes_pending_vla(monkeypatch):
    vla = _FakeVLA()
    _install_fake_builder(monkeypatch, lambda *args, **kwargs: vla)
    policy = _new_policy(_FailingReadyPolicy)
    policy._fail_ready_publish = True
    try:
        with pytest.raises(_PublishInterrupted, match="ready publication"):
            policy.to("rpu")

        assert vla.close_calls == 1
        assert policy._vla is None
        assert policy._rpu_ready is False
        assert policy._rpu_build_started is True
    finally:
        _release_live_instance(policy)


@pytest.mark.parametrize("device", ["cpu", "cuda", "rpu:1", "rpu-ish", 0])
def test_invalid_device_fails_before_builder_and_slot_claim(monkeypatch, device):
    called = False

    def build(*args, **kwargs):
        nonlocal called
        called = True
        return _FakeVLA()

    _install_fake_builder(monkeypatch, build)
    policy = _new_policy()
    with pytest.raises(RPUBackendError, match="only supports device"):
        policy.to(device)
    assert called is False

    # A bad device must not reserve the global slot.
    other = _new_policy()
    try:
        assert other.to("rpu") is other
    finally:
        _release_live_instance(policy)
        _release_live_instance(other)


def test_concurrent_to_on_same_policy_does_not_double_build(monkeypatch):
    entered = threading.Event()
    finish = threading.Event()
    builds = 0

    def build(*args, **kwargs):
        nonlocal builds
        builds += 1
        entered.set()
        assert finish.wait(timeout=5)
        return _FakeVLA()

    _install_fake_builder(monkeypatch, build)
    policy = _new_policy()
    errors = []

    def run_first():
        try:
            policy.to("rpu")
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    thread = threading.Thread(target=run_first)
    thread.start()
    try:
        assert entered.wait(timeout=5)
        with pytest.raises(RPUBackendError, match="another thread"):
            policy.to("rpu")
        finish.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert errors == []
        assert builds == 1
        assert policy._rpu_ready is True
    finally:
        finish.set()
        thread.join(timeout=5)
        _release_live_instance(policy)
