"""A private unbuilt-owner proof preserves coldness, never resets started."""
from types import SimpleNamespace

import pytest

from rpu_backend.api import _execution


def _session(owner=None, **callbacks):
    owner = SimpleNamespace(unbuilt=True) if owner is None else owner
    callbacks.setdefault("cold_config_only", lambda: owner.unbuilt is True)
    session = _execution.bind_execution_session(owner, {"prefill": {"chunk_size": 64}},
        entry_point="cold-config-test", graph_mode="COMPOSITE_CHILD", **callbacks)
    return owner, session


def test_pure_changes_and_equal_noop_remain_cold_until_actual_execute():
    owner, session = _session()
    for chunk in (96, 128):
        result = session.reconfigure({"prefill": {"chunk_size": chunk}})
        assert owner._rpu_execution is result
        session.require_cold()
        before = session.stats()
        assert session.reconfigure({}) is result and session.stats() == before
        session.require_cold()
    assert session.generation == 2
    with session.execute():
        pass
    session.reconfigure({"prefill": {"chunk_size": 64}})
    with pytest.raises(RuntimeError, match="cold session"):
        session.require_cold()


@pytest.mark.parametrize("hook", [None, lambda: False, lambda: 1])
def test_default_or_unproven_owner_keeps_original_started_semantics(hook):
    _, session = _session(cold_config_only=hook)
    session.reconfigure({})
    with pytest.raises(RuntimeError, match="cold session"):
        session.require_cold()


def test_force_rebuild_is_not_a_cold_configuration_exception():
    calls = []
    _, session = _session(apply=lambda *args, **kwargs: calls.append(kwargs))
    session._reconfigure({}, force_rebuild=True)
    assert calls == [{"force_rebuild": True}]
    with pytest.raises(RuntimeError, match="cold session"):
        session.require_cold()


@pytest.mark.parametrize("field,value", [
    ("_planner_costs", {"installed": object()}),
    ("_planner_native", {"installed": object()}),
    ("_planner_native_catalogs", {"installed": object()}),
    ("_planner_native_revalidators", {"installed": object()}),
    ("_planner_cost_observer", object()),
    ("_planner_calibration", object()),
])
def test_existing_native_or_cost_state_cannot_claim_configuration_only(field, value):
    _, session = _session()
    setattr(session, field, value)
    if field == "_planner_cost_observer":
        before = session.stats()
        with pytest.raises(RuntimeError, match="cost collection"):
            session._reconfigure({})
        assert session.stats() == before and not session._is_cold_config_only()
        return
    session._reconfigure({})
    with pytest.raises(RuntimeError, match="cold session"):
        session.require_cold()


def test_pure_validation_failure_keeps_cold_and_original_configuration():
    error = ValueError("rejected configuration")
    def reject(_new):
        raise error
    _, session = _session(validate=reject)
    old = session.config
    with pytest.raises(ValueError) as caught:
        session.reconfigure({"prefill": {"chunk_size": 96}})
    assert caught.value is error and session.config is old and session.generation == 0
    session.require_cold()


@pytest.mark.parametrize("boundary", ["apply", "publication"])
def test_pure_compensation_restores_identity_and_keeps_cold(boundary):
    error = KeyboardInterrupt("cold transaction interrupted")
    class Storage(dict):
        interrupt = False
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if self.interrupt:
                self.interrupt = False
                raise error
    class Owner:
        unbuilt = True
        def __init__(self):
            self.storage = Storage()
        @property
        def __dict__(self):
            return self.storage
    owner = Owner()
    calls = []
    def apply(*_args):
        calls.append("apply")
        if boundary == "apply":
            raise error
        owner.storage.interrupt = True
    _, session = _session(owner, apply=apply, rollback=lambda *_args: calls.append("rollback"))
    old = session.config
    owner.storage["_rpu_execution"] = old
    with pytest.raises(KeyboardInterrupt) as caught:
        session.reconfigure({"prefill": {"chunk_size": 96}})
    assert caught.value is error and calls == ["apply", "rollback"]
    assert session.config is old and owner.storage["_rpu_execution"] is old
    assert session.generation == 0 and session.stats()["commit_count"] == 0
    session.require_cold()


@pytest.mark.parametrize("boundary", ["validate", "apply", "rollback"])
@pytest.mark.parametrize("escape", ["execute", "native", "cost", "build"])
def test_callback_cannot_cross_runtime_boundary_and_return_to_cold(boundary, escape):
    error = ValueError("original callback failure")
    owner = SimpleNamespace(unbuilt=True)
    calls = []
    def cross():
        if escape == "execute":
            with session.execute():
                pass
        elif escape == "native":
            session._record_planner_native(owner, "", "prefill", ("causal_decoder", 17))
        elif escape == "cost":
            session._planner_cost_observer = object()
        else:
            owner.unbuilt = False
    def validate(_new):
        if boundary == "validate":
            cross()
            raise error
    def apply(*_args):
        if boundary == "apply":
            cross()
        raise error
    def rollback(*_args):
        calls.append("rollback")
        if boundary == "rollback":
            cross()
    _, session = _session(owner, validate=validate, apply=apply, rollback=rollback)
    with pytest.raises(ValueError) as caught:
        session.reconfigure({"prefill": {"chunk_size": 96}})
    assert caught.value is error
    assert calls == (["rollback"] if boundary == "rollback" else [])
    assert session.stats()["state"] == "POISONED" and session._started
    with pytest.raises(RuntimeError, match="poisoned"):
        with session.execute():
            pytest.fail("a crossed configuration transaction cannot execute")


def test_successful_callback_that_builds_resources_is_poisoned_before_publication():
    owner = SimpleNamespace(unbuilt=True)
    def apply(*_args):
        owner.unbuilt = False
    _, session = _session(owner, apply=apply, rollback=lambda *_args: pytest.fail("unsafe rollback"))
    old = session.config
    with pytest.raises(RuntimeError, match="crossed into runtime"):
        session.reconfigure({"prefill": {"chunk_size": 96}})
    assert session.config is old and session.stats()["state"] == "POISONED"


@pytest.mark.parametrize("rollback", [None, lambda *_args: (_ for _ in ()).throw(RuntimeError("rollback"))])
def test_uncompensated_cold_apply_failure_keeps_original_poison_rule(rollback):
    def apply(*_args):
        raise ValueError("apply")
    _, session = _session(apply=apply, rollback=rollback)
    with pytest.raises(ValueError, match="apply"):
        session.reconfigure({"prefill": {"chunk_size": 96}})
    assert session._started and session.stats()["state"] == "POISONED"


def test_private_callback_is_not_json_and_obeys_original_hook_immutability():
    owner, session = _session(cold_config_only=None)
    proof = lambda: True
    session.configure_callbacks(cold_config_only=proof)
    session.configure_callbacks(cold_config_only=proof)
    with pytest.raises(RuntimeError, match="already bound"):
        session.configure_callbacks(cold_config_only=lambda: True)
    with pytest.raises(TypeError, match="owner callback"):
        _session(cold_config_only=True)
    with pytest.raises(ValueError):
        session.reconfigure({"cold_config_only": True})
    session.require_cold()
    with session.execute():
        pass
    with pytest.raises(RuntimeError, match="immutable"):
        session.configure_callbacks(cold_config_only=lambda: True)


@pytest.mark.parametrize("cross_runtime", [False, True])
def test_view_snapshot_failure_never_calls_stale_apply_rollback(cross_runtime):
    error = KeyboardInterrupt("view snapshot interrupted")
    events = []
    _, session = _session(apply=lambda *_args: events.append("apply"),
                          rollback=lambda *_args: events.append("rollback"))
    class View:
        fail = False
        @property
        def __dict__(self):
            if self.fail:
                if cross_runtime:
                    with session.execute():
                        pass
                raise error
            return self.storage
        def __init__(self):
            self.storage = {}
    view = View()
    session.register_config_view(view)
    view.fail = True
    old = session.config
    with pytest.raises(KeyboardInterrupt) as caught:
        session.reconfigure({"prefill": {"chunk_size": 96}})
    assert caught.value is error and session.config is old and session.generation == 0
    assert events == []
    if cross_runtime:
        assert session._started and session._poisoned
    else:
        session.require_cold()


def test_a_new_view_during_apply_cannot_leave_a_stale_successful_mirror():
    class View:
        pass
    extra = View()
    def apply(*_args):
        session.register_config_view(extra)
    _, session = _session(apply=apply, rollback=lambda *_args: pytest.fail("view domain escaped"))
    old = session.config
    with pytest.raises(RuntimeError, match="crossed into runtime"):
        session.reconfigure({"prefill": {"chunk_size": 96}})
    assert session.config is old and extra._rpu_execution is old
    assert session._started and session._poisoned
