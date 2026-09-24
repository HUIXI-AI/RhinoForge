from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from rpu_backend.api._execution import (
    bind_execution_session,
    execution_guard,
    native_execution_reconfigure,
    reconfigure_rpu_execution,
    rpu_execution_stats,
)


SUPPORTED = {
    "prefill": ("chunk_size", "padding_rows", "padding_budget"),
    "vision": ("chunk_size",),
}
SUPPORTED_COMPONENTS = {
    "language_model": {"prefill": ("chunk_size",)},
    "vision_encoder": {"vision": ("chunk_size",)},
}


def _session(owner, *, apply=None, rollback=None, validate=None):
    return bind_execution_session(
        owner,
        {"prefill": {"chunk_size": 64}, "vision": {"chunk_size": 128}},
        entry_point="test-owner",
        supported=SUPPORTED,
        validate=validate,
        apply=apply,
        rollback=rollback,
        graph_mode="BOUNDED_ONESHOT",
    )


def test_process_unsafe_gate_stops_existing_and_future_sessions_before_callbacks(monkeypatch):
    from rpu_backend.api import _execution

    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    calls = []
    owner = SimpleNamespace()
    session = _session(owner, apply=lambda *args: calls.append("apply"))
    _execution._mark_execution_process_unsafe("retirement failed")
    _execution._mark_execution_process_unsafe("later error")
    assert _execution._UNSAFE_PROCESS_REASON == "retirement failed"
    for operation in (
        session.require_cold,
        lambda: session.reconfigure({"prefill": {"chunk_size": 96}}),
        lambda: session.shutdown(lambda: calls.append("retire")),
        lambda: session._install_planner_costs("", "prefill", lambda: calls.append("install")),
        lambda: session._record_planner_native(owner, "", "prefill", ("causal_decoder", 1)),
        lambda: _session(owner),
        lambda: _session(SimpleNamespace()),
    ):
        with pytest.raises(RuntimeError, match="process is unsafe"):
            operation()
    for context in (
        session.execute(),
        session._collect_planner_costs(lambda *args: calls.append("observe")),
        native_execution_reconfigure(SimpleNamespace()),
    ):
        with pytest.raises(RuntimeError, match="process is unsafe"):
            with context:
                calls.append("entered")
    assert calls == []
    assert session.generation == 0 and session.config["prefill"]["chunk_size"] == 64
    assert session.stats()["commit_count"] == 0  # Diagnostics remain readable.


def test_reconfigure_replaces_one_stage_and_commits_one_generation():
    owner = SimpleNamespace()
    calls = []
    session = _session(
        owner,
        validate=lambda new: calls.append(("validate", new)),
        apply=lambda old, new, generation: calls.append(
            ("apply", old, new, generation)
        ),
    )

    result = reconfigure_rpu_execution(
        owner, {"prefill": {"padding_budget": 16}}
    )

    assert result == {
        "prefill": {"padding_budget": 16},
        "vision": {"chunk_size": 128},
    }
    assert owner._rpu_execution is result
    assert session.generation == 1
    assert [call[0] for call in calls] == ["validate", "apply"]
    assert calls[-1][-1] == 1
    assert rpu_execution_stats(owner)["commit_count"] == 1


def test_empty_stage_removes_override_without_a_deletion_sentinel():
    owner = SimpleNamespace()
    _session(owner)

    assert reconfigure_rpu_execution(owner, {"prefill": {}}) == {
        "vision": {"chunk_size": 128}
    }


def test_component_capabilities_flow_through_session_and_patch_one_child():
    owner = SimpleNamespace()
    capabilities = {
        component_id: dict(stages)
        for component_id, stages in SUPPORTED_COMPONENTS.items()
    }
    session = bind_execution_session(
        owner,
        {
            "prefill": {"chunk_size": 64},
            "components": {
                "language_model": {"prefill": {"chunk_size": 96}},
                "vision_encoder": {"vision": {"chunk_size": 128}},
            },
        },
        entry_point="test-owner",
        supported_components=capabilities,
        graph_mode="COMPOSITE_CHILD",
    )

    capabilities["language_model"]["prefill"] = ("padding_rows",)

    assert session.reconfigure({
        "components": {
            "language_model": {"prefill": {"chunk_size": 160}},
        },
    }) == {
        "prefill": {"chunk_size": 64},
        "components": {
            "language_model": {"prefill": {"chunk_size": 160}},
            "vision_encoder": {"vision": {"chunk_size": 128}},
        },
    }
    assert session.reconfigure({
        "components": {"language_model": {"prefill": {}}},
    }) == {
        "prefill": {"chunk_size": 64},
        "components": {
            "vision_encoder": {"vision": {"chunk_size": 128}},
        },
    }

    with pytest.raises(ValueError, match="registered rpu_execution components changed"):
        bind_execution_session(
            owner,
            session.config,
            entry_point="test-owner",
            supported_components=capabilities,
            graph_mode="COMPOSITE_CHILD",
        )


def test_failed_apply_rolls_back_config_and_generation():
    owner = SimpleNamespace()
    calls = []

    def fail(_old, _new, _generation):
        calls.append("apply")
        raise RuntimeError("apply failed")

    session = _session(
        owner,
        apply=fail,
        rollback=lambda old, _new, generation: calls.append(
            ("rollback", old, generation)
        ),
    )
    old = session.config

    with pytest.raises(RuntimeError, match="apply failed"):
        reconfigure_rpu_execution(owner, {"prefill": {"chunk_size": 96}})

    assert session.config is old
    assert session.generation == 0
    assert calls == ["apply", ("rollback", old, 0)]
    assert rpu_execution_stats(owner)["state"] == "QUIESCENT"


def test_native_begin_interruption_aborts_the_exact_attempt():
    state = {"active": False, "attempt": None}

    class InterruptedToken:
        def __int__(self):
            raise KeyboardInterrupt

    def begin(attempt):
        state.update(active=True, attempt=attempt)
        return InterruptedToken()

    def abort_attempt(attempt):
        assert state == {"active": True, "attempt": attempt}
        state.update(active=False, attempt=None)
        return True

    ops = SimpleNamespace(
        execution_reconfigure_begin=begin,
        execution_reconfigure_commit=lambda _token: None,
        execution_reconfigure_abort=lambda _token: None,
        execution_reconfigure_abort_attempt=abort_attempt,
    )

    with pytest.raises(KeyboardInterrupt):
        with native_execution_reconfigure(ops):
            pytest.fail("begin must not yield an unknown token")

    assert state == {"active": False, "attempt": None}


def test_native_commit_interruption_aborts_a_still_active_token():
    state = {"active": False, "token": None, "fail_commit": True}

    def begin(_attempt):
        assert not state["active"]
        state.update(active=True, token=73)
        return 73

    def commit(token):
        assert state["active"] and token == state["token"]
        if state["fail_commit"]:
            state["fail_commit"] = False
            raise KeyboardInterrupt
        state.update(active=False, token=None)

    def abort(token):
        assert state["active"] and token == state["token"]
        state.update(active=False, token=None)

    ops = SimpleNamespace(
        execution_reconfigure_begin=begin,
        execution_reconfigure_commit=commit,
        execution_reconfigure_abort=abort,
        execution_reconfigure_abort_attempt=lambda _attempt: False,
    )

    with pytest.raises(KeyboardInterrupt):
        with native_execution_reconfigure(ops):
            pass
    assert not state["active"]

    with native_execution_reconfigure(ops):
        pass
    assert not state["active"]


@pytest.mark.parametrize("failure", ["stage", "commit_before_apply", "commit_after_apply"])
def test_native_journal_marks_possible_mutation_before_commit(failure):
    journal = {"mutation_started": False}
    state = {"live": 0}
    def commit(token):
        assert journal["mutation_started"] is True
        if failure == "commit_after_apply":
            state["live"] = 64
        raise RuntimeError("commit boundary")
    ops = SimpleNamespace(
        execution_reconfigure_begin=lambda attempt: 73,
        execution_reconfigure_commit=commit,
        execution_reconfigure_abort=lambda token: None,
        execution_reconfigure_abort_attempt=lambda attempt: False,
    )
    with pytest.raises(RuntimeError):
        with native_execution_reconfigure(ops, journal=journal):
            assert journal["mutation_started"] is False
            if failure == "stage":
                raise RuntimeError("stage boundary")
    assert journal["mutation_started"] is (failure != "stage")
    assert state["live"] == (64 if failure == "commit_after_apply" else 0)


def test_partial_publication_compensates_native_and_all_config_mirrors():
    class InterruptingDict(dict):
        interrupt = False

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if self.interrupt:
                self.interrupt = False
                raise KeyboardInterrupt

    class View:
        def __init__(self):
            self.storage = InterruptingDict()

        @property
        def __dict__(self):
            return self.storage

    owner = SimpleNamespace()
    native = {"config": "old"}
    rollback_generations = []

    def apply(_old, _new, _generation):
        native["config"] = "new"

    def rollback(_old, _new, generation):
        rollback_generations.append(generation)
        native["config"] = "old"

    session = _session(owner, apply=apply, rollback=rollback)
    view = View()
    session.register_config_view(view)
    old = session.config
    view.storage.interrupt = True

    with pytest.raises(KeyboardInterrupt):
        session.reconfigure({"prefill": {"chunk_size": 96}})

    assert native["config"] == "old"
    assert "_rpu_execution" not in vars(owner)
    assert view.storage["_rpu_execution"] is old
    assert session.config is old
    assert session.generation == 0
    assert rollback_generations == [0]
    assert session.stats()["commit_count"] == 0
    assert session.stats()["state"] == "QUIESCENT"


def test_failed_rollback_poison_session():
    owner = SimpleNamespace()

    def fail(*_args):
        raise RuntimeError("apply failed")

    def rollback_fail(*_args):
        raise RuntimeError("rollback failed")

    _session(owner, apply=fail, rollback=rollback_fail)
    with pytest.raises(RuntimeError, match="apply failed"):
        reconfigure_rpu_execution(owner, {"prefill": {"chunk_size": 96}})
    assert rpu_execution_stats(owner)["state"] == "POISONED"
    with pytest.raises(RuntimeError, match="poisoned"):
        with execution_guard(owner):
            pass


def test_successful_retirement_makes_poisoned_session_idempotently_closed():
    owner = SimpleNamespace()

    def fail(*_args):
        raise RuntimeError("apply failed")

    session = _session(owner, apply=fail, rollback=fail)
    with pytest.raises(RuntimeError, match="apply failed"):
        session.reconfigure({"prefill": {"chunk_size": 96}})

    assert session.shutdown(lambda: True) is True
    assert session.shutdown(lambda: pytest.fail("retired twice")) is None
    assert session.stats()["state"] == "CLOSED"


def test_failed_apply_without_rollback_poison_session():
    owner = SimpleNamespace()

    def fail(*_args):
        raise RuntimeError("partial apply failed")

    _session(owner, apply=fail)
    with pytest.raises(RuntimeError, match="partial apply failed"):
        reconfigure_rpu_execution(owner, {"prefill": {"chunk_size": 96}})
    assert rpu_execution_stats(owner)["state"] == "POISONED"


def test_bound_callbacks_cannot_be_replaced():
    owner = SimpleNamespace()
    _session(owner, apply=lambda *_args: None)

    with pytest.raises(RuntimeError, match="already bound"):
        _session(owner, apply=lambda *_args: None)


def test_registered_config_views_follow_every_publication():
    class View:
        pass

    owner = SimpleNamespace()
    mirror_a = View()
    mirror_b = View()
    session = _session(owner)
    session.register_config_view(mirror_a)
    session.register_config_view(mirror_b)

    published = session.reconfigure({"prefill": {"chunk_size": 96}})

    assert owner._rpu_execution is published
    assert mirror_a._rpu_execution is published
    assert mirror_b._rpu_execution is published


def test_reconfigure_waits_for_inflight_execution():
    owner = SimpleNamespace()
    applied = threading.Event()
    session = _session(
        owner,
        apply=lambda *_args: applied.set(),
    )
    entered = threading.Event()
    release = threading.Event()

    def execute():
        with execution_guard(owner):
            entered.set()
            assert release.wait(5)

    worker = threading.Thread(target=execute)
    worker.start()
    assert entered.wait(5)

    reconfigured = threading.Thread(
        target=lambda: reconfigure_rpu_execution(
            owner, {"prefill": {"chunk_size": 96}}
        )
    )
    reconfigured.start()
    assert not applied.wait(0.05)
    release.set()
    worker.join(5)
    reconfigured.join(5)

    assert not worker.is_alive()
    assert not reconfigured.is_alive()
    assert applied.is_set()
    assert session.generation == 1


def test_reconfigure_from_inside_forward_is_rejected():
    owner = SimpleNamespace()
    _session(owner)
    with execution_guard(owner):
        with pytest.raises(RuntimeError, match="inside a forward"):
            reconfigure_rpu_execution(
                owner, {"prefill": {"chunk_size": 96}}
            )
