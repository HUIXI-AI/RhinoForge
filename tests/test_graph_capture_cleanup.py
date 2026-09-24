"""Graph failure cleanup must release only the scope's borrowed reference."""
from types import SimpleNamespace
import weakref

import pytest


@pytest.fixture
def retained_capture(monkeypatch):
    from rpu_backend.graph import _runtime as runtime

    events = []
    faults = {}

    class Entry:
        def __init__(self):
            self.borrowed = weakref.WeakSet()
            self.status = 0

        def begin(self, sig):
            events.append("begin")
            if "begin" in faults:
                raise faults["begin"]
            self.status = 1

        def end(self):
            events.append("end")
            if "end" in faults:
                raise faults["end"]
            self.status = 2

        def abort(self):
            events.append("abort")
            self.status = 0
            if "abort" in faults:
                raise faults["abort"]

        def state(self):
            if "state" in faults:
                raise faults["state"]
            return self.status

    class BorrowedGraph:
        def __init__(self, entry):
            entry.borrowed.add(self)
            # Like a native method, a failing operation's Python traceback must
            # not itself own this external shared_ptr wrapper.
            for name in ("begin", "end", "abort", "state"):
                setattr(self, name, getattr(entry, name))

    class NativeCache:
        def __init__(self):
            self.entries = {}

        def get_or_create(self, sig):
            return BorrowedGraph(self.entries.setdefault(sig, Entry()))

        def touch_entry(self, sig, replay):
            events.append("touch")

        def evict(self, sig):
            if self.entries[sig].status == 1:
                raise RuntimeError("cannot evict an active graph")
            if self.entries[sig].borrowed:
                raise RuntimeError("cannot evict a graph with external references")
            del self.entries[sig]
            events.append("evict")

        def clear(self):
            if any(entry.status == 1 for entry in self.entries.values()):
                raise RuntimeError("cannot clear an active graph")
            if any(entry.borrowed for entry in self.entries.values()):
                raise RuntimeError("cannot clear a graph with external references")
            self.entries.clear()
            events.append("clear")

    class Profiler:
        def __enter__(self):
            if "profiler-enter" in faults:
                raise faults["profiler-enter"]
            return self

        def __exit__(self, *args):
            events.append("profiler-exit")
            if "profiler" in faults:
                raise faults["profiler"]

    def skip_idle():
        if "skip-idle" in faults:
            raise faults["skip-idle"]
        return False

    monkeypatch.setattr(runtime, "_skip_idle_record_function", skip_idle)
    monkeypatch.setattr("torch.profiler.record_function", lambda _: Profiler())
    cache = runtime.GraphCache()
    cache._impl = NativeCache()
    sig = SimpleNamespace(_impl="retained-failure", op_id="cpu-reference-lifetime")
    return SimpleNamespace(cache=cache, sig=sig, faults=faults, events=events)


@pytest.mark.parametrize("phase,abort_failure,profiler_failure", [
    ("begin", False, False),
    ("body", True, True),
    ("end", True, True),
    ("profiler", False, False),
    ("state", True, False),
    ("skip-idle", False, False),
    ("profiler-enter", False, False),
])
def test_failed_capture_releases_scope_reference_while_primary_traceback_is_alive(
    retained_capture, phase, abort_failure, profiler_failure,
):
    case = retained_capture
    if phase in {"state", "skip-idle", "profiler-enter"}:
        primary = KeyboardInterrupt(phase + " interrupted")
    else:
        primary = RuntimeError("build_batch SDK rc=5" if phase == "end" else phase + " failed")
    if phase != "body":
        case.faults[phase] = primary
    if abort_failure:
        case.faults["abort"] = RuntimeError("abort failed")
    if profiler_failure:
        case.faults["profiler"] = RuntimeError("profiler failed")
    scope = case.cache.capture(case.sig)
    try:
        with scope:
            if phase == "body":
                raise primary
    except BaseException as error:
        assert error is primary
        assert error.__traceback__ is not None
        notes = getattr(error, "__notes__", ())
        assert any("abort" in note.lower() for note in notes) == abort_failure
        assert any("profiler" in note.lower() for note in notes) == profiler_failure
        # The retained traceback and scope are intentionally alive here. These
        # are the real wrapper calls made by Vision's except and final close.
        case.cache.evict(case.sig)
        case.cache.clear()
    else:
        pytest.fail("capture should fail")
    assert not case.cache._impl.entries


def test_profiler_entry_exception_keeps_existing_nonfatal_capture_behavior(retained_capture):
    case = retained_capture
    case.faults["profiler-enter"] = RuntimeError("optional profiler unavailable")
    scope = case.cache.capture(case.sig)
    with scope:
        pass
    case.cache.evict(case.sig)
    case.cache.clear()
    assert case.events == ["begin", "end", "touch", "evict", "clear"]


def test_capture_releases_only_its_own_reference(retained_capture):
    case = retained_capture
    scope = case.cache.capture(case.sig)
    with scope as borrowed:
        pass
    # A caller that retains the returned graph remains a real external owner.
    with pytest.raises(RuntimeError, match="external references"):
        case.cache.evict(case.sig)
    with pytest.raises(RuntimeError, match="external references"):
        case.cache.clear()
    del borrowed
    case.cache.evict(case.sig)
    case.cache.clear()
    assert case.events == ["begin", "end", "touch", "profiler-exit", "evict", "clear"]


def test_graph_and_cache_policy_reference_is_read_only(monkeypatch):
    import rpu_backend.graph as graph
    from rpu_backend.graph import _runtime

    monkeypatch.setattr(graph, "so_path", None)
    monkeypatch.setattr(_runtime, "_cpp_loaded", False)
    policy = graph.GraphRuntimePolicy(force_oneshot_on_replay=True)
    for owner in (graph.Graph(runtime_policy=policy), graph.GraphCache(runtime_policy=policy)):
        with pytest.raises(AttributeError):
            owner.runtime_policy = graph.GraphRuntimePolicy()
        assert owner.runtime_policy is policy
        assert owner.runtime_policy.force_oneshot_on_replay
