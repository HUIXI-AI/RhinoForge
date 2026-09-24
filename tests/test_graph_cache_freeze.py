"""Hardware-free contract tests for GraphCache WARMING -> READY freeze."""
import ast
from pathlib import Path
from types import SimpleNamespace
import types

import pytest


def _load_graph_cache_class():
    source = (
        Path(__file__).resolve().parents[1]
        / "python"
        / "rpu_backend"
        / "graph"
        / "_runtime.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {"_GraphCaptureScope", "GraphCache"}
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    assert {node.name for node in classes} == names
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_skip_idle_record_function"
    )
    cache = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_SKIP_IDLE_RF_ENV"
            for target in node.targets
        )
    )
    namespace = {
        "_cpp_loaded": False,
        "types": types,
        "warnings": __import__("warnings"),
    }
    exec(
        compile(
            ast.Module(body=[helper, cache, *classes], type_ignores=[]),
            filename="rpu_backend/graph/_runtime.py",
            mode="exec",
        ),
        namespace,
    )
    return namespace["GraphCache"]


GraphCache = _load_graph_cache_class()


class _FakeGraph:
    def __init__(self):
        self._state = 0
        self._built_sig = None
        self._pending_sig = None
        self._replayable = True

    def begin(self, sig):
        if self._state == 2 and self._built_sig == sig and self._replayable:
            self._state = 3
        else:
            self._pending_sig = sig
            self._state = 1

    def end(self):
        if self._state == 1:
            self._built_sig = self._pending_sig
        self._state = 2

    def abort(self):
        self._state = 0

    def state(self):
        return self._state

    def replayable(self):
        return self._replayable

    def has_built_signature(self):
        return self._built_sig is not None

    def built_signature(self):
        return self._built_sig


class _FakeImpl:
    def __init__(self):
        self.entries = {}
        self.touches = []

    def get_or_create(self, sig):
        return self.entries.setdefault(sig, _FakeGraph())

    def lookup(self, sig):
        return self.entries.get(sig)

    def snapshot(self):
        return [SimpleNamespace(signature=sig) for sig in self.entries]

    def size(self):
        return len(self.entries)

    def max_entries(self):
        return 32

    def clear(self):
        self.entries.clear()

    def evict(self, sig):
        self.entries.pop(sig, None)

    def touch_entry(self, sig, replay):
        self.touches.append((sig, bool(replay)))


def _sig(raw):
    return SimpleNamespace(_impl=raw, op_id=str(raw))


def _cache():
    from test_execution_planner import _MODULE as planner

    cache = GraphCache.__new__(GraphCache)
    cache._impl = _FakeImpl()
    cache._phase = cache._CONFIGURING
    cache._execution_plans = planner.PreparedExecutionPlans()
    return cache


def test_ready_replays_prebuilt_profile_and_rejects_miss():
    cache, sig = _cache(), _sig("profile-a")
    cache.begin_warmup()
    plan = cache.prepare_plan(("shape", 64), lambda: SimpleNamespace(signature=sig))
    with cache.capture(sig) as graph:
        assert graph.state() == 1
    assert graph.state() == 2 and cache.size() == 1

    cache.freeze()
    assert cache.is_frozen() and cache.phase == "READY"
    with cache.capture(sig) as graph:
        assert graph.state() == 3
    assert graph.state() == 2 and cache.size() == 1

    with pytest.raises(RuntimeError, match="READY miss"):
        with cache.capture(_sig("profile-b")):
            pass
    assert cache.size() == 1

    def unexpected_replan():
        pytest.fail("READY must reuse the existing physical plan")

    retained_plan = cache.prepare_plan(("shape", 64), unexpected_replan)
    assert retained_plan is plan
    with cache.capture(retained_plan.signature) as graph:
        assert graph.state() == 3
    assert cache.is_frozen() and cache.size() == 1
    assert cache._impl.touches == [
        (sig._impl, False), (sig._impl, True), (sig._impl, True)
    ]


def test_ready_rejects_wall_oss_prefill_policy_version_change():
    cache = _cache()
    version_2 = _sig(("wall_oss_llm_embeds", 640, 2048, 320, 36, 2))
    version_3 = _sig(("wall_oss_llm_embeds", 640, 2048, 320, 36, 3))

    cache.begin_warmup()
    with cache.capture(version_2):
        pass
    cache.freeze()

    with pytest.raises(RuntimeError, match="READY miss"):
        with cache.capture(version_3):
            pass
    assert cache.size() == 1


def test_default_configuring_behavior_still_builds_new_signatures():
    cache = _cache()
    assert cache.phase == "CONFIGURING" and not cache.is_frozen()
    with cache.capture(_sig("profile-a")):
        pass
    with cache.capture(_sig("profile-b")):
        pass
    assert cache.size() == 2


def test_ready_rejects_invalidated_or_nonreplayable_entry():
    cache, sig = _cache(), _sig("profile-a")
    with cache.capture(sig):
        pass
    cache.freeze()

    cache.lookup(sig)._state = 0
    with pytest.raises(RuntimeError, match="not BUILT"):
        with cache.capture(sig):
            pass

    cache.begin_warmup()
    cache.lookup(sig)._state = 2
    cache.lookup(sig)._replayable = False
    with pytest.raises(RuntimeError, match="non-replayable"):
        cache.freeze()


def test_clear_does_not_unfreeze_and_begin_warmup_explicitly_unfreezes():
    cache, sig = _cache(), _sig("profile-a")
    with cache.capture(sig):
        pass
    cache.freeze()
    cache.clear()
    assert cache.is_frozen()

    with pytest.raises(RuntimeError, match="READY miss"):
        with cache.capture(sig):
            pass

    cache.begin_warmup()
    assert not cache.is_frozen() and cache.phase == "WARMING"
    with cache.capture(sig):
        pass
    assert cache.size() == 1


def test_freeze_requires_a_prebuilt_entry():
    with pytest.raises(RuntimeError, match="at least one prewarmed BUILT"):
        _cache().freeze()


@pytest.mark.parametrize("failure", ["begin", "body", "end"])
def test_failed_capture_discards_prepared_plan_and_retry_prepares_again(monkeypatch, failure):
    cache, sig = _cache(), _sig("prepared")
    first = cache.prepare_plan(("shape", 64), object)
    if failure == "begin":
        graph = cache._impl.get_or_create(sig._impl)
        begin = graph.begin

        def fail_begin(signature):
            begin(signature)
            raise RuntimeError("capture failed")

        monkeypatch.setattr(graph, "begin", fail_begin)
    elif failure == "end":
        graph = cache._impl.get_or_create(sig._impl)
        def fail_end():
            raise RuntimeError("capture failed")
        monkeypatch.setattr(graph, "end", fail_end)
    with pytest.raises(RuntimeError, match="capture failed"):
        with cache.capture(sig):
            if failure == "body":
                raise RuntimeError("capture failed")
    assert len(cache._execution_plans) == 0
    assert cache.prepare_plan(("shape", 64), object) is not first
