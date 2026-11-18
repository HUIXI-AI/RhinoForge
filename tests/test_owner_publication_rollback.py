"""Late facade publication must compensate real composite owner state."""
from contextlib import nullcontext
from types import SimpleNamespace
import ast
from pathlib import Path

import pytest

from rpu_backend.api._execution import bind_execution_session
from rpu_backend.adapters.internvla_n1 import execution as intern
from rpu_backend.adapters.hy_vla import runtime as hy


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


class Graph:
    def __init__(self):
        self.clears = 0

    def begin_warmup(self):
        pass

    def clear(self):
        self.clears += 1

    def cache_invariant_ok(self):
        return True


@pytest.mark.parametrize("relative,function", [
    ("wall_oss/runtime.py", "apply_execution_reconfigure"),
    ("gr00t/runtime.py", "apply_execution_reconfigure"),
    ("rhinovla/runtime.py", "apply"),
    ("qwen3_vl/__init__.py", "apply_execution_reconfigure"),
    ("qwen3_5/__init__.py", "_apply_qwen3_5_execution_config"),
    ("pi05/__init__.py", "apply_execution_reconfigure"),
    ("g05/runtime.py", "_apply_g05_execution_config"),
    ("lingbot_vla/runtime.py", "apply_execution_reconfigure"),
    ("lingbot_vla_v2/runtime.py", "apply_execution_reconfigure"),
    ("hy_vla/runtime.py", "apply_execution_reconfigure"),
    ("internvla_n1/execution.py", "apply"),
])
def test_owner_apply_does_not_retire_undo_before_shared_publication(relative, function):
    path = Path(__file__).resolve().parents[1] / "python/rpu_backend/adapters" / relative
    tree = ast.parse(path.read_text())
    node, = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name == function]
    for statement in node.body[1:]:
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Constant):
            assert not (statement.value.value is None and any(
                isinstance(target, ast.Attribute) and "journal" in target.attr
                for target in statement.targets)), (relative, statement.lineno)


@pytest.mark.parametrize("family", ["intern", "hy"])
def test_owner_undo_survives_shared_publication_interruption(monkeypatch, family):
    graph = Graph()
    if family == "intern":
        owner = SimpleNamespace()
        child = SimpleNamespace(_gc=graph)
        component = intern.NAVDP_COMPONENT
        controller = intern.InternVLAExecutionController(owner, {component: child})
        session = controller.session
        patch = {"components": {component: {"action": {"chunk_size": 32}}}}
        old_child = child._rpu_execution
        def assert_restored():
            assert child._rpu_execution == old_child
            assert child._fmb_execution_generation == 0
            assert controller._generation == 0
            assert controller._component_generations[component] == 0
    else:
        owner = hy.HyVlaRunner.__new__(hy.HyVlaRunner)
        root, children = hy.resolve_hy_vla_execution(None, entry_point="test")
        component = hy.HY_VLA_LANGUAGE_COMPONENT
        owner._closed = False
        owner._rpu_execution = root
        owner._rpu_execution_components = children
        owner._rpu_execution_component_generations = {key: 0 for key in children}
        owner._rpu_last_execution_receipts = {}
        owner._psubs = {"vlm": SimpleNamespace(handle=None, gc=graph)}
        owner.validate_execution_reconfigure = lambda config: None
        session = bind_execution_session(
            owner, root, entry_point="test Hy",
            supported_components=hy.HY_VLA_EXECUTION_COMPONENTS,
            apply=owner.apply_execution_reconfigure,
            rollback=owner.rollback_execution_reconfigure,
            graph_mode="COMPOSITE_CHILD",
        )
        patch = {"components": {component: {"prefill": {"chunk_size": 240}}}}
        def assert_restored():
            assert owner._rpu_execution_components == children
            assert owner._rpu_execution_component_generations == {key: 0 for key in children}
    view = View()
    session.register_config_view(view)
    old = session.config
    old_owner_config = owner._rpu_execution
    view.storage.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        session.reconfigure(patch)
    assert session.config is old and session.generation == 0
    assert session.stats()["state"] == "QUIESCENT"
    assert owner._rpu_execution is old_owner_config
    assert graph.clears == 2  # changed Graph retired on apply and compensation
    assert_restored()
    # A later successful transaction must replace, not reuse, the old journal.
    session.reconfigure(patch)
    assert session.generation == 1 and graph.clears == 3
