"""An explicitly closed policy needs no imports during interpreter shutdown."""
import builtins

from rpu_backend.adapters.rhinovla.pipeline import RhinoVLAOnRPU
from rpu_backend.adapters.rhinovla import runtime


def test_closed_owner_finalizes_after_imports_are_unavailable(monkeypatch):
    owner = object.__new__(RhinoVLAOnRPU)
    owner._rhinovla_closed = True

    def unavailable(*args, **kwargs):
        raise ImportError("sys.meta_path is None, Python is likely shutting down")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", unavailable)
        owner.__del__()


def test_live_owner_still_delegates_to_parent_gc_retirement(monkeypatch):
    owner = object.__new__(RhinoVLAOnRPU)
    calls = []

    def retire(value):
        calls.append(value)
        value._rhinovla_closed = True

    monkeypatch.setattr(runtime, "gc_close_rhinovla_runtime", retire)
    owner.__del__()
    assert calls == [owner]
    owner.__del__()
    assert calls == [owner]
