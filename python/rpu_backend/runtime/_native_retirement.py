"""Graph-first retirement shared by the decoder and Vision installers."""
from __future__ import annotations

import weakref
from contextlib import nullcontext


_FAILED_RETIREMENTS = []


class _InstalledNativeResource:
    """Keep graph addresses and weights alive until one successful destroy.

    The ordinary finalizer must not retain the model (or its Session): doing
    so would prevent that finalizer from ever running. Failed retirement is
    intentionally different: retain the complete failed installation forever.
    """

    def __init__(self, owner, handle, destroy, *, graphs, keepalive, label, handle_name,
                 raw_graphs=()):
        self.owner = weakref.ref(owner)
        self.handle = handle
        self.destroy = destroy
        self.graphs = tuple(graph for graph in graphs if graph is not None)
        self.raw_graphs = tuple(graph for graph in raw_graphs if graph is not None)
        self.keepalive = keepalive
        self.label = label
        self.handle_name = handle_name
        self.failed = None
        self.parent = None
        self.finalizer = weakref.finalize(owner, self._gc_retire)

    def take_ownership(self, parent):
        owner = self.owner()
        if (parent is None or owner is None or self.handle is None or self.failed is not None
                or getattr(owner, self.handle_name, None) != self.handle):
            raise RuntimeError(f"{self.label}: retirement handle identity changed")
        if self.parent is not None and self.parent() is not parent:
            raise RuntimeError(f"{self.label}: different retirement owner")
        try:
            self.parent = weakref.ref(parent)
        except TypeError:
            # Existing partial-build rollback capsules can be SimpleNamespace.
            # They have no GC callback; their explicit parent retires them.
            self.parent = lambda: parent
        if self.finalizer.alive:
            self.finalizer.detach()

    def require_replaceable(self):
        from rpu_backend.api._execution import _require_execution_process_safe

        _require_execution_process_safe()
        if self.failed is not None:
            raise RuntimeError(f"{self.label}: previous retirement failed") from self.failed
        if (self.handle is not None
                and getattr(self.owner(), self.handle_name, None) != self.handle):
            raise RuntimeError(f"{self.label}: retirement handle identity changed")
        if self.parent is not None:
            if self.parent() is None:
                raise RuntimeError(f"{self.label}: retirement parent was lost")
            raise RuntimeError(
                f"{self.label}: parent-owned resource must be replaced through its parent"
            )
        session = getattr(self.owner(), "_execution_session", None)
        if session is not None:
            session.require_cold()

    def retain_failure(self, error, *keepalive):
        from rpu_backend.api._execution import _mark_execution_process_unsafe

        if self.failed is None:
            self.failed = error
            _FAILED_RETIREMENTS.append(self)
        self.keepalive = (self.keepalive, self.owner(), keepalive)
        parent = self.parent() if self.parent is not None else self.owner()
        session = getattr(parent, "_execution_session", None)
        if session is not None:
            session.poison()
        _mark_execution_process_unsafe(f"{self.label} retirement failed: {error}")

    def retire(self):
        from rpu_backend.api._execution import _require_execution_process_safe

        if self.failed is not None:
            raise RuntimeError(f"{self.label}: retirement already failed") from self.failed
        if self.handle is None:
            return
        try:
            _require_execution_process_safe()
            seen = set()
            for graph in self.graphs:
                if id(graph) not in seen:
                    seen.add(id(graph))
                    graph.clear()
                    if not graph.cache_invariant_ok():
                        raise RuntimeError(f"{self.label}: GraphCache invariant failed")
            for graph in self.raw_graphs:
                if id(graph) not in seen:
                    seen.add(id(graph))
                    graph.invalidate()
            self.destroy(self.handle)
        except BaseException as error:
            self.retain_failure(error)
            raise
        self._native_destroyed(self.handle)

    def retire_owned(self, owner):
        """Explicit leaf retirement under its existing standalone/parent Session."""
        if self.parent is not None and self.handle is not None:
            raise RuntimeError(f"{self.label}: owned child must retire through its parent")

        def retire():
            if self.owner() is not owner or getattr(owner, self.handle_name, None) != self.handle:
                error = RuntimeError(f"{self.label}: retirement handle identity changed")
                self.retain_failure(error, owner)
                raise error
            self.retire()

        session = getattr(owner, "_execution_session", None)
        if session is None:
            retire()
        else:
            with session._lock:
                if session._active:
                    raise RuntimeError(f"cannot destroy {self.label} during a forward")
                if session._owner is owner:
                    session.shutdown(retire)
                else:
                    retire()
        if self.handle is not None:
            error = RuntimeError(f"{self.label}: closed Session still owns native resources")
            self.retain_failure(error, owner)
            raise error

    def _native_destroyed(self, handle):
        """Record the exact successful raw destroy performed by this owner."""
        if self.failed is not None or self.handle != handle or handle is None:
            raise RuntimeError(f"{self.label}: destroyed handle identity changed")
        owner = self.owner()
        if owner is not None and getattr(owner, self.handle_name, None) == self.handle:
            vars(owner)[self.handle_name] = None
        self.handle = None
        if self.finalizer.alive:
            self.finalizer.detach()
        self.graphs = ()
        self.raw_graphs = ()
        self.keepalive = ()

    def cleanup_failure(self, error, *keepalive):
        """Preserve the install exception; cleanup failure is an attached note."""
        try:
            self.retire()
        except BaseException as cleanup_error:
            self.retain_failure(cleanup_error, *keepalive)
            error.add_note(f"{self.label} cleanup failed: {cleanup_error!r}")
            return False
        return True

    def _gc_retire(self):
        # Weakref callbacks may run after the old process slot was handed to
        # a new owner. Never touch that owner's runtime on a late callback.
        from rpu_backend.api import causal_lm

        owner = self.owner()
        session = getattr(owner, "_execution_session", None)
        with session._lock if session is not None else nullcontext():
            with causal_lm._LIVE_LOCK:
                live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
                if (owner is None and live is not None) or (
                        session is not None and session.stats()["active"]):
                    self.retain_failure(RuntimeError(
                        f"{self.label}: live-owner handoff or active execution before retirement"))
                    return
                try:
                    self.retire()
                except BaseException:
                    # Failure is latched and strongly retained by retire().
                    # GC must never retry an uncertain native destroy.
                    pass
