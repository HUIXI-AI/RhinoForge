# Graph / GraphSignature / GraphCache Python wrappers.
#
# This module is configured by rpu_backend.__init__ after the C++ extension
# has been loaded. It intentionally keeps the public torch.rpu surface stable
# while keeping package initialisation small. Admission policy lives in
# admission.py; legacy submodule names are aliased below for compatibility.

import sys as _sys

from . import admission as _admission
from ._runtime import (
    GraphCache,
    GraphSignature,
    configure_backend as _configure_runtime_backend,
    get_default_graph_cache,
)


_sys.modules[__name__ + ".plan"] = _admission
_sys.modules[__name__ + ".signature_planner"] = _admission
_sys.modules[__name__ + ".entry_registry"] = _admission


rpu_backend = None
so_path = None


def configure_backend(backend, backend_path):
    """Bind the loaded C++ extension module used by these wrappers."""
    global rpu_backend, so_path
    rpu_backend = backend
    so_path = backend_path
    _configure_runtime_backend(backend)


class _GraphScope:
    """Context manager for graph capture/replay scope.

    Passing a signature enables signature-driven admission; omitting it keeps
    the compatibility path that records each scope independently.
    """
    def __init__(self, graph, sig=None):
        self._graph = graph
        self._sig = sig

    def __enter__(self):
        self._graph.begin(self._sig)
        return self._graph

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            try:
                self._graph.end()
            except BaseException as end_error:
                try:
                    self._graph.abort()
                except BaseException as abort_error:
                    if hasattr(end_error, "add_note"):
                        end_error.add_note(
                            "Graph abort after end failure also failed: "
                            f"{abort_error!r}"
                        )
                raise
        else:
            self._graph.abort()
        return False


class Graph:
    """RPU KernelGraph — capture/replay execution graph.

    Usage:
        from rpu_backend.graph import Graph

        g = Graph()
        with g.capture():
            output = model(input)
    """
    def __init__(self):
        if so_path:
            self._impl = rpu_backend.Graph()
        else:
            self._impl = None

    def capture(self, sig: "GraphSignature | None" = None):
        """Return a context manager for graph capture/replay scope.

        Passing a signature enables signature-driven admission; omitting it
        uses the compatibility recording path.
        """
        return _GraphScope(self, sig=sig)

    def begin(self, sig: "GraphSignature | None" = None):
        """Open graph scope (low-level). Prefer capture()."""
        if self._impl is None:
            return
        if sig is None:
            self._impl.begin()
        else:
            backend_sig = sig._impl if isinstance(sig, GraphSignature) else sig
            self._impl.begin(backend_sig)

    def end(self):
        """Close graph scope and submit."""
        if self._impl is not None:
            self._impl.end()

    def abort(self):
        """Abort scope on error, cleanup state."""
        if self._impl is not None:
            self._impl.abort()

    def invalidate(self):
        """Discard graph and prepared batch."""
        if self._impl is not None:
            self._impl.invalidate()

    def state(self) -> int:
        """Get state (0=PASSTHROUGH,1=RECORDING,2=BUILT,3=REPLAYING)."""
        if self._impl is not None:
            return self._impl.state()
        return 0

    def graph_size(self) -> int:
        """Get number of captured nodes."""
        if self._impl is not None:
            return self._impl.graph_size()
        return 0

    def replayable(self) -> bool:
        """Check if graph is replayable."""
        if self._impl is not None:
            return self._impl.replayable()
        return False

    def has_built_signature(self) -> bool:
        """Return whether this graph has a built signature for replay."""
        if self._impl is not None:
            return self._impl.has_built_signature()
        return False

    def built_signature(self):
        """Return the backend GraphSignature recorded at the last build."""
        if self._impl is not None:
            return self._impl.built_signature()
        return None

    def debug_stats(self):
        """Return last execute_graph_* statistics."""
        if self._impl is not None:
            return self._impl.debug_stats()
        return None

def layered_from(sig: "GraphSignature"):
    """Return the layered diagnostic signature for a GraphSignature."""
    if so_path is None:
        return None
    backend_sig = sig._impl if isinstance(sig, GraphSignature) else sig
    return rpu_backend.layered_from(backend_sig)


# Dynamo lazy-init guard hazard preflight.
#
# Keep this genuinely lazy: the graph package is wired whenever the native
# backend is available, but importing the backend must not pull Dynamo's model
# inspection helpers into a cold process.  ``from rpu_backend.graph import
# freeze_for_dynamo`` still works through PEP 562.
_LAZY_DYNAMO_EXPORTS = frozenset({
    "verify_lazy_init",
    "UninitializedLazyStateError",
    # Misnomers kept for the packaged runtime API; see lazy_init_guard.py.
    "freeze_for_dynamo",
    "DynamoUnsafeLazyStateError",
})


def __getattr__(name):
    if name in _LAZY_DYNAMO_EXPORTS:
        from . import lazy_init_guard as _lazy_init_guard
        return getattr(_lazy_init_guard, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Graph",
    "GraphCache",
    "GraphSignature",
    "get_default_graph_cache",
    "layered_from",
]
