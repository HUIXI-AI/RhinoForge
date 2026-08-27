"""Canonical GraphSignature and GraphCache runtime wrappers.

The public home is :mod:`rpu_backend.graph`. Keeping one implementation in a
leaf module preserves canonical wrappers and the ordered shutdown required by
the native queue/runtime lifecycle.
"""
from __future__ import annotations

import threading
import types
import warnings
import weakref
from typing import Any

import torch


_cpp_ext: Any = None
_cpp_loaded = False
_LIVE_GRAPH_CACHES: "weakref.WeakSet[GraphCache]" = weakref.WeakSet()
_graph_cache_tls = threading.local()


def configure_backend(cpp_ext: Any) -> None:
    """Bind the native symbol bag before any real graph object is created."""
    global _cpp_ext, _cpp_loaded
    _cpp_ext = cpp_ext
    _cpp_loaded = cpp_ext is not None


def _graph_fnv1a_64(value: str) -> int:
    """Return the signed int64 FNV1a graph operation identifier."""
    result = 0xCBF29CE484222325
    for byte in value.encode("utf-8"):
        result = ((result ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return result - (1 << 64) if result >= (1 << 63) else result


_GRAPH_DTYPE_CODE = {
    torch.float16: 5,
    torch.bfloat16: 15,
    torch.float32: 6,
    torch.int32: 3,
    torch.int64: 4,
    torch.bool: 11,
}


class GraphSignature:
    """Structured key for one graph-cache entry."""

    def __init__(
        self,
        op_id="",
        shapes=(),
        dyn_dims=(),
        dtypes=(),
        flags=0,
        branch_key=0,
        segment_key=0,
    ):
        self.op_id = str(op_id) if isinstance(op_id, (str, int)) else ""
        if not _cpp_loaded:
            self._impl = None
            return
        self._impl = _cpp_ext.GraphSignature()
        self._impl.op_id = (
            _graph_fnv1a_64(op_id) if isinstance(op_id, str) else int(op_id)
        )
        self._impl.op_id_str = self.op_id
        self._impl.shapes = [int(value) for value in shapes]
        self._impl.dyn_dims = [int(value) for value in dyn_dims]
        self._impl.dtypes = [
            _GRAPH_DTYPE_CODE[value]
            if isinstance(value, torch.dtype)
            else int(value)
            for value in dtypes
        ]
        self._impl.flags = int(flags)
        self._impl.branch_key = int(branch_key)
        self._impl.segment_key = int(segment_key)

    def __repr__(self):
        return (
            repr(self._impl)
            if self._impl is not None
            else "GraphSignature(<no backend>)"
        )

    def __eq__(self, other):
        if not isinstance(other, GraphSignature):
            return NotImplemented
        if self._impl is None or other._impl is None:
            return self._impl is other._impl
        return self._impl == other._impl

    def __hash__(self):
        return hash(self._impl) if self._impl is not None else 0


def _skip_idle_record_function():
    """Skip idle record scopes only when enabled and no profiler is active.

    The environment setting is cached, but profiler state is checked on every
    call so enabling profiling always preserves capture ranges.
    """
    global _SKIP_IDLE_RF_ENV
    if _SKIP_IDLE_RF_ENV is None:
        import os
        v = os.environ.get("RPU_SKIP_IDLE_RECORD_FUNCTION", "")
        _SKIP_IDLE_RF_ENV = v in ("1", "on", "true", "True")
    if not _SKIP_IDLE_RF_ENV:
        return False
    try:
        import torch
        return not torch.autograd._profiler_enabled()
    except Exception:
        return False


_SKIP_IDLE_RF_ENV = None


class _GraphCaptureScope:
    """Exception-safe BUILD/REPLAY scope with profiler attribution."""

    _STATE_BUILT = 2
    _STATE_REPLAYING = 3

    def __init__(self, cache, sig):
        self._cache = cache
        self._sig = sig
        self._graph = None
        self._enter_state = None
        self._prof_ctx = None

    def __enter__(self):
        if self._cache._impl is None or self._sig._impl is None:
            return None
        self._graph = self._cache._graph_for_capture(self._sig)
        self._graph.begin(self._sig._impl)
        self._enter_state = self._graph.state()
        if _skip_idle_record_function():
            # profiler **没启用**时，这个 `record_function` 是纯开销 ——
            # 每次都要建一个 Python 对象 + 走一趟 `_record_function_enter_new`
            # 的 C++ 调用，而 profiler 未启用时它什么也不记。
            # ⚠️ **只在 profiler 确实没启用时跳过** ⇒ 开着 profiler 时行为逐字不变，
            #    trace 里该有的区间一个不少（判据见 `_skip_idle_record_function`）。
            # ⚠️ 共享文件 ⇒ 由 `RPU_SKIP_IDLE_RECORD_FUNCTION` 控制，**全局默认 OFF**。
            self._prof_ctx = None
            return self._graph
        try:
            from torch.profiler import record_function

            name = getattr(self._sig, "op_id", "") or "rpu_graph_capture"
            self._prof_ctx = record_function(name)
            self._prof_ctx.__enter__()
        except ImportError:
            self._prof_ctx = None
        except Exception:
            import logging

            logging.getLogger("rpu_backend").debug(
                "GraphCaptureScope profiler init failed", exc_info=True
            )
            self._prof_ctx = None
        return self._graph

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._graph is None:
                return False
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
                if self._graph.state() == self._STATE_BUILT:
                    self._cache._impl.touch_entry(
                        self._sig._impl,
                        self._enter_state == self._STATE_REPLAYING,
                    )
            else:
                self._graph.abort()
            return False
        finally:
            if self._prof_ctx is not None:
                self._prof_ctx.__exit__(exc_type, exc_val, exc_tb)
                self._prof_ctx = None


class GraphCache:
    """Signature-indexed graph cache with CONFIGURING/WARMING/READY phases."""

    DEFAULT_MAX_ENTRIES = 32
    _CONFIGURING = "CONFIGURING"
    _WARMING = "WARMING"
    _READY = "READY"

    def __init__(self, max_entries=None):
        self._phase = self._CONFIGURING
        if not _cpp_loaded:
            self._impl = None
            return
        if max_entries is None:
            max_entries = self.DEFAULT_MAX_ENTRIES
        self._impl = _cpp_ext.GraphCache(int(max_entries))
        _LIVE_GRAPH_CACHES.add(self)

    def capture(self, sig):
        """Return the main ``with cache.capture(sig)`` scope."""
        return _GraphCaptureScope(self, sig)

    @staticmethod
    def _validate_ready_graph(graph, sig) -> None:
        if graph is None:
            raise RuntimeError(
                f"GraphCache READY miss for {sig!r}; online graph BUILD is disabled"
            )
        if graph.state() != 2:
            raise RuntimeError(
                f"GraphCache READY entry is not BUILT for {sig!r} "
                f"(state={graph.state()}); online recapture is disabled"
            )
        if not graph.replayable():
            raise RuntimeError(
                f"GraphCache READY entry is non-replayable for {sig!r}"
            )
        if not graph.has_built_signature() or graph.built_signature() != sig._impl:
            raise RuntimeError(
                f"GraphCache READY entry has no matching built signature for {sig!r}"
            )

    def _graph_for_capture(self, sig):
        if self._impl is None:
            return None
        if self._phase != self._READY:
            return self._impl.get_or_create(sig._impl)
        graph = self._impl.lookup(sig._impl)
        self._validate_ready_graph(graph, sig)
        return graph

    def begin_warmup(self):
        """Enter WARMING and explicitly allow graph BUILD/recapture."""
        self._phase = self._WARMING
        return self

    def freeze(self):
        """Enter lookup-only READY after validating all retained graphs."""
        if self._impl is None:
            raise RuntimeError("GraphCache.freeze requires the C++ backend")
        entries = self._impl.snapshot()
        if not entries:
            raise RuntimeError(
                "GraphCache.freeze requires at least one prewarmed BUILT entry"
            )
        for entry in entries:
            graph = self._impl.lookup(entry.signature)
            sig = types.SimpleNamespace(_impl=entry.signature, op_id="")
            self._validate_ready_graph(graph, sig)
        self._phase = self._READY
        return self

    def is_frozen(self) -> bool:
        return self._phase == self._READY

    @property
    def phase(self) -> str:
        return self._phase

    def get_or_create(self, sig):
        return self._graph_for_capture(sig)

    def lookup(self, sig):
        return self._impl.lookup(sig._impl) if self._impl else None

    def evict(self, sig):
        if self.is_frozen():
            raise RuntimeError(
                "GraphCache READY entries cannot be evicted; "
                "call begin_warmup() first"
            )
        if self._impl is not None:
            self._impl.evict(sig._impl)

    def clear(self):
        if self._impl is not None:
            self._impl.clear()

    def size(self) -> int:
        return self._impl.size() if self._impl is not None else 0

    def max_entries(self) -> int:
        return self._impl.max_entries() if self._impl is not None else 0

    def snapshot(self):
        return self._impl.snapshot() if self._impl is not None else []

    def cache_invariant_ok(self) -> bool:
        return self._impl.cache_invariant_ok() if self._impl is not None else True

    def touch_entry(self, sig, replay: bool) -> None:
        if self._impl is not None:
            self._impl.touch_entry(sig._impl, bool(replay))

    def debug_bucket_counts(self) -> dict:
        return dict(self._impl.debug_bucket_counts()) if self._impl is not None else {}

    def debug_branch_counts(self) -> dict:
        return dict(self._impl.debug_branch_counts()) if self._impl is not None else {}

    def dump_signature_tree(self) -> str:
        return self._impl.dump_signature_tree() if self._impl is not None else "<empty>"

    def explain_miss(self, sig) -> str:
        if self._impl is None:
            return "miss at GmId: backend unavailable"
        return self._impl.explain_miss(sig._impl)

    @classmethod
    def clear_all(cls) -> None:
        """Release live cache queues before native runtime teardown."""
        for cache in list(_LIVE_GRAPH_CACHES):
            try:
                cache.clear()
            except Exception as exc:
                warnings.warn(
                    "GraphCache.clear_all: clear failed for one instance: "
                    f"{type(exc).__name__}: {exc}",
                    stacklevel=2,
                )


def get_default_graph_cache() -> GraphCache:
    """Return the thread-local default production GraphCache."""
    cached = getattr(_graph_cache_tls, "cached", None)
    if cached is not None:
        return cached
    cache = GraphCache()
    _graph_cache_tls.cached = cache
    return cache


def install_torch_namespace(rpu_module: Any) -> None:
    """Install the canonical wrappers on ``torch.rpu``."""
    if not _cpp_loaded:
        return
    rpu_module.Graph = _cpp_ext.Graph
    rpu_module.GraphCache = GraphCache
    rpu_module.GraphSignature = GraphSignature
    rpu_module.get_default_graph_cache = get_default_graph_cache
    if hasattr(_cpp_ext, "SignatureLayer"):
        rpu_module.SignatureLayer = _cpp_ext.SignatureLayer
        rpu_module.LayeredSignature = _cpp_ext.LayeredSignature
        rpu_module.signature_layer_name = _cpp_ext.signature_layer_name
        rpu_module.layered_from = _cpp_ext.layered_from


GraphSignature.__module__ = "rpu_backend.graph"
GraphCache.__module__ = "rpu_backend.graph"


__all__ = [
    "GraphCache",
    "GraphSignature",
    "configure_backend",
    "get_default_graph_cache",
    "install_torch_namespace",
]
