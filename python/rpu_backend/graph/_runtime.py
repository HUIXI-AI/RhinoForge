"""Canonical GraphSignature and GraphCache runtime wrappers.

The public home is :mod:`rpu_backend.graph`.  Keeping the implementation in a
single leaf module avoids the historical duplicate wrappers in package
bootstrap and ``graph.__init__`` while preserving the ordered shutdown needed
by the native queue/runtime lifecycle.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
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


def _graph_policy_bool(name: str, *, default: bool = False) -> tuple[bool, str]:
    raw = os.environ.get(name)
    if raw is None:
        return default, f"{name}=<default:{int(default)}>"
    value = raw.strip().lower()
    if value in {"1", "true", "on"}:
        return True, f"{name}={raw!r}"
    if value in {"", "0", "false", "off"}:
        return False, f"{name}={raw!r}"
    raise ValueError(
        f"{name} must be one of 1/true/on/0/false/off/empty; got {raw!r}"
    )


def _graph_policy_exact_one(name: str) -> tuple[bool, str]:
    raw = os.environ.get(name)
    if raw is None or raw == "0":
        return False, f"{name}={raw!r}"
    if raw == "1":
        return True, f"{name}='1'"
    raise ValueError(f"{name} must be exactly '0' or '1'; got {raw!r}")


def _graph_host_op_defer_mode() -> tuple[int, str]:
    primary = os.environ.get("RPU_GRAPH_HOST_OP_DEFER_GATE")
    legacy = os.environ.get("RPU_GRAPH_DEFER_TO_COPY")
    if primary is not None and legacy is not None:
        raise ValueError(
            "RPU_GRAPH_HOST_OP_DEFER_GATE and legacy "
            "RPU_GRAPH_DEFER_TO_COPY cannot both be set"
        )
    name = (
        "RPU_GRAPH_HOST_OP_DEFER_GATE"
        if primary is not None
        else "RPU_GRAPH_DEFER_TO_COPY"
    )
    raw = primary if primary is not None else legacy
    if raw is None or raw.strip().lower() in {"", "auto"}:
        return 0, f"{name}=<auto>"
    value = raw.strip().lower()
    if value in {"0", "false", "off"}:
        return 1, f"{name}={raw!r}"
    if value in {"1", "true", "on"}:
        return 2, f"{name}={raw!r}"
    raise ValueError(f"{name} must be auto/on/off or a boolean token; got {raw!r}")


def _lkn_load_snapshot() -> tuple[int, int, int]:
    if not _cpp_loaded:
        return 65_536, 8, 64
    values = tuple(int(value) for value in _cpp_ext.get_lkn_batch_config())
    if len(values) != 6:
        raise RuntimeError("native LKN load snapshot must contain six integers")
    loaded = values[:3]
    limits = (1 << 22, 256, 1024)
    if any(value <= 0 or value > limit for value, limit in zip(loaded, limits)):
        raise RuntimeError(f"native LKN load snapshot is out of range: {loaded!r}")
    return loaded


@dataclass(frozen=True)
class GraphRuntimePolicy:
    """Cold, typed policy shared by a GraphCache and every Graph it creates."""

    fast_replay_skip_sync: bool = True
    force_oneshot_on_replay: bool = False
    host_op_defer_mode: int = 0
    siglip_isolate_patch_embed: bool = False
    lingbot2_multiview_spm_z2: bool = False
    fmb_fast_replay: bool = False
    fmb_deep_fast_replay: bool = False
    lkn_max_batch_entries: int = 65_536
    lkn_kd_buf_mb: int = 8
    lkn_instr_buf_mb: int = 64
    provenance: tuple[str, ...] = ()
    qwen35_legacy_27b_sdk_budget: bool = False
    execution_core_count: int = 8
    graph_arena_count: int = 0

    def __post_init__(self) -> None:
        for name in (
            "fast_replay_skip_sync",
            "force_oneshot_on_replay",
            "siglip_isolate_patch_embed",
            "lingbot2_multiview_spm_z2",
            "fmb_fast_replay",
            "fmb_deep_fast_replay",
            "qwen35_legacy_27b_sdk_budget",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if (
            type(self.host_op_defer_mode) is not int
            or self.host_op_defer_mode not in (0, 1, 2)
        ):
            raise ValueError("host_op_defer_mode must be 0 (auto), 1 (off), or 2 (on)")
        for name, limit in (
            ("lkn_max_batch_entries", 1 << 22),
            ("lkn_kd_buf_mb", 256),
            ("lkn_instr_buf_mb", 1024),
            ("execution_core_count", 8),
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0 or value > limit:
                raise ValueError(f"{name} must be an integer in [1, {limit}]")
        if not isinstance(self.provenance, tuple) or not all(
            isinstance(value, str) for value in self.provenance
        ):
            raise TypeError("provenance must be a tuple of strings")
        if type(self.graph_arena_count) is not int or not 0 <= self.graph_arena_count <= 4096:
            raise ValueError("graph_arena_count must be an integer in [0, 4096]")
        if self.qwen35_legacy_27b_sdk_budget and self.graph_arena_count:
            raise ValueError("graph_arena_count cannot override the legacy 27B arena plan")

    @classmethod
    def from_environment(
        cls, *, lingbot2_multiview_spm_z2: bool | None = None,
        qwen35_legacy_27b_sdk_budget: bool = False,
        execution_core_count: int = 8,
        graph_arena_count: int = 0,
    ) -> "GraphRuntimePolicy":
        if type(qwen35_legacy_27b_sdk_budget) is not bool:
            raise TypeError("qwen35_legacy_27b_sdk_budget override must be bool")
        skip_sync, skip_sync_src = _graph_policy_bool(
            "RPU_FASTREPLAY_SKIP_SYNC", default=True
        )
        force_oneshot, force_oneshot_src = _graph_policy_bool(
            "RPU_GRAPH_FORCE_ONESHOT_ON_REPLAY"
        )
        defer_mode, defer_src = _graph_host_op_defer_mode()
        isolate, isolate_src = _graph_policy_bool(
            "RPU_SIGLIP_ISOLATE_PATCH_EMBED"
        )
        if lingbot2_multiview_spm_z2 is None:
            z2, z2_src = _graph_policy_exact_one(
                "RPU_LINGBOT2_MULTIVIEW_SPM_Z2"
            )
        else:
            if type(lingbot2_multiview_spm_z2) is not bool:
                raise TypeError("lingbot2_multiview_spm_z2 override must be bool")
            z2 = lingbot2_multiview_spm_z2
            z2_src = "owner:lingbot2_multiview_spm_z2"
        fast_replay, fast_replay_src = _graph_policy_bool(
            "RPU_WALL_OSS_FAST_REPLAY"
        )
        deep_replay, deep_replay_src = _graph_policy_bool(
            "RPU_DEEP_FAST_REPLAY"
        )
        entries, kd_mb, instr_mb = _lkn_load_snapshot()
        return cls(
            fast_replay_skip_sync=skip_sync,
            force_oneshot_on_replay=force_oneshot,
            host_op_defer_mode=defer_mode,
            siglip_isolate_patch_embed=isolate,
            lingbot2_multiview_spm_z2=z2,
            fmb_fast_replay=fast_replay,
            fmb_deep_fast_replay=deep_replay,
            lkn_max_batch_entries=entries,
            lkn_kd_buf_mb=kd_mb,
            lkn_instr_buf_mb=instr_mb,
            qwen35_legacy_27b_sdk_budget=qwen35_legacy_27b_sdk_budget,
            execution_core_count=execution_core_count,
            graph_arena_count=graph_arena_count,
            provenance=(
                skip_sync_src,
                force_oneshot_src,
                defer_src,
                isolate_src,
                z2_src,
                fast_replay_src,
                deep_replay_src,
                "LKN=extension-load-snapshot",
                f"owner:qwen35_legacy_27b_sdk_budget={int(qwen35_legacy_27b_sdk_budget)}",
            ) + (("owner:execution_core_count",) if execution_core_count != 8 else ())
            + ((f"owner:graph_arena_count={graph_arena_count}",) if graph_arena_count else ()),
        )

    def prepare_arenas(self) -> bool:
        """Reserve the legacy 27B process plan before weights or instructions.

        Three SDK-sized arenas cover one decode entry, bounded raw prefill,
        and a serialized immediate launch. The successful SDK plan is frozen
        for the process; this is not a general multi-model arena planner.
        A host without an accessible device returns False without allocation.
        """
        if not self.qwen35_legacy_27b_sdk_budget and not self.graph_arena_count:
            raise ValueError("arena preparation requires an explicit owner arena plan")
        if not _cpp_loaded:
            return False
        return bool(self._native().prepare_arenas())

    def _native(self):
        modes = _cpp_ext.GraphHostOpDeferMode
        native_mode = (modes.Auto, modes.ForceOff, modes.ForceOn)[
            self.host_op_defer_mode
        ]
        return _cpp_ext.GraphRuntimePolicy(
            self.fast_replay_skip_sync,
            self.force_oneshot_on_replay,
            native_mode,
            self.siglip_isolate_patch_embed,
            self.lingbot2_multiview_spm_z2,
            self.fmb_fast_replay,
            self.fmb_deep_fast_replay,
            self.lkn_max_batch_entries,
            self.lkn_kd_buf_mb,
            self.lkn_instr_buf_mb,
            self.qwen35_legacy_27b_sdk_budget,
            *((self.execution_core_count, self.graph_arena_count) if self.graph_arena_count
              else (self.execution_core_count,) if self.execution_core_count != 8 else ()),
        )


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
        v = os.environ.get("RPU_SKIP_IDLE_RECORD_FUNCTION", "1")
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
        began = False
        try:
            self._graph = self._cache._graph_for_capture(self._sig)
            self._graph.begin(self._sig._impl)
            began = True
            self._enter_state = self._graph.state()
            if _skip_idle_record_function():
                # Skip record_function only when the opt-in helper confirms
                # profiling is inactive. Active profilers retain their ranges;
                # RPU_SKIP_IDLE_RECORD_FUNCTION is disabled by default.
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
        except BaseException as enter_error:
            # Admission can reject an unwarmed READY signature before any
            # graph begins. Existing plans remain valid in that case.
            try:
                if self._graph is not None:
                    self._cache._execution_plans.clear()
                    if began:
                        try:
                            self._graph.abort()
                        except BaseException as abort_error:
                            if hasattr(enter_error, "add_note"):
                                enter_error.add_note(
                                    "Graph abort after enter failure also failed: "
                                    f"{abort_error!r}"
                                )
            finally:
                self._prof_ctx = None
                self._graph = None
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        primary_error = exc_val
        try:
            if self._graph is None:
                return False
            if exc_type is None:
                try:
                    self._graph.end()
                except BaseException as end_error:
                    self._cache._execution_plans.clear()
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
                self._cache._execution_plans.clear()
                try:
                    self._graph.abort()
                except BaseException as abort_error:
                    if hasattr(exc_val, "add_note"):
                        exc_val.add_note(
                            "Graph abort after body failure also failed: "
                            f"{abort_error!r}"
                        )
            return False
        except BaseException as capture_error:
            primary_error = capture_error
            raise
        finally:
            try:
                if self._prof_ctx is not None:
                    try:
                        self._prof_ctx.__exit__(exc_type, exc_val, exc_tb)
                    except BaseException as profiler_error:
                        if primary_error is None:
                            raise
                        if hasattr(primary_error, "add_note"):
                            primary_error.add_note(
                                "Graph profiler exit after capture failure also failed: "
                                f"{profiler_error!r}"
                            )
            finally:
                self._prof_ctx = None
                # A retained exception traceback keeps this scope alive. Drop
                # only its borrowed shared_ptr so caller eviction/close can
                # still enforce the native external-reference guard.
                self._graph = None


class GraphCache:
    """Signature-indexed graph cache with CONFIGURING/WARMING/READY phases."""

    DEFAULT_MAX_ENTRIES = 32
    _CONFIGURING = "CONFIGURING"
    _WARMING = "WARMING"
    _READY = "READY"

    def __init__(self, max_entries=None, *, runtime_policy=None):
        from rpu_backend.runtime.execution_planner import PreparedExecutionPlans

        if max_entries is None:
            max_entries = self.DEFAULT_MAX_ENTRIES
        # A one-slot physical cache can still serve several prepared shapes
        # (for example vision groups captured in sequence).
        self._execution_plans = PreparedExecutionPlans(max(32, int(max_entries)))
        self._phase = self._CONFIGURING
        if runtime_policy is None:
            runtime_policy = GraphRuntimePolicy.from_environment()
        if not isinstance(runtime_policy, GraphRuntimePolicy):
            raise TypeError("runtime_policy must be a GraphRuntimePolicy")
        self._runtime_policy = runtime_policy
        if not _cpp_loaded:
            self._impl = None
            return
        self._impl = _cpp_ext.GraphCache(
            int(max_entries), runtime_policy._native()
        )
        _LIVE_GRAPH_CACHES.add(self)

    @property
    def runtime_policy(self) -> "GraphRuntimePolicy":
        """The same cold policy passed to the native cache at construction."""
        return self._runtime_policy

    def capture(self, sig):
        """Return the main ``with cache.capture(sig)`` scope."""
        return _GraphCaptureScope(self, sig)

    def prepare_plan(self, key, factory):
        """Reuse the physical plan before constructing its Graph signature."""
        return self._execution_plans.prepare(key, factory)

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
        self._execution_plans.begin_warmup()
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
        self._execution_plans.freeze()
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
        self._execution_plans.clear()
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


def install_torch_namespace(rpu_module: Any, graph_type: Any = None) -> None:
    """Install the canonical wrappers on ``torch.rpu``."""
    if not _cpp_loaded:
        return
    rpu_module.Graph = _cpp_ext.Graph if graph_type is None else graph_type
    rpu_module.GraphCache = GraphCache
    rpu_module.GraphRuntimePolicy = GraphRuntimePolicy
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
    "GraphRuntimePolicy",
    "GraphSignature",
    "configure_backend",
    "get_default_graph_cache",
    "install_torch_namespace",
]
