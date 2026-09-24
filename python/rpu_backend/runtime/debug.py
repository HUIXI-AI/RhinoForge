"""Debug helpers — set/get_debug, set/get_profile, SPM debug, profile accumulator reset.

Per ADR §6.4 (v5-04 final form): canonical home for debug control wrappers.
Lifted byte-equal from the v4.1 ``rpu_backend._internal.debug`` module
(deleted in this PR).

Imports C-ext symbols from ``rpu_backend._cpp_ext`` (synthetic module populated
by ``__init__.py`` from the .so loader). Each leaf reads ``_cpp_loaded`` to gate
the call.
"""
from __future__ import annotations

import contextlib

import torch


def _cpp_loaded() -> bool:
    """Return True iff the .so backend was loaded at import time.

    Late import of the synthetic _cpp_ext module avoids a module-load circular
    (rpu_backend.__init__ creates _cpp_ext, then runs Python files that may
    `from rpu_backend._internal import debug as _debug` — at THAT point _cpp_ext
    exists but _internal/debug.py was never re-imported).
    """
    import sys as _sys
    return "rpu_backend._cpp_ext" in _sys.modules and bool(getattr(_sys.modules.get("rpu_backend"), "_cpp_loaded", False))


def _cpp():
    """Return the C-extension symbol bag (or None)."""
    import sys as _sys
    return _sys.modules.get("rpu_backend._cpp_ext")


# -------------------------------
# Debug switch
# -------------------------------
def set_debug(enabled: bool) -> None:
    """Enable/disable debug logging (CPU_FALLBACK, MANUAL_FALLBACK messages)."""
    if _cpp_loaded():
        _cpp().set_debug(enabled)


def get_debug() -> bool:
    """Get debug logging state."""
    if _cpp_loaded() and hasattr(_cpp(), "get_debug"):
        return _cpp().get_debug()
    return False


# -------------------------------
# Profile switch
# -------------------------------
def set_profile(enabled: bool) -> None:
    """Enable/disable profile timing output (kernel timing information)."""
    if _cpp_loaded():
        _cpp().set_profile(enabled)


def get_profile() -> bool:
    """Get profile timing state."""
    if _cpp_loaded() and hasattr(_cpp(), "get_profile"):
        return _cpp().get_profile()
    return False


# -------------------------------
# Profile accumulator reset
# -------------------------------
def reset_profile_accumulators() -> None:
    """Reset all profile accumulators (call before profiling runs to clear warmup data)."""
    if _cpp_loaded() and hasattr(_cpp(), "reset_profile_accumulators"):
        _cpp().reset_profile_accumulators()


# -------------------------------
# HW perf trace (Chrome JSON)
# -------------------------------
def set_hw_perf_trace(
    enabled: bool,
    output_dir: str = ".rpu-hw-perf",
    max_dumps: int = 32,
) -> None:
    """Enable RPU HW cycle-precision kernel/DMA Chrome trace dump.

    Generates one JSON per BUILD and per REPLAY of every cached fused-decoder
    subgraph, up to ``max_dumps`` files per process. File names:
    ``<output_dir>/rpu_hwperf_pid<P>_tid<TID>_seq<S>_<GRAPH>_<build|replay>.json``.
    Drag a JSON into https://ui.perfetto.dev to visualize.

    Toggle semantics: lazy per-slot invalidate — calling this with ``enabled``
    flipped versus the previous setting does NOT eagerly reset GraphCache.
    Each cached graph slot remembers its BUILD-time perf state and is rebuilt
    the next time it's selected if the global flag has flipped. ``output_dir``
    and ``max_dumps`` changes take effect on the next dump (no rebuild needed);
    each call also resets the dump budget.
    """
    if _cpp_loaded() and hasattr(_cpp(), "set_hw_perf_trace"):
        _cpp().set_hw_perf_trace(enabled, output_dir, max_dumps)


def get_hw_perf_trace() -> bool:
    """Get HW perf trace enabled state."""
    if _cpp_loaded() and hasattr(_cpp(), "get_hw_perf_trace"):
        return _cpp().get_hw_perf_trace()
    return False


@contextlib.contextmanager
def hw_perf_trace(output_dir: str = ".rpu-hw-perf", max_dumps: int = 32):
    """Enable HW perf trace for the body of a ``with`` block.

    Always disables on exit, even on exception. Does NOT snapshot/restore
    any prior ``set_hw_perf_trace`` config — after the with-block the trace
    is OFF, with ``output_dir`` and ``max_dumps`` preserved on the C++ side
    (so a later ``set_hw_perf_trace(True)`` without args is harmless).

    Usage::

        with torch.rpu.hw_perf_trace("/tmp/run42", max_dumps=16):
            outputs = model.generate(...)
        # perf is OFF here; JSON files in /tmp/run42/

    See :func:`set_hw_perf_trace` for arg semantics.
    """
    set_hw_perf_trace(True, output_dir, max_dumps)
    try:
        yield
    finally:
        # Disable but keep output_dir + max_dumps so any post-block call to
        # set_hw_perf_trace(True) without explicit args picks up sensible
        # defaults. Side-effect: max_dumps gets re-stamped, which also
        # refreshes the budget counter — harmless when enabled is False.
        set_hw_perf_trace(False, output_dir, max_dumps)


# -------------------------------
# Debug tensor export
# -------------------------------










# -------------------------------
# SPM debug
# -------------------------------
def set_spm_debug(enabled: bool) -> None:
    """Enable/disable SPM/chunk size debug output."""
    if _cpp_loaded():
        _cpp().set_spm_debug(enabled)


def get_spm_debug() -> bool:
    """Get SPM debug state."""
    if _cpp_loaded() and hasattr(_cpp(), "get_spm_debug"):
        return _cpp().get_spm_debug()
    return False


def spm_alloc_dump(label: str = "") -> None:
    """Print SPM allocator usage (temporary/persistent/free)."""
    torch.ops.rpu.spm_alloc_dump(label)
