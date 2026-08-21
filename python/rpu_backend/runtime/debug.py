"""Debug helpers — set/get_debug, set/get_profile, debug tensor export, SPM debug, profile accumulator reset.

This is the canonical home for debug control wrappers.

Imports C-ext symbols from ``rpu_backend._cpp_ext`` (synthetic module populated
by ``__init__.py`` from the .so loader). Each leaf reads ``_cpp_loaded`` to gate
the call.
"""
from __future__ import annotations

import contextlib

import torch


def _cpp_loaded() -> bool:
    """Return True iff the .so backend was loaded at import time.

    Late import of the synthetic ``_cpp_ext`` module avoids a package-import
    cycle while ``rpu_backend.__init__`` is still populating native symbols.
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
    output_dir: str = ".",
    max_dumps: int = 32,
) -> None:
    """Enable RPU HW cycle-precision kernel/DMA Chrome trace dump.

    Generates one JSON per BUILD and per REPLAY segment of every cached
    subgraph, up to ``max_dumps`` files per process. File names:
    ``<output_dir>/rpu_hwperf_pid<P>_tid<TID>_seq<S>_<GRAPH>_<build|replay>_seg<N>.json``.
    Drag a JSON into https://ui.perfetto.dev to visualize. Raw files include
    DMA source/destination device addresses and execution structure; treat
    them as sensitive application artifacts.

    Enabling is prospective: it does not rebuild existing adapter-owned
    graphs. A graph's performance buffers are fixed at its first BUILD, so use
    a fresh process and enable tracing before model construction, explicit
    graph preparation, or the first forward. Disabling stops future files;
    ``output_dir`` and ``max_dumps`` changes apply to the next eligible dump.
    Each call resets the dump budget.
    """
    if _cpp_loaded() and hasattr(_cpp(), "set_hw_perf_trace"):
        _cpp().set_hw_perf_trace(enabled, output_dir, max_dumps)


def get_hw_perf_trace() -> bool:
    """Get HW perf trace enabled state."""
    if _cpp_loaded() and hasattr(_cpp(), "get_hw_perf_trace"):
        return _cpp().get_hw_perf_trace()
    return False


@contextlib.contextmanager
def hw_perf_trace(output_dir: str = ".", max_dumps: int = 32):
    """Enable HW perf trace for the body of a ``with`` block.

    Enter before model construction or any explicit graph preparation so the
    first Graph BUILD records performance data. Always disables on exit, even
    on exception. Does NOT snapshot/restore
    any prior ``set_hw_perf_trace`` config — after the with-block the trace
    is OFF; the next explicit call supplies its own output settings.

    Usage::

        with torch.rpu.hw_perf_trace("/tmp/run42", max_dumps=16):
            model = load_model()
            outputs = model.generate(...)
        # perf is OFF here; JSON files in /tmp/run42/

    See :func:`set_hw_perf_trace` for arg semantics.
    """
    set_hw_perf_trace(True, output_dir, max_dumps)
    try:
        yield
    finally:
        # Re-stamping the same values also resets the dump counter, harmless
        # while tracing is disabled.
        set_hw_perf_trace(False, output_dir, max_dumps)


# -------------------------------
# Debug tensor export
# -------------------------------
def set_debug_export(enabled: bool) -> None:
    """Enable/disable debug tensor export for fused decoder layer verification."""
    if _cpp_loaded():
        _cpp().set_debug_export(enabled)


def get_debug_export() -> bool:
    """Get debug tensor export state."""
    if _cpp_loaded() and hasattr(_cpp(), "get_debug_export"):
        return _cpp().get_debug_export()
    return False


def get_debug_tensor(name: str) -> torch.Tensor:
    """Get a debug tensor by name."""
    if _cpp_loaded() and hasattr(_cpp(), "get_debug_tensor"):
        return _cpp().get_debug_tensor(name)
    return torch.Tensor()


def clear_debug_tensors() -> None:
    """Clear all stored debug tensors."""
    if _cpp_loaded() and hasattr(_cpp(), "clear_debug_tensors"):
        _cpp().clear_debug_tensors()


def list_debug_tensors() -> list:
    """List all stored debug tensor names."""
    if _cpp_loaded() and hasattr(_cpp(), "list_debug_tensors"):
        return _cpp().list_debug_tensors()
    return []


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
