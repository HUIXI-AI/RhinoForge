"""Debug helpers — logging, timing counters, and SPM diagnostics.

This is the canonical home for debug control wrappers.

Imports C-ext symbols from ``rpu_backend._cpp_ext`` (synthetic module populated
by ``__init__.py`` from the .so loader). Each leaf reads ``_cpp_loaded`` to gate
the call.
"""
from __future__ import annotations

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


def get_debug_export() -> bool:
    """Compatibility query for model code; activation export is hard-disabled."""
    return False


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
