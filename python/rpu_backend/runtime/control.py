"""Runtime controls for allocation, synchronization, batching, and shutdown.

Chunk size affects SPM layout and graph signatures, so public entry points bind
cold ``rpu_execution`` settings to each handle. Low-level chunk setters remain
per-handle runtime plumbing.

Imports C-ext symbols from ``rpu_backend._cpp_ext`` (synthetic module populated
by ``__init__.py`` from the .so loader). Each leaf reads ``_cpp_loaded`` to gate
the call.
"""
from __future__ import annotations

import os
import torch

from rpu_backend._native_loader import device_nodes_accessible as _device_nodes_accessible


def _cpp_loaded() -> bool:
    """Return True iff the .so backend was loaded at import time."""
    import sys as _sys
    return "rpu_backend._cpp_ext" in _sys.modules and bool(getattr(_sys.modules.get("rpu_backend"), "_cpp_loaded", False))


def _cpp():
    """Return the C-extension symbol bag (or None)."""
    import sys as _sys
    return _sys.modules.get("rpu_backend._cpp_ext")


# -------------------------------
# AMP support
# -------------------------------
def get_amp_supported_dtype() -> list:
    """Return list of dtypes supported by AMP on RPU."""
    return [torch.float16]


def is_autocast_available() -> bool:
    """Check if autocast is available for RPU."""
    return True


# -------------------------------
# Availability check
# -------------------------------
def is_available() -> bool:
    """Return whether the native backend and RPU runtime are usable.

    ``torch.rpu`` is always scaffolded by the Python package.  Loading the
    extension alone is insufficient: the current process must retain access to
    the required device nodes and the native lifecycle must still be READY.
    """
    if not _cpp_loaded() or not _device_nodes_accessible():
        return False
    cpp = _cpp()
    return bool(
        cpp is not None
        and hasattr(cpp, "rpu_runtime_is_available")
        and cpp.rpu_runtime_is_available()
    )


# -------------------------------
# Caching allocator
# -------------------------------
def set_caching_allocator(enabled: bool) -> None:
    """Enable/disable caching allocator (default: disabled)."""
    if _cpp_loaded():
        _cpp().set_caching_allocator(enabled)


def get_caching_allocator() -> bool:
    """Get caching allocator state."""
    if _cpp_loaded() and hasattr(_cpp(), "get_caching_allocator"):
        return _cpp().get_caching_allocator()
    return False


def empty_cache() -> None:
    """Release cached RPU allocator blocks back to the runtime."""
    if _cpp_loaded() and hasattr(_cpp(), "empty_cache"):
        _cpp().empty_cache()


def get_memory_stats() -> dict:
    """Return RPU caching allocator stats when available."""
    if _cpp_loaded() and hasattr(_cpp(), "get_memory_stats"):
        return dict(_cpp().get_memory_stats())
    return {}


def memory_stats() -> dict:
    """Alias matching torch.cuda.memory_stats style."""
    return get_memory_stats()


def reset_peak_memory_stats() -> None:
    """Reset current peak counters in the RPU caching allocator."""
    if _cpp_loaded() and hasattr(_cpp(), "reset_peak_memory_stats"):
        _cpp().reset_peak_memory_stats()


def reset_accumulated_memory_stats() -> None:
    """Reset accumulated allocation/free counters in the RPU caching allocator."""
    if _cpp_loaded() and hasattr(_cpp(), "reset_accumulated_memory_stats"):
        _cpp().reset_accumulated_memory_stats()


# -------------------------------
# DDR flush
# -------------------------------
def set_ddr_flush(enabled: bool) -> None:
    """Enable/disable internal RPU kernel-to-kernel DDR flush (default: disabled).

    Controls intra-RPU sync points that are redundant when data never leaves
    DDR/SPM. CPU<->RPU boundary coherency is controlled separately by
    set_ddr_flush_force.
    """
    if _cpp_loaded():
        _cpp().set_ddr_flush(enabled)


def get_ddr_flush() -> bool:
    """Get DDR flush state."""
    if _cpp_loaded() and hasattr(_cpp(), "get_ddr_flush"):
        return _cpp().get_ddr_flush()
    return False


def set_ddr_flush_force(enabled: bool) -> None:
    """Enable/disable CPU<->RPU boundary force DDR flush (default: enabled).

    Required for cache coherency at zero-copy views, cross-device copies, and
    CPU fallback boundaries. Disabling is unsafe in production but useful for
    quantifying flush overhead in micro-benchmarks.
    """
    if _cpp_loaded() and hasattr(_cpp(), "set_ddr_flush_force"):
        _cpp().set_ddr_flush_force(enabled)


def get_ddr_flush_force() -> bool:
    """Get boundary force DDR flush state."""
    if _cpp_loaded() and hasattr(_cpp(), "get_ddr_flush_force"):
        return _cpp().get_ddr_flush_force()
    return True


# -------------------------------
# SPM mode
# -------------------------------
def get_spm_mode() -> bool:
    """Return False because eager operators have fixed SPM/fallback behavior.

    The function remains bound through ``torch.rpu`` for compatibility.
    """
    return False


# -------------------------------
# Chunk size is per handle rather than process-wide.
#
# `set_chunk_size(N)` decided a PER-HANDLE quantity: it keyed every live handle's
# SPM allocation and, because the graph signature did not encode it, a change
# after capture replayed a recorded graph under a layout it was not recorded for.
# Name a handle instead:
#     torch.ops.rpu.causal_decoder_set_chunk_size_override(handle, cs)
#     torch.ops.rpu.halo_image_flow_set_chunk_size_override(handle, cs)
# and for a bound on the AUTO search rather than a pin,
#     torch.ops.rpu.causal_decoder_set_chunk_size_cap(handle, cs)   # cold only
# -------------------------------


def get_allreduce_twostage() -> bool:
    """Return the current process two-stage all-reduce selector."""
    if not _cpp_loaded():
        return False
    cpp = _cpp()
    if cpp is None or not hasattr(cpp, "get_allreduce_twostage"):
        raise RuntimeError(
            "native backend lacks get_allreduce_twostage; rebuild rpu_backend"
        )
    return bool(cpp.get_allreduce_twostage())


def get_linear_acc32() -> bool:
    """Return the process-cached fused FP16 Linear ACC32 selector."""
    if not _cpp_loaded():
        return False
    cpp = _cpp()
    if cpp is None or not hasattr(cpp, "get_linear_acc32"):
        raise RuntimeError(
            "native backend lacks get_linear_acc32; rebuild rpu_backend"
        )
    return bool(cpp.get_linear_acc32())


# -------------------------------
# Cross-layer batch
# -------------------------------
def set_cross_layer_batch_prefill(enabled: bool) -> None:
    """Enable cross-layer batch for multi-chunk prefill."""
    if _cpp_loaded():
        _cpp().set_cross_layer_batch_prefill(enabled)


def get_cross_layer_batch_prefill() -> bool:
    """Check if cross-layer batch prefill is enabled."""
    if _cpp_loaded() and hasattr(_cpp(), "get_cross_layer_batch_prefill"):
        return _cpp().get_cross_layer_batch_prefill()
    return False


def set_cross_layer_batch_size(size: int) -> None:
    """Set number of layers per batch group (default 12)."""
    if _cpp_loaded():
        _cpp().set_cross_layer_batch_size(size)


def get_cross_layer_batch_size() -> int:
    """Get number of layers per batch group."""
    if _cpp_loaded() and hasattr(_cpp(), "get_cross_layer_batch_size"):
        return _cpp().get_cross_layer_batch_size()
    return 12


# -------------------------------
# Fused lm_head configuration is owned by the per-handle
# ``causal_decoder_set_lm_head(handle, lm_head_w)`` interface.
# -------------------------------


# -------------------------------
# SPM resets
# -------------------------------
def spm_alloc_reset_temporary() -> None:
    """Reset SPM temporary allocations (frees all temporary buffers)."""
    torch.ops.rpu.spm_alloc_reset_temporary()


def reset_temporary_spm() -> None:
    """Reset SPM temporary allocations (frees all temporary buffers)."""
    torch.ops.rpu.spm_alloc_reset_temporary()


def reset_all_spm() -> None:
    """Reset SPM temporary + persistent allocations (super-persistent preserved)."""
    torch.ops.rpu.spm_alloc_reset_all()


# -------------------------------
# Shutdown
# -------------------------------
def shutdown() -> None:
    """Release all cached resources (SPM pool, caching allocator).

    This function is automatically registered with atexit by __init__.py.
    """
    if _cpp_loaded():
        _cpp().shutdown()


# -------------------------------
# Warmup helper
# -------------------------------
def _get_rpu_warmup() -> int:
    """RPU_WARMUP=N env var → N warmup forwards before the measured forward."""
    try:
        return max(0, int(os.environ.get("RPU_WARMUP", "0")))
    except ValueError:
        return 0


# -------------------------------
# The one boolean env parser
# -------------------------------
_ENV_BOOL_TRUE = frozenset({"1", "true", "on"})
_ENV_BOOL_FALSE = frozenset({"0", "false", "off", ""})


def rpu_env_bool(
    name: str, default: bool = False, *, cpp_mirror: str | None = None
) -> bool:
    """Parse one boolean `RPU_*` switch. The ONLY boolean env reader in Python.

    The opening rules in ``docs/runtime_config.md`` document these semantics:

      unset            -> ``default``
      1 / true / on    -> True    (case-insensitive, surrounding space stripped)
      0 / false / off  -> False   (case-insensitive; "" counts as False)
      anything else    -> ``ValueError``

    One strict parser keeps Python-side environment semantics consistent and
    rejects typos instead of silently selecting the wrong execution path.

    ``cpp_mirror`` — pass ``"src/<file>.cpp"`` when the same variable is
    also read by ``std::getenv`` in the native half. There is no single rule
    over there (one site is ``e && s != "0" && s != "false"``, another is
    ``e[0] in {1,t,T}``), and where the two halves disagree the feature runs on
    one side only — e.g. a merger folded into the C++ graph AND re-applied in
    Python, which is a silently wrong result, not an error. ``0`` and ``1`` are
    the only spellings every parser in this repo reads alike, so with
    ``cpp_mirror`` set, anything else is refused with a pointer to the C++ site.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if cpp_mirror is not None and raw.strip() not in ("0", "1"):
        raise ValueError(
            f"{name} is read by BOTH Python and C++ ({cpp_mirror}), and the two "
            f"parsers do not agree outside '0'/'1' (C++ reads '', 'False' and "
            f"'off' as ON). Set it to 1 or 0; got {raw!r}."
        )
    if value in _ENV_BOOL_TRUE:
        return True
    if value in _ENV_BOOL_FALSE:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, or on/off (case-insensitive); "
        f"got {raw!r}. Use 1 to enable and 0 to disable — those are the only "
        f"two spellings the Python and C++ parsers agree on."
    )


# -------------------------------
# Fused lm_head configuration is per handle; no process-global setter is
# exposed here.
# -------------------------------
