"""Runtime control helpers — caching allocator, DDR flush, SPM mode, chunk size, cross-layer batch, shutdown, SPM resets.

Per ADR §6.4 (v5-04 final form): canonical home for SPM / system / cross-layer
batch / caching-allocator wrappers. Lifted byte-equal from the v4.1
``rpu_backend._internal.runtime`` module (deleted in this PR).

Per ADR §11 / §10 #15: deprecated ``set_chunk_size`` / ``set_spm_mode`` global
setters were tombstoned in v4.1 (use the model-specific public replacement);
the global ``set_fuse_lm_head*`` family was removed in v5-01b. MR-D finished the
chunk half: ``set_chunk_size`` / ``get_chunk_size`` and the ``_rpu_chunk_size``
module global are GONE, along with the C++ ``g_chunk_size_override`` they wrote.
Chunk size keys the SPM layout and graph signature, so public entry points bind
cold ``rpu_execution`` to each handle; the low-level per-handle setter is
internal runtime plumbing.

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
    """Select the process allocator before the first non-empty RPU tensor.

    The first call or allocation freezes the choice. Repeating the same value
    is safe; selecting the other mode requires a fresh Python process.
    """
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
    """Always False: the legacy per-op SPM selector (``g_rpu_use_spm_kernel``)
    was deleted. Kept because ``torch.rpu.get_spm_mode`` is still bound — the
    eager Linear now always stages through SPM and the eager RMSNorm always
    falls back to CPU, so there is nothing left to select.
    """
    return False


# -------------------------------
# Chunk size — MR-D: there is no process-wide chunk size any more.
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
# Fuse lm_head (qwen3-specific runtime) — v5-01b D-15/D-16/D-17 DELETED.
# Per ADR §6.4 + §10 #15: the global `set_fuse_lm_head` family is removed in
# v5-01b; the per-instance `causal_decoder_set_lm_head(handle, lm_head_w)` API
# lands in v5-06. The C-ext-side `g_fuse_lm_head*` cluster is retained until
# the v5-06 per-instance API replaces it.
# -------------------------------


# -------------------------------
# SPM resets
# -------------------------------
def spm_alloc_reset_temporary() -> None:
    """Reset SPM temporary allocations (frees all temporary buffers)."""
    torch.ops.rpu.spm_alloc_reset_temporary()


def reset_temporary_spm() -> None:
    """Reset SPM temporary allocations (frees all temporary buffers).
    Plan 01-01 alias for the existing torch.ops.rpu.spm_alloc_reset_temporary op."""
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

    Semantics = convention **D** of docs/runtime_config.md §"How values are
    parsed", promoted from a single adapter to the whole package:

      unset            -> ``default``
      1 / true / on    -> True    (case-insensitive, surrounding space stripped)
      0 / false / off  -> False   (case-insensitive; "" counts as False)
      anything else    -> ``ValueError``

    Why one parser and why it raises: the same variable used to be parsed a
    different way in every file (``== "1"`` here, ``not in ("0","false","")``
    there), so ``VAR=true`` meant ON in one module and OFF in the next, and a
    typo meant ON everywhere with no diagnostic. A boolean switch that silently
    resolves to the opposite of what the operator wrote is the worst failure
    mode this package has; refusing to guess is cheaper than any of them.

    ``cpp_mirror`` — pass ``"src/<file>.cpp:<line>"`` when the SAME variable is
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
# Qwen3-specific fused lm_head — v5-01b D-18 + setup_fuse_lm_head DELETED.
# Per ADR §6.4: `set_fused_lm_head_enabled` was a thin dispatcher; deleted in
# v5-01b alongside the wrappers above. `setup_fuse_lm_head` was its sole
# in-tree caller (set_fuse_lm_head_weights → C-ext); deleted in the same
# atomic plan per Karpathy meta-rule 4 (the family is dead; the helper is dead).
# Per-instance API (`causal_decoder_set_lm_head(handle, lm_head_w)`) lands
# in v5-06.
# -------------------------------
