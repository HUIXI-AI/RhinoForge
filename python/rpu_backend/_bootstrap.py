"""Initialize the RPU PyTorch backend and its four-name public surface.

Removed package attributes raise ``AttributeError`` with a current replacement
when one exists. See ``docs/api_reference.md#top-level-package``.
"""
from __future__ import annotations

# Keep module-scope imports private so they do not expand the package namespace.
import sys as _sys
import types as _types
import atexit as _atexit
import warnings as _warnings

# The Rhino Launch DMA allocator opens one mem-object fd per RPU
# tensor allocation; Pi0.5 (~3.3 GB across hundreds of weight tensors) drives
# the count past the default soft limit (1024), surfacing as `std::bad_alloc`
# when the underlying open() returns EMFILE. Lift soft to min(target, hard)
# BEFORE `import torch` (which also consumes fds during init).
from ._bootstrap_env import ensure_nofile_limit as _ensure_nofile_limit
_ensure_nofile_limit()
del _ensure_nofile_limit

import torch as _torch

_torch._C._rename_privateuse1_backend("rpu")

# Apply an explicit RPU_MODEL_CACHE override without changing the standard
# Hugging Face cache when no override was requested.
from ._bootstrap_env import configure_hf_cache as _configure_hf_cache  # noqa: E402
_configure_hf_cache()
del _configure_hf_cache

# torch.rpu module scaffold.
if "torch.rpu" not in _sys.modules:
    _mod = _types.ModuleType("torch.rpu")
    _mod.__dict__["__file__"] = "<rpu_backend_dynamic>"
    _sys.modules["torch.rpu"] = _mod
    _torch.rpu = _mod
else:
    _mod = _sys.modules["torch.rpu"]


# Isolate native symbols in ``rpu_backend._cpp_ext`` so they do not leak into
# the four-name package namespace.
from ._native_loader import load_cpp_extension as _load_cpp_extension
_cpp_ext, _cpp_loaded = _load_cpp_extension()
del _load_cpp_extension

# Eager-bind the four documented top-level names.
from rpu_backend.api.cache import RPUCache
from rpu_backend.api.causal_lm import RPUModelForCausalLM
from rpu_backend._version import DISTRIBUTION_VERSION as __version__

if _cpp_loaded and hasattr(_cpp_ext, "reset_graph_cache"):
    reset_graph_cache = _cpp_ext.reset_graph_cache
else:
    def reset_graph_cache():
        """No-op when .so is not loaded."""
        return None

# Keep the documented four-name list alphabetical and stable.
__all__ = [
    '__version__', 'RPUCache', 'RPUModelForCausalLM', 'reset_graph_cache',
]

# Preserve the raw Graph compatibility binding. Structured cache wrappers live
# under rpu_backend.graph; raw native classes remain in rpu_backend._cpp_ext.
if _cpp_loaded and hasattr(_cpp_ext, "Graph"):
    Graph = _cpp_ext.Graph

# The formal graph namespace owns the one production wrapper implementation.
# Configure it before installing torch.rpu aliases or importing any adapter.
from rpu_backend import graph as _graph  # noqa: E402
_graph.configure_backend(_cpp_ext if _cpp_loaded else None,
                         "<live>" if _cpp_loaded else None)
from rpu_backend.graph._runtime import (  # noqa: E402
    install_torch_namespace as _install_graph_torch_namespace,
)
_install_graph_torch_namespace(_mod)
del _install_graph_torch_namespace
_GraphCache = _graph.GraphCache

# Bind the torch.cuda-shaped device protocol plus runtime/debug/native helpers.
# The returned control module is retained only for ordered atexit shutdown.
from ._torch_rpu import configure_torch_rpu as _configure_torch_rpu
_control = _configure_torch_rpu(_torch, _mod, _cpp_ext, _cpp_loaded)
del _configure_torch_rpu

# Wire optional FakeTensor integration after the canonical graph runtime.
if _cpp_loaded:
    from ._graph_wiring import configure_graph_runtime as _configure_graph_runtime
    _configure_graph_runtime(_cpp_ext)
    del _configure_graph_runtime

# Register the fixed built-in adapter manifest. Built-in model modules stay
# unloaded until get_adapter() selects their architecture; external plugins
# are likewise deferred until an unknown architecture lookup or an explicit
# runtime.registry.discover_plugins() call.
# Delete bootstrap aliases after use so they do not leak into the package
# namespace.
from rpu_backend.adapters._manifest import BUILTIN_ADAPTERS as _BUILTIN_ADAPTERS
from rpu_backend.runtime.registry import discover_all as _discover_all
_discover_all(_BUILTIN_ADAPTERS)
del _BUILTIN_ADAPTERS, _discover_all


def __dir__():
    return list(__all__)


# Replacement hints for removed package attributes, alphabetical by key.
_V5_MIGRATION = {
    "build_cache": "use rpu_backend.RPUCache.from_model(model, ...)",
    "build_rpu_cache": "use rpu_backend.RPUCache.from_model(model, ...)",
    "convert_linear_weights_inplace": "use rpu_backend.runtime.weights.swizzle_model_inplace",
    "model_converter": "use rpu_backend.runtime.weights.swizzle_model_inplace",
    "patch_apply_rotary_pos_emb": "use the adapter flow in docs/model_porting.md#3-reuse-the-causal-decoder-when-it-matches",
    "patch_attention_for_rpu": "use the adapter flow in docs/model_porting.md#3-reuse-the-causal-decoder-when-it-matches",
    "patch_decoder_layer_for_rpu_fused": "use the adapter flow in docs/model_porting.md#3-reuse-the-causal-decoder-when-it-matches",
    "patch_qwen3_model_for_rpu_all_layers_once": "use the adapter flow in docs/model_porting.md#3-reuse-the-causal-decoder-when-it-matches",
    "patch_rmsnorm_class": "use the adapter flow in docs/model_porting.md#3-reuse-the-causal-decoder-when-it-matches",
    "Pi05Policy": "use rpu_backend.api.Pi05Policy (lazy import; not at top level)",
    "qwenpi05_fused_converter": "no importable replacement in this release",
    "RPUKVCache": "use rpu_backend.RPUCache",
    "set_chunk_size": "chunk size is per-handle: torch.ops.rpu.causal_decoder_set_chunk_size_override(handle, cs)",
    "set_fuse_lm_head": "use causal_decoder_set_lm_head(handle, lm_head_w) per-instance API",
    "set_fuse_lm_head_weights": "use causal_decoder_set_lm_head(handle, lm_head_w) per-instance API",
    "set_fused_lm_head_enabled": "use causal_decoder_set_lm_head(handle, lm_head_w) per-instance API",
    "set_spm_mode": "set per-instance attribute model._rpu_spm_mode",
    "setup_fuse_lm_head": "use causal_decoder_set_lm_head(handle, lm_head_w) per-instance API",
}


# Expose debug names lazily through PEP 562 without expanding ``__all__`` or
# ``vars(rpu_backend)``.
_LAZY_DEBUG_API = frozenset({"set_debug_level", "get_debug_level", "TRACE"})
# Preserve Dynamo convenience attributes without importing the guard module
# during a cold package import.
_LAZY_DYNAMO_API = frozenset({
    "verify_lazy_init",
    "UninitializedLazyStateError",
    # Misnomers kept for the packaged runtime API; see graph/lazy_init_guard.py.
    "freeze_for_dynamo",
    "DynamoUnsafeLazyStateError",
})
_DEPRECATED_GRAPH_API = {
    "GraphCache": "rpu_backend.graph.GraphCache",
    "GraphSignature": "rpu_backend.graph.GraphSignature",
    "get_default_graph_cache": "rpu_backend.graph.get_default_graph_cache",
}


def __getattr__(name: str):
    """Resolve narrow debug, Dynamo, graph, and removed-API attributes.

    Lazy branches return symbols without polluting ``vars(rpu_backend)`` or
    expanding ``__all__``.
    """
    if name in _LAZY_DEBUG_API:
        from rpu_backend.runtime import log as _log
        return getattr(_log, name)
    if name in _LAZY_DYNAMO_API:
        from rpu_backend.graph import lazy_init_guard as _lazy_init_guard
        return getattr(_lazy_init_guard, name)
    replacement = _DEPRECATED_GRAPH_API.get(name)
    if replacement is not None:
        _warnings.warn(
            f"rpu_backend.{name} is deprecated since API 5.0 and will be "
            f"removed in API 6.0; use {replacement}",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_graph, name)
    hint = _V5_MIGRATION.get(name)
    if hint:
        raise AttributeError(f"module 'rpu_backend' has no attribute {name!r}; Did you mean: {hint}?")
    raise AttributeError(f"module 'rpu_backend' has no attribute {name!r}")


# Removed process-global setters are intentionally absent and raise
# ``AttributeError`` naturally.

if _cpp_loaded:
    # 单个 atexit hook 串行:先 GraphCache.clear_all (释放所有 entry 的
    # private Queue_t),再 _control.shutdown (拆解 kernel/queue/buffer cache)。
    # Queue_t 必须在它使用的 buffer pool 之前释放。
    def _clear_all_at_exit() -> None:
        _GraphCache.clear_all()
        _control.shutdown()
    _atexit.register(_clear_all_at_exit)


_EXPORT_NAMES = [
    "__version__",
    "__all__",
    "RPUCache",
    "RPUModelForCausalLM",
    "reset_graph_cache",
    "__getattr__",
    "__dir__",
]

_PRIVATE_RUNTIME_EXPORT_NAMES = [
    "_cpp_ext",
    "_cpp_loaded",
]


def export_to_package(pkg_globals):
    """Populate the readable package shim with the runtime symbols users import."""
    for name in _EXPORT_NAMES:
        if name in globals():
            pkg_globals[name] = globals()[name]
    # The Dynamo backend is not exported because adapters already own graph
    # capture and nested capture is unsupported.
    for optional_name in ("Graph",):
        if optional_name in globals():
            pkg_globals[optional_name] = globals()[optional_name]
    for private_name in _PRIVATE_RUNTIME_EXPORT_NAMES:
        if private_name in globals():
            pkg_globals[private_name] = globals()[private_name]
