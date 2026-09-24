"""rpu_backend — v5.0 HF-Parity Library. Four public names; v5 is a pure
cutover: every removed v4 attribute raises AttributeError with a v5 migration
hint per ADR §2.4 + EXT-02.

NH1 iter-3 (D-53): module-scope imports use `_`-prefixed aliases.
NH2 iter-3: C-extension symbols isolated in synthetic `rpu_backend._cpp_ext`.
Phase 01.3 (v5-01b cutover): `_DEPRECATED_ROWS / _SILENT_MAP / _legacy_tombstones`
deleted; `set_chunk_size / set_spm_mode / set_fuse_lm_head*` global setters deleted;
`build_rpu_cache / build_cache` removed from public surface — use
`RPUCache.from_model(model, ...)` (ADR §2.5).

See docs/api_reference.md (after v5-13).
"""
from __future__ import annotations

# NH1 iter-3: all module-scope imports use `_`-prefixed aliases.
import sys as _sys
import types as _types
import atexit as _atexit
import warnings as _warnings

# The rhino-launch-kernel DMA allocator opens one mem-object fd per RPU
# tensor allocation; Pi0.5 (~3.3 GB across hundreds of weight tensors) drives
# the count past the default soft limit (1024), surfacing as `std::bad_alloc`
# when the underlying open() returns EMFILE. Lift soft to min(target, hard)
# BEFORE `import torch` (which also consumes fds during init).
from ._bootstrap_env import ensure_nofile_limit as _ensure_nofile_limit
_ensure_nofile_limit()
del _ensure_nofile_limit

import torch as _torch

_torch._C._rename_privateuse1_backend("rpu")

# Apply the model registry's cache defaults while preserving explicitly set
# Hugging Face cache paths. RPU_MODEL_CACHE relocates the model-alias root.
# See docs/api_reference.md#model-registry.
from ._bootstrap_env import configure_hf_cache as _configure_hf_cache  # noqa: E402
_configure_hf_cache()
del _configure_hf_cache

# torch.rpu module scaffold (preserved from old file).
if "torch.rpu" not in _sys.modules:
    _mod = _types.ModuleType("torch.rpu")
    _mod.__dict__["__file__"] = "<rpu_backend_dynamic>"
    _sys.modules["torch.rpu"] = _mod
    _torch.rpu = _mod
else:
    _mod = _sys.modules["torch.rpu"]


# NH2 iter-3: isolate C-ext symbols in a synthetic ``rpu_backend._cpp_ext``
# module so they do not leak into the four-name package namespace.
from ._native_loader import load_cpp_extension as _load_cpp_extension
_cpp_ext, _cpp_loaded = _load_cpp_extension()
del _load_cpp_extension

# Eager-bind the 4 public names per §1e whitelist (v5: __all__ = 4-name alphabetical list).
from rpu_backend.api.cache import RPUCache  # SKEL-05a (v5-01a-1 / D-02c): owning module is api.cache; D-36 path retained as thin re-export per NS-09a (d).
from rpu_backend.api.causal_lm import RPUModelForCausalLM  # v5-02 D-04: relocated from transformers/_causal_lm.py
from rpu_backend._version import DISTRIBUTION_VERSION as __version__

if _cpp_loaded and hasattr(_cpp_ext, "reset_graph_cache"):
    reset_graph_cache = _cpp_ext.reset_graph_cache
else:
    def reset_graph_cache():
        """No-op when .so is not loaded."""
        return None

# CUT-05 (v5-01b) lock per ADR §2.1: byte-identical alphabetical 4-name list.
__all__ = [
    '__version__', 'RPUCache', 'RPUModelForCausalLM', 'reset_graph_cache',
]

# Resolve the native symbol before configuring the formal graph namespace.
# The public compatibility alias below is replaced by the policy-owning Python
# wrapper; the raw C++ class remains available in rpu_backend._cpp_ext.
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
_install_graph_torch_namespace(_mod, _graph.Graph)
del _install_graph_torch_namespace
Graph = _graph.Graph
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
# NH1 iter-3: `_`-prefixed alias; `del` after invocation so _discover_all does
# NOT leak into dir(rpu_backend) and break the 4-name whitelist (v5: __all__ is
# the locked alphabetical 4-name list).
from rpu_backend.adapters._manifest import BUILTIN_ADAPTERS as _BUILTIN_ADAPTERS
from rpu_backend.runtime.registry import discover_all as _discover_all
_discover_all(_BUILTIN_ADAPTERS)
del _BUILTIN_ADAPTERS, _discover_all


def __dir__():
    return list(__all__)


# D-06 (v5-01b) EXT-02 migration table — see ADR §2.4 + REQUIREMENTS line 138.
# Alphabetical by key (case-insensitive) for grep-friendliness; 18 entries
# (R1: codex Stage-2 R1 MEDIUM-1 added qwenpi05_fused_converter per ADR §10 #18);
# ≤25 entries per CONTEXT D-23 budget.
_V5_MIGRATION = {
    "build_cache": "use rpu_backend.RPUCache.from_model(model, ...)",
    "build_rpu_cache": "use rpu_backend.RPUCache.from_model(model, ...)",
    "convert_linear_weights_inplace": "use rpu_backend.runtime.weights.swizzle_model_inplace (after v5-04)",
    "model_converter": "use rpu_backend.runtime.weights.swizzle_model_inplace (after v5-04)",
    "PaliGemmaPolicy": "use rpu_backend.api.PaliGemmaPolicy (lazy import; not at top level)",
    "patch_apply_rotary_pos_emb": "removed in v5; use FusedModelBase v3 framework",
    "patch_attention_for_rpu": "removed in v5; use FusedModelBase v3 framework",
    "patch_decoder_layer_for_rpu_fused": "removed in v5; use FusedModelBase v3 framework",
    "patch_qwen3_model_for_rpu_all_layers_once": "removed in v5; use FusedModelBase v3 framework",
    "patch_rmsnorm_class": "removed in v5; use FusedModelBase v3 framework",
    "Pi05Policy": "use rpu_backend.api.Pi05Policy (lazy import; not at top level)",
    "qwenpi05_fused_converter": "no importable replacement; use the supported policy conversion API",
    "RPUKVCache": "use rpu_backend.RPUCache (post-v5)",
    "set_chunk_size": "chunk size is per-handle: torch.ops.rpu.causal_decoder_set_chunk_size_override(handle, cs) (MR-D)",
    "set_fuse_lm_head": "use causal_decoder_set_lm_head(handle, lm_head_w) per-instance API (v5-06+)",
    "set_fuse_lm_head_weights": "use causal_decoder_set_lm_head(handle, lm_head_w) per-instance API (v5-06+)",
    "set_fused_lm_head_enabled": "use causal_decoder_set_lm_head(handle, lm_head_w) per-instance API (v5-06+)",
    "set_spm_mode": "set per-instance attribute model._rpu_spm_mode",
    "setup_fuse_lm_head": "use causal_decoder_set_lm_head(handle, lm_head_w) per-instance API (v5-06+)",
}


# P1b (debug-level session): names lazy-exposed via PEP 562 `__getattr__`.
# Kept OUT of `vars(rpu_backend)` to preserve the strict whitelist enforced
# by test_v4_public_api.py:101-128. `rpu_backend.set_debug_level(4)` still
# works at call time because attribute access routes through __getattr__.
_LAZY_DEBUG_API = frozenset({"set_debug_level", "get_debug_level", "TRACE"})
# Preserve the P7.1h convenience attributes without adding them to the locked
# four-name namespace or importing graph.lazy_init_guard during a cold package import.
_LAZY_DYNAMO_API = frozenset({
    "verify_lazy_init",
    "UninitializedLazyStateError",
    # Misnomers kept for the shipped wheel's API; see graph/lazy_init_guard.py.
    "freeze_for_dynamo",
    "DynamoUnsafeLazyStateError",
})
_DEPRECATED_GRAPH_API = {
    "GraphCache": "rpu_backend.graph.GraphCache",
    "GraphSignature": "rpu_backend.graph.GraphSignature",
    "get_default_graph_cache": "rpu_backend.graph.get_default_graph_cache",
}


def __getattr__(name: str):
    """v5.0 cutover with narrow debug/Dynamo lazy compatibility attributes.

    Migration hints per EXT-02 / ADR §2.4. Lazy branches return symbols without
    polluting ``vars(rpu_backend)`` or expanding ``__all__``.
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


# D-07 (v5-01b): NM2 iter-3 deprecation wrappers DELETED per CUT-02 +
# ADR §10 #1 (pure cutover; no DeprecationWarning, no tombstone shim).
# torch.rpu.set_chunk_size / set_spm_mode no longer bound; users get
# AttributeError naturally from the absent binding.

if _cpp_loaded:
    # P1a — 单个 atexit hook 串行:先 GraphCache.clear_all (释放所有 entry 的
    # private Queue_t),再 _control.shutdown (rhino-launch-kernel 的
    # KernelCache → QueueCache → BufferPool 拆解)。顺序错位会触发
    # `[WARN: ~HxilBufFactory:23]` 退出 noise (Queue_t 在 BufferPool 析构后
    # 才释放)。
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
    # `compile` / `dynamo_backend` used to ride along here. Dropped 2026-08-08:
    # the Dynamo frontend is FROZEN (graph/dynamo_backend.py docstring), and
    # shipping it on the wheel's public shim advertises a path that breaks the
    # moment it is pointed at an adapter that owns its own graph capture.
    # The module is still
    # importable for the frozen-state tests; it is just not an export.
    for optional_name in ("Graph",):
        if optional_name in globals():
            pkg_globals[optional_name] = globals()[optional_name]
    for private_name in _PRIVATE_RUNTIME_EXPORT_NAMES:
        if private_name in globals():
            pkg_globals[private_name] = globals()[private_name]
