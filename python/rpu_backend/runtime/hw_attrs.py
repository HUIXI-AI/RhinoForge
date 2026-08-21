"""Per-instance RPU hardware-attribute validation.

Design summary:
  - Validation is installed through a cached ``Validated_<OriginalClass>``
    subclass and ``__class__`` reassignment.
  - Every stamped module maps to one root-owned state in a
    ``WeakKeyDictionary``.
  - Public, internal, cold-only, and monotonic names have distinct write
    policies.
  - A root is marked installed only after its full module scan succeeds.

``_rpu_chunk_size`` remains writable only for compatibility paths that consume
it. Public CausalDecoder execution policy is the cold ``rpu_execution`` map.
"""
from __future__ import annotations

import weakref
from typing import Any, Optional

# Keep configuration-error imports function-local so runtime does not import
# the higher-level API package.


# Authoritative public attribute allowlist.
PUBLIC_HW_ATTRS: frozenset[str] = frozenset({
    "_rpu_chunk_size",
    "_rpu_execution",
    "_rpu_spm_mode",
    "_rpu_warmup",
    "_rpu_debug_export",
})

# Public attributes that cannot be written after the first forward:
#   _rpu_chunk_size:  hot-settable WITH manual reset_graph_cache() only on
#                     compatibility paths that consume it (currently Pi0.5
#                     Gemma). Other decoders use cold _rpu_execution.
#   _rpu_spm_mode:    cold-only — changes kernel code paths (SPM vs DDR);
#                     hot change unsound.
#   _rpu_warmup:      cold-only — constructor-time flag; runtime change
#                     has no effect and implies user misunderstanding.
#   _rpu_debug_export: hot-settable — debug flag only; no SPM or graph-cache
#                     implications.
COLD_ONLY_ATTRS: frozenset[str] = frozenset({
    "_rpu_execution",
    "_rpu_spm_mode",
    "_rpu_warmup",
})

# Adapter-internal state accepted by the validator. Handle names are neutral
# across architectures, and the generic decoder may rotate a handle/finalizer
# atomically on idempotent reinstallation.
#   _rpu_decoder_handle:      CausalLM all-layers-once (Qwen3 / Llama / future Phi+Mistral)
#   _rpu_vlm_decoder_handle:  Pi05 inner Gemma VLM all-layers-once
#   _rpu_action_handle:       Pi05 AdaRMS expert all-layers-once
#   _rpu_vision_handle:       Pi05 SigLIP vision tower all-layers-once
# The set includes metadata written by Qwen3, Pi0.5, SigLIP, fused-lm-head, and
# deterministic-noise paths before or after validator installation.
INTERNAL_HW_ATTRS_TRANSITIONAL: frozenset[str] = frozenset({
    "_rpu_swizzled",
    "_rpu_swizzle_started",
    "_rpu_attn_tp",
    "_rpu_effective_num_kv_heads",
    "_rpu_max_seq_len",
    "_rpu_cache",
    "_rpu_weights_converted",
    "_rpu_gemm_weight",
    "_rpu_pos_emb_fused",
    # Single-rule layout contract: the (partition, num_cores) that
    # `convert_linear_weights_inplace` actually chose for this Linear, recorded
    # so eager `aten::linear` honours it instead of re-deriving from the shape.
    # Written per-Linear at swizzle time, i.e. between validate_preinstall and
    # validate_postinstall. See runtime/weights.py `_record_linear_layout`.
    "_rpu_linear_partition",
    "_rpu_linear_num_cores",
    # Neutral cross-architecture handle names.
    "_rpu_decoder_handle",        # CausalLM (Qwen3 / Llama / future)
    "_rpu_decoder_handle_finalizer",
    "_rpu_vlm_decoder_handle",    # Pi05 Gemma VLM
    "_rpu_vlm_decoder_handle_finalizer",
    "_rpu_action_handle",         # Pi05 AdaRMS expert
    "_rpu_action_handle_finalizer",
    "_rpu_vision_handle",         # Pi05 SigLIP vision
    "_rpu_vision_handle_finalizer",
    # SigLIP embedding shape metadata set before validator installation.
    "_rpu_kh", "_rpu_kw",
    "_rpu_cin_orig", "_rpu_cin_padded",
    "_rpu_cout", "_rpu_stride",
    # Fused-lm-head Python-side reference keepalives (set/cleared by
    # set_fused_lm_head_enabled at runtime; toggles on/off → NOT monotonic).
    "_rpu_lm_head_w_keepalive",
    "_rpu_lm_head_scale_keepalive",
    "_rpu_lm_head_int8_cpu_ref",
    "_rpu_lm_head_scale_cpu_ref",
    "_rpu_lm_head_fp16_cpu_ref",
    "_rpu_embed_tokens_int8_cpu_ref",
    "_rpu_embed_tokens_scale_cpu_ref",
    # SigLIP conversion sentinel used to prevent a repeated transformation.
    "_rpu_siglip_weights_converted",
    # SigLIP per-layer constants. They are stable after conversion but remain
    # outside the monotonic set for idempotent adapter installation.
    "_rpu_layer_idx",
    "_rpu_num_layers",
    "_rpu_num_heads",
    "_rpu_head_dim",
    "_rpu_intermediate_size",
    # Pi0.5 deterministic-noise seed.
    "_rpu_noise_seed",
    # Dynamo lazy-init guard markers. Adapters
    # stamp these at the end of their patch fn so freeze_for_dynamo() can
    # enforce that all `_rpu_*` attrs forward() reads are pre-installed.
    # See `python/rpu_backend/graph/lazy_init_guard.py`.
    "_rpu_lazy_init_checked",
    # Compatibility spelling accepted for out-of-tree adapters. Both marker
    # names are checked by ``graph.lazy_init_guard``.
    "_rpu_dynamo_safe",
    "_rpu_required_attrs",
    # DeepStack language-layer indices stored by the decoder installer so adapters
    # can ask "is DeepStack active?" at forward time without re-reading the
    # C++ side. Always written for arch="qwen3" / "qwen3_vl_text" — empty
    # list `[]` for plain Qwen3 (no DeepStack), non-empty for Qwen3-VL text
    # decoder. Re-set on idempotent re-patching → NOT monotonic.
    "_rpu_deepstack_lang_layers",
    # Explicit model capability: only the plain Qwen3 installer publishes
    # True; Llama and Qwen3-VL text publish False. Re-set on re-patching.
    "_rpu_batch_decode_enabled",
    # Per-instance GraphCache and signature metadata written by the decoder
    # installer so the patched
    # `rpu_decoder_model_forward` can wrap the runner in
    # `with cache.capture(sig):`. Without this wrap the graph runtime
    # dispatches each kernel in immediate mode and the text decoder pays a
    # batch submission round-trip per operator.
    # All four are re-set on idempotent re-patching → NOT monotonic.
    "_rpu_decoder_graph_cache",
    "_rpu_decoder_num_layers",
    "_rpu_decoder_hidden_size",
    "_rpu_decoder_deepstack_hash",
    "_rpu_prefill_execution_alignment",
    # Per-instance GraphCache for the AdaRMS forward wrapper.
    # Same role as `_rpu_decoder_graph_cache` above — lets the patched
    # `rpu_adarms_model_forward` wrap `torch.ops.rpu.adarms_forward` in
    # `with cache.capture(sig):` so 5 denoise steps + cross-forward denoise
    # loops hit the sync-only REPLAY fast path. Its signature omits position
    # because this path resets position before each captured forward.
    "_rpu_adarms_graph_cache",
    # Per-instance GraphCache for the SigLIP encoder. SigLIP has no position
    # concept (bidirectional with fixed sequence length), so the signature
    # is naturally stable across forwards — 3 same-shape images per
    # inference share one cached entry (1 BUILD + 2 REPLAY within forward).
    "_rpu_siglip_graph_cache",
    # Per-instance GraphCache for the Pi0.5 Gemma VLM
    # prefill wrap in adapters/pi05/gemma.py. VLM prefill always runs at
    # position=0 after a cache reset, so the signature
    # drops position. 1 prefill call per inference → cross-forward same
    # prompt shape all REPLAY.
    "_rpu_gemma_graph_cache",
    # Pi0.5 shared planner hand-off: one ephemeral exact chunk chosen before
    # Gemma GraphCache capture, consumed by the patched forward, then removed.
    # It is intentionally runtime-mutable and never a public tuning knob.
    "_rpu_planned_prefill_chunk_size",
    # Pi05DenoiseStepModel handle and validation flag.
    # `_rpu_fused_denoise_handle` is None when fuse disabled (RPU_PI05_FUSED_DENOISE=0).
    # `_rpu_fuse_validated` is set True after first safety-net pass in _run_denoise.
    "_rpu_fused_denoise_handle",
    "_rpu_fused_denoise_handle_finalizer",
    "_rpu_fuse_validated",
    "_rpu_fused_denoise_graph_cache",
    # Pi05 warm-inference host caches. The RPU device is fixed by the one-live-
    # model policy; AdaRMS conditioning is deterministic per num_steps; and the
    # denoise mask is LRU-1 keyed by prefix-mask content + shape/device metadata.
    "_rpu_runtime_device",
    "_rpu_adarms_cond_cache",
    "_rpu_denoise_mask_cache",
    # Number of camera images
    # the RPU SigLIP forward packs into one seq=N*256 fused encoder forward.
    # Set on PI05Pytorch by Pi05Adapter; read by `_patched_embed_prefix`
    # (patches.py) to choose stack-batch vs per-image loop. Re-set on idempotent
    # re-patching → NOT monotonic.
    "_rpu_siglip_batch_n",
    # Qwen3.5 text runtime. The fused handle, cache, planner state, and bounded
    # prefill Graph are owned by the namespace rather than scattered over the
    # HF module, but the namespace name itself still crosses the validator.
    "_rpu_qwen3_5_text_install_started",
    "_rpu_qwen3_5",
    # Qwen3.5 lazy Vision install. The complete model tree is stamped after
    # text installation, so the first image forward must be able to publish
    # these transactional/runtime fields through the validator. They remain
    # maintainer-only and are not part of PUBLIC_HW_ATTRS.
    "_rpu_qwen3_5_vision_conversion_started",
    "_rpu_qwen3_5_vision_weights_converted",
    "_rpu_qwen3_5_had_instance_forward",
    "_rpu_qwen3_5_original_forward",
    "_rpu_q_w", "_rpu_k_w", "_rpu_v_w",
    "_rpu_q_b", "_rpu_k_b", "_rpu_v_b",
    "_rpu_o_b", "_rpu_fc1_b", "_rpu_fc2_b",
    "_rpu_vision_installing",
    "_rpu_vision_installing_handle",
    "_rpu_vision_installing_finalizer",
    "_rpu_vision_freq_cos", "_rpu_vision_freq_sin",
    "_rpu_vision_position_idx_keepalive",
    "_rpu_vision_position_idx_cpu", "_rpu_vision_position_idx_grid",
    "_rpu_vision_position_idx_cache_key",
    "_rpu_vision_step0",
    "_rpu_vision_patch_embed_w", "_rpu_vision_patch_embed_b",
    "_rpu_vision_pe_w_swz", "_rpu_vision_pe_b_rpu",
    "_rpu_vision_step0_pos", "_rpu_vision_step0_pos_grid",
    "_rpu_vision_step0_pos_temporal_num_frames",
    "_rpu_vision_step0_pos_camera_batch_count",
    "_rpu_vision_merger_refs", "_rpu_vision_has_merger",
    "_rpu_vision_kv_cache",
    "_rpu_vision_graph_disable", "_rpu_vision_rope_disable",
    "_rpu_vision_graph_cache",
    "_rpu_vision_graph_key", "_rpu_vision_graph_sig",
    "_rpu_vision_debug_graph",
    "_rpu_vision_spatial_merge_size",
    "_rpu_vision_num_layers", "_rpu_vision_hidden_size",
})

# Names that are monotonic after validator installation.
# Names outside this subset are mutable adapter state (for example graph/cache
# objects and Pi05 warm-inference caches).
_MONOTONIC_INTERNAL_HW_ATTRS: frozenset[str] = frozenset({
    "_rpu_swizzled",
    "_rpu_swizzle_started",
    "_rpu_attn_tp",
    "_rpu_effective_num_kv_heads",
    "_rpu_max_seq_len",
    "_rpu_weights_converted",
    "_rpu_gemm_weight",
    "_rpu_pos_emb_fused",
})


# Weak-reference state shared across each stamped module tree.
class RPUHwAttrState:
    """Per-ROOT state — first_forward_done flag. Shared by every stamped module
    in the tree so mixin ``__setattr__`` has O(1) access from any node."""

    __slots__ = ("first_forward_done",)

    def __init__(self) -> None:
        self.first_forward_done = False


# Every stamped module maps to the same root-owned state. Weak keys release the
# mapping when modules are collected.
_MODULE_STATE: "weakref.WeakKeyDictionary[Any, RPUHwAttrState]" = weakref.WeakKeyDictionary()
_VALIDATED_CLS_CACHE: "dict[type, type]" = {}

# Track installed roots so re-entry can be detected
# without iterating all stamped modules. Children live only in _MODULE_STATE.
_ROOTS: "weakref.WeakSet[Any]" = weakref.WeakSet()


# Attribute-validation mixin.
class HwAttrValidatorMixin:
    """Mixin stamped into every module in the RPU model tree via __class__.

    __setattr__ intercepts writes to `_rpu_*` names; delegates to
    ``_validate_hw_attr_write`` enforces the public and internal policies, then
    super().__setattr__.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_rpu_"):
            _validate_hw_attr_write(self, name, value)
        super().__setattr__(name, value)


def _validate_hw_attr_write(module: Any, name: str, value: Any) -> None:
    """Dispatch the cold-only, monotonic, mutable, and unknown-name policies.

    - name in PUBLIC_HW_ATTRS:
        - cold-set (first_forward_done=False): always OK.
        - hot-set (first_forward_done=True):
            - if name in COLD_ONLY_ATTRS: raise RPUConfigError.
            - else: OK after the caller resets the graph cache when required.
    - name in _MONOTONIC_INTERNAL_HW_ATTRS:
        - first write: OK
        - subsequent writes (including `= False`, clear, re-assign): raise RPUConfigError
    - name == "_rpu_cache":
        - always mutable (runtime updates per Pi05 select_action)
    - name in INTERNAL_HW_ATTRS_TRANSITIONAL \\ _MONOTONIC:
        - mutable adapter state and warm-inference caches
    - unknown `_rpu_*` name: raise RPUConfigError
    """
    if name in PUBLIC_HW_ATTRS:
        # Cold-only attributes cannot change after first forward.
        if name in COLD_ONLY_ATTRS:
            state = _MODULE_STATE.get(module)
            if state is not None and state.first_forward_done:
                _raise_config_error(
                    f"{name} is cold-only ({sorted(COLD_ONLY_ATTRS)}); "
                    f"writing after first forward is unsupported. Create a fresh "
                    "model instance to change it."
                )
        return  # cold-set + non-COLD_ONLY hot-set both valid

    if name == "_rpu_cache":
        return  # runtime-mutable

    if name in _MONOTONIC_INTERNAL_HW_ATTRS:
        # Check if already set on this module — if yes, raise.
        if name in vars(module):
            _raise_config_error(
                f"{name} is an adapter-internal monotonic flag on "
                f"{type(module).__name__}; modification after .to('rpu') is "
                f"forbidden. If you intend to re-initialize the adapter, "
                f"create a fresh instance (RPUSingleHandleError guards "
                f"against double-live)."
            )
        return  # first write — OK

    if name in INTERNAL_HW_ATTRS_TRANSITIONAL:
        return  # mutable transitional adapter state / warm-inference caches

    # User-facing errors list only public names; internal names remain an
    # implementation detail.
    _raise_config_error(
        f"Unknown hardware attribute `{name}` on {type(module).__name__}. "
        f"Allowed public names: {sorted(PUBLIC_HW_ATTRS)}. "
        f"(Adapter-internal `_rpu_*` names are accepted during swizzling "
        f"but not documented.)"
    )


def _raise_config_error(msg: str) -> None:
    """Raise ``RPUConfigError`` without a module-level API dependency."""
    from rpu_backend.runtime import RPUConfigError
    raise RPUConfigError(msg)


# Validation entry points.

def validate_preinstall(root: Any) -> None:
    """Scan before the adapter writes any
    INTERNAL_HW_ATTRS_TRANSITIONAL state. Allows ONLY PUBLIC_HW_ATTRS.

    Raises RPUConfigError if any `_rpu_*` attr exists on any module in the
    tree that is NOT in PUBLIC_HW_ATTRS.
    """
    _scan_modules(root, _validate_preinstall_on_one)


def _validate_preinstall_on_one(module: Any) -> None:
    for name in list(vars(module).keys()):
        if not name.startswith("_rpu_"):
            continue
        if name in PUBLIC_HW_ATTRS:
            continue
        # Internal names cannot exist before the adapter has written them.
        _raise_config_error(
            f"Unexpected `_rpu_*` attribute `{name}` on "
            f"{type(module).__name__} BEFORE `.to('rpu')` adapter setup. "
            f"Only PUBLIC_HW_ATTRS {sorted(PUBLIC_HW_ATTRS)} are allowed pre-install. "
            "Clear stale state or load a fresh model instance."
        )


def validate_postinstall(root: Any) -> None:
    """Scan after the adapter writes its
    internal transitional state but before it publishes `_rpu_swizzled=True`
    and before `install_hw_attr_validator` stamps the tree. Allows
    PUBLIC_HW_ATTRS ∪ INTERNAL_HW_ATTRS_TRANSITIONAL.
    """
    _scan_modules(root, _validate_postinstall_on_one)


def _validate_postinstall_on_one(module: Any) -> None:
    for name in list(vars(module).keys()):
        if not name.startswith("_rpu_"):
            continue
        if name in PUBLIC_HW_ATTRS:
            continue
        if name in INTERNAL_HW_ATTRS_TRANSITIONAL:
            continue
        _raise_config_error(
            f"Unexpected `_rpu_*` attribute `{name}` on "
            f"{type(module).__name__} AFTER adapter setup. "
            f"Allowed post-install: PUBLIC {sorted(PUBLIC_HW_ATTRS)} ∪ "
            f"TRANSITIONAL {sorted(INTERNAL_HW_ATTRS_TRANSITIONAL)}. "
            f"Either the adapter has a typo, or a user set an unexpected "
            f"`_rpu_*` name during patching."
        )


def validate_existing_hw_attrs(root: Any) -> None:
    """Compatibility alias with post-install validation semantics."""
    validate_postinstall(root)


def _scan_modules(root: Any, fn) -> None:
    """Walk root.modules() if it's an nn.Module; otherwise call fn on root."""
    modules_fn = getattr(root, "modules", None)
    if modules_fn is not None:
        for m in modules_fn():
            fn(m)
    else:
        fn(root)


def install_hw_attr_validator(root: Any) -> None:
    """Idempotent stamp. Walks root.modules(), swaps __class__ to
    Validated_<Orig> subclass, registers each module in _MODULE_STATE with
    the root's shared ``RPUHwAttrState`` object.

    Re-entry behavior:
      - If root already in _ROOTS: re-validate via `validate_postinstall`,
        preserve first_forward_done, do NOT re-stamp.

    Raises ``RPUConfigError`` if any module rejects ``__class__`` reassignment.
    The root is marked installed only after the full scan succeeds, allowing a
    failed partial install to be retried.
    """
    if root in _ROOTS:
        # Idempotent re-entry.
        validate_postinstall(root)
        return
    shared_state = RPUHwAttrState()
    # Scan first so a failure leaves the root set clean.
    _scan_modules(root, lambda m: _install_on_one(m, shared_state))
    # Mark success only after the scan completes. Partial
    # rollback on failure is best-effort (mixin stamps on already-walked
    # children remain; they are no-ops for mutation gates because those
    # children still have _MODULE_STATE entries and the shared state's
    # first_forward_done is still False).
    _ROOTS.add(root)


def _install_on_one(module: Any, shared_state: RPUHwAttrState) -> None:
    """Stamp `module`'s __class__ AND register it in _MODULE_STATE with
    the root's shared state object."""
    # Every stamped module in a tree shares one state.
    _MODULE_STATE[module] = shared_state

    orig_cls = type(module)
    if getattr(orig_cls, "__rpu_hw_attr_validated__", False):
        return  # already stamped
    if orig_cls not in _VALIDATED_CLS_CACHE:
        new_cls = type(
            f"Validated_{orig_cls.__name__}",
            (HwAttrValidatorMixin, orig_cls),
            {
                "__rpu_hw_attr_validated__": True,
                "__rpu_hw_attr_original_cls__": orig_cls,
            },
        )
        _VALIDATED_CLS_CACHE[orig_cls] = new_cls
    try:
        module.__class__ = _VALIDATED_CLS_CACHE[orig_cls]
    except TypeError as exc:
        # Surface the module type instead of silently skipping it.
        _raise_config_error(
            f"Cannot install the RPU hardware-attribute validator on "
            f"{orig_cls.__module__}.{orig_cls.__name__} (module={module!r}): {exc}. "
            f"This module type does not accept __class__ reassignment "
            f"(likely __slots__-bearing, C-backed, or frozen)."
        )


def mark_first_forward_done(root: Any) -> None:
    """Flip first_forward_done on the ROOT's shared state. Because
    _MODULE_STATE points every child to the SAME state object, every child's
    mixin sees the flip. Idempotent."""
    try:
        st = _MODULE_STATE.get(root)
    except TypeError:
        # Objects that cannot be weak-referenced cannot have been stamped by
        # install_hw_attr_validator. Treat synthetic/standalone forward owners
        # as uninstalled instead of changing their forward error contract.
        return
    if st is not None:
        st.first_forward_done = True


__all__ = [
    # Frozensets
    "PUBLIC_HW_ATTRS",
    "INTERNAL_HW_ATTRS_TRANSITIONAL",
    "COLD_ONLY_ATTRS",
    # Functions
    "validate_preinstall",
    "validate_postinstall",
    "validate_existing_hw_attrs",  # back-compat shim (dispatches to validate_postinstall)
    "install_hw_attr_validator",
    "mark_first_forward_done",
    # Types
    "HwAttrValidatorMixin",
    "RPUHwAttrState",
]
