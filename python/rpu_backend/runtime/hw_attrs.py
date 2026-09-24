"""A9 per-instance hardware-attribute validator.

CONTEXT D-16..D-32 (16 locked decisions) + reviews iter-2 amendments D-22/D-24/D-26
+ reviews iter-3 amendments D-29/D-52/D-53.
This module ships in Plan A (02-01-3); the wiring into Qwen3 + Pi05 adapters
ships in Plan B (02-02-3). Closest-shape analog: transformers/_causal_lm.py:22-97
weak-ref singleton pattern.

v5-04 HND-05: INTERNAL_HW_ATTRS_TRANSITIONAL was migrated from per-arch handle
names to neutral cross-arch names (`_rpu_decoder_handle /
_rpu_vlm_decoder_handle / _rpu_action_handle / _rpu_vision_handle`).

Design summary:
  - D-16: validation via `__class__` reassignment (NOT instance __setattr__).
  - D-18: dynamically-created subclass Validated_<OriginalClass> with
    (HwAttrValidatorMixin, OriginalCls) MRO.
  - D-19: cache subclasses keyed on original-class IDENTITY (not name),
    with sentinel __rpu_hw_attr_validated__ to prevent Validated_Validated_Foo.
  - D-21 + H4 iter-2: `_MODULE_STATE: WeakKeyDictionary[Module, RPUHwAttrState]`
    where every stamped module (root + all children) maps to a single shared
    state object — O(1) child→root lookup via `_MODULE_STATE.get(self)`.
  - D-22 + H6 iter-2: FIVE-function API (`validate_preinstall`, `validate_postinstall`,
    `install_hw_attr_validator`, `mark_first_forward_done`, `validate_existing_hw_attrs`
    back-compat shim). See context_amendment D-22.
  - D-26 + H5 iter-2: THREE frozensets — `PUBLIC_HW_ATTRS`,
    `INTERNAL_HW_ATTRS_TRANSITIONAL` (post-HND-05 neutral names), and
    `COLD_ONLY_ATTRS`.
  - D-28: monotonic policy — _rpu_swizzled/_swizzle_started/_weights_converted/
    _attn_tp/_max_seq_len/_gemm_weight/_pos_emb_fused are readonly after stamp;
    _rpu_cache is the one runtime-mutable internal name.
  - D-52 NM3 iter-3: `_ROOTS.add(root)` MUST run AFTER `_scan_modules` completes
    so a partial install failure leaves `_ROOTS` clean for retry.
  - D-29 NL1 iter-3: user-facing error message lists ONLY `PUBLIC_HW_ATTRS`;
    `INTERNAL_HW_ATTRS_TRANSITIONAL` is maintainer-only.

Pitfall P5: `_rpu_chunk_size` remains writable for compatibility paths that
actually consume it (currently Pi0.5 Gemma). Membership in `PUBLIC_HW_ATTRS`
only authorizes the write; it does not make the attribute a generic decoder
control. Public CausalDecoder policy uses `rpu_execution`; production hot
changes go through the shared `reconfigure_rpu_execution()` transaction.
"""
from __future__ import annotations

import weakref
from typing import Any, Optional

# v5-04 D3 ledger: errors moved to api.errors (was core.errors). Lazy import
# inside _raise_config_error preserves the runtime ↛ api DAG (DAG-02 / G8-acc):
# api/errors.py is a leaf module with no dependencies, but the DAG checker is
# strict — keep the import body-local so the module-level scan stays clean.


# --- D-26 + H5: authoritative whitelists (single source of truth) ---
PUBLIC_HW_ATTRS: frozenset[str] = frozenset({
    "_rpu_chunk_size",
    "_rpu_execution",
    "_rpu_spm_mode",
    "_rpu_warmup",
    "_rpu_debug_export",
})

# H5 reviews iter-2: subset of PUBLIC_HW_ATTRS that is COLD-ONLY (cannot be
# written after first forward). Matrix rationale:
#   _rpu_chunk_size:  hot-settable WITH manual reset_graph_cache() only on
#                     compatibility paths that consume it (currently Pi0.5
#                     Gemma). Other decoders use the shared execution session;
#                     hot changes use reconfigure_rpu_execution(), not assignment.
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

# Canonical reverse-census boundary for Python attributes that influence a
# physical execution plan, its route, or its replay identity.  This is kept
# separate from the write-admission sets below: several composite runtimes are
# not torch modules and therefore never pass through HwAttrValidatorMixin, but
# their planner state still requires one checked authority.
PLANNER_SENSITIVE_ATTRS: frozenset[str] = frozenset({
    # Public request and debug controls.
    "_rpu_chunk_size",
    "_rpu_debug_export",
    "_rpu_execution",
    "_rpu_spm_mode",
    "_rpu_warmup",
    # Planner capabilities, hand-off, and resolved-result state.
    "_rpu_decoder_topology",
    "_rpu_vision_num_cores",
    "_rpu_qwen3vl_vision_num_cores",
    "_rpu_execution_capabilities",
    "_rpu_execution_component_id",
    "_rpu_execution_component_generations",
    "_rpu_execution_components",
    "_rpu_decode_stage_descriptor",
    "_rpu_execution_generation",
    "_rpu_execution_graph_key_words",
    "_rpu_execution_handshake_required",
    "_rpu_execution_resolved",
    "_rpu_execution_resolved_generation",
    "_rpu_last_a6_plan",
    "_rpu_last_action_execution_plan",
    "_rpu_last_execution_plan",
    "_rpu_last_execution_plans",
    "_rpu_last_execution_receipts",
    "_rpu_legacy_prefix_pad16_request",
    "_rpu_legacy_kvinsert_pad16_request",
    "_rpu_g05_execution_component_generations",
    "_rpu_g05_execution_components",
    "_rpu_g05_execution_reconfigure_journal",
    "_rpu_planned_prefill_plan",
    "_rpu_vision_a6_plans",
    "_rpu_vision_control_snapshot",
    "_rpu_vision_last_a6_plan",
    "_rpu_vision_linear_acc32",
    "_rpu_vision_chunked_merger_w8a16",
    "_rpu_vision_chunked_merger_fp16",
    "_rpu_vision_compact_encoder_fp16",
    "_rpu_vision_compact_encoder_w8a16",
    "_rpu_vision_gelu_erf_mode",
    "_rpu_w8a16_imagetext_load_plan",
    "_rpu_qwen3_vl_awq_load_plan",
    "_rpu_qwen3_vl_runtime_load_plan",
    "_rpu_w8a16_staged_plan",
    # Physical geometry and layout identity.
    "_rpu_attn_tp",
    "_rpu_decoder_deepstack_hash",
    "_rpu_decoder_hidden_size",
    "_rpu_decoder_num_layers",
    "_rpu_deepstack_lang_layers",
    "_rpu_dinov3_hidden_size",
    "_rpu_dinov3_max_seq_len",
    "_rpu_dinov3_num_layers",
    "_rpu_effective_num_kv_heads",
    "_rpu_g05_max_seq_len",
    "_rpu_linear_num_cores",
    "_rpu_linear_partition",
    "_rpu_max_seq",
    "_rpu_max_seq_len",
    "_rpu_merger_hidden_size",
    "_rpu_num_layers",
    "_rpu_prefill_execution_alignment",
    "_rpu_kv_cache_layer_bank_size",
    "_rpu_rope_max_seq_len",
    "_rpu_siglip_batch_n",
    "_rpu_text_hidden_size",
    "_rpu_text_num_layers",
    "_rpu_vision_chunk_size_cap",
    "_rpu_vision_exact_chunk_size",
    "_rpu_vision_execution_chunk_size",
    "_rpu_vision_hidden_size",
    "_rpu_vision_num_layers",
    "_rpu_vision_padding_budget",
    "_rpu_vision_padding_rows",
    "_rpu_vision_spatial_merge_size",
    "_rpu_w_num_cores_1",
    # Graph lifecycle and replay-key state.
    "_rpu_adarms_graph_cache",
    "_rpu_decoder_graph_cache",
    "_rpu_dinov3_graph_cache",
    "_rpu_fused_denoise_graph_cache",
    "_rpu_gemma2_graph_cache",
    "_rpu_gemma4_graph_cache",
    "_rpu_gemma_graph_cache",
    "_rpu_prefill_kv_only",
    "_rpu_graph_cache",
    "_rpu_qwenpi05_graph_cache",
    "_rpu_siglip_graph_cache",
    "_rpu_text_graph_cache",
    "_rpu_vision_debug_graph",
    "_rpu_vision_graph_cache",
    "_rpu_vision_graph_key",
    "_rpu_vision_graph_sig",
    # KV ownership and physical route selectors.
    "_rpu_batch_decode_enabled",
    "_rpu_dinov3_kv_cache",
    "_rpu_kv_cache",
    "_rpu_vision_graph_disable",
    "_rpu_vision_has_dispatched",
    "_rpu_vision_kv_cache",
    "_rpu_vision_rope_disable",
})

# D-26 + v5-04 HND-05: transitional set used by adapter-internal state today.
# Phase 04 HND-05 collapsed the four per-arch handle names into four neutral
# cross-arch handle names listed below. Same NOT-monotonic semantics; the
# generic decoder atomically rotates its handle/finalizer on idempotent
# re-patching.
#   _rpu_decoder_handle:      CausalLM all-layers-once (Qwen3 / Llama / future Phi+Mistral)
#   _rpu_vlm_decoder_handle:  Pi05 inner Gemma VLM all-layers-once
#   _rpu_action_handle:       Pi05 AdaRMS expert all-layers-once
#   _rpu_vision_handle:       Pi05 SigLIP vision tower all-layers-once
#   _rpu_gemma2_handle:       PaliGemma2 inner Gemma2 all-layers-once
# Phase 2 02-REVIEW.md CR-01/CR-02/CR-03: extended with 8 names that the
# Qwen3 / Pi05 / fused-lm-head paths write through the validated tree.
# A9-SIGLIP-COLLISIONS (debug a9-validator-siglip-collisions, 2026-04-24):
# extended with 6 names that `_convert_siglip_weights_for_rpu` writes onto
# SigLIP encoder layers + the vit sentinel. Same shape as CR-02 but covers
# encoder-layer scope rather than embedding scope. Pi05 v2 SigLIP path
# (_adapter.py:1050) calls `_convert_siglip_weights_for_rpu(siglip_inner)`
# BEFORE `validate_postinstall(self._lerobot_policy)` runs at line 1178.
# A9-VALIDATOR-NOISE-SEED (debug a9-validator-noise-seed, 2026-04-24):
# test_pi05_e2e.py writes `_rpu_noise_seed` onto PI05Pytorch after validator
# install but before first forward so the RPU path's CPU noise generator can
# match the requested seed. It is test/internal harness state, not an A9 public
# hardware tuning attribute, and remains mutable for reruns on the same object.
INTERNAL_HW_ATTRS_TRANSITIONAL: frozenset[str] = frozenset({
    "_rpu_decoder_topology",
    "_rpu_vision_num_cores",
    "_rpu_qwen3vl_vision_num_cores",
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
    # v5-04 HND-05 neutral cross-arch handle names (replaces per-arch names).
    "_rpu_decoder_handle",        # CausalLM (Qwen3 / Llama / future)
    "_rpu_decoder_handle_finalizer",
    "_rpu_decoder_retirement_state",
    "_rpu_vlm_decoder_handle",    # Pi05 Gemma VLM
    "_rpu_vlm_decoder_handle_finalizer",
    "_rpu_vlm_decoder_retirement_state",
    "_rpu_action_handle",         # Pi05 AdaRMS expert
    "_rpu_action_handle_finalizer",
    "_rpu_action_retirement_state",
    "_rpu_vision_handle",         # Pi05 SigLIP vision
    "_rpu_vision_handle_finalizer",
    "_rpu_vision_retirement_state",
    "_rpu_gemma2_handle",         # PaliGemma2 Gemma2 decoder
    # CR-02: SigLIP embedding shape metadata (set by
    # _patch_siglip_embeddings_for_rpu before validator install). Phase 3 PR3
    # (PI05-05) collapses these into a single _rpu_emb_meta dict.
    "_rpu_kh", "_rpu_kw",
    "_rpu_cin_orig", "_rpu_cin_padded",
    "_rpu_cout", "_rpu_stride",
    # CR-03: fused-lm-head Python-side refcount keepalive (set/cleared by
    # set_fused_lm_head_enabled at runtime; toggles on/off → NOT monotonic).
    "_rpu_lm_head_w_keepalive",
    "_rpu_lm_head_scale_keepalive",
    "_rpu_lm_head_int8_cpu_ref",
    "_rpu_lm_head_scale_cpu_ref",
    "_rpu_lm_head_fp16_cpu_ref",
    "_rpu_embed_tokens_int8_cpu_ref",
    "_rpu_embed_tokens_scale_cpu_ref",
    # A9-SIGLIP: SigLIP encoder-conversion sentinel (set on the inner vit by
    # `_convert_siglip_weights_for_rpu` at siglip_converter.py:155 to
    # short-circuit double-swizzle on idempotent re-conversion). Same NOT-
    # monotonic shape as the neutral handle names.
    "_rpu_siglip_weights_converted",
    # A9-SIGLIP: per-encoder-layer constants (set by
    # `_convert_siglip_weights_for_rpu` at siglip_converter.py:149-153). They
    # are stable after first conversion but excluded from monotonic policy in
    # case a future re-conversion path re-sets them on the same instance.
    "_rpu_layer_idx",
    "_rpu_num_layers",
    "_rpu_num_heads",
    "_rpu_head_dim",
    "_rpu_intermediate_size",
    # A9-VALIDATOR-NOISE-SEED: Pi05 e2e harness deterministic-noise seed.
    "_rpu_noise_seed",
    # P7.1h L2: Dynamo lazy-init guard hazard preflight markers — adapters
    # stamp these at the end of their patch fn so freeze_for_dynamo() can
    # enforce that all `_rpu_*` attrs forward() reads are pre-installed.
    # See `python/rpu_backend/graph/lazy_init_guard.py`.
    "_rpu_lazy_init_checked",
    # Pre-2026-08-08 spelling of the marker above. Kept admitted because the
    # rename shipped with no back-compat, and BOTH of its failure modes are
    # bad: an out-of-tree adapter still stamping it either trips
    # `RPUConfigError: Unknown hardware attribute` here, or — worse, if the
    # write happens before the validator is installed — is silently SKIPPED by
    # `_verify_lazy_init`, which then returns CLEANLY and ships the lazy-init
    # hazard undetected. See `graph/lazy_init_guard.py::_LAZY_INIT_MARKERS`.
    "_rpu_dynamo_safe",
    "_rpu_required_attrs",
    # Qwen3-VL R-Phase 2: deepstack lang-layer indices stashed by
    # `_install_causal_decoder_forward` (runtime/decoder.py:505) so adapters
    # can ask "is DeepStack active?" at forward time without re-reading the
    # C++ side. Always written for arch="qwen3" / "qwen3_vl_text" — empty
    # list `[]` for plain Qwen3 (no DeepStack), non-empty for Qwen3-VL text
    # decoder. Re-set on idempotent re-patching → NOT monotonic.
    "_rpu_deepstack_lang_layers",
    # Explicit model capability: only the plain Qwen3 installer publishes
    # True; Llama and Qwen3-VL text publish False. Re-set on re-patching.
    "_rpu_batch_decode_enabled",
    # R-Phase 7 parity port: per-instance GraphCache + sig metadata written
    # by `_install_causal_decoder_forward` so the patched
    # `rpu_decoder_model_forward` can wrap the runner in
    # `with cache.capture(sig):`. Without this wrap the v5 graph runtime
    # dispatches each kernel in immediate mode and the text decoder pays a
    # `build_batch + enqueu_batch + invalidate` round-trip per kernel.
    # All four are re-set on idempotent re-patching → NOT monotonic.
    "_rpu_decoder_graph_cache",
    "_rpu_decoder_num_layers",
    "_rpu_decoder_hidden_size",
    "_rpu_decoder_deepstack_hash",
    "_rpu_prefill_execution_alignment",
    # Cold cache storage policy; published transactionally by the M-RoPE
    # installer. Existing cache views retain their own owners and layout.
    "_rpu_kv_cache_layer_bank_size",
    # Native singleton decode authority, published by the shared installer and
    # replaced only when the shared planner retires its generation/cost scope.
    "_rpu_decode_stage_descriptor",
    "_rpu_execution_generation",
    # Per-instance GraphCache for AdaRMS denoise forwards. Position is
    # updated through the replay inputs instead of defining a new signature.
    "_rpu_adarms_graph_cache",
    # Per-instance GraphCache for the SigLIP encoder. Images with the same
    # signature reuse a capture; the encoder has no mutable decode position.
    "_rpu_siglip_graph_cache",
    # Per-instance GraphCache for Pi0.5 Gemma prefill. Prefill resets the
    # cache position to zero, so matching prompt shapes can reuse a capture.
    "_rpu_gemma_graph_cache",
    # Dedicated Pi0.5 prefill entry returns only cache effects.
    "_rpu_prefill_kv_only",
    # The internal Gemma2 diagnostic owner uses the same Session retirement.
    "_rpu_gemma2_graph_cache",
    # Pi0.5 shared planner hand-off: one ephemeral immutable A6 result chosen
    # before Gemma GraphCache capture, consumed by the patched forward, then
    # removed. It is runtime-mutable and never a public tuning knob.
    "_rpu_planned_prefill_plan",
    # Task 4.2 (pi05-num-steps-fuse): Pi05DenoiseStepModel handle + safety-net
    # validated flag. Set on PI05Pytorch after _install_fused_denoise_handle.
    # `_rpu_fused_denoise_handle` is None when fuse disabled (RPU_PI05_FUSED_DENOISE=0).
    # `_rpu_fuse_validated` is set True after first safety-net pass in _run_denoise.
    "_rpu_fused_denoise_handle",
    "_rpu_fused_denoise_handle_finalizer",
    "_rpu_fused_denoise_retirement_state",
    "_rpu_fuse_validated",
    "_rpu_fused_denoise_graph_cache",
    # Pi05 warm-inference host caches. The RPU device is fixed by the one-live-
    # model policy; AdaRMS conditioning is deterministic per num_steps; and the
    # denoise mask is LRU-1 keyed by prefix-mask content + shape/device metadata.
    "_rpu_runtime_device",
    # Frozen by the Pi05 adapter/session at construction; published only after
    # the public-only preinstall scan and immutable once the tree is stamped.
    "_rpu_legacy_prefix_pad16_request",
    "_rpu_legacy_kvinsert_pad16_request",
    "_rpu_adarms_cond_cache",
    "_rpu_denoise_mask_cache",
    # SigLIP-batch (pi05-num-steps-fuse, 2026-05-31): number of camera images
    # the RPU SigLIP forward packs into one seq=N*256 fused encoder forward.
    # Set on PI05Pytorch by Pi05Adapter; read by `_patched_embed_prefix`
    # (patches.py) to choose stack-batch vs per-image loop. Re-set on idempotent
    # re-patching → NOT monotonic.
    "_rpu_siglip_batch_n",
    # Qwen3.5 text runtime. The fused handle, cache, planner state, and bounded
    # prefill Graph are owned by the namespace rather than scattered over the
    # HF module, but the namespace name itself still crosses the A9 boundary.
    "_rpu_qwen3_5_text_install_started",
    "_rpu_qwen3_5",
    "_rpu_qwen3_5_moe_text_install_started",
    "_rpu_qwen3_5_moe",
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
    "_rpu_vision_linear_acc32",
    "_rpu_vision_gelu_erf_mode",
    "_rpu_vision_graph_cache",
    "_rpu_vision_graph_key", "_rpu_vision_graph_sig",
    "_rpu_vision_debug_graph",
    "_rpu_vision_spatial_merge_size",
    "_rpu_vision_num_layers", "_rpu_vision_hidden_size",
    # Qwen3.5 lazy Vision planner controls and per-shape results.  These are
    # published after the full model tree has already been A9-stamped.
    "_rpu_vision_chunk_size_cap", "_rpu_vision_exact_chunk_size",
    "_rpu_vision_padding_rows", "_rpu_vision_padding_budget",
    "_rpu_vision_control_snapshot", "_rpu_vision_last_a6_plan",
    "_rpu_vision_a6_plans", "_rpu_vision_has_dispatched",
})

# D-28: names that are monotonic / readonly-after-stamp (subset of transitional).
# Names outside this subset are mutable adapter state (for example graph/cache
# objects and Pi05 warm-inference caches).
_MONOTONIC_INTERNAL_HW_ATTRS: frozenset[str] = frozenset({
    "_rpu_decoder_topology",
    "_rpu_vision_num_cores",
    "_rpu_qwen3vl_vision_num_cores",
    "_rpu_swizzled",
    "_rpu_swizzle_started",
    "_rpu_attn_tp",
    "_rpu_effective_num_kv_heads",
    "_rpu_max_seq_len",
    "_rpu_weights_converted",
    "_rpu_gemm_weight",
    "_rpu_pos_emb_fused",
    # Eager dispatch reads these as the layout of already-swizzled bytes.
    "_rpu_linear_partition",
    "_rpu_linear_num_cores",
    "_rpu_legacy_prefix_pad16_request",
    "_rpu_legacy_kvinsert_pad16_request",
})


# --- D-21 + H4: WeakKeyDictionary state (module-scope; GC-tracked) ---
class RPUHwAttrState:
    """Per-ROOT state — first_forward_done flag. Shared by every stamped module
    in the tree; H4 iter-2 maps every child module to the SAME state object via
    _MODULE_STATE so mixin __setattr__ has O(1) access from any node."""

    __slots__ = ("first_forward_done",)

    def __init__(self) -> None:
        self.first_forward_done = False


# H4 reviews iter-2: every stamped module (root + children) maps to the SAME
# shared state object. Mixin `__setattr__` looks up `state = _MODULE_STATE.get(self)`
# for O(1) child→root state access. `WeakKeyDictionary` cleans up entries when
# a module is GC'd. This REPLACES iter-1's `_STATE[root]` which only covered
# the root and silently broke the first-forward gate for every child write.
_MODULE_STATE: "weakref.WeakKeyDictionary[Any, RPUHwAttrState]" = weakref.WeakKeyDictionary()
_VALIDATED_CLS_CACHE: "dict[type, type]" = {}

# Tracks installed ROOTS (not children) so re-entry (D-23) can be detected
# without iterating all stamped modules. Children live only in _MODULE_STATE.
_ROOTS: "weakref.WeakSet[Any]" = weakref.WeakSet()


# --- D-18: mixin + D-28 + H5 policy ---
class HwAttrValidatorMixin:
    """Mixin stamped into every module in the RPU model tree via __class__.

    __setattr__ intercepts writes to `_rpu_*` names; delegates to
    _validate_hw_attr_write for D-26/D-27/D-28/H5-COLD_ONLY policy, then
    super().__setattr__.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_rpu_"):
            _validate_hw_attr_write(self, name, value)
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name.startswith("_rpu_"):
            _validate_hw_attr_write(self, name, None)
        super().__delattr__(name)


def _validate_hw_attr_write(module: Any, name: str, value: Any) -> None:
    """D-28 + H5 COLD_ONLY policy dispatch.

    - name in PUBLIC_HW_ATTRS:
        - cold-set (first_forward_done=False): always OK.
        - hot-set (first_forward_done=True):
            - if name in COLD_ONLY_ATTRS: raise RPUConfigError (H5 iter-2).
            - else: OK (user triggers reset_graph_cache() per D-25).
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
        # H5 iter-2: COLD_ONLY_ATTRS hot-set after first forward raises.
        if name in COLD_ONLY_ATTRS:
            state = _MODULE_STATE.get(module)
            if state is not None and state.first_forward_done:
                _raise_config_error(
                    f"{name} is cold-only (H5 matrix: {sorted(COLD_ONLY_ATTRS)}); "
                    f"writing after first forward is unsupported. Create a fresh "
                    f"model instance to change it. See docs/api_reference.md#Migration."
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

    # Unknown _rpu_* name (NL1 reviews iter-3: per D-29, user-facing error
    # messages list ONLY PUBLIC_HW_ATTRS. INTERNAL_HW_ATTRS_TRANSITIONAL is
    # maintainer-only — visible in code + plan docs + Plan B test, never in
    # the exception text).
    _raise_config_error(
        f"Unknown hardware attribute `{name}` on {type(module).__name__}. "
        f"Allowed public names: {sorted(PUBLIC_HW_ATTRS)}. "
        f"(Adapter-internal `_rpu_*` names are accepted during swizzling "
        f"but not documented.)"
    )


def _raise_config_error(msg: str) -> None:
    """Raise RPUConfigError. v5-04: late-import preserves the runtime ↛ api DAG."""
    from rpu_backend.runtime import RPUConfigError
    raise RPUConfigError(msg)


# --- D-22 + H6: five-function API surface ---

def validate_preinstall(root: Any) -> None:
    """H6 iter-2: PRE-stamp scan. Called BEFORE the adapter writes any
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
        # iter-1 allowed INTERNAL_HW_ATTRS_TRANSITIONAL here; H6 removes that
        # allowance because pre-install the adapter has NOT yet written those
        # names — any that exist are stale or user-injected.
        _raise_config_error(
            f"Unexpected `_rpu_*` attribute `{name}` on "
            f"{type(module).__name__} BEFORE `.to('rpu')` adapter setup. "
            f"Only PUBLIC_HW_ATTRS {sorted(PUBLIC_HW_ATTRS)} are allowed pre-install. "
            f"Clear stale state or load a fresh model instance (D-27)."
        )


def validate_postinstall(root: Any) -> None:
    """H6 iter-2: POST-state scan. Called after the adapter writes its
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
    """D-22 back-compat alias — dispatches to `validate_postinstall`.

    Historically this was the single validation entry point and its callers ran
    after adapter-internal state had been assembled. That is the post-state
    semantics. The H6 iter-2 split makes the true pre-install semantics
    available via `validate_preinstall`; this alias keeps existing callers
    working without defining ready-marker ordering.
    """
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
    the ROOT's shared RPUHwAttrState object (H4 iter-2).

    Re-entry behavior (D-23):
      - If root already in _ROOTS: re-validate via `validate_postinstall`,
        preserve first_forward_done, do NOT re-stamp.

    Raises RPUConfigError (D-20) if __class__ reassignment fails for any
    module (L2 iter-2 test exercises this via a __slots__-bearing class).

    NM3 reviews iter-3 (D-52): `_ROOTS.add(root)` MUST run AFTER the
    `_scan_modules` tree-walk completes. If any child raises (e.g., slotted
    child per L2), `_ROOTS` is NOT populated, so a future `install_*` call
    can retry. Iter-2's pre-walk `_ROOTS.add` corrupted re-entry (D-23
    returned success without finishing install).
    """
    if root in _ROOTS:
        # D-23 re-entry
        validate_postinstall(root)
        return
    shared_state = RPUHwAttrState()
    # NM3 iter-3 (D-52): scan FIRST. If scan raises, _ROOTS stays clean.
    _scan_modules(root, lambda m: _install_on_one(m, shared_state))
    # NM3 iter-3 (D-52): only mark success AFTER scan completes. Partial
    # rollback on failure is best-effort (mixin stamps on already-walked
    # children remain; they are no-ops for mutation gates because those
    # children still have _MODULE_STATE entries and the shared state's
    # first_forward_done is still False).
    _ROOTS.add(root)


def _install_on_one(module: Any, shared_state: RPUHwAttrState) -> None:
    """Stamp `module`'s __class__ AND register it in _MODULE_STATE with
    the ROOT's shared state object (H4 iter-2)."""
    # H4 iter-2: every stamped module (root + all children) shares one state.
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
        # D-20 + L2 iter-2: NO silent skip. Surface the path + type.
        _raise_config_error(
            f"Cannot install A9 validator on "
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
    "COLD_ONLY_ATTRS",  # H5 iter-2
    # Functions (D-22 + H6 API)
    "validate_preinstall",   # H6 iter-2 new
    "validate_postinstall",  # H6 iter-2 new
    "validate_existing_hw_attrs",  # back-compat shim (dispatches to validate_postinstall)
    "install_hw_attr_validator",
    "mark_first_forward_done",
    # Types (exported for test introspection)
    "HwAttrValidatorMixin",
    "RPUHwAttrState",
]
