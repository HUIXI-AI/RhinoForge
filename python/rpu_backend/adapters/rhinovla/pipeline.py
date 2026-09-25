#!/usr/bin/env python
"""RhinoVLA whole-pipeline orchestrator for one predict() call.

Qwen3-VL vision and text prefill produce the prefix KV. Text layers [10:28]
feed the RhinoVLA action expert and its denoise loop before action decoding.
RPU setup runs once in __init__; predict() reuses the installed pipeline."""
import math, os, sys, time
import torch
from .prefix import prepare_prefix_inputs_from_vision_outputs, get_cache_layer
from .action import denoise_schedule, rpu_denoise
from rpu_backend.api.errors import RPUBackendError
from rpu_backend.api._execution import execution_serialized
from rpu_backend.runtime import rpu_env_bool
from rpu_backend.runtime.chunk_envelope import ChunkEnvelope, make_lookup

# Preserve the 2B backbone boundary previously supplied by qwen3_vl.text.
# This is an inherited admission ceiling, not official full-chain certification.
_TEXT_ENVELOPE = make_lookup(
    {("qwen3_vl_text", 28, 2048): ChunkEnvelope(336, 320)},
    "rpu_backend/adapters/rhinovla/pipeline.py::_TEXT_ENVELOPE",
)


def _trace(msg):
    """Phase-pinpoint print, flushed to stderr so it survives a segfault.
    Gated by RHINO_TRACE so the perf path stays quiet."""
    if os.environ.get("RHINO_TRACE"):
        print(f"[rhino_e2e] {msg}", file=sys.stderr, flush=True)


def _env_truthy(name):
    return rpu_env_bool(name)


def _spm_dump(rb, label):
    """Per-core SPM usage dump, gated by RHINO_SPM (diagnostic only).
    Bound on torch.rpu (v5 debug shim), not the rpu_backend module."""
    if os.environ.get("RHINO_SPM"):
        try:
            torch.rpu.spm_alloc_dump(label)
        except Exception as e:  # noqa: BLE001
            print(f"[rhino_e2e] spm_alloc_dump({label}) failed: {e}", file=sys.stderr, flush=True)


def _limit_expert_mlp_dim_for_perf(expert, cfg, mlp_limit):
    """Experimental performance-only MLP width truncation.

    This changes model math and is only used to test whether an external
    baseline's much lower byte count corresponds to a narrower action MLP.
    """
    mlp_limit = int(mlp_limit)
    original = int(cfg.mlp_dim)
    if mlp_limit <= 0 or mlp_limit > original:
        raise ValueError(
            f"RPU_RHINOVLA_EXPERT_MLP_DIM_LIMIT must be in [1, {original}], "
            f"got {mlp_limit}"
        )
    if mlp_limit == original:
        return False

    for layer in expert.layers:
        gate = layer.mlp.gate_proj
        up = layer.mlp.up_proj
        down = layer.mlp.down_proj

        gate.weight = torch.nn.Parameter(
            gate.weight.detach()[:mlp_limit, :].contiguous())
        up.weight = torch.nn.Parameter(
            up.weight.detach()[:mlp_limit, :].contiguous())
        down.weight = torch.nn.Parameter(
            down.weight.detach()[:, :mlp_limit].contiguous())

        if gate.bias is not None:
            gate.bias = torch.nn.Parameter(
                gate.bias.detach()[:mlp_limit].contiguous())
        if up.bias is not None:
            up.bias = torch.nn.Parameter(
                up.bias.detach()[:mlp_limit].contiguous())

        gate.out_features = mlp_limit
        up.out_features = mlp_limit
        down.in_features = mlp_limit

    cfg.mlp_dim = mlp_limit
    expert.config.mlp_dim = mlp_limit
    return True


def _causal_prefix_attention_mask(attention_mask, prefix_mask, *, fast_replay, padding_rows):
    """Elide visibility for an all-visible logical causal prefix and its tail.

    The lower triangular mask prevents logical rows from seeing right padding.
    Action attention still receives the complete mask including the zero tail.
    """
    if fast_replay and prefix_mask.ndim == 2:
        logical = prefix_mask.shape[-1] - int(padding_rows)
        if (0 < logical <= prefix_mask.shape[-1]
                and bool((prefix_mask[:, :logical] == 1).all())
                and not bool(prefix_mask[:, logical:].any())):
            return None
    return attention_mask


def _pad_prefix_static_cache(cache, execution_len):
    """Cold padding for merger-scatter destinations; logical rows stay separate."""
    from rpu_backend.runtime.decoder import pad_mrope_prefill_inputs

    source = cache["inputs_embeds_base_rpu"]
    logical = int(source.shape[1])
    if execution_len == logical:
        return
    if execution_len < logical:
        raise ValueError("RhinoVLA cached prefix cannot truncate logical tokens")
    ie, mask, pos = pad_mrope_prefill_inputs(
        source, cache["attention_mask_rpu"], cache["position_ids_rpu"], execution_len)
    pad = execution_len - logical
    dense = cache["deepstack_zero_rpu"]
    updates = {
        "inputs_embeds_base_rpu": ie,
        "attention_mask_rpu": mask,
        "position_ids_rpu": pos,
        "deepstack_zero_rpu": torch.cat([dense, dense.new_zeros(pad, dense.shape[-1])]),
        "padded_prefix_mask": torch.cat([
            cache["prefix_mask"], cache["prefix_mask"].new_zeros(1, pad)], dim=-1),
        "logical_len": logical,
    }
    for key in ("rope_cos_il_rpu", "rope_sin_il_rpu"):
        if key in cache:
            value = cache[key]
            updates[key] = torch.cat([value, value[-1:].expand(pad, -1)]).contiguous()
    cache.update(updates)


def _preflight_rhinovla_cold_install(
    *, qin, prefix_len, steps, rpu_execution, model, action_bundle,
    flow_direction, expert_w8a16=None, full_expert_w8a16=None,
):
    """Resolve pipeline-owned host gates before selecting the allocator."""
    from rpu_backend.adapters.rhinovla.runtime import (
        RhinoVLAPipelineColdConfig,
        resolve_rhinovla_execution,
    )
    from rpu_backend.runtime.hw_attrs import validate_preinstall

    denoise_times, denoise_dt, denoise_schedule_words = denoise_schedule(
        steps, flow_direction
    )
    steps = int(steps)
    prefix_len = int(prefix_len)
    skip_vt = rpu_env_bool("RHINO_SKIP_VT")
    smoke_prefix_padding_rows = int(
        os.environ.get("RHINO_SMOKE_PREFIX_PADDING_ROWS", "0")
    )
    if flow_direction == "official_descending":
        from .factory import _reject_unsafe_diagnostics

        _reject_unsafe_diagnostics()

    chunk = int(os.environ.get("RHINO_CHUNK", "0")) if rpu_execution is None else 0
    effective_request = rpu_execution
    if effective_request is None and chunk > 0:
        effective_request = {"prefill": {"chunk_size": chunk}}
    pipeline_cold_config, effective_request = RhinoVLAPipelineColdConfig.resolve(
        effective_request
    )
    resolved = resolve_rhinovla_execution(
        effective_request, entry_point="RhinoVLAOnRPU"
    )

    if vars(model).get("_rpu_swizzle_started", False):
        raise RuntimeError(
            "RhinoVLA model conversion already started; reload the model"
        )
    validate_preinstall(model)
    try:
        iface = model.qwen if hasattr(model, "qwen") else model.qwen_vl_interface
        top = iface.model
        text_model = top.model.language_model
        vision_model = top.model.visual
        text_cfg = top.config.text_config
        vision_cfg = top.config.vision_config
        depth_total = int(text_cfg.num_hidden_layers)
        num_kv_heads = int(text_cfg.num_key_value_heads)
        head_dim = int(text_cfg.head_dim)
        expert, action_io, mask_proj, expert_cfg = action_bundle
        expert_depth = int(expert_cfg.depth)
        expert_mlp_dim = int(expert_cfg.mlp_dim)
        suffix_len = int(expert_cfg.action_horizon) + (
            1 if expert_cfg.use_state_token else 0
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise RPUBackendError(
            "RhinoVLA cold preflight found an incomplete or malformed model/config"
        ) from exc
    if min(depth_total, num_kv_heads, head_dim, expert_depth, expert_mlp_dim) <= 0:
        raise RPUBackendError(
            "RhinoVLA text/expert dimensions must be positive"
        )

    prefill_cold = pipeline_cold_config.stages["prefill"]
    vision_cold = pipeline_cold_config.stages["vision"]
    action_cold = pipeline_cold_config.stages["action"]
    from .fused import _RhinoVLANativeColdConfig

    native_cold = _RhinoVLANativeColdConfig.from_env()
    from .runtime import resolve_rhinovla_quantization
    expert_w8a16, full_expert_w8a16, full_w8a16 = resolve_rhinovla_quantization(
        expert_w8a16=expert_w8a16, full_expert_w8a16=full_expert_w8a16)
    precompute_gate_tanh = rpu_env_bool("RPU_RHINOVLA_PRECOMPUTE_GATE_TANH")
    high_precision = rpu_env_bool("RPU_RHINOVLA_HIGH_PRECISION")
    high_precision_fusions = rpu_env_bool("RPU_RHINOVLA_HIGH_PRECISION_FUSIONS")
    vector_k_norm = rpu_env_bool("RPU_RHINOVLA_VECTOR_K_NORM")
    vector_q_norm = rpu_env_bool("RPU_RHINOVLA_VECTOR_Q_NORM")
    partial_rope = rpu_env_bool("RPU_RHINOVLA_PARTIAL_ROPE")
    if partial_rope and not high_precision:
        raise ValueError("RhinoVLA PARTIAL_ROPE requires HIGH_PRECISION")
    if vector_q_norm and not high_precision:
        raise ValueError("RhinoVLA VECTOR_Q_NORM requires HIGH_PRECISION")
    if vector_k_norm and not (high_precision and native_cold.aligned_kv):
        raise ValueError("RhinoVLA VECTOR_K_NORM requires HIGH_PRECISION and ALIGNED_KV")
    if high_precision_fusions and not high_precision:
        raise ValueError("RhinoVLA HIGH_PRECISION_FUSIONS requires HIGH_PRECISION")
    if high_precision and not (full_w8a16 and native_cold.precompute_adarms
                               and native_cold.adarms_resident):
        raise ValueError("RhinoVLA HIGH_PRECISION requires full W8 and resident AdaRMS tables")
    if precompute_gate_tanh and not (full_w8a16 and native_cold.precompute_adarms
                                    and native_cold.adarms_resident):
        raise ValueError("RhinoVLA PRECOMPUTE_GATE_TANH requires full W8 and resident AdaRMS tables")
    if expert_w8a16 and native_cold.packed_qkv:
        raise ValueError("RhinoVLA EXPERT_W8A16 is incompatible with PACKED_QKV")
    if (native_cold.expert_fusions or native_cold.adarms_resident
            or native_cold.packed_qkv or native_cold.aligned_kv or expert_w8a16):
        from .convert import _direct_action_input

        if not (action_cold["denoise_unroll"] and _direct_action_input(action_io)
                and steps == 10 and expert_depth == 18 and suffix_len == 31
                and expert_mlp_dim == 3072):
            raise ValueError("RhinoVLA expert transfer requires the v3 18-layer, 31-token, ten-step native loop")
    if (native_cold.packed_qkv or native_cold.aligned_kv or expert_w8a16) and (
        getattr(expert_cfg, "width", None),
        getattr(expert_cfg, "num_attention_heads", None),
        getattr(expert_cfg, "num_key_value_heads", None),
        getattr(expert_cfg, "head_dim", None),
    ) != (1024, 16, 8, 128):
        raise ValueError("RhinoVLA PACKED_QKV/ALIGNED_KV/EXPERT_W8A16 requires H1024/Q16/KV8/D128")
    if full_expert_w8a16 and (not native_cold.precompute_adarms or
                              native_cold.skip_adarms_gemv):
        raise ValueError("RhinoVLA full expert W8 requires PRECOMPUTE_ADARMS without SKIP_ADARMS_GEMV")
    if full_w8a16:
        if not (native_cold.precompute_adarms and not native_cold.skip_adarms_gemv
                and vision_cold["rpu_patch_embed"] and vision_cold["rpu_mergers"]
                and vision_cold["fused_merger"] and not vision_cold["rpu_mergers_streaming"]):
            raise ValueError("RhinoVLA FULL_W8A16 requires precomputed AdaRMS and RPU patch/fused resident mergers")
        if tuple(getattr(text_cfg, name, None) for name in (
            "num_hidden_layers", "hidden_size", "intermediate_size",
            "num_attention_heads", "num_key_value_heads", "head_dim",
        )) != (28, 2048, 6144, 16, 8, 128):
            raise ValueError("RhinoVLA FULL_W8A16 requires the complete v3 2B text tower")
        if tuple(getattr(vision_cfg, name, None) for name in (
            "depth", "hidden_size", "intermediate_size", "num_heads",
        )) != (24, 1024, 4096, 16):
            raise ValueError("RhinoVLA FULL_W8A16 requires the complete v3 vision tower")
    if action_cold["denoise_unroll"]:
        from .convert import _direct_action_input
        if _direct_action_input(action_io) and (
            native_cold.precompute_time_proj or native_cold.fold_action_time_in
        ):
            raise ValueError(
                "RhinoVLA v3 direct action input has no action/time MLP; "
                "disable PRECOMPUTE_TIME_PROJ and FOLD_ACTION_TIME_IN"
            )
    prefill_stage = resolved[1].get("prefill", {})
    padding_capacity = max(
        16,
        int(prefill_stage.get("padding_budget", 0)),
        int(prefill_stage.get("padding_rows", 0))
        if isinstance(prefill_stage.get("padding_rows"), int)
        else 0,
    )
    max_seq = int(math.ceil((prefix_len + padding_capacity) / 16) * 16)

    depth_limit_raw = os.environ.get(
        "RPU_RHINOVLA_EXPERT_DEPTH_LIMIT", ""
    ).strip()
    depth_limit = int(depth_limit_raw) if depth_limit_raw else None
    if depth_limit is not None and not 0 < depth_limit <= expert_depth:
        raise ValueError(
            "RPU_RHINOVLA_EXPERT_DEPTH_LIMIT must be in "
            f"[1, {expert_depth}], got {depth_limit}"
        )
    effective_depth = depth_limit if depth_limit is not None else expert_depth
    mlp_limit_raw = os.environ.get(
        "RPU_RHINOVLA_EXPERT_MLP_DIM_LIMIT", ""
    ).strip()
    mlp_limit = int(mlp_limit_raw) if mlp_limit_raw else None
    if mlp_limit is not None and not 0 < mlp_limit <= expert_mlp_dim:
        raise ValueError(
            "RPU_RHINOVLA_EXPERT_MLP_DIM_LIMIT must be in "
            f"[1, {expert_mlp_dim}], got {mlp_limit}"
        )
    prefix_layer_offset = int(
        getattr(model, "prefix_layer_offset", depth_total - effective_depth)
    )
    if expert_w8a16 and (effective_depth != 18 or
                         (mlp_limit is not None and mlp_limit != 3072)):
        raise ValueError("RhinoVLA EXPERT_W8A16 requires all 18 layers and MLP3072")
    if depth_limit is not None and depth_limit != expert_depth:
        prefix_layer_offset = depth_total - effective_depth
    exp_max_seq = int(math.ceil((max_seq + suffix_len + 16) / 16) * 16)
    if action_cold["prefix_alias"]:
        # The expert appends its suffix into the text KV allocation. Reserving
        # only text padding (e.g. P230 -> 256) cannot hold P230 + S31. Keep the
        # prefix execution length unchanged and reserve the expert envelope.
        max_seq = exp_max_seq

    return {
        "denoise_times": denoise_times,
        "denoise_dt": denoise_dt,
        "denoise_schedule_words": denoise_schedule_words,
        "steps": steps,
        "qin": qin,
        "prefix_len": prefix_len,
        "skip_vt": skip_vt,
        "smoke_prefix_padding_rows": smoke_prefix_padding_rows,
        "chunk": chunk,
        "pipeline_cold_config": pipeline_cold_config,
        "rpu_execution": resolved[0],
        "text_execution": resolved[1],
        "vision_execution": resolved[2],
        "action_execution": resolved[3],
        "prefill_cold": prefill_cold,
        "vision_cold": vision_cold,
        "action_cold": action_cold,
        "model": model,
        "iface": iface,
        "top": top,
        "text_model": text_model,
        "vision_model": vision_model,
        "text_cfg": text_cfg,
        "vision_cfg": vision_cfg,
        "depth_total": depth_total,
        "max_seq": max_seq,
        "expert": expert,
        "expert_w8a16": expert_w8a16,
        "full_w8a16": full_w8a16,
        "full_expert_w8a16": full_expert_w8a16,
        "precompute_gate_tanh": precompute_gate_tanh,
        "high_precision": high_precision,
        "high_precision_fusions": high_precision_fusions,
        "vector_k_norm": vector_k_norm,
        "vector_q_norm": vector_q_norm,
        "partial_rope": partial_rope,
        "action_io": action_io,
        "mask_proj": mask_proj,
        "expert_cfg": expert_cfg,
        "expert_depth": expert_depth,
        "expert_mlp_dim": expert_mlp_dim,
        "depth_limit": depth_limit,
        "mlp_limit": mlp_limit,
        "prefix_layer_offset": prefix_layer_offset,
        "suffix_len": suffix_len,
        "exp_max_seq": exp_max_seq,
    }


class RhinoVLAOnRPU:
    def __init__(self, *, qin, prefix_len, steps=5, instance_id=0,
                 rpu_execution=None, model=None, action_bundle=None,
                 flow_direction="legacy_ascending",
                 expert_w8a16=None, full_expert_w8a16=None):
        if model is None or action_bundle is None:
            raise ValueError(
                "RhinoVLAOnRPU requires an explicitly loaded model and "
                "action_bundle; use RhinoVLAPolicy.from_factory with "
                "rpu_backend.adapters.rhinovla.factory.official_runtime_factory, "
                "or supply both arguments"
            )
        claim_acquired = False
        try:
            from rpu_backend.api._execution import _require_execution_process_safe
            from rpu_backend.api.causal_lm import (
                _claim_live_instance,
                _release_live_instance,
            )

            _require_execution_process_safe()
            cold = _preflight_rhinovla_cold_install(
                qin=qin,
                prefix_len=prefix_len,
                steps=steps,
                rpu_execution=rpu_execution,
                model=model,
                action_bundle=action_bundle,
                flow_direction=flow_direction,
                expert_w8a16=expert_w8a16, full_expert_w8a16=full_expert_w8a16,
            )
            _claim_live_instance(self)
            claim_acquired = True
            try:
                # RhinoVLA retains more than a thousand small weights. Freeze
                # the common allocator only after the exclusive owner exists.
                torch.rpu.set_caching_allocator(True)
            except BaseException:
                _release_live_instance(self)
                claim_acquired = False
                raise
            self._initialize(qin=qin, prefix_len=prefix_len, steps=steps,
                instance_id=instance_id, rpu_execution=rpu_execution, model=model,
                action_bundle=action_bundle, flow_direction=flow_direction,
                _cold_preflight=cold)
        except BaseException as error:
            owns_native_children = bool(
                vars(self).get("_rhinovla_retirement_children")
            )
            materialization_started = bool(
                vars(self).get("_rpu_swizzle_started", False)
            ) or owns_native_children
            if owns_native_children:
                try:
                    from .runtime import close_rhinovla_runtime

                    close_rhinovla_runtime(
                        self, _release_claim=not materialization_started
                    )
                except BaseException as cleanup_error:
                    error.add_note(f"RhinoVLA construction cleanup failed: {cleanup_error!r}")
            elif claim_acquired and not materialization_started:
                from rpu_backend.api.causal_lm import _release_live_instance
                _release_live_instance(self)
            raise

    def close(self):
        from .runtime import close_rhinovla_runtime
        close_rhinovla_runtime(self)

    def __del__(self):
        if vars(self).get("_rhinovla_closed"):
            return
        from .runtime import gc_close_rhinovla_runtime
        gc_close_rhinovla_runtime(self)

    def _initialize(self, *, qin, prefix_len, steps=5, instance_id=0,
                    rpu_execution=None, model=None, action_bundle=None,
                    flow_direction="legacy_ascending", _cold_preflight=None):
        if _cold_preflight is None:
            raise RuntimeError("RhinoVLA initialization requires cold preflight")
        cold = _cold_preflight
        self._flow_direction = flow_direction
        self._denoise_times = cold["denoise_times"]
        self._denoise_dt = cold["denoise_dt"]
        self._denoise_schedule_words = cold["denoise_schedule_words"]
        self._bound_denoise_schedule = (cold["steps"], flow_direction)
        self._last_prefix_rope_deltas = None
        self._skip_vt = cold["skip_vt"]
        self._smoke_prefix_padding_rows = cold["smoke_prefix_padding_rows"]
        import rpu_backend  # noqa: F401
        from rpu_backend.api.cache import RPUCache
        from rpu_backend.runtime.weights import convert_linear_weights_inplace
        # RhinoVLA-only vision fork: the RhinoVLA vision-on-RPU + perf hooks live
        # in adapters/rhinovla/vision.py (a self-contained fork), NOT the shared
        # adapters/qwen3_vl/vision.py — so gr00t/other qwen3_vl consumers on main
        # are untouched. Text prefill uses the shared M-RoPE decoder mechanism.
        from rpu_backend.adapters.rhinovla.vision import (
            install_qwen3_vl_vision_for_rpu,
            register_qwen3vl_vision_fused_merger,
        )
        from rpu_backend.adapters.rhinovla import (
            bind_rhinovla_execution_runtime,
            build_rpu_expert,
        )
        from rpu_backend.adapters.rhinovla.runtime import (
            own_rhinovla_child,
            RHINOVLA_TEXT_COMPONENT, RHINOVLA_VISION_COMPONENT, RHINOVLA_ACTION_COMPONENT,
        )
        from rpu_backend.runtime.decoder import install_mrope_text_decoder_for_rpu
        self.steps = cold["steps"]
        self.qin = cold["qin"]
        self.P = cold["prefix_len"]
        self._RPUCache = RPUCache
        self._run_decoder = __import__("rpu_backend.runtime.decoder", fromlist=["_run_causal_decoder_forward"])._run_causal_decoder_forward
        import rpu_backend as _rb
        self._rb = _rb
        self._chunk = cold["chunk"]
        self._pipeline_cold_config = cold["pipeline_cold_config"]
        self._rpu_execution = cold["rpu_execution"]
        self._text_execution = cold["text_execution"]
        self._vision_execution = cold["vision_execution"]
        self._action_execution = cold["action_execution"]
        prefill_cold = cold["prefill_cold"]
        vision_cold = cold["vision_cold"]
        action_cold = cold["action_cold"]
        m = cold["model"]
        iface = cold["iface"]
        # Everything below may mutate model weights or materialize RPU state.
        # A failed owner therefore keeps the process slot until that owner dies,
        # even when graph-first native cleanup succeeds.
        self._rpu_swizzle_started = True
        vars(m)["_rpu_swizzle_started"] = True
        iface.model = iface.model.to(torch.float16)
        self.m = m
        top = cold["top"]
        self.top = top
        self.text_model = cold["text_model"]
        self.vision_model = cold["vision_model"]
        self.text_cfg = cold["text_cfg"]
        self.vis_cfg = cold["vision_cfg"]
        self.depth_total = cold["depth_total"]
        self.max_seq = cold["max_seq"]
        rpu_patch_embed = vision_cold["rpu_patch_embed"]
        rpu_mergers = vision_cold["rpu_mergers"]
        rpu_mergers_streaming = vision_cold["rpu_mergers_streaming"]
        install_qwen3_vl_vision_for_rpu(
            self.vision_model,
            vision_config=self.vis_cfg,
            rpu_patch_embed=rpu_patch_embed,
            rpu_mergers=rpu_mergers,
            rpu_mergers_streaming=rpu_mergers_streaming,
            execution_chunk_size=self._vision_execution.get(
                "vision", {}
            ).get("chunk_size", "auto"),
            w8a16=cold["full_w8a16"],
        )
        own_rhinovla_child(self, self.vision_model, RHINOVLA_VISION_COMPONENT)
        if rpu_patch_embed:
            _trace("vision RPU patch_embed enabled")
        if rpu_mergers:
            merger_mode = "streaming" if rpu_mergers_streaming else "resident"
            _trace(f"vision RPU mergers enabled ({merger_mode})")
        if vision_cold["fast_replay"]:
            torch.ops.rpu.qwen3vl_vision_set_fast_replay(
                self.vision_model._rpu_vision_handle, True)
            _trace("vision fast replay enabled")
        # Skip the vision preload callback's host traversal on REPLAY.
        # Recorded preload nodes still execute through the segment launch.
        if vision_cold["preload_replay_skip"]:
            torch.ops.rpu.qwen3vl_vision_set_preload_replay_skip(
                self.vision_model._rpu_vision_handle, True)
            _trace("vision preload replay-skip enabled")
        # Bake the fused merger into the BUILT graph: skip the ~370-node post_fn
        # host re-emission on REPLAY (the merger nodes replay from segments_ anyway).
        # Bit-identical; requires the fused merger + vision fast-replay to be on.
        if vision_cold["bake_merger"]:
            torch.ops.rpu.qwen3vl_vision_set_bake_merger(
                self.vision_model._rpu_vision_handle, True)
            _trace("vision bake merger enabled")
        # In-graph fused vision merger (folds the 4 merger MLPs into the vision device
        # segment, killing the eager aten::linear in the vision->prefill host bubble).
        # Opt-in via RPU_QWEN3VL_VISION_FUSED_MERGER; registers weights + sets the
        # _rpu_vision_fused_merger flag consumed by the forward.
        if vision_cold["fused_merger"]:
            register_qwen3vl_vision_fused_merger(self.vision_model, w8a16=cold["full_w8a16"])
            _trace("vision fused merger registered")
        if cold["high_precision"]:
            torch.ops.rpu.qwen3vl_vision_set_rhinovla_high_precision(
                self.vision_model._rpu_vision_handle, True)
        # Clone elimination (RPU_RHINOVLA_PREFIX_NO_CLONE): the vision merger output
        # (pop_merged stable slots) + the prefix base buffers don't need a per-predict
        # clone in the RhinoVLA flow — vision runs once per predict and prefix assembly
        # copies the slots out immediately, and the prefix buffers' text/zero positions
        # are static (same prompt).
        self._prefix_no_clone = prefill_cold["prefix_no_clone"]
        if self._prefix_no_clone:
            self.vision_model._rpu_vision_merged_no_clone = True
            _trace("prefix no-clone enabled")
        _trace("vision installed"); _spm_dump(_rb, "after_vision_install")
        self.text_model = self.text_model.to(torch.float16)
        if cold["full_w8a16"]:
            from .precision import quantize_text_prefix
            quantize_text_prefix(self.text_model)
        convert_linear_weights_inplace(self.text_model, skip_names=set())
        self.text_model.to("rpu")
        prefix_scales = None
        if cold["full_w8a16"]:
            from .precision import text_scale_lists
            prefix_scales = text_scale_lists(self.text_model)
        install_mrope_text_decoder_for_rpu(
            self.text_model, text_config=self.text_cfg, max_seq_len=self.max_seq,
            arch="qwen3_vl_text", chunk_envelope_for=_TEXT_ENVELOPE,
            vision_config=self.vis_cfg, deepstack_lang_layers=list(range(3)),
            scale_lists=prefix_scales,
            execution_config={"prefill": {
                key: value for key, value in self._text_execution.get("prefill", {}).items()
                if key in ("chunk_size", "padding_rows", "padding_budget", "linear_acc32")
            }})
        if cold["high_precision"]:
            torch.ops.rpu.causal_decoder_set_rhinovla_high_precision(
                self.text_model._rpu_decoder_handle, True)
        if cold["high_precision_fusions"]:
            torch.ops.rpu.causal_decoder_set_rhinovla_high_precision_fusions(
                self.text_model._rpu_decoder_handle, True)
        own_rhinovla_child(self, self.text_model, RHINOVLA_TEXT_COMPONENT)
        from .precision import bind_pipeline_linear_accumulation
        bind_pipeline_linear_accumulation(
            self.text_model, self.vision_model,
            prefill=prefill_cold["linear_acc32"], vision=vision_cold["linear_acc32"])
        self.prefix_gc = self.text_model._rpu_text_graph_cache
        self._prefill_fast_replay = prefill_cold["fast_replay"]
        if self._prefill_fast_replay:
            torch.ops.rpu.causal_decoder_set_fast_replay(
                self.text_model._rpu_decoder_handle, True)
            _trace("prefill fast replay enabled")
        # Skip the text-decoder preload callback's host traversal on REPLAY.
        # Recorded nodes still reload persistent SPM; skip_op_stream advances the cursor.
        if prefill_cold["preload_replay_skip"]:
            torch.ops.rpu.causal_decoder_set_preload_replay_skip(
                self.text_model._rpu_decoder_handle, True)
            _trace("prefill preload replay-skip enabled")
        _trace("text installed"); _spm_dump(_rb, "after_text_install")
        # Persistent text-prefill KV cache, reused across predict() calls so the
        # prefill graph's baked cache addresses stay valid → REPLAY instead of
        # rebuild. reset_to_position(0) each _prefill re-inserts from scratch.
        self._text_cache = self._RPUCache(
            num_layers=self.depth_total, batch_size=1, max_seq_len=self.max_seq,
            num_kv_heads=int(self.text_cfg.num_key_value_heads),
            head_dim=int(self.text_cfg.head_dim))
        # The fused decoder takes inputs_embeds (input_ids=None) and never uses
        # embed_tokens, but prepare_prefix_inputs_from_vision_outputs looks the
        # token embeddings up per call with CPU input_ids. Keep embed_tokens on
        # CPU (validated path does this lookup before .to('rpu')).
        self.text_model.embed_tokens = self.text_model.embed_tokens.to("cpu")
        self._prefix_prep_cache_enabled = prefill_cold["prefix_prep_cache"]
        self._prefix_prep_rpu_enabled = prefill_cold["prefix_prep_rpu"]
        self._partial_mrope_enabled = prefill_cold["partial_mrope"]
        self._prefix_static_cache = (
            self._build_prefix_static_cache()
            if self._prefix_prep_cache_enabled
            else None
        )
        self.expert = cold["expert"]
        self.action_io = cold["action_io"]
        self.mask_proj = cold["mask_proj"]
        self.cfg = cold["expert_cfg"]
        self._expert_depth_limited = False
        self._expert_mlp_limited = False
        depth_limit = cold["depth_limit"]
        if depth_limit is not None:
            original_depth = cold["expert_depth"]
            if depth_limit != original_depth:
                self.expert.layers = torch.nn.ModuleList(
                    list(self.expert.layers)[:depth_limit]
                )
                self.cfg.depth = depth_limit
                self.expert.config.depth = depth_limit
                self._expert_depth_limited = True
                _trace(f"expert depth limited {original_depth} -> {depth_limit}")
        mlp_limit = cold["mlp_limit"]
        if mlp_limit is not None:
            original_mlp_dim = cold["expert_mlp_dim"]
            self._expert_mlp_limited = _limit_expert_mlp_dim_for_perf(
                self.expert, self.cfg, mlp_limit)
            if self._expert_mlp_limited:
                _trace(f"expert MLP dim limited {original_mlp_dim} -> {int(self.cfg.mlp_dim)}")
        self.prefix_layer_offset = cold["prefix_layer_offset"]
        self._shared_cache = action_cold["shared_cache"]
        # Prefix aliasing passes the selected text-cache layers directly to expert.
        # The expert writes suffix [P:P+S] and reads prefix [0:P], so shared_cache and
        # capacity for both ranges are required.
        self._prefix_alias = action_cold["prefix_alias"]
        if self._shared_cache:
            _trace(f"shared prefix cache enabled offset={self.prefix_layer_offset}")
        self.suffix_len = cold["suffix_len"]
        self.exp_max_seq = cold["exp_max_seq"]
        _trace("building expert")
        self.er = build_rpu_expert(
            self.expert,
            prefix_len=self.P,
            suffix_len=self.suffix_len,
            max_seq_len=self.exp_max_seq,
            w8a16=cold["expert_w8a16"],
            full_w8a16=cold["full_expert_w8a16"],
        )
        if cold["high_precision"]:
            torch.ops.rpu.rhino_vla_set_high_precision(self.er._rpu_handle, True)
        if cold["high_precision_fusions"]:
            torch.ops.rpu.rhino_vla_set_high_precision_fusions(self.er._rpu_handle, True)
        if cold["vector_k_norm"]:
            torch.ops.rpu.rhino_vla_set_vector_k_norm(self.er._rpu_handle, True)
        if cold["vector_q_norm"]:
            torch.ops.rpu.rhino_vla_set_vector_q_norm(self.er._rpu_handle, True)
        if cold["partial_rope"]:
            torch.ops.rpu.rhino_vla_set_partial_rope(self.er._rpu_handle, True)
        own_rhinovla_child(self, self.er, RHINOVLA_ACTION_COMPONENT)
        self._denoise_unroll = action_cold["denoise_unroll"]
        native_cold = self.er._rpu_native_cold_config
        self._precompute_adarms = native_cold.precompute_adarms
        self._precompute_time_proj = native_cold.precompute_time_proj
        self._fold_action_time_in = native_cold.fold_action_time_in
        self._denoise_cond_all_cpu = None
        self._denoise_adarms_tables = None
        self._denoise_time_proj_table = None
        self._denoise_loop_meta = None
        if self._denoise_unroll:
            from rpu_backend.adapters.rhinovla.convert import (
                _prepare_denoise_adarms_tables,
                _prepare_denoise_time_proj_table,
                _prepare_denoise_loop_weights,
            )

            loop_w = _prepare_denoise_loop_weights(
                self.expert, self.action_io, self.mask_proj, full_w8a16=cold["full_w8a16"])
            loop_setter = (torch.ops.rpu.rhino_vla_set_denoise_loop_weights_w8a16
                           if cold["full_w8a16"] else torch.ops.rpu.rhino_vla_set_denoise_loop_weights)
            loop_setter(
                self.er._rpu_handle,
                loop_w["action_in_w"], loop_w["action_in_b"],
                loop_w["action_time_in_w"], loop_w["action_time_in_b"],
                loop_w["time_in_action_w"], loop_w["time_in_time_w"],
                loop_w["time_in_b"],
                loop_w["time_out_w"], loop_w["time_out_b"],
                loop_w["state_w"], loop_w["state_b"],
                loop_w["state_mask_w"], loop_w["state_mask_b"],
                loop_w["action_mask_w"], loop_w["action_mask_b"],
                loop_w["final_norm_w"], loop_w["final_norm_b"],
                loop_w["action_out_w"], loop_w["action_out_b"],
                self.er._rpu_cos_cached, self.er._rpu_sin_cached,
                loop_w["action_dim"], loop_w["action_dim_pad"],
                loop_w["state_dim"], loop_w["state_dim_pad"],
                loop_w["action_horizon"], loop_w["suffix_len"],
                loop_w["direct_action_input"],
                *((loop_w["io_scales"],) if cold["full_w8a16"] else ()),
            )
            # Schedule is cold-bound before any dry plan/SPM allocation, not
            # first discovered during forward after a descriptor was sealed.
            torch.ops.rpu.rhino_vla_prepare_denoise_loop_schedule(
                self.er._rpu_handle, self.steps, self._denoise_dt,
            )
            self._denoise_loop_meta = loop_w
            _trace("denoise in-graph unroll weights bound")
            if self._precompute_adarms:
                action_mod = sys.modules[self.action_io.__class__.__module__]
                times = torch.tensor(self._denoise_times, dtype=torch.float32)
                cond_cpu = action_mod.create_sinusoidal_pos_embedding(
                    times, self.cfg.width).to(dtype=torch.float32).contiguous()
                if cold["full_expert_w8a16"]:
                    pair_table, final_table, weights, scales, biases = _prepare_denoise_adarms_tables(
                        self.er, cond_cpu, full_w8a16=True, loop_weights=loop_w,
                        quantize_final_norm=cold["full_w8a16"], final_norm=self.expert.norm,
                        precompute_gate_tanh=cold["precompute_gate_tanh"],
                        high_precision=cold["high_precision"])
                    torch.ops.rpu.rhino_vla_set_denoise_loop_adarms_tables_w8a16(
                        self.er._rpu_handle, pair_table, final_table, weights, scales, biases,
                        cold["precompute_gate_tanh"], cold["high_precision"])
                else:
                    pair_table, final_table = _prepare_denoise_adarms_tables(
                        self.expert, cond_cpu)
                    torch.ops.rpu.rhino_vla_set_denoise_loop_adarms_tables(
                        self.er._rpu_handle, pair_table, final_table)
                self._denoise_cond_all_cpu = cond_cpu
                self._denoise_adarms_tables = (pair_table, final_table)
                _trace("denoise AdaRMS precompute enabled")
            if self._precompute_time_proj:
                action_mod = sys.modules[self.action_io.__class__.__module__]
                if self._denoise_cond_all_cpu is not None:
                    cond_cpu = self._denoise_cond_all_cpu
                else:
                    times = torch.tensor(self._denoise_times, dtype=torch.float32)
                    cond_cpu = action_mod.create_sinusoidal_pos_embedding(
                        times, self.cfg.width).to(dtype=torch.float32).contiguous()
                time_proj_table = _prepare_denoise_time_proj_table(
                    self.action_io, cond_cpu)
                torch.ops.rpu.rhino_vla_set_denoise_loop_time_proj_table(
                    self.er._rpu_handle, time_proj_table)
                self._denoise_cond_all_cpu = cond_cpu
                self._denoise_time_proj_table = time_proj_table
                _trace("denoise time-proj precompute enabled")
        if action_cold["fast_replay"]:
            torch.ops.rpu.rhino_vla_set_fast_replay(self.er._rpu_handle, True)
            _trace("denoise fast replay enabled")
        self._execution_controller = bind_rhinovla_execution_runtime(
            self,
            text_model=self.text_model,
            vision_model=self.vision_model,
            action_expert=self.er,
            rpu_execution=self._rpu_execution,
        )
        # Option 2 merger-scatter (RPU_RHINOVLA_MERGER_SCATTER): the fused merger writes
        # its output DIRECTLY into persistent prefix buffers in-graph (during the vision
        # forward), at the visual run positions — killing the eager Python scatter copy_.
        # Build the persistent buffers + register them + the runs with the C++ merger NOW
        # (before the first vision forward, so the captured graph bakes their addresses).
        # Requires the fused merger + RPU prefix prep. Bit-identical (on/off action A/B).
        self._prefix_merger_scatter = (
            self._prefix_prep_rpu_enabled
            and getattr(self.vision_model, "_rpu_vision_fused_merger", False)
            and prefill_cold["merger_scatter"]
        )
        cache = self._prefix_static_cache
        if self._prefix_merger_scatter and cache is not None and "inputs_embeds_base_rpu" in cache:
            physical_len, _, _ = self._prefill_execution_plan(int(cache["inputs_embeds_base_rpu"].shape[1]))
            _pad_prefix_static_cache(cache, int(physical_len))
            ie_p = cache["inputs_embeds_base_rpu"].clone()
            cache["inputs_embeds_persist"] = ie_p
            n_ds = len(self.vision_model.deepstack_merger_list)
            dense_p = [cache["deepstack_zero_rpu"].clone() for _ in range(n_ds)]
            cache["deepstack_dense_persist"] = dense_p
            runs = cache["visual_runs"]
            # MR-C / T31 (B3): pass the TENSORS, not int(t.data_ptr()). The C++
            # side used to reinterpret_cast those integers while holding no
            # reference, so the two cache[...] assignments above were doing DOUBLE
            # duty: they are read back at :508/:511 as the actual prefix buffers,
            # AND they were the only thing keeping the scatter destinations alive
            # — by convention, with nothing checking it. The model retains the
            # tensors itself now, so only the first duty is left; the lifetime of
            # the DMA destination is a fact rather than an agreement between two
            # files.
            torch.ops.rpu.qwen3vl_vision_set_prefix_scatter(
                self.vision_model._rpu_vision_handle,
                ie_p,
                dense_p,
                [int(s) for s, _ in runs],
                [int(l) for _, l in runs],
            )
            _trace(f"prefix merger-scatter enabled: {len(runs)} runs, {n_ds} deepstack")
        else:
            self._prefix_merger_scatter = False
        _trace("expert built (__init__ done)"); _spm_dump(_rb, "after_expert_build")
        self.last_phase_ms = {}

    def _vision(self):
        with torch.no_grad():
            return self.vision_model(self.qin["pixel_values"],
                                     grid_thw=self.qin["image_grid_thw"], return_dict=True)

    def _build_prefix_static_cache(self):
        top = self.top
        outer = top.model
        input_ids = self.qin["input_ids"]
        attention_mask = self.qin["attention_mask"]
        image_grid_thw = self.qin["image_grid_thw"]
        visual_token_id = top.config.image_token_id
        visual_pos_mask = (input_ids == visual_token_id).squeeze(0).cpu().bool()
        mm_token_type_ids = self.qin.get(
            "mm_token_type_ids", (input_ids == visual_token_id).long()
        )
        attention_mask_cpu = attention_mask.detach().cpu().contiguous()
        with torch.no_grad():
            base_inputs_embeds = (
                self.text_model.get_input_embeddings()(input_ids)
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
            )
            position_ids, rope_deltas = outer.get_rope_index(
                input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                attention_mask=attention_mask,
            )
        seq_len = int(input_ids.shape[1])
        hidden_size = int(base_inputs_embeds.shape[-1])
        pos_cpu = position_ids.detach().cpu().contiguous()
        if pos_cpu.ndim == 3 and pos_cpu.shape[0] == 3 and pos_cpu.shape[1] == 1:
            pos_seq3 = pos_cpu.squeeze(1).transpose(0, 1).contiguous()
        else:
            pos_seq3 = pos_cpu
        cache = {
            "semantic_inputs": {
                key: self.qin[key].detach().cpu().clone() if key in self.qin else None
                for key in ("input_ids", "attention_mask", "image_grid_thw", "mm_token_type_ids")
            },
            "inputs_embeds_base": base_inputs_embeds,
            "position_ids": pos_cpu,
            "attention_mask": attention_mask_cpu,
            "prefix_mask": attention_mask_cpu.to(torch.bool),
            "prefix_rope_deltas": rope_deltas.detach().cpu().clone(),
            "visual_pos_mask": visual_pos_mask,
            "deepstack_zero": torch.zeros((seq_len, hidden_size), dtype=torch.float32),
            "prefix_len": int(attention_mask.sum()),
        }
        if self._partial_mrope_enabled:
            from rpu_backend.runtime.rope_partial import build_interleaved_mrope_cos_sin

            rope_params = getattr(self.text_cfg, "rope_parameters", None)
            if rope_params is None:
                rope_params = getattr(self.text_cfg, "rope_scaling", None) or {}
            mrope_section = [int(x) for x in rope_params["mrope_section"]]
            rope_theta = rope_params.get("rope_theta", None)
            if rope_theta is None:
                rope_theta = getattr(self.text_cfg, "rope_theta", 10000.0)
            head_dim = getattr(self.text_cfg, "head_dim", None)
            if head_dim is None:
                head_dim = int(self.text_cfg.hidden_size) // int(self.text_cfg.num_attention_heads)
            cos_il, sin_il = build_interleaved_mrope_cos_sin(
                pos_seq3.to(torch.int64),
                head_dim=int(head_dim),
                rope_theta=float(rope_theta),
                mrope_section=mrope_section,
            )
            cache["rope_cos_il"] = cos_il
            cache["rope_sin_il"] = sin_il
        if self._prefix_prep_rpu_enabled:
            cache.update(
                {
                    "inputs_embeds_base_rpu": base_inputs_embeds.to(
                        device="rpu", dtype=torch.float16
                    ).contiguous(),
                    "position_ids_rpu": pos_seq3.to(
                        device="rpu", dtype=torch.int32
                    ).contiguous(),
                    "attention_mask_rpu": attention_mask_cpu.to("rpu").contiguous(),
                    "visual_pos_mask_rpu": visual_pos_mask.to("rpu"),
                    "deepstack_zero_rpu": torch.zeros(
                        (seq_len, hidden_size),
                        dtype=torch.float16,
                        device="rpu",
                    ).contiguous(),
                }
            )
            # Contiguous runs of visual-token positions (one per camera view, 64
            # tokens each). index_put has no RPU kernel -> CPU fallback + RPU<->CPU
            # round-trips; these runs let prefix assembly write the merger output
            # with on-device slice copy_ instead. Computed once (mask is static).
            vp_idx = visual_pos_mask.nonzero(as_tuple=True)[0].tolist()
            runs = []
            if vp_idx:
                rs = rp = vp_idx[0]
                for j in vp_idx[1:]:
                    if j == rp + 1:
                        rp = j
                    else:
                        runs.append((rs, rp - rs + 1)); rs = rp = j
                runs.append((rs, rp - rs + 1))
            cache["visual_runs"] = runs
            if self._partial_mrope_enabled:
                cache["rope_cos_il_rpu"] = cache["rope_cos_il"].to("rpu").contiguous()
                cache["rope_sin_il_rpu"] = cache["rope_sin_il"].to("rpu").contiguous()
        return cache

    def _prepare_prefix_inputs(self, vision_output):
        cache = self._prefix_static_cache
        if cache is None:
            return prepare_prefix_inputs_from_vision_outputs(self.m, self.qin, vision_output)
        for key, original in cache["semantic_inputs"].items():
            present = key in self.qin
            if present != (original is not None) or (
                present and not torch.equal(original, self.qin[key].detach().cpu())
            ):
                raise ValueError(
                    f"RhinoVLA cached prefix {key} changed; construct a new runtime "
                    "for a different token/mask/grid input"
                )

        use_rpu_prep = (
            self._prefix_prep_rpu_enabled
            and vision_output.pooler_output.device.type == "rpu"
            and "inputs_embeds_base_rpu" in cache
        )
        if use_rpu_prep:
            runs = cache["visual_runs"]
            if self._prefix_merger_scatter:
                # Option 2: the fused merger already wrote inputs_embeds_persist +
                # dense_persist directly in-graph during the vision forward (at the
                # visual run positions) — no Python scatter; just return the buffers.
                return {
                    "inputs_embeds": cache["inputs_embeds_persist"],
                    "position_ids": cache["position_ids_rpu"],
                    "attention_mask": cache["attention_mask_rpu"],
                    "deepstack_dense": cache["deepstack_dense_persist"],
                    "prefix_len": cache["prefix_len"],
                    "prefix_mask": cache.get("padded_prefix_mask", cache["prefix_mask"]),
                    "logical_len": cache.get("logical_len", int(cache["inputs_embeds_persist"].shape[1])),
                    "prefix_rope_deltas": cache["prefix_rope_deltas"],
                    "visual_pos_mask": cache["visual_pos_mask"].unsqueeze(0),
                    "rope_cos_il": cache.get("rope_cos_il_rpu"),
                    "rope_sin_il": cache.get("rope_sin_il_rpu"),
                }
            # Clone elimination: reuse a persistent prefix buffer across predicts. The
            # text positions are static (same prompt); only the visual runs change, so
            # init once from the base and re-scatter only the visual runs — no
            # per-predict clone. Falls back to a fresh clone when off.
            if self._prefix_no_clone:
                inputs_embeds = cache.get("inputs_embeds_persist")
                if inputs_embeds is None:
                    with torch.inference_mode(False), torch.no_grad():
                        inputs_embeds = cache["inputs_embeds_base_rpu"].clone()
                    cache["inputs_embeds_persist"] = inputs_embeds
            else:
                inputs_embeds = cache["inputs_embeds_base_rpu"].clone()
            pooler = vision_output.pooler_output.detach().to(
                device="rpu", dtype=torch.float16
            )
            # On-device slice copy_ per contiguous visual run (replaces index_put,
            # which has no RPU kernel and falls back to CPU). pooler rows are in
            # view order = sequence order of the runs.
            off = 0
            for start, length in runs:
                inputs_embeds[0, start:start + length, :].copy_(pooler[off:off + length])
                off += length

            deepstack_dense = []
            dense_persist = cache.get("deepstack_dense_persist") if self._prefix_no_clone else None
            if self._prefix_no_clone and dense_persist is None:
                dense_persist = []
                cache["deepstack_dense_persist"] = dense_persist
            for di, feature in enumerate(vision_output.deepstack_features):
                if self._prefix_no_clone:
                    # Persistent dense buffer per deepstack layer: non-visual positions
                    # stay zero (init), only the visual runs are re-scattered each predict.
                    if di >= len(dense_persist):
                        with torch.inference_mode(False), torch.no_grad():
                            dense_persist.append(cache["deepstack_zero_rpu"].clone())
                    dense = dense_persist[di]
                else:
                    dense = cache["deepstack_zero_rpu"].clone()
                feat = feature.detach().to(device="rpu", dtype=torch.float16)
                off = 0
                for start, length in runs:
                    dense[start:start + length, :].copy_(feat[off:off + length])
                    off += length
                deepstack_dense.append(dense if dense.is_contiguous() else dense.contiguous())

            return {
                "inputs_embeds": inputs_embeds,
                "position_ids": cache["position_ids_rpu"],
                "attention_mask": cache["attention_mask_rpu"],
                "deepstack_dense": deepstack_dense,
                "prefix_len": cache["prefix_len"],
                "prefix_mask": cache["prefix_mask"],
                "prefix_rope_deltas": cache["prefix_rope_deltas"],
                "visual_pos_mask": cache["visual_pos_mask"].unsqueeze(0),
                "rope_cos_il": cache.get("rope_cos_il_rpu"),
                "rope_sin_il": cache.get("rope_sin_il_rpu"),
            }

        visual_pos = cache["visual_pos_mask"]
        inputs_embeds = cache["inputs_embeds_base"].clone()
        pooler = vision_output.pooler_output.detach().to(
            device="cpu", dtype=inputs_embeds.dtype
        )
        inputs_embeds[0, visual_pos, :] = pooler

        deepstack_dense = []
        for feature in vision_output.deepstack_features:
            dense = cache["deepstack_zero"].clone()
            dense[visual_pos, :] = feature.detach().to(device="cpu", dtype=torch.float32)
            deepstack_dense.append(dense.contiguous())

        return {
            "inputs_embeds": inputs_embeds,
            "position_ids": cache["position_ids"],
            "attention_mask": cache["attention_mask"],
            "deepstack_dense": deepstack_dense,
            "prefix_len": cache["prefix_len"],
            "prefix_mask": cache["prefix_mask"],
            "prefix_rope_deltas": cache["prefix_rope_deltas"],
            "visual_pos_mask": visual_pos.unsqueeze(0),
            "rope_cos_il": cache.get("rope_cos_il"),
            "rope_sin_il": cache.get("rope_sin_il"),
        }

    def _prefill_execution_plan(self, logical_len):
        """Inspect the same native child domain used by the official prefill.

        This cold query does not reset KV position, move inputs, or execute a
        Graph. Cost preparation and forward share its original AUTO/EXACT
        request, physical padding policy, and native owner binding.
        """
        from rpu_backend.runtime.decoder import plan_bounded_prefill_execution
        from rpu_backend.runtime.execution_planner import GRAPH_COMPOSITE_CHILD

        logical_len = int(logical_len)
        stage = self._rhinovla_execution_components[
            "language_model"
        ].get("prefill", {})
        padding_rows = stage.get("padding_rows", "auto")
        padding_budget = int(stage.get("padding_budget", 0))
        exact_chunk = stage.get("chunk_size")
        generation = int(
            self._rhinovla_execution_component_generations["language_model"]
        )
        plan_box = {}
        execution_len, chunk_size = plan_bounded_prefill_execution(
            logical_len,
            int(self._text_cache.max_seq_len),
            padding_budget,
            execution_owner=self.text_model,
            execution_native=("causal_decoder", int(self.text_model._rpu_decoder_handle)),
            plan_signature=(True, 0, int(self._partial_mrope_enabled), 4, logical_len),
            graph_cache=self.prefix_gc,
            position=0,
            alignment=1,
            padding_rows=padding_rows,
            exact_chunk_size=(exact_chunk if isinstance(exact_chunk, int) else None),
            resolve_stage_domain=lambda length: (
                torch.ops.rpu.causal_decoder_resolve_prefill_stage_domain(
                    self.text_model._rpu_decoder_handle,
                    int(length), 0, True, 0, int(self._partial_mrope_enabled), 4, logical_len,
                )
            ),
            request_id="rhinovla:language_model:prefill",
            plan_result_sink=lambda result: plan_box.__setitem__("result", result),
            graph_mode=GRAPH_COMPOSITE_CHILD,
            queue_owner_id=int(self.text_model._rpu_decoder_handle),
            physical_metadata=(
                ("component:language_model", 1),
                ("execution_generation", generation),
            ),
        )
        prefill_plan = plan_box["result"]
        return execution_len, chunk_size, prefill_plan

    def _prefill(self, vision_output):
        from rpu_backend.runtime.decoder import (
            pad_mrope_prefill_inputs as _pad_causal_prefill_inputs,
        )
        prep = self._prepare_prefix_inputs(vision_output)
        self._last_prefix_rope_deltas = (
            prep["prefix_rope_deltas"]
            if self._flow_direction == "official_descending" else None
        )
        cache = self._text_cache
        cache.reset_to_position(0)
        ie = prep["inputs_embeds"].to(device="rpu", dtype=torch.float16).contiguous()
        pos = prep["position_ids"].to("rpu")
        attn = prep["attention_mask"].to("rpu")
        ds = [t.to(device="rpu", dtype=torch.float16).contiguous() for t in prep["deepstack_dense"]]
        rope_cos_il = prep.get("rope_cos_il")
        rope_sin_il = prep.get("rope_sin_il")
        if rope_cos_il is not None:
            rope_cos_il = rope_cos_il.to(device="rpu", dtype=torch.float16).contiguous()
            rope_sin_il = rope_sin_il.to(device="rpu", dtype=torch.float16).contiguous()
        logical_len = int(prep.get("logical_len", ie.shape[1]))
        execution_len, chunk_size, prefill_plan = self._prefill_execution_plan(logical_len)
        generation = int(self._rhinovla_execution_component_generations["language_model"])
        pad = int(execution_len) - logical_len
        if ie.shape[1] != execution_len and ie.shape[1] != logical_len:
            # A permitted later plan can change padding. Keep the cold scatter
            # owner and rebuild only this text call's physical tail as needed.
            ie, attn = ie[:, :logical_len], attn[:, :logical_len]
            pos = pos[..., :logical_len] if pos.ndim == 3 else pos[:logical_len]
            ds = [item[:logical_len] for item in ds]
            if rope_cos_il is not None:
                rope_cos_il, rope_sin_il = rope_cos_il[:logical_len], rope_sin_il[:logical_len]
            prep["prefix_mask"] = prep["prefix_mask"][:, :logical_len]
        if pad and ie.shape[1] != execution_len:
            # The shared Qwen3-VL helper handles both [3,B,P] and the cached
            # RPU-preparation layout [P,3]; only the token axis is padded.
            ie, attn, pos = _pad_causal_prefill_inputs(ie, attn, pos, execution_len)
            ds = [
                torch.cat([item, item.new_zeros(pad, item.shape[-1])], dim=0)
                .contiguous()
                for item in ds
            ]
            if rope_cos_il is not None:
                rope_cos_il = torch.cat(
                    [rope_cos_il, rope_cos_il[-1:].expand(pad, -1)], dim=0
                ).contiguous()
                rope_sin_il = torch.cat(
                    [rope_sin_il, rope_sin_il[-1:].expand(pad, -1)], dim=0
                ).contiguous()
            prep["prefix_mask"] = torch.cat(
                [
                    prep["prefix_mask"],
                    prep["prefix_mask"].new_zeros(
                        prep["prefix_mask"].shape[0], pad
                    ),
                ],
                dim=-1,
            ).contiguous()
        attn = _causal_prefix_attention_mask(
            attn, prep["prefix_mask"],
            fast_replay=getattr(self, "_prefill_fast_replay", False),
            padding_rows=pad,
        )
        sig = self._rb.graph.GraphSignature(op_id="rhino_vla_prefix", shapes=[logical_len, execution_len, ie.shape[-1]],
                                      dyn_dims=[self.depth_total, int(cache.position),
                                                int(self._partial_mrope_enabled),
                                                *prefill_plan.graph_key_words()],
                                      dtypes=[torch.float16])
        with self.prefix_gc.capture(sig):
            self._run_decoder(self.text_model, self.text_model._rpu_decoder_handle,
                              input_ids=None, inputs_embeds=ie, attention_mask=attn,
                              position_ids=pos, past_key_values=cache, use_cache=True,
                              return_dict=True, deepstack_dense_visual_embeds=ds,
                              rope_cos_il=rope_cos_il, rope_sin_il=rope_sin_il,
                              prefill_plan=(execution_len, chunk_size, prefill_plan))

        native_text_receipt = getattr(
            self.text_model, "_rpu_last_execution_plan", None
        )
        resolved_chunk = int(prefill_plan.selected.stage_tuple.compute_chunk)
        if (
            not isinstance(native_text_receipt, dict)
            or native_text_receipt.get("logical_len") != logical_len
            or native_text_receipt.get("execution_len") != execution_len
            or native_text_receipt.get("position") != 0
            or native_text_receipt.get("chunk_size") != resolved_chunk
            or native_text_receipt.get("physical_descriptor")
            != prefill_plan.selected.stage_tuple.physical_descriptor
        ):
            raise RuntimeError(
                "RhinoVLA text forward did not retain its dispatched physical descriptor"
            )
        # The shared runner advances by its visible input rows. Preserve the
        # physical padded prefix because the action child consumes those cache
        # rows under an explicit mask.
        cache.reset_to_position(int(execution_len))
        receipt = prefill_plan.as_dict(include_candidates=False)
        receipt.update({
            "stage": "prefill",
            "component": "language_model",
            "generation": generation,
            "logical_len": logical_len,
            "execution_len": int(execution_len),
            "chunk_size": resolved_chunk,
            "padding_rows": pad,
            "position": 0,
            "authority": "NATIVE_A6_STAGE_DESCRIPTOR",
            "dry_forward_agreement": True,
        })
        vars(self.text_model)["_rpu_last_execution_plan"] = receipt
        if self._shared_cache:
            return None, prep["prefix_mask"]
        return self._export_text_prefix_kv_cpu(), prep["prefix_mask"]

    def _export_text_prefix_kv_cpu(self):
        dyn = self._text_cache.to_dynamic_cache("cpu")
        physical_prefix_len = int(self._text_cache.position)
        kv = []
        for i in range(self.prefix_layer_offset, self.prefix_layer_offset + int(self.cfg.depth)):
            k, v = get_cache_layer(dyn, i)
            kv.append((
                k[:, :, :physical_prefix_len, :].float().contiguous(),
                v[:, :, :physical_prefix_len, :].float().contiguous(),
            ))
        self._last_prefix_kv = kv
        return kv

    def export_last_prefix_kv_cpu(self):
        return self._export_text_prefix_kv_cpu()

    def _copy_shared_prefix_to_expert_cache(self):
        from rpu_backend.adapters.rhinovla.convert import copy_prefix_cache_from_text_cache

        copy_prefix_cache_from_text_cache(
            self.er._rpu_kv_cache,
            self._text_cache,
            source_layer_offset=self.prefix_layer_offset,
            num_layers=int(self.cfg.depth),
            prefix_len=int(self._text_cache.position),
        )

    def _denoise(self, prefix_kv, prefix_mask, state, x0, state_mask, action_mask):
        if (self.steps, self._flow_direction) != self._bound_denoise_schedule:
            raise ValueError("RhinoVLA cold denoise schedule changed; construct a new runtime")
        # Idempotency: old path resets to empty so the prefix-prefill guard
        # re-inserts; Phase2 shared-cache path resets then copies text-cache
        # prefix bytes directly and leaves cache.position == P.
        if self._shared_cache and prefix_kv is None:
            if not self._prefix_alias:
                self._copy_shared_prefix_to_expert_cache()
            # A3 (prefix alias): no copy — the denoise op reads the text-cache prefix
            # layers directly (text_cache is left at position P by _prefill).
        else:
            self.er._rpu_kv_cache.reset_to_position(0)
        if self._denoise_unroll:
            return self._denoise_loop(prefix_kv, prefix_mask, state, x0,
                                      state_mask, action_mask)
        x, _ = rpu_denoise(self.er, self.action_io, self.mask_proj,
                             prefix_key_values=prefix_kv, prefix_mask=prefix_mask,
                             state=state, state_mask=state_mask, action_mask=action_mask,
                             x0=x0, steps=self.steps, return_velocities=True,
                             flow_direction=self._flow_direction,
                             prefix_rope_deltas=self._last_prefix_rope_deltas)
        return x

    def _denoise_loop(self, prefix_kv, prefix_mask, state, x0, state_mask, action_mask):
        import sys as _sys
        import torch.nn.functional as Fnn
        from rpu_backend.adapters.rhinovla.convert import (
            _plan_rhino_prefix_kv_execution,
            _prefill_kv_cache,
        )
        from .fused import _rhino_action_math
        make_suffix_attn_mask, bool_mask_to_attention_bias, get_cache_layer = (
            _rhino_action_math(self.er)
        )

        prefix_plan = None
        prefix_route_arguments = ()
        prefix_route_argument_digest = ""
        if prefix_kv is not None:
            prefix_pairs = [
                get_cache_layer(prefix_kv, layer_idx)
                for layer_idx in range(int(self.cfg.depth))
            ]
            (
                prefix_plan,
                prefix_route_arguments,
                prefix_route_argument_digest,
            ) = _plan_rhino_prefix_kv_execution(
                prefix_pairs,
                component=getattr(
                    self.er, "_fmb_execution_component_id", "action_expert"
                ),
                generation=int(getattr(
                    self.er, "_fmb_execution_generation", 0
                )),
                queue_owner_id=int(self.er._rpu_handle),
            )
            _prefill_kv_cache(
                self.er._rpu_kv_cache,
                prefix_pairs,
                prefix_route_arguments,
            )

        meta = self._denoise_loop_meta
        AD = int(meta["action_dim"])
        ADP = int(meta["action_dim_pad"])
        SD = int(meta["state_dim"])
        SDP = int(meta["state_dim_pad"])
        AH = int(meta["action_horizon"])
        S = int(meta["suffix_len"])
        # A3: when aliasing, the prefix lives in the text cache; the expert
        # cache is unused. Otherwise use the expert cache filled just above.
        # Physical KV rows and logical RoPE rows intentionally differ for a
        # right-padded prefix.
        _pos_cache = self._text_cache if self._prefix_alias else self.er._rpu_kv_cache
        physical_prefix_len = int(_pos_cache.position)
        from rpu_backend.adapters.rhinovla.fused import (
            plan_rhino_action_execution,
            publish_rhino_action_execution_receipt,
            _rhino_suffix_rope_start,
            _aligned_rhino_action_prefix,
        )
        rope_position = _rhino_suffix_rope_start(
            runtime_prefix_len=physical_prefix_len,
            batch_size=int(prefix_mask.shape[0]),
            prefix_mask=prefix_mask,
            prefix_rope_deltas=self._last_prefix_rope_deltas,
        )
        if self.er._rpu_native_cold_config.aligned_kv:
            physical_prefix_len, prefix_mask = _aligned_rhino_action_prefix(
                prefix_mask, prefix_len=physical_prefix_len, suffix_len=S,
                capacity=int(_pos_cache.max_seq_len),
            )
        self._last_action_prefix_mask = prefix_mask
        if rope_position + S > int(self.er._rpu_rope_max_seq_len):
            raise RuntimeError(
                "RhinoVLA denoise unroll logical prefix + suffix exceeds "
                f"the installed RoPE table: {rope_position}+{S}>"
                f"{int(self.er._rpu_rope_max_seq_len)}")

        action_mod = _sys.modules[self.action_io.__class__.__module__]
        if self._denoise_cond_all_cpu is not None:
            cond_cpu = self._denoise_cond_all_cpu
        else:
            times = torch.tensor(self._denoise_times, dtype=torch.float32)
            cond_cpu = action_mod.create_sinusoidal_pos_embedding(
                times, self.cfg.width).to(dtype=torch.float32).contiguous()
        cond_all = cond_cpu.to(dtype=torch.float16, device="rpu").contiguous()

        # Rebuild masks per predict: identity-only caching cannot detect in-place
        # updates and may collide when Python object ids are reused.
        suffix_mask = torch.ones((1, S), dtype=torch.bool)
        bool_mask = make_suffix_attn_mask(prefix_mask, suffix_mask)
        attention_mask = bool_mask_to_attention_bias(
            bool_mask, torch.float16
        )
        # Match the per-step fused path: C++ reads v16-aligned K length, while
        # padded dummy lanes stay invisible through the attention bias.
        pad_k = (-attention_mask.shape[-1]) % 16
        if pad_k:
            attention_mask = Fnn.pad(
                attention_mask,
                (0, pad_k),
                value=torch.finfo(attention_mask.dtype).min,
            )
        attention_mask = attention_mask.cpu().contiguous()
        _am = action_mask if action_mask is not None else torch.ones(
            (1, AH, AD), dtype=torch.float32
        )
        if _am.ndim == 2:
            _am = _am[:, None, :]
        if _am.shape == (1, 1, AD):
            _am = _am.expand(1, AH, AD)
        if _am.shape != (1, AH, AD):
            raise ValueError("RhinoVLA action_mask must match [1, horizon, action_dim]")
        action_mask_pad = Fnn.pad(_am, (0, ADP - AD)).to(
            dtype=torch.float16, device="rpu"
        ).contiguous()
        _sm = state_mask if state_mask is not None else torch.ones(
            (state.shape[0], SD), dtype=torch.float32
        )
        _sm = _sm[:, 0] if _sm.ndim == 3 else _sm
        state_mask_pad = Fnn.pad(_sm, (0, SDP - SD)).to(
            dtype=torch.float16, device="rpu"
        ).contiguous()

        # Live per-predict: x0 (noise) + state (proprioception).
        if self._flow_direction == "official_descending":
            x0 = x0 * _am
        x0_pad = Fnn.pad(x0, (0, ADP - AD)).to(
            dtype=torch.float16, device="rpu").contiguous()
        state_2d = state[:, 0] if state.ndim == 3 else state
        if self._flow_direction == "official_descending":
            state_2d = state_2d * _sm
        state_pad = Fnn.pad(state_2d, (0, SDP - SD)).to(
            dtype=torch.float16, device="rpu").contiguous()

        # A3 (prefix alias): pass the text-cache prefix layers as the expert's KV
        # storage — zero copy. Stable DDR addresses → the captured graph bakes them
        # once; the expert writes the suffix at [P:P+S] and reads [0:P+S]. Otherwise
        # use the (copied) expert cache.
        depth = int(self.cfg.depth)
        if self._prefix_alias:
            ace_k = [self._text_cache.k_caches[self.prefix_layer_offset + i] for i in range(depth)]
            ace_v = [self._text_cache.v_caches[self.prefix_layer_offset + i] for i in range(depth)]
        else:
            ace_k = self.er._rpu_kv_cache.k_caches
            ace_v = self.er._rpu_kv_cache.v_caches

        x_out = torch.empty_like(x0_pad)
        torch.ops.rpu.rhino_vla_set_rope_position(
            self.er._rpu_handle, rope_position)
        action_plan = plan_rhino_action_execution(
            self.er,
            logical_len=S,
            execution_len=S,
            prefix_len=physical_prefix_len,
            kv_len=int(attention_mask.shape[-1]),
            use_attention_mask=True,
            is_causal=False,
            prefix_plan_key_words=(
                prefix_plan.graph_key_words()
                if prefix_plan is not None else ()
            ),
        )
        selected_action_plan = action_plan.selected
        if selected_action_plan is None:
            raise RuntimeError("RhinoVLA action planner selected no plan")
        sig = self._rb.graph.GraphSignature(
            op_id="rhino_vla_denoise_loop",
            shapes=[AH, ADP, self.cfg.width],
            dyn_dims=[
                int(self.cfg.depth), physical_prefix_len, rope_position,
                self.steps, S,
                *self._denoise_schedule_words,
                1 if self._precompute_adarms else 0,
                1 if self._precompute_time_proj else 0,
                1 if self._fold_action_time_in else 0,
                1 if self._prefix_alias else 0,
                *action_plan.graph_key_words(),
            ],
            dtypes=[torch.float16],
        )
        with self.er._rpu_graph_cache.capture(sig):
            out_pad = torch.ops.rpu.rhino_vla_denoise_loop_forward(
                self.er._rpu_handle,
                x0_pad,
                ace_k,
                ace_v,
                cond_all,
                attention_mask,
                state_pad,
                state_mask_pad,
                action_mask_pad,
                x_out,
                self._denoise_dt,
                physical_prefix_len,
                self.steps,
                selected_action_plan.stage_tuple.physical_descriptor,
            )
        publish_rhino_action_execution_receipt(
            self.er,
            action_plan,
            logical_len=S,
            prefix_plan=prefix_plan,
            prefix_route_argument_digest=prefix_route_argument_digest,
        )
        return out_pad[:, :, :AD].to("cpu", torch.float32)

    @execution_serialized
    @torch.no_grad()
    def predict(self, *, x0, state, state_mask, action_mask):
        # Isolation knob: skip vision+text FORWARDS, feed a synthetic prefix,
        # run expert-only. Models stay resident. If this OOMs, the co-resident
        # setup exhausts SDK SPM; if it runs, the vision/text forwards do.
        if self._skip_vt:
            from .action import build_rhino_inputs
            _trace("phase: SKIP_VT — synthetic prefix, expert-only")
            skv, smask = build_rhino_inputs(self.cfg, prefix_len=self.P)[:2]
            self._last_prefix_rope_deltas = None
            self._last_prefix_kv, self._last_prefix_mask = skv, smask
            t = time.perf_counter()
            act = self._denoise(skv, smask, state, x0, state_mask, action_mask)
            t_d = (time.perf_counter() - t) * 1000.0
            _trace("phase: denoise done (SKIP_VT)"); _spm_dump(self._rb, "after_denoise_skipvt")
            self.last_phase_ms = {"vision_ms": 0.0, "prefill_ms": 0.0, "denoise_ms": t_d}
            return act.detach().float().cpu()
        _trace("phase: vision start")
        t = time.perf_counter(); vo = self._vision(); t_v = (time.perf_counter() - t) * 1000.0
        _trace("phase: vision done"); _spm_dump(self._rb, "after_vision_fwd")
        t = time.perf_counter(); kv, pmask = self._prefill(vo); t_p = (time.perf_counter() - t) * 1000.0
        # Hardware certification hook for the logical-RoPE/physical-KV split.
        # The text cache keeps the planner's physical rows; the expert masks
        # any bounded trailing padding and starts suffix RoPE at sum(pmask).
        _prefix_padding_rows = self._smoke_prefix_padding_rows
        if _prefix_padding_rows:
            if not 0 < _prefix_padding_rows < int(pmask.shape[-1]):
                raise ValueError(
                    "RHINO_SMOKE_PREFIX_PADDING_ROWS must be in "
                    f"[1, {int(pmask.shape[-1]) - 1}], got "
                    f"{_prefix_padding_rows}")
            pmask = pmask.clone()
            pmask[:, -_prefix_padding_rows:] = False
        # Retain prefix KV for an optional CPU export after inference.
        # The shared-cache execution path does not require that export.
        self._last_prefix_kv, self._last_prefix_mask = kv, pmask
        _trace("phase: prefill done"); _spm_dump(self._rb, "after_prefill_fwd")
        # Subsystem boundary: free the text prefill's repo-arena temporary. The
        # vision/prefill graph caches are NO LONGER cleared — now that the prefix
        # insert stages through the repo arena (not SDK LocalSPM), the per-call
        # insert no longer competes with the cached graphs' SDK SPM pools, so
        # vision + prefill can stay resident and REPLAY across predict() calls
        # (vision input self.qin and the text cache are both stable). The run's
        # idempotent_cos guards correctness; if the 3 graphs' SDK SDPA pools
        # don't co-fit in 8MB, the expert build OOMs and we re-add the clear.
        torch.ops.rpu.spm_alloc_reset_temporary()
        _trace("phase: reset_temporary (graph caches kept for replay)")
        _spm_dump(self._rb, "after_prefill_reset")
        t = time.perf_counter(); act = self._denoise(kv, pmask, state, x0, state_mask, action_mask)
        t_d = (time.perf_counter() - t) * 1000.0
        _trace("phase: denoise done"); _spm_dump(self._rb, "after_denoise_fwd")
        self.last_phase_ms = {"vision_ms": t_v, "prefill_ms": t_p, "denoise_ms": t_d}
        return act.detach().float().cpu()
