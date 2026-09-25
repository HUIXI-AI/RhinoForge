"""GR00T-N1.7-3B image → action runtime.

The RPU Qwen3-VL backbone produces [1,S,2048] hidden states. The visual-language
encoder applies normalization and self-attention, then a graph-owned Euler
denoise loop combines state/action encoders, DiT and the action decoder.

Mutable-DMA inputs remain alive across each op call. Text and image MASK_2D
subsets have independent prepared native slots. The standalone runtime does
not require importing the upstream GR00T package."""
from __future__ import annotations
import os, math, glob
import inspect
from contextlib import nullcontext
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from safetensors import safe_open

import rpu_backend
from rpu_backend.api import RPUCache
from rpu_backend.api._execution import (
    bind_execution_session,
    execution_serialized,
    native_execution_reconfigure,
    normalize_rpu_execution,
    resolve_component_rpu_execution,
)
from rpu_backend.runtime import rpu_env_bool
from rpu_backend.runtime.decoder import (
    _run_causal_decoder_forward,
    chunk_policy_key,
    plan_bounded_prefill_execution,
)
from rpu_backend.runtime.execution_planner import (
    GRAPH_COMPOSITE_CHILD,
    GRAPH_NATIVE_COMPOSITE1,
    StageTuple,
    capability_selected,
    plan_a6,
)
from rpu_backend.runtime.weights import (
    convert_linear_weights_inplace, tp_col_swizzle_mc_weight, tp_row_swizzle_mc_weight,
)
from rpu_backend.adapters.qwen3_vl import (
    install_qwen3_vl_vision_for_rpu, install_qwen3_vl_text_for_rpu,
    enable_qwen3_vl_vision_merger_on_device, register_qwen3vl_vision_fused_merger,
    uninstall_qwen3_vl_runtime,
)
from rpu_backend.adapters.qwen3_vl.vision import (
    _prepare_qwen3vl_vision_input_for_spm_pipeline,
)
from rpu_backend.runtime.rope_partial import build_interleaved_mrope_cos_sin

# DiT geometry (config.json action_head_cfg)
_H, _NQ, _HD, _FF, _CROSS, _NL = 1536, 32, 48, 6144, 2048, 32
_VLD, _VLH, _VLHD, _VLFF, _VL_NL = 2048, 32, 64, 8192, 4   # vl_self_attention (A)
_AH, _AD, _AD_PAD = 40, 132, 144                            # action_horizon, action_dim, padded
_NSTEPS, _NBUCKETS = 4, 1000
_TP = 8
_AP = "action_head."
_NEG = float("-inf")
_SELECT_LAYER = 16
_VISION_PATCHES_PER_IMAGE = 256
_ACTION_QUERY_ROWS = _AH + 1
_ACTION_CHUNK_SIZE = ((_ACTION_QUERY_ROWS + 15) // 16) * 16
_PREFILL_PADDING_BUDGET = 64
_VISION_EXACT_CHUNKS = frozenset((144, 256, 512, 768))

_TEXT_COMPONENT = "text_backbone"
_VISION_COMPONENT = "vision_encoder"
_VL_COMPONENT = "vl_encoder"
_ACTION_COMPONENT = "action_dit"
_EXECUTION_COMPONENTS = {
    _TEXT_COMPONENT: {
        "prefill": ("chunk_size", "padding_rows", "padding_budget"),
    },
    _VISION_COMPONENT: {"vision": ("chunk_size",)},
    # The VL encoder has one shape-derived, full-sequence chunk.  Its exact
    # value is therefore meaningful only as a component override for the
    # current prompt, never as a second process-global parser/knob.
    _VL_COMPONENT: {"prefill": ("chunk_size",)},
    _ACTION_COMPONENT: {"action": ("chunk_size",)},
}


def _get_gr00t_rope_index(
    get_rope_index,
    *,
    input_ids,
    mm_token_type_ids,
    image_grid_thw,
    attention_mask,
):
    """Call the Qwen3-VL 4.57/5.x M-RoPE API without masking real errors."""
    kwargs = {
        "input_ids": input_ids.cpu(),
        "image_grid_thw": image_grid_thw.cpu(),
        "video_grid_thw": None,
        "attention_mask": attention_mask.cpu(),
    }
    if "mm_token_type_ids" in inspect.signature(get_rope_index).parameters:
        kwargs["mm_token_type_ids"] = mm_token_type_ids.cpu()
    return get_rope_index(**kwargs)


def _resolve_gr00t_execution(
    rpu_execution, *, entry_point="build_gr00t_vla", qwen_z1=False,
    fused_merger=None,
):
    root = normalize_rpu_execution(
        rpu_execution,
        entry_point=entry_point,
        supported_components=_EXECUTION_COMPONENTS,
    )
    text = resolve_component_rpu_execution(
        root,
        _TEXT_COMPONENT,
        entry_point=entry_point,
        supported_components=_EXECUTION_COMPONENTS,
        profile_auto={"prefill": {"chunk_size": "auto"}},
    )
    vision = resolve_component_rpu_execution(
        root,
        _VISION_COMPONENT,
        entry_point=entry_point,
        supported_components=_EXECUTION_COMPONENTS,
        profile_auto={"vision": {"chunk_size": "auto"}},
    )
    vl = resolve_component_rpu_execution(
        root,
        _VL_COMPONENT,
        entry_point=entry_point,
        supported_components=_EXECUTION_COMPONENTS,
        profile_auto={"prefill": {"chunk_size": "auto"}},
    )
    action = resolve_component_rpu_execution(
        root,
        _ACTION_COMPONENT,
        entry_point=entry_point,
        supported_components=_EXECUTION_COMPONENTS,
        profile_auto={"action": {"chunk_size": "auto"}},
    )
    _validate_gr00t_execution_geometry(
        text, vision, vl, action, entry_point=entry_point, qwen_z1=qwen_z1,
        fused_merger=fused_merger,
    )
    if qwen_z1:
        # AUTO and the admitted exact values name the same frozen native Z1
        # singleton.  Canonical child views keep its Graph identity stable
        # across an equivalent hot config publication.
        text = normalize_rpu_execution(
            {"prefill": {"chunk_size": "auto"}},
            entry_point=entry_point,
            supported=_EXECUTION_COMPONENTS[_TEXT_COMPONENT],
        )
        vision = normalize_rpu_execution(
            {"vision": {"chunk_size": "auto"}},
            entry_point=entry_point,
            supported=_EXECUTION_COMPONENTS[_VISION_COMPONENT],
        )
    return root, text, vision, vl, action


def _normalize_gr00t_execution(rpu_execution):
    return _resolve_gr00t_execution(rpu_execution)[0]


def _validate_gr00t_execution_geometry(
    text, vision, vl, action, *, entry_point, qwen_z1, fused_merger=None,
) -> None:
    """Reject fixed-stage requests that do not match GR00T's real geometry."""
    prefill_chunk = text.get("prefill", {}).get(
        "chunk_size", "auto"
    )
    if isinstance(prefill_chunk, int):
        raise ValueError(
            f"{entry_point}: fixed text_backbone prefill.chunk_size is not certified for "
            "the truncated 16-layer text geometry; use 'auto'. Hardware "
            "rejected exact 16/32/64/128 at the shipped S=82/S=148 inputs. "
            "Refusing before loading model weights."
        )
    vision_chunk = vision.get("vision", {}).get("chunk_size", "auto")
    if (
        isinstance(vision_chunk, int)
        and vision_chunk not in _VISION_EXACT_CHUNKS
    ):
        raise ValueError(
            f"{entry_point}: vision.chunk_size must be 'auto' or one of "
            f"{sorted(_VISION_EXACT_CHUNKS)}, the certified native Vision "
            f"dispatch chunks; got {vision_chunk}."
        )
    if fused_merger is None:
        fused_merger = rpu_env_bool("RPU_QWEN3VL_VISION_FUSED_MERGER")
    if vision_chunk == 144 and fused_merger:
        raise ValueError(
            f"{entry_point}: vision.chunk_size=144 is incompatible with "
            "RPU_QWEN3VL_VISION_FUSED_MERGER=1; choose 256 for a standard "
            "256-patch image or disable the fused merger. Refusing before "
            "loading model weights."
        )
    vl_chunk = vl.get("prefill", {}).get("chunk_size", "auto")
    if isinstance(vl_chunk, int) and vl_chunk > 256:
        raise ValueError(
            f"{entry_point}: vl_encoder prefill.chunk_size exceeds its "
            f"256-row cache/envelope; got {vl_chunk}."
        )
    action_chunk = action.get("action", {}).get("chunk_size", "auto")
    if isinstance(action_chunk, int) and action_chunk != _ACTION_CHUNK_SIZE:
        raise ValueError(
            f"{entry_point}: action.chunk_size must be 'auto' or exactly "
            f"{_ACTION_CHUNK_SIZE}, the single chunk for "
            f"{_ACTION_QUERY_ROWS} fixed action-query rows; got {action_chunk}. "
            "GR00T action does not expose variable chunking."
        )
    if qwen_z1:
        text_stage = text.get("prefill", {})
        padding_rows = text_stage.get("padding_rows", 0)
        padding_budget = text_stage.get("padding_budget", 0)
        if padding_rows not in (0, None) or padding_budget != 0:
            raise ValueError(
                f"{entry_point}: the frozen Qwen Z1 composite does not accept "
                "text padding overrides"
            )
        if vision_chunk not in ("auto", _QWEN_Z1_NUM_PATCHES):
            raise ValueError(
                f"{entry_point}: the frozen Qwen Z1 composite accepts only "
                "vision AUTO or exact chunk_size=256"
            )


def _pad_gr00t_prefill_inputs(
    inputs_embeds,
    attention_mask,
    position_ids,
    deepstack_dense,
    rope_cos_il,
    rope_sin_il,
    execution_len,
):
    """Right-pad every token-indexed GR00T text input in lockstep."""
    logical_len = int(inputs_embeds.shape[1])
    pad = int(execution_len) - logical_len
    if pad < 0:
        raise ValueError(
            f"GR00T prefill execution_len={execution_len} is smaller than "
            f"logical_len={logical_len}"
        )
    if pad == 0:
        return (
            inputs_embeds,
            attention_mask,
            position_ids,
            deepstack_dense,
            rope_cos_il,
            rope_sin_il,
        )
    if tuple(attention_mask.shape) != (1, logical_len):
        raise ValueError(
            "GR00T attention_mask must be [1, logical_len] before execution "
            f"padding, got {tuple(attention_mask.shape)}"
        )
    if tuple(position_ids.shape) != (3, 1, logical_len):
        raise ValueError(
            "GR00T M-RoPE position_ids must be [3, 1, logical_len] before "
            f"execution padding, got {tuple(position_ids.shape)}"
        )
    if any(tuple(dense.shape) != (logical_len, inputs_embeds.shape[-1])
           for dense in deepstack_dense):
        raise ValueError(
            "GR00T DeepStack inputs must all be [logical_len, hidden] before "
            "execution padding"
        )

    inputs_embeds = torch.cat(
        [inputs_embeds, inputs_embeds.new_zeros(1, pad, inputs_embeds.shape[-1])],
        dim=1,
    ).contiguous()
    attention_mask = torch.cat(
        [attention_mask, attention_mask.new_zeros(1, pad)], dim=-1,
    ).contiguous()
    position_ids = torch.cat(
        [position_ids, position_ids[..., -1:].expand(3, 1, pad)], dim=-1,
    ).contiguous()
    deepstack_dense = [
        torch.cat(
            [dense, dense.new_zeros(pad, dense.shape[-1])], dim=0,
        ).contiguous()
        for dense in deepstack_dense
    ]

    def _pad_partial_rope(table):
        if table is None:
            return None
        if table.ndim != 2 or table.shape[0] != logical_len:
            raise ValueError(
                "GR00T partial M-RoPE table must be [logical_len, head_dim/2] "
                f"before execution padding, got {tuple(table.shape)}"
            )
        return torch.cat(
            [table, table[-1:].expand(pad, table.shape[-1])], dim=0,
        ).contiguous()

    return (
        inputs_embeds,
        attention_mask,
        position_ids,
        deepstack_dense,
        _pad_partial_rope(rope_cos_il),
        _pad_partial_rope(rope_sin_il),
    )


def _restore_gr00t_logical_prefill(raw, cache, logical_len, execution_len):
    if int(execution_len) == int(logical_len):
        return raw
    cache.reset_to_position(int(logical_len))
    return raw[:, :int(logical_len)]

_QWEN_Z1_NUM_PATCHES = 256
_QWEN_Z1_EXECUTION_LEN = 128
_QWEN_Z1_REAL_LEN = 82
_QWEN_Z1_IMAGE_BEGIN = 4
_QWEN_Z1_IMAGE_END = 68
_QWEN_Z1_DEEPSTACK_COUNT = 3


def _half_rpu(t):
    return t.to(torch.float16).to("rpu").contiguous()


def _w8a16_on():
    return rpu_env_bool("RPU_GR00T_W8A16")


def _qwen3vl_spm_z1_on():
    return os.environ.get("RPU_GR00T_QWEN3VL_SPM_Z1", "") in ("1", "true")


def _partial_mrope_on():
    """RPU_GR00T_PARTIAL_MROPE: swap the backbone's 32 per-layer in-kernel
    llama_mrope_interleave (full-table + position_ids gather + strobe masks, redone every
    layer/forward) for partial_mrope (rotation only, fed host-baked per-token interleaved
    cos/sin computed ONCE per prompt). Bit-exact @ partial_rotary 1.0 — wall_oss precedent
    (RPU_WALL_OSS_PARTIAL_MROPE) + tests/model/wall_oss/test_partial_mrope_interleave_equiv.py.
    Default off → legacy path, byte-identical."""
    return rpu_env_bool("RPU_GR00T_PARTIAL_MROPE")


def _extract_mrope_params(text_config):
    """(head_dim, rope_theta, mrope_section) for the partial_mrope host bake — mirrors
    adapters/qwen3_vl/text.py exactly so the interleaved cos/sin is bit-identical to the
    in-kernel table the legacy path gathers from."""
    rp = getattr(text_config, "rope_parameters", None) or getattr(text_config, "rope_scaling", None) or {}
    mrope_section = [int(x) for x in rp["mrope_section"]]
    head_dim = getattr(text_config, "head_dim", None) or (
        text_config.hidden_size // text_config.num_attention_heads)
    rope_theta = rp.get("rope_theta", None) or getattr(text_config, "rope_theta", 10000.0)
    return int(head_dim), float(rope_theta), mrope_section


def _quantize_backbone_inplace(text_model):
    """On-the-fly W8A16 for the Qwen3-VL text decoder (backbone). Replace each layer's 7
    projection Linears' fp16 weight with per-output-channel int8 + register a fp16
    `weight_scale` buffer. MUST run BEFORE `convert_linear_weights_inplace` (which then
    swizzles the int8 weight at dwidth=1) and BEFORE `.to('rpu')`. Mirrors the qwen3 W8A16
    module layout (`quant/load.py::load_w8a16_model`) so `_install_causal_decoder_forward`'s
    scale_lists path applies unchanged. Norms/embedding/cos-sin stay fp16. All backbone dims
    are int8/8-core aligned (q=2048,k/v=1024,gate/up=6144 → /8 then /32)."""
    import torch.nn as nn
    from rpu_backend.quant._common import quantize_linear_per_channel
    for layer in text_model.layers:
        for mod in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
                    layer.self_attn.o_proj, layer.mlp.gate_proj, layer.mlp.up_proj,
                    layer.mlp.down_proj):
            w_int8, scale = quantize_linear_per_channel(mod.weight.data)
            mod.weight = nn.Parameter(w_int8, requires_grad=False)
            mod.register_buffer("weight_scale", scale.to(torch.float16))


def _backbone_scale_lists(text_model):
    """Gather the 7 per-output-channel fp16 scale lists (q,k,v,o,gate,up,down) from the
    quantized backbone modules (post `.to('rpu')`). Returns None if not int8-quantized."""
    layers = text_model.layers
    if getattr(layers[0].self_attn.q_proj.weight, "dtype", None) != torch.int8:
        return None
    N = len(layers)
    pick = lambda get: [get(layers[i]) for i in range(N)]
    return (
        pick(lambda l: l.self_attn.q_proj.weight_scale),
        pick(lambda l: l.self_attn.k_proj.weight_scale),
        pick(lambda l: l.self_attn.v_proj.weight_scale),
        pick(lambda l: l.self_attn.o_proj.weight_scale),
        pick(lambda l: l.mlp.gate_proj.weight_scale),
        pick(lambda l: l.mlp.up_proj.weight_scale),
        pick(lambda l: l.mlp.down_proj.weight_scale),
    )


def _destroy_gr00t_vl_encoder_handle(handle):
    torch.ops.rpu.gr00t_vl_encoder_destroy(handle)


def _destroy_gr00t_dit_handle(handle):
    torch.ops.rpu.gr00t_dit_destroy(handle)


_FAILED_RETIREMENTS = []


def _poison_gr00t_retirement(owner, error):
    if not any(value is owner for value in _FAILED_RETIREMENTS):
        _FAILED_RETIREMENTS.append(owner)
    if getattr(owner, "__dict__", None) is not None:
        owner._retirement_failed = error
    session = getattr(owner, "_execution_session", None)
    if session is not None:
        session.poison()
    from rpu_backend.api._execution import _mark_execution_process_unsafe
    _mark_execution_process_unsafe(f"GR00T retirement failed: {error}")


def _cleanup_gr00t_build_failure(error, owner, cleanup):
    from rpu_backend.api._execution import _require_execution_process_safe
    try:
        _require_execution_process_safe()
        cleanup()
    except BaseException as cleanup_error:
        _poison_gr00t_retirement(owner, cleanup_error)
        error.add_note(f"GR00T cleanup also failed: {cleanup_error}")


def _configure_gr00t_vl_encoder_handle(
    *, weight_args, scale_args, w8a16: bool
):
    from rpu_backend.api._execution import _require_execution_process_safe
    _require_execution_process_safe()
    handle = torch.ops.rpu.gr00t_vl_encoder_create()
    try:
        if w8a16:
            torch.ops.rpu.gr00t_vl_encoder_set_weights_w8a16(
                handle, *weight_args, *scale_args
            )
        else:
            torch.ops.rpu.gr00t_vl_encoder_set_weights(handle, *weight_args)
        return handle
    except BaseException as error:
        _cleanup_gr00t_build_failure(error, (handle, weight_args, scale_args),
                                    lambda: _destroy_gr00t_vl_encoder_handle(handle))
        raise


def _configure_gr00t_denoise_handle(
    *, dit_args, dit_scale_args, denoise_args, w8a16: bool
):
    from rpu_backend.api._execution import _require_execution_process_safe
    _require_execution_process_safe()
    handle = torch.ops.rpu.gr00t_dit_create()
    try:
        if w8a16:
            torch.ops.rpu.gr00t_dit_set_weights_w8a16(
                handle, *dit_args, *dit_scale_args
            )
        else:
            torch.ops.rpu.gr00t_dit_set_weights(handle, *dit_args)
        torch.ops.rpu.gr00t_denoise_set_weights(handle, *denoise_args)
        return handle
    except BaseException as error:
        _cleanup_gr00t_build_failure(error, (handle, dit_args, dit_scale_args, denoise_args),
                                    lambda: _destroy_gr00t_dit_handle(handle))
        raise


def _clear_graph_cache(owner, attr: str) -> None:
    cache = getattr(owner, attr, None)
    if cache is not None:
        cache.clear()
        if not cache.cache_invariant_ok():
            raise RuntimeError("GR00T GraphCache invariant failed during close")


def _unprepare_qwen3vl_spm_z1(
    *, pipeline_cache, vision_model, text_model, plan_hash
) -> None:
    """Retire the composite Graph before releasing either native handle."""
    if plan_hash is None:
        return
    if pipeline_cache is not None:
        pipeline_cache.clear()
        if not pipeline_cache.cache_invariant_ok():
            raise RuntimeError("GR00T composite GraphCache invariant failed during close")
    torch.ops.rpu.qwen3vl_pooler_spm_z1_unprepare(
        int(vision_model._rpu_vision_handle),
        int(text_model._rpu_decoder_handle),
        int(plan_hash),
    )


def _unprepare_gr00t_spm_z2(*, a_handle, d_handle, plan_hash) -> None:
    """Retire the temporary A/D layout authority after persistent preflight."""
    if plan_hash is None:
        return
    torch.ops.rpu.gr00t_spm_z2_unprepare(
        int(a_handle), int(d_handle), int(plan_hash)
    )


def _assert_qwen3vl_component_caches_empty(vision_model, text_model) -> None:
    """The composite Graph is the sole graph authority for the Z1 path."""
    caches = {
        "vision": vision_model._rpu_vision_graph_cache,
        "text": text_model._rpu_text_graph_cache,
        "decoder": text_model._rpu_decoder_graph_cache,
    }
    sizes = {name: int(cache.size()) for name, cache in caches.items()}
    if any(sizes.values()):
        raise RuntimeError(
            "GR00T Qwen3-VL SPM Z1 component GraphCache grew: "
            f"{sizes}"
        )


def _preflight_ready_graph(cache, signature, label: str) -> None:
    """Fail a frozen exact-profile miss before acquiring physical SPM."""
    if not cache.is_frozen():
        return
    graph = cache.lookup(signature)
    if (
        graph is None
        or graph.state() != 2
        or not graph.replayable()
        or not graph.has_built_signature()
        or graph.built_signature() != signature._impl
    ):
        raise RuntimeError(
            f"{label}: READY GraphCache has no matching replayable graph"
        )


def _take_gr00t_backbone_ownership(owner):
    # These installers are shared with standalone Qwen3-VL. GR00T's composite
    # now owns their lifetime; no integer-only callback may run ahead of it.
    for child, finalizer_name in (
        (getattr(owner, "_text", None), "_rpu_decoder_handle_finalizer"),
        (getattr(owner, "_vision", None), "_rpu_vision_handle_finalizer"),
    ):
        if child is None:
            continue
        finalizer = getattr(child, finalizer_name, None)
        if finalizer is not None:
            if not finalizer.alive and getattr(child, "_gr00t_retirement_owner", None) is None:
                raise RuntimeError("GR00T cannot adopt a consumed backbone finalizer")
            if finalizer.alive:
                finalizer.detach()
        child._gr00t_retirement_owner = owner
        resource = getattr(child, finalizer_name.replace("handle_finalizer", "retirement_state"), None)
        if resource is not None:
            resource.take_ownership(owner)


def _retire_gr00t_resources(owner):
    from rpu_backend.api._execution import _require_execution_process_safe
    for name in ("_qwen_z1_completed_owner", "_qwen_z1_pending_owner",
                 "_qwen_z1_completed_ranges", "_qwen_z1_pending_ranges",
                 "_qwen_z1_composite_plan_cache", "_qwen_z1_planner_parent",
                 "_qwen_z1_planner_children"):
        vars(owner).pop(name, None)
    if getattr(owner, "_retirement_failed", None) is not None:
        raise RuntimeError("GR00T cleanup failed; restart the process")
    if getattr(owner, "_closed", False):
        return
    try:
        _require_execution_process_safe()
        text, vision = getattr(owner, "_text", None), getattr(owner, "_vision", None)
        seen = set()
        for child, attrs in (
            (owner, ("_qwen_z1_gc", "_d_gc", "_a_gc")),
            (text, ("_rpu_text_graph_cache", "_rpu_decoder_graph_cache")),
            (vision, ("_rpu_vision_graph_cache",)),
        ):
            for attr in attrs:
                cache = getattr(child, attr, None)
                if cache is not None and id(cache) not in seen:
                    seen.add(id(cache))
                    _clear_graph_cache(child, attr)
        if getattr(owner, "_qwen_z1_plan_hash", None) is not None:
            _unprepare_qwen3vl_spm_z1(
                pipeline_cache=None, vision_model=vision, text_model=text,
                plan_hash=owner._qwen_z1_plan_hash)
            owner._qwen_z1_plan_hash = None
            owner._qwen_z1_vision_stage_descriptor = ()
        if getattr(owner, "_gr00t_z2_plan_hash", None) is not None:
            _unprepare_gr00t_spm_z2(a_handle=owner._a_handle, d_handle=owner._d_handle,
                                   plan_hash=owner._gr00t_z2_plan_hash)
            owner._gr00t_z2_plan_hash = None
        for child, attr, finalizer_name, destroy in (
            (owner, "_d_handle", "_d_handle_finalizer", _destroy_gr00t_dit_handle),
            (owner, "_a_handle", "_a_handle_finalizer", _destroy_gr00t_vl_encoder_handle),
            (text, "_rpu_decoder_handle", "_rpu_decoder_handle_finalizer",
             lambda handle: torch.ops.rpu.causal_decoder_destroy(handle)),
            (vision, "_rpu_vision_handle", "_rpu_vision_handle_finalizer",
             lambda handle: torch.ops.rpu.qwen3vl_vision_destroy(handle)),
        ):
            handle = getattr(child, attr, None)
            if handle is None:
                continue
            finalizer = getattr(child, finalizer_name, None)
            if (finalizer is not None and not finalizer.alive
                    and getattr(child, "_gr00t_retirement_owner", None) is None):
                raise RuntimeError("GR00T live handle has a consumed finalizer")
            resource = getattr(child, finalizer_name.replace("handle_finalizer", "retirement_state"), None)
            if resource is not None and resource.handle != handle:
                raise RuntimeError("GR00T backbone retirement handle identity changed")
            destroy(handle)
            if resource is not None:
                resource._native_destroyed(handle)
            setattr(child, attr, None)
            if finalizer is not None and finalizer.alive:
                finalizer.detach()
        if vision is not None:
            vars(vision).pop("_rpu_execution_graph_key_words", None)
        owner._closed = True
    except BaseException as error:
        for child, resource_attr in (
            (getattr(owner, "_text", None), "_rpu_decoder_retirement_state"),
            (getattr(owner, "_vision", None), "_rpu_vision_retirement_state"),
        ):
            resource = getattr(child, resource_attr, None)
            if resource is not None and resource.handle is not None:
                resource.retain_failure(error, owner)
        _poison_gr00t_retirement(owner, error)
        raise


def _cleanup_gr00t_partial_build(
    *, a_handle, d_handle, model, text_model, vision_model, forward_snapshots=(),
    qwen_z1_pipeline_cache=None, qwen_z1_plan_hash=None,
    gr00t_z2_plan_hash=None, a_keep=None, d_keep=None,
) -> None:
    """Retire raw component handles, then UNINSTALL the backbone.

    The backbone half delegates to `uninstall_qwen3_vl_runtime` — the
    uninstaller that pairs with the `install_qwen3_vl_{vision,text}_for_rpu`
    INSTALLERS this builder already reuses. What used to be here instead was
    `_cleanup_gr00t_backbone`: two graph-cache clears and two finalizer calls,
    deleting NO `_rpu_*` attribute and restoring NO `forward`. By the time this
    runs, `_quantize_backbone_inplace` and `convert_linear_weights_inplace`
    have already mutated the weights irreversibly, so leaving the module tree
    stamped as installed — with `_rpu_decoder_handle` pointing at a handle we
    just destroyed and the patched forwards still bound — hands the caller a
    model that looks usable and is not.
    """
    pending = SimpleNamespace(
        _bb=model, _text=text_model, _vision=vision_model,
        _a_handle=a_handle, _d_handle=d_handle, _a_keep=a_keep, _d_keep=d_keep,
        _qwen_z1_gc=qwen_z1_pipeline_cache, _qwen_z1_plan_hash=qwen_z1_plan_hash,
        _gr00t_z2_plan_hash=gr00t_z2_plan_hash, _closed=False,
    )
    try:
        _take_gr00t_backbone_ownership(pending)
        _retire_gr00t_resources(pending)
        if model is not None:
            # The shared uninstaller now sees no live handles; it only restores
            # the original forwards and removes the obsolete install attributes.
            uninstall_qwen3_vl_runtime(model, text_model, vision_model, forward_snapshots)
        from rpu_backend.api.causal_lm import _release_live_instance
        _release_live_instance(model)
    except BaseException as error:
        _poison_gr00t_retirement(pending, error)
        raise


class Gr00tN1d7VLA:
    """Image→action runtime (fully on-device glue). Build via `build_gr00t_vla(...)`."""

    def __init__(self, *, backbone, cfg, text_model, vision_model,
                 a_handle, a_keep, a_cache, d_handle, d_keep, d_cache, image_token_id,
                 execution_config=None, embodiment_id=20,
                 qwen_z1_plan_hash=None,
                 qwen_z1_vision_stage_descriptor=None):
        self._gc_retirement_enabled = False
        self._closed = False
        self._bb = backbone
        self._cfg = cfg
        self._text = text_model
        self._vision = vision_model
        self._a_handle = a_handle           # gr00t_vl_encoder
        self._a_keep = a_keep               # weight keepalive
        self._d_handle = d_handle           # gr00t_denoise (extends gr00t_dit)
        self._d_keep = d_keep
        self._qwen_z1_plan_hash = qwen_z1_plan_hash
        _take_gr00t_backbone_ownership(self)
        self._img_tok = image_token_id
        self._emb = embodiment_id
        self._vision_fused_merger = bool(getattr(
            vision_model, "_rpu_vision_fused_merger", False
        ))
        resolved_execution = _resolve_gr00t_execution(
            execution_config,
            entry_point="Gr00tN1d7VLA",
            qwen_z1=qwen_z1_plan_hash is not None,
            fused_merger=self._vision_fused_merger,
        )
        self._execution_generation = 0
        self._component_generations = {
            component: 0 for component in _EXECUTION_COMPONENTS
        }
        self._execution_reconfigure_journal = None
        self._rpu_last_execution_plan = {}
        self._a_gc = rpu_backend.graph.GraphCache()
        self._d_gc = rpu_backend.graph.GraphCache()
        self._qwen_z1_plan_hash = qwen_z1_plan_hash
        self._qwen_z1_vision_stage_descriptor = tuple(
            int(word) for word in (qwen_z1_vision_stage_descriptor or ())
        )
        if (
            qwen_z1_plan_hash is not None
            and not self._qwen_z1_vision_stage_descriptor
        ):
            raise RuntimeError(
                "GR00T Qwen3-VL SPM Z1 requires its prepared Vision descriptor"
            )
        self._qwen_z1_gc = (
            rpu_backend.graph.GraphCache(max_entries=1).begin_warmup()
            if qwen_z1_plan_hash is not None else None
        )
        if qwen_z1_plan_hash is not None:
            self._a_gc.begin_warmup()
            self._d_gc.begin_warmup()
        self._qwen_z1_text_hidden = None
        self._qwen_z1_position_ids = None
        self._a_cache = a_cache
        self._d_cache = d_cache
        # Opt1: backbone KV-cache hoisted out of the per-action _backbone (single non-incremental
        # prefill → one cache, reused via reset_to_position(0); kills the dominant per-forward
        # 16-layer alloc+zero_ and pins REPLAY KV addresses). cap 512 ≥ max prompt S.
        self._bb_cache = RPUCache(num_layers=cfg.text_config.num_hidden_layers, batch_size=1,
                                  max_seq_len=512, num_kv_heads=cfg.text_config.num_key_value_heads,
                                  head_dim=cfg.text_config.head_dim)
        # Opt2/3: per-prompt memo slots (keyed on input_ids identity). rope = CPU M-RoPE index math;
        # masks = the two cross-attn MASK_2D subsets (cached RPU tensors also pin denoise REPLAY addrs).
        self._rope_key = self._rope_ids = None
        self._pos_rpu = self._attn_rpu = None   # Strategy-2: RPU position_ids/attention_mask, memoized w/ _rope_ids
        self._mask_key = self._masks = None
        # Opt-a: inputs_embeds kept RESIDENT on RPU. The base text embedding is prompt-constant; the
        # image tokens form contiguous runs (1 per image), so each forward overwrites only those
        # slices with the (per-image) visual embeds on-device — no CPU masked_scatter, no RPU↔CPU
        # bounce, stable REPLAY address. Memoized on input_ids: the RPU text-embed buffer + the runs.
        self._emb_key = self._emb_text_rpu = self._img_runs = None
        # Opt #2: persistent deepstack dense buffers (one per merger), memoized on input_ids.
        self._ds_dense = None
        # partial_mrope (opt-in RPU_GR00T_PARTIAL_MROPE): host-bake the per-token interleaved
        # M-RoPE cos/sin ONCE per prompt (memoized on input_ids alongside _rope_ids) and stream it
        # to the backbone → the 32 per-layer kernels do rotation only (partial_mrope) instead of
        # re-deriving the interleave from the full table each call (llama_mrope_interleave).
        self._partial_mrope = (
            False if qwen_z1_plan_hash is not None else _partial_mrope_on()
        )
        self._mrope_hd, self._mrope_theta, self._mrope_section = _extract_mrope_params(cfg.text_config)
        self._rope_cos_il = self._rope_sin_il = None
        self._closed = False
        self._publish_execution_views(
            resolved_execution,
            generation=0,
            component_generations=self._component_generations,
        )
        self._execution_session = bind_execution_session(
            self,
            self._rpu_execution,
            entry_point="Gr00tN1d7VLA",
            supported_components=_EXECUTION_COMPONENTS,
            validate=self.validate_execution_reconfigure,
            apply=self.apply_execution_reconfigure,
            rollback=self.rollback_execution_reconfigure,
            graph_mode=(
                GRAPH_NATIVE_COMPOSITE1
                if qwen_z1_plan_hash is not None
                else GRAPH_COMPOSITE_CHILD
            ),
        )
        for child, component in (
            (self._text, _TEXT_COMPONENT), (self._vision, _VISION_COMPONENT),
            (self, _VL_COMPONENT), (self, _ACTION_COMPONENT),
        ):
            self._execution_session._bind_planner_owner(child, component)
        self._qwen_z1_catalog_identity = (
            self._qwen_z1_native_catalog_identity()
            if qwen_z1_plan_hash is not None else None
        )
        self._gc_retirement_enabled = True

    def _close_without_session(self) -> None:
        """Release all GR00T/backbone graph and native resources, once."""
        _retire_gr00t_resources(self)
        from rpu_backend.api.causal_lm import _release_live_instance
        _release_live_instance(self._bb)

    def close(self) -> None:
        """Retire all children through the shared stop-the-world session."""
        session = getattr(self, "_execution_session", None)
        try:
            if session is None:
                self._close_without_session()
            else:
                session.shutdown(self._close_without_session)
        except BaseException as error:
            from rpu_backend.api import _execution
            if _execution._UNSAFE_PROCESS_REASON is not None:
                _poison_gr00t_retirement(self, error)
            raise
        if not getattr(self, "_closed", False):
            error = RuntimeError("GR00T closed Session still owns native resources")
            _poison_gr00t_retirement(self, error)
            raise error

    def __del__(self):
        if (not getattr(self, "_gc_retirement_enabled", False)
                or getattr(self, "_closed", False)
                or getattr(self, "_retirement_failed", None) is not None):
            return
        session = getattr(self, "_execution_session", None)
        try:
            with session._lock if session is not None else nullcontext():
                if session is not None and session._active:
                    raise RuntimeError("GR00T GC during an active forward")
                from rpu_backend.api import causal_lm
                with causal_lm._LIVE_LOCK:
                    live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
                    if live is not None and live is not getattr(self, "_bb", None):
                        raise RuntimeError("GR00T GC after live-owner handoff")
                    self.close()
        except BaseException as error:
            _poison_gr00t_retirement(self, error)

    def _publish_execution_views(
        self, resolved, *, generation, component_generations=None,
    ) -> None:
        root, text, vision, vl, action = resolved
        if component_generations is not None:
            self._component_generations = dict(component_generations)
        self._rpu_execution = root
        self._text_execution = text
        self._vision_execution = vision
        self._vl_execution = vl
        self._action_execution = action
        self._execution_generation = int(generation)
        children = (
            (self._text, _TEXT_COMPONENT, text),
            (self._vision, _VISION_COMPONENT, vision),
        )
        for child, component, config in children:
            state = vars(child)
            state["_rpu_execution"] = config
            state["_fmb_execution_component_id"] = component
            state["_fmb_execution_generation"] = int(
                self._component_generations[component]
            )
        vars(self._vision)["_rpu_vision_execution_chunk_size"] = (
            vision.get("vision", {}).get("chunk_size", "auto")
        )
        vars(self._vision)["_rpu_execution_graph_key_words"] = (
            int(self._component_generations[_VISION_COMPONENT]),
        )
        self._execution_components = {
            _TEXT_COMPONENT: text,
            _VISION_COMPONENT: vision,
            _VL_COMPONENT: vl,
            _ACTION_COMPONENT: action,
        }

    @staticmethod
    def _native_chunk(config, stage):
        value = config.get(stage, {}).get("chunk_size", "auto")
        return 0 if value == "auto" else int(value)

    @staticmethod
    def _reset_graph_owner(cache, *, label):
        if cache is None:
            return
        cache.begin_warmup()
        cache.clear()
        if not cache.cache_invariant_ok():
            raise RuntimeError(
                f"GR00T {label} GraphCache invariant failed during reconfigure"
            )

    def _reset_component_graphs(self, changed) -> None:
        vars(self).pop("_qwen_z1_completed_owner", None)
        vars(self).pop("_qwen_z1_pending_owner", None)
        vars(self).pop("_qwen_z1_completed_ranges", None)
        vars(self).pop("_qwen_z1_pending_ranges", None)
        if _TEXT_COMPONENT in changed or _VISION_COMPONENT in changed:
            vars(self).pop("_qwen_z1_composite_plan_cache", None)
            vars(self).pop("_qwen_z1_planner_parent", None)
            vars(self).pop("_qwen_z1_planner_children", None)
        seen = set()

        def reset(cache, label):
            if cache is not None and id(cache) not in seen:
                seen.add(id(cache))
                self._reset_graph_owner(cache, label=label)

        if _TEXT_COMPONENT in changed:
            reset(getattr(self._text, "_rpu_text_graph_cache", None), "text")
            reset(
                getattr(self._text, "_rpu_decoder_graph_cache", None),
                "text decoder",
            )
            vars(self._text).pop("_rpu_last_execution_plan", None)
        if _VISION_COMPONENT in changed:
            reset(
                getattr(self._vision, "_rpu_vision_graph_cache", None),
                "vision",
            )
            vars(self._vision).pop("_rpu_last_execution_plan", None)
        if (_TEXT_COMPONENT in changed or _VISION_COMPONENT in changed) and (
            self._qwen_z1_gc is not None
        ):
            reset(self._qwen_z1_gc, "Qwen Z1 composite")
        if _VL_COMPONENT in changed:
            reset(self._a_gc, "VL encoder")
        if _ACTION_COMPONENT in changed:
            reset(self._d_gc, "action DiT")
        for component in changed:
            self._rpu_last_execution_plan.pop(component, None)

    def _resolve_execution(self, value, *, entry_point):
        return _resolve_gr00t_execution(
            value,
            entry_point=entry_point,
            qwen_z1=self._qwen_z1_plan_hash is not None,
            fused_merger=self._vision_fused_merger,
        )

    def validate_execution_reconfigure(self, execution_config) -> None:
        if self._closed:
            raise RuntimeError("GR00T runtime is closed")
        resolved = self._resolve_execution(
            execution_config, entry_point="Gr00tN1d7VLA.reconfigure"
        )
        if (
            self._qwen_z1_plan_hash is not None
            or resolved[2] == self._vision_execution
        ):
            return
        required = (
            "execution_reconfigure_begin",
            "execution_reconfigure_commit",
            "execution_reconfigure_abort",
            "execution_reconfigure_abort_attempt",
            "qwen3vl_vision_stage_chunk_size",
        )
        missing = [name for name in required if not hasattr(torch.ops.rpu, name)]
        if missing:
            raise RuntimeError(
                "GR00T binary lacks native execution-reconfigure op(s): "
                + ", ".join(missing)
            )

    def _apply_execution_state(
        self,
        resolved,
        *,
        generation,
        component_generations=None,
        force_components=(),
        reprepare_native=False,
    ) -> None:
        root, text, vision, vl, action = resolved
        current = (
            self._text_execution,
            self._vision_execution,
            self._vl_execution,
            self._action_execution,
        )
        requested = (text, vision, vl, action)
        component_ids = (
            _TEXT_COMPONENT,
            _VISION_COMPONENT,
            _VL_COMPONENT,
            _ACTION_COMPONENT,
        )
        changed = {
            component
            for component, before, after in zip(
                component_ids, current, requested
            )
            if before != after
        }
        changed.update(force_components)
        if component_generations is None:
            component_generations = dict(self._component_generations)
            for component in changed:
                component_generations[component] += 1

        native_vision_changed = (
            _VISION_COMPONENT in changed
            and self._qwen_z1_plan_hash is None
        )
        if native_vision_changed:
            with native_execution_reconfigure(
                torch.ops.rpu, journal=self._execution_reconfigure_journal,
            ) as token:
                torch.ops.rpu.qwen3vl_vision_stage_chunk_size(
                    int(self._vision._rpu_vision_handle),
                    int(token),
                    self._native_chunk(vision, "vision"),
                )

        if self._execution_reconfigure_journal is not None:
            # Descriptor-only controls also mutate Python Graph/config state.
            self._execution_reconfigure_journal["mutation_started"] = True
        if reprepare_native and self._qwen_z1_plan_hash is not None:
            self._refresh_qwen_z1_cost_plan(force=True)
            # Calibration is already staged on this same quiescent Session.
            # Consume its winner before online forward acquires Session.run.
            self._qwen_z1_component_plans()
        else:
            self._reset_component_graphs(changed)
        self._publish_execution_views(
            (root, text, vision, vl, action),
            generation=generation,
            component_generations=component_generations,
        )

    def apply_execution_reconfigure(
        self, old_config, new_config, generation, *, force_rebuild=False,
    ) -> None:
        self._execution_reconfigure_journal = {"mutation_started": False}
        old_resolved = self._resolve_execution(
            old_config, entry_point="Gr00tN1d7VLA.reconfigure rollback"
        )
        new_resolved = self._resolve_execution(
            new_config, entry_point="Gr00tN1d7VLA.reconfigure"
        )
        changed = {
            component
            for component, old, new in zip(
                _EXECUTION_COMPONENTS, old_resolved[1:], new_resolved[1:]
            )
            if old != new
        }
        if force_rebuild:
            changed.update(_EXECUTION_COMPONENTS)
        self._execution_reconfigure_journal = {
            "mutation_started": False,
            "resolved": old_resolved,
            "generation": self._execution_generation,
            "component_generations": dict(self._component_generations),
            "changed": changed,
        }
        self._apply_execution_state(
            new_resolved, generation=generation, force_components=changed,
            reprepare_native=force_rebuild,
        )
        # Retain undo state through the shared session's config publication.

    def rollback_execution_reconfigure(
        self, old_config, _new_config, generation,
    ) -> None:
        journal = self._execution_reconfigure_journal
        if journal is not None and journal.get("native_reprepare_started", False):
            self._execution_session.poison()
            raise RuntimeError("GR00T native Z1 reprepare cannot restore an old prepared authority; restart the process")
        if journal is not None and not journal.get("mutation_started", False):
            self._execution_reconfigure_journal = None
            return
        if journal is None:
            resolved = self._resolve_execution(
                old_config, entry_point="Gr00tN1d7VLA.reconfigure rollback"
            )
            generation = int(generation)
            changed = {
                component
                for component, current, old in zip(
                    _EXECUTION_COMPONENTS,
                    (
                        self._text_execution,
                        self._vision_execution,
                        self._vl_execution,
                        self._action_execution,
                    ),
                    resolved[1:],
                )
                if current != old
            }
            component_generations = dict(self._component_generations)
        else:
            resolved = journal["resolved"]
            generation = journal["generation"]
            changed = journal["changed"]
            component_generations = journal["component_generations"]
        self._apply_execution_state(
            resolved,
            generation=generation,
            component_generations=component_generations,
            force_components=changed,
        )
        self._execution_reconfigure_journal = None

    # ---- RPU Qwen3-VL backbone (P1) -> raw [1,S,2048] post-RMSNorm ----
    def _record_component_plan(
        self,
        component,
        stage,
        plan,
        *,
        logical_len,
        resolved_chunk_size,
        physical_descriptor=None,
        authority="NATIVE_A6_STAGE_DESCRIPTOR",
    ):
        selected = plan.selected
        if physical_descriptor is None:
            dispatched_descriptor = None
        elif (selected is not None and type(physical_descriptor) is tuple
              and physical_descriptor == selected.stage_tuple.physical_descriptor):
            dispatched_descriptor = selected.stage_tuple.physical_descriptor
        else:
            dispatched_descriptor = tuple(int(word) for word in physical_descriptor)
        if (
            selected is None
            or selected.stage_tuple.compute_chunk != int(resolved_chunk_size)
            or (
                dispatched_descriptor is not None
                and tuple(selected.stage_tuple.physical_descriptor)
                != dispatched_descriptor
            )
        ):
            raise RuntimeError(
                f"GR00T {component} dry/forward plan disagreement"
            )
        receipt = plan.as_dict(include_candidates=False)
        receipt.update({
            "component": component,
            "stage": stage,
            "generation": int(self._component_generations[component]),
            "logical_len": int(logical_len),
            "execution_len": int(selected.execution_len),
            "chunk_size": int(resolved_chunk_size),
            "padding_rows": int(selected.padding_rows),
            "position": 0,
            "authority": authority,
            "selection_scope": plan.selection_scope,
            "physical_plan_digest": plan.physical_plan_digest,
            "plan_digest": plan.plan_digest,
            "descriptor_words": len(
                selected.stage_tuple.physical_descriptor
            ),
            "dry_forward_agreement": True,
        })
        if dispatched_descriptor is not None:
            receipt["physical_descriptor"] = dispatched_descriptor
        self._rpu_last_execution_plan[component] = receipt
        return receipt

    def _prefill_execution_plan(self, logical_len):
        stage = self._text_execution.get("prefill", {})
        exact_chunk = stage.get("chunk_size")
        if "padding_rows" in stage:
            padding_rows = stage["padding_rows"]
        elif "padding_budget" in stage or isinstance(exact_chunk, int):
            padding_rows = "auto"
        else:
            # Preserve the existing unpadded GR00T default. Padding is selected
            # only when the caller opts into either padding field.
            padding_rows = 0
        padding_budget = int(stage.get(
            "padding_budget",
            _PREFILL_PADDING_BUDGET if padding_rows == "auto" else 0,
        ))
        plan_box = {}
        execution_len, chunk_size = plan_bounded_prefill_execution(
            int(logical_len),
            int(self._bb_cache.max_seq_len),
            padding_budget,
            execution_owner=self._text,
            execution_component=_TEXT_COMPONENT,
            execution_stage="prefill",
            execution_native=("causal_decoder", int(self._text._rpu_decoder_handle)),
            queue_owner_id=int(self._text._rpu_decoder_handle),
            resolve_stage_domain=lambda execution_len: (
                torch.ops.rpu.causal_decoder_resolve_prefill_stage_domain(
                    self._text._rpu_decoder_handle, int(execution_len), 0,
                    True, 0, 1, 4, int(logical_len),
                )
            ),
            position=0,
            # GR00T's certified S=82/S=148 prefills already execute off-grid;
            # only the selected chunk, not its logical/physical sequence, is
            # required to be 16-aligned.
            alignment=1,
            padding_rows=padding_rows,
            exact_chunk_size=(
                exact_chunk
                if isinstance(exact_chunk, int)
                else None
            ),
            request_id="gr00t:text_backbone:prefill",
            graph_mode=GRAPH_COMPOSITE_CHILD,
            physical_metadata=(
                ("execution_generation", int(
                    self._component_generations[_TEXT_COMPONENT]
                )),
                ("truncated_text_layers", _SELECT_LAYER),
            ),
            plan_result_sink=lambda result: plan_box.__setitem__(
                "result", result
            ),
            plan_signature=(True, 0, 1, 4),
            graph_cache=self._text._rpu_text_graph_cache,
        )
        return execution_len, chunk_size, plan_box["result"]

    def _validate_forward_execution(self, inputs):
        requested = self._vision_execution.get("vision", {}).get(
            "chunk_size"
        )
        grid = inputs.get("image_grid_thw")
        if not isinstance(grid, torch.Tensor) or grid.ndim != 2 or grid.shape[1] != 3:
            raise ValueError(
                "GR00T vision requires image_grid_thw with shape [N, 3]"
            )
        patches = [int(x) for x in grid.detach().cpu().prod(-1).tolist()]
        if not patches:
            raise ValueError(
                "GR00T vision requires at least one image grid"
            )
        if isinstance(requested, int) and any(
            rows != _VISION_PATCHES_PER_IMAGE for rows in patches
        ):
            raise ValueError(
                "GR00T exact vision chunk profiles are certified only for "
                f"{_VISION_PATCHES_PER_IMAGE} patches per image; got {patches}. "
                "Use 'auto' for another image geometry."
            )
        if requested == 144 and self._vision_fused_merger:
            raise ValueError(
                "GR00T vision.chunk_size=144 is incompatible with the fused "
                "merger, which requires one full-sequence chunk; choose 256 "
                "for the standard 256-patch image or disable "
                "RPU_QWEN3VL_VISION_FUSED_MERGER before building"
            )
        if requested in (512, 768):
            images_per_dispatch = requested // _VISION_PATCHES_PER_IMAGE
            # Native minibatching combines only consecutive identical full
            # grids. Equal patch counts are insufficient: RoPE and window
            # layout also depend on the full T/H/W grid. Prove grouping before
            # patch_embed or any other RPU mutation.
            grid_cpu = grid.detach().cpu()
            run_lengths = []
            run = 1
            for index in range(1, int(grid_cpu.shape[0])):
                if torch.equal(grid_cpu[index], grid_cpu[index - 1]):
                    run += 1
                else:
                    run_lengths.append(run)
                    run = 1
            run_lengths.append(run)
            if any(length % images_per_dispatch for length in run_lengths):
                raise ValueError(
                    "GR00T exact vision chunk_size="
                    f"{requested} requires consecutive identical-grid runs "
                    f"divisible by {images_per_dispatch}; got run lengths "
                    f"{run_lengths}. Use 'auto' or a realizable exact chunk."
                )

    def _validate_qwen_z1_inputs(self, inputs) -> None:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        grid = inputs["image_grid_thw"]
        if tuple(input_ids.shape) != (1, _QWEN_Z1_REAL_LEN):
            raise ValueError(
                "GR00T Qwen3-VL SPM Z1 accepts only input_ids [1,82]; "
                f"got {tuple(input_ids.shape)}"
            )
        if tuple(attention_mask.shape) != (1, _QWEN_Z1_REAL_LEN):
            raise ValueError(
                "GR00T Qwen3-VL SPM Z1 accepts only attention_mask [1,82]"
            )
        if not bool(torch.all(attention_mask.detach().cpu() == 1)):
            raise ValueError(
                "GR00T Qwen3-VL SPM Z1 requires an all-valid causal prefix"
            )
        if tuple(pixel_values.shape) != (_QWEN_Z1_NUM_PATCHES, 1536):
            raise ValueError(
                "GR00T Qwen3-VL SPM Z1 accepts only pixel_values [256,1536]; "
                f"got {tuple(pixel_values.shape)}"
            )
        grid_cpu = grid.detach().to(device="cpu", dtype=torch.long)
        expected_grid = torch.tensor([[1, 16, 16]], dtype=torch.long)
        if tuple(grid_cpu.shape) != (1, 3) or not torch.equal(
            grid_cpu, expected_grid
        ):
            raise ValueError(
                "GR00T Qwen3-VL SPM Z1 accepts only grid_thw=[[1,16,16]]"
            )
        ids_cpu = input_ids.detach().to(device="cpu")
        image_rows = (ids_cpu[0] == self._img_tok).nonzero().flatten()
        expected_rows = torch.arange(
            _QWEN_Z1_IMAGE_BEGIN, _QWEN_Z1_IMAGE_END,
            dtype=image_rows.dtype,
        )
        if not torch.equal(image_rows, expected_rows):
            raise ValueError(
                "GR00T Qwen3-VL SPM Z1 requires the exact image-token run "
                "[4,68)"
            )

    def _qwen_z1_native_catalog_identity(self):
        return (
            torch.ops.rpu.qwen3vl_vision_kvinsert_cost_catalog_sha256(
                int(self._vision._rpu_vision_handle)),
            torch.ops.rpu.causal_decoder_kvinsert_cost_catalog_sha256(
                int(self._text._rpu_decoder_handle)),
        )

    def _refresh_qwen_z1_cost_plan(self, *, force=False, selected=None):
        """Reprepare native ownership after trusted costs change, without reload.

        The shared installer calls the original prepare_plans both before and
        after binding a native catalog. Vision's prepared descriptor must be
        rebuilt here; clearing the outer Graph alone would retain the old route.
        """
        current = self._qwen_z1_native_catalog_identity()
        if not force and selected is None and current == self._qwen_z1_catalog_identity:
            return
        session = self._execution_session
        with session._lock:
            if session._active or session._closed or session._poisoned:
                session.poison()
                raise RuntimeError("GR00T Z1 cost reprepare requires a quiescent live Session")
            if self._execution_reconfigure_journal is not None:
                self._execution_reconfigure_journal["native_reprepare_started"] = True
            try:
                self._reset_component_graphs(set(_EXECUTION_COMPONENTS))
                _unprepare_qwen3vl_spm_z1(
                    pipeline_cache=None, vision_model=self._vision, text_model=self._text,
                    plan_hash=self._qwen_z1_plan_hash,
                )
                self._qwen_z1_plan_hash = None
                self._qwen_z1_vision_stage_descriptor = ()
                self._qwen_z1_catalog_identity = None
                # Preserve the builder's A/D persistent-layout preflight order.
                # Keep any live temporary identity on the existing retirement
                # field, so a failed prepare/cleanup cannot lose its owner.
                self._gr00t_z2_plan_hash = int(torch.ops.rpu.gr00t_spm_z2_prepare(
                    int(self._a_handle), int(self._d_handle), _QWEN_Z1_REAL_LEN))
                try:
                    self._qwen_z1_plan_hash = int(torch.ops.rpu.qwen3vl_pooler_spm_z1_prepare(
                        int(self._vision._rpu_vision_handle), int(self._text._rpu_decoder_handle),
                        _QWEN_Z1_NUM_PATCHES, _QWEN_Z1_EXECUTION_LEN, _QWEN_Z1_IMAGE_BEGIN,
                        _QWEN_Z1_REAL_LEN, _QWEN_Z1_DEEPSTACK_COUNT,
                        *([] if selected is None else [list(raw) for raw in selected]),
                    ))
                    descriptor = tuple(torch.ops.rpu.qwen3vl_pooler_spm_z1_vision_stage_descriptor(
                        int(self._vision._rpu_vision_handle), int(self._text._rpu_decoder_handle),
                        self._qwen_z1_plan_hash,
                    ))
                    if not descriptor:
                        raise RuntimeError("GR00T Z1 reprepare returned an empty Vision descriptor")
                finally:
                    _unprepare_gr00t_spm_z2(
                        a_handle=self._a_handle, d_handle=self._d_handle,
                        plan_hash=self._gr00t_z2_plan_hash,
                    )
                    self._gr00t_z2_plan_hash = None
                if self._qwen_z1_native_catalog_identity() != current:
                    raise RuntimeError("GR00T Z1 native costs changed during reprepare")
                self._qwen_z1_vision_stage_descriptor = descriptor
                actual = self._qwen_z1_stage_descriptors()
                if selected is not None and actual != selected:
                    raise RuntimeError("GR00T Z1 reprepare did not consume both selected descriptors")
                self._qwen_z1_catalog_identity = current
            except BaseException:
                # Never let a failed native rebuild publish or replay a prior
                # descriptor, even if the config transaction can compensate.
                self._qwen_z1_vision_stage_descriptor = ()
                self._qwen_z1_catalog_identity = None
                self._rpu_last_execution_plan.clear()
                vars(self._vision).pop("_rpu_last_execution_plan", None)
                vars(self._text).pop("_rpu_last_execution_plan", None)
                session.poison()
                raise

    def _qwen_z1_stage_descriptors(self):
        identity = (int(self._vision._rpu_vision_handle),
                    int(self._text._rpu_decoder_handle), int(self._qwen_z1_plan_hash))
        vision_descriptor = tuple(
            torch.ops.rpu.qwen3vl_pooler_spm_z1_vision_stage_descriptor(*identity)
        )
        text_descriptor = tuple(
            torch.ops.rpu.qwen3vl_pooler_spm_z1_text_stage_descriptor(
                *identity, False,
            )
        )
        if not vision_descriptor or vision_descriptor != self._qwen_z1_vision_stage_descriptor:
            raise RuntimeError("GR00T Qwen Z1 prepared Vision descriptor changed")
        if not text_descriptor:
            raise RuntimeError("GR00T Qwen Z1 Text COMPLETE descriptor is absent")
        return vision_descriptor, text_descriptor

    def _qwen_z1_component_plans(self):
        """Resolve the actual fixed native domains under the owning Session."""
        self._refresh_qwen_z1_cost_plan()
        queue_owner = id(self._qwen_z1_gc)
        lease_owner = id(self)
        vision_descriptor, text_descriptor = self._qwen_z1_stage_descriptors()
        plans = []
        for component, stage, child, prefix, handle, logical, execution, descriptor, config in (
            (_VISION_COMPONENT, "vision", self._vision, "qwen3vl_vision",
             self._vision._rpu_vision_handle, 256, 256,
             vision_descriptor, self._vision_execution),
            (_TEXT_COMPONENT, "prefill", self._text, "causal_decoder",
             self._text._rpu_decoder_handle, 82, 128, text_descriptor, self._text_execution),
        ):
            requested = config[stage]["chunk_size"]
            plan_bounded_prefill_execution(
                logical, execution, 0, padding_rows=execution - logical,
                exact_chunk_size=None if requested == "auto" else requested,
                resolve_stage_domain=lambda _length, raw=descriptor: [1, 1, len(raw), *raw],
                graph_mode=GRAPH_COMPOSITE_CHILD,
                execution_owner=child, execution_component=component,
                execution_stage=stage, execution_native=(prefix, int(handle)),
                physical_metadata=(
                    (f"component:{component}", 1),
                    ("execution_generation", self._component_generations[component]),
                    ("image_span_begin", _QWEN_Z1_IMAGE_BEGIN),
                    ("image_span_end", _QWEN_Z1_IMAGE_END),
                ),
                queue_owner_id=queue_owner, lease_owner_id=lease_owner,
                plan_result_sink=plans.append,
                plan_signature=(tuple(descriptor),),
                graph_cache=self._qwen_z1_gc,
            )
        selected = tuple(plan.selected.stage_tuple.physical_descriptor for plan in plans)
        if selected != (vision_descriptor, text_descriptor):
            self._refresh_qwen_z1_cost_plan(selected=selected)
        return tuple(plans)

    def _qwen_z1_selected_owner(self):
        rows = tuple(tuple(row) for row in
            torch.ops.rpu.qwen3vl_pooler_spm_z1_selected_owner(
                int(self._vision._rpu_vision_handle),
                int(self._text._rpu_decoder_handle), int(self._qwen_z1_plan_hash)))
        if len(rows) != 7 or rows[0] != (1, 1, 1):
            raise RuntimeError("GR00T Z1 native selected owner is incomplete")
        descriptor = rows[1]
        if (len(descriptor) < 24 or descriptor[:10] !=
                (1, 1, 3, 256, 128, 4, 82, 64, 2048, 3)
                or descriptor[10:12] != (int(self._vision._rpu_vision_handle),
                                         int(self._text._rpu_decoder_handle))
                or descriptor[21] != int(self._qwen_z1_plan_hash)
                or descriptor[22:24] != (len(rows[5]), len(rows[6]))
                or descriptor[24:] != rows[5] + rows[6]
                or len(rows[2]) != 2 or any(value <= 0 for value in rows[2])
                or any(len(profile) <= 11 or profile[10] <= 0 for profile in rows[3:5])):
            raise RuntimeError("GR00T Z1 native selected owner identity drift")
        return rows

    def _build_qwen_z1_composite_plan(self, native, child_plans):
        """Bind the fixed native geometry to the two selected child plans."""
        descriptor = tuple(native[1])
        children = tuple(child_plans)
        if len(children) != 2 or tuple(
            child.selected.stage_tuple.physical_descriptor for child in children
        ) != tuple(native[5:7]):
            raise RuntimeError("GR00T Z1 parent/child selected authority disagreement")
        component_generations = tuple(self._component_generations[component]
                                      for component in (_VISION_COMPONENT, _TEXT_COMPONENT))
        # Keep one owner-local result. The native rows are read and validated
        # by the caller on every execution; immutable child plans and both
        # execution generations must still match before reusing the parent.
        cached = getattr(self, "_qwen_z1_composite_plan_cache", None)
        if (cached is not None and native == cached[0]
                and all(child is previous for child, previous in zip(children, cached[1]))
                and component_generations == cached[2]):
            return cached[3].copy()
        if any(dict(child.selected.stage_tuple.physical_metadata).get("execution_generation")
               != generation for child, generation in zip(children, component_generations)):
            raise RuntimeError("GR00T Z1 child planner execution generation drift")
        digest_words = []
        for child in children:
            raw = bytes.fromhex(child.plan_digest)
            if len(raw) != 32:
                raise RuntimeError("GR00T Z1 child planner digest is incomplete")
            digest_words.extend(int.from_bytes(raw[offset:offset + 8], "big", signed=True)
                                for offset in range(0, 32, 8))
        descriptor += tuple(digest_words)
        metadata = [(f"descriptor_{index:03d}", word)
                    for index, word in enumerate(descriptor)]
        metadata.extend((f"child_{index}_installed_generation", generation)
                        for index, generation in enumerate(native[2]))
        # The coordinator owns exactly the native Vision -> Text route. Its
        # parent queue/lease identity is the installed Text endpoint handle;
        # the independent downstream A/D graphs are not children of this graph.
        parent = capability_selected(plan_a6(
            descriptor[4], tuple_domain={descriptor[4]: (StageTuple(
                descriptor[4], descriptor[4], descriptor[4], tuple(metadata), descriptor),)},
            physical_limit=descriptor[4], graph_mode=GRAPH_NATIVE_COMPOSITE1,
            queue_owner_id=descriptor[11], lease_owner_id=descriptor[11],
        ), selection_scope="NATIVE_COMPOSITE1_ORDERED_CHILDREN")
        raw = bytes.fromhex(parent.plan_digest)
        words = tuple(int.from_bytes(raw[offset:offset + 8], "big", signed=True)
                      for offset in range(0, 32, 8))
        self._qwen_z1_planner_parent = parent
        self._qwen_z1_planner_children = children
        result = {"planner_parent": parent, "planner_descriptor": descriptor,
                  "planner_digest_words": words,
                  "planner_component_generations": component_generations}
        self._qwen_z1_composite_plan_cache = (native, children, component_generations, result)
        return result.copy()

    def _qwen_z1_graph_identity(self, graph):
        if graph is None or graph.state() != 2 or not graph.replayable():
            raise RuntimeError("GR00T Z1 requires a completed replayable Graph")
        stats = graph.debug_stats()
        entries = tuple(entry for entry in self._qwen_z1_gc.snapshot()
                        if entry.graph_lifetime_id == stats.graph_lifetime_id)
        if len(entries) != 1 or any(getattr(entries[0], name) != getattr(stats, name)
                                   for name in ("build_generation", "execution_ordinal",
                                                "kernel_count", "segment_count")):
            raise RuntimeError("GR00T Z1 cache/Graph execution identity disagreement")
        values = (int(stats.graph_lifetime_id), int(stats.build_generation),
                  int(stats.execution_ordinal), int(entries[0].replay_count),
                  int(entries[0].recapture_count), int(stats.kernel_count),
                  int(stats.segment_count))
        if (any(value <= 0 for value in values[:3]) or values[4] != 0
                or values[5] <= 0 or values[6] <= 0
                or stats.non_replayable_reason):
            raise RuntimeError("GR00T Z1 Graph execution identity is incomplete")
        return values

    def _qwen_z1_check_completed_owner(self, retained):
        native, planned, execution, signature, generation = retained
        if (generation != self._execution_session.generation
                or planned["planner_component_generations"] != tuple(
                    self._component_generations[component]
                    for component in (_VISION_COMPONENT, _TEXT_COMPONENT))
                or native != self._qwen_z1_selected_owner()
                or self._qwen_z1_planner_parent is not planned["planner_parent"]
                or any(not cache.is_frozen() or not cache.cache_invariant_ok()
                       for cache in (self._qwen_z1_gc, self._a_gc, self._d_gc))
                or execution[2:] != self._qwen_z1_graph_identity(
                    self._qwen_z1_gc.lookup(signature))):
            raise RuntimeError("GR00T Z1 completed owner authority drift")
        return (native[0], planned["planner_descriptor"], planned["planner_digest_words"],
                native[2], native[3], native[4], execution, native[5], native[6])

    def _qwen_z1_recorded_ranges(self, graph):
        """Check actual native BUILD windows against their completed live Graph."""
        rows = tuple(tuple(row) for row in
            torch.ops.rpu.qwen3vl_pooler_spm_z1_recorded_ranges(
                int(self._vision._rpu_vision_handle),
                int(self._text._rpu_decoder_handle), int(self._qwen_z1_plan_hash)))
        if (len(rows) != 5 or rows[0] != (1, 3) or len(rows[1]) != 9
                or any(type(word) is not int or not -(1 << 63) <= word < (1 << 63)
                       for row in rows for word in row)):
            raise RuntimeError("GR00T Z1 recorded ranges are incomplete")
        identity = self._qwen_z1_graph_identity(graph)
        context = rows[1]
        unsigned = lambda word: word & ((1 << 64) - 1)
        if (context[:3] != (int(self._vision._rpu_vision_handle),
                            int(self._text._rpu_decoder_handle), int(self._qwen_z1_plan_hash))
                or tuple(map(unsigned, context[3:5])) != identity[:2]
                or unsigned(context[5]) != graph.built_signature_identity()
                or unsigned(context[6]) != graph.built_signature().segment_key
                or context[5] == 0 or context[7] == 0
                or unsigned(context[7]) != graph.build_topology_hash()
                or context[8] != graph.graph_size()):
            raise RuntimeError("GR00T Z1 recorded ranges changed Graph identity/topology")
        kinds = []
        for phase, row in enumerate(rows[2:], 1):
            if (len(row) < 9 or row[0] != phase or row[1] != len(kinds)
                    or row[2] <= row[1] or len(row[8:]) != row[2] - row[1]
                    or row[4] == 0 or any(kind not in (0, 8, 9) for kind in row[8:])
                    or row[3] != sum(1 << kind for kind in set(row[8:]))
                    or row[5:8] != tuple(row[8:].count(kind) for kind in (0, 8, 9))):
                raise RuntimeError("GR00T Z1 recorded component range is malformed")
            kinds.extend(row[8:])
        stats = graph.debug_stats()
        if (len(kinds) != context[8] or tuple(kinds.count(kind) for kind in (0, 8, 9))
                != (stats.kernel_count, stats.dma_count, stats.barrier_count)):
            raise RuntimeError("GR00T Z1 ranges do not cover the actual Graph")
        # SDK perf_batch_idx advances for Kernel/Dma, not Barrier. Keep the
        # actual node sequence and segment boundaries; node index is not a
        # packet index, and no historical kernel window assigns ownership.
        segments = tuple(stats.segment_census)
        cursor = 0
        for index, segment in enumerate(segments):
            if (segment.segment_id != index or segment.start_idx != cursor
                    or not cursor < segment.end_idx <= len(kinds)
                    or tuple(kinds[cursor:segment.end_idx].count(kind) for kind in (0, 8, 9))
                    != (segment.kernel_count, segment.dma_count, segment.barrier_count)):
                raise RuntimeError("GR00T Z1 ranges disagree with actual Graph segments")
            cursor = segment.end_idx
        if len(segments) != stats.segment_count or cursor != len(kinds):
            raise RuntimeError("GR00T Z1 ranges lack complete Graph segment coverage")
        return rows

    def _qwen_z1_check_completed_ranges(self, retained, ranges):
        if ranges is None or ranges != self._qwen_z1_recorded_ranges(
                self._qwen_z1_gc.lookup(retained[3])):
            raise RuntimeError("GR00T Z1 completed native ranges changed")
        return (ranges[0], ranges[1], retained[1]["planner_digest_words"], *ranges[2:])

    @property
    def composite_execution_plan(self):
        parent = getattr(self, "_qwen_z1_planner_parent", None)
        if parent is None:
            return None
        return {"parent": parent.as_dict(include_candidates=False),
                "descriptor": parent.selected.stage_tuple.physical_descriptor,
                "children": tuple({**child.as_dict(include_candidates=False),
                                   "component": component, "stage": stage}
                                  for child, component, stage in zip(
                                      self._qwen_z1_planner_children,
                                      (_VISION_COMPONENT, _TEXT_COMPONENT),
                                      ("vision", "prefill"), strict=True))}

    @property
    def composite_executed_owner(self):
        """Observe one successful action through its actual retained Graph."""
        with self._execution_session._lock:
            try:
                session = self._execution_session
                retained = getattr(self, "_qwen_z1_completed_owner", None)
                if (self._closed or session._closed or session._poisoned or session._active
                        or self._qwen_z1_plan_hash is None or retained is None):
                    raise RuntimeError("GR00T Z1 executed owner requires a successful idle action")
                return self._qwen_z1_check_completed_owner(retained)
            except BaseException:
                vars(self).pop("_qwen_z1_completed_owner", None)
                vars(self).pop("_qwen_z1_completed_ranges", None)
                raise

    @property
    def composite_executed_ranges(self):
        """Stable BUILD scope bound to the same successfully completed action."""
        with self._execution_session._lock:
            try:
                # Re-read selected native authority and actual L/B/ordinal under
                # the same execution lock before exposing retained BUILD ranges.
                self.composite_executed_owner
                return self._qwen_z1_check_completed_ranges(
                    self._qwen_z1_completed_owner,
                    getattr(self, "_qwen_z1_completed_ranges", None))
            except BaseException:
                vars(self).pop("_qwen_z1_completed_owner", None)
                vars(self).pop("_qwen_z1_completed_ranges", None)
                raise

    @torch.no_grad()
    def _backbone_qwen_z1(self, inputs):
        self._validate_qwen_z1_inputs(inputs)
        if self._qwen_z1_plan_hash is None or self._qwen_z1_gc is None:
            raise RuntimeError("GR00T Qwen3-VL SPM Z1 was not prepared")
        vision_plan, text_plan = self._qwen_z1_component_plans()
        native_owner = self._qwen_z1_selected_owner()
        parent_plan = self._build_qwen_z1_composite_plan(
            native_owner, (vision_plan, text_plan))

        m, cfg = self._bb, self._cfg
        text_model, vision_model = self._text, self._vision
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        grid = inputs["image_grid_thw"].to(torch.long)
        vision_input = _prepare_qwen3vl_vision_input_for_spm_pipeline(
            vision_model, inputs["pixel_values"].to(torch.float16), grid
        )

        prompt_changed = (
            self._emb_key is None or not torch.equal(input_ids, self._emb_key)
        )
        if prompt_changed:
            text_real = m.get_input_embeddings()(input_ids.to("rpu")).to(
                torch.float16
            ).contiguous()
            carrier = text_real.new_zeros(
                1, _QWEN_Z1_EXECUTION_LEN, cfg.text_config.hidden_size
            )
            carrier[:, :_QWEN_Z1_REAL_LEN, :].copy_(text_real)
            carrier[:, _QWEN_Z1_IMAGE_BEGIN:_QWEN_Z1_IMAGE_END, :].zero_()
            self._qwen_z1_text_hidden = carrier.contiguous()
            self._emb_key = input_ids.detach().clone()

        rope_changed = (
            self._rope_ids is None
            or not torch.equal(input_ids, self._rope_key)
            or not torch.equal(grid, self._rope_grid_key)
            or not torch.equal(attention_mask, self._rope_attn_key)
        )
        if rope_changed:
            mmtt = (input_ids == self._img_tok).to(torch.int32)
            self._rope_ids, _ = _get_gr00t_rope_index(
                m.model.get_rope_index, input_ids=input_ids,
                mm_token_type_ids=mmtt.cpu(),
                image_grid_thw=grid.cpu(),
                attention_mask=attention_mask.cpu(),
            )
            self._rope_key = input_ids.detach().clone()
            self._rope_grid_key = grid.detach().clone()
            self._rope_attn_key = attention_mask.detach().clone()
            pad_rows = _QWEN_Z1_EXECUTION_LEN - _QWEN_Z1_REAL_LEN
            padded = torch.cat(
                [
                    self._rope_ids,
                    self._rope_ids[..., -1:].expand(3, 1, pad_rows),
                ],
                dim=-1,
            )
            self._qwen_z1_position_ids = (
                padded[:, 0, :].transpose(0, 1)
                .to(device="rpu", dtype=torch.int32)
                .contiguous()
            )

        if self._qwen_z1_text_hidden is None or self._qwen_z1_position_ids is None:
            raise RuntimeError("GR00T Qwen3-VL SPM Z1 prompt state is incomplete")

        vision_cache = vision_model._rpu_vision_kv_cache
        text_cache = self._bb_cache
        vision_cache.reset_to_position(0)
        text_cache.reset_to_position(0)
        vision_k = [vision_cache.k_caches[i] for i in range(24)]
        vision_v = [vision_cache.v_caches[i] for i in range(24)]
        text_k = [text_cache.k_caches[i] for i in range(_SELECT_LAYER)]
        text_v = [text_cache.v_caches[i] for i in range(_SELECT_LAYER)]
        vision_handle = int(vision_model._rpu_vision_handle)
        text_handle = int(text_model._rpu_decoder_handle)
        plan_hash = int(self._qwen_z1_plan_hash)

        signature = rpu_backend.graph.GraphSignature(
            op_id="gr00t_qwen3vl_pooler_spm_z1_pipeline",
            shapes=[256, 1024, 64, 2048, 128],
            dyn_dims=[
                24, _SELECT_LAYER, 4, 68, _QWEN_Z1_REAL_LEN,
                _QWEN_Z1_DEEPSTACK_COUNT,
                5, 0, 11, 1, 17, 2, plan_hash,
                *vision_plan.graph_key_words(),
                *text_plan.graph_key_words(),
                *parent_plan["planner_parent"].graph_key_words(),
            ],
            dtypes=[torch.float16],
        )

        _assert_qwen3vl_component_caches_empty(vision_model, text_model)
        _preflight_ready_graph(
            self._qwen_z1_gc, signature,
            "GR00T Qwen3-VL SPM Z1 preflight",
        )
        epoch = None
        try:
            epoch = int(torch.ops.rpu.qwen3vl_pooler_spm_z1_begin(
                vision_handle,
                text_handle,
                _QWEN_Z1_NUM_PATCHES,
                _QWEN_Z1_EXECUTION_LEN,
                _QWEN_Z1_IMAGE_BEGIN,
                _QWEN_Z1_REAL_LEN,
                self._qwen_z1_position_ids,
                plan_hash,
            ))
            with self._qwen_z1_gc.capture(signature) as executing_graph:
                before = executing_graph.debug_stats()
                graph_before = (int(before.graph_lifetime_id), int(before.build_generation),
                                int(before.execution_ordinal))
                raw = torch.ops.rpu.qwen3vl_pooler_spm_z1_pipeline_forward(
                    vision_handle,
                    text_handle,
                    vision_input,
                    vision_k,
                    vision_v,
                    self._qwen_z1_text_hidden,
                    text_k,
                    text_v,
                    self._qwen_z1_position_ids,
                    epoch,
                    plan_hash,
                )
        except BaseException as forward_error:
            if epoch is not None:
                failed_epoch = epoch
                epoch = None
                try:
                    # GraphCaptureScope.__exit__ may fail after the composite
                    # body consumed its invocation.  Let the coordinator read
                    # the authoritative generation under its lock instead of
                    # guessing from Python control flow.
                    torch.ops.rpu.qwen3vl_pooler_spm_z1_cancel(
                        failed_epoch, plan_hash
                    )
                except BaseException as cleanup_error:
                    raise cleanup_error from forward_error
            raise
        else:
            if epoch is not None:
                completed_epoch = epoch
                epoch = None
                torch.ops.rpu.qwen3vl_pooler_spm_z1_end(
                    completed_epoch, plan_hash
                )

        graph_after = self._qwen_z1_graph_identity(executing_graph)
        if (graph_after[:2] != graph_before[:2]
                or graph_after[2] != graph_before[2] + 1
                or native_owner != self._qwen_z1_selected_owner()):
            raise RuntimeError("GR00T Z1 Graph did not execute the selected native owner")
        recorded_ranges = self._qwen_z1_recorded_ranges(executing_graph)

        if tuple(raw.shape) != (1, _QWEN_Z1_EXECUTION_LEN, 2048):
            raise RuntimeError(
                "GR00T Qwen3-VL SPM Z1 returned an unexpected Text shape "
                f"{tuple(raw.shape)}"
            )
        # Z1 consumes the descriptors sealed at prepare, not forward arguments.
        # Its identity-checked getters must still equal both actual winners
        # after the successful end, before either child publishes authority.
        vision_descriptor, text_descriptor = self._qwen_z1_stage_descriptors()
        if (vision_descriptor, text_descriptor) != (
            vision_plan.selected.stage_tuple.physical_descriptor,
            text_plan.selected.stage_tuple.physical_descriptor,
        ):
            raise RuntimeError("GR00T Qwen Z1 native dry/forward descriptor disagreement")
        vision_receipt = self._record_component_plan(
            _VISION_COMPONENT,
            "vision",
            vision_plan,
            logical_len=_QWEN_Z1_NUM_PATCHES,
            resolved_chunk_size=_QWEN_Z1_NUM_PATCHES,
            physical_descriptor=vision_descriptor,
        )
        text_receipt = self._record_component_plan(
            _TEXT_COMPONENT,
            "prefill",
            text_plan,
            logical_len=_QWEN_Z1_REAL_LEN,
            resolved_chunk_size=_QWEN_Z1_EXECUTION_LEN,
            physical_descriptor=text_descriptor,
        )
        vars(vision_model)["_rpu_last_execution_plan"] = vision_receipt
        vars(text_model)["_rpu_last_execution_plan"] = text_receipt
        _assert_qwen3vl_component_caches_empty(vision_model, text_model)
        invocation = getattr(self, "_qwen_z1_invocation_ordinal", 0) + 1
        self._qwen_z1_invocation_ordinal = invocation
        self._qwen_z1_pending_owner = (
            native_owner, parent_plan, (completed_epoch, invocation, *graph_after),
            signature, self._execution_session.generation)
        self._qwen_z1_pending_ranges = recorded_ranges
        return raw[:, :_QWEN_Z1_REAL_LEN, :], _QWEN_Z1_REAL_LEN

    @torch.no_grad()
    def _backbone(self, inputs):
        if self._qwen_z1_plan_hash is not None:
            return self._backbone_qwen_z1(inputs)
        m, cfg, text_model, vision_model = self._bb, self._cfg, self._text, self._vision
        input_ids = inputs["input_ids"]; attn = inputs["attention_mask"]
        logical_seq_len = int(input_ids.shape[1])
        pix = inputs["pixel_values"].to(torch.float16); grid = inputs["image_grid_thw"].to(torch.long)
        img_tok = m.config.image_token_id
        vout = vision_model(pix, grid_thw=grid)
        vision_receipt = getattr(
            vision_model, "_rpu_last_execution_plan", None
        )
        if (
            not isinstance(vision_receipt, dict)
            or vision_receipt.get("component") != _VISION_COMPONENT
            or vision_receipt.get("generation")
            != self._component_generations[_VISION_COMPONENT]
            or not vision_receipt.get("dry_forward_agreement")
        ):
            raise RuntimeError(
                "GR00T Vision forward did not publish the current component "
                "A6 receipt"
            )
        self._rpu_last_execution_plan[_VISION_COMPONENT] = dict(
            vision_receipt
        )
        # vout.pooler_output is already [sum(n_i/sm²), oh] in image order — the old
        # torch.cat(torch.split(pooler_output, per_image)) reconstructed the SAME tensor
        # (identity), so drop the redundant per-forward split + cat.
        visual_flat = vout.pooler_output
        # Opt-a: on-device visual inject. inputs_embeds lives on RPU (memoized: the base text
        # embedding + the contiguous image-token runs, 1 per image). Each forward overwrites only
        # the image slices with the visual embeds on-device (narrow+copy_ = RPU DMA) — the image
        # positions are overwrite-before-read, non-image positions are constant, so no clone is
        # needed and the buffer address stays pinned across REPLAY. Kills the CPU masked_scatter +
        # the RPU↔CPU bounces (the vision→backbone glue folds into device copies).
        if self._emb_text_rpu is None or not torch.equal(input_ids, self._emb_key):
            # These owners are refreshed in place on every request. Cold
            # allocation must not inherit the first request's inference mode.
            with torch.inference_mode(False), torch.no_grad():
                self._emb_text_rpu = m.get_input_embeddings()(input_ids.to("rpu")).to(
                    torch.float16).contiguous()                                    # [1,S,H] RPU
                ipos = (input_ids.reshape(-1) == img_tok).nonzero().reshape(-1).tolist()
                runs = []                                                          # contiguous (start,len) per image
                if ipos:
                    s = p = ipos[0]
                    for j in ipos[1:]:
                        if j == p + 1: p = j
                        else: runs.append((s, p - s + 1)); s = p = j
                    runs.append((s, p - s + 1))
                self._img_runs = runs
                self._emb_key = input_ids.clone()  # snapshot: catch in-place mutation of a reused input_ids
                # Opt #2: one persistent RPU zero buffer per deepstack merger. Image rows are
                # overwritten in-place each forward (same contiguous runs as the visual inject);
                # non-image rows stay zero forever (overwrite-before-read). Replaces the per-forward
                # torch.zeros + CPU index_put scatter (see the on-device dense build below).
                self._ds_dense = [torch.zeros((input_ids.shape[1], cfg.text_config.hidden_size),
                                              dtype=torch.float16, device="rpu").contiguous()
                                  for _ in range(len(vout.deepstack_features))]
        visual_rpu = visual_flat.to(device="rpu", dtype=torch.float16).contiguous()
        voff = 0
        for (start, length) in self._img_runs:
            self._emb_text_rpu[0, start:start + length, :].copy_(visual_rpu[voff:voff + length])
            voff += length
        inputs_embeds = self._emb_text_rpu
        seq_len = inputs_embeds.shape[1]
        # Opt3: get_rope_index is CPU M-RoPE index math, constant per prompt → memoize on input_ids.
        # get_rope_index depends on input_ids AND grid AND attention_mask — key on all three
        # (cloned snapshots, so a reused/in-place-mutated input tensor is not a false cache hit).
        if (self._rope_ids is None
                or not torch.equal(input_ids, self._rope_key)
                or not torch.equal(grid, self._rope_grid_key)
                or not torch.equal(attn, self._rope_attn_key)):
            mmtt = (input_ids == img_tok).to(torch.int32)
            self._rope_ids, _ = _get_gr00t_rope_index(m.model.get_rope_index, input_ids=input_ids, mm_token_type_ids=mmtt,
                                                       image_grid_thw=grid,
                                                       attention_mask=attn.cpu())
            self._rope_key = input_ids.clone()
            self._rope_grid_key = grid.clone()
            self._rope_attn_key = attn.clone()
            # Strategy-2: position_ids (= _rope_ids) and attention_mask are per-prompt CONSTANTS
            # fed to the backbone op every forward — memoize their RPU copies here (alongside
            # _rope_ids) so the warm loop skips the per-forward CPU→RPU copy AND the backbone
            # REPLAY reads a pinned address (same rationale as the denoise masks memo).
            self._pos_rpu = self._rope_ids.to("rpu").contiguous()
            self._attn_rpu = attn.to("rpu").contiguous()
            # partial_mrope: bake the interleaved cos/sin from the (constant-per-prompt) M-RoPE
            # position_ids once, here in the memoize block. get_rope_index → [3,1,S]; the helper
            # wants [S,3]. None when the gate is off → _run_causal_decoder_forward keeps legacy.
            if self._partial_mrope:
                pos23 = self._rope_ids[:, 0, :].transpose(0, 1).contiguous()        # [S,3]
                cos_il, sin_il = build_interleaved_mrope_cos_sin(
                    pos23.to(torch.int64), head_dim=self._mrope_hd,
                    rope_theta=self._mrope_theta, mrope_section=self._mrope_section)
                self._rope_cos_il = cos_il.to("rpu").contiguous()
                self._rope_sin_il = sin_il.to("rpu").contiguous()
        # Opt #2: on-device deepstack scatter. The deepstack features land at the same
        # contiguous image-token runs as the visual inject (opt-a), so the CPU index_put
        # `dense[mask]=feat` (×3/fwd) becomes pinned RPU narrow/copy_ into the persistent zero
        # buffers (non-image rows stay zero). Numerically identical to scatter_visual_embeds_to_dense;
        # kills the 3 CPU fallbacks + their RPU↔CPU bounces. Held member → mutable-DMA-safe.
        for i, feat in enumerate(vout.deepstack_features):
            feat_r = feat.to(device="rpu", dtype=torch.float16).contiguous()
            voff = 0
            for (start, length) in self._img_runs:
                self._ds_dense[i][start:start + length, :].copy_(feat_r[voff:voff + length])
                voff += length
        dense = self._ds_dense
        # The native dry planner accounts for currently available SPM.  Resolve
        # after Vision and its persistent glue have materialized, immediately
        # before padding/signature/forward consume the plan.
        execution_seq_len, planned_chunk_size, prefill_plan_result = (
            self._prefill_execution_plan(logical_seq_len)
        )
        (
            inputs_embeds,
            attn_rpu,
            pos_rpu,
            dense,
            rope_cos_il,
            rope_sin_il,
        ) = _pad_gr00t_prefill_inputs(
            inputs_embeds,
            self._attn_rpu,
            self._pos_rpu,
            dense,
            self._rope_cos_il,
            self._rope_sin_il,
            execution_seq_len,
        )
        cache = self._bb_cache                       # Opt1: reuse hoisted cache
        cache.reset_to_position(0)                    # prefill writes [0,S); overwrite-before-read
        sig = rpu_backend.graph.GraphSignature(op_id="gr00t_backbone",
                                         shapes=[seq_len, execution_seq_len,
                                                 text_model._rpu_text_hidden_size],
                                         dyn_dims=[text_model._rpu_text_num_layers,
                                                   text_model._rpu_text_deepstack_hash, 0,
                                                   chunk_policy_key(text_model._rpu_decoder_handle),
                                                   int(planned_chunk_size),
                                                   int(self._component_generations[
                                                       _TEXT_COMPONENT
                                                   ]),
                                                   *prefill_plan_result.graph_key_words()],
                                         dtypes=[torch.float16])
        with text_model._rpu_text_graph_cache.capture(sig):
            raw, _ = _run_causal_decoder_forward(
                text_model, text_model._rpu_decoder_handle, input_ids=None, inputs_embeds=inputs_embeds,
                attention_mask=attn_rpu, position_ids=pos_rpu,
                past_key_values=cache, use_cache=True, return_dict=True, deepstack_dense_visual_embeds=dense,
                rope_cos_il=rope_cos_il, rope_sin_il=rope_sin_il,
                prefill_plan=(execution_seq_len, planned_chunk_size,
                              prefill_plan_result))
        # Native exact-descriptor validation and the shared child receipt
        # already establish the dispatched capacity for this invocation.
        resolved_chunk_size = int(prefill_plan_result.selected.stage_tuple.compute_chunk)
        native_text_receipt = getattr(
            text_model, "_rpu_last_execution_plan", None
        )
        if (
            not isinstance(native_text_receipt, dict)
            or native_text_receipt.get("logical_len") != seq_len
            or native_text_receipt.get("execution_len") != execution_seq_len
            or native_text_receipt.get("chunk_size") != resolved_chunk_size
            or native_text_receipt.get("position") != 0
            or native_text_receipt.get("physical_descriptor")
            != prefill_plan_result.selected.stage_tuple.physical_descriptor
        ):
            raise RuntimeError(
                "GR00T text forward did not retain its dispatched physical "
                "descriptor"
            )
        raw = _restore_gr00t_logical_prefill(
            raw, cache, seq_len, execution_seq_len
        )
        text_receipt = self._record_component_plan(
            _TEXT_COMPONENT,
            "prefill",
            prefill_plan_result,
            logical_len=seq_len,
            resolved_chunk_size=resolved_chunk_size,
            physical_descriptor=native_text_receipt["physical_descriptor"],
        )
        vars(text_model)["_rpu_last_execution_plan"] = text_receipt
        return raw, seq_len

    # ---- A: gr00t_vl_encoder (vlln + 4-layer vl_self_attention) -> vl_embeds [1,S,2048] ----
    def _vl_execution_plan(self, seq_len):
        stage = self._vl_execution.get("prefill", {})
        exact = stage.get("chunk_size")
        plan_box = {}
        execution_len, chunk_size = plan_bounded_prefill_execution(
            int(seq_len),
            int(seq_len),
            0,
            execution_owner=self,
            execution_component=_VL_COMPONENT,
            execution_stage="prefill",
            execution_native=("gr00t_vl_encoder", int(self._a_handle)),
            queue_owner_id=int(self._a_handle),
            resolve_stage_domain=lambda length: (
                torch.ops.rpu.gr00t_vl_encoder_resolve_stage_domain(
                    int(self._a_handle), int(length)
                )
            ),
            position=0,
            alignment=1,
            padding_rows=0,
            exact_chunk_size=exact if isinstance(exact, int) else None,
            request_id="gr00t:vl_encoder:prefill",
            graph_mode=GRAPH_COMPOSITE_CHILD,
            physical_metadata=(
                ("execution_generation", int(
                    self._component_generations[_VL_COMPONENT]
                )),
                ("single_full_sequence_chunk", 1),
            ),
            plan_result_sink=lambda result: plan_box.__setitem__(
                "result", result
            ),
            plan_signature=(),
            graph_cache=self._a_gc,
        )
        result = plan_box["result"]
        if (
            execution_len != int(seq_len)
            or result.selected is None
            or not result.selected.stage_tuple.physical_descriptor
        ):
            raise RuntimeError(
                "GR00T VL encoder planner returned no consumable native "
                "stage descriptor"
            )
        return int(chunk_size), result

    @torch.no_grad()
    def _vl_encode(self, raw, S, execution_plan=None):
        raw_r = raw.to(torch.float16).to("rpu").contiguous()          # HELD (A input; post-RMSNorm)
        chunk_size, plan = (
            self._vl_execution_plan(S)
            if execution_plan is None else execution_plan
        )
        stage_descriptor = tuple(
            plan.selected.stage_tuple.physical_descriptor
        )
        a_sig = rpu_backend.graph.GraphSignature(op_id="gr00t_vl_encoder", shapes=[S, _VLD],
                                           dyn_dims=[
                                               _VL_NL,
                                               self._component_generations[
                                                   _VL_COMPONENT
                                               ],
                                               *plan.graph_key_words(),
                                           ], dtypes=[torch.float16])
        with self._a_gc.capture(a_sig):
            vl = torch.ops.rpu.gr00t_vl_encoder_forward(
                self._a_handle, raw_r, self._a_cache.k_caches,
                self._a_cache.v_caches,
                stage_descriptor,
            )
        resolved = int(
            torch.ops.rpu.gr00t_vl_encoder_get_resolved_chunk_size(
                int(self._a_handle)
            )
        )
        if resolved != chunk_size:
            raise RuntimeError(
                "GR00T VL encoder dry/forward chunk plan drift: "
                f"dry={chunk_size}, forward={resolved}, seq_len={S}"
            )
        self._record_component_plan(
            _VL_COMPONENT,
            "prefill",
            plan,
            logical_len=S,
            resolved_chunk_size=resolved,
            physical_descriptor=stage_descriptor,
        )
        return vl.to(torch.float16).contiguous()                      # HELD (denoise input)

    # ---- C: gr00t_denoise_unroll — encoders + 32-block DiT + norm_out + decoder + Euler ----
    def _action_execution_plan(self, cross_seq_len):
        stage = self._action_execution.get("action", {})
        exact = stage.get("chunk_size")
        plan_box = {}
        execution_len, chunk_size = plan_bounded_prefill_execution(
            _ACTION_QUERY_ROWS,
            _ACTION_QUERY_ROWS,
            0,
            execution_owner=self,
            execution_component=_ACTION_COMPONENT,
            execution_stage="action",
            execution_native=("gr00t_dit", int(self._d_handle)),
            queue_owner_id=int(self._d_handle),
            resolve_stage_domain=lambda length: (
                torch.ops.rpu.gr00t_denoise_resolve_stage_domain(
                    int(self._d_handle), int(length), int(cross_seq_len)
                )
            ),
            position=0,
            alignment=1,
            padding_rows=0,
            exact_chunk_size=exact if isinstance(exact, int) else None,
            request_id="gr00t:action_dit:action",
            graph_mode=GRAPH_COMPOSITE_CHILD,
            physical_metadata=(
                ("cross_kv_logical_length", int(cross_seq_len)),
                ("ddr_cross_handoff", 1),
                ("execution_generation", int(
                    self._component_generations[_ACTION_COMPONENT]
                )),
                ("single_full_sequence_chunk", 1),
            ),
            plan_result_sink=lambda result: plan_box.__setitem__(
                "result", result
            ),
            plan_signature=(int(cross_seq_len),),
            graph_cache=self._d_gc,
        )
        result = plan_box["result"]
        if (
            execution_len != _ACTION_QUERY_ROWS
            or result.selected is None
            or not result.selected.stage_tuple.physical_descriptor
        ):
            raise RuntimeError(
                "GR00T action planner returned no consumable native stage "
                "descriptor"
            )
        return int(chunk_size), result

    @torch.no_grad()
    def _denoise(self, vl_r, inputs, S, seed, execution_plan=None):
        action_chunk, action_plan = (
            self._action_execution_plan(S)
            if execution_plan is None else execution_plan
        )
        stage_descriptor = tuple(
            action_plan.selected.stage_tuple.physical_descriptor
        )
        # Opt2: text/image MASK_2D subsets (cross-attn over the backbone tokens) are constant per
        # prompt → memoize on input_ids. Cached RPU tensors also pin the denoise REPLAY mask addrs.
        # masks depend on input_ids (image subset) AND attention_mask (valid subset) — key on both.
        if (self._masks is None
                or not torch.equal(inputs["input_ids"], self._mask_key)
                or not torch.equal(inputs["attention_mask"], self._mask_attn_key)):
            im_mask = (inputs["input_ids"] == self._img_tok).reshape(-1).bool()
            bam = (inputs["attention_mask"] == 1).reshape(-1).bool()
            def mk(subset):
                return torch.where(subset.view(1, S).expand(_AH + 1, S),
                                   torch.zeros(_AH + 1, S), torch.full((_AH + 1, S), _NEG)).half()
            self._masks = (mk((~im_mask) & bam).to("rpu").contiguous(),
                           mk(im_mask & bam).to("rpu").contiguous())
            self._mask_key = inputs["input_ids"].clone()
            self._mask_attn_key = inputs["attention_mask"].clone()
        tmask_r, imask_r = self._masks
        state_r = _half_rpu(F.pad(inputs["state"].float(), (0, _AD_PAD - _AD)))      # [1,1,144]
        # Local generator: don't perturb the caller's process-global torch RNG (get_action is a
        # runtime API). Same seed → same noise as the old global manual_seed path.
        _gen = torch.Generator().manual_seed(int(seed))
        noise_r = _half_rpu(F.pad(torch.randn(1, _AH, _AD, generator=_gen), (0, _AD_PAD - _AD)))  # [1,40,144]
        d_sig = rpu_backend.graph.GraphSignature(op_id="gr00t_denoise_unroll", shapes=[_AH + 1, _H],
                                           dyn_dims=[
                                               _NL, S,
                                               self._component_generations[
                                                   _ACTION_COMPONENT
                                               ],
                                               *action_plan.graph_key_words(),
                                           ], dtypes=[torch.float16])
        with self._d_gc.capture(d_sig):
            act = torch.ops.rpu.gr00t_denoise_unroll_forward(
                self._d_handle, noise_r, state_r, self._d_cache.k_caches, self._d_cache.v_caches,
                vl_r, tmask_r, imask_r,
                stage_descriptor,
            )
        resolved = int(
            torch.ops.rpu.gr00t_denoise_get_resolved_chunk_size(
                int(self._d_handle)
            )
        )
        if resolved != action_chunk:
            raise RuntimeError(
                "GR00T action dry/forward chunk plan drift: "
                f"dry={action_chunk}, forward={resolved}, cross_seq_len={S}"
            )
        self._record_component_plan(
            _ACTION_COMPONENT,
            "action",
            action_plan,
            logical_len=_ACTION_QUERY_ROWS,
            resolved_chunk_size=resolved,
            physical_descriptor=stage_descriptor,
        )
        return act.float().cpu()[:, :, :_AD].reshape(1, _AH, _AD)     # crop 144->132 (normalized action)

    # ---- full image→action (fully on-device) ----
    @execution_serialized
    @torch.no_grad()
    def get_action(self, inputs, *, seed=0, num_steps=_NSTEPS):
        vars(self).pop("_qwen_z1_completed_owner", None)
        vars(self).pop("_qwen_z1_pending_owner", None)
        vars(self).pop("_qwen_z1_completed_ranges", None)
        vars(self).pop("_qwen_z1_pending_ranges", None)
        try:
            # The denoise handle bakes the Euler schedule (cond/tau/dt/body_iterations) at BUILD
            # time for _NSTEPS (see _build_denoise); per-call num_steps is NOT re-threaded, so a
            # different value would silently still run _NSTEPS. Reject rather than mislead.
            if num_steps != _NSTEPS:
                raise ValueError(
                    f"gr00t denoise is built for num_steps={_NSTEPS}; got {num_steps}. "
                    f"Rebuild via build_gr00t_vla(...) for a different schedule.")
            self._validate_forward_execution(inputs)
            # Capacity guard BEFORE the backbone: the backbone writes _bb_cache (512); _vl_encode /
            # _denoise use _a_cache / _d_cache (256). A longer prompt / more images would OOB-write a
            # KV cache in the C++ op (no capacity check there → wrong action / crash). Check the input
            # seq (== backbone seq) up front against the SMALLEST cache, so we never even run the
            # backbone with an over-capacity sequence.
            _S_in = int(inputs["input_ids"].shape[-1])
            _cap = min(getattr(self._bb_cache, "max_seq_len", 512),
                       getattr(self._a_cache, "max_seq_len", 256),
                       getattr(self._d_cache, "max_seq_len", 256))
            if _S_in > _cap:
                raise ValueError(f"GR00T input seq {_S_in} exceeds the KV-cache capacity {_cap} "
                                 f"(backbone/VL/denoise); rebuild with larger caches or shorten the input.")
            # Resolve the two downstream native descriptors before Vision/Text can
            # mutate caches.  This makes an impossible exact component override a
            # true preflight failure while AUTO and EXACT still use one A6 path.
            vl_execution_plan = self._vl_execution_plan(_S_in)
            action_execution_plan = self._action_execution_plan(_S_in)
            raw, S = self._backbone(inputs)
            if S != _S_in:
                raise RuntimeError(
                    "GR00T backbone changed the logical sequence length after "
                    f"planning: input={_S_in}, output={S}"
                )
            vl_r = self._vl_encode(raw, S, vl_execution_plan)
            action = self._denoise(
                vl_r, inputs, S, seed, action_execution_plan
            )
            if self._qwen_z1_plan_hash is not None:
                for cache in (self._qwen_z1_gc, self._a_gc, self._d_gc):
                    if not cache.is_frozen():
                        cache.freeze()
                retained = vars(self).pop("_qwen_z1_pending_owner", None)
                if retained is None:
                    raise RuntimeError("GR00T Z1 action has no completed native backbone")
                self._qwen_z1_check_completed_owner(retained)
                ranges = vars(self).pop("_qwen_z1_pending_ranges", None)
                self._qwen_z1_check_completed_ranges(retained, ranges)
                self._qwen_z1_completed_owner = retained
                self._qwen_z1_completed_ranges = ranges
            return action
        except BaseException:
            vars(self).pop("_qwen_z1_completed_owner", None)
            vars(self).pop("_qwen_z1_pending_owner", None)
            vars(self).pop("_qwen_z1_completed_ranges", None)
            vars(self).pop("_qwen_z1_pending_ranges", None)
            raise


# =============================================================================
# Build — backbone (Qwen3-VL 16L, RPU) + gr00t_vl_encoder (A) + gr00t_denoise (B/C)
# =============================================================================
def _ckpt_reader(ckpt_path):
    wmap = __import__("json").load(open(glob.glob(ckpt_path + "/*.index.json")[0]))["weight_map"]
    handles: dict = {}
    def gw(key):
        f = wmap[key]
        if f not in handles:
            handles[f] = safe_open(os.path.join(ckpt_path, f), framework="pt")
        return handles[f].get_tensor(key).float()
    return gw


def _build_vl_encoder(gw, w_rms):
    """A: swizzle 4 VL blocks + bake vlln + inv_w_rms. Returns (handle, keepalive).

    W8A16 (opt-in RPU_GR00T_W8A16): int8 the 6 GEMMs/block (q/k/v/o/ff0/ff2) — same int8-weight/
    fp16-act template as the DiT (the vl encoder is its structural twin). All dims int8-aligned
    (q/k/v=2048, ff0=8192, K∈{2048,8192}). Default off = fp16 (unchanged)."""
    hr = _half_rpu
    _w8a16 = _w8a16_on()
    if _w8a16:
        from rpu_backend.quant._common import quantize_linear_per_channel

    def _qw(w, swz):  # -> (rpu weight [int8 dwidth=1 if w8a16 else fp16], rpu fp16 scale or None)
        if _w8a16:
            wq, ws = quantize_linear_per_channel(w.half())
            return swz(wq, _TP, dwidth=1).contiguous().to("rpu"), ws.to(torch.float16).to("rpu").contiguous()
        return hr(swz(w.half(), _TP)), None

    L = {k: [] for k in ("n1w", "n1b", "qb", "kb", "vb", "ob", "n3w", "n3b", "f0b", "f2b")}
    GW = {k: [] for k in ("qw", "kw", "vw", "ow", "f0w", "f2w")}        # int8/fp16 weights
    SC = {k: [] for k in ("qw", "kw", "vw", "ow", "f0w", "f2w")}        # fp16 scales (None when fp16)
    for i in range(_VL_NL):
        p = f"{_AP}vl_self_attention.transformer_blocks.{i}."
        L["n1w"].append(hr(gw(p + "norm1.weight").half())); L["n1b"].append(hr(gw(p + "norm1.bias").half()))
        for key, name, swz in (("qw", "attn1.to_q.weight", tp_col_swizzle_mc_weight),
                               ("kw", "attn1.to_k.weight", tp_col_swizzle_mc_weight),
                               ("vw", "attn1.to_v.weight", tp_col_swizzle_mc_weight),
                               ("ow", "attn1.to_out.0.weight", tp_row_swizzle_mc_weight),
                               ("f0w", "ff.net.0.proj.weight", tp_col_swizzle_mc_weight),
                               ("f2w", "ff.net.2.weight", tp_row_swizzle_mc_weight)):
            w, s = _qw(gw(p + name), swz); GW[key].append(w); SC[key].append(s)
        L["qb"].append(hr(gw(p + "attn1.to_q.bias").half())); L["kb"].append(hr(gw(p + "attn1.to_k.bias").half()))
        L["vb"].append(hr(gw(p + "attn1.to_v.bias").half())); L["ob"].append(hr(gw(p + "attn1.to_out.0.bias").half()))
        L["n3w"].append(hr(gw(p + "norm3.weight").half())); L["n3b"].append(hr(gw(p + "norm3.bias").half()))
        L["f0b"].append(hr(gw(p + "ff.net.0.proj.bias").half())); L["f2b"].append(hr(gw(p + "ff.net.2.bias").half()))
    vlln_w = hr(gw(_AP + "vlln.weight").half()); vlln_b = hr(gw(_AP + "vlln.bias").half())
    inv_w_rms = hr((1.0 / w_rms).half())
    weight_args = (
        L["n1w"], L["n1b"], GW["qw"], L["qb"], GW["kw"], L["kb"],
        GW["vw"], L["vb"], GW["ow"], L["ob"], L["n3w"], L["n3b"],
        GW["f0w"], L["f0b"], GW["f2w"], L["f2b"], vlln_w, vlln_b,
        inv_w_rms, _VLH, _VLHD, _VLD, _VLFF, 1e-5,
    )
    scale_args = (
        SC["qw"], SC["kw"], SC["vw"], SC["ow"], SC["f0w"], SC["f2w"]
    )
    handle = _configure_gr00t_vl_encoder_handle(
        weight_args=weight_args, scale_args=scale_args, w8a16=_w8a16
    )
    return handle, (L, GW, SC, vlln_w, vlln_b, inv_w_rms)


def _build_denoise(gw, emb, num_steps=_NSTEPS):
    """B/C: 32 DiT blocks (gr00t_dit) + per-step glue (encoders/norm_out/decoder) + baked tables."""
    hr = _half_rpu

    def sc(w_lin, kpad=None, npad=None):       # -> (rpu weight, fp16 scale or None); single-core col-swizzle
        if kpad is not None and w_lin.size(1) < kpad:
            w_lin = F.pad(w_lin, (0, kpad - w_lin.size(1)))
        if npad is not None and w_lin.size(0) < npad:
            w_lin = F.pad(w_lin, (0, 0, 0, npad - w_lin.size(0)))
        # W8A16 (opt-in): int8 the glue Linear when K is int8-aligned (col-swizzle num_ele_32B=32, so
        # K%32; se1/ae1 K=144 fail it → stay fp16). Per-output-channel int8 + fp16 scale; the kernel
        # keys on (int8 weight + defined scale) → generated W8A16 ACC16/ACC32
        # auto-tile family. Default off = fp16.
        if _w8a16 and w_lin.size(1) % 32 == 0:
            wq, ws = quantize_linear_per_channel(w_lin.half())
            return (tp_col_swizzle_mc_weight(wq, 1, dwidth=1).contiguous().to("rpu"),
                    ws.to(torch.float16).to("rpu").contiguous())
        return hr(tp_col_swizzle_mc_weight(w_lin.half(), 1)), None

    def b(vec, npad=None):
        if npad is not None and vec.size(0) < npad:
            vec = F.pad(vec, (0, npad - vec.size(0)))
        return hr(vec.half())

    def catW(prefix):  return gw(prefix + ".W")[emb].T.contiguous()    # [in,out] -> [out,in]
    def catB(prefix):  return gw(prefix + ".b")[emb].contiguous()

    # --- 32 DiT blocks (col q/k/v/ff1, row to_out/ff2/adaln) ---
    # W8A16 (opt-in RPU_GR00T_W8A16) quantizes the 6 DiT GEMMs/block (q/k/v/o/ff1/ff2)
    # with per-output-channel symmetric int8 + fp16 row-scale and int8 swizzle (dwidth=1).
    # AdaLN GEMV stays fp16; sc() separately quantizes the 8 per-step K%32==0 GEMMs.
    # Default off = fp16.
    _w8a16 = _w8a16_on()
    if _w8a16:
        from rpu_backend.quant._common import quantize_linear_per_channel
    def _qw(w, swz):  # -> (rpu weight [int8 dwidth=1 if w8a16 else fp16], rpu fp16 scale or None)
        if _w8a16:
            wq, ws = quantize_linear_per_channel(w.half())
            return swz(wq, _TP, dwidth=1).contiguous().to("rpu"), ws.to(torch.float16).to("rpu").contiguous()
        return hr(swz(w.half(), _TP)), None
    DL = {k: [] for k in ("q", "k", "v", "o", "f1", "f2", "aw", "ab", "qb", "kb", "vb", "ob", "f1b", "f2b")}
    SC = {k: [] for k in ("q", "k", "v", "o", "f1", "f2", "aw")}
    is_cross = []
    for i in range(_NL):
        p = f"{_AP}model.transformer_blocks.{i}."
        for key, name, swz in (("q", "attn1.to_q.weight", tp_col_swizzle_mc_weight),
                               ("k", "attn1.to_k.weight", tp_col_swizzle_mc_weight),
                               ("v", "attn1.to_v.weight", tp_col_swizzle_mc_weight),
                               ("o", "attn1.to_out.0.weight", tp_row_swizzle_mc_weight),
                               ("f1", "ff.net.0.proj.weight", tp_col_swizzle_mc_weight),
                               ("f2", "ff.net.2.weight", tp_row_swizzle_mc_weight),
                               # AdaLN modulation uses the M=1 row GEMV layout;
                               # its optional INT8 weights retain the corresponding scale path.
                               ("aw", "norm1.linear.weight", tp_row_swizzle_mc_weight)):
            w, s = _qw(gw(p + name), swz); DL[key].append(w); SC[key].append(s)
        DL["ab"].append(hr(gw(p + "norm1.linear.bias").half()))
        DL["qb"].append(hr(gw(p + "attn1.to_q.bias").half())); DL["kb"].append(hr(gw(p + "attn1.to_k.bias").half()))
        DL["vb"].append(hr(gw(p + "attn1.to_v.bias").half())); DL["ob"].append(hr(gw(p + "attn1.to_out.0.bias").half()))
        DL["f1b"].append(hr(gw(p + "ff.net.0.proj.bias").half())); DL["f2b"].append(hr(gw(p + "ff.net.2.bias").half()))
        is_cross.append(1 if i % 2 == 0 else 0)
    dit_args = (
        DL["q"], DL["k"], DL["v"], DL["o"], DL["f1"], DL["f2"],
        DL["aw"], DL["ab"], DL["qb"], DL["kb"], DL["vb"], DL["ob"],
        DL["f1b"], DL["f2b"], is_cross,
        _NQ, _HD, _H, _FF, _CROSS, 1e-5,
    )
    dit_scale_args = (
        SC["q"], SC["k"], SC["v"], SC["o"], SC["f1"], SC["f2"], SC["aw"]
    )

    # --- per-step glue weights (single-core, emb-sliced, padded). W8A16: sc() int8s the 8 K%32==0
    #     Linears (se2/ae2a/ae2t/ae3/dec1/dec2/po1/po2) + returns fp16 scales; se1/ae1 (K=144) stay fp16. ---
    W2 = gw(_AP + "action_encoder.W2.W")[emb]                          # [3072,1536]
    se1, _se1_s = sc(catW(_AP + "state_encoder.layer1"), kpad=_AD_PAD); se1_b = b(catB(_AP + "state_encoder.layer1"))
    se2, se2_s = sc(catW(_AP + "state_encoder.layer2")); se2_b = b(catB(_AP + "state_encoder.layer2"))
    ae1, _ae1_s = sc(catW(_AP + "action_encoder.W1"), kpad=_AD_PAD); ae1_b = b(catB(_AP + "action_encoder.W1"))
    ae2a, ae2a_s = sc(W2[:_H].T.contiguous()); ae2t, ae2t_s = sc(W2[_H:].T.contiguous()); ae2_b = b(catB(_AP + "action_encoder.W2"))
    ae3, ae3_s = sc(catW(_AP + "action_encoder.W3")); ae3_b = b(catB(_AP + "action_encoder.W3"))
    dec1, dec1_s = sc(catW(_AP + "action_decoder.layer1")); dec1_b = b(catB(_AP + "action_decoder.layer1"))
    dec2, dec2_s = sc(catW(_AP + "action_decoder.layer2"), npad=_AD_PAD); dec2_b = b(catB(_AP + "action_decoder.layer2"), npad=_AD_PAD)
    po1, po1_s = sc(gw(_AP + "model.proj_out_1.weight")); po1_b = b(gw(_AP + "model.proj_out_1.bias"))
    po2, po2_s = sc(gw(_AP + "model.proj_out_2.weight")); po2_b = b(gw(_AP + "model.proj_out_2.bias"))

    # --- baked tables (cond[4], tau[4,40,1536], pos[40,1536]); fixed num_steps schedule ---
    def _timestep_cond(tb):                                            # SiLU(timestep_embedder(bucket))
        half = 128
        exp = (-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / (half - 1)).exp()
        e = torch.tensor([float(tb)]).unsqueeze(1) * exp.unsqueeze(0)
        e = torch.cat([e.sin(), e.cos()], -1); e = torch.cat([e[:, half:], e[:, :half]], -1)  # flip
        h1 = F.silu(F.linear(e, gw(_AP + "model.timestep_encoder.timestep_embedder.linear_1.weight"),
                                gw(_AP + "model.timestep_encoder.timestep_embedder.linear_1.bias")))
        temb = F.linear(h1, gw(_AP + "model.timestep_encoder.timestep_embedder.linear_2.weight"),
                            gw(_AP + "model.timestep_encoder.timestep_embedder.linear_2.bias"))
        return F.silu(temb).reshape(-1)
    cond = torch.stack([_timestep_cond(int((t / num_steps) * _NBUCKETS)) for t in range(num_steps)], 0)
    half = _H // 2
    exp = (-torch.arange(half, dtype=torch.float32) * (math.log(10000.0) / half)).exp()
    tau = torch.stack([torch.cat([(torch.full((_AH, 1), float(int((t / num_steps) * _NBUCKETS))) * exp).sin(),
                                   (torch.full((_AH, 1), float(int((t / num_steps) * _NBUCKETS))) * exp).cos()], -1)
                       for t in range(num_steps)], 0)
    pos = F.embedding(torch.arange(_AH), gw(_AP + "position_embedding.weight"))
    cond_t, tau_t, pos_t = hr(cond), hr(tau), hr(pos)

    denoise_args = (
        se1, se1_b, se2, se2_b, ae1, ae1_b, ae2a, ae2t, ae2_b,
        ae3, ae3_b, dec1, dec1_b, dec2, dec2_b, po1, po1_b, po2, po2_b,
        cond_t, tau_t, pos_t, _AD, _AD_PAD, num_steps, 1.0 / num_steps,
        se2_s, ae2a_s, ae2t_s, ae3_s, dec1_s, dec2_s, po1_s, po2_s,
    )  # scale entries are None when fp16
    handle = _configure_gr00t_denoise_handle(
        dit_args=dit_args,
        dit_scale_args=dit_scale_args,
        denoise_args=denoise_args,
        w8a16=_w8a16,
    )
    keep = (DL, SC, se1, se1_b, se2, se2_b, ae1, ae1_b, ae2a, ae2t, ae2_b, ae3, ae3_b,
            dec1, dec1_b, dec2, dec2_b, po1, po1_b, po2, po2_b, cond_t, tau_t, pos_t,
            se2_s, ae2a_s, ae2t_s, ae3_s, dec1_s, dec2_s, po1_s, po2_s)
    return handle, keep


def build_gr00t_vla(
    ckpt_path,
    qwen3vl_local,
    *,
    embodiment_id=20,
    rpu_execution=None,
):
    """Build the fully-on-device RPU GR00T VLA: backbone (Qwen3-VL 16L) + A + B/C fused ops."""
    from rpu_backend.api._execution import _require_execution_process_safe
    _require_execution_process_safe()
    qwen_z1_enabled = _qwen3vl_spm_z1_on()
    (
        execution_config,
        text_execution,
        vision_execution,
        _vl_execution,
        _action_execution,
    ) = _resolve_gr00t_execution(
        rpu_execution,
        entry_point="build_gr00t_vla",
        qwen_z1=qwen_z1_enabled,
    )
    if qwen_z1_enabled:
        if _w8a16_on():
            raise ValueError(
                "RPU_GR00T_QWEN3VL_SPM_Z1 currently supports only the exact "
                "FP16 Vision/Text profile; disable RPU_GR00T_W8A16"
            )
        fused_merger = os.environ.get(
            "RPU_QWEN3VL_VISION_FUSED_MERGER"
        )
        if fused_merger is not None and fused_merger.lower() not in (
            "1", "true"
        ):
            raise ValueError(
                "RPU_GR00T_QWEN3VL_SPM_Z1 requires "
                "RPU_QWEN3VL_VISION_FUSED_MERGER=1"
            )
        # Normalize accepted truthy spellings before the C++ cold gate is read.
        os.environ["RPU_QWEN3VL_VISION_FUSED_MERGER"] = "1"
        rope_spm = os.environ.get("RPU_QWEN3VL_VISION_ROPE_SPM")
        if rope_spm is not None and rope_spm.lower() not in ("1", "true"):
            raise ValueError(
                "RPU_GR00T_QWEN3VL_SPM_Z1 requires "
                "RPU_QWEN3VL_VISION_ROPE_SPM=1"
            )
        os.environ["RPU_QWEN3VL_VISION_ROPE_SPM"] = "1"
        kv_v16 = os.environ.get("RPU_KVINSERT_V16")
        if kv_v16 is not None and kv_v16.lower() not in ("1", "true"):
            raise ValueError(
                "RPU_GR00T_QWEN3VL_SPM_Z1 requires RPU_KVINSERT_V16=1"
            )
        os.environ["RPU_KVINSERT_V16"] = "1"
        partial_mrope = os.environ.get("RPU_GR00T_PARTIAL_MROPE")
        if partial_mrope is not None and partial_mrope.lower() in (
            "1", "true"
        ):
            raise ValueError(
                "RPU_GR00T_QWEN3VL_SPM_Z1 freezes legacy M-RoPE; set "
                "RPU_GR00T_PARTIAL_MROPE=0"
            )
        os.environ["RPU_GR00T_PARTIAL_MROPE"] = "0"
        host_fp32_patch = os.environ.get(
            "RPU_QWEN3VL_VISION_HOST_FP32_PATCH"
        )
        if host_fp32_patch is not None and host_fp32_patch.lower() in (
            "1", "true"
        ):
            raise ValueError(
                "RPU_GR00T_QWEN3VL_SPM_Z1 requires the CPU FP16 patch path; "
                "set RPU_QWEN3VL_VISION_HOST_FP32_PATCH=0"
            )
        os.environ["RPU_QWEN3VL_VISION_HOST_FP32_PATCH"] = "0"
        # The composite dispatch is one exact image and bypasses the ordinary
        # Vision minibatch planner.
        os.environ["RPU_QWEN3VL_VISION_BATCH"] = "0"
    # All reductions use the generator-scheduled ring path.
    # Vision 2D-RoPE reuses SPM-resident tables instead of streaming DDR tables per layer.
    # This builder supplies the default; explicit environment settings remain authoritative.
    os.environ.setdefault("RPU_QWEN3VL_VISION_ROPE_SPM", "1")
    # Denoise KV PAD16/V16 is frozen in the action child's COMPLETE descriptor.
    # Partial M-RoPE precomputes per-token interleaved cos/sin for the backbone,
    # so layers consume the prepared rotation tables. The exact composite profile
    # retains its separately bound legacy M-RoPE route.
    os.environ.setdefault("RPU_GR00T_PARTIAL_MROPE", "1")
    # Fast replay avoids walking recorded operations on the host. Backbone,
    # visual-language encoder and denoise can skip complete recorded bodies;
    # vision retains any required merger post hook. Explicit environment settings
    # remain authoritative over the builder's defaults.
    os.environ.setdefault("RPU_WALL_OSS_FAST_REPLAY", "1")
    os.environ.setdefault("RPU_FASTREPLAY_SKIP_SYNC", "1")
    os.environ.setdefault("RPU_DEEP_FAST_REPLAY", "1")
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    cfg = Qwen3VLConfig.from_pretrained(qwen3vl_local)
    cfg.text_config.num_hidden_layers = _SELECT_LAYER
    model = Qwen3VLForConditionalGeneration(cfg)
    model._rpu_execution = execution_config
    sd = {}
    for f in glob.glob(ckpt_path + "/model-*.safetensors"):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                if k.startswith("backbone.model."):
                    sd[k[len("backbone.model."):]] = sf.get_tensor(k)
    if not sd:
        raise ValueError(
            f"GR00T checkpoint at {ckpt_path}: no `backbone.model.*` tensors found (wrong bundle "
            f"or changed prefix) — the backbone would stay random-initialized.")
    # strict=False is intended: the ckpt carries the FULL Qwen3-VL backbone but we keep only
    # _SELECT_LAYER text layers (upper layers land in unexpected_keys) and drop the unused lm_head.
    # A MISSING per-layer backbone weight, though, = a silent random-init → fail loud.
    _res = model.load_state_dict(sd, strict=False)
    _missing_core = [k for k in _res.missing_keys
                     if ".layers." in k and (k.endswith(".weight") or k.endswith(".bias"))]
    if _missing_core:
        raise ValueError(
            f"GR00T backbone: {len(_missing_core)} per-layer weights not loaded "
            f"(e.g. {_missing_core[:4]}) — random-init risk; check the checkpoint/config.")
    from rpu_backend.api.causal_lm import _claim_live_instance
    _claim_live_instance(model)
    model._rpu_swizzle_started = True

    text_model = vision_model = None
    a_handle = d_handle = None
    a_keep = d_keep = None
    forward_snapshots = ()
    qwen_z1_plan_hash = None
    qwen_z1_vision_stage_descriptor = None
    gr00t_z2_plan_hash = None
    transferred = False
    runtime = None
    try:
        model = model.half().eval()
        w_rms = model.model.language_model.norm.weight.detach().float().cpu()

        text_model, vision_model = model.model.language_model, model.model.visual
        # Snapshot the instance forwards BEFORE the qwen3_vl installers replace
        # them, so the rollback can put them back (see
        # `uninstall_qwen3_vl_runtime`). Taken here, after text/vision are
        # resolved and before the first mutation.
        forward_snapshots = tuple(
            (owner, "forward" in vars(owner), vars(owner).get("forward"))
            for owner in (model, text_model, vision_model)
        )
        # Backbone W8A16 quantizes the text decoder's seven projections before
        # swizzle. INT8 uses dwidth=1, and scale_lists select the native W8A16 setter.
        # Quantization changes the backbone's numerical path and remains opt-in.
        if _w8a16_on():
            _quantize_backbone_inplace(text_model)
        convert_linear_weights_inplace(text_model, skip_names=set())
        text_model.to("rpu")
        bb_scale_lists = (
            _backbone_scale_lists(text_model) if _w8a16_on() else None
        )
        # Vision-ViT W8A16 (gated): int8 the 24-block qwen3vl vision encoder's 6 GEMMs/block. The
        # install w8a16 path is opt-in (default off → qwen3_vl/qwen3-vl E2E byte-identical). The merger
        # GEMMs (enable_..._merger_on_device below) stay fp16. Drift absorbed by the denoise.
        install_qwen3_vl_vision_for_rpu(
            vision_model,
            vision_config=cfg.vision_config,
            w8a16=_w8a16_on(),
            execution_chunk_size=vision_execution.get(
                "vision", {}
            ).get("chunk_size", "auto"),
        )
        # Opt4: run the patch-merger + deepstack-merger GEMMs on RPU (default qwen3_vl keeps them CPU
        # fp32). The merger is the single largest vision stage; the fp16 drift is absorbed by denoise.
        enable_qwen3_vl_vision_merger_on_device(vision_model)
        # ROUND-3 opt #1: the Python cold builder is the only selector
        # authority. Registering merger weights enables the native post_fn;
        # omitting them keeps the eager merger path.
        if rpu_env_bool("RPU_QWEN3VL_VISION_FUSED_MERGER"):
            register_qwen3vl_vision_fused_merger(vision_model)
        # RPU_QWEN3VL_VISION_BATCH groups consecutive equal-size images for one
        # vision forward while preserving per-image attention boundaries.
        # The vision adapter reads this explicit option; no setup is needed here.
        n_ds = len(getattr(cfg.vision_config, "deepstack_visual_indexes", []))
        install_qwen3_vl_text_for_rpu(
            text_model,
            text_config=cfg.text_config,
            vision_config=cfg.vision_config,
            deepstack_lang_layers=list(range(n_ds)),
            enable_deepstack=True,
            scale_lists=bb_scale_lists,
            execution_config=text_execution,
        )

        gw = _ckpt_reader(ckpt_path)
        a_handle, a_keep = _build_vl_encoder(gw, w_rms)
        d_handle, d_keep = _build_denoise(gw, embodiment_id)

        a_cache = RPUCache(num_layers=_VL_NL, batch_size=1, max_seq_len=256,
                           num_kv_heads=_VLH, head_dim=_VLHD, attn_tp=_TP)
        # Cross-attn KV holds the whole backbone sequence, not just action rows.
        # Bind both actual owner caches before the first composite preparation.
        d_cache = RPUCache(num_layers=_NL, batch_size=1, max_seq_len=256,
                           num_kv_heads=_NQ, head_dim=_HD, attn_tp=_TP)
        torch.ops.rpu.gr00t_vl_encoder_set_chunk_envelope(a_handle, int(a_cache.max_seq_len), 0)
        torch.ops.rpu.gr00t_dit_set_chunk_envelope(d_handle, int(d_cache.max_seq_len), 0)

        if qwen_z1_enabled:
            if n_ds != _QWEN_Z1_DEEPSTACK_COUNT:
                raise ValueError(
                    "GR00T Qwen3-VL SPM Z1 requires exactly three DeepStack "
                    f"routes, got {n_ds}"
                )
            # Freeze A/D's exact S=82 persistent/layout requirements before the
            # Qwen composite captures the process-wide persistent top.  The
            # ordinary A/D forwards remain legal because no Z2 lease is bound.
            gr00t_z2_plan_hash = int(
                torch.ops.rpu.gr00t_spm_z2_prepare(
                    int(a_handle), int(d_handle), _QWEN_Z1_REAL_LEN
                )
            )
            qwen_z1_plan_hash = int(
                torch.ops.rpu.qwen3vl_pooler_spm_z1_prepare(
                    int(vision_model._rpu_vision_handle),
                    int(text_model._rpu_decoder_handle),
                    _QWEN_Z1_NUM_PATCHES,
                    _QWEN_Z1_EXECUTION_LEN,
                    _QWEN_Z1_IMAGE_BEGIN,
                    _QWEN_Z1_REAL_LEN,
                    _QWEN_Z1_DEEPSTACK_COUNT,
                )
            )
            qwen_z1_vision_stage_descriptor = tuple(
                int(word) for word in
                torch.ops.rpu.qwen3vl_pooler_spm_z1_vision_stage_descriptor(
                    int(vision_model._rpu_vision_handle),
                    int(text_model._rpu_decoder_handle),
                    qwen_z1_plan_hash,
                )
            )
            if not qwen_z1_vision_stage_descriptor:
                raise RuntimeError(
                    "GR00T Qwen3-VL SPM Z1 prepared an empty Vision descriptor"
                )
            _unprepare_gr00t_spm_z2(
                a_handle=a_handle,
                d_handle=d_handle,
                plan_hash=gr00t_z2_plan_hash,
            )
            gr00t_z2_plan_hash = None

        runtime = Gr00tN1d7VLA.__new__(Gr00tN1d7VLA)
        vars(runtime).update(
            _bb=model, _text=text_model, _vision=vision_model,
            _a_handle=a_handle, _a_keep=a_keep, _d_handle=d_handle, _d_keep=d_keep,
            _qwen_z1_plan_hash=qwen_z1_plan_hash, _closed=False,
            _gc_retirement_enabled=False,
        )
        Gr00tN1d7VLA.__init__(runtime,
            backbone=model, cfg=cfg, text_model=text_model,
            vision_model=vision_model, a_handle=a_handle, a_keep=a_keep, a_cache=a_cache,
            d_handle=d_handle, d_keep=d_keep, d_cache=d_cache,
            image_token_id=model.config.image_token_id,
            execution_config=execution_config,
            embodiment_id=embodiment_id,
            qwen_z1_plan_hash=qwen_z1_plan_hash,
            qwen_z1_vision_stage_descriptor=(
                qwen_z1_vision_stage_descriptor
            ),
        )
        transferred = True
        try:
            model._rpu_swizzled = True
        except BaseException as error:
            _cleanup_gr00t_build_failure(error, runtime, runtime.close)
            raise
        return runtime
    except BaseException as error:
        if not transferred:
            if runtime is not None:
                runtime._gc_retirement_enabled = False
                def cleanup():
                    runtime.close()
                    uninstall_qwen3_vl_runtime(model, text_model, vision_model, forward_snapshots)
                _cleanup_gr00t_build_failure(error, runtime, cleanup)
            else:
                _cleanup_gr00t_build_failure(error, (model, a_handle, d_handle, a_keep, d_keep),
                    lambda: _cleanup_gr00t_partial_build(
                a_handle=a_handle,
                d_handle=d_handle,
                a_keep=a_keep,
                d_keep=d_keep,
                model=model,
                text_model=text_model,
                vision_model=vision_model,
                forward_snapshots=forward_snapshots,
                qwen_z1_plan_hash=qwen_z1_plan_hash,
                gr00t_z2_plan_hash=gr00t_z2_plan_hash,
                ))
        raise
