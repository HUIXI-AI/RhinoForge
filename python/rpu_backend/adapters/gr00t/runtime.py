"""GR00T-N1.7-3B VLA — RPU runtime (image → 132-d action, flow-matching denoise).

Self-contained standalone runtime (NOT a `register_adapter` HF patch: the GR00T package is not
importable in the RPU env). **"全程 RPU" — ALL per-action compute on-device** (the host-fp32 glue
was fused into graph subsystems, eliminating ~10 host↔device transfers/action):

  image+text → RPU Qwen3-VL backbone (reuse adapters/qwen3_vl)          → raw [1,S,2048]
            → gr00t_vl_encoder (vlln + 4-layer vl_self_attention, RPU) → vl_embeds [1,S,2048]
  state+noise → gr00t_denoise_unroll (4-step Euler denoise in one graph, RPU)
                  [state/action encoders → 32-block DiT → norm_out → decoder → Euler] → action [1,40,132]

Caller contracts:
  - op inputs (raw / vl_embeds / state / noise / masks) are held as named vars across the op call
    (mutable-DMA lifetime);
  - the text/image MASK_2D subsets each get their OWN prepared slot in the C++ (prep_mask_into).
"""
from __future__ import annotations
import os, math, glob
import weakref
import torch
import torch.nn.functional as F
from safetensors import safe_open

import rpu_backend
from rpu_backend.api import RPUCache
from rpu_backend.runtime import rpu_env_bool
from rpu_backend.runtime.decoder import (
    _run_causal_decoder_forward,
    chunk_policy_key,
    plan_bounded_prefill_execution,
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


def _normalize_gr00t_execution(rpu_execution):
    from rpu_backend.api._execution import normalize_rpu_execution
    execution_config = normalize_rpu_execution(
        rpu_execution,
        entry_point="build_gr00t_vla",
        supported={
            "prefill": ("chunk_size", "padding_rows", "padding_budget"),
            "vision": ("chunk_size",),
            "action": ("chunk_size",),
        },
    )
    _validate_gr00t_execution_geometry(execution_config)
    return execution_config


def _validate_gr00t_execution_geometry(execution_config) -> None:
    """Reject fixed-stage requests that do not match GR00T's real geometry."""
    prefill_chunk = execution_config.get("prefill", {}).get(
        "chunk_size", "auto"
    )
    if isinstance(prefill_chunk, int):
        raise ValueError(
            "build_gr00t_vla: fixed prefill.chunk_size is not certified for "
            "the truncated 16-layer text geometry; use 'auto'. Hardware "
            "rejected exact 16/32/64/128 at the shipped S=82/S=148 inputs. "
            "Refusing before loading model weights."
        )
    vision_chunk = execution_config.get("vision", {}).get("chunk_size", "auto")
    if (
        isinstance(vision_chunk, int)
        and vision_chunk not in _VISION_EXACT_CHUNKS
    ):
        raise ValueError(
            "build_gr00t_vla: vision.chunk_size must be 'auto' or one of "
            f"{sorted(_VISION_EXACT_CHUNKS)}, the certified native Vision "
            f"dispatch chunks; got {vision_chunk}."
        )
    if vision_chunk == 144 and rpu_env_bool(
        "RPU_QWEN3VL_VISION_FUSED_MERGER"
    ):
        raise ValueError(
            "build_gr00t_vla: vision.chunk_size=144 is incompatible with "
            "RPU_QWEN3VL_VISION_FUSED_MERGER=1; choose 256 for a standard "
            "256-patch image or disable the fused merger. Refusing before "
            "loading model weights."
        )
    action_chunk = execution_config.get("action", {}).get("chunk_size", "auto")
    if isinstance(action_chunk, int) and action_chunk != _ACTION_CHUNK_SIZE:
        raise ValueError(
            "build_gr00t_vla: action.chunk_size must be 'auto' or exactly "
            f"{_ACTION_CHUNK_SIZE}, the single chunk for "
            f"{_ACTION_QUERY_ROWS} fixed action-query rows; got {action_chunk}. "
            "GR00T action does not expose variable chunking."
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
    """Enable host-baked per-token M-RoPE tables for backbone rotation."""
    return rpu_env_bool("RPU_GR00T_PARTIAL_MROPE")


def _extract_mrope_params(text_config):
    """Return the geometry needed to build interleaved M-RoPE tables."""
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
    try:
        torch.ops.rpu.gr00t_vl_encoder_destroy(handle)
    except Exception:
        pass


def _destroy_gr00t_dit_handle(handle):
    try:
        torch.ops.rpu.gr00t_dit_destroy(handle)
    except Exception:
        pass


def _configure_gr00t_vl_encoder_handle(
    *, weight_args, scale_args, w8a16: bool
):
    handle = torch.ops.rpu.gr00t_vl_encoder_create()
    configured = False
    try:
        if w8a16:
            torch.ops.rpu.gr00t_vl_encoder_set_weights_w8a16(
                handle, *weight_args, *scale_args
            )
        else:
            torch.ops.rpu.gr00t_vl_encoder_set_weights(handle, *weight_args)
        configured = True
        return handle
    finally:
        if not configured:
            _destroy_gr00t_vl_encoder_handle(handle)


def _configure_gr00t_denoise_handle(
    *, dit_args, dit_scale_args, denoise_args, w8a16: bool
):
    handle = torch.ops.rpu.gr00t_dit_create()
    configured = False
    try:
        if w8a16:
            torch.ops.rpu.gr00t_dit_set_weights_w8a16(
                handle, *dit_args, *dit_scale_args
            )
        else:
            torch.ops.rpu.gr00t_dit_set_weights(handle, *dit_args)
        torch.ops.rpu.gr00t_denoise_set_weights(handle, *denoise_args)
        configured = True
        return handle
    finally:
        if not configured:
            _destroy_gr00t_dit_handle(handle)


def _clear_graph_cache(owner, attr: str) -> None:
    cache = getattr(owner, attr, None)
    if cache is not None:
        try:
            cache.clear()
        except Exception:
            pass


def _invoke_finalizer(owner, attr: str) -> None:
    finalizer = getattr(owner, attr, None)
    if finalizer is not None and getattr(finalizer, "alive", False):
        try:
            finalizer()
        except Exception:
            pass


def _unprepare_qwen3vl_spm_z1(
    *, pipeline_cache, vision_model, text_model, plan_hash
) -> None:
    """Retire the composite Graph before releasing either native handle."""
    if plan_hash is None:
        return
    if pipeline_cache is not None:
        pipeline_cache.clear()
    torch.ops.rpu.qwen3vl_pooler_spm_z1_unprepare(
        int(vision_model._rpu_vision_handle),
        int(text_model._rpu_decoder_handle),
        int(plan_hash),
    )


def _unprepare_gr00t_spm_z2(*, a_handle, d_handle, plan_hash) -> None:
    """Retire the temporary encoder/denoiser layout after persistent preflight."""
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


def _cleanup_gr00t_backbone(text_model, vision_model) -> None:
    """Clear Qwen3-VL graph state before retiring its tracked handles."""
    if text_model is not None:
        _clear_graph_cache(text_model, "_rpu_text_graph_cache")
        _clear_graph_cache(text_model, "_rpu_decoder_graph_cache")
        _invoke_finalizer(text_model, "_rpu_decoder_handle_finalizer")
    if vision_model is not None:
        _clear_graph_cache(vision_model, "_rpu_vision_graph_cache")
        _invoke_finalizer(vision_model, "_rpu_vision_handle_finalizer")


def _cleanup_gr00t_partial_build(
    *, a_handle, d_handle, model, text_model, vision_model, forward_snapshots=(),
    qwen_z1_pipeline_cache=None, qwen_z1_plan_hash=None,
    gr00t_z2_plan_hash=None,
) -> None:
    """Retire raw component handles, then UNINSTALL the backbone.

    The backbone half delegates to `uninstall_qwen3_vl_runtime`, paired with
    the `install_qwen3_vl_{vision,text}_for_rpu` installers. Cleanup clears
    caches and finalizers, removes published `_rpu_*` state, and restores bound
    forwards. Weight swizzles remain irreversible.
    """
    if qwen_z1_plan_hash is not None:
        _unprepare_qwen3vl_spm_z1(
            pipeline_cache=qwen_z1_pipeline_cache,
            vision_model=vision_model,
            text_model=text_model,
            plan_hash=qwen_z1_plan_hash,
        )
    if gr00t_z2_plan_hash is not None:
        _unprepare_gr00t_spm_z2(
            a_handle=a_handle,
            d_handle=d_handle,
            plan_hash=gr00t_z2_plan_hash,
        )
    if d_handle is not None:
        _destroy_gr00t_dit_handle(d_handle)
    if a_handle is not None:
        _destroy_gr00t_vl_encoder_handle(a_handle)
    if model is not None:
        uninstall_qwen3_vl_runtime(
            model, text_model, vision_model, forward_snapshots
        )


def _register_gr00t_handle_finalizers(owner, a_handle, d_handle):
    """Register both callbacks atomically; the caller owns raw handles on error."""
    a_finalizer = weakref.finalize(
        owner, _destroy_gr00t_vl_encoder_handle, a_handle
    )
    try:
        d_finalizer = weakref.finalize(
            owner, _destroy_gr00t_dit_handle, d_handle
        )
    except BaseException:
        a_finalizer.detach()
        raise
    return a_finalizer, d_finalizer


class Gr00tN1d7VLA:
    """Image→action runtime (fully on-device glue). Build via `build_gr00t_vla(...)`."""

    def __init__(self, *, backbone, cfg, text_model, vision_model,
                 a_handle, a_keep, d_handle, d_keep, image_token_id,
                 execution_config=None, embodiment_id=20,
                 qwen_z1_plan_hash=None):
        self._bb = backbone
        self._cfg = cfg
        self._text = text_model
        self._vision = vision_model
        self._a_handle = a_handle           # gr00t_vl_encoder
        self._a_keep = a_keep               # weight keepalive
        self._d_handle = d_handle           # gr00t_denoise (extends gr00t_dit)
        self._d_keep = d_keep
        self._img_tok = image_token_id
        self._emb = embodiment_id
        self._rpu_execution = (
            execution_config if execution_config is not None else {}
        )
        self._a_gc = rpu_backend.graph.GraphCache()
        self._d_gc = rpu_backend.graph.GraphCache()
        self._qwen_z1_plan_hash = qwen_z1_plan_hash
        self._qwen_z1_gc = (
            rpu_backend.graph.GraphCache(max_entries=1).begin_warmup()
            if qwen_z1_plan_hash is not None else None
        )
        if qwen_z1_plan_hash is not None:
            self._a_gc.begin_warmup()
            self._d_gc.begin_warmup()
        self._qwen_z1_text_hidden = None
        self._qwen_z1_position_ids = None
        self._a_cache = RPUCache(num_layers=_VL_NL, batch_size=1, max_seq_len=256,
                                 num_kv_heads=_VLH, head_dim=_VLHD, attn_tp=_TP)
        # Denoise cross-attn KV stores one entry per backbone token and supports
        # multi-image prompts up to the declared capacity.
        self._d_cache = RPUCache(num_layers=_NL, batch_size=1, max_seq_len=256,
                                 num_kv_heads=_NQ, head_dim=_HD, attn_tp=_TP)
        # Reuse one backbone prefill cache and reset its logical position; its
        # storage address remains stable for graph replay.
        self._bb_cache = RPUCache(num_layers=cfg.text_config.num_hidden_layers, batch_size=1,
                                  max_seq_len=512, num_kv_heads=cfg.text_config.num_key_value_heads,
                                  head_dim=cfg.text_config.head_dim)
        # Per-prompt memo slots are keyed on input_ids identity. rope = CPU M-RoPE index math;
        # masks = the two cross-attn MASK_2D subsets (cached RPU tensors also pin denoise REPLAY addrs).
        self._rope_key = self._rope_ids = None
        self._pos_rpu = self._attn_rpu = None   # RPU position_ids/attention_mask, memoized w/ _rope_ids
        self._mask_key = self._masks = None
        # Keep prompt-constant text embeddings at a stable RPU address and
        # overwrite only contiguous image-token runs each forward.
        self._emb_key = self._emb_text_rpu = self._img_runs = None
        # Persistent DeepStack dense buffers (one per merger), memoized on input_ids.
        self._ds_dense = None
        # Partial M-RoPE tables are baked once per prompt and memoized with
        # position IDs. The exact composite profile uses its frozen plan.
        self._partial_mrope = (
            False if qwen_z1_plan_hash is not None else _partial_mrope_on()
        )
        self._mrope_hd, self._mrope_theta, self._mrope_section = _extract_mrope_params(cfg.text_config)
        self._rope_cos_il = self._rope_sin_il = None
        self._a_handle_finalizer, self._d_handle_finalizer = (
            _register_gr00t_handle_finalizers(self, a_handle, d_handle)
        )
        self._closed = False

    def close(self) -> None:
        """Release all GR00T/backbone graph and native resources, once."""
        if self._closed:
            return
        if getattr(self, "_qwen_z1_plan_hash", None) is not None:
            _unprepare_qwen3vl_spm_z1(
                pipeline_cache=getattr(self, "_qwen_z1_gc", None),
                vision_model=self._vision,
                text_model=self._text,
                plan_hash=self._qwen_z1_plan_hash,
            )
            self._qwen_z1_plan_hash = None
            self._qwen_z1_gc = None
        self._closed = True
        _clear_graph_cache(self, "_d_gc")
        _clear_graph_cache(self, "_a_gc")
        _invoke_finalizer(self, "_d_handle_finalizer")
        _invoke_finalizer(self, "_a_handle_finalizer")
        _cleanup_gr00t_backbone(self._text, self._vision)

    # ---- RPU Qwen3-VL backbone -> raw [1,S,2048] post-RMSNorm ----
    def _prefill_execution_plan(self, logical_len):
        stage = self._rpu_execution.get("prefill", {})
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
        return plan_bounded_prefill_execution(
            int(logical_len),
            int(self._bb_cache.max_seq_len),
            padding_budget,
            lambda execution_len: (
                torch.ops.rpu.causal_decoder_resolve_prefill_chunk_size(
                    self._text._rpu_decoder_handle, int(execution_len), 0,
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
        )

    def _validate_forward_execution(self, inputs):
        requested = self._rpu_execution.get("vision", {}).get("chunk_size")
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
        if requested == 144 and rpu_env_bool(
            "RPU_QWEN3VL_VISION_FUSED_MERGER"
        ):
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

    @torch.no_grad()
    def _backbone_qwen_z1(self, inputs):
        self._validate_qwen_z1_inputs(inputs)
        if self._qwen_z1_plan_hash is None or self._qwen_z1_gc is None:
            raise RuntimeError("GR00T Qwen3-VL SPM Z1 was not prepared")

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
            self._rope_ids, _ = m.model.get_rope_index(
                input_ids.cpu(),
                mm_token_type_ids=mmtt.cpu(),
                image_grid_thw=grid.cpu(),
                video_grid_thw=None,
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
            with self._qwen_z1_gc.capture(signature):
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

        if tuple(raw.shape) != (1, _QWEN_Z1_EXECUTION_LEN, 2048):
            raise RuntimeError(
                "GR00T Qwen3-VL SPM Z1 returned an unexpected Text shape "
                f"{tuple(raw.shape)}"
            )
        _assert_qwen3vl_component_caches_empty(vision_model, text_model)
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
        # pooler_output is already flat in image order.
        visual_flat = vout.pooler_output
        # Image positions are overwrite-before-read; non-image positions remain
        # prompt-constant and the embedding buffer address stays stable for REPLAY.
        if self._emb_text_rpu is None or not torch.equal(input_ids, self._emb_key):
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
            # Each DeepStack merger owns a persistent dense buffer. Image rows
            # are overwritten in place; all other rows remain zero.
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
        # get_rope_index is CPU M-RoPE index math, constant per prompt → memoize on input_ids.
        # get_rope_index depends on input_ids AND grid AND attention_mask — key on all three
        # (cloned snapshots, so a reused/in-place-mutated input tensor is not a false cache hit).
        if (self._rope_ids is None
                or not torch.equal(input_ids, self._rope_key)
                or not torch.equal(grid, self._rope_grid_key)
                or not torch.equal(attn, self._rope_attn_key)):
            mmtt = (input_ids == img_tok).to(torch.int32)
            self._rope_ids, _ = m.model.get_rope_index(input_ids.cpu(), mm_token_type_ids=mmtt.cpu(),
                                                       image_grid_thw=grid.cpu(), video_grid_thw=None,
                                                       attention_mask=attn.cpu())
            self._rope_key = input_ids.clone()
            self._rope_grid_key = grid.clone()
            self._rope_attn_key = attn.clone()
            # Memoize prompt-constant position IDs and mask at stable RPU addresses.
            self._pos_rpu = self._rope_ids.to("rpu").contiguous()
            self._attn_rpu = attn.to("rpu").contiguous()
            # Bake interleaved tables from [3,1,S] position IDs; the helper
            # consumes [S,3]. Disabled mode leaves the tables unset.
            if self._partial_mrope:
                pos23 = self._rope_ids[:, 0, :].transpose(0, 1).contiguous()        # [S,3]
                cos_il, sin_il = build_interleaved_mrope_cos_sin(
                    pos23.to(torch.int64), head_dim=self._mrope_hd,
                    rope_theta=self._mrope_theta, mrope_section=self._mrope_section)
                self._rope_cos_il = cos_il.to("rpu").contiguous()
                self._rope_sin_il = sin_il.to("rpu").contiguous()
        # Scatter DeepStack features into the same contiguous image-token runs.
        # Member-owned buffers keep mutable-DMA addresses stable.
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
        execution_seq_len, planned_chunk_size = self._prefill_execution_plan(
            logical_seq_len
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
        cache = self._bb_cache                       # reuse the hoisted cache
        cache.reset_to_position(0)                    # prefill writes [0,S); overwrite-before-read
        sig = rpu_backend.graph.GraphSignature(op_id="gr00t_backbone",
                                         shapes=[seq_len, execution_seq_len,
                                                 text_model._rpu_text_hidden_size],
                                         dyn_dims=[text_model._rpu_text_num_layers,
                                                   text_model._rpu_text_deepstack_hash, 0,
                                                   chunk_policy_key(text_model._rpu_decoder_handle),
                                                   int(planned_chunk_size)],
                                         dtypes=[torch.float16])
        with text_model._rpu_text_graph_cache.capture(sig):
            raw, _ = _run_causal_decoder_forward(
                text_model, text_model._rpu_decoder_handle, input_ids=None, inputs_embeds=inputs_embeds,
                attention_mask=attn_rpu, position_ids=pos_rpu,
                past_key_values=cache, use_cache=True, return_dict=True, deepstack_dense_visual_embeds=dense,
                rope_cos_il=rope_cos_il, rope_sin_il=rope_sin_il)
        resolved_chunk_size = int(
            torch.ops.rpu.causal_decoder_get_resolved_chunk_size(
                text_model._rpu_decoder_handle
            )
        )
        if resolved_chunk_size != int(planned_chunk_size):
            raise RuntimeError(
                "GR00T prefill dry/forward chunk plan drift: "
                f"dry={planned_chunk_size}, forward={resolved_chunk_size}, "
                f"logical_len={seq_len}, execution_len={execution_seq_len}"
            )
        raw = _restore_gr00t_logical_prefill(
            raw, cache, seq_len, execution_seq_len
        )
        vars(text_model)["_rpu_last_execution_plan"] = {
            "stage": "prefill",
            "logical_len": int(seq_len),
            "execution_len": int(execution_seq_len),
            "chunk_size": resolved_chunk_size,
            "padding_rows": int(execution_seq_len - seq_len),
            "position": 0,
        }
        return raw, seq_len

    # ---- gr00t_vl_encoder (vlln + 4-layer vl_self_attention) -> vl_embeds [1,S,2048] ----
    @torch.no_grad()
    def _vl_encode(self, raw, S):
        raw_r = raw.to(torch.float16).to("rpu").contiguous()          # HELD (A input; post-RMSNorm)
        a_sig = rpu_backend.graph.GraphSignature(op_id="gr00t_vl_encoder", shapes=[S, _VLD],
                                           dyn_dims=[_VL_NL], dtypes=[torch.float16])
        with self._a_gc.capture(a_sig):
            vl = torch.ops.rpu.gr00t_vl_encoder_forward(
                self._a_handle, raw_r, self._a_cache.k_caches, self._a_cache.v_caches)
        return vl.to(torch.float16).contiguous()                      # HELD (denoise input)

    # ---- gr00t_denoise_unroll — encoders + 32-block DiT + norm_out + decoder + Euler ----
    @torch.no_grad()
    def _denoise(self, vl_r, inputs, S, seed):
        # Text/image MASK_2D subsets (cross-attn over the backbone tokens) are constant per
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
        # A local generator keeps the process-global RNG unchanged and makes
        # noise deterministic for the supplied seed.
        _gen = torch.Generator().manual_seed(int(seed))
        noise_r = _half_rpu(F.pad(torch.randn(1, _AH, _AD, generator=_gen), (0, _AD_PAD - _AD)))  # [1,40,144]
        d_sig = rpu_backend.graph.GraphSignature(op_id="gr00t_denoise_unroll", shapes=[_AH + 1, _H],
                                           dyn_dims=[_NL, S], dtypes=[torch.float16])
        with self._d_gc.capture(d_sig):
            act = torch.ops.rpu.gr00t_denoise_unroll_forward(
                self._d_handle, noise_r, state_r, self._d_cache.k_caches, self._d_cache.v_caches,
                vl_r, tmask_r, imask_r)
        return act.float().cpu()[:, :, :_AD].reshape(1, _AH, _AD)     # crop 144->132 (normalized action)

    # ---- full image→action (fully on-device) ----
    @torch.no_grad()
    def get_action(self, inputs, *, seed=0, num_steps=_NSTEPS):
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
        raw, S = self._backbone(inputs)
        vl_r = self._vl_encode(raw, S)
        action = self._denoise(vl_r, inputs, S, seed)
        if self._qwen_z1_plan_hash is not None:
            for cache in (self._qwen_z1_gc, self._a_gc, self._d_gc):
                if not cache.is_frozen():
                    cache.freeze()
        return action


# =============================================================================
# Build — backbone (Qwen3-VL 16L, RPU) + VL encoder + denoiser
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
    """Swizzle four VL blocks and bake vlln + inv_w_rms. Returns (handle, keepalive).

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
    """Build 32 DiT blocks plus per-step encoders, norm, decoder, and baked tables."""
    hr = _half_rpu

    def sc(w_lin, kpad=None, npad=None):       # -> (rpu weight, fp16 scale or None); single-core col-swizzle
        if kpad is not None and w_lin.size(1) < kpad:
            w_lin = F.pad(w_lin, (0, kpad - w_lin.size(1)))
        if npad is not None and w_lin.size(0) < npad:
            w_lin = F.pad(w_lin, (0, 0, 0, npad - w_lin.size(0)))
        # W8A16 (opt-in): int8 the glue Linear when K is int8-aligned (col-swizzle num_ele_32B=32, so
        # K%32; se1/ae1 K=144 fail it → stay fp16). Per-output-channel int8 + fp16 scale; the kernel
        # keys on (int8 weight + defined scale) → PARALLEL_LINEAR_W8A16. Default off = fp16.
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
    # Optional W8A16 applies per-output-channel int8 weights and fp16 scales to
    # q/k/v/o/ff1/ff2. AdaLN remains fp16; per-step glue is handled separately.
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
                               # AdaLN modulation dense (M=1 row GEMV): the last fp16
                               # parallel_linear in denoise; int8 reduces its DDR stream.
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
    """Build the fully-on-device RPU GR00T VLA with fused encoder and denoiser ops."""
    execution_config = _normalize_gr00t_execution(rpu_execution)
    qwen_z1_enabled = _qwen3vl_spm_z1_on()
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
        two_stage = os.environ.get("RPU_ALLREDUCE_TWOSTAGE")
        if two_stage is not None and two_stage.lower() in ("1", "true"):
            raise ValueError(
                "RPU_GR00T_QWEN3VL_SPM_Z1 uses the independently frozen "
                "single-stage all-reduce graph; set RPU_ALLREDUCE_TWOSTAGE=0"
            )
        os.environ["RPU_ALLREDUCE_TWOSTAGE"] = "0"
        ring_mode = os.environ.get("RPU_ALLREDUCE_RING")
        if ring_mode not in (None, "profile"):
            raise ValueError(
                "RPU_GR00T_QWEN3VL_SPM_Z1 freezes the ring graph; "
                "set RPU_ALLREDUCE_RING=profile"
            )
        os.environ["RPU_ALLREDUCE_RING"] = "profile"
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
        autotile = os.environ.get("RPU_LINEAR_AUTOTILE")
        if autotile is not None and autotile.lower() not in ("1", "true"):
            raise ValueError(
                "RPU_GR00T_QWEN3VL_SPM_Z1 requires RPU_LINEAR_AUTOTILE=1"
            )
        os.environ["RPU_LINEAR_AUTOTILE"] = "1"
        acc32 = os.environ.get("RPU_LINEAR_ACC32")
        if acc32 is not None and acc32.lower() in ("1", "true", "on"):
            raise ValueError(
                "RPU_GR00T_QWEN3VL_SPM_Z1 freezes the FP16/ACC16 graph; "
                "set RPU_LINEAR_ACC32=0"
            )
        os.environ["RPU_LINEAR_ACC32"] = "0"
        partial_mrope = os.environ.get("RPU_GR00T_PARTIAL_MROPE")
        if partial_mrope is not None and partial_mrope.lower() in (
            "1", "true"
        ):
            raise ValueError(
                "RPU_GR00T_QWEN3VL_SPM_Z1 freezes non-partial M-RoPE; set "
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
    # Enable the shared two-stage path before any graph is emitted. An explicit
    # enabled selector has priority over ring admission and preserves GR00T's
    # scatter+gather route.
    os.environ.setdefault("RPU_ALLREDUCE_TWOSTAGE", "1")
    # GR00T Vision uses SPM-resident 2D-RoPE tables.
    os.environ.setdefault("RPU_QWEN3VL_VISION_ROPE_SPM", "1")
    # Denoise KV-insert pads the per-block sequence to v16 alignment. Padded
    # cache rows are zeroed and SDPA reads the logical sequence length.
    os.environ.setdefault("RPU_GR00T_KVPAD16", "1")
    # Host-bake per-token interleaved M-RoPE tables once per prompt. The exact
    # composite profile keeps its independently frozen plan.
    os.environ.setdefault("RPU_GR00T_PARTIAL_MROPE", "1")
    # On a warm REPLAY, fast replay skips the host op-stream re-walk: backbone,
    # vl_encoder, and denoise graphs without post_fn use full skip; the Qwen3-VL
    # vision graph with fused-merger post_fn uses partial layer-body skip.
    # ``setdefault`` keeps an explicit environment or TOML override authoritative.
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
    forward_snapshots = ()
    qwen_z1_plan_hash = None
    gr00t_z2_plan_hash = None
    transferred = False
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
        # Backbone W8A16 (opt-in RPU_GR00T_W8A16, same switch as the DiT): int8 the 16-layer
        # Qwen3-VL text decoder's seven projection GEMMs. The
        # quant runs BEFORE the swizzle so convert_linear_weights_inplace swizzles int8 at dwidth=1;
        # scale_lists then route the install to causal_decoder_set_weights_w8a16.
        # The profile's numerical admission gate covers the resulting action output.
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
            execution_chunk_size=execution_config.get(
                "vision", {}
            ).get("chunk_size", "auto"),
        )
        # Run the patch-merger and DeepStack-merger GEMMs on RPU (default qwen3_vl keeps them CPU
        # fp32). The fp16 output feeds the denoiser.
        enable_qwen3_vl_vision_merger_on_device(vision_model)
        # Fold the patch merger into the Qwen3-VL vision graph (in-graph post_fn).
        # Gated on RPU_QWEN3VL_VISION_FUSED_MERGER (the same env the C++ post_fn reads) — register the
        # merger weights so merger_active() turns on; the deepstack mergers stay eager.
        #
        # ⚠️ SHARED SWITCH — MUST agree with the C++
        # `vision_fused_merger_enabled()` parser, which reads the same
        # variable as `e && s != "0" && s != "false"` (byte-exact). If C++ reads
        # ON while Python reads OFF, the merger runs TWICE — folded into the
        # graph and again eagerly — and the vision embedding is silently wrong
        # with no exception. `cpp_mirror` therefore refuses every spelling but
        # `1`/`0`, the only two the Python and C++ parsers read alike.
        if rpu_env_bool(
            "RPU_QWEN3VL_VISION_FUSED_MERGER",
            cpp_mirror="src/fused/rpu_qwen3vl_vision_model.cpp",
        ):
            register_qwen3vl_vision_fused_merger(vision_model)
        # Multi-image batch (per-image minibatch SDPA): env opt-in RPU_QWEN3VL_VISION_BATCH=1 packs
        # consecutive equal-size images into one vision forward. It uses the same
        # per-image math and is read in vision.py forward, with no setup needed here.
        # Left as an explicit environment opt-in
        # alongside RPU_GR00T_W8A16 / RPU_QWEN3VL_VISION_FUSED_MERGER (not setdefault'd).
        n_ds = len(getattr(cfg.vision_config, "deepstack_visual_indexes", []))
        install_qwen3_vl_text_for_rpu(
            text_model,
            text_config=cfg.text_config,
            vision_config=cfg.vision_config,
            deepstack_lang_layers=list(range(n_ds)),
            enable_deepstack=True,
            scale_lists=bb_scale_lists,
            execution_config=execution_config,
        )

        gw = _ckpt_reader(ckpt_path)
        a_handle, a_keep = _build_vl_encoder(gw, w_rms)
        d_handle, d_keep = _build_denoise(gw, embodiment_id)

        if qwen_z1_enabled:
            if n_ds != _QWEN_Z1_DEEPSTACK_COUNT:
                raise ValueError(
                    "GR00T Qwen3-VL SPM Z1 requires exactly three DeepStack "
                    f"routes, got {n_ds}"
                )
            # Freeze the encoder/denoiser S=82 persistent-layout requirements before the
            # Qwen composite captures the process-wide persistent top.  The
            # ordinary component forwards remain legal because no Z2 lease is bound.
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
            _unprepare_gr00t_spm_z2(
                a_handle=a_handle,
                d_handle=d_handle,
                plan_hash=gr00t_z2_plan_hash,
            )
            gr00t_z2_plan_hash = None

        runtime = Gr00tN1d7VLA(
            backbone=model, cfg=cfg, text_model=text_model,
            vision_model=vision_model, a_handle=a_handle, a_keep=a_keep,
            d_handle=d_handle, d_keep=d_keep,
            image_token_id=model.config.image_token_id,
            execution_config=execution_config,
            embodiment_id=embodiment_id,
            qwen_z1_plan_hash=qwen_z1_plan_hash,
        )
        transferred = True
        try:
            model._rpu_swizzled = True
        except BaseException:
            runtime.close()
            raise
        return runtime
    finally:
        if not transferred:
            _cleanup_gr00t_partial_build(
                a_handle=a_handle,
                d_handle=d_handle,
                model=model,
                text_model=text_model,
                vision_model=vision_model,
                forward_snapshots=forward_snapshots,
                qwen_z1_plan_hash=qwen_z1_plan_hash,
                gr00t_z2_plan_hash=gr00t_z2_plan_hash,
            )
