"""Pi0.5 W8A16 decoder validation and scale-list helpers."""
from __future__ import annotations

import torch
import torch.nn as nn

from rpu_backend.api.errors import RPUBackendError
from rpu_backend.runtime.weights import NUM_CORES


_W8A16_PROJ_PATHS = (
    ("self_attn", "q_proj"),
    ("self_attn", "k_proj"),
    ("self_attn", "v_proj"),
    ("self_attn", "o_proj"),
    ("mlp", "gate_proj"),
    ("mlp", "up_proj"),
    ("mlp", "down_proj"),
)


def _iter_decoder_projections(decoder, *, label: str):
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise RPUBackendError(
            f"Pi05 W8A16 validation failed: {label} has no .layers"
        )
    for layer_idx, layer in enumerate(layers):
        for parent_name, proj_name in _W8A16_PROJ_PATHS:
            parent = getattr(layer, parent_name, None)
            module = getattr(parent, proj_name, None)
            if module is None or not hasattr(module, "weight"):
                raise RPUBackendError(
                    "Pi05 W8A16 validation failed: missing "
                    f"{label}.layers[{layer_idx}].{parent_name}.{proj_name}.weight"
                )
            yield layer_idx, parent_name, proj_name, module


def detect_and_validate_gemma_decoder_w8a16(
    decoder,
    *,
    require_rpu: bool = False,
    label: str = "decoder",
) -> bool:
    """Return True for all-quantized decoder projections; reject partial loads."""
    projections = list(_iter_decoder_projections(decoder, label=label))
    quant_count = sum(
        1 for _, _, _, module in projections
        if module.weight.dtype in (torch.int8, torch.uint8)
    )
    if quant_count == 0:
        return False
    if quant_count != len(projections):
        raise RPUBackendError(
            f"Pi05 quant validation failed: partial {label} quantization "
            f"quantized={quant_count}/{len(projections)} (each projection must be "
            "int8(W8A16) or uint8(packed-INT4))"
        )

    for layer_idx, parent_name, proj_name, module in projections:
        scale = getattr(module, "weight_scale", None)
        fq_name = f"{label}.layers[{layer_idx}].{parent_name}.{proj_name}"
        if scale is None or scale.dtype != torch.float16:
            raise RPUBackendError(
                f"Pi05 quant validation failed: quantized {fq_name}.weight "
                f"({module.weight.dtype}) requires fp16 weight_scale"
            )
        if scale.dim() != 1 or scale.numel() != module.weight.size(0):
            raise RPUBackendError(
                f"Pi05 quant validation failed: {fq_name}.weight_scale shape "
                f"{tuple(scale.shape)} incompatible with weight output dim "
                f"{module.weight.size(0)}"
            )
        if require_rpu and (
            module.weight.device.type != "rpu" or scale.device.type != "rpu"
        ):
            raise RPUBackendError(
                f"Pi05 quant validation failed: {fq_name}.weight and "
                "weight_scale must be on RPU before set_weights"
            )
        if not module.weight.is_contiguous() or not scale.is_contiguous():
            raise RPUBackendError(
                f"Pi05 quant validation failed: {fq_name}.weight and "
                "weight_scale must be contiguous"
            )
    return True


def detect_and_validate_pi05_expert_w8a16(expert, *, require_rpu: bool = False) -> bool:
    """Return True for all-quantized Pi0.5 expert projections; reject partial loads."""
    return detect_and_validate_gemma_decoder_w8a16(
        expert, require_rpu=require_rpu, label="expert"
    )


def _set_or_replace_buffer(module, name: str, tensor: torch.Tensor) -> None:
    if name in module._buffers:
        module._buffers[name] = tensor
        return
    if hasattr(module, name):
        delattr(module, name)
    module.register_buffer(name, tensor)


def replicate_pi05_decoder_kv_heads_for_attention_tp(
    decoder,
    *,
    effective_num_kv_heads: int = NUM_CORES,
    label: str = "decoder",
) -> int:
    """Replicate K/V projection heads so Pi0.5 attention can run on 8 cores.

    Pi0.5 Gemma uses MQA (`num_key_value_heads=1`) with eight Q heads. The RPU
    attention path partitions Q/K/V/O by attention cores, so using
    `attn_tp=1` serializes all Q heads on core 0. This helper preserves MQA
    semantics by materializing identical effective K/V heads before the normal
    weight swizzle, then returns the effective KV-head count to pass to C++.
    """
    config = getattr(decoder, "config", None)
    layers = getattr(decoder, "layers", None)
    if config is None or layers is None:
        raise RPUBackendError(
            f"Pi05 KV replication failed: {label} requires .config and .layers"
        )

    original_nkv = int(config.num_key_value_heads)
    num_q_heads = int(config.num_attention_heads)
    head_dim = int(config.head_dim)
    effective_nkv = int(effective_num_kv_heads)
    if original_nkv <= 0 or effective_nkv <= 0 or head_dim <= 0:
        raise RPUBackendError(
            f"Pi05 KV replication failed for {label}: invalid "
            f"nkv={original_nkv}, effective_nkv={effective_nkv}, "
            f"head_dim={head_dim}"
        )
    if effective_nkv < original_nkv or effective_nkv % original_nkv != 0:
        raise RPUBackendError(
            f"Pi05 KV replication failed for {label}: effective_nkv="
            f"{effective_nkv} must be a multiple of config num_key_value_heads="
            f"{original_nkv}"
        )
    if num_q_heads % effective_nkv != 0:
        raise RPUBackendError(
            f"Pi05 KV replication failed for {label}: num_attention_heads="
            f"{num_q_heads} must be divisible by effective_nkv={effective_nkv}"
        )

    existing_effective = getattr(decoder, "_rpu_effective_num_kv_heads", None)
    target_rows = effective_nkv * head_dim
    original_rows = original_nkv * head_dim
    replication = effective_nkv // original_nkv

    if replication == 1:
        decoder._rpu_effective_num_kv_heads = effective_nkv
        return effective_nkv

    for layer_idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        for proj_name in ("k_proj", "v_proj"):
            proj = getattr(attn, proj_name, None)
            if not isinstance(proj, nn.Linear):
                raise RPUBackendError(
                    f"Pi05 KV replication failed: {label}.layers[{layer_idx}]"
                    f".self_attn.{proj_name} must be nn.Linear, got "
                    f"{type(proj).__name__}"
                )
            weight = proj.weight.detach()
            if weight.dim() != 2:
                raise RPUBackendError(
                    f"Pi05 KV replication failed: {label}.layers[{layer_idx}]"
                    f".self_attn.{proj_name}.weight must be 2D, got "
                    f"{weight.dim()}D"
                )
            hidden = int(weight.size(1))
            rows = int(weight.size(0))
            if rows == target_rows and existing_effective == effective_nkv:
                continue
            if rows != original_rows:
                raise RPUBackendError(
                    f"Pi05 KV replication failed: {label}.layers[{layer_idx}]"
                    f".self_attn.{proj_name}.weight rows={rows}, expected "
                    f"raw rows={original_rows} before swizzle or target rows="
                    f"{target_rows} for an already-replicated decoder"
                )

            with torch.no_grad():
                w_view = weight.view(original_nkv, head_dim, hidden)
                w_rep = w_view.repeat_interleave(replication, dim=0).contiguous()
                proj.weight = nn.Parameter(
                    w_rep.view(target_rows, hidden).contiguous(),
                    requires_grad=False,
                )
            proj.out_features = target_rows

            if proj.bias is not None:
                bias = proj.bias.detach()
                if bias.numel() != original_rows:
                    raise RPUBackendError(
                        f"Pi05 KV replication failed: {label}.layers[{layer_idx}]"
                        f".self_attn.{proj_name}.bias size={bias.numel()}, "
                        f"expected {original_rows}"
                    )
                with torch.no_grad():
                    b_view = bias.view(original_nkv, head_dim)
                    b_rep = b_view.repeat_interleave(replication, dim=0).contiguous()
                    proj.bias = nn.Parameter(
                        b_rep.view(target_rows).contiguous(),
                        requires_grad=False,
                    )

            scale = getattr(proj, "weight_scale", None)
            if proj.weight.dtype == torch.int8 and scale is None:
                raise RPUBackendError(
                    f"Pi05 KV replication failed: int8 {label}.layers[{layer_idx}]"
                    f".self_attn.{proj_name}.weight requires fp16 weight_scale"
                )
            if scale is not None:
                scale_detached = scale.detach()
                if scale_detached.dim() != 1 or scale_detached.numel() != original_rows:
                    raise RPUBackendError(
                        f"Pi05 KV replication failed: {label}.layers[{layer_idx}]"
                        f".self_attn.{proj_name}.weight_scale shape="
                        f"{tuple(scale_detached.shape)}, expected "
                        f"({original_rows},)"
                    )
                s_view = scale_detached.to(dtype=torch.float16).view(
                    original_nkv, head_dim
                )
                s_rep = s_view.repeat_interleave(replication, dim=0).contiguous()
                _set_or_replace_buffer(
                    proj, "weight_scale", s_rep.view(target_rows).contiguous()
                )

    decoder._rpu_effective_num_kv_heads = effective_nkv
    return effective_nkv


def gemma_decoder_scale_lists(
    decoder,
    *,
    require_rpu: bool = False,
    label: str = "decoder",
):
    """Return q/k/v/o/gate/up/down scale lists, or seven empty lists for fp16."""
    if not detect_and_validate_gemma_decoder_w8a16(
        decoder, require_rpu=require_rpu, label=label
    ):
        return ([], [], [], [], [], [], [])

    layers = decoder.layers
    return (
        [layers[i].self_attn.q_proj.weight_scale for i in range(len(layers))],
        [layers[i].self_attn.k_proj.weight_scale for i in range(len(layers))],
        [layers[i].self_attn.v_proj.weight_scale for i in range(len(layers))],
        [layers[i].self_attn.o_proj.weight_scale for i in range(len(layers))],
        [layers[i].mlp.gate_proj.weight_scale for i in range(len(layers))],
        [layers[i].mlp.up_proj.weight_scale for i in range(len(layers))],
        [layers[i].mlp.down_proj.weight_scale for i in range(len(layers))],
    )


def pi05_expert_scale_lists(expert, *, require_rpu: bool = False):
    """Return q/k/v/o/gate/up/down scale lists, or seven empty lists for fp16."""
    return gemma_decoder_scale_lists(
        expert, require_rpu=require_rpu, label="expert"
    )
