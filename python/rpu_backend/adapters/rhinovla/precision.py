"""Cold W8A16 preparation for RhinoVLA's Qwen3-VL text prefix."""
from __future__ import annotations

import torch

from rpu_backend.quant._common import quantize_linear_per_channel


TEXT_PROJECTIONS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


def quantize_text_prefix(model):
    """Stage all seven projections before committing natural INT8 weights.

    The caller owns half -> quantize -> byte-width swizzle -> RPU migration.
    Embedding, norms and rotary buffers retain their floating point precision.
    """
    cfg = model.config
    h, inter = int(cfg.hidden_size), int(cfg.intermediate_size)
    q = int(cfg.num_attention_heads) * int(cfg.head_dim)
    kv = int(cfg.num_key_value_heads) * int(cfg.head_dim)
    shapes = ((q, h), (kv, h), (kv, h), (h, q), (inter, h), (inter, h), (h, inter))
    if len(model.layers) != int(cfg.num_hidden_layers):
        raise ValueError("RhinoVLA text W8A16 requires every configured layer")
    prepared = []
    for index, layer in enumerate(model.layers):
        for name, shape in zip(TEXT_PROJECTIONS, shapes, strict=True):
            linear = layer.get_submodule(name)
            weight = linear.weight
            if (not isinstance(linear, torch.nn.Linear) or linear.bias is not None
                    or weight.device.type != "cpu" or weight.dtype != torch.float16
                    or tuple(weight.shape) != shape or hasattr(linear, "weight_scale")
                    or not torch.isfinite(weight).all()):
                raise ValueError(f"RhinoVLA text W8A16 requires fresh finite FP16 CPU layer {index}.{name} {shape}")
            quantized, scale = quantize_linear_per_channel(weight)
            if not (torch.isfinite(scale).all() and (scale > 0).all()):
                raise ValueError("RhinoVLA text W8A16 scale must be finite and positive")
            prepared.append((linear, quantized.contiguous(), scale.contiguous()))
    if {id(module) for module in model.modules() if isinstance(module, torch.nn.Linear)} != {
        id(linear) for linear, _, _ in prepared
    }:
        raise ValueError("RhinoVLA text W8A16 found an uncovered Linear")
    for linear, weight, scale in prepared:
        linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        linear.register_buffer("weight_scale", scale)


def text_scale_lists(model):
    """Return the actual natural scales in the native seven-projection order."""
    groups = tuple([layer.get_submodule(name) for layer in model.layers]
                   for name in TEXT_PROJECTIONS)
    for group in groups:
        for linear in group:
            scale = getattr(linear, "weight_scale", None)
            if (linear.weight.dtype != torch.int8 or scale is None
                    or scale.dtype != torch.float16 or scale.device != linear.weight.device
                    or tuple(scale.shape) != (linear.out_features,)
                    or not scale.is_contiguous()):
                raise ValueError("RhinoVLA text requires complete INT8 / FP16 scale owners")
    return tuple([linear.weight_scale for linear in group] for group in groups)


def bind_pipeline_linear_accumulation(text_model, vision_model, *, prefill, vision):
    """Bind existing text/vision Linear arithmetic before the first forward.

    Action, patch embedding and cold AdaRMS projections keep their existing
    arithmetic. The pipeline cold snapshot owns both booleans for its lifetime.
    """
    if type(prefill) is not bool or type(vision) is not bool:
        raise TypeError("RhinoVLA linear_acc32 must be bool")
    torch.ops.rpu.qwen3vl_vision_set_linear_acc32(
        vision_model._rpu_vision_handle, vision)
    torch.ops.rpu.causal_decoder_set_linear_acc32(
        text_model._rpu_decoder_handle, prefill)
    vision_model._rpu_vision_linear_acc32 = vision


def full_w8_inventory(runtime):
    """Validate actual installed full-pipeline W8 owners; never infer from dtype labels."""
    groups = {}

    def record(label, weights, scales):
        weights, scales = list(weights), list(scales)
        if not weights or len(weights) != len(scales):
            raise ValueError(f"RhinoVLA full W8 requires complete {label} owners")
        for weight, scale in zip(weights, scales, strict=True):
            if (not isinstance(weight, torch.Tensor) or weight.dtype != torch.int8
                    or weight.device.type != "rpu" or weight.ndim != 2
                    or not isinstance(scale, torch.Tensor) or scale.dtype != torch.float16
                    or scale.device != weight.device or scale.ndim != 1
                    or scale.numel() != weight.shape[0]
                    or not weight.is_contiguous() or not scale.is_contiguous()):
                raise ValueError(f"RhinoVLA full W8 requires INT8 / FP16 scale storage for {label}")
        groups[label] = {
            "physical_matrices": len(weights),
            "weight_bytes": sum(w.numel() * w.element_size() for w in weights),
            "scale_bytes": sum(s.numel() * s.element_size() for s in scales),
        }

    text = runtime.text_model
    if len(text.layers) != int(text.config.num_hidden_layers):
        raise ValueError("RhinoVLA full W8 requires every text layer")
    text_scales = text_scale_lists(text)
    record("text_prefix", (layer.get_submodule(name).weight for name in TEXT_PROJECTIONS
                           for layer in text.layers),
           (scale for group in text_scales for scale in group))
    visual = runtime.vision_model
    if (getattr(visual, "_rpu_vision_w8a16", False) is not True
            or getattr(visual, "_rpu_vision_fused_merger_w8a16", False) is not True):
        raise ValueError("RhinoVLA full W8 requires quantized vision blocks and fused mergers")
    vweights, vscales = visual._rpu_vision_projection_weights, visual._rpu_vision_projection_scales
    if (len(vweights) != 6 or len(vscales) != 6
            or any(len(group) != len(visual.blocks) for group in (*vweights, *vscales))):
        raise ValueError("RhinoVLA full W8 requires every vision projection")
    record("vision_blocks", (w for group in vweights for w in group),
           (s for group in vscales for s in group))
    record("patch_projection", [visual._rpu_vision_patch_embed_w_rpu],
           [visual._rpu_vision_patch_embed_scale])
    mergers = [visual.merger, *visual.deepstack_merger_list]
    record("vision_mergers", (w for merger in mergers for w in (
        merger._rpu_merger_fc1_w_rpu, merger._rpu_merger_fc2_fused_w_rpu)),
        (s for merger in mergers for s in (
            merger._rpu_merger_fc1_scale_rpu, merger._rpu_merger_fc2_scale_rpu)))
    loop = runtime._denoise_loop_meta
    if not isinstance(loop, dict) or len(loop.get("io_scales", ())) != 6:
        raise ValueError("RhinoVLA full W8 requires six action IO scales")
    io_names = ("action_in_w", "state_w", "state_mask_w", "action_mask_w", "action_out_w")
    record("action_io", (loop[name] for name in io_names),
           [*loop["io_scales"][:4], loop["io_scales"][5]])
    cold_weights, cold_scales, cold_biases = loop["adarms_cold_owners"]
    if (getattr(runtime.er, "_rpu_full_w8a16", False) is not True
            or len(cold_weights) != len(runtime.er.layers) + 1
            or len(cold_biases) != len(cold_weights)):
        raise ValueError("RhinoVLA full W8 requires every AdaRMS owner including final norm")
    record("adarms_cold", cold_weights, cold_scales)
    for bias, scale in zip(cold_biases, cold_scales, strict=True):
        if (not isinstance(bias, torch.Tensor) or bias.dtype != torch.float16
                or bias.device != scale.device or bias.shape != scale.shape
                or not bias.is_contiguous()):
            raise ValueError("RhinoVLA full W8 requires FP16 AdaRMS biases")
    groups["adarms_cold"]["logical_projections"] = 2 * len(runtime.er.layers) + 1
    return groups
