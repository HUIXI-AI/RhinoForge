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
