"""On-install W8A16 for the exact Llama-3.2-1B decoder.

The seven decoder projections become natural int8 [N, K] weights with one FP16
per-channel scale buffer each (symmetric, ``quantize_linear_per_channel``). The
caller then pads a six-lane MLP and applies the ordinary byte-width swizzle, as
for a W8A16 Qwen3 checkpoint. Packed G32 W4A16 and NVFP4 reuse the dense Qwen3
converters, which walk the same ``model.model.layers`` projections. The tied
output head, embedding and norms stay FP16.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rpu_backend.quant._common import quantize_linear_per_channel
from rpu_backend.quant.convert_qwen3_w4a16 import _PROJECTIONS, _get


def quantize_llama_w8a16_(model) -> None:
    """Quantize every decoder projection of a loaded FP16 Llama to W8A16 in place.

    Every projection is checked before the first one is replaced.
    """
    layers = model.model.layers
    for i, layer in enumerate(layers):
        for dotted, _ in _PROJECTIONS:
            linear = _get(layer, dotted)
            weight = linear.weight
            if (weight.dtype != torch.float16 or weight.device.type != "cpu"
                    or tuple(weight.shape) != (linear.out_features, linear.in_features)
                    or linear.bias is not None
                    or getattr(linear, "weight_scale", None) is not None):
                raise ValueError(f"layer{i}.{dotted}: W8A16 requires the original CPU FP16 weight")
    for layer in layers:
        for dotted, _ in _PROJECTIONS:
            linear = _get(layer, dotted)
            weight, scale = quantize_linear_per_channel(linear.weight, bits=8)
            linear.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
            # A buffer, not an attribute, so model.to('rpu') moves the scale too.
            linear.register_buffer("weight_scale", scale.contiguous(), persistent=False)
