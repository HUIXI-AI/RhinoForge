"""On-install W8A16 / G32 W4A16 for the exact DINOv3 ViT-B/16 encoder.

The six projections of every encoder layer are quantized once from the FP16
checkpoint, after the eight-core head padding (12 -> 16 heads, zero rows give
zero codes) and before the swizzle:

* ``w8a16``: natural int8 ``[N, K]`` plus one FP16 per-channel scale buffer
  (symmetric, ``quantize_linear_per_channel``); the ordinary byte-width swizzle
  then lays them out like any other Linear.
* ``w4a16``: packed uint8 ``[N, K/2]`` with a controller-striped G32 scale
  (``pack_linear_int4_``); these leaves carry their own per-core layout and
  must be skipped by the generic swizzle (pitfall P1).

Patch embedding, norms, LayerScale and biases stay FP16. Every projection is
checked before the first one is replaced.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rpu_backend.quant._common import quantize_linear_per_channel
from rpu_backend.quant.convert_qwen3_w4a16 import (
    DEFAULT_GROUP_SIZE, _check_int4_shape, _get, pack_linear_int4_,
)

# (attribute path on the encoder layer, partition) — partition must match what
# build_layer_subgraph passes to the linear launcher: 1 = col, 0 = row.
PROJECTIONS: tuple[tuple[str, int], ...] = (
    ("attention.q_proj", 1),
    ("attention.k_proj", 1),
    ("attention.v_proj", 1),
    ("attention.o_proj", 0),
    ("mlp.up_proj", 1),
    ("mlp.down_proj", 0),
)

W4_SKIP_LINEAR_NAMES = frozenset(p.rsplit(".", 1)[-1] for p, _ in PROJECTIONS)

ON_INSTALL_QUANTIZATION = ("w8a16", "w4a16")


def check_dinov3_quant_source(layers, quantization: str) -> None:
    """Admit only original, unswizzled CPU FP16 projections; reads no values."""
    if quantization not in ON_INSTALL_QUANTIZATION:
        raise ValueError(f"unknown DINOv3 on-install recipe {quantization!r}")
    for i, layer in enumerate(layers):
        for dotted, _ in PROJECTIONS:
            linear = _get(layer, dotted)
            weight = linear.weight
            if (not isinstance(linear, nn.Linear) or weight.dtype != torch.float16
                    or weight.device.type != "cpu" or weight.dim() != 2
                    or not weight.is_contiguous()
                    or getattr(linear, "weight_scale", None) is not None
                    or hasattr(linear, "_rpu_linear_partition")):
                raise ValueError(
                    f"layer{i}.{dotted}: DINOv3 {quantization} requires the original "
                    "unswizzled CPU FP16 weight")


def quantize_dinov3_projections_(layers, quantization: str, *, num_cores: int) -> None:
    """Quantize every (head-padded) encoder projection in place; all-or-nothing."""
    check_dinov3_quant_source(layers, quantization)
    for i, layer in enumerate(layers):
        for dotted, partition in PROJECTIONS:
            weight = _get(layer, dotted).weight
            if quantization == "w4a16":
                _check_int4_shape(f"layer{i}.{dotted}", *weight.shape, partition, num_cores)
    for i, layer in enumerate(layers):
        for dotted, partition in PROJECTIONS:
            linear = _get(layer, dotted)
            if quantization == "w4a16":
                pack_linear_int4_(linear, partition, name=f"layer{i}.{dotted}",
                                  group_size=DEFAULT_GROUP_SIZE, num_cores=num_cores)
                continue
            weight, scale = quantize_linear_per_channel(linear.weight.detach(), bits=8)
            linear.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
            linear.out_features, linear.in_features = weight.shape
            # A buffer, not an attribute, so model.to('rpu') moves the scale too.
            linear.register_buffer("weight_scale", scale.contiguous(), persistent=False)
