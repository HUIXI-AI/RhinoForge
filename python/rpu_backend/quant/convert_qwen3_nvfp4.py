"""NVFP4 (E2M1 weight, E4M3 block scale, FP32 tensor scale) for dense Qwen3 text.

The cold on-install recipe quantizes the seven decoder projections of the
official FP16 source once. Each projection becomes a packed uint8 [N, K/2]
weight plus a logical-shape uint8 [K/16, N] block-scale payload in the tiled
controller stripe; one FP32 tensor scale per projection and layer is returned
in seven padded tables for `causal_decoder_set_weights_nvfp4`. The output head,
embedding and norms stay FP16.

A six-lane MLP whose width is not 384-aligned (4B I9728, 14B I17408) is
zero-padded before quantization to the 384-aligned physical width
(``decoder_nvfp4_mlp_intermediate_size``): Gate/Up gain zero output rows,
Down gains zero input columns. Zero blocks quantize to zero codes and scales,
and the tensor scale (the projection amax) is unchanged. Linear features become
physical; model config stays logical. Four and eight lanes, and the already
aligned 0.6B/1.7B/8B widths, keep logical storage. The projections carry their
own swizzle and must be skipped by `convert_linear_weights_inplace` (pitfall P1).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rpu_backend.quant.convert_qwen3_w4a16 import _PROJECTIONS, _get
from rpu_backend.quant.nvfp4_pack import (
    quantize_nvfp4, swizzle_nvfp4_scales, swizzle_pack_nvfp4,
)
from rpu_backend.runtime.topology import decoder_nvfp4_mlp_intermediate_size


def _check_nvfp4_shape(name: str, n: int, k: int, partition: int, num_cores: int) -> None:
    if partition == 1:
        ok, rule = n % (16 * num_cores) == 0 and k % 64 == 0, f"col: N%{16 * num_cores}==0 and K%64==0"
    else:
        ok, rule = n % 16 == 0 and k % (64 * num_cores) == 0, f"row: N%16==0 and K%{64 * num_cores}==0"
    if not ok:
        raise ValueError(f"{name}: shape [N={n}, K={k}] violates the NVFP4 swizzle rule ({rule})")


def pad_nvfp4_tensor_scales(values) -> torch.Tensor:
    """One projection family's per-layer FP32 scales, padded with 1.0 for 32-byte loads."""
    if not values or any(
            not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or value.dtype != torch.float32 or value.ndim != 0 for value in values):
        raise ValueError("NVFP4 tensor scales must be CPU FP32 scalars, one per layer")
    scales = torch.stack(values)
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError("NVFP4 tensor scales must be finite and positive")
    return torch.nn.functional.pad(scales, (0, (-len(values)) % 8), value=1.0).contiguous()


def _physical_shape(dotted: str, linear: nn.Linear, partition: int, mlp_tp: int) -> tuple[int, int]:
    n, k = linear.weight.shape
    if not dotted.startswith("mlp."):
        return n, k
    logical = k if dotted.endswith("down_proj") else n
    physical = decoder_nvfp4_mlp_intermediate_size(logical, mlp_tp)
    if physical == logical:
        return n, k
    return (physical, k) if partition == 1 else (n, physical)


def convert_qwen3_to_nvfp4_(model, *, topology) -> tuple[torch.Tensor, ...]:
    """Quantize every decoder projection of a loaded FP16 Qwen3 to NVFP4 in place.

    Returns the seven CPU FP32 tensor-scale tables in `_PROJECTIONS` order.
    Every physical shape is checked before the first projection is replaced.
    """
    layers = model.model.layers
    for i, layer in enumerate(layers):
        for dotted, partition in _PROJECTIONS:
            linear = _get(layer, dotted)
            weight = linear.weight
            if weight.dtype != torch.float16 or weight.device.type != "cpu":
                raise TypeError(f"layer{i}.{dotted}: NVFP4 requires original CPU FP16 weights")
            if tuple(weight.shape) != (linear.out_features, linear.in_features):
                raise ValueError(f"layer{i}.{dotted}: NVFP4 requires the logical FP16 source shape")
            cores = topology.mlp_tp if dotted.startswith("mlp.") else topology.attn_tp
            _check_nvfp4_shape(f"layer{i}.{dotted}",
                               *_physical_shape(dotted, linear, partition, topology.mlp_tp),
                               partition, cores)
    tensor_scales = [[] for _ in _PROJECTIONS]
    for layer in layers:
        for index, (dotted, partition) in enumerate(_PROJECTIONS):
            linear = _get(layer, dotted)
            cores = topology.mlp_tp if dotted.startswith("mlp.") else topology.attn_tp
            n, k = _physical_shape(dotted, linear, partition, topology.mlp_tp)
            source = linear.weight.detach()
            if (n, k) != tuple(source.shape):
                source = torch.nn.functional.pad(
                    source, (0, k - source.shape[1], 0, n - source.shape[0]))
            codes, scales, tensor_scale = quantize_nvfp4(source)
            linear.weight = nn.Parameter(
                swizzle_pack_nvfp4(codes, partition, cores).reshape(n, k // 2), requires_grad=False)
            linear.out_features, linear.in_features = n, k
            # A buffer, not an attribute, so model.to('rpu') moves the scale too.
            if hasattr(linear, "weight_scale"):
                del linear.weight_scale
            linear.register_buffer("weight_scale", swizzle_nvfp4_scales(scales, partition, cores),
                                   persistent=False)
            tensor_scales[index].append(tensor_scale)
    return tuple(pad_nvfp4_tensor_scales(values) for values in tensor_scales)
