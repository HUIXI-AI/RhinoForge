"""Exact Expert-only NVFP4 preparation; no device state or model execution."""
from dataclasses import dataclass

import torch

from rpu_backend.quant.nvfp4_pack import (
    quantize_nvfp4, swizzle_nvfp4_scales, swizzle_pack_nvfp4,
)


@dataclass(frozen=True)
class HyVlaExpertNvfp4Metadata:
    """Quantization identity, separate from native tensor-list arguments."""

    enabled: bool = True
    method: str = "nvfp4a16"
    packing: str = "tiled_deinterleaved_k32"
    scale_packing: str = "controller_striped_k16"
    accumulation: str = "acc16"
    source_rounding: str = "fp16_before_quantize"
    resident_layers: int = 0


def quantize_expert_nvfp4_cpu(
    weight: torch.Tensor, partition: int, num_cores: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Logical CPU source -> packed codes, striped FP8 scales, FP32 scalar.

    Keep the same source-to-Half step as the W8 baseline. KV duplication belongs
    to the caller and precedes quantization. Returned bytes use the canonical
    generic NVFP4 physical ABI and must not undergo another weight transform.
    """
    # TP8 duplicates K/V to 1024 rows; the reduced four-core child keeps KV4.
    kv_rows = 1024 if num_cores == 8 else 512
    shapes = {(2048, 1024), (kv_rows, 1024)} if partition == 1 else {(1024, 2048)}
    if (not isinstance(weight, torch.Tensor) or weight.device.type != "cpu" or
            not weight.is_floating_point() or tuple(weight.shape) not in shapes or
            partition not in (0, 1) or num_cores not in (4, 8)):
        raise ValueError(
            "HyVLA NVFP4 requires an exact CPU Expert TP8 or four-core projection")
    codes, block_scale, tensor_scale = quantize_nvfp4(weight.half().contiguous())
    n, k = weight.shape
    packed = swizzle_pack_nvfp4(codes, partition, num_cores).reshape(n, k // 2)
    striped = swizzle_nvfp4_scales(block_scale, partition, num_cores)
    return packed.contiguous(), striped.contiguous(), tensor_scale.float().contiguous()


def pack_expert_gateup_nvfp4_installed(weight: torch.Tensor) -> torch.Tensor:
    """Installed TP8 DDR codes -> [8,131072] resident bytes, without requantizing.

    Nibbles are already deinterleaved K32 pairs. Only move the controller/core
    dimension outside [N16 block,K64 block,core,512B]; do not deinterleave K again.
    """
    if (not isinstance(weight, torch.Tensor) or weight.dtype != torch.uint8 or
            tuple(weight.shape) != (2048, 512)):
        raise ValueError("HyVLA NVFP4 resident Gate/Up requires installed uint8 [2048,512]")
    return (weight.cpu().contiguous().reshape(16, 16, 8, 512)
            .permute(2, 0, 1, 3).contiguous().reshape(8, 131072))
