"""NVFP4A16 weight quantization and packing for Rhino parallel linear.

Contract:
  weight       FP4 E2M1, logical [N,K], packed low-nibble first after TP swizzle
  block scale  FP8 E4M3FN [K/16,N] (N contiguous)
  tensor scale FP32 scalar; projection-family arrays are preloaded to SPM
"""
from __future__ import annotations

import numpy as np
import torch

_FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
_NUM_ELE_32B_FP4 = 64


def fp4_e2m1_decode(codes: torch.Tensor) -> torch.Tensor:
    """Decode uint8 low-nibble E2M1 codes to fp32."""
    code = codes.to(torch.uint8) & 0x0F
    magnitude = _FP4_VALUES.to(code.device)[(code & 0x07).to(torch.long)]
    return torch.where((code & 0x08) != 0, -magnitude, magnitude)


def fp4_e2m1_encode(values: torch.Tensor) -> torch.Tensor:
    """Encode fp32 to E2M1 with saturation and round-to-nearest-even."""
    x = values.detach().to(torch.float32)
    sign = torch.signbit(x).to(torch.uint8) << 3
    v = x.abs().clamp_max_(6.0)
    code = torch.zeros_like(v, dtype=torch.uint8)
    code[(v > 0.25) & (v < 0.75)] = 1
    code[(v >= 0.75) & (v <= 1.25)] = 2
    code[(v > 1.25) & (v < 1.75)] = 3
    code[(v >= 1.75) & (v <= 2.5)] = 4
    code[(v > 2.5) & (v < 3.5)] = 5
    code[(v >= 3.5) & (v <= 5.0)] = 6
    code[v > 5.0] = 7
    return code | sign


def quantize_nvfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize [N,K] weight.

    Returns ``(fp4_codes[N,K], fp8_scale_codes[K/16,N], tensor_scale[])``.
    The dequantized value is
    ``decode(fp4) * decode(fp8_scale[g,n]) * tensor_scale``.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected [N,K], got {tuple(weight.shape)}")
    n, k = weight.shape
    if k % 16 != 0:
        raise ValueError(f"NVFP4 requires K divisible by 16, got K={k}")

    w = weight.detach().to(torch.float32).contiguous()
    groups = k // 16
    blocked = w.view(n, groups, 16)
    block_max = blocked.abs().amax(dim=2)
    # Standard NVFP4 two-level scaling. The canonical Rhino kernel reads this
    # value from an FP32 projection-family array preloaded into SPM.
    tensor_scale = (w.abs().amax() / (6.0 * 448.0)).clamp_min_(2.0 ** -24)
    tensor_scale = tensor_scale.to(torch.float32)
    inner = (
        block_max / (6.0 * tensor_scale.to(torch.float32))
    ).clamp_(0.0, 448.0)
    fp8 = inner.to(torch.float8_e4m3fn)
    scale_codes_ng = fp8.view(torch.uint8)
    effective_scale = fp8.to(torch.float32) * tensor_scale

    safe_scale = torch.where(
        effective_scale == 0,
        torch.ones_like(effective_scale),
        effective_scale,
    )
    normalized = (blocked / safe_scale.unsqueeze(2)).clamp_(-6.0, 6.0)
    codes = fp4_e2m1_encode(normalized)
    codes = torch.where(
        (effective_scale == 0).unsqueeze(2),
        torch.zeros_like(codes),
        codes,
    ).view(n, k)
    return (
        codes.contiguous(),
        scale_codes_ng.t().contiguous(),
        tensor_scale.contiguous(),
    )


def dequantize_nvfp4(
    codes: torch.Tensor,
    scale_codes_gn: torch.Tensor,
    tensor_scale: torch.Tensor,
) -> torch.Tensor:
    """Decode the exact weight represented by ``quantize_nvfp4``."""
    groups, n = scale_codes_gn.shape
    k = codes.shape[1]
    if codes.shape != (n, groups * 16):
        raise ValueError(
            f"codes {tuple(codes.shape)} incompatible with scales "
            f"{tuple(scale_codes_gn.shape)}"
        )
    scale = scale_codes_gn.t().contiguous().view(torch.float8_e4m3fn)
    effective = scale.to(torch.float32) * tensor_scale.to(torch.float32)
    return (
        fp4_e2m1_decode(codes).view(n, groups, 16)
        * effective.unsqueeze(2)
    ).view(n, k)


def _swizzle_codes(
    codes: torch.Tensor, partition: int, num_cores: int
) -> torch.Tensor:
    n, k = codes.shape
    e = _NUM_ELE_32B_FP4
    v = codes.to(torch.uint8)
    if partition == 1:
        if n % (16 * num_cores) != 0 or k % e != 0:
            raise ValueError(
                "NVFP4 col swizzle needs N%(16*cores)==0 and K%64==0, "
                f"got N={n},K={k},cores={num_cores}"
            )
        v = v.view(
            num_cores, n // 16 // num_cores, 16, k // e, e
        ).permute(1, 3, 0, 2, 4)
    elif partition == 0:
        if n % 16 != 0 or k % (e * num_cores) != 0:
            raise ValueError(
                "NVFP4 row swizzle needs N%16==0 and K%(64*cores)==0, "
                f"got N={n},K={k},cores={num_cores}"
            )
        v = v.view(
            n // 16, 16, num_cores, k // e // num_cores, e
        ).permute(0, 3, 2, 1, 4)
    else:
        raise ValueError(f"partition must be 0(row) or 1(col), got {partition}")
    return v.contiguous().reshape(-1)


def swizzle_pack_nvfp4(
    codes: torch.Tensor, partition: int, num_cores: int
) -> torch.Tensor:
    """TP-swizzle logical FP4 codes and pack even/odd codes into low/high nibbles."""
    flat = _swizzle_codes(codes, partition, num_cores).cpu().numpy().astype(np.uint8)
    packed = ((flat[1::2] & 0x0F) << 4) | (flat[0::2] & 0x0F)
    return torch.from_numpy(packed).contiguous()
