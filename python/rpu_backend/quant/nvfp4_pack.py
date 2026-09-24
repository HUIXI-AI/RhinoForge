"""NVFP4A16 weight quantization and packing for Rhino parallel linear.

Contract:
  weight       FP4 E2M1, TP swizzle + deinterleaved K32 nibble pairs
  block scale  FP8 E4M3FN, K32-pair/core/N16/pair/lane DDR stripes
  tensor scale FP32 scalar; projection-family arrays are preloaded to SPM

The quantizer returns logical codes/scales. Both must be packed separately
before upload; the physical scale's [K/16,N] view is shape metadata only.
Matches the provider NVFP4 tiled ACC32 weight-layout contract.
"""
from __future__ import annotations

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
    if not weight.is_floating_point() or not torch.isfinite(weight).all() or weight.numel() == 0:
        raise ValueError("NVFP4 requires nonempty finite floating-point weights")
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
    """Pack K[0..15] in low nibbles and K[16..31] in high nibbles.

    Each swizzled K64 group contains two such 16-byte runs. Adjacent logical K
    values must not be packed together: the tiled kernel widens the two halves
    directly, without the legacy runtime deinterleave.
    """
    swizzled = _swizzle_codes(codes, partition, num_cores).reshape(-1, 2, 2, 16)
    packed = (swizzled[:, :, 0] & 0x0F) | ((swizzled[:, :, 1] & 0x0F) << 4)
    return packed.reshape(-1).cpu().contiguous()


def swizzle_nvfp4_scales(
    scale_codes_gn: torch.Tensor, partition: int, num_cores: int
) -> torch.Tensor:
    """Pack logical [K/16,N] FP8 codes into the tiled controller scale ABI.

    Storage is [localK/32, cores, localN/16, 2, 16], regardless of TP axis.
    The returned 2D view preserves logical dimensions for native validation;
    its bytes are physical stripes, not the logical [group,N] matrix.
    """
    if scale_codes_gn.dim() != 2 or scale_codes_gn.dtype != torch.uint8:
        raise ValueError("NVFP4 block scale must be logical uint8 [K/16,N]")
    if not isinstance(num_cores, int) or isinstance(num_cores, bool) or not 1 <= num_cores <= 8:
        raise ValueError("NVFP4 num_cores must be an integer in [1,8]")
    groups, n = scale_codes_gn.shape
    scale_ng = scale_codes_gn.t().contiguous()
    if partition == 1:
        if n % (16 * num_cores) != 0 or groups % 4 != 0:
            raise ValueError("NVFP4 col scale packing needs N%(16*cores)==0 and K%64==0")
        packed = scale_ng.reshape(num_cores, n // num_cores // 16, 16, groups // 2, 2)
        packed = packed.permute(3, 0, 1, 4, 2)
    elif partition == 0:
        if n % 16 != 0 or groups % (4 * num_cores) != 0:
            raise ValueError("NVFP4 row scale packing needs N%16==0 and K%(64*cores)==0")
        packed = scale_ng.reshape(n // 16, 16, num_cores, groups // num_cores // 2, 2)
        packed = packed.permute(3, 2, 0, 4, 1)
    else:
        raise ValueError(f"partition must be 0(row) or 1(col), got {partition}")
    return packed.contiguous().reshape(groups, n)


def swizzle_pack_nvfp4_v2(codes: torch.Tensor, scale_gn: torch.Tensor,
                          partition: int, num_cores: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rhino ACC32 ABI: deinterleaved nibbles and TP-striped FP8 scales.

    This entry retains the Pi0.5 striped_v2 metadata and tensor-shape contract;
    other callers use the generic tiled packing functions independently.
    """
    if codes.dtype != torch.uint8 or codes.dim() != 2 or (codes > 15).any():
        raise ValueError("NVFP4 v2 requires logical uint8 E2M1 codes [N,K] in [0,15]")
    n, k = codes.shape
    if scale_gn.dtype != torch.uint8 or tuple(scale_gn.shape) != (k // 16, n):
        raise ValueError("NVFP4 v2 requires uint8 FP8 scales [K/16,N]")
    order = torch.tensor([b * 32 + i + j for b in range(2)
                          for i in range(16) for j in (0, 16)], device=codes.device)
    v = _swizzle_codes(codes, partition, num_cores).reshape(-1, 64)[:, order].flatten()
    packed = ((v[1::2] << 4) | v[0::2]).reshape(n, k // 2).contiguous()
    logical = scale_gn.t().contiguous()
    if partition == 1:
        physical = logical.reshape(num_cores, n // num_cores // 16, 16, k // 32, 2).permute(3, 0, 1, 4, 2)
    else:
        physical = logical.reshape(n // 16, 16, num_cores, k // num_cores // 32, 2).permute(3, 2, 0, 4, 1)
    return packed, physical.contiguous().reshape(k // 16, n)
