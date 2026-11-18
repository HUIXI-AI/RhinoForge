"""Pure CPU helpers for Rhino RPU-compatible per-output-channel FP8 weights."""
from __future__ import annotations

import torch


FP8_E4M3_MAX = 448.0
# Shared grouped count ABI: smul.u16(expert_batch_start, 2) addresses m_sizes.
MAX_GROUPED_EXPERTS = 32768


def _validate_grouped_experts(experts: int) -> None:
    if not 1 <= experts <= MAX_GROUPED_EXPERTS:
        raise ValueError("grouped experts must be in [1,32768] for m_sizes byte offsets")


def quantize_fp8_e4m3_per_output(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw uint8 E4M3 codes and the FP16 scale hardware consumes."""
    if not isinstance(weight, torch.Tensor) or weight.dim() < 2:
        raise TypeError("FP8 weight must be a Tensor with at least two dimensions")
    if not weight.is_floating_point():
        raise TypeError(f"FP8 source weight must be floating, got {weight.dtype}")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("this PyTorch build does not expose torch.float8_e4m3fn")
    source = weight.detach().to("cpu").contiguous().float()
    if not bool(torch.isfinite(source).all()):
        raise ValueError("FP8 source weight contains NaN or Inf")
    max_abs = source.abs().amax(dim=-1)
    scale = (max_abs / FP8_E4M3_MAX).clamp_min(
        torch.finfo(torch.float16).tiny
    ).to(torch.float16)
    normalized = (source / scale.float().unsqueeze(-1)).clamp(
        -FP8_E4M3_MAX, FP8_E4M3_MAX
    )
    return (
        normalized.to(torch.float8_e4m3fn).view(torch.uint8).contiguous(),
        scale.contiguous(),
    )


def dequantize_fp8_e4m3_per_output(
    codes: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Decode the exact logical weight represented by the converter."""
    if codes.dtype != torch.uint8:
        raise TypeError(f"FP8 code storage must be uint8, got {codes.dtype}")
    if tuple(codes.shape[:-1]) != tuple(scale.shape):
        raise ValueError(
            f"FP8 scale shape {tuple(scale.shape)} does not match code rows "
            f"{tuple(codes.shape[:-1])}"
        )
    decoded = codes.detach().to("cpu").contiguous().view(torch.float8_e4m3fn)
    return decoded.float() * scale.detach().to("cpu").float().unsqueeze(-1)


def pack_grouped_weight(
    weight: torch.Tensor, *, partition: str, num_cores: int = 8
) -> torch.Tensor:
    """Apply the exact Rhino RPU grouped TP swizzle to logical ``[E,N,K]``."""
    if weight.dim() != 3:
        raise ValueError(f"grouped weight must be [E,N,K], got {tuple(weight.shape)}")
    if partition not in {"col", "row"}:
        raise ValueError(f"partition must be 'col' or 'row', got {partition!r}")
    group, n, k = map(int, weight.shape)
    _validate_grouped_experts(group)
    if group < 1 or num_cores < 1:
        raise ValueError("group and num_cores must be positive")
    dwidth = weight.element_size()
    elements_per_32b = 32 // dwidth
    source = weight.detach().to("cpu").contiguous()
    if partition == "col":
        if n % (16 * num_cores) or k % elements_per_32b:
            raise ValueError(
                f"grouped col shape {(group, n, k)} violates N%(16*tp)==0 "
                f"or K%{elements_per_32b}==0"
            )
        packed = source.view(
            group, num_cores, n // 16 // num_cores, 16,
            k // elements_per_32b, elements_per_32b,
        ).permute(0, 2, 4, 1, 3, 5)
    else:
        if n % 16 or k % (elements_per_32b * num_cores):
            raise ValueError(
                f"grouped row shape {(group, n, k)} violates N%16==0 or "
                f"K%({elements_per_32b}*tp)==0"
            )
        packed = source.view(
            group, n // 16, 16, num_cores,
            k // elements_per_32b // num_cores, elements_per_32b,
        ).permute(0, 1, 4, 3, 2, 5)
    return packed.contiguous().view(group, -1)


def pack_grouped_scale(scale: torch.Tensor) -> torch.Tensor:
    """Retain FP16 per-expert scales with the unified ABI's 256-byte stride.

    N remains the logical output width in the launcher. Padding is only at
    the end of each expert's scale row, including for column TP layouts.
    """
    if (not isinstance(scale, torch.Tensor) or scale.dim() != 2
            or scale.dtype != torch.float16 or scale.numel() == 0):
        raise ValueError("grouped scale requires nonempty FP16 [experts,N]")
    experts, n = scale.shape
    _validate_grouped_experts(experts)
    source = scale.detach().to("cpu")
    physical_n = (n + 127) // 128 * 128
    if physical_n == n:
        return source.contiguous()
    padded = source.new_zeros((experts, physical_n))
    padded[:, :n].copy_(source)
    return padded


__all__ = [
    "FP8_E4M3_MAX",
    "dequantize_fp8_e4m3_per_output",
    "pack_grouped_weight",
    "pack_grouped_scale",
    "quantize_fp8_e4m3_per_output",
]
