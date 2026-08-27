"""Shared per-channel linear quantization helpers."""
from __future__ import annotations

import torch


_FP16_TINY = torch.finfo(torch.float16).tiny


def quantize_linear_per_channel(
    weight: torch.Tensor,
    *,
    bits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a Linear weight [N, K] to int8 storage plus per-row fp16 scale.

    ``bits=4`` stores signed 4-bit values in int8 containers for fake-W4
    numerical probes. It deliberately does not pack nibbles or change kernels.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected 2-D Linear weight, got {tuple(weight.shape)}")
    if bits not in (4, 8):
        raise ValueError(f"expected bits to be 4 or 8, got {bits}")

    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))

    w_fp16 = weight.detach().to(torch.float16)
    amax = w_fp16.to(torch.float32).abs().amax(dim=1)
    scale = (amax / float(qmax)).to(torch.float16).clamp_min(_FP16_TINY)
    w_int8 = (
        torch.round(w_fp16.to(torch.float32) / scale.to(torch.float32).unsqueeze(1))
        .clamp_(qmin, qmax)
        .to(torch.int8)
    )
    return w_int8, scale


def dequantize_linear_per_channel(w_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize an int8-container per-channel Linear weight to fp16."""
    return scale.to(torch.float16).unsqueeze(1) * w_int8.to(torch.float16)
