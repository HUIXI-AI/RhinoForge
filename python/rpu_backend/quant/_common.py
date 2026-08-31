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


def quantize_linear_per_channel_activation_aware(
    weight: torch.Tensor,
    input_second_moment: torch.Tensor,
    *,
    bits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose each row scale by activation-weighted reconstruction error."""
    if weight.dim() != 2:
        raise ValueError(f"expected 2-D Linear weight, got {tuple(weight.shape)}")
    if bits != 8:
        raise ValueError("activation-aware quantization currently requires bits=8")
    if input_second_moment.shape != (weight.shape[1],):
        raise ValueError(
            "input_second_moment must match Linear input width: "
            f"got {tuple(input_second_moment.shape)}, expected {(weight.shape[1],)}"
        )

    importance = input_second_moment.detach().to(device=weight.device, dtype=torch.float32)
    if not torch.isfinite(importance).all() or (importance < 0).any():
        raise ValueError("input_second_moment must be finite and non-negative")
    if not torch.any(importance > 0):
        return quantize_linear_per_channel(weight, bits=bits)

    w = weight.detach().to(torch.float32)
    amax = w.abs().amax(dim=1)
    best_error = torch.full_like(amax, torch.inf)
    best_scale = torch.empty_like(amax, dtype=torch.float16)
    best_q = torch.empty_like(weight, dtype=torch.int8)
    for ratio in torch.linspace(1.0, 0.5, 11, device=w.device):
        scale = (amax * ratio / 127.0).to(torch.float16).clamp_min(_FP16_TINY)
        q = torch.round(w / scale.float().unsqueeze(1)).clamp_(-128, 127)
        error = ((w - q * scale.float().unsqueeze(1)).square() * importance).sum(dim=1)
        improved = error < best_error
        best_error = torch.where(improved, error, best_error)
        best_scale = torch.where(improved, scale, best_scale)
        best_q[improved] = q[improved].to(torch.int8)
    return best_q, best_scale


def dequantize_linear_per_channel(w_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize an int8-container per-channel Linear weight to fp16."""
    return scale.to(torch.float16).unsqueeze(1) * w_int8.to(torch.float16)
