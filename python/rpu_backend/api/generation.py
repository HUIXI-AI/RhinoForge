"""Token selection for explicit greedy generation loops."""
from __future__ import annotations

import torch


def greedy_token_ids(logits: torch.Tensor) -> torch.Tensor:
    """Return independent CPU int64 ``[batch, 1]`` next-token IDs.

    Accept ``[batch, vocab]`` or ``[batch, sequence, vocab]`` logits; for the
    latter, select the final sequence row. RPU logits must be FP16. CPU logits
    may be FP16, BF16, FP32, or FP64 and are never cast to lower precision.
    Ties select the first maximum, and NaNs select the first NaN, as in
    ``torch.argmax``. The input is not modified. Empty batches are accepted;
    empty sequence/vocabulary dimensions are rejected.

    Call after a model forward in a handwritten greedy loop. This helper does
    not apply sampling, logits processors, or EOS handling, and does not
    change Hugging Face ``generate`` or the device semantics of ``argmax``.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError("greedy_token_ids expects a Tensor")
    if logits.layout != torch.strided:
        raise TypeError("greedy_token_ids expects dense strided logits")
    if logits.ndim not in (2, 3):
        raise ValueError("greedy_token_ids expects [batch, vocab] or [batch, sequence, vocab]")
    if logits.shape[-1] == 0 or (logits.ndim == 3 and logits.shape[1] == 0):
        raise ValueError("greedy_token_ids requires nonempty sequence and vocabulary dimensions")
    if logits.device.type not in ("cpu", "rpu"):
        raise ValueError("greedy_token_ids expects CPU or RPU logits")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("greedy_token_ids expects floating-point logits")
    if logits.device.type == "rpu" and logits.dtype != torch.float16:
        raise TypeError("greedy_token_ids requires FP16 RPU logits")

    last = logits[:, -1, :] if logits.ndim == 3 else logits
    if last.device.type == "cpu":
        return last.argmax(dim=-1, keepdim=True)
    if last.shape[0] == 0:
        return torch.empty((0, 1), dtype=torch.int64, device="cpu")
    return torch.ops.rpu.argmax_lastdim_host(last.contiguous()).reshape(-1, 1)
