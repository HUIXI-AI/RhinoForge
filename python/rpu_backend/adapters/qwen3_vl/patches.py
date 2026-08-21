"""Class-level patches for Qwen3-VL.

Idempotent class swaps for `Qwen3VLTextRMSNorm` and `Qwen3VLTextRotaryEmbedding`
so the CPU-side reference path inside HF code still works after `.to('rpu')`.

These patches run once per process at adapter import time.
"""

from __future__ import annotations

from rpu_backend.runtime.decoder import (
    _install_rmsnorm_class_swap,
    _install_rotary_class_swap,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
)


def _idempotent_patch_qwen3_vl_rmsnorm(rmsnorm_class) -> None:
    if getattr(rmsnorm_class, "_rpu_patched_rmsnorm", False):
        return
    _install_rmsnorm_class_swap(rmsnorm_class)
    rmsnorm_class._rpu_patched_rmsnorm = True


def _idempotent_patch_qwen3_vl_rotary(rotary_emb_class) -> None:
    if getattr(rotary_emb_class, "_rpu_patched_rotary", False):
        return
    _install_rotary_class_swap(rotary_emb_class)
    rotary_emb_class._rpu_patched_rotary = True


_idempotent_patch_qwen3_vl_rmsnorm(Qwen3VLTextRMSNorm)
_idempotent_patch_qwen3_vl_rotary(Qwen3VLTextRotaryEmbedding)
