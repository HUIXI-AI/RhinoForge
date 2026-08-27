"""Reusable construction helpers for the RhinoVLA RPU action expert."""
from __future__ import annotations

import copy
from typing import Any


def build_rpu_expert(
    expert: Any,
    *,
    prefix_len: int,
    suffix_len: int,
    max_seq_len: int,
) -> Any:
    """Build one expert whose runtime prefix may vary within ``max_seq_len``.

    ``prefix_len`` is only the initial position used while installing the
    handle. Every forward resolves physical KV rows and logical RoPE rows from
    its actual prefix/cache and right-padding mask.
    """
    import torch

    from .convert import convert_expert_for_rpu
    from .fused import patch_rhino_vla_for_rpu

    torch.rpu.set_ddr_flush(True)
    converted = copy.deepcopy(expert)
    cpu_rotary = copy.deepcopy(converted.qwen_rotary_emb).cpu()
    converted = convert_expert_for_rpu(converted)
    patch_rhino_vla_for_rpu(
        converted,
        prefix_len=int(prefix_len),
        suffix_len=int(suffix_len),
        max_seq_len=int(max_seq_len),
        cpu_rotary_emb=cpu_rotary,
    )
    return converted


__all__ = ["build_rpu_expert"]
