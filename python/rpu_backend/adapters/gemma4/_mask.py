"""Pure-Python windowed additive mask — the single source of truth for the
sliding-window fill the C++ build_layer_subgraph mirrors. Import-safe (torch only)."""
from __future__ import annotations
import torch


def windowed_mask(offset: int, q_len: int, kv_len: int, sliding: int = 512) -> torch.Tensor:
    """Additive fp32 mask [q_len, kv_len]; 0=attend, -inf=mask.
    Query row q has absolute position abs=offset+q; attends iff abs-sliding < k <= abs."""
    NEG = float("-inf")
    q = torch.arange(q_len).unsqueeze(1)          # [q_len,1]
    k = torch.arange(kv_len).unsqueeze(0)         # [1,kv_len]
    abs_pos = offset + q
    allow = (k <= abs_pos) & (k > abs_pos - sliding)
    return torch.where(allow, torch.zeros(1), torch.full((1,), NEG))
