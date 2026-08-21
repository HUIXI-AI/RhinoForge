"""Pi05 packed-INT4 weight preparation.

For a real-W4 checkpoint (method=w4a16), the int4 projections are stored on disk
as int8 [N,K] (values in [-8,7]). At .to('rpu') time we nibble-pack them to uint8
[N,K/2] via quant.int4_pack.swizzle_pack_int4 so the production launcher dispatches
them to the wINT4 kernel. KV projections (in mixed_int8_suffixes) stay int8 (w8a16).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rpu_backend.quant.int4_pack import swizzle_pack_int4
from rpu_backend.runtime.weights import ROW_PARTITION_NAMES

# All Pi05/Gemma projection child names.
_ALL_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# Per-projection partition — MUST match the C++ fused-decoder launch sites:
#   q/k/v/o : rpu_gemma_model.cpp (VLM) / rpu_adarms_model.cpp (expert)
#   gate/up/down : fused_model_base.cpp::emit_mlp_pipeline
# Convention: q/k/v/gate/up = col(1), o/down = row(0). These projections are
# in `skip_names` when they are real-INT4, so the recursive walker never sees
# them and cannot record a layout for them — but the ROW half of the rule is
# THE SAME rule, so it is derived from `ROW_PARTITION_NAMES` rather than
# re-typed here and kept in lockstep by hand.
_PROJ_PARTITION = {
    name: (0 if name in ROW_PARTITION_NAMES else 1) for name in _ALL_PROJ
}

_ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}


def _projection_token(suffix: str) -> str:
    """Normalize 'k_proj.weight' or 'self_attn.k_proj.weight' -> 'k_proj'."""
    token = suffix[: -len(".weight")] if suffix.endswith(".weight") else suffix
    return token.rsplit(".", 1)[-1]


def int4_child_names(quant_config: dict) -> set[str]:
    """Child names of the projections that are real INT4 (packed), from quant config.

    = all projection tokens MINUS the mixed_int8_suffixes tokens. Empty if not w4a16.
    """
    if quant_config.get("method") != "w4a16":
        return set()
    # Accept both "k_proj.weight" and full suffixes like "self_attn.k_proj.weight".
    keep = {_projection_token(s) for s in quant_config.get("mixed_int8_suffixes", [])}
    unknown = keep - set(_ALL_PROJ)
    if unknown:
        raise RuntimeError(f"w4pack: unknown mixed_int8_suffixes projection token(s): {sorted(unknown)}")
    return set(_ALL_PROJ) - keep


def pack_int4_projections_inplace(decoder, int4_children: set[str], *, attn_num_cores: int) -> int:
    """Pack the int4 (int8-container [N,K]) projection weights in `decoder.layers`
    to uint8 [N,K/2] in place. Returns the number of projections packed.

    KV / non-int4 projections are left untouched (they go through the normal int8
    swizzle). Idempotent guard: a weight that is already uint8 is skipped.
    """
    n_packed = 0
    for layer in decoder.layers:
        for parent_name in ("self_attn", "mlp"):
            parent = getattr(layer, parent_name, None)
            if parent is None:
                continue
            for proj_name, module in parent.named_children():
                if proj_name not in int4_children or not isinstance(module, nn.Linear):
                    continue
                w = module.weight.data
                if w.dtype == torch.uint8:
                    continue  # already packed
                if w.dtype != torch.int8:
                    raise RuntimeError(
                        f"pack_int4: {parent_name}.{proj_name} expected int8 int4-container, "
                        f"got {w.dtype}")
                if proj_name not in _PROJ_PARTITION:
                    raise RuntimeError(
                        f"pack_int4: no partition mapping for projection {proj_name!r}; "
                        f"int4_children must be a subset of {sorted(_PROJ_PARTITION)}")
                partition = _PROJ_PARTITION[proj_name]
                num_cores = attn_num_cores if proj_name in _ATTN else 8
                N, K = int(module.out_features), int(module.in_features)
                packed = swizzle_pack_int4(w.detach().cpu(), partition, num_cores)
                packed = packed.reshape(N, K // 2).contiguous().to(w.device)
                module.weight = nn.Parameter(packed, requires_grad=False)
                n_packed += 1
    return n_packed
