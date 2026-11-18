"""Pi05 four-bit weight preparation, applied once before generic swizzle.

INT4 uses int8 [N,K] codes with per-channel or genuine K128 group scales and
packs to the controller-striped pgrp ABI. NVFP4 uses uint8 E2M1 codes and a
separate striped_v2 weight/FP8-scale ABI. K/V remains W8 in Action-only profiles.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rpu_backend.quant.int4_pgrp_pack import (pack_int4_per_channel_as_pgrp, swizzle_pack_int4_pgrp, swizzle_int4_pgrp_scale, w4_storage_shape, W4_WEIGHT_LAYOUT)
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
# re-typed here. (Before 2026-08-09 this was an independent copy that had to be
# kept in lockstep by hand.)
_PROJ_PARTITION = {
    name: (0 if name in ROW_PARTITION_NAMES else 1) for name in _ALL_PROJ
}

_ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}


def _projection_token(suffix: str) -> str:
    """Normalize 'k_proj.weight' or 'self_attn.k_proj.weight' -> 'k_proj'."""
    token = suffix[: -len(".weight")] if suffix.endswith(".weight") else suffix
    return token.rsplit(".", 1)[-1]


def int4_child_names(quant_config: dict, *, component: str | None = None) -> set[str]:
    """Child names of the INT4 or NVFP4 projections selected for packing.

    All projection tokens minus mixed_int8_suffixes, limited to the component.
    """
    if quant_config.get("method") not in ("w4a16", "nvfp4a16"):
        return set()
    # A scoped experimental checkpoint must never pack an untouched W8 component.
    components = quant_config.get("int4_components")
    if components is not None:
        if (not isinstance(components, list) or not components
                or any(name not in ("vlm", "expert") for name in components)
                or len(set(components)) != len(components)
                or component not in ("vlm", "expert")):
            raise ValueError("w4pack: scoped INT4 requires valid unique components and an explicit component")
        if component not in components:
            return set()
    # Accept both "k_proj.weight" and full suffixes like "self_attn.k_proj.weight".
    keep = {_projection_token(s) for s in quant_config.get("mixed_int8_suffixes", [])}
    unknown = keep - set(_ALL_PROJ)
    if unknown:
        raise RuntimeError(f"w4pack: unknown mixed_int8_suffixes projection token(s): {sorted(unknown)}")
    return set(_ALL_PROJ) - keep


def pack_int4_projections_inplace(decoder, int4_children: set[str], *, attn_num_cores: int,
                                  group_size: int = 32, mlp_num_cores: int = 8) -> int:
    """Pack selected [N,K] projections to N32K32 physical storage in place.

    Other projections use their normal swizzle. Packed INT4 is identified by
    an explicit N32K32 marker; NVFP4 retains its distinct packed marker.
    NVFP4 MLP projections stripe over ``mlp_num_cores`` lanes; INT4 keeps TP8.
    Returns the number of newly packed projections.
    """
    if type(group_size) is not int or group_size not in (32, 64, 128):
        raise ValueError("pack_int4: group_size must be 32, 64, or 128")
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
                if getattr(module, "_pi05_nvfp4_abi", None) == "striped_v2":
                    if getattr(module, "_pi05_nvfp4_packed", False):
                        continue
                    from rpu_backend.quant.nvfp4_pack import swizzle_pack_nvfp4_v2
                    scale = module.weight_scale
                    cores = attn_num_cores if proj_name in _ATTN else mlp_num_cores
                    packed, packed_scale = swizzle_pack_nvfp4_v2(
                        w.detach().cpu(), scale.detach().cpu(), _PROJ_PARTITION[proj_name], cores)
                    module.weight = nn.Parameter(packed.to(w.device), requires_grad=False)
                    module._buffers["weight_scale"] = packed_scale.to(scale.device)
                    module._pi05_nvfp4_packed = True
                    n_packed += 1
                    continue
                if w.dtype == torch.uint8:
                    scale = getattr(module, "weight_scale", None)
                    if (getattr(module, "_rpu_w4_layout", None) != W4_WEIGHT_LAYOUT
                            or scale is None or scale.dim() != 2 or scale.size(0) != group_size):
                        raise RuntimeError(
                            f"pack_int4: {parent_name}.{proj_name} is already uint8 "
                            "but does not carry an installed N32K32 weight and pgrp scale payload"
                        )
                    continue
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
                scale = getattr(module, "weight_scale", None)
                if scale is None:
                    raise RuntimeError(
                        f"pack_int4: {parent_name}.{proj_name} has no weight_scale"
                    )
                if scale.dim() == 2:
                    if (getattr(module, "_pi05_int4_group_size", None) != group_size
                            or tuple(scale.shape) != (K // group_size, N)):
                        raise RuntimeError("pack_int4: logical group scales/metadata mismatch")
                    packed = swizzle_pack_int4_pgrp(w.detach().cpu(), partition, num_cores)
                    packed_scale = swizzle_int4_pgrp_scale(scale.detach().cpu(), group_size, partition, num_cores)
                else:
                    packed, packed_scale = pack_int4_per_channel_as_pgrp(
                        w.detach().cpu(), scale.detach().cpu(),
                        partition, num_cores, group_size=group_size,
                    )
                packed = packed.reshape(w4_storage_shape(N, K, partition, num_cores)).contiguous().to(w.device)
                module.weight = nn.Parameter(packed, requires_grad=False)
                module._rpu_w4_layout = W4_WEIGHT_LAYOUT
                if "weight_scale" in module._buffers:
                    module._buffers["weight_scale"] = packed_scale.to(scale.device)
                else:
                    delattr(module, "weight_scale")
                    module.register_buffer(
                        "weight_scale", packed_scale.to(scale.device)
                    )
                n_packed += 1
    return n_packed
