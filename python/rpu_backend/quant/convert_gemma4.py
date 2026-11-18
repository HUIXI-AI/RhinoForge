"""On-install W8A16 / G32 W4A16 for the exact Gemma4 E4B text decoder.

The seven decoder projections of all 42 layers are quantized once from the
checkpoint's FP16 values while the adapter builds its swizzled payload:

* ``w8a16``: symmetric per-channel INT8 (``quantize_linear_per_channel``). At
  MLP6 the codes gain zero rows (gate/up) or columns (down) and gate/up scales
  gain unit entries; then the INT8 swizzle for the projection's lanes. The
  scale stays the natural FP16 ``[N]`` vector the launcher reads.
* ``w4a16``: G32 RTN ``pack_int4_weight`` for the projection's lanes, which pads
  MLP6 itself (zero values, unit scales) and stripes the FP16 group scale.

Attention projections run on two lanes at every budget, MLP projections on the
budget's lanes (8, 6 or 4). PLE, norms, embeddings and the tied CPU lm_head stay
as in the FP16 profile.
"""

from __future__ import annotations

from typing import Callable

import torch

from rpu_backend.quant._common import quantize_linear_per_channel
from rpu_backend.quant.convert_qwen3_w4a16 import DEFAULT_GROUP_SIZE, pack_int4_weight
from rpu_backend.runtime.weights import transform_linear_weight

ON_INSTALL_QUANTIZATION = ("w8a16", "w4a16")

# role -> (parameter suffix in the decoder layer, partition: 1 = col, 0 = row).
PROJECTIONS: dict[str, tuple[str, int]] = {
    "q": ("self_attn.q_proj.weight", 1),
    "k": ("self_attn.k_proj.weight", 1),
    "v": ("self_attn.v_proj.weight", 1),
    "o": ("self_attn.o_proj.weight", 0),
    "gate": ("mlp.gate_proj.weight", 1),
    "up": ("mlp.up_proj.weight", 1),
    "down": ("mlp.down_proj.weight", 0),
}
MLP_ROLES = ("gate", "up", "down")
_FLOAT_SOURCES = (torch.float16, torch.bfloat16, torch.float32)


def projection_source(pfx: str, layer: int, geometry, role: str) -> str:
    """Checkpoint name a role reads; KV-shared layers read their source's k/v."""
    owner = geometry.kv_source_layer if role in ("k", "v") else layer
    return f"{pfx}layers.{owner}.{PROJECTIONS[role][0]}"


def check_gemma4_quant_source(source: Callable[[str], torch.Tensor], geom, *, pfx: str,
                              quantization: str, hidden_size: int, num_q_heads: int,
                              intermediate_size: int) -> None:
    """Admit only unquantized 2-D CPU float projections of the logical E4B shapes.

    ``source(name)`` returns the checkpoint tensor without converting it; no
    values are read, and every projection is checked before the first is
    quantized.
    """
    if quantization not in ON_INSTALL_QUANTIZATION:
        raise ValueError(f"unknown Gemma4 on-install recipe {quantization!r}")
    for layer, g in enumerate(geom):
        attention, kv = num_q_heads * g.head_dim, g.num_kv_heads * g.head_dim
        shapes = {"q": (attention, hidden_size), "k": (kv, hidden_size),
                  "v": (kv, hidden_size), "o": (hidden_size, attention),
                  "gate": (intermediate_size, hidden_size),
                  "up": (intermediate_size, hidden_size),
                  "down": (hidden_size, intermediate_size)}
        for role, shape in shapes.items():
            name = projection_source(pfx, layer, g, role)
            try:
                weight = source(name)
            except KeyError:
                weight = None
            if (not torch.is_tensor(weight) or weight.dtype not in _FLOAT_SOURCES
                    or weight.device.type != "cpu" or tuple(weight.shape) != shape):
                raise ValueError(
                    f"{name}: Gemma4 {quantization} requires the original CPU float "
                    f"projection {list(shape)}")


def quantize_gemma4_projection(weight: torch.Tensor, quantization: str, *, role: str,
                               lanes: int, physical: int | None = None,
                               name: str = "") -> tuple[torch.Tensor, torch.Tensor]:
    """CPU float ``[N, K]`` -> (launcher payload, FP16 scale) for ``lanes`` cores.

    ``physical`` is the MLP width; when it exceeds the logical width (MLP6), the
    MLP projections are padded after logical quantization.
    """
    if quantization not in ON_INSTALL_QUANTIZATION:
        raise ValueError(f"unknown Gemma4 on-install recipe {quantization!r}")
    partition = PROJECTIONS[role][1]
    w = weight.to(torch.float16)
    logical = int(w.shape[1] if role == "down" else w.shape[0])
    padded = (int(physical) if role in MLP_ROLES and physical is not None
              and int(physical) != logical else None)
    if quantization == "w4a16":
        return pack_int4_weight(w, partition, name, DEFAULT_GROUP_SIZE,
                                num_cores=lanes, padded_size=padded)
    codes, scale = quantize_linear_per_channel(w, bits=8)
    if padded is not None:
        from rpu_backend.adapters.gemma4.topology import pad_mlp_projection

        codes = pad_mlp_projection(codes, role=role, physical=padded)
        if role != "down":
            scale = torch.cat([scale, scale.new_ones(padded - logical)])
    return (transform_linear_weight(codes, partition, num_cores=lanes).contiguous(),
            scale.contiguous())
