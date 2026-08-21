"""W4A16 (packed int4 weight / fp16 activation) for Qwen3.

This converter defaults to group-wise pgrp int4: `rpu_linear.cpp` routes a **packed uint8
[N, K/2]** weight plus a controller-striped FP16 scale payload to the tiled
wINT4a16 pgrp family. Weights are handed over through
`causal_decoder_set_weights_w8a16` — the same entry point as int8, because the
launcher dispatches on the weight's dtype (kChar = int8, kByte = packed int4),
not on the setter. This mirrors how the Wall-OSS adapter does w4a16.

Quantization is group-wise symmetric int4 (values -8..7), default group size
32. Per-channel mode remains an explicit A/B option. Both modes fuse nibble
packing with their row/col swizzle, so these projections MUST be excluded from
`convert_linear_weights_inplace` — running both would double-swizzle.

Scope / limits
--------------
* Only the 7 decoder projections are int4. `lm_head` and the embedding stay
  fp16: lm_head is col-partition swizzled by the generic converter.
* Group-wise pgrp supports M=1 through its tiled family. The optional
  per-channel mode has no M=1 kernel.
* Qwen3-0.6B needs no intermediate-size padding: every projection already
  satisfies the int4 swizzle rules (col: N%(16*cores)==0 and K%64==0;
  row: N%16==0 and K%(64*cores)==0). Larger Qwen3 sizes are not checked here.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rpu_backend.model_registry import model_path
from rpu_backend.quant.int4_pack import quantize_int4_per_channel, swizzle_pack_int4
from rpu_backend.quant.int4_pgrp_pack import (
    quantize_int4_group_wise,
    swizzle_int4_pgrp_scale,
    swizzle_pack_int4_pgrp,
)

NUM_CORES = 8
# Group-wise (pgrp / AWQ-style) group size along K. 32 matches the Wall-OSS
# w4a16-pgrp checkpoint and the parallel_linear_wint4a16_pgrp_* kernel ABI.
DEFAULT_GROUP_SIZE = 32

# (attribute path on the decoder layer, partition) — partition must match what
# build_layer_subgraph passes to the linear launcher: 1 = col, 0 = row.
_PROJECTIONS: tuple[tuple[str, int], ...] = (
    ("self_attn.q_proj", 1),
    ("self_attn.k_proj", 1),
    ("self_attn.v_proj", 1),
    ("self_attn.o_proj", 0),
    ("mlp.gate_proj", 1),
    ("mlp.up_proj", 1),
    ("mlp.down_proj", 0),
)

SKIP_LINEAR_NAMES = frozenset(p.rsplit(".", 1)[-1] for p, _ in _PROJECTIONS)


def _get(module: nn.Module, dotted: str) -> nn.Module:
    for part in dotted.split("."):
        module = getattr(module, part)
    return module


def _check_int4_shape(name: str, n: int, k: int, partition: int) -> None:
    if partition == 1:
        ok = (n % (16 * NUM_CORES) == 0) and (k % 64 == 0)
        rule = f"col: N%{16*NUM_CORES}==0 and K%64==0"
    else:
        ok = (n % 16 == 0) and (k % (64 * NUM_CORES) == 0)
        rule = f"row: N%16==0 and K%{64*NUM_CORES}==0"
    if not ok:
        raise ValueError(
            f"{name}: shape [N={n}, K={k}] violates the int4 swizzle rule ({rule}). "
            "This projection would need padding, which this converter does not do."
        )


def pack_linear_int4_(linear: nn.Linear, partition: int, name: str = "",
                      group_size: int | None = None) -> None:
    """In-place: fp16 [N,K] Linear -> packed uint8 [N,K/2] + fp16 scale.

    group_size None -> per-channel, scale [N] (1D).
    group_size  int -> group-wise pgrp, controller-striped 2D scale payload.
    The C++ launcher dispatches on scale.dim(), so the scale rank IS the mode
    selector; the two packings use different swizzles and are NOT interchangeable.
    """
    w = linear.weight.detach()
    n, k = int(w.shape[0]), int(w.shape[1])
    _check_int4_shape(name, n, k, partition)

    if group_size:
        if k % group_size != 0:
            raise ValueError(f"{name}: K={k} not divisible by group_size={group_size}")
        vals, logical_scale, _ = quantize_int4_group_wise(w, group_size)
        packed = swizzle_pack_int4_pgrp(vals, partition, NUM_CORES).reshape(n, k // 2)
        scale = swizzle_int4_pgrp_scale(
            logical_scale, group_size, partition, NUM_CORES
        )
    else:
        vals, scale = quantize_int4_per_channel(w)                     # scale [N]
        packed = swizzle_pack_int4(vals, partition, NUM_CORES).reshape(n, k // 2)

    linear.weight = nn.Parameter(
        packed.to(torch.uint8).contiguous(), requires_grad=False)
    # register_buffer, NOT a plain attribute: `model.to('rpu')` only migrates
    # parameters and buffers. A plain attribute leaves the scale on the host,
    # and the C++ side then takes a device address of a CPU pointer, producing
    # invalid outputs without a device error. quant/load.py does
    # the same thing for the W8A16 checkpoint scales.
    if hasattr(linear, "weight_scale"):
        del linear.weight_scale
    linear.register_buffer("weight_scale",
                           scale.to(torch.float16).contiguous(),
                           persistent=False)
    linear._rpu_int4_logical_shape = (n, k)


def convert_qwen3_to_w4a16_(model, group_size: int | None = None) -> None:
    """Quantize every decoder projection of a loaded fp16 Qwen3 to packed int4."""
    for i, layer in enumerate(model.model.layers):
        for dotted, partition in _PROJECTIONS:
            mod = _get(layer, dotted)
            if mod.weight.dtype != torch.float16:
                raise TypeError(
                    f"layer{i}.{dotted}: expected fp16 weight, got {mod.weight.dtype}")
            pack_linear_int4_(mod, partition, name=f"layer{i}.{dotted}",
                              group_size=group_size)


def load_w4a16_model(path: str | None = None,
                     group_size: int | None = DEFAULT_GROUP_SIZE):
    """Load Qwen3-0.6B and return it with int4 projections, ready for `.to('rpu')`.

    Group-wise quantization is the default. Per-channel int4 remains available
    with ``group_size=None`` for diagnostic comparison.

    The caller still does `.to('rpu')` and
    `_install_causal_decoder_forward(..., scale_lists=...)`.
    """
    from transformers import AutoModelForCausalLM
    from rpu_backend.runtime.weights import convert_linear_weights_inplace

    path = path or str(model_path("qwen3-0.6b"))
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True, local_files_only=True)
    model.eval()

    convert_qwen3_to_w4a16_(model, group_size=group_size)
    # Everything NOT int4 (notably lm_head) still needs the fp16 swizzle. The
    # int4 projections carry their swizzle inside the packing, hence the skip.
    convert_linear_weights_inplace(model, skip_names=set(SKIP_LINEAR_NAMES))
    return model


if __name__ == "__main__":
    import sys
    m = load_w4a16_model()
    q = m.model.layers[0].self_attn.q_proj
    print(f"q_proj.weight  dtype={q.weight.dtype} shape={tuple(q.weight.shape)}")
    print(f"q_proj.scale   dtype={q.weight_scale.dtype} shape={tuple(q.weight_scale.shape)}")
    print(f"lm_head.weight dtype={m.lm_head.weight.dtype} "
          f"shape={tuple(m.lm_head.weight.shape)}")
    sys.exit(0)
