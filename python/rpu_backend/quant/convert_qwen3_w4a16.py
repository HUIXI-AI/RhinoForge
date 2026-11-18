"""W4A16 (packed int4 weight / fp16 activation) for Qwen3.

Qwen3 shipped with a W8A16 offline converter but no 4-bit path. This converter
defaults to group-wise pgrp int4: `rpu_linear.cpp` routes N32K32-packed uint8
[padded global N, global K/2] weights and controller-striped FP16 scales to
the generated w4a16_pgrp family. Weights are handed over through
`causal_decoder_set_weights_w8a16` — the same entry point as int8, because the
launcher dispatches on the weight's dtype (kChar = int8, kByte = packed int4),
not on the setter. This mirrors how the Wall-OSS adapter does w4a16.

Quantization is group-wise symmetric int4 (values -8..7), default group size
32. Per-channel math remains an explicit A/B option and is represented by
repeating its scale over K/32 groups. Both modes use the same pgrp physical ABI,
so these projections MUST be excluded from
`convert_linear_weights_inplace` — running both would double-swizzle (pitfall P1).

Scope / limits
--------------
* The default recipe quantizes seven decoder projections. The explicit
  quantize_lm_head option also packs the head; embedding and norms stay FP16.
* Both quantization modes support M=1 through the generated pgrp tiled family.
* Public admission validates six exact dense Qwen3 source profiles. Every
  projection must satisfy col N%(16*TP)/K%32 or row N%16/K%(32*TP).
  A six-lane MLP whose width is not 192-aligned is quantized at its logical
  width, then zero-padded to the physical width before the single pgrp pack.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rpu_backend.model_registry import model_path
from rpu_backend.quant._common import quantize_linear_per_channel
from rpu_backend.quant.int4_pgrp_pack import (
    pad_int4_mlp_stripe,
    quantize_pack_int4_pgrp_col_bounded,
    quantize_int4_group_wise,
    swizzle_int4_pgrp_scale,
    swizzle_pack_int4_pgrp,
    w4_storage_shape,
)
from rpu_backend.runtime.topology import (
    decoder_lm_head_vocab_size,
    decoder_packed_w4_mlp_intermediate_size,
)

NUM_CORES = 8
# Group-wise (pgrp / AWQ-style) group size along K. 32 matches the Wall-OSS
# w4a16-pgrp checkpoint and the parallel_linear_w4a16_pgrp_* kernel ABI.
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


def _check_int4_shape(name: str, n: int, k: int, partition: int,
                      num_cores: int = NUM_CORES) -> None:
    if type(num_cores) is not int or not 1 <= num_cores <= NUM_CORES:
        raise ValueError("W4 packing requires an integer core count in [1,8]")
    if partition not in (0, 1):
        raise ValueError("W4 packing partition must be 0(row) or 1(col)")
    if partition == 1:
        ok = (n % (16 * num_cores) == 0) and (k % 32 == 0)
        rule = f"col: N%{16*num_cores}==0 and K%32==0"
    else:
        ok = (n % 16 == 0) and (k % (32 * num_cores) == 0)
        rule = f"row: N%16==0 and K%{32*num_cores}==0"
    if not ok:
        raise ValueError(
            f"{name}: shape [N={n}, K={k}] violates the int4 swizzle rule ({rule}). "
            "Only the admitted six-lane MLP width is padded by this converter."
        )


def packed_mlp_padded_size(role: str, linear: nn.Linear, mlp_tp: int) -> int | None:
    """Physical packed width of one Text7 MLP leaf, or None when unpadded."""
    if not role.startswith("mlp."):
        return None
    logical = linear.in_features if role.endswith("down_proj") else linear.out_features
    physical = decoder_packed_w4_mlp_intermediate_size(logical, mlp_tp)
    return physical if physical != logical else None


def pack_int4_weight(w: torch.Tensor, partition: int, name: str = "",
                     group_size: int | None = None, *, num_cores: int = NUM_CORES,
                     padded_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack fp16 [N,K] into N32K32 uint8 with TP-aware N padding and FP16 scale.

    group_size None -> logical per-channel, repeated over K/32 groups.
    group_size  int -> group-wise pgrp, controller-striped 2D scale payload.
    Both produce the same controller-striped 2-D physical scale ABI.
    padded_size widens an MLP stripe after logical quantization: col
    partition pads N, row partition pads K, with zero values and unit scales.
    """
    n, k = int(w.shape[0]), int(w.shape[1])
    if padded_size is not None:
        if padded_size < (n if partition == 1 else k):
            raise ValueError(f"{name}: padded width {padded_size} is smaller than its source")
        if padded_size % DEFAULT_GROUP_SIZE:
            raise ValueError(f"{name}: padded width {padded_size} must keep whole K/32 groups")
    pn, pk = ((padded_size, k) if partition == 1 else (n, padded_size)
              ) if padded_size is not None else (n, k)
    _check_int4_shape(name, pn, pk, partition, num_cores)

    if group_size:
        if k % group_size != 0:
            raise ValueError(f"{name}: K={k} not divisible by group_size={group_size}")
        vals, logical_scale, _ = quantize_int4_group_wise(w, group_size)
        scale_group = group_size
    else:
        # Per-channel math is one scale repeated over every K/32 group, as in
        # pack_int4_per_channel_as_pgrp; padding then follows the same path.
        scale_group = DEFAULT_GROUP_SIZE
        if k % scale_group:
            raise ValueError(f"{name}: K={k} not divisible by group_size={scale_group}")
        vals, channel_scale = quantize_linear_per_channel(w, bits=4)
        logical_scale = (channel_scale.to(torch.float16).reshape(1, n)
                         .expand(k // scale_group, n).contiguous())
    if padded_size is not None:
        vals, logical_scale = pad_int4_mlp_stripe(
            vals, logical_scale, partition=partition, padded_size=padded_size,
            group_size=scale_group)
    packed = swizzle_pack_int4_pgrp(vals, partition, num_cores).reshape(
        w4_storage_shape(pn, pk, partition, num_cores))
    scale = swizzle_int4_pgrp_scale(logical_scale, scale_group, partition, num_cores)
    return packed.to(torch.uint8).contiguous(), scale.to(torch.float16).contiguous()


def pack_linear_int4_(linear: nn.Linear, partition: int, name: str = "",
                      group_size: int | None = None, *, num_cores: int = NUM_CORES,
                      padded_size: int | None = None) -> None:
    """In-place ``pack_int4_weight`` of one Linear: packed uint8 weight + fp16 scale.

    Linear features become physical; model config stays logical.
    """
    n, k = int(linear.weight.shape[0]), int(linear.weight.shape[1])
    pn, pk = ((padded_size, k) if partition == 1 else (n, padded_size)
              ) if padded_size is not None else (n, k)
    packed, scale = pack_int4_weight(
        linear.weight.detach(), partition, name, group_size,
        num_cores=num_cores, padded_size=padded_size)
    linear.weight = nn.Parameter(packed, requires_grad=False)
    linear.out_features, linear.in_features = pn, pk
    # register_buffer, NOT a plain attribute: `model.to('rpu')` only migrates
    # parameters and buffers. A plain attribute leaves the scale on the host,
    # and the C++ side then takes a device address of a CPU pointer — no error,
    # just garbage. (Symptom: the model emits a constant token, so MMLU lands
    # exactly on the dataset's base rate for that letter.) quant/load.py does
    # the same thing for the W8A16 checkpoint scales.
    if hasattr(linear, "weight_scale"):
        del linear.weight_scale
    linear.register_buffer("weight_scale", scale, persistent=False)


def convert_qwen3_to_w4a16_(model, group_size: int | None = None,
                           *, quantize_lm_head: bool = False, topology=None) -> None:
    """Quantize every decoder projection of a loaded fp16 Qwen3 to packed int4."""
    attn_tp = topology.attn_tp if topology is not None else NUM_CORES
    mlp_tp = topology.mlp_tp if topology is not None else NUM_CORES
    head_tp = topology.lm_head_tp if topology is not None else NUM_CORES
    for i, layer in enumerate(model.model.layers):
        for dotted, partition in _PROJECTIONS:
            mod = _get(layer, dotted)
            n, k = mod.weight.shape
            cores = mlp_tp if dotted.startswith("mlp.") else attn_tp
            padded = packed_mlp_padded_size(dotted, mod, mlp_tp)
            if padded is not None:
                n, k = (padded, k) if partition == 1 else (n, padded)
            _check_int4_shape(f"layer{i}.{dotted}", n, k, partition, cores)
            if mod.weight.dtype != torch.float16:
                raise TypeError(f"layer{i}.{dotted}: expected fp16 weight, got {mod.weight.dtype}")
    if quantize_lm_head:
        if group_size != DEFAULT_GROUP_SIZE:
            raise ValueError("W4 lm_head requires group_size=32")
        if model.lm_head.weight.dtype != torch.float16:
            raise TypeError("W4 lm_head requires original FP16 weights")
        head_n, head_k = model.lm_head.weight.shape
        padded_head_n = decoder_lm_head_vocab_size(head_n, head_tp)
        _check_int4_shape("lm_head", padded_head_n, head_k, 1, head_tp)
    for i, layer in enumerate(model.model.layers):
        for dotted, partition in _PROJECTIONS:
            mod = _get(layer, dotted)
            cores = mlp_tp if dotted.startswith("mlp.") else attn_tp
            pack_linear_int4_(mod, partition, name=f"layer{i}.{dotted}",
                              group_size=group_size, num_cores=cores,
                              padded_size=packed_mlp_padded_size(dotted, mod, mlp_tp))
    if quantize_lm_head:
        if group_size != DEFAULT_GROUP_SIZE:
            raise ValueError("W4 lm_head requires group_size=32")
        head = model.lm_head
        if head.weight.dtype != torch.float16:
            raise TypeError("W4 lm_head requires original FP16 weights")
        head_w = head.weight.detach()
        padded_head_n = decoder_lm_head_vocab_size(int(head_w.shape[0]), head_tp)
        if padded_head_n != int(head_w.shape[0]):
            padded = head_w.new_zeros((padded_head_n, int(head_w.shape[1])))
            padded[:head_w.shape[0]].copy_(head_w)
            head_w = padded
        packed, scale = quantize_pack_int4_pgrp_col_bounded(
            head_w, group_size, num_cores=head_tp)
        # Replace only the head Parameter. The 0.6/1.7/4B source ties it to
        # embed_tokens; the embedding must retain the original FP16 storage.
        head.weight = nn.Parameter(packed, requires_grad=False)
        head.out_features = padded_head_n
        head.register_buffer("weight_scale", scale, persistent=False)


def load_w4a16_model(path: str | None = None,
                     group_size: int | None = DEFAULT_GROUP_SIZE):
    """Load Qwen3-0.6B and return it with int4 projections, ready for `.to('rpu')`.

    Uses group-wise quantization by default. ``group_size=None`` selects the
    alternative per-channel format for evaluation; it changes scale granularity
    and must not be treated as the same numerical profile.

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
