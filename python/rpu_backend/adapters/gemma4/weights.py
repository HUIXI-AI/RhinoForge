"""Gemma4 mixed-width weight binding.

Builds the per-layer weight lists + dual RoPE tables + per-layer int arrays for
``torch.ops.rpu.gemma4_set_weights`` from a (swizzled) HF Gemma4 text model.

Mixed geometry:
  * per-layer q/o projections have layer-dependent width (sliding nq*256 / global
    nq*512); ``convert_linear_weights_inplace`` swizzles each nn.Linear by its own
    (in,out) so a single ``attn_num_cores = min(8, num_kv_heads)`` (=2 for E4B)
    covers every layer.
  * KV-shared layers (idx >= first shared) have NO k_proj/v_proj/k_norm (HF drops
    them). The C++ schema is a DENSE ``Tensor[]``, so those slots are filled with
    the SOURCE layer's (already-swizzled) k/v/k_norm as harmless placeholders —
    the C++ skips them via ``is_kv_shared`` and SDPA reads the source KV slot.

``transform_linear_weight`` preserves the (out,in) shape (layout-only permute),
so the C++ shape validation (q rows == nq*head_dim, o cols == nq*head_dim) holds
on swizzled weights.

Imports torch.ops / transformers lazily so the module is import-light.
"""
from __future__ import annotations

from typing import Any

from .geometry import cpp_layer_arrays


def _w(module: Any, name: str):
    """layer-module attribute -> .weight (HF Linear / RMSNorm)."""
    return getattr(module, name).weight


def extract_decoder_weight_lists(layers, geom) -> dict[str, list]:
    """Per-layer weight lists (length == num_layers) for gemma4_set_weights.

    ``layers`` is the HF decoder layer list (``model.language_model.layers``);
    ``geom`` is the geometry table. KV-shared layers contribute the SOURCE
    layer's k/v/k_norm as a dense placeholder (C++ ignores it).
    """
    keys = ("q", "k", "v", "o", "q_norm", "k_norm",
            "input_norm", "post_attn_norm", "pre_ff_norm", "post_ff_norm",
            "gate", "up", "down")
    out: dict[str, list] = {k: [] for k in keys}
    for i, (L, g) in enumerate(zip(layers, geom)):
        a = L.self_attn
        out["q"].append(a.q_proj.weight)
        out["o"].append(a.o_proj.weight)
        out["q_norm"].append(a.q_norm.weight)
        if g.is_kv_shared:
            # dense placeholders = source layer's (swizzled) k/v/k_norm
            src = layers[g.kv_source_layer].self_attn
            out["k"].append(src.k_proj.weight)
            out["v"].append(src.v_proj.weight)
            out["k_norm"].append(src.k_norm.weight)
        else:
            out["k"].append(a.k_proj.weight)
            out["v"].append(a.v_proj.weight)
            out["k_norm"].append(a.k_norm.weight)
        out["input_norm"].append(L.input_layernorm.weight)
        out["post_attn_norm"].append(L.post_attention_layernorm.weight)
        out["pre_ff_norm"].append(L.pre_feedforward_layernorm.weight)
        out["post_ff_norm"].append(L.post_feedforward_layernorm.weight)
        out["gate"].append(L.mlp.gate_proj.weight)
        out["up"].append(L.mlp.up_proj.weight)
        out["down"].append(L.mlp.down_proj.weight)
    return out


def make_rope_tables(text_config, max_pos: int, device: str = "rpu", dtype=None):
    """Two KERNEL-FORMAT cos/sin tables (sliding, global) via HF, ready for the
    RPU SPM RoPE kernel. Returns (cos_sliding, sin_sliding, cos_global, sin_global)
    each ``[max_pos, head_dim/2]``. Needs the isolated transformers with Gemma4.

    The RPU ``llama_rope`` kernel uses NeoX d/2-split pairing, identical to HF
    Gemma4, and reads cos/sin
    in the half-width KERNEL format ``[max_pos, head_dim/2]`` (the unique half; the
    fused C++ feeds the stored pointer straight to the kernel with NO slicing). HF
    emits the duplicated full-width form ``cat([h, h], dim=-1)``, so slicing to the
    first ``head_dim/2`` columns recovers the kernel format exactly — including the
    global layer's partial-rotary passthrough dims (cos=1/sin=0), which live in the
    second part of the unique half and carry over untouched.
    """
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
    if dtype is None:
        dtype = torch.float16
    rot = Gemma4TextRotaryEmbedding(text_config)
    pos = torch.arange(max_pos).unsqueeze(0)
    x = torch.zeros(1, max_pos, 8)
    tables = {}
    for lt in ("sliding_attention", "full_attention"):
        cos, sin = rot(x, pos, layer_type=lt)          # [1, max_pos, head_dim] (HF full)
        half = cos.size(-1) // 2                         # head_dim/2 (kernel format)
        cos_h = cos.squeeze(0)[:, :half]
        sin_h = sin.squeeze(0)[:, :half]
        tables[lt] = (cos_h.to(dtype=dtype, device=device).contiguous(),
                      sin_h.to(dtype=dtype, device=device).contiguous())
    cs, ss = tables["sliding_attention"]
    cg, sg = tables["full_attention"]
    return cs, ss, cg, sg


def extract_ple_weights(layers, geom) -> dict:
    """Per-layer PLE weights for gemma4_set_weights, ready (swizzled, rpu).

    Mirrors the fused MLP col->row pattern: per_layer_input_gate is COL-swizzled
    (partition=1, num_cores=8) so its output is partitioned per core, and
    per_layer_projection is ROW-swizzled (partition=0) so it consumes that
    partition + all-reduces to a replicated result. Returns
    {"ple_gate","ple_proj","ple_post_norm"} lists + "layer_scalar" tensor.
    Caller must already have `.to('rpu')`-ed the layers (or pass _w to move).
    """
    import torch
    from rpu_backend.runtime.weights import transform_linear_weight
    _w = lambda t: t.to(dtype=torch.float16, device="rpu").contiguous()
    sw = lambda t, part: _w(transform_linear_weight(t.to(torch.float16), part, num_cores=8))
    out = {"ple_gate": [], "ple_proj": [], "ple_post_norm": []}
    scalars = []
    for L in layers:
        out["ple_gate"].append(sw(L.per_layer_input_gate.weight, 1))    # col-partition
        out["ple_proj"].append(sw(L.per_layer_projection.weight, 0))    # row-partition
        out["ple_post_norm"].append(_w(L.post_per_layer_input_norm.weight))
        scalars.append(float(L.layer_scalar.reshape(-1)[0]))
    out["layer_scalar"] = torch.tensor(scalars, dtype=torch.float16, device="rpu")
    return out


def set_gemma4_weights(
    handle: int, geom, lists: dict[str, list], rope_tables, final_norm_w, *,
    num_q_heads: int, num_kv_heads_max: int, head_dim_max: int,
    hidden_size: int, intermediate_size: int, eps: float, qk_scale: float = 1.0,
    ple: dict | None = None, ple_dim: int = 0, sliding_window: int = 512,
) -> None:
    """Assemble the per-layer int arrays + call torch.ops.rpu.gemma4_set_weights.

    ``lists`` = output of ``extract_decoder_weight_lists`` (already swizzled).
    ``rope_tables`` = (cos_sliding, sin_sliding, cos_global, sin_global).
    ``ple`` = output of ``extract_ple_weights``; None disables PLE.
    """
    import torch
    a = cpp_layer_arrays(geom)
    cos_sliding, sin_sliding, cos_global, sin_global = rope_tables
    N = len(lists["q"])
    if ple is not None:
        ple_gate, ple_proj, ple_post = ple["ple_gate"], ple["ple_proj"], ple["ple_post_norm"]
        layer_scalar = ple["layer_scalar"]
    else:
        ple_gate = ple_proj = ple_post = []                # empty -> has_ple_=false
        layer_scalar = torch.zeros(N, dtype=torch.float16, device="rpu")  # dummy
        ple_dim = 0
    torch.ops.rpu.gemma4_set_weights(
        handle,
        lists["q"], lists["k"], lists["v"], lists["o"],
        lists["q_norm"], lists["k_norm"],
        lists["input_norm"], lists["post_attn_norm"],
        lists["pre_ff_norm"], lists["post_ff_norm"],
        lists["gate"], lists["up"], lists["down"],
        cos_sliding, sin_sliding, cos_global, sin_global, final_norm_w,
        ple_gate, ple_proj, ple_post, layer_scalar,
        a["layer_type_ids"], a["head_dim_per_layer"],
        a["num_kv_heads_per_layer"], a["kv_source_layer"], a["is_kv_shared"],
        int(num_q_heads), int(num_kv_heads_max), int(head_dim_max),
        int(hidden_size), int(intermediate_size), int(ple_dim),
        float(eps), float(qk_scale), int(sliding_window),
    )
