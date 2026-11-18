"""Gemma4 E4B runtime glue (T7+): reusable builder shared by the gate scripts
and ``Gemma4Adapter.to_rpu``.

Three pure helpers driven by a ``raw(full_name) -> cpu float tensor`` getter, so
the SAME code serves both the offline safe_open gates (``f.get_tensor``) and the
adapter path (``model.state_dict()``):

  * ``swizzle_decoder_and_ple`` — per-layer swizzled q/k/v/o/norms/mlp + PLE
    weight lists (KV-shared layers reuse the SOURCE layer's k/v/k_norm as dense
    placeholders, matching the C++ ``is_kv_shared`` skip).
  * ``compute_side_input`` — the T6a PLE side-input, core-major laid out for the
    C++ scatter (per forward; depends on input_ids).
  * ``softcap_lm_logits`` — tied lm_head (embed_tokens.weight) + final softcap,
    the CPU FP32 reference.
  * ``upload_lm_head`` / ``device_lm_logits`` — the same tied lm_head as an
    RPU Linear (FP16 weight, FP32 accumulation) whose logits return to the CPU
    for the softcap; the adapter's forward uses this pair.

Proven swizzle partitions (T5/T6b/T7 gates; ``transform_linear_weight`` is
layout-only so the C++ shape checks hold): q/k/v=col1·nc2, o=row0·nc2,
gate/up=col1·ncM, down=row0·ncM, ple_gate=col1·ncP, ple_proj=row0·ncP, where
the cold budget (``topology.py``) selects M/P = 8/8 (default), 6/4 or 4/4.
At M=6 gate/up rows and down columns are zero-padded 10240 -> 10272 before
the swizzle; budget 8 keeps the original unpadded payload.

With an on-install recipe (``quantization="w8a16"`` / ``"w4a16"``) the seven
decoder projections are quantized for their lanes instead
(``rpu_backend.quant.convert_gemma4``; MLP6 pads to 10368) and their FP16
scales join the lists as ``<role>_scale``. PLE and norms stay FP16.
"""
from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F

from rpu_backend.runtime.weights import transform_linear_weight, NUM_CORES

from .topology import mlp_physical_intermediate, pad_mlp_projection, resolve_gemma4_topology

PFX = "model.language_model."

# Per-projection partition. Attention lanes are fixed at two (two KV heads);
# MLP/PLE lanes come from the cold topology (eight by default).
_ATTN = {"q": (1, 2), "k": (1, 2), "v": (1, 2), "o": (0, 2)}
_MLP = {"gate": 1, "up": 1, "down": 0}
_PLE = {"ple_gate": 1, "ple_proj": 0}


def _to_rpu(t: torch.Tensor) -> torch.Tensor:
    return t.to(device="rpu").contiguous()


def _rpu(w: torch.Tensor) -> torch.Tensor:
    return _to_rpu(w.to(torch.float16))


def _swz(w: torch.Tensor, part: int, nc: int) -> torch.Tensor:
    return _rpu(transform_linear_weight(w.to(torch.float16), part, num_cores=nc))


def _pad_rows(w: torch.Tensor, lanes: int) -> torch.Tensor:
    if w.ndim != 2:
        return w
    align = 16 * int(lanes)
    rem = int(w.shape[0]) % align
    return F.pad(w, (0, 0, 0, align - rem)) if rem else w


def swizzle_decoder_and_ple(raw: Callable[[str], torch.Tensor], geom, pfx: str = PFX,
                            *, topology=None, quantization=None):
    """Build the swizzled per-layer weight lists + PLE dict for set_gemma4_weights.

    ``raw(name)`` returns a CPU float tensor for the full param name. KV-shared
    layers (``g.is_kv_shared``) take their k/v/k_norm from ``g.kv_source_layer``
    (same layer type => same swizzle), which the C++ ignores but keeps the dense
    Tensor[] schema valid. ``topology`` (default: eight cores) selects the
    MLP/PLE lanes and the physical MLP width. ``quantization`` (None, ``"w8a16"``
    or ``"w4a16"``) must match the topology's storage; quantized lists also
    carry ``q_scale`` ... ``down_scale``.
    """
    topology = (resolve_gemma4_topology(8, quantization=quantization)
                if topology is None else topology)
    mlp_tp, ple_tp = topology.mlp_tp, topology.ple_tp
    physical = topology.physical_intermediate
    if physical != mlp_physical_intermediate(
            topology.logical_intermediate, mlp_tp, quantization):
        raise ValueError("Gemma4 topology was resolved for a different projection storage")
    if quantization is not None:
        from rpu_backend.quant.convert_gemma4 import quantize_gemma4_projection

    def mlp(name, role):
        weight = raw(name)
        if quantization is not None:
            return quantized(weight, role, mlp_tp, name)
        if topology.mlp_padding:
            weight = pad_mlp_projection(weight, role=role, physical=physical)
        return _swz(weight, _MLP[role], mlp_tp)

    def attention(name, role):
        if quantization is not None:
            return quantized(raw(name), role, _ATTN[role][1], name)
        return _swz(raw(name), *_ATTN[role])

    def quantized(weight, role, lanes, name):
        payload, scale = quantize_gemma4_projection(
            weight, quantization, role=role, lanes=lanes, physical=physical, name=name)
        lists[f"{role}_scale"].append(_to_rpu(scale))
        return _to_rpu(payload)

    keys = ("q", "k", "v", "o", "q_norm", "k_norm", "input_norm",
            "post_attn_norm", "pre_ff_norm", "post_ff_norm", "gate", "up", "down")
    if quantization is not None:
        keys += tuple(f"{role}_scale" for role in ("q", "k", "v", "o", "gate", "up", "down"))
    lists: dict[str, list] = {k: [] for k in keys}
    ple: dict[str, list] = {"ple_gate": [], "ple_proj": [], "ple_post_norm": []}
    scalars = []
    for i, g in enumerate(geom):
        p = f"{pfx}layers.{i}."
        ps = f"{pfx}layers.{g.kv_source_layer}."          # == p when not shared
        lists["q"].append(attention(p + "self_attn.q_proj.weight", "q"))
        lists["o"].append(attention(p + "self_attn.o_proj.weight", "o"))
        lists["q_norm"].append(_rpu(raw(p + "self_attn.q_norm.weight")))
        lists["k"].append(attention(ps + "self_attn.k_proj.weight", "k"))
        lists["v"].append(attention(ps + "self_attn.v_proj.weight", "v"))
        lists["k_norm"].append(_rpu(raw(ps + "self_attn.k_norm.weight")))
        lists["input_norm"].append(_rpu(raw(p + "input_layernorm.weight")))
        lists["post_attn_norm"].append(_rpu(raw(p + "post_attention_layernorm.weight")))
        lists["pre_ff_norm"].append(_rpu(raw(p + "pre_feedforward_layernorm.weight")))
        lists["post_ff_norm"].append(_rpu(raw(p + "post_feedforward_layernorm.weight")))
        lists["gate"].append(mlp(p + "mlp.gate_proj.weight", "gate"))
        lists["up"].append(mlp(p + "mlp.up_proj.weight", "up"))
        lists["down"].append(mlp(p + "mlp.down_proj.weight", "down"))
        ple["ple_gate"].append(_swz(raw(p + "per_layer_input_gate.weight"),
                                    _PLE["ple_gate"], ple_tp))
        ple["ple_proj"].append(_swz(raw(p + "per_layer_projection.weight"),
                                    _PLE["ple_proj"], ple_tp))
        ple["ple_post_norm"].append(_rpu(raw(p + "post_per_layer_input_norm.weight")))
        scalars.append(float(raw(p + "layer_scalar").reshape(-1)[0]))
    ple["layer_scalar"] = _rpu(torch.tensor(scalars, dtype=torch.float16))
    return lists, ple


def compute_side_input(
    input_ids: torch.Tensor, *, embed_weight: torch.Tensor,
    embed_per_layer_weight: torch.Tensor, model_projection_weight: torch.Tensor,
    projection_norm_weight: torch.Tensor, num_layers: int, hidden_size: int,
    ple_dim: int, eps: float, num_cores: int = NUM_CORES,
    model_projection_rpu: torch.Tensor | None = None,
    model_projection_lanes: int | None = None,
    raw_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    """T6a PLE side-input -> core-major fp16 CPU tensor [num_layers, C, S, ple_dim/C].

    ``num_cores`` (C) is the PLE lane count of the installed topology: eight by
    default, four at budgets 4 and 6.

    side = (proj_norm(per_layer_model_projection(embeds) * H^-0.5)
            + embed_tokens_per_layer(ids) * sqrt(ple_dim)) * 2^-0.5
    where embeds = embed_tokens(ids) * sqrt(H). Core-major so the C++ scatter
    reads core c <- columns [c*pc, (c+1)*pc).

    The four embedding/projection weights are passed directly (raw, any dtype):
    rows are GATHERED before upcast so the 5.6GB per-layer table is never
    fully materialized in fp32 (critical for per-token decode). When
    ``model_projection_rpu`` is supplied, the ``[num_layers * ple_dim, hidden]``
    projection runs on RPU in FP16 with FP32 accumulation instead of host CPU.
    """
    ids = input_ids.to("cpu", dtype=torch.long)
    B, S = ids.shape
    H, PD, L = hidden_size, ple_dim, num_layers
    emb_raw = raw_embeds if raw_embeds is not None else F.embedding(ids, embed_weight)
    tok_id = (F.embedding(ids, embed_per_layer_weight).float()
              * (PD ** 0.5)).reshape(B, S, L, PD)
    if isinstance(model_projection_rpu, torch.Tensor) and isinstance(emb_raw, torch.Tensor):
        proj_in = emb_raw.reshape(-1, H).to(dtype=torch.float16, device="rpu").contiguous()
        proj_lanes = int(model_projection_lanes or NUM_CORES)
        ctx = torch.ops.rpu.linear_with_accumulation(
            proj_in, model_projection_rpu, None, 1, proj_lanes, True
        ).to("cpu").float()
        if ctx.shape[-1] > L * PD:
            ctx = ctx[:, :L * PD]
        ctx = ctx.reshape(B, S, L, PD)
    else:
        embeds = emb_raw.float() * (H ** 0.5)
        ctx = (embeds @ model_projection_weight.float().t()) * (H ** -0.5)
        ctx = ctx.reshape(B, S, L, PD)
    ctx = _rms(ctx, projection_norm_weight, eps)
    side = (ctx + tok_id) * (2.0 ** -0.5)                  # [B,S,L,PD]
    pc = PD // num_cores
    return (side[0].permute(1, 0, 2)                       # [L,S,PD]
            .reshape(L, S, num_cores, pc).permute(0, 2, 1, 3)   # [L,C,S,pc]
            .contiguous().to(torch.float16))


def softcap_lm_logits(hidden: torch.Tensor, embed_weight: torch.Tensor,
                      softcap: float = 30.0) -> torch.Tensor:
    """Tied lm_head (logits = hidden @ embed_tokens.weight.T) + final softcap.

    ``hidden`` is the model's final-normed output; ``embed_weight`` is the RAW
    (unscaled) embed_tokens.weight (tie_word_embeddings). softcap: tanh(l/c)*c.
    Computed in fp32; device follows ``hidden``.
    """
    h = hidden.float()
    w = embed_weight.float().to(h.device)
    return _softcap(h @ w.t(), softcap)


def upload_model_projection(model_projection_weight: torch.Tensor, lanes: int) -> torch.Tensor:
    """PLE model projection weight col-swizzled (row-padded if needed) for ``lanes`` on RPU."""
    return _swz(_pad_rows(model_projection_weight, lanes), 1, lanes)


def upload_lm_head(embed_weight: torch.Tensor, lanes: int) -> torch.Tensor:
    """Tied lm_head weight (raw embed_tokens.weight) col-swizzled for ``lanes`` on RPU."""
    return _swz(_pad_rows(embed_weight, lanes), 1, lanes)


def device_lm_logits(hidden: torch.Tensor, head_weight: torch.Tensor, lanes: int,
                     softcap: float = 30.0, *, vocab_size: int | None = None) -> torch.Tensor:
    """Tied lm_head on RPU + final softcap on CPU -> fp32 CPU logits [B, S, vocab].

    ``hidden`` is the decoder's final-normed RPU output [B, S, H];
    ``head_weight`` comes from ``upload_lm_head`` with the same ``lanes``.
    The projection keeps FP16 weights and accumulates in FP32, matching the
    CPU reference's precision as closely as the Linear allows.
    """
    rows = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    logits = torch.ops.rpu.linear_with_accumulation(rows, head_weight, None, 1, lanes, True)
    logits = logits.to("cpu").float().reshape(*hidden.shape[:-1], -1)
    if vocab_size is not None and logits.shape[-1] > vocab_size:
        logits = logits[..., :vocab_size]
    if softcap and softcap > 0:
        logits.div_(softcap).tanh_().mul_(softcap)
    return logits


def _softcap(logits: torch.Tensor, softcap: float) -> torch.Tensor:
    if softcap and softcap > 0:
        logits = torch.tanh(logits / softcap) * softcap
    return logits


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * w.float()
