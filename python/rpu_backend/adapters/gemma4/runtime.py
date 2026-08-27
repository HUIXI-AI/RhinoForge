"""Gemma4 E4B runtime glue: reusable builder shared by validation callers
and ``Gemma4Adapter.to_rpu``.

Three pure helpers driven by a ``raw(full_name) -> cpu float tensor`` getter, so
the same code serves both offline checkpoint reads (``f.get_tensor``) and the
adapter path (``model.state_dict()``):

  * ``swizzle_decoder_and_ple`` — per-layer swizzled q/k/v/o/norms/mlp + PLE
    weight lists (KV-shared layers reuse the SOURCE layer's k/v/k_norm as dense
    placeholders, matching the C++ ``is_kv_shared`` skip).
  * ``compute_side_input`` — the PLE side-input, core-major laid out for the
    C++ scatter (per forward; depends on input_ids).
  * ``softcap_lm_logits`` — tied lm_head (embed_tokens.weight) + final softcap.

Swizzle partitions (``transform_linear_weight`` is layout-only, so the C++
shape checks hold): q/k/v=col1·nc2, o=row0·nc2,
gate/up=col1·nc8, down=row0·nc8, ple_gate=col1·nc8, ple_proj=row0·nc8.
"""
from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F

from rpu_backend.runtime.weights import transform_linear_weight, NUM_CORES

PFX = "model.language_model."

# Per-projection (partition, num_cores) — single source of truth for the swizzle.
_ATTN = {"q": (1, 2), "k": (1, 2), "v": (1, 2), "o": (0, 2)}
_MLP = {"gate": (1, 8), "up": (1, 8), "down": (0, 8)}
_PLE = {"ple_gate": (1, 8), "ple_proj": (0, 8)}


def _rpu(w: torch.Tensor) -> torch.Tensor:
    return w.to(dtype=torch.float16, device="rpu").contiguous()


def _swz(w: torch.Tensor, part: int, nc: int) -> torch.Tensor:
    return _rpu(transform_linear_weight(w.to(torch.float16), part, num_cores=nc))


def swizzle_decoder_and_ple(raw: Callable[[str], torch.Tensor], geom, pfx: str = PFX):
    """Build the swizzled per-layer weight lists + PLE dict for set_gemma4_weights.

    ``raw(name)`` returns a CPU float tensor for the full param name. KV-shared
    layers (``g.is_kv_shared``) take their k/v/k_norm from ``g.kv_source_layer``
    (same layer type => same swizzle), which the C++ ignores but keeps the dense
    Tensor[] schema valid.
    """
    keys = ("q", "k", "v", "o", "q_norm", "k_norm", "input_norm",
            "post_attn_norm", "pre_ff_norm", "post_ff_norm", "gate", "up", "down")
    lists: dict[str, list] = {k: [] for k in keys}
    ple: dict[str, list] = {"ple_gate": [], "ple_proj": [], "ple_post_norm": []}
    scalars = []
    for i, g in enumerate(geom):
        p = f"{pfx}layers.{i}."
        ps = f"{pfx}layers.{g.kv_source_layer}."          # == p when not shared
        lists["q"].append(_swz(raw(p + "self_attn.q_proj.weight"), *_ATTN["q"]))
        lists["o"].append(_swz(raw(p + "self_attn.o_proj.weight"), *_ATTN["o"]))
        lists["q_norm"].append(_rpu(raw(p + "self_attn.q_norm.weight")))
        lists["k"].append(_swz(raw(ps + "self_attn.k_proj.weight"), *_ATTN["k"]))
        lists["v"].append(_swz(raw(ps + "self_attn.v_proj.weight"), *_ATTN["v"]))
        lists["k_norm"].append(_rpu(raw(ps + "self_attn.k_norm.weight")))
        lists["input_norm"].append(_rpu(raw(p + "input_layernorm.weight")))
        lists["post_attn_norm"].append(_rpu(raw(p + "post_attention_layernorm.weight")))
        lists["pre_ff_norm"].append(_rpu(raw(p + "pre_feedforward_layernorm.weight")))
        lists["post_ff_norm"].append(_rpu(raw(p + "post_feedforward_layernorm.weight")))
        lists["gate"].append(_swz(raw(p + "mlp.gate_proj.weight"), *_MLP["gate"]))
        lists["up"].append(_swz(raw(p + "mlp.up_proj.weight"), *_MLP["up"]))
        lists["down"].append(_swz(raw(p + "mlp.down_proj.weight"), *_MLP["down"]))
        ple["ple_gate"].append(_swz(raw(p + "per_layer_input_gate.weight"), *_PLE["ple_gate"]))
        ple["ple_proj"].append(_swz(raw(p + "per_layer_projection.weight"), *_PLE["ple_proj"]))
        ple["ple_post_norm"].append(_rpu(raw(p + "post_per_layer_input_norm.weight")))
        scalars.append(float(raw(p + "layer_scalar").reshape(-1)[0]))
    ple["layer_scalar"] = torch.tensor(scalars, dtype=torch.float16, device="rpu")
    return lists, ple


def compute_side_input(
    input_ids: torch.Tensor, *, embed_weight: torch.Tensor,
    embed_per_layer_weight: torch.Tensor, model_projection_weight: torch.Tensor,
    projection_norm_weight: torch.Tensor, num_layers: int, hidden_size: int,
    ple_dim: int, eps: float,
) -> torch.Tensor:
    """Build the PLE side-input as a core-major FP16 RPU tensor.

    The output shape is ``[num_layers, NUM_CORES, S, ple_dim / NUM_CORES]``.

    side = (proj_norm(per_layer_model_projection(embeds) * H^-0.5)
            + embed_tokens_per_layer(ids) * sqrt(ple_dim)) * 2^-0.5
    where embeds = embed_tokens(ids) * sqrt(H). Core-major so the C++ scatter
    reads core c <- columns [c*pc, (c+1)*pc).

    The four embedding/projection weights are passed directly (raw, any dtype):
    rows are GATHERED before upcast so the 5.6GB per-layer table is never
    fully materialized in fp32 (critical for per-token decode).
    """
    ids = input_ids.to("cpu", dtype=torch.long)
    B, S = ids.shape
    H, PD, L = hidden_size, ple_dim, num_layers
    embeds = F.embedding(ids, embed_weight).float() * (H ** 0.5)
    tok_id = (F.embedding(ids, embed_per_layer_weight).float()
              * (PD ** 0.5)).reshape(B, S, L, PD)
    ctx = (embeds @ model_projection_weight.float().t()) * (H ** -0.5)
    ctx = ctx.reshape(B, S, L, PD)
    ctx = _rms(ctx, projection_norm_weight, eps)
    side = (ctx + tok_id) * (2.0 ** -0.5)                  # [B,S,L,PD]
    pc = PD // NUM_CORES
    return (side[0].permute(1, 0, 2)                       # [L,S,PD]
            .reshape(L, S, NUM_CORES, pc).permute(0, 2, 1, 3)   # [L,NUM_CORES,S,pc]
            .contiguous().to(torch.float16).to("rpu"))


def softcap_lm_logits(hidden: torch.Tensor, embed_weight: torch.Tensor,
                      softcap: float = 30.0) -> torch.Tensor:
    """Tied lm_head (logits = hidden @ embed_tokens.weight.T) + final softcap.

    ``hidden`` is the model's final-normed output; ``embed_weight`` is the RAW
    (unscaled) embed_tokens.weight (tie_word_embeddings). softcap: tanh(l/c)*c.
    Computed in fp32; device follows ``hidden``.
    """
    h = hidden.float()
    w = embed_weight.float().to(h.device)
    logits = h @ w.t()
    if softcap and softcap > 0:
        logits = torch.tanh(logits / softcap) * softcap
    return logits


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * w.float()
