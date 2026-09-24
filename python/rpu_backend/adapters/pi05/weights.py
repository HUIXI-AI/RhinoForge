"""Pi0.5 weight-prep helpers: leaf-Linear swizzle + projector pre-swizzle (PI05-01 reshape Plan 03-01).

Per D-3-04 refinement, weight-prep for SigLIP encoder + AdaRMS dense stays in
sibling converters (siglip_converter.py / adarms_converter.py) through Phase 3;
Phase 5 absorbs them. This file owns ONLY the 4 PI05Pytorch root-Linear swizzle
(CR-R5 BLOCKER 1) and the projector pre-swizzle (Pitfall-1 belt for the
multi_modal_projector).

v5-02 / B3: relocated from transformers/pi05/weights.py to adapters/pi05/weights.py.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from rpu_backend.runtime.log import _LOG


# =============================================================================
# CR-R5 BLOCKER 1 (F4 real fix) — direct leaf-Linear swizzle helper.
#
# The legacy recursive-walk helper in the model_converter shim (rpu_backend's
# v4.0 surface) walks `module.named_children()`. A leaf `nn.Linear` has NO
# children — iteration is empty, function returns WITHOUT touching `.weight.data`.
# Iteration 3/4 called that recursive-walk helper on each of the 4 root Linears
# directly,
# which was therefore a no-op: the 4 root Linears stayed row-major, and the
# `rpu_linear` kernel read them with the wrong layout -> MSE catastrophic.
#
# The correct operation: call `transform_linear_weight` directly on the leaf's
# `.weight.data`. This helper does exactly that, using the same
# partition-selection logic the recursive-walk helper would have applied had
# the Linear been a child of a larger module.
# =============================================================================
def _swizzle_leaf_linear(linear: nn.Linear, *, force_col_partition: bool = False) -> int:
    """Swizzle a leaf `nn.Linear`'s weight data IN-PLACE via transform_linear_weight.

    Returns the partition chosen (0 = row, 1 = col, -1 = skipped/fallback).

    Uses the same partition-selection as the legacy recursive-walk helper
    would have applied to a non-attention, non-{'dense','o_proj','down_proj'}
    Linear — which is exactly what the 4 PI05Pytorch root Linears are.
    """
    # Phase 12 D-H1: was the model_converter shim multi-import; canonical home is core.weights.
    from rpu_backend.runtime.weights import (
        transform_linear_weight, get_linear_partition, NUM_CORES,
    )

    if not isinstance(linear, nn.Linear):
        raise TypeError(f"expected nn.Linear, got {type(linear)}")

    in_features = linear.in_features
    out_features = linear.out_features
    dwidth = linear.weight.element_size()
    num_cores = NUM_CORES

    if force_col_partition:
        partition = get_linear_partition(in_features, out_features, dwidth,
                                          num_cores=num_cores)
        if partition == 0:
            partition = 1
    else:
        partition = get_linear_partition(in_features, out_features, dwidth,
                                          num_cores=num_cores)

    if partition < 0:
        # Fallback: skip swizzle (rpu_linear will fall back to CPU path).
        return partition

    with torch.no_grad():
        transformed = transform_linear_weight(
            linear.weight.data, partition, num_cores=num_cores)
        linear.weight.data = transformed

    return partition


def swizzle_projector_inplace(projector, *, num_cores=8) -> None:
    """Pi05-local SigLIP projector pre-swizzle (P1 double-swizzle belt).

    Sets `_rpu_weights_converted` sentinel BEFORE move-to-RPU so
    `_convert_siglip_projector_for_rpu` (siglip_converter.py:165) sees idempotent.
    Mirrors `_adapter.py:949-956` Step C verbatim.

    NOTE: `_rpu_weights_converted` is in INTERNAL_HW_ATTRS_TRANSITIONAL (verified
    in core/runtime/hw_attrs.py:101) — AM-3 guard satisfied; no whitelist
    extension needed.
    """
    from rpu_backend.adapters.siglip import _convert_siglip_projector_for_rpu
    _convert_siglip_projector_for_rpu(projector, num_cores=num_cores)


# =============================================================================
# D-B3 — Pi0.5 safetensors-key remapping (moved from pi05_converter.py:59 in
# Phase 5 Plan 05-01). Pi05-specific: renames lerobot's `model.…` keys to
# the HF Pi0.5 layout that the v4 loader expects. Not a generic
# `core/weights/` candidate (key rules are Pi05-specific).
# =============================================================================
def _remap_and_save(model_path: str, save_path: str):
    """First-time: load original safetensors, remap keys, save as new safetensors."""
    import os
    import glob
    from safetensors.torch import load_file, save_file

    _LOG.info("Loading original safetensors...")
    sf_files = sorted(glob.glob(os.path.join(model_path, "model*.safetensors")))
    sf_files = [f for f in sf_files if 'remapped' not in f]
    original = {}
    for sf in sf_files:
        original.update(load_file(sf))

    remapped = {}
    for key, value in original.items():
        new_key = key
        if key.startswith("action_time_mlp_in."):
            new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
        elif key.startswith("action_time_mlp_out."):
            new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
        if key.startswith("state_proj."):
            continue
        if (key == "paligemma_with_expert.paligemma.lm_head.weight" or
            key == "model.paligemma_with_expert.paligemma.lm_head.weight"):
            clone_key = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
            remapped[clone_key] = value.clone()
        if not new_key.startswith("model."):
            new_key = f"model.{new_key}"
        remapped[new_key] = value

    _LOG.info("Remapped %d keys, saving to %s...", len(remapped), save_path)
    save_file(remapped, save_path)
    _LOG.info("Done.")


# ----------------------------------------------------------------------
# Pi0.5 num_steps fuse — weights helpers.
# Weight preparation shared by the per-step and fused denoise paths.
# ----------------------------------------------------------------------

def _prepare_final_norm_weights(expert, *, num_cores=8) -> None:
    """Swizzle the model's final PiGemmaRMSNorm dense for AdaRMS-style GEMV.

    The per-layer AdaRMS converter (adarms.py:_prepare_adarms_dense_weights)
    already registers ``norm._rpu_dense_w_rp`` / ``_rpu_dense_b_rp`` on each
    layer's input/post_attention layernorm. The final norm (``expert.norm``)
    is not touched by that helper — it stays on CPU and runs in Python fp32
    by default (adarms.py:445). For the fused denoise step we need it on RPU
    in the same row-partition swizzled form.

    Idempotent: skips work if buffers are already registered.
    """
    import torch
    from rpu_backend.runtime.weights import transform_linear_weight, NUM_CORES

    norm = expert.norm  # PiGemmaRMSNorm with .dense Linear (3H, H)
    if hasattr(norm, "_rpu_dense_w_rp") and hasattr(norm, "_rpu_dense_b_rp"):
        return  # already prepped
    if not hasattr(norm, "dense") or norm.dense is None:
        raise RuntimeError(
            "_prepare_final_norm_weights: expert.norm has no .dense — "
            "is this a PiGemmaRMSNorm with AdaRMS?")
    if norm.dense.bias is None:
        raise RuntimeError(
            "_prepare_final_norm_weights: expert.norm.dense has no bias.")

    orig_w = (norm.dense.weight.detach().cpu()
              .to(dtype=torch.float16).contiguous())
    transformed_w = (transform_linear_weight(orig_w, partition=0,
                                              num_cores=num_cores)
                     .to('rpu').contiguous())
    orig_b = (norm.dense.bias.detach().cpu()
              .to(dtype=torch.float16).to('rpu').contiguous())
    norm.register_buffer('_rpu_dense_w_rp', transformed_w)
    norm.register_buffer('_rpu_dense_b_rp', orig_b)


def _swizzle_action_proj_weights(action_in_proj, action_out_proj) -> None:
    """Swizzle action_in/out_proj weights for num_cores=1 single-core SPM linear.

    K=32 for action_in_proj satisfies num_ele_32B=16 alignment (K%16==0). The
    8-core default converter (root-Linear walker) is bypassed — calling it
    AND this helper would double-swizzle (P1 pitfall). The transformed
    weight + bias are moved to RPU fp16. Caller MUST NOT also pass these
    modules through ``_convert_linear_weights_inplace``.

    Stored as ``_rpu_w_num_cores_1`` and ``_rpu_bias`` attributes (the bias
    name avoids colliding with the legacy ``bias`` parameter — the legacy
    CPU path may still need its CPU copy for the safety-net baseline run).

    Idempotent: skips if attrs already set.
    """
    import torch
    from rpu_backend.runtime.weights import transform_linear_weight

    for proj in (action_in_proj, action_out_proj):
        if hasattr(proj, "_rpu_w_num_cores_1") and hasattr(proj, "_rpu_bias"):
            continue
        orig_w = (proj.weight.detach().cpu()
                  .to(dtype=torch.float16).contiguous())
        transformed_w = (transform_linear_weight(orig_w, partition=1, num_cores=1)
                         .to('rpu').contiguous())
        if proj.bias is None:
            raise RuntimeError(
                "_swizzle_action_proj_weights: expected bias on action proj.")
        orig_b = (proj.bias.detach().cpu()
                  .to(dtype=torch.float16).to('rpu').contiguous())
        proj.register_buffer('_rpu_w_num_cores_1', transformed_w)
        proj.register_buffer('_rpu_bias', orig_b)


def _precompute_adarms_cond_all(policy, num_steps: int) -> "torch.Tensor":
    """Precompute the [num_steps, H_ada] adarms_cond tensor on RPU fp16.

    The conditioning is deterministic from the time schedule (independent of
    x_t), so we materialize all num_steps cond vectors up front. Mirrors the
    PERF E5 sin_lut precompute in runtime.py::_run_denoise.

    B1·A (2026-05-27): batched over num_steps to collapse 2N CPU linear
    dispatches into 2 GEMMs. Spec §4 + Rev 4 corrections C1/C2. tvs uses
    list-comprehension to bit-match `1.0 + s*dt` scalar sequence; sin_emb
    cast back to fp32 before time_mlp_in to avoid fp64 leak from upstream
    create_sinusoidal_pos_embedding intermediates.
    """
    import torch
    import torch.nn.functional as F
    from lerobot.policies.pi05.modeling_pi05 import create_sinusoidal_pos_embedding

    cache = getattr(policy, "_rpu_adarms_cond_cache", None)
    cache_key = int(num_steps)
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    cfg = policy.config
    dt = -1.0 / num_steps
    device = next(policy.time_mlp_in.parameters()).device
    sin_dim = policy.action_in_proj.out_features

    # Rev 4 C2: list-comprehension精确复刻原标量序列 `1.0 + s * dt`,
    # 避免 arange*dt+1.0 在某些 num_steps 下的浮点序差异.
    tvs = torch.tensor([1.0 + s * dt for s in range(num_steps)],
                       dtype=torch.float32, device=device)               # [num_steps]
    sin_emb = create_sinusoidal_pos_embedding(
        tvs, sin_dim, cfg.min_period, cfg.max_period, device=device)     # [num_steps, sin_dim]
    # Rev 4 C1: 保留原 .type(tvs.dtype) cast. upstream 在 CPU 上用 float64
    # intermediates (modeling_pi05.py:88 get_safe_dtype), 不 cast 会把 fp64
    # 喂进 fp32 time_mlp_in.
    sin_emb = sin_emb.type(tvs.dtype)                                    # fp32

    x = policy.time_mlp_in(sin_emb)                                      # [num_steps, hidden]
    x = F.silu(x)
    x = policy.time_mlp_out(x)                                           # [num_steps, H_ada]
    cond_all = F.silu(x)                                                 # [num_steps, H_ada]
    cond_all_rpu = cond_all.to(dtype=torch.float16, device='rpu')
    if cache is not None:
        cache[cache_key] = cond_all_rpu
    return cond_all_rpu


@torch.no_grad()
def _precompute_adarms_table(expert, cond_all, *, dense_w8a16: bool):
    """CPU-compute fixed-schedule modulation, using the installed dense dtype.

    Layout is [steps, attention/MLP per layer then final norm, scale/shift/gate].
    Like RhinoVLA's table, CPU fp32 accumulation can differ from the RPU ACC16
    GEMV/ring. This is a numerical optimization candidate, not bitwise parity.
    The native cold setter takes its own RPU copy and seals the step count.
    """
    from rpu_backend.quant._common import (
        quantize_linear_per_channel, dequantize_linear_per_channel,
    )

    cond = cond_all.detach().to(device="cpu", dtype=torch.float16).float()
    if cond.ndim != 2 or not cond.numel() or not torch.isfinite(cond).all():
        raise ValueError("Pi0.5 AdaRMS conditioning must be finite nonempty [steps,H]")
    width = cond.shape[1]
    norms = [norm for layer in expert.layers for norm in (
        layer.input_layernorm, layer.post_attention_layernorm)] + [expert.norm]
    rows = []
    for norm in norms:
        weight = norm.dense.weight.detach().to(device="cpu", dtype=torch.float16)
        bias = norm.dense.bias.detach().to(device="cpu", dtype=torch.float16)
        if weight.shape != (3 * width, width) or bias.shape != (3 * width,):
            raise ValueError("Pi0.5 AdaRMS dense must have weight [3H,H], bias [3H]")
        if dense_w8a16:
            weight = dequantize_linear_per_channel(*quantize_linear_per_channel(weight))
        row = torch.nn.functional.linear(cond, weight.float(), bias.float()).half()
        row[:, :width] += 1.0
        rows.append(row)
    table = torch.stack(rows, dim=1).contiguous()
    if not torch.isfinite(table).all():
        raise ValueError("Pi0.5 AdaRMS table contains nonfinite values")
    return table
