"""Pi0.5 weight preparation for root Linear modules and the multimodal projector."""
from __future__ import annotations
import torch
import torch.nn as nn

from rpu_backend.runtime.log import _LOG


# A leaf `nn.Linear` has no named children, so it must call
# `transform_linear_weight` directly rather than relying on a recursive walker.
def _swizzle_leaf_linear(linear: nn.Linear, *, force_col_partition: bool = False) -> int:
    """Swizzle a leaf `nn.Linear`'s weight data IN-PLACE via transform_linear_weight.

    Returns the partition chosen (0 = row, 1 = col, -1 = skipped/fallback).

    Uses the normal partition selection for a non-attention,
    non-{'dense','o_proj','down_proj'} Linear.
    """
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


def swizzle_projector_inplace(projector) -> None:
    """Pre-swizzle the Pi0.5 SigLIP projector exactly once.

    Sets `_rpu_weights_converted` sentinel BEFORE move-to-RPU so
    `_convert_siglip_projector_for_rpu` observes the idempotence marker.
    """
    if hasattr(projector, 'linear') and isinstance(projector.linear, nn.Linear):
        _swizzle_leaf_linear(projector.linear)
        projector._rpu_weights_converted = True
        projector.linear._rpu_weights_converted = True
    elif isinstance(projector, nn.Linear):
        _swizzle_leaf_linear(projector)
        projector._rpu_weights_converted = True


# Pi0.5-specific safetensors-key remapping from LeRobot keys to the HF layout.
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


# Pi0.5 fused num_steps weight helpers.

def _prepare_final_norm_weights(expert) -> None:
    """Swizzle the model's final PiGemmaRMSNorm dense for AdaRMS-style GEMV.

    The per-layer AdaRMS converter (adarms.py:_prepare_adarms_dense_weights)
    already registers ``norm._rpu_dense_w_rp`` / ``_rpu_dense_b_rp`` on each
    layer's input/post_attention layernorm. The final norm (``expert.norm``)
    is not touched by that helper — it stays on CPU and runs in Python fp32
    by default. For the fused denoise step we need it on RPU
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
                                              num_cores=NUM_CORES)
                     .to('rpu').contiguous())
    orig_b = (norm.dense.bias.detach().cpu()
              .to(dtype=torch.float16).to('rpu').contiguous())
    norm.register_buffer('_rpu_dense_w_rp', transformed_w)
    norm.register_buffer('_rpu_dense_b_rp', orig_b)


def _swizzle_action_proj_weights(action_in_proj, action_out_proj) -> None:
    """Swizzle action_in/out_proj weights for num_cores=1 single-core SPM linear.

    K=32 for action_in_proj satisfies num_ele_32B=16 alignment (K%16==0). The
    8-core default converter (root-Linear walker) is bypassed — calling it
    and this helper would double-swizzle. The transformed
    weight + bias are moved to RPU fp16. Caller MUST NOT also pass these
    modules through ``_convert_linear_weights_inplace``.

    Stored as ``_rpu_w_num_cores_1`` and ``_rpu_bias`` attributes (the bias
    name avoids colliding with the ``bias`` parameter; the CPU path may still
    need its CPU copy).

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
    x_t). ``tvs`` preserves the scalar ``1.0 + s*dt`` sequence, and sin_emb is
    cast to fp32 before time_mlp_in.
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

    # list-comprehension 精确复刻原标量序列 `1.0 + s * dt`,
    # 避免 arange*dt+1.0 在某些 num_steps 下的浮点序差异.
    tvs = torch.tensor([1.0 + s * dt for s in range(num_steps)],
                       dtype=torch.float32, device=device)               # [num_steps]
    sin_emb = create_sinusoidal_pos_embedding(
        tvs, sin_dim, cfg.min_period, cfg.max_period, device=device)     # [num_steps, sin_dim]
    # 保留 .type(tvs.dtype) cast；upstream 在 CPU 上用 float64
    # intermediates，不 cast 会把 fp64
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
