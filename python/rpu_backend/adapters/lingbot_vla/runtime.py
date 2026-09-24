"""LingBot-VLA (robbyant/lingbot-vla-4b) 的 image+text → action 运行时。

CPU FP32 Qwen2.5-VL ViT 产生视觉 embedding，与语言 embedding 拼接后，
RPU Qwen2 decoder 以双向 attention 写入共享 prefix KV。
state/noise 经 encoder 与逐步 AdaRMS 变换，交给 RPU action expert
执行 Euler denoise。该 checkpoint 的 vision 保持 CPU FP32 以避免
FP16 溢出。

VLM 只需填充一次 prefix KV；expert 跨 handle 读取该 DDR prefix，
仅覆写自己的 suffix。共享数据要求 DDR flush 与相应生命周期有效。"""
from __future__ import annotations
import glob
import math
import os
from contextlib import nullcontext
from collections.abc import Mapping
from numbers import Integral
import torch
import torch.nn.functional as F
from safetensors import safe_open

import rpu_backend
from rpu_backend.api import RPUCache
from rpu_backend.runtime.control import rpu_env_bool
from rpu_backend.api._execution import (
    bind_execution_session,
    execution_serialized,
    native_execution_reconfigure,
    normalize_rpu_execution,
    resolve_component_rpu_execution,
)
from rpu_backend.adapters.lingbot_vla.profile import (
    ACTION_DIM,
    ATTN_TP,
    DEFAULT_MAX_SEQ,
    EXP_HID,
    EXP_INTER,
    EXP_INTER_PAD,
    EXP_LAYERS,
    EXP_P,
    HD,
    MLP_CORES,
    N_ACTION,
    NKV,
    NQ,
    NUM_STEPS,
    RMS_EPS,
    ROPE_BASE,
    VIS_P,
    VLM_HID,
    VLM_INTER,
    VLM_LAYERS,
    VLM_P,
    V_DEPTH,
    V_FULLATT,
    V_HD,
    V_HEADS,
    V_HID,
    V_OUT,
    V_PATCH,
    V_SM,
    V_SMU,
    V_THETA,
    V_WINDOW,
    _expected_checkpoint_shapes,
    _runtime_options_from_env,
    _validate_checkpoint_profile,
)
from rpu_backend.runtime.decoder import (
    chunk_policy_key,
    plan_bounded_prefill_execution,
)
from rpu_backend.runtime.execution_planner import (
    GRAPH_COMPOSITE_CHILD,
    PlannerRejectError,
)
from rpu_backend.runtime.weights import tp_col_swizzle_mc_weight, tp_row_swizzle_mc_weight


def _destroy_causal_decoder_handle(handle) -> None:
    """Raw destruction: the owning retirement transaction handles failures."""
    if handle is not None:
        torch.ops.rpu.causal_decoder_destroy(handle)


def _destroy_denoise_handle(handle) -> None:
    if handle is not None:
        torch.ops.rpu.lingbot_denoise_destroy(handle)


# Failed native cleanup must retain the graph/tensor addresses even after GC.
# The process-terminal latch prevents another policy from adding resources.
_FAILED_RETIREMENTS = []


def _poison_lingbot_retirement(owner, error):
    if getattr(owner, "_cleanup_ok", None) is not False:
        owner._cleanup_ok = False
        _FAILED_RETIREMENTS.append(owner)
        session = getattr(owner, "_execution_session", None)
        if session is not None:
            session.poison()
    from rpu_backend.api import causal_lm
    from rpu_backend.api import _execution

    with causal_lm._LIVE_LOCK:
        if causal_lm._LIVE_TERMINAL_REASON is None:
            live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
            if live is None:
                live = owner
                causal_lm._claim_live_instance(live)
            causal_lm._poison_live_instance(live, f"LingBot-VLA cleanup failed: {error!r}", unsafe=True)
        # A previous normal process-terminal owner still needs the stronger
        # unsafe latch when an old owner's retirement fails after a handoff.
        _execution._mark_execution_process_unsafe(f"LingBot-VLA cleanup failed: {error!r}")


def _cleanup_lingbot_build(owner, error):
    try:
        owner._retire_resources()
    except BaseException as cleanup_error:
        error.add_note(f"LingBot-VLA builder cleanup also failed: {cleanup_error!r}")


def _cleanup_lingbot_pending_handle(handle, attr, keep, error):
    # Configuration failed before the public builder could acquire the handle.
    # Use the same retirement path, including failed-resource retention.
    pending = LingbotVlaPolicy.__new__(LingbotVlaPolicy)
    pending._cleanup_ok = None
    setattr(pending, attr, handle)
    pending._pending_keep = keep
    _cleanup_lingbot_build(pending, error)


def _h(t):
    """fp32/any → fp16 RPU contiguous."""
    return t.detach().to(torch.float16).to("rpu").contiguous()


# CPU FP32 Qwen2.5-VL vision tower，使用普通 Torch 与 eager attention。
def _vit_rmsnorm(x, weight, eps=RMS_EPS):
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return weight * x


def _vit_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _vit_apply_rope(q, k, cos, sin):
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_vit_rotate_half(q) * sin)
    k_embed = (k * cos) + (_vit_rotate_half(k) * sin)
    return q_embed, k_embed


def _vit_rot_pos_emb(grid_thw):
    rot_dim = V_HD // 2                                            # 40
    inv_freq = 1.0 / (V_THETA ** (torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim))
    sms = V_SM
    pos_ids = []
    for t, h, w in grid_thw.tolist():
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(h // sms, sms, w // sms, sms).permute(0, 2, 1, 3).flatten()
        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(h // sms, sms, w // sms, sms).permute(0, 2, 1, 3).flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0)
    max_grid_size = int(grid_thw[:, 1:].max())
    seq = torch.arange(max_grid_size, dtype=torch.float32)
    rotary_full = torch.outer(seq, inv_freq)
    rotary = rotary_full[pos_ids].flatten(1)                      # [seq, 40]
    return rotary


def _vit_get_window_index(grid_thw):
    window_index = []
    cu_window_seqlens = [0]
    window_index_id = 0
    vit_merger_window_size = V_WINDOW // V_SM // V_PATCH          # 4
    for grid_t, grid_h, grid_w in grid_thw.tolist():
        llm_grid_h = grid_h // V_SM
        llm_grid_w = grid_w // V_SM
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            grid_t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size)
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size)
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index.append(index_new + window_index_id)
        cu_seqlens_tmp = seqlens.cumsum(0) * V_SMU + cu_window_seqlens[-1]
        cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
        window_index_id += grid_t * llm_grid_h * llm_grid_w
    window_index = torch.cat(window_index, dim=0)
    return window_index, cu_window_seqlens


def _vit_preprocess_grid(grid_thw):
    rotary = _vit_rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = _vit_get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(cu_window_seqlens, dtype=torch.int32)
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    return rotary, window_index, cu_window_seqlens, cu_seqlens


def _vit_attention(x, cu_seqlens, cos, sin, W, p):
    seq = x.shape[0]
    qkv = F.linear(x, W[p + "attn.qkv.weight"], W[p + "attn.qkv.bias"])
    qkv = qkv.reshape(seq, 3, V_HEADS, V_HD).permute(1, 0, 2, 3)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = _vit_apply_rope(q, k, cos, sin)
    mask = torch.full((1, seq, seq), torch.finfo(torch.float32).min, dtype=torch.float32)
    cu = cu_seqlens.tolist()
    for i in range(1, len(cu)):
        mask[..., cu[i - 1]:cu[i], cu[i - 1]:cu[i]] = 0
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(V_HD)
    attn = attn + mask
    attn = torch.softmax(attn, dim=-1, dtype=torch.float32)
    out = torch.matmul(attn, v)
    out = out.transpose(0, 1).reshape(seq, -1)
    out = F.linear(out, W[p + "attn.proj.weight"], W[p + "attn.proj.bias"])
    return out


def _vit_mlp(x, W, p):
    gate = F.linear(x, W[p + "mlp.gate_proj.weight"], W[p + "mlp.gate_proj.bias"])
    up = F.linear(x, W[p + "mlp.up_proj.weight"], W[p + "mlp.up_proj.bias"])
    return F.linear(F.silu(gate) * up, W[p + "mlp.down_proj.weight"], W[p + "mlp.down_proj.bias"])


def _vit_block(x, cu_seqlens, cos, sin, W, p):
    x = x + _vit_attention(_vit_rmsnorm(x, W[p + "norm1.weight"]), cu_seqlens, cos, sin, W, p)
    x = x + _vit_mlp(_vit_rmsnorm(x, W[p + "norm2.weight"]), W, p)
    return x


def _vit_merger(x, W):
    x = _vit_rmsnorm(x, W[VIS_P + "merger.ln_q.weight"])
    x = x.view(-1, V_HID * V_SMU)
    x = F.linear(x, W[VIS_P + "merger.mlp.0.weight"], W[VIS_P + "merger.mlp.0.bias"])
    x = F.gelu(x)
    x = F.linear(x, W[VIS_P + "merger.mlp.2.weight"], W[VIS_P + "merger.mlp.2.bias"])
    return x


def run_vit(images, grid_thw, W):
    """CPU-fp32 Qwen2.5-VL vision tower → img_emb [1, n_out_tokens, 2048] (raster order)."""
    grid_thw = torch.as_tensor(grid_thw, dtype=torch.long)
    pixel_values = images.reshape(-1, images.shape[-1]).to(torch.float32).contiguous()
    pe_w = W[VIS_P + "patch_embed.proj.weight"].reshape(V_HID, -1).to(torch.float32)
    hs = F.linear(pixel_values, pe_w)

    rotary, window_index, cu_window_seqlens, cu_seqlens = _vit_preprocess_grid(grid_thw)
    seq = hs.shape[0]
    hs = hs.reshape(seq // V_SMU, V_SMU, -1)[window_index, :, :].reshape(seq, -1)
    rotary = rotary.reshape(seq // V_SMU, V_SMU, -1)[window_index, :, :].reshape(seq, -1)
    emb = torch.cat((rotary, rotary), dim=-1)
    cos, sin = emb.cos(), emb.sin()

    for L in range(V_DEPTH):
        cu = cu_seqlens if L in V_FULLATT else cu_window_seqlens
        hs = _vit_block(hs, cu, cos, sin, W, f"{VIS_P}blocks.{L}.")

    hs = _vit_merger(hs, W)
    reverse_indices = torch.argsort(window_index)
    hs = hs[reverse_indices, :]
    split_sizes = (grid_thw.prod(-1) // V_SMU).tolist()
    hs = torch.stack(torch.split(hs, split_sizes), dim=0)        # [num_images, n_tokens, 2048]
    return hs


# =============================================================================
# Host-fp32 encoders + AdaRMS fold (copied from the validated cpu_ref.py / verify_p3_expert.py)
# =============================================================================
def _sinusoidal_pos_embedding(time, dim, min_period, max_period):
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2 * math.pi
    sin_input = scaling[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def _make_att_2d_masks(pad_masks, att_masks):
    cumsum = torch.cumsum(att_masks.to(torch.int64), dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d & pad_2d


def embed_suffix(state, x_t, t, W):
    """(time_emb [1,768] = raw sinusoidal = ada_cond, suffix_embs [1,51,768]). Host fp32."""
    state_emb = F.linear(state, W["model.state_proj.weight"], W["model.state_proj.bias"])
    time_emb = _sinusoidal_pos_embedding(t, EXP_HID, min_period=4e-3, max_period=4.0)
    action_emb = F.linear(x_t, W["model.action_in_proj.weight"], W["model.action_in_proj.bias"])
    te_exp = time_emb[:, None, :].expand(-1, action_emb.shape[1], -1)
    ate = torch.cat([action_emb, te_exp], dim=-1)
    ate = F.linear(ate, W["model.action_time_mlp_in.weight"], W["model.action_time_mlp_in.bias"])
    ate = F.silu(ate)
    ate = F.linear(ate, W["model.action_time_mlp_out.weight"], W["model.action_time_mlp_out.bias"])
    suffix_embs = torch.cat([state_emb[:, None], ate], dim=1)
    return time_emb, suffix_embs


def embed_suffix_step(x_t, state_emb, time_emb, W):
    """Per-Euler-step suffix embed with the CALL-INVARIANT pieces (state_emb, time_emb)
    precomputed once. Only action_in_proj + the time-MLP depend on x_t. Bit-identical to
    embed_suffix's suffix_embs. Cuts the per-step state_proj + sinusoidal (host)."""
    action_emb = F.linear(x_t, W["model.action_in_proj.weight"], W["model.action_in_proj.bias"])
    te_exp = time_emb[:, None, :].expand(-1, action_emb.shape[1], -1)
    ate = torch.cat([action_emb, te_exp], dim=-1)
    ate = F.linear(ate, W["model.action_time_mlp_in.weight"], W["model.action_time_mlp_in.bias"])
    ate = F.silu(ate)
    ate = F.linear(ate, W["model.action_time_mlp_out.weight"], W["model.action_time_mlp_out.bias"])
    return torch.cat([state_emb[:, None], ate], dim=1)


def fold_scale_shift(norm_W, cond):
    """Per-(layer,norm-slot): scale=(1+gamma(cond))*rms_w, shift=beta(cond). cond [1,768].
    Returns 4 lists (input_scale, input_shift, post_scale, post_shift) of [768] fp16 RPU vectors."""
    in_s, in_sh, po_s, po_sh = [], [], [], []
    for L in range(EXP_LAYERS):
        for slot, sc, sh in (("input_layernorm", in_s, in_sh),
                             ("post_attention_layernorm", po_s, po_sh)):
            pf = f"{EXP_P}layers.{L}.{slot}"
            w = norm_W[pf + ".weight"]
            gamma = F.linear(cond, norm_W[pf + ".gamma.weight"], norm_W[pf + ".gamma.bias"])[0]
            beta = F.linear(cond, norm_W[pf + ".beta.weight"], norm_W[pf + ".beta.bias"])[0]
            sc.append(_h((1.0 + gamma) * w))
            sh.append(_h(beta))
    return in_s, in_sh, po_s, po_sh


def precompute_fold_all(norm_W, times):
    """Lever 1 (PERF_ROADMAP.md): batch the AdaRMS fold across all Euler steps.
    cond=sinusoidal(t) depends only on t (NOT x_t), so all steps precompute once and
    ship in 4 big `.to("rpu")` transfers instead of 36*2*2*num_steps tiny [768] ones
    (~1440 → 4, was ~2.9 s/call). Returns 4 rpu tensors [S, EXP_LAYERS, 768];
    slice `t[k]` per step → 36 contiguous [768] views for causal_decoder_set_adarms_step.
    Bit-identical to per-step fold_scale_shift (same math, same fp16 cast)."""
    S = len(times)
    in_s = torch.empty(S, EXP_LAYERS, EXP_HID, dtype=torch.float16)
    in_sh, po_s, po_sh = (torch.empty_like(in_s) for _ in range(3))
    for k, t in enumerate(times):
        cond = _sinusoidal_pos_embedding(torch.tensor([float(t)]), EXP_HID, 4e-3, 4.0)
        for L in range(EXP_LAYERS):
            for slot, sc, sh in (("input_layernorm", in_s, in_sh),
                                 ("post_attention_layernorm", po_s, po_sh)):
                pf = f"{EXP_P}layers.{L}.{slot}"
                w = norm_W[pf + ".weight"]
                gamma = F.linear(cond, norm_W[pf + ".gamma.weight"], norm_W[pf + ".gamma.bias"])[0]
                beta = F.linear(cond, norm_W[pf + ".beta.weight"], norm_W[pf + ".beta.bias"])[0]
                sc[k, L] = ((1.0 + gamma) * w).half()
                sh[k, L] = beta.half()
    return (in_s.to("rpu"), in_sh.to("rpu"), po_s.to("rpu"), po_sh.to("rpu"))


# =============================================================================
# RPU handle builders (copied from verify_p1_decoder.build_rpu_decoder /
# verify_p3_expert.build_rpu_expert; consume the checkpoint's projection weights).
# =============================================================================
def _build_rope_tables(head_dim, theta, max_seq):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    pos = torch.arange(max_seq, dtype=torch.float64)
    freqs = pos[:, None] * inv_freq[None, :]
    return _h(freqs.cos().float()), _h(freqs.sin().float())


_VLM_CERTIFIED_MAX_PREFIX = 178
_VLM_EXACT_CHUNK_CEILING = 176
_ACTION_CHUNK_SIZE = 64

_VISION_COMPONENT = "vision_encoder"
_VLM_COMPONENT = "language_model"
_ACTION_COMPONENT = "action_expert"
_EXECUTION_COMPONENTS = {
    # This checkpoint's Vision tower is intentionally CPU-fp32.  Keeping its
    # stable ID with an empty capability map makes that N/A explicit while the
    # shared parser rejects every physical Vision knob.
    _VISION_COMPONENT: {},
    _VLM_COMPONENT: {"prefill": ("chunk_size",)},
    _ACTION_COMPONENT: {"action": ("chunk_size",)},
}


def _resolve_execution_components(value, *, entry_point):
    """Normalize one public config and resolve all three stable children."""
    root = normalize_rpu_execution(
        value,
        entry_point=entry_point,
        supported_components=_EXECUTION_COMPONENTS,
    )
    vision = resolve_component_rpu_execution(
        root,
        _VISION_COMPONENT,
        entry_point=entry_point,
        supported_components=_EXECUTION_COMPONENTS,
        profile_auto={},
    )
    vlm = resolve_component_rpu_execution(
        root,
        _VLM_COMPONENT,
        entry_point=entry_point,
        supported_components=_EXECUTION_COMPONENTS,
        profile_auto={"prefill": {"chunk_size": "auto"}},
    )
    action = resolve_component_rpu_execution(
        root,
        _ACTION_COMPONENT,
        entry_point=entry_point,
        supported_components=_EXECUTION_COMPONENTS,
        profile_auto={"action": {"chunk_size": "auto"}},
    )
    if vision:
        raise RuntimeError("LingBot-VLA CPU Vision unexpectedly resolved physical controls")
    return root, vlm, action


def _native_chunk(config, stage):
    value = config.get(stage, {}).get("chunk_size", "auto")
    return 0 if value == "auto" else int(value)


def _validate_resolved_execution(vlm, action, *, entry_point):
    prefill_chunk = _native_chunk(vlm, "prefill")
    if prefill_chunk > _VLM_EXACT_CHUNK_CEILING:
        raise ValueError(
            f"{entry_point}: explicit prefill chunk_size exceeds the certified "
            f"ceiling {_VLM_EXACT_CHUNK_CEILING}; got {prefill_chunk}"
        )
    action_chunk = _native_chunk(action, "action")
    if action_chunk not in (0, _ACTION_CHUNK_SIZE):
        raise ValueError(
            f"{entry_point}: explicit action chunk_size must be 64 for the "
            f"fixed {N_ACTION + 1}-row suffix; got {action_chunk}"
        )
    return prefill_chunk, action_chunk


def _build_vlm_decoder(gw, max_seq, chunk_size=0):
    """RPU plain-Qwen2 VLM decoder (1D RoPE base 10000, mrope=[], bidirectional). → (handle, keep)."""
    q_w, k_w, v_w, o_w = [], [], [], []
    q_b, k_b, v_b = [], [], []
    gate_l, up_l, down_l = [], [], []
    in_norm, post_norm = [], []
    for L in range(VLM_LAYERS):
        p = f"{VLM_P}layers.{L}."
        q_w.append(_h(tp_col_swizzle_mc_weight(gw(p + "self_attn.q_proj.weight").half(), ATTN_TP)))
        k_w.append(_h(tp_col_swizzle_mc_weight(gw(p + "self_attn.k_proj.weight").half(), ATTN_TP)))
        v_w.append(_h(tp_col_swizzle_mc_weight(gw(p + "self_attn.v_proj.weight").half(), ATTN_TP)))
        o_w.append(_h(tp_row_swizzle_mc_weight(gw(p + "self_attn.o_proj.weight").half(), ATTN_TP)))
        q_b.append(_h(gw(p + "self_attn.q_proj.bias")))
        k_b.append(_h(gw(p + "self_attn.k_proj.bias")))
        v_b.append(_h(gw(p + "self_attn.v_proj.bias")))
        gate_l.append(_h(tp_col_swizzle_mc_weight(gw(p + "mlp.gate_proj.weight").half(), MLP_CORES)))
        up_l.append(_h(tp_col_swizzle_mc_weight(gw(p + "mlp.up_proj.weight").half(), MLP_CORES)))
        down_l.append(_h(tp_row_swizzle_mc_weight(gw(p + "mlp.down_proj.weight").half(), MLP_CORES)))
        in_norm.append(_h(gw(p + "input_layernorm.weight")))
        post_norm.append(_h(gw(p + "post_attention_layernorm.weight")))
    cos, sin = _build_rope_tables(HD, ROPE_BASE, max_seq)
    final_norm_w = _h(gw(VLM_P + "norm.weight"))
    keep = (q_w, k_w, v_w, o_w, q_b, k_b, v_b, gate_l, up_l, down_l,
            in_norm, post_norm, cos, sin, final_norm_w)
    handle = torch.ops.rpu.causal_decoder_create(False, False)
    try:
        torch.ops.rpu.causal_decoder_set_weights(
            handle, q_w, k_w, v_w, o_w, [], [], in_norm, post_norm,
            gate_l, up_l, down_l, cos, sin, final_norm_w,
            NQ, NKV, HD, VLM_HID, VLM_INTER, RMS_EPS, True, [], [], q_b, k_b, v_b)
        # Capability is independent of the current AUTO/EXACT request. Keep
        # the certified ceiling on both so later public chunk transactions
        # need not change this cold envelope. Native SDPA/SPM still filter each
        # concrete shape; a ceiling is not admission of every smaller chunk.
        envelope_max_kv = min(int(max_seq), _VLM_CERTIFIED_MAX_PREFIX)
        torch.ops.rpu.causal_decoder_set_chunk_envelope(
            handle, envelope_max_kv, _VLM_EXACT_CHUNK_CEILING)
        torch.ops.rpu.causal_decoder_set_chunk_size_override(
            handle, int(chunk_size))
        return handle, keep
    except BaseException as build_error:
        _cleanup_lingbot_pending_handle(handle, "_vlm", keep, build_error)
        raise


def _build_expert(gw, max_seq, chunk_size=0):
    """RPU AdaRMS Qwen2 action-expert (768-d, mlp-pad 2816, 1D RoPE base 10000). → (handle, keep).
    AdaRMS scale/shift are set per Euler step via causal_decoder_set_adarms_step."""
    q_w, k_w, v_w, o_w = [], [], [], []
    q_b, k_b, v_b = [], [], []
    gate_l, up_l, down_l = [], [], []
    in_norm, post_norm = [], []
    npad = EXP_INTER_PAD - EXP_INTER
    for L in range(EXP_LAYERS):
        p = f"{EXP_P}layers.{L}."
        q_w.append(_h(tp_col_swizzle_mc_weight(gw(p + "self_attn.q_proj.weight").half(), ATTN_TP)))
        k_w.append(_h(tp_col_swizzle_mc_weight(gw(p + "self_attn.k_proj.weight").half(), ATTN_TP)))
        v_w.append(_h(tp_col_swizzle_mc_weight(gw(p + "self_attn.v_proj.weight").half(), ATTN_TP)))
        o_w.append(_h(tp_row_swizzle_mc_weight(gw(p + "self_attn.o_proj.weight").half(), ATTN_TP)))
        q_b.append(_h(gw(p + "self_attn.q_proj.bias")))
        k_b.append(_h(gw(p + "self_attn.k_proj.bias")))
        v_b.append(_h(gw(p + "self_attn.v_proj.bias")))
        gw_ = F.pad(gw(p + "mlp.gate_proj.weight").half(), (0, 0, 0, npad))
        uw_ = F.pad(gw(p + "mlp.up_proj.weight").half(), (0, 0, 0, npad))
        dw_ = F.pad(gw(p + "mlp.down_proj.weight").half(), (0, npad))
        gate_l.append(_h(tp_col_swizzle_mc_weight(gw_, MLP_CORES)))
        up_l.append(_h(tp_col_swizzle_mc_weight(uw_, MLP_CORES)))
        down_l.append(_h(tp_row_swizzle_mc_weight(dw_, MLP_CORES)))
        in_norm.append(_h(gw(p + "input_layernorm.weight")))     # placeholder; overridden per step
        post_norm.append(_h(gw(p + "post_attention_layernorm.weight")))
    cos, sin = _build_rope_tables(HD, ROPE_BASE, max_seq)
    final_norm_w = _h(gw(EXP_P + "norm.weight"))                 # plain RMS (final_norm_adanorm=False)
    keep = (q_w, k_w, v_w, o_w, q_b, k_b, v_b, gate_l, up_l, down_l,
            in_norm, post_norm, cos, sin, final_norm_w)
    handle = torch.ops.rpu.causal_decoder_create(
        False, rpu_env_bool("RPU_ADARMS_FUSED_BCAST"))
    try:
        torch.ops.rpu.causal_decoder_set_weights(
            handle, q_w, k_w, v_w, o_w, [], [], in_norm, post_norm,
            gate_l, up_l, down_l, cos, sin, final_norm_w,
            NQ, NKV, HD, EXP_HID, EXP_INTER_PAD, RMS_EPS, True, [], [], q_b, k_b, v_b)
        # The action suffix is structurally 51 rows and must remain one
        # ceil16-aligned chunk. Auto and explicit requests therefore share 64.
        torch.ops.rpu.causal_decoder_set_chunk_envelope(
            handle, int(max_seq), _ACTION_CHUNK_SIZE)
        torch.ops.rpu.causal_decoder_set_chunk_size_override(
            handle, int(chunk_size))
        return handle, keep
    except BaseException as build_error:
        _cleanup_lingbot_pending_handle(handle, "_exp", keep, build_error)
        raise


def _build_suffix_mask(prefix_len):
    """Additive fp16 [51, prefix_len+51]: 0 attend, big_neg masked.
    prefix fully visible; suffix state(tok0)⊥action, action bidirectional."""
    B, S = 1, N_ACTION + 1                                       # 51
    suffix_att = torch.zeros(B, S, dtype=torch.bool); suffix_att[:, :2] = True
    suffix_pad = torch.ones(B, S, dtype=torch.bool)
    suffix_2d = _make_att_2d_masks(suffix_pad, suffix_att)       # [1,51,51]
    prefix_2d = torch.ones(B, S, prefix_len, dtype=torch.bool)   # [1,51,prefix_len]
    full = torch.cat([prefix_2d, suffix_2d], dim=2)[0]           # [51, prefix_len+51] bool
    add = torch.where(full, torch.tensor(0.0), torch.tensor(-50000.0)).half()
    return add.to("rpu").contiguous()


# =============================================================================
# Policy
# =============================================================================
class LingbotVlaPolicy:
    """Image+text → action [1,50,75] runtime. Build via `build_lingbot_vla(...)`."""

    _rpu_execution_capabilities = {
        "prefill": ("chunk_size",),
        "action": ("chunk_size",),
    }
    _rpu_execution_components = _EXECUTION_COMPONENTS

    def __init__(self, *, vis_W, head_W, norm_W, embed_w,
                 vlm_handle, vlm_keep, exp_handle, exp_keep,
                 max_seq=DEFAULT_MAX_SEQ, prefill_chunk_size=0,
                 action_chunk_size=0,
                 rpu_execution=None,
                 _runtime_options=None):
        # Establish cleanup state first: the public builder may call
        # _retire_resources() after any later constructor failure.
        self._closed = False
        self._cleanup_ok = None
        self._live_slot_released = False
        self._gc_retirement_enabled = True
        self._du_handle = None
        self._vis_W = vis_W          # CPU-fp32 ViT weights (dict keyed by full key)
        self._head_W = head_W        # CPU-fp32 encoder/head weights
        self._norm_W = norm_W        # CPU-fp32 expert AdaRMS norm/gamma/beta weights
        self._embed_w = embed_w      # CPU-fp32 [vocab, 2048] token embedding
        self._vlm = vlm_handle
        self._vlm_keep = vlm_keep
        self._exp = exp_handle
        self._exp_keep = exp_keep
        self._max_seq = max_seq
        root_execution, vlm_execution, action_execution = (
            _resolve_execution_components(
                rpu_execution,
                entry_point="LingbotVlaPolicy",
            )
        )
        resolved_prefill, resolved_action = _validate_resolved_execution(
            vlm_execution,
            action_execution,
            entry_point="LingbotVlaPolicy",
        )
        if (resolved_prefill, resolved_action) != (
            int(prefill_chunk_size), int(action_chunk_size)
        ):
            raise RuntimeError(
                "LingBot-VLA builder/component execution resolution drift"
            )
        self._rpu_execution = root_execution
        self._vlm_execution = vlm_execution
        self._action_execution = action_execution
        self._prefill_chunk_size = resolved_prefill
        self._action_chunk_override = resolved_action
        # Both expert paths have one physical 51-row -> C64 action ABI.  The
        # public AUTO value changes no physical geometry.
        self._action_chunk_size = _ACTION_CHUNK_SIZE
        self._execution_generation = 0
        self._execution_reconfigure_journal = None
        self._rpu_last_execution_plan = {}
        enable_reconfigure = getattr(
            torch.ops.rpu,
            "causal_decoder_enable_execution_reconfigure",
            None,
        )
        if enable_reconfigure is None:
            raise RuntimeError(
                "LingBot-VLA binary lacks causal-decoder hot-reconfigure support"
            )
        enable_reconfigure(self._vlm)
        enable_reconfigure(self._exp)
        # ONE shared RPUCache serves both the VLM (fills [0,S)) and the expert (reads [0,S), inserts [S,S+51)).
        self._cache = RPUCache(num_layers=VLM_LAYERS, batch_size=1, max_seq_len=max_seq,
                               num_kv_heads=NKV, head_dim=HD, attn_tp=ATTN_TP)
        self._vgc = rpu_backend.graph.GraphCache()
        self._egc = rpu_backend.graph.GraphCache()
        runtime_options = (
            _runtime_options_from_env()
            if _runtime_options is None
            else _runtime_options
        )
        # Lever 1 (PERF_ROADMAP.md): the AdaRMS fold is CALL-INVARIANT — cond=sinusoidal(t)
        # for the fixed NUM_STEPS timestep schedule, over fixed weights — so precompute it
        # ONCE at build and reuse every get_action. Removes ~2.9 s/call (2880 tiny host
        # F.linear that used to run per call). self._fold_all = 4 rpu tensors [NUM_STEPS,36,768].
        self._times, _t = [], 1.0
        for _ in range(NUM_STEPS):        # repeated add, matches the Euler loop's time schedule bit-for-bit
            self._times.append(_t)
            _t += -1.0 / NUM_STEPS
        self._fold_all = precompute_fold_all(norm_W, self._times)
        # Per-step sinusoidal time embeddings — call-invariant (fixed schedule) → precompute once.
        self._time_embs = [_sinusoidal_pos_embedding(torch.tensor([t]), EXP_HID, 4e-3, 4.0)
                           for t in self._times]
        # Build-once/replay expert denoise (RPU_LINGBOT_EXPERT_REPLAY, default on): one graph sig for
        # all NUM_STEPS steps + set_adarms_step_mutable (no per-step rebuild) + a STABLE suffix input
        # buffer (refreshed in place) → step 0 BUILDs, steps 1..N-1 REPLAY (fast-replay). Set =0 to
        # fall back to the per-step-build path (a distinct sig + set_adarms_step every step).
        self._expert_replay = runtime_options["expert_replay"]
        self._suffix_buf = torch.empty(1, N_ACTION + 1, EXP_HID, dtype=torch.float16, device="rpu")
        # 跨调用保留 VLM GraphCache 与稳定 prefix 输入缓冲。
        # 每个新 prefix 长度首次 BUILD，后续复用对应 Graph。
        # 二维缓冲的 [:S].unsqueeze(0) 保持连续布局和稳定基址。
        self._vlm_replay = runtime_options["vlm_replay"]
        self._prefix_buf = torch.empty(max_seq, VLM_HID, dtype=torch.float16, device="rpu")
        # Cross-call expert replay: with self._egc kept persistent, the expert graph (built on call 1,
        # step 0) replays on call 2+ — but only if its baked inputs stay at stable addresses. suffix_buf
        # + AdaRMS keepalives already are; the suffix MASK (per prefix length S) is cached here so its
        # DDR addr survives across calls (a fresh per-call mask would be freed → replay reads garbage).
        self._mask_cache: dict[int, torch.Tensor] = {}
        # Fused suffix encoder (RPU_LINGBOT_FUSED_ENCODER, default on): action_in_proj + time_mlp_in
        # collapse to ONE [action_dim→768] matmul (action) plus a CALL-INVARIANT per-step time bias,
        # ~2.8× fewer host FLOPs. Precomputed in fp32; a tiny matmul-reassociation drift vs embed_suffix
        # (verified action cos within gate). Set =0 to use the exact 3-linear embed_suffix.
        self._fused_encoder = runtime_options["fused_encoder"]
        # 局部限制 denoise host encoder 的线程数，减少小矩阵的并行派发。
        # 不影响此前的 CPU FP32 ViT；关闭该选项时使用调用方的线程设置。
        self._enc_1thread = runtime_options["encoder_1thread"]
        _Win = head_W["model.action_in_proj.weight"]                   # [768, action_dim]
        _bin = head_W["model.action_in_proj.bias"]                     # [768]
        _Wmi = head_W["model.action_time_mlp_in.weight"]              # [768, 1536]
        _bmi = head_W["model.action_time_mlp_in.bias"]               # [768]
        _A, _B = _Wmi[:, :EXP_HID], _Wmi[:, EXP_HID:]                 # action / time halves [768,768]
        self._enc_W_comp = (_A @ _Win).contiguous()                  # [768, action_dim]: h_action = x_t @ Wcompᵀ
        _bias_const = _bin @ _A.t() + _bmi                           # [768]
        self._enc_time_parts = [(_te @ _B.t() + _bias_const) for _te in self._time_embs]  # per step [1,768]
        # 设备 encoder 路径将 suffix encoder、out_proj 与 Euler 放到 RPU，
        # 使 x_t 在循环间保留于设备。权重采用 col-swizzle；8 核布局要求输出
        # 按 128、K 按 16 对齐，逻辑维 75 分别补到 128/80。
        # 设备 FP16 与 host FP32 encoder 的数值路径不同。
        self._ondevice_enc = runtime_options["ondevice_encoder"]
        if self._ondevice_enc:
            _aow = head_W["model.action_out_proj.weight"]; _aob = head_W["model.action_out_proj.bias"]
            self._od_ad = _aow.shape[0]                                        # 75
            _op = ((self._od_ad + 127) // 128) * 128                           # 128
            self._od_op_w = tp_col_swizzle_mc_weight(F.pad(_aow, (0, 0, 0, _op - self._od_ad)).half(), MLP_CORES).to("rpu").contiguous()
            self._od_op_b = _h(F.pad(_aob, (0, _op - self._od_ad)))
            self._od_kd = ((self._enc_W_comp.shape[1] + 15) // 16) * 16       # 75→80 (K%16)
            self._od_wc_w = tp_col_swizzle_mc_weight(F.pad(self._enc_W_comp, (0, self._od_kd - self._enc_W_comp.shape[1])).half(), MLP_CORES).to("rpu").contiguous()
            self._od_mo_w = tp_col_swizzle_mc_weight(head_W["model.action_time_mlp_out.weight"].half(), MLP_CORES).to("rpu").contiguous()
            self._od_mo_b = _h(head_W["model.action_time_mlp_out.bias"])
            self._od_time_parts = [_h(_tp) for _tp in self._enc_time_parts]   # [1,768] RPU
            self._od_xt_pad = torch.zeros(1, N_ACTION, self._od_kd, dtype=torch.float16, device="rpu")  # x_t padded, cols[ad:]=0

        # 图内 unroll 将 Euler 循环、suffix encoder 与 out_proj 合并执行。
        # x_t 保留在 SPM，逐步 AdaRMS 参数按 body_iter 偏移读取。
        # 部分 host FP32 计算改为设备 FP16，因此不是逐位等价变换。
        self._denoise_unroll = runtime_options["denoise_unroll"]
        self._dgc = rpu_backend.graph.GraphCache()      # persistent ⇒ the unroll graph replays across calls
        if self._denoise_unroll:
            self._build_denoise_unroll()
        else:
            # The first action domain is resolved before step 0. Bind the cold
            # AdaRMS mode now, so that descriptor/profile identity already names
            # the operation the expert will execute. Mutable setup owns stable
            # native clones; per-step refreshes preserve their addresses and do
            # not invalidate an existing Graph. The plain path retains its
            # deliberate per-step setter/invalidation behavior.
            first_fold = tuple(table[0] for table in self._fold_all)
            if self._expert_replay:
                torch.ops.rpu.causal_decoder_set_adarms_step_mutable(self._exp, *first_fold)
            else:
                torch.ops.rpu.causal_decoder_set_adarms_step(self._exp, *(list(table) for table in first_fold))
        self._execution_session = bind_execution_session(
            self,
            self._rpu_execution,
            entry_point="LingbotVlaPolicy",
            supported_components=_EXECUTION_COMPONENTS,
            validate=self.validate_execution_reconfigure,
            apply=lambda old, new, generation, *, force_rebuild=False: self.apply_execution_reconfigure(
                new, generation, rollback_config=old
            ),
            rollback=lambda old, _new, generation: (
                self.rollback_execution_reconfigure(old, generation)
            ),
            graph_mode=GRAPH_COMPOSITE_CHILD,
        )

        for component in _EXECUTION_COMPONENTS:
            self._execution_session._bind_planner_owner(self, component)

    @property
    def last_rpu_execution_plan(self) -> dict[str, dict[str, int | str]]:
        """Return a detached copy of the last per-stage native execution plan."""
        return {
            stage: dict(fields)
            for stage, fields in self._rpu_last_execution_plan.items()
        }

    def _plan_prefill_execution(self, logical_len: int):
        """Resolve the complete native A6 winner before any mutation."""
        logical_len = int(logical_len)
        if logical_len > _VLM_CERTIFIED_MAX_PREFIX:
            raise ValueError(
                "LingBot-VLA prefill is certified only through logical prefix "
                f"length {_VLM_CERTIFIED_MAX_PREFIX}; got {logical_len}. "
                "The larger cache capacity is not evidence that this SPM/SDPA "
                "shape is executable."
            )
        largest_exact = ((logical_len + 15) // 16) * 16
        if self._prefill_chunk_size and self._prefill_chunk_size > largest_exact:
            raise ValueError(
                "LingBot-VLA exact prefill chunk_size cannot be honored for "
                f"logical_len={logical_len}: requested={self._prefill_chunk_size}, "
                f"maximum exact value is ceil16(logical_len)={largest_exact}."
            )
        plan_box = {}
        try:
            execution_len, resolved = plan_bounded_prefill_execution(
                logical_len,
                min(
                    _VLM_CERTIFIED_MAX_PREFIX,
                    int(self._max_seq) - (N_ACTION + 1),
                ),
                0,
                execution_owner=self,
                execution_component="language_model",
                execution_stage="prefill",
                execution_native=("causal_decoder", int(self._vlm)),
                position=0,
                alignment=1,
                padding_rows=0,
                exact_chunk_size=(self._prefill_chunk_size or None),
                resolve_stage_domain=lambda length: (
                    torch.ops.rpu.causal_decoder_resolve_prefill_stage_domain(
                        self._vlm, int(length), 0, False, 0, 0, 4, logical_len
                    )
                ),
                request_id="lingbot-v1:language_model:prefill",
                plan_result_sink=lambda result: plan_box.__setitem__(
                    "result", result
                ),
                graph_mode=GRAPH_COMPOSITE_CHILD,
                queue_owner_id=int(self._vlm),
                lease_owner_id=int(self._vlm),
                plan_signature=(False, 0, 0, 4),
                graph_cache=self._vgc if self._vlm_replay else None,
            )
        except PlannerRejectError as exc:
            raise ValueError(
                "LingBot-VLA prefill has no certified native plan for "
                f"logical_len={logical_len}, requested_chunk="
                f"{self._prefill_chunk_size or 'auto'}: {exc}"
            ) from exc
        if execution_len != logical_len:
            raise RuntimeError(
                "LingBot-VLA unpadded prefill planner changed execution length: "
                f"logical={logical_len}, execution={execution_len}"
            )
        if self._prefill_chunk_size and resolved != self._prefill_chunk_size:
            raise ValueError(
                "LingBot-VLA exact prefill chunk_size was not honored by the "
                f"native dry planner: requested={self._prefill_chunk_size}, "
                f"resolved={resolved}, logical_len={logical_len}."
            )
        result = plan_box["result"]
        if not result.selected.stage_tuple.physical_descriptor:
            raise RuntimeError(
                "LingBot-VLA prefill A6 winner has no native descriptor"
            )
        return result

    def _plan_action_execution(self, prefix_len: int):
        """Admit the fixed C64 expert through its native stage domain."""
        denoise_unroll = getattr(self, "_denoise_unroll", False)
        handle = self._du_handle if denoise_unroll else self._exp
        plan_box = {}
        execution_len, resolved = plan_bounded_prefill_execution(
            N_ACTION + 1,
            N_ACTION + 1,
            0,
            execution_owner=self,
            execution_component="action_expert",
            execution_stage="action",
            execution_native=(
                "lingbot_denoise" if denoise_unroll else "causal_decoder", int(handle)
            ),
            position=int(prefix_len),
            alignment=1,
            padding_rows=0,
            # The native domain owns the fixed C64 ABI. Only a public override
            # is EXTERNAL_EXACT; AUTO can still calibrate native routes.
            exact_chunk_size=(getattr(self, "_action_chunk_override", 0) or None),
            resolve_stage_domain=lambda length: (
                torch.ops.rpu.lingbot_denoise_resolve_action_stage_domain(
                    handle, int(length), int(prefix_len),
                    int(prefix_len) + int(length),
                )
                if denoise_unroll
                else torch.ops.rpu.causal_decoder_resolve_prefill_stage_domain(
                    handle, int(length), int(prefix_len), False,
                    int(prefix_len) + int(length), 0, 4, N_ACTION + 1,
                )
            ),
            request_id="lingbot-v1:action_expert:action",
            plan_result_sink=lambda result: plan_box.__setitem__(
                "result", result
            ),
            graph_mode=GRAPH_COMPOSITE_CHILD,
            queue_owner_id=int(handle),
            lease_owner_id=int(handle),
            plan_signature=(bool(denoise_unroll),),
            graph_cache=(self._dgc if denoise_unroll else self._egc) if self._expert_replay else None,
        )
        if execution_len != N_ACTION + 1 or resolved != _ACTION_CHUNK_SIZE:
            raise RuntimeError(
                "LingBot-VLA fixed action plan drift: "
                f"execution={execution_len}, chunk={resolved}"
            )
        result = plan_box["result"]
        if not result.selected.stage_tuple.physical_descriptor:
            raise RuntimeError(
                "LingBot-VLA action A6 winner has no native descriptor"
            )
        return result

    def _record_execution_plan(
        self, stage: str, *, logical_len: int, chunk_size: int, position: int,
        plan=None,
    ) -> None:
        if chunk_size <= 0 or chunk_size % 16:
            raise RuntimeError(
                f"LingBot-VLA {stage} returned invalid native chunk_size={chunk_size}"
            )
        record = {
            "stage": stage,
            "component": (
                _VLM_COMPONENT if stage == "prefill" else _ACTION_COMPONENT
            ),
            "logical_len": int(logical_len),
            "execution_len": int(logical_len),
            "chunk_size": int(chunk_size),
            "padding_rows": 0,
            "position": int(position),
        }
        if plan is None:
            record.update({
                "authority": "FIXED_ABI_SINGLETON",
                "selection_scope": "FIXED_C64_DENOISE_ABI",
            })
        else:
            selected = plan.selected
            if (
                selected is None
                or selected.execution_len != int(logical_len)
                or selected.stage_tuple.compute_chunk != int(chunk_size)
            ):
                raise RuntimeError(
                    f"LingBot-VLA {stage} result disagrees with its A6 winner"
                )
            record.update({
                "authority": "NATIVE_A6_STAGE_DESCRIPTOR",
                "selection_scope": plan.selection_scope,
                "physical_plan_digest": plan.physical_plan_digest,
                "plan_digest": plan.plan_digest,
                "graph_mode": plan.graph_mode,
                "physical_descriptor": tuple(
                    selected.stage_tuple.physical_descriptor
                ),
            })
        self._rpu_last_execution_plan[stage] = record

    def validate_execution_reconfigure(self, execution_config) -> None:
        """Validate one hot generation without changing native/Python state."""
        if self._closed:
            raise RuntimeError("LingBot-VLA policy is closed")
        _root, vlm, action = _resolve_execution_components(
            execution_config,
            entry_point="LingbotVlaPolicy.reconfigure",
        )
        _validate_resolved_execution(
            vlm,
            action,
            entry_point="LingbotVlaPolicy.reconfigure",
        )
        required = (
            "execution_reconfigure_begin",
            "execution_reconfigure_commit",
            "execution_reconfigure_abort",
            "execution_reconfigure_abort_attempt",
            "causal_decoder_stage_chunk_size_override",
        )
        missing = [name for name in required if not hasattr(torch.ops.rpu, name)]
        if missing:
            raise RuntimeError(
                "LingBot-VLA binary lacks native execution-reconfigure op(s): "
                + ", ".join(missing)
            )

    @staticmethod
    def _reset_graph_owner(cache) -> None:
        if cache is None:
            return
        cache.begin_warmup()
        cache.clear()
        if not cache.cache_invariant_ok():
            raise RuntimeError(
                "LingBot-VLA GraphCache invariant failed during reconfigure"
            )

    def _reset_execution_runtime_state(self) -> None:
        """Retire exactly the Graph owners whose physical plan can change."""
        seen = set()
        for cache in (self._dgc, self._egc, self._vgc):
            if cache is not None and id(cache) not in seen:
                seen.add(id(cache))
                self._reset_graph_owner(cache)
        self._mask_cache.clear()
        self._rpu_last_execution_plan = {}

    def _stage_execution_chunks(self, vlm_chunk: int, action_chunk: int) -> None:
        """Stage both shared-KV decoder controls in one native transaction."""
        with native_execution_reconfigure(
            torch.ops.rpu, journal=self._execution_reconfigure_journal,
        ) as token:
            torch.ops.rpu.causal_decoder_stage_chunk_size_override(
                self._vlm, token, int(vlm_chunk)
            )
            torch.ops.rpu.causal_decoder_stage_chunk_size_override(
                self._exp, token, int(action_chunk)
            )

    def apply_execution_reconfigure(
        self, execution_config, generation, *, rollback_config=None
    ) -> None:
        """Commit both decoder knobs, then retire the old Graph generation."""
        self._execution_reconfigure_journal = {"mutation_started": False}
        self.validate_execution_reconfigure(execution_config)
        root, vlm, action = _resolve_execution_components(
            execution_config,
            entry_point="LingbotVlaPolicy.reconfigure",
        )
        prefill_chunk, action_chunk = _validate_resolved_execution(
            vlm,
            action,
            entry_point="LingbotVlaPolicy.reconfigure",
        )
        self._execution_reconfigure_journal = {
            "mutation_started": False,
            "config": (
                self._rpu_execution
                if rollback_config is None
                else rollback_config
            ),
            "generation": self._execution_generation,
            "prefill_chunk": self._prefill_chunk_size,
            "action_chunk": self._action_chunk_override,
            "vlm_execution": self._vlm_execution,
            "action_execution": self._action_execution,
        }
        self._stage_execution_chunks(prefill_chunk, action_chunk)
        self._prefill_chunk_size = prefill_chunk
        self._action_chunk_override = action_chunk
        self._reset_execution_runtime_state()
        self._rpu_execution = root
        self._vlm_execution = vlm
        self._action_execution = action
        self._execution_generation = int(generation)
        # Retain undo state through the shared session's config publication.

    def rollback_execution_reconfigure(self, execution_config, generation) -> None:
        """Restore the prior native controls; any failure poisons the session."""
        journal = self._execution_reconfigure_journal
        if journal is not None and not journal.get("mutation_started", False):
            self._execution_reconfigure_journal = None
            return
        if journal is None:
            root, vlm, action = _resolve_execution_components(
                execution_config,
                entry_point="LingbotVlaPolicy.rollback",
            )
            prefill_chunk, action_chunk = _validate_resolved_execution(
                vlm,
                action,
                entry_point="LingbotVlaPolicy.rollback",
            )
        else:
            root = journal["config"]
            vlm = journal["vlm_execution"]
            action = journal["action_execution"]
            prefill_chunk = journal["prefill_chunk"]
            action_chunk = journal["action_chunk"]
            generation = journal["generation"]

        self._stage_execution_chunks(prefill_chunk, action_chunk)
        self._reset_execution_runtime_state()
        self._prefill_chunk_size = int(prefill_chunk)
        self._action_chunk_override = int(action_chunk)
        self._rpu_execution = root
        self._vlm_execution = vlm
        self._action_execution = action
        self._execution_generation = int(generation)
        self._execution_reconfigure_journal = None

    def _drop_device_state(self) -> None:
        """Drop all policy-owned RPU tensors after confirmed native cleanup."""
        for attr in (
            "_cache", "_vgc", "_egc", "_dgc",
            "_vlm_keep", "_exp_keep", "_fold_all",
            "_suffix_buf", "_prefix_buf", "_mask_cache",
            "_od_op_w", "_od_op_b", "_od_wc_w", "_od_mo_w", "_od_mo_b",
            "_od_time_parts", "_od_xt_pad",
            "_du_keep", "_du_x0", "_du_x_traj",
            "_pending_keep",
            "_diagnostic_vision",
        ):
            if hasattr(self, attr):
                setattr(self, attr, None)
        self._vlm = None
        self._exp = None
        self._du_handle = None
        # A diagnostic tower is instance-bound; release its closure only after
        # the whole native retirement has succeeded.
        vars(self).pop("_vit_towers", None)

    def _retire_resources(self) -> bool:
        """Clear graph ownership, then retire all native handles once."""
        if getattr(self, "_cleanup_ok", None) is False:
            raise RuntimeError("LingBot-VLA cleanup failed; restart the process")
        if getattr(self, "_closed", False):
            return True
        try:
            from rpu_backend.api import causal_lm

            with causal_lm._LIVE_LOCK:
                if causal_lm._LIVE_TERMINAL_REASON is not None:
                    raise RuntimeError("LingBot-VLA process is unsafe; restart the process")
            vision = getattr(self, "_diagnostic_vision", None)
            caches = [getattr(self, attr, None) for attr in ("_dgc", "_egc", "_vgc")]
            caches.append(getattr(vision, "_graph_cache", None))
            seen = set()
            for cache in caches:
                if cache is not None and id(cache) not in seen:
                    seen.add(id(cache))
                    cache.clear()
                    if not cache.cache_invariant_ok():
                        raise RuntimeError("LingBot-VLA GraphCache invariant failed during close")
            if vision is not None:
                vision.close(_graphs_retired=True)
                self._diagnostic_vision = None
            for attr, destroy in (
                ("_du_handle", _destroy_denoise_handle),
                ("_exp", _destroy_causal_decoder_handle),
                ("_vlm", _destroy_causal_decoder_handle),
            ):
                handle = getattr(self, attr, None)
                if handle is not None:
                    destroy(handle)
                    setattr(self, attr, None)
            self._drop_device_state()
        except BaseException as cleanup_error:
            _poison_lingbot_retirement(self, cleanup_error)
            raise
        self._closed = True
        self._cleanup_ok = True
        return True

    def _close_without_session(self) -> None:
        """Retire resources while the caller owns the execution session."""
        self._retire_resources()
        if getattr(self, "_live_slot_released", False):
            return
        from rpu_backend.api.causal_lm import _release_live_instance
        _release_live_instance(self)
        self._live_slot_released = True

    def close(self) -> None:
        """Release resources through the common stop-the-world lifecycle."""
        session = getattr(self, "_execution_session", None)
        if session is None:
            self._close_without_session()
            return
        try:
            session.shutdown(self._close_without_session)
        except BaseException as error:
            from rpu_backend.api import _execution

            if _execution._UNSAFE_PROCESS_REASON is not None:
                _poison_lingbot_retirement(self, error)
            raise
        if not getattr(self, "_closed", False):
            error = RuntimeError("LingBot-VLA closed Session still owns native resources")
            _poison_lingbot_retirement(self, error)
            raise error

    def __del__(self):
        # Reuse explicit retirement; a raw-integer finalizer cannot observe
        # graph failures or remember which handles were already destroyed.
        if (not getattr(self, "_gc_retirement_enabled", False)
                or getattr(self, "_cleanup_ok", True) is not None):
            return
        session = getattr(self, "_execution_session", None)
        try:
            with session._lock if session is not None else nullcontext():
                if session is not None and session._active:
                    _poison_lingbot_retirement(
                        self, RuntimeError("LingBot-VLA GC during an active forward"))
                    return
                from rpu_backend.api import causal_lm

                # Cyclic GC clears weakrefs before __del__. Do not retire the
                # old native resources after another owner has claimed the slot.
                with causal_lm._LIVE_LOCK:
                    live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
                    if live is not None and live is not self:
                        _poison_lingbot_retirement(
                            self, RuntimeError("LingBot-VLA GC after live-owner handoff"))
                        return
                    self.close()
        except BaseException as error:
            # A shared unsafe-process guard may reject before retire() runs.
            _poison_lingbot_retirement(self, error)

    def _build_denoise_unroll(self):
        """Build the lingbot_denoise handle: expert decoder weights (reused from exp_keep) +
        single-core-swizzled fused encoder / out_proj + the call-invariant AdaRMS fold."""
        (q_w, k_w, v_w, o_w, q_b, k_b, v_b, gate_l, up_l, down_l,
         in_norm, post_norm, cos, sin, final_norm_w) = self._exp_keep
        # Single-core (partition=1) swizzled fused encoder + out_proj (NOT the 8-core _od_* set).
        ad = self._enc_W_comp.shape[1]                                   # 75 (action_dim)
        kp = ((ad + 15) // 16) * 16                                      # 80 (K pad, %16)
        wc = _h(tp_col_swizzle_mc_weight(F.pad(self._enc_W_comp, (0, kp - ad)).half(), 1))   # [768,80]
        mo = _h(tp_col_swizzle_mc_weight(self._head_W["model.action_time_mlp_out.weight"].half(), 1))  # [768,768]
        mo_b = _h(self._head_W["model.action_time_mlp_out.bias"])        # [768]
        aow = self._head_W["model.action_out_proj.weight"]              # [75,768]
        aob = self._head_W["model.action_out_proj.bias"]               # [75]
        npad = ((aow.shape[0] + 15) // 16) * 16                          # 80 (out pad; == kp for Euler)
        op = _h(tp_col_swizzle_mc_weight(F.pad(aow, (0, 0, 0, npad - aow.shape[0])).half(), 1))   # [80,768]
        op_b = _h(F.pad(aob, (0, npad - aob.shape[0])))                 # [80]
        time_all = _h(torch.stack([tp.reshape(-1) for tp in self._enc_time_parts]))   # [num_steps,768]
        in_s, in_sh, po_s, po_sh = self._fold_all                        # 4× [num_steps,36,768] RPU fp16
        self._du_ad = ad                                                 # 75, output slice width
        self._du_x0 = torch.zeros(1, N_ACTION + 1, kp, dtype=torch.float16, device="rpu")     # stable
        self._du_x_traj = torch.empty(NUM_STEPS, N_ACTION + 1, kp, dtype=torch.float16, device="rpu")  # stable
        handle = torch.ops.rpu.lingbot_denoise_create()
        try:
            torch.ops.rpu.lingbot_denoise_set_weights(
                handle, q_w, k_w, v_w, o_w, in_norm, post_norm,
                gate_l, up_l, down_l, cos, sin, final_norm_w,
                NQ, NKV, HD, EXP_HID, EXP_INTER_PAD, RMS_EPS, True, q_b, k_b, v_b,
                wc, mo, mo_b, op, op_b, time_all, in_s, in_sh, po_s, po_sh,
                ad, kp, NUM_STEPS, N_ACTION + 1)
            torch.ops.rpu.lingbot_denoise_set_chunk_size(
                handle, _ACTION_CHUNK_SIZE)
        except BaseException as build_error:
            _cleanup_lingbot_pending_handle(
                handle, "_du_handle", (self._exp_keep, wc, mo, mo_b, op, op_b,
                                       time_all, self._fold_all), build_error)
            raise
        self._du_handle = handle
        self._du_keep = (wc, mo, mo_b, op, op_b, time_all)               # keepalive

    def _denoise_unroll_run(
        self, noise, state_emb, dt, mask_r, S, num_steps, action_plan,
    ):
        """ONE fused unroll: refresh x0 + state, then lingbot_denoise_unroll_forward runs all
        num_steps Euler steps in-graph (shared prefix KV, per-step suffix insert at [S,S+51))."""
        self._du_x0.zero_()
        self._du_x0[:, 1:, :self._du_ad].copy_(_h(noise))               # rows 1:51, cols 0:75 = noise; row 0 = 0
        state_r = _h(state_emb.reshape(-1))                             # [768] emb_stage_ row 0
        self._cache.reset_to_position(S)                               # freeze prefix [0,S); insert suffix at S
        sig = rpu_backend.graph.GraphSignature(op_id="lingbot_denoise_unroll", shapes=[N_ACTION + 1, EXP_HID],
                                         dyn_dims=[EXP_LAYERS, S, num_steps,
                                                   self._action_chunk_size,
                                                   *action_plan.graph_key_words()],
                                         dtypes=[torch.float16])
        descriptor = list(
            action_plan.selected.stage_tuple.physical_descriptor
        )
        if not descriptor:
            raise RuntimeError(
                "LingBot-VLA fused action plan has no native descriptor"
            )
        with self._dgc.capture(sig):
            torch.ops.rpu.lingbot_denoise_unroll_forward(
                self._du_handle, self._du_x0, self._cache.k_caches, self._cache.v_caches,
                state_r, mask_r, self._du_x_traj, dt, S, num_steps,
                descriptor)
        resolved_chunk_size = int(
            torch.ops.rpu.lingbot_denoise_get_resolved_chunk_size(
                self._du_handle
            )
        )
        if resolved_chunk_size != _ACTION_CHUNK_SIZE:
            raise RuntimeError(
                "LingBot-VLA fused action chunk mismatch: "
                f"expected={_ACTION_CHUNK_SIZE}, resolved={resolved_chunk_size}"
            )
        self._record_execution_plan(
            "action",
            logical_len=N_ACTION + 1,
            chunk_size=resolved_chunk_size,
            position=S,
            plan=action_plan,
        )
        return self._du_x_traj[-1:][:, 1:, :self._du_ad].float().cpu()  # [1,50,75]

    def _validate_observation(self, obs):
        """Validate the standalone token-input contract before graph mutation."""
        if not isinstance(obs, Mapping):
            raise TypeError(f"obs must be a mapping, got {type(obs).__name__}.")
        required = ("images", "lang_tokens", "state", "noise")
        missing = [key for key in required if key not in obs]
        if missing:
            raise ValueError(f"lingbot observation is missing required key(s): {missing}.")

        values = {
            key: obs.get(key)
            for key in (*required, "img_masks", "lang_masks")
        }
        for key, value in values.items():
            if value is None and key in {"img_masks", "lang_masks"}:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"obs[{key!r}] must be a torch.Tensor.")
            if value.device.type != "cpu":
                raise ValueError(f"obs[{key!r}] must be a CPU tensor.")

        images = values["images"]
        if not images.is_floating_point() or images.is_complex():
            raise TypeError("obs['images'] must have a real floating-point dtype.")
        if images.ndim != 3 or images.shape[0] < 1 or images.shape[1] < 1:
            raise ValueError(
                "obs['images'] must have shape [num_images, num_patches, patch_width]."
            )
        if not bool(torch.isfinite(images).all()):
            raise ValueError("obs['images'] must contain only finite values.")
        num_patches = int(images.shape[1])
        grid = math.isqrt(num_patches)
        if grid * grid != num_patches or grid % V_SM != 0:
            raise ValueError(
                f"lingbot images require a square patch grid divisible by {V_SM}; "
                f"got {num_patches} patches."
            )
        patch_weight = self._vis_W[VIS_P + "patch_embed.proj.weight"]
        expected_patch_width = patch_weight.numel() // V_HID
        if patch_weight.numel() % V_HID != 0 or images.shape[2] != expected_patch_width:
            raise ValueError(
                f"obs['images'] patch width must be {expected_patch_width}; "
                f"got {images.shape[2]}."
            )

        def validate_mask(name, mask, expected_numel):
            if mask is None:
                return None
            if mask.numel() != expected_numel:
                raise ValueError(
                    f"obs[{name!r}] must contain {expected_numel} entries; "
                    f"got {mask.numel()}."
                )
            if mask.is_complex() or not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError(f"obs[{name!r}] must be binary (0/1).")
            return mask.reshape(-1).bool()

        img_mask = validate_mask("img_masks", values["img_masks"], images.shape[0])
        valid_images = images.shape[0] if img_mask is None else int(img_mask.sum().item())
        if valid_images < 1:
            raise ValueError("lingbot observation must contain at least one valid image.")

        lang_tokens = values["lang_tokens"]
        if lang_tokens.dtype not in {torch.int32, torch.int64}:
            raise TypeError("obs['lang_tokens'] must have torch.int32 or torch.int64 dtype.")
        lang_flat = lang_tokens.reshape(-1)
        lang_mask = validate_mask("lang_masks", values["lang_masks"], lang_flat.numel())
        if lang_mask is not None:
            lang_flat = lang_flat[lang_mask]
        if lang_flat.numel() < 1:
            raise ValueError("lingbot observation must contain at least one valid language token.")
        vocab_size = int(self._embed_w.shape[0])
        token_min = int(lang_flat.min().item())
        token_max = int(lang_flat.max().item())
        if token_min < 0 or token_max >= vocab_size:
            raise ValueError(
                f"obs['lang_tokens'] must be in [0, {vocab_size}); "
                f"got range [{token_min}, {token_max}]."
            )

        state = values["state"]
        noise = values["noise"]
        for name, value in (("state", state), ("noise", noise)):
            if not value.is_floating_point() or value.is_complex():
                raise TypeError(f"obs[{name!r}] must have a real floating-point dtype.")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"obs[{name!r}] must contain only finite values.")
        state_dim = int(self._head_W["model.state_proj.weight"].shape[1])
        action_dim = int(self._head_W["model.action_in_proj.weight"].shape[1])
        if tuple(state.shape) != (1, state_dim):
            raise ValueError(
                f"obs['state'] must have shape [1, {state_dim}]; got {tuple(state.shape)}."
            )
        if tuple(noise.shape) != (1, N_ACTION, action_dim):
            raise ValueError(
                f"obs['noise'] must have shape [1, {N_ACTION}, {action_dim}]; "
                f"got {tuple(noise.shape)}."
            )

        prefix_len = valid_images * (num_patches // V_SMU) + lang_flat.numel()
        if prefix_len + N_ACTION + 1 > self._max_seq:
            raise ValueError(
                f"lingbot: prefix S={prefix_len} + suffix {N_ACTION + 1} exceeds "
                f"KV-cache capacity {self._max_seq}; rebuild with a larger max_seq."
            )
        return (
            images,
            values["img_masks"],
            lang_flat.reshape(1, -1),
            state.float(),
            noise.float(),
        )

    def _vit_towers(self, images, img_masks):
        """Per-image CPU-fp32 ViT towers → concatenated img_emb [1, sum_tok, 2048]."""
        n = images.shape[0]
        num_patch = images.shape[1]
        g = int(round(math.isqrt(num_patch)))
        if g * g != num_patch:
            raise ValueError(f"lingbot ViT: {num_patch} patches is not a square grid; "
                             f"pass a square patch grid or an explicit grid_thw.")
        grid_thw = [[1, g, g]]
        embs = []
        for i in range(n):
            if img_masks is not None and not bool(img_masks.reshape(-1)[i]):
                continue
            embs.append(run_vit(images[i:i + 1].float(), grid_thw, self._vis_W))  # [1, tok, 2048]
        if not embs:
            raise ValueError("lingbot: no valid image (img_masks all False).")
        return torch.cat(embs, dim=1)                                             # [1, sum_tok, 2048]

    def _vlm_fill(self, prefix_r, S, vlm_sig, prefill_plan):
        planned_stage_descriptor = (
            prefill_plan.selected.stage_tuple.physical_descriptor
        )
        if not planned_stage_descriptor:
            raise RuntimeError("LingBot-VLA prefill plan has no native descriptor")
        self._cache.reset_to_position(0)
        with self._vgc.capture(vlm_sig):
            torch.ops.rpu.causal_decoder_forward(
                self._vlm, prefix_r,
                self._cache.k_caches, self._cache.v_caches,
                None, 0, False, None, [], None, None, -1, 0, False,
                0, planned_stage_descriptor,
            )
        resolved_chunk_size = int(
            torch.ops.rpu.causal_decoder_get_resolved_chunk_size(self._vlm)
        )
        if resolved_chunk_size != self._planned_prefill_chunk_size:
            raise RuntimeError(
                "LingBot-VLA prefill dry/forward chunk mismatch: "
                f"dry={self._planned_prefill_chunk_size}, "
                f"forward={resolved_chunk_size}, logical_len={S}"
            )
        self._record_execution_plan(
            "prefill",
            logical_len=S,
            chunk_size=resolved_chunk_size,
            position=0,
            plan=prefill_plan,
        )
        self._cache.update_position(S)

    def _embed_suffix_fused(self, x_t, state_emb, k):
        """Fused per-step suffix embed (see __init__): 2 matmuls instead of 3 + a precomputed
        per-step time bias. suffix_embs equals embed_suffix_step up to fp reassociation."""
        h = F.linear(x_t, self._enc_W_comp) + self._enc_time_parts[k]          # [1,50,768]
        h = F.silu(h)
        ate = F.linear(h, self._head_W["model.action_time_mlp_out.weight"],
                       self._head_W["model.action_time_mlp_out.bias"])
        return torch.cat([state_emb[:, None], ate], dim=1)                      # [1,51,768]

    def _denoise_ondevice(self, noise, state_emb, dt, mask_r, S, sig,
                          in_s, in_sh, po_s, po_sh, num_steps, action_plan):
        """On-device denoise loop: out_proj + Euler + fused encoder all run on RPU (eager F.linear),
        so x_t never leaves the device — no per-step CPU matmuls, no D2H/H2D. fp16 throughout."""
        x_t = _h(noise)                                            # [1,50,75] RPU fp16
        self._suffix_buf[:, 0].copy_(_h(state_emb))                # state token (constant → write once)
        for k in range(num_steps):
            self._od_xt_pad[:, :, :self._od_ad].copy_(x_t)         # x_t into padded [1,50,80] (cols[75:]=0)
            h = F.linear(self._od_xt_pad, self._od_wc_w) + self._od_time_parts[k]
            ate = F.linear(F.silu(h), self._od_mo_w, self._od_mo_b)        # [1,50,768]
            self._suffix_buf[:, 1:].copy_(ate)
            torch.ops.rpu.causal_decoder_set_adarms_step_mutable(self._exp, in_s[k], in_sh[k], po_s[k], po_sh[k])
            out = self._exp_run(
                self._suffix_buf, mask_r, S, sig, action_plan
            )                                                            # [1,51,768] RPU
            v_t = F.linear(out[:, -N_ACTION:], self._od_op_w, self._od_op_b)[..., :self._od_ad]  # [1,50,75]
            x_t = x_t + dt * v_t                                          # RPU add/mul (DDR, safe)
        return x_t.to("cpu").float()

    def _exp_run(self, suffix_r, mask_r, S, sig, action_plan):
        planned_stage_descriptor = (
            action_plan.selected.stage_tuple.physical_descriptor
        )
        if not planned_stage_descriptor:
            raise RuntimeError("LingBot-VLA action plan has no native descriptor")
        self._cache.reset_to_position(S)      # freeze prefix [0,S); insert suffix at S
        with self._egc.capture(sig):
            out = torch.ops.rpu.causal_decoder_forward(
                self._exp, suffix_r,
                self._cache.k_caches, self._cache.v_caches,
                mask_r, S, False, None, [], None, None, -1, 0, False,
                0, planned_stage_descriptor,
            )
        resolved_chunk_size = int(
            torch.ops.rpu.causal_decoder_get_resolved_chunk_size(self._exp)
        )
        if resolved_chunk_size != _ACTION_CHUNK_SIZE:
            raise RuntimeError(
                "LingBot-VLA action chunk mismatch: "
                f"expected={_ACTION_CHUNK_SIZE}, resolved={resolved_chunk_size}"
            )
        self._record_execution_plan(
            "action",
            logical_len=N_ACTION + 1,
            chunk_size=resolved_chunk_size,
            position=S,
            plan=action_plan,
        )
        return out

    @execution_serialized
    @torch.no_grad()
    def get_action(self, obs, *, num_steps=NUM_STEPS):
        if self._closed:
            raise RuntimeError("LingBot-VLA policy is closed.")
        if isinstance(num_steps, bool) or not isinstance(num_steps, Integral):
            raise TypeError(f"num_steps must be an int, got {num_steps!r}.")
        num_steps = int(num_steps)
        if num_steps != NUM_STEPS:
            raise ValueError(f"lingbot denoise built for num_steps={NUM_STEPS}; got {num_steps}.")
        images, img_masks, lang_ids, state, noise = self._validate_observation(obs)
        valid_images = (
            int(images.shape[0])
            if img_masks is None
            else int(img_masks.reshape(-1).bool().sum().item())
        )
        logical_prefix_len = (
            valid_images * (int(images.shape[1]) // V_SMU)
            + int(lang_ids.numel())
        )
        # Planning is read-only and precedes graph/cache/device mutation. This
        # catches both non-monotone SDPA rejects and real SPM-budget failures.
        prefill_plan = self._plan_prefill_execution(logical_prefix_len)
        self._planned_prefill_chunk_size = int(
            prefill_plan.selected.stage_tuple.compute_chunk
        )
        action_plan = self._plan_action_execution(logical_prefix_len)
        self._rpu_last_execution_plan = {
            "vision": {
                "stage": "vision",
                "component": _VISION_COMPONENT,
                "status": "NOT_APPLICABLE",
                "authority": "CPU_FP32",
            }
        }
        # causal_decoder_forward bakes the input tensor's DDR address at graph BUILD (fixed-DMA), so a
        # cached graph can only be REPLAYED across calls when the per-forward input lives at a STABLE
        # address. The VLM prefill now feeds through self._prefix_buf (stable) ⇒ self._vgc is kept
        # persistent and the VLM graph replays on the 2nd+ call (self._vlm_replay). The expert graph is
        # likewise persistent when replay is enabled because suffix + mask addresses are stable.
        # Toggle either off to clear its old graph owner and restore fresh-cache-per-call behavior
        # (RPU_LINGBOT_{VLM,EXPERT}_REPLAY=0).
        if not self._vlm_replay:
            self._vgc.clear()
            self._vgc = rpu_backend.graph.GraphCache()
        if not self._expert_replay:
            self._egc.clear()
            self._egc = rpu_backend.graph.GraphCache()  # fresh per call ⇒ rebuild step 0 each call
        # else: keep self._egc persistent ⇒ the expert graph replays across calls (stable mask + suffix
        # buffers below). set_adarms_step_mutable doesn't invalidate on the 2nd+ step, so call 2+ finds
        # the cached graph and REPLAYs all NUM_STEPS steps (no per-call step-0 rebuild).
        # --- prefix assembly (host) ---
        img_emb = self._vit_towers(images, img_masks)                             # [1, n_img_tok, 2048]
        lang_emb = F.embedding(lang_ids, self._embed_w)                           # [1, L, 2048]
        prefix_embs = torch.cat([img_emb.float(), lang_emb], dim=1)               # [1, S, 2048]
        S = prefix_embs.shape[1]
        if S != logical_prefix_len:
            raise RuntimeError(
                "LingBot-VLA prefix accounting mismatch: "
                f"planned={logical_prefix_len}, materialized={S}"
            )
        if self._expert_replay or self._denoise_unroll:
            mask_r = self._mask_cache.get(S)
            if mask_r is None:
                mask_r = _build_suffix_mask(S)
                self._mask_cache[S] = mask_r      # stable addr for cross-call expert/unroll replay
        else:
            mask_r = _build_suffix_mask(S)
        if self._vlm_replay:
            self._prefix_buf[:S].copy_(prefix_embs.half().squeeze(0).to("rpu"))   # refresh STABLE input
            prefix_r = self._prefix_buf[:S].unsqueeze(0)                          # [1,S,2048], stable base
        else:
            prefix_r = prefix_embs.half().to("rpu").contiguous()
        vlm_sig = rpu_backend.graph.GraphSignature(
            op_id="lingbot_vlm_prefill",
            shapes=[S, VLM_HID],
            dyn_dims=[VLM_LAYERS, chunk_policy_key(self._vlm),
                      *prefill_plan.graph_key_words()],
            dtypes=[torch.float16],
        )
        # 一次 VLM forward 填充 prefix KV，expert 随后读取同一份缓存。
        # 无需为同一个 prefix 重复执行填充。
        self._vlm_fill(
            prefix_r, S, vlm_sig, prefill_plan
        )   # BUILD the VLM graph → correct prefix KV in the shared cache

        aout_w = self._head_W["model.action_out_proj.weight"]
        aout_b = self._head_W["model.action_out_proj.bias"]

        # 固定 timestep 和 AdaRMS 常量在构建时预计算。
        # expert 首步 BUILD，后续步骤通过稳定 suffix 缓冲和 mutable AdaRMS
        # keepalive 更新输入并 REPLAY；state embedding 每次请求只需计算一次。
        x_t = noise.clone()
        dt = -1.0 / num_steps
        in_s_all, in_sh_all, po_s_all, po_sh_all = self._fold_all
        state_emb = F.linear(state, self._head_W["model.state_proj.weight"],
                             self._head_W["model.state_proj.bias"])
        # S MUST be in the signature: with self._expert_replay the persistent self._egc replays this
        # graph across calls, but S sets the 2D mask width [51, S+51], the SDPA kv_seq_len, the KV write
        # offset and the prefix position baked at BUILD. Omitting it would replay an old-S graph on a
        # call whose prefix length changed (different text len / image count) → silent wrong action.
        # (Matches _denoise_unroll_run, which already keys on S.)
        denoise_sig = rpu_backend.graph.GraphSignature(
            op_id="lingbot_expert_denoise",
            shapes=[N_ACTION + 1, EXP_HID],
            dyn_dims=[EXP_LAYERS, S, chunk_policy_key(self._exp),
                      *action_plan.graph_key_words()],
            dtypes=[torch.float16],
        )
        if self._denoise_unroll:
            return self._denoise_unroll_run(
                noise, state_emb, dt, mask_r, S, num_steps, action_plan
            )
        if self._ondevice_enc:
            return self._denoise_ondevice(noise, state_emb, dt, mask_r, S, denoise_sig,
                                          in_s_all, in_sh_all, po_s_all, po_sh_all,
                                          num_steps, action_plan)
        _nt = torch.get_num_threads()
        if self._enc_1thread:
            torch.set_num_threads(1)      # tiny per-step encoders are thread-overhead-bound (ViT already ran)
        try:
            for k in range(num_steps):
                if self._fused_encoder:
                    suffix = self._embed_suffix_fused(x_t, state_emb, k)                        # [1,51,768]
                else:
                    suffix = embed_suffix_step(x_t, state_emb, self._time_embs[k], self._head_W)
                if self._expert_replay:
                    # Mutable AdaRMS: pass the [EXP_LAYERS, EXP_HID] fold slices directly (no per-layer list).
                    torch.ops.rpu.causal_decoder_set_adarms_step_mutable(
                        self._exp, in_s_all[k], in_sh_all[k], po_s_all[k], po_sh_all[k])
                    self._suffix_buf.copy_(suffix.half())                        # refresh STABLE input (direct CPU→RPU)
                    out = self._exp_run(
                        self._suffix_buf, mask_r, S, denoise_sig, action_plan
                    )  # step 0 BUILD, k>0 REPLAY
                else:
                    ss = (list(in_s_all[k]), list(in_sh_all[k]), list(po_s_all[k]), list(po_sh_all[k]))
                    # Swaps the per-layer scale/shift tensors. It calls invalidate_model_state(),
                    # which only marks the C++ weights/preload dirty — it does NOT invalidate any
                    # RpuKernelGraph, so it cannot by itself force a rebuild.
                    torch.ops.rpu.causal_decoder_set_adarms_step(self._exp, *ss)
                    suffix_r = suffix.half().to("rpu").contiguous()
                    # 此分支不跨步或跨调用 REPLAY，因此签名不含 S。
                    # op_id 中的 k 隔离每个 denoise 步，非 replay 模式每次请求新建 GraphCache。
                    # 两项约束都必须保留：suffix 与 mask 地址在 BUILD 时绑定，
                    # 只给签名添加 S 不能替代这些生命周期要求。
                    step_sig = rpu_backend.graph.GraphSignature(
                        op_id=f"lingbot_expert_step_{k}",
                        shapes=[N_ACTION + 1, EXP_HID],
                        dyn_dims=[EXP_LAYERS, chunk_policy_key(self._exp),
                                  *action_plan.graph_key_words()],
                        dtypes=[torch.float16],
                    )
                    out = self._exp_run(
                        suffix_r, mask_r, S, step_sig, action_plan
                    )
                suffix_out = out.to("cpu").float()[:, -N_ACTION:]                  # [1,50,768]
                v_t = F.linear(suffix_out, aout_w, aout_b)                        # [1,50,75]
                x_t = x_t + dt * v_t
        finally:
            if self._enc_1thread:
                torch.set_num_threads(_nt)
        return x_t                                                                # [1,50,75]


# =============================================================================
# Build
# =============================================================================
def _ckpt_reader(ckpt_path):
    """Lazy fp32 weight getter over LingBot-VLA safetensors shards."""
    files = sorted(glob.glob(os.path.join(ckpt_path, "*.safetensors")))
    if not files:
        raise ValueError(f"lingbot: no .safetensors under {ckpt_path}")
    readers = [safe_open(f, framework="pt") for f in files]
    key_to_reader = {}
    for r in readers:
        for kk in r.keys():
            if kk in key_to_reader:
                raise ValueError(
                    f"lingbot: duplicate tensor key {kk!r} across safetensors files."
                )
            key_to_reader[kk] = r

    def gw(key):
        return key_to_reader[key].get_tensor(key).float()
    return gw, key_to_reader


def build_lingbot_vla(
    ckpt,
    qwen_base=None,
    *,
    max_seq=DEFAULT_MAX_SEQ,
    rpu_execution=None,
):
    """Build the LingBot-VLA RPU runtime.

    Args:
        ckpt:      path to the lingbot-vla-4b checkpoint dir (fp32 model.safetensors).
        qwen_base: Qwen2.5-VL-3B path (config/tokenizer). Accepted for API parity; the token-input
                   get_action path does not need it (obs supplies pre-tokenized lang_tokens).
        max_seq:   rope-table + KV-cache capacity (prefix S + suffix 51 must fit).
                   The default is 256, but the separately certified VLM execution
                   envelope is a continuous logical prefix length of at most 178.
        rpu_execution: Optional exact/auto chunk configuration for prefill and
                       action. Exact prefill values are 16-aligned, no greater
                       than 176, and revalidated against each logical shape.
    """
    execution_config, vlm_execution, action_execution = (
        _resolve_execution_components(
            rpu_execution,
            entry_point="build_lingbot_vla",
        )
    )
    prefill_chunk_size, action_chunk_size = _validate_resolved_execution(
        vlm_execution,
        action_execution,
        entry_point="build_lingbot_vla",
    )
    if isinstance(max_seq, bool) or not isinstance(max_seq, Integral):
        raise TypeError(f"max_seq must be an int, got {max_seq!r}.")
    max_seq = int(max_seq)
    minimum_seq = N_ACTION + 3  # 51 suffix + >=1 image token + >=1 text token
    if max_seq < minimum_seq:
        raise ValueError(
            f"max_seq must be at least {minimum_seq} to fit the "
            f"{N_ACTION + 1}-token action suffix and a non-empty image/text "
            f"prefix; got {max_seq}."
        )
    from rpu_backend.api.causal_lm import (
        _claim_live_instance,
        _release_live_instance,
    )

    # Claim before any RPU tensor/handle exists. A CPU-only preflight failure
    # releases immediately; once materialization starts, a failed owner stays
    # claimed until GC so traceback-held RPU tensors cannot overlap a new policy.
    policy = LingbotVlaPolicy.__new__(LingbotVlaPolicy)
    policy._rpu_execution = execution_config
    _claim_live_instance(policy)
    materialization_started = False
    try:
        gw, key_to_reader = _ckpt_reader(ckpt)
        _validate_checkpoint_profile(key_to_reader)

        # Resident CPU-fp32 weights for the host paths (ViT, encoders/heads,
        # AdaRMS norms, token embed).
        vis_W = {k: gw(k) for k in key_to_reader if k.startswith(VIS_P)}
        head_keys = [
            "model.state_proj.weight", "model.state_proj.bias",
            "model.action_in_proj.weight", "model.action_in_proj.bias",
            "model.action_out_proj.weight", "model.action_out_proj.bias",
            "model.action_time_mlp_in.weight", "model.action_time_mlp_in.bias",
            "model.action_time_mlp_out.weight", "model.action_time_mlp_out.bias",
        ]
        head_W = {k: gw(k) for k in head_keys}
        norm_W = {}
        for L in range(EXP_LAYERS):
            for slot in ("input_layernorm", "post_attention_layernorm"):
                pf = f"{EXP_P}layers.{L}.{slot}"
                for suf in (
                    ".weight", ".gamma.weight", ".gamma.bias",
                    ".beta.weight", ".beta.bias",
                ):
                    norm_W[pf + suf] = gw(pf + suf)
        embed_w = gw(VLM_P + "embed_tokens.weight")
        runtime_options = _runtime_options_from_env()

        policy._rpu_swizzle_started = True
        materialization_started = True
        policy._closed = False
        policy._cleanup_ok = None
        policy._live_slot_released = False
        # Shared cross-handle KV requires coherent DDR. One VLM fill is
        # sufficient; ddr_flush makes that prefix visible to the expert.
        torch.rpu.set_ddr_flush(True)
        # Fast replay skips the host op-stream walk on REPLAY. Explicit user
        # configuration still wins over these LingBot defaults.
        for flag in (
            "RPU_WALL_OSS_FAST_REPLAY",
            "RPU_FASTREPLAY_SKIP_SYNC",
            "RPU_DEEP_FAST_REPLAY",
        ):
            os.environ.setdefault(flag, "1")

        policy._vlm, policy._vlm_keep = _build_vlm_decoder(
            gw, max_seq, prefill_chunk_size)
        policy._exp, policy._exp_keep = _build_expert(
            gw, max_seq, action_chunk_size)

        # Seed the minimum cleanup state before entering the fallible policy
        # constructor; it may allocate caches/tensors or an optional third handle.
        policy._du_handle = None
        LingbotVlaPolicy.__init__(
            policy,
            vis_W=vis_W,
            head_W=head_W,
            norm_W=norm_W,
            embed_w=embed_w,
            vlm_handle=policy._vlm,
            vlm_keep=policy._vlm_keep,
            exp_handle=policy._exp,
            exp_keep=policy._exp_keep,
            max_seq=max_seq,
            prefill_chunk_size=prefill_chunk_size,
            action_chunk_size=action_chunk_size,
            rpu_execution=execution_config,
            _runtime_options=runtime_options,
        )
        policy._rpu_swizzled = True
        return policy
    except BaseException as build_error:
        if materialization_started:
            _cleanup_lingbot_build(policy, build_error)
        else:
            _release_live_instance(policy)
        raise
