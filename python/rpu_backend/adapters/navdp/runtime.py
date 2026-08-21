"""NavDP fused runtime — INCREMENT 2 (RPU runs the 16 decoder layers; host does glue+DDPM).

Weight source: extracted `navdp_head.safetensors` (`model.navdp.*`). The 16-layer
`nn.TransformerDecoder` runs on RPU (`torch.ops.rpu.navdp_forward`); the small glue
(input_embed, out_pos_embed, final LayerNorm, action_head, time_emb, cond_pos_embed) and
the DDPM 20-step loop run host-side in PyTorch.
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
import weakref

import torch
import torch.nn.functional as F

import rpu_backend
from rpu_backend.api import RPUCache
from rpu_backend.runtime.weights import tp_col_swizzle_mc_weight, tp_row_swizzle_mc_weight
from rpu_backend.adapters.internvla_n1._policy import (
    require_controlled_evaluation,
    verify_asset_manifest,
)

NUM_LAYERS, NUM_HEADS, HEAD_DIM, HIDDEN = 16, 8, 48, 384
FF_INTER, MEMORY_LEN, PREDICT_SIZE, ACTION_DIM = 1536, 34, 32, 3
EPS, DDPM_STEPS, TP = 1e-5, 20, 8
AD16 = ((ACTION_DIM + 15) // 16) * 16   # action_dim padded to 16 for kernel alignment (unroll)
_CACHE_LEN = 64  # >= max(predict_size, memory_len), mult of 16


def _destroy_navdp_handle(handle: int) -> None:
    try:
        torch.ops.rpu.navdp_destroy(handle)
    except Exception:
        pass


def _configure_navdp_handle(configure) -> int:
    """Create/configure one native handle, releasing it on failure."""
    handle = torch.ops.rpu.navdp_create()
    configured = False
    try:
        configure(handle)
        configured = True
        return handle
    finally:
        if not configured:
            _destroy_navdp_handle(handle)


def _hr(t):
    return t.to(torch.float16).to("rpu").contiguous()


def _col(w):
    return _hr(tp_col_swizzle_mc_weight(w.to(torch.float16), TP))


def _row(w):
    return _hr(tp_row_swizzle_mc_weight(w.to(torch.float16), TP))


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, x):
        hd = self.dim // 2
        emb = math.log(10000) / (hd - 1)
        emb = torch.exp(torch.arange(hd) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class NavdpRuntime:
    def __init__(self, handle, keep, glue, gc, cache):
        self._handle = handle
        self._keep = keep
        self._glue = glue          # host-side torch modules/params (fp32)
        self._gc = gc              # rpu_backend.graph.GraphCache
        self._cache = cache        # RPUCache
        self._time_emb = SinusoidalPosEmb(HIDDEN)
        self._sched = None
        self._handle_finalizer = weakref.finalize(
            self, _destroy_navdp_handle, handle
        )

    def _decoder_rpu(self, tgt_f32, memory_f32):
        """Run the 16 decoder layers on RPU. tgt[1,S,H], memory[1,M,H] fp32 -> hidden[1,S,H] fp32."""
        tgt_r = _hr(tgt_f32)          # [1,S,H]
        mem_r = _hr(memory_f32)       # [1,M,H]
        # Shared GraphCache + stable op_id → graph replay (memory drifts per step but the MUTABLE
        # broadcast rewrites its src from the device addr each replay; see rpu_navdp_model.cpp).
        sig = rpu_backend.graph.GraphSignature(
            op_id="navdp_decoder",
            shapes=[PREDICT_SIZE, HIDDEN],
            dyn_dims=[NUM_LAYERS, MEMORY_LEN],
            dtypes=[torch.float16],
        )
        with self._gc.capture(sig):
            hidden = torch.ops.rpu.navdp_forward(
                self._handle, tgt_r, mem_r, self._cache.k_caches, self._cache.v_caches, None)
        return hidden.float().cpu()

    def predict_noise(self, last_actions, timestep, goal_embed, rgbd_embed):
        """Run NavDP noise prediction with an RPU decoder and host-side glue."""
        g = self._glue
        action_embeds = F.linear(last_actions, g["ie_w"], g["ie_b"])            # input_embed 3->384 [B,32,384]
        time_embeds = self._time_emb(timestep).unsqueeze(1).to(last_actions.dtype)  # [1,1,384]
        # cond (memory) is SHARED across the B sampled trajectories: [1,34,384].
        cond = torch.cat([time_embeds, goal_embed[0:1], rgbd_embed[0:1]], dim=1) + g["cond_pos"][:, :MEMORY_LEN, :]
        inp = action_embeds + g["out_pos"][:, :PREDICT_SIZE, :]                 # [B,32,384]
        # RPU 16 layers, batch=1 per trajectory (host loops the sampler); memory `cond` shared.
        hidden = torch.cat([self._decoder_rpu(inp[b:b+1], cond) for b in range(inp.shape[0])], 0)
        out = F.layer_norm(hidden, (HIDDEN,), g["ln_w"], g["ln_b"], EPS)         # final layernorm
        return F.linear(out, g["ah_w"], g["ah_b"])                              # action_head 384->3

    def predict_action(self, goal_embed, rgbd_embed, sample_num=4, seed=20260706, init_noise=None):
        """Full System-1 denoise: DDPM 20-step reverse loop over the RPU decoder.

        goal_embed[1,1,384], rgbd_embed[1,32,384] -> action[sample_num,32,3].
        The scheduler stays on the host; each step runs the 16 decoder layers
        on RPU for each trajectory.
        """
        if os.environ.get("RPU_NAVDP_DENOISE_UNROLL", "0").strip() in ("1", "true", "True"):
            return self._predict_action_unroll(goal_embed, rgbd_embed, sample_num, seed, init_noise)
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        sched = DDPMScheduler(num_train_timesteps=DDPM_STEPS, beta_schedule="squaredcos_cap_v2",
                              clip_sample=True, prediction_type="epsilon")
        sched.set_timesteps(DDPM_STEPS)
        step_gen = torch.Generator().manual_seed(seed)  # deterministic DDPM step noise
        if init_noise is not None:
            naction = init_noise.clone()
        else:
            naction = torch.randn(sample_num, PREDICT_SIZE, ACTION_DIM, generator=torch.Generator().manual_seed(seed))
        for k in sched.timesteps:  # goal/rgbd stay batch-1 (shared memory); predict_noise loops trajectories
            noise_pred = self.predict_noise(naction, k.unsqueeze(0), goal_embed, rgbd_embed)
            naction = sched.step(model_output=noise_pred, timestep=k, sample=naction, generator=step_gen).prev_sample
        return naction

    def _ensure_glue_installed(self):
        """Push input_embed / action_head / final-LN weights + out_pos on-device (once).
        action_dim(3) is padded to AD16(16) for kernel alignment; padded cols/rows are zero."""
        if getattr(self, "_glue_installed", False):
            return
        g = self._glue
        p = AD16 - ACTION_DIM
        ie_w = F.pad(g["ie_w"], (0, p))              # [384,3] -> [384,16]
        ah_w = F.pad(g["ah_w"], (0, 0, 0, p))        # [3,384] -> [16,384]
        ah_b = F.pad(g["ah_b"], (0, p))              # [3] -> [16]
        # [A2]/[Z2] run the acc16 SPM-linear (partition=col) which expects a col-swizzled
        # weight (K-chunk interleave). nc=1 col-swizzle is identity for ie_w (K=16, one chunk)
        # but a real reorder for ah_w (K=384, 24 chunks) — raw ah_w → wrong K order → orthogonal.
        ie_w = tp_col_swizzle_mc_weight(ie_w.to(torch.float16), num_cores=1)
        ah_w = tp_col_swizzle_mc_weight(ah_w.to(torch.float16), num_cores=1)
        torch.ops.rpu.navdp_set_denoise_glue(
            self._handle,
            _hr(ie_w), _hr(g["ie_b"]), _hr(ah_w), _hr(ah_b),
            _hr(g["ln_w"]), _hr(g["ln_b"]), _hr(g["out_pos"][0, :PREDICT_SIZE, :]))
        self._glue_installed = True

    @torch.no_grad()
    def _predict_action_unroll(self, goal_embed, rgbd_embed, sample_num, seed, init_noise):
        """Run a 20-step in-graph DDPM unroll with host-precomputed coefficients."""
        batch_unroll = os.environ.get("RPU_NAVDP_BATCH", "1").strip() in ("1", "true", "True")
        batch_n = getattr(self, "_batch_n", None)
        if batch_unroll and batch_n is not None and batch_n != sample_num:
            raise ValueError(
                "NavDP batch-unroll sample_num is fixed after first dispatch: "
                f"expected {batch_n}, got {sample_num}; set RPU_NAVDP_BATCH=0 "
                "before model creation for variable sample_num"
            )
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        self._ensure_glue_installed()
        g = self._glue
        sched = DDPMScheduler(num_train_timesteps=DDPM_STEPS, beta_schedule="squaredcos_cap_v2",
                              clip_sample=True, prediction_type="epsilon")
        sched.set_timesteps(DDPM_STEPS)
        ac = sched.alphas_cumprod
        zg = torch.Generator().manual_seed(seed)
        coeff = torch.zeros(DDPM_STEPS, 4)                                   # A,B,C,D (CPU fp32, baked at BUILD)
        sigz = torch.zeros(sample_num, DDPM_STEPS, PREDICT_SIZE, AD16)       # padded action_dim
        mems = []
        for idx, k in enumerate(sched.timesteps):
            t = int(k); pt = t - 1
            ap = ac[t].item(); app = ac[pt].item() if pt >= 0 else 1.0
            bp = 1 - ap; bpp = 1 - app; ca = ap / app; cb = 1 - ca
            coeff[idx, 0] = 1 / ap**0.5; coeff[idx, 1] = -(bp**0.5) / ap**0.5
            coeff[idx, 2] = (app**0.5 * cb) / bp; coeff[idx, 3] = (ca**0.5 * bpp) / bp
            if t > 0:
                sig = (max((bpp / bp) * cb, 1e-20)) ** 0.5
                sigz[:, idx, :, :ACTION_DIM] = sig * torch.randn(sample_num, PREDICT_SIZE, ACTION_DIM, generator=zg)
            te = self._time_emb(k.unsqueeze(0)).unsqueeze(1)                 # [1,1,384]
            mems.append(torch.cat([te, goal_embed[0:1], rgbd_embed[0:1]], 1) + g["cond_pos"][:, :MEMORY_LEN, :])
        mem_all = torch.cat(mems, 0).to(torch.float16).to("rpu").contiguous()  # [N,34,384]
        coeff = coeff.contiguous()
        init = (init_noise.clone() if init_noise is not None
                else torch.randn(sample_num, PREDICT_SIZE, ACTION_DIM, generator=torch.Generator().manual_seed(seed)))
        init = F.pad(init, (0, AD16 - ACTION_DIM))                          # [sample_num,32,16]
        # batch=B trajectory batching (default on): fold the B trajectories into one unrolled graph.
        # C++ runs M=B*PS batched for every row-wise op; only the two attentions loop per-trajectory
        # (block-diagonal) — cross-attn memory is shared. RPU_NAVDP_BATCH=0 → serial fallback.
        if batch_unroll:
            BPS = sample_num * PREDICT_SIZE
            if batch_n is None:
                self._batch_cache = RPUCache(num_layers=NUM_LAYERS, batch_size=sample_num,
                                             max_seq_len=_CACHE_LEN, num_kv_heads=NUM_HEADS, head_dim=HEAD_DIM)
                self._batch_n = sample_num
            x0 = init.reshape(BPS, AD16).to(torch.float16).to("rpu").contiguous()             # [BPS,16] traj-major
            sz = sigz.permute(1, 0, 2, 3).reshape(DDPM_STEPS, BPS, AD16).contiguous() \
                     .to(torch.float16).to("rpu").contiguous()                                # [N,BPS,16]
            out = torch.empty(BPS, AD16, dtype=torch.float16, device="rpu")
            sig = rpu_backend.graph.GraphSignature(
                op_id="navdp_denoise_loop_b",
                shapes=[BPS, HIDDEN],
                dyn_dims=[NUM_LAYERS, MEMORY_LEN, DDPM_STEPS],
                dtypes=[torch.float16],
            )
            with self._gc.capture(sig):
                r = torch.ops.rpu.navdp_denoise_loop_forward(
                    self._handle, x0, mem_all, sz, coeff,
                    self._batch_cache.k_caches, self._batch_cache.v_caches, out)
            return r.reshape(sample_num, PREDICT_SIZE, AD16)[:, :, :ACTION_DIM].float().cpu()  # [B,PS,3]

        outs = []   # serial fallback (RPU_NAVDP_BATCH=0)
        for b in range(sample_num):
            x0 = init[b].to(torch.float16).to("rpu").contiguous()
            sz = sigz[b].to(torch.float16).to("rpu").contiguous()
            out = torch.empty(PREDICT_SIZE, AD16, dtype=torch.float16, device="rpu")
            sig = rpu_backend.graph.GraphSignature(
                op_id="navdp_denoise_loop",
                shapes=[PREDICT_SIZE, HIDDEN],
                dyn_dims=[NUM_LAYERS, MEMORY_LEN, DDPM_STEPS],
                dtypes=[torch.float16],
            )
            with self._gc.capture(sig):
                r = torch.ops.rpu.navdp_denoise_loop_forward(
                    self._handle, x0, mem_all, sz, coeff, self._cache.k_caches, self._cache.v_caches, out)
            outs.append(r[:, :ACTION_DIM].float().cpu())
        return torch.stack(outs, 0)

    def destroy(self):
        if getattr(self, "_handle", None) is None:
            return
        graph_cache = getattr(self, "_gc", None)
        if graph_cache is not None:
            graph_cache.clear()
        finalizer = getattr(self, "_handle_finalizer", None)
        if finalizer is not None and getattr(finalizer, "alive", False):
            torch.ops.rpu.navdp_destroy(self._handle)
            finalizer.detach()
        elif finalizer is None:
            torch.ops.rpu.navdp_destroy(self._handle)
        self._handle = None


def build_navdp(
    safetensors_path: str | Path,
    *,
    asset_manifest: Mapping[str | Path, str] | None = None,
) -> NavdpRuntime:
    require_controlled_evaluation(asset_manifest)
    checkpoint = Path(safetensors_path).expanduser().resolve()
    verify_asset_manifest((checkpoint,), asset_manifest)
    from safetensors.torch import load_file
    sd = load_file(str(checkpoint))

    actual_layers = {
        int(key.split(".", 3)[2])
        for key in sd
        if key.startswith("decoder.layers.") and key.split(".", 3)[2].isdigit()
    }
    if actual_layers != set(range(NUM_LAYERS)):
        raise ValueError(
            "InternVLA-N1 NavDP checkpoint does not match the fixed 16-layer "
            "profile: layer indices "
            f"{sorted(actual_layers)}, expected 0..{NUM_LAYERS - 1}"
        )
    required = {
        "input_embed.weight": (HIDDEN, ACTION_DIM),
        "input_embed.bias": (HIDDEN,),
        "out_pos_embed": (1, PREDICT_SIZE, HIDDEN),
        "cond_pos_embed": (1, MEMORY_LEN, HIDDEN),
        "layernorm.weight": (HIDDEN,),
        "layernorm.bias": (HIDDEN,),
        "action_head.weight": (ACTION_DIM, HIDDEN),
        "action_head.bias": (ACTION_DIM,),
    }
    for i in range(NUM_LAYERS):
        prefix = f"decoder.layers.{i}."
        required.update({
            prefix + "self_attn.in_proj_weight": (3 * HIDDEN, HIDDEN),
            prefix + "self_attn.in_proj_bias": (3 * HIDDEN,),
            prefix + "self_attn.out_proj.weight": (HIDDEN, HIDDEN),
            prefix + "self_attn.out_proj.bias": (HIDDEN,),
            prefix + "multihead_attn.in_proj_weight": (3 * HIDDEN, HIDDEN),
            prefix + "multihead_attn.in_proj_bias": (3 * HIDDEN,),
            prefix + "multihead_attn.out_proj.weight": (HIDDEN, HIDDEN),
            prefix + "multihead_attn.out_proj.bias": (HIDDEN,),
            prefix + "linear1.weight": (FF_INTER, HIDDEN),
            prefix + "linear1.bias": (FF_INTER,),
            prefix + "linear2.weight": (HIDDEN, FF_INTER),
            prefix + "linear2.bias": (HIDDEN,),
            **{
                prefix + f"norm{n}.{slot}": (HIDDEN,)
                for n in (1, 2, 3) for slot in ("weight", "bias")
            },
        })
    errors = [
        f"{key}: {tuple(sd[key].shape) if key in sd else 'missing'}, expected {shape}"
        for key, shape in required.items()
        if key not in sd
        or tuple(sd[key].shape) != shape
        or not sd[key].is_floating_point()
    ]
    if errors:
        raise ValueError(
            "InternVLA-N1 NavDP checkpoint does not match the fixed 16-layer "
            f"profile: {'; '.join(errors[:8])}"
        )

    def L(suf, i): return sd[f"decoder.layers.{i}.{suf}"]

    lists = {k: [] for k in (
        "sq_w sk_w sv_w so_w sq_b sk_b sv_b so_b cq_w ck_w cv_w co_w cq_b ck_b cv_b co_b "
        "ff1_w ff1_b ff2_w ff2_b n1_w n1_b n2_w n2_b n3_w n3_b").split()}
    for i in range(NUM_LAYERS):
        H = HIDDEN
        # split packed MHA in_proj [3H,H] -> q/k/v; bias [3H] -> q/k/v
        siw, sib = L("self_attn.in_proj_weight", i), L("self_attn.in_proj_bias", i)
        ciw, cib = L("multihead_attn.in_proj_weight", i), L("multihead_attn.in_proj_bias", i)
        lists["sq_w"].append(_col(siw[0:H]));   lists["sk_w"].append(_col(siw[H:2*H]));   lists["sv_w"].append(_col(siw[2*H:3*H]))
        lists["sq_b"].append(_hr(sib[0:H]));    lists["sk_b"].append(_hr(sib[H:2*H]));    lists["sv_b"].append(_hr(sib[2*H:3*H]))
        lists["so_w"].append(_row(L("self_attn.out_proj.weight", i)));  lists["so_b"].append(_hr(L("self_attn.out_proj.bias", i)))
        lists["cq_w"].append(_col(ciw[0:H]));   lists["ck_w"].append(_col(ciw[H:2*H]));   lists["cv_w"].append(_col(ciw[2*H:3*H]))
        lists["cq_b"].append(_hr(cib[0:H]));    lists["ck_b"].append(_hr(cib[H:2*H]));    lists["cv_b"].append(_hr(cib[2*H:3*H]))
        lists["co_w"].append(_row(L("multihead_attn.out_proj.weight", i)));  lists["co_b"].append(_hr(L("multihead_attn.out_proj.bias", i)))
        lists["ff1_w"].append(_col(L("linear1.weight", i)));  lists["ff1_b"].append(_hr(L("linear1.bias", i)))
        lists["ff2_w"].append(_row(L("linear2.weight", i)));  lists["ff2_b"].append(_hr(L("linear2.bias", i)))
        for n in ("n1", "n2", "n3"):
            src = {"n1": "norm1", "n2": "norm2", "n3": "norm3"}[n]
            lists[f"{n}_w"].append(_hr(L(f"{src}.weight", i)));  lists[f"{n}_b"].append(_hr(L(f"{src}.bias", i)))

    # host-side glue params (fp32) — input_embed / out_pos / cond_pos / final-LN / action_head
    glue = {
        "ie_w": sd["input_embed.weight"].float(), "ie_b": sd["input_embed.bias"].float(),
        "out_pos": sd["out_pos_embed"].float(), "cond_pos": sd["cond_pos_embed"].float(),
        "ln_w": sd["layernorm.weight"].float(), "ln_b": sd["layernorm.bias"].float(),
        "ah_w": sd["action_head.weight"].float(), "ah_b": sd["action_head.bias"].float(),
    }
    gc = rpu_backend.graph.GraphCache()
    cache = RPUCache(num_layers=NUM_LAYERS, batch_size=1, max_seq_len=_CACHE_LEN,
                     num_kv_heads=NUM_HEADS, head_dim=HEAD_DIM)

    def configure(handle):
        torch.ops.rpu.navdp_set_weights(
            handle,
            lists["sq_w"], lists["sk_w"], lists["sv_w"], lists["so_w"],
            lists["sq_b"], lists["sk_b"], lists["sv_b"], lists["so_b"],
            lists["cq_w"], lists["ck_w"], lists["cv_w"], lists["co_w"],
            lists["cq_b"], lists["ck_b"], lists["cv_b"], lists["co_b"],
            lists["ff1_w"], lists["ff1_b"], lists["ff2_w"], lists["ff2_b"],
            lists["n1_w"], lists["n1_b"], lists["n2_w"], lists["n2_b"],
            lists["n3_w"], lists["n3_b"], NUM_HEADS, HEAD_DIM, HIDDEN,
            FF_INTER, MEMORY_LEN, PREDICT_SIZE, ACTION_DIM, EPS)

    handle = _configure_navdp_handle(configure)
    try:
        return NavdpRuntime(handle, lists, glue, gc, cache)
    except BaseException:
        _destroy_navdp_handle(handle)
        raise
