"""Wall-OSS-0.5 action expert (expert-1) flow-matching denoise for RPU.

The action expert is a plain 1024-dim Qwen2.5
decoder (`use_adarms=FALSE`). The stock Wall-OSS checkpoint uses causal action
attention; RTC-trained checkpoints may use a bidirectional action block. It reuses the
same decoder core as expert-0, wrapped by the fused Wall-OSS action op for the
preprocessor, Euler update, and optional RTC hard pin. Expert-1 remains hidden 1024
(q_proj 1024→2048, o_proj 2048→1024) on the existing attn_tp=2 + optional-QKV-bias path.

Flow:
  1. expert-0 prefill (`WallOssLLM`) over the text prefix → fills the shared RPUCache
     with prefix K/V (positions [0, P)).
  2. Euler denoise over the checkpoint-specific time schedule: each step embeds the noisy action
     (`action_preprocessor.step`: w1/w2/w3 + sinusoidal time), runs the expert-1 decoder over
     the action chunk at `position=P` with the checkpoint's explicit action mask (action K/V
     appended at [P, P+H), attending the frozen prefix), reads
     `action_proj_back(last_hidden[:, :1024])` = velocity, and integrates `x += dt·v`.
     `reset_to_position(P)` keeps the prefix intact across steps.

The expert-1 decoder shares expert-0's RPUCache (same nkv=2, head_dim=128,
attn_tp=2 layout):
prefix K/V are expert-0's, action K/V are expert-1's, concatenated in the same cache — the
pi05 prefix-KV pattern. The C++ flushes the per-step CPU-computed action embedding and uses
a mutable layer-0 input DMA, so a fresh `.to('rpu')` tensor each step is replay-safe.
"""
from __future__ import annotations

import json
import math
import os
import weakref

import torch
import torch.nn.functional as F

import rpu_backend
from rpu_backend.runtime import rpu_env_bool
from rpu_backend.runtime.weights import (
    tp_col_swizzle_mc_weight,
    tp_row_swizzle_mc_weight,
)
from rpu_backend.adapters.wall_oss.llm import (
    _MLP_CORES, _resolve_ckpt, _eff_attn_tp, _replicate_kv,
    _SafeTensorStore, _split_qkv, _to_rpu_half, _to_rpu_int8, _pack_int4_rpu, _build_rope_tables,
    _chan_first_scale, _scale_to_rpu, _w4_is_group_wise,
    _pack_nvfp4_rpu, _nvfp4_tensor_scale_array,
    _partial_mrope_enabled, build_wall_oss_llm,
)
from rpu_backend.runtime.rope_partial import build_chunked_mrope_cos_sin


def _sinusoidal_pos_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    """action_head.SinusoidalPosEmb.forward (dim = action_hidden_size). t: [bs] fp32 → [bs, dim]."""
    half = dim // 2
    e = math.log(10000) / (half - 1)
    e = torch.exp(torch.arange(half, dtype=torch.float32) * -e)
    e = t[:, None] * e[None, :]
    return torch.cat([e.sin(), e.cos()], dim=-1)


def _dof_mask_is_row_constant(dof_mask: torch.Tensor) -> bool:
    first_row = dof_mask[0, :1].expand_as(dof_mask[0])
    return bool(torch.equal(dof_mask[0], first_row))


def _masked_dof_velocity_inputs(
    noise: torch.Tensor,
    dof_mask: torch.Tensor,
    padding_action: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return the reference contract's masked-DOF velocity blend inputs.

    The reference keeps inactive dimensions on a constant trajectory from the
    initial noise to normalized physical zero:
    ``v = model_v * dof_mask + (padding_action - noise) * (1 - dof_mask)``.
    Returning ``None`` for an all-active mask preserves the original op stream.
    """
    if not isinstance(dof_mask, torch.Tensor):
        raise TypeError(
            f"dof_mask must be a torch.Tensor, got {type(dof_mask).__name__}"
        )
    if tuple(dof_mask.shape) != tuple(noise.shape):
        raise ValueError(
            f"dof_mask must match noise shape {tuple(noise.shape)}, got "
            f"{tuple(dof_mask.shape)}"
        )
    mask = dof_mask.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if not bool(torch.logical_or(mask == 0, mask == 1).all()):
        raise ValueError("dof_mask must contain only 0 or 1")
    if bool((mask == 1).all()):
        return None, None
    if padding_action is None:
        raise ValueError(
            "padding_action is required when dof_mask contains inactive dimensions"
        )
    if not isinstance(padding_action, torch.Tensor):
        raise TypeError(
            "padding_action must be a torch.Tensor when dof_mask contains "
            "inactive dimensions"
        )
    if tuple(padding_action.shape) != tuple(noise.shape):
        raise ValueError(
            f"padding_action must match noise shape {tuple(noise.shape)}, got "
            f"{tuple(padding_action.shape)}"
        )
    padding = padding_action.detach().to(
        dtype=torch.float32, device="cpu").contiguous()
    initial = noise.detach().to(dtype=torch.float32, device="cpu")
    if not bool(torch.isfinite(padding).all()):
        raise ValueError("padding_action must contain only finite values")
    override = (padding - initial) * (1.0 - mask)
    return mask, override.contiguous()


def _action_horizon(noise: torch.Tensor, action_dim: int) -> int:
    if not isinstance(noise, torch.Tensor):
        raise TypeError(
            f"noise must be a torch.Tensor, got {type(noise).__name__}"
        )
    if (
        noise.dim() != 3
        or noise.size(0) != 1
        or noise.size(2) != action_dim
    ):
        raise ValueError(
            "noise must have shape [1, H, action_dim] with "
            f"action_dim={action_dim}, got {tuple(noise.shape)}"
        )
    return int(noise.size(1))


def _native_action_chunk_size(horizon: int) -> int:
    return ((int(horizon) + 15) // 16) * 16


def _validate_action_positions(position_ids: torch.Tensor, horizon: int) -> None:
    if not isinstance(position_ids, torch.Tensor):
        raise TypeError(
            "position_ids must be a torch.Tensor, got "
            f"{type(position_ids).__name__}"
        )
    if tuple(position_ids.shape) != (horizon, 3):
        raise ValueError(
            f"position_ids must be [H, 3], got {tuple(position_ids.shape)}"
        )


def _ensure_action_cache_capacity(cache, prefix_len: int, horizon: int) -> None:
    required = int(prefix_len) + int(horizon)
    capacity = int(cache.max_seq_len)
    if required > capacity:
        raise RuntimeError(
            "Wall-OSS action suffix exceeds the shared KV cache: "
            f"prefix_len={prefix_len}, horizon={horizon}, required={required}, "
            f"max_seq_len={capacity}. Reduce the prefix/padding or allocate a "
            "larger cache."
        )


def _destroy_action_handle(h, fused):
    """Release the expert-1 decoder handle (wall_oss_action_step when fused, else
    causal_decoder). Called by weakref.finalize on GC."""
    try:
        if fused:
            torch.ops.rpu.wall_oss_action_step_destroy(h)
        else:
            torch.ops.rpu.causal_decoder_destroy(h)
    except Exception:
        pass


def _configure_action_handle(
    *, fused, set_weights, weight_args, chunk_size_override=0
):
    """Create/configure an action handle, destroying it on failure."""
    create = (
        torch.ops.rpu.wall_oss_action_step_create
        if fused else torch.ops.rpu.causal_decoder_create
    )
    handle = create()
    configured = False
    try:
        set_weights(handle, *weight_args)
        if not fused:
            # Certified chunk envelope. Only the unfused path is a CausalDecoderModel;
            # wall_oss_action_step is a different C++ class that does not
            # participate in the envelope regime. Values are the text expert's
            # (llm.py): same geometry family, same 2D-rect-mask layout, and the
            # same geometry and mask contract.
            from rpu_backend.adapters.wall_oss.llm import (
                _CHUNK_ENVELOPE_MAX_KV,
                _PREFILL_CHUNK_SIZE_CAP_SAFE,
            )
            torch.ops.rpu.causal_decoder_set_chunk_envelope(
                handle, _CHUNK_ENVELOPE_MAX_KV, _PREFILL_CHUNK_SIZE_CAP_SAFE)
            if chunk_size_override:
                torch.ops.rpu.causal_decoder_set_chunk_size_override(
                    handle, chunk_size_override
                )
        configured = True
        return handle
    finally:
        if not configured:
            _destroy_action_handle(handle, fused)


class WallOssAction:
    """Expert-1 action denoiser on RPU.

    Construct via :func:`build_wall_oss_action`. Call :meth:`predict` with CPU `input_ids`
    `[1, P]` (text prefix), a fixed `noise` `[1, H, action_dim]` and `dof_mask` `[1, H, action_dim]`;
    returns a dict with the final `action` (CPU fp32). Per-step `xs`/`vs` trajectories
    are populated only with ``debug=True``; the production fused path leaves them as
    ``None`` to avoid intermediate device-to-host readbacks.
    """

    def __init__(self, *, llm, handle, graph_cache, hidden_size, num_layers,
                 action_dim, action_hidden, w1, w2, w3, proj_back,
                 head_dim, rope_theta, mrope_section,
                 fused=False, fused_extra=None, partial_mrope=None,
                 w2_rpu=None, w3_rpu=None, owns_llm=False,
                 execution_chunk_size="auto",
                 causal_action_attention_mask=True, flow_time_scale=1.0,
                 rtc_max_delay=0, rtc_rope_gap=10):
        partial_mrope = (
            _partial_mrope_enabled()
            if partial_mrope is None else bool(partial_mrope)
        )
        self.llm = llm                 # expert-0 WallOssLLM (owns shared RPUCache)
        self._owns_llm = bool(owns_llm)
        # partial_mrope: host-bake Qwen2.5-VL chunked cos/sin for the action chunk.
        self._head_dim = head_dim
        self._rope_theta = rope_theta
        self._mrope_section = mrope_section
        self._partial_mrope = partial_mrope
        self._handle = handle          # expert-1 decoder handle (causal_decoder OR wall_oss)
        self._graph_cache = graph_cache
        self.hidden_size = hidden_size  # 1024
        self.num_layers = num_layers
        self.action_dim = action_dim    # 26
        self.action_hidden = action_hidden  # 1024
        self._execution_chunk_size = execution_chunk_size
        self._causal_action_attention_mask = bool(
            causal_action_attention_mask)
        self._flow_time_scale = float(flow_time_scale)
        self._rtc_max_delay = int(rtc_max_delay)
        self._rtc_rope_gap = int(rtc_rope_gap)
        self._masked_dof_velocity = self._rtc_max_delay > 0
        # Memoized x-independent denoise bias inputs (frame-invariant for a fixed
        # dof_mask). RPU_WALL_OSS_HOST_CACHE=0 disables this cache.
        self._bias_cache = None
        self._rtc_bias_cache = None
        self._host_cache = rpu_env_bool("RPU_WALL_OSS_HOST_CACHE", default=True)
        # Memoized denoise-seam glue (prefill-graph → denoise-graph), both f(real_len)
        # for a fixed camera: action chunked cos/sin (keyed on next_pos) and the
        # action attention mask (keyed on (P,H,real_P)). Same RPU_WALL_OSS_HOST_CACHE gate.
        self._action_rope_cache = {}
        self._action_pos_cache = {}
        self._action_mask_cache = {}
        # HOST_CACHE=0 still keeps the immediately previous dynamic tensors alive while
        # allocating the next set. This prevents allocator address reuse from fooling
        # CausalDecoderModel's source-pointer upload memo into serving stale contents.
        self._last_action_rope_ref = None
        self._last_action_pos_ref = None
        self._last_action_mask_ref = None
        # Fused WallOssActionStepModel path (on-device preprocessor + proj_back). When set,
        # denoise_with_mask routes to the fused op; the CPU _action_embed/proj_back below
        # are unused. fused_extra carries the CPU fp32 pieces for the per-step B_step.
        self._fused = fused
        if fused:
            self._w1_dof = fused_extra["w1_dof"]
            self._w2_a = fused_extra["w2_a"]
            self._w2_t = fused_extra["w2_t"]
            self._action_dim_pad = fused_extra["action_dim_pad"]
            self._k_comb_pad = fused_extra["k_comb_pad"]
        # action_preprocessor weights. w1 (in=2*action_dim=52) and proj_back (out=26)
        # stay CPU fp32 — their dims aren't 16-aligned, so rpu_linear's partition search
        # rejects them (CPU fallback). w2 (2048->1024) / w3 (1024->1024) run on RPU fp16
        # (col-swizzled, 8-core), and the result stays on RPU for the decoder.
        self._w1, self._w2, self._w3, self._proj_back = w1, w2, w3, proj_back
        if not fused:  # 8-core swizzled w2/w3 only feed the CPU-host _action_embed path
            self._w2_rpu = (
                _to_rpu_half(tp_col_swizzle_mc_weight(w2.half(), _MLP_CORES))
                if w2_rpu is None else w2_rpu
            )  # [1024,2048]
            self._w3_rpu = (
                _to_rpu_half(tp_col_swizzle_mc_weight(w3.half(), _MLP_CORES))
                if w3_rpu is None else w3_rpu
            )  # [1024,1024]
        self._closed = False
        self._handle_finalizer = weakref.finalize(
            self, _destroy_action_handle, handle, fused
        )

    def close(self) -> None:
        """Release action resources and an internally owned LLM, once."""
        if self._closed:
            return
        self._closed = True
        graph_cache = getattr(self, "_graph_cache", None)
        if graph_cache is not None:
            try:
                graph_cache.clear()
            except Exception:
                pass
        finalizer = getattr(self, "_handle_finalizer", None)
        if finalizer is not None and getattr(finalizer, "alive", False):
            finalizer()
        if getattr(self, "_owns_llm", False):
            close_llm = getattr(self.llm, "close", None)
            if close_llm is not None:
                close_llm()

    def begin_graph_warmup(self):
        """Allow the action GraphCache to BUILD declared execution profiles."""
        return self._graph_cache.begin_warmup()

    def _velocity_blend_inputs(self, noise, dof_mask, padding_action):
        if not self._masked_dof_velocity:
            return None, None
        return _masked_dof_velocity_inputs(noise, dof_mask, padding_action)

    def freeze_graph_cache(self):
        """Enter lookup-only READY mode after all action profiles are BUILT."""
        return self._graph_cache.freeze()

    def _denoise_graph_signature(self, op_id: str, H: int, P: int,
                                 num_steps: int, *mode_dims: int):
        """Denoise graph identity: execution geometry only, never real prefix length.

        One key therefore serves every REAL prefix in an execution band. What makes that
        safe is NOT this key but the per-real-prefix mask pointer (see
        `_action_mask_cached`) driving the C++ (ptr, seq_q, seq_k) re-prepare. Gated on
        device by the mutable-mask replay contract.
        """
        return rpu_backend.graph.GraphSignature(
            op_id=op_id,
            shapes=[H, self.hidden_size],
            dyn_dims=[
                self.num_layers,
                P,
                _native_action_chunk_size(H),
                num_steps,
                *mode_dims,
            ],
            dtypes=[torch.float16],
        )

    def validate_execution_horizon(self, horizon: int) -> None:
        requested = getattr(self, "_execution_chunk_size", "auto")
        native_chunk = _native_action_chunk_size(horizon)
        if requested != "auto" and int(requested) != native_chunk:
            raise ValueError(
                "Wall-OSS action chunk_size must equal ceil16(runtime action "
                f"horizon): requested={requested}, horizon={horizon}, "
                f"required={native_chunk}"
            )

    def _bake_partial_mrope_tables(self, position_ids: torch.Tensor):
        """Host-bake Qwen2.5-VL chunked cos/sin [H, hd/2] → RPU fp16."""
        c, s = build_chunked_mrope_cos_sin(
            position_ids.to(torch.int64), head_dim=self._head_dim,
            rope_theta=self._rope_theta, mrope_section=self._mrope_section)
        return c.to("rpu").contiguous(), s.to("rpu").contiguous()

    def _partial_mrope_tables(self, position_ids: torch.Tensor, cache_key=None):
        """Memoized Qwen2.5-VL chunked cos/sin for the action positions.

        Action positions are ``arange(next_pos, next_pos + H)`` on all three
        axes, so the table is a function of the real prefix length. Stable RPU
        tensors are cached by ``next_pos`` and validated with ``torch.equal``.
        """
        key = (
            cache_key
            if self._host_cache and cache_key is not None
            else int(position_ids.reshape(-1)[0].item())
            if self._host_cache else None
        )
        c = self._action_rope_cache.get(key) if self._host_cache else None
        if c is not None and torch.equal(c[0], position_ids):
            return c[1], c[2]
        cos_il, sin_il = self._bake_partial_mrope_tables(position_ids)
        if self._host_cache:
            self._action_rope_cache[key] = (position_ids.clone(), cos_il, sin_il)
        else:
            self._last_action_rope_ref = (cos_il, sin_il)
        return cos_il, sin_il

    def _action_position_ids_rpu(self, position_ids: torch.Tensor,
                                 cache_key=None) -> torch.Tensor:
        """Return stable RPU int32 action positions, keyed by semantic next_pos."""
        key = (
            cache_key
            if self._host_cache and cache_key is not None
            else int(position_ids.reshape(-1)[0].item())
            if self._host_cache else None
        )
        cached = self._action_pos_cache.get(key) if self._host_cache else None
        if cached is not None and torch.equal(cached[0], position_ids):
            return cached[1]
        pos = position_ids.to(dtype=torch.int32, device="rpu").contiguous()
        if self._host_cache:
            self._action_pos_cache[key] = (position_ids.clone(), pos)
        else:
            self._last_action_pos_ref = pos
        return pos

    def _action_embed(self, t: float, x: torch.Tensor, dof_mask: torch.Tensor) -> torch.Tensor:
        """`ActionProcessor.step` (use_adarms=False). x/dof_mask:[1,H,action_dim] CPU fp32
        → [1,H,1024] **RPU fp16** (w2/w3 on RPU; ready to feed the decoder directly)."""
        H = x.size(1)
        na = torch.cat([x, dof_mask], dim=-1)                      # CPU [1,H,2*action_dim]
        ae = na @ self._w1.T                                       # CPU [1,H,1024]
        te = _sinusoidal_pos_emb(torch.tensor([t]), self.action_hidden)  # CPU [1,1024]
        te = te.unsqueeze(1).repeat(1, H, 1)                       # CPU [1,H,1024]
        ce_in = _to_rpu_half(torch.cat([ae, te], dim=-1))         # RPU fp16 [1,H,2048]
        ce = F.linear(ce_in, self._w2_rpu)                        # rpu_linear (col) [1,H,1024]
        return F.linear(F.silu(ce), self._w3_rpu)                # RPU fp16 [1,H,1024]

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, noise: torch.Tensor,
                dof_mask: torch.Tensor, num_steps: int = 10) -> dict:
        # predict() is the non-fused LTM (no-mask) path: it embeds on the host via _w2_rpu/_w3_rpu
        # (only built when not fused) and dispatches causal_decoder_forward on self._handle. In
        # fused mode self._handle is a wall_oss_action_step handle and _w2_rpu/_w3_rpu are unset,
        # so this path cannot run — fail clearly instead of an AttributeError / wrong-handle error.
        if self._fused:
            raise NotImplementedError(
                "WallOssAction.predict() (LTM no-mask path) is unsupported for fused models; "
                "use denoise_with_mask() (with WallOssVLA.predict for the full E2E flow).")
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(
                "input_ids must be a torch.Tensor, got "
                f"{type(input_ids).__name__}"
            )
        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise ValueError(
                f"input_ids must have shape [1, P], got {tuple(input_ids.shape)}"
            )
        H = _action_horizon(noise, self.action_dim)
        self.validate_execution_horizon(H)

        # Prefill and action share one cache. Reject the obvious logical
        # overflow before entering the LLM, then make its authoritative
        # prefill plan reserve H rows so optional padding cannot consume them.
        P = int(input_ids.size(1))
        _ensure_action_cache_capacity(self.llm.cache, P, H)
        self.llm.forward(input_ids, reserve_rows=H)
        cache = self.llm.cache
        if cache.position != P:
            raise RuntimeError(
                "Wall-OSS prefill cache position does not match input length: "
                f"cache={cache.position}, input={P}"
            )

        # mRoPE position_ids for the action chunk: absolute positions [P, P+H), 3 axes equal
        # (text-only context → no vision grid). Built once (stable keepalive across steps).
        pos = torch.arange(P, P + H, dtype=torch.int32)
        position_ids = pos[:, None].expand(H, 3).contiguous().to("rpu")

        sig = rpu_backend.graph.GraphSignature(
            op_id="wall_oss_action",
            shapes=[H, self.hidden_size],
            dyn_dims=[self.num_layers, P, _native_action_chunk_size(H)],
            dtypes=[torch.float16],
        )

        times = self._flow_times(num_steps)
        dt = (times[1] - times[0]).item()
        x = noise.clone()                                          # CPU fp32 Euler accumulator
        xs, vs = [x.clone()], []
        for i in range(num_steps):
            cache.reset_to_position(P)                             # keep prefix, drop prev action K/V
            emb = self._action_embed(times[i].item(), x, dof_mask)  # CPU fp32 [1,H,1024]
            emb_rpu = emb.to(dtype=torch.float16, device="rpu").contiguous()
            with self._graph_cache.capture(sig):
                raw = torch.ops.rpu.causal_decoder_forward(
                    self._handle, emb_rpu, cache.k_caches, cache.v_caches,
                    None,            # attention_mask (is_causal handles prefix+action causal)
                    P,               # position → action K/V inserted at [P, P+H)
                    True,            # is_causal
                    position_ids,    # mRoPE [H, 3]
                    [],              # deepstack dense embeds (none)
                )
            v = raw.float().cpu().reshape(1, H, self.hidden_size) @ self._proj_back.T  # [1,H,26]
            x = x + dt * v
            vs.append(v.clone()); xs.append(x.clone())

        return {
            "action": x, "xs": torch.cat(xs, 0), "vs": torch.cat(vs, 0),
            "prefix_len": P, "dt": dt, "times": times,
        }

    @staticmethod
    def _build_action_mask(P: int, H: int, real_P: int | None = None, *,
                           causal: bool = True,
                           rtc_delay: int = 0) -> torch.Tensor:
        """[H, P+H] fp16 additive mask: action row i attends cols [0, real_P) (REAL prefix)
        + the configured action block, -inf elsewhere. CPU tensor — the op's sdpa_prepare_mask
        copies it to a stable RPU DDR slot without a manual flush.
        Constant across denoise steps (P, H fixed) → baked once at graph BUILD.

        `real_P` < P only under the v16 KV-insert pad-to-16 path: the cache prefix occupies
        P (16-aligned) slots but the last (P-real_P) are masked padding (action must NOT
        attend to them). For RTC's bidirectional action block, committed queries
        ``[:rtc_delay]`` cannot attend denoising suffix keys ``[rtc_delay:]``;
        every other action-action direction is allowed."""
        real_P = P if real_P is None else int(real_P)
        # Vectorized build: every action row attends the real prefix [0, real_P).
        # The action block
        # [P, P+H) is either causal or bidirectional with the RTC prefix→suffix cut.
        m = torch.full((H, P + H), float("-inf"), dtype=torch.float16)
        m[:, :real_P] = 0.0                                            # real prefix, all rows
        if causal:
            allowed = torch.tril(torch.ones(H, H, dtype=torch.bool))
        else:
            allowed = torch.ones(H, H, dtype=torch.bool)
        if rtc_delay:
            if causal:
                raise ValueError("RTC action attention requires a bidirectional baseline")
            if not 0 < rtc_delay <= H:
                raise ValueError(
                    f"rtc_delay must be in [1, {H}], got {rtc_delay}")
            allowed[:rtc_delay, rtc_delay:] = False
        m[:, P : P + H].masked_fill_(allowed, 0.0)
        return m

    def _action_mask_cached(self, P: int, H: int, real_P: int | None = None,
                            rtc_delay: int = 0) -> torch.Tensor:
        """Memoized `_build_action_mask`, keyed on (P, H, real_P) — all f(real_len) for a
        fixed horizon → recurring prefix lengths reuse one mask. The mask is read-only
        downstream (the op copies it to a stable DDR slot), so sharing across frames is
        safe. Different real_P values retain different host pointers; alternating A→B→A
        therefore invalidates the C++ (pointer, seq_q, seq_k) upload memo while the same
        execution-profile graph safely REPLAYs. RPU_WALL_OSS_HOST_CACHE=0 disables."""
        causal = bool(getattr(self, "_causal_action_attention_mask", True))
        key = (P, H, P if real_P is None else int(real_P), causal,
               int(rtc_delay))
        if self._host_cache and key in self._action_mask_cache:
            return self._action_mask_cache[key]
        m = self._build_action_mask(
            P, H, real_P, causal=causal, rtc_delay=rtc_delay)
        if self._host_cache:
            self._action_mask_cache[key] = m
        else:
            self._last_action_mask_ref = m
        return m

    def _flow_times(self, num_steps: int) -> torch.Tensor:
        """Checkpoint-specific Euler schedule; stock Wall-OSS stays at 0..1."""
        times = torch.linspace(0, 1, num_steps + 1)
        scale = float(getattr(self, "_flow_time_scale", 1.0))
        return times if scale == 1.0 else times * scale

    def _rtc_inputs(self, position_ids: torch.Tensor,
                    prefix_actions: torch.Tensor | None,
                    rtc_delay: int, horizon: int):
        """Validate an RTC prefix and apply the one-time suffix RoPE gap."""
        delay = int(rtc_delay)
        if delay == 0:
            if prefix_actions is not None:
                raise ValueError("prefix_actions requires rtc_delay > 0")
            return position_ids, None, None
        max_delay = int(getattr(self, "_rtc_max_delay", 0))
        if max_delay <= 0:
            raise ValueError("this Wall-OSS action model was not built for RTC")
        if not 1 <= delay <= min(horizon, max_delay):
            raise ValueError(
                f"rtc_delay must be in [1, {min(horizon, max_delay)}], got {delay}")
        if not isinstance(prefix_actions, torch.Tensor):
            raise TypeError("prefix_actions must be a torch.Tensor for RTC")
        if (prefix_actions.dim() != 3 or prefix_actions.size(0) != 1
                or prefix_actions.size(2) != self.action_dim
                or prefix_actions.size(1) < delay):
            raise ValueError(
                "prefix_actions must be [1, L, action_dim] with "
                f"L >= rtc_delay and action_dim={self.action_dim}, got "
                f"{tuple(prefix_actions.shape)}")
        if prefix_actions.device.type != "cpu":
            raise ValueError("prefix_actions must be a CPU tensor")
        if not bool(torch.isfinite(prefix_actions).all()):
            raise ValueError("prefix_actions must contain only finite values")

        pos = position_ids.clone()
        gap = int(getattr(self, "_rtc_rope_gap", 10))
        if delay < horizon:
            pos[delay:] += gap
        first = int(position_ids.reshape(-1)[0].item())
        cache_key = ("rtc", first, horizon, delay, gap)
        return pos, prefix_actions[:, :delay].contiguous(), cache_key

    def _precompute_b_step(self, dof_mask: torch.Tensor, times: torch.Tensor,
                           num_steps: int) -> torch.Tensor:
        """X-independent per-step bias of the folded preprocessor:
        B_step[s] = (dof @ w1[:,AD:].T) @ w2[:,:AH].T + (te_s @ w2[:,AH:].T).
        Returns fp16 RPU [num_steps, AH] when dof is row-constant (→ C++ GEMM-bias path)
        else [num_steps, H, AH] (→ C++ same-shape add path)."""
        # base depends only on dof (const across steps). When dof is row-constant (the C++
        # GEMM-bias path), all H rows of base are identical, so compute one row
        # (M=1) instead of M=H.
        row_constant = _dof_mask_is_row_constant(dof_mask)
        dof_b = dof_mask[:, :1] if row_constant else dof_mask    # [1,1,AD] or [1,H,AD]
        ae_dof = dof_b @ self._w1_dof.T                 # [1,*,AH]
        base = ae_dof @ self._w2_a.T                    # [1,*,AH] (const across steps)
        rows = []
        for i in range(num_steps):
            te = _sinusoidal_pos_emb(torch.tensor([times[i].item()]), self.action_hidden)  # [1,AH]
            ct = te @ self._w2_t.T                       # [1,AH]
            rows.append(base + ct.unsqueeze(1))          # [1,*,AH]
        B = torch.cat(rows, 0)                           # [num_steps,*,AH]
        if row_constant:
            B = B[:, 0, :].contiguous()                  # [num_steps,AH]
        return B.to(dtype=torch.float16, device="rpu").contiguous()

    def _precompute_bstep_base_te(self, dof_mask: torch.Tensor, times: torch.Tensor,
                                  num_steps: int):
        """Return row-constant ``ae_dof`` and per-step time embeddings on RPU.

        The op computes both projections in SPM and combines them into ``B_step``.
        """
        ae_dof = (dof_mask[:, :1] @ self._w1_dof.T)[0, 0]  # [AH]  (one row; row-constant)
        te_all = torch.cat([
            _sinusoidal_pos_emb(torch.tensor([times[i].item()]), self.action_hidden)
            for i in range(num_steps)], 0)               # [num_steps, AH]
        return (ae_dof.to(dtype=torch.float16, device="rpu").contiguous(),
                te_all.to(dtype=torch.float16, device="rpu").contiguous())

    def _denoise_bias_inputs(self, dof_mask: torch.Tensor, times: torch.Tensor,
                             num_steps: int):
        """Memoized x-independent denoise bias inputs `(row_constant, b_all, te_all)`.
        These depend only on `(dof_mask, num_steps)` — the dof_mask is the embodiment's
        active-dim mask and `times = linspace(0, 1, num_steps+1)` is fixed, so they are
        constant across a rollout. Caching avoids rebuilding te_all and the
        ae_dof/base GEMM, and holds stable RPU tensors (reused across frames →
        also stabilizes the denoise op's input DDR addresses for replay). `b_all` is the
        row-constant ae_dof when row_constant else the full [steps,H,AH] B_step; `te_all`
        is the [steps,AH] table when row_constant else None — matching the two call sites.
        Validated by `torch.equal` on dof_mask so a dof/step change can't serve stale."""
        c = self._bias_cache
        if self._host_cache and c is not None and c[0] == num_steps and torch.equal(c[1], dof_mask):
            return c[2], c[3], c[4]
        row_constant = _dof_mask_is_row_constant(dof_mask)
        if row_constant:
            b_all, te_all = self._precompute_bstep_base_te(dof_mask, times, num_steps)
        else:
            b_all = self._precompute_b_step(dof_mask, times, num_steps)
            te_all = None
        if self._host_cache:
            self._bias_cache = (num_steps, dof_mask.clone(), row_constant, b_all, te_all)
        return row_constant, b_all, te_all

    def _rtc_bias_inputs(self, dof_mask: torch.Tensor, times: torch.Tensor,
                         num_steps: int, rtc_delay: int) -> torch.Tensor:
        """Full ``[N,H,hidden]`` bias with τ=1 on committed RTC rows."""
        c = getattr(self, "_rtc_bias_cache", None)
        if (self._host_cache and c is not None and c[0] == num_steps
                and c[1] == rtc_delay and torch.equal(c[2], dof_mask)):
            return c[3]

        base = (dof_mask @ self._w1_dof.T) @ self._w2_a.T
        rows = []
        H = int(dof_mask.size(1))
        for i in range(num_steps):
            tau = torch.full((H,), float(times[i].item()), dtype=torch.float32)
            tau[:rtc_delay] = 1.0
            te = _sinusoidal_pos_emb(tau, self.action_hidden).unsqueeze(0)
            rows.append(base + te @ self._w2_t.T)
        b_all = torch.cat(rows, dim=0).to(
            dtype=torch.float16, device="rpu").contiguous()
        if self._host_cache:
            self._rtc_bias_cache = (
                num_steps, rtc_delay, dof_mask.clone(), b_all)
        return b_all

    @torch.no_grad()
    def _denoise_with_mask_loop(self, prefix_len: int, noise: torch.Tensor,
                                dof_mask: torch.Tensor, position_ids: torch.Tensor,
                                num_steps: int = 10, real_prefix_len: int | None = None,
                                debug: bool = False,
                                rtc_prefix_actions: torch.Tensor | None = None,
                                rtc_delay: int = 0,
                                padding_action: torch.Tensor | None = None,
                                use_rtc: bool = False) -> dict:
        """In-graph unroll: one graph capture runs all `num_steps` Euler iterations
        (1 BUILD + 1 REPLAY). x persists in SPM across iterations; bias read at baked
        per-step offsets; final x = x_traj[-1]. Same return dict as `_denoise_with_mask_fused`."""
        H = _action_horizon(noise, self.action_dim)
        self.validate_execution_horizon(H)
        P = int(prefix_len)
        cache = self.llm.cache
        if cache.position != P:
            raise RuntimeError(
                f"unroll denoise: prefix K/V must be at position {P}, "
                f"cache at {cache.position}"
            )
        if not isinstance(use_rtc, bool):
            raise TypeError(f"use_rtc must be bool, got {use_rtc!r}")
        rtc_request = bool(use_rtc or rtc_delay)
        if rtc_request and int(getattr(self, "_rtc_max_delay", 0)) <= 0:
            raise ValueError("this Wall-OSS action model was not built for RTC")
        _validate_action_positions(position_ids, H)
        position_ids, rtc_prefix, rtc_cache_key = self._rtc_inputs(
            position_ids, rtc_prefix_actions, rtc_delay, H)
        pos = self._action_position_ids_rpu(position_ids, rtc_cache_key)
        if tuple(pos.shape) != (H, 3):
            raise RuntimeError(
                f"RPU position_ids must be [H, 3], got {tuple(pos.shape)}"
            )
        cos_il, sin_il = self._partial_mrope_tables(
            position_ids, rtc_cache_key)
        attn_mask = self._action_mask_cached(
            P, H, real_prefix_len, rtc_delay=rtc_delay)
        times = self._flow_times(num_steps)
        dt = (times[1] - times[0]).item()
        # Bias inputs mirror the step op: row-constant -> ae_dof[hidden] + te_all[steps,hidden]
        # (on-device base+ct); else the full host-precomputed [steps,H,hidden] B_step.
        # Memoized on (dof_mask, num_steps), which is constant across a rollout.
        if rtc_delay:
            row_constant = False
            b_all = self._rtc_bias_inputs(
                dof_mask, times, num_steps, rtc_delay)
            te_all = None
        else:
            row_constant, b_all, te_all = self._denoise_bias_inputs(
                dof_mask, times, num_steps)
        kp = self._k_comb_pad
        if kp != self._action_dim_pad:
            raise RuntimeError(
                f"in-graph Euler needs k_comb_pad({kp})=="
                f"action_dim_pad({self._action_dim_pad})"
            )
        x0_cpu = noise
        if rtc_delay:
            x0_cpu = noise.clone()
            x0_cpu[:, :rtc_delay] = rtc_prefix
        velocity_mask, velocity_override = self._velocity_blend_inputs(
            x0_cpu, dof_mask, padding_action)
        velocity_mask_rpu = velocity_override_rpu = None
        if velocity_mask is not None:
            velocity_mask_rpu = F.pad(
                velocity_mask, (0, kp - self.action_dim)).to(
                    dtype=torch.float16, device="rpu").contiguous()
            velocity_override_rpu = F.pad(
                velocity_override, (0, kp - self.action_dim)).to(
                    dtype=torch.float16, device="rpu").contiguous()
        # x0 lives on the RPU as [1,H,kp] fp16; the op keeps x in SPM across all steps.
        x0 = F.pad(x0_cpu, (0, kp - self.action_dim)).to(
            dtype=torch.float16, device="rpu").contiguous()                  # [1,H,kp]
        x_traj = torch.empty((num_steps, H, kp), dtype=torch.float16, device="rpu")
        v_traj = torch.empty((num_steps, H, self._action_dim_pad),
                             dtype=torch.float16, device="rpu")
        # num_steps + mode (have_te==b_step_mode==row_constant) discriminate
        # baked execution. real_prefix_len is dynamic mask CONTENT, not graph identity:
        # CausalDecoderModel::dynamic_config refreshes its stable mask slot before REPLAY.
        if rtc_request:
            sig = self._denoise_graph_signature(
                "wall_oss_action_rtc_denoise_loop_v2", H, P, num_steps,
                int(row_constant), rtc_delay, self._rtc_rope_gap,
                int(velocity_mask is not None),
            )
        elif velocity_mask is not None:
            sig = self._denoise_graph_signature(
                "wall_oss_action_denoise_loop_v2", H, P, num_steps,
                int(row_constant), 1,
            )
        else:
            sig = self._denoise_graph_signature(
                "wall_oss_action_denoise_loop", H, P, num_steps,
                int(row_constant),
            )
        with self._graph_cache.capture(sig):
            torch.ops.rpu.wall_oss_action_denoise_loop_forward(
                self._handle, x0, cache.k_caches, cache.v_caches,
                b_all, te_all, attn_mask, pos, x_traj, v_traj, dt, P,
                num_steps, cos_il, sin_il, x0 if rtc_delay else None,
                rtc_delay, velocity_mask_rpu, velocity_override_rpu)
        # Read trajectory ONLY after the capture exits (mutable SPM->DDR DMAs; op syncs).
        # Production (debug=False) reads back ONLY the final step (the action). The full
        # [num_steps,H,*] xs/vs trajectories are diagnostics, so production skips
        # their full readback, clone, and concatenation.
        x = x_traj[-1:][:, :, :self.action_dim].float().cpu()               # [1,H,action_dim]
        if debug:
            xs = torch.cat([
                x0_cpu.clone(),
                x_traj[:, :, :self.action_dim].float().cpu(),
            ], 0)
            vs = v_traj[:, :, :self.action_dim].float().cpu()
        else:
            xs = vs = None
        return {
            "x_norm": x, "xs": xs, "vs": vs,
            "prefix_len": P, "dt": dt, "times": times,
        }

    @torch.no_grad()
    def _denoise_with_mask_fused(self, prefix_len: int, noise: torch.Tensor,
                                 dof_mask: torch.Tensor, position_ids: torch.Tensor,
                                 num_steps: int = 10, real_prefix_len: int | None = None,
                                 debug: bool = False,
                                 rtc_prefix_actions: torch.Tensor | None = None,
                                 rtc_delay: int = 0,
                                 padding_action: torch.Tensor | None = None,
                                 use_rtc: bool = False) -> dict:
        """Fused denoise: on-device preprocessor + decoder + proj_back per step (1 BUILD +
        N-1 REPLAY). x never leaves the device between the preprocessor and proj_back."""
        fp32_tail = rpu_env_bool("RPU_WALL_OSS_ACTION_FP32_TAIL")
        if not isinstance(use_rtc, bool):
            raise TypeError(f"use_rtc must be bool, got {use_rtc!r}")
        rtc_request = bool(use_rtc or rtc_delay)
        if rtc_request:
            if fp32_tail:
                raise RuntimeError(
                    "Wall-OSS RTC does not support RPU_WALL_OSS_ACTION_FP32_TAIL")
            if not rpu_env_bool("RPU_WALL_OSS_DENOISE_UNROLL", default=True):
                raise RuntimeError(
                    "Wall-OSS RTC requires the fused denoise unroll")
            return self._denoise_with_mask_loop(
                prefix_len, noise, dof_mask, position_ids,
                num_steps=num_steps, real_prefix_len=real_prefix_len,
                debug=debug, rtc_prefix_actions=rtc_prefix_actions,
                rtc_delay=rtc_delay, padding_action=padding_action,
                use_rtc=True)
        if rtc_prefix_actions is not None:
            raise ValueError("rtc_prefix_actions requires rtc_delay > 0")
        # The in-graph unroll (1 BUILD + 1 REPLAY) is the default fused-denoise path;
        # set RPU_WALL_OSS_DENOISE_UNROLL=0 to fall back to the per-step loop (1 BUILD + 9 REPLAY).
        # The FP32-tail diagnostic must run stepwise because x_t is accumulated on the host.
        if (not fp32_tail and
                rpu_env_bool("RPU_WALL_OSS_DENOISE_UNROLL", default=True)):
            return self._denoise_with_mask_loop(
                prefix_len, noise, dof_mask, position_ids, num_steps=num_steps,
                real_prefix_len=real_prefix_len, debug=debug,
                padding_action=padding_action)
        H = _action_horizon(noise, self.action_dim)
        P = int(prefix_len)
        cache = self.llm.cache
        if cache.position != P:
            raise RuntimeError(
                f"fused denoise: prefix K/V must be at position {P}, "
                f"cache at {cache.position}"
            )
        _validate_action_positions(position_ids, H)
        pos = self._action_position_ids_rpu(position_ids)
        if tuple(pos.shape) != (H, 3):
            raise RuntimeError(
                f"RPU position_ids must be [H, 3], got {tuple(pos.shape)}"
            )
        cos_il, sin_il = self._partial_mrope_tables(position_ids)
        attn_mask = self._action_mask_cached(P, H, real_prefix_len)  # CPU fp16 [H, P+H], memoized
        times = self._flow_times(num_steps)
        dt = (times[1] - times[0]).item()
        # Row-constant masks pass ae_dof and per-step time embeddings separately;
        # the op forms B_step in SPM. Other masks use a precomputed full B_step.
        row_constant, _bias, te_all = self._denoise_bias_inputs(dof_mask, times, num_steps)
        velocity_mask, velocity_override = self._velocity_blend_inputs(
            noise, dof_mask, padding_action)
        if velocity_mask is not None and not fp32_tail:
            raise RuntimeError(
                "masked-DOF velocity override requires "
                "RPU_WALL_OSS_DENOISE_UNROLL=1"
            )
        ae_rpu = _bias if row_constant else None
        b_all = None if row_constant else _bias
        # row_constant selects the baked b_step/te_step mode. real_prefix_len
        # changes only the dynamically refreshed mask contents.
        sig = self._denoise_graph_signature(
            "wall_oss_action_step", H, P, num_steps, int(row_constant),
        )
        kp = self._k_comb_pad
        if kp != self._action_dim_pad:
            raise RuntimeError(
                f"in-graph Euler needs k_comb_pad({kp})=="
                f"action_dim_pad({self._action_dim_pad})"
            )
        # x stays on RPU as [1,H,kp] fp16; the post-hook performs the Euler
        # update in SPM using x_a/x_b as ping-pong buffers.
        # Cols [action_dim:kp] stay 0 (noise zero-padded; v_t cols [action_dim:] are 0).
        x_fp32 = (
            noise.detach().to(dtype=torch.float32, device="cpu").clone()
            if fp32_tail else None
        )
        x_a = F.pad(x_fp32 if fp32_tail else noise,
                    (0, kp - self.action_dim)).to(
            dtype=torch.float16, device="rpu").contiguous()              # [1,H,kp]  op input
        x_b = torch.empty_like(x_a)                                      # [1,H,kp]  op output
        v_t_buf = torch.empty((1, H, self._action_dim_pad),
                              dtype=torch.float16, device="rpu")
        # Trajectory collected as device clones (op syncs -> race-free); .cpu() once at end.
        xs_rpu, vs_rpu = [], []
        for i in range(num_steps):
            cache.reset_to_position(P)                    # keep prefix, drop prev action K/V
            if row_constant:
                b_step, te_step = ae_rpu, te_all.select(0, i).contiguous()     # ae_dof [AH], te [AH]
            else:
                b_step, te_step = b_all.select(0, i).contiguous(), None        # [H,AH], host B_step
            with self._graph_cache.capture(sig):
                raw = torch.ops.rpu.wall_oss_action_step_forward(
                    self._handle, x_a, cache.k_caches, cache.v_caches, b_step, te_step,
                    attn_mask, pos, v_t_buf, x_b, dt, P, cos_il, sin_il)  # x_b = x_a + dt·v
            if fp32_tail:
                # Match the GPU action seam: final hidden + original proj_back weight +
                # Euler state are FP32. The fused FP16 proj_back/Euler still execute in
                # the post-hook for graph identity, but their v_t_buf/x_b results are
                # deliberately ignored.
                if x_fp32 is None:
                    raise RuntimeError(
                        "FP32 action tail lost its host Euler accumulator"
                    )
                v_fp32 = (
                    raw.float().cpu().reshape(1, H, self.hidden_size)
                    @ self._proj_back.T
                )
                if velocity_mask is not None:
                    v_fp32 = velocity_override + v_fp32 * velocity_mask
                x_fp32 = x_fp32 + dt * v_fp32
                x_a = F.pad(x_fp32, (0, kp - self.action_dim)).to(
                    dtype=torch.float16, device="rpu").contiguous()
                if debug:
                    vs_rpu.append(v_fp32.clone())
                    xs_rpu.append(x_fp32.clone())
            else:
                x_a, x_b = x_b, x_a                       # x_a now holds the updated x
                if debug:
                    vs_rpu.append(v_t_buf[:, :, :self.action_dim].clone())
                    xs_rpu.append(x_a[:, :, :self.action_dim].clone())
        if fp32_tail:
            x = x_fp32
            if debug:
                xs = torch.cat([noise.clone()] + xs_rpu, 0)
                vs = torch.cat(vs_rpu, 0)
            else:
                xs = vs = None
        else:
            x = x_a[:, :, :self.action_dim].float().cpu()                # [1,H,action_dim]
            if debug:
                xs = torch.cat(
                    [noise.clone()] + [t.float().cpu() for t in xs_rpu], 0)
                vs = torch.cat([t.float().cpu() for t in vs_rpu], 0)
            else:
                xs = vs = None
        return {
            "x_norm": x, "xs": xs, "vs": vs,
            "prefix_len": P, "dt": dt, "times": times,
        }

    @torch.no_grad()
    def denoise_with_mask(self, prefix_len: int, noise: torch.Tensor,
                          dof_mask: torch.Tensor, position_ids: torch.Tensor,
                          num_steps: int = 10, real_prefix_len: int | None = None,
                          debug: bool = False,
                          rtc_prefix_actions: torch.Tensor | None = None,
                          rtc_delay: int = 0,
                          padding_action: torch.Tensor | None = None,
                          use_rtc: bool = False) -> dict:
        """10-step Euler denoise over a prefix of arbitrary length.

        Unlike :meth:`predict`, this does NOT prefill — the caller (WallOssVLA) has
        already filled the shared cache prefix K/V at [0, prefix_len) via
        ``WallOssLLM.forward_embeds``. Each step runs the expert-1 decoder over the
        action chunk with an explicit 2D mask (is_causal=False), so the prefix length
        need not be %16-aligned (the LTM path requires that).

        Args:
            prefix_len: P, the shared prefix (vision+text) length already in the cache.
            noise: [1, H, action_dim] fixed Euler init.
            dof_mask: [1, H, action_dim].
            padding_action: normalized physical-zero action [1, H, action_dim].
                Required when any dof_mask entry is zero.
            position_ids: [H, 3] int32 mRoPE positions for the action chunk (continuing
                after the prefix).
        """
        H = _action_horizon(noise, self.action_dim)
        self.validate_execution_horizon(H)
        P = int(prefix_len)
        _ensure_action_cache_capacity(self.llm.cache, P, H)
        if self._fused:
            return self._denoise_with_mask_fused(
                prefix_len, noise, dof_mask, position_ids, num_steps=num_steps,
                real_prefix_len=real_prefix_len, debug=debug,
                rtc_prefix_actions=rtc_prefix_actions, rtc_delay=rtc_delay,
                padding_action=padding_action, use_rtc=use_rtc)
        if use_rtc or rtc_delay or rtc_prefix_actions is not None:
            raise RuntimeError("Wall-OSS RTC requires the fused action denoiser")
        cache = self.llm.cache
        if cache.position != P:
            raise RuntimeError(
                f"denoise_with_mask: prefix K/V must be at position {P}, "
                f"cache at {cache.position} (call llm.forward_embeds first)"
            )

        _validate_action_positions(position_ids, H)
        pos = self._action_position_ids_rpu(position_ids)
        if tuple(pos.shape) != (H, 3):
            raise RuntimeError(
                f"RPU position_ids must be [H, 3], got {tuple(pos.shape)}"
            )
        attn_mask = self._action_mask_cached(P, H, real_prefix_len)  # CPU fp16 [H, P+H], memoized

        sig = self._denoise_graph_signature(
            "wall_oss_action_masked", H, P, num_steps,  # distinct from predict's LTM path
        )
        times = self._flow_times(num_steps)
        dt = (times[1] - times[0]).item()
        velocity_mask, velocity_override = self._velocity_blend_inputs(
            noise, dof_mask, padding_action)
        x = noise.clone()
        xs, vs = [x.clone()], []
        for i in range(num_steps):
            cache.reset_to_position(P)                        # keep prefix, drop prev action K/V
            emb = self._action_embed(times[i].item(), x, dof_mask)            # CPU fp32 [1,H,1024]
            emb_rpu = emb.to(dtype=torch.float16, device="rpu").contiguous()
            with self._graph_cache.capture(sig):
                raw = torch.ops.rpu.causal_decoder_forward(
                    self._handle, emb_rpu, cache.k_caches, cache.v_caches,
                    attn_mask,       # explicit 2D additive mask (CPU fp16)
                    P,               # action K/V inserted at [P, P+H)
                    False,           # is_causal=False → MASK_2D path
                    pos,             # mRoPE [H, 3]
                    [],
                )
            v = raw.float().cpu().reshape(1, H, self.hidden_size) @ self._proj_back.T  # [1,H,26]
            if velocity_mask is not None:
                v = velocity_override + v * velocity_mask
            x = x + dt * v
            vs.append(v.clone()); xs.append(x.clone())

        return {
            "x_norm": x, "xs": torch.cat(xs, 0), "vs": torch.cat(vs, 0),
            "prefix_len": P, "dt": dt, "times": times,
        }


def _load_expert1_decoder(st, cfg, num_layers, max_seq_len, *, w8a16: bool = False,
                          w4a16: bool = False, nvfp4a16: bool = False,
                          fused_action: dict | None = None,
                          action_chunk_size: int = 0):
    """Load expert-1 (action) decoder weights and install the RPU decoder handle.

    Same split/swizzle@attn_tp=2 + QKV-bias path as `build_wall_oss_llm`, but at the
    action-expert dims (hidden 1024, intermediate 2048). q_proj 1024→2048, o_proj 2048→1024.

    If ``fused_action`` is given (dict with single-core-swizzled w_comb/w3/proj_back +
    action_dim/k_comb_pad/chunk_size), routes to the fused WallOssActionStepModel op
    (decoder + on-device preprocessor + proj_back) instead of the plain causal_decoder.
    """
    NQ = cfg["num_attention_heads"]
    NKV = cfg["num_key_value_heads"]
    HD = cfg.get("head_dim") or (cfg["hidden_size"] // NQ)
    H = cfg["experts"][1]["hidden_size"]            # 1024
    INTER = cfg["experts"][1]["intermediate_size"]  # 2048
    EPS = cfg["rms_norm_eps"]
    THETA = cfg["rope_theta"]
    MROPE = list(cfg["rope_scaling"]["mrope_section"])
    ATTN_TP = _eff_attn_tp(NKV)     # 2 (default) or 8 via RPU_WALL_OSS_ATTN_TP8
    REP = ATTN_TP // NKV            # KV-head replication factor (1 or 4)
    NKV_EFF = NKV * REP             # effective KV heads the C++ decoder sees (== ATTN_TP)
    # group-wise (pgrp) vs per-channel int4 — from the checkpoint's quant config.
    w4_group_wise = w4a16 and _w4_is_group_wise(str(st.root))

    def gf(key):
        return st.get_tensor(key).float()

    def gr(key):
        return st.get_tensor(key).contiguous()

    q_w_list, k_w_list, v_w_list, o_w_list = [], [], [], []
    q_b_list, k_b_list, v_b_list = [], [], []
    gate_list, up_list, down_list = [], [], []
    q_s_list, k_s_list, v_s_list, o_s_list = [], [], [], []
    gate_s_list, up_s_list, down_s_list = [], [], []
    q_ts_list, k_ts_list, v_ts_list, o_ts_list = [], [], [], []
    gate_ts_list, up_ts_list, down_ts_list = [], [], []
    in_norm_list, post_norm_list = [], []

    for L in range(num_layers):
        p = f"model.layers.{L}"
        qkv_w = gr(f"{p}.self_attn.qkv_proj_experts.1.weight")  # [2560, 1024]
        qkv_b = gf(f"{p}.self_attn.qkv_proj_experts.1.bias")    # [2560]
        qw, kw, vw = _split_qkv(qkv_w, NQ, NKV, HD)
        qb, kb, vb = _split_qkv(qkv_b, NQ, NKV, HD)
        # ATTN_TP8: replicate each real KV head REP× so the col-swizzle sees
        # NKV_EFF==ATTN_TP heads (1 whole head/core). No-op (REP==1) by default.
        kw, vw = _replicate_kv(kw, NKV, HD, REP), _replicate_kv(vw, NKV, HD, REP)
        kb, vb = _replicate_kv(kb, NKV, HD, REP), _replicate_kv(vb, NKV, HD, REP)
        ow = gr(f"{p}.self_attn.o_proj_experts.1.weight")       # [1024, nq*hd]
        if w8a16 or w4a16:
            qkv_s = _chan_first_scale(gr(f"{p}.self_attn.qkv_proj_experts.1.weight_scale"))
            qs, ks, vs = _split_qkv(qkv_s, NQ, NKV, HD)
            ks, vs = _replicate_kv(ks, NKV, HD, REP), _replicate_kv(vs, NKV, HD, REP)
            os = _chan_first_scale(gr(f"{p}.self_attn.o_proj_experts.1.weight_scale"))
            if w8a16:
                q_w_list.append(_to_rpu_int8(tp_col_swizzle_mc_weight(qw, ATTN_TP, dwidth=1)))
                k_w_list.append(_to_rpu_int8(tp_col_swizzle_mc_weight(kw, ATTN_TP, dwidth=1)))
                v_w_list.append(_to_rpu_int8(tp_col_swizzle_mc_weight(vw, ATTN_TP, dwidth=1)))
                o_w_list.append(_to_rpu_int8(tp_row_swizzle_mc_weight(ow, ATTN_TP, dwidth=1)))
            else:  # w4a16: int4 values (int8 container) -> packed uint8 [N,K/2]
                q_w_list.append(_pack_int4_rpu(qw, 1, ATTN_TP, w4_group_wise))
                k_w_list.append(_pack_int4_rpu(kw, 1, ATTN_TP, w4_group_wise))
                v_w_list.append(_pack_int4_rpu(vw, 1, ATTN_TP, w4_group_wise))
                o_w_list.append(_pack_int4_rpu(ow, 0, ATTN_TP, w4_group_wise))
            q_s_list.append(_scale_to_rpu(
                qs, in_features=H, partition=1, num_cores=ATTN_TP))
            k_s_list.append(_scale_to_rpu(
                ks, in_features=H, partition=1, num_cores=ATTN_TP))
            v_s_list.append(_scale_to_rpu(
                vs, in_features=H, partition=1, num_cores=ATTN_TP))
            o_s_list.append(_scale_to_rpu(
                os, in_features=NQ * HD, partition=0, num_cores=ATTN_TP))
        elif nvfp4a16:
            q_w, q_s, q_ts = _pack_nvfp4_rpu(qw, 1, ATTN_TP)
            k_w, k_s, k_ts = _pack_nvfp4_rpu(kw, 1, ATTN_TP)
            v_w, v_s, v_ts = _pack_nvfp4_rpu(vw, 1, ATTN_TP)
            o_w, o_s, o_ts = _pack_nvfp4_rpu(ow, 0, ATTN_TP)
            q_w_list.append(q_w); q_s_list.append(q_s); q_ts_list.append(q_ts)
            k_w_list.append(k_w); k_s_list.append(k_s); k_ts_list.append(k_ts)
            v_w_list.append(v_w); v_s_list.append(v_s); v_ts_list.append(v_ts)
            o_w_list.append(o_w); o_s_list.append(o_s); o_ts_list.append(o_ts)
        else:
            q_w_list.append(_to_rpu_half(tp_col_swizzle_mc_weight(qw.half(), ATTN_TP)))
            k_w_list.append(_to_rpu_half(tp_col_swizzle_mc_weight(kw.half(), ATTN_TP)))
            v_w_list.append(_to_rpu_half(tp_col_swizzle_mc_weight(vw.half(), ATTN_TP)))
            o_w_list.append(_to_rpu_half(tp_row_swizzle_mc_weight(ow.half(), ATTN_TP)))
        q_b_list.append(_to_rpu_half(qb))
        k_b_list.append(_to_rpu_half(kb))
        v_b_list.append(_to_rpu_half(vb))

        gu = gr(f"{p}.moe.experts.1.gate_up_proj.weight")       # [2*INTER, 1024]
        gate_w = gu[:INTER].contiguous()
        up_w = gu[INTER:2 * INTER].contiguous()
        down_w = gr(f"{p}.moe.experts.1.down_proj.weight")      # [1024, INTER]
        if w8a16 or w4a16:
            gu_s = _chan_first_scale(gr(f"{p}.moe.experts.1.gate_up_proj.weight_scale"))
            gate_s = gu_s[:INTER].contiguous()
            up_s = gu_s[INTER:2 * INTER].contiguous()
            down_s = _chan_first_scale(gr(f"{p}.moe.experts.1.down_proj.weight_scale"))
            if w8a16:
                gate_list.append(_to_rpu_int8(tp_col_swizzle_mc_weight(gate_w, _MLP_CORES, dwidth=1)))
                up_list.append(_to_rpu_int8(tp_col_swizzle_mc_weight(up_w, _MLP_CORES, dwidth=1)))
                down_list.append(_to_rpu_int8(tp_row_swizzle_mc_weight(down_w, _MLP_CORES, dwidth=1)))
            else:  # w4a16: expert-1 INTER=2048, so down (K=2048) IS int4 (2048 % 512 == 0)
                gate_list.append(_pack_int4_rpu(gate_w, 1, _MLP_CORES, w4_group_wise))
                up_list.append(_pack_int4_rpu(up_w, 1, _MLP_CORES, w4_group_wise))
                down_list.append(_pack_int4_rpu(down_w, 0, _MLP_CORES, w4_group_wise))
            gate_s_list.append(_scale_to_rpu(
                gate_s, in_features=H, partition=1, num_cores=_MLP_CORES))
            up_s_list.append(_scale_to_rpu(
                up_s, in_features=H, partition=1, num_cores=_MLP_CORES))
            down_s_list.append(_scale_to_rpu(
                down_s, in_features=INTER, partition=0,
                num_cores=_MLP_CORES))
        elif nvfp4a16:
            gate_wq, gate_s, gate_ts = _pack_nvfp4_rpu(gate_w, 1, _MLP_CORES)
            up_wq, up_s, up_ts = _pack_nvfp4_rpu(up_w, 1, _MLP_CORES)
            down_wq, down_s, down_ts = _pack_nvfp4_rpu(down_w, 0, _MLP_CORES)
            gate_list.append(gate_wq); gate_s_list.append(gate_s); gate_ts_list.append(gate_ts)
            up_list.append(up_wq); up_s_list.append(up_s); up_ts_list.append(up_ts)
            down_list.append(down_wq); down_s_list.append(down_s); down_ts_list.append(down_ts)
        else:
            gate_list.append(_to_rpu_half(tp_col_swizzle_mc_weight(gate_w.half(), _MLP_CORES)))
            up_list.append(_to_rpu_half(tp_col_swizzle_mc_weight(up_w.half(), _MLP_CORES)))
            down_list.append(_to_rpu_half(tp_row_swizzle_mc_weight(down_w.half(), _MLP_CORES)))

        in_norm_list.append(_to_rpu_half(gf(f"{p}.input_layernorms.1.weight")))
        post_norm_list.append(_to_rpu_half(gf(f"{p}.post_attention_layernorms.1.weight")))

    final_norm_w = _to_rpu_half(gf("model.norms.1.weight"))
    cos, sin = _build_rope_tables(HD, THETA, max_seq_len)

    if fused_action is not None:
        # Fused WallOssActionStepModel: same decoder weights routed through the wall_oss
        # op (which forwards to CausalDecoderModel::set_weights) + the on-device action
        # preprocessor / proj_back weights. Scale lists are empty for fp16, populated for
        # Quantized decoder scales are passed through the inherited decoder setter.
        weight_args = [
            q_w_list, k_w_list, v_w_list, o_w_list,
            in_norm_list, post_norm_list,
            gate_list, up_list, down_list,
            cos, sin, final_norm_w,
            NQ, NKV_EFF, HD, H, INTER, EPS, True,      # use_silu (SwiGLU)
            MROPE,
            q_b_list, k_b_list, v_b_list,
            q_s_list, k_s_list, v_s_list, o_s_list,
            gate_s_list, up_s_list, down_s_list,
            fused_action["w_comb"], fused_action["w3"], fused_action["proj_back"],
            fused_action["w2_t"], fused_action["w2_a"],
            fused_action["action_dim"], fused_action["k_comb_pad"],
            fused_action.get("chunk_size", 0),  # 0 = derive horizon at forward
        ]
        if nvfp4a16:
            weight_args.extend([
                _nvfp4_tensor_scale_array(q_ts_list),
                _nvfp4_tensor_scale_array(k_ts_list),
                _nvfp4_tensor_scale_array(v_ts_list),
                _nvfp4_tensor_scale_array(o_ts_list),
                _nvfp4_tensor_scale_array(gate_ts_list),
                _nvfp4_tensor_scale_array(up_ts_list),
                _nvfp4_tensor_scale_array(down_ts_list),
            ])
        handle = _configure_action_handle(
            fused=True,
            set_weights=torch.ops.rpu.wall_oss_action_step_set_weights,
            weight_args=weight_args,
        )
        return handle, H, INTER

    set_weights = (
        torch.ops.rpu.causal_decoder_set_weights_nvfp4
        if nvfp4a16 else (
            torch.ops.rpu.causal_decoder_set_weights_w8a16
            if (w8a16 or w4a16) else torch.ops.rpu.causal_decoder_set_weights
        )
    )
    args = [
        q_w_list, k_w_list, v_w_list, o_w_list,
        [], [],                       # q_norm / k_norm — Qwen2.5 has none
        in_norm_list, post_norm_list,
        gate_list, up_list, down_list,
        cos, sin, final_norm_w,
        NQ, NKV_EFF, HD, H, INTER,
        EPS, True,                    # use_silu (SwiGLU)
        MROPE, [],                    # mrope_section, deepstack_lang_layers
    ]
    if nvfp4a16:
        args.extend([
            q_s_list, k_s_list, v_s_list, o_s_list,
            gate_s_list, up_s_list, down_s_list,
            _nvfp4_tensor_scale_array(q_ts_list),
            _nvfp4_tensor_scale_array(k_ts_list),
            _nvfp4_tensor_scale_array(v_ts_list),
            _nvfp4_tensor_scale_array(o_ts_list),
            _nvfp4_tensor_scale_array(gate_ts_list),
            _nvfp4_tensor_scale_array(up_ts_list),
            _nvfp4_tensor_scale_array(down_ts_list),
            q_b_list, k_b_list, v_b_list,
        ])
    elif w8a16 or w4a16:
        args.extend([
            q_s_list, k_s_list, v_s_list, o_s_list,
            gate_s_list, up_s_list, down_s_list,
            q_b_list, k_b_list, v_b_list,
        ])
    else:
        args.extend([q_b_list, k_b_list, v_b_list])
    handle = _configure_action_handle(
        fused=False, set_weights=set_weights, weight_args=args,
        chunk_size_override=action_chunk_size,
    )
    return handle, H, INTER


def _build_fused_action_weights(g, AH, AD, chunk_size):
    """Fold the action preprocessor by linearity into single-core-swizzled
    on-device weights + the CPU fp32 pieces the per-step B_step precompute needs.

    Returns (fused_action dict for the C-API, fused_extra dict for WallOssAction).
    """
    w1 = g("action_preprocessor.w1.weight")        # [AH, 2*AD]
    w2 = g("action_preprocessor.w2.weight")        # [AH, 2*AH]
    w3 = g("action_preprocessor.w3.weight")        # [AH, AH]
    proj_back = g("action_preprocessor.action_proj_back.weight")   # [AD, AH]
    k_comb_pad = ((AD + 15) // 16) * 16            # 26 -> 32 (single-core col-GEMM K)
    action_dim_pad = ((AD + 15) // 16) * 16        # proj_back out padded to 16
    # W_comb = w2[:,:AH] @ w1[:,:AD]  -> [AH, AD]; pad K (in) AD->k_comb_pad (zero cols).
    w_comb = w2[:, :AH] @ w1[:, :AD]               # [AH, AD]
    w_comb_p = F.pad(w_comb, (0, k_comb_pad - AD))                 # [AH, k_comb_pad]
    proj_back_p = F.pad(proj_back, (0, 0, 0, action_dim_pad - AD))  # [action_dim_pad, AH]
    fused_action = {
        "w_comb":    _to_rpu_half(tp_col_swizzle_mc_weight(w_comb_p.half(), 1)),
        "w3":        _to_rpu_half(tp_col_swizzle_mc_weight(w3.half(), 1)),
        "proj_back": _to_rpu_half(tp_col_swizzle_mc_weight(proj_back_p.half(), 1)),
        # w2_t = w2[:,AH:] (time proj) and w2_a = w2[:,:AH] (action proj); single-core
        # col-swizzled. Enable the on-device base = ae_dof·w2_a and ct = te·w2_t (the whole
        # x-independent B_step on-device; host keeps only ae_dof = dof·w1_dof + sinusoidal).
        "w2_t":      _to_rpu_half(tp_col_swizzle_mc_weight(w2[:, AH:].contiguous().half(), 1)),
        "w2_a":      _to_rpu_half(tp_col_swizzle_mc_weight(w2[:, :AH].contiguous().half(), 1)),
        "action_dim": AD, "k_comb_pad": k_comb_pad, "chunk_size": chunk_size,
    }
    fused_extra = {  # CPU fp32 pieces for the per-step x-independent B_step
        "w1_dof": w1[:, AD:].contiguous(),         # [AH, AD]
        "w2_a":   w2[:, :AH].contiguous(),         # [AH, AH]
        "w2_t":   w2[:, AH:].contiguous(),         # [AH, AH]
        "action_dim_pad": action_dim_pad, "k_comb_pad": k_comb_pad,
    }
    return fused_action, fused_extra


def build_wall_oss_action(
    ckpt_dir: str | None = None,
    *,
    llm: "WallOssLLM | None" = None,
    max_seq_len: int = 2048,
    w8a16: bool = False,
    w4a16: bool = False,
    nvfp4a16: bool = False,
    fp16_ckpt_dir: str | None = None,
    fused: bool = False,
    execution_config=None,
    rtc_max_delay: int = 0,
    rtc_rope_gap: int = 10,
    flow_time_scale: float = 1.0,
) -> WallOssAction:
    """Build the expert-0 prefill and expert-1 action denoiser sharing one RPUCache.

    Args:
        ckpt_dir: Wall-OSS-0.5 checkpoint directory. ``None`` (default) resolves at
            call time under the current ``$RPU_MODEL_CACHE`` (W8A16 vs fp16 path).
        llm: optional pre-built expert-0 ``WallOssLLM``; if None, builds one here. Its
             RPUCache is shared with the action expert (same nkv/head_dim/attn_tp layout).
        max_seq_len: RoPE table + shared KV cache capacity (must hold prefix + horizon).
        w8a16: load expert-0/expert-1 decoder weights from the Wall-OSS W8A16
            checkpoint. CPU action_preprocessor/proj_back stay fp32.
        fp16_ckpt_dir: source checkpoint for CPU fp32 action_preprocessor/proj_back
            when `w8a16=True`.
        fused: route the denoise step through the fused WallOssActionStepModel op
            (on-device preprocessor + proj_back; folded by linearity). Gated upstream by
            RPU_WALL_OSS_FUSED_DENOISE.
    """
    precision_modes = {
        "w8a16": w8a16,
        "w4a16": w4a16,
        "nvfp4a16": nvfp4a16,
    }
    invalid_modes = {
        name: value
        for name, value in precision_modes.items()
        if not isinstance(value, bool)
    }
    if invalid_modes:
        raise ValueError(
            f"Wall-OSS precision flags must be bool, got {invalid_modes}"
        )
    selected_modes = [name for name, enabled in precision_modes.items() if enabled]
    if len(selected_modes) > 1:
        raise ValueError(
            "pick one of w8a16 / w4a16 / nvfp4a16, got "
            f"{selected_modes}"
        )
    from rpu_backend.api._execution import normalize_rpu_execution
    action_execution = normalize_rpu_execution(
        {
            "action": ({} if execution_config is None else
                       execution_config.get("action", {}))
        },
        entry_point="build_wall_oss_action",
        supported={"action": ("chunk_size",)},
    ).get("action", {})
    requested_action_chunk = action_execution.get("chunk_size", "auto")
    action_chunk_size = (
        0 if requested_action_chunk == "auto" else int(requested_action_chunk)
    )
    if w4a16 and ckpt_dir is None:
        from rpu_backend.model_registry import model_path
        ckpt_dir = str(model_path("wall-oss-0.5-w4a16"))
    else:
        ckpt_dir = _resolve_ckpt(ckpt_dir, w8a16)
    fp16_ckpt_dir = _resolve_ckpt(fp16_ckpt_dir, False)
    cfg = json.load(open(f"{ckpt_dir}/config.json"))
    causal_action_attention_mask = bool(
        cfg.get("causal_action_attention_mask", True))
    if rtc_max_delay:
        if not fused:
            raise ValueError("Wall-OSS RTC requires fused=True")
        if causal_action_attention_mask:
            raise ValueError(
                "Wall-OSS RTC requires causal_action_attention_mask=false")
    owns_llm = llm is None
    if owns_llm:
        llm = build_wall_oss_llm(ckpt_dir, expert=0, max_seq_len=max_seq_len,
                                 w8a16=w8a16, w4a16=w4a16,
                                 nvfp4a16=nvfp4a16,
                                 fp16_ckpt_dir=fp16_ckpt_dir,
                                 execution_config=execution_config)
    try:
        num_layers = cfg["num_hidden_layers"]
        if llm.num_layers != num_layers:
            raise ValueError(
                f"shared-cache requires expert-0 num_layers={llm.num_layers} "
                f"== {num_layers}"
            )

        st = _SafeTensorStore(ckpt_dir)
        cpu_st = (
            _SafeTensorStore(fp16_ckpt_dir)
            if (w8a16 or w4a16 or nvfp4a16) else st
        )

        def g(key):
            return cpu_st.get_tensor(key).float()

        AH = cfg["action_hidden_size"]                                     # 1024
        proj_back = g("action_preprocessor.action_proj_back.weight")
        AD = proj_back.size(0)                                              # 26
        w1 = g("action_preprocessor.w1.weight")
        w2 = g("action_preprocessor.w2.weight")
        w3 = g("action_preprocessor.w3.weight")
        fused_action, fused_extra = (
            _build_fused_action_weights(g, AH, AD, action_chunk_size)
            if fused else (None, None)
        )
        partial_mrope = _partial_mrope_enabled()
        graph_cache = rpu_backend.graph.GraphCache()
        w2_rpu = w3_rpu = None
        if not fused:
            w2_rpu = _to_rpu_half(
                tp_col_swizzle_mc_weight(w2.half(), _MLP_CORES)
            )
            w3_rpu = _to_rpu_half(
                tp_col_swizzle_mc_weight(w3.half(), _MLP_CORES)
            )

        handle = None
        transferred = False
        try:
            handle, H, _INTER = _load_expert1_decoder(
                st, cfg, num_layers, max_seq_len,
                w8a16=w8a16, w4a16=w4a16, nvfp4a16=nvfp4a16,
                fused_action=fused_action,
                action_chunk_size=action_chunk_size,
            )
            model = WallOssAction(
                llm=llm, handle=handle, graph_cache=graph_cache,
                hidden_size=H, num_layers=num_layers,
                action_dim=AD,                                             # 26
                action_hidden=AH,                                          # 1024
                w1=w1, w2=w2, w3=w3, proj_back=proj_back,
                head_dim=(
                    cfg.get("head_dim")
                    or (cfg["hidden_size"] // cfg["num_attention_heads"])
                ),
                rope_theta=cfg["rope_theta"],
                mrope_section=list(cfg["rope_scaling"]["mrope_section"]),
                fused=fused, fused_extra=fused_extra,
                partial_mrope=partial_mrope,
                w2_rpu=w2_rpu, w3_rpu=w3_rpu,
                owns_llm=owns_llm,
                execution_chunk_size=requested_action_chunk,
                causal_action_attention_mask=causal_action_attention_mask,
                flow_time_scale=flow_time_scale,
                rtc_max_delay=rtc_max_delay,
                rtc_rope_gap=rtc_rope_gap,
            )
            transferred = True
            return model
        finally:
            if handle is not None and not transferred:
                _destroy_action_handle(handle, fused)
    except BaseException:
        if owns_llm:
            close_llm = getattr(llm, "close", None)
            if close_llm is not None:
                try:
                    close_llm()
                except Exception:
                    pass
        raise
