"""LingBot-VLA V2 (robbyant/lingbot-vla-v2-6b) — RPU runtime (image+text → action, flow-matching).

Self-contained standalone runtime; it does not import the LingBot-VLA V1 adapter.

  image  → Qwen3-VL ViT on RPU (install_qwen3_vl_vision_for_rpu)   → img_emb  [n_vis, 2560]
  text   → embed_tokens[lang_tokens]                               → lang_emb [1, L, 2560]
  prefix = image-placeholder tokens + lang tokens, visual embeds masked_scatter'd in
         → RPU Qwen3-VL text decoder (M-RoPE; vision features from [5,11,17]
           injected into text layers [0,1,2]; CAUSAL — YAML vlm_causal: true)
           fills the SHARED RPUCache [0, S)
  state+noise → host encoders + per-(step,layer,norm) AdaRMS fold
         → RPU sparse-MoE AdaRMS expert (768-d, 32 experts, top_k 4, joint shared-prefix-KV,
           explicit 2D mask), 10-step host Euler (default) → action [1, 50, 55]

Runtime boundaries
------------------
* Vision uses the Qwen3-VL RPU tower.
* The action expert does not own a separate RoPE base; q/k use the VLM rotary
  embedding and its configured theta (5e6 for Qwen3-VL-4B).
* The host Euler loop is the default. An optional, default-off unroll uses the
  existing sparse-MoE handle, configurable launch batch limits, and FP16 state
  between ACC32 linears.

Controlled evaluation is limited to the exact-Z2 W8A16 three-image
REAL215..225/P225 envelope; FP16, W4 and other profiles remain outside it.
Every path stays fail-closed
unless exact ``RPU_LINGBOT2_ALLOW_UNVALIDATED=1`` is set; the env name is retained
for compatibility.
"""
from __future__ import annotations

import math
import os
import threading
import weakref
from numbers import Integral

import torch
import torch.nn.functional as F

import rpu_backend
from rpu_backend.api import RPUCache
from rpu_backend.adapters.qwen3_vl import (
    install_qwen3_vl_text_for_rpu,
    install_qwen3_vl_vision_for_rpu,
    register_qwen3vl_vision_fused_merger,
    scatter_visual_embeds_to_dense,
)
from rpu_backend.adapters.qwen3_vl.vision import (
    _prepare_qwen3vl_vision_input_for_spm_pipeline,
)
from rpu_backend.runtime.decoder import _run_causal_decoder_forward
from rpu_backend.runtime.rope_partial import build_interleaved_mrope_cos_sin
from rpu_backend.runtime.weights import convert_linear_weights_inplace

from rpu_backend.adapters.lingbot_vla_v2 import convert as _cv
from rpu_backend.adapters.lingbot_vla_v2.convert import (
    ACTION_DIM, EXP_HID, EXP_LAYERS, HD, NKV, N_ACTION, NUM_STEPS, STATE_DIM, VLM_P,
    build_expert_moe, ckpt_reader, load_align_weights, load_host_weights,
)
from rpu_backend.adapters.lingbot_vla_v2.prefix import (
    AlignConfig, SpecialTokenIds, apply_suffix_prefix_blocking_, build_prefix_layout,
    compute_align_tokens, pad_prefix_layout_for_execution,
    plan_prefix_execution, suffix_position_start,
)

# VLM (Qwen3-VL-4B-Instruct) — checkpoint-verified.
VLM_LAYERS = 36
VLM_HID = 2560
ATTN_TP = 8                     # C++ attn_tp() == min(NUM_CORES, num_kv_heads); 8 for BOTH halves,
                                # which is exactly why one shared RPUCache serves both.
DEFAULT_MAX_SEQ = 1024          # rope-table + KV-cache capacity (prefix S + suffix 51 must fit)
DEFAULT_CHUNK_SIZE = 0          # 0 = C++ auto-select.
PREFIX_PADDING_MULTIPLE = 16
PREFIX_POLICY_VERSION = 1
MULTIVIEW_SPM_Z2_EXECUTION_LEN = 225


def _destroy_native_handle(handle, op_name: str) -> bool:
    """Best-effort native-handle cleanup for explicit close/finalizers."""
    if handle is None:
        return True
    try:
        getattr(torch.ops.rpu, op_name)(handle)
    except Exception:
        return False
    return True


def _invoke_handle_finalizer(finalizer, handle, op_name: str) -> bool:
    """Destroy a child handle and disarm its best-effort finalizer."""
    if finalizer is not None:
        if not getattr(finalizer, "alive", False):
            return True
        if handle is None:
            return False
        if not _destroy_native_handle(handle, op_name):
            return False
        try:
            finalizer.detach()
        except Exception:
            return False
        return True
    return _destroy_native_handle(handle, op_name)


def _destroy_policy_handles(
    expert_handle,
    text_handle,
    text_finalizer,
    vision_handle,
    vision_finalizer,
    multiview_spm_z2_state=None,
) -> bool:
    """Retire LingBot2 native handles in reverse construction order."""
    if not _close_multiview_spm_z2_state(multiview_spm_z2_state):
        return False
    if not _destroy_native_handle(expert_handle, "lingbot_v2_moe_destroy"):
        return False
    if not _invoke_handle_finalizer(
        text_finalizer, text_handle, "causal_decoder_destroy"
    ):
        return False
    return _invoke_handle_finalizer(
        vision_finalizer, vision_handle, "qwen3vl_vision_destroy"
    )


def _clear_graph_cache(owner, attr: str) -> bool:
    if owner is None:
        return True
    cache = getattr(owner, attr, None)
    if cache is None:
        return True
    try:
        cache.clear()
    except Exception:
        return False
    return True


def _resolve_qwen3vl_base():
    """Locate the Qwen3-VL-4B-Instruct dir used ONLY for `Qwen3VLConfig.from_pretrained`.

    No weights are read from it (they come from the V2 checkpoint), so only its
    config.json (~1.5 KB) is needed. First candidate that actually holds a
    config.json wins; a missing candidate is skipped, so this never returns a path
    that does not exist. Resolution order:

      1. ``RPU_LINGBOT2_QWEN3VL_BASE`` env — deployment override (validated at use).
      2. a copy bundled next to rpu_backend.so by a packaged runtime
         (``rpu_backend/qwen3_vl_base/``).
      3. the copy shipped INSIDE this adapter package
         (``adapters/lingbot_vla_v2/_qwen3vl_base/``) — present in EVERY install, so
         callers need no extra configuration or second model directory.
      4. ``model_registry.model_path("qwen3-vl-4b")`` — source-checkout fallback.

    Returns ``None`` when nothing resolves; the caller raises a clear, actionable
    error instead of letting HF misread a missing path as a repo id.
    """
    env = os.environ.get("RPU_LINGBOT2_QWEN3VL_BASE", "").strip()
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.dirname(os.path.dirname(here))  # .../rpu_backend
    try:
        from rpu_backend.model_registry import model_path

        registry_base = os.fspath(model_path("qwen3-vl-4b"))
    except (KeyError, OSError):
        registry_base = None
    for cand in (
        os.path.join(pkg_root, "qwen3_vl_base"),   # packaged runtime bundle (next to the .so)
        os.path.join(here, "_qwen3vl_base"),       # in-package data — always ships
        registry_base,
    ):
        if cand and os.path.isfile(os.path.join(cand, "config.json")):
            return cand
    return None


def _env_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() in ("1", "true", "True", "on")


def _env_exact_one(name: str) -> bool:
    """Return true only for the literal opt-in used by physical Z2."""
    return os.environ.get(name, "") == "1"


def _close_multiview_spm_z2_state(state) -> bool:
    """Close retained physical Z2 before any child GraphCache or handle.

    ``state`` is deliberately a mutable dictionary captured by the native-handle
    finalizer.  That lets direct-builder teardown preserve the same ordering as
    explicit ``close()`` even though the Z2 plan is prepared lazily later.
    """
    if not state or not state.get("prepare_attempted", False):
        return True
    if state.get("owner_thread") != threading.get_ident():
        return False
    try:
        torch.ops.rpu.lingbot2_multiview_spm_z2_close(
            state["vision_handle"],
            state["text_handle"],
            state["expert_handle"],
            state["plan_hash"] if state.get("prepared", False) else 0,
        )
    except Exception:
        return False
    state["prepared"] = False
    state["prepare_attempted"] = False
    state["plan_hash"] = None
    return True


def _h(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float16).to("rpu").contiguous()


def resolve_rope_theta(text_config) -> float:
    """The expert's RoPE theta — sourced from the REAL config, never a hardcoded constant.

    The action expert does not own its RoPE: the VLM rotary embedding rotates
    expert q/k and supplies theta (5e6 for Qwen3-VL-4B). This raises instead of
    using a plausible but incorrect default.
    """
    rope_params = getattr(text_config, "rope_parameters", None)
    if rope_params is None:
        rope_params = getattr(text_config, "rope_scaling", None) or {}
    theta = rope_params.get("rope_theta", None)
    if theta is None:
        theta = getattr(text_config, "rope_theta", None)
    if theta is None:
        raise ValueError(
            "lingbot2: cannot resolve the VLM's rope_theta from the text config. The action "
            "expert is rotated by the VLM's rotary_emb, so its theta MUST come from the real "
            "config (5e6 for Qwen3-VL-4B); no fallback value is safe.")
    return float(theta)


def resolve_mrope_geometry(text_config) -> tuple[int, list[int]]:
    """Return the checkpoint-authoritative Qwen3-VL head dim and THW split."""
    rope_params = getattr(text_config, "rope_parameters", None)
    if rope_params is None:
        rope_params = getattr(text_config, "rope_scaling", None) or {}
    section = rope_params.get("mrope_section", None)
    if section is None:
        raise ValueError(
            "lingbot2: text config is missing rope_parameters['mrope_section']."
        )
    section = [int(value) for value in section]
    head_dim = getattr(text_config, "head_dim", None)
    if head_dim is None:
        head_dim = (
            int(text_config.hidden_size) // int(text_config.num_attention_heads)
        )
    head_dim = int(head_dim)
    if len(section) != 3 or sum(section) != head_dim // 2:
        raise ValueError(
            "lingbot2: invalid Qwen3-VL M-RoPE geometry: "
            f"head_dim={head_dim}, mrope_section={section}."
        )
    return head_dim, section


def _build_prefill_mrope_inputs(
    position_ids: torch.Tensor,
    *,
    head_dim: int,
    rope_theta: float,
    mrope_section: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize one HF M-RoPE index and bake decoder-native partial tables."""
    if tuple(position_ids.shape[:2]) != (3, 1) or position_ids.dim() != 3:
        raise ValueError(
            "lingbot2: prefill position_ids must be [3,1,S], got "
            f"{tuple(position_ids.shape)}")
    position_ids_for_decoder = (
        position_ids.squeeze(1).transpose(0, 1)
        .to(device="cpu", dtype=torch.int32).contiguous()
    )
    rope_cos_il, rope_sin_il = build_interleaved_mrope_cos_sin(
        position_ids_for_decoder.to(torch.int64),
        head_dim=head_dim,
        rope_theta=rope_theta,
        mrope_section=mrope_section,
    )
    return position_ids_for_decoder, rope_cos_il, rope_sin_il


def build_suffix_rope_window_cpu(
    position_start: int,
    *,
    seq_len: int,
    head_dim: int,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the exact fp16 1-D RoPE rows consumed by the action suffix."""
    position_start = int(position_start)
    seq_len = int(seq_len)
    head_dim = int(head_dim)
    if position_start < 0 or seq_len <= 0 or head_dim <= 0 or head_dim % 2:
        raise ValueError(
            "invalid suffix RoPE geometry: "
            f"start={position_start}, seq_len={seq_len}, head_dim={head_dim}")
    if not math.isfinite(float(rope_theta)) or float(rope_theta) <= 0:
        raise ValueError(f"rope_theta must be finite and positive, got {rope_theta}")
    inv_freq = 1.0 / (
        float(rope_theta) ** (
            torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim
        )
    )
    positions = torch.arange(
        position_start, position_start + seq_len, dtype=torch.float64)
    freqs = positions[:, None] * inv_freq[None, :]
    # Matches convert._build_rope_tables: double trig -> fp32 -> fp16.
    return (
        freqs.cos().float().half().contiguous(),
        freqs.sin().float().half().contiguous(),
    )


def _op_has_cos_sin_offset() -> bool:
    """Does the built .so expose the trailing `cos_sin_offset` on lingbot_v2_moe_forward?

    The trailing ``int cos_sin_offset=-1`` uses -1 for "same as position",
    >=0 == the RoPE position base while `position` keeps driving the KV-insert offset.

    We read the REGISTERED SCHEMA rather than try/except-ing a call: a TypeError from a real
    call could equally mean a genuine bug, and silently falling back would mis-route the RoPE
    base (exactly the silent-wrong-action failure this whole guard exists to prevent).
    """
    try:
        schema = torch.ops.rpu.lingbot_v2_moe_forward.default._schema
        return any(a.name == "cos_sin_offset" for a in schema.arguments)
    except Exception:
        return False


def _op_has_adarms_direct_schedule() -> bool:
    """Whether native LingBot2 supports one-time schedule bind + forward step select."""
    try:
        bind_schema = torch.ops.rpu.lingbot_v2_moe_bind_adarms_schedule.default._schema
        forward_schema = torch.ops.rpu.lingbot_v2_moe_forward.default._schema
        bind_args = [arg.name for arg in bind_schema.arguments]
        forward_args = [arg.name for arg in forward_schema.arguments]
        return (
            bind_args == ["handle", "input_scale", "input_shift", "post_scale", "post_shift"]
            and forward_args == [
                "handle", "hidden_states", "k_caches", "v_caches", "attention_mask",
                "position", "cos_sin_offset", "adarms_schedule_step",
            ]
            and "int adarms_schedule_step=-1" in str(forward_schema)
        )
    except Exception:
        return False


def _validate_denoise_unroll() -> bool:
    """Cold preflight for the controlled 10-step fused-denoise path."""
    enabled = _env_on("RPU_LINGBOT2_DENOISE_UNROLL", "0")
    if not enabled:
        return False
    if _env_on("RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE", "0"):
        raise ValueError(
            "RPU_LINGBOT2_DENOISE_UNROLL=1 and "
            "RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE=1 are mutually exclusive; "
            "the unrolled graph consumes the indexed schedule directly.")
    try:
        torch.ops.rpu.lingbot_v2_moe_set_denoise_unroll.default._schema
        torch.ops.rpu.lingbot_v2_moe_denoise_unroll_forward.default._schema
    except Exception as exc:
        raise RuntimeError(
            "RPU_LINGBOT2_DENOISE_UNROLL=1 requires a rebuilt backend with "
            "lingbot_v2_moe_set_denoise_unroll and "
            "lingbot_v2_moe_denoise_unroll_forward.") from exc
    return True


_MULTIVIEW_SPM_Z2_COMMON_ENV: dict[str, str] = {
    "RPU_LINGBOT2_DENOISE_UNROLL": "1",
    "RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE": "0",
    "RPU_LINGBOT2_EXPERT_REPLAY": "1",
    "RPU_LINGBOT2_FP16_TOP4": "1",
    "RPU_QWEN3VL_VISION_FUSED_MERGER": "1",
    "RPU_QWEN3VL_VISION_ROPE_SPM": "1",
    # The separate denoise Graph has no physical-pipeline lease or semantic
    # producer yield, so it may skip its full body and the
    # resulting redundant param sync.  Leased Vision/Text members remain on
    # the outer composite occurrence walk.  Deep replay is dtype-profile scoped.
    "RPU_WALL_OSS_FAST_REPLAY": "1",
    "RPU_FASTREPLAY_SKIP_SYNC": "1",
    "RPU_LINGBOT2_VISION_DIRECT_PREFIX": "0",
    "RPU_LINGBOT2_VISION_DIRECT_PREFIX_NO_OUTPUT": "0",
    "RPU_LINGBOT2_VISION_FAST_REPLAY": "0",
    "RPU_LINGBOT2_PREFILL_FAST_REPLAY": "0",
    "RPU_LINGBOT2_VISION_PACKED_PREP": "1",
    "RPU_QWEN3VL_VISION_HOST_FP32_PATCH": "0",
    "RPU_QWEN3VL_VISION_BATCH": "0",
    "RPU_KVINSERT_HYBRID_V16": "0",
    "RPU_KVINSERT_HYBRID3_V16": "0",
    "RPU_ADARMS_FUSED_BCAST": "0",
    "RPU_LINGBOT2_DOWN_ACC32_TILE": "0",
    "RPU_ALLREDUCE_TWOSTAGE": "0",
    "RPU_ALLREDUCE_CHUNK_V2": "0",
    # The exact profile freezes both ring variants in its graph census.
    "RPU_ALLREDUCE_RING": "profile",
}

_MULTIVIEW_SPM_Z2_PROFILE_ENV: dict[str, dict[str, str]] = {
    "dense_fp16": {
        "RPU_DEEP_FAST_REPLAY": "0",
        "RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE": "0",
        "RPU_LINGBOT2_DENOISE_TWOSTAGE": "0",
        "RPU_LINGBOT2_OUTER_TWOSTAGE": "0",
        "RPU_LINGBOT2_GROUPED_EXPERTS": "0",
        "RPU_LINGBOT2_EXPERT_W8A16": "0",
        "RPU_LINGBOT2_EXPERT_W4A16": "0",
        "RPU_LINGBOT2_PREFILL_W8A16": "0",
        "RPU_LINGBOT2_PREFILL_W4A16": "0",
        "RPU_LINGBOT2_BASE_W8A16": "0",
        "RPU_LINGBOT2_VISION_W8A16": "0",
        "RPU_L2_RCHUNK": "0",
    },
    "grouped_w8a16": {
        "RPU_DEEP_FAST_REPLAY": "1",
        "RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE": "1",
        "RPU_LINGBOT2_DENOISE_TWOSTAGE": "1",
        "RPU_LINGBOT2_OUTER_TWOSTAGE": "1",
        "RPU_LINGBOT2_GROUPED_EXPERTS": "1",
        "RPU_LINGBOT2_EXPERT_W8A16": "1",
        "RPU_LINGBOT2_EXPERT_W4A16": "0",
        "RPU_LINGBOT2_PREFILL_W8A16": "1",
        "RPU_LINGBOT2_PREFILL_W4A16": "0",
        "RPU_LINGBOT2_BASE_W8A16": "1",
        "RPU_LINGBOT2_VISION_W8A16": "1",
        "RPU_L2_RCHUNK": "1632",
        "RPU_LINEAR_AUTOTILE": "1",
    },
    "grouped_w4a16": {
        "RPU_DEEP_FAST_REPLAY": "1",
        "RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE": "1",
        "RPU_LINGBOT2_DENOISE_TWOSTAGE": "1",
        "RPU_LINGBOT2_OUTER_TWOSTAGE": "1",
        "RPU_LINGBOT2_GROUPED_EXPERTS": "1",
        "RPU_LINGBOT2_EXPERT_W8A16": "0",
        "RPU_LINGBOT2_EXPERT_W4A16": "1",
        "RPU_LINGBOT2_PREFILL_W8A16": "1",
        "RPU_LINGBOT2_PREFILL_W4A16": "0",
        "RPU_LINGBOT2_BASE_W8A16": "1",
        "RPU_LINGBOT2_VISION_W8A16": "1",
        "RPU_L2_RCHUNK": "1632",
        "RPU_LINEAR_AUTOTILE": "1",
    },
}


def _resolve_multiview_spm_z2_profile() -> str:
    """Return the one sealed multiview-Z2 profile represented by the live env."""
    matches = [
        profile
        for profile, required in _MULTIVIEW_SPM_Z2_PROFILE_ENV.items()
        if all(
            os.environ.get(name, "0") == expected
            for name, expected in required.items()
        )
    ]
    if len(matches) != 1:
        names = sorted({
            name
            for required in _MULTIVIEW_SPM_Z2_PROFILE_ENV.values()
            for name in required
        })
        actual = {name: os.environ.get(name, "0") for name in names}
        raise RuntimeError(
            "lingbot2 multiview SPM Z2 requires exactly one sealed "
            "Dense FP16, Grouped W8A16, or Grouped W4A16 profile; got "
            f"{actual}"
        )
    return matches[0]


def _validate_multiview_spm_z2() -> bool:
    """Cold preflight for a sealed production Vision->Text physical Z2."""
    if not _env_exact_one("RPU_LINGBOT2_MULTIVIEW_SPM_Z2"):
        return False

    # The facade selects by dtype; the direct builder has no dtype token here,
    # so re-derive the exact bundle from the live conversion env.  Native cold
    # prime remains authoritative for every bound W8/W4 tensor and scale shape.
    _resolve_multiview_spm_z2_profile()
    required = _MULTIVIEW_SPM_Z2_COMMON_ENV
    drift = {
        name: os.environ.get(name)
        for name, expected in required.items()
        if os.environ.get(name, "0") != expected
    }
    preproc = os.environ.get("RPU_LINGBOT2_PREPROC", "exact").strip().lower()
    if os.environ.get("RPU_LINGBOT2_LEGACY_PREPROC", "0") != "0":
        drift["RPU_LINGBOT2_LEGACY_PREPROC"] = os.environ.get(
            "RPU_LINGBOT2_LEGACY_PREPROC"
        )
    if preproc not in ("", "exact"):
        drift["RPU_LINGBOT2_PREPROC"] = preproc

    if drift:
        raise RuntimeError(
            "lingbot2 multiview SPM Z2 sealed-profile environment drifted: "
            f"{drift}"
        )

    schemas = {
        "lingbot2_multiview_spm_z2_prepare": [
            "vision_handle", "text_handle", "expert_handle", "position_ids",
            "rope_cos_il", "rope_sin_il",
        ],
        "lingbot2_multiview_spm_z2_forward": [
            "vision_handle", "text_handle", "expert_handle", "vision_inputs",
            "vision_k_caches", "vision_v_caches", "text_hidden",
            "text_k_caches", "text_v_caches", "position_ids", "rope_cos_il",
            "rope_sin_il", "plan_hash",
        ],
        "lingbot2_multiview_spm_z2_refresh_text_inputs": [
            "vision_handle", "text_handle", "expert_handle", "position_ids",
            "rope_cos_il", "rope_sin_il", "plan_hash",
        ],
        "lingbot2_multiview_spm_z2_stats": [
            "vision_handle", "text_handle", "expert_handle", "plan_hash"
        ],
        "lingbot2_multiview_spm_z2_close": [
            "vision_handle", "text_handle", "expert_handle", "plan_hash"
        ],
    }
    try:
        for name, expected_args in schemas.items():
            schema = getattr(torch.ops.rpu, name).default._schema
            actual_args = [argument.name for argument in schema.arguments]
            if actual_args != expected_args:
                raise RuntimeError(
                    f"{name} schema arguments drifted: {actual_args} != "
                    f"{expected_args}"
                )
    except Exception as exc:
        raise RuntimeError(
            "RPU_LINGBOT2_MULTIVIEW_SPM_Z2=1 requires a rebuilt backend "
            "with the exact LingBot2 production prepare/refresh/forward/stats/close ops."
        ) from exc
    return True


def _validate_adarms_direct_schedule() -> bool:
    """Cold preflight before any LingBot2 RPU materialization; return whether enabled."""
    enabled = _env_on("RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE", "0")
    if not enabled:
        return False
    if _env_on("RPU_LINGBOT2_DENOISE_UNROLL", "0"):
        raise ValueError(
            "RPU_LINGBOT2_DENOISE_UNROLL=1 and "
            "RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE=1 are mutually exclusive; "
            "the unrolled graph consumes the indexed schedule directly.")
    if not _env_on("RPU_LINGBOT2_EXPERT_REPLAY", "1"):
        raise ValueError(
            "RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE=1 requires "
            "RPU_LINGBOT2_EXPERT_REPLAY=1.")
    if not _env_on("RPU_ADARMS_FUSED_BCAST", "0"):
        raise ValueError(
            "RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE=1 requires "
            "RPU_ADARMS_FUSED_BCAST=1.")
    if not _op_has_adarms_direct_schedule():
        raise RuntimeError(
            "RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE=1 requires a rebuilt backend with "
            "lingbot_v2_moe_bind_adarms_schedule and the forward "
            "adarms_schedule_step argument.")
    return True


# =============================================================================
# Host encoders + AdaRMS fold. Sinusoidal constants are min_period 4e-3,
# max_period 4.0, and dim = proj_width = 768.
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


def precompute_enc_fused(W, time_embs):
    r"""Fold the consecutive suffix-encoder linears before the nonlinearity.

    With ``Wa = mlp_in.weight[:, :768]`` and ``Wt`` the remaining columns:

        (x @ Wp.T + bp) @ Wa.T + (t @ Wt.T + b)
            == x @ (Wa @ Wp).T  +  (bp @ Wa.T + t @ Wt.T + b)

    Returns (W_fused [768,55] contiguous, [num_steps] list of [1,1,768] step biases).
    """
    Win = W["model.action_time_mlp_in.weight"]                 # [768, 1536]
    b_mlp = W["model.action_time_mlp_in.bias"]                 # [768]
    d = Win.shape[1] // 2                                      # 768 = action_emb width
    Wa, Wt = Win[:, :d], Win[:, d:]
    Wp = W["model.action_in_proj.weight"]                      # [768, 55]
    bp = W["model.action_in_proj.bias"]                        # [768]
    W_fused = (Wa @ Wp).contiguous()                           # [768, 55]
    b_proj = F.linear(bp, Wa)                                  # bp @ Wa.T -> [768]
    parts = [(F.linear(te, Wt, b_mlp) + b_proj).reshape(1, 1, -1) for te in time_embs]
    return W_fused, parts


def embed_suffix_step(x_t, state_emb, time_part, W, W_fused):
    """Per-Euler-step suffix embed. state_emb / time_part are call-invariant → precomputed.

    `time_part` carries BOTH action_time_mlp_in's bias and action_in_proj's bias mapped through
    Wa (see precompute_enc_fused), so the projection below is intentionally bias-free.
    """
    ate = F.silu(F.linear(x_t, W_fused) + time_part)
    ate = F.linear(ate, W["model.action_time_mlp_out.weight"], W["model.action_time_mlp_out.bias"])
    return torch.cat([state_emb[:, None], ate], dim=1)                      # [1, 51, 768]


def precompute_fold_all(norm_W, times):
    """Batch the AdaRMS fold across all Euler steps.

    ``ada_cond`` depends only on timestep. Returns four RPU tensors shaped
    ``[num_steps, EXP_LAYERS, 768]``; slice ``[k]`` supplies one step.

    gamma/beta are [768,768] LINEARS (not norm scales): scale = (1 + gamma(cond)) * rms_w,
    shift = beta(cond).
    """
    S = len(times)
    in_s = torch.empty(S, EXP_LAYERS, EXP_HID, dtype=torch.float16)
    in_sh, po_s, po_sh = (torch.empty_like(in_s) for _ in range(3))
    for k, t in enumerate(times):
        cond = _sinusoidal_pos_embedding(torch.tensor([float(t)]), EXP_HID, 4e-3, 4.0)
        for L in range(EXP_LAYERS):
            for slot, sc, sh in (("input_layernorm", in_s, in_sh),
                                 ("post_attention_layernorm", po_s, po_sh)):
                pf = f"{_cv.EXP_P}layers.{L}.{slot}"
                w = norm_W[pf + ".weight"]
                gamma = F.linear(cond, norm_W[pf + ".gamma.weight"], norm_W[pf + ".gamma.bias"])[0]
                beta = F.linear(cond, norm_W[pf + ".beta.weight"], norm_W[pf + ".beta.bias"])[0]
                sc[k, L] = ((1.0 + gamma) * w).half()
                sh[k, L] = beta.half()
    return (in_s.to("rpu"), in_sh.to("rpu"), po_s.to("rpu"), po_sh.to("rpu"))


def build_suffix_mask_bool(prefix_len, align_cfg, *, real_prefix_len=None):
    """Bool [51, prefix_len + 51] suffix attention mask, True == visible. Pure CPU.

    This matches the reference ``predict_velocity`` mask semantics:
      * suffix -> prefix  = the REAL-prefix mask, MINUS the blocked align spans
        (block_future_depth_to_action / block_suffix_to_future_video). Adapter-level
        execution padding is a masked tail and is never interpreted as align rows.
      * suffix -> suffix  = make_att_2d_masks(suffix_pad_masks, suffix_att_masks): state(tok0) is
        its own block, the 50 action tokens form one bidirectional block that also sees state.
    """
    prefix_len = int(prefix_len)
    real_prefix_len = (
        prefix_len if real_prefix_len is None else int(real_prefix_len)
    )
    if prefix_len <= 0 or not 0 < real_prefix_len <= prefix_len:
        raise ValueError(
            "require 0 < real_prefix_len <= prefix_len, got "
            f"{real_prefix_len}/{prefix_len}"
        )
    B, Q = 1, N_ACTION + 1                                       # 51
    suffix_att = torch.zeros(B, Q, dtype=torch.bool)
    suffix_att[:, :2] = True
    suffix_pad = torch.ones(B, Q, dtype=torch.bool)
    suffix_2d = _make_att_2d_masks(suffix_pad, suffix_att)[0]    # [51, 51]
    prefix_2d = torch.zeros(Q, prefix_len, dtype=torch.bool)     # [51, execution]
    prefix_2d[:, :real_prefix_len] = True
    apply_suffix_prefix_blocking_(prefix_2d, real_prefix_len, align_cfg)
    return torch.cat([prefix_2d, suffix_2d], dim=1)              # [51, prefix_len+51]


def _contiguous_runs(mask_1d):
    """Contiguous True runs of a 1-D bool mask → [(start, length), ...].

    Prefix image placeholders form contiguous runs, allowing one slice copy per
    run. Deriving runs from the mask keeps the result valid if layout changes;
    the caller validates the total against the Vision token count.
    """
    idx = mask_1d.nonzero().flatten().tolist()
    runs, start, prev = [], None, None
    for i in idx:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            runs.append((start, prev - start + 1))
            start = prev = i
    if start is not None:
        runs.append((start, prev - start + 1))
    return runs


def _scatter_rows_(dst, src, runs, offs):
    """dst[run] = src[contiguous slice], for each (run, offset) pair. In place, on any device."""
    for (st, ln), off in zip(runs, offs):
        dst.narrow(0, st, ln).copy_(src.narrow(0, off, ln))


def _refresh_prefix_memo_(memo, *, memo_key, embeds, position_ids,
                          rope_cos_il, rope_sin_il,
                          deepstack_features, runs, offs, p0):
    """Refresh one execution bucket without changing any Graph-baked address."""
    # An outer inference_mode() must not suppress the version epoch on these
    # normal mutable owners.  C++ uses that epoch to decide whether its stable
    # M-RoPE keepalive needs an in-place refill for a same-bucket prompt.
    with torch.inference_mode(False):
        memo["embeds"].copy_(embeds)
        for dense, feat in zip(memo["dense"], deepstack_features):
            dense.zero_()
            _scatter_rows_(dense, feat, runs, offs)
        memo["position_ids"].copy_(position_ids)
        # Pass the decoder-native [S,3] layout as the stable owner itself.
        # Otherwise decoder.py would materialize a new contiguous temporary on
        # every call and C++ would never observe this owner's version counter.
        position_ids_for_decoder = (
            memo["position_ids"].squeeze(1).transpose(0, 1)
            .to(dtype=torch.int32).contiguous()
        )
        memo["position_ids_rpu"].copy_(position_ids_for_decoder)
        memo["rope_cos_il"].copy_(rope_cos_il)
        memo["rope_sin_il"].copy_(rope_sin_il)
    memo.update({
        "key": memo_key,
        "p0": int(p0),
        "runs": list(runs),
        "offs": list(offs),
    })
    return memo


def _validate_direct_prefix_group(grid_thw, *, patches_per_image, batch_cap,
                                  cache_capacity):
    """Prove that Qwen3-VL will execute all images as exactly one packed group.

    Prefix-scatter consumes one contiguous merger output and walks every prefix
    run.  It is therefore unsafe when the generic Vision adapter splits the
    images into two or more forwards: each forward would walk all runs while
    owning only its group's merger rows.
    """
    grid = torch.as_tensor(grid_thw, dtype=torch.long).reshape(-1, 3).cpu()
    n_images = int(grid.shape[0])
    if n_images <= 0:
        raise ValueError("lingbot2 direct-prefix: at least one image is required.")
    if batch_cap < n_images:
        raise ValueError(
            f"lingbot2 direct-prefix: Vision batch cap {batch_cap} is smaller than "
            f"the {n_images} valid images; one packed group is required.")
    if any(not torch.equal(row, grid[0]) for row in grid[1:]):
        raise ValueError(
            "lingbot2 direct-prefix: every valid image must have exactly the same "
            "grid_thw so Vision emits one packed group.")

    implied = [int(row.prod().item()) for row in grid]
    if any(n != int(patches_per_image) for n in implied):
        raise ValueError(
            "lingbot2 direct-prefix: grid_thw and pixel rows disagree: "
            f"grid implies {implied}, input carries {patches_per_image} patches/image.")
    packed_patches = n_images * int(patches_per_image)
    if packed_patches > int(cache_capacity):
        raise ValueError(
            f"lingbot2 direct-prefix: packed Vision sequence {packed_patches} exceeds "
            f"KV capacity {cache_capacity}; one packed group is impossible.")

    # Keep this identical to qwen3_vl.vision._minibatch_sdpa_supported.  Every
    # incremental group size must pass because the generic grouping loop stops
    # permanently at the first unsupported size.
    if n_images > 1:
        blocks_per_image, remainder = divmod(int(patches_per_image), 128)
        if remainder or blocks_per_image < 2:
            raise ValueError(
                "lingbot2 direct-prefix: minibatch SDPA requires patches/image to be "
                f"a multiple of 128 with at least two blocks; got {patches_per_image}.")
        for group_size in range(2, n_images + 1):
            grid_dim_x = blocks_per_image * group_size
            if grid_dim_x > 8 and grid_dim_x % 8:
                raise ValueError(
                    "lingbot2 direct-prefix: minibatch SDPA rejects intermediate group "
                    f"size {group_size} (grid_dim_x={grid_dim_x}); Vision would split the "
                    "images, which prefix-scatter cannot consume safely.")


def _validate_prefix_scatter_runs(runs, *, prefix_rows, expected_rows,
                                  expected_runs):
    """Validate fixed-DMA destinations before publishing their addresses."""
    if len(runs) != int(expected_runs):
        raise ValueError(
            f"lingbot2 direct-prefix: layout produced {len(runs)} visual runs for "
            f"{expected_runs} images.")
    total = 0
    previous_end = 0
    for idx, (start, length) in enumerate(runs):
        start, length = int(start), int(length)
        end = start + length
        if start < previous_end or length <= 0 or end > int(prefix_rows):
            raise ValueError(
                "lingbot2 direct-prefix: invalid visual run "
                f"#{idx} ({start}, {length}) for prefix rows {prefix_rows}.")
        total += length
        previous_end = end
    if total != int(expected_rows):
        raise ValueError(
            f"lingbot2 direct-prefix: visual runs cover {total} rows, expected "
            f"{expected_rows} merger rows.")


def _clear_graph_caches_for_rebind(caches):
    """Drop baked addresses only while online BUILD is already permitted.

    READY is an explicit lookup-only deployment contract.  Prompt drift must
    fail before mutating either cache rather than silently thawing in infer.
    """
    for cache, label in caches:
        if cache is None:
            raise RuntimeError(
                f"lingbot2 direct-prefix: missing {label} GraphCache.")
        if getattr(cache, "is_frozen", lambda: False)():
            raise RuntimeError(
                "lingbot2 direct-prefix: prompt geometry changed while "
                f"{label} GraphCache is READY; rebuild/re-warm a fresh policy.")
    for cache, _label in caches:
        cache.clear()


def _build_suffix_mask(prefix_len, align_cfg, *, real_prefix_len=None):
    """Additive fp16 [51, prefix_len + 51] on RPU: 0 attend, big_neg masked.

    Built on CPU then moved because direct RPU zero allocation is unsupported.
    """
    full = build_suffix_mask_bool(
        prefix_len, align_cfg, real_prefix_len=real_prefix_len
    )
    add = torch.where(full, torch.tensor(0.0), torch.tensor(-50000.0)).half()
    return add.to("rpu").contiguous()


# =============================================================================
# Policy
# =============================================================================
class LingbotVlaV2Policy:
    """Image+text → action [1, 50, 55]. Build via `build_lingbot_vla_v2(...)`."""

    def __init__(self, *, vlm, cfg, head_W, norm_W, align_W, align_cfg, rope_theta,
                 exp_handle, exp_keep, max_seq=DEFAULT_MAX_SEQ, chunk_size=DEFAULT_CHUNK_SIZE):
        # Establish cleanup state first: the public builder may call
        # _retire_resources() after any later constructor failure.
        self._closed = False
        self._cleanup_ok = None
        self._live_slot_released = False
        self._handle_finalizer = None
        self._multiview_spm_z2 = False
        self._z2_state = {
            "prepare_attempted": False,
            "prepared": False,
            "poisoned": False,
            "plan_hash": None,
            "owner_thread": None,
            "vision_handle": None,
            "text_handle": None,
            "expert_handle": None,
        }
        self._z2_prompt_key = None
        self._z2_prompt_metadata = None
        self._z2_text_hidden = None
        self._z2_position_ids = None
        self._z2_rope_cos_il = None
        self._z2_rope_sin_il = None
        self._vlm = vlm                              # HF Qwen3VLModel (visual + language_model on RPU)
        self._cfg = cfg
        self._text = vlm.language_model
        self._visual = vlm.visual
        self._head_W = head_W                        # CPU-fp32 encoders / action head
        self._norm_W = norm_W                        # CPU-fp32 AdaRMS norm + gamma/beta linears
        self._exp = exp_handle
        self._exp_keep = exp_keep                    # ⚠️ C++ holds raw pointers — never drop this
        self._max_seq = max_seq
        self._align_cfg = align_cfg
        self._ids = SpecialTokenIds.from_config(cfg)
        self._spatial_merge = int(cfg.vision_config.spatial_merge_size)
        self._rope_theta = rope_theta      # sourced from the VLM config
        self._mrope_head_dim, self._mrope_section = resolve_mrope_geometry(
            cfg.text_config)
        # The 8 current_depth + 8 future_depth prefix query tokens. Call-invariant (they depend
        # only on weights) → computed ONCE at build. [n_task, VLM_HID] fp32 each.
        self._align_tokens = compute_align_tokens(align_W, align_cfg)
        # Whether the built .so exposes the trailing `cos_sin_offset` arg on lingbot_v2_moe_forward.
        self._has_cos_sin_offset = _op_has_cos_sin_offset()

        # ONE shared RPUCache: the VLM fills [0, S); the expert reads that prefix KV cross-handle
        # and writes only its own suffix [S, S+51). Both halves are 8 KV heads / head_dim 128 /
        # attn_tp 8, so the 7-D swizzle layout is identical — this is what makes sharing legal.
        self._cache = RPUCache(num_layers=VLM_LAYERS, batch_size=1, max_seq_len=max_seq,
                               num_kv_heads=NKV, head_dim=HD, attn_tp=ATTN_TP)

        # PER-MODEL graph caches (never get_default_graph_cache() — a shared default would let a
        # sibling model's signature collide with ours).
        self._vgc = rpu_backend.graph.GraphCache()
        self._egc = rpu_backend.graph.GraphCache()
        self._prepared_graph_profile = None
        self._graphs_ready = False
        self._last_prefix_metadata = None

        # Chunk size is a per-instance attribute read by the model forward;
        # torch.rpu.set_chunk_size() does not reach fused paths. Set it explicitly so the
        # chunk plan (and therefore the fp16 reduce order) is pinned rather than left to the C++
        # binary-search auto-select. Must be set AFTER .to('rpu').
        self._chunk_size = int(chunk_size)
        self._text._rpu_chunk_size = self._chunk_size

        # The action expert owns its own 1-D RoPE tables.  READY mode refreshes
        # rows [0,51) in place with the current semantic p0 window and captures
        # one constant-offset graph per execution bucket.  The table objects and
        # therefore the Graph-baked DDR addresses never change.
        if len(exp_keep) < 20:
            raise RuntimeError("lingbot2 expert keepalive is missing RoPE owners")
        self._expert_rope_cos = exp_keep[17]
        self._expert_rope_sin = exp_keep[18]
        self._ready_rope_p0 = None
        self._ready_rope_cpu = {}

        # Call-invariant Euler schedule: iterative `time += dt` (NOT closed-form) so the timesteps
        # match the official loop bit-for-bit.
        self._times, _t = [], 1.0
        for _ in range(NUM_STEPS):
            self._times.append(_t)
            _t += -1.0 / NUM_STEPS
        self._fold_all = precompute_fold_all(norm_W, self._times)
        self._time_embs = [_sinusoidal_pos_embedding(torch.tensor([t]), EXP_HID, 4e-3, 4.0)
                           for t in self._times]
        # Collapse action_in_proj + the action half of action_time_mlp_in into one [768,55]
        # projection, and hoist the step-constant time term out of the Euler loop.
        self._enc_Wf, self._enc_time_parts = precompute_enc_fused(head_W, self._time_embs)

        # Build-once/replay expert denoise: one graph signature
        # for all NUM_STEPS steps + set_adarms_step_mutable (no per-step rebuild) + a STABLE suffix
        # input buffer refreshed in place ⇒ step 0 BUILDs, steps 1..9 REPLAY.
        # `lingbot_v2_moe_forward` bakes the input tensor's DDR address at BUILD (fixed-DMA), so the
        # per-forward input MUST live at a stable address for a replay to be valid.
        self._expert_replay = _env_on("RPU_LINGBOT2_EXPERT_REPLAY", "1")
        self._denoise_unroll = _validate_denoise_unroll()
        self._multiview_spm_z2 = _validate_multiview_spm_z2()
        self._adarms_direct_schedule = _validate_adarms_direct_schedule()
        if self._adarms_direct_schedule:
            torch.ops.rpu.lingbot_v2_moe_bind_adarms_schedule(
                self._exp, *self._fold_all)
        self._suffix_buf = None
        if not self._denoise_unroll:
            self._suffix_buf = torch.empty(
                1, N_ACTION + 1, EXP_HID, dtype=torch.float16, device="rpu")
        self._dgc = None
        if self._denoise_unroll:
            self._dgc = rpu_backend.graph.GraphCache()
            self._build_denoise_unroll()
        if self._multiview_spm_z2:
            vc = self._cfg.vision_config
            tc = self._cfg.text_config
            vision_geometry = (
                int(vc.depth), int(vc.hidden_size), int(vc.intermediate_size),
                int(vc.num_heads), int(vc.hidden_size) // int(vc.num_heads),
                int(vc.out_hidden_size), int(vc.spatial_merge_size),
                tuple(int(index) for index in vc.deepstack_visual_indexes),
            )
            text_geometry = (
                int(tc.num_hidden_layers), int(tc.hidden_size),
                int(tc.intermediate_size), int(tc.num_attention_heads),
                int(tc.num_key_value_heads), int(tc.head_dim),
            )
            if vision_geometry != (
                24, 1024, 4096, 16, 64, 2560, 2, (5, 11, 17)
            ) or text_geometry != (36, 2560, 9728, 32, 8, 128):
                raise RuntimeError(
                    "lingbot2 multiview SPM Z2 requires the exact Qwen3-VL-4B "
                    f"Vision/Text profile, got {vision_geometry} / {text_geometry}."
                )
            if self._align_cfg != AlignConfig() or self._max_seq < 276:
                raise RuntimeError(
                    "lingbot2 multiview SPM Z2 requires the authoritative "
                    "kuavo_v2_depth align layout and KV capacity >=276."
                )
            if not self._denoise_unroll:
                raise RuntimeError(
                    "lingbot2 multiview SPM Z2 requires fused denoise unroll."
                )
            z2_embedding = getattr(self, "_z2_embedding_cpu", None)
            if (not isinstance(z2_embedding, torch.Tensor)
                    or z2_embedding.device.type != "cpu"
                    or z2_embedding.dtype != torch.float16
                    or z2_embedding.dim() != 2
                    or z2_embedding.size(1) != VLM_HID
                    or not z2_embedding.is_contiguous()):
                raise RuntimeError(
                    "lingbot2 multiview SPM Z2 requires the retained CPU FP16 "
                    "token-embedding table for variable prompt refresh.")
            self._z2_state.update({
                "vision_handle": getattr(
                    self._visual, "_rpu_vision_handle", None),
                "text_handle": getattr(
                    self._text, "_rpu_decoder_handle", None),
                "expert_handle": self._exp,
            })
            if not all(
                type(self._z2_state[name]) is int
                and self._z2_state[name] > 0
                for name in (
                    "vision_handle", "text_handle", "expert_handle"
                )
            ):
                raise RuntimeError(
                    "lingbot2 multiview SPM Z2 requires live integer "
                    "Vision/Text/expert handles."
                )
        # The suffix MASK is cached per prefix length S so its DDR address survives across calls
        # (a fresh per-call mask would be freed → a replayed graph would read garbage).
        self._mask_cache: dict[int, torch.Tensor] = {}
        # Optionally run the host encoder with one thread; 0 preserves the
        # process-wide PyTorch setting.
        self._enc_1thread = _env_on("RPU_LINGBOT2_ENCODER_1THREAD", "1")
        # Prompt-constant prefix assembly defaults on. The
        # whole prefix except the visual rows is a pure function of the PROMPT (image count,
        # tokens/image, language tokens, grid), which a deployed policy holds fixed for an entire
        # episode while camera frames change every control tick. It is built once
        # into RPU-resident buffers and only the visual rows are overwritten.
        self._prefix_opt = _env_on("RPU_LINGBOT2_HOST_PREFIX_OPT", "1")
        self._pfx_memo = None          # prompt-keyed prefix buffers (see _build_prefix)
        # One stable-address carrier per execution length.  Graph fixed-DMA
        # bindings key on shape, not token values; keeping only the most recent
        # prompt used to allocate a new carrier for a same-length instruction
        # and then REPLAY against the freed old address.  Each bucket below owns
        # the one address baked by that execution signature and refreshes its
        # contents in place when prompt identity changes.
        self._pfx_memos = {}
        self._pos_ids_src = None       # identity key for the position_ids H2D memo (_vlm_fill)
        self._pos_rpu = None
        # Cross-call VLM prefill replay is safe only when `_prefix_opt` keeps
        # inputs_embeds, all three dense deepstack tensors, and the device copy
        # of position_ids at stable addresses and refreshes their contents in
        # place. Therefore replay follows `_prefix_opt` and remains off without it.
        self._vlm_replay = _env_on("RPU_LINGBOT2_VLM_REPLAY",
                                   "1" if self._prefix_opt else "0")
        if self._vlm_replay and not self._prefix_opt:
            # Replay with per-call allocations would leave fixed-DMA graph nodes
            # pointing at stale addresses, so reject the combination before capture.
            raise RuntimeError(
                "lingbot2: RPU_LINGBOT2_VLM_REPLAY=1 with RPU_LINGBOT2_HOST_PREFIX_OPT=0. "
                "The prefill graph is replayed against inputs_embeds / dense deepstack / "
                "position_ids that are re-ALLOCATED every call, so the replayed graph may read "
                "freed addresses. Persistent prefix buffers are required. Set "
                "HOST_PREFIX_OPT=1, or VLM_REPLAY=0.")

        # The optional fused Vision
        # merger writes pooler + all three DeepStack outputs directly into the
        # persistent VLM prefix tensors, eliminating twelve synchronous Python
        # narrow().copy_() launches per frame.  The native DMA destinations are
        # fixed at Vision Graph BUILD, so this is deliberately cold and tightly
        # gated to the single-packed-group path.
        self._vision_direct_prefix = _env_on(
            "RPU_LINGBOT2_VISION_DIRECT_PREFIX", "0")
        self._vision_direct_prefix_no_output = _env_on(
            "RPU_LINGBOT2_VISION_DIRECT_PREFIX_NO_OUTPUT", "0")
        if (self._vision_direct_prefix_no_output
                and not self._vision_direct_prefix):
            raise RuntimeError(
                "lingbot2 VISION_DIRECT_PREFIX_NO_OUTPUT=1 requires "
                "VISION_DIRECT_PREFIX=1.")
        if self._vision_direct_prefix:
            if not self._prefix_opt or not self._vlm_replay:
                raise RuntimeError(
                    "lingbot2 direct-prefix requires HOST_PREFIX_OPT=1 and "
                    "VLM_REPLAY=1 so every fixed-DMA target remains strongly owned.")
            if os.environ.get("RPU_QWEN3VL_VISION_BATCH", "0") != "1":
                raise RuntimeError(
                    "lingbot2 direct-prefix requires RPU_QWEN3VL_VISION_BATCH=1.")
            if not getattr(self._visual, "_rpu_vision_fused_merger", False):
                raise RuntimeError(
                    "lingbot2 direct-prefix requires the fused Vision merger.")
            if self._spatial_merge != 2:
                raise RuntimeError(
                    "lingbot2 direct-prefix requires spatial_merge_size=2; "
                    f"installed Vision uses {self._spatial_merge}.")
            if not hasattr(
                    torch.ops.rpu, "qwen3vl_vision_set_prefix_scatter"):
                raise RuntimeError(
                    "lingbot2 direct-prefix requires a backend build with "
                    "qwen3vl_vision_set_prefix_scatter.")
            n_deepstack = len(getattr(
                self._visual, "_rpu_vision_deepstack_indexes", ()))
            if n_deepstack != 3:
                raise RuntimeError(
                    "lingbot2 direct-prefix requires exactly three fused DeepStack "
                    f"merger outputs; installed Vision has {n_deepstack}.")
            if self._vision_direct_prefix_no_output:
                self._visual._rpu_prefix_scatter_consume_only_allowed = True

        text_finalizer = getattr(
            self._text, "_rpu_decoder_handle_finalizer", None
        )
        vision_finalizer = getattr(
            self._visual, "_rpu_vision_handle_finalizer", None
        )
        self._handle_finalizer = weakref.finalize(
            self,
            _destroy_policy_handles,
            self._exp,
            getattr(self._text, "_rpu_decoder_handle", None),
            text_finalizer,
            getattr(self._visual, "_rpu_vision_handle", None),
            vision_finalizer,
            self._z2_state,
        )

    def _drop_device_state(self) -> None:
        """Drop policy-owned RPU tensors after confirmed native cleanup."""
        for attr in (
            "_cache",
            "_vgc",
            "_egc",
            "_dgc",
            "_exp_keep",
            "_fold_all",
            "_suffix_buf",
            "_denoise_keep",
            "_denoise_state_host",
            "_denoise_x0_host",
            "_denoise_state",
            "_denoise_x0",
            "_denoise_x_final",
            "_mask_cache",
            "_pfx_memo",
            "_pfx_memos",
            "_pos_rpu",
            "_z2_text_hidden",
            "_z2_position_ids",
            "_z2_rope_cos_il",
            "_z2_rope_sin_il",
            "_z2_prompt_key",
            "_z2_prompt_metadata",
            "_z2_embedding_cpu",
            "_expert_rope_cos",
            "_expert_rope_sin",
            "_ready_rope_cpu",
            "_prepared_graph_profile",
            "_last_prefix_metadata",
        ):
            if hasattr(self, attr):
                setattr(self, attr, None)
        self._graphs_ready = False
        self._exp = None
        self._text = None
        self._visual = None
        self._vlm = None

    def _retire_resources(self) -> bool:
        """Clear graph ownership, then retire all native handles once."""
        if getattr(self, "_closed", False):
            return getattr(self, "_cleanup_ok", False) is True
        self._closed = True
        # The retained physical Graph owns a parked arena guard.  It must be
        # retired before any child GraphCache or handle is touched.  On failure
        # stop immediately: clearing a child cache or destroying a handle would
        # strand the retained Graph with stale model/arena identity.
        if not _close_multiview_spm_z2_state(
            getattr(self, "_z2_state", None)
        ):
            self._cleanup_ok = False
            return False
        cleanup_ok = True

        text = getattr(self, "_text", None)
        visual = getattr(self, "_visual", None)
        for owner, attr in (
            (self, "_dgc"),
            (self, "_egc"),
            (self, "_vgc"),
            (text, "_rpu_text_graph_cache"),
            (text, "_rpu_decoder_graph_cache"),
            (visual, "_rpu_vision_graph_cache"),
        ):
            if not _clear_graph_cache(owner, attr):
                cleanup_ok = False

        finalizer = getattr(self, "_handle_finalizer", None)
        if finalizer is not None:
            if getattr(finalizer, "alive", False) and finalizer() is not True:
                cleanup_ok = False
        else:
            if not _destroy_policy_handles(
                getattr(self, "_exp", None),
                getattr(text, "_rpu_decoder_handle", None),
                getattr(text, "_rpu_decoder_handle_finalizer", None),
                getattr(visual, "_rpu_vision_handle", None),
                getattr(visual, "_rpu_vision_handle_finalizer", None),
            ):
                cleanup_ok = False

        if cleanup_ok:
            self._drop_device_state()
        self._cleanup_ok = cleanup_ok
        return cleanup_ok

    def close(self) -> None:
        """Release graph/native resources and the process-wide policy slot."""
        if not self._retire_resources():
            raise RuntimeError(
                "LingBot-VLA-V2 cleanup failed; the process-wide RPU slot "
                "remains claimed. Restart the process before loading another "
                "policy."
            )
        if getattr(self, "_live_slot_released", False):
            return
        from rpu_backend.api.causal_lm import _release_live_instance

        _release_live_instance(self)
        self._live_slot_released = True

    # ---------------------------------------------------------------- prefix
    def _prefix_execution_plan(self, real_len):
        """Resolve one REAL prefix through the active finite profile."""
        profile = self._prepared_graph_profile
        if profile is not None:
            try:
                execution_len, chunk_size = profile["real_to_execution"][
                    int(real_len)
                ]
            except KeyError as exc:
                lo, hi = profile["prefix_length_range"]
                raise RuntimeError(
                    "LingBot2 REAL prefix is outside the prepared range: "
                    f"real_len={real_len}, prepared=[{lo}, {hi}]. No online "
                    "graph BUILD was attempted."
                ) from exc
            return plan_prefix_execution(
                int(real_len), max_seq=self._max_seq,
                suffix_len=N_ACTION + 1, chunk_size=int(chunk_size),
                fixed_execution_len=int(execution_len),
            )
        return plan_prefix_execution(
            int(real_len), max_seq=self._max_seq,
            suffix_len=N_ACTION + 1, chunk_size=self._chunk_size,
            padding_multiple=PREFIX_PADDING_MULTIPLE,
            fixed_execution_len=(
                MULTIVIEW_SPM_Z2_EXECUTION_LEN
                if self._multiview_spm_z2 else None
            ),
        )

    def _prefix_execution_profiles(self, prefix_length_range):
        """Map an inclusive REAL-prefix envelope to finite execution buckets."""
        if (not isinstance(prefix_length_range, (tuple, list))
                or len(prefix_length_range) != 2):
            raise ValueError(
                "prefix_length_range must be a two-item (min, max) sequence")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in prefix_length_range
        ):
            raise ValueError(
                "prefix_length_range entries must be integers, got "
                f"{prefix_length_range!r}")
        lo, hi = (int(prefix_length_range[0]), int(prefix_length_range[1]))
        if lo <= 0 or hi < lo:
            raise ValueError(
                f"invalid prefix_length_range=({lo}, {hi}); require 0 < min <= max")

        grouped = {}
        real_to_execution = {}
        for real_len in range(lo, hi + 1):
            plan = plan_prefix_execution(
                real_len,
                max_seq=self._max_seq,
                suffix_len=N_ACTION + 1,
                chunk_size=self._chunk_size,
                padding_multiple=PREFIX_PADDING_MULTIPLE,
                fixed_execution_len=(
                    MULTIVIEW_SPM_Z2_EXECUTION_LEN
                    if self._multiview_spm_z2 else None
                ),
            )
            key = plan.key
            real_to_execution[real_len] = key
            if key not in grouped:
                grouped[key] = [real_len, real_len]
            else:
                grouped[key][1] = real_len
        profiles = tuple(
            {
                "execution_len": execution_len,
                "chunk_size": chunk_size,
                "real_min": bounds[0],
                "real_max": bounds[1],
            }
            for (execution_len, chunk_size), bounds in sorted(grouped.items())
        )
        return profiles, real_to_execution

    def _image_grid(self, images):
        """obs['images'] [n_img, num_patch, patch_dim] → grid_thw [[1, g, g]] per image."""
        num_patch = images.shape[1]
        g = int(round(math.isqrt(num_patch)))
        if g * g != num_patch:
            raise ValueError(f"lingbot2 ViT: {num_patch} patches is not a square grid; "
                             f"pass a square patch grid.")
        if g % self._spatial_merge != 0:
            raise ValueError(f"lingbot2 ViT: grid {g} not divisible by spatial_merge_size "
                             f"{self._spatial_merge}.")
        return g

    def _check_grid_thw(self, grid_thw, images, keep_img):
        """Validate an EXPLICIT [n_img,3] (t,h,w) patch grid, as produced by the official
        Qwen3VLProcessor / Qwen2VLImageProcessorFast alongside `pixel_values`.

        This is the only way to express a NON-square image: 640x480 at patch_size=16 is
        grid [1,30,40], which `_image_grid`'s isqrt() cannot represent (1200 is not a
        square). Nothing is inferred here — the grid comes from the processor.

        Returns (grid_thw restricted to VALID images, post-merge tokens per image).
        """
        gt = torch.as_tensor(grid_thw, dtype=torch.long).reshape(-1, 3)
        if gt.shape[0] != images.shape[0]:
            raise ValueError(f"lingbot2: image_grid_thw has {gt.shape[0]} rows but obs['images'] "
                             f"has {images.shape[0]} images.")
        gt = gt[keep_img]
        m = self._spatial_merge
        toks = []
        for t, h, w in gt.tolist():
            if t != 1:
                raise ValueError(f"lingbot2: image_grid_thw t={t}; still images require t==1.")
            if h % m or w % m:
                raise ValueError(f"lingbot2: grid {h}x{w} not divisible by spatial_merge_size {m}.")
            toks.append((h // m) * (w // m))
        # build_prefix_layout takes ONE num_patch for every image (embed_prefix emits one
        # identical <image>*P run per image), so mixed resolutions would silently mislay the
        # prefix. Refuse instead of guessing.
        if len(set(toks)) != 1:
            raise ValueError(f"lingbot2: all images must yield the same post-merge token count; "
                             f"got {toks}. Use equal-resolution images.")
        want = int((gt[:, 0] * gt[:, 1] * gt[:, 2]).sum())
        have = len(keep_img) * int(images.shape[1])
        if want != have:
            raise ValueError(f"lingbot2: image_grid_thw implies {want} patches but obs['images'] "
                             f"carries {have}.")
        return gt, toks[0]

    def _request_prefix_geometry(
        self, images, img_masks, lang_tokens, lang_masks, grid_thw
    ):
        """Resolve REAL prefix geometry using CPU metadata only.

        READY admission calls this before Vision, cache reset, prefix DMA or any
        Graph capture.  A request outside the prepared envelope therefore cannot
        partially execute an RPU subsystem before it is rejected.
        """
        if (not isinstance(images, torch.Tensor) or images.device.type != "cpu"
                or images.dim() != 3):
            raise ValueError(
                "lingbot2: obs['images'] must be a CPU "
                "[n_img,num_patch,patch_dim] tensor")
        if (img_masks is not None
                and (not isinstance(img_masks, torch.Tensor)
                     or img_masks.device.type != "cpu")):
            raise ValueError("lingbot2: img_masks must be a CPU tensor")
        if img_masks is not None and img_masks.numel() != images.shape[0]:
            raise ValueError(
                "lingbot2: img_masks must have one entry per image, got "
                f"{img_masks.numel()} for {images.shape[0]}")
        keep_img = [
            index for index in range(images.shape[0])
            if img_masks is None or bool(img_masks.reshape(-1)[index])
        ]
        if not keep_img:
            raise ValueError("lingbot2: no valid image (img_masks all False).")

        if (grid_thw is not None and isinstance(grid_thw, torch.Tensor)
                and grid_thw.device.type != "cpu"):
            raise ValueError("lingbot2: image_grid_thw must be a CPU tensor")
        if grid_thw is None:
            g = self._image_grid(images)
            normalized_grid = torch.tensor(
                [[1, g, g]] * len(keep_img), dtype=torch.long)
            merged_per_img = (g // self._spatial_merge) ** 2
        else:
            normalized_grid, merged_per_img = self._check_grid_thw(
                grid_thw, images, keep_img)

        if (not isinstance(lang_tokens, torch.Tensor)
                or lang_tokens.device.type != "cpu"):
            raise ValueError("lingbot2: lang_tokens must be a CPU tensor")
        flat_lang = lang_tokens.reshape(-1)
        if lang_masks is not None:
            if (not isinstance(lang_masks, torch.Tensor)
                    or lang_masks.device.type != "cpu"
                    or lang_masks.numel() != flat_lang.numel()):
                raise ValueError(
                    "lingbot2: lang_masks must match lang_tokens element-for-element")
            flat_lang = flat_lang[lang_masks.reshape(-1).bool()]
        if flat_lang.numel() <= 0:
            raise ValueError("lingbot2: the filtered instruction must contain a token")
        lang_ids = flat_lang.reshape(1, -1).to(
            device="cpu", dtype=torch.long).contiguous()
        layout = build_prefix_layout(
            len(keep_img), int(merged_per_img), lang_ids,
            self._align_cfg, self._ids,
        )
        return {
            "real_len": int(layout.S),
            "language_len": int(lang_ids.size(1)),
            "fixed_real_rows": int(layout.S - lang_ids.size(1)),
            "keep_img": tuple(int(index) for index in keep_img),
            "grid_key": tuple(
                int(value) for value in normalized_grid.reshape(-1).tolist()),
            "merged_per_img": int(merged_per_img),
            "image_shape": tuple(int(value) for value in images.shape),
        }

    def _denoise_mode(self):
        return {
            "unroll": bool(self._denoise_unroll),
            "expert_replay": bool(self._expert_replay),
            "multiview_spm_z2": bool(self._multiview_spm_z2),
        }

    @property
    def graphs_ready(self):
        return bool(self._graphs_ready)

    @property
    def graph_profile(self):
        profile = self._prepared_graph_profile
        if profile is None or not self._graphs_ready:
            return None
        result = {
            key: value for key, value in profile.items()
            if key not in {"real_to_execution"}
        }
        result["execution_profiles"] = tuple(
            dict(item) for item in profile["execution_profiles"])
        result["graph_counts"] = dict(profile.get("graph_counts", {}))
        return result

    def _validate_prepared_request(self, metadata, *, num_steps):
        profile = self._prepared_graph_profile
        if profile is None:
            return
        if int(num_steps) != profile["num_steps"]:
            raise RuntimeError(
                f"LingBot2 num_steps={num_steps} differs from prepared "
                f"num_steps={profile['num_steps']}. No online graph BUILD was attempted.")
        if self._denoise_mode() != profile["denoise_mode"]:
            raise RuntimeError(
                "LingBot2 execution mode changed after graph preparation: "
                f"current={self._denoise_mode()}, prepared={profile['denoise_mode']}. "
                "No online graph BUILD was attempted.")
        if metadata["grid_key"] != profile["grid_key"]:
            raise RuntimeError(
                "LingBot2 image geometry differs from the prepared Vision profile: "
                f"grid={metadata['grid_key']}, prepared={profile['grid_key']}")
        if (metadata["keep_img"] != profile["keep_img"]
                or metadata["image_shape"] != profile["image_shape"]
                or metadata["fixed_real_rows"] != profile["fixed_real_rows"]):
            raise RuntimeError(
                "LingBot2 image/prefix structure differs from the prepared profile; "
                "no RPU graph was touched.")
        real_len = metadata["real_len"]
        lo, hi = profile["prefix_length_range"]
        if not lo <= real_len <= hi:
            raise RuntimeError(
                "LingBot2 REAL prefix is outside the prepared range: "
                f"real_len={real_len}, prepared=[{lo}, {hi}]. No online graph BUILD "
                "was attempted.")
        execution_key = profile["real_to_execution"].get(real_len)
        if execution_key not in profile["execution_keys"]:
            raise RuntimeError(
                "LingBot2 request maps to an unprepared execution profile: "
                f"real_len={real_len}, execution={execution_key}")

    @staticmethod
    def _filtered_language_tokens(obs):
        tokens = obs["lang_tokens"].reshape(-1).to(
            device="cpu", dtype=torch.long)
        masks = obs.get("lang_masks")
        if masks is not None:
            tokens = tokens[masks.reshape(-1).bool().cpu()]
        return tokens.contiguous()

    def _profile_probe_obs(self, obs, metadata, real_len, *, variant):
        """Create one shape/content probe without changing image/state/noise."""
        lang_len = int(real_len) - int(metadata["fixed_real_rows"])
        if lang_len <= 0:
            raise ValueError(
                "prepared REAL range leaves no instruction tokens: "
                f"real_len={real_len}, fixed_rows={metadata['fixed_real_rows']}")
        source = self._filtered_language_tokens(obs)
        if source.numel() <= 0:
            raise ValueError("prepare_graphs requires a non-empty representative prompt")
        repeats = (lang_len + source.numel() - 1) // source.numel()
        tokens = source.repeat(repeats)[:lang_len].clone()
        if variant:
            vocab_size = int(getattr(self._cfg.text_config, "vocab_size", 0))
            if vocab_size <= 1:
                raise RuntimeError("lingbot2 text vocab_size is unavailable")
            tokens[0] = (int(tokens[0]) + 1) % vocab_size
        probe = dict(obs)
        probe["lang_tokens"] = tokens.reshape(1, -1)
        probe["lang_masks"] = torch.ones(
            (1, lang_len), dtype=torch.bool)
        return probe

    @torch.no_grad()
    def prepare_graphs(
        self, obs, *, prefix_length_range, num_steps=NUM_STEPS
    ):
        """Prebuild a finite REAL-prefix envelope and enter lookup-only READY.

        The inclusive range is measured after filtering tokenizer padding and
        before the adapter appends execution-tail padding.  Preparation follows
        Wall-OSS' lifecycle: CPU admission/capacity checks, real-path warmup,
        per-bucket A/B/A content+length probes, freeze, then one real READY replay.
        """
        if self._prepared_graph_profile is not None:
            raise RuntimeError("LingBot2 graphs are already prepared and frozen")
        if int(num_steps) != NUM_STEPS:
            raise ValueError(
                f"lingbot2 denoise is built for num_steps={NUM_STEPS}; got {num_steps}")
        if self._vision_direct_prefix:
            raise RuntimeError(
                "LingBot2 direct-prefix owns one Vision Graph-baked scatter target "
                "and is not a multi-prompt READY path. Disable "
                "RPU_LINGBOT2_VISION_DIRECT_PREFIX or use multiview SPM Z2.")
        if not self._vlm_replay or not self._expert_replay:
            raise RuntimeError(
                "LingBot2 prepare_graphs requires VLM_REPLAY=1 and EXPERT_REPLAY=1; "
                "otherwise infer would replace a frozen cache.")

        images = obs["images"]
        metadata = self._request_prefix_geometry(
            images, obs.get("img_masks"), obs["lang_tokens"],
            obs.get("lang_masks"), obs.get("image_grid_thw"),
        )
        profiles, real_to_execution = self._prefix_execution_profiles(
            prefix_length_range)
        lo = profiles[0]["real_min"]
        hi = profiles[-1]["real_max"]
        if not lo <= metadata["real_len"] <= hi:
            raise ValueError(
                "prepare_graphs representative is outside prefix_length_range: "
                f"real_len={metadata['real_len']}, range=[{lo}, {hi}]")
        if lo <= metadata["fixed_real_rows"]:
            raise ValueError(
                "prefix_length_range includes an empty instruction: "
                f"min={lo}, fixed_rows={metadata['fixed_real_rows']}")

        denoise_cache = self._dgc if self._denoise_unroll else self._egc
        if denoise_cache is None:
            raise RuntimeError("LingBot2 active denoise GraphCache is missing")
        execution_keys = frozenset(real_to_execution.values())
        capacity = denoise_cache.max_entries()
        if capacity and len(profiles) > capacity:
            raise ValueError(
                "prefix_length_range creates more denoise execution profiles "
                f"than GraphCache capacity: profiles={len(profiles)}, "
                f"capacity={capacity}")
        if not self._multiview_spm_z2:
            vlm_capacity = self._vgc.max_entries()
            if vlm_capacity and len(profiles) > vlm_capacity:
                raise ValueError(
                    "prefix_length_range creates more VLM execution profiles "
                    f"than GraphCache capacity: profiles={len(profiles)}, "
                    f"capacity={vlm_capacity}")

        profile = {
            "prefix_length_range": (lo, hi),
            "execution_keys": execution_keys,
            "execution_profiles": tuple(dict(item) for item in profiles),
            "real_to_execution": dict(real_to_execution),
            "grid_key": metadata["grid_key"],
            "keep_img": metadata["keep_img"],
            "image_shape": metadata["image_shape"],
            "fixed_real_rows": metadata["fixed_real_rows"],
            "num_steps": int(num_steps),
            "denoise_mode": self._denoise_mode(),
            "policy_version": PREFIX_POLICY_VERSION,
        }
        caches = [("denoise", denoise_cache)]
        if not self._multiview_spm_z2:
            caches = [
                ("vision", self._visual._rpu_vision_graph_cache),
                ("prefill", self._vgc),
                *caches,
            ]
        for _name, cache in caches:
            cache.begin_warmup()
            cache.clear()
        self._ready_rope_p0 = None
        self._graphs_ready = False
        self._prepared_graph_profile = profile
        try:
            warm_out = self.get_action(dict(obs), num_steps=num_steps).clone()
            z2_warm_stats = None
            if self._multiview_spm_z2:
                z2 = self._z2_state
                z2_warm_stats = list(
                    torch.ops.rpu.lingbot2_multiview_spm_z2_stats(
                        z2["vision_handle"], z2["text_handle"],
                        z2["expert_handle"], z2["plan_hash"],
                    )
                )
            vision_count = (
                0 if self._multiview_spm_z2
                else self._visual._rpu_vision_graph_cache.size()
            )
            if not self._multiview_spm_z2 and vision_count < 1:
                raise RuntimeError("LingBot2 Vision prewarm produced no graph entry")

            for item in profiles:
                real_a = item["real_min"]
                real_b = item["real_max"]
                probe_a = self._profile_probe_obs(
                    obs, metadata, real_a, variant=False)
                probe_b = self._profile_probe_obs(
                    obs, metadata, real_b,
                    variant=(real_b == real_a))
                if not self._multiview_spm_z2:
                    torch.ops.rpu.spm_alloc_reset_temporary()
                out_a = self.get_action(probe_a, num_steps=num_steps).clone()
                self.get_action(probe_b, num_steps=num_steps)
                out_a2 = self.get_action(probe_a, num_steps=num_steps).clone()
                if not torch.equal(out_a, out_a2):
                    max_abs = float((out_a.float() - out_a2.float()).abs().max())
                    raise RuntimeError(
                        "LingBot2 prompt A/B/A replay returned stale/different data: "
                        f"execution={item['execution_len']}, max_abs={max_abs:.7g}")

            expected = {"denoise": len(profiles)}
            if not self._multiview_spm_z2:
                expected.update({
                    "vision": vision_count,
                    "prefill": len(profiles),
                })
            actual = {name: cache.size() for name, cache in caches}
            if actual != expected:
                raise RuntimeError(
                    f"LingBot2 graph prewarm produced {actual}, expected {expected}")
            for _name, cache in caches:
                if not cache.cache_invariant_ok():
                    raise RuntimeError(
                        f"LingBot2 {_name} GraphCache invariant failed before READY")
                cache.freeze()
            ready_out = self.get_action(dict(obs), num_steps=num_steps)
            ready_counts = {name: cache.size() for name, cache in caches}
            if ready_counts != actual:
                raise RuntimeError(
                    "LingBot2 READY replay changed graph counts: "
                    f"before={actual}, after={ready_counts}")
            if not torch.equal(warm_out, ready_out):
                max_abs = float(
                    (warm_out.float() - ready_out.float()).abs().max())
                raise RuntimeError(
                    "LingBot2 READY replay differs from WARMING output after "
                    f"A/B/A probes (max_abs={max_abs:.7g})")
            if self._multiview_spm_z2:
                z2 = self._z2_state
                z2_ready_stats = list(
                    torch.ops.rpu.lingbot2_multiview_spm_z2_stats(
                        z2["vision_handle"], z2["text_handle"],
                        z2["expert_handle"], z2["plan_hash"],
                    )
                )
                if (
                    len(z2_ready_stats) < 24
                    or z2_ready_stats[0] != 3
                    or z2_ready_stats[2] != z2["plan_hash"]
                    or z2_ready_stats[9] != 1
                    or z2_warm_stats is None
                    or z2_ready_stats[10] <= z2_warm_stats[10]
                    or z2_ready_stats[12] != z2_warm_stats[12]
                    or z2_ready_stats[13] != z2_warm_stats[13]
                    or z2_ready_stats[20:24] != [1, 1, 0, 0]
                ):
                    raise RuntimeError(
                        "LingBot2 Z2 did not retain one stable BUILD followed by "
                        f"REPLAY under a parked guard: warm={z2_warm_stats}, "
                        f"ready={z2_ready_stats}")
                profile["z2_retained_build_count"] = z2_ready_stats[9]
                profile["z2_replay_count"] = z2_ready_stats[10]
            profile["graph_counts"] = dict(actual)
            profile["ready_replay_validated"] = True
            profile["example_real_prefix_len"] = metadata["real_len"]
            profile["phase"] = "READY"
            self._graphs_ready = True
        except Exception:
            for _name, cache in caches:
                cache.begin_warmup()
            self._prepared_graph_profile = None
            self._graphs_ready = False
            self._ready_rope_p0 = None
            raise
        return self.graph_profile

    def _build_prefix_direct(self, images, keep_img, lang_tokens, lang_masks,
                             grid_thw, num_patch):
        """Build/bind persistent prefix targets before the fused Vision Graph BUILD."""
        batch_cap = max(1, int(os.environ.get(
            "RPU_QWEN3VL_VISION_BATCH_CAP", "3")))
        vision_cache = self._visual._rpu_vision_kv_cache
        _validate_direct_prefix_group(
            grid_thw,
            patches_per_image=int(images.shape[1]),
            batch_cap=batch_cap,
            cache_capacity=int(vision_cache.max_seq_len),
        )

        memo_key = (
            PREFIX_POLICY_VERSION, tuple(keep_img), int(num_patch),
            tuple(lang_tokens.reshape(-1).tolist()),
            None if lang_masks is None else tuple(
                lang_masks.reshape(-1).bool().tolist()),
            tuple(grid_thw.reshape(-1).tolist()),
        )
        memo = self._pfx_memo
        old_memo_keepalive = None
        if memo is not None and memo["key"] != memo_key:
            # Both graphs baked addresses owned by the old memo.  Refuse READY
            # before mutating either cache; otherwise clear both and retain the
            # old tensors locally until the new binding has executed once.
            old_memo_keepalive = memo
            _clear_graph_caches_for_rebind((
                (self._visual._rpu_vision_graph_cache, "Vision"),
                (self._vgc, "VLM"),
            ))
            self._pos_ids_src = None
            self._pos_rpu = None
            memo = None

        if memo is None:
            if (old_memo_keepalive is None
                    and getattr(self._visual, "_rpu_vision_has_dispatched", False)):
                raise RuntimeError(
                    "lingbot2 direct-prefix must bind its fixed DMA targets before "
                    "the first Vision forward.")

            if lang_masks is not None:
                keep = lang_masks.reshape(-1).bool()
                lang_ids = lang_tokens.reshape(-1)[keep].reshape(1, -1)
            else:
                lang_ids = lang_tokens.reshape(1, -1)
            lang_ids = lang_ids.to("cpu", dtype=torch.long)

            lay = build_prefix_layout(
                len(keep_img), num_patch, lang_ids, self._align_cfg, self._ids)
            plan = self._prefix_execution_plan(lay.S)
            execution = pad_prefix_layout_for_execution(
                lay, plan.execution_len, pad_token_id=self._ids.eos)
            real_S = int(lay.S)
            S = int(execution.execution_len)

            _mm_types = getattr(self, "_rope_index_takes_mm_types", None)
            if _mm_types is None:
                import inspect
                _mm_types = "mm_token_type_ids" in inspect.signature(
                    self._vlm.get_rope_index).parameters
                self._rope_index_takes_mm_types = _mm_types
            _rope_kw = dict(
                input_ids=execution.input_ids,
                image_grid_thw=grid_thw,
                video_grid_thw=None,
                attention_mask=execution.pad_masks.long(),
            )
            if _mm_types:
                _rope_kw["mm_token_type_ids"] = execution.mm_token_type_ids
            position_ids, _delta = self._vlm.get_rope_index(**_rope_kw)
            p0 = suffix_position_start(position_ids, execution.pad_masks)

            # Build the same prompt base as the legacy path, but leave visual
            # placeholders untouched: the in-graph merger DMA overwrites them.
            embeds = self._vlm.get_input_embeddings()(
                execution.input_ids.to("rpu")).to("cpu", torch.float16)
            for seg, tok in self._align_tokens.items():
                a, b = lay.spans[seg]
                embeds[0, a:b] = tok.to(torch.float16)
            embeds = embeds.to(
                device="rpu", dtype=torch.float16).contiguous()

            runs = _contiguous_runs(
                execution.visual_pos_masks.squeeze(0))
            _validate_prefix_scatter_runs(
                runs,
                prefix_rows=S,
                expected_rows=len(keep_img) * int(num_patch),
                expected_runs=len(keep_img),
            )
            dense_ds = [
                torch.zeros((S, VLM_HID), dtype=torch.float16)
                .to("rpu").contiguous()
                for _ in range(3)
            ]
            memo = {
                "key": memo_key,
                "embeds": embeds,
                "dense": dense_ds,
                "position_ids": position_ids,
                "position_ids_rpu": position_ids.to("rpu"),
                "S": S,
                "real_S": real_S,
                "p0": p0,
                "runs": runs,
            }
            # `memo` itself strongly owns every target while the setter runs.
            # Publish it only after the setter succeeds: otherwise a retry with
            # the same key could skip binding and feed untouched placeholders.
            torch.ops.rpu.qwen3vl_vision_set_prefix_scatter(
                self._visual._rpu_vision_handle,
                int(embeds.data_ptr()),
                [int(dense.data_ptr()) for dense in dense_ds],
                [int(start) for start, _length in runs],
                [int(length) for _start, length in runs],
            )
            self._pfx_memo = memo
            # Prefix preparation used immediate RPU ops before the Vision
            # subsystem.  Release only temporary SPM allocations; persistent
            # targets above live in DDR and remain strongly owned by the memo.
            torch.ops.rpu.spm_alloc_reset_temporary()

        if len(keep_img) == images.shape[0]:
            pix = images.reshape(-1, images.shape[-1])
            if not pix.is_contiguous():
                pix = pix.contiguous()
        else:
            pix = images[keep_img].reshape(
                -1, images.shape[-1]).contiguous()
        with torch.no_grad():
            # The no-output flag skips generic pop/clone results after direct
            # prefix writes. In either mode, the prefix targets above are the
            # only outputs consumed by LingBot2.
            self._visual(
                pix,
                grid_thw=grid_thw,
                _rpu_prefix_scatter_consume_only=(
                    self._vision_direct_prefix_no_output),
            )
        return (memo["embeds"], memo["position_ids"],
                memo["position_ids_rpu"], memo["dense"], None, None,
                memo["S"], memo["real_S"], memo["p0"])

    def _fill_prefix_multiview_spm_z2(
        self, images, img_masks, lang_tokens, lang_masks, grid_thw
    ):
        """Run the exact three-camera Vision->Text physical Z2.

        All prompt/layout validation and every RPU input allocation happen before
        the native prepare.  Once prepare acquires the physical arena, its first
        forward is the immediately following RPU operation.  Later calls may
        change images and any admitted REAL prompt.  Prompt owners keep their
        addresses; bytes are refreshed in place and position/M-RoPE changes are
        explicitly re-primed outside Graph capture before retained replay.
        """
        z2 = self._z2_state
        if z2.get("poisoned", False):
            raise RuntimeError(
                "lingbot2 multiview SPM Z2 previously failed; restart the "
                "process instead of retrying a possibly poisoned lifecycle."
            )
        current_thread = threading.get_ident()
        owner_thread = z2.get("owner_thread")
        if owner_thread is not None and owner_thread != current_thread:
            raise RuntimeError(
                "lingbot2 multiview SPM Z2 is thread-affine; get_action and "
                "close must run on the thread that performed the first call."
            )

        if not isinstance(images, torch.Tensor) or images.device.type != "cpu":
            raise ValueError(
                "lingbot2 multiview SPM Z2 requires CPU image patches."
            )
        if images.dim() != 3 or images.size(0) != 3 or images.size(1) != 256:
            raise ValueError(
                "lingbot2 multiview SPM Z2 requires images [3,256,patch_dim], "
                f"got {tuple(images.shape)}."
            )
        if img_masks is not None:
            if (
                not isinstance(img_masks, torch.Tensor)
                or img_masks.device.type != "cpu"
                or img_masks.numel() != 3
                or not bool(img_masks.reshape(-1).bool().all())
            ):
                raise ValueError(
                    "lingbot2 multiview SPM Z2 requires all three cameras valid."
                )
        if (
            not isinstance(grid_thw, torch.Tensor)
            or grid_thw.device.type != "cpu"
            or tuple(grid_thw.shape) != (3, 3)
            or grid_thw.tolist() != [[1, 16, 16]] * 3
        ):
            raise ValueError(
                "lingbot2 multiview SPM Z2 requires image_grid_thw="
                "[[1,16,16]]*3."
            )
        if (
            not isinstance(lang_tokens, torch.Tensor)
            or lang_tokens.device.type != "cpu"
        ):
            raise ValueError(
                "lingbot2 multiview SPM Z2 requires CPU language tokens."
            )
        if lang_masks is not None:
            if (
                not isinstance(lang_masks, torch.Tensor)
                or lang_masks.device.type != "cpu"
                or lang_masks.numel() != lang_tokens.numel()
            ):
                raise ValueError(
                    "lingbot2 multiview SPM Z2 language mask must be a CPU "
                    "tensor matching lang_tokens."
                )
            keep_lang = lang_masks.reshape(-1).bool()
            lang_ids = lang_tokens.reshape(-1)[keep_lang].reshape(1, -1)
        else:
            lang_ids = lang_tokens.reshape(1, -1)
        lang_ids = lang_ids.to(device="cpu", dtype=torch.long).contiguous()

        # Z2 keeps its physical Text execution fixed at P225, but REAL prompt
        # rows are variable and end before an adapter-owned masked tail.
        layout = build_prefix_layout(
            3, 64, lang_ids, self._align_cfg, self._ids)
        plan = self._prefix_execution_plan(layout.S)
        execution = pad_prefix_layout_for_execution(
            layout, plan.execution_len, pad_token_id=self._ids.eos)
        real_S = int(layout.S)
        S = int(execution.execution_len)
        runs = _contiguous_runs(execution.visual_pos_masks.squeeze(0))
        if S != MULTIVIEW_SPM_Z2_EXECUTION_LEN or runs != [
            (1, 64), (67, 64), (133, 64)
        ]:
            raise RuntimeError(
                "lingbot2 multiview SPM Z2 execution geometry drifted: "
                f"REAL={real_S}, P={S}, visual_runs={runs}.")

        rope_takes_mm_types = getattr(
            self, "_rope_index_takes_mm_types", None)
        if rope_takes_mm_types is None:
            import inspect
            rope_takes_mm_types = "mm_token_type_ids" in inspect.signature(
                self._vlm.get_rope_index).parameters
            self._rope_index_takes_mm_types = rope_takes_mm_types
        rope_kwargs = {
            "input_ids": execution.input_ids,
            "image_grid_thw": grid_thw,
            "video_grid_thw": None,
            "attention_mask": execution.pad_masks.long(),
        }
        if rope_takes_mm_types:
            rope_kwargs["mm_token_type_ids"] = execution.mm_token_type_ids
        position_ids_3d, _rope_delta = self._vlm.get_rope_index(**rope_kwargs)
        if tuple(position_ids_3d.shape) != (3, 1, S):
            raise RuntimeError(
                "lingbot2 multiview SPM Z2 M-RoPE shape drifted: "
                f"{tuple(position_ids_3d.shape)}.")
        p0 = int(suffix_position_start(
            position_ids_3d, execution.pad_masks))
        position_ids_cpu = (
            position_ids_3d[:, 0, :].transpose(0, 1)
            .to(device="cpu", dtype=torch.int32).contiguous()
        )
        mrope_head_dim, mrope_section = resolve_mrope_geometry(
            self._cfg.text_config)
        rope_cos_il_cpu, rope_sin_il_cpu = build_interleaved_mrope_cos_sin(
            position_ids_cpu.to(torch.int64),
            head_dim=mrope_head_dim,
            rope_theta=self._rope_theta,
            mrope_section=mrope_section,
        )
        prompt_key = (
            PREFIX_POLICY_VERSION,
            tuple(execution.input_ids.reshape(-1).tolist()),
            tuple(execution.pad_masks.reshape(-1).bool().tolist()),
            tuple(position_ids_3d.reshape(-1).tolist()),
            real_S, p0,
        )

        prompt_metadata = self._z2_prompt_metadata
        refresh_text_inputs = False
        if self._z2_prompt_key is None:
            # Prompt bytes are assembled entirely on CPU.  This matters after
            # the physical guard is retained: no unrelated device embedding op
            # is allowed between composite replays.
            text_hidden_cpu = F.embedding(
                execution.input_ids, self._z2_embedding_cpu)
            text_hidden_cpu.masked_fill_(
                execution.visual_pos_masks.unsqueeze(-1), 0)
            for segment, token in self._align_tokens.items():
                begin, end = layout.spans[segment]
                text_hidden_cpu[0, begin:end] = token.to(torch.float16)
            text_hidden_cpu = text_hidden_cpu.contiguous()
            # A caller may wrap inference in torch.inference_mode().  Native Z2
            # version-gates the position/RoPE owners and mutates the text
            # carrier, so create all four as normal tensors.
            with torch.inference_mode(False):
                self._z2_text_hidden = text_hidden_cpu.to(
                    device="rpu", dtype=torch.float16).contiguous()
                self._z2_position_ids = position_ids_cpu.to(
                    device="rpu", dtype=torch.int32).contiguous()
                self._z2_rope_cos_il = rope_cos_il_cpu.to(
                    device="rpu", dtype=torch.float16).contiguous()
                self._z2_rope_sin_il = rope_sin_il_cpu.to(
                    device="rpu", dtype=torch.float16).contiguous()
            if any(torch.is_inference(tensor) for tensor in (
                self._z2_text_hidden,
                self._z2_position_ids,
                self._z2_rope_cos_il,
                self._z2_rope_sin_il,
            )):
                raise RuntimeError(
                    "lingbot2 multiview SPM Z2 M-RoPE owners unexpectedly lack "
                    "version counters.")
        else:
            if (
                not isinstance(prompt_metadata, dict)
                or self._z2_text_hidden is None
                or self._z2_position_ids is None
                or self._z2_rope_cos_il is None
                or self._z2_rope_sin_il is None
            ):
                raise RuntimeError(
                    "lingbot2 multiview SPM Z2 stable prompt owners were dropped.")
            if prompt_key != self._z2_prompt_key:
                try:
                    text_hidden_cpu = F.embedding(
                        execution.input_ids, self._z2_embedding_cpu)
                    text_hidden_cpu.masked_fill_(
                        execution.visual_pos_masks.unsqueeze(-1), 0)
                    for segment, token in self._align_tokens.items():
                        begin, end = layout.spans[segment]
                        text_hidden_cpu[0, begin:end] = token.to(torch.float16)
                    self._z2_text_hidden.copy_(text_hidden_cpu)
                    position_changed = not torch.equal(
                        prompt_metadata["position_ids_cpu"],
                        position_ids_cpu)
                    if position_changed:
                        with torch.inference_mode(False):
                            self._z2_position_ids.copy_(position_ids_cpu)
                            self._z2_rope_cos_il.copy_(rope_cos_il_cpu)
                            self._z2_rope_sin_il.copy_(rope_sin_il_cpu)
                        if not z2.get("prepared", False):
                            raise RuntimeError(
                                "lingbot2 multiview SPM Z2 prompt changed before "
                                "the retained graph was built")
                        refresh_text_inputs = True
                except BaseException:
                    z2["poisoned"] = True
                    raise

        if prompt_key != self._z2_prompt_key:
            self._z2_prompt_metadata = {
                "lang_ids": lang_ids.clone(),
                "position_ids_cpu": position_ids_cpu.clone(),
                "rope_cos_il_cpu": rope_cos_il_cpu.clone(),
                "rope_sin_il_cpu": rope_sin_il_cpu.clone(),
                "S": S,
                "real_S": real_S,
                "p0": p0,
            }
            self._z2_prompt_key = prompt_key

        packed_patches = images.to(dtype=torch.float16).contiguous().view(
            3 * 256, images.size(2)
        )
        # This is preparation-only packing: one patch Linear/H2D/position add
        # owns three adjacent RPU inputs, while native Vision remains three
        # sequential N=256 occurrences.  W8 Z2 explicitly selects the staged
        # device Linear; shared helper callers retain their CPU default.  Each
        # narrow view keeps the storage alive through the synchronous forward.
        vision_input_owner = (
            _prepare_qwen3vl_vision_input_for_spm_pipeline(
                self._visual,
                packed_patches,
                grid_thw,
                device_patch_embed=bool(getattr(
                    self._visual, "_rpu_patch_embed_on_device", False
                )),
            )
        )
        vision_inputs = [
            vision_input_owner.narrow(0, index, 1)
            for index in range(3)
        ]
        vision_cache = getattr(self._visual, "_rpu_vision_kv_cache", None)
        if vision_cache is None:
            raise RuntimeError(
                "lingbot2 multiview SPM Z2 requires the installed Vision cache."
            )
        vision_cache.reset_to_position(0)
        self._cache.reset_to_position(0)
        z2["owner_thread"] = current_thread

        if not z2.get("prepared", False):
            # This reset and all allocations above precede prepare.  After the
            # physical lease is acquired, forward is intentionally the next RPU
            # operation; even stats are deferred until the caller regains control.
            torch.ops.rpu.spm_alloc_reset_temporary()
            z2["prepare_attempted"] = True
            try:
                z2["plan_hash"] = int(
                    torch.ops.rpu.lingbot2_multiview_spm_z2_prepare(
                        z2["vision_handle"], z2["text_handle"],
                        z2["expert_handle"], self._z2_position_ids,
                        self._z2_rope_cos_il, self._z2_rope_sin_il,
                    )
                )
                z2["prepared"] = True
            except BaseException:
                z2["poisoned"] = True
                raise

        if refresh_text_inputs:
            try:
                # Re-prime last: the retained Text forward is the immediately
                # following RPU operation, matching the physical-owner contract.
                torch.ops.rpu.lingbot2_multiview_spm_z2_refresh_text_inputs(
                    z2["vision_handle"], z2["text_handle"],
                    z2["expert_handle"], self._z2_position_ids,
                    self._z2_rope_cos_il, self._z2_rope_sin_il,
                    z2["plan_hash"],
                )
            except BaseException:
                z2["poisoned"] = True
                raise

        try:
            torch.ops.rpu.lingbot2_multiview_spm_z2_forward(
                z2["vision_handle"], z2["text_handle"],
                z2["expert_handle"], vision_inputs,
                vision_cache.k_caches, vision_cache.v_caches,
                self._z2_text_hidden, self._cache.k_caches,
                self._cache.v_caches, self._z2_position_ids,
                self._z2_rope_cos_il, self._z2_rope_sin_il,
                z2["plan_hash"],
            )
        except BaseException:
            z2["poisoned"] = True
            raise
        # The native Text execution wrote [0,225), but it deliberately bypasses
        # the Python HF wrapper that normally advances this bookkeeping field.
        self._cache.reset_to_position(S)
        return S, real_S, p0

    def _build_prefix(self, images, img_masks, lang_tokens, lang_masks, grid_thw=None):
        """Run the RPU ViT and assemble the prefix EXACTLY as the official embed_prefix does.

        Layout (authoritative YAML): [<vision_start> <image>*P <vision_end>]*n_img ++ lang
                                     ++ <eos>*8 (current_depth) ++ <eos>*8 (future_depth)

        Returns (inputs_embeds [1,S,2560] RPU fp16, position_ids [3,1,S] CPU, dense_ds, S, p0).
        """
        keep_img = [i for i in range(images.shape[0])
                    if img_masks is None or bool(img_masks.reshape(-1)[i])]
        if not keep_img:
            raise ValueError("lingbot2: no valid image (img_masks all False).")

        # rope_grid_thw carries one row per valid image; get_rope_index
        # consumes one grid per contiguous <image> run.
        if grid_thw is None:
            # Infer a square grid from the patch count when no grid is supplied.
            g = self._image_grid(images)
            grid_thw = torch.tensor([[1, g, g]] * len(keep_img), dtype=torch.long)
            merged_per_img = (g // self._spatial_merge) ** 2
        else:
            # Use the supplied real (t,h,w) grid, which may be non-square.
            grid_thw, merged_per_img = self._check_grid_thw(grid_thw, images, keep_img)
        if self._vision_direct_prefix:
            return self._build_prefix_direct(
                images, keep_img, lang_tokens, lang_masks, grid_thw,
                merged_per_img)
        if self._prefix_opt and len(keep_img) == images.shape[0]:
            # Every image kept ⇒ `images[keep_img]` is advanced indexing that copies the whole
            # [n_img, num_patch, patch_dim] fp32 tensor only to hand the ViT the
            # same values. Reshape is a view when the source is contiguous; the
            # ViT only reads it.
            pix = images.reshape(-1, images.shape[-1])
            if not pix.is_contiguous():
                pix = pix.contiguous()
        else:
            pix = images[keep_img].reshape(-1, images.shape[-1]).contiguous()
        with torch.no_grad():
            vout = self._visual(pix, grid_thw=grid_thw)
        pooler = vout.pooler_output                                  # [n_img*P, 2560]
        # fp16, mirroring the `pooler.to("cpu", torch.float16)` cast below: the ViT is fed
        # fp32 pixels,
        # so its outputs come back fp32. scatter_visual_embeds_to_dense inherits the dtype
        # of what it is handed, and causal_decoder_forward then rejects the fp32 dense
        # deepstack tensors with "expected scalar type Half but found Float".
        deepstack_features = [d.to(torch.float16) for d in vout.deepstack_features]

        num_patch = merged_per_img
        if pooler.shape[0] != len(keep_img) * num_patch:
            raise ValueError(f"lingbot2: ViT returned {pooler.shape[0]} tokens, expected "
                             f"{len(keep_img)} x {num_patch}")

        # ---- prompt-constant path ------------------------------------------------------------
        # RPU-resident prefix buffers retain stable replay addresses. Only
        # contiguous visual rows are overwritten for each frame.
        #
        # The key covers every input the cached bytes depend on. img_masks enters through
        # len(keep_img) AND through which images survive — keep_img itself is in the key, not just
        # its length, so dropping camera 0 vs camera 2 cannot collide.
        memo_key = None
        if self._prefix_opt:
            memo_key = (PREFIX_POLICY_VERSION, tuple(keep_img), int(num_patch),
                        tuple(lang_tokens.reshape(-1).tolist()),
                        None if lang_masks is None else tuple(lang_masks.reshape(-1).bool().tolist()),
                        tuple(grid_thw.reshape(-1).tolist()))
            memo = self._pfx_memo
            if memo is not None and memo["key"] == memo_key:
                _scatter_rows_(memo["embeds"][0], pooler.to(torch.float16),
                               memo["runs"], memo["offs"])
                for dense, feat in zip(memo["dense"], deepstack_features):
                    _scatter_rows_(dense, feat, memo["runs"], memo["offs"])
                return (memo["embeds"], memo["position_ids"],
                        memo["position_ids_rpu"], memo["dense"],
                        memo["rope_cos_il"], memo["rope_sin_il"],
                        memo["S"], memo["real_S"], memo["p0"])

        if lang_masks is not None:
            keep = lang_masks.reshape(-1).bool()
            lang_ids = lang_tokens.reshape(-1)[keep].reshape(1, -1)
        else:
            lang_ids = lang_tokens.reshape(1, -1)
        lang_ids = lang_ids.to("cpu", dtype=torch.long)

        lay = build_prefix_layout(
            len(keep_img), num_patch, lang_ids, self._align_cfg, self._ids
        )
        plan = self._prefix_execution_plan(lay.S)
        execution = pad_prefix_layout_for_execution(
            lay, plan.execution_len, pad_token_id=self._ids.eos
        )
        real_S = int(lay.S)
        S = int(execution.execution_len)

        # ---- M-RoPE position IDs ----
        # transformers 5.x takes an explicit `mm_token_type_ids` to mark the vision spans;
        # 4.57.x has no such parameter and derives the same spans by scanning `input_ids`
        # for the image/vision special tokens. Both produce the same position_ids for our
        # prefix, so pass the argument only where it exists. Probe via the
        # signature rather than try/except TypeError, which would also swallow a genuine
        # argument error. Cached per instance — this is on the get_action path.
        _mm_types = getattr(self, "_rope_index_takes_mm_types", None)
        if _mm_types is None:
            import inspect
            _mm_types = "mm_token_type_ids" in inspect.signature(
                self._vlm.get_rope_index).parameters
            self._rope_index_takes_mm_types = _mm_types
        _rope_kw = dict(
            input_ids=execution.input_ids,
            image_grid_thw=grid_thw,
            video_grid_thw=None,
            attention_mask=execution.pad_masks.long(),
        )
        if _mm_types:
            _rope_kw["mm_token_type_ids"] = execution.mm_token_type_ids
        position_ids, _delta = self._vlm.get_rope_index(**_rope_kw)
        p0 = suffix_position_start(position_ids, execution.pad_masks)
        (
            position_ids_for_decoder,
            rope_cos_il_cpu,
            rope_sin_il_cpu,
        ) = _build_prefill_mrope_inputs(
            position_ids,
            head_dim=self._mrope_head_dim,
            rope_theta=self._rope_theta,
            mrope_section=self._mrope_section,
        )

        # ---- embeds: token embeds, then overwrite the placeholder spans ----
        # embed_tokens.weight lives on RPU (text_model.to("rpu") in build_lingbot_vla_v2), so the
        # lookup itself must run there — torch.embedding(rpu_weight, cpu_ids) raises
        # "rpu_to_cpu_zerocopy: input must be on RPU device". Pull the RESULT straight back to
        # CPU so the placeholder assembly below (masked_scatter / align-token writes) stays
        # unchanged; the single final transfer below moves it to RPU.
        embeds_cpu = self._vlm.get_input_embeddings()(
            execution.input_ids.to("rpu")
        ).to("cpu", torch.float16)
        # <image> placeholders  -> ViT patch embeddings. <vision_start>/<vision_end> KEEP their
        # real token embeddings; special tokens use the language-token table.
        embeds_cpu = embeds_cpu.masked_scatter(
            execution.visual_pos_masks.unsqueeze(-1).expand_as(embeds_cpu),
            pooler.to("cpu", torch.float16),
        )
        # <eos> align placeholders -> the align query tokens (their ids are fake; only the
        # embeddings are real).
        for seg, tok in self._align_tokens.items():
            a, b = lay.spans[seg]
            embeds_cpu[0, a:b] = tok.to(torch.float16)

        if self._prefix_opt:
            # Memo MISS (first call, or the prompt changed): build the dense deepstack tensors the
            # narrow()/copy_() way as well, so the buffers that get cached are the ones the warm
            # path will keep writing into, and record where the visual rows live.
            runs = _contiguous_runs(execution.visual_pos_masks.squeeze(0))
            offs, acc = [], 0
            for _st, ln in runs:
                offs.append(acc)
                acc += ln
            if acc != pooler.shape[0]:
                raise AssertionError(
                    f"lingbot2: visual mask covers {acc} rows but the ViT returned "
                    f"{pooler.shape[0]} tokens — the prefix layout and the ViT disagree.")
            execution_key = plan.key
            memo = self._pfx_memos.get(execution_key)
            if memo is None:
                capacity = self._vgc.max_entries()
                if capacity and len(self._pfx_memos) >= capacity:
                    raise RuntimeError(
                        "lingbot2 prefix execution-profile capacity exhausted: "
                        f"profiles={len(self._pfx_memos)}, capacity={capacity}. "
                        "Prepare a finite REAL-prefix envelope or construct a "
                        "fresh policy with a narrower prompt range."
                    )
                # These are mutable stable-address owners.  A caller may wrap
                # inference in torch.inference_mode(); create normal tensors so
                # prompt refresh remains legal and position_ids_rpu exposes the
                # version epoch consumed by CausalDecoderModel's keepalive gate.
                with torch.inference_mode(False):
                    embeds = embeds_cpu.to(
                        device="rpu", dtype=torch.float16
                    ).contiguous()
                    dense_ds = []
                    for feat in deepstack_features:
                        # Allocate zeros on CPU, then transfer; direct RPU zero
                        # allocation is unsupported here.
                        dense = torch.zeros(
                            (S, VLM_HID), dtype=torch.float16
                        ).to("rpu").contiguous()
                        _scatter_rows_(dense, feat, runs, offs)
                        dense_ds.append(dense)
                    stable_position_ids = position_ids.clone()
                    stable_position_ids_rpu = position_ids_for_decoder.to(
                        "rpu").contiguous()
                    stable_rope_cos_il = rope_cos_il_cpu.to(
                        "rpu").contiguous()
                    stable_rope_sin_il = rope_sin_il_cpu.to(
                        "rpu").contiguous()
                if any(torch.is_inference(tensor) for tensor in (
                    stable_position_ids_rpu,
                    stable_rope_cos_il,
                    stable_rope_sin_il,
                )):
                    raise RuntimeError(
                        "lingbot2 generic position/M-RoPE owner unexpectedly "
                        "lacks a content version counter")
                memo = {
                    "key": memo_key,
                    "embeds": embeds,
                    "dense": dense_ds,
                    "position_ids": stable_position_ids,
                    "position_ids_rpu": stable_position_ids_rpu,
                    "rope_cos_il": stable_rope_cos_il,
                    "rope_sin_il": stable_rope_sin_il,
                    "S": S,
                    "real_S": real_S,
                    "p0": p0,
                    "runs": runs,
                    "offs": offs,
                }
                self._pfx_memos[execution_key] = memo
            else:
                # Same execution signature, different prompt identity.  Every
                # fixed-DMA owner keeps its address; only bytes and semantic
                # metadata change.  Clear dense carriers before scattering so
                # a changed visual-run layout cannot leave stale rows behind.
                _refresh_prefix_memo_(
                    memo,
                    memo_key=memo_key,
                    embeds=embeds_cpu,
                    position_ids=position_ids,
                    rope_cos_il=rope_cos_il_cpu,
                    rope_sin_il=rope_sin_il_cpu,
                    deepstack_features=deepstack_features,
                    runs=runs,
                    offs=offs,
                    p0=p0,
                )
                memo["real_S"] = real_S
            self._pfx_memo = memo
            embeds = memo["embeds"]
            position_ids = memo["position_ids"]
            position_ids_rpu = memo["position_ids_rpu"]
            dense_ds = memo["dense"]
            rope_cos_il = memo["rope_cos_il"]
            rope_sin_il = memo["rope_sin_il"]
        else:
            embeds = embeds_cpu.to(
                device="rpu", dtype=torch.float16
            ).contiguous()
            position_ids_rpu = position_ids_for_decoder.to("rpu")
            rope_cos_il = rope_cos_il_cpu.to("rpu").contiguous()
            rope_sin_il = rope_sin_il_cpu.to("rpu").contiguous()
            dense_ds = scatter_visual_embeds_to_dense(
                deepstack_features,
                execution.visual_pos_masks.squeeze(0), S, VLM_HID,
            )
        return (
            embeds, position_ids, position_ids_rpu, dense_ds,
            rope_cos_il, rope_sin_il,
            S, real_S, p0,
        )

    def _vlm_fill(self, embeds, position_ids, position_ids_rpu, dense_ds,
                  rope_cos_il, rope_sin_il, S):
        """ONE VLM prefill writes the prefix KV into the shared cache [0, S)."""
        self._cache.reset_to_position(0)
        if position_ids_rpu is None:
            # Legacy non-memo caller.  Memoized buckets pass their own stable
            # RPU owner explicitly so A(length-x)->B(length-y)->A(length-x)
            # returns to the address originally baked for x.
            if self._pos_rpu is None or position_ids is not self._pos_ids_src:
                self._pos_ids_src = position_ids
                self._pos_rpu = position_ids.to("rpu")
            position_ids_rpu = self._pos_rpu
        # S rides in `shapes`; position 0 in dyn_dims (the prefill always starts at 0).
        sig = rpu_backend.graph.GraphSignature(
            op_id="lingbot2_vlm_prefill",
            shapes=[S, VLM_HID, self._chunk_size],
            dyn_dims=[VLM_LAYERS, S, 0, PREFIX_POLICY_VERSION],
            dtypes=[torch.float16],
        )
        with self._vgc.capture(sig):
            _run_causal_decoder_forward(
                self._text, self._text._rpu_decoder_handle,
                input_ids=None, inputs_embeds=embeds,
                attention_mask=None, position_ids=position_ids_rpu,
                past_key_values=self._cache, use_cache=True, return_dict=True,
                deepstack_dense_visual_embeds=dense_ds,
                rope_cos_il=rope_cos_il, rope_sin_il=rope_sin_il)
        # _run_causal_decoder_forward advances the cache position by seq_len itself.

    def _prepare_ready_suffix_rope(self, p0):
        """Refresh a stable expert RoPE window and return the graph offset.

        Outside a prepared envelope the existing full-table + semantic ``p0``
        path remains untouched.  During WARMING/READY, rows [0,51) of the
        expert-owned tables are refreshed in place, so the graph always records
        offset zero while different REAL prompt lengths retain their exact RoPE
        positions through mutable table *content*.
        """
        p0 = int(p0)
        if self._prepared_graph_profile is None:
            return p0
        if self._ready_rope_p0 == p0:
            return 0
        window = self._ready_rope_cpu.get(p0)
        if window is None:
            window = build_suffix_rope_window_cpu(
                p0,
                seq_len=N_ACTION + 1,
                head_dim=HD,
                rope_theta=self._rope_theta,
            )
            self._ready_rope_cpu[p0] = window
        cos_cpu, sin_cpu = window
        if (self._expert_rope_cos.size(0) < N_ACTION + 1
                or self._expert_rope_sin.size(0) < N_ACTION + 1
                or self._expert_rope_cos.size(1) != HD // 2
                or self._expert_rope_sin.size(1) != HD // 2):
            raise RuntimeError("lingbot2 expert RoPE owner geometry drifted")
        self._expert_rope_cos.narrow(0, 0, N_ACTION + 1).copy_(cos_cpu)
        self._expert_rope_sin.narrow(0, 0, N_ACTION + 1).copy_(sin_cpu)
        self._ready_rope_p0 = p0
        return 0

    def _exp_run(self, suffix_r, mask_r, S, p0, sig, adarms_step=None):
        """position=S drives the KV-insert offset; cos_sin_offset=p0 drives the RoPE base."""
        self._cache.reset_to_position(S)      # freeze prefix [0, S); insert suffix at S
        with self._egc.capture(sig):
            if self._adarms_direct_schedule:
                if adarms_step is None:
                    raise RuntimeError("lingbot2 direct AdaRMS schedule requires a step index")
                return torch.ops.rpu.lingbot_v2_moe_forward(
                    self._exp, suffix_r, self._cache.k_caches, self._cache.v_caches,
                    mask_r, S, p0, int(adarms_step))
            if self._has_cos_sin_offset:
                return torch.ops.rpu.lingbot_v2_moe_forward(
                    self._exp, suffix_r, self._cache.k_caches, self._cache.v_caches,
                    mask_r, S, p0)
            # A schema without cos_sin_offset is reachable only when p0 == S, so the
            # single `position` is unambiguous and this is exactly equivalent.
            return torch.ops.rpu.lingbot_v2_moe_forward(
                self._exp, suffix_r, self._cache.k_caches, self._cache.v_caches, mask_r, S)

    def _build_denoise_unroll(self):
        """Bind the FP16-resident action head and full indexed AdaRMS schedule once."""
        from rpu_backend.runtime.weights import tp_col_swizzle_mc_weight

        state_w_cpu = self._head_W["model.state_proj.weight"]
        state_b_cpu = self._head_W["model.state_proj.bias"]
        mo_w_cpu = self._head_W["model.action_time_mlp_out.weight"]
        mo_b_cpu = self._head_W["model.action_time_mlp_out.bias"]
        op_w_cpu = self._head_W["model.action_out_proj.weight"]
        op_b_cpu = self._head_W["model.action_out_proj.bias"]
        expected = {
            "state_w": (EXP_HID, STATE_DIM),
            "wc": (EXP_HID, ACTION_DIM),
            "mo": (EXP_HID, EXP_HID),
            "op": (ACTION_DIM, EXP_HID),
        }
        actual = {
            "state_w": tuple(state_w_cpu.shape),
            "wc": tuple(self._enc_Wf.shape),
            "mo": tuple(mo_w_cpu.shape),
            "op": tuple(op_w_cpu.shape),
        }
        if actual != expected:
            raise RuntimeError(
                "lingbot2 denoise unroll is restricted to the 55->64 action/state "
                f"profile; expected {expected}, got {actual}.")

        state_pad = (STATE_DIM + 15) // 16 * 16
        action_pad = (ACTION_DIM + 15) // 16 * 16

        def col1(weight):
            weight_h = weight.detach().to(torch.float16).contiguous()
            return tp_col_swizzle_mc_weight(weight_h, 1).to("rpu").contiguous()

        # Order is load-bearing: fp16 -> zero-pad -> single-core col-swizzle -> RPU.
        state_w = col1(F.pad(state_w_cpu.half(), (0, state_pad - STATE_DIM)))
        wc = col1(F.pad(self._enc_Wf.half(), (0, action_pad - ACTION_DIM)))
        mo = col1(mo_w_cpu)
        op = col1(F.pad(op_w_cpu.half(), (0, 0, 0, action_pad - ACTION_DIM)))
        state_b = _h(state_b_cpu)
        mo_b = _h(mo_b_cpu)
        op_b = _h(F.pad(op_b_cpu, (0, action_pad - ACTION_DIM)))
        time_all = _h(torch.stack(
            [part.reshape(-1) for part in self._enc_time_parts]))

        torch.ops.rpu.lingbot_v2_moe_set_denoise_unroll(
            self._exp, state_w, state_b, wc, mo, mo_b, op, op_b, time_all,
            *self._fold_all,
            STATE_DIM, ACTION_DIM, state_pad, action_pad, NUM_STEPS)

        self._denoise_state_pad = state_pad
        self._denoise_action_pad = action_pad
        self._denoise_state = torch.zeros(
            1, state_pad, dtype=torch.float16, device="rpu")
        self._denoise_x0 = torch.zeros(
            1, N_ACTION + 1, action_pad, dtype=torch.float16, device="rpu")
        self._denoise_state_host = torch.zeros(1, state_pad, dtype=torch.float16)
        self._denoise_x0_host = torch.zeros(
            1, N_ACTION + 1, action_pad, dtype=torch.float16)
        # Native unroll only reads these DDR inputs into SPM; it never writes
        # them back. Thus the state pad and x0 row-0/pad initialized here stay
        # zero while each call overwrites only the complete logical regions.
        self._denoise_x_final = torch.empty_like(self._denoise_x0)
        self._denoise_keep = (state_w, state_b, wc, mo, mo_b, op, op_b, time_all)

    def _denoise_unroll_run(self, state, noise, mask_r, S, real_S, p0,
                            dt, num_steps):
        """Refresh stable padded inputs, then execute all Euler steps in one graph op."""
        # Cross-device copy requires a contiguous destination. Fill the logical
        # regions in small stable CPU staging slabs, then copy each complete
        # padded slab into its stable RPU graph input.
        self._denoise_state_host[:, :STATE_DIM].copy_(state)
        self._denoise_x0_host[:, 1:, :ACTION_DIM].copy_(noise)
        self._denoise_state.copy_(self._denoise_state_host)
        self._denoise_x0.copy_(self._denoise_x0_host)
        self._cache.reset_to_position(S)
        prepared = self._prepared_graph_profile is not None
        dyn_dims = (
            [EXP_LAYERS, S, NUM_STEPS, PREFIX_POLICY_VERSION]
            if prepared else
            [EXP_LAYERS, S, real_S, p0, NUM_STEPS, PREFIX_POLICY_VERSION]
        )
        sig = rpu_backend.graph.GraphSignature(
            op_id="lingbot2_denoise_unroll", shapes=[N_ACTION + 1, EXP_HID],
            dyn_dims=dyn_dims,
            dtypes=[torch.float16])
        with self._dgc.capture(sig):
            torch.ops.rpu.lingbot_v2_moe_denoise_unroll_forward(
                self._exp, self._denoise_x0, self._denoise_state,
                self._cache.k_caches, self._cache.v_caches, mask_r,
                self._denoise_x_final, dt, S, p0, num_steps)
        return self._denoise_x_final[:, 1:, :ACTION_DIM].to("cpu").float()

    # ---------------------------------------------------------------- action
    @torch.no_grad()
    def get_action(self, obs, *, num_steps=NUM_STEPS):
        if getattr(self, "_closed", False):
            raise RuntimeError("LingBot-VLA-V2 policy is closed.")
        if num_steps != NUM_STEPS:
            raise ValueError(f"lingbot2 denoise built for num_steps={NUM_STEPS}; got {num_steps}.")
        images = obs["images"]
        img_masks = obs.get("img_masks")
        lang_tokens = obs["lang_tokens"]
        lang_masks = obs.get("lang_masks")
        state = obs["state"]
        noise = obs["noise"]

        # CPU-only READY admission is deliberately first.  In particular this
        # precedes the optional cache replacement below and every Vision/Z2 op.
        request_metadata = self._request_prefix_geometry(
            images, img_masks, lang_tokens, lang_masks,
            obs.get("image_grid_thw"),
        )
        self._validate_prepared_request(
            request_metadata, num_steps=num_steps)
        self._last_prefix_metadata = dict(request_metadata)
        state = state.float()
        noise = noise.float()

        if not self._vlm_replay:
            self._vgc = rpu_backend.graph.GraphCache()
        if not self._expert_replay:
            if self._denoise_unroll:
                self._dgc = rpu_backend.graph.GraphCache()
            else:
                self._egc = rpu_backend.graph.GraphCache()

        if self._multiview_spm_z2:
            S, real_S, p0 = self._fill_prefix_multiview_spm_z2(
                images, img_masks, lang_tokens, lang_masks,
                obs.get("image_grid_thw"),
            )
            embeds = position_ids = position_ids_rpu = dense_ds = None
            rope_cos_il = rope_sin_il = None
        else:
            (
                embeds, position_ids, position_ids_rpu, dense_ds,
                rope_cos_il, rope_sin_il,
                S, real_S, p0,
            ) = self._build_prefix(
                images, img_masks, lang_tokens, lang_masks,
                obs.get("image_grid_thw"),
            )

        # ── RoPE base vs KV offset ────────────────────────────────────────────────────────────
        # The suffix's RoPE start is p0 = max(prefix M-RoPE position) + 1,
        # which is not the KV-insert offset S whenever the
        # prefix holds images — M-RoPE advances only max(h,w)//spatial_merge_size per image, not
        # one per visual token. The trailing `cos_sin_offset` argument routes p0
        # into the non-M-RoPE decoder while `position` remains the KV offset. We pass
        # position=S + cos_sin_offset=p0 and the image path is fully expressible.
        #
        # On a legacy .so WITHOUT the arg, a single `position` must serve both. That is only
        # correct when p0 == S (text-only prefixes), so refuse loudly rather than mis-route.
        if p0 != S and not self._has_cos_sin_offset:
            raise NotImplementedError(
                f"lingbot2: suffix RoPE start p0={p0} != KV offset S={S} (M-RoPE compresses the "
                f"image span), but the built rpu_backend .so's `lingbot_v2_moe_forward` does not "
                f"expose the trailing `cos_sin_offset` argument. Rebuild the backend with the "
                f"`int cos_sin_offset=-1` argument on lingbot_v2_moe_forward so the RoPE base "
                f"can be set to p0 independently of the cache offset. Refusing to emit a silently-wrong "
                f"action."
            )

        mask_key = (S, real_S)
        mask_r = self._mask_cache.get(mask_key)
        if mask_r is None:
            mask_r = _build_suffix_mask(
                S, self._align_cfg, real_prefix_len=real_S
            )
            # Each (execution, REAL) signature keeps one stable mask address.
            self._mask_cache[mask_key] = mask_r

        if not self._multiview_spm_z2:
            self._vlm_fill(
                embeds, position_ids, position_ids_rpu, dense_ds,
                rope_cos_il, rope_sin_il, S
            )

        graph_rope_offset = self._prepare_ready_suffix_rope(p0)

        if self._denoise_unroll:
            return self._denoise_unroll_run(
                state, noise, mask_r, S, real_S, graph_rope_offset,
                -1.0 / num_steps, num_steps,
            )

        # ── Euler denoise (host loop; step 0 BUILD, steps 1..9 REPLAY) ──
        x_t = noise.clone()
        dt = -1.0 / num_steps
        in_s_all, in_sh_all, po_s_all, po_sh_all = self._fold_all
        state_emb = F.linear(state, self._head_W["model.state_proj.weight"],
                             self._head_W["model.state_proj.bias"])     # call-invariant
        aout_w = self._head_W["model.action_out_proj.weight"]
        aout_b = self._head_W["model.action_out_proj.bias"]
        # Outside a prepared envelope S/REAL/p0 stay in the identity exactly as
        # before. READY has already copied the semantic p0 rows into the stable
        # expert table at offset zero, while the explicit mask is refreshed by
        # CausalDecoderModel::dynamic_config; only the execution bucket is then
        # graph identity.
        denoise_dyn_dims = (
            [EXP_LAYERS, S, PREFIX_POLICY_VERSION]
            if self._prepared_graph_profile is not None else
            [EXP_LAYERS, S, real_S, p0, PREFIX_POLICY_VERSION]
        )
        denoise_sig = rpu_backend.graph.GraphSignature(
            op_id="lingbot2_expert_denoise", shapes=[N_ACTION + 1, EXP_HID],
            dyn_dims=denoise_dyn_dims,
            dtypes=[torch.float16])

        _nt = torch.get_num_threads()
        if self._enc_1thread:
            torch.set_num_threads(1)
        try:
            for k in range(num_steps):
                suffix = embed_suffix_step(x_t, state_emb, self._enc_time_parts[k],
                                           self._head_W, self._enc_Wf)
                if not self._adarms_direct_schedule:
                    torch.ops.rpu.lingbot_v2_moe_set_adarms_step_mutable(
                        self._exp, in_s_all[k], in_sh_all[k], po_s_all[k], po_sh_all[k])
                self._suffix_buf.copy_(suffix.half())          # refresh the STABLE input in place
                out = self._exp_run(
                    self._suffix_buf, mask_r, S, graph_rope_offset, denoise_sig,
                    adarms_step=k if self._adarms_direct_schedule else None)
                # Slice on device before D2H so only the rows consumed by the
                # output projection are transferred. Keep `x_t + dt * v_t` as
                # written: `x_t.add(v_t, alpha=dt)` changes fp32 reassociation.
                suffix_out = out[:, -N_ACTION:].to("cpu").float()        # [1, 50, 768]
                v_t = F.linear(suffix_out, aout_w, aout_b)               # [1, 50, 55]
                x_t = x_t + dt * v_t
        finally:
            if self._enc_1thread:
                torch.set_num_threads(_nt)
        return x_t                                                       # [1, 50, 55]


# =============================================================================
# Build
# =============================================================================
def _materialize_derived_buffers(root):
    """Give real CPU storage to every buffer left on the meta device after
    `load_state_dict(..., assign=True)` on a meta-initialised scaffold.

    Why this is needed: the scaffold is built under `with torch.device("meta")`, and the rope
    `inv_freq` buffers are `persistent=False` — they are DERIVED from the config, never stored in
    the checkpoint — so `assign=True` never gives them storage and they stay on `meta`. The
    subsequent `.to("rpu")` then raises "Cannot copy out of meta tensor; no data!".

    How: re-instantiate the owning module with its OWN official constructor, on CPU, using the
    ctor arguments the module itself recorded, and copy the derived buffers across. The RoPE
    formula is never re-derived by hand — transformers computes it, exactly as it would in a
    normal (non-meta) build.

    Transformers derives three non-persistent buffers:
      * ``Qwen3VLVisionRotaryEmbedding`` owns ``inv_freq`` and is constructed
        from ``head_dim // 2`` with the default theta.
      * ``Qwen3VLTextRotaryEmbedding`` owns ``inv_freq`` and
        ``original_inv_freq`` and is constructed from the model config.

    Transformers 4.57.x does not retain the Vision constructor arguments, so
    they are recovered from ``inv_freq.shape`` and the constructor default.
    Both classes store every ctor argument they need as a plain Python attribute, which meta-init
    does not touch — so the reconstruction uses the same values the original build used.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding,
        Qwen3VLVisionRotaryEmbedding,
    )

    cpu = torch.device("cpu")
    fixed = []
    for mod in root.modules():
        metas = [n for n, b in mod.named_buffers(recurse=False) if b is not None and b.is_meta]
        if not metas:
            continue
        # Rebuilt OUTSIDE any meta context => real CPU storage.
        if isinstance(mod, Qwen3VLVisionRotaryEmbedding):
            # transformers 5.x records self.dim/self.theta; 4.57.x does NOT (its __init__
            # only computes inv_freq and registers it). Recover the ctor args instead of
            # requiring them: inv_freq is `1/(theta**(arange(0,dim,2)/dim))`, so its length
            # is exactly dim//2 — and shape survives meta-init even though storage does not.
            # Theta is the constructor default in supported Transformers versions;
            # the model constructs this class without an explicit theta value.
            dim = getattr(mod, "dim", None)
            if dim is None:
                dim = 2 * mod.inv_freq.shape[0]
            fresh = Qwen3VLVisionRotaryEmbedding(dim, getattr(mod, "theta", 10000.0))
        elif isinstance(mod, Qwen3VLTextRotaryEmbedding):
            fresh = Qwen3VLTextRotaryEmbedding(mod.config, device=cpu)
        else:
            raise RuntimeError(
                f"lingbot2: {type(mod).__name__} owns meta buffer(s) {metas} but is not a known "
                f"config-derived module. Refusing to guess how to materialize it.")
        for n in metas:
            src = getattr(fresh, n, None)
            if src is None or src.is_meta:
                raise RuntimeError(
                    f"lingbot2: official ctor of {type(mod).__name__} did not produce a real "
                    f"buffer {n!r}; refusing to fabricate one.")
            mod.register_buffer(n, src.detach().clone(), persistent=False)
            fixed.append(f"{type(mod).__name__}.{n}{tuple(src.shape)}")
        # Non-buffer derived scalars that the ctor also computes.
        if hasattr(fresh, "attention_scaling"):
            mod.attention_scaling = fresh.attention_scaling
    print(f"[lingbot2] materialized {len(fixed)} config-derived meta buffer(s): {sorted(set(fixed))}")
    return fixed


def _assert_no_meta(root, where):
    """Hard gate before `.to('rpu')`: nothing may remain on meta.

    A meta PARAMETER (or a persistent buffer) at this point means the checkpoint did not cover it
    — that is a real weight-loading bug, never something to paper over. Fail loudly.
    """
    mp = [n for n, t in root.named_parameters() if t.is_meta]
    mb = [n for n, t in root.named_buffers() if t is not None and t.is_meta]
    print(f"[lingbot2] meta check ({where}): params={len(mp)} buffers={len(mb)}")
    if mp or mb:
        raise RuntimeError(
            f"lingbot2: meta tensors remain {where}: "
            f"{len(mp)} parameter(s) {mp[:8]} / {len(mb)} buffer(s) {mb[:8]}. "
            f"Parameters and persistent buffers MUST come from the checkpoint — refusing to "
            f"to_empty()/fabricate them.")
    return len(mp), len(mb)




# RPU_LINGBOT2_PREFILL_W4A16 applies int4 only to these MLP projections;
# attention q/k/v/o remain int8.
_TEXT_INT4_NAMES = {"gate_proj", "up_proj", "down_proj"}


def _quantize_text_w4a16(text_model) -> int:
    """Mixed int8/int4: int8 all 7 projections, then re-quantize the MLP three to int4 values
    (int8 container). Packing to uint8 happens later via the shared INT4 swizzle primitive.
    Runs BEFORE convert_linear_weights_inplace + .to('rpu'). Returns #int4 projections."""
    import torch.nn as nn
    from rpu_backend.quant._common import quantize_linear_per_channel
    n4 = 0
    for layer in text_model.layers:
        for pname, mod in (("q_proj", layer.self_attn.q_proj), ("k_proj", layer.self_attn.k_proj),
                           ("v_proj", layer.self_attn.v_proj), ("o_proj", layer.self_attn.o_proj),
                           ("gate_proj", layer.mlp.gate_proj), ("up_proj", layer.mlp.up_proj),
                           ("down_proj", layer.mlp.down_proj)):
            bits = 4 if pname in _TEXT_INT4_NAMES else 8
            w_q, scale = quantize_linear_per_channel(mod.weight.data, bits=bits)  # int8 container
            mod.weight = nn.Parameter(w_q, requires_grad=False)
            mod.register_buffer("weight_scale", scale.to(torch.float16))
            if bits == 4:
                n4 += 1
    print(f"LINGBOT2_PREFILL_W4A16 int4 {n4} MLP projections ({len(text_model.layers)} layers x 3); "
          f"attention stays int8", flush=True)
    return n4


def _pack_text_w4a16(text_model) -> int:
    """Nibble-pack the three LingBot2 text-MLP INT4 projections in place."""
    import torch.nn as nn
    from rpu_backend.quant.int4_pack import swizzle_pack_int4

    n_packed = 0
    for layer in text_model.layers:
        for name in _TEXT_INT4_NAMES:
            module = getattr(layer.mlp, name)
            if not isinstance(module, nn.Linear):
                raise RuntimeError(f"lingbot2: mlp.{name} is not nn.Linear")
            weight = module.weight.data
            if weight.dtype != torch.int8:
                raise RuntimeError(
                    f"lingbot2: mlp.{name} expected int8 INT4 container, got {weight.dtype}"
                )
            partition = 0 if name == "down_proj" else 1
            n, k = int(module.out_features), int(module.in_features)
            packed = swizzle_pack_int4(
                weight.detach().cpu(), partition=partition, num_cores=ATTN_TP
            ).reshape(n, k // 2)
            module.weight = nn.Parameter(
                packed.contiguous().to(weight.device), requires_grad=False
            )
            n_packed += 1
    return n_packed

def _quantize_text_w8a16(text_model) -> None:
    """On-the-fly W8A16 for the Qwen3-VL TEXT decoder (the VLM prefill).

    Replace each layer's seven projection Linear fp16 weights with
    per-output-channel int8 + a fp16 `weight_scale` buffer.

    ORDER IS LOAD-BEARING: must run BEFORE convert_linear_weights_inplace (which then swizzles the
    int8 weight at dwidth=1) and BEFORE .to('rpu').

    All-or-nothing by construction: causal_decoder_set_weights_w8a16 takes a 7-tuple of scale
    lists and routes the whole layer onto the int8 setter, so a per-class subset is not
    expressible through this op. Nothing else in the tower is touched (norms / embeddings /
    cos-sin stay fp16); the ViT and the action expert are separate models entirely.
    """
    import torch.nn as nn
    from rpu_backend.quant._common import quantize_linear_per_channel
    n = 0
    for layer in text_model.layers:
        for mod in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
                    layer.self_attn.o_proj, layer.mlp.gate_proj, layer.mlp.up_proj,
                    layer.mlp.down_proj):
            w_int8, scale = quantize_linear_per_channel(mod.weight.data)
            mod.weight = nn.Parameter(w_int8, requires_grad=False)
            mod.register_buffer("weight_scale", scale.to(torch.float16))
            n += 1
    print(f"LINGBOT2_PREFILL_W8A16 quantized {n} linears "
          f"({len(text_model.layers)} layers x 7)", flush=True)


def _text_scale_lists(text_model):
    """Gather the 7 per-output-channel fp16 scale lists post `.to('rpu')`.

    Returns None when the tower is not int8 -> install_qwen3_vl_text_for_rpu then takes the
    plain fp16 setter.
    """
    layers = text_model.layers
    _wd = getattr(layers[0].mlp.gate_proj.weight, "dtype", None)
    if _wd not in (torch.int8, torch.uint8):
        return None
    pick = lambda get: [get(layers[i]) for i in range(len(layers))]
    return (pick(lambda l: l.self_attn.q_proj.weight_scale),
            pick(lambda l: l.self_attn.k_proj.weight_scale),
            pick(lambda l: l.self_attn.v_proj.weight_scale),
            pick(lambda l: l.self_attn.o_proj.weight_scale),
            pick(lambda l: l.mlp.gate_proj.weight_scale),
            pick(lambda l: l.mlp.up_proj.weight_scale),
            pick(lambda l: l.mlp.down_proj.weight_scale))


def _build_lingbot_vla_v2_impl(
    policy,
    ckpt,
    qwen3vl_base=None,
    *,
    max_seq=DEFAULT_MAX_SEQ,
    chunk_size=DEFAULT_CHUNK_SIZE,
    align_cfg=None,
    training_config_path=None,
):
    """Build the LingBot-VLA V2 RPU runtime.

    Args:
        ckpt:        path to the lingbot-vla-v2-6b checkpoint dir (sharded fp32 safetensors).
        qwen3vl_base: Qwen3-VL-4B-Instruct dir, for the config ONLY (the scaffold is filled from
                     `ckpt`; no base weights are read). Defaults to the shared read-only mount.
        max_seq:     rope-table + KV-cache capacity (prefix S + suffix 51 must fit).
        chunk_size:  per-instance `_rpu_chunk_size` for the VLM text decoder. 0 = C++ auto.
        align_cfg:   prefix align_params flags. Defaults to AlignConfig(), whose defaults ARE the
                     authoritative kuavo_v2_depth YAML. Pass an explicit AlignConfig for a
                     checkpoint trained with different align_params.
        training_config_path: the checkpoint's own `lingbotvla_cli.yaml`. Its MoE/action geometry
                     (token_num_experts / token_top_k / token_moe_intermediate_size /
                     token_shared_intermediate_size / routed_scaling_factor / chunk_size / …) is
                     read from it instead of the hard-coded kuavo_v2 constants, and cross-checked
                     against the checkpoint weights (fail-loud on mismatch). None ⇒ auto-find a
                     `lingbotvla_cli*.yaml` next to `ckpt`, else the kuavo_v2 defaults.
    """
    _validate_denoise_unroll()
    _validate_adarms_direct_schedule()

    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    # The expert reads the VLM's prefix KV cross-handle out of the shared DDR cache; ddr_flush
    # keeps that cache coherent between the two handles.
    torch.rpu.set_ddr_flush(True)
    # This direct builder does not mutate shared or global defaults because native
    # helpers may sample process-wide environment state. The public facade scopes
    # and restores its runtime settings; direct callers must set opt-ins explicitly.
    # Two-stage all-reduce and its chunked mode remain disabled unless the caller
    # explicitly enables them for a supported profile.
    # Vision switches are caller-controlled and default off. ROPE_SPM uses
    # SPM-resident tables; BATCH groups equal-size cameras and splits differing
    # shapes; FUSED_MERGER captures patch and DeepStack mergers in the graph and
    # uses device fp16, which can change numerical results.
    # HOST_FP32_PATCH changes accumulation order and remains explicit opt-in.
    # The main path returns fresh tensors because callers may retain them.
    # Expert-only W8A16 (convert.py:build_expert_moe). int8 + per-output-channel fp16 scale on the
    # ROUTED expert gate/up/down ONLY; router, attention, AdaRMS, shared expert, residual and the
    # action head all stay fp16. Reuses the shipped quantiser (quant/_common) and the acc16 GEMM's
    # native int8 path — no new quantisation machinery. Requires the grouped path (it quantises the
    # packed weights), so it is inert when RPU_LINGBOT2_GROUPED_EXPERTS is off.
    # W8A16 changes numerics and is opt-in only through
    # RPU_LINGBOT2_EXPERT_W8A16=1.

    # Qwen3-VL base dir resolution — CALL time, explicit source for diagnostics.
    # Precedence: explicit qwen3vl_base arg (threaded from from_checkpoint /
    # runtime_env by api/lingbot2.py) > RPU_LINGBOT2_QWEN3VL_BASE env (direct build()
    # callers) > package default. Never depends on the import-time constant.
    if qwen3vl_base:
        base, base_src = str(qwen3vl_base), "explicit qwen3vl_base arg"
    else:
        _env = os.environ.get("RPU_LINGBOT2_QWEN3VL_BASE", "").strip()
        if _env:
            base, base_src = _env, "RPU_LINGBOT2_QWEN3VL_BASE env"
        else:
            base, base_src = _resolve_qwen3vl_base(), "package default"
    # expand ~ and make absolute BEFORE handing the local path to Transformers, so a
    # user/relative dir is never misread as a HF repo id.
    if base:
        base = os.path.abspath(os.path.expanduser(str(base)))
    print(f"[lingbot2] Qwen3-VL base resolved: source={base_src} value={base!r}")
    if not base or not os.path.isdir(base) or not os.path.isfile(os.path.join(base, "config.json")):
        raise FileNotFoundError(
            f"lingbot2: Qwen3-VL config dir not usable — source={base_src}, resolved value={base!r}. "
            "A local directory containing config.json is required (weights come from the V2 "
            "checkpoint, not from here). Pass qwen3vl_base=<dir>, or "
            "runtime_env={'RPU_LINGBOT2_QWEN3VL_BASE': '<dir>'}; the package-bundled default "
            "(rpu_backend/qwen3_vl_base/) is used only when neither is given."
        )
    cfg = Qwen3VLConfig.from_pretrained(base)
    gw, key_to_reader = ckpt_reader(ckpt)

    # ---- Qwen3-VL scaffold from the base CONFIG, weights from the V2 checkpoint -------------
    # meta init + assign=True ⇒ no double allocation of the 4B backbone.
    with torch.device("meta"):
        vlm = Qwen3VLModel(cfg)
    sd = {k[len(VLM_P):]: gw(k) for k in key_to_reader if k.startswith(VLM_P)}
    missing, unexpected = vlm.load_state_dict(sd, strict=False, assign=True)
    missing = [m for m in missing if not m.endswith(".inv_freq")]
    if missing or unexpected:
        raise ValueError(
            f"lingbot2: Qwen3-VL scaffold/checkpoint mismatch.\n  missing={missing[:8]}\n"
            f"  unexpected={unexpected[:8]}")

    _materialize_derived_buffers(vlm)
    _assert_no_meta(vlm, "after checkpoint load + derived-buffer materialization")
    vlm.eval()

    text_model, vision_model = vlm.language_model, vlm.visual
    # Publish partial ownership before the first irreversible swizzle/RPU
    # allocation. The outer transaction can then retire any text/vision handle
    # installed before a later stage fails.
    policy._vlm = vlm
    policy._text = text_model
    policy._visual = vision_model
    policy._rpu_materialization_started = True
    policy._rpu_swizzle_started = True

    # ---- Swizzle text linears on CPU, then move them to RPU ----
    # ckpt_reader normalizes source BF16/FP32 weights to CPU FP32, so the swizzle would
    # read element_size()==4 and emit a layout the fp16 kernel cannot consume. .half() must
    # precede BOTH the swizzle and .to("rpu"). The vision side casts its own weights explicitly.
    text_model.half()
    # VLM prefill W8A16 — opt-in, DEFAULT OFF (RPU_LINGBOT2_PREFILL_W8A16=1). Must land between
    # .half() and convert_linear_weights_inplace/.to('rpu'). Unset => the fp16 tower, unchanged.
    _prefill_w4 = _env_on("RPU_LINGBOT2_PREFILL_W4A16")
    if _prefill_w4:
        _quantize_text_w4a16(text_model)
        # INT4 weights skip the dtype-based swizzle and use the shared quant primitive.
        convert_linear_weights_inplace(text_model, skip_names=set(_TEXT_INT4_NAMES))
        _n4 = _pack_text_w4a16(text_model)
        print(f"LINGBOT2_PREFILL_W4A16 packed {_n4} int4 projections to uint8", flush=True)
    elif _env_on("RPU_LINGBOT2_PREFILL_W8A16"):
        _quantize_text_w8a16(text_model)
        convert_linear_weights_inplace(text_model, skip_names=set())
    else:
        convert_linear_weights_inplace(text_model, skip_names=set())
    if _env_exact_one("RPU_LINGBOT2_MULTIVIEW_SPM_Z2"):
        # A retained physical arena cannot run an out-of-band RPU embedding
        # lookup between replays.  Keep one host FP16 token table so arbitrary
        # admitted prompt bytes can refresh the stable P225 carrier using CPU
        # F.embedding + a DDR copy only.  This is Z2-only; generic execution
        # retains the ordinary device embedding path and no extra host copy.
        policy._z2_embedding_cpu = (
            text_model.get_input_embeddings().weight.detach()
            .to(device="cpu", dtype=torch.float16).clone().contiguous()
        )
    text_model.to("rpu")
    # ---- Step D: vision encoder (does its own fused-QKV split + swizzle; moves its own weights)
    # Vision W8A16 — opt-in, DEFAULT OFF (RPU_LINGBOT2_VISION_W8A16=1). int8 the 6 ViT GEMMs
    # (q/k/v/o/fc1/fc2) per-output-channel; the vision install + qwen3vl_vision_set_weights_w8a16
    # op support it. Unset keeps the fp16 ViT.
    _vis_w8 = _env_on("RPU_LINGBOT2_VISION_W8A16")
    install_qwen3_vl_vision_for_rpu(vision_model, vision_config=cfg.vision_config, w8a16=_vis_w8)
    # Registering fused patch-merger weights enables its native graph post-function.
    _fused_merger_on = os.environ.get("RPU_QWEN3VL_VISION_FUSED_MERGER", "0") not in ("0", "false", "")
    if _fused_merger_on:
        register_qwen3vl_vision_fused_merger(vision_model)
    # ---- patch_embed on RPU ---------------------------------------------------------------
    # Stage folded patch-embed weights independently of the merger path. Device
    # fp16 changes numerics, so this remains opt-in:
    #   RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE=1
    if os.environ.get("RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE") == "1":
        if not getattr(vision_model, "_rpu_patch_embed_on_device", False):
            from rpu_backend.runtime.weights import tp_col_swizzle_mc_weight

            _pe_w = vision_model._rpu_vision_patch_embed_w
            _pe_b = vision_model._rpu_vision_patch_embed_b
            # The folded Conv3d weight is row-major [embed_dim, cin*tp*ps*ps] and MUST be
            # col-swizzled for rpu_linear, else it silently produces garbage.
            vision_model._rpu_patch_embed_w_rpu = tp_col_swizzle_mc_weight(
                _pe_w.half(), 8).to(device="rpu").contiguous()
            vision_model._rpu_patch_embed_b_rpu = (
                _pe_b.half().to(device="rpu").contiguous() if _pe_b is not None else None)
            vision_model._rpu_patch_embed_on_device = True
    # ---- Vision replay --------------------------------------------------------------------
    # Fast replay enables preload skipping for captured weight nodes. When the
    # fused merger is active, bake its post-function into the replay path.
    if _env_on("RPU_LINGBOT2_VISION_FAST_REPLAY", "0"):
        _vh = vision_model._rpu_vision_handle
        torch.ops.rpu.qwen3vl_vision_set_fast_replay(_vh, True)
        torch.ops.rpu.qwen3vl_vision_set_preload_replay_skip(_vh, True)
        if _fused_merger_on:
            torch.ops.rpu.qwen3vl_vision_set_bake_merger(_vh, True)
    # ---- Step E: text decoder handle -------------------------------------
    # Vision DeepStack outputs are injected into the first matching text layers
    # in list order.
    deepstack_text_layers = list(range(len(cfg.vision_config.deepstack_visual_indexes)))
    install_qwen3_vl_text_for_rpu(
        text_model, text_config=cfg.text_config, vision_config=cfg.vision_config,
        deepstack_lang_layers=deepstack_text_layers,
        enable_deepstack=True, max_seq_len=max_seq,
        scale_lists=_text_scale_lists(text_model))
    # NOTE: no lm_head. The checkpoint has none and the YAML sets `use_lm_head: false` — the VLM
    # exists only to write the prefix KV.
    # Prefill fast replay uses the same preload-skip contract. It is inert unless
    # RPU_LINGBOT2_VLM_REPLAY=1 because a recreated cache always builds first.
    if _env_on("RPU_LINGBOT2_PREFILL_FAST_REPLAY", "0"):
        _th = text_model._rpu_decoder_handle
        torch.ops.rpu.causal_decoder_set_fast_replay(_th, True)
        torch.ops.rpu.causal_decoder_set_preload_replay_skip(_th, True)

    # ---- Expert (sparse MoE) ----
    rope_theta = resolve_rope_theta(cfg.text_config)
    # Resolve action-expert geometry from an explicit or adjacent training YAML,
    # fall back to the profile defaults, and validate against weights/runtime.
    geom = _cv.resolve_geometry(gw, training_config_path=training_config_path, ckpt_dir=ckpt)
    head_W, norm_W = load_host_weights(gw, geom=geom)
    align_W = load_align_weights(gw)
    exp_handle, exp_keep = build_expert_moe(
        gw,
        max_seq=max_seq,
        rope_theta=rope_theta,
        geom=geom,
        on_handle_created=lambda handle: setattr(policy, "_exp", handle),
    )
    policy._exp = exp_handle
    policy._exp_keep = exp_keep

    LingbotVlaV2Policy.__init__(
        policy,
        vlm=vlm,
        cfg=cfg,
        head_W=head_W,
        norm_W=norm_W,
        align_W=align_W,
        align_cfg=align_cfg or AlignConfig(),
        rope_theta=rope_theta,
        exp_handle=exp_handle,
        exp_keep=exp_keep,
        max_seq=max_seq,
        chunk_size=chunk_size,
    )
    policy._rpu_swizzled = True
    return policy


def build_lingbot_vla_v2(
    ckpt,
    qwen3vl_base=None,
    *,
    max_seq=DEFAULT_MAX_SEQ,
    chunk_size=DEFAULT_CHUNK_SIZE,
    align_cfg=None,
    training_config_path=None,
):
    """Build one process-exclusive LingBot-VLA-V2 RPU runtime.

    CPU-only preflight failures release the process slot immediately. Once
    swizzling/RPU materialization starts, a failed owner keeps the slot claimed
    until its traceback is collected, preventing traceback-held RPU tensors from
    overlapping a replacement policy.
    """
    if os.environ.get("RPU_LINGBOT2_ALLOW_UNVALIDATED") != "1":
        raise RuntimeError(
            "LingBot-VLA-V2 is fail-closed by default. Controlled evaluation "
            "is limited to exact-Z2 W8A16, three-image REAL215..225/P225; "
            "FP16, W4 and other profiles are outside this envelope. Set exactly "
            "RPU_LINGBOT2_ALLOW_UNVALIDATED=1 to enter a controlled path; "
            "none is formally or robot certified."
        )
    from rpu_backend.api.causal_lm import (
        _claim_live_instance,
        _release_live_instance,
    )

    policy = LingbotVlaV2Policy.__new__(LingbotVlaV2Policy)
    policy._closed = False
    policy._cleanup_ok = None
    policy._live_slot_released = False
    policy._handle_finalizer = None
    policy._rpu_materialization_started = False
    policy._vlm = None
    policy._text = None
    policy._visual = None
    policy._exp = None
    policy._exp_keep = None
    _claim_live_instance(policy)

    completed = False
    try:
        result = _build_lingbot_vla_v2_impl(
            policy,
            ckpt,
            qwen3vl_base=qwen3vl_base,
            max_seq=max_seq,
            chunk_size=chunk_size,
            align_cfg=align_cfg,
            training_config_path=training_config_path,
        )
        completed = True
        return result
    finally:
        if not completed:
            if getattr(policy, "_rpu_materialization_started", False):
                # Cleanup is immediate, but the claim remains until GC because
                # the exception traceback may still own RPU tensors.
                policy._retire_resources()
            else:
                _release_live_instance(policy)
