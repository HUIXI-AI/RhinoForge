"""LingBot-VLA V2 (robbyant/lingbot-vla-v2-6b) — RPU runtime (image+text → action, flow-matching).

Self-contained standalone runtime, mirroring `adapters/lingbot_vla/runtime.py` (V1) structure. V1 is
NOT touched and NOT imported; this is an independent, default-OFF, opt-in package.

  image  → Qwen3-VL ViT on RPU (install_qwen3_vl_vision_for_rpu)   → img_emb  [n_vis, 2560]
  text   → embed_tokens[lang_tokens]                               → lang_emb [1, L, 2560]
  prefix = image-placeholder tokens + lang tokens, visual embeds masked_scatter'd in
         → RPU Qwen3-VL text decoder (M-RoPE; vision features from [5,11,17]
           injected into text layers [0,1,2]; CAUSAL — YAML vlm_causal: true)
           fills the SHARED RPUCache [0, S)
  state+noise → host encoders + per-(step,layer,norm) AdaRMS fold
         → RPU sparse-MoE AdaRMS expert (768-d, 32 experts, top_k 4, joint shared-prefix-KV,
           explicit 2D mask), 10-step host Euler (default) → action [1, 50, 55]

Deliberately NOT inherited from V1
----------------------------------
* V1's CPU-fp32 ViT. That was a V1-checkpoint-specific fp16-overflow workaround
  (`vit_fp16_outlier_overflow`); V2 uses the validated Qwen3-VL RPU vision tower.
* V1's ROPE_BASE = 10000. The V2 expert does NOT own its RoPE — it is rotated by the VLM's
  rotary_emb (theta 5e6). See convert._build_rope_tables.
* V1's separate in-graph denoise handle. V2 keeps the HOST Euler loop as its default path and
  offers a controlled, default-OFF unroll on the existing sparse-MoE handle. That experiment
  raises the launcher's configurable batch limits and keeps FP16 state between ACC32 linears.

The exact-Z2 W8A16 three-image REAL215..225/P225 envelope has controlled
temporary support under blocking CI, but is not formally or robot certified.
FP16, W4 and other profiles remain evaluation-only. Every path stays fail-closed
unless exact ``RPU_LINGBOT2_ALLOW_UNVALIDATED=1`` is set; the env name is retained
for compatibility. Nothing here changes V1.
"""
from __future__ import annotations

import dataclasses
import math
import os
import threading
import weakref
from contextlib import contextmanager, nullcontext
from itertools import count
from numbers import Integral

import torch
import torch.nn.functional as F

import rpu_backend
from rpu_backend.api import RPUCache
from rpu_backend.runtime import rpu_env_bool
from rpu_backend.adapters.qwen3_vl import (
    install_qwen3_vl_text_for_rpu,
    install_qwen3_vl_vision_for_rpu,
    register_qwen3vl_vision_fused_merger,
    scatter_visual_embeds_to_dense,
)
from rpu_backend.adapters.qwen3_vl.vision import (
    _prepare_qwen3vl_vision_input_for_spm_pipeline,
)
from rpu_backend.api._execution import (
    bind_execution_session,
    execution_serialized,
    native_execution_reconfigure,
    normalize_rpu_execution,
    resolve_component_rpu_execution,
)
from rpu_backend.runtime.decoder import (
    _run_causal_decoder_forward,
    plan_bounded_prefill_execution,
)
from rpu_backend.runtime.execution_planner import (
    GRAPH_COMPOSITE_CHILD,
    GRAPH_NATIVE_COMPOSITE1,
    PlannerRejectError,
    StageTuple,
    capability_selected,
    plan_a6,
    plan_fixed_component_execution,
)
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
    PrefixExecutionPlan, plan_prefix_execution, suffix_position_start,
)

# VLM (Qwen3-VL-4B-Instruct) — checkpoint-verified.
VLM_LAYERS = 36
VLM_HID = 2560
ATTN_TP = 8                     # C++ attn_tp() == min(NUM_CORES, num_kv_heads); 8 for BOTH halves,
                                # which is exactly why one shared RPUCache serves both.
DEFAULT_MAX_SEQ = 1024          # rope-table + KV-cache capacity (prefix S + suffix 51 must fit)
PREFIX_PADDING_MULTIPLE = 16
PREFIX_POLICY_VERSION = 1
MULTIVIEW_SPM_Z2_EXECUTION_LEN = 225
MULTIVIEW_SPM_Z2_MIN_REAL_LEN = 215
MULTIVIEW_SPM_Z2_VISION_CHUNK = 256
MULTIVIEW_SPM_Z2_PREFILL_CHUNK = 240
MULTIVIEW_SPM_Z2_ACTION_CHUNK = 64
_Z2_NATIVE_HEADER_WORDS = 36
_Z2_VISION_DESCRIPTOR_SIZE_INDEX = _Z2_NATIVE_HEADER_WORDS
_Z2_CHILD_DIGEST_WORDS = 3 * 4
_READY_ROPE_DIRTY = object()
_Z2_PROMPT_DIRTY = object()
_Z2_PREPARE_ATTEMPTS = count(1)
_Z2_RECONFIGURE_ATTEMPTS = count(1)
_MISSING = object()

LINGBOT2_VISION_COMPONENT = "vision_encoder"
LINGBOT2_TEXT_COMPONENT = "language_model"
LINGBOT2_ACTION_COMPONENT = "action_expert"
LINGBOT2_EXECUTION_COMPONENTS = {
    LINGBOT2_VISION_COMPONENT: {"vision": ("chunk_size",)},
    LINGBOT2_TEXT_COMPONENT: {
        "prefill": ("chunk_size", "padding_rows", "padding_budget"),
    },
    LINGBOT2_ACTION_COMPONENT: {"action": ("chunk_size",)},
}


def resolve_lingbot2_execution_components(value, *, entry_point: str):
    """Resolve one canonical config into LingBot2's three stable children."""
    root = normalize_rpu_execution(
        value,
        entry_point=entry_point,
        supported_components=LINGBOT2_EXECUTION_COMPONENTS,
    )
    vision = resolve_component_rpu_execution(
        root,
        LINGBOT2_VISION_COMPONENT,
        entry_point=entry_point,
        supported_components=LINGBOT2_EXECUTION_COMPONENTS,
        profile_auto={"vision": {"chunk_size": "auto"}},
    )
    text = resolve_component_rpu_execution(
        root,
        LINGBOT2_TEXT_COMPONENT,
        entry_point=entry_point,
        supported_components=LINGBOT2_EXECUTION_COMPONENTS,
        profile_auto={
            "prefill": {
                "chunk_size": "auto",
                "padding_rows": "auto",
                "padding_budget": PREFIX_PADDING_MULTIPLE - 1,
            },
        },
    )
    action = resolve_component_rpu_execution(
        root,
        LINGBOT2_ACTION_COMPONENT,
        entry_point=entry_point,
        supported_components=LINGBOT2_EXECUTION_COMPONENTS,
        profile_auto={"action": {"chunk_size": "auto"}},
    )
    return root, vision, text, action


def _plan_digest_words(digest: str) -> tuple[int, int, int, int]:
    raw = bytes.fromhex(digest)
    if len(raw) != 32:
        raise ValueError("planner digest must contain exactly 32 bytes")
    return tuple(
        int.from_bytes(raw[index:index + 8], "big", signed=True)
        for index in range(0, 32, 8)
    )


def _plan_fixed_lingbot2_component(*, stage: str, **kwargs):
    """Attach the stable LingBot2 child stage to generic fixed-plan errors."""
    try:
        return plan_fixed_component_execution(
            stage=stage, **kwargs
        )
    except PlannerRejectError as exc:
        raise PlannerRejectError(
            exc.code,
            stage=stage.upper(),
            requested=exc.requested,
            limit=exc.limit,
            resolved=int(kwargs["chunk_size"]),
            clamped=exc.clamped,
            detail=str(exc),
        ) from exc


def _plan_z2_native_child(
    descriptor, *, logical_len, component_id, stage, generation,
    requested_chunk, execution_owner, native_prefix, native_handle,
    parent_handle, physical_metadata=(),
):
    """Plan one actual native occurrence, preserving its COMPLETE stage axes."""
    plans = []
    plan_bounded_prefill_execution(
        logical_len, logical_len, 0, padding_rows=0,
        exact_chunk_size=None if requested_chunk == "auto" else requested_chunk,
        resolve_stage_domain=lambda _length: [1, 1, len(descriptor), *descriptor],
        graph_mode=GRAPH_COMPOSITE_CHILD,
        physical_metadata=((f"component:{component_id}", 1),
                           ("execution_generation", generation), *physical_metadata),
        queue_owner_id=parent_handle, lease_owner_id=parent_handle,
        execution_owner=execution_owner, execution_component=component_id,
        execution_stage=stage, execution_native=(native_prefix, native_handle),
        plan_result_sink=plans.append,
        plan_signature=(tuple(descriptor),),
        graph_cache=None,
    )
    return plans[0]


def _validate_lingbot2_cold_execution_config(
    execution_config, *, multiview_spm_z2: bool
) -> None:
    """Reject fixed-child requests before any model materialization."""
    root, vision, text, action = resolve_lingbot2_execution_components(
        execution_config,
        entry_point="LingBot2 cold execution preflight",
    )
    del root
    stages = (("action", N_ACTION + 1, MULTIVIEW_SPM_Z2_ACTION_CHUNK),)
    if multiview_spm_z2:
        stages = (
            ("vision", 256, MULTIVIEW_SPM_Z2_VISION_CHUNK),
            ("prefill", MULTIVIEW_SPM_Z2_EXECUTION_LEN,
             MULTIVIEW_SPM_Z2_PREFILL_CHUNK),
            *stages,
        )
        padding_rows = text.get("prefill", {}).get(
            "padding_rows", "auto"
        )
        if isinstance(padding_rows, int) and padding_rows > (
            MULTIVIEW_SPM_Z2_EXECUTION_LEN
            - MULTIVIEW_SPM_Z2_MIN_REAL_LEN
        ):
            raise PlannerRejectError(
                "CAPABILITY", stage="PREFILL", requested=padding_rows,
                limit=10,
                detail="LingBot2 Z2 exact padding has no REAL215..225 request",
            )
    stage_configs = {"vision": vision, "prefill": text, "action": action}
    component_ids = {
        "vision": LINGBOT2_VISION_COMPONENT,
        "prefill": LINGBOT2_TEXT_COMPONENT,
        "action": LINGBOT2_ACTION_COMPONENT,
    }
    for stage, logical_len, tile in stages:
        _plan_fixed_lingbot2_component(
            logical_len=logical_len,
            execution_len=logical_len,
            chunk_size=tile,
            component_id=component_ids[stage],
            stage=stage,
            stage_config={
                "chunk_size": stage_configs[stage].get(
                    stage, {}
                ).get("chunk_size", "auto"),
            },
            physical_metadata=(("cold_preflight", 1),),
            queue_owner_id=0,
            lease_owner_id=0,
        )


def _decode_z2_native_descriptor(raw, handles):
    descriptor = tuple(int(value) for value in raw)
    if len(descriptor) <= _Z2_VISION_DESCRIPTOR_SIZE_INDEX:
        raise RuntimeError(
            "LingBot2 Z2 native planner returned an invalid descriptor length: "
            f"got {len(descriptor)}"
        )
    vision_descriptor_words = descriptor[_Z2_VISION_DESCRIPTOR_SIZE_INDEX]
    native = _z2_native_descriptor_from_planner(descriptor)
    if len(descriptor) != len(native):
        raise RuntimeError(
            "LingBot2 Z2 native planner returned a stale child descriptor "
            f"boundary: count={vision_descriptor_words}, total={len(descriptor)}"
        )
    for child in _z2_native_child_descriptors(descriptor):
        if len(child) < 2 or child[1] != len(child):
            raise RuntimeError("LingBot2 Z2 native planner returned an invalid FMB child descriptor")
    expected = {
        0: 3, 1: 3, 2: 3,
        3: 0, 4: 3, 5: 256, 6: 256,
        9: 1, 10: 225, 11: 240, 12: 225, 13: 1,
        16: 2, 17: 51, 18: 64,
        23: 1, 24: 3, 25: 9, 26: 64,
        27: 1, 28: 67, 29: 133,
        30: 1_093_120, 31: 8_205_312, 32: 8_303_616,
        33: int(handles[0]), 34: int(handles[1]), 35: int(handles[2]),
    }
    mismatches = {
        index: (descriptor[index], value)
        for index, value in expected.items()
        if descriptor[index] != value
    }
    if mismatches:
        raise RuntimeError(
            "LingBot2 Z2 native planner descriptor drifted: "
            f"{mismatches}"
        )
    for index in (7, 8, 14, 15, 19, 20, 21, 22):
        if descriptor[index] == 0:
            raise RuntimeError(
                "LingBot2 Z2 native planner descriptor contains a zero "
                f"identity at word {index}"
            )
    return descriptor


def _z2_native_descriptor_from_planner(planner_descriptor):
    values = tuple(int(value) for value in planner_descriptor)
    if len(values) <= _Z2_VISION_DESCRIPTOR_SIZE_INDEX:
        raise RuntimeError("LingBot2 Z2 planner descriptor is truncated")
    vision_end = (
        _Z2_VISION_DESCRIPTOR_SIZE_INDEX + 1
        + values[_Z2_VISION_DESCRIPTOR_SIZE_INDEX]
    )
    if not _Z2_VISION_DESCRIPTOR_SIZE_INDEX + 1 < vision_end < len(values):
        raise RuntimeError("LingBot2 Z2 planner Vision descriptor is empty or truncated")
    native_words = vision_end + 1 + values[vision_end]
    if not vision_end + 1 < native_words <= len(values):
        raise RuntimeError("LingBot2 Z2 planner Text descriptor is empty or truncated")
    return values[:native_words]


def _z2_native_child_descriptors(native):
    vision_end = _Z2_VISION_DESCRIPTOR_SIZE_INDEX + 1 + native[_Z2_VISION_DESCRIPTOR_SIZE_INDEX]
    return native[_Z2_VISION_DESCRIPTOR_SIZE_INDEX + 1:vision_end], native[vision_end + 1:]


_FAILED_RETIREMENTS = []


def _poison_lingbot2_retirement(owner, error):
    """Keep unretired native addresses alive; failed cleanup is never retried."""
    if getattr(owner, "_cleanup_ok", None) is not False:
        owner._cleanup_ok = False
        _FAILED_RETIREMENTS.append(owner)
    session = getattr(owner, "_execution_session", None)
    try:
        if session is not None:
            session.poison()
    except ReferenceError:
        pass  # The facade's Session may already have been collected.
    for child, name in ((getattr(owner, "_text", None), "decoder"),
                        (getattr(owner, "_visual", None), "vision")):
        resource = getattr(child, f"_rpu_{name}_retirement_state", None)
        if resource is not None and resource.handle is not None:
            try:
                resource.retain_failure(error, owner)
            except ReferenceError:
                pass  # Resource was retained before its expired Session proxy.
    from rpu_backend.api import causal_lm, _execution

    with causal_lm._LIVE_LOCK:
        if causal_lm._LIVE_TERMINAL_REASON is None:
            live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
            if live is None and _execution._UNSAFE_PROCESS_REASON is None:
                live = owner
                causal_lm._claim_live_instance(live)
            if live is not None:
                causal_lm._poison_live_instance(live, f"LingBot2 cleanup failed: {error!r}", unsafe=True)
        _execution._mark_execution_process_unsafe(f"LingBot2 cleanup failed: {error!r}")


def _take_lingbot2_component_retirement(owner, child, name):
    resource = getattr(child, f"_rpu_{name}_retirement_state", None)
    if resource is None:
        raise RuntimeError(f"LingBot2 {name} has no installed retirement state")
    prior = getattr(child, "_lingbot2_retirement_owner", None)
    if prior is not None and prior is not owner:
        raise RuntimeError(f"LingBot2 {name} has a different retirement owner")
    resource.take_ownership(owner)
    vars(child)["_lingbot2_retirement_owner"] = owner


def _resolve_qwen3vl_base():
    """Locate the Qwen3-VL-4B-Instruct dir used ONLY for `Qwen3VLConfig.from_pretrained`.

    No weights are read from it (they come from the V2 checkpoint), so only its
    config.json (~1.5 KB) is needed. First candidate that actually holds a
    config.json wins; a missing candidate is skipped, so this never returns a path
    that does not exist. Resolution order:

      1. ``RPU_LINGBOT2_QWEN3VL_BASE`` env — deployment override (validated at use).
      2. the copy bundled next to rpu_backend.so by the closed-source wheel build
         (``rpu_backend/qwen3_vl_base/``) — legacy co-located location.
      3. the copy shipped INSIDE this adapter package
         (``adapters/lingbot_vla_v2/_qwen3vl_base/``) — present in EVERY install, so
         the customer needs no configuration and no second model directory. This is
         what removes the old hard-coded-dev-mount dependency.
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
        os.path.join(pkg_root, "qwen3_vl_base"),   # closed-source wheel bundle (next to the .so)
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


def _lingbot2_exact_bool(name: str, *, default: bool = False) -> bool:
    """Translate a legacy exact 0/1 switch before native materialization."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw not in ("0", "1"):
        raise ValueError(f"{name} must be exactly 0 or 1, got {raw!r}")
    return raw == "1"


def _lingbot2_nonnegative_int(name: str, *, default: int) -> int:
    """Translate one canonical decimal integer; reject ambiguous C atol input."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    if not raw or not raw.isascii() or not raw.isdigit():
        raise ValueError(
            f"{name} must be a canonical non-negative decimal integer, got {raw!r}"
        )
    value = int(raw)
    if raw != str(value):
        raise ValueError(
            f"{name} must use canonical decimal spelling, got {raw!r}"
        )
    return value


@dataclasses.dataclass(frozen=True)
class _Lingbot2MoeColdConfig:
    """Single Python authority for native MoE execution controls."""

    bufonly: bool
    addr: bool
    schunk: int
    rchunk_requested: int
    dense_soft_router: bool
    fp16_top4: bool
    routed_only: bool
    down_acc16: bool

    @classmethod
    def from_env(cls) -> "_Lingbot2MoeColdConfig":
        dense_soft_router = _lingbot2_exact_bool(
            "RPU_LINGBOT2_DEBUG_DENSE_SOFT_ROUTER"
        )
        schunk = _lingbot2_nonnegative_int(
            "RPU_L2_SCHUNK", default=256 * 127
        )
        return cls(
            # These two historical controls intentionally use presence
            # semantics: an explicitly present "0" still enables them.
            bufonly="RPU_L2_BUFONLY" in os.environ,
            addr="RPU_L2_ADDR" in os.environ,
            schunk=schunk if schunk > 0 else 256 * 127,
            rchunk_requested=_lingbot2_nonnegative_int(
                "RPU_L2_RCHUNK", default=0
            ),
            dense_soft_router=dense_soft_router,
            fp16_top4=_lingbot2_exact_bool(
                "RPU_LINGBOT2_FP16_TOP4",
                default=not dense_soft_router,
            ),
            routed_only=_lingbot2_exact_bool("RPU_L2_ROUTED_ONLY"),
            down_acc16=_lingbot2_exact_bool("RPU_L2_DOWN_ACC16"),
        )


def _close_multiview_spm_z2_state(state) -> bool:
    """Close retained physical Z2 before any child GraphCache or handle.

    The parent retains this state for both explicit close and graph-aware GC.
    """
    if not state or not state.get("prepare_attempted", False):
        return True
    if state.get("retirement_error") is not None:
        return False
    if state.get("owner_thread") != threading.get_ident():
        state["retirement_error"] = RuntimeError("LingBot2 Z2 retirement on a foreign thread")
        return False
    try:
        args = (
            state["vision_handle"], state["text_handle"],
            state["expert_handle"],
        )
        identity = (
            state.get("planner_descriptor") or (),
            state.get("planner_digest_words") or (),
        )
        if state.get("prepared", False):
            torch.ops.rpu.lingbot2_multiview_spm_z2_close(
                *args, state["plan_hash"], *identity
            )
        else:
            try:
                torch.ops.rpu.lingbot2_multiview_spm_z2_cancel_prepare(
                    *args, state["prepare_attempt_token"], *identity
                )
            except Exception as cancel_error:
                # Preserve the pre-existing explicit retry for a native
                # prepare failure whose internal teardown remained Poisoned.
                try:
                    torch.ops.rpu.lingbot2_multiview_spm_z2_close(
                        *args, 0, *identity
                    )
                except BaseException as close_error:
                    cancel_error.add_note(f"LingBot2 poisoned Z2 close also failed: {close_error!r}")
                    raise cancel_error
    except BaseException as error:
        state["retirement_error"] = error
        return False
    state["prepared"] = False
    state["prepare_attempted"] = False
    state["poisoned"] = False
    state["owner_thread"] = None
    state["plan_hash"] = None
    state["prepare_attempt_token"] = None
    return True


def _h(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float16).to("rpu").contiguous()


def resolve_rope_theta(text_config) -> float:
    """The expert's RoPE theta — sourced from the REAL config, never a hardcoded constant.

    ⚠️ V1 hardcodes ROPE_BASE = 10000. Inheriting that here is silently wrong: the V2 action
    expert does NOT own its RoPE. `apply_mrope` (modeling_lingbot_vla_v2.py:275-277) rotates the
    expert's q/k with `self.qwenvl.model.language_model.rotary_emb` — the VLM's, whose theta is
    5e6 for Qwen3-VL-4B. Raises rather than defaulting: a wrong-but-plausible theta would produce
    a silently-wrong action, and 10000 is exactly the plausible wrong value.
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
            "config (5e6 for Qwen3-VL-4B) — refusing to fall back to V1's 10000.")
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
    """Does the built .so expose the complete planned LingBot2 action ABI?

    The C++ side is (in parallel) adding `int cos_sin_offset=-1`, mirroring
    causal_decoder_forward (rpu_backend.cpp:2693): -1 == "same as position" (back-compat),
    >=0 == the RoPE position base while `position` keeps driving the KV-insert offset.

    We read the REGISTERED SCHEMA rather than try/except-ing a call: a TypeError from a real
    call could equally mean a genuine bug, and silently falling back would mis-route the RoPE
    base (exactly the silent-wrong-action failure this whole guard exists to prevent).
    """
    try:
        schema = torch.ops.rpu.lingbot_v2_moe_forward.default._schema
        return [argument.name for argument in schema.arguments] == [
            "handle", "hidden_states", "k_caches", "v_caches",
            "attention_mask", "position", "cos_sin_offset",
            "adarms_schedule_step", "planned_stage_descriptor",
        ]
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
                "planned_stage_descriptor",
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
    # Stage-2: the separate denoise Graph has no physical-pipeline lease or
    # semantic producer yield, so it may skip its audited full body and the
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
}

_MULTIVIEW_SPM_Z2_PROFILE_ENV: dict[str, dict[str, str]] = {
    "dense_fp16": {
        "RPU_DEEP_FAST_REPLAY": "0",
        "RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE": "0",
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
        "RPU_LINGBOT2_GROUPED_EXPERTS": "1",
        "RPU_LINGBOT2_EXPERT_W8A16": "1",
        "RPU_LINGBOT2_EXPERT_W4A16": "0",
        "RPU_LINGBOT2_PREFILL_W8A16": "1",
        "RPU_LINGBOT2_PREFILL_W4A16": "0",
        "RPU_LINGBOT2_BASE_W8A16": "1",
        "RPU_LINGBOT2_VISION_W8A16": "1",
        "RPU_L2_RCHUNK": "1632",
    },
    "grouped_w4a16": {
        "RPU_DEEP_FAST_REPLAY": "1",
        "RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE": "1",
        "RPU_LINGBOT2_GROUPED_EXPERTS": "1",
        "RPU_LINGBOT2_EXPERT_W8A16": "0",
        "RPU_LINGBOT2_EXPERT_W4A16": "1",
        "RPU_LINGBOT2_PREFILL_W8A16": "1",
        "RPU_LINGBOT2_PREFILL_W4A16": "0",
        "RPU_LINGBOT2_BASE_W8A16": "1",
        "RPU_LINGBOT2_VISION_W8A16": "1",
        "RPU_L2_RCHUNK": "1632",
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
        "lingbot2_multiview_spm_z2_dry_probe": [
            "vision_handle", "text_handle", "expert_handle",
        ],
        "lingbot2_multiview_spm_z2_prepare": [
            "vision_handle", "text_handle", "expert_handle", "position_ids",
            "rope_cos_il", "rope_sin_il", "planner_descriptor",
            "planner_digest", "prepare_attempt_token",
        ],
        "lingbot2_multiview_spm_z2_cancel_prepare": [
            "vision_handle", "text_handle", "expert_handle",
            "prepare_attempt_token", "planner_descriptor", "planner_digest",
        ],
        "lingbot2_multiview_spm_z2_reconfigure_begin": [
            "vision_handle", "text_handle", "expert_handle", "plan_hash",
            "planner_descriptor", "planner_digest", "reconfigure_attempt_token",
        ],
        "lingbot2_multiview_spm_z2_forward": [
            "vision_handle", "text_handle", "expert_handle", "vision_inputs",
            "vision_k_caches", "vision_v_caches", "text_hidden",
            "text_k_caches", "text_v_caches", "position_ids", "rope_cos_il",
            "rope_sin_il", "plan_hash", "planner_descriptor",
            "planner_digest",
        ],
        "lingbot2_multiview_spm_z2_refresh_text_inputs": [
            "vision_handle", "text_handle", "expert_handle", "position_ids",
            "rope_cos_il", "rope_sin_il", "plan_hash",
            "planner_descriptor", "planner_digest",
        ],
        "lingbot2_multiview_spm_z2_stats": [
            "vision_handle", "text_handle", "expert_handle", "plan_hash",
            "planner_descriptor", "planner_digest",
        ],
        "lingbot2_multiview_spm_z2_executed_owner": [
            "vision_handle", "text_handle", "expert_handle", "plan_hash",
            "planner_descriptor", "planner_digest",
        ],
        "lingbot2_multiview_spm_z2_close": [
            "vision_handle", "text_handle", "expert_handle", "plan_hash",
            "planner_descriptor", "planner_digest",
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
# Host encoders + AdaRMS fold (periods verified against the official
# create_sinusoidal_pos_embedding: min_period 4e-3, max_period 4.0, dim = proj_width = 768)
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
    """Fold the consecutive linears at the start of the suffix encoder.

    There is no nonlinearity between action_in_proj and action_time_mlp_in.
    With Wa = mlp_in.weight[:, :768] and Wt = mlp_in.weight[:, 768:]:
        (x @ Wp.T + bp) @ Wa.T + (t @ Wt.T + b)
            = x @ (Wa @ Wp).T + (bp @ Wa.T + t @ Wt.T + b)
    The time term is constant per Euler step. Folding removes the expanded
    time concatenation, but floating-point reassociation can change rounding.

    Return contiguous W_fused [768,55] and [1,1,768] biases for each step."""
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
    """Batch the AdaRMS fold across all Euler steps (V1 runtime.py:295-316, Lever 1).

    ada_cond = sinusoidal(t) depends ONLY on t (adanorm_time: true), so the whole schedule folds
    once at build and ships in 4 big `.to('rpu')` transfers instead of 36*2*2*num_steps tiny [768]
    ones. Returns 4 rpu tensors [num_steps, EXP_LAYERS, 768]; slice [k] per step → 36 contiguous
    [768] rows for lingbot_v2_moe_set_adarms_step_mutable.

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

    Mirrors predict_velocity (:1024-1042):
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

    The prefix layout puts the <image> placeholders in exactly `n_images` contiguous runs
    ([<vision_start> <image>*P <vision_end>] repeated), so scattering the visual embeddings needs
    no boolean mask at all — `narrow().copy_()` per run is the same write with an RPU kernel
    behind it, instead of an `index_put_` that has none (it falls back to the CPU, round-tripping
    the whole [S, hidden] tensor twice).

    Derived from the mask rather than from the layout arithmetic on purpose: if the layout ever
    changes shape this keeps producing the right runs, and the caller asserts the total against
    the ViT token count.
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

    Built on CPU then moved (pitfall B-1: never torch.zeros(device='rpu')).
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
                 exp_handle, exp_keep, max_seq=DEFAULT_MAX_SEQ, rpu_execution=None,
                 execution_session=None, execution_generation=0,
                 component_generations=None):
        # Establish cleanup state first: the public builder may call
        # _retire_resources() after any later constructor failure.
        self._closed = False
        self._cleanup_ok = None
        self._live_slot_released = False
        self._gc_retirement_enabled = True
        self._multiview_spm_z2 = _validate_multiview_spm_z2()
        self._graph_runtime_policy = (
            rpu_backend.graph.GraphRuntimePolicy.from_environment(
                lingbot2_multiview_spm_z2=self._multiview_spm_z2
            )
        )
        self._z2_state = {
            "prepare_attempted": False,
            "prepared": False,
            "poisoned": False,
            "plan_hash": None,
            "prepare_attempt_token": None,
            "reconfigure_token": None,
            "planner_descriptor": None,
            "planner_digest": None,
            "planner_digest_words": None,
            "planner_parent": None,
            "planner_children": None,
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
        resolved_execution = resolve_lingbot2_execution_components(
            rpu_execution,
            entry_point="LingbotVlaV2Policy",
        )
        self._rpu_execution = resolved_execution[0]
        self._execution_generation = int(execution_generation)
        self._component_generations = (
            {
                component: self._execution_generation
                for component in LINGBOT2_EXECUTION_COMPONENTS
            }
            if component_generations is None
            else {
                component: int(component_generations[component])
                for component in LINGBOT2_EXECUTION_COMPONENTS
            }
        )
        self._vision_execution = resolved_execution[1]
        self._text_execution = resolved_execution[2]
        self._action_execution = resolved_execution[3]
        self._execution_components = {
            LINGBOT2_VISION_COMPONENT: self._vision_execution,
            LINGBOT2_TEXT_COMPONENT: self._text_execution,
            LINGBOT2_ACTION_COMPONENT: self._action_execution,
        }
        self._last_execution_plan = {}
        self._execution_reconfigure_journal = None
        self._align_cfg = align_cfg
        self._ids = SpecialTokenIds.from_config(cfg)
        self._spatial_merge = int(cfg.vision_config.spatial_merge_size)
        self._rope_theta = rope_theta      # 5e6 — sourced from config, NEVER V1's 10000
        self._mrope_head_dim, self._mrope_section = resolve_mrope_geometry(
            cfg.text_config)
        # The 8 current_depth + 8 future_depth prefix query tokens. Call-invariant (they depend
        # only on weights) → computed ONCE at build. [n_task, VLM_HID] fp32 each.
        self._align_tokens = compute_align_tokens(align_W, align_cfg)
        # Whether the built .so exposes the trailing `cos_sin_offset` arg on lingbot_v2_moe_forward.
        self._has_cos_sin_offset = _op_has_cos_sin_offset()
        if not self._has_cos_sin_offset:
            raise RuntimeError(
                "LingBot2 requires a backend whose action forward accepts "
                "cos_sin_offset and the complete planned_stage_descriptor"
            )

        # ONE shared RPUCache: the VLM fills [0, S); the expert reads that prefix KV cross-handle
        # and writes only its own suffix [S, S+51). Both halves are 8 KV heads / head_dim 128 /
        # attn_tp 8, so the 7-D swizzle layout is identical — this is what makes sharing legal.
        self._cache = RPUCache(num_layers=VLM_LAYERS, batch_size=1, max_seq_len=max_seq,
                               num_kv_heads=NKV, head_dim=HD, attn_tp=ATTN_TP)

        # PER-MODEL graph caches (never get_default_graph_cache() — a shared default would let a
        # sibling model's signature collide with ours).
        self._vgc = rpu_backend.graph.GraphCache(
            runtime_policy=self._graph_runtime_policy
        )
        self._egc = rpu_backend.graph.GraphCache(
            runtime_policy=self._graph_runtime_policy
        )
        self._prepared_graph_profile = None
        self._graphs_ready = False
        self._last_prefix_metadata = None

        # P5 (docs/pitfalls.md:226-288): chunk size is a PER-INSTANCE attribute read by the patched
        # forward; torch.rpu.set_chunk_size() does NOT reach fused paths. Set it explicitly so the
        # chunk plan (and therefore the fp16 reduce order) is pinned rather than left to the C++
        # binary-search auto-select. Must be set AFTER .to('rpu').
        requested_prefill_chunk = self._text_execution.get(
            "prefill", {}
        ).get("chunk_size", "auto")
        self._chunk_size = (
            0 if (
                requested_prefill_chunk == "auto"
                or _env_exact_one("RPU_LINGBOT2_MULTIVIEW_SPM_Z2")
            )
            else int(requested_prefill_chunk)
        )
        self._text._rpu_chunk_size = self._chunk_size
        enable_reconfigure = getattr(
            torch.ops.rpu,
            "causal_decoder_enable_execution_reconfigure",
            None,
        )
        if enable_reconfigure is None:
            raise RuntimeError(
                "LingBot2 binary lacks causal-decoder hot-reconfigure support"
            )
        enable_reconfigure(self._text._rpu_decoder_handle)

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

        # Build-once/replay expert denoise (default ON — V1's proven default): one graph signature
        # for all NUM_STEPS steps + set_adarms_step_mutable (no per-step rebuild) + a STABLE suffix
        # input buffer refreshed in place ⇒ step 0 BUILDs, steps 1..9 REPLAY.
        # `lingbot_v2_moe_forward` bakes the input tensor's DDR address at BUILD (fixed-DMA), so the
        # per-forward input MUST live at a stable address for a replay to be valid.
        self._expert_replay = _env_on("RPU_LINGBOT2_EXPERT_REPLAY", "1")
        self._denoise_unroll = _validate_denoise_unroll()
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
            self._dgc = rpu_backend.graph.GraphCache(
                runtime_policy=self._graph_runtime_policy
            )
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
        # Restrict the small per-step host encoder matmuls to one thread when enabled.
        # Set the option to "0" to retain the caller's ambient thread configuration.
        self._enc_1thread = _env_on("RPU_LINGBOT2_ENCODER_1THREAD", "1")
        # Prefix constants depend on image count, tokens per image, language tokens
        # and grid. Cache them in RPU buffers and overwrite only the visual rows;
        # disabling this option rebuilds the prefix for each request.
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
        # Cross-call VLM replay requires stable inputs_embeds, all DeepStack buffers
        # and position_ids. Prefix preparation owns and updates those buffers in place.
        # The replay default follows _prefix_opt and remains off without that ownership.
        self._vlm_replay = _env_on("RPU_LINGBOT2_VLM_REPLAY",
                                   "1" if self._prefix_opt else "0")
        if self._vlm_replay and not self._prefix_opt:
            # This combination is the one the original note was right about. It is reachable:
            # api/lingbot2.py:66-70 `_STRUCTURAL_ENV` sets RPU_LINGBOT2_VLM_REPLAY=1
            # UNCONDITIONALLY for every Lingbot2Policy, so a caller who passes
            # runtime_env={"RPU_LINGBOT2_HOST_PREFIX_OPT": "0"} lands here — replay ON with the
            # per-call allocations back. Refuse the unsafe combination before the first
            # capture; stale fixed-DMA addresses are a correctness failure, not a warning.
            raise RuntimeError(
                "lingbot2: RPU_LINGBOT2_VLM_REPLAY=1 with RPU_LINGBOT2_HOST_PREFIX_OPT=0. "
                "The prefill graph is replayed against inputs_embeds / dense deepstack / "
                "position_ids that are re-ALLOCATED every call, so the replayed graph may read "
                "freed addresses. Replay was only proven sound WITH the persistent prefix "
                "buffers (results/lingbot2/vishost_fix/vlmreplay_soundness.json). Set "
                "HOST_PREFIX_OPT=1, or VLM_REPLAY=0.")

        # Controlled Wall-OSS-style host-glue experiment.  The fused Vision
        # merger writes pooler + all three DeepStack outputs directly into the
        # persistent VLM prefix tensors, eliminating twelve synchronous Python
        # narrow().copy_() launches per frame.  The native DMA destinations are
        # fixed at Vision Graph BUILD, so this is deliberately cold and tightly
        # gated to the single-packed-group path proven below.
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
            if not self._visual._rpu_vision_batch_enabled:
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

        _take_lingbot2_component_retirement(self, self._text, "decoder")
        _take_lingbot2_component_retirement(self, self._visual, "vision")
        self._publish_execution_views(
            resolved_execution,
            generation=self._execution_generation,
            component_generations=self._component_generations,
        )
        graph_mode = (
            GRAPH_NATIVE_COMPOSITE1
            if self._multiview_spm_z2 else GRAPH_COMPOSITE_CHILD
        )
        if execution_session is None:
            self._execution_session = bind_execution_session(
                self,
                self._rpu_execution,
                entry_point="LingbotVlaV2Policy",
                supported_components=LINGBOT2_EXECUTION_COMPONENTS,
                validate=self.validate_execution_reconfigure,
                apply=lambda old, new, generation, *, force_rebuild=False: (
                    self.apply_execution_reconfigure(
                        new, generation, rollback_config=old,
                        force_rebuild=force_rebuild,
                    )
                ),
                rollback=lambda old, _new, generation: (
                    self.rollback_execution_reconfigure(old, generation)
                ),
                graph_mode=graph_mode,
            )
        else:
            if execution_session.config != self._rpu_execution:
                raise ValueError(
                    "LingbotVlaV2Policy: inherited execution session config "
                    "does not match the runtime config"
                )
            if execution_session.graph_mode != graph_mode:
                raise ValueError(
                    "LingbotVlaV2Policy: inherited execution session graph "
                    "mode does not match the runtime lifecycle"
                )
            execution_session.register_config_view(self)
            self._execution_session = weakref.proxy(execution_session)
        for child, component in (
            (self._text, LINGBOT2_TEXT_COMPONENT),
            (self._visual, LINGBOT2_VISION_COMPONENT),
            (self, LINGBOT2_ACTION_COMPONENT),
        ):
            self._execution_session._bind_planner_owner(child, component)

    def _resolve_execution(self, value, *, entry_point: str):
        return resolve_lingbot2_execution_components(
            value, entry_point=entry_point
        )

    def _publish_execution_views(
        self, resolved, *, generation: int, component_generations=None,
    ) -> None:
        root, vision, text, action = resolved
        self._rpu_execution = root
        self._vision_execution = vision
        self._text_execution = text
        self._action_execution = action
        self._execution_generation = int(generation)
        if component_generations is not None:
            self._component_generations = dict(component_generations)
        self._execution_components = {
            LINGBOT2_VISION_COMPONENT: vision,
            LINGBOT2_TEXT_COMPONENT: text,
            LINGBOT2_ACTION_COMPONENT: action,
        }
        for child, component, config in (
            (self._visual, LINGBOT2_VISION_COMPONENT, vision),
            (self._text, LINGBOT2_TEXT_COMPONENT, text),
        ):
            state = vars(child)
            state["_fmb_execution_component_id"] = component
            state["_fmb_execution_component_config"] = config
            state["_fmb_execution_generation"] = int(
                self._component_generations[component]
            )
        # The sparse action expert is a native handle, not a Python module.
        # Keep its stable child identity and view on the owning policy.
        self._action_execution_component_id = LINGBOT2_ACTION_COMPONENT
        self._action_execution_generation = int(
            self._component_generations[LINGBOT2_ACTION_COMPONENT]
        )
        vars(self._text)["_rpu_execution"] = text

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
        """Retire physical Z2, all Python graphs, then native siblings once."""
        if getattr(self, "_cleanup_ok", None) is False:
            raise RuntimeError("LingBot2 retirement already failed; restart the process")
        if getattr(self, "_closed", False):
            return getattr(self, "_cleanup_ok", False) is True
        text = getattr(self, "_text", None)
        visual = getattr(self, "_visual", None)
        try:
            from rpu_backend.api._execution import _require_execution_process_safe

            _require_execution_process_safe()
            # Validate every child before releasing any sibling. Dead child
            # callbacks mean parent ownership, not successful native destroy.
            children = []
            for child, name, op in (
                (text, "decoder", "causal_decoder_destroy"),
                (visual, "vision", "qwen3vl_vision_destroy"),
            ):
                handle = getattr(child, f"_rpu_{name}_handle", None)
                resource = getattr(child, f"_rpu_{name}_retirement_state", None)
                if handle is not None:
                    if (resource is None or resource.handle != handle
                            or resource.failed is not None
                            or getattr(child, "_lingbot2_retirement_owner", None) is not self
                            or resource.parent is None
                            or (resource.parent() is not None and resource.parent() is not self)
                            or (resource.owner() is not None and resource.owner() is not child)
                            or resource.finalizer.alive):
                        raise RuntimeError(f"LingBot2 {name} retirement ownership changed")
                    children.append((child, name, handle, resource, op))
                elif resource is not None and resource.handle is not None:
                    raise RuntimeError(f"LingBot2 {name} retirement handle was lost")
            z2 = getattr(self, "_z2_state", None)
            if not _close_multiview_spm_z2_state(z2):
                raise (z2 or {}).get("retirement_error") or RuntimeError(
                    "LingBot2 physical Z2 retirement failed")
            graphs = [getattr(owner, attr, None) for owner, attr in (
                (self, "_dgc"), (self, "_egc"), (self, "_vgc"),
                (text, "_rpu_text_graph_cache"),
                (text, "_rpu_decoder_graph_cache"),
                (visual, "_rpu_vision_graph_cache"),
            )]
            for _, _, _, resource, _ in children:
                graphs.extend(resource.graphs)
            seen = set()
            for graph in graphs:
                if graph is not None and id(graph) not in seen:
                    seen.add(id(graph))
                    graph.clear()
                    if not graph.cache_invariant_ok():
                        raise RuntimeError("LingBot2 GraphCache invariant failed")
            if getattr(self, "_exp", None) is not None:
                torch.ops.rpu.lingbot_v2_moe_destroy(self._exp)
                self._exp = None
            for child, name, handle, resource, op in children:
                getattr(torch.ops.rpu, op)(handle)
                resource._native_destroyed(handle)
                # Cyclic GC may already have cleared resource.owner's weakref.
                vars(child)[f"_rpu_{name}_handle"] = None
                vars(child).pop("_lingbot2_retirement_owner", None)
            self._drop_device_state()
            self._closed = True
            self._cleanup_ok = True
            return True
        except BaseException as error:
            _poison_lingbot2_retirement(self, error)
            raise

    def _close_without_session(self) -> None:
        """Retire resources while the caller owns the execution session."""
        self._retire_resources()
        if getattr(self, "_live_slot_released", False):
            return
        from rpu_backend.api.causal_lm import _release_live_instance

        _release_live_instance(self)
        self._live_slot_released = True

    def close(self) -> None:
        """Release graph/native resources and the process-wide policy slot."""
        session = getattr(self, "_execution_session", None)
        if session is None:
            self._close_without_session()
            return
        try:
            shutdown = session.shutdown
        except ReferenceError:
            # A facade-owned runtime keeps only a weak session proxy. If the
            # facade is already gone, no facade execution can still be active.
            self._close_without_session()
        else:
            try:
                shutdown(self._close_without_session)
            except BaseException as error:
                from rpu_backend.api import _execution

                if _execution._UNSAFE_PROCESS_REASON is not None:
                    _poison_lingbot2_retirement(self, error)
                raise
        if not getattr(self, "_closed", False):
            error = RuntimeError("LingBot2 closed Session still owns native resources")
            _poison_lingbot2_retirement(self, error)
            raise error

    def _gc_retire(self) -> bool:
        if (not getattr(self, "_gc_retirement_enabled", False)
                or getattr(self, "_cleanup_ok", None) is not None):
            return getattr(self, "_cleanup_ok", None) is True
        session = getattr(self, "_execution_session", None)
        try:
            try:
                lock = session._lock if session is not None else nullcontext()
            except ReferenceError:
                session, lock = None, nullcontext()
            with lock:
                if session is not None and session._active:
                    raise RuntimeError("LingBot2 GC during an active forward")
                from rpu_backend.api import causal_lm

                with causal_lm._LIVE_LOCK:
                    live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
                    if live is not None and live is not self:
                        raise RuntimeError("LingBot2 GC after live-owner handoff")
                    self.close()
            return True
        except BaseException as error:
            _poison_lingbot2_retirement(self, error)
            return False

    def __del__(self):
        self._gc_retire()

    # ---------------------------------------------------------------- prefix
    def _prefix_execution_plan(self, real_len):
        """Resolve one REAL prefix through the shared native A6 domain."""
        real_len = int(real_len)
        if self._multiview_spm_z2:
            # Z2 is one NATIVE_COMPOSITE1 parent with three ordered children;
            # its Text child is not a standalone GraphCache owner.
            return plan_prefix_execution(
                real_len, max_seq=self._max_seq,
                suffix_len=N_ACTION + 1, chunk_size=self._chunk_size,
                fixed_execution_len=MULTIVIEW_SPM_Z2_EXECUTION_LEN,
            )
        profile = self._prepared_graph_profile
        prepared_key = None
        if profile is not None:
            try:
                prepared_key = profile["real_to_execution"][real_len]
            except KeyError as exc:
                lo, hi = profile["prefix_length_range"]
                raise RuntimeError(
                    "LingBot2 REAL prefix is outside the prepared range: "
                    f"real_len={real_len}, prepared=[{lo}, {hi}]. No online "
                    "graph BUILD was attempted."
                ) from exc
        # READY freezes admissible results, not the caller's search request.
        # Replacing AUTO with its last winner would invent an external EXACT
        # pin and change the domain bound to an installed cost certificate.
        prefill = self._text_execution.get("prefill", {})
        padding_rows = prefill.get("padding_rows", "auto")
        padding_budget = (
            0 if isinstance(padding_rows, int)
            else int(prefill.get(
                "padding_budget", PREFIX_PADDING_MULTIPLE - 1,
            ))
        )
        exact_chunk_size = self._chunk_size or None

        plan_box = {}
        execution_len, chunk_size = plan_bounded_prefill_execution(
            real_len,
            self._max_seq - (N_ACTION + 1),
            padding_budget,
            execution_owner=self._text,
            execution_component=LINGBOT2_TEXT_COMPONENT,
            execution_stage="prefill",
            execution_native=("causal_decoder", int(self._text._rpu_decoder_handle)),
            alignment=PREFIX_PADDING_MULTIPLE,
            padding_rows=padding_rows,
            exact_chunk_size=exact_chunk_size,
            resolve_stage_domain=lambda length: (
                torch.ops.rpu.causal_decoder_resolve_prefill_stage_domain(
                    self._text._rpu_decoder_handle,
                    int(length), 0, True, 0, 1, 4, real_len,
                )
            ),
            physical_metadata=(
                (f"component:{LINGBOT2_TEXT_COMPONENT}", 1),
                ("execution_generation", self._component_generations[LINGBOT2_TEXT_COMPONENT]),
            ),
            request_id="lingbot2:prefill",
            plan_result_sink=lambda result: plan_box.__setitem__("result", result),
            graph_mode=GRAPH_COMPOSITE_CHILD,
            queue_owner_id=int(self._text._rpu_decoder_handle),
            lease_owner_id=int(self._text._rpu_decoder_handle),
            plan_signature=(True, 0, 1, 4),
            graph_cache=self._vgc if self._vlm_replay else None,
        )
        if prepared_key is not None and (execution_len, chunk_size) != tuple(prepared_key):
            raise RuntimeError(
                "LingBot2 planner changed a prepared execution profile; "
                "rebuild the graphs before replay. No online graph BUILD was attempted."
            )
        return PrefixExecutionPlan(
            real_len, int(execution_len), int(chunk_size), plan_box["result"]
        )

    def _action_execution_plan(self, prefix_len, rope_offset):
        """Admit the fixed C64 action child through the same authority."""
        requested = self._action_execution.get("action", {}).get(
            "chunk_size", "auto"
        )
        if requested != "auto" and int(requested) != MULTIVIEW_SPM_Z2_ACTION_CHUNK:
            raise PlannerRejectError(
                "EXACT_MISMATCH",
                stage="ACTION",
                requested=int(requested),
                resolved=MULTIVIEW_SPM_Z2_ACTION_CHUNK,
                detail="LingBot2 action suffix has one fixed C64 tile",
            )
        exact_chunk = None if requested == "auto" else int(requested)
        plan_box = {}
        execution_len, resolved = plan_bounded_prefill_execution(
            N_ACTION + 1,
            N_ACTION + 1,
            0,
            execution_owner=self,
            execution_component=LINGBOT2_ACTION_COMPONENT,
            execution_stage="action",
            execution_native=("lingbot_v2_moe", int(self._exp)),
            position=int(prefix_len),
            alignment=1,
            padding_rows=0,
            exact_chunk_size=exact_chunk,
            resolve_stage_domain=lambda length: (
                torch.ops.rpu.lingbot_v2_moe_resolve_action_stage_domain(
                    self._exp, int(length), int(prefix_len),
                    int(prefix_len) + int(length), int(rope_offset),
                )
            ),
            physical_metadata=(
                (f"component:{LINGBOT2_ACTION_COMPONENT}", 1),
                ("execution_generation", self._component_generations[LINGBOT2_ACTION_COMPONENT]),
            ),
            request_id="lingbot2:action_expert:action",
            plan_result_sink=lambda result: plan_box.__setitem__(
                "result", result
            ),
            graph_mode=GRAPH_COMPOSITE_CHILD,
            queue_owner_id=int(self._exp),
            lease_owner_id=int(self._exp),
            plan_signature=(int(rope_offset),),
            graph_cache=(self._dgc if self._denoise_unroll else self._egc) if self._expert_replay else None,
        )
        result = plan_box["result"]
        if (
            execution_len != N_ACTION + 1
            or resolved != 64
            or result.selected is None
            or not result.selected.stage_tuple.physical_descriptor
        ):
            raise RuntimeError(
                "LingBot2 action native planner drift: "
                f"execution={execution_len}, chunk={resolved}"
            )
        return result

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
            plan = self._prefix_execution_plan(real_len)
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

    def _vision_execution_plans(self, metadata):
        """Plan every Qwen3-VL dispatch group before its first RPU op."""
        image_count = len(metadata["keep_img"])
        grid_words = metadata["grid_key"]
        grids = tuple(
            tuple(grid_words[index:index + 3])
            for index in range(0, len(grid_words), 3)
        )
        if len(grids) != image_count:
            raise RuntimeError("LingBot2 Vision grid metadata is inconsistent")
        patches_per_image = int(metadata["image_shape"][1])
        cache_capacity = int(self._visual._rpu_vision_kv_cache.max_seq_len)
        batch_on = self._visual._rpu_vision_batch_enabled
        batch_cap = self._visual._rpu_vision_batch_cap if batch_on else 1
        fused_merger = bool(getattr(
            self._visual, "_rpu_vision_fused_merger", False
        ))
        requested = self._vision_execution.get("vision", {}).get(
            "chunk_size", "auto"
        )
        exact_chunk = None if requested == "auto" else int(requested)

        def minibatch_supported(count):
            blocks, remainder = divmod(patches_per_image, 128)
            grid_x = blocks * count
            return (
                remainder == 0
                and blocks >= 2
                and (grid_x <= 8 or grid_x % 8 == 0)
            )

        groups = []
        first = 0
        while first < image_count:
            if exact_chunk is None:
                count = 1
                while (
                    count < batch_cap
                    and first + count < image_count
                    and grids[first + count] == grids[first]
                    and (count + 1) * patches_per_image <= cache_capacity
                    and minibatch_supported(count + 1)
                ):
                    count += 1
            else:
                run = 1
                while (
                    first + run < image_count
                    and grids[first + run] == grids[first]
                ):
                    run += 1
                candidates = []
                for candidate in range(1, min(run, 3) + 1):
                    packed = candidate * patches_per_image
                    if (
                        run % candidate
                        or packed > cache_capacity
                        or (candidate > 1 and not minibatch_supported(candidate))
                    ):
                        continue
                    full_chunk = ((packed + 15) // 16) * 16
                    if exact_chunk == 144:
                        if candidate == 1 and not fused_merger:
                            candidates.append(candidate)
                    elif full_chunk == exact_chunk:
                        candidates.append(candidate)
                if not candidates:
                    raise PlannerRejectError(
                        "EXACT_MISMATCH",
                        stage="VISION",
                        requested=exact_chunk,
                        detail=(
                            "LingBot2 Vision cannot realize the requested "
                            f"chunk for group@{first}"
                        ),
                    )
                count = max(candidates)
            groups.append((first, count))
            first += count

        plans = []
        packed_offset = 0
        handle = int(self._visual._rpu_vision_handle)
        for ordinal, (first, count) in enumerate(groups):
            packed = count * patches_per_image
            t, h, w = grids[first]
            group_metadata = (
                ("fused_merger", int(fused_merger)),
                ("grid_h", int(h)),
                ("grid_t", int(t)),
                ("grid_w", int(w)),
                ("group_ordinal", int(ordinal)),
                ("image_batch_count", int(count)),
                ("packed_offset", int(packed_offset)),
                ("patches_per_image", int(patches_per_image)),
            )
            plan_box = {}
            execution_len, resolved = plan_bounded_prefill_execution(
                packed,
                packed,
                0,
                execution_owner=self._visual,
                execution_component=LINGBOT2_VISION_COMPONENT,
                execution_stage="vision",
                execution_native=("qwen3vl_vision", int(handle)),
                position=0,
                alignment=1,
                padding_rows=0,
                exact_chunk_size=exact_chunk,
                resolve_stage_domain=lambda length, count=count: (
                    torch.ops.rpu.qwen3vl_vision_resolve_stage_domain(
                        handle, int(length), int(count)
                    )
                ),
                physical_metadata=(
                    (f"component:{LINGBOT2_VISION_COMPONENT}", 1),
                    ("execution_generation", self._component_generations[LINGBOT2_VISION_COMPONENT]),
                    *group_metadata,
                ),
                request_id=f"lingbot2:vision_encoder:vision:{ordinal}",
                plan_result_sink=lambda result: plan_box.__setitem__(
                    "result", result
                ),
                graph_mode=GRAPH_COMPOSITE_CHILD,
                queue_owner_id=handle,
                lease_owner_id=handle,
                plan_signature=(int(count),),
                graph_cache=self._visual._rpu_vision_graph_cache,
            )
            result = plan_box["result"]
            if (
                execution_len != packed
                or result.selected is None
                or resolved != result.selected.stage_tuple.compute_chunk
                or not result.selected.stage_tuple.physical_descriptor
            ):
                raise RuntimeError(
                    "LingBot2 Vision native child planner drift: "
                    f"group={ordinal}, execution={execution_len}, chunk={resolved}"
            )
            plans.append(result)
            packed_offset += packed
        return tuple(plans)

    @contextmanager
    def _vision_graph_identity(self, plans):
        """Publish parent plan words only for the serialized Vision call."""
        state = vars(self._visual)
        previous = state.get("_rpu_execution_graph_key_words", _MISSING)
        state["_rpu_execution_graph_key_words"] = tuple(
            word for plan in plans for word in plan.graph_key_words()
        )
        try:
            yield
        finally:
            if previous is _MISSING:
                state.pop("_rpu_execution_graph_key_words", None)
            else:
                state["_rpu_execution_graph_key_words"] = previous

    def _validate_vision_execution(self, plans):
        """The executed child must consume the complete preflight selection."""
        receipt = getattr(self._visual, "_rpu_last_execution_plan", {})
        expected = tuple(
            tuple(plan.selected.stage_tuple.physical_descriptor)
            for plan in plans
        )
        if (
            not expected or not all(expected)
            or receipt.get("dry_forward_agreement") is not True
            or receipt.get("authority") != "NATIVE_A6_STAGE_DESCRIPTOR"
            or tuple(receipt.get("physical_descriptors", ())) != expected
            or tuple(receipt.get("dispatch_chunk_sizes", ())) != tuple(
                plan.selected.stage_tuple.compute_chunk for plan in plans
            )
        ):
            raise RuntimeError(
                "LingBot2 Vision dry/forward COMPLETE descriptor drift"
            )

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

    @property
    def composite_execution_plan(self):
        parent = self._z2_state.get("planner_parent")
        children = self._z2_state.get("planner_children")
        if parent is None or children is None:
            return None
        return {
            "parent": parent.as_dict(include_candidates=False),
            "children": tuple(
                {**child.as_dict(include_candidates=False),
                 "component": component, "stage": stage}
                for child, component, stage in zip(children,
                    (LINGBOT2_VISION_COMPONENT, LINGBOT2_TEXT_COMPONENT, LINGBOT2_ACTION_COMPONENT),
                    ("vision", "prefill", "action"), strict=True)
            ),
            "descriptor": tuple(self._z2_state["planner_descriptor"]),
        }

    @property
    def composite_executed_owner(self):
        """Read the actual retained native parent; no candidate is invented."""
        with self._execution_session._lock:
            z2 = self._z2_state
            if self._closed or not self._multiview_spm_z2 or not z2.get("prepared"):
                raise RuntimeError("LingBot2 executed owner requires a live prepared Z2")
            return tuple(tuple(row) for row in
                torch.ops.rpu.lingbot2_multiview_spm_z2_executed_owner(
                    z2["vision_handle"], z2["text_handle"], z2["expert_handle"],
                    z2["plan_hash"], z2["planner_descriptor"],
                    z2["planner_digest_words"],
                ))

    def validate_execution_reconfigure(self, execution_config) -> None:
        """Pure/live-read admission before the session mutates any owner."""
        if self._closed:
            raise RuntimeError("LingBot-VLA-V2 policy is closed")
        z2 = self._z2_state
        if z2.get("poisoned", False):
            raise RuntimeError(
                "LingBot2 Z2 lifecycle is poisoned; restart the process"
            )
        if (
            z2.get("prepare_attempted", False)
            and z2.get("owner_thread") != threading.get_ident()
        ):
            raise RuntimeError(
                "LingBot2 Z2 reconfigure must run on its native owner thread"
            )
        resolved = self._resolve_execution(
            execution_config,
            entry_point="LingbotVlaV2Policy.reconfigure",
        )
        _root, vision, text, action = resolved
        action_chunk = action.get("action", {}).get(
            "chunk_size", "auto"
        )
        if (
            action_chunk != "auto"
            and int(action_chunk) != MULTIVIEW_SPM_Z2_ACTION_CHUNK
        ):
            raise PlannerRejectError(
                "EXACT_MISMATCH",
                stage="ACTION",
                requested=int(action_chunk),
                resolved=MULTIVIEW_SPM_Z2_ACTION_CHUNK,
                detail="LingBot2 action suffix has one fixed C64 tile",
            )

        required = {
            "execution_reconfigure_begin",
            "execution_reconfigure_commit",
            "execution_reconfigure_abort",
            "execution_reconfigure_abort_attempt",
        }
        target_prefill = text.get("prefill", {}).get(
            "chunk_size", "auto"
        )
        target_prefill_native = (
            0 if target_prefill == "auto" else int(target_prefill)
        )
        if (
            not self._multiview_spm_z2
            and target_prefill_native != self._chunk_size
        ):
            required.add("causal_decoder_stage_chunk_size_override")
        if self._multiview_spm_z2 and z2.get("prepare_attempted", False):
            required.add("lingbot2_multiview_spm_z2_reconfigure_begin")
        if not self._multiview_spm_z2:
            target_vision = vision.get("vision", {}).get(
                "chunk_size", "auto"
            )
            current_vision = getattr(
                self._visual, "_rpu_vision_execution_chunk_size", "auto"
            )
            if target_vision != current_vision:
                required.add("qwen3vl_vision_stage_chunk_size")
        missing = sorted(
            name for name in required if not hasattr(torch.ops.rpu, name)
        )
        if missing:
            raise RuntimeError(
                "LingBot2 binary lacks native execution-reconfigure op(s): "
                + ", ".join(missing)
            )

        if not self._multiview_spm_z2:
            return

        handles = (
            z2["vision_handle"], z2["text_handle"], z2["expert_handle"]
        )
        if z2.get("prepare_attempted", False):
            native = _decode_z2_native_descriptor(
                _z2_native_descriptor_from_planner(
                    z2["planner_descriptor"]
                ),
                handles,
            )
        else:
            native = _decode_z2_native_descriptor(
                torch.ops.rpu.lingbot2_multiview_spm_z2_dry_probe(*handles),
                handles,
            )
        proposed_generations = dict(self._component_generations)
        for component, before, after in zip(
            LINGBOT2_EXECUTION_COMPONENTS,
            (self._vision_execution, self._text_execution,
             self._action_execution),
            resolved[1:],
        ):
            if before != after:
                proposed_generations[component] += 1
        for stage, component, logical_len, tile, config in (
            ("vision", LINGBOT2_VISION_COMPONENT, native[5], native[6],
             vision.get("vision", {})),
            ("prefill", LINGBOT2_TEXT_COMPONENT, native[10], native[11],
             text.get("prefill", {})),
            ("action", LINGBOT2_ACTION_COMPONENT, native[17], native[18],
             action.get("action", {})),
        ):
            _plan_fixed_lingbot2_component(
                logical_len=logical_len,
                execution_len=logical_len,
                chunk_size=tile,
                component_id=component,
                stage=stage,
                generation=proposed_generations[component],
                stage_config={
                    "chunk_size": config.get("chunk_size", "auto"),
                },
                queue_owner_id=handles[2],
                lease_owner_id=handles[2],
            )

        padding_rows = text.get("prefill", {}).get(
            "padding_rows", "auto"
        )
        if isinstance(padding_rows, int) and padding_rows > 10:
            raise PlannerRejectError(
                "CAPABILITY", stage="PREFILL", requested=padding_rows,
                limit=10,
                detail="LingBot2 Z2 exact padding has no REAL215..225 request",
            )

    @staticmethod
    def _warm_and_clear_execution_cache(cache) -> None:
        if cache is None:
            return
        cache.begin_warmup()
        cache.clear()
        if not cache.cache_invariant_ok():
            raise RuntimeError(
                "LingBot2 GraphCache invariant failed during reconfigure"
            )

    def _reset_execution_runtime_state(self, changed=None) -> None:
        """Retire only Graph owners affected by changed child plans."""
        if changed is None or self._multiview_spm_z2:
            changed = set(LINGBOT2_EXECUTION_COMPONENTS)
        else:
            changed = set(changed)
        caches = []
        if LINGBOT2_VISION_COMPONENT in changed:
            caches.append(getattr(
                self._visual, "_rpu_vision_graph_cache", None
            ))
        if LINGBOT2_TEXT_COMPONENT in changed:
            caches.extend((
                self._vgc,
                getattr(self._text, "_rpu_text_graph_cache", None),
                getattr(self._text, "_rpu_decoder_graph_cache", None),
            ))
        if (
            LINGBOT2_ACTION_COMPONENT in changed
            or LINGBOT2_TEXT_COMPONENT in changed
        ):
            caches.extend((self._dgc, self._egc))
        seen = set()
        for cache in caches:
            if cache is not None and id(cache) not in seen:
                seen.add(id(cache))
                self._warm_and_clear_execution_cache(cache)

        if (
            LINGBOT2_ACTION_COMPONENT in changed
            or LINGBOT2_TEXT_COMPONENT in changed
        ):
            self._restore_cold_suffix_rope()
            self._mask_cache.clear()
        self._prepared_graph_profile = None
        self._graphs_ready = False
        self._last_execution_plan = {}
        if (
            not self._multiview_spm_z2
            and LINGBOT2_TEXT_COMPONENT in changed
        ):
            self._pfx_memo = None
            self._pfx_memos.clear()
            self._pos_ids_src = None
            self._pos_rpu = None
        elif self._multiview_spm_z2:
            # Prompt refresh is multi-write.  After a confirmed native close,
            # discard both its key and owners so a partial A->B update can
            # never masquerade as the old A payload on the next cold forward.
            self._z2_prompt_key = None
            self._z2_prompt_metadata = None
            self._z2_text_hidden = None
            self._z2_position_ids = None
            self._z2_rope_cos_il = None
            self._z2_rope_sin_il = None

    def _poison_z2_execution(self) -> None:
        """Keep the native-composite and public session states consistent."""
        self._z2_state["poisoned"] = True
        warming = (
            getattr(self, "_prepared_graph_profile", None) is not None
            and not getattr(self, "_graphs_ready", False)
        )
        if not warming:
            session = getattr(self, "_execution_session", None)
            if session is not None:
                session.poison()

    def _begin_z2_execution_reconfigure(self) -> int:
        z2 = self._z2_state
        attempt_token = next(_Z2_RECONFIGURE_ATTEMPTS)
        token = None
        try:
            token = int(
                torch.ops.rpu.lingbot2_multiview_spm_z2_reconfigure_begin(
                    z2["vision_handle"], z2["text_handle"],
                    z2["expert_handle"], z2["plan_hash"],
                    z2["planner_descriptor"], z2["planner_digest_words"],
                    attempt_token,
                )
            )
            z2["reconfigure_token"] = token
            z2["prepared"] = False
            z2["prepare_attempted"] = False
            z2["plan_hash"] = None
            z2["prepare_attempt_token"] = None
            return token
        except BaseException as error:
            aborted_active_attempt = False
            try:
                if token is None:
                    aborted_active_attempt = bool(
                        torch.ops.rpu.execution_reconfigure_abort_attempt(
                            attempt_token
                        )
                    )
                else:
                    torch.ops.rpu.execution_reconfigure_abort(token)
            except BaseException as abort_error:
                if hasattr(error, "add_note"):
                    error.add_note(
                        "LingBot2 Z2 reconfigure begin recovery failed: "
                        f"{abort_error!r}"
                    )
            else:
                z2["reconfigure_token"] = None
                if token is not None or aborted_active_attempt:
                    z2["prepared"] = False
                    z2["prepare_attempted"] = False
                    z2["plan_hash"] = None
                    z2["prepare_attempt_token"] = None
            raise

    def apply_execution_reconfigure(
        self, execution_config, generation, *, rollback_config=None,
        force_native=False, force_rebuild=False,
    ) -> None:
        """Refresh existing handles; the next call builds generation ``generation``."""
        self._execution_reconfigure_journal = {"mutation_started": False}
        self.validate_execution_reconfigure(execution_config)
        old_resolved = self._resolve_execution(
            self._rpu_execution if rollback_config is None else rollback_config,
            entry_point="LingbotVlaV2Policy.reconfigure rollback",
        )
        new_resolved = self._resolve_execution(
            execution_config,
            entry_point="LingbotVlaV2Policy.reconfigure",
        )
        component_ids = tuple(LINGBOT2_EXECUTION_COMPONENTS)
        changed = {
            component
            for component, old, new in zip(
                component_ids, old_resolved[1:], new_resolved[1:]
            )
            if old != new
        }
        if force_native or force_rebuild:
            changed.update(component_ids)
        component_generations = dict(self._component_generations)
        for component in changed:
            component_generations[component] += 1
        if not changed and not force_native:
            self._execution_reconfigure_journal = {
                "resolved": old_resolved,
                "generation": self._execution_generation,
                "component_generations": dict(self._component_generations),
                "publication_only": True,
                "mutation_started": True,
            }
            self._publish_execution_views(
                new_resolved,
                generation=generation,
                component_generations=component_generations,
            )
            return

        z2 = self._z2_state
        _root, vision, text, _action = new_resolved
        prefill_chunk = text.get("prefill", {}).get(
            "chunk_size", "auto"
        )
        vision_chunk = vision.get("vision", {}).get(
            "chunk_size", "auto"
        )
        native_prefill_chunk = (
            0 if prefill_chunk == "auto" else int(prefill_chunk)
        )
        native_vision_chunk = (
            0 if vision_chunk == "auto" else int(vision_chunk)
        )
        current_vision_chunk = getattr(
            self._visual, "_rpu_vision_execution_chunk_size", "auto"
        )
        old_plan = {
            key: z2.get(key)
            for key in (
                "planner_descriptor", "planner_digest",
                "planner_digest_words", "planner_parent",
                "planner_children",
            )
        }
        self._execution_reconfigure_journal = {
            "resolved": old_resolved,
            "generation": self._execution_generation,
            "component_generations": dict(self._component_generations),
            "chunk_size": self._chunk_size,
            "vision_chunk": current_vision_chunk,
            "plan": old_plan,
            "changed": set(changed),
            "mutation_started": False,
        }
        native_token = None
        if self._multiview_spm_z2 and z2.get("prepare_attempted", False):
            native_token = self._begin_z2_execution_reconfigure()
            self._execution_reconfigure_journal["mutation_started"] = True

        needs_native = (
            native_token is not None
            or (
                not self._multiview_spm_z2
                and (
                    force_native
                    or LINGBOT2_TEXT_COMPONENT in changed
                    or LINGBOT2_VISION_COMPONENT in changed
                )
            )
        )
        if needs_native:
            with native_execution_reconfigure(
                torch.ops.rpu, token=native_token
            ) as token:
                if (
                    not self._multiview_spm_z2
                    and (force_native or LINGBOT2_TEXT_COMPONENT in changed)
                ):
                    torch.ops.rpu.causal_decoder_stage_chunk_size_override(
                        self._text._rpu_decoder_handle,
                        token,
                        native_prefill_chunk,
                    )
                    self._execution_reconfigure_journal[
                        "mutation_started"
                    ] = True
                if (
                    not self._multiview_spm_z2
                    and (force_native or LINGBOT2_VISION_COMPONENT in changed)
                ):
                    torch.ops.rpu.qwen3vl_vision_stage_chunk_size(
                        self._visual._rpu_vision_handle,
                        token,
                        native_vision_chunk,
                    )
                    self._execution_reconfigure_journal[
                        "mutation_started"
                    ] = True

        z2["reconfigure_token"] = None
        # Record committed native controls before fallible cache cleanup so a
        # session rollback stages the old values instead of treating them as
        # unchanged.
        self._chunk_size = (
            0 if self._multiview_spm_z2 else native_prefill_chunk
        )
        self._text._rpu_chunk_size = self._chunk_size
        if not self._multiview_spm_z2:
            self._visual._rpu_vision_execution_chunk_size = vision_chunk

        self._execution_reconfigure_journal["mutation_started"] = True
        self._reset_execution_runtime_state(changed)
        self._publish_execution_views(
            new_resolved,
            generation=generation,
            component_generations=component_generations,
        )
        # A retained selected descriptor is not the new generation's AUTO
        # oracle. After retirement, the next forward must obtain fresh native
        # admission before applying the new request/cost/calibration selection.
        z2.update({
            "planner_descriptor": None,
            "planner_digest": None,
            "planner_digest_words": None,
            "planner_parent": None,
            "planner_children": None,
        })
        # Retain undo state through the shared session's config publication.

    def rollback_execution_reconfigure(self, execution_config, generation) -> None:
        """Restore a closed native-composite owner from its reconfigure journal."""
        journal = self._execution_reconfigure_journal
        if journal is None:
            return
        if not journal.get("mutation_started", True):
            self._execution_reconfigure_journal = None
            return
        if journal.get("publication_only", False):
            self._publish_execution_views(
                journal["resolved"],
                generation=journal["generation"],
                component_generations=journal["component_generations"],
            )
            self._execution_reconfigure_journal = None
            return

        z2 = self._z2_state
        pending_abort_error = None
        pending_token = z2.get("reconfigure_token")
        if pending_token is not None:
            try:
                torch.ops.rpu.execution_reconfigure_abort(pending_token)
            except BaseException as error:
                # Commit may already have completed before Python observed an
                # interrupt.  The compensating transaction below distinguishes
                # that stale-token case from a still-active failed abort.
                pending_abort_error = error
            z2["reconfigure_token"] = None
        try:
            changed = journal.get(
                "changed", set(LINGBOT2_EXECUTION_COMPONENTS)
            )
            native_token = (
                self._begin_z2_execution_reconfigure()
                if z2.get("prepare_attempted", False) else None
            )
            needs_native_compensation = (
                native_token is not None
                or (
                    not self._multiview_spm_z2
                    and bool(changed.intersection({
                        LINGBOT2_TEXT_COMPONENT,
                        LINGBOT2_VISION_COMPONENT,
                    }))
                )
            )
            if needs_native_compensation:
                with native_execution_reconfigure(
                    torch.ops.rpu, token=native_token
                ) as token:
                    if not self._multiview_spm_z2:
                        if LINGBOT2_TEXT_COMPONENT in changed:
                            torch.ops.rpu.causal_decoder_stage_chunk_size_override(
                                self._text._rpu_decoder_handle,
                                token,
                                int(journal["chunk_size"]),
                            )
                        if LINGBOT2_VISION_COMPONENT in changed:
                            old_vision = journal.get("vision_chunk", "auto")
                            torch.ops.rpu.qwen3vl_vision_stage_chunk_size(
                                self._visual._rpu_vision_handle,
                                token,
                                0 if old_vision == "auto" else int(old_vision),
                            )
            z2["reconfigure_token"] = None
            self._reset_execution_runtime_state(journal.get("changed"))
            z2.update({key: None for key in journal["plan"]}
                      if self._multiview_spm_z2 else journal["plan"])
            z2.update({
                "prepared": False,
                "prepare_attempted": False,
                "plan_hash": None,
            })
        except BaseException as error:
            if pending_abort_error is not None and hasattr(error, "add_note"):
                error.add_note(
                    "pending LingBot2 Z2 reconfigure abort failed: "
                    f"{pending_abort_error!r}"
                )
            z2["poisoned"] = True
            raise

        self._chunk_size = int(journal["chunk_size"])
        self._text._rpu_chunk_size = self._chunk_size
        old_vision = journal.get("vision_chunk", "auto")
        if not self._multiview_spm_z2:
            self._visual._rpu_vision_execution_chunk_size = old_vision
        resolved = journal.get("resolved")
        if resolved is None:
            resolved = self._resolve_execution(
                journal.get("config", execution_config),
                entry_point="LingbotVlaV2Policy.reconfigure rollback",
            )
        self._publish_execution_views(
            resolved,
            generation=journal["generation"],
            component_generations=journal.get(
                "component_generations", self._component_generations
            ),
        )
        self._execution_reconfigure_journal = None

    def _validate_z2_planner_request(self, metadata):
        if (
            metadata["keep_img"] != (0, 1, 2)
            or metadata["grid_key"] != (1, 16, 16) * 3
            or metadata["image_shape"][:2] != (3, 256)
            or metadata["merged_per_img"] != 64
            or not MULTIVIEW_SPM_Z2_MIN_REAL_LEN <= metadata["real_len"]
            <= MULTIVIEW_SPM_Z2_EXECUTION_LEN
        ):
            raise PlannerRejectError(
                "CAPABILITY",
                stage="COMPOSITE",
                requested=int(metadata["real_len"]),
                limit=225,
                detail="LingBot2 Z2 admits exactly three N256 cameras and REAL215..225",
            )
        padding = MULTIVIEW_SPM_Z2_EXECUTION_LEN - int(metadata["real_len"])
        text_execution = getattr(self, "_text_execution", None)
        if text_execution is None:
            text_execution = self._resolve_execution(
                self._rpu_execution,
                entry_point="LingBot2 Z2 planner request",
            )[2]
        prefill = text_execution.get("prefill", {})
        requested_padding = prefill.get("padding_rows", "auto")
        if isinstance(requested_padding, int) and requested_padding != padding:
            raise PlannerRejectError(
                "EXACT_MISMATCH",
                stage="PREFILL",
                requested=requested_padding,
                resolved=padding,
                detail="fixed Z2 P225 requires a different padding row count",
            )
        if "padding_budget" in prefill and padding > int(
            prefill["padding_budget"]
        ):
            raise PlannerRejectError(
                "CAPABILITY",
                stage="PREFILL",
                requested=padding,
                limit=int(prefill["padding_budget"]),
                detail="fixed Z2 P225 exceeds the configured padding budget",
            )

    def _ensure_z2_composite_plan(self, metadata):
        self._validate_z2_planner_request(metadata)
        existing = self._z2_state.get("planner_parent")
        if existing is not None:
            return existing

        plan = self._build_z2_composite_plan(
            self._rpu_execution,
            self._execution_generation,
            component_generations=self._component_generations,
        )
        self._z2_state.update(plan)
        return plan["planner_parent"]

    def _build_z2_composite_plan(
        self, execution, generation, *, native=None,
        component_generations=None,
    ):
        """Build one immutable planner identity without publishing it."""

        z2 = self._z2_state
        root, vision, text, action = self._resolve_execution(
            execution,
            entry_point="LingBot2 Z2 execution plan",
        )
        del root
        generations = (
            self._component_generations
            if component_generations is None else component_generations
        )
        handles = (
            z2["vision_handle"], z2["text_handle"], z2["expert_handle"]
        )
        if native is None:
            native = _decode_z2_native_descriptor(
                torch.ops.rpu.lingbot2_multiview_spm_z2_dry_probe(*handles),
                handles,
            )
        else:
            native = _decode_z2_native_descriptor(native, handles)
        vision_descriptor, text_descriptor = _z2_native_child_descriptors(native)
        children = (
            _plan_z2_native_child(
                vision_descriptor,
                logical_len=native[5],
                component_id=LINGBOT2_VISION_COMPONENT,
                stage="vision",
                generation=generations[LINGBOT2_VISION_COMPONENT],
                requested_chunk=vision.get("vision", {}).get("chunk_size", "auto"),
                execution_owner=self._visual, native_prefix="qwen3vl_vision",
                native_handle=handles[0], parent_handle=handles[2],
                physical_metadata=(
                    ("child_ordinal", native[3]),
                    ("occurrence_count", native[4]),
                    ("layout_hash_hi", (native[8] >> 32) & 0xFFFFFFFF),
                    ("layout_hash_lo", native[8] & 0xFFFFFFFF),
                    ("temporary_bytes", native[7]),
                ),
            ),
            _plan_z2_native_child(
                text_descriptor,
                logical_len=native[10],
                component_id=LINGBOT2_TEXT_COMPONENT,
                stage="prefill",
                generation=generations[LINGBOT2_TEXT_COMPONENT],
                requested_chunk=text.get("prefill", {}).get("chunk_size", "auto"),
                execution_owner=self._text, native_prefix="causal_decoder",
                native_handle=handles[1], parent_handle=handles[2],
                physical_metadata=(
                    ("allocation_chunk", native[12]),
                    ("child_ordinal", native[9]),
                    ("layout_hash_hi", (native[15] >> 32) & 0xFFFFFFFF),
                    ("layout_hash_lo", native[15] & 0xFFFFFFFF),
                    ("temporary_bytes", native[14]),
                ),
            ),
            # This is the parent's reserved expert ABI, not the independently
            # executed Action plan, which is published by _action_execution_plan.
            _plan_fixed_lingbot2_component(
                logical_len=native[17],
                execution_len=native[17],
                chunk_size=native[18],
                component_id=LINGBOT2_ACTION_COMPONENT,
                stage="action",
                generation=generations[LINGBOT2_ACTION_COMPONENT],
                stage_config={
                    "chunk_size": action.get("action", {}).get(
                        "chunk_size", "auto"
                    ),
                },
                physical_metadata=(
                    ("child_ordinal", native[16]),
                    ("layout_hash", native[21]),
                    ("profile_hash", native[20]),
                    ("profile_kind", native[19]),
                    ("temporary_bytes", native[22]),
                ),
                queue_owner_id=handles[2],
                lease_owner_id=handles[2],
            ),
        )
        # Forward receives exactly the shared planner's selected child bytes,
        # including any admitted native calibration route, not a second AUTO.
        vision_descriptor, text_descriptor = (
            child.selected.stage_tuple.physical_descriptor for child in children[:2]
        )
        header = list(native[:_Z2_NATIVE_HEADER_WORDS])
        for index, child in zip((8, 15), children[:2]):
            metadata = dict(child.selected.stage_tuple.physical_metadata)
            layout = (metadata["layout_hash_hi"] << 32) | metadata["layout_hash_lo"]
            header[index] = layout if layout < 1 << 63 else layout - (1 << 64)
        native = (*header, len(vision_descriptor),
                  *vision_descriptor, len(text_descriptor), *text_descriptor)
        descriptor = native + tuple(
            word
            for child in children
            for word in _plan_digest_words(child.plan_digest)
        )
        if len(descriptor) != len(native) + _Z2_CHILD_DIGEST_WORDS:
            raise RuntimeError("LingBot2 Z2 planner descriptor assembly drifted")
        parent_metadata = tuple(
            (f"descriptor_{index:02d}", value)
            for index, value in enumerate(descriptor)
        )
        parent = capability_selected(
            plan_a6(
                MULTIVIEW_SPM_Z2_EXECUTION_LEN,
                tuple_domain={
                    MULTIVIEW_SPM_Z2_EXECUTION_LEN: (
                        StageTuple(
                            native[6], native[11], native[18], parent_metadata,
                            descriptor,
                        ),
                    )
                },
                physical_limit=MULTIVIEW_SPM_Z2_EXECUTION_LEN,
                graph_mode=GRAPH_NATIVE_COMPOSITE1,
                queue_owner_id=handles[2],
                lease_owner_id=handles[2],
            ),
            selection_scope="NATIVE_COMPOSITE1_ORDERED_CHILDREN",
        )
        digest_words = _plan_digest_words(parent.plan_digest)
        if not any(digest_words):
            raise RuntimeError("LingBot2 Z2 planner produced a zero digest")
        return {
            "planner_descriptor": descriptor,
            "planner_digest": parent.plan_digest,
            "planner_digest_words": digest_words,
            "planner_parent": parent,
            "planner_children": children,
        }

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

    @execution_serialized
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
        if self._multiview_spm_z2:
            if not (
                MULTIVIEW_SPM_Z2_MIN_REAL_LEN
                <= lo <= hi <= MULTIVIEW_SPM_Z2_EXECUTION_LEN
            ):
                raise ValueError(
                    "LingBot2 Z2 prepared REAL range must stay within "
                    f"[{MULTIVIEW_SPM_Z2_MIN_REAL_LEN}, "
                    f"{MULTIVIEW_SPM_Z2_EXECUTION_LEN}], got [{lo}, {hi}]"
                )
            for real_len in range(lo, hi + 1):
                self._validate_z2_planner_request({
                    **metadata, "real_len": real_len,
                })
            self._ensure_z2_composite_plan(metadata)
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
                        z2["planner_descriptor"],
                        z2["planner_digest_words"],
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
                        z2["planner_descriptor"],
                        z2["planner_digest_words"],
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
        except BaseException as error:
            z2 = self._z2_state
            if (
                self._multiview_spm_z2
                and z2.get("prepare_attempted", False)
                and not _close_multiview_spm_z2_state(z2)
            ):
                z2["poisoned"] = True
                session = getattr(self, "_execution_session", None)
                if session is not None:
                    session.poison()
                _poison_lingbot2_retirement(self, z2.get("retirement_error") or error)
                error.add_note(
                    "LingBot2 Z2 cleanup failed; child caches and RoPE were "
                    "left untouched"
                )
                raise
            try:
                self._reset_execution_runtime_state()
            except BaseException as cleanup_error:
                z2["poisoned"] = True
                _poison_lingbot2_retirement(self, cleanup_error)
                error.add_note(f"LingBot2 prepare_graphs cleanup also failed: {cleanup_error!r}")
                raise error
            raise
        return self.graph_profile

    def _build_prefix_direct(
        self, images, keep_img, lang_tokens, lang_masks, grid_thw, num_patch,
        prefill_plan,
    ):
        """Build/bind persistent prefix targets before the fused Vision Graph BUILD."""
        batch_cap = self._visual._rpu_vision_batch_cap
        vision_cache = self._visual._rpu_vision_kv_cache
        _validate_direct_prefix_group(
            grid_thw,
            patches_per_image=int(images.shape[1]),
            batch_cap=batch_cap,
            cache_capacity=int(vision_cache.max_seq_len),
        )

        if lang_masks is not None:
            keep = lang_masks.reshape(-1).bool()
            lang_ids = lang_tokens.reshape(-1)[keep].reshape(1, -1)
        else:
            lang_ids = lang_tokens.reshape(1, -1)
        lang_ids = lang_ids.to("cpu", dtype=torch.long)
        lay = build_prefix_layout(
            len(keep_img), num_patch, lang_ids, self._align_cfg, self._ids
        )
        if int(lay.S) != prefill_plan.real_len:
            raise RuntimeError(
                "LingBot2 CPU prefix geometry drifted after planner admission: "
                f"admitted={prefill_plan.real_len}, built={lay.S}"
            )
        plan = prefill_plan

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
                "a6_plan": plan.a6_plan,
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
            # Phase 1 leaves generic pop/clone intact to isolate the twelve
            # removed Python scatters.  The phase-2 private flag skips those
            # stale outputs as well.  In both cases the prefix targets above are
            # the only outputs consumed by LingBot2.
            self._visual(
                pix,
                grid_thw=grid_thw,
                _rpu_prefix_scatter_consume_only=(
                    self._vision_direct_prefix_no_output),
            )
        return (memo["embeds"], memo["position_ids"],
                memo["position_ids_rpu"], memo["dense"], None, None,
                memo["S"], memo["real_S"], memo["p0"], memo["a6_plan"])

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
        current_prompt_key = self._z2_prompt_key
        prompt_changed = prompt_key != current_prompt_key
        next_prompt_metadata = None
        if prompt_changed:
            # Finish every fallible CPU copy before mutating stable RPU owners.
            next_prompt_metadata = {
                "lang_ids": lang_ids.clone(),
                "position_ids_cpu": position_ids_cpu.clone(),
                "rope_cos_il_cpu": rope_cos_il_cpu.clone(),
                "rope_sin_il_cpu": rope_sin_il_cpu.clone(),
                "S": S,
                "real_S": real_S,
                "p0": p0,
            }
        refresh_text_inputs = False
        if current_prompt_key is None:
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
            if prompt_changed:
                self._z2_prompt_key = _Z2_PROMPT_DIRTY
                try:
                    text_hidden_cpu = F.embedding(
                        execution.input_ids, self._z2_embedding_cpu)
                    text_hidden_cpu.masked_fill_(
                        execution.visual_pos_masks.unsqueeze(-1), 0)
                    for segment, token in self._align_tokens.items():
                        begin, end = layout.spans[segment]
                        text_hidden_cpu[0, begin:end] = token.to(torch.float16)
                    self._z2_text_hidden.copy_(text_hidden_cpu)
                    position_changed = (
                        current_prompt_key is _Z2_PROMPT_DIRTY
                        or not torch.equal(
                            prompt_metadata["position_ids_cpu"],
                            position_ids_cpu,
                        )
                    )
                    if position_changed:
                        with torch.inference_mode(False):
                            self._z2_position_ids.copy_(position_ids_cpu)
                            self._z2_rope_cos_il.copy_(rope_cos_il_cpu)
                            self._z2_rope_sin_il.copy_(rope_sin_il_cpu)
                        if z2.get("prepared", False):
                            refresh_text_inputs = True
                except BaseException:
                    self._poison_z2_execution()
                    raise

        if prompt_changed:
            self._z2_prompt_metadata = next_prompt_metadata
            if not refresh_text_inputs:
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
            z2["prepare_attempt_token"] = next(_Z2_PREPARE_ATTEMPTS)
            z2["prepare_attempted"] = True
            try:
                z2["plan_hash"] = int(
                    torch.ops.rpu.lingbot2_multiview_spm_z2_prepare(
                        z2["vision_handle"], z2["text_handle"],
                        z2["expert_handle"], self._z2_position_ids,
                        self._z2_rope_cos_il, self._z2_rope_sin_il,
                        z2["planner_descriptor"],
                        z2["planner_digest_words"],
                        z2["prepare_attempt_token"],
                    )
                )
                z2["prepared"] = True
            except BaseException:
                self._poison_z2_execution()
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
                    z2["planner_descriptor"],
                    z2["planner_digest_words"],
                )
                self._z2_prompt_key = prompt_key
            except BaseException:
                self._poison_z2_execution()
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
                z2["planner_descriptor"],
                z2["planner_digest_words"],
            )
        except BaseException:
            self._poison_z2_execution()
            raise
        # The native Text execution wrote [0,225), but it deliberately bypasses
        # the Python HF wrapper that normally advances this bookkeeping field.
        self._cache.reset_to_position(S)
        return S, real_S, p0

    def _build_prefix(
        self, images, img_masks, lang_tokens, lang_masks, prefill_plan,
        grid_thw=None,
    ):
        """Run the RPU ViT and assemble the prefix EXACTLY as the official embed_prefix does.

        Layout (authoritative YAML): [<vision_start> <image>*P <vision_end>]*n_img ++ lang
                                     ++ <eos>*8 (current_depth) ++ <eos>*8 (future_depth)

        Returns (inputs_embeds [1,S,2560] RPU fp16, position_ids [3,1,S] CPU, dense_ds, S, p0).
        """
        keep_img = [i for i in range(images.shape[0])
                    if img_masks is None or bool(img_masks.reshape(-1)[i])]
        if not keep_img:
            raise ValueError("lingbot2: no valid image (img_masks all False).")

        # rope_grid_thw carries ONE row per VALID image (embed_prefix :714-717): get_rope_index
        # consumes one grid per contiguous <image> run.
        if grid_thw is None:
            # Legacy path: square grid inferred from the patch count. Unchanged.
            g = self._image_grid(images)
            grid_thw = torch.tensor([[1, g, g]] * len(keep_img), dtype=torch.long)
            merged_per_img = (g // self._spatial_merge) ** 2
        else:
            # Official-processor path: the real (t,h,w) grid, which may be non-square.
            grid_thw, merged_per_img = self._check_grid_thw(grid_thw, images, keep_img)
        if self._vision_direct_prefix:
            return self._build_prefix_direct(
                images, keep_img, lang_tokens, lang_masks, grid_thw,
                merged_per_img, prefill_plan)
        if self._prefix_opt and len(keep_img) == images.shape[0]:
            # When every image is retained, reshape the contiguous processor output
            # instead of using advanced indexing to copy the same read-only values.
            pix = images.reshape(-1, images.shape[-1])
            if not pix.is_contiguous():
                pix = pix.contiguous()
        else:
            pix = images[keep_img].reshape(-1, images.shape[-1]).contiguous()
        with torch.no_grad():
            vout = self._visual(pix, grid_thw=grid_thw)
        pooler = vout.pooler_output                                  # [n_img*P, 2560]
        # fp16, mirroring the `pooler.to("cpu", torch.float16)` cast below: the ViT is fed
        # fp32 pixels (the official policy does the same, lingbot_vla_v2_policy.py:71-72),
        # so its outputs come back fp32. scatter_visual_embeds_to_dense inherits the dtype
        # of what it is handed, and causal_decoder_forward then rejects the fp32 dense
        # deepstack tensors with "expected scalar type Half but found Float".
        deepstack_features = [d.to(torch.float16) for d in vout.deepstack_features]

        num_patch = merged_per_img
        if pooler.shape[0] != len(keep_img) * num_patch:
            raise ValueError(f"lingbot2: ViT returned {pooler.shape[0]} tokens, expected "
                             f"{len(keep_img)} x {num_patch}")

        # Prompt constants depend on image count, tokens per image, language tokens
        # and grid. Build their RPU buffers once, then overwrite only each image's
        # contiguous visual rows. Stable addresses support VLM Graph replay.
        # The key also includes which images survive filtering, so dropping different
        # cameras cannot reuse an incompatible prefix layout.
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
                        memo["S"], memo["real_S"], memo["p0"],
                        memo["a6_plan"])

        if lang_masks is not None:
            keep = lang_masks.reshape(-1).bool()
            lang_ids = lang_tokens.reshape(-1)[keep].reshape(1, -1)
        else:
            lang_ids = lang_tokens.reshape(1, -1)
        lang_ids = lang_ids.to("cpu", dtype=torch.long)

        lay = build_prefix_layout(
            len(keep_img), num_patch, lang_ids, self._align_cfg, self._ids
        )
        if int(lay.S) != prefill_plan.real_len:
            raise RuntimeError(
                "LingBot2 CPU prefix geometry drifted after planner admission: "
                f"admitted={prefill_plan.real_len}, built={lay.S}"
            )
        plan = prefill_plan
        execution = pad_prefix_layout_for_execution(
            lay, plan.execution_len, pad_token_id=self._ids.eos
        )
        real_S = int(lay.S)
        S = int(execution.execution_len)

        # ---- M-RoPE position ids (build_prefix_position_ids :265-273) ----
        # transformers 5.x takes an explicit `mm_token_type_ids` to mark the vision spans;
        # 4.57.x has no such parameter and derives the same spans by scanning `input_ids`
        # for the image/vision special tokens. Both produce the same position_ids for our
        # prefix (verified end-to-end: the action tensor is bit-identical across the two
        # versions), so we simply pass the argument only where it exists. Probed via the
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
        # exactly as it was; :345 does the single final .to("rpu").
        embeds_cpu = self._vlm.get_input_embeddings()(
            execution.input_ids.to("rpu")
        ).to("cpu", torch.float16)
        # <image> placeholders  -> ViT patch embeddings. <vision_start>/<vision_end> KEEP their
        # real token embeddings (embed_special_token is just embed_language_tokens, :260-263).
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
                        # zeros on CPU then hop (prefix.py pitfall B-1: never
                        # torch.zeros(device='rpu')).
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
                    "a6_plan": plan.a6_plan,
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
                memo["a6_plan"] = plan.a6_plan
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
            S, real_S, p0, plan.a6_plan,
        )

    def _vlm_fill(self, embeds, position_ids, position_ids_rpu, dense_ds,
                  rope_cos_il, rope_sin_il, S, a6_plan):
        """ONE VLM prefill writes the prefix KV into the shared cache [0, S)."""
        if a6_plan is None or a6_plan.selected is None:
            raise RuntimeError("LingBot2 generic prefill is missing its A6 plan")
        chunk_size = int(a6_plan.selected.stage_tuple.compute_chunk)
        if int(a6_plan.selected.execution_len) != int(S):
            raise RuntimeError("LingBot2 generic prefill A6 execution length drifted")
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
            shapes=[S, VLM_HID, chunk_size],
            dyn_dims=[VLM_LAYERS, S, 0, PREFIX_POLICY_VERSION,
                      *a6_plan.graph_key_words()],
            dtypes=[torch.float16],
        )
        with self._vgc.capture(sig):
            _run_causal_decoder_forward(
                self._text, self._text._rpu_decoder_handle,
                input_ids=None, inputs_embeds=embeds,
                attention_mask=None, position_ids=position_ids_rpu,
                past_key_values=self._cache, use_cache=True, return_dict=True,
                deepstack_dense_visual_embeds=dense_ds,
                rope_cos_il=rope_cos_il, rope_sin_il=rope_sin_il,
                prefill_plan=(S, chunk_size, a6_plan))
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
        # Distinguish an incomplete write from a completely installed p0.  A
        # failed cold restore must remain retryable instead of short-circuiting
        # on p0 == 0 with only one table restored.
        self._ready_rope_p0 = _READY_ROPE_DIRTY
        self._expert_rope_cos.narrow(0, 0, N_ACTION + 1).copy_(cos_cpu)
        self._expert_rope_sin.narrow(0, 0, N_ACTION + 1).copy_(sin_cpu)
        self._ready_rope_p0 = p0
        return 0

    def _restore_cold_suffix_rope(self):
        """Restore the full-table rows overwritten by READY's offset-0 window."""
        if self._ready_rope_p0 is None:
            return
        self._prepare_ready_suffix_rope(0)
        self._ready_rope_p0 = None

    def _exp_run(
        self, suffix_r, mask_r, S, p0, sig, action_plan,
        adarms_step=None,
    ):
        """position=S drives the KV-insert offset; cos_sin_offset=p0 drives the RoPE base."""
        self._cache.reset_to_position(S)      # freeze prefix [0, S); insert suffix at S
        descriptor = list(
            action_plan.selected.stage_tuple.physical_descriptor
        )
        if not descriptor:
            raise RuntimeError("LingBot2 action plan has no native descriptor")
        with self._egc.capture(sig):
            if self._adarms_direct_schedule:
                if adarms_step is None:
                    raise RuntimeError("lingbot2 direct AdaRMS schedule requires a step index")
                return torch.ops.rpu.lingbot_v2_moe_forward(
                    self._exp, suffix_r, self._cache.k_caches, self._cache.v_caches,
                    mask_r, S, p0, int(adarms_step), descriptor)
            return torch.ops.rpu.lingbot_v2_moe_forward(
                self._exp, suffix_r, self._cache.k_caches, self._cache.v_caches,
                mask_r, S, p0, -1, descriptor)

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

        # P1 order is load-bearing: fp16 -> zero-pad -> single-core col-swizzle -> RPU.
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
                            dt, num_steps, action_plan):
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
            dyn_dims=[*dyn_dims, *action_plan.graph_key_words()],
            dtypes=[torch.float16])
        with self._dgc.capture(sig):
            torch.ops.rpu.lingbot_v2_moe_denoise_unroll_forward(
                self._exp, self._denoise_x0, self._denoise_state,
                self._cache.k_caches, self._cache.v_caches, mask_r,
                self._denoise_x_final, dt, S, p0, num_steps,
                action_plan.selected.stage_tuple.physical_descriptor)
        return self._denoise_x_final[:, 1:, :ACTION_DIM].to("cpu").float()

    # ---------------------------------------------------------------- action
    @execution_serialized
    @torch.no_grad()
    def get_action(self, obs, *, num_steps=NUM_STEPS):
        if getattr(self, "_closed", False):
            raise RuntimeError("LingBot-VLA-V2 policy is closed.")
        if self._z2_state.get("poisoned", False):
            raise RuntimeError(
                "LingBot2 execution lifecycle is poisoned; restart the process"
            )
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
        if self._multiview_spm_z2:
            self._ensure_z2_composite_plan(request_metadata)
            prefill_plan = None
            vision_plans = ()
        else:
            prefill_plan = self._prefix_execution_plan(
                request_metadata["real_len"]
            )
            self._action_execution_plan(prefill_plan.execution_len, 0)
            vision_plans = self._vision_execution_plans(request_metadata)
        self._last_prefix_metadata = dict(request_metadata)
        state = state.float()
        noise = noise.float()

        if not self._vlm_replay:
            self._vgc = rpu_backend.graph.GraphCache(
                runtime_policy=self._graph_runtime_policy
            )
        if not self._expert_replay:
            if self._denoise_unroll:
                self._dgc = rpu_backend.graph.GraphCache(
                    runtime_policy=self._graph_runtime_policy
                )
            else:
                self._egc = rpu_backend.graph.GraphCache(
                    runtime_policy=self._graph_runtime_policy
                )

        if self._multiview_spm_z2:
            S, real_S, p0 = self._fill_prefix_multiview_spm_z2(
                images, img_masks, lang_tokens, lang_masks,
                obs.get("image_grid_thw"),
            )
            embeds = position_ids = position_ids_rpu = dense_ds = None
            rope_cos_il = rope_sin_il = None
        else:
            with self._vision_graph_identity(vision_plans):
                (
                    embeds, position_ids, position_ids_rpu, dense_ds,
                    rope_cos_il, rope_sin_il,
                    S, real_S, p0, prefill_plan,
                ) = self._build_prefix(
                    images, img_masks, lang_tokens, lang_masks,
                    prefill_plan, obs.get("image_grid_thw"),
                )
            self._validate_vision_execution(vision_plans)

        # ── RoPE base vs KV offset ────────────────────────────────────────────────────────────
        # The suffix's RoPE start is p0 = max(prefix M-RoPE position) + 1
        # (modeling_lingbot_vla_v2.py:741-747), which is NOT the KV-insert offset S whenever the
        # prefix holds images — M-RoPE advances only max(h,w)//spatial_merge_size per image, not
        # one per visual token. The C++ honours `rope_position_base_` for non-mrope decoders
        # (rpu_qwen3_model.h:1635/1748; read at rpu_lingbot_v2_moe_model.cpp:427); the trailing
        # `cos_sin_offset` arg is what routes p0 into it. With that arg present we pass
        # position=S + cos_sin_offset=p0 and the image path is fully expressible.
        #
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
                rope_cos_il, rope_sin_il, S, prefill_plan
            )

        graph_rope_offset = self._prepare_ready_suffix_rope(p0)
        action_plan = self._action_execution_plan(S, graph_rope_offset)

        composite_plan = self.composite_execution_plan if self._multiview_spm_z2 else None
        completed_plan = {
            "vision": (
                composite_plan["children"][0]
                if self._multiview_spm_z2 else {
                    "component": LINGBOT2_VISION_COMPONENT,
                    "stage": "vision",
                    "dispatches": tuple(
                        plan.as_dict(include_candidates=False)
                        for plan in vision_plans
                    ),
                    "physical_plan_digests": tuple(
                        plan.physical_plan_digest for plan in vision_plans
                    ),
                    "execution_generation": self._component_generations[
                        LINGBOT2_VISION_COMPONENT
                    ],
                }
            ),
            "prefill": (
                composite_plan["children"][1]
                if self._multiview_spm_z2 else {
                    **prefill_plan.a6_plan.as_dict(include_candidates=False),
                    "component": LINGBOT2_TEXT_COMPONENT,
                    "stage": "prefill",
                    "execution_generation": self._component_generations[
                        LINGBOT2_TEXT_COMPONENT
                    ],
                }
            ),
            # The suffix is executed by the separate _dgc Graph below, not
            # the retained native Vision/Text parent. Publish the very plan
            # handed to that action forward, including its COMPLETE manifest.
            "action": {
                **action_plan.as_dict(include_candidates=False),
                "component": LINGBOT2_ACTION_COMPONENT,
                "stage": "action",
                "execution_generation": self._component_generations[
                    LINGBOT2_ACTION_COMPONENT
                ],
            },
        }

        if self._denoise_unroll:
            result = self._denoise_unroll_run(
                state, noise, mask_r, S, real_S, graph_rope_offset,
                -1.0 / num_steps, num_steps, action_plan,
            )
            self._last_execution_plan = completed_plan
            return result

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
            dyn_dims=[*denoise_dyn_dims, *action_plan.graph_key_words()],
            dtypes=[torch.float16])

        # The caching allocator may reuse the temporary suffix tensors across steps.
        # Keep their ownership local to the current forward.
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
                    action_plan,
                    adarms_step=k if self._adarms_direct_schedule else None)
                # Slice the action rows on device before reading them back and widening.
                # Keep x_t + dt * v_t as separate multiply and add: an alpha-add formulation
                # can fuse them and change floating-point rounding.
                suffix_out = out[:, -N_ACTION:].to("cpu").float()        # [1, 50, 768]
                v_t = F.linear(suffix_out, aout_w, aout_b)               # [1, 50, 55]
                x_t = x_t + dt * v_t
        finally:
            if self._enc_1thread:
                torch.set_num_threads(_nt)
        self._last_execution_plan = completed_plan
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

    Ground truth (transformers/models/qwen3_vl/modeling_qwen3_vl.py, the ONLY 3 register_buffer
    sites in that file, all persistent=False):
      :95-100  Qwen3VLVisionRotaryEmbedding.__init__(dim, theta=10000.0)
                 -> self.dim, self.theta ; register_buffer("inv_freq", ..., persistent=False)
                 real call site :637  Qwen3VLVisionRotaryEmbedding(head_dim // 2)
                 ⚠️ transformers 4.57.x does NOT set self.dim/self.theta — recovered from
                 inv_freq.shape + the ctor default at the call site below.
      :295-309 Qwen3VLTextRotaryEmbedding.__init__(config, device=None)
                 -> self.config ; register_buffer("inv_freq"/"original_inv_freq", persistent=False)
                 real call site :845  Qwen3VLTextRotaryEmbedding(config=config)
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
            # theta is the ctor default on both versions: the only call site in either is
            # `Qwen3VLVisionRotaryEmbedding(head_dim // 2)` (4.57.3:582, 5.3.0:637), which
            # never passes it. So the rebuilt inv_freq is bit-identical to a normal build.
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




# The MLP projections are the bandwidth-dominant weights in each text layer (intermediate ~4x
# hidden). RPU_LINGBOT2_PREFILL_W4A16 int4's ONLY these three; attention q/k/v/o stay int8. This
# matches "only quantise the genuinely bandwidth-bound big matrices" — the small attn projections
# are launch-bound at prefill and int4 buys ~nothing (the denoise B1 lesson).
_TEXT_INT4_NAMES = {"gate_proj", "up_proj", "down_proj"}


def _quantize_text_w4a16(text_model) -> int:
    """Mixed int8/int4: int8 all 7 projections, then re-quantize the MLP three to int4 values
    (int8 container). Packing to uint8 happens later through the pgrp ABI.
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
    """Pack the three text-MLP INT4 projections through the pgrp ABI."""
    import torch.nn as nn
    from rpu_backend.quant.int4_pgrp_pack import pack_int4_per_channel_as_pgrp

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
            scale = module.weight_scale
            packed, packed_scale = pack_int4_per_channel_as_pgrp(
                weight.detach().cpu(), scale.detach().cpu().to(torch.float16),
                partition=partition, num_cores=ATTN_TP,
            )
            module.weight = nn.Parameter(
                packed.reshape(n, k // 2).contiguous().to(weight.device),
                requires_grad=False,
            )
            module._buffers["weight_scale"] = packed_scale.to(scale.device)
            n_packed += 1
    return n_packed

def _quantize_text_w8a16(text_model) -> None:
    """On-the-fly W8A16 for the Qwen3-VL TEXT decoder (the VLM prefill).

    Replace each layer's 7 projection Linears' fp16 weight with
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
    plain fp16 setter and the path is byte-identical to v3.
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
    rpu_execution=None,
    align_cfg=None,
    training_config_path=None,
    execution_session=None,
    execution_generation=0,
    component_generations=None,
):
    """Build the LingBot-VLA V2 RPU runtime.

    Args:
        ckpt:        path to the lingbot-vla-v2-6b checkpoint dir (sharded fp32 safetensors).
        qwen3vl_base: Qwen3-VL-4B-Instruct dir, for the config ONLY (the scaffold is filled from
                     `ckpt`; no base weights are read). Defaults to the shared read-only mount.
        max_seq:     rope-table + KV-cache capacity (prefix S + suffix 51 must fit).
        rpu_execution: cold canonical planner requests for the independent
                     prefill, Vision and action children.
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
    native_cold = _Lingbot2MoeColdConfig.from_env()
    policy._lingbot2_moe_cold_config = native_cold
    _validate_denoise_unroll()
    _validate_adarms_direct_schedule()
    (
        rpu_execution,
        vision_execution,
        text_execution,
        _action_execution,
    ) = resolve_lingbot2_execution_components(
        rpu_execution,
        entry_point="build_lingbot_vla_v2",
    )

    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    # The expert reads the VLM's prefix KV cross-handle out of the shared DDR cache; ddr_flush
    # keeps that cache coherent between the two handles (V1 proved this is required).
    torch.rpu.set_ddr_flush(True)
    # The direct builder leaves process-wide defaults to the caller; the public
    # facade scopes and restores its runtime bundle. KV-insert hybrid routes
    # handle aligned bulk and unaligned head/tail ranges without changing positions.
    # AdaRMS fused broadcast requires compatible persistent per-layer SPM slots.
    #
    # Vision RoPE tables may reside in SPM. Image batching groups equal grids while
    # preserving per-image attention. Fused mergers execute inside the vision Graph
    # and change accumulation from the eager host path to device arithmetic.
    # Patch embedding precision remains a separate explicit runtime choice.
    # Outputs retain independent storage so callers can hold them across requests.
    #
    # Expert W8A16 quantizes only routed gate/up/down weights with per-channel
    # FP16 scales. Router, attention, AdaRMS, shared expert, residual and action
    # head remain FP16. It requires grouped experts and explicit opt-in.

    # Qwen3-VL base dir resolution — CALL time, explicit source for diagnostics.
    # Precedence: explicit qwen3vl_base arg (threaded from from_checkpoint /
    # runtime_env by api/lingbot2.py) > RPU_LINGBOT2_QWEN3VL_BASE env (direct build()
    # callers) > bundled/dev default. Never depends on the import-time constant.
    if qwen3vl_base:
        base, base_src = str(qwen3vl_base), "explicit qwen3vl_base arg"
    else:
        _env = os.environ.get("RPU_LINGBOT2_QWEN3VL_BASE", "").strip()
        if _env:
            base, base_src = _env, "RPU_LINGBOT2_QWEN3VL_BASE env"
        else:
            base, base_src = _resolve_qwen3vl_base(), "bundled/dev default"
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
            "runtime_env={'RPU_LINGBOT2_QWEN3VL_BASE': '<dir>'}; the wheel-bundled default "
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

    # SDK07B limits one process to 4096 live DDR mappings. Exact Z2 retains
    # thousands of expert slices, so pack them with the existing allocator.
    if _env_exact_one("RPU_LINGBOT2_MULTIVIEW_SPM_Z2"):
        torch.rpu.set_caching_allocator(True)

    text_model, vision_model = vlm.language_model, vlm.visual
    # Publish partial ownership before the first irreversible swizzle/RPU
    # allocation. The outer transaction can then retire any text/vision handle
    # installed before a later stage fails.
    policy._vlm = vlm
    policy._text = text_model
    policy._visual = vision_model
    policy._rpu_materialization_started = True
    policy._gc_retirement_enabled = True
    policy._rpu_swizzle_started = True

    # ---- Step A/C: swizzle text linears on CPU, THEN move to RPU (P1 order) ----
    # RC-1: ckpt_reader normalizes source BF16/FP32 weights to CPU FP32, so the swizzle would
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
    # op already support it (gr00t uses this). Unset => fp16 ViT, byte-identical to v5.
    _vis_w8 = _env_on("RPU_LINGBOT2_VISION_W8A16")
    z2_enabled = _env_exact_one("RPU_LINGBOT2_MULTIVIEW_SPM_Z2")
    vision_chunk = vision_execution.get("vision", {}).get(
        "chunk_size", "auto"
    )
    install_qwen3_vl_vision_for_rpu(
        vision_model,
        vision_config=cfg.vision_config,
        w8a16=_vis_w8,
        execution_chunk_size="auto" if z2_enabled else vision_chunk,
    )
    _take_lingbot2_component_retirement(policy, vision_model, "vision")
    # Registering the merger weights is the one native authority: it enables
    # the post_fn that runs LayerNorm -> m0 -> GELU -> m2 -> all-reduce inside
    # the Vision graph.
    _fused_merger_on = rpu_env_bool("RPU_QWEN3VL_VISION_FUSED_MERGER")
    if _fused_merger_on:
        register_qwen3vl_vision_fused_merger(vision_model)
    # Stage patch embedding for the RPU independently of the fused merger setup.
    # RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE selects this path; changing the host
    # FP32 GEMM to device FP16 changes its accumulation and rounding.
    if os.environ.get("RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE") == "1":
        if not getattr(vision_model, "_rpu_patch_embed_on_device", False):
            from rpu_backend.runtime.weights import tp_col_swizzle_mc_weight

            _pe_w = vision_model._rpu_vision_patch_embed_w
            _pe_b = vision_model._rpu_vision_patch_embed_b
            # F31: the folded Conv3d weight is row-major [embed_dim, cin*tp*ps*ps] and MUST be
            # col-swizzled for rpu_linear, else it silently produces garbage.
            vision_model._rpu_patch_embed_w_rpu = tp_col_swizzle_mc_weight(
                _pe_w.half(), 8).to(device="rpu").contiguous()
            vision_model._rpu_patch_embed_b_rpu = (
                _pe_b.half().to(device="rpu").contiguous() if _pe_b is not None else None)
            vision_model._rpu_patch_embed_on_device = True
    # Vision fast replay skips the recorded layer body's host traversal.
    # Preload skip additionally avoids re-emitting host callbacks while recorded
    # preload nodes still execute. Baking the fused merger permits skipping its
    # post hook too, and requires the fused merger and vision fast-replay options.
    if _env_on("RPU_LINGBOT2_VISION_FAST_REPLAY", "0"):
        _vh = vision_model._rpu_vision_handle
        torch.ops.rpu.qwen3vl_vision_set_fast_replay(_vh, True)
        torch.ops.rpu.qwen3vl_vision_set_preload_replay_skip(_vh, True)
        if _fused_merger_on:
            torch.ops.rpu.qwen3vl_vision_set_bake_merger(_vh, True)
    # ---- Step E: text decoder handle -------------------------------------
    # deepstack_visual_indexes selects which VISION blocks emit the three feature tensors.
    # Qwen3-VL injects those tensors into TEXT layers 0, 1 and 2 in list order; using the
    # vision indices [5, 11, 17] here was a branch bug and changed the model architecture.
    deepstack_text_layers = list(range(len(cfg.vision_config.deepstack_visual_indexes)))
    requested_text_chunk = text_execution.get("prefill", {}).get(
        "chunk_size", "auto"
    )
    if z2_enabled and requested_text_chunk != "auto":
        text_execution = {
            stage: dict(fields) for stage, fields in text_execution.items()
        }
        text_execution.setdefault("prefill", {})["chunk_size"] = "auto"
    install_qwen3_vl_text_for_rpu(
        text_model, text_config=cfg.text_config, vision_config=cfg.vision_config,
        deepstack_lang_layers=deepstack_text_layers,
        enable_deepstack=True, max_seq_len=max_seq,
        scale_lists=_text_scale_lists(text_model),
        execution_config=text_execution)
    _take_lingbot2_component_retirement(policy, text_model, "decoder")
    # The VLM only writes prefix KV; this checkpoint has no lm_head.
    # Prefill fast replay/preload skipping requires a persistent VLM GraphCache;
    # a cache rebuilt on every request cannot enter REPLAY.
    if _env_on("RPU_LINGBOT2_PREFILL_FAST_REPLAY", "0"):
        _th = text_model._rpu_decoder_handle
        torch.ops.rpu.causal_decoder_set_fast_replay(_th, True)
        torch.ops.rpu.causal_decoder_set_preload_replay_skip(_th, True)

    # ---- Expert (sparse MoE) ----
    rope_theta = resolve_rope_theta(cfg.text_config)
    # Per-checkpoint action-expert geometry: read the checkpoint's lingbotvla_cli.yaml (passed
    # explicitly, or auto-found next to the weights) instead of hard-coding kuavo_v2, then fail
    # loud if it disagrees with the actual weights or the as-built .so. With no YAML this returns
    # the kuavo_v2 defaults, so the current build stays bit-identical.
    geom = _cv.resolve_geometry(gw, training_config_path=training_config_path, ckpt_dir=ckpt)
    head_W, norm_W = load_host_weights(gw, geom=geom)
    align_W = load_align_weights(gw)
    exp_handle, exp_keep = build_expert_moe(
        gw,
        max_seq=max_seq,
        rope_theta=rope_theta,
        geom=geom,
        runtime_config=native_cold,
        on_handle_created=lambda handle, keep: vars(policy).update(
            _exp=handle, _exp_keep=keep),
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
        rpu_execution=rpu_execution,
        execution_session=execution_session,
        execution_generation=execution_generation,
        component_generations=component_generations,
    )
    policy._rpu_swizzled = True
    return policy


def build_lingbot_vla_v2(
    ckpt,
    qwen3vl_base=None,
    *,
    max_seq=DEFAULT_MAX_SEQ,
    chunk_size=None,
    rpu_execution=None,
    align_cfg=None,
    training_config_path=None,
    _execution_session=None,
    _execution_generation=0,
    _component_generations=None,
):
    """Build one process-exclusive LingBot-VLA-V2 RPU runtime.

    CPU-only preflight failures release the process slot immediately. Once
    swizzling/RPU materialization starts, a failed owner keeps the slot claimed
    until its traceback is collected, preventing traceback-held RPU tensors from
    overlapping a replacement policy.
    """
    from rpu_backend.api._execution import normalize_rpu_execution

    execution_config = normalize_rpu_execution(
        rpu_execution,
        entry_point="build_lingbot_vla_v2",
        vlm_chunk_size=chunk_size,
        supported_components=LINGBOT2_EXECUTION_COMPONENTS,
    )
    _validate_lingbot2_cold_execution_config(
        execution_config,
        multiview_spm_z2=_env_exact_one("RPU_LINGBOT2_MULTIVIEW_SPM_Z2"),
    )
    if os.environ.get("RPU_LINGBOT2_ALLOW_UNVALIDATED") != "1":
        raise RuntimeError(
            "LingBot-VLA-V2 is fail-closed by default. Blocking CI grants "
            "controlled temporary support only to exact-Z2 W8A16, three-image "
            "REAL215..225/P225; FP16, W4 and other profiles remain "
            "evaluation-only. Set exactly "
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
    policy._gc_retirement_enabled = False
    if _execution_session is not None:
        policy._execution_session = weakref.proxy(_execution_session)
    policy._rpu_materialization_started = False
    policy._vlm = None
    policy._text = None
    policy._visual = None
    policy._exp = None
    policy._exp_keep = None
    _claim_live_instance(policy)

    try:
        result = _build_lingbot_vla_v2_impl(
            policy,
            ckpt,
            qwen3vl_base=qwen3vl_base,
            max_seq=max_seq,
            rpu_execution=execution_config,
            align_cfg=align_cfg,
            training_config_path=training_config_path,
            execution_session=_execution_session,
            execution_generation=_execution_generation,
            component_generations=_component_generations,
        )
        return result
    except BaseException as error:
        if getattr(policy, "_rpu_materialization_started", False):
            # Keep the live claim: the original traceback may still hold weights.
            try:
                session = getattr(policy, "_execution_session", None)
                if session is None:
                    policy._retire_resources()
                else:
                    session.shutdown(policy._retire_resources)
                if not policy._closed:
                    raise RuntimeError("LingBot2 builder closed Session still owns native resources")
            except BaseException as cleanup_error:
                _poison_lingbot2_retirement(policy, cleanup_error)
                error.add_note(f"LingBot2 builder cleanup also failed: {cleanup_error!r}")
        else:
            _release_live_instance(policy)
        raise
