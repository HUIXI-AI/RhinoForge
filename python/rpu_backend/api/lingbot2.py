"""Public LingBot-VLA-V2 policy facade for downstream RPU inference.

The canonical user import is:

    from rpu_backend.api import Lingbot2Policy

Shape choices, driven by the LingBot-VLA-V2 adapter's real interface:

- There is no offline prepare-artifact and no pre-quantized checkpoint. The VLA is
  built straight from the source checkpoint dir (BF16 and FP32 safetensors are
  accepted and normalized to CPU FP32 while loading), so the entry point is
  ``from_checkpoint(...)``; ``.to('rpu')`` triggers that eager build. Quantization
  (``w8a16`` / ``w4a16``) happens **at load time** from those same weights, selected
  by env flags — unlike Wall-OSS, you do NOT need a separate ``-w8a16`` checkpoint.
- ``get_action`` returns the raw action grid ``[1, n_action_steps, max_action_dim]``
  = ``[1, 50, 55]``. Only the FIRST 16 dims are real robot DoF (dual-arm 14 +
  effector 2); dims 16..54 are unsupervised padding. ``Lingbot2ActionOutput.actions``
  is the trimmed ``[H, 16]`` view; ``actions_full`` keeps the untrimmed grid.
- The denoise step count is fixed at 10 and is not a per-call argument. The
  ordinary generic profiles default to one ten-step graph with FP16 Euler state;
  exact ``runtime_env={"RPU_LINGBOT2_DENOISE_UNROLL": "0"}`` restores the
  host-FP32 Euler loop for a controlled comparison.

Router: the MoE router is strict fp16-storage top-4, and it is the DEFAULT — no flag
needed. (``RPU_LINGBOT2_DEBUG_DENSE_SOFT_ROUTER=1`` still selects the dense-soft
diagnostic path, which is ACCURACY_UNVALIDATED and must not be used in production.)

Controlled evaluation is limited to exact-Z2 W8A16 with three-image
REAL215..225/P225 inputs. FP16, W4 and other profiles are outside this envelope.
``.to('rpu')`` stays
fail-closed unless the compatibility-named
``RPU_LINGBOT2_ALLOW_UNVALIDATED=1`` is set exactly.

Adapter code is lazy-imported inside methods so ``rpu_backend.api`` stays cheap to
import in downstream control/data tooling.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import threading
import time
import weakref
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from rpu_backend.api._runtime_env import normalize_runtime_env

# Real robot DoF count. The model emits 55 action dims; 16..54 are unsupervised
# padding and must not be fed to a robot or scored for accuracy.
VALID_ACTION_DIMS = 16

# Ordinary generic profiles share this optimization bundle.  Replay controls
# stay handle-scoped: the unrolled expert, Vision and Prefill handles each carry
# their own audited opt-in, without requiring process-wide deep replay.  Vision
# merger/device-patch and FP16-state Euler change numerics, but LingBot2 is
# already controlled-evaluation-only and records those deltas separately.
_GENERIC_OPT_ENV: dict[str, str] = {
    "RPU_ALLREDUCE_RING": "profile",
    "RPU_FASTREPLAY_SKIP_SYNC": "1",
    "RPU_KVINSERT_HYBRID_V16": "1",
    "RPU_ADARMS_FUSED_BCAST": "1",
    "RPU_QWEN3VL_VISION_ROPE_SPM": "1",
    "RPU_QWEN3VL_VISION_BATCH": "1",
    "RPU_QWEN3VL_VISION_FUSED_MERGER": "1",
    "RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE": "1",
    "RPU_LINGBOT2_PREFILL_FAST_REPLAY": "1",
    "RPU_LINGBOT2_VISION_FAST_REPLAY": "1",
    "RPU_LINGBOT2_DENOISE_UNROLL": "1",
}

# Per-profile env. Read by the Python converter and by the C++ model at BUILD time
# (inside build_lingbot_vla_v2), so setting them in .to('rpu') is early enough.
_PROFILE_ENV: dict[str, dict[str, str]] = {
    # FP16 uses the same grouped-expert topology as W8/W4, while keeping FP16
    # packed weights.  This cuts the routed-expert launch fan-out without
    # changing the public precision contract.  The grouped down projection
    # accumulates internally in FP32 and writes FP16. Keep the m304/n80 shape
    # override for this profile; it still uses the current autotile family.
    # Retain the validated global replay selector as the fallback when a caller
    # explicitly disables denoise unroll.
    "fp16": {
        **_GENERIC_OPT_ENV,
        "RPU_LINGBOT2_GROUPED_EXPERTS": "1",
        "RPU_WALL_OSS_FAST_REPLAY": "1",
        "RPU_LINGBOT2_DOWN_ACC32_TILE": "1",
        "RPU_L2_RCHUNK": "1632",
    },
    "w8a16": {
        **_GENERIC_OPT_ENV,
        "RPU_LINGBOT2_GROUPED_EXPERTS": "1",
        "RPU_LINGBOT2_EXPERT_W8A16": "1",
        "RPU_LINGBOT2_PREFILL_W8A16": "1",
        "RPU_LINGBOT2_BASE_W8A16": "1",
        "RPU_LINGBOT2_VISION_W8A16": "1",
        "RPU_L2_RCHUNK": "1632",
    },
    "w4a16": {
        **_GENERIC_OPT_ENV,
        "RPU_LINGBOT2_GROUPED_EXPERTS": "1",
        "RPU_LINGBOT2_EXPERT_W4A16": "1",
        "RPU_LINGBOT2_PREFILL_W8A16": "1",
        "RPU_LINGBOT2_BASE_W8A16": "1",
        "RPU_LINGBOT2_VISION_W8A16": "1",
        "RPU_L2_RCHUNK": "1632",
    },
}

# Structural defaults. The ordinary precision bundles live in _PROFILE_ENV
# above; callers can still override every switch explicitly through
# runtime_env. Physical-Z2-only controls remain off here.
_STRUCTURAL_ENV: dict[str, str] = {
    "RPU_LINGBOT2_GROUPED_EXPERTS": "0",
    "RPU_LINGBOT2_FP16_TOP4": "1",
    "RPU_LINGBOT2_DEBUG_DENSE_SOFT_ROUTER": "0",
    "RPU_LINGBOT2_DEBUG_DUMP_ROUTER_H": "0",
    "RPU_LINGBOT2_EXPERT_REPLAY": "1",
    "RPU_LINGBOT2_HOST_PREFIX_OPT": "1",
    "RPU_LINGBOT2_VISION_DIRECT_PREFIX": "0",
    "RPU_LINGBOT2_VISION_DIRECT_PREFIX_NO_OUTPUT": "0",
    "RPU_LINGBOT2_MULTIVIEW_SPM_Z2": "0",
    "RPU_LINGBOT2_DENOISE_UNROLL": "0",
    "RPU_LINGBOT2_DENOISE_TWOSTAGE": "0",
    "RPU_LINGBOT2_OUTER_TWOSTAGE": "0",
    "RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE": "0",
    "RPU_LINGBOT2_DOWN_ACC32_TILE": "0",
    "RPU_ALLREDUCE_TWOSTAGE": "0",
    "RPU_ALLREDUCE_CHUNK_V2": "0",
    "RPU_ALLREDUCE_RING": "0",
    "RPU_LINGBOT2_VLM_REPLAY": "1",
    # Shared/global optimizations remain structurally off. Profile-scoped values
    # above override only the handles whose replay/input contracts were selected;
    # close() restores the caller's prior environment.
    "RPU_WALL_OSS_FAST_REPLAY": "0",
    "RPU_FASTREPLAY_SKIP_SYNC": "0",
    "RPU_DEEP_FAST_REPLAY": "0",
    "RPU_KVINSERT_HYBRID_V16": "0",
    "RPU_KVINSERT_HYBRID3_V16": "0",
    "RPU_ADARMS_FUSED_BCAST": "0",
    "RPU_QWEN3VL_VISION_ROPE_SPM": "0",
    "RPU_QWEN3VL_VISION_BATCH": "0",
    "RPU_QWEN3VL_VISION_FUSED_MERGER": "0",
    "RPU_QWEN3VL_VISION_HOST_FP32_PATCH": "0",
    "RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE": "0",
    "RPU_LINGBOT2_PREFILL_FAST_REPLAY": "0",
    "RPU_LINGBOT2_VISION_FAST_REPLAY": "0",
}

# One ten-step V2 denoise graph is larger than the SDK's conservative Queue
# defaults. These are facade-scoped defaults only: an explicit runtime_env value
# or a value already present in the caller's process environment wins.
_DENOISE_UNROLL_CAPACITY_DEFAULTS: dict[str, str] = {
    "LKN_MAX_BATCH_ENTRIES": "131072",
    "LKN_KD_BUF_MB": "32",
    "LKN_INSTR_BUF_MB": "128",
}

# Dense FP16 emits a larger ten-step Graph than grouped W8/W4, so it uses a
# larger queue and instruction-buffer envelope.
_DENSE_FP16_DENOISE_UNROLL_CAPACITY_DEFAULTS: dict[str, str] = {
    "LKN_MAX_BATCH_ENTRIES": "262144",
    "LKN_KD_BUF_MB": "64",
    "LKN_INSTR_BUF_MB": "512",
}

# All multiview-Z2 profiles use the larger dense-FP16 denoise-Graph envelope.
_MULTIVIEW_SPM_Z2_QUEUE_ENV: dict[str, str] = {
    "LKN_MAX_BATCH_ENTRIES": "262144",
    # Reserve conservative kernel-descriptor and instruction-buffer tiers.
    "LKN_KD_BUF_MB": "64",
    "LKN_INSTR_BUF_MB": "512",
}

# The physical-Z2 path admits exactly three typed layouts.  Its replay portion
# enables the generic full-body replay and redundant-sync skip consumed by the
# separate denoise Graph.  Physical Vision/Text members carry a non-zero
# pipeline lease epoch, so they cannot enter the ordinary FMB skip path; their
# outer owner remains on the audited occurrence walk.  The exact host ingress
# packs three raw patch slabs without changing those native occurrences.  The
# sealed dtype profile owns CPU-versus-device patch embedding and the bounded
# denoise deep-replay result.  Component-local Vision/Prefill replay stays off.
_MULTIVIEW_SPM_Z2_COMMON_ENV: dict[str, str] = {
    "RPU_LINGBOT2_DENOISE_UNROLL": "1",
    "RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE": "0",
    "RPU_LINGBOT2_EXPERT_REPLAY": "1",
    "RPU_LINGBOT2_FP16_TOP4": "1",
    "RPU_QWEN3VL_VISION_FUSED_MERGER": "1",
    "RPU_QWEN3VL_VISION_ROPE_SPM": "1",
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

_MULTIVIEW_SPM_Z2_DTYPE_PROFILE: dict[str, str] = {
    "fp16": "dense_fp16",
    "w8a16": "grouped_w8a16",
    "w4a16": "grouped_w4a16",
}

# Build a complete profile snapshot on every construction. Without explicit OFF
# values, a W4 policy leaves process-global flags behind and a later W8/FP16 policy
# silently reuses W4 conversion (W4 wins when both flags are present).
_PROFILE_DEFAULT_ENV: dict[str, str] = {
    key: "0" for profile in _PROFILE_ENV.values() for key in profile
}
_PROFILE_DEFAULT_ENV.update({
    "RPU_LINGBOT2_PREFILL_W4A16": "0",
    "RPU_LINGBOT2_LEGACY_PREPROC": "0",
    "RPU_L2_CAPTURE_L0": "0",
    "RPU_L2_CAPTURE_INNORM": "0",
    "RPU_L2_CAPTURE_GATE": "0",
    "RPU_L2_CAP_RESID": "0",
    "RPU_L2_STAGES": "0",
    "RPU_L2_ROUTED_ONLY": "0",
    "RPU_L2_DOWN_ACC16": "0",
    "RPU_L2_DBG_PACKED": "0",
    "RPU_L2_RCHUNK": "0",
    "RPU_L2_SCHUNK": "0",
})

# These debug switches are enabled by mere presence in the native code. They
# must be removed, not assigned "0", before applying a fresh profile snapshot.
_PROFILE_CLEAR_ENV = frozenset({"RPU_L2_BUFONLY", "RPU_L2_ADDR"})

_RUNTIME_ENV_ALLOWLIST = frozenset({
    *_PROFILE_DEFAULT_ENV,
    *_STRUCTURAL_ENV,
    *_DENOISE_UNROLL_CAPACITY_DEFAULTS,
    *_DENSE_FP16_DENOISE_UNROLL_CAPACITY_DEFAULTS,
    *_MULTIVIEW_SPM_Z2_QUEUE_ENV,
    *_MULTIVIEW_SPM_Z2_COMMON_ENV,
    *(key for profile in _MULTIVIEW_SPM_Z2_PROFILE_ENV.values()
      for key in profile),
    *_PROFILE_CLEAR_ENV,
    "RPU_LINGBOT2_ALLOW_UNVALIDATED",
    "RPU_LINGBOT2_ENCODER_1THREAD",
    "RPU_LINGBOT2_PREPROC",
    "RPU_LINGBOT2_QWEN3VL_BASE",
    "RPU_QWEN3VL_VISION_BATCH_CAP",
})

# LingBot2's native graph state and its profile switches are process-global.
# Serialize every facade build/inference/close/finalizer transaction so a second
# policy cannot temporarily overwrite the environment while the live policy is
# executing.  The per-instance lock below still protects each facade's state.
_RPU_POLICY_RUNTIME_LOCK = threading.RLock()


def _restore_environment(snapshot: Mapping[str, str | None] | None) -> None:
    if not snapshot:
        return
    for key, old_value in snapshot.items():
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


def _serialized_policy_call(method):
    """Serialize inference with construction/close over shared graph scratch."""
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        with _RPU_POLICY_RUNTIME_LOCK:
            with self._rpu_build_lock:
                return method(self, *args, **kwargs)
    return wrapped


def _finalize_runtime_policy(policy, env_snapshot=None) -> bool:
    """Best-effort facade finalizer; the runtime owns native-handle details."""
    with _RPU_POLICY_RUNTIME_LOCK:
        try:
            policy.close()
        except Exception:
            return False
        _restore_environment(env_snapshot)
        return True


def _runtime_policy_complete(policy) -> bool:
    finalizer = getattr(policy, "_handle_finalizer", None)
    return bool(
        policy is not None
        and not getattr(policy, "_closed", True)
        and getattr(policy, "_exp", None) is not None
        and getattr(policy, "_text", None) is not None
        and getattr(policy, "_visual", None) is not None
        and finalizer is not None
        and getattr(finalizer, "alive", False)
    )


@dataclasses.dataclass(frozen=True)
class Lingbot2ActionOutput:
    """One LingBot-VLA-V2 action-chunk result.

    ``actions`` is the de-normalized physical-unit ``[H, 16]`` action — the real robot
    DoF after the ``FeatureTransform.unapply`` equivalent. It is only available when
    the policy was built with a ``robot_norm_path`` plus the matching
    ``training_config_path`` (or an explicit ``action_norm_spec``; see
    ``from_checkpoint``). This means only that de-normalization was configured; it is
    not numerical or robot-control certification. Otherwise reading ``.actions``
    raises, because the model emits a *normalized* grid that must not be consumed as
    physical units.

    ``actions_normalized`` is the ``[H, 16]`` real-DoF slice in the model's normalized
    space; ``actions_full`` is the untrimmed
    ``[H, 55]`` normalized grid the model emits (dims 16..54 are unsupervised padding).
    Use these for comparisons against normalized-space reference output.
    """

    actions_normalized: torch.Tensor
    actions_full: torch.Tensor
    latency_ms: float
    phase_ms: dict[str, float]
    actions_physical: torch.Tensor | None = None
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def actions(self) -> torch.Tensor:
        """De-normalized physical-unit action ``[H, 16]``.

        Raises if de-normalization was not configured. Availability does not imply
        numerical or robot-control certification.
        """
        if self.actions_physical is None:
            from rpu_backend.api.errors import RPUBackendError

            raise RPUBackendError(
                "LingBot-VLA-V2 actions are normalized, not physical-unit actions: "
                "build the policy with "
                "Lingbot2Policy.from_checkpoint(..., robot_norm_path=<robot "
                "norm-stats JSON>, training_config_path=<lingbotvla_cli.yaml>) to get "
                "de-normalized physical-unit actions for controlled evaluation. This "
                "does not certify robot-control use. To inspect the normalized grid on "
                "purpose, read .actions_normalized / .actions_full instead."
            )
        return self.actions_physical

    @property
    def robot_ready(self) -> bool:
        """Compatibility flag: true iff a physical-unit denormalizer was applied.

        The name does not assert numerical or robot-control certification.
        """
        return self.actions_physical is not None


class Lingbot2Policy:
    """Public LingBot-VLA-V2 (Qwen3-VL + 32-expert MoE action expert) RPU entry point.

    Checkpoint, camera roster and precision profile are fixed at construction.
    Camera pixels, the task instruction and proprioception may change per call.
    """

    def __init__(self) -> None:
        raise RuntimeError(
            "Lingbot2Policy() is not a public constructor. Use "
            "Lingbot2Policy.from_checkpoint(...).to('rpu')."
        )

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: str | os.PathLike[str],
        *,
        dtype: str = "fp16",
        n_cameras: int = 3,
        image_size: int = 256,
        max_lang_tokens: int = 72,
        robot_norm_path: str | os.PathLike[str] | None = None,
        training_config_path: str | os.PathLike[str] | None = None,
        robot_config_path: str | os.PathLike[str] | None = None,
        action_norm_spec: Any | None = None,
        runtime_env: Mapping[str, str] | None = None,
        qwen3vl_base: str | os.PathLike[str] | None = None,
    ) -> "Lingbot2Policy":
        """Create a policy bound to a LingBot-VLA-V2 checkpoint directory.

        ``ckpt_dir`` must hold ``config.json``, the BF16/FP32 ``model-*.safetensors``
        shards and the Qwen3-VL processor/tokenizer files. The SAME directory
        serves every ``dtype``: w8a16/w4a16 quantize from these weights at load
        time, so there is no separate quantized checkpoint to ship.

        ``dtype`` selects the precision profile:
          ``fp16``   — unquantized fp16
          ``w8a16``  — int8 weights / fp16 activations on expert+prefill+base+vision
          ``w4a16``  — group-wise int4 routed experts, the rest int8

        Controlled evaluation is limited to exact-Z2 ``w8a16`` at three-image
        REAL215..225/P225. ``fp16``, ``w4a16`` and other profiles are outside
        this envelope.

        ``max_lang_tokens`` is a tokenizer truncation/admission cap (72 for the
        public checkpoint), not the semantic prompt length.  The tokenizer may
        pad its one-item batch to this width, but the adapter filters those rows
        by ``attention_mask`` and preserves the resulting variable REAL prefix.
        A prepared generic profile then adds its own masked 16-row execution-tail
        padding; those execution rows are never instruction tokens.

        Action de-normalization is required to obtain physical-unit actions, but is
        not sufficient for robot-control certification. The model emits a *normalized*
        action grid; the official pipeline maps it back to physical units
        via ``FeatureTransform.unapply``. Pass ``robot_norm_path`` (the checkpoint's
        robot norm-stats JSON) together with ``training_config_path`` (the matching
        ``lingbotvla_cli.yaml``) to enable this — ``infer().actions`` then returns the
        physical ``[H, 16]`` action. The training config is mandatory because the norm
        JSON alone cannot prove per-joint order or norm types. ``robot_config_path``
        may describe delta/relative actions, but this candidate rejects such configs:
        its flat padded state input does not carry enough metadata to prove joint-offset
        alignment. ``action_norm_spec`` accepts an explicit list of ``ActionGroup`` and
        overrides all of the above. The resolved action groups must total exactly 16
        dimensions. NONE of these bake in a specific robot — every value comes from the
        files you pass.
        """
        dtype = (dtype or "fp16").strip().lower()
        if dtype not in _PROFILE_ENV:
            raise ValueError(
                f"dtype must be one of {sorted(_PROFILE_ENV)}, got {dtype!r}"
            )
        inst = cls.__new__(cls)
        inst._ckpt_dir = os.fspath(ckpt_dir)
        inst._dtype = dtype
        inst._n_cameras = int(n_cameras)
        inst._image_size = int(image_size)
        if isinstance(max_lang_tokens, bool) or not isinstance(max_lang_tokens, int):
            raise ValueError(
                f"max_lang_tokens must be a positive integer, got {max_lang_tokens!r}")
        if max_lang_tokens <= 0:
            raise ValueError(
                f"max_lang_tokens must be positive, got {max_lang_tokens}")
        inst._max_lang_tokens = int(max_lang_tokens)
        inst._robot_norm_path = None if robot_norm_path is None else os.fspath(robot_norm_path)
        inst._training_config_path = None if training_config_path is None else os.fspath(training_config_path)
        inst._robot_config_path = None if robot_config_path is None else os.fspath(robot_config_path)
        inst._action_norm_spec = action_norm_spec
        inst._denorm = None
        inst._runtime_env = normalize_runtime_env(
            runtime_env,
            owner="Lingbot2Policy",
            allowed=_RUNTIME_ENV_ALLOWLIST,
        )
        inst._qwen3vl_base = None if qwen3vl_base is None else os.fspath(qwen3vl_base)
        inst._rpu_ready = False
        inst._rpu_build_started = False
        inst._rpu_build_lock = threading.RLock()
        inst._closed = False
        inst._policy = None
        inst._policy_finalizer = None
        inst._env_snapshot = None
        inst._processor = None
        inst._prepare_ms = {}
        inst._graph_profile = None
        inst._preproc_mode = "exact"        # real value set in .to() once env is applied
        return inst

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def robot_ready(self) -> bool:
        """Compatibility flag: true iff ``.actions`` can return physical units.

        This does not assert numerical or robot-control certification.
        """
        return bool(
            self._rpu_ready
            and _runtime_policy_complete(self._policy)
            and getattr(self, "_denorm", None) is not None
        )

    @property
    def graphs_ready(self) -> bool:
        """Whether the finite LingBot2 graph envelope is lookup-only READY."""
        return bool(
            _runtime_policy_complete(self._policy)
            and self._graph_profile is not None
            and getattr(self._policy, "graphs_ready", False)
        )

    @property
    def graph_profile(self) -> dict[str, Any] | None:
        """Copy of the prepared REAL-prefix envelope, or ``None``."""
        if not self.graphs_ready:
            return None
        profile = dict(self._graph_profile)
        profile["execution_profiles"] = tuple(
            dict(item) for item in profile["execution_profiles"])
        profile["graph_counts"] = dict(profile["graph_counts"])
        return profile

    @_serialized_policy_call
    def to(self, device: Any) -> "Lingbot2Policy":
        target = str(device)
        if not target.startswith("rpu"):
            from rpu_backend.api.errors import RPUBackendError

            raise RPUBackendError("Lingbot2Policy only supports .to('rpu').")
        if self._closed:
            from rpu_backend.api.errors import RPUBackendError

            raise RPUBackendError(
                "Lingbot2Policy is closed; construct a fresh policy."
            )
        if self._rpu_ready:
            if _runtime_policy_complete(self._policy):
                return self
            from rpu_backend.api.errors import RPUBackendError

            raise RPUBackendError(
                "Lingbot2Policy is marked ready but its runtime ownership is "
                "incomplete or closed. Delete it, run gc.collect(), and "
                "construct a fresh policy."
            )

        allow_unvalidated = self._runtime_env.get(
            "RPU_LINGBOT2_ALLOW_UNVALIDATED",
            os.environ.get("RPU_LINGBOT2_ALLOW_UNVALIDATED", "0"),
        )
        if allow_unvalidated != "1":
            from rpu_backend.api.errors import RPUBackendError

            raise RPUBackendError(
                "LingBot-VLA-V2 is fail-closed by default. Controlled "
                "evaluation is limited to exact-Z2 W8A16, three-image "
                "REAL215..225/P225; FP16, W4 and other profiles are outside "
                "this envelope. Set exactly "
                "RPU_LINGBOT2_ALLOW_UNVALIDATED=1 to enter a controlled path; "
                "none is formally or robot certified."
            )

        if not self._rpu_build_lock.acquire(blocking=False):
            from rpu_backend.api.errors import RPUBackendError

            raise RPUBackendError(
                "Lingbot2Policy.to('rpu') is already building this policy "
                "in another thread."
            )
        try:
            if self._closed:
                from rpu_backend.api.errors import RPUBackendError

                raise RPUBackendError(
                    "Lingbot2Policy is closed; construct a fresh policy."
                )
            if self._rpu_ready:
                if _runtime_policy_complete(self._policy):
                    return self
                from rpu_backend.api.errors import RPUBackendError

                raise RPUBackendError(
                    "Lingbot2Policy is marked ready but its runtime ownership "
                    "is incomplete or closed. Delete it, run gc.collect(), "
                    "and construct a fresh policy."
                )
            if self._rpu_build_started:
                from rpu_backend.api.errors import RPUBackendError

                raise RPUBackendError(
                    "Lingbot2Policy.to('rpu') previously failed after the "
                    "runtime build started. Delete this policy, run "
                    "gc.collect(), and construct a fresh instance; retrying "
                    "may reuse partially initialized or swizzled state."
                )

            # Order matters: structural defaults first, then the profile, then
            # the caller's runtime_env last so an explicit override always wins.
            env: dict[str, str] = dict(_PROFILE_DEFAULT_ENV)
            env.update(_STRUCTURAL_ENV)
            env.update(_PROFILE_ENV[self._dtype])
            env.update(self._runtime_env)

            # The generic profile defaults to V2 unroll; an exact caller value
            # remains authoritative, and every other value disables it. Apply its
            # Queue limits before importing the adapter (and therefore before
            # any Queue/GraphCache can be constructed), while preserving every
            # caller-supplied capacity. The unrolled model owns the indexed
            # AdaRMS schedule; the separate direct-schedule experiment is
            # incompatible and must stay off for this build.
            denoise_unroll = env.get("RPU_LINGBOT2_DENOISE_UNROLL") == "1"
            env["RPU_LINGBOT2_DENOISE_UNROLL"] = (
                "1" if denoise_unroll else "0"
            )
            if denoise_unroll:
                env["RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE"] = "0"
                grouped_experts = (
                    env.get("RPU_LINGBOT2_GROUPED_EXPERTS") == "1"
                )
                capacity_defaults = (
                    _DENOISE_UNROLL_CAPACITY_DEFAULTS
                    if grouped_experts
                    else _DENSE_FP16_DENOISE_UNROLL_CAPACITY_DEFAULTS
                )
                for key, value in capacity_defaults.items():
                    if key not in self._runtime_env and key not in os.environ:
                        env[key] = value

            # The production multiview Z2 path owns one retained physical
            # Vision->Text Graph and then hands the arena to fused denoise.
            # Ambient process state must never opt a policy into this
            # thread-affine lifecycle: only an exact per-policy "1" does so.
            multiview_spm_z2 = (
                self._runtime_env.get("RPU_LINGBOT2_MULTIVIEW_SPM_Z2") == "1"
            )
            env["RPU_LINGBOT2_MULTIVIEW_SPM_Z2"] = (
                "1" if multiview_spm_z2 else "0"
            )
            if multiview_spm_z2:
                z2_profile = _MULTIVIEW_SPM_Z2_DTYPE_PROFILE.get(self._dtype)
                if z2_profile is None:
                    raise ValueError(
                        "RPU_LINGBOT2_MULTIVIEW_SPM_Z2=1 supports only the "
                        "sealed Dense FP16, Grouped W8A16, or Grouped W4A16 "
                        "profiles."
                    )
                if self._n_cameras != 3 or self._image_size != 256:
                    raise ValueError(
                        "RPU_LINGBOT2_MULTIVIEW_SPM_Z2=1 requires exactly "
                        "3 cameras at image_size=256."
                    )
                if not denoise_unroll:
                    raise ValueError(
                        "RPU_LINGBOT2_MULTIVIEW_SPM_Z2=1 requires exact "
                        "RPU_LINGBOT2_DENOISE_UNROLL=1."
                    )

                preproc = self._runtime_env.get(
                    "RPU_LINGBOT2_PREPROC",
                    os.environ.get("RPU_LINGBOT2_PREPROC", "exact"),
                ).strip().lower()
                legacy_preproc = self._runtime_env.get(
                    "RPU_LINGBOT2_LEGACY_PREPROC", "0"
                )
                if preproc not in ("", "exact") or legacy_preproc != "0":
                    raise ValueError(
                        "RPU_LINGBOT2_MULTIVIEW_SPM_Z2=1 requires the exact "
                        "LingBot2 image preprocessor."
                    )

                required_env = {
                    **_MULTIVIEW_SPM_Z2_QUEUE_ENV,
                    **_MULTIVIEW_SPM_Z2_COMMON_ENV,
                    **_MULTIVIEW_SPM_Z2_PROFILE_ENV[z2_profile],
                }
                conflicts = {
                    key: self._runtime_env[key]
                    for key, required in required_env.items()
                    if key in self._runtime_env
                    and self._runtime_env[key] != required
                }
                if conflicts:
                    raise ValueError(
                        "RPU_LINGBOT2_MULTIVIEW_SPM_Z2=1 conflicts with "
                        f"sealed-profile runtime_env values: {conflicts}."
                    )
                if (
                    z2_profile in ("grouped_w8a16", "grouped_w4a16")
                    and os.environ.get("RPU_LINEAR_AUTOTILE") != "1"
                ):
                    raise ValueError(
                        "Grouped W8A16/W4A16 multiview Z2 requires literal "
                        "RPU_LINEAR_AUTOTILE=1 in the process environment as "
                        "the sealed profile's compatibility marker; it does "
                        "not select kernels."
                    )
                env.update(required_env)
                env["RPU_LINGBOT2_PREPROC"] = "exact"

            touched = set(env) | set(_PROFILE_CLEAR_ENV)
            self._env_snapshot = {key: os.environ.get(key) for key in touched}
            for key in _PROFILE_CLEAR_ENV:
                os.environ.pop(key, None)
            for key, value in env.items():
                os.environ[str(key)] = str(value)

            # Read AFTER the env above is applied so it can also be driven via
            # runtime_env. RPU_LINGBOT2_PREPROC = exact | fast | legacy.
            _pp = (os.environ.get("RPU_LINGBOT2_PREPROC") or "").strip().lower()
            if os.environ.get("RPU_LINGBOT2_LEGACY_PREPROC") == "1":
                _pp = "legacy"
            if _pp not in ("exact", "fast", "legacy"):
                _pp = "exact"
            self._preproc_mode = _pp

            # Thread the config-only Qwen3-VL base explicitly into the builder.
            qv = (
                self._qwen3vl_base
                or self._runtime_env.get("RPU_LINGBOT2_QWEN3VL_BASE")
                or None
            )
            if qv:
                qv = os.path.abspath(os.path.expanduser(str(qv)))

            # Robot-action config is CPU-only preflight. Resolve it before any
            # swizzle or native handle is created so a bad/mismatched stats file
            # cannot leave a partially materialized process behind.
            denorm = self._build_denormalizer()

            from rpu_backend.adapters.lingbot_vla_v2 import build_lingbot_vla_v2

            self._rpu_build_started = True
            pending_policy = None
            pending_finalizer = None
            try:
                t0 = time.perf_counter()
                # The training config also drives the checkpoint-specific MoE
                # geometry. The runtime builder owns the process-wide claim.
                pending_policy = build_lingbot_vla_v2(
                    self._ckpt_dir,
                    qwen3vl_base=qv,
                    training_config_path=self._training_config_path,
                )
                if not _runtime_policy_complete(pending_policy):
                    from rpu_backend.api.errors import RPUBackendError

                    raise RPUBackendError(
                        "LingBot-VLA-V2 builder returned without complete "
                        "expert/text/vision runtime ownership."
                    )
                prepare_ms = {
                    "build_ms": (time.perf_counter() - t0) * 1000.0
                }
                pending_finalizer = weakref.finalize(
                    self,
                    _finalize_runtime_policy,
                    pending_policy,
                    dict(self._env_snapshot),
                )
            except BaseException as exc:
                if pending_policy is not None:
                    try:
                        pending_policy.close()
                    except Exception:
                        raise RuntimeError(
                            "LingBot-VLA-V2 construction failed and native "
                            "runtime cleanup also failed; restart the process "
                            "before loading another policy."
                        ) from exc
                raise

            # Publish only after runtime construction, de-normalizer setup and
            # finalizer registration have all succeeded.
            self._policy = pending_policy
            self._policy_finalizer = pending_finalizer
            self._prepare_ms = prepare_ms
            self._denorm = denorm
            self._rpu_ready = True
            return self
        except BaseException:
            if not self._rpu_ready:
                _restore_environment(self._env_snapshot)
                self._env_snapshot = None
            raise
        finally:
            self._rpu_build_lock.release()

    @_serialized_policy_call
    def close(self) -> None:
        """Release the owned runtime and make this facade permanently closed."""
        with self._rpu_build_lock:
            if self._closed:
                return
            finalizer = getattr(self, "_policy_finalizer", None)
            policy = getattr(self, "_policy", None)
            if finalizer is not None and getattr(finalizer, "alive", False):
                if finalizer() is not True:
                    raise RuntimeError(
                        "LingBot-VLA-V2 cleanup failed; restart the process "
                        "before loading another policy."
                    )
            elif policy is not None:
                policy.close()
            _restore_environment(self._env_snapshot)
            self._env_snapshot = None
            self._policy = None
            self._policy_finalizer = None
            self._graph_profile = None
            self._rpu_ready = False
            self._closed = True

    def _build_denormalizer(self):
        """Construct the config-driven action de-normalizer, or None if not configured."""
        from rpu_backend.api.action_denorm import ActionDenormalizer

        if self._action_norm_spec is not None:
            if self._robot_norm_path is None:
                raise ValueError("action_norm_spec requires robot_norm_path (the norm-stats JSON).")
            denorm = ActionDenormalizer.from_spec(
                self._robot_norm_path, self._action_norm_spec
            )
        elif self._robot_norm_path is None:
            return None
        elif self._training_config_path is not None:
            denorm = ActionDenormalizer.from_training_config(
                self._robot_norm_path, self._training_config_path,
                robot_config=self._robot_config_path,
            )
        else:
            raise ValueError(
                "robot_norm_path requires the matching training_config_path or an "
                "explicit action_norm_spec; the norm JSON alone cannot prove action "
                "group order and normalization types"
            )

        if denorm.real_dim != VALID_ACTION_DIMS:
            raise ValueError(
                "LingBot-VLA-V2 robot action config resolves to "
                f"{denorm.real_dim} dims; the checkpoint contract requires exactly "
                f"{VALID_ACTION_DIMS}. Refusing to mark a partial/misaligned action "
                "as a complete physical-unit action."
            )
        delta_groups = [g.key for g in denorm.groups if g.subtract_state]
        if delta_groups:
            raise ValueError(
                "LingBot-VLA-V2 delta/relative action de-normalization is not "
                "certified: the runtime receives a flat padded state tensor and "
                "cannot prove its joint offsets match the action groups "
                f"{delta_groups}. Use an absolute-action checkpoint/config."
            )
        return denorm

    # ── preprocessing ───────────────────────────────────────────────────────
    def _get_processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self._ckpt_dir)
        return self._processor

    @staticmethod
    def _as_rgb_uint8_array(im: Any) -> Any:
        """Normalize one camera frame to an HWC uint8 RGB numpy array.

        Accepts three input types:
          * a PIL image, converted with ``np.array(im.convert("RGB"))``;
          * an HWC uint8 RGB numpy array;
          * an HWC uint8 torch tensor.

        Numpy/tensor inputs must already be RGB; no colourspace conversion is applied.
        """
        import numpy as np

        if isinstance(im, np.ndarray):
            arr = im
        elif torch.is_tensor(im):
            arr = im.detach().cpu().numpy()
        else:  # PIL.Image, or anything exposing .convert()
            # np.array provides the writable copy required by torch.as_tensor.
            return np.array(im.convert("RGB"))
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"expected an HWC RGB frame with shape (H, W, 3), got array of "
                f"shape {tuple(arr.shape)} / dtype {arr.dtype}"
            )
        if arr.dtype != np.uint8:
            raise ValueError(
                f"expected an HWC uint8 RGB frame, got dtype {arr.dtype}"
            )
        return np.ascontiguousarray(arr)

    def _encode_images(self, images: Sequence[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Camera frames -> (pixel_values [N, P, D], image_grid_thw [N, 3]).

        Each frame may be a PIL image or an HWC uint8 RGB numpy array / tensor.

        Resizes each camera frame to the square the checkpoint's evaluation contract
        assumes, then runs the Qwen3-VL image processor.

        Exact mode preserves the fp32 torchvision resize pipeline and batches the
        processor call. Fast mode uses PIL resize. Compatibility mode retains the
        per-camera processor path.

        Resampling differs from the compatibility path (torchvision antialiased
        bilinear on fp32 vs PIL bilinear on uint8). Set
        ``RPU_LINGBOT2_LEGACY_PREPROC=1`` to restore the compatibility path.
        """
        import numpy as np

        if len(images) != self._n_cameras:
            raise ValueError(
                f"expected {self._n_cameras} images (n_cameras), got {len(images)}"
            )
        sz = self._image_size
        n = len(images)
        proc = self._get_processor()
        ip = getattr(proc, "image_processor", proc)
        mode = getattr(self, "_preproc_mode", "exact")

        if mode == "fast":
            from PIL import Image as _Image

            # Fast mode uses PIL bilinear resize and one batched processor call;
            # its resampling intentionally differs from exact mode.
            arrs = [np.asarray(
                        _Image.fromarray(self._as_rgb_uint8_array(im))
                        .resize((sz, sz), _Image.BILINEAR))
                    for im in images]
            enc = ip(images=arrs, return_tensors="pt")
            pixel_values = enc["pixel_values"]
            return pixel_values.reshape(n, -1, pixel_values.shape[-1]).contiguous().float(), \
                enc["image_grid_thw"].reshape(n, 3).long()

        if mode == "exact":
            # Preserve the compatibility pipeline's fp32 antialiased resize while
            # batching the processor call; pixel_values remain identical.
            from torchvision.transforms.v2 import Resize

            resize = Resize((sz, sz))
            arrs = []
            for im in images:
                t = torch.as_tensor(self._as_rgb_uint8_array(im)).permute(2, 0, 1).contiguous()
                t = resize(t.to(torch.float32))
                arrs.append(t.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).numpy())
            enc = ip(images=arrs, return_tensors="pt")
            pixel_values = enc["pixel_values"]
            return pixel_values.reshape(n, -1, pixel_values.shape[-1]).contiguous().float(), \
                enc["image_grid_thw"].reshape(n, 3).long()

        # Compatibility preprocessing mode.
        from PIL import Image as _Image
        from torchvision.transforms.v2 import Resize

        resize = Resize((sz, sz))
        pv, grid = [], []
        for im in images:
            t = torch.as_tensor(self._as_rgb_uint8_array(im)).permute(2, 0, 1).contiguous()
            t = resize(t.to(torch.float32))
            arr = t.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).numpy()
            enc = ip(images=[_Image.fromarray(arr)], return_tensors="pt")
            pv.append(enc["pixel_values"])
            grid.append(enc["image_grid_thw"][0])
        pixel_values = torch.cat(pv, 0)
        return pixel_values.reshape(n, -1, pixel_values.shape[-1]).contiguous().float(), \
            torch.stack(grid, 0).long()

    def _encode_instruction(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize to the admission cap; ``attention_mask`` defines REAL rows."""
        proc = self._get_processor()
        tok = proc.tokenizer(
            instruction,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self._max_lang_tokens,
        )
        return tok["input_ids"].long(), tok["attention_mask"].bool()

    def _build_raw_observation(
        self,
        *,
        images: Sequence[Any],
        instruction: str,
        proprioception: Sequence[float] | torch.Tensor | None,
        noise: torch.Tensor | None,
        noise_seed: int,
    ) -> dict[str, torch.Tensor]:
        pixel_values, grid = self._encode_images(images)
        lang_tokens, lang_masks = self._encode_instruction(instruction)
        state_dim = (
            int(self._policy.max_state_dim)
            if hasattr(self._policy, "max_state_dim") else 55
        )
        if proprioception is None:
            state = torch.zeros(1, state_dim, dtype=torch.float32)
        else:
            source = torch.as_tensor(
                proprioception, dtype=torch.float32).reshape(1, -1)
            if source.shape[1] > state_dim:
                raise ValueError(
                    f"proprioception has {source.shape[1]} dims, max is {state_dim}")
            state = torch.zeros(1, state_dim, dtype=torch.float32)
            state[:, :source.shape[1]] = source
        if noise is None:
            generator = torch.Generator().manual_seed(int(noise_seed))
            noise = torch.randn(
                (1, 50, state_dim), generator=generator,
                dtype=torch.float32)
        return {
            "images": pixel_values,
            "image_grid_thw": grid,
            "img_masks": torch.ones(grid.shape[0], dtype=torch.bool),
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
            "state": state,
            "noise": noise.float(),
        }

    @_serialized_policy_call
    @torch.no_grad()
    def prepare_graphs(
        self,
        *,
        images: Sequence[Any] | None = None,
        instruction: str | None = None,
        proprioception: Sequence[float] | torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        noise_seed: int = 0,
        obs: Mapping[str, torch.Tensor] | None = None,
        prefix_length_range: Sequence[int] | None = None,
    ) -> "Lingbot2Policy":
        """Prebuild/freeze a finite variable-prompt execution envelope.

        ``prefix_length_range`` is inclusive REAL prefix length: tokenizer
        padding has already been filtered, while adapter execution-tail padding
        has not yet been appended.  When omitted, generic execution covers one
        through ``max_lang_tokens`` REAL instruction tokens for the representative
        image geometry.  Fixed-P225 multiview Z2 truthfully caps that default at
        its physical envelope instead of falling back to generic execution.
        """
        if not self._rpu_ready or not _runtime_policy_complete(self._policy):
            from rpu_backend.api.errors import RPUBackendError

            raise RPUBackendError("call .to('rpu') before prepare_graphs().")
        if self._graph_profile is not None:
            raise RuntimeError("Lingbot2Policy graphs are already prepared")
        if obs is None:
            if images is None or instruction is None:
                raise ValueError(
                    "pass either obs=..., or images=... and instruction=...")
            obs = self._build_raw_observation(
                images=images,
                instruction=instruction,
                proprioception=proprioception,
                noise=noise,
                noise_seed=noise_seed,
            )
        else:
            obs = dict(obs)

        metadata = self._policy._request_prefix_geometry(
            obs["images"], obs.get("img_masks"), obs["lang_tokens"],
            obs.get("lang_masks"), obs.get("image_grid_thw"),
        )
        if prefix_length_range is None:
            lo = metadata["fixed_real_rows"] + 1
            hi = metadata["fixed_real_rows"] + self._max_lang_tokens
            if getattr(self._policy, "_multiview_spm_z2", False):
                hi = min(hi, 225)
            if hi < lo:
                raise ValueError(
                    "representative image geometry does not fit the fixed P225 "
                    "multiview Z2 prompt envelope")
            prefix_length_range = (lo, hi)

        t0 = time.perf_counter()
        profile = self._policy.prepare_graphs(
            obs, prefix_length_range=prefix_length_range)
        self._prepare_ms["graph_warmup_ms"] = (
            time.perf_counter() - t0) * 1000.0
        self._graph_profile = profile
        return self

    # ── inference ───────────────────────────────────────────────────────────
    @_serialized_policy_call
    def infer(
        self,
        *,
        images: Sequence[Any] | None = None,
        instruction: str | None = None,
        proprioception: Sequence[float] | torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        noise_seed: int = 0,
        obs: Mapping[str, torch.Tensor] | None = None,
    ) -> Lingbot2ActionOutput:
        """Run one action-chunk inference.

        Either pass the raw inputs (``images`` + ``instruction`` + ``proprioception``)
        and let the policy preprocess them, or pass a fully-formed ``obs`` mapping to
        reproduce a recorded preprocessed input byte-for-byte.
        """
        if not self._rpu_ready or not _runtime_policy_complete(self._policy):
            from rpu_backend.api.errors import RPUBackendError

            raise RPUBackendError("call .to('rpu') before infer().")

        if obs is None:
            if images is None or instruction is None:
                raise ValueError("pass either obs=..., or images=... and instruction=...")
            obs = self._build_raw_observation(
                images=images,
                instruction=instruction,
                proprioception=proprioception,
                noise=noise,
                noise_seed=noise_seed,
            )

        t0 = time.perf_counter()
        with torch.no_grad():
            raw = self._policy.get_action(dict(obs))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        full = raw.detach().float().cpu()
        if full.dim() == 3:
            full = full[0]
        if not bool(torch.isfinite(full).all()):
            from rpu_backend.api.errors import RPUBackendError

            raise RPUBackendError("LingBot-VLA-V2 produced a non-finite action chunk.")

        norm16 = full[:, :VALID_ACTION_DIMS].contiguous()
        physical = None
        if self._denorm is not None:
            # The public facade currently accepts only absolute-action configs; state
            # remains passed through so ActionDenormalizer's standalone contract stays
            # explicit, but _build_denormalizer rejects subtract_state before this point.
            state = obs.get("state") if isinstance(obs, Mapping) else None
            physical = self._denorm.unapply(norm16, state=state).contiguous()
            if not bool(torch.isfinite(physical).all()):
                from rpu_backend.api.errors import RPUBackendError

                raise RPUBackendError(
                    "LingBot-VLA-V2 de-normalization produced a non-finite physical "
                    "action chunk; refusing physical-unit output."
                )
        prefix_extra = {}
        prefix_metadata = getattr(
            self._policy, "_last_prefix_metadata", None)
        if isinstance(prefix_metadata, dict):
            prefix_extra["real_prefix_len"] = int(
                prefix_metadata["real_len"])
            plan = self._policy._prefix_execution_plan(
                prefix_metadata["real_len"])
            prefix_extra["execution_prefix_len"] = int(
                plan.execution_len)
            prefix_extra["prefill_chunk_size"] = int(plan.chunk_size)
        return Lingbot2ActionOutput(
            actions_normalized=norm16,
            actions_full=full,
            latency_ms=latency_ms,
            phase_ms=dict(self._prepare_ms),
            actions_physical=physical,
            extra={
                "dtype": self._dtype,
                "valid_action_dims": VALID_ACTION_DIMS,
                "denormalized": physical is not None,
                "robot_ready": physical is not None,
                "graphs_ready": self.graphs_ready,
                **prefix_extra,
            },
        )

    def predict_action_chunk(self, **kwargs: Any) -> torch.Tensor:
        """Convenience wrapper: returns just the ``[H, 16]`` real-DoF action chunk."""
        return self.infer(**kwargs).actions
