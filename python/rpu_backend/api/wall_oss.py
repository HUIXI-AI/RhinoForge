"""Public Wall-OSS-0.5 policy facade for downstream RPU inference.

The canonical user import is:

    from rpu_backend.api import WallOssPolicy

Two shape choices are driven by the Wall-OSS adapter's real interface:

- Wall-OSS has no offline prepare-artifact. The VLA is built straight from a
  checkpoint dir, so the entry point is ``from_checkpoint(...)`` (not
  ``from_prepare_artifact``). ``.to('rpu')`` triggers that eager build — there is
  no separate device move to make afterward.
- ``WallOssVLA.predict`` returns the *de-normalized* (physical-unit) action chunk
  (the per-dataset normalizer runs inside the adapter). So ``WallOssActionOutput``
  carries ``actions`` (physical), not ``actions_norm``. Downstream should not
  re-denormalize.

Adapter code is lazy-imported inside methods so ``rpu_backend.api`` stays cheap to
import in downstream control/data tooling.
"""

from __future__ import annotations

import dataclasses
import math
import os
import threading
import time
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import torch

from rpu_backend.api._runtime_env import (
    LKN_CAPACITY_ENV,
    normalize_runtime_env,
)


_SUPPORTED_CAMERA_NAMES = frozenset(
    {
        "face_view",
        "left_wrist_view",
        "right_wrist_view",
        "move1_view",
        "move2_view",
        "wall_view",
        "top_view",
        "side_view",
        "global_view",
    }
)
_RUNTIME_ENV_ALLOWLIST = frozenset({
    "RPU_DEEP_FAST_REPLAY",
    "RPU_FASTREPLAY_SKIP_SYNC",
    "RPU_LINEAR_ACC32",
    "RPU_RMSNORM_NEWTON",
    "RPU_WALL_OSS_ACTION_FP32_TAIL",
    "RPU_WALL_OSS_ATTN_TP8",
    "RPU_WALL_OSS_BATCH_MAX_SEQ",
    "RPU_WALL_OSS_BATCH_VISION",
    "RPU_WALL_OSS_DENOISE_UNROLL",
    "RPU_WALL_OSS_DEVICE_PATCH_EMBED",
    "RPU_WALL_OSS_EXPERT0_DOWN_INT4",
    "RPU_WALL_OSS_FAST_IMGPROC",
    "RPU_WALL_OSS_FAST_PROCESSOR",
    "RPU_WALL_OSS_FAST_REPLAY",
    "RPU_WALL_OSS_FUSED_ASSEMBLE",
    "RPU_WALL_OSS_FUSED_DENOISE",
    "RPU_WALL_OSS_GENERIC_PROLOGUE",
    "RPU_WALL_OSS_HOST_CACHE",
    "RPU_WALL_OSS_INSTRUMENT",
    "RPU_WALL_OSS_KVINSERT_PAD16",
    "RPU_WALL_OSS_MULTISUITE_PROLOGUE",
    "RPU_WALL_OSS_ONDEVICE_EMBED",
    "RPU_WALL_OSS_PARTIAL_MROPE",
    "RPU_WALL_OSS_PE_NORM_FOLD",
    "RPU_WALL_OSS_PROLOGUE_FILLER",
    "RPU_WALL_OSS_SHORT_PROMPT",
    "RPU_WALL_OSS_VISION_DEVICE_MERGED",
    "RPU_WALL_OSS_VISION_FUSED_MERGER",
    "RPU_WALL_OSS_VISION_LAYER_GROUP",
    "RPU_WALL_OSS_VISION_ROPE_SPM",
    "WALL_OSS_PROFILE_PREFIX",
})


def _strict_bool(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"WallOssPolicy {name} must be bool, got {value!r}")
    return value


def _integer_at_least(name: str, value: int, minimum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError(
            f"WallOssPolicy {name} must be an integer >= {minimum}, got {value!r}"
        )
    return int(value)


def _integer_value(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(
            f"WallOssPolicy {name} must be an integer, got {value!r}"
        )
    return int(value)


def _positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"WallOssPolicy {name} must be a positive finite number, got {value!r}"
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(
            f"WallOssPolicy {name} must be a positive finite number, got {value!r}"
        )
    return result


def _camera_roster(camera_names: Sequence[str]) -> list[str]:
    if isinstance(camera_names, (str, bytes)) or not isinstance(
        camera_names, Sequence
    ):
        raise ValueError(
            "WallOssPolicy camera_names must be a non-empty sequence of names"
        )
    result = list(camera_names)
    if not result or any(not isinstance(name, str) or not name for name in result):
        raise ValueError(
            "WallOssPolicy camera_names must be a non-empty sequence of names"
        )
    unknown = sorted(set(result) - _SUPPORTED_CAMERA_NAMES)
    if unknown:
        raise ValueError(
            f"WallOssPolicy unsupported camera_names {unknown}; supported names: "
            f"{sorted(_SUPPORTED_CAMERA_NAMES)}"
        )
    return result


def _active_slot_list(active_slots: Sequence[int] | None) -> list[int] | None:
    if active_slots is None:
        return None
    if isinstance(active_slots, (str, bytes)) or not isinstance(
        active_slots, Sequence
    ):
        raise ValueError(
            "WallOssPolicy active_slots must be a sequence of non-negative integers"
        )
    return [
        _integer_at_least("active_slots entry", slot, 0) for slot in active_slots
    ]


def _runtime_environment(runtime_env: Mapping[str, str] | None) -> dict[str, str]:
    return normalize_runtime_env(
        runtime_env,
        owner="WallOssPolicy",
        allowed=_RUNTIME_ENV_ALLOWLIST,
        preimport_only=LKN_CAPACITY_ENV,
    )


def _cpu_finite_tensor(value, *, name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            f"WallOssPolicy {name} must be a CPU tensor-like value"
        ) from exc
    if tensor.device.type != "cpu":
        raise ValueError(
            f"WallOssPolicy {name} must be on CPU, got {tensor.device}"
        )
    if tensor.numel() == 0:
        raise ValueError(f"WallOssPolicy {name} must not be empty")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"WallOssPolicy {name} must contain only finite values")
    return tensor


def _binary_mask(value, *, name: str) -> torch.Tensor:
    tensor = _cpu_finite_tensor(value, name=name)
    if not bool(torch.all((tensor == 0) | (tensor == 1))):
        raise ValueError(f"WallOssPolicy {name} must contain only 0 or 1")
    return tensor


def _prefix_range(value: Sequence[int]) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            "WallOssPolicy prefix_length_range must be a two-item sequence"
        )
    if len(value) != 2:
        raise ValueError(
            "WallOssPolicy prefix_length_range must be a two-item sequence"
        )
    lo = _integer_at_least("prefix_length_range minimum", value[0], 1)
    hi = _integer_at_least("prefix_length_range maximum", value[1], 1)
    if hi < lo:
        raise ValueError(
            f"WallOssPolicy prefix_length_range must satisfy min <= max, got "
            f"({lo}, {hi})"
        )
    return lo, hi


@dataclasses.dataclass(frozen=True)
class WallOssActionOutput:
    """One Wall-OSS action-chunk result.

    ``actions`` is the de-normalized (physical-unit) chunk with shape
    ``[H, action_dim]``.
    """

    actions: torch.Tensor
    action_hz: float
    latency_ms: float
    phase_ms: dict[str, float]
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


def _wall_oss_vla_runtime_complete(vla: Any) -> bool:
    check = getattr(vla, "_runtime_complete", None)
    if not callable(check):
        return False
    try:
        return check() is True
    except Exception:
        return False


def _wall_oss_policy_runtime_complete(policy: Any) -> bool:
    return bool(
        getattr(policy, "_rpu_ready", False) is True
        and getattr(policy, "_rpu_build_started", False) is True
        and _wall_oss_vla_runtime_complete(getattr(policy, "_vla", None))
    )


def _close_wall_oss_vla(vla: Any) -> None:
    close = getattr(vla, "close", None)
    if callable(close):
        try:
            close()
        except BaseException:
            pass


class WallOssPolicy:
    """Public Wall-OSS-0.5 (Qwen2.5-VL VLA) RPU entry point.

    Prompt layout (FLOW system/observation template), camera roster and dataset
    normalizer are fixed at construction. Camera pixels, proprioception and the
    task instruction may change per call.
    """

    def __init__(self) -> None:
        raise RuntimeError(
            "WallOssPolicy() is not a public constructor. Use "
            "WallOssPolicy.from_checkpoint(...).to('rpu')."
        )

    @property
    def action_dim(self) -> int:
        """Per-step action vector length the model emits. Available only after .to('rpu')."""
        if not _wall_oss_policy_runtime_complete(self):
            raise RuntimeError("WallOssPolicy.action_dim is available only after .to('rpu')")
        return _integer_at_least("runtime action_dim", self._vla.action_dim, 1)

    @property
    def action_horizon(self) -> int:
        """Number of action steps per inference chunk (H)."""
        return int(self._action_horizon)

    @property
    def camera_names(self) -> list[str]:
        """Camera roster (one input image expected per entry, in this order)."""
        return list(self._camera_names)

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: str | os.PathLike[str],
        *,
        dataset_key: str = "berkeley_autolab_ur5",
        camera_names: Sequence[str] = ("face_view",),
        delta_action: bool = False,
        state_bins: int = 256,
        action_hz: float = 32.0,
        action_horizon: int = 32,
        num_steps: int = 10,
        max_seq_len: int = 2048,
        w8a16: bool = False,
        w4a16: bool = False,
        nvfp4a16: bool = False,
        fp16_ckpt_dir: str | os.PathLike[str] | None = None,
        active_slots: Sequence[int] | None = None,
        runtime_env: Mapping[str, str] | None = None,
        rpu_execution=None,
    ) -> "WallOssPolicy":
        """Create a policy bound to a Wall-OSS-0.5 checkpoint directory.

        ``ckpt_dir`` must hold ``config.json``, the decoder/vision weights, the
        Qwen2.5-VL processor files and the ``normalizer_action.pth`` /
        ``normalizer_propri.pth`` sidecars. ``dataset_key`` selects the per-robot
        normalizer and the FLOW-prompt Embodiment tag; it must have both
        ``min.<key>`` and ``delta.<key>`` entries in both sidecars. See
        ``rpu_backend.adapters.wall_oss.build_wall_oss_vla`` for the field meanings.

        ``w8a16``, ``w4a16`` and ``nvfp4a16`` require an explicit
        ``fp16_ckpt_dir`` for processor, normalizer and CPU glue weights.
        For ``nvfp4a16``, ``ckpt_dir`` is the matching W8A16 checkpoint:
        Vision and expert-0 ``down_proj`` stay W8 while the same decoder
        projections selected by WINT4A16 use NVFP4. This is an experimental
        final-action mode and does not claim FP16-equivalent prefix parity.

        """

        from rpu_backend.api._execution import normalize_rpu_execution
        execution_config = normalize_rpu_execution(
            rpu_execution,
            entry_point="WallOssPolicy.from_checkpoint",
            supported={
                "prefill": ("chunk_size", "padding_rows", "padding_budget"),
                "vision": ("chunk_size",),
                "action": ("chunk_size",),
            },
        )
        delta_action = _strict_bool("delta_action", delta_action)
        w8a16 = _strict_bool("w8a16", w8a16)
        w4a16 = _strict_bool("w4a16", w4a16)
        nvfp4a16 = _strict_bool("nvfp4a16", nvfp4a16)
        precision_modes = {
            "w8a16": w8a16,
            "w4a16": w4a16,
            "nvfp4a16": nvfp4a16,
        }
        selected = [name for name, enabled in precision_modes.items() if enabled]
        if len(selected) > 1:
            raise ValueError(
                "WallOssPolicy precision modes are mutually exclusive, got "
                f"{selected}")
        if selected and fp16_ckpt_dir is None:
            raise ValueError(
                f"{selected[0]} requires fp16_ckpt_dir; quantized checkpoints "
                "do not contain directly usable FP16 CPU glue weights")
        inst = cls.__new__(cls)
        inst._ckpt_dir = os.fspath(ckpt_dir)
        if not isinstance(dataset_key, str) or not dataset_key:
            raise ValueError(
                f"WallOssPolicy dataset_key must be a non-empty string, got "
                f"{dataset_key!r}"
            )
        inst._dataset_key = dataset_key
        inst._camera_names = _camera_roster(camera_names)
        inst._delta_action = delta_action
        inst._state_bins = _integer_at_least("state_bins", state_bins, 2)
        inst._action_hz = _positive_float("action_hz", action_hz)
        inst._action_horizon = _integer_at_least(
            "action_horizon", action_horizon, 1
        )
        action_chunk = execution_config.get("action", {}).get(
            "chunk_size", "auto"
        )
        required_action_chunk = ((inst._action_horizon + 15) // 16) * 16
        if (
            action_chunk != "auto"
            and int(action_chunk) != required_action_chunk
        ):
            raise ValueError(
                "WallOssPolicy action chunk_size must equal "
                "ceil16(action_horizon): "
                f"requested={action_chunk}, "
                f"action_horizon={inst._action_horizon}, "
                f"required={required_action_chunk}"
            )
        inst._num_steps = _integer_at_least("num_steps", num_steps, 1)
        inst._max_seq_len = _integer_at_least("max_seq_len", max_seq_len, 1)
        if inst._max_seq_len <= inst._action_horizon:
            raise ValueError(
                "WallOssPolicy max_seq_len must be greater than "
                "action_horizon so the shared cache retains at least one "
                f"prefill row; got max_seq_len={inst._max_seq_len}, "
                f"action_horizon={inst._action_horizon}"
            )
        active_slot_list = _active_slot_list(active_slots)
        inst._w8a16 = w8a16
        inst._w4a16 = w4a16
        inst._nvfp4a16 = nvfp4a16
        inst._precision = (
            "w8a16" if w8a16 else (
                "w4a16" if w4a16 else (
                    "nvfp4a16" if nvfp4a16 else "fp16"
                )
            )
        )
        inst._fp16_ckpt_dir = os.fspath(fp16_ckpt_dir) if fp16_ckpt_dir is not None else inst._ckpt_dir
        inst._active_slots = active_slot_list
        inst._runtime_env = _runtime_environment(runtime_env)
        inst._rpu_execution = execution_config
        inst._rpu_ready = False
        inst._rpu_build_started = False
        inst._rpu_build_lock = threading.Lock()
        inst._vla = None
        inst._prepare_ms = {}
        inst._graph_profile = None
        return inst

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        **kwargs: Any,
    ) -> "WallOssPolicy":
        """HF-idiomatic alias for :meth:`from_checkpoint`.

        ``pretrained_model_name_or_path`` is a **local Wall-OSS-0.5 HF model
        directory**: standard HF assets (``config.json`` + ``model.safetensors``
        [+ ``model.safetensors.index.json`` for shards] + Qwen2.5-VL processor
        files) plus the two wall-x normalizer sidecars (``normalizer_action.pth`` /
        ``normalizer_propri.pth``). Weights load HF-natively —
        ``AutoConfig``/``AutoProcessor.from_pretrained`` + ``safetensors.safe_open``
        on ``model.safetensors``; the ``.pth`` sidecars are preprocessing stats
        (action/proprioception mean+delta), NOT model weights.

        Weights are delivered out of band (not on a public Hub), so this resolves a
        **local directory** — fetch the model dir first (``git lfs clone`` or
        ``huggingface_hub.snapshot_download``) and pass its path. Remaining keyword
        args are forwarded verbatim to :meth:`from_checkpoint`.
        """
        path = os.fspath(pretrained_model_name_or_path)
        if not os.path.isdir(path):
            from rpu_backend.api.errors import RPUConfigError

            raise RPUConfigError(
                "WallOssPolicy.from_pretrained expects a local Wall-OSS HF model "
                f"directory; not found: {path}. Fetch it first (git lfs clone / "
                "huggingface_hub.snapshot_download) and pass the resolved path."
            )
        return cls.from_checkpoint(path, **kwargs)

    def to(self, device: Any) -> "WallOssPolicy":
        from rpu_backend.api.errors import RPUBackendError
        from rpu_backend.api._loading import resolve_loader_device

        try:
            is_rpu = resolve_loader_device(
                device, entry_point="WallOssPolicy.to"
            )
        except ValueError as exc:
            raise RPUBackendError(
                "WallOssPolicy only supports device rpu or rpu:0."
            ) from exc
        if not is_rpu:
            raise RPUBackendError(
                "WallOssPolicy only supports device rpu or rpu:0."
            )
        if _wall_oss_policy_runtime_complete(self):
            return self
        if self._rpu_ready:
            raise RPUBackendError(
                "WallOssPolicy is marked ready but its VLA runtime ownership is "
                "incomplete or closed. Delete this policy, run gc.collect(), "
                "and construct a fresh instance."
            )

        if not self._rpu_build_lock.acquire(blocking=False):
            raise RPUBackendError(
                "WallOssPolicy.to('rpu') is already building this policy in "
                "another thread."
            )
        try:
            if _wall_oss_policy_runtime_complete(self):
                return self
            if self._rpu_ready:
                raise RPUBackendError(
                    "WallOssPolicy is marked ready but its VLA runtime ownership "
                    "is incomplete or closed. Delete this policy, run "
                    "gc.collect(), and construct a fresh instance."
                )
            if self._rpu_build_started:
                raise RPUBackendError(
                    "WallOssPolicy.to('rpu') previously failed after claiming "
                    "the process-wide RPU slot. Delete this policy, run "
                    "gc.collect(), and construct a fresh instance; retrying may "
                    "reuse partially initialized or swizzled state."
                )

            from rpu_backend.api.causal_lm import _claim_live_instance
            _claim_live_instance(self)
            self._rpu_build_started = True

            pending_vla = None
            try:
                for key, value in self._runtime_env.items():
                    os.environ[str(key)] = str(value)

                from rpu_backend.adapters.wall_oss import build_wall_oss_vla

                t0 = time.perf_counter()
                pending_vla = build_wall_oss_vla(
                    self._ckpt_dir,
                    dataset_key=self._dataset_key,
                    camera_names=self._camera_names,
                    delta_action=self._delta_action,
                    state_bins=self._state_bins,
                    max_seq_len=self._max_seq_len,
                    w8a16=self._w8a16,
                    w4a16=self._w4a16,
                    nvfp4_wint4_scope=self._nvfp4a16,
                    fp16_ckpt_dir=self._fp16_ckpt_dir,
                    rpu_execution=self._rpu_execution,
                )
                self._rpu_execution = pending_vla._rpu_execution
                if not _wall_oss_vla_runtime_complete(pending_vla):
                    raise RPUBackendError(
                        "WallOssPolicy builder returned without complete live "
                        "Vision/LLM/action runtime ownership."
                    )
                self._prepare_ms = {
                    "build_ms": (time.perf_counter() - t0) * 1000.0
                }
                self._vla = pending_vla
                self._rpu_ready = True
                return self
            except BaseException:
                self._rpu_ready = False
                _close_wall_oss_vla(pending_vla)
                self._vla = None
                self._prepare_ms = {}
                self._graph_profile = None
                raise
        finally:
            self._rpu_build_lock.release()

    @staticmethod
    def _make_default_noise(*, horizon: int, action_dim: int, noise_seed: int) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(noise_seed)
        return torch.randn(
            (1, horizon, action_dim), generator=gen, dtype=torch.float32
        )

    @staticmethod
    def _as_action_grid(
        value,
        *,
        horizon: int,
        action_dim: int,
        name: str,
        binary: bool = False,
    ) -> torch.Tensor:
        t = (
            _binary_mask(value, name=name)
            if binary
            else _cpu_finite_tensor(value, name=name)
        )
        if tuple(t.shape) == (horizon, action_dim):
            t = t.unsqueeze(0)
        expected = (1, horizon, action_dim)
        if tuple(t.shape) != expected:
            raise ValueError(f"{name} shape {tuple(t.shape)} != expected {expected}")
        return t.contiguous()

    def _observation_inputs(
        self,
        *,
        images,
        instruction,
        proprioception,
        state_mask,
    ) -> tuple[list[Any], str, torch.Tensor, torch.Tensor]:
        image_list = list(images) if isinstance(images, (list, tuple)) else [images]
        if len(image_list) != len(self._camera_names):
            raise ValueError(
                f"WallOssPolicy expected {len(self._camera_names)} images for "
                f"camera_names={self._camera_names}, got {len(image_list)}"
            )
        if any(image is None for image in image_list):
            raise ValueError("WallOssPolicy images must not contain None")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("WallOssPolicy instruction must be a non-empty string")

        prop = _cpu_finite_tensor(
            proprioception, name="proprioception"
        ).reshape(1, 1, -1)
        default_mask = torch.ones_like(prop)
        mask = (
            default_mask
            if state_mask is None
            else _binary_mask(state_mask, name="state_mask").reshape(1, 1, -1)
        )
        if mask.numel() != prop.numel():
            raise ValueError(
                f"WallOssPolicy state_mask has {mask.numel()} entries but "
                f"proprioception has {prop.numel()}"
            )
        return image_list, instruction, prop, mask

    def _default_dof_mask(self, *, horizon: int, action_dim: int) -> torch.Tensor:
        if self._active_slots is None:
            return torch.ones((1, horizon, action_dim), dtype=torch.float32)
        row = torch.zeros((action_dim,), dtype=torch.float32)
        for slot in self._active_slots:
            if slot >= action_dim:
                raise ValueError(
                    f"active_slots entry {slot} exceeds action_dim={action_dim}"
                )
            row[slot] = 1.0
        return row[None, None, :].expand(1, horizon, action_dim).clone()

    def _dof_mask_input(
        self, dof_mask, *, horizon: int, action_dim: int
    ) -> torch.Tensor:
        mask = (
            self._default_dof_mask(horizon=horizon, action_dim=action_dim)
            if dof_mask is None
            else self._as_action_grid(
                dof_mask,
                horizon=horizon,
                action_dim=action_dim,
                name="dof_mask",
                binary=True,
            )
        )
        return mask

    @property
    def graphs_ready(self) -> bool:
        """Whether all prepared Wall-OSS execution profiles are frozen in READY."""
        return bool(
            _wall_oss_policy_runtime_complete(self)
            and self._graph_profile is not None
        )

    @property
    def precision(self) -> str:
        """Canonical precision mode selected at construction."""
        return str(self._precision)

    @property
    def graph_profile(self) -> dict[str, Any] | None:
        """Copy of the prepared finite execution envelope, or ``None``."""
        if (
            not _wall_oss_policy_runtime_complete(self)
            or self._graph_profile is None
        ):
            return None
        profile = dict(self._graph_profile)
        profile["execution_profiles"] = tuple(
            dict(item) for item in profile["execution_profiles"])
        profile["graph_counts"] = dict(profile["graph_counts"])
        return profile

    @torch.no_grad()
    def prepare_graphs(
        self,
        *,
        images,
        instruction: str,
        proprioception,
        prefix_length_range: Sequence[int],
        state_mask=None,
        dof_mask=None,
        num_steps: int | None = None,
        noise_seed: int = 0,
    ) -> "WallOssPolicy":
        """Prebuild and freeze every graph needed by an inclusive prefix range.

        The range is the real processor-output length after image-token expansion
        and before execution padding. ``images``/``instruction``/``proprioception``
        are a representative production observation: preparation runs it once
        end-to-end to warm Vision and glue, then builds the remaining finite
        prefill/denoise profiles. After this succeeds, an out-of-range prefix,
        different image geometry, horizon, denoise-step count or dof-mask execution
        mode fails before graph capture instead of triggering an online BUILD.
        """
        if not _wall_oss_policy_runtime_complete(self):
            from rpu_backend.api.errors import RPUBackendError

            raise RPUBackendError(
                "WallOssPolicy.to('rpu') must own a complete live runtime "
                "before prepare_graphs().")
        if self._graph_profile is not None:
            raise RuntimeError("WallOssPolicy graphs are already prepared")

        image_list, instruction, prop, state_mask_t = self._observation_inputs(
            images=images,
            instruction=instruction,
            proprioception=proprioception,
            state_mask=state_mask,
        )
        action_dim = self.action_dim
        horizon = self._action_horizon
        dof_t = self._dof_mask_input(
            dof_mask, horizon=horizon, action_dim=action_dim
        )
        seed = _integer_value("noise_seed", noise_seed)
        noise_t = self._make_default_noise(
            horizon=horizon, action_dim=action_dim, noise_seed=seed
        )
        steps = (
            self._num_steps
            if num_steps is None
            else _integer_at_least("num_steps", num_steps, 1)
        )
        prefix_range = _prefix_range(prefix_length_range)

        t0 = time.perf_counter()
        profile = self._vla.prepare_graphs(
            images=image_list,
            instruction=instruction,
            proprioception=prop,
            state_mask=state_mask_t,
            noise=noise_t,
            dof_mask=dof_t,
            prefix_length_range=prefix_range,
            num_steps=steps,
        )
        self._prepare_ms["graph_warmup_ms"] = (
            time.perf_counter() - t0) * 1000.0
        self._graph_profile = profile
        return self

    @torch.no_grad()
    def infer(
        self,
        *,
        images,
        instruction: str,
        proprioception,
        state_mask=None,
        noise=None,
        dof_mask=None,
        num_steps: int | None = None,
        noise_seed: int = 0,
    ) -> WallOssActionOutput:
        """Run one Wall-OSS action-chunk inference and return CPU physical actions.

        ``images``: one PIL image per entry in ``camera_names``. ``proprioception``:
        raw robot state ``[D]`` (normalized to the FLOW state string internally).
        ``state_mask``: active proprioception dims (default all active). ``noise`` /
        ``dof_mask``: ``[1, H, action_dim]`` (defaults: seeded Gaussian / active-slot
        mask). Returns de-normalized actions with shape ``[H, action_dim]``.
        """

        if not _wall_oss_policy_runtime_complete(self):
            from rpu_backend.api.errors import RPUBackendError

            raise RPUBackendError(
                "WallOssPolicy.to('rpu') must own a complete live runtime "
                "before inference."
            )

        image_list, instruction, prop, state_mask_t = self._observation_inputs(
            images=images,
            instruction=instruction,
            proprioception=proprioception,
            state_mask=state_mask,
        )
        action_dim = self.action_dim
        horizon = self._action_horizon
        seed = _integer_value("noise_seed", noise_seed)

        noise_t = (
            self._make_default_noise(
                horizon=horizon, action_dim=action_dim, noise_seed=seed
            )
            if noise is None
            else self._as_action_grid(
                noise, horizon=horizon, action_dim=action_dim, name="noise"
            )
        )
        dof_t = self._dof_mask_input(
            dof_mask, horizon=horizon, action_dim=action_dim
        )
        steps = (
            self._num_steps
            if num_steps is None
            else _integer_at_least("num_steps", num_steps, 1)
        )
        if self._graph_profile is not None:
            prepared_steps = int(self._graph_profile["num_steps"])
            if steps != prepared_steps:
                raise RuntimeError(
                    f"num_steps={steps} differs from prepared num_steps="
                    f"{prepared_steps}; online graph BUILD is disabled")
            row_constant = bool(torch.allclose(
                dof_t[0], dof_t[0, :1].expand_as(dof_t[0])))
            if row_constant != bool(self._graph_profile["row_constant"]):
                raise RuntimeError(
                    "dof_mask execution mode differs from the prepared graph; "
                    "online graph BUILD is disabled")

        t0 = time.perf_counter()
        out = self._vla.predict(
            image_list, instruction, prop, state_mask_t, noise_t, dof_t,
            num_steps=steps)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        from rpu_backend.api.errors import RPUBackendError

        if not isinstance(out, Mapping) or "action" not in out:
            raise RPUBackendError(
                "WallOssPolicy runtime must return a mapping containing 'action'"
            )
        actions = out["action"]
        if not isinstance(actions, torch.Tensor):
            raise RPUBackendError(
                "WallOssPolicy runtime 'action' must be a torch.Tensor"
            )
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        actions = actions.detach().float().cpu().contiguous()
        expected_shape = (horizon, action_dim)
        if tuple(actions.shape) != expected_shape:
            raise RPUBackendError(
                f"WallOssPolicy runtime action shape {tuple(actions.shape)} != "
                f"expected {expected_shape}"
            )
        if not bool(torch.isfinite(actions).all()):
            raise RPUBackendError(
                "WallOssPolicy runtime action contains non-finite values"
            )

        n_images = len(image_list)
        return WallOssActionOutput(
            actions=actions,
            action_hz=self._action_hz,
            latency_ms=latency_ms,
            phase_ms={},  # ponytail: predict() returns no per-phase timing; wrap sub-stages when a caller needs it
            extra={
                "ckpt_dir": self._ckpt_dir,
                "dataset_key": str(out.get("dataset_key", self._dataset_key)),
                "prefix_len": int(out.get("prefix_len", 0)),
                "real_prefix_len": int(out.get("real_prefix_len", 0)),
                "num_images": int(n_images),
                "num_steps": int(steps),
                "action_chunk_size_resolved": (
                    (int(horizon) + 15) // 16
                ) * 16,
                "noise_seed": seed,
                "prepare_ms": dict(self._prepare_ms),
                "graphs_ready": self.graphs_ready,
                "precision": self.precision,
            },
        )

    @torch.no_grad()
    def predict_action_chunk(self, **kwargs: Any) -> torch.Tensor:
        """Return only the CPU physical action chunk with shape ``[H, action_dim]``."""

        return self.infer(**kwargs).actions
