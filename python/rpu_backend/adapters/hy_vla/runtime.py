"""Hy-Embodied-0.5-VLA 的子系统生命周期与 image → action 运行时。

相机图像经 ViT 和 merger 组装为 prefix，VLM prefill 写入共享 KV，
action expert 执行 Euler denoise。三个 native 子系统由本模块统一管理。

权重必须在 forward 前全部就位，create 与 set_weights 成对执行。
首次 BUILD 的输出参与正常计算。持久执行路径复用 handle 与 GraphCache，
布局改变时整体失效重建；非持久路径按阶段释放子系统资源。
临时 SPM 回收必须晚于输出物化，可变输入使用相应的稳定槽或 mutable DMA。"""
from __future__ import annotations

import dataclasses
import copy
import os
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

import rpu_backend
from rpu_backend.api._execution import (
    bind_execution_session,
    execution_serialized,
    native_execution_reconfigure,
    normalize_rpu_execution,
    resolve_component_rpu_execution,
)
from rpu_backend.api.cache import RPUCache
from rpu_backend.adapters.hy_vla.weights import (HyVlaConfig, HyVlaWeights,
                                                 _HyVlaWeightColdPlan,
                                                 build_rope_tables,
                                                 build_time_embed,
                                                 build_unroll_weights,
                                                 load_hy_vla_weights,
                                                 preflight_hy_vla_checkpoint)
from rpu_backend.runtime.decoder import plan_bounded_prefill_execution
from rpu_backend.runtime.execution_planner import GRAPH_COMPOSITE_CHILD


HY_VLA_VISION_COMPONENT = "vision_encoder"
HY_VLA_LANGUAGE_COMPONENT = "language_model"
HY_VLA_ACTION_COMPONENT = "action_expert"
HY_VLA_EXECUTION_COMPONENTS = {
    HY_VLA_VISION_COMPONENT: {"vision": ("chunk_size",)},
    HY_VLA_LANGUAGE_COMPONENT: {"prefill": ("chunk_size",)},
    HY_VLA_ACTION_COMPONENT: {"action": ("chunk_size",)},
}
_HY_VLA_COMPONENT_AUTO = {
    HY_VLA_VISION_COMPONENT: {"vision": {"chunk_size": "auto"}},
    HY_VLA_LANGUAGE_COMPONENT: {"prefill": {"chunk_size": "auto"}},
    HY_VLA_ACTION_COMPONENT: {"action": {"chunk_size": "auto"}},
}
_HY_VLA_SUB_NAME = {
    HY_VLA_VISION_COMPONENT: "vit",
    HY_VLA_LANGUAGE_COMPONENT: "vlm",
    HY_VLA_ACTION_COMPONENT: "expert",
}
_RMSNORM_CAP_BASE = 1
_RMSNORM_CAP_V16 = 2
_RMSNORM_CAP_V32 = 4


def resolve_hy_vla_execution(value, *, entry_point: str):
    """Resolve the one public mapping into three stable physical children."""
    root = normalize_rpu_execution(
        value,
        entry_point=entry_point,
        supported_components=HY_VLA_EXECUTION_COMPONENTS,
    )
    children = {
        component: resolve_component_rpu_execution(
            root,
            component,
            entry_point=entry_point,
            supported_components=HY_VLA_EXECUTION_COMPONENTS,
            profile_auto=_HY_VLA_COMPONENT_AUTO[component],
        )
        for component in HY_VLA_EXECUTION_COMPONENTS
    }
    return root, children


def _native_chunk(config, stage: str) -> int:
    value = config[stage]["chunk_size"]
    return 0 if value == "auto" else int(value)


def _validate_hy_vla_execution_geometry(
    children,
    *, cfg: HyVlaConfig,
    prefix_len: int,
    packed_vision: bool,
    entry_point: str,
) -> None:
    """Validate fixed ABI geometry; native FMB remains the tiling authority."""
    vlm_chunk = _native_chunk(children[HY_VLA_LANGUAGE_COMPONENT], "prefill")
    if vlm_chunk not in (0, int(prefix_len)):
        raise ValueError(
            f"{entry_point}: language_model prefill.chunk_size must be 'auto' "
            f"or exactly prefix_len={prefix_len}; got {vlm_chunk}"
        )
    action_physical = ((int(cfg.suffix_len) + 15) // 16) * 16
    action_chunk = _native_chunk(children[HY_VLA_ACTION_COMPONENT], "action")
    if action_chunk not in (0, action_physical):
        raise ValueError(
            f"{entry_point}: action_expert action.chunk_size must be 'auto' "
            f"or exactly {action_physical} for the {cfg.suffix_len}-row suffix; "
            f"got {action_chunk}"
        )
    vision_chunk = _native_chunk(children[HY_VLA_VISION_COMPONENT], "vision")
    vision_len = int(cfg.vit_seq) * (int(cfg.num_cameras) if packed_vision else 1)
    vision_physical = ((vision_len + 15) // 16) * 16
    if packed_vision and vision_chunk not in (0, vision_physical):
        raise ValueError(
            f"{entry_point}: packed vision_encoder vision.chunk_size must be "
            f"'auto' or exactly {vision_physical}; got {vision_chunk}"
        )
    if not packed_vision and vision_chunk not in (0, vision_physical):
        raise ValueError(
            f"{entry_point}: vision_encoder vision.chunk_size must be 'auto' "
            f"or exactly ceil16(vit_seq)={vision_physical}; got {vision_chunk}"
        )


_HY_VLA_ENV_DEFAULTS = {
    "RPU_HY_VLA_DENOISE_UNROLL": "1",
    "RPU_HY_VLA_PERSIST_HANDLES": "1",
    "RPU_HY_VLA_VIT_PACKED": "1",
    "RPU_KVINSERT_HYBRID_V16": "1",
    "RPU_KVINSERT_V16_ANY_TP": "1",
    "RPU_HY_VLA_FUSED_MERGER": "1",
    "RPU_HY_VLA_MERGER_IN_GRAPH": "1",
    "RPU_HY_VLA_PATCH_EMBED_MC": "1",
    "RPU_HY_VLA_PATCH_EMBED_IN_GRAPH": "1",
    "RPU_HY_VLA_FAST_REPLAY": "1",
    "RPU_HY_VLA_FAST_REPLAY_PRELOAD": "1",
    "RPU_HY_VLA_ATTN_TP8": "1",
    "RPU_HY_VLA_MASK_ONCE": "1",
    "RPU_HY_VLA_Q_INPLACE": "1",
    "RPU_HY_VLA_KVPAD16": "1",
    "RPU_HY_VLA_SILU_MUL": "expert",
    "RPU_RMSNORM_VWARP": "auto",
    "RPU_SKIP_IDLE_RECORD_FUNCTION": "1",
    "RPU_HY_VLA_PARTIAL_ROPE": "expert,vlm",
    "RPU_FASTREPLAY_SKIP_SYNC": "1",
}
_HY_VLA_COMPONENTS = frozenset(("vit", "vlm", "expert"))


def _setdefault_env(
    key: str,
    cold_plan: "_HyVlaColdPlan",
    snapshot: dict[str, str | None] | None,
) -> str:
    """Publish one value from the pre-claim plan and record the key we own."""
    planned = dict(cold_plan.default_environment)
    if key not in planned:
        raise KeyError(f"{key} is not a HyVLA builder-owned default")
    if snapshot is not None and key not in snapshot:
        snapshot[key] = os.environ.get(key)
    os.environ[key] = planned[key]
    return planned[key]


def _environment_value(
    environment: Mapping[str, str],
    key: str,
    default: str = "",
) -> str:
    return str(environment.get(key, default)).strip()


def _component_selector(
    environment: Mapping[str, str],
    key: str,
) -> frozenset[str]:
    """Parse one exact per-owner selector; typos must never silently disable it."""
    raw = _environment_value(environment, key)
    value = raw.lower()
    if value in ("", "0", "off", "false"):
        return frozenset()
    if value in ("1", "on", "true", "all"):
        return _HY_VLA_COMPONENTS
    selected = frozenset(
        token.strip().lower() for token in raw.split(",") if token.strip()
    )
    invalid = selected - _HY_VLA_COMPONENTS
    if invalid:
        raise ValueError(
            f"{key}={raw!r}: unknown components {sorted(invalid)}; "
            "use vit, vlm, expert, all, or off"
        )
    if not selected:
        raise ValueError(
            f"{key}={raw!r}: invalid empty component selector; "
            "use vit, vlm, expert, all, or off"
        )
    return selected


def _parse_bool(
    environment: Mapping[str, str],
    key: str,
    *,
    empty: bool,
) -> bool:
    value = _environment_value(environment, key).lower()
    if value == "":
        return empty
    if value in ("0", "off", "false"):
        return False
    if value in ("1", "on", "true"):
        return True
    raise ValueError(
        f"{key} must be one of 0/off/false or 1/on/true; got {value!r}"
    )


def _hyvla_mot_norm_nomerge(
    environment: Mapping[str, str] | None = None,
) -> int:
    source = os.environ if environment is None else environment
    key = "RPU_HY_VLA_MOT_NORM_NOMERGE"
    raw = _environment_value(source, key)
    value = raw.lower()
    if value in ("0", "off", "false"):
        return 0
    if value in ("", "1", "on", "true", "both"):
        return 3
    tokens = frozenset(
        token.strip() for token in value.split(",") if token.strip()
    )
    invalid = tokens - {"qkv", "mlp"}
    if invalid or not tokens:
        raise ValueError(
            f"{key}={raw!r}: unknown components {sorted(invalid)}; "
            "use qkv, mlp, both, or off"
        )
    return (1 if "qkv" in tokens else 0) | (2 if "mlp" in tokens else 0)


def _rmsnorm_capability_from_env(
    environment: Mapping[str, str] | None = None,
) -> int:
    source = os.environ if environment is None else environment
    value = _environment_value(source, "RPU_RMSNORM_VWARP").lower()
    if value in ("", "0", "off", "false"):
        return _RMSNORM_CAP_BASE
    if value == "16":
        return _RMSNORM_CAP_BASE | _RMSNORM_CAP_V16
    if value == "32":
        return _RMSNORM_CAP_BASE | _RMSNORM_CAP_V32
    if value == "auto":
        return _RMSNORM_CAP_BASE | _RMSNORM_CAP_V16 | _RMSNORM_CAP_V32
    raise ValueError(
        "RPU_RMSNORM_VWARP must be one of 0, 16, 32, or auto; "
        f"got {value!r}"
    )


@dataclasses.dataclass(frozen=True)
class _HyVlaNativeOwnerConfig:
    fast_replay: bool = False
    fast_replay_preload: bool = False
    mask_once: bool = False
    silu_mul: bool = False
    kvpad16: bool = False
    partial_rope: bool = False
    rmsnorm_pad16: bool = False
    mot_norm_nomerge: int = 0
    q_inplace: bool = False
    rmsnorm_capability: int = _RMSNORM_CAP_BASE


@dataclasses.dataclass(frozen=True)
class _HyVlaNativeColdSnapshot:
    """Immutable Python authority for the three native physical owners."""
    vit: _HyVlaNativeOwnerConfig
    vlm: _HyVlaNativeOwnerConfig
    expert: _HyVlaNativeOwnerConfig

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "_HyVlaNativeColdSnapshot":
        source = os.environ if environment is None else environment
        selectors = {
            key: _component_selector(source, key)
            for key in (
                "RPU_HY_VLA_FAST_REPLAY",
                "RPU_HY_VLA_FAST_REPLAY_PRELOAD",
                "RPU_HY_VLA_MASK_ONCE",
                "RPU_HY_VLA_SILU_MUL",
                "RPU_HY_VLA_KVPAD16",
                "RPU_HY_VLA_PARTIAL_ROPE",
                "RPU_HY_VLA_RMSNORM_PAD16",
            )
        }

        def selected(key: str, component: str) -> bool:
            return component in selectors[key]

        return cls(
            vit=_HyVlaNativeOwnerConfig(
                fast_replay=selected("RPU_HY_VLA_FAST_REPLAY", "vit"),
                fast_replay_preload=selected(
                    "RPU_HY_VLA_FAST_REPLAY_PRELOAD", "vit"
                ),
                mask_once=selected("RPU_HY_VLA_MASK_ONCE", "vit"),
                kvpad16=selected("RPU_HY_VLA_KVPAD16", "vit"),
                q_inplace=_parse_bool(
                    source, "RPU_HY_VLA_Q_INPLACE", empty=False
                ),
            ),
            vlm=_HyVlaNativeOwnerConfig(
                fast_replay=selected("RPU_HY_VLA_FAST_REPLAY", "vlm"),
                fast_replay_preload=selected(
                    "RPU_HY_VLA_FAST_REPLAY_PRELOAD", "vlm"
                ),
                mask_once=selected("RPU_HY_VLA_MASK_ONCE", "vlm"),
                silu_mul=selected("RPU_HY_VLA_SILU_MUL", "vlm"),
                partial_rope=selected("RPU_HY_VLA_PARTIAL_ROPE", "vlm"),
                mot_norm_nomerge=_hyvla_mot_norm_nomerge(source),
                rmsnorm_capability=_rmsnorm_capability_from_env(source),
            ),
            expert=_HyVlaNativeOwnerConfig(
                fast_replay=selected("RPU_HY_VLA_FAST_REPLAY", "expert"),
                fast_replay_preload=selected(
                    "RPU_HY_VLA_FAST_REPLAY_PRELOAD", "expert"
                ),
                mask_once=selected("RPU_HY_VLA_MASK_ONCE", "expert"),
                silu_mul=selected("RPU_HY_VLA_SILU_MUL", "expert"),
                kvpad16=selected("RPU_HY_VLA_KVPAD16", "expert"),
                partial_rope=selected("RPU_HY_VLA_PARTIAL_ROPE", "expert"),
                rmsnorm_pad16=selected(
                    "RPU_HY_VLA_RMSNORM_PAD16", "expert"
                ),
            ),
        )

    @classmethod
    def from_env(cls) -> "_HyVlaNativeColdSnapshot":
        """Compatibility wrapper; production construction passes a frozen map."""
        return cls.from_environment()


def _parse_persist_components(
    environment: Mapping[str, str],
) -> frozenset[str]:
    key = "RPU_HY_VLA_PERSIST_HANDLES"
    raw = _environment_value(environment, key)
    value = raw.lower()
    if value in ("1", "true", "on", "all"):
        return _HY_VLA_COMPONENTS
    if value in ("0", "", "false", "off"):
        return frozenset()
    selected = frozenset(
        token.strip().lower() for token in raw.split(",") if token.strip()
    )
    invalid = selected - _HY_VLA_COMPONENTS
    if invalid:
        raise ValueError(
            f"{key}={raw!r}: unknown components {sorted(invalid)}; "
            "use vit, vlm, expert, all, or off"
        )
    if not selected:
        raise ValueError(
            f"{key}={raw!r}: invalid empty component selector; "
            "use vit, vlm, expert, all, or off"
        )
    return selected


@dataclasses.dataclass(frozen=True)
class _HyVlaRunnerColdSnapshot:
    unroll: bool
    vit_packed: bool
    fused_merger: bool
    merger_in_graph: bool
    patch_embed_in_graph: bool
    proj1_in_merger: bool
    prefix_template: bool
    persist: frozenset[str]

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str],
    ) -> "_HyVlaRunnerColdSnapshot":
        return cls(
            unroll=_parse_bool(
                environment, "RPU_HY_VLA_DENOISE_UNROLL", empty=False
            ),
            vit_packed=_parse_bool(
                environment, "RPU_HY_VLA_VIT_PACKED", empty=False
            ),
            fused_merger=_parse_bool(
                environment, "RPU_HY_VLA_FUSED_MERGER", empty=False
            ),
            merger_in_graph=_parse_bool(
                environment, "RPU_HY_VLA_MERGER_IN_GRAPH", empty=False
            ),
            patch_embed_in_graph=_parse_bool(
                environment, "RPU_HY_VLA_PATCH_EMBED_IN_GRAPH", empty=True
            ),
            proj1_in_merger=_parse_bool(
                environment, "RPU_HY_VLA_PROJ1_IN_MERGER", empty=True
            ),
            prefix_template=_parse_bool(
                environment, "RPU_HY_VLA_PREFIX_TEMPLATE", empty=True
            ),
            persist=_parse_persist_components(environment),
        )


@dataclasses.dataclass(frozen=True)
class _HyVlaColdPlan:
    """Complete immutable host plan resolved before process ownership."""

    native: _HyVlaNativeColdSnapshot
    weights: _HyVlaWeightColdPlan
    runner: _HyVlaRunnerColdSnapshot
    caching_allocator: bool
    default_environment: tuple[tuple[str, str], ...]


def prepare_hy_vla_cold_plan(
    overrides: Mapping[str, str] | None = None,
) -> _HyVlaColdPlan:
    """Resolve every HyVLA cold/layout/quant selector without global mutation."""
    effective = dict(os.environ)
    if overrides is not None:
        effective.update({str(key): str(value) for key, value in overrides.items()})
    for key, value in _HY_VLA_ENV_DEFAULTS.items():
        effective.setdefault(key, value)
    return _HyVlaColdPlan(
        native=_HyVlaNativeColdSnapshot.from_environment(effective),
        weights=_HyVlaWeightColdPlan.from_environment(effective),
        runner=_HyVlaRunnerColdSnapshot.from_environment(effective),
        caching_allocator=_parse_bool(
            effective, "RPU_HY_VLA_CACHING_ALLOC", empty=True
        ),
        default_environment=tuple(
            (key, effective[key]) for key in _HY_VLA_ENV_DEFAULTS
        ),
    )


class _DirectHyVlaOwner:
    """Weak-referenceable process owner for the low-level builder entry."""


# RPU_HY_VLA_PERSIST_HANDLES 控制跨帧 handle 与 GraphCache 复用。
# RPU_HY_VLA_VIT_PACKED 将相机序列合并，attention 仍须保持逐图隔离；
# minibatch SDPA 的每图范围由 per_image_ctx 指定。
MASK_NEG = -50000.0   # fp16 安全的 additive mask 常量。vendor 用 -2.38e38（bf16
                      # 的 -inf 近似），在 fp16 上必然溢出；score 量级 ~1e2，
                      # 加 -50000 后仍在 fp16 范围内，softmax 结果为 0。


def _rpu16(t: torch.Tensor) -> torch.Tensor:
    return t.to(dtype=torch.float16, device="rpu").contiguous()


class _Sub:
    """一座子系统的生命周期：create+bind → 用 → destroy。

    create 与 bind 绑成一步（不留空 handle）；每次 open
    配一个新的 `GraphCache`；成功 `close()` 清图、destroy 后才 `reset_all()`。
    任一阶段失败即保留未退休资源并禁止重试，避免重置仍被 Graph 引用的地址。
    首次 BUILD 的输出可直接使用。
    """

    def __init__(self, name: str, create, destroy, bind: Callable[[int], None],
                 sig: "rpu_backend.graph.GraphSignature", persist: bool = False,
                 variant=None, owner=None):
        self._name, self._create, self._destroy, self._bind = name, create, destroy, bind
        self.sig = sig
        self.persist = persist        # True ⇒ 出作用域不 destroy，跨帧存活
        self.variant = variant        # bind 闭包的语义指纹；变了就必须重建
        self.handle: Optional[int] = None
        self.gc: Optional["rpu_backend.graph.GraphCache"] = None
        self._owner = owner
        self._failed = False

    def _require_usable(self) -> None:
        if self._failed or getattr(self._owner, "_retirement_failed", False):
            raise RuntimeError(f"{self._name}: native lifecycle failed; restart the process")

    def _fail(self) -> None:
        if not self._failed:
            self._failed = True
            if self._owner is not None:
                self._owner._poison_retirement(self)

    def open(self) -> None:
        self._require_usable()
        assert self.handle is None, f"{self._name}: 已经 open"
        try:
            self.handle = int(self._create())
            self._bind(self.handle)     # ← 与 create 绑成一步：不留空 handle
            self.gc = rpu_backend.graph.GraphCache()
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass  # Preserve the original bind/create error and failed owner.
            self._fail()
            raise

    def close(self, *, _reset_spm=True, _graphs_cleared=False) -> None:
        self._require_usable()
        if self.handle is None:
            return
        try:
            if (_reset_spm and self._owner is not None and any(
                    sub is not self and sub.handle is not None
                    for sub in self._owner._psubs.values())):
                # Partial-persist diagnostics retain their configuration, but a
                # global reset must retire every sibling Graph first.
                self._owner._psubs[self._name] = self
                self._owner._drop_psubs()
                return
            if self.gc is not None and not _graphs_cleared:
                self.gc.clear()
            self._destroy(self.handle)
            self.handle = None
            if _reset_spm:
                torch.ops.rpu.spm_alloc_reset_all()
            self.gc = None
        except BaseException:
            self._fail()
            raise

    def __enter__(self):
        self._require_usable()
        if self.handle is None:       # 持久模式下只有第一帧真正 open
            self.open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            # A failed capture/forward has no proven retirement boundary.
            self._fail()
        elif not self.persist:
            self.close()
        return False


@dataclasses.dataclass
class HyVlaRunner:
    """一次 `get_action` = 一条完整的 image → action 链路。

    `RPU_HY_VLA_PERSIST_HANDLES=1`（默认）使三座子系统跨帧存活。
    prompt-layout 改变时，`_cached` 更新常量并作废旧图；复用前提见 `_sub`。
    设为 `0` 则每帧重建三座，并重新将权重从 DDR preload 到 SPM。
    权重本身只在 `build_hy_vla()` 时 swizzle 一次，常驻 RPU DDR。
    """
    w: HyVlaWeights
    prefix_len: int                 # S = ceil16(有效 prefix 行数)
    _native_cold: _HyVlaNativeColdSnapshot = dataclasses.field(repr=False)
    _unroll: bool
    _vit_packed: bool
    _fused_merger: bool
    _merger_in_graph: bool
    _pe_in_graph: bool
    _proj1_in_merger: bool
    _prefix_template: bool
    _persist: frozenset[str]
    rpu_execution: dataclasses.InitVar[Mapping[str, Any] | None] = None
    _embed: Callable = dataclasses.field(repr=False, default=None)
    _vit_cache: "RPUCache" = dataclasses.field(repr=False, default=None)
    _cache: "RPUCache" = dataclasses.field(repr=False, default=None)
    _out_proj_wt: torch.Tensor = dataclasses.field(repr=False, default=None)
    _out_proj_b: torch.Tensor = dataclasses.field(repr=False, default=None)
    _const_key: tuple = dataclasses.field(repr=False, default=None)
    _const_val: tuple = dataclasses.field(repr=False, default=None)
    _prefix_key: tuple = dataclasses.field(repr=False, default=None)
    _prefix_val: tuple = dataclasses.field(repr=False, default=None)
    _psubs: dict = dataclasses.field(repr=False, default=None)
    _uw: tuple = dataclasses.field(repr=False, default=None)
    _ux0: torch.Tensor = dataclasses.field(repr=False, default=None)
    _utraj: torch.Tensor = dataclasses.field(repr=False, default=None)
    _closed: bool = dataclasses.field(repr=False, default=False, init=False)
    _retirement_failed: bool = dataclasses.field(repr=False, default=False, init=False)

    def __post_init__(self, rpu_execution):
        root, children = resolve_hy_vla_execution(
            rpu_execution, entry_point="build_hy_vla"
        )
        _validate_hy_vla_execution_geometry(
            children,
            cfg=self.w.cfg,
            prefix_len=self.prefix_len,
            packed_vision=self._vit_packed,
            entry_point="build_hy_vla",
        )
        self._rpu_execution = root
        self._rpu_execution_components = children
        self._rpu_execution_component_generations = {
            component: 0 for component in HY_VLA_EXECUTION_COMPONENTS
        }
        self._rpu_last_execution_receipts = {}
        self._execution_reconfigure_journal = None
        self._embed = build_time_embed(self.w.cfg, self.w.host)
        c = self.w.cfg
        # action_out_proj 使用预转置权重与 addmm，避免每步重新处理小投影布局。
        # 较大的 encoder GEMM 保留各自的 F.linear 布局。
        self._out_proj_wt = self.w.host["action_out_proj.weight"].t().contiguous()
        self._out_proj_b = self.w.host["action_out_proj.bias"]
        self._const_key, self._const_val = {}, {}
        self._psubs = {}
        if self._unroll:
            # 图内 unroll 的常驻缓冲：x0 行 0 是 state 位（恒 0，encoder 对它的输出
            # 从不进 emb_stage_，是死路），行 1: 每帧填 noise；x_traj 收逐步轨迹，
            # 终值 = [-1]。两者地址跨帧稳定 ⇒ mutable DMA 每帧只改基址。
            self._uw = build_unroll_weights(
                c, self.w.host, self.w.action_mlp_cores)
            self._ux0 = torch.zeros(1, c.suffix_len, c.action_dim,
                                    dtype=torch.float16, device="rpu")
            self._utraj = torch.empty(c.num_steps, c.suffix_len, c.action_dim,
                                      dtype=torch.float16, device="rpu")
        # KV cache 在 runner 生命周期内按固定 cfg 与 prefix_len 分配并跨帧复用。
        # 复用依赖 overwrite-before-read：ViT/prefill 重置到 0，denoise 重置到 S。
        # native attention 只读取逻辑 kv_seq_len，不读取额外的物理 padding 行。
        self._total = ((self.prefix_len + c.suffix_len + 15) // 16) * 16
        # packed 路径把 N 相机拼成一条 N*196 的序列 ⇒ ViT 的 KV cache 要够长。
        self._vit_cache = RPUCache(
            num_layers=c.vit_layers, batch_size=1,
            max_seq_len=c.vit_seq * (c.num_cameras if self._vit_packed else 1),
            num_kv_heads=c.vit_heads, head_dim=c.vit_head_dim_padded, attn_tp=8)
        # ⚠️ VLM 与 expert **共享这一个 cache**（expert 把 suffix KV 追加在
        # prefix 之后）⇒ 候选 G 的 KV 槽位复制必须**两座一起做**，不能只改一座。
        self._cache = RPUCache(
            num_layers=c.layers, batch_size=1, max_seq_len=self._total,
            num_kv_heads=self.w.kv_heads_rpu, head_dim=c.head_dim,
            attn_tp=self.w.attn_tp)
        self._execution_session = bind_execution_session(
            self,
            root,
            entry_point="HyVlaRunner",
            supported_components=HY_VLA_EXECUTION_COMPONENTS,
            validate=self.validate_execution_reconfigure,
            apply=self.apply_execution_reconfigure,
            rollback=self.rollback_execution_reconfigure,
            graph_mode=GRAPH_COMPOSITE_CHILD,
        )
        for component in HY_VLA_EXECUTION_COMPONENTS:
            self._execution_session._bind_planner_owner(self, component)

    @property
    def last_rpu_execution_plan(self) -> dict[str, dict[str, Any]]:
        """Return detached per-child physical planning receipts."""
        return copy.deepcopy(self._rpu_last_execution_receipts)

    def _component_config(self, component: str, stage: str):
        return self._rpu_execution_components[component][stage]

    def _component_generation(self, component: str) -> int:
        return int(self._rpu_execution_component_generations[component])

    def _initialize_native_execution(
        self, component: str, handle: int
    ) -> None:
        stage = {
            HY_VLA_VISION_COMPONENT: "vision",
            HY_VLA_LANGUAGE_COMPONENT: "prefill",
            HY_VLA_ACTION_COMPONENT: "action",
        }[component]
        prefix = {
            HY_VLA_VISION_COMPONENT: "hyvit2",
            HY_VLA_LANGUAGE_COMPONENT: "hyvla_vlm",
            HY_VLA_ACTION_COMPONENT: "hyvla_expert",
        }[component]
        chunk = _native_chunk(
            self._rpu_execution_components[component], stage
        )
        getattr(torch.ops.rpu, f"{prefix}_set_chunk_size_override")(
            int(handle), int(chunk)
        )
        getattr(torch.ops.rpu, f"{prefix}_enable_execution_reconfigure")(
            int(handle)
        )

    def _plan_execution(
        self,
        *,
        component: str,
        stage: str,
        logical_len: int,
        position: int,
        sub: _Sub,
        resolve_stage_domain,
        envelope: Mapping[str, Any],
        spans: Sequence[Mapping[str, Any]],
        kv_route: str,
        route_reason: str,
        plan_signature=(),
    ):
        """Select one native A6 descriptor for a physical child owner."""
        requested = self._component_config(component, stage)["chunk_size"]
        generation = self._component_generation(component)
        box = {}
        execution_len, chunk_size = plan_bounded_prefill_execution(
            int(logical_len),
            int(logical_len),
            0,
            execution_owner=self,
            execution_component=component,
            execution_stage=stage,
            execution_native=({
                HY_VLA_VISION_COMPONENT: "hyvit2",
                HY_VLA_LANGUAGE_COMPONENT: "hyvla_vlm",
                HY_VLA_ACTION_COMPONENT: "hyvla_expert",
            }[component], int(sub.handle)),
            position=int(position),
            alignment=1,
            padding_rows=0,
            exact_chunk_size=(
                None if requested == "auto" else int(requested)
            ),
            resolve_stage_domain=resolve_stage_domain,
            request_id=f"hy-vla:{component}:{stage}",
            plan_result_sink=lambda result: box.__setitem__("result", result),
            graph_mode=GRAPH_COMPOSITE_CHILD,
            queue_owner_id=id(sub.gc),
            physical_metadata=(
                (f"component:{component}", 1),
                ("component_generation", generation),
                ("kv_insert_ddr_required", 1 if kv_route == "DDR_REQUIRED" else 0),
                (f"route_reason:{route_reason}", 1),
            ),
            plan_signature=(tuple(plan_signature), tuple(sorted(envelope.items())), tuple(tuple(sorted(span.items())) for span in spans)),
            graph_cache=sub.gc,
        )
        if execution_len != int(logical_len):
            raise RuntimeError(
                f"Hy-VLA {component} planner changed an unpadded length: "
                f"logical={logical_len}, execution={execution_len}"
            )
        plan = box["result"]
        descriptor = plan.selected.stage_tuple.physical_descriptor
        if not descriptor:
            raise RuntimeError(
                f"Hy-VLA {component} A6 winner has no native descriptor"
            )
        metadata = dict(plan.selected.stage_tuple.physical_metadata)
        if metadata.get("manifest_state") != 1:
            raise RuntimeError(
                f"Hy-VLA {component} A6 winner has no COMPLETE physical "
                "manifest for descriptor forward"
            )
        raw_attention = metadata.get("raw_attention_site_count", 0) > 0
        ddr_attention = metadata.get("capability_attention_ddr_site_count", 0) > 0
        if raw_attention == ddr_attention:
            raise RuntimeError(
                f"Hy-VLA {component} descriptor must select one attention policy"
            )
        return plan, tuple(descriptor), int(chunk_size), {
            "envelope": dict(envelope),
            "spans": [dict(span) for span in spans],
            "kv_route": kv_route,
            "attention_route": "RAW_SPM" if raw_attention else "DDR_REQUIRED",
            "kv_insert_route": kv_route,
            "attention_ddr_required": int(ddr_attention),
            "kv_insert_ddr_required": int(kv_route == "DDR_REQUIRED"),
            "route_reason": route_reason,
        }

    def _record_execution_receipt(
        self,
        *,
        component: str,
        stage: str,
        sub: _Sub,
        plan,
        descriptor,
        native_chunk: int,
        details: Mapping[str, Any],
    ) -> None:
        selected = plan.selected
        if (
            selected is None
            or int(selected.stage_tuple.compute_chunk) != int(native_chunk)
            or tuple(selected.stage_tuple.physical_descriptor) != tuple(descriptor)
        ):
            raise RuntimeError(
                f"Hy-VLA {component} native forward disagrees with its A6 winner"
            )
        receipt = plan.as_dict(include_candidates=False)
        receipt.update({
            "component": component,
            "stage": stage,
            "authority": "NATIVE_A6_STAGE_DESCRIPTOR",
            "capability": {
                stage: {
                    "chunk_size": ("auto", "exact"),
                    "full_descriptor_forward": True,
                }
            },
            # Agreement above authorizes reuse of the canonical immutable
            # integer tuple; avoid converting the same wire on every receipt.
            "descriptor": selected.stage_tuple.physical_descriptor,
            "native_resolved_chunk_size": int(native_chunk),
            "graph_owner": {
                "mode": GRAPH_COMPOSITE_CHILD,
                "id": id(sub.gc),
                "component_generation": self._component_generation(component),
                "session_generation": int(self._execution_session.generation),
            },
            **copy.deepcopy(dict(details)),
        })
        self._rpu_last_execution_receipts[component] = receipt

    @staticmethod
    def _clear_graph_owner(cache, *, component: str) -> None:
        if cache is None:
            return
        begin_warmup = getattr(cache, "begin_warmup", None)
        if begin_warmup is not None:
            begin_warmup()
        cache.clear()
        invariant = getattr(cache, "cache_invariant_ok", None)
        if invariant is not None and not invariant():
            raise RuntimeError(
                f"Hy-VLA {component} GraphCache invariant failed during reconfigure"
            )

    def validate_execution_reconfigure(self, execution_config) -> None:
        if self._closed:
            raise RuntimeError("HyVlaRunner is closed")
        _root, children = resolve_hy_vla_execution(
            execution_config, entry_point="HyVlaRunner.reconfigure"
        )
        _validate_hy_vla_execution_geometry(
            children,
            cfg=self.w.cfg,
            prefix_len=self.prefix_len,
            packed_vision=self._vit_packed,
            entry_point="HyVlaRunner.reconfigure",
        )
        required = (
            "execution_reconfigure_begin",
            "execution_reconfigure_commit",
            "execution_reconfigure_abort",
            "execution_reconfigure_abort_attempt",
            "hyvit2_stage_chunk_size_override",
            "hyvla_vlm_stage_chunk_size_override",
            "hyvla_expert_stage_chunk_size_override",
        )
        missing = [name for name in required if not hasattr(torch.ops.rpu, name)]
        if missing:
            raise RuntimeError(
                "Hy-VLA binary lacks native execution-reconfigure op(s): "
                + ", ".join(missing)
            )

    def _stage_changed_native_chunks(self, changed, children) -> bool:
        live = []
        for component in changed:
            sub = self._psubs.get(_HY_VLA_SUB_NAME[component])
            if sub is not None and sub.handle is not None:
                live.append((component, sub.handle))
        if not live:
            return False
        with native_execution_reconfigure(
            torch.ops.rpu, journal=self._execution_reconfigure_journal,
        ) as token:
            for component, handle in live:
                stage, prefix = {
                    HY_VLA_VISION_COMPONENT: ("vision", "hyvit2"),
                    HY_VLA_LANGUAGE_COMPONENT: ("prefill", "hyvla_vlm"),
                    HY_VLA_ACTION_COMPONENT: ("action", "hyvla_expert"),
                }[component]
                getattr(
                    torch.ops.rpu, f"{prefix}_stage_chunk_size_override"
                )(
                    int(handle), int(token),
                    _native_chunk(children[component], stage),
                )
        return True

    def apply_execution_reconfigure(
        self, old_config, new_config, generation: int, *, force_rebuild=False
    ) -> None:
        self._execution_reconfigure_journal = None
        self.validate_execution_reconfigure(new_config)
        _old_root, old_children = resolve_hy_vla_execution(
            old_config, entry_point="HyVlaRunner.reconfigure"
        )
        new_root, new_children = resolve_hy_vla_execution(
            new_config, entry_point="HyVlaRunner.reconfigure"
        )
        changed = {
            component for component in HY_VLA_EXECUTION_COMPONENTS
            if old_children[component] != new_children[component]
        }
        if force_rebuild:
            changed.update(HY_VLA_EXECUTION_COMPONENTS)
        self._execution_reconfigure_journal = {
            "mutation_started": False,
            "root": self._rpu_execution,
            "children": self._rpu_execution_components,
            "generations": dict(self._rpu_execution_component_generations),
            "changed": changed,
        }
        mutation_started = self._stage_changed_native_chunks(
            changed, new_children
        )
        self._execution_reconfigure_journal["mutation_started"] = mutation_started
        for component in changed:
            sub = self._psubs.get(_HY_VLA_SUB_NAME[component])
            if sub is not None:
                self._clear_graph_owner(sub.gc, component=component)
            self._rpu_last_execution_receipts.pop(component, None)
            self._rpu_execution_component_generations[component] += 1
        self._rpu_execution = new_root
        self._rpu_execution_components = new_children
        # ponytail: retain one undo snapshot through shared facade publication;
        # the next transaction replaces it, so a late exception can compensate.

    def rollback_execution_reconfigure(
        self, old_config, _new_config, _generation: int
    ) -> None:
        journal = self._execution_reconfigure_journal
        if journal is None:
            return
        if journal["mutation_started"]:
            _root, old_children = resolve_hy_vla_execution(
                old_config, entry_point="HyVlaRunner.rollback"
            )
            self._stage_changed_native_chunks(journal["changed"], old_children)
        for component in journal["changed"]:
            sub = self._psubs.get(_HY_VLA_SUB_NAME[component])
            if sub is not None:
                self._clear_graph_owner(sub.gc, component=component)
            self._rpu_last_execution_receipts.pop(component, None)
        self._rpu_execution = journal["root"]
        self._rpu_execution_components = journal["children"]
        self._rpu_execution_component_generations = journal["generations"]
        self._execution_reconfigure_journal = None

    # ── 子系统获取（持久 or 每帧新建）────────────────────────────────────
    def _sub(self, name, create, destroy, bind, sig, variant=None) -> "_Sub":
        """持久模式下三座子系统跨帧存活 —— handle 与 GraphCache 都不重建。

        复用前提是图里 bake 的地址跨帧稳定，或由 mutable DMA 显式更新：
          · 层 0 的 hidden 输入走 `hidden_in_src_base_` 的 **mutable DMA**，每次
            REPLAY 重新解引用（`fused_model_base.cpp:1078`）⇒ 调用方每帧新建的
            `hid` / `x_prefix` 地址漂移**不影响**；
          · 显式 2D mask 由 `dynamic_config` 拷进稳定 DDR 槽（同上注释）；
          · 权重、KV cache（`__post_init__` 建一次）、SPM 绝对地址全部稳定 ——
            SPM 是确定性 bump 分配器，每帧三段的分配序列完全相同 ⇒ 地址逐帧相同；
          · rope 表 / additive mask / x0 / x_traj 由步 6 与 unroll 钉成跨帧同一批张量。
        前提被打破的唯一入口是 prompt 变了（rope/mask 要重算），由 `_cached` 的
        miss 分支负责把三座全部关掉重建。

        三座共存时，持久 SPM 分配必须为 prefill 临时工作区留出足够空间。
        """
        if self._retirement_failed:
            raise RuntimeError("HyVlaRunner native lifecycle failed; restart the process")
        if self._closed:
            raise RuntimeError("HyVlaRunner 已 close，不能重建子系统 handle。")
        component = {
            "vit": HY_VLA_VISION_COMPONENT,
            "vlm": HY_VLA_LANGUAGE_COMPONENT,
            "expert": HY_VLA_ACTION_COMPONENT,
        }[name]

        def bind_and_guard(handle):
            bind(handle)
            self._initialize_native_execution(component, handle)

        if name not in self._persist:
            if name in self._psubs:
                self._drop_psubs()
            return _Sub(name, create, destroy, bind_and_guard, sig, owner=self)
        s = self._psubs.get(name)
        if s is not None and s.variant != variant:
            # bind 的语义变了（例如 expert 的 unroll 开关被翻转）⇒ 缓存的 handle 绑的是
            # 旧语义。Global reset 必须等所有旧 Graph/handle 退休；DDR KV 仍归 runner。
            self._drop_psubs()
            s = None
        if s is None:
            s = _Sub(
                name, create, destroy, bind_and_guard, sig,
                persist=True, variant=variant, owner=self,
            )
            self._psubs[name] = s
        return s

    def _poison_retirement(self, sub=None) -> None:
        self._retirement_failed = True
        if sub is not None:
            # A transient child also needs an owner after __enter__/__exit__ fails.
            self._psubs[sub._name] = sub
        session = getattr(self, "_execution_session", None)
        if session is not None:
            try:
                session.poison()
            except BaseException:
                pass  # The original native failure must remain the raised error.

    def _drop_psubs(self) -> None:
        """Retire all child Graph owners before the single process-wide reset."""
        if self._retirement_failed:
            raise RuntimeError("HyVlaRunner native lifecycle failed; restart the process")
        subs = tuple(self._psubs.values())
        live = any(sub.handle is not None for sub in subs)
        try:
            for sub in subs:
                sub._require_usable()
                if sub.gc is not None:
                    sub.gc.clear()
            for sub in subs:
                sub.close(_reset_spm=False, _graphs_cleared=True)
            if live:
                torch.ops.rpu.spm_alloc_reset_all()
        except BaseException:
            self._poison_retirement()
            raise
        self._psubs.clear()

    def _close_without_session(self) -> None:
        if self._retirement_failed:
            raise RuntimeError("HyVlaRunner native lifecycle failed; restart the process")
        if self._closed:
            return
        self._drop_psubs()
        self._closed = True

    def close(self) -> None:
        """Deterministically retire persistent graph/handle ownership."""
        session = getattr(self, "_execution_session", None)
        if session is None:
            self._close_without_session()
            return
        session.shutdown(self._close_without_session)

    def __del__(self):
        # 持久 handle 必须显式释放：super-persistent 只在活实例数归零时整体回收，
        # 泄漏会让下一个 runner 在 S=240 撞 SPM。测试里 `del runner; gc.collect()`
        # 之后才建第二个 runner，依赖的就是这里。
        try:
            self.close()
        except Exception:
            pass

    # ── 跨帧常量缓存 ──────────────────────────────────────────────────────
    def _cached(self, tag: str, key: Sequence[torch.Tensor], build: Callable):
        """缓存**只依赖 prompt 布局**的每帧常量（rope 表 / additive mask / modality）。

        它们是 `prefill_pos` / `prefill_mask` / `modality_mask` / `denoise_mask` /
        `suffix_pos` 的纯函数，而这几个输入在同一条 prompt 上跨帧不变 —— 但**不能
        假定**不变（prompt 变了长度就变了），所以用 `torch.equal` 按内容比对。
        命中时复用常量及其 RPU 张量；失配时重建相关常量并使旧图失效。
        """
        prev = self._const_key.get(tag)
        if prev is not None and all(
                a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
                for a, b in zip(prev, key)):
            return self._const_val[tag]
        # miss ⇒ rope 表 / mask 要换一批张量，而它们的地址被已 BUILD 的图 bake 了
        # ⇒ 持久子系统必须全部作废重建（见 `_sub` 的前提清单）。
        # A first constant has never been baked into an empty cold child graph.
        # Preserve fully bound cold handles for pre-forward cost collection;
        # changed constants or executed graphs retain the retirement boundary.
        if prev is not None or any(
                sub.gc is not None and sub.gc.snapshot()
                for sub in self._psubs.values()):
            self._drop_psubs()
        val = build()
        self._const_key[tag] = [t.clone() for t in key]
        self._const_val[tag] = val
        return val

    # ── prefix 组装（ViT → VLM 的接缝）────────────────────────────────────
    def assemble_prefix(self, img_embs: Sequence[torch.Tensor],
                        lang_tokens: torch.Tensor) -> torch.Tensor:
        """`[bos, hy_user] + 每图(vs + 7行×(7 patch + 1 split) + ve) + lang`。

        `embed_tokens` 与 `lm_head` 是 tied（ckpt 无独立 embed_tokens），所以直接
        用 `tok_emb` 查表；5 个特殊 token 的 id 由反查得到，见 `HyVlaConfig`。
        """
        c, tok = self.w.cfg, self.w.tok_emb
        g, D = c.grid, c.proj_dim
        if self._prefix_template:
            buf, slots, _ = self._prefix_tmpl(len(img_embs), lang_tokens)
            for sl, ie in zip(slots, img_embs):
                sl.copy_(ie.reshape(g, g, D))
            return buf
        split = tok[c.tok_vision_split]
        parts = [tok[c.tok_bos][None], tok[c.tok_user][None]]
        for ie in img_embs:
            parts.append(tok[c.tok_vision_start][None])
            grid = ie.reshape(g, g, D)
            grid = torch.cat([grid, split[None, None].expand(g, 1, D)], dim=1)
            parts.append(grid.reshape(g * (g + 1), D))
            parts.append(tok[c.tok_vision_end][None])
        parts.append(tok[lang_tokens.long()])
        return torch.cat(parts, 0)[None]

    def _prefix_tmpl(self, n_img: int, lang_tokens: torch.Tensor):
        """按图像数及语言 token 内容缓存 prefix 常量骨架和图像行视图。

        BOS、角色、视觉边界和语言 token 保持不变，每帧只更新每图的 49 行
        视觉 embedding。返回值复用同一块 host buffer，供本帧 prefill 立即
        上传；调用方若需跨帧保留，必须自行 clone。"""
        prev = self._prefix_key
        if not (prev is not None and prev[0] == n_img
                and torch.equal(prev[1], lang_tokens)):
            c, tok, g, D = self.w.cfg, self.w.tok_emb, self.w.cfg.grid, self.w.cfg.proj_dim
            rows = g + 1                       # 每行 7 个 patch + 1 个 split
            per = g * rows                     # 每图 56 行
            lang = tok[lang_tokens.long()]
            # prefix 骨架直接采用其消费者所需的 FP16，避免每帧再转换常量部分。
            # 逐元素转换不依赖这些行是在拼接前还是拼接后转换。
            buf = torch.empty((1, 2 + n_img * (per + 2) + lang.shape[0], D),
                              dtype=torch.float16)
            buf[0, 0] = tok[c.tok_bos]
            buf[0, 1] = tok[c.tok_user]
            slots = []
            for i in range(n_img):
                base = 2 + i * (per + 2)
                buf[0, base] = tok[c.tok_vision_start]
                blk = buf[0, base + 1: base + 1 + per].view(g, rows, D)
                blk[:, g] = tok[c.tok_vision_split]     # 每行末尾的 split
                slots.append(blk[:, :g])                # ← 逐帧只写这个视图
                buf[0, base + 1 + per] = tok[c.tok_vision_end]
            buf[0, 2 + n_img * (per + 2):] = lang
            self._prefix_key = (n_img, lang_tokens.clone())
            self._prefix_val = (buf, slots, per)
        return self._prefix_val

    # ── 三段 ──────────────────────────────────────────────────────────────
    def _prepare_vit(self, n_img: int) -> _Sub:
        c, w = self.w.cfg, self.w
        cache = self._vit_cache

        def bind(h):
            cold = self._native_cold.vit
            torch.ops.rpu.hyvit2_set_runtime_config(
                h,
                cold.fast_replay,
                cold.fast_replay_preload,
                cold.mask_once,
                cold.kvpad16,
                cold.q_inplace,
            )
            # W8A16 与 fp16 走**同一个** C++ set_weights，只是多传 6 条 scale 列表
            # （`rpu_hyvit2_set_weights` 自己就是拿空列表调它）。scale 非空 ⇒
            # `check_hyvit2_w8a16_scale_lists` 强制权重必须是 int8。
            args = (h, w.vit["qw"], w.vit["kw"], w.vit["vw"], w.vit["ow"],
                    w.vit["f1w"], w.vit["f2w"], w.vit["n1w"], w.vit["n1b"],
                    w.vit["n2w"], w.vit["n2b"], w.vit["qb"], w.vit["kb"], w.vit["vb"],
                    w.vit["ob"], w.vit["f1b"], w.vit["f2b"],
                    w.vit_proj1_w, w.vit_proj1_b,
                    c.vit_heads, c.vit_head_dim_padded, c.vit_hidden,
                    c.vit_inter_padded, c.proj_dim, c.vit_eps,
                    self._fused_merger, self._proj1_in_merger,
                    w.patch_embed_cores)
            if w.vit["qws"]:
                torch.ops.rpu.hyvit2_set_weights_w8a16(
                    *args, w.vit["qws"], w.vit["kws"], w.vit["vws"],
                    w.vit["ows"], w.vit["f1ws"], w.vit["f2ws"])
            else:
                torch.ops.rpu.hyvit2_set_weights(*args)
            torch.ops.rpu.hyvit2_set_chunk_envelope(h, int(cache.max_seq_len), 0)
            torch.ops.rpu.hyvit2_model_set_patch_emb(
                h, w.vit_patch_gemm_w, w.vit_pos_fused, 16, 16)

        vit_seq = c.vit_seq * (n_img if self._vit_packed else 1)
        return self._sub("vit", torch.ops.rpu.hyvit2_create,
                        torch.ops.rpu.hyvit2_destroy, bind,
                        rpu_backend.graph.GraphSignature(
                       op_id="hyvit2_vision_compute",
                       shapes=[vit_seq, c.vit_layers, c.vit_head_dim_padded],
                       dyn_dims=[c.vit_heads, 8, n_img if self._vit_packed else 1, 0],
                       dtypes=[torch.float16]))

    def _plan_vision_execution(self, sub: _Sub, image_count: int):
        logical_len = self.w.cfg.vit_seq * image_count
        span_len = logical_len // image_count
        return self._plan_execution(
            component=HY_VLA_VISION_COMPONENT, stage="vision",
            logical_len=logical_len, position=0, sub=sub,
            resolve_stage_domain=lambda length: (
                torch.ops.rpu.hyvit2_resolve_stage_domain(
                    sub.handle, int(length), image_count,
                    bool(self._vit_packed and self._pe_in_graph))),
            envelope={
                "logical_len": logical_len, "execution_len": logical_len,
                "image_count": image_count, "input_boundary": "KEEP_LOCAL",
                "qkv_boundary": "ALLOW_CROSS", "compute_boundary": "ALLOW_CROSS",
            },
            spans=[{"offset": image * span_len, "length": span_len,
                    "group": image, "boundary": "KEEP_LOCAL"}
                   for image in range(image_count)],
            kv_route="DDR_REQUIRED",
            route_reason="DDR_KV_CACHE_STORAGE_REQUIRED",
            plan_signature=(int(image_count), bool(self._vit_packed and self._pe_in_graph)),
        )

    def _run_vit(self, images: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        c, cache, n_img = self.w.cfg, self._vit_cache, len(images)
        sub = self._prepare_vit(n_img)

        def fwd(img):
            cache.reset_to_position(0)
            # 逐图路径在 ViT capture 外执行 patch embedding。
            hid = torch.ops.rpu.hyvit2_patch_embed(sub.handle, _rpu16(img))
            with sub.gc.capture(vision_sig):
                return torch.ops.rpu.hyvit2_forward(
                    sub.handle, hid,
                    [cache.k_caches[i] for i in range(c.vit_layers)],
                    [cache.v_caches[i] for i in range(c.vit_layers)],
                    vision_descriptor)

        with sub:
            image_count = n_img if self._vit_packed else 1
            logical_len = c.vit_seq * image_count
            vision_plan, vision_descriptor, _vision_chunk, vision_details = (
                self._plan_vision_execution(sub, image_count)
            )
            vision_sig = rpu_backend.graph.GraphSignature(
                op_id="hyvit2_vision_compute",
                shapes=[logical_len, c.vit_layers, c.vit_head_dim_padded],
                dyn_dims=[
                    c.vit_heads, 8, image_count,
                    self._component_generation(HY_VLA_VISION_COMPONENT),
                    *vision_plan.graph_key_words(),
                ],
                dtypes=[torch.float16],
            )
            # 首次 BUILD 的输出参与正常计算。先完成各图的 vision forward，
            # 再执行批量 merger，减少分散的 eager 调用。
            # 输出必须在 reset_temporary() 前物化；各图输出在拼接前保持存活。
            if self._vit_packed:
                # 将 N 个相机输入拼为 N*196 行；patch_embed_multi 设置图像分组数，
                # native attention 据此保持逐图隔离。
                # _pe_in_graph 控制 patch embedding 是否进入 capture，多核权重布局
                # 由同一份 cold 配置绑定。H2D 输入准备仍在 capture 外完成。
                cache.reset_to_position(0)
                imgs_rpu = [_rpu16(im) for im in images]
                kc = [cache.k_caches[i] for i in range(c.vit_layers)]
                vc = [cache.v_caches[i] for i in range(c.vit_layers)]
                if self._pe_in_graph:
                    with sub.gc.capture(vision_sig):
                        o = torch.ops.rpu.hyvit2_forward_multi(
                            sub.handle, imgs_rpu, kc, vc,
                            vision_descriptor)
                else:
                    hid = torch.ops.rpu.hyvit2_patch_embed_multi(
                        sub.handle, imgs_rpu)
                    with sub.gc.capture(vision_sig):
                        o = torch.ops.rpu.hyvit2_forward_packed(
                            sub.handle, hid, n_img, kc, vc,
                            vision_descriptor)
                stacked = o.reshape(n_img, -1, self._vit_out_dim())
            else:
                outs = [fwd(im) for im in images]
                stacked = torch.cat(
                    [o.reshape(1, -1, self._vit_out_dim()) for o in outs], 0)
            native_chunk = int(
                torch.ops.rpu.hyvit2_get_resolved_chunk_size(sub.handle)
            )
            self._record_execution_receipt(
                component=HY_VLA_VISION_COMPONENT,
                stage="vision",
                sub=sub,
                plan=vision_plan,
                descriptor=vision_descriptor,
                native_chunk=native_chunk,
                details=vision_details,
            )
            # 先读回 CPU，再按消费者需要加宽到 FP32。
            # 使用 FP16 prefix 模板时保留 FP16，避免不必要的往返类型转换。
            if self._merger_in_graph:
                # 融合 merger 需要自己的临时 SPM，先回收 ViT temporary。
                # 此时 o 已是物化的 DDR 张量；CPU 读回仍须在 capture 完成后执行。
                torch.ops.rpu.spm_alloc_reset_temporary()
                with sub.gc.capture(self._merger_sig(n_img)):
                    emb_rpu = self._merger_rest(stacked)
                embs = emb_rpu.cpu()
            else:
                embs = self._merger_rest(stacked).cpu()
            if not self._prefix_template:
                embs = embs.float()   # 老 `assemble_prefix` 要与 fp32 的 tok 表 cat
            torch.ops.rpu.spm_alloc_reset_temporary()
        return [embs[i] for i in range(len(images))]

    def _vit_out_dim(self) -> int:
        """ViT 塔的输出维。`_proj1_in_merger` 打开时 proj1 不在塔尾 ⇒ 是 hidden。"""
        c = self.w.cfg
        return (c.vit_hidden if (self._proj1_in_merger and self._fused_merger)
                else c.proj_dim)

    def _merger_sig(self, n_img: int) -> "rpu_backend.graph.GraphSignature":
        """merger 图的签名。与 ViT 共用同一个 `GraphCache`（两个 entry），
        因为它俩的生命周期完全一致（都挂在 `_Sub("vit")` 上）。"""
        c = self.w.cfg
        return rpu_backend.graph.GraphSignature(
            op_id="hyvla_merger_fused",
            # ⚠️ 输入维必须进签名：`_proj1_in_merger` 翻转会改变本图的输入形状，
            #    否则切开关时会命中上一档的缓存图。
            shapes=[n_img * c.vit_seq // 4, self._vit_out_dim(), 4],
            dyn_dims=[
                8, n_img,
                self._component_generation(HY_VLA_VISION_COMPONENT), 0,
            ],
            dtypes=[torch.float16])

    def _merger_rest(self, proj1_out: torch.Tensor) -> torch.Tensor:
        """merger 余部：DwPooler group-softmax → GELU → proj2。

        batch 维对应相机数。普通路径在 host 完成组轴重排、mean、softmax
        和加权求和，在 RPU 执行 GEMM，避免在组轴归约间反复切换设备。
        host 张量保持 FP16，归约使用 FP32 累加；因此末尾 sum 的舍入路径
        与逐次 FP16 加法不同。融合路径使用其专用的 member-major 布局。"""
        M, D = self.w.merger_rest, self.w.cfg.proj_dim
        B0 = proj1_out.shape[0] if proj1_out.dim() == 3 else 1
        if self._merger_in_graph:
            # 融合 op 在同一图内执行 Linear、GELU、pool 与 combine，减少 eager 派发。
            assert self._fused_merger, \
                "RPU_HY_VLA_MERGER_IN_GRAPH 需要 member-major 布局，必须同时开 " \
                "RPU_HY_VLA_FUSED_MERGER"
            # proj1 进图 ⇒ 输入是 [4,G,vit_hidden]，权重/偏置多传两条；
            # 关掉时传 None，走与落地前逐位相同的老链路。
            p1 = self._proj1_in_merger
            return torch.ops.rpu.hyvla_merger_fused(
                proj1_out.reshape(4, -1, self._vit_out_dim()),
                M["pooler.predictor.0.weight_a"], M["pooler.predictor.0.bias"],
                M["pooler.predictor.0.weight_b"],
                M["pooler.predictor.2.weight"], M["pooler.predictor.2.bias"],
                M["proj2.weight"], M["proj2.bias"],
                self.w.vit_proj1_w if p1 else None,
                self.w.vit_proj1_b if p1 else None).reshape(B0, -1, D)
        # ⚠️ 下面全是 host 回退路径。C++ 的 ViT 尾部只看 PROJ1_IN_MERGER +
        #    member-major，**不看 MERGER_IN_GRAPH** ⇒ 关掉图内 merger 却留着
        #    PROJ1_IN_MERGER，host 侧会拿到没 proj1 过的 [S,1152]，**静默算错**。
        assert not (self._proj1_in_merger and self._fused_merger), (
            "RPU_HY_VLA_PROJ1_IN_MERGER=1 只在图内 merger 下成立；"
            "关掉 RPU_HY_VLA_MERGER_IN_GRAPH 时请一并设 "
            "RPU_HY_VLA_PROJ1_IN_MERGER=0")
        lin = lambda t, k: F.linear(
            t.reshape(-1, t.shape[-1]).contiguous(), M[k + ".weight"], M[k + ".bias"]
        ).reshape(*t.shape[:-1], -1)
        if self._fused_merger:
            # C++ 尾部（`RPU_HY_VLA_FUSED_MERGER`）已经把行序换成 member-major
            # ⇒ 这里的 reshape 是纯 view，**没有 permute、没有 host 往返**。
            nxc = proj1_out.reshape(4, -1, D)
            # pooled = Σ_m nxc[m]，平铺成 [4,G,D]。**没除 4** —— 1/4 折进了 Wb
            # （fp16 里乘 0.25 是精确的 2 的幂缩放），少一次 kernel。
            pooled = torch.ops.rpu.hyvla_merger_pool(nxc)
            # predictor.0 按 K 拆成两半 ⇒ 不需要 `cat`（原来那次 4.8 MB 落地）。
            # 两半输出同形直接相加，**顺带避开首轴隐式广播**（那会静默产 NaN）。
            sc = (F.linear(nxc.reshape(-1, D), M["pooler.predictor.0.weight_a"],
                           M["pooler.predictor.0.bias"])
                  + F.linear(pooled.reshape(-1, D), M["pooler.predictor.0.weight_b"]))
            sc = lin(F.gelu(sc), "pooler.predictor.2").reshape(4, -1, D)
            out = torch.ops.rpu.hyvla_merger_combine(nxc, sc)
            return lin(F.gelu(out), "proj2").reshape(B0, -1, D)
        nxc = proj1_out.cpu().reshape(B0, 7, 2, 7, 2, D).permute(
            0, 1, 3, 2, 4, 5).reshape(B0, 7, 7, 4, D).contiguous()
        pooled = nxc.mean(-2, keepdim=True).expand(-1, -1, -1, 4, -1)
        sc = lin(_rpu16(torch.cat([nxc, pooled], -1)), "pooler.predictor.0")
        sc = lin(F.gelu(sc), "pooler.predictor.2")
        out = _rpu16((nxc * F.softmax(sc.cpu(), dim=-2)).sum(-2))
        return lin(F.gelu(out), "proj2").reshape(B0, -1, D)

    # ── 对外接口 ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def prepare_execution(self, *, images, prefill_pos, prefill_mask,
                          modality_mask, suffix_pos, denoise_mask):
        """Prepare the existing three child planners before the first forward.

        This binds weights/constants, not Graphs or KV contents. Cost collection
        and the normal forwards reuse the same child planning methods below.
        """
        session = self._execution_session
        with session._lock:
            session.require_cold()
            if self._persist != frozenset(("vit", "vlm", "expert")):
                raise ValueError("Hy-VLA cold planning requires persistent children")
            if len(images) != self.w.cfg.num_cameras:
                raise ValueError("Hy-VLA cold planning requires the configured camera count")
            if any(sub.gc is not None and sub.gc.snapshot()
                   for sub in self._psubs.values()):
                raise RuntimeError("Hy-VLA cold planning must precede all child forwards")
            try:
                # Retire any replaced constant set before retaining child refs.
                # Both regular forwards use these same cached constants.
                self._prefill_constants(prefill_pos, prefill_mask, modality_mask)
                self._denoise_constants(
                    self._total, prefill_pos, suffix_pos, denoise_mask)
                vit = self._prepare_vit(len(images))
                vlm, _ = self._prepare_vlm(prefill_pos, prefill_mask, modality_mask)
                expert, _ = self._prepare_expert(
                    self._total, prefill_pos, suffix_pos, denoise_mask)
                for sub in (vit, vlm, expert):
                    if sub.handle is None:
                        sub.open()
                return {
                    HY_VLA_VISION_COMPONENT: self._plan_vision_execution(
                        vit, len(images) if self._vit_packed else 1),
                    HY_VLA_LANGUAGE_COMPONENT: self._plan_language_execution(vlm),
                    HY_VLA_ACTION_COMPONENT: self._plan_action_execution(expert),
                }
            except BaseException:
                session.poison()
                raise

    @execution_serialized
    @torch.no_grad()
    def get_action(self, images: Sequence[torch.Tensor],
                   lang_tokens: torch.Tensor,
                   prefill_mask: torch.Tensor, prefill_pos: torch.Tensor,
                   modality_mask: torch.Tensor,
                   denoise_mask: torch.Tensor, suffix_pos: torch.Tensor,
                   noise: torch.Tensor,
                   state: Optional[torch.Tensor] = None,
                   state_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """一次完整推理 → `action [1, 50, 32]`。

        Args:
            images:        3 × `[1,3,224,224]` fp32。
            lang_tokens:   `[64]` int，prompt 的 token id（含 padding）。
            prefill_mask:  `[240,240]` bool，vendor 的 prefill attention mask。
            prefill_pos:   `[240]`，prefix 的 position_ids。
            modality_mask: `[240]`，True = vision 行（走 `_v` 塔）。
            denoise_mask:  `[51,291]` bool，denoise 的 attention mask。
            suffix_pos:    `[51]`，suffix 的 position_ids（从 prefix **有效**长度起算）。
            noise:         `[1,50,32]` fp32，Euler 的初始 x_t。
            state / state_emb: 二选一。`state` 走 `state_proj`；`state_emb`
                直接提供已投影的 `[1,1,1024]` state token。
        """
        if self._closed:
            raise RuntimeError("HyVlaRunner 已 close，不能再次推理。")
        w = self.w
        assert (state is None) != (state_emb is None), "state / state_emb 二选一"

        # ── 1) ViT ────────────────────────────────────────────────────────
        img_embs = self._run_vit(images)

        # ── 2) 组装 prefix（host）──────────────────────────────────────────
        prefix = self.assemble_prefix(img_embs, lang_tokens)

        # ── 3) VLM prefill ────────────────────────────────────────────────
        cache, total = self._cache, self._total
        self._run_vlm_prefill(cache, prefix, prefill_pos, prefill_mask,
                              modality_mask)

        # ── 4) expert denoise ×10 ─────────────────────────────────────────
        st = (state_emb if state_emb is not None else
              F.linear(state, w.host["state_proj.weight"],
                       w.host["state_proj.bias"])[:, None, :])
        return self._run_denoise(cache, total, prefill_pos, suffix_pos,
                                 denoise_mask, noise, st)

    # ── 分段实现 ──────────────────────────────────────────────────────────
    # 三段各自成方法，是为了**可打点**：perf harness 用实例级方法遮蔽来累计
    # 每段 wall（同 lingbot 的 `_vit_towers` / `_vlm_fill` / `_exp_run`）。
    # 内联闭包遮蔽不了。

    def _prefill_constants(self, prefill_pos, prefill_mask, modality_mask):
        c, S = self.w.cfg, self.prefix_len

        def build_consts():
            vcos, vsin = build_rope_tables(prefill_pos[:S], c)
            vm = prefill_mask[:S, :S].clone()
            valid = prefill_mask.any(-1)[:S]
            for i in range(S):
                if not valid[i]:
                    vm[i, i] = True  # padding 行开对角口：CPU softmax 对全 -inf 行
                                     # 退化成均匀分布，RPU kernel 未必；因果性保证
                                     # 这些行的值流不回有效行。
            return (vcos, vsin,
                    _rpu16(torch.where(vm, 0.0, MASK_NEG).reshape(1, 1, S, S)),
                    _rpu16(1.0 - modality_mask[:S].float()))

        return self._cached(
            "prefill", (prefill_pos, prefill_mask, modality_mask), build_consts)

    def _prepare_vlm(self, prefill_pos, prefill_mask, modality_mask):
        """Bind the original MoT weights and prompt constants without a forward."""
        c, w, S = self.w.cfg, self.w, self.prefix_len
        vcos, vsin, vmask, vmod = self._prefill_constants(
            prefill_pos, prefill_mask, modality_mask)

        # W8A16 时改调 `*_w8a16` 变体（多传 7 条 scale），C++ 侧落到同一个
        # set_weights_hyvla / set_moe_weights。**两条孪生各自独立**：
        # text 与 `_v` 是两组不同权重，各带各的 scale 列表。
        _P = ("q_w", "k_w", "v_w", "o_w", "q_norm", "k_norm",
              "input_norm", "post_norm", "gate_w", "up_w", "down_w")
        _S = ("q_ws", "k_ws", "v_ws", "o_ws", "gate_ws", "up_ws", "down_ws")

        def bind_vlm(h):
            cold = self._native_cold.vlm
            torch.ops.rpu.hyvla_vlm_set_runtime_config(
                h,
                cold.fast_replay,
                cold.fast_replay_preload,
                cold.mask_once,
                cold.partial_rope,
                cold.mot_norm_nomerge,
                cold.silu_mul,
                cold.rmsnorm_capability,
            )
            targs = (h, *[w.vlm_text[k] for k in _P],
                     vcos, vsin, w.vlm_final_norm,
                     c.num_q_heads, w.kv_heads_rpu, c.head_dim, c.vlm_hidden,
                     c.vlm_inter, c.eps, True, [], [], [])
            margs = (h, *[w.vlm_vision[k] for k in _P],
                     [], [], [], w.vlm_final_norm, vmod, S)
            # ⚠️ R42：两条孪生**各判各的**。原来一个 `if` 同时决定两边 ——
            # 那把"两条孪生位宽必然相同"写成了硬耦合，而 C++ 侧没有这个约束：
            # `set_weights_hyvla` 与 `set_moe_weights` 是两个独立入口，各自按
            # "有没有给 scale 列表"（`moe_q8`）独立校验 dtype，全程按张量
            # `scalar_type()` 派发 kernel，没有任何模型级的 w8a16 标志。
            if w.vlm_text["q_ws"]:
                torch.ops.rpu.hyvla_vlm_set_weights_w8a16(
                    *targs, *[w.vlm_text[k] for k in _S])
            else:
                torch.ops.rpu.hyvla_vlm_set_weights(*targs)
            if w.vlm_vision["q_ws"]:
                torch.ops.rpu.hyvla_vlm_set_moe_weights_w8a16(
                    *margs, *[w.vlm_vision[k] for k in _S])
            else:
                torch.ops.rpu.hyvla_vlm_set_moe_weights(*margs)

        vlm = self._sub("vlm", torch.ops.rpu.hyvla_vlm_create,
                        torch.ops.rpu.hyvla_vlm_destroy, bind_vlm,
                        rpu_backend.graph.GraphSignature(
                       op_id="hyvla_vlm_prefill", shapes=[S, c.vlm_hidden],
                       dyn_dims=[c.layers, 0, w.attn_tp, 0], dtypes=[torch.float16]))
        return vlm, (vcos, vsin, vmask)

    def _plan_language_execution(self, sub: _Sub):
        S = self.prefix_len
        return self._plan_execution(
            component=HY_VLA_LANGUAGE_COMPONENT, stage="prefill",
            logical_len=S, position=0, sub=sub,
            resolve_stage_domain=lambda length: (
                torch.ops.rpu.hyvla_vlm_resolve_stage_domain(
                    sub.handle, int(length), 0, int(length))),
            envelope={
                "logical_len": S, "execution_len": S,
                "certified_prefix_min": 16, "certified_prefix_max": 240,
                "single_chunk_required": True,
            },
            spans=[{"offset": 0, "length": S, "group": 0,
                    "boundary": "KEEP_LOCAL"}],
            kv_route="DDR_REQUIRED",
            route_reason="PREFIX_HISTORY_DDR_REQUIRED_SHARED_RPUCACHE_ABI",
            plan_signature=(),
        )

    def _run_vlm_prefill(self, cache: RPUCache, prefix: torch.Tensor,
                         prefill_pos: torch.Tensor, prefill_mask: torch.Tensor,
                         modality_mask: torch.Tensor) -> None:
        """MoT 双权重 prefill —— 把 prefix KV 写进 `cache` 的 `[0,S)`。"""
        c, S = self.w.cfg, self.prefix_len
        vlm, (vcos, vsin, vmask) = self._prepare_vlm(
            prefill_pos, prefill_mask, modality_mask)
        x_prefix = _rpu16(prefix[:, :S])

        def prefill():
            cache.reset_to_position(0)
            with vlm.gc.capture(vlm_sig):
                torch.ops.rpu.hyvla_vlm_step_forward(
                    vlm.handle, x_prefix, cache.k_caches, cache.v_caches,
                    vcos, vsin, vmask, 0, vlm_descriptor)
            native_chunk = int(
                torch.ops.rpu.hyvla_vlm_get_resolved_chunk_size(vlm.handle)
            )
            self._record_execution_receipt(
                component=HY_VLA_LANGUAGE_COMPONENT,
                stage="prefill",
                sub=vlm,
                plan=vlm_plan,
                descriptor=vlm_descriptor,
                native_chunk=native_chunk,
                details=vlm_details,
            )
            torch.ops.rpu.spm_alloc_reset_temporary()

        with vlm:
            vlm_plan, vlm_descriptor, _vlm_chunk, vlm_details = (
                self._plan_language_execution(vlm)
            )
            vlm_sig = rpu_backend.graph.GraphSignature(
                op_id="hyvla_vlm_prefill",
                shapes=[S, c.vlm_hidden],
                dyn_dims=[
                    c.layers, 0, self.w.attn_tp,
                    self._component_generation(HY_VLA_LANGUAGE_COMPONENT),
                    *vlm_plan.graph_key_words(),
                ],
                dtypes=[torch.float16],
            )
            prefill()          # 唯一一次 forward（= BUILD），cache 里留下的就是 prefix KV

    def _denoise_constants(self, total, prefill_pos, suffix_pos, denoise_mask):
        c, S, L = self.w.cfg, self.prefix_len, self.w.cfg.suffix_len

        def build_consts():
            pos_tbl = torch.zeros(total, dtype=torch.float64)
            pos_tbl[:S] = prefill_pos[:S].double()
            pos_tbl[S:S + L] = suffix_pos.double()
            ecos, esin = build_rope_tables(pos_tbl, c)
            # denoise mask 的 prefix 块必须跟着 S 走，总长度为 S + 51。
            dm = torch.cat([denoise_mask[:, :S], denoise_mask[:, -L:]], dim=1)
            return (ecos, esin,
                    _rpu16(torch.where(dm, 0.0, MASK_NEG).reshape(1, 1, L, S + L)))

        return self._cached(
            "denoise", (prefill_pos, suffix_pos, denoise_mask), build_consts)

    def _prepare_expert(self, total, prefill_pos, suffix_pos, denoise_mask):
        """Bind the original suffix/unroll weights and constants without executing."""
        c, w, S, L = self.w.cfg, self.w, self.prefix_len, self.w.cfg.suffix_len
        ecos, esin, emask = self._denoise_constants(
            total, prefill_pos, suffix_pos, denoise_mask)

        def bind_exp(h):
            cold = self._native_cold.expert
            torch.ops.rpu.hyvla_expert_set_runtime_config(
                h,
                cold.fast_replay,
                cold.fast_replay_preload,
                cold.mask_once,
                cold.silu_mul,
                cold.kvpad16,
                cold.partial_rope,
                cold.rmsnorm_pad16,
            )
            eargs = (h, *[w.expert[k] for k in ("q_w", "k_w", "v_w", "o_w",
                                                "q_norm", "k_norm", "input_norm",
                                                "post_norm", "gate_w", "up_w",
                                                "down_w")],
                     ecos, esin, w.expert_final_norm,
                     c.num_q_heads, w.kv_heads_rpu, c.head_dim, c.expert_hidden,
                     c.expert_inter, c.eps, L, w.action_mlp_cores)
            if w.expert["q_ws"]:
                # unroll 的 encoder/out_proj 权重保持浮点，不随 expert 量化选择改变。
                torch.ops.rpu.hyvla_expert_set_weights_w8a16(
                    *eargs, *[w.expert[k] for k in ("q_ws", "k_ws", "v_ws",
                                                    "o_ws", "gate_ws", "up_ws",
                                                    "down_ws")])
            else:
                torch.ops.rpu.hyvla_expert_set_weights(*eargs)
            if self._unroll:
                torch.ops.rpu.hyvla_expert_set_action_weights(
                    h, *self._uw, c.action_dim, c.num_steps)

        exp = self._sub("expert", torch.ops.rpu.hyvla_expert_create,
                        torch.ops.rpu.hyvla_expert_destroy, bind_exp,
                        rpu_backend.graph.GraphSignature(
                       op_id="hyvla_expert_denoise_unroll" if self._unroll
                       else "hyvla_expert_denoise",
                       shapes=[L, c.expert_hidden],
                       dyn_dims=[c.layers, S, w.attn_tp,
                                 c.num_steps if self._unroll else 0],
                       dtypes=[torch.float16]),
                        variant=self._unroll)
        return exp, (ecos, esin, emask)

    def _plan_action_execution(self, sub: _Sub):
        S, L = self.prefix_len, self.w.cfg.suffix_len
        return self._plan_execution(
            component=HY_VLA_ACTION_COMPONENT, stage="action",
            logical_len=L, position=S, sub=sub,
            resolve_stage_domain=lambda length: (
                torch.ops.rpu.hyvla_expert_resolve_stage_domain(
                    sub.handle, int(length), S, S + int(length))),
            envelope={
                "logical_len": L, "execution_len": L,
                "physical_chunk": ((L + 15) // 16) * 16,
                "prefix_history_rows": S, "single_chunk_required": True,
            },
            spans=[{"offset": 0, "length": L, "group": 0,
                    "boundary": "KEEP_LOCAL"}],
            kv_route="DDR_REQUIRED",
            route_reason="PREFIX_HISTORY_DDR_REQUIRED_SHARED_RPUCACHE_ABI",
            plan_signature=(),
        )

    def _run_denoise(self, cache: RPUCache, total: int,
                     prefill_pos: torch.Tensor, suffix_pos: torch.Tensor,
                     denoise_mask: torch.Tensor, noise: torch.Tensor,
                     st: torch.Tensor) -> torch.Tensor:
        """10 步 Euler，在 `cache` 的 `[S,S+51)` 上追加 suffix KV → `action`。"""
        c, S, L = self.w.cfg, self.prefix_len, self.w.cfg.suffix_len
        exp, (ecos, esin, emask) = self._prepare_expert(
            total, prefill_pos, suffix_pos, denoise_mask)

        dt = -1.0 / c.num_steps

        if self._unroll:
            # 一次 forward 跑完 10 步。cache 只在进图前 reset 一次 —— 图内每个 body
            # 都以 position=S 插 KV，覆写同一段 51 行，与逐步 reset 等价。
            self._ux0[:, 1:, :].copy_(_rpu16(noise))
            with exp:
                action_plan, action_descriptor, _action_chunk, action_details = (
                    self._plan_action_execution(exp)
                )
                action_sig = rpu_backend.graph.GraphSignature(
                    op_id="hyvla_expert_denoise_unroll",
                    shapes=[L, c.expert_hidden],
                    dyn_dims=[
                        c.layers, S, self.w.attn_tp, c.num_steps,
                        self._component_generation(HY_VLA_ACTION_COMPONENT),
                        *action_plan.graph_key_words(),
                    ],
                    dtypes=[torch.float16],
                )
                cache.reset_to_position(S)
                with exp.gc.capture(action_sig):
                    torch.ops.rpu.hyvla_expert_unroll_forward(
                        exp.handle, self._ux0, cache.k_caches, cache.v_caches,
                        _rpu16(st.reshape(-1)), ecos, esin, emask,
                        self._utraj, dt, S, c.num_steps, action_descriptor)
                native_chunk = int(
                    torch.ops.rpu.hyvla_expert_get_resolved_chunk_size(
                        exp.handle
                    )
                )
                self._record_execution_receipt(
                    component=HY_VLA_ACTION_COMPONENT,
                    stage="action",
                    sub=exp,
                    plan=action_plan,
                    descriptor=action_descriptor,
                    native_chunk=native_chunk,
                    details=action_details,
                )
                out = self._utraj[-1:].cpu().float()   # 读回早于 reset_temporary
                torch.ops.rpu.spm_alloc_reset_temporary()
            return out[:, 1:, :]

        def one_step(xt, k):
            emb = _rpu16(torch.cat([st, self._embed(xt, k)], dim=1))
            # 每步 reset 到 prefix 尾 —— 等价于 vendor 的 copy.deepcopy 冻结
            # prefix KV：suffix 的 51 行每步覆写同一段，用完即弃。
            cache.reset_to_position(S)
            with exp.gc.capture(action_sig):
                o = torch.ops.rpu.hyvla_expert_step_forward(
                    exp.handle, emb, cache.k_caches, cache.v_caches,
                    ecos, esin, emask, S, action_descriptor)
            native_chunk = int(
                torch.ops.rpu.hyvla_expert_get_resolved_chunk_size(exp.handle)
            )
            self._record_execution_receipt(
                component=HY_VLA_ACTION_COMPONENT,
                stage="action",
                sub=exp,
                plan=action_plan,
                descriptor=action_descriptor,
                native_chunk=native_chunk,
                details=action_details,
            )
            so = o.float().cpu()              # capture 退出后立刻读回
            torch.ops.rpu.spm_alloc_reset_temporary()
            up = torch.addmm(self._out_proj_b, so[0, -c.n_action:],
                             self._out_proj_wt)          # 见 `__post_init__`
            return xt + dt * up.reshape(1, -1, c.action_dim)

        # Euler 循环中的 host GEMM 较小，局部限制线程以避免反复并行派发。
        # 该线程设置仅覆盖此 denoise 循环。
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
          with exp:
            action_plan, action_descriptor, _action_chunk, action_details = (
                self._plan_action_execution(exp)
            )
            action_sig = rpu_backend.graph.GraphSignature(
                op_id="hyvla_expert_denoise",
                shapes=[L, c.expert_hidden],
                dyn_dims=[
                    c.layers, S, self.w.attn_tp,
                    self._component_generation(HY_VLA_ACTION_COMPONENT),
                    *action_plan.graph_key_words(),
                ],
                dtypes=[torch.float16],
            )
            # k=0 的 BUILD 输出直接用于当前 Euler 步。
            xt = noise.clone()
            for k in range(c.num_steps):
                xt = one_step(xt, k)
            return xt
        finally:
            torch.set_num_threads(prev_threads)


def build_hy_vla(
    ckpt_path: "str | Path",
    pos_embedding: torch.Tensor | None = None,
    prefix_len: int = 240,
    cfg: HyVlaConfig = HyVlaConfig(),
    *,
    rpu_execution: Mapping[str, Any] | None = None,
    _env_snapshot: dict[str, str | None] | None = None,
    _owner: object | None = None,
    _cold_plan: _HyVlaColdPlan | None = None,
) -> HyVlaRunner:
    """装载 Hy-VLA：权重一次性全部上 RPU，返回可反复调用的 runner。

    Args:
        pos_embedding: ViT 位置编码。**默认 `None` = 从 ckpt 自己的 `pos_embed`
            重采样**（`weights.sample_vit_pos_embedding`）。传张量则原样使用。
        prefix_len: `S = ceil16(有效 prefix 行数)`。合法区间 `{16,32,…,240}`。
            上限 240 同时是需求上限（`tokenizer_max_length=64` ⇒ 最坏
            `ceil16(176+64)=240`）和能力上限（S=256 差 375 KB 撞 SPM，
            S=272 撞 SDPA 的 `grid×gqa ≤ 8`）。默认取 240 —— 它同时是交付契约
            里的最坏情形，对任何合法 prompt 都够用；短 prompt 想省时间才需要
            显式传 `ceil16(有效行数)`。

        ⚠️ 这是低层入口，但不是生命周期逃生口。一旦开始原生权重物化，
        该 Python 进程同样不能再构建另一个 RPU runner/model；需要重启进程。
    """
    cold_plan = (
        prepare_hy_vla_cold_plan()
        if _cold_plan is None else _cold_plan
    )
    if not isinstance(cold_plan, _HyVlaColdPlan):
        raise TypeError("_cold_plan must be a _HyVlaColdPlan")
    execution_root, execution_children = resolve_hy_vla_execution(
        rpu_execution, entry_point="build_hy_vla"
    )
    if prefix_len % 16 or not 16 <= prefix_len <= 240:
        raise ValueError(f"prefix_len 必须是 16 的倍数且在 [16,240]，得到 {prefix_len}")
    packed_vision = cold_plan.runner.vit_packed
    _validate_hy_vla_execution_geometry(
        execution_children,
        cfg=cfg,
        prefix_len=prefix_len,
        packed_vision=packed_vision,
        entry_point="build_hy_vla",
    )
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"Hy-VLA 权重不存在: {ckpt_path}")
    # Validate the complete safetensors schema before claiming or poisoning the
    # process-wide RPU owner.  The loader repeats this check on its live handle.
    preflight_hy_vla_checkpoint(ckpt_path, cfg)
    from rpu_backend.api.causal_lm import (
        _claim_live_instance,
        _poison_live_instance,
        _release_live_instance,
    )

    direct_owner = _owner is None
    lifecycle_owner = _DirectHyVlaOwner() if direct_owner else _owner
    _claim_live_instance(lifecycle_owner)
    try:
        torch.rpu.set_caching_allocator(cold_plan.caching_allocator)
    except BaseException:
        _release_live_instance(lifecycle_owner)
        raise
    _poison_live_instance(
        lifecycle_owner,
        "Hy-VLA native materialization freezes process-global runtime switches",
    )
    # 将 encoder、action_out_proj 与 Euler 更新收进 expert 的图内 unroll，
    # x_t 在 denoise 期间留在 SPM。该路径把部分 host FP32 计算改为设备
    # FP16，并涉及矩阵重结合；RPU_HY_VLA_DENOISE_UNROLL=0 使用逐步路径。
    _setdefault_env("RPU_HY_VLA_DENOISE_UNROLL", cold_plan, _env_snapshot)
    # 持久模式跨帧保留三个子系统的 handle 与 GraphCache。
    # 布局或 prompt 相关缓存键改变时，runner 负责失效并重建全部依赖资源；
    # 不能将失效的 persistent SPM 槽继续用于 REPLAY。
    _setdefault_env("RPU_HY_VLA_PERSIST_HANDLES", cold_plan, _env_snapshot)
    # 打包三路相机的 ViT 输入，共享一次权重遍历。
    # 逐图 attention 范围仍由 native 图像分组布局隔离。
    _setdefault_env("RPU_HY_VLA_VIT_PACKED", cold_plan, _env_snapshot)
    # Hybrid KV-insert 将对齐的主体交给 V16，余下尾部交给 V2。
    # 两段写入位置不重叠；每条路径仍受核数及 KV head 布局条件约束。
    _setdefault_env("RPU_KVINSERT_HYBRID_V16", cold_plan, _env_snapshot)
    # 允许满足完整 KV-head 分片条件的 attention TP 使用 V16。
    # 不能将按列拆分的不完整 KV head 布局当作完整 head 输入。
    _setdefault_env("RPU_KVINSERT_V16_ANY_TP", cold_plan, _env_snapshot)
    # member-major 布局让 DwPooler 的组轴运算交给专用 pool/combine op。
    # predictor 按 K 拆分以避免拼接。此路径改变 GEMM 累加及 pooled/softmax
    # 精度；Python 的布局选择随 ViT set_weights 绑定到 native。
    _setdefault_env("RPU_HY_VLA_FUSED_MERGER", cold_plan, _env_snapshot)
    # 融合 merger 将 Linear、GELU、pool 与 combine 放入同一图内 op。
    # 它依赖 RPU_HY_VLA_FUSED_MERGER 的 member-major 布局，并保持相应
    # GEMM 的 ACC32 累加策略。
    _setdefault_env("RPU_HY_VLA_MERGER_IN_GRAPH", cold_plan, _env_snapshot)
    # patch embedding 的多核布局和图内执行分别由两个 cold 选项控制。
    # weights.py 使用同一份解析结果 swizzle，并将核数绑定到 native handle。
    _setdefault_env("RPU_HY_VLA_PATCH_EMBED_MC", cold_plan, _env_snapshot)
    _setdefault_env("RPU_HY_VLA_PATCH_EMBED_IN_GRAPH", cold_plan, _env_snapshot)
    # Fast replay 跳过已记录 layer body 的 host 遍历，设备图仍正常提交。
    # 要求该 body 没有必须逐次执行的 host 副作用：hidden、mask 和 unroll
    # 输入/输出分别通过 mutable DMA 或稳定槽更新。
    _setdefault_env("RPU_HY_VLA_FAST_REPLAY", cold_plan, _env_snapshot)
    # Preload replay skip 省去 host 回调重走，已记录的 preload DMA
    # 仍随 segment 执行。仅在权重保持有效且 layer-loop skip 生效时启用。
    _setdefault_env("RPU_HY_VLA_FAST_REPLAY_PRELOAD", cold_plan, _env_snapshot)
    # 将每个 KV head 复制到两个物理槽，使 attention 使用 8 核。
    # VLM 与 expert 共享 KV cache，必须共同使用此布局；复制会增加权重和
    # cache 存储，并改变跨核归约顺序。
    _setdefault_env("RPU_HY_VLA_ATTN_TP8", cold_plan, _env_snapshot)
    # 层间恒定的显式 mask 可在首层加载后保持于 SPM。
    # 其生命周期必须覆盖全部消费者，避免后续 Q/K/V 临时槽覆盖 mask。
    _setdefault_env("RPU_HY_VLA_MASK_ONCE", cold_plan, _env_snapshot)
    # 单 chunk ViT 在 KV_FIRST 两相间保留 Q，避免 DDR 中转。
    # q_kv 使用覆盖两相的生命周期，q_comp 别名指向同一槽；
    # 多 query-chunk 路径不能直接复用该约定。
    _setdefault_env("RPU_HY_VLA_Q_INPLACE", cold_plan, _env_snapshot)
    # KV-insert 可使用补齐到 16 行的 SPM 容量。
    # 额外写入的物理尾行必须位于 cache 容量内，attention 的 kv_seq_len
    # 仍只包含真实 token，不能把 padding 当作有效上下文。
    _setdefault_env("RPU_HY_VLA_KVPAD16", cold_plan, _env_snapshot)
    # SwiGLU 融合路径先计算独立的 gate/up GEMM，再合并 SiLU 与乘法。
    # 该模型按 tower 选择能力；launcher 必须遵守每块元素数的范围约束，
    # 超过范围时分块，不能仅凭总张量大小启用单块执行。
    _setdefault_env("RPU_HY_VLA_SILU_MUL", cold_plan, _env_snapshot)
    # RMSNorm vector 能力由 cold 配置翻译为 per-handle capability。
    # AUTO 根据 M 选择可用的向量宽度，仍要求 M % V == 0；不满足条件
    # 时使用对应回退。显式 vector 路径缺少所需 kernel 时应拒绝。
    _setdefault_env("RPU_RMSNORM_VWARP", cold_plan, _env_snapshot)
    # 仅在 profiler 未启用时省去 capture 的 record_function 开销；
    # 启用 profiler 时保留正常 trace 区间。
    _setdefault_env("RPU_SKIP_IDLE_RECORD_FUNCTION", cold_plan, _env_snapshot)
    # ACC32 GEMM 根据实际形状选择可用 tiling。
    # partial_mrope 使用逐 token 的 [max_seq, head_dim/2] 表，
    # cos_sin_start 表示表内行偏移，不需要另建 position gather 缓冲。
    # RoPE 能力按 expert/vlm 的 cold 选项绑定到各自 handle。
    _setdefault_env("RPU_HY_VLA_PARTIAL_ROPE", cold_plan, _env_snapshot)
    # R40-①：允许通用 Graph executor 在 replay 没有触碰 kernel 参数、且当前
    # segment 不含 mutable DMA 时省略 `sync_mutable_params()`。在 pinned runtime
    # 上，mutable-DMA segment 省略该同步的本机 ABBA 回放已证不安全，底层原因
    # 尚未完成定位；因此 Graph 保守地根据 segment 结构保留同步，不依赖
    # 模型名、层号或 shape 特判。
    _setdefault_env("RPU_FASTREPLAY_SKIP_SYNC", cold_plan, _env_snapshot)
    # W8A16 不在这里自动启用，部署方需显式选择量化的 tower。
    # 可按 vit/vlm/expert 组合选择，量化会改变数值路径。
    # DDR caching allocator 由 cold plan 选择；关闭它可使用独立分配路径。
    runner = HyVlaRunner(
        w=load_hy_vla_weights(
            ckpt_path,
            pos_embedding,
            cfg,
            _cold_plan=cold_plan.weights,
        ),
        prefix_len=prefix_len,
        _native_cold=cold_plan.native,
        rpu_execution=execution_root,
        _unroll=cold_plan.runner.unroll,
        _vit_packed=cold_plan.runner.vit_packed,
        _fused_merger=cold_plan.runner.fused_merger,
        _merger_in_graph=cold_plan.runner.merger_in_graph,
        _pe_in_graph=cold_plan.runner.patch_embed_in_graph,
        _proj1_in_merger=cold_plan.runner.proj1_in_merger,
        _prefix_template=cold_plan.runner.prefix_template,
        _persist=cold_plan.runner.persist,
    )
    if direct_owner:
        # Keep the anonymous low-level claim alive with its runner. Never add
        # this back-reference for the facade: weakref.finalize(policy, ..., runner)
        # strongly owns runner, so runner -> policy would make policy immortal.
        runner._lifecycle_owner = lifecycle_owner
    return runner
