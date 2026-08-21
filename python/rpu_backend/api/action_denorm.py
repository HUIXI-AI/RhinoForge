"""Config-driven action de-normalization for LingBot-VLA-V2 on the RPU.

WHY THIS EXISTS
    The LingBot-VLA-V2 model emits actions in a *normalized* space. Deployment
    must map them back to physical units before handing them to the robot. This
    module implements the required split, unnormalization, and optional delta
    ``action += state`` transformation.

REQUIREMENTS
    * NOTHING about a specific robot's stats is hardcoded. The norm JSON path, the joint
      order/dims and each joint's norm_type are all supplied by config.
    * Uses the checkpoint configuration's normalization formulas.

The formulas preserve the reference ``+1e-6`` epsilons and clamp bounds so
results compose with other LingBot-VLA-V2 runtimes.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import torch

# Norm types whose unnormalize we have replicated verbatim from the official Normalizer.
_SUPPORTED_NORM_TYPES = frozenset({
    "meanstd", "std", "minmax", "minmax_woclip",
    "bounds_98", "bounds_98_woclip", "bounds_99", "bounds_99_woclip",
    "identity",
    # NOTE: "sincos" deliberately unsupported here -- it is state-only in the official
    # code (raises for non-state keys), so it can never apply to an action group.
})

_REQUIRED_STATS = {
    "meanstd": ("mean", "std"),
    "std": ("std",),
    "minmax": ("min", "max"),
    "minmax_woclip": ("min", "max"),
    "bounds_98": ("q02", "q98"),
    "bounds_98_woclip": ("q02", "q98"),
    "bounds_99": ("q01", "q99"),
    "bounds_99_woclip": ("q01", "q99"),
    "identity": (),
}


@dataclass(frozen=True)
class ActionGroup:
    """One contiguous action sub-vector, e.g. ('action.arm.position', 14, 'meanstd')."""
    key: str
    dim: int
    norm_type: str
    subtract_state: bool = False        # True => relative/delta action: physical = unnorm + state
    state_key: str | None = None        # the matching observation.state.* key, for delta actions


@dataclass
class ActionDenormalizer:
    """Maps a normalized action grid [H, D_real] back to physical units, per group.

    Build it with one of the classmethods; never hardcode a specific robot here.
    """
    groups: Sequence[ActionGroup]
    norm_stats: Mapping[str, Mapping[str, np.ndarray]]
    _offsets: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        if not self.groups:
            raise ValueError("action de-normalizer requires at least one action group")
        off, acc = [], 0
        for g in self.groups:
            if (not isinstance(g.dim, (int, np.integer)) or
                    isinstance(g.dim, (bool, np.bool_)) or int(g.dim) <= 0):
                raise ValueError(
                    f"action group {g.key!r} has invalid dim {g.dim!r}; "
                    "dim must be a positive integer"
                )
            dim = int(g.dim)
            if g.norm_type not in _SUPPORTED_NORM_TYPES:
                raise ValueError(
                    f"action group {g.key!r} has norm_type {g.norm_type!r}, which this "
                    f"denormalizer has not replicated. Supported: {sorted(_SUPPORTED_NORM_TYPES)}."
                )
            if g.norm_type != "identity" and g.key not in self.norm_stats:
                raise KeyError(
                    f"norm_stats is missing key {g.key!r} (have: {list(self.norm_stats)[:8]}...). "
                    f"Wrong robot_norm JSON for this checkpoint?"
                )
            if g.norm_type != "identity":
                stats = self.norm_stats[g.key]
                if not isinstance(stats, Mapping):
                    raise ValueError(f"norm_stats[{g.key!r}] must be a mapping")
                for name in _REQUIRED_STATS[g.norm_type]:
                    if name not in stats:
                        raise KeyError(
                            f"norm_stats[{g.key!r}] is missing required {name!r} "
                            f"for norm_type {g.norm_type!r}"
                        )
                    value = np.asarray(stats[name], dtype=np.float64)
                    if value.ndim not in (1, 2):
                        raise ValueError(
                            f"{g.key}.{name} must be [dim] or [horizon, dim], "
                            f"got shape {value.shape}"
                        )
                    if value.shape[-1] != dim:
                        raise ValueError(
                            f"{g.key}.{name} action dim {value.shape[-1]} != "
                            f"configured group dim {dim}"
                        )
                    if value.ndim == 2 and value.shape[0] == 0:
                        raise ValueError(f"{g.key}.{name} has an empty horizon")
                    if not np.isfinite(value).all():
                        raise ValueError(f"{g.key}.{name} contains non-finite values")
            off.append((acc, acc + dim))
            acc += dim
        self._offsets = off
        self.real_dim = acc

    # ── stat helper (mirrors Normalizer._get_stat for the 1-D-stat case) ──────────
    def _stat(self, key: str, name: str, horizon: int) -> np.ndarray:
        s = np.asarray(self.norm_stats[key][name], dtype=np.float64)
        if s.ndim == 2:                       # [chunk, dim] stats -> align horizon
            if s.shape[0] < horizon:
                raise ValueError(f"{key}.{name} horizon {s.shape[0]} < action horizon {horizon}")
            s = s[:horizon]
        return s

    def _unnormalize_group(self, v: np.ndarray, g: ActionGroup) -> np.ndarray:
        """v: [H, dim] normalized. Returns [H, dim] physical. Verbatim official formulas."""
        H = v.shape[0]
        nt = g.norm_type
        if nt == "identity":
            return v
        if nt == "meanstd":
            mean = self._stat(g.key, "mean", H)
            std = self._stat(g.key, "std", H)
            return v * (std + 1e-6) + mean
        if nt == "std":
            std = self._stat(g.key, "std", H)
            return v * (std + 1e-6)
        if nt in ("minmax", "minmax_woclip"):
            lo = self._stat(g.key, "min", H)
            hi = self._stat(g.key, "max", H)
            return (v + 1.0) / 2.0 * (hi - lo + 1e-6) + lo
        if nt in ("bounds_98", "bounds_98_woclip"):
            lo = self._stat(g.key, "q02", H)
            hi = self._stat(g.key, "q98", H)
            return (v + 1.0) / 2.0 * (hi - lo + 1e-6) + lo
        if nt in ("bounds_99", "bounds_99_woclip"):
            lo = self._stat(g.key, "q01", H)
            hi = self._stat(g.key, "q99", H)
            return (v + 1.0) / 2.0 * (hi - lo + 1e-6) + lo
        raise AssertionError(nt)   # unreachable (guarded in __post_init__)

    # ── public entry ──────────────────────────────────────────────────────────────
    def unapply(self, action_norm, state=None) -> torch.Tensor:
        """Normalized action [H, D_real] (or [1,H,D] / [H,D_full]) -> physical [H, D_real].

        Only the first ``real_dim`` columns are treated as real DoF (the model pads the
        rest). ``state`` (physical proprioception, [D_state]) is required iff any group
        is a delta action (subtract_state=True).
        """
        a = action_norm.detach().float().cpu() if isinstance(action_norm, torch.Tensor) \
            else torch.as_tensor(action_norm, dtype=torch.float32)
        if a.dim() == 3:
            if a.shape[0] != 1:
                raise ValueError(f"expected batch 1, got {tuple(a.shape)}")
            a = a[0]
        if a.dim() != 2:
            raise ValueError(f"action must be [H, D], got {tuple(a.shape)}")
        if a.shape[1] < self.real_dim:
            raise ValueError(f"action has {a.shape[1]} dims < real_dim {self.real_dim}")
        if not bool(torch.isfinite(a).all()):
            raise ValueError("normalized action contains non-finite values")

        arr = a[:, : self.real_dim].numpy().astype(np.float64)
        out = np.empty_like(arr)
        for g, (lo, hi) in zip(self.groups, self._offsets):
            phys = self._unnormalize_group(arr[:, lo:hi], g)
            if g.subtract_state:
                if state is None:
                    raise ValueError(
                        f"group {g.key!r} is a delta action (subtract_state=True) but no "
                        f"state was provided to unapply()."
                    )
                sv = torch.as_tensor(state, dtype=torch.float64).reshape(-1).numpy()
                if sv.size < hi:
                    raise ValueError(
                        f"state has {sv.size} dims, but group {g.key!r} ends at {hi}"
                    )
                if not np.isfinite(sv[lo:hi]).all():
                    raise ValueError(f"state slice for group {g.key!r} is non-finite")
                phys = phys + sv[lo:hi]      # physical = unnorm(delta) + current state
            out[:, lo:hi] = phys
        result = torch.from_numpy(out).to(torch.float32)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("de-normalized physical action contains non-finite values")
        return result

    # ── builders (config-driven; nothing robot-specific baked in) ──────────────────
    @classmethod
    def from_spec(cls, norm_stats_path: str, groups: Sequence[ActionGroup]) -> "ActionDenormalizer":
        """Explicit spec: caller supplies the ordered ActionGroups."""
        with open(norm_stats_path) as f:
            js = json.load(f)
        stats = js.get("norm_stats", js)
        return cls(groups=list(groups), norm_stats=stats)

    @classmethod
    def from_norm_json(
        cls,
        norm_stats_path: str,
        *,
        default_norm_type: str = "meanstd",
        robot_config: Mapping | str | None = None,
    ) -> "ActionDenormalizer":
        """Diagnostic builder: derive action groups straight from the norm JSON.

        Each ``action.*`` key becomes a group (in JSON order); its dim = ``len(mean)``,
        its norm_type = ``default_norm_type`` (meanstd). This cannot prove the checkpoint's
        joint order or norm types and is therefore not used by the public robot-ready facade;
        use ``from_training_config`` or an explicit spec for physical robot actions.
        """
        with open(norm_stats_path) as f:
            js = json.load(f)
        stats = js.get("norm_stats", js)
        subtract = _delta_flags_from_robot_config(robot_config)
        groups = []
        for key, st in stats.items():
            if not key.startswith("action."):
                continue
            mean = np.asarray(st["mean"])
            if mean.ndim == 0:
                raise ValueError(f"{key}.mean must have an action dimension")
            # Stats may be [dim] or horizon-specific [horizon, dim].
            dim = int(mean.shape[-1])
            name = key.split("action.", 1)[1]
            groups.append(ActionGroup(
                key=key, dim=dim, norm_type=default_norm_type,
                subtract_state=bool(subtract.get(key, False)),
                state_key=f"observation.state.{name}",
            ))
        if not groups:
            raise ValueError(f"no 'action.*' keys found in {norm_stats_path}")
        return cls.from_spec(norm_stats_path, groups)

    @classmethod
    def from_training_config(
        cls,
        norm_stats_path: str,
        training_config: Mapping | str,
        *,
        robot_config: Mapping | str | None = None,
    ) -> "ActionDenormalizer":
        """Derive the action groups from the official training config (lingbotvla_cli.yaml).

        Reads ``data.joints`` (order + per-joint dim) and ``data.norm_type`` (per-joint
        norm type) exactly as ``FeatureTransform`` does. ``robot_config`` (the
        ``configs/robot_configs/*.yaml``) is optional and only consulted for
        ``subtract_state`` / delta-action flags; without it, actions are treated as
        absolute (subtract_state=False), which is correct for absolute-joint checkpoints.
        """
        tc = _load_yaml_or_obj(training_config)
        data = tc["data"] if "data" in tc else tc
        # joints: list of "{'arm.position': 14}" -> ordered [(name, dim), ...]
        joint_dims: list[tuple[str, int]] = []
        for entry in data["joints"]:
            d = ast.literal_eval(entry) if isinstance(entry, str) else dict(entry)
            (name, dim), = d.items()
            joint_dims.append((name, int(dim)))
        # norm_type: list of "{'arm.position': 'meanstd'}" -> {name: type}
        ntype: dict[str, str] = {}
        for entry in data["norm_type"]:
            d = ast.literal_eval(entry) if isinstance(entry, str) else dict(entry)
            for name, t in d.items():
                ntype[name] = t

        missing_norm_types = [name for name, _ in joint_dims if name not in ntype]
        if missing_norm_types:
            raise ValueError(
                "training config is missing explicit norm_type entries for action "
                f"groups {missing_norm_types}; refusing to guess identity"
            )

        subtract = _delta_flags_from_robot_config(robot_config)

        groups = []
        for name, dim in joint_dims:
            key = f"action.{name}"
            groups.append(ActionGroup(
                key=key, dim=dim,
                norm_type=ntype[name],
                subtract_state=bool(subtract.get(key, False)),
                state_key=f"observation.state.{name}",
            ))
        return cls.from_spec(norm_stats_path, groups)


def _load_yaml_or_obj(x):
    if isinstance(x, Mapping):
        return x
    import yaml
    with open(x) as f:
        return yaml.safe_load(f)


def _delta_flags_from_robot_config(robot_config) -> dict[str, bool]:
    """Extract per-action subtract_state flags from a robot_config, if given."""
    if robot_config is None:
        return {}
    rc = _load_yaml_or_obj(robot_config)
    flags: dict[str, bool] = {}
    for entry in rc.get("actions", []) or []:
        if isinstance(entry, Mapping):
            (tgt, info), = entry.items()
            if isinstance(info, Mapping) and "subtract_state" in info:
                flags[tgt] = bool(info["subtract_state"])
    return flags
