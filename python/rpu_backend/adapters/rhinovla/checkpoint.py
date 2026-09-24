"""Explicit loader for the RhinoVLA action-expert checkpoint component."""
from __future__ import annotations

import hashlib
import importlib.util
import math
import re
import sys
import threading
import warnings
from collections.abc import Mapping
from numbers import Integral, Real
from os import PathLike
from pathlib import Path
from typing import Any


_LAYER_KEY = re.compile(r"^action_expert\.layers\.(\d+)\.")
_MODULE_LOAD_LOCK = threading.RLock()


def _path_argument(name: str, value: str | PathLike[str]) -> Path:
    if not isinstance(value, (str, PathLike)):
        raise TypeError(
            f"RhinoVLA {name} must be str or PathLike, "
            f"got {type(value).__name__}"
        )
    if isinstance(value, str) and not value:
        raise ValueError(f"RhinoVLA {name} must not be empty")
    try:
        path = Path(value)
    except TypeError as exc:
        raise TypeError(
            f"RhinoVLA {name} PathLike must resolve to str"
        ) from exc
    return path


def _integer_at_least(name: str, value: int, minimum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError(
            f"RhinoVLA {name} must be an integer >= {minimum}, got {value!r}"
        )
    return int(value)


def _positive_real(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"RhinoVLA {name} must be a positive finite number, got {value!r}"
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(
            f"RhinoVLA {name} must be a positive finite number, got {value!r}"
        )
    return result


def _expert_source(model_root: Path) -> Path:
    source = (
        model_root
        / "rhinovla/model/modules/action_model/rhino_action_expert.py"
    )
    if not source.is_file():
        raise FileNotFoundError(
            f"RhinoVLA action-expert source not found: {source}"
        )
    return source


def _load_expert_module(model_root: Path):
    return _load_expert_source(_expert_source(model_root))


def _official_expert_source(model_root: Path) -> Path:
    source = model_root / "rhinovla/model/modules/action_expert.py"
    if not source.is_file():
        raise FileNotFoundError(
            f"RhinoVLA official action-expert source not found: {source}"
        )
    return source


def _load_official_expert_module(model_root: Path):
    return _load_expert_source(_official_expert_source(model_root))


def _load_expert_source(source: Path):
    source_bytes = source.read_bytes()
    digest_input = str(source.resolve()).encode() + b"\0" + source_bytes
    digest = hashlib.sha256(digest_input).hexdigest()[:16]
    module_name = f"_rpu_backend_rhinovla_action_{digest}"
    with _MODULE_LOAD_LOCK:
        cached = sys.modules.get(module_name)
        if cached is not None:
            return cached
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load RhinoVLA action-expert source: {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            code = compile(source_bytes, str(source), "exec")
            exec(code, module.__dict__)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module


def _state_tensor(
    state: Mapping[str, Any],
    key: str,
    *,
    ndim: int | None = None,
):
    import torch

    if key not in state:
        raise ValueError(f"RhinoVLA checkpoint is missing required tensor {key!r}")
    value = state[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"RhinoVLA checkpoint {key!r} must be a torch.Tensor, "
            f"got {type(value).__name__}"
        )
    if value.device.type != "cpu":
        raise ValueError(
            f"RhinoVLA checkpoint {key!r} must be on CPU, got {value.device}"
        )
    if not value.is_floating_point() or value.layout is not torch.strided:
        raise TypeError(
            f"RhinoVLA checkpoint {key!r} must be a dense floating tensor"
        )
    if ndim is not None and value.ndim != ndim:
        raise ValueError(
            f"RhinoVLA checkpoint {key!r} must be {ndim}-D, "
            f"got shape {tuple(value.shape)}"
        )
    if any(dim < 1 for dim in value.shape):
        raise ValueError(
            f"RhinoVLA checkpoint {key!r} has an empty dimension: "
            f"{tuple(value.shape)}"
        )
    return value


def _validate_checkpoint_state(
    state: Mapping[str, Any],
    *,
    instance_id: int,
    lora_rank: int,
    plain_weights: bool = False,
) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise TypeError(
            "RhinoVLA checkpoint must contain a state-dict mapping, "
            f"got {type(state).__name__}"
        )
    if any(not isinstance(name, str) for name in state):
        raise TypeError("RhinoVLA checkpoint state-dict keys must be strings")

    relevant_prefixes = (
        "action_expert.",
        "action_io.",
        "state_mask_proj.",
        "action_mask_proj.",
    )
    for name in state:
        if name.startswith(relevant_prefixes):
            _state_tensor(state, name)

    q_weight = _state_tensor(
        state, "action_expert.layers.0.self_attn.q_proj.weight", ndim=2
    )
    k_weight = _state_tensor(
        state, "action_expert.layers.0.self_attn.k_proj.weight", ndim=2
    )
    q_norm = _state_tensor(
        state, "action_expert.layers.0.self_attn.q_norm.weight", ndim=1
    )
    gate_suffix = "weight" if plain_weights else "base.weight"
    gate_weight = _state_tensor(
        state, f"action_expert.layers.0.mlp.gate_proj.{gate_suffix}", ndim=2
    )
    action_weight = _state_tensor(
        state, "action_io.action_in_proj.weight", ndim=2
    )
    state_weight = _state_tensor(
        state, "action_io.state_proj.weight", ndim=2
    )

    head_dim = int(q_norm.shape[0])
    width = int(q_weight.shape[1])
    if k_weight.shape[1] != width:
        raise ValueError(
            "RhinoVLA q_proj/k_proj input widths disagree: "
            f"{tuple(q_weight.shape)} versus {tuple(k_weight.shape)}"
        )
    if q_weight.shape[0] % head_dim or k_weight.shape[0] % head_dim:
        raise ValueError(
            "RhinoVLA q_proj/k_proj rows must be divisible by head_dim, got "
            f"q={q_weight.shape[0]}, k={k_weight.shape[0]}, head_dim={head_dim}"
        )
    num_heads = int(q_weight.shape[0] // head_dim)
    num_kv_heads = int(k_weight.shape[0] // head_dim)
    if num_heads % num_kv_heads:
        raise ValueError(
            "RhinoVLA num_attention_heads must be divisible by "
            f"num_key_value_heads, got {num_heads}/{num_kv_heads}"
        )
    if gate_weight.shape[1] != width:
        raise ValueError(
            "RhinoVLA gate_proj input width must match q_proj width, got "
            f"{tuple(gate_weight.shape)} versus {width}"
        )
    mlp_dim = int(gate_weight.shape[0])

    if action_weight.shape[0] != width or state_weight.shape[0] != width:
        raise ValueError(
            "RhinoVLA action/state projection output widths must match the "
            f"expert width {width}, got {tuple(action_weight.shape)} and "
            f"{tuple(state_weight.shape)}"
        )
    action_dim = int(action_weight.shape[1])
    state_dim = int(state_weight.shape[1])
    expected_mask_shapes = {
        "state_mask_proj.weight": (width, state_dim),
        "state_mask_proj.bias": (width,),
        "action_mask_proj.weight": (width, action_dim),
        "action_mask_proj.bias": (width,),
    }
    for name, expected in expected_mask_shapes.items():
        value = _state_tensor(state, name, ndim=len(expected))
        if tuple(value.shape) != expected:
            raise ValueError(
                f"RhinoVLA checkpoint {name!r} must have shape {expected}, "
                f"got {tuple(value.shape)}"
            )

    layer_indices = {
        int(match.group(1))
        for name in state
        if (match := _LAYER_KEY.match(name)) is not None
    }
    if not layer_indices:
        raise ValueError("RhinoVLA checkpoint contains no action expert layers")
    expected_layers = set(range(max(layer_indices) + 1))
    if layer_indices != expected_layers:
        raise ValueError(
            "RhinoVLA checkpoint action expert layers must be contiguous from "
            f"0, got {sorted(layer_indices)}"
        )
    depth = len(layer_indices)

    profile = {
        "head_dim": head_dim,
        "width": width,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "mlp_dim": mlp_dim,
        "depth": depth,
        "action_dim": action_dim,
        "state_dim": state_dim,
    }
    if plain_weights:
        if any(
            name.startswith("action_expert.")
            and (".base." in name or name.endswith((".lora_a", ".lora_b")))
            for name in state
        ):
            raise ValueError(
                "RhinoVLA official checkpoint requires plain weights, "
                "not instance-LoRA or mixed weights"
            )
        return profile

    base_suffix = ".base.weight"
    lora_a_suffix = ".lora_a"
    lora_b_suffix = ".lora_b"
    base_parents = {
        name[: -len(base_suffix)]
        for name in state
        if name.startswith("action_expert.") and name.endswith(base_suffix)
    }
    lora_a_parents = {
        name[: -len(lora_a_suffix)]
        for name in state
        if name.startswith("action_expert.") and name.endswith(lora_a_suffix)
    }
    lora_b_parents = {
        name[: -len(lora_b_suffix)]
        for name in state
        if name.startswith("action_expert.") and name.endswith(lora_b_suffix)
    }
    if not base_parents:
        raise ValueError("RhinoVLA checkpoint contains no instance-LoRA weights")
    if base_parents != lora_a_parents or base_parents != lora_b_parents:
        raise ValueError(
            "RhinoVLA every action-expert base.weight must have exactly one "
            "matching lora_a and lora_b tensor"
        )

    instance_count = None
    lora_groups = {}
    for parent in sorted(base_parents):
        base = _state_tensor(state, f"{parent}{base_suffix}", ndim=2)
        lora_a = _state_tensor(state, f"{parent}{lora_a_suffix}", ndim=3)
        lora_b = _state_tensor(state, f"{parent}{lora_b_suffix}", ndim=3)
        current_instances = int(lora_a.shape[0])
        if lora_b.shape[0] != current_instances:
            raise ValueError(
                f"RhinoVLA LoRA instance counts disagree for {parent}: "
                f"{lora_a.shape[0]} versus {lora_b.shape[0]}"
            )
        if instance_count is None:
            instance_count = current_instances
        elif current_instances != instance_count:
            raise ValueError(
                "RhinoVLA LoRA instance count must be identical across all "
                f"layers, got {instance_count} and {current_instances} at {parent}"
            )
        expected_a = (current_instances, base.shape[1], lora_rank)
        expected_b = (current_instances, lora_rank, base.shape[0])
        if tuple(lora_a.shape) != expected_a or tuple(lora_b.shape) != expected_b:
            raise ValueError(
                f"RhinoVLA LoRA shapes for {parent} must be {expected_a} and "
                f"{expected_b}, got {tuple(lora_a.shape)} and "
                f"{tuple(lora_b.shape)}"
            )
        lora_groups[parent] = (base, lora_a, lora_b)

    if instance_count is None or instance_id >= instance_count:
        raise IndexError(
            f"RhinoVLA instance_id {instance_id} outside [0, {instance_count})"
        )

    profile["lora_groups"] = lora_groups
    return profile


def _create_rotary(
    *,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    rope_theta: float,
    max_position: int = 4096,
):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextConfig,
        Qwen3VLTextRotaryEmbedding,
    )

    config = Qwen3VLTextConfig(
        hidden_size=head_dim * num_heads,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=max_position,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": rope_theta,
            "mrope_section": [24, 20, 20],
        },
    )
    return Qwen3VLTextRotaryEmbedding(config=config)


def load_action_bundle(
    checkpoint: str | Path,
    *,
    model_root: str | Path,
    action_horizon: int,
    instance_id: int = 0,
    lora_alpha: float = 32,
    lora_rank: int = 32,
    trust_model_code: bool = False,
) -> tuple[Any, Any, Any, Any]:
    """Load ``(expert, action_io, mask_proj, config)`` on CPU in FP32.

    Instance-LoRA MLP weights are merged into their base weights before the
    bundle is returned. The caller must then use :func:`build_rpu_expert` for
    the irreversible FP16 swizzle and RPU installation. ``model_root`` contains
    executable Python model definitions, so callers must explicitly set
    ``trust_model_code=True`` for a reviewed source tree.
    """
    return _load_action_bundle(
        checkpoint, model_root=model_root, action_horizon=action_horizon,
        instance_id=instance_id, lora_alpha=lora_alpha, lora_rank=lora_rank,
        trust_model_code=trust_model_code,
    )


def load_official_action_bundle(
    checkpoint: str | Path,
    *,
    model_root: str | Path,
    action_horizon: int,
    trust_model_code: bool = False,
) -> tuple[Any, Any, Any, Any]:
    """Load the official plain-weight pretrain expert as a CPU FP32 bundle.

    This is an explicit format, not a fallback from the instance-LoRA loader.
    ``model_root`` must be a reviewed official source tree containing
    ``rhinovla/model/modules/action_expert.py``. The returned expert is its
    hidden-input transformer, matching :func:`build_rpu_expert`, not its newer
    complete-policy wrapper. All component state dicts are loaded strictly.

    The official non-RTC policy uses time 1→0 and Qwen prefix RoPE deltas;
    callers must preserve those semantics separately from weight loading.
    """
    return _load_action_bundle(
        checkpoint, model_root=model_root, action_horizon=action_horizon,
        trust_model_code=trust_model_code, official=True,
    )


def _load_action_bundle(
    checkpoint: str | Path,
    *,
    model_root: str | Path,
    action_horizon: int,
    instance_id: int = 0,
    lora_alpha: float = 32,
    lora_rank: int = 32,
    trust_model_code: bool = False,
    official: bool = False,
) -> tuple[Any, Any, Any, Any]:
    checkpoint = _path_argument("checkpoint", checkpoint)
    model_root = _path_argument("model_root", model_root)
    instance_id = _integer_at_least("instance_id", instance_id, 0)
    action_horizon = _integer_at_least("action_horizon", action_horizon, 1)
    lora_rank = _integer_at_least("lora_rank", lora_rank, 1)
    lora_alpha = _positive_real("lora_alpha", lora_alpha)
    if not isinstance(trust_model_code, bool):
        raise TypeError(
            "RhinoVLA trust_model_code must be bool, "
            f"got {trust_model_code!r}"
        )
    if not trust_model_code:
        raise PermissionError(
            "RhinoVLA load_action_bundle executes the action-expert Python "
            "source under model_root; pass trust_model_code=True only for a "
            "reviewed model repository"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"RhinoVLA checkpoint not found: {checkpoint}")

    import torch
    import torch.nn as nn

    state = torch.load(
        checkpoint, map_location="cpu", mmap=True, weights_only=True
    )
    profile = _validate_checkpoint_state(
        state, instance_id=instance_id, lora_rank=lora_rank,
        plain_weights=official,
    )
    if official:
        _official_expert_source(model_root)
    else:
        _expert_source(model_root)
    warnings.warn(
        "RhinoVLA trust_model_code=True executes Python source from the "
        "supplied model_root; use only a reviewed repository.",
        UserWarning,
        stacklevel=2,
    )
    module = (
        _load_official_expert_module(model_root)
        if official else _load_expert_module(model_root)
    )

    width = profile["width"]
    head_dim = profile["head_dim"]
    num_heads = profile["num_heads"]
    num_kv_heads = profile["num_kv_heads"]
    mlp_dim = profile["mlp_dim"]
    depth = profile["depth"]
    action_dim = profile["action_dim"]
    state_dim = profile["state_dim"]
    rope_theta = 5_000_000.0

    config = module.RhinoActionExpertConfig(
        action_dim=action_dim,
        state_dim=state_dim,
        action_horizon=action_horizon,
        width=width,
        depth=depth,
        mlp_dim=mlp_dim,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        rms_norm_eps=1e-6,
        rope_theta=rope_theta,
        hidden_act="silu",
        use_state_token=True,
        use_adarms=True,
        **({"use_mask_condition": True} if official else {}),
    )
    rotary = _create_rotary(
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        rope_theta=rope_theta,
    )
    expert_class = (
        module._RhinoActionTransformer if official else module.RhinoActionExpert
    )
    expert = expert_class(config, qwen_rotary_emb=rotary)
    action_io = module.RhinoActionIO(config)

    scaling = lora_alpha / lora_rank
    expert_state = {}
    for name, value in state.items():
        if not name.startswith("action_expert."):
            continue
        subname = name[len("action_expert.") :]
        if subname.endswith((".lora_a", ".lora_b")):
            continue
        if subname.endswith(".base.weight"):
            parent = name[: -len(".base.weight")]
            _, lora_a, lora_b = profile["lora_groups"][parent]
            delta = torch.matmul(
                lora_a[instance_id].float(),
                lora_b[instance_id].float(),
            ).t().mul(scaling)
            expert_state[subname.replace(".base.weight", ".weight")] = (
                value.float() + delta
            )
        else:
            expert_state[subname] = value.float().clone()
    expert.load_state_dict(expert_state, strict=True)

    action_io.load_state_dict(
        {
            name[len("action_io.") :]: value.float().clone()
            for name, value in state.items()
            if name.startswith("action_io.")
        },
        strict=True,
    )
    mask_proj = nn.ModuleDict(
        {
            "state_mask_proj": nn.Linear(state_dim, width),
            "action_mask_proj": nn.Linear(action_dim, width),
        }
    )
    for prefix in ("state", "action"):
        component_prefix = f"{prefix}_mask_proj."
        mask_proj[f"{prefix}_mask_proj"].load_state_dict(
            {
                name[len(component_prefix):]: value.float().clone()
                for name, value in state.items()
                if name.startswith(component_prefix)
            },
            strict=True,
        )

    expert.eval()
    action_io.eval()
    mask_proj.eval()
    return expert, action_io, mask_proj, config


__all__ = ["load_action_bundle", "load_official_action_bundle"]
