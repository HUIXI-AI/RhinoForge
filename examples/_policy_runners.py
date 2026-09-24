"""Public inference paths for the registered VLA policy examples.

The shared runner owns timing, output materialization and cleanup.  This module
only resolves caller-owned inputs, constructs the documented public policy, and
returns one repeatable inference call plus its directly closable runtime owner.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace


def _required_file(inputs, key):
    value = inputs.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"[input].{key} must name an existing caller-owned file")
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"[input].{key}: {path}")
    return path


def _load_mapping(inputs, key):
    import torch

    path = _required_file(inputs, key)
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"{path} must be a tensor-only mapping loadable with weights_only=True"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def _images(inputs):
    from PIL import Image

    paths = inputs.get("images")
    if not isinstance(paths, list) or not paths:
        raise ValueError("[input].images must contain at least one image path")
    result = []
    for value in paths:
        path = _required_file({"image": value}, "image")
        with Image.open(path) as image:
            result.append(image.convert("RGB"))
    return result


def _checkpoint(config):
    from rpu_backend.model_registry import model_path

    checkpoint = config["input"].get("checkpoint")
    return Path(checkpoint).expanduser() if checkpoint else Path(model_path(config["example"]["registry_alias"]))


def _model_path(alias, *, label):
    if not isinstance(alias, str) or not alias:
        raise ValueError(f"{label} must be a non-empty model-registry alias")
    from rpu_backend.model_registry import model_path

    return Path(model_path(alias))


def _wall_oss(runner):
    from rpu_backend.api import WallOssPolicy

    config, inputs = runner.source_config, runner.source_config["input"]
    images = _images(inputs)
    dtype = config["example"].get("dtype", "fp16")
    if dtype not in {"fp16", "w8a16", "w4a16", "w4a16-pgrp"}:
        raise ValueError(f"unsupported Wall-OSS example dtype: {dtype!r}")
    quantized = dtype != "fp16"
    fp16_checkpoint = None
    if quantized:
        fp16_checkpoint = _model_path(
            inputs.get("fp16_registry_alias"),
            label="[input].fp16_registry_alias",
        )
    camera_names = inputs.get("camera_names")
    if not isinstance(camera_names, list) or len(camera_names) != len(images):
        raise ValueError("[input].camera_names must match [input].images")

    policy = WallOssPolicy.from_checkpoint(
        str(_checkpoint(config)),
        dataset_key=inputs.get("dataset_key", "berkeley_autolab_ur5"),
        camera_names=tuple(camera_names),
        delta_action=inputs.get("delta_action", False),
        state_bins=inputs.get("state_bins", 256),
        action_hz=inputs.get("action_hz", 32.0),
        action_horizon=inputs.get("action_horizon", 32),
        num_steps=inputs.get("num_steps", 10),
        max_seq_len=inputs.get("max_seq_len", 2048),
        w8a16=dtype == "w8a16",
        w4a16=dtype in {"w4a16", "w4a16-pgrp"},
        fp16_ckpt_dir=str(fp16_checkpoint) if fp16_checkpoint else None,
        active_slots=inputs.get("active_slots"),
        runtime_env=config["example"].get("opt_in"),
        rpu_execution=config.get("rpu_execution"),
    )
    runner.owner = policy
    policy.to("rpu")
    request = {
        "images": images,
        "instruction": inputs["instruction"],
        "proprioception": inputs["proprioception"],
        "noise_seed": inputs.get("noise_seed", 0),
    }
    for name in ("state_mask", "dof_mask", "num_steps"):
        if name in inputs:
            request[name] = inputs[name]
    prefix_range = inputs.get("prefix_length_range")
    if prefix_range is not None:
        policy.prepare_graphs(**request, prefix_length_range=prefix_range)
    return lambda: policy.infer(**request).actions


def _hy_vla(runner):
    from rpu_backend.api import HyEmbodiedPolicy

    config, inputs = runner.source_config, runner.source_config["input"]
    images = _images(inputs)
    if len(images) != 3:
        raise ValueError("HY-VLA UMI profile requires exactly three camera images")
    policy = HyEmbodiedPolicy.from_checkpoint(
        str(_checkpoint(config)),
        dtype=config["example"].get("dtype", "fp16"),
        prefix_len=inputs.get("prefix_len", 240),
        runtime_env=config["example"].get("opt_in"),
        rpu_execution=config.get("rpu_execution"),
    )
    runner.owner = policy
    policy.to("rpu")
    request = {
        "images": images,
        "instruction": inputs["instruction"],
        "state": inputs.get("state"),
        "noise_seed": inputs.get("noise_seed", 0),
    }
    return lambda: policy.infer(**request).actions_normalized


def _lingbot2(runner):
    from rpu_backend.api import Lingbot2Policy

    config, inputs = runner.source_config, runner.source_config["input"]
    if "observation" in inputs:
        request = {"obs": _load_mapping(inputs, "observation")}
    else:
        request = {"images": _images(inputs), "instruction": inputs["instruction"],
                   "proprioception": inputs["proprioception"],
                   "noise_seed": inputs.get("seed", 0)}
    policy = Lingbot2Policy.from_checkpoint(
        str(_checkpoint(config)), dtype=config["example"]["dtype"],
        runtime_env=config["example"].get("opt_in"),
        rpu_execution=config.get("rpu_execution"),
    )
    runner.owner = policy
    policy.to("rpu")
    policy.prepare_graphs(**request, prefix_length_range=inputs.get("prefix_length_range"))
    return lambda: policy.infer(**request).actions_full


def _halo(runner):
    from rpu_backend.adapters.halo import HaloConfig, build_halo_action_expert

    config = runner.source_config
    fixture = _load_mapping(config["input"], "fixture")
    values = fixture.get("kwargs", fixture)
    cfg = HaloConfig()
    positions = values["packed_position_ids"]
    if positions.unique().numel() != 1:
        raise ValueError("HALO single action-step requires one constant packed_position_ids value")
    if (values["packed_text_indexes"].tolist() != list(cfg.text_rows)
            or values["packed_action_token_indexes"].tolist() != list(cfg.action_rows)
            or int(values["key_values_lens"][0]) != cfg.prefix_len):
        raise ValueError("HALO fixture text/action rows or prefix length do not match the single-step profile")
    cache = values["past_key_values"]
    if not isinstance(cache, dict):
        raise ValueError("HALO past_key_values must be a tensor-only key_cache/value_cache mapping")
    checkpoint = _checkpoint(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    policy = build_halo_action_expert(
        checkpoint, rope_position=int(positions.flatten()[0]),
        rpu_execution=config.get("rpu_execution"),
    )
    runner.owner = policy
    policy.insert_prefix([cache["key_cache"][i] for i in range(cfg.num_layers)],
                         [cache["value_cache"][i] for i in range(cfg.num_layers)])
    return lambda: policy.step(values["x_t"], values["timestep"],
                               values["packed_action_position_ids"], values["packed_text_ids"])


_RHINOVLA_QUANT_KEYS = ("expert_w8a16", "full_w8a16")


def _rhinovla_quantization(config):
    """Bind example precision without importing the backend or loading weights."""
    inputs = config["input"]
    for name in _RHINOVLA_QUANT_KEYS:
        if name in inputs and type(inputs[name]) is not bool:
            raise ValueError(f"[input].{name} must be boolean")
    expert = inputs.get("expert_w8a16", False)
    full = inputs.get("full_w8a16", False)
    if config["example"]["dtype"] == "w8a16":
        if any(name not in inputs for name in _RHINOVLA_QUANT_KEYS) or not expert:
            raise ValueError("RhinoVLA w8a16 requires explicit expert_w8a16=true and full_w8a16 boolean")
    elif expert or full:
        raise ValueError("RhinoVLA fp16 conflicts with enabled expert quantization")
    if full and not expert:
        raise ValueError("RhinoVLA full_w8a16 requires expert_w8a16=true")
    profiles = {
        "rhinovla.v3.w8a16.expert.trusted-factory": False,
        "rhinovla.v3.w8a16.full-expert.trusted-factory": True,
    }
    profile_id = config["example"].get("profile_id")
    if profile_id in profiles and full is not profiles[profile_id]:
        raise ValueError("RhinoVLA full_w8a16 conflicts with the selected profile_id")
    return {"expert_w8a16": expert, "full_w8a16": full}


def _bind_rhinovla_quantization(config, factory_config):
    expected = _rhinovla_quantization(config)
    for name, value in expected.items():
        if name in factory_config and (type(factory_config[name]) is not bool
                                       or factory_config[name] is not value):
            raise ValueError(f"RhinoVLA factory_config.{name} conflicts with the selected example")
        # Existing FP16 factories may reject unknown keys. Preserve their call
        # contract unless the caller explicitly selected this quantization API.
        if name in config["input"]:
            factory_config[name] = value
    return expected


def _confirm_rhinovla_quantization(policy, expected):
    """Inspect the installed expert, not a factory-provided precision label."""
    import torch
    from rpu_backend.adapters.rhinovla.convert import _gather_expert_scales
    from rpu_backend.adapters.rhinovla.runtime import _RhinoVLAExecutionController

    runtime = policy.runtime
    controller = getattr(runtime, "_rhinovla_execution_controller", None)
    if (not isinstance(controller, _RhinoVLAExecutionController)
            or controller.runtime is not runtime
            or getattr(policy, "_rhinovla_execution_controller", None) is not controller):
        raise ValueError("RhinoVLA quantization requires the bound runtime execution controller")
    expert = controller.action_expert
    children = getattr(runtime, "_rhinovla_retirement_children", {})
    if children.get("action_expert") is not expert:
        raise ValueError("RhinoVLA quantization action-expert owner does not match the runtime")
    if not getattr(expert, "layers", None):
        raise ValueError("RhinoVLA quantization requires the installed expert's projection weights")
    if bool(_gather_expert_scales(expert)) is not expected["expert_w8a16"]:
        raise ValueError("RhinoVLA factory ignored the selected expert_w8a16 precision")
    if getattr(expert, "_rpu_full_w8a16", None) is not expected["full_w8a16"]:
        raise ValueError("RhinoVLA factory ignored the selected full_w8a16 expert precision")
    if expected["full_w8a16"]:
        owners = getattr(expert, "_rpu_full_w8_cond_owners", None)
        weights = owners.get("weights", ()) if isinstance(owners, dict) else ()
        scales = owners.get("scales", ()) if isinstance(owners, dict) else ()
        if len(weights) != 6 or len(scales) != 3:
            raise ValueError("RhinoVLA full expert W8 requires installed AdaRMS weights and scales")
        for group, scale_group in zip(weights[::2], scales):
            if (len(group) != len(expert.layers) or len(scale_group) != len(group)
                    or any(not isinstance(weight, torch.Tensor) or weight.dtype != torch.int8
                           or not isinstance(scale, torch.Tensor) or scale.dtype != torch.float16
                           or scale.ndim != 1 or not scale.numel() or scale.device != weight.device
                           for weight, scale in zip(group, scale_group))):
                raise ValueError("RhinoVLA full expert W8 requires INT8 AdaRMS weights with FP16 scales")


def _rhinovla(runner):
    from rpu_backend.api import RhinoVLAPolicy

    config, inputs = runner.source_config, runner.source_config["input"]
    request = _load_mapping(inputs, "request")
    path = _required_file(inputs, "factory_config")
    factory_config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(factory_config, dict):
        raise ValueError("RhinoVLA factory_config must contain a JSON object")
    quantization = _bind_rhinovla_quantization(config, factory_config)
    if inputs["runtime_factory"].startswith("your_package."):
        raise ValueError("set [input].runtime_factory to your trusted model integration's module:callable")
    for name, value in (("checkpoint", str(_checkpoint(config))), ("steps", inputs.get("num_steps", 10))):
        if name in factory_config and factory_config[name] != value:
            raise ValueError(f"RhinoVLA factory_config.{name} conflicts with the selected example")
        factory_config[name] = value
    factory_config["trust_model_code"] = True
    policy = RhinoVLAPolicy.from_factory(
        inputs["runtime_factory"], factory_config,
        rpu_execution=config.get("rpu_execution"),
    )
    runner.owner = policy
    if any(name in inputs or name in factory_config for name in _RHINOVLA_QUANT_KEYS):
        _confirm_rhinovla_quantization(policy, quantization)
    return lambda: policy.predict(**request)


PREPARE = {"wall_oss": _wall_oss, "hy_vla": _hy_vla,
           "lingbot2": _lingbot2, "halo": _halo, "rhinovla": _rhinovla}


def validate_policy_config(config):
    """Validate caller-facing fields without importing Torch or opening assets."""
    example, inputs = config["example"], config["input"]
    target, dtype = example["target"], example.get("dtype")
    if target not in PREPARE:
        raise ValueError(f"no public catalog policy entry for {target!r}")
    input_keys = {
        "wall_oss": {"images", "camera_names", "instruction", "proprioception", "noise_seed",
                     "dataset_key", "delta_action", "state_bins", "action_hz", "action_horizon",
                     "num_steps", "max_seq_len", "fp16_registry_alias", "active_slots",
                     "state_mask", "dof_mask", "prefix_length_range"},
        "hy_vla": {"images", "instruction", "state", "noise_seed", "prefix_len"},
        "lingbot2": {"observation", "images", "instruction", "proprioception", "seed", "prefix_length_range"},
        "halo": {"fixture"},
        "rhinovla": {"runtime_factory", "factory_config", "request", "trust_model_code", "num_steps",
                      "expert_w8a16", "full_w8a16"},
    }
    unknown = set(inputs) - (input_keys[target] | {"checkpoint"})
    if unknown:
        raise ValueError(f"unsupported {target} input fields: {sorted(unknown)}")
    allowed = {
        "wall_oss": {"fp16", "w8a16", "w4a16", "w4a16-pgrp"},
        "hy_vla": {"fp16", "w8a16", "w4a16"},
        "lingbot2": {"fp16", "w8a16", "w4a16"},
        "halo": {"fp16"}, "rhinovla": {"fp16", "w8a16"},
    }
    if dtype not in allowed[target]:
        raise ValueError(f"{target} dtype must be one of {sorted(allowed[target])}")
    def string(key):
        if not isinstance(inputs.get(key), str) or not inputs[key].strip():
            raise ValueError(f"[input].{key} must be a nonempty string")
    def integer(key, default, minimum=1):
        if type(inputs.get(key, default)) is not int or inputs.get(key, default) < minimum:
            raise ValueError(f"[input].{key} must be an integer >= {minimum}")
    def images(count=None):
        paths = inputs.get("images")
        if (not isinstance(paths, list) or not paths
                or not all(isinstance(path, str) and path for path in paths)
                or (count is not None and len(paths) != count)):
            raise ValueError("[input].images must contain the profile's camera image paths")
        string("instruction")
    def numeric(key):
        values = inputs.get(key)
        if not isinstance(values, list) or any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError(f"[input].{key} must be a finite numeric array")
    if "checkpoint" in inputs:
        string("checkpoint")
    if target in ("wall_oss", "hy_vla"):
        images(3 if target == "hy_vla" else None)
        integer("noise_seed", 0, 0)
    if target == "wall_oss":
        numeric("proprioception")
        cameras = inputs.get("camera_names")
        if (not isinstance(cameras, list) or len(cameras) != len(inputs["images"])
                or not all(isinstance(name, str) and name for name in cameras)):
            raise ValueError("[input].camera_names must match [input].images")
        if dtype != "fp16":
            string("fp16_registry_alias")
        for key, default in (("num_steps", 10), ("action_horizon", 32), ("max_seq_len", 2048), ("state_bins", 256)):
            integer(key, default)
    elif target == "hy_vla":
        if "state" in inputs:
            numeric("state")
        integer("prefix_len", 240)
    elif target == "lingbot2":
        if "observation" in inputs:
            string("observation")
        else:
            images(3)
            numeric("proprioception")
        if str(example.get("opt_in", {}).get("RPU_LINGBOT2_ALLOW_UNVALIDATED")) != "1":
            raise ValueError("LingBot2 requires explicit RPU_LINGBOT2_ALLOW_UNVALIDATED=1")
    elif target == "halo":
        string("fixture")
    elif target == "rhinovla":
        _rhinovla_quantization(config)
        for key in ("runtime_factory", "factory_config", "request"):
            string(key)
        if inputs["runtime_factory"].count(":") != 1 or not all(inputs["runtime_factory"].split(":")):
            raise ValueError("[input].runtime_factory must use module:callable syntax")
        if inputs.get("trust_model_code") is not True:
            raise ValueError("RhinoVLA requires explicit [input].trust_model_code=true")
        integer("num_steps", 10)
    if "prefix_length_range" in inputs:
        interval = inputs["prefix_length_range"]
        if (not isinstance(interval, list) or len(interval) != 2
                or any(type(value) is not int or value < 1 for value in interval)
                or interval[0] > interval[1]):
            raise ValueError("prefix_length_range must be an ordered positive two-integer interval")


def prepare_policy(config):
    """Prepare one reusable public inference call and its direct runtime owner."""
    validate_policy_config(config)
    runner = SimpleNamespace(source_config=config, owner=None)
    try:
        infer = PREPARE[config["example"]["target"]](runner)
    except BaseException as error:
        if runner.owner is not None:
            from _runner import _close
            try:
                _close(runner.owner, config["example"]["target"])
            except BaseException as cleanup_error:
                error.add_note(f"model setup cleanup failed: {cleanup_error!r}")
        raise
    return infer, runner.owner
