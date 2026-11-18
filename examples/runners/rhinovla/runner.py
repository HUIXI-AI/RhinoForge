"""Official and caller-factory RhinoVLA policy construction."""
from __future__ import annotations

import json

from examples.common.inputs import _required_file, _load_mapping, _policy_checkpoint as _checkpoint

_RHINOVLA_QUANT_KEYS = ("expert_w8a16", "full_w8a16")
_RHINOVLA_V3_FACTORY = "rpu_backend.adapters.rhinovla.factory:v3_runtime_factory"
_RHINOVLA_FULL_PIPELINE_PROFILES = frozenset({
    "rhinovla.v3.w8a16.full-pipeline.trusted-factory",
    "rhinovla.v3.w8a16.full-pipeline-high.trusted-factory",
})
_RHINOVLA_RTC_PROFILES = {
    "rhinovla.v3.rtc." + suffix: "rhinovla.v3." + suffix
    for suffix in (
        "fp16.trusted-factory",
        "w8a16.expert.trusted-factory",
        "w8a16.full-expert.trusted-factory",
        "w8a16.full-pipeline.trusted-factory",
        "w8a16.full-pipeline-high.trusted-factory",
    )
}


def _rhinovla_precision_profile(config):
    profile_id = config["example"].get("profile_id")
    return _RHINOVLA_RTC_PROFILES.get(profile_id, profile_id)


def _stage_boundaries(policy):
    """Vision, Prefill and Denoise calls of the RPU pipeline; a caller runtime without them stays untimed."""
    runtime = getattr(policy, "runtime", None)
    stages = (("vision", "_vision"), ("prefill", "_prefill"), ("denoise", "_denoise"))
    if all(callable(getattr(runtime, name, None)) for _, name in stages):
        return tuple((stage, runtime, name) for stage, name in stages)
    return ()


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
        **dict.fromkeys(_RHINOVLA_FULL_PIPELINE_PROFILES, True),
    }
    profile_id = _rhinovla_precision_profile(config)
    if profile_id in profiles and full is not profiles[profile_id]:
        raise ValueError("RhinoVLA full_w8a16 conflicts with the selected profile_id")
    if profile_id in _RHINOVLA_FULL_PIPELINE_PROFILES and any(
            config["example"].get("opt_in", {}).get(name) != "1"
            for name in ("RPU_RHINOVLA_FULL_W8A16", "RPU_RHINOVLA_EXPERT_W8A16")):
        raise ValueError("RhinoVLA full-pipeline profile requires both explicit W8 opt-ins")
    return {"expert_w8a16": expert, "full_w8a16": full}

_V3_SOURCE_KEYS = frozenset({"source_dir", "checkpoint", "fixture", "trust_model_code", "num_steps"})
_W8_SWITCHES = ("RPU_RHINOVLA_EXPERT_W8A16", "RPU_RHINOVLA_FULL_W8A16")


def validate_v3_source_config(config):
    """Check a v3 source entry (pinned source, checkpoint and CPU fixture)."""
    inputs = config["input"]
    if set(inputs) - _V3_SOURCE_KEYS:
        raise ValueError(f"unsupported RhinoVLA v3 source fields: {sorted(set(inputs) - _V3_SOURCE_KEYS)}")
    for key in ("source_dir", "checkpoint", "fixture"):
        if not isinstance(inputs.get(key), str) or not inputs[key].strip():
            raise ValueError(f"RhinoVLA v3 source requires a nonempty {key} path")
    if inputs.get("trust_model_code") is not True:
        raise ValueError("RhinoVLA v3 source requires explicit trust_model_code=true")
    if type(inputs.get("num_steps", 10)) is not int or inputs.get("num_steps", 10) != 10:
        raise ValueError("RhinoVLA v3 source requires num_steps=10")
    expected = _v3_source_quantization(config)
    if config["example"].get("dtype") != ("w8a16" if expected["expert_w8a16"] else "fp16"):
        raise ValueError("RhinoVLA v3 source precision must match its W8A16 switches")


def _v3_source_quantization(config):
    """Expert/full W8 selection named by the profile's pinned switches."""
    opt_in = config["example"].get("opt_in", {})
    if any(opt_in.get(name) not in ("0", "1") for name in _W8_SWITCHES):
        raise ValueError("RhinoVLA v3 source must pin both W8A16 switches to \"0\" or \"1\"")
    full = opt_in["RPU_RHINOVLA_FULL_W8A16"] == "1"
    return {"expert_w8a16": full or opt_in["RPU_RHINOVLA_EXPERT_W8A16"] == "1", "full_w8a16": full}


def validate_rhinovla_config(config):
    """Reject invalid factory input before importing Torch or opening assets."""
    inputs = config["input"]
    if "source_dir" in inputs:
        validate_v3_source_config(config)
        return
    allowed = {"checkpoint", "runtime_factory", "factory_config", "request",
               "trust_model_code", "num_steps", *_RHINOVLA_QUANT_KEYS}
    if set(inputs) - allowed:
        raise ValueError(f"unsupported rhinovla input fields: {sorted(set(inputs) - allowed)}")
    if config["example"].get("dtype") not in {"fp16", "w8a16"}:
        raise ValueError("RhinoVLA dtype must be fp16 or w8a16")
    _rhinovla_quantization(config)
    if (config["example"].get("profile_id") in _RHINOVLA_RTC_PROFILES
            and inputs.get("runtime_factory") != _RHINOVLA_V3_FACTORY):
        raise ValueError("RhinoVLA RTC requires the built-in v3_runtime_factory")
    for key in ("runtime_factory", "factory_config", "request"):
        if not isinstance(inputs.get(key), str) or not inputs[key].strip():
            raise ValueError(f"[input].{key} must be a nonempty string")
    if (inputs["runtime_factory"].count(":") != 1
            or not all(inputs["runtime_factory"].split(":"))):
        raise ValueError("[input].runtime_factory must use module:callable syntax")
    if inputs.get("trust_model_code") is not True:
        raise ValueError("RhinoVLA requires explicit [input].trust_model_code=true")
    if type(inputs.get("num_steps", 10)) is not int or inputs.get("num_steps", 10) != 10:
        raise ValueError("RhinoVLA requires num_steps=10")

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

def _validate_rhinovla_rtc_request(config, factory_config, request):
    """Bind the built-in RTC mode before loading qin or constructing a model."""
    if config["input"]["runtime_factory"] != _RHINOVLA_V3_FACTORY:
        return
    from rpu_backend.adapters.rhinovla.rtc import validate_rtc_request

    rtc = factory_config.get("rtc", False)
    profile_id = config["example"].get("profile_id")
    if profile_id in _RHINOVLA_RTC_PROFILES:
        if rtc is not True:
            raise ValueError("RhinoVLA RTC profile requires factory_config.rtc=true")
    elif rtc is True and profile_id and profile_id.startswith("rhinovla.v3."):
        raise ValueError("RhinoVLA factory_config.rtc=true requires the RTC profile_id")
    validate_rtc_request(
        rtc=rtc, action_prefix=request.get("action_prefix"), delay=request.get("delay"))


def _confirm_rhinovla_quantization(policy, expected, *, full_pipeline=False):
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
    if full_pipeline:
        from rpu_backend.adapters.rhinovla.precision import full_w8_inventory
        if (runtime.er is not expert
                or controller.text_model is not runtime.text_model
                or controller.vision_model is not runtime.vision_model
                or children.get("language_model") is not runtime.text_model
                or children.get("vision_encoder") is not runtime.vision_model):
            raise ValueError("RhinoVLA full W8 child owners do not match the runtime")
        full_w8_inventory(runtime)

def _factory(runner):
    from rpu_backend.api import RhinoVLAPolicy

    config, inputs = runner.source_config, runner.source_config["input"]
    request = _load_mapping(inputs, "request")
    path = _required_file(inputs, "factory_config")
    factory_config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(factory_config, dict):
        raise ValueError("RhinoVLA factory_config must contain a JSON object")
    quantization = _bind_rhinovla_quantization(config, factory_config)
    _validate_rhinovla_rtc_request(config, factory_config, request)
    if inputs["runtime_factory"] == _RHINOVLA_V3_FACTORY:
        factory_config["qin"] = _load_mapping(factory_config, "qin")
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
    runner.stage_boundaries = _stage_boundaries(policy)
    if any(name in inputs or name in factory_config for name in _RHINOVLA_QUANT_KEYS):
        _confirm_rhinovla_quantization(
            policy, quantization,
            full_pipeline=_rhinovla_precision_profile(config) in _RHINOVLA_FULL_PIPELINE_PROFILES)
    return lambda: policy.predict(**request)

def _official(runner):
    from examples.common.inputs import _controlled_checkpoint as _checkpoint
    from rpu_backend.api import RhinoVLAPolicy
    from rpu_backend.adapters.rhinovla.factory import (
        OFFICIAL_CHECKPOINT_SHA256, OFFICIAL_SOURCE_SHA,
    )

    config, inputs = runner.source_config, runner.source_config["input"]
    if inputs.get("trust_model_code") is not True:
        raise PermissionError("official RhinoVLA source requires [input].trust_model_code=true")
    if inputs.get("num_steps", 10) != 10:
        raise ValueError("official RhinoVLA example profile requires num_steps=10")
    root = _checkpoint(config)
    fixture = _load_mapping(
        {"fixture": str(root / "fixture/cpu_fixture.pt")}, "fixture")
    metadata = fixture["metadata"]
    if (metadata.get("kind") != "rhinovla_official_cpu_fp32_nonrtc"
            or metadata.get("source_commit") != OFFICIAL_SOURCE_SHA
            or metadata.get("checkpoint_sha256") != OFFICIAL_CHECKPOINT_SHA256
            or metadata.get("rtc") is not False):
        raise ValueError("RhinoVLA fixture is not the pinned official non-RTC input")
    policy = RhinoVLAPolicy.from_factory(
        "rpu_backend.adapters.rhinovla.factory:official_runtime_factory",
        {"model_root": root / "source",
         "checkpoint": root / "weights/rhinovla_pretrain.ckpt",
         "qin": fixture["qwen_inputs"], "steps": inputs.get("num_steps", 10),
         "trust_model_code": True},
        rpu_execution=config.get("rpu_execution"),
    )
    runner.owner = policy
    runner.stage_boundaries = _stage_boundaries(policy)
    request = {"x0": fixture["noise"], "state": fixture["state"].unsqueeze(0),
               "state_mask": fixture["state_mask"].unsqueeze(0),
               "action_mask": fixture["action_mask"].view(1, 1, -1)}
    return lambda: policy.predict(**request)



def _v3_source(runner):
    """Build the pinned v3 runtime as validate_v3.py does; time fixture case 0."""
    from pathlib import Path
    from rpu_backend.api import RhinoVLAPolicy
    from rpu_backend.adapters.rhinovla.factory import V3_CHECKPOINT_SHA256, V3_SOURCE_SHA

    config, inputs = runner.source_config, runner.source_config["input"]
    validate_v3_source_config(config)
    quantization = _v3_source_quantization(config)
    fixture = _load_mapping(inputs, "fixture")
    if fixture.get("metadata") != {
        "source_commit": V3_SOURCE_SHA, "checkpoint_sha256": V3_CHECKPOINT_SHA256,
        "dtype": "float32", "steps": inputs.get("num_steps", 10), "rtc": False,
        "action_dim": 96, "action_horizon": 30,
    }:
        raise ValueError("RhinoVLA v3 fixture does not match the pinned steps_150000 non-RTC source")
    case = fixture["cases"][0]
    policy = RhinoVLAPolicy.from_factory(
        "rpu_backend.adapters.rhinovla.factory:v3_runtime_factory",
        {"model_root": Path(inputs["source_dir"]).expanduser(),
         "checkpoint": Path(inputs["checkpoint"]).expanduser(),
         "qin": {key: value.clone() for key, value in case["qwen_inputs"].items()},
         "steps": inputs.get("num_steps", 10), "trust_model_code": True, **quantization},
        rpu_execution=config.get("rpu_execution"),
    )
    runner.owner = policy
    runner.stage_boundaries = _stage_boundaries(policy)
    _confirm_v3_source_quantization(policy.runtime, quantization)
    return lambda: policy.predict(**case["request"])


def _confirm_v3_source_quantization(runtime, expected):
    """Inspect the installed expert as validate_v3.py does, not the requested label."""
    from rpu_backend.adapters.rhinovla.convert import _gather_expert_scales
    from rpu_backend.adapters.rhinovla.precision import full_w8_inventory

    if bool(_gather_expert_scales(runtime.er)) is not expected["expert_w8a16"]:
        raise ValueError("RhinoVLA v3 source expert precision does not match the selected profile")
    if expected["full_w8a16"]:
        full_w8_inventory(runtime)


def prepare(runner):
    if "source_dir" in runner.source_config["input"]:
        return _v3_source(runner)
    if "runtime_factory" in runner.source_config["input"]:
        validate_rhinovla_config(runner.source_config)
        return _factory(runner)
    return _official(runner)
