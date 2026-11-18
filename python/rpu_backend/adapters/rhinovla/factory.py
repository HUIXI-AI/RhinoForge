"""Explicit factories for pinned official RhinoVLA profiles."""
from collections.abc import Mapping
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


OFFICIAL_SOURCE_SHA = "a3edf0c1addf4bf849192642bff3ad15559db3f4"
OFFICIAL_CHECKPOINT_SHA256 = "93be25953542bf59e1b036bcb3be6c161b9a20dc070aaceae768ddb1c6d2b145"
V3_SOURCE_SHA = "e60650377dca38a78bcd4a2e2d58f9e6a631a275"
V3_CHECKPOINT_SHA256 = "2303f9e44cc5847a9990a617cf56c30dfb0ca05f38b6f08b152bf95e67e2f70f"


def _reject_unsafe_diagnostics():
    from rpu_backend.runtime import rpu_env_bool

    for name in ("RPU_RHINOVLA_EXPERT_DEPTH_LIMIT", "RPU_RHINOVLA_EXPERT_MLP_DIM_LIMIT"):
        if os.environ.get(name, "").strip():
            raise ValueError(f"official RhinoVLA forbids math-changing diagnostic {name}")
    for name in ("RPU_RHINOVLA_SKIP_ADARMS_GEMV", "RHINO_SKIP_VT"):
        if rpu_env_bool(name):
            raise ValueError(f"official RhinoVLA forbids math-changing diagnostic {name}")
    if int(os.environ.get("RHINO_SMOKE_PREFIX_PADDING_ROWS", "0")) != 0:
        raise ValueError("official RhinoVLA forbids diagnostic prefix visibility truncation")


def _validate_official_assets(
    model_root, checkpoint, *, source_sha=OFFICIAL_SOURCE_SHA,
    checkpoint_sha256=OFFICIAL_CHECKPOINT_SHA256,
):
    revision = subprocess.check_output(
        ["git", "-C", str(model_root), "rev-parse", "HEAD"], text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(model_root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    if revision != source_sha or dirty:
        raise ValueError("RhinoVLA official factory requires the clean reviewed source revision")
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != checkpoint_sha256:
        raise ValueError("RhinoVLA official factory checkpoint SHA-256 mismatch")


def _load_official_model(model_root, checkpoint, *, rtc=False):
    """Strict CPU FP32 source model; this function never imports the native backend."""
    import torch
    from omegaconf import OmegaConf

    # Python packages are process-global. Do not accidentally reuse a historical
    # `rhinovla` module after the caller selected the new official source tree.
    for name, module in tuple(sys.modules.items()):
        if name == "rhinovla" or name.startswith("rhinovla."):
            source = getattr(module, "__file__", None)
            if source is None or not Path(source).resolve().is_relative_to(model_root):
                raise RuntimeError("a different RhinoVLA source is already imported; use a fresh process")
    sys.path.insert(0, str(model_root))
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        framework = importlib.import_module("rhinovla.model.framework")
        trainer = importlib.import_module("rhinovla.training.trainer_utils.trainer_tools")
        cfg = OmegaConf.load(model_root / "configs/training/demo_full_finetune.yaml")
        cfg.framework.qwenvl.base_vlm = str(model_root / "rhinovla/assets/qwen3_vl_processor")
        cfg.framework.qwenvl.attn_implementation = "sdpa"
        if rtc:
            cfg.framework.action_expert.training_time_rtc = {"max_delay": 6}
        elif "training_time_rtc" in cfg.framework.action_expert:
            cfg.framework.action_expert.training_time_rtc.max_delay = 0
        model = framework.build_framework(cfg).float().eval().requires_grad_(False)
        state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        remapped, stats = trainer.TrainerUtils._adapt_checkpoint_state_dict(model.state_dict(), state)
        trainer.TrainerUtils._raise_on_unexpected_checkpoint_tensors(stats, "<full_model>")
        model.load_state_dict(remapped, strict=True)
        return model
    finally:
        sys.dont_write_bytecode = previous_bytecode
        sys.path.remove(str(model_root))


def _validate_qin_profile(qin, model_config):
    """Validate actual tensor inputs before constructing or mutating weights."""
    import torch

    ids, mask = qin["input_ids"], qin["attention_mask"]
    if not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask.any()):
        raise ValueError("RhinoVLA attention_mask must be nonempty binary visibility")
    if bool((mask[:, 1:].long() > mask[:, :-1].long()).any()):
        raise ValueError("RhinoVLA attention_mask supports right-padding only")
    vocab_size = int(model_config["text_config"]["vocab_size"])
    if bool(((ids < 0) | (ids >= vocab_size)).any()):
        raise ValueError("RhinoVLA input_ids are outside the source vocabulary")
    grid = qin["image_grid_thw"]
    if grid.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("RhinoVLA image_grid_thw must be integer")
    if grid.ndim != 2 or grid.shape[1] != 3 or grid.shape[0] < 1 or bool((grid <= 0).any()):
        raise ValueError("RhinoVLA image_grid_thw must contain positive [N,3] rows")
    if not bool((grid[:, 0] == 1).all()):
        raise ValueError("RhinoVLA ordinary image profile does not admit temporal video grids")
    vision = model_config["vision_config"]
    merge = int(vision["spatial_merge_size"])
    if bool((grid[:, 1:] % merge != 0).any()):
        raise ValueError("RhinoVLA image grid must align with spatial_merge_size")
    patches = int(grid.long().prod(dim=-1).sum())
    patch_width = int(vision["in_channels"] * vision["temporal_patch_size"] * vision["patch_size"] ** 2)
    pixels = qin["pixel_values"]
    if not pixels.is_floating_point() or pixels.shape != (patches, patch_width):
        raise ValueError("RhinoVLA pixel_values shape/dtype disagrees with the source patch geometry")
    if not bool(torch.isfinite(pixels).all()):
        raise ValueError("RhinoVLA pixel_values must be finite")
    image_tokens = ids == int(model_config["image_token_id"])
    if int(image_tokens.sum()) != patches // (merge * merge):
        raise ValueError("RhinoVLA image tokens do not match the patch grid")
    if "mm_token_type_ids" in qin and not torch.equal(qin["mm_token_type_ids"], image_tokens.long()):
        raise ValueError("RhinoVLA mm_token_type_ids disagree with the image tokens")


def official_runtime_factory(config, *, rpu_execution=None):
    """Build through ``RhinoVLAPolicy.from_factory`` using preprocessed CPU qin.

    Config fields: model_root, checkpoint, qin, steps (default 10), and explicit
    trust_model_code=True. Input preprocessing remains the official processor's
    responsibility. RTC is not accepted; no second executor/config parser exists.
    """
    return _runtime_factory(config, rpu_execution=rpu_execution)


def v3_runtime_factory(config, *, rpu_execution=None):
    """Load the pinned v3 96D, horizon-30, depth-18 profile.

    Use the v3 checkpoint remapping, descending flow and prefix RoPE continuation;
    training-only OAT/LoRA tensors are ignored. Configuration and prepared CPU
    inputs follow ``official_runtime_factory``. ``rtc=True`` selects the
    ten-step RTC path with request delays 0 through 6 for the selected FP16
    or W8 precision and execution configuration. Instead of ``qin``,
    ``raw_inputs`` = {images, instruction, view_roles} has the loaded source
    model's own processor build ``qin`` at 256x256. The v1 asset contract
    is separate."""
    return _runtime_factory(config, rpu_execution=rpu_execution, v3=True)


V3_IMAGE_SIZE = (256, 256)


def _check_raw_inputs(raw):
    """Cheap shape checks before the source loads; the source checks the role vocabulary."""
    from PIL import Image

    if not isinstance(raw, Mapping) or set(raw) != {"images", "instruction", "view_roles"}:
        raise ValueError("RhinoVLA raw_inputs must contain exactly images, instruction and view_roles")
    images, roles = raw["images"], raw["view_roles"]
    if not isinstance(images, (list, tuple)) or len(images) not in (1, 2, 3):
        raise ValueError("RhinoVLA raw_inputs.images must hold one to three images")
    if any(not isinstance(image, Image.Image) or image.mode != "RGB" or image.size != V3_IMAGE_SIZE
           for image in images):
        raise ValueError("RhinoVLA raw_inputs.images must be RGB PIL images resized to 256x256")
    if not isinstance(raw["instruction"], str) or not raw["instruction"].strip():
        raise ValueError("RhinoVLA raw_inputs.instruction must be nonempty text")
    if (not isinstance(roles, (list, tuple)) or len(roles) != len(images)
            or any(not isinstance(role, str) for role in roles)):
        raise ValueError("RhinoVLA raw_inputs.view_roles must name one role per image")


def _source_qin(model, raw):
    """validate_v3.py's recipe: the source processor, chat template and RGB modality."""
    qin = model.qwen.build_qwenvl_inputs(
        images=[list(raw["images"])], instructions=[raw["instruction"]],
        view_roles=[list(raw["view_roles"])], view_modalities=[["rgb"] * len(raw["images"])],
    )
    return dict(qin)


def _check_qin_tensors(qin):
    import torch

    if not isinstance(qin, Mapping):
        raise TypeError("RhinoVLA qin must contain preprocessed CPU tensors")
    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    if not required.issubset(qin):
        raise ValueError(f"RhinoVLA qin is missing {sorted(required - set(qin))}")
    extra = set(qin) - required - {"mm_token_type_ids"}
    if extra:
        raise ValueError(f"unsupported RhinoVLA qin fields: {sorted(extra)}")
    if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu"
           for value in qin.values()):
        raise TypeError("RhinoVLA qin values must be CPU tensors")
    ids = qin["input_ids"]
    if ids.dtype != torch.long or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
        raise ValueError("RhinoVLA official qin requires batch=1 and nonempty input_ids")
    if qin["attention_mask"].shape != ids.shape:
        raise ValueError("RhinoVLA input_ids/attention_mask shapes disagree")


def _runtime_factory(config, *, rpu_execution=None, v3=False):
    if not isinstance(config, Mapping):
        raise TypeError("RhinoVLA official factory config must be a mapping")
    allowed = {"model_root", "checkpoint", "qin", "steps", "trust_model_code",
               "expert_w8a16", "full_w8a16", "rtc"} | ({"raw_inputs"} if v3 else set())
    if set(config) - allowed:
        raise ValueError(f"unsupported RhinoVLA official factory keys: {sorted(set(config) - allowed)}")
    if ("qin" in config) == ("raw_inputs" in config):
        raise ValueError("RhinoVLA factory takes exactly one of qin or raw_inputs")
    if config.get("trust_model_code") is not True:
        raise PermissionError("official model code requires explicit trust_model_code=True")
    rtc = config.get("rtc", False)
    if type(rtc) is not bool:
        raise TypeError("RhinoVLA rtc must be bool")
    if rtc and not v3:
        raise ValueError("RhinoVLA RTC requires v3_runtime_factory")
    _reject_unsafe_diagnostics()
    from .runtime import resolve_rhinovla_quantization
    expert_w8a16, full_expert_w8a16, _ = resolve_rhinovla_quantization(
        expert_w8a16=config.get("expert_w8a16"),
        full_expert_w8a16=config.get("full_w8a16"))
    from .checkpoint import _path_argument
    from .action import denoise_schedule
    import torch

    root = _path_argument("model_root", config["model_root"]).resolve()
    checkpoint = _path_argument("checkpoint", config["checkpoint"]).resolve()
    steps = config.get("steps", 10)
    denoise_schedule(steps, "official_descending")
    if rtc:
        from .rtc import require_rtc_native, validate_rtc_install
        from .runtime import RhinoVLAPipelineColdConfig
        from .fused import _RhinoVLANativeColdConfig
        from rpu_backend.runtime import rpu_env_bool
        from rpu_backend.runtime.topology import execution_core_count

        if rpu_env_bool("RPU_GRAPH_FORCE_ONESHOT_ON_REPLAY"):
            raise ValueError("RhinoVLA RTC requires ordinary Graph replay, not forced oneshot")
        cores = execution_core_count(rpu_execution)
        pipeline_cold, _ = RhinoVLAPipelineColdConfig.resolve(rpu_execution)
        validate_rtc_install(
            rtc=rtc, steps=steps, flow_direction="official_descending",
            expert_w8a16=expert_w8a16, full_expert_w8a16=full_expert_w8a16,
            native_cold=_RhinoVLANativeColdConfig.from_env(num_cores=cores),
            action_cold=pipeline_cold.stages["action"])
        require_rtc_native()
    raw = config.get("raw_inputs")
    if raw is None:
        qin = config["qin"]
        _check_qin_tensors(qin)
    else:
        _check_raw_inputs(raw)
    if v3:
        _validate_official_assets(
            root, checkpoint, source_sha=V3_SOURCE_SHA,
            checkpoint_sha256=V3_CHECKPOINT_SHA256,
        )
    else:
        _validate_official_assets(root, checkpoint)
    model_config = json.loads((root / "rhinovla/assets/qwen3_vl_processor/config.json").read_text())
    if raw is None:
        _validate_qin_profile(qin, model_config)
    model = (_load_official_model(root, checkpoint, rtc=True) if rtc
             else _load_official_model(root, checkpoint))
    if raw is not None:
        # Built by the loaded source, then held to the same checks before any RPU install.
        qin = _source_qin(model, raw)
        _check_qin_tensors(qin)
        _validate_qin_profile(qin, model_config)
    ids = qin["input_ids"]
    full_expert = model.action_expert
    expert = full_expert.transformer
    # Upstream shares this small module with Qwen; Qwen's later .half() must
    # not mutate the CPU action reference's rotary embedding.
    expert.qwen_rotary_emb = copy.deepcopy(expert.qwen_rotary_emb)
    masks = torch.nn.ModuleDict({
        "state_mask_proj": full_expert.state_mask_proj,
        "action_mask_proj": full_expert.action_mask_proj,
    })
    bundle = (expert, full_expert.io, masks, model.expert_config)
    from .pipeline import RhinoVLAOnRPU
    return RhinoVLAOnRPU(
        qin=qin, prefix_len=int(ids.shape[1]), steps=steps,
        model=model, action_bundle=bundle, flow_direction="official_descending",
        rpu_execution=rpu_execution,
        expert_w8a16=expert_w8a16, full_expert_w8a16=full_expert_w8a16,
        rtc=rtc,
    )


__all__ = ["official_runtime_factory", "v3_runtime_factory"]
