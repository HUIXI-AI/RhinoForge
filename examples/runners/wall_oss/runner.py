"""Public wall_oss model construction and input binding."""
from __future__ import annotations


from examples.common.inputs import _images, _policy_checkpoint as _checkpoint, _model_path

# WallOssPolicy (w8a16, w4a16) flags by the resolved per-module specs of the profile's
# recipe. Both W4 recipes take the W4 route; the loader reads the group layout from the weights.
_ROUTE_PROBES = (("vision",), ("prefill",), ("prefill", "Down"), ("action",), ("action_preprocessor",))
_ROUTES = {("fp16",) * 5: (False, False), ("w8a16",) * 5: (True, False),
           ("w8a16", "w4a16-channel", "w4a16-channel", "w4a16-channel", "fp16"): (False, True),
           ("w8a16", "w4a16-g32", "w8a16", "w4a16-g32", "fp16"): (False, True)}


def wall_stage_boundaries(vla):
    """Vision, Prefill and Denoise calls of WallOssVLA.predict (both Vision entries)."""
    return tuple((stage, getattr(vla, part, None), name) for stage, part, name in (
        ("vision", "vision", "forward"), ("vision", "vision", "_forward_parts"),
        ("prefill", "llm", "forward_embeds"), ("denoise", "action", "denoise_with_mask")))


def _wall_oss(runner):
    from rpu_backend.api import WallOssPolicy

    config, inputs = runner.source_config, runner.source_config["input"]
    images = _images(inputs)
    from examples.common.config import owner_route
    w8, w4 = owner_route("WallOssPolicy", "wall_oss", _ROUTE_PROBES, _ROUTES,
                         selector=config["example"].get("dtype", "fp16"))
    quantized = w8 or w4
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
        w8a16=w8,
        w4a16=w4,
        fp16_ckpt_dir=str(fp16_checkpoint) if fp16_checkpoint else None,
        active_slots=inputs.get("active_slots"),
        runtime_env=config["example"].get("opt_in"),
        rpu_execution=config.get("rpu_execution"),
    ).to("rpu")
    # The facade has no close method; expose the runtime that owns all children.
    runner.owner = policy._vla
    runner.stage_boundaries = wall_stage_boundaries(getattr(policy, "_vla", None))
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


prepare = _wall_oss
