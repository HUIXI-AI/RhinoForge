"""Wall-OSS RTC policy construction."""
from __future__ import annotations

from pathlib import Path

from examples.common.inputs import _controlled_images as _images, _controlled_checkpoint as _checkpoint
from examples.runners.wall_oss.runner import wall_stage_boundaries

# The policy's W8A16 flag by the resolved per-module specs of the profile's recipe.
_ROUTE_PROBES = (("vision",), ("prefill",), ("action",))
_ROUTES = {("fp16",) * 3: False, ("w8a16",) * 3: True}


def _wall_rtc(runner):
    from rpu_backend.api import WallOssPolicy
    from rpu_backend.graph import GraphRuntimePolicy
    from rpu_backend.model_registry import model_path

    config, inputs = runner.source_config, runner.source_config["input"]
    request = {"images": _images(inputs), "instruction": inputs["instruction"],
               "proprioception": inputs["proprioception"],
               "noise_seed": inputs.get("seed", 0)}
    for name in ("state_mask", "dof_mask"):
        if name in inputs:
            request[name] = inputs[name]
    from examples.common.config import owner_route
    w8 = owner_route("Wall-OSS RTC", "wall_oss_rtc", _ROUTE_PROBES, _ROUTES, selector=config["example"].get("dtype", "fp16"))
    fp16_alias = inputs.get("fp16_registry_alias")
    fp16_checkpoint = inputs.get("fp16_checkpoint")
    if w8 and not (fp16_alias or fp16_checkpoint):
        raise ValueError("Wall RTC W8 requires [input].fp16_checkpoint or fp16_registry_alias for CPU glue weights")
    if fp16_checkpoint and not Path(fp16_checkpoint).is_dir():
        raise FileNotFoundError(f"[input].fp16_checkpoint: {fp16_checkpoint}")
    # P640 retains Vision, Prefill and twelve denoise graphs. Patch
    # embedding needs one transient queue; reserve before loading weights.
    if not GraphRuntimePolicy.from_environment(graph_arena_count=15).prepare_arenas():
        raise RuntimeError("Wall RTC could not prepare fifteen Graph arenas")
    policy = WallOssPolicy.from_checkpoint(
        _checkpoint(config), dataset_key="ex_normal", delta_action=True,
        camera_names=("face_view", "left_wrist_view", "right_wrist_view"),
        state_bins=512, action_horizon=20, num_steps=6, max_seq_len=2048,
        w8a16=w8, fp16_ckpt_dir=(fp16_checkpoint or model_path(fp16_alias)) if w8 else None,
        runtime_env=config["example"].get("opt_in"), rtc=True,
        rpu_execution=config.get("rpu_execution"),
    ).to("rpu")
    runner.owner = policy
    runner.stage_boundaries = wall_stage_boundaries(getattr(policy, "_vla", None))
    policy.prepare_graphs(
        **request, prefix_length_range=inputs.get("prefix_length_range", [599, 640]))
    request.update(use_rtc=True, inference_delay=inputs.get("inference_delay", 0))
    if "rtc_prefix_actions" in inputs:
        request["rtc_prefix_actions"] = inputs["rtc_prefix_actions"]
    return lambda: policy.infer(**request).actions


prepare = _wall_rtc
