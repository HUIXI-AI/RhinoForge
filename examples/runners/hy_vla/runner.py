"""Public hy_vla model construction and input binding."""
from __future__ import annotations


from examples.common.inputs import _images, _policy_checkpoint as _checkpoint

# HyEmbodiedPolicy dtype by the resolved specs of the profile's precision recipe:
# FP16, all-W8A16, or W8A16 Vision with G32 RTN W4A16 VLM and action expert.
_ROUTE_PROBES = (("vision",), ("vlm",), ("action",))
_ROUTES = {("fp16",) * 3: "fp16", ("w8a16",) * 3: "w8a16",
           ("w8a16", "w4a16-g32-rtn", "w4a16-g32-rtn"): "w4a16"}


def policy_dtype(example):
    from examples.common.config import owner_route
    return owner_route("HyEmbodiedPolicy", "hy_vla", _ROUTE_PROBES, _ROUTES, selector=example.get("dtype", "fp16"))

def _hy_vla(runner):
    from rpu_backend.api import HyEmbodiedPolicy

    config, inputs = runner.source_config, runner.source_config["input"]
    images = _images(inputs)
    if len(images) not in (1, 2, 3):
        raise ValueError("HY-VLA UMI profile takes one to three camera images, in camera_names order")
    # auto (or omitted) lets the policy take ceil16(2 + 58C + 64), which covers all
    # 64 language slots of C cameras; an explicit bucket is checked by the policy.
    prefix_len = inputs.get("prefix_len", "auto")
    policy = HyEmbodiedPolicy.from_checkpoint(
        str(_checkpoint(config)),
        dtype=policy_dtype(config["example"]),
        prefix_len=None if prefix_len == "auto" else prefix_len,
        n_cameras=len(images),
        runtime_env=config["example"].get("opt_in"),
        rpu_execution=config.get("rpu_execution"),
    ).to("rpu")
    runner.owner = policy
    runner.stage_boundaries = tuple((stage, getattr(policy, "_runner", None), name) for stage, name in (
        ("vision", "_run_vit"), ("prefill", "_run_vlm_prefill"), ("denoise", "_run_denoise")))
    request = {
        "images": images,
        "instruction": inputs["instruction"],
        "state": inputs.get("state"),
        "noise_seed": inputs.get("noise_seed", 0),
    }
    return lambda: policy.infer(**request).actions_normalized


prepare = _hy_vla
