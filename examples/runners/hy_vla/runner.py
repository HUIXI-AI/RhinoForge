"""Public hy_vla model construction and input binding."""
from __future__ import annotations


from examples.common.inputs import _images, _policy_checkpoint as _checkpoint

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
