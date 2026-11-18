"""Public lingbot2 model construction and input binding."""
from __future__ import annotations


from examples.common.inputs import _load_mapping, _controlled_images as _images, _controlled_checkpoint as _checkpoint

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
        _checkpoint(config), dtype="w8a16",
        runtime_env=config["example"].get("opt_in"),
        rpu_execution=config.get("rpu_execution"),
    ).to("rpu")
    runner.owner = policy
    runner.stage_boundaries = tuple((stage, getattr(policy, "_policy", None), name) for stage, name in (
        ("vision", "_build_prefix"), ("prefill", "_vlm_fill"),
        ("vl", "_fill_prefix_multiview_spm_z2"), ("action", "_denoise_unroll_run"),
        ("action", "_exp_run")))
    policy.prepare_graphs(
        **request, prefix_length_range=inputs.get("prefix_length_range", [215, 225]))
    return lambda: policy.infer(**request).actions_full


prepare = _lingbot2
