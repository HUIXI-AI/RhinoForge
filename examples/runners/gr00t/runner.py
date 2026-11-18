"""Public gr00t model construction and input binding."""
from __future__ import annotations

import os

from examples.common.inputs import _load_mapping, _policy_checkpoint as _checkpoint, _model_path

def _gr00t(runner):
    from rpu_backend.adapters.gr00t import build_gr00t_vla
    from examples.runners.gr00t.config import (
        precision_environment, precision_mode, validate_config, validate_observation)

    config, inputs = runner.source_config, runner.source_config["input"]
    selectors = precision_environment(precision_mode(config["example"]))
    geometry_config = validate_config(config)
    observation = _load_mapping(inputs, "observation")
    # Existing benchmark archives wrap the same tensor mapping in inputs/meta.
    # Geometry and text length below are checked against tensors, not metadata.
    if isinstance(observation.get("inputs"), dict):
        observation = observation["inputs"]
    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "state"}
    missing = sorted(required.difference(observation))
    if missing:
        raise ValueError(f"GR00T observation is missing fields: {missing}")
    prefix_rows = int(observation["input_ids"].shape[-1])
    max_prefix_len = inputs.get("max_prefix_len", max(256, (prefix_rows + 15) // 16 * 16))
    if (type(max_prefix_len) is not int or not 0 < max_prefix_len <= 512
            or max_prefix_len % 16 or prefix_rows > max_prefix_len):
        raise ValueError("GR00T input.max_prefix_len must cover the observation, be 16-aligned, and be in 16..512")
    qwen3vl = _model_path(
        inputs.get("qwen3vl_registry_alias"),
        label="[input].qwen3vl_registry_alias",
    )
    if geometry_config:
        from transformers import AutoTokenizer
        validate_observation(config, observation, AutoTokenizer.from_pretrained(str(qwen3vl)))
        import torch
        from rpu_backend.graph import GraphRuntimePolicy
        # Four persistent graphs plus one transient patch-embedding queue.
        if not GraphRuntimePolicy.from_environment(graph_arena_count=5).prepare_arenas():
            raise RuntimeError("GR00T could not prepare five Graph arenas")
        torch.rpu.set_ddr_flush(False)
        torch.rpu.set_ddr_flush_force(True)
    # These selectors are read only while installing/swizzling weights. Keep
    # the explicit profile authoritative without changing the caller's environment.
    previous = {name: os.environ.get(name) for name in selectors}
    os.environ.update(selectors)
    try:
        runtime = build_gr00t_vla(
            str(_checkpoint(config)),
            str(qwen3vl),
            embodiment_id=inputs.get("embodiment_id", 20),
            num_steps=inputs.get("num_steps", 4),
            max_prefix_len=max_prefix_len,
            rpu_execution=config.get("rpu_execution"),
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    runner.owner = runtime
    runner.stage_boundaries = (("backbone", runtime, "_backbone"), ("vl", runtime, "_vl_encode"),
                               ("denoise", runtime, "_denoise"))
    geometry = {"input_ids_shape": list(observation["input_ids"].shape)}
    for _key in ("pixel_values", "pixel_values_videos", "attention_mask", "mm_token_type_ids"):
        _val = observation.get(_key)
        if _val is not None and hasattr(_val, "shape"):
            geometry[_key + "_shape"] = list(_val.shape)
    for _key in ("image_grid_thw", "video_grid_thw"):
        if observation.get(_key) is not None:
            geometry[_key] = observation[_key]
    runner.input_geometry = geometry
    runner.input_geometry.update(state_shape=list(observation["state"].shape),
        prefix_rows=prefix_rows, max_prefix_len=max_prefix_len,
        num_steps=inputs.get("num_steps", 4), embodiment_id=inputs.get("embodiment_id", 20))
    if geometry_config:
        runner.input_geometry["validated_geometry"] = {key: inputs[key]
            for key in ("cameras", "image_size", "text_tokens", "action_horizon")}
    call = lambda: runtime.get_action(
        dict(observation),
        seed=inputs.get("seed", 0),
        num_steps=inputs.get("num_steps", 4),
    )
    return call


prepare = _gr00t
