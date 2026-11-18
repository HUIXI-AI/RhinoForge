"""Public g05 model construction and input binding."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

from examples.common.inputs import _load_mapping, _controlled_checkpoint as _checkpoint

def _g05(runner):
    from rpu_backend.adapters.g05 import patch_g05_policy_for_rpu
    from rpu_backend.adapters.g05.release_replay import (
        _checkpoint_manifest, _release_import_transaction, _resolve_release_paths,
    )

    config, inputs = runner.source_config, runner.source_config["input"]
    camera_names = None
    if "g05" in config:
        from examples.common.config import _g05_runner

        camera_names = _g05_runner().input_camera_names(inputs)
    prepare = getattr(runner, "_g05_prepare_inputs", False) is True
    batch = None if prepare else _load_mapping(inputs, "batch")
    if not prepare and not {"samples", "pixel_values"} <= batch.keys():
        raise ValueError("G0.5 batch requires preprocessed samples and pixel_values")
    if not prepare and "g05" in config:
        images = batch["pixel_values"]
        cameras = len(camera_names)
        if not isinstance(images, dict) or len(images) != cameras or any(
                value.ndim != 5 or tuple(value.shape[:3]) != (1, 1, 3)
                for value in images.values()):
            raise ValueError(f"G05 K1 example requires {cameras} actual [1,1,3,H,W] camera tensors")
    release_root = inputs.get("release_root")
    root, checkpoint = _resolve_release_paths(
        _checkpoint(config), Path(release_root) if release_root else None)
    checkpoint_manifest = _checkpoint_manifest(checkpoint)
    with _release_import_transaction(root) as commit:
        from vla_rtc.loader import load_for_eval

        if not Path(sys.modules["vla_rtc.loader"].__file__).resolve().is_relative_to(root):
            raise RuntimeError("vla_rtc.loader is already imported from a different release root")
        policy, _processor = load_for_eval(checkpoint, device="cpu")
        commit()
    if camera_names is not None and len(camera_names) == 2:
        from rpu_backend.adapters.g05 import configure_g05_camera_profile

        configure_g05_camera_profile(policy, _processor, camera_names=camera_names)
    if prepare:
        from examples.runners.g05.inputs import prepare_inputs

        batch, metadata, alternate = prepare_inputs(
            policy, _processor, inputs, config["run"], checkpoint, checkpoint_manifest)
        runner._g05_batch = batch
        runner._g05_input_metadata = metadata
        runner._g05_alternate_batch = alternate
    # CPU input preparation owns no device resources. Publish the owner only when
    # patching begins so input failures do not trigger device retirement cleanup.
    runner.owner = policy
    model = getattr(policy, "model", None)
    runner.stage_boundaries = (("vision", model, "_forward_embed"),
                               ("prefill", getattr(model, "vlm", None), "forward"),
                               ("denoise", model, "inference_fm"))
    patch_g05_policy_for_rpu(
        policy.eval(), rpu_execution=config.get("rpu_execution"),
        **getattr(runner, "_g05_patch_options", {}))
    if not getattr(policy, "_rpu_g05_fm_only_customer_release", False):
        raise ValueError("G0.5 example requires the exact K1 RTC/FM customer release")
    if "g05" in config:
        from rpu_backend.adapters.g05 import configure_g05_vision_camera_batch

        configure_g05_vision_camera_batch(
            policy, config["g05"].get("vision_camera_batch", 1))
    if prepare:
        from examples.runners.g05.inputs import configure_text_envelope

        configure_text_envelope(policy, runner._g05_input_metadata)
        runner._g05_infer_batch = lambda value: policy.predict_action(copy.deepcopy(value))["action"]
    # predict_action updates its batch; a repeated benchmark starts from the
    # original observation, including the caller's RTC prefix each time.
    return lambda: policy.predict_action(copy.deepcopy(batch))["action"]


prepare = _g05
