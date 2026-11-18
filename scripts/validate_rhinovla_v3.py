"""Prepare public RhinoVLA v3 inputs and check inference replay."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tomllib


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _factory():
    return _module(Path(__file__).resolve().parents[1] /
                   "python/rpu_backend/adapters/rhinovla/factory.py", "_v3_factory")


def _identity(factory):
    return {"kind": "rhinovla_v3_inputs", "source_commit": factory.V3_SOURCE_SHA,
            "checkpoint_sha256": factory.V3_CHECKPOINT_SHA256,
            "steps": 10, "rtc": False, "action_dim": 96, "action_horizon": 30}


def _request(seed):
    import torch
    generator = torch.Generator().manual_seed(seed)
    result = {"x0": torch.randn(1, 30, 96, generator=generator),
              "state": torch.randn(1, 96, generator=generator) * 0.1,
              "state_mask": torch.zeros(1, 96), "action_mask": torch.zeros(1, 96)}
    result["state_mask"][:, :16] = 1
    result["action_mask"][:, :16] = 1
    result["x0"] *= result["action_mask"][:, None]
    return result


def _rtc_request(request, *, delay, seed):
    """Use reproducible committed actions in the checkpoint's normalized space."""
    import torch
    generator = torch.Generator().manual_seed(seed + 100)
    return {**request, "delay": delay,
            "action_prefix": torch.randn(1, delay, 96, generator=generator) * 0.1}


def _resize_v3_image(image, *, smart_resize, crop_size=None):
    from PIL import Image

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {(width, height)}")
    scale = 256.0 / min(width, height)
    target_height, target_width = smart_resize(
        height * scale,
        width * scale,
        factor=32,
        min_pixels=256 * 256,
        max_pixels=256 * 512,
    )
    resized = image.convert("RGB").resize(
        (int(target_width), int(target_height)), Image.Resampling.BICUBIC,
    )
    if crop_size is None:
        return resized
    crop_width, crop_height = crop_size
    if crop_width > resized.width or crop_height > resized.height:
        raise ValueError(
            f"top_head crop exceeds resized image: image={resized.size}, crop={crop_size}"
        )
    left = (resized.width - crop_width) // 2
    top = (resized.height - crop_height) // 2
    return resized.crop((left, top, left + crop_width, top + crop_height))


def prepare(args):
    import av
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from qwen_vl_utils.vision_process import smart_resize as qwen_smart_resize
    from transformers import AutoProcessor

    factory = _factory()
    factory._validate_official_assets(
        args.source, args.checkpoint, source_sha=factory.V3_SOURCE_SHA,
        checkpoint_sha256=factory.V3_CHECKPOINT_SHA256)
    qwen = _module(args.source / "rhinovla/model/modules/qwen.py", "_v3_qwen_inputs")
    config = OmegaConf.load(args.source / "configs/training/demo_full_finetune.yaml")
    data = config.datasets.vla_data
    min_pixels, max_pixels = qwen._resolve_pixel_budget(config)
    processor = AutoProcessor.from_pretrained(
        args.source / "rhinovla/assets/qwen3_vl_processor", min_pixels=min_pixels,
        max_pixels=max_pixels, local_files_only=True, trust_remote_code=False)
    processor.tokenizer.padding_side = "left"
    images = []
    for camera in ("head_color", "hand_left", "hand_right"):
        path = args.source / (
            f"datasets/example_lerobot_v3/videos/observation.images.{camera}/"
            "chunk-000/file-000.mp4")
        with av.open(str(path)) as container:
            images.append(_resize_v3_image(
                next(container.decode(video=0)).to_image(),
                smart_resize=qwen_smart_resize,
                crop_size=(320, 256) if camera == "head_color" else None,
            ))
    cases = []
    for seed in (0, 1):
        frames = images if seed == 0 else [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                                          for image in images]
        qin = qwen.build_qwenvl_cpu_inputs(
            processor, images=[frames], instructions=["pick up the object"],
            view_roles=[["top_head", "hand_left", "hand_right"]],
            view_modalities=[["rgb"] * 3],
            allowed_roles=data.get("view_role_vocab", qwen._DEFAULT_VIEW_ROLE_VOCAB),
            allowed_modalities=data.get("view_modality_vocab", qwen._DEFAULT_VIEW_MODALITY_VOCAB),
            cot_prompt=data.get("CoT_prompt", "") if "CoT_prompt" in data else None,
            cot_prompt_present="CoT_prompt" in data)
        cases.append({"qwen_inputs": {name: value.detach().cpu().clone()
                                       for name, value in qin.items()},
                      "request": _request(seed)})
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": _identity(factory), "cases": cases}, args.fixture)
    if args.example_dir is not None:
        directory = args.example_dir.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(cases[0]["qwen_inputs"], directory / "qin.pt")
        torch.save(cases[0]["request"], directory / "request.pt")
        (directory / "factory.json").write_text(json.dumps({
            "model_root": str(args.source), "qin": str(directory / "qin.pt")}, indent=2) + "\n")
    print(f"Prepared inputs: {args.fixture}")


_CHILD_CACHES = {"vision": ("vision_model", "_rpu_vision_graph_cache"),
                 "text": ("text_model", "_rpu_text_graph_cache"),
                 "action": ("er", "_rpu_graph_cache")}


def _caches(runtime):
    return {name: getattr(getattr(runtime, owner), cache)
            for name, (owner, cache) in _CHILD_CACHES.items()}


def _snapshot(runtime):
    result = {}
    for name, cache in _caches(runtime).items():
        if not cache.cache_invariant_ok():
            raise RuntimeError(f"{name}: Graph cache invariant failed")
        rows = list(cache.snapshot())
        entries = {repr(row.signature): {field: int(getattr(row, field)) for field in (
            "kernel_count", "segment_count", "replay_count", "recapture_count",
            "graph_lifetime_id", "build_generation")} for row in rows}
        if not rows or len(entries) != len(rows) or cache.size() != len(rows):
            raise RuntimeError(f"{name}: missing or duplicate Graph entry")
        result[name] = entries
    return result


def _check_replay(before, after):
    if before.keys() != after.keys():
        raise RuntimeError("Graph component set changed")
    for component, entries in before.items():
        if entries.keys() != after[component].keys():
            raise RuntimeError(f"{component}: Graph cache grew during replay")
        for signature, old in entries.items():
            new = after[component][signature]
            if (old["kernel_count"] <= 1 or old["segment_count"] < 1
                    or any(new[key] != old[key] for key in (
                        "kernel_count", "segment_count", "graph_lifetime_id", "build_generation"))
                    or old["graph_lifetime_id"] < 1 or old["build_generation"] < 1
                    or old["recapture_count"] != 0 or new["recapture_count"] != 0
                    or new["replay_count"] <= old["replay_count"]):
                raise RuntimeError(f"{component}: Graph did not replay its original build")


def _check_action(action, request):
    import torch
    if (not isinstance(action, torch.Tensor) or tuple(action.shape) != (1, 30, 96)
            or not action.is_floating_point() or not bool(torch.isfinite(action).all())):
        raise RuntimeError("Action must be a finite floating tensor with shape [1,30,96]")
    inactive = request["action_mask"][:, None, :] == 0
    if bool(action.masked_select(inactive.expand_as(action)).any()):
        raise RuntimeError("Inactive action slots must remain zero")
    if "delay" in request:
        delay = request["delay"]
        prefix = request["action_prefix"][:, :delay] * request["action_mask"][:, None]
        if not torch.equal(action[:, :delay].cpu(), prefix):
            raise RuntimeError("RTC committed prefix was not preserved exactly")


def _load_fixture(path, factory):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("metadata") != _identity(factory):
        raise ValueError("Input fixture identity mismatch; rerun prepare")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 2:
        raise ValueError("Input fixture requires two cases")
    for case in cases:
        request = case["request"]
        for name, shape in (("x0", (1, 30, 96)), ("state", (1, 96)),
                            ("state_mask", (1, 96)), ("action_mask", (1, 96))):
            value = request[name]
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
                raise ValueError(f"Input fixture has invalid {name}")
        qin = case["qwen_inputs"]
        if not isinstance(qin, dict) or not qin or any(
                not isinstance(value, torch.Tensor) for value in qin.values()):
            raise ValueError("Input fixture requires tensor qwen_inputs")
    first, second = (case["qwen_inputs"] for case in cases)
    if first.keys() != second.keys() or any(first[key].shape != second[key].shape
                                          or first[key].dtype != second[key].dtype for key in first):
        raise ValueError("Replay cases must use the same input geometry")
    return payload


def run(args):
    import torch
    config = tomllib.loads(args.config.read_text()) if args.config else {}
    if "model" in config:
        if "env" in config:
            raise ValueError("Controlled configuration cannot combine [env] and [model]")
        common = _module(Path(__file__).resolve().parents[1] / "examples/common/config.py", "_v3_common")
        config = common.normalize_config(config)
        switches = config.get("example", {}).get("opt_in", {})
    else:
        if set(config) - {"env", "rpu_execution"}:
            raise ValueError("Controlled configuration accepts only env and rpu_execution")
        switches = config.get("env", {})
    for key, value in switches.items():
        os.environ[key] = str(value)
    from rpu_backend.api import RhinoVLAPolicy

    factory = _factory()
    payload = _load_fixture(args.fixture, factory)
    rtc = getattr(args, "rtc", False)
    delay = getattr(args, "delay", None)
    if delay is not None and not rtc:
        raise ValueError("--delay requires --rtc")
    if rtc:
        delay = 6 if delay is None else delay
        if type(delay) is not int or not 0 <= delay <= 6:
            raise ValueError("RTC delay must be in [0, 6]")
    live_qin = {key: value.clone() for key, value in payload["cases"][0]["qwen_inputs"].items()}
    policy = RhinoVLAPolicy.from_factory(
        "rpu_backend.adapters.rhinovla.factory:v3_runtime_factory",
        {"model_root": args.source, "checkpoint": args.checkpoint, "qin": live_qin,
         "steps": 10, "trust_model_code": True, "rtc": rtc},
        rpu_execution=config.get("rpu_execution", {"action": {"denoise_unroll": True}}))
    actions, live_actions = [], []
    try:
        for cache in _caches(policy.runtime).values():
            cache.begin_warmup()
        for index in [0, 1, 0] + [0] * args.runs:
            case = payload["cases"][index]
            for key, value in case["qwen_inputs"].items():
                policy.runtime.qin[key].copy_(value)
            before = _snapshot(policy.runtime) if actions else None
            request = (_rtc_request(case["request"], delay=delay, seed=index)
                       if rtc else case["request"])
            action = policy.predict(**request)
            _check_action(action, request)
            snapshot = action.detach().cpu().clone()
            for previous, saved in zip(live_actions, actions):
                if not torch.equal(previous.cpu(), saved):
                    raise RuntimeError("A retained action changed during a later request")
                if action.data_ptr() == previous.data_ptr():
                    raise RuntimeError("Action outputs share storage between requests")
            if index == 0 and actions and not torch.equal(snapshot, actions[0]):
                raise RuntimeError("Repeated input changed the action")
            if index == 1 and torch.equal(snapshot, actions[0]):
                raise RuntimeError("Changed input had no effect")
            if before is not None:
                _check_replay(before, _snapshot(policy.runtime))
            else:
                _snapshot(policy.runtime)
                for cache in _caches(policy.runtime).values():
                    cache.freeze()
            live_actions.append(action)
            actions.append(snapshot)
    finally:
        policy.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {**payload["metadata"], "rtc": rtc}
    if rtc:
        metadata["delay"] = delay
    torch.save({"metadata": metadata, "actions": actions}, args.output)
    print(f"Replay checks passed; actions: {args.output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--example-dir", type=Path, help="prepare also writes inputs for the factory TOML examples")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--rtc", action="store_true", help="enable RTC with any controlled precision recipe")
    parser.add_argument("--delay", type=int, choices=range(7), help="RTC committed action count (default: 6)")
    args = parser.parse_args(argv)
    if args.phase == "run" and args.output is None:
        parser.error("run requires --output")
    if args.runs < 1 or args.threads < 1:
        parser.error("runs and threads must be positive")
    if args.delay is not None and not args.rtc:
        parser.error("--delay requires --rtc")
    if args.phase != "run" and args.rtc:
        parser.error("--rtc is a run option; prepared fixtures work with both modes")
    args.source = args.source.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    import torch
    torch.set_num_threads(args.threads)
    (prepare if args.phase == "prepare" else run)(args)


if __name__ == "__main__":
    main()
