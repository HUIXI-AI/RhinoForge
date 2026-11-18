"""CPU input preparation; configuration checks do not import Torch or backend."""
from __future__ import annotations

import json
import math
from pathlib import Path

from examples.common.libraries import file_sha256


def validate_input_config(inputs):
    for key in ("checkpoint", "training_config"):
        if not isinstance(inputs.get(key), str) or not inputs[key].strip():
            raise ValueError(f"[input].{key} must be a nonempty path")
    modes = [key for key in ("case", "observation", "images") if key in inputs]
    if len(modes) != 1:
        raise ValueError("[input] requires exactly one of case, observation, or images")
    mode = modes[0]
    if mode in ("case", "observation"):
        if not isinstance(inputs[mode], str) or not inputs[mode].strip():
            raise ValueError(f"[input].{mode} must be a nonempty path")
        if {"instruction", "state", "noise"} & inputs.keys():
            raise ValueError("case/observation owns its instruction, state and noise")
    else:
        paths = inputs[mode]
        if not isinstance(paths, list) or len(paths) not in (1, 2, 3) or any(not isinstance(p, str) or not p.strip() for p in paths):
            raise ValueError("[input].images requires one to three image paths")
        if not isinstance(inputs.get("instruction"), str) or not inputs["instruction"].strip():
            raise ValueError("raw images require a nonempty instruction")
        state = inputs.get("state")
        if (not isinstance(state, list) or not 1 <= len(state) <= 55 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in state)):
            raise ValueError("raw images require 1..55 finite [input].state values")
        if "noise" in inputs and (not isinstance(inputs["noise"], str) or not inputs["noise"].strip()):
            raise ValueError("[input].noise must be a tensor file path")
    if "cameras" in inputs and (type(inputs["cameras"]) is not int or inputs["cameras"] not in (1, 2, 3)):
        raise ValueError("[input].cameras must be 1, 2 or 3")
    cap = inputs.get("max_language_tokens", 72)
    if type(cap) is not int or not 1 <= cap <= 160:
        raise ValueError("[input].max_language_tokens must be in 1..160 (default 72)")
    if "expected_language_tokens" in inputs:
        count = inputs["expected_language_tokens"]
        if type(count) is not int or not 1 <= count <= cap:
            raise ValueError(f"[input].expected_language_tokens must be 1..{cap}, the language cap "
                             "[input].max_language_tokens (default 72, at most 160)")


def training_alignment(train, checkpoint_config):
    """Mirror official init_depth_heads/init_video_heads defaults, not YAML guesses.

    AlignConfig's convenience defaults describe one training YAML, so pass every
    relevant field explicitly. Video options inherit depth options as upstream.
    """
    align = train.get("align_params") or {}
    depth = align.get("depth") or {}
    video = dict(depth) | (align.get("video") or {})
    def flag(table, key, default=False):
        value = table.get(key, default)
        if type(value) is not bool:
            raise ValueError(f"training/checkpoint flag {key} must be boolean")
        return value
    enabled = bool(align)
    future_video = enabled and flag(align, "use_future_video")
    patch = future_video and flag(video, "use_patch_loss", True)
    current_patch = future_video and flag(video, "use_current_patch_loss")
    result = dict(use_depth_align=enabled, align_type=align.get("mode", "query"),
        num_task_tokens=align.get("num_task_tokens", 8),
        num_backbone_tokens=depth.get("num_backbone_tokens", 256),
        use_future_depth=enabled and flag(depth, "use_future_depth"),
        use_future_video=future_video, use_future_video_patch=patch,
        use_current_video_patch=current_patch,
        use_current_shared_task_proj=future_video and flag(video, "use_current_shared_task_proj", current_patch),
        use_future_video_cls=future_video and flag(video, "use_cls_loss"),
        future_video_share_future_depth_query=future_video and flag(video, "share_future_depth_query"),
        use_shared_future_task_proj=future_video and flag(video, "use_shared_future_task_proj"),
        block_future_depth_to_action=enabled and flag(depth, "block_future_depth_to_action"),
        block_suffix_to_future_video=future_video and flag(video, "block_suffix_to_future_video"),
        qwen3vl_use_vision_boundaries=flag(checkpoint_config, "qwen3vl_use_vision_boundaries", True),
        vlm_causal=flag(train, "vlm_causal", True))
    if result["align_type"] != "query":
        raise ValueError("LingBot v2 examples require the existing query alignment implementation")
    if future_video and video.get("context_mode", "img_query") != "img_query":
        raise ValueError("video context_mode must match the adapter's img_query implementation")
    if future_video and result["future_video_share_future_depth_query"] and video.get("num_backbone_tokens", 256) != result["num_backbone_tokens"]:
        raise ValueError("shared future video/depth queries require equal backbone token counts")
    for key, default in (("token_top_k", 4), ("token_num_experts", 32), ("chunk_size", 50)):
        if train.get(key, default) != default:
            raise ValueError(f"training {key} differs from the native LingBot v2 contract ({default})")
    if train.get("token_moe_layers", list(range(36))) != list(range(36)):
        raise ValueError("training must retain all 36 MoE Action layers")
    return result


def load_tensor_file(path):
    import torch
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def validate_observation(obs, inputs):
    import torch
    required = {"images", "image_grid_thw", "img_masks", "lang_tokens", "lang_masks", "state", "noise"}
    if not isinstance(obs, dict) or set(obs) != required:
        raise ValueError("observation must contain exactly images, image_grid_thw, img_masks, lang_tokens, lang_masks, state, noise")
    for name, value in obs.items():
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or not torch.isfinite(value).all():
            raise ValueError(f"observation {name} must be a finite CPU tensor")
    images = obs["images"]
    cameras = images.shape[0] if images.ndim == 3 else 0
    if cameras not in (1, 2, 3) or tuple(images.shape[1:]) != (256, 1536) or images.dtype != torch.float32:
        raise ValueError("images must be Float32 [1|2|3,256,1536] (256x256 input images)")
    if obs["img_masks"].dtype != torch.bool or tuple(obs["img_masks"].shape) != (cameras,) or not obs["img_masks"].all():
        raise ValueError("all configured cameras must have true Boolean image masks")
    grid = obs["image_grid_thw"]
    if grid.dtype != torch.int64 or tuple(grid.shape) != (cameras, 3) or not torch.equal(grid, torch.tensor([[1, 16, 16]]).repeat(cameras, 1)):
        raise ValueError("image_grid_thw must be Int64 [cameras,3] with [1,16,16] rows")
    tokens, masks = obs["lang_tokens"], obs["lang_masks"]
    if (tokens.dtype != torch.int64 or masks.dtype != torch.bool or tokens.ndim != 2 or
            tokens.shape[0] != 1 or masks.shape != tokens.shape or not 1 <= tokens.shape[1] <= inputs.get("max_language_tokens", 72)):
        raise ValueError("language tokens/masks require Int64/Boolean [1,L], 1<=L<=language cap")
    count = int(masks.sum())
    if not count or (tokens < 0).any():
        raise ValueError("language must contain nonnegative token IDs and at least one valid token")
    for key, shape in (("state", (1, 55)), ("noise", (1, 50, 55))):
        if obs[key].dtype != torch.float32 or tuple(obs[key].shape) != shape:
            raise ValueError(f"{key} must be Float32 {shape}")
    if inputs.get("cameras", cameras) != cameras or inputs.get("expected_language_tokens", count) != count:
        raise ValueError("observed camera/language counts differ from configured input envelope")
    return {"cameras": cameras, "language_tokens": count, "language_storage_tokens": tokens.shape[1],
            "max_language_tokens": inputs.get("max_language_tokens", 72),
            "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in obs.items()}}


def tokenize_instruction(tokenizer, instruction, max_language_tokens=72):
    """Preserve the complete tokenizer result; the cap is never a truncation target."""
    language = tokenizer(instruction, return_tensors="pt", padding="max_length",
                         truncation=False, max_length=max_language_tokens)
    tokens, mask = language["input_ids"], language["attention_mask"]
    if tokens.ndim != 2 or tokens.shape[0] != 1 or mask.shape != tokens.shape:
        raise ValueError("tokenizer must return matching [1,L] token IDs and attention mask")
    count = int(mask.sum())
    if not count or count > max_language_tokens or tokens.shape[1] > max_language_tokens:
        raise ValueError(f"complete instruction has {count} valid tokens; max_language_tokens="
                         f"{max_language_tokens}; choose a sufficient explicit cap (up to 160)")
    return language


def raw_observation(inputs, checkpoint, seed):
    """Same exact CPU image resize/tokenizer path as Lingbot2Policy's default."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms.v2 import Resize
    from transformers import AutoProcessor
    from .optimized import expanded_path
    processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=False, local_files_only=True)
    arrays = []
    resize = Resize((256, 256))
    for value in inputs["images"]:
        with Image.open(expanded_path(value)) as image:
            tensor = torch.as_tensor(np.array(image.convert("RGB"))).permute(2, 0, 1).contiguous()
        tensor = resize(tensor.float())
        arrays.append(tensor.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).numpy())
    encoded = processor.image_processor(images=arrays, return_tensors="pt")
    language = tokenize_instruction(processor.tokenizer, inputs["instruction"],
                                    inputs.get("max_language_tokens", 72))
    state = torch.zeros(1, 55, dtype=torch.float32)
    state[0, :len(inputs["state"])] = torch.tensor(inputs["state"], dtype=torch.float32)
    if "noise" in inputs:
        noise = load_tensor_file(expanded_path(inputs["noise"]))
        if isinstance(noise, dict):
            if set(noise) != {"noise"}: raise ValueError("noise file must contain only the noise tensor")
            noise = noise["noise"]
    else:
        noise = torch.randn((1, 50, 55), generator=torch.Generator(device="cpu").manual_seed(seed))
    pixel = encoded["pixel_values"]
    return dict(images=pixel.reshape(len(arrays), -1, pixel.shape[-1]).contiguous().float(),
        image_grid_thw=encoded["image_grid_thw"].reshape(len(arrays), 3).long(),
        img_masks=torch.ones(len(arrays), dtype=torch.bool), lang_tokens=language["input_ids"].long(),
        lang_masks=language["attention_mask"].bool(), state=state, noise=noise)


def load_inputs(inputs, *, seed):
    import torch
    import yaml
    from .optimized import expanded_path
    checkpoint = expanded_path(inputs["checkpoint"])
    training_path = expanded_path(inputs["training_config"])
    train = yaml.safe_load(training_path.read_text())["train"]
    checkpoint_config = json.loads((checkpoint / "config.json").read_text())
    align = training_alignment(train, checkpoint_config)
    masks = {key: align[key] for key in ("block_future_depth_to_action", "block_suffix_to_future_video")}
    assets = {"checkpoint": str(checkpoint), "training_config": str(training_path),
        "training_config_sha256": file_sha256(training_path), "mask_flags": masks,
        "alignment": align, "num_steps": 10}
    golden = None
    if "case" in inputs:
        directory = expanded_path(inputs["case"])
        case = json.loads((directory / "case.json").read_text())
        call = case.get("call", {})
        if (case.get("model_id") != "lingbot-vla-v2-6b" or call.get("denoise_steps") != 10 or
                call.get("output_stage") != "full55-before-action-postprocessing" or
                call.get("policy_postprocessor") is not False or call.get("profile_mask_defaults") != masks):
            raise ValueError("case must retain ten native steps, full55 normalized actions and the training alignment masks")
        files = {}
        for key in ("input", "golden"):
            item = case[key]
            path = (directory / item["path"]).resolve(strict=True)
            if not path.is_relative_to(directory) or file_sha256(path) != item["sha256"]:
                raise ValueError(f"case {key} file checksum/path differs")
            files[key] = load_tensor_file(path)
        obs, golden = files["input"], files["golden"]
        if (not isinstance(golden, dict) or "normalized_actions" not in golden or
                golden["normalized_actions"].shape != (1, 50, 55) or not torch.isfinite(golden["normalized_actions"]).all()):
            raise ValueError("case reference requires finite full55 normalized actions")
        if "normalized_action_steps" in golden and golden["normalized_action_steps"].shape != (10, 1, 50, 55):
            raise ValueError("case reference steps must retain ten native steps")
        assets.update(input_mode="case", case=str(directory), case_metadata=case)
    elif "observation" in inputs:
        path = expanded_path(inputs["observation"])
        obs = load_tensor_file(path)
        assets.update(input_mode="observation", observation_path=str(path), observation_sha256=file_sha256(path))
    else:
        obs = raw_observation(inputs, checkpoint, seed)
        assets.update(input_mode="raw_images", images=[str(expanded_path(p)) for p in inputs["images"]],
            instruction=inputs["instruction"], instruction_truncated=False, noise_source="file" if "noise" in inputs else "seeded_cpu_generator")
    if not isinstance(obs, dict):
        raise ValueError("input tensor file must contain an observation mapping")
    obs = {("images" if k == "pixel_values" else "noise" if k == "initial_noise" else k): v for k, v in obs.items()}
    assets["observation"] = validate_observation(obs, inputs)
    return obs, golden, assets, align
