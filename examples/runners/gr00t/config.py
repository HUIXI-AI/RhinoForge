"""Cold policy for the GR00T public examples.

Config checking stays stdlib-only. Tensor/tokenizer checks run before model
installation; native COMPLETE descriptors still select the actual SPM/DDR path.
"""
from __future__ import annotations

import os


def validate_config(config):
    inputs, example = config["input"], config["example"]
    if "expansion_ref" in example:
        raise ValueError("GR00T example.expansion_ref is retired; use the shared main/expansion runtime libraries")
    geometry = {"cameras", "image_size", "text_tokens"}
    if not geometry.intersection(inputs):
        return False  # Existing caller-owned observations keep their old policy.
    for name, choices in (("image_size", (224, 256)),
                          ("num_steps", (4, 10)), ("action_horizon", (40,))):
        value = inputs.get(name)
        if type(value) is not int or value not in choices:
            raise ValueError(f"GR00T input.{name} must be one of {choices}")
    # Both image sizes admit two/three cameras and the same instruction lengths;
    # keep the existing single-camera 256 compatibility. Actual native capacity
    # admission remains authoritative for every shape and precision.
    choices_by_field = {
        "cameras": (1, 2, 3) if inputs["image_size"] == 256 else (2, 3),
        "text_tokens": (32, 64, 128, 160, 192, 224, 256),
    }
    for name, choices in choices_by_field.items():
        value = inputs.get(name)
        if type(value) is not int or value not in choices:
            raise ValueError(f"GR00T input.{name} must be one of {choices} at image_size={inputs['image_size']}")
    if example.get("dtype") not in ("fp16", "w8a16"):
        raise ValueError("GR00T example requires explicit fp16 or w8a16 dtype")
    observation = inputs.get("observation")
    if not isinstance(observation, str) or not observation:
        raise ValueError("GR00T input.observation must name a file")
    return True


def runtime_environment(config):
    if not validate_config(config):
        return {}
    w8 = str(int(config["example"]["dtype"] == "w8a16"))
    return {
        "RPU_GR00T_W8A16": w8,
        "RPU_QWEN3VL_VISION_FUSED_MERGER": "1",
        "RPU_QWEN3VL_VISION_BATCH": "1",
        "RPU_WALL_OSS_FAST_REPLAY": "1",
        "RPU_GR00T_CROSS_KV_REUSE": "1",
        "RPU_GR00T_PRECOMPUTE_ADALN": "1",
        "RPU_GR00T_PRECOMPUTE_TIME_GLUE": "1",
        "RPU_GR00T_KV_SPM_ONLY": "1",
        "RPU_GR00T_CROSS_V_TRANSPOSE_REUSE": "1",
        "RPU_GR00T_FF1_RESIDENT_W8": w8,
        "RPU_GR00T_SELF_QKV_MULTIBASE": w8,
        "RPU_GR00T_RING_XOR3": "1",
        "RPU_GR00T_VL_KV_SPM_ONLY": "1",
        "RPU_GR00T_NATIVE_POSITION_CACHE": "1",
        "RPU_GR00T_QWEN3VL_SPM_Z1": "0",
    }


def configure_environment(config):
    """Bind all cold choices before importing torch/backend; reject conflicts."""
    env = runtime_environment(config)
    if not env:
        return env
    for name, value in env.items():
        inherited = os.environ.get(name)
        if inherited is not None and inherited != value:
            raise ValueError(f"inherited {name} conflicts with the GR00T config; use a fresh process")
    os.environ.update(env)
    return env


def validate_observation(config, observation, tokenizer):
    """Validate real tensors and pure instruction length, not saved metadata."""
    import torch

    inputs = config["input"]
    cameras = inputs["cameras"]
    edge = inputs["image_size"] // 16
    patches = cameras * edge * edge
    for name, shape in (("image_grid_thw", (cameras, 3)),
                        ("pixel_values", (patches, 1536)), ("state", (1, 1, 132))):
        value = observation.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"GR00T observation.{name} must have shape {shape}")
    if observation["image_grid_thw"].tolist() != [[1, edge, edge]] * cameras:
        raise ValueError("GR00T observation.image_grid_thw does not match cameras/image_size")
    for name in ("pixel_values", "state"):
        value = observation[name]
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"GR00T observation.{name} must contain finite floating-point values")
    ids = observation.get("input_ids")
    mask = observation.get("attention_mask")
    if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1
            or ids.dtype not in (torch.int32, torch.int64)
            or not isinstance(mask, torch.Tensor) or mask.shape != ids.shape
            or not bool((mask == 1).all())):
        raise ValueError("GR00T observation requires B1 integer input_ids and an unpadded attention_mask")
    encode = lambda value: tokenizer.encode(value, add_special_tokens=False)
    image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image = (encode("<|vision_start|>") + [image_pad] * (edge // 2) ** 2
             + encode("<|vision_end|>"))
    prefix = encode("<|im_start|>user\n") + image * cameras
    suffix = encode("<|im_end|>\n<|im_start|>assistant\n")
    tokens = ids[0].tolist()
    text = tokens[len(prefix):-len(suffix)]
    if (tokens[:len(prefix)] != prefix or tokens[-len(suffix):] != suffix
            or len(text) != inputs["text_tokens"]
            or any(token in tokenizer.all_special_ids for token in text)):
        raise ValueError(f"GR00T observation must contain {cameras} images followed by text_tokens instruction tokens in the Qwen user template")
    if "mm_token_type_ids" in observation:
        types = observation["mm_token_type_ids"]
        if (not isinstance(types, torch.Tensor) or types.shape != ids.shape
                or not torch.equal(types, (ids == image_pad).to(types.dtype))):
            raise ValueError("GR00T observation.mm_token_type_ids does not match image tokens")
    if "embodiment_id" in observation:
        embodiment = observation["embodiment_id"]
        values = embodiment.reshape(-1).tolist() if isinstance(embodiment, torch.Tensor) else [embodiment]
        if values != [inputs.get("embodiment_id", 20)]:
            raise ValueError("GR00T observation.embodiment_id differs from input.embodiment_id")
