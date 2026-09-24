"""Offline, bounded-memory loading for the audited 2B hybrid delivery only.

The public HF-compatible loader is unchanged. This entry consumes ordinary
FP16/BF16 safetensors, converts one tensor at a time, and retains only the
delivery's final FP16/int8 tensors. No runtime swizzle happens here.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForImageTextToText, GenerationConfig

from rpu_backend.quant._common import quantize_linear_per_channel
from rpu_backend.quant.load import _materialize_imagetext_meta_buffers, _raise_on_imagetext_meta


def _checkpoint_inventory(root, expected):
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(str(root / "model.safetensors"), framework="pt") as handle:
            weight_map = {name: "model.safetensors" for name in handle.keys()}
    # Validate the complete inventory and shapes before installing any weight.
    if set(weight_map) not in (set(expected), set(expected) - {"lm_head.weight"}):
        raise ValueError("delivery checkpoint has missing or unexpected tensor names")
    for filename in sorted(set(weight_map.values())):
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts or not filename.endswith(".safetensors"):
            raise ValueError("checkpoint shard must be a relative safetensors path")
        shard = (root / relative).resolve()
        if root not in shard.parents:
            raise ValueError("checkpoint shard escapes the checkpoint directory")
        names = {name for name, target in weight_map.items() if target == filename}
        with safe_open(str(shard), framework="pt") as handle:
            if set(handle.keys()) != names:
                raise ValueError("shard inventory disagrees with its index")
            for name in names:
                view = handle.get_slice(name)
                if tuple(view.get_shape()) != tuple(expected[name].shape):
                    raise ValueError(f"checkpoint shape mismatch: {name}")
                if view.get_dtype() not in {"F16", "BF16"}:
                    raise ValueError(f"delivery checkpoint must be FP16/BF16: {name}")
    return weight_map


def load_delivery_model(path, *, rpu_execution=None):
    """Exact 28-layer 2B: W8 layers 0..26 plus layer-27 MLP; FP16 otherwise."""
    from . import (Qwen3VLAdapter, _QWEN3_VL_DELIVERY_PROFILE,
                   _require_delivery_opt_in, _validate_delivery_config,
                   _normalize_delivery_execution)

    _require_delivery_opt_in()

    root = Path(path).resolve(strict=True)
    config = AutoConfig.from_pretrained(root, local_files_only=True)
    _validate_delivery_config(config)
    text, vision = config.text_config, config.vision_config
    text_fields = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                   "num_key_value_heads", "head_dim", "vocab_size")
    vision_fields = ("depth", "hidden_size", "intermediate_size", "num_heads", "out_hidden_size",
                     "patch_size", "temporal_patch_size", "spatial_merge_size")
    if (config.architectures != ["Qwen3VLForConditionalGeneration"]
            or tuple(getattr(text, key) for key in text_fields) != (2048, 6144, 28, 16, 8, 128, 151936)
            or tuple(getattr(vision, key) for key in vision_fields) != (24, 1024, 4096, 16, 2048, 16, 2, 2)
            or list(vision.deepstack_visual_indexes) != [5, 11, 17]
            or not config.tie_word_embeddings
            or any(getattr(cfg, key, None) is not None for cfg in (config, text, vision)
                   for key in ("quantization_config", "quant_config"))):
        raise ValueError("streamed delivery loading requires the exact unquantized Qwen3-VL-2B profile")
    execution = _normalize_delivery_execution(rpu_execution)
    Qwen3VLAdapter.preflight(config)
    Qwen3VLAdapter.preflight_execution(config, execution)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config, dtype=torch.float16)
    parameters = dict(model.named_parameters(remove_duplicate=False))
    weight_map = _checkpoint_inventory(root, parameters)
    modules = dict(model.named_modules())
    prefix = "model.language_model.layers."
    projections = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    for name, filename in weight_map.items():
        # Closing the mapping after each tensor prevents resident pages from
        # accumulating across a multi-GB shard. Retained tensors own storage.
        with safe_open(str(root / filename), framework="pt") as handle:
            raw = handle.get_tensor(name)
            weight = raw.to(dtype=torch.float16, copy=True)
            del raw
        module_name, parameter_name = name.rsplit(".", 1)
        module = modules[module_name]
        parts = name[len(prefix):].split(".") if name.startswith(prefix) else []
        if len(parts) == 4 and parts[2] in projections and parts[3] == "weight":
            quantized = int(parts[0]) < 27 or parts[2] in {"gate_proj", "up_proj", "down_proj"}
            if quantized:
                weight, scale = quantize_linear_per_channel(weight)
            else:
                scale = torch.ones(weight.shape[0], dtype=torch.float16)
            module.register_buffer("weight_scale", scale)
        setattr(module, parameter_name, torch.nn.Parameter(weight, requires_grad=False))
    if "lm_head.weight" not in weight_map:
        model.lm_head.weight = model.model.language_model.embed_tokens.weight
    _materialize_imagetext_meta_buffers(model)
    _raise_on_imagetext_meta(model)
    generation = root / "generation_config.json"
    if generation.is_file():
        model.generation_config = GenerationConfig.from_pretrained(root, local_files_only=True)
    model._qwen3_vl_delivery_profile = _QWEN3_VL_DELIVERY_PROFILE
    model._rpu_execution = execution
    adapter = Qwen3VLAdapter(model)
    adapter._rpu_execution = execution
    return model.eval()
