"""Exact runtime-scope Qwen3-VL quantization, separate from Text-W8 and AWQ.

Registered runtime checkpoints quantize Text7 and the independent head.
Vision/embed/norm tensors remain FP16 here. W4 source storage is logical signed
two's-complement, not compressed-tensors offset nibbles or controller storage.
The existing 4B W8 checkpoint continues to use its original loader.
"""
from __future__ import annotations

import copy
import gc
from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from transformers import AutoModelForImageTextToText

from .convert_qwen3_vl import _get, matches_runtime_geometry, runtime_quant_config
from .int4_pgrp_pack import swizzle_int4_pgrp_scale, swizzle_pack_int4_pgrp
from .load_qwen3_vl_awq import (
    _ROLES, _config_signature, _inventory, _projection_inventory,
    _allocate_awq_scale_views,
)

_PLAN_ATTR = "_rpu_qwen3_vl_runtime_load_plan"


def is_qwen3_vl_runtime_quant_config(config) -> bool:
    """Exact runtime profiles; never legacy Text-W8 or raw AWQ assets."""
    try:
        text, vision = _get(config, "text_config"), _get(config, "vision_config")
        qc = _get(config, "quant_config")
        if not isinstance(qc, dict) or not matches_runtime_geometry(config):
            return False
        h = _get(text, "hidden_size")
        bits = {"w8a16": 8, "w4a16": 4}.get(qc.get("method"))
        if bits is None or (h == 2560 and bits == 8):
            return False
        return (
            _get(config, "tie_word_embeddings") is False
            and _get(text, "tie_word_embeddings") is False
            and _get(config, "quantization_config") is None
            and all(_get(owner, key) is None for owner in (text, vision)
                    for key in ("quant_config", "quantization_config"))
            # JSON identity also rejects bool-as-int and 32.0-as-32 controls.
            and json.dumps(qc, sort_keys=True) == json.dumps(runtime_quant_config(
                h, bits, source_quantization=qc.get("source_quantization")), sort_keys=True)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _quantized_modules(model):
    modules = _projection_inventory(model)
    head = model.lm_head
    if not isinstance(head, nn.Linear) or head.bias is not None:
        raise ValueError("runtime quantization requires an independent bias-free Linear head")
    if head.weight is model.model.language_model.embed_tokens.weight:
        raise ValueError("runtime quantized head must not alias embedding")
    return [*modules, ("lm_head", head, 1)]


def _pack_w4_projection(weight, scale, *, n, k, partition):
    """Logical signed nibbles -> the existing sole pgrp physical ABI, once."""
    if (weight.device.type != "cpu" or weight.dtype != torch.uint8
            or tuple(weight.shape) != (n, k // 2) or not weight.is_contiguous()
            or k % 32 or n <= 0 or k <= 0):
        raise ValueError("runtime W4 weight requires contiguous CPU uint8[N,K/2], K divisible by 32")
    if (scale.device.type != "cpu" or scale.dtype != torch.float16
            or tuple(scale.shape) != (k // 32, n) or not scale.is_contiguous()
            or not bool(torch.isfinite(scale).all() and (scale > 0).all())):
        raise ValueError("runtime W4 scale requires finite positive contiguous CPU FP16[G,N]")
    nibbles = torch.stack((weight & 15, weight >> 4), dim=-1).reshape(n, k).to(torch.int8)
    values = torch.where(nibbles >= 8, nibbles - 16, nibbles)
    return (swizzle_pack_int4_pgrp(values, partition, 8).reshape(n, k // 2),
            swizzle_int4_pgrp_scale(scale, 32, partition, 8))


def _copy_w8_projection(weight, scale, *, n, k, copy_weight=True):
    if (weight.device.type != "cpu" or weight.dtype != torch.int8
            or tuple(weight.shape) != (n, k) or not weight.is_contiguous()):
        raise ValueError("runtime W8 weight requires contiguous CPU int8[N,K]")
    if (scale.device.type != "cpu" or scale.dtype != torch.float16
            or tuple(scale.shape) != (n,) or not scale.is_contiguous()
            or not bool(torch.isfinite(scale).all() and (scale > 0).all())):
        raise ValueError("runtime W8 scale requires finite positive contiguous CPU FP16[N]")
    # Some Torch CPU builds retain freed large allocations in a private
    # allocator (mimalloc on the tested board). Keep the checkpoint copy in
    # NumPy-owned storage so replacing one layer releases its raw CPU pages,
    # instead of retaining a second full model while HostDDR banks grow.
    # The subsequent swizzle and device copy keep their existing byte layout.
    # The one-step 32B RPU loader keeps safetensors-backed CPU weights until
    # the existing per-layer swizzle copies them into device banks. Copying
    # the entire raw model first exhausts non-CMA CPU RAM on the target board.
    return (torch.from_numpy(weight.numpy().copy()) if copy_weight else weight), scale.clone()


def _module_inventory(model):
    return tuple((name, id(module), module.in_features, module.out_features,
                  getattr(module, "_rpu_linear_partition", None),
                  getattr(module, "_rpu_linear_num_cores", None))
                 for name, module, _ in _quantized_modules(model))


@dataclass(frozen=True)
class _RuntimeLoadPlan:
    model_id: int
    config_signature: str
    tensor_inventory: tuple
    projections: tuple


def _validate_qwen3_vl_runtime_model(model) -> bool:
    """Validate the sealed CPU owner before allocation or irreversible swizzle."""
    plan = vars(model).get(_PLAN_ATTR)
    selected = is_qwen3_vl_runtime_quant_config(getattr(model, "config", None))
    if plan is None and not selected:
        return False
    if not selected or not isinstance(plan, _RuntimeLoadPlan) or plan.model_id != id(model):
        raise RuntimeError("Qwen3-VL runtime quantization requires its original CPU load plan")
    if (_config_signature(model.config) != plan.config_signature
            or _inventory(model) != plan.tensor_inventory):
        raise RuntimeError("Qwen3-VL runtime CPU configuration or tensor inventory changed after loading")
    if any(t.device.type != "cpu" for t in (*model.parameters(), *model.buffers())):
        raise RuntimeError("Qwen3-VL runtime load plan requires CPU tensors before installation")
    if _module_inventory(model) != plan.projections:
        raise RuntimeError("Qwen3-VL runtime packed projection/head layout changed after loading")
    return True


@torch.inference_mode(False)
def load_qwen3_vl_runtime_imagetext(
    config, ckpt_dir: str, *, dtype=torch.float16, _mmap_w8_weights=False, **resolve_kwargs,
):
    """Strict CPU loading; W8 stays raw and W4 is packed exactly once.

    The internal 32B W8 one-step RPU path retains checkpoint mappings instead
    of copying all raw weights. Checkpoint files must remain unchanged during
    loading/migration; tensors own their mappings after safe_open closes.
    Ordinary CPU loading still returns independent weight copies.
    """
    from .load import (
        _resolve_checkpoint_dir, _safetensor_shards, _install_parameter,
        _materialize_imagetext_meta_buffers, _raise_on_imagetext_meta,
    )
    if dtype is not torch.float16:
        raise TypeError("Qwen3-VL runtime quantization requires torch.float16 activations")
    if not is_qwen3_vl_runtime_quant_config(config):
        raise ValueError("runtime loader requires an exact registered runtime quant profile")
    if _mmap_w8_weights and not (
        config.text_config.hidden_size == 5120 and config.quant_config["method"] == "w8a16"
    ):
        raise ValueError("mapped runtime weights require the exact 32B W8 RPU loading path")
    directory = _resolve_checkpoint_dir(ckpt_dir, **resolve_kwargs)
    source_config = json.loads((directory / "config.json").read_text())
    if (not is_qwen3_vl_runtime_quant_config(source_config)
            or _get(source_config["text_config"], "hidden_size") != config.text_config.hidden_size
            or source_config["quant_config"] != config.quant_config):
        raise ValueError("runtime checkpoint configuration differs from selected profile")
    skeleton = copy.deepcopy(config)
    del skeleton.quant_config
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(skeleton)
    model.config.quant_config = copy.deepcopy(config.quant_config)
    projections = _quantized_modules(model)
    if len(projections) != model.config.text_config.num_hidden_layers * 7 + 1:
        raise ValueError("runtime loader requires Text7 for every layer plus one independent head")
    bits = 4 if config.quant_config["method"] == "w4a16" else 8
    shapes = {name: tuple(t.shape) for name, t in model.named_parameters(remove_duplicate=False)}
    expected = {name: (shape, "F16") for name, shape in shapes.items()}
    for name, module, partition in projections:
        n, k = module.out_features, module.in_features
        if bits == 4 and ((partition == 1 and (n % 128 or k % 64))
                         or (partition == 0 and (n % 16 or k % 512))):
            raise ValueError(f"runtime W4 projection is outside TP8 pgrp geometry: {name}")
        expected[name + ".weight"] = ((n, k if bits == 8 else k // 2), "I8" if bits == 8 else "U8")
        expected[name + ".weight_scale"] = ((n,) if bits == 8 else (k // 32, n), "F16")
    tensor_shards = {}
    for shard in _safetensor_shards(directory):
        with safe_open(shard, framework="pt") as handle:
            for name in handle.keys():
                if name in tensor_shards or name not in expected:
                    raise ValueError(f"Unexpected or duplicate runtime tensor: {name}")
                view = handle.get_slice(name)
                if (tuple(view.get_shape()), view.get_dtype()) != expected[name]:
                    raise ValueError(f"Invalid runtime tensor metadata: {name}")
                tensor_shards[name] = str(shard)
    if set(tensor_shards) != set(expected):
        raise ValueError(f"runtime tensor inventory differs: missing {sorted(set(expected)-set(tensor_shards))[:8]}")
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        declared = json.loads(index.read_text())["weight_map"]
        if set(declared) != set(tensor_shards) or any(
                Path(tensor_shards[n]).resolve() != (directory / declared[n]).resolve() for n in declared):
            raise ValueError("runtime shard index differs from actual tensor locations")
    with ExitStack() as stack:
        handles = {p: stack.enter_context(safe_open(p, framework="pt")) for p in set(tensor_shards.values())}
        def tensor(name):
            return handles[tensor_shards[name]].get_tensor(name)
        for name, module, partition in projections:
            weight, scale = tensor(name + ".weight"), tensor(name + ".weight_scale")
            if bits == 4:
                weight, scale = _pack_w4_projection(weight, scale,
                    n=module.out_features, k=module.in_features, partition=partition)
                module._rpu_linear_partition = partition
                module._rpu_linear_num_cores = 8
            else:
                weight, scale = _copy_w8_projection(
                    weight, scale, n=module.out_features, k=module.in_features,
                    copy_weight=not _mmap_w8_weights)
            module.weight = nn.Parameter(weight, requires_grad=False)
            module.register_buffer("weight_scale", scale, persistent=False)
        quantized_names = {name + ".weight" for name, _, _ in projections}
        for name in sorted(set(shapes) - quantized_names):
            value = tensor(name)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Non-finite runtime floating tensor: {name}")
            _install_parameter(model, name, value.clone().contiguous())
    _materialize_imagetext_meta_buffers(model)
    _raise_on_imagetext_meta(model)
    model.requires_grad_(False).eval()
    setattr(model, _PLAN_ATTR, _RuntimeLoadPlan(id(model), _config_signature(model.config),
        _inventory(model), _module_inventory(model)))
    _validate_qwen3_vl_runtime_model(model)
    return model


@torch.inference_mode(False)
def _move_materialized_runtime_decoder_for_rpu(model) -> None:
    """Use existing banks; packed W4 copies without another swizzle. Head stays CPU."""
    from .load import _allocate_decoder_projection_views, _move_materialized_decoder_for_rpu
    if vars(model).get("_rpu_swizzle_started") is not True:
        raise RuntimeError("runtime decoder migration requires adapter ownership/poison first")
    if not _validate_qwen3_vl_runtime_model(model):
        raise ValueError("runtime decoder migration requires the exact loaded model")
    per_layer = model.config.text_config.hidden_size in (4096, 5120)
    if model.config.quant_config["method"] == "w8a16":
        if per_layer:
            _move_materialized_decoder_for_rpu(model, dtype=torch.int8, per_layer=True)
        else:
            _move_materialized_decoder_for_rpu(model, dtype=torch.int8)
        return
    names = tuple(tuple(f"model.language_model.layers.{i}.{role}.weight" for role, _ in _ROLES)
                  for i in range(len(model.model.language_model.layers)))
    views = (None if per_layer else
             _allocate_decoder_projection_views(model, names, dtype=torch.uint8))
    scale_views = _allocate_awq_scale_views(model)
    for i, layer in enumerate(model.model.language_model.layers):
        if per_layer:
            views = _allocate_decoder_projection_views(model, (names[i],), dtype=torch.uint8)
        for role, _ in _ROLES:
            name = f"model.language_model.layers.{i}.{role}"
            module = layer.get_submodule(role)
            target = views[name + ".weight"]
            target.copy_(module.weight.detach())
            module.weight = nn.Parameter(target, requires_grad=False)
            module.weight_scale = scale_views[name]
        layer.to("rpu")
        if per_layer:
            gc.collect()
