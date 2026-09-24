"""Bounded installation of the exact plain Qwen3-32B quantized profiles.

W4 staging reads metadata only. The adapter owns the live-model claim and
poison marker before consuming that plan; no complete FP16 model is loaded.
W8 retains the existing safetensors-backed CPU loader and banks one layer at
a time. These helpers do not admit another model, recipe or execution profile.
"""
from __future__ import annotations

import copy
from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from .convert_qwen3_w4a16 import _PROJECTIONS, pack_linear_int4_
from .int4_pgrp_pack import quantize_pack_int4_pgrp_col_bounded
from .load import (
    _allocate_decoder_projection_views, _install_parameter,
    _materialize_meta_buffers, _move_materialized_decoder_for_rpu,
    _resolve_checkpoint_dir, _safetensor_shards,
)
from .qwen3_profiles import (
    is_qwen3_32b_w8a16_config, qwen3_dense_quant_profile,
)


_PLAN_ATTR = "_qwen3_w4a16_load_plan"
_RECIPES = frozenset({"w4a16", "w4a16_lm_head"})


def _source_profile(config, quantization):
    if quantization not in _RECIPES:
        raise ValueError("Qwen3 staged W4 requires w4a16 or w4a16_lm_head")
    profile = qwen3_dense_quant_profile(config)
    if profile.num_layers != 64:
        raise ValueError("Qwen3 staged W4 requires the exact 32B dense source")
    return profile


def _config_signature(config):
    values = config.to_dict()
    # HF supplies these loader provenance fields independently of semantics.
    for name in ("_name_or_path", "_commit_hash", "transformers_version"):
        values.pop(name, None)
    return json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)


def _tensor_inventory(model):
    return tuple((name, id(value), value._version, tuple(value.shape),
                  tuple(value.stride()), str(value.dtype), str(value.device))
                 for name, value in (
                     *model.named_parameters(remove_duplicate=False),
                     *model.named_buffers(remove_duplicate=False)))


def _projection_inventory(model):
    result = []
    for index, layer in enumerate(model.model.layers):
        for role, partition in _PROJECTIONS:
            module = layer.get_submodule(role)
            if not isinstance(module, nn.Linear) or module.bias is not None:
                raise ValueError(f"Qwen3 staged W4 requires bias-free Text7: {index}.{role}")
            result.append((f"model.layers.{index}.{role}", id(module),
                           module.in_features, module.out_features, partition,
                           getattr(module, "_rpu_linear_partition", None),
                           getattr(module, "_rpu_linear_num_cores", None)))
    head = model.lm_head
    if not isinstance(head, nn.Linear) or head.bias is not None:
        raise ValueError("Qwen3 staged W4 requires a bias-free independent lm_head")
    result.append(("lm_head", id(head), head.in_features, head.out_features, 1,
                   getattr(head, "_rpu_linear_partition", None),
                   getattr(head, "_rpu_linear_num_cores", None)))
    return tuple(result)


def _file_identity(path):
    stat = Path(path).stat()
    return (str(Path(path).resolve()), stat.st_dev, stat.st_ino,
            stat.st_size, stat.st_mtime_ns)


@dataclass(frozen=True)
class _Qwen3W4LoadPlan:
    model_id: int
    quantization: str
    config_signature: str
    tensors: tuple
    projections: tuple
    tensor_shards: tuple[tuple[str, str], ...]
    layer_tensor_names: tuple[tuple[str, ...], ...]
    nondecoder_tensor_names: tuple[str, ...]
    files: tuple


@torch.inference_mode(False)
def stage_qwen3_w4a16_for_rpu(
    config, ckpt_dir: str, *, quantization: str,
    dtype: torch.dtype = torch.float16, **resolve_kwargs,
):
    """Return an unmaterialized, identity-bound 32B source for one-step RPU load."""
    _source_profile(config, quantization)
    if dtype is not torch.float16:
        raise TypeError("Qwen3 staged W4 requires FP16 activations")
    directory = _resolve_checkpoint_dir(ckpt_dir, **resolve_kwargs)
    source = AutoConfig.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False)
    _source_profile(source, quantization)
    if _config_signature(source) != _config_signature(config):
        raise ValueError("Qwen3 staged source config differs from the selected config")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(copy.deepcopy(config))
    shapes = {name: tuple(value.shape)
              for name, value in model.named_parameters(remove_duplicate=False)}
    projections = _projection_inventory(model)
    if len(projections) != config.num_hidden_layers * 7 + 1:
        raise ValueError("Qwen3 staged source requires every Text7 projection and independent head")
    tensor_shards = {}
    shards = _safetensor_shards(directory)
    for shard in shards:
        with safe_open(shard, framework="pt") as handle:
            for name in handle.keys():
                if name in tensor_shards or name not in shapes:
                    raise ValueError(f"Unexpected or duplicate Qwen3 source tensor: {name}")
                view = handle.get_slice(name)
                if tuple(view.get_shape()) != shapes[name] or view.get_dtype() not in {"F16", "BF16"}:
                    raise ValueError(f"Invalid Qwen3 source tensor metadata: {name}")
                tensor_shards[name] = str(shard)
    if set(tensor_shards) != set(shapes):
        raise ValueError(f"Qwen3 source tensor inventory differs: missing {sorted(set(shapes)-set(tensor_shards))[:8]}")
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        declared = json.loads(index.read_text())["weight_map"]
        if set(declared) != set(tensor_shards) or any(
                Path(tensor_shards[name]).resolve() != (directory / declared[name]).resolve()
                for name in declared):
            raise ValueError("Qwen3 source shard index differs from actual tensor locations")
    layer_names = tuple(tuple(sorted(name for name in shapes
        if name.startswith(f"model.layers.{index}.")))
        for index in range(len(model.model.layers)))
    decoder_names = {name for names in layer_names for name in names}
    files = [directory / "config.json", *shards]
    if index.is_file():
        files.append(index)
    plan = _Qwen3W4LoadPlan(
        id(model), quantization, _config_signature(model.config),
        _tensor_inventory(model), projections, tuple(sorted(tensor_shards.items())),
        layer_names, tuple(sorted(set(shapes) - decoder_names)),
        tuple(_file_identity(path) for path in files))
    setattr(model, _PLAN_ATTR, plan)
    validate_staged_qwen3_w4a16(model, quantization=quantization)
    return model


def validate_staged_qwen3_w4a16(model, *, quantization: str) -> bool:
    """Validate the original meta owner and unchanged source before any claim."""
    _source_profile(model.config, quantization)
    plan = vars(model).get(_PLAN_ATTR)
    if not isinstance(plan, _Qwen3W4LoadPlan) or plan.model_id != id(model):
        raise RuntimeError("Qwen3 staged W4 requires its original source load plan")
    if (plan.quantization != quantization
            or _config_signature(model.config) != plan.config_signature
            or _tensor_inventory(model) != plan.tensors
            or _projection_inventory(model) != plan.projections):
        raise RuntimeError("Qwen3 staged W4 config, recipe or tensor/module inventory changed")
    if (any(not value.is_meta for value in model.parameters())
            or any(value.device.type not in {"cpu", "meta"} for value in model.buffers())):
        raise RuntimeError("Qwen3 staged W4 was materialized before adapter ownership")
    if any(_file_identity(identity[0]) != identity for identity in plan.files):
        raise RuntimeError("Qwen3 staged W4 checkpoint changed after metadata validation")
    return True


def _move_packed_layer(model, index):
    """One weight bank and one scale bank; retain native device-alignment checks."""
    layer = model.model.layers[index]
    names = tuple(f"model.layers.{index}.{role}.weight" for role, _ in _PROJECTIONS)
    weights = _allocate_decoder_projection_views(model, (names,), dtype=torch.uint8)
    layout, size = [], 0
    for role, _ in _PROJECTIONS:
        scale = layer.get_submodule(role).weight_scale
        offset = (size + 4095) // 4096 * 4096
        layout.append((role, offset // 2, tuple(scale.shape), scale.numel()))
        size = offset + scale.numel() * 2
    scales = torch.empty(size // 2, dtype=torch.float16, device="rpu")
    for role, offset, shape, count in layout:
        module = layer.get_submodule(role)
        target = weights[f"model.layers.{index}.{role}.weight"]
        target.copy_(module.weight.detach())
        module.weight = nn.Parameter(target, requires_grad=False)
        scale = scales.narrow(0, offset, count).view(shape)
        scale.copy_(module.weight_scale)
        module.weight_scale = scale
    layer.to("rpu")


@torch.inference_mode(False)
def materialize_staged_qwen3_w4a16_for_rpu(model, *, quantization: str) -> None:
    """Consume the plan into final RPU layouts after adapter claim and poison."""
    from rpu_backend.runtime.weights import (
        _release_cpu_weight_pages, transform_linear_weight,
    )
    if vars(model).get("_rpu_swizzle_started") is not True:
        raise RuntimeError("Qwen3 staged W4 migration requires adapter ownership/poison first")
    validate_staged_qwen3_w4a16(model, quantization=quantization)
    plan = vars(model)[_PLAN_ATTR]
    locations = dict(plan.tensor_shards)
    with ExitStack() as stack:
        handles = {path: stack.enter_context(safe_open(path, framework="pt"))
                   for path in set(locations.values())}

        def install(name):
            value = handles[locations[name]].get_tensor(name).to(torch.float16)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Non-finite Qwen3 source tensor: {name}")
            _install_parameter(model, name, value)

        for index, names in enumerate(plan.layer_tensor_names):
            for name in names:
                install(name)
            layer = model.model.layers[index]
            for role, partition in _PROJECTIONS:
                projection = layer.get_submodule(role)
                pack_linear_int4_(projection, partition, name=f"{index}.{role}", group_size=32)
                projection._rpu_linear_partition = partition
                projection._rpu_linear_num_cores = 8
            _move_packed_layer(model, index)
            _release_cpu_weight_pages()
        for name in plan.nondecoder_tensor_names:
            install(name)
            if name == "lm_head.weight":
                head = model.lm_head
                if quantization == "w4a16_lm_head":
                    weight, scale = quantize_pack_int4_pgrp_col_bounded(head.weight.detach())
                    head.weight = nn.Parameter(weight, requires_grad=False)
                    head.register_buffer("weight_scale", scale, persistent=False)
                    del weight, scale
                else:
                    head.weight = nn.Parameter(
                        transform_linear_weight(head.weight.detach(), 1), requires_grad=False)
                head._rpu_linear_partition = 1
                head._rpu_linear_num_cores = 8
                head.to("rpu")
            else:
                module_name, parameter_name = name.rsplit(".", 1)
                module = model.get_submodule(module_name)
                module._parameters[parameter_name] = nn.Parameter(
                    module._parameters[parameter_name].detach().to("rpu"), requires_grad=False)
            _release_cpu_weight_pages()
    _materialize_meta_buffers(model)
    if any(value.is_meta for value in (*model.parameters(), *model.buffers())):
        raise RuntimeError("Qwen3 staged W4 left unresolved meta tensors")
    # Success consumes the plan; a failure keeps it and the adapter's poison.
    vars(model).pop(_PLAN_ATTR)
    model.requires_grad_(False).eval()


def move_qwen3_w8a16_decoder_for_rpu(model) -> None:
    """Migrate the exact mmap-loaded W8 decoder into one weight bank per layer."""
    if not is_qwen3_32b_w8a16_config(model.config):
        raise ValueError("Qwen3 W8 bank migration requires the exact 32B decoder/head checkpoint")
    projections = [layer.get_submodule(role) for layer in model.model.layers
                   for role, _ in _PROJECTIONS]
    for module in (*projections, model.lm_head):
        weight, scale = module.weight, getattr(module, "weight_scale", None)
        if (weight.device.type != "cpu" or weight.dtype != torch.int8
                or tuple(weight.shape) != (module.out_features, module.in_features)
                or not weight.is_contiguous()
                or not isinstance(scale, torch.Tensor) or scale.device.type != "cpu"
                or scale.dtype != torch.float16 or tuple(scale.shape) != (module.out_features,)
                or not scale.is_contiguous()
                or not bool(torch.isfinite(scale).all() and (scale > 0).all())
                or getattr(module, "_rpu_linear_partition", None) is not None):
            raise ValueError("Qwen3 32B W8 requires unswizzled CPU INT8 projections/head and positive FP16 scales")
    _move_materialized_decoder_for_rpu(
        model, dtype=torch.int8, per_layer=True,
        decoder=model.model, decoder_prefix="model")
