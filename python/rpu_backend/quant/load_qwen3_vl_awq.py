"""Exact Qwen3-VL Text-W4 G32 checkpoint loading, without HF quantizer hooks.

The compressed-tensors source uses offset nibbles; the runtime uses its existing
pgrp ABI. Floating parameters come from the same AWQ asset, including smoothed
norms. This module does not quantize weights or configure runtime execution.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from transformers import AutoModelForImageTextToText

from rpu_backend.quant.int4_pgrp_pack import (
    swizzle_int4_pgrp_scale,
    swizzle_pack_int4_pgrp,
)

_ROLES = (("self_attn.q_proj", 1), ("self_attn.k_proj", 1),
          ("self_attn.v_proj", 1), ("self_attn.o_proj", 0),
          ("mlp.gate_proj", 1), ("mlp.up_proj", 1), ("mlp.down_proj", 0))
_PLAN_ATTR = "_rpu_qwen3_vl_awq_load_plan"


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def is_qwen3_vl_awq_config(config, *, allow_padded_8b=False) -> bool:
    """Raw Text-AWQ admits 2B/4B; the converter alone may admit exact 8B sources."""
    try:
        text, vision = _get(config, "text_config"), _get(config, "vision_config")
        h = _get(text, "hidden_size")
        profiles = {2048: (6144, 28, 16, True), 2560: (9728, 36, 32, False)}
        if allow_padded_8b is True:
            profiles[4096] = (12288, 36, 32, False)
        expected = profiles.get(h)
        if expected is None:
            return False
        inter, layers, heads, tied = expected
        padded = h == 4096
        vh, vi, depth, taps = ((1152, 4304, 27, (8, 16, 24)) if padded
                               else (1024, 4096, 24, (5, 11, 17)))
        text_values = tuple(_get(text, k) for k in (
            "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "vocab_size"))
        if (any(type(x) is not int for x in (h, *text_values)) or
                text_values != (inter, layers, heads, 8, 128, 151936)):
            return False
        if (tuple(_get(config, "architectures", ())) != ("Qwen3VLForConditionalGeneration",)
                or _get(config, "model_type") != "qwen3_vl"
                or _get(text, "model_type") != "qwen3_vl_text"
                or _get(config, "tie_word_embeddings") is not tied
                or (_get(text, "tie_word_embeddings") is not None if padded else
                    _get(text, "tie_word_embeddings") is not True)
                or _get(text, "attention_bias") is not False
                or _get(text, "attention_dropout", 0.0) != 0.0
                or _get(text, "hidden_act") != "silu"
                or _get(text, "rms_norm_eps") != 1e-6
                or _get(text, "max_position_embeddings") != 262144
                or _get(text, "use_cache") is not True):
            return False
        rope = _get(text, "rope_parameters") or _get(text, "rope_scaling") or {}
        if (rope.get("rope_type", "default") != "default"
                or rope.get("mrope_interleaved") is not True
                or tuple(rope.get("mrope_section", ())) != (24, 20, 20)
                or rope.get("rope_theta", _get(text, "rope_theta")) != 5000000):
            return False
        if tuple(_get(vision, k) for k in (
                "model_type", "hidden_size", "intermediate_size", "depth", "num_heads",
                "patch_size", "temporal_patch_size", "spatial_merge_size", "out_hidden_size",
                "hidden_act", "in_channels", "num_position_embeddings")) != (
                "qwen3_vl", vh, vi, depth, 16, 16, 2, 2, h,
                "gelu_pytorch_tanh", 3, 2304):
            return False
        if tuple(_get(vision, "deepstack_visual_indexes", ())) != taps:
            return False
        if tuple(_get(config, k) for k in (
                "image_token_id", "video_token_id", "vision_start_token_id",
                "vision_end_token_id")) != (151655, 151656, 151652, 151653):
            return False
        if any(_get(owner, "quant_config") is not None for owner in (config, text, vision)):
            return False
        if any(_get(owner, "quantization_config") is not None for owner in (text, vision)):
            return False
        qc = _get(config, "quantization_config")
        if (not isinstance(qc, dict) or qc.get("quant_method") != "compressed-tensors"
                or qc.get("format") != "pack-quantized"
                or qc.get("quantization_status") != "compressed"
                or qc.get("kv_cache_scheme") is not None
                or qc.get("sparsity_config") not in (None, {})
                or qc.get("transform_config") not in (None, {})):
            return False
        groups = qc.get("config_groups")
        if not isinstance(groups, dict) or set(groups) != {"group_0"}:
            return False
        group = groups["group_0"]
        w = group.get("weights")
        if (group.get("format") != "pack-quantized" or group.get("targets") != ["Linear"]
                or group.get("input_activations") is not None
                or group.get("output_activations") is not None
                or not isinstance(w, dict)
                or type(w.get("num_bits")) is not int or w.get("num_bits") != 4
                or type(w.get("group_size")) is not int or w.get("group_size") != 32
                or w.get("symmetric") is not True or w.get("dynamic") is not False
                or w.get("strategy") != "group" or w.get("type") != "int"
                or w.get("actorder") is not None or w.get("block_structure") is not None
                or set(w) - {"num_bits", "group_size", "symmetric", "dynamic",
                             "strategy", "type", "actorder", "block_structure",
                             "observer", "observer_kwargs"}):
            return False
        ignored = {f"model.visual.blocks.{i}.{role}" for i in range(depth) for role in
                   ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2")}
        ignored |= {f"model.visual.{owner}.linear_fc{fc}" for owner in
                    ("merger", *(f"deepstack_merger_list.{i}" for i in range(3))) for fc in (1, 2)}
        ignored.add("lm_head")
        names = qc.get("ignore")
        return isinstance(names, list) and len(names) == len(ignored) and set(names) == ignored
    except (AttributeError, TypeError, ValueError):
        return False


def _pack_projection(packed, shape, scale, *, n, k, partition):
    """Source offset nibble -> existing physical pgrp; never requantize."""
    if packed.dtype != torch.int32 or tuple(packed.shape) != (n, k // 8):
        raise ValueError("AWQ weight_packed must be INT32[N,K/8]")
    if shape.dtype != torch.int64 or tuple(shape.shape) != (2,) or shape.tolist() != [n, k]:
        raise ValueError("AWQ weight_shape must be INT64 [N,K]")
    if scale.dtype not in (torch.float16, torch.bfloat16) or tuple(scale.shape) != (n, k // 32):
        raise ValueError("AWQ weight_scale must be BF16/FP16[N,K/32]")
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError("AWQ scales must be finite and positive")
    scale_half = scale.half()
    if not bool(torch.isfinite(scale_half).all()) or not bool((scale_half > 0).all()):
        raise ValueError("AWQ scales are not representable as positive FP16")
    shifts = torch.arange(8, dtype=torch.int32) * 4
    values = ((packed.unsqueeze(-1) >> shifts) & 15).sub_(8).to(torch.int8).reshape(n, k)
    weight = swizzle_pack_int4_pgrp(values, partition, 8).reshape(n, k // 2)
    scales = swizzle_int4_pgrp_scale(scale_half.t().contiguous(), 32, partition, 8)
    return weight, scales


def _config_signature(config):
    return json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"), default=str)


def _inventory(model):
    tensors = list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False))
    return tuple((name, id(t), t._version, t.data_ptr(), tuple(t.shape), tuple(t.stride()),
                  str(t.dtype), str(t.device)) for name, t in tensors)


def _projection_inventory(model):
    result = []
    for i, layer in enumerate(model.model.language_model.layers):
        for role, partition in _ROLES:
            module = layer.get_submodule(role)
            if not isinstance(module, nn.Linear) or module.bias is not None:
                raise ValueError(f"AWQ Text7 requires bias-free Linear: layer{i}.{role}")
            result.append((f"model.language_model.layers.{i}.{role}", module, partition))
    return result


@dataclass(frozen=True)
class _AWQLoadPlan:
    model_id: int
    config_signature: str
    tensor_inventory: tuple
    projections: tuple


def _validate_qwen3_vl_awq_model(model) -> bool:
    """Validate the original CPU owner before adapter construction and claim."""
    plan = vars(model).get(_PLAN_ATTR)
    selected = is_qwen3_vl_awq_config(getattr(model, "config", None))
    if plan is None and not selected:
        return False
    if not selected or not isinstance(plan, _AWQLoadPlan) or plan.model_id != id(model):
        raise RuntimeError("Qwen3-VL AWQ requires its original CPU load plan and exact configuration")
    if _config_signature(model.config) != plan.config_signature or _inventory(model) != plan.tensor_inventory:
        raise RuntimeError("Qwen3-VL AWQ CPU configuration or tensor inventory changed after loading")
    if any(t.device.type != "cpu" for t in (*model.parameters(), *model.buffers())):
        raise RuntimeError("Qwen3-VL AWQ load plan requires CPU tensors before installation")
    actual = tuple((name, id(module), module.in_features, module.out_features,
                    getattr(module, "_rpu_linear_partition", None),
                    getattr(module, "_rpu_linear_num_cores", None))
                   for name, module, _ in _projection_inventory(model))
    if actual != plan.projections:
        raise RuntimeError("Qwen3-VL AWQ packed projection layout changed after loading")
    return True


def _checked_float(name, tensor):
    if tensor.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"AWQ floating tensor {name} requires BF16/FP16 source")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"AWQ floating tensor {name} is non-finite")
    half = tensor.to(dtype=torch.float16, copy=True).contiguous()
    if not bool(torch.isfinite(half).all()):
        raise ValueError(f"AWQ floating tensor {name} overflows FP16")
    return half


# Cold owners need version counters even when callers load inside inference_mode.
@torch.inference_mode(False)
def load_qwen3_vl_awq_imagetext(config, ckpt_dir: str, *, dtype=torch.float16, **resolve_kwargs):
    """Load both exact Text-W4 profiles on CPU, preserving actual AWQ constants."""
    from rpu_backend.quant.load import (
        _resolve_checkpoint_dir, _safetensor_shards, _install_parameter,
        _materialize_imagetext_meta_buffers, _raise_on_imagetext_meta,
    )
    if dtype is not torch.float16:
        raise TypeError("Qwen3-VL AWQ requires torch.float16 activations")
    if not is_qwen3_vl_awq_config(config):
        raise ValueError("Qwen3-VL AWQ requires exact2B/4B symmetric Text-W4 G32 TP8 profile")
    directory = _resolve_checkpoint_dir(ckpt_dir, **resolve_kwargs)
    source_config = json.loads((directory / "config.json").read_text())
    if (not is_qwen3_vl_awq_config(source_config)
            or source_config["text_config"]["hidden_size"] != config.text_config.hidden_size
            or source_config["quantization_config"] != config.quantization_config):
        raise ValueError("AWQ checkpoint configuration differs from the selected profile")
    skeleton_config = copy.deepcopy(config)
    del skeleton_config.quantization_config
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(skeleton_config)
    model.config.quantization_config = copy.deepcopy(config.quantization_config)
    projections = _projection_inventory(model)
    if len(projections) != model.config.text_config.num_hidden_layers * 7:
        raise ValueError("AWQ requires all seven projections of every Text layer")
    shapes = {name: tuple(t.shape) for name, t in model.named_parameters(remove_duplicate=False)}
    if model.config.tie_word_embeddings:
        shapes.pop("lm_head.weight")
    expected = dict(shapes)
    for name, module, partition in projections:
        n, k = module.out_features, module.in_features
        if (k % 32 or (partition == 1 and (n % 128 or k % 64))
                or (partition == 0 and (n % 16 or k % 512))):
            raise ValueError(f"AWQ projection is outside TP8 pgrp geometry: {name}")
        del expected[name + ".weight"]
        expected.update({name + ".weight_packed": (n, k // 8),
                         name + ".weight_shape": (2,), name + ".weight_scale": (n, k // 32)})
    tensor_shards = {}
    for shard in _safetensor_shards(directory):
        with safe_open(shard, framework="pt") as handle:
            for name in handle.keys():
                if name in tensor_shards or name not in expected:
                    raise ValueError(f"Unexpected or duplicate AWQ tensor: {name}")
                view = handle.get_slice(name)
                suffix = name.rsplit(".", 1)[-1]
                allowed = {"weight_packed": {"I32"}, "weight_shape": {"I64"}}.get(suffix, {"F16", "BF16"})
                if tuple(view.get_shape()) != expected[name] or view.get_dtype() not in allowed:
                    raise ValueError(f"Invalid AWQ tensor metadata: {name}")
                tensor_shards[name] = str(shard)
    if set(tensor_shards) != set(expected):
        raise ValueError(f"AWQ checkpoint tensor inventory differs: missing {sorted(set(expected)-set(tensor_shards))[:8]}")
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        declared = json.loads(index.read_text())["weight_map"]
        if set(declared) != set(tensor_shards) or any(
                Path(tensor_shards[n]).resolve() != (directory / declared[n]).resolve() for n in declared):
            raise ValueError("AWQ shard index differs from actual tensor locations")
    # Open each shard once; tensors are copied/packed before these mappings close.
    from contextlib import ExitStack
    with ExitStack() as stack:
        handles = {s: stack.enter_context(safe_open(s, framework="pt")) for s in set(tensor_shards.values())}
        def tensor(name):
            return handles[tensor_shards[name]].get_tensor(name)
        for name, module, partition in projections:
            weight, scale = _pack_projection(
                tensor(name + ".weight_packed"), tensor(name + ".weight_shape"), tensor(name + ".weight_scale"),
                n=module.out_features, k=module.in_features, partition=partition)
            module.weight = nn.Parameter(weight, requires_grad=False)
            module.register_buffer("weight_scale", scale, persistent=False)
            module._rpu_linear_partition = partition
            module._rpu_linear_num_cores = 8
        packed_names = {name + ".weight" for name, _, _ in projections}
        for name in sorted(set(shapes) - packed_names):
            _install_parameter(model, name, _checked_float(name, tensor(name)))
    if model.config.tie_word_embeddings:
        model.lm_head.weight = model.model.language_model.embed_tokens.weight
    _materialize_imagetext_meta_buffers(model)
    _raise_on_imagetext_meta(model)
    model.requires_grad_(False).eval()
    plan = _AWQLoadPlan(id(model), _config_signature(model.config), _inventory(model),
        tuple((name, id(module), module.in_features, module.out_features, partition, 8)
              for name, module, partition in projections))
    setattr(model, _PLAN_ATTR, plan)
    _validate_qwen3_vl_awq_model(model)
    return model


def _allocate_awq_scale_views(model):
    """One FP16 scale owner; native retention still checks device alignment.

    Relative 4096B offsets preserve the pgrp controller alignment when the
    bank's actual device base is aligned. The allocator only promises 32B;
    never infer device alignment from a host pointer or bypass native retain.
    """
    layout, size = [], 0
    for name, module, _ in _projection_inventory(model):
        scale = module.weight_scale
        if (scale.device.type != "cpu" or scale.dtype != torch.float16
                or scale.dim() != 2 or scale.size(0) != 32
                or not scale.is_contiguous()):
            raise ValueError(f"AWQ scale bank requires contiguous CPU FP16[32,*]: {name}")
        offset = (size + 4095) // 4096 * 4096
        layout.append((name, scale, offset // 2))
        size = offset + scale.numel() * 2
    storage = torch.empty(size // 2, dtype=torch.float16, device="rpu")
    views = {}
    for name, scale, offset in layout:
        target = storage.narrow(0, offset, scale.numel()).view(scale.shape)
        target.copy_(scale)
        views[name] = target
    return views


@torch.inference_mode(False)
def _move_materialized_awq_decoder_for_rpu(model) -> None:
    """Copy already-packed Text7 once; native scale retention owns alignment."""
    from rpu_backend.quant.load import _allocate_decoder_projection_views
    if vars(model).get("_rpu_swizzle_started") is not True:
        raise RuntimeError("AWQ decoder migration requires adapter ownership/poison first")
    if not _validate_qwen3_vl_awq_model(model):
        raise ValueError("AWQ decoder migration requires the exact loaded model")
    names = tuple(tuple(f"model.language_model.layers.{i}.{role}.weight" for role, _ in _ROLES)
                  for i in range(len(model.model.language_model.layers)))
    views = _allocate_decoder_projection_views(model, names, dtype=torch.uint8)
    # Only the 4B path changes storage. Keep the existing 2B
    # scale migration unchanged, including native set_weights retention.
    scale_views = (_allocate_awq_scale_views(model)
                   if model.config.text_config.hidden_size == 2560 else None)
    for i, layer in enumerate(model.model.language_model.layers):
        for role, _ in _ROLES:
            module = layer.get_submodule(role)
            target = views[f"model.language_model.layers.{i}.{role}.weight"]
            target.copy_(module.weight.detach())
            module.weight = nn.Parameter(target, requires_grad=False)
            if scale_views is not None:
                module.weight_scale = scale_views[f"model.language_model.layers.{i}.{role}"]
        # Bank views are already RPU buffers; only norms still move on 4B.
        # Native set_weights retains an aligned cold copy whenever a scale's
        # actual device address is not already 4096B-aligned.
        layer.to("rpu")
