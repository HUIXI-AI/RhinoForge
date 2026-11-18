"""Cold, explicit W8A16 preparation for the exact G0.5 K1 experiment.

Activations, normalization, GDN scalar projections, action I/O and host AdaLN
keep their existing precision. This is not a checkpoint-format certification.
"""
from __future__ import annotations

import torch


def normalize_weight_dtype(value):
    if value not in ("fp16", "w8a16"):
        raise ValueError("G0.5 weight_dtype must be 'fp16' or 'w8a16'")
    return value


def check_installed_weight_dtype(owner, requested):
    requested = normalize_weight_dtype(requested)
    if getattr(owner, "_rpu_g05_weight_dtype", "fp16") != requested:
        raise ValueError(
            "G0.5 weight_dtype conflicts with the installed runtime; "
            "reload the checkpoint to change precision")


def normalize_linear_accumulation(value):
    if not isinstance(value, str) or value not in ("acc32", "acc16"):
        raise ValueError("G0.5 linear_accumulation must be 'acc32' or 'acc16'")
    return value


def validate_linear_accumulation(value, *, weight_dtype, exact_k1):
    value = normalize_linear_accumulation(value)
    normalize_weight_dtype(weight_dtype)
    if value == "acc16" and not exact_k1:
        raise ValueError("G0.5 ACC16 is limited to the exact K1 FP16/W8A16 experiment")
    return value


def check_installed_linear_accumulation(owner, requested):
    requested = normalize_linear_accumulation(requested)
    if getattr(owner, "_rpu_g05_linear_accumulation", "acc32") != requested:
        raise ValueError(
            "G0.5 linear_accumulation conflicts with the installed runtime; "
            "reload the checkpoint to change precision")


def normalize_math_precision(value):
    from rpu_backend.adapters.qwen3_5._g05_cold import normalize_math_precision as check

    return check(value)


def validate_math_precision(value, *, exact_k1):
    value = normalize_math_precision(value)
    if value == "high" and not exact_k1:
        raise ValueError("G0.5 HIGH math requires the exact K1 profile")
    return value


def check_installed_math_precision(owner, requested):
    requested = normalize_math_precision(requested)
    if getattr(owner, "_rpu_g05_math_precision", "base") != requested:
        raise ValueError(
            "G0.5 math_precision conflicts with the installed runtime; "
            "reload the checkpoint to change precision")


def require_g05_high_math_native(*, vision=False):
    from rpu_backend.adapters.qwen3_5._g05_cold import require_g05_high_math_native as check

    return check(vision=vision)


def validate_g05_high_math_install(config, *, cpu_stage_weights):
    from rpu_backend.adapters.qwen3_5._g05_cold import validate_g05_high_math_install as check

    return check(config, cpu_stage_weights=cpu_stage_weights)


def linear_execution_config(config, mode, *, stage="prefill"):
    """Bind a private shared installer view without changing the caller's plan."""
    import copy

    # Normalized component configs are immutable Mapping objects whose
    # deepcopy deliberately returns themselves. Only these two levels need
    # mutation; preserve the caller's frozen plan and all unrelated options.
    result = copy.deepcopy(dict(config or {}))
    choices = dict(result.get(stage, {}))
    result[stage] = choices
    enabled = normalize_linear_accumulation(mode) == "acc32"
    if "linear_acc32" in choices and choices["linear_acc32"] is not enabled:
        raise ValueError("G0.5 execution config conflicts with linear_accumulation")
    choices["linear_acc32"] = enabled
    return result


def g05_w8_quant_config(*, action=False):
    from rpu_backend.adapters.qwen3_5._g05_cold import g05_w8_quant_config as check

    return check(action=action)


def validate_g05_w8_install(config, *, cpu_stage_weights):
    from rpu_backend.adapters.qwen3_5._g05_cold import validate_g05_w8_install as check

    return check(config, cpu_stage_weights=cpu_stage_weights)


def quantize_g05_text_weights(inner, *, action=False):
    """Quantize CPU FP16 sources once, before the shared irreversible swizzle."""
    from rpu_backend.adapters.qwen3_5.quant_scope import (
        is_full_from_layer_types, resolve_quant_specs,
    )
    from rpu_backend.quant._common import quantize_linear_per_channel

    if getattr(inner, "_rpu_g05_quantization_started", False):
        raise RuntimeError("G0.5 quantization already started; reload the checkpoint")
    if getattr(inner.config, "quant_config", None):
        raise ValueError("G0.5 W8A16 preparation requires an unquantized checkpoint")
    config = g05_w8_quant_config(action=action)
    specs = resolve_quant_specs(
        is_full_from_layer_types(inner.config.layer_types), config)
    projections = []
    # Validate all owners before replacing the first parameter.
    for name, spec in specs.items():
        linear = inner.get_submodule(name)
        weight = linear.weight
        if (weight.device.type != "cpu" or weight.dtype != torch.float16
                or weight.ndim != 2 or not weight.is_contiguous()
                or not bool(torch.isfinite(weight).all())
                or getattr(linear, "weight_scale", None) is not None):
            raise ValueError(f"G0.5 quantization needs finite, unquantized CPU FP16 {name}")
        projections.append((name, spec, linear))
    inner._rpu_g05_quantization_started = True
    inventory = []
    for name, spec, linear in projections:
        if spec.method == "w8a16":
            weight, scale = quantize_linear_per_channel(linear.weight, bits=8)
            linear.weight = torch.nn.Parameter(weight.contiguous(), requires_grad=False)
            linear.register_buffer("weight_scale", scale.contiguous())
        weight, scale = linear.weight, getattr(linear, "weight_scale", None)
        inventory.append({
            "name": name, "method": spec.method,
            "weight_shape": list(weight.shape), "weight_dtype": str(weight.dtype),
            "weight_bytes": weight.numel() * weight.element_size(),
            "scale_shape": list(scale.shape) if scale is not None else [],
            "scale_dtype": str(scale.dtype) if scale is not None else None,
            "scale_bytes": scale.numel() * scale.element_size() if scale is not None else 0,
        })
    inner.config.quant_config = config
    inner._rpu_g05_quant_inventory = inventory
    return inventory



def normalize_action_weight_dtype(value, *, weight_dtype):
    """Resolve the Action override without broadening Vision/Text precision."""
    base = normalize_weight_dtype(weight_dtype)
    value = base if value is None else value
    if value not in (base, "nvfp4a16"):
        raise ValueError("G0.5 action_weight_dtype must inherit weight_dtype or be 'nvfp4a16'")
    if value == "nvfp4a16" and base != "w8a16":
        raise ValueError("G0.5 Action NVFP4 requires Vision/Text weight_dtype='w8a16'")
    return value


def check_installed_action_weight_dtype(owner, requested, *, weight_dtype):
    effective = normalize_action_weight_dtype(requested, weight_dtype=weight_dtype)
    installed = getattr(owner, "_rpu_g05_action_weight_dtype",
                        getattr(owner, "_rpu_g05_weight_dtype", "fp16"))
    if installed != effective:
        raise ValueError("G0.5 action_weight_dtype conflicts with the installed runtime; "
                         "reload the checkpoint to change precision")


def require_g05_action_nvfp4_native():
    from rpu_backend.adapters.qwen3_5._g05_cold import require_g05_action_nvfp4_native as check

    return check()


def prepare_g05_action_nvfp4(expert, lanes=8):
    """Pack five families directly from original CPU FP16, before W8 staging.

    The shared installer retains W8 K/V/attention gate; this private payload
    replaces only finalized Q/O/MLP lists, without changing public Qwen scope.
    ``lanes`` is the expert's execution width (8, or 4 at a reduced budget);
    the logical tensor shapes do not depend on it.
    """
    if lanes not in (4, 8):
        raise ValueError("G0.5 Action NVFP4 packs for four or eight lanes")
    from rpu_backend.adapters.g05.action import _check_g05_action_profile
    from rpu_backend.adapters.qwen3_5.text import split_q_gate
    from rpu_backend.quant.nvfp4_pack import quantize_nvfp4, swizzle_pack_nvfp4_v2

    _check_g05_action_profile(expert.config)
    require_g05_action_nvfp4_native()
    if (getattr(expert, "_rpu_g05_quantization_started", False)
            or getattr(expert.config, "quant_config", None)):
        raise ValueError("G0.5 Action NVFP4 requires original unquantized CPU FP16 weights")
    families = ("q", "o", "gate", "up", "down")
    shapes = ((4096, 1024), (1024, 2048), (4096, 1024),
              (4096, 1024), (1024, 4096))
    sources = []
    for layer in expert.layers:
        modules = (layer.self_attn.q_proj, layer.self_attn.o_proj,
                   layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj)
        for name, module, shape in zip(families, modules, shapes):
            w = module.weight
            if (w.device.type != "cpu" or w.dtype != torch.float16
                    or tuple(w.shape) != shape or not w.is_contiguous()
                    or not bool(torch.isfinite(w).all())
                    or getattr(module, "weight_scale", None) is not None):
                raise ValueError(f"G0.5 Action NVFP4 needs finite original CPU FP16 {name} {shape}")
        sources.append(modules)
    if len(sources) != 24:
        raise ValueError("G0.5 Action NVFP4 requires all 24 layers")
    weights, scales, tensor_scales = ([[] for _ in families] for _ in range(3))
    for modules in sources:
        for index, module in enumerate(modules):
            weight = module.weight.detach()
            if index == 0:
                weight, _ = split_q_gate(weight, 8, 256)
            codes, block_scale, tensor_scale = quantize_nvfp4(weight)
            packed, striped = swizzle_pack_nvfp4_v2(
                codes, block_scale, 0 if index in (1, 4) else 1, lanes)
            weights[index].append(packed)
            scales[index].append(striped)
            tensor_scales[index].append(tensor_scale.reshape(()))
    payload = {"abi": "striped_v2", "block_size": 16,
               "weights": weights, "scales": scales,
               "tensor_scales": [torch.stack(values).contiguous()
                                 for values in tensor_scales]}
    validate_g05_action_nvfp4_payload(payload)
    return payload


def validate_g05_action_nvfp4_payload(payload):
    from rpu_backend.adapters.qwen3_5._g05_cold import validate_g05_action_nvfp4_payload as check

    return check(payload)


def g05_action_nvfp4_inventory(payload, w8_inventory):
    """Describe installed packed owners, replacing temporary W8 staging rows."""
    family_paths = ("self_attn.q_proj.query", "self_attn.o_proj", "mlp.gate_proj",
                    "mlp.up_proj", "mlp.down_proj")
    replaced = {name.removesuffix(".query") for name in family_paths}
    result = []
    for item in w8_inventory:
        role = item["name"].split(".", 2)[2]
        if role not in replaced:
            result.append(dict(item))
        elif role == "self_attn.q_proj":
            gate = dict(item, name=item["name"] + ".attention_gate")
            gate.update(weight_shape=[2048, 1024], weight_bytes=2048 * 1024,
                        scale_shape=[2048], scale_bytes=2048 * 2)
            result.append(gate)
    for index, role in enumerate(family_paths):
        for layer, (weight, scale) in enumerate(zip(payload["weights"][index], payload["scales"][index])):
            result.append({"name": f"layers.{layer}.{role}", "method": "nvfp4a16",
                           "abi": "striped_v2", "block_size": 16,
                           "weight_shape": list(weight.shape), "weight_dtype": str(weight.dtype),
                           "weight_bytes": weight.numel() * weight.element_size(),
                           "scale_shape": list(scale.shape), "scale_dtype": str(scale.dtype),
                           "scale_bytes": scale.numel() * scale.element_size(),
                           "tensor_scale_dtype": "torch.float32", "tensor_scale_bytes": 4})
    return result
