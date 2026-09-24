"""Cold storage preparation for the exact legacy Qwen3.8-27B text policy."""
import torch

from .quant_scope import FP16, W8A16, is_full_from_layer_types, resolve_quant_specs


def validate_legacy_cold_weights(inner, config, *, cold_cpu):
    """Validate every raw leaf before claiming a reduced model or swizzling it."""
    tc = getattr(config, "text_config", config)
    specs = resolve_quant_specs(is_full_from_layer_types(tc.layer_types), config.quant_config)
    devices = ("cpu",) if cold_cpu else ("cpu", "rpu", "privateuseone")
    quant_weights = {name + ".weight" for name, spec in specs.items() if spec.method == W8A16}
    for name, value in (*inner.named_parameters(), *inner.named_buffers()):
        dtype = torch.int8 if name in quant_weights else torch.float16
        if (value.device.type not in devices or value.dtype != dtype
                or not value.is_contiguous()):
            raise ValueError(f"legacy Qwen3.8-27B cold tensor {name} has incompatible storage")
    for name, spec in specs.items():
        projection = inner.get_submodule(name)
        weight = projection.weight
        scale = getattr(projection, "weight_scale", None)
        if (not isinstance(projection, torch.nn.Linear) or projection.bias is not None
                or tuple(weight.shape) != (projection.out_features, projection.in_features)
                or hasattr(projection, "_rpu_linear_partition")):
            raise ValueError(f"legacy Qwen3.8-27B projection {name} must retain raw Linear geometry")
        if spec.method == W8A16:
            if (not isinstance(scale, torch.Tensor) or scale.dtype != torch.float16
                    or scale.device.type not in devices or not scale.is_contiguous()
                    or tuple(scale.shape) != (weight.shape[0],)):
                raise ValueError(f"legacy Qwen3.8-27B projection {name} requires raw FP16 scale [N]")
        elif spec.method != FP16 or (scale is not None and scale.numel()):
            raise ValueError(f"legacy Qwen3.8-27B GDN projection {name} requires unscaled FP16")


def pad_legacy_mlp_projection(weight, *, projection_name, logical_size, physical_size):
    """Zero-pad the admitted int8 MLP6 payload; retain logical HF parameters."""
    if (logical_size, physical_size) != (17408, 17472):
        raise ValueError("no admitted legacy Qwen3.8-27B MLP padding geometry")
    if projection_name not in ("gate_proj", "up_proj", "down_proj"):
        raise ValueError("legacy MLP padding requires a known projection role")
    shape = (5120, logical_size) if projection_name == "down_proj" else (logical_size, 5120)
    if (weight.device.type != "cpu" or weight.dtype != torch.int8
            or not weight.is_contiguous() or tuple(weight.shape) != shape):
        raise ValueError("legacy MLP padding requires original contiguous CPU int8 weights")
    padded_shape = (5120, physical_size) if projection_name == "down_proj" else (physical_size, 5120)
    with torch.no_grad():
        padded = weight.new_zeros(padded_shape)
        padded[:shape[0], :shape[1]].copy_(weight)
    return padded


def pad_legacy_mlp_scale(scale, *, projection_name):
    """New gate/up rows dequantize to zero with unit scale; down keeps [5120]."""
    if projection_name not in ("gate_proj", "up_proj", "down_proj"):
        raise ValueError("legacy MLP scale padding requires a known projection role")
    expected = 5120 if projection_name == "down_proj" else 17408
    if (scale.device.type != "cpu" or scale.dtype != torch.float16
            or not scale.is_contiguous() or tuple(scale.shape) != (expected,)):
        raise ValueError("legacy MLP scale padding requires original CPU FP16 scale [N]")
    if projection_name == "down_proj":
        return scale
    padded = scale.new_ones(17472)
    padded[:expected].copy_(scale)
    return padded


def pad_gdn_scalar(weight, *, num_heads, num_cores):
    """Core-major FP16 head parameters, at least one aligned16-byte row/core."""
    if (type(num_cores) is not int or num_cores not in (4, 8)
            or type(num_heads) is not int or num_heads not in (16, 32, 48)
            or weight.dtype != torch.float16 or tuple(weight.shape) != (num_heads,)):
        raise ValueError("Qwen3.5 GDN scalar requires admitted FP16 head geometry")
    local_heads = num_heads // num_cores
    slots = ((local_heads + 7) // 8) * 8
    padded = torch.zeros((num_cores, slots), dtype=weight.dtype)
    padded[:, :local_heads].copy_(weight.detach().to("cpu").reshape(num_cores, local_heads))
    return padded.reshape(-1).contiguous()
