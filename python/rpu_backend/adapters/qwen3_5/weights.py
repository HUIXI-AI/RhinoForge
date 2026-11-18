"""Cold storage preparation for exact large Qwen3.5 text profiles."""
import torch

from .quant_scope import FP16, W8A16, W4A16_PGRP, W4A16_AWQ, is_full_from_layer_types, resolve_quant_specs


def validate_modern_27b_cold_weights(inner, config, *, cold_cpu):
    """Validate official 27B raw leaves before any irreversible upload.

    The same geometry is used by the dense FP16 and uniform quantized
    checkpoints.  Quantized projections carry integer codes plus scales;
    FP16 projections keep their native floating weights and no scale.
    """
    tc = getattr(config, "text_config", config)
    specs = resolve_quant_specs(
        is_full_from_layer_types(tc.layer_types), getattr(config, "quant_config", None))
    devices = ("cpu",) if cold_cpu else ("cpu", "rpu", "privateuseone")
    quant_weights = {name + ".weight": spec for name, spec in specs.items()
                     if spec.method != FP16}
    quant_zeros = {name + ".weight_zero" for name, spec in specs.items()
                   if spec.method == W4A16_AWQ}
    for name, value in (*inner.named_parameters(), *inner.named_buffers()):
        # Nonpersistent rotary buffers retain their source FP32 dtype.
        if name.endswith(("inv_freq", "original_inv_freq")):
            continue
        spec = quant_weights.get(name)
        dtype = (torch.uint8 if name in quant_zeros or (spec and spec.method == W4A16_AWQ)
                 else torch.int8 if spec else torch.float16)
        if (value.device.type not in devices or value.dtype != dtype
                or not value.is_contiguous()):
            raise ValueError(f"Qwen3.5-27B cold tensor {name} has incompatible storage")
    for name, spec in specs.items():
        projection = inner.get_submodule(name)
        weight = projection.weight
        if (not isinstance(projection, torch.nn.Linear) or projection.bias is not None
                or tuple(weight.shape) != (projection.out_features, projection.in_features)
                or hasattr(projection, "_rpu_linear_partition")):
            raise ValueError(f"Qwen3.5-27B projection {name} requires original raw Linear geometry")
        if spec.method == FP16:
            scale = getattr(projection, "weight_scale", None)
            if scale is not None and scale.numel():
                raise ValueError(f"Qwen3.5-27B FP16 projection {name} must not carry a scale")
            continue
        if spec.method not in (W8A16, W4A16_PGRP, W4A16_AWQ):
            raise ValueError("Qwen3.5-27B cold installation requires an admitted projection recipe")
        scale = getattr(projection, "weight_scale", None)
        expected = ((weight.shape[0],) if spec.method == W8A16 else
                    (weight.shape[1] // spec.group_size, weight.shape[0]))
        if (not isinstance(scale, torch.Tensor) or scale.device != weight.device
                or scale.dtype != torch.float16 or not scale.is_contiguous()
                or tuple(scale.shape) != expected):
            raise ValueError(f"Qwen3.5-27B projection {name} requires raw FP16 scale {expected}")
        if cold_cpu and (not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all())):
            raise ValueError(f"Qwen3.5-27B projection {name} requires finite positive scales")
        if cold_cpu and spec.method == W4A16_PGRP:
            lower, upper = torch.aminmax(weight)
            if int(lower) < -8 or int(upper) > 7:
                raise ValueError(f"Qwen3.5-27B projection {name} requires signed INT4 values [-8, 7]")
        if spec.method == W4A16_AWQ:
            zero = getattr(projection, "weight_zero", None)
            if (not isinstance(zero, torch.Tensor) or zero.dtype != torch.uint8
                    or zero.device != weight.device or not zero.is_contiguous()
                    or tuple(zero.shape) != expected or int(zero.max()) > 15
                    or int(weight.max()) > 15):
                raise ValueError(f"Qwen3.5-27B AWQ projection {name} requires direct UINT4 codes and zeros")
        elif isinstance(getattr(projection, "weight_zero", None), torch.Tensor) and projection.weight_zero.numel():
            raise ValueError(f"Qwen3.5-27B projection {name} must not carry a zero tensor")


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


def pad_dense_27b_mlp_projection(weight, *, projection_name, logical_size, physical_size):
    """Zero-pad official 27B FP16 or integer MLP weights before swizzling."""
    if (logical_size, physical_size) not in ((17408, 17472), (17408, 17664)):
        raise ValueError("no admitted official Qwen3.5-27B MLP padding geometry")
    if projection_name not in ("gate_proj", "up_proj", "down_proj"):
        raise ValueError("official MLP padding requires a known projection role")
    axis = 1 if projection_name == "down_proj" else 0
    if (weight.device.type != "cpu" or weight.dtype not in (torch.float16, torch.int8)
            or weight.ndim != 2 or not weight.is_contiguous()
            or weight.shape[axis] != logical_size):
        raise ValueError("official MLP padding requires original contiguous CPU FP16 or INT8 weights")
    shape = list(weight.shape)
    shape[axis] = physical_size
    with torch.no_grad():
        padded = weight.new_zeros(shape)
        padded[:weight.shape[0], :weight.shape[1]].copy_(weight)
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


def pad_dense_27b_mlp_scale(scale, *, projection_name):
    return pad_legacy_mlp_scale(scale, projection_name=projection_name)


def pad_quant_mlp_projection(weight, *, projection_name, logical_size, physical_size):
    """Pad raw integer weights before swizzling, leaving logical metadata intact."""
    if ((logical_size, physical_size) not in
            ((3584, 3648), (3584, 3840), (17408, 17472), (17408, 17664))
            or projection_name not in ("gate_proj", "up_proj", "down_proj")
            or weight.device.type != "cpu" or weight.dtype not in (torch.int8, torch.uint8)
            or weight.dim() != 2 or not weight.is_contiguous()):
        raise ValueError("quantized MLP padding requires an admitted raw integer geometry")
    axis = 1 if projection_name == "down_proj" else 0
    if weight.shape[axis] != logical_size:
        raise ValueError("quantized MLP padding requires the logical projection width")
    shape = list(weight.shape)
    shape[axis] = physical_size
    padded = weight.new_zeros(shape)
    padded[:weight.shape[0], :weight.shape[1]].copy_(weight)
    return padded


def pad_quant_mlp_scale(scale, *, projection_name, logical_size, physical_size, group_size=0):
    """Pad channel-first W8/W4 scales to match the zero-filled weight stripe."""
    if (projection_name not in ("gate_proj", "up_proj", "down_proj")
            or scale.device.type != "cpu" or scale.dtype != torch.float16
            or not scale.is_contiguous() or scale.dim() not in (1, 2)):
        raise ValueError("quantized MLP padding requires raw FP16 scale storage")
    down = projection_name == "down_proj"
    if down and scale.dim() == 1:
        return scale
    axis = 1 if down else 0
    old = logical_size // group_size if down else logical_size
    new = physical_size // group_size if down else physical_size
    if (down and (group_size not in (32, 128) or logical_size % group_size or physical_size % group_size)
            or scale.shape[axis] != old):
        raise ValueError("quantized MLP scale disagrees with the logical projection width")
    shape = list(scale.shape)
    shape[axis] = new
    padded = scale.new_ones(shape)
    if scale.dim() == 1:
        padded[:old].copy_(scale)
    else:
        padded[:scale.shape[0], :scale.shape[1]].copy_(scale)
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
