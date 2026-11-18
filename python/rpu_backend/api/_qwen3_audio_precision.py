"""Cold decoder precision and physical weight packing for speech models."""
from __future__ import annotations


def normalize_audio_precision(precision, rpu_execution, default_acc32):
    if precision not in ("fp16", "w8a16", "w4a16_g32", "nvfp4"):
        raise ValueError("speech precision must be fp16, w8a16, w4a16_g32 or nvfp4")
    execution = {} if rpu_execution is None else rpu_execution
    if not isinstance(execution, dict) or set(execution) - {"model", "prefill"}:
        raise ValueError("speech execution accepts model.num_cores and prefill.linear_acc32")
    model, prefill = execution.get("model", {}), execution.get("prefill", {})
    if not isinstance(model, dict) or set(model) - {"num_cores"}:
        raise ValueError("speech model execution accepts only num_cores")
    cores = model.get("num_cores", 8)
    if type(cores) is not int or cores not in (4, 6, 8):
        raise ValueError("speech num_cores must be 4, 6 or 8")
    if not isinstance(prefill, dict) or set(prefill) - {"linear_acc32"}:
        raise ValueError("speech prefill execution accepts only linear_acc32")
    acc32 = prefill.get("linear_acc32", default_acc32)
    if type(acc32) is not bool:
        raise ValueError("speech linear_acc32 must be a boolean")
    return precision, acc32


def asr_decoder_profile(config):
    """Admit the two reviewed ASR text components after equal-axis RoPE conversion."""
    from rpu_backend.runtime.topology import DecoderGeometryProfile

    names = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
             "num_key_value_heads", "head_dim", "vocab_size")
    values = tuple(getattr(config, name, None) for name in names)
    allowed = ((1024, 3072, 28, 16, 8, 128, 151936),
               (2048, 6144, 28, 16, 8, 128, 151936))
    rope = getattr(config, "rope_parameters", None)
    layers = getattr(config, "layer_types", None)
    if (any(type(value) is not int for value in values) or values not in allowed
            or getattr(config, "model_type", None) != "qwen3"
            or getattr(config, "architectures", None) is not None
            or getattr(config, "hidden_act", None) != "silu"
            or getattr(config, "rms_norm_eps", None) != 1e-6
            or getattr(config, "attention_bias", False)
            or getattr(config, "mlp_bias", False)
            or getattr(config, "attention_dropout", 0) != 0
            or getattr(config, "use_sliding_window", False)
            or getattr(config, "tie_word_embeddings", None) is not True
            or getattr(config, "max_position_embeddings", None) != 65536
            or getattr(config, "partial_rotary_factor", 1.0) != 1.0
            or getattr(config, "quantization_config", None) is not None
            or getattr(config, "quant_config", None) is not None
            or not isinstance(rope, dict)
            or rope.get("rope_type") != "default" or rope.get("rope_theta") != 1000000
            or (layers is not None and (len(layers) != 28
                                       or any(kind != "full_attention" for kind in layers)))):
        raise ValueError("ASR text requires its exact reviewed dense decoder geometry and semantics")
    return DecoderGeometryProfile(*values, 6, six_core_lm_head_tp=6)


def push_asr_lm_head(causal_lm, handle, precision):
    """Bind an explicitly packed speech head without ordinary text recipe markers."""
    import torch
    from rpu_backend.runtime.topology import (
        decoder_lm_head_vocab_size,
        execution_core_count,
        resolve_decoder_topology,
    )

    normalize_audio_precision(precision, None, False)
    if precision == "nvfp4":
        raise ValueError("NVFP4 speech decoders require an FP16 output head")
    model, head = causal_lm.model, causal_lm.lm_head
    profile = asr_decoder_profile(model.config)
    topology = getattr(model, "_rpu_decoder_topology", None)
    if (getattr(model, "_rpu_audio_component", None) != "asr"
            or getattr(model, "_rpu_decoder_handle", None) != handle
            or topology is None or topology.num_cores not in (4, 6, 8)
            or causal_lm.config is not model.config
            or causal_lm._rpu_execution != model._rpu_execution
            or not isinstance(head, torch.nn.Linear) or head.bias is not None
            or (head.out_features, head.in_features) != (profile.vocab_size, profile.hidden_size)):
        raise ValueError("ASR head requires the installed matching speech decoder")
    if (execution_core_count(model._rpu_execution) != topology.num_cores
            or topology != resolve_decoder_topology(num_cores=topology.num_cores, **profile.geometry())):
        raise ValueError("ASR head requires the decoder's cold component topology")
    weight, scale = head.weight.detach(), getattr(head, "weight_scale", None)
    dtype = {"fp16": torch.float16, "w8a16": torch.int8, "w4a16_g32": torch.uint8}[precision]
    from rpu_backend.quant.int4_pgrp_pack import w4_storage_shape
    padded_vocab = decoder_lm_head_vocab_size(profile.vocab_size, topology.lm_head_tp)
    shape = (w4_storage_shape(padded_vocab, profile.hidden_size, 1, topology.lm_head_tp)
             if precision == "w4a16_g32" else (padded_vocab, profile.hidden_size))
    if (weight.dtype != dtype or weight.device.type != "rpu" or tuple(weight.shape) != shape
            or not weight.is_contiguous() or getattr(head, "_rpu_linear_partition", None) != 1
            or getattr(head, "_rpu_linear_num_cores", None) != topology.lm_head_tp):
        raise ValueError("ASR head requires its physical component weight payload")
    if precision == "fp16":
        if scale is not None:
            raise ValueError("FP16 ASR head must not carry quantization scales")
    else:
        if precision == "w8a16":
            scale_shape = (padded_vocab,)
        else:
            groups = (profile.hidden_size // 32 + 3) // 4 * 4
            rows = (padded_vocab // topology.lm_head_tp + 63) // 64 * 64
            scale_shape = (32, topology.lm_head_tp * groups * rows // 32)
        if (not isinstance(scale, torch.Tensor) or scale.dtype != torch.float16
                or scale.device != weight.device or not scale.is_contiguous()
                or tuple(scale.shape) != scale_shape):
            raise ValueError("ASR head scale payload differs from the selected precision")
    causal_lm._rpu_lm_head_w_keepalive = weight
    causal_lm._rpu_lm_head_scale_keepalive = scale
    resource = model._rpu_decoder_retirement_state
    resource.keepalive = (resource.keepalive, weight, scale)
    torch.ops.rpu.causal_decoder_set_lm_head(handle, weight, scale)


def pack_audio_linear(weight, partition, precision, *, bounded_head=False, num_cores=8):
    """Return CPU physical weights/scales without changing source embeddings."""
    import torch
    from rpu_backend.runtime.weights import transform_linear_weight
    from rpu_backend.runtime.topology import execution_core_count

    normalize_audio_precision(precision, None, False)
    execution_core_count({"model": {"num_cores": num_cores}})
    if precision == "nvfp4":
        raise ValueError("NVFP4 decoder projections require pack_audio_nvfp4 and tensor scales")
    if (type(partition) is not int or partition not in (0, 1)
            or weight.device.type != "cpu" or weight.ndim != 2
            or not weight.is_floating_point()):
        raise ValueError("speech packing requires a CPU floating-point matrix and partition 0 or 1")
    if bounded_head and partition != 1:
        raise ValueError("speech output heads require column partitioning")
    weight = weight.detach().to(torch.float16).contiguous()
    if not torch.isfinite(weight).all():
        raise ValueError("speech weights must remain finite in FP16")
    from rpu_backend.runtime.topology import decoder_lm_head_vocab_size
    padded_n = decoder_lm_head_vocab_size(int(weight.shape[0]), num_cores) if bounded_head else int(weight.shape[0])
    if precision == "fp16":
        if padded_n != int(weight.shape[0]):
            padded = weight.new_zeros((padded_n, int(weight.shape[1])))
            padded[:weight.shape[0]].copy_(weight)
            weight = padded
        return transform_linear_weight(weight, partition, num_cores), None
    if precision == "w8a16":
        from rpu_backend.quant._common import quantize_linear_per_channel

        packed, scale = quantize_linear_per_channel(weight, bits=8)
        if padded_n != int(packed.shape[0]):
            padded_packed = packed.new_zeros((padded_n, int(packed.shape[1])))
            padded_packed[:packed.shape[0]].copy_(packed)
            padded_scale = scale.new_ones(padded_n)
            padded_scale[:scale.shape[0]].copy_(scale)
            packed, scale = padded_packed, padded_scale
        return transform_linear_weight(packed, partition, num_cores), scale.contiguous()
    from rpu_backend.quant.convert_qwen3_w4a16 import pack_linear_int4_
    from rpu_backend.quant.int4_pgrp_pack import quantize_pack_int4_pgrp_col_bounded

    if bounded_head:
        if padded_n != int(weight.shape[0]):
            padded = weight.new_zeros((padded_n, int(weight.shape[1])))
            padded[:weight.shape[0]].copy_(weight)
            weight = padded
        return quantize_pack_int4_pgrp_col_bounded(weight, group_size=32, num_cores=num_cores)
    # A meta shell avoids allocating another full floating-point matrix.
    with torch.device("meta"):
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    pack_linear_int4_(linear, partition, group_size=32, num_cores=num_cores)
    return linear.weight.detach(), linear.weight_scale


def pack_audio_nvfp4(weight, partition, *, num_cores=8):
    """Return physical E2M1 weight, E4M3 block scales and one FP32 tensor scale."""
    import torch
    from rpu_backend.quant.nvfp4_pack import (
        quantize_nvfp4, swizzle_pack_nvfp4, swizzle_nvfp4_scales,
    )
    from rpu_backend.runtime.topology import execution_core_count

    execution_core_count({"model": {"num_cores": num_cores}})

    if (type(partition) is not int or partition not in (0, 1)
            or weight.device.type != "cpu" or weight.ndim != 2
            or not weight.is_floating_point()):
        raise ValueError("speech NVFP4 packing requires a CPU matrix and partition 0 or 1")
    n, k = weight.shape
    codes, scales, tensor_scale = quantize_nvfp4(weight.detach().to(torch.float16))
    return (swizzle_pack_nvfp4(codes, partition, num_cores).reshape(n, k // 2),
            swizzle_nvfp4_scales(scales, partition, num_cores), tensor_scale)


def pack_audio_tensor_scales(values):
    """Pad one decoder projection family's FP32 scales for native vector loads."""
    import torch

    if len(values) not in (5, 28) or any(
            not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or value.dtype != torch.float32 or value.ndim != 0 for value in values):
        raise ValueError("speech NVFP4 requires five or twenty-eight CPU FP32 tensor scales")
    scales = torch.stack(values)
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError("speech NVFP4 tensor scales must be finite and positive")
    return torch.nn.functional.pad(scales, (0, (-len(values)) % 8), value=1.0).contiguous()
