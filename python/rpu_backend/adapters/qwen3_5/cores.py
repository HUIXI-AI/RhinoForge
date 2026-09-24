"""Exact cold core profiles for dense Qwen3.5 text and image inference."""
from collections.abc import Mapping
from numbers import Integral


REDUCED_CORE_ASSET_ALIASES = frozenset({
    "qwen3_5-0.8b", "qwen3_5-2b", "qwen3_5-4b", "qwen3_5-9b",
})
# H -> (I, layers, Q heads, KV heads, GDN value heads).
_TEXT_PROFILES = {
    1024: (3584, 24, 8, 2, 16),
    2048: (6144, 24, 8, 2, 16),
    2560: (9216, 32, 16, 4, 32),
    4096: (12288, 32, 16, 4, 32),
}


def core_topology(execution=None):
    from rpu_backend.runtime.topology import DecoderTopology, execution_core_count

    cores = execution_core_count(execution)
    attention = 8 if cores == 8 else 4
    return DecoderTopology(cores, attention, cores, attention)


def topology_tuple(topology):
    return (topology.num_cores, topology.attn_tp, topology.mlp_tp,
            topology.lm_head_tp, topology.physical_kv_cores)


def vision_core_count(execution=None):
    return core_topology(execution).attn_tp


def _exact_integer(value, expected):
    return isinstance(value, Integral) and not isinstance(value, bool) and value == expected


def validate_vision_core_profile(config, num_cores):
    if type(num_cores) is not int or num_cores not in (4, 8):
        raise ValueError("Qwen3.5 Vision num_cores must be 4 or 8")
    if num_cores == 8:
        return
    _validate_dense_vision_profile(config)


def _validate_dense_vision_profile(config):
    geometry = tuple(getattr(config, name, None) for name in (
        "depth", "num_heads", "hidden_size", "intermediate_size", "out_hidden_size"))
    allowed = {(12, 12, 768, 3072, 1024),
               (24, 16, 1024, 4096, 2048), (24, 16, 1024, 4096, 2560),
               (27, 16, 1152, 4304, 4096)}
    valid = (all(isinstance(x, Integral) and not isinstance(x, bool) for x in geometry)
             and geometry in allowed
             and getattr(config, "hidden_act", None) == "gelu_pytorch_tanh"
             and not getattr(config, "deepstack_visual_indexes", ())
             and all(_exact_integer(getattr(config, name, None), value) for name, value in (
                 ("in_channels", 3), ("patch_size", 16), ("temporal_patch_size", 2),
                 ("spatial_merge_size", 2), ("num_position_embeddings", 2304)))
             and not any(getattr(config, name, None) for name in (
                 "quant_config", "quantization_config")))
    if not valid:
        raise ValueError("Vision requires an exact dense FP16 Qwen3.5 0.8B/2B/4B/9B tower")


def validate_core_profile(config, execution=None):
    """Reject reduced near-neighbours before any model/weight/handle mutation.

    Checkpoint dtype metadata may say BF16; the public loader explicitly loads
    and materializes FP16. Actual tensor dtype is checked by the installer.
    Legacy eight-core profiles retain their existing independent admission.
    """
    topology = core_topology(execution)
    if topology.num_cores == 8:
        return topology
    from rpu_backend.adapters.qwen3_5.quant_scope import validate_legacy_text_profile
    if validate_legacy_text_profile(config):
        tc = getattr(config, "text_config", config)
        if (getattr(tc, "model_type", None) != "qwen3_5_text"
                or getattr(tc, "vocab_size", None) != 248320
                or getattr(tc, "use_cache", None) is not True
                or getattr(tc, "full_attention_interval", 4) != 4
                or any(getattr(owner, name, None) for owner in (config, tc)
                       for name in ("quantization_config", "num_experts", "adaptive_mode"))):
            raise ValueError("reduced Qwen3.8-27B requires the exact legacy W8 text profile")
        return topology
    _validate_dense_text_profile(config)
    return topology


def _reduced_w8_specs(config):
    """Exact 2B W8 main/GDN projections with FP16 scalar b/a projections.

    Reuse the checkpoint QuantSpec resolver; a profile label alone never admits
    different tensors or a different per-projection arithmetic contract.
    """
    quant = getattr(config, "quant_config", None)
    if not quant:
        return None
    from .quant_scope import FP16, W8A16, is_full_from_layer_types, resolve_quant_specs
    tc = getattr(config, "text_config", config)
    if not _exact_integer(getattr(tc, "hidden_size", None), 2048):
        raise ValueError("reduced Qwen3.5 quantization requires the exact 2B W8 main/GDN profile")
    specs = resolve_quant_specs(is_full_from_layer_types(getattr(tc, "layer_types", ())), quant)
    scalar_roles = (".linear_attn.in_proj_b", ".linear_attn.in_proj_a")
    if not specs or any(spec.method != (FP16 if name.endswith(scalar_roles) else W8A16)
                        for name, spec in specs.items()):
        raise ValueError("reduced Qwen3.5 2B requires W8 main/GDN projections and FP16 GDN b/a")
    return specs


def _validate_dense_text_profile(config, *, uniform_quant=False):
    if uniform_quant and getattr(config, "quant_config", None):
        from .quant_scope import W8A16, W4A16_PGRP, is_full_from_layer_types, resolve_quant_specs
        tc = getattr(config, "text_config", config)
        specs = set(resolve_quant_specs(
            is_full_from_layer_types(getattr(tc, "layer_types", ())), config.quant_config).values())
        if (len(specs) != 1 or next(iter(specs)).method not in (W8A16, W4A16_PGRP)
                or (next(iter(specs)).method == W4A16_PGRP and next(iter(specs)).group_size != 32)):
            raise ValueError("retained 9B text requires uniform W8 or W4 group32 projections")
    else:
        _reduced_w8_specs(config)
    tc = getattr(config, "text_config", config)
    hidden = getattr(tc, "hidden_size", None)
    row = _TEXT_PROFILES.get(hidden) if isinstance(hidden, Integral) and not isinstance(hidden, bool) else None
    fields = ("intermediate_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "linear_num_value_heads")
    valid = row is not None and all(_exact_integer(getattr(tc, name, None), value)
                                   for name, value in zip(fields, row or ()))
    valid = valid and all(_exact_integer(getattr(tc, name, None), value) for name, value in (
        ("head_dim", 256), ("linear_num_key_heads", 16), ("linear_key_head_dim", 128),
        ("linear_value_head_dim", 128), ("linear_conv_kernel_dim", 4), ("vocab_size", 248320)))
    rope = getattr(tc, "rope_parameters", None) or {}
    valid = (valid and getattr(tc, "model_type", None) == "qwen3_5_text"
             and getattr(tc, "hidden_act", None) == "silu"
             and getattr(tc, "rms_norm_eps", None) == 1e-6
             and getattr(tc, "attention_bias", None) is False
             and getattr(tc, "attn_output_gate", True) is True
             and getattr(tc, "use_cache", None) is True
             and getattr(tc, "full_attention_interval", 4) == 4
             and not getattr(tc, "mlp_only_layers", ())
             and not any(getattr(owner, name, None) for owner in (config, tc) for name in (
                 "quantization_config", "num_experts", "adaptive_mode"))
             and (tc is config or not getattr(tc, "quant_config", None))
             and isinstance(rope, Mapping)
             and rope.get("rope_type", rope.get("type")) == "default"
             and rope.get("rope_theta") == 10_000_000
             and rope.get("partial_rotary_factor") == 0.25
             and rope.get("mrope_interleaved") is True
             and isinstance(rope.get("mrope_section"), (list, tuple))
             and tuple(rope["mrope_section"]) == (11, 11, 10))
    if valid:
        expected = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(row[1])]
        valid = list(getattr(tc, "layer_types", ()) or ()) == expected
    if not valid:
        raise ValueError("model.num_cores=4/6 requires exact dense Qwen3.5 FP16 0.8B/2B/4B/9B or 2B W8 main/GDN hybrid text with partial interleaved M-RoPE")
    if hasattr(config, "text_config"):
        vc = getattr(config, "vision_config", None)
        if (getattr(config, "model_type", None) != "qwen3_5"
                or getattr(config, "tie_word_embeddings", None) is not (hidden != 4096)
                or getattr(config, "image_token_id", None) != 248056):
            raise ValueError("reduced Qwen3.5 requires the exact conditional-generation config")
        _validate_dense_vision_profile(vc)
        if vc.out_hidden_size != hidden:
            raise ValueError("Qwen3.5 Vision merger and text widths disagree")


def validate_text_weight_geometry(inner, config, topology):
    """Validate raw reduced-profile shapes before irreversible swizzling."""
    if topology.num_cores == 8:
        return
    _validate_dense_text_weights(inner, config)
    from .quant_scope import validate_legacy_text_profile
    if validate_legacy_text_profile(config):
        return  # The existing legacy-specific cold validator owns its W8/FP16 policy.
    specs = _reduced_w8_specs(config)
    if specs is not None:
        _validate_reduced_w8_weights(inner, config, specs)


def _validate_reduced_w8_weights(inner, config, specs, *, cold_cpu=False):
    """Check raw projection storage; the cold VL owner also requires CPU scales."""
    import torch
    from .quant_scope import W8A16, is_full_from_layer_types
    from .text import _iter_text_projections
    tc = getattr(config, "text_config", config)
    is_full = is_full_from_layer_types(tc.layer_types)
    for linear, name in _iter_text_projections(inner, is_full):
        weight = linear.weight
        w8 = specs[name].method == W8A16
        allowed_float = ((torch.float16, torch.bfloat16, torch.float32)
                         if weight.device.type == "cpu" else (torch.float16,))
        devices = ("cpu",) if cold_cpu else ("cpu", "rpu", "privateuseone")
        if (weight.device.type not in devices
                or not weight.is_contiguous()
                or hasattr(linear, "_rpu_linear_partition")
                or (weight.dtype != torch.int8 if w8 else weight.dtype not in allowed_float)
                or getattr(linear, "bias", None) is not None):
            raise ValueError(f"Qwen3.5 reduced projection {name} has incompatible raw W8/FP16 storage")
        scale = getattr(linear, "weight_scale", None)
        if w8:
            if (not isinstance(scale, torch.Tensor) or scale.dtype != torch.float16
                    or scale.device != weight.device or not scale.is_contiguous()
                    or tuple(scale.shape) != (weight.shape[0],)):
                raise ValueError(f"Qwen3.5 reduced W8 projection {name} requires contiguous FP16 scale [N]")
            if cold_cpu and (not bool(torch.isfinite(scale).all())
                             or not bool((scale > 0).all())):
                raise ValueError(f"Qwen3.5 cold W8 projection {name} requires finite positive scales")
        elif isinstance(scale, torch.Tensor) and scale.numel():
            raise ValueError(f"Qwen3.5 reduced FP16 projection {name} must not carry a scale")


def validate_text_mlp_padding_storage(inner, config, topology, *, cold_cpu=False):
    """Check every padded raw leaf before claiming or transforming the model.

    The outer loader checks CPU floating weights before its explicit half cast;
    the text installer checks FP16 raw weights, which may already live on RPU.
    The installer pads CPU copies, leaving logical checkpoint metadata intact.
    """
    from rpu_backend.runtime.topology import decoder_mlp_intermediate_size
    import torch

    from rpu_backend.adapters.qwen3_5.quant_scope import validate_legacy_text_profile
    if topology.num_cores != 8 and validate_legacy_text_profile(config):
        from rpu_backend.adapters.qwen3_5.weights import validate_legacy_cold_weights

        validate_legacy_cold_weights(inner, config, cold_cpu=cold_cpu)
        return
    tc = getattr(config, "text_config", config)
    logical = tc.intermediate_size
    if decoder_mlp_intermediate_size(logical, topology.mlp_tp) == logical:
        return
    allowed_dtype = ((torch.float16, torch.bfloat16, torch.float32)
                     if cold_cpu else (torch.float16,))
    allowed_device = ("cpu",) if cold_cpu else ("cpu", "rpu", "privateuseone")
    for layer in inner.layers:
        for name, shape in (("gate_proj", (logical, tc.hidden_size)),
                            ("up_proj", (logical, tc.hidden_size)),
                            ("down_proj", (tc.hidden_size, logical))):
            projection = getattr(layer.mlp, name)
            weight = projection.weight
            if (not isinstance(projection, torch.nn.Linear)
                    or weight.device.type not in allowed_device
                    or weight.dtype not in allowed_dtype or not weight.is_contiguous()
                    or tuple(weight.shape) != shape
                    or (projection.out_features, projection.in_features) != shape
                    or projection.bias is not None
                    or hasattr(projection, "_rpu_linear_partition")):
                raise ValueError("Qwen3.5 MLP padding requires original contiguous "
                                 "floating Linear weights with logical geometry")


def _validate_dense_text_weights(inner, config):
    tc = getattr(config, "text_config", config)
    if len(inner.layers) != tc.num_hidden_layers:
        raise ValueError("Qwen3.5 layer count disagrees with the reduced profile")
    h, inter = tc.hidden_size, tc.intermediate_size

    def shape(owner, name, expected):
        value = getattr(owner, name, None)
        if value is None or tuple(value.shape) != expected:
            raise ValueError(f"Qwen3.5 reduced weight {name} requires shape {expected}")

    shape(inner.embed_tokens, "weight", (tc.vocab_size, h))
    shape(inner.norm, "weight", (h,))
    for index, layer in enumerate(inner.layers):
        for name in ("input_layernorm", "post_attention_layernorm"):
            shape(getattr(layer, name), "weight", (h,))
        for name, expected in (("gate_proj", (inter, h)), ("up_proj", (inter, h)),
                               ("down_proj", (h, inter))):
            shape(getattr(layer.mlp, name), "weight", expected)
        if tc.layer_types[index] == "full_attention":
            attention = layer.self_attn
            for name, expected in (("q_proj", (2 * tc.num_attention_heads * 256, h)),
                                   ("k_proj", (tc.num_key_value_heads * 256, h)),
                                   ("v_proj", (tc.num_key_value_heads * 256, h)),
                                   ("o_proj", (h, tc.num_attention_heads * 256))):
                shape(getattr(attention, name), "weight", expected)
            shape(attention.q_norm, "weight", (256,))
            shape(attention.k_norm, "weight", (256,))
        else:
            attention = layer.linear_attn
            width = tc.linear_num_value_heads * 128
            for name, expected in (("in_proj_qkv", (4096 + width, h)),
                                   ("in_proj_z", (width, h)),
                                   ("in_proj_b", (tc.linear_num_value_heads, h)),
                                   ("in_proj_a", (tc.linear_num_value_heads, h)),
                                   ("out_proj", (h, width))):
                shape(getattr(attention, name), "weight", expected)
            shape(attention.conv1d, "weight", (4096 + width, 1, 4))
            shape(attention, "A_log", (tc.linear_num_value_heads,))
            shape(attention, "dt_bias", (tc.linear_num_value_heads,))
            shape(attention.norm, "weight", (128,))


def is_dense_9b_vl_candidate(config):
    """Select the cold resource check; this is not profile admission."""
    return (getattr(getattr(config, "text_config", None), "hidden_size", None) == 4096
            and getattr(config, "vision_config", None) is not None)


def validate_dense_9b_vl_profile(config):
    if not is_dense_9b_vl_candidate(config):
        raise ValueError("the four-arena owner requires exact dense Qwen3.5 9B VL")
    specs = _dense_9b_vl_w8_specs(config)
    _validate_dense_text_profile(config, uniform_quant=specs is not None)


def _dense_9b_vl_w8_specs(config):
    """The VL arena plan admits modern uniform W8, independently of text W4."""
    quant = getattr(config, "quant_config", None)
    if quant is None:
        return None
    from .quant_scope import W8A16, is_full_from_layer_types, resolve_quant_specs

    tc = getattr(config, "text_config", config)
    specs = resolve_quant_specs(is_full_from_layer_types(tc.layer_types), quant)
    if not specs or any(spec.method != W8A16 for spec in specs.values()):
        raise ValueError("Qwen3.5 9B VL requires uniform W8 text projections or FP16")
    return specs


def validate_dense_9b_retained_text_profile(config):
    """Exact public TP8 text lifecycle; no reduced-core or Vision admission."""
    if not is_dense_9b_vl_candidate(config):
        raise ValueError("retained 9B text requires the exact dense Qwen3.5 config")
    _validate_dense_text_profile(config, uniform_quant=True)


def validate_dense_9b_vision_profile(config):
    _validate_dense_vision_profile(config)
    if config.hidden_size != 1152 or config.out_hidden_size != 4096:
        raise ValueError("the four-arena Vision owner requires exact dense Qwen3.5 9B")


def validate_dense_9b_vision_weights(visual, config):
    """Inspect raw CPU geometry before lazy Vision can mutate any weights."""
    validate_dense_9b_vision_profile(config)

    def shape(owner, name, expected):
        value = getattr(owner, name, None)
        if value is None or tuple(value.shape) != expected:
            raise ValueError(f"Qwen3.5 9B Vision weight {name} requires shape {expected}")

    if len(visual.blocks) != 27:
        raise ValueError("Qwen3.5 9B Vision requires 27 raw blocks")
    for block in visual.blocks:
        for owner, out_size, in_size in (
            (block.attn.qkv, 3456, 1152), (block.attn.proj, 1152, 1152),
            (block.mlp.linear_fc1, 4304, 1152), (block.mlp.linear_fc2, 1152, 4304),
        ):
            shape(owner, "weight", (out_size, in_size))
            shape(owner, "bias", (out_size,))
        for norm in (block.norm1, block.norm2):
            shape(norm, "weight", (1152,))
            shape(norm, "bias", (1152,))
    shape(visual.patch_embed.proj, "weight", (1152, 3, 2, 16, 16))
    shape(visual.patch_embed.proj, "bias", (1152,))
    shape(visual.pos_embed, "weight", (2304, 1152))
    shape(visual.merger.norm, "weight", (1152,))
    shape(visual.merger.norm, "bias", (1152,))
    for owner, out_size in ((visual.merger.linear_fc1, 4608), (visual.merger.linear_fc2, 4096)):
        shape(owner, "weight", (out_size, 4608))
        shape(owner, "bias", (out_size,))


def validate_dense_9b_vl_weights(model):
    validate_dense_9b_vl_profile(model.config)
    inner = model.model.language_model
    _validate_dense_text_profile(inner.config)
    _validate_dense_text_weights(inner, model.config)
    output = model.get_output_embeddings()
    if tuple(output.weight.shape) != (248320, 4096):
        raise ValueError("Qwen3.5 9B LM-head weight shape disagrees with config")
    validate_dense_9b_vision_weights(model.model.visual, model.config.vision_config)
    validate_cold_weight_dtype(model)


def validate_cold_weight_dtype(model, *, fp16=False):
    import torch

    config = getattr(model, "config", None)
    specs = (_dense_9b_vl_w8_specs(config)
             if is_dense_9b_vl_candidate(config) else None)
    quantized = set()
    if specs is not None:
        _validate_reduced_w8_weights(
            model.model.language_model, config, specs, cold_cpu=True)
        quantized = {f"model.language_model.{name}.weight" for name in specs}
    allowed = ((torch.float16,) if fp16 or specs is not None
               else (torch.float16, torch.bfloat16, torch.float32))
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if name in quantized:
            continue  # Exact INT8 weight and FP16 scale were checked above.
        if parameter.device.type != "cpu" or parameter.dtype not in allowed:
            raise ValueError(f"Qwen3.5 9B cold weight {name} must be CPU "
                             + ("FP16" if fp16 or specs is not None else "floating point"))
