"""Exact dense Qwen3 geometry for the TP8 quantized loader.

This geometry table does not expand FP16 or reduced-core admission.
"""
from collections.abc import Mapping
from numbers import Integral

from rpu_backend.runtime.topology import DecoderGeometryProfile


_PROFILES = tuple(DecoderGeometryProfile(h, i, layers, heads, 8, 128, 151936, 8)
                  for h, i, layers, heads in (
                      (1024, 3072, 28, 16), (2048, 6144, 28, 16),
                      (2560, 9728, 36, 32), (4096, 12288, 36, 32),
                      (5120, 17408, 40, 40), (5120, 25600, 64, 64)))

_W8_METADATA = {
    "method": "w8a16", "mode": "per_channel_symmetric", "qaxis": 0,
    "skip_modules": [], "quantized_lm_head": True,
    "quantized_embed_tokens": False, "lm_head_untied": True,
    "embed_tokens_untied": False,
}


def _profile(config):
    fields = ("hidden_size", "intermediate_size", "num_hidden_layers",
              "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size")
    values = tuple(getattr(config, key, None) for key in fields)
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in values):
        raise ValueError("Qwen3 quantization requires exact integer dense geometry")
    profile = next((p for p in _PROFILES if values == (
        p.hidden_size, p.intermediate_size, p.num_layers, p.num_q_heads,
        p.num_kv_heads, p.head_dim, p.vocab_size)), None)
    rope = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    layer_types = getattr(config, "layer_types", None)
    capacity = getattr(config, "max_position_embeddings", None)
    if (profile is None
            or getattr(config, "model_type", None) != "qwen3"
            or tuple(getattr(config, "architectures", ()) or ()) != ("Qwen3ForCausalLM",)
            or getattr(config, "rms_norm_eps", None) != 1e-6
            or getattr(config, "attention_bias", False)
            or getattr(config, "mlp_bias", False)
            or getattr(config, "use_sliding_window", False)
            or getattr(config, "hidden_act", "silu") != "silu"
            or getattr(config, "partial_rotary_factor", 1.0) != 1.0
            or not isinstance(rope, Mapping)
            or rope.get("rope_type", rope.get("type", "default")) != "default"
            or (layer_types is not None and
                (len(layer_types) != profile.num_layers or
                 any(kind != "full_attention" for kind in layer_types)))
            or isinstance(capacity, bool) or not isinstance(capacity, Integral) or capacity <= 0):
        raise ValueError("Qwen3 quantization requires an exact dense full-attention profile with default RoPE")
    return profile


def qwen3_dense_quant_profile(config):
    """Admit an unquantized source for the explicit symmetric G32 recipe."""
    profile = _profile(config)
    if (getattr(config, "quant_config", None) is not None
            or getattr(config, "quantization_config", None) is not None
            or getattr(config, "tie_word_embeddings", None) is not (profile.hidden_size < 4096)):
        raise ValueError("Qwen3 W4 requires the official dense source, not a pre-quantized checkpoint")
    return profile


def is_qwen3_32b_w8a16_config(config):
    """The converter's exact 32B W8 decoder/head checkpoint contract."""
    try:
        profile = _profile(config)
    except (TypeError, ValueError):
        return False
    metadata = getattr(config, "quant_config", None)
    return (profile.num_layers == 64
            and getattr(config, "tie_word_embeddings", None) is False
            and getattr(config, "quantization_config", None) is None
            and isinstance(metadata, dict) and metadata == _W8_METADATA
            and all(type(metadata[key]) is type(value) for key, value in _W8_METADATA.items()))
