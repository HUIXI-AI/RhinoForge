"""Derive exact Qwen3-VL runtime-scope W8A16 or W4A16 checkpoints.

Text projections and the independent LM head use per-channel INT8 or signed
INT4/G32 with logical FP16 scales. Embeddings stay FP16. Vision weights stay FP16 in
the checkpoint and are quantized by the W8 Vision installer, including patch
embedding and the final/DeepStack mergers. This is not an existing FP16
certification or a claim of byte parity with separately supplied assets.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ._common import quantize_linear_per_channel
from .convert_qwen3 import _checkpoint_layout, _copy_sidecar_files
from .int4_pgrp_pack import quantize_int4_group_wise


PROFILE = "qwen3_vl_4b_w8a16_runtime_align_v1"
QUANT_CONFIG = {
    "method": "w8a16",
    "profile": PROFILE,
    "mode": "per_channel_symmetric",
    "qaxis": 0,
    "skip_modules": ["model.language_model.embed_tokens"],
    "quantized_lm_head": True,
    "quantized_embed_tokens": False,
    "lm_head_untied": True,
    "embed_tokens_untied": False,
    "vision_quantization": "per_channel_on_install",
}
TEXT_PREFIX = "model.language_model.layers."
EMBED_NAME = "model.language_model.embed_tokens.weight"
HEAD_NAME = "lm_head.weight"
AWQ_SOURCE_QUANTIZATION = "compressed_tensors_awq"
AWQ_RUNTIME_RECIPE = "preserve_awq_text7_fp16_rtn_head_w4_vision_w8_v1"
RTN_RUNTIME_RECIPE = "fp16_rtn_symmetric_text7_head_g32_vision_w8_v1"
_TEXT_GEOMETRIES = {
    2048: (2, 6144, 28, 16),
    2560: (4, 9728, 36, 32),
    4096: (8, 12288, 36, 32),
    5120: (32, 25600, 64, 64),
}


def runtime_quant_config(hidden_size: int, bits: int, *, source_quantization=None) -> dict:
    """Exact runtime scope; preserve the existing 4B/W8 format verbatim."""
    if type(hidden_size) is not int or hidden_size not in _TEXT_GEOMETRIES:
        raise ValueError("runtime quantization requires an exact registered hidden size")
    if type(bits) is not int or bits not in (8, 4):
        raise ValueError("runtime quantization requires bits=8 or bits=4")
    if (source_quantization not in (None, AWQ_SOURCE_QUANTIZATION)
            or (source_quantization is not None and bits != 4)):
        raise ValueError("AWQ preservation requires exact compressed_tensors_awq and bits=4")
    if hidden_size == 5120 and source_quantization is not None:
        raise ValueError("32B runtime W4 requires symmetric RTN/G32, not an AWQ recipe")
    if hidden_size == 4096 and bits == 4 and source_quantization != AWQ_SOURCE_QUANTIZATION:
        raise ValueError("8B runtime W4 requires preserved compressed_tensors_awq Text7")
    result = copy.deepcopy(QUANT_CONFIG)
    size = _TEXT_GEOMETRIES[hidden_size][0]
    result["profile"] = f"qwen3_vl_{size}b_w{bits}a16_runtime_align_v1"
    if bits == 4:
        result.update(method="w4a16", mode="per_group_symmetric", group_size=32,
                      weight_storage="logical_twos_complement_uint8_n_kdiv2",
                      scale_storage="logical_fp16_g_n")
        if hidden_size == 5120:
            result["recipe"] = RTN_RUNTIME_RECIPE
    if source_quantization is not None:
        result.update(profile=f"qwen3_vl_{size}b_w4a16_awq_runtime_align_v1",
                      source_quantization=AWQ_SOURCE_QUANTIZATION, recipe=AWQ_RUNTIME_RECIPE)
    return result


def _get(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def matches_4b_geometry(config) -> bool:
    """Exact official dense geometry/semantics, independent of storage dtype."""
    return _matches_geometry(config, 2560)


def matches_runtime_geometry(config) -> bool:
    text, vision = _get(config, "text_config"), _get(config, "vision_config")
    hidden = _get(text, "hidden_size")
    return (type(hidden) is int and hidden in _TEXT_GEOMETRIES and _matches_geometry(config, hidden)
            and _get(text, "attention_bias") is False and _get(text, "use_cache") is True
            and all(type(_get(text, key)) is int for key in (
                "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                "head_dim", "vocab_size", "max_position_embeddings"))
            and all(type(_get(vision, key)) is int for key in (
                "hidden_size", "intermediate_size", "depth", "num_heads", "patch_size",
                "temporal_patch_size", "spatial_merge_size", "out_hidden_size", "in_channels", "num_position_embeddings")))


def _matches_geometry(config, hidden_size) -> bool:
    try:
        text = _get(config, "text_config")
        vision = _get(config, "vision_config")
        rope = _get(text, "rope_parameters") or _get(text, "rope_scaling") or {}
        _, intermediate, layers, heads = _TEXT_GEOMETRIES[hidden_size]
        vh, vi, depth, taps = ((1152, 4304, 27, (8, 16, 24)) if hidden_size in (4096, 5120)
                               else (1024, 4096, 24, (5, 11, 17)))
        return (
            _get(config, "model_type") == "qwen3_vl"
            and tuple(_get(config, "architectures", ())) == ("Qwen3VLForConditionalGeneration",)
            and tuple(_get(text, key) for key in (
                "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
                "hidden_act", "rms_norm_eps", "attention_bias", "use_cache", "max_position_embeddings",
            )) == ("qwen3_vl_text", hidden_size, intermediate, layers, heads, 8, 128, 151936,
                  "silu", 1e-6, False, True, 262144)
            and _get(text, "attention_dropout", 0.0) == 0.0
            and rope.get("rope_type") == "default"
            and float(rope.get("rope_theta", _get(text, "rope_theta", 0))) == 5_000_000.0
            and rope.get("mrope_interleaved") is True
            and tuple(rope.get("mrope_section", ())) == (24, 20, 20)
            and tuple(_get(vision, key) for key in (
                "model_type", "hidden_size", "intermediate_size", "depth", "num_heads",
                "patch_size", "temporal_patch_size", "spatial_merge_size", "out_hidden_size",
                "hidden_act", "in_channels", "num_position_embeddings",
            )) == ("qwen3_vl", vh, vi, depth, 16, 16, 2, 2, hidden_size,
                  "gelu_pytorch_tanh", 3, 2304)
            and tuple(_get(vision, "deepstack_visual_indexes", ())) == taps
            and tuple(_get(config, key) for key in (
                "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id",
            )) == (151655, 151656, 151652, 151653)
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def quantize_weight(weight: torch.Tensor, *, bits: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound FP32 temporaries while preserving the common rowwise quantizer."""
    if (weight.ndim != 2 or not weight.is_floating_point() or weight.device.type != "cpu"
            or type(bits) is not int or bits not in (8, 4) or 0 in weight.shape):
        raise ValueError("runtime source weight must be a nonempty floating CPU [N,K] tensor with bits=8/4")
    n, k = weight.shape
    if bits == 4 and k % 32:
        raise ValueError("runtime W4 source requires K divisible by group size 32")
    values = torch.empty((n, k if bits == 8 else k // 2), dtype=torch.int8 if bits == 8 else torch.uint8)
    scales = torch.empty((n,) if bits == 8 else (k // 32, n), dtype=torch.float16)
    for start in range(0, weight.shape[0], 4096):
        stop = min(start + 4096, weight.shape[0])
        block = weight[start:stop].half()
        if not bool(torch.isfinite(block).all()):
            raise ValueError("runtime source weight must be finite and representable as FP16")
        if bits == 8:
            values[start:stop], scales[start:stop] = quantize_linear_per_channel(block)
        else:
            q, scale, _ = quantize_int4_group_wise(block, 32)
            nibble = q.to(torch.uint8).bitwise_and_(15)
            values[start:stop] = nibble[:, 0::2] | (nibble[:, 1::2] << 4)
            scales[:, start:stop] = scale
    if not bool(torch.isfinite(scales).all() and (scales > 0).all()):
        raise ValueError("runtime scales must be finite positive FP16")
    return values, scales


def _checkpoint_tensor_shapes(config):
    text = config["text_config"]
    h, inter, heads = (text[k] for k in ("hidden_size", "intermediate_size", "num_attention_heads"))
    projections = {
        "self_attn.q_proj": (heads * 128, h), "self_attn.k_proj": (1024, h),
        "self_attn.v_proj": (1024, h), "self_attn.o_proj": (h, heads * 128),
        "mlp.gate_proj": (inter, h), "mlp.up_proj": (inter, h),
        "mlp.down_proj": (h, inter),
    }
    return {f"{TEXT_PREFIX}{i}.{name}.weight": shape for i in range(text["num_hidden_layers"])
            for name, shape in projections.items()}, (151936, h)


def _awq_logical_projection(packed, shape, scale, *, n, k):
    """Preserve AWQ integers: offset nibbles -> logical signed nibbles, no swizzle."""
    from .load_qwen3_vl_awq import _checked_float

    if (packed.device.type != "cpu" or packed.dtype != torch.int32
            or tuple(packed.shape) != (n, k // 8) or k % 32):
        raise ValueError("AWQ weight_packed must be CPU INT32[N,K/8] with G32")
    if (shape.dtype != torch.int64 or tuple(shape.shape) != (2,)
            or shape.tolist() != [n, k]):
        raise ValueError("AWQ weight_shape must be INT64 [N,K]")
    if scale.device.type != "cpu" or tuple(scale.shape) != (n, k // 32):
        raise ValueError("AWQ weight_scale must be CPU BF16/FP16[N,K/32]")
    scale_half = _checked_float("weight_scale", scale)
    if not bool((scale_half > 0).all()):
        raise ValueError("AWQ scales must be representable as positive FP16")
    # For u in [0,15], two's-complement(u-8) == u XOR 8. Four bytes
    # per source word retain the original increasing-K order without depending
    # on host endianness or introducing a second quantization/physical packing.
    shifts = torch.arange(4, dtype=torch.int32) * 8
    logical = (((packed.unsqueeze(-1) >> shifts) & 255) ^ 0x88).to(torch.uint8)
    return logical.reshape(n, k // 2), scale_half.t().contiguous()


def _awq_checkpoint_schema(config):
    """Use the same unquantized meta inventory as the strict AWQ CPU loader."""
    from transformers import AutoModelForImageTextToText, Qwen3VLConfig

    skeleton = Qwen3VLConfig.from_dict(copy.deepcopy(config))
    del skeleton.quantization_config
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(skeleton)
    shapes = {name: tuple(t.shape) for name, t in model.named_parameters(remove_duplicate=False)}
    if config["tie_word_embeddings"]:
        shapes.pop(HEAD_NAME)
    projections, _ = _checkpoint_tensor_shapes(config)
    expected = {name: (shape, {"BF16", "F16"}) for name, shape in shapes.items()}
    for name, (n, k) in projections.items():
        if shapes.get(name) != (n, k):
            raise ValueError(f"AWQ source projection geometry differs from model: {name}")
        del expected[name]
        base = name[:-7]
        expected.update({base + ".weight_packed": ((n, k // 8), {"I32"}),
                         base + ".weight_shape": ((2,), {"I64"}),
                         base + ".weight_scale": ((n, k // 32), {"BF16", "F16"})})
    return projections, expected


def _convert_awq_checkpoint(src, dst, config, *, bits):
    """Preserve calibrated Text7 and every AWQ floating constant; quantize only head."""
    if not matches_runtime_geometry(config):
        raise ValueError("AWQ conversion requires exact Qwen3-VL runtime geometry")
    from .load_qwen3_vl_awq import _checked_float

    quant_config = runtime_quant_config(config["text_config"]["hidden_size"], bits,
                                        source_quantization=AWQ_SOURCE_QUANTIZATION)
    if dst.exists():
        raise FileExistsError(f"destination already exists: {dst}")
    sharded, shards, declared, index = _checkpoint_layout(src)
    projections, expected = _awq_checkpoint_schema(config)
    actual = {}
    for shard in shards:
        with safe_open(src / shard, framework="pt") as handle:
            for name in handle.keys():
                if name in actual or name not in expected:
                    raise ValueError(f"Unexpected or duplicate AWQ source tensor: {name}")
                view = handle.get_slice(name)
                shape, dtypes = expected[name]
                if tuple(view.get_shape()) != shape or view.get_dtype() not in dtypes:
                    raise ValueError(f"Invalid AWQ source tensor header: {name}")
                actual[name] = shard
    if set(actual) != set(expected):
        raise ValueError(f"AWQ source inventory differs: missing {sorted(set(expected)-set(actual))[:8]}")
    if sharded and (set(declared) != set(actual) or any(
            (src / declared[name]).resolve() != (src / actual[name]).resolve() for name in actual)):
        raise ValueError("AWQ source shard index differs from actual tensor locations")
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    if tmp.exists():
        raise FileExistsError(f"temporary output already exists: {tmp}")
    stats = {"profile": quant_config["profile"], "source_quantization": AWQ_SOURCE_QUANTIZATION,
             "recipe": AWQ_RUNTIME_RECIPE, "n_quantized": 0,
             "n_preserved_text_projections": 0, "bytes_out": 0, "n_shards": 0}
    output_map = {}
    projection_bases = {name[:-7]: shape for name, shape in projections.items()}
    try:
        tmp.mkdir(parents=True)
        with ExitStack() as stack:
            handles = {s: stack.enter_context(safe_open(src / s, framework="pt")) for s in shards}
            def tensor(name):
                return handles[actual[name]].get_tensor(name)
            for shard in shards:
                print(f"[convert_qwen3_vl AWQ preserve] {shard}", flush=True)
                out = {}
                for name in handles[shard].keys():
                    base, suffix = name.rsplit(".", 1)
                    if base in projection_bases:
                        if suffix != "weight_packed":
                            continue
                        n, k = projection_bases[base]
                        out[base + ".weight"], out[base + ".weight_scale"] = _awq_logical_projection(
                            tensor(name), tensor(base + ".weight_shape"), tensor(base + ".weight_scale"), n=n, k=k)
                        stats["n_preserved_text_projections"] += 1
                    elif name == HEAD_NAME:
                        out[name], out["lm_head.weight_scale"] = quantize_weight(_checked_float(name, tensor(name)), bits=4)
                        stats["n_quantized"] += 1
                    else:
                        out[name] = _checked_float(name, tensor(name))
                        if name == EMBED_NAME and config["tie_word_embeddings"]:
                            out[HEAD_NAME], out["lm_head.weight_scale"] = quantize_weight(out[name], bits=4)
                            stats["n_quantized"] += 1
                if not out:
                    continue
                save_file(out, str(tmp / shard), metadata={"format": "pt"})
                stats["n_shards"] += 1
                for name, value in out.items():
                    if name in output_map:
                        raise ValueError(f"duplicate converted AWQ tensor: {name}")
                    output_map[name] = shard
                    stats["bytes_out"] += value.numel() * value.element_size()
                del out
        if stats["n_preserved_text_projections"] != len(projections) or stats["n_quantized"] != 1:
            raise ValueError("AWQ conversion requires every original Text7 projection and one new head")
        if sharded:
            index = copy.deepcopy(index)
            index["metadata"] = {**index.get("metadata", {}), "total_size": stats["bytes_out"]}
            index["weight_map"] = dict(sorted(output_map.items()))
            (tmp / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
        config = copy.deepcopy(config)
        del config["quantization_config"]
        for owner in (config, config["text_config"], config["vision_config"]):
            owner["dtype"] = "float16"
        config["tie_word_embeddings"] = config["text_config"]["tie_word_embeddings"] = False
        config["quant_config"] = quant_config
        (tmp / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        _copy_sidecar_files(src, tmp, set(shards))
        tmp.rename(dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return stats


def convert_checkpoint(src: str | Path, dst: str | Path, *, bits: int = 8) -> dict:
    src, dst = Path(src).expanduser().resolve(), Path(dst).expanduser().resolve()
    config = json.loads((src / "config.json").read_text())
    from .load_qwen3_vl_awq import is_qwen3_vl_awq_config
    if is_qwen3_vl_awq_config(config, allow_padded_8b=True):
        return _convert_awq_checkpoint(src, dst, config, bits=bits)
    text = config.get("text_config", {})
    # The top-level ImageText flag controls the head. Large official sources
    # omit TextConfig.tie_word_embeddings. That text-only field cannot permit
    # synthesizing a missing untied ImageText head.
    large = text.get("hidden_size") in (4096, 5120)
    source_tied = config.get("tie_word_embeddings")
    valid_tie = ((source_tied is False and
                  (text.get("tie_word_embeddings") is None or
                   type(text.get("tie_word_embeddings")) is bool)) if large else
                 (source_tied is True and text.get("tie_word_embeddings") is True))
    if (not matches_runtime_geometry(config) or not valid_tie
            or any(_get(owner, key) is not None for owner in
                   (config, config["text_config"], config["vision_config"])
                   for key in ("quant_config", "quantization_config"))):
        raise ValueError("conversion requires an exact unquantized official Qwen3-VL-Instruct config")
    quant_config = runtime_quant_config(config["text_config"]["hidden_size"], bits)
    if dst.exists():
        raise FileExistsError(f"destination already exists: {dst}")
    sharded, shards, weight_map, index = _checkpoint_layout(src)
    expected, head_shape = _checkpoint_tensor_shapes(config)
    actual_map = {}
    # The source index is only a declaration. Check every physical header before
    # quantization or output creation; a stale index must not synthesize a valid
    # inventory while the checkpoint is missing its raw embedding.
    for shard in shards:
        with safe_open(str(src / shard), framework="pt") as handle:
            for name in handle.keys():
                if name in actual_map:
                    raise ValueError(f"duplicate source tensor: {name}")
                shape = expected.get(name, head_shape if name in (HEAD_NAME, EMBED_NAME) else None)
                if shape is not None and tuple(handle.get_slice(name).get_shape()) != shape:
                    raise ValueError(f"source tensor {name} header shape differs from expected {shape}")
                actual_map[name] = shard
    if sharded:
        if set(weight_map) != set(actual_map) or any(
                (src / weight_map[name]).resolve() != (src / actual_map[name]).resolve()
                for name in actual_map):
            raise ValueError("source shard index differs from actual tensor locations")
    source_names = set(actual_map)
    if EMBED_NAME not in source_names:
        raise KeyError(f"source checkpoint is missing {EMBED_NAME}")
    if source_tied is False and HEAD_NAME not in source_names:
        raise KeyError(f"untied source checkpoint is missing {HEAD_NAME}")
    if not set(expected).issubset(source_names):
        raise KeyError(f"missing text projections: {sorted(set(expected)-source_names)[:8]}")
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    if tmp.exists():
        raise FileExistsError(f"temporary output already exists: {tmp}")
    stats = {"profile": quant_config["profile"], "n_quantized": 0, "bytes_out": 0, "n_shards": len(shards)}
    output_map = {}
    try:
        tmp.mkdir(parents=True)
        for shard in shards:
            print(f"[convert_qwen3_vl] {shard}", flush=True)
            tensors = load_file(str(src / shard))
            out = {}
            for name, tensor in tensors.items():
                if not tensor.is_floating_point():
                    raise TypeError(f"unquantized source tensor {name} is {tensor.dtype}")
                shape = expected.get(name, head_shape if name in (HEAD_NAME, EMBED_NAME) else None)
                if shape is not None and tuple(tensor.shape) != shape:
                    raise ValueError(f"source tensor {name} has shape {tuple(tensor.shape)}, expected {shape}")
                if name in expected or name == HEAD_NAME:
                    out[name], out[name[:-7] + ".weight_scale"] = quantize_weight(tensor, bits=bits)
                    stats["n_quantized"] += 1
                else:
                    out[name] = tensor.to(torch.float16).contiguous()
                    if not bool(torch.isfinite(out[name]).all()):
                        raise ValueError(f"source tensor {name} is not finite FP16")
                if name == EMBED_NAME and HEAD_NAME not in source_names:
                    out[HEAD_NAME], out["lm_head.weight_scale"] = quantize_weight(tensor, bits=bits)
                    stats["n_quantized"] += 1
            save_file(out, str(tmp / shard), metadata={"format": "pt"})
            for name, tensor in out.items():
                if name in output_map:
                    raise KeyError(f"duplicate source tensor: {name}")
                output_map[name] = shard
                stats["bytes_out"] += tensor.numel() * tensor.element_size()
            del tensors, out
        if stats["n_quantized"] != len(expected) + 1:
            raise ValueError(f"expected {len(expected)} text projections and one head, got {stats['n_quantized']}")
        if sharded:
            index = copy.deepcopy(index)
            index["metadata"] = {**index.get("metadata", {}), "total_size": stats["bytes_out"]}
            index["weight_map"] = dict(sorted(output_map.items()))
            (tmp / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
        for owner in (config, config["text_config"], config["vision_config"]):
            owner["dtype"] = "float16"
        config["tie_word_embeddings"] = False
        config["text_config"]["tie_word_embeddings"] = False
        config["quant_config"] = quant_config
        (tmp / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        _copy_sidecar_files(src, tmp, set(shards))
        tmp.rename(dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--bits", type=int, choices=(8, 4), default=8)
    args = parser.parse_args()
    print(json.dumps(convert_checkpoint(args.src, args.dst, bits=args.bits), indent=2))


if __name__ == "__main__":
    main()
