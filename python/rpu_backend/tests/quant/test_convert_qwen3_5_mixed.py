"""Converter coverage for the Qwen3.5 mixed W4/W8/FP16 profile."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import pytest
from safetensors.torch import load_file, save_file


_QUANT = Path(__file__).resolve().parents[2] / "quant"
sys.path.insert(0, str(_QUANT))
import convert_qwen3_5 as convert  # noqa: E402
import precision_recipes  # noqa: E402


def _mini_checkpoint(path):
    layer_types = [convert.LINEAR_ATTENTION, convert.FULL_ATTENTION]
    is_full = convert.is_full_from_layer_types(layer_types)
    tensors = {
        f"{convert.TEXT_PREFIX}{relpath}.weight": torch.randn(64, 64)
        for relpath in convert.resolve_quant_specs(is_full)
    }
    tensors["model.visual.patch_embed.proj.weight"] = torch.randn(64, 64)
    path.mkdir(parents=True)
    save_file(tensors, path / "model.safetensors", metadata={"format": "pt"})
    with (path / "config.json").open("w") as f:
        json.dump({
            "architectures": [convert.ARCHITECTURE],
            "text_config": {
                "num_hidden_layers": len(layer_types),
                "layer_types": layer_types,
            },
        }, f)
    return is_full


def test_mixed_converter_emits_declared_tensor_formats(tmp_path):
    torch.manual_seed(0)
    src, dst = tmp_path / "fp16", tmp_path / "mixed"
    is_full = _mini_checkpoint(src)
    (src / "SHA256SUMS").write_text("source checkpoint checksum\n")

    stats = convert.convert_checkpoint(
        src, dst, profile="w4_main_w8_gdn"
    )
    assert not (dst / "SHA256SUMS").exists()
    # No registered recipe is this mix, so the output carries no record.
    assert stats["recipe"] is None and precision_recipes.read_record(dst) is None

    with (dst / "config.json").open() as f:
        quant_config = json.load(f)["quant_config"]
    specs = convert.resolve_quant_specs(is_full, quant_config)
    tensors = load_file(dst / "model.safetensors")

    assert quant_config["format_version"] == 2
    assert quant_config["projection_counts"] == {
        convert.FP16: 2,
        convert.W8A16: 3,
        convert.W4A16_PGRP: 10,
    }
    assert stats["projection_counts"] == quant_config["projection_counts"]
    assert stats["n_quantized"] == 13

    for relpath, spec in specs.items():
        weight_key = f"{convert.TEXT_PREFIX}{relpath}.weight"
        scale_name = convert.scale_key(weight_key)
        weight = tensors[weight_key]
        if spec.method == convert.FP16:
            assert weight.dtype == torch.float16
            assert scale_name not in tensors
        elif spec.method == convert.W8A16:
            assert weight.dtype == torch.int8
            assert tensors[scale_name].shape == (weight.shape[0],)
            assert tensors[scale_name].dtype == torch.float16
        else:
            assert weight.dtype == torch.int8
            assert int(weight.min()) >= -8 and int(weight.max()) <= 7
            assert tensors[scale_name].shape == (
                weight.shape[1] // spec.group_size,
                weight.shape[0],
            )
            assert tensors[scale_name].dtype == torch.float16

    assert tensors["model.visual.patch_embed.proj.weight"].dtype == torch.float16


def test_mixed_converter_is_deterministic(tmp_path):
    torch.manual_seed(1)
    src = tmp_path / "fp16"
    _mini_checkpoint(src)
    outputs = []
    for name in ("mixed_a", "mixed_b"):
        dst = tmp_path / name
        convert.convert_checkpoint(src, dst, profile="w4_main_w8_gdn")
        outputs.append(load_file(dst / "model.safetensors"))

    assert outputs[0].keys() == outputs[1].keys()
    for name in outputs[0]:
        assert torch.equal(outputs[0][name], outputs[1][name]), name


def test_w8_decode_profile_keeps_gdn_ba_fp16(tmp_path):
    torch.manual_seed(2)
    src, dst = tmp_path / "fp16", tmp_path / "mixed_w8"
    is_full = _mini_checkpoint(src)

    stats = convert.convert_checkpoint(
        src, dst, profile="w8_main_w8_gdn"
    )

    with (dst / "config.json").open() as f:
        quant_config = json.load(f)["quant_config"]
    specs = convert.resolve_quant_specs(is_full, quant_config)
    tensors = load_file(dst / "model.safetensors")

    assert quant_config["projection_counts"] == {
        convert.FP16: 2,
        convert.W8A16: 13,
        convert.W4A16_PGRP: 0,
    }
    assert stats["projection_counts"] == quant_config["projection_counts"]
    assert stats["n_quantized"] == 13
    for relpath, spec in specs.items():
        weight_key = f"{convert.TEXT_PREFIX}{relpath}.weight"
        scale_name = convert.scale_key(weight_key)
        if spec.method == convert.FP16:
            assert tensors[weight_key].dtype == torch.float16
            assert scale_name not in tensors
        else:
            assert spec.method == convert.W8A16
            assert tensors[weight_key].dtype == torch.int8
            assert tensors[scale_name].shape == (64,)


def test_uniform_v1_converter_remains_supported(tmp_path):
    torch.manual_seed(3)
    src = tmp_path / "fp16"
    _mini_checkpoint(src)

    for method in (convert.W8A16, convert.W4A16_PGRP):
        dst = tmp_path / method
        stats = convert.convert_checkpoint(src, dst, method=method)
        with (dst / "config.json").open() as f:
            quant_config = json.load(f)["quant_config"]

        assert quant_config["format_version"] == 1
        assert quant_config["method"] == method
        assert quant_config["projection_counts"] == {
            convert.FP16: 0,
            convert.W8A16: 15 if method == convert.W8A16 else 0,
            convert.W4A16_PGRP: 15 if method == convert.W4A16_PGRP else 0,
        }
        assert stats["n_quantized"] == 15
        assert precision_recipes.read_record(dst) == ("qwen3_5", convert.produced_recipe(method=method))
    assert precision_recipes.read_record(tmp_path / convert.W8A16) == ("qwen3_5", "w8a16_text")

    # A requested recipe the arguments do not produce fails before any output exists.
    with pytest.raises(ValueError, match="produce qwen3_5/w8a16_text, not qwen3_5/w4a16_mlp_rtn_g32"):
        convert.convert_checkpoint(src, tmp_path / "wrong", method=convert.W8A16, recipe="w4a16_mlp_rtn_g32")
    assert not (tmp_path / "wrong").exists()
    # A record copied with the source sidecars never labels a different conversion.
    precision_recipes.write_record(src, "qwen3_5", "fp16")
    convert.convert_checkpoint(src, tmp_path / "again", profile="w8_main_w8_gdn")
    assert precision_recipes.read_record(tmp_path / "again") is None


def test_mlp_only_cli_preserves_uniform_mlp_codes_and_source_float_precision(tmp_path):
    torch.manual_seed(4)
    src, dst, uniform = tmp_path / "source", tmp_path / "mlp", tmp_path / "uniform"
    is_full = _mini_checkpoint(src)
    source = {key: value.clone() for key, value in load_file(src / "model.safetensors").items()}
    # FP32 non-quantized weights must not lose precision through BF16.
    norm_key = "model.language_model.layers.0.linear_attn.norm.weight"
    source[norm_key] = torch.full((64,), 1.003, dtype=torch.float32)
    save_file(source, src / "model.safetensors", metadata={"format": "pt"})
    assert not torch.equal(source[norm_key].half(), source[norm_key].bfloat16().half())

    assert convert.main(["--src", str(src), "--dst", str(dst),
                         "--recipe", "w4a16_mlp_rtn_g32"]) == 0
    assert precision_recipes.read_record(dst) == ("qwen3_5", "w4a16_mlp_rtn_g32")
    with pytest.raises(SystemExit):
        convert.main(["--src", str(src), "--dst", str(tmp_path / "x"),
                      "--recipe", "w4a16_mlp_rtn_g32", "--group-size", "32"])
    convert.convert_checkpoint(src, uniform, method=convert.W4A16_PGRP)
    tensors = load_file(dst / "model.safetensors")
    uniform_tensors = load_file(uniform / "model.safetensors")
    quant = json.loads((dst / "config.json").read_text())["quant_config"]
    specs = convert.resolve_quant_specs(is_full, quant)
    assert quant["projection_counts"] == {convert.FP16: 9, convert.W8A16: 0,
                                           convert.W4A16_PGRP: 6}
    assert quant["n_quantized"] == 6
    assert specs["layers.0.mlp.gate_proj"].method == convert.W4A16_PGRP
    for key, original in source.items():
        scale = convert.scale_key(key)
        if ".mlp." in key:
            assert torch.equal(tensors[key], uniform_tensors[key]), key
            assert torch.equal(tensors[scale], uniform_tensors[scale]), scale
        else:
            assert tensors[key].dtype == torch.float16
            assert torch.equal(tensors[key], original.half()), key
            assert scale not in tensors
    assert set(tensors) == set(source) | {
        convert.scale_key(key) for key in source if ".mlp." in key}
    assert "quant_config" not in json.loads((src / "config.json").read_text())


def test_mlp_only_2b_scope_and_nondefault_group_rejection(tmp_path):
    is_full = convert.is_full_from_layer_types(
        [convert.LINEAR_ATTENTION] * 3 + [convert.FULL_ATTENTION]) * 6
    quant = convert._quant_config(is_full, None, 32, "w4_mlp_only")
    assert quant["projection_counts"] == {convert.FP16: 114, convert.W8A16: 0,
                                           convert.W4A16_PGRP: 72}
    src, dst = tmp_path / "source", tmp_path / "rejected"
    _mini_checkpoint(src)
    with pytest.raises(ValueError, match="requires the default --group-size=32"):
        convert.convert_checkpoint(src, dst, profile="w4_mlp_only", group_size=64)
    assert not dst.exists()


def test_mlp_w4_attention_gdn_w8_preserves_each_quantizer_output(tmp_path):
    """The mixed recipe must match uniform W4/W8, including small GDN b/a."""
    torch.manual_seed(5)
    src = tmp_path / "source"
    is_full = _mini_checkpoint(src)
    source = {key: value.clone() for key, value in load_file(src / "model.safetensors").items()}
    for role in ("in_proj_b", "in_proj_a"):
        key = f"{convert.TEXT_PREFIX}layers.0.linear_attn.{role}.weight"
        source[key] = source[key][:16].contiguous()
    save_file(source, src / "model.safetensors", metadata={"format": "pt"})
    mixed, w4, w8 = (tmp_path / name for name in ("mixed", "w4", "w8"))
    assert convert.main(["--src", str(src), "--dst", str(mixed),
                         "--profile", "w4_mlp_w8_attn_gdn"]) == 0
    convert.convert_checkpoint(src, w4, method=convert.W4A16_PGRP)
    convert.convert_checkpoint(src, w8, method=convert.W8A16)
    quant = json.loads((mixed / "config.json").read_text())["quant_config"]
    specs = convert.resolve_quant_specs(is_full, quant)
    assert quant["format_version"] == 2
    assert quant["projection_counts"] == {convert.FP16: 0, convert.W8A16: 9,
                                           convert.W4A16_PGRP: 6}
    got = load_file(mixed / "model.safetensors")
    references = {convert.W4A16_PGRP: load_file(w4 / "model.safetensors"),
                  convert.W8A16: load_file(w8 / "model.safetensors")}
    for path, spec in specs.items():
        assert spec.method == (convert.W4A16_PGRP if ".mlp." in path else convert.W8A16)
        key = f"{convert.TEXT_PREFIX}{path}.weight"
        for name in (key, convert.scale_key(key)):
            assert torch.equal(got[name], references[spec.method][name]), name
    assert torch.equal(got["model.visual.patch_embed.proj.weight"],
                       source["model.visual.patch_embed.proj.weight"].half())


def test_mlp_w4_attention_gdn_w8_2b_scope_and_group_rejection(tmp_path):
    is_full = convert.is_full_from_layer_types(
        [convert.LINEAR_ATTENTION] * 3 + [convert.FULL_ATTENTION]) * 6
    quant = convert._quant_config(is_full, None, 32, "w4_mlp_w8_attn_gdn")
    assert quant["projection_counts"] == {convert.FP16: 0, convert.W8A16: 114,
                                           convert.W4A16_PGRP: 72}
    src, dst = tmp_path / "source", tmp_path / "rejected"
    _mini_checkpoint(src)
    with pytest.raises(ValueError, match="requires the default --group-size=32"):
        convert.convert_checkpoint(src, dst, profile="w4_mlp_w8_attn_gdn", group_size=64)
    assert not dst.exists()
