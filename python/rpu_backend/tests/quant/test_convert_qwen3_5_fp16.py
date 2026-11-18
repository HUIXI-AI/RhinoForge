"""FP16 export preserves stored values and publishes only complete checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "quant"))
import convert_qwen3_5 as convert


def checkpoint(path):
    path.mkdir()
    tensors = {f"{convert.TEXT_PREFIX}{name}.weight": torch.arange(32).reshape(4, 8).bfloat16()
               for name in convert.resolve_quant_specs([False])}
    tensors["model.language_model.layers.0.linear_attn.norm.weight"] = torch.full((8,), 1.003)
    tensors["model.visual.patch_embed.proj.weight"] = torch.ones(4, 8, dtype=torch.float16)
    tensors["model.language_model.position_ids"] = torch.arange(8, dtype=torch.int64)
    tensors["model.visual.valid_mask"] = torch.tensor([True, False])
    names = sorted(tensors)
    mapping = {name: f"model-{i % 2 + 1}.safetensors" for i, name in enumerate(names)}
    for shard in sorted(set(mapping.values())):
        save_file({k: v for k, v in tensors.items() if mapping[k] == shard}, path / shard)
    config = {"architectures": [convert.ARCHITECTURE], "dtype": "bfloat16",
              "text_config": {"num_hidden_layers": 1, "layer_types": [convert.LINEAR_ATTENTION],
                              "dtype": "bfloat16"}, "vision_config": {"torch_dtype": "bfloat16"}}
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 1}, "weight_map": mapping}))
    (path / "tokenizer_config.json").write_text('{"model_max_length": 128}')
    (path / "SHA256SUMS").write_text("old source checksum; must not describe converted bytes\n")
    return tensors, mapping


def test_fp16_cli_exports_exact_casts_and_bound_source(tmp_path):
    src, dst = tmp_path / "source", tmp_path / "fp16"
    tensors, mapping = checkpoint(src)
    source_config = (src / "config.json").read_bytes()
    assert convert.main(["--src", str(src), "--dst", str(dst), "--method", "fp16"]) == 0
    config = json.loads((dst / "config.json").read_text())
    assert "quant_config" not in config and "quantization_config" not in config
    for owner in (config, config["text_config"], config["vision_config"]):
        assert owner["dtype"] == owner["torch_dtype"] == "float16"
    output = {}
    for shard in set(mapping.values()):
        output.update(load_file(dst / shard))
    assert output.keys() == tensors.keys()
    for name, original in tensors.items():
        expected = original.half() if original.is_floating_point() else original
        assert output[name].dtype == expected.dtype
        assert torch.equal(output[name], expected), name
    norm = tensors["model.language_model.layers.0.linear_attn.norm.weight"]
    assert not torch.equal(norm.half(), norm.bfloat16().half())
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == mapping
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in output.values())
    provenance = config["weight_conversion"]
    assert provenance["method"] == "fp16"
    expected_files = set(mapping.values()) | {"config.json", "model.safetensors.index.json", "tokenizer_config.json"}
    assert set(provenance["source_files_sha256"]) == expected_files
    for name, digest in provenance["source_files_sha256"].items():
        assert digest == hashlib.sha256((src / name).read_bytes()).hexdigest()
    assert provenance["source_sha256"] == hashlib.sha256(json.dumps(provenance["source_files_sha256"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert not (dst / "SHA256SUMS").exists()
    assert (src / "config.json").read_bytes() == source_config


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 70000.0])
def test_fp16_rejects_nonfinite_and_overflow_without_publishing(tmp_path, bad):
    src, dst = tmp_path / "source", tmp_path / "fp16"
    _, mapping = checkpoint(src)
    name = "model.language_model.layers.0.linear_attn.norm.weight"
    shard = src / mapping[name]
    tensors = load_file(shard)
    tensors[name] = torch.full((8,), bad, dtype=torch.float32)
    save_file(tensors, shard)
    with pytest.raises(ValueError, match="nonfinite value or FP16 overflow"):
        convert.convert_checkpoint(src, dst, method="fp16")
    assert not dst.exists() and not list(tmp_path.glob(".fp16.tmp-*"))


@pytest.mark.parametrize("owner,key", [("root", "quant_config"), ("root", "quantization_config"),
                                       ("text_config", "quant_config"), ("vision_config", "quantization_config")])
def test_fp16_rejects_quantized_input_declarations(tmp_path, owner, key):
    src, dst = tmp_path / "source", tmp_path / "fp16"
    checkpoint(src)
    config = json.loads((src / "config.json").read_text())
    (config if owner == "root" else config[owner])[key] = {"method": "int8"}
    (src / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="quant"):
        convert.convert_checkpoint(src, dst, method="fp16")
    assert not dst.exists()


def test_fp16_rejects_undeclared_integer_weights_but_preserves_buffers(tmp_path):
    src, dst = tmp_path / "source", tmp_path / "fp16"
    _, mapping = checkpoint(src)
    name = next(n for n in mapping if n.endswith(".weight"))
    p = src / mapping[name]
    tensors = load_file(p)
    tensors[name] = tensors[name].to(torch.int8)
    save_file(tensors, p)
    with pytest.raises(TypeError, match="floating weights"):
        convert.convert_checkpoint(src, dst, method="fp16")
    assert not dst.exists()


def test_fp16_rejects_incomplete_index_and_preserves_existing_destination(tmp_path):
    src, dst = tmp_path / "source", tmp_path / "fp16"
    checkpoint(src)
    index_path = src / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].pop("model.visual.valid_mask")
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="disagree with weight map"):
        convert.convert_checkpoint(src, dst, method="fp16")
    assert not dst.exists()
    dst.mkdir()
    (dst / "existing").write_text("keep")
    with pytest.raises(FileExistsError):
        convert.convert_checkpoint(src, dst, method="fp16")
    assert (dst / "existing").read_text() == "keep"


def test_fp16_shard_write_failure_removes_partial_output(tmp_path, monkeypatch):
    src, dst = tmp_path / "source", tmp_path / "fp16"
    checkpoint(src)
    original = convert.save_file
    calls = []
    def fail_second(*args, **kwargs):
        calls.append(args[1])
        if len(calls) == 2:
            raise OSError("disk full")
        return original(*args, **kwargs)
    monkeypatch.setattr(convert, "save_file", fail_second)
    with pytest.raises(OSError, match="disk full"):
        convert.convert_checkpoint(src, dst, method="fp16")
    assert len(calls) == 2
    assert not dst.exists() and not list(tmp_path.glob(".fp16.tmp-*"))
