"""Offline mixed-E4M3 serialization through the real CPU checkpoint loader."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen3_5MoeConfig

from rpu_backend.adapters.qwen3_5_moe import loader, quant_scope
from rpu_backend.quant import convert_qwen3_5_moe_fp8 as converter
from rpu_backend.quant.fp8 import dequantize_fp8_e4m3_per_output


def source(tmp_path, *, sharded=True):
    directory = tmp_path / "parent"
    directory.mkdir()
    config = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "tie_word_embeddings": False,
        "text_config": {
            "num_hidden_layers": 2, "num_experts": 2, "num_experts_per_tok": 1,
            "hidden_size": 16, "moe_intermediate_size": 8,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
            "attn_output_gate": True,
            "linear_num_key_heads": 2, "linear_key_head_dim": 4,
            "linear_num_value_heads": 2, "linear_value_head_dim": 8,
            "layer_types": ["linear_attention", "full_attention"],
            "dtype": "bfloat16",
        },
        "vision_config": {"dtype": "bfloat16"},
    }
    (directory / "config.json").write_text(json.dumps(config))
    for filename, value in {
        "generation_config.json": {"eos_token_id": [11, 12]},
        "tokenizer_config.json": {"chat_template": "preserve me"},
        "preprocessor_config.json": {"patch_size": 16},
    }.items():
        (directory / filename).write_text(json.dumps(value))
    tensors = {}
    for name in quant_scope.expected_fp8_weight_names(config["text_config"]["layer_types"]):
        if name.endswith("gate_up_proj"):
            shape = (2, 16, 16)
        elif name.endswith("down_proj"):
            shape = (2, 16, 8)
        else:
            rows = 32 if name.endswith(("q_proj.weight", "in_proj_qkv.weight")) else (
                8 if name.endswith(("k_proj.weight", "v_proj.weight")) else 16)
            shape = (rows, 16)
        # Exactly representable E4M3 values with nonuniform per-output scales.
        row = torch.tensor([-448., -224., 0., 224., 448., 0., -1., 1.])
        row = row.repeat(shape[-1] // 8)
        scales = torch.arange(1, 1 + torch.tensor(shape[:-1]).prod().item()).reshape(shape[:-1]) / 4
        tensors[name] = (scales.unsqueeze(-1) * row).bfloat16()
    tensors.update({
        "model.language_model.layers.0.mlp.gate.weight": torch.ones(2, 16).bfloat16(),
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": torch.ones(16, 8),
        "model.language_model.layers.0.linear_attn.in_proj_a.weight": torch.ones(2, 16),
        "model.language_model.layers.0.linear_attn.conv1d.weight": torch.ones(32, 1, 4),
        "model.language_model.norm.weight": torch.full((16,), 0.25),
        "model.language_model.embed_tokens.weight": torch.ones(32, 16).bfloat16(),
        "lm_head.weight": torch.ones(32, 16).bfloat16(),
        "model.visual.patch_embed.proj.weight": torch.ones(16, 3, 2, 2),
        "mtp.layers.0.mlp.experts.gate_up_proj": torch.ones(2, 16, 16).bfloat16(),
    })
    groups = [dict(list(tensors.items())[::2]), dict(list(tensors.items())[1::2])] if sharded else [tensors]
    weight_map = {}
    for number, values in enumerate(groups):
        filename = f"part-{number}.safetensors" if sharded else "model.safetensors"
        save_file(values, directory / filename)
        weight_map.update({key: filename for key in values})
    if sharded:
        (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return directory, tensors


class Skeleton(torch.nn.Module):
    def __init__(self, tensors):
        super().__init__()
        for name, tensor in tensors.items():
            parts = name.split(".")
            module = self
            for part in parts[:-1]:
                if part not in module._modules:
                    module.add_module(part, torch.nn.Module())
                module = module._modules[part]
            module.register_parameter(parts[-1], torch.nn.Parameter(torch.empty(tensor.shape)))

    def tie_weights(self):
        pass


@pytest.mark.parametrize("sharded,chunk", [(True, 1), (False, 3)])
def test_convert_and_load_preserves_codes_scales_and_sidecars(tmp_path, monkeypatch, sharded, chunk):
    src, original = source(tmp_path, sharded=sharded)
    # Shrink geometry only; keep the real scope, format and loader machinery.
    monkeypatch.setattr(converter, "validate_exact_profile", lambda *a, **kw: None)
    dst = tmp_path / "output"
    stats = converter.convert_checkpoint(src, dst, expert_chunk=chunk)
    cfg = Qwen3_5MoeConfig.from_dict(json.loads((dst / "config.json").read_text()))
    quant = quant_scope.validate_quant_config(cfg.quant_config, cfg.text_config.layer_types)
    assert cfg.text_config.dtype == cfg.vision_config.dtype == torch.float16
    assert stats["n_quantized"] == 11
    assert stats["n_shards"] == (2 if sharded else 1)
    monkeypatch.setattr(loader, "parse_quant_config", lambda _: quant)
    monkeypatch.setattr(loader.AutoModelForImageTextToText, "from_config", lambda _: Skeleton(original))
    model = loader.load_qwen3_5_moe_model(dst, config=cfg)
    parameters = dict(model.named_parameters())
    for name, expected in original.items():
        actual = parameters[name]
        assert actual.device.type == "cpu"
        if name in quant["quantized_tensors"]:
            module_name, parameter_name = name.rsplit(".", 1)
            scale = getattr(model.get_submodule(module_name), parameter_name + "_scale")
            assert actual.dtype == torch.uint8 and scale.dtype == torch.float16
            assert torch.equal(dequantize_fp8_e4m3_per_output(actual, scale), expected.float())
        else:
            assert actual.dtype == torch.float16
            assert torch.equal(actual, expected.half())
    assert model.generation_config.eos_token_id == [11, 12]
    for filename in ("generation_config.json", "tokenizer_config.json", "preprocessor_config.json"):
        assert (src / filename).read_bytes() == (dst / filename).read_bytes()
    if sharded:
        index = json.loads((dst / "model.safetensors.index.json").read_text())
        payload_bytes = 0
        for filename in set(index["weight_map"].values()):
            tensors = load_file(dst / filename)
            assert set(tensors) == {k for k, v in index["weight_map"].items() if v == filename}
            payload_bytes += sum(t.numel() * t.element_size() for t in tensors.values())
        assert index["metadata"]["total_size"] == payload_bytes == stats["bytes_out"]
    with pytest.raises(FileExistsError):
        converter.convert_checkpoint(src, dst)


@pytest.mark.parametrize("fault,match", [
    ("shape", "shape .* must be"), ("mixer_shape", "shape .* must be"),
    ("nan", "NaN or Inf"),
    ("overflow", "overflows FP16"), ("missing", "complete mixed-FP8 inventory"),
    ("index", "inventory mismatch"), ("extra_layer", "complete mixed-FP8 inventory"),
    ("quantized", "unquantized"), ("fp8_dtype", "source tensor must be FP16/BF16/FP32"),
])
def test_bad_sources_fail_without_publishing_output(tmp_path, monkeypatch, fault, match):
    src, _ = source(tmp_path)
    monkeypatch.setattr(converter, "validate_exact_profile", lambda *a, **kw: None)
    index_path = src / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    if fault == "overflow":
        name = "model.language_model.norm.weight"
    elif fault == "mixer_shape":
        name = "model.language_model.layers.1.self_attn.q_proj.weight"
    filename = index["weight_map"][name]
    tensors = load_file(src / filename)
    if fault == "shape":
        tensors[name] = tensors[name][:, :-1].contiguous()
    elif fault == "mixer_shape":
        tensors[name] = tensors[name][:-1].contiguous()
    elif fault == "nan":
        tensors[name][0, 0, 0] = float("nan")
    elif fault == "overflow":
        tensors[name] = torch.full((16,), 1e10)
    elif fault == "missing":
        tensors.pop(name)
        index["weight_map"].pop(name)
    elif fault == "index":
        index["weight_map"][name] = "part-1.safetensors" if filename == "part-0.safetensors" else "part-0.safetensors"
    elif fault == "extra_layer":
        new_name = name.replace("layers.0", "layers.2")
        tensors[new_name] = tensors.pop(name)
        index["weight_map"][new_name] = index["weight_map"].pop(name)
    elif fault == "fp8_dtype":
        tensors[name] = tensors[name].to(torch.float8_e4m3fn)
    else:
        config = json.loads((src / "config.json").read_text())
        config["text_config"]["quantization_config"] = None
        (src / "config.json").write_text(json.dumps(config))
    save_file(tensors, src / filename)
    index_path.write_text(json.dumps(index))
    with pytest.raises((ValueError, TypeError), match=match):
        converter.convert_checkpoint(src, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output.tmp-*"))


def test_missing_sidecar_and_nested_output_fail_before_conversion(tmp_path, monkeypatch):
    src, _ = source(tmp_path)
    monkeypatch.setattr(converter, "validate_exact_profile", lambda *a, **kw: None)
    with pytest.raises(ValueError, match="outside the source"):
        converter.convert_checkpoint(src, src / "output")
    (src / "generation_config.json").unlink()
    with pytest.raises(FileNotFoundError, match="generation_config"):
        converter.convert_checkpoint(src, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_exact_unquantized_config_is_accepted():
    from test_qwen35_moe_python_integration import _exact_config

    config = _exact_config()
    config.pop("quant_config")
    layers, experts, intermediate, hidden, layer_types = converter._validate_config(config)
    assert (layers, experts, intermediate, hidden) == (40, 256, 512, 2048)
    assert len(quant_scope.expected_fp8_weight_names(layer_types)) == 210


def test_interrupt_does_not_publish_partial_checkpoint(tmp_path, monkeypatch):
    src, _ = source(tmp_path)
    monkeypatch.setattr(converter, "validate_exact_profile", lambda *a, **kw: None)
    save = converter.save_file

    def interrupt_after_save(*args, **kwargs):
        save(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(converter, "save_file", interrupt_after_save)
    with pytest.raises(KeyboardInterrupt):
        converter.convert_checkpoint(src, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output.tmp-*"))


def test_standalone_cli_runs_on_cpu_and_rejects_wrong_profile(tmp_path):
    src, _ = source(tmp_path)
    script = Path(converter.__file__).resolve()
    repo_root = script.parents[3].resolve()
    excluded = {repo_root, (repo_root / "python").resolve()}
    dep_paths = [
        p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if p and Path(p).resolve() not in excluded
    ]
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONNOUSERSITE="1")
    if dep_paths:
        env["PYTHONPATH"] = os.pathsep.join(dep_paths)
    else:
        env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-s", "-P", str(script), "--src", str(src),
                             "--dst", str(tmp_path / "output"), "--threads", "1"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode != 0
    assert "exact 35B-A3B" in result.stdout + result.stderr
    assert not (tmp_path / "output").exists()
