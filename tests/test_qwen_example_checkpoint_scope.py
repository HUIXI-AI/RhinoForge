"""Real example labels must reject another valid checkpoint precision/size."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def common(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    spec = importlib.util.spec_from_file_location("_common", ROOT / "examples/_common.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_common", module)
    spec.loader.exec_module(module)
    return module


def qwen3_metadata(size, quantized=False):
    from rpu_backend.quant.qwen3_profiles import _W8_METADATA

    layers, hidden, intermediate, heads = {
        "0_6b": (28, 1024, 3072, 16), "1_7b": (28, 2048, 6144, 16),
        "4b": (36, 2560, 9728, 32), "8b": (36, 4096, 12288, 32),
        "14b": (40, 5120, 17408, 40)}[size]
    cfg = dict(model_type="qwen3", architectures=["Qwen3ForCausalLM"],
               num_hidden_layers=layers, hidden_size=hidden, intermediate_size=intermediate,
               num_attention_heads=heads, num_key_value_heads=8, head_dim=128,
               vocab_size=151936, tie_word_embeddings=hidden < 4096)
    if quantized:
        cfg.update(quant_config=copy.deepcopy(_W8_METADATA), tie_word_embeddings=False)
    return cfg


def write_metadata(tmp_path, metadata):
    # Deliberately unrelated to any model/alias directory name.
    checkpoint = tmp_path / "custom-compatible-checkpoint"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "config.json").write_text(json.dumps(metadata))
    return str(checkpoint)


@pytest.mark.parametrize("size", ["0_6b", "1_7b", "4b", "8b", "14b"])
def test_qwen3_w8_declared_scope_accepts_metadata_not_path(common, tmp_path, size):
    cfg = common.load_config(ROOT / f"examples/configs/qwen3/text/{size}/w8a16.toml")
    common.validate_checkpoint_profile(cfg, write_metadata(tmp_path, qwen3_metadata(size, True)))


@pytest.mark.parametrize("size,precision,quantized", [
    ("0_6b", "fp16", False), ("0_6b", "fp16_instruct", False),
    ("4b", "fp16_instruct", False), ("8b", "fp16", False)])
def test_dense_checkpoint_scope_accepts_original_example(common, tmp_path, size, precision, quantized):
    cfg = common.load_config(ROOT / f"examples/configs/qwen3/text/{size}/{precision}.toml")
    common.validate_checkpoint_profile(cfg, write_metadata(tmp_path, qwen3_metadata(size, quantized)))


@pytest.mark.parametrize("precision,quantized", [("fp16", True), ("w8a16", False)])
def test_qwen3_valid_other_precision_cannot_inherit_label(common, tmp_path, precision, quantized):
    cfg = common.load_config(ROOT / f"examples/configs/qwen3/text/0_6b/{precision}.toml")
    with pytest.raises(common.ConfigError, match="precision/scope"):
        common.validate_checkpoint_profile(cfg, write_metadata(tmp_path, qwen3_metadata("0_6b", quantized)))


@pytest.mark.parametrize("field,value", [
    ("quantized_lm_head", False), ("quantized_lm_head", 1),
    ("skip_modules", ["lm_head"]), ("quantized_embed_tokens", True)])
def test_qwen3_w8_head_embedding_and_skip_scope_are_bound(common, tmp_path, field, value):
    cfg = common.load_config(ROOT / "examples/configs/qwen3/text/0_6b/w8a16.toml")
    metadata = qwen3_metadata("0_6b", True)
    metadata["quant_config"][field] = value
    with pytest.raises(common.ConfigError, match="precision/scope"):
        common.validate_checkpoint_profile(cfg, write_metadata(tmp_path, metadata))


def test_another_valid_qwen3_size_cannot_inherit_label(common, tmp_path):
    cfg = common.load_config(ROOT / "examples/configs/qwen3/text/0_6b/w8a16.toml")
    with pytest.raises(common.ConfigError, match="architecture and size"):
        common.validate_checkpoint_profile(cfg, write_metadata(tmp_path, qwen3_metadata("1_7b", True)))


@pytest.mark.parametrize("component", ["text", "vl", "vision"])
def test_qwen3_vl_fp16_cannot_silently_load_runtime_quant(common, tmp_path, component):
    from test_qwen3_vl_runtime_quant_loading import config

    cfg = common.load_config(ROOT / f"examples/configs/qwen3_vl/{component}/2b/fp16.toml")
    metadata = config(2, 8, source=True)
    common.validate_checkpoint_profile(cfg, write_metadata(tmp_path, metadata.to_dict()))
    metadata = config(2, 8)
    with pytest.raises(common.ConfigError, match="precision/scope"):
        common.validate_checkpoint_profile(cfg, write_metadata(tmp_path, metadata.to_dict()))


def test_legacy32_accepts_only_legacy_metadata_not_runtime_w8(common, tmp_path):
    from rpu_backend.quant.load import _QWEN3_VL_32B_W8A16_QUANT_CONFIG
    from test_qwen3_vl_envelope_preflight import runtime_32b_config

    cfg = common.load_config(ROOT / "examples/configs/qwen3_vl/vl/32b/w8a16_legacy.toml")
    runtime = runtime_32b_config(8)
    with pytest.raises(common.ConfigError, match="precision/scope"):
        common.validate_checkpoint_profile(cfg, write_metadata(tmp_path, runtime.to_dict()))
    runtime.quant_config = copy.deepcopy(_QWEN3_VL_32B_W8A16_QUANT_CONFIG)
    common.validate_checkpoint_profile(cfg, write_metadata(tmp_path, runtime.to_dict()))


def test_wrong_qwen3_precision_rejects_before_tokenizer_and_weights(common, tmp_path, monkeypatch):
    import transformers
    import rpu_backend.model_registry

    cfg = common.load_config(ROOT / "examples/configs/qwen3/text/0_6b/w8a16.toml")
    checkpoint = write_metadata(tmp_path, qwen3_metadata("0_6b", False))
    monkeypatch.setattr(rpu_backend.model_registry, "model_path", lambda _: Path(checkpoint))
    tokenizer = Mock(side_effect=AssertionError("must reject before tokenizer/weight loading"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    spec = importlib.util.spec_from_file_location("_scope_text_runner", ROOT / "examples/_text_vision_runners.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    with pytest.raises(common.ConfigError, match="precision/scope"):
        runner._prepare_text_vision(cfg, SimpleNamespace(owner=None))
    tokenizer.assert_not_called()
