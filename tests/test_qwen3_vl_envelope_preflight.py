"""Reject excessive text chunks at the public loader boundary, before weights."""
import copy

import pytest

from rpu_backend.adapters.qwen3_vl import Qwen3VLAdapter
from rpu_backend.api import RPUModelForConditionalGeneration
from rpu_backend.api import conditional_generation
from rpu_backend.api.errors import UnsupportedModelError
from rpu_backend.quant import convert_qwen3_vl, load
from test_qwen3_vl_runtime_quant_loading import config


def runtime_32b_config(bits):
    cfg = config(4, bits)
    text, vision = cfg.text_config, cfg.vision_config
    text.hidden_size, text.intermediate_size = 5120, 25600
    text.num_hidden_layers, text.num_attention_heads = 64, 64
    vision.hidden_size, vision.intermediate_size, vision.depth = 1152, 4304, 27
    vision.out_hidden_size, vision.deepstack_visual_indexes = 5120, [8, 16, 24]
    cfg.quant_config = convert_qwen3_vl.runtime_quant_config(5120, bits)
    return cfg


@pytest.mark.parametrize("execution", [
    {"prefill": {"chunk_size": 80}},
    {"prefill": {"chunk_size": 128}},
    {"prefill": {"chunk_size": 256}},
    {"components": {"language_model": {"prefill": {"chunk_size": 80}}}},
    {"components": {"language_model": {"prefill": {"chunk_size": 256}}}},
])
@pytest.mark.parametrize("device", [None, "rpu"])
def test_legacy_32b_excess_chunk_rejects_before_any_checkpoint_loader(monkeypatch, execution, device):
    cfg = runtime_32b_config(8)
    cfg.quant_config = copy.deepcopy(load._QWEN3_VL_32B_W8A16_QUANT_CONFIG)
    assert load.is_qwen3_vl_32b_w8a16_config(cfg)
    monkeypatch.setenv("QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED", "1")
    # Controlled metadata admission does not widen this recipe's own chunk bound.
    Qwen3VLAdapter.preflight(cfg)
    monkeypatch.setattr(conditional_generation.AutoConfig, "from_pretrained", lambda *a, **k: cfg)

    def forbidden(*args, **kwargs):
        pytest.fail("excessive chunk reached checkpoint loading or staging")

    monkeypatch.setattr(load, "load_w8a16_imagetext", forbidden)
    monkeypatch.setattr(load, "stage_w8a16_imagetext_for_rpu", forbidden)
    monkeypatch.setattr(conditional_generation.AutoModelForImageTextToText, "from_pretrained", forbidden)
    with pytest.raises(UnsupportedModelError, match="planning ceiling 64"):
        RPUModelForConditionalGeneration.from_pretrained(
            "unused-local-checkpoint", device=device, rpu_execution=execution,
            local_files_only=True)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("chunk", ["auto", 64])
def test_runtime_32b_keeps_its_distinct_existing_envelope(monkeypatch, bits, chunk):
    monkeypatch.delenv("QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED", raising=False)
    cfg = runtime_32b_config(bits)
    assert not load.is_qwen3_vl_32b_w8a16_config(cfg)
    Qwen3VLAdapter.preflight(cfg)
    Qwen3VLAdapter.preflight_execution(cfg, {"prefill": {"chunk_size": chunk}})


@pytest.mark.parametrize("size", [2, 4])
def test_existing_dense_auto_envelope_remains_admitted(size):
    cfg = config(size, source=True)
    Qwen3VLAdapter.preflight(cfg)
    Qwen3VLAdapter.preflight_execution(cfg, None)
