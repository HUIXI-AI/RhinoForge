"""Configured Qwen3-VL images size the cold Vision RoPE table and KV cache.

Actual example runner, public loader and adapter; only the checkpoint,
processor and native install are CPU doubles.
"""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters import qwen3_vl
from rpu_backend.adapters.qwen3_vl import vision
from rpu_backend.api import conditional_generation
from rpu_backend.tests.acceptance.test_qwen3_vl_install_lifecycle import install_runtime, _Model  # noqa: F401


class _Loaded(_Model):
    def eval(self):
        return self


def _public_loader(monkeypatch, adapter=qwen3_vl.Qwen3VLAdapter):
    model = _Loaded()
    model.config.architectures = ["Qwen3VLForConditionalGeneration"]
    loads = []
    monkeypatch.setattr(conditional_generation, "AutoConfig",
                        SimpleNamespace(from_pretrained=lambda *_a, **_kw: model.config))
    monkeypatch.setattr(conditional_generation, "AutoModelForImageTextToText", SimpleNamespace(
        from_pretrained=lambda *_a, **_kw: loads.append(True) or model))
    monkeypatch.setattr(conditional_generation, "get_adapter", lambda _arch: adapter)
    return model, loads


def _encode(_processor, images, _inputs):
    # Qwen3-VL processor geometry for sizes already on its 32 px grid
    # (patch 16 x merge 2): one frame of H/16 x W/16 patches per image.
    grid = torch.tensor([[1, image.size[1] // 16, image.size[0] // 16] for image in images])
    tokens = int(grid.prod(-1).sum()) // 4
    return {"input_ids": torch.ones((1, tokens + 8), dtype=torch.long),
            "attention_mask": torch.ones((1, tokens + 8), dtype=torch.long),
            "pixel_values": torch.zeros((int(grid.prod(-1).sum()), 1536)),
            "image_grid_thw": grid}


@pytest.mark.parametrize("sizes_wh, max_hw, max_seq_len", [
    ([(896, 896)], 56, 56 * 56),            # failed on the board with max_hw 48
    ([(640, 480)], 48, 2048),               # fits: original install unchanged
    ([(640, 480), (1024, 512)], 64, 2048),  # widest side, not the largest image
])
def test_configured_images_size_the_cold_vision_install(
        monkeypatch, install_runtime, sizes_wh, max_hw, max_seq_len):
    import transformers
    from examples.common import config as common, generation
    from examples.runners.qwen3_vl import runner
    from rpu_backend import model_registry

    monkeypatch.setattr(common, "validate_checkpoint_profile", lambda *_: None)
    monkeypatch.setattr(model_registry, "model_path", lambda _: "/cache/qwen3-vl")
    monkeypatch.setattr(transformers, "AutoProcessor", SimpleNamespace(
        from_pretrained=lambda *_a, **_kw: SimpleNamespace(tokenizer=None)))
    monkeypatch.setattr(generation, "_image_inputs",
                        lambda _: [SimpleNamespace(size=size) for size in sizes_wh])
    monkeypatch.setattr(generation, "qwen3_vl_encode", _encode)
    model, _loads = _public_loader(monkeypatch)
    config = {"example": {"target": "qwen3_vl", "registry_alias": "qwen3-vl-2b",
                          "profile_id": "qwen3-vl-2b.fp16.vision"},
              "input": {"component": "vision", "images": ["configured.png"] * len(sizes_wh)}}

    state = SimpleNamespace(owner=None)
    _infer, owner = runner.prepare(config, state)

    assert owner is state.owner is model
    assert state.input_geometry["source_image_sizes_wh"] == [list(size) for size in sizes_wh]
    installed = install_runtime["vision_kwargs"]
    assert (installed["max_hw"], installed["max_seq_len"]) == (max_hw, max_seq_len)


def test_omitted_grids_keep_the_original_install(monkeypatch, install_runtime):
    model, _loads = _public_loader(monkeypatch)
    assert conditional_generation.RPUModelForConditionalGeneration.from_pretrained(
        "/cache/qwen3-vl", device="rpu") is model
    installed = install_runtime["vision_kwargs"]
    assert (installed["max_hw"], installed["max_seq_len"]) == (48, 2048)


@pytest.mark.parametrize("grid, message", [
    ([[1, 64, 66]], "4224 patches exceeds the native 4096-patch limit"),
    ([[1, 0, 32]], r"\[N,3\] positive integer"),
    ([[1.0, 32.0, 32.0]], r"\[N,3\] positive integer"),
    ([1, 32, 32], r"\[N,3\] positive integer"),
])
def test_unsupported_grids_fail_before_checkpoint_weights(monkeypatch, install_runtime, grid, message):
    _model, loads = _public_loader(monkeypatch)
    with pytest.raises(ValueError, match=message):
        conditional_generation.RPUModelForConditionalGeneration.from_pretrained(
            "/cache/qwen3-vl", device="rpu", image_grid_thw=grid)
    assert loads == [] and install_runtime["events"] == []


def test_native_ceiling_is_admitted_exactly():
    assert vision.qwen3_vl_vision_capacity([[1, 64, 64]]) == (64, 4096)
    assert vision.qwen3_vl_vision_capacity([[1, 128, 32]]) == (128, 4096)


def test_other_adapters_reject_vision_grids_before_weights(monkeypatch, install_runtime):
    class Adapter:
        pass

    _model, loads = _public_loader(monkeypatch, Adapter)
    with pytest.raises(ValueError, match="image_grid_thw sizes the Qwen3-VL Vision install"):
        conditional_generation.RPUModelForConditionalGeneration.from_pretrained(
            "/cache/qwen3-vl", image_grid_thw=[[1, 56, 56]])
    assert loads == []
