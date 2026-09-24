"""HALO must reject incompatible layouts before loading or allocating resources."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters import halo
from rpu_backend.adapters.halo import image_flow, runtime, text_cache, text_decode
from rpu_backend.adapters.halo import vit_cache, weights


@pytest.mark.parametrize("entry,field,value", [
    ("action", "attn_tp", 4),
    ("text_cache", "attn_tp", 4),
    ("text_decode", "mlp_cores", 4),
    ("image_flow", "mlp_cores", 4),
    ("vit_cache", "attn_tp", 4),
    ("prefix_banks", "attn_tp", 4),
    ("cfg_branch", "attn_tp", 4),
    ("load_branches", "mlp_cores", 4),
    ("load_gen_branch", "attn_tp", 4),
])
def test_entry_rejects_layout_before_resources(monkeypatch, entry, field, value):
    cfg = replace(weights.HaloConfig(), **{field: value})
    touched = []

    def forbidden(*args, **kwargs):
        touched.append("resource")
        raise AssertionError("invalid HALO layout reached a resource boundary")

    for module in (runtime, text_cache, text_decode, image_flow, vit_cache):
        monkeypatch.setattr(module, "_coerce_halo_execution_controller", forbidden)
        monkeypatch.setattr(module, "RPUCache", forbidden)
    for module in (weights, vit_cache):
        monkeypatch.setattr(module, "_HaloTensorStore", forbidden)
    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch, "zeros", forbidden)

    calls = {
        "action": lambda: runtime.build_halo_action_expert(
            "unused", rope_position=0, cfg=cfg),
        "text_cache": lambda: text_cache.build_halo_text_cache("unused", cfg=cfg),
        "text_decode": lambda: text_decode.build_halo_text_decode("unused", cfg=cfg),
        "image_flow": lambda: image_flow.build_halo_image_flow("unused", cfg=cfg),
        "vit_cache": lambda: vit_cache.build_halo_vit_cache("unused", cfg=cfg),
        # A supplied runner bypasses weight loading, so validate its config too.
        "prefix_banks": lambda: halo.build_native_prefix_banks(
            "unused", cfg=cfg, img_uncond_text_pack={},
            text_cache=SimpleNamespace(prefill=forbidden)),
        "cfg_branch": lambda: runtime.build_cfg_branch(
            "test", 0, [None] * cfg.num_layers, [None] * cfg.num_layers,
            rope_position=0, prefix_len=0, epoch=1, cfg=cfg),
        "load_branches": lambda: weights.load_halo_branches("unused", cfg),
        "load_gen_branch": lambda: weights.load_halo_gen_branch("unused", cfg),
    }
    with pytest.raises(ValueError, match=f"HALO {field}=.*expected"):
        calls[entry]()
    assert touched == []


@pytest.mark.parametrize("field,value", [
    ("attn_tp", True), ("attn_tp", 2.0), ("mlp_cores", 8.0),
    ("num_kv_heads", False), ("num_kv_heads", 2.0), ("num_kv_heads", 0),
])
def test_layout_does_not_coerce_invalid_config(field, value):
    cfg = replace(weights.HaloConfig(), **{field: value})
    with pytest.raises(ValueError, match=f"HALO {field}"):
        weights.validate_halo_core_layout(cfg)


def test_direct_action_runner_rejects_layout_before_input_allocation(monkeypatch):
    runner = object.__new__(runtime.HaloStepRunner)
    runner.cfg = replace(weights.HaloConfig(), mlp_cores=4)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid HALO layout allocated its input buffer")

    monkeypatch.setattr(torch, "empty", forbidden)
    with pytest.raises(ValueError, match="HALO mlp_cores="):
        runner.__post_init__()


def test_valid_layout_preserves_config_and_reaches_checkpoint(monkeypatch):
    cfg = replace(weights.HaloConfig(), prefix_len=16)
    original = vars(cfg).copy()
    opened = []

    def missing_checkpoint(path):
        opened.append(path)
        raise FileNotFoundError("expected checkpoint boundary")

    monkeypatch.setattr(weights, "_HaloTensorStore", missing_checkpoint)
    with pytest.raises(FileNotFoundError, match="expected checkpoint boundary"):
        weights.load_halo_branches("unused", cfg)
    assert opened == ["unused"]
    assert vars(cfg) == original
