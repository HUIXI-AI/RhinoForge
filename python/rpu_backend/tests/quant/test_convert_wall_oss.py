import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from rpu_backend.quant.convert_wall_oss import (
    WALL_OSS_QUANT_SUFFIXES,
    convert_checkpoint,
)
from rpu_backend.adapters.wall_oss.checkpoint_validation import (
    validate_rtc_w4_checkpoint_pair,
)


def _mini_ckpt(d):
    # one layer, both experts, wall-oss-shaped (down e0 K=11008, down e1 K=2048).
    t = {
        "model.layers.0.self_attn.qkv_proj_experts.0.weight": torch.randn(2560, 2048),
        "model.layers.0.self_attn.qkv_proj_experts.0.bias": torch.randn(2560),
        "model.layers.0.self_attn.o_proj_experts.0.weight": torch.randn(2048, 2048),
        "model.layers.0.moe.experts.0.gate_up_proj.weight": torch.randn(22016, 2048),
        "model.layers.0.moe.experts.0.down_proj.weight": torch.randn(2048, 11008),
        "model.layers.0.self_attn.qkv_proj_experts.1.weight": torch.randn(2560, 1024),
        "model.layers.0.moe.experts.1.down_proj.weight": torch.randn(1024, 2048),
        "model.norms.0.weight": torch.randn(2048),
    }
    d.mkdir(parents=True)
    save_file(t, str(d / "model.safetensors"), metadata={"format": "pt"})
    json.dump({"hidden_size": 2048}, (d / "config.json").open("w"))


def test_convert_wall_oss_mixed(tmp_path):
    src, dst = tmp_path / "fp16", tmp_path / "w4a16"
    _mini_ckpt(src)
    convert_checkpoint(src, dst)

    cfg = json.load((dst / "rpu_quant_config.json").open())
    assert cfg["method"] == "w4a16"
    assert cfg["source"] == "fp16"
    assert cfg["mode"] == "group_wise_symmetric"
    assert cfg["group_size"] == 32
    assert cfg["kernel"] == "w4a16_pgrp"
    # No forced-int8: expert-0 down_proj is now int4 (made legal by the runtime
    # INTER-pad). int8_suffixes is empty unless --keep-int8 is passed.
    assert cfg["int8_suffixes"] == []
    assert "moe.experts.0.down_proj.weight" in cfg["int4_suffixes"]
    assert "moe.experts.1.down_proj.weight" in cfg["int4_suffixes"]

    out = load_file(str(dst / "model.safetensors"))
    # everything int4: values fit [-8,7] in the int8 container.
    q0 = out["model.layers.0.self_attn.qkv_proj_experts.0.weight"]
    assert q0.dtype == torch.int8 and int(q0.abs().max()) <= 8
    d0 = out["model.layers.0.moe.experts.0.down_proj.weight"]
    assert d0.dtype == torch.int8 and int(d0.abs().max()) <= 8  # int4 (expert-0 down)
    d1 = out["model.layers.0.moe.experts.1.down_proj.weight"]
    assert int(d1.abs().max()) <= 8  # int4

    # scales present + fp16; non-quant tensors copied as fp16
    q0_scale = out["model.layers.0.self_attn.qkv_proj_experts.0.weight_scale"]
    assert q0_scale.dtype == torch.float16
    assert q0_scale.dim() == 2
    assert out["model.norms.0.weight"].dtype == torch.float16
    assert out["model.layers.0.self_attn.qkv_proj_experts.0.bias"].dtype == torch.float16


def test_w8a16_quant_config_has_no_abs_path():
    # regression: the separately-delivered rpu_quant_config.json must not leak the
    # build-host absolute src path.
    from pathlib import Path

    from rpu_backend.quant.convert_wall_oss_w8a16 import _quant_config

    cfg = _quant_config(Path("/path/to/local/user/some_ckpt_dir"), (), [])
    assert cfg["source"] == "some_ckpt_dir"
    assert "/" not in cfg["source"]
    assert "/path/to/local" not in json.dumps(cfg)


def test_rtc_w4_checkpoint_pair_requires_exact_pgrp_composition(tmp_path):
    fp16, w4 = tmp_path / "fp16", tmp_path / "w4"
    fp16.mkdir()
    w4.mkdir()
    config = {"hidden_size": 2048, "causal_action_attention_mask": False}
    json.dump(config, (fp16 / "config.json").open("w"))
    json.dump(config, (w4 / "config.json").open("w"))
    quant = {
        "method": "w4a16",
        "source": "fp16",
        "mode": "group_wise_symmetric",
        "group_size": 32,
        "int4_suffixes": [
            suffix
            for suffix in WALL_OSS_QUANT_SUFFIXES
            if suffix != "moe.experts.0.down_proj.weight"
        ],
        "int8_suffixes": ["moe.experts.0.down_proj.weight"],
    }
    json.dump(quant, (w4 / "rpu_quant_config.json").open("w"))

    assert validate_rtc_w4_checkpoint_pair(w4, fp16) == (config, config)

    quant["int8_suffixes"].append("self_attn.qkv_proj_experts.0.weight")
    json.dump(quant, (w4 / "rpu_quant_config.json").open("w"))
    with pytest.raises(ValueError, match="expert-0 down_proj.*retained as INT8"):
        validate_rtc_w4_checkpoint_pair(w4, fp16)

    quant["int8_suffixes"] = ["moe.experts.0.down_proj.weight"]
    quant["int4_suffixes"] = quant["int4_suffixes"][:-1]
    json.dump(quant, (w4 / "rpu_quant_config.json").open("w"))
    with pytest.raises(ValueError, match="exact seven group-wise INT4"):
        validate_rtc_w4_checkpoint_pair(w4, fp16)
