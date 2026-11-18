"""Named RTN recipe admission stays separate from generic mixed-v2 conversion."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from test_qwen35_core_profiles import _config
from test_qwen35_dense27_profiles import config27
from rpu_backend.adapters.qwen3_5 import Qwen3_5Adapter, _rpu_qwen3_5_forward
from rpu_backend.runtime import UnsupportedModelError
from rpu_backend.adapters.qwen3_5.cores import validate_core_profile
from rpu_backend.adapters.qwen3_5.quant_scope import (
    is_full_from_layer_types, validate_mlp_only_text_profile,
)
from rpu_backend.quant.convert_qwen3_5 import _quant_config


def _recipe(size=2):
    config = config27() if size == 27 else _config(size)
    config.quant_config = _quant_config(
        is_full_from_layer_types(config.text_config.layer_types), None, 32,
        profile="w4_mlp_only")
    return config


def test_named_mlp_recipe_admits_exact_2b_metadata_and_runtime(monkeypatch):
    monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
    config = _recipe()
    metadata = config.to_dict()
    before = deepcopy(metadata)
    assert validate_mlp_only_text_profile(metadata)
    assert metadata == before
    assert validate_core_profile(config).num_cores == 8
    Qwen3_5Adapter.preflight_execution(config, {"model": {"num_cores": 8}})


@pytest.mark.parametrize("cores", [4, 6, 8])
def test_named_mlp_recipe_admits_official_27b_geometry(cores):
    config = _recipe(27)
    metadata = config.to_dict()
    assert validate_mlp_only_text_profile(metadata)
    from rpu_backend.adapters.qwen3_5.quant_scope import validate_modern_27b_profile
    assert not validate_modern_27b_profile(config, allow_fp16=True)
    topology = validate_core_profile(config, {"model": {"num_cores": cores}})
    assert topology.num_cores == cores


def test_mlp_w4_attention_gdn_w8_keeps_generic_v2_text_admission(monkeypatch):
    monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
    config = _config(2)
    config.quant_config = _quant_config(
        is_full_from_layer_types(config.text_config.layer_types), None, 32,
        profile="w4_mlp_w8_attn_gdn")
    execution = {"model": {"num_cores": 8}, "prefill": {"linear_acc32": False}}
    assert not validate_mlp_only_text_profile(config.to_dict())
    assert validate_core_profile(config, execution).num_cores == 8
    Qwen3_5Adapter.preflight_execution(config, execution)


@pytest.mark.parametrize("size", [0.8, 4, 9])
def test_named_mlp_recipe_does_not_expand_converter_scope_into_other_sizes(size):
    config = _recipe(size)
    with pytest.raises(ValueError, match="exact 2B"):
        validate_mlp_only_text_profile(config.to_dict())
    with pytest.raises(ValueError, match="exact 2B"):
        validate_core_profile(config)
    # The existing arbitrary mixed-v2 admission is independent of the new name.
    config.quant_config["profile"] = "custom_mixed"
    assert validate_core_profile(config).num_cores == 8


@pytest.mark.parametrize("cores", [4, 6])
def test_named_mlp_recipe_accepts_reduced_topology(cores):
    assert validate_core_profile(_recipe(), {"model": {"num_cores": cores}}).num_cores == cores


@pytest.mark.parametrize("mutation", ["scope", "group", "geometry", "semantics"])
def test_named_mlp_recipe_cannot_hide_near_neighbour_behind_label(mutation):
    config = _recipe()
    if mutation == "scope":
        config.quant_config["projection_overrides"]["mlp.gate_proj"] = {"method": "fp16"}
    elif mutation == "group":
        config.quant_config["projection_overrides"]["mlp.gate_proj"]["group_size"] = 128
    elif mutation == "geometry":
        config.text_config.num_attention_heads = 16
    else:
        config.text_config.attn_output_gate = False
    with pytest.raises(ValueError):
        validate_core_profile(config)


@pytest.mark.parametrize("overrides", [{}, {"0": {"default": {"method": "fp16"}}}])
def test_named_mlp_recipe_rejects_unsupported_layer_override_declarations(overrides):
    config = _recipe()
    config.quant_config["layer_overrides"] = overrides
    with pytest.raises(ValueError, match="does not support layer_overrides"):
        validate_mlp_only_text_profile(config.to_dict())
    with pytest.raises(ValueError, match="does not support layer_overrides"):
        validate_core_profile(config)
    config.quant_config["profile"] = "custom_mixed"
    assert not validate_mlp_only_text_profile(config)
    assert validate_core_profile(config).num_cores == 8


@pytest.mark.parametrize("opt_in", [False, True])
def test_named_mlp_recipe_rejects_vision_before_loading(monkeypatch, opt_in):
    execution = {"model": {"num_cores": 8}}
    if opt_in:
        monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    else:
        monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
        execution["vision"] = {"enabled": True}
    with pytest.raises(UnsupportedModelError, match="pure text"):
        Qwen3_5Adapter.preflight_execution(_recipe(), execution)


@pytest.mark.parametrize("kwargs", [
    {"input_ids": torch.ones(2, 1, dtype=torch.long)},
    {"input_ids": torch.ones(1, 1, dtype=torch.long), "pixel_values": torch.ones(1)},
    {"input_ids": torch.ones(1, 1, dtype=torch.long), "mm_token_type_ids": torch.ones(1, 1)},
])
def test_named_mlp_recipe_request_rejects_before_cache_and_embedding(kwargs):
    owner = SimpleNamespace(config=_recipe())
    before = dict(vars(owner))
    with pytest.raises(NotImplementedError, match="batch size 1|pure text"):
        _rpu_qwen3_5_forward(owner, **kwargs)
    assert vars(owner) == before
