"""Pi0.5 cold configuration and padding preserve the original model function."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.api._execution import resolve_pi05_execution_components
from rpu_backend.adapters.pi05.cores import (
    core_topology, pad_decoder_mlp_inplace, physical_intermediate_size,
    vision_batch_enabled,
)


@pytest.mark.parametrize("cores", (4, 6, 8))
def test_pi05_root_core_choice_preserves_component_authority(cores, monkeypatch):
    monkeypatch.setenv("RPU_PI05_SIGLIP_BATCH", "1")
    root, vision, text, action = resolve_pi05_execution_components({
        "model": {"num_cores": cores},
        "components": {"language_model": {"prefill": {"chunk_size": 400}}},
    }, entry_point="test")
    topology = core_topology(root)
    assert (topology.num_cores, topology.attn_tp, topology.mlp_tp, topology.physical_kv_cores) == (
        cores, 8 if cores == 8 else 4, cores, 8)
    assert text == {"prefill": {"chunk_size": 400}}
    assert vision == {"vision": {"chunk_size": "auto"}}
    assert action == {"action": {"chunk_size": "auto"}}
    assert vision_batch_enabled(root) is (cores == 8)


@pytest.mark.parametrize("value", (5, True, 4.0, "6"))
def test_pi05_invalid_core_rejects_before_loading(value):
    with pytest.raises(ValueError):
        resolve_pi05_execution_components({"model": {"num_cores": value}}, entry_point="test")


def _decoder(dtype, layers=1):
    generator = torch.Generator().manual_seed(12)
    result = []
    for _ in range(layers):
        mlp = torch.nn.Module()
        for name, shape in (("gate_proj", (4096, 16)), ("up_proj", (4096, 16)),
                            ("down_proj", (16, 4096))):
            leaf = torch.nn.Linear(shape[1], shape[0], bias=False)
            value = torch.randn(shape, generator=generator) * .02
            if dtype == torch.int8:
                value = (value * 100).round().to(dtype)
                leaf.register_buffer("weight_scale", torch.full((shape[0],), .01, dtype=torch.float16))
            leaf.weight = torch.nn.Parameter(value.to(dtype), requires_grad=False)
            setattr(mlp, name, leaf)
        result.append(SimpleNamespace(mlp=mlp))
    return SimpleNamespace(config=SimpleNamespace(intermediate_size=4096), layers=result)


def _output(mlp, x):
    def weight(leaf):
        result = leaf.weight.float()
        if result.shape[0] and hasattr(leaf, "weight_scale"):
            result = result * leaf.weight_scale.float()[:, None]
        return result
    gate = torch.nn.functional.gelu(x @ weight(mlp.gate_proj).T, approximate="tanh")
    return (gate * (x @ weight(mlp.up_proj).T)) @ weight(mlp.down_proj).T


@pytest.mark.parametrize("dtype,physical,prefix", ((torch.float16, 4128, 16416), (torch.int8, 4224, 16512)))
def test_pi05_mlp6_cold_padding_preserves_output_and_scales(dtype, physical, prefix):
    model = _decoder(dtype)
    x = torch.randn(3, 16, generator=torch.Generator().manual_seed(13))
    before = _output(model.layers[0].mlp, x)
    down_scale = getattr(model.layers[0].mlp.down_proj, "weight_scale", None)
    pad_decoder_mlp_inplace(model, core_topology({"model": {"num_cores": 6}}))
    assert model.config.intermediate_size == 4096
    assert model.layers[0].mlp.gate_proj.weight.shape == (physical, 16)
    assert physical_intermediate_size(16384, 6, dtype) == prefix
    torch.testing.assert_close(_output(model.layers[0].mlp, x), before, atol=1e-7, rtol=1e-5)
    if dtype == torch.int8:
        assert model.layers[0].mlp.down_proj.weight_scale is down_scale
        assert torch.equal(model.layers[0].mlp.gate_proj.weight_scale[4096:], torch.ones(physical-4096, dtype=torch.float16))
    with pytest.raises(ValueError, match="cold logical"):
        pad_decoder_mlp_inplace(model, core_topology({"model": {"num_cores": 6}}))


def test_pi05_padding_preflights_late_invalid_layer_before_mutation():
    model = _decoder(torch.int8, layers=2)
    first = model.layers[0].mlp.gate_proj.weight
    model.layers[1].mlp.up_proj.weight_scale = torch.ones(1, dtype=torch.float16)
    with pytest.raises(ValueError, match="output-channel scales"):
        pad_decoder_mlp_inplace(model, core_topology({"model": {"num_cores": 6}}))
    assert model.layers[0].mlp.gate_proj.weight is first
    assert first.shape == (4096, 16)


def test_pi05_hot_core_change_fails_before_mutating_or_native_calls():
    from rpu_backend.adapters.pi05 import Pi05Adapter

    execution = {"model": {"num_cores": 4}}
    adapter = object.__new__(Pi05Adapter)
    adapter._rpu_is_ready = True
    adapter._rpu_execution = execution
    adapter._legacy_vlm_chunk_size = 0
    adapter._closed = True
    with pytest.raises(ValueError, match="cold-only"):
        adapter.validate_execution_reconfigure({"model": {"num_cores": 6}})
    assert adapter._rpu_execution is execution


@pytest.mark.parametrize("stage,component", (("vision", "vision_encoder"), ("prefill", "language_model"), ("action", "action_expert")))
@pytest.mark.parametrize("mode", (False, True))
def test_pi05_accumulation_component_override_and_cold_reconfigure(stage, component, mode, monkeypatch):
    from rpu_backend.adapters.pi05 import Pi05Adapter
    monkeypatch.setenv("RPU_LINEAR_ACC32", "1")
    default = resolve_pi05_execution_components({}, entry_point="test")
    assert all(not part.get(key, {}).get("linear_acc32", False)
               for part, key in zip(default[1:], ("vision", "prefill", "action")))
    execution = {stage: {"linear_acc32": not mode}, "components": {
        component: {stage: {"linear_acc32": mode}}}}
    resolved = resolve_pi05_execution_components(execution, entry_point="test")
    index = ("vision", "prefill", "action").index(stage) + 1
    assert resolved[index][stage]["linear_acc32"] is mode
    adapter = object.__new__(Pi05Adapter)
    adapter._rpu_is_ready = True
    adapter._rpu_execution = execution
    adapter._legacy_vlm_chunk_size = 0
    adapter._closed = True
    with pytest.raises(ValueError, match="linear_acc32 is cold-only"):
        adapter.validate_execution_reconfigure({stage: {"linear_acc32": not mode}})
    assert adapter._rpu_execution is execution
