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
    monkeypatch.setenv("RPU_PI05_VISION_KV_SPM_ONLY", "1")
    monkeypatch.setenv("RPU_PI05_VISION_Q_RESIDENT", "1")
    assert vision_batch_enabled(root) is True
    monkeypatch.setenv("RPU_PI05_SIGLIP_W4A16", "1")
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


def _nvfp4_decoder():
    from rpu_backend.quant.nvfp4_pack import quantize_nvfp4

    generator = torch.Generator().manual_seed(14)
    mlp = torch.nn.Module()
    for name, shape in (("gate_proj", (4096, 16)), ("up_proj", (4096, 16)),
                        ("down_proj", (16, 4096))):
        leaf = torch.nn.Linear(shape[1], shape[0], bias=False, device="meta")
        codes, scale, tensor_scale = quantize_nvfp4(torch.randn(shape, generator=generator) * .02)
        leaf.weight = torch.nn.Parameter(codes, requires_grad=False)
        leaf.register_buffer("weight_scale", scale)
        leaf.register_buffer("tensor_scale", tensor_scale)
        leaf._pi05_nvfp4_abi = "striped_v2"
        setattr(mlp, name, leaf)
    return SimpleNamespace(config=SimpleNamespace(intermediate_size=4096),
                           layers=[SimpleNamespace(mlp=mlp)])


def _nvfp4_output(mlp, x):
    from rpu_backend.quant.nvfp4_pack import fp4_e2m1_decode

    def weight(leaf):
        block = leaf.weight_scale.view(torch.float8_e4m3fn).float().T.repeat_interleave(16, dim=1)
        return fp4_e2m1_decode(leaf.weight) * block * leaf.tensor_scale
    gate = torch.nn.functional.gelu(x @ weight(mlp.gate_proj).T, approximate="tanh")
    return (gate * (x @ weight(mlp.up_proj).T)) @ weight(mlp.down_proj).T


def test_pi05_action_nvfp4_mlp6_pads_codes_and_block_scales():
    model = _nvfp4_decoder()
    mlp = model.layers[0].mlp
    x = torch.randn(3, 16, generator=torch.Generator().manual_seed(15))
    before = _nvfp4_output(mlp, x)
    tensor_scales = [getattr(mlp, name).tensor_scale for name in ("gate_proj", "up_proj", "down_proj")]
    pad_decoder_mlp_inplace(model, core_topology({"model": {"num_cores": 4}}))
    assert mlp.gate_proj.weight.shape == (4096, 16)  # Four lanes need no pad.
    pad_decoder_mlp_inplace(model, core_topology({"model": {"num_cores": 6}}))
    # 704 E2M1 values per Down lane: eleven 64-wide K tiles.
    assert physical_intermediate_size(4096, 6, torch.uint8) == 4224
    assert mlp.gate_proj.weight.shape == mlp.up_proj.weight.shape == (4224, 16)
    assert mlp.down_proj.weight.shape == (16, 4224)
    assert mlp.gate_proj.weight_scale.shape == (1, 4224)
    assert mlp.down_proj.weight_scale.shape == (264, 16)
    assert not mlp.gate_proj.weight_scale[:, 4096:].any()
    assert not mlp.down_proj.weight_scale[256:].any()
    assert [getattr(mlp, name).tensor_scale for name in ("gate_proj", "up_proj", "down_proj")] == tensor_scales
    torch.testing.assert_close(_nvfp4_output(mlp, x), before, atol=0, rtol=0)
    with pytest.raises(ValueError, match="Action NVFP4"):
        physical_intermediate_size(16384, 6, torch.uint8)


def test_pi05_nvfp4_padding_rejects_packed_codes_before_mutation():
    model = _nvfp4_decoder()
    first = model.layers[0].mlp.gate_proj.weight
    model.layers[0].mlp.down_proj._pi05_nvfp4_packed = True
    with pytest.raises(ValueError, match="unpacked logical"):
        pad_decoder_mlp_inplace(model, core_topology({"model": {"num_cores": 6}}))
    assert model.layers[0].mlp.gate_proj.weight is first


def _full_policy(nvfp4):
    """Exact Pi0.5 geometry; one uninitialized module set is shared by every layer."""
    def leaf(rows, cols, fp4=False, dtype=torch.int8):
        result = torch.nn.Linear(cols, rows, bias=False, device="meta")
        result.weight = torch.nn.Parameter(
            torch.empty(rows, cols, dtype=torch.uint8 if fp4 else dtype), requires_grad=False)
        if fp4:
            result.register_buffer("weight_scale", torch.zeros(cols // 16, rows, dtype=torch.uint8))
            result.register_buffer("tensor_scale", torch.tensor(.01))
            result._pi05_nvfp4_abi = "striped_v2"
        else:
            result.register_buffer("weight_scale", torch.ones(rows, dtype=torch.float16))
        return result

    def decoder(hidden, intermediate, expert):
        fp4 = lambda name: nvfp4 and expert and name not in ("k_proj", "v_proj")
        layer = SimpleNamespace(self_attn=SimpleNamespace(), mlp=SimpleNamespace())
        for parent, name, shape in (
                ("self_attn", "q_proj", (2048, hidden)), ("self_attn", "k_proj", (256, hidden)),
                ("self_attn", "v_proj", (256, hidden)), ("self_attn", "o_proj", (hidden, 2048)),
                ("mlp", "gate_proj", (intermediate, hidden)), ("mlp", "up_proj", (intermediate, hidden)),
                ("mlp", "down_proj", (hidden, intermediate))):
            setattr(getattr(layer, parent), name, leaf(*shape, fp4=fp4(name)))
        config = SimpleNamespace(num_hidden_layers=18, hidden_size=hidden, intermediate_size=intermediate,
                                 num_attention_heads=8, num_key_value_heads=1, head_dim=256)
        return SimpleNamespace(config=config, layers=[layer] * 18)

    vision = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=27, hidden_size=1152,
                                                    intermediate_size=4304, num_attention_heads=16),
                             encoder=SimpleNamespace(layers=[None] * 27))
    pawe = SimpleNamespace(
        paligemma=SimpleNamespace(model=SimpleNamespace(
            language_model=decoder(2048, 16384, False), vision_tower=vision)),
        gemma_expert=SimpleNamespace(model=decoder(1024, 4096, True)))
    model = SimpleNamespace(paligemma_with_expert=pawe,
                            config=SimpleNamespace(max_action_dim=32, chunk_size=50),
                            action_in_proj=SimpleNamespace(out_features=1024))
    quant = {"method": "nvfp4a16", "nvfp4_abi": "striped_v2", "nvfp4_block_size": 16,
             "int4_components": ["expert"], "mixed_int8_suffixes": ["k_proj.weight", "v_proj.weight"]}
    return SimpleNamespace(model=model, _pi05_quant_config=quant if nvfp4 else {"method": "w8a16"})


@pytest.mark.parametrize("cores", (4, 6))
def test_pi05_reduced_policy_admits_only_the_action_nvfp4_scope(cores):
    from rpu_backend.adapters.pi05.cores import validate_reduced_policy

    topology = core_topology({"model": {"num_cores": cores}})
    for nvfp4 in (False, True):
        validate_reduced_policy(_full_policy(nvfp4), topology)
    for key, value in (("mixed_int8_suffixes", ["k_proj.weight"]),
                       ("int4_components", ["vlm", "expert"]),
                       ("nvfp4_abi", None), ("nvfp4_block_size", 32)):
        policy = _full_policy(True)
        policy._pi05_quant_config[key] = value
        with pytest.raises(ValueError, match="Action-only"):
            validate_reduced_policy(policy, topology)
    expert = lambda policy: policy.model.paligemma_with_expert.gemma_expert.model.layers[0]
    policy = _full_policy(True)
    expert(policy).self_attn.q_proj._pi05_nvfp4_packed = True
    with pytest.raises(ValueError, match="unpacked striped_v2"):
        validate_reduced_policy(policy, topology)
    policy = _full_policy(True)
    expert(policy).mlp.up_proj._pi05_nvfp4_abi = None
    with pytest.raises(ValueError, match="unpacked striped_v2"):
        validate_reduced_policy(policy, topology)
    # A W8 checkpoint cannot carry FP4 codes under the W8A16 method.
    policy = _full_policy(True)
    policy._pi05_quant_config = {"method": "w8a16"}
    with pytest.raises(ValueError, match="original cold logical"):
        validate_reduced_policy(policy, topology)


def test_pi05_reduced_example_derives_profile_from_core_geometry(tmp_path):
    from examples.common.config import ConfigError, example_metadata, load_config
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for cameras, scope in ((3, ""), (2, "-2cam")):
        # Two cameras: the same template with preset 2cam_t32 and a 2-camera batch.
        two = {} if cameras == 3 else {"input_preset": "2cam_t32",
                                       "overrides": {"input": {"batch": "/path/to/pi05_2cam_t32_batch.pt"}}}
        for precision in ("fp16", "w8a16"):
            template = root / f"examples/configs/pi05/h50/{precision}.toml"
            assert load_config(template, **two)["rpu_execution"]["model"]["num_cores"] == 8
            for cores in (4, 6):
                path = tmp_path / f"{cameras}-{precision}-selected-{cores}.toml"
                path.write_text(template.read_text().replace("num_cores = 8", f"num_cores = {cores}", 1))
                config = load_config(path, **two)
                assert config["rpu_execution"]["model"]["num_cores"] == cores
                assert example_metadata(config)["profile_id"].endswith(f".vla{scope}-tp{cores}")
                changed = tmp_path / f"{cameras}-{precision}-{cores}.toml"
                changed.write_text(path.read_text().replace(f"num_cores = {cores}", f"num_cores = {10-cores}"))
                assert example_metadata(load_config(changed, **two))["profile_id"].endswith(f".vla{scope}-tp{10-cores}")
                changed.write_text(path.read_text().replace(f"num_cores = {cores}", "num_cores = 5"))
                with pytest.raises(ConfigError, match="num_cores"):
                    load_config(changed, **two)


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


def _vision_w8a8_execution(selector, enabled):
    return {"components": {"vision_encoder": {"vision": {selector: enabled}}}}


@pytest.mark.parametrize("selector", ("qkv_w8a8", "mlp_w8a8"))
@pytest.mark.parametrize("enabled", (False, True))
def test_pi05_vision_w8a8_is_a_component_boolean(selector, enabled):
    _, vision, text, action = resolve_pi05_execution_components(
        _vision_w8a8_execution(selector, enabled), entry_point="test")
    assert vision["vision"][selector] is enabled
    assert selector not in text["prefill"]
    assert selector not in action["action"]
    default = resolve_pi05_execution_components({}, entry_point="test")
    assert not default[1]["vision"].get(selector, False)
    other = "mlp_w8a8" if selector == "qkv_w8a8" else "qkv_w8a8"
    assert not vision["vision"].get(other, False)


@pytest.mark.parametrize("selector", ("qkv_w8a8", "mlp_w8a8"))
@pytest.mark.parametrize("value", (0, 1, "true", None))
def test_pi05_vision_w8a8_rejects_non_boolean_before_loading(selector, value):
    with pytest.raises(TypeError, match=f"{selector}.*bool"):
        resolve_pi05_execution_components(_vision_w8a8_execution(selector, value), entry_point="test")


@pytest.mark.parametrize("selector", ("qkv_w8a8", "mlp_w8a8"))
def test_vision_w8a8_cannot_enable_other_stages_or_model_capabilities(selector):
    from rpu_backend.api._execution import normalize_rpu_execution

    for execution in (
        {"vision": {selector: True}},
        {"components": {"language_model": {"prefill": {selector: True}}}},
        {"components": {"action_expert": {"action": {selector: True}}}},
    ):
        with pytest.raises(ValueError, match=f"{selector}.*not supported"):
            resolve_pi05_execution_components(execution, entry_point="test")
    # The same-named component of another model does not inherit Pi's opt-in.
    with pytest.raises(ValueError, match=f"{selector}.*not supported"):
        normalize_rpu_execution(
            _vision_w8a8_execution(selector, True), entry_point="non-Pi model",
            supported={"vision": ("chunk_size", "linear_acc32")},
            supported_components={"vision_encoder": {"vision": ("chunk_size", "linear_acc32")}},
        )


@pytest.mark.parametrize("selector", ("qkv_w8a8", "mlp_w8a8"))
@pytest.mark.parametrize("enabled", (False, True))
def test_pi05_vision_w8a8_hot_change_fails_before_native_or_state_mutation(selector, enabled):
    from rpu_backend.adapters.pi05 import Pi05Adapter

    execution = _vision_w8a8_execution(selector, enabled)
    adapter = object.__new__(Pi05Adapter)
    adapter._rpu_is_ready = True
    adapter._rpu_execution = execution
    adapter._legacy_vlm_chunk_size = 0
    adapter._closed = True
    with pytest.raises(ValueError, match=f"{selector} is cold-only"):
        adapter.validate_execution_reconfigure(_vision_w8a8_execution(selector, not enabled))
    assert adapter._rpu_execution is execution


@pytest.mark.parametrize("selector", ("qkv_w8a8", "mlp_w8a8"))
@pytest.mark.parametrize("horizon", (10, 50))
@pytest.mark.parametrize("cores,cameras,acc32", (
    (8, 2, False), (8, 3, False), (4, 2, False), (6, 3, False),
    (8, 1, False), (8, 4, False), (8, 2, True),
))
def test_pi05_vision_w8a8_geometry_rejects_unsupported_before_loading(
    selector, horizon, cores, cameras, acc32, monkeypatch,
):
    from rpu_backend.adapters.pi05 import Pi05Adapter, _validate_execution_geometry

    monkeypatch.setenv("RPU_PI05_SIGLIP_BATCH", "1")
    execution = _vision_w8a8_execution(selector, True)
    execution["model"] = {"num_cores": cores}
    execution["components"]["vision_encoder"]["vision"]["linear_acc32"] = acc32
    adapter = object.__new__(Pi05Adapter)
    adapter._rpu_execution = execution
    adapter._legacy_vlm_chunk_size = 0
    adapter._closed = True
    adapter._lerobot_policy = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(
        chunk_size=horizon, image_features={f"camera{i}": None for i in range(cameras)},
    )))
    if cores == 8 and cameras in (2, 3) and not acc32:
        _validate_execution_geometry(adapter)
    else:
        with pytest.raises(ValueError, match=f"{selector} requires"):
            _validate_execution_geometry(adapter)
    assert adapter._rpu_execution is execution
