"""Qwen3-VL 2B/4B NVFP4 Vision: host contracts only.

Recipe and profile binding, the runner's cold-install selection, preflight and
installer refusals before any weight changes, the NVFP4 packing and tensor
scales of the real tower's six block projections at 4 and 8 lanes, the
adapter's Vision install arguments, and the native owner's declared op, table
buffers and launches. None of this is a board run or numerical evidence for
the NVFP4 Vision route.
"""
import copy
import re
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from examples.common import config as COMMON
from examples.runners.qwen3_vl import runner as qwen3_vl_runner
from rpu_backend.adapters import qwen3_vl
from rpu_backend.adapters.qwen3_vl import Qwen3VLAdapter, vision
from rpu_backend.api import conditional_generation as public
from rpu_backend.api.errors import UnsupportedModelError
from rpu_backend.quant.nvfp4_pack import quantize_nvfp4, swizzle_nvfp4_scales, swizzle_pack_nvfp4
from test_qwen3_vl_awq_packing import exact_config
from test_qwen3_vl_large_quantized_vision import _block as _padded_block
from test_qwen3_vl_runtime_quant_loading import awq_runtime_config, config as vl_config
from test_qwen3_vl_text_nvfp4 import _tiny_vl_model, install_runtime, vl_install  # noqa: F401
from test_qwen3_vl_vision_w4 import _tower_block


ROOT = Path(__file__).resolve().parents[1]
VL = ROOT / "examples/configs/qwen3_vl/vl"
RECIPE = "nvfp4_text_vision"
ROLES = (("q", 1), ("k", 1), ("v", 1), ("o", 0), ("fc1", 1), ("fc2", 0))


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


# ------------------------------------------------------------- profiles and recipes


@pytest.mark.parametrize("size", ("2b", "4b"))
def test_vision_nvfp4_profiles_bind_recipe_and_cores(size):
    config = COMMON.load_config(VL / size / f"{RECIPE}.toml")
    profiles = COMMON.planner_profiles()
    profile = profiles[config["example"]["profile_id"]]
    base = profiles[f"qwen3-vl-{size}.nvfp4.text"]
    assert profile["id"] == f"qwen3-vl-{size}.nvfp4-vision.multimodal"
    assert (profile["precision"], profile["variant"], profile["family"], profile["component"]) == (
        "nvfp4", "nvfp4-vision", "qwen3-vl", config["input"]["component"])
    assert config["example"]["registry_alias"] == profile["asset_alias"] == base["asset_alias"]
    assert profile["support"]["status"] == profile["hw_status"] == "IMPLEMENTED_UNVERIFIED_HW"
    # Same dense source, preflight gate and text geometry as the text NVFP4 profile.
    assert profile["controlled_gates"] == base["controlled_gates"]
    limits = copy.deepcopy(profile["input_envelope"]["limits"])
    expected = copy.deepcopy(base["input_envelope"]["limits"])
    assert limits.pop("vision_on_install_quantization") == "nvfp4"
    assert limits["quantization"].pop("vision").startswith(
        "blocks q/k/v/o/fc1/fc2 NVFP4 on install") and expected["quantization"].pop("vision") == "FP16"
    # One 640x480 image instead of the text-only prompt.
    assert (limits.pop("image_size_wh"), limits.pop("image_grid_thw")) == ([640, 480], [1, 30, 40])
    assert limits.pop("image_content").startswith("one image")
    assert limits.pop("example_geometry") == {
        **expected.pop("example_geometry"), "images": 1,
        "logical_prefill_rows": config["input"]["prefill_tokens"]}
    assert limits == expected and limits["allowed_num_cores"] == [4, 6, 8]
    assert COMMON.core_support(config) == {4: None, 6: None, 8: None}

    item = COMMON.precision_recipe("qwen3_vl", RECIPE, profile["id"])
    assert (item.spec("text").key(), item.spec("vision").key(), item.spec("lm_head").key()) == (
        "nvfp4", "nvfp4", "fp16")
    assert item.spec("vision", "patch_embed").key() == item.spec("vision", "merger").key() == "fp16"
    # Each recipe binds its own profiles; 8B keeps FP16 Vision.
    for recipe, profile_id in (("nvfp4_text", profile["id"]), (RECIPE, base["id"]),
                               (RECIPE, "qwen3-vl-8b.nvfp4.text")):
        with pytest.raises(COMMON.ConfigError, match="not registered for profile"):
            COMMON.precision_recipe("qwen3_vl", recipe, profile_id)


def test_runner_selects_vision_nvfp4_from_the_recipe():
    for size in ("2b", "4b"):
        assert qwen3_vl_runner.install_quantization(
            f"qwen3-vl-{size}.nvfp4-vision.multimodal") == "vision_nvfp4"
        assert qwen3_vl_runner.install_quantization(f"qwen3-vl-{size}.nvfp4.text") == "nvfp4"
        assert qwen3_vl_runner.install_quantization(f"qwen3-vl-{size}.fp16.multimodal") is None


# ------------------------------------------------------------- admission before weights change


@pytest.mark.parametrize("size", (2, 4))
@pytest.mark.parametrize("cores", (4, 6, 8))
def test_preflight_admits_the_dense_2b_4b_source(size, cores):
    Qwen3VLAdapter.preflight_quantization(
        vl_config(size, source=True), {"model": {"num_cores": cores}}, "vision_nvfp4")


def _changed_tower(field, value):
    config = vl_config(2, source=True)
    setattr(config.vision_config, field, value)
    return config


@pytest.mark.parametrize("config", [
    pytest.param(lambda: vl_config(8, source=True), id="8b-source"),
    pytest.param(lambda: _changed_tower("depth", 23), id="depth"),
    pytest.param(lambda: _changed_tower("deepstack_visual_indexes", [8, 16, 24]), id="deepstack"),
    pytest.param(lambda: exact_config(2), id="awq-source"),
    pytest.param(lambda: awq_runtime_config(4), id="runtime-w4"),
])
def test_preflight_refuses_other_sources_and_towers(config):
    with pytest.raises(UnsupportedModelError, match="2B/4B for NVFP4 Vision"):
        Qwen3VLAdapter.preflight_quantization(config(), {"model": {"num_cores": 8}}, "vision_nvfp4")


def test_public_vision_nvfp4_rejects_8b_before_weight_load(monkeypatch):
    monkeypatch.setattr(public.AutoConfig, "from_pretrained", lambda *a, **k: vl_config(8, source=True))
    monkeypatch.setattr(public.AutoModelForImageTextToText, "from_pretrained",
                        lambda *a, **k: pytest.fail("invalid NVFP4 Vision source loaded weights"))
    with pytest.raises(UnsupportedModelError, match="exact 2B/4B tower"):
        public.RPUModelForConditionalGeneration.from_pretrained(
            "source", quantization="vision_nvfp4", rpu_execution={"model": {"num_cores": 8}})


@pytest.mark.parametrize("flags", [
    {"w8a16": True}, {"w4a16": True}, {"_allow_graph_blocked_32b": True}, {"_allow_padded_8b": True},
    {"_allow_runtime_quantized_large": True}, {"_pack_fp16_block_weights": True},
], ids=lambda flags: next(iter(flags)))
def test_installer_refuses_nvfp4_with_other_paths_before_any_block(flags):
    # Sentinel blocks: any conversion attempt would fail differently.
    tower = NS(config=exact_config(2).vision_config, blocks=[object()] * 24)
    with pytest.raises(ValueError, match="NVFP4 Vision requires the exact"):
        vision.install_qwen3_vl_vision_for_rpu(tower, nvfp4=True, **flags)


def test_installer_refuses_nvfp4_on_other_towers_and_non_bool():
    tower = NS(config=exact_config(8).vision_config, blocks=[object()] * 27)
    with pytest.raises(ValueError, match="NVFP4 Vision requires the exact"):
        vision.install_qwen3_vl_vision_for_rpu(tower, nvfp4=True)
    tower = NS(config=exact_config(2).vision_config, blocks=[object()] * 24)
    with pytest.raises(TypeError, match="nvfp4 must be a bool"):
        vision.install_qwen3_vl_vision_for_rpu(tower, nvfp4=1)


# ------------------------------------------------------------- block packing


@pytest.mark.parametrize("cores", (4, 8))
def test_block_packing_matches_the_nvfp4_reference_at_the_vision_lanes(cores):
    block = _tower_block()
    qkv = block.attn.qkv.weight.detach().clone()
    raw = {"q": qkv[:1024], "k": qkv[1024:2048], "v": qkv[2048:],
           "o": block.attn.proj.weight.detach().clone(),
           "fc1": block.mlp.linear_fc1.weight.detach().clone(),
           "fc2": block.mlp.linear_fc2.weight.detach().clone()}
    biases = {"q": block.attn.qkv.bias.detach()[:1024].clone(),
              "o": block.attn.proj.bias.detach().clone(),
              "fc2": block.mlp.linear_fc2.bias.detach().clone()}
    vision._convert_vision_block_weights_for_rpu(block, 16, 1024, nvfp4=True, num_cores=cores)
    packed = {"q": block._rpu_q_w, "k": block._rpu_k_w, "v": block._rpu_v_w,
              "o": block.attn.proj.weight.detach(), "fc1": block.mlp.linear_fc1.weight.detach(),
              "fc2": block.mlp.linear_fc2.weight.detach()}
    for role, partition in ROLES:
        n, k = raw[role].shape
        codes, block_scale, tensor_scale = quantize_nvfp4(raw[role].half())
        weight, scale = packed[role], getattr(block, f"_rpu_{role}_ws")
        # The native set_weights check: aligned per-lane geometry, [N,K/2] and [K/16,N] bytes.
        local_n, local_k = (n // cores, k) if partition == 1 else (n, k // cores)
        assert local_n % 16 == 0 and local_k % 64 == 0, role
        assert weight.dtype == scale.dtype == torch.uint8, role
        assert tuple(weight.shape) == (n, k // 2) and tuple(scale.shape) == (k // 16, n), role
        reference = swizzle_pack_nvfp4(codes, partition, cores).reshape(n, k // 2)
        assert torch.equal(weight, reference), role
        # The partition matters: the other lane split is a different payload.
        assert not torch.equal(weight, swizzle_pack_nvfp4(codes, 1 - partition, cores).reshape(n, k // 2))
        assert torch.equal(scale, swizzle_nvfp4_scales(block_scale, partition, cores)), role
        ts = getattr(block, f"_rpu_{role}_ts")
        assert ts.dtype == torch.float32 and ts.dim() == 0 and torch.equal(ts, tensor_scale), role
    # Biases stay FP16 logical vectors; the native owner places the row biases on core 0.
    for role, bias in biases.items():
        assert torch.equal(getattr(block, f"_rpu_{role}_b"), bias), role
    assert block._rpu_qwen3vl_vision_weights_converted == "nvfp4"

    # Same mode: a no-op. Another mode or core count: refused, nothing repacked.
    vision._convert_vision_block_weights_for_rpu(block, 16, 1024, nvfp4=True, num_cores=cores)
    assert block._rpu_q_w is packed["q"]
    with pytest.raises(RuntimeError, match="already converted with mode='nvfp4'"):
        vision._convert_vision_block_weights_for_rpu(block, 16, 1024, w4a16=True, num_cores=cores)
    with pytest.raises(RuntimeError, match="core count cannot change"):
        vision._convert_vision_block_weights_for_rpu(
            block, 16, 1024, nvfp4=True, num_cores=12 - cores)
    assert block._rpu_q_w is packed["q"]


def test_block_conversion_refuses_nvfp4_before_mutation():
    block = _tower_block()
    for flags in ({"w8a16": True}, {"w4a16": True}):
        with pytest.raises(ValueError, match="one of w8a16, w4a16 and nvfp4"):
            vision._convert_vision_block_weights_for_rpu(block, 16, 1024, nvfp4=True, **flags)
    with pytest.raises(TypeError, match="nvfp4 must be a bool"):
        vision._convert_vision_block_weights_for_rpu(block, 16, 1024, nvfp4=1)
    with torch.no_grad():
        block.mlp.linear_fc2.weight[3, 5] = float("inf")
    source = block.attn.qkv.weight
    with pytest.raises(ValueError, match="must be finite in FP16"):
        vision._convert_vision_block_weights_for_rpu(block, 16, 1024, nvfp4=True)
    assert block.attn.qkv.weight is source and not hasattr(block, "_rpu_q_w")
    assert getattr(block, "_rpu_qwen3vl_vision_weights_converted", None) is None
    padded = _padded_block()
    with pytest.raises(ValueError, match="NVFP4 Vision does not admit the padded 8B/32B tower"):
        vision._convert_vision_block_weights_for_rpu(
            padded, 16, 1152, nvfp4=True, physical_head_dim=80, physical_intermediate_size=4352)
    assert padded.attn.qkv.weight.dtype == torch.float16 and not hasattr(padded, "_rpu_q_w")


# ------------------------------------------------------------- adapter install


@pytest.fixture
def vision_install(monkeypatch, vl_install):  # noqa: F811
    for name in ("qwen3vl_vision_set_fast_replay", "qwen3vl_vision_set_preload_replay_skip"):
        monkeypatch.setattr(torch.ops.rpu, name, lambda handle, enabled, name=name: (
            vl_install["events"].append((name, handle, enabled))), raising=False)
    return vl_install


@pytest.mark.parametrize("size,cores", [("2b", 8), ("4b", 8), ("2b", 4), ("4b", 6)])
def test_adapter_installs_nvfp4_vision_with_fp16_mergers(monkeypatch, vision_install, size, cores):
    state = vision_install
    from rpu_backend.quant import load as quant_load
    monkeypatch.setattr(quant_load, "_move_materialized_decoder_for_rpu",
                        lambda *a, **k: pytest.fail("NVFP4 took the FP16 storage-bank decoder move"))
    monkeypatch.setattr(qwen3_vl, "_validate_qwen3_vl_reduced_weights", lambda *a: None)
    model, _moves = _tiny_vl_model(size, cores)
    owner = qwen3_vl.Qwen3VLAdapter(model, quantization="vision_nvfp4")
    assert owner.to_rpu() is model
    # Text takes the same NVFP4 payload as the text-only recipe.
    assert len(state["text_kwargs"]["nvfp4_tensor_scales"]) == 7
    kwargs = state["vision_kwargs"]
    assert kwargs["nvfp4"] is True and kwargs.get("num_cores", 8) == (8 if cores == 8 else 4)
    assert not any(kwargs.get(flag, False) for flag in (
        "w8a16", "w4a16", "_pack_fp16_block_weights", "_allow_graph_blocked_32b",
        "_allow_padded_8b", "_allow_runtime_quantized_large"))
    events = state["events"]
    if cores == 8:
        # FP16 device mergers; the encoder replays like the FP16 banks.
        assert ("enable_vision_merger", {}) in events
        assert ("qwen3vl_vision_set_fast_replay", 11, True) in events
        assert ("qwen3vl_vision_set_preload_replay_skip", 11, True) in events
        assert not any(event[0] == "fused_vision_merger" for event in events)
    owner.close()


# ------------------------------------------------------------- native owner


def test_native_owner_declares_nvfp4_and_launches_with_the_tables():
    source = (ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp").read_text(encoding="utf-8")
    assert "static void check_qwen3vl_vision_nvfp4_projection(" in source
    # Column projections are checked at partition 1, row projections at 0.
    for role, partition in ROLES:
        assert re.search(rf'"{role}_w",\s+packed_w4 \? {partition} : -1, cores, nvfp4\);', source), role
    op = source[source.index("void rpu_qwen3vl_vision_set_weights_nvfp4("):]
    assert "/*packed_w4=*/true, nvfp4_tensor_scales);" in op[:op.index("\n}\n")]
    # Each block projection reads its own table at the layer index; mergers stay FP16.
    for index in range(6):
        assert source.count(f"nvfp4_table_addr({index}), nvfp4_layer_id(layer_idx),") == 1, index
    assert source.count("/*nvfp4_tensor_scale_spm_addr=*/0, /*nvfp4_layer_id=*/0,") >= 2
    for path in ("src/core/rpu_kernel_decls.h", "src/core/rpu_dispatch_registrations.inc"):
        assert "qwen3vl_vision_set_weights_nvfp4" in (ROOT / path).read_text(encoding="utf-8"), path
    installer = (ROOT / "python/rpu_backend/adapters/qwen3_vl/vision.py").read_text(encoding="utf-8")
    assert re.search(r"if nvfp4:\n\s+torch\.ops\.rpu\.qwen3vl_vision_set_weights_nvfp4\(", installer)
