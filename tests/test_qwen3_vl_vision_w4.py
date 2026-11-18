"""Qwen3-VL packed W4 Vision: host contracts only.

Every size takes budgets 4/6/8. The 2B/4B Vision runs on 8 cores at budget 8;
the padded 8B/32B Vision runs on 4 cores at every budget. Recipe and profile
binding, the runner's cold-install selection, the adapter and installer
refusals before any weight changes, the G32 pgrp packing of both towers' six
block projections at the lanes they run on, and the native owner's declared op,
G32 check and row-bias order. None of this is a board run or numerical evidence
for the W4 Vision route.
"""
import copy
import re
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from examples.common import config as COMMON
from examples.runners.qwen3_vl import runner as qwen3_vl_runner
from rpu_backend.adapters.qwen3_vl import Qwen3VLAdapter, vision
from rpu_backend.api.errors import UnsupportedModelError
from rpu_backend.quant import precision_recipes
from rpu_backend.quant.int4_pgrp_pack import quantize_int4_group_wise
from test_qwen3_vl_awq_packing import exact_config, inverse_rpu_scale, inverse_rpu_weight
from test_qwen3_vl_large_quantized_vision import _block as _padded_block
from test_qwen3_vl_runtime_quant_loading import awq_runtime_config, config as runtime_config


ROOT = Path(__file__).resolve().parents[1]
VL = ROOT / "examples/configs/qwen3_vl/vl"
RECIPE = "w4a16_text_awq_head_vision_rtn_g32"
RUNTIME = "w4a16_text_awq_head_rtn_g32_vision_w8a16"
RECIPE_32B = "w4a16_text_head_vision_rtn_g32"
RUNTIME_32B = "w4a16_text_head_rtn_g32_vision_w8a16"
# (size, Vision W4 recipe, its runtime W4 checkpoint recipe, text method)
SIZES = (("2b", RECIPE, RUNTIME, "awq"), ("4b", RECIPE, RUNTIME, "awq"),
         ("8b", RECIPE, RUNTIME, "awq"), ("32b", RECIPE_32B, RUNTIME_32B, "rtn"))


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


# ------------------------------------------------------------- profiles and recipes


@pytest.mark.parametrize("size,recipe,runtime,method", SIZES, ids=[s[0] for s in SIZES])
def test_vision_w4_profiles_bind_recipe_and_cores(size, recipe, runtime, method):
    config = COMMON.load_config(VL / size / f"{recipe}.toml")
    profiles = COMMON.planner_profiles()
    profile = profiles[config["example"]["profile_id"]]
    base = profiles[f"qwen3-vl-{size}-runtime-w4a16.w4a16.multimodal"]
    assert profile["id"] == f"qwen3-vl-{size}-runtime-w4a16.w4a16-vision-g32.multimodal"
    assert (profile["precision"], profile["variant"], profile["family"], profile["owner"]) == (
        "w4a16", "runtime-align-v1-vision-w4", "qwen3-vl", base["owner"])
    assert config["example"]["registry_alias"] == profile["asset_alias"] == base["asset_alias"]
    assert profile["support"]["status"] == profile["hw_status"] == "IMPLEMENTED_UNVERIFIED_HW"
    # Same checkpoint, preflight and envelope as the W8 Vision profile; only Vision differs.
    assert profile["controlled_gates"] == base["controlled_gates"]
    limits = copy.deepcopy(profile["input_envelope"]["limits"])
    expected = copy.deepcopy(base["input_envelope"]["limits"])
    assert limits.pop("vision_on_install_quantization") == "w4a16"
    assert limits["quantization"].pop("vision").startswith(
        "blocks q/k/v/o/fc1/fc2 symmetric RTN W4/G32 on install; patch embedding and mergers W8")
    expected["quantization"].pop("vision")
    assert limits == expected
    assert profile["input_envelope"]["limits"]["allowed_num_cores"] == [4, 6, 8]
    assert COMMON.core_support(config) == dict.fromkeys((4, 6, 8))
    assert config["rpu_execution"]["model"]["num_cores"] == 8

    item = COMMON.precision_recipe("qwen3_vl", recipe, profile["id"])
    assert precision_recipes.checkpoint_recipe(item) == ("qwen3_vl", runtime)
    assert (item.spec("text").key(), item.spec("lm_head").key(), item.spec("vision").key()) == (
        f"w4a16-g32-{method}", "w4a16-g32-rtn", "w4a16-g32-rtn")
    assert item.spec("vision", "patch_embed").key() == item.spec("vision", "merger").key() == "w8a16"
    # Each recipe binds its own profiles: not the base profile, and not the other
    # size's Vision W4 recipe.
    other = RECIPE_32B if recipe == RECIPE else RECIPE
    for name, profile_id in ((runtime, profile["id"]), (recipe, base["id"]), (other, profile["id"])):
        with pytest.raises(COMMON.ConfigError, match="not registered for profile"):
            COMMON.precision_recipe("qwen3_vl", name, profile_id)


def test_runner_selects_the_cold_install_quantization_from_the_recipe():
    for size in ("2b", "4b", "8b", "32b"):
        assert qwen3_vl_runner.install_quantization(
            f"qwen3-vl-{size}-runtime-w4a16.w4a16-vision-g32.multimodal") == "vision_w4a16"
        assert qwen3_vl_runner.install_quantization(
            f"qwen3-vl-{size}-runtime-w4a16.w4a16.multimodal") is None
    for size in ("2b", "4b"):
        assert qwen3_vl_runner.install_quantization(f"qwen3-vl-{size}.nvfp4.text") == "nvfp4"
        assert qwen3_vl_runner.install_quantization(f"qwen3-vl-{size}.fp16.multimodal") is None


# ------------------------------------------------------------- admission before weights change


_RUNTIME_W4 = {"2b": lambda: awq_runtime_config(2), "4b": lambda: awq_runtime_config(4),
               "8b": lambda: awq_runtime_config(8), "32b": lambda: runtime_config(32, 4)}


@pytest.mark.parametrize("size", [s[0] for s in SIZES])
def test_preflight_admits_runtime_w4_checkpoints_at_every_budget(size):
    for cores in (4, 6, 8):
        Qwen3VLAdapter.preflight_quantization(
            _RUNTIME_W4[size](), {"model": {"num_cores": cores}}, "vision_w4a16")


@pytest.mark.parametrize("config", [
    pytest.param(lambda: runtime_config(2, 8), id="runtime-w8"),
    pytest.param(lambda: runtime_config(32, 8), id="runtime-w8-32b"),
    pytest.param(lambda: exact_config(4), id="awq-source"),
])
@pytest.mark.parametrize("cores", (4, 8))
def test_preflight_refuses_other_checkpoints(config, cores):
    with pytest.raises(UnsupportedModelError, match="exact Qwen3-VL runtime W4A16"):
        Qwen3VLAdapter.preflight_quantization(config(), {"model": {"num_cores": cores}}, "vision_w4a16")


def test_w4_tower_is_the_exact_2b_4b_vision():
    for size in (2, 4):
        assert vision._is_qwen3_vl_w4_vision_tower(exact_config(size).vision_config, 24)
    assert not vision._is_qwen3_vl_w4_vision_tower(exact_config(8).vision_config, 27)
    assert not vision._is_qwen3_vl_w4_vision_tower(exact_config(2).vision_config, 23)
    for field, value in (("deepstack_visual_indexes", [8, 16, 24]), ("hidden_act", "gelu"),
                         ("out_hidden_size", 4096), ("intermediate_size", 4096.0)):
        changed = copy.deepcopy(exact_config(2).vision_config)
        object.__setattr__(changed, field, value)
        assert not vision._is_qwen3_vl_w4_vision_tower(changed, 24), field


@pytest.mark.parametrize("flags", [
    {"w8a16": True}, {"_allow_graph_blocked_32b": True}, {"_allow_padded_8b": True},
    {"_allow_runtime_quantized_large": True}, {"_pack_fp16_block_weights": True},
], ids=lambda flags: next(iter(flags)))
def test_installer_refuses_w4_with_other_paths_before_any_block(flags):
    # Sentinel blocks: any conversion attempt would fail differently.
    tower = NS(config=exact_config(2).vision_config, blocks=[object()] * 24)
    with pytest.raises(ValueError, match="packed W4 Vision requires the exact"):
        vision.install_qwen3_vl_vision_for_rpu(tower, w4a16=True, **flags)


def test_installer_refuses_w4_on_other_towers_and_non_bool(monkeypatch):
    tower = NS(config=exact_config(8).vision_config, blocks=[object()] * 27)
    with pytest.raises(ValueError, match="packed W4 Vision requires the exact"):
        vision.install_qwen3_vl_vision_for_rpu(tower, w4a16=True)
    # The large tower takes W4 only through the runtime-quantized path, on 4 cores.
    with pytest.raises(ValueError, match="packed W4 Vision requires the exact"):
        vision.install_qwen3_vl_vision_for_rpu(tower, w4a16=True, _allow_runtime_quantized_large=True)
    # Sentinel blocks: skip the four-core module-tree check, keep the tower checks.
    monkeypatch.setattr(vision, "validate_qwen3_vl_vision_core_profile", lambda *args, **kwargs: None)
    small = NS(config=exact_config(2).vision_config, blocks=[object()] * 24)
    with pytest.raises(ValueError, match="requires the exact 8B/32B tower"):
        vision.install_qwen3_vl_vision_for_rpu(
            small, w4a16=True, _allow_runtime_quantized_large=True, num_cores=4)
    # The 8B tower at 4 cores passes every check and reaches the first block.
    with pytest.raises(AttributeError, match="'object' object has no attribute 'attn'"):
        vision.install_qwen3_vl_vision_for_rpu(
            tower, w4a16=True, _allow_runtime_quantized_large=True, num_cores=4)
    tower = NS(config=exact_config(2).vision_config, blocks=[object()] * 24)
    with pytest.raises(TypeError, match="w4a16 must be a bool"):
        vision.install_qwen3_vl_vision_for_rpu(tower, w4a16=1)


# ------------------------------------------------------------- block packing


def _linear(k, n, salt):
    layer = torch.nn.Linear(k, n, dtype=torch.float16)
    with torch.no_grad():
        rows, cols = torch.arange(n)[:, None], torch.arange(k)[None, :]
        weight = ((rows * 7 + cols * 3 + salt) % 61 - 30).float() / 256
        weight[:, :32] *= 40  # one wide group: its scale must not spread to the next group
        layer.weight.copy_(weight)
        layer.bias.copy_((torch.arange(n) % 13 - 6).float() / 128)
    return layer


def _tower_block():
    # One block of the exact 2B/4B tower: H1024, I4096, 16 heads x 64.
    return NS(attn=NS(qkv=_linear(1024, 3072, 1), proj=_linear(1024, 1024, 2)),
              mlp=NS(linear_fc1=_linear(1024, 4096, 3), linear_fc2=_linear(4096, 1024, 4)))


@pytest.mark.parametrize("cores", (4, 8))
def test_block_packing_decodes_to_the_rtn_codes_at_the_vision_lanes(cores):
    block = _tower_block()
    qkv = block.attn.qkv.weight.detach().clone()
    raw = {"q": qkv[:1024], "k": qkv[1024:2048], "v": qkv[2048:],
           "o": block.attn.proj.weight.detach().clone(),
           "fc1": block.mlp.linear_fc1.weight.detach().clone(),
           "fc2": block.mlp.linear_fc2.weight.detach().clone()}
    biases = {"q": block.attn.qkv.bias.detach()[:1024].clone(),
              "o": block.attn.proj.bias.detach().clone(),
              "fc2": block.mlp.linear_fc2.bias.detach().clone()}
    vision._convert_vision_block_weights_for_rpu(block, 16, 1024, w4a16=True, num_cores=cores)
    packed = {"q": block._rpu_q_w, "k": block._rpu_k_w, "v": block._rpu_v_w,
              "o": block.attn.proj.weight.detach(), "fc1": block.mlp.linear_fc1.weight.detach(),
              "fc2": block.mlp.linear_fc2.weight.detach()}
    for role, partition in (("q", 1), ("k", 1), ("v", 1), ("o", 0), ("fc1", 1), ("fc2", 0)):
        n, k = raw[role].shape
        weight, scale = packed[role], getattr(block, f"_rpu_{role}_ws")
        codes, scale_gn, _ = quantize_int4_group_wise(raw[role].half(), 32)
        assert weight.dtype == torch.uint8 and tuple(weight.shape) == (n, k // 2), role
        # The native set_weights check: aligned per-lane geometry and the exact G32 count.
        local_n, local_k = (n // cores, k) if partition == 1 else (n, k // cores)
        assert local_n % 16 == 0 and local_k % 64 == 0, role
        assert scale.dtype == torch.float16 and scale.dim() == 2 and scale.size(0) == 32, role
        assert scale.numel() == cores * ((local_k // 32 + 3) // 4) * ((local_n + 63) // 64) * 256
        assert torch.equal(inverse_rpu_weight(weight, n, k, partition, cores), codes), role
        assert torch.equal(inverse_rpu_scale(scale, n, k, partition, cores), scale_gn.t()), role
    # Biases stay FP16 logical vectors; the native owner places the row biases on core 0.
    for role, bias in biases.items():
        assert torch.equal(getattr(block, f"_rpu_{role}_b"), bias), role
    assert block._rpu_qwen3vl_vision_weights_converted == "w4a16"

    # Same mode: a no-op. Another mode or core count: refused, nothing repacked.
    vision._convert_vision_block_weights_for_rpu(block, 16, 1024, w4a16=True, num_cores=cores)
    assert block._rpu_q_w is packed["q"]
    with pytest.raises(RuntimeError, match="already converted with mode='w4a16'"):
        vision._convert_vision_block_weights_for_rpu(block, 16, 1024, w8a16=True, num_cores=cores)
    with pytest.raises(RuntimeError, match="core count cannot change"):
        vision._convert_vision_block_weights_for_rpu(
            block, 16, 1024, w4a16=True, num_cores=12 - cores)
    assert block._rpu_q_w is packed["q"]


def test_block_conversion_refuses_before_mutation():
    block = _tower_block()
    with pytest.raises(ValueError, match="one of w8a16, w4a16 and nvfp4"):
        vision._convert_vision_block_weights_for_rpu(block, 16, 1024, w8a16=True, w4a16=True)
    with torch.no_grad():
        block.mlp.linear_fc2.weight[3, 5] = float("inf")
    source = block.attn.qkv.weight
    with pytest.raises(ValueError, match="must be finite in FP16"):
        vision._convert_vision_block_weights_for_rpu(block, 16, 1024, w4a16=True)
    assert block.attn.qkv.weight is source and not hasattr(block, "_rpu_q_w")
    assert getattr(block, "_rpu_qwen3vl_vision_weights_converted", None) is None
    padded = _padded_block()
    with pytest.raises(ValueError, match="padded 8B/32B tower on 4 cores only"):
        vision._convert_vision_block_weights_for_rpu(
            padded, 16, 1152, w4a16=True, physical_head_dim=80, physical_intermediate_size=4352)
    assert padded.attn.qkv.weight.dtype == torch.float16 and not hasattr(padded, "_rpu_q_w")


def test_padded_block_packs_the_physical_projections_at_four_lanes():
    # 8B/32B: head 72->80 and MLP 4304->4352 are zero-padded first, then quantized,
    # so the G32 groups run over the physical K and the padding decodes to zero.
    block = _padded_block()
    qkv = block.attn.qkv.weight.detach().clone()
    qkv_b = block.attn.qkv.bias.detach().clone()

    def rows(weight):
        return torch.nn.functional.pad(weight.view(16, 72, -1), (0, 0, 0, 8)).reshape(1280, -1)

    raw = {"q": rows(qkv[:1152]), "k": rows(qkv[1152:2304]), "v": rows(qkv[2304:]),
           "o": torch.nn.functional.pad(block.attn.proj.weight.detach().view(1152, 16, 72),
                                        (0, 8)).reshape(1152, 1280),
           "fc1": torch.nn.functional.pad(block.mlp.linear_fc1.weight.detach(), (0, 0, 0, 48)),
           "fc2": torch.nn.functional.pad(block.mlp.linear_fc2.weight.detach(), (0, 48))}
    vision._convert_vision_block_weights_for_rpu(
        block, 16, 1152, w4a16=True, num_cores=4,
        physical_head_dim=80, physical_intermediate_size=4352)
    packed = {"q": block._rpu_q_w, "k": block._rpu_k_w, "v": block._rpu_v_w,
              "o": block.attn.proj.weight.detach(), "fc1": block.mlp.linear_fc1.weight.detach(),
              "fc2": block.mlp.linear_fc2.weight.detach()}
    for role, partition in (("q", 1), ("k", 1), ("v", 1), ("o", 0), ("fc1", 1), ("fc2", 0)):
        n, k = raw[role].shape
        weight, scale = packed[role], getattr(block, f"_rpu_{role}_ws")
        codes, scale_gn, _ = quantize_int4_group_wise(raw[role].half(), 32)
        assert weight.dtype == torch.uint8 and tuple(weight.shape) == (n, k // 2), role
        local_n, local_k = (n // 4, k) if partition == 1 else (n, k // 4)
        assert local_n % 16 == 0 and local_k % 64 == 0, role
        assert scale.numel() == 4 * ((local_k // 32 + 3) // 4) * ((local_n + 63) // 64) * 256
        decoded = inverse_rpu_weight(weight, n, k, partition, 4)
        assert torch.equal(decoded, codes), role
        assert torch.equal(inverse_rpu_scale(scale, n, k, partition, 4), scale_gn.t()), role
        # The padded rows / columns hold zero codes.
        if role in ("q", "k", "v"):
            assert torch.count_nonzero(decoded.view(16, 80, -1)[:, 72:]) == 0, role
        elif role == "o":
            assert torch.count_nonzero(decoded.view(1152, 16, 80)[..., 72:]) == 0
        else:
            assert torch.count_nonzero(decoded[4304:] if role == "fc1" else decoded[:, 4304:]) == 0
    padded_q_b = getattr(block, "_rpu_q_b").view(16, 80)
    assert torch.equal(padded_q_b[:, :72].reshape(1152), qkv_b[:1152])
    assert torch.count_nonzero(padded_q_b[:, 72:]) == 0
    assert block._rpu_qwen3vl_vision_weights_converted == "w4a16"
    assert block._rpu_qwen3vl_vision_physical_geometry == (80, 4352)


# ------------------------------------------------------------- native owner


def test_native_owner_declares_w4_and_adds_row_bias_before_the_reduction():
    source = (ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp").read_text(encoding="utf-8")
    assert "static void check_qwen3vl_vision_w4a16_projection(" in source
    # Column projections are checked at partition 1, row projections at 0.
    for role, partition in (("q_w", 1), ("k_w", 1), ("v_w", 1), ("o_w", 0),
                            ("fc1_w", 1), ("fc2_w", 0)):
        assert re.search(rf'"{role}",\s+packed_w4 \? {partition} : -1, cores, nvfp4\);', source), role
    op = source[source.index("void rpu_qwen3vl_vision_set_weights_w4a16("):]
    assert "/*packed_w4=*/true);" in op[:op.index("\n}\n")]
    # The packed W4 row Linear refuses in-kernel bias: launch without it, add the
    # core-0-only row bias to every partial, then reduce.
    for flag, bias in (("w4_o", "o_bias"), ("w4_fc2", "fc2_bias")):
        launch = source.index(f'{flag} ? 0 : layer_addr(layer_idx, 0, "{bias}")')
        add = source.index(f'if ({flag}) add_row_bias_partial(layer_addr(layer_idx, 0, "{bias}")')
        reduce = source.index("rpu_launch_all_reduce_sum_residual_kernel(", add)
        assert launch < add < reduce, flag
    for path in ("src/core/rpu_kernel_decls.h", "src/core/rpu_dispatch_registrations.inc"):
        assert "qwen3vl_vision_set_weights_w4a16" in (ROOT / path).read_text(encoding="utf-8"), path
    installer = (ROOT / "python/rpu_backend/adapters/qwen3_vl/vision.py").read_text(encoding="utf-8")
    assert re.search(r"if w4a16:\n\s+torch\.ops\.rpu\.qwen3vl_vision_set_weights_w4a16\(", installer)
