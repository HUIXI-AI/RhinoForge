"""HY-Embodied UMI / HyVLA W8A16 and G32 W4A16: host contracts only.

Recipe and profile binding, the cold environment each entry selects, the G32
pgrp packing at four, six and eight lanes, and the native admission. None of
this is a board run or numerical evidence for the quantized routes.
"""
import copy
import os
from pathlib import Path

import pytest
import torch

from examples.common import config as COMMON
from examples.runners.hy_vla import optimized as hy_vla_runner
from examples.runners.hy_vla import runner as umi_runner
from rpu_backend.adapters.hy_vla import weights
from rpu_backend.adapters.hy_vla import runtime
from rpu_backend.adapters.hy_vla.runtime import prepare_hy_vla_cold_plan
from rpu_backend.api import hy_embodied as api
from rpu_backend.quant.int4_pgrp_pack import quantize_int4_group_wise
from test_qwen3_vl_awq_packing import inverse_rpu_scale, inverse_rpu_weight


ROOT = Path(__file__).resolve().parents[1]
UMI = ROOT / "examples/configs/hy_embodied/umi"
# TOML recipe -> (profile variant, profile precision, runner selector).
RECIPES = {
    "w8a16": ("w8a16", "w8a16", "w8a16"),
    "w4a16_rtn_g32_vision_w8a16": ("w4a16-g32", "w4a16", "w4a16"),
}
# Selector -> (vit, vlm text, vlm vision, expert) bits.
BITS = {"fp16": (16, 16, 16, 16), "w8a16": (8, 8, 8, 8), "w4a16": (8, 4, 4, 4),
        "w4a16-vision": (4, 4, 4, 4)}


@pytest.fixture(autouse=True)
def _no_hyvla_selectors(monkeypatch):
    for name in [name for name in os.environ
                 if name.startswith("RPU_HY_VLA_") or name == "RPU_RMSNORM_VWARP"]:
        monkeypatch.delenv(name)


def _bits(plan):
    w = plan.weights
    return (w.vit_bits, w.vlm_text_bits, w.vlm_vision_bits, w.expert_bits)


# ------------------------------------------------------------- profiles and recipes


@pytest.mark.parametrize("recipe", RECIPES)
def test_quant_profiles_bind_exact_recipe_and_cores(recipe):
    variant, precision, selector = RECIPES[recipe]
    config = COMMON.load_config(UMI / f"{recipe}.toml")
    profiles = COMMON.planner_profiles()
    profile = profiles[config["example"]["profile_id"]]
    fp16 = profiles["hy-embodied-0.5-vla-umi.fp16.vla"]
    assert profile["id"] == f"hy-embodied-0.5-vla-umi.{variant}.vla"
    assert (profile["precision"], profile["variant"], profile["family"]) == (precision, variant, "hy-vla")
    assert profile["support"]["status"] == "IMPLEMENTED_UNVERIFIED_HW"
    limits = profile["input_envelope"]["limits"]
    expected = dict(fp16["input_envelope"]["limits"])
    if precision == "w4a16":
        expected["group_size"] = 32
    assert limits == expected and limits["allowed_num_cores"] == [4, 6, 8]
    assert config["example"]["dtype"] == selector
    assert umi_runner.policy_dtype(config["example"]) == selector
    assert COMMON.core_support(config) == {4: None, 6: None, 8: None}
    for cores in (4, 6, 8):
        COMMON._validate_execution({"model": {"num_cores": cores}}, profile=profile)
    for cores in (2, 5, 7):
        with pytest.raises((ValueError, COMMON.ConfigError)):
            COMMON._validate_execution({"model": {"num_cores": cores}}, profile=profile)
    unknown = copy.deepcopy(profile)
    unknown["id"] = "hy-embodied-0.5-vla-umi.other.vla"
    with pytest.raises(COMMON.ConfigError, match="registered UMI FP16, W8A16 or W4A16"):
        COMMON._validate_execution({"model": {"num_cores": 6}}, profile=unknown)
    # Each profile realizes only its own recipe.
    other = next(name for name in RECIPES if name != recipe)
    with pytest.raises(COMMON.ConfigError, match="not registered for profile"):
        COMMON.precision_recipe("hy_vla", other, profile["id"])
    with pytest.raises(COMMON.ConfigError, match="not registered for profile"):
        COMMON.precision_recipe("hy_vla", "fp16", profile["id"])


def test_template_w4_recipe_routes_and_wrong_specs_are_refused():
    assert hy_vla_runner.precision_route("w4a16") == "w4a16"
    assert umi_runner.policy_dtype({"dtype": "w4a16"}) == "w4a16"
    config = COMMON.load_config(ROOT / "examples/configs/hy_vla/w4a16_rtn_g32_vision_w8a16.toml")
    args = hy_vla_runner.example_arguments(config)
    assert args.precision == "w4a16" and args.resident_fp8_scale_layers == 0
    config["hy_vla"]["resident_fp8_scale_layers"] = 4
    with pytest.raises(ValueError, match="requires Action NVFP4"):
        hy_vla_runner.validate_example_config(config)


# ------------------------------------------------------------- cold environment


@pytest.mark.parametrize("cores", (4, 6, 8))
@pytest.mark.parametrize("dtype", ("fp16", "w8a16", "w4a16", "w4a16-vision"))
def test_policy_dtype_selects_the_cold_plan(dtype, cores):
    env = {"RPU_HY_VLA_W8A16": api._PROFILE_ENV[dtype],
           "RPU_HY_VLA_W4A16": api._PROFILE_W4_ENV.get(dtype, "0")}
    plan = prepare_hy_vla_cold_plan(env, num_cores=cores)
    assert _bits(plan) == BITS[dtype]
    assert not plan.weights.gateup_resident and not plan.weights.expert_nvfp4
    assert plan.weights.vision_lanes == (4 if dtype == "w4a16-vision" or cores != 8 else 8)
    # The legacy W8 names keep W4 off.
    assert all(api._PROFILE_W4_ENV.get(name, "0") == "0"
               for name in api._PROFILE_ENV if not name.startswith("w4a16"))


@pytest.mark.parametrize("cores", (4, 6, 8))
def test_template_environment_is_w4_without_residency(cores):
    config = COMMON.load_config(ROOT / "examples/configs/hy_vla/w4a16_rtn_g32_vision_w8a16.toml")
    if cores != 8:
        config.setdefault("rpu_execution", {})["model"] = {"num_cores": cores}
    env = hy_vla_runner.profile_environment(hy_vla_runner.example_arguments(config))
    assert (env["RPU_HY_VLA_W8A16"], env["RPU_HY_VLA_W4A16"]) == ("vit", "vlm,expert")
    assert env["RPU_HY_VLA_EXPERT_GATEUP_RESIDENT"] == "0"
    plan = prepare_hy_vla_cold_plan(env, num_cores=cores)
    assert _bits(plan) == BITS["w4a16"] and not plan.weights.gateup_resident
    assert plan.weights.attn_tp8 is (cores == 8)


def test_to_rpu_passes_the_w4_selection_before_ownership(tmp_path, monkeypatch):
    for name in ("model.safetensors", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json"):
        (tmp_path / name).write_bytes(b"")
    seen = {}

    class Stop(Exception):
        pass

    def plan(overrides, *, num_cores):
        seen.update(overrides, cores=num_cores)
        raise Stop

    monkeypatch.setattr(runtime, "prepare_hy_vla_cold_plan", plan)
    policy = api.HyEmbodiedPolicy.from_checkpoint(
        tmp_path, dtype="w4a16", rpu_execution={"model": {"num_cores": 6}})
    with pytest.raises(Stop):
        policy.to("rpu")
    assert seen == {"RPU_HY_VLA_W8A16": "vit", "RPU_HY_VLA_W4A16": "vlm,expert", "cores": 6}
    assert "RPU_HY_VLA_W4A16" not in os.environ


def test_w4_refuses_resident_gateup():
    with pytest.raises(ValueError, match="requires FP16/W8A16 expert"):
        prepare_hy_vla_cold_plan({"RPU_HY_VLA_W8A16": "vit", "RPU_HY_VLA_W4A16": "vlm,expert",
                                  "RPU_HY_VLA_ATTN_TP8": "1",
                                  "RPU_HY_VLA_EXPERT_GATEUP_RESIDENT": "1"})


# ------------------------------------------------------------- G32 packing


class _Source:
    def __getitem__(self, key):
        return torch.zeros(2, 2)


@pytest.mark.parametrize("cores", (4, 6, 8))
def test_w4_reaches_every_vlm_and_expert_projection(monkeypatch, cores):
    cold = weights._HyVlaWeightColdPlan.from_environment(
        {"RPU_HY_VLA_W8A16": "vit", "RPU_HY_VLA_W4A16": "vlm,expert",
         **({"RPU_HY_VLA_ATTN_TP8": "1"} if cores == 8 else {})}, num_cores=cores)
    calls = []
    monkeypatch.setattr(weights, "_quant", lambda raw, partition, lanes, bits: (
        calls.append((partition, lanes, bits)), (raw, raw))[1])
    monkeypatch.setattr(weights, "_rpu16", lambda value: value)
    monkeypatch.setattr(weights, "_dup_kv", lambda raw, head_dim, *, attn_tp8: raw)
    cfg = weights.HyVlaConfig(layers=1, vit_layers=1)
    attn, mlp = (8, 8) if cores == 8 else (4, cores)
    for suffix in ("", "_v"):
        calls.clear()
        weights._build_mot_branch(_Source(), cfg, suffix, cold)
        assert calls == [(1, attn, 4)] * 3 + [(0, attn, 4)] + [(1, mlp, 4)] * 2 + [(0, mlp, 4)]
    calls.clear()
    weights._build_expert(_Source(), _Source(), cfg, cold)
    lanes = 8 if cores == 8 else 4
    assert calls == [(1, attn, 4)] * 3 + [(0, attn, 4)] + [(1, lanes, 4)] * 2 + [(0, lanes, 4)]


# (name, N, K, partition, lanes): the real VLM (H2048/I6144, KV4 or TP8-duplicated
# KV8) and expert (H1024/I2048) projections at each lane count they run on.
_GEOMETRIES = [
    ("vlm k/v kv4", 512, 2048, 1, 4), ("vlm o", 2048, 2048, 0, 4),
    ("vlm gate six", 6144, 2048, 1, 6), ("vlm down six", 2048, 6144, 0, 6),
    ("vlm gate four", 6144, 2048, 1, 4), ("vlm down four", 2048, 6144, 0, 4),
    ("vlm k/v tp8", 1024, 2048, 1, 8), ("vlm down tp8", 2048, 6144, 0, 8),
    ("expert q", 2048, 1024, 1, 4), ("expert o", 1024, 2048, 0, 4),
    ("expert down", 1024, 2048, 0, 4), ("expert gate tp8", 2048, 1024, 1, 8),
    ("expert o tp8", 1024, 2048, 0, 8),
]


@pytest.mark.parametrize("name,n,k,partition,lanes", _GEOMETRIES, ids=[g[0] for g in _GEOMETRIES])
def test_g32_packing_decodes_to_the_rtn_weights(name, n, k, partition, lanes):
    w = (((torch.arange(n * k) * 7919) % 1021 - 510).float() / 4096).reshape(n, k)
    w[:, :32] *= 40  # one wide group: its scale must not spread to the next group
    packed, scale = weights._pack_w4_g32(w, partition, lanes)
    codes, scale_gn, _ = quantize_int4_group_wise(w.half(), 32)
    assert packed.dtype == torch.uint8 and tuple(packed.shape) == (n, k // 2)
    assert scale.dtype == torch.float16 and scale.dim() == 2 and scale.size(0) == 32
    # The reduced native check: aligned per-core geometry and the exact G32 count.
    local_n, local_k = (n // lanes, k) if partition == 1 else (n, k // lanes)
    assert local_n % 16 == 0 and local_k % 64 == 0
    assert scale.numel() == lanes * ((local_k // 32 + 3) // 4) * ((local_n + 63) // 64) * 256
    assert torch.equal(inverse_rpu_weight(packed, n, k, partition, lanes), codes)
    assert torch.equal(inverse_rpu_scale(scale, n, k, partition, lanes), scale_gn.t())
    assert not torch.equal(scale_gn[0], scale_gn[1])


def test_vision_row_projections_have_no_eight_lane_w4_layout():
    # ViT o_proj K1280 and fc2 K4352: K % (64 x 8) != 0 while four lanes fit.
    for k in (1280, 4352):
        assert k % (64 * 8) and not k % (64 * 4)


# ------------------------------------------------------------- native admission


def test_native_reduced_decoder_admits_hyvla_g32_and_fused_routes_refuse_it():
    model = (ROOT / "src/fused/rpu_qwen3_model.h").read_text(encoding="utf-8")
    vlm = model[model.index("const bool hyvla_vlm_profile"):model.index("const bool hyvla_expert_profile")]
    expert = model[model.index("const bool hyvla_expert_profile"):model.index("const bool hyvla_profile")]
    assert "reduced_packed_w4" in vlm and "reduced_packed_w4" in expert
    bounded = model[model.index("const bool bounded_text_w4"):]
    assert "hyvla_profile" in bounded[:bounded.index(";")]
    # Fused HyVLA routes admit FP16, INT8 or NVFP4 (Byte weight with Byte block
    # scale); packed W4 (Byte weight, FP16 G32 scale) keeps auto-tile Linear.
    precision = (ROOT / "src/fused/hyvla_linear_precision.h").read_text(encoding="utf-8")
    nvfp4 = precision[precision.index("} else if (dtype == at::kByte) {"):]
    assert "pair.second->scalar_type() != at::kByte" in nvfp4[:nvfp4.index("} else if", 10)]
