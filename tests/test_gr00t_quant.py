"""GR00T N1.7 G32 W4A16, with W8 or W4 Vision: host contracts only.

Recipe and profile binding of both W4 recipes, the cold environment the runner
selects, the refusals before any weight loads (Vision W4 also checks the tower),
the G32 pgrp packing of the text prefix, VL encoder and DiT at the lanes they
run on, and the DiT / VL encoder owners' G32 checks and row-bias order. The W4
Vision blocks themselves are V1's packing (test_qwen3_vl_vision_w4.py). None of
this is a board run or numerical evidence for the W4 routes.
"""
import copy
import inspect
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from examples.common import config as COMMON
from examples.runners.gr00t import config as runner_config
from rpu_backend.adapters.gr00t import runtime as gr00t
from rpu_backend.quant.int4_pgrp_pack import quantize_int4_group_wise
from test_qwen3_vl_awq_packing import exact_config, inverse_rpu_scale, inverse_rpu_weight


ROOT = Path(__file__).resolve().parents[1]
DROID = ROOT / "examples/configs/gr00t/n1_7_droid"
RECIPE = "w4a16_rtn_g32_vision_glue_adaln_w8a16"
PROFILE = "gr00t-n1d7-3b.w4a16-g32.vla"
VISION_RECIPE = "w4a16_rtn_g32_glue_adaln_w8a16"
VISION_PROFILE = "gr00t-n1d7-3b.w4a16-g32-vision.vla"
# (recipe, profile, variant, runner mode); both recipes carry selector w4a16.
W4_ENTRIES = ((RECIPE, PROFILE, "default-w4a16-g32", "w4a16"),
              (VISION_RECIPE, VISION_PROFILE, "default-w4a16-g32-vision", "w4a16-vision"))
W4_REFUSED = ("RPU_GR00T_FF1_RESIDENT_W8", "RPU_GR00T_SELF_QKV_MULTIBASE", "RPU_GR00T_RING_XOR3")


@pytest.fixture(autouse=True)
def _no_gr00t_selectors(monkeypatch):
    for name in [name for name in os.environ if name.startswith("RPU_GR00T_")]:
        monkeypatch.delenv(name)


# ------------------------------------------------------------- profiles and recipes


@pytest.mark.parametrize("recipe,profile_id,variant,mode", W4_ENTRIES, ids=("w4", "w4-vision"))
@pytest.mark.parametrize("size", ("224", "256"))
def test_quant_profiles_bind_exact_recipe_and_cores(size, recipe, profile_id, variant, mode):
    config = COMMON.load_config(DROID / size / f"{recipe}.toml")
    profiles = COMMON.planner_profiles()
    profile = profiles[config["example"]["profile_id"]]
    w8 = profiles["gr00t-n1d7-3b.w8a16.vla"]
    assert profile["id"] == profile_id
    assert (profile["precision"], profile["variant"], profile["family"]) == (
        "w4a16", variant, "gr00t")
    assert profile["support"]["status"] == "IMPLEMENTED_UNVERIFIED_HW"
    limits = profile["input_envelope"]["limits"]
    assert limits == {**w8["input_envelope"]["limits"], "group_size": 32}
    assert limits["allowed_num_cores"] == [4, 6, 8]
    assert runner_config.precision_mode(config["example"]) == mode
    assert COMMON.core_support(config) == {4: None, 6: None, 8: None}
    for cores in (4, 6, 8):
        COMMON._validate_execution({"model": {"num_cores": cores}}, profile=profile)
    for cores in (2, 5, 7):
        with pytest.raises((ValueError, COMMON.ConfigError)):
            COMMON._validate_execution({"model": {"num_cores": cores}}, profile=profile)
    unknown = copy.deepcopy(profile)
    unknown["id"] = "gr00t-n1d7-3b.other.vla"
    with pytest.raises(COMMON.ConfigError, match="registered FP16, W8A16 or W4A16 VLA"):
        COMMON._validate_execution({"model": {"num_cores": 6}}, profile=unknown)
    sibling = VISION_RECIPE if recipe == RECIPE else RECIPE
    for other in ("w8a16", "fp16", sibling):
        with pytest.raises(COMMON.ConfigError, match="not registered for profile"):
            COMMON.precision_recipe("gr00t", other, profile["id"])


def test_shared_w4_selector_resolves_only_through_the_profile():
    with pytest.raises(COMMON.ConfigError, match="'w4a16' names 2 recipes, not one"):
        runner_config.precision_mode({"dtype": "w4a16"})
    with pytest.raises(COMMON.ConfigError, match="whose selector is not 'w4a16'"):
        runner_config.precision_mode({"dtype": "w4a16", "profile_id": "gr00t-n1d7-3b.w8a16.vla"})
    # A selector one recipe carries still names it alone; the profile is not read.
    assert runner_config.precision_mode({"dtype": "w8a16", "profile_id": "unregistered"}) == "w8a16"


def test_quantized_recipes_declare_the_int8_adaln_the_builder_installs():
    registry = json.loads((ROOT / "manifests/precision_recipes.v1.json").read_text())
    recipes = registry["families"]["gr00t"]["recipes"]
    # The builder quantizes AdaLN (norm1.linear) to INT8 whenever W8A16 is on;
    # only the merger stays FP16. Vision follows the recipe: W8, or G32 W4.
    w4 = {"format": "w4a16", "group": 32, "method": "rtn"}
    for name, vision in (("w8a16", "w8a16"), (RECIPE, "w8a16"), (VISION_RECIPE, w4)):
        rules = {rule["component"]: rule["spec"] for rule in recipes[name]["rules"]}
        assert (rules["adaln"], rules["merger"], rules["vision"]) == ("w8a16", "fp16", vision), name
    w8 = COMMON.load_config(DROID / "224" / "w8a16.toml")
    assert runner_config.precision_mode(w8["example"]) == "w8a16"


# ------------------------------------------------------------- cold environment


@pytest.mark.parametrize("recipe,vision", ((RECIPE, "0"), (VISION_RECIPE, "1")), ids=("w4", "w4-vision"))
@pytest.mark.parametrize("cores", (4, 6, 8))
def test_runtime_environment_selects_w4_without_w8_only_routes(cores, recipe, vision):
    config = COMMON.load_config(DROID / "224" / f"{recipe}.toml")
    if cores != 8:
        config.setdefault("rpu_execution", {})["model"] = {"num_cores": cores}
    env = runner_config.runtime_environment(config)
    assert (env["RPU_GR00T_W8A16"], env["RPU_GR00T_W4A16"]) == ("1", "1")
    assert env["RPU_GR00T_VISION_W4A16"] == vision
    assert env["RPU_QWEN3VL_VISION_BATCH"] == ("0" if vision == "1" and cores != 8 else "1")
    assert {env[name] for name in W4_REFUSED} == {"0"}
    assert env["RPU_GR00T_QWEN3VL_SPM_Z1"] == "0"
    w8 = COMMON.load_config(DROID / "224" / "w8a16.toml")
    if cores != 8:
        w8.setdefault("rpu_execution", {})["model"] = {"num_cores": cores}
    w8_env = runner_config.runtime_environment(w8)
    assert (w8_env["RPU_GR00T_W4A16"], w8_env["RPU_GR00T_VISION_W4A16"]) == ("0", "0")
    assert runner_config.precision_environment("fp16") == {
        "RPU_GR00T_W8A16": "0", "RPU_GR00T_W4A16": "0", "RPU_GR00T_VISION_W4A16": "0"}


def test_builder_refuses_w4_conflicts_before_loading(monkeypatch):
    # Eight cores: these selectors exist there, so the W4 refusal is what fires.
    monkeypatch.delenv("RPU_QWEN3VL_VISION_FUSED_MERGER", raising=False)

    def build():
        gr00t.build_gr00t_vla("/missing/ckpt", "/missing/qwen3vl")

    monkeypatch.setenv("RPU_GR00T_W4A16", "1")
    with pytest.raises(ValueError, match="RPU_GR00T_W4A16=1 requires RPU_GR00T_W8A16=1"):
        build()
    monkeypatch.setenv("RPU_GR00T_W8A16", "1")
    for name in W4_REFUSED:
        monkeypatch.setenv(name, "1")
        with pytest.raises(ValueError, match=f"RPU_GR00T_W4A16 has no route for {name}"):
            build()
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("RPU_GR00T_W4A16", "0")
    assert gr00t._w4a16_on() is False


def test_builder_refuses_vision_w4_without_w4_before_loading(monkeypatch):
    monkeypatch.delenv("RPU_QWEN3VL_VISION_FUSED_MERGER", raising=False)

    def build():
        gr00t.build_gr00t_vla("/missing/ckpt", "/missing/qwen3vl")

    monkeypatch.setenv("RPU_GR00T_VISION_W4A16", "1")
    for w8 in ("0", "1"):
        monkeypatch.setenv("RPU_GR00T_W8A16", w8)
        with pytest.raises(ValueError, match="RPU_GR00T_VISION_W4A16=1 requires RPU_GR00T_W4A16=1"):
            build()
    # Vision W4 is W4: it inherits the W8 requirement and every W4 refusal.
    monkeypatch.setenv("RPU_GR00T_W4A16", "1")
    monkeypatch.setenv("RPU_GR00T_W8A16", "0")
    with pytest.raises(ValueError, match="RPU_GR00T_W4A16=1 requires RPU_GR00T_W8A16=1"):
        build()
    monkeypatch.setenv("RPU_GR00T_W8A16", "1")
    for name in W4_REFUSED:
        monkeypatch.setenv(name, "1")
        with pytest.raises(ValueError, match=f"RPU_GR00T_W4A16 has no route for {name}"):
            build()
        monkeypatch.setenv(name, "0")


def test_vision_installer_precision_follows_the_selectors(monkeypatch):
    assert gr00t._gr00t_vision_precision() == {"w8a16": False}
    monkeypatch.setenv("RPU_GR00T_W8A16", "1")
    assert gr00t._gr00t_vision_precision() == {"w8a16": True}
    monkeypatch.setenv("RPU_GR00T_W4A16", "1")
    assert gr00t._gr00t_vision_precision() == {"w8a16": True}
    monkeypatch.setenv("RPU_GR00T_VISION_W4A16", "1")
    assert gr00t._gr00t_vision_precision() == {"w4a16": True}
    # The installer takes exactly these flags; it refuses w4a16 together with w8a16.
    assert "**vision_precision," in inspect.getsource(gr00t.build_gr00t_vla)


class _Built(Exception):
    pass


@pytest.mark.parametrize("size", (2, 8))
def test_builder_checks_the_vision_tower_before_building(size, tmp_path, monkeypatch):
    import transformers

    # Present already, so the builder's setdefaults leave the environment as it was.
    for name in ("RPU_QWEN3VL_VISION_ROPE_SPM", "RPU_GR00T_PARTIAL_MROPE", "RPU_WALL_OSS_FAST_REPLAY",
                 "RPU_FASTREPLAY_SKIP_SYNC", "RPU_DEEP_FAST_REPLAY"):
        monkeypatch.setenv(name, "1")
    for name in ("RPU_GR00T_W8A16", "RPU_GR00T_W4A16", "RPU_GR00T_VISION_W4A16"):
        monkeypatch.setenv(name, "1")
    exact_config(size).save_pretrained(tmp_path)

    def built(_cfg):
        raise _Built

    monkeypatch.setattr(transformers, "Qwen3VLForConditionalGeneration", built)
    if size == 8:
        with pytest.raises(ValueError, match="requires the exact Qwen3-VL-2B Vision tower"):
            gr00t.build_gr00t_vla("/missing/ckpt", str(tmp_path))
    else:
        with pytest.raises(_Built):
            gr00t.build_gr00t_vla("/missing/ckpt", str(tmp_path))


# ------------------------------------------------------------- G32 packing


# (name, N, K, partition, lanes): the real DiT (H1536/FF6144, cross K2048) and VL
# encoder (D2048/FF8192) block projections at the four and eight lanes they run on.
_GEOMETRIES = [
    (f"{name} x{lanes}", n, k, partition, lanes)
    for lanes in (4, 8)
    for name, n, k, partition in (
        ("dit q", 1536, 1536, 1), ("dit cross k", 1536, 2048, 1), ("dit o", 1536, 1536, 0),
        ("dit ff1", 6144, 1536, 1), ("dit ff2", 1536, 6144, 0),
        ("vl q", 2048, 2048, 1), ("vl o", 2048, 2048, 0),
        ("vl ff0", 8192, 2048, 1), ("vl ff2", 2048, 8192, 0))
]


def _weight(n, k):
    w = (((torch.arange(n * k) * 7919) % 1021 - 510).float() / 4096).reshape(n, k)
    w[:, :32] *= 40  # one wide group: its scale must not spread to the next group
    return w


@pytest.mark.parametrize("name,n,k,partition,lanes", _GEOMETRIES, ids=[g[0] for g in _GEOMETRIES])
def test_block_packing_decodes_to_the_rtn_weights(name, n, k, partition, lanes):
    w = _weight(n, k)
    packed, scale = gr00t._pack_gr00t_w4(w, partition, lanes)
    codes, scale_gn, _ = quantize_int4_group_wise(w.half(), 32)
    assert packed.dtype == torch.uint8 and tuple(packed.shape) == (n, k // 2)
    assert scale.dtype == torch.float16 and scale.dim() == 2 and scale.size(0) == 32
    # The native set_weights check: aligned per-lane geometry and the exact G32 count.
    local_n, local_k = (n // lanes, k) if partition == 1 else (n, k // lanes)
    assert local_n % 16 == 0 and local_k % 64 == 0
    assert scale.numel() == lanes * ((local_k // 32 + 3) // 4) * ((local_n + 63) // 64) * 256
    assert torch.equal(inverse_rpu_weight(packed, n, k, partition, lanes), codes)
    assert torch.equal(inverse_rpu_scale(scale, n, k, partition, lanes), scale_gn.t())


class _TextLayer(nn.Module):
    # One Qwen3-VL-2B text layer: Q16/KV8 x D128, H2048, I6144.
    def __init__(self, dtype=torch.float16):
        super().__init__()
        self.self_attn, self.mlp = nn.Module(), nn.Module()
        shapes = {"self_attn.q_proj": (2048, 2048), "self_attn.k_proj": (1024, 2048),
                  "self_attn.v_proj": (1024, 2048), "self_attn.o_proj": (2048, 2048),
                  "mlp.gate_proj": (6144, 2048), "mlp.up_proj": (6144, 2048),
                  "mlp.down_proj": (2048, 6144)}
        for role, (n, k) in shapes.items():
            owner, leaf = role.split(".")
            linear = nn.Linear(k, n, bias=False)
            linear.weight = nn.Parameter(_weight(n, k).to(dtype), requires_grad=False)
            setattr(getattr(self, owner), leaf, linear)


@pytest.mark.parametrize("attn,mlp", ((8, 8), (4, 4), (4, 6)), ids=("tp8", "core4", "core6"))
def test_text_prefix_packs_every_projection_at_its_topology(attn, mlp):
    text = SimpleNamespace(layers=[_TextLayer()])
    topology = None if attn == 8 else SimpleNamespace(attn_tp=attn, mlp_tp=mlp)
    source = {role: text.layers[0].get_submodule(role).weight.clone()
              for role in ("self_attn.k_proj", "mlp.down_proj")}
    gr00t._quantize_backbone_w4_inplace(text, topology)
    gr00t._mark_backbone_w4_layout(text, topology)
    for role, partition in (("self_attn.q_proj", 1), ("self_attn.k_proj", 1),
                            ("self_attn.v_proj", 1), ("self_attn.o_proj", 0),
                            ("mlp.gate_proj", 1), ("mlp.up_proj", 1), ("mlp.down_proj", 0)):
        linear = text.layers[0].get_submodule(role)
        lanes = mlp if role.startswith("mlp.") else attn
        assert linear.weight.dtype == torch.uint8
        assert (linear._rpu_linear_partition, linear._rpu_linear_num_cores) == (partition, lanes)
        # The reduced decoder validates the registered buffer, not a copy.
        assert "weight_scale" in dict(linear.named_buffers())
    for role, partition in (("self_attn.k_proj", 1), ("mlp.down_proj", 0)):
        linear = text.layers[0].get_submodule(role)
        lanes = mlp if role.startswith("mlp.") else attn
        n, k = source[role].shape
        codes, _, _ = quantize_int4_group_wise(source[role], 32)
        assert tuple(linear.weight.shape) == (n, k // 2)  # I6144 is never padded
        assert torch.equal(inverse_rpu_weight(linear.weight, n, k, partition, lanes), codes)


def test_text_prefix_checks_every_projection_before_packing():
    layers = [_TextLayer(), _TextLayer()]
    layers[1].mlp.down_proj.weight = nn.Parameter(
        layers[1].mlp.down_proj.weight.float(), requires_grad=False)
    with pytest.raises(TypeError, match="layer1.mlp.down_proj requires its FP16 weight"):
        gr00t._quantize_backbone_w4_inplace(SimpleNamespace(layers=layers))
    assert layers[0].self_attn.q_proj.weight.dtype == torch.float16


# ------------------------------------------------------------- native admission


def test_native_owners_admit_g32_and_add_row_bias_before_the_reduction():
    # The text prefix admission runs in test_qwen3_spm_kv_by_mha_source_contract's
    # native harness; the DiT and VL encoder owners are checked here.
    for name, ff1 in (("rpu_gr00t_dit_model.cpp", "ff1"), ("rpu_gr00t_vl_encoder_model.cpp", "ff0")):
        source = (ROOT / "src/fused" / name).read_text(encoding="utf-8")
        assert "static void check_gr00t_w4a16_projection(" in source, name
        for role, partition in (("to_q", 1), ("to_k", 1), ("to_v", 1), ("to_out", 0),
                                (ff1, 1), ("ff2", 0)):
            assert re.search(rf'"{role}",\s+{partition}, cores\);', source), (name, role)
        # The packed W4 row Linear refuses in-kernel bias: launch without it, add the
        # core-0-only row bias to every partial, then reduce.
        for flag, bias in (("w4_o", "out_bias"), ("w4_ff2", "ff2_bias")):
            launch = source.index(f"{flag} ? 0 : layer_addr(layer_idx, 0, \"{bias}\")")
            add = source.index(f"if ({flag}) add_row_bias_partial(layer_addr(layer_idx, 0, \"{bias}\")")
            reduce = source.index("rpu_launch_all_reduce_sum_residual_kernel(", add)
            assert launch < add < reduce, (name, flag)
    dit = (ROOT / "src/fused/rpu_gr00t_dit_model.cpp").read_text(encoding="utf-8")
    assert 'check_gr00t_w8a16_scale_list(N, adaln_w_list, adaln_scale_list, c, "adaln_w");' in dit
