"""HyVLA HyViT2 Vision W4 (G32 RTN, quantized at load): host contracts only.

The all-W4 recipe and its runner route, the cold plan's four-lane Vision under
every budget and its refusals, the resolver's core map, the weight builder's
four-lane packing of every ViT projection, the bind dispatch, and the native
owner's declared op, its four-lane check and the row-bias order. None of this
is a board run or numerical evidence for the W4 Vision route.
"""
import ast
import os
import re
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from examples.common import config as COMMON
from examples.runners.hy_vla import optimized as runner
from rpu_backend.adapters.hy_vla import runtime, weights
from rpu_backend.adapters.hy_vla.runtime import (
    HY_VLA_ACTION_COMPONENT,
    HY_VLA_LANGUAGE_COMPONENT,
    HY_VLA_VISION_COMPONENT,
    prepare_hy_vla_cold_plan,
    resolve_hy_vla_execution,
)
from rpu_backend.api import hy_embodied as api
from rpu_backend.quant import precision_recipes
from rpu_backend.quant.int4_pgrp_pack import quantize_int4_group_wise
from test_qwen3_vl_awq_packing import inverse_rpu_scale, inverse_rpu_weight


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/configs/hy_vla/w4a16_rtn_g32.toml"
W4 = {"RPU_HY_VLA_W8A16": "0", "RPU_HY_VLA_W4A16": "vit,vlm,expert"}


@pytest.fixture(autouse=True)
def _no_hyvla_selectors(monkeypatch):
    for name in [name for name in os.environ
                 if name.startswith("RPU_HY_VLA_") or name == "RPU_RMSNORM_VWARP"]:
        monkeypatch.delenv(name)


def _with_budget(config, cores):
    if cores != 8:
        config.setdefault("rpu_execution", {})["model"] = {"num_cores": cores}
    return config


# ------------------------------------------------------------- recipe and runner


def test_recipe_routes_every_component_to_w4():
    family = precision_recipes.load(COMMON.PRECISION_RECIPES)["hy_vla"]
    item = family.recipes["w4a16_rtn_g32"]
    assert (item.label, item.selector) == ("w4a16", "w4a16_vision")
    assert {item.spec(component).key() for component in ("vision", "vlm", "action")} == {"w4a16-g32-rtn"}
    assert runner.precision_route("w4a16_vision") == "w4a16_vision"
    # The Vision-W8 W4 recipe keeps its own route.
    assert runner.precision_route("w4a16") == "w4a16"


@pytest.mark.parametrize("cores", (4, 6, 8))
def test_example_environment_puts_vision_on_four_lanes_at_every_budget(cores):
    config = _with_budget(COMMON.load_config(CONFIG), cores)
    assert COMMON.core_support(config) == {4: None, 6: None, 8: None}
    args = runner.example_arguments(config)
    assert args.precision == "w4a16_vision" and args.resident_fp8_scale_layers == 0
    env = runner.profile_environment(args)
    assert (env["RPU_HY_VLA_W8A16"], env["RPU_HY_VLA_W4A16"]) == ("0", "vit,vlm,expert")
    assert env["RPU_HY_VLA_VISION_KV_SPM_ONLY"] == env["RPU_HY_VLA_EXPERT_GATEUP_RESIDENT"] == "0"
    plan = prepare_hy_vla_cold_plan(env, num_cores=cores)
    w = plan.weights
    assert (w.vit_bits, w.vlm_text_bits, w.vlm_vision_bits, w.expert_bits) == (4, 4, 4, 4)
    assert w.vision_lanes == w.patch_embed_cores == 4
    assert w.attn_tp8 is (cores == 8) and not plan.native.vit.kv_spm_only
    if cores == 8:
        # Against the Vision-W8 W4 template only the precision and the
        # eight-core Vision KV residency change.
        base = runner.profile_environment(runner.example_arguments(COMMON.load_config(
            ROOT / "examples/configs/hy_vla/w4a16_rtn_g32_vision_w8a16.toml")))
        assert {key for key in env if env[key] != base[key]} == {
            "RPU_HY_VLA_W8A16", "RPU_HY_VLA_W4A16", "RPU_HY_VLA_VISION_KV_SPM_ONLY"}


def test_policy_dtype_passes_the_all_w4_selection_before_ownership(tmp_path, monkeypatch):
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
    policy = api.HyEmbodiedPolicy.from_checkpoint(tmp_path, dtype="w4a16-vision")
    with pytest.raises(Stop):
        policy.to("rpu")
    assert seen == {**W4, "cores": 8}
    assert "RPU_HY_VLA_W4A16" not in os.environ


# ------------------------------------------------------------- cold plan and core map


@pytest.mark.parametrize("cores", (4, 6, 8))
def test_cold_plan_refuses_the_eight_lane_vision_routes_for_w4(cores):
    assert prepare_hy_vla_cold_plan(W4, num_cores=cores).weights.vision_lanes == 4
    w8 = {"RPU_HY_VLA_W8A16": "all"}
    assert prepare_hy_vla_cold_plan(w8, num_cores=cores).weights.vision_lanes == (8 if cores == 8 else 4)
    match = "W4 Vision runs on four lanes" if cores == 8 else f"num_cores={cores} has no route"
    for name, value in (("RPU_HY_VLA_VISION_KV_SPM_ONLY", "1"), ("RPU_HY_VLA_VIT_PACKED", "0"),
                        ("RPU_HY_VLA_FUSED_MERGER", "0"), ("RPU_HY_VLA_MERGER_IN_GRAPH", "0"),
                        ("RPU_HY_VLA_PROJ1_IN_MERGER", "0")):
        with pytest.raises(ValueError, match=match):
            prepare_hy_vla_cold_plan({**W4, name: value}, num_cores=cores)
    if cores == 8:
        # Eight-lane W8 Vision keeps its SPM-resident KV.
        assert prepare_hy_vla_cold_plan(
            {**w8, "RPU_HY_VLA_VISION_KV_SPM_ONLY": "1"}, num_cores=8).native.vit.kv_spm_only


def test_resolver_and_runner_put_only_vision_on_four_lanes_at_budget_eight():
    _root, children = resolve_hy_vla_execution({}, entry_point="test", vision_lanes=4)
    assert children[HY_VLA_VISION_COMPONENT]["model"] == {"num_cores": 4}
    assert all("model" not in children[component]
               for component in (HY_VLA_LANGUAGE_COMPONENT, HY_VLA_ACTION_COMPONENT))
    assert (resolve_hy_vla_execution({}, entry_point="test", vision_lanes=8)[1]
            == resolve_hy_vla_execution({}, entry_point="test")[1])
    for budget in (4, 6):
        request = {"model": {"num_cores": budget}}
        assert (resolve_hy_vla_execution(request, entry_point="test", vision_lanes=4)[1]
                == resolve_hy_vla_execution(request, entry_point="test")[1])
    owner = runtime.HyVlaRunner.__new__(runtime.HyVlaRunner)
    owner.w, owner._core_budget = NS(vision_lanes=4), 8
    assert owner._vision_lanes == 4
    assert owner._component_cores(HY_VLA_LANGUAGE_COMPONENT) == 8
    assert owner._component_cores(HY_VLA_ACTION_COMPONENT) == 8


def test_reconfigure_and_rollback_keep_w4_vision_on_four_lanes_at_budget_eight():
    old_root, old_children = resolve_hy_vla_execution(None, entry_point="test", vision_lanes=4)
    new_root, _ = resolve_hy_vla_execution(
        {"components": {HY_VLA_LANGUAGE_COMPONENT: {"prefill": {"chunk_size": 240}}}},
        entry_point="test")
    owner = runtime.HyVlaRunner.__new__(runtime.HyVlaRunner)
    owner.w, owner._closed, owner._psubs = NS(vision_lanes=4), False, {}
    owner._rpu_execution, owner._rpu_execution_components = old_root, old_children
    owner._rpu_execution_component_generations = {component: 0 for component in old_children}
    owner._rpu_last_execution_receipts = {}
    owner.validate_execution_reconfigure = lambda _value: None
    owner.apply_execution_reconfigure(old_root, new_root, generation=1)
    children = owner._rpu_execution_components
    # Only the language chunk changed; Vision keeps its installed four lanes.
    assert children[HY_VLA_VISION_COMPONENT] == old_children[HY_VLA_VISION_COMPONENT]
    assert children[HY_VLA_VISION_COMPONENT]["model"] == {"num_cores": 4}
    assert owner._rpu_execution_component_generations == {
        HY_VLA_VISION_COMPONENT: 0, HY_VLA_LANGUAGE_COMPONENT: 1, HY_VLA_ACTION_COMPONENT: 0}
    owner._execution_reconfigure_journal["mutation_started"] = True
    owner.rollback_execution_reconfigure(old_root, new_root, 1)
    assert owner._rpu_execution_components == old_children


# ------------------------------------------------------------- weights and bind


class _Vit:
    def __getitem__(self, key):
        shape = {"attn.qkv.weight": (3 * 16 * 72, 1152), "attn.qkv.bias": (3 * 16 * 72,),
                 "attn.proj.weight": (1152, 16 * 72), "mlp.fc1.weight": (4304, 1152),
                 "mlp.fc1.bias": (4304,), "mlp.fc2.weight": (1152, 4304)}.get(
                     key.split(".", 2)[2], (1152,))
        return torch.zeros(shape)


@pytest.mark.parametrize("cores", (4, 6, 8))
def test_weight_builder_packs_every_vit_projection_on_four_lanes(monkeypatch, cores):
    calls = []
    monkeypatch.setattr(weights, "_quant", lambda raw, partition, lanes, bits: (
        calls.append((tuple(raw.shape), partition, lanes, bits)), (raw, raw))[1])
    monkeypatch.setattr(weights, "_rpu16", lambda value: value)
    cold = weights._HyVlaWeightColdPlan.from_environment(
        {**W4, **({"RPU_HY_VLA_ATTN_TP8": "1"} if cores == 8 else {})}, num_cores=cores)
    out = weights._build_vit(_Vit(), weights.HyVlaConfig(vit_layers=1), cold)
    assert calls == [((1280, 1152), 1, 4, 4)] * 3 + [((1152, 1280), 0, 4, 4),
                     ((4352, 1152), 1, 4, 4), ((1152, 4352), 0, 4, 4)]
    assert all(len(out[name]) == 1 for name in ("qws", "kws", "vws", "ows", "f1ws", "f2ws"))


# (name, N, K, partition): the padded HyViT2 projections at four lanes.
_GEOMETRIES = [("q/k/v", 1280, 1152, 1), ("o", 1152, 1280, 0),
               ("fc1", 4352, 1152, 1), ("fc2", 1152, 4352, 0)]


@pytest.mark.parametrize("name,n,k,partition", _GEOMETRIES, ids=[g[0] for g in _GEOMETRIES])
def test_four_lane_packing_decodes_to_the_rtn_weights(name, n, k, partition):
    lanes = 4
    w = (((torch.arange(n * k) * 7919) % 1021 - 510).float() / 4096).reshape(n, k)
    w[:, :32] *= 40
    packed, scale = weights._pack_w4_g32(w, partition, lanes)
    codes, scale_gn, _ = quantize_int4_group_wise(w.half(), 32)
    assert packed.dtype == torch.uint8 and tuple(packed.shape) == (n, k // 2)
    # The native check: aligned per-core geometry and the exact G32 payload.
    local_n, local_k = (n // lanes, k) if partition == 1 else (n, k // lanes)
    assert local_n % 16 == 0 and local_k % 64 == 0
    assert scale.size(0) == 32
    assert scale.numel() == lanes * ((local_k // 32 + 3) // 4) * ((local_n + 63) // 64) * 256
    assert torch.equal(inverse_rpu_weight(packed, n, k, partition, lanes), codes)
    assert torch.equal(inverse_rpu_scale(scale, n, k, partition, lanes), scale_gn.t())
    if partition == 0:
        # Eight row lanes would leave 160/544 K per core.
        assert (k // 8) % 64


def test_bind_installs_packed_w4_on_four_lanes(monkeypatch):
    tree = ast.parse((ROOT / "python/rpu_backend/adapters/hy_vla/runtime.py").read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HyVlaRunner")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare_vit")
    bind = next(n for n in method.body if isinstance(n, ast.FunctionDef) and n.name == "bind")
    value = object()
    cfg = NS(vit_heads=16, vit_head_dim_padded=80, vit_hidden=1152, vit_inter=4304,
             vit_inter_padded=4352, proj_dim=2048, vit_eps=1e-6)
    vit = {name: [value] for name in (
        "qw kw vw ow f1w f2w n1w n1b n2w n2b qb kb vb ob f1b f2b qws kws vws ows f1ws f2ws").split()}
    vit["qw"] = [NS(dtype=torch.uint8)]
    w = NS(vit=vit, vit_proj1_w=value, vit_proj1_b=value, patch_embed_cores=4,
           vit_patch_gemm_w=value, vit_pos_fused=value)
    events = []
    runner_ = NS(_native_cold=NS(vit=NS(fast_replay=False, fast_replay_preload=False,
                                        mask_once=False, kvpad16=False, q_inplace=False)),
                 _fused_merger=True, _proj1_in_merger=True)
    namespace = dict(torch=torch, c=cfg, w=w, cache=NS(max_seq_len=588), self=runner_, lanes=4,
                     _require_native_topology=lambda stage, topology, expected: events.append(
                         ("topology", stage, expected)))
    exec(compile(ast.Module(body=[bind], type_ignores=[]), "<actual-hyvit-bind>", "exec"), namespace)
    for op in ("hyvit2_set_runtime_config", "hyvit2_set_weights", "hyvit2_set_weights_w8a16",
               "hyvit2_set_weights_w4a16", "hyvit2_model_set_patch_emb"):
        monkeypatch.setattr(torch.ops.rpu, op, lambda *args, op=op: events.append(
            (op, len(args))), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "hyvit2_set_execution_core_count",
                        lambda handle, lanes: events.append(("cores", lanes)), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "hyvit2_get_execution_topology", lambda handle: (), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "hyvit2_set_chunk_envelope",
                        lambda handle, capacity, chunk: events.append(("envelope", capacity)), raising=False)
    namespace["bind"](7)
    # The 28 FP16/W8 arguments plus the six G32 scale lists.
    assert events == [("cores", 4), ("hyvit2_set_runtime_config", 6), ("hyvit2_set_weights_w4a16", 34),
                      ("topology", "Vision", (1, 4, 4, 4, 1, 8, 4304, 4352)),
                      ("envelope", 588), ("hyvit2_model_set_patch_emb", 5)]


# ------------------------------------------------------------- native owner


def test_native_owner_declares_w4_on_four_lanes_and_adds_row_bias_before_the_reduction():
    source = (ROOT / "src/fused/rpu_hyvit2_vision_model.cpp").read_text(encoding="utf-8")
    check = source[source.index("static void check_hyvit2_w4a16_scale_lists("):]
    check = check[:check.index("\n}\n")]
    # Column q/k/v/fc1 and row o/fc2 at the actual lanes, with the G32 payload size.
    assert "n % (col ? 16 * cores : 16) == 0" in check and "k % (col ? 64 : 64 * cores) == 0" in check
    assert "s.size(0) == 32" in check and "s.numel() == expected" in check
    for role, col in (("q", "true"), ("k", "true"), ("v", "true"), ("o", "false"),
                      ("fc1", "true"), ("fc2", "false")):
        assert re.search(rf"check_projection\({role}_w,\s+{role}_ws,\s+{col},", check), role
    body = source[source.index("if (packed_w4) {"):]
    assert re.match(r"if \(packed_w4\) \{(\s+//[^\n]*)+\s+TORCH_CHECK\(num_cores\(\) == 4 && N == 27", body)
    op = source[source.index("void rpu_hyvit2_set_weights_w4a16("):]
    assert "/*packed_w4=*/true);" in op[:op.index("\n}\n")]
    # One precision for every layer: FP16, INT8 (bits 0-5) or packed W4 (bits 6-11).
    assert "if (w4) layer_mask |= 1LL << (6 + j);" in source
    assert "precision_mask == 0 || precision_mask == 0x3f ||" in source
    assert "precision_mask == 0xfc0" in source
    # The packed W4 row Linear refuses in-kernel bias: launch without it, add the
    # core-0-only row bias to every partial, then reduce.
    for flag, bias in (("w4_o", "o_bias"), ("w4_fc2", "fc2_bias")):
        launch = source.index(f'{flag} ? 0 : layer_addr(layer_idx, 0, "{bias}")')
        add = source.index(f'if ({flag}) add_row_bias_partial(layer_addr(layer_idx, 0, "{bias}")')
        reduce = source.index("rpu_launch_all_reduce_sum_residual_kernel(", add)
        assert launch < add < reduce, flag
    for path in ("src/core/rpu_kernel_decls.h", "src/core/rpu_dispatch_registrations.inc"):
        assert "hyvit2_set_weights_w4a16" in (ROOT / path).read_text(encoding="utf-8"), path
