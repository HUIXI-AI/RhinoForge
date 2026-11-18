"""LingBot v2 public templates at a cold 4/6/8-core budget: host contracts only.

The example TOMLs, the composite resolver, the eight-core selector refusals,
the runner environment, the cold-only reconfigure and the action-expert lane
layout are checked here with small doubles. None of this is a board run or
numerical evidence for the reduced-core kernels.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.common import config as common
from examples.runners.lingbot2 import optimized as runner
import rpu_backend.adapters.lingbot_vla_v2.convert as convert
import rpu_backend.adapters.lingbot_vla_v2.runtime as runtime
from rpu_backend.adapters.lingbot_vla_v2.runtime import (
    LINGBOT2_ACTION_COMPONENT,
    LINGBOT2_TEXT_COMPONENT,
    LINGBOT2_VISION_COMPONENT,
    resolve_lingbot2_execution_components,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = sorted((ROOT / "examples/configs/lingbot/v2").glob("*.toml"))


@pytest.fixture(autouse=True)
def _no_lingbot2_selectors(monkeypatch):
    # Other LingBot2 tests apply profile selectors to the process environment.
    for name in (*runtime._LINGBOT2_EIGHT_CORE_SELECTORS, "RPU_LINGBOT2_SPARSE_EXPERTS",
                 "RPU_LINGBOT2_BASE_W8A16"):
        monkeypatch.delenv(name, raising=False)


def _cores(child):
    return child.get("model", {}).get("num_cores", 8)


# --------------------------------------------------------------------------- examples


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: path.name)
def test_every_public_template_admits_each_budget(path):
    assert common.core_support(common.load_config(path)) == {4: None, 6: None, 8: None}


def test_exact_z2_stays_eight_core():
    support = common.core_support(common.load_config(
        ROOT / "examples/configs/lingbot/legacy/exact_z2/6b/w8a16.toml"))
    assert support[8] is None and "exact-Z2" in support[4] and support[4] == support[6]


@pytest.mark.parametrize(("request_", "match"), [
    ({"model": {"num_cores": 5}}, "4, 6 or 8"),
    ({"components": {"language_model": {"model": {"num_cores": 4}}}}, "model"),
])
def test_runner_refuses_other_and_child_budgets(request_, match):
    with pytest.raises(ValueError, match=match):
        runner.execution_config(request_)


@pytest.mark.parametrize("precision", ("fp16", "w8a16"))
@pytest.mark.parametrize("cores", (4, 6))
def test_reduced_runner_environment_is_admitted_by_the_builder(monkeypatch, precision, cores):
    env = runner.profile_environment(precision, cores)
    eight = runner.profile_environment(precision)
    assert eight == runner.profile_environment(precision, 8)
    # Only the eight-core selectors change; the sparse expert route stays on.
    assert {name for name in env if env[name] != eight[name]} == {
        name for name in runner.EIGHT_CORE_SELECTORS if eight[name] != "0"}
    assert env["RPU_LINGBOT2_SPARSE_EXPERTS"] == "1"
    for name in runtime._LINGBOT2_EIGHT_CORE_SELECTORS:
        monkeypatch.setenv(name, env.get(name, "0"))
    monkeypatch.setenv("RPU_LINGBOT2_SPARSE_EXPERTS", env["RPU_LINGBOT2_SPARSE_EXPERTS"])
    runtime._validate_lingbot2_reduced_environment(cores)
    # The eight-core template turns at least one of them on, so it is refused.
    for name in runtime._LINGBOT2_EIGHT_CORE_SELECTORS:
        monkeypatch.setenv(name, eight.get(name, "0"))
    with pytest.raises(ValueError, match="refuses eight-core-only selectors"):
        runtime._validate_lingbot2_reduced_environment(cores)


def test_runner_reports_and_forwards_the_budget():
    assert runner.core_budget(runner.execution_config({"model": {"num_cores": 6}})) == 6
    assert runner.core_budget(runner.execution_config(None)) == 8


# --------------------------------------------------------------------------- resolver


@pytest.mark.parametrize("budget", (4, 6, 8))
def test_resolver_derives_each_child_budget(budget):
    request = {} if budget == 8 else {"model": {"num_cores": budget}}
    root, vision, text, action = resolve_lingbot2_execution_components(
        request, entry_point="test")
    assert runtime._lingbot2_core_budget(root) == budget
    if budget == 8:
        assert all("model" not in child for child in (vision, text, action))
        return
    assert _cores(text) == budget
    assert _cores(vision) == _cores(action) == 4
    assert text["prefill"]["chunk_size"] == "auto"
    assert runtime._lingbot2_component_cores(budget, LINGBOT2_TEXT_COMPONENT) == budget
    for component in (LINGBOT2_VISION_COMPONENT, LINGBOT2_ACTION_COMPONENT):
        assert runtime._lingbot2_component_cores(budget, component) == 4


@pytest.mark.parametrize(("request_", "match"), [
    ({"model": {"num_cores": 5}}, "4, 6 or 8"),
    ({"components": {LINGBOT2_TEXT_COMPONENT: {"model": {"num_cores": 4}}}}, "model"),
])
def test_resolver_refuses_other_and_child_budgets(request_, match):
    with pytest.raises(ValueError, match=match):
        resolve_lingbot2_execution_components(request_, entry_point="test")


def test_reconfigure_keeps_the_cold_budget():
    policy = runtime.LingbotVlaV2Policy.__new__(runtime.LingbotVlaV2Policy)
    policy._closed = False
    policy._z2_state = {}
    policy._core_budget = 6
    for change in ({}, {"model": {"num_cores": 4}}):
        with pytest.raises(ValueError, match="cold-only"):
            policy.validate_execution_reconfigure(change)


def test_each_child_graph_policy_bounds_its_own_core_count():
    from rpu_backend.graph import GraphRuntimePolicy

    policy = GraphRuntimePolicy.from_environment()
    assert runtime._lingbot2_graph_policy(policy, policy.execution_core_count) is policy
    reduced = runtime._lingbot2_graph_policy(policy, 4)
    assert reduced.execution_core_count == 4
    assert reduced.provenance[-1] == "owner:execution_core_count"


# --------------------------------------------------------------------------- refusals


@pytest.mark.parametrize("selector", runtime._LINGBOT2_EIGHT_CORE_SELECTORS)
@pytest.mark.parametrize("cores", (4, 6))
def test_reduced_build_refuses_each_eight_core_selector(monkeypatch, selector, cores):
    monkeypatch.setenv("RPU_LINGBOT2_SPARSE_EXPERTS", "1")
    monkeypatch.setenv(selector, "1")
    with pytest.raises(ValueError, match=f"model.num_cores={cores} refuses .*{selector}"):
        runtime.build_lingbot_vla_v2(
            "/missing/ckpt", rpu_execution={"model": {"num_cores": cores}})


def test_reduced_build_requires_sparse_experts_and_eight_is_unchanged(monkeypatch):
    with pytest.raises(ValueError, match="requires RPU_LINGBOT2_SPARSE_EXPERTS=1"):
        runtime._validate_lingbot2_reduced_environment(4)
    monkeypatch.setenv("RPU_LINGBOT2_RUNTIME_GRAPH", "1")
    runtime._validate_lingbot2_reduced_environment(8)


# --------------------------------------------------------------------------- expert lanes


def _expert_doubles(monkeypatch):
    from rpu_backend.runtime import weights

    swizzles, packers, creates = [], [], []

    def swizzle(kind):
        def record(value, cores, **_kwargs):
            swizzles.append((kind, tuple(value.shape), cores))
            return value
        return record

    def packer(name, result):
        def record(*_args, num_cores, **_kwargs):
            packers.append((name, num_cores))
            return result
        return record

    monkeypatch.setattr(weights, "tp_col_swizzle_mc_weight", swizzle("col"))
    monkeypatch.setattr(weights, "tp_row_swizzle_mc_weight", swizzle("row"))
    monkeypatch.setattr(convert, "_h", lambda value: value)
    monkeypatch.setattr(convert, "_q", lambda value: value)
    monkeypatch.setattr(convert, "_build_rope_tables", lambda *_args: (None, None))
    monkeypatch.setattr(convert, "load_router_correction_biases",
                        lambda _gw, geom: [torch.zeros(geom["NUM_EXPERTS"])] * geom["EXP_LAYERS"])
    monkeypatch.setattr(convert, "pack_shared_gate_up", packer(
        "shared", (torch.zeros(1536, 768), None, None, None)))
    monkeypatch.setattr(convert, "pack_sparse_expert_gate_up", packer("gate_up", (torch.zeros(1), None)))
    monkeypatch.setattr(convert, "pack_sparse_expert_projection", packer("down", (torch.zeros(1), None)))
    monkeypatch.setattr(torch.ops.rpu, "lingbot_v2_moe_create",
                        lambda *args, **kwargs: (creates.append(kwargs), 7)[1], raising=False)
    for name in ("set_weights", "set_base_scales", "set_sparse_weights",
                 "set_shared_gate_up_weights"):
        monkeypatch.setattr(torch.ops.rpu, f"lingbot_v2_moe_{name}",
                            lambda *args, **kwargs: None, raising=False)
    return swizzles, packers, creates


def _build_expert(num_cores):
    geom = {**convert._default_geom(), "EXP_LAYERS": 1}

    def gw(name):
        if ".mlp.experts." in name:
            return torch.zeros(geom["NUM_EXPERTS"], 4, 4)
        return torch.zeros(4, 4)

    config = SimpleNamespace(sparse_experts=True, sparse_w8a16=False, bufonly=False,
                             addr=False, schunk=32512, rchunk_requested=0,
                             dense_soft_router=False, fp16_top4=True, routed_only=False,
                             down_acc16=False)
    return convert.build_expert_moe(gw, max_seq=16, rope_theta=1e6, runtime_config=config,
                                    geom=geom, num_cores=num_cores)


def test_reduced_expert_swizzles_every_owner_on_four_lanes(monkeypatch):
    swizzles, packers, creates = _expert_doubles(monkeypatch)
    assert _build_expert(4)[0] == 7
    # Attention, shared Gate/Up/Down and dense expert banks on four lanes;
    # the router keeps its single-core layout.
    assert {cores for _kind, _shape, cores in swizzles} == {4, convert.ROUTER_CORES}
    assert ("shared", 4) not in packers and ("shared", 8) not in packers
    assert set(packers) == {("gate_up", 4), ("down", 4)}
    assert creates == [{"num_cores": 4}]


def test_eight_core_expert_keeps_its_call_and_joint_shared_route(monkeypatch):
    swizzles, packers, creates = _expert_doubles(monkeypatch)
    _build_expert(8)
    assert {cores for _kind, _shape, cores in swizzles} == {8, convert.ROUTER_CORES}
    assert set(packers) == {("shared", 8), ("gate_up", 8), ("down", 8)}
    assert creates == [{}]


@pytest.mark.parametrize("cores", (6, 2, True))
def test_expert_refuses_other_lane_counts(cores):
    with pytest.raises(ValueError, match="admits 8 or 4 cores"):
        convert.build_expert_moe(lambda name: None, max_seq=16, rope_theta=1e6,
                                 runtime_config=SimpleNamespace(), num_cores=cores)


def test_reduced_persistent_spm_reserves_all_three_owners_cold(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3vl_vision_prepare_persistent_spm",
        lambda handle, length: calls.append(("vision", handle, length)),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "causal_decoder_prepare_persistent_spm",
        lambda handle, length, pos, causal, mask_kv=0: calls.append(
            ("text", handle, length, pos, causal, mask_kv)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "lingbot_v2_moe_prepare_resident_spm",
        lambda handle, length, pos, causal, mask_kv: calls.append(
            ("action", handle, length, pos, causal, mask_kv)
        ),
        raising=False,
    )
    runtime._prepare_lingbot2_reduced_persistent_spm(11, 22, 33)
    assert calls == [
        ("vision", 11, 256),
        ("text", 22, 256, 0, True, 0),
        ("action", 33, runtime.N_ACTION + 1, 256, False, 256 + runtime.N_ACTION + 1),
    ]


def test_reduced_persistent_spm_rejects_partial_native_api(monkeypatch):
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3vl_vision_prepare_persistent_spm",
        lambda handle, length: None,
        raising=False,
    )
    monkeypatch.delattr(torch.ops.rpu, "causal_decoder_prepare_persistent_spm", raising=False)
    monkeypatch.delattr(torch.ops.rpu, "lingbot_v2_moe_prepare_resident_spm", raising=False)
    with pytest.raises(RuntimeError, match="Persistent preparation API is incomplete"):
        runtime._prepare_lingbot2_reduced_persistent_spm(11, 22, 33)
