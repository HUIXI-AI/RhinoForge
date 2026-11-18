"""HyVLA at a cold 4/6/8-core budget: host contracts only.

The example TOMLs, the composite resolver, the cold-plan refusals, the weight
lane layouts and the native handle order are checked here with small doubles.
None of this is a board run or numerical evidence for the reduced-core
kernels.
"""
import os
from pathlib import Path

import pytest
import torch

from examples.common import config as common
from examples.runners.hy_vla import optimized as runner
from rpu_backend.adapters.hy_vla import nvfp4, runtime, weights
from rpu_backend.adapters.hy_vla.runtime import (
    HY_VLA_ACTION_COMPONENT,
    HY_VLA_LANGUAGE_COMPONENT,
    HY_VLA_VISION_COMPONENT,
    HyVlaRunner,
    _HyVlaNativeColdSnapshot,
    prepare_hy_vla_cold_plan,
    resolve_hy_vla_execution,
)
from rpu_backend.adapters.hy_vla.weights import HyVlaConfig


@pytest.fixture
def cold_owner(monkeypatch):
    """Real preparation/planner/forward glue; only device operations are CPU doubles."""
    from contextlib import contextmanager
    from types import SimpleNamespace
    from rpu_backend.api._execution import bind_execution_session
    from rpu_backend.runtime.execution_planner import PreparedExecutionPlans

    events = []
    scopes = {"vit": (HY_VLA_VISION_COMPONENT, "vision", "hyvit2", 592),
              "vlm": (HY_VLA_LANGUAGE_COMPONENT, "prefill", "hyvla_vlm", 240),
              "expert": (HY_VLA_ACTION_COMPONENT, "action", "hyvla_expert", 64)}
    descriptors = {
        "vit": (3, 61, 196, 592, 592, 0, 0, 0, 1, 0, 0, 3, 1, 1, 3, 0, 0, 196, 196, 1, 196, 196, 392, 2, 392, 196, 588, 0, 0, 588, 588, 0, 0, 588, 588, 0, 196, 0, 196, 196, 1, 392, 196, 2, 1, 588, 588, 0, 588, 588, 4, 1, 1, 2466643912, 1351873722, 1, 1142732678571512451, 2, 0, 0, 0),
        "vlm": (3, 47, 240, 240, 240, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 240, 240, 0, 0, 240, 240, 0, 0, 240, 240, 0, 240, -1, 1, 240, 240, 0, 240, 240, 4, 1, 1, 2094591820, 710271203, 1, 4354974985578103569, 1, 0, 0, 0),
        "expert": (3, 47, 64, 64, 64, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 51, 291, 0, 0, 51, 291, 0, 0, 51, 291, 0, 51, -1, 1, 51, 51, 0, 291, 51, 4, 1, 1, 1380704381, 3355739691, 1, 8372732983358876378, 1, 0, 0, 0),
    }

    class Graph:
        def __init__(self):
            self.entries = {}
            self.plans = PreparedExecutionPlans()

        def is_frozen(self):
            return False

        def prepare_plan(self, key, factory):
            return self.plans.prepare(key, factory)

        @contextmanager
        def capture(self, signature):
            self.entries[signature] = self.entries.get(signature, 0) + 1
            yield

        def snapshot(self):
            return tuple(self.entries.items())

        def clear(self):
            self.entries.clear()

    monkeypatch.setattr(runtime.rpu_backend.graph, "GraphCache", Graph)
    monkeypatch.setattr(runtime.rpu_backend.graph, "GraphSignature",
        lambda **fields: tuple((key, tuple(value) if isinstance(value, list) else value)
                               for key, value in sorted(fields.items())))
    monkeypatch.setattr(runtime, "_rpu16", lambda value: value.half().contiguous())
    monkeypatch.setattr(weights, "_rpu16", runtime._rpu16)
    monkeypatch.setenv("RPU_HY_VLA_ATTN_TP8", "1")

    handles = {}
    def operation(name):
        def call(*args):
            events.append((name, args))
            if name.endswith("_create"):
                handle = 100 + len(handles)
                handles[handle] = name.removesuffix("_create")
                return handle
            if name.endswith("_resolve_stage_domain"):
                child = next(child for child, scope in scopes.items() if scope[2] == handles[args[0]])
                words = descriptors[child]
                return (1, 1, len(words), *words)
            if name.endswith("_get_resolved_chunk_size"):
                return next(scope[3] for scope in scopes.values() if scope[2] == handles[args[0]])
            if name.endswith("_planner_cache_identity"):
                return (args[0], 1)
            if name == "hyvit2_forward_multi":
                return torch.ones(1, 588, 1152, dtype=torch.float16)
            if name == "hyvla_merger_fused":
                return torch.ones(147, 2048, dtype=torch.float16)
            if name == "hyvla_expert_unroll_forward":
                if args[-1] is not None:
                    args[1][:, 1:].copy_(args[-1])
                args[8].copy_(args[1].expand_as(args[8]))
        return call

    native_names = ["spm_alloc_reset_all", "spm_alloc_reset_temporary", "hyvla_merger_fused",
                    "hyvit2_forward_multi", "hyvla_vlm_step_forward", "hyvla_expert_unroll_forward",
                    "hyvit2_set_chunk_envelope", "hyvit2_model_set_patch_emb",
                    "hyvla_vlm_set_moe_weights", "hyvla_expert_set_action_weights"]
    for _component, _stage, prefix, _chunk in scopes.values():
        native_names.extend(f"{prefix}_{suffix}" for suffix in (
            "create", "destroy", "set_runtime_config", "set_weights", "set_chunk_size_override",
            "enable_execution_reconfigure", "resolve_stage_domain", "get_resolved_chunk_size",
            "planner_cache_identity"))
    for name in native_names:
        monkeypatch.setattr(torch.ops.rpu, name, operation(name), raising=False)

    owner = HyVlaRunner.__new__(HyVlaRunner)
    tensor = torch.ones(1, dtype=torch.float16)
    linear = {key: [tensor] for key in ("q_w", "k_w", "v_w", "o_w", "q_norm", "k_norm",
              "input_norm", "post_norm", "gate_w", "up_w", "down_w")}
    linear["q_ws"] = []
    vit = {key: [tensor] for key in ("qw", "kw", "vw", "ow", "f1w", "f2w", "n1w", "n1b",
              "n2w", "n2b", "qb", "kb", "vb", "ob", "f1b", "f2b")}
    vit["qws"] = []
    owner.w = SimpleNamespace(cfg=HyVlaConfig(), vit=vit, vlm_text=linear,
        vlm_vision=linear.copy(), expert=linear.copy(), vit_proj1_w=tensor, vit_proj1_b=tensor,
        vit_patch_gemm_w=tensor, vit_pos_fused=tensor, patch_embed_cores=8, action_mlp_cores=8,
        attn_tp=8, kv_heads_rpu=8,
        vlm_final_norm=tensor, expert_final_norm=tensor,
        merger_rest={key: tensor for key in ("pooler.predictor.0.weight_a", "pooler.predictor.0.bias",
            "pooler.predictor.0.weight_b", "pooler.predictor.2.weight", "pooler.predictor.2.bias",
            "proj2.weight", "proj2.bias")})
    owner._closed = owner._retirement_failed = False
    owner.prefix_len, owner._total = 240, 304
    owner._native_cold = _HyVlaNativeColdSnapshot.from_env()
    owner._persist = frozenset(scopes)
    owner._psubs, owner._const_key, owner._const_val = {}, {}, {}
    owner._rpu_execution, owner._rpu_execution_components = resolve_hy_vla_execution(None, entry_point="test")
    owner._rpu_execution_component_generations = dict.fromkeys(owner._rpu_execution_components, 0)
    owner._rpu_last_execution_receipts = {}
    owner._vit_packed = owner._pe_in_graph = owner._unroll = True
    owner._merger_in_graph = owner._fused_merger = owner._proj1_in_merger = owner._prefix_template = True
    owner._ux0 = torch.zeros(1, 51, 32, dtype=torch.float16)
    owner._utraj = torch.zeros(10, 51, 32, dtype=torch.float16)
    owner._uw = (tensor,) * 6
    def cache(capacity, layers):
        return SimpleNamespace(max_seq_len=capacity, k_caches=[tensor.clone() for _ in range(layers)],
            v_caches=[tensor.clone() for _ in range(layers)],
            reset_to_position=lambda position: events.append(("kv_reset", (capacity, position))))
    owner._vit_cache, owner._cache = cache(588, 27), cache(304, 32)
    owner.assemble_prefix = lambda _embeddings, _tokens: torch.ones(1, 240, 2048, dtype=torch.float16)
    owner._execution_session = bind_execution_session(owner, {}, entry_point="HY cold CPU bridge",
        supported_components=runtime.HY_VLA_EXECUTION_COMPONENTS, graph_mode="COMPOSITE_CHILD")
    for component in owner._rpu_execution_components:
        owner._execution_session._bind_planner_owner(owner, component)
    inputs = dict(images=[torch.ones(1, 3, 224, 224) for _ in range(3)],
        prefill_pos=torch.arange(240), prefill_mask=torch.ones(240, 240, dtype=torch.bool).tril(),
        modality_mask=torch.arange(240) < 147, suffix_pos=torch.arange(181, 232),
        denoise_mask=torch.ones(51, 291, dtype=torch.bool))
    yield owner, inputs, events, scopes, descriptors
    owner.close()


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples/configs/hy_vla").glob("*.toml"))


@pytest.fixture(autouse=True)
def _no_hyvla_selectors(monkeypatch):
    # Other HyVLA tests apply profile selectors to the process environment.
    for name in [name for name in os.environ
                 if name.startswith("RPU_HY_VLA_") or name == "RPU_RMSNORM_VWARP"]:
        monkeypatch.delenv(name)


def _bind_budget(owner, budget):
    owner._rpu_execution, owner._rpu_execution_components = resolve_hy_vla_execution(
        {} if budget == 8 else {"model": {"num_cores": budget}}, entry_point="test")
    owner._core_budget = budget


def _with_budget(config, cores):
    config.setdefault("rpu_execution", {})["model"] = {"num_cores": cores}
    return config


def _cores(child):
    return child.get("model", {}).get("num_cores", 8)


# --------------------------------------------------------------------------- examples


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.name)
def test_every_template_admits_each_budget(path):
    assert common.core_support(common.load_config(path)) == {4: None, 6: None, 8: None}


@pytest.mark.parametrize("cores", (4, 6))
def test_reduced_template_refuses_eight_core_options(cores):
    config = _with_budget(common.load_config(ROOT / "examples/configs/hy_vla/fp16.toml"), cores)
    config["hy_vla"]["rmsnorm_vwarp"] = "16"
    with pytest.raises(ValueError, match='rmsnorm_vwarp = "16" is eight-core only'):
        runner.validate_example_config(config)
    config = _with_budget(common.load_config(
        ROOT / "examples/configs/hy_vla/w8a16_action_nvfp4.toml"), cores)
    config["hy_vla"]["resident_fp8_scale_layers"] = 4
    with pytest.raises(ValueError, match="resident_fp8_scale_layers is eight-core only"):
        runner.validate_example_config(config)
    config = _with_budget(common.load_config(ROOT / "examples/configs/hy_vla/fp16.toml"), 5)
    with pytest.raises(ValueError, match="4, 6 or 8"):
        runner.validate_example_config(config)


@pytest.mark.parametrize("cores", (4, 6, 8))
@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.name)
def test_example_environment_is_admitted_by_the_cold_plan(path, cores):
    config = common.load_config(path)
    if cores != 8:
        _with_budget(config, cores)
    args = runner.example_arguments(config)
    assert _cores(args.rpu_execution) == cores
    assert ("model" in args.rpu_execution) == (cores != 8)
    env = runner.profile_environment(args)
    plan = prepare_hy_vla_cold_plan(env, num_cores=cores)
    assert plan.num_cores == plan.weights.num_cores == cores
    if cores == 8:
        # Packed W4 has no SPM-resident Gate/Up layout; the others keep it.
        resident = args.precision not in ("w4a16", "w4a16_vision")
        assert env["RPU_HY_VLA_ATTN_TP8"] == "1" and plan.weights.attn_tp8
        assert env["RPU_HY_VLA_EXPERT_GATEUP_RESIDENT"] == str(int(resident))
        assert plan.weights.gateup_resident is resident
        return
    assert {env[name] for name in ("RPU_HY_VLA_ATTN_TP8", "RPU_HY_VLA_RMSNORM_PAD16",
            "RPU_HY_VLA_VISION_KV_SPM_ONLY", "RPU_HY_VLA_EXPERT_GATEUP_RESIDENT",
            "RPU_RMSNORM_VWARP")} == {"0"}
    assert env["RPU_HY_VLA_PREFILL_KV_ONLY"] == "1"
    weights_plan = plan.weights
    assert not weights_plan.attn_tp8 and not weights_plan.gateup_resident
    assert weights_plan.action_mlp_cores == weights_plan.patch_embed_cores == 4
    assert plan.native.vlm.rmsnorm_capability == runtime._RMSNORM_CAP_BASE
    assert not plan.native.expert.rmsnorm_pad16 and not plan.native.vit.kv_spm_only


def test_umi_profile_admits_each_budget_with_its_default_environment():
    from rpu_backend.api.hy_embodied import HyEmbodiedPolicy

    config = common.load_config(ROOT / "examples/configs/hy_embodied/umi/fp16.toml")
    assert common.core_support(config) == {4: None, 6: None, 8: None}
    for cores in (4, 6):
        policy = HyEmbodiedPolicy.from_checkpoint(
            "/missing/ckpt", rpu_execution={"model": {"num_cores": cores}})
        assert runtime._hy_vla_core_budget(policy._rpu_execution) == cores
        # The same cold overrides HyEmbodiedPolicy.to("rpu") passes for FP16.
        plan = prepare_hy_vla_cold_plan(
            {"RPU_HY_VLA_W8A16": "0", "RPU_HY_VLA_W4A16": "0"}, num_cores=cores)
        assert not plan.weights.attn_tp8 and plan.weights.action_mlp_cores == 4
        assert plan.native.vlm.rmsnorm_capability == runtime._RMSNORM_CAP_BASE


# --------------------------------------------------------------------------- resolver


@pytest.mark.parametrize("budget", (4, 6, 8))
def test_resolver_derives_each_child_budget(budget):
    request = {} if budget == 8 else {"model": {"num_cores": budget}}
    root, children = resolve_hy_vla_execution(request, entry_point="test")
    assert runtime._hy_vla_core_budget(root) == budget
    if budget == 8:
        assert all("model" not in child for child in children.values())
        return
    assert _cores(children[HY_VLA_LANGUAGE_COMPONENT]) == budget
    assert _cores(children[HY_VLA_VISION_COMPONENT]) == _cores(children[HY_VLA_ACTION_COMPONENT]) == 4
    assert children[HY_VLA_LANGUAGE_COMPONENT]["prefill"]["chunk_size"] == "auto"


@pytest.mark.parametrize(("request_", "match"), [
    ({"model": {"num_cores": 5}}, "4, 6 or 8"),
    ({"components": {HY_VLA_LANGUAGE_COMPONENT: {"model": {"num_cores": 4}}}}, "model"),
])
def test_resolver_refuses_other_and_child_budgets(request_, match):
    with pytest.raises(ValueError, match=match):
        resolve_hy_vla_execution(request_, entry_point="test")


def test_reconfigure_keeps_the_cold_budget():
    owner = HyVlaRunner.__new__(HyVlaRunner)
    owner._closed = False
    _bind_budget(owner, 6)
    for change in ({}, {"model": {"num_cores": 4}}):
        with pytest.raises(ValueError, match="cold-only"):
            owner.validate_execution_reconfigure(change)


# --------------------------------------------------------------------------- cold plan


@pytest.mark.parametrize(("env", "match"), [
    ({"RPU_HY_VLA_ATTN_TP8": "1"}, "RPU_HY_VLA_ATTN_TP8"),
    ({"RPU_RMSNORM_VWARP": "auto"}, "RPU_RMSNORM_VWARP"),
    ({"RPU_HY_VLA_RMSNORM_PAD16": "expert"}, "RPU_HY_VLA_RMSNORM_PAD16"),
    ({"RPU_HY_VLA_VISION_KV_SPM_ONLY": "1"}, "RPU_HY_VLA_VISION_KV_SPM_ONLY"),
    ({"RPU_HY_VLA_VIT_PACKED": "0"}, "RPU_HY_VLA_VIT_PACKED=0"),
    ({"RPU_HY_VLA_PROJ1_IN_MERGER": "0"}, "RPU_HY_VLA_PROJ1_IN_MERGER=0"),
    ({"RPU_HY_VLA_EXPERT_GATEUP_RESIDENT": "1"}, "RPU_HY_VLA_EXPERT_GATEUP_RESIDENT"),
    ({"RPU_HY_VLA_EXPERT_O_RESIDENT_LAYERS": "12"}, "RPU_HY_VLA_EXPERT_O_RESIDENT_LAYERS"),
])
@pytest.mark.parametrize("cores", (4, 6))
def test_cold_plan_refuses_eight_core_routes_before_loading(env, match, cores):
    with pytest.raises(ValueError, match=f"model.num_cores={cores} has no route for {match}"):
        prepare_hy_vla_cold_plan(env, num_cores=cores)


def test_reduced_nvfp4_needs_the_four_core_action_mlp():
    env = {"RPU_HY_VLA_EXPERT_NVFP4": "1", "RPU_HY_VLA_W8A16": "vit,vlm"}
    plan = prepare_hy_vla_cold_plan(env, num_cores=6)
    assert plan.weights.expert_nvfp4 and plan.weights.action_mlp_cores == 4
    with pytest.raises(ValueError, match="requires the four-core action MLP"):
        prepare_hy_vla_cold_plan(env | {"RPU_HY_VLA_ACTION_MLP_MC": "0"}, num_cores=6)
    # Eight cores keep their exact NVFP4 contract unchanged.
    with pytest.raises(ValueError, match="requires attention TP8, action MLP TP8"):
        prepare_hy_vla_cold_plan(env | {"RPU_HY_VLA_ATTN_TP8": "0"})


def test_build_refuses_a_cold_plan_for_another_budget():
    plan = prepare_hy_vla_cold_plan()
    with pytest.raises(ValueError, match="cold plan was prepared for model.num_cores=8"):
        runtime.build_hy_vla("/missing/ckpt", rpu_execution={"model": {"num_cores": 6}},
                             _cold_plan=plan)
    with pytest.raises(ValueError, match="one of 4, 6 or 8"):
        prepare_hy_vla_cold_plan(num_cores=True)


# --------------------------------------------------------------------------- weights


class _Source:
    def __getitem__(self, key):
        return torch.zeros(2, 2)


def _record_quant(monkeypatch):
    calls = []
    monkeypatch.setattr(weights, "_quant",
                        lambda raw, partition, cores, bits: (calls.append((partition, cores)), (raw, None))[1])
    monkeypatch.setattr(weights, "_rpu16", lambda value: value)
    monkeypatch.setattr(weights, "_dup_kv", lambda raw, head_dim, *, attn_tp8: (
        pytest.fail("KV duplicated below eight cores") if attn_tp8 else raw))
    return calls


@pytest.mark.parametrize("cores", (4, 6))
def test_weight_lanes_follow_each_child(monkeypatch, cores):
    cfg = weights.HyVlaConfig(layers=1, vit_layers=1)
    cold = weights._HyVlaWeightColdPlan.from_environment({}, num_cores=cores)
    calls = _record_quant(monkeypatch)
    weights._build_mot_branch(_Source(), cfg, "_v", cold)
    # q/k/v/o on the native KV4 attention lanes; Gate/Up/Down on the budget.
    assert calls == [(1, 4), (1, 4), (1, 4), (0, 4), (1, cores), (1, cores), (0, cores)]
    calls.clear()
    weights._build_expert(_Source(), _Source(), cfg, cold)
    assert calls == [(1, 4), (1, 4), (1, 4), (0, 4), (1, 4), (1, 4), (0, 4)]


def test_vision_lanes_are_four_below_eight_cores(monkeypatch):
    cfg = weights.HyVlaConfig(vit_layers=1)
    calls = []
    monkeypatch.setattr(weights, "_quant",
                        lambda raw, partition, cores, bits: (calls.append((partition, cores)), (raw, None))[1])
    monkeypatch.setattr(weights, "_rpu16", lambda value: value)

    class Vit:
        def __getitem__(self, key):
            shape = {"attn.qkv.weight": (3 * 16 * 72, 1152), "attn.qkv.bias": (3 * 16 * 72,),
                     "attn.proj.weight": (1152, 16 * 72), "mlp.fc1.weight": (4304, 1152),
                     "mlp.fc1.bias": (4304,), "mlp.fc2.weight": (1152, 4304)}.get(
                         key.split(".", 2)[2], (1152,))
            return torch.zeros(shape)

    for cores, lanes in ((8, 8), (6, 4), (4, 4)):
        calls.clear()
        weights._build_vit(Vit(), cfg, weights._HyVlaWeightColdPlan.from_environment(
            {"RPU_HY_VLA_ATTN_TP8": "1"} if cores == 8 else {}, num_cores=cores))
        assert {lane for _partition, lane in calls} == {lanes}


def test_four_lane_nvfp4_keeps_kv4_and_refuses_other_lanes():
    raw = ((torch.arange(512 * 1024, dtype=torch.float32) % 83 - 41) / 37).reshape(512, 1024)
    packed, striped, _scale = nvfp4.quantize_expert_nvfp4_cpu(raw, 1, 4)
    assert packed.shape == (512, 512) and striped.shape == (64, 512)
    with pytest.raises(ValueError, match="TP8 or four-core projection"):
        nvfp4.quantize_expert_nvfp4_cpu(torch.zeros(1024, 1024), 1, 4)
    with pytest.raises(ValueError, match="TP8 or four-core projection"):
        nvfp4.quantize_expert_nvfp4_cpu(torch.zeros(2048, 1024), 1, 6)


# --------------------------------------------------------------------------- native order


def _reduced_native_owner(cold_owner, monkeypatch, budget):  # noqa: F811
    owner, inputs, events, _scopes, _descriptors = cold_owner
    _bind_budget(owner, budget)
    owner.w.attn_tp = owner.w.kv_heads_rpu = 4
    policies = {}
    graph = runtime.rpu_backend.graph.GraphCache

    class Graph(graph):
        def __init__(self, runtime_policy=None):
            super().__init__()
            policies[len(policies)] = runtime_policy

    monkeypatch.setattr(runtime.rpu_backend.graph, "GraphCache", Graph)
    topology = {"hyvit2": (1, 4, 4, 4, 1, 8, 4304, 4352),
                "hyvla_vlm": (budget, 4, budget, 4, 8),
                "hyvla_expert": (4, 4, 4, 4, 8)}
    for prefix, expected in topology.items():
        monkeypatch.setattr(torch.ops.rpu, f"{prefix}_set_execution_core_count",
                            lambda handle, cores, prefix=prefix: events.append(
                                (f"{prefix}_set_execution_core_count", (handle, cores))),
                            raising=False)
        monkeypatch.setattr(torch.ops.rpu, f"{prefix}_get_execution_topology",
                            lambda handle, expected=expected, prefix=prefix: (events.append(
                                (f"{prefix}_get_execution_topology", (handle,))), list(expected))[1],
                            raising=False)
    monkeypatch.setattr(owner, "_plan_execution", lambda **request: request)
    return owner, inputs, events, policies


@pytest.mark.parametrize("budget", (4, 6))
def test_children_bind_core_count_first_and_read_topology_back(cold_owner, monkeypatch, budget):  # noqa: F811
    owner, inputs, events, policies = _reduced_native_owner(cold_owner, monkeypatch, budget)
    owner.prepare_execution(**inputs)
    names = [name for name, _args in events]
    for prefix, cores in (("hyvit2", 4), ("hyvla_vlm", budget), ("hyvla_expert", 4)):
        first = names.index(f"{prefix}_set_execution_core_count")
        assert events[first][1][1] == cores
        assert first < names.index(f"{prefix}_set_runtime_config")
        setter = "hyvla_vlm_set_moe_weights" if prefix == "hyvla_vlm" else f"{prefix}_set_weights"
        assert names.index(setter) < names.index(f"{prefix}_get_execution_topology")
    lanes = sorted(policy.execution_core_count for policy in policies.values())
    assert lanes == sorted([4, budget, 4])


def test_native_topology_that_differs_from_the_layout_fails_closed(cold_owner, monkeypatch):  # noqa: F811
    owner, inputs, _events, _policies = _reduced_native_owner(cold_owner, monkeypatch, 6)
    monkeypatch.setattr(torch.ops.rpu, "hyvla_expert_get_execution_topology",
                        lambda handle: [8, 8, 8, 8, 8], raising=False)
    with pytest.raises(RuntimeError, match="action expert native topology differs"):
        owner.prepare_execution(**inputs)
    # Like any failed native bind, the mismatch poisons the owner: restart required.
    assert owner._retirement_failed
    with pytest.raises(RuntimeError, match="restart the process"):
        owner.close()
    monkeypatch.setattr(owner, "close", lambda: None)  # the fixture's teardown close


@pytest.mark.parametrize("budget", (4, 6, 8))
def test_merger_passes_its_lane_count_only_below_eight(cold_owner, monkeypatch, budget):  # noqa: F811
    owner, _inputs, _events, _scopes, _descriptors = cold_owner
    _bind_budget(owner, budget)
    seen = []
    monkeypatch.setattr(torch.ops.rpu, "hyvla_merger_fused",
                        lambda *args, **kwargs: (seen.append(kwargs), torch.ones(147, 2048))[1],
                        raising=False)
    owner._merger_rest(torch.ones(588, 1152, dtype=torch.float16))
    assert seen == [{} if budget == 8 else {"num_cores": 4}]
