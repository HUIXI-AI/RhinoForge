"""Wall-OSS base, RTC v728 and parcel sorting at a cold 4/6/8-core budget: host contracts only.

The example TOMLs, the composite resolver, the builders' refusals, the native
handle order and the execution-topology rows are checked here with small
doubles. None of this is a board run or numerical evidence for the
reduced-core kernels.
"""
from pathlib import Path
import os
import shutil
import subprocess

import pytest
import torch

from examples.common import config as common
from examples.runners.wall_oss import base as runner
from rpu_backend.adapters.wall_oss import action as wall_action
from rpu_backend.adapters.wall_oss import llm as wall_llm
from rpu_backend.adapters.wall_oss import runtime as wall
from rpu_backend.adapters.wall_oss.vision import build_wall_oss_vision


ROOT = Path(__file__).resolve().parents[1]
BASE = sorted((ROOT / "examples/configs/wall_oss/base").rglob("*.toml"))
EIGHT_ONLY = sorted(
    [*(ROOT / "examples/configs/wall_oss/legacy").rglob("*.toml"),
     ROOT / "examples/configs/wall_oss/sorting/w8a16_action_w4a16_g32.toml"])
REDUCED_RTC = sorted(
    [*(ROOT / "examples/configs/wall_oss/rtc").rglob("*.toml"),
     ROOT / "examples/configs/wall_oss/sorting/fp16.toml",
     ROOT / "examples/configs/wall_oss/sorting/w8a16.toml"])
EIGHT_ONLY_ENV = (
    "RPU_WALL_OSS_ATTN_TP8", "RPU_WALL_OSS_BATCH_VISION",
    "RPU_WALL_OSS_VISION_ROW_WEIGHT_OUTER",
    "RPU_WALL_OSS_FUSED_ASSEMBLE", "RPU_WALL_OSS_FP16_TRANSFER_MASK",
    "RPU_WALL_OSS_SORTING_SILU_MUL",
)


@pytest.fixture(autouse=True)
def _no_wall_selectors(monkeypatch):
    # Other Wall tests apply profile selectors to the process environment.
    for name in [name for name in os.environ if name.startswith("RPU_WALL_OSS_")]:
        monkeypatch.delenv(name)


def _cores(child):
    return child.get("model", {}).get("num_cores", 8)


def _with_budget(config, cores):
    config.setdefault("rpu_execution", {})["model"] = {"num_cores": cores}
    return config


# --------------------------------------------------------------------------- examples


@pytest.mark.parametrize("path", BASE, ids=lambda path: str(path.relative_to(ROOT)))
def test_every_base_example_admits_each_budget(path):
    assert common.core_support(common.load_config(path)) == {4: None, 6: None, 8: None}


@pytest.mark.parametrize("path", REDUCED_RTC, ids=lambda path: str(path.relative_to(ROOT)))
def test_rtc_and_sorting_examples_admit_each_budget(path):
    assert common.core_support(common.load_config(path)) == {4: None, 6: None, 8: None}


@pytest.mark.parametrize("path", EIGHT_ONLY, ids=lambda path: str(path.relative_to(ROOT)))
def test_legacy_and_action_w4_sorting_stay_eight_core_with_a_reason(path):
    support = common.core_support(common.load_config(path))
    assert support[8] is None
    assert support[4] == support[6]
    assert support[4].startswith("旧版档案" if "legacy" in path.parts else "原生实现只有 8 核")


@pytest.mark.parametrize("cores", (4, 6))
def test_reduced_base_refuses_eight_core_example_options(cores):
    config = common.load_config(ROOT / "examples/configs/wall_oss/base/3cam/256/fp16.toml")
    residency = _with_budget(config, cores)
    residency["wall_oss"]["instruction_residency"] = "auto"
    with pytest.raises(ValueError, match="instruction_residency is eight-core only"):
        runner.validate_example_config(residency)
    config = _with_budget(common.load_config(
        ROOT / "examples/configs/wall_oss/base/3cam/256/fp16.toml"), cores)
    config["rpu_execution"]["vision"]["linear_acc32"] = True
    with pytest.raises(ValueError, match="linear_acc32 is eight-core only.*vision"):
        runner.validate_example_config(config)


@pytest.mark.parametrize("cores", (4, 6, 8))
def test_argv_drops_pair_and_residency_below_eight_cores(cores):
    pair = common.load_config(ROOT / "examples/configs/wall_oss/base/3cam/256/w8a16.toml")
    window = common.load_config(ROOT / "examples/configs/wall_oss/base/2cam/224/fp16.toml")
    if cores != 8:
        _with_budget(pair, cores)
        _with_budget(window, cores)
    pair_argv, window_argv = runner.example_argv(pair), runner.example_argv(window)
    assert ("--vision-pair" in pair_argv) == ("--action-residency" in pair_argv) == (cores == 8)
    # 224 keeps its window request: the four-lane owner serves it per window.
    assert "--vision-window" in window_argv


def _environment(monkeypatch, tmp_path, path, cores):
    engine = runner.validation_runner()
    config = common.load_config(path)
    if cores != 8:
        _with_budget(config, cores)
    template = str(tmp_path / "{cameras}cam{image_size}_t{text_tokens}.pt")
    config["input"].update(fp16_checkpoint=str(tmp_path), batch_template=template)
    Path(runner.batch_path(config["input"])).touch()
    config["run"]["output_dir"] = str(tmp_path / "out")
    args = engine.arguments(runner.example_argv(config))
    args.rpu_execution = runner.execution_config(config.get("rpu_execution"))
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setitem(engine._LIBRARIES, "current_libraries", lambda root: (
        {"RPU_KERNEL_LIB_PATH": "main", "RPU_KERNEL_LIB_PATH_EXPANSION": "expansion"}, None))
    monkeypatch.setitem(engine._LIBRARIES, "verify_libraries", lambda root, selected: ({}, None))
    return engine.cold_environment(args, root=tmp_path)


@pytest.mark.parametrize("cores", (4, 6, 8))
def test_cold_environment_turns_eight_core_selectors_off(monkeypatch, tmp_path, cores):
    env = _environment(monkeypatch, tmp_path,
                       ROOT / "examples/configs/wall_oss/base/3cam/256/fp16.toml", cores)
    if cores == 8:
        assert env["RPU_WALL_OSS_ATTN_TP8"] == env["RPU_WALL_OSS_VISION_PAIR_252"] == "1"
        assert env["RPU_WALL_OSS_FUSED_ASSEMBLE"] == env["RPU_WALL_OSS_SORTING_SILU_MUL"] == "1"
        return
    assert {env[name] for name in EIGHT_ONLY_ENV} == {"0"}
    # The four-lane Vision owner keeps SPM-resident RoPE tables + fused merger.
    assert env["RPU_WALL_OSS_VISION_ROPE_SPM"] == env["RPU_WALL_OSS_VISION_FUSED_MERGER"] == "1"
    assert "RPU_WALL_OSS_VISION_PAIR_252" not in env
    assert env["RPU_WALL_OSS_ACTION_WEIGHT_RESIDENCY"] == "0"
    assert env["RPU_WALL_OSS_FUSED_DENOISE"] == "1"


def test_reduced_224_window_selects_per_window_sdpa(monkeypatch, tmp_path):
    env = _environment(monkeypatch, tmp_path,
                       ROOT / "examples/configs/wall_oss/base/2cam/224/fp16.toml", 6)
    assert env["RPU_WALL_OSS_VISION_NATIVE_WINDOW_SDPA"] == "1"
    assert env["RPU_WALL_OSS_BATCH_VISION"] == "0"


# --------------------------------------------------------------------------- resolver


@pytest.mark.parametrize("budget", (4, 6, 8))
def test_resolver_derives_each_child_budget(budget):
    request = {} if budget == 8 else {"model": {"num_cores": budget}}
    root, text, vision, action = wall.resolve_wall_oss_execution_components(
        request, entry_point="test")
    assert wall._wall_core_count(root) == budget
    builder = wall._action_builder_execution(text, action, budget)
    if budget == 8:
        assert all("model" not in child for child in (text, vision, action, builder))
        return
    assert _cores(text) == _cores(builder) == budget
    assert _cores(vision) == _cores(action) == 4


@pytest.mark.parametrize(("request_", "match"), [
    ({"model": {"num_cores": 4}, "vision": {"chunk_size": 768}}, "needs eight cores"),
    ({"model": {"num_cores": 5}}, "4, 6 or 8"),
    ({"components": {"text_backbone": {"model": {"num_cores": 4}}}}, "model"),
])
def test_resolver_refuses_packed_vision_and_child_budgets(request_, match):
    with pytest.raises(ValueError, match=match):
        wall.resolve_wall_oss_execution_components(request_, entry_point="test")


def test_reconfigure_keeps_the_cold_budget():
    owner = wall.WallOssVLA.__new__(wall.WallOssVLA)
    root, text, vision, action = wall.resolve_wall_oss_execution_components(
        {"model": {"num_cores": 6}}, entry_point="test")
    vars(owner).update(_closed=False, _rpu_execution=root, _text_execution=text,
                       _vision_execution=vision, _action_execution=action)
    for change in ({}, {"model": {"num_cores": 4}}):
        with pytest.raises(ValueError, match="cold-only"):
            owner.validate_execution_reconfigure(change)


@pytest.mark.parametrize(("options", "env", "match"), [
    ({"rtc": True, "w4a16": True}, None, "no route for w4a16"),
    ({"rtc": True, "w8a16": True, "action_w4a16_ckpt_dir": "/missing/w4"}, None,
     "no route for action_w4a16_ckpt_dir"),
    ({"nvfp4a16": True}, None, "no route for nvfp4a16"),
    ({}, "RPU_WALL_OSS_ATTN_TP8", "no route for RPU_WALL_OSS_ATTN_TP8"),
    ({}, "RPU_WALL_OSS_VISION_PAIR_252", "no route for RPU_WALL_OSS_VISION_PAIR_252"),
])
def test_composite_refuses_eight_core_routes_before_loading(options, env, match, monkeypatch):
    for name in ("RPU_WALL_OSS_ATTN_TP8", "RPU_WALL_OSS_FUSED_ASSEMBLE",
                 "RPU_WALL_OSS_VISION_ROPE_SPM", "RPU_WALL_OSS_VISION_FUSED_MERGER",
                 "RPU_WALL_OSS_VISION_PAIR_252"):
        monkeypatch.delenv(name, raising=False)
    if env is not None:
        monkeypatch.setenv(env, "1")
    with pytest.raises(ValueError, match=match):
        wall.build_wall_oss_vla("/missing/ckpt", rpu_execution={"model": {"num_cores": 6}},
                                **options)


def test_composite_refuses_a_shared_graph_policy(monkeypatch):
    from rpu_backend.graph import GraphRuntimePolicy
    for name in ("RPU_WALL_OSS_ATTN_TP8", "RPU_WALL_OSS_VISION_PAIR_252"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="graph_runtime_policy is eight-core only"):
        wall.build_wall_oss_vla("/missing/ckpt", rpu_execution={"model": {"num_cores": 4}},
                                graph_runtime_policy=GraphRuntimePolicy.from_environment())


# --------------------------------------------------------------------------- builders


def test_attention_lanes_follow_the_budget(monkeypatch):
    monkeypatch.delenv("RPU_WALL_OSS_ATTN_TP8", raising=False)
    assert wall_llm._eff_attn_tp(2) == 2
    assert wall_llm._eff_attn_tp(2, 4) == wall_llm._eff_attn_tp(2, 6) == 4
    monkeypatch.setenv("RPU_WALL_OSS_ATTN_TP8", "1")
    assert wall_llm._eff_attn_tp(2) == 8
    with pytest.raises(ValueError, match="eight-core layout"):
        wall_llm._eff_attn_tp(2, 6)


def test_text_builder_refuses_quantized_and_mixed_routes(monkeypatch):
    monkeypatch.delenv("RPU_WALL_OSS_PREFILL_MIXED_PRECISION", raising=False)
    with pytest.raises(ValueError, match="6-core Wall-OSS text.*refused: w4a16"):
        wall_llm.build_wall_oss_llm("/missing/ckpt", expert=0, w4a16=True,
                                    execution_config={"model": {"num_cores": 6}})


@pytest.mark.parametrize(("options", "env", "refused"), [
    ({"fused": False}, None, "fused=False"),
    ({"fused": True, "w4a16": True}, None, "w4a16"),
    ({"fused": True}, "RPU_WALL_OSS_ACTION_WEIGHT_RESIDENCY", "RPU_WALL_OSS_ACTION_WEIGHT_RESIDENCY"),
])
def test_action_builder_admits_only_the_fused_ordinary_expert(options, env, refused, monkeypatch):
    for name in ("RPU_WALL_OSS_ACTION_WEIGHT_RESIDENCY", "RPU_WALL_OSS_ACTION_KV_OWNER_FANOUT"):
        monkeypatch.delenv(name, raising=False)
    if env is not None:
        monkeypatch.setenv(env, "1")
    with pytest.raises(ValueError, match=f"4-core Wall-OSS action.*refused: {refused}"):
        wall_action.build_wall_oss_action("/missing/ckpt", execution_config={
            "model": {"num_cores": 4}}, **options)


def test_vision_builder_admits_w8_on_four_lanes_only(monkeypatch):
    monkeypatch.setenv("RPU_WALL_OSS_VISION_MIXED_PRECISION", "1")
    with pytest.raises(ValueError, match="refused: RPU_WALL_OSS_VISION_MIXED_PRECISION$"):
        build_wall_oss_vision("/missing/ckpt", w8a16=True, num_cores=4)
    with pytest.raises(ValueError, match="must be 4 or 8"):
        build_wall_oss_vision("/missing/ckpt", num_cores=6)


# --------------------------------------------------------------------------- native handles


def _ops(monkeypatch, prefix, topology, names):
    calls = []

    def op(name, result=None):
        def call(*args, **kwargs):
            calls.append((name, kwargs) if kwargs else name)
            return result
        monkeypatch.setattr(torch.ops.rpu, f"{prefix}_{name}", call, raising=False)
    op("create", 7)
    op("topology", topology)
    op("destroy")
    for name in names:
        op(name)
    return calls


def test_action_handle_binds_four_lanes_and_checks_the_readback(monkeypatch):
    calls = _ops(monkeypatch, "wall_oss_action_step", [4, 4, 4, 4, 8], ())
    handle = wall_action._configure_action_handle(
        fused=True, set_weights=lambda *args: calls.append("set_weights"),
        weight_args=(), num_cores=4)
    assert handle == 7
    assert calls == [("create", {"num_cores": 4}), "set_weights", "topology"]
    calls.clear()
    monkeypatch.setattr(torch.ops.rpu, "wall_oss_action_step_topology",
                        lambda handle: [8, 8, 8, 8, 8], raising=False)
    with pytest.raises(RuntimeError, match="action topology differs"):
        wall_action._configure_action_handle(
            fused=True, set_weights=lambda *args: None, weight_args=(), num_cores=4)
    assert calls[-1] == "destroy"
    with pytest.raises(ValueError, match="requires the fused owner"):
        wall_action._configure_action_handle(
            fused=False, set_weights=lambda *args: None, weight_args=(), num_cores=4)


def test_text_handle_creates_its_prefix_before_weights(monkeypatch):
    setters = ("set_wall_prefill_routes", "set_chunk_size_cap", "set_chunk_envelope",
               "set_equal_two_prefill", "set_chunk_size_override")
    calls = _ops(monkeypatch, "causal_decoder", [6, 4, 6, 4, 8], setters)
    expected = (6, 4, 6, 4, 8)
    assert wall_llm._configure_causal_decoder_handle(
        set_weights=lambda *args: calls.append("set_weights"), weight_args=(),
        chunk_size_cap=256, equal_two_prefill=False, num_cores=6,
        expected_topology=expected) == 7
    assert calls[:4] == [("create", {"num_cores": 6, "vocab_size": 0}),
                         "set_wall_prefill_routes", "set_weights", "topology"]
    calls.clear()
    with pytest.raises(RuntimeError, match="decoder topology differs"):
        wall_llm._configure_causal_decoder_handle(
            set_weights=lambda *args: None, weight_args=(), chunk_size_cap=256,
            equal_two_prefill=False, num_cores=6, expected_topology=(6, 4, 4, 4, 8))
    assert calls[-1] == "destroy"


def test_native_registrations_expose_the_action_budget():
    registrations = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    assert 'm.def("wall_oss_action_step_create(int num_cores=8) -> int"' in registrations
    assert 'm.def("wall_oss_action_step_topology(int handle) -> int[]"' in registrations
    source = (ROOT / "src/fused/rpu_wall_oss_action_step_model.h").read_text()
    assert "bool reduced_wall_action_owner() const override { return true; }" in source


def test_native_topology_rows_admit_only_the_exact_wall_decoders(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    cpp = r'''
#include "execution_topology.h"
#include <cassert>
using namespace v3;
template <class F> void reject(F f) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    assert(false);
}
bool same(DecoderExecutionTopology t, int cores, int attention, int mlp) {
    return t.num_cores == cores && t.attention_tp == attention && t.mlp_tp == mlp;
}
int main() {
    assert(wall_oss_text_profile_num_layers(16, 4, 128, 2048, 11008) == 36);
    assert(wall_oss_action_profile_num_layers(16, 4, 128, 1024, 2048) == 36);
    assert(decoder_mlp_intermediate_size(11008, 6) == 11136);
    assert(decoder_mlp_intermediate_size(11008, 4) == 11008);
    assert(decoder_mlp_intermediate_size(11008, 8) == 11008);
    for (int cores : {4, 6}) {
        assert(same(resolve_decoder_execution_topology(cores, 16, 4, 128, 2048, 11008),
                    cores, 4, cores));
        // The checkpoint's two KV heads must arrive replicated to four lanes;
        // a padded width is never an admitted logical geometry.
        reject([cores] { resolve_decoder_execution_topology(cores, 16, 2, 128, 2048, 11008); });
        reject([cores] { resolve_decoder_execution_topology(cores, 16, 4, 128, 2048, 11136); });
        reject([cores] { resolve_decoder_execution_topology(cores, 16, 4, 128, 1024, 2560); });
    }
    assert(same(resolve_decoder_execution_topology(4, 16, 4, 128, 1024, 2048), 4, 4, 4));
    // Eight cores keep the original Wall layout (attention 2, MLP 8).
    assert(same(resolve_decoder_execution_topology(8, 16, 2, 128, 2048, 11008), 8, 2, 8));
    reject([] { resolve_decoder_execution_topology(5, 16, 4, 128, 2048, 11008); });
}
'''
    path, binary = tmp_path / "wall_cores.cpp", tmp_path / "wall_cores"
    path.write_text(cpp)
    compiled = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                               "-I", str(ROOT / "src/core"), str(path), "-o", str(binary)],
                              capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr


# --------------------------------------------------------------------------- RTC v728


def test_residual_scale_is_four_below_eight_cores_only():
    assert wall_llm._validate_residual_scale(1, 8, "owner") == 1
    assert wall_llm._validate_residual_scale(4, 6, "owner") == 4
    with pytest.raises(ValueError, match="below eight cores only"):
        wall_llm._validate_residual_scale(4, 8, "owner")
    for value in (2, 8, True, 4.0):
        with pytest.raises(ValueError, match=r"must be one of \(1, 4\)"):
            wall_llm._validate_residual_scale(value, 4, "owner")


def test_residual_fold_divides_fp16_weights_exactly():
    weight = torch.tensor([[1024.0, -3.5], [0.25, 6.0e4]], dtype=torch.float16)
    assert wall_llm._fold_residual(weight, 1) is weight
    folded = wall_llm._fold_residual(weight, 4)
    assert folded.dtype == torch.float16
    assert torch.equal(folded * 4, weight)


def test_w8_fold_keeps_every_dequantized_weight_fp16_normal():
    floor = 2.0 ** -14
    codes = torch.tensor([[127, -64, 1], [127, -64, 3], [5, -5, 0]], dtype=torch.int8)
    # Channel 0 stays normal at 1/4, channel 1 would drop below 2^-14, channel 2 is padding.
    scale = torch.tensor([0.5, 1.0e-4, 0.0], dtype=torch.float16)
    folded, target = wall_llm._fold_w8_residual(codes, scale, 4)
    assert folded.dtype == torch.int8 and target.dtype == torch.float16
    assert torch.equal(folded[0], codes[0]) and float(target[0]) == float(scale[0]) / 4
    assert float(target[1]) == floor
    expected = codes[1].float() * float(scale[1]) / 4
    assert torch.allclose(folded[1].float() * floor, expected, atol=floor / 2)
    assert torch.equal(folded[2], codes[2]) and float(target[2]) == 0.0
    dequantized = (folded.float() * target.float()[:, None]).abs()
    assert bool(((dequantized == 0) | (dequantized >= floor)).all())
    assert wall_llm._fold_w8_residual(codes, scale, 1) == (codes, scale)
    with pytest.raises(ValueError, match="per-channel"):
        wall_llm._fold_w8_residual(codes, scale[:, None], 4)


def test_scaled_residual_stream_keeps_every_norm_output():
    """Folding 1/s into embedding, O and Up (eps/s^2) leaves each RMSNorm output unchanged."""
    torch.manual_seed(0)
    hidden, inter, scale, eps = 32, 48, 4, 1e-6
    embed = torch.randn(5, hidden) * 300
    o, gate, up, down = (torch.randn(*shape) * 0.2 for shape in (
        (hidden, hidden), (inter, hidden), (inter, hidden), (hidden, inter)))
    gate, down = gate.double(), down.double()

    def norm(x, eps):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def block(h, o, up, eps):
        h = h + norm(h, eps) @ o.T
        x = norm(h, eps)
        return h + (torch.nn.functional.silu(x @ gate.T) * (x @ up.T)) @ down.T

    reference = block(embed.double(), o.double(), up.double(), eps)
    fold = [wall_llm._fold_residual(t, scale).double() for t in (embed, o, up)]
    scaled = block(*fold, eps / scale ** 2)
    assert torch.allclose(scaled * scale, reference, rtol=1e-9, atol=1e-9)
    assert torch.allclose(norm(scaled, eps / scale ** 2), norm(reference, eps), atol=1e-9)


def test_text_and_vision_builders_refuse_misplaced_residual_scales():
    with pytest.raises(ValueError, match="below eight cores only"):
        wall_llm.build_wall_oss_llm("/missing/ckpt", expert=0, residual_scale=4)
    with pytest.raises(ValueError, match="expert 0 only"):
        wall_llm.build_wall_oss_llm("/missing/ckpt", expert=1, residual_scale=4,
                                    execution_config={"model": {"num_cores": 6}})
    with pytest.raises(ValueError, match="below eight cores only"):
        build_wall_oss_vision("/missing/ckpt", residual_scale=4)


def test_reduced_rtc_action_needs_the_scaled_expert0(tmp_path):
    (tmp_path / "config.json").write_text('{"causal_action_attention_mask": false}')

    class Unscaled:
        residual_scale = 1

    for llm in (None, Unscaled()):
        with pytest.raises(ValueError, match="4-core Wall-OSS RTC needs the residual-scaled"):
            wall_action.build_wall_oss_action(
                str(tmp_path), fp16_ckpt_dir=str(tmp_path), fused=True, rtc_max_delay=10,
                llm=llm, execution_config={"model": {"num_cores": 4}})


@pytest.mark.parametrize("cores", (4, 6, 8))
def test_reduced_rtc_delegates_fixed_buckets_to_text_planner(cores):
    calls = []

    class Text:
        residual_scale = 1 if cores == 8 else 4

        def prefill_execution_plan(self, real_len, *, reserve_rows):
            calls.append((real_len, reserve_rows))
            return real_len, 224

    owner = wall.WallOssVLA.__new__(wall.WallOssVLA)
    vars(owner).update(rtc_enabled=True, llm=Text())
    plans = [owner._prefill_execution_plan(real, horizon=20) for real in (599, 640, 641, 671)]
    if cores == 8:
        assert plans == [(640, 320), (640, 320), (672, 336), (672, 336)] and not calls
    else:
        # The P640/P672 buckets stay; the reduced planner owns each bucket's chunk.
        assert plans == [(640, 224), (640, 224), (672, 224), (672, 224)]
        assert calls == [(640, 20), (640, 20), (672, 20), (672, 20)]
    for real in (598, 672):
        with pytest.raises(ValueError, match="outside the supported graph envelope"):
            owner._prefill_execution_plan(real, horizon=20)


@pytest.mark.parametrize("w8a16", (False, True))
def test_reduced_rtc_policy_pins_ordinary_routes(w8a16):
    from rpu_backend.api import WallOssPolicy
    policy = WallOssPolicy.from_checkpoint(
        "unused/checkpoint", rtc=True, w8a16=w8a16, dataset_key="ex_normal",
        camera_names=("face_view", "left_wrist_view", "right_wrist_view"),
        delta_action=True, state_bins=512, action_horizon=20, num_steps=6,
        fp16_ckpt_dir="/models/wrc-sf-v728-fp16" if w8a16 else None,
        rpu_execution={"model": {"num_cores": 6}})
    # Uniform W8A16: the eight-core hybrid's FP16 boundaries are eight-lane kernels.
    assert policy.precision == ("w8a16" if w8a16 else "fp16")
    env = policy._runtime_env
    assert (env["RPU_WALL_OSS_FUSED_DENOISE"]
            == env["RPU_WALL_OSS_DENOISE_UNROLL"]
            == env["RPU_WALL_OSS_VISION_FUSED_MERGER"] == "1")
    assert {env[name] for name in (
        "RPU_WALL_OSS_ATTN_TP8",
        "RPU_WALL_OSS_VISION_WEIGHT_OUTER", "RPU_WALL_OSS_VISION_MIXED_PRECISION",
        "RPU_WALL_OSS_PREFILL_MIXED_PRECISION")} == {"0"}


@pytest.mark.parametrize("cores", (4, 6, 8))
def test_sorting_environment_drops_the_eight_core_schedules(monkeypatch, cores):
    from examples.runners.wall_oss import sorting as sorting_runner
    from rpu_backend.adapters.wall_oss._sorting import WallSortingConfig, sorting_config_from_env
    engine = sorting_runner.validation_runner()
    for mode in ("fp16", "w8"):
        env = engine.exact_environment(ROOT, mode, cores)
        monkeypatch.setattr(os, "environ", {**os.environ, **env})
        if cores == 8:
            assert sorting_config_from_env() != WallSortingConfig()
            assert env["RPU_WALL_OSS_ATTN_TP8"] == env["RPU_WALL_OSS_PREFILL_MIXED_PRECISION"] == "1"
            continue
        # Every selector a reduced builder refuses is off before any import.
        assert sorting_config_from_env() == WallSortingConfig()
        assert {env[name] for name in engine._REDUCED_OFF} == {"0"}
        assert (env["RPU_WALL_OSS_FUSED_DENOISE"]
                == env["RPU_WALL_OSS_VISION_ROPE_SPM"]
                == env["RPU_WALL_OSS_VISION_FUSED_MERGER"] == "1")
    with pytest.raises(ValueError, match="eight-core owner only"):
        engine.exact_environment(ROOT, "action-w4", 4)


def test_sorting_example_passes_its_budget_and_refuses_action_w4(tmp_path):
    from examples.runners.wall_oss import sorting as sorting_runner
    w8 = _with_budget(common.load_config(ROOT / "examples/configs/wall_oss/sorting/w8a16.toml"), 6)
    sorting_runner.validate_example_config(w8)
    assert sorting_runner.num_cores(w8) == 6
    w4 = _with_budget(common.load_config(
        ROOT / "examples/configs/wall_oss/sorting/w8a16_action_w4a16_g32.toml"), 4)
    with pytest.raises(ValueError, match="eight-core Action W4 owner only"):
        sorting_runner.validate_example_config(w4)
    w8["rpu_execution"]["prefill"] = {"chunk_size": 256}
    with pytest.raises(ValueError, match="accepts only model.num_cores"):
        sorting_runner.validate_example_config(w8)


def test_reduced_wall_stage3_resident_prefill_and_sisc_vision_contracts():
    decoder = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    fmb = (ROOT / "src/core/fused_model_base.cpp").read_text()
    vision = (ROOT / "src/fused/rpu_qwen25vl_vision_model.cpp").read_text()
    action = (ROOT / "src/fused/rpu_wall_oss_action_step_model.cpp").read_text()
    assert "bool reduced_wall_resident_prefill(" in decoder
    assert "cfg.inter_layer_io = InterLayerIO::SPM_RESIDENT;" in decoder
    assert "(rows * (intermediate_size() / mlp_tp())) % 256 == 0" in decoder
    assert "reduced_core_silu_mul_ok" in fmb
    assert "reduced_core_sisc_raw_kv_only_enabled()" in vision
    assert "reduced_core_sisc_resident_mask_enabled()" in vision
    assert "reduced_core_sisc_single_raw_query_enabled(seq_len)" in vision
    assert "reduced_sisc_silu_mul" in vision
    assert "reduced_sisc_rows" in vision
    assert "const bool eight_cores = (num_cores() == 8);" in action


