"""GR00T N1.7 at a cold 4/6/8-core budget: host contracts only.

The example TOMLs, the composite resolver, the cold-only reconfigure check, the
16-layer text prefix admission and the native VL encoder / DiT handle order are
checked here with small doubles. None of this is a board run or numerical
evidence for the reduced-core kernels.
"""
from pathlib import Path
import re

import pytest
import torch

from examples.common import config as common
from examples.runners.gr00t import config as runner_config
from rpu_backend.adapters.gr00t import runtime as gr00t
from rpu_backend.adapters.qwen3_vl import _check_profile
from rpu_backend.api._execution import qwen3_vl_text_core_profile, qwen3_vl_text_topology
from rpu_backend.api.errors import UnsupportedModelError
from test_qwen3vl_core_config import _config as _vl_config
from test_qwen3vl_text_core46 import _install, _model, native  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "examples/configs/gr00t").rglob("*.toml"))
VLA_PROFILES = ("gr00t-n1d7-3b.fp16.vla", "gr00t-n1d7-3b.w8a16.vla", "gr00t-n1d7-3b.w4a16-g32.vla",
                "gr00t-n1d7-3b.w4a16-g32-vision.vla")
SHARED_FAST_ENV = (
    "RPU_QWEN3VL_VISION_FUSED_MERGER", "RPU_QWEN3VL_VISION_BATCH",
    "RPU_GR00T_CROSS_KV_REUSE", "RPU_GR00T_PRECOMPUTE_ADALN",
    "RPU_GR00T_PRECOMPUTE_TIME_GLUE", "RPU_GR00T_KV_SPM_ONLY",
    "RPU_GR00T_CROSS_V_TRANSPOSE_REUSE", "RPU_GR00T_VL_KV_SPM_ONLY",
)
EIGHT_ONLY_ENV = (
    "RPU_GR00T_FF1_RESIDENT_W8",
    "RPU_GR00T_SELF_QKV_MULTIBASE", "RPU_GR00T_RING_XOR3",
)


def _cores(child):
    return child.get("model", {}).get("num_cores", 8)


# --------------------------------------------------------------------------- examples


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: str(path.relative_to(ROOT)))
def test_every_example_admits_each_budget(path):
    config = common.load_config(path)
    assert config["example"]["profile_id"] in VLA_PROFILES
    assert common.core_support(config) == {4: None, 6: None, 8: None}


def test_core_choice_requires_a_registered_vla_profile():
    profiles = common.example_profiles()
    for profile_id in VLA_PROFILES:
        assert profiles[profile_id]["input_envelope"]["limits"]["allowed_num_cores"] == [4, 6, 8]
        common._validate_execution({"model": {"num_cores": 6}}, profile=profiles[profile_id])
    exact = profiles["gr00t-n1d7-3b.fp16.exact-z1"]
    assert "allowed_num_cores" not in exact["input_envelope"]["limits"]
    with pytest.raises(common.ConfigError, match="registered FP16, W8A16 or W4A16 VLA"):
        common._validate_execution({"model": {"num_cores": 4}}, profile=exact)
    common._validate_execution({"model": {"num_cores": 8}}, profile=exact)


@pytest.mark.parametrize("precision", ("fp16", "w8a16"))
@pytest.mark.parametrize("cores", (4, 6, 8))
def test_runner_binds_eight_core_selectors_off_at_reduced_budgets(precision, cores):
    config = common.load_config(ROOT / f"examples/configs/gr00t/n1_7_droid/224/{precision}.toml")
    if cores != 8:
        config.setdefault("rpu_execution", {})["model"] = {"num_cores": cores}
    env = runner_config.runtime_environment(config)
    w8 = "1" if precision == "w8a16" else "0"
    assert env["RPU_GR00T_W8A16"] == w8
    assert env["RPU_WALL_OSS_FAST_REPLAY"] == env["RPU_GR00T_NATIVE_POSITION_CACHE"] == "1"
    assert env["RPU_GR00T_QWEN3VL_SPM_Z1"] == "0"
    assert {env[name] for name in SHARED_FAST_ENV} == {"1"}
    if cores == 8:
        assert env["RPU_GR00T_FF1_RESIDENT_W8"] == env["RPU_GR00T_SELF_QKV_MULTIBASE"] == w8
        assert env["RPU_GR00T_RING_XOR3"] == "1"
    else:
        assert {env[name] for name in EIGHT_ONLY_ENV} == {"0"}


# --------------------------------------------------------------------------- resolver


@pytest.mark.parametrize("budget", (4, 6, 8))
def test_resolver_derives_each_child_budget(budget, monkeypatch):
    monkeypatch.delenv("RPU_QWEN3VL_VISION_FUSED_MERGER", raising=False)
    request = {} if budget == 8 else {"model": {"num_cores": budget}}
    root, text, vision, vl, action = gr00t._resolve_gr00t_execution(
        request, entry_point="test", qwen_z1=False, max_prefix_len=256)
    assert gr00t._gr00t_core_count(root) == budget
    if budget == 8:
        assert all("model" not in child for child in (text, vision, vl, action))
        return
    assert _cores(text) == budget
    assert _cores(vision) == _cores(vl) == _cores(action) == 4


@pytest.mark.parametrize(("request_", "z1", "match"), [
    ({"model": {"num_cores": 6}}, True, "eight-core composite"),
    ({"model": {"num_cores": 4}, "vision": {"chunk_size": 640}}, False, "must be 'auto' or one of"),
    ({"model": {"num_cores": 5}}, False, "4, 6 or 8"),
    ({"components": {"action_dit": {"model": {"num_cores": 4}}}}, False, "unknown stage"),
])
def test_resolver_refuses_eight_core_composites_and_child_budgets(request_, z1, match, monkeypatch):
    monkeypatch.delenv("RPU_QWEN3VL_VISION_FUSED_MERGER", raising=False)
    with pytest.raises(ValueError, match=match):
        gr00t._resolve_gr00t_execution(request_, entry_point="test", qwen_z1=z1, max_prefix_len=256)


def test_reconfigure_keeps_the_cold_budget(monkeypatch):
    monkeypatch.delenv("RPU_QWEN3VL_VISION_FUSED_MERGER", raising=False)
    owner = gr00t.Gr00tN1d7VLA.__new__(gr00t.Gr00tN1d7VLA)
    resolved = gr00t._resolve_gr00t_execution(
        {"model": {"num_cores": 6}}, entry_point="test", qwen_z1=False, max_prefix_len=256)
    vars(owner).update(_closed=False, _num_cores=6, _qwen_z1_plan_hash=None,
                       _vision_fused_merger=False, _max_prefix_len=256,
                       _vision_execution=resolved[2])
    owner.validate_execution_reconfigure({"model": {"num_cores": 6}})
    for change in ({}, {"model": {"num_cores": 4}}):
        with pytest.raises(ValueError, match="cold-only"):
            owner.validate_execution_reconfigure(change)


def test_resolver_admits_packed_vision_chunks_at_reduced_budgets(monkeypatch):
    monkeypatch.delenv("RPU_QWEN3VL_VISION_FUSED_MERGER", raising=False)
    for chunk in (512, 768):
        _, _, vision_cfg, _, _ = gr00t._resolve_gr00t_execution(
            {"model": {"num_cores": 6}, "vision": {"chunk_size": chunk}},
            entry_point="test", qwen_z1=False, max_prefix_len=256)
        assert vision_cfg["vision"]["chunk_size"] == chunk
        assert _cores(vision_cfg) == 4


# --------------------------------------------------------------------------- text prefix


def _prefix_config(layers=16, hidden=2048):
    model, _ = _model("2b", 6)
    model.config.num_hidden_layers = layers
    model.config.hidden_size = hidden
    return model.config


@pytest.mark.parametrize("cores", (4, 6))
def test_text_prefix_is_admitted_only_for_the_2b_sixteen_layer_backbone(cores):
    profile = qwen3_vl_text_core_profile(_prefix_config(), cores)
    assert profile.num_layers == gr00t._SELECT_LAYER == 16
    topology = qwen3_vl_text_topology(_prefix_config(), cores)
    assert (topology.num_cores, topology.attn_tp, topology.mlp_tp) == (cores, 4, cores)
    for config in (_prefix_config(layers=15), _prefix_config(layers=16, hidden=2560)):
        with pytest.raises(ValueError, match="exact dense Qwen3-VL"):
            qwen3_vl_text_core_profile(config, cores)
    # The ordinary Qwen3-VL adapter still admits only complete registered towers.
    truncated = _vl_config(2)
    truncated.text_config.num_hidden_layers = 16
    with pytest.raises(UnsupportedModelError):
        _check_profile(truncated)
    _check_profile(_vl_config(2))


@pytest.mark.parametrize("cores", (4, 6))
def test_text_prefix_installs_on_the_reduced_template(native, cores):  # noqa: F811
    model, topology = _model("2b", cores)
    model.config.num_hidden_layers = 16
    del model.layers[16:]
    bound = qwen3_vl_text_topology(model.config, cores)
    assert bound == topology
    assert _install(model, bound) == 1
    assert native["create_topologies"] == [{"num_cores": cores, "vocab_size": 151936}]
    model._rpu_decoder_retirement_state.retire()


# --------------------------------------------------------------------------- native handles


def _ops(monkeypatch, prefix, topology):
    calls = []
    def op(name, result=None):
        def call(*args):
            calls.append(name)
            return result
        monkeypatch.setattr(torch.ops.rpu, f"{prefix}_{name}", call, raising=False)
    op("create", 7)
    op("set_execution_core_count")
    op("get_execution_topology", topology)
    op("set_weights")
    op("destroy")
    return calls


def test_vl_encoder_binds_four_lanes_before_weights(monkeypatch):
    calls = _ops(monkeypatch, "gr00t_vl_encoder", [4, 4, 4, 8, 32, 8192])
    assert gr00t._configure_gr00t_vl_encoder_handle(
        weight_args=(), scale_args=(), w8a16=False, cores=4) == 7
    assert calls == ["create", "set_execution_core_count", "set_weights", "get_execution_topology"]
    calls.clear()
    monkeypatch.setattr(torch.ops.rpu, "gr00t_vl_encoder_get_execution_topology",
                        lambda handle: [8, 8, 8, 8, 32, 8192], raising=False)
    with pytest.raises(RuntimeError, match="cold weight layout"):
        gr00t._configure_gr00t_vl_encoder_handle(
            weight_args=(), scale_args=(), w8a16=False, cores=4)
    assert calls[-1] == "destroy"


def test_dit_prepares_four_lane_tables_when_requested(monkeypatch):
    calls = _ops(monkeypatch, "gr00t_dit", [4, 4, 4, 8, 32, 6144])
    monkeypatch.setattr(torch.ops.rpu, "gr00t_denoise_set_weights",
                        lambda *args: calls.append("denoise_set_weights"), raising=False)
    for table in ("prepare_adaln_table", "prepare_time_tables"):
        monkeypatch.setattr(torch.ops.rpu, f"gr00t_dit_{table}",
                            lambda *args, table=table: calls.append(table), raising=False)
    monkeypatch.setenv("RPU_GR00T_PRECOMPUTE_ADALN", "1")
    monkeypatch.setenv("RPU_GR00T_PRECOMPUTE_TIME_GLUE", "1")
    assert gr00t._configure_gr00t_denoise_handle(
        dit_args=(), dit_scale_args=(), denoise_args=(), w8a16=False, cores=4) == 7
    assert calls == ["create", "set_execution_core_count", "set_weights",
                     "denoise_set_weights", "get_execution_topology",
                     "prepare_adaln_table", "prepare_time_tables"]


def test_native_owners_have_no_fixed_eight_core_layout():
    for name in ("rpu_gr00t_vl_encoder_model.cpp", "rpu_gr00t_dit_model.cpp"):
        source = (ROOT / "src/fused" / name).read_text()
        assert not re.search(r"\bNUM_CORES\b", source), name
        assert "set_execution_core_count(static_cast<int>(cores))" in source, name
    registrations = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    for prefix in ("gr00t_vl_encoder", "gr00t_dit"):
        assert f'm.def("{prefix}_set_execution_core_count(int handle, int cores) -> ()"' in registrations
        assert f'm.def("{prefix}_get_execution_topology(int handle) -> int[]"' in registrations
