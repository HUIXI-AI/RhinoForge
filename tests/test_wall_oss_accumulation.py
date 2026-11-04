"""Cold Wall ACC selection through real API/builders; no device execution."""
import json
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.api import WallOssPolicy
from rpu_backend.adapters.wall_oss import action, llm, runtime
from rpu_backend.adapters.wall_oss._sorting import WallSortingConfig
from rpu_backend.tests.acceptance.test_wall_oss_builder_lifecycle import (
    native_ops, _patch_vla_preflight,
)


@pytest.mark.parametrize("kind", ["prefill", "action"])
def test_decoder_envelope_uses_requested_capacity(native_ops, monkeypatch, kind):
    calls = []
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_chunk_envelope",
                        lambda *args: calls.append(args))
    kwargs = dict(set_weights=lambda handle, *_: native_ops["step"]("weights", handle),
                  weight_args=(), max_seq_len=16384)
    if kind == "prefill":
        handle = llm._configure_causal_decoder_handle(
            **kwargs, chunk_size_cap=384, equal_two_prefill=False)
    else:
        handle = action._configure_action_handle(**kwargs, fused=False)
    try:
        assert calls[0][:2] == (handle, 16384)
    finally:
        torch.ops.rpu.causal_decoder_destroy(handle)


@pytest.mark.parametrize("stage", ["prefill", "vision", "action"])
def test_public_accumulation_bool_and_rtc_rejection(stage):
    for value in (False, True):
        requested = {stage: {"linear_acc32": value}}
        policy = WallOssPolicy.from_checkpoint("/unused", rpu_execution=requested)
        assert policy._rpu_execution == requested
    for value in (0, 1, "true", None):
        with pytest.raises(TypeError, match="linear_acc32"):
            WallOssPolicy.from_checkpoint(
                "/unused", rpu_execution={stage: {"linear_acc32": value}})
    with pytest.raises(ValueError, match="RTC.*linear_acc32"):
        WallOssPolicy.from_checkpoint(
            "/unused", rtc=True, rpu_execution={stage: {"linear_acc32": True}})
    with pytest.raises(ValueError, match="RTC.*linear_acc32"):
        runtime.build_wall_oss_vla(
            "/must-not-open", rtc=True,
            rpu_execution={"components": {
                {"prefill": runtime.WALL_OSS_TEXT_COMPONENT,
                 "vision": runtime.WALL_OSS_VISION_COMPONENT,
                 "action": runtime.WALL_OSS_ACTION_COMPONENT}[stage]: {
                    stage: {"linear_acc32": True}}}})


@pytest.mark.parametrize("kind", ["prefill", "action", "unfused_decoder"])
@pytest.mark.parametrize("enabled", [False, True])
def test_accumulation_bound_before_weights(native_ops, monkeypatch, kind, enabled):
    events = native_ops["events"]
    setter = ("wall_oss_action_step_set_linear_acc32" if kind == "action"
              else "causal_decoder_set_linear_acc32")
    def set_mode(handle, value):
        assert value is True
        native_ops["step"]("acc32", handle)
    monkeypatch.setattr(torch.ops.rpu, setter, set_mode, raising=False)
    kwargs = dict(set_weights=lambda handle, *_: native_ops["step"]("weights", handle),
                  weight_args=(), linear_acc32=enabled)
    if kind == "prefill":
        handle = llm._configure_causal_decoder_handle(
            **kwargs, chunk_size_cap=384, equal_two_prefill=True)
    else:
        handle = action._configure_action_handle(**kwargs, fused=kind == "action")
    names = [name for name, _ in events]
    assert names.count("acc32") == int(enabled)
    if enabled:
        assert names.index("acc32") < names.index("weights")
    destroy = (torch.ops.rpu.wall_oss_action_step_destroy if kind == "action"
               else torch.ops.rpu.causal_decoder_destroy)
    destroy(handle)
    assert native_ops["alive"] == {}


@pytest.mark.parametrize("kind", ["prefill", "action"])
def test_failed_cold_accumulation_setter_retires_handle(native_ops, monkeypatch, kind):
    def reject(*args):
        raise RuntimeError("ACC setter failed")
    setter = ("wall_oss_action_step_set_linear_acc32" if kind == "action"
              else "causal_decoder_set_linear_acc32")
    monkeypatch.setattr(torch.ops.rpu, setter, reject, raising=False)
    kwargs = dict(set_weights=lambda *_: pytest.fail("must not install weights"),
                  weight_args=(), linear_acc32=True)
    with pytest.raises(RuntimeError, match="ACC setter failed"):
        if kind == "prefill":
            llm._configure_causal_decoder_handle(
                **kwargs, chunk_size_cap=384, equal_two_prefill=True)
        else:
            action._configure_action_handle(**kwargs, fused=True)
    assert native_ops["alive"] == {}


@pytest.mark.parametrize("env", ["RPU_WALL_OSS_ACTION_WEIGHT_RESIDENCY",
                                  "RPU_WALL_OSS_ACTION_KV_OWNER_FANOUT"])
def test_action_acc32_conflicts_reject_before_assets(monkeypatch, env):
    monkeypatch.setenv(env, "1")
    with pytest.raises(ValueError, match="ACC32.*residency.*fanout"):
        action.build_wall_oss_action(
            "/must-not-open", fused=True,
            execution_config={"action": {"linear_acc32": True}})
    with pytest.raises(ValueError, match="ACC32.*residency.*fanout"):
        action._load_expert1_decoder(
            None, {}, 36, 2048, linear_acc32=True, sorting=WallSortingConfig(),
            **{"weight_residency" if "RESIDENCY" in env else "kv_owner_fanout": True})


def test_action_acc32_unfused_and_rtc_and_prefill_mixed_reject_before_assets(monkeypatch):
    for kwargs, message in (({"fused": False}, "requires fused=True"),
                            ({"fused": True, "rtc_max_delay": 10}, "RTC")):
        with pytest.raises(ValueError, match=message):
            action.build_wall_oss_action(
                "/must-not-open", **kwargs,
                execution_config={"action": {"linear_acc32": True}})
    monkeypatch.setenv("RPU_WALL_OSS_PREFILL_MIXED_PRECISION", "1")
    with pytest.raises(ValueError, match="mixed RTC prefill"):
        llm.build_wall_oss_llm(
            "/must-not-open", execution_config={"prefill": {"linear_acc32": True}})


@pytest.mark.parametrize("hybrid", [False, True])
def test_runtime_keeps_independent_component_accumulation(monkeypatch, hybrid):
    # Use the real builder up to the three component allocation boundaries.
    monkeypatch.setattr(runtime.os, "environ", runtime.os.environ.copy())
    _patch_vla_preflight(monkeypatch)
    monkeypatch.setattr(runtime, "_fused_denoise_enabled", lambda: True)
    monkeypatch.setattr(runtime, "validate_w8_nvfp4_checkpoint_pair", lambda *_: None)
    calls = {}
    def component(name, result):
        def build(*args, **kwargs):
            calls[name] = kwargs["execution_config"]
            return result
        return build
    text = object()
    monkeypatch.setattr(runtime, "build_wall_oss_llm", component("prefill", text))
    monkeypatch.setattr(runtime, "build_wall_oss_action", component("action", SimpleNamespace(llm=text)))
    monkeypatch.setattr(runtime, "build_wall_oss_vision", component("vision", object()))
    monkeypatch.setattr(runtime, "WallOssVLA", lambda **kwargs: SimpleNamespace(**kwargs))
    config = {"prefill": {"linear_acc32": True}, "vision": {"linear_acc32": False},
              "action": {"linear_acc32": True}}
    built = runtime.build_wall_oss_vla(
        "/unused", fp16_ckpt_dir="/unused", dataset_key="dataset",
        w8_nvfp4a16=hybrid, rpu_execution=config)
    assert built.execution_config == config
    assert calls["action"]["prefill"]["linear_acc32"] is True
    assert calls["action"]["action"]["linear_acc32"] is True
    assert calls["vision"]["vision"]["linear_acc32"] is False
    if hybrid:
        assert calls["prefill"]["prefill"]["linear_acc32"] is True


@pytest.mark.parametrize("enabled", [False, True])
def test_llm_builder_forwards_accumulation_to_cold_handle(monkeypatch, tmp_path, enabled):
    monkeypatch.delenv("RPU_WALL_OSS_PREFILL_MIXED_PRECISION", raising=False)
    cfg = {"hidden_size": 16, "num_attention_heads": 1, "num_key_value_heads": 1,
           "head_dim": 16, "intermediate_size": 16, "rms_norm_eps": 1e-6,
           "rope_theta": 10000, "rope_scaling": {"mrope_section": [2, 2, 4]},
           "num_hidden_layers": 0}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    weight = torch.ones(16)
    monkeypatch.setattr(llm, "_SafeTensorStore", lambda *_: SimpleNamespace(get_tensor=lambda _: weight))
    monkeypatch.setattr(llm, "_to_rpu_half", lambda value: value)
    monkeypatch.setattr(llm, "_build_rope_tables", lambda *_: (weight, weight))
    monkeypatch.setattr(llm, "RPUCache", lambda **_: object())
    from rpu_backend.graph import GraphRuntimePolicy
    graph_policy = GraphRuntimePolicy(batch_snapshot_mode="off") if enabled else None
    graph_calls = []
    monkeypatch.setattr(llm.rpu_backend.graph, "GraphCache",
                        lambda **kwargs: graph_calls.append(kwargs) or object())
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_weights", lambda *_: None, raising=False)
    def stop_at_native(**kwargs):
        assert kwargs["linear_acc32"] is enabled
        raise RuntimeError("reached real cold handle boundary")
    monkeypatch.setattr(llm, "_configure_causal_decoder_handle", stop_at_native)
    with pytest.raises(RuntimeError, match="reached real cold handle boundary"):
        llm.build_wall_oss_llm(str(tmp_path), execution_config={
            "prefill": {"linear_acc32": enabled}},
            **({"graph_runtime_policy": graph_policy} if graph_policy is not None else {}))
    assert graph_calls == ([{"runtime_policy": graph_policy}] if enabled else [{}])


@pytest.mark.parametrize("enabled", [False, True])
def test_action_builder_and_loader_forward_accumulation(monkeypatch, tmp_path, enabled):
    # Stop before native installation; no layer math or device transfer is needed
    # to exercise the builder -> expert loader -> cold handle configuration.
    cfg = {"hidden_size": 16, "num_attention_heads": 1, "num_key_value_heads": 1,
           "head_dim": 16, "rms_norm_eps": 1e-6, "rope_theta": 10000,
           "rope_scaling": {"mrope_section": [2, 2, 4]}, "num_hidden_layers": 0,
           "action_hidden_size": 16,
           "experts": [{}, {"hidden_size": 16, "intermediate_size": 16}]}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    weight = torch.ones(16, 16)
    monkeypatch.setattr(action, "_SafeTensorStore", lambda *_: SimpleNamespace(get_tensor=lambda _: weight))
    monkeypatch.setattr(action, "_to_rpu_half", lambda value: value)
    monkeypatch.setattr(action, "_build_rope_tables", lambda *_: (weight, weight))
    from rpu_backend.graph import GraphRuntimePolicy
    graph_policy = GraphRuntimePolicy(batch_snapshot_mode="off") if enabled else None
    graph_calls = []
    monkeypatch.setattr(action.rpu_backend.graph, "GraphCache",
                        lambda **kwargs: graph_calls.append(kwargs) or object())
    fused = {name: weight for name in ("w_comb", "w3", "proj_back", "w2_t", "w2_a")}
    fused.update(action_dim=16, k_comb_pad=16)
    monkeypatch.setattr(action, "_build_fused_action_weights", lambda *_: (fused, {}))
    monkeypatch.setattr(torch.ops.rpu, "wall_oss_action_step_set_weights", lambda *_: None, raising=False)
    def stop_at_native(**kwargs):
        assert kwargs["linear_acc32"] is enabled
        assert kwargs["fused"] is True
        raise RuntimeError("reached real action cold handle boundary")
    monkeypatch.setattr(action, "_configure_action_handle", stop_at_native)
    with pytest.raises(RuntimeError, match="reached real action cold handle boundary"):
        action.build_wall_oss_action(
            str(tmp_path), fp16_ckpt_dir=str(tmp_path), fused=True,
            llm=SimpleNamespace(num_layers=0),
            execution_config={"action": {"linear_acc32": enabled}},
            **({"graph_runtime_policy": graph_policy} if graph_policy is not None else {}))
    assert graph_calls == ([{"runtime_policy": graph_policy}] if enabled else [{}])


@pytest.mark.parametrize("mode", ("auto", "off"))
def test_wall_public_graph_policy_reaches_every_component_builder(monkeypatch, tmp_path, mode):
    from rpu_backend.graph import GraphRuntimePolicy
    requested = GraphRuntimePolicy(batch_snapshot_mode=mode)
    public = WallOssPolicy.from_pretrained(tmp_path, graph_runtime_policy=requested)
    assert public._graph_runtime_policy is requested
    monkeypatch.setattr(runtime.os, "environ", runtime.os.environ.copy())
    _patch_vla_preflight(monkeypatch)
    monkeypatch.setattr(runtime, "_fused_denoise_enabled", lambda: True)
    calls = {}
    text = object()
    def build(name, result):
        def component(*args, **kwargs):
            calls[name] = kwargs["graph_runtime_policy"]
            return result
        return component
    monkeypatch.setattr(runtime, "build_wall_oss_action", build("action", SimpleNamespace(llm=text)))
    monkeypatch.setattr(runtime, "build_wall_oss_vision", build("vision", object()))
    monkeypatch.setattr(runtime, "WallOssVLA", lambda **kwargs: SimpleNamespace(**kwargs))
    runtime.build_wall_oss_vla("/unused", dataset_key="dataset", graph_runtime_policy=requested)
    assert calls == {"action": requested, "vision": requested}


@pytest.mark.parametrize("value", (True, "auto", {}, 1))
def test_wall_graph_policy_type_rejects_before_assets(value):
    from rpu_backend.adapters.wall_oss import vision
    for call in (
        lambda: WallOssPolicy.from_checkpoint("/must-not-load", graph_runtime_policy=value),
        lambda: runtime.build_wall_oss_vla("/must-not-load", graph_runtime_policy=value),
        lambda: llm.build_wall_oss_llm("/must-not-load", graph_runtime_policy=value),
        lambda: vision.build_wall_oss_vision("/must-not-load", graph_runtime_policy=value),
        lambda: action.build_wall_oss_action("/must-not-load", graph_runtime_policy=value),
    ):
        with pytest.raises(TypeError, match="graph_runtime_policy"):
            call()
