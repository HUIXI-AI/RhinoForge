"""Board-free ownership contracts for Wall-OSS component builders."""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.wall_oss import action, llm, runtime, vision


_REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(autouse=True)
def isolated_retirement(monkeypatch):
    from rpu_backend.api import _execution, causal_lm
    from rpu_backend.adapters.wall_oss import _retirement
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(causal_lm, "_LIVE_TERMINAL_REASON", None)
    monkeypatch.setattr(_retirement, "_FAILED_RETIREMENTS", [])


class _RuntimeDict(dict):
    """Test double that supports runtime metadata publication."""


@pytest.mark.parametrize("enabled,available", [(False, False), (True, True), (True, False)])
def test_base_pair_reserves_arenas_before_component_allocation(monkeypatch, enabled, available):
    from rpu_backend.graph import GraphRuntimePolicy

    events = []
    normalizer = {"min.dataset": torch.zeros(26), "delta.dataset": torch.ones(26)}
    monkeypatch.setenv("RPU_WALL_OSS_VISION_PAIR_252", str(int(enabled)))
    monkeypatch.setattr(runtime.torch, "load", lambda *a, **k: normalizer)
    monkeypatch.setattr(runtime.AutoProcessor, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(runtime.AutoConfig, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(runtime, "_build_rope_helper", lambda *a, **k: object())
    monkeypatch.setattr(runtime, "_fused_denoise_enabled", lambda: True)

    def reserve(policy):
        assert events == []
        assert policy.graph_arena_count == 5
        events.append("reserve")
        return available

    def component(name, value):
        def build(*args, **kwargs):
            assert events == (["reserve"] if enabled else []) + (["action"] if name == "vision" else [])
            events.append(name)
            return value
        return build

    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas", reserve)
    monkeypatch.setattr(runtime, "build_wall_oss_action", component("action", SimpleNamespace(llm=object())))
    monkeypatch.setattr(runtime, "build_wall_oss_vision", component("vision", object()))
    monkeypatch.setattr(runtime, "WallOssVLA", lambda **kwargs: _RuntimeDict(kwargs))
    kwargs = dict(ckpt_dir="/models/base", fp16_ckpt_dir="/models/base", dataset_key="dataset")
    if enabled and not available:
        with pytest.raises(RuntimeError, match="five cold Graph arenas"):
            runtime.build_wall_oss_vla(**kwargs)
        assert events == ["reserve"]
    else:
        built = runtime.build_wall_oss_vla(**kwargs)
        assert events == (["reserve"] if enabled else []) + ["action", "vision"]
        if enabled:
            assert built._graph_arena_policy.graph_arena_count == 5
        else:
            assert not hasattr(built, "_graph_arena_policy")


class _Cache:
    def __init__(self, events, name):
        self._events = events
        self._name = name

    def clear(self):
        self._events.append(("clear", self._name))

    def cache_invariant_ok(self):
        return True

    def capture(self, signature):
        raise AssertionError("runtime-completeness checks must not capture graphs")


@pytest.fixture
def native_ops(monkeypatch):
    state = {"next": 1, "alive": {}, "events": [], "fail": None}
    state["envelopes"] = []

    def step(name, handle, result=None):
        state["events"].append((name, handle))
        if state["fail"] == name:
            raise RuntimeError(f"synthetic {name} failure")
        return result

    def create(kind):
        handle = state["next"]
        state["next"] += 1
        state["alive"][handle] = kind
        state["events"].append((f"{kind}_create", handle))
        return handle

    def destroy(kind, handle):
        state["events"].append((f"{kind}_destroy", handle))
        assert state["alive"].pop(handle) == kind

    monkeypatch.setattr(
        torch.ops.rpu, "qwen25vl_vision_create",
        lambda: create("vision"), raising=False,
    )
    def envelope(handle, capacity, chunk):
        assert state["events"][-1] == ("vision_weights", handle)
        state["envelopes"].append((handle, capacity, chunk))
        if state["fail"] == "vision_envelope":
            raise RuntimeError("synthetic vision_envelope failure")
    monkeypatch.setattr(torch.ops.rpu, "qwen25vl_vision_set_chunk_envelope", envelope, raising=False)
    monkeypatch.setattr(
        torch.ops.rpu, "qwen25vl_vision_set_rope",
        lambda handle, *_: step("vision_rope", handle), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "qwen25vl_vision_set_execution_routes",
        lambda handle, *_: step("vision_routes", handle), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "qwen25vl_vision_set_chunk_size",
        lambda handle, *_: step("vision_chunk", handle), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "qwen25vl_vision_position_idx_keepalive",
        lambda handle: step("vision_keepalive", handle, object()), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "qwen25vl_vision_set_merger_weights",
        lambda handle, *_: step("vision_merger", handle), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "qwen25vl_vision_destroy",
        lambda handle: destroy("vision", handle), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "causal_decoder_create",
        lambda: create("decoder"), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "causal_decoder_set_wall_prefill_routes",
        lambda handle, *_: step("decoder_routes", handle), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "causal_decoder_set_chunk_size_cap",
        lambda handle, *_: step("decoder_chunk", handle), raising=False,
    )
    # MR-A: certified chunk envelope, declared between the cap and equal-two
    # steps in _configure_causal_decoder_handle.
    monkeypatch.setattr(
        torch.ops.rpu, "causal_decoder_set_chunk_envelope",
        lambda handle, *_: step("decoder_envelope", handle), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "causal_decoder_set_equal_two_prefill",
        lambda handle, *_: step("decoder_equal_two", handle), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "causal_decoder_set_chunk_size_override",
        lambda handle, *_: step("decoder_override", handle), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "causal_decoder_destroy",
        lambda handle: destroy("decoder", handle), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "wall_oss_action_step_create",
        lambda: create("action"), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "wall_oss_action_step_destroy",
        lambda handle: destroy("action", handle), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "wall_oss_action_step_set_kv_owner_fanout",
        lambda handle, *_: step("action_kv_owner", handle), raising=False,
    )
    state["step"] = step
    return state


@pytest.mark.parametrize(
    "failure",
    [
        "vision_weights",
        "vision_envelope",
        "vision_routes",
        "vision_chunk",
        "vision_rope",
        "vision_keepalive",
        "vision_merger",
    ],
)
def test_vision_native_configuration_failure_destroys_handle(
    native_ops, failure
):
    native_ops["fail"] = failure

    with pytest.raises(RuntimeError, match=failure):
        vision._configure_qwen25vl_vision_handle(
            set_weights=lambda handle, *_: native_ops["step"](
                "vision_weights", handle
            ),
            weight_args=(), freq_cos=object(), freq_sin=object(),
            merger_args=(), max_seq_len=1536,
        )

    assert native_ops["alive"] == {}
    assert native_ops["events"][-1][0] == "vision_destroy"


def test_vision_exact_chunk_is_installed_before_rope(native_ops):
    handle, _ = vision._configure_qwen25vl_vision_handle(
        set_weights=lambda value, *_: native_ops["step"](
            "vision_weights", value
        ),
        weight_args=(), freq_cos=object(), freq_sin=object(),
        merger_args=(), max_seq_len=1536, chunk_size=112,
    )

    assert [name for name, _ in native_ops["events"]] == [
        "vision_create",
        "vision_routes",
        "vision_weights",
        "vision_chunk",
        "vision_rope",
        "vision_keepalive",
        "vision_merger",
    ]
    torch.ops.rpu.qwen25vl_vision_destroy(handle)


def test_vision_cold_envelope_receives_supplied_capacity(native_ops):
    handle, _ = vision._configure_qwen25vl_vision_handle(
        set_weights=lambda h, *_: native_ops["step"]("vision_weights", h),
        weight_args=(), freq_cos=object(), freq_sin=object(), merger_args=(),
        max_seq_len=8192,
    )
    try:
        assert native_ops["envelopes"] == [(handle, 8192, 0)]
    finally:
        torch.ops.rpu.qwen25vl_vision_destroy(handle)


@pytest.mark.parametrize(
    "failure",
    [
        "decoder_routes", "decoder_weights", "decoder_chunk",
        "decoder_equal_two", "decoder_override",
    ],
)
def test_llm_native_configuration_failure_destroys_handle(native_ops, failure):
    native_ops["fail"] = failure

    with pytest.raises(RuntimeError, match=failure):
        llm._configure_causal_decoder_handle(
            set_weights=lambda handle, *_: native_ops["step"](
                "decoder_weights", handle
            ),
            weight_args=(), chunk_size_cap=256, equal_two_prefill=True,
        )

    assert native_ops["alive"] == {}
    assert native_ops["events"][-1][0] == "decoder_destroy"


def test_llm_exact_chunk_is_installed_after_cold_policy(native_ops):
    handle = llm._configure_causal_decoder_handle(
        set_weights=lambda value, *_: native_ops["step"](
            "decoder_weights", value
        ),
        weight_args=(),
        chunk_size_cap=384,
        equal_two_prefill=False,
        chunk_size_override=320,
    )

    assert [name for name, _ in native_ops["events"]] == [
        "decoder_create",
        "decoder_routes",
        "decoder_weights",
        "decoder_chunk",
        "decoder_envelope",
        "decoder_equal_two",
        "decoder_override",
    ]
    torch.ops.rpu.causal_decoder_destroy(handle)


@pytest.mark.parametrize("owner_direct", [False, True])
def test_llm_owner_direct_route_is_installed_before_weights(native_ops, monkeypatch, owner_direct):
    routes = []

    def set_routes(handle, mixed, staged, rmsnorm, owner):
        routes.append((mixed, staged, rmsnorm, owner))
        native_ops["step"]("decoder_routes", handle)

    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_wall_prefill_routes", set_routes)
    handle = llm._configure_causal_decoder_handle(
        set_weights=lambda value, *_: native_ops["step"]("decoder_weights", value),
        weight_args=(), chunk_size_cap=384, equal_two_prefill=True,
        wall_prefill_routes=(True, True, llm._RMSNORM_CAP_BASE, owner_direct),
    )
    try:
        assert routes == [(True, True, llm._RMSNORM_CAP_BASE, owner_direct)]
        names = [name for name, _ in native_ops["events"]]
        assert names.index("decoder_routes") < names.index("decoder_weights")
    finally:
        torch.ops.rpu.causal_decoder_destroy(handle)


def test_llm_default_plan_uses_native_dry_chunk(monkeypatch):
    calls = []

    def resolve(
        handle, execution_len, position, _causal, _mask_kv_len,
        _rope_mode, _graph_lifecycle, logical_len,
    ):
        calls.append((handle, execution_len, position, logical_len))
        return [(96, 96, 96)]

    monkeypatch.setattr(
        torch.ops.rpu,
        "causal_decoder_resolve_prefill_stage_domain",
        resolve,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "causal_decoder_planner_cache_identity",
        lambda _handle: (1,), raising=False,
    )
    owner = object.__new__(llm.WallOssLLM)
    owner._handle = 7
    owner._graph_cache = None
    owner.cache = SimpleNamespace(max_seq_len=512)
    owner._rpu_execution = {}
    owner._prefill_chunk_size_cap = 256
    owner._prefill_pad16 = True
    owner._prefill_equal_two = False
    owner._prefill_mixed_precision = False
    owner._attn_geometry = {}

    assert owner.prefill_execution_plan(192) == (192, 96)
    assert owner.prefill_execution_plan(180) == (192, 96)
    assert calls == [(7, 192, 0, 192), (7, 192, 0, 180)]


@pytest.mark.parametrize("fused,kind", [(False, "decoder"), (True, "action")])
def test_action_native_configuration_failure_destroys_handle(
    native_ops, fused, kind
):
    native_ops["fail"] = "action_weights"

    with pytest.raises(RuntimeError, match="action_weights"):
        action._configure_action_handle(
            fused=fused,
            set_weights=lambda handle, *_: native_ops["step"](
                "action_weights", handle
            ),
            weight_args=(),
        )

    assert native_ops["alive"] == {}
    assert native_ops["events"][-1][0] == f"{kind}_destroy"


def test_unfused_action_exact_chunk_is_installed_after_envelope(native_ops):
    handle = action._configure_action_handle(
        fused=False,
        set_weights=lambda value, *_: native_ops["step"](
            "action_weights", value
        ),
        weight_args=(),
        chunk_size_override=32,
    )

    assert [name for name, _ in native_ops["events"]] == [
        "decoder_create",
        "action_weights",
        "decoder_envelope",
        "decoder_override",
    ]
    torch.ops.rpu.causal_decoder_destroy(handle)


def test_fused_action_installs_kv_owner_after_weights(native_ops):
    handle = action._configure_action_handle(
        fused=True,
        set_weights=lambda value, *_: native_ops["step"](
            "action_weights", value
        ),
        weight_args=(),
        kv_owner_fanout=True,
    )

    assert [name for name, _ in native_ops["events"]] == [
        "action_create",
        "action_weights",
        "action_kv_owner",
    ]
    torch.ops.rpu.wall_oss_action_step_destroy(handle)


def test_component_close_clears_graph_and_finalizes_exactly_once(monkeypatch):
    events = []
    monkeypatch.setattr(
        vision, "_destroy_qwen25vl_vision_handle",
        lambda handle: events.append(("destroy", "vision", handle)),
    )
    monkeypatch.setattr(
        llm, "_destroy_causal_decoder_handle",
        lambda handle: events.append(("destroy", "llm", handle)),
    )
    monkeypatch.setattr(
        action, "_destroy_action_handle",
        lambda handle, fused: events.append(("destroy", "action", handle, fused)),
    )
    monkeypatch.setattr(
        torch.ops.rpu, "qwen25vl_vision_destroy",
        lambda handle: events.append(("destroy", "vision", handle)),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "causal_decoder_destroy",
        lambda handle: events.append(("destroy", "llm", handle)),
        raising=False,
    )

    vision_model = vision.WallOssVision(
        handle=11, cache=object(), graph_cache=_Cache(events, "vision"),
        keepalive=object(), patch_w=object(), merger=lambda value: value,
        hidden_size=8, num_layers=1, spatial_merge_size=2, window=False,
        window_size=112, patch_size=14, experimental_schedule=(0, False),
    )
    llm_model = llm.WallOssLLM(
        handle=12, cache=object(), graph_cache=_Cache(events, "llm"),
        embed_w=object(), hidden_size=8, num_layers=1, head_dim=16,
        rope_theta=10000.0, mrope_section=[], prefill_chunk_size_cap=16,
        prefill_pad16=False, prefill_equal_two=False, partial_mrope=True,
        attn_geometry=dict(num_q_heads=1, num_kv_heads=1, num_cores=1,
                           head_dim=16, attn_mask_type=1),
    )
    action_model = action.WallOssAction(
        llm=llm_model, handle=13, graph_cache=_Cache(events, "action"),
        hidden_size=8, num_layers=1, action_dim=2, action_hidden=4,
        w1=object(), w2=object(), w3=object(), proj_back=object(),
        head_dim=16, rope_theta=10000.0, mrope_section=[], fused=True,
        fused_extra={
            "w1_dof": object(), "w2_a": object(), "w2_t": object(),
            "action_dim_pad": 16, "k_comb_pad": 16,
        },
        partial_mrope=True, owns_llm=True,
    )

    vision_model.close()
    vision_model.close()
    action_model.close()
    action_model.close()

    assert events == [
        ("clear", "vision"),
        ("destroy", "vision", 11),
        ("clear", "action"),
        ("clear", "llm"),
        ("destroy", "action", 13, True),
        ("destroy", "llm", 12),
    ]
    assert all(owner._closed and owner._handle is None
               for owner in (vision_model, action_model, llm_model))


@pytest.mark.parametrize(
    "owner_cls,destroy_name,kind",
    [
        (vision.WallOssVision, "qwen25vl_vision_destroy", "vision"),
        (llm.WallOssLLM, "causal_decoder_destroy", "llm"),
    ],
)
def test_component_close_destroy_failure_is_terminal_and_never_retried(
    monkeypatch, owner_cls, destroy_name, kind
):
    events = []

    class Finalizer:
        alive = True

        def detach(self):
            events.append(("detach", kind))
            self.alive = False

    destroy_calls = 0

    def destroy(handle):
        nonlocal destroy_calls
        destroy_calls += 1
        events.append(("destroy", kind, handle))
        if destroy_calls == 1:
            raise RuntimeError(f"synthetic {kind} destroy failure")

    monkeypatch.setattr(torch.ops.rpu, destroy_name, destroy, raising=False)
    owner = owner_cls.__new__(owner_cls)
    owner._handle = 17
    owner._graph_cache = _Cache(events, kind)
    owner._handle_finalizer = Finalizer()
    owner._closed = False

    with pytest.raises(RuntimeError, match=f"{kind} destroy failure"):
        owner.close()

    assert owner._closed is False
    assert owner._handle_finalizer.alive
    assert events == [("clear", kind), ("destroy", kind, 17)]

    with pytest.raises(RuntimeError, match="cleanup failed"):
        owner.close()
    assert owner._closed is False and owner._handle == 17
    assert owner._handle_finalizer.alive
    assert events == [("clear", kind), ("destroy", kind, 17)]


def _patch_vla_preflight(monkeypatch):
    normalizer = {"min.dataset": torch.tensor(0), "delta.dataset": torch.tensor(1)}
    monkeypatch.setattr(runtime.torch, "load", lambda *args, **kwargs: normalizer)
    monkeypatch.setattr(
        runtime.AutoProcessor, "from_pretrained", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        runtime.AutoConfig, "from_pretrained", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        runtime, "_build_rope_helper", lambda config, **kwargs: object()
    )
    monkeypatch.setattr(runtime, "_fused_denoise_enabled", lambda: False)


def test_rtc_w4_requires_exact_w8_companion_before_runtime_env_mutation(monkeypatch):
    monkeypatch.delenv("RPU_WALL_OSS_PREFILL_MIXED_PRECISION", raising=False)
    monkeypatch.setenv("RPU_LINEAR_ACC32", "1")
    monkeypatch.setenv("RPU_RMSNORM_NEWTON", "1")
    monkeypatch.setenv("RPU_WALL_OSS_ACTION_FP32_TAIL", "1")
    monkeypatch.setenv("RPU_WALL_OSS_VISION_WEIGHT_OUTER", "1")
    monkeypatch.setenv("RPU_WALL_OSS_VISION_LAYER_GROUP", "32")
    with pytest.raises(ValueError, match="rtc_w8_ckpt_dir"):
        runtime.build_wall_oss_vla(
            "/checkpoint", fp16_ckpt_dir="/checkpoint", rtc=True, w4a16=True
        )
    assert runtime.os.environ["RPU_LINEAR_ACC32"] == "1"
    assert runtime.os.environ["RPU_RMSNORM_NEWTON"] == "1"
    assert runtime.os.environ["RPU_WALL_OSS_ACTION_FP32_TAIL"] == "1"
    assert runtime.os.environ["RPU_WALL_OSS_VISION_WEIGHT_OUTER"] == "1"
    assert runtime.os.environ["RPU_WALL_OSS_VISION_LAYER_GROUP"] == "32"
    assert "RPU_WALL_OSS_PREFILL_MIXED_PRECISION" not in runtime.os.environ


def test_rtc_w4_default_pair_validation_precedes_runtime_env_mutation(monkeypatch):
    from rpu_backend import model_registry

    seen = []

    def reject_pair(w4_ckpt_dir, fp16_ckpt_dir):
        seen.append((str(w4_ckpt_dir), str(fp16_ckpt_dir)))
        raise ValueError("synthetic invalid W4 pair")

    monkeypatch.setattr(
        model_registry,
        "model_path",
        lambda name: Path(f"/models/{name}"),
    )
    monkeypatch.setattr(runtime, "validate_rtc_w4_checkpoint_pair", reject_pair)
    monkeypatch.setenv("RPU_LINEAR_ACC32", "1")
    monkeypatch.delenv("RPU_WALL_OSS_PREFILL_MIXED_PRECISION", raising=False)

    with pytest.raises(ValueError, match="synthetic invalid W4 pair"):
        runtime.build_wall_oss_vla(
            None,
            rtc=True,
            w4a16=True,
            rtc_w8_ckpt_dir="/models/wall-oss-0.5-w8a16",
        )

    assert seen == [(
        "/models/wall-oss-0.5-w4a16-pgrp",
        "/models/wall-oss-0.5",
    )]
    assert runtime.os.environ["RPU_LINEAR_ACC32"] == "1"
    assert "RPU_WALL_OSS_PREFILL_MIXED_PRECISION" not in runtime.os.environ


@pytest.mark.parametrize(
    ("weight_outer", "expected_weight_outer", "vwarp", "expected_vwarp"),
    [(None, "1", None, None), ("0", "0", "0", "0")],
)
@pytest.mark.parametrize("action_w4", [False, True])
def test_rtc_w8a16_keeps_only_the_overflow_boundaries_fp16(
    monkeypatch, weight_outer, expected_weight_outer, vwarp, expected_vwarp, action_w4,
):
    normalizer = {
        "min.dataset": torch.zeros(20, 26),
        "delta.dataset": torch.ones(20, 26),
    }
    config = SimpleNamespace(
        causal_action_attention_mask=False,
        noise_scheduler={"s": 0.999},
        vision_config=SimpleNamespace(tokens_per_second=2),
    )
    calls = []
    fake_llm = object()
    fake_action = SimpleNamespace(llm=fake_llm)
    if weight_outer is None:
        monkeypatch.delenv("RPU_WALL_OSS_VISION_WEIGHT_OUTER", raising=False)
    else:
        monkeypatch.setenv("RPU_WALL_OSS_VISION_WEIGHT_OUTER", weight_outer)
    if vwarp is None:
        monkeypatch.delenv("RPU_RMSNORM_VWARP", raising=False)
    else:
        monkeypatch.setenv("RPU_RMSNORM_VWARP", vwarp)
    monkeypatch.setattr(runtime.torch, "load", lambda *a, **k: normalizer)
    monkeypatch.setattr(
        runtime.AutoProcessor, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(
        runtime.AutoConfig, "from_pretrained", lambda *a, **k: config)
    monkeypatch.setattr(runtime, "_validate_rtc_training_metadata", lambda *_: None)
    monkeypatch.setattr(runtime, "_build_rope_helper", lambda *a, **k: object())
    monkeypatch.setattr(runtime, "_fused_denoise_enabled", lambda: True)
    monkeypatch.setattr(
        runtime, "validate_w8_nvfp4_checkpoint_pair",
        lambda *args: calls.append(("validate", args)),
    )
    monkeypatch.setattr(
        runtime, "build_wall_oss_llm",
        lambda *args, **kwargs: (
            calls.append(("llm", args, kwargs)), fake_llm)[1],
    )
    monkeypatch.setattr(
        runtime, "build_wall_oss_action",
        lambda *args, **kwargs: (
            calls.append(("action", args, kwargs)), fake_action)[1],
    )
    monkeypatch.setattr(
        runtime, "build_wall_oss_vision",
        lambda *args, **kwargs: (
            calls.append(("vision", args, kwargs)), object())[1],
    )
    monkeypatch.setattr(
        runtime, "WallOssVLA", lambda **kwargs: _RuntimeDict(kwargs)
    )

    pairs = []
    monkeypatch.setattr(
        runtime, "validate_rtc_w4_checkpoint_pair",
        lambda *args: pairs.append(args),
    )
    built = runtime.build_wall_oss_vla(
        "/models/v728-w8",
        fp16_ckpt_dir="/models/v728-fp16",
        dataset_key="dataset",
        rtc=True,
        w8a16=True,
        action_w4a16_ckpt_dir="/models/v728-w4" if action_w4 else None,
    )

    assert calls[0] == (
        "validate", ("/models/v728-w8", "/models/v728-fp16"))
    assert calls[1][0:2] == ("llm", ("/models/v728-w8",))
    assert {
        key: value
        for key, value in calls[1][2].items()
        if key != "execution_config"
    } == {
        "expert": 0,
        "max_seq_len": 2048,
        "w8a16": True,
        "fp16_ckpt_dir": "/models/v728-fp16",
        "rmsnorm_fixed32": True,
    }
    text_execution = calls[1][2]["execution_config"]
    assert text_execution == {"prefill": {"chunk_size": "auto"}}
    assert pairs == ([("/models/v728-w4", "/models/v728-fp16")] if action_w4 else [])
    assert calls[2][0:2] == (
        "action", ("/models/v728-w4" if action_w4 else "/models/v728-w8",)
    )
    assert calls[2][2]["llm"] is fake_llm
    assert calls[2][2]["w4a16" if action_w4 else "w8a16"] is True
    assert ("w8a16" if action_w4 else "w4a16") not in calls[2][2]
    assert calls[2][2]["fp16_ckpt_dir"] == "/models/v728-fp16"
    assert calls[2][2]["rtc_max_delay"] == 10
    assert calls[2][2]["execution_config"] == {
        "prefill": {"chunk_size": "auto"},
        "action": {"chunk_size": "auto"},
    }
    assert calls[3][0:2] == ("vision", ("/models/v728-w8",))
    assert {
        key: value
        for key, value in calls[3][2].items()
        if key != "execution_config"
    } == {
        "window": True,
        "w8a16": True,
        "fp16_ckpt_dir": "/models/v728-fp16",
        "num_cores": 8,
        "per_window_sdpa": False,
        "rmsnorm_fixed32": True,
    }
    assert calls[3][2]["execution_config"] == {
        "vision": {"chunk_size": "auto"}
    }
    assert built["execution_config"] == {}
    assert built["action"] is fake_action
    assert (
        runtime.os.environ["RPU_WALL_OSS_VISION_WEIGHT_OUTER"]
        == expected_weight_outer
    )
    assert runtime.os.environ.get("RPU_RMSNORM_VWARP") == expected_vwarp
    assert runtime.os.environ["RPU_WALL_OSS_VISION_MIXED_PRECISION"] == "1"
    assert runtime.os.environ["RPU_WALL_OSS_PREFILL_MIXED_PRECISION"] == "1"


def test_rtc_w4a16_routes_exact_three_checkpoint_composition(monkeypatch):
    normalizer = {
        "min.dataset": torch.zeros(20, 26),
        "delta.dataset": torch.ones(20, 26),
    }
    config = SimpleNamespace(
        causal_action_attention_mask=False,
        noise_scheduler={"s": 0.999},
        vision_config=SimpleNamespace(tokens_per_second=2),
    )
    calls = []
    fake_llm = object()
    fake_action = SimpleNamespace(llm=fake_llm)
    monkeypatch.setattr(runtime.torch, "load", lambda *a, **k: normalizer)
    monkeypatch.setattr(
        runtime.AutoProcessor, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(
        runtime.AutoConfig, "from_pretrained", lambda *a, **k: config)
    monkeypatch.setattr(runtime, "_validate_rtc_training_metadata", lambda *_: None)
    monkeypatch.setattr(runtime, "_build_rope_helper", lambda *a, **k: object())
    monkeypatch.setattr(runtime, "_fused_denoise_enabled", lambda: True)
    monkeypatch.setattr(
        runtime, "validate_rtc_w4_checkpoint_pair",
        lambda *args: calls.append(("validate_w4", args)),
    )
    monkeypatch.setattr(
        runtime, "validate_w8_nvfp4_checkpoint_pair",
        lambda *args: calls.append(("validate_w8", args)),
    )
    monkeypatch.setattr(
        runtime, "build_wall_oss_llm",
        lambda *args, **kwargs: (
            calls.append(("llm", args, kwargs)), fake_llm)[1],
    )
    monkeypatch.setattr(
        runtime, "build_wall_oss_action",
        lambda *args, **kwargs: (
            calls.append(("action", args, kwargs)), fake_action)[1],
    )
    monkeypatch.setattr(
        runtime, "build_wall_oss_vision",
        lambda *args, **kwargs: (
            calls.append(("vision", args, kwargs)), object())[1],
    )
    monkeypatch.setattr(
        runtime, "WallOssVLA", lambda **kwargs: SimpleNamespace(**kwargs))

    built = runtime.build_wall_oss_vla(
        "/models/v728-w4",
        fp16_ckpt_dir="/models/v728-fp16",
        rtc_w8_ckpt_dir="/models/v728-w8",
        dataset_key="dataset",
        rtc=True,
        w4a16=True,
    )

    assert calls[:2] == [
        ("validate_w4", ("/models/v728-w4", "/models/v728-fp16")),
        ("validate_w8", ("/models/v728-w8", "/models/v728-fp16")),
    ]
    assert calls[2][0:2] == ("llm", ("/models/v728-w4",))
    assert calls[2][2] == {
        "expert": 0,
        "max_seq_len": 2048,
        "fp16_ckpt_dir": "/models/v728-fp16",
        "execution_config": {"prefill": {"chunk_size": "auto"}},
        "w4a16": True,
    }
    assert calls[3][0:2] == ("action", ("/models/v728-w4",))
    assert calls[3][2]["llm"] is fake_llm
    assert calls[3][2]["w4a16"] is True
    assert calls[3][2]["execution_config"] == {
        "prefill": {"chunk_size": "auto"},
        "action": {"chunk_size": "auto"},
    }
    assert calls[4][0:2] == ("vision", ("/models/v728-w8",))
    assert calls[4][2] == {
        "window": True,
        "w8a16": True,
        "fp16_ckpt_dir": "/models/v728-fp16",
        "execution_config": {"vision": {"chunk_size": "auto"}},
        "num_cores": 8,
        "per_window_sdpa": False,
    }
    assert built.execution_config == {}
    assert built.action is fake_action
    assert runtime.os.environ["RPU_WALL_OSS_VISION_WEIGHT_OUTER"] == "0"
    assert runtime.os.environ["RPU_WALL_OSS_EXPERT0_DOWN_INT4"] == "0"


def test_rtc_builder_keeps_the_checkpoint_slow_image_processor():
    source = inspect.getsource(runtime.build_wall_oss_vla)
    assert '({"use_fast": False} if rtc else {})' in source


def test_vla_builder_cleans_completed_action_when_vision_fails(monkeypatch):
    _patch_vla_preflight(monkeypatch)
    events = []
    fake_llm = SimpleNamespace(close=lambda: events.append("llm_close"))

    class FakeAction:
        _owns_llm = True
        llm = fake_llm

        def _retire_native(self):
            events.append("action_close")

        def close(self):
            self._retire_native()
            self.llm.close()

    monkeypatch.setattr(runtime, "build_wall_oss_action", lambda *a, **k: FakeAction())
    monkeypatch.setattr(
        runtime, "build_wall_oss_vision",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("vision failed")),
    )

    with pytest.raises(RuntimeError, match="vision failed"):
        runtime.build_wall_oss_vla(
            "/checkpoint", fp16_ckpt_dir="/checkpoint", dataset_key="dataset"
        )

    assert events == ["action_close", "llm_close"]


def test_vla_builder_cleans_explicit_llm_when_action_fails(monkeypatch):
    _patch_vla_preflight(monkeypatch)
    monkeypatch.setattr(
        runtime, "validate_w8_nvfp4_checkpoint_pair", lambda *args: None
    )
    events = []
    fake_llm = SimpleNamespace(close=lambda: events.append("llm_close"))
    monkeypatch.setattr(runtime, "build_wall_oss_llm", lambda *a, **k: fake_llm)
    monkeypatch.setattr(
        runtime, "build_wall_oss_action",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("action failed")),
    )

    with pytest.raises(RuntimeError, match="action failed"):
        runtime.build_wall_oss_vla(
            "/checkpoint", fp16_ckpt_dir="/checkpoint", dataset_key="dataset",
            nvfp4_wint4_scope=True,
        )

    assert events == ["llm_close"]


def test_vla_close_releases_externally_shared_llm_once():
    events = []
    vla = runtime.WallOssVLA.__new__(runtime.WallOssVLA)
    vla.vision = SimpleNamespace(close=lambda: events.append("vision_close"))
    vla.action = SimpleNamespace(
        _owns_llm=False, close=lambda: events.append("action_close")
    )
    vla.llm = SimpleNamespace(close=lambda: events.append("llm_close"))
    vla._closed = False

    vla.close()
    vla.close()

    assert events == ["vision_close", "action_close", "llm_close"]


def test_vla_runtime_complete_requires_all_live_component_owners():
    class Finalizer:
        alive = True

    events = []
    llm_component = SimpleNamespace(
        _closed=False,
        _handle=11,
        _handle_finalizer=Finalizer(),
        _graph_cache=_Cache(events, "llm"),
        cache=object(),
    )
    action_component = SimpleNamespace(
        _closed=False,
        _handle=12,
        _handle_finalizer=Finalizer(),
        _graph_cache=_Cache(events, "action"),
        llm=llm_component,
    )
    vision_component = SimpleNamespace(
        _closed=False,
        _handle=13,
        _handle_finalizer=Finalizer(),
        _graph_cache=_Cache(events, "vision"),
        cache=object(),
        _keepalive=object(),
    )
    vla = runtime.WallOssVLA.__new__(runtime.WallOssVLA)
    vla._closed = False
    vla.vision = vision_component
    vla.action = action_component
    vla.llm = llm_component
    vla.action_dim = 26
    vla._processor = object()
    vla._rope = object()
    vla._normalizer = object()
    vla._normalizer_propri = object()
    for component in (llm_component, action_component, vision_component):
        component._retirement_owner = vla

    assert vla._runtime_complete() is True

    action_component._retirement_owner = object()
    assert vla._runtime_complete() is False
    action_component._retirement_owner = vla
    vision_component._closed = True
    assert vla._runtime_complete() is False
    vision_component._closed = False
    action_component.llm = object()
    assert vla._runtime_complete() is False


def test_vision_chunk_control_op_uses_default_dispatch_registration():
    """Integer-only control ops cannot be routed through PrivateUse1."""
    source = (_REPO_ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    direct = (
        'm.def("qwen25vl_vision_set_chunk_size(int handle, int chunk_size) '
        '-> ()",\n        TORCH_FN(rpu_qwen25vl_vision_set_chunk_size));'
    )
    assert direct in source
    assert 'm.impl("qwen25vl_vision_set_chunk_size"' not in source


def test_action_kv_owner_control_op_uses_default_dispatch_registration():
    """A handle plus bool has no Tensor from which to select PrivateUse1."""
    source = (_REPO_ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    direct = (
        'm.def("wall_oss_action_step_set_kv_owner_fanout(int handle, bool '
        'enabled) -> ()",\n'
        '        TORCH_FN(rpu_wall_oss_action_step_set_kv_owner_fanout));'
    )
    assert direct in source
    assert 'm.impl("wall_oss_action_step_set_kv_owner_fanout"' not in source


@pytest.mark.parametrize("flags", [{}, {"w8a16": True}, {"rtc": True}, {"rtc": True, "w4a16": True}])
def test_action_only_w4_rejects_non_rtc_w8_compositions_before_assets(monkeypatch, flags):
    with pytest.raises(ValueError, match="action_w4a16_ckpt_dir requires"):
        runtime.build_wall_oss_vla(
            "/must-not-be-read", action_w4a16_ckpt_dir="/must-not-be-read", **flags
        )


def test_action_only_w4_validates_pair_before_rtc_side_effects(monkeypatch):
    monkeypatch.delenv("RPU_WALL_OSS_PREFILL_MIXED_PRECISION", raising=False)
    monkeypatch.setattr(runtime, "validate_w8_nvfp4_checkpoint_pair", lambda *_: None)

    def reject_pair(*_):
        raise ValueError("invalid Action W4 pair")

    monkeypatch.setattr(runtime, "validate_rtc_w4_checkpoint_pair", reject_pair)
    with pytest.raises(ValueError, match="invalid Action W4 pair"):
        runtime.build_wall_oss_vla(
            "/models/v728-w8", fp16_ckpt_dir="/models/v728-fp16",
            action_w4a16_ckpt_dir="/models/invalid-w4", rtc=True, w8a16=True,
        )
    assert "RPU_WALL_OSS_PREFILL_MIXED_PRECISION" not in runtime.os.environ


@pytest.mark.parametrize("kind", ["vision", "prefill", "action", "unfused_action"])
def test_sorting_routes_are_bound_before_weights(native_ops, monkeypatch, kind):
    from rpu_backend.adapters.wall_oss._sorting import WallSortingConfig

    sorting = WallSortingConfig(fp16_mask=7, mlp_m224=True, silu_mul=True)
    calls = []
    name = {
        "vision": "qwen25vl_vision_set_wall_sorting_routes",
        "prefill": "causal_decoder_set_wall_sorting_routes",
        "action": "wall_oss_action_step_set_wall_sorting_routes",
        "unfused_action": "causal_decoder_set_wall_sorting_routes",
    }[kind]
    monkeypatch.setattr(torch.ops.rpu, name,
                        lambda _handle, *args: calls.append(("sorting", args)), raising=False)
    def weights(handle, *_):
        calls.append(("weights", ()))
        native_ops["step"]("vision_weights" if kind == "vision" else "weights", handle)

    if kind == "vision":
        handle, _ = vision._configure_qwen25vl_vision_handle(
            set_weights=weights, weight_args=(), freq_cos=object(), freq_sin=object(),
            merger_args=(), max_seq_len=2048, sorting=sorting,
        )
        destroy = torch.ops.rpu.qwen25vl_vision_destroy
    elif kind == "prefill":
        handle = llm._configure_causal_decoder_handle(
            set_weights=weights, weight_args=(), chunk_size_cap=384,
            equal_two_prefill=True, sorting=sorting,
        )
        destroy = torch.ops.rpu.causal_decoder_destroy
    else:
        handle = action._configure_action_handle(
            fused=kind == "action", set_weights=weights, weight_args=(), sorting=sorting,
        )
        destroy = (torch.ops.rpu.wall_oss_action_step_destroy if kind == "action"
                   else torch.ops.rpu.causal_decoder_destroy)
    try:
        assert calls == [
            ("sorting", sorting.vision_args() if kind == "vision" else sorting.decoder_args()),
            ("weights", ()),
        ]
    finally:
        destroy(handle)


@pytest.mark.parametrize("name,value", [
    ("RPU_WALL_OSS_FP16_TRANSFER_MASK", "8"),
    ("RPU_WALL_OSS_W16_REUSE_MASK", "1"),
    ("RPU_WALL_OSS_W16_NEXT3_MASK", "2"),
    ("RPU_WALL_OSS_W16_NEXT4_MASK", "4"),
])
def test_sorting_rejects_unshipped_routes_before_checkpoint_access(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        action.build_wall_oss_action("/must-not-be-read", fused=True)


def test_sorting_snapshot_keeps_its_cold_values(monkeypatch):
    from rpu_backend.adapters.wall_oss._sorting import sorting_config_from_env

    monkeypatch.setenv("RPU_WALL_OSS_FP16_TRANSFER_MASK", "7")
    monkeypatch.setenv("RPU_WALL_OSS_W16_REUSE_MASK", "8")
    monkeypatch.setenv("RPU_WALL_OSS_W16_NEXT3_MASK", "1")
    monkeypatch.setenv("RPU_WALL_OSS_W16_NEXT4_MASK", "3")
    snapshot = sorting_config_from_env()
    monkeypatch.setenv("RPU_WALL_OSS_FP16_TRANSFER_MASK", "0")
    monkeypatch.setenv("RPU_WALL_OSS_W16_NEXT4_MASK", "0")
    assert snapshot.fp16_mask == 7
    assert snapshot.action_reuse and snapshot.prefill_row
    assert snapshot.prefill_mlp and snapshot.prefill_paired


@pytest.mark.parametrize("failure", ["count", "upload", "setter"])
@pytest.mark.parametrize("destroy_fails", [False, True])
def test_action_residency_failure_preserves_pending_resources(monkeypatch, failure, destroy_fails):
    from rpu_backend.api import _execution
    from rpu_backend.adapters.wall_oss import _retirement
    from rpu_backend.adapters.wall_oss._sorting import WallSortingConfig

    cfg = {
        "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 128,
        "experts": [{}, {"hidden_size": 1024, "intermediate_size": 2048}],
        "rms_norm_eps": 1e-6, "rope_theta": 10000,
        "rope_scaling": {"mrope_section": [16, 24, 24]},
    }
    weight = torch.zeros(4, 2)
    store = SimpleNamespace(get_tensor=lambda _: weight)
    fused = {name: weight for name in ("w_comb", "w3", "proj_back", "w2_t", "w2_a")}
    fused.update(action_dim=26, k_comb_pad=16)
    monkeypatch.setattr(action, "_eff_attn_tp", lambda _, num_cores=8: 8)
    monkeypatch.setattr(action, "_split_qkv", lambda value, *_: (value, value, value))
    for name in ("_replicate_kv", "_to_rpu_half", "_to_rpu_int8", "_chan_first_scale",
                 "tp_col_swizzle_mc_weight", "tp_row_swizzle_mc_weight"):
        monkeypatch.setattr(action, name, lambda value, *_, **__: value)
    monkeypatch.setattr(action, "_build_rope_tables", lambda *_: (weight, weight))
    monkeypatch.setattr(action, "_pack_action_resident_weight_bundle",
                        lambda *_: torch.zeros(8, 4, dtype=torch.int8))
    captured = {}

    def configured(**kwargs):
        captured["weight_args"] = kwargs["weight_args"]
        return 17

    monkeypatch.setattr(action, "_configure_action_handle", configured)
    # The pending handle and actual CPU tensors exercise the real loader cleanup;
    # only native creation, layout work, and device transfer are replaced.
    original = RuntimeError(f"synthetic resident {failure} failure")
    tensor_to = torch.Tensor.to

    def transfer(tensor, *args, **kwargs):
        if kwargs.get("device") == "rpu":
            if failure == "upload":
                raise original
            return tensor.clone()
        return tensor_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", transfer)

    def set_residency(handle, bundle):
        captured["resident"] = bundle
        assert handle == 17
        raise original

    monkeypatch.setattr(torch.ops.rpu, "wall_oss_action_step_set_weights", lambda *_: None, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "wall_oss_action_step_set_weight_residency", set_residency, raising=False)
    destroys = []

    def destroy(handle, fused):
        destroys.append((handle, fused))
        if destroy_fails:
            raise RuntimeError("synthetic resident destroy failure")

    monkeypatch.setattr(action, "_destroy_action_handle", destroy)
    message = "failed to pack all Action resident weights" if failure == "count" else str(original)
    with pytest.raises(RuntimeError, match=message) as caught:
        action._load_expert1_decoder(
            store, cfg, 0 if failure == "count" else 5, 2048,
            w8a16=True, fused_action=fused, weight_residency=True,
            sorting=WallSortingConfig(),
        )
    if failure != "count":
        assert caught.value is original
    assert destroys == [(17, True)]
    if destroy_fails:
        assert "synthetic resident destroy failure" in caught.value.__notes__[0]
        assert _execution._UNSAFE_PROCESS_REASON is not None
        retained, = _retirement._FAILED_RETIREMENTS
        assert retained[0] == 17
        assert retained[1] is captured["weight_args"]
        assert len(retained[2]) == (0 if failure == "count" else 4)
        assert (retained[3] is None) == (failure == "count")
        assert retained[4] is captured.get("resident")
        with pytest.raises(RuntimeError, match="restart"):
            _execution._require_execution_process_safe()
        assert destroys == [(17, True)]
    else:
        assert _execution._UNSAFE_PROCESS_REASON is None
        assert not _retirement._FAILED_RETIREMENTS


@pytest.fixture
def action_boundary(monkeypatch):
    """Stop the real Action caller at its first device-input helper."""
    owner = action.WallOssAction.__new__(action.WallOssAction)
    owner.action_dim = 3
    owner._fused = True
    owner._action_fp32_tail = False
    owner._denoise_unroll = True
    owner._rtc_max_delay = 2
    owner.llm = SimpleNamespace(cache=SimpleNamespace(position=8, max_seq_len=64))
    events = []
    persistent = object()
    arena = {"temporary": 0, "persistent": persistent}

    def reset():
        events.append("reset")
        arena["temporary"] = 0

    def first_input(*_):
        events.append("action_input")
        assert arena["temporary"] == 0
        assert arena["persistent"] is persistent
        raise RuntimeError("stop at Action input")

    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", reset, raising=False)
    owner._action_position_ids_rpu = first_input
    return owner, events, arena


@pytest.mark.parametrize("mode", ["unfused", "stepwise", "unrolled", "rtc"])
def test_action_boundary_resets_once_before_device_inputs(action_boundary, mode):
    owner, events, arena = action_boundary
    owner._fused = mode != "unfused"
    owner._denoise_unroll = mode != "stepwise"
    kwargs = ({"use_rtc": True, "rtc_delay": 1,
               "rtc_prefix_actions": torch.zeros(1, 1, 3)} if mode == "rtc" else {})
    # Initial entry and a rebuild after Prefill left its peak in the shared arena.
    for temporary in (0, 7168000):
        events.clear()
        arena["temporary"] = temporary
        with pytest.raises(RuntimeError, match="stop at Action input"):
            owner.denoise_with_mask(
                8, torch.zeros(1, 2, 3), torch.ones(1, 2, 3),
                torch.zeros(2, 3, dtype=torch.int32), **kwargs)
        assert events == ["reset", "action_input"]


@pytest.mark.parametrize("failure", ["cache_position", "positions", "rtc_prefix"])
def test_action_boundary_rejected_input_does_not_reset(action_boundary, failure):
    owner, events, arena = action_boundary
    arena["temporary"] = 7168000
    positions = torch.zeros(2, 3, dtype=torch.int32)
    kwargs = {}
    if failure == "cache_position":
        owner.llm.cache.position = 7
    elif failure == "positions":
        positions = positions[:, :2]
    else:
        kwargs = {"use_rtc": True, "rtc_delay": 1,
                  "rtc_prefix_actions": torch.full((1, 1, 3), float("nan"))}
    with pytest.raises((ValueError, RuntimeError)):
        owner.denoise_with_mask(
            8, torch.zeros(1, 2, 3), torch.ones(1, 2, 3), positions, **kwargs)
    assert events == []
    assert arena["temporary"] == 7168000


@pytest.mark.parametrize("entry", ["predict", "prepare_graphs"])
def test_runtime_enters_action_with_fresh_temporary_arena(action_boundary, entry):
    owner, events, arena = action_boundary
    vla = runtime.WallOssVLA.__new__(runtime.WallOssVLA)
    vla.camera_names = ("face_view",)
    vla.dataset_key = "fixture"
    vla.delta_action = vla.rtc_enabled = False
    vla._prepared_graph_profile = None
    vla._ondevice_embed = True
    vla._proprioception_string = lambda *_: ""
    ids = torch.zeros(1, 8, dtype=torch.long)
    ids[0, 0] = runtime.IMAGE_PAD_ID
    vla._process_inputs = lambda *_: (ids, torch.zeros(4, 8), torch.tensor([[1, 2, 2]]))
    vla._validate_prepared_request = lambda **_: None
    vla._prefill_execution_plan = lambda *_, **__: (8, 8)
    vla._assemble_inputs_embeds_ondevice = lambda *_: torch.zeros(1, 8, 4)
    vla._rope_index_cached = lambda *_: (torch.zeros(3, 1, 8), torch.zeros(8, 3))
    graph_cache = SimpleNamespace(begin_warmup=lambda: None, clear=lambda: None)
    vla.vision = SimpleNamespace(
        forward=lambda *_: torch.zeros(1, 4), _graph_cache=graph_cache,
        _fused_merger=False)
    vla.llm = owner.llm
    vla.llm._graph_cache = owner._graph_cache = graph_cache
    vla.action = owner

    def prefill(*_, **__):
        events.append("prefill")
        arena["temporary"] = 7168000
        return torch.zeros(1, 4)

    vla.llm.forward_embeds = prefill
    kwargs = dict(images=[object()], instruction="instruction",
                  proprioception=torch.zeros(1), state_mask=torch.ones(1, dtype=torch.bool),
                  noise=torch.zeros(1, 2, 3), dof_mask=torch.ones(1, 2, 3))
    if entry == "prepare_graphs":
        kwargs["prefix_length_range"] = (8, 8)
    with pytest.raises(RuntimeError, match="stop at Action input"):
        if entry == "predict":
            vla.predict(kwargs.pop("images"), **kwargs)
        else:
            vla.prepare_graphs(**kwargs)
    assert events == ["reset", "reset", "prefill", "reset", "action_input"]


def test_standalone_action_predict_resets_after_prefill(action_boundary, monkeypatch):
    owner, events, arena = action_boundary
    owner._fused = False

    def prefill(*_, **__):
        events.append("prefill")
        arena["temporary"] = 7168000

    owner.llm.forward = prefill
    original_to = torch.Tensor.to

    def transfer(tensor, *args, **kwargs):
        if args == ("rpu",):
            owner._action_position_ids_rpu(tensor)
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", transfer)
    with pytest.raises(RuntimeError, match="stop at Action input"):
        owner.predict(torch.zeros(1, 8, dtype=torch.long),
                      torch.zeros(1, 2, 3), torch.ones(1, 2, 3))
    assert events == ["prefill", "reset", "action_input"]


@pytest.mark.parametrize("mixed,physical,logical_lengths,reserve_rows,canonical", [
    (True, 640, (599, 639, 640), 20, True),
    (True, 672, (641, 671), 20, True),
    (False, 640, (639, 640), 20, False),
    (False, 640, (545, 570, 639, 640), 32, True),
    (False, 672, (641, 671), 32, False),
])
def test_prefill_bucket_keeps_native_plan_authority(
    monkeypatch, mixed, physical, logical_lengths, reserve_rows, canonical
):
    from contextlib import contextmanager
    from rpu_backend.runtime.execution_planner import PreparedExecutionPlans, StageTuple

    owner = llm.WallOssLLM.__new__(llm.WallOssLLM)
    owner._handle = 7
    owner.hidden_size, owner.num_layers = 4, 1
    owner._rpu_execution = {}
    owner._prefill_chunk_size_cap = 384
    owner._prefill_pad16 = owner._prefill_equal_two = True
    owner._prefill_mixed_precision = mixed
    owner._attn_geometry = {}
    owner.cache = SimpleNamespace(max_seq_len=2048, k_caches=[], v_caches=[])
    owner.cache.reset_to_position = lambda p: setattr(owner.cache, "position", p)
    owner.cache.update_position = lambda p: setattr(owner.cache, "position", p)
    owner._prefill_position_ids_rpu = lambda pos, _: pos
    owner._prefill_cos_sin = lambda pos, _: (pos, pos)
    planner_calls = []
    real_plan = owner.prefill_execution_plan

    def plan(logical, *, reserve_rows, plan_result_sink):
        planner_calls.append((logical, reserve_rows))
        return real_plan(logical, reserve_rows=reserve_rows,
                         plan_result_sink=plan_result_sink)

    owner.prefill_execution_plan = plan
    monkeypatch.setattr(llm, "_to_rpu_half", lambda tensor: tensor)
    monkeypatch.setattr(llm, "_equal_two_chunk_supported", lambda *_: True)
    # Keep real A6 planning and signature construction; only native ABI/device
    # leaves are doubles. Include logical identity as the native manifest does.
    revision = [1]
    domains, descriptors, signatures = [], [], []

    def domain(handle, seq, position, causal, mask, rope, lifecycle, logical):
        domains.append((seq, logical))
        return [StageTuple(
            seq // 2, seq // 2, seq // 2,
            (("logical_length", logical), ("execution_padding_rows", seq - logical)),
            (seq, logical, revision[0]),
        )]

    @contextmanager
    def capture(signature):
        signatures.append(tuple(signature.dyn_dims))
        yield

    plans = PreparedExecutionPlans()
    owner._graph_cache = SimpleNamespace(
        is_frozen=lambda: False, prepare_plan=plans.prepare, capture=capture)
    monkeypatch.setattr(llm.rpu_backend.graph, "GraphSignature", SimpleNamespace)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_planner_cache_identity",
                        lambda _: (7, revision[0]), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_resolve_prefill_stage_domain",
                        domain, raising=False)

    def forward(*args):
        descriptors.append(tuple(args[-1]))
        return args[1]

    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_forward", forward, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_get_resolved_chunk_size",
                        lambda _: physical // 2, raising=False)

    def run(logical):
        owner.forward_embeds(torch.zeros(1, physical, 4), torch.zeros(physical, 3),
                             logical_len=logical, reserve_rows=reserve_rows)
        receipt = owner._rpu_last_execution_plan
        assert receipt["logical_len"] == logical
        assert receipt["padding_rows"] == physical - logical
        assert receipt["physical_descriptor"] == descriptors[-1]
        assert owner.cache.position == physical

    for logical in logical_lengths:
        run(logical)
    expected = 1 if canonical else len(logical_lengths)
    assert len(set(signatures)) == len(set(descriptors)) == expected
    assert planner_calls == [
        (physical if canonical else logical, reserve_rows)
        for logical in logical_lengths
    ]
    assert domains == ([(physical, physical)] if canonical else
                       [(physical, logical) for logical in logical_lengths])
    previous_signature = signatures[-1]
    revision[0] += 1
    run(logical_lengths[-1])
    assert signatures[-1] != previous_signature
    assert descriptors[-1][-1] == 2


@pytest.mark.parametrize("overrides", [
    {"_prefill_pad16": False},
    {"_prefill_equal_two": False},
    {"_rpu_execution": {"prefill": {"chunk_size": 320}}},
    {"_rpu_execution": {"prefill": {"padding_rows": 0}}},
    {"_rpu_execution": {"prefill": {"padding_rows": "auto"}}},
    {"_rpu_execution": {"prefill": {"padding_budget": 64}}},
    {"op_id": "wall_oss_llm"},
    {"logical_len": 544},
    {"reserve_rows": 20},
])
def test_base_p640_bucket_preserves_nondefault_planner_requests(overrides):
    owner = llm.WallOssLLM.__new__(llm.WallOssLLM)
    owner._rpu_execution = {}
    owner._prefill_mixed_precision = False
    owner._prefill_pad16 = owner._prefill_equal_two = True
    request = dict(op_id="wall_oss_llm_embeds", logical_len=570, reserve_rows=32)
    for name, value in overrides.items():
        if name.startswith("_"):
            setattr(owner, name, value)
        else:
            request[name] = value
    calls = []

    def plan(logical, *, reserve_rows, plan_result_sink):
        calls.append((logical, reserve_rows))
        raise RuntimeError("ordinary planner reached")

    owner.prefill_execution_plan = plan
    with pytest.raises(RuntimeError, match="ordinary planner reached"):
        owner._run_prefill(torch.zeros(1, 640, 4), torch.zeros(640, 3), **request)
    assert calls == [(request["logical_len"], request["reserve_rows"])]
