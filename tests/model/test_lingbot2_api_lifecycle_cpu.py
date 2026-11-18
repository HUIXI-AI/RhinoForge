"""Host-only concurrency contracts for the public LingBot2 policy facade.

The module is executed directly from its file path so this test never imports the
``rpu_backend`` package or loads the native backend.  Tiny stub modules provide the
two lazy imports exercised by ``Lingbot2Policy.to()``.
"""
from __future__ import annotations

import gc
import importlib.util
import os
import sys
import threading
import types
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


MODULE = (Path(__file__).resolve().parents[2] / "python" / "rpu_backend"
          / "api" / "lingbot2.py")
EXECUTION_MODULE = MODULE.with_name("_execution.py")
PROFILE_KEY = "RPU_LINGBOT2_EXPERT_W4A16"
GENERIC_PERF_DEFAULTS = {
    "RPU_FASTREPLAY_SKIP_SYNC": "1",
    "RPU_KVINSERT_HYBRID_V16": "1",
    "RPU_ADARMS_FUSED_BCAST": "1",
    "RPU_QWEN3VL_VISION_ROPE_SPM": "1",
    "RPU_QWEN3VL_VISION_BATCH": "1",
    "RPU_QWEN3VL_VISION_FUSED_MERGER": "1",
    "RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE": "1",
    "RPU_LINGBOT2_PREFILL_FAST_REPLAY": "1",
    "RPU_LINGBOT2_VISION_FAST_REPLAY": "1",
    "RPU_LINGBOT2_DENOISE_UNROLL": "1",
}
GROUPED_EXPERT_DEFAULTS = {
    "RPU_LINGBOT2_GROUPED_EXPERTS": "1",
    "RPU_L2_RCHUNK": "1632",
}
FP16_ONLY_DEFAULTS = {
    "RPU_WALL_OSS_FAST_REPLAY": "1",
}
FP16_PERF_DEFAULTS = {
    **GENERIC_PERF_DEFAULTS,
    **GROUPED_EXPERT_DEFAULTS,
    **FP16_ONLY_DEFAULTS,
}
FP16_CONTROLLED_ONLY = {
    "RPU_LINGBOT2_VISION_DIRECT_PREFIX": "0",
    "RPU_LINGBOT2_VISION_DIRECT_PREFIX_NO_OUTPUT": "0",
    "RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE": "0",
    "RPU_DEEP_FAST_REPLAY": "0",
    "RPU_KVINSERT_HYBRID3_V16": "0",
}

Z2_QUEUE_ENV = {
    "LKN_MAX_BATCH_ENTRIES": "262144",
    "LKN_KD_BUF_MB": "64",
    "LKN_INSTR_BUF_MB": "512",
}
Z2_COMMON_ENV = {
    **Z2_QUEUE_ENV,
    "RPU_LINGBOT2_DENOISE_UNROLL": "1",
    "RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE": "0",
    "RPU_LINGBOT2_EXPERT_REPLAY": "1",
    "RPU_LINGBOT2_FP16_TOP4": "1",
    "RPU_QWEN3VL_VISION_FUSED_MERGER": "1",
    "RPU_QWEN3VL_VISION_ROPE_SPM": "1",
    "RPU_WALL_OSS_FAST_REPLAY": "1",
    "RPU_FASTREPLAY_SKIP_SYNC": "1",
    "RPU_LINGBOT2_VISION_DIRECT_PREFIX": "0",
    "RPU_LINGBOT2_VISION_DIRECT_PREFIX_NO_OUTPUT": "0",
    "RPU_LINGBOT2_VISION_FAST_REPLAY": "0",
    "RPU_LINGBOT2_PREFILL_FAST_REPLAY": "0",
    "RPU_QWEN3VL_VISION_HOST_FP32_PATCH": "0",
    "RPU_QWEN3VL_VISION_BATCH": "0",
    "RPU_KVINSERT_HYBRID_V16": "0",
    "RPU_KVINSERT_HYBRID3_V16": "0",
    "RPU_ADARMS_FUSED_BCAST": "0",
}
Z2_DENSE_FP16_ENV = {
    "RPU_DEEP_FAST_REPLAY": "0",
    "RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE": "0",
    "RPU_LINGBOT2_GROUPED_EXPERTS": "0",
    "RPU_LINGBOT2_EXPERT_W8A16": "0",
    "RPU_LINGBOT2_EXPERT_W4A16": "0",
    "RPU_LINGBOT2_PREFILL_W8A16": "0",
    "RPU_LINGBOT2_PREFILL_W4A16": "0",
    "RPU_LINGBOT2_BASE_W8A16": "0",
    "RPU_LINGBOT2_VISION_W8A16": "0",
    "RPU_L2_RCHUNK": "0",
}
Z2_GROUPED_W8A16_ENV = {
    "RPU_DEEP_FAST_REPLAY": "1",
    "RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE": "1",
    "RPU_LINGBOT2_GROUPED_EXPERTS": "1",
    "RPU_LINGBOT2_EXPERT_W8A16": "1",
    "RPU_LINGBOT2_EXPERT_W4A16": "0",
    "RPU_LINGBOT2_PREFILL_W8A16": "1",
    "RPU_LINGBOT2_PREFILL_W4A16": "0",
    "RPU_LINGBOT2_BASE_W8A16": "1",
    "RPU_LINGBOT2_VISION_W8A16": "1",
    "RPU_L2_RCHUNK": "1632",
}
Z2_GROUPED_W4A16_ENV = {
    "RPU_DEEP_FAST_REPLAY": "1",
    "RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE": "1",
    "RPU_LINGBOT2_GROUPED_EXPERTS": "1",
    "RPU_LINGBOT2_EXPERT_W8A16": "0",
    "RPU_LINGBOT2_EXPERT_W4A16": "1",
    "RPU_LINGBOT2_PREFILL_W8A16": "1",
    "RPU_LINGBOT2_PREFILL_W4A16": "0",
    "RPU_LINGBOT2_BASE_W8A16": "1",
    "RPU_LINGBOT2_VISION_W8A16": "1",
    "RPU_L2_RCHUNK": "1632",
}


def _load_module(monkeypatch, builder):
    name = f"lingbot2_api_lifecycle_cpu_{id(builder)}"
    spec = importlib.util.spec_from_file_location(name, MODULE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)

    root = types.ModuleType("rpu_backend")
    root.__path__ = []
    adapters = types.ModuleType("rpu_backend.adapters")
    adapters.__path__ = []
    adapter = types.ModuleType("rpu_backend.adapters.lingbot_vla_v2")
    adapter_runtime = types.ModuleType(
        "rpu_backend.adapters.lingbot_vla_v2.runtime"
    )

    def validate_cold(config, *, multiview_spm_z2):
        fixed = {"action": 64}
        if multiview_spm_z2:
            fixed.update({"prefill": 240, "vision": 256})
        resolved = resolve_components(
            config, entry_point="test LingBot2 cold preflight"
        )
        views = {
            "vision": resolved[1],
            "prefill": resolved[2],
            "action": resolved[3],
        }
        for stage, expected in fixed.items():
            requested = views[stage].get(stage, {}).get(
                "chunk_size", "auto"
            )
            if requested != "auto" and int(requested) != expected:
                raise RuntimeError(f"{stage.upper()} fixed child mismatch")

    api = types.ModuleType("rpu_backend.api")
    api.__path__ = []
    action_denorm = types.ModuleType("rpu_backend.api.action_denorm")
    action_denorm.ActionDenormalizer = object
    errors = types.ModuleType("rpu_backend.api.errors")
    errors.RPUBackendError = type("RPUBackendError", (RuntimeError,), {})
    execution_spec = importlib.util.spec_from_file_location(
        "rpu_backend.api._execution", EXECUTION_MODULE
    )
    execution = importlib.util.module_from_spec(execution_spec)
    execution_spec.loader.exec_module(execution)

    vision_component = "vision_encoder"
    text_component = "language_model"
    action_component = "action_expert"
    components = {
        vision_component: {"vision": ("chunk_size",)},
        text_component: {
            "prefill": ("chunk_size", "padding_rows", "padding_budget"),
        },
        action_component: {"action": ("chunk_size",)},
    }

    def resolve_components(value, *, entry_point):
        root_config = execution.normalize_rpu_execution(
            value, entry_point=entry_point, supported_components=components
        )
        views = tuple(
            execution.resolve_component_rpu_execution(
                root_config,
                component,
                entry_point=entry_point,
                supported_components=components,
                profile_auto=profile,
            )
            for component, profile in (
                (vision_component, {"vision": {"chunk_size": "auto"}}),
                (text_component, {"prefill": {
                    "chunk_size": "auto", "padding_rows": "auto",
                    "padding_budget": 15,
                }}),
                (action_component, {"action": {"chunk_size": "auto"}}),
            )
        )
        return (root_config, *views)

    def build_with_inherited_session(*args, **kwargs):
        result = builder(*args, **kwargs)
        session = kwargs.get("_execution_session")
        if session is not None:
            resolved = resolve_components(
                kwargs["rpu_execution"], entry_point="fake LingBot2 runtime"
            )
            result._execution_session = weakref.proxy(session)
            result._execution_generation = kwargs["_execution_generation"]
            result._component_generations = dict(
                kwargs["_component_generations"]
            )
            (
                result._rpu_execution,
                result._vision_execution,
                result._text_execution,
                result._action_execution,
            ) = resolved
            result._execution_components = dict(zip(
                components, resolved[1:]
            ))
        return result

    adapter.build_lingbot_vla_v2 = build_with_inherited_session
    adapter_runtime.LINGBOT2_VISION_COMPONENT = vision_component
    adapter_runtime.LINGBOT2_TEXT_COMPONENT = text_component
    adapter_runtime.LINGBOT2_ACTION_COMPONENT = action_component
    adapter_runtime.LINGBOT2_EXECUTION_COMPONENTS = components
    adapter_runtime.resolve_lingbot2_execution_components = resolve_components
    adapter_runtime._validate_lingbot2_cold_execution_config = validate_cold
    for module_name, stub in (
        ("rpu_backend", root),
        ("rpu_backend.adapters", adapters),
        ("rpu_backend.adapters.lingbot_vla_v2", adapter),
        ("rpu_backend.adapters.lingbot_vla_v2.runtime", adapter_runtime),
        ("rpu_backend.api", api),
        ("rpu_backend.api.action_denorm", action_denorm),
        ("rpu_backend.api.errors", errors),
        ("rpu_backend.api._execution", execution),
    ):
        monkeypatch.setitem(sys.modules, module_name, stub)
    return module


class _HandleFinalizer:
    def __init__(self):
        self.alive = True


class _FakeRuntime:
    def __init__(self, *, get_action=None, close_started=None, close_release=None):
        self._closed = False
        self._exp = object()
        self._text = object()
        self._visual = object()
        self._handle_finalizer = _HandleFinalizer()
        self._gc_retirement_enabled = True
        self._cleanup_ok = None
        self._rpu_swizzled = True
        self._get_action = get_action
        self._close_started = close_started
        self._close_release = close_release

    def get_action(self, obs):
        if self._get_action is not None:
            return self._get_action(obs)
        return torch.zeros(1, 50, 55)

    def close(self):
        if self._closed:
            return
        if self._close_started is not None:
            self._close_started.set()
        if self._close_release is not None:
            assert self._close_release.wait(5), "timed out waiting to finish close"
        self._closed = True
        self._handle_finalizer.alive = False
        self._cleanup_ok = True

    _close_without_session = close

    def _gc_retire(self):
        self.close()
        return True


def _policy(api, dtype):
    return api.Lingbot2Policy.from_checkpoint(
        "/unused",
        dtype=dtype,
    )


def _join(thread):
    thread.join(5)
    assert not thread.is_alive(), "policy lifecycle thread deadlocked"


def test_rpu_execution_is_cold_normalized_and_forwarded(monkeypatch):
    captured = {}

    def builder(*_args, **kwargs):
        captured.update(kwargs)
        return _FakeRuntime()

    api = _load_module(monkeypatch, builder)
    policy = api.Lingbot2Policy.from_checkpoint(
        "/unused",
        rpu_execution={
            "prefill": {"chunk_size": 240, "padding_budget": 10},
            "vision": {"chunk_size": 256},
            "action": {"chunk_size": 64},
        },
    )
    assert dict(policy._rpu_execution["prefill"]) == {
        "chunk_size": 240,
        "padding_budget": 10,
    }
    with pytest.raises(TypeError, match="read-only"):
        policy._rpu_execution["prefill"].chunk_size = 128

    policy.to("rpu")
    assert captured["rpu_execution"] is policy._rpu_execution
    policy.close()


def test_rpu_execution_hot_reconfigure_uses_runtime_generation(monkeypatch):
    calls = []

    def builder(*_args, **_kwargs):
        runtime = _FakeRuntime()
        runtime.validate_execution_reconfigure = lambda config: calls.append(
            ("validate", config)
        )
        runtime.apply_execution_reconfigure = lambda config, generation: calls.append(
            ("apply", config, generation)
        )
        return runtime

    api = _load_module(monkeypatch, builder)
    policy = _policy(api, "fp16").to("rpu")

    result = policy._execution_session.reconfigure({
        "prefill": {"chunk_size": 240},
        "vision": {"chunk_size": 256},
        "action": {"chunk_size": 64},
    })

    assert result is policy._rpu_execution
    assert policy._execution_session.generation == 1
    assert [call[0] for call in calls] == ["validate", "apply"]
    assert calls[-1][-1] == 1
    policy.close()


def test_rpu_execution_reconfigure_before_build_seeds_runtime_generation(monkeypatch):
    captured = {}

    def builder(*_args, **kwargs):
        captured.update(kwargs)
        return _FakeRuntime()

    api = _load_module(monkeypatch, builder)
    policy = _policy(api, "fp16")
    policy._execution_session.reconfigure({
        "prefill": {"chunk_size": 240}
    })
    policy.to("rpu")

    assert captured["rpu_execution"] == {
        "prefill": {"chunk_size": 240}
    }
    assert policy._policy._execution_generation == 1
    assert (
        policy._policy._execution_session.generation
        == policy._execution_session.generation
    )
    policy.close()


def test_rpu_execution_hot_failure_reapplies_old_runtime_config(monkeypatch):
    calls = []

    def builder(*_args, **_kwargs):
        runtime = _FakeRuntime()
        runtime.validate_execution_reconfigure = lambda _config: None

        def apply(config, generation):
            calls.append((config, generation))
            if config.get("prefill", {}).get("chunk_size") == 240:
                raise RuntimeError("native refresh failed")

        runtime.apply_execution_reconfigure = apply
        runtime.rollback_execution_reconfigure = (
            lambda config, generation: calls.append((config, generation))
        )
        return runtime

    api = _load_module(monkeypatch, builder)
    policy = _policy(api, "fp16").to("rpu")
    old = policy._rpu_execution

    with pytest.raises(RuntimeError, match="native refresh failed"):
        policy._execution_session.reconfigure({
            "prefill": {"chunk_size": 240}
        })

    assert policy._rpu_execution is old
    assert policy._execution_session.generation == 0
    assert calls[-1] == (old, 0)
    policy.close()


def test_rpu_execution_rejects_unsupported_fields_before_build(monkeypatch):
    calls = []
    api = _load_module(monkeypatch, lambda *_args, **_kwargs: calls.append(1))

    with pytest.raises(ValueError, match="not supported"):
        api.Lingbot2Policy.from_checkpoint(
            "/unused",
            rpu_execution={"vision": {"padding_rows": 0}},
        )
    assert calls == []


def test_fixed_action_config_rejects_before_facade_builder(monkeypatch):
    calls = []
    api = _load_module(monkeypatch, lambda *_args, **_kwargs: calls.append(1))

    with pytest.raises(RuntimeError, match="ACTION"):
        api.Lingbot2Policy.from_checkpoint(
            "/unused",
            rpu_execution={"action": {"chunk_size": 32}},
        )
    assert calls == []


def test_camera_frame_normalizer_accepts_hwc_numpy_and_tensor(monkeypatch):
    api = _load_module(monkeypatch, lambda *_args, **_kwargs: _FakeRuntime())
    frame = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

    from_numpy = api.Lingbot2Policy._as_rgb_uint8_array(frame)
    from_tensor = api.Lingbot2Policy._as_rgb_uint8_array(torch.from_numpy(frame))

    assert np.array_equal(from_numpy, frame)
    assert np.array_equal(from_tensor, frame)
    assert from_numpy.flags.c_contiguous
    assert from_tensor.flags.c_contiguous


def test_camera_frame_normalizer_rejects_chw_tensor(monkeypatch):
    api = _load_module(monkeypatch, lambda *_args, **_kwargs: _FakeRuntime())

    with pytest.raises(ValueError, match="expected an HWC RGB frame"):
        api.Lingbot2Policy._as_rgb_uint8_array(torch.zeros(3, 4, 5))


@pytest.mark.parametrize(
    "frame",
    (
        np.full((4, 5, 3), 0.5, dtype=np.float32),
        torch.full((4, 5, 3), 256, dtype=torch.uint16),
    ),
)
def test_camera_frame_normalizer_rejects_non_uint8(monkeypatch, frame):
    api = _load_module(monkeypatch, lambda *_args, **_kwargs: _FakeRuntime())

    with pytest.raises(ValueError, match="expected an HWC uint8 RGB frame"):
        api.Lingbot2Policy._as_rgb_uint8_array(frame)


@pytest.mark.parametrize("mode", ("exact", "fast", "legacy"))
def test_camera_input_types_encode_identically(monkeypatch, mode):
    api = _load_module(monkeypatch, lambda *_args, **_kwargs: _FakeRuntime())

    class FakeImageProcessor:
        def __call__(self, *, images, return_tensors):
            assert return_tensors == "pt"
            values = [
                torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).reshape(-1, 3)
                for image in images
            ]
            return {
                "pixel_values": torch.stack(values),
                "image_grid_thw": torch.tensor([[1, 2, 2]] * len(values)),
            }

    policy = object.__new__(api.Lingbot2Policy)
    policy._n_cameras = 1
    policy._image_size = 4
    policy._preproc_mode = mode
    policy._processor = FakeImageProcessor()
    frame = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)

    pil_encoded = policy._encode_images([Image.fromarray(frame)])
    numpy_encoded = policy._encode_images([frame])
    tensor_encoded = policy._encode_images([torch.from_numpy(frame)])

    assert torch.equal(pil_encoded[0], numpy_encoded[0])
    assert torch.equal(pil_encoded[0], tensor_encoded[0])
    assert torch.equal(pil_encoded[1], numpy_encoded[1])
    assert torch.equal(pil_encoded[1], tensor_encoded[1])


def test_rejected_second_build_cannot_change_live_infer_environment(monkeypatch):
    monkeypatch.delenv(PROFILE_KEY, raising=False)
    infer_entered = threading.Event()
    allow_infer = threading.Event()
    build_attempted = threading.Event()
    second_builder_entered = threading.Event()
    seen = {}
    build_count = 0
    build_count_lock = threading.Lock()

    def get_action(_obs):
        infer_entered.set()
        assert allow_infer.wait(5), "timed out waiting to finish infer"
        seen["during_infer"] = os.environ.get(PROFILE_KEY)
        return torch.zeros(1, 50, 55)

    first_runtime = _FakeRuntime(get_action=get_action)

    def builder(*_args, **_kwargs):
        nonlocal build_count
        with build_count_lock:
            build_count += 1
            call = build_count
        if call == 1:
            return first_runtime
        second_builder_entered.set()
        raise RuntimeError("synthetic live-slot rejection")

    api = _load_module(monkeypatch, builder)
    first = _policy(api, "w4a16").to("rpu")
    second = _policy(api, "fp16")
    assert os.environ.get(PROFILE_KEY) == "1"

    infer_errors = []
    build_errors = []
    infer_thread = threading.Thread(
        target=lambda: _capture(infer_errors, first.infer,
                                obs={"state": torch.zeros(1, 55)})
    )
    def build_second():
        build_attempted.set()
        _capture(build_errors, second.to, "rpu")

    build_thread = threading.Thread(target=build_second)
    infer_thread.start()
    assert infer_entered.wait(5)
    build_thread.start()
    assert build_attempted.wait(5)

    # The second facade must be blocked before it applies its fp16 profile or
    # reaches the builder while the first facade owns the runtime lock.
    assert not second_builder_entered.wait(0.2)
    assert os.environ.get(PROFILE_KEY) == "1"
    allow_infer.set()
    _join(infer_thread)
    _join(build_thread)

    assert infer_errors == []
    assert len(build_errors) == 1
    assert str(build_errors[0]) == "synthetic live-slot rejection"
    assert seen == {"during_infer": "1"}
    assert os.environ.get(PROFILE_KEY) == "1"
    first.close()
    second.close()
    assert PROFILE_KEY not in os.environ


def _capture(errors, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except BaseException as exc:  # thread boundary: report the exact failure to the test
        errors.append(exc)


def test_fp16_perf_defaults_are_profile_scoped_and_restored(monkeypatch):
    keys = {**FP16_PERF_DEFAULTS, **FP16_CONTROLLED_ONLY}
    for key in keys:
        monkeypatch.setenv(key, "caller-value")

    seen = {}

    def builder(*_args, **_kwargs):
        seen.update({key: os.environ.get(key) for key in keys})
        return _FakeRuntime()

    api = _load_module(monkeypatch, builder)
    policy = _policy(api, "fp16").to("rpu")
    assert seen == keys
    assert {key: os.environ.get(key) for key in keys} == keys

    policy.close()
    assert {key: os.environ.get(key) for key in keys} == {
        key: "caller-value" for key in keys
    }


def test_runtime_env_can_opt_out_of_fp16_perf_default(monkeypatch):
    seen = {}

    def builder(*_args, **_kwargs):
        seen["fast_replay"] = os.environ.get("RPU_WALL_OSS_FAST_REPLAY")
        seen["grouped_experts"] = os.environ.get(
            "RPU_LINGBOT2_GROUPED_EXPERTS")
        seen["rchunk"] = os.environ.get("RPU_L2_RCHUNK")
        seen["unroll"] = os.environ.get("RPU_LINGBOT2_DENOISE_UNROLL")
        seen["device_patch"] = os.environ.get(
            "RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE")
        return _FakeRuntime()

    api = _load_module(monkeypatch, builder)
    policy = api.Lingbot2Policy.from_checkpoint(
        "/unused",
        dtype="fp16",
        runtime_env={
            "RPU_WALL_OSS_FAST_REPLAY": "0",
            "RPU_LINGBOT2_GROUPED_EXPERTS": "0",
            "RPU_L2_RCHUNK": "0",
            "RPU_LINGBOT2_DENOISE_UNROLL": "0",
            "RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE": "0",
        },
    ).to("rpu")
    assert seen == {
        "fast_replay": "0",
        "grouped_experts": "0",
        "rchunk": "0",
        "unroll": "0",
        "device_patch": "0",
    }
    policy.close()


def test_denoise_unroll_scopes_queue_defaults_and_forces_indexed_schedule(
    monkeypatch,
):
    capacity_defaults = {
        "LKN_MAX_BATCH_ENTRIES": "131072",
        "LKN_KD_BUF_MB": "32",
        "LKN_INSTR_BUF_MB": "128",
    }
    keys = {
        "RPU_LINGBOT2_DENOISE_UNROLL",
        "RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE",
        *capacity_defaults,
    }
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    seen = {}

    def builder(*_args, **_kwargs):
        seen.update({key: os.environ.get(key) for key in keys})
        return _FakeRuntime()

    api = _load_module(monkeypatch, builder)
    policy = api.Lingbot2Policy.from_checkpoint(
        "/unused",
        runtime_env={
            "RPU_LINGBOT2_DENOISE_UNROLL": "1",
            # The unrolled path must use its indexed schedule even if a
            # controlled caller also requested the old direct-DMA experiment.
            "RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE": "1",
        },
    ).to("rpu")

    assert seen == {
        "RPU_LINGBOT2_DENOISE_UNROLL": "1",
        "RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE": "0",
        **capacity_defaults,
    }
    assert {key: os.environ.get(key) for key in keys} == seen

    policy.close()
    assert all(key not in os.environ for key in keys)


def test_dense_fp16_opt_out_uses_larger_unroll_capacity(monkeypatch):
    capacity_defaults = {
        "LKN_MAX_BATCH_ENTRIES": "262144",
        "LKN_KD_BUF_MB": "64",
        "LKN_INSTR_BUF_MB": "512",
    }
    for key in capacity_defaults:
        monkeypatch.delenv(key, raising=False)

    seen = {}

    def builder(*_args, **_kwargs):
        seen.update({key: os.environ.get(key) for key in capacity_defaults})
        return _FakeRuntime()

    api = _load_module(monkeypatch, builder)
    policy = api.Lingbot2Policy.from_checkpoint(
        "/unused",
        dtype="fp16",
        runtime_env={
            "RPU_LINGBOT2_GROUPED_EXPERTS": "0",
        },
    ).to("rpu")

    assert seen == capacity_defaults
    policy.close()
    assert all(key not in os.environ for key in capacity_defaults)


def test_multiview_z2_scopes_the_dense_fp16_bundle(monkeypatch):
    expected = {**Z2_COMMON_ENV, **Z2_DENSE_FP16_ENV}
    for key in expected:
        monkeypatch.delenv(key, raising=False)
    seen = {}

    def builder(*_args, **_kwargs):
        seen.update({key: os.environ.get(key) for key in expected})
        return _FakeRuntime()

    api = _load_module(monkeypatch, builder)
    policy = api.Lingbot2Policy.from_checkpoint(
        "/unused",
        dtype="fp16",
        n_cameras=3,
        image_size=256,
        runtime_env={
            "RPU_LINGBOT2_DENOISE_UNROLL": "1",
            "RPU_LINGBOT2_MULTIVIEW_SPM_Z2": "1",
        },
    ).to("rpu")

    assert seen == expected
    policy.close()
    assert all(key not in os.environ for key in expected)


def test_multiview_z2_scopes_the_grouped_w8a16_bundle(monkeypatch):
    expected = {**Z2_COMMON_ENV, **Z2_GROUPED_W8A16_ENV}
    for key in expected:
        monkeypatch.delenv(key, raising=False)
    seen = {}

    def builder(*_args, **_kwargs):
        seen.update({key: os.environ.get(key) for key in expected})
        return _FakeRuntime()

    api = _load_module(monkeypatch, builder)
    policy = api.Lingbot2Policy.from_checkpoint(
        "/unused",
        dtype="w8a16",
        n_cameras=3,
        image_size=256,
        runtime_env={
            "RPU_LINGBOT2_DENOISE_UNROLL": "1",
            "RPU_LINGBOT2_MULTIVIEW_SPM_Z2": "1",
        },
    ).to("rpu")

    assert seen == expected
    policy.close()
    assert all(key not in os.environ for key in expected)


def test_multiview_z2_scopes_the_grouped_w4a16_bundle(monkeypatch):
    expected = {**Z2_COMMON_ENV, **Z2_GROUPED_W4A16_ENV}
    for key in expected:
        monkeypatch.delenv(key, raising=False)
    seen = {}

    def builder(*_args, **_kwargs):
        seen.update({key: os.environ.get(key) for key in expected})
        return _FakeRuntime()

    api = _load_module(monkeypatch, builder)
    policy = api.Lingbot2Policy.from_checkpoint(
        "/unused",
        dtype="w4a16",
        n_cameras=3,
        image_size=256,
        runtime_env={
            "RPU_LINGBOT2_DENOISE_UNROLL": "1",
            "RPU_LINGBOT2_MULTIVIEW_SPM_Z2": "1",
        },
    ).to("rpu")

    assert seen == expected
    policy.close()
    assert all(key not in os.environ for key in expected)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("RPU_LINGBOT2_GROUPED_EXPERTS", "0"),
        ("RPU_LINGBOT2_EXPERT_W8A16", "0"),
        ("RPU_LINGBOT2_EXPERT_W4A16", "1"),
        ("RPU_LINGBOT2_PREFILL_W8A16", "0"),
        ("RPU_LINGBOT2_PREFILL_W4A16", "1"),
        ("RPU_LINGBOT2_BASE_W8A16", "0"),
        ("RPU_LINGBOT2_VISION_W8A16", "0"),
        ("RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE", "0"),
        ("RPU_DEEP_FAST_REPLAY", "0"),
        ("RPU_L2_RCHUNK", "1024"),
    ),
)
def test_multiview_z2_rejects_partial_or_w4_w8_bundle(
    monkeypatch, name, value,
):
    calls = []

    def builder(*_args, **_kwargs):
        calls.append(True)
        return _FakeRuntime()

    api = _load_module(monkeypatch, builder)
    policy = api.Lingbot2Policy.from_checkpoint(
        "/unused",
        dtype="w8a16",
        n_cameras=3,
        image_size=256,
        runtime_env={
            "RPU_LINGBOT2_DENOISE_UNROLL": "1",
            "RPU_LINGBOT2_MULTIVIEW_SPM_Z2": "1",
            name: value,
        },
    )

    with pytest.raises(ValueError, match="sealed-profile runtime_env"):
        policy.to("rpu")
    assert calls == []


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("RPU_LINGBOT2_GROUPED_EXPERTS", "0"),
        ("RPU_LINGBOT2_EXPERT_W8A16", "1"),
        ("RPU_LINGBOT2_EXPERT_W4A16", "0"),
        ("RPU_LINGBOT2_PREFILL_W8A16", "0"),
        ("RPU_LINGBOT2_PREFILL_W4A16", "1"),
        ("RPU_LINGBOT2_BASE_W8A16", "0"),
        ("RPU_LINGBOT2_VISION_W8A16", "0"),
        ("RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE", "0"),
        ("RPU_DEEP_FAST_REPLAY", "0"),
        ("RPU_L2_RCHUNK", "1024"),
    ),
)
def test_multiview_z2_rejects_partial_w4_bundle(monkeypatch, name, value):
    calls = []

    def builder(*_args, **_kwargs):
        calls.append(True)
        return _FakeRuntime()

    api = _load_module(monkeypatch, builder)
    policy = api.Lingbot2Policy.from_checkpoint(
        "/unused",
        dtype="w4a16",
        n_cameras=3,
        image_size=256,
        runtime_env={
            "RPU_LINGBOT2_DENOISE_UNROLL": "1",
            "RPU_LINGBOT2_MULTIVIEW_SPM_Z2": "1",
            name: value,
        },
    )

    with pytest.raises(ValueError, match="sealed-profile runtime_env"):
        policy.to("rpu")
    assert calls == []


@pytest.mark.parametrize(
    "name",
    (
        "RPU_DEEP_FAST_REPLAY",
        "RPU_LINGBOT2_VISION_FAST_REPLAY",
        "RPU_LINGBOT2_PREFILL_FAST_REPLAY",
        "RPU_QWEN3VL_VISION_BATCH",
        "RPU_KVINSERT_HYBRID_V16",
        "RPU_KVINSERT_HYBRID3_V16",
        "RPU_ADARMS_FUSED_BCAST",
    ),
)
def test_multiview_z2_rejects_stage2_switches(monkeypatch, name):
    calls = []

    def builder(*_args, **_kwargs):
        calls.append(True)
        return _FakeRuntime()

    api = _load_module(monkeypatch, builder)
    policy = api.Lingbot2Policy.from_checkpoint(
        "/unused",
        dtype="fp16",
        n_cameras=3,
        image_size=256,
        runtime_env={
            "RPU_LINGBOT2_DENOISE_UNROLL": "1",
            "RPU_LINGBOT2_MULTIVIEW_SPM_Z2": "1",
            name: "1",
        },
    )

    with pytest.raises(ValueError, match="sealed-profile runtime_env"):
        policy.to("rpu")
    assert calls == []


def test_multiview_z2_rejects_the_obsolete_131072_entry_override(monkeypatch):
    calls = []

    def builder(*_args, **_kwargs):
        calls.append(True)
        return _FakeRuntime()

    api = _load_module(monkeypatch, builder)
    policy = api.Lingbot2Policy.from_checkpoint(
        "/unused",
        dtype="fp16",
        n_cameras=3,
        image_size=256,
        runtime_env={
            "RPU_LINGBOT2_DENOISE_UNROLL": "1",
            "RPU_LINGBOT2_MULTIVIEW_SPM_Z2": "1",
            "LKN_MAX_BATCH_ENTRIES": "131072",
        },
    )
    with pytest.raises(ValueError, match="sealed-profile runtime_env"):
        policy.to("rpu")
    assert calls == []


def test_denoise_unroll_preserves_explicit_queue_limits_and_restores_on_failure(
    monkeypatch,
):
    capacity = {
        "LKN_MAX_BATCH_ENTRIES": "65536",
        "LKN_KD_BUF_MB": "16",
        "LKN_INSTR_BUF_MB": "64",
    }
    for key, value in capacity.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE", "1")

    def builder(*_args, **_kwargs):
        assert os.environ["RPU_LINGBOT2_DENOISE_UNROLL"] == "1"
        assert os.environ["RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE"] == "0"
        assert os.environ["LKN_MAX_BATCH_ENTRIES"] == "262144"
        assert os.environ["LKN_KD_BUF_MB"] == "16"
        assert os.environ["LKN_INSTR_BUF_MB"] == "64"
        raise RuntimeError("synthetic unroll build failure")

    api = _load_module(monkeypatch, builder)
    policy = api.Lingbot2Policy.from_checkpoint(
        "/unused",
        runtime_env={
            "RPU_LINGBOT2_DENOISE_UNROLL": "1",
            "LKN_MAX_BATCH_ENTRIES": "262144",
        },
    )
    with pytest.raises(RuntimeError, match="synthetic unroll build failure"):
        policy.to("rpu")

    assert {key: os.environ.get(key) for key in capacity} == capacity
    assert os.environ["RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE"] == "1"
    assert "RPU_LINGBOT2_DENOISE_UNROLL" not in os.environ
    assert policy._env_snapshot is None


def test_denoise_unroll_builds_without_the_retired_controlled_path_opt_in(monkeypatch):
    # RPU_LINGBOT2_ALLOW_UNVALIDATED was retired (2026-10-19): .to("rpu") needs
    # no opt-in, and the unroll selector alone still reaches the real builder.
    for key in (
        "RPU_LINGBOT2_DENOISE_UNROLL",
        "RPU_LINGBOT2_ALLOW_UNVALIDATED",
        "LKN_MAX_BATCH_ENTRIES",
        "LKN_KD_BUF_MB",
        "LKN_INSTR_BUF_MB",
    ):
        monkeypatch.delenv(key, raising=False)
    calls = []

    def builder(*_args, **_kwargs):
        calls.append(True)
        return _FakeRuntime()

    api = _load_module(monkeypatch, builder)
    policy = api.Lingbot2Policy.from_checkpoint(
        "/unused",
        runtime_env={"RPU_LINGBOT2_DENOISE_UNROLL": "1"},
    )
    policy.to("rpu")

    assert calls == [True]
    policy.close()
    assert "RPU_LINGBOT2_DENOISE_UNROLL" not in os.environ


def test_denoise_unroll_requires_exact_runtime_env_one(monkeypatch):
    for key in (
        "RPU_LINGBOT2_DENOISE_UNROLL",
        "LKN_MAX_BATCH_ENTRIES",
        "LKN_KD_BUF_MB",
        "LKN_INSTR_BUF_MB",
    ):
        monkeypatch.delenv(key, raising=False)
    seen = {}

    def builder(*_args, **_kwargs):
        seen["unroll"] = os.environ.get("RPU_LINGBOT2_DENOISE_UNROLL")
        seen["capacity"] = {
            key: os.environ.get(key)
            for key in (
                "LKN_MAX_BATCH_ENTRIES",
                "LKN_KD_BUF_MB",
                "LKN_INSTR_BUF_MB",
            )
        }
        return _FakeRuntime()

    api = _load_module(monkeypatch, builder)
    policy = api.Lingbot2Policy.from_checkpoint(
        "/unused",
        runtime_env={
            "RPU_LINGBOT2_DENOISE_UNROLL": "true",
        },
    ).to("rpu")

    assert seen == {
        "unroll": "0",
        "capacity": {
            "LKN_MAX_BATCH_ENTRIES": None,
            "LKN_KD_BUF_MB": None,
            "LKN_INSTR_BUF_MB": None,
        },
    }
    policy.close()


@pytest.mark.parametrize("dtype", ("w8a16", "w4a16"))
def test_generic_defaults_apply_without_fp16_only_flags(monkeypatch, dtype):
    seen = {}

    def builder(*_args, **_kwargs):
        seen.update({
            key: os.environ.get(key)
            for key in (
                *GENERIC_PERF_DEFAULTS,
                *GROUPED_EXPERT_DEFAULTS,
                *FP16_ONLY_DEFAULTS,
            )
        })
        return _FakeRuntime()

    api = _load_module(monkeypatch, builder)
    policy = _policy(api, dtype).to("rpu")
    assert seen == {
            **GENERIC_PERF_DEFAULTS,
            **GROUPED_EXPERT_DEFAULTS,
            **{key: "0" for key in FP16_ONLY_DEFAULTS},
    }
    policy.close()


def test_explicit_close_blocks_successor_build_until_environment_restore(monkeypatch):
    monkeypatch.delenv(PROFILE_KEY, raising=False)
    close_started = threading.Event()
    close_release = threading.Event()
    build_attempted = threading.Event()
    successor_builder_entered = threading.Event()
    runtimes = [
        _FakeRuntime(close_started=close_started, close_release=close_release),
        _FakeRuntime(),
    ]

    def builder(*_args, **_kwargs):
        runtime = runtimes.pop(0)
        if len(runtimes) == 0:
            successor_builder_entered.set()
        return runtime

    api = _load_module(monkeypatch, builder)
    first = _policy(api, "w4a16").to("rpu")
    second = _policy(api, "fp16")
    close_errors = []
    build_errors = []
    close_thread = threading.Thread(target=lambda: _capture(close_errors, first.close))
    def build_second():
        build_attempted.set()
        _capture(build_errors, second.to, "rpu")

    build_thread = threading.Thread(target=build_second)

    close_thread.start()
    assert close_started.wait(5)
    build_thread.start()
    assert build_attempted.wait(5)
    assert not successor_builder_entered.wait(0.2)
    assert os.environ.get(PROFILE_KEY) == "1"
    close_release.set()
    _join(close_thread)
    _join(build_thread)

    assert close_errors == []
    assert build_errors == []
    assert first._closed
    assert second._rpu_ready
    assert os.environ.get(PROFILE_KEY) == "0"
    second.close()
    assert PROFILE_KEY not in os.environ


def test_gc_finalizer_blocks_successor_build_until_environment_restore(monkeypatch):
    monkeypatch.delenv(PROFILE_KEY, raising=False)
    close_started = threading.Event()
    close_release = threading.Event()
    build_attempted = threading.Event()
    successor_builder_entered = threading.Event()
    runtimes = [
        _FakeRuntime(close_started=close_started, close_release=close_release),
        _FakeRuntime(),
    ]

    def builder(*_args, **_kwargs):
        runtime = runtimes.pop(0)
        if len(runtimes) == 0:
            successor_builder_entered.set()
        return runtime

    api = _load_module(monkeypatch, builder)
    holder = [_policy(api, "w4a16").to("rpu")]
    first_ref = weakref.ref(holder[0])
    second = _policy(api, "fp16")
    gc_errors = []
    build_errors = []

    def retire_by_gc():
        try:
            holder.clear()
            gc.collect()
        except BaseException as exc:
            gc_errors.append(exc)

    gc_thread = threading.Thread(target=retire_by_gc)
    def build_second():
        build_attempted.set()
        _capture(build_errors, second.to, "rpu")

    build_thread = threading.Thread(target=build_second)
    gc_thread.start()
    assert close_started.wait(5)
    build_thread.start()
    assert build_attempted.wait(5)
    assert not successor_builder_entered.wait(0.2)
    assert os.environ.get(PROFILE_KEY) == "1"
    close_release.set()
    _join(gc_thread)
    _join(build_thread)

    assert gc_errors == []
    assert build_errors == []
    assert first_ref() is None
    assert second._rpu_ready
    assert os.environ.get(PROFILE_KEY) == "0"
    second.close()
    assert PROFILE_KEY not in os.environ


@pytest.mark.parametrize(
    ("real_dim", "groups", "match"),
    [
        (15, [], "requires exactly 16"),
        (17, [], "requires exactly 16"),
        (
            16,
            [types.SimpleNamespace(key="action.arm", subtract_state=True)],
            "delta/relative action de-normalization is not certified",
        ),
    ],
)
def test_robot_ready_denormalizer_fails_closed_on_unsafe_contract(
    monkeypatch, real_dim, groups, match
):
    class FakeDenormalizer:
        @classmethod
        def from_spec(cls, *_args, **_kwargs):
            return types.SimpleNamespace(real_dim=real_dim, groups=groups)

    api = _load_module(monkeypatch, lambda *_args, **_kwargs: _FakeRuntime())
    sys.modules["rpu_backend.api.action_denorm"].ActionDenormalizer = FakeDenormalizer
    policy = api.Lingbot2Policy.from_checkpoint(
        "/unused",
        robot_norm_path="/unused/norm.json",
        action_norm_spec=[object()],
    )
    with pytest.raises(ValueError, match=match):
        policy._build_denormalizer()


def test_robot_norm_without_ordered_contract_fails_before_runtime_build(monkeypatch):
    calls = []

    def builder(*_args, **_kwargs):
        calls.append(True)
        return _FakeRuntime()

    api = _load_module(monkeypatch, builder)
    policy = api.Lingbot2Policy.from_checkpoint(
        "/unused",
        robot_norm_path="/unused/norm.json",
    )
    with pytest.raises(ValueError, match="matching training_config_path"):
        policy.to("rpu")

    assert calls == []
    assert not policy._rpu_build_started
    assert policy._env_snapshot is None


def test_nonfinite_physical_action_fails_closed(monkeypatch):
    api = _load_module(monkeypatch, lambda *_args, **_kwargs: _FakeRuntime())
    policy = _policy(api, "fp16").to("rpu")
    policy._denorm = types.SimpleNamespace(
        unapply=lambda *_args, **_kwargs: torch.full((50, 16), float("nan"))
    )

    with pytest.raises(RuntimeError, match="non-finite physical action"):
        policy.infer(obs={"state": torch.zeros(1, 55)})
    policy.close()


def test_infer_removes_the_runtime_batch_dimension(monkeypatch):
    api = _load_module(monkeypatch, lambda *_args, **_kwargs: _FakeRuntime())
    policy = _policy(api, "fp16").to("rpu")

    output = policy.infer(obs={"state": torch.zeros(1, 55)})

    assert tuple(output.actions_full.shape) == (50, 55)
    assert tuple(output.actions_normalized.shape) == (50, 16)
    policy.close()
