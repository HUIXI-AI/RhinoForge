from pathlib import Path
import importlib.util
import inspect
import json
import sys
import types

import pytest


MODULE = (
    Path(__file__).resolve().parents[1]
    / "python"
    / "rpu_backend"
    / "api"
    / "wall_oss.py"
)
VALIDATION_MODULE = (
    Path(__file__).resolve().parents[1]
    / "python"
    / "rpu_backend"
    / "adapters"
    / "wall_oss"
    / "checkpoint_validation.py"
)
P4B_E2E = (
    Path(__file__).resolve().parents[1]
    / "python"
    / "rpu_backend"
    / "tests"
    / "model"
    / "wall_oss"
    / "test_wall_oss_p4b_e2e.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "wall_oss_api_precision_config_test", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_validation_module():
    spec = importlib.util.spec_from_file_location(
        "wall_oss_checkpoint_validation_test", VALIDATION_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_pair(tmp_path, *, method="w8a16", source="fp16", mismatch=False):
    config = {
        "model_type": "qwen2_5_vl",
        "hidden_size": 2048,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "intermediate_size": 11008,
        "action_hidden_size": 1024,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
        "vocab_size": 151936,
        "num_experts": 2,
        "experts": [{"hidden_size": 2048}, {"hidden_size": 1024}],
        "dim_inputs": [2048, 1024],
        "dof_config": {"arm": 7},
        "vision_config": {"hidden_size": 1280, "patch_size": 14},
    }
    w8 = tmp_path / "w8"
    fp16 = tmp_path / "fp16"
    w8.mkdir()
    fp16.mkdir()
    w8_config = dict(config)
    if mismatch:
        w8_config["rope_theta"] = 10_000.0
    (w8 / "config.json").write_text(json.dumps(w8_config), encoding="utf-8")
    (fp16 / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (w8 / "rpu_quant_config.json").write_text(
        json.dumps({"method": method, "source": source}),
        encoding="utf-8",
    )
    return w8, fp16




def test_wint4_scoped_nvfp4_is_public_but_action_hybrid_is_not():
    api = _load_module()
    parameters = inspect.signature(
        api.WallOssPolicy.from_checkpoint
    ).parameters
    assert "nvfp4a16" in parameters
    assert "w8_nvfp4a16" not in parameters


def test_precision_modes_are_mutually_exclusive_at_construction():
    api = _load_module()
    with pytest.raises(ValueError, match="mutually exclusive"):
        api.WallOssPolicy.from_checkpoint(
            "/models/wall-oss",
            w8a16=True,
            nvfp4a16=True,
        )


def test_public_precision_property_remains_canonical():
    api = _load_module()
    policy = api.WallOssPolicy.from_checkpoint(
        "/models/wall-oss-w8a16",
        w8a16=True,
        fp16_ckpt_dir="/models/wall-oss-fp16",
    )
    assert policy.precision == "w8a16"


def test_public_nvfp4_precision_property_is_canonical():
    api = _load_module()
    policy = api.WallOssPolicy.from_checkpoint(
        "/models/wall-oss-w8a16",
        nvfp4a16=True,
        fp16_ckpt_dir="/models/wall-oss-fp16",
    )
    assert policy.precision == "nvfp4a16"


def test_public_nvfp4_maps_to_wint4_scoped_runtime(monkeypatch):
    api = _load_module()
    calls = {}
    adapter = types.ModuleType("rpu_backend.adapters.wall_oss")
    errors = types.ModuleType("rpu_backend.api.errors")
    loading = types.ModuleType("rpu_backend.api._loading")
    causal_lm = types.ModuleType("rpu_backend.api.causal_lm")
    execution = types.ModuleType("rpu_backend.api._execution")

    errors.RPUBackendError = RuntimeError
    errors.RPUConfigError = ValueError
    loading.resolve_loader_device = lambda device, **_kwargs: str(device) in {
        "rpu", "rpu:0"
    }
    causal_lm._claim_live_instance = lambda _policy: None
    execution.normalize_rpu_execution = (
        lambda value, **_kwargs: {} if value is None else value
    )

    def build_wall_oss_vla(checkpoint, **kwargs):
        calls["checkpoint"] = checkpoint
        calls["kwargs"] = kwargs
        return types.SimpleNamespace(
            _runtime_complete=lambda: True,
            _rpu_execution=kwargs["rpu_execution"],
        )

    adapter.build_wall_oss_vla = build_wall_oss_vla
    monkeypatch.setitem(sys.modules, "rpu_backend.adapters.wall_oss", adapter)
    monkeypatch.setitem(sys.modules, "rpu_backend.api.errors", errors)
    monkeypatch.setitem(sys.modules, "rpu_backend.api._loading", loading)
    monkeypatch.setitem(sys.modules, "rpu_backend.api.causal_lm", causal_lm)
    monkeypatch.setitem(sys.modules, "rpu_backend.api._execution", execution)
    policy = api.WallOssPolicy.from_checkpoint(
        "/models/wall-oss-w8a16",
        nvfp4a16=True,
        fp16_ckpt_dir="/models/wall-oss-fp16",
        rpu_execution={"prefill": {"linear_acc32": True},
                       "vision": {"linear_acc32": False},
                       "action": {"linear_acc32": True}},
    ).to("rpu")

    assert policy.precision == "nvfp4a16"
    assert calls["checkpoint"] == "/models/wall-oss-w8a16"
    assert calls["kwargs"]["nvfp4_wint4_scope"] is True
    assert calls["kwargs"]["fp16_ckpt_dir"] == "/models/wall-oss-fp16"
    assert policy._rpu_execution is calls["kwargs"]["rpu_execution"]
    assert calls["kwargs"]["rpu_execution"] == {
        "prefill": {"linear_acc32": True},
        "vision": {"linear_acc32": False},
        "action": {"linear_acc32": True},
    }


def test_public_rtc_w8a16_maps_to_hybrid_runtime(monkeypatch):
    api = _load_module()
    calls = {}
    adapter = types.ModuleType("rpu_backend.adapters.wall_oss")
    errors = types.ModuleType("rpu_backend.api.errors")
    loading = types.ModuleType("rpu_backend.api._loading")
    causal_lm = types.ModuleType("rpu_backend.api.causal_lm")

    errors.RPUBackendError = RuntimeError
    errors.RPUConfigError = ValueError
    loading.resolve_loader_device = lambda device, **_kwargs: str(device) in {
        "rpu", "rpu:0"
    }
    causal_lm._claim_live_instance = lambda _policy: None

    def build_wall_oss_vla(checkpoint, **kwargs):
        calls["checkpoint"] = checkpoint
        calls["kwargs"] = kwargs
        return types.SimpleNamespace(
            _runtime_complete=lambda: True,
            _rpu_execution=kwargs["rpu_execution"],
        )

    adapter.build_wall_oss_vla = build_wall_oss_vla
    monkeypatch.setitem(sys.modules, "rpu_backend.adapters.wall_oss", adapter)
    monkeypatch.setitem(sys.modules, "rpu_backend.api.errors", errors)
    monkeypatch.setitem(sys.modules, "rpu_backend.api._loading", loading)
    monkeypatch.setitem(sys.modules, "rpu_backend.api.causal_lm", causal_lm)
    policy = api.WallOssPolicy.from_checkpoint(
        "/models/v728-w8a16",
        rtc=True,
        dataset_key="ex_normal",
        camera_names=("face_view", "left_wrist_view", "right_wrist_view"),
        delta_action=True,
        state_bins=512,
        action_horizon=20,
        num_steps=6,
        w8a16=True,
        fp16_ckpt_dir="/models/v728-fp16",
    ).to("rpu")

    assert policy.precision == "w8a16-hybrid"
    assert calls["checkpoint"] == "/models/v728-w8a16"
    assert calls["kwargs"]["rtc"] is True
    assert calls["kwargs"]["w8a16"] is True
    assert calls["kwargs"]["fp16_ckpt_dir"] == "/models/v728-fp16"


@pytest.mark.parametrize("mode", ["w8a16", "w4a16", "nvfp4a16"])
def test_public_quantized_modes_require_explicit_fp16_glue(mode):
    api = _load_module()
    with pytest.raises(ValueError, match="requires fp16_ckpt_dir"):
        api.WallOssPolicy.from_checkpoint(
            "/models/wall-oss-quantized",
            **{mode: True},
        )


def test_hybrid_checkpoint_pair_accepts_declared_matching_source(tmp_path):
    validation = _load_validation_module()
    w8, fp16 = _write_pair(tmp_path)
    w8_config, fp16_config = validation.validate_w8_nvfp4_checkpoint_pair(
        w8, fp16)
    assert w8_config["hidden_size"] == fp16_config["hidden_size"] == 2048


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"method": "nvfp4"}, "method='w8a16'"),
        ({"source": "another-checkpoint"}, "cannot be resolved exactly"),
        ({"fp16_quantized": True}, "must be an unquantized source"),
        ({"mismatch": True}, "architecture-compatible"),
    ],
)
def test_hybrid_checkpoint_pair_rejects_unsafe_inputs(
    tmp_path, override, match
):
    validation = _load_validation_module()
    w8, fp16 = _write_pair(
        tmp_path,
        method=override.get("method", "w8a16"),
        source=override.get("source", "fp16"),
        mismatch=override.get("mismatch", False),
    )
    if override.get("fp16_quantized"):
        (fp16 / "rpu_quant_config.json").write_text(
            json.dumps({"method": "w8a16", "source": "source"}),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match=match):
        validation.validate_w8_nvfp4_checkpoint_pair(w8, fp16)


def test_hybrid_pair_rejects_same_basename_in_another_directory(tmp_path):
    validation = _load_validation_module()
    w8, fp16 = _write_pair(tmp_path)
    other_parent = tmp_path / "other"
    other_parent.mkdir()
    wrong_fp16 = other_parent / fp16.name
    wrong_fp16.mkdir()
    (wrong_fp16 / "config.json").write_text(
        (fp16 / "config.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="declared source pair"):
        validation.validate_w8_nvfp4_checkpoint_pair(w8, wrong_fp16)
