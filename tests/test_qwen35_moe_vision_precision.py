"""MoE-only cold Vision precision through public configuration and native binding."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters import qwen3_5_moe as moe
from rpu_backend.adapters.qwen3_5 import vision
from rpu_backend.api._execution import normalize_rpu_execution
from test_qwen35_moe_core_profiles import config as moe_config


PROFILES = (
    vision.QWEN3_5_MOE_35B_VISION_PROFILE,
    vision.QWEN3_5_MOE_35B_TP4_VISION_PROFILE,
    vision.QWEN3_5_MOE_35B_TP6_VISION_PROFILE,
)
CHOICES = ({}, {"linear_acc32": False, "gelu_erf_ultra": False},
           {"linear_acc32": True}, {"gelu_erf_ultra": True},
           {"linear_acc32": True, "gelu_erf_ultra": True})


def normalize(stage):
    return normalize_rpu_execution({"vision": stage}, entry_point="MoE precision test",
                                   supported=moe.Qwen3_5MoeAdapter._EXECUTION_SUPPORTED)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("stage", CHOICES)
def test_independent_moe_choices_and_defaults(profile, stage):
    config = normalize(stage)
    assert vision._qwen35_vision_precision(config, prepared_profile=profile) == (
        stage.get("linear_acc32", False), stage.get("gelu_erf_ultra", False))
    with pytest.raises(TypeError):
        config["vision"] = {"gelu_erf_ultra": True}


@pytest.mark.parametrize("field", ["linear_acc32", "gelu_erf_ultra"])
@pytest.mark.parametrize("value", [0, 1, "true", "auto", None])
def test_moe_controls_require_boolean(field, value):
    with pytest.raises(TypeError, match="must be bool"):
        normalize({field: value})
    with pytest.raises(TypeError, match="must be bool"):
        vision._qwen35_vision_precision(
            {"vision": {field: value}}, prepared_profile=PROFILES[0])


@pytest.mark.parametrize("cores", [4, 6, 8])
@pytest.mark.parametrize("stage", CHOICES)
def test_public_loader_passes_choices_to_real_moe_adapter(monkeypatch, cores, stage):
    from rpu_backend.api import conditional_generation as public

    model = torch.nn.Module()
    model.config = cfg = moe_config()
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.model.visual = torch.nn.Module()
    model._qwen3_5_moe_profile = moe.EXACT_PROFILE_ID
    seen = []
    monkeypatch.setattr(public.AutoConfig, "from_pretrained", lambda *_, **__: cfg)

    def load(path, *, config, dtype, **kwargs):
        assert path == "/mock-moe" and config is cfg and dtype is torch.float16
        seen.append("load")
        return model

    def to_rpu(instance):
        assert instance.model is model
        assert dict(instance._rpu_execution.get("vision", {})) == stage
        assert instance._rpu_execution == model._execution_session.config
        seen.append("install")
        return model

    monkeypatch.setattr(moe.Qwen3_5MoeAdapter, "load_pretrained", staticmethod(load))
    monkeypatch.setattr(moe.Qwen3_5MoeAdapter, "to_rpu", to_rpu)
    result = public.RPUModelForConditionalGeneration.from_pretrained(
        "/mock-moe", device="rpu",
        rpu_execution={"model": {"num_cores": cores}, "vision": stage})
    assert result is model and seen == ["load", "install"]


@pytest.mark.parametrize("enabled", [False, True])
def test_dense_rejects_moe_only_field_before_weight_load(monkeypatch, enabled):
    from rpu_backend.api import conditional_generation as public
    from rpu_backend.adapters import qwen3_5
    from test_qwen35_core_profiles import _config

    monkeypatch.setattr(public.AutoConfig, "from_pretrained", lambda *_, **__: _config())
    monkeypatch.setattr(qwen3_5.Qwen3_5Adapter, "load_pretrained",
                        lambda *_, **__: pytest.fail("unsupported field reached weights"))
    with pytest.raises(ValueError, match="gelu_erf_ultra.*not supported"):
        public.RPUModelForConditionalGeneration.from_pretrained(
            "/mock-dense", rpu_execution={"vision": {"gelu_erf_ultra": enabled}})
    with pytest.raises(ValueError, match="only by Qwen3.5-MoE"):
        vision._qwen35_vision_precision({"vision": {"gelu_erf_ultra": enabled}})


@pytest.mark.parametrize("field", ["gelu_erf_ultra"])
def test_moe_text_rejects_vision_precision(field):
    with pytest.raises(ValueError, match="not supported"):
        normalize_rpu_execution({"prefill": {field: True}}, entry_point="MoE text",
                                supported=moe.Qwen3_5MoeAdapter._EXECUTION_SUPPORTED)


@pytest.mark.parametrize("field", ["linear_acc32", "gelu_erf_ultra"])
def test_moe_hot_precision_change_rejects_before_mutation(monkeypatch, field):
    instance = SimpleNamespace(model=SimpleNamespace(config=moe_config()),
                               preflight_execution=moe.Qwen3_5MoeAdapter.preflight_execution)
    monkeypatch.setattr(moe, "_qwen3_5_moe_cold_config_only", lambda _: False)
    monkeypatch.setattr(torch.ops.rpu, "execution_reconfigure_begin",
                        lambda *_: pytest.fail("native mutation"), raising=False)
    with pytest.raises(ValueError, match="immutable after device installation"):
        moe._validate_qwen3_5_moe_execution_reconfigure(instance, normalize({field: True}))


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("stage", CHOICES)
def test_moe_installer_binds_both_controls_before_weight_conversion(monkeypatch, profile, stage):
    from test_qwen3_5_vision_helpers import _runtime_capable_vision
    from test_qwen35_moe_vision_bridge import _vision_config

    model = _runtime_capable_vision()
    model.config = _vision_config()
    model.blocks = [SimpleNamespace() for _ in range(27)]
    cores = vision._qwen3_5_moe_vision_profile_cores(profile)
    events = []
    for name, callback in (
        ("create", lambda: events.append("create") or 17),
        ("destroy", lambda _: None),
        ("set_execution_cores", lambda _, value: events.append(("cores", value))),
        ("set_moe_precision", lambda _, acc, erf: events.append(("precision", acc, erf))),
        ("set_linear_acc32", lambda *_: pytest.fail("MoE used coupled dense setter")),
        ("set_gelu_erf_ultra", lambda *_: pytest.fail("MoE used coupled dense setter")),
        ("set_chunk_size_cap", lambda *_: None),
        ("set_cache_capacity", lambda *_: None),
        ("set_prefill_chunk_size", lambda *_: None),
        ("resolve_prefill_domain", lambda *_: None),
    ):
        monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_" + name, callback, raising=False)

    def stop(*_, **__):
        events.append("convert")
        raise RuntimeError("stop before weights")

    monkeypatch.setattr(vision, "_convert_vision_block_weights_for_rpu", stop)
    try:
        with pytest.raises(RuntimeError, match="stop before weights"):
            vision._install_qwen3_5_vision_for_rpu_impl(
                model, execution_config=normalize(stage), num_cores=cores,
                _prepared_profile=profile, _cold_numeric_opt_in=True)
        assert events == ["create", *([("cores", 4)] if cores == 4 else []),
                          ("precision", stage.get("linear_acc32", False),
                           stage.get("gelu_erf_ultra", False)), "convert"]
    finally:
        model._rpu_vision_installing_finalizer()


def test_moe_missing_native_precision_op_rejects_before_handle(monkeypatch):
    from test_qwen3_5_vision_helpers import _runtime_capable_vision
    from test_qwen35_moe_vision_bridge import _vision_config

    model = _runtime_capable_vision()
    model.config = _vision_config()
    model.blocks = [SimpleNamespace() for _ in range(27)]
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_set_moe_precision", None, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_create",
                        lambda: pytest.fail("missing op created handle"), raising=False)
    with pytest.raises(RuntimeError, match="lacks qwen3_5_vision_set_moe_precision"):
        vision._install_qwen3_5_vision_for_rpu_impl(
            model, _prepared_profile=PROFILES[0], _cold_numeric_opt_in=True)


def test_native_moe_choices_are_atomic_cold_and_keep_dense_constraint(tmp_path):
    from test_qwen35_moe_core_native import _definition, _run_cpp

    source = (Path(__file__).resolve().parents[1] /
              "src/fused/rpu_qwen3_5_vision_model.cpp").read_text()
    methods = "\n".join(_definition(source, "void Qwen3_5VisionModel::" + name + "(")
                        for name in ("set_moe_precision", "set_linear_acc32", "set_gelu_erf_ultra"))
    _run_cpp(tmp_path, r'''
#include <cassert>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(cond, ...) if (!(cond)) throw std::runtime_error("rejected")
struct Qwen3_5VisionModel {
    int cores = 8, layers = 0, invalidations = 0;
    std::vector<int> layer_weights_;
    bool linear_acc32_ = false, gelu_erf_ultra_ = false, moe_precision_ = false;
    bool high_precision_ = false;
    int num_cores() const { return cores; }
    int num_layers() const { return layers; }
    void invalidate_model_state() { ++invalidations; }
    void set_moe_precision(bool, bool);
    void set_linear_acc32(bool);
    void set_gelu_erf_ultra(bool);
};
''' + methods + r'''
template<class F> void rejects(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}
int main() {
    for (int cores : {4, 8}) for (bool acc : {false, true}) for (bool erf : {false, true}) {
        Qwen3_5VisionModel m; m.cores = cores;
        m.set_moe_precision(acc, erf);
        assert(m.moe_precision_ && m.linear_acc32_ == acc && m.gelu_erf_ultra_ == erf);
        assert(m.invalidations == 1);
        m.layers = 27;
        rejects([&] { m.set_moe_precision(!acc, !erf); });
        assert(m.linear_acc32_ == acc && m.gelu_erf_ultra_ == erf && m.invalidations == 1);
    }
    for (int cores : {1, 2, 6}) {
        Qwen3_5VisionModel m; m.cores = cores;
        rejects([&] { m.set_moe_precision(false, true); });
        assert(!m.moe_precision_ && !m.gelu_erf_ultra_ && m.invalidations == 0);
    }
    Qwen3_5VisionModel dense;
    rejects([&] { dense.set_gelu_erf_ultra(true); });
    dense.set_linear_acc32(true);
    dense.set_gelu_erf_ultra(true);
    rejects([&] { dense.set_linear_acc32(false); });
    assert(!dense.moe_precision_);
}
''')
