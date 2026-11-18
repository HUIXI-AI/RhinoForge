"""Bounded plain-Qwen3 larger-chunk admission; no hardware performance claim."""
from pathlib import Path
import subprocess

import pytest
import torch

from rpu_backend.adapters import qwen3
from rpu_backend.api import _execution
from rpu_backend.runtime import decoder
from test_qwen3_fused_lm_head_install import _Model, runtime
from test_qwen3_w8_core_profiles import _config, SHAPES
from test_fmb_standalone_layout_rebuild import _definition

ROOT = Path(__file__).resolve().parents[1]


def _execution_config(chunk, cores=8):
    return {"model": {"num_cores": cores}, "prefill": {"chunk_size": chunk}}


@pytest.mark.parametrize("chunk, expected", [("auto", (0, 64)), (16, (0, 64)),
                                             (64, (0, 64)), (128, (512, 128)),
                                             (192, (512, 192)), (256, (512, 256))])
def test_exact_w8_tp8_request_selects_one_bounded_envelope(chunk, expected):
    cfg, execution = _config(SHAPES[-1]), _execution_config(chunk)
    qwen3.Qwen3Adapter.preflight_execution(cfg, execution)
    lookup = qwen3._qwen3_chunk_envelope_lookup(cfg, execution, quantized=True)
    assert lookup("qwen3", 64, 5120) == expected
    # Default public tables and unrelated owners never inherit the candidate.
    assert qwen3.lookup_quantized_causal_decoder("qwen3", 64, 5120) == (0, 64)
    assert qwen3.lookup_causal_decoder("qwen3", 64, 5120) == (0, 64)
    if chunk in (128, 192, 256):
        with pytest.raises(RuntimeError, match="no certified chunk envelope"):
            lookup("qwen3_vl_text", 64, 5120)


@pytest.mark.parametrize("fault", ["cores4", "cores6", "w4_source", "foreign_recipe", "foreign_geometry"])
@pytest.mark.parametrize("chunk", [128, 192, 256])
def test_c128_refuses_other_precision_topology_or_geometry(fault, chunk):
    cfg, execution = _config(SHAPES[-1]), _execution_config(chunk)
    if fault.startswith("cores"):
        execution["model"]["num_cores"] = int(fault[-1])
    elif fault == "w4_source":
        cfg.quant_config = None
    elif fault == "foreign_recipe":
        cfg.quant_config = dict(cfg.quant_config, quantized_lm_head=False)
    else:
        cfg.num_attention_heads = 32
    with pytest.raises(qwen3.UnsupportedModelError):
        qwen3.Qwen3Adapter.preflight_execution(cfg, execution)
    assert qwen3._qwen3_chunk_envelope_lookup(cfg, execution, quantized=True)(
        "qwen3", 64, 5120) == (0, 64)


def _install_small_model(runtime, monkeypatch, chunk, *, a8=False):
    # Existing installation fixture exercises the actual adapter/session; small
    # tensor carriers stand in for the already separately validated W8 payload.
    model = _Model(8)
    model.config = _config(SHAPES[-1])
    model._rpu_execution = _execution_config(chunk)
    if a8:
        model._rpu_execution["prefill"]["linear_w8a8"] = True
    from rpu_backend.runtime import weights
    from rpu_backend.quant import load_qwen3_quantized
    from rpu_backend.graph import GraphRuntimePolicy
    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas", lambda self: True)
    monkeypatch.setattr(qwen3, "transform_linear_weight", lambda weight, **kwargs: weight.clone())
    monkeypatch.setattr(weights, "validate_decoder_mlp_padding", lambda *_a, **_k: None)
    monkeypatch.setattr(load_qwen3_quantized, "move_qwen3_w8a16_decoder_for_rpu", lambda model: None)
    monkeypatch.setattr(qwen3, "_convert_qwen3_remaining_weights",
                        lambda model: qwen3.convert_linear_weights_inplace(model))
    install = qwen3._install_causal_decoder_forward
    def install_with_topology(inner, **kwargs):
        handle = install(inner, **kwargs)
        inner._rpu_decoder_topology = kwargs["topology"]
        return handle
    monkeypatch.setattr(qwen3, "_install_causal_decoder_forward", install_with_topology)
    adapter = qwen3.Qwen3Adapter(model)
    adapter.to_rpu()
    return model, adapter


@pytest.mark.parametrize("chunk", ["auto", 16, 32, 64, 128, 192, 256, 512])
def test_a8_exact_profile_uses_bounded_envelope(chunk):
    cfg, execution = _config(SHAPES[-1]), _execution_config(chunk)
    execution["prefill"]["linear_w8a8"] = True
    qwen3.Qwen3Adapter.preflight_execution(cfg, execution)
    assert qwen3._qwen3_chunk_envelope_lookup(cfg, execution, quantized=True)(
        "qwen3", 64, 5120) == (512, chunk if chunk in (192, 256, 512) else 128)


@pytest.mark.parametrize("fault", ["a16", "acc32", "cores4", "cores6", "w4", "recipe", "geometry"])
def test_c512_refuses_non_a8_or_foreign_profile(fault):
    cfg, execution = _config(SHAPES[-1]), _execution_config(512)
    execution["prefill"]["linear_w8a8"] = fault != "a16"
    if fault == "acc32":
        execution["prefill"]["linear_acc32"] = True
    elif fault.startswith("cores"):
        execution["model"]["num_cores"] = int(fault[-1])
    elif fault == "w4":
        cfg.quant_config = None
    elif fault == "recipe":
        cfg.quant_config = dict(cfg.quant_config, quantized_lm_head=False)
    elif fault == "geometry":
        cfg.num_attention_heads = 32
    with pytest.raises(qwen3.UnsupportedModelError):
        qwen3.Qwen3Adapter.preflight_execution(cfg, execution)


@pytest.mark.parametrize("fault", ["cores4", "recipe", "geometry", "acc32", "w4", "chunk384"])
def test_a8_rejects_outside_exact_profile_even_with_auto_chunk(fault):
    cfg, execution = _config(SHAPES[-1]), _execution_config("auto")
    execution["prefill"]["linear_w8a8"] = True
    if fault == "cores4":
        execution["model"]["num_cores"] = 4
    elif fault == "recipe":
        cfg.quant_config = dict(cfg.quant_config, qaxis=1)
    elif fault == "geometry":
        cfg.num_attention_heads = 32
    elif fault == "acc32":
        execution["prefill"]["linear_acc32"] = True
    elif fault == "w4":
        cfg.quant_config = None
    else:
        execution["prefill"]["chunk_size"] = 384
    with pytest.raises(qwen3.UnsupportedModelError):
        qwen3.Qwen3Adapter.preflight_execution(cfg, execution)


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_a8_flag_requires_bool(value):
    execution = _execution_config("auto")
    execution["prefill"]["linear_w8a8"] = value
    with pytest.raises(qwen3.UnsupportedModelError, match="boolean"):
        qwen3.Qwen3Adapter.preflight_execution(_config(SHAPES[-1]), execution)


@pytest.mark.parametrize("chunk", ["auto", 64, 128, 192, 256, 512])
def test_a8_actual_adapter_passes_cold_config(runtime, monkeypatch, chunk):
    model, adapter = _install_small_model(runtime, monkeypatch, chunk, a8=True)
    try:
        installed = runtime["install_kwargs"]
        assert installed["chunk_envelope_for"]("qwen3", 64, 5120) == (512, chunk if chunk in (192, 256, 512) else 128)
        assert installed["execution_config"]["prefill"]["linear_w8a8"] is True
        assert adapter._bound_prefill_w8a8 is True
        assert adapter._bound_c512_prefill is (chunk == 512)
        assert installed["scale_lists"] is not None
    finally:
        decoder._close_causal_lm_model(model)


@pytest.mark.parametrize("old", [False, True])
def test_a8_mode_change_rejected_before_native_graph_mutation(runtime, monkeypatch, old):
    model, _ = _install_small_model(runtime, monkeypatch, 128, a8=old)
    native = _record_reconfiguration(runtime, monkeypatch, model)
    try:
        before, config = list(runtime["events"]), model._rpu_execution
        generation = model._execution_session.generation
        with pytest.raises(qwen3.UnsupportedModelError, match="cold-only"):
            _execution.reconfigure_rpu_execution(model, {
                "prefill": {"chunk_size": 128, "linear_w8a8": not old}})
        assert not native and runtime["events"] == before
        assert model._rpu_execution is config
        assert model._execution_session.generation == generation
    finally:
        decoder._close_causal_lm_model(model)


def test_a8_omitted_flag_preserved_by_valid_reconfiguration(runtime, monkeypatch):
    model, _ = _install_small_model(runtime, monkeypatch, 128, a8=True)
    native = _record_reconfiguration(runtime, monkeypatch, model)
    try:
        _execution.reconfigure_rpu_execution(model, {
            "prefill": {"chunk_size": 128, "padding_rows": 16}})
        assert model._rpu_execution["prefill"]["linear_w8a8"] is True
        assert not native
    finally:
        decoder._close_causal_lm_model(model)


@pytest.mark.parametrize("chunk, expected", [("auto", (0, 64)), (64, (0, 64)),
                                            (128, (512, 128)), (192, (512, 192)), (256, (512, 256))])
def test_actual_adapter_installs_selected_envelope(runtime, monkeypatch, chunk, expected):
    model, adapter = _install_small_model(runtime, monkeypatch, chunk)
    try:
        installed = runtime["install_kwargs"]
        assert installed["chunk_envelope_for"]("qwen3", 64, 5120) == expected
        assert installed["execution_config"]["prefill"]["chunk_size"] == chunk
        assert installed["topology"].identity() == (2, 8, 8, 8, 8, 8)
        assert installed["scale_lists"] is not None
        assert adapter.preflight_execution.__func__ is qwen3.Qwen3Adapter._preflight_bound_execution
    finally:
        decoder._close_causal_lm_model(model)


@pytest.mark.parametrize("old,new,a8", [(64, 128, False), ("auto", 128, False),
    (128, "auto", False), (128, 64, False), (128, 192, False), (192, 128, False),
    (192, 64, False), (64, 192, False), (192, "auto", False), (128, 192, True),
    (192, 128, True), (192, 64, True), (64, 192, True), (192, "auto", True),
    (128, 256, False), (192, 256, False), (256, 192, False), (256, 128, False),
    (256, "auto", False), (64, 256, False), (128, 256, True), (192, 256, True),
    (256, 192, True), (256, 128, True), (256, "auto", True),
    (128, 512, True), (192, 512, True), (256, 512, True), ("auto", 512, True),
    (512, 128, True), (512, 192, True), (512, 256, True), (512, "auto", True)])
def test_crossing_c128_mode_requires_reload_before_native_or_graph_mutation(runtime, monkeypatch, old, new, a8):
    model, _ = _install_small_model(runtime, monkeypatch, old, a8=a8)
    try:
        before, config = list(runtime["events"]), model._rpu_execution
        with pytest.raises(qwen3.UnsupportedModelError, match="cold-only"):
            _execution.reconfigure_rpu_execution(model, {"prefill": {"chunk_size": new}})
        assert model._rpu_execution is config and runtime["events"] == before
    finally:
        decoder._close_causal_lm_model(model)


def _record_reconfiguration(runtime, monkeypatch, model):
    native = []
    for name, operation in {
        "execution_reconfigure_begin": lambda token: native.append("begin") or token,
        "execution_reconfigure_commit": lambda token: native.append("commit"),
        "execution_reconfigure_abort": lambda token: native.append("abort"),
        "execution_reconfigure_abort_attempt": lambda token: False,
        "causal_decoder_stage_chunk_size_override": lambda *args: native.append("stage"),
    }.items():
        monkeypatch.setattr(torch.ops.rpu, name, operation, raising=False)
    model.model._rpu_decoder_graph_cache.begin_warmup = lambda: runtime["events"].append("warmup")
    return native


@pytest.mark.parametrize("fault", ["recipe", "geometry"])
@pytest.mark.parametrize("bound,requested", [(128, "auto"), (128, 128), (192, "auto"), (192, 192), (256, "auto"), (256, 256), (512, "auto"), (512, 512)])
def test_bound_c128_rejects_mutable_hf_profile_drift(runtime, monkeypatch, fault, bound, requested):
    model, _ = _install_small_model(runtime, monkeypatch, bound, a8=bound == 512)
    native = _record_reconfiguration(runtime, monkeypatch, model)
    try:
        if fault == "recipe":
            model.config.quant_config = dict(model.config.quant_config, qaxis=1)
        else:
            model.config.num_attention_heads = 32
        before, config = list(runtime["events"]), model._rpu_execution
        generation = model._execution_session.generation
        with pytest.raises(qwen3.UnsupportedModelError, match=f"bound C{bound}.*profile changed"):
            _execution.reconfigure_rpu_execution(model, {
                "prefill": {"chunk_size": requested, "padding_rows": 16}})
        assert not native
        assert model._rpu_execution is config and runtime["events"] == before
        assert model._execution_session.generation == generation
    finally:
        decoder._close_causal_lm_model(model)


def test_bound_c128_keeps_valid_padding_reconfiguration(runtime, monkeypatch):
    model, adapter = _install_small_model(runtime, monkeypatch, 128)
    native = _record_reconfiguration(runtime, monkeypatch, model)
    try:
        generation = model._execution_session.generation
        _execution.reconfigure_rpu_execution(model, {
            "prefill": {"chunk_size": 128, "padding_rows": 16}})
        assert adapter._bound_c128_prefill is True
        assert model._rpu_execution["prefill"]["chunk_size"] == 128
        assert model._rpu_execution["prefill"]["padding_rows"] == 16
        assert model._execution_session.generation == generation + 1
        assert not native  # The unchanged chunk needs no native override stage.
        assert runtime["events"][-2:] == ["warmup", "clear"]
    finally:
        decoder._close_causal_lm_model(model)


@pytest.mark.parametrize("a8", [False, True])
@pytest.mark.parametrize("chunk", [192, 256])
def test_c192_rejects_acc32_and_keeps_padding_only_reconfiguration(runtime, monkeypatch, a8, chunk):
    config, execution = _config(SHAPES[-1]), _execution_config(chunk)
    execution["prefill"].update(linear_w8a8=a8, linear_acc32=True)
    with pytest.raises(qwen3.UnsupportedModelError, match="linear_acc32=false"):
        qwen3.Qwen3Adapter.preflight_execution(config, execution)
    model, adapter = _install_small_model(runtime, monkeypatch, chunk, a8=a8)
    native = _record_reconfiguration(runtime, monkeypatch, model)
    try:
        _execution.reconfigure_rpu_execution(model, {
            "prefill": {"chunk_size": chunk, "padding_rows": 16}})
        assert adapter._bound_c192_prefill is (chunk == 192)
        assert adapter._bound_c256_prefill is (chunk == 256)
        assert adapter._bound_c128_prefill is False
        assert model._rpu_execution["prefill"]["chunk_size"] == chunk
        assert not native
    finally:
        decoder._close_causal_lm_model(model)


@pytest.mark.parametrize("chunk", [128, 192, 256])
def test_existing_native_envelope_enforces_p512_and_continuation_bound(tmp_path, chunk):
    # Execute the actual shared guard with the adapter-selected cold envelope.
    body = _definition((ROOT / "src/core/fused_model_base.cpp").read_text(),
                       "int64_t FusedModelBase::enforce_chunk_envelope(")
    prefix = r'''
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#define TORCH_CHECK(ok, ...) do { if(!(ok)) throw std::runtime_error("reject"); } while(false)
struct FusedModelBase {
 struct ChunkEnvelope { int64_t max_kv_len, chunk; bool declared()const{return max_kv_len>0;} };
 struct Impl { ChunkEnvelope chunk_envelope_; } impl;
 Impl* pimpl_=&impl;
 int64_t enforce_chunk_envelope(int64_t,int64_t,const char*)const;
};
'''
    envelope = qwen3._qwen3_chunk_envelope_lookup(
        _config(SHAPES[-1]), _execution_config(chunk), quantized=True)("qwen3",64,5120)
    program = prefix + body + r'''
int main(){
 FusedModelBase m;m.impl.chunk_envelope_={MAX_KV,MAX_CHUNK};
 assert(m.enforce_chunk_envelope(512,0,"plain")==MAX_CHUNK);
 assert(m.enforce_chunk_envelope(384,128,"plain")==MAX_CHUNK);
 for(auto pair : {std::pair<int64_t,int64_t>{513,0},{384,129},{4096,0}}){
  bool rejected=false;try{m.enforce_chunk_envelope(pair.first,pair.second,"plain");}
  catch(const std::runtime_error&){rejected=true;}assert(rejected);
 }
 // Existing single-token decode still follows its separate KV/RoPE guards.
 assert(m.enforce_chunk_envelope(1,512,"plain")==MAX_CHUNK);
}
'''.replace("MAX_KV",str(envelope[0])).replace("MAX_CHUNK",str(envelope[1]))
    cpp, exe = tmp_path / "envelope.cpp", tmp_path / "envelope"
    cpp.write_text(program)
    built = subprocess.run(["c++","-std=c++17",str(cpp),"-o",str(exe)],capture_output=True,text=True,timeout=30)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(exe)],capture_output=True,text=True,timeout=10)
    assert ran.returncode == 0, ran.stderr
