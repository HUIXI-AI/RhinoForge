"""Cold A8 admission and shared installer rollback; synthetic CPU carriers only."""
import pytest
import torch

from rpu_backend.api import _execution
from rpu_backend.runtime import decoder
from rpu_backend.tests.acceptance.test_decoder_install_transaction import (
    _FakeTensor, _finalize, _model, lookup_causal_decoder, native_install,
)
from test_qwen3_w8_core_profiles import _config, SHAPES


@pytest.mark.parametrize("profile", ["exact", "smaller", "llama", "w4", "nvfp4"])
@pytest.mark.parametrize("chunk", [128, 192, 256, 512])
def test_public_loader_admits_only_exact_a8_before_weight_loading(monkeypatch, profile, chunk):
    from transformers import AutoConfig, LlamaConfig
    from rpu_backend.api.causal_lm import RPUModelForCausalLM
    from rpu_backend.quant import load
    from rpu_backend.adapters.qwen3 import UnsupportedModelError
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    config = _config(SHAPES[-1] if profile != "smaller" else SHAPES[0])
    if profile == "llama":
        config = LlamaConfig(architectures=["LlamaForCausalLM"])
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *_a, **_k: config)
    calls = []
    class ReachedWeights(Exception):
        pass
    def load_weights(*args, **kwargs):
        calls.append("weights")
        raise ReachedWeights
    monkeypatch.setattr(load, "load_w8a16_model", load_weights)
    if profile == "nvfp4":
        from rpu_backend.quant import convert_qwen3_nvfp4
        monkeypatch.setattr(convert_qwen3_nvfp4, "convert_qwen3_to_nvfp4_", load_weights)
    expected = ReachedWeights if profile == "exact" else (UnsupportedModelError, ValueError)
    expected_message = "Prefill W8A8 requires existing W8 weights" if profile == "nvfp4" else None
    with pytest.raises(expected, match=expected_message):
        RPUModelForCausalLM.from_pretrained("synthetic", rpu_execution={
            "model": {"num_cores": 8}, "prefill": {"linear_w8a8": True, "chunk_size": chunk}},
            quantization={"w4": "w4a16", "nvfp4": "nvfp4"}.get(profile))
    assert calls == (["weights"] if profile == "exact" else [])


def test_a8_capability_is_scoped_to_qwen3():
    config = {"prefill": {"linear_w8a8": True}}
    normalized = _execution.normalize_rpu_execution(
        config, entry_point="test", supported=_execution._QWEN3_EXECUTION_SUPPORTED)
    assert normalized["prefill"]["linear_w8a8"] is True
    with pytest.raises(ValueError, match="not supported"):
        _execution.normalize_rpu_execution(
            config, entry_point="other", supported={"prefill": ("chunk_size",)})


@pytest.mark.parametrize("failure", [False, True])
def test_a8_installs_after_weights_before_descriptor_and_rolls_back(native_install, monkeypatch, failure):
    model = _model(model_type="qwen3")
    state, calls = native_install, []
    for name in ("set_weights_w8a16", "set_chunk_envelope", "set_linear_acc32",
                 "set_chunk_size_override"):
        monkeypatch.setattr(torch.ops.rpu, "causal_decoder_" + name,
                            lambda *args, _name=name: calls.append(_name), raising=False)

    def set_a8(handle, enabled):
        assert handle == 1 and enabled is True
        calls.append("set_prefill_w8a8")
        if failure:
            raise RuntimeError("synthetic A8 setter failure")

    def descriptor(handle):
        calls.append("descriptor")
        return (3, handle, 1)

    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_prefill_w8a8", set_a8, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_resolve_decode_stage_descriptor", descriptor, raising=False)
    table = _FakeTensor()

    def install():
        return decoder._install_causal_decoder_forward(
            model, arch="qwen3", cos_sin=(table, table),
            chunk_envelope_for=lookup_causal_decoder, scale_lists=([],) * 7,
            execution_config={"prefill": {"linear_w8a8": True}})

    try:
        if failure:
            with pytest.raises(RuntimeError, match="synthetic A8 setter failure"):
                install()
            assert not state["alive"]
            assert state["events"].count(("destroy", 1)) == 1
            assert not hasattr(model, "_rpu_decoder_handle")
            assert calls == ["set_chunk_envelope", "set_weights_w8a16", "set_linear_acc32", "set_prefill_w8a8"]
        else:
            assert install() == 1
            assert calls == ["set_chunk_envelope", "set_weights_w8a16", "set_linear_acc32",
                             "set_prefill_w8a8", "set_chunk_size_override", "descriptor"]
    finally:
        _finalize(model)


@pytest.mark.parametrize("arch, scales", [("llama", ([],) * 7), ("qwen3", None)])
def test_shared_installer_rejects_foreign_a8_before_create(native_install, arch, scales):
    table = _FakeTensor()
    with pytest.raises(ValueError, match="exact plain W8 Qwen3"):
        decoder._install_causal_decoder_forward(
            _model(model_type=arch), arch=arch, cos_sin=(table, table),
            chunk_envelope_for=lookup_causal_decoder, scale_lists=scales,
            execution_config={"prefill": {"linear_w8a8": True}})
    assert not native_install["events"]


def _exact_plain_32b_installer_model():
    model = _model(model_type="qwen3")
    model.layers *= 64
    block = model.layers[0]
    block.self_attn.head_dim = 128
    block.self_attn.q_proj.out_features = 8192
    block.self_attn.k_proj.out_features = block.self_attn.v_proj.out_features = 1024
    block.self_attn.o_proj.out_features = 5120
    block.mlp.gate_proj.out_features = block.mlp.up_proj.out_features = 25600
    model.config.hidden_size = 5120
    return model


@pytest.mark.parametrize("chunk,a8,fail_at", [
    (chunk, a8, fail_at) for chunk in (192, 256)
    for a8, fail_at in ((False, None), (True, None), (True, "cold"))
] + [(512, True, None), (512, True, "cold"), (512, True, "a8")])
def test_c192_installs_before_a8_and_rolls_back(native_install, monkeypatch, a8, fail_at, chunk):
    model = _exact_plain_32b_installer_model()
    calls = []
    for name in ("set_weights_w8a16", "set_chunk_envelope", "set_linear_acc32",
                 "set_prefill_w8a8", "set_chunk_size_override"):
        monkeypatch.setattr(torch.ops.rpu, "causal_decoder_" + name,
                            lambda *args, _name=name: calls.append(_name), raising=False)

    def set_c192(handle, enabled):
        assert handle == 1 and enabled is True
        calls.append(f"set_qwen3_c{chunk}")
        if fail_at == "cold":
            raise RuntimeError(f"synthetic C{chunk} setter failure")

    def set_a8(handle, enabled):
        assert handle == 1 and enabled is True
        calls.append("set_prefill_w8a8")
        if fail_at == "a8":
            raise RuntimeError("synthetic A8 setter failure")

    def descriptor(handle):
        calls.append("descriptor")
        return (3, handle, 1)

    monkeypatch.setattr(torch.ops.rpu, f"causal_decoder_set_qwen3_c{chunk}", set_c192, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_prefill_w8a8", set_a8, raising=False)
    for other in {192, 256, 512} - {chunk}:
        monkeypatch.setattr(torch.ops.rpu, f"causal_decoder_set_qwen3_c{other}",
                            lambda *_: pytest.fail("wrong cold chunk setter"), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_resolve_decode_stage_descriptor", descriptor, raising=False)
    table = _FakeTensor()
    prefix = ["set_chunk_envelope", "set_weights_w8a16", "set_linear_acc32", f"set_qwen3_c{chunk}"]
    def install():
        return decoder._install_causal_decoder_forward(
            model, arch="qwen3", cos_sin=(table, table),
            chunk_envelope_for=lambda *_: (512, chunk), scale_lists=([],) * 7,
            execution_config={"prefill": {"chunk_size": chunk, "linear_w8a8": a8}})
    try:
        if fail_at:
            failed = f"C{chunk}" if fail_at == "cold" else "A8"
            with pytest.raises(RuntimeError, match=f"synthetic {failed} setter failure"):
                install()
            assert not native_install["alive"]
            assert native_install["events"].count(("destroy", 1)) == 1
            assert not hasattr(model, "_rpu_decoder_handle")
            assert calls == prefix + (["set_prefill_w8a8"] if fail_at == "a8" else [])
        else:
            assert install() == 1
            assert calls == prefix + (["set_prefill_w8a8"] if a8 else []) + [
                "set_chunk_size_override", "descriptor"]
    finally:
        _finalize(model)


@pytest.mark.parametrize("a8,acc32", [(False, False), (True, True)])
def test_c512_direct_installer_refuses_invalid_math_before_native_create(native_install, monkeypatch, a8, acc32):
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_weights_w8a16",
                        lambda *_: pytest.fail("invalid C512 reached native weight install"), raising=False)
    model, table = _exact_plain_32b_installer_model(), _FakeTensor()
    with pytest.raises(ValueError, match="C512 requires.*linear_w8a8=true.*linear_acc32=false"):
        decoder._install_causal_decoder_forward(
            model, arch="qwen3", cos_sin=(table, table),
            chunk_envelope_for=lambda *_: (512, 512), scale_lists=([],) * 7,
            execution_config={"prefill": {"chunk_size": 512, "linear_w8a8": a8,
                                           "linear_acc32": acc32}})
    assert not native_install["events"]


@pytest.mark.parametrize("chunk", [192, 256])
def test_other_qwen3_c192_keeps_existing_install_path(native_install, monkeypatch, chunk):
    monkeypatch.setattr(torch.ops.rpu, f"causal_decoder_set_qwen3_c{chunk}",
                        lambda *_: pytest.fail(f"C{chunk} exact32B route called for another geometry"), raising=False)
    model, table = _model(model_type="qwen3"), _FakeTensor()
    try:
        assert decoder._install_causal_decoder_forward(
            model, arch="qwen3", cos_sin=(table, table),
            chunk_envelope_for=lambda *_: (4096, 512),
            execution_config={"prefill": {"chunk_size": chunk}}) == 1
    finally:
        _finalize(model)
