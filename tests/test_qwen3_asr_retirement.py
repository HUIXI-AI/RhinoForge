"""ASR audio handles share the decoder retirement and process-owner contract."""
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from rpu_backend.api import Qwen3ASRModel, qwen3_asr
from rpu_backend.api._qwen3_asr_audio import ResidentAudioEncoder
from rpu_backend.api import causal_lm, _execution
from rpu_backend.runtime import _native_retirement


@pytest.mark.parametrize("failure", ["graph", "destroy"])
def test_audio_failure_keeps_parent_and_never_retries_after_gc(monkeypatch, failure):
    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.setattr(_native_retirement, "_FAILED_RETIREMENTS", [])
    events = []
    model = Qwen3ASRModel()
    model._closed = False
    model._head_weight = object()
    causal_lm._claim_live_instance(model)
    encoder = model._audio_encoder = ResidentAudioEncoder(None, {})
    encoder.handle = 7

    def clear():
        events.append("clear")
        if failure == "graph":
            raise RuntimeError("graph retirement failed")

    def destroy(handle):
        assert handle == 7
        events.append("destroy")
        raise RuntimeError("native retirement failed")

    encoder.graph = SimpleNamespace(clear=clear, cache_invariant_ok=lambda: True)
    resource = encoder._resource = _native_retirement._InstalledNativeResource(
        encoder, encoder.handle, destroy, graphs=(encoder.graph,), keepalive={},
        label="test ASR audio", handle_name="handle")
    resource.take_ownership(model)
    assert not resource.finalizer.alive
    with pytest.raises(RuntimeError, match="retirement failed"):
        model.close()
    previous = list(events)
    for operation in (model.close, model.execution_stats,
                      lambda: model.transcribe(torch.zeros(8000))):
        with pytest.raises(RuntimeError, match="restart"):
            operation()
    assert events == previous
    assert encoder.handle == 7 and not model._closed
    retained = weakref.ref(model)
    del model, encoder, resource
    gc.collect()
    assert retained() is not None and retained()._head_weight is not None
    assert events == previous
    with pytest.raises(RuntimeError, match="unsafe"):
        causal_lm._claim_live_instance(Qwen3ASRModel())
    assert len(_native_retirement._FAILED_RETIREMENTS) == 1


def test_partial_audio_install_is_owned_before_weight_conversion(monkeypatch):
    from rpu_backend.runtime import weights

    events = []
    layer = SimpleNamespace(
        self_attn_layer_norm=SimpleNamespace(eps=1e-5),
        final_layer_norm=SimpleNamespace(eps=1e-5),
        self_attn=SimpleNamespace(**{name: torch.nn.Linear(1, 1) for name in (
            "q_proj", "k_proj", "v_proj", "out_proj")}),
        fc1=torch.nn.Linear(1, 1), fc2=torch.nn.Linear(1, 1))
    layer.self_attn.q_proj = torch.nn.Linear(896, 896)
    audio = SimpleNamespace(config=SimpleNamespace(
        d_model=896, encoder_layers=18, encoder_attention_heads=14,
        encoder_ffn_dim=3584, activation_function="gelu", n_window=50,
        n_window_infer=800, output_dim=1024), layers=[layer] * 18,
        ln_post=SimpleNamespace(eps=1e-5))
    graph = SimpleNamespace(clear=lambda: events.append("clear"),
                            cache_invariant_ok=lambda: True)
    monkeypatch.setattr(torch.rpu, "GraphCache", lambda **kwargs: graph, raising=False)
    monkeypatch.setattr(torch.ops, "rpu", SimpleNamespace(
        siglip_create=lambda value: 7,
        siglip_destroy=lambda handle: events.append(("destroy", handle))))

    def fail(*args):
        raise ValueError("weight conversion failed")

    monkeypatch.setattr(weights, "transform_linear_weight", fail)
    # The audio adapter imported this helper directly.
    from rpu_backend.api import _qwen3_asr_audio
    monkeypatch.setattr(_qwen3_asr_audio, "transform_linear_weight", fail)
    parent = Qwen3ASRModel()
    encoder = parent._audio_encoder = ResidentAudioEncoder(audio, {})
    with pytest.raises(ValueError, match="weight conversion failed"):
        encoder.install(parent)
    assert encoder._resource.parent() is parent
    assert not encoder._resource.finalizer.alive
    encoder.close()
    encoder.close()
    assert events == ["clear", ("destroy", 7)]
    assert encoder.handle is None and encoder.graph is None


def test_install_failure_keeps_original_error_when_cleanup_fails(monkeypatch):
    monkeypatch.setattr(causal_lm, "_claim_live_instance", lambda owner: None)
    monkeypatch.setattr(qwen3_asr, "_preflight", lambda *args: {})
    monkeypatch.setattr(qwen3_asr, "_load_processor", lambda *args: (None, None, ""))

    def load(*args):
        raise ValueError("original install failure")

    def close(self):
        raise RuntimeError("cleanup failure")

    monkeypatch.setattr(qwen3_asr, "_load_components", load)
    monkeypatch.setattr(Qwen3ASRModel, "close", close)
    with pytest.raises(ValueError, match="original install failure") as caught:
        Qwen3ASRModel.from_pretrained("unused")
    assert "cleanup failure" in caught.value.__notes__[0]
