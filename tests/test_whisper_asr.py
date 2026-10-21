"""Host-only admission and audio-boundary checks for Whisper base."""

import numpy as np
import pytest
import torch

from rpu_backend.api import WhisperASRModel, WhisperTranscription
from rpu_backend.api import whisper_asr
from rpu_backend.api.errors import RPUUnsupportedDtypeError, UnsupportedModelError
from rpu_backend.api.errors import RPUSingleHandleError


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, "fp16"])
def test_dtype_rejected_before_checkpoint_access(tmp_path, monkeypatch, dtype):
    monkeypatch.setattr(whisper_asr, "validate_checkpoint",
                        lambda path: pytest.fail("checkpoint accessed"))
    with pytest.raises(RPUUnsupportedDtypeError):
        WhisperASRModel.from_pretrained(tmp_path / "missing", dtype=dtype)


def test_unreviewed_checkpoint_rejected_before_native_access(tmp_path, monkeypatch):
    monkeypatch.setattr(whisper_asr, "_require_native_backend",
                        lambda: pytest.fail("native backend accessed"))
    with pytest.raises(UnsupportedModelError, match="checkpoint"):
        WhisperASRModel.from_pretrained(tmp_path)


def test_native_backend_absence_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(whisper_asr, "validate_checkpoint", lambda path: {})
    with pytest.raises(UnsupportedModelError, match="native backend"):
        WhisperASRModel.from_pretrained(tmp_path)


def test_from_pretrained_loads_and_owns_both_runtimes(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in (
        "base-encoder.onnx", "base-decoder.onnx", "base-tokens.txt"
    )}
    created = []

    class Runtime:
        def __init__(self, *args):
            self.args = args
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

    tokens = object()
    monkeypatch.setattr(whisper_asr, "validate_checkpoint", lambda path: paths)
    monkeypatch.setattr(whisper_asr, "_require_native_backend", lambda: None)
    monkeypatch.setattr(whisper_asr, "WhisperEncoderRuntime", Runtime)
    monkeypatch.setattr(whisper_asr, "WhisperDecoderRuntime", Runtime)
    monkeypatch.setattr(whisper_asr._TokenTable, "load", lambda path: tokens)
    allocator = []
    monkeypatch.setattr(torch.rpu, "set_caching_allocator", allocator.append)

    model = WhisperASRModel.from_pretrained(tmp_path)
    model.close()

    assert model._tokens is tokens
    assert [runtime.args for runtime in created] == [
        (paths["base-encoder.onnx"],),
        (paths["base-encoder.onnx"], paths["base-decoder.onnx"]),
    ]
    assert all(runtime.closed for runtime in created)
    assert allocator == [True]


def test_from_pretrained_closes_encoder_when_decoder_loading_fails(
        tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in (
        "base-encoder.onnx", "base-decoder.onnx", "base-tokens.txt"
    )}

    class Encoder:
        def __init__(self, path):
            self.closed = False
            Encoder.instance = self

        def close(self):
            self.closed = True

    monkeypatch.setattr(whisper_asr, "validate_checkpoint", lambda path: paths)
    monkeypatch.setattr(whisper_asr, "_require_native_backend", lambda: None)
    monkeypatch.setattr(whisper_asr, "WhisperEncoderRuntime", Encoder)
    monkeypatch.setattr(
        whisper_asr, "WhisperDecoderRuntime",
        lambda *args: (_ for _ in ()).throw(RuntimeError("decoder load")),
    )

    with pytest.raises(RuntimeError, match="decoder load"):
        WhisperASRModel.from_pretrained(tmp_path)
    assert Encoder.instance.closed


def test_from_pretrained_preserves_load_error_when_cleanup_fails(
        tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in (
        "base-encoder.onnx", "base-decoder.onnx", "base-tokens.txt"
    )}
    created = []

    class Runtime:
        def __init__(self, *args):
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True
            if len(created) == 2 and self is created[1]:
                raise RuntimeError("decoder cleanup")

    monkeypatch.setattr(whisper_asr, "validate_checkpoint", lambda path: paths)
    monkeypatch.setattr(whisper_asr, "_require_native_backend", lambda: None)
    monkeypatch.setattr(whisper_asr, "WhisperEncoderRuntime", Runtime)
    monkeypatch.setattr(whisper_asr, "WhisperDecoderRuntime", Runtime)
    monkeypatch.setattr(whisper_asr._TokenTable, "load",
                        lambda path: (_ for _ in ()).throw(ValueError("token load")))
    monkeypatch.setattr(torch.rpu, "set_caching_allocator", lambda enabled: None)

    with pytest.raises(ValueError, match="token load") as caught:
        WhisperASRModel.from_pretrained(tmp_path)

    assert all(runtime.closed for runtime in created)
    assert any("decoder cleanup" in note for note in caught.value.__notes__)


def test_from_pretrained_rejects_a_second_live_model(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in (
        "base-encoder.onnx", "base-decoder.onnx", "base-tokens.txt"
    )}

    class Runtime:
        def __init__(self, *args):
            pass

        def close(self):
            pass

    monkeypatch.setattr(whisper_asr, "validate_checkpoint", lambda path: paths)
    monkeypatch.setattr(whisper_asr, "_require_native_backend", lambda: None)
    monkeypatch.setattr(whisper_asr, "WhisperEncoderRuntime", Runtime)
    monkeypatch.setattr(whisper_asr, "WhisperDecoderRuntime", Runtime)
    monkeypatch.setattr(whisper_asr._TokenTable, "load", lambda path: object())
    first = WhisperASRModel.from_pretrained(tmp_path)
    try:
        with pytest.raises(RPUSingleHandleError):
            WhisperASRModel.from_pretrained(tmp_path)
    finally:
        first.close()


@pytest.mark.parametrize("audio,rate,tokens", [
    (np.array([], dtype=np.float32), 16000, 448),
    (np.zeros(16000, dtype=np.float32), 16000, 448),
    (np.ones(480001, dtype=np.float32), 16000, 448),
    (np.ones((1, 16000), dtype=np.float32), 16000, 448),
    (np.ones(16000, dtype=np.int16), 16000, 448),
    (np.full(16000, np.nan), 16000, 448),
    (np.full(16000, np.inf), 16000, 448),
    (np.ones(16000, dtype=np.float32), 0, 448),
    (np.ones(16000, dtype=np.float32), True, 448),
    (np.ones(16000, dtype=np.float32), 16000, 449),
    (np.ones(16000, dtype=np.float32), 16000, True),
    (np.ones(16000, dtype=np.float32), 16000, 1.5),
])
def test_invalid_audio_rejected_without_inference(audio, rate, tokens):
    with pytest.raises(ValueError):
        whisper_asr._validate_audio(audio, rate, tokens)


@pytest.mark.parametrize("rate,samples", [(16000, 1), (16000, 480000), (8000, 240000)])
def test_valid_audio_boundaries(rate, samples):
    audio = torch.ones(samples, dtype=torch.float16)
    value = whisper_asr._validate_audio(audio, rate, 448)
    assert value.shape == (samples,)
    assert value.dtype == np.float32


def test_public_result_is_immutable():
    result = WhisperTranscription("hello", "en", (1, 2), "eos")
    with pytest.raises(AttributeError):
        result.text = "changed"


def test_closed_model_rejects_valid_audio():
    model = WhisperASRModel()
    with pytest.raises(RuntimeError, match="closed"):
        model.transcribe(np.ones(16000, dtype=np.float32))
    model.close()
