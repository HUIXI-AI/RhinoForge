"""Host-only admission and audio-boundary checks for Whisper base."""

import numpy as np
import pytest
import torch

from rpu_backend.api import WhisperASRModel, WhisperTranscription
from rpu_backend.api import whisper_asr
from rpu_backend.api.errors import RPUUnsupportedDtypeError, UnsupportedModelError


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
