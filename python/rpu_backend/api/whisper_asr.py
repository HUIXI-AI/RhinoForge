"""Public admission and audio contract for the frozen Whisper base profile."""

from __future__ import annotations

import operator
import threading
from dataclasses import dataclass

import numpy as np
import torch

from rpu_backend.adapters.whisper import validate_checkpoint

from .errors import RPUUnsupportedDtypeError, UnsupportedModelError


_NATIVE_OPS = (
    "whisper_encoder_create",
    "whisper_encoder_destroy",
    "whisper_encoder_set_weights",
    "whisper_encoder_forward",
    "whisper_decoder_create",
    "whisper_decoder_destroy",
    "whisper_decoder_set_weights",
    "whisper_decoder_prefill",
    "whisper_decoder_decode",
)


@dataclass(frozen=True)
class WhisperTranscription:
    text: str
    language: str
    token_ids: tuple[int, ...]
    finish_reason: str


def _validate_audio(audio, sample_rate: int, max_new_tokens: int) -> np.ndarray:
    if isinstance(sample_rate, bool):
        raise ValueError("sample_rate must be a positive integer")
    try:
        rate = operator.index(sample_rate)
    except TypeError as exc:
        raise ValueError("sample_rate must be a positive integer") from exc
    if rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if isinstance(max_new_tokens, bool):
        raise ValueError("max_new_tokens must be an integer in [1, 448]")
    try:
        count = operator.index(max_new_tokens)
    except TypeError as exc:
        raise ValueError("max_new_tokens must be an integer in [1, 448]") from exc
    if not 1 <= count <= 448:
        raise ValueError("max_new_tokens must be an integer in [1, 448]")
    if torch.is_tensor(audio):
        if audio.device.type != "cpu" or not audio.is_floating_point():
            raise ValueError("audio tensor must contain CPU floating-point samples")
        audio = audio.detach().float().numpy()
    waveform = np.asarray(audio)
    if waveform.ndim != 1 or waveform.dtype.kind != "f":
        raise ValueError("audio must be one mono floating-point waveform")
    if waveform.size == 0 or waveform.size > rate * 30 or not np.isfinite(waveform).all():
        raise ValueError("audio must be finite, non-empty and at most 30 seconds")
    waveform = np.asarray(waveform, dtype=np.float32)
    if not np.isfinite(waveform).all() or not np.any(waveform):
        raise ValueError("audio must remain finite and must not be silent")
    return waveform


def _require_native_backend() -> None:
    missing = [name for name in _NATIVE_OPS if not hasattr(torch.ops.rpu, name)]
    if missing:
        raise UnsupportedModelError(
            "Whisper native backend is not available: " + ", ".join(missing))


class WhisperASRModel:
    """One frozen Whisper base FP16 model; native execution is added in later stages."""

    def __init__(self):
        self._lock = threading.Lock()
        self._closed = True

    @classmethod
    def from_pretrained(cls, checkpoint, *, dtype=torch.float16) -> "WhisperASRModel":
        if dtype is not torch.float16:
            raise RPUUnsupportedDtypeError("Whisper base supports only torch.float16")
        validate_checkpoint(checkpoint)
        _require_native_backend()
        raise UnsupportedModelError("Whisper native backend loading is not implemented")

    def transcribe(self, audio, *, sample_rate=16000, max_new_tokens=448) -> WhisperTranscription:
        """Validate one waveform before any RPU access."""
        _validate_audio(audio, sample_rate, max_new_tokens)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Whisper requests and close must be serial")
        try:
            if self._closed:
                raise RuntimeError("Whisper model is closed")
            raise UnsupportedModelError("Whisper native backend loading is not implemented")
        finally:
            self._lock.release()

    def close(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Whisper requests and close must be serial")
        try:
            self._closed = True
        finally:
            self._lock.release()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Whisper model is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
