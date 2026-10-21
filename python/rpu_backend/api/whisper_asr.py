"""Public admission and audio contract for the frozen Whisper base profile."""

from __future__ import annotations

import base64
import operator
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rpu_backend.adapters.whisper import validate_checkpoint
from rpu_backend.adapters.whisper.runtime import (
    WhisperDecoderRuntime,
    WhisperEncoderRuntime,
)

from .errors import RPUUnsupportedDtypeError, UnsupportedModelError


_NATIVE_OPS = (
    "whisper_encoder_create",
    "whisper_encoder_destroy",
    "whisper_encoder_set_weights",
    "whisper_encoder_set_final_norm",
    "whisper_encoder_set_frontend",
    "whisper_encoder_set_chunk_envelope",
    "whisper_encoder_resolve_stage_domain",
    "whisper_encoder_forward",
    "whisper_decoder_create",
    "whisper_decoder_destroy",
    "whisper_decoder_set_weights",
    "whisper_decoder_set_cross_weights",
    "whisper_decoder_set_io",
    "whisper_decoder_reset",
    "whisper_decoder_prepare_cross",
    "whisper_decoder_prefill_logits",
    "whisper_decoder_decode_logits",
    "gather_embedding",
    "spm_alloc_reset_temporary",
)

_EOT = 50257
_SOT = 50258
_BLANK = 220
_TRANSLATE = 50358
_TRANSCRIBE = 50359
_NO_SPEECH = 50362
_NO_TIMESTAMPS = 50363
_LANGUAGE_CODES = (
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl",
    "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk",
    "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr",
    "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn",
    "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne",
    "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn",
    "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi",
    "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my",
    "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
)
assert len(_LANGUAGE_CODES) == 99


@dataclass(frozen=True)
class _TokenTable:
    pieces: tuple[bytes, ...]

    @classmethod
    def load(cls, path: str | Path, *, expected_size: int = 50257) -> "_TokenTable":
        pieces: list[bytes | None] = [None] * expected_size
        try:
            with Path(path).open(encoding="utf-8") as stream:
                for line in stream:
                    encoded, raw_index = line.split()
                    index = int(raw_index)
                    if not 0 <= index < expected_size or pieces[index] is not None:
                        raise ValueError
                    pieces[index] = base64.b64decode(encoded)
            if any(piece is None for piece in pieces):
                raise ValueError
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError("Whisper token table is malformed") from exc
        return cls(tuple(piece for piece in pieces if piece is not None))

    def decode(self, token_ids) -> str:
        value = b"".join(
            self.pieces[token_id]
            for token_id in token_ids
            if 0 <= token_id < len(self.pieces)
        )
        return value.decode("utf-8", errors="replace")


def _hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
    mels = frequencies / (200.0 / 3.0)
    logarithmic = frequencies >= 1000.0
    mels[logarithmic] = 15.0 + np.log(frequencies[logarithmic] / 1000.0) / (
        np.log(6.4) / 27.0
    )
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    frequencies = (200.0 / 3.0) * mels
    logarithmic = mels >= 15.0
    frequencies[logarithmic] = 1000.0 * np.exp(
        (np.log(6.4) / 27.0) * (mels[logarithmic] - 15.0)
    )
    return frequencies


def _mel_filterbank() -> torch.Tensor:
    frequencies = np.linspace(0.0, 8000.0, 201, dtype=np.float32)
    mel_points = np.linspace(
        _hz_to_mel(np.array([0.0], dtype=np.float32))[0],
        _hz_to_mel(np.array([8000.0], dtype=np.float32))[0],
        82,
        dtype=np.float32,
    )
    mel_frequencies = _mel_to_hz(mel_points)
    differences = np.diff(mel_frequencies)
    ramps = mel_frequencies[:, None] - frequencies[None, :]
    lower = -ramps[:-2] / differences[:-1, None]
    upper = ramps[2:] / differences[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_frequencies[2:] - mel_frequencies[:-2]))[:, None]
    return torch.from_numpy(weights)


_MEL_FILTERBANK = _mel_filterbank()


def _resample(samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if sample_rate == 16000:
        return samples
    ratio = 16000 / sample_rate
    output = torch.empty(max(1, round(samples.numel() * ratio)), dtype=torch.float32)
    offsets = torch.arange(-31, 33)
    cutoff = min(1.0, ratio)
    for start in range(0, output.numel(), 32768):
        stop = min(start + 32768, output.numel())
        positions = torch.arange(start, stop, dtype=torch.float32) / ratio
        indices = positions.floor().long()[:, None] + offsets
        distance = positions[:, None] - indices
        weights = cutoff * torch.sinc(cutoff * distance)
        weights *= 0.5 + 0.5 * torch.cos(torch.pi * distance / 32)
        valid = (indices >= 0) & (indices < samples.numel())
        weights *= valid
        weights /= weights.sum(dim=1, keepdim=True)
        output[start:stop] = (
            samples[indices.clamp(0, samples.numel() - 1)] * weights
        ).sum(dim=1)
    return output


def _log_mel_spectrogram(waveform: np.ndarray, sample_rate: int) -> torch.Tensor:
    samples = torch.from_numpy(waveform).float()
    samples = _resample(samples, sample_rate)
    samples = F.pad(samples, (0, max(0, 160 - samples.numel())))
    window = torch.hann_window(400)
    spectrum = torch.stft(
        samples,
        400,
        160,
        window=window,
        return_complex=True,
        pad_mode="reflect" if samples.numel() > 200 else "constant",
    ).abs().square()[:, :-1]
    mel = _MEL_FILTERBANK @ spectrum
    mel = mel.clamp_min(1e-10).log10()
    mel = torch.maximum(mel, mel.max() - 8.0)
    mel = (mel + 4.0) / 4.0
    mel = F.pad(mel, (0, 1500))
    if mel.shape[1] > 3000:
        mel = F.pad(mel[:, :2950], (0, 50))
    else:
        mel = F.pad(mel, (0, 3000 - mel.shape[1]))
    return mel.unsqueeze(0).half().contiguous()


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
    """One frozen Whisper base FP16 model with host-side greedy generation."""

    def __init__(self, encoder=None, decoder=None, tokens=None):
        self._lock = threading.Lock()
        self._encoder = encoder
        self._decoder = decoder
        self._tokens = tokens
        self._last_timing = None
        self._closed = encoder is None or decoder is None or tokens is None

    @classmethod
    def from_pretrained(cls, checkpoint, *, dtype=torch.float16) -> "WhisperASRModel":
        from .causal_lm import _claim_live_instance, _release_live_instance

        if dtype is not torch.float16:
            raise RPUUnsupportedDtypeError("Whisper base supports only torch.float16")
        paths = validate_checkpoint(checkpoint)
        _require_native_backend()
        model = cls()
        _claim_live_instance(model)
        encoder = None
        decoder = None
        try:
            torch.rpu.set_caching_allocator(True)
            encoder = WhisperEncoderRuntime(paths["base-encoder.onnx"])
            decoder = WhisperDecoderRuntime(
                paths["base-encoder.onnx"], paths["base-decoder.onnx"]
            )
            tokens = _TokenTable.load(paths["base-tokens.txt"])
        except BaseException as error:
            for name, runtime in (("decoder", decoder), ("encoder", encoder)):
                if runtime is None:
                    continue
                try:
                    runtime.close()
                except BaseException as cleanup_error:
                    error.add_note(
                        f"Whisper {name} cleanup failed: {cleanup_error!r}"
                    )
            try:
                _release_live_instance(model)
            except BaseException as cleanup_error:
                error.add_note(
                    f"Whisper ownership cleanup failed: {cleanup_error!r}"
                )
            raise
        model._encoder, model._decoder, model._tokens = encoder, decoder, tokens
        model._closed = False
        return model

    def transcribe(self, audio, *, sample_rate=16000, max_new_tokens=448) -> WhisperTranscription:
        """Validate one waveform before any RPU access."""
        waveform = _validate_audio(audio, sample_rate, max_new_tokens)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Whisper requests and close must be serial")
        try:
            if self._closed:
                raise RuntimeError("Whisper model is closed")
            with torch.inference_mode():
                total_start = time.perf_counter()
                started = total_start
                mel = _log_mel_spectrogram(waveform, operator.index(sample_rate))
                timing = {"preprocess_ms": (time.perf_counter() - started) * 1000}
                started = time.perf_counter()
                hidden = self._encoder.encode(
                    mel
                )
                timing["encoder_ms"] = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                self._decoder.prepare_audio(hidden)
                timing["cross_ms"] = (time.perf_counter() - started) * 1000
                result = self._generate(operator.index(max_new_tokens), timing)
                timing["total_ms"] = (time.perf_counter() - total_start) * 1000
                self._last_timing = timing
                return result
        finally:
            self._lock.release()

    def _generate(self, max_new_tokens: int, timing: dict) -> WhisperTranscription:
        started = time.perf_counter()
        language_logits = self._decoder.prefill(_SOT)[0, _SOT + 1:_TRANSLATE]
        if not torch.isfinite(language_logits).all():
            raise RuntimeError("Whisper language logits are not finite")
        language_index = int(torch.argmax(language_logits))
        language_token = _SOT + 1 + language_index
        timing["language_ms"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        self._decoder.decode(language_token)
        self._decoder.decode(_TRANSCRIBE)
        logits = self._decoder.decode(_NO_TIMESTAMPS)
        timing["prompt_ms"] = (time.perf_counter() - started) * 1000

        generated = []
        decode_steps = []
        finish_reason = "length"
        for step in range(min(max_new_tokens, 448 - 4)):
            scores = logits[0].clone()
            if not torch.isfinite(scores).all():
                raise RuntimeError("Whisper decoder logits are not finite")
            scores[[_NO_TIMESTAMPS, _SOT, _NO_SPEECH, _TRANSLATE]] = -torch.inf
            if step == 0:
                scores[[_EOT, _BLANK]] = -torch.inf
            token_id = int(torch.argmax(scores))
            if token_id == _EOT:
                finish_reason = "eos"
                break
            generated.append(token_id)
            if step + 1 < min(max_new_tokens, 448 - 4):
                started = time.perf_counter()
                logits = self._decoder.decode(token_id)
                decode_steps.append((time.perf_counter() - started) * 1000)
        timing["decode_step_ms"] = tuple(decode_steps)
        timing["decode_ms"] = sum(decode_steps)
        return WhisperTranscription(
            self._tokens.decode(generated).strip(),
            _LANGUAGE_CODES[language_index],
            tuple(generated),
            finish_reason,
        )

    def close(self) -> None:
        from .causal_lm import _release_live_instance

        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Whisper requests and close must be serial")
        try:
            if self._closed:
                return
            try:
                self._decoder.close()
            finally:
                self._encoder.close()
            self._closed = True
            _release_live_instance(self)
        finally:
            self._lock.release()

    def execution_stats(self) -> dict:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Whisper requests, statistics and close must be serial")
        try:
            if self._closed:
                raise RuntimeError("Whisper model is closed")
            encoder = self._encoder.execution_stats()
            decoder = self._decoder.execution_stats()
            return {
                "encoder": encoder,
                **decoder,
                "timing": dict(self._last_timing or {}),
                "graph_invariant_ok": encoder["invariant_ok"] and all(
                    stage["invariant_ok"] for stage in decoder.values()
                ),
            }
        finally:
            self._lock.release()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Whisper model is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
