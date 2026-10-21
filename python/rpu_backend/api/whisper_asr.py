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
_MAX_AUDIO_SECONDS = 30 * 60
_STREAM_WINDOW_SECONDS = 28
_STREAM_OVERLAP_SECONDS = 2
_SILENCE_FRAME_SAMPLES = 320
_SILENCE_FRAMES = 50
_SILENCE_RMS = 10 ** (-40 / 20)
# shortcut: fixed RMS VAD and exact token overlap; upgrade for noisy or continuous speech.
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


@dataclass(frozen=True)
class WhisperStreamingTranscription:
    confirmed_text: str
    partial_text: str
    language: str | None
    audio_seconds: float
    is_final: bool


def _validate_audio(
    audio, sample_rate: int, max_new_tokens: int, *, max_seconds: int = 30
) -> np.ndarray:
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
    if waveform.size == 0 or waveform.size > rate * max_seconds or not np.isfinite(waveform).all():
        limit = "30 minutes" if max_seconds == _MAX_AUDIO_SECONDS else f"{max_seconds} seconds"
        raise ValueError(f"audio must be finite, non-empty and at most {limit}")
    waveform = np.asarray(waveform, dtype=np.float32)
    if not np.isfinite(waveform).all() or not np.any(waveform):
        raise ValueError("audio must remain finite and must not be silent")
    return waveform


def _validate_audio_chunk(audio) -> np.ndarray:
    if torch.is_tensor(audio):
        if audio.device.type != "cpu" or not audio.is_floating_point():
            raise ValueError("audio tensor must contain CPU floating-point samples")
        audio = audio.detach().float().numpy()
    waveform = np.asarray(audio)
    if (waveform.ndim != 1 or waveform.dtype.kind != "f" or waveform.size == 0
            or not np.isfinite(waveform).all()):
        raise ValueError("audio chunk must be one non-empty finite floating-point waveform")
    return np.asarray(waveform, dtype=np.float32)


def _to_16khz(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    return _resample(torch.from_numpy(waveform), sample_rate).numpy()


def _speech_frames(samples: np.ndarray) -> np.ndarray:
    frame_count = samples.size // _SILENCE_FRAME_SAMPLES
    frames = samples[:frame_count * _SILENCE_FRAME_SAMPLES].reshape(
        frame_count, _SILENCE_FRAME_SAMPLES
    )
    power = np.einsum("ij,ij->i", frames, frames, dtype=np.float64)
    return power >= _SILENCE_RMS ** 2 * _SILENCE_FRAME_SAMPLES


def _speech_state(waveform: np.ndarray, sample_rate: int) -> tuple[bool, int | None]:
    samples = _to_16khz(waveform, sample_rate)
    speech = _speech_frames(samples)
    if speech.size:
        has_speech = False
        silence_run = 0
        for index, active in enumerate(speech):
            if active:
                has_speech = True
                silence_run = 0
            elif has_speech:
                silence_run += 1
                if silence_run == _SILENCE_FRAMES:
                    boundary_16k = (index + 1) * _SILENCE_FRAME_SAMPLES
                    boundary = round(boundary_16k * sample_rate / 16000)
                    return True, min(boundary, waveform.size)
        return has_speech, None
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    return rms >= _SILENCE_RMS, None


def _long_audio_segments(waveform: np.ndarray):
    frame = _SILENCE_FRAME_SAMPLES
    speech = _speech_frames(waveform)
    start = 0
    overlaps_previous = False
    window = _STREAM_WINDOW_SECONDS * 16000
    overlap = _STREAM_OVERLAP_SECONDS * 16000
    while start < waveform.size:
        frame_index = start // frame
        if not overlaps_previous:
            speech_index = frame_index
            while speech_index < speech.size and not speech[speech_index]:
                speech_index += 1
            if speech_index >= speech.size:
                break
            first_speech = speech_index * frame
            start = max(start, first_speech // 16000 * 16000)
            frame_index = start // frame
        limit = min(start + window, waveform.size)
        silence_run = 0
        has_speech = False
        natural_end = None
        for index in range(frame_index, min(speech.size, limit // frame)):
            if speech[index]:
                has_speech = True
                silence_run = 0
            elif has_speech:
                silence_run += 1
            if has_speech and silence_run == _SILENCE_FRAMES:
                natural_end = (index + 1) * frame
                break
        if natural_end is not None:
            yield waveform[start:natural_end], overlaps_previous
            start = natural_end
            overlaps_previous = False
        elif limit < waveform.size:
            yield waveform[start:limit], overlaps_previous
            start = limit - overlap
            overlaps_previous = True
        else:
            yield waveform[start:], overlaps_previous
            break


def _trim_token_overlap(existing: tuple[int, ...], incoming: tuple[int, ...]) -> tuple[int, ...]:
    for count in range(min(len(existing), len(incoming)), 0, -1):
        if existing[-count:] == incoming[:count]:
            return incoming[count:]
    return incoming


def _aggregate_timings(rows: list[dict], total_ms: float) -> dict:
    names = ("preprocess_ms", "encoder_ms", "cross_ms", "language_ms",
             "prompt_ms", "decode_ms")
    timing = {name: sum(row[name] for row in rows) for name in names}
    timing["decode_step_ms"] = tuple(
        value for row in rows for value in row["decode_step_ms"]
    )
    timing["total_ms"] = total_ms
    return timing


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
        """Transcribe one finite waveform of at most 30 minutes."""
        waveform = _validate_audio(
            audio, sample_rate, max_new_tokens, max_seconds=_MAX_AUDIO_SECONDS
        )
        rate = operator.index(sample_rate)
        count = operator.index(max_new_tokens)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Whisper requests and close must be serial")
        try:
            if self._closed:
                raise RuntimeError("Whisper model is closed")
            with torch.inference_mode():
                total_start = time.perf_counter()
                if waveform.size <= rate * 30:
                    result, timing = self._run_one_locked(
                        waveform, rate, count
                    )
                else:
                    result, timing = self._run_long_locked(
                        _to_16khz(waveform, rate), count, total_start
                    )
                timing["total_ms"] = (time.perf_counter() - total_start) * 1000
                self._last_timing = timing
                return result
        finally:
            self._lock.release()

    def create_stream(
        self, *, sample_rate=16000, max_new_tokens=448
    ) -> "WhisperASRStream":
        """Create one host-side streaming session backed by this model."""
        _validate_audio(
            np.ones(1, dtype=np.float32), sample_rate, max_new_tokens,
            max_seconds=_MAX_AUDIO_SECONDS,
        )
        if self._closed:
            raise RuntimeError("Whisper model is closed")
        return WhisperASRStream(
            self, operator.index(sample_rate), operator.index(max_new_tokens)
        )

    def _run_one_locked(
        self, waveform: np.ndarray, sample_rate: int, max_new_tokens: int,
        language_index: int | None = None,
    ) -> tuple[WhisperTranscription, dict]:
        total_start = time.perf_counter()
        started = total_start
        mel = _log_mel_spectrogram(waveform, sample_rate)
        timing = {"preprocess_ms": (time.perf_counter() - started) * 1000}
        started = time.perf_counter()
        hidden = self._encoder.encode(mel)
        timing["encoder_ms"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        self._decoder.prepare_audio(hidden)
        timing["cross_ms"] = (time.perf_counter() - started) * 1000
        result = self._generate(max_new_tokens, timing, language_index)
        timing["total_ms"] = (time.perf_counter() - total_start) * 1000
        return result, timing

    def _run_long_locked(
        self, waveform: np.ndarray, max_new_tokens: int, total_start: float
    ) -> tuple[WhisperTranscription, dict]:
        tokens: tuple[int, ...] = ()
        language_index = None
        finish_reason = "eos"
        timings = []
        for segment, overlaps_previous in _long_audio_segments(waveform):
            result, timing = self._run_one_locked(
                segment, 16000, max_new_tokens, language_index
            )
            timings.append(timing)
            incoming = result.token_ids
            if overlaps_previous:
                incoming = _trim_token_overlap(tokens, incoming)
            tokens += incoming
            if language_index is None:
                language_index = _LANGUAGE_CODES.index(result.language)
            if result.finish_reason == "length":
                finish_reason = "length"
        if language_index is None:
            raise RuntimeError("Whisper long-form segmentation found no speech")
        timing = _aggregate_timings(
            timings, (time.perf_counter() - total_start) * 1000
        )
        return WhisperTranscription(
            self._tokens.decode(tokens).strip(),
            _LANGUAGE_CODES[language_index],
            tokens,
            finish_reason,
        ), timing

    def _transcribe_stream_segment(
        self, waveform: np.ndarray, sample_rate: int, max_new_tokens: int,
        language_index: int | None,
    ) -> WhisperTranscription:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Whisper requests and close must be serial")
        try:
            if self._closed:
                raise RuntimeError("Whisper model is closed")
            with torch.inference_mode():
                result, timing = self._run_one_locked(
                    waveform, sample_rate, max_new_tokens, language_index
                )
                self._last_timing = timing
                return result
        finally:
            self._lock.release()

    def _generate(
        self, max_new_tokens: int, timing: dict, language_index: int | None = None
    ) -> WhisperTranscription:
        started = time.perf_counter()
        language_logits = self._decoder.prefill(_SOT)[0, _SOT + 1:_TRANSLATE]
        if not torch.isfinite(language_logits).all():
            raise RuntimeError("Whisper language logits are not finite")
        if language_index is None:
            language_index = int(torch.argmax(language_logits))
        elif not 0 <= language_index < len(_LANGUAGE_CODES):
            raise ValueError("Whisper language index is out of range")
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


class WhisperASRStream:
    """One bounded host-side streaming session for a resident Whisper model."""

    def __init__(self, model: WhisperASRModel, sample_rate: int, max_new_tokens: int):
        self._model = model
        self._sample_rate = sample_rate
        self._max_new_tokens = max_new_tokens
        self._queued = np.empty(0, dtype=np.float32)
        self._current = np.empty(0, dtype=np.float32)
        self._confirmed_tokens: tuple[int, ...] = ()
        self._language_index = None
        self._received_samples = 0
        self._processed_samples = 0
        self._overlaps_previous = False
        self._finished = False

    def accept_audio(self, audio) -> tuple[WhisperStreamingTranscription, ...]:
        self._require_active()
        waveform = _validate_audio_chunk(audio)
        limit = self._sample_rate * _MAX_AUDIO_SECONDS
        if self._received_samples + waveform.size > limit:
            raise ValueError("stream audio must be at most 30 minutes")
        self._received_samples += waveform.size
        self._queued = np.concatenate((self._queued, waveform))
        updates = []
        while self._queued.size >= self._sample_rate:
            second = self._queued[:self._sample_rate]
            self._queued = self._queued[self._sample_rate:].copy()
            self._current = np.concatenate((self._current, second))
            self._processed_samples += self._sample_rate
            updates.append(self._update(self._processed_samples / self._sample_rate))
        return tuple(updates)

    def finish(self) -> WhisperStreamingTranscription:
        self._require_active()
        if self._queued.size:
            self._current = np.concatenate((self._current, self._queued))
            self._processed_samples += self._queued.size
            self._queued = np.empty(0, dtype=np.float32)
        has_speech, _ = (
            _speech_state(self._current, self._sample_rate)
            if self._current.size else (False, False)
        )
        while has_speech:
            _, boundary = _speech_state(self._current, self._sample_rate)
            segment = self._current if boundary is None else self._current[:boundary]
            result = self._infer(segment)
            incoming = self._deduplicate(result.token_ids)
            self._confirmed_tokens += incoming
            self._pin_language(result.language)
            if boundary is None:
                self._current = np.empty(0, dtype=np.float32)
            else:
                self._current = self._current[boundary:].copy()
            self._overlaps_previous = False
            has_speech, _ = (
                _speech_state(self._current, self._sample_rate)
                if self._current.size else (False, None)
            )
        self._current = np.empty(0, dtype=np.float32)
        self._finished = True
        return self._snapshot((), None, self._received_samples / self._sample_rate, True)

    def _update(self, audio_seconds: float) -> WhisperStreamingTranscription:
        has_speech, silence_boundary = _speech_state(
            self._current, self._sample_rate
        )
        if not has_speech:
            self._current = np.empty(0, dtype=np.float32)
            return self._snapshot((), None, audio_seconds, False)

        if silence_boundary is not None:
            result = self._infer(self._current[:silence_boundary])
            incoming = self._deduplicate(result.token_ids)
            self._confirmed_tokens += incoming
            self._pin_language(result.language)
            self._current = self._current[silence_boundary:].copy()
            self._overlaps_previous = False
            remainder_has_speech, _ = (
                _speech_state(self._current, self._sample_rate)
                if self._current.size else (False, None)
            )
            if not remainder_has_speech:
                self._current = np.empty(0, dtype=np.float32)
                return self._snapshot((), None, audio_seconds, False)
            result = self._infer(self._current)
            return self._snapshot(
                result.token_ids, result.language, audio_seconds, False
            )

        result = self._infer(self._current)
        incoming = self._deduplicate(result.token_ids)
        language = result.language
        if self._current.size >= self._sample_rate * _STREAM_WINDOW_SECONDS:
            self._confirmed_tokens += incoming
            self._pin_language(language)
            keep = self._sample_rate * _STREAM_OVERLAP_SECONDS
            self._current = self._current[-keep:].copy()
            self._overlaps_previous = True
            incoming = ()
        return self._snapshot(incoming, language, audio_seconds, False)

    def _infer(self, waveform: np.ndarray) -> WhisperTranscription:
        return self._model._transcribe_stream_segment(
            waveform,
            self._sample_rate,
            self._max_new_tokens,
            self._language_index,
        )

    def _deduplicate(self, tokens: tuple[int, ...]) -> tuple[int, ...]:
        if self._overlaps_previous:
            return _trim_token_overlap(self._confirmed_tokens, tokens)
        return tokens

    def _pin_language(self, language: str) -> None:
        if self._language_index is None:
            self._language_index = _LANGUAGE_CODES.index(language)

    def _snapshot(
        self, partial_tokens: tuple[int, ...], language: str | None,
        audio_seconds: float, is_final: bool,
    ) -> WhisperStreamingTranscription:
        confirmed = self._model._tokens.decode(self._confirmed_tokens).strip()
        combined = self._model._tokens.decode(
            self._confirmed_tokens + partial_tokens
        ).strip()
        if combined.startswith(confirmed):
            partial = combined[len(confirmed):]
        else:
            value = self._model._tokens.decode(partial_tokens).strip()
            partial = (" " if confirmed and value else "") + value
        current_language = (
            _LANGUAGE_CODES[self._language_index]
            if self._language_index is not None else language
        )
        return WhisperStreamingTranscription(
            confirmed, partial, current_language, audio_seconds, is_final
        )

    def _require_active(self) -> None:
        if self._finished:
            raise RuntimeError("Whisper stream is already finished")
        if self._model._closed:
            raise RuntimeError("Whisper model is closed")
