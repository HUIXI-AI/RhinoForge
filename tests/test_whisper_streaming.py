"""Long-form and streaming contracts for the frozen Whisper base profile."""

from types import MethodType

import numpy as np
import pytest

from rpu_backend.api import (
    WhisperASRModel,
    WhisperStreamingTranscription,
    WhisperTranscription,
)
from rpu_backend.api import whisper_asr


class _Runtime:
    def close(self):
        pass


def _model(pieces):
    return WhisperASRModel(
        _Runtime(), _Runtime(), whisper_asr._TokenTable(tuple(pieces))
    )


def _result(token_id, language="en"):
    return WhisperTranscription("unused", language, (token_id,), "eos")


def test_stream_updates_each_audio_second_and_confirms_after_silence():
    model = _model((b" first", b" revised", b" final", b" next"))
    responses = iter((
        _result(0, "en"),
        _result(1, "zh"),
        _result(2, "en"),
        _result(3, "en"),
        _result(3, "en"),
    ))
    calls = []

    def infer(self, waveform, sample_rate, max_new_tokens, language_index):
        calls.append((len(waveform), sample_rate, max_new_tokens, language_index))
        return next(responses)

    model._transcribe_stream_segment = MethodType(infer, model)
    stream = model.create_stream(sample_rate=16000)

    assert stream.accept_audio(np.full(6400, 0.1, np.float32)) == ()
    first = stream.accept_audio(np.full(9600, 0.1, np.float32))
    second = stream.accept_audio(np.full(16000, 0.1, np.float32))
    confirmed = stream.accept_audio(np.zeros(16000, np.float32))
    following = stream.accept_audio(np.full(16000, 0.1, np.float32))
    final = stream.finish()

    assert first[0].partial_text == "first" and first[0].language == "en"
    assert second[0].partial_text == "revised" and second[0].language == "zh"
    assert confirmed[0].confirmed_text == "final"
    assert confirmed[0].partial_text == "" and confirmed[0].language == "en"
    assert following[0].confirmed_text == "final"
    assert following[0].partial_text == " next"
    assert calls[-2][-1] == calls[-1][-1] == 0
    assert final.confirmed_text == "final next" and final.partial_text == ""
    assert final.is_final and final.audio_seconds == 4.0
    with pytest.raises(RuntimeError, match="finished"):
        stream.accept_audio(np.ones(1, np.float32))
    model.close()


def test_stream_large_input_returns_every_intermediate_update():
    model = _model((b" speech",))
    calls = []

    def infer(self, waveform, sample_rate, max_new_tokens, language_index):
        calls.append(len(waveform))
        return _result(0)

    model._transcribe_stream_segment = MethodType(infer, model)
    stream = model.create_stream(sample_rate=16000)

    updates = stream.accept_audio(np.full(32000, 0.1, np.float32))

    assert len(updates) == 2
    assert [item.audio_seconds for item in updates] == [1.0, 2.0]
    assert calls == [16000, 32000]
    model.close()


def test_stream_confirms_silence_between_update_boundaries():
    model = _model((b" draft", b" final", b" next"))
    responses = iter((_result(0), _result(1), _result(2)))

    def infer(self, waveform, sample_rate, max_new_tokens, language_index):
        return next(responses)

    model._transcribe_stream_segment = MethodType(infer, model)
    stream = model.create_stream(sample_rate=16000)
    audio = np.concatenate((
        np.full(8000, 0.1, np.float32),
        np.zeros(16000, np.float32),
        np.full(8000, 0.1, np.float32),
    ))

    updates = stream.accept_audio(audio)

    assert updates[0].partial_text == "draft"
    assert updates[1].confirmed_text == "final"
    assert updates[1].partial_text == " next"
    model.close()


def test_stream_silence_returns_empty_updates_without_rpu_access():
    model = _model((b" speech",))
    model._transcribe_stream_segment = lambda *args: pytest.fail("RPU accessed")
    stream = model.create_stream(sample_rate=16000)

    updates = stream.accept_audio(np.zeros(32000, np.float32))
    final = stream.finish()

    assert len(updates) == 2
    assert all(item.confirmed_text == item.partial_text == "" for item in updates)
    assert all(item.language is None for item in updates)
    assert final.is_final and final.language is None
    model.close()


def test_stream_forced_window_keeps_two_second_overlap_without_duplicate_tokens():
    model = _model((b" alpha", b" beta"))
    calls = []

    def infer(self, waveform, sample_rate, max_new_tokens, language_index):
        calls.append((len(waveform), language_index))
        tokens = (0, 1) if language_index is not None and len(waveform) <= 3 * sample_rate else (0,)
        return WhisperTranscription("unused", "en", tokens, "eos")

    model._transcribe_stream_segment = MethodType(infer, model)
    stream = model.create_stream(sample_rate=16000)

    updates = stream.accept_audio(np.full(29 * 16000, 0.1, np.float32))
    final = stream.finish()

    assert updates[27].confirmed_text == "alpha"
    assert updates[27].partial_text == ""
    assert updates[28].confirmed_text == "alpha"
    assert updates[28].partial_text == " beta"
    assert calls[27] == (28 * 16000, None)
    assert calls[28] == (3 * 16000, 0)
    assert final.confirmed_text == "alpha beta"
    model.close()


def test_offline_forced_window_keeps_leading_silence_in_overlap():
    audio = np.full(29 * 16000, 0.1, np.float32)
    audio[26 * 16000:26 * 16000 + 12800] = 0.0

    segments = list(whisper_asr._long_audio_segments(audio))

    assert [(len(segment), overlaps) for segment, overlaps in segments] == [
        (28 * 16000, False),
        (3 * 16000, True),
    ]


def test_offline_keeps_subsecond_leading_silence_like_streaming():
    audio = np.full(31 * 16000, 0.1, np.float32)
    audio[:12800] = 0.0

    segments = list(whisper_asr._long_audio_segments(audio))

    assert len(segments[0][0]) == 28 * 16000
    assert not segments[0][1]


def test_stream_rejects_thirty_minute_overflow_before_processing():
    model = _model((b" speech",))
    model._transcribe_stream_segment = lambda *args: pytest.fail("RPU accessed")
    stream = model.create_stream(sample_rate=1)

    with pytest.raises(ValueError, match="30 minutes"):
        stream.accept_audio(np.ones(1801, np.float32))

    model.close()


def test_streaming_result_is_immutable():
    result = WhisperStreamingTranscription("done", " live", "en", 2.0, False)
    with pytest.raises(AttributeError):
        result.partial_text = "changed"


def test_long_transcribe_splits_on_silence_and_pins_language():
    model = _model((b" hello", b" world"))
    calls = []

    def run_one(self, waveform, sample_rate, max_new_tokens, language_index=None):
        calls.append((len(waveform), sample_rate, language_index))
        token = len(calls) - 1
        timing = {
            "preprocess_ms": 1.0, "encoder_ms": 2.0, "cross_ms": 3.0,
            "language_ms": 4.0, "prompt_ms": 5.0, "decode_ms": 6.0,
            "decode_step_ms": (6.0,), "total_ms": 21.0,
        }
        return _result(token), timing

    model._run_one_locked = MethodType(run_one, model)
    audio = np.concatenate((
        np.full(15 * 16000, 0.1, np.float32),
        np.zeros(16000, np.float32),
        np.full(15 * 16000, 0.1, np.float32),
    ))

    result = model.transcribe(audio)

    assert result.text == "hello world"
    assert result.token_ids == (0, 1) and result.language == "en"
    assert len(calls) == 2 and calls[0][2] is None and calls[1][2] == 0
    assert model._last_timing["encoder_ms"] == 4.0
    model.close()


def test_long_audio_validation_accepts_thirty_minutes_only():
    valid = whisper_asr._validate_audio(
        np.ones(1800, np.float32), 1, 448, max_seconds=1800
    )
    assert valid.shape == (1800,)
    with pytest.raises(ValueError, match="30 minutes"):
        whisper_asr._validate_audio(
            np.ones(1801, np.float32), 1, 448, max_seconds=1800
        )
