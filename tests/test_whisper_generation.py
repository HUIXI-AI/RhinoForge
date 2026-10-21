"""Host generation, token decoding and feature preparation for Whisper base."""

import base64

import numpy as np
import pytest
import torch

from rpu_backend.api import whisper_asr


def test_token_table_decodes_base64_bytes_and_rejects_gaps(tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_text(
        f"{base64.b64encode(b' hello').decode()} 0\n"
        f"{base64.b64encode(' 世界'.encode()).decode()} 1\n"
    )

    table = whisper_asr._TokenTable.load(path, expected_size=2)

    assert table.decode((0, 1)) == " hello 世界"
    path.write_text("IQ== 1\n")
    with pytest.raises(ValueError, match="token table"):
        whisper_asr._TokenTable.load(path, expected_size=2)


def test_log_mel_is_fixed_finite_and_resamples():
    wave = np.sin(np.linspace(0, 40, 8000, dtype=np.float32))

    mel = whisper_asr._log_mel_spectrogram(wave, 8000)

    assert mel.shape == (1, 80, 3000)
    assert mel.dtype is torch.float16
    assert torch.isfinite(mel).all()
    assert torch.count_nonzero(mel[:, :, 1600:]) == 0


class _Encoder:
    def __init__(self):
        self.closed = False

    def encode(self, mel):
        assert mel.shape == (1, 80, 3000)
        return torch.ones((1, 1500, 512), dtype=torch.float16)

    def close(self):
        self.closed = True

class _Decoder:
    def __init__(self, *, eos=True):
        self.calls = []
        self.eos = eos
        self.closed = False

    def prepare_audio(self, hidden):
        assert hidden.shape == (1, 1500, 512)

    def _logits(self, token):
        logits = torch.zeros((1, 51865), dtype=torch.float16)
        logits[0, token] = 10
        return logits

    def prefill(self, token):
        self.calls.append(token)
        logits = self._logits(5)
        logits[0, 50259] = 9
        return logits

    def decode(self, token):
        self.calls.append(token)
        if token == 50363 or token == 10:
            return self._logits(50257 if token == 10 and self.eos else 10)
        return self._logits(4)

    def close(self):
        self.closed = True


def _model(decoder):
    table = whisper_asr._TokenTable((b"",) * 10 + (b" hello",))
    return whisper_asr.WhisperASRModel(_Encoder(), decoder, table)


def test_generation_restricts_language_and_stops_on_eos(monkeypatch):
    monkeypatch.setattr(
        whisper_asr, "_log_mel_spectrogram",
        lambda wave, rate: torch.zeros((1, 80, 3000), dtype=torch.float16),
    )
    decoder = _Decoder()
    model = _model(decoder)

    result = model.transcribe(np.ones(16000, dtype=np.float32), max_new_tokens=8)

    assert result.language == "en"
    assert result.token_ids == (10,)
    assert result.text == "hello"
    assert result.finish_reason == "eos"
    assert decoder.calls == [50258, 50259, 50359, 50363, 10]


def test_generation_reports_length_and_close_owns_both_runtimes(monkeypatch):
    monkeypatch.setattr(
        whisper_asr, "_log_mel_spectrogram",
        lambda wave, rate: torch.zeros((1, 80, 3000), dtype=torch.float16),
    )
    decoder = _Decoder(eos=False)
    model = _model(decoder)

    result = model.transcribe(np.ones(16000, dtype=np.float32), max_new_tokens=2)
    model.close()

    assert result.token_ids == (10, 10)
    assert result.finish_reason == "length"
    assert model._encoder.closed and decoder.closed


def test_close_attempts_encoder_cleanup_when_decoder_cleanup_fails():
    decoder = _Decoder()
    decoder.close = lambda: (_ for _ in ()).throw(RuntimeError("decoder close"))
    model = _model(decoder)

    with pytest.raises(RuntimeError, match="decoder close"):
        model.close()

    assert model._encoder.closed
