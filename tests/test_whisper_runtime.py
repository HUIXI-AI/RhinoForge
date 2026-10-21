"""Host tests for Whisper encoder loading and lifecycle."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _varint(value):
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _field(number, wire, payload):
    key = _varint((number << 3) | wire)
    if wire == 2:
        return key + _varint(len(payload)) + payload
    return key + payload


def test_minimal_onnx_weight_reader_validates_shape(tmp_path):
    from rpu_backend.adapters.whisper.weights import load_float_initializers
    from rpu_backend.api.errors import UnsupportedModelError

    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    dimensions = b"".join(_varint(item) for item in values.shape)
    tensor = b"".join((
        _field(1, 2, dimensions),
        _field(2, 0, _varint(1)),
        _field(8, 2, b"weight"),
        _field(9, 2, values.tobytes()),
    ))
    graph = _field(5, 2, tensor)
    model = _field(7, 2, graph)
    path = tmp_path / "encoder.onnx"
    path.write_bytes(model)

    loaded = load_float_initializers(path, {"weight": (2, 3)})

    np.testing.assert_array_equal(loaded["weight"], values)
    with pytest.raises(UnsupportedModelError, match="initializer mismatch"):
        load_float_initializers(path, {"weight": (3, 2)})


def test_decoder_weight_map_matches_frozen_layer_names(monkeypatch):
    from rpu_backend.adapters.whisper import weights

    requested = {}

    def load(_path, expected):
        requested.update(expected)
        return {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in expected.items()
        }

    monkeypatch.setattr(weights, "load_float_initializers", load)
    loaded = weights.load_decoder_weights("decoder.onnx", layer_count=1)

    assert "onnx::MatMul_4196" in requested
    assert "onnx::MatMul_4240" in requested
    assert loaded["layers"][0]["ff2_w"].shape == (512, 2048)


def test_full_decoder_weight_map_includes_tied_io_and_encoder_cross_kv(monkeypatch):
    from rpu_backend.adapters.whisper import weights

    requests = []

    def load(_path, expected):
        requests.append(expected)
        return {
            name: np.broadcast_to(np.float32(0), shape)
            for name, shape in expected.items()
        }

    monkeypatch.setattr(weights, "load_float_initializers", load)
    encoder = weights.load_encoder_weights("encoder.onnx")
    decoder = weights.load_decoder_weights("decoder.onnx")

    assert requests[0]["onnx::MatMul_1363"] == (512, 512)
    assert requests[0]["onnx::Add_1390"] == (512,)
    assert encoder["cross_layers"][5]["v_w"].shape == (512, 512)
    assert requests[1]["textDecoder.token_embedding.weight"] == (51865, 512)
    assert requests[1]["textDecoder.positional_embedding"] == (448, 512)
    assert decoder["token_embedding"].shape == (51865, 512)
    assert decoder["final_w"].shape == (512,)


class _Graph:
    def __init__(self, calls):
        self.calls = calls
        self.frozen = False

    def capture(self, signature):
        self.calls.append(("capture", signature))
        return nullcontext()

    def freeze(self):
        self.calls.append(("freeze",))
        self.frozen = True

    def is_frozen(self):
        return self.frozen

    def clear(self):
        self.calls.append(("clear",))

    def cache_invariant_ok(self):
        return True


class _Cache:
    max_seq_len = 1536

    def __init__(self):
        self.k_caches = [torch.zeros(1)]
        self.v_caches = [torch.zeros(1)]

    def reset_to_position(self, position):
        assert position == 0


def _install_mocks(monkeypatch, *, fail_frontend=False):
    from rpu_backend.adapters.whisper import runtime

    calls = []
    prepared = SimpleNamespace(
        weight_args=("weights",),
        final_args=(torch.ones(1), torch.zeros(1)),
        frontend_args=("frontend",),
        keepalive=("keepalive",),
    )
    monkeypatch.setattr(runtime, "load_encoder_weights", lambda path: {})
    monkeypatch.setattr(runtime, "_prepare_encoder_tensors", lambda weights: prepared)
    monkeypatch.setattr(runtime, "_make_encoder_cache", lambda: _Cache())
    monkeypatch.setattr(runtime, "_make_mel_staging", lambda: torch.empty(1))
    monkeypatch.setattr(runtime, "_copy_mel_to_staging", lambda staging, mel: None)
    monkeypatch.setattr(
        runtime,
        "_plan_whisper_encoder",
        lambda handle, graph: SimpleNamespace(
            descriptor=(1, 2), signature="signature"),
    )
    monkeypatch.setattr(
        runtime.rpu_backend.graph,
        "GraphCache",
        lambda max_entries=1: _Graph(calls),
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "whisper_encoder_create",
        lambda: calls.append(("create",)) or 17,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "whisper_encoder_destroy",
        lambda handle: calls.append(("destroy", handle)),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "whisper_encoder_set_weights",
        lambda handle, *args: calls.append(("weights", handle, args)),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "whisper_encoder_set_final_norm",
        lambda handle, *args: calls.append(("final", handle, args)),
        raising=False,
    )

    def set_frontend(handle, *args):
        calls.append(("frontend", handle, args))
        if fail_frontend:
            raise RuntimeError("frontend failed")

    monkeypatch.setattr(
        torch.ops.rpu,
        "whisper_encoder_set_frontend",
        set_frontend,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "whisper_encoder_set_chunk_envelope",
        lambda handle, *args: calls.append(("envelope", handle, args)),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "whisper_encoder_forward",
        lambda *args: torch.arange(6, dtype=torch.float16).reshape(1, 2, 3),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "spm_alloc_reset_temporary",
        lambda: calls.append(("spm_reset",)),
        raising=False,
    )
    return runtime, calls


def test_encoder_build_replay_output_ownership_and_idempotent_close(monkeypatch):
    runtime, calls = _install_mocks(monkeypatch)
    encoder = runtime.WhisperEncoderRuntime("encoder.onnx")
    mel = torch.zeros(1, 80, 3000, dtype=torch.float16)

    first = encoder.encode(mel)
    second = encoder.encode(mel)

    assert first.data_ptr() != second.data_ptr()
    assert sum(call[0] == "create" for call in calls) == 1
    assert sum(call[0] == "freeze" for call in calls) == 1
    assert sum(call[0] == "capture" for call in calls) == 2
    encoder.close()
    encoder.close()
    assert sum(call[0] == "destroy" for call in calls) == 1
    assert sum(call[0] == "clear" for call in calls) == 1


def test_encoder_partial_install_failure_cleans_graph_and_handle(monkeypatch):
    runtime, calls = _install_mocks(monkeypatch, fail_frontend=True)

    with pytest.raises(RuntimeError, match="frontend failed"):
        runtime.WhisperEncoderRuntime("encoder.onnx")

    assert sum(call[0] == "destroy" for call in calls) == 1
    assert sum(call[0] == "clear" for call in calls) == 1


def _install_decoder_mocks(monkeypatch, *, fail_cross=False):
    from rpu_backend.adapters.whisper import runtime

    calls = []
    prepared = SimpleNamespace(
        weight_args=("layers",), cross_args=("cross",), io_args=("io",),
        token_embedding=torch.ones(1), keepalive=("keepalive",),
    )
    monkeypatch.setattr(runtime, "load_encoder_weights", lambda path: {})
    monkeypatch.setattr(runtime, "load_decoder_weights", lambda path: {})
    monkeypatch.setattr(runtime, "_prepare_decoder_tensors", lambda *args: prepared)
    monkeypatch.setattr(runtime, "_make_decoder_cache", lambda length: _Cache())
    monkeypatch.setattr(
        runtime, "_make_decoder_staging",
        lambda: (torch.empty(1), [torch.empty(1), torch.empty(1)],
                 [torch.empty(1), torch.empty(1)],
                 [torch.empty(1), torch.empty(1)]),
    )
    monkeypatch.setattr(
        runtime.rpu_backend.graph, "GraphCache",
        lambda max_entries=1: _Graph(calls),
    )
    monkeypatch.setattr(
        torch.ops.rpu, "whisper_decoder_create",
        lambda: calls.append(("create",)) or 23, raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "whisper_decoder_destroy",
        lambda handle: calls.append(("destroy", handle)), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "whisper_decoder_set_weights",
        lambda handle, *args: calls.append(("weights", handle, args)), raising=False,
    )

    def set_cross(handle, *args):
        calls.append(("cross", handle, args))
        if fail_cross:
            raise RuntimeError("cross failed")

    monkeypatch.setattr(
        torch.ops.rpu, "whisper_decoder_set_cross_weights",
        set_cross, raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "whisper_decoder_set_io",
        lambda handle, *args: calls.append(("io", handle, args)), raising=False,
    )
    return runtime, calls


def test_decoder_install_and_idempotent_close(monkeypatch):
    runtime, calls = _install_decoder_mocks(monkeypatch)

    decoder = runtime.WhisperDecoderRuntime("encoder.onnx", "decoder.onnx")
    decoder.close()
    decoder.close()

    assert sum(call[0] == "create" for call in calls) == 1
    assert sum(call[0] == "weights" for call in calls) == 1
    assert sum(call[0] == "cross" for call in calls) == 1
    assert sum(call[0] == "io" for call in calls) == 1
    assert sum(call[0] == "destroy" for call in calls) == 1
    assert sum(call[0] == "clear" for call in calls) == 3


def test_decoder_partial_install_failure_cleans_graphs_and_handle(monkeypatch):
    runtime, calls = _install_decoder_mocks(monkeypatch, fail_cross=True)

    with pytest.raises(RuntimeError, match="cross failed"):
        runtime.WhisperDecoderRuntime("encoder.onnx", "decoder.onnx")

    assert sum(call[0] == "destroy" for call in calls) == 1
    assert sum(call[0] == "clear" for call in calls) == 3
