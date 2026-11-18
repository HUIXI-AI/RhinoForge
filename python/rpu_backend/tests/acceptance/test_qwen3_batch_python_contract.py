"""Board-free contracts for Qwen3 batched prefill/decode admission."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.api.cache import RPUCache
from rpu_backend.runtime import decoder


def _cache(batch: int, max_seq_len: int = 64) -> RPUCache:
    return RPUCache(
        num_layers=1,
        batch_size=batch,
        max_seq_len=max_seq_len,
        num_kv_heads=1,
        head_dim=16,
        device="cpu",
    )


def _capture_log(events):
    @contextmanager
    def capture(batch, seq_len, prefill_plan=None):
        events.append(("capture", batch, seq_len, prefill_plan))
        yield

    return capture


def _fake_runner(events):
    def run(model, handle, *, input_ids, past_key_values, batch_slot,
            attention_mask=None, **kwargs):
        events.append((
            "run",
            batch_slot,
            input_ids.clone(),
            None if attention_mask is None else attention_mask.clone(),
        ))
        past_key_values.update_position(input_ids.shape[1])
        return input_ids.unsqueeze(-1).to(torch.float16), past_key_values

    return run


def _planned_prefill(execution_len=32, chunk_size=32, logical_len=17):
    result = SimpleNamespace(
        graph_mode=decoder.GRAPH_RETAINED_CACHE,
        selected=SimpleNamespace(
            execution_len=execution_len, padding_rows=execution_len - logical_len,
            stage_tuple=SimpleNamespace(physical_descriptor=(1, 15), compute_chunk=chunk_size)
        ),
        graph_key_words=lambda: (7,) * 8,
    )
    return execution_len, chunk_size, result


@pytest.mark.parametrize("with_mask", [False, True])
def test_batched_prefill_accepts_equal_length_real_rows(
    monkeypatch, with_mask
):
    events = []
    monkeypatch.setattr(
        decoder, "_run_causal_decoder_forward", _fake_runner(events)
    )
    ids = torch.tensor([[1, 2, 3], [7, 8, 9]], dtype=torch.long)
    mask = torch.ones_like(ids) if with_mask else None
    cache = _cache(batch=2)

    raw, returned_cache = decoder._run_batched_prefill(
        SimpleNamespace(_rpu_batch_decode_enabled=True),
        11,
        _capture_log(events),
        past_key_values=cache,
        input_ids=ids,
        attention_mask=mask,
    )

    assert returned_cache is cache
    assert cache.position == ids.shape[1]
    assert torch.equal(raw[..., 0].to(torch.long), ids)
    runs = [event for event in events if event[0] == "run"]
    assert [event[1] for event in runs] == [0, 1]
    assert [event[2].tolist() for event in runs] == [[[1, 2, 3]], [[7, 8, 9]]]
    if with_mask:
        assert [event[3].tolist() for event in runs] == [
            [[1, 1, 1]],
            [[1, 1, 1]],
        ]
    else:
        assert all(event[3] is None for event in runs)


def test_batched_prefill_rejects_padding_before_cache_or_graph_mutation(
    monkeypatch,
):
    events = []
    monkeypatch.setattr(
        decoder, "_run_causal_decoder_forward", _fake_runner(events)
    )
    cache = _cache(batch=2)
    cache.reset_to_position(4)

    with pytest.raises(AssertionError, match="does not support padding"):
        decoder._run_batched_prefill(
            SimpleNamespace(_rpu_batch_decode_enabled=True),
            11,
            _capture_log(events),
            past_key_values=cache,
            input_ids=torch.tensor([[1, 2], [3, 4]]),
            attention_mask=torch.tensor([
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0],
            ]),
        )

    assert cache.position == 4
    assert events == []


def test_batched_prefill_rolls_back_when_capture_exit_fails(monkeypatch):
    events = []
    monkeypatch.setattr(
        decoder, "_run_causal_decoder_forward", _fake_runner(events)
    )
    cache = _cache(batch=2)
    cache.reset_to_position(4)

    @contextmanager
    def failing_capture(batch, seq_len, prefill_plan=None):
        events.append(("capture", batch, seq_len, prefill_plan))
        yield
        raise RuntimeError("synthetic capture exit failure")

    with pytest.raises(RuntimeError, match="capture exit failure"):
        decoder._run_batched_prefill(
            SimpleNamespace(_rpu_batch_decode_enabled=True),
            11,
            failing_capture,
            past_key_values=cache,
            input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        )

    assert cache.position == 4


def test_batched_prefill_rolls_back_when_output_concat_fails(monkeypatch):
    events = []
    monkeypatch.setattr(
        decoder, "_run_causal_decoder_forward", _fake_runner(events)
    )
    monkeypatch.setattr(
        decoder.torch,
        "cat",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic concat failure")
        ),
    )
    cache = _cache(batch=2)
    cache.reset_to_position(4)

    with pytest.raises(RuntimeError, match="concat failure"):
        decoder._run_batched_prefill(
            SimpleNamespace(_rpu_batch_decode_enabled=True),
            11,
            _capture_log(events),
            past_key_values=cache,
            input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        )

    assert cache.position == 4
    assert len([event for event in events if event[0] == "run"]) == 2


def test_batched_prefill_all_ones_mask_keeps_mask_and_pads_s17(monkeypatch):
    calls = []

    def native_forward(*args):
        calls.append(args)
        return args[1]

    monkeypatch.setattr(
        decoder.torch.ops.rpu,
        "causal_decoder_forward",
        native_forward,
        raising=False,
    )
    monkeypatch.setattr(
        "rpu_backend.runtime.hw_attrs.mark_first_forward_done",
        lambda model: None,
    )
    monkeypatch.setattr(
        decoder,
        "_text_prefill_execution_plan",
        lambda model, handle, cache, seq_len: _planned_prefill(),
    )
    monkeypatch.setattr(
        decoder.torch.ops.rpu,
        "causal_decoder_get_resolved_chunk_size",
        lambda handle: 32,
        raising=False,
    )
    hidden = torch.arange(2 * 17 * 16, dtype=torch.float16).reshape(2, 17, 16)
    mask = torch.ones((2, 17), dtype=torch.long)
    cache = _cache(batch=2)

    raw, returned_cache = decoder._run_batched_prefill(
        SimpleNamespace(_rpu_batch_decode_enabled=True),
        11,
        _capture_log([]),
        past_key_values=cache,
        inputs_embeds=hidden,
        attention_mask=mask,
    )

    assert returned_cache is cache
    assert cache.position == 17
    assert raw.shape == (2, 17, 16)
    assert [tuple(call[1].shape) for call in calls] == [
        (1, 32, 16),
        (1, 32, 16),
    ]
    assert [tuple(call[4].shape) for call in calls] == [(1, 17), (1, 17)]
    assert all(bool(torch.all(call[4] == 1)) for call in calls)
    assert [call[-4] for call in calls] == [0, 1]
    assert all(call[-3] is True for call in calls)
    assert all(call[-2] == 0 and call[-1] == (1, 15) for call in calls)


@pytest.mark.parametrize("position_batch", [1, 2])
def test_batched_prefill_canonicalizes_generate_positions_before_capture(
    monkeypatch, position_batch
):
    calls = []
    captures = []

    def native_forward(*args):
        calls.append(args)
        return args[1]

    monkeypatch.setattr(
        decoder.torch.ops.rpu,
        "causal_decoder_forward",
        native_forward,
        raising=False,
    )
    monkeypatch.setattr(
        "rpu_backend.runtime.hw_attrs.mark_first_forward_done",
        lambda model: None,
    )
    monkeypatch.setattr(
        decoder,
        "_text_prefill_execution_plan",
        lambda model, handle, cache, seq_len: _planned_prefill(),
    )
    monkeypatch.setattr(
        decoder.torch.ops.rpu,
        "causal_decoder_get_resolved_chunk_size",
        lambda handle: 32,
        raising=False,
    )
    hidden = torch.zeros((2, 17, 16), dtype=torch.float16)
    positions = torch.arange(17).unsqueeze(0).expand(position_batch, -1)
    cache = _cache(batch=2)

    raw, returned_cache = decoder._run_batched_prefill(
        SimpleNamespace(_rpu_batch_decode_enabled=True),
        11,
        _capture_log(captures),
        past_key_values=cache,
        inputs_embeds=hidden,
        attention_mask=torch.ones((2, 17), dtype=torch.long),
        position_ids=positions,
        cache_position=torch.arange(17),
    )

    assert returned_cache is cache
    assert cache.position == 17
    assert raw.shape == (2, 17, 16)
    assert [event[:3] for event in captures] == [
        ("capture", 1, 17), ("capture", 1, 17),
    ]
    assert all(event[3][:2] == (32, 32) for event in captures)
    assert [tuple(call[1].shape) for call in calls] == [
        (1, 32, 16),
        (1, 32, 16),
    ]
    assert all(call[7] is None for call in calls)


def test_public_boolean_cannot_forge_all_ones_padding_token(monkeypatch):
    calls = []

    def native_forward(*args):
        calls.append(args)
        return args[1]

    monkeypatch.setattr(
        decoder.torch.ops.rpu,
        "causal_decoder_forward",
        native_forward,
        raising=False,
    )
    monkeypatch.setattr(
        "rpu_backend.runtime.hw_attrs.mark_first_forward_done",
        lambda model: None,
    )
    monkeypatch.setattr(
        decoder,
        "_text_prefill_execution_plan",
        lambda model, handle, cache, seq_len: _planned_prefill(),
    )
    monkeypatch.setattr(
        decoder.torch.ops.rpu,
        "causal_decoder_get_resolved_chunk_size",
        lambda handle: 32,
        raising=False,
    )

    decoder._run_causal_decoder_forward(
        SimpleNamespace(_rpu_batch_decode_enabled=True),
        11,
        inputs_embeds=torch.zeros((1, 17, 16), dtype=torch.float16),
        attention_mask=torch.ones((1, 17), dtype=torch.long),
        past_key_values=_cache(batch=1),
        _validated_all_ones_attention_mask=True,
    )

    assert tuple(calls[0][1].shape) == (1, 17, 16)


def test_batched_prefill_rejects_non_qwen3_capability(monkeypatch):
    events = []
    monkeypatch.setattr(
        decoder, "_run_causal_decoder_forward", _fake_runner(events)
    )
    cache = _cache(batch=2)

    with pytest.raises(AssertionError, match="only.*Qwen3"):
        decoder._run_batched_prefill(
            SimpleNamespace(_rpu_batch_decode_enabled=False),
            11,
            _capture_log(events),
            past_key_values=cache,
            input_ids=torch.tensor([[1, 2], [3, 4]]),
        )

    assert cache.position == 0
    assert events == []


def test_decoder_op_receives_explicit_qwen3_batch_capability(monkeypatch):
    calls = []

    def native_forward(*args):
        calls.append(args)
        return args[1]

    monkeypatch.setattr(
        decoder.torch.ops.rpu,
        "causal_decoder_forward",
        native_forward,
        raising=False,
    )
    monkeypatch.setattr(
        "rpu_backend.runtime.hw_attrs.mark_first_forward_done",
        lambda model: None,
    )
    monkeypatch.setattr(
        decoder.torch.ops.rpu,
        "causal_decoder_get_resolved_chunk_size",
        lambda handle: 1,
        raising=False,
    )
    model = SimpleNamespace(
        _rpu_batch_decode_enabled=True,
        _rpu_decode_stage_descriptor=(1, 15),
    )
    hidden = torch.zeros((2, 1, 16), dtype=torch.float16)
    cache = _cache(batch=2)

    raw, _ = decoder._run_causal_decoder_forward(
        model,
        11,
        inputs_embeds=hidden,
        past_key_values=cache,
    )

    assert raw is hidden
    assert calls[0][-3] is True
    assert calls[0][-2:] == (0, (1, 15))
    assert cache.position == 1


def test_batched_decode_canonicalizes_standard_hf_controls():
    cache = _cache(batch=2)
    cache.reset_to_position(16)

    assert decoder._canonicalize_plain_text_controls(
        1,
        cache,
        torch.ones((2, 17), dtype=torch.long),
        torch.tensor([[16], [16]], dtype=torch.long),
        torch.tensor([16], dtype=torch.long),
        batch_size=2,
    ) == (None, None, None)

    padded = torch.ones((2, 17), dtype=torch.long)
    padded[1, -1] = 0
    with pytest.raises(NotImplementedError, match="all-ones"):
        decoder._canonicalize_plain_text_controls(
            1, cache, padded, None, None, batch_size=2
        )


def test_decoder_rejects_non_qwen3_batch_before_native_call(monkeypatch):
    called = False

    def native_forward(*args):
        nonlocal called
        called = True
        return args[1]

    monkeypatch.setattr(
        decoder.torch.ops.rpu,
        "causal_decoder_forward",
        native_forward,
        raising=False,
    )
    monkeypatch.setattr(
        "rpu_backend.runtime.hw_attrs.mark_first_forward_done",
        lambda model: None,
    )

    with pytest.raises(AssertionError, match="only.*Qwen3"):
        decoder._run_causal_decoder_forward(
            SimpleNamespace(_rpu_batch_decode_enabled=False),
            11,
            inputs_embeds=torch.zeros((2, 1, 16), dtype=torch.float16),
            past_key_values=_cache(batch=2),
        )

    assert called is False
