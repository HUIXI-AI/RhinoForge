from __future__ import annotations
import pytest
import torch
import rpu_backend
GEOM = {'qwen3-0.6b': dict(L=28, h=1024, inter=3072, nq=16, nkv=8, hd=128), 'qwen3-4b': dict(L=36, h=2560, inter=9728, nq=32, nkv=8, hd=128)}
VALID_CS_1024 = {'qwen3-0.6b': {16, 32, 64, 128, 256, 320, 448, 512}, 'qwen3-4b': {16, 32, 64, 128, 256}}
AUTO_CS_1024 = {'qwen3-0.6b': 512, 'qwen3-4b': 256}
PIN = 256
CERTIFIED_4B_MAX_KV = 1024

def _rpu(*shape):
    return torch.zeros(*shape, dtype=torch.float16, device='rpu')

def _make_handle(geom):
    (L, h, inter) = (geom['L'], geom['h'], geom['inter'])
    (nq, nkv, hd) = (geom['nq'], geom['nkv'], geom['hd'])
    q = _rpu(nq * hd, h)
    kv = _rpu(nkv * hd, h)
    o = _rpu(h, nq * hd)
    head_norm = _rpu(hd)
    hidden_norm = _rpu(h)
    gate_up = _rpu(inter, h)
    down = _rpu(h, inter)
    rope = _rpu(8192, hd // 2)
    handle = torch.ops.rpu.causal_decoder_create()
    try:
        torch.ops.rpu.causal_decoder_set_weights(handle, [q] * L, [kv] * L, [kv] * L, [o] * L, [head_norm] * L, [head_norm] * L, [hidden_norm] * L, [hidden_norm] * L, [gate_up] * L, [gate_up] * L, [down] * L, rope, rope, hidden_norm, nq, nkv, hd, h, inter, 1e-06, True)
    except Exception:
        torch.ops.rpu.causal_decoder_destroy(handle)
        raise
    return handle

@pytest.fixture
def h06():
    handle = _make_handle(GEOM['qwen3-0.6b'])
    yield handle
    torch.ops.rpu.causal_decoder_destroy(handle)

@pytest.fixture
def h4b():
    handle = _make_handle(GEOM['qwen3-4b'])
    yield handle
    torch.ops.rpu.causal_decoder_destroy(handle)

@pytest.fixture
def no_global_chunk():
    yield

def _resolve(handle, seq_len, position=0):
    return torch.ops.rpu.causal_decoder_resolve_prefill_chunk_size(handle, seq_len, position)

def test_undeclared_handle_refuses_prefill(h06):
    with pytest.raises(RuntimeError, match='no certified chunk envelope declared'):
        _resolve(h06, 1024)

def test_undeclared_handle_still_allows_decode(h06):
    assert _resolve(h06, 1) > 0

@pytest.mark.parametrize('model', ['qwen3-0.6b', 'qwen3-4b'])
def test_auto_row_reproduces_todays_green_behaviour(model, no_global_chunk):
    handle = _make_handle(GEOM[model])
    try:
        torch.ops.rpu.causal_decoder_set_chunk_envelope(handle, 1024, 0)
        assert _resolve(handle, 1024) == AUTO_CS_1024[model]
    finally:
        torch.ops.rpu.causal_decoder_destroy(handle)

def test_pinned_chunk_is_honoured_exactly(h06, no_global_chunk):
    torch.ops.rpu.causal_decoder_set_chunk_envelope(h06, 4096, PIN)
    assert _resolve(h06, 1024) == PIN
    assert PIN < AUTO_CS_1024['qwen3-0.6b'], 'the pin must actually bind — if auto already picks it, this asserts nothing'

def test_pinned_chunk_holds_across_prefill_lengths(h06, no_global_chunk):
    torch.ops.rpu.causal_decoder_set_chunk_envelope(h06, 4096, PIN)
    assert {_resolve(h06, n) for n in (512, 1024, 2048, 4096)} == {PIN}

def test_ceiling_binds_even_when_the_pin_is_not_a_legal_chunk(h06, no_global_chunk):
    torch.ops.rpu.causal_decoder_set_chunk_envelope(h06, 4096, 224)
    resolved = _resolve(h06, 1024)
    assert resolved == 128
    assert resolved <= 224

def test_explicit_override_above_envelope_is_rejected(h06, no_global_chunk):
    torch.ops.rpu.causal_decoder_set_chunk_envelope(h06, 4096, PIN)
    torch.ops.rpu.causal_decoder_set_chunk_size_override(h06, 512)
    assert 512 in VALID_CS_1024['qwen3-0.6b'], 'the override must be rejected BY THE ENVELOPE, not by SDPA validity — otherwise this test would pass with the guard removed'
    with pytest.raises(RuntimeError, match='fails validity check'):
        _resolve(h06, 1024)

def test_auto_only_row_does_not_certify_an_explicit_override(h06, no_global_chunk):
    torch.ops.rpu.causal_decoder_set_chunk_envelope(h06, 1024, 0)
    torch.ops.rpu.causal_decoder_set_chunk_size_override(h06, 128)
    with pytest.raises(RuntimeError, match='fails validity check'):
        _resolve(h06, 1024)

def test_explicit_override_within_envelope_still_works(h06, no_global_chunk):
    torch.ops.rpu.causal_decoder_set_chunk_envelope(h06, 4096, PIN)
    torch.ops.rpu.causal_decoder_set_chunk_size_override(h06, 128)
    assert _resolve(h06, 1024) == 128

def test_qwen3_4b_above_certified_prefill_is_rejected(h4b, no_global_chunk):
    torch.ops.rpu.causal_decoder_set_chunk_envelope(h4b, CERTIFIED_4B_MAX_KV, 0)
    assert _resolve(h4b, CERTIFIED_4B_MAX_KV) > 0
    with pytest.raises(RuntimeError, match="exceeds this handle's installed"):
        _resolve(h4b, 3072)

def test_position_counts_toward_the_envelope(h4b, no_global_chunk):
    torch.ops.rpu.causal_decoder_set_chunk_envelope(h4b, CERTIFIED_4B_MAX_KV, 0)
    with pytest.raises(RuntimeError, match="exceeds this handle's installed"):
        _resolve(h4b, 512, position=CERTIFIED_4B_MAX_KV)

def test_qwen3_5_promotes_the_mr54_language_envelope():
    from rpu_backend.adapters.qwen3_5 import text as q35
    assert q35.PREFILL_CHUNK_CEILING == {(24, 1024): 512, (24, 2048): 640, (32, 2560): 448, (32, 4096): 256, (64, 5120): 128}

def test_qwen3_5_language_handle_does_not_use_the_vision_envelope(monkeypatch):
    from rpu_backend.adapters.qwen3_5 import text as q35
    calls = []
    monkeypatch.setattr(torch.ops.rpu, 'qwen3_5_set_chunk_envelope', lambda handle, max_kv, chunk: calls.append((handle, max_kv, chunk)), raising=False)
    q35._declare_qwen3_5_chunk_envelope('language-handle', 32, 2560, 16384)
    assert calls == [('language-handle', 16512, 448)]

def test_chunk_caps_are_ceilings_not_exact_aligned_requests():
    causal = torch.ops.rpu.causal_decoder_create()
    q35 = torch.ops.rpu.qwen3_5_create()
    try:
        torch.ops.rpu.causal_decoder_set_chunk_size_cap(causal, 95)
        torch.ops.rpu.qwen3_5_set_chunk_size_cap(q35, 96)
    finally:
        torch.ops.rpu.causal_decoder_destroy(causal)
        torch.ops.rpu.qwen3_5_destroy(q35)

def test_a_dry_plan_does_not_consume_the_cold_window(h06, no_global_chunk):
    torch.ops.rpu.causal_decoder_set_chunk_envelope(h06, 4096, PIN)
    _resolve(h06, 1024)
    torch.ops.rpu.causal_decoder_set_chunk_envelope(h06, 8192, 512)
    assert _resolve(h06, 1024) == 512
    torch.ops.rpu.causal_decoder_set_chunk_size_cap(h06, 128)
    assert _resolve(h06, 1024) == 128

def test_envelope_rejects_the_undeclared_sentinel(h06):
    with pytest.raises(RuntimeError, match='max_kv_len must be > 0'):
        torch.ops.rpu.causal_decoder_set_chunk_envelope(h06, 0, 224)
