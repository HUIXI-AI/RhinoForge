"""Board-free lifecycle checks for the Qwen3.5 text installation."""
from __future__ import annotations

import types
import ast
from pathlib import Path

import pytest
import torch

import rpu_backend
import rpu_backend.adapters.qwen3_5 as adapter_module
import rpu_backend.adapters.qwen3_5.text as text_module
import rpu_backend.adapters.qwen3_5.vision as vision_module
from rpu_backend.api._execution import (
    normalize_rpu_execution,
    reconfigure_rpu_execution,
)
from rpu_backend.api.errors import PlannerRejectError
from rpu_backend.runtime.execution_planner import _physical_manifest_fingerprint


@pytest.fixture(autouse=True)
def _fake_planner_native_identity(monkeypatch):
    # These host fixtures already replace native planning; keep its lifetime
    # query in the same fake native boundary. No installed model is consulted.
    for prefix in ("qwen3_5", "qwen3_5_vision"):
        monkeypatch.setattr(torch.ops.rpu, prefix + "_planner_cache_identity",
                            lambda handle: (int(handle), 1), raising=False)
    # Public installation checks this cold ABI before entering the injected
    # lifecycle failures. The host fixture supplies the complete native seam.
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_linear_acc32",
                        lambda _handle, _enabled: None, raising=False)


def _native_fixture_state(request, **fields):
    from rpu_backend.runtime._native_retirement import _InstalledNativeResource

    state = text_module._Qwen35NativeState(**fields)
    # Native/graph lifetime is exercised by test_q35_g05_strict_retirement;
    # these admission fixtures contain no device tensors or recorded graphs.
    resource = _InstalledNativeResource(state, state.handle, lambda _: None,
        graphs=(), keepalive=(), label="Q35 lifecycle fixture", handle_name="handle")
    state.retirement, state.handle_finalizer = resource, resource.finalizer
    def finish_fixture():
        resource.finalizer.detach()
        vars(state).pop("_retirement_close", None)
    request.addfinalizer(finish_fixture)
    return state


def _vision_descriptor(
    execution_len,
    input_chunk,
    compute_chunk,
    *,
    qkv_chunk=None,
    span_count=1,
    temporal_num_frames=1,
    camera_batch_count=1,
    compact_input=False,
    stage_fingerprint=1,
    attention_selector=1,
):
    qkv_chunk = compute_chunk if qkv_chunk is None else qkv_chunk

    def stage(capacity):
        chunks = []
        offset = 0
        while offset < execution_len:
            length = min(capacity, execution_len - offset)
            chunks.extend((len(chunks) // 4, offset, length, offset + length))
            offset += length
        return chunks

    stages = [stage(cap) for cap in (input_chunk, qkv_chunk, compute_chunk)]
    span_len = execution_len // span_count
    spans = [
        word
        for ordinal in range(span_count)
        for word in (ordinal * span_len, span_len, ordinal)
    ]
    manifest_values = [
        1, execution_len, execution_len, 0, execution_len, execution_len,
        1, 1, 1, 1, 101, attention_selector, 0, 0,
    ]
    manifest_fingerprint = _physical_manifest_fingerprint(manifest_values)
    inner = [
        3, 0, input_chunk, qkv_chunk, compute_chunk,
        stage_fingerprint >> 32, stage_fingerprint & 0xFFFFFFFF,
        1, 0, 0, 0,
        *(len(stage_chunks) // 4 for stage_chunks in stages),
        span_count,
        *(word for stage_chunks in stages for word in stage_chunks),
        *spans,
        1, execution_len, execution_len, 0, execution_len, execution_len,
        1, 1, 1,
        manifest_fingerprint >> 32,
        manifest_fingerprint & 0xFFFFFFFF,
        1, 101, attention_selector, 0, 0, 0,
    ]
    inner[1] = len(inner)
    outer_spans = [
        word
        for ordinal in range(span_count)
        for word in (ordinal * span_len, span_len)
    ]
    return [
        3, execution_len, input_chunk, qkv_chunk, compute_chunk,
        512, 1, stage_fingerprint >> 32,
        stage_fingerprint & 0xFFFFFFFF, 0, 9, span_count, 1, 1,
        temporal_num_frames, camera_batch_count, int(compact_input),
        *outer_spans,
        *inner,
    ]


def test_quant_policy_requires_exact_opt_in_before_model_access(monkeypatch):
    config = types.SimpleNamespace(quant_config={"method": "w8a16"})
    model = types.SimpleNamespace(config=config)

    for value in (None, "true"):
        if value is None:
            monkeypatch.delenv("QWEN3_5_QUANT_ALLOW_UNCERTIFIED", raising=False)
        else:
            monkeypatch.setenv("QWEN3_5_QUANT_ALLOW_UNCERTIFIED", value)
        with pytest.raises(adapter_module.UnsupportedModelError, match="uncertified"):
            adapter_module.Qwen3_5Adapter.preflight(config)
        with pytest.raises(adapter_module.UnsupportedModelError, match="uncertified"):
            adapter_module.Qwen3_5Adapter(model)

    monkeypatch.setenv("QWEN3_5_QUANT_ALLOW_UNCERTIFIED", "1")
    assert adapter_module._check_quant_policy(config) is None
    assert adapter_module._check_quant_policy(
        types.SimpleNamespace(quant_config={})
    ) is None


def test_qwen3_5_language_envelope_contains_the_mr54_rows():
    expected = {
        (24, 1024): 512,
        (24, 2048): 640,
        (32, 2560): 448,
        (32, 4096): 256,
    }
    assert {key: text_module.PREFILL_CHUNK_CEILING[key] for key in expected} == expected


def test_vision_config_does_not_change_language_chunk_envelope(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_set_chunk_envelope",
        lambda handle, max_kv_len, chunk: calls.append(
            (handle, max_kv_len, chunk)
        ),
        raising=False,
    )

    profiles = ((24, 2048), (32, 2560), (24, 1024), (32, 4096))
    for handle, key in enumerate(profiles):
        text_module._declare_qwen3_5_chunk_envelope(handle, *key, 16384)
    assert calls == [
        (0, 16512, 640),
        (1, 16512, 448),
        (2, 16512, 512),
        (3, 16512, 256),
    ]

    calls.clear()
    text_module._declare_qwen3_5_chunk_envelope(4, 24, 2048, 32768)
    assert calls == [(4, 32896, 640)]


def test_multimodal_request_uses_the_shared_stage_ceiling():
    result = []
    assert text_module._plan_prefill_execution(
        384,
        384,
        0,
        max_stage_chunk_size=128,
        resolve_stage_domain=lambda _execution_len: (
            (128, 128, 128), (192, 192, 192), (640, 640, 640)
        ),
        plan_result_sink=result.append,
    ) == (384, 128)
    assert [
        candidate.reject_reason for candidate in result[0].candidates
        if not candidate.feasible
    ] == ["stage_chunk_limit", "stage_chunk_limit"]


def test_text_exact_long_chunk_is_admitted_by_the_shared_planner():
    assert text_module._plan_prefill_execution(
        4096,
        4096,
        0,
        exact_chunk_size=640,
        resolve_stage_domain=lambda _execution_len: ((640, 640, 640),),
    ) == (4096, 640)


def test_multimodal_exact_long_chunk_fails_instead_of_clamping():
    with pytest.raises(RuntimeError, match="EXACT_MISMATCH"):
        text_module._plan_prefill_execution(
            384,
            384,
            0,
            exact_chunk_size=640,
            max_stage_chunk_size=128,
            resolve_stage_domain=lambda _execution_len: (
                (128, 128, 128), (640, 640, 640)
            ),
        )


def _text_preflight_owner(*, exact_chunk_size=None, retained=False):
    state = types.SimpleNamespace(
        handle=7,
        num_layers=24,
        hidden_size=2048,
        max_seq_len=8192,
        chunk_envelope=(8320, 640),
        padding_budget=0,
        padding_rows="auto",
        exact_chunk_size=exact_chunk_size,
        prefill_plan_key=None,
        prefill_plan=None,
        last_a6_plan=None,
        graph_cache=rpu_backend.graph.GraphCache(),
        prefill_graph_cache=rpu_backend.graph.GraphCache() if retained else None,
    )
    return types.SimpleNamespace(_rpu_qwen3_5=state), types.SimpleNamespace(
        position=0,
        max_seq_len=8192,
        allocated_max_seq_len=8255,
    )


@pytest.mark.parametrize("profile", [(24, 1024), (24, 2048), (32, 2560), (32, 4096)])
@pytest.mark.parametrize("exact", [False, True])
def test_image_and_text_requests_share_the_actual_language_domain(monkeypatch, profile, exact):
    from test_execution_planner import _native_candidate_v2

    ceiling = text_module.PREFILL_CHUNK_CEILING[profile]
    envelope = (8320, ceiling)
    text, cache = _text_preflight_owner(exact_chunk_size=ceiling if exact else None)
    state = text._rpu_qwen3_5
    state.num_layers, state.hidden_size = profile
    state.chunk_envelope = envelope
    calls = []
    def domain(handle, execution, logical):
        calls.append((handle, execution, logical))
        descriptors = [_native_candidate_v2(execution, (chunk,) * 3,
            descriptor_version=3, graph_lifecycle=2) for chunk in (128, ceiling)]
        return (1, len(descriptors), *(word for descriptor in descriptors
                    for word in (len(descriptor), *descriptor)))
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_resolve_prefill_stage_domain", domain, raising=False)

    results = [text_module._preflight_qwen3_5_text(
        text, cache, 4096, multimodal_prefill=image)[2] for image in (False, True)]
    assert calls == [(7, 4096, 4096)] * 2
    assert results[0].as_dict() == results[1].as_dict()
    assert results[1].selected.stage_tuple.compute_chunk == ceiling
    assert results[1].graph_mode == "BOUNDED_ONESHOT"


@pytest.mark.parametrize("image", [False, True])
def test_language_preflight_rejects_beyond_installed_envelope_without_native_probe(monkeypatch, image):
    text, cache = _text_preflight_owner()
    probe = []
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_resolve_prefill_stage_domain",
        lambda *_args: probe.append(True),
        raising=False,
    )

    with pytest.raises(ValueError, match="limit=8192"):
        text_module._preflight_qwen3_5_text(
            text, cache, 8193, multimodal_prefill=image
        )
    assert probe == []


@pytest.mark.parametrize("image", [False, True])
def test_language_preflight_keeps_specialized_native_envelope_and_cold_cap(monkeypatch, image):
    text, cache = _text_preflight_owner()
    text._rpu_qwen3_5.prefill_capability_envelope = (256, 256)
    calls = []
    def domain(*args):
        calls.append(args)
        # The actual native domain still owns a legacy cold cap of 128.
        return ((128, 128, 128),)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_resolve_prefill_stage_domain", domain, raising=False)
    result = text_module._preflight_qwen3_5_text(text, cache, 256, multimodal_prefill=image)[2]
    assert result.selected.stage_tuple.compute_chunk == 128
    assert calls == [(7, 256, 256)]
    with pytest.raises(adapter_module.UnsupportedModelError, match="limit 256"):
        text_module._preflight_qwen3_5_text(text, cache, 257, multimodal_prefill=image)
    assert len(calls) == 1


@pytest.mark.parametrize("fault", ["stage_ceiling", "native_spm"])
def test_multimodal_language_does_not_bypass_shared_or_native_rejections(monkeypatch, fault):
    text, cache = _text_preflight_owner(exact_chunk_size=656)
    def domain(*_args):
        if fault == "native_spm":
            raise RuntimeError("native SPM capacity rejected")
        return ((656, 656, 656),)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_resolve_prefill_stage_domain", domain, raising=False)
    with pytest.raises(RuntimeError, match="EXACT_MISMATCH|native SPM capacity rejected"):
        text_module._preflight_qwen3_5_text(text, cache, 4096, multimodal_prefill=True)


def test_text_preflight_passes_logical_length_and_graph_lifecycle(monkeypatch):
    text, cache = _text_preflight_owner(retained=True)
    calls = []
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_resolve_prefill_stage_domain",
        lambda handle, execution_len, logical_len: (
            calls.append((handle, execution_len, logical_len))
            or ((128, 128, 128),)
        ),
        raising=False,
    )

    _, _, result = text_module._preflight_qwen3_5_text(text, cache, 128)
    assert calls == [(7, 128, 128)]
    assert result.graph_mode == "RETAINED_CACHE"


def test_multimodal_text_horizon_is_selected_from_the_forward_request():
    source = (Path(adapter_module.__file__)).read_text(encoding="utf-8")
    assert "multimodal_prefill=pixel_values is not None" in source


def test_text_install_options_are_validated_before_mutation(monkeypatch):
    monkeypatch.setenv("QWEN3_5_TEXT_CHUNK", "128")
    monkeypatch.setenv("QWEN3_5_TEXT_PADDING_BUDGET", "32")
    assert text_module._validate_text_install_options(4096) == (4096, 128, 32)

    for invalid in (True, 0, -1, 1.5, "4096"):
        with pytest.raises(ValueError, match="positive integer"):
            text_module._validate_text_install_options(invalid)

    monkeypatch.setenv("QWEN3_5_TEXT_CHUNK", "63")
    with pytest.raises(ValueError, match="at least 64"):
        text_module._validate_text_install_options(4096)
    monkeypatch.setenv("QWEN3_5_TEXT_CHUNK", "0")
    monkeypatch.setenv("QWEN3_5_TEXT_PADDING_BUDGET", "-1")
    with pytest.raises(ValueError, match="non-negative"):
        text_module._validate_text_install_options(4096)
    monkeypatch.setenv("QWEN3_5_TEXT_PADDING_BUDGET", "65")
    with pytest.raises(ValueError, match="cache-backed maximum 64"):
        text_module._validate_text_install_options(4096)


def test_canonical_auto_ignores_legacy_chunk_cap(monkeypatch):
    monkeypatch.setenv("QWEN3_5_TEXT_CHUNK", "128")
    monkeypatch.setenv("QWEN3_5_TEXT_PADDING_BUDGET", "32")
    assert text_module._validate_text_install_options(
        4096,
        execution_config={"prefill": {"chunk_size": "auto"}},
    ) == (4096, 0, 32)


def test_canonical_chunk_conflict_is_rejected_before_install(monkeypatch):
    monkeypatch.setenv("QWEN3_5_TEXT_CHUNK", "128")
    with pytest.raises(ValueError, match="conflicts with canonical"):
        text_module._validate_text_install_options(
            4096,
            execution_config={"prefill": {"chunk_size": 256}},
        )


def test_text_control_snapshot_is_immutable_and_records_source(monkeypatch):
    monkeypatch.delenv("QWEN3_5_TEXT_CHUNK", raising=False)
    result = text_module._resolve_text_install_options(
        4096,
        execution_config={"prefill": {"chunk_size": "auto", "padding_budget": 16}},
    )
    snapshot = result[3]
    assert snapshot["chunk_source"] == "canonical"
    assert snapshot["padding_source"] == "canonical"
    with pytest.raises(TypeError):
        snapshot["chunk_source"] = "legacy_env"


def test_vision_group_uses_canonical_a6_and_exact_constraint(monkeypatch):
    owner = types.SimpleNamespace(
        _rpu_vision_exact_chunk_size=128,
        _rpu_vision_padding_rows="auto",
        _rpu_vision_padding_budget=0,
        _rpu_vision_kv_cache=types.SimpleNamespace(max_seq_len=512),
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_vision_resolve_prefill_domain",
        lambda _handle, execution_len, *_geometry: [_vision_descriptor(
            execution_len, 128, 128
        )],
        raising=False,
    )
    result = vision_module._plan_qwen3_5_vision_group(
        owner, 7, 256, group_id=0
    )
    assert result.selected is not None
    assert result.selected.stage_tuple.input_chunk == 128

    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_vision_resolve_prefill_domain",
        lambda _handle, execution_len, *_geometry: [_vision_descriptor(
            execution_len, 128, 64
        )],
        raising=False,
    )
    # Changing the fake native capability is a new native lifetime.
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_planner_cache_identity",
                        lambda handle: (handle, 2))
    with pytest.raises(PlannerRejectError, match="EXACT_MISMATCH"):
        vision_module._plan_qwen3_5_vision_group(owner, 7, 256, group_id=0)


def test_vision_group_default_auto_uses_zero_padding(monkeypatch):
    owner = types.SimpleNamespace(
        _rpu_vision_exact_chunk_size=None,
        _rpu_vision_padding_rows="auto",
        _rpu_vision_padding_budget=0,
        _rpu_vision_kv_cache=types.SimpleNamespace(max_seq_len=512),
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_vision_resolve_prefill_domain",
        lambda _handle, execution_len, *_geometry: [_vision_descriptor(
            execution_len, 896, 128
        )],
        raising=False,
    )
    result = vision_module._plan_qwen3_5_vision_group(
        owner, 7, 256, group_id=0
    )
    assert result.selected is not None
    assert result.selected.execution_len == 256
    assert result.selected.stage_tuple.input_chunk == 896
    assert result.selected.stage_tuple.compute_chunk == 128
    assert result.graph_mode == "RETAINED_CACHE"
    assert result.optimality == "CAPABILITY_SELECTED"


def test_vision_full_domain_keeps_chunk_and_same_chunk_route_alternatives(monkeypatch):
    from rpu_backend.api._execution import bind_execution_session

    owner = types.SimpleNamespace(
        _rpu_vision_exact_chunk_size=None, _rpu_vision_padding_rows="auto",
        _rpu_vision_padding_budget=0,
        _rpu_vision_kv_cache=types.SimpleNamespace(max_seq_len=512),
    )
    session = bind_execution_session(
        owner, {}, entry_point="Q35VisionHost", supported={"vision": ("chunk_size",)},
        graph_mode="RETAINED_CACHE")
    session._bind_planner_owner(owner)
    domain = [
        _vision_descriptor(256, 128, 64),
        _vision_descriptor(256, 128, 256, attention_selector=1),
        _vision_descriptor(256, 128, 256, attention_selector=2),
    ]
    calls = []
    def resolve(*args):
        calls.append(args)
        return domain
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_resolve_prefill_domain",
                        resolve, raising=False)
    observations = []
    with session._collect_planner_costs(lambda *args: observations.append(args)):
        result = vision_module._plan_qwen3_5_vision_group(owner, 17, 256, group_id=0)
    assert calls == [(17, 256, 1, 1, False)]
    assert session._planner_native[("", "vision", "qwen3_5_vision")] == (owner, 17)
    assert len(result.candidates) == 3
    assert result.selected.stage_tuple.compute_chunk == 256
    assert dict(result.selected.stage_tuple.physical_metadata)["raw_attention_site_count"] == 1
    assert [item[4] for item in observations] == ["domain", "result"]
    assert all(item[:4] == (owner, "", "vision", ("qwen3_5_vision", 17))
               for item in observations)
    assert len(observations[0][5][1]) == 3
    assert observations[1][5] == (256, 0, result)
    assert {row.physical_descriptor for row in observations[0][5][1]} == {
        tuple(raw[19:]) for raw in domain}


def test_text_preparation_reuses_shapes_but_observes_real_domains_and_native_lifetime(monkeypatch):
    from rpu_backend.api import _execution

    text, cache = _text_preflight_owner(retained=True)
    calls = []
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_resolve_prefill_stage_domain",
                        lambda *args: calls.append(args) or ((128, 128, 128),),
                        raising=False)
    session = _execution.bind_execution_session(
        text, {}, entry_point="Q35TextHost", supported={"prefill": ("chunk_size",)},
        graph_mode="RETAINED_CACHE")
    session._bind_planner_owner(text)
    first = text_module._preflight_qwen3_5_text(text, cache, 128)[2]
    text_module._preflight_qwen3_5_text(text, cache, 256)
    assert text_module._preflight_qwen3_5_text(text, cache, 128)[2] is first
    assert len(calls) == 2
    observations = []
    with session._collect_planner_costs(lambda *args: observations.append(args)):
        text_module._preflight_qwen3_5_text(text, cache, 128)
        text_module._preflight_qwen3_5_text(text, cache, 128)
    assert len(calls) == 4
    assert [item[4] for item in observations] == ["domain", "result"] * 2
    # A same-handle native replacement must not reuse the earlier search.
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_planner_cache_identity", lambda handle: (handle, 2))
    newer = text_module._preflight_qwen3_5_text(text, cache, 128)[2]
    assert newer is not first and len(calls) == 5
    assert text_module._preflight_qwen3_5_text(text, cache, 128)[2] is newer
    assert len(calls) == 5
    assert text._rpu_qwen3_5.prefill_plan[2] is newer


@pytest.mark.parametrize("cores,vision_bytes", [(4, 399360), (8, 373760)])
@pytest.mark.parametrize("text_started", [False, True])
@pytest.mark.parametrize("exact", [False, True])
def test_multimodal_persistent_barrier_replans_against_peer_floor(
        monkeypatch, cores, vision_bytes, text_started, exact):
    """Real Python planner/prepare flow, native allocation and domain as doubles.

    The byte sizes reproduce the observed 2B peer-floor boundary; this is not
    native layout or hardware credit. An accepted cold memo must not escape
    the first-image barrier, and EXACT must still reject instead of shrinking.
    """
    from test_execution_planner import _native_candidate_v2

    oversized, safe = (512, 256) if cores == 4 else (640, 512)
    required = 7935744 if cores == 4 else 7864320
    text, cache = _text_preflight_owner(exact_chunk_size=oversized if exact else None, retained=True)
    vision = types.SimpleNamespace(config=types.SimpleNamespace(spatial_merge_size=2))
    model = types.SimpleNamespace(
        model=types.SimpleNamespace(language_model=text, visual=vision),
        config=types.SimpleNamespace(image_token_id=99))
    ids = torch.ones(1, 4095, dtype=torch.long)
    ids[:, :64] = 99
    pixels, grid = torch.zeros(256, 1536), torch.tensor([[1, 16, 16]])
    own = 225280
    reserved = {"text": own if text_started else 0, "vision": 0}
    events, budgets = [], []
    state = text._rpu_qwen3_5
    vision._rpu_vision_graph_cache = rpu_backend.graph.GraphCache()
    for label, graph in (("decode", state.graph_cache),
                         ("prefill", state.prefill_graph_cache),
                         ("vision", vision._rpu_vision_graph_cache)):
        original = graph.clear
        def clear(original=original, label=label):
            events.append("clear-" + label)
            original()
        monkeypatch.setattr(graph, "clear", clear)
    state.prefill_graph = types.SimpleNamespace(invalidate=lambda: events.append("clear-raw-text"))
    vision._rpu_vision_debug_graph = types.SimpleNamespace(invalidate=lambda: events.append("clear-raw-vision"))

    def install(*args, **kwargs):
        events.append("install")
        vision._rpu_vision_handle = 17
        return 17

    def reserve_vision(handle, length):
        assert (handle, length) == (17, 256)
        events.append("vision-fixed")
        reserved["vision"] = vision_bytes

    def reserve_text(handle, length):
        assert (handle, length) == (7, 4096)
        events.append("text-fixed")
        reserved["text"] = own

    def domain(handle, execution, logical):
        assert (handle, execution, logical) == (7, 4096, 4095)
        usable = 8 * 1024 * 1024 - 1024
        available = min(usable * 99 // 100 - own,
                        usable - 65536 - own - reserved["vision"])
        budgets.append(available)
        rows = [_native_candidate_v2(execution, (chunk,) * 3,
                logical_len=logical, descriptor_version=3, graph_lifecycle=1)
                for chunk, temporary in ((safe, 6291456), (oversized, required))
                if temporary <= available]
        return (1, len(rows), *(word for row in rows for word in (len(row), *row)))

    monkeypatch.setattr(vision_module, "_install_lazy_qwen3_5_vision", install)
    for name, fn in {
        "qwen3_5_vision_get_resolved_chunk_size": lambda handle: 0,
        "qwen3_5_get_resolved_chunk_size": lambda handle: 128 if text_started else 0,
        "qwen3_5_vision_prepare_persistent_spm": reserve_vision,
        "qwen3_5_prepare_persistent_spm": reserve_text,
        "qwen3_5_resolve_prefill_stage_domain": domain,
    }.items():
        monkeypatch.setattr(torch.ops.rpu, name, fn, raising=False)
    early = text_module._preflight_qwen3_5_text(text, cache, 4095, multimodal_prefill=True)
    assert early[1] == oversized
    assert vision_module._prepare_qwen3_5_multimodal_spm(model, ids, pixels, grid, early[0])
    assert events == ["install", "clear-decode", "clear-prefill",
                      "clear-vision", "clear-raw-text", "clear-raw-vision",
                      "vision-fixed"] + ([] if text_started else ["text-fixed"])
    assert reserved == {"text": own, "vision": vision_bytes}
    if exact:
        with pytest.raises(PlannerRejectError):
            text_module._preflight_qwen3_5_text(text, cache, 4095, multimodal_prefill=True)
    else:
        final = text_module._preflight_qwen3_5_text(text, cache, 4095, multimodal_prefill=True)
        assert final[1] == safe and final[2] is not early[2]
        assert text_module._preflight_qwen3_5_text(text, cache, 4095, multimodal_prefill=True)[2] is final[2]
    assert budgets == [8078428, 7697408 if cores == 4 else 7723008]
    assert not vision_module._prepare_qwen3_5_multimodal_spm(model, ids, pixels, grid, early[0])
    assert text._rpu_qwen3_5.multimodal_spm_identity == (7, 17)


@pytest.mark.parametrize("failure", ["grid", "tokens", "pixels", "ready"])
def test_multimodal_persistent_prepare_rejects_before_install(monkeypatch, failure):
    text, _ = _text_preflight_owner()
    vision = types.SimpleNamespace(config=types.SimpleNamespace(spatial_merge_size=2))
    model = types.SimpleNamespace(model=types.SimpleNamespace(language_model=text, visual=vision),
                                 config=types.SimpleNamespace(image_token_id=99))
    ids, pixels, grid = torch.full((1, 64), 99), torch.zeros(256, 1536), torch.tensor([[1, 16, 16]])
    if failure == "grid":
        grid[0, 0] = 2
    elif failure == "tokens":
        ids[0, 0] = 1
    elif failure == "pixels":
        pixels = pixels[:255]
    else:
        text._rpu_qwen3_5.graph_cache = types.SimpleNamespace(is_frozen=lambda: True)
    monkeypatch.setattr(vision_module, "_install_lazy_qwen3_5_vision",
                        lambda *a: pytest.fail("rejection must precede installation"))
    with pytest.raises((ValueError, RuntimeError)):
        vision_module._prepare_qwen3_5_multimodal_spm(model, ids, pixels, grid, 4096)
    assert not hasattr(text._rpu_qwen3_5, "multimodal_spm_identity")


def test_vision_descriptor_rejects_different_outer_and_inner_span_partitions():
    descriptor = _vision_descriptor(
        384, 896, 128, span_count=3, camera_batch_count=3,
    )
    descriptor[17:23] = [0, 112, 112, 144, 256, 128]
    with pytest.raises(RuntimeError, match="outer/inner descriptor spans disagree"):
        vision_module._decode_qwen3_5_vision_plan_descriptor(
            descriptor, logical_len=384, temporal_num_frames=1,
            camera_batch_count=3, compact_input=False,
        )


def test_temporal_vision_uses_same_planner_with_independent_stage_fields(
    monkeypatch,
):
    owner = types.SimpleNamespace(
        _rpu_vision_exact_chunk_size=None,
        _rpu_vision_padding_rows="auto",
        _rpu_vision_padding_budget=0,
        _rpu_vision_kv_cache=types.SimpleNamespace(max_seq_len=4096),
    )
    descriptor = _vision_descriptor(
        4608,
        768,
        4608,
        span_count=3,
        temporal_num_frames=6,
        camera_batch_count=3,
        compact_input=True,
        stage_fingerprint=(0x89ABCDEF << 32) | 0x01234567,
    )
    descriptor[9:11] = [0x76543210, 0xFEDCBA98]
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_vision_resolve_prefill_domain",
        lambda *_args: [descriptor],
        raising=False,
    )

    result = vision_module._plan_qwen3_5_vision_group(
        owner,
        7,
        4608,
        group_id=0,
        temporal_num_frames=6,
        camera_batch_count=3,
        compact_input=True,
    )

    assert result.selected is not None
    assert result.selected.stage_tuple.as_tuple() == (768, 4608, 4608)
    metadata = dict(result.selected.stage_tuple.physical_metadata)
    assert metadata["span_count"] == 3
    assert metadata["merger_chunk"] == 512
    assert metadata["stage_plan_fingerprint_hi"] == 0x89ABCDEF
    assert metadata["layout_hash_hi"] == 0x76543210


@pytest.mark.parametrize(
    "message",
    [
        "RPU_PLANNER_REJECT:EXACT_MISMATCH:chunk fails validity check",
        "RPU_PLANNER_REJECT:EXACT_MISMATCH:chunk exceeds SPM budget",
    ],
)
def test_vision_exact_native_rejection_is_typed(monkeypatch, message):
    owner = types.SimpleNamespace(
        _rpu_vision_exact_chunk_size=512,
        _rpu_vision_padding_rows="auto",
        _rpu_vision_padding_budget=0,
        _rpu_vision_kv_cache=types.SimpleNamespace(max_seq_len=512),
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_vision_resolve_prefill_domain",
        lambda *_args: (_ for _ in ()).throw(RuntimeError(message)),
        raising=False,
    )
    with pytest.raises(PlannerRejectError) as info:
        vision_module._plan_qwen3_5_vision_group(
            owner, 7, 512, group_id=0
        )
    assert info.value.code == "EXACT_MISMATCH"
    assert info.value.requested == 512


def test_vision_group_does_not_mask_native_programming_error(monkeypatch):
    owner = types.SimpleNamespace(
        _rpu_vision_exact_chunk_size=None,
        _rpu_vision_padding_rows="auto",
        _rpu_vision_padding_budget=0,
        _rpu_vision_kv_cache=types.SimpleNamespace(max_seq_len=512),
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_vision_resolve_prefill_domain",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError("invalid or stale vision handle")
        ),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="invalid or stale vision handle"):
        vision_module._plan_qwen3_5_vision_group(
            owner, 7, 256, group_id=0
        )


def test_invalid_cold_option_fails_before_process_owner_claim(monkeypatch):
    model = types.SimpleNamespace()
    adapter = adapter_module.Qwen3_5Adapter.__new__(
        adapter_module.Qwen3_5Adapter
    )
    adapter.model = model
    adapter._rpu_is_ready = False
    claimed = []
    monkeypatch.setattr(
        adapter_module, "_claim_live_instance", lambda owner: claimed.append(owner)
    )
    monkeypatch.setenv("QWEN3_5_TEXT_CHUNK", "63")

    with pytest.raises(ValueError, match="at least 64"):
        adapter.to_rpu(max_seq_len=128)

    assert claimed == []
    assert not hasattr(model, "_rpu_swizzle_started")


def test_text_install_clears_prepared_graph_cache_on_base_exception(monkeypatch):
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_linear_acc32", lambda *args: None, raising=False)
    events = []
    execution = normalize_rpu_execution(None, entry_point="test")
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_linear_acc32", lambda *_: None, raising=False)

    class GraphCache:
        def cache_invariant_ok(self):
            return True
        def clear(self):
            events.append("graph-cache-clear")

    class InstallInterrupted(BaseException):
        pass

    monkeypatch.setattr(rpu_backend.graph, "GraphCache", GraphCache)
    monkeypatch.setattr(rpu_backend.graph, "Graph", lambda: types.SimpleNamespace(invalidate=lambda: None))

    def interrupt(*_args, **kwargs):
        assert kwargs["execution_config"] is execution
        with pytest.raises(TypeError):
            execution["prefill"] = {"chunk_size": 64}
        raise InstallInterrupted

    monkeypatch.setattr(
        text_module, "_install_qwen3_5_text_for_rpu_impl", interrupt
    )
    with pytest.raises(InstallInterrupted):
        text_module.install_qwen3_5_text_for_rpu(
            types.SimpleNamespace(), types.SimpleNamespace(intermediate_size=6144),
            execution_config=execution
        )
    assert events == ["graph-cache-clear"]


def test_failed_outer_install_clears_graph_before_handle_and_keeps_poison(monkeypatch):
    events = []

    class GraphCache:
        def cache_invariant_ok(self):
            return True
        def clear(self):
            events.append("graph-cache-clear")

    class Finalizer:
        alive = True

        def detach(self):
            self.alive = False

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_destroy", lambda _h: events.append("native-destroy"), raising=False)
    original_forward = object()
    model = types.SimpleNamespace(
        forward=object(),
        _lm_head_w_rpu_keepalive=object(),
        _rpu_swizzle_started=True,
    )
    inner = types.SimpleNamespace(
        _rpu_qwen3_5=types.SimpleNamespace(
            handle=17, graph_cache=GraphCache(), handle_finalizer=Finalizer()
        )
    )
    adapter = types.SimpleNamespace(
        _handle=17, _graph_cache=object(), _rpu_is_ready=True
    )

    adapter_module._cleanup_failed_text_install(
        adapter, model, inner, True, original_forward
    )

    assert events == ["graph-cache-clear", "native-destroy"]
    assert not hasattr(inner, "_rpu_qwen3_5")
    assert not hasattr(model, "_lm_head_w_rpu_keepalive")
    assert model.forward is original_forward
    assert model._rpu_swizzle_started is True
    assert adapter._handle is None
    assert adapter._graph_cache is None
    assert adapter._rpu_is_ready is False


def test_outer_adapter_never_releases_owner_after_failed_mutation(monkeypatch):
    from rpu_backend.api import causal_lm
    from rpu_backend.api._execution import bind_execution_session

    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(causal_lm, "_LIVE_TERMINAL_REASON", None)
    monkeypatch.setattr(adapter_module, "_claim_live_instance", causal_lm._claim_live_instance)
    primary = RuntimeError("failed during weight mutation")

    class Model:
        def __init__(self):
            self.config, self.model = object(), types.SimpleNamespace()

        def half(self):
            raise primary

    model = Model()
    owner = adapter_module.Qwen3_5Adapter.__new__(adapter_module.Qwen3_5Adapter)
    owner.model, owner._rpu_is_ready = model, False
    owner._execution_session = bind_execution_session(
        model, {}, entry_point="failed mutation ownership", graph_mode="BOUNDED_ONESHOT")
    try:
        with pytest.raises(RuntimeError) as caught:
            owner.to_rpu(max_seq_len=128)
        assert caught.value is primary
        assert model._rpu_swizzle_started and causal_lm._LIVE_REF() is model
        with pytest.raises(causal_lm.RPUSingleHandleError):
            causal_lm._claim_live_instance(Model())
    finally:
        causal_lm._release_live_instance(model)
    retirement = Path(text_module.__file__).read_text().split(
        "def _retire_qwen3_5_text_state(", 1)[1].split(
        "def _retire_failed_qwen3_5_text_install(", 1)[0]
    assert retirement.index("cache.clear()") < retirement.index("qwen3_5_destroy(handle)") < retirement.index("finalizer.detach()")


def test_failed_install_removes_forward_override_when_none_existed():
    class Model:
        def forward(self):
            return "class-forward"

    model = Model()
    model.forward = object()
    adapter = types.SimpleNamespace(
        _handle=None, _graph_cache=None, _rpu_is_ready=False
    )

    adapter_module._cleanup_failed_text_install(
        adapter, model, None, False, None
    )

    assert "forward" not in vars(model)
    assert model.forward() == "class-forward"


def test_outer_to_rpu_failure_keeps_model_poisoned_and_cleans_runtime(
    monkeypatch, request,
):
    events = []

    class GraphCache:
        def cache_invariant_ok(self):
            return True
        def clear(self):
            events.append("graph-cache-clear")

    class Finalizer:
        alive = True

        def detach(self):
            self.alive = False

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_destroy", lambda _h: events.append("native-destroy"), raising=False)
    class Inner:
        embed_tokens = object()

        def to(self, device):
            events.append(f"inner-to-{device}")
            return self

    class Model:
        def __init__(self):
            self.model = Inner()
            self.config = object()
            self.forward = object()

        def half(self):
            events.append("half")
            return self

        def get_output_embeddings(self):
            return types.SimpleNamespace(weight=torch.ones(1, 1))

    model = Model()
    original_forward = model.forward
    adapter = adapter_module.Qwen3_5Adapter.__new__(
        adapter_module.Qwen3_5Adapter
    )
    adapter.model = model
    adapter._rpu_is_ready = False
    from rpu_backend.api._execution import bind_execution_session

    adapter._execution_session = bind_execution_session(model, {}, entry_point="outer Q35 fixture",
                                                       graph_mode="BOUNDED_ONESHOT")
    adapter._execution_session._bind_planner_owner(model.model)

    def claim(owner):
        assert owner is model
        events.append("claim")

    def install(
        inner, _config, _max_seq_len, *, execution_config=None,
        _resolved_options=None, _graph_runtime_policy=None, _public_prefill_graph_mode=None,
    ):
        assert _resolved_options is not None
        assert _public_prefill_graph_mode == "BOUNDED_ONESHOT"
        inner._rpu_qwen3_5 = _native_fixture_state(request,
            handle=31, graph_cache=GraphCache(), handle_finalizer=Finalizer()
        )
        return 31

    def fail_lm_head_transform(*_args, **_kwargs):
        raise RuntimeError("lm-head-transform-failed")

    monkeypatch.setattr(adapter_module, "_claim_live_instance", claim)
    monkeypatch.setattr(
        adapter_module, "install_qwen3_5_text_for_rpu", install
    )
    monkeypatch.setattr(
        adapter_module,
        "transform_linear_weight",
        fail_lm_head_transform,
    )

    with pytest.raises(RuntimeError, match="lm-head-transform-failed"):
        adapter.to_rpu(max_seq_len=128)

    assert events == [
        "claim",
        "half",
        "inner-to-rpu",
        "graph-cache-clear",
        "native-destroy",
    ]
    assert model._rpu_swizzle_started is True
    assert not hasattr(model.model, "_rpu_qwen3_5")
    assert model.forward is original_forward
    assert adapter._handle is None
    assert adapter._graph_cache is None

    with pytest.raises(adapter_module.RPUBackendError, match="prior install failed"):
        adapter.to_rpu(max_seq_len=128)


class _ReadyFinalizer:
    def __init__(self, *, alive=True):
        self.alive = alive


def _complete_text_model(request):
    class Model:
        config = object()

        def to(self, *args, **kwargs):
            return self

    model = Model()
    inner = types.SimpleNamespace()
    state = _native_fixture_state(request,
        handle=47,
        handle_finalizer=_ReadyFinalizer(),
        graph_cache=object(),
        max_seq_len=128,
        num_layers=1,
        hidden_size=16,
        rotary_dim=8,
        rope_theta=10000.0,
        mrope_section=[],
        padding_budget=0,
        padding_rows="auto",
        execution_config={},
        control_snapshot={},
        exact_chunk_size=None,
        chunk_envelope=(192, 128),
        vision_control_options=(0, None, "auto", 0, {}),
        last_a6_plan=None,
        prefill_graph=object(),
    )
    inner._rpu_qwen3_5 = state
    inner._rpu_qwen3_5_text_install_started = True
    model.model = inner
    model._lm_head_w_rpu_keepalive = object()
    model._rpu_swizzle_started = True
    model._rpu_swizzled = True
    model.forward = types.MethodType(
        adapter_module._rpu_qwen3_5_forward, model
    )
    return model, state


def test_complete_text_runtime_restores_repeated_adapter_mirrors(monkeypatch, request):
    model, state = _complete_text_model(request)
    monkeypatch.setattr(adapter_module, "_check_profile", lambda config: None)
    monkeypatch.setattr(
        adapter_module, "_validate_layer_types", lambda config: None
    )
    monkeypatch.setattr(
        adapter_module,
        "_claim_live_instance",
        lambda owner: pytest.fail("complete runtime must not be reclaimed"),
    )
    monkeypatch.setattr(
        adapter_module,
        "_validate_qwen3_5_execution_reconfigure",
        lambda _adapter, _config: None,
    )

    def apply(adapter, config, _generation, *, force_native=False):
        assert force_native is False
        adapter._rpu_execution = config
        state.execution_config = config

    monkeypatch.setattr(
        adapter_module, "_apply_qwen3_5_execution_config", apply
    )

    adapter = adapter_module.Qwen3_5Adapter(model)
    repeated = adapter_module.Qwen3_5Adapter(model)

    assert adapter._rpu_is_ready is True
    assert adapter._handle == state.handle
    assert adapter._graph_cache is state.graph_cache
    assert adapter.to_rpu(max_seq_len=1) is model
    assert repeated._execution_session is adapter._execution_session
    assert repeated._rpu_execution is adapter._rpu_execution
    assert repeated._rpu_is_ready is True
    assert repeated._handle == state.handle
    assert repeated._graph_cache is state.graph_cache
    assert repeated.to_rpu(max_seq_len=1) is model

    published = reconfigure_rpu_execution(
        repeated, {"prefill": {"chunk_size": 128}}
    )
    assert model._rpu_execution is published
    assert adapter._rpu_execution is published
    assert repeated._rpu_execution is published
    assert state.execution_config is published


@pytest.mark.parametrize("dead_finalizer", [False, True])
def test_incomplete_text_ready_marker_is_rejected_before_claim(
    monkeypatch, dead_finalizer, request
):
    if dead_finalizer:
        model, state = _complete_text_model(request)
        state.handle_finalizer.detach()
    else:
        model = types.SimpleNamespace(
            config=object(),
            model=types.SimpleNamespace(),
            _rpu_swizzled=True,
            to=lambda *args, **kwargs: model,
        )
    monkeypatch.setattr(adapter_module, "_check_profile", lambda config: None)
    monkeypatch.setattr(
        adapter_module, "_validate_layer_types", lambda config: None
    )
    claimed = []
    monkeypatch.setattr(
        adapter_module, "_claim_live_instance", lambda owner: claimed.append(owner)
    )
    adapter = adapter_module.Qwen3_5Adapter(model)

    with pytest.raises(
        adapter_module.RPUBackendError, match="runtime ownership is incomplete"
    ):
        adapter.to_rpu(max_seq_len=128)

    assert claimed == []
    assert adapter._handle is None
    assert adapter._graph_cache is None


def test_text_ready_publish_base_exception_cleans_and_poison_model(monkeypatch, request):
    events = []

    class InstallInterrupted(BaseException):
        pass

    class GraphCache:
        def cache_invariant_ok(self):
            return True

        def clear(self):
            events.append("graph-cache-clear")

    class Finalizer:
        alive = True

        def detach(self):
            self.alive = False

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_destroy", lambda _h: events.append("native-destroy"), raising=False)
    class Inner:
        embed_tokens = object()

        def to(self, device):
            events.append(f"inner-to-{device}")
            return self

    class Model:
        def __init__(self):
            self.config = object()
            self.model = Inner()
            self._fail_ready_publish = False

        def __setattr__(self, name, value):
            if name == "_rpu_swizzled" and self.__dict__.get(
                "_fail_ready_publish", False
            ):
                self.__dict__["_fail_ready_publish"] = False
                raise InstallInterrupted("synthetic ready publish interruption")
            super().__setattr__(name, value)

        def to(self, *args, **kwargs):
            return self

        def half(self):
            events.append("half")
            return self

        def forward(self, *args, **kwargs):
            return "original-forward"

        def get_output_embeddings(self):
            return types.SimpleNamespace(weight=torch.ones(1, 1))

    class Movable:
        def to(self, device):
            return self

    def install(
        inner, _config, _max_seq_len, *, execution_config=None,
        _resolved_options=None, _graph_runtime_policy=None, _public_prefill_graph_mode=None,
    ):
        assert _resolved_options is not None
        assert _public_prefill_graph_mode == "BOUNDED_ONESHOT"
        inner._rpu_qwen3_5_text_install_started = True
        inner._rpu_qwen3_5 = _native_fixture_state(request,
            handle=53, graph_cache=GraphCache(), handle_finalizer=Finalizer()
        )
        return 53

    model = Model()
    original_forward = model.forward
    monkeypatch.setattr(adapter_module, "_check_profile", lambda config: None)
    monkeypatch.setattr(
        adapter_module, "_validate_layer_types", lambda config: None
    )
    monkeypatch.setattr(
        adapter_module, "_claim_live_instance", lambda owner: events.append("claim")
    )
    monkeypatch.setattr(
        adapter_module, "install_qwen3_5_text_for_rpu", install
    )
    monkeypatch.setattr(
        adapter_module, "transform_linear_weight", lambda *args: Movable()
    )
    adapter = adapter_module.Qwen3_5Adapter(model)
    model._fail_ready_publish = True

    with pytest.raises(InstallInterrupted, match="ready publish interruption"):
        adapter.to_rpu(max_seq_len=128)

    assert events == [
        "claim",
        "half",
        "inner-to-rpu",
        "graph-cache-clear",
        "native-destroy",
    ]
    assert adapter._rpu_is_ready is False
    assert adapter._handle is None
    assert adapter._graph_cache is None
    assert model._rpu_swizzle_started is True
    assert not hasattr(model, "_rpu_swizzled")
    assert not hasattr(model.model, "_rpu_qwen3_5")
    assert model.forward.__func__ is original_forward.__func__
    with pytest.raises(adapter_module.RPUBackendError, match="swizzled|device"):
        model.to("cpu")


def test_vision_preparation_keys_geometry_and_reuses_previous_group(monkeypatch):
    owner = types.SimpleNamespace(
        _rpu_vision_kv_cache=types.SimpleNamespace(max_seq_len=256),
        _rpu_vision_graph_cache=rpu_backend.graph.GraphCache(max_entries=1))
    calls = []
    def resolve(handle, length, frames, cameras, compact):
        calls.append((handle, length, frames, cameras, compact))
        return [_vision_descriptor(length, 128, 128, span_count=cameras,
                                   temporal_num_frames=frames,
                                   camera_batch_count=cameras, compact_input=compact)]
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_resolve_prefill_domain", resolve, raising=False)
    first = vision_module._plan_qwen3_5_vision_group(owner, 17, 256, group_id=0)
    vision_module._plan_qwen3_5_vision_group(owner, 17, 256, group_id=1, camera_batch_count=2)
    assert vision_module._plan_qwen3_5_vision_group(owner, 17, 256, group_id=2) is first
    assert calls == [(17, 256, 1, 1, False), (17, 256, 1, 2, False)]
    assert len(calls) == 2


@pytest.mark.parametrize("changed", ["use_attention_mask", "is_causal"])
def test_rhino_action_preparation_does_not_hide_semantic_rejection(monkeypatch, changed):
    from rpu_backend.adapters.rhinovla.fused import plan_rhino_action_execution
    from test_execution_planner import _native_candidate_v2

    owner = types.SimpleNamespace(_rpu_handle=17, _rpu_graph_cache=rpu_backend.graph.GraphCache())
    calls = []
    def resolve(handle, length, logical, prefix, kv_len, mask, causal, exact):
        calls.append((mask, causal))
        if not mask or causal:
            raise RuntimeError("native action attention semantics rejected")
        descriptor = _native_candidate_v2(length, (32,) * 3, logical_len=logical,
            position=prefix, descriptor_version=3, graph_lifecycle=4,
            routes=((1, 101, 2, 0, (), 0),))
        return [1, 1, len(descriptor), *descriptor]
    monkeypatch.setattr(torch.ops.rpu, "rhino_vla_planner_cache_identity", lambda handle: (handle, 1), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "rhino_vla_resolve_action_stage_domain", resolve, raising=False)
    arguments = dict(logical_len=32, execution_len=32, prefix_len=80, kv_len=112,
                     use_attention_mask=True, is_causal=False)
    first = plan_rhino_action_execution(owner, **arguments)
    assert plan_rhino_action_execution(owner, **arguments) is first
    arguments[changed] = not arguments[changed]
    for _ in range(2):
        with pytest.raises(RuntimeError, match="native action attention semantics rejected"):
            plan_rhino_action_execution(owner, **arguments)
    arguments[changed] = not arguments[changed]
    assert plan_rhino_action_execution(owner, **arguments) is first
    assert len(calls) == 3  # Failed preparation publishes no plan and keeps the valid signature.
