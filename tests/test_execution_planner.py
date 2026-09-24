"""Host-only contracts for the canonical A6 execution planner."""
from __future__ import annotations

import ast
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "rpu_execution_planner_under_test",
    _ROOT / "python/rpu_backend/runtime/execution_planner.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _domain(_execution_len):
    # Deliberately non-monotone feasibility is supplied below; the planner must
    # enumerate every tuple before applying the fixed lexicographic score.
    return [(64, 64, 64), (128, 96, 128), (256, 256, 256)]


def _native_candidate(execution_len, chunks, *, position=0, fingerprint=1):
    schedules = []
    counts = []
    for capacity in chunks:
        stage = []
        offset = 0
        while offset < execution_len:
            length = min(capacity, execution_len - offset)
            stage.extend((len(stage) // 4, offset, length,
                          position + offset + length))
            offset += length
        schedules.extend(stage)
        counts.append(len(stage) // 4)
    descriptor = [
        1, 0, *chunks, fingerprint >> 32, fingerprint & 0xFFFFFFFF,
        0, 0, 0, 0, *counts, 1,
        *schedules, 0, execution_len, -1,
    ]
    descriptor[1] = len(descriptor)
    return descriptor


def _native_candidate_v2(
    execution_len, chunks, *, logical_len=None, position=0, fingerprint=1,
    graph_lifecycle=1, descriptor_version=2, routes=None,
):
    descriptor = _native_candidate(
        execution_len, chunks, position=position, fingerprint=fingerprint,
    )
    descriptor[0] = descriptor_version
    logical_len = execution_len if logical_len is None else logical_len
    execution_padding_rows = execution_len - logical_len
    if routes is None:
        routes = (
            (1, 101, 2, 0, (), 0),
            (5, 501, 4, 1, (0, execution_len - 16, 16), 0),
        )
    manifest_values = [
        1, logical_len, execution_len, execution_padding_rows,
        position + logical_len, execution_len, graph_lifecycle, 3, len(routes),
    ]
    for family, site_id, selector, flags, arguments, invocation in routes:
        manifest_values.extend((
            family, site_id, selector, flags, len(arguments), *arguments,
        ))
        if invocation:
            manifest_values.append(invocation)
    manifest_fingerprint = _MODULE._physical_manifest_fingerprint(
        manifest_values
    )
    descriptor.extend((
        1, logical_len, execution_len, execution_padding_rows,
        position + logical_len, execution_len, graph_lifecycle, 3, len(routes),
        manifest_fingerprint >> 32,
        manifest_fingerprint & 0xFFFFFFFF,
    ))
    for family, site_id, selector, flags, arguments, invocation in routes:
        descriptor.extend((
            family, site_id, selector, flags, len(arguments),
        ))
        if descriptor_version == 3:
            descriptor.append(invocation)
        descriptor.extend(arguments)
    descriptor[1] = len(descriptor)
    return descriptor


def _native_domain(execution_len, *chunks, position=0):
    descriptors = [
        _native_candidate(execution_len, row, position=position,
                          fingerprint=index + 1)
        for index, row in enumerate(chunks)
    ]
    return [
        1, len(descriptors),
        *(word for descriptor in descriptors
          for word in (len(descriptor), *descriptor)),
    ]


def _native_domain_v2(
    execution_len, chunks, *, logical_len=None, position=0,
    graph_lifecycle=1,
):
    descriptor = _native_candidate_v2(
        execution_len,
        chunks,
        logical_len=logical_len,
        position=position,
        graph_lifecycle=graph_lifecycle,
    )
    return [1, 1, len(descriptor), *descriptor]


def _native_domain_v3(execution_len, chunks, *, position=0):
    descriptor = _native_candidate_v2(
        execution_len,
        chunks,
        position=position,
        descriptor_version=3,
        routes=(
            (1, 101, 2, 0, (), 0),
            (1, 101, 2, 0, (), 1),
            (5, 501, 1, 0, (0, execution_len), 0),
        ),
    )
    return [1, 1, len(descriptor), *descriptor]


def test_a6_enumerates_full_domain_and_uses_fixed_score():
    seen = []

    def feasible(candidate):
        seen.append(candidate.as_dict())
        return candidate.stage_tuple.input_chunk != 128 or candidate.execution_len != 128

    result = _MODULE.plan_a6(
        100,
        request={"padding_budget": 28},
        alignment=4,
        tuple_domain=_domain,
        feasible=feasible,
    )

    assert result.status == "ACCEPTED"
    assert result.search_complete is True
    assert result.optimality == "EXACT_DOMAIN"
    assert result.selected is not None
    assert result.selected.score == min(
        item["score"] for item in seen if item["feasible"]
    )
    assert result.candidate_count == 24  # 8 execution lengths × 3 tuples
    assert result.feasible_count < result.candidate_count
    assert len(seen) == result.candidate_count


def test_a6_comparator_has_no_hidden_execution_alignment_preference():
    result = _MODULE.plan_a6(
        15,
        request={"padding_budget": 1},
        tuple_domain=lambda _n: [(16, 16, 16)],
    )
    assert result.selected is not None
    assert result.selected.execution_len == 15
    assert result.selected.score == (1, 0, 1, 16, 16, 16)
    assert len(result.selected.score) == 6


def test_zero_padding_budget_still_allows_mandatory_alignment():
    result = _MODULE.plan_a6(
        65,
        request={"padding_budget": 0},
        alignment=64,
        physical_limit=256,
        tuple_domain=lambda _n: [(64, 64, 64)],
    )
    assert [
        candidate.execution_len for candidate in result.candidates
    ] == [128]
    assert result.selected is not None
    assert result.selected.padding_rows == 63


def test_padding_budget_extends_search_after_mandatory_alignment():
    result = _MODULE.plan_a6(
        833,
        request={"padding_budget": 64},
        alignment=64,
        physical_limit=1024,
        tuple_domain=lambda _n: [(64, 64, 64)],
    )
    assert sorted({
        candidate.execution_len for candidate in result.candidates
    }) == [896, 960]


def test_pi05_expresses_v16_as_feasibility_not_a_private_score_callback():
    source = (
        _ROOT
        / "python/rpu_backend/adapters/pi05/runtime.py"
    ).read_text(encoding="utf-8")
    start = source.index("def _plan_pi05_prefix_execution(")
    end = source.index("\ndef _prefill_prefix_embs(", start)
    helper = source[start:end]
    assert helper.count("plan_bounded_prefill_execution(") == 1
    assert "prefix_pad16 = _cold_prefix_pad16_request(self)" in helper
    assert "alignment=1 if explicit_padding_rows else" in helper
    assert "_prefix_pad16_enabled()" not in helper
    assert "gemma_resolve_prefill_stage_domain" in helper
    assert "preferred_execution_multiple" not in helper
    assert "candidate_score" not in helper
    assert "queue_owner_id=int(handle)" in helper
    assert '"component:language_model"' in helper
    assert '"execution_generation", component_generation' in helper


def test_prefill_owner_and_component_generation_bind_graph_identity():
    def plan(generation):
        return _MODULE.plan_prefill(
            64,
            64,
            0,
            padding_rows=0,
            queue_owner_id=17,
            physical_metadata=(
                ("component:language_model", 1),
                ("execution_generation", generation),
            ),
            resolve_stage_domain=lambda _n: [(64, 64, 64)],
        )

    first = plan(0)
    second = plan(1)
    assert first.as_dict(include_candidates=False)[
        "owner_identity_status"
    ] == "RESOLVED"
    assert first.graph_key_words() != second.graph_key_words()
    assert dict(first.selected.stage_tuple.physical_metadata) == {
        "component:language_model": 1,
        "execution_generation": 0,
    }


@pytest.mark.parametrize(
    "relative_path, planner_names",
    [
        ("python/rpu_backend/runtime/decoder.py", {"plan_bounded_prefill_execution"}),
        ("python/rpu_backend/adapters/qwen3_vl/__init__.py", {"_plan_causal_prefill_execution"}),
        ("python/rpu_backend/adapters/qwen3_vl/vision.py", {"plan_bounded_prefill_execution"}),
        ("python/rpu_backend/adapters/qwen3_5/text.py", {"plan_bounded_prefill_execution", "_plan_prefill_execution"}),
        ("python/rpu_backend/adapters/qwen3_5/vision.py", {"plan_bounded_prefill_execution"}),
        ("python/rpu_backend/adapters/dinov3.py", {"plan_bounded_prefill_execution"}),
        ("python/rpu_backend/adapters/internvla_n1/backbone.py", {"plan_bounded_prefill_execution"}),
        ("python/rpu_backend/adapters/internvla_n1/nextdit.py", {"plan_bounded_prefill_execution"}),
        (
            "python/rpu_backend/adapters/internvla_n1/execution.py",
            {"plan_native_component_execution"},
        ),
        (
            "python/rpu_backend/adapters/navdp/runtime.py",
            {"plan_native_component_execution"},
        ),
        ("python/rpu_backend/adapters/internvla_n1/rgbd_rpu.py", {"plan_bounded_prefill_execution"}),
    ],
)
def test_production_a6_calls_bind_stable_queue_owner(
    relative_path, planner_names,
):
    path = _ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in planner_names
    ]
    assert calls, relative_path
    for call in calls:
        owner = next(
            (kw.value for kw in call.keywords if kw.arg == "queue_owner_id"),
            None,
        )
        assert owner is not None, f"{relative_path}:{call.lineno} has no queue owner"
        assert not (
            isinstance(owner, ast.Constant) and owner.value in (None, 0, False)
        ), f"{relative_path}:{call.lineno} leaves owner identity unresolved"


@pytest.mark.parametrize(
    "relative_path",
    [
        "python/rpu_backend/runtime/decoder.py",
        "python/rpu_backend/adapters/gr00t/runtime.py",
        "python/rpu_backend/adapters/halo/text_cache.py",
        "python/rpu_backend/adapters/halo/vit_cache.py",
        "python/rpu_backend/adapters/internvla_n1/backbone.py",
        "python/rpu_backend/adapters/pi05/runtime.py",
        "python/rpu_backend/adapters/qwen3_5/text.py",
        "python/rpu_backend/adapters/qwen3_vl/__init__.py",
        "python/rpu_backend/adapters/wall_oss/action.py",
        "python/rpu_backend/adapters/wall_oss/llm.py",
    ],
)
def test_production_prefill_planner_calls_fail_closed_without_stage_domain(
    relative_path,
):
    path = _ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id in {
                "plan_bounded_prefill_execution",
                "_plan_causal_prefill_execution",
                "_plan_halo_fmb_execution",
            }
        )
    ]
    assert calls, relative_path
    for call in calls:
        resolver = next(
            (
                keyword.value
                for keyword in call.keywords
                if keyword.arg == "resolve_stage_domain"
            ),
            None,
        )
        assert resolver is not None, (
            f"{relative_path}:{call.lineno} silently uses the scalar bridge"
        )
        if isinstance(call.func, ast.Name) and call.func.id == "_plan_halo_fmb_execution":
            assert any(
                isinstance(node, ast.Attribute)
                and node.attr in {
                    "causal_decoder_resolve_prefill_stage_domain",
                    "siglip_resolve_stage_domain",
                }
                for node in ast.walk(resolver)
            ), f"{relative_path}:{call.lineno} wrapper has no native stage domain"


def test_native_domain_ops_and_forward_winner_are_registered():
    registrations = (
        _ROOT / "src/core/rpu_dispatch_registrations.inc"
    ).read_text(encoding="utf-8")
    assert (
        'm.def("causal_decoder_resolve_prefill_stage_domain('
        "int handle, int execution_len, int position, bool is_causal, "
        "int mask_kv_len, int rope_mode=0, int graph_lifecycle=1, int logical_len=0, "
        "int planning_chunk_size_override=-1) -> int[]\""
    ) in registrations
    assert (
        'm.def("qwen3_5_resolve_prefill_stage_domain('
        "int handle, int execution_len, int logical_len, "
        "int planning_chunk_size_override=-1) -> int[]\""
    ) in registrations
    assert 'm.def("gemma_resolve_prefill_stage_domain(int handle,' in registrations
    assert registrations.count("int[] planned_stage_descriptor=[]") >= 3


def _load_pi05_prefix_planner(*, pad16_enabled):
    path = _ROOT / "python/rpu_backend/adapters/pi05/runtime.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_plan_pi05_prefix_execution"
    )

    def shared_prefill(real_len, physical_limit, padding_budget, **kwargs):
        sink = kwargs.pop("plan_result_sink")
        owner = kwargs.pop("execution_owner")
        assert owner.paligemma_with_expert
        assert kwargs.pop("plan_signature") == ()
        assert kwargs.pop("graph_cache") is owner.paligemma_with_expert.paligemma.model.language_model._rpu_gemma_graph_cache
        assert kwargs.pop("execution_component") == "language_model"
        assert kwargs.pop("execution_stage") == "prefill"
        assert kwargs.pop("execution_native") == ("gemma", 17)
        result = _MODULE.plan_prefill(
            real_len,
            physical_limit,
            padding_budget,
            **kwargs,
        )
        sink(result)
        assert result.selected is not None
        return result.selected.execution_len, result.selected.stage_tuple.compute_chunk

    fake_rpu = SimpleNamespace(
        gemma_resolve_prefill_stage_domain=(
            lambda _handle, execution_len, requested, logical_len: _native_domain(
                execution_len,
                (requested or 16,) * 3,
            )
        ),
    )
    namespace = {
        "torch": SimpleNamespace(ops=SimpleNamespace(rpu=fake_rpu)),
        "plan_bounded_prefill_execution": shared_prefill,
        "GRAPH_COMPOSITE_CHILD": _MODULE.GRAPH_COMPOSITE_CHILD,
        "PI05_TEXT_COMPONENT": "language_model",
        "_cold_prefix_pad16_request": lambda _owner: pad16_enabled,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["_plan_pi05_prefix_execution"]


def _pi05_planner_owner(*, execution=None, chunk_size=0, generation=0):
    language_model = SimpleNamespace(
        _rpu_gemma_graph_cache=object(),
        _rpu_vlm_decoder_handle=17,
        _rpu_chunk_size=chunk_size,
        _rpu_max_seq_len=2048,
    )
    return SimpleNamespace(
        paligemma_with_expert=SimpleNamespace(
            paligemma=SimpleNamespace(
                model=SimpleNamespace(language_model=language_model)
            )
        ),
        config=SimpleNamespace(chunk_size=50),
        _rpu_execution={} if execution is None else execution,
        _pi05_execution_component_generations={"language_model": generation},
    )


def test_pi05_default_and_external_prefix_controls_share_one_a6_path():
    enabled = _load_pi05_prefix_planner(pad16_enabled=True)
    disabled = _load_pi05_prefix_planner(pad16_enabled=False)

    default_on = enabled(_pi05_planner_owner(), 17)
    default_off = disabled(_pi05_planner_owner(), 17)
    exact_padding = enabled(
        _pi05_planner_owner(execution={"prefill": {"padding_rows": 3}}), 17
    )
    auto_padding = enabled(
        _pi05_planner_owner(execution={"prefill": {"padding_budget": 15}}), 17
    )
    exact_chunk = enabled(_pi05_planner_owner(chunk_size=32), 17)
    next_generation = enabled(_pi05_planner_owner(generation=1), 17)

    assert default_on[:2] == (32, 16)
    assert default_off[:2] == (17, 16)
    assert exact_padding[:2] == (20, 16)
    assert auto_padding[:2] == (32, 16)
    assert exact_chunk[:2] == (32, 32)
    for _, _, result in (
        default_on, default_off, exact_padding, auto_padding, exact_chunk
    ):
        assert result.plan_digest
        assert result.domain_digest
        assert result.optimality == ("EXTERNAL_EXACT" if result.request.mode == "EXACT" else "CAPABILITY_SELECTED")
        assert result.as_dict(include_candidates=False)[
            "owner_identity_status"
        ] == "RESOLVED"
        assert result.selection_scope == ("EXTERNALLY_PINNED_NATIVE_CANDIDATE" if result.request.mode == "EXACT"
                                          else "STAGE_DOMAIN_ROUTE_CAPABILITY_SELECTED")
    assert default_on[2].graph_key_words() != next_generation[2].graph_key_words()


def test_exact_request_is_a_constraint_not_a_second_selector():
    result = _MODULE.plan_a6(
        100,
        request={"mode": "EXACT", "chunk_size": 128},
        alignment=4,
        tuple_domain=lambda _n: [(64, 64, 64), (128, 128, 128)],
    )
    assert result.selected is not None
    assert result.selected.stage_tuple.as_tuple() == (128, 128, 128)

    caller_owned_input = _MODULE.plan_a6(
        100,
        request={"mode": "EXACT", "chunk_size": 128},
        tuple_domain=lambda _n: [(100, 128, 128)],
    )
    assert caller_owned_input.selected is not None
    assert caller_owned_input.selected.stage_tuple.input_chunk == 100

    with pytest.raises(_MODULE.PlannerRejectError, match="EXACT_MISMATCH"):
        _MODULE.plan_a6(
            100,
            request={"mode": "EXACT", "chunk_size": 512},
            alignment=4,
            tuple_domain=lambda _n: [(64, 64, 64), (128, 128, 128)],
        )

    inferred = _MODULE.PlannerRequest.from_mapping({"chunk_size": 128})
    assert inferred.mode == _MODULE.REQUEST_EXACT
    with pytest.raises(ValueError, match="AUTO.*conflicts"):
        _MODULE.PlannerRequest.from_mapping(
            {"mode": "AUTO", "chunk_size": 128}
        )


def test_unregistered_compat_clamp_is_rejected():
    with pytest.raises(ValueError, match="must be one of"):
        _MODULE.PlannerRequest.from_mapping({
            "mode": "COMPAT_CLAMP", "chunk_size": 64,
        })


def test_domain_order_and_callback_errors_are_not_silently_recovered():
    with pytest.raises(_MODULE.PlannerRejectError, match="unsorted"):
        _MODULE.plan_a6(
            32,
            tuple_domain=lambda _n: [(128, 128, 128), (64, 64, 64)],
        )

    def broken(_candidate):
        raise RuntimeError("oracle bug")

    with pytest.raises(RuntimeError, match="oracle bug"):
        _MODULE.plan_a6(32, tuple_domain=lambda _n: [(64, 64, 64)], feasible=broken)


@pytest.mark.parametrize("verdict", [None, 0, object(), ""])
def test_feasibility_callback_has_a_strict_return_contract(verdict):
    with pytest.raises(TypeError, match="feasible callback must return"):
        _MODULE.plan_a6(
            32,
            tuple_domain=lambda _n: [(64, 64, 64)],
            feasible=lambda _candidate: verdict,
        )


def test_execution_length_limit_is_checked_before_native_oracle():
    calls = []

    with pytest.raises(_MODULE.PlannerRejectError, match="SEARCH_LIMIT"):
        _MODULE.plan_prefill(
            1,
            1000,
            999,
            resolve_stage_domain=lambda execution_len: (
                calls.append(execution_len) or ((16, 16, 16),)
            ),
        )

    assert calls == []


def test_execution_length_limit_is_checked_before_range_materialization(
    monkeypatch,
):
    monkeypatch.setattr(
        _MODULE,
        "range",
        lambda *_args: pytest.fail("candidate range was materialized"),
        raising=False,
    )
    with pytest.raises(_MODULE.PlannerRejectError, match="SEARCH_LIMIT"):
        _MODULE.plan_a6(
            1,
            request={"padding_budget": 10_000},
            physical_limit=10_000,
            tuple_domain=lambda _n: pytest.fail("native oracle was called"),
        )


def test_exact_padding_rejects_native_int64_overflow_before_oracle():
    with pytest.raises(_MODULE.PlannerRejectError, match="CAPABILITY"):
        _MODULE.plan_a6(
            (1 << 63) - 1,
            request={"padding_rows": 1},
            tuple_domain=lambda _n: pytest.fail("native oracle was called"),
        )


def test_auto_padding_rejects_native_int64_overflow_before_oracle():
    with pytest.raises(_MODULE.PlannerRejectError, match="CAPABILITY"):
        _MODULE.plan_a6(
            (1 << 63) - 1,
            request={"padding_budget": 1},
            tuple_domain=lambda _n: pytest.fail("native oracle was called"),
        )


def test_prefill_bounds_structured_domain_before_materializing_it():
    calls = []
    yielded = []

    def resolve(execution_len):
        calls.append(execution_len)
        for index in range(_MODULE.MAX_A6_PAIRS + 2):
            yielded.append(index)
            yield (16, 16, 16)

    with pytest.raises(_MODULE.PlannerRejectError, match="SEARCH_LIMIT"):
        _MODULE.plan_prefill(16, 16, 0, resolve_stage_domain=resolve)

    assert calls == [16]
    assert len(yielded) == _MODULE.MAX_A6_PAIRS + 1


def test_planner_rejection_is_part_of_the_public_backend_error_hierarchy():
    from rpu_backend.api import PlannerRejectError as public_error
    from rpu_backend.api.errors import RPUBackendError

    assert issubclass(public_error, RPUBackendError)
    assert issubclass(public_error, RuntimeError)


def test_pair_limit_bounds_domain_iteration():
    yielded = []

    def domain(_execution_len):
        for chunk in (16, 32, 48, 64):
            yielded.append(chunk)
            yield (chunk, chunk, chunk)

    with pytest.raises(_MODULE.PlannerRejectError, match="SEARCH_LIMIT"):
        _MODULE.plan_a6(32, tuple_domain=domain, max_pairs=2)

    assert yielded == [16, 32, 48]


def test_candidate_log_is_bounded_and_digest_is_identity_sensitive():
    records = []
    one = _MODULE.plan_a6(
        32,
        tuple_domain=lambda _n: [(32, 32, 32), (64, 64, 64)],
        logger=records.append,
        max_diag_records=1,
    )
    two = _MODULE.plan_a6(
        32,
        tuple_domain=lambda _n: [(32, 32, 32), (64, 64, 64)],
        graph_mode=_MODULE.GRAPH_BOUNDED_ONESHOT,
    )
    assert one.truncated is True
    assert records[0]["kind"] == "candidate"
    assert records[-1]["kind"] == "summary"
    assert one.plan_digest != two.plan_digest
    assert one.domain_digest == two.domain_digest


def test_physical_metadata_is_canonical_and_part_of_plan_identity():
    one = _MODULE.plan_a6(
        64,
        tuple_domain=lambda _n: [
            _MODULE.StageTuple(
                64, 64, 64,
                (("stage_plan_fingerprint", 7), ("merger_chunk", 512)),
            )
        ],
    )
    two = _MODULE.plan_a6(
        64,
        tuple_domain=lambda _n: [
            _MODULE.StageTuple(
                64, 64, 64,
                (("merger_chunk", 256), ("stage_plan_fingerprint", 7)),
            )
        ],
    )
    assert one.plan_digest != two.plan_digest
    assert one.selected is not None
    assert one.selected.as_dict()["physical_metadata"] == {
        "merger_chunk": 512,
        "stage_plan_fingerprint": 7,
    }


def test_stage_tuple_rejects_fractional_values_without_truncation():
    with pytest.raises(ValueError, match="integer"):
        _MODULE.StageTuple.from_value((64.5, 64, 64))
    with pytest.raises(ValueError, match="integer"):
        _MODULE.StageTuple.from_value(_MODULE.StageTuple(64.0, 64, 64))

    # INPUT may describe an already-produced full semantic span; only the
    # native QKV/compute stages have the universal v16 alignment contract.
    assert _MODULE.StageTuple.from_value((17, 64, 64)).input_chunk == 17
    with pytest.raises(ValueError, match="QKV/compute"):
        _MODULE.StageTuple.from_value((64, 17, 64))


def test_request_and_graph_metadata_are_preserved_in_result():
    result = _MODULE.plan_a6(
        32,
        request={"mode": "EXACT", "chunk_size": 64, "padding_budget": 8},
        tuple_domain=lambda _n: [(64, 64, 64)],
        graph_mode=_MODULE.GRAPH_BOUNDED_ONESHOT,
    )
    payload = result.as_dict(include_candidates=False)
    assert payload["request"] == {
        "mode": "EXACT", "chunk_size": 64,
        "padding_rows": None, "padding_budget": 8,
    }
    assert payload["requested_chunk_size"] == 64
    assert payload["resolved_chunk_size"] == 64
    assert payload["graph_mode"] == _MODULE.GRAPH_BOUNDED_ONESHOT


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ({"mode": "EXACT"}, "requires chunk_size"),
        ({"chunk_size": 17}, "multiple of 16"),
        ({"padding_rows": 1, "padding_budget": 2}, "conflicts"),
        ({"unexpected": 1}, "unknown field"),
    ],
)
def test_request_validation(value, match):
    with pytest.raises((ValueError, TypeError), match=match):
        _MODULE.PlannerRequest.from_mapping(value)


def test_prefill_stage_domain_logs_typed_rejections():
    records = []

    def resolve(execution_len):
        if execution_len == 104:
            raise _MODULE.PlannerRejectError(
                "CAPABILITY", requested=execution_len,
                detail="SPM candidate rejected",
            )
        return ((64, 64, 64),)

    result = _MODULE.plan_prefill(
        100,
        120,
        8,
        resolve_stage_domain=resolve,
        logger=records.append,
        request_id="prefill-test",
    )
    assert result.selected is not None
    assert result.selected.execution_len == 100
    assert result.optimality == "CAPABILITY_SELECTED"
    assert result.selection_scope == "STAGE_DOMAIN_ROUTE_CAPABILITY_SELECTED"
    assert any(item["kind"] == "domain_reject" for item in records)
    assert sum(item["kind"] == "domain_reject" for item in records) == 1
    assert result.candidate_count == result.feasible_count + result.rejected_count
    assert result.rejected_count == 1
    assert result.domain_digest


def test_prefill_enumerates_full_native_stage_domain_without_claiming_route_optimality():
    seen = []
    records = []

    def stage_domain(execution_len):
        seen.append(execution_len)
        # Domain-v1 is the production native ABI. Deliberately expose two
        # candidates so the host comparator selects after full enumeration.
        return _native_domain(
            execution_len, (64, 64, 64), (128, 128, 128),
        )

    result = _MODULE.plan_prefill(
        100,
        104,
        4,
        alignment=4,
        resolve_stage_domain=stage_domain,
        logger=records.append,
    )

    assert seen == [100, 104]
    assert result.candidate_count == 4
    assert result.search_complete is True
    assert result.selected is not None
    assert result.selected.score == min(
        candidate.score for candidate in result.candidates if candidate.feasible
    )
    assert result.selected.execution_len == 100
    assert result.selected.stage_tuple.as_tuple() == (128, 128, 128)
    assert result.optimality == "CAPABILITY_SELECTED"
    assert result.selection_scope == "STAGE_DOMAIN_ROUTE_CAPABILITY_SELECTED"
    assert records[-1]["kind"] == "summary"
    assert records[-1]["optimality"] == "CAPABILITY_SELECTED"
    assert all(
        record.get("optimality") != "EXACT_DOMAIN" for record in records
    )


def test_native_domain_v1_drives_physical_score_descriptor_and_graph_key():
    result = _MODULE.plan_prefill(
        100,
        100,
        0,
        resolve_stage_domain=lambda execution_len: _native_domain(
            execution_len, (100, 64, 64),
        ),
    )
    assert result.selected is not None
    assert result.selected.num_chunks == (1, 2, 2)
    assert result.selected.tail_deficit == (0, 28, 28)
    assert result.selected.score == (2, 0, 28, 100, 64, 64)
    assert result.selected.stage_tuple.physical_descriptor
    payload = result.as_dict(include_candidates=True)
    assert payload["selected"]["physical_descriptor"] == (
        result.selected.stage_tuple.physical_descriptor
    )
    assert all("physical_descriptor" not in row for row in payload["candidates"])
    assert result.graph_key_words() == tuple(
        int(result.physical_plan_digest[index:index + 8], 16)
        for index in range(0, 64, 8)
    )


def test_native_candidate_v2_preserves_complete_physical_route_manifest():
    result = _MODULE.plan_prefill(
        96,
        112,
        16,
        padding_rows=16,
        resolve_stage_domain=lambda execution_len: _native_domain_v2(
            execution_len,
            (112, 112, 112),
            logical_len=96,
            graph_lifecycle=2,
        ),
        graph_mode=_MODULE.GRAPH_BOUNDED_ONESHOT,
    )
    assert result.selected is not None
    assert result.selected.execution_len == 112
    metadata = dict(result.selected.stage_tuple.physical_metadata)
    assert metadata["descriptor_version"] == 2
    assert metadata["manifest_state"] == 1
    assert metadata["logical_length"] == 96
    assert metadata["physical_length"] == 112
    assert metadata["execution_padding_rows"] == 16
    assert metadata["kv_logical_length"] == 96
    assert metadata["kv_insert_physical_rows"] == 112
    assert metadata["graph_lifecycle"] == 2
    assert metadata["linear_acc_policy"] == 3
    assert metadata["route_count"] == 2
    assert metadata["manifest_fingerprint"] != 0


def test_native_candidate_v3_accepts_repeated_site_invocations():
    result = _MODULE.plan_prefill(
        64,
        64,
        0,
        resolve_stage_domain=lambda execution_len: _native_domain_v3(
            execution_len, (32, 32, 32),
        ),
    )
    assert result.selected is not None
    metadata = dict(result.selected.stage_tuple.physical_metadata)
    assert metadata["descriptor_version"] == 3
    assert metadata["route_count"] == 3
    assert metadata["manifest_fingerprint"] != 0


def test_native_candidate_v3_accepts_collective_route_family():
    descriptor = _native_candidate_v2(
        64,
        (32, 32, 32),
        descriptor_version=3,
        routes=((10, 1001, 1, 0, (), 0),),
    )
    result = _MODULE.plan_prefill(
        64,
        64,
        0,
        resolve_stage_domain=lambda _execution_len: [
            1, 1, len(descriptor), *descriptor,
        ],
    )
    assert result.selected is not None
    assert dict(result.selected.stage_tuple.physical_metadata)[
        "route_count"
    ] == 1


def test_native_candidate_v3_joint_physical_domain_uses_capability_rank():
    spm = _native_candidate_v2(
        64,
        (64, 64, 64),
        descriptor_version=3,
        routes=(
            (1, 101, 2, 0, (), 0),
            (1, 101, 2, 0, (), 1),
            (4, 401, 2, 1 << 57, (), 0),
        ),
    )
    ddr = _native_candidate_v2(
        64,
        (64, 64, 64),
        descriptor_version=3,
        routes=(
            (1, 101, 1, 0, (), 0),
            (4, 401, 1, 1 << 56, (), 0),
        ),
    )
    result = _MODULE.plan_prefill(
        64,
        64,
        0,
        resolve_stage_domain=lambda _execution_len: [
            1, 2, len(spm), *spm, len(ddr), *ddr,
        ],
    )

    assert len(result.candidates) == 2
    assert result.selected is not None
    metadata = dict(result.selected.stage_tuple.physical_metadata)
    assert metadata["capability_attention_ddr_site_count"] == 0
    assert metadata["raw_attention_site_count"] == 1
    assert metadata["rope_table_residency"] == 2
    assert metadata["capability_rope_residency_rank"] == 0
    assert metadata["hardware_cost_certificate"] == 0
    assert metadata["selector_state"] == 1
    assert result.optimality == "CAPABILITY_SELECTED"
    assert result.selection_scope == "STAGE_DOMAIN_ROUTE_CAPABILITY_SELECTED"

    ddr_only = _MODULE.plan_prefill(
        64,
        64,
        0,
        resolve_stage_domain=lambda _execution_len: [
            1, 1, len(ddr), *ddr,
        ],
    )
    assert result.physical_plan_digest != ddr_only.physical_plan_digest
    assert result.graph_key_words() != ddr_only.graph_key_words()


def test_hardware_cost_reader_binds_profile_dependencies_and_full_domain():
    # Mock data tests the reader, never installed as a production certificate.
    descriptors = [
        _native_candidate_v2(
            64, (chunk, chunk, chunk), descriptor_version=3,
            routes=((1, 101, attention, 0, (), 0),
                    (4, 401, rope, 1 << (55 + rope), (), 0)),
        )
        for chunk, attention, rope in ((32, 1, 1), (64, 2, 2))
    ]
    raw = [1, len(descriptors)]
    for descriptor in descriptors:
        raw.extend((len(descriptor), *descriptor))
    rows = _MODULE._decode_native_stage_domain(
        raw, execution_len=64, logical_len=64, position=0,
        graph_mode=_MODULE.GRAPH_RETAINED_CACHE,
    )
    scope = _MODULE.PlannerCostScope("a" * 64, "b" * 64, "e" * 64)
    domain_digest, keys = _MODULE.planner_cost_domain(
        {64: rows}, scope=scope, logical_len=64, position=0,
        graph_mode=_MODULE.GRAPH_RETAINED_CACHE,
    )
    certificate = _MODULE.PlannerCostCertificate(
        scope, domain_digest,
        tuple((key, 100 if row.compute_chunk == 32 else 200)
              for (_length, row), key in keys.items()),
        "c" * 64, sealed=True,
    )
    def plan(**kwargs):
        return _MODULE.plan_prefill(
            64, 64, 0, resolve_stage_domain=lambda _length: raw, **kwargs,
        )

    selected = plan(cost_scope=scope, cost_certificates=(certificate,))
    assert selected.selected.stage_tuple.compute_chunk == 32
    assert selected.optimality == "HARDWARE_COST_CERTIFIED"
    assert selected.selection_scope == "MEASURED_NATIVE_DOMAIN_ONLY"
    expected_key = next(key for (_length, row), key in keys.items() if row.compute_chunk == 32)
    assert selected.selected_cost_candidate_digest == expected_key
    assert selected.as_dict()["selected_cost_candidate_digest"] == expected_key
    assert selected.as_dict()["hardware_cost"] == {
        "status": "SEALED_EXACT_DOMAIN",
        "profile_identity": scope.profile_identity,
        "dependency_identity": scope.dependency_identity,
        "runtime_identity": scope.runtime_identity,
        "domain_digest": domain_digest,
        "receipt_sha256": "c" * 64,
        "metric": "same_session_median_device_ns",
    }
    # An external exact chunk remains authoritative even if slower measured.
    exact = plan(cost_scope=scope, cost_certificates=(certificate,), exact_chunk_size=64)
    assert exact.selected.stage_tuple.compute_chunk == 64
    assert exact.optimality == "EXTERNAL_EXACT"
    assert exact.selection_scope == "EXTERNALLY_PINNED_NATIVE_CANDIDATE"
    # AUTO's feasible set is not the caller's narrower EXACT cost domain.
    assert exact.cost_certificate is None
    assert exact.cost_domain_digest != domain_digest
    assert len(exact.cost_feasible_pairs) == 1
    assert exact.native_cost_feasible_pairs == selected.native_cost_feasible_pairs
    assert len(exact.native_cost_feasible_pairs) == 2
    assert exact.selected_cost_candidate_digest == next(
        key for (_length, row), key in keys.items() if row.compute_chunk == 64)
    # The measured catalog is per call/owner: it cannot leak into another owner
    # with identical geometry or mutate a retained result after selection.
    assert plan(cost_scope=scope).optimality == "CAPABILITY_SELECTED"
    assert plan(cost_scope=scope).selected_cost_candidate_digest in keys.values()
    assert plan().as_dict()["selected_cost_candidate_digest"] is None
    assert selected.cost_certificate is certificate
    for generation_field in ("execution_generation", "component_generation"):
        before, after = (
            plan(cost_scope=scope, cost_certificates=(certificate,),
                 physical_metadata=((generation_field, generation),))
            for generation in (0, 1)
        )
        assert before.cost_domain_digest == after.cost_domain_digest == domain_digest
        assert before.selected.stage_tuple.compute_chunk == after.selected.stage_tuple.compute_chunk == 32
        assert after.optimality == "HARDWARE_COST_CERTIFIED"
        assert before.selected_cost_candidate_digest == after.selected_cost_candidate_digest == expected_key
        assert before.graph_key_words() != after.graph_key_words()
        assert before.plan_digest != after.plan_digest
    assert plan(
        cost_scope=scope, cost_certificates=(certificate,),
        physical_metadata=(("physical_feature", 1),),
    ).optimality == "CAPABILITY_SELECTED"
    with pytest.raises(TypeError, match="immutable certificate tuple"):
        plan(cost_scope=scope, cost_certificates=[certificate])
    with pytest.raises(ValueError, match="bound cost_scope"):
        plan(cost_certificates=(certificate,))
    assert plan().optimality == "CAPABILITY_SELECTED"
    assert plan(request_id=scope.profile_identity).as_dict()["hardware_cost"]["status"] == "PROFILE_IDENTITY_UNBOUND"
    for changed in (
        replace(scope, profile_identity="d" * 64),
        replace(scope, dependency_identity="d" * 64),
        replace(scope, runtime_identity="d" * 64),
    ):
        result = plan(cost_scope=changed, cost_certificates=(certificate,))
        assert result.optimality == "CAPABILITY_SELECTED"
        assert result.as_dict()["hardware_cost"]["status"] == "NO_MATCHING_SEALED_CERTIFICATE"

    # Same profile and chunk do not authorize another site/shape/domain.
    altered = replace(rows[0], physical_descriptor=(*rows[0].physical_descriptor, 42))
    different_domain = _MODULE.planner_cost_domain(
        {64: (altered, rows[1])}, scope=scope, logical_len=64, position=0,
        graph_mode=_MODULE.GRAPH_RETAINED_CACHE,
    )[0]
    assert different_domain != domain_digest
    rank, found, changed_digest, changed_keys = _MODULE._apply_cost_certificate(
        {64: (altered, rows[1])}, scope=scope, logical_len=64, position=0,
        certificates=(certificate,),
        graph_mode=_MODULE.GRAPH_RETAINED_CACHE,
    )
    assert found is None
    assert changed_digest == different_domain
    assert rank[64][0] == altered
    for field, value in (("logical_len", 63), ("position", 16),
                         ("graph_mode", _MODULE.GRAPH_BOUNDED_ONESHOT)):
        arguments = dict(scope=scope, logical_len=64, position=0,
                         graph_mode=_MODULE.GRAPH_RETAINED_CACHE)
        arguments[field] = value
        assert _MODULE.planner_cost_domain({64: rows}, **arguments)[0] != domain_digest

    assert plan(cost_scope=scope, cost_certificates=(
        replace(certificate, sealed=False),
    )).optimality == "CAPABILITY_SELECTED"
    with pytest.raises(ValueError, match="exact candidate domain"):
        plan(cost_scope=scope, cost_certificates=(
            replace(certificate, candidate_costs=certificate.candidate_costs[:1]),
        ))
    with pytest.raises(ValueError, match="SHA-256"):
        _MODULE.PlannerCostScope("owner-class", "b" * 64, "e" * 64)


def test_costs_price_only_actual_feasible_rows_and_bind_rejected_raw_domain():
    raw = [1, 2]
    for chunk in (32, 64):
        descriptor = _native_candidate_v2(
            80, (chunk, chunk, chunk), logical_len=64,
            descriptor_version=3, routes=((1, 101, 1, 0, (), 0),))
        raw.extend((len(descriptor), *descriptor))
    scope = _MODULE.PlannerCostScope("a" * 64, "b" * 64, "c" * 64)

    def plan(**kwargs):
        return _MODULE.plan_prefill(
            64, 80, 0, padding_rows=16, cost_scope=scope,
            resolve_stage_domain=lambda _n: raw, **kwargs)

    # Both tails would be padding-only at P64/C32 or C64. This must fail
    # before any certificate can authorize a launch.
    records = []
    with pytest.raises(_MODULE.PlannerRejectError, match="NO_FEASIBLE"):
        plan(logger=records.append)
    candidates = [record for record in records if record["kind"] == "candidate"]
    assert len(candidates) == 2
    assert all(record["reject_reason"] == "padding_only_tail" for record in candidates)
    assert records[-1]["kind"] == "summary" and records[-1]["status"] == "REJECTED"

    raw = [1, 2]
    for chunk in (32, 64):
        descriptor = _native_candidate_v2(
            64, (chunk, chunk, chunk), descriptor_version=3,
            routes=((1, 101, 1, 0, (), 0),))
        raw.extend((len(descriptor), *descriptor))
    rows = _MODULE._decode_native_stage_domain(
        raw, execution_len=64, logical_len=64, position=0,
        graph_mode=_MODULE.GRAPH_RETAINED_CACHE)
    def capped(**kwargs):
        return _MODULE.plan_prefill(
            64, 64, 0, max_stage_chunk_size=32, cost_scope=scope,
            resolve_stage_domain=lambda _n: raw, **kwargs)
    baseline = capped()
    assert baseline.candidate_count == 2 and baseline.feasible_count == 1
    assert baseline.cost_feasible_pairs == ((64, _MODULE.StageTuple.from_value(rows[0])),)
    assert baseline.native_cost_feasible_pairs == baseline.cost_feasible_pairs
    exact = capped(exact_chunk_size=32)
    assert exact.native_cost_feasible_pairs == baseline.native_cost_feasible_pairs
    assert exact.optimality == "EXTERNAL_EXACT"
    arguments = dict(scope=scope, logical_len=64, position=0,
                     graph_mode=_MODULE.GRAPH_RETAINED_CACHE)
    digest, keys = _MODULE.planner_cost_domain(
        {64: rows}, **arguments, feasible_pairs=baseline.cost_feasible_pairs)
    assert len(keys) == 1 and digest == baseline.cost_domain_digest
    # Neither dropping the rejected raw row nor pretending it is feasible can
    # borrow the same certificate, even though both may choose C32.
    assert digest != _MODULE.planner_cost_domain({64: rows[:1]}, **arguments)[0]
    assert digest != _MODULE.planner_cost_domain({64: rows}, **arguments)[0]
    certificate = _MODULE.PlannerCostCertificate(
        scope, digest, tuple((key, 10) for key in keys.values()), "d" * 64, True)
    ranked = capped(cost_certificates=(certificate,))
    assert ranked.cost_certificate is certificate
    assert ranked.cost_feasible_pairs == baseline.cost_feasible_pairs
    rejected = next(candidate for candidate in ranked.candidates if not candidate.feasible)
    assert rejected.reject_reason == "stage_chunk_limit"
    assert "hardware_cost_score" not in dict(rejected.stage_tuple.physical_metadata)


def test_calibration_selects_actual_same_chunk_route_without_shrinking_auto_domain():
    raw = [1, 2]
    for attention in (2, 1):
        descriptor = _native_candidate_v2(
            64, (64, 64, 64), descriptor_version=3,
            routes=((1, 101, attention, 0, (), 0),))
        raw.extend((len(descriptor), *descriptor))
    scope = _MODULE.PlannerCostScope("a" * 64, "b" * 64, "c" * 64)
    def plan(**kwargs):
        return _MODULE.plan_prefill(64, 64, 0, cost_scope=scope,
                                    resolve_stage_domain=lambda _n: raw, **kwargs)
    baseline = plan()
    rows = tuple(candidate.stage_tuple for candidate in baseline.candidates)
    digest, keys = _MODULE.planner_cost_domain(
        {64: rows}, scope=scope, logical_len=64, position=0,
        graph_mode=_MODULE.GRAPH_RETAINED_CACHE,
        feasible_pairs=baseline.cost_feasible_pairs)
    target = next(row for row in rows if row != baseline.selected.stage_tuple)
    selection = _MODULE.PlannerCalibrationSelection(scope, digest, keys[(64, target)])
    actual = plan(calibration_selection=selection)
    assert actual.selected.stage_tuple is target or actual.selected.stage_tuple == target
    assert actual.candidates == baseline.candidates
    assert actual.cost_feasible_pairs == baseline.cost_feasible_pairs
    assert actual.request == baseline.request and actual.request.mode == "AUTO"
    assert actual.selected_cost_candidate_digest == keys[(64, target)]
    assert actual.optimality == actual.selection_scope == "P7_CALIBRATION_ONLY"
    assert actual.cost_certificate is None
    assert actual.graph_key_words() != baseline.graph_key_words()
    assert actual.as_dict()["selected"]["physical_descriptor"] == target.physical_descriptor
    assert all("hardware_cost_score" not in dict(row.physical_metadata) for row in rows)
    for bad in (replace(selection, domain_digest="d" * 64),
                replace(selection, candidate_digest="d" * 64)):
        with pytest.raises(ValueError, match="actual.*domain"):
            plan(calibration_selection=bad)
    with pytest.raises(ValueError, match="AUTO domain"):
        plan(calibration_selection=selection, exact_chunk_size=64)


def test_domain_scoped_calibration_keeps_non_target_binding_and_validation():
    scope = _MODULE.PlannerCostScope("a" * 64, "b" * 64, "c" * 64)
    foreign = replace(scope, profile_identity="f" * 64)
    def domain(length):
        descriptors = [_native_candidate_v2(
            length, (chunk, chunk, chunk), descriptor_version=3,
            routes=((1, 101, 1, 0, (), 0),)) for chunk in (32, 64)]
        return (1, len(descriptors), *(word for descriptor in descriptors
                for word in (len(descriptor), *descriptor)))
    def plan(length=64, **kwargs):
        return _MODULE.plan_prefill(length, length, 0, resolve_stage_domain=domain, **kwargs)
    baseline = plan(cost_scope=scope)
    target = next(candidate for candidate in baseline.candidates
                  if candidate.stage_tuple != baseline.selected.stage_tuple)
    _, keys = _MODULE.planner_cost_domain(
        {64: tuple(candidate.stage_tuple for candidate in baseline.candidates)},
        scope=scope, logical_len=64, position=0,
        graph_mode=_MODULE.GRAPH_RETAINED_CACHE,
        feasible_pairs=baseline.cost_feasible_pairs)
    selection = _MODULE.PlannerCalibrationSelection(
        scope, baseline.cost_domain_digest, keys[(64, target.stage_tuple)], domain_scoped=True)
    matched = plan(calibration_selection=selection)
    assert matched.selected.stage_tuple == target.stage_tuple
    assert matched.cost_scope == scope and matched.selection_scope == "P7_CALIBRATION_ONLY"
    assert matched.candidates == baseline.candidates
    # Visiting another shape neither grants the calibration's scope nor alters
    # the original owner's costs. A sealed non-target winner remains binding.
    other = plan(128, cost_scope=foreign)
    digest, other_keys = _MODULE.planner_cost_domain(
        {128: tuple(candidate.stage_tuple for candidate in other.candidates)},
        scope=foreign, logical_len=128, position=0,
        graph_mode=_MODULE.GRAPH_RETAINED_CACHE,
        feasible_pairs=other.cost_feasible_pairs)
    certificate = _MODULE.PlannerCostCertificate(
        foreign, digest, tuple((key, 10 if row.compute_chunk == 32 else 20)
                              for (_length, row), key in other_keys.items()), "d" * 64, True)
    for options in ({}, {"cost_scope": foreign},
                    {"cost_scope": foreign, "cost_certificates": (certificate,)},
                    {"cost_scope": foreign, "exact_chunk_size": 32}):
        actual = plan(128, calibration_selection=selection, **options)
        assert actual.as_dict() == plan(128, **options).as_dict()
    with pytest.raises(TypeError, match="immutable certificate tuple"):
        plan(128, calibration_selection=selection, cost_scope=foreign,
             cost_certificates=[certificate])
    with pytest.raises(ValueError, match="bound cost_scope"):
        plan(128, calibration_selection=selection, cost_certificates=(certificate,))
    # Matching geometry cannot override another installed scope/certificate.
    for options in ({"cost_scope": foreign},
                    {"cost_scope": scope, "cost_certificates": (certificate,)}):
        with pytest.raises(ValueError, match="exact scope and no outer"):
            plan(calibration_selection=selection, **options)
    with pytest.raises(ValueError, match="actual domain"):
        plan(calibration_selection=replace(selection, candidate_digest="e" * 64))
    # The prior strict contract remains strict, even for a valid other shape.
    with pytest.raises(ValueError, match="AUTO domain"):
        plan(128, cost_scope=scope,
             calibration_selection=replace(selection, domain_scoped=False))
    for invalid in (1, "true", None):
        with pytest.raises(ValueError, match="must be bool"):
            replace(selection, domain_scoped=invalid)


def test_native_calibration_readmits_the_winner_and_keeps_original_cost_key():
    scope = _MODULE.PlannerCostScope("a" * 64, "b" * 64, "c" * 64)
    live = []
    def domain(length):
        live.append(length)
        descriptor = _native_candidate_v2(
            length, (96, 96, 96), logical_len=64, descriptor_version=3,
            routes=((1, 101, 1, 0, (), 0),))
        return (1, 1, len(descriptor), *descriptor)
    def plan(**kwargs):
        return _MODULE.plan_prefill(64, 80, 16, alignment=16,
                                    cost_scope=scope, resolve_stage_domain=domain, **kwargs)
    baseline = plan()
    target = baseline.candidates[0]
    digest, keys = _MODULE.planner_cost_domain(
        {item.execution_len: (item.stage_tuple,) for item in baseline.candidates},
        scope=scope, logical_len=64, position=0, graph_mode=_MODULE.GRAPH_RETAINED_CACHE,
        feasible_pairs=baseline.cost_feasible_pairs)
    key = keys[(target.execution_len, target.stage_tuple)]
    selection = _MODULE.PlannerCalibrationSelection(scope, digest, key, (501, 0, 2))
    minted = []
    def mint(length, row, route):
        assert live == [64, 80, 64] and length == 64 and route == (501, 0, 2)
        descriptor = _native_candidate_v2(
            length, row.as_tuple(), descriptor_version=3,
            routes=((1, 101, 2, 0, (), 0),))
        decoded, = _MODULE._decode_native_stage_domain(
            (1, 1, len(descriptor), *descriptor), execution_len=length,
            logical_len=64, position=0, graph_mode=_MODULE.GRAPH_RETAINED_CACHE)
        minted.append(_MODULE.StageTuple.from_value(decoded))
        return minted[0]
    live.clear()
    result = plan(calibration_selection=selection, calibration_mint=mint)
    assert live == [64, 80, 64] and result.selected.stage_tuple == minted[0]
    assert result.selected_cost_candidate_digest == key
    assert result.cost_domain_digest == digest
    assert result.candidates == baseline.candidates
    assert result.graph_key_words() != baseline.graph_key_words()
    # Re-admission is a proof check, not permission to choose a fresh domain.
    live.clear()
    def changed(length):
        raw = domain(length)
        if len(live) == 3:
            descriptor = _native_candidate_v2(
                length, (96, 96, 96), descriptor_version=3,
                routes=((1, 101, 2, 0, (), 0),))
            return (1, 1, len(descriptor), *descriptor)
        return raw
    with pytest.raises(ValueError, match="changed during actual re-admission"):
        _MODULE.plan_prefill(64, 80, 16, alignment=16, cost_scope=scope,
                             calibration_selection=selection, calibration_mint=mint,
                             resolve_stage_domain=changed)


@pytest.mark.parametrize("fault", [None, "outer_exact", "domain", "infeasible_candidate", "mint_topology"])
def test_native_exact_calibration_retains_request_and_never_claims_automatic_optimality(fault):
    scope = _MODULE.PlannerCostScope("a" * 64, "b" * 64, "c" * 64)
    raw = [1, 3]
    for chunk in (32, 64, 128):
        words = _native_candidate_v2(128, (chunk,) * 3, descriptor_version=3)
        raw.extend((len(words), *words))
    def plan(**kwargs):
        return _MODULE.plan_prefill(128, 128, 0, padding_rows=0, exact_chunk_size=128,
            cost_scope=scope, resolve_stage_domain=lambda _length: raw, **kwargs)
    baseline = plan()
    selected = baseline.selected.stage_tuple
    digest, keys = _MODULE.planner_cost_domain(
        {128: tuple(row.stage_tuple for row in baseline.candidates)}, scope=scope,
        logical_len=128, position=0, graph_mode="RETAINED_CACHE",
        feasible_pairs=baseline.cost_feasible_pairs)
    selection = _MODULE.PlannerCalibrationSelection(scope, digest, keys[(128, selected)], (501, 0, 2))
    if fault == "outer_exact":
        selection = replace(selection, native_route=None)
    elif fault == "domain":
        selection = replace(selection, domain_digest="d" * 64)
    elif fault == "infeasible_candidate":
        _, raw_keys = _MODULE.planner_cost_domain(
            {128: tuple(row.stage_tuple for row in baseline.candidates)}, scope=scope,
            logical_len=128, position=0, graph_mode="RETAINED_CACHE",
            feasible_pairs=baseline.native_cost_feasible_pairs)
        selection = replace(selection, candidate_digest=next(
            key for (_length, row), key in raw_keys.items() if row.compute_chunk == 64))
    minted = []
    def mint(length, row, route):
        assert length == 128 and row == selected and route == (501, 0, 2)
        chunk = 64 if fault == "mint_topology" else 128
        words = _native_candidate_v2(length, (chunk,) * 3, descriptor_version=3,
            routes=((5, 501, 2, 0, (), 0),))
        value, = _MODULE._decode_native_stage_domain((1, 1, len(words), *words), execution_len=length,
            logical_len=128, position=0, graph_mode="RETAINED_CACHE")
        minted.append(_MODULE.StageTuple.from_value(value))
        return minted[0]
    if fault:
        with pytest.raises(ValueError, match="actual.*domain|same native stage"):
            plan(calibration_selection=selection, calibration_mint=mint)
        assert bool(minted) is (fault == "mint_topology")
    else:
        actual = plan(calibration_selection=selection, calibration_mint=mint)
        assert actual.request == baseline.request and actual.request.mode == "EXACT"
        assert actual.request.chunk_size == 128 and actual.request.padding_rows == 0
        assert actual.candidates == baseline.candidates
        assert actual.cost_feasible_pairs == baseline.cost_feasible_pairs
        assert actual.native_cost_feasible_pairs == baseline.native_cost_feasible_pairs
        assert actual.selected.stage_tuple == minted[0]
        assert actual.optimality == actual.selection_scope == "P7_CALIBRATION_ONLY"
        assert actual.cost_certificate is None and actual.selected_cost_candidate_digest == selection.candidate_digest


def test_native_candidate_v3_rejects_conflicting_rope_residency():
    descriptor = _native_candidate_v2(
        64,
        (64, 64, 64),
        descriptor_version=3,
        routes=((4, 401, 2, (1 << 56) | (1 << 57), (), 0),),
    )
    with pytest.raises(ValueError, match="conflicting table residency"):
        _MODULE.plan_prefill(
            64,
            64,
            0,
            resolve_stage_domain=lambda _execution_len: [
                1, 1, len(descriptor), *descriptor,
            ],
        )


def test_native_candidate_v2_rejects_stale_route_manifest_fingerprint():
    domain = _native_domain_v2(64, (64, 64, 64))
    # Domain header + record length + candidate stage payload + physical header
    # lands on the first route selector.
    candidate = domain[3:]
    stage_words = 15 + 4 * sum(candidate[11:14]) + 3 * candidate[14]
    candidate[stage_words + 11 + 2] ^= 1
    domain = [1, 1, len(candidate), *candidate]
    with pytest.raises(ValueError, match="manifest fingerprint is stale"):
        _MODULE.plan_prefill(
            64, 64, 0, resolve_stage_domain=lambda _execution_len: domain,
        )


def test_native_candidate_v2_rejects_truncated_route_arguments():
    candidate = _native_domain_v2(64, (64, 64, 64))[3:]
    stage_words = 15 + 4 * sum(candidate[11:14]) + 3 * candidate[14]
    candidate[stage_words + 11 + 4] = len(candidate)
    domain = [1, 1, len(candidate), *candidate]
    with pytest.raises(ValueError, match="route arguments are truncated"):
        _MODULE.plan_prefill(
            64, 64, 0, resolve_stage_domain=lambda _execution_len: domain,
        )


@pytest.mark.parametrize(
    ("physical_field", "invalid_value"),
    ((3, 1), (4, 63), (5, 63)),
)
def test_native_candidate_v2_keeps_execution_and_kv_lengths_distinct(
    physical_field, invalid_value,
):
    candidate = _native_domain_v2(64, (64, 64, 64))[3:]
    stage_words = 15 + 4 * sum(candidate[11:14]) + 3 * candidate[14]
    candidate[stage_words + physical_field] = invalid_value
    domain = [1, 1, len(candidate), *candidate]
    with pytest.raises(ValueError, match="COMPLETE.*inconsistent"):
        _MODULE.plan_prefill(
            64, 64, 0, resolve_stage_domain=lambda _execution_len: domain,
        )


@pytest.mark.parametrize(
    ("physical_field", "invalid_value", "graph_mode"),
    (
        (1, 63, _MODULE.GRAPH_RETAINED_CACHE),
        (3, 1, _MODULE.GRAPH_RETAINED_CACHE),
        (6, 2, _MODULE.GRAPH_RETAINED_CACHE),
    ),
)
def test_native_candidate_v2_must_match_outer_plan_contract(
    physical_field, invalid_value, graph_mode,
):
    candidate = _native_domain_v2(64, (64, 64, 64))[3:]
    stage_words = 15 + 4 * sum(candidate[11:14]) + 3 * candidate[14]
    candidate[stage_words + physical_field] = invalid_value
    domain = [1, 1, len(candidate), *candidate]
    with pytest.raises(ValueError, match="COMPLETE.*inconsistent"):
        _MODULE.plan_prefill(
            64,
            64,
            0,
            graph_mode=graph_mode,
            resolve_stage_domain=lambda _execution_len: domain,
        )


def test_native_candidate_v2_kv_logical_length_covers_position():
    candidate = _native_domain_v2(
        64, (64, 64, 64), position=32,
    )[3:]
    stage_words = 15 + 4 * sum(candidate[11:14]) + 3 * candidate[14]
    candidate[stage_words + 4] = 64
    domain = [1, 1, len(candidate), *candidate]
    with pytest.raises(ValueError, match="COMPLETE.*inconsistent"):
        _MODULE.plan_prefill(
            64,
            64,
            0,
            position=32,
            resolve_stage_domain=lambda _execution_len: domain,
        )


def test_graph_key_tracks_physical_winner_not_equivalent_request():
    one = _MODULE.plan_a6(
        63,
        request={"padding_rows": 1},
        tuple_domain={64: [(64, 64, 64)]},
    )
    two = _MODULE.plan_a6(
        64,
        request={"padding_rows": 0},
        tuple_domain={64: [(64, 64, 64)]},
    )
    assert one.plan_digest != two.plan_digest
    assert one.graph_key_words() == two.graph_key_words()


def test_native_domain_v1_rejects_legacy_flat_or_trailing_wire():
    with pytest.raises(ValueError, match="invalid v1 header"):
        _MODULE.plan_prefill(
            32, 32, 0,
            resolve_stage_domain=lambda _execution_len: (32, 32, 32),
        )
    with pytest.raises(ValueError, match="trailing"):
        _MODULE.plan_prefill(
            32, 32, 0,
            resolve_stage_domain=lambda execution_len: (
                *_native_domain(execution_len, (32, 32, 32)), 99
            ),
        )


def test_shared_stage_ceiling_filters_without_model_private_domain_parser():
    result = _MODULE.plan_prefill(
        64, 64, 0,
        max_stage_chunk_size=64,
        resolve_stage_domain=lambda _execution_len: (
            (64, 64, 64), (128, 128, 128)
        ),
    )
    assert result.selected is not None
    assert result.selected.stage_tuple.compute_chunk == 64
    assert result.candidates[1].reject_reason == "stage_chunk_limit"


def test_prefill_propagates_graph_mode_and_does_not_duplicate_rejects():
    records = []

    def resolve(execution_len):
        if execution_len == 32:
            raise _MODULE.PlannerRejectError(
                "CAPABILITY", requested=execution_len,
                detail="shape rejected",
            )
        return ((64, 64, 64),)

    result = _MODULE.plan_prefill(
        32, 64, 8, resolve_stage_domain=resolve, logger=records.append,
        graph_mode=_MODULE.GRAPH_BOUNDED_ONESHOT,
    )
    assert result.graph_mode == _MODULE.GRAPH_BOUNDED_ONESHOT
    assert sum(item["kind"] == "domain_reject" for item in records) == 1
    assert records[-1]["graph_mode"] == _MODULE.GRAPH_BOUNDED_ONESHOT


def test_prefill_exact_request_is_not_silently_clamped():
    def resolve(execution_len):
        chunk = 128 if execution_len >= 113 else 112
        return ((chunk, chunk, chunk),)

    result = _MODULE.plan_prefill(
        100, 256, 16,
        resolve_stage_domain=resolve,
        exact_chunk_size=128,
    )
    assert result.selected is not None
    assert result.selected.execution_len == 113


def test_prefill_auto_accepts_explicit_zero_padding():
    result = _MODULE.plan_prefill(
        256, 512, 0,
        resolve_stage_domain=lambda _execution_len: ((128, 128, 128),),
        padding_rows=0,
    )
    assert result.selected is not None
    assert result.selected.execution_len == 256
    assert result.request.padding_rows == 0
    assert result.request.padding_budget == 0


@pytest.mark.parametrize("logical_len,padding_rows,chunk", [
    (4095, "auto", 512),
    (4095, 1, 256),
    (4096, "auto", 256),
    (4096, 0, 128),
])
def test_long_prefill_selects_admitted_boundary_with_room_for_decode(
    logical_len, padding_rows, chunk,
):
    seen = []

    def resolve(execution_len):
        seen.append(execution_len)
        # Supply one admitted native domain. This checks host handling of the
        # oracle's boundary; it does not emulate native SDPA admissibility.
        if execution_len == 4096:
            return _native_domain_v2(
                execution_len, (chunk, chunk, chunk), logical_len=logical_len,
            )
        raise _MODULE.PlannerRejectError(
            "CAPABILITY", requested=execution_len, limit=4096,
            detail="no admitted native domain for this physical length",
        )

    result = _MODULE.plan_prefill(
        logical_len, logical_len + 64, 64,
        padding_rows=padding_rows, resolve_stage_domain=resolve,
    )
    assert result.selected is not None
    assert result.selected.padding_rows == 4096 - logical_len
    assert result.selected.execution_len == 4096
    assert result.selected.stage_tuple.as_tuple() == (chunk, chunk, chunk)
    assert result.selected.stage_tuple.physical_descriptor
    if padding_rows == "auto":
        assert max(seen) > 4096  # Cache room does not expand native admission.
        assert result.rejected_count == len(seen) - 1
    else:
        assert seen == [4096]


def test_long_prefill_preserves_native_4097_rejection_despite_larger_cache():
    seen = []

    def reject(execution_len):
        seen.append(execution_len)
        raise _MODULE.PlannerRejectError(
            "CAPABILITY", requested=execution_len, limit=4096,
            detail="native prefill envelope exceeded",
        )

    with pytest.raises(_MODULE.PlannerRejectError) as error:
        _MODULE.plan_prefill(4097, 4161, 64, resolve_stage_domain=reject)
    assert error.value.code == "CAPABILITY"
    assert error.value.requested == 4097
    assert error.value.limit == 4096
    assert seen[0] == 4097 and seen[-1] == 4161


def test_prefill_propagates_non_native_callback_failures():
    def broken(_execution_len):
        raise KeyError("programming bug")

    with pytest.raises(KeyError, match="programming bug"):
        _MODULE.plan_prefill(32, 64, 0, resolve_stage_domain=broken)

    def runtime_bug(_execution_len):
        raise RuntimeError("device or programming bug")

    with pytest.raises(RuntimeError, match="device or programming bug"):
        _MODULE.plan_prefill(32, 64, 0, resolve_stage_domain=runtime_bug)


def test_native_chunk_error_translation_is_narrow_and_preserves_stage():
    no_candidate = RuntimeError(
        "RPU_PLANNER_REJECT:CAPABILITY:no feasible native stage tuple"
    )
    reject = _MODULE.native_chunk_reject(
        no_candidate, stage="VISION", requested=256
    )
    assert reject is not None
    assert reject.code == "CAPABILITY"
    assert reject.stage == "VISION"
    assert reject.requested == 256
    assert _MODULE.native_chunk_reject(
        RuntimeError("invalid handle"), stage="VISION", requested=256
    ) is None

    reject = _MODULE.native_chunk_reject(
        RuntimeError(
            "RPU_PLANNER_REJECT:EXACT_MISMATCH:requested chunk exceeds SPM"
        ),
        stage="VISION",
        requested=512,
    )
    assert reject is not None
    assert reject.code == "EXACT_MISMATCH"


def test_all_typed_domain_rejections_preserve_first_structured_error():
    def reject(execution_len):
        raise _MODULE.PlannerRejectError(
            "CAPABILITY", stage="VISION", requested=execution_len,
            detail="no native candidate",
        )

    with pytest.raises(_MODULE.PlannerRejectError) as info:
        _MODULE.plan_prefill(
            256, 512, 0,
            resolve_stage_domain=reject,
            padding_rows=0,
        )
    assert info.value.code == "CAPABILITY"
    assert info.value.stage == "VISION"
    assert info.value.requested == 256


def test_fixed_component_plan_is_one_real_domain_and_exact_is_fail_closed():
    plan = _MODULE.plan_fixed_component_execution(
        50,
        execution_len=50,
        chunk_size=64,
        component_id="action_expert",
        stage="action",
        generation=3,
        position=800,
        stage_config={"chunk_size": "auto"},
    )
    assert plan.candidate_count == 1
    assert plan.feasible_count == 1
    assert plan.selection_scope == "FIXED_ABI_SINGLETON_NO_SEARCH"
    assert plan.optimality == "CAPABILITY_SELECTED"
    assert plan.selected is not None
    assert plan.selected.stage_tuple.compute_chunk == 64
    assert len(plan.graph_key_words()) == 8

    with pytest.raises(_MODULE.PlannerRejectError) as info:
        _MODULE.plan_fixed_component_execution(
            50,
            execution_len=50,
            chunk_size=64,
            component_id="action_expert",
            stage="action",
            stage_config={"chunk_size": 80},
        )
    assert info.value.code == "EXACT_MISMATCH"


def test_fixed_component_physical_identity_covers_child_and_generation():
    def digest(component, generation):
        return _MODULE.plan_fixed_component_execution(
            256,
            execution_len=256,
            chunk_size=256,
            component_id=component,
            stage="vision",
            generation=generation,
        ).physical_plan_digest

    baseline = digest("vision_encoder", 0)
    assert baseline == digest("vision_encoder", 0)
    assert baseline != digest("vision_encoder", 1)
    assert baseline != digest("another_vision", 0)


def test_fixed_component_identity_retains_native_physical_descriptor():
    def plan(descriptor):
        return _MODULE.plan_fixed_component_execution(
            256,
            execution_len=256,
            chunk_size=256,
            component_id="vision_encoder",
            stage="vision",
            graph_mode=_MODULE.GRAPH_NATIVE_COMPOSITE1,
            physical_descriptor=descriptor,
            queue_owner_id=17,
            lease_owner_id=19,
        )

    descriptor = (3, 5, 256, 256, 256)
    selected = plan(descriptor)
    assert selected.selected is not None
    assert selected.selected.stage_tuple.physical_descriptor == descriptor
    assert selected.physical_plan_digest != plan((*descriptor, 1)).physical_plan_digest
    assert selected.graph_key_words() != plan((*descriptor, 1)).graph_key_words()


def test_fixed_component_identity_covers_physical_metadata_and_owner():
    def plan(image_batch_count, owner):
        return _MODULE.plan_fixed_component_execution(
            512,
            execution_len=512,
            chunk_size=512,
            component_id="vision_encoder",
            stage="vision",
            physical_metadata=(("image_batch_count", image_batch_count),),
            queue_owner_id=owner,
        )

    single = plan(1, 17)
    assert single.as_dict(include_candidates=False)["owner_identity_status"] == "RESOLVED"
    assert single.physical_plan_digest != plan(2, 17).physical_plan_digest
    assert single.physical_plan_digest != plan(1, 18).physical_plan_digest

    with pytest.raises(ValueError, match="unique"):
        _MODULE.plan_fixed_component_execution(
            64,
            execution_len=64,
            chunk_size=64,
            component_id="action_expert",
            stage="action",
            physical_metadata=(("position", 1),),
        )
