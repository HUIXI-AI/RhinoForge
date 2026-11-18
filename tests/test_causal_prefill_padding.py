"""Plain-text bounded-prefill planning contracts (host-only)."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
DECODER = ROOT / "python" / "rpu_backend" / "runtime" / "decoder.py"
SOURCE = DECODER.read_text()
QWEN3_SOURCE = (
    ROOT / "python" / "rpu_backend" / "adapters" / "qwen3.py"
).read_text()
QWEN3_MODEL_H = (
    ROOT / "src" / "fused" / "rpu_qwen3_model.h"
).read_text()

_SPEC = importlib.util.spec_from_file_location(
    "causal_prefill_execution_planner_under_test",
    ROOT / "python/rpu_backend/runtime/execution_planner.py",
)
_PLANNER = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _PLANNER
_SPEC.loader.exec_module(_PLANNER)


def _stage_domain(resolve):
    def domain(execution_len):
        chunk = resolve(execution_len)
        return ((chunk, chunk, chunk),)

    return domain


def _load_function(name):
    tree = ast.parse(SOURCE, filename=str(DECODER))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    namespace = {
        "torch": torch,
        "os": os,
        "json": json,
        "_LOG": SimpleNamespace(info=lambda *_args, **_kwargs: None),
        "PlannerRejectError": _PLANNER.PlannerRejectError,
        "native_chunk_reject": _PLANNER.native_chunk_reject,
        "plan_prefill": _PLANNER.plan_prefill,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]),
                 str(DECODER), "exec"), namespace)
    return namespace, namespace[name]


class _ConfigView:
    def __init__(self, **values):
        vars(self).update(values)


def test_causal_padding_env_binds_once_to_the_shared_cold_session(monkeypatch):
    from rpu_backend.runtime import decoder
    from rpu_backend.api._execution import reconfigure_rpu_execution
    _bind_causal_decoder_execution_session = decoder._bind_causal_decoder_execution_session

    original = {"prefill": {"chunk_size": "auto"}}
    inner = _ConfigView()
    model = SimpleNamespace(model=inner, _rpu_execution=original)
    adapter = _ConfigView(model=model)
    sibling = SimpleNamespace(_rpu_execution={"vision": {"chunk_size": 144}})
    model.sibling = sibling
    sibling_config = sibling._rpu_execution
    monkeypatch.setenv("RPU_CAUSAL_PREFILL_PADDING_BUDGET", "7")
    session = _bind_causal_decoder_execution_session(adapter, entry_point="cold-padding-test")
    expected = {"prefill": {"chunk_size": "auto", "padding_budget": 7}}
    assert session.config == expected
    assert original == {"prefill": {"chunk_size": "auto"}}
    assert model._rpu_execution is adapter._rpu_execution is inner._rpu_execution is session.config
    assert sibling._rpu_execution is sibling_config

    # Post-install binding and a second facade must preserve the first cold
    # translation, even if the process environment becomes invalid meanwhile.
    monkeypatch.setenv("RPU_CAUSAL_PREFILL_PADDING_BUDGET", "invalid-after-cold")
    second_adapter = _ConfigView(model=model)
    assert _bind_causal_decoder_execution_session(
        second_adapter, entry_point="cold-padding-test") is session
    assert session.config == expected
    assert second_adapter._rpu_execution is adapter._rpu_execution is session.config
    session.require_cold()
    assert session.generation == 0
    with session.execute():
        pass
    assert _bind_causal_decoder_execution_session(adapter, entry_point="cold-padding-test") is session
    assert model._rpu_execution is inner._rpu_execution is session.config
    assert session.config == expected and session.generation == 0
    with pytest.raises(RuntimeError, match="cold session"):
        session.require_cold()

    # The translated value is ordinary canonical configuration afterward;
    # public reconfigure updates every facade without consulting the env.
    model.config = SimpleNamespace()
    adapter.preflight_execution = lambda *_args: None
    monkeypatch.setattr(decoder, "_causal_lm_runtime_complete", lambda _: True)
    for name in ("execution_reconfigure_begin", "execution_reconfigure_commit",
                 "execution_reconfigure_abort", "execution_reconfigure_abort_attempt",
                 "causal_decoder_stage_chunk_size_override"):
        monkeypatch.setattr(torch.ops.rpu, name,
            lambda *_args: pytest.fail("same chunk must not stage native controls"), raising=False)
    events = []
    inner._rpu_decoder_graph_cache = SimpleNamespace(begin_warmup=lambda: events.append("warm"),
        clear=lambda: events.append("clear"), cache_invariant_ok=lambda: True)
    published = reconfigure_rpu_execution(model, {"prefill": {"chunk_size": "auto", "padding_budget": 3}})
    assert events == ["warm", "clear"] and session.generation == 1
    assert model._rpu_execution is inner._rpu_execution is adapter._rpu_execution is published
    assert second_adapter._rpu_execution is session.config is published
    assert sibling._rpu_execution is sibling_config
    assert _bind_causal_decoder_execution_session(second_adapter, entry_point="cold-padding-test") is session
    assert session.config["prefill"]["padding_budget"] == 3 and session.generation == 1


@pytest.mark.parametrize("prefill", [
    {"chunk_size": 128, "padding_budget": 3},
    {"chunk_size": "auto", "padding_rows": 0},
    {"chunk_size": "auto", "padding_rows": 11},
])
def test_causal_canonical_padding_precedes_legacy_env(monkeypatch, prefill):
    from rpu_backend.runtime.decoder import _bind_causal_decoder_execution_session

    monkeypatch.setenv("RPU_CAUSAL_PREFILL_PADDING_BUDGET", "invalid-ignored")
    original = {"prefill": dict(prefill)}
    model = SimpleNamespace(model=_ConfigView(), _rpu_execution=original)
    adapter = _ConfigView(model=model)
    session = _bind_causal_decoder_execution_session(adapter, entry_point="cold-padding-test")
    assert session.config == original
    assert model._rpu_execution is model.model._rpu_execution is adapter._rpu_execution is session.config
    session.require_cold()


@pytest.mark.parametrize("value,match", [("bad", "integer"), ("-1", "non-negative")])
def test_invalid_legacy_padding_fails_before_session_binding(monkeypatch, value, match):
    from rpu_backend.runtime.decoder import _bind_causal_decoder_execution_session

    monkeypatch.setenv("RPU_CAUSAL_PREFILL_PADDING_BUDGET", value)
    model = SimpleNamespace(model=_ConfigView(), _rpu_execution={"prefill": {"chunk_size": "auto"}})
    with pytest.raises(ValueError, match=match):
        _bind_causal_decoder_execution_session(_ConfigView(model=model), entry_point="cold-padding-test")
    assert not hasattr(model, "_execution_session")
    assert not hasattr(model.model, "_execution_session")


@pytest.mark.parametrize("arch", ["qwen3", "llama", "qwen3_vl_text"])
@pytest.mark.parametrize("bound", [False, True])
def test_installer_only_translates_unbound_plain_decoder_padding(monkeypatch, arch, bound):
    from rpu_backend.api import _execution
    from rpu_backend.runtime import decoder

    original = {"prefill": {"chunk_size": "auto"}}
    model = SimpleNamespace(_rpu_execution=original, config=SimpleNamespace(
        model_type="llama" if arch == "qwen3" else "qwen3"))
    if bound:
        session = _execution.bind_execution_session(model, original, entry_point="already-bound",
            supported=_execution._CAUSAL_DECODER_EXECUTION_SUPPORTED, graph_mode="RETAINED_CACHE")
        model._rpu_execution = session.config
    should_translate = not bound and arch != "qwen3_vl_text"
    monkeypatch.setenv("RPU_CAUSAL_PREFILL_PADDING_BUDGET", "7" if should_translate else "invalid-unused")
    translated = []
    translate = _execution._cold_causal_decoder_execution
    def observe(config):
        result = translate(config)
        translated.append(result)
        return result
    monkeypatch.setattr(_execution, "_cold_causal_decoder_execution", observe)
    # Execute the real install prefix, stopping at its next config-only guard.
    # This never reaches weights, native handles or device operations.
    with pytest.raises(ValueError, match="mismatches.*model_type"):
        decoder._install_causal_decoder_forward(model, arch=arch,
            mrope_section=[1] if arch == "qwen3_vl_text" else None,
            chunk_envelope_for=lambda *_: pytest.fail("past config-only guard"))
    assert translated == ([{"prefill": {"chunk_size": "auto", "padding_budget": 7}}]
                          if should_translate else [])
    assert original == {"prefill": {"chunk_size": "auto"}}
    if bound:
        assert model._rpu_execution is session.config
        session.require_cold()


def test_plain_text_starts_at_the_raw_length_and_only_pads_for_fewer_launches():
    _, plan = _load_function("plan_bounded_prefill_execution")

    # Equal launch count keeps the raw rows, even when a padded split has no
    # tail. Ordinary Qwen/Llama sequence lengths are not 16-aligned by contract.
    assert plan(
        1000, 8192, 64, resolve_stage_domain=_stage_domain(lambda _n: 512)
    ) == (1000, 512)
    assert plan(
        33, 8192, 0, resolve_stage_domain=_stage_domain(lambda _n: 32)
    ) == (33, 32)

    # Padding is worthwhile when it removes a launch: raw 1000 needs two 512
    # chunks, while 1024 has a legal one-chunk plan.
    fewer_launches = lambda n: 1024 if n >= 1024 else 512  # noqa: E731
    domain = _stage_domain(fewer_launches)
    assert plan(1000, 8192, 64, resolve_stage_domain=domain) == (1024, 1024)
    assert plan(1000, 8192, 23, resolve_stage_domain=domain) == (1000, 512)


def test_prefill_facade_requires_full_domain_and_only_translates_native_rejects():
    _, plan = _load_function("plan_bounded_prefill_execution")

    with pytest.raises(TypeError, match="resolve_stage_domain"):
        plan(32, 64, 0)

    typed = _PLANNER.PlannerRejectError("CAPABILITY", requested=32)
    with pytest.raises(_PLANNER.PlannerRejectError) as info:
        plan(
            32, 64, 0,
            resolve_stage_domain=lambda _length: (_ for _ in ()).throw(typed),
        )
    assert info.value is typed

    with pytest.raises(_PLANNER.PlannerRejectError, match="EXACT_MISMATCH"):
        plan(
            32, 64, 0,
            resolve_stage_domain=lambda _length: (_ for _ in ()).throw(
                RuntimeError("RPU_PLANNER_REJECT:EXACT_MISMATCH:no exact tuple")
            ),
        )

    with pytest.raises(RuntimeError, match="device failure") as info:
        plan(
            32, 64, 0,
            resolve_stage_domain=lambda _length: (_ for _ in ()).throw(
                RuntimeError("device failure")
            ),
        )
    assert type(info.value) is RuntimeError


def test_alignment_and_exact_padding_are_adapter_owned():
    _, plan = _load_function("plan_bounded_prefill_execution")
    resolve = lambda _n: 160

    domain = _stage_domain(resolve)
    assert plan(313, 8192, 64, resolve_stage_domain=domain, alignment=1,
                padding_rows=0) == (313, 160)
    assert plan(313, 8192, 64, resolve_stage_domain=domain, alignment=16,
                padding_rows=7) == (320, 160)

    with pytest.raises(_PLANNER.PlannerRejectError, match="ALIGNMENT"):
        plan(313, 8192, 64, resolve_stage_domain=domain,
             alignment=16, padding_rows=0)
    with pytest.raises(ValueError, match="non-negative"):
        plan(313, 8192, 64, resolve_stage_domain=domain, padding_rows=-1)
    with pytest.raises(ValueError, match=">= 1"):
        plan(313, 8192, 64, resolve_stage_domain=domain, alignment=0)


def test_exact_chunk_searches_padding_instead_of_accepting_native_clamp():
    _, plan = _load_function("plan_bounded_prefill_execution")

    def clamping_resolver(execution_len):
        return min(128, ((execution_len + 15) // 16) * 16)

    # The native resolver clamps 128 to 112 through execution_len=112.  The
    # shared planner must keep searching until 128 is the value really used.
    assert plan(
        100,
        256,
        16,
        resolve_stage_domain=_stage_domain(clamping_resolver),
        exact_chunk_size=128,
    ) == (113, 128)

    with pytest.raises(_PLANNER.PlannerRejectError, match="EXACT_MISMATCH"):
        plan(
            100,
            256,
            12,
            resolve_stage_domain=_stage_domain(clamping_resolver),
            exact_chunk_size=128,
        )
    with pytest.raises(_PLANNER.PlannerRejectError, match="EXACT_MISMATCH"):
        plan(
            100,
            256,
            64,
            resolve_stage_domain=_stage_domain(clamping_resolver),
            padding_rows=0,
            exact_chunk_size=128,
        )
    with pytest.raises(ValueError, match="positive multiple of 16"):
        plan(
            100, 256, 64,
            resolve_stage_domain=_stage_domain(clamping_resolver),
            exact_chunk_size=127,
        )


def test_padding_search_is_bounded_by_budget_and_physical_rows():
    _, plan = _load_function("plan_bounded_prefill_execution")
    fewer_launches = lambda n: 1024 if n >= 1024 else 512  # noqa: E731

    domain = _stage_domain(fewer_launches)
    assert plan(1000, 8192, 24, resolve_stage_domain=domain) == (1024, 1024)
    assert plan(1000, 8192, 23, resolve_stage_domain=domain) == (1000, 512)
    assert plan(1000, 1023, 64, resolve_stage_domain=domain) == (1000, 512)
    with pytest.raises(_PLANNER.PlannerRejectError, match="NO_FEASIBLE"):
        plan(1000, 999, 64, resolve_stage_domain=domain)


def test_plain_text_native_guard_rejects_only_causal_sub16_physical_tails(tmp_path):
    # Compile the actual early guards, not a Python copy of the predicate.
    # Passing here only reaches the remaining envelope/SPM/SDPA checks; it is
    # neither full native admission nor numerical/hardware certification.
    function = QWEN3_MODEL_H[QWEN3_MODEL_H.index("bool subclass_chunk_size_valid"):]
    guards = function[function.index("{") + 1:function.index("// MR-A:")]
    source = r"""
#include <cassert>
#include <cstdint>
#include <optional>
#include "rpu_qwen3_w8a8_contract.h"
namespace qwen3_w8a8 = v3::qwen3_w8a8;
// This plain-text probe keeps the independent W8A8 cold capabilities off.
constexpr bool qwen3_prefill_w8a8_ = false;
constexpr bool qwen3_c192_ = false, qwen3_c256_ = false, qwen3_c512_ = false;
struct Context { bool is_causal; std::optional<int> attention_mask; } context;
const Context& ctx() { return context; }
bool reaches_remaining_checks(int64_t cs, int64_t seq_len, int64_t position = 0) {
""" + guards + r"""
    return true;
}
int main() {
    for (bool causal : {false, true}) {
        context = {causal, std::nullopt};
        assert(reaches_remaining_checks(16, 65) == !causal);
        assert(reaches_remaining_checks(16, 178) == !causal);
        assert(reaches_remaining_checks(512, 513) == !causal);
        assert(reaches_remaining_checks(512, 2049) == !causal);
        assert(reaches_remaining_checks(32, 80));  // a full v16 tail
        assert(reaches_remaining_checks(80, 75)); // single physical chunk
        assert(reaches_remaining_checks(16, 1));  // one-token decode
        context.attention_mask = 1;
        assert(!reaches_remaining_checks(16, 65));
        assert(!reaches_remaining_checks(16, 178));
        assert(!reaches_remaining_checks(32, 80));
        assert(reaches_remaining_checks(80, 75));
    }
}
"""
    executable = tmp_path / "causal_tail_guard"
    subprocess.run(["g++", "-std=c++17", "-I", str(ROOT / "src/fused"),
                    "-x", "c++", "-", "-o", str(executable)],
                   input=source, text=True, capture_output=True, check=True)
    subprocess.run([str(executable)], capture_output=True, check=True)


def test_short_prefill_boundary_keeps_one_physical_chunk():
    _, plan = _load_function("plan_bounded_prefill_execution")

    cases = ((2, 16), (15, 16), (16, 16), (17, 32))
    for logical_len, expected_chunk in cases:
        execution_len, chunk_size = plan(
            logical_len,
            logical_len + 64,
            64,
            resolve_stage_domain=_stage_domain(
                lambda n: ((n + 15) // 16) * 16
            ),
        )
        assert execution_len == logical_len
        assert chunk_size == expected_chunk


def test_offgrid_policy_keeps_an_admitted_raw_plan_or_pads_exact_512():
    _, plan = _load_function("plan_bounded_prefill_execution")

    def auto_resolver(execution_len):
        # Model a native resolver that can find a safe raw candidate. The real
        # winner is geometry-specific (0.6B hardware picks E520/cs480 at P513),
        # so this contract checks the outer policy: keep raw when admitted.
        valid = [
            chunk_size
            for chunk_size in range(512, 15, -16)
            if execution_len <= chunk_size
            or execution_len % chunk_size == 0
            or execution_len % chunk_size >= 16
        ]
        return min(
            valid,
            key=lambda chunk_size: (
                (execution_len + chunk_size - 1) // chunk_size,
                ((execution_len + chunk_size - 1) // chunk_size)
                * chunk_size - execution_len,
            ),
        )

    for logical_len in (513, 2049):
        execution_len, chunk_size = plan(
            logical_len, logical_len + 64, 64,
            resolve_stage_domain=_stage_domain(auto_resolver),
        )
        assert execution_len == logical_len
        assert chunk_size % 16 == 0
        assert execution_len % chunk_size >= 16

    def exact_512_domain(execution_len):
        tail = execution_len % 512
        if tail and tail < 16:
            return ()
        return ((512, 512, 512),)

    assert plan(
        513, 577, 64,
        resolve_stage_domain=exact_512_domain,
        exact_chunk_size=512,
    ) == (528, 512)
    assert plan(
        2049, 2113, 64,
        resolve_stage_domain=exact_512_domain,
        exact_chunk_size=512,
    ) == (2064, 512)


def test_resolved_chunk_is_always_a_positive_multiple_of_16():
    _, plan = _load_function("plan_bounded_prefill_execution")

    for chunk_size, match in ((0, "integer >= 1"), (17, "multiples of 16")):
        with pytest.raises(ValueError, match=match):
            plan(
                32, 64, 0,
                resolve_stage_domain=_stage_domain(
                    lambda _n, cs=chunk_size: cs
                ),
            )


def test_prefill_planner_can_emit_bounded_candidate_records():
    _, plan = _load_function("plan_bounded_prefill_execution")
    records = []
    assert plan(
        100, 128, 16,
        resolve_stage_domain=_stage_domain(lambda _n: 64),
        candidate_logger=records.append, request_id="host-test",
    ) == (100, 64)
    assert records[-1]["request_id"] == "host-test"
    assert records[-1]["kind"] == "summary"
    assert records[-1]["status"] == "ACCEPTED"
    assert records[-1]["selected"]["execution_len"] == 100
    assert records[-1]["resolved_chunk_size"] == 64
    candidates = [r for r in records if r["kind"] == "candidate"]
    assert len(candidates) == 17
    assert all(r["request_id"] == "host-test" for r in candidates)


def test_fixed_comparator_has_no_execution_alignment_penalty():
    _, plan = _load_function("plan_bounded_prefill_execution")
    assert plan(
        15,
        16,
        1,
        resolve_stage_domain=_stage_domain(lambda _n: 16),
    ) == (15, 16)
    assert "preferred_execution_multiple" not in SOURCE
    assert "alignment_penalty" not in SOURCE


def test_text_helper_uses_public_stage_config_and_real_cache_position(monkeypatch):
    namespace, helper = _load_function("_text_prefill_execution_plan")
    facade_namespace, facade = _load_function("plan_bounded_prefill_execution")
    namespace["plan_bounded_prefill_execution"] = facade
    calls = []

    class Ops:
        class rpu:
            causal_decoder_planner_cache_identity = staticmethod(lambda handle: (1, 1))

            @staticmethod
            def causal_decoder_resolve_prefill_stage_domain(
                handle, length, position, _is_causal, _mask_kv_len,
                rope_mode, lifecycle, logical_len,
            ):
                assert (rope_mode, lifecycle, logical_len) == (0, 1, 33)
                calls.append((handle, length, position))
                return ((48, 48, 48),)

    namespace["torch"] = facade_namespace["torch"] = type("T", (), {"ops": Ops})

    class Cache:
        max_seq_len = 128
        position = 16

    class Model:
        _rpu_execution = {"prefill": {"padding_rows": "auto",
                                       "padding_budget": 0}}

    assert helper(Model(), 7, Cache(), 33)[:2] == (33, 48)
    assert calls == [(7, 33, 16)]

    Model._rpu_execution = {"prefill": {"padding_rows": 15}}
    monkeypatch.setenv("RPU_CAUSAL_PREFILL_PADDING_BUDGET", "not-an-integer")
    calls.clear()
    assert helper(Model(), 7, Cache(), 33)[:2] == (48, 48)
    assert calls == [(7, 48, 16)]

    class ExactModel:
        _rpu_execution = {
            "prefill": {"chunk_size": 48, "padding_budget": 15}
        }

    calls.clear()
    assert helper(ExactModel(), 7, Cache(), 33)[:2] == (33, 48)
    assert calls == [(7, n, 16) for n in range(33, 49)]


def test_plain_hf_controls_are_validated_then_removed_for_planning():
    _, canonicalize = _load_function("_canonicalize_plain_text_controls")

    class Cache:
        position = 16

    mask = torch.ones((1, 35), dtype=torch.long)
    positions = torch.arange(16, 35).unsqueeze(0)
    cache_positions = torch.arange(16, 35)
    assert canonicalize(
        19, Cache(), mask, positions, cache_positions
    ) == (None, None, None)

    bad_mask = mask.clone()
    bad_mask[0, 0] = 0
    with pytest.raises(NotImplementedError, match="all-ones"):
        canonicalize(19, Cache(), bad_mask, positions, cache_positions)
    with pytest.raises(NotImplementedError, match="attention_mask shape"):
        canonicalize(19, Cache(), torch.ones((1, 34)), positions,
                     cache_positions)
    with pytest.raises(NotImplementedError, match="contiguous cache range"):
        canonicalize(19, Cache(), mask, positions + 1, cache_positions)
    with pytest.raises(NotImplementedError, match="contiguous cache range"):
        canonicalize(19, Cache(), mask, positions, cache_positions[:-1])


def test_all_explicit_padding_modes_fail_when_controls_prevent_planning():
    _, reject = _load_function("_reject_unconsumed_text_padding")

    class Model:
        _rpu_execution = {"prefill": {"padding_rows": 7}}

    reject(Model(), 33, None, None)
    reject(Model(), 1, object(), None)
    with pytest.raises(ValueError, match="padding.*cannot be applied"):
        reject(Model(), 33, object(), None)
    with pytest.raises(ValueError, match="padding.*cannot be applied"):
        reject(Model(), 33, None, object())

    Model._rpu_execution = {"prefill": {"padding_rows": "auto"}}
    with pytest.raises(ValueError, match="padding.*cannot be applied"):
        reject(Model(), 33, object(), object())
    Model._rpu_execution = {"prefill": {"padding_budget": 7}}
    with pytest.raises(ValueError, match="padding.*cannot be applied"):
        reject(Model(), 33, object(), object())
    Model._rpu_execution = {"prefill": {"padding_rows": 0}}
    reject(Model(), 33, object(), object())


def test_padding_is_confined_to_plain_unmasked_multitoken_forwards():
    tree = ast.parse(SOURCE, filename=str(DECODER))
    forward = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_causal_decoder_forward"
    )
    guard = next(
        node for node in ast.walk(forward)
        if isinstance(node, ast.If)
        and any(
            "_text_prefill_execution_plan" in ast.unparse(statement)
            for statement in node.body
        )
    )
    condition = ast.unparse(guard.test)
    for required in ("seq_len > 1", "position_ids is None",
                     "attention_mask is None"):
        assert required in condition
    assert "position == 0" not in condition


@pytest.mark.parametrize("position", [0, 17])
@pytest.mark.parametrize("native_rejects", [False, True])
def test_selected_prefill_publishes_only_after_native_descriptor_admission(
        monkeypatch, position, native_rejects):
    from rpu_backend.api.cache import RPUCache
    from rpu_backend.runtime import decoder
    from rpu_backend.runtime.execution_planner import plan_prefill, _decode_native_stage_domain
    from test_execution_planner import _native_candidate_v2

    wire = tuple(_native_candidate_v2(32, (16,) * 3, position=position,
        descriptor_version=3, routes=((1, 101, 2, 0, (), 0),)))
    plan = plan_prefill(32, 32, 0, position=position,
        resolve_stage_domain=lambda length: (1, 1, len(wire), *wire))
    previous = {"sentinel": "previous completed call"}
    owner = SimpleNamespace(_rpu_last_execution_plan=previous)
    cache = object.__new__(RPUCache)
    cache.position, cache.max_seq_len, cache.batch_size = position, 128, 1
    cache.k_caches, cache.v_caches = [torch.zeros(1)], [torch.zeros(1)]
    seen = []
    def native(handle, hidden, keys, values, mask, actual_position, *rest):
        assert tuple(rest[-1]) == wire and actual_position == position
        _decode_native_stage_domain((1, 1, len(wire), *wire),
            execution_len=32, logical_len=32, position=actual_position,
            graph_mode="RETAINED_CACHE")
        seen.append(rest[-1])
        if native_rejects:
            raise RuntimeError("RPU_PLANNER_REJECT:EXACT_MISMATCH: live compute/qkv mismatch")
        return hidden.clone()
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_forward", native, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_get_resolved_chunk_size",
        lambda _: pytest.fail("selected prefill already consumed its exact descriptor"), raising=False)
    def run():
        return decoder._run_causal_decoder_forward(owner, 7,
            inputs_embeds=torch.ones(1, 32, 4, dtype=torch.float16),
            past_key_values=cache, prefill_plan=(32, 16, plan))
    if native_rejects:
        with pytest.raises(RuntimeError, match="EXACT_MISMATCH"):
            run()
        assert owner._rpu_last_execution_plan is previous and cache.position == position
    else:
        output, actual_cache = run()
        assert actual_cache is cache and cache.position == position + 32
        assert owner._rpu_last_execution_plan["chunk_size"] == 16
        assert owner._rpu_last_execution_plan["physical_descriptor"] == wire
        assert owner._rpu_last_execution_plan["position"] == position
        assert torch.equal(output, torch.ones_like(output))
    assert len(seen) == 1


def test_plan_is_resolved_before_capture_and_is_part_of_the_signature():
    start = SOURCE.index("def rpu_decoder_model_forward")
    closure = SOURCE[start:]
    resolve = closure.index("prefill_plan = _text_prefill_execution_plan")
    capture = closure.index("def _capture(")
    assert resolve < capture
    capture_source = closure[capture:]
    assert "int(seq_len)" in capture_source
    assert "execution_len" in capture_source
    assert "self._rpu_decoder_hidden_size" in capture_source
    assert "planned_chunk_size" in capture_source


def test_qwen3vl_mrope_positions_still_move_to_rpu():
    tree = ast.parse(SOURCE, filename=str(DECODER))
    forward = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "rpu_decoder_model_forward"
    )
    guards = [
        ast.unparse(node.test)
        for node in ast.walk(forward)
        if isinstance(node, ast.If)
        and "position_ids = position_ids.to('rpu')" in ast.unparse(node)
    ]
    assert any(
        "not batched_prefill" in guard and "position_ids is not None" in guard
        for guard in guards
    )


def test_unconsumed_padding_is_rejected_before_plain_decoder_capture():
    start = SOURCE.index("def rpu_decoder_model_forward")
    closure = SOURCE[start:]
    reject = closure.index("_reject_unconsumed_text_padding(")
    capture = closure.index("def _capture(")
    assert reject < capture


def test_fused_lm_head_uses_the_same_precomputed_execution_plan():
    start = QWEN3_SOURCE.index("def fused_forward")
    closure = QWEN3_SOURCE[start:]
    resolve = closure.index("prefill_plan = _text_prefill_execution_plan")
    capture = closure.index("def _capture(")
    assert resolve < capture
    capture_source = closure[capture:]
    assert "int(seq_len)" in capture_source
    assert "execution_len" in capture_source
    assert "planned_chunk_size" in capture_source
    assert "prefill_plan=prefill_plan" in closure


def test_unconsumed_padding_is_rejected_before_fused_lm_head_capture():
    start = QWEN3_SOURCE.index("def fused_forward")
    closure = QWEN3_SOURCE[start:]
    reject = closure.index("_reject_unconsumed_text_padding(")
    capture = closure.index("def _capture(")
    assert reject < capture


def test_padding_is_sliced_and_reported_without_advancing_logical_position():
    assert "raw = raw[:, :logical_seq_len]" in SOURCE
    assert "past_key_values.update_position(logical_seq_len)" in SOURCE
    for field in ("logical_len", "execution_len", "chunk_size",
                  "padding_rows", "position"):
        assert f'"{field}"' in SOURCE


def test_plain_text_budget_zero_preserves_arbitrary_lengths():
    plan = _load_function("plan_bounded_prefill_execution")[1]

    def safe_resolver(length):
        for chunk_size in range(((length + 15) // 16) * 16, 15, -16):
            tail = length % chunk_size
            if tail == 0 or tail >= 16:
                return chunk_size
        raise RuntimeError("no safe chunk")

    for length in list(range(17, 65)) + [513, 999, 1000, 2049, 2063]:
        execution_len, chunk_size = plan(
            length,
            8192,
            0,
            resolve_stage_domain=_stage_domain(safe_resolver),
        )
        tail_rows = execution_len - (
            (execution_len + chunk_size - 1) // chunk_size - 1
        ) * chunk_size
        assert execution_len == length
        assert tail_rows >= 16
