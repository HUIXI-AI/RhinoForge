"""Qwen3-VL bounded-prefill planning and padding contracts."""
from __future__ import annotations

import ast
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[1]
_ADAPTER = (
    _ROOT / "python" / "rpu_backend" / "adapters" / "qwen3_vl" / "__init__.py"
)
_MODEL_H = _ROOT / "src" / "fused" / "rpu_qwen3_model.h"
_MODEL_CPP = _ROOT / "src" / "fused" / "rpu_qwen3_model.cpp"
_DECLS = _ROOT / "src" / "core" / "rpu_kernel_decls.h"
# The op registration moved out of rpu_backend.cpp when 40a24100 split the
# backend TU into four .inc files (kernel cache / tensor bridge / dispatcher
# registration / pybind). Point at the dispatcher .inc, which is where
# TORCH_LIBRARY now lives — this assertion had silently rotted because the
# file was in no CI job.
_BACKEND = _ROOT / "src" / "core" / "rpu_dispatch_registrations.inc"


# The bounded-prefill planner now lives in runtime/, shared with the plain text
# decoder (which pads for the same reason: a better chunk plan). M-RoPE token
# padding is also shared with composite policies; visual scatter stays here.
_DECODER = _ROOT / "python" / "rpu_backend" / "runtime" / "decoder.py"

_PLANNER_SPEC = importlib.util.spec_from_file_location(
    "qwen3vl_execution_planner_under_test",
    _ROOT / "python/rpu_backend/runtime/execution_planner.py",
)
_PLANNER = importlib.util.module_from_spec(_PLANNER_SPEC)
assert _PLANNER_SPEC.loader is not None
sys.modules[_PLANNER_SPEC.name] = _PLANNER
_PLANNER_SPEC.loader.exec_module(_PLANNER)


def _stage_domain(resolve):
    def domain(execution_len):
        chunk = resolve(execution_len)
        return ((chunk, chunk, chunk),)

    return domain


def _load_function(name, source=None):
    src = source or _ADAPTER
    tree = ast.parse(src.read_text(), filename=str(src))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    namespace = {
        "torch": torch,
        "os": os,
        "json": json,
        "_LOG": SimpleNamespace(info=lambda *_args, **_kwargs: None),
        "RPUCache": object,
        "PlannerRejectError": _PLANNER.PlannerRejectError,
        "native_chunk_reject": _PLANNER.native_chunk_reject,
        "plan_prefill": _PLANNER.plan_prefill,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]),
                 str(src), "exec"), namespace)
    return namespace[name]


def test_qwen3_vl_preflight_admits_only_the_exact_padded_8b_profile(monkeypatch):
    from rpu_backend.adapters import qwen3_vl
    emitted = []
    monkeypatch.setattr(qwen3_vl, "_LOG", SimpleNamespace(
        warning=lambda *args, **kwargs: emitted.append((args, kwargs))))
    check = qwen3_vl._check_profile

    config = SimpleNamespace(
        model_type="qwen3_vl",
        tie_word_embeddings=False,
        text_config=SimpleNamespace(
            model_type="qwen3_vl_text",
            hidden_size=4096,
            intermediate_size=12288,
            num_hidden_layers=36,
            num_key_value_heads=8,
            num_attention_heads=32,
            head_dim=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            vocab_size=151936,
            attention_bias=False,
            use_cache=True,
            rope_scaling={
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
                "rope_type": "default",
                "rope_theta": 5_000_000,
            },
        ),
        vision_config=SimpleNamespace(
            model_type="qwen3_vl",
            in_channels=3,
            num_position_embeddings=2304,
            hidden_size=1152,
            intermediate_size=4304,
            depth=27,
            num_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            deepstack_visual_indexes=(8, 16, 24),
            hidden_act="gelu_pytorch_tanh",
            out_hidden_size=4096,
        ),
    )
    check(config)
    assert emitted == []

    config.vision_config.depth = 26
    with pytest.raises(qwen3_vl.UnsupportedModelError, match="unsupported vision profile"):
        check(config)


def test_qwen3_vl_313_prefill_uses_exact_bounded_plan():
    plan = _load_function("plan_bounded_prefill_execution", _DECODER)
    calls = []

    def resolve(execution_len):
        calls.append(execution_len)
        if execution_len == 368:
            return ()
        chunk = {
            320: 160,
            336: 112,
            352: 176,
            384: 192,
        }[execution_len]
        return ((chunk, chunk, chunk),)

    assert plan(
        313, 384, 64, resolve_stage_domain=resolve, alignment=16
    ) == (320, 160)
    assert calls == [320, 336, 352, 368, 384]

    # Reject a final chunk containing only padding.
    assert plan(
        17, 64, 32,
        resolve_stage_domain=_stage_domain(lambda _: 16),
        alignment=16,
    ) == (32, 16)

    with pytest.raises(_PLANNER.PlannerRejectError, match="NO_FEASIBLE"):
        plan(
            313, 319, 64,
            resolve_stage_domain=_stage_domain(lambda _: 160),
            alignment=16,
        )


def test_qwen3_vl_336_is_a_real_bounded_search_candidate():
    """A legal padded candidate can reduce the number of launches."""
    plan = _load_function("plan_bounded_prefill_execution", _DECODER)
    calls = []

    def resolve(length):
        calls.append(length)
        chunk = {320: 112, 336: 176, 352: 176}[length]
        return ((chunk, chunk, chunk),)

    # 320 needs three launches, while 336 needs two, so the joint planner may
    # spend 23 padding rows. This proves 336 is searched and can win; it does
    # not claim the model-specific hardware envelope has certified it.
    assert plan(
        313, 352, 32, resolve_stage_domain=resolve, alignment=16
    ) == (336, 176)
    assert calls == [320, 336, 352]


def test_qwen3_vl_long_prefill_envelope_is_profile_specific():
    source = (
        _ROOT / "python" / "rpu_backend" / "adapters" / "qwen3_vl" / "text.py"
    ).read_text()
    tree = ast.parse(source)
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_CHUNK_ENVELOPE"
                for target in node.targets)
    )
    rows = {}
    for key, value in zip(assignment.value.keys, assignment.value.values):
        literal_key = ast.literal_eval(key)
        if literal_key[:1] == ("qwen3_vl_text",) and literal_key[1] in (28, 36):
            rows[literal_key] = tuple(ast.literal_eval(arg) for arg in value.args)
    assert rows[("qwen3_vl_text", 28, 2048)] == (4176, 320)
    assert rows[("qwen3_vl_text", 36, 2560)] == (4176, 256)
    assert rows[("qwen3_vl_text", 36, 4096)] == (4176, 128)


def test_prefill_planner_refuses_to_plan_past_the_8192_physical_limit():
    """Reject requests beyond the actual M-RoPE keepalive capacity.

    The forward passes physical_limit = min(max_seq_len-position,
    sKeyVx*sKeyChunk, sValVx*sValChunk, _QWEN3_VL_MROPE_KEEPALIVE_ROWS);
    the M-RoPE keepalive is 8192 rows
    (adapters/qwen3_vl/__init__.py), sized to match RPUCache's 8192-token
    capacity (src/fused/rpu_qwen3_model.h QWEN3_MROPE_MAX_KEEPALIVE_SEQ).

    So 8192 is the LAST plannable execution length. One token past it the
    ceil16 base leaves the candidate window empty and the planner must RAISE —
    silently truncating, or wrapping into row 0 of the keepalive table, would
    corrupt positions instead of failing.
    """
    plan = _load_function("plan_bounded_prefill_execution", _DECODER)
    limit = 8192
    resolve = _stage_domain(lambda _n: 512)

    # 8191 and 8192 both ceil16 to exactly the limit -> still plannable.
    assert plan(
        8191, limit, 64, resolve_stage_domain=resolve, alignment=16
    )[0] == limit
    assert plan(
        8192, limit, 64, resolve_stage_domain=resolve, alignment=16
    )[0] == limit

    # 8193 ceil16s to 8208 > limit; the padding budget must NOT rescue it.
    for real_len in (8193, 8208, 9000):
        with pytest.raises(_PLANNER.PlannerRejectError, match="NO_FEASIBLE"):
            plan(
                real_len, limit, 64,
                resolve_stage_domain=resolve,
                alignment=16,
            )


def test_keepalive_bound_matches_the_cpp_cache_capacity():
    """The Python keepalive row count and the C++ one must not drift apart.

    They are two independent literals describing one buffer; if they diverge the
    planner would admit a length the fused decoder cannot address.
    """
    assert "_QWEN3_VL_MROPE_KEEPALIVE_ROWS = 8192" in _ADAPTER.read_text()
    assert "QWEN3_MROPE_MAX_KEEPALIVE_SEQ = 8192" in _MODEL_H.read_text()


def test_qwen3_vl_prefill_padding_covers_both_position_layouts_and_restore():
    pad_inputs = _load_function("pad_mrope_prefill_inputs", _DECODER)
    from rpu_backend.adapters.qwen3_vl.text import scatter_visual_embeds_to_dense
    restore = _load_function("_restore_logical_prefill")

    hidden = torch.ones(1, 313, 4)
    mask = torch.ones(1, 313, dtype=torch.long)
    positions_3d = torch.arange(313).view(1, 1, 313).expand(3, 1, 313)
    hidden, mask, positions_3d = pad_inputs(
        hidden, mask, positions_3d, 320,
    )
    assert hidden.shape == (1, 320, 4)
    assert mask.shape == (1, 320)
    assert positions_3d.shape == (3, 1, 320)
    assert not hidden[:, 313:].any()
    assert not mask[:, 313:].any()
    assert torch.equal(
        positions_3d[..., 313:],
        positions_3d[..., 312:313].expand(3, 1, 7),
    )

    positions_2d = torch.arange(313).view(313, 1).expand(313, 3)
    _, _, positions_2d = pad_inputs(
        torch.ones(1, 313, 4), None, positions_2d, 320,
    )
    assert positions_2d.shape == (320, 3)
    assert torch.equal(
        positions_2d[313:],
        positions_2d[312:313].expand(7, 3),
    )

    deepstack = scatter_visual_embeds_to_dense(
        [torch.ones(313, 4) for _ in range(3)],
        torch.ones(313, dtype=torch.bool), 313, 4,
        device="cpu", execution_len=320,
    )
    assert all(t[:313].all() for t in deepstack)
    assert all(t.shape == (320, 4) for t in deepstack)
    assert all(not t[313:].any() for t in deepstack)

    class Cache:
        position = 320

        def reset_to_position(self, position):
            self.position = position

    cache = Cache()
    raw = torch.arange(320 * 4).view(1, 320, 4)
    out = restore(raw, cache, 0, 313, 320)
    assert torch.equal(out, raw[:, :313])
    assert cache.position == 313


def test_qwen3_vl_continuation_preserves_explicit_native_plans():
    from rpu_backend.runtime.causal_append import continuation_segments

    # Explicit chunk/padding and auto padding stay with the native planner.
    for config in ({"padding_rows": "auto"}, {"padding_rows": 7},
                   {"padding_budget": 16}, {"chunk_size": 160}):
        assert continuation_segments(17, 32, config, capacity=384) is None
    # Only unpadded auto continuation decomposes its unaligned head/tail.
    spans = continuation_segments(17, 32, {"padding_rows": 0}, capacity=384)
    assert spans == (*((offset, 1) for offset in range(15)), (15, 16), (31, 1))
    with pytest.raises(ValueError, match="logical KV-cache horizon"):
        continuation_segments(380, 5, {}, capacity=384)


def test_qwen3_vl_prefill_planner_wiring_contract():
    source = _ADAPTER.read_text()
    tree = ast.parse(source, filename=str(_ADAPTER))
    forward = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_rpu_qwen3vl_forward"
    )
    zero_call = next(
        node for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "make_zero_visual_embeds"
    )
    assert ast.unparse(zero_call.args[1]) == "execution_seq_len"

    forward_source = ast.get_source_segment(source, forward)
    assert (
        forward_source.index("continuation_segments(")
        < forward_source.index("_qwen3_vl_text_graph_signature(")
    )
    def function_source(name):
        node = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
        return ast.get_source_segment(source, node)

    assert "GraphSignature(" in function_source("_qwen3_vl_text_graph_signature")
    plan_source = function_source("_qwen3_vl_prefill_plan")
    assert "cache.sKeyVx" in plan_source
    assert "cache.sValVx" in plan_source
    assert "cache.max_seq_len" in plan_source
    assert "_QWEN3_VL_MROPE_KEEPALIVE_ROWS" in plan_source
    assert "causal_decoder_resolve_prefill_stage_domain" in plan_source
    assert "causal_decoder_resolve_prefill_chunk_size" not in plan_source
    assert forward_source.index("_qwen3_vl_prefill_plan(") < forward_source.index("run_continuation_segments(")
    assert "native_execution_receipt" in forward_source
    assert 'native_execution_receipt.get("chunk_size") != resolved_chunk_size' in forward_source
    assert "resolved_chunk_size != planned_chunk_size" in forward_source

    assert "resolve_prefill_chunk_size" in _MODEL_H.read_text()
    assert "resolve_chunk_size_for_shape(" in _MODEL_H.read_text()
    assert "rpu_causal_decoder_resolve_prefill_chunk_size" in _MODEL_CPP.read_text()
    assert "rpu_causal_decoder_resolve_prefill_chunk_size" in _DECLS.read_text()
    assert "causal_decoder_resolve_prefill_chunk_size" in _BACKEND.read_text()


# These host tests exercise padding-plan search with a fake resolver.
# Validating the native planner requires RPU hardware and Qwen3-VL weights;
# host-only results do not establish hardware correctness.
