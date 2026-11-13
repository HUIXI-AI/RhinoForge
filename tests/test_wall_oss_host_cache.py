"""Inference runtime helpers and execution contracts."""
import importlib.util
import inspect
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

# The host-safe package now exposes shared planner/errors/mask helpers without
# loading a device when its native extension is unavailable. Load those real
# dependencies before the legacy adapter-specific mocks below.
import rpu_backend.runtime
from rpu_backend.runtime.execution_planner import PreparedExecutionPlans

from _module_stubs import real_rpu_env_bool, stub_modules

# --- stub rpu_backend (+ submodules the adapters import) so the real adapter
#     source files load on a plain CPU python (import rpu_backend SIGSEGVs sans root).
#     The stubs are SCOPED to the load below: installing them into sys.modules
#     permanently poisons the interpreter, and every later real
#     `import rpu_backend.adapters.*` in the same process then dies with
#     "No module named 'rpu_backend.quant._common'; 'rpu_backend.quant' is not a
#     package".
#     The scope is stub_modules(), NOT patch.dict — patch.dict restores a whole
#     sys.modules snapshot on exit and so evicts the transformers/torchvision/
#     safetensors chain that runtime.py imports below, which then cannot be
#     re-imported in this process. See tests/_module_stubs.py.
#     Only ABSENT names are stubbed (the old `setdefault` semantics): when a
#     real submodule is already imported, the adapters keep using it. ---
_REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_WALL = os.path.join(_REPO, "python/rpu_backend/adapters/wall_oss")
_stubs = {
    name: MagicMock()
    for name in (
        "rpu_backend", "rpu_backend.api", "rpu_backend.api.cache",
        "rpu_backend.runtime", "rpu_backend.runtime.decoder",
        "rpu_backend.runtime.weights", "rpu_backend.model_registry",
        "rpu_backend.quant", "rpu_backend.quant.int4_pgrp_pack",
        "rpu_backend.quant.nvfp4_pack",
        "rpu_backend.adapters", "rpu_backend.adapters.wall_oss",
        "rpu_backend.adapters.wall_oss.checkpoint_validation",
        "rpu_backend.adapters.wall_oss.llm", "rpu_backend.adapters.wall_oss.action",
        "rpu_backend.adapters.wall_oss.vision", "rpu_backend.runtime.rope_partial",
    )
    if name not in sys.modules
}
# ... except `rpu_backend.runtime`, whose `rpu_env_bool` decides every adapter
# switch: a MagicMock is truthy for all of them, so the run silently uses a
# configuration no test asked for. Without this, running THIS FILE ALONE (no
# earlier test having imported the real module) failed 4 of its own assertions.
if "rpu_backend.runtime" in _stubs:
    _stubs["rpu_backend.runtime"] = types.ModuleType("rpu_backend.runtime")
    _stubs["rpu_backend.runtime"].rpu_env_bool = real_rpu_env_bool()

# Cold configuration and planner leaves use only the stdlib at import. Keep
# their real constants/decorators even when no earlier test imported the package.
for _leaf, _source in (
    ("rpu_backend.runtime.execution_planner", "runtime/execution_planner.py"),
    ("rpu_backend.api._execution", "api/_execution.py"),
    ("rpu_backend.adapters.wall_oss._sorting", "adapters/wall_oss/_sorting.py"),
):
    if _leaf not in sys.modules:
        _spec = importlib.util.spec_from_file_location(
            "_walloss_" + _leaf.rsplit(".", 1)[-1],
            os.path.join(_REPO, "python/rpu_backend", _source),
        )
        _module = importlib.util.module_from_spec(_spec)
        with stub_modules({_spec.name: _module}):
            _spec.loader.exec_module(_module)
        _stubs[_leaf] = _module


def _load(modname, filename):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_WALL, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


with stub_modules(_stubs):
    vis = _load("walloss_vision_real", "vision.py")
    act = _load("walloss_action_real", "action.py")
    llm = _load("walloss_llm_real", "llm.py")
    # runtime.py does `from ...wall_oss.action import _dof_mask_is_row_constant`,
    # so the module it resolves must carry the real function before runtime.py runs.
    sys.modules[
        "rpu_backend.adapters.wall_oss.action"
    ]._dof_mask_is_row_constant = act._dof_mask_is_row_constant
    rt = _load("walloss_runtime_real", "runtime.py")  # transformers real; rpu_backend mocked


def _make_vision(window=True, host_cache=True, per_window_sdpa=False):
    v = vis.WallOssVision.__new__(vis.WallOssVision)
    v._sms, v._window, v._window_size, v._patch_size = 2, window, 112, 14
    v._batch_vision, v._batch_max_seq = True, 1728
    v._layer_group, v._weight_outer = 0, False
    v._layout_cache = {}
    v._group_layout_cache = {}
    v._index_rpu_cache = {}
    v._position_keepalive_snapshot = None
    v._device_patch_embed = False
    v._linear_acc32 = False
    v._patch_w_rpu = v._patch_w_fold_rpu = None
    v._pe_buf_cache = {}
    v._host_cache = host_cache
    v._last_window_mask_ref = None
    v._per_window_sdpa = per_window_sdpa
    return v


def _make_action(host_cache=True, partial_mrope=True):
    a = act.WallOssAction.__new__(act.WallOssAction)
    a._bias_cache = None
    a._host_cache = host_cache
    a._partial_mrope = partial_mrope
    a._kv_owner_fanout = False
    a._action_rope_cache = {}
    a._action_pos_cache = {}
    a._action_mask_cache = {}
    a._last_action_rope_ref = None
    a._last_action_pos_ref = None
    a._last_action_mask_ref = None
    a._causal_action_attention_mask = True
    a._flow_time_scale = 1.0
    a._rtc_bias_cache = None
    return a


def _action_pos(next_pos, H=32):
    return torch.arange(next_pos, next_pos + H, dtype=torch.int32)[:, None].expand(H, 3).contiguous()


def _make_llm(partial_mrope=True, host_cache=True):
    o = llm.WallOssLLM.__new__(llm.WallOssLLM)
    o._partial_mrope, o._host_cache, o._prefill_rope_cache = partial_mrope, host_cache, {}
    o._prefill_pos_cache, o._last_prefill_pos_ref = {}, None
    o._last_prefill_rope_ref = None
    return o


def _make_vla(host_cache=True):
    import types as _t
    o = rt.WallOssVLA.__new__(rt.WallOssVLA)
    o._rope_cache, o._host_cache = {}, host_cache
    o._embed_ids_cache = None
    return o


def _make_prepared_vla():
    o = _make_vla()
    o.action = types.SimpleNamespace(
        _fused=True,
        _action_fp32_tail=False,
        _denoise_unroll=True,
    )
    o.llm = types.SimpleNamespace(
        prefill_execution_plan=(
            lambda _real_len, *, reserve_rows=0: (640, 320)
        )
    )
    o._prepared_graph_profile = {
        "prefix_length_range": (545, 640),
        "execution_keys": frozenset({(640, 320)}),
        "grid_key": (1, 18, 32, 1, 18, 32, 1, 18, 32),
        "horizon": 32,
        "num_steps": 10,
        "row_constant": True,
        "denoise_mode": {
            "fused": True,
            "fp32_tail": False,
            "unroll": True,
        },
    }
    return o


def test_process_inputs_repeats_image_token_ids_exactly():
    text = "11<|image_pad|>12<|image_pad|>13<|image_pad|>14"
    base = torch.tensor([[11, rt.IMAGE_PAD_ID, 12, rt.IMAGE_PAD_ID,
                          13, rt.IMAGE_PAD_ID, 14]])
    grid = torch.tensor([[1, 4, 4], [1, 2, 4], [1, 2, 2]])
    expected = torch.tensor([[11] + [rt.IMAGE_PAD_ID] * 4 + [12]
                             + [rt.IMAGE_PAD_ID] * 2 + [13]
                             + [rt.IMAGE_PAD_ID] + [14]])
    tokenized = []

    class Processor:
        image_processor = types.SimpleNamespace(merge_size=2)
        image_token_id = rt.IMAGE_PAD_ID

        def tokenizer(self, texts, return_tensors):
            tokenized.append(texts[0])
            return {"input_ids": base.clone()}

        def __call__(self, **_):
            return {
                "input_ids": expected.clone(),
                "pixel_values": torch.zeros(1, 2),
                "image_grid_thw": grid,
            }

    o = rt.WallOssVLA.__new__(rt.WallOssVLA)
    o._processor = Processor()
    o._fast_processor = True
    o._fast_proc_ok = None
    o._image_preproc = lambda _: (torch.zeros(1, 2), grid)

    actual, _, _ = o._process_inputs(text, [object(), object(), object()])

    assert torch.equal(actual, expected)
    assert tokenized == [text]
    assert o._fast_proc_ok is True


def test_process_inputs_marker_grid_mismatch_disables_fast_path():
    calls = {"tokenizer": 0, "wrapper": 0}
    grid = torch.tensor([[1, 2, 2], [1, 2, 2]])

    class Processor:
        image_processor = types.SimpleNamespace(merge_size=2)
        image_token_id = rt.IMAGE_PAD_ID

        def tokenizer(self, *_args, **_kwargs):
            calls["tokenizer"] += 1
            return {"input_ids": torch.tensor([[rt.IMAGE_PAD_ID]])}

        def __call__(self, **_):
            calls["wrapper"] += 1
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "pixel_values": torch.zeros(1, 2),
                "image_grid_thw": grid,
            }

    o = rt.WallOssVLA.__new__(rt.WallOssVLA)
    o._processor = Processor()
    o._fast_processor = True
    o._fast_proc_ok = None
    o._image_preproc = lambda _: (torch.zeros(1, 2), grid)

    with pytest.warns(UserWarning, match="image marker count mismatch"):
        o._process_inputs("<|image_pad|>", [object(), object()])
    o._process_inputs("<|image_pad|>", [object(), object()])

    assert calls == {"tokenizer": 1, "wrapper": 2}
    assert o._fast_proc_ok is False


def _validate_prepared_vla(o):
    o._validate_prepared_request(
        real_len=592,
        grid=torch.tensor([[1, 18, 32]] * 3),
        horizon=32,
        num_steps=10,
        row_constant=True,
    )


def test_ready_profile_accepts_unchanged_denoise_mode():
    o = _make_prepared_vla()
    _validate_prepared_vla(o)


@pytest.mark.parametrize(
    ("env_name", "changed_value"),
    [
        ("RPU_WALL_OSS_ACTION_FP32_TAIL", "1"),
        ("RPU_WALL_OSS_DENOISE_UNROLL", "0"),
    ],
)
def test_ready_profile_ignores_denoise_env_drift(env_name, changed_value):
    o = _make_prepared_vla()
    env = {
        "RPU_WALL_OSS_ACTION_FP32_TAIL": "0",
        "RPU_WALL_OSS_DENOISE_UNROLL": "1",
        env_name: changed_value,
    }
    with patch.dict(os.environ, env):
        _validate_prepared_vla(o)


@pytest.mark.parametrize(
    ("attr_name", "changed_value"),
    [
        ("_action_fp32_tail", True),
        ("_denoise_unroll", False),
    ],
)
def test_ready_profile_rejects_instance_mode_drift(attr_name, changed_value):
    o = _make_prepared_vla()
    setattr(o.action, attr_name, changed_value)
    with pytest.raises(RuntimeError, match="execution mode changed"):
        _validate_prepared_vla(o)


def test_ready_profile_rejects_fused_mode_drift():
    o = _make_prepared_vla()
    o.action._fused = False
    with pytest.raises(RuntimeError, match="execution mode changed"):
        _validate_prepared_vla(o)


# ─────────────────────── prefill chunk-size cap (llm.py) ────────────────────
def test_prefill_chunk_cap_tp8_pad16_is_384():
    with patch.dict(os.environ, {}, clear=True):
        assert llm._prefill_chunk_size_cap(8) == 384


def test_prefill_chunk_cap_tp8_without_pad16_stays_256():
    with patch.dict(os.environ, {"RPU_WALL_OSS_KVINSERT_PAD16": "0"}):
        assert llm._prefill_chunk_size_cap(8) == 256


def test_prefill_chunk_cap_tp2_stays_256():
    with patch.dict(os.environ, {"RPU_WALL_OSS_KVINSERT_PAD16": "1"}):
        assert llm._prefill_chunk_size_cap(2) == 256


# ───────────────────── equal-two prefill execution plan ────────────────────
# WHICH chunk sizes are admissible is NOT decided here. `rpu_backend` is stubbed
# in this module, so `_equal_two_chunk_supported`'s C++ predicate would come back
# as a truthy MagicMock and every candidate would "pass" — the exact shape of the
# defect these tests exist to catch. The admissibility TRUTH is gated against the
# real predicate in tests/ops/test_wall_oss_prefill_envelope.py (hw_mech). What
# is gated here is the SEARCH STRUCTURE around it, with the predicate injected.
_MEASURED_EQUAL_TWO = {   # 2026-08-10, sdpa_helpers.is_valid_chunk_size
    160: True, 176: True, 192: True, 208: False, 224: True, 240: False,
    256: True, 288: False, 304: False, 320: True, 352: False, 384: False,
}


def _plan(real_len, cap=384, **kw):
    """_prefill_execution_plan with the measured predicate injected."""
    kw.setdefault("pad16", True)
    kw.setdefault("equal_two", True)
    with patch.object(llm, "_equal_two_chunk_supported",
                      lambda cs, exec_len, geom: _MEASURED_EQUAL_TWO[cs]):
        return llm._prefill_execution_plan(real_len, cap, geom={}, **kw)


def test_prefill_execution_plan_equal_two_matrix():
    expected = {
        385: (448, 224),
        576: (640, 320),
        577: (640, 320),
        592: (640, 320),
        608: (640, 320),
        624: (640, 320),
        640: (640, 320),
    }
    for real_len, plan in expected.items():
        assert _plan(real_len) == plan


def test_prefill_execution_plan_release_envelope_is_one_profile():
    assert {_plan(real_len) for real_len in range(545, 641)} == {(640, 320)}


def test_prefill_execution_plan_refuses_a_plan_the_kernel_rejects():
    """737-768 used to plan (768, 384) — cleanly, on the host, and then hard-fail
    in compute_chunks_impl, because the host modelled tile_m as the largest
    divisor of chunk_size/16 while the runtime tiler caps tile_m_v16 at 8 above
    sQry=176. cs=384's SECOND call is grid=3 * gqa=2 == 6, the sync shape
    rpu_helpers.h:373 rejects. There is no legal two-chunk plan up to the 384
    cap, so the host must now say so instead of certifying one."""
    for real_len in (737, 768):
        with pytest.raises(ValueError,
                           match="no safe two-chunk execution profile"):
            _plan(real_len)


def test_rtc_prefill_execution_plan_splits_graph_profiles_at_640():
    o = _make_vla()
    o.rtc_enabled = True
    profiles = o._prefix_execution_profiles((599, 671), horizon=20)
    assert [
        (p["real_min"], p["real_max"], p["execution_len"], p["chunk_size"])
        for p in profiles
    ] == [
        (599, 640, 640, 320),
        (641, 671, 672, 336),
    ]
    assert o._prefill_execution_plan(640, horizon=20) == (640, 320)
    assert o._prefill_execution_plan(641, horizon=20) == (672, 336)
    with pytest.raises(ValueError, match=r"\[599, 671\]"):
        o._prefill_execution_plan(598, horizon=20)
    with pytest.raises(ValueError, match=r"\[599, 671\]"):
        o._prefill_execution_plan(672, horizon=20)


def test_prefill_graph_signature_policy_version_changes_cache_key():
    def fake_signature(**kwargs):
        return (
            kwargs["op_id"],
            tuple(kwargs["shapes"]),
            tuple(kwargs["dyn_dims"]),
            tuple(kwargs["dtypes"]),
        )

    with patch.object(
        llm.rpu_backend.graph, "GraphSignature", side_effect=fake_signature
    ):
        with patch.object(llm, "_PREFILL_CHUNK_POLICY_VERSION", 3):
            version_3 = llm._prefill_graph_signature(
                "wall_oss_llm_embeds", 640, 2048, 320, 36,
                plan_key_words=(10, 11),
            )
        with patch.object(llm, "_PREFILL_CHUNK_POLICY_VERSION", 4):
            version_4 = llm._prefill_graph_signature(
                "wall_oss_llm_embeds", 640, 2048, 320, 36,
                plan_key_words=(10, 11),
            )
        mixed = llm._prefill_graph_signature(
            "wall_oss_llm_embeds", 640, 2048, 320, 36, True,
            plan_key_words=(10, 11),
        )

    assert version_3[2] == (36, 3, 0, 10, 11)
    assert version_4[2] == (36, 4, 0, 10, 11)
    assert mixed[2] == (36, 4, 1, 10, 11)
    assert version_3 != version_4
    assert version_3 != mixed


def test_prefill_execution_plan_keeps_single_chunk():
    assert _plan(383) == (384, 384)


def test_prefill_execution_plan_rejects_more_than_two_chunks():
    try:
        _plan(769)
    except ValueError as exc:
        assert "has no safe two-chunk execution profile" in str(exc)
    else:
        raise AssertionError("769-token prefill must not silently become three chunks")


def test_equal_two_chunk_support_delegates_to_the_cpp_predicate():
    """The host must not re-derive kernel feasibility. Only C288 (an ACC16
    prefix-gate finding the SDPA predicate does not model) is decided locally."""
    ext = types.SimpleNamespace(sdpa_helpers=MagicMock())
    pred = ext.sdpa_helpers.is_valid_chunk_size
    pred.return_value = False
    with patch.dict(sys.modules, {"rpu_backend._cpp_ext": ext}):
        assert not llm._equal_two_chunk_supported(288, 576, {"x": 1})
        pred.assert_not_called()                       # short-circuited locally

        assert not llm._equal_two_chunk_supported(320, 640, {"x": 1})
        pred.assert_called_once_with(x=1, cs=320, seq_len=640, position=0)

        pred.return_value = True
        assert llm._equal_two_chunk_supported(320, 640, {"x": 1})


def _attach_cold_planner(model, monkeypatch):
    plans = PreparedExecutionPlans()
    model._graph_cache = types.SimpleNamespace(
        is_frozen=lambda: False, prepare_plan=plans.prepare)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_planner_cache_identity",
                        lambda handle: (handle, 1), raising=False)


def test_p672_c336_is_scoped_to_mixed_prefill(monkeypatch):
    model = llm.WallOssLLM.__new__(llm.WallOssLLM)
    _attach_cold_planner(model, monkeypatch)
    model._handle = 7
    model._graph_cache = None  # cold planning, no native Graph on the host
    model.cache = types.SimpleNamespace(max_seq_len=2048)
    model._rpu_execution = {"prefill": {}}
    model._prefill_chunk_size_cap = 384
    model._prefill_pad16 = True
    model._prefill_equal_two = True
    model._attn_geometry = {}
    model._prefill_mixed_precision = True
    monkeypatch.setattr(
        torch.ops.rpu, "causal_decoder_planner_cache_identity",
        lambda handle: (handle, 1), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "causal_decoder_resolve_prefill_stage_domain",
        lambda _handle, _seq, _position, _causal, _mask_kv,
               rope_mode, graph_lifecycle, logical_len: [
            (336, 336, 336)
        ] if (rope_mode, graph_lifecycle, logical_len) == (1, 4, 672) else [],
        raising=False,
    )
    for requested in ("auto", 336):
        model._rpu_execution = {"prefill": {"chunk_size": requested}}
        plans = []
        assert model.prefill_execution_plan(672, plan_result_sink=plans.append) == (672, 336)
        assert plans[0].request.mode == ("AUTO" if requested == "auto" else "EXACT")
    model._rpu_execution = {"prefill": {}}

    model._prefill_mixed_precision = False
    with patch.object(llm, "_equal_two_chunk_supported", return_value=False):
        with pytest.raises(ValueError, match="no safe two-chunk"):
            model.prefill_execution_plan(672)


def test_prefill_execution_plan_pad16_optout_keeps_legacy_shape():
    assert _plan(592, 256, pad16=False, equal_two=False) == (592, 256)


def _public_plan(monkeypatch, execution, resolver):
    model = llm.WallOssLLM.__new__(llm.WallOssLLM)
    _attach_cold_planner(model, monkeypatch)
    model._handle = 7
    model._graph_cache = None  # cold planning, no native Graph on the host
    model.cache = types.SimpleNamespace(max_seq_len=2048)
    model._prefill_chunk_size_cap = 384
    model._prefill_pad16 = True
    model._prefill_equal_two = False
    model._prefill_mixed_precision = False
    model._attn_geometry = {}
    model._rpu_execution = {"prefill": execution}
    monkeypatch.setattr(
        torch.ops.rpu, "causal_decoder_planner_cache_identity",
        lambda handle: (handle, 1), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "causal_decoder_resolve_prefill_stage_domain",
        lambda handle, seq, position, _causal, _mask_kv,
               rope_mode, graph_lifecycle, logical_len: [(
            resolver(handle, seq, position),
        ) * 3] if (rope_mode, graph_lifecycle) == (1, 4) and 0 < logical_len <= seq else [],
        raising=False,
    )
    return model.prefill_execution_plan


def test_public_prefill_exact_padding_and_chunk_share_one_native_plan(monkeypatch):
    calls = []

    def resolve(handle, seq, position):
        calls.append((handle, seq, position))
        return 320

    plan = _public_plan(
        monkeypatch,
        {"chunk_size": 320, "padding_rows": 48},
        resolve,
    )
    assert plan(592) == (640, 320)
    assert calls == [(7, 640, 0)]


def test_public_prefill_auto_padding_buys_fewer_chunks(monkeypatch):
    plan = _public_plan(
        monkeypatch,
        {"chunk_size": "auto", "padding_budget": 48},
        lambda _handle, seq, _position: 320 if seq == 640 else 160,
    )
    assert plan(592) == (640, 320)


def test_public_prefill_auto_padding_stays_inside_action_reserve(monkeypatch):
    calls = []

    def resolve(_handle, seq, _position):
        calls.append(seq)
        return 1008

    plan = _public_plan(
        monkeypatch,
        {"chunk_size": "auto", "padding_budget": 64},
        resolve,
    )

    assert plan(2001, reserve_rows=32) == (2016, 1008)
    assert calls == [2016]


def test_public_prefill_reserves_action_rows_before_padding(monkeypatch):
    plan = _public_plan(
        monkeypatch,
        {"chunk_size": "auto", "padding_budget": 64},
        lambda _handle, _seq, _position: 32,
    )
    with pytest.raises(RuntimeError, match="NO_FEASIBLE"):
        plan(2017, reserve_rows=32)


def test_legacy_prefill_plan_cannot_consume_reserved_action_rows(monkeypatch):
    plan = _public_plan(
        monkeypatch,
        {},
        lambda _handle, _seq, _position: 32,
    )
    with pytest.raises(ValueError, match="reserved action rows"):
        plan(2017, reserve_rows=32)


def test_action_cache_capacity_guard_rejects_prefix_plus_horizon_overflow():
    cache = types.SimpleNamespace(max_seq_len=2048)
    with pytest.raises(RuntimeError, match="required=2064"):
        act._ensure_action_cache_capacity(cache, 2032, 32)


def test_low_level_precision_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="pick one"):
        llm.build_wall_oss_llm(
            "/unused",
            w8a16=True,
            nvfp4a16=True,
        )


def test_low_level_nvfp4_builder_validates_w8_down_pair_before_loading():
    with patch.object(
        llm,
        "validate_w8_nvfp4_checkpoint_pair",
        side_effect=ValueError("checkpoint pair mismatch"),
    ) as validate:
        with pytest.raises(ValueError, match="checkpoint pair mismatch"):
            llm.build_wall_oss_llm(
                "/fp16",
                expert=0,
                nvfp4a16=True,
                w8_down_ckpt_dir="/w8",
            )
    validate.assert_called_once_with("/w8", "/fp16")


# ─────────────────────────── vision window layout ───────────────────────────
def test_window_layout_cache_hit_returns_same_object():
    v = _make_vision()
    grid, seq = torch.tensor([[1, 16, 16]]), 256
    assert v._window_layout(grid, seq) is v._window_layout(grid, seq)


def test_window_layout_bit_identical_to_fresh():
    v = _make_vision()
    grid, seq, smu = torch.tensor([[1, 16, 16]]), 256, 4
    pos_idx, win, rev, mask = v._window_layout(grid, seq)
    fpos = vis._compute_vision_position_idx_cpu(grid, 2)
    fwin, fcu, frev = vis._get_window_index(grid, 112, 2, 14)
    fpos = fpos.reshape(seq // smu, smu, 2)[fwin].reshape(seq, 2)
    fmask = vis._block_diag_mask(fcu, seq)
    assert torch.equal(pos_idx, fpos)
    assert torch.equal(win, fwin)
    assert torch.equal(rev, frev)
    assert torch.equal(mask, fmask)


def test_window_layout_grid_change_recomputes():
    v = _make_vision()
    a = v._window_layout(torch.tensor([[1, 16, 16]]), 256)
    b = v._window_layout(torch.tensor([[1, 8, 8]]), 64)
    assert a is not b
    assert a[0].size(0) == 256 and b[0].size(0) == 64


def test_window_layout_non_window_mode():
    v = _make_vision(window=False)
    pos_idx, win, rev, mask = v._window_layout(torch.tensor([[1, 16, 16]]), 256)
    assert win is None and rev is None and mask is None
    assert torch.equal(pos_idx, vis._compute_vision_position_idx_cpu(torch.tensor([[1, 16, 16]]), 2))


def test_window_layout_toggle_off_recomputes():
    v = _make_vision(host_cache=False)
    grid, seq = torch.tensor([[1, 16, 16]]), 256
    a = v._window_layout(grid, seq)
    b = v._window_layout(grid, seq)
    assert a is not b                      # RPU_WALL_OSS_HOST_CACHE=0 → no memoization
    assert not v._layout_cache             # nothing stored
    assert torch.equal(a[0], b[0])         # still correct


def test_window_layout_toggle_off_keeps_previous_mask_during_next_allocation():
    v = _make_vision(host_cache=False)
    grids = (
        torch.tensor([[1, 18, 32]]),
        torch.tensor([[1, 24, 24]]),
    )
    prior_refs = []

    def build_mask(_cu_window_seqlens, _seq):
        prior_refs.append(v._last_window_mask_ref)
        return torch.full((1,), len(prior_refs), dtype=torch.float16)

    with patch.object(vis, "_block_diag_mask", side_effect=build_mask):
        first = v._window_layout(grids[0], 576)
        second = v._window_layout(grids[1], 576)

    assert prior_refs == [None, first[3]]
    assert v._last_window_mask_ref is second[3]
    assert not torch.equal(first[3], second[3])


def test_per_window_layout_omits_dense_mask_and_keeps_boundaries():
    v = _make_vision(per_window_sdpa=True)
    grid, seq = torch.tensor([[1, 28, 28]]), 784
    with patch.object(
        vis, "_block_diag_mask", side_effect=AssertionError("dense mask built")
    ):
        _pos, _win, _rev, mask, cu = v._window_layout(grid, seq)
    assert mask is None
    assert cu[0].item() == 0 and cu[-1].item() == seq
    assert bool((cu[1:] > cu[:-1]).all())
    assert bool((cu.remainder(16) == 0).all())


def test_per_window_signature_keys_full_same_seq_different_grid_boundaries():
    v = _make_vision(per_window_sdpa=True)
    grid_a = torch.tensor([[1, 28, 28]])
    grid_b = torch.tensor([[1, 14, 56]])
    assert int(grid_a.prod().item()) == int(grid_b.prod().item()) == 784
    cu_a = v._window_layout(grid_a, 784)[4]
    cu_b = v._window_layout(grid_b, 784)[4]
    base = [32, 1, 0, 0]
    sig_a = vis._vision_graph_dyn_dims(base, tuple(cu_a.tolist()))
    sig_b = vis._vision_graph_dyn_dims(base, tuple(cu_b.tolist()))
    assert sig_a != sig_b
    assert sig_a[-cu_a.numel():] == cu_a.tolist()
    assert sig_b[-cu_b.numel():] == cu_b.tolist()
    assert vis._vision_graph_dyn_dims(base, None) == base


def test_group_layout_exposes_exact_3x576_window_boundaries():
    v = _make_vision()
    layout = v._group_layout(torch.tensor([[1, 18, 32]] * 3), 1728, 4)
    cu = layout[4]
    expected = [0, 64, 128, 192, 256, 320, 384,
                448, 512, 528, 544, 560, 576]
    assert cu.tolist() == expected
    base = [32, 576, 1, 3, 0, 1, 1, 0]
    assert vis._vision_graph_dyn_dims(base, tuple(cu.tolist())) == [*base, 1, *expected]


def test_vision_planner_and_graph_share_current_boundary_snapshot(monkeypatch):
    v = _make_vision(per_window_sdpa=True)
    v._handle, v._graph_cache = 7, object()
    signatures, native_inputs = [], []
    plan = types.SimpleNamespace(selected=types.SimpleNamespace(
        stage_tuple=types.SimpleNamespace(physical_descriptor=(3, 17))))
    def planner(length, *args, **kwargs):
        signatures.append(kwargs["plan_signature"])
        kwargs["resolve_stage_domain"](length)
        kwargs["plan_result_sink"](plan)
        return length, 64
    monkeypatch.setattr(vis, "plan_bounded_prefill_execution", planner)
    monkeypatch.setattr(torch.ops.rpu, "qwen25vl_vision_resolve_stage_domain",
        lambda handle, length, images, mask, boundaries, exact:
            native_inputs.append(boundaries), raising=False)
    cu = torch.tensor([0, 64, 128], dtype=torch.long)
    _, actual, descriptor, first = v._plan_native_execution(128, 1, None, cu)
    assert actual is plan and descriptor == (3, 17)
    assert native_inputs[-1] is cu
    assert first is signatures[-1][-1] and first == (0, 64, 128)
    # The Graph key belongs to the same content snapshot as the planner key.
    cu[1] = 48
    assert vis._vision_graph_dyn_dims([32], first) == [32, 1, 0, 64, 128]
    _, _, _, second = v._plan_native_execution(128, 1, None, cu)
    assert second is signatures[-1][-1] and second == (0, 48, 128)
    assert vis._vision_graph_dyn_dims([32], second) != vis._vision_graph_dyn_dims([32], first)
    assert v._plan_native_execution(128, 1, None, None)[3] is None
    assert signatures[-1][-1] is None and native_inputs[-1] is None


def test_group_layout_toggle_off_keeps_previous_mask_during_next_allocation():
    v = _make_vision(host_cache=False)
    grids = (
        torch.tensor([[1, 18, 32]] * 3),
        torch.tensor([[1, 24, 24]] * 3),
    )
    prior_refs = []

    def pack_group(grid_group, _window_size, sms, _patch_size):
        prior_refs.append(v._last_window_mask_ref)
        seq = int(grid_group.prod(-1).sum().item())
        merge_units = seq // (sms * sms)
        mask = torch.full((1,), len(prior_refs), dtype=torch.float16)
        cu = torch.tensor([0, seq // grid_group.size(0)], dtype=torch.long)
        return torch.arange(merge_units), (), mask, cu

    with patch.object(vis, "_pack_multi_image", side_effect=pack_group):
        first = v._group_layout(grids[0], 1728, 4)
        second = v._group_layout(grids[1], 1728, 4)

    assert prior_refs == [None, first[3]]
    assert v._last_window_mask_ref is second[3]
    assert not torch.equal(first[3], second[3])


@pytest.mark.parametrize(
    ("layer_group", "weight_outer", "match"),
    [
        (32, False, "validated only for 3x576 patches"),
        (0, True, "validated only for 3x576 patches"),
    ],
)
def test_experimental_vision_schedule_rejects_single_image_before_dispatch(
    layer_group, weight_outer, match,
):
    v = _make_vision()
    v._layer_group = layer_group
    v._weight_outer = weight_outer
    v._fused_assemble = False
    v._pending_rev = None
    v._forward_one = MagicMock()
    env = {
        "RPU_WALL_OSS_BATCH_VISION": "1",
        "RPU_WALL_OSS_VISION_LAYER_GROUP": str(layer_group),
        "RPU_WALL_OSS_VISION_WEIGHT_OUTER": "1" if weight_outer else "0",
        "RPU_WALL_OSS_VISION_MIXED_PRECISION": "0",
    }

    with patch.dict(os.environ, env):
        with pytest.raises(ValueError, match=match):
            v.forward(
                torch.zeros(576, 1176),
                torch.tensor([[1, 24, 24]]),
            )

    v._forward_one.assert_not_called()


# ─────────────────────────── denoise bias inputs ────────────────────────────
def _stub_precompute(a):
    """Replace both precompute paths with shared call-counting CPU stubs."""
    calls = []
    a._precompute_bstep_base_te = lambda dof, times, n: (
        calls.append("te"), (torch.zeros(n, 4), torch.zeros(n, 4)))[1]
    a._precompute_b_step = lambda dof, times, n: (
        calls.append("b"), torch.zeros(n, dof.size(1), 4))[1]
    return calls


def test_bias_cache_hit_single_compute():
    a = _make_action()
    calls = _stub_precompute(a)
    dof, times = torch.ones(1, 32, 26), torch.linspace(0, 1, 11)
    row1, b1, te1 = a._denoise_bias_inputs(dof, times, 10)
    row2, b2, te2 = a._denoise_bias_inputs(dof, times, 10)
    assert len(calls) == 1                 # 2nd call served from cache
    assert row1 is True                    # ones → row-constant
    assert b1 is b2 and te1 is te2          # same cached tensors


def test_bias_cache_dof_change_recomputes():
    a = _make_action()
    calls = _stub_precompute(a)
    times = torch.linspace(0, 1, 11)
    a._denoise_bias_inputs(torch.ones(1, 32, 26), times, 10)
    a._denoise_bias_inputs(torch.full((1, 32, 26), 0.5), times, 10)  # different dof
    assert len(calls) == 2


def test_bias_near_constant_dof_uses_full_rowwise_path():
    a = _make_action()
    calls = _stub_precompute(a)
    dof = torch.ones(1, 32, 26)
    dof[0, 1, 0] += 1e-6

    row_constant, b_all, te_all = a._denoise_bias_inputs(
        dof, torch.linspace(0, 1, 11), 10
    )

    assert row_constant is False
    assert calls == ["b"]
    assert b_all.shape == (10, 32, 4)
    assert te_all is None


def test_bias_cache_numsteps_change_recomputes():
    a = _make_action()
    calls = _stub_precompute(a)
    dof = torch.ones(1, 32, 26)
    a._denoise_bias_inputs(dof, torch.linspace(0, 1, 11), 10)
    a._denoise_bias_inputs(dof, torch.linspace(0, 1, 9), 8)         # different num_steps
    assert len(calls) == 2


# ───────────────────── prefill position/cos-sin cache (llm.py) ──────────────
def _patch_rpu_tensor_to(on_alloc):
    original_to = torch.Tensor.to
    device_storages = {}

    def fake_to(tensor, *args, **kwargs):
        positional_device = bool(args and args[0] == "rpu")
        if positional_device or kwargs.get("device") == "rpu":
            if positional_device:
                args = ("cpu", *args[1:])
            else:
                kwargs["device"] = "cpu"
            out = original_to(tensor, *args, **kwargs)
            if (tensor.untyped_storage()._cdata in device_storages
                    and out is tensor):
                return tensor  # Same-device to() preserves uploaded row views.
            out = out.clone()
            device_storages[out.untyped_storage()._cdata] = out.untyped_storage()
            on_alloc(out)
            return out
        return original_to(tensor, *args, **kwargs)

    return patch.object(torch.Tensor, "to", fake_to)


def test_prefill_position_cache_hit_reuses_stable_rpu_tensor():
    o = _make_llm()
    allocations = []
    pos = torch.zeros(192, 3, dtype=torch.int64)
    with _patch_rpu_tensor_to(allocations.append):
        p1 = o._prefill_position_ids_rpu(pos, 192)
        p2 = o._prefill_position_ids_rpu(pos.clone(), 192)
    assert len(allocations) == 1
    assert p1 is p2
    assert p1.dtype == torch.int32


def test_prefill_position_same_seq_content_change_allocates_before_replace():
    o = _make_llm()
    old_at_allocation = []

    def observe(_new):
        cached = o._prefill_pos_cache.get(192)
        old_at_allocation.append(None if cached is None else cached[1])

    a = torch.zeros(192, 3, dtype=torch.int64)
    b = torch.ones(192, 3, dtype=torch.int64)
    with _patch_rpu_tensor_to(observe):
        p_a = o._prefill_position_ids_rpu(a, 192)
        p_b = o._prefill_position_ids_rpu(b, 192)
    assert old_at_allocation == [None, p_a]
    assert p_b is not p_a
    assert o._prefill_pos_cache[192][1] is p_b
    assert torch.equal(o._prefill_pos_cache[192][0], b)


def _stub_bake(o):
    calls = []
    o._bake_prefill_cos_sin = lambda pos: (calls.append(1), (torch.zeros(4), torch.zeros(4)))[1]
    return calls


def test_prefill_cos_sin_hit_single_bake():
    o = _make_llm()
    calls = _stub_bake(o)
    pos = torch.zeros(192, 3, dtype=torch.long)
    c1 = o._prefill_cos_sin(pos, 192)
    c2 = o._prefill_cos_sin(pos, 192)
    assert len(calls) == 1 and c1[0] is c2[0]      # 2nd served from cache, same tensors


def test_prefill_cos_sin_diff_positions_same_seq_recomputes():
    o = _make_llm()
    calls = _stub_bake(o)
    o._prefill_cos_sin(torch.zeros(192, 3, dtype=torch.long), 192)
    o._prefill_cos_sin(torch.ones(192, 3, dtype=torch.long), 192)   # same seq, diff positions
    assert len(calls) == 2                          # torch.equal guard catches it


def test_prefill_cos_sin_diff_seq_recomputes():
    o = _make_llm()
    calls = _stub_bake(o)
    o._prefill_cos_sin(torch.zeros(192, 3, dtype=torch.long), 192)
    o._prefill_cos_sin(torch.zeros(176, 3, dtype=torch.long), 176)
    assert len(calls) == 2


def test_prefill_cos_sin_legacy_flag_cannot_disable_required_tables():
    o = _make_llm(partial_mrope=False)
    calls = _stub_bake(o)
    cos, sin = o._prefill_cos_sin(torch.zeros(192, 3, dtype=torch.long), 192)
    assert cos is not None and sin is not None and len(calls) == 1


def test_prefill_cos_sin_toggle_off_recomputes():
    o = _make_llm(host_cache=False)
    calls = _stub_bake(o)
    pos = torch.zeros(192, 3, dtype=torch.long)
    o._prefill_cos_sin(pos, 192)
    latest = o._prefill_cos_sin(pos, 192)
    assert len(calls) == 2 and not o._prefill_rope_cache
    assert o._last_prefill_rope_ref[0] is latest[0]
    assert o._last_prefill_rope_ref[1] is latest[1]


# ─────────────── denoise-seam: action cos/sin + mask (action.py) ────────────
def _stub_action_bake(a):
    calls = []
    a._bake_partial_mrope_tables = lambda pos: (calls.append(1), (torch.zeros(4), torch.zeros(4)))[1]
    return calls


def test_action_cos_sin_hit_single_bake():
    a = _make_action()
    calls = _stub_action_bake(a)
    pos = _action_pos(181)
    c1 = a._partial_mrope_tables(pos)
    c2 = a._partial_mrope_tables(pos)
    assert len(calls) == 1 and c1[0] is c2[0]


def test_action_cos_sin_diff_nextpos_recomputes():
    a = _make_action()
    calls = _stub_action_bake(a)
    a._partial_mrope_tables(_action_pos(181))
    a._partial_mrope_tables(_action_pos(165))      # different prefix length → different next_pos
    assert len(calls) == 2


def test_action_cos_sin_legacy_flag_cannot_disable_required_tables():
    a = _make_action(partial_mrope=False)
    calls = _stub_action_bake(a)
    cos, sin = a._partial_mrope_tables(_action_pos(181))
    assert cos is not None and sin is not None and len(calls) == 1


def test_action_cos_sin_toggle_off_recomputes():
    a = _make_action(host_cache=False)
    calls = _stub_action_bake(a)
    pos = _action_pos(181)
    a._partial_mrope_tables(pos)
    a._partial_mrope_tables(pos)
    assert len(calls) == 2 and not a._action_rope_cache
    assert a._last_action_rope_ref is not None


def test_action_position_a_b_a_reuses_stable_rpu_tensor():
    a = _make_action()
    allocations = []
    pos_a = _action_pos(591)
    pos_b = _action_pos(592)
    with _patch_rpu_tensor_to(allocations.append):
        p_a1 = a._action_position_ids_rpu(pos_a)
        p_b = a._action_position_ids_rpu(pos_b)
        p_a2 = a._action_position_ids_rpu(pos_a.clone())
    assert len(allocations) == 2
    assert p_a1 is p_a2 and p_a1 is not p_b
    assert p_a1.dtype == p_b.dtype == torch.int32


def test_action_position_same_nextpos_content_change_allocates_before_replace():
    a = _make_action()
    old_at_allocation = []

    def observe(_new):
        cached = a._action_pos_cache.get(591)
        old_at_allocation.append(None if cached is None else cached[1])

    pos_a = _action_pos(591)
    pos_changed = pos_a.clone()
    pos_changed[-1, -1] += 1
    with _patch_rpu_tensor_to(observe):
        p_a = a._action_position_ids_rpu(pos_a)
        p_changed = a._action_position_ids_rpu(pos_changed)
    assert old_at_allocation == [None, p_a]
    assert p_changed is not p_a
    assert a._action_pos_cache[591][1] is p_changed
    assert torch.equal(a._action_pos_cache[591][0], pos_changed)


def test_action_mask_cache_hit_and_bit_identical():
    a = _make_action()
    m1 = a._action_mask_cached(192, 32, 180)
    m2 = a._action_mask_cached(192, 32, 180)
    assert m1 is m2                                            # cache hit
    assert torch.equal(m1, a._build_action_mask(192, 32, 180))  # bit-identical to fresh


def test_action_mask_diff_key_recomputes():
    a = _make_action()
    m1 = a._action_mask_cached(192, 32, 180)
    m2 = a._action_mask_cached(176, 32, 164)
    assert m1 is not m2 and m1.shape != m2.shape


def test_action_mask_a_b_a_same_execution_shape_refreshes_by_pointer():
    a = _make_action()
    m_a1 = a._action_mask_cached(608, 32, 591)
    m_b = a._action_mask_cached(608, 32, 592)
    m_a2 = a._action_mask_cached(608, 32, 591)
    assert m_a1 is m_a2 and m_a1 is not m_b
    assert m_a1.shape == m_b.shape == (32, 640)
    assert not torch.equal(m_a1, m_b)


@pytest.mark.parametrize("action_dim", [26, 32])
@pytest.mark.parametrize("steps", [1, 10])
@pytest.mark.parametrize("debug", [False, True])
def test_unroll_final_action_reads_one_contiguous_padded_step(
        monkeypatch, action_dim, steps, debug):
    from contextlib import contextmanager

    class Graph:
        pending = None

        @contextmanager
        def capture(self, _signature):
            yield
            self.pending()
            self.pending = None

    graph, horizon, prefix, padded_dim = Graph(), 20, 16, 32
    plan = types.SimpleNamespace(
        selected=types.SimpleNamespace(stage_tuple=types.SimpleNamespace(physical_descriptor=(1,))),
        graph_key_words=lambda: ())
    owner = types.SimpleNamespace(
        action_dim=action_dim, _k_comb_pad=padded_dim, _action_dim_pad=padded_dim,
        _handle=7, _graph_cache=graph,
        llm=types.SimpleNamespace(cache=types.SimpleNamespace(position=prefix, k_caches=[], v_caches=[])),
        validate_execution_horizon=lambda rows: None,
        _rtc_inputs=lambda pos, actions, delay, rows: (pos, None, None),
        _action_position_ids_rpu=lambda pos, key: pos,
        _partial_mrope_tables=lambda pos, key: (None, None),
        _action_mask_cached=lambda *args, **kwargs: None,
        _flow_times=lambda count: torch.linspace(1, 0, count + 1),
        _denoise_bias_inputs=lambda *args: (True, torch.ones(8), torch.ones(steps, 8)),
        _velocity_blend_inputs=lambda *args: (None, None),
        _plan_action_chunk=lambda *args, **kwargs: (16, plan),
        _denoise_graph_signature=lambda *args, **kwargs: None,
        _record_action_plan=lambda *args, **kwargs: None)
    trajectories, reads = [], []
    original_to, original_empty = torch.Tensor.to, torch.empty

    def host_to(value, *args, **kwargs):
        if (kwargs.get("device") == "cpu" and kwargs.get("dtype") == torch.float32
                and trajectories
                and value.untyped_storage()._cdata == trajectories[-1].untyped_storage()._cdata):
            assert graph.pending is None  # Native DMA finished at capture exit.
            reads.append((tuple(value.shape), value.is_contiguous(),
                          value.storage_offset(), value.numel() * value.element_size()))
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return original_to(value, *args, **kwargs)

    def host_empty(*args, **kwargs):
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    def native(_handle, x0, _k, _v, _bias, _time, _mask, _pos, x_traj, v_traj, *args):
        trajectories.append(x_traj)

        def execute():
            x_traj.fill_(77)
            v_traj.fill_(-7)
            x_traj[-1:, :, :action_dim].copy_(x0[:, :, :action_dim] + 0.25)
        graph.pending = execute

    monkeypatch.setattr(torch.Tensor, "to", host_to)
    monkeypatch.setattr(torch, "empty", host_empty)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", lambda: None, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "wall_oss_action_denoise_loop_forward", native, raising=False)
    noise = torch.linspace(-0.99, 1.01, horizon * action_dim).reshape(1, horizon, action_dim)
    positions = torch.arange(prefix, prefix + horizon)[:, None].expand(horizon, 3).contiguous()
    outputs = []
    for offset in (0.0, 0.5, 0.0):
        current = noise + offset
        result = act.WallOssAction._denoise_with_mask_loop(
            owner, prefix, current, torch.ones_like(current), positions,
            num_steps=steps, debug=debug)
        output, expected = result["x_norm"], (current.half() + 0.25).float()
        assert output.shape == (1, horizon, action_dim)
        assert output.dtype == torch.float32 and output.device.type == "cpu"
        assert output.is_contiguous() and torch.equal(output, expected)
        assert reads[-1] == ((1, horizon, padded_dim), True,
                             (steps - 1) * horizon * padded_dim, 1280)
        if debug:
            assert result["xs"].shape == (steps + 1, horizon, action_dim)
            assert result["vs"].shape == (steps, horizon, action_dim)
            assert torch.equal(result["xs"][-1:], expected)
            assert torch.all(result["vs"] == -7)
        else:
            assert result["xs"] is result["vs"] is None
        trajectories[-1].fill_(-33)
        assert torch.equal(output, expected)
        outputs.append(output)
    assert torch.equal(outputs[0], outputs[2])
    assert not torch.equal(outputs[0], outputs[1])
    assert len({output.data_ptr() for output in outputs}) == 3


def test_rtc_action_mask_is_bidirectional_except_prefix_to_suffix():
    m = act.WallOssAction._build_action_mask(
        3, 4, 2, causal=False, rtc_delay=2)

    assert torch.all(m[:, :2] == 0)          # every action row sees REAL prefix
    assert torch.all(torch.isneginf(m[:, 2:3]))
    action = m[:, 3:]
    assert torch.all(action[:2, :2] == 0)
    assert torch.all(torch.isneginf(action[:2, 2:]))
    assert torch.all(action[2:] == 0)         # suffix sees prefix + suffix both ways


def test_rtc_rope_gap_changes_suffix_on_all_three_axes_once():
    a = _make_action()
    a.action_dim = 26
    a._rtc_max_delay = 10
    a._rtc_rope_gap = 10
    pos = _action_pos(100, H=4)
    prefix = torch.zeros(1, 4, 26)

    rtc_pos, rtc_prefix, key = a._rtc_inputs(pos, prefix, 2, 4)

    assert torch.equal(rtc_pos[:2], pos[:2])
    assert torch.equal(rtc_pos[2:], pos[2:] + 10)
    assert rtc_prefix.shape == (1, 2, 26)
    assert key == ("rtc", 100, 4, 2, 10)


def test_rtc_per_token_tau_uses_one_for_prefix_and_step_time_for_suffix():
    a = _make_action()
    a.action_hidden = 4
    a._w1_dof = torch.zeros(4, 2)
    a._w2_a = torch.eye(4)
    a._w2_t = torch.eye(4)
    dof = torch.zeros(1, 3, 2)
    times = torch.tensor([0.0, 0.4, 0.8])
    allocations = []

    with _patch_rpu_tensor_to(allocations.append):
        b = a._rtc_bias_inputs(dof, times, num_steps=2, rtc_delay=1)

    phi_one = act._sinusoidal_pos_emb(torch.tensor([1.0]), 4)[0]
    phi_zero = act._sinusoidal_pos_emb(torch.tensor([0.0]), 4)[0]
    phi_step = act._sinusoidal_pos_emb(torch.tensor([0.4]), 4)[0]
    assert torch.equal(b[0, 0], phi_one.half())
    assert torch.equal(b[1, 0], phi_one.half())
    assert torch.equal(b[0, 1], phi_zero.half())
    assert torch.equal(b[1, 1], phi_step.half())


def test_v728_flow_schedule_ends_at_point_999():
    a = _make_action()
    a._flow_time_scale = 0.999
    assert torch.equal(
        a._flow_times(6),
        torch.linspace(0.0, 1.0, 7) * 0.999,
    )


def test_masked_dof_velocity_matches_customer_padding_trajectory():
    noise = torch.tensor([[[0.2, 0.5], [-0.4, 2.0]]])
    dof = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    padding = torch.tensor([[[0.0, -1.0], [0.0, -1.0]]])

    mask, override = act._masked_dof_velocity_inputs(noise, dof, padding)
    assert torch.equal(mask, dof)
    assert torch.equal(
        override, torch.tensor([[[0.0, -1.5], [0.0, -3.0]]]))

    x = noise.clone()
    model_v = torch.tensor([[[4.0, 99.0], [8.0, -99.0]]])
    for _ in range(6):
        v = override + model_v * mask
        x += 0.999 / 6 * v
    assert torch.allclose(x[..., 1], -0.999 + 0.001 * noise[..., 1])


def test_masked_dof_velocity_requires_padding_but_all_active_is_noop():
    noise = torch.zeros(1, 2, 3)
    assert act._masked_dof_velocity_inputs(
        noise, torch.ones_like(noise), None) == (None, None)
    with pytest.raises(ValueError, match="padding_action is required"):
        act._masked_dof_velocity_inputs(
            noise, torch.tensor([[[1.0, 1.0, 0.0]]]).expand_as(noise), None)


def test_masked_dof_velocity_is_scoped_to_the_rtc_action_profile():
    action = _make_action()
    noise = torch.zeros(1, 2, 3)
    dof = torch.tensor([[[1.0, 1.0, 0.0]]]).expand_as(noise)
    padding = torch.ones_like(noise)

    action._masked_dof_velocity = False
    assert action._velocity_blend_inputs(noise, dof, None) == (None, None)

    action._masked_dof_velocity = True
    mask, override = action._velocity_blend_inputs(noise, dof, padding)
    assert torch.equal(mask, dof)
    assert torch.equal(override, 1.0 - dof)


def test_rtc_prefix_normalization_uses_matching_step_stats_without_clamp():
    o = rt.WallOssVLA.__new__(rt.WallOssVLA)
    o.action_dim = 2
    o.dataset_key = "ex_normal"
    o.rtc_enabled = True
    o._rtc_max_delay = 2
    o._normalizer_clamp = False
    o._normalizer = {
        "min.ex_normal": torch.tensor([[0.0, 10.0], [2.0, 20.0]]),
        "delta.ex_normal": torch.tensor([[2.0, 10.0], [4.0, 20.0]]),
    }
    prefix = torch.tensor([[3.0, 30.0], [6.0, 60.0]])

    norm = o._normalize_rtc_prefix(prefix, rtc_delay=2, horizon=2)

    assert torch.equal(norm, torch.tensor([[[2.0, 3.0], [1.0, 3.0]]]))


def test_padding_action_normalizes_physical_zero_per_step_without_clamp():
    o = rt.WallOssVLA.__new__(rt.WallOssVLA)
    o.action_dim = 2
    o.dataset_key = "ex_normal"
    o._normalizer_clamp = False
    o._normalizer = {
        "min.ex_normal": torch.tensor([[0.0, 2.0], [-1.0, 4.0]]),
        "delta.ex_normal": torch.tensor([[1.0, 2.0], [2.0, 4.0]]),
    }

    padding = o._normalized_padding_action(2)

    assert torch.equal(
        padding, torch.tensor([[[-1.0, -3.0], [0.0, -3.0]]]))


def _fake_signature(**kwargs):
    return (
        kwargs["op_id"],
        tuple(kwargs["shapes"]),
        tuple(kwargs["dyn_dims"]),
        tuple(kwargs["dtypes"]),
    )


def test_denoise_execution_profile_a_b_a_keeps_one_graph_key():
    """One key per EXECUTION profile — and the mask is what carries the real prefix.

    ⚠️ Rewritten 2026-08-09. This test used to build its three keys inside
    `for _real_prefix_len in (591, 592, 591)` while passing only literals to
    `_denoise_graph_signature`, which has no real_prefix_len parameter at all.
    Three calls to a deterministic function with byte-identical arguments are
    equal by construction, so `assert keys[0] == keys[1] == keys[2]` could not
    fail — and the property it advertised (the real prefix does not leak into
    the key) was never exercised, because the real prefix never entered.

    The one-key half is still by construction and is left here only as the
    statement of intent; the load-bearing halves are the two below it, plus
    test_denoise_graph_key_discriminates_the_execution_profile, which is the
    assertion that catches a field going missing from the signature. Execution
    geometry belongs in the key; the real prefix length is represented by the
    mutable mask instead.
    """
    a = _make_action()
    a.hidden_size, a.num_layers = 1024, 36

    keys, masks = [], []
    with patch.object(act.rpu_backend.graph, "GraphSignature", side_effect=_fake_signature):
        for real_prefix_len in (591, 592, 591):
            # Same execution profile P=608/H=32; only the REAL prefix moves.
            masks.append(a._action_mask_cached(608, 32, real_prefix_len))
            keys.append(a._denoise_graph_signature(
                "wall_oss_action_denoise_loop", 32, 608, 10, 1))

    assert len(set(keys)) == 1

    # What makes one shared key safe is that the mask is NOT shared: it is the
    # dynamic content refreshed before REPLAY. If these ever became equal, the
    # single key would be serving two different attention masks.
    assert not torch.equal(masks[0], masks[1]), (
        "real prefix 591 and 592 produced the SAME mask, so the shared graph "
        "key is no longer covered by a per-real-prefix mask")
    # ...and A->B->A must hand back A's own tensor, because the C++ upload memo
    # is keyed on (pointer, seq_q, seq_k) (action.py:384-386).
    assert masks[0] is masks[2]
    assert masks[0].shape == masks[1].shape == (32, 640)


def test_denoise_graph_key_discriminates_the_execution_profile():
    """Every field that changes the executed graph MUST change the key.

    This is the direction that matters. A signature is dangerous when it is too
    COARSE — two different executions colliding on one key means the second
    replays the first one's graph and returns silently wrong numbers. Each case
    below drops exactly one field's variation past the signature; if any of
    them stops changing the key, someone removed that field from `shapes` /
    `dyn_dims` and the collision is live.
    """
    a = _make_action()
    a.hidden_size, a.num_layers = 1024, 36
    base_args = ("wall_oss_action_denoise_loop", 32, 608, 10, 1)

    def key(args=base_args, hidden=1024, layers=36):
        a.hidden_size, a.num_layers = hidden, layers
        with patch.object(act.rpu_backend.graph, "GraphSignature",
                          side_effect=_fake_signature):
            return a._denoise_graph_signature(*args)

    base = key()
    variants = {
        "op_id":       ("wall_oss_action_step", 32, 608, 10, 1),
        "H":           ("wall_oss_action_denoise_loop", 48, 608, 10, 1),
        "P":           ("wall_oss_action_denoise_loop", 32, 624, 10, 1),
        "num_steps":   ("wall_oss_action_denoise_loop", 32, 608, 4, 1),
        "row_constant": ("wall_oss_action_denoise_loop", 32, 608, 10, 0),
    }
    for field, args in variants.items():
        assert key(args) != base, (
            f"{field} no longer reaches the denoise graph key: two different "
            f"executions would share one graph and REPLAY each other's")

    assert key(hidden=1536) != base, "hidden_size dropped out of the key"
    assert key(layers=28) != base, "num_layers dropped out of the key"
    a._precision_mode = 1
    assert key() != base, "weight precision dropped out of the key"
    # The mode tail is variadic; a second mode dim must not be swallowed.
    assert key(base_args + (1,)) != base, "an extra mode dim dropped out of the key"


def test_action_mask_toggle_off_recomputes():
    a = _make_action(host_cache=False)
    m1 = a._action_mask_cached(192, 32, 180)
    m2 = a._action_mask_cached(192, 32, 180)
    assert m1 is not m2 and not a._action_mask_cache
    assert a._last_action_mask_ref is m2


# ───────────────────── get_rope_index cache (runtime.py) ────────────────────
def _stub_rope(o):
    calls = []

    def fake(input_ids, mm, image_grid_thw=None, attention_mask=None):
        calls.append(1)
        return torch.zeros(3, 1, input_ids.size(1), dtype=torch.long), None
    o._rope = types.SimpleNamespace(get_rope_index=fake)
    return calls


def test_rope_index_cache_hit_single_call():
    o = _make_vla()
    calls = _stub_rope(o)
    ids, grid = torch.zeros(1, 180, dtype=torch.long), torch.tensor([[1, 16, 16]])
    a = o._rope_index_cached(ids, grid, 180, 180)
    b = o._rope_index_cached(ids, grid, 180, 180)
    assert len(calls) == 1 and a[1] is b[1]


def test_rope_index_cache_diff_len_recomputes():
    o = _make_vla()
    calls = _stub_rope(o)
    grid = torch.tensor([[1, 16, 16]])
    o._rope_index_cached(torch.zeros(1, 180, dtype=torch.long), grid, 180, 192)
    o._rope_index_cached(torch.zeros(1, 164, dtype=torch.long), grid, 164, 176)
    assert len(calls) == 2


def test_rope_index_cache_same_len_grid_different_image_layout_recomputes():
    o = _make_vla()
    calls = _stub_rope(o)
    grid = torch.tensor([[1, 2, 2]])
    ids_a = torch.zeros(1, 12, dtype=torch.long)
    ids_b = torch.zeros(1, 12, dtype=torch.long)
    ids_a[0, 2:6] = rt.IMAGE_PAD_ID
    ids_b[0, 5:9] = rt.IMAGE_PAD_ID

    o._rope_index_cached(ids_a, grid, 12, 16)
    o._rope_index_cached(ids_b, grid, 12, 16)

    assert len(calls) == 2


def test_rope_index_cache_different_padded_len_recomputes():
    o = _make_vla()
    calls = _stub_rope(o)
    ids = torch.zeros(1, 12, dtype=torch.long)
    ids[0, 2:6] = rt.IMAGE_PAD_ID
    grid = torch.tensor([[1, 2, 2]])

    first = o._rope_index_cached(ids, grid, 12, 16)
    second = o._rope_index_cached(ids, grid, 12, 32)

    assert len(calls) == 2
    assert first[1].shape == (16, 3)
    assert second[1].shape == (32, 3)


def test_rope_index_cache_toggle_off_recomputes():
    o = _make_vla(host_cache=False)
    calls = _stub_rope(o)
    ids, grid = torch.zeros(1, 180, dtype=torch.long), torch.tensor([[1, 16, 16]])
    o._rope_index_cached(ids, grid, 180, 180)
    o._rope_index_cached(ids, grid, 180, 180)
    assert len(calls) == 2 and not o._rope_cache


# ─────────────── subsystem boundary: previous Denoise → next Vision ─────────
def test_predict_resets_temporary_spm_before_vision_entry():
    o = rt.WallOssVLA.__new__(rt.WallOssVLA)
    events = []
    o.camera_names = ("face_view",)
    o.dataset_key = "fixture"
    o.delta_action = False
    o._ondevice_embed = False
    o._proprioception_string = lambda *_: ""
    o._process_inputs = lambda *_: (
        torch.zeros(1, 8, dtype=torch.long),
        torch.zeros(4, 8),
        torch.tensor([[1, 2, 2]], dtype=torch.long),
    )
    o._validate_prepared_request = lambda **_: events.append("admission")
    o.action = types.SimpleNamespace(
        validate_execution_horizon=lambda _horizon: None
    )
    o.llm = types.SimpleNamespace(
        cache=types.SimpleNamespace(max_seq_len=16)
    )

    def vision_forward(*_):
        events.append("vision")
        raise RuntimeError("stop after Vision entry")

    o.vision = types.SimpleNamespace(forward=vision_forward)
    reset = MagicMock(side_effect=lambda: events.append("reset"))
    with patch.object(
        rt.torch.ops.rpu, "spm_alloc_reset_temporary", reset, create=True
    ):
        try:
            o.predict(
                [object()], "instruction",
                torch.zeros(1), torch.ones(1, dtype=torch.bool),
                torch.zeros(1, 2, 3), torch.ones(1, 2, 3),
            )
        except RuntimeError as exc:
            assert str(exc) == "stop after Vision entry"
        else:
            raise AssertionError("Vision sentinel must stop the host-only test")

    assert events == ["admission", "reset", "vision"]


def _assert_patch_embed_reset_before_capture(fn):
    source = inspect.getsource(fn)
    patch_pos = source.index("embed_rpu = self._patch_embed")
    reset_pos = source.index("torch.ops.rpu.spm_alloc_reset_temporary()")
    capture_pos = source.index("with self._graph_cache.capture(sig):")
    assert patch_pos < reset_pos < capture_pos


def test_vision_single_resets_after_patch_embed_before_capture():
    _assert_patch_embed_reset_before_capture(vis.WallOssVision._forward_one)


def test_vision_group_resets_after_patch_embed_before_capture():
    _assert_patch_embed_reset_before_capture(vis.WallOssVision._forward_group)


@pytest.mark.parametrize("host_cache", [False, True])
def test_vision_layout_uploads_compare_contents_and_preserve_old_owners(host_cache):
    v = _make_vision(host_cache=host_cache)
    uploads = []
    window = torch.tensor([1, 0])
    with _patch_rpu_tensor_to(uploads.append):
        first = v._patch_perm_ids(window, 8, 4)
        same = v._patch_perm_ids(window.clone(), 8, 4)
        window.copy_(torch.tensor([0, 1]))
        changed = v._patch_perm_ids(window, 8, 4)
        reverse = v._index_rpu("reverse", torch.arange(8).flip(0))
    assert torch.equal(first, torch.tensor([4, 5, 6, 7, 0, 1, 2, 3]))
    assert torch.equal(changed, torch.arange(8))
    assert torch.equal(reverse, torch.arange(8).flip(0))
    assert (same is first) is host_cache
    assert first.data_ptr() != changed.data_ptr()
    assert len(uploads) == (3 if host_cache else 4)


@pytest.mark.parametrize("host_cache", [False, True])
def test_vision_position_keepalive_refreshes_changes_and_replacement(monkeypatch, host_cache):
    v = _make_vision(host_cache=host_cache)
    v._keepalive = torch.full((8, 2), -1, dtype=torch.int16)
    copied = []
    original_copy = torch.Tensor.copy_

    def copy(dst, src, *args, **kwargs):
        if dst.data_ptr() == v._keepalive.data_ptr():
            copied.append(src.clone())
        return original_copy(dst, src, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    pos = torch.arange(8, dtype=torch.int16).reshape(4, 2)
    expected = pos.clone()
    v._refresh_position_keepalive(pos)
    v._refresh_position_keepalive(pos.clone())
    pos[0, 0] = 9
    v._refresh_position_keepalive(pos)
    assert v._keepalive[0, 0] == 9
    assert torch.equal(v._keepalive[4:], torch.full((4, 2), -1, dtype=torch.int16))
    v._refresh_position_keepalive(expected)
    assert torch.equal(v._keepalive[:4], expected)
    v._keepalive = torch.zeros(8, 2, dtype=torch.int16)
    v._refresh_position_keepalive(expected)
    assert torch.equal(v._keepalive[:4], expected)
    assert len(copied) == (4 if host_cache else 5)


@pytest.mark.parametrize("host_cache", [False, True])
def test_embedding_ids_cache_tracks_text_values_padding_and_inplace_changes(host_cache):
    vla = _make_vla(host_cache=host_cache)
    uploads = []
    ids = torch.tensor([11, rt.IMAGE_PAD_ID, 12, 0])
    with _patch_rpu_tensor_to(uploads.append):
        first = vla._embedding_ids_rpu(ids)
        same = vla._embedding_ids_rpu(ids.clone())
        ids[0] = 13
        changed = vla._embedding_ids_rpu(ids)
        reordered = vla._embedding_ids_rpu(ids.flip(0))
        padded = vla._embedding_ids_rpu(torch.cat([ids, torch.zeros(1, dtype=ids.dtype)]))
    assert (same is first) is host_cache
    assert torch.equal(first, torch.tensor([11, rt.IMAGE_PAD_ID, 12, 0]))
    assert torch.equal(changed, ids)
    assert torch.equal(reordered, ids.flip(0))
    assert padded.numel() == 5 and padded[-1] == 0
    assert len(uploads) == (4 if host_cache else 5)


@pytest.mark.parametrize("component", ["prefill", "action"])
def test_cached_mrope_owners_stay_versioned_inside_inference_mode(monkeypatch, component):
    owner = _make_llm() if component == "prefill" else _make_action()
    module = llm if component == "prefill" else act
    owner._head_dim, owner._rope_theta, owner._mrope_section = 128, 10000, (16, 24, 24)
    uploads = []

    # Exercise the real bake/upload and cache methods. The table arithmetic is
    # independent of the version/ownership boundary covered here.
    def tables(positions, **_):
        values = positions.sum(dim=1, keepdim=True).to(torch.float16)
        return values.expand(-1, 64).contiguous(), (-values).expand(-1, 64).contiguous()

    monkeypatch.setattr(module, "build_chunked_mrope_cos_sin", tables)

    def get(positions):
        if component == "prefill":
            return (owner._prefill_position_ids_rpu(positions, positions.size(0)),
                    *owner._prefill_cos_sin(positions, positions.size(0)))
        return (owner._action_position_ids_rpu(positions),
                *owner._partial_mrope_tables(positions))

    with _patch_rpu_tensor_to(uploads.append), torch.inference_mode():
        positions = _action_pos(80, 4)
        first = get(positions)
        original = tuple(value.clone() for value in first)
        same = get(positions.clone())
        positions[-1, -1] += 7
        changed = get(positions)
        for old, hit, new, snapshot in zip(first, same, changed, original):
            assert old is hit and old is not new
            assert not torch.is_inference(old) and not torch.is_inference(new)
            assert torch.equal(old, snapshot)
            assert not torch.equal(old, new)
    assert len(uploads) == 6
    # Native callers can still detect an explicit mutation through an alias.
    for value in first:
        version = value._version
        value.add_(1)
        assert value._version == version + 1


def test_vision_internal_parts_keep_camera_order_and_public_tensor_result():
    v = _make_vision()
    v._fused_assemble = False
    v._forward_group = lambda *_: pytest.fail("576-patch cameras must stay separate")

    def one(pixels, _grid):
        return pixels[:1, :1].expand(144, 16).clone()

    v._forward_one = one
    grid = torch.tensor([[1, 18, 32]] * 3)
    pixels = torch.repeat_interleave(torch.tensor([1., 2., 3.]), 576)[:, None]
    parts = v._forward_parts(pixels, grid)
    assert isinstance(parts, tuple) and len(parts) == 3
    assert [part[0, 0].item() for part in parts] == [1., 2., 3.]
    expected = torch.cat(parts)
    public = v.forward(pixels, grid)
    assert isinstance(public, torch.Tensor) and torch.equal(public, expected)
    changed = v._forward_parts(pixels + 10, grid)
    assert torch.equal(torch.cat(parts), expected)
    assert [part[0, 0].item() for part in changed] == [11., 12., 13.]
    assert all(left.data_ptr() != right.data_ptr() for left, right in zip(parts, changed))


def test_pair_camera_parts_reach_device_assembler_without_intermediate_concat(monkeypatch):
    monkeypatch.setattr(rt, "IMAGE_PAD_ID", 13)
    monkeypatch.setattr(torch.rpu, "get_debug_export", lambda: False, raising=False)
    vision = _make_vision()
    vision._pair_252 = True
    vision._fused_assemble = False
    vision._batch_max_seq = 768
    produced = []

    def encode(pixels, _grid):
        out = pixels.reshape(-1, 4, 1).sum(1).expand(-1, 16).contiguous().to("rpu")
        produced.append(out)
        return out

    vision._forward_pair_252 = vision._forward_one = encode
    owner = _make_vla()
    owner.vision = vision
    owner._embed_w_rpu = torch.arange(32 * 16, dtype=torch.float16).reshape(32, 16)
    ids = torch.tensor([[11] + [13] * 81 + [12] + [13] * 81 + [14] + [13] * 81])
    grid = torch.tensor([[1, 18, 18]] * 3)
    pixels = torch.arange(972, dtype=torch.float16).reshape(-1, 1)
    original_cat = torch.cat
    consumed = []

    def assemble(vocab, token_ids, pieces, runs):
        assert runs.tolist() == [[1, 81], [83, 81], [165, 81]]
        assert [part.size(0) for part in pieces] == [81, 81, 81]
        assert [part.untyped_storage()._cdata for part in pieces] == [
            produced[-2].untyped_storage()._cdata,
            produced[-2].untyped_storage()._cdata,
            produced[-1].untyped_storage()._cdata,
        ]
        out = vocab[token_ids.long()].clone()
        for part, (start, count) in zip(pieces, runs.tolist()):
            out[start:start + count].copy_(part)
        consumed.append(out)
        return out

    monkeypatch.setattr(torch.ops.rpu, "assemble_inputs_embeds_parts", assemble, raising=False)
    outputs = []
    with _patch_rpu_tensor_to(lambda _: None):
        for offset in (0, 1, 0):
            with monkeypatch.context() as stage:
                stage.setattr(torch, "cat", lambda *_a, **_k: pytest.fail("pair parts were concatenated"))
                parts = vision._forward_parts(pixels + offset, grid)
            assert isinstance(parts, tuple) and tuple(map(len, parts)) == (162, 81)
            out = owner._assemble_inputs_embeds_ondevice(ids, parts, 256)
            expected = (pixels + offset).reshape(-1, 4, 1).sum(1).expand(-1, 16)
            assert torch.equal(out[0, :ids.size(1)][ids[0] == 13], expected)
            outputs.append(out)
        public = vision.forward(pixels, grid)
    assert torch.equal(public, original_cat((produced[-2], produced[-1])))
    assert torch.equal(outputs[0], outputs[2]) and not torch.equal(outputs[0], outputs[1])
    assert outputs[0].data_ptr() != outputs[2].data_ptr() and len(consumed) == 3


@pytest.mark.parametrize("partition", [(144, 144, 144), (200, 232), (432,)])
@pytest.mark.parametrize("joined_runs", [False, True])
def test_device_assembly_maps_subgroup_parts_to_image_runs_and_retains_outputs(
        monkeypatch, partition, joined_runs):
    monkeypatch.setattr(rt, "IMAGE_PAD_ID", 13)
    owner = _make_vla()
    owner._embed_w_rpu = torch.arange(32 * 16, dtype=torch.float16).reshape(32, 16)
    owner.vision = types.SimpleNamespace(_pending_rev=None)
    ids = (torch.tensor([[11] + [13] * 432 + [12]]) if joined_runs else
           torch.tensor([[11] + [13] * 144 + [12] + [13] * 144 + [14] + [13] * 144]))
    padded = ((ids.size(1) + 15) // 16) * 16
    merged = torch.arange(432, dtype=torch.float16)[:, None].expand(432, 16).contiguous()
    dispatches = []

    def assemble(vocab, token_ids, pieces, runs):
        assert runs.device.type == "cpu" and runs.dtype == torch.long
        assert runs.shape == (len(pieces), 2)
        assert sum(part.size(0) for part in pieces) == 432
        output = vocab[token_ids.long()].clone()
        for part, (start, length) in zip(pieces, runs.tolist()):
            assert part.is_contiguous() and part.shape == (length, 16)
            output[start:start + length].copy_(part)
        dispatches.append(runs.clone())
        return output

    monkeypatch.setattr(torch.ops.rpu, "assemble_inputs_embeds_parts", assemble, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "assemble_inputs_embeds",
                        lambda vocab, ids, part, start: assemble(
                            vocab, ids, [part], torch.tensor([[start, part.size(0)]])),
                        raising=False)
    outputs = []
    with _patch_rpu_tensor_to(lambda _out: None):
        for offset in (0, 10, 0):
            current_ids = ids.clone()
            current_ids[0, 0] += offset
            current_merged = merged + offset
            parts = tuple(current_merged.split(partition))
            output = owner._assemble_inputs_embeds_ondevice(current_ids, parts, padded)
            padded_ids = torch.nn.functional.pad(current_ids, (0, padded - ids.size(1)))
            expected = owner._embed_w_rpu[padded_ids]
            expected[0, :ids.size(1)][current_ids[0] == 13] = current_merged
            assert torch.equal(output, expected)
            outputs.append(output)
    assert torch.equal(outputs[0], outputs[2])
    assert not torch.equal(outputs[0], outputs[1])
    assert outputs[0].data_ptr() != outputs[2].data_ptr()
    # A boundary inside one Vision owner and a boundary spanning two owners
    # must both split into exactly the rows consumed by the destination runs.
    if partition == (200, 232) and not joined_runs:
        assert dispatches[0][:, 1].tolist() == [144, 56, 88, 144]


def test_device_assembly_rejects_wrong_merged_row_count_before_dispatch(monkeypatch):
    monkeypatch.setattr(rt, "IMAGE_PAD_ID", 13)
    owner = _make_vla()
    owner._embed_w_rpu = torch.zeros(32, 16, dtype=torch.float16)
    owner.vision = types.SimpleNamespace(_pending_rev=None)
    dispatch = MagicMock()
    monkeypatch.setattr(torch.ops.rpu, "assemble_inputs_embeds_parts", dispatch, raising=False)
    with _patch_rpu_tensor_to(lambda _out: None):
        with pytest.raises(RuntimeError, match="image-pad tokens 2 != merged rows 3"):
            owner._assemble_inputs_embeds_ondevice(
                torch.tensor([[11, 13, 12, 13]]), (torch.zeros(3, 16),), 16)
    dispatch.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
@pytest.mark.parametrize("host_cache", [False, True])
def test_camera_patch_upload_shares_one_fresh_owner_and_refreshes_pixels(dtype, host_cache):
    v = _make_vision(host_cache=host_cache)
    v._fused_assemble = False
    v._batch_vision = False
    v._device_patch_embed = True
    v._pe_k = 16
    v._patch_w_rpu = v._patch_w_fold_rpu = torch.eye(16, dtype=torch.float16)
    inputs = []

    def one(pixels, grid, prepared_pixels=None):
        assert prepared_pixels is not None
        inputs.append(prepared_pixels)
        return v._patch_embed(pixels, None, pixels.size(0), 4, prepared_pixels).squeeze(0)

    v._forward_one = one
    grid = torch.tensor([[1, 18, 32], [1, 16, 16], [1, 8, 16]])
    pixels = (torch.arange(960 * 3).remainder(127).reshape(960, 3).to(dtype))
    if dtype == torch.float32:
        pixels = pixels / 3.7
    uploads, results = [], []
    with _patch_rpu_tensor_to(uploads.append):
        for offset in (0, 1, 0):
            changed = pixels + offset
            parts = v._forward_parts(changed, grid)
            expected = torch.nn.functional.pad(changed.half(), (0, 13))
            if dtype == torch.uint8:
                expected[:, 3] = 1
            output = torch.cat(parts)
            assert torch.equal(output, expected)
            results.append(output)
    assert len(uploads) == 3
    for request in range(3):
        assert all(part.untyped_storage().data_ptr() == uploads[request].data_ptr()
                   for part in inputs[request * 3:(request + 1) * 3])
    assert uploads[0].data_ptr() != uploads[1].data_ptr()
    assert torch.equal(results[0], results[2])
    assert not torch.equal(results[0], results[1])


@pytest.mark.parametrize("route", ["cpu_gemm", "missing_weight", "unaligned", "rpu_pixels"])
def test_camera_patch_upload_preserves_original_fallbacks(route):
    v = _make_vision()
    v._fused_assemble = False
    v._batch_vision = False
    v._pe_k = 16
    v._device_patch_embed = route != "cpu_gemm"
    v._patch_w_rpu = None if route == "missing_weight" else torch.eye(16, dtype=torch.float16)
    grid = (torch.tensor([[1, 2, 2], [1, 2, 4]]) if route == "unaligned"
            else torch.tensor([[1, 18, 32], [1, 16, 16]]))
    pixels = torch.ones(int(grid.prod(-1).sum()), 3)
    if route == "rpu_pixels":
        class DevicePixels(torch.Tensor):
            @property
            def device(self):
                return types.SimpleNamespace(type="rpu")
        pixels = pixels.as_subclass(DevicePixels)
    calls = []

    def one(pixels, _grid, prepared_pixels=None):
        assert prepared_pixels is None
        calls.append(pixels.size(0))
        return torch.zeros(pixels.size(0) // 4, 16)

    v._forward_one = one
    v._prepare_patch_pixels = lambda _: pytest.fail("fallback uploaded a shared patch slab")
    parts = v._forward_parts(pixels, grid)
    assert len(parts) == len(grid) and calls == grid.prod(-1).tolist()


def test_rpu_cat_forces_host_ddr_boundary_flushes():
    """The multi-subgroup Vision path cats device tensors through host memcpy."""
    path = os.path.join(_REPO, "src/ops/rpu_concat.cpp")
    with open(path, encoding="utf-8") as source_file:
        source = source_file.read()
    body = source[
        source.index("at::Tensor rpu_cat("):
        source.index("at::Tensor& rpu_cat_out(")
    ]
    assert body.count("rpu_ddr_flush_force_sized(") == 2
    assert "rpu_ddr_flush(" not in body
    assert body.count("c10::DeviceType::PrivateUse1") >= 2


@pytest.mark.parametrize("side,fold", [(224, True), (256, True), (256, False)])
def test_batched_image_preproc_matches_processor_resize(side, fold):
    import numpy as np
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast

    ip = Qwen2VLImageProcessorFast()
    o = rt.WallOssVLA.__new__(rt.WallOssVLA)
    o._processor = types.SimpleNamespace(image_processor=ip)
    o._pe_norm_fold = fold
    pixels = np.random.default_rng(8).integers(0, 256, (side, side, 3), dtype=np.uint8)
    images = [Image.fromarray(pixels), Image.fromarray(255 - pixels),
              Image.fromarray(np.flip(pixels, axis=1).copy())]
    original = [np.asarray(im).copy() for im in images]
    pv, grid = o._fast_image_preproc(images)
    ref = ip(images=images, return_tensors="pt")
    assert torch.equal(grid, ref["image_grid_thw"])
    assert grid.tolist() == [[1, 16 if side == 224 else 18, 16 if side == 224 else 18]] * 3
    if fold:
        raw = ip(images=images, do_rescale=False, do_normalize=False, return_tensors="pt")
        # Includes exact tensor-bicubic uint8 rounding, camera order and patch layout.
        assert torch.equal(pv, raw["pixel_values"])
        pv = o._reconstruct_norm(pv)
    torch.testing.assert_close(pv, ref["pixel_values"], atol=1e-6, rtol=1e-6)
    assert all(np.array_equal(np.asarray(im), before) for im, before in zip(images, original))


def test_batched_image_resize_keeps_slow_processor_and_mixed_shape_fallback():
    import numpy as np
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast

    image = Image.fromarray(np.full((256, 256, 3), 127, dtype=np.uint8))
    o = rt.WallOssVLA.__new__(rt.WallOssVLA)
    o._pe_norm_fold = True
    o._fast_imgproc, o._fast_imgproc_ok = True, True
    for ip in [Qwen2VLImageProcessor(), Qwen2VLImageProcessorFast(do_resize=False)]:
        o._processor = types.SimpleNamespace(image_processor=ip)
        assert o._fast_image_preproc([image] * 3) is None
    ip = Qwen2VLImageProcessor()
    o._processor = types.SimpleNamespace(image_processor=ip)
    pv, grid = o._image_preproc([image] * 3)
    ref = ip(images=[image] * 3, return_tensors="pt")
    assert torch.equal(pv, ref["pixel_values"])
    assert torch.equal(grid, ref["image_grid_thw"])
    o._processor = types.SimpleNamespace(image_processor=Qwen2VLImageProcessorFast())
    assert o._fast_image_preproc([image, image.resize((224, 224))]) is None
    assert o._fast_image_preproc([image.resize((280, 256))] * 3) is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")
