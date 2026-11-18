"""CPU-only contracts for the Qwen3.5 Python adapter/cache wiring."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tomllib
import types
from unittest.mock import patch

import pytest
import torch


_PKG = Path(__file__).resolve().parents[1] / "python" / "rpu_backend"
_ADAPTER = _PKG / "adapters" / "qwen3_5" / "__init__.py"
_TEXT = _PKG / "adapters" / "qwen3_5" / "text.py"
_DECODER = _PKG / "runtime" / "decoder.py"
_CACHE = _PKG / "api" / "qwen3_5_cache.py"
_API = _PKG / "api" / "__init__.py"
_MODEL_CPP = Path(__file__).resolve().parents[1] / "src" / "fused" / "rpu_qwen3_5_model.cpp"
_MODEL_H = Path(__file__).resolve().parents[1] / "src" / "fused" / "rpu_qwen3_5_model.h"
_VISION_CPP = (
    Path(__file__).resolve().parents[1]
    / "src" / "fused" / "rpu_qwen3_5_vision_model.cpp"
)
_PARTIAL_CPP = (
    Path(__file__).resolve().parents[1] / "src" / "ops" / "rpu_partial_mrope.cpp"
)
_ELTWISE_CPP = (
    Path(__file__).resolve().parents[1] / "src" / "ops" / "rpu_eltwise_binary.cpp"
)

_PLANNER_SPEC = importlib.util.spec_from_file_location(
    "qwen3_5_execution_planner_under_test",
    _PKG / "runtime" / "execution_planner.py",
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


class _FakeRPUCache:
    def __init__(self, num_layers, batch_size, max_seq_len, num_kv_heads,
                 head_dim, device="cpu", dtype=torch.float16, attn_tp=8,
                 *, _kv_layer_mask=None):
        self.allocated_max_seq_len = max_seq_len
        self.max_seq_len = max_seq_len
        self.position = self._seen_tokens = 0
        mask = _kv_layer_mask or (True,) * num_layers
        self.k_caches = [torch.ones(1 if active else 0) for active in mask]
        self.v_caches = [torch.ones(1 if active else 0) for active in mask]

    def reset(self):
        self.position = self._seen_tokens = 0
        for tensor in self.k_caches + self.v_caches:
            tensor.zero_()


def _load_cache_module():
    from rpu_backend.api.cache import _zero_whole_rpu_tensors

    root = types.ModuleType("rpu_backend")
    root.__path__ = [str(_PKG)]
    api = types.ModuleType("rpu_backend.api")
    api.__path__ = [str(_PKG / "api")]
    base = types.ModuleType("rpu_backend.api.cache")
    base.RPUCache = _FakeRPUCache
    base._zero_whole_rpu_tensors = _zero_whole_rpu_tensors
    stubs = {
        "rpu_backend": root,
        "rpu_backend.api": api,
        "rpu_backend.api.cache": base,
    }
    spec = importlib.util.spec_from_file_location("qwen3_5_cache_under_test", _CACHE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


def test_qwen3_5_cache_aligns_physical_capacity_and_resets_gdn_state():
    cls = _load_cache_module().Qwen3_5Cache
    cache = cls(
        num_layers=2, batch_size=1, max_seq_len=65,
        num_kv_heads=8, head_dim=16,
        layer_types=["full_attention", "linear_attention"],
        num_v_heads=1, key_head_dim=1, value_head_dim=1,
        conv_dim=8, conv_kernel_dim=2, device="cpu",
    )
    assert cache.allocated_max_seq_len == 192
    assert cache.max_seq_len == 65

    # Route-B reserves 63 mandatory alignment rows plus the complete 64-row
    # optional search window, then rounds physical storage to 16 rows.
    cache_100 = cls(
        num_layers=1, batch_size=1, max_seq_len=100,
        num_kv_heads=8, head_dim=16, layer_types=["full_attention"],
        num_v_heads=1, key_head_dim=1, value_head_dim=1,
        conv_dim=8, conv_kernel_dim=2, device="cpu",
    )
    assert cache_100.allocated_max_seq_len == 240

    cache.position = cache._seen_tokens = 17
    cache.recurrent_states[1].fill_(1)
    cache.conv_states[1].fill_(1)
    with pytest.raises(ValueError, match="non-zero position"):
        cache.reset_to_position(8)
    cache.reset_to_position(0)
    assert cache.position == 0
    assert not cache.recurrent_states[1].any()
    assert not cache.conv_states[1].any()


def test_qwen3_5_adapter_lifecycle_static_contract():
    source = _ADAPTER.read_text()
    compile(source, str(_ADAPTER), "exec")
    assert "rpu_aware_to.__rpu_wrapped__ = True" in source
    assert source.index("_claim_live_instance(model)") < source.index("model.half()")
    assert 'inner.to("rpu")' in source
    assert 'type(model).to(model, "rpu")' not in source
    assert "model._rpu_swizzled = True" in source
    # Explicit retirement and a rejected cold, pre-mutation install may release
    # the claim. Real live-owner tests cover the failed-mutation exclusion.
    tree = ast.parse(source)
    release_owners = [
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and "_release_live_instance" in ast.get_source_segment(source, node)
    ]
    assert release_owners == ["_close_qwen3_5_model", "Qwen3_5Adapter"]
    assert "_cleanup_failed_text_install" in source
    assert "output_hidden_states=None" in source
    assert "does not support output_hidden_states=True" in source
    assert "Qwen3_5CausalLMOutputWithPast" in source
    assert "hidden_states=(out,)" not in source


def test_qwen3_5_text_handle_and_graph_static_contract():
    source = _TEXT.read_text()
    compile(source, str(_TEXT), "exec")
    assert "pending_state.retirement = _InstalledNativeResource(" in source
    assert "raw_graphs=(prefill_graph,)" in source
    assert "torch.ops.rpu.qwen3_5_destroy(handle)" in source
    assert "handle_finalizer=handle_finalizer" in source
    assert "prefill_rope_keepalive" not in source
    assert "cache.position + real_len > cache.max_seq_len" in source
    decode_graph = source.split('op_id="rpu_qwen3_5"', 1)[1]
    assert 'num_layers, int(getattr(st, "execution_generation", 0))' in decode_graph
    assert "start_layer" not in source
    assert "end_layer" not in source
    assert "qwen3_5_resolve_prefill_stage_domain" in source
    assert "st.handle, n, real_len" in source
    decode_signature = source.split("def _qwen3_5_decode_signature(", 1)[1]
    decode_signature = decode_signature.split("\ndef ", 1)[0]
    assert 'op_id="rpu_qwen3_5"' in decode_signature
    assert "*plan.graph_key_words()" in decode_signature
    assert (
        "graph_cache.capture(_qwen3_5_decode_signature(st, decode_plan_result))"
        in source
    )
    install = source.split("def _install_qwen3_5_text_for_rpu_impl(", 1)[1]
    unit_norm = install.split("    for i, layer in enumerate(inner.layers):", 1)[0]
    assert "unit_norm = None" in unit_norm
    assert unit_norm.index("if is_adaptive:") < unit_norm.index(".new_ones(")


def test_qwen3_5_cpp_weight_and_rope_input_contracts():
    source = _MODEL_CPP.read_text()
    set_weights = source[source.index("void Qwen3_5Model::set_weights("):]
    validation_end = set_weights.index("has_qk_norm_")
    for name in (
        "q_w_list", "input_norm_list", "gdn_in_z_list", "gdn_q_list",
        "gdn_a_bg_list",
    ):
        assert f'check_list({name}, "{name}")' in set_weights[:validation_end]
    assert "keep_256b_aligned_rpu_copy(cos" in set_weights
    assert "keep_256b_aligned_rpu_copy(sin" in set_weights
    assert set_weights.index("cos.sizes() == sin.sizes()") < validation_end


def test_qwen3_5_physical_manifest_is_the_kv_attention_authority():
    text = _MODEL_CPP.read_text()
    text_header = _MODEL_H.read_text()
    vision = _VISION_CPP.read_text()

    for source in (text, vision):
        assert "physical_manifest_for_candidate(" in source
        assert "FmbPhysicalManifestState::COMPLETE" in source
        assert "rpu_launch_insert_kvcache_spm_unified_with_plan(" in source
        assert "rpu_launch_insert_kvcache_spm_unified(" not in source

    assert "KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION" in text
    assert "rpu_rebase_kvinsert_segment_plan_position(" in text
    assert "/*invocation=*/chunk.idx" in text
    assert "const int64_t invocation = action_mode_" in text
    assert "? (action_prefix_lens_[layer_idx] > 0 ? 1 : 0)" in text
    assert "QWEN35_TEXT_LINEAR_SITE = 4115846627077743284LL" in text
    assert text.count("QWEN35_TEXT_LINEAR_SITE") >= 3
    assert "linear_site_id(" not in text
    assert "LinearSiteRole" not in text_header
    assert "chunk_idx * 8 + static_cast<int64_t>(selector)" in text_header
    assert "bool needs_auto_tile = false" in text
    assert "bool needs_gemv = false" in text
    assert "linear_invocation(chunk_idx, selector)" in text
    assert "emit_mlp_pipeline(" not in text
    for source in (text, vision):
        assert "append_fmb_shared_runtime_routes(" in source
        assert "FMB_SHARED_LAYER_INPUT_DMA" in source
    assert "return fmb_ring_all_reduce_route_selector(rows, cols, num_cores());" in text
    assert "Qwen35TextAllReduceRoute::PREPARE_RING_INPUT" in text
    assert "std::sort(" in text

    routed_text_sites = (
        "QWEN35_TEXT_CORE_TOPOLOGY_SITE",
        "QWEN35_TEXT_GDN_PREPARE_ALL_REDUCE_SITE",
        "QWEN35_TEXT_FULL_PREPARE_ALL_REDUCE_SITE",
        "QWEN35_TEXT_FULL_ALL_REDUCE_SITE",
        "QWEN35_TEXT_GDN_ALL_REDUCE_SITE",
        "QWEN35_TEXT_MLP_ALL_REDUCE_SITE",
        "QWEN35_TEXT_PARTIAL_ROPE_1D_SITE",
        "QWEN35_TEXT_PARTIAL_MROPE_SITE",
        "QWEN35_TEXT_Q_NORM_ROPE_SITE",
        "QWEN35_TEXT_K_NORM_ROPE_SITE",
        "QWEN35_TEXT_Q_ROPE_SITE",
        "QWEN35_TEXT_K_ROPE_SITE",
        "QWEN35_TEXT_ADAPTIVE_MOD_DMA_SITE",
        "QWEN35_TEXT_ACTION_INPUT_DMA_SITE",
        "QWEN35_TEXT_ACTION_RTC_PREFIX_DMA_SITE",
        "QWEN35_TEXT_ACTION_TRACE_PRE_DMA_SITE",
        "QWEN35_TEXT_ACTION_TRACE_EMBED_DMA_SITE",
        "QWEN35_TEXT_ACTION_TRACE_HIDDEN_DMA_SITE",
        "QWEN35_TEXT_ACTION_KEEP_MASK_DMA_SITE",
        "QWEN35_TEXT_ACTION_TRACE_VELOCITY_DMA_SITE",
        "QWEN35_TEXT_ACTION_TRACE_POST_DMA_SITE",
        "QWEN35_TEXT_ACTION_OUTPUT_DMA_SITE",
        "QWEN35_TEXT_ACTION_INPUT_BIAS_PRELOAD_SITE",
        "QWEN35_TEXT_ACTION_OUTPUT_BIAS_PRELOAD_SITE",
        "QWEN35_TEXT_Q_NORM_PRELOAD_SITE",
        "QWEN35_TEXT_K_NORM_PRELOAD_SITE",
        "QWEN35_TEXT_ACTION_SCHEDULE_SITE",
        "QWEN35_TEXT_QK_NORM_SCHEDULE_SITE",
        "QWEN35_TEXT_MLP_SILU_SITE",
        "QWEN35_TEXT_GDN_STATE_LOAD_DMA_SITE",
        "QWEN35_TEXT_GDN_CONV_DECODE_LOAD_DMA_SITE",
        "QWEN35_TEXT_GDN_CONV_PREFILL_LOAD_DMA_SITE",
        "QWEN35_TEXT_GDN_CONV_PREFILL_STORE_DMA_SITE",
        "QWEN35_TEXT_GDN_STATE_PREFILL_LOAD_DMA_SITE",
        "QWEN35_TEXT_GDN_STATE_PREFILL_STORE_DMA_SITE",
        "QWEN35_TEXT_GDN_STATE_DECODE_STORE_DMA_SITE",
        "QWEN35_TEXT_GDN_CONV_DECODE_STORE_DMA_SITE",
    )
    for site in routed_text_sites:
        # Definition + descriptor emission + production consume.
        assert text.count(site) >= 3, site

    for site_id in (
        "1711359184452140201LL",
        "7862402590903398933LL",
        "97797974637846192LL",
        "3433090853780504513LL",
        "6909027152890455551LL",
        "3308017190679002622LL",
    ):
        assert site_id in vision
    assert "layout.physical_manifest_fingerprint" in vision
    full_domain = vision.split(
        "Qwen3_5VisionModel::resolve_prefill_domain(", 1
    )[1].split("Qwen3_5VisionModel::resolve_prefill_plan(", 1)[0]
    assert "resolve_prefill_stage_domain_for_shape(" in full_domain
    assert "for (const auto& candidate : physical_candidates)" in full_domain
    assert "const auto& physical_manifest = candidate.physical_manifest;" in full_domain
    assert "encode_fmb_prefill_stage_candidate(candidate)" in full_domain
    assert "finish_kvinsert_cost_domain_oracle(returned_candidates)" in full_domain
    assert "return v3::fmb_ring_all_reduce_route_selector(rows, cols, cores);" in vision
    assert "Qwen35VisionAllReduceRoute::PREPARE_RING_INPUT" in vision
    assert "std::sort(" in vision

    routed_vision_sites = (
        "QWEN35_VISION_STEP0_COMPACT_PIXEL_DMA_SITE",
        "QWEN35_VISION_STEP0_PIXEL_DMA_SITE",
        "QWEN35_VISION_STEP0_LINEAR_SITE",
        "QWEN35_VISION_STEP0_POS_SCATTER_DMA_SITE",
        "QWEN35_VISION_STEP0_POS_BROADCAST_DMA_SITE",
        "QWEN35_VISION_MERGER_FC1_SITE",
        "QWEN35_VISION_MERGER_FC2_SITE",
        "QWEN35_VISION_MERGER_ALL_REDUCE_SITE",
        "QWEN35_VISION_MERGER_OUTPUT_DMA_SITE",
        "QWEN35_VISION_TEMPORAL_Q_SITE",
        "QWEN35_VISION_TEMPORAL_K_SITE",
        "QWEN35_VISION_TEMPORAL_V_SITE",
        "QWEN35_VISION_SPATIAL_Q_SITE",
        "QWEN35_VISION_SPATIAL_K_SITE",
        "QWEN35_VISION_SPATIAL_V_SITE",
        "QWEN35_VISION_SPATIAL_Q_ROPE_SITE",
        "QWEN35_VISION_SPATIAL_K_ROPE_SITE",
        "QWEN35_VISION_SPATIAL_O_SITE",
        "QWEN35_VISION_SPATIAL_ATTN_ALL_REDUCE_SITE",
        "QWEN35_VISION_SPATIAL_FC1_SITE",
        "QWEN35_VISION_SPATIAL_FC2_SITE",
        "QWEN35_VISION_SPATIAL_MLP_ALL_REDUCE_SITE",
        "QWEN35_VISION_Q_SITE",
        "QWEN35_VISION_K_SITE",
        "QWEN35_VISION_V_SITE",
        "QWEN35_VISION_Q_ROPE_SITE",
        "QWEN35_VISION_K_ROPE_SITE",
        "QWEN35_VISION_PREPARE_ALL_REDUCE_SITE",
        "QWEN35_VISION_O_SITE",
        "QWEN35_VISION_ATTN_ALL_REDUCE_SITE",
        "QWEN35_VISION_FC1_SITE",
        "QWEN35_VISION_FC2_SITE",
        "QWEN35_VISION_MLP_ALL_REDUCE_SITE",
    )
    for site in routed_vision_sites:
        # Definition + descriptor emission + production consume.
        assert vision.count(site) >= 3, site

def test_qwen3_5_registry_matches_every_active_descriptor_route_site():
    root = Path(__file__).resolve().parents[1]
    owners_path = root / "manifests/planner_owners.v1.json"
    if not owners_path.is_file():
        pytest.skip("manifests/planner_owners.v1.json is internal-only")
    registry = json.loads(owners_path.read_text())
    sources = {
        "fmb-qwen35-text": _MODEL_CPP.read_text() + "\n" + _MODEL_H.read_text(),
        "fmb-qwen35-vision": _VISION_CPP.read_text(),
    }
    for owner, source in sources.items():
        all_routes = [
            site for site in registry["sites"]
            if site["owner"] == owner
            and site["control_kind"] == "descriptor_route"
        ]
        routes = [site for site in all_routes if site["state"] != "PARTIAL"]
        # A typed launcher without a planner descriptor stays a PARTIAL gap;
        # it must not claim a site id the source declares as bound.
        unbound = [site for site in all_routes if site["state"] == "PARTIAL"]
        for site in unbound:
            assert site["reason_code"] == "STATIC_ROUTE_BINDING_UNPROVEN", site["id"]
            assert site["hw_status"] == "NOT_RUN", site["id"]
            assert [e["kind"] for e in site["evidence"]] == ["gap"], site["id"]
        declarations = re.findall(
            r"(?:static\s+)?constexpr\s+int64_t\s+(\w*[Ss][Ii][Tt][Ee])"
            r"\s*=\s*(\d+)LL",
            source,
        )
        assert len(declarations) == len({name for name, _ in declarations})
        assert len(declarations) == len({site_id for _, site_id in declarations})
        # The text NVFP4 table tag only salts the layout hash and cost identity;
        # it is never a manifest route key, so it has no registry route.
        identity_salts = {"kTextNvfp4TablesSite"} if owner == "fmb-qwen35-text" else set()
        for name in identity_salts:
            assert source.count(name) == 1 + source.count(f"layout_mix(hash, {name})") + \
                source.count(f"identity.push_back({name})"), name
        declared_by_id = {site_id: name for name, site_id in declarations
                          if name not in identity_salts}
        capability_routes = [
            route for route in registry.get("capability_routes", ())
            if route["owner"] == owner
        ]
        capability_ids = {
            str(route["route_key"]["site_id"])
            for route in capability_routes
        }
        route_ids = {str(route["route_key"]["site_id"]) for route in routes}
        assert len(capability_routes) == len(capability_ids)
        canonical_routes = {route["id"]: route for route in routes
                            if "route_alias_of" not in route}
        assert len(canonical_routes) == len(route_ids)
        for route in routes:
            if "route_alias_of" in route:
                canonical = canonical_routes[route["route_alias_of"]]
                assert route["route_key"] == canonical["route_key"]
                assert route["source"] != canonical["source"]
                assert all(route[field] == canonical[field] for field in
                           ("owner", "component", "stage", "family", "state"))
        assert capability_ids <= declared_by_id.keys()
        assert route_ids == declared_by_id.keys() - capability_ids
        assert not {
            str(site["route_key"]["site_id"]) for site in unbound
        } & declared_by_id.keys()
        for route in routes:
            site_id = str(route["route_key"]["site_id"])
            site_name = declared_by_id[site_id]
            # Stable declaration + conditional manifest + launcher-side receipt.
            assert source.count(site_name) >= 3, route["id"]

    assert "37266457839629843" not in json.dumps(registry)


def test_qwen3_5_vision_and_kernel_fail_fast_contracts():
    vision = _VISION_CPP.read_text()
    forward = vision[vision.index("at::Tensor Qwen3_5VisionModel::forward("):]
    state_write = forward.index("current_num_patches_ = num_patches_in")
    assert "input.scalar_type() == at::kHalf" in forward[:state_write]
    assert "num_patches_in % QWEN3_5_SPATIAL_MERGE_UNIT == 0" in forward[:state_write]

    partial = _PARTIAL_CPP.read_text()
    assert partial.count("& 0xFF) == 0") == 4
    assert partial.index("cos_raw_addr & 0xFF") < partial.index(
        "const uint64_t cos_addr = cos_raw_addr >> 8")

    eltwise = _ELTWISE_CPP.read_text()
    bx = eltwise[eltwise.index(
        "void rpu_launch_eltwise_binary_Bx1xC_BxNxC_spm_kernel(") :]
    assert "c10::Half imm3 = 1.0, imm4 = 1.0" in bx
    assert "set_regs(59, imm3.x)" in bx
    assert "set_regs(60, imm4.x)" in bx


def test_qwen3_5_text_prefill_delta_and_public_cache_export():
    adapter_source = _ADAPTER.read_text()
    assert "pixel_values is None and cache is not None and int(cache.position) == 0" in adapter_source
    assert "qwen3_5_set_mrope_position_delta(st.handle, 0)" in adapter_source
    api_source = _API.read_text()
    assert "from .qwen3_5_cache import Qwen3_5Cache" in api_source
    assert '"Qwen3_5Cache"' in api_source


def _load_function(path, name, namespace=None):
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    namespace = dict(namespace or {})
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace[name]


def _load_text_function(name):
    namespace = {}
    if name == "_plan_prefill_execution":
        namespace["plan_bounded_prefill_execution"] = _load_function(
            _DECODER,
            "plan_bounded_prefill_execution",
            {
                "os": os,
                "json": json,
                "_LOG": types.SimpleNamespace(
                    info=lambda *_args, **_kwargs: None
                ),
                "PlannerRejectError": _PLANNER.PlannerRejectError,
                "native_chunk_reject": _PLANNER.native_chunk_reject,
                "plan_prefill": _PLANNER.plan_prefill,
            },
        )
    return _load_function(_TEXT, name, namespace)


def test_qwen3_5_prefill_execution_plan_uses_bounded_padding():
    plan = _load_text_function("_plan_prefill_execution")
    resolve_256 = lambda _: 256
    domain_256 = _stage_domain(resolve_256)
    # Equal chunk count does not justify extra compute just to erase the tail.
    assert plan(
        704, 768, 64, resolve_stage_domain=domain_256, queue_owner_id=17
    ) == (704, 256)
    assert plan(
        704, 768, 0, resolve_stage_domain=domain_256, queue_owner_id=17
    ) == (704, 256)
    assert plan(
        704, 704, 64, resolve_stage_domain=domain_256, queue_owner_id=17
    ) == (704, 256)

    # Padding is worthwhile when it unlocks a strictly smaller chunk count.
    resolve_drop = lambda n: {832: 256, 896: 448}[n]
    assert plan(
        832, 896, 64, resolve_stage_domain=_stage_domain(resolve_drop),
        queue_owner_id=17,
    ) == (896, 448)

    # E=192 would create a third cs=64 chunk containing no real rows.
    assert plan(
        65, 192, 64, resolve_stage_domain=_stage_domain(lambda _: 64),
        queue_owner_id=17,
    ) == (128, 64)


def test_qwen3_5_prefill_plan_skips_rejected_candidates():
    plan = _load_text_function("_plan_prefill_execution")

    def resolve(execution_len):
        if execution_len == 320:
            return ()
        assert execution_len == 384
        return ((192, 192, 192),)

    assert plan(
        320, 384, 64, resolve_stage_domain=resolve, queue_owner_id=17
    ) == (384, 192)

    with pytest.raises(_PLANNER.PlannerRejectError, match="CAPABILITY"):
        plan(
            320, 384, 64,
            resolve_stage_domain=lambda _execution_len: (),
            queue_owner_id=17,
        )


def test_qwen3_5_prefill_position_ids_accept_hf_generation_form():
    normalize = _load_text_function("_normalize_prefill_position_ids")
    text = torch.arange(5).view(1, 1, 5)
    vision = torch.tensor([
        [[10, 11, 12, 13, 14]],
        [[20, 21, 22, 23, 24]],
        [[30, 31, 32, 33, 34]],
    ])
    out = normalize(torch.cat([text, vision], dim=0),
                    real_len=5, padded_len=8)
    assert out.shape == (3, 8)
    assert torch.equal(out[:, :5], vision[:, 0])


def test_qwen3_5_prefill_position_ids_reject_batch_gt_one():
    normalize = _load_text_function("_normalize_prefill_position_ids")
    with pytest.raises(ValueError, match="batch_size == 1"):
        normalize(torch.zeros(4, 2, 5, dtype=torch.long),
                  real_len=5, padded_len=64)


def test_qwen3_5_prefill_mrope_cache_is_exact_and_opt_in():
    tree = ast.parse(_TEXT.read_text(), filename=str(_TEXT))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_get_prefill_mrope_tables"
    )
    calls = 0

    def build(position_ids, *_args):
        nonlocal calls
        calls += 1
        table = torch.full((position_ids.shape[1], 1), calls)
        return table, -table

    namespace = {"_build_interleaved_mrope_cos_sin": build}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(_TEXT), "exec"),
         namespace)
    get_tables = namespace["_get_prefill_mrope_tables"]

    generic = types.SimpleNamespace()
    pid = torch.arange(6).view(1, -1).expand(3, -1).clone()
    get_tables(generic, pid, 64, 10_000_000.0, (11, 11, 10))
    get_tables(generic, pid.clone(), 64, 10_000_000.0, (11, 11, 10))
    assert calls == 2

    state = types.SimpleNamespace(prefill_rope_cache=None)
    expected_pid = pid.clone()
    first = get_tables(state, pid, 64, 10_000_000.0, (11, 11, 10))
    pid[0, 0] = 99
    second = get_tables(
        state, expected_pid, 64, 10_000_000.0, (11, 11, 10))
    assert calls == 3
    assert first[0].data_ptr() == second[0].data_ptr()
    assert first[1].data_ptr() == second[1].data_ptr()

    changed = expected_pid.clone()
    changed[0, 0] = 1
    get_tables(state, changed, 64, 10_000_000.0, (11, 11, 10))
    get_tables(state, changed[:, :-1], 64, 10_000_000.0, (11, 11, 10))
    assert calls == 5


def test_qwen3_5_text_rejects_nonzero_multi_token_prefill():
    source = _TEXT.read_text()
    assert "real_len > 1 and cache.position != 0" in source
    assert "reset the cache and re-feed the full prefix" in source












class _ForwardCache:
    def __init__(self, position=0):
        self.position = position


def _load_forward(*, run=None, fuse=None, preflight=None, prepare=None):
    tree = ast.parse(_ADAPTER.read_text(), filename=str(_ADAPTER))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_rpu_qwen3_5_forward"
    )
    namespace = {
        "os": __import__("os"),
        "torch": torch,
        "quanttrio_awq_config": lambda _config: None,
        "is_mlp_only_text_profile": lambda _config: False,
        "F": torch.nn.functional,
        "Qwen3_5Cache": _ForwardCache,
        "Qwen3_5CausalLMOutputWithPast": object,
        "_preflight_qwen3_5_text": preflight or (
            lambda _model, _cache, real_len, **_kw: (real_len, 0, None)
        ),
        "run_qwen3_5_text": run or (lambda *a, **kw: None),
        "fuse_visual_embeds": fuse or (lambda *a, **kw: None),
        "_prepare_qwen3_5_multimodal_spm": prepare or (lambda *a, **kw: False),
        "execution_serialized": lambda fn: fn,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(_ADAPTER), "exec"),
         namespace)
    forward = namespace["_rpu_qwen3_5_forward"]
    hw_attrs = types.ModuleType("rpu_backend.runtime.hw_attrs")
    hw_attrs.mark_first_forward_done = lambda _model: None

    def isolated_forward(*args, **kwargs):
        # Stub the parents too: importing a real runtime while its leaf is
        # replaced otherwise leaves runtime.hw_attrs pointing at this mock.
        with patch.dict(
            sys.modules, {
                "rpu_backend": types.ModuleType("rpu_backend"),
                "rpu_backend.runtime": types.ModuleType("rpu_backend.runtime"),
                "rpu_backend.runtime.hw_attrs": hw_attrs,
            }
        ):
            return forward(*args, **kwargs)

    return isolated_forward


def _forward_self():
    inner = types.SimpleNamespace()
    return types.SimpleNamespace(model=inner)


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"past_key_values": _ForwardCache()}, ValueError, "exactly one"),
        ({"input_ids": torch.ones(1, 1, dtype=torch.long),
          "inputs_embeds": torch.ones(1, 1, 2),
          "past_key_values": _ForwardCache()}, ValueError, "exactly one"),
        ({"input_ids": torch.ones(1, 1, dtype=torch.long)}, TypeError,
         "requires Qwen3_5Cache"),
        ({"input_ids": torch.ones(1, 1, dtype=torch.long),
          "past_key_values": _ForwardCache(), "use_cache": False},
         NotImplementedError, "use_cache=True"),
        ({"input_ids": torch.ones(1, 1, dtype=torch.long),
          "past_key_values": _ForwardCache(), "return_dict": False},
         NotImplementedError, "return_dict=True"),
        ({"input_ids": torch.ones(1, 2, dtype=torch.long),
          "past_key_values": _ForwardCache(),
          "attention_mask": torch.tensor([[1, 0]])},
         NotImplementedError, "padding/zero"),
        ({"input_ids": torch.ones(1, 1, dtype=torch.long),
          "past_key_values": _ForwardCache(),
          "pixel_values_videos": torch.ones(1)},
         NotImplementedError, "video inputs"),
        ({"input_ids": torch.ones(1, 1, dtype=torch.long),
          "past_key_values": _ForwardCache(), "unknown_option": 1},
         TypeError, "unknown_option"),
        ({"input_ids": torch.ones(1, 1, dtype=torch.long),
          "past_key_values": _ForwardCache(position=1),
          "pixel_values": torch.ones(1),
          "image_grid_thw": torch.tensor([[1, 2, 2]]),
          "mm_token_type_ids": torch.ones(1, 1, dtype=torch.int)},
         NotImplementedError, "prefill position 0"),
    ],
)
def test_qwen3_5_forward_rejects_unsupported_contracts(kwargs, error, match):
    with pytest.raises(error, match=match):
        _load_forward()(_forward_self(), **kwargs)


def test_qwen3_5_forward_stages_cpu_inputs_embeds_before_fused_text():
    original = torch.randn(1, 2, 4)
    seen = {}

    def fake_to(tensor, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if str(device).startswith("rpu"):
            seen["staged"] = True
            return tensor
        return original_to(tensor, *args, **kwargs)

    def fake_run(_model, hidden, *_args, **_kwargs):
        seen["same_tensor"] = hidden.data_ptr() == original.data_ptr()
        raise RuntimeError("stop-after-text-boundary")

    original_to = torch.Tensor.to
    forward = _load_forward(run=fake_run,
                            prepare=lambda *a: pytest.fail("text-only must not install vision"))
    kwargs = {
        "inputs_embeds": original,
        "past_key_values": _ForwardCache(position=1),
    }
    with patch.object(torch.Tensor, "to", fake_to):
        with pytest.raises(RuntimeError, match="stop-after-text-boundary"):
            forward(_forward_self(), **kwargs)
    assert seen == {"staged": True, "same_tensor": True}


@pytest.mark.parametrize(
    "seq_len, code", [(384, "EXACT_MISMATCH"), (385, "CAPABILITY")]
)
def test_qwen3_5_text_planner_rejects_before_vision_fusion(seq_len, code):
    events = []

    def reject(*_args, **_kwargs):
        events.append("text-preflight")
        raise RuntimeError(code)

    def fuse(*_args, **_kwargs):
        events.append("vision")

    forward = _load_forward(preflight=reject, fuse=fuse,
                            prepare=lambda *a: events.append("prepare"))
    with pytest.raises(RuntimeError, match=code):
        forward(
            _forward_self(),
            input_ids=torch.ones(1, seq_len, dtype=torch.long),
            past_key_values=_ForwardCache(),
            pixel_values=torch.ones(1),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            mm_token_type_ids=torch.ones(1, seq_len, dtype=torch.long),
        )

    assert events == ["text-preflight"]


def test_qwen3_5_final_text_plan_follows_shared_spm_prepare_before_embedding():
    events = []

    def plan(*args, **kwargs):
        events.append("plan")
        if len(events) == 3:
            raise RuntimeError("exact chunk no longer fits shared SPM")
        return 4096, 640, None

    def prepare(*args):
        events.append("reserve-fixed")
        return True

    forward = _load_forward(preflight=plan, prepare=prepare,
                            fuse=lambda *a: pytest.fail("must not execute vision"))
    with pytest.raises(RuntimeError, match="no longer fits"):
        forward(_forward_self(), input_ids=torch.ones(1, 4095, dtype=torch.long),
                past_key_values=_ForwardCache(), pixel_values=torch.ones(256, 1536),
                image_grid_thw=torch.tensor([[1, 16, 16]]),
                mm_token_type_ids=torch.ones(1, 4095, dtype=torch.long))
    assert events == ["plan", "reserve-fixed", "plan"]


def test_qwen3_5_vision_chunk_cap_has_one_cold_authority():
    source = _VISION_CPP.read_text(encoding="utf-8")
    body = source.split(
        "int64_t Qwen3_5VisionModel::subclass_chunk_size_cap(", 1
    )[1].split("}\n", 1)[0]
    assert "getenv" not in body
    assert "return chunk_size_cap_;" in body
