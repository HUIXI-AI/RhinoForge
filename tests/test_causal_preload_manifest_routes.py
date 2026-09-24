"""Host-only coverage for inherited CausalDecoder persistent preload routes.

The model-specific COMPLETE-manifest owners below reuse
``CausalDecoderModel::declare_buffers``.  Its preload callbacks are therefore
real physical consumers even when the subclass replaces the decoder layer
body.  Exercise the production route helper and the production QKV-bias
consumer expressions so an owner cannot silently omit those inherited routes.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(
    os.environ.get(
        "RHINOFORGE_TEST_ROOT",
        Path(__file__).resolve().parents[1],
    )
)
QWEN = ROOT / "src/fused/rpu_qwen3_model.h"
HYVLA = ROOT / "src/fused/rpu_hyvla_vlm_model.cpp"
PLANNER = ROOT / "src/core/fmb_three_stage_chunk_plan.cpp"


def _balanced_block(source: str, marker: str, *, start: int = 0) -> str:
    begin = source.index(marker, start)
    opening = source.index("{", begin)
    depth = 0
    for cursor in range(opening, len(source)):
        if source[cursor] == "{":
            depth += 1
        elif source[cursor] == "}":
            depth -= 1
            if depth == 0:
                return source[begin : cursor + 1]
    raise AssertionError(f"unterminated production block: {marker}")


def _method_block(source: str, marker: str) -> str:
    """Find an inline/defined method, skipping a declaration if present."""
    cursor = 0
    while True:
        begin = source.index(marker, cursor)
        opening = source.find("{", begin)
        semicolon = source.find(";", begin)
        if opening >= 0 and (semicolon < 0 or opening < semicolon):
            return _balanced_block(source, marker, start=cursor)
        cursor = begin + len(marker)


def _balanced_call(region: str, marker: str) -> str:
    begin = region.index(marker)
    opening = region.index("(", begin)
    depth = 0
    for cursor in range(opening, len(region)):
        if region[cursor] == "(":
            depth += 1
        elif region[cursor] == ")":
            depth -= 1
            if depth == 0:
                semicolon = region.index(";", cursor)
                return region[begin : semicolon + 1]
    raise AssertionError(f"unterminated production call: {marker}")


def _constant(source: str, name: str) -> str:
    found = re.search(
        rf"(?:inline\s+)?constexpr\s+int64_t\s+{name}\s*=\s*[^;]+;",
        source,
    )
    assert found is not None, name
    return found.group(0)


def _production_preload_helper(source: str) -> str:
    return _method_block(
        source, "    void append_causal_decoder_preload_manifest_routes("
    )


def test_helper_site_set_matches_base_buffer_consumers() -> None:
    qwen = QWEN.read_text(encoding="utf-8")
    declare_buffers = _method_block(
        qwen, "    std::vector<BufferDecl> causal_baseline_buffer_declarations("
    )
    helper = _production_preload_helper(qwen)
    site_pattern = re.compile(r"([A-Z][A-Z0-9_]*_SITE)")

    consumer_sites = set(site_pattern.findall(declare_buffers))
    producer_sites = set(site_pattern.findall(helper))
    assert consumer_sites
    assert producer_sites == consumer_sites


def test_every_inherited_preload_owner_uses_the_common_manifest_helper() -> None:
    qwen = QWEN.read_text(encoding="utf-8")
    marker = "    void append_causal_decoder_preload_manifest_routes("
    assert marker in qwen, "the CausalDecoder persistent preload helper is missing"

    base_manifest = _method_block(
        qwen, "    FmbPhysicalExecutionManifest physical_manifest_for_candidate("
    )
    assert base_manifest.count("append_causal_decoder_preload_manifest_routes(") == 1

    # These are the complete set of direct subclasses that both reuse the base
    # BufferDecls and build a COMPLETE manifest from scratch.  The two other
    # base-buffer users extend the base manifest and therefore inherit its call.
    direct_owners = {
        "src/fused/rpu_hyvla_vlm_model.cpp":
            "HyVlaVlmModel::physical_manifest_for_candidate(",
        "src/fused/rpu_hyvla_expert_model.cpp":
            "HyVlaExpertModel::physical_manifest_for_candidate(",
        "src/fused/rpu_halo_action_expert_model.cpp":
            "HaloActionExpertModel::physical_manifest_for_candidate(",
        "src/fused/rpu_lingbot_v2_moe_model.cpp":
            "LingbotV2MoeExpertModel::physical_manifest_for_candidate(",
    }
    delegated_owners = {
        "src/fused/rpu_lingbot_denoise_model.cpp":
            "LingbotDenoiseModel::physical_manifest_for_candidate(",
        "src/fused/rpu_wall_oss_action_step_model.cpp":
            "WallOssActionStepModel::physical_manifest_for_candidate(",
    }

    discovered_base_buffer_users = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "src/fused").glob("*.cpp")
        if re.search(
            r"(?:CausalDecoderModel::declare_buffers|causal_baseline_buffer_declarations)\s*\(",
            path.read_text(encoding="utf-8"),
        )
    }
    assert discovered_base_buffer_users == set(direct_owners) | set(delegated_owners)

    for relative, method_marker in direct_owners.items():
        source = (ROOT / relative).read_text(encoding="utf-8")
        method = _method_block(source, method_marker)
        assert method.count("append_causal_decoder_preload_manifest_routes(") == 1
        assert "CausalDecoderModel::physical_manifest_for_candidate(" not in method

    for relative, method_marker in delegated_owners.items():
        source = (ROOT / relative).read_text(encoding="utf-8")
        method = _method_block(source, method_marker)
        assert method.count("CausalDecoderModel::physical_manifest_for_candidate(") == 1

    # Halo image-flow owns a fully independent BufferDecl graph.  Keep that
    # boundary explicit so the shared helper is not attached by class ancestry
    # alone.
    halo_image = (ROOT / "src/fused/rpu_halo_image_flow_model.cpp").read_text(
        encoding="utf-8"
    )
    halo_decls = _method_block(
        halo_image, "HaloImageFlowModel::declare_buffers("
    )
    halo_manifest = _method_block(
        halo_image, "HaloImageFlowModel::physical_manifest_for_candidate("
    )
    assert "CausalDecoderModel::declare_buffers(" not in halo_decls
    assert "append_causal_decoder_preload_manifest_routes(" not in halo_manifest


def test_production_preload_routes_match_text_and_act_consumers(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required for the host manifest test")

    qwen = QWEN.read_text(encoding="utf-8")
    hyvla = HYVLA.read_text(encoding="utf-8")
    planner = PLANNER.read_text(encoding="utf-8")

    preload_helper = _production_preload_helper(qwen)
    causal_dma_enum = _balanced_block(
        qwen, "enum class CausalDecoderMutableDmaRoute : int64_t"
    ) + ";"
    planning_mode_enum = _balanced_block(
        qwen, "enum class CausalDecoderPlanningMode : int64_t"
    ) + ";"
    hyvla_dma_enum = _balanced_block(
        hyvla, "enum class HyVlaVlmDmaRoute : int64_t"
    ) + ";"

    preload_names = (
        "CAUSAL_DECODER_INPUT_NORM_POOLER_PRELOAD_SITE",
        "CAUSAL_DECODER_INPUT_NORM_DIRECT_PRELOAD_SITE",
        "CAUSAL_DECODER_POST_NORM_POOLER_PRELOAD_SITE",
        "CAUSAL_DECODER_POST_NORM_DIRECT_PRELOAD_SITE",
        "CAUSAL_DECODER_ADARMS_INPUT_SHIFT_PRELOAD_SITE",
        "CAUSAL_DECODER_ADARMS_POST_SHIFT_PRELOAD_SITE",
        "CAUSAL_DECODER_Q_NORM_POOLER_PRELOAD_SITE",
        "CAUSAL_DECODER_Q_NORM_DIRECT_PRELOAD_SITE",
        "CAUSAL_DECODER_K_NORM_POOLER_PRELOAD_SITE",
        "CAUSAL_DECODER_K_NORM_DIRECT_PRELOAD_SITE",
        "CAUSAL_DECODER_QKV_BIAS_PRELOAD_SITE",
        "CAUSAL_DECODER_NVFP4_SCALE_PRELOAD_SITE",
        "CAUSAL_DECODER_FINAL_NORM_POOLER_PRELOAD_SITE",
        "CAUSAL_DECODER_FINAL_NORM_DIRECT_PRELOAD_SITE",
        "QWEN3VL_DELIVERY_NORM_SLAB_SITE",
    )
    constants = "\n".join(_constant(qwen, name) for name in preload_names)
    constants += "\n" + _constant(hyvla, "HYVLA_VLM_QKV_BIAS_PRELOAD_DMA_SITE")

    manifest_method = _method_block(
        hyvla, "HyVlaVlmModel::physical_manifest_for_candidate("
    )
    bias_begin = manifest_method.index("    if (has_qkv_bias_) {")
    act_manifest_block = _balanced_block(
        manifest_method, "    if (has_qkv_bias_) {", start=bias_begin
    )

    base_bias_region = _balanced_block(
        qwen, "        if (has_qkv_bias_) {",
        start=qwen.index("        // q_bias / k_bias / v_bias only present"),
    )
    text_consume = _balanced_call(
        base_bias_region, "this->ctx().consume_physical_route("
    )
    act_bias_region = hyvla[
        hyvla.index("        // bias 按 col-partition scatter") :
        hyvla.index("    // (HALO 在此还声明了一个", hyvla.index(
            "        // bias 按 col-partition scatter"
        ))
    ]
    act_consume = _balanced_call(act_bias_region, "ctx().consume_physical_route(")

    consume_route = _method_block(
        planner,
        "const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(",
    ).replace(
        "FmbPhysicalManifestConsumer::consume_route", "consume_physical_route"
    )
    validator = _method_block(planner, "void validate_physical_manifest_impl(")

    program = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#define TORCH_INTERNAL_ASSERT(condition) assert(condition)
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error(#condition); } while (false)
#define DWIDTH 2
#define NUM_CORES 8

enum class FmbPhysicalManifestState : int { UNSPECIFIED = 0, COMPLETE = 1 };
enum class FmbGraphLifecycle : int {
    UNSPECIFIED = 0, RETAINED_CACHE = 1, BOUNDED_ONESHOT = 2,
    NATIVE_COMPOSITE1 = 3, COMPOSITE_CHILD = 4,
};
enum class FmbLinearAccumulationPolicy : int {
    UNSPECIFIED = 0, ACC16 = 1, ACC32 = 2, MIXED_BY_SITE = 3,
};
enum class FmbRouteFamily : int {
    ATTENTION = 1, LINEAR = 2, ALL_REDUCE = 3, ROPE = 4,
    KV_INSERT = 5, MUTABLE_DMA = 6, NORMALIZATION = 7,
    ACTIVATION = 8, GRAPH_SCHEDULE = 9, COLLECTIVE = 10,
};
enum class AttentionExecutionPolicy : int { AUTO = 0, DDR_KV = 1 };
enum class FmbRopeTableResidency : int { UNSPECIFIED = 0 };
__CAUSAL_DMA_ENUM__
__PLANNING_MODE_ENUM__
__HYVLA_DMA_ENUM__
__CONSTANTS__

struct FmbRouteManifestEntry {
    int64_t site_id = 0;
    FmbRouteFamily family = FmbRouteFamily::ATTENTION;
    int64_t selector = 0;
    int64_t flags = 0;
    std::vector<int64_t> arguments;
    int64_t invocation = 0;
};
struct FmbPhysicalExecutionManifest {
    FmbPhysicalManifestState state = FmbPhysicalManifestState::UNSPECIFIED;
    int64_t logical_length = 0, physical_length = 0;
    int64_t execution_padding_rows = 0, kv_logical_length = 0;
    int64_t kv_insert_physical_rows = 0;
    FmbGraphLifecycle graph_lifecycle = FmbGraphLifecycle::UNSPECIFIED;
    FmbLinearAccumulationPolicy linear_accumulation =
        FmbLinearAccumulationPolicy::UNSPECIFIED;
    std::vector<FmbRouteManifestEntry> routes;
};
AttentionExecutionPolicy fmb_attention_execution_policy(
    const FmbRouteManifestEntry&) { return AttentionExecutionPolicy::DDR_KV; }
FmbRopeTableResidency fmb_rope_table_residency(
    const FmbPhysicalExecutionManifest&) {
    return FmbRopeTableResidency::UNSPECIFIED;
}
__VALIDATOR__

namespace at {
using IntArrayRef = std::vector<int64_t>;
struct Tensor {
    int64_t elements = 0;
    int64_t numel() const { return elements; }
};
}  // namespace at

struct Context {
    FmbPhysicalExecutionManifest manifest_;
    mutable std::vector<uint8_t> consumed_;
    bool has_complete_physical_manifest() const { return true; }
    const FmbPhysicalExecutionManifest& manifest() const { return manifest_; }
    const FmbRouteManifestEntry& find_route(
        FmbRouteFamily family, int64_t site, int64_t invocation) const {
        const auto target = std::make_tuple(
            static_cast<int64_t>(family), site, invocation);
        const auto found = std::lower_bound(
            manifest_.routes.begin(), manifest_.routes.end(), target,
            [](const FmbRouteManifestEntry& route, const auto& identity) {
                return std::make_tuple(static_cast<int64_t>(route.family),
                                       route.site_id, route.invocation) < identity;
            });
        if (found == manifest_.routes.end() || found->family != family ||
            found->site_id != site || found->invocation != invocation) {
            throw std::runtime_error("missing physical route");
        }
        return *found;
    }
__CONSUME_ROUTE__
    void bind(FmbPhysicalExecutionManifest manifest) {
        manifest_ = std::move(manifest);
        consumed_.assign(manifest_.routes.size(), 0);
    }
};

struct Harness {
    int64_t h = 2048, hd = 128, nq = 16, nkv = 4, layers = 2, tp = 4;
    bool has_qk_norm_ = true, has_qkv_bias_ = false;
    bool adarms_ = false, adarms_mutable_ = false;
    bool adarms_fused_bcast_enabled_ = false, nvfp4_ = false;
    bool norm_slab = false;
    bool qwen3vl_pooler_z1_dispatch_ = false;
    bool qwen3vl_multiview_text_composite_dispatch_ = false;
    CausalDecoderPlanningMode planning_mode_ = CausalDecoderPlanningMode::ORDINARY;
    at::Tensor norm_slab_w_{23};
    at::Tensor q_nvfp4_tensor_scales_{11}, k_nvfp4_tensor_scales_{12};
    at::Tensor v_nvfp4_tensor_scales_{13}, o_nvfp4_tensor_scales_{14};
    at::Tensor gate_nvfp4_tensor_scales_{15}, up_nvfp4_tensor_scales_{16};
    at::Tensor down_nvfp4_tensor_scales_{17};
    Context context;

    int64_t hidden_size() const { return h; }
    int64_t head_dim() const { return hd; }
    int64_t num_q_heads() const { return nq; }
    int64_t num_kv_heads() const { return nkv; }
    int64_t num_layers() const { return layers; }
    int64_t attn_tp() const { return tp; }
    int num_cores() const { return 8; }
    int mlp_tp() const { return 8; }
    int lm_head_tp() const { return 8; }
__TOPOLOGY_METHODS__
    bool norm_slab_active() const { return norm_slab; }
    Context& ctx() { return context; }
    const Context& ctx() const { return context; }

__PRELOAD_HELPER__

    void append_act_bias_routes(FmbPhysicalExecutionManifest& manifest) const {
        const int64_t nq = num_q_heads();
        const int64_t nkv = num_kv_heads();
        const int64_t hd = head_dim();
        const int64_t tp = attn_tp();
__ACT_MANIFEST_BLOCK__
    }

    void consume_text_bias(int64_t L, int64_t role) {
        const int tp_ = static_cast<int>(attn_tp());
        const at::Tensor bw{role == 0 ? nq * hd : nkv * hd};
        const int64_t per_core = bw.numel() / tp_;
__TEXT_CONSUME__
    }

    void consume_act_bias(int64_t L, int64_t bias_invocation) {
        (void)L;  // The act PersistentPerLayer callback intentionally reuses role identity.
        const int tp_ = static_cast<int>(attn_tp());
        const at::Tensor bw{bias_invocation == 0 ? nq * hd : nkv * hd};
        const int64_t per_core = bw.numel() / tp_;
__ACT_CONSUME__
    }
};

using Identity = std::tuple<int64_t, int64_t, int64_t>;
Identity identity(const FmbRouteManifestEntry& route) {
    return {static_cast<int64_t>(route.family), route.site_id, route.invocation};
}
void canonicalize(FmbPhysicalExecutionManifest& manifest) {
    std::sort(manifest.routes.begin(), manifest.routes.end(),
              [](const auto& lhs, const auto& rhs) {
                  return identity(lhs) < identity(rhs);
              });
}
FmbPhysicalExecutionManifest build(Harness& model) {
    FmbPhysicalExecutionManifest manifest;
    manifest.state = FmbPhysicalManifestState::COMPLETE;
    manifest.logical_length = 240;
    manifest.physical_length = 240;
    manifest.kv_logical_length = 240;
    manifest.kv_insert_physical_rows = 240;
    manifest.graph_lifecycle = FmbGraphLifecycle::COMPOSITE_CHILD;
    manifest.linear_accumulation = FmbLinearAccumulationPolicy::ACC16;
    model.append_causal_decoder_preload_manifest_routes(manifest);
    model.append_act_bias_routes(manifest);
    canonicalize(manifest);
    validate_physical_manifest_impl(manifest, 240, 0);
    return manifest;
}
const FmbRouteManifestEntry& route(
    const FmbPhysicalExecutionManifest& manifest, int64_t site,
    int64_t invocation) {
    for (const auto& entry : manifest.routes) {
        if (entry.family == FmbRouteFamily::MUTABLE_DMA &&
            entry.site_id == site && entry.invocation == invocation) return entry;
    }
    throw std::runtime_error("route not found");
}
int count_site(const FmbPhysicalExecutionManifest& manifest, int64_t site) {
    return static_cast<int>(std::count_if(
        manifest.routes.begin(), manifest.routes.end(),
        [site](const auto& entry) { return entry.site_id == site; }));
}
void expect(const FmbRouteManifestEntry& got, int64_t selector,
            std::vector<int64_t> arguments) {
    assert(got.selector == selector && got.flags == 0 &&
           got.arguments == arguments);
}

void direct_and_pooler() {
    for (int mode = 0; mode < 5; ++mode) {
        Harness model;
        if (mode == 1) model.qwen3vl_pooler_z1_dispatch_ = true;
        if (mode == 2) model.qwen3vl_multiview_text_composite_dispatch_ = true;
        if (mode == 3)
            model.planning_mode_ = CausalDecoderPlanningMode::QWEN3VL_POOLER_Z1;
        if (mode == 4)
            model.planning_mode_ = CausalDecoderPlanningMode::QWEN3VL_MULTIVIEW_TEXT;
        const bool tensor_preload = mode != 0;
        auto manifest = build(model);
        assert(manifest.routes.size() == 4 * model.layers + 1);
        const int64_t input_site = tensor_preload
            ? CAUSAL_DECODER_INPUT_NORM_POOLER_PRELOAD_SITE
            : CAUSAL_DECODER_INPUT_NORM_DIRECT_PRELOAD_SITE;
        const int64_t post_site = tensor_preload
            ? CAUSAL_DECODER_POST_NORM_POOLER_PRELOAD_SITE
            : CAUSAL_DECODER_POST_NORM_DIRECT_PRELOAD_SITE;
        const int64_t q_site = tensor_preload
            ? CAUSAL_DECODER_Q_NORM_POOLER_PRELOAD_SITE
            : CAUSAL_DECODER_Q_NORM_DIRECT_PRELOAD_SITE;
        const int64_t k_site = tensor_preload
            ? CAUSAL_DECODER_K_NORM_POOLER_PRELOAD_SITE
            : CAUSAL_DECODER_K_NORM_DIRECT_PRELOAD_SITE;
        const int64_t final_site = tensor_preload
            ? CAUSAL_DECODER_FINAL_NORM_POOLER_PRELOAD_SITE
            : CAUSAL_DECODER_FINAL_NORM_DIRECT_PRELOAD_SITE;
        for (int64_t L = 0; L < model.layers; ++L) {
            expect(route(manifest, input_site, L), 1,
                   {L, model.h, tensor_preload ? 1 : 0});
            expect(route(manifest, post_site, L), 1,
                   {L, model.h, tensor_preload ? 1 : 0});
            expect(route(manifest, q_site, L), 1,
                   {L, model.hd, 1, tensor_preload ? 1 : 0});
            expect(route(manifest, k_site, L), 1,
                   {L, model.hd, 1, tensor_preload ? 1 : 0});
        }
        expect(route(manifest, final_site, 0), 1,
               {model.h, tensor_preload ? 1 : 0});
        assert(count_site(manifest, CAUSAL_DECODER_QKV_BIAS_PRELOAD_SITE) == 0);
        assert(count_site(manifest, HYVLA_VLM_QKV_BIAS_PRELOAD_DMA_SITE) == 0);
    }
}

void norm_slab() {
    Harness model;
    model.norm_slab = true;
    // Per-layer norm/bias routes and the independent final-norm route are
    // replaced by the one delivery slab preload.
    model.has_qkv_bias_ = false;
    auto manifest = build(model);
    assert(manifest.routes.size() == 1);
    expect(route(manifest, QWEN3VL_DELIVERY_NORM_SLAB_SITE, 0), 1, {23});
    assert(count_site(manifest, CAUSAL_DECODER_INPUT_NORM_DIRECT_PRELOAD_SITE) == 0);
    assert(count_site(manifest, CAUSAL_DECODER_FINAL_NORM_DIRECT_PRELOAD_SITE) == 0);
    assert(count_site(manifest, CAUSAL_DECODER_QKV_BIAS_PRELOAD_SITE) == 0);
}

void bias_consumers_and_uniqueness() {
    Harness model;
    model.has_qkv_bias_ = true;
    auto manifest = build(model);
    assert(manifest.routes.size() == (4 + 3) * model.layers + 1 + 3);
    assert(count_site(manifest, CAUSAL_DECODER_QKV_BIAS_PRELOAD_SITE) ==
           3 * model.layers);
    assert(count_site(manifest, HYVLA_VLM_QKV_BIAS_PRELOAD_DMA_SITE) == 3);

    for (int64_t L = 0; L < model.layers; ++L) {
        for (int64_t role = 0; role < 3; ++role) {
            const int64_t per_core = role == 0 ? 512 : 128;
            expect(route(manifest, CAUSAL_DECODER_QKV_BIAS_PRELOAD_SITE,
                         3 * L + role), 3,
                   {L, role, per_core, per_core * DWIDTH, model.tp, 1});
        }
    }
    for (int64_t role = 0; role < 3; ++role) {
        const int64_t total = role == 0 ? 2048 : 512;
        const int64_t per_core = total / model.tp;
        expect(route(manifest, HYVLA_VLM_QKV_BIAS_PRELOAD_DMA_SITE, role), 1,
               {1, total, per_core, per_core * DWIDTH, model.tp});
    }

    model.context.bind(manifest);
    for (int64_t L = 0; L < model.layers; ++L) {
        for (int64_t role = 0; role < 3; ++role) {
            model.consume_text_bias(L, role);
            model.consume_act_bias(L, role);
        }
    }
    for (size_t i = 0; i < manifest.routes.size(); ++i) {
        const auto site = manifest.routes[i].site_id;
        if (site == CAUSAL_DECODER_QKV_BIAS_PRELOAD_SITE ||
            site == HYVLA_VLM_QKV_BIAS_PRELOAD_DMA_SITE) {
            assert(model.context.consumed_[i] == 1);
        }
    }

    // The production validator must reject a duplicate physical identity.
    auto duplicate = manifest;
    duplicate.routes.push_back(route(
        manifest, CAUSAL_DECODER_QKV_BIAS_PRELOAD_SITE, 0));
    canonicalize(duplicate);
    bool rejected = false;
    try { validate_physical_manifest_impl(duplicate, 240, 0); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);

    // The production consumers reject stale invocation/argument authority.
    auto stale = manifest;
    for (auto& entry : stale.routes) {
        if (entry.site_id == CAUSAL_DECODER_QKV_BIAS_PRELOAD_SITE &&
            entry.invocation == 4) ++entry.arguments[2];
    }
    model.context.bind(stale);
    rejected = false;
    try { model.consume_text_bias(1, 1); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}

void adarms_and_nvfp4() {
    Harness model;
    model.adarms_ = true;
    model.adarms_mutable_ = true;
    model.adarms_fused_bcast_enabled_ = true;
    model.nvfp4_ = true;
    auto manifest = build(model);
    assert(manifest.routes.size() == 4 * model.layers + 1 +
                                     2 * model.layers + 7);
    for (int64_t L = 0; L < model.layers; ++L) {
        expect(route(manifest, CAUSAL_DECODER_ADARMS_INPUT_SHIFT_PRELOAD_SITE, L),
               1, {L, model.h, 1, 1, 1});
        expect(route(manifest, CAUSAL_DECODER_ADARMS_POST_SHIFT_PRELOAD_SITE, L),
               1, {L, model.h, 1, 1, 1});
    }
    for (int64_t role = 0; role < 7; ++role) {
        const int64_t elements = 11 + role;
        expect(route(manifest, CAUSAL_DECODER_NVFP4_SCALE_PRELOAD_SITE, role),
               1, {role, elements, 2 * elements, 1});
    }
}

int main() {
    direct_and_pooler();
    norm_slab();
    bias_consumers_and_uniqueness();
    adarms_and_nvfp4();
}
'''
    replacements = {
        "CAUSAL_DMA_ENUM": causal_dma_enum,
        "PLANNING_MODE_ENUM": planning_mode_enum,
        "HYVLA_DMA_ENUM": hyvla_dma_enum,
        "CONSTANTS": constants,
        "VALIDATOR": validator,
        "CONSUME_ROUTE": consume_route,
        "PRELOAD_HELPER": preload_helper,
        "TOPOLOGY_METHODS": "\n".join(_method_block(qwen, marker) for marker in (
            "    std::array<int64_t, 6> cold_topology_arguments() const",
            "    std::vector<int64_t> bind_cold_topology_arguments(",
        )),
        "ACT_MANIFEST_BLOCK": act_manifest_block,
        "TEXT_CONSUME": text_consume,
        "ACT_CONSUME": act_consume,
    }
    for key, value in replacements.items():
        program = program.replace(f"__{key}__", value)

    cpp = tmp_path / "causal-preload-manifest.cpp"
    binary = tmp_path / "causal-preload-manifest"
    cpp.write_text(program, encoding="utf-8")
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", str(cpp), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 0, result.stderr
