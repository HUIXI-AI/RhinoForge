"""Board-free contract for the bounded Qwen3 raw-SPM attention adopter."""

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEADER = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
RUNTIME_CONFIG = (ROOT / "docs/runtime_config.md").read_text()


def _section(start: str, end: str) -> str:
    begin = HEADER.index(start)
    return HEADER[begin : HEADER.index(end, begin)]


def test_switch_is_validated_and_snapshotted_per_handle(monkeypatch):
    from rpu_backend.runtime.control import rpu_env_bool

    name = "RPU_QWEN3_SPM_KV_BY_MHA"
    monkeypatch.delenv(name, raising=False)
    assert rpu_env_bool(name, default=True)
    for value, expected in (("0", False), ("1", True), ("false", False), ("on", True)):
        monkeypatch.setenv(name, value)
        assert rpu_env_bool(name, default=True) is expected
    monkeypatch.setenv(name, "maybe")
    with pytest.raises(ValueError):
        rpu_env_bool(name, default=True)

    # Cold Python installation validates once and passes the snapshot into the
    # handle constructor; forward/planning reads the handle field only.
    installer = (ROOT / "python/rpu_backend/runtime/decoder.py").read_text()
    assert 'handle = torch.ops.rpu.causal_decoder_create(\n        rpu_env_bool("RPU_QWEN3_SPM_KV_BY_MHA", default=True),' in installer
    native = (ROOT / "src/fused/rpu_qwen3_model.cpp").read_text()
    create = native[native.index("int64_t rpu_causal_decoder_create(") :
                    native.index("void rpu_causal_decoder_destroy(")]
    assert "->configure_cold_routes(" in create
    configure = _section(
        "    void configure_cold_routes(",
        "    // `position` is the cache position",
    )
    assert "get_last_resolved_chunk_size() == 0" in configure
    assert "qwen3_spm_kv_by_mha_enabled_ = qwen3_spm_kv_by_mha;" in configure
    assert "!cold_routes_configured_ && num_layers() == 0" in configure
    assert "cold_routes_configured_ = true;" in configure
    assert "bool qwen3_spm_kv_by_mha_enabled_ = true;" in HEADER
    assert 'getenv("RPU_QWEN3_SPM_KV_BY_MHA")' not in HEADER
    dynamic = _section(
        "ModelDynamicConfig dynamic_config(",
        "bool subclass_chunk_size_valid(",
    )
    assert "qwen3_spm_kv_by_mha_enabled_" in dynamic
    assert "? AttentionExecutionPolicy::AUTO" in dynamic
    assert ": AttentionExecutionPolicy::DDR_KV" in dynamic
    assert "qwen3_spm_kv_by_mha_enabled()" not in dynamic


def test_switch_documents_auto_and_durable_ddr_fallback():
    assert "`RPU_QWEN3_SPM_KV_BY_MHA`" in RUNTIME_CONFIG
    assert "when the exact plan fits" in RUNTIME_CONFIG
    assert "`0` pins the DDR-cache attention path" in RUNTIME_CONFIG
    assert "unsupported shapes fall back to DDR automatically" in RUNTIME_CONFIG


def test_only_exact_plain_qwen3_short_prefill_is_eligible():
    eligible = _section(
        "bool subclass_spm_kv_by_mha_eligible(",
        "// ═══════════════════════════════════════════════════════════════════════\n"
        "    // static/dynamic config",
    )
    for contract in (
        "typeid(*this) != typeid(CausalDecoderModel)",
        "position != 0",
        "!layout.is_causal",
        "layout.use_attn_mask",
        "layout.batch_size != 1",
        "plan.chunk_mode != ChunkMode::SEQUENTIAL",
        "plan.compute.chunks.size() != 1",
        "seq != 16 && seq != 32 && seq != 64 && seq != 128",
        "num_q_heads() == 16",
        "num_kv_heads() == 8",
        "head_dim() == 128",
        "num_layers() == 28",
        "hidden_size() == 1024 && intermediate_size() == 3072",
        "hidden_size() == 2048 && intermediate_size() == 6144",
        "!has_qk_norm_",
        "has_qkv_bias_",
        "nvfp4_",
        "has_mrope_",
        "!deepstack_lang_layers_.empty()",
        "adarms_",
        "batch_decode_active_",
        "sdpa_kernel_ != SdpaKernelType::FLASH_ATTN_SPM",
        "layer.q_w.scalar_type() == at::kHalf",
        "layer.down_w.scalar_type() == at::kHalf",
    ):
        assert contract in eligible


def test_spm_layout_reuses_non_aliasing_sdpa_tmp_for_v_transpose():
    buffers = _section(
        "std::vector<BufferDecl> declare_buffers(",
        "int64_t subclass_layout_hash() const override",
    )
    assert "const bool spm_kv_by_mha" in buffers
    assert "tmp = std::max(tmp, A(bs * comp_cs * local_kv * DWIDTH));" in buffers
    assert "spm_kv_by_mha ? 4" in buffers
    assert 'decls.push_back({"v",' in buffers
    assert 'decls.push_back({"sdpa_tmp",' in buffers


def test_spm_attention_mirrors_kv_to_ddr_before_raw_dispatch():
    phase4 = _section(
        "// Phase 4: KV cache insert + SDPA + O_proj",
        "// Layer 0 establishes a full residual2",
    )
    consume_plan = phase4.index("const KvInsertSegmentPlan kv_plan = consume_causal_kv_plan(")
    kv_insert = phase4.index("rpu_launch_insert_kvcache_spm_unified_with_plan(")
    consume_attention = phase4.index("CAUSAL_DECODER_RAW_SPM_ATTN_SITE")
    transpose = phase4.index("rpu_launch_v_transpose_spm(")
    by_mha = phase4.index("rpu_launch_sdpa_by_mha_spm(")
    fallback = phase4.index("rpu_launch_sdpa_spm_dispatch(", by_mha)
    assert consume_plan < kv_insert < consume_attention < transpose < by_mha < fallback
    assert "k_cache, v_cache," in phase4[kv_insert:consume_attention]
    assert "kv_insert_spm_rows_, kv_plan);" in phase4[kv_insert:consume_attention]
    assert "mask_type == 1" in phase4
    assert "1.0 / std::sqrt(static_cast<double>(hd))" in phase4
    assert HEADER.count("rpu_launch_v_transpose_spm(") == 1
    assert HEADER.count("rpu_launch_sdpa_by_mha_spm(") == 1


def test_norm_preload_consumes_the_actual_cold_manifest_arguments(tmp_path):
    """Run native producer/helper/preload/consumer bodies without an RPU.

    Exercise the actual preload fragment's append lambda and the five ordinary
    norm callbacks against the production consumer. The main manifest builder
    has a separate append lambda and cannot certify these preload routes.
    """
    import re
    import shutil
    import subprocess

    compiler = shutil.which("g++")
    assert compiler, "native preload contract requires g++"
    header = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    consumer_source = (ROOT / "src/core/fmb_three_stage_chunk_plan.cpp").read_text()

    def balanced(source, start, opening="{", closing="}"):
        begin = source.index(opening, start)
        depth, end = 1, begin + 1
        while depth:
            depth += (source[end] == opening) - (source[end] == closing)
            end += 1
        return source[start:end]

    methods = "\n".join(balanced(header, header.index(signature)) for signature in (
        "std::array<int64_t, 6> cold_topology_arguments() const",
        "std::vector<int64_t> bind_cold_topology_arguments(",
    ))
    consumer = balanced(consumer_source, consumer_source.index(
        "const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route("))
    manifest = balanced(header, header.index(
        "void append_causal_decoder_preload_manifest_routes("))
    append_start = manifest.index("const auto append =")
    append_body = manifest.index(") {", append_start) + 2
    append = manifest[append_start:append_body] + balanced(manifest, append_body) + ";"
    preloads = _section("std::vector<BufferDecl> declare_buffers(",
                        "int64_t subclass_layout_hash() const override")
    sites = ("INPUT_NORM", "POST_NORM", "Q_NORM", "K_NORM", "FINAL_NORM")
    definitions, producers, consumers = [], [], []
    for role in sites:
        direct = f"CAUSAL_DECODER_{role}_DIRECT_PRELOAD_SITE"
        for mode in ("DIRECT", "POOLER"):
            symbol = f"CAUSAL_DECODER_{role}_{mode}_PRELOAD_SITE"
            definitions.append(re.search(
                rf"inline constexpr int64_t {symbol}\s*=\s*\d+LL;", header).group())
        position = manifest.index(direct)
        start = manifest.rindex("append(", 0, position)
        producers.append(balanced(manifest, start, "(", ")") + ";")
        position = preloads.index(direct)
        start = preloads.rindex("this->ctx().consume_physical_route(", 0, position)
        call = start + len("this->ctx().")
        consumers.append(preloads[start:call] + balanced(preloads, call, "(", ")") + ";")

    cpp = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
#define TORCH_INTERNAL_ASSERT(condition) assert(condition)
namespace at { using IntArrayRef = std::vector<int64_t>; }
enum class FmbRouteFamily { LINEAR, ALL_REDUCE, MUTABLE_DMA };
enum class CausalDecoderMutableDmaRoute { DDR_BROADCAST_TO_SPM = 1 };
struct FmbRouteManifestEntry {
    int64_t site_id; FmbRouteFamily family; int64_t selector, flags;
    std::vector<int64_t> arguments; int64_t invocation;
};
struct Manifest { std::vector<FmbRouteManifestEntry> routes; };
struct FmbPhysicalManifestConsumer {
    Manifest manifest_;
    const auto& manifest() const { return manifest_; }
    mutable std::vector<bool> consumed_;
    const FmbRouteManifestEntry& find_route(FmbRouteFamily family, int64_t site, int64_t invocation) const {
        for (const auto& route : manifest_.routes)
            if (route.family == family && route.site_id == site && route.invocation == invocation) return route;
        throw std::runtime_error("missing route");
    }
    const FmbRouteManifestEntry& consume_route(FmbRouteFamily, int64_t, int64_t,
        int64_t, at::IntArrayRef, int64_t) const;
    const FmbRouteManifestEntry& consume_physical_route(FmbRouteFamily family,
        int64_t site, int64_t selector, int64_t flags, at::IntArrayRef arguments,
        int64_t invocation = 0) const {
        return consume_route(family, site, selector, flags, arguments, invocation);
    }
};
__CONSUMER__
__DEFINITIONS__
struct Model {
    static constexpr int NUM_CORES = 8;
    int cores = 8, hidden = 1024, intermediate = 3072, layer = 7;
    bool has_qk_norm_ = true, qwen3vl_pooler_z1_dispatch_ = false;
    bool qwen3vl_multiview_text_composite_dispatch_ = false;
    FmbPhysicalManifestConsumer context;
    int num_cores() const { return cores; }
    int attn_tp() const { return cores == 8 ? 8 : 4; }
    int mlp_tp() const { return cores; }
    int lm_head_tp() const { return cores == 8 ? 8 : 4; }
    int hidden_size() const { return hidden; }
    int head_dim() const { return 128; }
    FmbPhysicalManifestConsumer& ctx() { return context; }
    __METHODS__
    void build_manifest() {
        auto& manifest = context.manifest_;
        __APPEND__
        const bool tensor_preload = false;
        const int64_t layer = this->layer;
        const int64_t broadcast_selector = static_cast<int64_t>(
            CausalDecoderMutableDmaRoute::DDR_BROADCAST_TO_SPM);
        __PRODUCERS__
        context.consumed_.assign(manifest.routes.size(), false);
    }
    void preload() {
        const int L = layer;
        __PRELOADS__
    }
};
int main() {
    for (const auto dims : std::vector<std::vector<int>>{
            {1024,3072,28}, {2048,6144,28}, {2560,9728,36}, {4096,12288,36}})
    for (int cores : {8, 6, 4, 8}) {
        Model model; model.cores = cores; model.hidden = dims[0];
        model.intermediate = dims[1]; model.layer = dims[2] - 1;
        model.build_manifest(); model.preload();
        assert(std::all_of(model.context.consumed_.begin(), model.context.consumed_.end(), [](bool v) { return v; }));
        const size_t original_widths[] = {3,3,4,4,2};
        for (size_t i = 0; i < 5; ++i) {
            const auto& args = model.context.manifest_.routes[i].arguments;
            assert(args.size() == original_widths[i] + (cores == 8 ? 0 : 6));
            if (cores != 8) {
                const std::vector<int64_t> expected{2, cores, 4, model.mlp_tp(), 4, 8};
                assert(std::equal(expected.begin(), expected.end(), args.end() - 6));
            }
        }
        // Exercise both directions of stale descriptor reuse, plus each of the
        // five originally incomplete preload argument vectors independently.
        for (size_t i = 0; i < 5; ++i) {
            Model broken = model; broken.context.manifest_.routes.clear(); broken.cores = cores; broken.build_manifest();
            auto& args = broken.context.manifest_.routes[i].arguments;
            if (cores == 8) args.insert(args.end(), {2,6,4,6,4,8});
            else args.resize(original_widths[i]);
            bool rejected = false;
            try { broken.preload(); } catch (const std::runtime_error&) { rejected = true; }
            assert(rejected && !broken.context.consumed_[i]);
        }
        if (cores == 6) {
            // Same execution budget and attention TP, different MLP template:
            // a 4B descriptor cannot reuse the previous N6/M6 cold identity.
            Model stale = model;
            stale.context.manifest_.routes.clear(); stale.build_manifest();
            auto& args = stale.context.manifest_.routes[0].arguments;
            args[args.size() - 3] = model.mlp_tp() == 4 ? 6 : 4;
            bool rejected = false;
            try { stale.preload(); } catch (const std::runtime_error&) { rejected = true; }
            assert(rejected && !stale.context.consumed_[0]);
        }
    }
}
'''
    for marker, value in (("__CONSUMER__", consumer), ("__METHODS__", methods),
                          ("__DEFINITIONS__", "\n".join(definitions)),
                          ("__APPEND__", append), ("__PRODUCERS__", "\n".join(producers)),
                          ("__PRELOADS__", "\n".join(consumers))):
        cpp = cpp.replace(marker, value)
    source = tmp_path / "preload_manifest.cpp"
    source.write_text(cpp)
    binary = source.with_suffix("")
    built = subprocess.run([compiler, "-std=c++17", "-O0", str(source), "-o", str(binary)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_reduced_native_forward_checks_real_cache_layout_before_mutation(tmp_path):
    import shutil
    import subprocess
    import torch
    from rpu_backend.api.cache import RPUCache
    from rpu_backend.runtime.decoder import pad_mrope_prefill_inputs

    compiler = shutil.which("g++")
    assert compiler, "native cache admission requires g++"
    source = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    begin = source.index('        TORCH_CHECK(hidden_states.device().type() == at::kPrivateUse1,',
                         source.index('"CausalDecoderModel::forward called before set_weights.'))
    end = source.index("        const bool saved_batch_decode_active = ", begin)
    admission = source[begin:end]
    common = (ROOT / "src/core/fmb_three_stage_chunk_plan.cpp").read_text()
    start = common.index("void validate_fmb_planning_shape(")
    shape_guard = common[start:common.index("\n}", start) + 2]
    fmb = (ROOT / "src/core/fused_model_base.cpp").read_text()
    start = fmb.index("    // 3. mask normalization")
    mask_normalization = fmb[start:fmb.index("    // 4. KV cache validation", start)]
    # Get physical dimensions from the real CPU RPUCache caller, without
    # allocating an RPU or teaching the fixture a duplicate swizzle formula.
    cache = RPUCache(1, 1, 608, 8, 128, attn_tp=4, device="cpu")
    old_cache = RPUCache(1, 1, 608, 8, 128, device="cpu")
    codec_cache = RPUCache(8, 1, 64, 16, 64, attn_tp=4, device="cpu")
    codec_old_cache = RPUCache(8, 1, 64, 16, 64, attn_tp=8, device="cpu")
    # The real VL padding helper appends invalid rows to the token metadata;
    # the native causal path must still protect and return the logical prefix.
    padded, padded_mask, _ = pad_mrope_prefill_inputs(
        torch.ones(1, 313, 4), torch.ones(1, 313, dtype=torch.long),
        torch.arange(313).view(1, 1, 313).expand(3, 1, 313), 320,
    )
    assert padded_mask[:, :313].all() and not padded_mask[:, 313:].any()
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
#include <c10/core/ScalarType.h>
#undef TORCH_CHECK
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
namespace at {
constexpr int kPrivateUse1 = 1;
using ScalarType = c10::ScalarType;
constexpr auto kHalf = ScalarType::Half, kFloat = ScalarType::Float;
struct Device {
    int kind = kPrivateUse1, index = 0;
    int type() const { return kind; }
    bool operator==(const Device& other) const { return kind == other.kind && index == other.index; }
};
struct Sizes : std::vector<int64_t> {
    using std::vector<int64_t>::vector;
    std::vector<int64_t> vec() const { return {begin(), end()}; }
};
struct Tensor {
    Sizes shape;
    ScalarType dtype = kHalf;
    Device where;
    bool dense = true, valid = true;
    Tensor(std::initializer_list<int64_t> dims) : shape(dims) {}
    bool defined() const { return valid; }
    bool is_contiguous() const { return dense; }
    Device device() const { return where; }
    ScalarType scalar_type() const { return dtype; }
    int64_t dim() const { return shape.size(); }
    int64_t size(int64_t axis) const { return shape.at(axis); }
    Sizes sizes() const { return shape; }
};
}
namespace detail { __SHAPE_GUARD__ }
constexpr int NUM_CORES = 8;
struct Model {
    int cores = 6, admitted = 0, hidden = 1024, layers = 28;
    int kv_heads = 8, dimension = 128, attention_cores = 4;
    int64_t position = 0, batch_slot = 0;
    bool is_causal = true, allow_batch_decode = true;
    bool has_mrope_ = false, adarms_ = false, decoder_mixed_precision_ = false;
    bool nvfp4_ = false, linear_acc32_ = false;
    bool reduced_qwen3vl_profile_ = false, reduced_internvla_profile_ = false;
    bool reduced_wall_text_profile_ = false, reduced_wall_action_profile_ = false;
    bool reduced_hyvla_profile_ = false, reduced_lingbot2_expert_profile_ = false;
    bool fast_replay_skip_layer_loop_ = false, preload_replay_skip_ = false;
    std::optional<at::Tensor> attention_mask;
    at::Tensor hidden_states{1,16,1024};
    std::vector<at::Tensor> k_caches{28, at::Tensor{__K_SHAPE__}};
    std::vector<at::Tensor> v_caches{28, at::Tensor{__V_SHAPE__}};
    int num_cores() const { return cores; }
    int hidden_size() const { return hidden; }
    int num_layers() const { return layers; }
    int num_kv_heads() const { return kv_heads; }
    int head_dim() const { return dimension; }
    int attn_tp() const { return attention_cores; }
    void configure(int width, int count, int budget) {
        hidden = width; layers = count; cores = budget;
        hidden_states.shape[2] = hidden;
        k_caches.assign(layers, at::Tensor{__K_SHAPE__});
        v_caches.assign(layers, at::Tensor{__V_SHAPE__});
    }
    void forward_admission() {
        __ADMISSION__
        int64_t seq_len = seq_len_in;
        __MASK_NORMALIZATION__
        assert(!effective_mask.has_value());
        assert(max_kv_seq_len == position + seq_len);
        ++admitted;
    }
};
int main() {
    for (const auto dims : std::vector<std::vector<int>>{
            {1024,28}, {2048,28}, {2560,36}, {4096,36}})
    for (int cores : {4,6}) for (bool speech_nvfp4 : {false,true}) {
        if (speech_nvfp4 && dims[0] > 2048) continue;
        for (int64_t position : {0,576,592}) {
            Model model; model.configure(dims[0], dims[1], cores); model.position = position;
            model.nvfp4_ = speech_nvfp4;
            model.forward_admission(); assert(model.admitted == 1);
        }
        const std::vector<std::function<void(Model&)>> corruptions{
            [](Model& m) { m.k_caches[0].shape = {__OLD_K_SHAPE__}; },
            [](Model& m) { m.v_caches.back().shape = {__OLD_V_SHAPE__}; },
            [](Model& m) { --m.v_caches.back().shape[1]; },
            [](Model& m) { m.k_caches.pop_back(); },
            [](Model& m) { m.v_caches.pop_back(); },
            [](Model& m) { m.k_caches.resize(m.layers == 28 ? 36 : 28, m.k_caches[0]); },
            [](Model& m) { m.v_caches.resize(m.layers == 28 ? 36 : 28, m.v_caches[0]); },
            [](Model& m) { m.k_caches.back().dtype = at::kFloat; },
            [](Model& m) { m.v_caches.back().dense = false; },
            [](Model& m) { m.k_caches.back().where.index = 1; },
            [](Model& m) { m.v_caches.back().where.kind = 0; },
            [](Model& m) { m.k_caches.back().valid = false; },
            [](Model& m) { m.v_caches.back().shape.pop_back(); },
            [](Model& m) { m.k_caches[0].shape[0] = 2; },
            [](Model& m) { m.hidden_states.shape[0] = 2; },
            [](Model& m) { m.hidden_states.shape[1] = 0; },
            [](Model& m) { m.hidden_states.shape[2] += 16; },
            [](Model& m) { m.hidden_states.dtype = at::kFloat; },
            [](Model& m) { m.hidden_states.dense = false; },
            [](Model& m) { m.position = 593; },
            [](Model& m) { m.position = INT64_MAX; },
            [](Model& m) { m.batch_slot = 1; },
            [](Model& m) { m.k_caches[0].shape[1] = INT64_MAX; },
            [](Model& m) { m.is_causal = false; },
            [](Model& m) { m.attention_mask = at::Tensor{1,16}; },
            [](Model& m) { m.has_mrope_ = true; },
            [](Model& m) { m.adarms_ = true; },
            [](Model& m) { m.decoder_mixed_precision_ = true; },
        };
        for (const auto& corrupt : corruptions) {
            Model model; model.configure(dims[0], dims[1], cores);
            model.nvfp4_ = speech_nvfp4; corrupt(model);
            bool rejected = false;
            try { model.forward_admission(); }
            catch (const std::runtime_error&) { rejected = true; }
            assert(rejected && model.admitted == 0);
        }
    }
    // Codec is an exact headless model with 16 KV heads of dimension 64.
    // Use shapes allocated by RPUCache, including its eight physical stripes.
    auto codec = []() {
        Model m; m.configure(512, 8, 4); m.kv_heads = 16; m.dimension = 64;
        m.k_caches.assign(8, at::Tensor{__CODEC_K_SHAPE__});
        m.v_caches.assign(8, at::Tensor{__CODEC_V_SHAPE__});
        return m;
    };
    for (int64_t rows : {1,16,32,48,64}) {
        auto m = codec(); m.hidden_states.shape[1] = rows;
        m.forward_admission(); assert(m.admitted == 1);
    }
    auto final_decode = codec(); final_decode.hidden_states.shape[1] = 1;
    final_decode.position = 63;
    final_decode.forward_admission(); assert(final_decode.admitted == 1);
    const std::vector<std::function<void(Model&)>> invalid_codec{
        [](Model& m) { m.k_caches[0].shape = {__CODEC_OLD_K_SHAPE__}; },
        [](Model& m) { m.v_caches.back().shape = {__CODEC_OLD_V_SHAPE__}; },
        [](Model& m) { m.k_caches[0].shape[2] = 2; m.k_caches[0].shape[3] = 8; },
        [](Model& m) { m.v_caches.back().shape[4] = 4; },
        [](Model& m) { --m.v_caches.back().shape[1]; },
        [](Model& m) { m.k_caches.pop_back(); },
        [](Model& m) { m.v_caches.back().dtype = at::kFloat; },
        [](Model& m) { m.k_caches.back().where.index = 1; },
        [](Model& m) { m.v_caches.back().dense = false; },
        [](Model& m) { m.position = 49; },
        [](Model& m) { m.hidden_states.shape[1] = 65; },
        [](Model& m) { m.kv_heads = 15; },
        [](Model& m) { m.dimension = 63; },
        [](Model& m) { m.attention_cores = 6; },
    };
    for (const auto& corrupt : invalid_codec) {
        auto m = codec(); corrupt(m); bool rejected = false;
        try { m.forward_admission(); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && m.admitted == 0);
    }
    // Equal element count does not make a codec layout valid for text.
    Model wrong_text; wrong_text.k_caches[0].shape[2] = 4;
    wrong_text.k_caches[0].shape[3] = 4;
    bool rejected_text = false;
    try { wrong_text.forward_admission(); } catch (const std::runtime_error&) { rejected_text = true; }
    assert(rejected_text && wrong_text.admitted == 0);
    // Default-eight cache geometry and existing batch capability remain owned
    // by the existing validation; the reduced-template gate does not narrow it.
    Model legacy; legacy.cores = 8;
    legacy.k_caches.assign(28, at::Tensor{__OLD_K_SHAPE__});
    legacy.v_caches.assign(28, at::Tensor{__OLD_V_SHAPE__});
    legacy.forward_admission(); assert(legacy.admitted == 1);
    // Exact InternVLA text carries M-RoPE but no VL mask domain.
    for (int cores : {4,6}) {
        Model text; text.configure(2048, 28, cores);
        text.reduced_internvla_profile_ = text.has_mrope_ = true;
        text.forward_admission(); assert(text.admitted == 1);
        Model masked; masked.configure(2048, 28, cores);
        masked.reduced_internvla_profile_ = masked.has_mrope_ = true;
        masked.attention_mask = at::Tensor{1,16};
        masked.attention_mask->dtype = c10::ScalarType::Long;
        bool rejected = false;
        try { masked.forward_admission(); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && masked.admitted == 0);
    }
    // Exact VL install permits M-RoPE/ACC32 without weakening physical caches.
    for (int cores : {4,6}) for (int width : {2048,2560,4096}) {
        Model vl; vl.configure(width, width == 2048 ? 28 : 36, cores);
        vl.reduced_qwen3vl_profile_ = true; vl.has_mrope_ = true; vl.linear_acc32_ = true;
        vl.forward_admission(); assert(vl.admitted == 1);
        vl.v_caches.back().shape = {__OLD_V_SHAPE__};
        bool rejected = false;
        try { vl.forward_admission(); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && vl.admitted == 1);
        // Exercise unpadded prefill, the actual adapter's padded shape and
        // decode; FMB must normalize all these causal token masks to no mask.
        for (int64_t sequence : {int64_t(1), int64_t(16), int64_t(__PADDED_LEN__)})
        for (auto dtype : {c10::ScalarType::Bool, c10::ScalarType::Byte,
                           c10::ScalarType::Short, c10::ScalarType::Int,
                           c10::ScalarType::Long, c10::ScalarType::Half,
                           c10::ScalarType::Float, c10::ScalarType::Double}) {
            Model masked; masked.configure(width, width == 2048 ? 28 : 36, cores);
            masked.reduced_qwen3vl_profile_ = true;
            masked.has_mrope_ = masked.linear_acc32_ = true;
            masked.hidden_states.shape[1] = sequence;
            masked.position = sequence == 1 ? 313 : 0;
            masked.attention_mask = at::Tensor{1, sequence};
            masked.attention_mask->dtype = dtype;
            masked.forward_admission(); assert(masked.admitted == 1);
        }
        const std::vector<std::function<void(Model&)>> invalid_masks{
            [](Model& m) { m.attention_mask->valid = false; },
            [](Model& m) { m.attention_mask->dense = false; },
            [](Model& m) { m.attention_mask->where.kind = 0; },
            [](Model& m) { m.attention_mask->where.index = 1; },
            [](Model& m) { m.attention_mask->shape = {16}; },
            [](Model& m) { m.attention_mask->shape = {1,1,16,16}; },
            [](Model& m) { m.attention_mask->shape[0] = 2; },
            [](Model& m) { m.attention_mask->shape[1] = 15; },
            [](Model& m) { m.attention_mask->dtype = c10::ScalarType::ComplexFloat; },
            [](Model& m) { m.attention_mask->dtype = c10::ScalarType::QInt8; },
            [](Model& m) { m.is_causal = false; },
            [](Model& m) { m.v_caches.back().shape = {__OLD_V_SHAPE__}; },
        };
        for (const auto& corrupt : invalid_masks) {
            Model masked; masked.configure(width, width == 2048 ? 28 : 36, cores);
            masked.reduced_qwen3vl_profile_ = true;
            masked.has_mrope_ = masked.linear_acc32_ = true;
            masked.attention_mask = at::Tensor{1,16};
            masked.attention_mask->dtype = c10::ScalarType::Long;
            corrupt(masked);
            bool rejected = false;
            try { masked.forward_admission(); }
            catch (const std::runtime_error&) { rejected = true; }
            assert(rejected && masked.admitted == 0);
        }
    }
}
'''
    values = {
        "__SHAPE_GUARD__": shape_guard, "__ADMISSION__": admission,
        "__MASK_NORMALIZATION__": mask_normalization,
        "__PADDED_LEN__": str(padded.size(1)),
        "__K_SHAPE__": ",".join(map(str, cache.k_caches[0].shape)),
        "__V_SHAPE__": ",".join(map(str, cache.v_caches[0].shape)),
        "__OLD_K_SHAPE__": ",".join(map(str, old_cache.k_caches[0].shape)),
        "__OLD_V_SHAPE__": ",".join(map(str, old_cache.v_caches[0].shape)),
        "__CODEC_K_SHAPE__": ",".join(map(str, codec_cache.k_caches[0].shape)),
        "__CODEC_V_SHAPE__": ",".join(map(str, codec_cache.v_caches[0].shape)),
        "__CODEC_OLD_K_SHAPE__": ",".join(map(str, codec_old_cache.k_caches[0].shape)),
        "__CODEC_OLD_V_SHAPE__": ",".join(map(str, codec_old_cache.v_caches[0].shape)),
    }
    for marker, value in values.items():
        cpp = cpp.replace(marker, value)
    path = tmp_path / "native_cache_admission.cpp"
    path.write_text(cpp)
    binary = path.with_suffix("")
    built = subprocess.run([compiler, "-std=c++17", "-O0",
                            "-I", str(Path(torch.__file__).resolve().parent / "include"),
                            str(path), "-o", str(binary)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_reduced_family_weights_admit_only_exact_profiles_before_commit(tmp_path):
    """Compile the actual pre-commit weight gate and topology resolver.

    The public Python installer checks tensor shapes before swizzle. This probe
    specifically verifies the native metadata/dtype gate and its mutation order.
    """
    import shutil
    import subprocess

    compiler = shutil.which("g++")
    assert compiler, "native family admission requires g++"
    header = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    commit = header.index("// Commit state")
    begin = header.index("        const bool reduced_packed_w4", commit)
    end = header.index("        set_model_params(", begin)
    gate = header[begin:end]
    from test_fmb_standalone_layout_rebuild import _definition
    acc32_setter = (_definition(header, "int64_t qwen3_inplace_capacity(") +
                    _definition(header, "void set_linear_acc32(bool enabled)"))
    topology_override = _definition(header, "DecoderExecutionTopology resolve_model_execution_topology(")
    width_override = _definition(header, "int64_t resolve_model_mlp_intermediate_size(int64_t logical_size) const override")
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <typeinfo>
#include <vector>
#include "src/core/execution_topology.h"
#include "src/core/rpu_wall_sorting_config.h"
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
namespace at {
enum ScalarType { kHalf, kChar, kByte, kFloat };
struct Tensor {
    ScalarType dtype = kHalf;
    int64_t rows = 0, cols = 0;
    int64_t size(int axis) const { return axis ? cols : rows; }
    int dim() const { return 2; }
    int64_t numel() const { return rows * cols; }
    ScalarType scalar_type() const { return dtype; }
};
using TensorList = std::vector<Tensor>;
}
using v3::qwen3_fp16_profile_num_layers;
using v3::llama_fp16_profile_num_layers;
using v3::internvla_text_profile_num_layers;
using v3::qwen3_vl_text_prefix_num_layers;
using v3::wall_oss_text_profile_num_layers;
using v3::wall_oss_action_profile_num_layers;
using v3::hyvla_vlm_profile_num_layers;
using v3::hyvla_expert_profile_num_layers;
using v3::lingbot2_expert_profile_num_layers;
using v3::resolve_decoder_execution_topology;
using v3::decoder_mlp_intermediate_size;
using v3::decoder_packed_w4_mlp_intermediate_size;
using v3::decoder_nvfp4_mlp_intermediate_size;
using v3::DecoderExecutionTopology;
struct FusedModelBase {
    int budget = 6, bound_mlp_tp = 0;
    virtual ~FusedModelBase() = default;
    int num_cores() const { return budget; }
    int mlp_tp() const { return bound_mlp_tp; }
    virtual DecoderExecutionTopology resolve_model_execution_topology(
        int64_t q, int64_t kv, int64_t d, int64_t h, int64_t i) const {
        return v3::resolve_decoder_execution_topology(budget,q,kv,d,h,i);
    }
    virtual int64_t resolve_model_mlp_intermediate_size(int64_t logical_size) const {
        return v3::decoder_mlp_intermediate_size(logical_size, bound_mlp_tp);
    }
};
struct CausalDecoderModel : FusedModelBase {
    virtual ~CausalDecoderModel() = default;
    int commits = 0, head_tp = 4;
    int64_t hidden_size = 1024, intermediate_size = 3072;
    int64_t num_q_heads = 16, num_kv_heads = 8, head_dim = 128;
    int64_t configured_vocab_size_ = 151936;
    bool linear_acc32_ = false, has_scale = false, has_nvfp4 = false;
    bool qwen3_prefill_w8a8_ = false, qwen3_c192_ = false, qwen3_c256_ = false, qwen3_c512_ = false;
    bool reduced_qwen3vl_profile_ = false;
    bool reduced_qwen3_w4_profile_ = false, reduced_qwen3_nvfp4_profile_ = false;
    bool reduced_llama_profile_ = false;
    bool reduced_internvla_profile_ = false;
    bool reduced_wall_text_profile_ = false, reduced_wall_action_profile_ = false;
    bool reduced_hyvla_profile_ = false, reduced_lingbot2_expert_profile_ = false;
    static constexpr int REDUCED_HYVLA_VLM_OWNER = 1, REDUCED_HYVLA_EXPERT_OWNER = 2;
    WallSortingConfig wall_sorting_;
    int64_t logical_intermediate_size_ = 0, physical_intermediate_ = 0;
    bool wall_prefill_mixed_requested_ = false, wall_prefill_staged_requested_ = false;
    bool cold_routes_configured_ = true, use_silu = true;
    bool qk_norm_present = true, qkv_bias_present = false;
    double eps = 1e-6;
    std::vector<int64_t> mrope_section, deepstack_lang_layers;
    at::TensorList q_w_list, k_w_list, v_w_list, o_w_list, gate_list, up_list, down_list;
    at::TensorList q_w_scale_list, k_w_scale_list, v_w_scale_list, o_w_scale_list,
        gate_scale_list, up_scale_list, down_scale_list;
    int num_cores() const { return budget; }
    int lm_head_tp() const { return head_tp; }
    virtual bool reduced_wall_action_owner() const { return false; }
    virtual int reduced_hyvla_owner() const { return 0; }
    virtual bool reduced_lingbot2_expert_owner() const { return false; }
    int num_layers() const { return commits ? q_w_list.size() : 0; }
    int get_last_resolved_chunk_size() const { return 0; }
    void invalidate_model_state() {}
    __ACC32_SETTER__
    __TOPOLOGY_OVERRIDE__
    __WIDTH_OVERRIDE__
    void configure(const std::vector<int64_t>& dims, int cores) {
        hidden_size = dims[0]; intermediate_size = dims[1];
        num_q_heads = dims[2]; budget = cores;
        head_tp = (cores == 8 ? 8 :
                   ((cores == 6 && (configured_vocab_size_ == 151936 || configured_vocab_size_ == 128256))
                    ? 6 : 4));
        for (auto* list : {&q_w_list, &k_w_list, &v_w_list, &o_w_list,
                           &gate_list, &up_list, &down_list})
            list->assign(dims[3], at::Tensor{});
        for (auto& w : q_w_list) { w.rows = num_q_heads * head_dim; w.cols = hidden_size; }
        for (auto* list : {&k_w_list, &v_w_list})
            for (auto& w : *list) { w.rows = num_kv_heads * head_dim; w.cols = hidden_size; }
        for (auto& w : o_w_list) { w.rows = hidden_size; w.cols = num_q_heads * head_dim; }
        const auto physical = v3::decoder_mlp_intermediate_size(intermediate_size, budget);
        for (auto* list : {&gate_list, &up_list})
            for (auto& w : *list) { w.rows = physical; w.cols = hidden_size; }
        for (auto& w : down_list) { w.rows = hidden_size; w.cols = physical; }
    }
    void pack_w4() {
        has_scale = true;
        // Packed G32 keeps MLP6; unaligned widths carry 192-aligned zero padding.
        const int mlp = budget;
        const auto physical = v3::decoder_packed_w4_mlp_intermediate_size(intermediate_size, mlp);
        auto pack = [&](auto& weights, auto& scales, int64_t n, int64_t k,
                        int partition, int cores) {
            const auto local_n = partition ? n / cores : n;
            const auto local_k = partition ? k : k / cores;
            const auto count = cores * ((local_k / 32 + 3) / 4) *
                ((local_n + 63) / 64) * 256;
            for (auto& weight : weights) weight = {at::kByte, n, k / 2};
            scales.assign(weights.size(), {at::kHalf, 32, count / 32});
        };
        pack(q_w_list,q_w_scale_list,num_q_heads * head_dim,hidden_size,1,4);
        pack(k_w_list,k_w_scale_list,num_kv_heads * head_dim,hidden_size,1,4);
        pack(v_w_list,v_w_scale_list,num_kv_heads * head_dim,hidden_size,1,4);
        pack(o_w_list,o_w_scale_list,hidden_size,num_q_heads * head_dim,0,4);
        pack(gate_list,gate_scale_list,physical,hidden_size,1,mlp);
        pack(up_list,up_scale_list,physical,hidden_size,1,mlp);
        pack(down_list,down_scale_list,hidden_size,physical,0,mlp);
    }
    void pack_nvfp4() {
        has_nvfp4 = true; has_scale = true;
        std::vector<at::TensorList*> weights{&q_w_list,&k_w_list,&v_w_list,
            &o_w_list,&gate_list,&up_list,&down_list};
        std::vector<at::TensorList*> scales{&q_w_scale_list,&k_w_scale_list,&v_w_scale_list,
            &o_w_scale_list,&gate_scale_list,&up_scale_list,&down_scale_list};
        for (size_t j=0;j<weights.size();++j) {
            scales[j]->clear();
            for (auto& w : *weights[j]) {
                scales[j]->push_back({at::kByte,w.cols/16,w.rows});
                w.cols /= 2; w.dtype = at::kByte;
            }
        }
    }
    void pack_nvfp4_physical() {
        // Dense NVFP4 stores the 384-aligned MLP width at six lanes.
        const auto physical = v3::decoder_nvfp4_mlp_intermediate_size(intermediate_size, budget);
        for (auto* list : {&gate_list, &up_list}) for (auto& w : *list) w.rows = physical;
        for (auto& w : down_list) w.cols = physical;
        pack_nvfp4();
    }
    void pack_w8() {
        has_scale = true;
        for (auto* list : {&q_w_list,&k_w_list,&v_w_list,&o_w_list,
                           &gate_list,&up_list,&down_list})
            for (auto& w : *list) w.dtype=at::kChar;
    }
    void set_weights_gate() {
        int64_t N = static_cast<int64_t>(q_w_list.size());
        // The complete real entry checks list equality/ranks before this gate;
        // every corruption below retains matching nonempty list arity.
        __GATE__
        // Mirror FusedModelBase::set_model_params: bind TP, then physical width.
        bound_mlp_tp = resolve_model_execution_topology(
            num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size).mlp_tp;
        physical_intermediate_ = resolve_model_mlp_intermediate_size(intermediate_size);
        ++commits;
    }
};
struct OtherDecoder : CausalDecoderModel {};
int main() {
    const std::vector<std::vector<int64_t>> profiles{
        {1024,3072,16,28}, {2048,6144,16,28},
        {2560,9728,32,36}, {4096,12288,32,36},
        {5120,17408,40,40}, {5120,25600,64,64}};
    for (const auto& profile : profiles) for (int budget : {4,6}) {
        CausalDecoderModel good; good.configure(profile,budget);
        good.set_weights_gate(); assert(good.commits == 1);
        auto topology = v3::resolve_decoder_execution_topology(
            budget, profile[2],8,128,profile[0],profile[1]);
        assert(topology.attention_tp == 4 && topology.physical_kv_cores == 8);
        assert(topology.mlp_tp == budget);
        const std::vector<std::function<void(CausalDecoderModel&)>> corruptions{
            [](auto& m) { m.gate_list.back().rows -= 16; },
            [](auto& m) { m.up_list.back().cols += 16; },
            [](auto& m) { m.down_list.back().cols -= 16; },
            [](auto& m) { m.hidden_size += 16; },
            [](auto& m) { m.intermediate_size += 16; },
            [](auto& m) { m.intermediate_size = m.intermediate_size == 3072 ? 6144 : 3072; },
            [](auto& m) { m.num_q_heads = m.num_q_heads == 16 ? 32 : 16; },
            [](auto& m) { m.num_kv_heads = 4; },
            [](auto& m) { m.head_dim = 64; },
            [](auto& m) {
                for (auto* l : {&m.q_w_list,&m.k_w_list,&m.v_w_list,&m.o_w_list,
                                &m.gate_list,&m.up_list,&m.down_list})
                    l->resize(l->size() == 28 ? 36 : 28);
            },
            [](auto& m) { m.configured_vocab_size_ += 16; },
            [](auto& m) { m.head_tp = (m.budget == 6 ? 4 : 6); },
            [](auto& m) { m.cold_routes_configured_ = false; },
            [](auto& m) { m.has_scale = true; },
            [](auto& m) { m.has_nvfp4 = true; },
            [](auto& m) { m.mrope_section = {24,20,20}; },
            [](auto& m) { m.deepstack_lang_layers = {5,11,17}; },
            [](auto& m) { m.wall_prefill_mixed_requested_ = true; },
            [](auto& m) { m.wall_prefill_staged_requested_ = true; },
            [](auto& m) { m.use_silu = false; },
            [](auto& m) { m.qk_norm_present = false; },
            [](auto& m) { m.qkv_bias_present = true; },
            [](auto& m) { m.budget = 5; },
        };
        for (const auto& corrupt : corruptions) {
            CausalDecoderModel bad; bad.configure(profile,budget); corrupt(bad);
            bool rejected = false;
            try { bad.set_weights_gate(); } catch (const std::runtime_error&) { rejected = true; }
            assert(rejected && bad.commits == 0);
        }
        for (int projection = 0; projection < 7; ++projection)
        for (auto dtype : {at::kChar,at::kByte,at::kFloat}) {
            CausalDecoderModel bad; bad.configure(profile,budget);
            std::vector<at::TensorList*> lists{&bad.q_w_list,&bad.k_w_list,
                &bad.v_w_list,&bad.o_w_list,&bad.gate_list,&bad.up_list,&bad.down_list};
            lists[projection]->back().dtype = dtype;
            bool rejected = false;
            try { bad.set_weights_gate(); } catch (const std::runtime_error&) { rejected = true; }
            assert(rejected && bad.commits == 0);
        }
        OtherDecoder other; other.configure(profile,budget);
        bool rejected = false;
        try { other.set_weights_gate(); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && other.commits == 0);
        for (bool acc32 : {false, true}) {
            CausalDecoderModel packed; packed.configure(profile,budget);
            packed.pack_w4(); packed.set_linear_acc32(acc32);
            packed.set_weights_gate();
            assert(packed.commits == 1 && packed.reduced_qwen3_w4_profile_);
        }
        const std::vector<std::function<void(CausalDecoderModel&)>> packed_faults{
            [](auto& m) { m.down_list.back().cols -= 16; },
            [](auto& m) { m.q_w_list.back().cols *= 2; },
            [](auto& m) { m.gate_scale_list.back().cols += 8; },
            [](auto& m) { m.o_w_scale_list.back().rows = 64; },
            [](auto& m) { m.up_list.back().dtype = at::kChar; },
            [](auto& m) { m.has_nvfp4 = true; },
        };
        for (const auto& fault : packed_faults) {
            CausalDecoderModel bad; bad.configure(profile,budget); bad.pack_w4(); fault(bad);
            bool rejected = false;
            try { bad.set_weights_gate(); } catch (const std::runtime_error&) { rejected = true; }
            assert(rejected && bad.commits == 0);
        }
        {
            CausalDecoderModel packed; packed.configure(profile,budget); packed.pack_w4();
            packed.set_weights_gate();
            const auto tp = packed.resolve_model_execution_topology(
                profile[2],8,128,profile[0],profile[1]);
            assert(tp.attention_tp == 4 && tp.mlp_tp == budget);
            // SPM/GEMM width: 9984/17664/25728 at MLP6; logical at MLP4.
            assert(packed.physical_intermediate_ ==
                   v3::decoder_packed_w4_mlp_intermediate_size(profile[1], budget));
            assert(packed.physical_intermediate_ % (32 * budget) == 0);
            assert(good.physical_intermediate_ ==
                   v3::decoder_mlp_intermediate_size(profile[1], budget));
        }
        // The FP16/W8 six-lane width (9792/17472) is not a legal packed stripe.
        const auto dense = v3::decoder_mlp_intermediate_size(profile[1], budget);
        if (dense != v3::decoder_packed_w4_mlp_intermediate_size(profile[1], budget)) {
            CausalDecoderModel bad; bad.configure(profile,budget); bad.pack_w4();
            for (auto* list : {&bad.gate_list, &bad.up_list}) for (auto& w : *list) w.rows = dense;
            for (auto& w : bad.down_list) w.cols = dense / 2;
            bool rejected = false;
            try { bad.set_weights_gate(); } catch (const std::runtime_error&) { rejected = true; }
            assert(rejected && bad.commits == 0);
        }
        // Dense 4B/8B/14B NVFP4 mirrors the packed-W4 width; 32B has no recipe,
        // and 0.6B/1.7B keep their speech-shared admission (checked below).
        if (profile[0] >= 2560) {
            CausalDecoderModel nvfp4; nvfp4.configure(profile,budget); nvfp4.pack_nvfp4_physical();
            bool rejected = false;
            try { nvfp4.set_weights_gate(); } catch (const std::runtime_error&) { rejected = true; }
            assert(rejected == (profile[3] == 64));
            if (!rejected) {
                assert(nvfp4.commits == 1 && nvfp4.reduced_qwen3_nvfp4_profile_ &&
                       !nvfp4.reduced_qwen3_w4_profile_);
                assert(nvfp4.physical_intermediate_ ==
                       v3::decoder_nvfp4_mlp_intermediate_size(profile[1], budget));
                // A later install cannot drop the bound packed width.
                nvfp4.has_nvfp4 = nvfp4.has_scale = false; nvfp4.configure(profile,budget);
                rejected = false;
                try { nvfp4.set_weights_gate(); } catch (const std::runtime_error&) { rejected = true; }
                assert(rejected && nvfp4.commits == 1);
            }
            if (profile[3] != 64 && dense != v3::decoder_nvfp4_mlp_intermediate_size(profile[1], budget)) {
                // The FP16/W8 six-lane width (9792/17472) is not the NVFP4 stripe.
                CausalDecoderModel bad; bad.configure(profile,budget); bad.pack_nvfp4();
                rejected = false;
                try { bad.set_weights_gate(); } catch (const std::runtime_error&) { rejected = true; }
                assert(rejected && bad.commits == 0);
            }
        }
    }
    // Speech cold identities distinguish full talkers, five-layer predictors,
    // and the headless codec. Exercise the real resolver after admission too.
    for (int budget : {4,6}) for (int h : {1024,2048})
    for (int vocab : {151936,3072,2048}) for (int precision : {0,1,2,3}) {
        if (vocab == 2048 && h != 1024) continue;
        const int layers = vocab == 2048 ? 5 : 28;
        auto make = [&]() {
            CausalDecoderModel m; m.configured_vocab_size_=vocab;
            m.configure({h,h*3,16,layers},budget);
            if (precision == 1) m.pack_w8();
            if (precision == 2) m.pack_w4();
            if (precision == 3) m.pack_nvfp4();
            return m;
        };
        auto good=make();good.set_weights_gate();assert(good.commits==1);
        const auto tp=good.resolve_model_execution_topology(16,8,128,h,h*3);
        assert(tp.attention_tp==4 && tp.mlp_tp==budget);
        const std::vector<std::function<void(CausalDecoderModel&)>> faults{
            [](auto& m){m.configured_vocab_size_=2049;},
            [](auto& m){m.q_w_list.back().cols-=16;},
            [](auto& m){m.k_w_list.back().rows-=16;},
            [](auto& m){m.qkv_bias_present=true;},
            [](auto& m){m.qk_norm_present=false;},
            [](auto& m){m.mrope_section={24,20,20};},
            [](auto& m){m.deepstack_lang_layers={0,1,2};},
            [](auto& m){m.q_w_list.back().dtype=at::kFloat;}
        };
        for (size_t f=0;f<faults.size();++f) {
            // Plain FP16/W8 shapes are checked by the public installer and the
            // native common shape gate; this extracted fragment owns the speech
            // marker and packed admission.
            if (vocab==151936 && precision<2 && (f==1 || f==2)) continue;
            auto bad=make();faults[f](bad);bool rejected=false;
            try{bad.set_weights_gate();}catch(const std::runtime_error&){rejected=true;}
            assert(rejected && bad.commits==0);
        }
        if (precision==3) for (int family=0;family<7;++family) {
            auto bad=make();std::vector<at::TensorList*> scales{
                &bad.q_w_scale_list,&bad.k_w_scale_list,&bad.v_w_scale_list,
                &bad.o_w_scale_list,&bad.gate_scale_list,&bad.up_scale_list,&bad.down_scale_list};
            scales[family]->back().cols+=16;bool rejected=false;
            try{bad.set_weights_gate();}catch(const std::runtime_error&){rejected=true;}
            assert(rejected && bad.commits==0);
            bad=make();bad.eps=1e-5;rejected=false;
            try{bad.set_weights_gate();}catch(const std::runtime_error&){rejected=true;}
            assert(rejected && bad.commits==0);
        }
    }
    auto codec = []() {
        CausalDecoderModel m;m.configured_vocab_size_=0;m.num_kv_heads=16;
        m.head_dim=64;m.eps=1e-5;m.qk_norm_present=false;
        m.configure({512,1024,16,8},4);return m;
    };
    auto c=codec();c.set_weights_gate();assert(c.commits==1);
    const auto ct=c.resolve_model_execution_topology(16,16,64,512,1024);
    assert(ct.num_cores==4 && ct.attention_tp==4 && ct.mlp_tp==4);
    for (const auto& fault : std::vector<std::function<void(CausalDecoderModel&)>>{
        [](auto& m){m.budget=6;},[](auto& m){m.configured_vocab_size_=3072;},
        [](auto& m){m.eps=1e-6;},[](auto& m){m.qk_norm_present=true;},
        [](auto& m){m.pack_w8();},[](auto& m){m.pack_w4();},
        [](auto& m){m.q_w_list.back().cols-=16;},[](auto& m){m.qkv_bias_present=true;}}) {
        auto bad=codec();fault(bad);bool rejected=false;
        try{bad.set_weights_gate();}catch(const std::runtime_error&){rejected=true;}
        assert(rejected && bad.commits==0);
    }
    // Exact Llama-3.2-1B reuses the template: D64, no QK-norm, vocab 128256,
    // MLP6 on the padded 8256 physical width (W4: 8448); every other variant is
    // refused.
    auto llama = [](int budget) {
        CausalDecoderModel m;m.configured_vocab_size_=128256;m.head_dim=64;
        m.eps=1e-5;m.qk_norm_present=false;
        m.configure({2048,8192,32,16},budget);
        if (budget == 6) m.head_tp = 6;
        return m;
    };
    for (int budget : {4,6}) {
        auto l=llama(budget);l.set_weights_gate();
        assert(l.commits==1 && l.reduced_llama_profile_ && !l.reduced_qwen3vl_profile_);
        const auto lt=l.resolve_model_execution_topology(32,8,64,2048,8192);
        assert(lt.num_cores==budget && lt.attention_tp==4 && lt.mlp_tp==budget);
        assert(l.physical_intermediate_==(budget==6 ? 8256 : 8192));
        assert(l.gate_list.back().rows==l.physical_intermediate_);
        // W8A16 and G32 packed W4 share the same bounded Llama admission.
        for (bool packed : {false,true}) {
            auto q=llama(budget);if(packed)q.pack_w4();else q.pack_w8();
            q.set_weights_gate();assert(q.commits==1 && q.reduced_llama_profile_);
            assert(q.physical_intermediate_==(budget==4 ? 8192 : 8256));
        }
        for (const auto& fault : std::vector<std::function<void(CausalDecoderModel&)>>{
            [](auto& m){m.qk_norm_present=true;},[](auto& m){m.configured_vocab_size_=151936;},
            [](auto& m){m.mrope_section={24,20,20};},[](auto& m){m.deepstack_lang_layers={0,1,2};},
            [](auto& m){m.qkv_bias_present=true;},[](auto& m){m.use_silu=false;},
            [](auto& m){m.gate_list.back().rows-=16;},[](auto& m){m.down_list.back().cols-=16;},
            [](auto& m){m.head_tp=(m.budget==6?4:6);},[](auto& m){m.budget=5;},
            [](auto& m){
                for (auto* l : {&m.q_w_list,&m.k_w_list,&m.v_w_list,&m.o_w_list,
                                &m.gate_list,&m.up_list,&m.down_list}) l->resize(15);
            },
            [](auto& m){m.q_w_list.back().dtype=at::kFloat;}}) {
            auto bad=llama(budget);fault(bad);bool rejected=false;
            try{bad.set_weights_gate();}catch(const std::runtime_error&){rejected=true;}
            assert(rejected && bad.commits==0);
        }
        OtherDecoder other; other.configured_vocab_size_=128256; other.head_dim=64;
        other.qk_norm_present=false; other.configure({2048,8192,32,16},budget);
        bool rejected=false;
        try{other.set_weights_gate();}catch(const std::runtime_error&){rejected=true;}
        assert(rejected && other.commits==0);
        // Qwen geometry cannot borrow the Llama vocabulary, with or without QK-norm.
        for (bool norm : {true,false}) {
            CausalDecoderModel qwen; qwen.configured_vocab_size_=128256;
            qwen.qk_norm_present=norm; qwen.configure({2048,6144,16,28},budget);
            rejected=false;
            try{qwen.set_weights_gate();}catch(const std::runtime_error&){rejected=true;}
            assert(rejected && qwen.commits==0);
        }
    }
    // Eight cores never mark the reduced Llama profile (K RoPE stays unchanged).
    {
        auto l=llama(8);l.set_weights_gate();
        assert(l.commits==1 && !l.reduced_llama_profile_);
    }
    // Exact headless InternVLA System-2 text (Qwen2.5-VL-7B W8A16): QKV bias,
    // no QK-norm, M-RoPE [16,24,24]; attention 4, MLP6 on the padded 19008.
    auto internvla = [](int budget) {
        CausalDecoderModel m;m.configured_vocab_size_=0;m.num_kv_heads=4;
        m.qk_norm_present=false;m.qkv_bias_present=true;m.mrope_section={16,24,24};
        m.configure({3584,18944,28,28},budget);m.pack_w8();return m;
    };
    for (int budget : {4,6}) {
        auto v=internvla(budget);v.set_weights_gate();
        assert(v.commits==1 && v.reduced_internvla_profile_ && !v.reduced_qwen3vl_profile_);
        const auto vt=v.resolve_model_execution_topology(28,4,128,3584,18944);
        assert(vt.num_cores==budget && vt.attention_tp==4 && vt.mlp_tp==budget);
        assert(v.physical_intermediate_==(budget==6 ? 19008 : 18944));
        for (const auto& fault : std::vector<std::function<void(CausalDecoderModel&)>>{
            [](auto& m){m.qk_norm_present=true;},[](auto& m){m.qkv_bias_present=false;},
            [](auto& m){m.configured_vocab_size_=151936;},
            [](auto& m){m.mrope_section={24,20,20};},[](auto& m){m.mrope_section.clear();},
            [](auto& m){m.deepstack_lang_layers={0,1,2};},
            [](auto& m){
                m.has_scale=false;
                for (auto* l : {&m.q_w_list,&m.k_w_list,&m.v_w_list,&m.o_w_list,
                                &m.gate_list,&m.up_list,&m.down_list})
                    for (auto& w : *l) w.dtype=at::kHalf;
            },
            [](auto& m){m.pack_w4();},[](auto& m){m.pack_nvfp4();},
            [](auto& m){m.eps=1e-5;},[](auto& m){m.use_silu=false;},
            [](auto& m){m.gate_list.back().rows-=16;},[](auto& m){m.head_tp=6;},
            [](auto& m){
                for (auto* l : {&m.q_w_list,&m.k_w_list,&m.v_w_list,&m.o_w_list,
                                &m.gate_list,&m.up_list,&m.down_list}) l->resize(27);
            }}) {
            auto bad=internvla(budget);fault(bad);bool rejected=false;
            try{bad.set_weights_gate();}catch(const std::runtime_error&){rejected=true;}
            assert(rejected && bad.commits==0);
        }
        OtherDecoder other; other.configured_vocab_size_=0; other.num_kv_heads=4;
        other.qk_norm_present=false; other.qkv_bias_present=true;
        other.mrope_section={16,24,24}; other.configure({3584,18944,28,28},budget);
        other.pack_w8();
        bool rejected=false;
        try{other.set_weights_gate();}catch(const std::runtime_error&){rejected=true;}
        assert(rejected && other.commits==0);
    }
    // GR00T's headless 16-layer 2B prefix: FP16, W8A16 and G32 packed W4 at
    // four and six lanes (I6144 is 384-aligned, never padded); NVFP4 stays
    // with the full models.
    for(int budget:{4,6})for(int precision:{0,1,2,3}) {
        CausalDecoderModel prefix;  // the 2B cold vocabulary topology
        prefix.configure({2048,6144,16,16},budget);
        prefix.mrope_section={24,20,20};prefix.deepstack_lang_layers={0,1,2};
        if(precision==1)prefix.pack_w8();
        if(precision==2)prefix.pack_w4();
        if(precision==3)prefix.pack_nvfp4();
        bool rejected=false;
        try{prefix.set_weights_gate();}catch(const std::runtime_error&){rejected=true;}
        assert(rejected==(precision==3) && prefix.commits==(precision==3 ? 0 : 1));
        if(precision==2)assert(prefix.gate_list.back().rows==6144);
    }
    // VL quantization retains exact M-RoPE/DeepStack and cold component domains.
    for(const auto& profile:profiles)for(int budget:{4,6})for(bool packed:{false,true}) {
        if(profile[0]==1024 || (profile[0]==5120 && profile[3]!=64))continue;
        CausalDecoderModel vl;vl.configure(profile,budget);
        vl.mrope_section={24,20,20};vl.deepstack_lang_layers={0,1,2};
        if(packed)vl.pack_w4();else vl.pack_w8();
        vl.set_weights_gate();assert(vl.commits==1 && vl.reduced_qwen3vl_profile_);
        const auto tp=vl.resolve_model_execution_topology(profile[2],8,128,profile[0],profile[1]);
        assert(tp.attention_tp==4 && tp.mlp_tp==budget);
    }
    // Default-eight non-family owners remain outside reduced-family admission.
    OtherDecoder legacy; legacy.configure({3072,8192,24,12},8);
    legacy.num_kv_heads = 6; legacy.qk_norm_present = false;
    legacy.set_weights_gate(); assert(legacy.commits == 1);
    for (const auto& profile : profiles) for (int budget : {4,6}) {
        if (profile[0] == 1024 || profile[0] == 5120) continue;
        CausalDecoderModel vl; vl.configure(profile,budget);
        vl.set_linear_acc32(true); assert(vl.linear_acc32_);
        vl.mrope_section = {24,20,20}; vl.deepstack_lang_layers = {0,1,2};
        vl.set_weights_gate(); assert(vl.commits == 1 && vl.reduced_qwen3vl_profile_);
        vl.set_linear_acc32(true); vl.set_weights_gate(); assert(vl.commits == 2);
        // A rebind cannot turn VL into plain Qwen3, or vice versa.
        vl.mrope_section.clear(); vl.deepstack_lang_layers.clear(); vl.linear_acc32_ = false;
        bool rejected = false;
        try { vl.set_weights_gate(); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && vl.commits == 2);
        CausalDecoderModel plain; plain.configure(profile,budget); plain.set_weights_gate();
        plain.set_linear_acc32(true); assert(plain.linear_acc32_);
        plain.mrope_section = {24,20,20}; plain.deepstack_lang_layers = {0,1,2};
        rejected = false;
        try { plain.set_weights_gate(); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && plain.commits == 1);
        for (const auto bad_sections : std::vector<std::vector<int64_t>>{{16,24,24},{24,20},{}}) {
            CausalDecoderModel bad; bad.configure(profile,budget);
            bad.mrope_section = bad_sections; bad.deepstack_lang_layers = {0,1,2};
            rejected = false;
            try { bad.set_weights_gate(); } catch (const std::runtime_error&) { rejected = true; }
            assert(rejected && bad.commits == 0);
        }
    }
}
'''.replace("__GATE__", gate).replace("__ACC32_SETTER__", acc32_setter).replace("__TOPOLOGY_OVERRIDE__", topology_override).replace("__WIDTH_OVERRIDE__", width_override)
    path = tmp_path / "native_family_weight_admission.cpp"
    path.write_text(cpp)
    binary = path.with_suffix("")
    built = subprocess.run(
        [compiler, "-std=c++17", "-O0", "-I", str(ROOT), str(path), "-o", str(binary)],
        capture_output=True, text=True,
    )
    assert built.returncode == 0, built.stderr
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_nvfp4_requires_all_seven_native_tensor_scale_tables(tmp_path):
    """Execute the existing native tensor-scale boundary for both speech depths."""
    import shutil
    import subprocess

    start = HEADER.index("        auto check_nvfp4_tensor_scales =")
    end = HEADER.index("        // ─", start)
    boundary = HEADER[start:end]
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error("rejected"); } while(false)
namespace c10 {enum class DeviceType{PrivateUse1,CPU};}
namespace at {
enum ScalarType{kHalf,kFloat,kByte};
struct Device {c10::DeviceType kind=c10::DeviceType::PrivateUse1; auto type()const{return kind;}};
struct Tensor {
    bool valid=true,dense=true;int64_t length=32,rank=1;ScalarType dtype=kFloat;Device where;
    bool defined()const{return valid;}int64_t dim()const{return rank;}
    int64_t size(int)const{return length;}int64_t numel()const{return length;}
    ScalarType scalar_type()const{return dtype;}Device device()const{return where;}
    bool is_contiguous()const{return dense;}
};
}
struct Model {
    int64_t N=28;bool has_nvfp4=true;
    at::Tensor q_nvfp4_tensor_scales,k_nvfp4_tensor_scales,v_nvfp4_tensor_scales,
        o_nvfp4_tensor_scales,gate_nvfp4_tensor_scales,up_nvfp4_tensor_scales,down_nvfp4_tensor_scales;
    std::vector<at::Tensor*> tables(){return {&q_nvfp4_tensor_scales,&k_nvfp4_tensor_scales,
        &v_nvfp4_tensor_scales,&o_nvfp4_tensor_scales,&gate_nvfp4_tensor_scales,
        &up_nvfp4_tensor_scales,&down_nvfp4_tensor_scales};}
    void validate(){__BOUNDARY__}
};
int main(){for(int layers:{5,28}) {
    Model good;good.N=layers;for(auto* t:good.tables())t->length=((layers+7)/8)*8;good.validate();
    for(int table=0;table<7;++table)for(int fault=0;fault<7;++fault){Model bad=good;auto* t=bad.tables()[table];
        if(fault==0)t->valid=false;if(fault==1)t->rank=2;if(fault==2)t->length=layers-1;
        if(fault==3)t->length=layers;if(fault==4)t->dtype=at::kHalf;
        if(fault==5)t->where.kind=c10::DeviceType::CPU;if(fault==6)t->dense=false;
        bool rejected=false;try{bad.validate();}catch(const std::runtime_error&){rejected=true;}assert(rejected);
    }
    Model ordinary=good;ordinary.has_nvfp4=false;
    bool rejected=false;try{ordinary.validate();}catch(const std::runtime_error&){rejected=true;}assert(rejected);
    for(auto* t:ordinary.tables())t->valid=false;ordinary.validate();
}}
'''.replace("__BOUNDARY__", boundary)
    path = tmp_path / "native_speech_tensor_scales.cpp"
    path.write_text(cpp)
    compiler = shutil.which("g++")
    assert compiler
    built = subprocess.run([compiler, "-std=c++17", str(path), "-o", str(path.with_suffix(""))], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    subprocess.run([str(path.with_suffix(""))], check=True, capture_output=True, text=True)
