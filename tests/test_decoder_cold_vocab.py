"""Host regression for native decoder topology before optional lm_head binding."""

from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _function(source, signature):
    start = source.index(signature)
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def native_cold_vocab_probe(tmp_path_factory):
    compiler = shutil.which("g++")
    assert compiler, "native decoder contract checks require g++"
    header = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    source = (ROOT / "src/fused/rpu_qwen3_model.cpp").read_text()
    methods = "\n".join(_function(header, signature) for signature in (
        "void configure_cold_routes(", "int lm_head_tp() const",
        "std::vector<int64_t> execution_topology() const", "void set_lm_head(",
        "void clear_lm_head()",
    ))
    field_names = ("configured_vocab_size_", "configured_lm_head_tp_", "cold_routes_configured_")
    fields = "\n".join(re.search(rf"    (?:int64_t|int|bool) {name} = [^;]+;", header)[0]
                       for name in field_names)
    reserve = re.search(r'\{"lm_head_out",\s*(A\([^\n]+\)),', header)[1]
    helper = _function(header, "inline int resolve_causal_decoder_lm_head_tp(")
    create = _function(source, "int64_t rpu_causal_decoder_create(")
    harness = r'''
#include "execution_topology.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
constexpr int NUM_CORES = 8;
namespace at {
enum ScalarType { kHalf, kChar, kByte, kFloat };
enum DeviceType { kPrivateUse1, kCPU };
struct Device { DeviceType value; DeviceType type() const { return value; } };
struct Tensor {
    std::vector<int64_t> shape;
    ScalarType dtype = kHalf;
    DeviceType place = kPrivateUse1;
    bool defined() const { return !shape.empty(); }
    int dim() const { return shape.size(); }
    int64_t size(int dim) const { return shape.at(dim); }
    int64_t numel() const { int64_t n = 1; for (auto d : shape) n *= d; return n; }
    ScalarType scalar_type() const { return dtype; }
    Device device() const { return {place}; }
};
}
namespace v3 {
inline constexpr int64_t QWEN3_FUSED_LM_HEAD_MAX_VOCAB = 200000;
at::Tensor rpu_retain_linear_quant_scale(const at::Tensor&, const at::Tensor& scale) { return scale; }
__HELPER__
struct CausalDecoderModel {
    int cores = 8, layers = 0, invalidations = 0;
    int64_t resolved_chunk = 0;
    bool qwen3_spm_kv_by_mha_enabled_ = true, adarms_fused_bcast_enabled_ = false;
    __FIELDS__
    at::Tensor lm_head_w_, lm_head_w_scale_, lm_head_logits_out_;
    std::vector<at::Tensor> tts_predictor_heads_;
    bool fuse_lm_head_ = false, fuse_prefill_last_lm_head_ = false;
    bool qwen3vl_2b_w8_profile_ = false, has_mrope_ = false, reuse_lm_head_output_ = false;
    int64_t vocab_size_ = 0, lm_head_exact_candidates_ = 0;
    uintptr_t lm_head_logits_dst_base_ = 0;
    int num_cores() const { return cores; }
    int num_layers() const { return layers; }
    int hidden_size() const { return 1024; }
    int intermediate_size() const { return 2816; }
    int num_q_heads() const { return 16; }
    int num_kv_heads() const { return 8; }
    int head_dim() const { return 64; }
    struct Layer { at::Tensor q_w, q_w_scale; };
    std::vector<Layer> layer_weights_;
    int attn_tp() const { return cores != 8 ? 4 : 8; }
    int mlp_tp() const { return cores; }
    int64_t get_last_resolved_chunk_size() const { return resolved_chunk; }
    void set_execution_core_count(int count) { assert(layers == 0); cores = count; }
    void invalidate_model_state() { ++invalidations; }
    static int64_t A(int64_t bytes) { return (bytes + 255) / 256 * 256; }
    int64_t reserved_bytes(int64_t bs) const { constexpr int DWIDTH = 2; return __RESERVE__; }
    __METHODS__
};
}
struct CausalDecoderRegistry {
    inline static int creates = 0;
    inline static std::unique_ptr<v3::CausalDecoderModel> model;
    static int64_t create() { ++creates; model = std::make_unique<v3::CausalDecoderModel>(); return creates; }
    static v3::CausalDecoderModel* get(int64_t, const char*) { return model.get(); }
};
__CREATE__
template<class F> void rejects(F&& function) {
    bool failed = false;
    try { function(); } catch (const std::runtime_error&) { failed = true; }
    assert(failed);
}
int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string mode = argv[1];
    const at::Tensor real_weight{{151936, 1024}};
    if (mode == "cold_head_lifecycle4" || mode == "cold_head_lifecycle6") {
        const int budget = mode.back() - '0';
        const auto handle = rpu_causal_decoder_create(true, false, budget, 151936);
        auto& model = *CausalDecoderRegistry::get(handle, "probe");
        model.layers = 28;
        const std::vector<int64_t> expected{budget, 4, budget, 4, 8};
        assert(model.execution_topology() == expected);
        assert(model.vocab_size_ == 0 && !model.fuse_lm_head_);
        const auto cold_bytes = model.reserved_bytes(1);
        assert(cold_bytes == model.A(200000 / 4 * 2));
        for (int cycle = 0; cycle < 2; ++cycle) {
            model.set_lm_head(real_weight);
            assert(model.vocab_size_ == 151936 && model.fuse_lm_head_);
            assert(model.execution_topology() == expected);
            assert(model.reserved_bytes(1) == cold_bytes);
            assert(model.reserved_bytes(1) >= real_weight.size(0) / 4 * 2);
            const int before_invalid = model.invalidations;
            rejects([&] { model.set_lm_head(at::Tensor{{152000, 1024}}); });
            for (auto dtype : {at::kChar, at::kByte}) {
                rejects([&] { model.set_lm_head(at::Tensor{{151936, 1024}, dtype}, at::Tensor{{151936}}); });
            }
            rejects([&] { model.set_lm_head(real_weight, at::Tensor{{151936}}); });
            rejects([&] { model.set_lm_head(real_weight, std::nullopt, true, 2); });
            assert(model.invalidations == before_invalid);
            assert(model.lm_head_w_.size(0) == 151936 && model.execution_topology() == expected);
            model.clear_lm_head();
            assert(model.vocab_size_ == 0 && !model.fuse_lm_head_);
            assert(model.execution_topology() == expected);
            assert(model.reserved_bytes(1) == cold_bytes);
        }
        assert(model.invalidations == 4);
    } else if (mode == "admission") {
        for (int64_t vocab : {0, -1, 151938, 152000, 200000}) {
            rejects([&] { rpu_causal_decoder_create(true, false, 6, vocab); });
            assert(CausalDecoderRegistry::creates == 0);
        }
        rejects([&] { rpu_causal_decoder_create(true, false, (int64_t{1} << 32) + 6, 151936); });
        assert(CausalDecoderRegistry::creates == 0);
        v3::CausalDecoderModel model;
        model.configure_cold_routes(true, false, 6, 151936);
        rejects([&] { model.configure_cold_routes(true, false, 8, 0); });
        assert(model.num_cores() == 6 && model.lm_head_tp() == 4);
        v3::CausalDecoderModel after_weights;
        after_weights.layers = 28;
        rejects([&] { after_weights.configure_cold_routes(true, false, 6, 151936); });
        v3::CausalDecoderModel after_planning;
        after_planning.resolved_chunk = 16;
        rejects([&] { after_planning.configure_cold_routes(true, false, 6, 151936); });
    } else if (mode == "legacy_eight") {
        v3::CausalDecoderModel legacy;
        legacy.configure_cold_routes(true, false);
        legacy.layers = 28;
        const std::vector<int64_t> expected{8, 8, 8, 8, 8};
        assert(legacy.execution_topology() == expected);
        assert(legacy.reserved_bytes(1) == legacy.A(200000 / 8 * 2));
        for (int vocab : {12288, 151936}) {
            legacy.set_lm_head(at::Tensor{{vocab, 1024}});
            assert(legacy.execution_topology() == expected);
            legacy.clear_lm_head();
            assert(legacy.execution_topology() == expected);
        }
        v3::CausalDecoderModel configured;
        configured.configure_cold_routes(true, false, 8, 151936);
        configured.layers = 28;
        configured.set_lm_head(real_weight);
        rejects([&] { configured.set_lm_head(at::Tensor{{12288, 1024}}); });
        // Hidden-only default-eight models retain their existing admission;
        // the fused lm_head upper bound remains enforced by set_lm_head.
        v3::CausalDecoderModel larger;
        larger.configure_cold_routes(true, false, 8, 256000);
        larger.layers = 28;
        assert(larger.execution_topology() == expected);
        rejects([&] { larger.set_lm_head(at::Tensor{{256000, 1024}}); });
    } else { assert(false); }
}
'''
    for marker, value in (("__HELPER__", helper), ("__METHODS__", methods),
                          ("__FIELDS__", fields), ("__RESERVE__", reserve),
                          ("__CREATE__", create)):
        harness = harness.replace(marker, value)
    directory = tmp_path_factory.mktemp("decoder-cold-vocab")
    source_file = directory / "probe.cpp"
    source_file.write_text(harness)
    binary = directory / "probe"
    built = subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(ROOT / "src/core"),
                    str(source_file), "-o", str(binary)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    return binary


@pytest.mark.parametrize("mode", ["cold_head_lifecycle4", "cold_head_lifecycle6", "admission", "legacy_eight"])
def test_native_decoder_cold_vocabulary(native_cold_vocab_probe, mode):
    subprocess.run([str(native_cold_vocab_probe), mode], check=True, capture_output=True, text=True)


def test_native_create_schema_keeps_vocab_optional_and_defaults_compatible():
    declarations = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    declaration = declarations.split("int64_t rpu_causal_decoder_create(", 1)[1].split(";", 1)[0]
    assert "int64_t num_cores = 8" in declaration
    assert "int64_t vocab_size = 0" in declaration
    registration = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    assert "int num_cores=8, int vocab_size=0) -> int" in registration
