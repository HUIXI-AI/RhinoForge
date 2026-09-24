"""Execute the production cold admission and math-route helpers without a board."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition


ROOT = Path(__file__).resolve().parents[1]


def test_action_precision_is_cold_transactional_and_reaches_norm_launch(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/fused/rpu_rhino_vla_model.cpp").read_text()
    methods = "\n".join(_definition(source, prefix) for prefix in (
        "void set_high_precision(",
        "void set_high_precision_fusions(",
        "void set_vector_k_norm(",
        "void set_vector_q_norm(",
        "void set_partial_rope(",
        "int64_t high_precision_route_flags(",
        "int64_t rope_route_selector(",
        "RpuUnaryPrecision unary_precision(",
        "RpuRmsNormSpmRoute rmsnorm_route(",
        "int64_t k_norm_rows(",
        "int64_t q_norm_rows(",
        "void emit_qk_norm(",
        "void emit_aligned_kv_tail_zero(",
        "std::vector<int64_t> high_precision_route_arguments(",
        "void consume_high_precision_route(",
        "void emit_norm_shift(",
    ))
    # TensorList defaults contain {}, so take the admission block before the
    # first tensor-validation lambda instead of treating those as a body.
    table = source[source.index("void set_denoise_loop_adarms_tables("):]
    table_guard = table[table.index("TORCH_CHECK"):table.index("auto check_rpu")]
    site = re.search(r"constexpr int64_t RHINO_HIGH_PRECISION_SITE = (\d+)LL", source)[1]
    rope_enum = re.search(r"enum class RhinoRopeRoute[^}]+};", source)[0]
    rope_wrapper = _definition((ROOT / "src/ops/rpu_rope.cpp").read_text(),
                               "void rpu_launch_rhinovla_rope_spm_kernel(")
    program = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
#define NUM_CORES 8
#define DWIDTH 2
namespace c10 { struct Half { explicit Half(float) {} }; }
enum class RpuUnaryPrecision { BASE, HIGH };
enum class RpuRmsNormSpmRoute { BASE, NEWTON, NEWTON_V32_GRID8 };
enum class ValuOpType { ADD };
enum class FmbRouteFamily { GRAPH_SCHEDULE };
ROPE_ENUM
enum class KernelId { PARTIAL_MROPE };
bool graph_active = false, transaction_active = false, payload_ready = true;
int payload_checks = 0, fused_calls = 0, norm_calls = 0, add_calls = 0, high_fused_calls = 0;
bool rope_payload_ready = true;
int rope_payload_checks = 0;
bool get_kernel(KernelId id) {
    assert(id == KernelId::PARTIAL_MROPE); ++rope_payload_checks; return rope_payload_ready;
}
#define GET_KERNEL(id) get_kernel(id)
struct Tensor {
    uintptr_t pointer;
    template<class T> T* data_ptr() const { return reinterpret_cast<T*>(pointer); }
};
namespace at { using Tensor = ::Tensor; }
struct RopeCall {
    bool partial;
    uint32_t input, output;
    const Tensor *cos_owner, *sin_owner;
    c10::Half *cos, *sin;
    int64_t position, rows, heads, hd, rotary, cores;
};
std::vector<RopeCall> ropes;
void rpu_launch_partial_mrope_spm_kernel(uint32_t input, uint32_t output,
        const Tensor& cos, const Tensor& sin, int64_t position, int64_t rows,
        int64_t heads, int64_t hd, int64_t rotary, int64_t cores) {
    ropes.push_back({true,input,output,&cos,&sin,cos.data_ptr<c10::Half>(),
                    sin.data_ptr<c10::Half>(),position,rows,heads,hd,rotary,cores});
}
void rpu_launch_rope_spm_kernel(uint32_t input, uint32_t output,
        c10::Half* cos, c10::Half* sin, int64_t rows, int64_t heads,
        int64_t hd, int64_t position, int cores) {
    ropes.push_back({false,input,output,nullptr,nullptr,cos,sin,position,rows,heads,hd,hd,cores});
}
ROPE_WRAPPER
RpuRmsNormSpmRoute last_norm = RpuRmsNormSpmRoute::BASE;
struct NormCall { uint32_t input, output, weight; int64_t rows, cols; RpuRmsNormSpmRoute route; };
std::vector<NormCall> norms;
std::vector<std::string> events;
bool k_tail_zero = false, q_tail_zero = false;
struct RpuExecutionCoordinator {
    static void require_graph_quiescent(const char*) { TORCH_CHECK(!graph_active); }
    static void check_current_thread_execution_allowed(const char*) { TORCH_CHECK(!transaction_active); }
};
void rpu_require_high_precision_fusion_kernels() { ++payload_checks; TORCH_CHECK(payload_ready); }
void rpu_require_high_precision_math_kernels() { ++payload_checks; TORCH_CHECK(payload_ready); }
RpuRmsNormSpmRoute rpu_resolve_high_precision_rmsnorm_spm_route(int64_t rows, int64_t cols) {
    assert(rows > 0 && (cols == 128 || cols == 1024));
    return rows % 32 == 0 ? RpuRmsNormSpmRoute::NEWTON_V32_GRID8 : RpuRmsNormSpmRoute::NEWTON;
}
void rpu_launch_rhinovla_newton_norm_shift_spm_kernel(uint32_t, uint32_t, uint32_t, uint32_t, int64_t, double, int) {
    ++high_fused_calls;
}
void rpu_launch_pi05_adarms_norm_shift_spm_kernel(uint32_t, uint32_t, uint32_t, uint32_t, int64_t, double, int) {
    ++fused_calls;
}
void rpu_launch_rmsnorm_spm_kernel(uint32_t input, uint32_t output, uint32_t weight,
                                  int64_t rows, int64_t cols, double, RpuRmsNormSpmRoute route) {
    ++norm_calls; last_norm = route;
    norms.push_back({input, output, weight, rows, cols, route});
    events.push_back(input == 0x1000 ? "q_norm" : input == 0x2000 ? "k_norm" : "norm");
    if (input == 0x2000 && rows == 32) assert(k_tail_zero);
    if (input == 0x1000 && rows == 64) assert(q_tail_zero);
}
void rpu_launch_memset_spm_multicore(uint32_t output, int64_t count, int cores) {
    assert(cores == 8);
    if (output == 0x1000 + 62 * 128 * 2) {
        assert(count == 256); q_tail_zero = true; events.push_back("zero_q"); return;
    }
    assert(count == 128);
    if (output == 0x2000 + 31 * 128 * 2) { k_tail_zero = true; events.push_back("zero_k"); }
    else { assert(output == 0x4000 + 31 * 128 * 2); events.push_back("zero_v"); }
}
void rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(uint32_t, uint32_t, uint32_t, int64_t, int64_t, c10::Half, ValuOpType, bool) {
    ++add_calls;
}
struct Context {
    int64_t site = 0, selector = 0, flags = 0;
    std::vector<int64_t> args;
    void consume_physical_route(FmbRouteFamily, int64_t s, int64_t sel, int64_t f, std::vector<int64_t> a) {
        site = s; selector = sel; flags = f; args = a;
    }
};
constexpr int64_t RHINO_HIGH_PRECISION_SITE = __SITE_VALUE__;
struct Model {
    struct Config { bool precompute_adarms = true, skip_adarms_gemv = false,
        packed_qkv = false, expert_fusions = true, aligned_kv = false; } cold_config_;
    bool high_precision_ = false, high_precision_bound_ = false, cold_config_bound_ = true;
    bool high_precision_fusions_ = false, high_precision_fusions_bound_ = false;
    bool vector_k_norm_ = false, vector_k_norm_bound_ = false;
    bool vector_q_norm_ = false, vector_q_norm_bound_ = false;
    bool partial_rope_ = false, partial_rope_bound_ = false;
    Tensor cos_ref_{0x100000}, sin_ref_{0x200000};
    bool denoise_loop_weights_ready_ = false, denoise_adarms_tables_ready_ = false;
    bool expert_w8a16_ = true, full_action_w8a16_ = true, gates_tanh_precomputed_ = false;
    int64_t suffix_len_ = 31, num_steps_ = 10, layers = 18, h = 1024, is = 3072;
    int64_t nq = 16, nkv = 8, hd = 128, tp = 8, last_chunk = 0, generation = 5;
    int64_t local_q_heads_ = 2;
    double eps_ = 1e-6;
    Context context;
    int64_t num_layers() const { return layers; }
    int64_t hidden_size() const { return h; }
    int64_t intermediate_size() const { return is; }
    int64_t num_q_heads() const { return nq; }
    int64_t num_kv_heads() const { return nkv; }
    int64_t head_dim() const { return hd; }
    int64_t attn_tp() const { return tp; }
    int64_t get_last_resolved_chunk_size() const { return last_chunk; }
    int64_t installed_model_state_generation() const { return generation; }
    void invalidate_model_state() { ++generation; }
    void validate_expert_transfer(int64_t rows) const { TORCH_CHECK(rows == 31); }
    uint32_t addr(int core, const char* name) const {
        assert(core == 0);
        const std::string value(name);
        if (value == "q") return 0x1000;
        if (value == "k") return 0x2000;
        if (value == "v") return 0x4000;
        if (value == "output") return 0x8000;
        assert(value == "input_norm"); return 0xa000;
    }
    uint32_t layer_addr(int layer, int core, const char* name) const {
        assert(core == 0);
        const std::string value(name);
        assert(value == "q_norm_w" || value == "k_norm_w");
        return 0x30000 + layer * 0x1000 + (value == "k_norm_w" ? 0x100 : 0);
    }
    Context& ctx() { return context; }
    METHODS
    void check_table_admission(bool high_precision) { TABLE_GUARD }
};
template<class F> void reject_unchanged(Model& model, F operation) {
    auto generation = model.generation;
    auto precision = model.high_precision_;
    auto bound = model.high_precision_bound_;
    auto vector = model.vector_k_norm_, vector_bound = model.vector_k_norm_bound_;
    auto q_vector = model.vector_q_norm_, q_vector_bound = model.vector_q_norm_bound_;
    auto rope = model.partial_rope_, rope_bound = model.partial_rope_bound_;
    auto fused = model.high_precision_fusions_, fused_bound = model.high_precision_fusions_bound_;
    bool rejected = false;
    try { operation(); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected && model.generation == generation && model.high_precision_ == precision && model.high_precision_bound_ == bound);
    assert(model.vector_k_norm_ == vector && model.vector_k_norm_bound_ == vector_bound &&
           model.vector_q_norm_ == q_vector && model.vector_q_norm_bound_ == q_vector_bound &&
           model.partial_rope_ == rope && model.partial_rope_bound_ == rope_bound &&
           model.high_precision_fusions_ == fused && model.high_precision_fusions_bound_ == fused_bound);
}
int main() {
    for (int damage = 0; damage < 10; ++damage) {
        Model model;
        if (damage == 0) graph_active = true;
        if (damage == 1) transaction_active = true;
        if (damage == 2) model.last_chunk = 31;
        if (damage == 3) model.denoise_loop_weights_ready_ = true;
        if (damage == 4) model.denoise_adarms_tables_ready_ = true;
        if (damage == 5) model.full_action_w8a16_ = false;
        if (damage == 6) model.layers = 17;
        if (damage == 7) model.cold_config_.skip_adarms_gemv = true;
        if (damage == 8) model.nq = 8;
        if (damage == 9) payload_ready = false;
        reject_unchanged(model, [&] { model.set_high_precision(true); });
        graph_active = transaction_active = false; payload_ready = true;
    }
    Model base;
    base.emit_norm_shift(256, 512, 768, 1024, 31);
    assert(fused_calls == 1 && norm_calls == 0 && add_calls == 0);
    assert(base.rmsnorm_route(62,128) == RpuRmsNormSpmRoute::BASE);
    assert(base.unary_precision() == RpuUnaryPrecision::BASE);
    base.check_table_admission(false);
    reject_unchanged(base, [&] { base.check_table_admission(true); });
    Model model;
    model.set_high_precision(true);
    assert(model.high_precision_ && model.high_precision_bound_ && model.generation == 6);
    reject_unchanged(model, [&] { model.set_high_precision(false); });
    reject_unchanged(model, [&] { model.check_table_admission(false); });
    model.check_table_admission(true);
    model.last_chunk = 31;
    reject_unchanged(model, [&] { model.check_table_admission(true); });
    model.last_chunk = 0;
    assert(model.unary_precision() == RpuUnaryPrecision::HIGH);
    assert(model.rmsnorm_route(62,128) == RpuRmsNormSpmRoute::NEWTON);
    assert(model.rmsnorm_route(31,128) == RpuRmsNormSpmRoute::NEWTON);
    model.emit_norm_shift(256, 512, 768, 1024, 31);
    assert(fused_calls == 1 && norm_calls == 1 && add_calls == 1 && last_norm == RpuRmsNormSpmRoute::NEWTON);
    model.gates_tanh_precomputed_ = true;
    model.consume_high_precision_route();
    assert(model.context.site == RHINO_HIGH_PRECISION_SITE && model.context.selector == 1 && model.context.flags == 1);
    assert(model.context.args == std::vector<int64_t>({1,31,1024,3072,18,10,16,8,128,6}));
    reject_unchanged(base, [&] { base.set_high_precision_fusions(true); });
    payload_ready = false;
    reject_unchanged(model, [&] { model.set_high_precision_fusions(true); });
    assert(!model.high_precision_fusions_ && !model.high_precision_fusions_bound_);
    payload_ready = true;
    model.set_high_precision_fusions(true);
    assert(model.high_precision_fusions_ && model.high_precision_fusions_bound_ && model.generation == 7);
    model.emit_norm_shift(256,512,768,1024,31);
    assert(high_fused_calls == 1 && norm_calls == 1 && add_calls == 1);
    model.consume_high_precision_route();
    assert(model.context.flags == 3);
    reject_unchanged(model, [&] { model.set_high_precision_fusions(false); });
    Model disabled;
    disabled.full_action_w8a16_ = false; payload_ready = false;
    int checks = payload_checks;
    disabled.set_high_precision(false);
    assert(!disabled.high_precision_ && disabled.high_precision_bound_ && payload_checks == checks);

    payload_ready = true;
    for (int damage = 0; damage < 9; ++damage) {
        Model bad;
        bad.set_high_precision(true); bad.cold_config_.aligned_kv = true;
        if (damage == 0) graph_active = true;
        if (damage == 1) transaction_active = true;
        if (damage == 2) bad.last_chunk = 31;
        if (damage == 3) bad.denoise_loop_weights_ready_ = true;
        if (damage == 4) bad.denoise_adarms_tables_ready_ = true;
        if (damage == 5) bad.high_precision_bound_ = false;
        if (damage == 6) bad.high_precision_ = false;
        if (damage == 7) bad.cold_config_.aligned_kv = false;
        if (damage == 8) bad.vector_k_norm_bound_ = true;
        reject_unchanged(bad, [&] { bad.set_vector_k_norm(true); });
        graph_active = transaction_active = false;
    }
    Model scalar;
    assert(scalar.k_norm_rows(17,2,64) == 34);
    events.clear(); norms.clear();
    scalar.emit_qk_norm(31,1,128,7);
    assert(events == std::vector<std::string>({"q_norm","k_norm"}));
    assert(norms[0].rows == 62 && norms[1].rows == 31 &&
           norms[0].route == RpuRmsNormSpmRoute::BASE && norms[1].route == RpuRmsNormSpmRoute::BASE);
    scalar.set_high_precision(true);
    scalar.set_vector_k_norm(false);
    assert(!scalar.vector_k_norm_ && scalar.vector_k_norm_bound_ && scalar.high_precision_route_flags() == 0);
    reject_unchanged(scalar, [&] { scalar.set_vector_k_norm(true); });
    events.clear(); norms.clear();
    scalar.emit_qk_norm(31,1,128,7);
    assert(events == std::vector<std::string>({"q_norm","k_norm"}));
    assert(norms[0].rows == 62 && norms[1].rows == 31 &&
           norms[0].route == RpuRmsNormSpmRoute::NEWTON && norms[1].route == RpuRmsNormSpmRoute::NEWTON);

    Model vector;
    vector.set_high_precision(true); vector.cold_config_.aligned_kv = true;
    checks = payload_checks;
    vector.set_vector_k_norm(true);
    assert(vector.vector_k_norm_ && vector.vector_k_norm_bound_ && vector.generation == 7 && payload_checks == checks);
    reject_unchanged(vector, [&] { vector.set_vector_k_norm(false); });
    assert(vector.k_norm_rows(31,1,128) == 32 && vector.high_precision_route_flags() == 4);
    events.clear(); norms.clear(); k_tail_zero = false;
    vector.emit_qk_norm(31,1,128,7);
    assert(events == std::vector<std::string>({"zero_k","zero_v","q_norm","k_norm"}));
    assert(norms.size() == 2 && norms[0].rows == 62 && norms[1].rows == 32 &&
           norms[0].cols == 128 && norms[1].cols == 128 &&
           norms[0].route == RpuRmsNormSpmRoute::NEWTON && norms[1].route == RpuRmsNormSpmRoute::NEWTON_V32_GRID8);
    assert(norms[0].input == 0x1000 && norms[0].output == 0x8000 && norms[0].weight == 0x37000);
    assert(norms[1].input == 0x2000 && norms[1].output == 0xa000 && norms[1].weight == 0x37100);
    for (int damage = 0; damage < 5; ++damage) {
        Model bad = vector;
        if (damage == 3) bad.high_precision_ = false;
        if (damage == 4) bad.cold_config_.aligned_kv = false;
        events.clear(); norms.clear();
        reject_unchanged(bad, [&] { bad.emit_qk_norm(damage == 0 ? 30 : 31,
            damage == 1 ? 2 : 1, damage == 2 ? 64 : 128, 7); });
        assert(events.empty() && norms.empty());
    }
    vector.gates_tanh_precomputed_ = true;
    vector.set_high_precision_fusions(true);
    vector.consume_high_precision_route();
    assert(vector.context.flags == 7);

    for (int damage = 0; damage < 8; ++damage) {
        Model bad;
        bad.set_high_precision(true);
        if (damage == 0) graph_active = true;
        if (damage == 1) transaction_active = true;
        if (damage == 2) bad.last_chunk = 31;
        if (damage == 3) bad.denoise_loop_weights_ready_ = true;
        if (damage == 4) bad.denoise_adarms_tables_ready_ = true;
        if (damage == 5) bad.high_precision_bound_ = false;
        if (damage == 6) bad.high_precision_ = false;
        if (damage == 7) bad.vector_q_norm_bound_ = true;
        reject_unchanged(bad, [&] { bad.set_vector_q_norm(true); });
        graph_active = transaction_active = false;
    }
    assert(scalar.q_norm_rows(17,3,64) == 51);
    scalar.set_vector_q_norm(false);
    assert(!scalar.vector_q_norm_ && scalar.vector_q_norm_bound_);
    reject_unchanged(scalar, [&] { scalar.set_vector_q_norm(true); });
    Model qvector;
    qvector.set_high_precision(true);
    checks = payload_checks;
    qvector.set_vector_q_norm(true);
    assert(qvector.vector_q_norm_ && qvector.vector_q_norm_bound_ && qvector.generation == 7 &&
           !qvector.vector_k_norm_ && !qvector.cold_config_.aligned_kv && payload_checks == checks);
    reject_unchanged(qvector, [&] { qvector.set_vector_q_norm(false); });
    assert(qvector.q_norm_rows(31,2,128) == 64 && qvector.high_precision_route_flags() == 8);
    events.clear(); norms.clear(); q_tail_zero = false;
    qvector.emit_qk_norm(31,1,128,7);
    assert(events == std::vector<std::string>({"zero_q","q_norm","k_norm"}));
    assert(norms.size() == 2 && norms[0].rows == 64 && norms[1].rows == 31 &&
           norms[0].route == RpuRmsNormSpmRoute::NEWTON_V32_GRID8 && norms[1].route == RpuRmsNormSpmRoute::NEWTON);
    for (int damage = 0; damage < 4; ++damage) {
        Model bad = qvector;
        if (damage == 1) bad.local_q_heads_ = 1;
        if (damage == 3) bad.high_precision_ = false;
        events.clear(); norms.clear();
        reject_unchanged(bad, [&] { bad.emit_qk_norm(damage == 0 ? 30 : 31,
            1, damage == 2 ? 64 : 128, 7); });
        assert(events.empty() && norms.empty());
    }
    // Both switches preserve independent admission and clear all tails before either norm.
    vector.set_vector_q_norm(true);
    events.clear(); norms.clear(); k_tail_zero = q_tail_zero = false;
    vector.emit_qk_norm(31,1,128,7);
    assert(events == std::vector<std::string>({"zero_k","zero_v","zero_q","q_norm","k_norm"}));
    assert(norms[0].rows == 64 && norms[1].rows == 32 &&
           norms[0].route == RpuRmsNormSpmRoute::NEWTON_V32_GRID8 && norms[1].route == RpuRmsNormSpmRoute::NEWTON_V32_GRID8);
    vector.consume_high_precision_route();
    assert(vector.context.flags == 15);

    for (int damage = 0; damage < 9; ++damage) {
        Model bad;
        bad.set_high_precision(true);
        if (damage == 0) graph_active = true;
        if (damage == 1) transaction_active = true;
        if (damage == 2) bad.last_chunk = 31;
        if (damage == 3) bad.denoise_loop_weights_ready_ = true;
        if (damage == 4) bad.denoise_adarms_tables_ready_ = true;
        if (damage == 5) bad.high_precision_bound_ = false;
        if (damage == 6) bad.high_precision_ = false;
        if (damage == 7) bad.partial_rope_bound_ = true;
        if (damage == 8) rope_payload_ready = false;
        reject_unchanged(bad, [&] { bad.set_partial_rope(true); });
        graph_active = transaction_active = false; rope_payload_ready = true;
    }
    Model rope_base;
    rope_base.set_high_precision(true);
    rope_payload_ready = false; checks = rope_payload_checks;
    rope_base.set_partial_rope(false);
    assert(!rope_base.partial_rope_ && rope_base.partial_rope_bound_ &&
           rope_payload_checks == checks && rope_base.rope_route_selector() == 1 &&
           rope_base.high_precision_route_flags() == 0);
    reject_unchanged(rope_base, [&] { rope_base.set_partial_rope(true); });
    rope_payload_ready = true;
    Model partial;
    partial.set_high_precision(true); checks = rope_payload_checks;
    partial.set_partial_rope(true);
    assert(partial.partial_rope_ && partial.partial_rope_bound_ && partial.generation == 7 &&
           rope_payload_checks == checks + 1 && partial.rope_route_selector() == 2 &&
           partial.high_precision_route_flags() == 16);
    reject_unchanged(partial, [&] { partial.set_partial_rope(false); });
    // Both Q and K use all 128 channels, the same DDR tables and the logical
    // position, even when the physical prefix is separately padded (45 != 48).
    for (int64_t position : {0, 45, 230}) {
        for (int64_t heads : {1, 2}) {
            const uint32_t input = heads == 2 ? 0x8000 : 0xa000;
            const uint32_t output = heads == 2 ? 0x1000 : 0x2000;
            ropes.clear();
            rpu_launch_rhinovla_rope_spm_kernel(input,output,rope_base.cos_ref_,rope_base.sin_ref_,
                31,heads,128,position,rope_base.partial_rope_,8);
            rpu_launch_rhinovla_rope_spm_kernel(input,output,partial.cos_ref_,partial.sin_ref_,
                31,heads,128,position,partial.partial_rope_,8);
            assert(ropes.size() == 2 && !ropes[0].partial && ropes[1].partial);
            for (const auto& call : ropes) {
                assert(call.input == input && call.output == output && call.position == position &&
                       call.rows == 31 && call.heads == heads && call.hd == 128 && call.rotary == 128 &&
                       call.cores == 8 && call.cos == partial.cos_ref_.data_ptr<c10::Half>() &&
                       call.sin == partial.sin_ref_.data_ptr<c10::Half>());
            }
            assert(ropes[1].cos_owner == &partial.cos_ref_ && ropes[1].sin_owner == &partial.sin_ref_);
        }
    }
    for (int damage = 0; damage < 5; ++damage) {
        Model bad = partial;
        ropes.clear();
        reject_unchanged(bad, [&] { rpu_launch_rhinovla_rope_spm_kernel(0x8000,0x1000,
            bad.cos_ref_,bad.sin_ref_,damage == 0 ? 32 : 31,damage == 1 ? 3 : 2,
            damage == 2 ? 64 : 128,45,true,damage == 3 ? 0 : damage == 4 ? 9 : 8); });
        assert(ropes.empty());
    }
    vector.set_partial_rope(true);
    vector.consume_high_precision_route();
    assert(vector.context.flags == 31 && vector.rope_route_selector() == 2);
}
'''.replace("__SITE_VALUE__", site).replace("ROPE_ENUM", rope_enum).replace("ROPE_WRAPPER", rope_wrapper).replace("METHODS", methods).replace("TABLE_GUARD", table_guard)
    cpp, executable = tmp_path / "precision.cpp", tmp_path / "precision"
    cpp.write_text(program)
    build = subprocess.run([compiler, "-std=c++17", "-O0", str(cpp), "-o", str(executable)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(executable)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
