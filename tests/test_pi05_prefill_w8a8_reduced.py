"""Execute the real Pi reduced W8A8 selector, owner, manifest and MLP emission without RPU allocation."""
from pathlib import Path

from test_siglip_attention_dry_live_contract import _function, _run_native


ROOT = Path(__file__).resolve().parents[1]


def test_tile_picks_and_register_abi(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    code = r'''
#include "ops/rpu_pi_wall_tiling.h"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
using namespace rpu_pl_tiling;
struct Kernel {
    std::map<int, uint16_t> regs;
    void set_regs(int index, uint16_t value) { regs[index] = value; }
    void set_regs(int, uint32_t) = delete;
};
int main() {
    // Reduced P800/P544 chunks are M32 and M256 at K2048; the local N are
    // 16384/4 and 16512/6.
    for (int n : {4096, 2752}) {
        for (int m : {16, 32}) {
            auto s = rpu_pi_wall::select(rpu_pi_wall::Operation::A8Online, WeightType::Int8, true, m, n, 2048);
            assert(s.tile.m_tile == 48 && s.tile.n_tile == 128 && s.tile.k_tile == 512);
            assert(std::string(rpu_pi_wall::candidate_names[s.ordinal]) == "linear_w8a8_online_i32_m48n128k512");
        }
        {
            auto s = rpu_pi_wall::select(rpu_pi_wall::Operation::A8Online, WeightType::Int8, true, 64, n, 2048);
            assert(s.tile.m_tile == 112 && s.tile.n_tile == 128 && s.tile.k_tile == 256);
        }
        {
            auto s = rpu_pi_wall::select(rpu_pi_wall::Operation::A8Online, WeightType::Int8, true, 128, n, 2048);
            assert(s.tile.m_tile == 176 && s.tile.n_tile == 128 && s.tile.k_tile == 128);
        }
        {
            auto s = rpu_pi_wall::select(rpu_pi_wall::Operation::A8Online, WeightType::Int8, true, 192, n, 2048);
            assert(s.tile.m_tile == 240 && s.tile.n_tile == 128 && s.tile.k_tile == 64);
        }
        {
            auto s = rpu_pi_wall::select(rpu_pi_wall::Operation::A8Online, WeightType::Int8, true, 256, n, 2048);
            assert(s.tile.m_tile == 272 && s.tile.n_tile == 112 && s.tile.k_tile == 64);
            assert(std::string(rpu_pi_wall::candidate_names[s.ordinal]) == "linear_w8a8_online_i32_m272n112k64");
        }
    }

    const auto sel = rpu_pi_wall::select(rpu_pi_wall::Operation::A8Online, WeightType::Int8, true, 256, 2752, 2048);
    Kernel kernel;
    set_parallel_linear_regs(kernel, {0x10000, 0x123456700ull, 0x20000, 0, 256, 2752, 2048,
                                      0, 6, 4096, 5504, 0xabcdef100ull, 1, 1, 2});
    kernel.set_regs(50, static_cast<uint16_t>(0x57f0));
    kernel.set_regs(51, static_cast<uint16_t>(0xd7f0));
    kernel.set_regs(64, sel.grid_x);
    kernel.set_regs(65, sel.grid_y);
    kernel.set_regs(66, static_cast<uint16_t>(1));
    const std::map<int, uint16_t> expected{
        {0, 0}, {1, 1}, {2, 0x4567}, {3, 0x0123}, {4, 0}, {5, 2}, {6, 0}, {7, 0},
        {8, 1}, {9, 256}, {10, 2752}, {11, 0}, {12, 2048}, {13, 128}, {14, 0}, {15, 6},
        {16, 4096}, {17, 0}, {18, 5504}, {19, 0}, {20, 0xdef1}, {21, 0x0abc},
        {22, 0}, {23, 0}, {24, 0}, {25, 0}, {26, 2}, {27, 1}, {28, 0}, {29, 0},
        {50, 0x57f0}, {51, 0xd7f0}, {64, 25}, {65, 1}, {66, 1}};
    assert(kernel.regs == expected);
    std::cout << "W8A8 picks and registers: PASS\n";
}
'''
    assert "PASS" in _run_native(tmp_path, code, torch)


def test_reduced_owner_manifest_and_emission(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    source = (ROOT / "src/fused/rpu_gemma_model.cpp").read_text()
    cache = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    names = ("bool pi05_prefill_a8_reduced(",
             "std::vector<int64_t> pi05_prefill_a8_reduced_arguments(",
             "bool pi05_prefill_a8_reduced_owner(",
             "void pi05_prefill_a8_reduced_validate_manifest(",
             "FmbLinearAccumulationPolicy pi05_prefill_linear_accumulation(")
    methods = "\n".join(_function(source, "    " + name) for name in names)
    # Only the device boundary is substituted; geometry, tensor dtype/shape,
    # every cold flag and payload admission stay real.
    methods = methods.replace("at::kPrivateUse1", "at::kCPU")
    constants = "\n".join(line for line in source.splitlines()
                          if line.startswith("constexpr int64_t PI05_GATEUP_A8_REDUCED_"))
    dry = _function(source, "    FmbPhysicalExecutionManifest physical_manifest_for_candidate(")
    begin = dry.index("        TORCH_CHECK(!pi05_prefill_a8_reduced() ||")
    guard = dry[begin:dry.index(");", begin) + 2]
    append = _function(dry, "        if (pi05_prefill_a8_reduced()) {")
    emission = _function(source, "    void emit_pi05_prefill_a8_reduced_mlp(")
    availability = _function(cache, "class KernelPayloadAvailability final")
    kernel_ids = _function(cache, "enum class KernelId")
    pi_wall_id = _function(cache, "inline KernelId pi_wall_kernel_id(")
    pi_wall_loaded = _function((ROOT / "src/ops/rpu_pi_wall_runtime.h").read_text(),
                               "inline bool rpu_pi_wall_operation_loaded(")
    code = r'''
#include "core/fused_model_base.h"
#include "ops/rpu_pi_wall_tiling.h"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <array>
#include <atomic>
#include <cassert>
#include <iostream>
#include <map>
#include <sstream>
using namespace v3;
''' + kernel_ids + ";\n" + pi_wall_id + "\n" + availability + r''';
struct KernelCache {
    KernelPayloadAvailability availability;
    static KernelCache& instance() { static KernelCache cache; return cache; }
    bool has_loaded(KernelId id) const { return availability.has_loaded(id); }
};
''' + pi_wall_loaded + "\n" + constants + r'''
enum class ValuOpType { ADD, MUL };
enum class GeluMode { TANH };
static std::vector<std::string> events;
template <class... Args> static void record(const Args&... args) {
    std::ostringstream line;
    ((line << args << ' '), ...);
    events.push_back(line.str());
}
static void rpu_launch_linear_w8a8_spm_to_spm_kernel(uint32_t x, const at::Tensor& weight, uint32_t y,
    const at::Tensor& scale, int64_t m, int64_t n, int64_t k, int cores) {
    assert(weight.scalar_type() == at::kChar && scale.scalar_type() == at::kFloat);
    record("w8a8", x, y, m, n, k, cores);
}
static void rpu_launch_eltwise_unary_spm_kernel(uint32_t x, uint32_t y, int64_t elems, ValuOpType, GeluMode, int cores) {
    record("gelu", x, y, elems, cores);
}
static void rpu_launch_eltwise_binary_spm_kernel(uint32_t a, uint32_t b, uint32_t y, int64_t elems, ValuOpType op, c10::Half, int cores) {
    assert(op == ValuOpType::MUL);
    record("mul", a, b, y, elems, cores);
}
static void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t x, const at::Tensor&, uint32_t y, int64_t m, int64_t k, int64_t n,
    int, int cores, int, const at::Tensor&, int, int, bool acc32) {
    assert(!acc32);
    record("down", x, y, m, k, n, cores);
}
static void rpu_launch_all_reduce_sum_residual_kernel(uint32_t x, uint32_t residual, uint32_t y, int64_t m, int64_t h, int reduce, int cores, bool xor3) {
    assert(!xor3);
    record("ring", x, residual, y, m, h, reduce, cores);
}
namespace v3 {
struct Owner {
    bool linear_acc32_=false;
    bool pi05_prefill_a8_=true, pi05_prefill_gateup_pair_=false;
    bool pi05_prefill_o_pair_=false;
    bool pi05_prefill_qkv_pair_=false, pi05_prefill_down_pair_=false;
    bool pi05_prefill_owner_norm_=false, pi05_prefill_kv1_pair_direct_=false;
    bool pi05_prefill_kv_only_=false, pi05_ring_xor3_opt_in_=false;
    int cores, attention_tp=4, mlp_cores, layers=18, dma_policy=0;
    int64_t hidden=2048, intermediate, logical_intermediate_size_=16384, q_heads=8, kv_heads=4, hd=256;
    struct Weights { at::Tensor gate_proj_w,up_proj_w,down_proj_w,gate_ws,up_ws,down_ws; };
    std::vector<Weights> layer_weights_;
    std::vector<std::array<at::Tensor,2>> pi05_a8_scales_;
    mutable FmbPhysicalManifestConsumer consumer;
    struct Ctx {
        const FmbPhysicalManifestConsumer& consumer;
        const FmbRouteManifestEntry& consume_physical_route(FmbRouteFamily family, int64_t site, int64_t selector,
            int64_t flags, at::IntArrayRef arguments = {}, int64_t invocation = 0) const {
            return consumer.consume_route(family, site, selector, flags, arguments, invocation);
        }
    };
    Ctx ctx() const { return {consumer}; }
    uint32_t addr(int, const char* name) const {
        static const std::map<std::string,uint32_t> slots{{"residual1",0x1000},{"residual2",0x2000},
            {"gate",0x3000},{"up",0x4000},{"down",0x5000}};
        return slots.at(name);
    }
    int num_cores() const { return cores; }
    int attn_tp() const { return attention_tp; }
    int mlp_tp() const { return mlp_cores; }
    int num_layers() const { return layers; }
    int64_t hidden_size() const { return hidden; }
    int64_t intermediate_size() const { return intermediate; }
    int64_t num_q_heads() const { return q_heads; }
    int64_t num_kv_heads() const { return kv_heads; }
    int64_t head_dim() const { return hd; }
    int pi05_prefill_dma_policy() const { return dma_policy; }
    explicit Owner(int budget) : cores(budget), mlp_cores(budget), intermediate(budget == 6 ? 16512 : 16384) {
        const auto weight=at::empty({intermediate,2048},at::TensorOptions().dtype(at::kChar));
        const auto scale=at::ones({intermediate},at::TensorOptions().dtype(at::kHalf));
        const auto promoted=at::ones({intermediate},at::TensorOptions().dtype(at::kFloat));
        layer_weights_.assign(18,{weight,weight,weight,scale,scale,scale});
        pi05_a8_scales_.assign(18,{promoted,promoted});
    }
''' + methods + r'''
    FmbPhysicalExecutionManifest dry(const FmbThreeStageChunkPlan& plan, int64_t rows) const {
''' + guard + r'''
        FmbPhysicalExecutionManifest manifest;
        manifest.state=FmbPhysicalManifestState::COMPLETE;
        manifest.logical_length=rows; manifest.physical_length=rows;
        manifest.execution_padding_rows=0;
        manifest.kv_logical_length=rows;
        manifest.kv_insert_physical_rows=rows;
        manifest.graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;
        manifest.linear_accumulation=pi05_prefill_linear_accumulation();
        auto append=[&](FmbRouteFamily family,int64_t site,int64_t selector,
                       int64_t flags=0,std::vector<int64_t> args={},int64_t invocation=0) {
            manifest.routes.push_back({site,family,selector,flags,std::move(args),invocation});
        };
''' + append + r'''
        append_fmb_shared_runtime_routes(manifest, plan, hidden_size(),
            FMB_SHARED_MLP_AUTO_TILE | FMB_SHARED_MLP_RING_REDUCE, 1, false, num_cores(), mlp_tp());
        return manifest;
    }
''' + emission + r'''
};
} // namespace v3
template<class F> void rejects(F run) {
    bool failed=false;
    try { run(); } catch (const c10::Error&) { failed=true; }
    assert(failed);
}
static void publish(bool tiles, int missing=-1) {
    using Availability=KernelPayloadAvailability;
    Availability::ExtendedPayloads extended;
    int i=0;
    for (const auto& p : rpu_pi_wall::profiles) {
        if (p.operation != rpu_pi_wall::Operation::A8Online ||
            p.weight != rpu_pl_tiling::WeightType::Int8) continue;
        if (tiles && i!=missing)
            extended.set(static_cast<size_t>(pi_wall_kernel_id(p.ordinal))-Availability::kExtendedBegin);
        ++i;
    }
    assert(i==28);
    KernelCache::instance().availability.publish(true,true,false,false,false,false,extended);
}
static void budget(int cores) {
    publish(true);
    Owner owner(cores);
    assert(owner.pi05_prefill_a8_reduced() && owner.pi05_prefill_a8_reduced_owner());
    const int64_t n=owner.intermediate_size(), local_n=n/cores;
    // 3-camera P800 at T32: three M256 chunks and the M32 tail.
    FmbThreeStageChunkPlan plan;
    plan.chunk_mode=ChunkMode::KV_FIRST;
    plan.compute.chunks={{0,0,256,256},{1,256,256,512},{2,512,256,768},{3,768,32,800}};
    plan.qkv.chunks=plan.compute.chunks;
    auto manifest=owner.dry(plan,800);
    assert(manifest.routes.size()==4+1+4);
    const auto m256=owner.pi05_prefill_a8_reduced_arguments(256);
    const auto m32=owner.pi05_prefill_a8_reduced_arguments(32);
    assert((m256==std::vector<int64_t>{1,256,n,2048,cores,272,112,64,17}));
    assert((m32==std::vector<int64_t>{1,32,n,2048,cores,48,128,512,17}));
    owner.pi05_prefill_a8_reduced_validate_manifest(manifest);
    validate_fmb_physical_manifest(manifest,800,0);
    owner.consumer.reset(manifest);
    for (const auto& chunk : plan.compute.chunks) {
        events.clear();
        owner.emit_pi05_prefill_a8_reduced_mlp(3,chunk,"residual1");
        const auto elems=chunk.len*local_n;
        const auto text=[](auto... args){ std::ostringstream line; ((line<<args<<' '),...); return line.str(); };
        const std::vector<std::string> expected{
            text("w8a8",0x1000,0x3000,chunk.len,n,2048,cores),
            text("gelu",0x3000,0x3000,elems,cores),
            // Up reads the same input Gate read: the tile does not write X.
            text("w8a8",0x1000,0x4000,chunk.len,n,2048,cores),
            text("mul",0x3000,0x4000,0x3000,elems,cores),
            text("down",0x3000,0x5000,chunk.len,2048,n,cores),
            text("ring",0x5000,0x2000,0x1000,chunk.len,2048,cores,cores)};
        assert(events==expected);
    }
    owner.consumer.require_all_consumed();

    // A descriptor with another budget's arguments is foreign.
    auto foreign=manifest;
    foreign.routes.front().arguments[4]=8;
    rejects([&]{owner.pi05_prefill_a8_reduced_validate_manifest(foreign);});
    auto acc=manifest;
    acc.linear_accumulation=FmbLinearAccumulationPolicy::ACC16;
    rejects([&]{owner.pi05_prefill_a8_reduced_validate_manifest(acc);});
    rejects([&]{owner.emit_pi05_prefill_a8_reduced_mlp(3,plan.compute.chunks[0],"residual2");});

    for(auto flag : {&owner.linear_acc32_,&owner.pi05_prefill_o_pair_,&owner.pi05_prefill_qkv_pair_,
        &owner.pi05_prefill_down_pair_,&owner.pi05_prefill_owner_norm_,&owner.pi05_prefill_kv1_pair_direct_,
        &owner.pi05_prefill_kv_only_,&owner.pi05_ring_xor3_opt_in_}) {
        if(flag==&owner.linear_acc32_) continue;
        *flag=true;
        assert(!owner.pi05_prefill_a8_reduced_owner());
        rejects([&]{owner.dry(plan,800);});
        rejects([&]{owner.pi05_prefill_a8_reduced_validate_manifest(manifest);});
        *flag=false;
    }
    for(auto value : {&owner.attention_tp,&owner.mlp_cores,&owner.layers,&owner.dma_policy}) {
        const int saved=*value;*value=5;
        assert(!owner.pi05_prefill_a8_reduced_owner());*value=saved;
    }
    for(auto value : {&owner.hidden,&owner.logical_intermediate_size_,&owner.q_heads,&owner.kv_heads,&owner.hd}) {
        const int64_t saved=*value;*value=saved+1;
        assert(!owner.pi05_prefill_a8_reduced_owner());*value=saved;
    }
    auto& tail=owner.layer_weights_.back();
    const auto saved_weight=tail.up_proj_w,saved_scale=tail.gate_ws;
    tail.up_proj_w=at::empty({n,2048},at::TensorOptions().dtype(at::kHalf));
    assert(!owner.pi05_prefill_a8_reduced_owner());tail.up_proj_w=saved_weight;
    tail.up_proj_w=at::empty({n,1024},at::TensorOptions().dtype(at::kChar));
    assert(!owner.pi05_prefill_a8_reduced_owner());tail.up_proj_w=saved_weight;
    tail.gate_ws=at::ones({n},at::TensorOptions().dtype(at::kFloat));
    assert(!owner.pi05_prefill_a8_reduced_owner());tail.gate_ws=saved_scale;
    // An unavailable tile cannot be admitted: the REF is the cold capability.
    for(int missing=0;missing!=28;++missing) {
        publish(true,missing);
        assert(!owner.pi05_prefill_a8_reduced_owner());
    }
    publish(true);
    assert(owner.pi05_prefill_a8_reduced_owner());
    // A budget that is not four or six has no reduced owner.
    owner.cores=5;owner.mlp_cores=5;
    assert(owner.pi05_prefill_a8_reduced() && !owner.pi05_prefill_a8_reduced_owner());
    // Eight cores keep the fixed owners: this predicate must not claim them.
    owner.cores=8;owner.mlp_cores=8;
    assert(!owner.pi05_prefill_a8_reduced());
    owner.cores=cores;owner.mlp_cores=cores;
    // Paired Gate/Up names the eight-core owner and is never the reduced form.
    owner.pi05_prefill_gateup_pair_=true;
    assert(!owner.pi05_prefill_a8_reduced());
    owner.pi05_prefill_gateup_pair_=false;
    owner.pi05_prefill_a8_=false;
    assert(!owner.pi05_prefill_a8_reduced());
    // Without the A8 selection a stale site is still foreign.
    rejects([&]{owner.pi05_prefill_a8_reduced_validate_manifest(manifest);});
}
int main() {
    at::set_num_threads(1);
    budget(4);
    budget(6);
    std::cout << "Pi reduced A8 owner/manifest/emission: PASS\n";
}
'''
    result = _run_native(tmp_path, code, torch,
                         extra_sources=(ROOT / "src/core/fmb_three_stage_chunk_plan.cpp",))
    assert "PASS" in result


def test_terminal_layer_keeps_the_w8_pipeline():
    # pi05_a8_scales_ covers every layer, so the runtime checks cannot catch a
    # reduced branch that also claims layer 17.
    source = (ROOT / "src/fused/rpu_gemma_model.cpp").read_text()
    assert "else if (pi05_prefill_a8_reduced() && !is_last_layer) {" in source


def test_launcher_is_broadcast_and_declares_both_ddr_operands():
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    launcher = _function(source, "void rpu_launch_linear_w8a8_spm_to_spm_kernel(")
    assert "queue->set_broadcast_mode(true);" in launcher
    assert "GraphDdrRegisterRole::Int8ModelWeight" in launcher
    assert "GraphDdrRegisterRole::Fp32PerChannelScale" in launcher
    assert "kernel->set_regs(50, static_cast<uint16_t>(0x57f0));" in launcher
    assert "kernel->set_regs(51, static_cast<uint16_t>(0xd7f0));" in launcher
    assert "kernel->set_regs(64, selected.grid_x);" in launcher
