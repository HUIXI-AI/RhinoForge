"""Run production launch/admission bodies with host-only queue/SPM recorders.

This verifies the emitted core domain and register/SCM geometry. It does not
replace numerical tests of the generated kernels on the board.
"""

from pathlib import Path
from pi_wall_host_catalog import with_actual_linear_contracts
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _function(path: str, signature: str) -> str:
    source = (ROOT / path).read_text()
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    position = opening + 1
    while depth:
        depth += (source[position] == "{") - (source[position] == "}")
        position += 1
    return source[start:position]


def test_plain_qwen_rope_caller_preserves_attention_core_domain(tmp_path):
    compiler = shutil.which("g++")
    assert compiler, "production RoPE caller checks require g++"
    wrapper = _function("src/ops/rpu_rope.cpp", "void rpu_launch_rope_spm_kernel(\n    uint32_t")
    header = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    helper = header.index("void emit_qk_rope(")
    start = header.index("rpu_launch_rope_spm_kernel(", helper)
    # Compile the exact plain-1D branch call, retaining the real default-eight
    # declaration: omitting tp recreates the rejected eight-core launch.
    call = header[start:header.index(";", start) + 1]
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    start = decls.index("void rpu_launch_rope_spm_kernel(\n    uint32_t")
    declaration = decls[start:decls.index(";", start) + 1]
    harness = r'''
#include <cassert>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
#define ROPE_NUM_CORES 8
namespace c10 { using Half = uint16_t; }
enum class KernelId { ROPE };
struct Kernel_t {
    std::map<int, uint16_t> regs;
    void reset_regs() { regs.clear(); }
    void set_regs(int slot, uint16_t value) { regs[slot] = value; }
} kernel;
Kernel_t* get_kernel(KernelId) { return &kernel; }
#define GET_KERNEL(id) get_kernel(id)
int flushes = 0;
void rpu_ddr_flush(c10::Half*) { ++flushes; }
uint64_t RpuGetDevAddr(c10::Half* ptr) { return reinterpret_cast<uint64_t>(ptr); }
int budget = 8, requested = 0;
struct Queue {
    bool broadcast = false;
    std::vector<uint8_t> cores;
    std::vector<uint16_t> grid;
    void set_broadcast_mode(bool enabled) { broadcast = enabled; }
    void enqueu_kernel(Kernel_t&, std::vector<uint16_t> dims, std::vector<uint8_t> ids) {
        assert(broadcast && ids.size() <= static_cast<size_t>(budget));
        assert(ids.size() == static_cast<size_t>(requested));
        for (size_t i = 0; i < ids.size(); ++i) assert(ids[i] == i);
        cores = ids; grid = dims;
    }
} queue;
Queue* get_queue(int count) { requested = count; queue.broadcast = false; return &queue; }
#define GET_QUEUE(count) get_queue(count)
__DECLARATION__
__WRAPPER__
struct Tensor {
    c10::Half data[16]{};
    template<class T> T* data_ptr() { return data; }
};
struct Decoder {
    Tensor cos_, sin_;
    uint32_t addr(int, const char* slot) const { return std::string(slot) == "input" ? 0x10000 : 0x20000; }
    void plain_rope(int tp, int64_t seq_len, int64_t n_heads, int64_t cos_sin_start) {
        const char* in_slot = "input";
        const char* out_slot = "output";
        const uint32_t spm_offset_bytes = 256;
        const int64_t dim = 128;
        __CALL__
    }
};
int main() {
    Decoder model;
    // The six-core decoder uses TP4 for Q/K; exercise prefill and decode,
    // including replay position changes and return to the default-eight path.
    for (int cores : {8, 4, 4, 8}) {
        budget = cores == 4 ? 6 : 8;
        for (int64_t rows : {16, 1}) {
            model.plain_rope(cores, rows, 16 / cores, 32);
            assert(requested == cores && kernel.regs[21] == cores);
            assert(queue.cores.size() == static_cast<size_t>(cores));
            assert(kernel.regs[0] == 32 && kernel.regs[2] == 128);
            assert(kernel.regs[3] == 16 / cores && queue.grid[0] == rows);
            assert((kernel.regs[4] | (uint32_t(kernel.regs[5]) << 16)) == 0x10100);
            assert((kernel.regs[10] | (uint32_t(kernel.regs[11]) << 16)) == 0x20100);
        }
    }
    // A legacy caller that intentionally omits the trailing argument stays8.
    budget = 8;
    rpu_launch_rope_spm_kernel(0x10000, 0x20000, model.cos_.data, model.sin_.data, 1, 2, 128, 0);
    assert(requested == 8 && kernel.regs[21] == 8);
    for (const auto& args : std::vector<std::vector<int64_t>>{
            {0, 2, 128, 0, 4}, {65536, 2, 128, 0, 4},
            {1, 0, 128, 0, 4}, {1, 65536, 128, 0, 4},
            {1, 2, 0, 0, 4}, {1, 2, 127, 0, 4}, {1, 2, 65536, 0, 4},
            {1, 2, 128, -1, 4}, {2, 2, 128, UINT32_MAX, 4},
            {1, 2, 128, 0, 0}, {1, 2, 128, 0, 9}}) {
        const int before = flushes;
        bool failed = false;
        try {
            rpu_launch_rope_spm_kernel(0x10000, 0x20000, model.cos_.data,
                model.sin_.data, args[0], args[1], args[2], args[3], args[4]);
        } catch (const std::runtime_error&) { failed = true; }
        assert(failed && flushes == before);
    }
}
'''
    for marker, value in (("__DECLARATION__", declaration),
                          ("__WRAPPER__", wrapper), ("__CALL__", call)):
        harness = harness.replace(marker, value)
    source = tmp_path / "qwen_rope_cores.cpp"
    source.write_text(with_actual_linear_contracts(harness))
    binary = tmp_path / "qwen_rope_cores"
    subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(ROOT / "src/core"), str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)



def test_pi05_gemma_calls_keep_compute_budget_and_physical_cache_separate(tmp_path):
    compiler = shutil.which("g++")
    assert compiler, "Gemma core-domain regression requires g++"
    gemma = _function("src/fused/rpu_gemma_model.cpp", "void build_layer_subgraph(")
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    prepare_name = "rpu_prepare_ring_all_reduce_input("
    sdpa_name = "rpu_launch_sdpa_spm_unified_kernel_v2("
    calls = {}
    for name in (prepare_name, sdpa_name):
        start = gemma.index(name)
        calls[name] = gemma[start:gemma.index(";", start) + 1]
    start = decls.index("void " + prepare_name)
    declaration = decls[start:decls.index(";", start) + 1]
    prepare = _function("src/core/rpu_runtime_extras.cpp", "void " + prepare_name)
    core_guard = _function("src/graph/graph_runtime_execute.cpp", "uint8_t validated_kernel_core_extent(")
    memset_name = "rpu_launch_memset_spm_multicore("
    memset_calls = []
    for signature in ("void Pi05DenoiseStepModel::emit_pre_layers_body(",
                      "void Pi05DenoiseStepModel::build_layer_subgraph("):
        body = _function("src/fused/rpu_pi05_denoise_step_model.cpp", signature)
        offset = 0
        while (offset := body.find(memset_name, offset)) >= 0:
            stop = body.index(";", offset) + 1
            memset_calls.append(body[offset:stop])
            offset = stop
    start = decls.index("void " + memset_name)
    memset_declaration = decls[start:decls.index(";", start) + 1]
    memset_helpers = "\n".join((
        _function("src/ops/rpu_memcpy.cpp", "static void compute_store_split("),
        _function("src/ops/rpu_memcpy.cpp", "void " + memset_name)))
    denoise = (ROOT / "src/fused/rpu_pi05_denoise_step_model.cpp").read_text()
    shared_header = (ROOT / "src/core/fused_model_base.h").read_text()
    append_name = "append_fmb_shared_runtime_routes("
    reduce_name = "fmb_ring_all_reduce_route_selector("
    dma_name = "rpu_launch_ddr_broadcast_spm_dma_mutable("
    route_declarations = []
    for signature in ("void " + append_name, "int64_t " + reduce_name):
        # The second overload accepts the Pi XOR3 flag. Retain its real default
        # core counts so an omitted owner/MLP argument reproduces the bug.
        start = shared_header.index(signature, shared_header.index(signature) + 1)
        route_declarations.append(shared_header[start:shared_header.index(";", start) + 1])
    start = decls.index("void " + dma_name)
    route_declarations.append("struct DeferredDmaChannels;\n" +
                              decls[start:decls.index(";", start) + 1])
    denoise_calls = []
    adarms = _function("src/fused/rpu_pi05_denoise_step_model.cpp",
                       "uint32_t Pi05DenoiseStepModel::adarms_gemv_to_spm(")
    for body, signature in ((denoise, append_name),
                            (denoise, "const int64_t expected_reduce = " + reduce_name),
                            (adarms, dma_name)):
        start = body.index(signature)
        denoise_calls.append(body[start:body.index(";", start) + 1])
    harness = r"""
#include <cassert>
#include <cstdint>
#include <optional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
namespace c10 { using Half = uint16_t; constexpr auto nullopt = std::nullopt; }
struct SpmAllocator { static constexpr uint64_t ALIGN = 256; };
constexpr int kRingCoreNum = 8;
struct GraphRuntimePolicy { int execution_core_count = 8; } policy;
__CORE_GUARD__
std::vector<uint8_t> core_ids(int count) {
    std::vector<uint8_t> result;
    for (int i = 0; i < count; ++i) result.push_back(i);
    return result;
}
struct MemsetCall { int cores; uint32_t address; uint32_t elements; };
std::vector<MemsetCall> memset_calls;
enum class KernelId { MEMSET_SPM_MULTI_CORE_V2 };
struct Kernel_t {
    std::map<int, uint16_t> regs;
    void reset_regs() { regs.clear(); }
    void set_regs(int index, uint16_t value) { regs[index] = value; }
} kernel;
#define GET_KERNEL(id) (&::kernel)
int requested_cores = 0;
struct Queue {
    bool broadcast = false;
    void set_broadcast_mode(bool enabled) { broadcast = enabled; }
    void enqueu_kernel(Kernel_t& kernel, std::vector<uint16_t> grid, std::vector<uint8_t> ids) {
        assert(broadcast && ids.size() == static_cast<size_t>(requested_cores));
        validated_kernel_core_extent(policy, ids);
        const auto u32 = [&](int lo) {
            return kernel.regs[lo] | (uint32_t(kernel.regs[lo + 1]) << 16);
        };
        assert(u32(4) == 0 && grid[1] == 1 && grid[2] == 1);
        memset_calls.push_back({requested_cores, u32(6), u32(0) * grid[0]});
    }
} queue;
Queue* get_queue(int cores) { requested_cores = cores; queue.broadcast = false; return &queue; }
#define GET_QUEUE(cores) get_queue(cores)
constexpr int SPM_BANK_SIZE = 32, MAX_BLOCKS_MEMCPY = 8;
constexpr int64_t STORE_INNER_STEP = 32256;
__MEMSET_DECLARATION__
__MEMSET_HELPERS__
int fill_count = 0, fill_cores = 0, sdpa_cores = 0, sdpa_cache_cores = 0;
void rpu_launch_fill_spm_kernel(uint32_t, int64_t elements, c10::Half value, int cores) {
    validated_kernel_core_extent(policy, core_ids(cores));
    assert(elements == 400 * 2048 && value == 0);
    ++fill_count; fill_cores = cores;
}
__DECLARATION__
__PREPARE__
// Record the real model call's two independent domains. Physical cache lanes
// affect DDR addressing; only the compute domain is checked as a launch budget.
void rpu_launch_sdpa_spm_unified_kernel_v2(
        int, int, int, std::optional<double>, uint32_t, uint32_t, uint32_t, uint32_t,
        int64_t, int64_t nq, int64_t nkv, int64_t hd, int64_t, int cores, int cache_cores) {
    validated_kernel_core_extent(policy, core_ids(cores));
    assert(nq == 8 && nkv == cores && hd == 256);
    sdpa_cores = cores; sdpa_cache_cores = cache_cores;
}
struct FmbPhysicalExecutionManifest { int64_t physical_length = 50; };
struct FmbThreeStageChunkPlan {};
constexpr uint32_t FMB_SHARED_LAYER_INPUT_DMA = 1, FMB_SHARED_MLP_AUTO_TILE = 2,
                   FMB_SHARED_MLP_RING_REDUCE = 4;
__ROUTE_DECLARATIONS__
int route_owner_cores = 0, route_mlp_cores = 0, reduce_cores = 0, table_dma_cores = 0;
void append_fmb_shared_runtime_routes(
        FmbPhysicalExecutionManifest&, const FmbThreeStageChunkPlan&,
        int64_t hidden, uint32_t mask, int64_t multiplier,
        bool xor3, int cores, int mlp_cores) {
    assert(hidden == 1024 && mask == 7 && multiplier == 1 && !xor3);
    route_owner_cores = cores; route_mlp_cores = mlp_cores;
}
int64_t fmb_ring_all_reduce_route_selector(int64_t rows, int64_t cols, bool xor3, int cores) {
    assert(rows == 50 && cols == 1024 && !xor3);
    reduce_cores = cores;
    return cores;
}
void rpu_launch_ddr_broadcast_spm_dma_mutable(
        const uint64_t* live_src, int64_t offset, int64_t elements, uint32_t address, int cores,
        DeferredDmaChannels* deferred, int64_t sync_elements) {
    assert(deferred == nullptr && sync_elements == 0);
    validated_kernel_core_extent(policy, core_ids(cores));
    assert(*live_src == 0x700000 && offset == 128 && elements == 3 * 1024 && address == 0x10000);
    table_dma_cores = cores;
}
struct Model {
    int owner_cores;
    int num_cores() const { return owner_cores; }
    int mlp_tp() const { return owner_cores; }
    int64_t hidden_size() const { return 1024; }
    bool use_pi05_ring_xor3(int64_t) const { return false; }
    uint32_t addr(int, const char*) const { return 0x10000; }
    struct Offset { uint32_t value = 0x10000; };
    Offset addr_offset(const char*) const { return {}; }
    int pi05_pair_owner_bytes() const { return 400 * 2048 / 8 * 2; }
    void emit() {
        const int tp = owner_cores == 8 ? 8 : 4;
        const int64_t seq_len = 400, h = 2048, nq = 8, nkv = tp, hd = 256;
        const int64_t total_kv_seq_len = 800;
        const int k_cache = 0, v_cache = 1;
        const uint32_t mask_offset = 0x20000;
        const bool pi05_prefill_o_pair_ = false, paired_down = false;
        struct Chunk { int idx = 0; } chunk;
        struct Mask { int mask_type = 4; };
        const std::vector<Mask> chunk_masks_(1);
        __SDPA_CALL__
        __PREPARE_CALL__
    }
    void denoise_memsets() {
        const int tp = owner_cores == 8 ? 8 : 4;
        const int64_t cs = 50, h = 1024, pad_elems = 14 * 256;
        const uint32_t offset = 50 * 256 * sizeof(c10::Half);
        __MEMSET_CALLS__
    }
    void denoise_routes_and_table() {
        FmbPhysicalExecutionManifest manifest;
        FmbThreeStageChunkPlan plan;
        const bool spm_staging = false, pi05_xor3_gated_mlp_opt_in_ = false;
        const int64_t physical_len = 50;
        uint64_t adarms_table_src_base_ = 0x700000;
        const int64_t offset = 128, out_features = 3 * hidden_size();
        const uint32_t gemv_out_spm_addr = 0x10000;
        __DENOISE_CALLS__
        assert(expected_reduce == owner_cores);
    }
};
int main() {
    // Alternate owners in one process to expose any leaked eight-core default.
    for (const int budget : {8, 4, 6, 4, 8}) {
        policy.execution_core_count = budget;
        fill_count = fill_cores = sdpa_cores = sdpa_cache_cores = 0;
        Model{budget}.emit();
        assert(sdpa_cores == (budget == 8 ? 8 : 4));
        assert(sdpa_cache_cores == 8);  // RPUCache keeps eight controller stripes.
        assert(fill_count == (budget == 6 ? 1 : 0));
        assert(fill_cores == (budget == 6 ? 6 : 0));
        memset_calls.clear();
        Model{budget}.denoise_memsets();
        // Final residual belongs to the whole owner; K/V padding belongs only
        // to attention producers. The real memset wrapper emits each domain.
        assert(memset_calls.front().cores == budget);
        assert(memset_calls.front().address == 0x10000);
        assert(memset_calls.front().elements == 50 * 1024);
        for (size_t i = 1; i < memset_calls.size(); ++i) {
            assert(memset_calls[i].cores == (budget == 8 ? 8 : 4));
            assert(memset_calls[i].address == 0x10000 + 50 * 256 * sizeof(c10::Half));
            assert(memset_calls[i].elements == 14 * 256);
        }
        route_owner_cores = route_mlp_cores = reduce_cores = table_dma_cores = 0;
        Model{budget}.denoise_routes_and_table();
        // TP6 attention computes on four cores, but its shared MLP, descriptor
        // reduction and fallback AdaRMS broadcast belong to all six owner cores.
        assert(route_owner_cores == budget && route_mlp_cores == budget);
        assert(reduce_cores == budget && table_dma_cores == budget);
    }
}
"""
    for marker, value in (("__CORE_GUARD__", core_guard), ("__DECLARATION__", declaration),
                          ("__PREPARE__", prepare), ("__SDPA_CALL__", calls[sdpa_name]),
                          ("__PREPARE_CALL__", calls[prepare_name]),
                          ("__MEMSET_DECLARATION__", memset_declaration),
                          ("__MEMSET_HELPERS__", memset_helpers),
                          ("__MEMSET_CALLS__", "\n".join(memset_calls)),
                          ("__ROUTE_DECLARATIONS__", "\n".join(route_declarations)),
                          ("__DENOISE_CALLS__", "\n".join(denoise_calls))):
        harness = harness.replace(marker, value)
    source = tmp_path / "pi05_gemma_core_calls.cpp"
    source.write_text(with_actual_linear_contracts(harness))
    binary = tmp_path / "pi05_gemma_core_calls"
    built = subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(ROOT / "src/core"), str(source), "-o", str(binary)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)

def test_runtime_core_counts_drive_actual_launches(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ is required for production wrapper host checks")
    functions = [
        _function("src/ops/rpu_linear.cpp", "void validate_parallel_linear_geometry("),
        _function("src/ops/rpu_linear.cpp", "void validate_parallel_linear_grid("),
        _function("src/core/rpu_runtime_extras.cpp", "bool ring_all_reduce_sum_residual_admitted("),
        _function("src/core/rpu_runtime_extras.cpp", "struct RingAdaRmsOperands") + ";",
        _function("src/core/rpu_runtime_extras.cpp", "struct G05RingRtcEpilogue") + ";",
        _function("src/core/rpu_runtime_extras.cpp", "struct G05ActionTailNorm") + ";",
        _function("src/core/rpu_runtime_extras.cpp", "void launch_ring_all_reduce_sum_residual("),
        _function("src/core/rpu_runtime_extras.cpp", "void rpu_launch_all_reduce_sum_residual_impl("),
        _function("src/core/rpu_runtime_extras.cpp", "void rpu_prepare_ring_all_reduce_input("),
        _function("src/core/rpu_runtime_extras.cpp", "void rpu_launch_all_reduce_sum_residual_kernel(\n    uint32_t input_spm,\n"),
        _function("src/core/rpu_runtime_extras.cpp", "void rpu_launch_all_reduce_sum_residual_kernel(\n    uint32_t input_spm, uint32_t residual_spm, uint32_t output_spm,"),
        _function("src/ops/rpu_rmsnorm.cpp", "void rpu_launch_rmsnorm_spm_kernel("),
    ]
    # Compile the real capability selectors used by the wrapper. A stub that
    # always admits XOR3 would hide its separate exact-shape/TP8 boundary.
    shared_header = (ROOT / "src/core/fused_model_base.h").read_text()
    start = shared_header.index("enum class FmbSharedAllReduceRouteSelector")
    ring_contract = shared_header[start:shared_header.index("};", start) + 2]
    ring_selectors = "\n".join((
        _function("src/core/fmb_three_stage_chunk_plan.cpp",
                  "int64_t fmb_ring_all_reduce_route_selector(int64_t rows, int64_t cols,\n"),
        _function("src/core/fmb_three_stage_chunk_plan.cpp",
                  "int64_t fmb_ring_all_reduce_route_selector(int64_t rows, int64_t cols, bool pi05_xor3,"),
    ))
    declarations = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    rms_contract = declarations[
        declarations.index("inline constexpr int64_t RPU_RMSNORM_SCM_U16_MAX"):
        declarations.index("enum class RpuMixedRmsNormSpmRoute")
    ]
    kernel_ids = _function("src/core/rpu_kernel_cache.h", "enum class KernelId") + ";"
    harness = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
#define TORCH_INTERNAL_ASSERT(condition) assert(condition)
#define NUM_CORES 8
// Geometry-only Half double; launch arithmetic is not evaluated on the host.
namespace c10 { struct Half {
    uint16_t x;
    explicit Half(double value) : x(value == 0 ? 0 : 1) {}
}; }
constexpr int kKernelRegOk = 0;
constexpr int UNIFIED_NUM_CORES = 8;
constexpr int kRingCoreNum = 8;
constexpr uint32_t kRingWarpNum = 8;
constexpr uint32_t kRingRoundElemMax = 31U * 256U * kRingWarpNum;
constexpr uint64_t kRingNopaceChunkBytes = 11U * 1024U;
__KERNEL_IDS__
enum class GraphKernelNoDdrProof { RingAllReduceSpmResidual };
namespace v3 {
__RING_CONTRACT__
__RING_SELECTORS__
}
__RMS_CONTRACT__
struct SpmAllocator {
    static constexpr uint64_t SPM_USABLE = 8 * 1024 * 1024 - 1024;
    static constexpr uint64_t ALIGN = 256;
};
struct Allocator {
    bool initialized = true;
    bool is_initialized() const { return initialized; }
    uint32_t addr(int core, uint32_t offset) const {
        return 0x10000U + core * (8U * 1024U * 1024U) + offset;
    }
} SPM_ALLOC;
struct Kernel_t {
    std::map<uint32_t, uint16_t> regs;
    int reset_regs() { regs.clear(); return kKernelRegOk; }
    int set_regs(uint32_t index, uint16_t value) { regs[index] = value; return kKernelRegOk; }
} kernel;
struct KernelCache {
    static KernelCache& instance() { static KernelCache cache; return cache; }
    Kernel_t* get(KernelId) const { return &kernel; }
};
KernelId last_kernel;
Kernel_t* get_kernel(KernelId id) { last_kernel = id; return &kernel; }
#define GET_KERNEL(id) get_kernel(id)
struct RpuKernelGraph {
    static RpuKernelGraph& active() { static RpuKernelGraph graph; return graph; }
    bool kernel_register_census_active() const { return false; }
    void stage_kernel_no_ddr(KernelId, GraphKernelNoDdrProof) {}
};
void rpu_set_legacy_scm_u16_checked(Kernel_t* k, uint32_t i, uint16_t v, const char*) {
    k->set_regs(i, v);
}
uint64_t RpuGetDevAddr(c10::Half* ptr) { return reinterpret_cast<uint64_t>(ptr); }
uint16_t half_to_u16(c10::Half value) { return value.x; }
struct Queue {
    bool broadcast = false;
    int count = 0;
    int enqueues = 0;
    std::vector<uint8_t> cores;
    std::vector<uint16_t> grid;
    void set_broadcast_mode(bool enabled) { broadcast = enabled; }
    void enqueu_kernel(Kernel_t&, std::vector<uint16_t> dims, std::vector<uint8_t> active) {
        assert(broadcast);
        ++enqueues;
        grid = dims;
        cores = active;
        assert(static_cast<int>(cores.size()) == count);
        for (int i = 0; i < count; ++i) assert(cores[i] == i);
    }
} queue;
Queue* get_queue(int cores) {
    assert(cores >= 1 && cores <= 8);
    queue.count = cores;
    queue.broadcast = false;
    return &queue;
}
#define GET_QUEUE(count) get_queue(count)
int fill_count = 0;
int fill_cores = 0;
void rpu_launch_fill_spm_kernel(uint32_t, int64_t, c10::Half value, int cores) {
    assert(value.x == 0);
    fill_count++;
    fill_cores = cores;
}
template<class F> void rejects(F function) {
    bool failed = false;
    try { function(); } catch (const std::runtime_error&) { failed = true; }
    assert(failed);
}
'''
    main = r'''
int main() {
    const uint32_t input = SPM_ALLOC.addr(0, 0);
    const uint32_t output = SPM_ALLOC.addr(0, 2 * 1024 * 1024);
    const uint32_t residual = SPM_ALLOC.addr(0, 4 * 1024 * 1024);
    // A 6-core MLP is legal; attention Q/KV and lm_head are separate domains.
    validate_parallel_linear_geometry(16, 3072, 1024, 1, 6);
    validate_parallel_linear_geometry(16, 1024, 3072, 0, 6);
    validate_parallel_linear_geometry(1, 151936, 1024, 1, 4);
    rejects([] { validate_parallel_linear_geometry(1, 151936, 1024, 1, 6); });
    rejects([] { validate_parallel_linear_geometry(1, 1024, 3072, 0, 0); });
    rejects([] { validate_parallel_linear_geometry(1, 3072, 1024, 2, 6); });
    rejects([] { validate_parallel_linear_geometry(1, 1024, 1024, 0, 6); });

    rejects([] { validate_parallel_linear_geometry(INT64_MAX, 1024, 1024, 1, 4); });
    rejects([] { validate_parallel_linear_geometry(1, INT64_MAX - 15, 1024, 1, 4); });
    rejects([] { validate_parallel_linear_geometry(1, 1024, INT64_MAX - 15, 0, 4); });
    validate_parallel_linear_grid(65535, 65535, 1, 1);
    rejects([] { validate_parallel_linear_grid(65536, 128, 1, 16); });
    rejects([] { validate_parallel_linear_grid(128, 65536, 16, 1); });
    rejects([] { validate_parallel_linear_grid(128, 128, 0, 16); });
    // Alternate core counts in one process, including equal local geometry.
    for (const int cores : {8, 6, 4, 6, 1, 8}) {
        for (const int elements : {1, 257, 32768, 524288}) {
            launch_ring_all_reduce_sum_residual(
                input, residual, nullptr, output, 1, elements, cores);
            assert(queue.count == cores && kernel.regs[0] == cores);
            assert(queue.grid[0] == 8);  // warp count is a hardware constant
            uint32_t total = 0;
            const uint32_t avg = ((elements + cores - 1) / cores + 15) / 16 * 16;
            for (int core = 0; core < cores; ++core) {
                const uint32_t offset = kernel.regs[4096 + 32 + core] |
                    (uint32_t(kernel.regs[4096 + 40 + core]) << 16);
                const uint32_t size = kernel.regs[4096 + 48 + core] |
                    (uint32_t(kernel.regs[4096 + 56 + core]) << 16);
                assert(offset == total * sizeof(c10::Half));
                total += size;
            }
            assert(total == uint32_t(elements));
            assert(last_kernel == (avg * 2 < 11 * 1024
                ? KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_NOPACE
                : KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_PACED));
        }
        rpu_launch_rmsnorm_spm_kernel(input, output, residual, 16, 128, 1e-6,
                                     RpuRmsNormSpmRoute::BASE, cores);
        assert(queue.count == cores && queue.grid[0] == 16);
        rpu_launch_rmsnorm_spm_kernel(input, output, residual, 32, 128, 1e-6,
                                     RpuRmsNormSpmRoute::V32, cores);
        assert(queue.count == cores && queue.grid[0] == 1 && kernel.regs[3] == 32);
    }
    // The corrected expansion payload has the same register geometry while
    // selecting a different kernel from the main library's generic V16.
    rpu_launch_rmsnorm_spm_kernel(input, output, residual, 64, 1024, 1e-5,
                                 RpuRmsNormSpmRoute::V16_GRID8, 8);
    const auto main_v16_regs = kernel.regs;
    assert(last_kernel == KernelId::RMS_NORM_SPM_V16);
    rpu_launch_rmsnorm_spm_kernel(input, output, residual, 64, 1024, 1e-5,
                                 RpuRmsNormSpmRoute::HYVLA_V16_BITCOPY_GRID8, 8);
    assert(last_kernel == KernelId::HYVLA_EXPERT_RMSNORM_V16_BITCOPY);
    assert(kernel.regs == main_v16_regs && queue.count == 8 && queue.grid[0] == 8);
    rejects([&] { rpu_launch_rmsnorm_spm_kernel(input, output, residual, 64, 1024, 1e-5,
        RpuRmsNormSpmRoute::HYVLA_V16_BITCOPY_GRID8, 4); });
    rejects([&] { rpu_launch_rmsnorm_spm_kernel(input, output, residual, 51, 1024, 1e-5,
        RpuRmsNormSpmRoute::HYVLA_V16_BITCOPY_GRID8, 8); });
    // 4->6 preparation clears exactly the consumer domain; 6->6 is a no-op.
    rpu_prepare_ring_all_reduce_input(input, 16, 1024, 4, 6);
    assert(fill_count == 1 && fill_cores == 6);
    rpu_prepare_ring_all_reduce_input(input, 16, 1024, 6, 6);
    assert(fill_count == 1);
    rpu_launch_all_reduce_sum_residual_impl(input, residual, nullptr, output, 16, 1024, 4, 6);
    assert(queue.count == 6);
    rejects([&] { rpu_prepare_ring_all_reduce_input(input, 1, 257, 4, 6); });
    rejects([&] { rpu_prepare_ring_all_reduce_input(input, 16, 1024, 8, 6); });
    rejects([&] { rpu_launch_all_reduce_sum_residual_impl(input, residual, nullptr, output, 16, 1024, 8, 6); });
    rejects([&] { launch_ring_all_reduce_sum_residual(input, residual, nullptr, output, 1, 256, 0); });
    rejects([&] { launch_ring_all_reduce_sum_residual(input, residual, nullptr, output, 1, 256, 9); });
    rejects([&] { launch_ring_all_reduce_sum_residual(input, residual, nullptr, input, 1, 256, 6); });
    // The explicit Pi0.5 route retains its own TP8 boundary; admitting the
    // ordinary ring at four/six cores must not make this route legal there.
    struct PiShape { int64_t rows, cols; KernelId kernel; };
    for (const auto& shape : {
            PiShape{50, 1024, KernelId::PI05_RING_XOR3_M50N1024},
            PiShape{512, 1152, KernelId::PI05_RING_XOR3_M512N1152},
            PiShape{304, 2048, KernelId::PI05_RING_XOR3_M304N2048}}) {
        rpu_launch_all_reduce_sum_residual_kernel(
            input, residual, output, shape.rows, shape.cols, 8, 8, true);
        assert(queue.count == 8 && queue.cores.size() == 8 && last_kernel == shape.kernel);
        const int before = queue.enqueues;
        for (const auto& cores : {std::array<int, 2>{4, 8}, {6, 8}, {8, 6}, {4, 4}, {6, 6}}) {
            rejects([&] { rpu_launch_all_reduce_sum_residual_kernel(
                input, residual, output, shape.rows, shape.cols, cores[0], cores[1], true); });
        }
        for (const int cores : {4, 6}) {
            rejects([&] { launch_ring_all_reduce_sum_residual(input, residual, nullptr,
                output, shape.rows, shape.cols, cores, false, false, true); });
        }
        assert(queue.enqueues == before);
    }
    const int before_invalid_xor3 = queue.enqueues;
    rejects([&] { rpu_launch_all_reduce_sum_residual_kernel(
        input, residual, output, 16, 1024, 8, 8, true); });
    assert(queue.enqueues == before_invalid_xor3);
    rpu_launch_all_reduce_sum_residual_kernel(input, residual, output, 16, 1024, 4, 6, false);
    assert(queue.count == 6 && last_kernel == KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_NOPACE);
    // The exact local residual family retains its separate eight-core gate.
    assert(!ring_all_reduce_sum_residual_admitted(input, residual, nullptr, output, 320, 2048, 6, true));
    rejects([&] { rpu_launch_rmsnorm_spm_kernel(input, output, residual, 1, 128, 1e-6, RpuRmsNormSpmRoute::BASE, 0); });
    rejects([&] { rpu_launch_rmsnorm_spm_kernel(input, output, residual, 1, 128, 1e-6, RpuRmsNormSpmRoute::BASE, 9); });
    rejects([&] { rpu_launch_rmsnorm_spm_kernel(input, output, residual, 0, 128, 1e-6, RpuRmsNormSpmRoute::BASE, 4); });
    rejects([&] { rpu_launch_rmsnorm_spm_kernel(input, output, residual, 1, 127, 1e-6, RpuRmsNormSpmRoute::BASE, 4); });
    rejects([&] { rpu_launch_rmsnorm_spm_kernel(input, output, residual, 65536, 128, 1e-6, RpuRmsNormSpmRoute::BASE, 4); });
    rejects([&] { rpu_launch_rmsnorm_spm_kernel(input, output, residual, 16, 32768, 1e-6, RpuRmsNormSpmRoute::V16, 6); });
    rejects([&] { rpu_launch_rmsnorm_spm_kernel(input, output, residual, 32 * 65536LL, 128, 1e-6, RpuRmsNormSpmRoute::V32, 6); });
    rejects([&] { rpu_launch_rmsnorm_spm_kernel(input, output, residual, 64, 128, 1e-6, RpuRmsNormSpmRoute::QWEN3VL_V64, 6); });
    assert(!ring_all_reduce_sum_residual_admitted(input, residual, nullptr, input, 576, 1280, 6, false, false, true));
    SPM_ALLOC.initialized = false;
    assert(!ring_all_reduce_sum_residual_admitted(input, residual, nullptr, output, 1, 256, 6));
}
'''
    source = tmp_path / "core_count_wrappers.cpp"
    binary = tmp_path / "core_count_wrappers"
    source.write_text(with_actual_linear_contracts(harness.replace("__RMS_CONTRACT__", rms_contract)
                      .replace("__KERNEL_IDS__", kernel_ids)
                      .replace("__RING_CONTRACT__", ring_contract)
                      .replace("__RING_SELECTORS__", ring_selectors)
                      + "\n".join(functions) + main))
    subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(ROOT / "src/core"), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_explicit_ddr_linear_threads_core_count_through_transport():
    body = _function("src/ops/rpu_linear.cpp", "void rpu_launch_linear_ddr_kernel(")
    # A forgotten TP8 DMA/gather would corrupt otherwise-valid TP4 lm_head output.
    assert "NUM_CORES" not in body
    assert "{0, 1, 2, 3, 4, 5, 6, 7}" not in body
    assert "cur_m, n, k, partition, num_cores" in body
    assert "validate_parallel_linear_geometry(" in body
    legacy = _function("src/ops/rpu_linear.cpp", "Tensor rpu_linear_with_core_count(")
    assert "rpu_linear_with_accumulation(" in legacy
    assert "partition, num_cores, true, c10::nullopt" in legacy
    public = _function("src/ops/rpu_linear.cpp", "Tensor rpu_linear_with_accumulation(")
    assert "static_cast<int>(num_cores)" in public
    assert "!graph_dma::active()" in public
    assert "weight.device() == input.device()" in public
    assert "at::empty(output_shape, input.options())" in public


def test_eager_m1_column_writeback_preserves_rows_and_graph_transport(tmp_path):
    """Execute the real route and writeback blocks over per-core result bits.

    The fixture does not implement GEMM. Distinct row/core values expose a
    mistaken core-major DDR layout, a lost mutable offset, or a changed route.
    """
    compiler = shutil.which("g++")
    assert compiler, "Linear writeback contract requires a C++ compiler"
    launcher = _function("src/ops/rpu_linear.cpp", "void rpu_launch_linear_ddr_kernel(")
    selection = launcher[launcher.index("  size_t row_bytes ="):launcher.index("  const uint32_t x_addr =")]
    writeback = launcher[
        launcher.index("    if (partition != 0) {\n      // Write-back destination."):
        launcher.index("    // Row partition (eager only): per-core")
    ]
    harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
using Half = uint16_t;
struct SpmAllocator {
    static constexpr size_t ALIGN=256;
    static constexpr size_t SPM_PLANNING_BUDGET=2*1024*1024+127;
};
size_t Align(size_t value,size_t alignment) { return (value+alignment-1)/alignment*alignment; }
struct Pool {
    size_t available,used=0;
    size_t free_space() const { return available; }
    size_t total_used() const { return used; }
} SPM_ALLOC;
std::vector<std::vector<Half>> shard;
std::vector<Half> gathered;
int gathers=0,scatters=0,mutable_writes=0,copies=0,row_batches=0;
void rpu_ddr_flush(Half*) {}
int rpu_resolve_all_gather_schedule(size_t,size_t) { return 0; }
void rpu_launch_all_gather_spm_kernel(uint32_t,uint32_t,size_t rows,size_t cols,size_t width,int cores,int) {
    assert(width==sizeof(Half) && shard.size()==static_cast<size_t>(cores));
    ++gathers; gathered.resize(rows*cols*cores);
    for(size_t row=0;row<rows;++row) for(int core=0;core<cores;++core)
        std::copy_n(shard[core].begin()+row*cols,cols,gathered.begin()+(row*cores+core)*cols);
}
void rpu_launch_spm_copy_ddr_dma(uint32_t,Half* dst,size_t count) {
    ++copies; assert(count==gathered.size()); std::copy(gathered.begin(),gathered.end(),dst);
}
void rpu_launch_spm_copy_ddr_dma_mutable(uint32_t src,const uint64_t* base,int64_t offset,size_t count) {
    ++mutable_writes;
    rpu_launch_spm_copy_ddr_dma(src,reinterpret_cast<Half*>(*base+offset),count);
}
void rpu_launch_spm_scatter_ddr_dma(uint32_t,Half* dst,int64_t count,int64_t stride,int cores) {
    ++scatters; assert(count>0 && count*2%16==0 && stride%16==0);
    assert(shard.size()==static_cast<size_t>(cores));
    for(int core=0;core<cores;++core) {
        assert(shard[core].size()==static_cast<size_t>(count));
        auto* target=reinterpret_cast<Half*>(reinterpret_cast<uint8_t*>(dst)+core*stride);
        std::copy(shard[core].begin(),shard[core].end(),target);
    }
}
void rpu_launch_spm_scatter_ddr_dma_mutable(uint32_t src,const uint64_t* base,int64_t offset,
                                         int64_t count,int64_t stride,int cores) {
    ++mutable_writes;
    rpu_launch_spm_scatter_ddr_dma(src,reinterpret_cast<Half*>(*base+offset),count,stride,cores);
}
void rpu_launch_spm_scatter_rows_ddr_dma_immediate(uint32_t,Half* dst,int64_t rows,int64_t count,
        int64_t spm_row_stride,int64_t ddr_row_stride,int64_t core_stride,int cores) {
    ++row_batches; assert(rows>0 && count>0 && spm_row_stride==count*2);
    assert(shard.size()==static_cast<size_t>(cores));
    for(int core=0;core<cores;++core) {
        assert(shard[core].size()==static_cast<size_t>(rows*count));
        for(int64_t row=0;row<rows;++row) {
            auto* target=reinterpret_cast<Half*>(reinterpret_cast<uint8_t*>(dst)+row*ddr_row_stride+core*core_stride);
            std::copy_n(shard[core].begin()+row*count,count,target);
        }
    }
}
Half value(size_t row,size_t column,bool bias) { return 10+row*1000+column+(bias?10000:0); }
void run(size_t m,int num_cores,bool in_graph,bool has_bias,bool pressure,bool use_mutable,
         int partition=1,int precision=16,size_t used=0,int64_t halo=0,bool expect_rows=false) {
    const bool w8a16=precision==8,w4a16=precision==4;
    const size_t n=32*num_cores,k=16*num_cores,a_dwidth=sizeof(Half);
    const size_t local_n=partition==0?n:n/num_cores;
    const size_t local_k=partition==0?k/num_cores:k;
    const bool kernel_applies_bias=has_bias && partition!=0;
    // Equal to the old gather admission boundary: still enough for x+y.
    SPM_ALLOC.available=pressure
        ? (local_k+local_n+n)*a_dwidth+3*SpmAllocator::ALIGN+(kernel_applies_bias?Align(local_n*a_dwidth,SpmAllocator::ALIGN):0)
        : 1024*1024;
    SPM_ALLOC.used=used;
    // A plain GEMM row occupies the whole local K; only the conv-slab route
    // narrows the pitch to one channel row and adds halo rows that end a chunk
    // without starting a window of their own. `halo` stays 0 for every case
    // below that checks the writeback, which the slab route does not change.
    const size_t in_pitch=local_k;
    // The launcher's conv-slab pointer is non-null exactly when it has a halo.
    const int64_t* conv_slab=halo?&halo:nullptr;
    constexpr Half sentinel=0xffff;
    constexpr size_t prefix=16,suffix=16;
    std::vector<Half> storage(prefix+m*n+suffix,sentinel);
    Half* yp=storage.data()+prefix;
    const uint64_t base=reinterpret_cast<uint64_t>(storage.data());
    const uint64_t* live_dst_base=use_mutable?&base:nullptr;
    __SELECTION__
    if(partition==0) { assert(!col_scatter && !col_scatter_rows); return; }
    assert(col_scatter_rows==expect_rows);
    gathers=scatters=mutable_writes=copies=row_batches=0;
    const int64_t dst_offset_bytes=prefix*a_dwidth;
    const uint32_t y_addr=0x10000,g_addr=0x20000;
    for(size_t row0=0;row0<m;row0+=m_chunk) {
        const size_t cur_m=std::min(m-row0,m_chunk);
        shard.assign(num_cores,std::vector<Half>(cur_m*local_n));
        for(int core=0;core<num_cores;++core) for(size_t row=0;row<cur_m;++row)
            for(size_t column=0;column<local_n;++column)
                shard[core][row*local_n+column]=value(row0+row,core*local_n+column,has_bias);
        __WRITEBACK__
    }
    for(size_t row=0;row<m;++row) for(size_t column=0;column<n;++column)
        assert(yp[row*n+column]==value(row,column,has_bias));
    assert(std::all_of(storage.begin(),storage.begin()+prefix,[](Half v){return v==sentinel;}));
    assert(std::all_of(storage.end()-suffix,storage.end(),[](Half v){return v==sentinel;}));
    const int chunks=static_cast<int>((m+m_chunk-1)/m_chunk);
    if(expect_rows) {
        // One per-row batch per chunk, and fewer chunks than the gather allows.
        assert(chunks==1 && row_batches==1);
        assert(scatters==0 && gathers==0 && copies==0);
    } else if(pressure || (m==1 && !in_graph)) {
        assert(scatters==static_cast<int>(m) && gathers==0 && copies==0 && row_batches==0);
    } else {
        assert(gathers==chunks && copies==chunks && scatters==0 && row_batches==0);
    }
    assert(mutable_writes==(use_mutable?scatters+copies:0));
}
int main() {
    for(int cores:{1,2,4,6,8}) for(bool bias:{false,true}) {
        run(1,cores,false,bias,false,false); // new eager single-row route
        run(3,cores,false,bias,false,false); // unchanged multi-row gather
        run(1,cores,true,bias,false,true);   // unchanged captured single-row gather
        run(3,cores,true,bias,false,true);   // captured multi-row destination offset
        run(3,cores,false,bias,true,false);  // old low-SPM row-by-row fallback
        run(1,cores,true,bias,true,true);    // old captured low-SPM mutable scatter
        run(1,cores,false,bias,false,false,0); // row partition never selects scatter
    }
    // Quantized eager projections share the same M1 scatter selection. Their
    // separate admission contract allows TP8 and rejects graph capture.
    for(int precision:{8,4}) for(bool bias:{false,true}) {
        run(1,8,false,bias,false,false,1,precision);
        run(3,8,false,bias,false,false,1,precision);
        run(3,8,false,bias,true,false,1,precision);
        bool rejected=false;
        try {
            run(1,8,false,bias,false,false,1,precision,SpmAllocator::SPM_PLANNING_BUDGET);
        } catch(const std::runtime_error&) { rejected=true; }
        assert(rejected); // physical free space cannot bypass the aligned budget
    }
    // The slab's halo rows are a fixed surcharge on the chunk, not a per-row
    // cost, so they have to be charged to the SPM budget and not only to the
    // staging DMA. Six halo rows at a 128-half pitch outweigh the all-gather
    // landing zone that the pressure budget just affords, so the geometry that
    // is admitted without a halo must be refused with one.
    run(3,8,false,false,false,false,1,16,0,6);
    // Eager multi-row GEMM whose gather landing zone would split M into two
    // chunks: per-row scatter batches write it in one, row-major. Captured and
    // live-destination calls keep the gather.
    for(bool bias:{false,true}) {
        run(2000,8,false,bias,false,false,1,16,0,0,true);
        run(2000,8,true,bias,false,true); // two gathered chunks
    }
    bool halo_rejected=false;
    try { run(3,8,false,false,true,false,1,16,0,6); }
    catch(const std::runtime_error&) { halo_rejected=true; }
    assert(halo_rejected);
}
'''
    cpp, binary = tmp_path / "linear_writeback.cpp", tmp_path / "linear_writeback"
    cpp.write_text(with_actual_linear_contracts(harness.replace("__SELECTION__", selection).replace("__WRITEBACK__", writeback)))
    built = subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(ROOT / "src/core"), str(cpp), "-o", str(binary)],
                           capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, built.stdout + built.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


def test_sdpa_rejects_invalid_core_geometry_before_cache_staging(tmp_path):
    compiler = shutil.which("g++")
    assert compiler, "SDPA admission checks require g++"
    function = _function("src/ops/rpu_sdpa.cpp", "void launch_sdpa_spm_unified_impl(")
    wrapper = _function("src/ops/rpu_sdpa.cpp", "void rpu_launch_sdpa_spm_unified_kernel_v2(")
    kernel_ids = _function("src/core/rpu_kernel_cache.h", "enum class KernelId") + ";"
    sdpa_contract = _function("src/core/rpu_helpers.h", "struct SdpaTiling") + ";\n"
    sdpa_contract += _function(
        "src/ops/rpu_hyvla_sdpa_tiling.h", "inline bool rpu_hyvla_s51_key_rows_supported(")
    sdpa_contract += _function("src/ops/rpu_sdpa.cpp",
        "bool rpu_lingbot2_sdpa_tail_t51_key_rows_supported(")
    sdpa_contract += "\nconstexpr int MASK_2D = 4;\n"
    # Execute the real admission prefix, ending where cache staging begins.
    # No tensor/SDK simulation is needed to prove divide-by-zero and narrowing
    # are rejected before those side effects.
    prefix = function[:function.index("  size_t dwidth = ")]
    harness = r'''
#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
constexpr int MAX_CORES = 8;
namespace at { struct Tensor {}; }
__KERNEL_IDS__
int64_t CeilDiv(int64_t value, int64_t divisor) { assert(divisor > 0); return (value + divisor - 1) / divisor; }
__SDPA_CONTRACT__
struct KernelCache {static KernelCache& instance(){static KernelCache c;return c;}
 bool has_loaded(KernelId)const{throw std::runtime_error("unexpected LingBot2 payload admission");}};
void* GET_QUEUE(int){throw std::runtime_error("unexpected LingBot2 queue admission");}
int staged = 0;
__ADMISSION__
  ++staged;
}
__WRAPPER__
int main() {
    at::Tensor cache;
    auto run = [&](int64_t rows, int64_t q, int64_t kv, int64_t dim,
                   int64_t length, int cores, int virtual_cores) {
        rpu_launch_sdpa_spm_unified_kernel_v2(cache, cache, 1, {},
            0x10000, 0x20000, 0x30000, 0x40000, rows, q, kv, dim, length,
            cores, virtual_cores, 0, false, false);
    };
    for (int cores : {4, 6, 8}) {
        for (int vtp : {-1, 0, 8}) {
            run(1, cores * 2, cores, 128, 608, cores, vtp);
            run(16, cores * 2, 2, 128, 608, cores, vtp); // replicated KV
        }
    }
    for (const auto& args : std::vector<std::vector<int64_t>>{
            {1,16,8,128,608,0,0}, {1,16,8,128,608,9,0},
            {1,16,8,128,608,4,-2}, {1,16,8,128,608,4,9},
            {0,16,8,128,608,4,0}, {65536,16,8,128,608,4,0},
            {1,0,8,128,608,4,0}, {1,16,0,128,608,4,0},
            {1,16,8,127,608,4,0}, {1,16,8,65536,608,4,0},
            {1,16,8,128,0,4,0}, {1,16,8,128,INT64_MAX,4,0},
            {1,16,8,128,608,6,0}, {1,12,8,128,608,6,0},
            {1,24,16,128,608,8,0},
            {1,65535,1,128,608,1,8}}) {
        const int before = staged;
        bool rejected = false;
        try { run(args[0],args[1],args[2],args[3],args[4],args[5],args[6]); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && staged == before);
    }
}
'''.replace("__ADMISSION__", prefix).replace("__WRAPPER__", wrapper).replace(
        "__KERNEL_IDS__", kernel_ids).replace("__SDPA_CONTRACT__", sdpa_contract)
    source = tmp_path / "sdpa_admission.cpp"
    source.write_text(with_actual_linear_contracts(harness))
    binary = source.with_suffix("")
    subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(ROOT / "src/core"), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_explicit_linear_public_caller_validates_before_allocating_and_returns_fresh(tmp_path):
    compiler = shutil.which("g++")
    assert compiler, "Linear public-entry checks require g++"
    functions = "\n".join((
        _function("src/ops/rpu_linear.cpp", "static void check_linear_shapes("),
        _function("src/ops/rpu_linear.cpp", "void validate_parallel_linear_geometry("),
        _function("src/ops/rpu_linear.cpp", "static void check_linear_w4a16_operands("),
        _function("src/ops/rpu_linear.cpp", "Tensor rpu_linear_with_accumulation("),
        _function("src/ops/rpu_linear.cpp", "Tensor rpu_linear_with_core_count("),
    ))
    declarations = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    begin = declarations.index("at::Tensor rpu_linear_with_accumulation(")
    functions = declarations[begin:declarations.index(";", begin) + 1] + "\n" + functions
    # As in test_qwen3vl_transport_contract, use a small tensor double while
    # compiling the production body. The kernel recorder checks dispatch and
    # output ownership; it does not pretend to compute GEMM or swizzled weights.
    harness = r'''
#include <cassert>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
#define RECORD_FUNCTION(...)
#define NUM_CORES 8
namespace c10 {
template<class T> using optional = std::optional<T>;
using std::nullopt;
}
namespace graph_dma { bool capturing = false; bool active() { return capturing; } }
int allocations = 0, copies = 0, launches = 0, last_cores = 0, last_partition = -1;
int retained_scales = 0;
bool last_acc32 = false;
std::shared_ptr<int> last_scale;
namespace at {
enum ScalarType { kHalf, kFloat, kChar, kByte };
constexpr int kPrivateUse1 = 1;
struct Device {
    int kind = kPrivateUse1, index = 0;
    int type() const { return kind; }
    bool operator==(const Device& rhs) const { return kind == rhs.kind && index == rhs.index; }
};
struct Options { ScalarType value = kHalf; Device where; ScalarType dtype() const { return value; } };
struct Sizes : std::vector<int64_t> {
    using std::vector<int64_t>::vector;
    std::vector<int64_t> vec() const { return {begin(), end()}; }
};
struct Tensor {
    Sizes shape;
    Options opts;
    bool dense = true;
    std::shared_ptr<int> storage = std::make_shared<int>(0);
    Tensor() = default;
    Tensor(std::initializer_list<int64_t> dims) : shape(dims) {}
    bool defined() const { return !shape.empty(); }
    int64_t dim() const { return shape.size(); }
    int64_t size(int64_t axis) const { return shape.at(axis < 0 ? shape.size()+axis : axis); }
    int64_t numel() const { return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>()); }
    Device device() const { return opts.where; }
    ScalarType scalar_type() const { return opts.value; }
    bool is_contiguous() const { return dense; }
    Sizes sizes() const { return shape; }
    Options options() const { return opts; }
    Tensor contiguous() const { ++copies; auto result = *this; result.dense = true; return result; }
    Tensor view(std::initializer_list<int64_t> dims) const { auto result = *this; result.shape = dims; return result; }
};
Tensor empty(const std::vector<int64_t>& dims, Options opts) {
    ++allocations;
    Tensor result; result.shape.assign(dims.begin(), dims.end()); result.opts = opts;
    return result;
}
}
using Tensor = at::Tensor;
Tensor rpu_retain_linear_quant_scale(const Tensor& weight, const Tensor& scale, int cores = 8) {
    assert(weight.scalar_type() == at::kByte && scale.defined());
    assert(cores == 4 || cores == 6 || cores == 8);
    ++retained_scales;
    return scale;
}
void rpu_launch_linear_ddr_kernel(const Tensor& input, const Tensor& weight,
        Tensor& output, const Tensor& bias, bool has_bias, int partition,
        const uint64_t* live_dst, int64_t offset, const uint64_t* live_bias, int cores,
        const Tensor& scale, c10::optional<bool> linear_acc32) {
    assert(!live_dst && !live_bias && offset == 0);
    assert(input.dim() == 2 && output.dim() == 2 && input.dense);
    const bool packed = weight.scalar_type() == at::kByte;
    assert(input.size(0) == output.size(0));
    assert(input.size(1) == weight.size(1) * (packed ? 2 : 1));
    assert(output.size(1) == weight.size(0));
    assert(output.storage != input.storage && output.storage != weight.storage);
    if (has_bias) assert(bias.numel() == output.size(1));
    assert(linear_acc32.has_value());
    last_acc32 = *linear_acc32;
    last_scale = scale.defined() ? scale.storage : nullptr;
    ++launches; last_cores = cores; last_partition = partition;
    *output.storage = launches;
}
__FUNCTIONS__
int main() {
    Tensor input({2,16,1024}), weight({3072,1024}), bias({3072});
    input.dense = false;
    for (int64_t cores : {4,6,8}) {
        auto first = rpu_linear_with_core_count(input, weight, bias, 1, cores);
        const int old_value = *first.storage;
        auto second = rpu_linear_with_core_count(input, weight, {}, 1, cores);
        assert((first.shape == at::Sizes{2,16,3072}));
        assert(first.storage != second.storage && *first.storage == old_value);
        assert(last_cores == cores && last_partition == 1 && last_acc32 && !last_scale);
        Tensor row_input({16,3072}), row_weight({1024,3072});
        auto row = rpu_linear_with_core_count(row_input, row_weight, {}, 0, cores);
        assert(last_cores == cores && last_partition == 0 && row.size(1) == 1024 && last_acc32);
        // The public cold-policy caller transports both accumulation choices
        // without changing core geometry or handing out shared output storage.
        for (bool acc32 : {false, true, false}) {
            auto explicit_first = rpu_linear_with_accumulation(input, weight, bias, 1, cores, acc32, {});
            const int retained_value = *explicit_first.storage;
            auto explicit_second = rpu_linear_with_accumulation(input, weight, {}, 1, cores, acc32, {});
            assert(explicit_first.shape == first.shape && last_acc32 == acc32 && !last_scale);
            assert(explicit_first.storage != explicit_second.storage && *explicit_first.storage == retained_value);
            assert(last_cores == cores && last_partition == 1);
            auto explicit_row = rpu_linear_with_accumulation(row_input, row_weight, {}, 0, cores, acc32, {});
            assert(explicit_row.size(1) == 1024 && last_cores == cores && last_partition == 0 && last_acc32 == acc32);

            Tensor quant_weight = weight, scale({3072});
            quant_weight.opts.value = at::kChar;
            auto quant = rpu_linear_with_accumulation(input, quant_weight, bias, 1, cores, acc32, scale);
            assert(quant.shape == first.shape && last_scale == scale.storage);
            assert(last_cores == cores && last_partition == 1 && last_acc32 == acc32);
        }
    }
    Tensor packed_weight({3072,512}), packed_scale({32,3072});
    packed_weight.opts.value = at::kByte;
    for (bool acc32 : {false, true}) {
        const int before_retained = retained_scales;
        auto packed = rpu_linear_with_accumulation(input, packed_weight, bias, 1, 8, acc32, packed_scale);
        assert((packed.shape == at::Sizes{2,16,3072}));
        assert(last_cores == 8 && last_partition == 1 && last_acc32 == acc32);
        assert(last_scale == packed_scale.storage && retained_scales == before_retained + 1);
    }
    auto reject = [&](Tensor x, Tensor w, c10::optional<Tensor> b, int64_t part, int64_t cores) {
        const int before = allocations, before_copies = copies, before_launches = launches;
        for (int mode : {-1, 0, 1}) {
            bool rejected = false;
            try {
                if (mode < 0) rpu_linear_with_core_count(x, w, b, part, cores);
                else rpu_linear_with_accumulation(x, w, b, part, cores, mode == 1, {});
            } catch (const std::runtime_error&) { rejected = true; }
            assert(rejected && allocations == before && copies == before_copies && launches == before_launches);
        }
    };
    reject(input,weight,{},1,0); reject(input,weight,{},1,9);
    reject(input,weight,{},1,INT64_MAX); reject(input,weight,{},2,4);
    reject(Tensor({1,1024}),Tensor({151936,1024}),{},1,6);
    reject(Tensor({1,1024}),Tensor({1024,1024}),{},0,6);
    reject(Tensor({0,1024}),weight,{},1,4);
    reject(Tensor({1,1023}),weight,{},1,4);
    auto invalid = weight; invalid.dense = false; reject(input,invalid,{},1,4);
    invalid = weight; invalid.opts.value = at::kFloat; reject(input,invalid,{},1,4);
    invalid = weight; invalid.opts.where.index = 1; reject(input,invalid,{},1,4);
    auto host = input; host.opts.where.kind = 0; reject(host,weight,{},1,4);
    auto wrong_bias = bias; wrong_bias.opts.where.kind = 0; reject(input,weight,wrong_bias,1,4);
    wrong_bias = bias; wrong_bias.dense = false; reject(input,weight,wrong_bias,1,4);
    wrong_bias = bias; wrong_bias.opts.value = at::kFloat; reject(input,weight,wrong_bias,1,4);
    reject(input,weight,Tensor({}),1,4); reject(input,weight,Tensor({1}),1,4);
    reject(input,packed_weight,{},1,8); // Packed weights require an explicit scale before allocation.
    graph_dma::capturing = true; reject(input,weight,{},1,4);
}
'''.replace("__FUNCTIONS__", functions)
    source = tmp_path / "linear_public.cpp"
    source.write_text(with_actual_linear_contracts(harness))
    binary = source.with_suffix("")
    subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(ROOT / "src/core"), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
