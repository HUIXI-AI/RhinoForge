// rpu_llama_kvcache.cpp
// LLaMA KV-Cache insert kernel launch functions

#include "rpu_ops.h"
#include "fused_model_base.h"
#include "rpu_spm_allocator.h"
#include "rhino_launch_buffer.h"
#include "rhino_launch_queue.h"
#include <c10/util/Half.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <iostream>
#include <limits>

using namespace rhino_lkn;

#define NUM_CORES 8

struct LegacyKvInsertSettings {
    KvInsertLegacyPolicy policy;
    bool trace = false;
};

// Compatibility boundary for the old launcher ABI.  No plan-taking launcher or
// pure resolver reads process state.  Keep the original switch lifetimes and
// spellings here: v16/ANY_TP/trace are first-use cached, while the two hybrid
// switches remain hot reads.
static LegacyKvInsertSettings legacy_kvinsert_settings_from_env() {
    const auto on = [](const char* value) {
        return value && std::string(value) != "" &&
            std::string(value) != "0" && std::string(value) != "false" &&
            std::string(value) != "False";
    };
    static const bool v16_enabled = [] {
        const char* value = std::getenv("RPU_KVINSERT_V16");
        return !(value && (std::string(value) == "0" ||
                           std::string(value) == "false"));
    }();
    static const bool any_tp_v16 = [] {
        const char* value = std::getenv("RPU_KVINSERT_V16_ANY_TP");
        return value && std::string(value) != "" &&
            std::string(value) != "0" && std::string(value) != "false" &&
            std::string(value) != "False";
    }();
    static const bool trace = [] {
        const char* value = std::getenv("RPU_KVINSERT_V16_TRACE");
        return value && std::string(value) != "0" &&
            std::string(value) != "false";
    }();

    const bool hybrid2 =
        on(std::getenv("RPU_KVINSERT_HYBRID_V16")) ||
        on(std::getenv("RPU_RHINOVLA_KVINSERT_HYBRID_V16"));
    const bool hybrid3 = on(std::getenv("RPU_KVINSERT_HYBRID3_V16"));
    return {{v16_enabled, any_tp_v16, hybrid2, hybrid3}, trace};
}

static const char* kvinsert_segment_path_tag(
    KvInsertRoute route, size_t segment_index) {
    switch (route) {
        case KvInsertRoute::V2: return "v2";
        case KvInsertRoute::ALIGNED_V16: return "v16";
        case KvInsertRoute::PAD16_V16: return "v16_pad16";
        case KvInsertRoute::HYBRID2:
            return segment_index == 0 ? "v16_hybrid_bulk" :
                                        "v2_hybrid_tail";
        case KvInsertRoute::HYBRID3:
            if (segment_index == 0) return "v2_hybrid3_head";
            if (segment_index == 1) return "v16_hybrid3_bulk";
            return "v2_hybrid3_tail";
        default: return "invalid";
    }
}

// =============================================================================
// LLaMA V-Cache Insert Kernel (8-core, multi-warp)
//
// Inserts V values into V-Cache with swizzle layout transformation.
//
// Input layout:  [batch, seq_len, num_cores * num_kv_percore_heads, head_dim]
// Output layout: [batch, sValVx, nKVHeadVx, headDimVx, nKVHeadChunk, headDimChunk, sValChunk]
//
// Where:
//   - sValVx = seq_len / sValChunk (e.g., 32/16 = 2)
//   - nKVHeadVx = num_kv_percore_heads (e.g., 32/8 = 4)
//   - headDimVx = head_dim / 16 (e.g., 64/16 = 4)
//   - nKVHeadChunk = NUM_CORES (8)
//   - headDimChunk = 16 (V16 elements)
//   - sValChunk = 16 (V16 elements)
//
// Kernel parameters:
//   param0/1: position (int32 value)
//   param2: num_kv_percore_heads
//   param3: head_dim
//   param4: seq_len
//   param6/7: input V SPM address
//   param8/9: V-Cache DDR address
// =============================================================================

void rpu_launch_llama_insert_vcache(
    const at::Tensor &v_input,   // [batch, seq_len, num_cores * num_kv_percore_heads, head_dim] DDR
    at::Tensor &v_cache,         // [batch, sValVx, nKVHeadVx, headDimVx, nKVHeadChunk, headDimChunk, sValChunk] DDR
    int64_t position             // 插入位置 (cache 中的起始 position)
) {
    // ========== 1. 参数验证 ==========
    TORCH_CHECK(v_input.dim() == 4,
                "v_input must be 4D [batch, seq_len, num_cores*num_kv_percore_heads, head_dim], got ",
                v_input.dim(), "D");
    TORCH_CHECK(v_input.scalar_type() == at::kHalf,
                "v_input must be FP16, got ", v_input.scalar_type());
    TORCH_CHECK(v_input.is_contiguous(),
                "v_input must be contiguous");
    TORCH_CHECK(v_input.device().type() == c10::DeviceType::PrivateUse1,
                "v_input must be on RPU device");

    int64_t batch = v_input.size(0);
    int64_t seq_len = v_input.size(1);
    int64_t total_kv_heads = v_input.size(2);  // num_cores * num_kv_percore_heads
    int64_t head_dim = v_input.size(3);

    TORCH_CHECK(batch == 1, "batch must be 1, got ", batch);
    TORCH_CHECK(total_kv_heads % NUM_CORES == 0 || NUM_CORES % total_kv_heads == 0,
                "total_kv_heads (", total_kv_heads, ") must divide or be divisible by ", NUM_CORES);

    // Per-core KV elements: nhkv * head_dim / NUM_CORES
    int64_t local_kv_dim = total_kv_heads * head_dim / NUM_CORES;
    int64_t num_kv_percore_heads = (total_kv_heads >= NUM_CORES)
        ? total_kv_heads / NUM_CORES : 1;

    // ========== 2. 验证输出维度 ==========
    constexpr int64_t sValChunk = 16;
    constexpr int64_t headDimChunk = 16;
    int64_t sValVx = CeilDiv(seq_len, sValChunk);
    int64_t headDimVx = head_dim / headDimChunk;
    // DDR insert always uses NUM_CORES → effective_kv_slots = total_kv_heads
    int64_t nKVHeadChunk = std::min((int64_t)NUM_CORES, total_kv_heads);
    int64_t nKVHeadVx = CeilDiv(total_kv_heads, nKVHeadChunk);

    TORCH_CHECK(v_cache.dim() == 7,
                "v_cache must be 7D, got ", v_cache.dim(), "D");
    TORCH_CHECK(v_cache.size(0) == batch,
                "v_cache batch mismatch: expected ", batch, ", got ", v_cache.size(0));
    TORCH_CHECK(v_cache.size(1) == sValVx,
                "v_cache sValVx mismatch: expected ", sValVx, ", got ", v_cache.size(1));
    TORCH_CHECK(v_cache.size(2) == nKVHeadVx,
                "v_cache nKVHeadVx mismatch: expected ", nKVHeadVx, ", got ", v_cache.size(2));
    TORCH_CHECK(v_cache.size(3) == headDimVx,
                "v_cache headDimVx mismatch: expected ", headDimVx, ", got ", v_cache.size(3));
    TORCH_CHECK(v_cache.size(4) == nKVHeadChunk,
                "v_cache nKVHeadChunk mismatch: expected ", nKVHeadChunk, ", got ", v_cache.size(4));
    TORCH_CHECK(v_cache.size(5) == headDimChunk,
                "v_cache headDimChunk mismatch: expected ", headDimChunk, ", got ", v_cache.size(5));
    TORCH_CHECK(v_cache.size(6) == sValChunk,
                "v_cache sValChunk mismatch: expected ", sValChunk, ", got ", v_cache.size(6));
    TORCH_CHECK(v_cache.scalar_type() == at::kHalf,
                "v_cache must be FP16");
    TORCH_CHECK(v_cache.is_contiguous(),
                "v_cache must be contiguous");

    // ========== 3. 获取 Kernel 和 Queue ==========
    Kernel_t* kernel = GET_KERNEL(KernelId::LLAMA_INSERT_VCACHE);
    if (kernel == nullptr) {
        kernel = KernelCache::instance().get_kernel("llama_insert_vcache_multiwarp");
    }
    TORCH_CHECK(kernel != nullptr, "Failed to get llama_insert_vcache_multiwarp kernel");
    kernel->reset_regs();

    auto* wq = GET_QUEUE(NUM_CORES);
    wq->set_broadcast_mode(true);

    // ========== 4. 分配每核 SPM 内存 ==========
    // 每核数据量: seq_len * local_kv_dim * sizeof(FP16)
    size_t per_core_v_bytes = seq_len * local_kv_dim * sizeof(c10::Half);

    std::vector<LocalSPM_t*> spm_v(NUM_CORES);

    for (int i = 0; i < NUM_CORES; ++i) {
        spm_v[i] = graph_acquire_local_spm(per_core_v_bytes, read_write, kStride32B, 2, i);
    }

    // ========== 5. 复制数据到 SPM ==========
    const c10::Half* v_ptr = v_input.data_ptr<c10::Half>();

    // 输入布局: [batch, seq_len, total_kv_heads, head_dim]
    // When nhkv >= NUM_CORES: Core i gets heads [i*percore : (i+1)*percore]
    // When nhkv < NUM_CORES: Core i gets head_dim slice [i*local_kv_dim : (i+1)*local_kv_dim]

    for (int core = 0; core < NUM_CORES; ++core) {
        c10::Half* spm_v_ptr = static_cast<c10::Half*>(spm_v[core]->get_cpu_ptr());

        for (int64_t s = 0; s < seq_len; ++s) {
            const c10::Half* src;
            if (total_kv_heads >= NUM_CORES) {
                // Split by KV heads: core i gets its heads
                src = v_ptr + s * total_kv_heads * head_dim
                            + core * num_kv_percore_heads * head_dim;
            } else {
                // Split by head_dim: core i gets its portion of head_dim
                src = v_ptr + s * total_kv_heads * head_dim
                            + core * local_kv_dim;
            }
            c10::Half* dst = spm_v_ptr + s * local_kv_dim;
            ddr_to_spm(dst, src, local_kv_dim * sizeof(c10::Half));
        }
    }

    // ========== 6. 准备 DDR 地址 ==========
    rpu_ddr_flush(const_cast<c10::Half*>(v_input.data_ptr<c10::Half>()));
    rpu_ddr_flush(v_cache.data_ptr<c10::Half>());

    uint64_t cache_addr = RpuGetDevAddr(v_cache.data_ptr<c10::Half>()) >> 8;

    // ========== 7. 设置寄存器 ==========
    // param0/1: position (int32 value, 直接是数值)
    // param2: num_kv_percore_heads (每核的 head 数)
    // param3: head_dim
    // param4: seq_len
    // param6/7: input V SPM address (64-bit)
    // param8/9: V-Cache DDR address (64-bit)

    // 使用 Core 0 的 SPM 地址作为基准 (broadcast mode 下所有核使用相同的相对偏移)
    uint32_t v_spm_addr = spm_v[0]->get_rpu_addr();
    uint32_t position_val = static_cast<uint32_t>(position);

    // Per-core element count is param2 * param3.
    int64_t kern_heads = num_kv_percore_heads;
    int64_t kern_hdim = head_dim;
    if (total_kv_heads < NUM_CORES) {
        kern_heads = 1;
        kern_hdim = local_kv_dim;
    }

    kernel->set_regs(0, (uint16_t)(position_val & 0xFFFF));      // position 低16位
    kernel->set_regs(1, (uint16_t)(position_val >> 16));         // position 高16位
    kernel->set_regs(2, (uint16_t)kern_heads);                    // heads (per core)
    kernel->set_regs(3, (uint16_t)kern_hdim);                     // head_dim (per core)
    kernel->set_regs(4, (uint16_t)seq_len);                      // seq_len
    kernel->set_regs(6, (uint16_t)(v_spm_addr & 0xFFFF));        // V SPM 地址低16位
    kernel->set_regs(7, (uint16_t)(v_spm_addr >> 16));           // V SPM 地址高16位
    kernel->set_regs(8, (uint16_t)(cache_addr & 0xFFFF));        // V-Cache DDR 地址低16位
    kernel->set_regs(9, (uint16_t)((cache_addr >> 16) & 0xFFFF)); // V-Cache DDR 地址高16位

    // ========== 8. 启动 Kernel ==========
    // grid: [seq_len, 1, 1]
    wq->enqueu_kernel(*kernel,
        {(uint16_t)seq_len, (uint16_t)1, (uint16_t)1},
        {0, 1, 2, 3, 4, 5, 6, 7});

    // ========== 9. 同步和清理 ==========
    rpu_ddr_flush(v_cache.data_ptr<c10::Half>());

    // Reset kernel registers after sync point to prevent state corruption on consecutive runs

    for (int i = 0; i < NUM_CORES; ++i) {
        graph_release_local_spm(spm_v[i]);
    }
}

// =============================================================================
// LLaMA K-Cache Insert Kernel (8-core, multi-warp)
//
// Inserts K values into K-Cache with swizzle layout transformation.
// Parameter layout is identical to V-Cache for consistency.
//
// Input layout:  [batch, seq_len, num_cores * num_kv_percore_heads, head_dim]
// Output layout: [batch, sKeyVx, nKVHeadVx, headDimVx, nKVHeadChunk, sKeyChunk, headDimChunk]
//
// Kernel parameters (same as V-Cache):
//   param0/1: position (int32 value)
//   param2: num_kv_percore_heads
//   param3: head_dim
//   param4: seq_len
//   param6/7: input K SPM address (64-bit)
//   param8/9: K-Cache DDR address (64-bit)
// =============================================================================

void rpu_launch_llama_insert_kcache(
    const at::Tensor &k_input,   // [batch, seq_len, num_kv_heads, head_dim] DDR
    at::Tensor &k_cache,         // [batch, sKeyVx, nKVHeadVx, headDimVx, nKVHeadChunk, sKeyChunk, headDimChunk] DDR
    int64_t position             // 插入位置 (cache 中的起始 position)
) {
    // ========== 1. 参数验证 ==========
    TORCH_CHECK(k_input.dim() == 4,
                "k_input must be 4D [batch, seq_len, num_kv_heads, head_dim], got ",
                k_input.dim(), "D");
    TORCH_CHECK(k_input.scalar_type() == at::kHalf,
                "k_input must be FP16, got ", k_input.scalar_type());
    TORCH_CHECK(k_input.is_contiguous(),
                "k_input must be contiguous");
    TORCH_CHECK(k_input.device().type() == c10::DeviceType::PrivateUse1,
                "k_input must be on RPU device");

    int64_t batch = k_input.size(0);
    int64_t seq_len = k_input.size(1);
    int64_t total_kv_heads = k_input.size(2);
    int64_t head_dim = k_input.size(3);

    TORCH_CHECK(batch == 1, "batch must be 1, got ", batch);
    TORCH_CHECK(total_kv_heads % NUM_CORES == 0 || NUM_CORES % total_kv_heads == 0,
                "total_kv_heads (", total_kv_heads, ") must divide or be divisible by ", NUM_CORES);

    int64_t local_kv_dim = total_kv_heads * head_dim / NUM_CORES;
    int64_t num_kv_percore_heads = (total_kv_heads >= NUM_CORES)
        ? total_kv_heads / NUM_CORES : 1;

    // ========== 2. 验证输出维度 ==========
    // K-Cache layout: [batch, sKeyVx, nKVHeadVx, headDimVx, nKVHeadChunk, sKeyChunk, headDimChunk]
    constexpr int64_t sKeyChunk = 16;
    constexpr int64_t headDimChunk = 16;
    int64_t sKeyVx = CeilDiv(seq_len, sKeyChunk);
    int64_t headDimVx = head_dim / headDimChunk;
    // DDR insert always uses NUM_CORES → effective_kv_slots = total_kv_heads
    int64_t nKVHeadChunk = std::min((int64_t)NUM_CORES, total_kv_heads);
    int64_t nKVHeadVx = CeilDiv(total_kv_heads, nKVHeadChunk);

    TORCH_CHECK(k_cache.dim() == 7,
                "k_cache must be 7D, got ", k_cache.dim(), "D");
    TORCH_CHECK(k_cache.size(0) == batch,
                "k_cache batch mismatch: expected ", batch, ", got ", k_cache.size(0));
    TORCH_CHECK(k_cache.size(1) == sKeyVx,
                "k_cache sKeyVx mismatch: expected ", sKeyVx, ", got ", k_cache.size(1));
    TORCH_CHECK(k_cache.size(2) == nKVHeadVx,
                "k_cache nKVHeadVx mismatch: expected ", nKVHeadVx, ", got ", k_cache.size(2));
    TORCH_CHECK(k_cache.size(3) == headDimVx,
                "k_cache headDimVx mismatch: expected ", headDimVx, ", got ", k_cache.size(3));
    TORCH_CHECK(k_cache.size(4) == nKVHeadChunk,
                "k_cache nKVHeadChunk mismatch: expected ", nKVHeadChunk, ", got ", k_cache.size(4));
    TORCH_CHECK(k_cache.size(5) == sKeyChunk,
                "k_cache sKeyChunk mismatch: expected ", sKeyChunk, ", got ", k_cache.size(5));
    TORCH_CHECK(k_cache.size(6) == headDimChunk,
                "k_cache headDimChunk mismatch: expected ", headDimChunk, ", got ", k_cache.size(6));
    TORCH_CHECK(k_cache.scalar_type() == at::kHalf,
                "k_cache must be FP16");
    TORCH_CHECK(k_cache.is_contiguous(),
                "k_cache must be contiguous");

    // ========== 3. 获取 Kernel 和 Queue ==========
    Kernel_t* kernel = GET_KERNEL(KernelId::LLAMA_INSERT_KCACHE);
    if (kernel == nullptr) {
        kernel = KernelCache::instance().get_kernel("llama_insert_kcache_multiwarp");
    }
    TORCH_CHECK(kernel != nullptr, "Failed to get llama_insert_kcache_multiwarp kernel");
    kernel->reset_regs();

    auto* wq = GET_QUEUE(NUM_CORES);
    wq->set_broadcast_mode(true);

    // ========== 4. 分配每核 SPM 内存 ==========
    size_t per_core_k_bytes = seq_len * local_kv_dim * sizeof(c10::Half);

    std::vector<LocalSPM_t*> spm_k(NUM_CORES);

    for (int i = 0; i < NUM_CORES; ++i) {
        spm_k[i] = graph_acquire_local_spm(per_core_k_bytes, read_write, kStride32B, 2, i);
    }

    // ========== 5. 复制数据到 SPM ==========
    const c10::Half* k_ptr = k_input.data_ptr<c10::Half>();

    // 输入布局: [batch, seq_len, total_kv_heads, head_dim]
    // When nhkv >= NUM_CORES: Core i gets heads [i*percore : (i+1)*percore]
    // When nhkv < NUM_CORES: Core i gets head_dim slice [i*local_kv_dim : (i+1)*local_kv_dim]

    for (int core = 0; core < NUM_CORES; ++core) {
        c10::Half* spm_k_ptr = static_cast<c10::Half*>(spm_k[core]->get_cpu_ptr());

        for (int64_t s = 0; s < seq_len; ++s) {
            const c10::Half* src;
            if (total_kv_heads >= NUM_CORES) {
                src = k_ptr + s * total_kv_heads * head_dim
                            + core * num_kv_percore_heads * head_dim;
            } else {
                src = k_ptr + s * total_kv_heads * head_dim
                            + core * local_kv_dim;
            }
            c10::Half* dst = spm_k_ptr + s * local_kv_dim;
            ddr_to_spm(dst, src, local_kv_dim * sizeof(c10::Half));
        }
    }

    // ========== 6. 准备 DDR 地址 ==========
    rpu_ddr_flush(const_cast<c10::Half*>(k_input.data_ptr<c10::Half>()));
    rpu_ddr_flush(k_cache.data_ptr<c10::Half>());

    uint64_t cache_addr = RpuGetDevAddr(k_cache.data_ptr<c10::Half>()) >> 8;

    // ========== 7. 设置寄存器 (与 V-Cache 完全一致) ==========
    // param0/1: position (int32 value)
    // param2: num_kv_percore_heads
    // param3: head_dim
    // param4: seq_len
    // param6/7: input K SPM address (64-bit)
    // param8/9: K-Cache DDR address (64-bit)

    uint32_t k_spm_addr = spm_k[0]->get_rpu_addr();
    uint32_t position_val = static_cast<uint32_t>(position);

    // Per-core element count is param2 * param3.
    int64_t kern_heads = num_kv_percore_heads;
    int64_t kern_hdim = head_dim;
    if (total_kv_heads < NUM_CORES) {
        kern_heads = 1;
        kern_hdim = local_kv_dim;
    }

    kernel->set_regs(0, (uint16_t)(position_val & 0xFFFF));       // position 低16位
    kernel->set_regs(1, (uint16_t)(position_val >> 16));          // position 高16位
    kernel->set_regs(2, (uint16_t)kern_heads);                     // heads (per core)
    kernel->set_regs(3, (uint16_t)kern_hdim);                      // head_dim (per core)
    kernel->set_regs(4, (uint16_t)seq_len);                       // seq_len
    kernel->set_regs(6, (uint16_t)(k_spm_addr & 0xFFFF));         // K SPM 地址低16位
    kernel->set_regs(7, (uint16_t)(k_spm_addr >> 16));            // K SPM 地址高16位
    kernel->set_regs(8, (uint16_t)(cache_addr & 0xFFFF));         // K-Cache DDR 地址低16位
    kernel->set_regs(9, (uint16_t)((cache_addr >> 16) & 0xFFFF)); // K-Cache DDR 地址高16位

    // ========== 8. 启动 Kernel ==========
    // grid: [seq_len, 1, 1]
    wq->enqueu_kernel(*kernel,
        {(uint16_t)seq_len, (uint16_t)1, (uint16_t)1},
        {0, 1, 2, 3, 4, 5, 6, 7});

    // ========== 9. 同步和清理 ==========
    rpu_ddr_flush(k_cache.data_ptr<c10::Half>());

    // Reset kernel registers after sync point

    for (int i = 0; i < NUM_CORES; ++i) {
        graph_release_local_spm(spm_k[i]);
    }
}

// =============================================================================
// SPM Unified Layout K-Cache Insert Kernel
// =============================================================================
// Input is in SPM at unified layout position (K buffer)
// No DDR -> SPM copy needed - data is already in SPM from RoPE kernel
// Output: K-Cache in DDR (swizzle format)
// =============================================================================

// Batch decode: shift a cache base address to batch slot b. The 7-D cache is
// contiguous with batch outermost, so slot b starts cache_batch_offset_elems
// elements in. The kernels take v128 (256-byte) addresses, so the byte offset
// must be 256-aligned — it always is (the innermost swizzle block alone is
// nKVHeadChunk*sChunk*headDimChunk = 8*16*16 elements = 4 KB), but assert it
// rather than silently truncating in the >> 8.
static at::Tensor kv_cache_batch_operand(
    const at::Tensor &cache,
    int64_t cache_batch_offset_elems,
    const char *what) {
    if (cache_batch_offset_elems == 0) return cache;

    TORCH_CHECK(cache_batch_offset_elems > 0 && cache.dim() == 7 &&
                    cache.is_contiguous() && cache.stride(0) > 0 &&
                    cache_batch_offset_elems % cache.stride(0) == 0,
                what, ": cache_batch_offset_elems=", cache_batch_offset_elems,
                " must select one contiguous batch slot (stride0=",
                cache.stride(0), ")");
    const int64_t batch_slot = cache_batch_offset_elems / cache.stride(0);
    TORCH_CHECK(batch_slot < cache.size(0),
                what, ": batch slot ", batch_slot,
                " out of range for batch=", cache.size(0));
    const int64_t off_bytes =
        cache_batch_offset_elems * static_cast<int64_t>(sizeof(c10::Half));
    TORCH_CHECK(off_bytes % 256 == 0,
                what, ": batch slot byte offset ", off_bytes,
                " must be 256-aligned for the v128 cache address");
    return cache.select(0, batch_slot);
}

static void validate_kvinsert_spm_rows(
    const KvInsertSegmentPlan& plan, int64_t spm_rows) {
    TORCH_CHECK(
        (spm_rows == 0 && plan.physical_rows() == plan.logical_rows()) ||
            spm_rows >= plan.physical_rows(),
        "KV-insert spm_rows must cover the plan's physical rows; got "
        "spm_rows=", spm_rows, " physical_rows=", plan.physical_rows(),
        " logical_rows=", plan.logical_rows());
}

void rpu_launch_insert_kcache_spm_unified_with_plan(
    at::Tensor &k_cache,         // DDR K-cache (swizzle format)
    uint32_t k_off,              // SPM offset for K buffer
    int64_t num_kv_heads, int64_t head_dim,
    int num_cores,               // number of cores (attn_tp)
    int64_t cache_batch_offset_elems,
    int64_t spm_rows,
    const KvInsertSegmentPlan& plan,
    bool trace)
{
    // Validate the complete immutable route before staging any graph node.
    rpu_validate_kvinsert_segment_plan(
        plan, num_cores, num_kv_heads, head_dim);
    validate_kvinsert_spm_rows(plan, spm_rows);
    const int64_t position = plan.segment(0).position;
    int64_t total_kv_heads = num_kv_heads;
    // Per-core KV dim: when nhkv >= num_cores, each core has nhkv/num_cores complete heads;
    // when nhkv < num_cores, each core has a portion of head_dim (col partition).
    int64_t local_kv_dim = num_kv_heads * head_dim / num_cores;
    int64_t num_kv_percore_heads = (num_kv_heads >= num_cores) ? num_kv_heads / num_cores : 1;

    // Validate cache dimensions
    constexpr int64_t sKeyChunk = 16;

    int64_t sKeyVx_needed =
        CeilDiv(position + plan.physical_rows(), sKeyChunk);

    TORCH_CHECK(k_cache.dim() == 7, "k_cache must be 7D");
    TORCH_CHECK(k_cache.size(1) >= sKeyVx_needed,
                "k_cache sKeyVx too small: got ", k_cache.size(1), ", need >= ", sKeyVx_needed);
    TORCH_CHECK(k_cache.scalar_type() == at::kHalf, "k_cache must be FP16");
    TORCH_CHECK(k_cache.is_contiguous(), "k_cache must be contiguous");

    TORCH_CHECK(
        cache_batch_offset_elems == 0 ||
            !RpuKernelGraph::active().kernel_register_census_active(),
        "typed DDR-register census does not yet encode a KV-cache batch "
        "offset; refusing an unproven K-cache subrange");
    const at::Tensor k_cache_operand = kv_cache_batch_operand(
        k_cache, cache_batch_offset_elems, "insert_kcache_spm_unified");
    // ===== 设置寄存器 =====
    uint32_t k_spm_addr = SPM_ALLOC.addr(0, k_off);
    // Per-core element count per position is param2 * param3.
    // When nhkv >= num_cores: param2=local_kv_heads, param3=head_dim
    // When nhkv < num_cores: param2=1, param3=local_kv_dim (head_dim split across cores)
    int64_t kern_heads = num_kv_percore_heads;
    int64_t kern_hdim = head_dim;
    if (total_kv_heads < num_cores) {
        kern_heads = 1;
        kern_hdim = local_kv_dim;
    }

    // 定义寄存器配置 lambda (per-part: hybrid v16-bulk + v2-tail reuse it)
    auto setup_regs = [=](Kernel_t* kernel,
                          int64_t part_position,
                          int64_t part_seq_len,
                          uint32_t part_spm_addr) {
        uint32_t position_val = static_cast<uint32_t>(part_position);
        kernel->set_regs(0, (uint16_t)(position_val & 0xFFFF));
        kernel->set_regs(1, (uint16_t)(position_val >> 16));
        kernel->set_regs(2, (uint16_t)kern_heads);
        kernel->set_regs(3, (uint16_t)kern_hdim);
        kernel->set_regs(4, (uint16_t)part_seq_len);
        kernel->set_regs(6, (uint16_t)(part_spm_addr & 0xFFFF));
        kernel->set_regs(7, (uint16_t)(part_spm_addr >> 16));
        kernel->set_regs(21, (uint16_t)(num_cores));
    };

    auto launch_once = [&](const KvInsertSegment& part,
                           uint32_t part_spm_addr,
                           const char* path_tag) {
        const bool use_v16 = part.kernel == KvInsertKernel::V16;
        const KernelId kernel_id = use_v16
            ? KernelId::LLAMA_INSERT_KCACHE_V16
            : KernelId::LLAMA_INSERT_KCACHE;
        auto register_writer =
            RpuKernelGraph::active().stage_kernel_ddr_registers(
                kernel_id,
                std::vector<GraphDdrRegisterOperandSpec>{
                    GraphDdrRegisterOperandSpec{
                        GraphDdrRegisterAbi{
                            GraphDdrRegisterRole::KeyCache,
                            GraphDdrRegisterAccess::Write,
                            GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                            8, 9},
                        k_cache_operand}});
        rpu_ddr_flush(k_cache_operand.data_ptr<c10::Half>());
        Kernel_t* kernel = GET_KERNEL(kernel_id);
        if (kernel == nullptr && !use_v16) {  // legacy v2 string fallback only
            TORCH_CHECK(
                !RpuKernelGraph::active().kernel_register_census_active(),
                "typed DDR-register census forbids raw K-cache kernel fallback");
            kernel = KernelCache::instance().get_kernel("llama_insert_kcache_multiwarp");
        }
        TORCH_CHECK(kernel != nullptr, "Failed to get llama_insert_kcache",
                    use_v16 ? "_v16" : "", " kernel");
        kernel->reset_regs();
        setup_regs(kernel, part.position, part.rows, part_spm_addr);
        register_writer.write(*kernel);

        if (trace) {
            std::cerr << "[KVINSERT_TRACE] kind=K path=" << path_tag
                      << " pos=" << part.position << " seq=" << part.rows
                      << " nkv=" << num_kv_heads << " hdim=" << kern_hdim
                      << " grid_x=" << (use_v16 ? (part.rows / 16) : part.rows)
                      << "\n";
        }

        const uint16_t grid_x = use_v16
            ? static_cast<uint16_t>(part.rows / 16)
            : static_cast<uint16_t>(part.rows);
        auto* wq = GET_QUEUE(num_cores);
        wq->set_broadcast_mode(true);
        std::vector<uint8_t> core_list;
        for (int i = 0; i < num_cores; ++i) core_list.push_back(i);
        wq->enqueu_kernel(*kernel, {grid_x, (uint16_t)1, (uint16_t)1}, core_list);
    };

    const int64_t bytes_per_row =
        kern_heads * kern_hdim * static_cast<int64_t>(sizeof(c10::Half));
    for (size_t i = 0; i < plan.segment_count(); ++i) {
        const KvInsertSegment& part = plan.segment(i);
        TORCH_CHECK(part.token_offset <=
                        (static_cast<int64_t>(
                             std::numeric_limits<uint32_t>::max()) -
                         k_spm_addr) / bytes_per_row,
                    "KV-insert K segment SPM address overflows uint32");
    }
    for (size_t i = 0; i < plan.segment_count(); ++i) {
        const KvInsertSegment& part = plan.segment(i);
        const uint32_t part_spm_addr = k_spm_addr +
            static_cast<uint32_t>(part.token_offset * bytes_per_row);
        launch_once(part, part_spm_addr,
                    kvinsert_segment_path_tag(plan.route(), i));
    }

      // ===== 同步 =====
      rpu_ddr_flush(k_cache_operand.data_ptr<c10::Half>());

    // 注意: 不清理 SPM - 由调用方 (rpu_fused_qkv_attention_unified) 负责
}

void rpu_launch_insert_kcache_spm_unified(
    at::Tensor &k_cache,
    int64_t position,
    uint32_t k_off,
    int64_t seq_len, int64_t num_kv_heads, int64_t head_dim,
    int num_cores,
    int64_t cache_batch_offset_elems,
    bool allow_non8_v16,
    bool allow_hybrid_v16,
    int64_t spm_rows)
{
    const LegacyKvInsertSettings settings =
        legacy_kvinsert_settings_from_env();
    const KvInsertSegmentPlan plan =
        rpu_resolve_legacy_kvinsert_segment_plan(
            position, seq_len, spm_rows, num_cores, num_kv_heads, head_dim,
            allow_non8_v16, allow_hybrid_v16, settings.policy);
    rpu_launch_insert_kcache_spm_unified_with_plan(
        k_cache, k_off, num_kv_heads, head_dim, num_cores,
        cache_batch_offset_elems, spm_rows, plan, settings.trace);
}

// =============================================================================
// SPM Unified Layout V-Cache Insert Kernel
// =============================================================================
// Input is in SPM at unified layout position (V buffer)
// No DDR -> SPM copy needed - data is already in SPM from Linear kernel
// Output: V-Cache in DDR (swizzle format)
// =============================================================================

void rpu_launch_insert_vcache_spm_unified_with_plan(
    at::Tensor &v_cache,         // DDR V-cache (swizzle format)
    uint32_t v_off,              // SPM offset for V buffer
    int64_t num_kv_heads, int64_t head_dim,
    int num_cores,               // number of cores (attn_tp)
    int64_t cache_batch_offset_elems,
    int64_t spm_rows,
    const KvInsertSegmentPlan& plan,
    bool trace)
{
    // Validate the complete immutable route before staging any graph node.
    rpu_validate_kvinsert_segment_plan(
        plan, num_cores, num_kv_heads, head_dim);
    validate_kvinsert_spm_rows(plan, spm_rows);
    const int64_t position = plan.segment(0).position;
    int64_t total_kv_heads = num_kv_heads;
    int64_t local_kv_dim = num_kv_heads * head_dim / num_cores;
    int64_t num_kv_percore_heads = (num_kv_heads >= num_cores) ? num_kv_heads / num_cores : 1;

    // Validate cache dimensions
    constexpr int64_t sValChunk = 16;

    int64_t sValVx_needed =
        CeilDiv(position + plan.physical_rows(), sValChunk);

    TORCH_CHECK(v_cache.dim() == 7, "v_cache must be 7D");
    TORCH_CHECK(v_cache.size(1) >= sValVx_needed,
                "v_cache sValVx too small: got ", v_cache.size(1), ", need >= ", sValVx_needed);
    TORCH_CHECK(v_cache.scalar_type() == at::kHalf, "v_cache must be FP16");
    TORCH_CHECK(v_cache.is_contiguous(), "v_cache must be contiguous");

    TORCH_CHECK(
        cache_batch_offset_elems == 0 ||
            !RpuKernelGraph::active().kernel_register_census_active(),
        "typed DDR-register census does not yet encode a KV-cache batch "
        "offset; refusing an unproven V-cache subrange");
    const at::Tensor v_cache_operand = kv_cache_batch_operand(
        v_cache, cache_batch_offset_elems, "insert_vcache_spm_unified");
    // ===== 设置寄存器 =====
    uint32_t v_spm_addr = SPM_ALLOC.addr(0, v_off);

    // Per-core element count is param2 * param3.
    int64_t kern_heads = num_kv_percore_heads;
    int64_t kern_hdim = head_dim;
    if (total_kv_heads < num_cores) {
        kern_heads = 1;
        kern_hdim = local_kv_dim;
    }

    // 定义寄存器配置 lambda (per-part: hybrid v16-bulk + v2-tail reuse it)
    auto setup_regs = [=](Kernel_t* kernel,
                          int64_t part_position,
                          int64_t part_seq_len,
                          uint32_t part_spm_addr) {
        uint32_t position_val = static_cast<uint32_t>(part_position);
        kernel->set_regs(0, (uint16_t)(position_val & 0xFFFF));
        kernel->set_regs(1, (uint16_t)(position_val >> 16));
        kernel->set_regs(2, (uint16_t)kern_heads);
        kernel->set_regs(3, (uint16_t)kern_hdim);
        kernel->set_regs(4, (uint16_t)part_seq_len);
        kernel->set_regs(6, (uint16_t)(part_spm_addr & 0xFFFF));
        kernel->set_regs(7, (uint16_t)(part_spm_addr >> 16));
        kernel->set_regs(21, (uint16_t)(num_cores));
    };

    auto launch_once = [&](const KvInsertSegment& part,
                           uint32_t part_spm_addr,
                           const char* path_tag) {
        const bool use_v16 = part.kernel == KvInsertKernel::V16;
        const KernelId kernel_id = use_v16
            ? KernelId::LLAMA_INSERT_VCACHE_V16
            : KernelId::LLAMA_INSERT_VCACHE;
        auto register_writer =
            RpuKernelGraph::active().stage_kernel_ddr_registers(
                kernel_id,
                std::vector<GraphDdrRegisterOperandSpec>{
                    GraphDdrRegisterOperandSpec{
                        GraphDdrRegisterAbi{
                            GraphDdrRegisterRole::ValueCache,
                            GraphDdrRegisterAccess::Write,
                            GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                            8, 9},
                        v_cache_operand}});
        rpu_ddr_flush(v_cache_operand.data_ptr<c10::Half>());
        Kernel_t* kernel = GET_KERNEL(kernel_id);
        if (kernel == nullptr && !use_v16) {
            TORCH_CHECK(
                !RpuKernelGraph::active().kernel_register_census_active(),
                "typed DDR-register census forbids raw V-cache kernel fallback");
            kernel = KernelCache::instance().get_kernel("llama_insert_vcache_multiwarp");
        }
        TORCH_CHECK(kernel != nullptr, "Failed to get llama_insert_vcache",
                    use_v16 ? "_v16" : "", " kernel");
        kernel->reset_regs();
        setup_regs(kernel, part.position, part.rows, part_spm_addr);
        register_writer.write(*kernel);

        if (trace) {
            std::cerr << "[KVINSERT_TRACE] kind=V path=" << path_tag
                      << " pos=" << part.position << " seq=" << part.rows
                      << " nkv=" << num_kv_heads << " hdim=" << kern_hdim
                      << " grid_x=" << (use_v16 ? (part.rows / 16) : part.rows)
                      << "\n";
        }

        const uint16_t grid_x = use_v16
            ? static_cast<uint16_t>(part.rows / 16)
            : static_cast<uint16_t>(part.rows);
        auto* wq = GET_QUEUE(num_cores);
        wq->set_broadcast_mode(true);
        std::vector<uint8_t> core_list;
        for (int i = 0; i < num_cores; ++i) core_list.push_back(i);
        wq->enqueu_kernel(*kernel, {grid_x, (uint16_t)1, (uint16_t)1}, core_list);
    };

    const int64_t bytes_per_row =
        kern_heads * kern_hdim * static_cast<int64_t>(sizeof(c10::Half));
    for (size_t i = 0; i < plan.segment_count(); ++i) {
        const KvInsertSegment& part = plan.segment(i);
        TORCH_CHECK(part.token_offset <=
                        (static_cast<int64_t>(
                             std::numeric_limits<uint32_t>::max()) -
                         v_spm_addr) / bytes_per_row,
                    "KV-insert V segment SPM address overflows uint32");
    }
    for (size_t i = 0; i < plan.segment_count(); ++i) {
        const KvInsertSegment& part = plan.segment(i);
        const uint32_t part_spm_addr = v_spm_addr +
            static_cast<uint32_t>(part.token_offset * bytes_per_row);
        launch_once(part, part_spm_addr,
                    kvinsert_segment_path_tag(plan.route(), i));
    }

      // ===== 同步 =====
      rpu_ddr_flush(v_cache_operand.data_ptr<c10::Half>());

    // 注意: 不清理 SPM - 由调用方 (rpu_fused_qkv_attention_unified) 负责
}

void rpu_launch_insert_kvcache_spm_unified_with_plan(
    at::Tensor &k_cache, at::Tensor &v_cache,
    uint32_t k_off, uint32_t v_off,
    int64_t num_kv_heads, int64_t head_dim,
    int num_cores,
    int64_t k_cache_batch_offset_elems,
    int64_t v_cache_batch_offset_elems,
    int64_t spm_rows,
    const KvInsertSegmentPlan& plan,
    bool trace)
{
    rpu_launch_insert_kcache_spm_unified_with_plan(
        k_cache, k_off, num_kv_heads, head_dim, num_cores,
        k_cache_batch_offset_elems, spm_rows, plan, trace);
    rpu_launch_insert_vcache_spm_unified_with_plan(
        v_cache, v_off, num_kv_heads, head_dim, num_cores,
        v_cache_batch_offset_elems, spm_rows, plan, trace);
}

void rpu_launch_insert_vcache_spm_unified(
    at::Tensor &v_cache,
    int64_t position,
    uint32_t v_off,
    int64_t seq_len, int64_t num_kv_heads, int64_t head_dim,
    int num_cores,
    int64_t cache_batch_offset_elems,
    bool allow_non8_v16,
    bool allow_hybrid_v16,
    int64_t spm_rows)
{
    const LegacyKvInsertSettings settings =
        legacy_kvinsert_settings_from_env();
    const KvInsertSegmentPlan plan =
        rpu_resolve_legacy_kvinsert_segment_plan(
            position, seq_len, spm_rows, num_cores, num_kv_heads, head_dim,
            allow_non8_v16, allow_hybrid_v16, settings.policy);
    rpu_launch_insert_vcache_spm_unified_with_plan(
        v_cache, v_off, num_kv_heads, head_dim, num_cores,
        cache_batch_offset_elems, spm_rows, plan, settings.trace);
}

void rpu_launch_insert_kvcache_spm_unified(
    at::Tensor &k_cache, at::Tensor &v_cache,
    int64_t position,
    uint32_t k_off, uint32_t v_off,
    int64_t seq_len, int64_t num_kv_heads, int64_t head_dim,
    int num_cores,
    int64_t k_cache_batch_offset_elems,
    int64_t v_cache_batch_offset_elems,
    bool allow_non8_v16,
    bool allow_hybrid_v16,
    int64_t spm_rows)
{
    const LegacyKvInsertSettings settings =
        legacy_kvinsert_settings_from_env();
    const KvInsertSegmentPlan plan =
        rpu_resolve_legacy_kvinsert_segment_plan(
            position, seq_len, spm_rows, num_cores, num_kv_heads, head_dim,
            allow_non8_v16, allow_hybrid_v16, settings.policy);
    rpu_launch_insert_kvcache_spm_unified_with_plan(
        k_cache, v_cache, k_off, v_off, num_kv_heads, head_dim, num_cores,
        k_cache_batch_offset_elems, v_cache_batch_offset_elems, spm_rows,
        plan, settings.trace);
}

namespace {

void launch_owned_paired_kv_insert(
    at::Tensor& k_cache, at::Tensor& v_cache, int64_t position,
    uint32_t k_off, uint32_t v_off, int64_t seq_len,
    int64_t num_kv_heads, int64_t head_dim, int num_cores, bool decode) {
    TORCH_CHECK(num_kv_heads == 8 && head_dim == 128 && num_cores == 8 &&
                    seq_len == (decode ? 1 : 192) && position >= 0 &&
                    (decode || position % 16 == 0),
                "paired KV insert requires Qwen3-VL TP8/HD128 "
                "decode1 or 16-aligned prefill192");
    const auto cache_shape = [](const at::Tensor& cache) {
        return cache.defined() &&
            cache.device().type() == c10::DeviceType::PrivateUse1 &&
            cache.scalar_type() == at::kHalf && cache.dim() == 7 &&
            cache.is_contiguous() && cache.size(0) == 1 &&
            cache.size(1) > 0 && cache.size(2) == 1 && cache.size(3) == 8 &&
            cache.size(4) == 8 && cache.size(5) == 16 && cache.size(6) == 16;
    };
    TORCH_CHECK(cache_shape(k_cache) && cache_shape(v_cache) &&
                    k_cache.device() == v_cache.device() &&
                    k_cache.size(1) == v_cache.size(1),
                "paired KV insert requires matching owned B1 FP16 RPU "
                "7D KV8/HD128 swizzled caches");
    TORCH_CHECK(static_cast<uint64_t>(position) <= UINT32_MAX - seq_len &&
                    (position + seq_len + 15) / 16 <= k_cache.size(1),
                "paired KV insert position/rows exceed cache owner capacity");
    const auto k_begin = reinterpret_cast<uintptr_t>(k_cache.data_ptr());
    const auto v_begin = reinterpret_cast<uintptr_t>(v_cache.data_ptr());
    // Contiguous tensor views can have different pointers while overlapping.
    // Compare distances instead of adding extents to potentially high addresses.
    const bool disjoint = k_begin <= v_begin
        ? v_begin - k_begin >= k_cache.nbytes()
        : k_begin - v_begin >= v_cache.nbytes();
    TORCH_CHECK(disjoint,
                "paired KV insert requires non-overlapping K/V byte ranges");

    const KernelId kernel_id = decode
        ? KernelId::QWEN3VL_INSERT_KV_CACHE_DECODE
        : KernelId::QWEN3VL_INSERT_KV_CACHE_V16;
    auto register_writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
        kernel_id,
        std::vector<GraphDdrRegisterOperandSpec>{
            GraphDdrRegisterOperandSpec{
                GraphDdrRegisterAbi{
                    GraphDdrRegisterRole::KeyCache,
                    GraphDdrRegisterAccess::Write,
                    GraphDdrRegisterEncoding::DevAddrShift8LoHi, 8, 9},
                k_cache},
            GraphDdrRegisterOperandSpec{
                GraphDdrRegisterAbi{
                    GraphDdrRegisterRole::ValueCache,
                    GraphDdrRegisterAccess::Write,
                    GraphDdrRegisterEncoding::DevAddrShift8LoHi, 12, 13},
                v_cache}});
    rpu_ddr_flush(k_cache.data_ptr<c10::Half>());
    rpu_ddr_flush(v_cache.data_ptr<c10::Half>());
    const uint32_t k_spm_addr = SPM_ALLOC.addr(0, k_off);
    const uint32_t v_spm_addr = SPM_ALLOC.addr(0, v_off);
    Kernel_t* kernel = GET_KERNEL(kernel_id);
    TORCH_CHECK(kernel != nullptr, "selected paired KV insert kernel unavailable");
    kernel->reset_regs();
    const uint32_t position_val = static_cast<uint32_t>(position);
    kernel->set_regs(0, static_cast<uint16_t>(position_val & 0xFFFF));
    kernel->set_regs(1, static_cast<uint16_t>(position_val >> 16));
    kernel->set_regs(2, static_cast<uint16_t>(1));
    kernel->set_regs(3, static_cast<uint16_t>(128));
    kernel->set_regs(6, static_cast<uint16_t>(k_spm_addr & 0xFFFF));
    kernel->set_regs(7, static_cast<uint16_t>(k_spm_addr >> 16));
    kernel->set_regs(10, static_cast<uint16_t>(v_spm_addr & 0xFFFF));
    kernel->set_regs(11, static_cast<uint16_t>(v_spm_addr >> 16));
    kernel->set_regs(21, static_cast<uint16_t>(num_cores));
    register_writer.write(*kernel);
    auto* queue = GET_QUEUE(num_cores);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel,
        {static_cast<uint16_t>(decode ? 1 : seq_len / 16), 1, 1},
        {0, 1, 2, 3, 4, 5, 6, 7});
    rpu_ddr_flush(k_cache.data_ptr<c10::Half>());
    rpu_ddr_flush(v_cache.data_ptr<c10::Half>());
}

}  // namespace

void rpu_launch_insert_kv_cache_spm_unified_v16(
    at::Tensor& k_cache, at::Tensor& v_cache, int64_t position,
    uint32_t k_off, uint32_t v_off, int64_t seq_len,
    int64_t num_kv_heads, int64_t head_dim, int num_cores) {
    launch_owned_paired_kv_insert(k_cache, v_cache, position, k_off, v_off,
        seq_len, num_kv_heads, head_dim, num_cores, false);
}

void rpu_launch_insert_kv_cache_spm_unified_decode(
    at::Tensor& k_cache, at::Tensor& v_cache, int64_t position,
    uint32_t k_off, uint32_t v_off, int64_t seq_len,
    int64_t num_kv_heads, int64_t head_dim, int num_cores) {
    launch_owned_paired_kv_insert(k_cache, v_cache, position, k_off, v_off,
        seq_len, num_kv_heads, head_dim, num_cores, true);
}

// =============================================================================
// Arena-staged KV insert (RhinoVLA): stage external DDR K/V through the repo
// SPM_ALLOC arena, then reuse the unified swizzle insert. Caller resets the
// temporary arena after the per-layer prefix loop (so a per-layer prefill loop
// accumulates distinct offsets, one reset at end — avoids a kernel-vs-reset
// race on a reused offset).
// =============================================================================

static void validate_kv_arena_input(
    const at::Tensor& kv_input, int num_cores) {
    TORCH_CHECK(kv_input.dim() == 4,
                "arena insert: input must be 4D [B, seq_len, num_kv_heads, head_dim], got ",
                kv_input.dim(), "D");
    TORCH_CHECK(kv_input.scalar_type() == at::kHalf, "arena insert: input must be FP16");
    TORCH_CHECK(kv_input.is_contiguous(), "arena insert: input must be contiguous");
    TORCH_CHECK(kv_input.device().type() == at::kPrivateUse1,
                "arena insert: input must be on the RPU device");
    // The address math below indexes core-local slices and ignores the batch dim,
    // and the 8-core head split requires nkv to divide (or be divided by) num_cores
    // — mirror the checks in rpu_launch_llama_insert_{k,v}cache so a bad shape fails
    // here instead of silently inserting only batch 0 / a wrong head layout.
    TORCH_CHECK(kv_input.size(0) == 1,
                "arena insert: batch must be 1, got ", kv_input.size(0));
    TORCH_CHECK(kv_input.size(2) > 0 && kv_input.size(3) > 0,
                "arena insert: KV heads and head dimension must be positive");
    TORCH_CHECK(kv_input.size(2) % num_cores == 0 || num_cores % kv_input.size(2) == 0,
                "arena insert: num_kv_heads (", kv_input.size(2),
                ") must divide or be divisible by ", num_cores);
}

static uint32_t stage_kv_to_arena(const at::Tensor& kv_input, int num_cores,
                                  int64_t& seq_len_out, int64_t& total_kv_heads_out,
                                  int64_t& head_dim_out) {
    validate_kv_arena_input(kv_input, num_cores);

    // Self-init: in the standalone expert path the prefix insert runs before
    // the first graph capture that would lazily init SPM_ALLOC.
    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();

    const int64_t seq_len        = kv_input.size(1);
    const int64_t total_kv_heads = kv_input.size(2);
    const int64_t head_dim       = kv_input.size(3);
    const int64_t local_kv_dim   = total_kv_heads * head_dim / num_cores;
    const int64_t percore_heads  = (total_kv_heads >= num_cores) ? total_kv_heads / num_cores : 1;

    const size_t per_core_bytes = (size_t)seq_len * local_kv_dim * sizeof(c10::Half);
    const uint32_t off = SPM_ALLOC.alloc_temporary(per_core_bytes);

    rpu_ddr_flush(const_cast<c10::Half*>(kv_input.data_ptr<c10::Half>()));
    const c10::Half* in_ptr = kv_input.data_ptr<c10::Half>();
    for (int core = 0; core < num_cores; ++core) {
        c10::Half* spm_ptr = static_cast<c10::Half*>(SPM_ALLOC.cpu_ptr(core, off));
        for (int64_t s = 0; s < seq_len; ++s) {
            const c10::Half* src =
                (total_kv_heads >= num_cores)
                    ? in_ptr + s * total_kv_heads * head_dim + core * percore_heads * head_dim
                    : in_ptr + s * total_kv_heads * head_dim + core * local_kv_dim;
            ddr_to_spm(spm_ptr + s * local_kv_dim, src, local_kv_dim * sizeof(c10::Half));
        }
    }
    seq_len_out = seq_len; total_kv_heads_out = total_kv_heads; head_dim_out = head_dim;
    return off;
}

std::vector<int64_t> rpu_resolve_llama_kvcache_arena_plan(
    int64_t position, int64_t rows, int64_t num_kv_heads,
    int64_t head_dim) {
    const KvInsertSegmentPlan plan =
        rpu_resolve_kvinsert_segment_plan_auto(
            position, rows, rows, NUM_CORES, num_kv_heads, head_dim,
            KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16);
    const KvInsertRouteArguments arguments = rpu_kvinsert_route_arguments(
        plan, NUM_CORES, num_kv_heads, head_dim);
    return {arguments.begin(), arguments.end()};
}

static KvInsertSegmentPlan kv_arena_plan_from_arguments(
    const at::Tensor& input, int64_t position,
    at::IntArrayRef planned_route_arguments) {
    validate_kv_arena_input(input, NUM_CORES);
    const KvInsertSegmentPlan plan =
        rpu_kvinsert_segment_plan_from_route_arguments(
            planned_route_arguments, NUM_CORES, input.size(2), input.size(3));
    TORCH_CHECK(
        plan.segment(0).position == position &&
            plan.logical_rows() == input.size(1) &&
            plan.physical_rows() == input.size(1),
        "arena insert: planned KV route does not match input geometry");
    return plan;
}

void rpu_launch_llama_insert_kcache_arena(
    const at::Tensor &k_input, at::Tensor &k_cache, int64_t position,
    at::IntArrayRef planned_route_arguments) {
    const KvInsertSegmentPlan plan = kv_arena_plan_from_arguments(
        k_input, position, planned_route_arguments);
    int64_t seq_len, nkv, hd;
    const uint32_t off = stage_kv_to_arena(k_input, NUM_CORES, seq_len, nkv, hd);
    rpu_launch_insert_kcache_spm_unified_with_plan(
        k_cache, off, nkv, hd, NUM_CORES,
        /*cache_batch_offset_elems=*/0, seq_len, plan);
}

void rpu_launch_llama_insert_vcache_arena(
    const at::Tensor &v_input, at::Tensor &v_cache, int64_t position,
    at::IntArrayRef planned_route_arguments) {
    const KvInsertSegmentPlan plan = kv_arena_plan_from_arguments(
        v_input, position, planned_route_arguments);
    int64_t seq_len, nkv, hd;
    const uint32_t off = stage_kv_to_arena(v_input, NUM_CORES, seq_len, nkv, hd);
    rpu_launch_insert_vcache_spm_unified_with_plan(
        v_cache, off, nkv, hd, NUM_CORES,
        /*cache_batch_offset_elems=*/0, seq_len, plan);
}

void rpu_launch_llama_insert_kvcache_arena(
    const at::Tensor &k_input, const at::Tensor &v_input,
    at::Tensor &k_cache, at::Tensor &v_cache, int64_t position,
    at::IntArrayRef planned_route_arguments) {
    validate_kv_arena_input(k_input, NUM_CORES);
    validate_kv_arena_input(v_input, NUM_CORES);
    TORCH_CHECK(k_input.sizes() == v_input.sizes(),
                "arena paired insert: K/V input shapes must match; got ",
                k_input.sizes(), " and ", v_input.sizes());
    const KvInsertSegmentPlan plan = kv_arena_plan_from_arguments(
        k_input, position, planned_route_arguments);
    int64_t seq_len, nkv, hd;
    int64_t v_seq_len, v_nkv, v_hd;
    const uint32_t k_off =
        stage_kv_to_arena(k_input, NUM_CORES, seq_len, nkv, hd);
    const uint32_t v_off =
        stage_kv_to_arena(v_input, NUM_CORES, v_seq_len, v_nkv, v_hd);
    TORCH_CHECK(seq_len == v_seq_len && nkv == v_nkv && hd == v_hd,
                "arena paired insert: staged K/V geometry must match");
    rpu_launch_insert_kvcache_spm_unified_with_plan(
        k_cache, v_cache, k_off, v_off, nkv, hd, NUM_CORES,
        /*k_cache_batch_offset_elems=*/0,
        /*v_cache_batch_offset_elems=*/0,
        /*spm_rows=*/seq_len, plan);
}

void rpu_launch_pi05_kv1_direct_cache_m50d256p64_kernel(
    uint32_t k_compact_spm_offset,
    uint32_t v_compact_spm_offset,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t logical_rope_position,
    const KvInsertSegmentPlan& plan) {
    constexpr uint32_t kActiveRows = 50;
    constexpr uint32_t kPhysicalRows = 64;
    constexpr uint32_t kHeadDim = 256;
    constexpr uint16_t kPhysicalKvHeads = 8;
    constexpr uint32_t kCapacity = 2048;
    constexpr uint16_t kCompactWidth = 32;
    constexpr uint16_t kPairWidth = 16;
    constexpr uint16_t kAbiVersion = 1;
    constexpr size_t kCompactActiveBytes =
        kActiveRows * kCompactWidth * sizeof(c10::Half);
    constexpr size_t kCacheBytes =
        size_t{1} * 128 * 1 * 16 * 8 * 16 * 16 * sizeof(c10::Half);
    constexpr KernelId kKernelId =
        KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M50D256P64;

    TORCH_CHECK(
        RpuKernelGraph::has_active(),
        "Pi0.5 KV1 direct-cache requires Graph BUILD or REPLAY");
    TORCH_CHECK(
        plan.route() == KvInsertRoute::PAD16_V16 &&
            plan.logical_rows() == kActiveRows &&
            plan.physical_rows() == kPhysicalRows &&
            plan.segment_count() == 1 &&
            plan.segment(0).kernel == KvInsertKernel::V16 &&
            (plan.segment(0).position == 544 || plan.segment(0).position == 800 ||
             plan.segment(0).position == 832 || plan.segment(0).position == 576 ||
             plan.segment(0).position == 864 || plan.segment(0).position == 896 || plan.segment(0).position == 608 || plan.segment(0).position == 640) &&
            plan.segment(0).token_offset == 0 &&
            plan.segment(0).rows == kPhysicalRows,
        "Pi0.5 KV1 direct-cache requires the exact P544/P576/P608/P640/P800/P832/P864/P896/M50/PAD64 plan");
    TORCH_CHECK(
        logical_rope_position >= 0 && logical_rope_position <= UINT32_MAX,
        "Pi0.5 KV1 direct-cache logical RoPE position exceeds u32");

    const auto cache_ok = [](const at::Tensor& cache) {
        return cache.defined() &&
            cache.device().type() == c10::DeviceType::PrivateUse1 &&
            cache.scalar_type() == at::kHalf && cache.is_contiguous() &&
            cache.storage_offset() == 0 && cache.dim() == 7 &&
            cache.size(0) == 1 && cache.size(1) == 128 &&
            cache.size(2) == 1 && cache.size(3) == 16 &&
            cache.size(4) == 8 && cache.size(5) == 16 &&
            cache.size(6) == 16 && cache.nbytes() == kCacheBytes;
    };
    TORCH_CHECK(
        cache_ok(k_cache) && cache_ok(v_cache) &&
            k_cache.device() == v_cache.device(),
        "Pi0.5 KV1 direct-cache requires full contiguous FP16 K/V owners "
        "[1,128,1,16,8,16,16]");
    const auto table_ok = [logical_rope_position](const at::Tensor& table) {
        return table.defined() &&
            table.device().type() == c10::DeviceType::PrivateUse1 &&
            table.scalar_type() == at::kHalf && table.is_contiguous() &&
            table.storage_offset() == 0 && table.dim() == 2 &&
            table.size(1) == 128 && logical_rope_position <= table.size(0) &&
            kActiveRows <= table.size(0) - logical_rope_position &&
            table.nbytes() ==
                static_cast<size_t>(table.size(0) * table.size(1)) *
                    sizeof(c10::Half);
    };
    TORCH_CHECK(
        table_ok(cos) && table_ok(sin) && cos.sizes() == sin.sizes() &&
            cos.device() == k_cache.device() && sin.device() == k_cache.device(),
        "Pi0.5 KV1 direct-cache requires retained contiguous FP16 cos/sin "
        "[table_rows,128] covering the logical action rows");

    TORCH_CHECK(
        SPM_ALLOC.is_initialized(),
        "Pi0.5 KV1 direct-cache requires the initialized SPM allocator");
    const uint64_t spm_base = SPM_ALLOC.addr(0, 0);
    auto absolute_spm = [spm_base](uint32_t offset, const char* role) {
        TORCH_CHECK(
            (offset & 255u) == 0 &&
                kCompactActiveBytes <= SpmAllocator::SPM_USABLE &&
                offset <= SpmAllocator::SPM_USABLE - kCompactActiveBytes,
            "Pi0.5 KV1 direct-cache ", role,
            " typed SPM offset is outside the active compact extent");
        const uint64_t absolute = SPM_ALLOC.addr(0, offset);
        TORCH_CHECK(
            absolute == spm_base + offset && (absolute & 255u) == 0 &&
                absolute - spm_base <=
                    SpmAllocator::SPM_USABLE - kCompactActiveBytes &&
                absolute <= UINT32_MAX,
            "Pi0.5 KV1 direct-cache ", role,
            " failed its single typed-offset to absolute-SPM conversion");
        return static_cast<uint32_t>(absolute);
    };
    const uint32_t k_spm = absolute_spm(k_compact_spm_offset, "K input");
    const uint32_t v_spm = absolute_spm(v_compact_spm_offset, "V input");
    TORCH_CHECK(
        uint64_t{k_spm} + kCompactActiveBytes <= v_spm ||
            uint64_t{v_spm} + kCompactActiveBytes <= k_spm,
        "Pi0.5 KV1 direct-cache compact K/V active SPM intervals overlap");

    const uint64_t k_cache_addr = RpuGetDevAddr(k_cache.data_ptr<c10::Half>());
    const uint64_t v_cache_addr = RpuGetDevAddr(v_cache.data_ptr<c10::Half>());
    const uint64_t cos_addr = RpuGetDevAddr(cos.data_ptr<c10::Half>());
    const uint64_t sin_addr = RpuGetDevAddr(sin.data_ptr<c10::Half>());
    const auto shift8_ok = [](uint64_t address) {
        return address != 0 && (address & 255u) == 0 &&
            (address >> 8) <= UINT32_MAX;
    };
    TORCH_CHECK(
        shift8_ok(k_cache_addr) && shift8_ok(v_cache_addr) &&
            shift8_ok(cos_addr) && shift8_ok(sin_addr),
        "Pi0.5 KV1 direct-cache DDR owners violate the shift8 ABI");
    const auto disjoint = [](uint64_t lhs, size_t lhs_bytes,
                             uint64_t rhs, size_t rhs_bytes) {
        return lhs + lhs_bytes <= rhs || rhs + rhs_bytes <= lhs;
    };
    const size_t table_bytes = cos.nbytes();
    TORCH_CHECK(
        disjoint(k_cache_addr, kCacheBytes, v_cache_addr, kCacheBytes) &&
            disjoint(k_cache_addr, kCacheBytes, cos_addr, table_bytes) &&
            disjoint(k_cache_addr, kCacheBytes, sin_addr, table_bytes) &&
            disjoint(v_cache_addr, kCacheBytes, cos_addr, table_bytes) &&
            disjoint(v_cache_addr, kCacheBytes, sin_addr, table_bytes) &&
            disjoint(cos_addr, table_bytes, sin_addr, table_bytes),
        "Pi0.5 KV1 direct-cache requires four independent DDR owners");

    auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
        kKernelId,
        std::vector<GraphDdrRegisterOperandSpec>{
            GraphDdrRegisterOperandSpec{
                GraphDdrRegisterAbi{
                    GraphDdrRegisterRole::KeyCache,
                    GraphDdrRegisterAccess::Write,
                    GraphDdrRegisterEncoding::DevAddrShift8LoHi, 4, 5},
                k_cache},
            GraphDdrRegisterOperandSpec{
                GraphDdrRegisterAbi{
                    GraphDdrRegisterRole::ValueCache,
                    GraphDdrRegisterAccess::Write,
                    GraphDdrRegisterEncoding::DevAddrShift8LoHi, 6, 7},
                v_cache},
            GraphDdrRegisterOperandSpec{
                GraphDdrRegisterAbi{
                    GraphDdrRegisterRole::SemanticInput,
                    GraphDdrRegisterAccess::Read,
                    GraphDdrRegisterEncoding::DevAddrShift8LoHi, 8, 9},
                cos},
            GraphDdrRegisterOperandSpec{
                GraphDdrRegisterAbi{
                    GraphDdrRegisterRole::SemanticInput,
                    GraphDdrRegisterAccess::Read,
                    GraphDdrRegisterEncoding::DevAddrShift8LoHi, 10, 11},
                sin}});
    auto& graph = RpuKernelGraph::active();
    graph.keep_alive(k_cache);
    graph.keep_alive(v_cache);
    graph.keep_alive(cos);
    graph.keep_alive(sin);
    rpu_ddr_flush(k_cache.data_ptr<c10::Half>());
    rpu_ddr_flush(v_cache.data_ptr<c10::Half>());
    rpu_ddr_flush(cos.data_ptr<c10::Half>());
    rpu_ddr_flush(sin.data_ptr<c10::Half>());

    Kernel_t* kernel = GET_KERNEL(kKernelId);
    TORCH_CHECK(
        kernel != nullptr,
        "Pi0.5 KV1 direct-cache optional expansion symbol is absent");
    kernel->reset_regs();
    const auto set_reg = [kernel](uint32_t index, uint16_t value) {
        TORCH_CHECK(
            kernel->set_regs(index, value) == kKernelRegOk,
            "Pi0.5 KV1 direct-cache rejected parameter ", index);
    };
    const auto set_pair = [&set_reg](uint32_t index, uint32_t value) {
        set_reg(index, static_cast<uint16_t>(value & 0xffffu));
        set_reg(index + 1, static_cast<uint16_t>(value >> 16));
    };
    set_pair(0, k_spm);
    set_pair(2, v_spm);
    // The four typed DDR pairs remain zero until writer.write below.
    for (uint32_t index = 4; index <= 10; index += 2) set_pair(index, 0);
    set_pair(12, static_cast<uint32_t>(logical_rope_position));
    set_pair(14, static_cast<uint32_t>(plan.segment(0).position));
    set_reg(16, kActiveRows);
    set_reg(17, kPhysicalRows);
    set_reg(18, kHeadDim);
    set_reg(19, kPhysicalKvHeads);
    set_reg(20, kCapacity);
    set_reg(21, kCompactWidth);
    set_reg(22, kPairWidth);
    set_reg(23, kAbiVersion);
    for (uint32_t index = 24; index < 64; ++index) set_reg(index, 0);
    set_reg(64, 4);
    set_reg(65, 1);
    set_reg(66, 1);
    set_reg(67, 0);
    writer.write(*kernel);

    auto* queue = GET_QUEUE(NUM_CORES);
    TORCH_CHECK(queue != nullptr,
                "Pi0.5 KV1 direct-cache 8-core queue is absent");
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(
        *kernel, {4, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_pi05_prefill_kv1_direct_cache_m400d256p400_kernel(
    v3::SpmOffset k_compact_spm_offset,
    v3::SpmOffset v_compact_spm_offset,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t logical_rope_position,
    const KvInsertSegmentPlan& plan) {
    TORCH_CHECK(plan.logical_rows() == 272 || plan.logical_rows() == 288 ||
                plan.logical_rows() == 304 || plan.logical_rows() == 320 ||
                plan.logical_rows() == 400 ||
                plan.logical_rows() == 416 || plan.logical_rows() == 432 || plan.logical_rows() == 448,
                "Pi0.5 prefill KV1 direct-cache requires C272/C288/C304/C320/C400/C416/C432/C448");
    const uint32_t kActiveRows = static_cast<uint32_t>(plan.logical_rows());
    const uint32_t kPhysicalRows = kActiveRows;
    constexpr uint32_t kHeadDim = 256;
    constexpr uint16_t kPhysicalKvHeads = 8;
    constexpr uint32_t kCapacity = 2048;
    constexpr uint16_t kCompactWidth = 32;
    constexpr uint16_t kPairWidth = 16;
    constexpr uint16_t kAbiVersion = 1;
    const size_t kCompactActiveBytes =
        kActiveRows * kCompactWidth * sizeof(c10::Half);
    constexpr size_t kCacheBytes =
        size_t{1} * 128 * 1 * 16 * 8 * 16 * 16 * sizeof(c10::Half);
    const KernelId kKernelId = kActiveRows == 432
        ? KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M432D256P432
        : kActiveRows == 448
        ? KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M448D256P448
        : kActiveRows == 416
        ? KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M416D256P416
        : kActiveRows == 288
        ? KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M288D256P288
        : kActiveRows == 304
        ? KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M304D256P304
        : kActiveRows == 320
        ? KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M320D256P320
        : kActiveRows == 272
        ? KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M272D256P272
        : KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M400D256P400;

    TORCH_CHECK(
        RpuKernelGraph::has_active(),
        "Pi0.5 prefill KV1 direct-cache requires Graph BUILD or REPLAY");
    TORCH_CHECK(
        plan.route() == KvInsertRoute::ALIGNED_V16 &&
            plan.logical_rows() == kActiveRows &&
            plan.physical_rows() == kPhysicalRows &&
            plan.segment_count() == 1 &&
            plan.segment(0).kernel == KvInsertKernel::V16 &&
            (plan.segment(0).position == 0 ||
             plan.segment(0).position == kActiveRows) &&
            plan.segment(0).token_offset == 0 &&
            plan.segment(0).rows == kPhysicalRows,
        "Pi0.5 prefill KV1 direct-cache requires an exact P544/C272, P576/C288, P608/C304, P640/C320, P800/C400, P832/C416, P864/C432 or P896/C448 "
        "ALIGNED_V16 plan at physical start 0 or chunk rows");
    TORCH_CHECK(
        logical_rope_position == plan.segment(0).position,
        "Pi0.5 unpadded prefill KV1 direct-cache logical and physical starts must match");

    const auto cache_ok = [](const at::Tensor& cache) {
        return cache.defined() &&
            cache.device().type() == c10::DeviceType::PrivateUse1 &&
            cache.scalar_type() == at::kHalf && cache.is_contiguous() &&
            cache.storage_offset() == 0 && cache.dim() == 7 &&
            cache.size(0) == 1 && cache.size(1) == 128 &&
            cache.size(2) == 1 && cache.size(3) == 16 &&
            cache.size(4) == 8 && cache.size(5) == 16 &&
            cache.size(6) == 16 && cache.nbytes() == kCacheBytes;
    };
    TORCH_CHECK(
        cache_ok(k_cache) && cache_ok(v_cache) &&
            k_cache.device() == v_cache.device(),
        "Pi0.5 prefill KV1 direct-cache requires full contiguous FP16 K/V "
        "owners [1,128,1,16,8,16,16]");
    const auto table_ok = [logical_rope_position, kActiveRows](const at::Tensor& table) {
        return table.defined() &&
            table.device().type() == c10::DeviceType::PrivateUse1 &&
            table.scalar_type() == at::kHalf && table.is_contiguous() &&
            table.storage_offset() == 0 && table.dim() == 2 &&
            table.size(1) == 128 && table.size(0) >= 2 * kActiveRows &&
            logical_rope_position <= table.size(0) &&
            kActiveRows <= table.size(0) - logical_rope_position &&
            table.nbytes() ==
                static_cast<size_t>(table.size(0) * table.size(1)) *
                    sizeof(c10::Half);
    };
    TORCH_CHECK(
        table_ok(cos) && table_ok(sin) && cos.sizes() == sin.sizes() &&
            cos.device() == k_cache.device() && sin.device() == k_cache.device(),
        "Pi0.5 prefill KV1 direct-cache requires retained contiguous FP16 "
        "cos/sin [table_rows>=prefix,128]");

    TORCH_CHECK(
        SPM_ALLOC.is_initialized(),
        "Pi0.5 prefill KV1 direct-cache requires the initialized SPM "
        "allocator");
    const uint64_t spm_base = SPM_ALLOC.addr(0, 0);
    auto absolute_spm = [spm_base, kCompactActiveBytes](v3::SpmOffset offset,
                                   const char* role) {
        const uint32_t raw_offset = offset.value;
        TORCH_CHECK(
            (raw_offset & 255u) == 0 &&
                kCompactActiveBytes <= SpmAllocator::SPM_USABLE &&
                raw_offset <= SpmAllocator::SPM_USABLE - kCompactActiveBytes,
            "Pi0.5 prefill KV1 direct-cache ", role,
            " typed SPM offset is outside the active compact extent");
        const uint64_t absolute = SPM_ALLOC.addr(0, raw_offset);
        TORCH_CHECK(
            absolute == spm_base + raw_offset && (absolute & 255u) == 0 &&
                absolute - spm_base <=
                    SpmAllocator::SPM_USABLE - kCompactActiveBytes &&
                absolute <= UINT32_MAX,
            "Pi0.5 prefill KV1 direct-cache ", role,
            " failed its single typed-offset to absolute-SPM conversion");
        return static_cast<uint32_t>(absolute);
    };
    const uint32_t k_spm = absolute_spm(k_compact_spm_offset, "K input");
    const uint32_t v_spm = absolute_spm(v_compact_spm_offset, "V input");
    TORCH_CHECK(
        uint64_t{k_spm} + kCompactActiveBytes <= v_spm ||
            uint64_t{v_spm} + kCompactActiveBytes <= k_spm,
        "Pi0.5 prefill KV1 direct-cache compact K/V active SPM intervals "
        "overlap");

    const uint64_t k_cache_addr =
        RpuGetDevAddr(k_cache.data_ptr<c10::Half>());
    const uint64_t v_cache_addr =
        RpuGetDevAddr(v_cache.data_ptr<c10::Half>());
    const uint64_t cos_addr = RpuGetDevAddr(cos.data_ptr<c10::Half>());
    const uint64_t sin_addr = RpuGetDevAddr(sin.data_ptr<c10::Half>());
    const auto shift8_ok = [](uint64_t address) {
        return address != 0 && (address & 255u) == 0 &&
            (address >> 8) <= UINT32_MAX;
    };
    TORCH_CHECK(
        shift8_ok(k_cache_addr) && shift8_ok(v_cache_addr) &&
            shift8_ok(cos_addr) && shift8_ok(sin_addr),
        "Pi0.5 prefill KV1 direct-cache DDR owners violate the shift8 ABI");
    const auto disjoint = [](uint64_t lhs, size_t lhs_bytes,
                             uint64_t rhs, size_t rhs_bytes) {
        return lhs + lhs_bytes <= rhs || rhs + rhs_bytes <= lhs;
    };
    const size_t table_bytes = cos.nbytes();
    TORCH_CHECK(
        disjoint(k_cache_addr, kCacheBytes, v_cache_addr, kCacheBytes) &&
            disjoint(k_cache_addr, kCacheBytes, cos_addr, table_bytes) &&
            disjoint(k_cache_addr, kCacheBytes, sin_addr, table_bytes) &&
            disjoint(v_cache_addr, kCacheBytes, cos_addr, table_bytes) &&
            disjoint(v_cache_addr, kCacheBytes, sin_addr, table_bytes) &&
            disjoint(cos_addr, table_bytes, sin_addr, table_bytes),
        "Pi0.5 prefill KV1 direct-cache requires four independent DDR "
        "owners");

    auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
        kKernelId,
        std::vector<GraphDdrRegisterOperandSpec>{
            GraphDdrRegisterOperandSpec{
                GraphDdrRegisterAbi{
                    GraphDdrRegisterRole::KeyCache,
                    GraphDdrRegisterAccess::Write,
                    GraphDdrRegisterEncoding::DevAddrShift8LoHi, 4, 5},
                k_cache},
            GraphDdrRegisterOperandSpec{
                GraphDdrRegisterAbi{
                    GraphDdrRegisterRole::ValueCache,
                    GraphDdrRegisterAccess::Write,
                    GraphDdrRegisterEncoding::DevAddrShift8LoHi, 6, 7},
                v_cache},
            GraphDdrRegisterOperandSpec{
                GraphDdrRegisterAbi{
                    GraphDdrRegisterRole::SemanticInput,
                    GraphDdrRegisterAccess::Read,
                    GraphDdrRegisterEncoding::DevAddrShift8LoHi, 8, 9},
                cos},
            GraphDdrRegisterOperandSpec{
                GraphDdrRegisterAbi{
                    GraphDdrRegisterRole::SemanticInput,
                    GraphDdrRegisterAccess::Read,
                    GraphDdrRegisterEncoding::DevAddrShift8LoHi, 10, 11},
                sin}});
    auto& graph = RpuKernelGraph::active();
    graph.keep_alive(k_cache);
    graph.keep_alive(v_cache);
    graph.keep_alive(cos);
    graph.keep_alive(sin);
    rpu_ddr_flush(k_cache.data_ptr<c10::Half>());
    rpu_ddr_flush(v_cache.data_ptr<c10::Half>());
    rpu_ddr_flush(cos.data_ptr<c10::Half>());
    rpu_ddr_flush(sin.data_ptr<c10::Half>());

    Kernel_t* kernel = GET_KERNEL(kKernelId);
    TORCH_CHECK(
        kernel != nullptr,
        "Pi0.5 prefill KV1 direct-cache optional expansion symbol is absent");
    kernel->reset_regs();
    const auto set_reg = [kernel](uint32_t index, uint16_t value) {
        TORCH_CHECK(
            kernel->set_regs(index, value) == kKernelRegOk,
            "Pi0.5 prefill KV1 direct-cache rejected parameter ", index);
    };
    const auto set_pair = [&set_reg](uint32_t index, uint32_t value) {
        set_reg(index, static_cast<uint16_t>(value & 0xffffu));
        set_reg(index + 1, static_cast<uint16_t>(value >> 16));
    };
    set_pair(0, k_spm);
    set_pair(2, v_spm);
    // The four typed DDR pairs stay zero until the occurrence writer binds the
    // exact retained owners after every other register is final.
    for (uint32_t index = 4; index <= 10; index += 2) set_pair(index, 0);
    set_pair(12, static_cast<uint32_t>(logical_rope_position));
    set_pair(14, static_cast<uint32_t>(plan.segment(0).position));
    set_reg(16, kActiveRows);
    set_reg(17, kPhysicalRows);
    set_reg(18, kHeadDim);
    set_reg(19, kPhysicalKvHeads);
    set_reg(20, kCapacity);
    set_reg(21, kCompactWidth);
    set_reg(22, kPairWidth);
    set_reg(23, kAbiVersion);
    for (uint32_t index = 24; index < 64; ++index) set_reg(index, 0);
    const uint16_t blocks = kActiveRows / 16;
    set_reg(64, blocks);
    set_reg(65, 1);
    set_reg(66, 1);
    set_reg(67, 0);
    writer.write(*kernel);

    auto* queue = GET_QUEUE(NUM_CORES);
    TORCH_CHECK(queue != nullptr,
                "Pi0.5 prefill KV1 direct-cache 8-core queue is absent");
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(
        *kernel, {blocks, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}
