// rpu_llama_kvcache.cpp
// LLaMA KV-Cache insert kernel launch functions

#include "rpu_ops.h"
#include "fused_model_base.h"
#include "rpu_spm_allocator.h"
#include "rhino_launch_buffer.h"
#include "rhino_launch_queue.h"
#include <c10/util/Half.h>
#include <vector>
#include <array>
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
                "v_input must be on device");

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
    // 参考 ASM 文档 (用户更正后):
    // param0/1: position (int32 value, 直接是数值)
    // param2: num_kv_percore_heads (每核的 head 数)
    // param3: head_dim
    // param4: seq_len
    // param6/7: input V SPM address (64-bit)
    // param8/9: V-Cache DDR address (64-bit)

    // 使用 Core 0 的 SPM 地址作为基准 (broadcast mode 下所有核使用相同的相对偏移)
    uint32_t v_spm_addr = spm_v[0]->get_rpu_addr();
    uint32_t position_val = static_cast<uint32_t>(position);

    // ASM kernel: data_size = param2 * param3
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
    // 每个 warp 使用 %blkid.x 来获取它处理的 seq 索引
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
                "k_input must be on device");

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

    // ASM kernel: data_size = param2 * param3
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

void rpu_launch_wall_prefill_owner_direct_kv_spm_kernel(
    at::Tensor& k_cache,
    at::Tensor& v_cache,
    uint32_t k_spm_addr,
    uint32_t v_spm_addr,
    const at::Tensor& cos,
    const at::Tensor& sin) {
    constexpr int64_t kRows = 640;
    constexpr int64_t kRotaryHalf = 64;
    constexpr uint64_t kCompactBytes =
        kRows * kRotaryHalf * sizeof(c10::Half);

    auto check_cache = [](const at::Tensor& cache, const char* name) {
        TORCH_CHECK(
            cache.defined() &&
                cache.device().type() == c10::DeviceType::PrivateUse1 &&
                cache.scalar_type() == at::kHalf && cache.is_contiguous() &&
                cache.dim() == 7 && cache.size(0) == 1 &&
                cache.size(1) >= 40 && cache.size(2) == 1 &&
                cache.size(3) == 8 && cache.size(4) == 8 &&
                cache.size(5) == 16 && cache.size(6) == 16,
            "Wall P640 owner-direct ", name,
            " cache must be contiguous RPU fp16 "
            "[1,>=40,1,8,8,16,16], got ", cache.sizes());
    };
    check_cache(k_cache, "K");
    check_cache(v_cache, "V");
    TORCH_CHECK(
        cos.defined() && sin.defined() &&
            cos.device().type() == c10::DeviceType::PrivateUse1 &&
            sin.device().type() == c10::DeviceType::PrivateUse1 &&
            cos.scalar_type() == at::kHalf && sin.scalar_type() == at::kHalf &&
            cos.is_contiguous() && sin.is_contiguous() &&
            cos.dim() == 2 && sin.dim() == 2 && cos.sizes() == sin.sizes() &&
            cos.size(0) >= kRows && cos.size(1) == kRotaryHalf,
        "Wall P640 owner-direct cos/sin must be matching contiguous RPU fp16 "
        "tables [>=640,64]");

    const uint32_t spm_base0 = SPM_ALLOC.addr(0, 0);
    auto checked_offset = [&](uint32_t address, const char* name) {
        TORCH_CHECK(
            address >= spm_base0 && address % 256 == 0 &&
                static_cast<uint64_t>(address - spm_base0) <=
                    SpmAllocator::SPM_USABLE - kCompactBytes,
            "Wall P640 owner-direct ", name,
            " must be a 256-byte-aligned local SPM range");
        return address - spm_base0;
    };
    const uint32_t k_offset = checked_offset(k_spm_addr, "K input");
    const uint32_t v_offset = checked_offset(v_spm_addr, "V input");
    TORCH_CHECK(
        static_cast<uint64_t>(k_offset) + kCompactBytes <= v_offset ||
            static_cast<uint64_t>(v_offset) + kCompactBytes <= k_offset,
        "Wall P640 owner-direct K/V SPM input ranges must not overlap");

    auto register_writer =
        RpuKernelGraph::active().stage_kernel_ddr_registers(
            KernelId::WALL_PREFILL_OWNER_DIRECT_KV_M640,
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

    const at::Tensor* ddr_operands[] = {&k_cache, &v_cache, &cos, &sin};
    for (const at::Tensor* tensor : ddr_operands) {
        rpu_ddr_flush(tensor->data_ptr());
        TORCH_CHECK(
            RpuGetDevAddr(tensor->data_ptr()) % 256 == 0,
            "Wall P640 owner-direct DDR operand is not 256-byte aligned");
    }
    const uint64_t k_cache_addr = RpuGetDevAddr(k_cache.data_ptr());
    const uint64_t v_cache_addr = RpuGetDevAddr(v_cache.data_ptr());
    const uint64_t k_cache_bytes =
        static_cast<uint64_t>(k_cache.numel()) * sizeof(c10::Half);
    const uint64_t v_cache_bytes =
        static_cast<uint64_t>(v_cache.numel()) * sizeof(c10::Half);
    TORCH_CHECK(
        k_cache_addr + k_cache_bytes <= v_cache_addr ||
            v_cache_addr + v_cache_bytes <= k_cache_addr,
        "Wall P640 owner-direct K/V DDR cache ranges must not overlap");

    Kernel_t* kernel =
        GET_KERNEL(KernelId::WALL_PREFILL_OWNER_DIRECT_KV_M640);
    TORCH_CHECK(kernel != nullptr,
                "owner_direct_kv_m640 missing from expansion ref");
    kernel->reset_regs();
    kernel->set_regs(0, static_cast<uint16_t>(k_offset));
    kernel->set_regs(1, static_cast<uint16_t>(k_offset >> 16));
    kernel->set_regs(2, static_cast<uint16_t>(v_offset));
    kernel->set_regs(3, static_cast<uint16_t>(v_offset >> 16));
    kernel->set_regs(12, static_cast<uint16_t>(0));
    register_writer.write(*kernel);

    auto* queue = GET_QUEUE(NUM_CORES);
    queue->set_broadcast_mode(true);
    const std::vector<uint8_t> cores{0, 1, 2, 3, 4, 5, 6, 7};
    queue->enqueu_kernel(*kernel, {40, 1, 1}, cores);

    rpu_ddr_flush(k_cache.data_ptr<c10::Half>());
    rpu_ddr_flush(v_cache.data_ptr<c10::Half>());
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
    // ASM kernel: data_size = param2 * param3 = elements per core per position.
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

    // ASM kernel: data_size = param2 * param3
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
    int64_t num_kv_heads, int64_t head_dim, int num_cores, bool decode,
    bool lingbot_m224 = false) {
    TORCH_CHECK(num_kv_heads == 8 && head_dim == 128 && num_cores == 8 &&
                    !(lingbot_m224 && decode) &&
                    seq_len == (decode ? 1 : (lingbot_m224 ? 224 : 192)) && position >= 0 &&
                    (decode || position % 16 == 0),
                "paired KV insert requires admitted TP8/HD128 decode1, "
                "Qwen3-VL aligned prefill192 or LingBot2 aligned prefill224");
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
    uint32_t k_spm_addr = 0;
    uint32_t v_spm_addr = 0;
    if (lingbot_m224) {
        constexpr uint64_t kInputBytes = 224 * 128 * sizeof(c10::Half);
        TORCH_CHECK(k_cache.storage_offset() == 0 && v_cache.storage_offset() == 0 &&
                        k_cache.storage().nbytes() == k_cache.nbytes() &&
                        v_cache.storage().nbytes() == v_cache.nbytes(),
                    "LingBot2 M224 paired KV requires two full cache owners");
        TORCH_CHECK(SPM_ALLOC.is_initialized(),
                    "LingBot2 M224 paired KV requires initialized SPM");
        const uint64_t spm_base = SPM_ALLOC.addr(0, 0);
        const auto absolute_spm = [spm_base](uint32_t offset) {
            TORCH_CHECK((offset & 255u) == 0 &&
                            offset <= SpmAllocator::SPM_USABLE - kInputBytes,
                        "LingBot2 M224 KV input offset violates its full57344B extent");
            const uint64_t address = SPM_ALLOC.addr(0, offset);
            TORCH_CHECK(address == spm_base + offset && (address & 255u) == 0 &&
                            address <= UINT32_MAX - (kInputBytes - 1),
                        "LingBot2 M224 KV input failed its offset-to-absolute conversion");
            return static_cast<uint32_t>(address);
        };
        k_spm_addr = absolute_spm(k_off);
        v_spm_addr = absolute_spm(v_off);
        TORCH_CHECK(uint64_t{k_spm_addr} + kInputBytes <= v_spm_addr ||
                        uint64_t{v_spm_addr} + kInputBytes <= k_spm_addr,
                    "LingBot2 M224 KV input SPM intervals overlap");
        const uint64_t k_addr = RpuGetDevAddr(k_cache.data_ptr<c10::Half>());
        const uint64_t v_addr = RpuGetDevAddr(v_cache.data_ptr<c10::Half>());
        const auto ddr_ok = [](uint64_t address, size_t bytes) {
            constexpr uint64_t limit = uint64_t{1} << 40;
            return address != 0 && (address & 255u) == 0 &&
                address < limit && bytes <= limit - address;
        };
        TORCH_CHECK(ddr_ok(k_addr, k_cache.nbytes()) && ddr_ok(v_addr, v_cache.nbytes()),
                    "LingBot2 M224 full KV owners violate the shift8 DDR address ABI");
        const bool ddr_disjoint = k_addr <= v_addr
            ? v_addr - k_addr >= k_cache.nbytes()
            : k_addr - v_addr >= v_cache.nbytes();
        TORCH_CHECK(ddr_disjoint,
                    "LingBot2 M224 requires non-overlapping full DDR KV owners");
    }
    TORCH_CHECK(KernelCache::instance().get(kernel_id) != nullptr,
                "selected paired KV insert kernel unavailable");
    auto* queue = GET_QUEUE(num_cores);
    TORCH_CHECK(queue != nullptr, "selected paired KV insert queue unavailable");
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
    Kernel_t* kernel = GET_KERNEL(kernel_id);
    if (lingbot_m224) {
        RpuKernelGraph::active().keep_alive(k_cache);
        RpuKernelGraph::active().keep_alive(v_cache);
    }
    rpu_ddr_flush(k_cache.data_ptr<c10::Half>());
    rpu_ddr_flush(v_cache.data_ptr<c10::Half>());
    if (!lingbot_m224) {
        k_spm_addr = SPM_ALLOC.addr(0, k_off);
        v_spm_addr = SPM_ALLOC.addr(0, v_off);
    }
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
    if (lingbot_m224) {
        // The unchanged paired V16 payload locks and releases core counter0.
        // Every recorded launch includes its zero-valued full u32 pair so
        // snapshot replay starts each core with the same fresh counter state.
        TORCH_CHECK(kernel->set_regs(128, uint32_t{0}) == kKernelRegOk,
                    "LingBot2 M224 paired KV rejected atomic counter0 reset");
    }
    register_writer.write(*kernel);
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

void rpu_launch_wall_base_paired_kv_spm_kernel(
    at::Tensor& k_cache, at::Tensor& v_cache,
    uint32_t k_off, uint32_t v_off, int64_t spm_rows,
    const KvInsertSegmentPlan& plan) {
    rpu_validate_kvinsert_segment_plan(plan, 8, 8, 128);
    TORCH_CHECK(plan.route() == KvInsertRoute::ALIGNED_V16 &&
                    plan.logical_rows() == 32 && plan.physical_rows() == 32 &&
                    plan.segment_count() == 1 && spm_rows == 32,
                "Wall paired KV requires one actual H32 ALIGNED_V16 plan");
    const auto& segment = plan.segment(0);
    const int64_t position = segment.position;
    TORCH_CHECK((position == 336 || position == 384) && segment.rows == 32 &&
                    segment.token_offset == 0 && segment.kernel == KvInsertKernel::V16,
                "Wall paired KV requires P336/P384 and the full 32-row input");
    const auto cache_shape = [](const at::Tensor& cache) {
        return cache.defined() &&
            cache.device().type() == c10::DeviceType::PrivateUse1 &&
            cache.scalar_type() == at::kHalf && cache.is_contiguous() &&
            cache.dim() == 7 && cache.size(0) == 1 && cache.size(1) > 0 &&
            cache.size(2) == 1 && cache.size(3) == 8 && cache.size(4) == 8 &&
            cache.size(5) == 16 && cache.size(6) == 16;
    };
    TORCH_CHECK(cache_shape(k_cache) && cache_shape(v_cache) &&
                    k_cache.device() == v_cache.device() &&
                    k_cache.size(1) == v_cache.size(1) &&
                    k_cache.size(1) >= (position + 32) / 16,
                "Wall paired KV requires matching owned FP16 7D KV8/D128 caches");
    const auto kb = reinterpret_cast<uintptr_t>(k_cache.data_ptr());
    const auto vb = reinterpret_cast<uintptr_t>(v_cache.data_ptr());
    TORCH_CHECK(kb <= vb ? vb - kb >= k_cache.nbytes()
                         : kb - vb >= v_cache.nbytes(),
                "Wall paired KV requires disjoint K/V cache byte ranges");
    constexpr uint32_t kBytes = 32 * 128 * sizeof(c10::Half);
    TORCH_CHECK(SPM_ALLOC.is_initialized() && k_off % 32 == 0 && v_off % 32 == 0 &&
                    k_off <= SpmAllocator::SPM_USABLE - kBytes &&
                    v_off <= SpmAllocator::SPM_USABLE - kBytes &&
                    (k_off <= v_off ? v_off - k_off >= kBytes
                                     : k_off - v_off >= kBytes),
                "Wall paired KV requires independent complete 8192-byte SPM inputs");
    for (int core = 0; core < 8; ++core) {
        const uint64_t base = SPM_ALLOC.addr(core, 0);
        TORCH_CHECK(base + k_off + kBytes - 1 <= UINT32_MAX &&
                        base + v_off + kBytes - 1 <= UINT32_MAX,
                    "Wall paired KV absolute SPM input exceeds uint32");
    }
    // All geometry/owner/range admission precedes typed address staging.
    const auto id = KernelId::QWEN3VL_INSERT_KV_CACHE_V16;
    auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
        id, std::vector<GraphDdrRegisterOperandSpec>{
            {GraphDdrRegisterAbi{GraphDdrRegisterRole::KeyCache,
                GraphDdrRegisterAccess::Write,
                GraphDdrRegisterEncoding::DevAddrShift8LoHi, 8, 9}, k_cache},
            {GraphDdrRegisterAbi{GraphDdrRegisterRole::ValueCache,
                GraphDdrRegisterAccess::Write,
                GraphDdrRegisterEncoding::DevAddrShift8LoHi, 12, 13}, v_cache}});
    rpu_ddr_flush(k_cache.data_ptr<c10::Half>());
    rpu_ddr_flush(v_cache.data_ptr<c10::Half>());
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel != nullptr, "Wall paired KV payload unavailable");
    kernel->reset_regs();
    const auto set16 = [kernel](uint32_t reg, uint16_t value) {
        TORCH_CHECK(kernel->set_regs(reg, value) == 0,
                    "Wall paired KV register write rejected at ", reg);
    };
    const uint32_t ka = SPM_ALLOC.addr(0, k_off);
    const uint32_t va = SPM_ALLOC.addr(0, v_off);
    set16(0, static_cast<uint16_t>(position));
    set16(1, 0);
    set16(2, 1);
    set16(3, 128);
    set16(6, static_cast<uint16_t>(ka));
    set16(7, static_cast<uint16_t>(ka >> 16));
    set16(10, static_cast<uint16_t>(va));
    set16(11, static_cast<uint16_t>(va >> 16));
    set16(21, 8);
    writer.write(*kernel);
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {2, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
    rpu_ddr_flush(k_cache.data_ptr<c10::Half>());
    rpu_ddr_flush(v_cache.data_ptr<c10::Half>());
}

void rpu_launch_hyvla_expert_paired_kv_spm_kernel(
    at::Tensor& k_cache, at::Tensor& v_cache,
    uint32_t k_off, uint32_t v_off, int64_t spm_rows,
    const KvInsertSegmentPlan& plan) {
    rpu_validate_kvinsert_segment_plan(plan, 8, 8, 128);
    TORCH_CHECK(plan.route() == KvInsertRoute::PAD16_V16 &&
                    plan.logical_rows() == 51 && plan.physical_rows() == 64 &&
                    plan.segment_count() == 1 && spm_rows == 64,
                "HyVLA paired KV requires one actual S51/S64 PAD16_V16 plan");
    const auto& segment = plan.segment(0);
    const int64_t position = segment.position;
    TORCH_CHECK((position >= 128 && position <= 240 && position % 16 == 0) &&
                    segment.rows == 64 &&
                    segment.token_offset == 0 && segment.kernel == KvInsertKernel::V16,
                "HyVLA paired KV requires P128..240 aligned16 and the full 64-row input");
    const auto cache_shape = [](const at::Tensor& cache) {
        return cache.defined() &&
            cache.device().type() == c10::DeviceType::PrivateUse1 &&
            cache.scalar_type() == at::kHalf && cache.is_contiguous() &&
            cache.dim() == 7 && cache.size(0) == 1 && cache.size(1) > 0 &&
            cache.size(2) == 1 && cache.size(3) == 8 && cache.size(4) == 8 &&
            cache.size(5) == 16 && cache.size(6) == 16;
    };
    TORCH_CHECK(cache_shape(k_cache) && cache_shape(v_cache) &&
                    k_cache.device() == v_cache.device() &&
                    k_cache.size(1) == v_cache.size(1) &&
                    k_cache.size(1) >= (position + 64 + 15) / 16,
                "HyVLA paired KV requires matching owned FP16 7D KV8/D128 caches");
    const auto kb = reinterpret_cast<uintptr_t>(k_cache.data_ptr());
    const auto vb = reinterpret_cast<uintptr_t>(v_cache.data_ptr());
    TORCH_CHECK(kb <= vb ? vb - kb >= k_cache.nbytes()
                         : kb - vb >= v_cache.nbytes(),
                "HyVLA paired KV requires disjoint K/V cache byte ranges");
    constexpr uint32_t kBytes = 64 * 128 * sizeof(c10::Half);
    TORCH_CHECK(SPM_ALLOC.is_initialized() && k_off % 32 == 0 && v_off % 32 == 0 &&
                    k_off <= SpmAllocator::SPM_USABLE - kBytes &&
                    v_off <= SpmAllocator::SPM_USABLE - kBytes &&
                    (k_off <= v_off ? v_off - k_off >= kBytes
                                     : k_off - v_off >= kBytes),
                "HyVLA paired KV requires independent complete 16384-byte SPM inputs");
    for (int core = 0; core < 8; ++core) {
        const uint64_t base = SPM_ALLOC.addr(core, 0);
        TORCH_CHECK(base + k_off + kBytes - 1 <= UINT32_MAX &&
                        base + v_off + kBytes - 1 <= UINT32_MAX,
                    "HyVLA paired KV absolute SPM input exceeds uint32");
    }
    // All geometry/owner/range admission precedes typed address staging.
    const auto id = KernelId::QWEN3VL_INSERT_KV_CACHE_V16;
    auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
        id, std::vector<GraphDdrRegisterOperandSpec>{
            {GraphDdrRegisterAbi{GraphDdrRegisterRole::KeyCache,
                GraphDdrRegisterAccess::Write,
                GraphDdrRegisterEncoding::DevAddrShift8LoHi, 8, 9}, k_cache},
            {GraphDdrRegisterAbi{GraphDdrRegisterRole::ValueCache,
                GraphDdrRegisterAccess::Write,
                GraphDdrRegisterEncoding::DevAddrShift8LoHi, 12, 13}, v_cache}});
    rpu_ddr_flush(k_cache.data_ptr<c10::Half>());
    rpu_ddr_flush(v_cache.data_ptr<c10::Half>());
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel != nullptr, "HyVLA paired KV payload unavailable");
    kernel->reset_regs();
    const auto set16 = [kernel](uint32_t reg, uint16_t value) {
        TORCH_CHECK(kernel->set_regs(reg, value) == 0,
                    "HyVLA paired KV register write rejected at ", reg);
    };
    const uint32_t ka = SPM_ALLOC.addr(0, k_off);
    const uint32_t va = SPM_ALLOC.addr(0, v_off);
    set16(0, static_cast<uint16_t>(position));
    set16(1, 0);
    set16(2, 1);
    set16(3, 128);
    set16(6, static_cast<uint16_t>(ka));
    set16(7, static_cast<uint16_t>(ka >> 16));
    set16(10, static_cast<uint16_t>(va));
    set16(11, static_cast<uint16_t>(va >> 16));
    set16(21, 8);
    writer.write(*kernel);
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {4, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
    rpu_ddr_flush(k_cache.data_ptr<c10::Half>());
    rpu_ddr_flush(v_cache.data_ptr<c10::Half>());
}

void rpu_launch_insert_kv_cache_spm_unified_decode(
    at::Tensor& k_cache, at::Tensor& v_cache, int64_t position,
    uint32_t k_off, uint32_t v_off, int64_t seq_len,
    int64_t num_kv_heads, int64_t head_dim, int num_cores) {
    launch_owned_paired_kv_insert(k_cache, v_cache, position, k_off, v_off,
        seq_len, num_kv_heads, head_dim, num_cores, true);
}

void rpu_launch_lingbot2_insert_kv_m224_spm_with_plan(
    at::Tensor& k_cache, at::Tensor& v_cache,
    uint32_t k_off, uint32_t v_off, int64_t spm_rows,
    const KvInsertSegmentPlan& plan) {
    TORCH_CHECK(RpuKernelGraph::has_active(),
                "LingBot2 M224 paired KV requires an active Graph context");
    rpu_validate_kvinsert_segment_plan(plan, 8, 8, 128);
    TORCH_CHECK(spm_rows >= 224 && plan.logical_rows() == 224 &&
                    plan.physical_rows() == 224 &&
                    plan.route() == KvInsertRoute::ALIGNED_V16 &&
                    plan.segment_count() == 1 &&
                    plan.segment(0).kernel == KvInsertKernel::V16 &&
                    plan.segment(0).rows == 224 && plan.segment(0).token_offset == 0,
                "LingBot2 M224 paired KV requires the exact single224-row V16 plan");
    launch_owned_paired_kv_insert(k_cache, v_cache, plan.segment(0).position,
        k_off, v_off, 224, 8, 128, 8, false, true);
}

void rpu_launch_lingbot2_insert_kv_t51_spm_kernel(
    at::Tensor& k_cache, at::Tensor& v_cache, int64_t position,
    uint32_t k_off, uint32_t v_off, int64_t spm_rows) {
    constexpr int64_t kRows = 51;
    constexpr uint64_t kInputBytes = kRows * 128 * sizeof(c10::Half);
    constexpr KernelId kKernelId = KernelId::LINGBOT2_INSERT_KV_T51;
    TORCH_CHECK(RpuKernelGraph::has_active(),
                "LingBot2 T51 KV insert requires an active Graph context");
    TORCH_CHECK(spm_rows >= kRows && position >= 0 && position % 16 == 0 &&
                    static_cast<uint64_t>(position) <= UINT32_MAX - kRows,
                "LingBot2 T51 KV insert requires51 available rows and an aligned u32 position");
    const auto cache_ok = [](const at::Tensor& cache) {
        return cache.defined() &&
            cache.device().type() == c10::DeviceType::PrivateUse1 &&
            cache.scalar_type() == at::kHalf && cache.is_contiguous() &&
            cache.storage_offset() == 0 && cache.dim() == 7 &&
            cache.size(0) == 1 && cache.size(1) > 0 && cache.size(2) == 1 &&
            cache.size(3) == 8 && cache.size(4) == 8 &&
            cache.size(5) == 16 && cache.size(6) == 16 &&
            cache.storage().nbytes() == cache.nbytes();
    };
    TORCH_CHECK(cache_ok(k_cache) && cache_ok(v_cache) &&
                    k_cache.device() == v_cache.device() &&
                    k_cache.size(1) == v_cache.size(1) &&
                    (position + kRows + 15) / 16 <= k_cache.size(1),
                "LingBot2 T51 KV insert requires matching full FP16 RPU "
                "[1,S16,1,8,8,16,16] owners covering the logical append");
    TORCH_CHECK(SPM_ALLOC.is_initialized(),
                "LingBot2 T51 KV insert requires initialized SPM");
    const uint64_t spm_base = SPM_ALLOC.addr(0, 0);
    const auto absolute_spm = [spm_base](uint32_t offset) {
        TORCH_CHECK((offset & 255u) == 0 &&
                        offset <= SpmAllocator::SPM_USABLE - kInputBytes,
                    "LingBot2 T51 KV input offset is misaligned or outside its full13056B extent");
        const uint64_t address = SPM_ALLOC.addr(0, offset);
        TORCH_CHECK(address == spm_base + offset && (address & 255u) == 0 &&
                        address <= UINT32_MAX - (kInputBytes - 1),
                    "LingBot2 T51 KV input failed its single offset-to-absolute conversion");
        return static_cast<uint32_t>(address);
    };
    const uint32_t k_spm = absolute_spm(k_off);
    const uint32_t v_spm = absolute_spm(v_off);
    TORCH_CHECK(uint64_t{k_spm} + kInputBytes <= v_spm ||
                    uint64_t{v_spm} + kInputBytes <= k_spm,
                "LingBot2 T51 KV input SPM intervals overlap");
    const uint64_t k_addr = RpuGetDevAddr(k_cache.data_ptr<c10::Half>());
    const uint64_t v_addr = RpuGetDevAddr(v_cache.data_ptr<c10::Half>());
    const auto ddr_ok = [](uint64_t address, size_t bytes) {
        return address != 0 && (address & 255u) == 0 &&
            (address >> 8) <= UINT32_MAX && bytes <= (uint64_t{1} << 40) - address;
    };
    TORCH_CHECK(ddr_ok(k_addr, k_cache.nbytes()) && ddr_ok(v_addr, v_cache.nbytes()),
                "LingBot2 T51 KV full DDR owners violate the shift8 address ABI");
    const bool disjoint = k_addr <= v_addr
        ? v_addr - k_addr >= k_cache.nbytes()
        : k_addr - v_addr >= v_cache.nbytes();
    TORCH_CHECK(disjoint, "LingBot2 T51 KV requires non-overlapping full K/V owners");
    TORCH_CHECK(KernelCache::instance().get(kKernelId) != nullptr,
                "LingBot2 T51 KV optional expansion symbol is absent");
    auto* queue = GET_QUEUE(8);
    TORCH_CHECK(queue != nullptr, "LingBot2 T51 KV eight-core queue is absent");

    auto& graph = RpuKernelGraph::active();
    auto writer = graph.stage_kernel_ddr_registers(kKernelId,
        std::vector<GraphDdrRegisterOperandSpec>{
            {GraphDdrRegisterAbi{GraphDdrRegisterRole::KeyCache,
                GraphDdrRegisterAccess::Write,
                GraphDdrRegisterEncoding::DevAddrShift8LoHi, 8, 9}, k_cache},
            {GraphDdrRegisterAbi{GraphDdrRegisterRole::ValueCache,
                GraphDdrRegisterAccess::Write,
                GraphDdrRegisterEncoding::DevAddrShift8LoHi, 12, 13}, v_cache}});
    Kernel_t* kernel = GET_KERNEL(kKernelId);
    graph.keep_alive(k_cache);
    graph.keep_alive(v_cache);
    rpu_ddr_flush(k_cache.data_ptr<c10::Half>());
    rpu_ddr_flush(v_cache.data_ptr<c10::Half>());
    kernel->reset_regs();
    const auto set_reg = [kernel](uint32_t index, uint16_t value) {
        TORCH_CHECK(kernel->set_regs(index, value) == kKernelRegOk,
                    "LingBot2 T51 KV rejected parameter ", index);
    };
    const auto set_pair = [&set_reg](uint32_t index, uint32_t value) {
        set_reg(index, static_cast<uint16_t>(value));
        set_reg(index + 1, static_cast<uint16_t>(value >> 16));
    };
    set_pair(0, static_cast<uint32_t>(position));
    set_reg(2, 1);
    set_reg(3, 128);
    set_reg(4, kRows);
    set_pair(6, k_spm);
    set_pair(10, v_spm);
    set_reg(21, 8);
    writer.write(*kernel);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {4, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
    rpu_ddr_flush(k_cache.data_ptr<c10::Half>());
    rpu_ddr_flush(v_cache.data_ptr<c10::Half>());
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
    const KvInsertSegmentPlan& plan,
    uint16_t rotated_skip_mask,
    const Pi05SplitKvRopePack* q_pack) {
    // M50/PAD64 (four 16-row blocks, grid 4) or the H10 M10/PAD16 sibling
    // (one block, grid 1); the logical row count selects the payload.
    const bool h10 = plan.logical_rows() == 10;
    const uint32_t kActiveRows = h10 ? 10 : 50;
    const uint32_t kPhysicalRows = h10 ? 16 : 64;
    const uint16_t kGrid = h10 ? 1 : 4;
    constexpr uint32_t kHeadDim = 256;
    constexpr uint16_t kPhysicalKvHeads = 8;
    constexpr uint32_t kCapacity = 2048;
    constexpr uint16_t kCompactWidth = 32;
    constexpr uint16_t kPairWidth = 16;
    constexpr uint16_t kAbiVersion = 1;
    // One FP16 cos or sin row, in the DDR tables and in the SPM rows alike.
    constexpr uint32_t kRopeRowBytes = kHeadDim / 2 * sizeof(c10::Half);
    const size_t kCompactActiveBytes =
        kActiveRows * kCompactWidth * sizeof(c10::Half);
    constexpr size_t kCacheBytes =
        size_t{1} * 128 * 1 * 16 * 8 * 16 * 16 * sizeof(c10::Half);
    // A launch that packs has the chunk's cos/sin rows in SPM, so its writer
    // reads them there too when the REF has that sibling (registers 40..45).
    const KernelId kKernelId = pi05_kv1_direct_cache_q_rope_pack(kActiveRows)
        ? (q_pack && pi05_kv1_direct_cache_spm_rope()
               ? KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_SPM_ROPE_M64
               : KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_M64)
        : pi05_kv1_direct_cache_rotated(kActiveRows)
        ? KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_ROT
        : pi05_kv1_direct_cache_overlapped(kActiveRows)
        ? KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_OVERLAP
        : h10 ? KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M10D256P16
              : KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M50D256P64;
    const bool spm_rope =
        kKernelId == KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_SPM_ROPE_M64;
    // The fused payload is the rotated writer with the RoPE pack's blocks
    // after its own and its registers 32 higher.
    const bool fused = spm_rope ||
        kKernelId == KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_M64;
    const uint32_t base = fused ? 32 : 0;
    TORCH_CHECK(
        rotated_skip_mask == 0 || fused ||
            kKernelId == KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_ROT,
        "Pi0.5 KV1 direct-cache: only the rotated writer skips head slots");
    TORCH_CHECK(
        q_pack == nullptr || (fused && q_pack->plan.seq_q == kActiveRows),
        "Pi0.5 KV1 direct-cache: only the fused payload packs the query rows, "
        "and only its own rows");

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
             plan.segment(0).position == 864 || plan.segment(0).position == 896 || plan.segment(0).position == 608 || plan.segment(0).position == 640 ||
             plan.segment(0).position == 672 || plan.segment(0).position == 928 ||
             plan.segment(0).position == 960) &&
            plan.segment(0).token_offset == 0 &&
            plan.segment(0).rows == kPhysicalRows,
        "Pi0.5 KV1 direct-cache requires the exact P544/P576/P608/P640/P672/P800/P832/P864/P896/P928/P960 M50/PAD64 or M10/PAD16 plan");
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
    const auto table_ok = [logical_rope_position, kActiveRows](const at::Tensor& table) {
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
    auto absolute_spm = [spm_base, kCompactActiveBytes](uint32_t offset, const char* role) {
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
    if (q_pack) {
        pi05_split_kv_check_rope_pack_spm(q_pack->plan, q_pack->rope_rows_off);
        // The pack writes every core's packed rows while the cache body reads
        // its compact K/V rows; it writes no other SPM, the cache body none.
        const uint64_t qp = SPM_ALLOC.addr(0, q_pack->qp_off);
        const uint64_t qp_bytes = static_cast<uint64_t>(q_pack->plan.qp_bytes());
        const auto clear_of = [qp, qp_bytes, kCompactActiveBytes](uint32_t spm) {
            return qp + qp_bytes <= spm || uint64_t{spm} + kCompactActiveBytes <= qp;
        };
        TORCH_CHECK(
            clear_of(k_spm) && clear_of(v_spm),
            "Pi0.5 KV1 direct-cache + RoPE pack: the packed query rows overlap "
            "the compact K/V rows");
    }

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

    std::vector<GraphDdrRegisterOperandSpec> ddr_operands{
        GraphDdrRegisterOperandSpec{
            GraphDdrRegisterAbi{
                GraphDdrRegisterRole::KeyCache,
                GraphDdrRegisterAccess::Write,
                GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                base + 4, base + 5},
            k_cache},
        GraphDdrRegisterOperandSpec{
            GraphDdrRegisterAbi{
                GraphDdrRegisterRole::ValueCache,
                GraphDdrRegisterAccess::Write,
                GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                base + 6, base + 7},
            v_cache}};
    // The SPM-RoPE sibling's registers 8..11 hold SPM addresses, set below.
    if (!spm_rope) {
        ddr_operands.push_back(GraphDdrRegisterOperandSpec{
            GraphDdrRegisterAbi{
                GraphDdrRegisterRole::SemanticInput,
                GraphDdrRegisterAccess::Read,
                GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                base + 8, base + 9},
            cos});
        ddr_operands.push_back(GraphDdrRegisterOperandSpec{
            GraphDdrRegisterAbi{
                GraphDdrRegisterRole::SemanticInput,
                GraphDdrRegisterAccess::Read,
                GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                base + 10, base + 11},
            sin});
    }
    auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
        kKernelId, std::move(ddr_operands));
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
    // The pack's registers 0..14 and SCM slot 0; its blocks follow the cache's.
    const uint16_t pack_blocks = q_pack
        ? pi05_split_kv_write_rope_pack_spm_regs(
              kernel, q_pack->rope_rows_off, q_pack->q_off, q_pack->qp_off, q_pack->plan)
        : 0;
    const auto set_raw = [kernel](uint32_t index, uint16_t value) {
        TORCH_CHECK(
            kernel->set_regs(index, value) == kKernelRegOk,
            "Pi0.5 KV1 direct-cache rejected parameter ", index);
    };
    const auto set_reg = [&set_raw, base](uint32_t index, uint16_t value) {
        set_raw(base + index, value);
    };
    const auto set_pair = [&set_reg](uint32_t index, uint32_t value) {
        set_reg(index, static_cast<uint16_t>(value & 0xffffu));
        set_reg(index + 1, static_cast<uint16_t>(value >> 16));
    };
    set_pair(0, k_spm);
    set_pair(2, v_spm);
    // The typed DDR pairs remain zero until writer.write below.
    for (uint32_t index = 4; index <= 10; index += 2) set_pair(index, 0);
    set_pair(12, static_cast<uint32_t>(logical_rope_position));
    if (spm_rope) {
        // The pack's own cos/sin rows (its params 0..3; its T is kActiveRows),
        // which start at the chunk's first row, so the logical row is 0.
        const uint32_t cos_rows =
            SPM_ALLOC.addr(0, static_cast<uint32_t>(q_pack->rope_rows_off));
        set_pair(8, cos_rows);
        set_pair(10, cos_rows + kActiveRows * kRopeRowBytes);
        set_pair(12, 0);
    }
    set_pair(14, static_cast<uint32_t>(plan.segment(0).position));
    set_reg(16, kActiveRows);
    set_reg(17, kPhysicalRows);
    set_reg(18, kHeadDim);
    set_reg(19, kPhysicalKvHeads);
    set_reg(20, kCapacity);
    set_reg(21, kCompactWidth);
    set_reg(22, kPairWidth);
    set_reg(23, kAbiVersion);
    for (uint32_t index = 24; index < 64 - base; ++index) set_reg(index, 0);
    // Bit c: reader core c reads none of the written blocks (rotated writer only).
    set_reg(25, rotated_skip_mask);
    // Blocks below param28 run the cache body.
    if (fused) set_raw(28, kGrid);
    const uint16_t grid = kGrid + pack_blocks;
    set_raw(64, grid);
    set_raw(65, 1);
    set_raw(66, 1);
    set_raw(67, 0);
    writer.write(*kernel);

    auto* queue = GET_QUEUE(NUM_CORES);
    TORCH_CHECK(queue != nullptr,
                "Pi0.5 KV1 direct-cache 8-core queue is absent");
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(
        *kernel, {grid, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
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
                plan.logical_rows() == 336 || plan.logical_rows() == 400 ||
                plan.logical_rows() == 416 || plan.logical_rows() == 432 || plan.logical_rows() == 448 || plan.logical_rows() == 464 ||
                plan.logical_rows() == 480,
                "Pi0.5 prefill KV1 direct-cache requires C272/C288/C304/C320/C336/C400/C416/C432/C448/C464/C480");
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
    const KernelId kKernelId = kActiveRows == 480
        ? KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M480D256P480
        : kActiveRows == 432
        ? KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M432D256P432
        : kActiveRows == 464
        ? KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M464D256P464
        : kActiveRows == 336
        ? KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M336D256P336
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
        "Pi0.5 prefill KV1 direct-cache requires an exact P544/C272, P576/C288, P608/C304, P640/C320, P672/C336, P800/C400, P832/C416, P864/C432, P896/C448, P928/C464 or P960/C480 "
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

// Exact G05 W8 Action continuation. All kernels launch cores0..7: the LKN
// packet encodes a contiguous start/count, so {0,4} is not a sparse core mask.
namespace {
constexpr uint32_t kG05OwnerRows = 32;
constexpr uint32_t kG05OwnerDim = 256;
constexpr uint32_t kG05OwnerBytes = kG05OwnerRows * kG05OwnerDim * 2;

void g05_require_owner_cache_payloads() {
    for (auto id : {KernelId::G05_K_OWNER_RMS_NORM_M32D256,
                    KernelId::G05_K_OWNER_PARTIAL_MROPE_M32D256R64,
                    KernelId::G05_OWNER_INSERT_KCACHE_MULTIWARP,
                    KernelId::G05_OWNER_INSERT_KCACHE_MULTIWARP_V16,
                    KernelId::G05_OWNER_INSERT_VCACHE_MULTIWARP,
                    KernelId::G05_OWNER_INSERT_VCACHE_MULTIWARP_V16}) {
        TORCH_CHECK(KernelCache::instance().has_loaded(id),
                    "G05 owner continuation requires all six optional payloads");
    }
}

bool g05_owner_intervals_disjoint(uint64_t a, size_t an, uint64_t b, size_t bn) {
    return a + an <= b || b + bn <= a;
}

void g05_validate_owner_spm_absolute(uint32_t address, size_t bytes) {
    TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 owner requires initialized SPM");
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    TORCH_CHECK(bytes <= SpmAllocator::SPM_USABLE && address >= base &&
                    (address & 255u) == 0 &&
                    uint64_t{address} - base <= SpmAllocator::SPM_USABLE - bytes,
                "G05 owner SPM interval/alignment is invalid");
}

uint64_t g05_validate_owner_ddr(const at::Tensor& owner) {
    const uint64_t address = RpuGetDevAddr(owner.data_ptr());
    constexpr uint64_t limit = UINT64_C(1) << 40;
    TORCH_CHECK(address != 0 && (address & 255u) == 0 && address < limit &&
                    owner.nbytes() <= limit - address,
                "G05 owner DDR interval/alignment is invalid");
    return address;
}

void g05_set_owner_pair(Kernel_t& kernel, uint32_t index, uint32_t value) {
    kernel.set_regs(index, static_cast<uint16_t>(value));
    kernel.set_regs(index + 1, static_cast<uint16_t>(value >> 16));
}
}  // namespace

void rpu_launch_g05_k_owner_norm_rope_spm_kernel(
    uint32_t key_spm_addr, uint32_t norm_spm_addr,
    const at::Tensor& cos, const at::Tensor& sin, int64_t rope_position) {
    TORCH_CHECK(RpuKernelGraph::has_active(), "G05 owner norm/RoPE requires Graph");
    g05_validate_owner_spm_absolute(key_spm_addr, kG05OwnerBytes);
    g05_validate_owner_spm_absolute(norm_spm_addr, kG05OwnerDim * 2);
    TORCH_CHECK(g05_owner_intervals_disjoint(
                    key_spm_addr, kG05OwnerBytes, norm_spm_addr, kG05OwnerDim * 2),
                "G05 owner K and norm SPM intervals overlap");
    const auto table_ok = [rope_position](const at::Tensor& table) {
        return table.defined() &&
            table.device().type() == c10::DeviceType::PrivateUse1 &&
            table.scalar_type() == at::kHalf && table.is_contiguous() &&
            table.storage_offset() == 0 && table.dim() == 2 && table.size(1) == 32 &&
            rope_position >= 0 && rope_position <= UINT32_MAX &&
            rope_position <= table.size(0) &&
            kG05OwnerRows <= table.size(0) - rope_position;
    };
    TORCH_CHECK(table_ok(cos) && table_ok(sin) && cos.sizes() == sin.sizes() &&
                    cos.device() == sin.device(),
                "G05 owner RoPE needs FP16 [rows,32] owners covering all32 rows");
    const uint64_t cos_addr = g05_validate_owner_ddr(cos);
    const uint64_t sin_addr = g05_validate_owner_ddr(sin);
    TORCH_CHECK(g05_owner_intervals_disjoint(cos_addr, cos.nbytes(), sin_addr, sin.nbytes()),
                "G05 owner cos/sin DDR intervals overlap");
    g05_require_owner_cache_payloads();
    auto* queue = GET_QUEUE(8);
    TORCH_CHECK(queue != nullptr, "G05 owner eight-core queue is absent");
    queue->set_broadcast_mode(true);

    // The guard is followed by the complete BASE RMSNorm payload. It has no
    // DDR operands and uses the same intrinsic no-DDR census as BASE RMSNorm.
    auto* kernel = GET_KERNEL(KernelId::G05_K_OWNER_RMS_NORM_M32D256);
    TORCH_CHECK(kernel != nullptr, "G05 owner BASE norm payload disappeared");
    kernel->reset_regs();
    kernel->set_regs(0, uint16_t{256});
    kernel->set_regs(1, uint16_t{0x1c00});  // FP16(1/256), unchanged BASE ABI.
    kernel->set_regs(2, uint16_t{512});
    kernel->set_regs(3, uint16_t{0});
    g05_set_owner_pair(*kernel, 4, key_spm_addr);
    g05_set_owner_pair(*kernel, 6, key_spm_addr);
    g05_set_owner_pair(*kernel, 8, norm_spm_addr);
    kernel->set_regs(10, uint16_t{0x0011});  // FP16(1e-6), original epsilon.
    queue->enqueu_kernel(*kernel, {32, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});

    constexpr auto id = KernelId::G05_K_OWNER_PARTIAL_MROPE_M32D256R64;
    auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
        id, std::vector<GraphDdrRegisterOperandSpec>{
            {GraphDdrRegisterAbi{GraphDdrRegisterRole::SemanticInput,
                GraphDdrRegisterAccess::Read,
                GraphDdrRegisterEncoding::DevAddrShift8LoHi, 0, 1}, cos},
            {GraphDdrRegisterAbi{GraphDdrRegisterRole::SemanticInput,
                GraphDdrRegisterAccess::Read,
                GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, sin}});
    rpu_ddr_flush(cos.data_ptr<c10::Half>());
    rpu_ddr_flush(sin.data_ptr<c10::Half>());
    kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel != nullptr, "G05 owner partial RoPE payload disappeared");
    kernel->reset_regs();
    g05_set_owner_pair(*kernel, 4, key_spm_addr);
    g05_set_owner_pair(*kernel, 6, key_spm_addr);
    g05_set_owner_pair(*kernel, 8, static_cast<uint32_t>(rope_position));
    kernel->set_regs(10, uint16_t{256});
    kernel->set_regs(11, uint16_t{1});
    kernel->set_regs(12, uint16_t{32});
    kernel->set_regs(13, uint16_t{64});
    kernel->set_regs(14, uint16_t{1});
    writer.write(*kernel);
    queue->enqueu_kernel(*kernel, {1, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

// Exact G05 Action continuation. Each warp retires four independent BASE
// RMSNorm rows before applying the original partial-MRoPE to those same rows.
void rpu_launch_g05_qk_norm_rope_group4_spm_kernel(
    uint32_t query_spm_addr, uint32_t key_spm_addr,
    uint32_t query_norm_spm_addr, uint32_t key_norm_spm_addr,
    const at::Tensor& query_cos, const at::Tensor& query_sin, int64_t query_position,
    const at::Tensor& key_cos, const at::Tensor& key_sin, int64_t key_position) {
    TORCH_CHECK(RpuKernelGraph::has_active(), "G05 group4 requires Graph");
    const std::array<std::pair<uint32_t,size_t>,4> spans{{
        {query_spm_addr,kG05OwnerBytes},{key_spm_addr,kG05OwnerBytes},
        {query_norm_spm_addr,kG05OwnerDim*2},{key_norm_spm_addr,kG05OwnerDim*2}}};
    for (size_t i=0;i<spans.size();++i) {
        g05_validate_owner_spm_absolute(spans[i].first,spans[i].second);
        for (size_t j=0;j<i;++j)
            TORCH_CHECK(g05_owner_intervals_disjoint(spans[i].first,spans[i].second,
                            spans[j].first,spans[j].second),
                        "G05 group4 Q/K and gamma SPM intervals overlap");
    }
    const std::array<const at::Tensor*,2> cos{{&query_cos,&key_cos}},sin{{&query_sin,&key_sin}};
    const std::array<int64_t,2> positions{{query_position,key_position}};
    for (size_t family=0;family<2;++family) {
        const auto table_ok=[&](const at::Tensor& t) {
            const int64_t position=positions[family];
            return t.defined() && t.device().type()==c10::DeviceType::PrivateUse1 &&
                t.scalar_type()==at::kHalf && t.is_contiguous() && t.storage_offset()==0 &&
                t.dim()==2 && t.size(1)==32 && position>=0 && position<=UINT32_MAX &&
                position<=t.size(0) && 32<=t.size(0)-position;
        };
        const auto& c=*cos[family];const auto& s=*sin[family];
        TORCH_CHECK(table_ok(c) && table_ok(s) && c.sizes()==s.sizes() && c.device()==s.device(),
                    "G05 group4 requires owned FP16 [rows,32] tables covering each independent M32 position");
        const auto ca=g05_validate_owner_ddr(c),sa=g05_validate_owner_ddr(s);
        TORCH_CHECK(g05_owner_intervals_disjoint(ca,c.nbytes(),sa,s.nbytes()),
                    "G05 group4 cosine/sine DDR owners overlap");
    }
    constexpr std::array<KernelId,2> ids{{KernelId::G05_Q_NORM_PARTIAL_ROPE_M32D256R64_GROUP4,
        KernelId::G05_K_OWNER_NORM_PARTIAL_ROPE_M32D256R64_GROUP4}};
    for (auto id:ids) TORCH_CHECK(KernelCache::instance().has_loaded(id),
        "G05 group4 optional payload absent; cold owner must retain original route");
    auto* queue=GET_QUEUE(8);TORCH_CHECK(queue,"G05 group4 queue absent");
    queue->set_broadcast_mode(true);
    for (size_t family=0;family<2;++family) {
        rpu_ddr_flush(cos[family]->data_ptr<c10::Half>());rpu_ddr_flush(sin[family]->data_ptr<c10::Half>());
    }
    for (size_t family=0;family<2;++family) {
        const auto id=ids[family];
        auto writer=RpuKernelGraph::active().stage_kernel_ddr_registers(id,
            std::vector<GraphDdrRegisterOperandSpec>{
                {{GraphDdrRegisterRole::SemanticInput,GraphDdrRegisterAccess::Read,
                  GraphDdrRegisterEncoding::DevAddrShift8LoHi,16,17},*cos[family]},
                {{GraphDdrRegisterRole::SemanticInput,GraphDdrRegisterAccess::Read,
                  GraphDdrRegisterEncoding::DevAddrShift8LoHi,18,19},*sin[family]}});
        auto* kernel=GET_KERNEL(id);TORCH_CHECK(kernel,"G05 group4 payload disappeared");
        kernel->reset_regs();
        kernel->set_regs(0,uint16_t{256});kernel->set_regs(1,uint16_t{0x1c00});
        kernel->set_regs(2,uint16_t{512});kernel->set_regs(3,uint16_t{0});
        g05_set_owner_pair(*kernel,4,spans[family].first);
        g05_set_owner_pair(*kernel,6,spans[family].first);
        g05_set_owner_pair(*kernel,8,spans[family+2].first);
        kernel->set_regs(10,uint16_t{0x0011}); // Original FP16 epsilon1e-6.
        g05_set_owner_pair(*kernel,20,static_cast<uint32_t>(positions[family]));
        kernel->set_regs(22,uint16_t{1});kernel->set_regs(23,uint16_t{4});
        kernel->set_regs(24,uint16_t{64});kernel->set_regs(25,uint16_t{1});
        writer.write(*kernel);
        queue->enqueu_kernel(*kernel,{8,1,1},{0,1,2,3,4,5,6,7});
    }
}

void rpu_launch_g05_owner_insert_kvcache_spm_with_plan(
    at::Tensor& key_cache, at::Tensor& value_cache,
    uint32_t key_spm_offset, uint32_t value_spm_offset,
    const KvInsertSegmentPlan& plan) {
    TORCH_CHECK(RpuKernelGraph::has_active(), "G05 owner cache requires Graph");
    rpu_validate_kvinsert_segment_plan(plan, 8, 8, 256);
    const auto part_is = [&](size_t index, KvInsertKernel kernel,
                             int64_t position, int64_t offset, int64_t rows) {
        const auto& part = plan.segment(index);
        return part.kernel == kernel && part.position == position &&
            part.token_offset == offset && part.rows == rows;
    };
    const bool p0 = plan.route() == KvInsertRoute::ALIGNED_V16 &&
        plan.segment_count() == 1 && part_is(0, KvInsertKernel::V16, 0, 0, 32);
    const bool p245 = plan.route() == KvInsertRoute::HYBRID3 &&
        plan.segment_count() == 3 && part_is(0, KvInsertKernel::V2, 245, 0, 11) &&
        part_is(1, KvInsertKernel::V16, 256, 11, 16) &&
        part_is(2, KvInsertKernel::V2, 272, 27, 5);
    const int64_t prefix = plan.segment(0).position;
    const bool p1 = prefix >= 161 && prefix <= 449 && prefix % 16 == 1 &&
        plan.route() == KvInsertRoute::HYBRID3 && plan.segment_count() == 3 &&
        part_is(0,KvInsertKernel::V2,prefix,0,15) &&
        part_is(1,KvInsertKernel::V16,prefix+15,15,16) &&
        part_is(2,KvInsertKernel::V2,prefix+31,31,1);
    const bool p181 = plan.route() == KvInsertRoute::HYBRID3 &&
        plan.segment_count() == 3 && part_is(0,KvInsertKernel::V2,181,0,11) &&
        part_is(1,KvInsertKernel::V16,192,11,16) &&
        part_is(2,KvInsertKernel::V2,208,27,5);
    TORCH_CHECK(plan.logical_rows() == 32 && plan.physical_rows() == 32 &&
                    (p0 || p245 || p1 || p181),
                "G05 owner cache requires exact unpadded bounded P0/P181/P245/P%16=1 M32 segment plan");
    TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 owner cache requires initialized SPM");
    for (auto offset : {key_spm_offset, value_spm_offset}) {
        TORCH_CHECK((offset & 255u) == 0 &&
                        offset <= SpmAllocator::SPM_USABLE - kG05OwnerBytes,
                    "G05 owner cache source offset exceeds full32-row SPM extent");
        const uint64_t absolute = SPM_ALLOC.addr(0, offset);
        TORCH_CHECK(absolute == uint64_t{SPM_ALLOC.addr(0, 0)} + offset && absolute <= UINT32_MAX,
                    "G05 owner cache typed offset conversion overflow");
    }
    TORCH_CHECK(g05_owner_intervals_disjoint(
                    key_spm_offset, kG05OwnerBytes, value_spm_offset, kG05OwnerBytes),
                "G05 owner K/V SPM intervals overlap");
    const int64_t end_row = plan.segment(0).position + 32;
    const auto cache_ok = [end_row](const at::Tensor& cache) {
        // Capacity is an actual owner bound, not part of the fixed inner ABI.
        // Hardware performance evidence is S136; writers are capacity-independent.
        return cache.defined() &&
            cache.device().type() == c10::DeviceType::PrivateUse1 &&
            cache.scalar_type() == at::kHalf && cache.is_contiguous() &&
            cache.storage_offset() == 0 && cache.dim() == 7 &&
            cache.size(0) == 1 && cache.size(1) >= (end_row + 15) / 16 &&
            cache.size(2) == 1 && cache.size(3) == 16 && cache.size(4) == 8 &&
            cache.size(5) == 16 && cache.size(6) == 16 &&
            cache.nbytes() == static_cast<size_t>(cache.size(1)) * 65536;
    };
    TORCH_CHECK(cache_ok(key_cache) && cache_ok(value_cache) &&
                    key_cache.sizes() == value_cache.sizes() &&
                    key_cache.device() == value_cache.device(),
                "G05 owner cache needs full FP16 [1,S,1,16,8,16,16] owners covering the plan");
    const uint64_t key_addr = g05_validate_owner_ddr(key_cache);
    const uint64_t value_addr = g05_validate_owner_ddr(value_cache);
    TORCH_CHECK(g05_owner_intervals_disjoint(
                    key_addr, key_cache.nbytes(), value_addr, value_cache.nbytes()),
                "G05 owner cache DDR intervals overlap");
    g05_require_owner_cache_payloads();
    auto* queue = GET_QUEUE(8);
    TORCH_CHECK(queue != nullptr, "G05 owner cache eight-core queue is absent");
    queue->set_broadcast_mode(true);
    // Original order: all K segments, then all V segments. Every node binds its
    // actual cache owner. Source GP6/7 remains an offset for unified owner loads.
    for (int kind = 0; kind < 2; ++kind) {
        const auto& cache = kind == 0 ? key_cache : value_cache;
        for (size_t i = 0; i < plan.segment_count(); ++i) {
            const auto& part = plan.segment(i);
            const bool v16 = part.kernel == KvInsertKernel::V16;
            const auto id = kind == 0
                ? (v16 ? KernelId::G05_OWNER_INSERT_KCACHE_MULTIWARP_V16
                       : KernelId::G05_OWNER_INSERT_KCACHE_MULTIWARP)
                : (v16 ? KernelId::G05_OWNER_INSERT_VCACHE_MULTIWARP_V16
                       : KernelId::G05_OWNER_INSERT_VCACHE_MULTIWARP);
            auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
                id, std::vector<GraphDdrRegisterOperandSpec>{
                    {GraphDdrRegisterAbi{
                        kind == 0 ? GraphDdrRegisterRole::KeyCache : GraphDdrRegisterRole::ValueCache,
                        GraphDdrRegisterAccess::Write,
                        GraphDdrRegisterEncoding::DevAddrShift8LoHi, 8, 9}, cache}});
            rpu_ddr_flush(cache.data_ptr<c10::Half>());
            auto* kernel = GET_KERNEL(id);
            TORCH_CHECK(kernel != nullptr, "G05 owner cache writer payload disappeared");
            kernel->reset_regs();
            g05_set_owner_pair(*kernel, 0, static_cast<uint32_t>(part.position));
            kernel->set_regs(2, uint16_t{1});
            kernel->set_regs(3, uint16_t{256});
            kernel->set_regs(4, static_cast<uint16_t>(part.rows));
            const uint32_t offset = (kind == 0 ? key_spm_offset : value_spm_offset) +
                static_cast<uint32_t>(part.token_offset * 512);
            g05_set_owner_pair(*kernel, 6, offset);
            kernel->set_regs(21, uint16_t{8});
            writer.write(*kernel);
            queue->enqueu_kernel(*kernel,
                {static_cast<uint16_t>(v16 ? part.rows / 16 : part.rows), 1, 1},
                {0, 1, 2, 3, 4, 5, 6, 7});
        }
    }
}

// One exact standard KV22 plan, all eight physical cache stripes. The private
// measured composition preserves every original writer's transpose/locks/fences.
void rpu_launch_g05_owner_insert_kvcache_joint_spm_with_plan(
    at::Tensor& key_cache, at::Tensor& value_cache,
    uint32_t key_spm_offset, uint32_t value_spm_offset,
    const KvInsertSegmentPlan& plan) {
    TORCH_CHECK(RpuKernelGraph::has_active(), "G05 joint owner cache requires Graph");
    rpu_validate_kvinsert_segment_plan(plan, 8, 8, 256);
    const auto part_is = [&](size_t index, KvInsertKernel kernel,
                             int64_t position, int64_t offset, int64_t rows) {
        const auto& part = plan.segment(index);
        return part.kernel == kernel && part.position == position &&
            part.token_offset == offset && part.rows == rows;
    };
    const bool p0 = plan.route() == KvInsertRoute::ALIGNED_V16 &&
        plan.segment_count() == 1 && part_is(0, KvInsertKernel::V16, 0, 0, 32);
    const bool p245 = plan.route() == KvInsertRoute::HYBRID3 &&
        plan.segment_count() == 3 && part_is(0, KvInsertKernel::V2, 245, 0, 11) &&
        part_is(1, KvInsertKernel::V16, 256, 11, 16) &&
        part_is(2, KvInsertKernel::V2, 272, 27, 5);
    TORCH_CHECK(plan.logical_rows() == 32 && plan.physical_rows() == 32 && (p0 || p245),
                "G05 joint owner cache requires exact unpadded P0/P245 M32 segment plan");
    TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 joint owner cache requires initialized SPM");
    for (auto offset : {key_spm_offset, value_spm_offset}) {
        TORCH_CHECK((offset & 255u) == 0 &&
                        offset <= SpmAllocator::SPM_USABLE - kG05OwnerBytes,
                    "G05 joint owner cache source offset exceeds full32-row SPM extent");
        const uint64_t absolute = SPM_ALLOC.addr(0, offset);
        TORCH_CHECK(absolute == uint64_t{SPM_ALLOC.addr(0, 0)} + offset && absolute <= UINT32_MAX,
                    "G05 joint owner cache typed offset conversion overflow");
    }
    TORCH_CHECK(g05_owner_intervals_disjoint(
                    key_spm_offset, kG05OwnerBytes, value_spm_offset, kG05OwnerBytes),
                "G05 owner K/V SPM intervals overlap");
    const int64_t end_row = plan.segment(0).position + 32;
    const auto cache_ok = [end_row](const at::Tensor& cache) {
        // Capacity is an actual owner bound, not part of the fixed inner ABI.
        // Hardware performance evidence is S136; writers are capacity-independent.
        return cache.defined() &&
            cache.device().type() == c10::DeviceType::PrivateUse1 &&
            cache.scalar_type() == at::kHalf && cache.is_contiguous() &&
            cache.storage_offset() == 0 && cache.dim() == 7 &&
            cache.size(0) == 1 && cache.size(1) >= (end_row + 15) / 16 &&
            cache.size(2) == 1 && cache.size(3) == 16 && cache.size(4) == 8 &&
            cache.size(5) == 16 && cache.size(6) == 16 &&
            cache.nbytes() == static_cast<size_t>(cache.size(1)) * 65536;
    };
    TORCH_CHECK(cache_ok(key_cache) && cache_ok(value_cache) &&
                    key_cache.sizes() == value_cache.sizes() &&
                    key_cache.device() == value_cache.device(),
                "G05 joint owner cache needs full FP16 [1,S,1,16,8,16,16] owners covering the plan");
    const uint64_t key_addr = g05_validate_owner_ddr(key_cache);
    const uint64_t value_addr = g05_validate_owner_ddr(value_cache);
    TORCH_CHECK(g05_owner_intervals_disjoint(
                    key_addr, key_cache.nbytes(), value_addr, value_cache.nbytes()),
                "G05 joint owner cache DDR intervals overlap");

    const auto id = p0 ? KernelId::G05_OWNER_INSERT_KV_CACHE_M32D256_P0
                       : KernelId::G05_OWNER_INSERT_KV_CACHE_M32D256_P245;
    TORCH_CHECK(KernelCache::instance().has_loaded(id),
                "G05 joint owner cache payload absent; cold owner must retain separate writers");
    auto* queue = GET_QUEUE(8);
    TORCH_CHECK(queue != nullptr, "G05 joint owner cache eight-core queue is absent");
    queue->set_broadcast_mode(true);
    auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
        id, std::vector<GraphDdrRegisterOperandSpec>{
            {{GraphDdrRegisterRole::KeyCache, GraphDdrRegisterAccess::Write,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 8, 9}, key_cache},
            {{GraphDdrRegisterRole::ValueCache, GraphDdrRegisterAccess::Write,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 12, 13}, value_cache}});
    rpu_ddr_flush(key_cache.data_ptr<c10::Half>());
    rpu_ddr_flush(value_cache.data_ptr<c10::Half>());
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel != nullptr, "G05 joint owner cache payload disappeared");
    kernel->reset_regs();
    g05_set_owner_pair(*kernel, 0, static_cast<uint32_t>(plan.segment(0).position));
    kernel->set_regs(2, uint16_t{1});
    kernel->set_regs(3, uint16_t{256});
    kernel->set_regs(4, uint16_t{32});
    g05_set_owner_pair(*kernel, 6, key_spm_offset);
    g05_set_owner_pair(*kernel, 10, value_spm_offset);
    kernel->set_regs(21, uint16_t{8});
    writer.write(*kernel);
    queue->enqueu_kernel(*kernel,
        {static_cast<uint16_t>(p0 ? 2 : 17), 2, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}
