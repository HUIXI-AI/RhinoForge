// rpu_llama_kvcache.cpp
// LLaMA KV-Cache insert kernel launch functions

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include "rhino_launch_buffer.h"
#include "rhino_launch_queue.h"
#include <c10/util/Half.h>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <string>
#include <iostream>

using namespace rhino_lkn;

#define NUM_CORES 8

// KV-insert v16 is selected for aligned prefill with whole-head partitions.
// RPU_KVINSERT_V16=0 disables it.
static bool kvinsert_v16_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_KVINSERT_V16");
        cached = (e && (std::string(e) == "0" || std::string(e) == "false")) ? 0 : 1;
    }
    return cached != 0;
}
// RPU_KVINSERT_V16_ANY_TP admits whole-head partitions below eight cores.
// Default off.
static bool kvinsert_v16_any_tp_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_KVINSERT_V16_ANY_TP");
        cached = (e && std::string(e) != "" && std::string(e) != "0" &&
                  std::string(e) != "false" && std::string(e) != "False") ? 1 : 0;
    }
    return cached != 0;
}

// Pure shape conditions with no environment read of their own. One copy backs
// both the whole-sequence decision and the hybrid bulk decision. `allow_any_tp`
// is threaded in by the callers (they call
// kvinsert_v16_any_tp_enabled()), so this predicate stays a pure function of its
// arguments and the exposed `rpu_kvinsert_shape_plan` signature is unchanged.
static inline bool kvinsert_v16_shape_ok(int64_t position, int64_t seq_len, int num_cores,
                                         int64_t num_kv_heads, int64_t head_dim,
                                         bool allow_non8_v16 = false) {
    return (position % 16 == 0) && (seq_len % 16 == 0)
        && (seq_len >= 16)
        && (num_cores == 8 || allow_non8_v16)
        && (num_kv_heads % num_cores == 0)
        && (head_dim % 16 == 0);
}

// Declared in rpu_kernel_decls.h and shared by the K and V launch paths.
KvInsertShapePlan rpu_kvinsert_shape_plan(int64_t position, int64_t seq_len,
                                          int num_cores, int64_t num_kv_heads,
                                          int64_t head_dim) {
    KvInsertShapePlan p;
    p.bulk_seq = (seq_len / 16) * 16;
    p.tail_seq = seq_len - p.bulk_seq;
    const bool any_tp = kvinsert_v16_any_tp_enabled() && num_cores > 0
        && NUM_CORES % num_cores == 0;
    p.v16_shape_ok =
        kvinsert_v16_shape_ok(position, seq_len, num_cores, num_kv_heads, head_dim, any_tp);
    p.hybrid_shape_ok =
        p.tail_seq > 0 && p.bulk_seq >= 16 &&
        kvinsert_v16_shape_ok(position, p.bulk_seq, num_cores, num_kv_heads, head_dim, any_tp);
    return p;
}

static inline bool kvinsert_use_v16(int64_t position, int64_t seq_len, int num_cores,
                                    int64_t num_kv_heads, int64_t head_dim,
                                    bool allow_non8_v16 = false) {
    const bool any_tp = kvinsert_v16_any_tp_enabled() && num_cores > 0
        && NUM_CORES % num_cores == 0;
    return kvinsert_v16_enabled() && kvinsert_v16_shape_ok(
        position, seq_len, num_cores, num_kv_heads, head_dim,
        allow_non8_v16 || any_tp);
}
static bool kvinsert_v16_trace_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_KVINSERT_V16_TRACE");
        cached = (e && std::string(e) != "0" && std::string(e) != "false") ? 1 : 0;
    }
    return cached != 0;
}
// Hybrid KV insert splits a non-16-aligned prefill into an aligned v16 bulk and
// a v2 tail. It is opt-in; other profiles retain their ordinary route.
static bool kvinsert_hybrid_v16_enabled() {
    const char* e = std::getenv("RPU_KVINSERT_HYBRID_V16");
    const char* rhino = std::getenv("RPU_RHINOVLA_KVINSERT_HYBRID_V16");
    auto on = [](const char* v) {
        return v && std::string(v) != "" && std::string(v) != "0" &&
               std::string(v) != "false" && std::string(v) != "False";
    };
    return on(e) || on(rhino);
}

// Three-way KV insert adds a v2 head when position is not 16-aligned, followed
// by an aligned v16 bulk and a v2 tail. It is opt-in.
static bool kvinsert_hybrid3_v16_enabled() {
    const char* e = std::getenv("RPU_KVINSERT_HYBRID3_V16");
    return e && std::string(e) != "" && std::string(e) != "0" &&
           std::string(e) != "false" && std::string(e) != "False";
}

// =============================================================================
// LLaMA V-Cache Insert Kernel
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
    // Host launch register contract:
    // param0/1: position (int32 value, 直接是数值)
    // param2: num_kv_percore_heads (每核的 head 数)
    // param3: head_dim
    // param4: seq_len
    // param6/7: input V SPM address (64-bit)
    // param8/9: V-Cache DDR address (64-bit)

    // 使用 Core 0 的 SPM 地址作为基准 (broadcast mode 下所有核使用相同的相对偏移)
    uint32_t v_spm_addr = spm_v[0]->get_rpu_addr();
    uint32_t position_val = static_cast<uint32_t>(position);

    // Operator data_size = param2 * param3.
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
// LLaMA K-Cache Insert Kernel
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

    // Operator data_size = param2 * param3.
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

static void validate_kvinsert_spm_rows(int64_t seq_len, int64_t spm_rows) {
    TORCH_CHECK(spm_rows == 0 || spm_rows >= seq_len,
                "KV-insert spm_rows must cover seq_len; got spm_rows=",
                spm_rows, " seq_len=", seq_len);
}

void rpu_launch_insert_kcache_spm_unified(
    at::Tensor &k_cache,         // DDR K-cache (swizzle format)
    int64_t position,
    uint32_t k_off,              // SPM offset for K buffer
    int64_t seq_len, int64_t num_kv_heads, int64_t head_dim,
    int num_cores,               // number of cores (attn_tp)
    int64_t cache_batch_offset_elems,
    bool allow_non8_v16,
    bool allow_hybrid_v16,
    int64_t spm_rows)
{
    validate_kvinsert_spm_rows(seq_len, spm_rows);
    int64_t total_kv_heads = num_kv_heads;
    // Per-core KV dim: when nhkv >= num_cores, each core has nhkv/num_cores complete heads;
    // when nhkv < num_cores, each core has a portion of head_dim (col partition).
    int64_t local_kv_dim = num_kv_heads * head_dim / num_cores;
    int64_t num_kv_percore_heads = (num_kv_heads >= num_cores) ? num_kv_heads / num_cores : 1;

    // Validate cache dimensions
    constexpr int64_t sKeyChunk = 16;
    constexpr int64_t headDimChunk = 16;

    // PAD16（见 rpu_kernel_decls.h 的声明注释）：调用方声明了 SPM 行容量且够
    // ceil16(seq_len) 时，把插入长度补齐到 16 的倍数 ⇒ "v16 主体 + v2 尾巴"
    // 两次发射塌成**一次** v16。补出来的那几行是 SPM 里的既有字节，写到
    // cache 的 [position+seq_len, position+pad) —— 那一段本来就是陈旧数据，
    // 一直靠 SDPA 的 kv_seq_len 挡在外面 ⇒ 输出逐位不变。
    const int64_t pad_seq_ = ((seq_len + 15) / 16) * 16;
    const int64_t eff_seq  = (spm_rows >= pad_seq_ && pad_seq_ != seq_len
                              && position % 16 == 0) ? pad_seq_ : seq_len;
    int64_t sKeyVx_needed = CeilDiv(position + eff_seq, sKeyChunk);  // Need space up to position+eff_seq
    int64_t headDimVx = head_dim / headDimChunk;
    // Cache layout: effective_kv_slots = NUM_CORES * nhkv / attn_tp.
    int64_t effective_kv_slots = (int64_t)NUM_CORES * total_kv_heads / num_cores;
    int64_t nKVHeadChunk = std::min((int64_t)NUM_CORES, effective_kv_slots);
    int64_t nKVHeadVx = CeilDiv(effective_kv_slots, nKVHeadChunk);

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
    // Operator data_size = param2 * param3 = elements per core per position.
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

    auto launch_once = [&](int64_t part_position,
                           uint32_t part_spm_addr,
                           int64_t part_seq_len,
                           bool use_v16,
                           const char* path_tag) {
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
        setup_regs(kernel, part_position, part_seq_len, part_spm_addr);
        register_writer.write(*kernel);

        if (kvinsert_v16_trace_enabled()) {
            std::cerr << "[KVINSERT_TRACE] kind=K path=" << path_tag
                      << " pos=" << part_position << " seq=" << part_seq_len
                      << " nkv=" << num_kv_heads << " hdim=" << kern_hdim
                      << " grid_x=" << (use_v16 ? (part_seq_len / 16) : part_seq_len)
                      << "\n";
        }

        const uint16_t grid_x = use_v16 ? (uint16_t)(part_seq_len / 16) : (uint16_t)part_seq_len;
        auto* wq = GET_QUEUE(num_cores);
        wq->set_broadcast_mode(true);
        std::vector<uint8_t> core_list;
        for (int i = 0; i < num_cores; ++i) core_list.push_back(i);
        wq->enqueu_kernel(*kernel, {grid_x, (uint16_t)1, (uint16_t)1}, core_list);
    };

    // ⚠️ 传 `eff_seq` 而不是 `seq_len`：eff_seq 是本次**真正会发射**的长度（spm_rows
    // padding 生效时才 > seq_len）。spm_rows 默认 0 ⇒ 其余全部调用方 eff_seq == seq_len，
    // 计划与发射序列逐字节不变。
    const KvInsertShapePlan plan =
        rpu_kvinsert_shape_plan(position, eff_seq, num_cores, num_kv_heads, kern_hdim);
    const int64_t bulk_seq = plan.bulk_seq;
    const int64_t tail_seq = plan.tail_seq;
    const bool hybrid_v16 =
        (kvinsert_hybrid_v16_enabled() || allow_hybrid_v16) &&
        kvinsert_v16_enabled() && tail_seq > 0 && bulk_seq >= 16 &&
        kvinsert_v16_shape_ok(position, bulk_seq, num_cores, num_kv_heads,
                              kern_hdim, allow_non8_v16);
    // hybrid3: position 非 16 对齐时，用一段短 v2 head 把 bulk 顶到 16 边界上。
    const int64_t head3 = (16 - (position % 16)) % 16;
    const int64_t rest3 = seq_len - head3;
    const int64_t bulk3 = rest3 > 0 ? (rest3 / 16) * 16 : 0;
    const int64_t tail3 = rest3 > 0 ? rest3 - bulk3 : 0;
    const bool hybrid3_v16 =
        kvinsert_hybrid3_v16_enabled() &&
        head3 > 0 && rest3 > 0 && bulk3 >= 16 &&
        kvinsert_use_v16(position + head3, bulk3, num_cores, num_kv_heads,
                         kern_hdim, allow_non8_v16);
    const bool use_v16 =
        kvinsert_use_v16(position, eff_seq, num_cores, num_kv_heads,
                         kern_hdim, allow_non8_v16);
    if (RpuKernelGraph::active().kernel_register_census_active()) {
        TORCH_CHECK(
            !hybrid_v16 && !hybrid3_v16,
            "typed DDR-register census requires a single-launch K-cache route");
    }
    // 每 token 的 SPM 步长与两段 hybrid 的 tail 偏移算法一致。
    auto seg_addr3 = [&](int64_t tok_off) -> uint32_t {
        return k_spm_addr + static_cast<uint32_t>(
            tok_off * kern_heads * kern_hdim * static_cast<int64_t>(sizeof(c10::Half)));
    };
    if (hybrid3_v16) {
        launch_once(position, seg_addr3(0), head3, false, "v2_hybrid3_head");
        launch_once(position + head3, seg_addr3(head3), bulk3, true, "v16_hybrid3_bulk");
        if (tail3 > 0)
            launch_once(position + head3 + bulk3, seg_addr3(head3 + bulk3), tail3,
                        false, "v2_hybrid3_tail");
    } else if (hybrid_v16) {
        launch_once(position, k_spm_addr, bulk_seq, true, "v16_hybrid_bulk");
        const uint32_t tail_spm_addr = k_spm_addr + static_cast<uint32_t>(
            bulk_seq * kern_heads * kern_hdim * static_cast<int64_t>(sizeof(c10::Half)));
        launch_once(position + bulk_seq, tail_spm_addr, tail_seq, false, "v2_hybrid_tail");
    } else {
        launch_once(position, k_spm_addr, eff_seq, use_v16,
                    use_v16 ? (eff_seq != seq_len ? "v16_pad16" : "v16") : "v2");
    }

      // ===== 同步 =====
      rpu_ddr_flush(k_cache_operand.data_ptr<c10::Half>());

    // 注意: 不清理 SPM - 由调用方 (rpu_fused_qkv_attention_unified) 负责
}

// =============================================================================
// SPM Unified Layout V-Cache Insert Kernel
// =============================================================================
// Input is in SPM at unified layout position (V buffer)
// No DDR -> SPM copy needed - data is already in SPM from Linear kernel
// Output: V-Cache in DDR (swizzle format)
// =============================================================================

void rpu_launch_insert_vcache_spm_unified(
    at::Tensor &v_cache,         // DDR V-cache (swizzle format)
    int64_t position,
    uint32_t v_off,              // SPM offset for V buffer
    int64_t seq_len, int64_t num_kv_heads, int64_t head_dim,
    int num_cores,               // number of cores (attn_tp)
    int64_t cache_batch_offset_elems,
    bool allow_non8_v16,
    bool allow_hybrid_v16,
    int64_t spm_rows)
{
    validate_kvinsert_spm_rows(seq_len, spm_rows);
    int64_t total_kv_heads = num_kv_heads;
    int64_t local_kv_dim = num_kv_heads * head_dim / num_cores;
    int64_t num_kv_percore_heads = (num_kv_heads >= num_cores) ? num_kv_heads / num_cores : 1;

    // Validate cache dimensions
    constexpr int64_t sValChunk = 16;
    constexpr int64_t headDimChunk = 16;

    // PAD16（见 rpu_kernel_decls.h 的声明注释）：调用方声明了 SPM 行容量且够
    // ceil16(seq_len) 时，把插入长度补齐到 16 的倍数 ⇒ "v16 主体 + v2 尾巴"
    // 两次发射塌成**一次** v16。补出来的那几行是 SPM 里的既有字节，写到
    // cache 的 [position+seq_len, position+pad) —— 那一段本来就是陈旧数据，
    // 一直靠 SDPA 的 kv_seq_len 挡在外面 ⇒ 输出逐位不变。
    const int64_t pad_seq_ = ((seq_len + 15) / 16) * 16;
    const int64_t eff_seq  = (spm_rows >= pad_seq_ && pad_seq_ != seq_len
                              && position % 16 == 0) ? pad_seq_ : seq_len;
    int64_t sValVx_needed = CeilDiv(position + eff_seq, sValChunk);  // Need space up to position+eff_seq
    int64_t headDimVx = head_dim / headDimChunk;
    // Cache layout: effective_kv_slots = NUM_CORES * nhkv / attn_tp.
    int64_t effective_kv_slots = (int64_t)NUM_CORES * total_kv_heads / num_cores;
    int64_t nKVHeadChunk = std::min((int64_t)NUM_CORES, effective_kv_slots);
    int64_t nKVHeadVx = CeilDiv(effective_kv_slots, nKVHeadChunk);

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

    // Operator data_size = param2 * param3.
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

    auto launch_once = [&](int64_t part_position,
                           uint32_t part_spm_addr,
                           int64_t part_seq_len,
                           bool use_v16,
                           const char* path_tag) {
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
        setup_regs(kernel, part_position, part_seq_len, part_spm_addr);
        register_writer.write(*kernel);

        if (kvinsert_v16_trace_enabled()) {
            std::cerr << "[KVINSERT_TRACE] kind=V path=" << path_tag
                      << " pos=" << part_position << " seq=" << part_seq_len
                      << " nkv=" << num_kv_heads << " hdim=" << kern_hdim
                      << " grid_x=" << (use_v16 ? (part_seq_len / 16) : part_seq_len)
                      << "\n";
        }

        const uint16_t grid_x = use_v16 ? (uint16_t)(part_seq_len / 16) : (uint16_t)part_seq_len;
        auto* wq = GET_QUEUE(num_cores);
        wq->set_broadcast_mode(true);
        std::vector<uint8_t> core_list;
        for (int i = 0; i < num_cores; ++i) core_list.push_back(i);
        wq->enqueu_kernel(*kernel, {grid_x, (uint16_t)1, (uint16_t)1}, core_list);
    };

    // ⚠️ 传 `eff_seq` 而不是 `seq_len`：eff_seq 是本次**真正会发射**的长度（spm_rows
    // padding 生效时才 > seq_len）。spm_rows 默认 0 ⇒ 其余全部调用方 eff_seq == seq_len，
    // 计划与发射序列逐字节不变。
    const KvInsertShapePlan plan =
        rpu_kvinsert_shape_plan(position, eff_seq, num_cores, num_kv_heads, kern_hdim);
    const int64_t bulk_seq = plan.bulk_seq;
    const int64_t tail_seq = plan.tail_seq;
    const bool hybrid_v16 =
        (kvinsert_hybrid_v16_enabled() || allow_hybrid_v16) &&
        kvinsert_v16_enabled() && tail_seq > 0 && bulk_seq >= 16 &&
        kvinsert_v16_shape_ok(position, bulk_seq, num_cores, num_kv_heads,
                              kern_hdim, allow_non8_v16);
    // hybrid3: position 非 16 对齐时，用一段短 v2 head 把 bulk 顶到 16 边界上。
    const int64_t head3 = (16 - (position % 16)) % 16;
    const int64_t rest3 = seq_len - head3;
    const int64_t bulk3 = rest3 > 0 ? (rest3 / 16) * 16 : 0;
    const int64_t tail3 = rest3 > 0 ? rest3 - bulk3 : 0;
    const bool hybrid3_v16 =
        kvinsert_hybrid3_v16_enabled() &&
        head3 > 0 && rest3 > 0 && bulk3 >= 16 &&
        kvinsert_use_v16(position + head3, bulk3, num_cores, num_kv_heads,
                         kern_hdim, allow_non8_v16);
    const bool use_v16 =
        kvinsert_use_v16(position, eff_seq, num_cores, num_kv_heads,
                         kern_hdim, allow_non8_v16);
    if (RpuKernelGraph::active().kernel_register_census_active()) {
        TORCH_CHECK(
            !hybrid_v16 && !hybrid3_v16,
            "typed DDR-register census requires a single-launch V-cache route");
    }
    // 每 token 的 SPM 步长与两段 hybrid 的 tail 偏移算法一致。
    auto seg_addr3 = [&](int64_t tok_off) -> uint32_t {
        return v_spm_addr + static_cast<uint32_t>(
            tok_off * kern_heads * kern_hdim * static_cast<int64_t>(sizeof(c10::Half)));
    };
    if (hybrid3_v16) {
        launch_once(position, seg_addr3(0), head3, false, "v2_hybrid3_head");
        launch_once(position + head3, seg_addr3(head3), bulk3, true, "v16_hybrid3_bulk");
        if (tail3 > 0)
            launch_once(position + head3 + bulk3, seg_addr3(head3 + bulk3), tail3,
                        false, "v2_hybrid3_tail");
    } else if (hybrid_v16) {
        launch_once(position, v_spm_addr, bulk_seq, true, "v16_hybrid_bulk");
        const uint32_t tail_spm_addr = v_spm_addr + static_cast<uint32_t>(
            bulk_seq * kern_heads * kern_hdim * static_cast<int64_t>(sizeof(c10::Half)));
        launch_once(position + bulk_seq, tail_spm_addr, tail_seq, false, "v2_hybrid_tail");
    } else {
        launch_once(position, v_spm_addr, eff_seq, use_v16,
                    use_v16 ? (eff_seq != seq_len ? "v16_pad16" : "v16") : "v2");
    }

      // ===== 同步 =====
      rpu_ddr_flush(v_cache_operand.data_ptr<c10::Half>());

    // 注意: 不清理 SPM - 由调用方 (rpu_fused_qkv_attention_unified) 负责
}

// =============================================================================
// Arena-staged KV insert (RhinoVLA): stage external DDR K/V through the repo
// SPM_ALLOC arena, then reuse the unified swizzle insert. Caller resets the
// temporary arena after the per-layer prefix loop (so a per-layer prefill loop
// accumulates distinct offsets, one reset at end — avoids a kernel-vs-reset
// race on a reused offset).
// =============================================================================

static uint32_t stage_kv_to_arena(const at::Tensor& kv_input, int num_cores,
                                  int64_t& seq_len_out, int64_t& total_kv_heads_out,
                                  int64_t& head_dim_out) {
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
    TORCH_CHECK(kv_input.size(2) % num_cores == 0 || num_cores % kv_input.size(2) == 0,
                "arena insert: num_kv_heads (", kv_input.size(2),
                ") must divide or be divisible by ", num_cores);

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

void rpu_launch_llama_insert_kcache_arena(
    const at::Tensor &k_input, at::Tensor &k_cache, int64_t position) {
    int64_t seq_len, nkv, hd;
    const uint32_t off = stage_kv_to_arena(k_input, NUM_CORES, seq_len, nkv, hd);
    rpu_launch_insert_kcache_spm_unified(k_cache, position, off, seq_len, nkv, hd, NUM_CORES);
}

void rpu_launch_llama_insert_vcache_arena(
    const at::Tensor &v_input, at::Tensor &v_cache, int64_t position) {
    int64_t seq_len, nkv, hd;
    const uint32_t off = stage_kv_to_arena(v_input, NUM_CORES, seq_len, nkv, hd);
    rpu_launch_insert_vcache_spm_unified(v_cache, position, off, seq_len, nkv, hd, NUM_CORES);
}
