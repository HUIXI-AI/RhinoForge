#pragma once
// =============================================================================
// DecoderSpmBuffers — offset-based 基类，使用 SpmAllocator
//
// 所有 buffer 存为 uint32_t offset（所有 core 共用）。
// 通过 SPM_ALLOC.addr(core, offset) 获取硬件地址。
// =============================================================================

#include "rpu_profile.h"   // Align, CeilDiv
#include "rpu_helpers.h"   // sdpa_compute_tiling, SdpaConfig, SdpaTiling
#include "rpu_spm_allocator.h"

#ifndef RPU_NUM_CORES
#define RPU_NUM_CORES 8
#endif
#ifndef RPU_DWIDTH
#define RPU_DWIDTH 2
#endif

// =============================================================================
// spm:: 工具函数
// =============================================================================

namespace spm {

constexpr size_t ALIGN = 256;

inline size_t buf_size(int64_t elements) {
    return Align(elements * RPU_DWIDTH, ALIGN);
}

inline size_t sdpa_tmp_bytes(int64_t seq_len, int64_t head_dim,
                             int64_t num_q_heads, int64_t num_kv_heads,
                             int attn_tp) {
    // Use full sdpa_compute_tiling so we get the post-VLM-shrink tile
    // shape (matters when head_dim is large enough to trigger shrink).
    SdpaConfig cfg{SdpaKernelType::FLASH_ATTN_SPM,
                   head_dim, num_q_heads, num_kv_heads,
                   /*cores*/attn_tp, /*mask*/1};
    SdpaTiling t = sdpa_compute_tiling(cfg, seq_len);
    int64_t nkv = CeilDiv(num_kv_heads, (int64_t)attn_tp);
    return Align(t.tile_n_v16 * t.tile_k * nkv * CeilDiv(seq_len, t.tile_m) * 32, ALIGN);
}

}  // namespace spm

// =============================================================================
// DecoderSpmBuffers 基类 (offset-based)
// =============================================================================

class DecoderSpmBuffers {
public:
    // --- 公共 buffer offsets (所有 core 共用) ---
    uint32_t q_off = 0, k_off = 0, v_off = 0, output_off = 0, oproj_off = 0;
    uint32_t residual1_off = 0, residual2_off = 0;
    uint32_t gate_off = 0, up_off = 0, down_off = 0, input_norm_off = 0;
    uint32_t sdpa_tmp_off = 0, sdpa_mask_off = 0;
    uint32_t input_norm_weight_off = 0, post_attn_norm_weight_off = 0;

    // --- 公共 sizes ---
    size_t q_size = 0, k_size = 0, v_size = 0, oproj_size = 0;
    size_t residual_size = 0, gate_size = 0, down_size = 0, input_norm_size = 0;
    size_t hidden_norm_weight_size = 0, sdpa_tmp_size = 0, sdpa_mask_size = 0;
    int attn_tp = RPU_NUM_CORES;

    explicit DecoderSpmBuffers(const char* subsystem) : sub_(subsystem) {}
    virtual ~DecoderSpmBuffers() = default;

    const char* subsystem() const { return sub_; }

    // =========================================================================
    // 计算公共 size
    // =========================================================================
    void compute_common_sizes(int64_t seq_len, int64_t hidden_size, int64_t head_dim,
                              int64_t intermediate_size, int64_t num_q_heads,
                              int64_t num_kv_heads) {
        attn_tp = std::min((int)RPU_NUM_CORES, (int)num_kv_heads);
        int64_t lqh = num_q_heads / attn_tp;
        int64_t lkv = num_kv_heads * head_dim / attn_tp;

        q_size          = spm::buf_size(seq_len * lqh * head_dim);
        k_size          = spm::buf_size(seq_len * lkv);
        v_size          = k_size;
        oproj_size      = spm::buf_size(seq_len * hidden_size);
        residual_size   = oproj_size;
        gate_size       = spm::buf_size(seq_len * intermediate_size / RPU_NUM_CORES);
        down_size       = oproj_size;
        input_norm_size = oproj_size;
        hidden_norm_weight_size = spm::buf_size(hidden_size);
        sdpa_tmp_size   = spm::sdpa_tmp_bytes(
            seq_len, head_dim, num_q_heads, num_kv_heads, attn_tp);
    }

    // =========================================================================
    // 分配 (all temporary via SpmAllocator)
    // =========================================================================

    // Always-alive buffers + optional MLP buffers
    // include_mlp=true (default): also allocate gate/up/down separately
    // include_mlp=false: MLP handled by allocate_attention_and_mlp()
    void allocate_shared(bool include_mlp = true) {
        if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
        residual1_off    = SPM_ALLOC.alloc_temporary(residual_size);
        input_norm_off   = SPM_ALLOC.alloc_temporary(input_norm_size);
        residual2_off    = input_norm_off;  // alias
        if (include_mlp) {
            gate_off     = SPM_ALLOC.alloc_temporary(gate_size);
            up_off       = SPM_ALLOC.alloc_temporary(gate_size);
            down_off     = SPM_ALLOC.alloc_temporary(down_size);
        }
    }

    void allocate_norm_weights() {
        input_norm_weight_off    = SPM_ALLOC.alloc_temporary(hidden_norm_weight_size);
        post_attn_norm_weight_off = SPM_ALLOC.alloc_temporary(hidden_norm_weight_size);
    }

    // Attention + MLP buffers: lifecycle 不重叠，共享同一 SPM 区域
    //
    // Attention (Phase 2-5): q, k, v, output, oproj, sdpa_tmp
    // MLP      (Phase 7-8): gate, up, down
    //
    // 分配 max(attn_total, mlp_total) 的区域，两组 buffer 各自从头偏移
    void allocate_attention_and_mlp() {
        size_t attn_total = q_size + k_size + v_size + q_size /*output*/ + oproj_size
                          + sdpa_tmp_size + sdpa_mask_size;
        size_t mlp_total  = gate_size + gate_size /*up*/ + down_size;
        size_t shared_size = std::max(attn_total, mlp_total);

        uint32_t base = SPM_ALLOC.alloc_temporary(shared_size);

        // Attention layout (from base)
        q_off        = base;
        k_off        = q_off + q_size;
        v_off        = k_off + k_size;
        output_off   = v_off + v_size;
        oproj_off    = output_off + q_size;  // output uses q_size
        sdpa_tmp_off = oproj_off + oproj_size;
        if (sdpa_mask_size > 0)
            sdpa_mask_off = sdpa_tmp_off + sdpa_tmp_size;

        // MLP layout (from SAME base — overlaps attention, safe because lifecycle disjoint)
        gate_off = base;
        up_off   = base + gate_size;
        down_off = base + gate_size + gate_size;
    }

    // Legacy: separate allocation (for subsystems that don't overlap attn/mlp)
    void allocate_attention() {
        q_off        = SPM_ALLOC.alloc_temporary(q_size);
        k_off        = SPM_ALLOC.alloc_temporary(k_size);
        v_off        = SPM_ALLOC.alloc_temporary(v_size);
        output_off   = SPM_ALLOC.alloc_temporary(q_size);
        oproj_off    = SPM_ALLOC.alloc_temporary(oproj_size);
        sdpa_tmp_off = SPM_ALLOC.alloc_temporary(sdpa_tmp_size);
        if (sdpa_mask_size > 0)
            sdpa_mask_off = SPM_ALLOC.alloc_temporary(sdpa_mask_size);
    }

    // =========================================================================
    // 释放
    // =========================================================================
    virtual void release_all() {
        SPM_ALLOC.reset_temporary();
    }

    // Debug
    void dump_buffers(const char* tag) const {
        printf("[%s:%s] core0: res=0x%08x norm=0x%08x gate=0x%08x up=0x%08x down=0x%08x "
               "q=0x%08x k=0x%08x v=0x%08x out=0x%08x oproj=0x%08x tmp=0x%08x nw=0x%08x\n",
               sub_, tag,
               SPM_ALLOC.addr(0, residual1_off), SPM_ALLOC.addr(0, input_norm_off),
               SPM_ALLOC.addr(0, gate_off), SPM_ALLOC.addr(0, up_off), SPM_ALLOC.addr(0, down_off),
               SPM_ALLOC.addr(0, q_off), SPM_ALLOC.addr(0, k_off), SPM_ALLOC.addr(0, v_off),
               SPM_ALLOC.addr(0, output_off), SPM_ALLOC.addr(0, oproj_off),
               SPM_ALLOC.addr(0, sdpa_tmp_off), SPM_ALLOC.addr(0, input_norm_weight_off));
    }

protected:
    const char* sub_;
};
