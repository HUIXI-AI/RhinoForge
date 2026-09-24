// rpu_rhino_vla_model.cpp — RhinoVLA Action Expert all-layers-once model
//                           (v3 FusedModelBase port)
//
// FORK of rpu_qwenpi05_model.cpp (2026-06-18). RhinoVLA's action expert is the
// same math/shape family as QwenPI05's (depth-N AdaRMS + q_norm/k_norm + M-RoPE
// + prefix-KV cross-attn + SwiGLU + tanh-gated residual), so this started as a
// verbatim copy with symbols renamed qwenpi05->rhino_vla. It is now a SEPARATE
// op (torch.ops.rpu.rhino_vla_*) so RhinoVLA (72-D / depth-18) can be optimized
// independently of QwenPI05 — changes here do NOT touch QwenPI05 and vice versa.
// Renamed SPM buffer labels also avoid allocator name-collision if both load.
// RhinoVLA-specific bits (72-D IO, mask_condition, instance-LoRA merge, 0->1
// denoise) live in the Python adapter (python/rpu_backend/adapters/rhinovla/);
// LoRA is merged into base weights BEFORE set_weights, so this op never sees it.
//
// RhinoVLA is structurally the closest analog of AdaRMS (flat-phase,
// SEQUENTIAL, per-forward cond + per-layer GEMV) PLUS q_norm/k_norm per-layer
// (Qwen3 pattern). No new framework features required — this port reuses
// machinery already validated by 01-04 (AdaRMS flat-phase) and 01-02 (Qwen3
// q_norm/k_norm DMA).
//
// Key differences vs AdaRMS:
//   - QK norm: per-head RMSNorm with [head_dim] weights (from Qwen3)
//   - RoPE: cos/sin passed per-forward (M-RoPE pre-computed on Python side)
//   - MLP activation: SiLU (SwiGLU) instead of GELU (GeGLU)
//   - Prefix KV: non-causal attention with VLM prefix cache; kv_insert_pos
//     = ctx().position + chunk.offset
//   - RoPE uses a logical prefix position; KV insert uses the physical row
//   - Gated residual = SUB -> TANH -> 1xC->NxC MUL -> ADD (4 steps,
//     vs AdaRMS 3-step without TANH)
//
// Key transformations vs v2 (mechanical; compute logic untouched):
//   - Inherit from v3::FusedModelBase (lives in `namespace v3`).
//   - Replace `buf(name)` at SDPA + KV-insert call sites with
//     `addr_offset(name).value` (Pitfall 3 structural fix via typed SpmOffset).
//   - Access model params via pimpl getters.
//   - Drop the legacy subclass-side tensor-tracking calls — class-member
//     cond_ref_/cos_ref_/sin_ref_ assignment alone keeps the tensors alive for
//     the synchronous forward().
//   - set_weights ENDS with `invalidate_model_state();` as last non-empty
//     statement (D-503 per-function awk contract).
//   - Delete v2-only virtual overrides (chunk-size clip, KV-insert / compute
//     subgraph stubs) — all absent from the v3 subclass contract per D-203.
//
// Scope preserved from v2:
//   - Per-forward cond DMA (layer 0 / chunk 0 gate) via cond_loaded_this_forward_.
//   - Per-layer GEMV (cond * dense_w + bias -> [scale|shift|gate]) at
//     chunk.idx == 0, result persists in SPM across chunks of this layer.
//   - (1+scale) preprocessing via scalar ADD to the live GEMV output SPM buffer.
//   - Gated residual 4-step: SUB full-size -> TANH gate slice -> 1xC->NxC MUL
//     with gate slice -> ADD full-size.
//   - Per-layer q_norm_w/k_norm_w DMA + RMSNorm before RoPE.
//   - Always SEQUENTIAL (no KV_FIRST); flat phases (all LayerWide).
//   - cfg.cross_layer_batch_size = num_layers() (force single group).
//   - Debug export flush machinery for per-core SPM dumps.
//
// Design contract: docs/architecture.md#fusedmodelbase-v3--framework-contract

#include "fused_model_base.h"
#include "model_handle_registry.h"
#include "execution_coordinator.h"
#include "rpu_ops.h"
#include "rpu_helpers.h"
#include "rpu_spm_allocator.h"
#include "rpu_spm_residency.h"
#include "rpu_runtime_state.h"  // v5-07: shared runtime globals (g_chunk_size_override, g_debug_tensors, get_cross_layer_batch_size, get_debug_export)

#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

namespace v3 {

namespace {

// Stable descriptor identities. RHINOVLA_FIXED_KERNEL_BASIS: remaining
// launches implement fixed AdaRMS/QK-norm/gated-residual/Euler math,
// immutable preload, debug export, or result staging. They are not runtime
// route choices.
constexpr int64_t RHINO_MLP_GATE_LINEAR_SITE = 7548654080061359966LL;
constexpr int64_t RHINO_MLP_UP_FUSED_LINEAR_SITE = 3623241155541228399LL;
constexpr int64_t RHINO_MLP_UP_PLAIN_LINEAR_SITE = 718838271294101232LL;
constexpr int64_t RHINO_MLP_DOWN_LINEAR_SITE = 1932999177669688573LL;
constexpr int64_t RHINO_MLP_ALL_REDUCE_SITE = 5474499638086968110LL;
constexpr int64_t RHINO_COND_DMA_SITE = 2873533677515363161LL;
constexpr int64_t RHINO_ADARMS_TABLE_DMA_SITE = 8078644669678580208LL;
constexpr int64_t RHINO_Q_LINEAR_SITE = 454271712610923214LL;
constexpr int64_t RHINO_K_LINEAR_SITE = 3251487821785778755LL;
constexpr int64_t RHINO_V_LINEAR_SITE = 6812248271289393368LL;
constexpr int64_t RHINO_Q_ROPE_SITE = 3231791188675653111LL;
constexpr int64_t RHINO_K_ROPE_SITE = 8158949646342380956LL;
constexpr int64_t RHINO_KV_INSERT_SITE = 8134769895626279333LL;
constexpr int64_t RHINO_ATTENTION_SITE = 1426919686904353831LL;
constexpr int64_t RHINO_PREPARE_ALL_REDUCE_SITE = 88222149256769702LL;
constexpr int64_t RHINO_O_LINEAR_SITE = 7249337375525702841LL;
constexpr int64_t RHINO_ATTN_ALL_REDUCE_SITE = 6601348243361589476LL;
constexpr int64_t RHINO_ADARMS_LINEAR_SITE = 8641947192376565010LL;
constexpr int64_t RHINO_ADARMS_ALL_REDUCE_SITE = 8915163346183048699LL;
constexpr int64_t RHINO_ADARMS_PAIR_LINEAR_SITE = 8298134678041751444LL;
constexpr int64_t RHINO_ADARMS_PAIR_ALL_REDUCE_SITE = 7480414339061081683LL;
constexpr int64_t RHINO_PRE_X_DMA_SITE = 7714069497149412415LL;
constexpr int64_t RHINO_PRE_ACTION_MASK_DMA_SITE = 3632590092765186033LL;
constexpr int64_t RHINO_PRE_STATE_DMA_SITE = 6842173452739152635LL;
constexpr int64_t RHINO_PRE_STATE_MASK_DMA_SITE = 8907848472450172079LL;
constexpr int64_t RHINO_PRE_TIME_DMA_SITE = 7616441398499207362LL;
constexpr int64_t RHINO_PRE_COND_DMA_SITE = 6299967694144645109LL;
constexpr int64_t RHINO_PRE_FOLDED_ACTION_LINEAR_SITE = 7043281795651389233LL;
constexpr int64_t RHINO_PRE_ACTION_LINEAR_SITE = 8994515274536934858LL;
constexpr int64_t RHINO_PRE_TIME_ACTION_LINEAR_SITE = 4753004938800951432LL;
constexpr int64_t RHINO_PRE_TIME_TABLE_DMA_SITE = 4168668624893311273LL;
constexpr int64_t RHINO_PRE_TIME_LINEAR_SITE = 1328710968743913034LL;
constexpr int64_t RHINO_PRE_TIME_OUT_LINEAR_SITE = 8622824304681576620LL;
constexpr int64_t RHINO_PRE_ACTION_MASK_LINEAR_SITE = 4957523574588877102LL;
constexpr int64_t RHINO_PRE_STATE_LINEAR_SITE = 1340638388713089312LL;
constexpr int64_t RHINO_PRE_STATE_MASK_LINEAR_SITE = 4225731628177473256LL;
constexpr int64_t RHINO_POST_ADARMS_DMA_SITE = 4403769495735413474LL;
constexpr int64_t RHINO_POST_ACTION_LINEAR_SITE = 1703582534478793943LL;
constexpr int64_t RHINO_POST_OUTPUT_DMA_SITE = 4522756399139884487LL;
constexpr int64_t RHINO_GATED_RESIDUAL_SCHEDULE_SITE = 2303822066361518852LL;
constexpr int64_t RHINO_DENOISE_LOOP_SCHEDULE_SITE = 4844693638692986856LL;
constexpr int64_t RHINO_MASK_SPM_SCHEDULE_SITE = 2981318290883577335LL;
constexpr int64_t RHINO_EXPERT_FUSIONS_SITE = 7993340808033946649LL;
constexpr int64_t RHINO_RESIDENT_PAIR_DMA_SITE = 8943177739067329637LL;
constexpr int64_t RHINO_RESIDENT_FINAL_DMA_SITE = 4321632818720823484LL;
constexpr int64_t RHINO_ADARMS_RESIDENT_SITE = 7331327527407828867LL;
constexpr int64_t RHINO_PACKED_QKV_LINEAR_SITE = 5257906499650722911LL;
constexpr int64_t RHINO_EXPERT_W8A16_SITE = 2568242320735694809LL;
constexpr int64_t RHINO_LINEAR_W8A16_PER_CHANNEL = 1;
constexpr int64_t RHINO_FULL_W8A16_SITE = 7661653982490056347LL;
constexpr int64_t RHINO_HIGH_PRECISION_SITE = 1749523321564235290LL;

void check_rhino_full_w8_projection(const at::Tensor& weight,
                                   const at::Tensor& scale,
                                   int64_t n, int64_t k) {
    TORCH_CHECK(weight.defined() && weight.device().type() == at::kPrivateUse1 &&
                    weight.scalar_type() == at::kChar && weight.is_contiguous() &&
                    weight.dim() == 2 && weight.size(0) == n && weight.size(1) == k,
                "RhinoVLA full W8 requires contiguous INT8 RPU projection [", n, ",", k, "]");
    TORCH_CHECK(scale.defined() && scale.device() == weight.device() &&
                    scale.scalar_type() == at::kHalf && scale.is_contiguous() &&
                    scale.dim() == 1 && scale.numel() == n,
                "RhinoVLA full W8 requires contiguous natural FP16 scale [", n, "]");
    rpu_ddr_flush_force_sized(scale.data_ptr(), scale.nbytes());
    const auto* values = scale.data_ptr<c10::Half>();
    for (int64_t i = 0; i < n; ++i) {
        const float value = static_cast<float>(values[i]);
        TORCH_CHECK(std::isfinite(value) && value > 0,
                    "RhinoVLA full W8 scales must be finite and positive");
    }
}

enum class RhinoMaskSpmSchedule : int64_t {
    EVERY_LAYER = 1,
    FIRST_BODY_FIRST_LAYER = 2,
};

constexpr uint32_t RHINO_KV_CAPABILITIES =
    KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 |
    KV_INSERT_CAP_HYBRID2 | KV_INSERT_CAP_HYBRID3;
constexpr int64_t RHINO_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED = 1;

enum class RhinoRopeRoute : int64_t {
    ROPE_1D_LOGICAL_SUFFIX = 1,
    PARTIAL_MROPE_FULL_HEAD_LOGICAL_SUFFIX = 2,
};

enum class RhinoAllReduceRoute : int64_t {
    PREPARE_RING_INPUT = 3,
};

constexpr int64_t rhino_gated_residual_selector(bool no_sub) {
    return no_sub ? 2 : 1;  // Selectors are positive; args retain the actual bool.
}

enum class RhinoMutableDmaRoute : int64_t {
    DDR_BROADCAST_TO_SPM = 1,
    DDR_SCATTER_TO_SPM = 2,
    SPM_COPY_TO_DDR = 3,
};

// Reuse the all-gather payload's per-core diagonal copy with its ring disabled.
// Unlike the generic slice payload, this ABI indexes its global-address tables
// by core id, so one broadcast launch copies each core's own Q/K/V columns.
void rpu_launch_rhino_packed_qkv_split_spm_kernel(
    uint32_t packed_spm_addr, uint32_t output_spm_addr,
    int64_t rows, int64_t begin_columns, int64_t columns) {
    constexpr const char* caller = "rhino_packed_qkv_split_spm_kernel";
    constexpr uint32_t source_row_bytes = 512 * sizeof(c10::Half);
    constexpr uint32_t warp_count = 8;
    constexpr uint32_t line_bytes = 4096;
    constexpr uint32_t block_bytes = line_bytes * 7;
    TORCH_CHECK(rows == 31 &&
                    ((begin_columns == 0 && columns == 256) ||
                     ((begin_columns == 256 || begin_columns == 384) &&
                      columns == 128)),
                caller, ": requires exact v3 M31 Q256/K128/V128 split");

    const uint32_t chunk_bytes =
        static_cast<uint32_t>(columns * sizeof(c10::Half));
    const uint32_t begin_bytes =
        static_cast<uint32_t>(begin_columns * sizeof(c10::Half));
    const uint64_t source_bytes = static_cast<uint64_t>(rows) * source_row_bytes;
    const uint64_t output_bytes = static_cast<uint64_t>(rows) * chunk_bytes;
    const uint32_t spm_base = SPM_ALLOC.addr(0, 0);
    TORCH_CHECK(packed_spm_addr >= spm_base && output_spm_addr >= spm_base,
                caller, ": operands must be core-0 unified SPM addresses");
    const uint64_t packed_offset = packed_spm_addr - spm_base;
    const uint64_t output_offset = output_spm_addr - spm_base;
    TORCH_CHECK(packed_offset + source_bytes <= SpmAllocator::SPM_USABLE &&
                    output_offset + output_bytes <= SpmAllocator::SPM_USABLE,
                caller, ": source or destination exceeds per-core SPM");
    TORCH_CHECK((packed_offset % 32) == 0 && (output_offset % 32) == 0,
                caller, ": SPM operands must be 32-byte aligned");
    TORCH_CHECK(packed_offset + source_bytes <= output_offset ||
                    output_offset + output_bytes <= packed_offset,
                caller, ": packed source and destination must not overlap");

    const uint32_t chunk_vectors = (chunk_bytes + 31) / 32;
    const uint32_t rows_per_thread = 64 / chunk_vectors;
    const uint16_t blocks = static_cast<uint16_t>(
        (chunk_bytes + block_bytes - 1) / block_bytes);
    constexpr auto kernel_id = KernelId::ALL_GATHER_MULTI_CORE_LITTLE_CHUNK;
    RpuKernelGraph::active().stage_kernel_no_ddr(
        kernel_id, GraphKernelNoDdrProof::AllGatherSpm);
    Kernel_t* kernel = GET_KERNEL(kernel_id);
    TORCH_CHECK(kernel != nullptr,
                caller, ": missing all_gather_multi_core_little_chunk payload");
    kernel->reset_regs();

    // core_num=1 eliminates the ring; the queue still launches all eight
    // physical cores. Source rows retain the 1024-byte packed stride, while
    // destination rows are contiguous Q (512 bytes) or K/V (256 bytes).
    kernel->set_regs(0, static_cast<uint16_t>(1));
    kernel->set_regs(1, static_cast<uint16_t>(rows));
    kernel->set_regs(2, static_cast<uint16_t>(chunk_bytes & 0xFFFF));
    kernel->set_regs(3, static_cast<uint16_t>(chunk_bytes >> 16));
    kernel->set_regs(4, static_cast<uint16_t>(line_bytes));
    kernel->set_regs(5, static_cast<uint16_t>(block_bytes));
    kernel->set_regs(6, static_cast<uint16_t>(chunk_vectors));
    kernel->set_regs(7, static_cast<uint16_t>(rows_per_thread));
    kernel->set_regs(64, static_cast<uint16_t>(warp_count));
    kernel->set_regs(65, static_cast<uint16_t>(1));
    kernel->set_regs(66, static_cast<uint16_t>(1));

    constexpr uint32_t S = 4096;
    for (int core = 0; core < NUM_CORES; ++core) {
        rpu_set_legacy_scm_u16_checked(
            kernel, S + core, static_cast<uint16_t>(chunk_bytes & 0xFFFF), caller);
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 8 + core, static_cast<uint16_t>(chunk_bytes >> 16), caller);
        rpu_set_legacy_scm_u16_checked(kernel, S + 16 + core, blocks, caller);
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 24 + core,
            static_cast<uint16_t>(source_row_bytes & 0xFFFF), caller);
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 32 + core,
            static_cast<uint16_t>(source_row_bytes >> 16), caller);

        const uint32_t source = SPM_ALLOC.addr(
            core, static_cast<uint32_t>(packed_offset + begin_bytes)) - spm_base;
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 40 + core, static_cast<uint16_t>(source & 0xFFFF), caller);
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 48 + core, static_cast<uint16_t>(source >> 16), caller);

        const uint32_t destination = SPM_ALLOC.addr(
            core, static_cast<uint32_t>(output_offset)) - spm_base;
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 56 + core * 16 + core,
            static_cast<uint16_t>(destination & 0xFFFF), caller);
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 64 + core * 16 + core,
            static_cast<uint16_t>(destination >> 16), caller);
    }

    auto* queue = GET_QUEUE(NUM_CORES);
    queue->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    for (int core = 0; core < NUM_CORES; ++core) {
        cores.push_back(static_cast<uint8_t>(core));
    }
    queue->enqueu_kernel(
        *kernel, {static_cast<uint16_t>(warp_count), 1, 1}, cores);
}

}  // namespace

// Python resolves the legacy environment controls once before handle creation
// and binds this immutable value object before any weights. Descriptor
// generation and graph emission therefore share one per-handle authority.
struct RhinoVLAColdConfig {
    bool denoise_static_context_cache = false;
    bool fused_adarms = false;
    bool skip_adarms_gemv = false;
    bool precompute_adarms = false;
    bool precompute_time_proj = false;
    bool fold_action_time_in = false;
    bool gated_no_sub = false;
    bool fused_silu_mul = false;
    bool expert_fusions = false;
    bool adarms_resident = false;
    bool packed_qkv = false;
    bool aligned_kv = false;
};

// =============================================================================
// RhinoVLAModel — v3::FusedModelBase subclass (RhinoVLA Action Expert)
// =============================================================================

class RhinoVLAModel : public FusedModelBase {
public:
    struct LayerWeights {
        // Attention
        at::Tensor q_w, k_w, v_w, o_w;
        // MLP
        at::Tensor gate_proj_w, up_proj_w, down_proj_w;
        // AdaRMS GEMV (per-layer)
        at::Tensor attn_dense_w;    // [3*hidden, hidden] fp16, row-partition swizzle (K split 8 cores)
        at::Tensor attn_dense_b;    // [3*hidden] fp16
        at::Tensor mlp_dense_w;     // [3*hidden, hidden] fp16, row-partition swizzle (K split 8 cores)
        at::Tensor mlp_dense_b;     // [3*hidden] fp16
        at::Tensor pair_dense_w;    // [6*hidden, hidden] fp16, fused attn+mlp cond projection
        at::Tensor pair_dense_b;    // [6*hidden] fp16
        // QK norm (per-layer, new vs AdaRMS)
        at::Tensor q_norm_w;        // [head_dim] fp16
        at::Tensor k_norm_w;        // [head_dim] fp16
        at::Tensor packed_qkv_w;    // [Q + K + V, hidden], cold byte-packed TP8 owner
        // Natural output-channel FP16 [N] scales; only the seven expert
        // projections may carry INT8 weights. AdaRMS, norms and IO stay FP16.
        at::Tensor q_ws, k_ws, v_ws, o_ws;
        at::Tensor gate_ws, up_ws, down_ws;
        at::Tensor attn_dense_ws, mlp_dense_ws, pair_dense_ws;
    };

    RhinoVLAModel() = default;

    void set_runtime_config(
        bool denoise_static_context_cache,
        bool fused_adarms,
        bool skip_adarms_gemv,
        bool precompute_adarms,
        bool precompute_time_proj,
        bool fold_action_time_in,
        bool gated_no_sub,
        bool fused_silu_mul,
        bool expert_fusions,
        bool adarms_resident,
        bool packed_qkv = false,
        bool aligned_kv = false) {
        TORCH_CHECK(!cold_config_bound_,
                    "RhinoVLA runtime config may be bound only once");
        TORCH_CHECK(num_layers() == 0 && layer_weights_.empty() &&
                        !denoise_loop_weights_ready_,
                    "RhinoVLA runtime config must be bound before weights");
        TORCH_CHECK(!(expert_fusions || adarms_resident) ||
                        (precompute_adarms && !skip_adarms_gemv),
                    "RhinoVLA expert transfer requires precomputed AdaRMS tables");
        TORCH_CHECK(!expert_fusions || gated_no_sub,
                    "RhinoVLA expert fusions require gated_no_sub");
        // Bind payload availability while installing the cold owner. The dry
        // planner runs under a reconfiguration guard and must not GET_KERNEL.
        TORCH_CHECK(!expert_fusions ||
                        (GET_KERNEL(KernelId::PI05_ADARMS_NORM_SHIFT_H1024) &&
                         GET_KERNEL(KernelId::PI05_GATED_RESIDUAL_H1024)),
                    "RhinoVLA expert fusions require both Pi05 H1024 payloads");
        cold_config_ = {
            denoise_static_context_cache,
            fused_adarms,
            skip_adarms_gemv,
            precompute_adarms,
            precompute_time_proj,
            fold_action_time_in,
            gated_no_sub,
            fused_silu_mul,
            expert_fusions,
            adarms_resident,
            packed_qkv,
            aligned_kv,
        };
        cold_config_bound_ = true;
        invalidate_model_state();
    }

    void set_fast_replay_skip_layer_loop(bool enabled) {
        fast_replay_skip_layer_loop_ = enabled;
    }

    void set_high_precision(bool enabled) {
        constexpr const char* operation = "rhino_vla_set_high_precision";
        RpuExecutionCoordinator::require_graph_quiescent(operation);
        RpuExecutionCoordinator::check_current_thread_execution_allowed(operation);
        TORCH_CHECK(!high_precision_bound_ && get_last_resolved_chunk_size() == 0 &&
                        !denoise_loop_weights_ready_ && !denoise_adarms_tables_ready_,
                    "RhinoVLA precision must be bound once before loop IO/tables and dispatch");
        TORCH_CHECK(cold_config_bound_ && num_layers() > 0,
                    "RhinoVLA precision requires installed expert weights");
        TORCH_CHECK(!enabled ||
                        (expert_w8a16_ && full_action_w8a16_ &&
                         cold_config_.precompute_adarms && !cold_config_.skip_adarms_gemv &&
                         !cold_config_.packed_qkv && num_layers() == 18 &&
                         hidden_size() == 1024 && intermediate_size() == 3072 &&
                         num_q_heads() == 16 && num_kv_heads() == 8 &&
                         head_dim() == 128 && attn_tp() == 8),
                    "RhinoVLA high precision requires exact full W8 v3 with precomputed AdaRMS");
        // Payload admission is cold and precedes every owner mutation. The
        // planner and replay never query the kernel cache or process env.
        if (enabled) rpu_require_high_precision_math_kernels();
        high_precision_ = enabled;
        high_precision_bound_ = true;
        invalidate_model_state();
    }


    void set_high_precision_fusions(bool enabled) {
        constexpr const char* operation = "rhino_vla_set_high_precision_fusions";
        RpuExecutionCoordinator::require_graph_quiescent(operation);
        RpuExecutionCoordinator::check_current_thread_execution_allowed(operation);
        TORCH_CHECK(!high_precision_fusions_bound_ && high_precision_bound_ && high_precision_ &&
                        get_last_resolved_chunk_size() == 0 && !denoise_loop_weights_ready_ &&
                        !denoise_adarms_tables_ready_,
                    "RhinoVLA HIGH fusions require a cold HIGH owner before loop IO/tables and dispatch");
        if (enabled) rpu_require_high_precision_fusion_kernels();
        high_precision_fusions_ = enabled;
        high_precision_fusions_bound_ = true;
        invalidate_model_state();
    }

    void set_vector_k_norm(bool enabled) {
        constexpr const char* operation = "rhino_vla_set_vector_k_norm";
        RpuExecutionCoordinator::require_graph_quiescent(operation);
        RpuExecutionCoordinator::check_current_thread_execution_allowed(operation);
        TORCH_CHECK(!vector_k_norm_bound_ && high_precision_bound_ && high_precision_ &&
                        get_last_resolved_chunk_size() == 0 && !denoise_loop_weights_ready_ &&
                        !denoise_adarms_tables_ready_,
                    "RhinoVLA vector K norm requires a cold HIGH owner before loop IO/tables and dispatch");
        TORCH_CHECK(!enabled || cold_config_.aligned_kv,
                    "RhinoVLA vector K norm requires aligned KV storage");
        // HIGH admission already resolved the Newton V32 payload. No extra
        // storage: K has 32 rows and input_norm has 31*1024 FP16 elements.
        vector_k_norm_ = enabled;
        vector_k_norm_bound_ = true;
        invalidate_model_state();
    }

    void set_vector_q_norm(bool enabled) {
        constexpr const char* operation = "rhino_vla_set_vector_q_norm";
        RpuExecutionCoordinator::require_graph_quiescent(operation);
        RpuExecutionCoordinator::check_current_thread_execution_allowed(operation);
        TORCH_CHECK(!vector_q_norm_bound_ && high_precision_bound_ && high_precision_ &&
                        get_last_resolved_chunk_size() == 0 && !denoise_loop_weights_ready_ &&
                        !denoise_adarms_tables_ready_,
                    "RhinoVLA vector Q norm requires a cold HIGH owner before loop IO/tables and dispatch");
        // HIGH admission resolves Newton V32 and the exact v3 expert geometry.
        // The planner owns the extra two rows in both q and output buffers.
        vector_q_norm_ = enabled;
        vector_q_norm_bound_ = true;
        invalidate_model_state();
    }

    void set_partial_rope(bool enabled) {
        constexpr const char* operation = "rhino_vla_set_partial_rope";
        RpuExecutionCoordinator::require_graph_quiescent(operation);
        RpuExecutionCoordinator::check_current_thread_execution_allowed(operation);
        TORCH_CHECK(!partial_rope_bound_ && high_precision_bound_ && high_precision_ &&
                        get_last_resolved_chunk_size() == 0 && !denoise_loop_weights_ready_ &&
                        !denoise_adarms_tables_ready_,
                    "RhinoVLA partial RoPE requires a cold HIGH owner before loop IO/tables and dispatch");
        TORCH_CHECK(!enabled || GET_KERNEL(KernelId::PARTIAL_MROPE),
                    "RhinoVLA partial RoPE requires the partial_mrope payload");
        partial_rope_ = enabled;
        partial_rope_bound_ = true;
        invalidate_model_state();
    }


    std::vector<int64_t> planner_cache_identity() const {
        auto identity = FusedModelBase::planner_cache_identity();
        identity.push_back(rope_position_);
        identity.push_back(static_cast<int64_t>(fast_replay_skip_layer_loop_));
        return identity;
    }

    void set_rope_position(int64_t position) {
        TORCH_CHECK(position >= 0,
                    "RhinoVLA logical RoPE position must be non-negative");
        rope_position_ = position;
    }

    void set_configured_chunk_size(int64_t chunk_size) {
        TORCH_CHECK(chunk_size == 0 ||
                        (chunk_size >= 16 && chunk_size % 16 == 0),
                    "RhinoVLA action chunk size must be 0 (auto) or a "
                    "positive multiple of 16, got ", chunk_size);
        set_control_chunk_size_override(
            chunk_size, "RhinoVLAModel::set_configured_chunk_size");
    }

    void stage_configured_chunk_size(uint64_t token, int64_t chunk_size) {
        stage_control_chunk_size_override(
            token, chunk_size,
            "RhinoVLAModel::stage_configured_chunk_size");
    }

    std::vector<int64_t> resolve_action_stage_domain(
        int64_t execution_len, int64_t logical_len, int64_t position,
        int64_t kv_len, bool use_attention_mask, bool is_causal,
        int64_t requested_chunk_size)
    {
        TORCH_CHECK(!loop_mode_ || dt_pinned_,
                    "RhinoVLA action planner requires "
                    "prepare_denoise_loop_schedule before loop planning");
        detail::validate_fmb_planning_shape(
            execution_len, position, "RhinoVLA action planner");
        TORCH_CHECK(logical_len > 0 && logical_len <= execution_len,
                    "RPU_PLANNER_REJECT:CAPABILITY: RhinoVLA action logical "
                    "length must be in [1, execution_len]");
        TORCH_CHECK(kv_len >= position + execution_len,
                    "RPU_PLANNER_REJECT:CAPABILITY: RhinoVLA action KV "
                    "length does not cover prefix plus execution rows");
        std::optional<at::Tensor> mask_shape = std::nullopt;
        if (use_attention_mask) {
            mask_shape = at::empty(
                {1, 1, execution_len, kv_len},
                at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
        }
        return encode_fmb_prefill_stage_domain(
            resolve_prefill_stage_domain_for_shape(
                execution_len, position, mask_shape, is_causal,
                requested_chunk_size, logical_len));
    }

    void set_denoise_loop_weights(
        const at::Tensor& action_in_w, const at::Tensor& action_in_b,
        const at::Tensor& action_time_in_w, const at::Tensor& action_time_in_b,
        const at::Tensor& time_in_action_w, const at::Tensor& time_in_time_w,
        const at::Tensor& time_in_b,
        const at::Tensor& time_out_w, const at::Tensor& time_out_b,
        const at::Tensor& state_w, const at::Tensor& state_b,
        const at::Tensor& state_mask_w, const at::Tensor& state_mask_b,
        const at::Tensor& action_mask_w, const at::Tensor& action_mask_b,
        const at::Tensor& final_norm_w, const at::Tensor& final_norm_b,
        const at::Tensor& action_out_w, const at::Tensor& action_out_b,
        const at::Tensor& cos, const at::Tensor& sin,
        int64_t action_dim, int64_t action_dim_pad,
        int64_t state_dim, int64_t state_dim_pad,
        int64_t action_horizon, int64_t suffix_len, bool direct_action_input,
        at::TensorList io_scales = {}) {
        TORCH_CHECK(num_layers() > 0,
                    "RhinoVLAModel::set_denoise_loop_weights called before set_weights");
        if (high_precision_) {
            RpuExecutionCoordinator::require_graph_quiescent("rhino_vla_set_high_precision_io");
            RpuExecutionCoordinator::check_current_thread_execution_allowed("rhino_vla_set_high_precision_io");
            TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                        "RhinoVLA high precision IO must be installed before dispatch");
        }
        TORCH_CHECK(action_dim > 0 && action_dim_pad >= action_dim &&
                    state_dim > 0 && state_dim_pad >= state_dim &&
                    action_horizon > 0 && suffix_len == action_horizon + 1,
                    "RhinoVLAModel::set_denoise_loop_weights: invalid dims");
        TORCH_CHECK(!expert_w8a16_ ||
                        (direct_action_input && action_dim == 96 && state_dim == 96 &&
                         action_horizon == 30 && suffix_len == 31),
                    "RhinoVLA W8A16 requires the ordinary v3 96D/H30/M31 action loop");
        TORCH_CHECK(full_action_w8a16_ ? io_scales.size() == 6 : io_scales.empty(),
                    "RhinoVLA full W8 IO requires exactly six scales on a full W8 expert owner");
        if (full_action_w8a16_) {
            TORCH_CHECK(!RpuKernelGraph::has_active() && action_dim_pad == 96 && state_dim_pad == 96,
                        "RhinoVLA full W8 IO requires cold exact 96D installation");
            check_rhino_full_w8_projection(action_in_w, io_scales[0], 1024, 96);
            check_rhino_full_w8_projection(state_w, io_scales[1], 1024, 96);
            check_rhino_full_w8_projection(state_mask_w, io_scales[2], 1024, 96);
            check_rhino_full_w8_projection(action_mask_w, io_scales[3], 1024, 96);
            check_rhino_full_w8_projection(final_norm_w, io_scales[4], 3072, 1024);
            check_rhino_full_w8_projection(action_out_w, io_scales[5], 96, 1024);
        }
        auto check_rpu = [](const at::Tensor& t, const char* name) {
            TORCH_CHECK(t.defined() && t.device().type() == at::kPrivateUse1 &&
                        t.scalar_type() == at::kHalf && t.is_contiguous(),
                        "RhinoVLAModel::set_denoise_loop_weights: ", name,
                        " must be contiguous fp16 RPU tensor");
        };
        if (!full_action_w8a16_) check_rpu(action_in_w, "action_in_w");
        check_rpu(action_in_b, "action_in_b");
        check_rpu(action_time_in_w, "action_time_in_w");
        check_rpu(action_time_in_b, "action_time_in_b");
        check_rpu(time_in_action_w, "time_in_action_w");
        check_rpu(time_in_time_w, "time_in_time_w");
        check_rpu(time_in_b, "time_in_b");
        check_rpu(time_out_w, "time_out_w");
        check_rpu(time_out_b, "time_out_b");
        if (!full_action_w8a16_) check_rpu(state_w, "state_w");
        check_rpu(state_b, "state_b");
        if (!full_action_w8a16_) check_rpu(state_mask_w, "state_mask_w");
        check_rpu(state_mask_b, "state_mask_b");
        if (!full_action_w8a16_) check_rpu(action_mask_w, "action_mask_w");
        check_rpu(action_mask_b, "action_mask_b");
        if (!full_action_w8a16_) check_rpu(final_norm_w, "final_norm_w");
        check_rpu(final_norm_b, "final_norm_b");
        if (!full_action_w8a16_) check_rpu(action_out_w, "action_out_w");
        check_rpu(action_out_b, "action_out_b");
        check_rpu(cos, "cos");
        check_rpu(sin, "sin");
        // Shape contract (mirrors forward()): build_layer_subgraph reads cos/sin as
        // a raw [max_pos, head_dim/2] table. A full-dim or short table silently
        // misreads / runs off the end — validate here, not at first replay.
        TORCH_CHECK(cos.sizes() == sin.sizes(),
                    "RhinoVLAModel::set_denoise_loop_weights: cos/sin shape mismatch: cos=",
                    cos.sizes(), " sin=", sin.sizes());
        TORCH_CHECK(cos.dim() == 2 && cos.size(1) == head_dim() / 2,
                    "RhinoVLAModel::set_denoise_loop_weights: cos/sin must be 2D "
                    "[max_pos, head_dim/2=", head_dim() / 2, "]; got ", cos.sizes());
        TORCH_CHECK(cos.size(0) >= suffix_len,
                    "RhinoVLAModel::set_denoise_loop_weights: cos/sin rows ", cos.size(0),
                    " < suffix_len ", suffix_len, " — the denoise RoPE kernel reads "
                    "[0, suffix_len) and would run past the table end");

        TORCH_CHECK(!direct_action_input ||
                    (!cold_config_.precompute_time_proj && !cold_config_.fold_action_time_in),
                    "RhinoVLA direct action input has no action/time MLP; "
                    "time projection precompute and folding are invalid");
        std::vector<at::Tensor> next_io_scales(io_scales.begin(), io_scales.end());
        if (full_action_w8a16_) {
            for (const auto& bias : {action_in_b, state_b, state_mask_b, action_mask_b}) {
                TORCH_CHECK(bias.dim() == 1 && bias.numel() == 1024,
                            "RhinoVLA full W8 IO bias must be [1024]");
            }
            TORCH_CHECK(final_norm_b.dim() == 1 && final_norm_b.numel() == 3072 &&
                            action_out_b.dim() == 1 && action_out_b.numel() == 96,
                        "RhinoVLA full W8 final biases must be [3072]/[96]");
        }
        direct_action_input_ = direct_action_input;
        full_io_scales_.swap(next_io_scales);
        if (full_action_w8a16_) {
            denoise_adarms_tables_ready_ = false;
            gates_tanh_precomputed_ = false;
            full_cold_weights_.clear();
            full_cold_scales_.clear();
            full_cold_biases_.clear();
            adarms_pair_table_ = at::Tensor();
            adarms_final_table_ = at::Tensor();
        }
        action_in_w_ = action_in_w;
        action_in_b_ = action_in_b;
        action_time_in_w_ = action_time_in_w;
        action_time_in_b_ = action_time_in_b;
        time_in_action_w_ = time_in_action_w;
        time_in_time_w_ = time_in_time_w;
        time_in_b_ = time_in_b;
        time_out_w_ = time_out_w;
        time_out_b_ = time_out_b;
        state_w_ = state_w;
        state_b_ = state_b;
        state_mask_w_ = state_mask_w;
        state_mask_b_ = state_mask_b;
        action_mask_w_ = action_mask_w;
        action_mask_b_ = action_mask_b;
        final_norm_w_ = final_norm_w;
        final_norm_b_ = final_norm_b;
        action_out_w_ = action_out_w;
        action_out_b_ = action_out_b;
        cos_loop_ = cos;
        sin_loop_ = sin;
        action_dim_ = action_dim;
        action_dim_pad_ = action_dim_pad;
        state_dim_ = state_dim;
        state_dim_pad_ = state_dim_pad;
        action_horizon_ = action_horizon;
        suffix_len_ = suffix_len;
        action_emb_stage_ = at::empty(
            {1, suffix_len_, hidden_size()},
            at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
        denoise_loop_weights_ready_ = true;
        // Select the loop shape before planning; prepare_denoise_loop_schedule
        // must also bind its steps and dt before a dry descriptor is admitted.
        loop_mode_ = true;
        invalidate_model_state();
    }

    void prepare_denoise_loop_schedule(int64_t num_steps, double dt) {
        TORCH_CHECK(denoise_loop_weights_ready_,
                    "RhinoVLA denoise loop schedule requires set_denoise_loop_weights");
        TORCH_CHECK(!high_precision_ || num_steps == 10,
                    "RhinoVLA high precision requires the ordinary ten-step v3 loop");
        TORCH_CHECK(num_steps >= 1, "RhinoVLA denoise loop num_steps must be >= 1");
        // Compare before multiplying so an untrusted int64 step count cannot
        // overflow the existing conservative graph-node budget.
        TORCH_CHECK(num_steps <= (32000 - 1) / 1400,
                    "RhinoVLA denoise loop node budget exceeded: num_steps=",
                    num_steps);
        TORCH_CHECK(std::isfinite(dt) && dt != 0.0,
                    "RhinoVLA denoise loop dt must be finite and nonzero");
        const c10::Half dt_half = c10::Half(static_cast<float>(dt));
        TORCH_CHECK(std::isfinite(static_cast<float>(dt_half)) &&
                        static_cast<float>(dt_half) != 0.0f,
                    "RhinoVLA denoise loop dt must remain finite and nonzero in fp16");
        if (get_last_resolved_chunk_size() != 0) {
            TORCH_CHECK(loop_mode_ && dt_pinned_ && num_steps_ == num_steps,
                        "RhinoVLA denoise loop schedule changed after dispatch; "
                        "rebuild the owner");
            TORCH_CHECK(dt_.x == dt_half.x,
                        "RhinoVLA denoise loop dt changed across replays; "
                        "rebuild the owner");
            return;
        }
        if (!loop_mode_ || !dt_pinned_ || num_steps_ != num_steps ||
                dt_.x != dt_half.x) {
            loop_mode_ = true;
            num_steps_ = num_steps;
            dt_ = dt_half;
            dt_pinned_ = true;
            invalidate_model_state();
        }
    }

    void set_denoise_loop_adarms_tables(
        const at::Tensor& pair_table,
        const at::Tensor& final_table,
        at::TensorList cold_weights = {}, at::TensorList cold_scales = {},
        at::TensorList cold_biases = {}, bool gates_tanh_precomputed = false,
        bool high_precision = false) {
        TORCH_CHECK(num_layers() > 0,
                    "RhinoVLAModel::set_denoise_loop_adarms_tables before set_weights");
        TORCH_CHECK(high_precision == high_precision_,
                    "RhinoVLA AdaRMS table precision must match the installed owner");
        if (high_precision_) {
            RpuExecutionCoordinator::require_graph_quiescent("rhino_vla_set_high_precision_tables");
            RpuExecutionCoordinator::check_current_thread_execution_allowed("rhino_vla_set_high_precision_tables");
            TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                        "RhinoVLA high precision tables must be installed before dispatch");
        }
        auto check_rpu = [](const at::Tensor& t, const char* name) {
            TORCH_CHECK(t.defined() && t.device().type() == at::kPrivateUse1 &&
                        t.scalar_type() == at::kHalf && t.is_contiguous(),
                        "RhinoVLAModel::set_denoise_loop_adarms_tables: ", name,
                        " must be contiguous fp16 RPU tensor");
        };
        check_rpu(pair_table, "pair_table");
        check_rpu(final_table, "final_table");
        TORCH_CHECK(pair_table.dim() == 3,
                    "RhinoVLA AdaRMS pair_table must be [steps,layers,6H], got ",
                    pair_table.sizes());
        TORCH_CHECK(final_table.dim() == 2,
                    "RhinoVLA AdaRMS final_table must be [steps,3H], got ",
                    final_table.sizes());
        TORCH_CHECK(pair_table.size(0) == final_table.size(0),
                    "RhinoVLA AdaRMS table steps mismatch pair=", pair_table.size(0),
                    " final=", final_table.size(0));
        TORCH_CHECK(pair_table.size(1) == num_layers(),
                    "RhinoVLA AdaRMS pair_table layer count ", pair_table.size(1),
                    " != num_layers=", num_layers());
        TORCH_CHECK(pair_table.size(2) == 6 * hidden_size(),
                    "RhinoVLA AdaRMS pair_table last dim ", pair_table.size(2),
                    " != 6H=", 6 * hidden_size());
        TORCH_CHECK(final_table.size(1) == 3 * hidden_size(),
                    "RhinoVLA AdaRMS final_table last dim ", final_table.size(1),
                    " != 3H=", 3 * hidden_size());
        TORCH_CHECK(full_action_w8a16_
                        ? (cold_weights.size() == 19 && cold_scales.size() == 19 && cold_biases.size() == 19)
                        : (cold_weights.empty() && cold_scales.empty() && cold_biases.empty()),
                    "RhinoVLA full W8 AdaRMS tables require nineteen cold W8 projection owners");
        TORCH_CHECK(!gates_tanh_precomputed ||
                        (full_action_w8a16_ && cold_config_.precompute_adarms &&
                         cold_config_.adarms_resident && !cold_config_.skip_adarms_gemv &&
                         denoise_loop_weights_ready_ && loop_mode_ && direct_action_input_ &&
                         num_layers() == 18 && hidden_size() == 1024 &&
                         intermediate_size() == 3072 && num_q_heads() == 16 &&
                         num_kv_heads() == 8 && head_dim() == 128 && attn_tp() == 8 &&
                         num_steps_ == 10 && action_dim_ == 96 && state_dim_ == 96 &&
                         action_horizon_ == 30 && suffix_len_ == 31),
                    "RhinoVLA gate TANH precompute requires cold full W8 resident AdaRMS exact v3 loop");
        if (full_action_w8a16_) {
            TORCH_CHECK(!RpuKernelGraph::has_active() && pair_table.size(0) == 10,
                        "RhinoVLA full W8 AdaRMS tables require cold ten-step installation");
            for (int64_t i = 0; i < 19; ++i) {
                const int64_t n = i == 18 ? 3072 : 6144;
                check_rhino_full_w8_projection(cold_weights[i], cold_scales[i], n, 1024);
                check_rpu(cold_biases[i], "cold_bias");
                TORCH_CHECK(cold_biases[i].dim() == 1 && cold_biases[i].numel() == n,
                            "RhinoVLA full W8 AdaRMS cold bias shape mismatch");
            }
        }
        std::vector<at::Tensor> next_cold_weights(cold_weights.begin(), cold_weights.end());
        std::vector<at::Tensor> next_cold_scales(cold_scales.begin(), cold_scales.end());
        std::vector<at::Tensor> next_cold_biases(cold_biases.begin(), cold_biases.end());

        full_cold_weights_.swap(next_cold_weights);
        full_cold_scales_.swap(next_cold_scales);
        full_cold_biases_.swap(next_cold_biases);
        gates_tanh_precomputed_ = gates_tanh_precomputed;
        adarms_pair_table_ = pair_table;
        adarms_final_table_ = final_table;
        adarms_pair_table_src_base_ =
            ::rhino_lkn::RpuGetDevAddr(adarms_pair_table_.data_ptr());
        adarms_final_table_src_base_ =
            ::rhino_lkn::RpuGetDevAddr(adarms_final_table_.data_ptr());
        rpu_ddr_flush_force(adarms_pair_table_.data_ptr<c10::Half>());
        rpu_ddr_flush_force(adarms_final_table_.data_ptr<c10::Half>());
        denoise_adarms_tables_ready_ = true;
        invalidate_model_state();
    }

    void set_denoise_loop_time_proj_table(const at::Tensor& table) {
        TORCH_CHECK(num_layers() > 0,
                    "RhinoVLAModel::set_denoise_loop_time_proj_table before set_weights");
        TORCH_CHECK(table.defined() && table.device().type() == at::kPrivateUse1 &&
                    table.scalar_type() == at::kHalf && table.is_contiguous(),
                    "RhinoVLA time-proj table must be contiguous fp16 RPU tensor");
        TORCH_CHECK(table.dim() == 2,
                    "RhinoVLA time-proj table must be [steps,H], got ",
                    table.sizes());
        TORCH_CHECK(table.size(1) == hidden_size(),
                    "RhinoVLA time-proj table last dim ", table.size(1),
                    " != H=", hidden_size());

        time_proj_table_ = table;
        time_proj_table_src_base_ =
            ::rhino_lkn::RpuGetDevAddr(time_proj_table_.data_ptr());
        rpu_ddr_flush_force(time_proj_table_.data_ptr<c10::Half>());
        denoise_time_proj_table_ready_ = true;
        invalidate_model_state();
    }

    // ========================================================================
    // set_weights — accepts per-layer weights including QK norm
    //
    // Unlike AdaRMS, cos/sin are NOT stored here (they are per-forward).
    // QK norm weights (q_norm, k_norm) are per-layer [head_dim] fp16.
    // ========================================================================
    void set_weights(
        at::TensorList q_w_list, at::TensorList k_w_list,
        at::TensorList v_w_list, at::TensorList o_w_list,
        at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
        at::TensorList attn_dense_w_list, at::TensorList attn_dense_b_list,
        at::TensorList mlp_dense_w_list,  at::TensorList mlp_dense_b_list,
        at::TensorList pair_dense_w_list, at::TensorList pair_dense_b_list,
        at::TensorList q_norm_list, at::TensorList k_norm_list,
        int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size,
        double eps,
        at::TensorList q_w_scale_list = {}, at::TensorList k_w_scale_list = {},
        at::TensorList v_w_scale_list = {}, at::TensorList o_w_scale_list = {},
        at::TensorList gate_scale_list = {}, at::TensorList up_scale_list = {},
        at::TensorList down_scale_list = {},
        at::TensorList attn_dense_scale_list = {}, at::TensorList mlp_dense_scale_list = {},
        at::TensorList pair_dense_scale_list = {})
    {
        TORCH_CHECK(!high_precision_bound_,
                    "RhinoVLA expert weights must be installed before binding precision; rebuild the owner");
        TORCH_CHECK(cold_config_bound_,
                    "rhino_vla_set_runtime_config must be called before "
                    "rhino_vla_set_weights");
        int64_t N = static_cast<int64_t>(q_w_list.size());
        TORCH_CHECK(N > 0, "rhino_vla_set_weights: empty weight lists");
        TORCH_CHECK(num_q_heads > 0 && num_kv_heads > 0 && head_dim > 0
                    && hidden_size > 0 && intermediate_size > 0,
                    "rhino_vla_set_weights: dim params must be positive");
        TORCH_CHECK(num_q_heads % num_kv_heads == 0,
                    "rhino_vla_set_weights: num_q_heads (", num_q_heads,
                    ") must be divisible by num_kv_heads (", num_kv_heads, ")");

        // All per-layer lists must have same length
        auto check_list = [&](const at::TensorList& l, const char* n) {
            TORCH_CHECK(static_cast<int64_t>(l.size()) == N,
                        "rhino_vla_set_weights: ", n, ".size()=", l.size(),
                        " != num_layers=", N);
        };
        check_list(k_w_list,          "k_w_list");
        check_list(v_w_list,          "v_w_list");
        check_list(o_w_list,          "o_w_list");
        check_list(gate_list,         "gate_list");
        check_list(up_list,           "up_list");
        check_list(down_list,         "down_list");
        check_list(attn_dense_w_list, "attn_dense_w_list");
        check_list(attn_dense_b_list, "attn_dense_b_list");
        check_list(mlp_dense_w_list,  "mlp_dense_w_list");
        check_list(mlp_dense_b_list,  "mlp_dense_b_list");
        check_list(pair_dense_w_list, "pair_dense_w_list");
        check_list(pair_dense_b_list, "pair_dense_b_list");
        check_list(q_norm_list,       "q_norm_list");
        check_list(k_norm_list,       "k_norm_list");

        // Per-layer defined/rank checks: 2D for linear weights, 1D for bias/norm
        auto check_rank = [&](const at::TensorList& list, const char* name,
                              int64_t expected_rank) {
            for (int64_t i = 0; i < N; i++) {
                TORCH_CHECK(list[i].defined(),
                            "rhino_vla_set_weights: ", name, "[", i, "] undefined");
                TORCH_CHECK(list[i].dim() == expected_rank,
                            "rhino_vla_set_weights: ", name, "[", i, "] must be ",
                            expected_rank, "D, got ", list[i].dim(), "D");
            }
        };
        check_rank(q_w_list,          "q_w_list",          2);
        check_rank(k_w_list,          "k_w_list",          2);
        check_rank(v_w_list,          "v_w_list",          2);
        check_rank(o_w_list,          "o_w_list",          2);
        check_rank(gate_list,         "gate_list",         2);
        check_rank(up_list,           "up_list",           2);
        check_rank(down_list,         "down_list",         2);
        check_rank(attn_dense_w_list, "attn_dense_w_list", 2);
        check_rank(attn_dense_b_list, "attn_dense_b_list", 1);
        check_rank(mlp_dense_w_list,  "mlp_dense_w_list",  2);
        check_rank(mlp_dense_b_list,  "mlp_dense_b_list",  1);
        check_rank(pair_dense_w_list, "pair_dense_w_list", 2);
        check_rank(pair_dense_b_list, "pair_dense_b_list", 1);
        check_rank(q_norm_list,       "q_norm_list",       1);
        check_rank(k_norm_list,       "k_norm_list",       1);

        const bool w8a16 = !q_w_scale_list.empty();
        const bool full_w8a16 = !attn_dense_scale_list.empty();
        TORCH_CHECK(!high_precision_ || full_w8a16,
                    "RhinoVLA high precision owner cannot replace full W8 weights with another precision");
        TORCH_CHECK(!full_w8a16 ||
                        (w8a16 && cold_config_.precompute_adarms && !cold_config_.skip_adarms_gemv),
                    "RhinoVLA full W8 requires expert W8 and precomputed AdaRMS without skip");
        for (const auto scales : {attn_dense_scale_list, mlp_dense_scale_list, pair_dense_scale_list}) {
            TORCH_CHECK(full_w8a16 ? static_cast<int64_t>(scales.size()) == N : scales.empty(),
                        "RhinoVLA full W8 requires all three AdaRMS scale lists");
        }
        TORCH_CHECK(!w8a16 || !cold_config_.packed_qkv,
                    "RhinoVLA W8A16 is incompatible with packed_qkv");
        for (const at::TensorList scales : {
                q_w_scale_list, k_w_scale_list, v_w_scale_list, o_w_scale_list,
                gate_scale_list, up_scale_list, down_scale_list}) {
            TORCH_CHECK(w8a16 ? static_cast<int64_t>(scales.size()) == N : scales.empty(),
                        "RhinoVLA W8A16 requires all seven scale lists with one tensor per layer");
        }
        for (const at::TensorList weights : {
                q_w_list, k_w_list, v_w_list, o_w_list,
                gate_list, up_list, down_list}) {
            for (const auto& weight : weights) {
                TORCH_CHECK(weight.scalar_type() == (w8a16 ? at::kChar : at::kHalf),
                            "RhinoVLA projections must be uniformly FP16 without scales or INT8 with all seven scales");
            }
        }

        // Transfer opt-ins and W8A16 are scoped to the exact v3 expert. Validate
        // all projections before committing any cold installation.
        if (cold_config_.packed_qkv || cold_config_.aligned_kv || w8a16) {
            TORCH_CHECK(!RpuKernelGraph::has_active(),
                        "RhinoVLA QKV/KV optimization requires cold installation");
            TORCH_CHECK(N == 18 && hidden_size == 1024 && intermediate_size == 3072 &&
                            num_q_heads == 16 && num_kv_heads == 8 && head_dim == 128,
                        "RhinoVLA QKV/KV optimization requires exact v3 18L/H1024/I3072/Q16/KV8/D128/TP8");
            const auto check_projection = [&](const at::Tensor& weight,
                                              int64_t rows, int64_t cols) {
                TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1 &&
                                weight.scalar_type() == (w8a16 ? at::kChar : at::kHalf) &&
                                weight.is_contiguous() &&
                                weight.size(0) == rows && weight.size(1) == cols &&
                                weight.nbytes() == static_cast<size_t>(rows * cols * (w8a16 ? 1 : DWIDTH)),
                            "RhinoVLA QKV/KV optimization requires contiguous uniform FP16/INT8 RPU projection [",
                            rows, ",", cols, "]");
            };
            for (int64_t i = 0; i < N; ++i) {
                check_projection(q_w_list[i], 2048, 1024);
                check_projection(k_w_list[i], 1024, 1024);
                check_projection(v_w_list[i], 1024, 1024);
                check_projection(o_w_list[i], 1024, 2048);
                check_projection(gate_list[i], 3072, 1024);
                check_projection(up_list[i], 3072, 1024);
                check_projection(down_list[i], 1024, 3072);
            }
        }

        if (w8a16) {
            const auto check_scale = [&](const at::Tensor& scale, int64_t outputs) {
                TORCH_CHECK(scale.defined() && scale.device() == q_w_list[0].device() &&
                                scale.scalar_type() == at::kHalf && scale.is_contiguous() &&
                                scale.dim() == 1 && scale.numel() == outputs,
                            "RhinoVLA W8A16 requires natural contiguous FP16 RPU scale [N=", outputs, "]");
                // Installation is cold. Flush the source before reading its
                // host mapping; no content reads or conversion occur in replay.
                rpu_ddr_flush_force_sized(scale.data_ptr(), scale.nbytes());
                const auto* values = scale.data_ptr<c10::Half>();
                for (int64_t channel = 0; channel < outputs; ++channel) {
                    const float value = static_cast<float>(values[channel]);
                    TORCH_CHECK(std::isfinite(value) && value > 0,
                                "RhinoVLA W8A16 scales must be finite and positive");
                }
            };
            const auto check_fp16 = [&](const at::Tensor& tensor,
                                         at::IntArrayRef shape) {
                TORCH_CHECK(tensor.device() == q_w_list[0].device() &&
                                tensor.scalar_type() == at::kHalf && tensor.is_contiguous() &&
                                tensor.sizes() == shape,
                            "RhinoVLA W8A16 keeps exact-shape AdaRMS and QK norms in FP16");
            };
            for (int64_t i = 0; i < N; ++i) {
                check_scale(q_w_scale_list[i], 2048);
                check_scale(k_w_scale_list[i], 1024);
                check_scale(v_w_scale_list[i], 1024);
                check_scale(o_w_scale_list[i], 1024);
                check_scale(gate_scale_list[i], 3072);
                check_scale(up_scale_list[i], 3072);
                check_scale(down_scale_list[i], 1024);
                if (full_w8a16) {
                    check_rhino_full_w8_projection(attn_dense_w_list[i], attn_dense_scale_list[i], 3072, 1024);
                    check_rhino_full_w8_projection(mlp_dense_w_list[i], mlp_dense_scale_list[i], 3072, 1024);
                    check_rhino_full_w8_projection(pair_dense_w_list[i], pair_dense_scale_list[i], 6144, 1024);
                } else {
                    check_fp16(attn_dense_w_list[i], {3072, 1024});
                    check_fp16(mlp_dense_w_list[i], {3072, 1024});
                    check_fp16(pair_dense_w_list[i], {6144, 1024});
                }
                check_fp16(attn_dense_b_list[i], {3072});
                check_fp16(mlp_dense_b_list[i], {3072});
                check_fp16(pair_dense_b_list[i], {6144});
                check_fp16(q_norm_list[i], {128});
                check_fp16(k_norm_list[i], {128});
            }
        }

        // Build every packed owner before changing model state. Col-swizzle is
        // [local_N/16, K/16, core, 16, 16], so concatenating the complete
        // swizzled Q/K/V byte ranges produces per-core [Q256 | K128 | V128].
        // No second swizzle and no floating-point arithmetic are involved.
        std::vector<at::Tensor> packed_qkv_weights(N);
        if (cold_config_.packed_qkv) {
            for (int64_t i = 0; i < N; ++i) {
                auto packed = at::empty({4096, 1024}, q_w_list[i].options());
                auto* destination = static_cast<uint8_t*>(packed.data_ptr());
                size_t offset = 0;
                for (const at::Tensor* source : {&q_w_list[i], &k_w_list[i], &v_w_list[i]}) {
                    rpu_ddr_flush_force_sized(source->data_ptr(), source->nbytes());
                    std::memcpy(destination + offset, source->data_ptr(), source->nbytes());
                    offset += source->nbytes();
                }
                rpu_ddr_flush_force_sized(destination, packed.nbytes());
                packed_qkv_weights[i] = std::move(packed);
            }
        }

        // Build the entire replacement owner before mutating live model state.
        std::vector<LayerWeights> next_layer_weights;
        next_layer_weights.reserve(N);
        for (int64_t i = 0; i < N; i++) {
            next_layer_weights.push_back({
                q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
                gate_list[i], up_list[i], down_list[i],
                attn_dense_w_list[i], attn_dense_b_list[i],
                mlp_dense_w_list[i],  mlp_dense_b_list[i],
                pair_dense_w_list[i], pair_dense_b_list[i],
                q_norm_list[i], k_norm_list[i],
                packed_qkv_weights[i],
                w8a16 ? q_w_scale_list[i] : at::Tensor(),
                w8a16 ? k_w_scale_list[i] : at::Tensor(),
                w8a16 ? v_w_scale_list[i] : at::Tensor(),
                w8a16 ? o_w_scale_list[i] : at::Tensor(),
                w8a16 ? gate_scale_list[i] : at::Tensor(),
                w8a16 ? up_scale_list[i] : at::Tensor(),
                w8a16 ? down_scale_list[i] : at::Tensor(),
                full_w8a16 ? attn_dense_scale_list[i] : at::Tensor(),
                full_w8a16 ? mlp_dense_scale_list[i] : at::Tensor(),
                full_w8a16 ? pair_dense_scale_list[i] : at::Tensor(),
            });
        }

        set_model_params(num_q_heads, num_kv_heads, head_dim,
                         hidden_size, intermediate_size);
        set_num_layers(N);
        eps_ = eps;
        local_q_heads_ = num_q_heads / attn_tp();
        local_kv_dim_ = num_kv_heads * head_dim / attn_tp();
        layer_weights_.swap(next_layer_weights);
        expert_w8a16_ = w8a16;
        const bool retire_full_derived = full_w8a16 || full_action_w8a16_;
        full_action_w8a16_ = full_w8a16;
        if (retire_full_derived) {
            gates_tanh_precomputed_ = false;
            full_io_scales_.clear();
            full_cold_weights_.clear();
            full_cold_scales_.clear();
            full_cold_biases_.clear();
            adarms_pair_table_ = at::Tensor();
            adarms_final_table_ = at::Tensor();
            denoise_adarms_tables_ready_ = false;
            denoise_loop_weights_ready_ = false;
            loop_mode_ = false;
            dt_pinned_ = false;
        }

        invalidate_model_state();  // D-503: last non-empty statement of set_weights
    }

    // ========================================================================
    // forward — cond + cos/sin are per-forward arguments
    //
    // Unlike AdaRMS which stores cos/sin in set_weights, RhinoVLA receives
    // cos/sin per-forward (M-RoPE pre-computed on Python side).
    // Three tensors kept alive via class members: cond_ref_, cos_ref_, sin_ref_.
    // ========================================================================
    at::Tensor forward(
        const at::Tensor& hidden_states,
        const at::Tensor& cond,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const at::Tensor& cos,
        const at::Tensor& sin,
        const std::optional<at::Tensor>& attention_mask,
        int64_t position,
        bool is_causal,
        at::IntArrayRef planned_stage_descriptor = {})
    {
        TORCH_CHECK(num_layers() > 0,
                    "RhinoVLAModel::forward called before set_weights");
        TORCH_CHECK(!loop_mode_,
                    "RhinoVLAModel::forward cannot replace an installed "
                    "denoise loop with step execution; rebuild the owner");
        TORCH_CHECK(hidden_states.device().type() == at::kPrivateUse1,
                    "RhinoVLAModel::forward: hidden_states must be on RPU device");
        TORCH_CHECK(hidden_states.is_contiguous(),
                    "RhinoVLAModel::forward: hidden_states must be contiguous "
                    "(downstream kernels use raw data_ptr())");
        TORCH_CHECK(position >= 0,
                    "RhinoVLAModel::forward: position must be non-negative, got ", position);

        // cond validation (mirrors AdaRMS:224-240 — cross-boundary tensors
        // consumed as raw c10::Half* in the graph MUST be validated at the
        // forward entry. Codex-review finding #3 — previous code only
        // checked shape; dtype/device/contiguity slipped through.)
        TORCH_CHECK(cond.defined() && cond.dim() == 1 &&
                    cond.size(0) == hidden_size(),
                    "RhinoVLAModel::forward: cond must be 1D [hidden_size=",
                    hidden_size(), "], got dim=", (cond.defined() ? cond.dim() : -1),
                    " size=", (cond.defined() ? cond.size(0) : -1));
        TORCH_CHECK(cond.device().type() == at::kPrivateUse1,
                    "RhinoVLAModel::forward: cond must be on RPU device");
        TORCH_CHECK(cond.scalar_type() == at::kHalf,
                    "RhinoVLAModel::forward: cond must be fp16");
        TORCH_CHECK(cond.is_contiguous(),
                    "RhinoVLAModel::forward: cond must be contiguous "
                    "(DMA source pointer assumes row-major layout)");

        // cos/sin validation — consumed as raw c10::Half* in rotary kernels.
        TORCH_CHECK(cos.defined() && sin.defined(),
                    "RhinoVLAModel::forward: cos/sin must be defined");
        TORCH_CHECK(cos.device().type() == at::kPrivateUse1 &&
                    sin.device().type() == at::kPrivateUse1,
                    "RhinoVLAModel::forward: cos/sin must be on RPU device");
        TORCH_CHECK(cos.scalar_type() == at::kHalf &&
                    sin.scalar_type() == at::kHalf,
                    "RhinoVLAModel::forward: cos/sin must be fp16");
        TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(),
                    "RhinoVLAModel::forward: cos/sin must be contiguous "
                    "(rotary kernel assumes row-major layout)");
        TORCH_CHECK(cos.sizes() == sin.sizes(),
                    "RhinoVLAModel::forward: cos/sin shapes must match, got cos=",
                    cos.sizes(), " sin=", sin.sizes());
        // Round-2 codex-review + Round-3 shape-contract fix: the rotary
        // kernel reads cos_ref_/sin_ref_ as raw c10::Half* starting at
        // `cos_sin_start = rope_position_ + chunk.offset` for `seq_len` rows
        // × head_dim/2
        // columns (rpu_rhino_vla_model.cpp:455, :651-:656; rpu_rope.cpp:370
        // documents the DDR table contract as [max_seq, head_dim/2]).
        //
        // Round-4 tightening: ONLY accept head_dim/2. The round-3 relaxation
        // ("accept either half- or full-dim") was incorrect — the kernel
        // consumes raw data_ptr with stride = head_dim/2 fp16 elements per
        // row. A full-dim caller would still reach kernel dispatch (validation
        // passes) but misindex positions (rhino_vla_fused_converter.py:299
        // comments note full-dim tables misindex). The canonical fused
        // producer always returns [suffix_len, head_dim/2]
        // (rhino_vla_fused_converter.py:305-:323). Any caller that wants to
        // pass full-dim must slice to half-dim before the call.
        TORCH_CHECK(cos.dim() == 2,
                    "RhinoVLAModel::forward: cos must be 2D [max_pos, head_dim/2], got dim=",
                    cos.dim());
        TORCH_CHECK(sin.dim() == 2,
                    "RhinoVLAModel::forward: sin must be 2D [max_pos, head_dim/2], got dim=",
                    sin.dim());
        TORCH_CHECK(cos.size(1) == head_dim() / 2,
                    "RhinoVLAModel::forward: cos last-dim must equal head_dim/2=",
                    head_dim() / 2, " (kernel DDR contract per rpu_rope.cpp:370); got ",
                    cos.size(1),
                    ". If the caller has full-dim [max_pos, head_dim] tables, "
                    "slice to half-dim before .forward(): "
                    "cos_half = cos[..., :head_dim//2].contiguous().");
        TORCH_CHECK(rope_position_ >= 0,
                    "RhinoVLAModel::forward: logical RoPE position was not set");
        TORCH_CHECK(cos.size(0) >= rope_position_ + hidden_states.size(1),
                    "RhinoVLAModel::forward: cos row count (", cos.size(0),
                    ") must cover logical position + seq_len (", rope_position_,
                    "+", hidden_states.size(1), ")");

        // Reset per-forward cond state: a new denoise step is a new cond.
        // cond_ref_/cos_ref_/sin_ref_ keep the tensors alive for the
        // synchronous forward (run_all_layers batch end executes before
        // return); cond DDR pointer cursor-patched per REPLAY via mutable DMA.
        cond_loaded_this_forward_ = false;
        cond_ref_ = cond;
        cos_ref_ = cos;
        sin_ref_ = sin;
        // cond_src_base_ feeds rpu_launch_ddr_scatter_spm_dma_mutable in the
        // cond scatter site below; REPLAY end() reads this live address. Per-
        // forward flush mirrors fused_model_base's hidden_in_src_base_ pattern
        // (caller may have CPU-dirty cache lines from an upstream op).
        cond_src_base_ = ::rhino_lkn::RpuGetDevAddr(cond.data_ptr());
        rpu_ddr_flush_force(cond.data_ptr<c10::Half>());

        prepare_sdpa_mask_cached(attention_mask, is_causal && hidden_states.size(1) > 1,
                                 hidden_states.size(1), position + hidden_states.size(1));
        return run_all_layers(hidden_states, k_caches, v_caches,
                              attention_mask, position, is_causal,
                              /*planned_chunk_size=*/0,
                              planned_stage_descriptor);
    }

    at::Tensor denoise_loop_forward(
        const at::Tensor& x0_rpu,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const at::Tensor& cond_all,
        const at::Tensor& attention_mask,
        const at::Tensor& state_rpu,
        const at::Tensor& state_mask_rpu,
        const at::Tensor& action_mask_rpu,
        at::Tensor& x_out,
        double dt,
        int64_t prefix_len,
        int64_t num_steps,
        at::IntArrayRef planned_stage_descriptor = {}) {
        prepare_denoise_loop_schedule(num_steps, dt);
        TORCH_CHECK(x0_rpu.dim() == 3 && x0_rpu.size(0) == 1 &&
                    x0_rpu.size(1) == action_horizon_ &&
                    x0_rpu.size(2) == action_dim_pad_ &&
                    x0_rpu.scalar_type() == at::kHalf &&
                    x0_rpu.device().type() == at::kPrivateUse1 &&
                    x0_rpu.is_contiguous(),
                    "RhinoVLA denoise loop x0 must be [1,", action_horizon_,
                    ",", action_dim_pad_, "] contiguous fp16 RPU");
        TORCH_CHECK(action_mask_rpu.sizes() == x0_rpu.sizes() &&
                    action_mask_rpu.scalar_type() == at::kHalf &&
                    action_mask_rpu.device().type() == at::kPrivateUse1 &&
                    action_mask_rpu.is_contiguous(),
                    "RhinoVLA denoise loop action_mask must match padded x0");
        TORCH_CHECK(x_out.sizes() == x0_rpu.sizes() &&
                    x_out.scalar_type() == at::kHalf &&
                    x_out.device().type() == at::kPrivateUse1 &&
                    x_out.is_contiguous(),
                    "RhinoVLA denoise loop x_out must match padded x0");
        TORCH_CHECK(state_rpu.dim() == 2 && state_rpu.size(0) == 1 &&
                    state_rpu.size(1) == state_dim_pad_ &&
                    state_rpu.scalar_type() == at::kHalf &&
                    state_rpu.device().type() == at::kPrivateUse1 &&
                    state_rpu.is_contiguous(),
                    "RhinoVLA denoise loop state must be [1,", state_dim_pad_,
                    "] contiguous fp16 RPU");
        TORCH_CHECK(state_mask_rpu.sizes() == state_rpu.sizes() &&
                    state_mask_rpu.scalar_type() == at::kHalf &&
                    state_mask_rpu.device().type() == at::kPrivateUse1 &&
                    state_mask_rpu.is_contiguous(),
                    "RhinoVLA denoise loop state_mask must match padded state");
        TORCH_CHECK(cond_all.dim() == 2 && cond_all.size(0) == num_steps &&
                    cond_all.size(1) == hidden_size() &&
                    cond_all.scalar_type() == at::kHalf &&
                    cond_all.device().type() == at::kPrivateUse1 &&
                    cond_all.is_contiguous(),
                    "RhinoVLA denoise loop cond_all must be [num_steps, hidden]");
        if (cold_config_.precompute_adarms) {
            TORCH_CHECK(denoise_adarms_tables_ready_,
                        "RPU_RHINOVLA_PRECOMPUTE_ADARMS=1 but AdaRMS tables were not bound");
            TORCH_CHECK(adarms_pair_table_.size(0) >= num_steps &&
                        adarms_final_table_.size(0) >= num_steps,
                        "RhinoVLA AdaRMS tables do not cover num_steps=", num_steps,
                        " pair_steps=", adarms_pair_table_.size(0),
                        " final_steps=", adarms_final_table_.size(0));
            adarms_pair_table_src_base_ =
                ::rhino_lkn::RpuGetDevAddr(adarms_pair_table_.data_ptr());
            adarms_final_table_src_base_ =
                ::rhino_lkn::RpuGetDevAddr(adarms_final_table_.data_ptr());
            rpu_ddr_flush_force(adarms_pair_table_.data_ptr<c10::Half>());
            rpu_ddr_flush_force(adarms_final_table_.data_ptr<c10::Half>());
        }
        if (cold_config_.precompute_time_proj) {
            TORCH_CHECK(denoise_time_proj_table_ready_,
                        "RPU_RHINOVLA_PRECOMPUTE_TIME_PROJ=1 but time-proj table was not bound");
            TORCH_CHECK(time_proj_table_.size(0) >= num_steps,
                        "RhinoVLA time-proj table does not cover num_steps=",
                        num_steps, " table_steps=", time_proj_table_.size(0));
            time_proj_table_src_base_ =
                ::rhino_lkn::RpuGetDevAddr(time_proj_table_.data_ptr());
            rpu_ddr_flush_force(time_proj_table_.data_ptr<c10::Half>());
        }
        TORCH_CHECK(prefix_len >= 0, "RhinoVLA denoise loop prefix_len must be >= 0");
        if (expert_w8a16_) validate_expert_transfer(suffix_len_);
        if (cold_config_.aligned_kv) {
            validate_expert_transfer(suffix_len_);
            TORCH_CHECK(prefix_len % 16 == 0 && attention_mask.defined(),
                        "RhinoVLA aligned KV requires an aligned physical prefix and explicit mask");
        }
        TORCH_CHECK(rope_position_ >= 0 &&
                        cos_loop_.size(0) >= rope_position_ + suffix_len_,
                    "RhinoVLA denoise loop RoPE table does not cover logical "
                    "position ", rope_position_, " + suffix_len ", suffix_len_);
        if (!k_caches.empty()) {
            const at::Tensor& k0 = k_caches[0];
            const int64_t cache_max_seq = k0.size(1) * k0.size(5);
            const int64_t kv_write_rows = cold_config_.aligned_kv ? 32 : suffix_len_;
            TORCH_CHECK(prefix_len + kv_write_rows <= cache_max_seq,
                        "RhinoVLA denoise loop prefix_len(", prefix_len,
                        ") + KV write rows(", kv_write_rows,
                        ") exceeds k_cache max_seq_len=", cache_max_seq);
        }
        cond_ref_ = cond_all;
        cos_ref_ = cos_loop_;
        sin_ref_ = sin_loop_;
        x0_ref_ = x0_rpu;
        state_ref_ = state_rpu;
        state_mask_ref_ = state_mask_rpu;
        action_mask_ref_ = action_mask_rpu;
        x_out_ref_ = x_out;
        rpu_ddr_flush_force(x0_rpu.data_ptr<c10::Half>());
        rpu_ddr_flush_force(cond_all.data_ptr<c10::Half>());
        rpu_ddr_flush_force(state_rpu.data_ptr<c10::Half>());
        rpu_ddr_flush_force(state_mask_rpu.data_ptr<c10::Half>());
        rpu_ddr_flush_force(action_mask_rpu.data_ptr<c10::Half>());
        rpu_ddr_flush_force(x_out.data_ptr<c10::Half>());
        x0_src_base_ = ::rhino_lkn::RpuGetDevAddr(x0_rpu.data_ptr());
        cond_all_src_base_ = ::rhino_lkn::RpuGetDevAddr(cond_all.data_ptr());
        state_src_base_ = ::rhino_lkn::RpuGetDevAddr(state_rpu.data_ptr());
        state_mask_src_base_ = ::rhino_lkn::RpuGetDevAddr(state_mask_rpu.data_ptr());
        action_mask_src_base_ = ::rhino_lkn::RpuGetDevAddr(action_mask_rpu.data_ptr());
        x_out_dst_base_ = ::rhino_lkn::RpuGetDevAddr(x_out.data_ptr());

        prepare_sdpa_mask_cached(std::optional<at::Tensor>(attention_mask), false,
                                 suffix_len_, prefix_len + suffix_len_);
        (void) run_all_layers(action_emb_stage_, k_caches, v_caches,
                              std::optional<at::Tensor>(attention_mask),
                              prefix_len, /*is_causal=*/false,
                              /*planned_chunk_size=*/0,
                              planned_stage_descriptor);
        return x_out;
    }

    // ========================================================================
    // debug_flush_dumps — materialize per-core SPM reads after forward.
    // Called by Python via rhino_vla_debug_flush_dumps op after a debug-export
    // forward finishes, to populate g_debug_tensors.
    // ========================================================================
    void debug_flush_dumps() {
        for (const auto& req : debug_dump_requests_) {
            at::Tensor t = at::empty({req.num_cores, req.elems_per_core},
                at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
            c10::Half* dst = t.data_ptr<c10::Half>();
            for (int c = 0; c < req.num_cores; ++c) {
                const void* src = SPM_ALLOC.cpu_ptr(c, req.buf_off);
                std::memcpy(dst + c * req.elems_per_core, src,
                            req.elems_per_core * sizeof(c10::Half));
            }
            g_debug_tensors[req.name] = t;
        }
        debug_dump_requests_.clear();
    }

protected:
    std::vector<int64_t> kvinsert_cost_weight_identity() const override {
        if (layer_weights_.empty()) return {};
        int64_t eps_bits = 0;
        static_assert(sizeof(eps_bits) == sizeof(eps_));
        std::memcpy(&eps_bits, &eps_, sizeof(eps_bits));
        std::vector<int64_t> identity{
            1, static_cast<int64_t>(layer_weights_.size()), eps_bits,
            cold_config_.denoise_static_context_cache, cold_config_.fused_adarms,
            cold_config_.skip_adarms_gemv, cold_config_.precompute_adarms,
            cold_config_.precompute_time_proj, cold_config_.fold_action_time_in,
            cold_config_.gated_no_sub, cold_config_.fused_silu_mul,
            fast_replay_skip_layer_loop_, loop_mode_, denoise_loop_weights_ready_,
            num_steps_, static_cast<int64_t>(dt_.x), action_dim_, action_dim_pad_,
            state_dim_, state_dim_pad_, action_horizon_, suffix_len_,
            denoise_adarms_tables_ready_, denoise_time_proj_table_ready_,
            direct_action_input_, cold_config_.expert_fusions,
            cold_config_.adarms_resident, cold_config_.packed_qkv,
            cold_config_.aligned_kv, expert_w8a16_, full_action_w8a16_, gates_tanh_precomputed_,
            high_precision_, high_precision_ ? 1 : 0};
        if (high_precision_) {
            identity.push_back(high_precision_fusions_);
            identity.push_back(vector_k_norm_);
            identity.push_back(vector_q_norm_);
            identity.push_back(partial_rope_);
        }
        for (const auto& layer : layer_weights_) {
            for (const auto* tensor : {
                    &layer.q_w, &layer.k_w, &layer.v_w, &layer.o_w,
                    &layer.gate_proj_w, &layer.up_proj_w, &layer.down_proj_w,
                    &layer.attn_dense_w, &layer.attn_dense_b,
                    &layer.mlp_dense_w, &layer.mlp_dense_b,
                    &layer.pair_dense_w, &layer.pair_dense_b,
                    &layer.q_norm_w, &layer.k_norm_w, &layer.packed_qkv_w,
                    &layer.q_ws, &layer.k_ws, &layer.v_ws, &layer.o_ws,
                    &layer.gate_ws, &layer.up_ws, &layer.down_ws,
                    &layer.attn_dense_ws, &layer.mlp_dense_ws, &layer.pair_dense_ws}) {
                append_kvinsert_cost_tensor_identity(identity, *tensor);
            }
        }
        for (const auto* tensor : {
                &action_in_w_, &action_in_b_, &action_time_in_w_, &action_time_in_b_,
                &time_in_action_w_, &time_in_time_w_, &time_in_b_,
                &time_out_w_, &time_out_b_, &state_w_, &state_b_,
                &state_mask_w_, &state_mask_b_, &action_mask_w_, &action_mask_b_,
                &final_norm_w_, &final_norm_b_, &action_out_w_, &action_out_b_,
                &cos_loop_, &sin_loop_, &adarms_pair_table_, &adarms_final_table_,
                &time_proj_table_}) {
            append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
        for (const auto* owners : {&full_io_scales_, &full_cold_weights_, &full_cold_scales_, &full_cold_biases_}) {
            identity.push_back(static_cast<int64_t>(owners->size()));
            for (const auto& tensor : *owners) append_kvinsert_cost_tensor_identity(identity, tensor);
        }
        return identity;
    }

    // ========================================================================
    // static_config — graph-naive flat-phase subclass; no legacy graph slot.
    //
    // RhinoVLA is a flat-phase, SEQUENTIAL subclass — like AdaRMS. All four
    // D-501 pointer-to-member slots (preload_fn / kv_first_fn /
    // kv_first_chunk_plan_fn / post_fn) stay nullptr by default. No persistent
    // weights, no KV_FIRST, no post-graph.
    // ========================================================================
    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        cfg.num_layers       = num_layers();
        // Pin to a single fused graph regardless of the global
        // `g_cross_layer_batch_size` (which is shared with Qwen3 text
        // decode where callers sometimes set it to 7 / 12 / 36). For
        // RhinoVLA action expert we always want all layers in one graph
        // replay per DDIM step — setting it equal to num_layers() makes
        // `compute_num_groups` in FusedModelBase return 1 unconditionally.
        cfg.cross_layer_batch_size = num_layers();
        cfg.fast_replay_skip_layer_loop = fast_replay_skip_layer_loop_;
        if (loop_mode_) {
            cfg.pre_layers_fn = reinterpret_cast<void (FusedModelBase::*)()>(
                &RhinoVLAModel::emit_loop_pre_layers_body);
            cfg.post_layers_fn = reinterpret_cast<void (FusedModelBase::*)()>(
                &RhinoVLAModel::emit_loop_post_layers_body);
            cfg.body_iterations = num_steps_;
            cfg.fast_replay_skip_full_body = true;
        }
        return cfg;
    }

    // ========================================================================
    // dynamic_config — always SEQUENTIAL
    // ========================================================================
    ModelDynamicConfig dynamic_config(const ChunkPlan& /*plan*/) override {
        ModelDynamicConfig cfg;
        cfg.chunk_mode     = ChunkMode::SEQUENTIAL;
        cfg.inter_layer_io = InterLayerIO::AUTO;
        // Prefix-history attention is backed by the DDR KV cache. Do not
        // advertise the unrelated SPM-KV/MHA route.
        cfg.attention_policy = AttentionExecutionPolicy::DDR_KV;
        return cfg;
    }

    bool subclass_chunk_size_valid(
        int64_t chunk_size, int64_t seq_len,
        int64_t /*position*/) const override {
        // Prefix-history action attention is non-causal over the complete
        // suffix. Splitting it would change suffix-to-suffix visibility.
        return chunk_size >= seq_len;
    }

    FmbPhysicalExecutionManifest physical_manifest_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t physical_len, int64_t logical_len,
        int64_t position) const override {
        TORCH_CHECK(
            plan.compute.chunks.size() == 1,
            "RhinoVLA COMPLETE descriptor requires one action suffix chunk");
        const ChunkInfo& chunk = plan.compute.chunks.front();
        validate_expert_transfer(chunk.len);
        if (cold_config_.aligned_kv) {
            TORCH_CHECK(physical_len == 31 && logical_len == 31 &&
                            chunk.offset == 0 && position % 16 == 0 &&
                            layout.use_attn_mask && !layout.is_causal,
                        "RhinoVLA aligned KV requires one real M31 suffix, aligned physical prefix and non-causal mask");
        }
        const int64_t invocation = chunk.idx;
        const bool fused_adarms = cold_config_.fused_adarms;
        const bool skip_adarms = cold_config_.skip_adarms_gemv;
        const bool precomputed_adarms =
            loop_mode_ && cold_config_.precompute_adarms;
        const bool precomputed_time =
            loop_mode_ && cold_config_.precompute_time_proj;
        const bool fold_action_time =
            loop_mode_ && cold_config_.fold_action_time_in;
        const bool fused_silu_mul = cold_config_.fused_silu_mul && !high_precision_;
        const bool gated_no_sub = cold_config_.gated_no_sub;
        const bool custom_mlp = fused_silu_mul || gated_no_sub || high_precision_;

        FmbPhysicalExecutionManifest manifest;
        manifest.state = FmbPhysicalManifestState::COMPLETE;
        manifest.logical_length = logical_len;
        manifest.physical_length = physical_len;
        manifest.execution_padding_rows = physical_len - logical_len;
        manifest.kv_logical_length = position + physical_len;
        manifest.kv_insert_physical_rows = cold_config_.aligned_kv ? 32 : chunk.len;
        manifest.graph_lifecycle = FmbGraphLifecycle::COMPOSITE_CHILD;
        manifest.linear_accumulation = FmbLinearAccumulationPolicy::ACC16;
        const auto append = [&](FmbRouteFamily family, int64_t site_id,
                                int64_t selector, int64_t flags = 0,
                                std::vector<int64_t> arguments = {}) {
            manifest.routes.push_back({
                site_id, family, selector, flags,
                std::move(arguments), invocation});
        };
        const auto append_linear = [&](int64_t site_id) {
            append(FmbRouteFamily::LINEAR, site_id,
                   static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE));
        };
        const auto append_expert_linear = [&](int64_t site_id, int64_t n,
                                               int64_t k, int partition) {
            append(FmbRouteFamily::LINEAR, site_id,
                   static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                   expert_w8a16_ ? RHINO_LINEAR_W8A16_PER_CHANNEL : 0,
                   expert_projection_route_arguments(chunk.len, n, k, partition));
        };
        const auto append_io_linear = [&](int64_t site_id, int64_t rows, int64_t n, int64_t k) {
            append(FmbRouteFamily::LINEAR, site_id,
                   static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                   full_action_w8a16_ ? RHINO_LINEAR_W8A16_PER_CHANNEL : 0,
                   full_io_route_arguments(rows, n, k));
        };
        const auto append_dma = [&](int64_t site_id,
                                    RhinoMutableDmaRoute route) {
            append(FmbRouteFamily::MUTABLE_DMA, site_id,
                   static_cast<int64_t>(route));
        };

        if (cold_config_.expert_fusions) {
            append(FmbRouteFamily::GRAPH_SCHEDULE, RHINO_EXPERT_FUSIONS_SITE,
                   1, 0, {chunk.len, hidden_size(), num_layers(), num_steps_});
        }
        if (expert_w8a16_) {
            append(FmbRouteFamily::GRAPH_SCHEDULE, RHINO_EXPERT_W8A16_SITE,
                   1, 0, expert_w8a16_route_arguments(chunk.len));
        }
        if (full_action_w8a16_) {
            append(FmbRouteFamily::GRAPH_SCHEDULE, RHINO_FULL_W8A16_SITE,
                   1, gates_tanh_precomputed_ ? 1 : 0, {num_layers(), num_steps_, 37, 5, 8, 16});
        }
        if (high_precision_) {
            append(FmbRouteFamily::GRAPH_SCHEDULE, RHINO_HIGH_PRECISION_SITE,
                   1, high_precision_route_flags(), high_precision_route_arguments());
        }
        if (cold_config_.adarms_resident) {
            append(FmbRouteFamily::GRAPH_SCHEDULE, RHINO_ADARMS_RESIDENT_SITE,
                   1, 0, {num_steps_, num_layers(), hidden_size(),
                          num_steps_ * (num_layers() * 6 + 3) * hidden_size() * DWIDTH});
            append(FmbRouteFamily::MUTABLE_DMA, RHINO_RESIDENT_PAIR_DMA_SITE,
                   static_cast<int64_t>(RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                   0, {num_steps_ * num_layers() * 6 * hidden_size()});
            append(FmbRouteFamily::MUTABLE_DMA, RHINO_RESIDENT_FINAL_DMA_SITE,
                   static_cast<int64_t>(RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                   0, {num_steps_ * 3 * hidden_size()});
        }

        if (layout.use_attn_mask && !(layout.is_causal && physical_len > 1)) {
            auto declarations = const_cast<RhinoVLAModel*>(this)
                ->baseline_buffer_declarations(layout);
            const auto mask_plan = detail::plan_forward_spm_residency(
                declarations, {"sdpa_mask"});
            append(FmbRouteFamily::GRAPH_SCHEDULE, RHINO_MASK_SPM_SCHEDULE_SITE,
                   static_cast<int64_t>(mask_plan.retained
                       ? RhinoMaskSpmSchedule::FIRST_BODY_FIRST_LAYER
                       : RhinoMaskSpmSchedule::EVERY_LAYER), 0,
                   {physical_len, Align(position + physical_len, int64_t{16}),
                    attn_tp(), NUM_CORES, loop_mode_ ? num_steps_ : 1, num_layers(),
                    mask_plan.original_peak_bytes, mask_plan.retained_peak_bytes,
                    position + physical_len});
        }

        if (!loop_mode_) {
            append(FmbRouteFamily::MUTABLE_DMA, RHINO_COND_DMA_SITE,
                   static_cast<int64_t>(
                       RhinoMutableDmaRoute::DDR_SCATTER_TO_SPM),
                   /*flags=*/0, {loop_mode_ ? 1 : 0});
        }
        if (precomputed_adarms) {
            if (!cold_config_.adarms_resident) {
                append_dma(RHINO_ADARMS_TABLE_DMA_SITE,
                           RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM);
            }
        } else if (!skip_adarms && fused_adarms) {
            append_linear(RHINO_ADARMS_PAIR_LINEAR_SITE);
            append(FmbRouteFamily::ALL_REDUCE,
                   RHINO_ADARMS_PAIR_ALL_REDUCE_SITE,
                   fmb_ring_all_reduce_route_selector(
                       /*rows=*/1, /*cols=*/6 * hidden_size()));
        }
        const bool uses_regular_adarms =
            (!skip_adarms && !precomputed_adarms && !fused_adarms) ||
            (loop_mode_ && !cold_config_.precompute_adarms);
        if (uses_regular_adarms) {
            append_linear(RHINO_ADARMS_LINEAR_SITE);
            append(FmbRouteFamily::ALL_REDUCE,
                   RHINO_ADARMS_ALL_REDUCE_SITE,
                   fmb_ring_all_reduce_route_selector(
                       /*rows=*/1, /*cols=*/3 * hidden_size()));
        }
        if (cold_config_.packed_qkv) {
            append(FmbRouteFamily::LINEAR, RHINO_PACKED_QKV_LINEAR_SITE,
                   static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                   0, {chunk.len, 4096, 1024, 1, NUM_CORES});
        } else {
            append_expert_linear(RHINO_Q_LINEAR_SITE, num_q_heads() * head_dim(), hidden_size(), 1);
            append_expert_linear(RHINO_K_LINEAR_SITE, num_kv_heads() * head_dim(), hidden_size(), 1);
            append_expert_linear(RHINO_V_LINEAR_SITE, num_kv_heads() * head_dim(), hidden_size(), 1);
        }
        append(FmbRouteFamily::ROPE, RHINO_Q_ROPE_SITE,
               rope_route_selector(),
               /*flags=*/0, {rope_position_ + chunk.offset});
        append(FmbRouteFamily::ROPE, RHINO_K_ROPE_SITE,
               rope_route_selector(),
               /*flags=*/0, {rope_position_ + chunk.offset});
        const KvInsertSegmentPlan kv_plan =
            resolve_kvinsert_plan_auto(
                RHINO_KV_INSERT_SITE, manifest.graph_lifecycle,
                position + chunk.offset, chunk.len, manifest.kv_insert_physical_rows,
                attn_tp(), num_kv_heads(), head_dim(),
                RHINO_KV_CAPABILITIES |
                    (cold_config_.aligned_kv ? KV_INSERT_CAP_PAD16 : 0u));
        const KvInsertRouteArguments kv_arguments =
            rpu_kvinsert_route_arguments(
                kv_plan, attn_tp(), num_kv_heads(), head_dim());
        append(FmbRouteFamily::KV_INSERT, RHINO_KV_INSERT_SITE,
               static_cast<int64_t>(kv_plan.route()),
               RHINO_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED,
               {kv_arguments.begin(), kv_arguments.end()});
        append(FmbRouteFamily::ATTENTION, RHINO_ATTENTION_SITE,
               static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV));
        append(FmbRouteFamily::ALL_REDUCE,
               RHINO_PREPARE_ALL_REDUCE_SITE,
               static_cast<int64_t>(
                   RhinoAllReduceRoute::PREPARE_RING_INPUT));
        append_expert_linear(RHINO_O_LINEAR_SITE, hidden_size(), num_q_heads() * head_dim(), 0);
        append(FmbRouteFamily::ALL_REDUCE,
               RHINO_ATTN_ALL_REDUCE_SITE,
               fmb_ring_all_reduce_route_selector(chunk.len, hidden_size()));
        append(FmbRouteFamily::GRAPH_SCHEDULE,
               RHINO_GATED_RESIDUAL_SCHEDULE_SITE,
               rhino_gated_residual_selector(gated_no_sub),
               /*flags=*/0, {gated_no_sub ? 1 : 0});
        if (custom_mlp) {
            append_expert_linear(RHINO_MLP_GATE_LINEAR_SITE, intermediate_size(), hidden_size(), 1);
            append_expert_linear(fused_silu_mul
                              ? RHINO_MLP_UP_FUSED_LINEAR_SITE
                              : RHINO_MLP_UP_PLAIN_LINEAR_SITE,
                                 intermediate_size(), hidden_size(), 1);
            append_expert_linear(RHINO_MLP_DOWN_LINEAR_SITE, hidden_size(), intermediate_size(), 0);
            append(FmbRouteFamily::ALL_REDUCE,
                   RHINO_MLP_ALL_REDUCE_SITE,
                   fmb_ring_all_reduce_route_selector(
                       chunk.len, hidden_size()));
        }
        if (loop_mode_) {
            append(FmbRouteFamily::GRAPH_SCHEDULE,
                   RHINO_DENOISE_LOOP_SCHEDULE_SITE,
                   /*selector=*/1, /*flags=*/0,
                   {num_steps_, static_cast<int64_t>(dt_.x)});
            append_dma(RHINO_PRE_X_DMA_SITE,
                       RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM);
            append_dma(RHINO_PRE_ACTION_MASK_DMA_SITE,
                       RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM);
            append_dma(RHINO_PRE_STATE_DMA_SITE,
                       RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM);
            append_dma(RHINO_PRE_STATE_MASK_DMA_SITE,
                       RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM);
            if (!direct_action_input_ && !precomputed_time) {
                append_dma(RHINO_PRE_TIME_DMA_SITE,
                           RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM);
            }
            if (!precomputed_adarms) {
                append_dma(RHINO_PRE_COND_DMA_SITE,
                           RhinoMutableDmaRoute::DDR_SCATTER_TO_SPM);
            }
            if (direct_action_input_) {
                append_io_linear(RHINO_PRE_ACTION_LINEAR_SITE, action_horizon_, hidden_size(), action_dim_pad_);
            } else {
                if (fold_action_time) {
                    append_linear(RHINO_PRE_FOLDED_ACTION_LINEAR_SITE);
                } else {
                    append_io_linear(RHINO_PRE_ACTION_LINEAR_SITE, action_horizon_, hidden_size(), action_dim_pad_);
                    append_linear(RHINO_PRE_TIME_ACTION_LINEAR_SITE);
                }
                if (precomputed_time) {
                    append_dma(RHINO_PRE_TIME_TABLE_DMA_SITE,
                               RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM);
                } else {
                    append_linear(RHINO_PRE_TIME_LINEAR_SITE);
                }
                append_linear(RHINO_PRE_TIME_OUT_LINEAR_SITE);
            }
            append_io_linear(RHINO_PRE_ACTION_MASK_LINEAR_SITE, action_horizon_, hidden_size(), action_dim_pad_);
            append_io_linear(RHINO_PRE_STATE_LINEAR_SITE, 1, hidden_size(), state_dim_pad_);
            append_io_linear(RHINO_PRE_STATE_MASK_LINEAR_SITE, 1, hidden_size(), state_dim_pad_);
            if (precomputed_adarms && !cold_config_.adarms_resident) {
                append(FmbRouteFamily::MUTABLE_DMA,
                       RHINO_POST_ADARMS_DMA_SITE,
                       static_cast<int64_t>(
                           RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                       /*flags=*/0,
                       {cold_config_.precompute_adarms ? 1 : 0});
            }
            append_io_linear(RHINO_POST_ACTION_LINEAR_SITE, action_horizon_, action_dim_pad_, hidden_size());
            append_dma(RHINO_POST_OUTPUT_DMA_SITE,
                       RhinoMutableDmaRoute::SPM_COPY_TO_DDR);
        }
        uint32_t shared_routes = FMB_SHARED_LAYER_INPUT_DMA;
        if (!custom_mlp) {
            shared_routes |=
                FMB_SHARED_MLP_AUTO_TILE | FMB_SHARED_MLP_RING_REDUCE;
        }
        append_fmb_shared_runtime_routes(
            manifest, plan, hidden_size(), shared_routes);
        std::sort(
            manifest.routes.begin(), manifest.routes.end(),
            [](const FmbRouteManifestEntry& lhs,
               const FmbRouteManifestEntry& rhs) {
                return std::make_tuple(
                           static_cast<int64_t>(lhs.family), lhs.site_id,
                           lhs.invocation) <
                    std::make_tuple(
                           static_cast<int64_t>(rhs.family), rhs.site_id,
                           rhs.invocation);
            });
        return manifest;
    }

    FmbPhysicalManifestForwardCapability
    physical_manifest_forward_capability(
        const FmbPhysicalExecutionManifest& /*manifest*/) const override {
        return {true, FmbGraphLifecycle::COMPOSITE_CHILD};
    }

    // ========================================================================
    // declare_buffers — flat phases, ALL LayerWide
    //
    // Same as AdaRMS but adds q_norm_w and k_norm_w buffers for QK norm plus
    // debug-snapshot buffers allocated unconditionally so the graph layout
    // stays stable across debug/non-debug invocations.
    //
    // SPM budget (hidden=1024, intermediate=3072, heads=16, kv=8, head_dim=128):
    //   attn_tp = min(8, 8) = 8
    //   local_q_heads = 16/8 = 2
    //   local_kv_dim = 8*128/8 = 128
    //   residual ~ 2KB, q ~ 512B, kv ~ 256B, oproj ~ 2KB, mlp ~ 768B
    //   cond ~ 2KB, gemv ~ 6KB, q/k_norm_w ~ 256B each
    //   Total per core ~ 15KB << 8109 KB SPM budget
    // ========================================================================
    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override {
        auto declarations = baseline_buffer_declarations(ctx);
        if (ctx.use_attn_mask && !(ctx.is_causal && ctx.chunk_size > 1)) {
            detail::plan_forward_spm_residency(declarations, {"sdpa_mask"});
        }
        return declarations;
    }

    std::vector<BufferDecl> baseline_buffer_declarations(const LayoutContext& ctx) {
        int64_t cs = ctx.chunk_size;
        int64_t h  = hidden_size();
        int64_t nq = num_q_heads();
        int64_t nkv = num_kv_heads();
        int64_t hd = head_dim();
        int64_t is_ = intermediate_size();

        int64_t local_q  = nq / attn_tp();
        int64_t local_kv = nkv * hd / attn_tp();
        auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };

        int64_t res   = A(cs * h * DWIDTH);
        const int64_t logical_q_bytes = A(cs * local_q * hd * DWIDTH);
        const int64_t q_rows = vector_q_norm_ ? Align(cs * local_q, int64_t{32}) : cs * local_q;
        int64_t q     = A(q_rows * hd * DWIDTH);
        const int64_t kv_rows = cold_config_.aligned_kv ? Align(cs, int64_t{16}) : cs;
        int64_t kv    = A(kv_rows * local_kv * DWIDTH);
        int64_t out   = q;
        int64_t oproj = A(cs * h * DWIDTH);
        int64_t mlp   = A(cs * (is_ / NUM_CORES) * DWIDTH);

        // SDPA tmp and mask sizing
        SdpaConfig sdpa_cfg = make_sdpa_config(ctx.use_attn_mask ? 4 : 1);
        int64_t tmp = A(sdpa_compute_tmp_v16_size(sdpa_cfg, cs) * 32);
        int64_t mask_sz = ctx.use_attn_mask
            ? A(cs * CeilDiv(ctx.max_kv_seq_len, (int64_t)16) * 32)
            : 0;

        // AdaRMS extras (sized per-core; broadcast DMA means same addr on all cores)
        int64_t cond_sz     = A(h * DWIDTH);
        int64_t gemv_out_sz = A(3 * h * DWIDTH);
        int64_t pair_gemv_out_sz = A(6 * h * DWIDTH);

        // QK norm weights (new vs AdaRMS)
        int64_t norm_w_sz = A(hd * DWIDTH);

        constexpr BufferScope ALL = BufferScope::LayerWide;
        int64_t nl = num_layers();

        std::vector<BufferDecl> decls;
        if (cold_config_.packed_qkv) {
            decls.push_back({"packed_qkv",
                A(cs * (local_q * hd + 2 * local_kv) * DWIDTH),
                0, 0, StorageClass::Temp, 0, nullptr, ALL});
        }
        if (cold_config_.adarms_resident) {
            decls.push_back({"adarms_pair_resident",
                num_steps_ * nl * 6 * h * DWIDTH,
                0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"adarms_final_resident",
                num_steps_ * 3 * h * DWIDTH,
                0, 0, StorageClass::Temp, 0, nullptr, ALL});
        }

        // Structural (mirrors AdaRMS, flat phases 0,0)
        decls.push_back({"residual1",    res,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"input_norm",   res,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"residual2",    res,     0, 0, StorageClass::Temp, 0, nullptr, ALL});

        // Phase 1 small-shape fusion: a persistent zeroed [seq, hidden] buffer
        // fed to all_reduce as the "residual" (add 0), so the gated-residual
        // path drops its redundant SUB. Zeroed once at BUILD via preload; never
        // written afterwards (P2-safe). Only declared when the flag is on.
        if (cold_config_.gated_no_sub) {
            BufferDecl zr;
            zr.name    = "zero_resid";
            zr.size    = res;
            zr.storage = StorageClass::Persistent;
            zr.scope   = ALL;
            zr.preload_callback =
                [res](FusedModelBase&, int, uint32_t core0_addr) {
                    rpu_launch_memset_spm_multicore(core0_addr, res / DWIDTH);
                };
            decls.push_back(zr);
        }
        decls.push_back({"q",            q,       0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"k",            kv,      0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"v",            kv,      0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"output",       out,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"oproj",        oproj,   0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"sdpa_tmp",     tmp,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"sdpa_mask",    mask_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"gate",         mlp,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"up",           mlp,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"down",         res,     0, 0, StorageClass::Temp, 0, nullptr, ALL});

        // AdaRMS-specific
        decls.push_back({"cond",         cond_sz,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"attn_gemv",    gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"mlp_gemv",     gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"bias_temp",    gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"gemv_partial", gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"adarms_pair_gemv",       pair_gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"adarms_pair_bias_temp",  pair_gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"adarms_pair_partial",    pair_gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});

        // DEBUG snapshot buffers (only used when get_debug_export() is true;
        // allocated unconditionally so the graph layout stays stable).
        // These capture Q/K at Phase 3 (prenorm) and Phase 3b (post-norm)
        // without conflicting with later phases that reuse q/k/output/input_norm.
        decls.push_back({"dbg_q_prenorm", logical_q_bytes, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"dbg_k_prenorm", kv,    0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"dbg_q_norm",    logical_q_bytes, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"dbg_k_norm",    kv,    0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"dbg_sdpa_out",  logical_q_bytes, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"dbg_oproj_out", oproj, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"dbg_layer_out", res,   0, 0, StorageClass::Temp, 0, nullptr, ALL});

        if (denoise_loop_weights_ready_) {
            int64_t ah = action_horizon_;
            int64_t adp = action_dim_pad_;
            int64_t sdp = state_dim_pad_;
            int64_t x_t_sz = A(ah * adp * DWIDTH);
            int64_t state_sz = A(sdp * DWIDTH);
            int64_t action_h_sz = A(ah * h * DWIDTH);
            int64_t token_h_sz = A(h * DWIDTH);
            int64_t v_t_sz = A(ah * adp * DWIDTH);

            auto preload_h = [](const at::Tensor* t, int64_t elems) {
                return [t, elems](FusedModelBase&, int, uint32_t core0_addr) {
                    rpu_launch_ddr_broadcast_spm_dma(
                        t->data_ptr<c10::Half>(), elems, core0_addr, /*num_cores=*/1);
                };
            };
            decls.push_back({"x_t_spm", x_t_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"action_mask_spm", x_t_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"state_spm", state_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"state_mask_spm", state_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"action_raw_spm", action_h_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            if (!direct_action_input_) {
                decls.push_back({"action_hidden_spm", action_h_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
                decls.push_back({"time_spm", token_h_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
                decls.push_back({"time_proj_spm", token_h_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            }
            decls.push_back({"action_mask_hidden_spm", action_h_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"state_token_spm", token_h_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"state_mask_hidden_spm", token_h_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"final_gemv", gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"final_out_spm", res, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"v_t_core0_spm", v_t_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});

            decls.push_back({"action_in_b_spm", token_h_sz, 0, 0,
                StorageClass::Persistent, 0, nullptr, ALL,
                preload_h(&action_in_b_, h)});
            if (!direct_action_input_) {
                decls.push_back({"action_time_in_b_spm", token_h_sz, 0, 0,
                    StorageClass::Persistent, 0, nullptr, ALL,
                    preload_h(&action_time_in_b_, h)});
                decls.push_back({"time_in_b_spm", token_h_sz, 0, 0,
                    StorageClass::Persistent, 0, nullptr, ALL,
                    preload_h(&time_in_b_, h)});
                decls.push_back({"time_out_b_spm", token_h_sz, 0, 0,
                    StorageClass::Persistent, 0, nullptr, ALL,
                    preload_h(&time_out_b_, h)});
            }
            decls.push_back({"state_b_spm", token_h_sz, 0, 0,
                StorageClass::Persistent, 0, nullptr, ALL,
                preload_h(&state_b_, h)});
            decls.push_back({"state_mask_b_spm", token_h_sz, 0, 0,
                StorageClass::Persistent, 0, nullptr, ALL,
                preload_h(&state_mask_b_, h)});
            decls.push_back({"action_mask_b_spm", token_h_sz, 0, 0,
                StorageClass::Persistent, 0, nullptr, ALL,
                preload_h(&action_mask_b_, h)});
            decls.push_back({"action_out_b_spm", A(adp * DWIDTH), 0, 0,
                StorageClass::Persistent, 0, nullptr, ALL,
                preload_h(&action_out_b_, adp)});
        }

        // ─────────────────────────────────────────────────────────────────────
        // QK norm preloads — D-502 PersistentPerLayer + .preload_callback.
        //
        // Hoists the per-(layer, chunk) `rpu_launch_ddr2spm_multicore` calls
        // out of build_layer_subgraph: the legacy launcher has a size-sensitive
        // correctness bug, and a per-chunk broadcast-DMA conversion would
        // overflow the firmware batch cap. Pattern matches Qwen3 / gemma.
        // ─────────────────────────────────────────────────────────────────────
        {
            BufferDecl q_norm;
            q_norm.name      = "q_norm_w";
            q_norm.size      = norm_w_sz;
            q_norm.storage   = StorageClass::PersistentPerLayer;
            q_norm.per_layer = nl;
            q_norm.scope     = BufferScope::LayerWide;
            q_norm.preload_callback =
                [this](FusedModelBase&, int L, uint32_t core0_addr) {
                    rpu_launch_ddr_broadcast_spm_dma(
                        layer_weights_[L].q_norm_w.data_ptr<c10::Half>(),
                        head_dim(), core0_addr);
                };
            decls.push_back(q_norm);
        }
        {
            BufferDecl k_norm;
            k_norm.name      = "k_norm_w";
            k_norm.size      = norm_w_sz;
            k_norm.storage   = StorageClass::PersistentPerLayer;
            k_norm.per_layer = nl;
            k_norm.scope     = BufferScope::LayerWide;
            k_norm.preload_callback =
                [this](FusedModelBase&, int L, uint32_t core0_addr) {
                    rpu_launch_ddr_broadcast_spm_dma(
                        layer_weights_[L].k_norm_w.data_ptr<c10::Half>(),
                        head_dim(), core0_addr);
                };
            decls.push_back(k_norm);
        }

        return decls;
    }

    // The denoise-loop block above adds non-aliased Temp decls sized from
    // action_horizon_ / action_dim_pad_ / state_dim_pad_, and none of the three
    // reaches the framework's params hash. The persistent guard does not cover
    // them either: `ah` and `sdp` appear in no Persistent size at all, and the
    // one Persistent that carries `adp` is Align(adp*2, 256) = 256 for every
    // adp <= 128. These arrive through set_denoise_loop_weights(), which is a
    // SEPARATE op from set_weights with no C++-enforced ordering against
    // forward() — so denoise_loop_weights_ready_ flipping false->true after a
    // first forward is reachable, and that flip alone repacks the temp arena.
    int64_t subclass_layout_hash() const override {
        int64_t h = detail::layout_mix(0, denoise_loop_weights_ready_ ? 1 : 0);
        h = detail::layout_mix(h, direct_action_input_ ? 1 : 0);
        h = detail::layout_mix(h, action_horizon_);
        h = detail::layout_mix(h, action_dim_pad_);
        h = detail::layout_mix(h, state_dim_pad_);
        h = detail::layout_mix(h, cold_config_.expert_fusions);
        h = detail::layout_mix(h, cold_config_.adarms_resident);
        h = detail::layout_mix(h, cold_config_.packed_qkv);
        h = detail::layout_mix(h, cold_config_.aligned_kv);
        h = detail::layout_mix(h, expert_w8a16_);
        h = detail::layout_mix(h, full_action_w8a16_);
        h = detail::layout_mix(h, gates_tanh_precomputed_);
        h = detail::layout_mix(h, high_precision_);
        if (high_precision_) {
            h = detail::layout_mix(h, high_precision_fusions_);
            h = detail::layout_mix(h, vector_k_norm_);
            h = detail::layout_mix(h, vector_q_norm_);
            h = detail::layout_mix(h, partial_rope_);
        }
        if (cold_config_.adarms_resident) h = detail::layout_mix(h, num_steps_);
        return h;
    }

    // ========================================================================
    // build_layer_subgraph — complete 16-operation per-layer pipeline
    //
    // Pipeline: cond DMA -> GEMV -> input DMA -> AdaRMSNorm(attn) -> QKV ->
    //   QKNorm -> RoPE -> KV insert -> SDPA -> OProj -> all_reduce_sum_residual
    //   -> gated attn residual(4-step) -> AdaRMSNorm(MLP) -> MLP(SiLU) ->
    //   gated MLP residual(4-step) -> output DMA
    //
    // Key differences from AdaRMS:
    //   - RoPE position and physical KV row are intentionally independent
    //   - kv_insert_pos = ctx().position + chunk.offset (absolute cache position)
    //   - QK Norm before RoPE (Qwen3 pattern)
    //   - Gated residual = SUB -> TANH -> broadcast MUL -> ADD (4 steps, not 3)
    //   - ActivationKind::SILU (not GELU)
    //   - cos_ref_/sin_ref_ DDR pointers (not cos_/sin_)
    //
    // Pitfall 3 structural fix: SDPA + KV-insert use addr_offset(name).value
    // (typed SpmOffset, compile-time distinct from absolute addr() uint32_t).
    // ========================================================================

    // RhinoVLA-only MLP pipeline variant for the small-shape fusion flags.
    // Mirrors FusedModelBase::emit_mlp_pipeline (SiLU path) but optionally:
    //   fuse_silu_mul: gate -> up -> silu_mul(gate,up,gate)  (drops SiLU unary)
    //   zero_residual: all_reduce gets zero_resid  -> residual1 = reduce(down),
    //                  so the caller's gated-residual SUB can be dropped.
    // Activations stay FP16; optional natural per-channel scales select W8A16
    // in the same generic ACC16 projection launcher used by the shared path.
    void emit_rhino_mlp_pipeline(const at::Tensor& gate_w,
                                 const at::Tensor& up_w,
                                 const at::Tensor& down_w,
                                 int64_t seq_len,
                                 bool fuse_silu_mul,
                                 bool zero_residual,
                                 const at::Tensor& gate_ws,
                                 const at::Tensor& up_ws,
                                 const at::Tensor& down_ws) {
        const int64_t h   = hidden_size();
        const int64_t is_ = intermediate_size();
        const int64_t elems = seq_len * (is_ / NUM_CORES);

        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, RHINO_MLP_GATE_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            expert_w8a16_ ? RHINO_LINEAR_W8A16_PER_CHANNEL : 0,
            expert_projection_route_arguments(seq_len, is_, h, 1));
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "residual1"), gate_w, addr(0, "gate"),
            seq_len, is_, h, /*partition=*/1, NUM_CORES, /*bias=*/0, gate_ws);

        if (fuse_silu_mul) {
            // up first, then fused silu(gate)*up -> gate
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR,
                RHINO_MLP_UP_FUSED_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                expert_w8a16_ ? RHINO_LINEAR_W8A16_PER_CHANNEL : 0,
                expert_projection_route_arguments(seq_len, is_, h, 1));
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "residual1"), up_w, addr(0, "up"),
                seq_len, is_, h, /*partition=*/1, NUM_CORES, /*bias=*/0, up_ws);
            rpu_launch_silu_mul_spm_kernel(
                addr(0, "gate"), addr(0, "up"), addr(0, "gate"),
                elems, NUM_CORES);
        } else {
            if (!high_precision_fusions_) {
                rpu_launch_eltwise_unary_spm_kernel(
                    addr(0, "gate"), addr(0, "gate"), elems, ValuOpType::SILU,
                    GeluMode::NONE, NUM_CORES, unary_precision());
            }
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR,
                RHINO_MLP_UP_PLAIN_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                expert_w8a16_ ? RHINO_LINEAR_W8A16_PER_CHANNEL : 0,
                expert_projection_route_arguments(seq_len, is_, h, 1));
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "residual1"), up_w, addr(0, "up"),
                seq_len, is_, h, /*partition=*/1, NUM_CORES, /*bias=*/0, up_ws);
            if (high_precision_fusions_) {
                rpu_launch_rhinovla_silu_high_mul_spm_kernel(
                    addr(0, "gate"), addr(0, "up"), addr(0, "gate"), elems, NUM_CORES);
            } else {
                rpu_launch_eltwise_binary_spm_kernel(
                    addr(0, "gate"), addr(0, "up"), addr(0, "gate"),
                    elems, ValuOpType::MUL, c10::Half(1.0));
            }
        }

        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, RHINO_MLP_DOWN_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            expert_w8a16_ ? RHINO_LINEAR_W8A16_PER_CHANNEL : 0,
            expert_projection_route_arguments(seq_len, h, is_, 0));
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "gate"), down_w, addr(0, "down"),
            seq_len, h, is_, /*partition=*/0, NUM_CORES, /*bias=*/0, down_ws);

        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE, RHINO_MLP_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(seq_len, h),
            /*resolved_flags=*/0);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "down"),
            zero_residual ? addr(0, "zero_resid") : addr(0, "residual2"),
            addr(0, "residual1"),
            seq_len, h);
    }

    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {
        const auto& lw = layer_weights_[layer_idx];
        int64_t seq_len = chunk.len;
        int64_t cos_sin_start = rope_position_ + chunk.offset;
        int64_t kv_insert_pos = ctx().position + chunk.offset;  // absolute cache pos
        int64_t h = hidden_size();
        int64_t nq = num_q_heads();
        int64_t nkv = num_kv_heads();
        int64_t hd = head_dim();
        int tp = attn_tp();
        int64_t num_elems_full = seq_len * h;
        bool is_first_layer_first_chunk =
            (layer_idx == 0) && (chunk.idx == 0) && !cond_loaded_this_forward_;
        bool is_chunk_zero = (chunk.idx == 0);
        bool fused_adarms = cold_config_.fused_adarms;
        bool skip_adarms_gemv = cold_config_.skip_adarms_gemv;
        bool precomputed_adarms =
            loop_mode_ && cold_config_.precompute_adarms;
        if (cold_config_.expert_fusions) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE, RHINO_EXPERT_FUSIONS_SITE,
                1, 0, {seq_len, h, num_layers(), num_steps_}, chunk.idx);
        }
        if (expert_w8a16_) consume_expert_w8a16_route(chunk);

        // --------------------------------------------------------------------
        // Phase 0: cond DMA (once per forward)
        //
        // Row-partition GEMV consumes the cond as a per-core K-slice:
        // core c needs cond[c*H/8 : (c+1)*H/8] at its SPM base offset.
        // The GEMV kernel does NOT stride into a replicated buffer; it
        // just reads `local_k = H/8` elements from `core_base + cond_addr`
        // on each core. So we SCATTER cond from DDR to per-core SPM
        // (core c reads cond[c*H/8:(c+1)*H/8] from DDR and writes to the
        // SAME local SPM offset on its own core).
        //
        // NOTE: an earlier implementation used `ddr2spm_multicore`
        // (broadcast) here — that caused every core to see cond[0:H/8]
        // only, which gave correct core-0 partial and garbage on cores
        // 1-7. Root-caused via standalone test on 2026-04-15.
        // --------------------------------------------------------------------
        if (!loop_mode_ && is_first_layer_first_chunk) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA, RHINO_COND_DMA_SITE,
                static_cast<int64_t>(
                    RhinoMutableDmaRoute::DDR_SCATTER_TO_SPM),
                /*resolved_flags=*/0, {loop_mode_ ? 1 : 0}, chunk.idx);
            int64_t local_k           = h / NUM_CORES;
            int64_t core_stride_bytes = local_k * DWIDTH;
            rpu_launch_ddr_scatter_spm_dma_mutable(
                &cond_src_base_,
                /*src_offset_bytes=*/0,
                /*elements_per_core=*/local_k,
                /*core_stride_bytes=*/core_stride_bytes,
                addr(0, "cond"),
                /*num_cores=*/NUM_CORES);
            cond_loaded_this_forward_ = true;
        }

        // --------------------------------------------------------------------
        // Phase A: per-layer GEMV (chunk 0 only)
        // Computes [scale | shift | gate] from cond * dense_w + bias, with
        // +1.0 baked into the scale slice. Result persists in SPM for this
        // layer's remaining chunks.
        // --------------------------------------------------------------------
        if (is_chunk_zero) {
            if (skip_adarms_gemv) {
                identity_adarms_pair_to_spm(addr(0, "adarms_pair_gemv"));
            } else if (precomputed_adarms) {
                if (!cold_config_.adarms_resident) {
                    int64_t step = ctx().body_iter;
                    int64_t offset =
                        ((step * num_layers() + layer_idx) * 6 * h) * DWIDTH;
                    ctx().consume_physical_route(
                        FmbRouteFamily::MUTABLE_DMA,
                        RHINO_ADARMS_TABLE_DMA_SITE,
                        static_cast<int64_t>(
                            RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                        /*resolved_flags=*/0, {}, chunk.idx);
                    rpu_launch_ddr_broadcast_spm_dma_mutable(
                        &adarms_pair_table_src_base_,
                        offset,
                        6 * h,
                        addr(0, "adarms_pair_gemv"),
                        NUM_CORES);
                }
            } else if (fused_adarms) {
                adarms_pair_gemv_to_spm(addr(0, "cond"),
                                         lw.pair_dense_w, lw.pair_dense_b,
                                         addr(0, "adarms_pair_gemv"),
                                         addr(0, "adarms_pair_bias_temp"),
                                         addr(0, "adarms_pair_partial"));
            } else {
                adarms_gemv_to_spm(addr(0, "cond"),
                                   lw.attn_dense_w, lw.attn_dense_b,
                                   addr(0, "attn_gemv"), addr(0, "bias_temp"),
                                   addr(0, "gemv_partial"));
                adarms_gemv_to_spm(addr(0, "cond"),
                                   lw.mlp_dense_w, lw.mlp_dense_b,
                                   addr(0, "mlp_gemv"), addr(0, "bias_temp"),
                                   addr(0, "gemv_partial"));
            }
        }

        // GEMV slice offsets (computed from core 0 base address; broadcast
        // DMA means every core shares the same offsets).
        const bool use_pair_adarms_buffer =
            skip_adarms_gemv || fused_adarms || precomputed_adarms;
        const uint32_t pair_base = cold_config_.adarms_resident
            ? addr(0, "adarms_pair_resident") +
                (ctx().body_iter * num_layers() + layer_idx) * 6 * h * DWIDTH
            : addr(0, "adarms_pair_gemv");
        uint32_t attn_scale = use_pair_adarms_buffer ? pair_base : addr(0, "attn_gemv");
        uint32_t attn_shift = attn_scale + (uint32_t)(h * DWIDTH);
        uint32_t attn_gate  = attn_scale + (uint32_t)(h * DWIDTH * 2);
        uint32_t mlp_scale  = use_pair_adarms_buffer
            ? pair_base + (uint32_t)(h * DWIDTH * 3)
            : addr(0, "mlp_gemv");
        uint32_t mlp_shift  = mlp_scale + (uint32_t)(h * DWIDTH);
        uint32_t mlp_gate   = mlp_scale + (uint32_t)(h * DWIDTH * 2);

        // --------------------------------------------------------------------
        // Phase 1: layer input DMA
        // --------------------------------------------------------------------
        if (!ctx().input_in_spm) {
            emit_layer_input_dma(layer_idx, chunk);
        }

        // --------------------------------------------------------------------
        // Phase 2: AdaRMSNorm (attn): residual1 -> input_norm
        // (1+scale) was applied during GEMV, so RMSNorm uses attn_scale directly.
        // Then add shift (broadcast 1xC onto NxC rows).
        // --------------------------------------------------------------------
        emit_norm_shift(
            addr(0, "residual1"), addr(0, "input_norm"),
            attn_scale, attn_shift, seq_len);

        // --------------------------------------------------------------------
        // Phase 3: QKV Linear (col partition)
        // --------------------------------------------------------------------
        if (cold_config_.packed_qkv) {
            emit_packed_qkv(lw, chunk);
        } else {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, RHINO_Q_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                expert_w8a16_ ? RHINO_LINEAR_W8A16_PER_CHANNEL : 0,
                expert_projection_route_arguments(seq_len, nq * hd, h, 1), chunk.idx);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "input_norm"), lw.q_w, addr(0, "q"),
                seq_len, nq * hd, h,
                /*partition=*/1, tp, /*bias=*/0, lw.q_ws);
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, RHINO_K_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                expert_w8a16_ ? RHINO_LINEAR_W8A16_PER_CHANNEL : 0,
                expert_projection_route_arguments(seq_len, nkv * hd, h, 1), chunk.idx);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "input_norm"), lw.k_w, addr(0, "k"),
                seq_len, nkv * hd, h,
                /*partition=*/1, tp, /*bias=*/0, lw.k_ws);
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, RHINO_V_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                expert_w8a16_ ? RHINO_LINEAR_W8A16_PER_CHANNEL : 0,
                expert_projection_route_arguments(seq_len, nkv * hd, h, 1), chunk.idx);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "input_norm"), lw.v_w, addr(0, "v"),
                seq_len, nkv * hd, h,
                /*partition=*/1, tp, /*bias=*/0, lw.v_ws);
        }

        // DEBUG EXPORT (layer 0): Phase 3 Q/K prenorm snapshot.
        if (layer_idx == 0 && chunk.idx == 0 && get_debug_export()) {
            debug_dump_spm_per_core("rhino_vla_cond_L0",
                addr(0, "cond"), h, NUM_CORES,
                addr_offset("cond").value);
            debug_dump_spm_per_core("rhino_vla_qnormw_L0",
                layer_addr(layer_idx, 0, "q_norm_w"), hd, NUM_CORES,
                layer_addr_offset(layer_idx, "q_norm_w").value);
            debug_dump_spm_per_core("rhino_vla_residual1_L0",
                addr(0, "residual1"), seq_len * h, NUM_CORES,
                addr_offset("residual1").value);
            debug_dump_spm_per_core("rhino_vla_input_norm_L0",
                addr(0, "input_norm"), seq_len * h, NUM_CORES,
                addr_offset("input_norm").value);

            // In-graph copy: dbg_q_prenorm <- q
            int64_t q_elems_per_core = seq_len * local_q_heads_ * hd;
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "q"), c10::Half(0.0), addr(0, "dbg_q_prenorm"),
                q_elems_per_core, ValuOpType::ADD);
            debug_dump_spm_per_core("rhino_vla_q_prenorm_L0",
                addr(0, "dbg_q_prenorm"),
                q_elems_per_core, tp,
                addr_offset("dbg_q_prenorm").value);

            int64_t local_kv_heads_dbg = nkv / tp;
            int64_t k_elems_per_core = seq_len * local_kv_heads_dbg * hd;
            // In-graph copy: dbg_k_prenorm <- k
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "k"), c10::Half(0.0), addr(0, "dbg_k_prenorm"),
                k_elems_per_core, ValuOpType::ADD);
            debug_dump_spm_per_core("rhino_vla_k_prenorm_L0",
                addr(0, "dbg_k_prenorm"),
                k_elems_per_core, tp,
                addr_offset("dbg_k_prenorm").value);

            // V buffer — Phase 3 V linear output, never modified after Phase 5
            // V-cache insert.
            debug_dump_spm_per_core("rhino_vla_v_L0",
                addr(0, "v"),
                k_elems_per_core, tp,
                addr_offset("v").value);
        }

        // --------------------------------------------------------------------
        // Phase 3b: QK Norm (Qwen3 pattern: RMSNorm before RoPE)
        // --------------------------------------------------------------------
        int64_t local_kv_heads = nkv / tp;
        emit_qk_norm(seq_len, local_kv_heads, hd, layer_idx);

        // DEBUG EXPORT (layer 0, chunk 0 only): snapshot Q norm and K norm
        // output into dedicated buffers BEFORE Phase 4 RoPE / Phase 6 SDPA
        // overwrite `q`/`output`/`input_norm`.
        if (layer_idx == 0 && chunk.idx == 0 && get_debug_export()) {
            int64_t q_elems_per_core = seq_len * local_q_heads_ * hd;
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "output"), c10::Half(0.0), addr(0, "dbg_q_norm"),
                q_elems_per_core, ValuOpType::ADD);
            debug_dump_spm_per_core("rhino_vla_q_norm_L0",
                addr(0, "dbg_q_norm"),
                q_elems_per_core, tp,
                addr_offset("dbg_q_norm").value);

            int64_t k_elems_per_core = seq_len * local_kv_heads * hd;
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "input_norm"), c10::Half(0.0), addr(0, "dbg_k_norm"),
                k_elems_per_core, ValuOpType::ADD);
            debug_dump_spm_per_core("rhino_vla_k_norm_L0",
                addr(0, "dbg_k_norm"),
                k_elems_per_core, tp,
                addr_offset("dbg_k_norm").value);
        }

        // --------------------------------------------------------------------
        // Phase 4: RoPE (per-forward table, logical start is independent of
        // the physical KV insertion row).
        // --------------------------------------------------------------------
        ctx().consume_physical_route(
            FmbRouteFamily::ROPE, RHINO_Q_ROPE_SITE,
            rope_route_selector(),
            /*resolved_flags=*/0, {cos_sin_start}, chunk.idx);
        rpu_launch_rhinovla_rope_spm_kernel(
            addr(0, "output"), addr(0, "q"), cos_ref_, sin_ref_,
            seq_len, local_q_heads_, hd, cos_sin_start, partial_rope_, NUM_CORES);
        ctx().consume_physical_route(
            FmbRouteFamily::ROPE, RHINO_K_ROPE_SITE,
            rope_route_selector(),
            /*resolved_flags=*/0, {cos_sin_start}, chunk.idx);
        rpu_launch_rhinovla_rope_spm_kernel(
            addr(0, "input_norm"), addr(0, "k"), cos_ref_, sin_ref_,
            seq_len, local_kv_heads, hd, cos_sin_start, partial_rope_, NUM_CORES);

        // DEBUG EXPORT (layer 0, chunk 0 only): dump Q/K after RoPE
        if (layer_idx == 0 && chunk.idx == 0 && get_debug_export()) {
            debug_dump_spm_per_core("rhino_vla_q_rope_L0",
                addr(0, "q"),
                seq_len * local_q_heads_ * hd, tp,
                addr_offset("q").value);
            debug_dump_spm_per_core("rhino_vla_k_rope_L0",
                addr(0, "k"),
                seq_len * local_kv_heads * hd, tp,
                addr_offset("k").value);
        }

        // --------------------------------------------------------------------
        // Phase 5: KV cache insert (absolute position, NOT cos_sin_start).
        //
        // Pitfall 3 structural fix: KV-insert takes SPM OFFSETS, not absolute
        // addresses. addr_offset("name").value is compile-time distinct from
        // addr(core, "name").
        // --------------------------------------------------------------------
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        const auto& kv_route = ctx().find_physical_route(
            FmbRouteFamily::KV_INSERT, RHINO_KV_INSERT_SITE,
            chunk.idx);
        const KvInsertSegmentPlan kv_plan =
            restore_kvinsert_plan(
                RHINO_KV_INSERT_SITE, kv_route.arguments, tp, nkv, hd);
        const int64_t kv_physical_rows = cold_config_.aligned_kv ? 32 : seq_len;
        TORCH_CHECK(kv_plan.segment(0).position == kv_insert_pos &&
                        kv_plan.logical_rows() == seq_len &&
                        kv_plan.physical_rows() == kv_physical_rows,
                    "RhinoVLA KV descriptor geometry drift");
        ctx().consume_physical_route(
            FmbRouteFamily::KV_INSERT, RHINO_KV_INSERT_SITE,
            static_cast<int64_t>(kv_plan.route()),
            RHINO_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED,
            kv_route.arguments, chunk.idx);
        if (cold_config_.aligned_kv && !vector_k_norm_) {
            emit_aligned_kv_tail_zero(seq_len);
        }
        rpu_launch_insert_kvcache_spm_unified_with_plan(
            k_cache, v_cache,
            addr_offset("k").value, addr_offset("v").value,
            nkv, hd, tp,
            /*k_cache_batch_offset_elems=*/0,
            /*v_cache_batch_offset_elems=*/0,
            /*spm_rows=*/cold_config_.aligned_kv ? kv_physical_rows : 0, kv_plan);

        // --------------------------------------------------------------------
        // Phase 6: SDPA (2D mask from Python, non-causal for prefix KV).
        //
        // Pitfall 3 structural fix: SDPA args (q / output / sdpa_tmp /
        // sdpa_mask) take SPM offsets via addr_offset(...).value.
        // --------------------------------------------------------------------
        int64_t kv_seq_len = ctx().position + ctx().seq_len;
        ctx().consume_physical_attention_route(
            RHINO_ATTENTION_SITE, AttentionExecutionPolicy::DDR_KV,
            chunk.idx);
        const bool has_2d_mask =
            ctx().attention_mask.has_value() && ctx().attention_mask->defined();
        // The SPM flash-attn kernel mishandles invalid lanes in a non-initial
        // partial K v16 tile. RhinoVLA's 2D mask path pads and masks dummy lanes.
        int64_t sdpa_kv_seq_len = has_2d_mask ? Align(kv_seq_len, (int64_t)16) : kv_seq_len;
        int mask_type = emit_sdpa_mask(
            layer_idx, chunk, seq_len, kv_seq_len, sdpa_kv_seq_len, tp,
            addr_offset("sdpa_mask").value);
        rpu_launch_sdpa_spm_unified_kernel_v2(
            k_cache, v_cache, mask_type, c10::nullopt,
            addr_offset("q").value,
            addr_offset("output").value,
            addr_offset("sdpa_tmp").value,
            addr_offset("sdpa_mask").value,
            seq_len, nq, nkv, hd,
            sdpa_kv_seq_len, tp, NUM_CORES);

        // DEBUG EXPORT: snapshot SDPA output before O_proj consumes it
        if (layer_idx == 0 && chunk.idx == 0 && get_debug_export()) {
            int64_t q_elems_per_core = seq_len * local_q_heads_ * hd;
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "output"), c10::Half(0.0), addr(0, "dbg_sdpa_out"),
                q_elems_per_core, ValuOpType::ADD);
            debug_dump_spm_per_core("rhino_vla_sdpa_out_L0",
                addr(0, "dbg_sdpa_out"),
                q_elems_per_core, tp,
                addr_offset("dbg_sdpa_out").value);
        }

        // --------------------------------------------------------------------
        // Phase 7: O_proj (row partition)
        // --------------------------------------------------------------------
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE,
            RHINO_PREPARE_ALL_REDUCE_SITE,
            static_cast<int64_t>(RhinoAllReduceRoute::PREPARE_RING_INPUT),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_prepare_ring_all_reduce_input(
            addr(0, "oproj"), seq_len, h, tp);
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, RHINO_O_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            expert_w8a16_ ? RHINO_LINEAR_W8A16_PER_CHANNEL : 0,
            expert_projection_route_arguments(seq_len, h, nq * hd, 0), chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "output"), lw.o_w, addr(0, "oproj"),
            seq_len, h, nq * hd,
            /*partition=*/0, tp, /*bias=*/0, lw.o_ws);

        // DEBUG EXPORT: snapshot O_proj per-core output before all_reduce
        if (layer_idx == 0 && chunk.idx == 0 && get_debug_export()) {
            int64_t oproj_elems_per_core = seq_len * h;
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "oproj"), c10::Half(0.0), addr(0, "dbg_oproj_out"),
                oproj_elems_per_core, ValuOpType::ADD);
            debug_dump_spm_per_core("rhino_vla_oproj_out_L0",
                addr(0, "dbg_oproj_out"),
                oproj_elems_per_core, NUM_CORES,
                addr_offset("dbg_oproj_out").value);
        }

        // --------------------------------------------------------------------
        // Phase 8: all_reduce_sum_residual
        // residual2 = reduce_sum(oproj) + residual1
        //   no_sub: feed zero residual -> residual2 = reduce_sum(oproj), so the
        //   redundant Phase-9 SUB (which only undoes the +residual1) is dropped.
        // --------------------------------------------------------------------
        const bool gated_no_sub = cold_config_.gated_no_sub;
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE,
            RHINO_GATED_RESIDUAL_SCHEDULE_SITE,
            rhino_gated_residual_selector(cold_config_.gated_no_sub),
            /*resolved_flags=*/0,
            {cold_config_.gated_no_sub ? 1 : 0}, chunk.idx);
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE,
            RHINO_ATTN_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(seq_len, h),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"),
            gated_no_sub ? addr(0, "zero_resid") : addr(0, "residual1"),
            addr(0, "residual2"),
            seq_len, h, tp, NUM_CORES);

        // --------------------------------------------------------------------
        // Phase 9: Gated attn residual (SUB -> TANH -> MUL -> ADD)
        // After all_reduce: residual2 = reduce(oproj) (+residual1 unless no_sub)
        // Goal: residual2 = residual1 + tanh(attn_gate) * reduce(oproj)
        //
        // SUB: residual2 = residual2 - residual1 = reduce(oproj) [skip if no_sub]
        // --------------------------------------------------------------------
        if (!gated_no_sub) {
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "residual2"), addr(0, "residual1"), addr(0, "residual2"),
                num_elems_full, ValuOpType::SUB, c10::Half(1.0), NUM_CORES);
        }
        // TANH: attn_gate = tanh(attn_gate) in-place [1,H]
        if (!gates_tanh_precomputed_) {
            rpu_launch_eltwise_unary_spm_kernel(
                attn_gate, attn_gate,
                h, ValuOpType::TANH,
                /*is_gelu=*/false, /*num_cores=*/NUM_CORES, unary_precision());
        }
        // MUL: residual2 = attn_gate [1,H] * residual2 [S,H]
        if (cold_config_.expert_fusions) {
            rpu_launch_pi05_gated_residual_spm_kernel(
                addr(0, "residual2"), addr(0, "residual1"), attn_gate,
                addr(0, "residual2"), seq_len, NUM_CORES);
        } else {
            rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                attn_gate, addr(0, "residual2"), addr(0, "residual2"),
                seq_len, h,
                c10::Half(1.0), ValuOpType::MUL, /*is_bopa=*/false);
            // ADD: residual2 = residual2 + residual1 (gated result in residual2)
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "residual2"), addr(0, "residual1"), addr(0, "residual2"),
                num_elems_full, ValuOpType::ADD, c10::Half(1.0), NUM_CORES);
        }

        // --------------------------------------------------------------------
        // Phase 10: AdaRMSNorm (MLP): residual2 -> residual1
        // --------------------------------------------------------------------
        emit_norm_shift(
            addr(0, "residual2"), addr(0, "residual1"),
            mlp_scale, mlp_shift, seq_len);

        // --------------------------------------------------------------------
        // Phase 11: MLP pipeline (SiLU, NOT GELU)
        // emit_mlp_pipeline: reads residual1, writes residual1 = reduce(down) + residual2
        //   silu_mul: fuse SiLU(gate)+(gate*up) into one kernel.
        //   no_sub:   feed zero residual -> residual1 = reduce(down), dropping
        //             the redundant Phase-13 SUB below.
        // --------------------------------------------------------------------
        const bool fuse_silu_mul = cold_config_.fused_silu_mul && !high_precision_;
        if (fuse_silu_mul || gated_no_sub || high_precision_) {
            emit_rhino_mlp_pipeline(lw.gate_proj_w, lw.up_proj_w, lw.down_proj_w,
                                    seq_len, fuse_silu_mul, gated_no_sub,
                                    lw.gate_ws, lw.up_ws, lw.down_ws);
        } else {
            emit_mlp_pipeline(lw.gate_proj_w, lw.up_proj_w, lw.down_proj_w,
                              seq_len, ActivationKind::SILU,
                              lw.gate_ws, lw.up_ws, lw.down_ws);
        }

        // --------------------------------------------------------------------
        // Phase 13: Gated MLP residual (SUB -> TANH -> MUL -> ADD)
        // After MLP: residual1 = reduce(down) (+residual2 unless no_sub)
        // Goal: residual1 = residual2 + tanh(mlp_gate) * reduce(down)
        // --------------------------------------------------------------------
        if (!gated_no_sub) {
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "residual1"), addr(0, "residual2"), addr(0, "residual1"),
                num_elems_full, ValuOpType::SUB, c10::Half(1.0), NUM_CORES);
        }
        // TANH: mlp_gate = tanh(mlp_gate) in-place [1,H]
        if (!gates_tanh_precomputed_) {
            rpu_launch_eltwise_unary_spm_kernel(
                mlp_gate, mlp_gate,
                h, ValuOpType::TANH,
                /*is_gelu=*/false, /*num_cores=*/NUM_CORES, unary_precision());
        }
        // MUL: residual1 = mlp_gate [1,H] * residual1 [S,H]
        if (cold_config_.expert_fusions) {
            rpu_launch_pi05_gated_residual_spm_kernel(
                addr(0, "residual1"), addr(0, "residual2"), mlp_gate,
                addr(0, "residual1"), seq_len, NUM_CORES);
        } else {
            rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                mlp_gate, addr(0, "residual1"), addr(0, "residual1"),
                seq_len, h,
                c10::Half(1.0), ValuOpType::MUL, /*is_bopa=*/false);
            // ADD: residual1 = residual1 + residual2 (final gated MLP result)
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "residual1"), addr(0, "residual2"), addr(0, "residual1"),
                num_elems_full, ValuOpType::ADD, c10::Half(1.0), NUM_CORES);
        }

        // DEBUG EXPORT: snapshot per-layer final output (residual1)
        if (layer_idx == 0 && chunk.idx == 0 && get_debug_export()) {
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "residual1"), c10::Half(0.0), addr(0, "dbg_layer_out"),
                num_elems_full, ValuOpType::ADD);
            debug_dump_spm_per_core("rhino_vla_layer_out_L0",
                addr(0, "dbg_layer_out"),
                num_elems_full, NUM_CORES,
                addr_offset("dbg_layer_out").value);
        }

        // --------------------------------------------------------------------
        // Phase 14: Output DMA
        // --------------------------------------------------------------------
        if (!ctx().output_to_spm) {
            emit_layer_output_dma(layer_idx, chunk);
        }
    }

private:
    RpuUnaryPrecision unary_precision() const {
        return high_precision_ ? RpuUnaryPrecision::HIGH : RpuUnaryPrecision::BASE;
    }

    RpuRmsNormSpmRoute rmsnorm_route(int64_t rows, int64_t cols) const {
        return high_precision_ ? rpu_resolve_high_precision_rmsnorm_spm_route(rows, cols)
                               : RpuRmsNormSpmRoute::BASE;
    }

    int64_t k_norm_rows(int64_t rows, int64_t local_kv_heads, int64_t cols) const {
        if (!vector_k_norm_) return rows * local_kv_heads;
        validate_expert_transfer(rows);
        TORCH_CHECK(high_precision_ && cold_config_.aligned_kv && rows == 31 &&
                        local_kv_heads == 1 && cols == 128,
                    "RhinoVLA vector K norm requires exact aligned v3 K geometry");
        return 32;
    }

    int64_t q_norm_rows(int64_t rows, int64_t local_q_heads, int64_t cols) const {
        if (!vector_q_norm_) return rows * local_q_heads;
        validate_expert_transfer(rows);
        TORCH_CHECK(high_precision_ && rows == 31 && local_q_heads == 2 && cols == 128,
                    "RhinoVLA vector Q norm requires exact v3 Q geometry");
        return 64;
    }

    void emit_qk_norm(int64_t seq_len, int64_t local_kv_heads, int64_t hd, int layer_idx) {
        const int64_t q_rows = q_norm_rows(seq_len, local_q_heads_, hd);
        const int64_t k_rows = k_norm_rows(seq_len, local_kv_heads, hd);
        // Move the existing once-per-layer K/V tail clear before Newton V32
        // reads K. RoPE below still writes only the 31 logical rows, leaving
        // the aligned K/V cache-insert tail at +0. No extra launch or SPM.
        if (vector_k_norm_) emit_aligned_kv_tail_zero(seq_len);
        if (vector_q_norm_) {
            // Refresh the two padded rows each layer/replay. RoPE and SDPA
            // still consume only logical tokens; no extra head or KV row.
            const int64_t logical_q_rows = seq_len * local_q_heads_;
            const uint32_t tail_offset = static_cast<uint32_t>(logical_q_rows * hd * DWIDTH);
            rpu_launch_memset_spm_multicore(addr(0, "q") + tail_offset,
                (q_rows - logical_q_rows) * hd, NUM_CORES);
        }
        // PersistentPerLayer norm weights are preloaded at BUILD.
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "q"), addr(0, "output"),
            layer_addr(layer_idx, 0, "q_norm_w"),
            q_rows, hd, eps_, rmsnorm_route(q_rows, hd));
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "k"), addr(0, "input_norm"),
            layer_addr(layer_idx, 0, "k_norm_w"),
            k_rows, hd, eps_, rmsnorm_route(k_rows, hd));
    }

    int64_t high_precision_route_flags() const {
        return (gates_tanh_precomputed_ ? 1 : 0) | (high_precision_fusions_ ? 2 : 0) |
               (vector_k_norm_ ? 4 : 0) | (vector_q_norm_ ? 8 : 0) |
               (partial_rope_ ? 16 : 0);
    }

    int64_t rope_route_selector() const {
        return static_cast<int64_t>(partial_rope_
            ? RhinoRopeRoute::PARTIAL_MROPE_FULL_HEAD_LOGICAL_SUFFIX
            : RhinoRopeRoute::ROPE_1D_LOGICAL_SUFFIX);
    }

    std::vector<int64_t> high_precision_route_arguments() const {
        // Version 1: FP16 activations, Newton norms, high SiLU/TANH.
        return {1, suffix_len_, hidden_size(), intermediate_size(), num_layers(),
                num_steps_, num_q_heads(), num_kv_heads(), head_dim(),
                static_cast<int64_t>(installed_model_state_generation())};
    }

    void consume_high_precision_route() {
        validate_expert_transfer(suffix_len_);
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            RHINO_HIGH_PRECISION_SITE, 1, high_precision_route_flags(),
            high_precision_route_arguments());
    }

    std::vector<int64_t> full_io_route_arguments(int64_t rows, int64_t n, int64_t k) const {
        return full_action_w8a16_ ? std::vector<int64_t>{rows, n, k, 1, 1} : std::vector<int64_t>{};
    }

    void consume_full_action_w8a16_route() {
        validate_expert_transfer(suffix_len_);
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            RHINO_FULL_W8A16_SITE, 1, gates_tanh_precomputed_ ? 1 : 0,
            {num_layers(), num_steps_, 37, 5, 8, 16});
    }

    std::vector<int64_t> expert_projection_route_arguments(
        int64_t rows, int64_t n, int64_t k, int partition) const {
        if (!expert_w8a16_) return {};
        return {rows, n, k, partition, NUM_CORES};
    }

    std::vector<int64_t> expert_w8a16_route_arguments(int64_t rows) const {
        return {rows, hidden_size(), intermediate_size(), num_layers(),
                num_steps_, 7, 8, 16, NUM_CORES};
    }

    void consume_expert_w8a16_route(const ChunkInfo& chunk) {
        validate_expert_transfer(chunk.len);
        // This owner policy also covers the shared MLP fallback, whose common
        // AUTO_TILE descriptor is independent of projection storage dtype.
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, RHINO_EXPERT_W8A16_SITE,
            1, 0, expert_w8a16_route_arguments(chunk.len), chunk.idx);
    }

    void validate_expert_transfer(int64_t rows) const {
        if (!(cold_config_.expert_fusions || cold_config_.adarms_resident ||
              cold_config_.packed_qkv || cold_config_.aligned_kv || expert_w8a16_)) return;
        TORCH_CHECK(loop_mode_ && direct_action_input_ && num_steps_ == 10 &&
                        num_layers() == 18 && hidden_size() == 1024 &&
                        intermediate_size() == 3072 && num_q_heads() == 16 &&
                        num_kv_heads() == 8 && head_dim() == 128 && attn_tp() == 8 &&
                        action_dim_ == 96 && state_dim_ == 96 &&
                        action_horizon_ == 30 && suffix_len_ == 31 && rows == 31,
                    "RhinoVLA expert transfer requires exact v3 H1024/18L/96D/M31/10-step FP16 activation loop");
        TORCH_CHECK(!(cold_config_.expert_fusions || cold_config_.adarms_resident) ||
                        (cold_config_.precompute_adarms && !cold_config_.skip_adarms_gemv &&
                         denoise_adarms_tables_ready_),
                    "RhinoVLA expert transfer requires ready precomputed AdaRMS tables");
        TORCH_CHECK(!full_action_w8a16_ ||
                        (cold_config_.precompute_adarms && !cold_config_.skip_adarms_gemv &&
                         denoise_adarms_tables_ready_ && full_io_scales_.size() == 6 &&
                         full_cold_weights_.size() == 19 && full_cold_scales_.size() == 19 &&
                         full_cold_biases_.size() == 19),
                    "RhinoVLA full W8 requires bound IO scales and cold W8 AdaRMS table owners");
    }

    void emit_aligned_kv_tail_zero(int64_t rows) {
        validate_expert_transfer(rows);
        // The last PAD16 lane is visible to the aligned SDPA tile under its
        // finite mask bias. Explicit +0 prevents stale NaNs from leaking through
        // QK or value accumulation, and is refreshed on every layer and replay.
        const int64_t local_kv = num_kv_heads() * head_dim() / attn_tp();
        const uint32_t tail_offset = static_cast<uint32_t>(rows * local_kv * DWIDTH);
        rpu_launch_memset_spm_multicore(addr(0, "k") + tail_offset, local_kv, NUM_CORES);
        rpu_launch_memset_spm_multicore(addr(0, "v") + tail_offset, local_kv, NUM_CORES);
    }

    void emit_packed_qkv(const LayerWeights& weights, const ChunkInfo& chunk) {
        validate_expert_transfer(chunk.len);
        TORCH_CHECK(weights.packed_qkv_w.defined(),
                    "RhinoVLA packed QKV owner was not installed");
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, RHINO_PACKED_QKV_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            0, {chunk.len, 4096, 1024, 1, NUM_CORES}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), weights.packed_qkv_w, addr(0, "packed_qkv"),
            chunk.len, 4096, 1024, /*partition=*/1, NUM_CORES);
        // Each core owns two Q heads, one K head and one V head. Split the
        // packed rows into the original non-overlapping buffers before QK norm
        // and RoPE using exact per-core byte copies, without numerical conversion.
        rpu_launch_rhino_packed_qkv_split_spm_kernel(
            addr(0, "packed_qkv"), addr(0, "q"),
            chunk.len, 0, 256);
        rpu_launch_rhino_packed_qkv_split_spm_kernel(
            addr(0, "packed_qkv"), addr(0, "k"),
            chunk.len, 256, 128);
        rpu_launch_rhino_packed_qkv_split_spm_kernel(
            addr(0, "packed_qkv"), addr(0, "v"),
            chunk.len, 384, 128);
    }

    void emit_norm_shift(uint32_t input, uint32_t output, uint32_t scale,
                         uint32_t shift, int64_t rows) {
        // Both fusions preserve the norm's FP16 rounding before ADD. HIGH
        // uses an additive Newton payload with the same live operand slots.
        if (high_precision_fusions_) {
            rpu_launch_rhinovla_newton_norm_shift_spm_kernel(
                input, output, scale, shift, rows, eps_, NUM_CORES);
        } else if (cold_config_.expert_fusions && !high_precision_) {
            rpu_launch_pi05_adarms_norm_shift_spm_kernel(
                input, output, scale, shift, rows, eps_, NUM_CORES);
        } else {
            rpu_launch_rmsnorm_spm_kernel(input, output, scale, rows, hidden_size(), eps_,
                                        rmsnorm_route(rows, hidden_size()));
            rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                shift, output, output, rows, hidden_size(),
                c10::Half(1.0), ValuOpType::ADD, /*is_bopa=*/false);
        }
    }

    // Host preparation runs at both public forward entries, before FMB may
    // skip the host body on replay. The shape-keyed DDR slot remains stable;
    // tracked identity/version is only a content memo, never SPM residency.
    void prepare_sdpa_mask_cached(const std::optional<at::Tensor>& attention_mask,
                                 bool is_causal, int64_t seq_q, int64_t physical_kv) {
        const bool has_mask = attention_mask.has_value() && attention_mask->defined();
        const int64_t seq_k = has_mask ? Align(physical_kv, int64_t{16}) : physical_kv;
        const at::Tensor input = has_mask ? *attention_mask : at::Tensor{};
        const bool version_enabled = has_mask &&
            input.unsafeGetTensorImpl()->version_counter().enabled();
        const int64_t version = version_enabled
            ? static_cast<int64_t>(input.unsafeGetTensorImpl()->version_counter().current_version()) : -1;
        const bool hit = version_enabled && prepared_mask_input_ref_.defined() &&
            prepared_mask_input_ref_.is_same(input) && prepared_mask_version_ == version &&
            prepared_mask_q_ == seq_q && prepared_mask_k_ == seq_k &&
            prepared_mask_causal_ == is_causal;
        if (!hit) {
            // Preparation can overwrite the stable slot before failing. Retire
            // the old stamp first so retry cannot reuse its outdated content.
            prepared_mask_input_ref_ = at::Tensor{};
            prepared_mask_version_ = -1;
            auto next = sdpa_prepare_mask(attention_mask, is_causal, seq_q, seq_k,
                                          sdpa_stable_mask_cache());
            TORCH_CHECK(!version_enabled ||
                input.unsafeGetTensorImpl()->version_counter().current_version() == version,
                "RhinoVLA mask mutated while preparing its stable Graph input");
            prepared_mask_ = std::move(next);
            prepared_mask_input_ref_ = input;
            prepared_mask_version_ = version;
            prepared_mask_q_ = seq_q;
            prepared_mask_k_ = seq_k;
            prepared_mask_causal_ = is_causal;
        }
    }

    int emit_sdpa_mask(int layer_idx, const ChunkInfo& chunk,
                       int64_t seq_q, int64_t physical_kv, int64_t sdpa_kv,
                       int tp, uint32_t mask_off) {
        bool retained = false;  // non-COMPLETE callers preserve per-layer DMA
        if (prepared_mask_.mask_type == 4 && ctx().has_complete_physical_manifest()) {
            const auto& route = ctx().find_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE, RHINO_MASK_SPM_SCHEDULE_SITE, chunk.idx);
            retained = route.selector == static_cast<int64_t>(RhinoMaskSpmSchedule::FIRST_BODY_FIRST_LAYER);
            TORCH_CHECK((retained || route.selector == static_cast<int64_t>(RhinoMaskSpmSchedule::EVERY_LAYER)) &&
                        route.flags == 0 && route.arguments.size() == 9,
                        "RhinoVLA explicit mask schedule is malformed");
            TORCH_CHECK(route.arguments[6] >= 0 && route.arguments[7] >= 0 &&
                        (!retained || route.arguments[7] <= route.arguments[6]),
                        "RhinoVLA retained mask must not grow temporary SPM");
            if (ctx().body_iter == 0 && layer_idx == 0 && chunk.idx == 0) {
                ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
                    RHINO_MASK_SPM_SCHEDULE_SITE, route.selector, 0,
                    {seq_q, sdpa_kv, tp, NUM_CORES, loop_mode_ ? num_steps_ : 1,
                     num_layers(), route.arguments[6], route.arguments[7], physical_kv}, chunk.idx);
            }
        }
        if (!retained || (ctx().body_iter == 0 && layer_idx == 0 && chunk.idx == 0)) {
            sdpa_dma_mask_to_spm(prepared_mask_, mask_off, seq_q, sdpa_kv, tp);
        }
        return prepared_mask_.mask_type;
    }

    SdpaConfig make_sdpa_config(int mask = 1) const {
        return {SdpaKernelType::FLASH_ATTN_SPM, head_dim(), num_q_heads(), num_kv_heads(),
                attn_tp(), mask};
    }

    // ========================================================================
    // GEMV helper: row-partition 8-core + fused all_reduce+bias.
    //
    // Weight is the natural [3H, H] tensor row-partition swizzled so each core
    // holds K/8=H/8 columns of all 3H output rows. Layout of the final output
    // (on every core after reduce): [scale(hidden) | shift(hidden) | gate(hidden)].
    // The scale portion gets +1.0 in place so later RMSNorm sees (1+scale)
    // directly.
    //
    // Three distinct SPM buffers are required (all_reduce_sum_residual expects
    // input, residual, and output to be non-overlapping SPM regions):
    //   gemv_out_spm_addr     : final reduced+biased [3H] (broadcast to all cores)
    //   bias_temp_spm_addr    : DMA'd bias, used as the reduce-residual [3H]
    //   partial_spm_addr      : per-core partial [3H] from the row-partition GEMV
    //
    // Steps:
    //   1. DMA bias broadcast -> bias_temp (same bias on all 8 cores)
    //   2. GEMV (row partition, 8 cores): cond * dense_w -> partial
    //      Each core computes a partial [3H] along its local_k=H/8 slice.
    //   3. all_reduce_sum_residual(partial, bias) -> gemv_out
    //      Broadcast full [3H] to all cores.
    //   4. scale_slice += 1.0 (only first hidden_size elements)
    // ========================================================================
    void adarms_gemv_to_spm(uint32_t cond_spm_addr,
                            const at::Tensor& dense_w,
                            const at::Tensor& dense_b,
                            uint32_t gemv_out_spm_addr,
                            uint32_t bias_temp_spm_addr,
                            uint32_t partial_spm_addr)
    {
        int64_t h = hidden_size();
        int64_t out_features = 3 * h;

        // Step 1: DMA bias -> bias_temp (replicated across all 8 cores, acts
        // as the residual input to the fused all_reduce+bias kernel).
        rpu_launch_ddr_broadcast_spm_dma(
            dense_b.data_ptr<c10::Half>(), out_features, bias_temp_spm_addr);
        // Step 2: Row-partition GEMV. partial_spm_addr holds the per-core [3H]
        // partial along the local_k = H/NUM_CORES split.
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, RHINO_ADARMS_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            cond_spm_addr, dense_w, partial_spm_addr,
            /*M=*/1, /*N=*/out_features, /*K=*/h,
            /*partition=*/0, /*num_cores=*/NUM_CORES,
            /*bias_spm_addr=*/0);
        // Step 3: Fused all_reduce + bias. reduce(partial) + bias -> gemv_out
        // broadcast to all output cores. Input and output MUST be distinct
        // SPM regions.
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE,
            RHINO_ADARMS_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(
                /*rows=*/1, /*cols=*/out_features),
            /*resolved_flags=*/0);
        rpu_launch_all_reduce_sum_residual_kernel(
            partial_spm_addr, bias_temp_spm_addr, gemv_out_spm_addr,
            /*M=*/1, /*N=*/out_features,
            /*input_num_cores=*/NUM_CORES, /*output_num_cores=*/NUM_CORES);
        // Step 4: (1+scale) -- add +1.0 only to the first hidden_size elements
        rpu_launch_eltwise_binary_scalar_spm_kernel(
            gemv_out_spm_addr, c10::Half(1.0),
            gemv_out_spm_addr, h, ValuOpType::ADD);
    }

    void adarms_pair_gemv_to_spm(uint32_t cond_spm_addr,
                                 const at::Tensor& dense_w,
                                 const at::Tensor& dense_b,
                                 uint32_t gemv_out_spm_addr,
                                 uint32_t bias_temp_spm_addr,
                                 uint32_t partial_spm_addr)
    {
        int64_t h = hidden_size();
        int64_t out_features = 6 * h;

        rpu_launch_ddr_broadcast_spm_dma(
            dense_b.data_ptr<c10::Half>(), out_features, bias_temp_spm_addr);
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, RHINO_ADARMS_PAIR_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            cond_spm_addr, dense_w, partial_spm_addr,
            /*M=*/1, /*N=*/out_features, /*K=*/h,
            /*partition=*/0, /*num_cores=*/NUM_CORES,
            /*bias_spm_addr=*/0);
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE,
            RHINO_ADARMS_PAIR_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(
                /*rows=*/1, /*cols=*/out_features),
            /*resolved_flags=*/0);
        rpu_launch_all_reduce_sum_residual_kernel(
            partial_spm_addr, bias_temp_spm_addr, gemv_out_spm_addr,
            /*M=*/1, /*N=*/out_features,
            /*input_num_cores=*/NUM_CORES, /*output_num_cores=*/NUM_CORES);

        rpu_launch_eltwise_binary_scalar_spm_kernel(
            gemv_out_spm_addr, c10::Half(1.0),
            gemv_out_spm_addr, h, ValuOpType::ADD);
        rpu_launch_eltwise_binary_scalar_spm_kernel(
            gemv_out_spm_addr + (uint32_t)(3 * h * DWIDTH), c10::Half(1.0),
            gemv_out_spm_addr + (uint32_t)(3 * h * DWIDTH), h, ValuOpType::ADD);
    }

    void identity_adarms_pair_to_spm(uint32_t gemv_out_spm_addr)
    {
        int64_t h = hidden_size();
        rpu_launch_memset_spm_multicore(gemv_out_spm_addr, 6 * h);
        rpu_launch_eltwise_binary_scalar_spm_kernel(
            gemv_out_spm_addr, c10::Half(1.0),
            gemv_out_spm_addr, h, ValuOpType::ADD);
        rpu_launch_eltwise_binary_scalar_spm_kernel(
            gemv_out_spm_addr + (uint32_t)(3 * h * DWIDTH), c10::Half(1.0),
            gemv_out_spm_addr + (uint32_t)(3 * h * DWIDTH), h, ValuOpType::ADD);
    }

    void emit_resident_adarms_tables() {
        const int64_t h = hidden_size();
        // Whole-loop temporary storage: retaining this across model stages
        // would subtract 2.17 MiB from the Vision planner's SPM budget.
        // Reload every request, before any in-place TANH of the gate slices.
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, RHINO_ADARMS_RESIDENT_SITE,
            1, 0, {num_steps_, num_layers(), h,
                   num_steps_ * (num_layers() * 6 + 3) * h * DWIDTH});
        ctx().consume_physical_route(
            FmbRouteFamily::MUTABLE_DMA, RHINO_RESIDENT_PAIR_DMA_SITE,
            static_cast<int64_t>(RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM),
            0, {num_steps_ * num_layers() * 6 * h});
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &adarms_pair_table_src_base_, 0, num_steps_ * num_layers() * 6 * h,
            addr(0, "adarms_pair_resident"), NUM_CORES);
        ctx().consume_physical_route(
            FmbRouteFamily::MUTABLE_DMA, RHINO_RESIDENT_FINAL_DMA_SITE,
            static_cast<int64_t>(RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM),
            0, {num_steps_ * 3 * h});
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &adarms_final_table_src_base_, 0, num_steps_ * 3 * h,
            addr(0, "adarms_final_resident"), NUM_CORES);
    }

    void emit_loop_pre_layers_body() {
        const int64_t h = hidden_size();
        const int64_t ah = action_horizon_;
        const int64_t adp = action_dim_pad_;
        const int64_t sdp = state_dim_pad_;
        const int64_t bit = ctx().body_iter;
        const bool emit_static_context =
            !cold_config_.denoise_static_context_cache || bit == 0;
        const bool precomputed_adarms = cold_config_.precompute_adarms;
        const bool precomputed_time_proj = cold_config_.precompute_time_proj;
        const bool fold_action_time_in = cold_config_.fold_action_time_in;

        if (bit == 0) {
            if (full_action_w8a16_) consume_full_action_w8a16_route();
            if (high_precision_) consume_high_precision_route();
            if (cold_config_.adarms_resident) emit_resident_adarms_tables();
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA, RHINO_PRE_X_DMA_SITE,
                static_cast<int64_t>(
                    RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                /*resolved_flags=*/0);
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &x0_src_base_, 0, ah * adp, addr(0, "x_t_spm"), /*num_cores=*/1);
        }
        if (emit_static_context) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                RHINO_PRE_ACTION_MASK_DMA_SITE,
                static_cast<int64_t>(
                    RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                /*resolved_flags=*/0);
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &action_mask_src_base_, 0, ah * adp,
                addr(0, "action_mask_spm"), /*num_cores=*/1);
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA, RHINO_PRE_STATE_DMA_SITE,
                static_cast<int64_t>(
                    RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                /*resolved_flags=*/0);
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &state_src_base_, 0, sdp, addr(0, "state_spm"), /*num_cores=*/1);
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                RHINO_PRE_STATE_MASK_DMA_SITE,
                static_cast<int64_t>(
                    RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                /*resolved_flags=*/0);
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &state_mask_src_base_, 0, sdp, addr(0, "state_mask_spm"), /*num_cores=*/1);
        }
        if (!direct_action_input_ && !precomputed_time_proj) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA, RHINO_PRE_TIME_DMA_SITE,
                static_cast<int64_t>(
                    RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                /*resolved_flags=*/0);
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &cond_all_src_base_, bit * h * DWIDTH, h,
                addr(0, "time_spm"), /*num_cores=*/1);
        }
        if (!precomputed_adarms) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA, RHINO_PRE_COND_DMA_SITE,
                static_cast<int64_t>(
                    RhinoMutableDmaRoute::DDR_SCATTER_TO_SPM),
                /*resolved_flags=*/0);
            const int64_t local_k = h / NUM_CORES;
            const int64_t core_stride_bytes = local_k * DWIDTH;
            rpu_launch_ddr_scatter_spm_dma_mutable(
                &cond_all_src_base_, bit * h * DWIDTH,
                local_k, core_stride_bytes, addr(0, "cond"), NUM_CORES);
        }

        if (fold_action_time_in) {
            // Fold action_time_mlp_in(action_in_proj(x_t)) by linearity:
            // W_time_action * (W_action * x + b_action).
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR,
                RHINO_PRE_FOLDED_ACTION_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "x_t_spm"), action_time_in_w_,
                addr(0, "action_hidden_spm"),
                ah, h, adp, /*partition=*/1, /*num_cores=*/1,
                addr(0, "action_time_in_b_spm"));
        } else {
            // action_embeds = action_in_proj(x_t)
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, RHINO_PRE_ACTION_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                full_action_w8a16_ ? RHINO_LINEAR_W8A16_PER_CHANNEL : 0,
                full_io_route_arguments(ah, h, adp));
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "x_t_spm"), action_in_w_, addr(0, "action_raw_spm"),
                ah, h, adp, /*partition=*/1, /*num_cores=*/1,
                addr(0, "action_in_b_spm"), full_action_w8a16_ ? full_io_scales_[0] : at::Tensor());

            if (!direct_action_input_) {
                // action_time_mlp_in([action_embeds, time_emb]) split by linearity:
                // action part [H] + time part [H] + bias, then SiLU + mlp_out.
                ctx().consume_physical_route(
                    FmbRouteFamily::LINEAR,
                    RHINO_PRE_TIME_ACTION_LINEAR_SITE,
                    static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                    /*resolved_flags=*/0);
                rpu_launch_linear_spm_to_spm_acc16_kernel(
                    addr(0, "action_raw_spm"), time_in_action_w_,
                    addr(0, "action_hidden_spm"),
                    ah, h, h, /*partition=*/1, /*num_cores=*/1);
            }
        }
        if (!direct_action_input_) {
            if (precomputed_time_proj) {
                ctx().consume_physical_route(
                    FmbRouteFamily::MUTABLE_DMA,
                    RHINO_PRE_TIME_TABLE_DMA_SITE,
                    static_cast<int64_t>(
                        RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                    /*resolved_flags=*/0);
                rpu_launch_ddr_broadcast_spm_dma_mutable(
                    &time_proj_table_src_base_,
                    bit * h * DWIDTH,
                    h,
                    addr(0, "time_proj_spm"),
                    /*num_cores=*/1);
            } else {
                ctx().consume_physical_route(
                    FmbRouteFamily::LINEAR, RHINO_PRE_TIME_LINEAR_SITE,
                    static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                    /*resolved_flags=*/0);
                rpu_launch_linear_spm_to_spm_acc16_kernel(
                    addr(0, "time_spm"), time_in_time_w_,
                    addr(0, "time_proj_spm"),
                    1, h, h, /*partition=*/1, /*num_cores=*/1,
                    addr(0, "time_in_b_spm"));
            }
            rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                addr(0, "time_proj_spm"),
                addr(0, "action_hidden_spm"),
                addr(0, "action_hidden_spm"),
                ah, h, c10::Half(1.0), ValuOpType::ADD, /*is_bopa=*/false);
            rpu_launch_eltwise_unary_spm_kernel(
                addr(0, "action_hidden_spm"),
                addr(0, "action_hidden_spm"),
                ah * h, ValuOpType::SILU,
                /*is_gelu=*/false, /*num_cores=*/1);
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, RHINO_PRE_TIME_OUT_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "action_hidden_spm"), time_out_w_,
                addr(0, "action_raw_spm"),
                ah, h, h, /*partition=*/1, /*num_cores=*/1,
                addr(0, "time_out_b_spm"));
        }

        // mask_condition: add action_mask_proj(action_mask) to action tokens.
        if (emit_static_context) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR,
                RHINO_PRE_ACTION_MASK_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                full_action_w8a16_ ? RHINO_LINEAR_W8A16_PER_CHANNEL : 0,
                full_io_route_arguments(ah, h, adp));
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "action_mask_spm"), action_mask_w_,
                addr(0, "action_mask_hidden_spm"),
                ah, h, adp, /*partition=*/1, /*num_cores=*/1,
                addr(0, "action_mask_b_spm"), full_action_w8a16_ ? full_io_scales_[3] : at::Tensor());
        }
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "action_raw_spm"),
            addr(0, "action_mask_hidden_spm"),
            addr(0, "action_raw_spm"),
            ah * h, ValuOpType::ADD, c10::Half(1.0), /*num_cores=*/1);

        // state token = state_proj(state) + state_mask_proj(state_mask).
        if (emit_static_context) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, RHINO_PRE_STATE_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                full_action_w8a16_ ? RHINO_LINEAR_W8A16_PER_CHANNEL : 0,
                full_io_route_arguments(1, h, sdp));
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "state_spm"), state_w_, addr(0, "state_token_spm"),
                1, h, sdp, /*partition=*/1, /*num_cores=*/1,
                addr(0, "state_b_spm"), full_action_w8a16_ ? full_io_scales_[1] : at::Tensor());
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR,
                RHINO_PRE_STATE_MASK_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                full_action_w8a16_ ? RHINO_LINEAR_W8A16_PER_CHANNEL : 0,
                full_io_route_arguments(1, h, sdp));
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "state_mask_spm"), state_mask_w_,
                addr(0, "state_mask_hidden_spm"),
                1, h, sdp, /*partition=*/1, /*num_cores=*/1,
                addr(0, "state_mask_b_spm"), full_action_w8a16_ ? full_io_scales_[2] : at::Tensor());
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "state_token_spm"),
                addr(0, "state_mask_hidden_spm"),
                addr(0, "state_token_spm"),
                h, ValuOpType::ADD, c10::Half(1.0), /*num_cores=*/1);
        }

        if (get_debug_export()) {
            const std::string suffix = "_b" + std::to_string(bit);
            debug_dump_spm_per_core("rhino_vla_loop_state_token" + suffix,
                addr(0, "state_token_spm"), h, 1,
                addr_offset("state_token_spm").value);
            debug_dump_spm_per_core("rhino_vla_loop_action_hidden" + suffix,
                addr(0, "action_raw_spm"), ah * h, 1,
                addr_offset("action_raw_spm").value);
        }

        // Stage [state_token, action_tokens] into stable DDR for layer-0 input DMA.
        c10::Half* stage = action_emb_stage_.data_ptr<c10::Half>();
        rpu_launch_spm_copy_ddr_dma(
            addr(0, "state_token_spm"), stage, h);
        rpu_launch_spm_copy_ddr_dma(
            addr(0, "action_raw_spm"), stage + h, ah * h);
    }

    void emit_loop_post_layers_body() {
        const int64_t h = hidden_size();
        const int64_t ah = action_horizon_;
        const int64_t adp = action_dim_pad_;

        if (cold_config_.precompute_adarms) {
            if (!cold_config_.adarms_resident) {
                int64_t step = ctx().body_iter;
                int64_t offset = (step * 3 * h) * DWIDTH;
                ctx().consume_physical_route(
                    FmbRouteFamily::MUTABLE_DMA,
                    RHINO_POST_ADARMS_DMA_SITE,
                    static_cast<int64_t>(
                        RhinoMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                    /*resolved_flags=*/0,
                    {cold_config_.precompute_adarms ? 1 : 0});
                rpu_launch_ddr_broadcast_spm_dma_mutable(
                    &adarms_final_table_src_base_,
                    offset,
                    3 * h,
                    addr(0, "final_gemv"),
                    NUM_CORES);
            }
        } else {
            adarms_gemv_to_spm(addr(0, "cond"),
                               final_norm_w_, final_norm_b_,
                               addr(0, "final_gemv"),
                               addr(0, "bias_temp"),
                               addr(0, "gemv_partial"));
        }
        const uint32_t final_scale = cold_config_.adarms_resident
            ? addr(0, "adarms_final_resident") + ctx().body_iter * 3 * h * DWIDTH
            : addr(0, "final_gemv");
        const uint32_t final_shift = final_scale + (uint32_t)(h * DWIDTH);
        emit_norm_shift(
            addr(0, "residual1"), addr(0, "final_out_spm"),
            final_scale, final_shift, suffix_len_);

        if (get_debug_export()) {
            const std::string suffix = "_b" + std::to_string(ctx().body_iter);
            debug_dump_spm_per_core("rhino_vla_loop_final_out" + suffix,
                addr(0, "final_out_spm"), suffix_len_ * h, 1,
                addr_offset("final_out_spm").value);
        }

        // Decode only action tokens; row 0 is the state token.
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, RHINO_POST_ACTION_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            full_action_w8a16_ ? RHINO_LINEAR_W8A16_PER_CHANNEL : 0,
            full_io_route_arguments(ah, adp, h));
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "final_out_spm") + (uint32_t)(h * DWIDTH),
            action_out_w_, addr(0, "v_t_core0_spm"),
            ah, adp, h, /*partition=*/1, /*num_cores=*/1,
            addr(0, "action_out_b_spm"), full_action_w8a16_ ? full_io_scales_[5] : at::Tensor());
        // Legacy ascending flow masks velocity too. Official descending flow
        // masks only the Euler result; fractional masks make this distinction
        // observable. dt_ is immutable for the installed denoise loop.
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, RHINO_DENOISE_LOOP_SCHEDULE_SITE,
            /*resolved_selector=*/1, /*resolved_flags=*/0,
            {num_steps_, static_cast<int64_t>(dt_.x)});
        if (static_cast<float>(dt_) > 0.0f) {
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "v_t_core0_spm"),
                addr(0, "action_mask_spm"),
                addr(0, "v_t_core0_spm"),
                ah * adp, ValuOpType::MUL, c10::Half(1.0), /*num_cores=*/1);
        }

        if (get_debug_export()) {
            const std::string suffix = "_b" + std::to_string(ctx().body_iter);
            debug_dump_spm_per_core("rhino_vla_loop_v_t" + suffix,
                addr(0, "v_t_core0_spm"), ah * adp, 1,
                addr_offset("v_t_core0_spm").value);
            debug_dump_spm_per_core("rhino_vla_loop_action_mask" + suffix,
                addr(0, "action_mask_spm"), ah * adp, 1,
                addr_offset("action_mask_spm").value);
        }

        // On-device fp16 Euler + mask: x = (x + dt*v) * action_mask.
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "x_t_spm"),
            addr(0, "v_t_core0_spm"),
            addr(0, "x_t_spm"),
            ah * adp, ValuOpType::ADD, dt_, /*num_cores=*/1);
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "x_t_spm"),
            addr(0, "action_mask_spm"),
            addr(0, "x_t_spm"),
            ah * adp, ValuOpType::MUL, c10::Half(1.0), /*num_cores=*/1);
        ctx().consume_physical_route(
            FmbRouteFamily::MUTABLE_DMA, RHINO_POST_OUTPUT_DMA_SITE,
            static_cast<int64_t>(RhinoMutableDmaRoute::SPM_COPY_TO_DDR),
            /*resolved_flags=*/0);
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "x_t_spm"), &x_out_dst_base_, 0, ah * adp);
    }

    // ========================================================================
    // debug_dump_spm_per_core — record metadata for a per-core SPM dump.
    //
    // The actual readback happens OUTSIDE the graph replay. We direct-map
    // per-core SPM via SPM_ALLOC.cpu_ptr(core, buf_off) and memcpy into an
    // at::Tensor. This avoids any kernel-level scatter and is a pure
    // post-execution operation.
    //
    // Call `debug_flush_dumps()` from the Python side after the forward
    // completes and before the next forward.
    // ========================================================================
    struct DebugDumpRequest {
        std::string name;
        uint32_t buf_off;
        int64_t elems_per_core;
        int num_cores;
    };
    std::vector<DebugDumpRequest> debug_dump_requests_;

    void debug_dump_spm_per_core(const std::string& name,
                                 uint32_t /*spm_core0_addr*/,
                                 int64_t elems_per_core,
                                 int num_cores,
                                 uint32_t buf_off)
    {
        if (!get_debug_export()) return;
        DebugDumpRequest req;
        req.name = name;
        req.buf_off = buf_off;
        req.elems_per_core = elems_per_core;
        req.num_cores = num_cores;
        debug_dump_requests_.push_back(req);
    }

    // ----- Model state -----
    RhinoVLAColdConfig cold_config_;
    bool cold_config_bound_ = false;
    std::vector<LayerWeights> layer_weights_;
    bool expert_w8a16_ = false;
    bool full_action_w8a16_ = false;
    bool gates_tanh_precomputed_ = false;
    bool high_precision_ = false;
    bool high_precision_bound_ = false;
    bool vector_k_norm_ = false;
    bool vector_k_norm_bound_ = false;
    bool vector_q_norm_ = false;
    bool vector_q_norm_bound_ = false;
    bool partial_rope_ = false;
    bool partial_rope_bound_ = false;
    bool high_precision_fusions_ = false;
    bool high_precision_fusions_bound_ = false;
    std::vector<at::Tensor> full_io_scales_;
    std::vector<at::Tensor> full_cold_weights_, full_cold_scales_, full_cold_biases_;
    double eps_ = 1e-6;

    // Derived dims (computed in set_weights, cached for hot path)
    int64_t local_q_heads_ = 0;
    int64_t local_kv_dim_  = 0;

    // Per-forward tensor refs (kept alive for graph execution/replay via
    // class-member assignment; run_all_layers is synchronous so refcount
    // covers the full dispatch window). `cond_src_base_` is the live RPU
    // dev addr of cond_ref_'s data_ptr — feeds the mutable-DMA cond scatter
    // so REPLAY end() picks up the per-forward base from this storage.
    at::Tensor cond_ref_;
    uint64_t   cond_src_base_ = 0;
    at::Tensor cos_ref_;
    at::Tensor sin_ref_;
    PreparedMask prepared_mask_;
    at::Tensor prepared_mask_input_ref_;
    int64_t prepared_mask_version_ = -1, prepared_mask_q_ = -1, prepared_mask_k_ = -1;
    bool prepared_mask_causal_ = false;
    int64_t rope_position_ = -1;
    bool cond_loaded_this_forward_ = false;
    bool fast_replay_skip_layer_loop_ = false;

    // Phase 3 denoise-loop weights and mutable bases.
    bool denoise_loop_weights_ready_ = false;
    bool direct_action_input_ = false;
    bool loop_mode_ = false;
    int64_t num_steps_ = 1;
    c10::Half dt_ = c10::Half(0.0f);
    bool dt_pinned_ = false;
    int64_t action_dim_ = 0;
    int64_t action_dim_pad_ = 0;
    int64_t state_dim_ = 0;
    int64_t state_dim_pad_ = 0;
    int64_t action_horizon_ = 0;
    int64_t suffix_len_ = 0;
    at::Tensor action_in_w_, action_in_b_;
    at::Tensor action_time_in_w_, action_time_in_b_;
    at::Tensor time_in_action_w_, time_in_time_w_, time_in_b_;
    at::Tensor time_out_w_, time_out_b_;
    at::Tensor state_w_, state_b_;
    at::Tensor state_mask_w_, state_mask_b_;
    at::Tensor action_mask_w_, action_mask_b_;
    at::Tensor final_norm_w_, final_norm_b_;
    at::Tensor action_out_w_, action_out_b_;
    at::Tensor cos_loop_, sin_loop_;
    bool denoise_adarms_tables_ready_ = false;
    at::Tensor adarms_pair_table_, adarms_final_table_;
    uint64_t adarms_pair_table_src_base_ = 0;
    uint64_t adarms_final_table_src_base_ = 0;
    bool denoise_time_proj_table_ready_ = false;
    at::Tensor time_proj_table_;
    uint64_t time_proj_table_src_base_ = 0;
    at::Tensor action_emb_stage_;
    at::Tensor x0_ref_, state_ref_, state_mask_ref_, action_mask_ref_, x_out_ref_;
    uint64_t x0_src_base_ = 0;
    uint64_t cond_all_src_base_ = 0;
    uint64_t state_src_base_ = 0;
    uint64_t state_mask_src_base_ = 0;
    uint64_t action_mask_src_base_ = 0;
    uint64_t x_out_dst_base_ = 0;

};

}  // namespace v3

// =============================================================================
// Instance registry — uses ModelHandleRegistry<v3::RhinoVLAModel> template
// =============================================================================

using RhinoVLARegistry = ModelHandleRegistry<v3::RhinoVLAModel>;

std::vector<int64_t> rpu_rhino_vla_planner_cache_identity(int64_t handle) {
    return RhinoVLARegistry::get(handle, "rpu_rhino_vla_planner_cache_identity")
        ->planner_cache_identity();
}

void rpu_rhino_vla_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_chunk_envelope")
        ->set_chunk_envelope(max_kv_len, chunk);
}

void rpu_rhino_vla_bind_kvinsert_costs(
        int64_t handle, at::IntArrayRef identity,
        const std::string& catalog_sha256, at::IntArrayRef certificate_rows) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_bind_kvinsert_costs")
        ->bind_kvinsert_costs(identity, catalog_sha256, certificate_rows);
}

std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_rhino_vla_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor, int64_t site_id,
    int64_t invocation, int64_t route) {
    return RhinoVLARegistry::get(handle, "rpu_rhino_vla_kvinsert_exact_candidate")
        ->mint_kvinsert_exact_candidate(descriptor, site_id, invocation, route);
}

KvInsertCostDomainQuery rpu_rhino_vla_kvinsert_cost_domain(
        int64_t handle, at::IntArrayRef descriptor) {
    return RhinoVLARegistry::get(handle, "rpu_rhino_vla_kvinsert_cost_domain")
        ->kvinsert_cost_domain("rhino_vla", descriptor);
}

std::string rpu_rhino_vla_kvinsert_cost_catalog_sha256(int64_t handle) {
    return RhinoVLARegistry::get(
        handle, "rpu_rhino_vla_kvinsert_cost_catalog_sha256")
        ->kvinsert_cost_catalog_sha256();
}

// =============================================================================
// Public C API for TORCH_LIBRARY_IMPL wrappers (file-scope, not namespaced)
// =============================================================================

int64_t rpu_rhino_vla_create() {
    return RhinoVLARegistry::create();
}

void rpu_rhino_vla_destroy(int64_t handle) {
    RhinoVLARegistry::destroy(handle, "rpu_rhino_vla_destroy");
}

void rpu_rhino_vla_set_runtime_config(
    int64_t handle,
    bool denoise_static_context_cache,
    bool fused_adarms,
    bool skip_adarms_gemv,
    bool precompute_adarms,
    bool precompute_time_proj,
    bool fold_action_time_in,
    bool gated_no_sub,
    bool fused_silu_mul,
    bool expert_fusions,
    bool adarms_resident,
    bool packed_qkv,
    bool aligned_kv) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_runtime_config")
        ->set_runtime_config(
            denoise_static_context_cache,
            fused_adarms,
            skip_adarms_gemv,
            precompute_adarms,
            precompute_time_proj,
            fold_action_time_in,
            gated_no_sub,
            fused_silu_mul,
            expert_fusions,
            adarms_resident,
            packed_qkv,
            aligned_kv);
}

void rpu_rhino_vla_set_fast_replay(int64_t handle, bool enabled) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_fast_replay")
        ->set_fast_replay_skip_layer_loop(enabled);
}

void rpu_rhino_vla_set_high_precision(int64_t handle, bool enabled) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_high_precision")
        ->set_high_precision(enabled);
}

void rpu_rhino_vla_set_high_precision_fusions(int64_t handle, bool enabled) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_high_precision_fusions")
        ->set_high_precision_fusions(enabled);
}

void rpu_rhino_vla_set_vector_k_norm(int64_t handle, bool enabled) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_vector_k_norm")
        ->set_vector_k_norm(enabled);
}

void rpu_rhino_vla_set_vector_q_norm(int64_t handle, bool enabled) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_vector_q_norm")
        ->set_vector_q_norm(enabled);
}

void rpu_rhino_vla_set_partial_rope(int64_t handle, bool enabled) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_partial_rope")
        ->set_partial_rope(enabled);
}

at::Tensor rpu_rhino_vla_tanh_high_precision(const at::Tensor& input) {
    constexpr const char* operation = "rhino_vla_tanh_high_precision";
    RpuExecutionCoordinator::require_graph_quiescent(operation);
    RpuExecutionCoordinator::check_current_thread_execution_allowed(operation);
    TORCH_CHECK(input.defined() && input.device().type() == at::kPrivateUse1 &&
                    input.scalar_type() == at::kHalf && input.is_contiguous() &&
                    input.dim() == 2 && input.size(0) == 10 && input.size(1) == 2048,
                "RhinoVLA high precision gate TANH requires cold contiguous FP16 RPU [10,2048]");
    rpu_require_high_precision_math_kernels();
    at::Tensor output = at::empty_like(input);
    rpu_launch_eltwise_unary_kernel(input, output, ValuOpType::TANH, RpuUnaryPrecision::HIGH);
    return output;
}

void rpu_rhino_vla_set_rope_position(int64_t handle, int64_t position) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_rope_position")
        ->set_rope_position(position);
}

void rpu_rhino_vla_set_denoise_loop_weights(
    int64_t handle,
    const at::Tensor& action_in_w, const at::Tensor& action_in_b,
    const at::Tensor& action_time_in_w, const at::Tensor& action_time_in_b,
    const at::Tensor& time_in_action_w, const at::Tensor& time_in_time_w,
    const at::Tensor& time_in_b,
    const at::Tensor& time_out_w, const at::Tensor& time_out_b,
    const at::Tensor& state_w, const at::Tensor& state_b,
    const at::Tensor& state_mask_w, const at::Tensor& state_mask_b,
    const at::Tensor& action_mask_w, const at::Tensor& action_mask_b,
    const at::Tensor& final_norm_w, const at::Tensor& final_norm_b,
    const at::Tensor& action_out_w, const at::Tensor& action_out_b,
    const at::Tensor& cos, const at::Tensor& sin,
    int64_t action_dim, int64_t action_dim_pad,
    int64_t state_dim, int64_t state_dim_pad,
    int64_t action_horizon, int64_t suffix_len, bool direct_action_input) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_denoise_loop_weights")
        ->set_denoise_loop_weights(
            action_in_w, action_in_b,
            action_time_in_w, action_time_in_b,
            time_in_action_w, time_in_time_w, time_in_b,
            time_out_w, time_out_b,
            state_w, state_b,
            state_mask_w, state_mask_b,
            action_mask_w, action_mask_b,
            final_norm_w, final_norm_b,
            action_out_w, action_out_b,
            cos, sin,
            action_dim, action_dim_pad,
            state_dim, state_dim_pad,
            action_horizon, suffix_len, direct_action_input);
}

void rpu_rhino_vla_set_denoise_loop_weights_w8a16(
    int64_t handle,
    const at::Tensor& action_in_w, const at::Tensor& action_in_b,
    const at::Tensor& action_time_in_w, const at::Tensor& action_time_in_b,
    const at::Tensor& time_in_action_w, const at::Tensor& time_in_time_w,
    const at::Tensor& time_in_b,
    const at::Tensor& time_out_w, const at::Tensor& time_out_b,
    const at::Tensor& state_w, const at::Tensor& state_b,
    const at::Tensor& state_mask_w, const at::Tensor& state_mask_b,
    const at::Tensor& action_mask_w, const at::Tensor& action_mask_b,
    const at::Tensor& final_norm_w, const at::Tensor& final_norm_b,
    const at::Tensor& action_out_w, const at::Tensor& action_out_b,
    const at::Tensor& cos, const at::Tensor& sin,
    int64_t action_dim, int64_t action_dim_pad,
    int64_t state_dim, int64_t state_dim_pad,
    int64_t action_horizon, int64_t suffix_len, bool direct_action_input,
    at::TensorList io_scales) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_denoise_loop_weights_w8a16")
        ->set_denoise_loop_weights(
            action_in_w, action_in_b,
            action_time_in_w, action_time_in_b,
            time_in_action_w, time_in_time_w, time_in_b,
            time_out_w, time_out_b,
            state_w, state_b,
            state_mask_w, state_mask_b,
            action_mask_w, action_mask_b,
            final_norm_w, final_norm_b,
            action_out_w, action_out_b,
            cos, sin,
            action_dim, action_dim_pad,
            state_dim, state_dim_pad,
            action_horizon, suffix_len, direct_action_input, io_scales);
}

void rpu_rhino_vla_set_denoise_loop_adarms_tables(
    int64_t handle,
    const at::Tensor& pair_table,
    const at::Tensor& final_table) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_denoise_loop_adarms_tables")
        ->set_denoise_loop_adarms_tables(pair_table, final_table);
}

void rpu_rhino_vla_set_denoise_loop_adarms_tables_w8a16(
    int64_t handle,
    const at::Tensor& pair_table,
    const at::Tensor& final_table,
    at::TensorList cold_weights, at::TensorList cold_scales, at::TensorList cold_biases,
    bool gates_tanh_precomputed, bool high_precision) {
    TORCH_CHECK(cold_weights.size() == 19, "RhinoVLA full W8 requires nineteen cold projections");
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_denoise_loop_adarms_tables_w8a16")
        ->set_denoise_loop_adarms_tables(pair_table, final_table, cold_weights, cold_scales,
                                       cold_biases, gates_tanh_precomputed, high_precision);
}

void rpu_rhino_vla_prepare_denoise_loop_schedule(
    int64_t handle, int64_t num_steps, double dt) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_prepare_denoise_loop_schedule")
        ->prepare_denoise_loop_schedule(num_steps, dt);
}

void rpu_rhino_vla_set_denoise_loop_time_proj_table(
    int64_t handle,
    const at::Tensor& table) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_denoise_loop_time_proj_table")
        ->set_denoise_loop_time_proj_table(table);
}

void rpu_rhino_vla_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    at::TensorList attn_dense_w_list, at::TensorList attn_dense_b_list,
    at::TensorList mlp_dense_w_list,  at::TensorList mlp_dense_b_list,
    at::TensorList pair_dense_w_list, at::TensorList pair_dense_b_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps)
{
    RhinoVLARegistry::get(handle, "rpu_rhino_vla")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        gate_list, up_list, down_list,
        attn_dense_w_list, attn_dense_b_list,
        mlp_dense_w_list,  mlp_dense_b_list,
        pair_dense_w_list, pair_dense_b_list,
        q_norm_list, k_norm_list,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size, eps);
}

void rpu_rhino_vla_set_weights_w8a16(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    at::TensorList attn_dense_w_list, at::TensorList attn_dense_b_list,
    at::TensorList mlp_dense_w_list, at::TensorList mlp_dense_b_list,
    at::TensorList pair_dense_w_list, at::TensorList pair_dense_b_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps,
    at::TensorList q_w_scale_list, at::TensorList k_w_scale_list,
    at::TensorList v_w_scale_list, at::TensorList o_w_scale_list,
    at::TensorList gate_scale_list, at::TensorList up_scale_list,
    at::TensorList down_scale_list) {
    TORCH_CHECK(!q_w_scale_list.empty(),
                "rhino_vla_set_weights_w8a16 requires all seven scale lists");
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_weights_w8a16")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        gate_list, up_list, down_list,
        attn_dense_w_list, attn_dense_b_list,
        mlp_dense_w_list, mlp_dense_b_list,
        pair_dense_w_list, pair_dense_b_list,
        q_norm_list, k_norm_list,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size, eps,
        q_w_scale_list, k_w_scale_list, v_w_scale_list, o_w_scale_list,
        gate_scale_list, up_scale_list, down_scale_list);
}

void rpu_rhino_vla_set_weights_full_w8a16(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    at::TensorList attn_dense_w_list, at::TensorList attn_dense_b_list,
    at::TensorList mlp_dense_w_list, at::TensorList mlp_dense_b_list,
    at::TensorList pair_dense_w_list, at::TensorList pair_dense_b_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps,
    at::TensorList q_w_scale_list, at::TensorList k_w_scale_list,
    at::TensorList v_w_scale_list, at::TensorList o_w_scale_list,
    at::TensorList gate_scale_list, at::TensorList up_scale_list,
    at::TensorList down_scale_list,
    at::TensorList attn_dense_scale_list, at::TensorList mlp_dense_scale_list,
    at::TensorList pair_dense_scale_list) {
    TORCH_CHECK(!attn_dense_scale_list.empty(), "RhinoVLA full W8 requires AdaRMS scales");
    TORCH_CHECK(!q_w_scale_list.empty(),
                "rhino_vla_set_weights_w8a16 requires all seven scale lists");
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_weights_full_w8a16")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        gate_list, up_list, down_list,
        attn_dense_w_list, attn_dense_b_list,
        mlp_dense_w_list, mlp_dense_b_list,
        pair_dense_w_list, pair_dense_b_list,
        q_norm_list, k_norm_list,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size, eps,
        q_w_scale_list, k_w_scale_list, v_w_scale_list, o_w_scale_list,
        gate_scale_list, up_scale_list, down_scale_list,
        attn_dense_scale_list, mlp_dense_scale_list, pair_dense_scale_list);
}

at::Tensor rpu_rhino_vla_forward(
    int64_t handle,
    const at::Tensor& hidden_states,
    const at::Tensor& cond,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position,
    bool is_causal,
    at::IntArrayRef planned_stage_descriptor)
{
    // TensorList -> std::vector<at::Tensor> (shallow copy; tensors are refcounted)
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());

    return RhinoVLARegistry::get(handle, "rpu_rhino_vla")->forward(
        hidden_states, cond, k_caches, v_caches,
        cos, sin, attention_mask, position, is_causal,
        planned_stage_descriptor);
}

at::Tensor rpu_rhino_vla_denoise_loop_forward(
    int64_t handle,
    const at::Tensor& x0_rpu,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const at::Tensor& cond_all,
    const at::Tensor& attention_mask,
    const at::Tensor& state_rpu,
    const at::Tensor& state_mask_rpu,
    const at::Tensor& action_mask_rpu,
    at::Tensor x_out,
    double dt,
    int64_t prefix_len,
    int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor) {
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return RhinoVLARegistry::get(handle, "rpu_rhino_vla_denoise_loop_forward")
        ->denoise_loop_forward(
            x0_rpu, k_caches, v_caches, cond_all, attention_mask,
            state_rpu, state_mask_rpu, action_mask_rpu, x_out,
            dt, prefix_len, num_steps, planned_stage_descriptor);
}

void rpu_rhino_vla_set_chunk_size(int64_t handle, int64_t chunk_size) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_chunk_size")
        ->set_configured_chunk_size(chunk_size);
}

void rpu_rhino_vla_enable_execution_reconfigure(int64_t handle) {
    RhinoVLARegistry::get(
        handle, "rpu_rhino_vla_enable_execution_reconfigure")
        ->enable_execution_reconfigure_guard();
}

void rpu_rhino_vla_stage_chunk_size(
        int64_t handle, int64_t token, int64_t chunk_size) {
    TORCH_CHECK(token > 0,
                "RhinoVLA action hot-reconfigure token must be positive");
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_stage_chunk_size")
        ->stage_configured_chunk_size(
            static_cast<uint64_t>(token), chunk_size);
}

int64_t rpu_rhino_vla_get_resolved_chunk_size(int64_t handle) {
    return RhinoVLARegistry::get(
               handle, "rpu_rhino_vla_get_resolved_chunk_size")
        ->get_last_resolved_chunk_size();
}

std::vector<int64_t> rpu_rhino_vla_resolve_action_stage_domain(
        int64_t handle, int64_t execution_len, int64_t logical_len,
        int64_t position, int64_t kv_len, bool use_attention_mask,
        bool is_causal, int64_t requested_chunk_size) {
    return RhinoVLARegistry::get(
               handle, "rpu_rhino_vla_resolve_action_stage_domain")
        ->resolve_action_stage_domain(
            execution_len, logical_len, position, kv_len,
            use_attention_mask, is_causal,
            requested_chunk_size);
}

// Post-forward helper: materialize per-core SPM dumps requested during the
// graph build (only active when debug export is enabled). This MUST be
// called after a forward finishes; it reads SPM via direct CPU-mapped
// pointers and populates g_debug_tensors, which Python reads via
// rpu_backend.get_debug_tensor(name).
void rpu_rhino_vla_debug_flush_dumps(int64_t handle) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_debug_flush_dumps")->debug_flush_dumps();
}
