// rpu_qwen3_model.h — extracted v3::CausalDecoderModel (Task 3 of the wall_oss
// fused-denoise plan). The class + its 4 former rpu_qwen3_model.cpp-local constants
// are moved here VERBATIM (zero behavior change) so subclasses (WallOssActionStepModel)
// can inherit the plain Qwen2.5 decoder body. rpu_qwen3_model.cpp now #includes this
// header and keeps only the registry + C-API free functions.
//
// NOTE: the inline method bodies were written against `using namespace at;` and
// `using namespace ::rhino_lkn;` (unqualified Align/CeilDiv/RpuGetDevAddr). To preserve
// the verbatim move WITHOUT polluting the global namespace of includers, those
// using-directives are scoped INSIDE `namespace v3` below.
#pragma once

#include "fused_model_base.h"
#include "execution_topology.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_spm_buffers.h"
#include "rpu_spm_residency.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_kvinsert_segment_plan.h"
#include "rpu_linear_tiling.h"
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <c10/util/ScopeExit.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <typeinfo>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <tuple>
#include <utility>
#include <string>

#include "rpu_runtime_state.h"  // shared debug and cross-layer runtime state

namespace v3 {

struct SpmDense2DSpec;
class SpmPortView;
class SpmPortId;

// Namespace-scoped using-directives (see header note): preserve unqualified-name lookup
// for the verbatim-moved inline bodies without leaking into the global namespace.
using namespace at;
using namespace ::rhino_lkn;

// Former rpu_qwen3_model.cpp file-local constants (#define NUM_CORES/DWIDTH + two
// constexpr), now `inline constexpr` so the includers share one definition. Values verbatim.
inline constexpr int NUM_CORES = 8;
inline constexpr int DWIDTH    = 2;

// Stable semantic call-site identities from planner_owners.v1.json.  These
// identify the ordinary CausalDecoderModel layer-body sites, independent of
// the concrete launcher selected by a COMPLETE physical descriptor.
inline constexpr int64_t CAUSAL_DECODER_KV_INSERT_SITE =
    6070783309895644632LL;
inline constexpr int64_t CAUSAL_DECODER_MASKED_ATTENTION_SITE =
    2102573421850928983LL;
inline constexpr int64_t CAUSAL_DECODER_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED = 2;
inline constexpr int64_t CAUSAL_DECODER_MASK_SPM_SCHEDULE_SITE = 301270362681343383LL;
enum class CausalMaskSpmSchedule : int64_t {
    EVERY_LAYER = 1,
    FIRST_BODY_FIRST_LAYER = 2,
};

inline constexpr int64_t CAUSAL_DECODER_ROPE_1D_SITE = 3305504109428563154LL;
inline constexpr int64_t CAUSAL_DECODER_PARTIAL_MROPE_TENSOR_SITE =
    3695191896991652915LL;
inline constexpr int64_t CAUSAL_DECODER_PARTIAL_MROPE_POINTER_SITE =
    7486129449658149720LL;
inline constexpr int64_t CAUSAL_DECODER_MROPE_TENSOR_SITE =
    8192467531688182946LL;
inline constexpr int64_t CAUSAL_DECODER_MROPE_POINTER_SITE =
    1285209676163169104LL;
inline constexpr int64_t CAUSAL_DECODER_ADARMS_MUTABLE_SITE =
    8107707913403098115LL;

inline constexpr int64_t CAUSAL_DECODER_KV_FIRST_Q_SITE =
    1963944858872995083LL;
inline constexpr int64_t CAUSAL_DECODER_KV_FIRST_K_SITE =
    3178756187656776518LL;
inline constexpr int64_t CAUSAL_DECODER_KV_FIRST_V_SITE =
    6814956996628673573LL;
inline constexpr int64_t CAUSAL_DECODER_KV_FIRST_KV_SITE =
    3361545584592853925LL;

inline constexpr int64_t CAUSAL_DECODER_KV_FIRST_ATTN_SITE =
    4507982168063975685LL;
inline constexpr int64_t CAUSAL_DECODER_KV_FIRST_PREPARE_SITE =
    5596699635173740728LL;
inline constexpr int64_t CAUSAL_DECODER_KV_FIRST_O_SITE =
    1455889208566655265LL;
inline constexpr int64_t CAUSAL_DECODER_KV_FIRST_REDUCE_SITE =
    6940573292902316011LL;
inline constexpr int64_t CAUSAL_DECODER_KV_FIRST_DEEPSTACK_SITE =
    3361145283719152462LL;
inline constexpr int64_t CAUSAL_DECODER_KV_FIRST_LM_HEAD_SITE =
    2051550011469097529LL;
inline constexpr int64_t CAUSAL_DECODER_KV_FIRST_LM_HEAD_DMA_SITE =
    4026804102159716270LL;

inline constexpr int64_t CAUSAL_DECODER_Q_SITE = 403971716841024546LL;
inline constexpr int64_t CAUSAL_DECODER_K_SITE = 4795854722057375216LL;
inline constexpr int64_t CAUSAL_DECODER_V_SITE = 6513089886571723927LL;
inline constexpr int64_t CAUSAL_DECODER_RAW_SPM_ATTN_SITE =
    1958683583016123762LL;
inline constexpr int64_t CAUSAL_DECODER_PREPARE_SITE =
    6944851813645512725LL;
inline constexpr int64_t CAUSAL_DECODER_O_SITE = 8202287092212169288LL;
inline constexpr int64_t CAUSAL_DECODER_REDUCE_SITE =
    6623643466707134842LL;
inline constexpr int64_t CAUSAL_DECODER_DEEPSTACK_SITE =
    532029389006794381LL;
inline constexpr int64_t CAUSAL_DECODER_LM_HEAD_SITE =
    8243380633913809255LL;
inline constexpr int64_t CAUSAL_DECODER_LM_HEAD_DMA_SITE =
    7937005304416076778LL;
inline constexpr int64_t QWEN3VL_4B_DECODE_QK_NORM_MROPE_SITE =
    3063655359104379742LL;
inline constexpr int64_t QWEN3VL_4B_DECODE_QK_NORM_MROPE_KV_INSERT_SITE =
    3722835822343098908LL;
inline constexpr int64_t QWEN3VL_4B_DECODE_O_RING_NORM_SITE =
    3714126432099747378LL;
inline constexpr int64_t QWEN3VL_4B_PREFILL_GATEUP_SITE =
    5140288894752052630LL;
inline constexpr int64_t QWEN3VL_4B_PREFILL_DOWN_SITE =
    1441787235352378190LL;
inline constexpr int64_t QWEN3VL_4B_PREFILL_MLP_REDUCE_SITE =
    8917083250595985403LL;
// Preserve the original route IDs; unfused decode MLP arguments now carry
// the admitted 2B or 4B geometry. The combined Gate/Up payload stays 4B-only.
inline constexpr int64_t QWEN3VL_4B_DECODE_MLP_GATE_SITE =
    2531548710636001806LL;
inline constexpr int64_t QWEN3VL_4B_DECODE_MLP_GATEUP_FUSED_SITE =
    2455230034228367297LL;
inline constexpr int64_t QWEN3VL_4B_DECODE_MLP_UP_SITE =
    1067125770787623802LL;
inline constexpr int64_t QWEN3VL_4B_DECODE_MLP_ACTIVATION_SITE =
    4257145402999266975LL;
inline constexpr int64_t QWEN3VL_4B_DECODE_MLP_DOWN_SITE =
    8611354113246523844LL;
inline constexpr int64_t QWEN3VL_4B_DECODE_MLP_REDUCE_SITE =
    6953872594742217143LL;

// Persistent preload routes.  These are part of the same COMPLETE descriptor
// as the layer body: Graph BUILD and REPLAY execute the callbacks inside the
// captured op stream, so their branch/shape authority cannot remain local.
inline constexpr int64_t CAUSAL_DECODER_INPUT_NORM_POOLER_PRELOAD_SITE =
    3935150780636094510LL;
inline constexpr int64_t CAUSAL_DECODER_INPUT_NORM_DIRECT_PRELOAD_SITE =
    5681769358418745203LL;
inline constexpr int64_t CAUSAL_DECODER_POST_NORM_POOLER_PRELOAD_SITE =
    3473869795228497796LL;
inline constexpr int64_t CAUSAL_DECODER_POST_NORM_DIRECT_PRELOAD_SITE =
    7884695408020680785LL;
inline constexpr int64_t CAUSAL_DECODER_ADARMS_INPUT_SHIFT_PRELOAD_SITE =
    6775618881931855491LL;
inline constexpr int64_t CAUSAL_DECODER_ADARMS_POST_SHIFT_PRELOAD_SITE =
    2212764701262785194LL;
inline constexpr int64_t CAUSAL_DECODER_Q_NORM_POOLER_PRELOAD_SITE =
    1399000759814080097LL;
inline constexpr int64_t CAUSAL_DECODER_Q_NORM_DIRECT_PRELOAD_SITE =
    8890152244259718343LL;
inline constexpr int64_t CAUSAL_DECODER_K_NORM_POOLER_PRELOAD_SITE =
    4679467455528904800LL;
inline constexpr int64_t CAUSAL_DECODER_K_NORM_DIRECT_PRELOAD_SITE =
    703369264709138330LL;
inline constexpr int64_t CAUSAL_DECODER_QKV_BIAS_PRELOAD_SITE =
    2415264593810463607LL;
inline constexpr int64_t CAUSAL_DECODER_NVFP4_SCALE_PRELOAD_SITE =
    5605101652280608153LL;
inline constexpr int64_t CAUSAL_DECODER_FINAL_NORM_POOLER_PRELOAD_SITE =
    2840945300572718145LL;
inline constexpr int64_t CAUSAL_DECODER_FINAL_NORM_DIRECT_PRELOAD_SITE =
    4330740528604353982LL;

// Non-launch topology decisions carried by the same descriptor.  Site identity
// distinguishes the setting; arguments are the complete cold configuration.
inline constexpr int64_t CAUSAL_DECODER_POOLER_TOPOLOGY_SITE =
    6670355129004946231LL;
inline constexpr int64_t CAUSAL_DECODER_ADARMS_TOPOLOGY_SITE =
    4043535920254917841LL;
inline constexpr int64_t CAUSAL_DECODER_QK_NORM_TOPOLOGY_SITE =
    8857035797511211633LL;
inline constexpr int64_t CAUSAL_DECODER_QKV_BIAS_TOPOLOGY_SITE =
    7465994957384313179LL;
inline constexpr int64_t CAUSAL_DECODER_NVFP4_TOPOLOGY_SITE =
    3693225195438645897LL;
inline constexpr int64_t CAUSAL_DECODER_DEEPSTACK_TOPOLOGY_SITE =
    7158328860829261528LL;
inline constexpr int64_t CAUSAL_DECODER_LM_HEAD_TOPOLOGY_SITE =
    3145128204432384004LL;
// Controlled 4B W8 ordinary prefill uses the shared chunk-major traversal.
inline constexpr int64_t CAUSAL_DECODER_4B_W8_PREFILL_SCHEDULE_SITE =
    4913775020614323787LL;
// Schedule-only migration: ordinary 2B W8 and 2B/4B FP16/W4 keep their
// original projection, activation and accumulation routes.
inline constexpr int64_t CAUSAL_DECODER_VL_PREFILL_SCHEDULE_SITE =
    1348148692389361716LL;
inline constexpr int64_t CAUSAL_DECODER_4B_W8_PACKED_QKV_SITE =
    1079292426267549655LL;

// Exact controlled delivery sites, represented in every COMPLETE descriptor.
inline constexpr int64_t QWEN3VL_DELIVERY_TOPOLOGY_SITE = 7689168168149885712LL;
inline constexpr int64_t QWEN3VL_DELIVERY_NORM_SLAB_SITE = 7657073863368167104LL;
inline constexpr int64_t QWEN3VL_DELIVERY_RMS_SITE = 1596693722638272420LL;
inline constexpr int64_t QWEN3VL_DELIVERY_QKV_SITE = 8283348924956943329LL;
inline constexpr int64_t QWEN3VL_DELIVERY_ROPE_SITE = 9014667968434229459LL;
inline constexpr int64_t QWEN3VL_DELIVERY_MLP_LINEAR_SITE = 5098457485624789093LL;
constexpr int64_t QWEN3VL_DELIVERY_MLP_DOWN_SITE = 2912376799355870491LL;
constexpr int64_t QWEN3VL_DELIVERY_MLP_DECODE_ACTIVATION_SITE = 8268150559597324482LL;
constexpr int64_t QWEN3VL_DELIVERY_KV_DECODE_PAIR_SITE = 5003444885314301181LL;
inline constexpr int64_t QWEN3VL_DELIVERY_MLP_ACTIVATION_SITE = 1837157632576326358LL;
inline constexpr int64_t QWEN3VL_DELIVERY_MLP_REDUCE_SITE = 957155555766991356LL;
inline constexpr int64_t QWEN3VL_DELIVERY_KV_PAIR_SITE = 24242269098580500LL;
inline constexpr int64_t QWEN3VL_DELIVERY_LM_HIDDEN_DMA_SITE = 1179089008795781732LL;

// Ordinary direct-SPM normalization routes are downstream choices of
// the already-owned compute chunks, not independent planner search axes.
// Keep these sites out of the generated owner inventory.
inline constexpr int64_t RHINOVLA_TEXT_HIGH_RMSNORM_SITE = 6934604613252246886LL;
inline constexpr int64_t RHINOVLA_TEXT_HIGH_ACTIVATION_SITE = 5910322037581299718LL;
inline constexpr int64_t CAUSAL_DECODER_ORDINARY_RMSNORM_SITE =
    1776983747673121799LL;

enum class CausalDecoderAllReduceRoute : int64_t {
    PREPARE_RING_INPUT = 3,
};

enum class CausalDecoderRopeRoute : int64_t {
    ROPE_1D = 1,
    MROPE = 2,
    PARTIAL_MROPE = 3,
};

enum class CausalDecoderMutableDmaRoute : int64_t {
    DDR_BROADCAST_TO_SPM = 1,
    SPM_SCATTER_TO_DDR = 2,
    DDR_SCATTER_TO_SPM = 3,
};

enum class CausalDecoderGraphScheduleRoute : int64_t {
    TOPOLOGY = 1,
};

enum class CausalDecoderLinearRoute : int64_t {
    AUTO_TILE = static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
    GEMV = static_cast<int64_t>(FmbLinearRouteSelector::GEMV),
    ROW_WEIGHT_REUSE = static_cast<int64_t>(FmbLinearRouteSelector::ROW_WEIGHT_REUSE),
};

enum class CausalDecoderPlanningMode : int64_t {
    ORDINARY = 0,
    WALL_Z1 = 1,
    QWEN3VL_POOLER_Z1 = 2,
    QWEN3VL_MULTIVIEW_TEXT = 3,
};

// CAUSAL_DECODER_FIXED_KERNEL_BASIS: remaining non-manifest launchers implement
// fixed normalization/activation, preload/cache bookkeeping, or BufferDecl
// transport. A selectable implementation must first become a typed route.

inline int64_t causal_ring_route(int64_t rows, int64_t cols) {
    return fmb_ring_all_reduce_route_selector(rows, cols);
}

inline std::vector<int64_t> causal_deepstack_topology_arguments(
    const std::vector<int64_t>& layers) {
    std::vector<int64_t> arguments;
    arguments.reserve(layers.size() + 1);
    arguments.push_back(static_cast<int64_t>(layers.size()));
    arguments.insert(arguments.end(), layers.begin(), layers.end());
    return arguments;
}

// Phase 2.5: lm_head 输出 SPM buffer 大小上限.
// declare_buffers 在 fuse_lm_head_ 状态未知时被框架调用 (ensure_allocated 在 set_lm_head
// 之前可能已经发生过), 所以我们用一个固定上限分配 logits buffer, 避免引入跨调用
// 顺序的 SPM 重分配复杂度。
//   Qwen3 系列 vocab 一致 = 151936 (0.6B/1.7B/4B/8B/14B/32B)
//   per-core: 151936/8 = 18992 halves ≈ 38 KB
//   上限设 200000 (per-core 25000 halves = 50 KB), 给未来更大 vocab 模型留余量
// Eight-core head reserve: 50 KB/core; reduced-core TP4 reserves 100 KB/core.
inline constexpr int64_t QWEN3_FUSED_LM_HEAD_MAX_VOCAB = 200000;

inline int resolve_causal_decoder_lm_head_tp(int core_count, int64_t vocab_size) {
    TORCH_CHECK(core_count == 8 || core_count == 6 || core_count == 4,
                "causal decoder admits the default eight-core or reduced-core Qwen3 profile");
    TORCH_CHECK(vocab_size >= 0, "causal decoder configured vocab_size must be nonnegative");
    TORCH_CHECK(core_count == 8 || vocab_size == 151936,
                "reduced-core Qwen3 profile requires the actual Qwen3 vocab_size=151936 at creation");
    // The reserve ceiling describes capacity, not model geometry. TP remains
    // bound even when the optional fused weight is absent or later cleared.
    return core_count == 8 ? 8 : 4;
}

// AdaRMS fused scale/shift broadcast is a cold planner capability.
// Each role has a contiguous [num_layers, hidden] DDR slab. reverse_layer_alloc
// makes the corresponding persistent SPM slots ascend in the same layer order,
// so a whole-role broadcast can replace individual per-layer transfers.
// The transformation changes data movement, not arithmetic.
//
// M-RoPE position IDs use a stable [seq_len,3] INT32 DDR buffer indexed by
// ctx().position + chunk.offset. Allocate capacity for the cache envelope and
// refresh the relevant rows each forward. Only models with mrope_section need
// this buffer; plain Qwen3/Llama paths do not allocate it.
inline constexpr int64_t QWEN3_MROPE_MAX_KEEPALIVE_SEQ = 8192;

// =============================================================================
// CausalDecoderModel — v3::FusedModelBase subclass (Qwen3 + Llama; future Phi/Mistral)
// =============================================================================

class CausalDecoderModel : public FusedModelBase {
public:
    // Fixed model-owned auxiliary storages, exposed only for cold ownership
    // binding. This is not a complete weight inventory or replay permission.
    struct DecodeAuxiliaryTensorOwners {
        uint64_t model_state_generation = 0;
        at::Tensor packed_qkv_scales;
        at::Tensor mrope_position_ids;
        // Layer-major Gate, Up. Empty unless the native fused decode leaf
        // retained its independently padded scale storages.
        std::vector<at::Tensor> gate_up_scales;
    };

    DecodeAuxiliaryTensorOwners decode_auxiliary_tensor_owners() const {
        TORCH_CHECK(!RpuKernelGraph::has_active(),
                    "decode auxiliary owners require cold binding outside Graph capture");
        TORCH_CHECK(num_layers() > 0,
                    "decode auxiliary owners require installed model weights");
        DecodeAuxiliaryTensorOwners owners{
            installed_model_state_generation(),
            qwen3vl_4b_qkv_scale_bank_, position_ids_keepalive_, {}};
        if (qwen3vl_4b_decode_gateup_fused_) {
            owners.gate_up_scales.reserve(layer_weights_.size() * 2);
            for (const auto& weights : layer_weights_) {
                owners.gate_up_scales.push_back(weights.gate_ws);
                owners.gate_up_scales.push_back(weights.up_ws);
            }
        }
        return owners;
    }

    std::vector<int64_t> execution_topology() const {
        TORCH_CHECK(num_layers() > 0,
                    "decoder topology requires installed model weights");
        return {num_cores(), attn_tp(), mlp_tp(), lm_head_tp(), NUM_CORES};
    }

    struct LayerWeights {
        at::Tensor q_w, k_w, v_w, o_w;
        // Exact 2B delivery owns a physical concatenation; controlled 4B
        // decode can instead retain a view of adjacent already-swizzled Q/K/V.
        at::Tensor qkv_w;
        at::Tensor q_norm_w, k_norm_w;
        at::Tensor input_norm_w, post_norm_w;
        at::Tensor gate_w, up_w, down_w;
        at::Tensor gate_up_w;
        // Optional QKV bias (Qwen2.5 / Wall-OSS-0.5). Undefined when
        // has_qkv_bias_ == false (Qwen3 / Llama path).
        at::Tensor q_bias_w, k_bias_w, v_bias_w;
        at::Tensor q_ws, k_ws, v_ws, o_ws, gate_ws, up_ws, down_ws;
        at::Tensor qkv_ws, gate_up_ws;
    };

    // codex HIGH-1+HIGH-2 (Phase 06.1 / D-6-17): explicit qk-norm presence boolean.
    // Qwen3 set_weights populates q_norm_list / k_norm_list -> has_qk_norm_=true.
    // Llama set_weights passes empty q_norm_list / k_norm_list -> has_qk_norm_=false.
    // build_layer_subgraph branches on this explicit flag (not the
    // (nkv >= NUM_CORES) heuristic which incorrectly fired for Llama nkv==8).
    bool has_qk_norm_ = false;

    // Optional QKV bias presence (Qwen2.5 / Wall-OSS-0.5). All three bias lists
    // empty -> has_qkv_bias_=false (Qwen3 / Llama, unchanged). All three size==N
    // -> true: a persistent per-layer SPM bias buffer is scattered per col-partition
    // and its addr is passed to the Q/K/V linear launchers (no GEMM-side add).
    bool has_qkv_bias_ = false;

    // RhinoVLA fast-replay opt-in (per-model; flows into ModelStaticConfig,
    // honored by run_all_layers). Default false → qwen3/llama/wall_oss unaffected.
    bool fast_replay_skip_layer_loop_ = false;
    bool preload_replay_skip_ = false;
    // Ordinary Qwen3-VL initial prefill may use the existing FMB cursor skip,
    // but only when its KV owners match the last body emitted for that graph.
    // The graph signature does not contain KV addresses; keep this small owner
    // witness here so a same-shape replacement cache falls back to normal
    // emission instead of replaying registers bound to the old cache. Bind the
    // witness to one graph capture: another signature may still contain the
    // previous cache addresses even after this model has used the new cache.
    struct FastReplayOwner {
        at::Tensor tensor;  // retain the owner; raw pointer equality is not ABA-safe
        const c10::TensorImpl* impl = nullptr;
        const c10::StorageImpl* storage = nullptr;
        at::ScalarType dtype = at::ScalarType::Undefined;
        std::vector<int64_t> sizes;
        std::vector<int64_t> strides;
        int64_t storage_offset = 0;
        uint64_t address = 0;
        size_t storage_bytes = 0;
    };
    std::vector<FastReplayOwner> fast_replay_k_owners_;
    std::vector<FastReplayOwner> fast_replay_v_owners_;
    GraphOpStreamStamp fast_replay_prefill_stamp_, fast_replay_bound_stamp_;
    uint64_t fast_replay_prefill_lifetime_ = 0, fast_replay_bound_lifetime_ = 0;
    uint64_t fast_replay_prefill_generation_ = 0, fast_replay_bound_generation_ = 0;
    bool fast_replay_prefill_candidate_ = false;
    bool fast_replay_owner_bound_ = false;

    static FastReplayOwner fast_replay_owner(const at::Tensor& tensor) {
        if (!tensor.defined() ||
            tensor.device().type() != c10::DeviceType::PrivateUse1 ||
            tensor.layout() != c10::Layout::Strided || !tensor.is_contiguous()) {
            return {};
        }
        return {tensor,
                tensor.unsafeGetTensorImpl(),
                tensor.storage().unsafeGetStorageImpl(),
                tensor.scalar_type(),
                tensor.sizes().vec(),
                tensor.strides().vec(),
                tensor.storage_offset(),
                static_cast<uint64_t>(::rhino_lkn::RpuGetDevAddr(tensor.data_ptr())),
                tensor.storage().nbytes()};
    }

    static bool fast_replay_owner_equal(const FastReplayOwner& lhs,
                                        const at::Tensor& rhs) {
        const auto current = fast_replay_owner(rhs);
        return lhs.impl != nullptr && current.impl == lhs.impl &&
            current.storage == lhs.storage && current.address == lhs.address &&
            current.storage_bytes == lhs.storage_bytes &&
            current.dtype == lhs.dtype &&
            current.sizes == lhs.sizes && current.strides == lhs.strides &&
            current.storage_offset == lhs.storage_offset;
    }

    void prepare_fast_replay_prefill_owners(at::TensorList k_caches,
                                            at::TensorList v_caches,
                                            bool candidate) {
        fast_replay_prefill_candidate_ = false;
        fast_replay_owner_bound_ = false;
        if (!candidate || k_caches.size() != v_caches.size() ||
            k_caches.empty() || !RpuKernelGraph::has_active()) {
            return;
        }
        auto& graph = RpuKernelGraph::active();
        if (!graph.replayable() ||
            (graph.state() != RpuKernelGraph::State::RECORDING &&
             graph.state() != RpuKernelGraph::State::REPLAYING)) return;
        fast_replay_prefill_stamp_ = graph.op_stream_stamp();
        fast_replay_prefill_lifetime_ = graph.debug_stats().graph_lifetime_id;
        fast_replay_prefill_generation_ = installed_model_state_generation();
        fast_replay_prefill_candidate_ = true;
        if (graph.state() == RpuKernelGraph::State::REPLAYING &&
            fast_replay_bound_lifetime_ == fast_replay_prefill_lifetime_ &&
            fast_replay_bound_generation_ == fast_replay_prefill_generation_ &&
            graph.is_op_stream_stamp_current(fast_replay_bound_stamp_) &&
            fast_replay_bound_stamp_.position == fast_replay_prefill_stamp_.position &&
            fast_replay_k_owners_.size() == k_caches.size() &&
            fast_replay_v_owners_.size() == v_caches.size()) {
            bool same = true;
            for (size_t i = 0; i < k_caches.size() && same; ++i) {
                same = fast_replay_owner_equal(fast_replay_k_owners_[i], k_caches[i]) &&
                    fast_replay_owner_equal(fast_replay_v_owners_[i], v_caches[i]);
            }
            fast_replay_owner_bound_ = same;
        }
    }

    void commit_fast_replay_prefill_owners(at::TensorList k_caches,
                                           at::TensorList v_caches) {
        if (!fast_replay_prefill_candidate_ ||
            k_caches.size() != v_caches.size() || k_caches.empty() ||
            !RpuKernelGraph::has_active()) return;
        auto& graph = RpuKernelGraph::active();
        if (graph.debug_stats().graph_lifetime_id != fast_replay_prefill_lifetime_ ||
            installed_model_state_generation() != fast_replay_prefill_generation_ ||
            !graph.is_op_stream_stamp_current(fast_replay_prefill_stamp_)) return;
        fast_replay_k_owners_.clear();
        fast_replay_v_owners_.clear();
        fast_replay_k_owners_.reserve(k_caches.size());
        fast_replay_v_owners_.reserve(v_caches.size());
        for (size_t i = 0; i < k_caches.size(); ++i) {
            fast_replay_k_owners_.push_back(fast_replay_owner(k_caches[i]));
            fast_replay_v_owners_.push_back(fast_replay_owner(v_caches[i]));
        }
        fast_replay_bound_stamp_ = fast_replay_prefill_stamp_;
        fast_replay_bound_lifetime_ = fast_replay_prefill_lifetime_;
        fast_replay_bound_generation_ = fast_replay_prefill_generation_;
        fast_replay_owner_bound_ = true;
    }

    // Batch decode: KV-cache slot for a batch==1 forward (batched prefill).
    // Stamped per forward; consumed by build_layer_subgraph.
    int64_t cache_batch_slot_ = 0;
    bool batch_decode_active_ = false;
    void set_fast_replay_skip_layer_loop(bool e) {
        if (fast_replay_skip_layer_loop_ != e) {
            fast_replay_skip_layer_loop_ = e;
            fast_replay_k_owners_.clear();
            fast_replay_v_owners_.clear();
            fast_replay_bound_stamp_ = {};
            fast_replay_prefill_stamp_ = {};
            fast_replay_prefill_candidate_ = false;
            fast_replay_owner_bound_ = false;
            invalidate_model_state();
        }
    }
    void set_preload_replay_skip(bool e) {
        if (preload_replay_skip_ != e) {
            preload_replay_skip_ = e;
            invalidate_model_state();
        }
    }
    void set_configured_chunk_size_cap(int64_t cs) {
        TORCH_CHECK(cs == 0 || cs >= 16,
                    "configured chunk_size cap must be 0 or at least 16, got ", cs);
        TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                    "configured chunk_size cap must be set before the first forward");
        configured_chunk_size_cap_ = cs;
        set_chunk_size_override(0);
        invalidate_model_state();
    }
    void set_equal_two_prefill(bool enabled) {
        TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                    "equal-two prefill policy must be set before the first forward");
        equal_two_prefill_ = enabled;
        set_chunk_size_override(0);
        invalidate_model_state();
    }
    void set_rhinovla_high_precision(bool enabled) {
        RpuExecutionCoordinator::require_graph_quiescent("rhinovla_text_high_precision");
        RpuExecutionCoordinator::check_current_thread_execution_allowed("rhinovla_text_high_precision");
        TORCH_CHECK(!RpuKernelGraph::has_active() && get_last_resolved_chunk_size() == 0 &&
                        !rhinovla_precision_bound_,
                    "RhinoVLA text precision must be bound once before forward");
        if (enabled) {
            TORCH_CHECK(typeid(*this) == typeid(CausalDecoderModel) &&
                            num_cores() == 8 && num_layers() == 28 && hidden_size() == 2048 &&
                            intermediate_size() == 6144 && num_q_heads() == 16 &&
                            num_kv_heads() == 8 && head_dim() == 128 &&
                            has_qk_norm_ && has_mrope_ && use_silu_ && !adarms_ &&
                            !qwen3vl_2b_w8_profile_ &&
                            !decode_gate_up_fusion_,
                        "RhinoVLA high-precision text requires exact ordinary v3 W8 prefix");
            for (const auto& layer : layer_weights_) {
                for (const auto* weight : {&layer.q_w, &layer.k_w, &layer.v_w,
                        &layer.o_w, &layer.gate_w, &layer.up_w, &layer.down_w})
                    TORCH_CHECK(weight->scalar_type() == at::kChar,
                                "RhinoVLA high-precision text requires uniform W8 Linears");
            }
            rpu_require_high_precision_math_kernels();
        }
        rhinovla_high_precision_ = enabled;
        rhinovla_precision_bound_ = true;
        invalidate_model_state();
    }

    void set_rhinovla_high_precision_fusions(bool enabled) {
        constexpr const char* operation = "rhinovla_text_high_precision_fusions";
        RpuExecutionCoordinator::require_graph_quiescent(operation);
        RpuExecutionCoordinator::check_current_thread_execution_allowed(operation);
        TORCH_CHECK(!RpuKernelGraph::has_active() && get_last_resolved_chunk_size() == 0 &&
                        !rhinovla_precision_fusions_bound_,
                    "RhinoVLA text precision fusions must be bound once before forward");
        TORCH_CHECK(rhinovla_precision_bound_ && rhinovla_high_precision_,
                    "RhinoVLA text precision fusions require an enabled HIGH owner");
        if (enabled) rpu_require_high_precision_fusion_kernels();
        rhinovla_high_precision_fusions_ = enabled;
        rhinovla_precision_fusions_bound_ = true;
        invalidate_model_state();
    }

    void set_linear_acc32(bool enabled) {
        TORCH_CHECK(
            get_last_resolved_chunk_size() == 0,
            "causal decoder linear accumulation mode must be set before the first forward");
        linear_acc32_ = enabled;
        invalidate_model_state();
    }

    // Ordinary Causal handles opt in explicitly before installing weights.
    // Derived owners must provide their complete baseline declarations and
    // opt in internally; a public setter cannot silently change a subclass.
    void set_explicit_mask_spm_residency(bool enabled) {
        TORCH_CHECK(typeid(*this) == typeid(CausalDecoderModel) &&
                        !explicit_mask_residency_configured_ &&
                        num_layers() == 0 && get_last_resolved_chunk_size() == 0,
                    "explicit mask SPM residency must be configured once on a "
                    "plain decoder before weights or planning");
        explicit_mask_residency_requested_ = enabled;
        explicit_mask_residency_configured_ = true;
        invalidate_model_state();
    }

    CausalDecoderModel() = default;

    void configure_cold_routes(
        bool qwen3_spm_kv_by_mha, bool adarms_fused_bcast,
        int core_count = 8, int64_t vocab_size = 0) {
        TORCH_CHECK(
            !cold_routes_configured_ && num_layers() == 0 &&
                get_last_resolved_chunk_size() == 0,
            "causal decoder cold routes must be configured once before weights or planning");
        const int head_tp = resolve_causal_decoder_lm_head_tp(core_count, vocab_size);
        if (core_count != num_cores()) set_execution_core_count(core_count);
        configured_vocab_size_ = vocab_size;
        configured_lm_head_tp_ = head_tp;
        qwen3_spm_kv_by_mha_enabled_ = qwen3_spm_kv_by_mha;
        adarms_fused_bcast_enabled_ = adarms_fused_bcast;
        cold_routes_configured_ = true;
    }

    // `position` is the cache position the planned forward will start at, and
    // it is not decorative: chunk validity depends on the KV context length
    // (position + execution_len), so a plan resolved at position 0 can be
    // REJECTED by compute_chunks at position P. That is what a second
    // conversational turn hit — the planner accepted 528 and the forward
    // answered "no chunk_size in [16, 528] ... satisfies BOTH SPM budget AND
    // subclass kernel-validity" — which is why this is an argument now instead
    // of a hardcoded 0.
    int64_t resolve_prefill_chunk_size(int64_t execution_len,
                                       int64_t position) {
        // MR-D: the dry plan and the execution that follows it read the SAME
        // per-handle chunk_size_override_, so they cannot resolve differently.
        // (Pre-MR-D both had to re-apply the process global here to stay in
        // agreement, which is why the global reached the dry planner at all.)
        return resolve_chunk_size_for_shape(
            execution_len, position, std::nullopt,
            /*is_causal=*/true);
    }

    std::vector<int64_t> resolve_prefill_stage_domain(
        int64_t execution_len, int64_t position, bool is_causal,
        int64_t mask_kv_len, int64_t rope_mode = 0,
        int64_t graph_lifecycle = static_cast<int64_t>(
            FmbGraphLifecycle::RETAINED_CACHE), int64_t logical_len = 0,
        int64_t planning_chunk_size_override = -1) {
        detail::validate_fmb_planning_shape(
            execution_len, position,
            "causal decoder prefill stage-domain planner");
        TORCH_CHECK(mask_kv_len >= 0,
                    "RPU_PLANNER_REJECT:CAPABILITY: causal decoder "
                    "stage-domain mask width must be "
                    "non-negative, got ", mask_kv_len);
        TORCH_CHECK(!(is_causal && mask_kv_len > 0),
                    "RPU_PLANNER_REJECT:CAPABILITY: causal decoder "
                    "stage-domain explicit mask requires "
                    "is_causal=false");
        TORCH_CHECK(
            rope_mode == 0 || rope_mode == 1,
            "RPU_PLANNER_REJECT:CAPABILITY: causal decoder rope_mode must "
            "be 0 (model default) or 1 (partial M-RoPE), got ", rope_mode);
        TORCH_CHECK(
            rope_mode == 0 || has_mrope_,
            "RPU_PLANNER_REJECT:CAPABILITY: partial M-RoPE requires an "
            "M-RoPE decoder handle");
        (void)subclass_chunk_size_cap(execution_len, position);
        TORCH_CHECK(
            mask_kv_len == 0 ||
                (chunk_envelope().declared() &&
                 mask_kv_len <= chunk_envelope().max_kv_len),
            "RPU_PLANNER_REJECT:CAPABILITY: causal decoder stage-domain "
            "mask width exceeds the certified envelope");
        std::optional<at::Tensor> mask = std::nullopt;
        if (mask_kv_len > 0) {
            mask = at::empty(
                {1, mask_kv_len},
                at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
        }
        return encode_fmb_prefill_stage_domain(
            resolve_prefill_stage_domain_with_physical_context(
                execution_len, position, mask, is_causal,
                /*requested_chunk_size=*/0, logical_len,
                rope_mode,
                static_cast<FmbGraphLifecycle>(graph_lifecycle),
                CausalDecoderPlanningMode::ORDINARY,
                planning_chunk_size_override));
    }

    std::vector<int64_t> resolve_decode_stage_descriptor(int64_t position = 0) {
        TORCH_CHECK(qwen3vl_2b_w8_profile_
                ? ((position == 0 && !fuse_lm_head_) || (position >= 576 && position <= 579))
                : position == 0,
            "decode descriptor position is outside this model's admitted lifecycle");
        auto candidates = resolve_prefill_stage_domain_with_physical_context(
            /*seq_len=*/1, position, std::nullopt,
            /*is_causal=*/true, /*requested_chunk_size=*/0,
            /*logical_len=*/1, /*rope_mode=*/qwen3vl_2b_w8_profile_ ? 1 : 0,
            FmbGraphLifecycle::RETAINED_CACHE);
        TORCH_CHECK(
            candidates.size() == 1,
            "RPU_PLANNER_REJECT:CAPABILITY: causal decoder decode must "
            "resolve exactly one native stage candidate, got ",
            candidates.size());
        return encode_fmb_prefill_stage_candidate(candidates.front());
    }

    const std::vector<int64_t>& multiview_text_stage_descriptor() const {
        validate_qwen3vl_multiview_text_contract();
        TORCH_CHECK(qwen3vl_multiview_text_composite_inputs_primed_ &&
                        !qwen3vl_multiview_text_composite_dispatch_ &&
                        !qwen3vl_multiview_text_stage_descriptor_.empty(),
                    "multiview Text descriptor requires the actual primed owner outside dispatch");
        (void)kvinsert_cost_domain("causal_decoder", qwen3vl_multiview_text_stage_descriptor_);
        return qwen3vl_multiview_text_stage_descriptor_;
    }

    std::vector<int64_t> qwen3vl_pooler_z1_stage_descriptor(bool partial_mrope);

    // ── Model state (set once via set_weights, invalidates graph cache) ──

    void set_weights(
        at::TensorList q_w_list, at::TensorList k_w_list,
        at::TensorList v_w_list, at::TensorList o_w_list,
        at::TensorList q_norm_list, at::TensorList k_norm_list,
        at::TensorList input_norm_list, at::TensorList post_norm_list,
        at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
        const at::Tensor& cos, const at::Tensor& sin,
        const at::Tensor& final_norm_w,
        int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size,
        double eps, bool use_silu,
        // R-Phase 0.5 (Qwen3-VL re-port): empty defaults preserve existing
        // 1D-RoPE / no-injection behavior.  Phase 1 (M-RoPE) and Phase 2
        // (DeepStack) will lift the empty-only gate below.
        at::IntArrayRef mrope_section = {},
        at::IntArrayRef deepstack_lang_layers = {},
        // Wall-OSS-0.5 (P1): optional QKV bias. Empty defaults preserve the
        // existing no-bias path (Qwen3 / Llama). All three size==N -> biased.
        at::TensorList q_bias_list = {},
        at::TensorList k_bias_list = {},
        at::TensorList v_bias_list = {},
        at::TensorList q_w_scale_list = {},
        at::TensorList k_w_scale_list = {},
        at::TensorList v_w_scale_list = {},
        at::TensorList o_w_scale_list = {},
        at::TensorList gate_scale_list = {},
        at::TensorList up_scale_list = {},
        at::TensorList down_scale_list = {},
        at::Tensor q_nvfp4_tensor_scales = {},
        at::Tensor k_nvfp4_tensor_scales = {},
        at::Tensor v_nvfp4_tensor_scales = {},
        at::Tensor o_nvfp4_tensor_scales = {},
        at::Tensor gate_nvfp4_tensor_scales = {},
        at::Tensor up_nvfp4_tensor_scales = {},
        at::Tensor down_nvfp4_tensor_scales = {})
    {
        // ─────────────────────────────────────────────────────────────────────
        // Basic sanity
        // ─────────────────────────────────────────────────────────────────────
        int64_t N = static_cast<int64_t>(q_w_list.size());
        TORCH_CHECK(N > 0, "causal_decoder_set_weights: empty weight lists");
        TORCH_CHECK(num_q_heads > 0 && num_kv_heads > 0 && head_dim > 0
                    && hidden_size > 0 && intermediate_size > 0,
                    "causal_decoder_set_weights: model dim params must be positive "
                    "(num_q_heads=", num_q_heads, ", num_kv_heads=", num_kv_heads,
                    ", head_dim=", head_dim, ", hidden_size=", hidden_size,
                    ", intermediate_size=", intermediate_size, ")");
        // Note: Qwen3 permits num_q_heads * head_dim != hidden_size (q-projection
        // dim can exceed hidden_size, e.g. Qwen3-0.6B has 16*128=2048 vs 1024).
        // o_proj maps (num_q_heads*head_dim) → hidden_size.
        TORCH_CHECK(num_q_heads % num_kv_heads == 0,
                    "causal_decoder_set_weights: num_q_heads (", num_q_heads,
                    ") must be divisible by num_kv_heads (", num_kv_heads,
                    ") — Qwen3 uses GQA");

        // ─────────────────────────────────────────────────────────────────────
        // R-Phase 2 (Qwen3-VL): DeepStack entry validation
        //
        // Indices must reference valid layers, be distinct, and not exceed 3
        // (Qwen3-VL ships 3 deepstack mergers; more would require extending
        // the deepstack_dense_src_base_ array below).
        // ─────────────────────────────────────────────────────────────────────
        const bool has_deepstack = (deepstack_lang_layers.size() > 0);
        if (has_deepstack) {
            TORCH_CHECK(deepstack_lang_layers.size() <= 3,
                        "causal_decoder_set_weights: deepstack_lang_layers must have "
                        "at most 3 entries (Qwen3-VL deepstack merger count), got size=",
                        deepstack_lang_layers.size());
            for (size_t i = 0; i < deepstack_lang_layers.size(); ++i) {
                const int64_t L = deepstack_lang_layers[i];
                TORCH_CHECK(L >= 0 && L < N,
                            "causal_decoder_set_weights: deepstack_lang_layers[", i,
                            "]=", L, " must be in [0, num_layers=", N, ")");
                for (size_t j = i + 1; j < deepstack_lang_layers.size(); ++j) {
                    TORCH_CHECK(deepstack_lang_layers[j] != L,
                                "causal_decoder_set_weights: deepstack_lang_layers must "
                                "be distinct; entries ", i, " and ", j,
                                " both equal ", L);
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // All 11 weight lists must have the same length
        // ─────────────────────────────────────────────────────────────────────
        auto check_list = [&](const at::TensorList& list, const char* name) {
            TORCH_CHECK(static_cast<int64_t>(list.size()) == N,
                        "causal_decoder_set_weights: ", name, ".size()=", list.size(),
                        " != num_layers=", N,
                        " (inferred from q_w_list.size()). "
                        "All per-layer weight lists must have identical length.");
        };
        check_list(k_w_list,       "k_w_list");
        check_list(v_w_list,       "v_w_list");
        check_list(o_w_list,       "o_w_list");
        // codex HIGH-1+HIGH-2 (Phase 06.1 / D-6-17): q_norm/k_norm length checks
        // gated on non-empty input. Qwen3 (qk_norm_lists populated) -> both lists
        // size==N -> has_qk_norm_=true. Llama (empty lists from
        // _patch_decoder_common(qk_norm_lists=None)) -> has_qk_norm_=false.
        // Reject the asymmetric case (one list non-empty, the other empty) loudly.
        TORCH_CHECK(
            (q_norm_list.size() > 0) == (k_norm_list.size() > 0),
            "causal_decoder_set_weights: q_norm_list and k_norm_list must both be empty "
            "(no QK head-norm; e.g. Llama) OR both have size==num_layers (e.g. "
            "Qwen3). Got q_norm_list.size()=", q_norm_list.size(),
            " vs k_norm_list.size()=", k_norm_list.size(), ".");
        const bool qk_norm_present = !q_norm_list.empty();
        if (qk_norm_present) {
            check_list(q_norm_list,    "q_norm_list");
            check_list(k_norm_list,    "k_norm_list");
        }
        check_list(input_norm_list,"input_norm_list");
        check_list(post_norm_list, "post_norm_list");
        check_list(gate_list,      "gate_list");
        check_list(up_list,        "up_list");
        check_list(down_list,      "down_list");

        const bool has_scale = !q_w_scale_list.empty();
        auto check_optional_scale_list = [&](const at::TensorList& list,
                                             const char* name) {
            if (has_scale) {
                check_list(list, name);
            } else {
                TORCH_CHECK(list.empty(),
                            "causal_decoder_set_weights: ", name,
                            " must be empty unless all quantized scale lists are provided");
            }
        };
        check_optional_scale_list(k_w_scale_list,  "k_w_scale_list");
        check_optional_scale_list(v_w_scale_list,  "v_w_scale_list");
        check_optional_scale_list(o_w_scale_list,  "o_w_scale_list");
        check_optional_scale_list(gate_scale_list, "gate_scale_list");
        check_optional_scale_list(up_scale_list,   "up_scale_list");
        check_optional_scale_list(down_scale_list, "down_scale_list");
        const bool has_nvfp4 =
            has_scale && q_w_scale_list[0].scalar_type() == at::kByte;
        if (has_scale) {
            check_list(q_w_scale_list, "q_w_scale_list");
        }

        // ─────────────────────────────────────────────────────────────────────
        // Global tensor shape checks
        // ─────────────────────────────────────────────────────────────────────
        TORCH_CHECK(cos.defined() && sin.defined() && final_norm_w.defined(),
                    "causal_decoder_set_weights: cos/sin/final_norm_w must be defined tensors");
        // cos/sin: either [max_pos, head_dim] (HF expanded format) or [max_pos, head_dim/2]
        // (kernel format, half-dim since RoPE applies to pairs). Both are accepted.
        TORCH_CHECK(cos.dim() == 2,
                    "causal_decoder_set_weights: cos must be 2D, got ", cos.dim(), "D");
        TORCH_CHECK(cos.size(-1) == head_dim || cos.size(-1) == head_dim / 2,
                    "causal_decoder_set_weights: cos last dim must be head_dim (", head_dim,
                    ") or head_dim/2 (", head_dim / 2, "), got ", cos.size(-1));
        TORCH_CHECK(sin.sizes() == cos.sizes(),
                    "causal_decoder_set_weights: sin.sizes() must equal cos.sizes(), got ",
                    sin.sizes(), " vs ", cos.sizes());
        TORCH_CHECK(final_norm_w.dim() == 1 && final_norm_w.size(0) == hidden_size,
                    "causal_decoder_set_weights: final_norm_w must be 1D with size=hidden_size (",
                    hidden_size, "), got shape ", final_norm_w.sizes());

        // ─────────────────────────────────────────────────────────────────────
        // Per-layer defined/rank check (quick structural sanity)
        // Full weight-shape verification is subclass-internal detail; we just
        // make sure no tensor is undefined or wrong rank.
        // ─────────────────────────────────────────────────────────────────────
        auto check_all_defined_rank = [&](const at::TensorList& list,
                                          const char* name, int64_t expected_rank) {
            for (int64_t i = 0; i < N; i++) {
                TORCH_CHECK(list[i].defined(),
                            "causal_decoder_set_weights: ", name, "[", i, "] is undefined");
                TORCH_CHECK(list[i].dim() == expected_rank,
                            "causal_decoder_set_weights: ", name, "[", i, "] must be ",
                            expected_rank, "D, got ", list[i].dim(), "D");
            }
        };
        check_all_defined_rank(q_w_list,        "q_w_list",        2);
        check_all_defined_rank(k_w_list,        "k_w_list",        2);
        check_all_defined_rank(v_w_list,        "v_w_list",        2);
        check_all_defined_rank(o_w_list,        "o_w_list",        2);
        // codex HIGH-1+HIGH-2: skip QK-norm rank checks when has_qk_norm_=false
        // (Llama path: q_norm_list and k_norm_list are intentionally empty).
        if (qk_norm_present) {
            check_all_defined_rank(q_norm_list,     "q_norm_list",     1);
            check_all_defined_rank(k_norm_list,     "k_norm_list",     1);
        }
        check_all_defined_rank(input_norm_list, "input_norm_list", 1);
        check_all_defined_rank(post_norm_list,  "post_norm_list",  1);
        check_all_defined_rank(gate_list,       "gate_list",       2);
        check_all_defined_rank(up_list,         "up_list",         2);
        check_all_defined_rank(down_list,       "down_list",       2);
        if (has_scale) {
            auto check_quant_pair = [&](const at::TensorList& weights,
                                        const at::TensorList& scales,
                                        const char* name) {
                for (int64_t i = 0; i < N; ++i) {
                    TORCH_CHECK(scales[i].defined(),
                                "causal_decoder_set_weights: ", name,
                                " scale[", i, "] is undefined");
                    if (has_nvfp4 && scales[i].scalar_type() == at::kByte) {
                        TORCH_CHECK(weights[i].scalar_type() == at::kByte,
                                    "causal_decoder_set_weights: ", name,
                                    " NVFP4 weight[", i, "] must be packed uint8");
                        TORCH_CHECK(
                            scales[i].size(0) * 16 == weights[i].size(1) * 2
                                && scales[i].size(1) == weights[i].size(0),
                            "causal_decoder_set_weights: ", name,
                            " NVFP4 scale must be [K/16,N]; weight=",
                            weights[i].sizes(), " scale=", scales[i].sizes());
                        continue;
                    }
                    TORCH_CHECK(
                        scales[i].scalar_type() == at::kHalf,
                        "causal_decoder_set_weights: ", name, " scale[", i,
                        "] must be fp16 for W8/W4 or uint8 FP8 E4M3FN for "
                        "NVFP4, got ", scales[i].scalar_type());
                    if (weights[i].scalar_type() == at::kHalf) {
                        TORCH_CHECK(
                            !has_nvfp4,
                            "causal_decoder_set_weights: mixed NVFP4 ", name,
                            " weight[", i, "] must use the W8A16 int8 fallback, got FP16");
                        TORCH_CHECK(
                            scales[i].dim() == 1 &&
                                scales[i].size(0) == weights[i].size(0),
                            "causal_decoder_set_weights: FP16 ", name,
                            " placeholder scale[", i,
                            "] must be per-channel [N], got ", scales[i].sizes());
                    } else if (weights[i].scalar_type() == at::kChar) {
                        TORCH_CHECK(
                            scales[i].dim() == 1 &&
                                scales[i].size(0) == weights[i].size(0),
                            "causal_decoder_set_weights: W8A16 ", name,
                            " scale[", i, "] must be per-channel [N], got ",
                            scales[i].sizes());
                    } else {
                        TORCH_CHECK(
                            weights[i].scalar_type() == at::kByte,
                            "causal_decoder_set_weights: quantized ", name,
                            " weight[", i, "] must be int8 W8A16 or packed uint8 ",
                            "W4A16, got ", weights[i].scalar_type());
                        TORCH_CHECK(
                            scales[i].dim() == 2,
                            "causal_decoder_set_weights: packed W4A16 ", name,
                            " scale[", i, "] must use the controller-striped ",
                            "pgrp 2D payload, got ", scales[i].sizes());
                    }
                }
            };
            check_quant_pair(q_w_list,    q_w_scale_list,  "q");
            check_quant_pair(k_w_list,    k_w_scale_list,  "k");
            check_quant_pair(v_w_list,    v_w_scale_list,  "v");
            check_quant_pair(o_w_list,    o_w_scale_list,  "o");
            check_quant_pair(gate_list,   gate_scale_list, "gate");
            check_quant_pair(up_list,     up_scale_list,   "up");
            check_quant_pair(down_list,   down_scale_list, "down");
        }

        auto check_nvfp4_tensor_scales = [&](const at::Tensor& ts, const char* name) {
            if (!has_nvfp4) {
                TORCH_CHECK(!ts.defined(),
                            "causal_decoder_set_weights: ", name,
                            " is only valid with uint8 NVFP4 block scales");
                return;
            }
            TORCH_CHECK(ts.defined() && ts.dim() == 1 && ts.size(0) >= N,
                        "causal_decoder_set_weights: ", name,
                        " must be fp32 RPU [at_least_num_layers=", N, "], got ",
                        ts.defined() ? ts.sizes() : at::IntArrayRef{});
            TORCH_CHECK(ts.numel() % 8 == 0,
                        "causal_decoder_set_weights: ", name,
                        " must cover complete 32-byte vector-load blocks "
                        "(numel must be divisible by 8), got ", ts.numel());
            TORCH_CHECK(ts.scalar_type() == at::kFloat,
                        "causal_decoder_set_weights: ", name,
                        " must be fp32, got ", ts.scalar_type());
            TORCH_CHECK(ts.device().type() == c10::DeviceType::PrivateUse1 &&
                            ts.is_contiguous(),
                        "causal_decoder_set_weights: ", name,
                        " must be contiguous on RPU");
        };
        check_nvfp4_tensor_scales(q_nvfp4_tensor_scales, "q_nvfp4_tensor_scales");
        check_nvfp4_tensor_scales(k_nvfp4_tensor_scales, "k_nvfp4_tensor_scales");
        check_nvfp4_tensor_scales(v_nvfp4_tensor_scales, "v_nvfp4_tensor_scales");
        check_nvfp4_tensor_scales(o_nvfp4_tensor_scales, "o_nvfp4_tensor_scales");
        check_nvfp4_tensor_scales(gate_nvfp4_tensor_scales, "gate_nvfp4_tensor_scales");
        check_nvfp4_tensor_scales(up_nvfp4_tensor_scales, "up_nvfp4_tensor_scales");
        check_nvfp4_tensor_scales(down_nvfp4_tensor_scales, "down_nvfp4_tensor_scales");
        // ─────────────────────────────────────────────────────────────────────
        // Wall-OSS-0.5 (P1): optional QKV bias presence.
        // Reject the asymmetric case (some empty, some populated) loudly.
        // All three empty -> no bias (Qwen3 / Llama, unchanged).
        // All three size==N -> 1D per-layer bias vectors gathered into SPM.
        // ─────────────────────────────────────────────────────────────────────
        const bool qkv_bias_present = (q_bias_list.size() > 0);
        TORCH_CHECK(
            (q_bias_list.size() > 0) == (k_bias_list.size() > 0)
            && (k_bias_list.size() > 0) == (v_bias_list.size() > 0),
            "causal_decoder_set_weights: q/k/v_bias_list must all be empty "
            "(no QKV bias; e.g. Qwen3/Llama) OR all have size==num_layers "
            "(e.g. Qwen2.5/Wall-OSS). Got sizes q=", q_bias_list.size(),
            " k=", k_bias_list.size(), " v=", v_bias_list.size(), ".");
        if (qkv_bias_present) {
            check_list(q_bias_list, "q_bias_list");
            check_list(k_bias_list, "k_bias_list");
            check_list(v_bias_list, "v_bias_list");
            check_all_defined_rank(q_bias_list, "q_bias_list", 1);
            check_all_defined_rank(k_bias_list, "k_bias_list", 1);
            check_all_defined_rank(v_bias_list, "v_bias_list", 1);
        }

        const RpuRmsNormSpmContract ordinary_rmsnorm_spm_contract =
            rpu_snapshot_rmsnorm_spm_contract();

        // ─────────────────────────────────────────────────────────────────────
        // Commit state
        // ─────────────────────────────────────────────────────────────────────
        if (num_cores() != 8) {
            const bool vl_semantics =
                hidden_size != 1024 &&
                mrope_section == std::vector<int64_t>{24, 20, 20} &&
                deepstack_lang_layers == std::vector<int64_t>{0, 1, 2};
            TORCH_CHECK(num_layers() == 0 || reduced_qwen3vl_profile_ == vl_semantics,
                        "reduced-core decoder cannot change its installed text semantics");
            const bool bounded_text_w8 = has_scale && !has_nvfp4 &&
                N == 28 && num_q_heads == 16 &&
                num_kv_heads == 8 && head_dim == 128 && hidden_size == 2048 &&
                intermediate_size == 6144 &&
                (vl_semantics || (mrope_section.empty() && deepstack_lang_layers.empty()));
            TORCH_CHECK((!has_scale || bounded_text_w8) && !has_nvfp4 &&
                            (vl_semantics || (mrope_section.empty() &&
                                              deepstack_lang_layers.empty())) ,
                        "reduced-core decoder requires plain Qwen3 / Qwen3-VL FP16 or exact Qwen3-1.7B / Qwen3-VL-2B W8A16");
            TORCH_CHECK(cold_routes_configured_ && configured_vocab_size_ == 151936 &&
                            lm_head_tp() == 4,
                        "reduced-core Qwen3 profile requires cold vocabulary topology before weights");
            TORCH_CHECK((num_cores() == 4 || num_cores() == 6) &&
                            N == qwen3_fp16_profile_num_layers(
                                num_q_heads, num_kv_heads, head_dim,
                                hidden_size, intermediate_size) && use_silu &&
                            qk_norm_present && !qkv_bias_present &&
                            typeid(*this) == typeid(CausalDecoderModel),
                        "reduced-core decoder requires an exact FP16 Qwen3 or Qwen3-VL geometry and layer count");
            for (const auto* weights : {&q_w_list, &k_w_list, &v_w_list,
                                      &o_w_list, &gate_list, &up_list, &down_list}) {
                for (const auto& weight : *weights) {
                    TORCH_CHECK(weight.scalar_type() == (bounded_text_w8 ? at::kChar : at::kHalf),
                                "reduced-core decoder requires a uniform admitted FP16 or INT8 projection inventory");
                }
            }
            const auto topology = resolve_decoder_execution_topology(
                num_cores(), num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size);
            const int64_t physical_intermediate = decoder_mlp_intermediate_size(
                intermediate_size, topology.mlp_tp);
            for (int64_t i = 0; i < N; ++i) {
                TORCH_CHECK(gate_list[i].size(0) == physical_intermediate &&
                                gate_list[i].size(1) == hidden_size &&
                                up_list[i].size(0) == physical_intermediate &&
                                up_list[i].size(1) == hidden_size &&
                                down_list[i].size(0) == hidden_size &&
                                down_list[i].size(1) == physical_intermediate,
                            "reduced-core Qwen3 MLP weights must match the physical cold geometry at layer ", i);
            }
        }
        reduced_qwen3vl_profile_ = num_cores() != 8 && !mrope_section.empty();
        set_model_params(num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size);
        set_num_layers(N);
        has_qk_norm_ = qk_norm_present;
        has_qkv_bias_ = qkv_bias_present;
        eps_ = eps;
        use_silu_ = use_silu;
        ordinary_rmsnorm_spm_contract_ = ordinary_rmsnorm_spm_contract;
        rhinovla_high_precision_ = false;
        rhinovla_precision_bound_ = false;
        rhinovla_high_precision_fusions_ = false;
        rhinovla_precision_fusions_bound_ = false;

        const bool delivery_deepstack = deepstack_lang_layers == std::vector<int64_t>{5, 11, 17};
        bool exact_delivery_weights = N == 28 && has_scale;
        for (int64_t i = 0; exact_delivery_weights && i < N; ++i) {
            const auto attention_dtype = i < 27 ? at::kChar : at::kHalf;
            exact_delivery_weights = q_w_list[i].scalar_type() == attention_dtype &&
                k_w_list[i].scalar_type() == attention_dtype &&
                v_w_list[i].scalar_type() == attention_dtype &&
                o_w_list[i].scalar_type() == attention_dtype &&
                gate_list[i].scalar_type() == at::kChar &&
                up_list[i].scalar_type() == at::kChar &&
                down_list[i].scalar_type() == at::kChar;
        }
        const bool qwen3vl_2b_w8_topology = exact_delivery_weights && !has_nvfp4 &&
            mrope_section == std::vector<int64_t>{24, 20, 20} &&
            delivery_deepstack && num_q_heads == 16 && num_kv_heads == 8 &&
            head_dim == 128 && hidden_size == 2048 && intermediate_size == 6144 &&
            use_silu && has_qk_norm_ && !has_qkv_bias_;
        qwen3vl_2b_w8_profile_ = qwen3vl_2b_w8_topology;
        // This cold capability changes only causal chunk/layer traversal.
        // W4 retains the validated packed weight / controller-striped scale
        // ABI; FP16 retains its per-handle accumulation policy. The existing
        // 4B W8 schedule remains separately admitted and manifested.
        const auto resident_projection_dtype = q_w_list[0].scalar_type();
        const bool resident_2b_geometry = N == 28 && hidden_size == 2048 &&
            intermediate_size == 6144 && num_q_heads == 16;
        const bool resident_4b_geometry = N == 36 && hidden_size == 2560 &&
            intermediate_size == 9728 && num_q_heads == 32;
        qwen3vl_ordinary_resident_prefill_profile_ =
            typeid(*this) == typeid(CausalDecoderModel) && num_cores() == 8 &&
            (resident_2b_geometry || resident_4b_geometry) && !has_nvfp4 &&
            num_kv_heads == 8 && head_dim == 128 && use_silu &&
            has_qk_norm_ && !has_qkv_bias_ &&
            mrope_section == std::vector<int64_t>{24, 20, 20} &&
            deepstack_lang_layers == std::vector<int64_t>{0, 1, 2} &&
            ((resident_projection_dtype == at::kHalf && !has_scale) ||
             (resident_projection_dtype == at::kByte && has_scale) ||
             (resident_2b_geometry && resident_projection_dtype == at::kChar && has_scale));
        for (int64_t i = 0; qwen3vl_ordinary_resident_prefill_profile_ && i < N; ++i) {
            qwen3vl_ordinary_resident_prefill_profile_ =
                q_w_list[i].scalar_type() == resident_projection_dtype &&
                k_w_list[i].scalar_type() == resident_projection_dtype &&
                v_w_list[i].scalar_type() == resident_projection_dtype &&
                o_w_list[i].scalar_type() == resident_projection_dtype &&
                gate_list[i].scalar_type() == resident_projection_dtype &&
                up_list[i].scalar_type() == resident_projection_dtype &&
                down_list[i].scalar_type() == resident_projection_dtype;
        }
        // Reuse the cold all-layer dtype/topology proof for the measured FP16
        // decode profile. Default ACC32 still selects AUTO_TILE at dispatch.
        qwen3vl_fp16_decode_profile_ =
            qwen3vl_ordinary_resident_prefill_profile_ &&
            resident_projection_dtype == at::kHalf;
        // Ordinary HF DeepStack consumers are 0/1/2. This separate profile
        // never inherits the fixed delivery's packed weights or position graphs.
        qwen3vl_4b_w8_profile_ = N == 36 && num_cores() == 8 && has_scale &&
            !has_nvfp4 && hidden_size == 2560 && intermediate_size == 9728 &&
            num_q_heads == 32 && num_kv_heads == 8 && head_dim == 128 &&
            mrope_section == std::vector<int64_t>{24, 20, 20} &&
            deepstack_lang_layers == std::vector<int64_t>{0, 1, 2} &&
            use_silu && has_qk_norm_ && !has_qkv_bias_;
        for (int64_t i = 0; qwen3vl_4b_w8_profile_ && i < N; ++i) {
            qwen3vl_4b_w8_profile_ = q_w_list[i].scalar_type() == at::kChar &&
                k_w_list[i].scalar_type() == at::kChar &&
                v_w_list[i].scalar_type() == at::kChar &&
                o_w_list[i].scalar_type() == at::kChar &&
                gate_list[i].scalar_type() == at::kChar &&
                up_list[i].scalar_type() == at::kChar &&
                down_list[i].scalar_type() == at::kChar;
        }
        // QK postprocessing consumes FP16 activations and BASE RMSNorm, not
        // projection weights. Its exact 4B geometry also applies to FP16
        // ACC32 and packed W4 without changing either Linear route.
        const auto projection_dtype = q_w_list[0].scalar_type();
        bool qwen3vl_4b_qk_profile = N == 36 && num_cores() == 8 &&
            !has_nvfp4 && hidden_size == 2560 && intermediate_size == 9728 &&
            num_q_heads == 32 && num_kv_heads == 8 && head_dim == 128 &&
            mrope_section == std::vector<int64_t>{24, 20, 20} &&
            deepstack_lang_layers == std::vector<int64_t>{0, 1, 2} &&
            use_silu && has_qk_norm_ && !has_qkv_bias_ &&
            ((projection_dtype == at::kHalf && !has_scale) ||
             ((projection_dtype == at::kChar || projection_dtype == at::kByte) && has_scale));
        for (int64_t i = 0; qwen3vl_4b_qk_profile && i < N; ++i) {
            qwen3vl_4b_qk_profile = q_w_list[i].scalar_type() == projection_dtype &&
                k_w_list[i].scalar_type() == projection_dtype &&
                v_w_list[i].scalar_type() == projection_dtype &&
                o_w_list[i].scalar_type() == projection_dtype &&
                gate_list[i].scalar_type() == projection_dtype &&
                up_list[i].scalar_type() == projection_dtype &&
                down_list[i].scalar_type() == projection_dtype;
        }
        // Bind the optional payload once, before planning/capture. Its presence
        // changes the actual COMPLETE route; an old REF keeps the original MLP.
        qwen3vl_4b_decode_gateup_fused_ = qwen3vl_4b_w8_profile_ &&
            KernelCache::instance().has_loaded(KernelId::QWEN3VL_GATEUP_SWIGLU_GEMV);
        qwen3vl_4b_decode_qk_fused_ = qwen3vl_4b_qk_profile &&
            KernelCache::instance().has_loaded(KernelId::QWEN3VL_QK_NORM_MROPE_D128);
        qwen3vl_4b_decode_qk_kv_fused_ = qwen3vl_4b_qk_profile &&
            KernelCache::instance().has_loaded(KernelId::QWEN3VL_QK_NORM_MROPE_KV_INSERT_D128);
        qwen3vl_4b_decode_o_ring_norm_fused_ = qwen3vl_4b_w8_profile_ &&
            KernelCache::instance().has_loaded(KernelId::QWEN3VL_O_RING_NORM_M1_H2560_W8);
        // Cold opt-in by actual storage, not a model-name route override.
        // General native callers with independent Q/K/V keep separate GEMVs.
        qwen3vl_4b_packed_qkv_weights_ = qwen3vl_4b_w8_profile_;
        for (int64_t i = 0; qwen3vl_4b_packed_qkv_weights_ && i < N; ++i) {
            qwen3vl_4b_packed_qkv_weights_ = qwen3vl_4b_qkv_storage_contiguous(
                q_w_list[i], k_w_list[i], v_w_list[i]) &&
                q_w_scale_list[i].is_contiguous() &&
                k_w_scale_list[i].is_contiguous() && v_w_scale_list[i].is_contiguous();
        }
        qwen3vl_4b_qkv_scale_bank_ = qwen3vl_4b_packed_qkv_weights_
            ? at::empty({N, 6144}, q_w_scale_list[0].options()) : at::Tensor();
        decode_qkv_fusion_ = qwen3vl_2b_w8_topology;
        decode_gate_up_fusion_ = qwen3vl_2b_w8_topology;
        prefill_packed_gate_up_strided_ = qwen3vl_2b_w8_topology;
        prefill_packed_qkv_deinterleave_ = qwen3vl_2b_w8_topology;
        prefill_packed_qkv_planar_direct_ = qwen3vl_2b_w8_topology;
        fused_prefill_kv_insert_ = qwen3vl_2b_w8_topology;
        fused_decode_kv_insert_ = qwen3vl_2b_w8_topology;
        fused_prefill_qk_mrope_ = qwen3vl_2b_w8_topology;
        fused_decode_qk_mrope_ = qwen3vl_2b_w8_topology;

        layer_weights_.clear();
        layer_weights_.reserve(N);
        for (int64_t i = 0; i < N; i++) {
            at::Tensor qkv_w;
            at::Tensor qkv_ws;
            at::Tensor gate_up_w;
            at::Tensor gate_up_ws;
            const int64_t q_dim = num_q_heads * head_dim;
            const int64_t kv_dim = num_kv_heads * head_dim;
            const int64_t qkv_dim = q_dim + 2 * kv_dim;
            const bool layer_qkv_w8 =
                q_w_list[i].scalar_type() == at::kChar &&
                k_w_list[i].scalar_type() == at::kChar &&
                v_w_list[i].scalar_type() == at::kChar;
            if (decode_qkv_fusion_ && layer_qkv_w8) {
                qkv_w = at::empty({qkv_dim, hidden_size}, q_w_list[i].options());
                auto* w_dst = static_cast<uint8_t*>(qkv_w.data_ptr());
                const size_t q_bytes = q_w_list[i].nbytes();
                const size_t k_bytes = k_w_list[i].nbytes();
                const size_t v_bytes = v_w_list[i].nbytes();
                std::memcpy(w_dst, q_w_list[i].data_ptr(), q_bytes);
                std::memcpy(w_dst + q_bytes, k_w_list[i].data_ptr(), k_bytes);
                std::memcpy(w_dst + q_bytes + k_bytes,
                            v_w_list[i].data_ptr(), v_bytes);
                rpu_ddr_flush_force_sized(w_dst, q_bytes + k_bytes + v_bytes);

                qkv_ws = at::empty({qkv_dim}, q_w_scale_list[i].options());
                auto* s_dst = qkv_ws.data_ptr<c10::Half>();
                const auto* qs = q_w_scale_list[i].data_ptr<c10::Half>();
                const auto* ks = k_w_scale_list[i].data_ptr<c10::Half>();
                const auto* vs = v_w_scale_list[i].data_ptr<c10::Half>();
                const int64_t q_local = q_dim / NUM_CORES;
                const int64_t kv_local = kv_dim / NUM_CORES;
                const int64_t qkv_local = qkv_dim / NUM_CORES;
                for (int core = 0; core < NUM_CORES; ++core) {
                    auto* core_dst = s_dst + core * qkv_local;
                    std::memcpy(core_dst, qs + core * q_local,
                                q_local * sizeof(c10::Half));
                    std::memcpy(core_dst + q_local, ks + core * kv_local,
                                kv_local * sizeof(c10::Half));
                    std::memcpy(core_dst + q_local + kv_local,
                                vs + core * kv_local,
                                kv_local * sizeof(c10::Half));
                }
                rpu_ddr_flush_force_sized(
                    s_dst, static_cast<size_t>(qkv_dim) * sizeof(c10::Half));
            } else if (qwen3vl_4b_packed_qkv_weights_) {
                // as_strided's storage offset is int64, including layers above
                // 2 GiB in the single model-owned INT8 bank. No second swizzle,
                // memcpy of weights, or per-layer DDR allocation is performed.
                qkv_w = q_w_list[i].as_strided(
                    {qkv_dim, hidden_size}, {hidden_size, 1},
                    q_w_list[i].storage_offset());
                qkv_ws = qwen3vl_4b_qkv_scale_bank_.select(0, i);
                copy_qwen3vl_4b_qkv_scales(
                    qkv_ws, q_w_scale_list[i], k_w_scale_list[i], v_w_scale_list[i]);
                rpu_ddr_flush_force_sized(qkv_ws.data_ptr(), qkv_ws.nbytes());
            }

            const bool layer_gate_up_w8 =
                gate_list[i].scalar_type() == at::kChar &&
                up_list[i].scalar_type() == at::kChar;
            if (decode_gate_up_fusion_ && layer_gate_up_w8) {
                const int64_t gate_up_dim = 2 * intermediate_size;
                gate_up_w = at::empty(
                    {gate_up_dim, hidden_size}, gate_list[i].options());
                auto* w_dst = static_cast<uint8_t*>(gate_up_w.data_ptr());
                const size_t gate_bytes = gate_list[i].nbytes();
                const size_t up_bytes = up_list[i].nbytes();
                std::memcpy(w_dst, gate_list[i].data_ptr(), gate_bytes);
                std::memcpy(w_dst + gate_bytes, up_list[i].data_ptr(), up_bytes);
                rpu_ddr_flush_force_sized(w_dst, gate_bytes + up_bytes);

                gate_up_ws = at::empty(
                    {gate_up_dim}, gate_scale_list[i].options());
                auto* s_dst = gate_up_ws.data_ptr<c10::Half>();
                const auto* gate_s = gate_scale_list[i].data_ptr<c10::Half>();
                const auto* up_s = up_scale_list[i].data_ptr<c10::Half>();
                const int64_t local = intermediate_size / NUM_CORES;
                const int64_t fused_local = 2 * local;
                for (int core = 0; core < NUM_CORES; ++core) {
                    auto* core_dst = s_dst + core * fused_local;
                    std::memcpy(core_dst, gate_s + core * local,
                                local * sizeof(c10::Half));
                    std::memcpy(core_dst + local, up_s + core * local,
                                local * sizeof(c10::Half));
                }
                rpu_ddr_flush_force_sized(
                    s_dst,
                    static_cast<size_t>(gate_up_dim) * sizeof(c10::Half));
            }
            layer_weights_.push_back({
                q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
                qkv_w,
                // codex HIGH-1+HIGH-2: undefined tensors when no qk-norm
                // (Llama path). build_layer_subgraph branches on has_qk_norm_
                // and never dereferences q_norm_w / k_norm_w in the false branch.
                has_qk_norm_ ? q_norm_list[i] : at::Tensor(),
                has_qk_norm_ ? k_norm_list[i] : at::Tensor(),
                input_norm_list[i], post_norm_list[i],
                gate_list[i], up_list[i], down_list[i],
                gate_up_w,
                // Optional QKV bias (undefined when has_qkv_bias_==false).
                has_qkv_bias_ ? q_bias_list[i] : at::Tensor(),
                has_qkv_bias_ ? k_bias_list[i] : at::Tensor(),
                has_qkv_bias_ ? v_bias_list[i] : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(q_w_list[i], q_w_scale_list[i])
                          : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(k_w_list[i], k_w_scale_list[i])
                          : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(v_w_list[i], v_w_scale_list[i])
                          : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(o_w_list[i], o_w_scale_list[i])
                          : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(gate_list[i], gate_scale_list[i])
                          : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(up_list[i], up_scale_list[i])
                          : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(down_list[i], down_scale_list[i])
                          : at::Tensor(),
                qkv_ws, gate_up_ws,
            });
            if (qwen3vl_4b_decode_gateup_fused_) {
                auto& weights = layer_weights_.back();
                weights.gate_ws = retain_qwen3vl_4b_gateup_scale(weights.gate_ws);
                weights.up_ws = retain_qwen3vl_4b_gateup_scale(weights.up_ws);
            }
            if (qwen3vl_4b_decode_o_ring_norm_fused_) {
                auto& weights = layer_weights_.back();
                weights.o_ws = retain_qwen3vl_4b_o_scale(weights.o_ws);
            }
        }
        cos_ = cos;
        sin_ = sin;
        final_norm_w_ = final_norm_w;
        nvfp4_ = has_nvfp4;
        q_nvfp4_tensor_scales_ = q_nvfp4_tensor_scales;
        k_nvfp4_tensor_scales_ = k_nvfp4_tensor_scales;
        v_nvfp4_tensor_scales_ = v_nvfp4_tensor_scales;
        o_nvfp4_tensor_scales_ = o_nvfp4_tensor_scales;
        gate_nvfp4_tensor_scales_ = gate_nvfp4_tensor_scales;
        up_nvfp4_tensor_scales_ = up_nvfp4_tensor_scales;
        down_nvfp4_tensor_scales_ = down_nvfp4_tensor_scales;

        // Qwen3/Qwen3-VL has four static RMSNorm weights per layer plus one
        // final RMSNorm. Pack them into one stable DDR allocation so graph
        // BUILD can preload one contiguous SPM slab instead of 4*N+1 sources.
        // QKV-bias and dynamic AdaRMS decoder variants retain their existing
        // per-layer layout and refresh rules.
        norm_slab_w_ = at::Tensor{};
        if (norm_slab_model_eligible()) {
            const int64_t slab_bytes = norm_slab_size_bytes();
            TORCH_CHECK(
                slab_bytes > 0 && slab_bytes % DWIDTH == 0,
                "Qwen norm slab size must be positive and fp16 aligned, got ",
                slab_bytes);
            norm_slab_w_ = at::empty(
                {slab_bytes / DWIDTH}, input_norm_list[0].options());
            auto* dst = reinterpret_cast<uint8_t*>(norm_slab_w_.data_ptr());
            std::memset(dst, 0, static_cast<size_t>(slab_bytes));

            auto copy_norm = [&](const at::Tensor& src, int64_t dst_offset,
                                 int64_t expected_elements, const char* name,
                                 int64_t layer) {
                TORCH_CHECK(
                    src.device().type() == at::kPrivateUse1 &&
                        src.scalar_type() == at::kHalf && src.is_contiguous() &&
                        src.numel() == expected_elements,
                    "Qwen norm slab requires contiguous fp16 RPU ", name,
                    " at layer ", layer, "; got device=", src.device(),
                    " dtype=", src.scalar_type(),
                    " contiguous=", src.is_contiguous(),
                    " numel=", src.numel(), " expected=", expected_elements);
                std::memcpy(
                    dst + dst_offset, src.data_ptr(),
                    static_cast<size_t>(expected_elements * DWIDTH));
            };
            for (int64_t L = 0; L < N; ++L) {
                copy_norm(input_norm_list[L], norm_input_offset_bytes(L),
                          hidden_size, "input_norm", L);
                copy_norm(post_norm_list[L], norm_post_offset_bytes(L),
                          hidden_size, "post_norm", L);
                copy_norm(q_norm_list[L], norm_q_offset_bytes(L),
                          head_dim, "q_norm", L);
                copy_norm(k_norm_list[L], norm_k_offset_bytes(L),
                          head_dim, "k_norm", L);
            }
            copy_norm(final_norm_w, norm_final_offset_bytes(), hidden_size,
                      "final_norm", -1);
            rpu_ddr_flush_force_sized(dst, static_cast<size_t>(slab_bytes));
        }

        // ─────────────────────────────────────────────────────────────────────
        // R-Phase 1 (Qwen3-VL): M-RoPE setup
        //
        // mrope_section empty → 1D RoPE path (Qwen3 / Llama transparent).
        // mrope_section non-empty → validate (3 axes, sum == head_dim/2, kernel-format
        //   cos/sin) and compute strobe masks + allocate position_ids keepalive.
        // ─────────────────────────────────────────────────────────────────────
        const bool has_mrope = (mrope_section.size() > 0);
        if (has_mrope) {
            // C1: mrope_section must be exactly 3 axes (T, H, W), all non-negative,
            // sum == head_dim/2 (rotary half-dim).
            TORCH_CHECK(mrope_section.size() == 3,
                        "causal_decoder_set_weights: mrope_section must have exactly "
                        "3 entries (T, H, W), got size=", mrope_section.size());
            for (size_t i = 0; i < 3; ++i) {
                TORCH_CHECK(mrope_section[i] >= 0,
                            "causal_decoder_set_weights: mrope_section[", i,
                            "] must be non-negative, got ", mrope_section[i]);
            }
            const int64_t mrope_sum = mrope_section[0] + mrope_section[1] + mrope_section[2];
            TORCH_CHECK(mrope_sum == head_dim / 2,
                        "causal_decoder_set_weights: sum(mrope_section)=", mrope_sum,
                        " must equal head_dim/2=", head_dim / 2);

            // C2: mrope kernel reads cos/sin as [max_seq, head_dim/2] strictly —
            // HF-expanded [max_seq, head_dim] is rejected (entry check C2).
            TORCH_CHECK(cos.size(-1) == head_dim / 2,
                        "causal_decoder_set_weights: mrope path requires cos.last_dim "
                        "== head_dim/2 (kernel format), got ", cos.size(-1),
                        " (head_dim=", head_dim, ").  Adapter must pass kernel-format "
                        "cos/sin via cos_sin= override.");

            mrope_section_.assign(mrope_section.begin(), mrope_section.end());
            mrope_strobe_masks_ = rpu_compute_mrope_strobe_masks(
                mrope_section_, static_cast<int32_t>(head_dim));

            // Lazy-allocate keepalive on RPU device. Stable across forwards;
            // forward() copies per-forward [seq_len, 3] position_ids into the
            // appropriate row range at index `ctx().position`.
            if (qwen3vl_4b_decode_qk_fused_ || qwen3vl_4b_decode_qk_kv_fused_) {
                // The fused leaf keeps original M-RoPE's 32B position load.
                // Initialize backing for its final-row overfetch, while the
                // public/forward logical capacity and shape remain unchanged.
                auto backing = at::zeros(
                    {QWEN3_MROPE_MAX_KEEPALIVE_SEQ + 8, 3},
                    at::TensorOptions().dtype(at::kInt).device(at::kPrivateUse1));
                position_ids_keepalive_ = backing.narrow(
                    0, 0, QWEN3_MROPE_MAX_KEEPALIVE_SEQ);
            } else if (!position_ids_keepalive_.defined()
                || position_ids_keepalive_.size(0) != QWEN3_MROPE_MAX_KEEPALIVE_SEQ
                || position_ids_keepalive_.size(1) != 3
                || position_ids_keepalive_.device().type() != at::kPrivateUse1) {
                position_ids_keepalive_ = at::zeros(
                    {QWEN3_MROPE_MAX_KEEPALIVE_SEQ, 3},
                    at::TensorOptions().dtype(at::kInt).device(at::kPrivateUse1));
            } else {
                position_ids_keepalive_.zero_();
            }

            // partial_mrope keepalives [MAX_KEEPALIVE_SEQ, head_dim/2] fp16 —
            // populated per-forward only when the adapter supplies interleaved
            // cos/sin; idle (allocated, never written) for the legacy mrope path.
            const int64_t half = head_dim / 2;
            if (!cos_il_keepalive_.defined()
                || cos_il_keepalive_.size(0) != QWEN3_MROPE_MAX_KEEPALIVE_SEQ
                || cos_il_keepalive_.size(1) != half
                || cos_il_keepalive_.device().type() != at::kPrivateUse1) {
                auto il_opts = at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1);
                cos_il_keepalive_ = at::zeros({QWEN3_MROPE_MAX_KEEPALIVE_SEQ, half}, il_opts);
                sin_il_keepalive_ = at::zeros({QWEN3_MROPE_MAX_KEEPALIVE_SEQ, half}, il_opts);
            }
            // This block (re)allocated and/or zeroed the keepalives. Retained
            // source stamps must therefore never suppress the next refill.
            ka_pos_sources_.clear();
            ka_cos_sources_.clear();
            ka_sin_sources_.clear();
            ka_last_position_ = -1;
            ka_last_seq_ = -1;
            ka_last_pos_src_ = nullptr;
            ka_last_pos_src_version_ = -1;
            ka_last_cos_src_ = nullptr;
            ka_last_cos_src_version_ = -1;
            ka_last_sin_src_ = nullptr;
            ka_last_sin_src_version_ = -1;
            has_mrope_ = true;
        } else {
            has_mrope_ = false;
            mrope_section_.clear();
            mrope_strobe_masks_ = {};
            position_ids_keepalive_ = at::Tensor{};
        }

        // ─────────────────────────────────────────────────────────────────────
        // R-Phase 2 (Qwen3-VL): DeepStack state commit
        //
        // Store the layer indices that receive visual_embed injection. The dense
        // src bases stay zero-initialized until the first forward — the mutable
        // DMA recorded in build_layer_subgraph will deref &deepstack_dense_src_base_
        // each REPLAY, picking up the live dev_addr that forward() writes from
        // caller-supplied tensors.
        // ─────────────────────────────────────────────────────────────────────
        if (has_deepstack) {
            deepstack_lang_layers_.assign(
                deepstack_lang_layers.begin(), deepstack_lang_layers.end());
        } else {
            deepstack_lang_layers_.clear();
        }
        deepstack_dense_src_base_.fill(0);
        deepstack_dense_refs_.clear();

        // AdaRMS FiLM state (LingBot-VLA expert) is established per set_adarms_*() call,
        // which always runs AFTER set_weights. Clear it here (symmetric with the mrope /
        // deepstack resets above) so a handle re-used as a PLAIN decoder after AdaRMS use
        // does not retain stale flags / shift buffers and keep adding FiLM shifts.
        adarms_ = false;
        adarms_mutable_ = false;
        adarms_unroll_ = false;
        adarms_schedule_select_ = false;
        adarms_schedule_step_ = -1;
        input_shift_.clear();
        post_shift_.clear();
        input_scale_ka_ = at::Tensor{};
        post_scale_ka_  = at::Tensor{};
        input_shift_ka_ = at::Tensor{};
        post_shift_ka_  = at::Tensor{};
        adarms_is_ = at::Tensor{};
        adarms_ish_ = at::Tensor{};
        adarms_ps_ = at::Tensor{};
        adarms_psh_ = at::Tensor{};
        adarms_is_schedule_base_ = adarms_ish_schedule_base_ = 0;
        adarms_ps_schedule_base_ = adarms_psh_schedule_base_ = 0;
        adarms_is_live_base_ = adarms_ish_live_base_ = 0;
        adarms_ps_live_base_ = adarms_psh_live_base_ = 0;

        invalidate_model_state();  // D-503: last non-empty statement of set_weights
    }

    // AdaRMS FiLM step-update (LingBot-VLA expert). The Euler denoise loop calls
    // this before each forward with the step's folded scale/shift. The scale rides
    // on the input/post norm-weight slots (rmsnorm uses it as the weight); the shift
    // is broadcast-added after each norm. All tensors [hidden] fp16 RPU. Must be
    // called AFTER set_weights (needs num_layers()/layer_weights_).
    void set_adarms_step(
        at::TensorList input_scales, at::TensorList input_shifts,
        at::TensorList post_scales,  at::TensorList post_shifts)
    {
        TORCH_CHECK(num_cores() == 8, "set_adarms_step requires an eight-core profile");
        const int64_t N = num_layers();
        TORCH_CHECK(!layer_weights_.empty(), "set_adarms_step: call set_weights first");
        TORCH_CHECK(!adarms_schedule_select_,
                    "set_adarms_step cannot replace a bound AdaRMS schedule");
        TORCH_CHECK((int64_t)input_scales.size()==N && (int64_t)input_shifts.size()==N
                 && (int64_t)post_scales.size()==N  && (int64_t)post_shifts.size()==N,
                    "set_adarms_step: all 4 lists must have size num_layers=", N);
        const int64_t h = hidden_size();
        bool planning_domain_changed = !adarms_ || adarms_mutable_ || adarms_unroll_;
        const auto same_layout = [](const at::Tensor& before, const at::Tensor& after) {
            return before.defined() && before.scalar_type() == after.scalar_type() &&
                before.device() == after.device() && before.sizes() == after.sizes() &&
                before.strides() == after.strides();
        };
        input_shift_.resize(N);
        post_shift_.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            auto chk = [&](const at::Tensor& t, const char* nm){
                TORCH_CHECK(t.defined() && t.dim()==1 && t.size(0)==h
                         && t.scalar_type()==at::kHalf
                         && t.device().type()==at::kPrivateUse1,
                         "set_adarms_step: ", nm, "[", i, "] must be [", h, "] fp16 RPU");
            };
            chk(input_scales[i],"input_scales"); chk(input_shifts[i],"input_shifts");
            chk(post_scales[i], "post_scales");  chk(post_shifts[i], "post_shifts");
            planning_domain_changed = planning_domain_changed ||
                !same_layout(layer_weights_[i].input_norm_w, input_scales[i]) ||
                !same_layout(layer_weights_[i].post_norm_w, post_scales[i]) ||
                !same_layout(input_shift_[i], input_shifts[i]) ||
                !same_layout(post_shift_[i], post_shifts[i]);
            layer_weights_[i].input_norm_w = input_scales[i];   // scale rides on norm_w
            layer_weights_[i].post_norm_w  = post_scales[i];
            input_shift_[i] = input_shifts[i];
            post_shift_[i]  = post_shifts[i];
        }
        adarms_ = true;
        adarms_mutable_ = false;  // plain build-per-step path: disable mutable/unroll so a
        adarms_unroll_  = false;  // handle downgrading from those modes doesn't keep the old path
        // Value/address changes still rebuild the plain Graph; only its unchanged
        // FP16 layout and plain AdaRMS route may reuse the native candidate plan.
        invalidate_model_state(planning_domain_changed);
    }

    // set_adarms_step_mutable — REPLAY-safe AdaRMS for the denoise loop (LingBot-VLA).
    // The plain set_adarms_step above swaps the per-layer scale/shift tensors and
    // invalidates model state, requiring a fresh graph build each Euler step.
    // This variant instead holds the scale/shift in STABLE-ADDR keepalives whose CONTENTS
    // are refreshed in place per step (no invalidate), and build_*_body emits a per-layer
    // graph-body broadcast DMA that re-loads them into the norm_w/shift SPM slots on every
    // REPLAY (emit_adarms_mut_refresh) — the same discipline as the partial_mrope cos/sin
    // keepalive. So the graph is BUILT once and REPLAYED for steps 1..N (fast-replay-safe).
    // Inputs are [num_layers, hidden] fp16 RPU (one contiguous tensor per role, not lists).
    void set_adarms_step_mutable(
        const at::Tensor& input_scale, const at::Tensor& input_shift,
        const at::Tensor& post_scale,  const at::Tensor& post_shift)
    {
        TORCH_CHECK(num_cores() == 8, "set_adarms_step_mutable requires an eight-core profile");
        const int64_t N = num_layers();
        const int64_t h = hidden_size();
        TORCH_CHECK(!layer_weights_.empty(), "set_adarms_step_mutable: call set_weights first");
        TORCH_CHECK(!adarms_schedule_select_,
                    "set_adarms_step_mutable cannot replace a bound AdaRMS schedule");
        auto chk = [&](const at::Tensor& t, const char* nm){
            TORCH_CHECK(t.defined() && t.dim()==2 && t.size(0)==N && t.size(1)==h
                     && t.scalar_type()==at::kHalf && t.device().type()==at::kPrivateUse1,
                     "set_adarms_step_mutable: ", nm, " must be [", N, ",", h, "] fp16 RPU");
        };
        chk(input_scale,"input_scale"); chk(input_shift,"input_shift");
        chk(post_scale,"post_scale");   chk(post_shift,"post_shift");
        if (!adarms_mutable_) {
            // First step: allocate stable keepalives (own storage) + point the per-layer
            // scale/shift at [hidden] views of them, then BUILD once (emit the refresh DMAs).
            input_scale_ka_ = input_scale.clone();
            input_shift_ka_ = input_shift.clone();
            post_scale_ka_  = post_scale.clone();
            post_shift_ka_  = post_shift.clone();
            input_shift_.resize(N);
            post_shift_.resize(N);
            for (int64_t i = 0; i < N; ++i) {
                layer_weights_[i].input_norm_w = input_scale_ka_[i];  // [hidden] view, stable addr
                layer_weights_[i].post_norm_w  = post_scale_ka_[i];
                input_shift_[i] = input_shift_ka_[i];
                post_shift_[i]  = post_shift_ka_[i];
            }
            adarms_ = true;
            adarms_mutable_ = true;
            invalidate_model_state();  // build the mutable graph once
        } else {
            // Subsequent steps: refresh keepalive CONTENTS in place (addr stable) + flush.
            // No invalidate ⇒ the built graph REPLAYs; its body DMAs re-read these buffers.
            input_scale_ka_.copy_(input_scale);
            input_shift_ka_.copy_(input_shift);
            post_scale_ka_.copy_(post_scale);
            post_shift_ka_.copy_(post_shift);
            rpu_ddr_flush_force(input_scale_ka_.data_ptr<c10::Half>());
            rpu_ddr_flush_force(input_shift_ka_.data_ptr<c10::Half>());
            rpu_ddr_flush_force(post_scale_ka_.data_ptr<c10::Half>());
            rpu_ddr_flush_force(post_shift_ka_.data_ptr<c10::Half>());
        }
    }

    // set_adarms_unroll — bind the CALL-INVARIANT per-step AdaRMS fold for the in-graph
    // denoise unroll (LingBot-VLA lingbot_denoise op). The 4 buffers are
    // [num_steps, num_layers, hidden] fp16 RPU (adapter's precomputed fold_all); the
    // per-body refresh (emit_adarms_mut_refresh_*) reads slice (body_iter*N + L) each
    // iteration, so one BUILD unrolls all num_steps steps with per-step FiLM. The
    // preload_callbacks + build-time layer_weights_ views point at step 0 (body_iter 0).
    // Must be called AFTER set_weights.
    void set_adarms_unroll(
        const at::Tensor& input_scale, const at::Tensor& input_shift,
        const at::Tensor& post_scale,  const at::Tensor& post_shift)
    {
        TORCH_CHECK(num_cores() == 8, "set_adarms_unroll requires an eight-core profile");
        const int64_t N = num_layers();
        const int64_t h = hidden_size();
        TORCH_CHECK(!layer_weights_.empty(), "set_adarms_unroll: call set_weights first");
        TORCH_CHECK(!adarms_schedule_select_,
                    "set_adarms_unroll cannot replace a bound AdaRMS schedule");
        auto chk = [&](const at::Tensor& t, const char* nm){
            TORCH_CHECK(t.defined() && t.dim()==3 && t.size(1)==N && t.size(2)==h
                     && t.scalar_type()==at::kHalf && t.device().type()==at::kPrivateUse1,
                     "set_adarms_unroll: ", nm, " must be [num_steps,", N, ",", h, "] fp16 RPU");
        };
        chk(input_scale,"input_scale"); chk(input_shift,"input_shift");
        chk(post_scale,"post_scale");   chk(post_shift,"post_shift");
        TORCH_CHECK(input_scale.size(0)==input_shift.size(0)
                 && input_scale.size(0)==post_scale.size(0)
                 && input_scale.size(0)==post_shift.size(0),
                 "set_adarms_unroll: all 4 buffers must share num_steps");
        // Own the storage (stable addr across the synchronous forward + REPLAYs).
        adarms_is_  = input_scale.clone();
        adarms_ish_ = input_shift.clone();
        adarms_ps_  = post_scale.clone();
        adarms_psh_ = post_shift.clone();
        // Point the per-layer scale/shift at body_iter-0 [hidden] views so the
        // declare_buffers preload_callbacks (BUILD-time) load step 0; the per-body
        // refresh DMAs then re-read the correct body_iter slice on every iteration.
        input_shift_.resize(N);
        post_shift_.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            layer_weights_[i].input_norm_w = adarms_is_[0][i];   // scale rides on norm_w
            layer_weights_[i].post_norm_w  = adarms_ps_[0][i];
            input_shift_[i] = adarms_ish_[0][i];
            post_shift_[i]  = adarms_psh_[0][i];
        }
        adarms_         = true;
        adarms_mutable_ = true;   // declare_buffers shift slots + build-body refresh + FiLM add
        adarms_unroll_  = true;   // refresh reads the body_iter slice (below)
        invalidate_model_state();
    }

    // LingBot2 host-glue optimization: bind the immutable full Euler AdaRMS schedule once,
    // then select a step by changing four stable member live-bases immediately before forward.
    // The captured graph records pointers to those member fields, so full fast replay can skip
    // the C++ op walk and still re-dereference the current step during queue preparation.
    // No schedule copy or flush is needed in the per-step path.
    void bind_adarms_schedule(
        const at::Tensor& input_scale, const at::Tensor& input_shift,
        const at::Tensor& post_scale,  const at::Tensor& post_shift)
    {
        const int64_t N = num_layers();
        const int64_t h = hidden_size();
        TORCH_CHECK(!layer_weights_.empty(), "bind_adarms_schedule: call set_weights first");
        TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                    "bind_adarms_schedule must run before the first forward");
        TORCH_CHECK(!adarms_schedule_select_,
                    "bind_adarms_schedule may only bind once per set_weights");
        TORCH_CHECK(!adarms_ && !adarms_mutable_ && !adarms_unroll_,
                    "bind_adarms_schedule cannot replace another AdaRMS mode");
        TORCH_CHECK(adarms_fused_bcast_enabled_,
                    "bind_adarms_schedule requires the AdaRMS fused-broadcast "
                    "cold planner capability");
        auto chk = [&](const at::Tensor& t, const char* nm){
            TORCH_CHECK(t.defined() && t.dim()==3 && t.size(0)>0
                     && t.size(1)==N && t.size(2)==h && t.is_contiguous()
                     && t.scalar_type()==at::kHalf && t.device().type()==at::kPrivateUse1,
                     "bind_adarms_schedule: ", nm,
                     " must be contiguous [num_steps,", N, ",", h, "] fp16 RPU");
        };
        chk(input_scale,"input_scale"); chk(input_shift,"input_shift");
        chk(post_scale,"post_scale");   chk(post_shift,"post_shift");
        const int64_t steps = input_scale.size(0);
        TORCH_CHECK(input_shift.size(0)==steps && post_scale.size(0)==steps
                 && post_shift.size(0)==steps,
                    "bind_adarms_schedule: all 4 buffers must share num_steps");

        // Direct tensor references keep the caller's RPU storage alive. Contents are immutable;
        // flush each complete slab once here, instead of four copy_ + four flushes per step.
        adarms_is_  = input_scale;
        adarms_ish_ = input_shift;
        adarms_ps_  = post_scale;
        adarms_psh_ = post_shift;
        rpu_ddr_flush_force_sized(adarms_is_.data_ptr<c10::Half>(), adarms_is_.nbytes());
        rpu_ddr_flush_force_sized(adarms_ish_.data_ptr<c10::Half>(), adarms_ish_.nbytes());
        rpu_ddr_flush_force_sized(adarms_ps_.data_ptr<c10::Half>(), adarms_ps_.nbytes());
        rpu_ddr_flush_force_sized(adarms_psh_.data_ptr<c10::Half>(), adarms_psh_.nbytes());
        adarms_is_schedule_base_ = ::rhino_lkn::RpuGetDevAddr(adarms_is_.data_ptr<c10::Half>());
        adarms_ish_schedule_base_ = ::rhino_lkn::RpuGetDevAddr(adarms_ish_.data_ptr<c10::Half>());
        adarms_ps_schedule_base_ = ::rhino_lkn::RpuGetDevAddr(adarms_ps_.data_ptr<c10::Half>());
        adarms_psh_schedule_base_ = ::rhino_lkn::RpuGetDevAddr(adarms_psh_.data_ptr<c10::Half>());

        input_shift_.resize(N);
        post_shift_.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            layer_weights_[i].input_norm_w = adarms_is_[0][i];
            layer_weights_[i].post_norm_w  = adarms_ps_[0][i];
            input_shift_[i] = adarms_ish_[0][i];
            post_shift_[i]  = adarms_psh_[0][i];
        }
        adarms_ = true;
        adarms_mutable_ = true;
        adarms_unroll_ = false;
        adarms_schedule_select_ = true;
        select_adarms_schedule_step(0);
        invalidate_model_state();
    }

    void select_adarms_schedule_step(int64_t step) {
        TORCH_CHECK(adarms_schedule_select_,
                    "select_adarms_schedule_step requires bind_adarms_schedule");
        TORCH_CHECK(step >= 0 && step < adarms_is_.size(0),
                    "AdaRMS schedule step ", step, " out of range [0,", adarms_is_.size(0), ")");
        const uint64_t offset = static_cast<uint64_t>(step * num_layers() * hidden_size()
                                                      * sizeof(c10::Half));
        adarms_is_live_base_  = adarms_is_schedule_base_  + offset;
        adarms_ish_live_base_ = adarms_ish_schedule_base_ + offset;
        adarms_ps_live_base_  = adarms_ps_schedule_base_  + offset;
        adarms_psh_live_base_ = adarms_psh_schedule_base_ + offset;
        adarms_schedule_step_ = step;
    }

    bool has_bound_adarms_schedule() const { return adarms_schedule_select_; }

    // Fixed Wall-OSS-0.5 merger Z1 canary. The coordinator owns one complete
    // replicated [execution, hidden] destination port and fills it before the
    // decoder runs. This component adopts only its normal temporary arena and
    // consumes that external port as layer 0's input residual stream.
    SpmPipelineComponentLayout prepare_wall_oss_z1_layout(
        int64_t execution_len,
        int64_t real_len);
    SpmDense2DSpec wall_oss_z1_destination_spec(
        int64_t execution_len) const;
    void prime_wall_oss_z1_inputs(
        const at::Tensor& position_ids,
        const at::Tensor& rope_cos_il,
        const at::Tensor& rope_sin_il,
        int64_t execution_len);
    void rollback_wall_oss_z1_inputs();
    void unprepare_wall_oss_z1_layout();
    void adopt_wall_oss_z1_layout(
        const SpmPipelineLease& lease,
        const SpmTensorView& scratch);
    void bind_wall_oss_z1_destination(
        const SpmPipelineLease& lease,
        const SpmPortView& destination);
    void validate_wall_oss_z1_layout(
        const SpmPipelineLease& lease) const;
    void clear_wall_oss_z1_layout(uint64_t epoch, uint64_t plan_hash);
    at::Tensor forward_wall_oss_z1(
        const at::Tensor& hidden_shape_carrier,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const at::Tensor& position_ids,
        const at::Tensor& rope_cos_il,
        const at::Tensor& rope_sin_il,
        uint64_t epoch,
        uint64_t plan_hash);
    void check_wall_oss_z1_destroy_allowed() const;

    // Qwen3-VL pooler Z1 canary.  This is deliberately independent from
    // the proven Wall-OSS state machine above: one fixed Qwen3-VL-2B FP16
    // profile consumes a replicated [128, 2048] destination port as layer 0's
    // input without changing the ordinary or Wall-OSS dispatch contracts.  The
    // optional retained DeepStack modes consume either the first or all three
    // replicated [64, 2048] ports at text layers {0} / {0,1,2}, rows [4, 68),
    // without materializing dense DDR DeepStack tensors.  The only second
    // profile is GR00T's exact 16-layer, S82, retained-DS3 production graph.
    SpmPipelineComponentLayout prepare_qwen3vl_pooler_z1_layout(
        int64_t execution_len,
        int64_t real_len,
        int64_t retained_deepstack_count = 0,
        at::IntArrayRef selected_descriptor = {});
    SpmDense2DSpec qwen3vl_pooler_z1_destination_spec(
        int64_t execution_len) const;
    SpmDense2DSpec qwen3vl_pooler_z1_deepstack_spec(
        int64_t ordinal) const;
    SpmDense2DSpec qwen3vl_pooler_z1_deepstack1_spec() const;
    void prime_qwen3vl_pooler_z1_inputs(
        const at::Tensor& position_ids,
        int64_t execution_len,
        const std::optional<at::Tensor>& rope_cos_il = std::nullopt,
        const std::optional<at::Tensor>& rope_sin_il = std::nullopt);
    void rollback_qwen3vl_pooler_z1_inputs();
    void unprepare_qwen3vl_pooler_z1_layout();
    void adopt_qwen3vl_pooler_z1_layout(
        const SpmPipelineLease& lease,
        const SpmTensorView& scratch);
    void bind_qwen3vl_pooler_z1_destination(
        const SpmPipelineLease& lease,
        const SpmPortView& destination);
    void bind_qwen3vl_pooler_z1_deepstack(
        const SpmPipelineLease& lease,
        const SpmPortView& deepstack,
        int64_t ordinal);
    void bind_qwen3vl_pooler_z1_deepstack1(
        const SpmPipelineLease& lease,
        const SpmPortView& deepstack1);
    void validate_qwen3vl_pooler_z1_layout(
        const SpmPipelineLease& lease) const;
    void clear_qwen3vl_pooler_z1_layout(
        uint64_t epoch,
        uint64_t plan_hash);
    at::Tensor forward_qwen3vl_pooler_z1(
        const at::Tensor& hidden_shape_carrier,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const at::Tensor& position_ids,
        uint64_t epoch,
        uint64_t plan_hash);
    void stage_qwen3vl_pooler_z1_outer_fast_component(
        ::GraphKernelRegisterCensusGuard& guard,
        at::TensorList k_caches,
        at::TensorList v_caches) const;
    at::Tensor qwen3vl_pooler_z1_outer_fast_output() const;
    void check_qwen3vl_pooler_z1_destroy_allowed() const;

    // CPU-dry profile probe for the multi-view phase-overlay path.  It is
    // deliberately separate from the fixed H2048/S128 physical Z1 state and
    // cannot adopt, bind, dispatch, or acquire a physical arena.
    SpmPipelineCausalPrefillDryLayout
    prepare_qwen3vl_multiview_text_dry_layout(int64_t execution_len);
    void cancel_qwen3vl_multiview_text_dry_layout();

    // Live Text consumer for the three-image N256 composite canary.  Preparation
    // and endpoint sealing are address-opaque; execution remains fail-closed
    // until the generic composite physical compiler binds all nine retained
    // DeepStack payloads in Step 5.
    SpmPipelineComponentLayout
    prepare_qwen3vl_multiview_text_composite_layout(
        int64_t execution_len, at::IntArrayRef selected_descriptor = {});
    SpmFmbResolvedExecutionProfile
    resolve_qwen3vl_multiview_text_composite_profile();
    SpmFmbResolvedPhaseManifest
    seal_qwen3vl_multiview_text_composite_manifest(
        const SpmFmbResolvedExecutionProfile& profile,
        SpmScratchId arena,
        uint32_t arena_base) const;
    SpmFmbConsumerRowSliceEndpoint
    seal_qwen3vl_multiview_text_consumer_endpoint(
        const SpmFmbResolvedExecutionProfile& profile,
        const SpmFmbResolvedPhaseManifest& manifest,
        SpmPortId destination_port,
        int64_t row_begin,
        int layer_idx) const;
    void prime_qwen3vl_multiview_text_inputs(
        const at::Tensor& position_ids,
        int64_t execution_len,
        const std::optional<at::Tensor>& rope_cos_il = std::nullopt,
        const std::optional<at::Tensor>& rope_sin_il = std::nullopt);
    void rollback_qwen3vl_multiview_text_inputs();
    void unprepare_qwen3vl_multiview_text_composite_layout();
    at::Tensor forward_qwen3vl_multiview_text_composite(
        const at::Tensor& hidden_states,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const at::Tensor& position_ids,
        const std::optional<at::Tensor>& rope_cos_il = std::nullopt,
        const std::optional<at::Tensor>& rope_sin_il = std::nullopt);
    void stage_qwen3vl_multiview_text_outer_fast_component(
        ::GraphKernelRegisterCensusGuard& guard,
        at::TensorList k_caches,
        at::TensorList v_caches) const;
    void bind_qwen3vl_multiview_text_outer_fast_input(
        ::GraphKernelRegisterCensusGuard& guard,
        const at::Tensor& hidden_states);
    at::Tensor qwen3vl_multiview_text_outer_fast_output() const;

    // ── Public entry (called from TORCH_LIBRARY wrapper) ──

    at::Tensor forward(
        const at::Tensor& hidden_states,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const std::optional<at::Tensor>& attention_mask,
        int64_t position,
        bool is_causal,
        // R-Phase 0.5 (Qwen3-VL re-port): None defaults preserve existing
        // 1D-RoPE / no-injection behavior.  Phase 1 / Phase 2 will lift
        // the must-be-None gates below.
        const std::optional<at::Tensor>& position_ids = std::nullopt,
        const std::optional<std::vector<at::Tensor>>& deepstack_dense_visual_embeds = std::nullopt,
        // partial_mrope path: per-forward interleaved cos/sin [seq_len, head_dim/2]
        // fp16 RPU. When supplied (M-RoPE active), the M-RoPE launchers swap to
        // partial_mrope. None (default) preserves the llama_mrope_interleave path.
        const std::optional<at::Tensor>& rope_cos_il = std::nullopt,
        const std::optional<at::Tensor>& rope_sin_il = std::nullopt,
        // HALO WS-2: 1D-RoPE position base. -1 = legacy (cos_sin_start =
        // ctx().position + chunk.offset). >=0 decouples the RoPE table index
        // from the cache-insert offset (see rope_position_base_ for rationale).
        int64_t cos_sin_offset = -1,
        // Batch decode: which KV-cache batch slot this (batch=1) forward reads
        // and writes. Batched PREFILL runs one call per sequence with
        // batch_slot = b; batched DECODE passes batch_slot = 0 and covers slots
        // 0..bs-1 internally. The cache base rides regs 8/9 of the KV-insert /
        // SDPA kernels, which `sync_mutable_params()` re-pushes on every REPLAY
        // exactly like `position` — so all B prefills REPLAY one graph rather
        // than building one graph per slot. That is why batch_slot must NOT
        // enter the GraphCache signature.
        int64_t batch_slot = 0,
        // Fail-closed public capability. Python sets this only for the plain
        // Qwen3 adapter; shared CausalDecoderModel users remain batch=1.
        bool allow_batch_decode = false,
        // Transitional scalar authority is retained in the public ABI only so
        // stale callers fail with a precise message below. Production Causal
        // decoder forwards require one COMPLETE A6 descriptor for both prefill
        // and decode.
        int64_t planned_chunk_size = 0,
        at::IntArrayRef planned_stage_descriptor = {})
    {
        TORCH_CHECK(!qwen3vl_multiview_text_dry_prepared_,
                    "CausalDecoderModel ordinary forward is unavailable while "
                    "its multiview dry layout is prepared");
        TORCH_CHECK(!qwen3vl_multiview_text_composite_prepared_ ||
                        qwen3vl_multiview_text_composite_dispatch_,
                    "CausalDecoderModel ordinary forward is unavailable while "
                    "its multiview composite layout is prepared");
        TORCH_CHECK((!wall_z1_active_ && !wall_z1_inputs_primed_) ||
                        wall_z1_dispatch_,
                    "CausalDecoderModel ordinary forward is unavailable while "
                    "its Wall-OSS Z1 inputs are primed or its lease is active");
        TORCH_CHECK(qwen3vl_pooler_z1_prepared_execution_len_ == 0 ||
                        qwen3vl_pooler_z1_dispatch_,
                    "CausalDecoderModel ordinary forward is unavailable while "
                    "its Qwen3-VL pooler Z1 layout is prepared");
        // ─────────────────────────────────────────────────────────────────────
        // Model must be configured (set_weights called) before forward
        // ─────────────────────────────────────────────────────────────────────
        TORCH_CHECK(num_layers() > 0,
                    "CausalDecoderModel::forward called before set_weights. "
                    "Call torch.ops.rpu.causal_decoder_set_weights(...) first.");
        TORCH_CHECK(!layer_weights_.empty() && final_norm_w_.defined(),
                    "CausalDecoderModel::forward: internal state inconsistent, "
                    "set_weights must have been called successfully");
        TORCH_CHECK(
            planned_chunk_size == 0 && !planned_stage_descriptor.empty(),
            "RPU_PLANNER_REJECT:CAPABILITY: CausalDecoderModel production "
            "forward requires one COMPLETE stage descriptor as its sole "
            "physical authority");

        if (qwen3vl_2b_w8_profile_) {
            TORCH_CHECK(hidden_states.dim() == 3 && hidden_states.size(0) == 1 &&
                is_causal && !attention_mask.has_value() &&
                ((hidden_states.size(1) == 576 && position == 0) ||
                 (hidden_states.size(1) == 1 && position >= 576 && position <= 579)),
                "fixed Qwen3-VL delivery admits only P576 and four decode positions 576..579");
            const std::vector<int64_t> expected_cache_shape{1, 38, 1, 8, 8, 16, 16};
            TORCH_CHECK(k_caches.size() == 28 && v_caches.size() == 28,
                        "fixed delivery requires 28 paired KV caches");
            for (size_t layer = 0; layer < k_caches.size(); ++layer) {
                TORCH_CHECK(k_caches[layer].defined() && v_caches[layer].defined() &&
                    k_caches[layer].sizes().vec() == expected_cache_shape &&
                    v_caches[layer].sizes().vec() == expected_cache_shape,
                    "fixed delivery requires actual paired KV capacity 608 and exact geometry");
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Shape / layout contract — run_all_layers does full hidden_states and
        // k/v cache checks (device / dtype / contiguous / per-layer defined).
        // Keep only the subclass-specific early checks that framework doesn't do.
        // ─────────────────────────────────────────────────────────────────────
        TORCH_CHECK(hidden_states.device().type() == at::kPrivateUse1,
                    "CausalDecoderModel::forward: hidden_states must be on RPU device, got ",
                    hidden_states.device());
        TORCH_CHECK(position >= 0,
                    "CausalDecoderModel::forward: position must be non-negative, got ", position);
        TORCH_CHECK(num_cores() == 8 ||
                        (hidden_states.dim() == 3 && hidden_states.size(0) == 1 && is_causal &&
                         (reduced_qwen3vl_profile_ || !attention_mask.has_value()) &&
                         (reduced_qwen3vl_profile_ || !has_mrope_) &&
                         !adarms_ && !nvfp4_),
                    "reduced-core decoder requires B1 causal FP16 execution for its installed profile");
        const int64_t seq_len_in = hidden_states.size(1);
        detail::validate_fmb_planning_shape(
            seq_len_in, position, "CausalDecoderModel::forward");

        if (num_cores() != 8) {
            if (attention_mask.has_value()) {
                // Exact VL's adapter admits all-valid caller tokens and only
                // appends right-padding. FMB's causal path ignores this 2D
                // token mask: LTM protects the logical prefix, and the adapter
                // slices output/rewinds KV to its logical length. Validate the
                // argument without reading its data or adding mask DMA/ops.
                const at::Tensor& mask = *attention_mask;
                TORCH_CHECK(reduced_qwen3vl_profile_ && mask.defined() &&
                                mask.device() == hidden_states.device() &&
                                mask.is_contiguous() &&
                                mask.dim() == 2 && mask.size(0) == 1 &&
                                mask.size(1) == seq_len_in &&
                                (c10::isIntegralType(mask.scalar_type(), /*includeBool=*/true) ||
                                 c10::isFloatingType(mask.scalar_type())),
                            "reduced-core Qwen3-VL causal token mask must be contiguous "
                            "Boolean/integer/floating RPU [1, execution_len] on the hidden-state device");
            }
            TORCH_CHECK(hidden_states.scalar_type() == at::kHalf &&
                            hidden_states.is_contiguous() &&
                            hidden_states.size(2) == hidden_size(),
                        "reduced-core Qwen3 requires contiguous FP16 hidden states");
            TORCH_CHECK(static_cast<int64_t>(k_caches.size()) == num_layers() &&
                            static_cast<int64_t>(v_caches.size()) == num_layers(),
                        "reduced-core Qwen3 requires one paired KV cache per model layer");
            int64_t cache_blocks = 0;
            for (size_t layer = 0; layer < k_caches.size(); ++layer) {
                for (const at::Tensor& cache : {k_caches[layer], v_caches[layer]}) {
                    TORCH_CHECK(cache.defined() && cache.device() == hidden_states.device() &&
                                    cache.scalar_type() == at::kHalf && cache.is_contiguous() &&
                                    cache.dim() == 7,
                                "reduced-core Qwen3 requires contiguous FP16 RPU 7D KV caches");
                    if (cache_blocks == 0) cache_blocks = cache.size(1);
                    TORCH_CHECK(cache_blocks > 0 &&
                                    cache.sizes().vec() == std::vector<int64_t>(
                                        {1, cache_blocks, 2, 8, 8, 16, 16}) &&
                                    cache_blocks <= std::numeric_limits<int64_t>::max() / 16 &&
                                    seq_len_in <= cache_blocks * 16 &&
                                    position <= cache_blocks * 16 - seq_len_in,
                                "reduced-core Qwen3 KV layout/capacity must match attention TP4");
                }
            }
        }

        // Batch decode: stamp the per-forward KV-cache slot (read by
        // build_layer_subgraph). A non-zero slot is only meaningful for the
        // one-sequence-at-a-time batched prefill; batched decode drives slots
        // itself from hidden_states.size(0).
        TORCH_CHECK(batch_slot >= 0,
                    "CausalDecoderModel::forward: batch_slot must be non-negative, got ",
                    batch_slot);
        TORCH_CHECK(batch_slot == 0 || hidden_states.size(0) == 1,
                    "CausalDecoderModel::forward: batch_slot > 0 selects a single "
                    "cache slot, so it requires hidden_states batch == 1; got batch=",
                    hidden_states.size(0), ", batch_slot=", batch_slot);
        const bool has_batched_cache =
            !k_caches.empty() && k_caches.front().defined() &&
            k_caches.front().dim() > 0 && k_caches.front().size(0) > 1;
        const bool batch_decode_request = hidden_states.size(0) > 1 ||
            batch_slot > 0 || has_batched_cache;
        TORCH_CHECK(num_cores() == 8 || !batch_decode_request,
                    "reduced-core Qwen3 requires a B1 cache");
        TORCH_CHECK(!batch_decode_request || allow_batch_decode,
                    "CausalDecoderModel::forward: batched cache/decode is "
                    "enabled only by the plain Qwen3 adapter");
        TORCH_CHECK(!batch_decode_request ||
                        (!fast_replay_skip_layer_loop_ &&
                         !preload_replay_skip_),
                    "CausalDecoderModel::forward: batched cache/decode is "
                    "incompatible with fast-replay host op-stream skipping");
        const bool saved_batch_decode_active = batch_decode_active_;
        batch_decode_active_ = batch_decode_request;
        auto restore_batch_decode_active = c10::make_scope_exit(
            [&] { batch_decode_active_ = saved_batch_decode_active; });
        cache_batch_slot_ = batch_slot;

        // Generic FMB fast replay skips the layer body. The graph signature
        // does not carry KV addresses, so only admit the stable initial B1
        // causal prefill after verifying that the cache owners match the
        // BUILD witness. Decode and continuation use normal emission or the
        // checked replay path below.
        const bool fast_replay_prefill_candidate =
            fast_replay_skip_layer_loop_ && seq_len_in > 1 && position == 0 &&
            is_causal && !attention_mask.has_value() && !batch_decode_request &&
            !qwen3vl_multiview_text_composite_prepared_;
        prepare_fast_replay_prefill_owners(
            k_caches, v_caches, fast_replay_prefill_candidate);

        // HALO WS-2: stamp per-forward 1D-RoPE position base (read by
        // build_layer_subgraph at cos_sin_start). M-RoPE couples cos_sin_start to
        // the keepalive fill row (ctx().position), so reject the override there.
        TORCH_CHECK(cos_sin_offset < 0 || !has_mrope_,
                    "CausalDecoderModel::forward: cos_sin_offset (1D-RoPE position "
                    "base override) is not supported with M-RoPE active; got "
                    "cos_sin_offset=", cos_sin_offset);
        rope_position_base_ = cos_sin_offset;

        // ─────────────────────────────────────────────────────────────────────
        // Attention mask modes (P4 Wall-OSS):
        //   - is_causal=true:  原 LTM 路径。attention_mask 可传 (HF 默认 4D causal
        //                      mask) 但被 base 静默忽略 (use_explicit_mask=false)。
        //   - is_causal=false: 必须带显式 2D additive mask。base 把它
        //                      (fused_model_base.cpp:788) 归一成 use_explicit_mask
        //                      → ctx().attention_mask；动作去噪 (任意长 prefix,
        //                      rectangular [H, P+H]) 走 MASK_2D。
        // ─────────────────────────────────────────────────────────────────────
        //   - is_causal=false + 无 mask: 故意的全双向 MASK_NONE。HALO WS-2 vit_cache
        //     und prefill (packed_position_ids 全 0,SigLIP->LM 整段双向 full attention)
        //     走这条:build_layer_subgraph 的 sdpa_causal = is_causal && (seq_len>1) =
        //     false -> mask_type=0 (MASK_NONE),与 text_decode 的 seq=1 路是同一条已板证
        //     kernel 分支 (c10::nullopt mask, mask_off=0)。故不再硬拒 is_causal=false 无
        //     mask;调用方对"双向无 mask vs 漏传 mask"负责。

        // Chunk authority is per-handle. Exact SPM-pipeline dispatches scope a
        // prepared chunk to this call only; ordinary forwards keep the cold
        // adapter-installed value and never re-read process-global state.
        const int64_t saved_chunk_size_override = get_chunk_size_override();
        const bool scoped_dispatch_chunk =
            wall_z1_dispatch_ || qwen3vl_pooler_z1_dispatch_ ||
            qwen3vl_multiview_text_composite_dispatch_;
        if (wall_z1_dispatch_) {
            set_chunk_size_override(wall_z1_prepared_chunk_size_);
        } else if (qwen3vl_pooler_z1_dispatch_) {
            set_chunk_size_override(qwen3vl_pooler_z1_prepared_chunk_size_);
        } else if (qwen3vl_multiview_text_composite_dispatch_) {
            set_chunk_size_override(0);
        }
        auto restore_dispatch_chunk = c10::make_scope_exit(
            [this, scoped_dispatch_chunk, saved_chunk_size_override] {
                if (scoped_dispatch_chunk) {
                    set_chunk_size_override(saved_chunk_size_override);
                }
            });

        // ─────────────────────────────────────────────────────────────────────
        // R-Phase 1 (Qwen3-VL): M-RoPE forward-entry handling
        //
        // Contract:
        //   1. set_weights allocated position_ids_keepalive_ at
        //      [MAX_KEEPALIVE_SEQ, 3] int32 RPU DDR (stable across forwards).
        //   2. Caller passes per-forward position_ids [seq_len, 3] int32 RPU;
        //      we copy it into the keepalive at row `ctx().position` and flush.
        //   3. build_layer_subgraph dispatches rpu_launch_mrope_spm_kernel with
        //      pos_offset = ctx().position + chunk.offset (absolute index into
        //      the keepalive — NOT a per-forward offset).
        // ─────────────────────────────────────────────────────────────────────
        // bound the 1D-RoPE table override.
        // cos_sin_start = rope_position_base_ + chunk.offset is read by BOTH
        // build_layer_subgraph (L1481) and emit_kv_first_body (L1341); chunks
        // partition [0, seq_len), so the max table row touched is
        // rope_position_base_ + seq_len_in. Without this, an out-of-range
        // cos_sin_offset makes the RoPE kernels read past cos_/sin_ (OOB DDR) —
        // the KV-cache capacity check (keyed on `position`) does not cover the
        // decoupled table index. -1 is the legacy/auto sentinel (reject below it).
        TORCH_CHECK(cos_sin_offset >= -1,
                    "CausalDecoderModel::forward: cos_sin_offset must be >= -1 "
                    "(-1 = legacy/auto), got ", cos_sin_offset);
        if (cos_sin_offset >= 0) {
            TORCH_CHECK(cos_.defined() && sin_.defined(),
                        "CausalDecoderModel::forward: cos_sin_offset override "
                        "requires RoPE tables to be set (call set_weights first)");
            TORCH_CHECK(seq_len_in <= cos_.size(0)
                            && cos_sin_offset <= cos_.size(0) - seq_len_in
                            && seq_len_in <= sin_.size(0)
                            && cos_sin_offset <= sin_.size(0) - seq_len_in,
                        "CausalDecoderModel::forward: cos_sin_offset overflows "
                        "RoPE table: cos_sin_offset=", cos_sin_offset,
                        " + seq_len=", seq_len_in, " exceeds table rows (cos=",
                        cos_.size(0), ", sin=", sin_.size(0),
                        "). Caller compressed the RoPE position base incorrectly.");
        } else if (!has_mrope_) {
            // Legacy/auto path (cos_sin_offset == -1): cos_sin_start becomes
            // ctx().position + chunk.offset, so the highest table row touched is
            // position + seq_len_in — exactly the same OOB exposure the override
            // branch above guards, but it was UNCHECKED. A short table therefore
            // read past cos_/sin_ silently: decode at position >= table rows
            // returned garbage whose visibility depended on what the allocator
            // had placed after the table (bs==1 happened to look fine, bs>=2 did
            // not). Fail loudly instead.
            TORCH_CHECK(cos_.defined() && sin_.defined(),
                        "CausalDecoderModel::forward: RoPE tables must be set "
                        "(call set_weights first)");
            TORCH_CHECK(seq_len_in <= cos_.size(0)
                            && position <= cos_.size(0) - seq_len_in
                            && seq_len_in <= sin_.size(0)
                            && position <= sin_.size(0) - seq_len_in,
                        "CausalDecoderModel::forward: position overflows the RoPE "
                        "table: position=", position, " + seq_len=", seq_len_in,
                        " exceeds table rows (cos=", cos_.size(0), ", sin=",
                        sin_.size(0), "). The table must cover every reachable "
                        "position — see _install_causal_decoder_forward.");
        }

        if (has_mrope_) {
            RECORD_FUNCTION("rpu_causal::mrope_keepalive", {});
            TORCH_CHECK(position_ids.has_value(),
                        "CausalDecoderModel::forward: position_ids is required "
                        "when set_weights was called with non-empty mrope_section "
                        "(M-RoPE active).");
            const at::Tensor& pos = *position_ids;
            TORCH_CHECK(pos.dim() == 2,
                        "CausalDecoderModel::forward: position_ids must be 2D "
                        "[seq_len, 3], got ", pos.dim(), "D");
            TORCH_CHECK(pos.size(0) == seq_len_in,
                        "CausalDecoderModel::forward: position_ids.size(0)=",
                        pos.size(0), " must equal seq_len=", seq_len_in);
            TORCH_CHECK(pos.size(1) == 3,
                        "CausalDecoderModel::forward: position_ids.size(1) must be 3, got ",
                        pos.size(1));
            TORCH_CHECK(pos.scalar_type() == at::kInt,
                        "CausalDecoderModel::forward: position_ids must be int32, got ",
                        pos.scalar_type());
            TORCH_CHECK(pos.is_contiguous(),
                        "CausalDecoderModel::forward: position_ids must be contiguous");
            TORCH_CHECK(pos.device().type() == at::kPrivateUse1,
                        "CausalDecoderModel::forward: position_ids must be on RPU device, got ",
                        pos.device());
            TORCH_CHECK(position + seq_len_in <= QWEN3_MROPE_MAX_KEEPALIVE_SEQ,
                        "CausalDecoderModel::forward: position+seq_len (",
                        position, "+", seq_len_in, ") exceeds keepalive capacity (",
                        QWEN3_MROPE_MAX_KEEPALIVE_SEQ,
                        "). Bump QWEN3_MROPE_MAX_KEEPALIVE_SEQ in rpu_qwen3_model.cpp.");
            TORCH_CHECK(position_ids_keepalive_.defined(),
                        "CausalDecoderModel::forward: position_ids_keepalive_ undefined "
                        "(set_weights with non-empty mrope_section must precede forward).");
            // Cache non-overlapping keepalive segments, not merely the previous
            // call. A sequential service revisits decode positions after every
            // request; retaining those tiny source tensors avoids three RPU
            // copy/flush operations per token on the next request. Each stamp
            // includes Tensor identity and content version, while overlapping
            // writes invalidate older stamps before they can become false hits.
            const KeepaliveSegmentKey ka_key{position, seq_len_in};
            const bool ka_geom_same =
                (position == ka_last_position_ && seq_len_in == ka_last_seq_);
            const auto& pos_version_counter =
                pos.unsafeGetTensorImpl()->version_counter();
            const int64_t pos_src_version = pos_version_counter.enabled()
                ? static_cast<int64_t>(pos_version_counter.current_version())
                : -1;
            // Inference tensors can be overwritten in a stable adapter slot
            // without a version bump. An unavailable counter cannot certify
            // unchanged contents, even when the pointer and geometry match.
            const bool pos_content_changed =
                pos_src_version < 0 ||
                pos_src_version != ka_last_pos_src_version_;
            if (qwen3vl_2b_w8_profile_
                    ? !keepalive_segment_hit(ka_pos_sources_, ka_key, pos)
                    : (!ka_geom_same || pos.data_ptr() != ka_last_pos_src_ ||
                       pos_content_changed)) {
                position_ids_keepalive_.narrow(0, position, seq_len_in).copy_(pos);
                rpu_ddr_flush_force_sized(
                    position_ids_keepalive_.narrow(0, position, seq_len_in).data_ptr<int32_t>(),
                    static_cast<size_t>(seq_len_in) * 3 * sizeof(int32_t));
                ka_last_pos_src_ = pos.data_ptr();
                ka_last_pos_src_version_ = pos_src_version;
                if (qwen3vl_2b_w8_profile_) {
                    record_keepalive_segment(ka_pos_sources_, ka_key, pos);
                }
            }

            // partial_mrope: when the adapter supplies host-baked interleaved
            // cos/sin, refresh the stable keepalive for this forward. This runs in
            // the prologue (before run_all_layers / the FAST_REPLAY body-skip), so
            // the BUILD-baked keepalive addr stays valid while contents update —
            // same discipline as position_ids (see fast-replay drift fix).
            partial_mrope_active_ = rope_cos_il.has_value();
            if (partial_mrope_active_) {
                TORCH_CHECK(rope_sin_il.has_value(),
                            "CausalDecoderModel::forward: rope_cos_il supplied without rope_sin_il");
                const at::Tensor& cil = *rope_cos_il;
                const at::Tensor& sil = *rope_sin_il;
                const int64_t half = cos_il_keepalive_.size(1);
                TORCH_CHECK(cil.dim() == 2 && cil.size(0) == seq_len_in
                            && cil.size(1) == half && sil.sizes() == cil.sizes(),
                            "CausalDecoderModel::forward: rope_cos_il/sin_il must be "
                            "[seq_len, head_dim/2]=[", seq_len_in, ",", half, "], got cos ",
                            cil.sizes(), " sin ", sil.sizes());
                TORCH_CHECK(cil.scalar_type() == at::kHalf && sil.scalar_type() == at::kHalf,
                            "CausalDecoderModel::forward: rope_cos_il/sin_il must be fp16");
                TORCH_CHECK(cil.is_contiguous() && sil.is_contiguous()
                            && cil.device().type() == at::kPrivateUse1
                            && sil.device().type() == at::kPrivateUse1,
                            "CausalDecoderModel::forward: rope_cos_il/sin_il must be "
                            "contiguous on RPU");
                const size_t il_bytes =
                    static_cast<size_t>(seq_len_in) * half * sizeof(c10::Half);
                const auto& cos_version_counter =
                    cil.unsafeGetTensorImpl()->version_counter();
                const int64_t cos_src_version = cos_version_counter.enabled()
                    ? static_cast<int64_t>(cos_version_counter.current_version())
                    : -1;
                const bool cos_content_changed =
                    cos_src_version < 0 ||
                    cos_src_version != ka_last_cos_src_version_;
                if (qwen3vl_2b_w8_profile_
                        ? !keepalive_segment_hit(ka_cos_sources_, ka_key, cil)
                        : (!ka_geom_same || cil.data_ptr() != ka_last_cos_src_ ||
                           cos_content_changed)) {
                    cos_il_keepalive_.narrow(0, position, seq_len_in).copy_(cil);
                    rpu_ddr_flush_force_sized(
                        cos_il_keepalive_.narrow(0, position, seq_len_in).data_ptr<c10::Half>(),
                        il_bytes);
                    ka_last_cos_src_ = cil.data_ptr();
                    ka_last_cos_src_version_ = cos_src_version;
                    if (qwen3vl_2b_w8_profile_) {
                        record_keepalive_segment(ka_cos_sources_, ka_key, cil);
                    }
                }
                const auto& sin_version_counter =
                    sil.unsafeGetTensorImpl()->version_counter();
                const int64_t sin_src_version = sin_version_counter.enabled()
                    ? static_cast<int64_t>(sin_version_counter.current_version())
                    : -1;
                const bool sin_content_changed =
                    sin_src_version < 0 ||
                    sin_src_version != ka_last_sin_src_version_;
                if (qwen3vl_2b_w8_profile_
                        ? !keepalive_segment_hit(ka_sin_sources_, ka_key, sil)
                        : (!ka_geom_same || sil.data_ptr() != ka_last_sin_src_ ||
                           sin_content_changed)) {
                    sin_il_keepalive_.narrow(0, position, seq_len_in).copy_(sil);
                    rpu_ddr_flush_force_sized(
                        sin_il_keepalive_.narrow(0, position, seq_len_in).data_ptr<c10::Half>(),
                        il_bytes);
                    ka_last_sin_src_ = sil.data_ptr();
                    ka_last_sin_src_version_ = sin_src_version;
                    if (qwen3vl_2b_w8_profile_) {
                        record_keepalive_segment(ka_sin_sources_, ka_key, sil);
                    }
                }
            }
            ka_last_position_ = position;
            ka_last_seq_ = seq_len_in;
        } else {
            partial_mrope_active_ = false;
        }
        // If position_ids is passed without M-RoPE active, silently ignore
        // (HF callers always supply position_ids; 1D-RoPE path doesn't need them).

        // ─────────────────────────────────────────────────────────────────────
        // R-Phase 2 (Qwen3-VL): DeepStack forward-entry handling
        //
        // Lifecycle (mirrors AdaRMS cond_src_base_ / cond_ref_ template):
        //   1. set_weights stamped deepstack_lang_layers_; build_layer_subgraph
        //      will emit a mutable DMA reading &deepstack_dense_src_base_[i] for
        //      each layer in that list (one src slot per merger index).
        //   2. Caller-supplied dense tensors are held in deepstack_dense_refs_
        //      for the synchronous forward (batch end runs before forward
        //      returns) so their storage cannot be reused by the caching
        //      allocator mid-graph.
        //   3. Per-replay, the mutable DMA cursor-patches the live DDR address.
        //   4. A sized boundary flush on each dense tensor ensures CPU-side writes
        //      (Python scatter-fill at adapter level) are visible to the RPU
        //      DMA engine.
        // ─────────────────────────────────────────────────────────────────────
        const bool qwen3vl_pooler_z1_retained_deepstack_dispatch =
            qwen3vl_pooler_z1_dispatch_ &&
            qwen3vl_pooler_z1_retained_deepstack_count_ > 0;
        const bool qwen3vl_multiview_retained_deepstack_dispatch =
            qwen3vl_multiview_text_composite_dispatch_;
        if (!deepstack_lang_layers_.empty() &&
            !qwen3vl_pooler_z1_retained_deepstack_dispatch &&
            !qwen3vl_multiview_retained_deepstack_dispatch) {
            RECORD_FUNCTION("rpu_causal::deepstack_inputs", {});
            TORCH_CHECK(deepstack_dense_visual_embeds.has_value(),
                        "CausalDecoderModel::forward: deepstack_dense_visual_embeds "
                        "is required when set_weights was called with non-empty "
                        "deepstack_lang_layers (got None).");
            const auto& embeds = *deepstack_dense_visual_embeds;
            TORCH_CHECK(embeds.size() == deepstack_lang_layers_.size(),
                        "CausalDecoderModel::forward: deepstack_dense_visual_embeds "
                        "must have length == deepstack_lang_layers (",
                        deepstack_lang_layers_.size(), "), got ", embeds.size());
            const int64_t h_local = hidden_size();
            deepstack_dense_refs_.clear();
            deepstack_dense_refs_.reserve(embeds.size());
            const bool shared_source =
                qwen3vl_2b_w8_profile_ &&
                !embeds.empty() && embeds.front().defined() &&
                std::all_of(
                    embeds.begin() + 1, embeds.end(),
                    [&](const at::Tensor& other) {
                        return other.is_same(embeds.front());
                    });
            const int64_t shared_source_version =
                shared_source && embeds.front().unsafeGetTensorImpl()
                    ->version_counter().enabled()
                ? static_cast<int64_t>(embeds.front().unsafeGetTensorImpl()
                    ->version_counter().current_version()) : -1;
            const bool shared_source_flush_hit =
                shared_source_version >= 0 &&
                shared_source && deepstack_shared_flush_source_.defined() &&
                deepstack_shared_flush_source_.is_same(embeds.front()) &&
                deepstack_shared_flush_version_ == shared_source_version;
            for (size_t i = 0; i < embeds.size(); ++i) {
                const at::Tensor& d = embeds[i];
                TORCH_CHECK(d.defined(),
                            "CausalDecoderModel::forward: deepstack_dense_visual_embeds[",
                            i, "] is undefined");
                TORCH_CHECK(d.dim() == 2,
                            "CausalDecoderModel::forward: deepstack_dense_visual_embeds[",
                            i, "] must be 2D [seq_len, hidden], got ", d.dim(), "D");
                TORCH_CHECK(d.size(0) >= seq_len_in,
                            "CausalDecoderModel::forward: deepstack_dense_visual_embeds[",
                            i, "].size(0)=", d.size(0),
                            " must be >= seq_len=", seq_len_in,
                            " (max src_offset_bytes baked into BUILD-time graph is "
                            "(seq_len - chunk.len) * hidden * 2)");
                TORCH_CHECK(d.size(1) == h_local,
                            "CausalDecoderModel::forward: deepstack_dense_visual_embeds[",
                            i, "].size(1)=", d.size(1),
                            " must equal hidden_size=", h_local);
                TORCH_CHECK(d.scalar_type() == at::kHalf,
                            "CausalDecoderModel::forward: deepstack_dense_visual_embeds[",
                            i, "] must be fp16, got ", d.scalar_type());
                TORCH_CHECK(d.is_contiguous(),
                            "CausalDecoderModel::forward: deepstack_dense_visual_embeds[",
                            i, "] must be contiguous");
                TORCH_CHECK(d.device().type() == at::kPrivateUse1,
                            "CausalDecoderModel::forward: deepstack_dense_visual_embeds[",
                            i, "] must be on RPU device, got ", d.device());
                deepstack_dense_refs_.push_back(d);
                deepstack_dense_src_base_[i] =
                    ::rhino_lkn::RpuGetDevAddr(d.data_ptr<c10::Half>());
                bool duplicate_source = false;
                for (size_t j = 0; qwen3vl_2b_w8_profile_ && j < i; ++j) {
                    if (embeds[j].is_same(d)) {
                        duplicate_source = true;
                        break;
                    }
                }
                if (!duplicate_source && !shared_source_flush_hit) {
                    RECORD_FUNCTION("rpu_causal::deepstack_boundary_flush", {});
                    // Chunk DMAs read only this contiguous view. A decode view
                    // is one row of a much larger zero keepalive; flushing its
                    // whole allocator segment repeats tens of MiB per merger.
                    rpu_ddr_flush_force_sized(d.data_ptr<c10::Half>(), d.nbytes());
                }
            }
            if (shared_source) {
                deepstack_shared_flush_source_ = embeds.front();
                deepstack_shared_flush_version_ = shared_source_version;
            }
        } else if (qwen3vl_pooler_z1_retained_deepstack_dispatch ||
                   qwen3vl_multiview_retained_deepstack_dispatch) {
            TORCH_CHECK(
                !deepstack_dense_visual_embeds.has_value() ||
                    deepstack_dense_visual_embeds->empty(),
                "CausalDecoderModel::forward: Qwen3-VL retained DeepStack "
                "consumes SPM payloads and rejects dense DDR embeds");
            deepstack_dense_refs_.clear();
            deepstack_dense_src_base_.fill(0);
        } else if (deepstack_dense_visual_embeds.has_value()
                   && !deepstack_dense_visual_embeds->empty()) {
            // DeepStack disabled in set_weights; caller passed dense tensors
            // anyway. Silently ignore (same pattern as position_ids on 1D-RoPE
            // path) — avoids breaking Qwen3-VL callers that build dense embeds
            // unconditionally and let the C++ side decide.
        }

        const bool decode_fused_lm_head = fuse_lm_head_ && seq_len_in == 1;
        const bool fused_lm_head_this_forward = decode_fused_lm_head ||
            (fuse_lm_head_ && fuse_prefill_last_lm_head_ &&
             is_causal && seq_len_in > 1);
        if (fused_lm_head_this_forward) {
            RECORD_FUNCTION("rpu_causal::lm_head_output_setup", {});
            TORCH_CHECK(vocab_size_ > 0 && vocab_size_ % lm_head_tp() == 0,
                        "CausalDecoderModel::forward: fuse_lm_head requires "
                        "vocab_size_=", vocab_size_, " > 0 and divisible by ",
                        lm_head_tp());
            const int64_t output_width = vocab_size_ +
                (lm_head_exact_candidates_ > 0 ? hidden_size() : 0);
            if (!reuse_lm_head_output_ ||
                !lm_head_logits_out_.defined() ||
                lm_head_logits_out_.size(0) != hidden_states.size(0) ||
                lm_head_logits_out_.size(2) != output_width) {
                lm_head_logits_out_ = at::empty(
                    {hidden_states.size(0), 1, output_width},
                    hidden_states.options());
            }
            lm_head_logits_dst_base_ =
                ::rhino_lkn::RpuGetDevAddr(lm_head_logits_out_.data_ptr<c10::Half>());
        } else {
            lm_head_logits_out_ = at::Tensor{};
            lm_head_logits_dst_base_ = 0;
        }

        // Q staging slot is consumed ONLY by the KV_FIRST no-mask branch
        // (build_layer_subgraph: !ctx().is_causal && !ctx().attention_mask) — gate the
        // alloc on the SAME condition (forward params → ctx, equivalent). Causal /
        // explicit-mask forwards never read q_ddr_slots_; minting a permanent per-shape
        // slot for them would grow the map (and trip the 128 cap) on a long-lived
        // CAUSAL decoder that sees many distinct seq_lens — a non-HALO regression
        // (cold-panel round-4). The original single grow-only q_ddr_buf_ also allocated
        // unconditionally, but as ONE reused buffer it had no count growth/cap; the
        // map+cap made unconditional alloc unsafe, so gate it. Stable per-shape slot
        // keyed by {seq_len, width}: created once, never reallocated → an already-
        // captured smaller-seq graph keeps a valid fixed-DMA address after a larger-seq
        // forward; width in the key also covers set_weights head-dim changes.
        if (!is_causal && !attention_mask.has_value()) {
            const int64_t q_ddr_local_dim = (num_q_heads() / attn_tp()) * head_dim();
            const std::pair<int64_t, int64_t> q_key{seq_len_in, q_ddr_local_dim};
            if (q_ddr_slots_.find(q_key) == q_ddr_slots_.end()) {
                TORCH_CHECK(q_ddr_slots_.size() < 128,
                            "CausalDecoderModel: q_ddr_slots_ exceeded 128 distinct "
                            "(seq_len,width) KV_FIRST shapes on one handle — likely a "
                            "runaway shape loop (per-shape Q staging is never freed for "
                            "replay-address stability; normal use sees a handful)");
                q_ddr_slots_.emplace(q_key, at::empty(
                    {static_cast<int64_t>(NUM_CORES), seq_len_in, q_ddr_local_dim},
                    at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1)));
            }
        }
        seq_len_ = seq_len_in;

        auto clear_checked_decode = c10::make_scope_exit([&] {
            qwen3vl_decode_replay_scope_active_ = false;
            cancel_qwen3vl_decode_replay();
        });
        const bool checked_decode_scope = begin_qwen3vl_decode_replay(
            hidden_states, k_caches, v_caches, attention_mask, position,
            is_causal, planned_stage_descriptor);
        qwen3vl_decode_replay_scope_active_ = checked_decode_scope;

        // 透传给 run_all_layers (入口做 batch==1 / dim==3 / contiguous / hidden_size 检查)
        at::Tensor result;
        {
            // Includes descriptor decode/rebase in the public FMB wrapper,
            // which precedes the existing run_all_layers_impl profiler scope.
            RECORD_FUNCTION("rpu_causal::fmb_dispatch", {});
            result = run_all_layers(hidden_states, k_caches, v_caches,
                                    attention_mask, position, is_causal,
                                    planned_chunk_size,
                                    planned_stage_descriptor);
        }

        // Commit only after the layer body completed. A failed forward cannot
        // make a later replay appear owner-safe.
        commit_fast_replay_prefill_owners(k_caches, v_caches);

        if (checked_decode_scope) finish_qwen3vl_decode_replay();

        // ★ Phase 2.5 fused lm_head 后处理:
        //   build_layer_subgraph 在最后一层追加 lm_head GEMM,随后用 graph-aware
        //   mutable scatter DMA 将 8 个 core 的 SPM shard 写入 lm_head_logits_out_.
        //   返回的 logits tensor 因此由 GraphCache end() 统一 BUILD/REPLAY 写入,
        //   不再在 graph scope 内提前 host memcpy 读取尚未执行的 SPM 内容.
        //
        //   条件: decode，或已显式启用并验证的 prefill-last W4 路径。
        if (fused_lm_head_this_forward) {
            rpu_ddr_flush(lm_head_logits_out_.data_ptr<c10::Half>());
            return lm_head_logits_out_;
        }

        return result;
    }

    // ── Phase 2.5: fused lm_head 入口 (per-instance state, called from C API) ──

    void set_lm_head(
        const at::Tensor& lm_head_weight,
        const std::optional<at::Tensor>& lm_head_scale = std::nullopt,
        bool fuse_prefill_last = false,
        int64_t exact_candidates = 0) {
        TORCH_CHECK(lm_head_weight.defined(),
                    "set_lm_head: lm_head_weight must be defined");
        const bool reduced_qwen3_w8_head =
            !has_mrope_ && num_layers() == 28 &&
            hidden_size() == 2048 && intermediate_size() == 6144 &&
            num_q_heads() == 16 && num_kv_heads() == 8 && head_dim() == 128 &&
            !layer_weights_.empty() && layer_weights_[0].q_w.scalar_type() == at::kChar &&
            lm_head_weight.scalar_type() == at::kChar;
        TORCH_CHECK(num_cores() == 8 ||
                        (((lm_head_weight.scalar_type() == at::kHalf &&
                         (!lm_head_scale.has_value() || !lm_head_scale->defined() ||
                          lm_head_scale->numel() == 0)) || reduced_qwen3_w8_head) &&
                         !fuse_prefill_last && exact_candidates == 0),
                    "reduced-core decoder requires FP16 or exact Qwen3-1.7B W8A16 decode head");
        const bool lm_head_is_w8a16 =
            lm_head_weight.scalar_type() == at::kChar;
        const bool lm_head_is_w4a16 =
            lm_head_weight.scalar_type() == at::kByte;
        TORCH_CHECK(lm_head_weight.dim() == 2,
                    "set_lm_head: lm_head_weight must be 2D, got ",
                    lm_head_weight.dim(), "D");
        const int64_t expected_storage_k =
            lm_head_is_w4a16 ? hidden_size() / 2 : hidden_size();
        TORCH_CHECK(lm_head_weight.size(1) == expected_storage_k,
                    "set_lm_head: lm_head_weight.size(1)=", lm_head_weight.size(1),
                    " must match ", expected_storage_k,
                    lm_head_is_w4a16 ? " packed bytes" : " hidden elements");
        TORCH_CHECK(configured_vocab_size_ == 0 ||
                        lm_head_weight.size(0) == configured_vocab_size_,
                    "set_lm_head: weight vocab_size=", lm_head_weight.size(0),
                    " does not match cold configured vocab_size=", configured_vocab_size_);
        TORCH_CHECK(lm_head_weight.size(0) > 0
                    && lm_head_weight.size(0) % lm_head_tp() == 0,
                    "set_lm_head: vocab_size=", lm_head_weight.size(0),
                    " must be > 0 and divisible by lm_head_tp=", lm_head_tp());
        TORCH_CHECK(lm_head_weight.size(0) <= QWEN3_FUSED_LM_HEAD_MAX_VOCAB,
                    "set_lm_head: vocab_size=", lm_head_weight.size(0),
                    " exceeds QWEN3_FUSED_LM_HEAD_MAX_VOCAB=",
                    QWEN3_FUSED_LM_HEAD_MAX_VOCAB,
                    ". Bump the constant in rpu_qwen3_model.cpp if a larger vocab "
                    "model is being added (also re-check the 8109 KB/core SPM budget).");
        TORCH_CHECK(lm_head_weight.scalar_type() == at::kHalf ||
                        lm_head_is_w8a16 || lm_head_is_w4a16,
                    "set_lm_head: lm_head_weight must be FP16, int8, or "
                    "packed uint8, got ",
                    lm_head_weight.scalar_type());
        TORCH_CHECK(lm_head_weight.device().type() == at::kPrivateUse1,
                    "set_lm_head: lm_head_weight must be on RPU device, got ",
                    lm_head_weight.device());
        TORCH_CHECK(exact_candidates == 0 || exact_candidates == 2,
                    "set_lm_head: exact_candidates must be 0 or 2, got ",
                    exact_candidates);
        TORCH_CHECK((!fuse_prefill_last && exact_candidates == 0) ||
                        (qwen3vl_2b_w8_profile_ && lm_head_is_w4a16 && has_mrope_ &&
                         num_layers() == 28 && hidden_size() == 2048 &&
                         lm_head_weight.size(0) == 151936),
                    "set_lm_head: prefill-last/exact-rerank is restricted to "
                    "the validated Qwen3-VL-2B W4 profile");
        if (lm_head_is_w8a16 || lm_head_is_w4a16) {
            TORCH_CHECK(lm_head_scale.has_value() && lm_head_scale->defined(),
                        "set_lm_head: quantized lm_head_weight requires "
                        "lm_head_scale");
            TORCH_CHECK(lm_head_scale->scalar_type() == at::kHalf,
                        "set_lm_head: lm_head_scale must be FP16 (half), got ",
                        lm_head_scale->scalar_type());
            TORCH_CHECK(lm_head_scale->device().type() == at::kPrivateUse1,
                        "set_lm_head: lm_head_scale must be on RPU device, got ",
                        lm_head_scale->device());
            if (lm_head_is_w8a16) {
                TORCH_CHECK(lm_head_scale->dim() == 1 &&
                                lm_head_scale->numel() == lm_head_weight.size(0),
                            "set_lm_head: int8 scale must be [vocab]");
                lm_head_w_scale_ = *lm_head_scale;
            } else {
                TORCH_CHECK(lm_head_scale->dim() == 2,
                            "set_lm_head: packed int4 scale must be 2-D, got ",
                            lm_head_scale->dim(), "D");
                const int64_t group_size = lm_head_scale->size(0);
                TORCH_CHECK(group_size == 32,
                            "set_lm_head: packed int4 group_size must be 32, got ",
                            group_size);
                const int64_t local_groups = hidden_size() / group_size;
                const int64_t local_n = lm_head_weight.size(0) / NUM_CORES;
                const int64_t expected_scale_elements =
                    ((local_groups + 3) / 4) * ((local_n + 63) / 64) *
                    NUM_CORES * 4 * 64;
                TORCH_CHECK(
                    lm_head_scale->numel() == expected_scale_elements,
                            "set_lm_head: packed int4 scale must carry the "
                            "group_size=32 controller-striped payload; got ",
                            lm_head_scale->sizes(), " with ",
                            lm_head_scale->numel(), " elements, expected ",
                            expected_scale_elements);
                lm_head_w_scale_ = rpu_retain_linear_quant_scale(
                    lm_head_weight, *lm_head_scale);
            }
        } else {
            lm_head_w_scale_ = at::Tensor{};
        }
        lm_head_w_ = lm_head_weight;
        vocab_size_ = lm_head_weight.size(0);
        fuse_prefill_last_lm_head_ = fuse_prefill_last;
        lm_head_exact_candidates_ = exact_candidates;
        reuse_lm_head_output_ = false; // Returned logits own storage across subsequent forwards.
        fuse_lm_head_ = true;
        invalidate_model_state();  // D-503: last non-empty statement of set_lm_head
    }

    void clear_lm_head() {
        lm_head_w_ = at::Tensor{};
        lm_head_w_scale_ = at::Tensor{};
        vocab_size_ = 0;
        fuse_prefill_last_lm_head_ = false;
        lm_head_exact_candidates_ = 0;
        reuse_lm_head_output_ = false;
        lm_head_logits_out_ = at::Tensor{};
        lm_head_logits_dst_base_ = 0;
        fuse_lm_head_ = false;
        invalidate_model_state();  // D-503: last non-empty statement of clear_lm_head
    }

protected:
    KvCostLayoutScope capture_kvinsert_cost_layout_scope() override {
        return capture_kvinsert_cost_layout_fields(
            planning_rope_mode_, planning_graph_lifecycle_,
            planning_mode_, qwen3vl_multiview_text_dry_prepared_,
            qwen3vl_multiview_text_composite_prepared_);
    }

    std::vector<int64_t> bind_multiview_text_stage_descriptor(
        const FmbThreeStageChunkPlan& stage_plan, LayoutContext& layout,
        bool partial_mrope, at::IntArrayRef selected_descriptor = {});

    // Subclasses with a native composite forward use the same A6 resolver but
    // must stamp their RoPE mode and lifecycle before the COMPLETE descriptor
    // is minted. Keep those planning-only values scoped to this call so one
    // component cannot leak policy into a later ordinary decoder forward.
    std::vector<FmbPrefillStageCandidate>
    resolve_prefill_stage_domain_with_physical_context(
        int64_t seq_len, int64_t position,
        const std::optional<at::Tensor>& attention_mask, bool is_causal,
        int64_t requested_chunk_size, int64_t logical_len,
        int64_t rope_mode, FmbGraphLifecycle graph_lifecycle,
        CausalDecoderPlanningMode planning_mode =
            CausalDecoderPlanningMode::ORDINARY,
        int64_t planning_chunk_size_override = -1) {
        TORCH_CHECK(
            rope_mode == 0 || rope_mode == 1,
            "RPU_PLANNER_REJECT:CAPABILITY: causal decoder rope_mode must "
            "be 0 (model default) or 1 (partial M-RoPE), got ", rope_mode);
        TORCH_CHECK(
            rope_mode == 0 || has_mrope_,
            "RPU_PLANNER_REJECT:CAPABILITY: partial M-RoPE requires an "
            "M-RoPE decoder handle");
        TORCH_CHECK(
            graph_lifecycle == FmbGraphLifecycle::RETAINED_CACHE ||
                graph_lifecycle == FmbGraphLifecycle::BOUNDED_ONESHOT ||
                graph_lifecycle == FmbGraphLifecycle::NATIVE_COMPOSITE1 ||
                graph_lifecycle == FmbGraphLifecycle::COMPOSITE_CHILD,
            "RPU_PLANNER_REJECT:CAPABILITY: causal decoder graph lifecycle "
            "is invalid: ", static_cast<int64_t>(graph_lifecycle));
        const int64_t saved_rope_mode = planning_rope_mode_;
        const FmbGraphLifecycle saved_lifecycle = planning_graph_lifecycle_;
        const CausalDecoderPlanningMode saved_mode = planning_mode_;
        planning_rope_mode_ = rope_mode;
        planning_graph_lifecycle_ = graph_lifecycle;
        planning_mode_ = planning_mode;
        auto restore = c10::make_scope_exit([&] {
            planning_rope_mode_ = saved_rope_mode;
            planning_graph_lifecycle_ = saved_lifecycle;
            planning_mode_ = saved_mode;
        });
        return resolve_prefill_stage_domain_for_shape(
            seq_len, position, attention_mask, is_causal,
            requested_chunk_size, logical_len, planning_chunk_size_override);
    }

    std::vector<int64_t> encode_causal_physical_stage_descriptor(
        const FmbThreeStageChunkPlan& plan, const LayoutContext& layout,
        int64_t input_chunk_size, int64_t qkv_chunk_size,
        int64_t compute_chunk_size, int64_t physical_len,
        int64_t logical_len, int64_t position, int64_t rope_mode,
        FmbGraphLifecycle graph_lifecycle,
        CausalDecoderPlanningMode planning_mode) {
        const int64_t saved_rope_mode = planning_rope_mode_;
        const FmbGraphLifecycle saved_lifecycle = planning_graph_lifecycle_;
        const CausalDecoderPlanningMode saved_mode = planning_mode_;
        planning_rope_mode_ = rope_mode;
        planning_graph_lifecycle_ = graph_lifecycle;
        planning_mode_ = planning_mode;
        auto restore = c10::make_scope_exit([&] {
            planning_rope_mode_ = saved_rope_mode;
            planning_graph_lifecycle_ = saved_lifecycle;
            planning_mode_ = saved_mode;
        });
        begin_kvinsert_cost_domain_oracle();
        bool completed_oracle = false;
        auto discard_failed_oracle = c10::make_scope_exit([&] {
            if (!completed_oracle) begin_kvinsert_cost_domain_oracle();
        });
        auto context = ctx();
        context.position = position;
        context.seq_len = physical_len;
        context.batch_size = layout.batch_size;
        context.is_causal = layout.is_causal;
        context.stage_plan = plan;
        remember_kvinsert_cost_planning_context(context);
        auto candidates = observe_kvinsert_cost_candidates([&] {
            FmbPhysicalExecutionManifest manifest = physical_manifest_for_candidate(
                plan, layout, physical_len, logical_len, position);
            validate_fmb_physical_manifest(manifest, physical_len, position);
            return std::vector<FmbPrefillStageCandidate>{{
                input_chunk_size, qkv_chunk_size, compute_chunk_size,
                plan, std::move(manifest)}};
        }, &layout);
        finish_kvinsert_cost_domain_oracle(candidates);
        auto descriptor = encode_fmb_prefill_stage_candidate(candidates.front());
        completed_oracle = true;
        return descriptor;
    }

    int deepstack_index_for_layer(int layer_idx) const {
        const auto it = std::find(
            deepstack_lang_layers_.begin(), deepstack_lang_layers_.end(),
            static_cast<int64_t>(layer_idx));
        return it == deepstack_lang_layers_.end()
            ? -1
            : static_cast<int>(
                std::distance(deepstack_lang_layers_.begin(), it));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // declare_buffers
    //
    // Static Qwen3/Qwen3-VL norm weights use one Persistent DDR/SPM slab and
    // one `.preload_callback`. Dynamic/bias decoder variants retain the
    // PersistentPerLayer input/post/q/k slots plus Persistent final_norm slot.
    // Both paths hoist weight DMA out of the per-(layer, chunk) loop.
    //
    // Mirror of the gemma/gemma2 pattern (see rpu_gemma_model.cpp:504-575).
    // ═══════════════════════════════════════════════════════════════════════

    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override {
        auto declarations = explicit_mask_baseline_buffer_declarations(ctx);
        if (explicit_mask_residency_requested_ && ctx.use_attn_mask) {
            (void)detail::plan_forward_spm_residency(declarations, {"sdpa_mask"});
        }
        return declarations;
    }

    virtual std::vector<BufferDecl> explicit_mask_baseline_buffer_declarations(
            const LayoutContext& ctx) {
        return causal_baseline_buffer_declarations(ctx);
    }
    virtual int64_t explicit_mask_body_iterations() const { return 1; }

    // Derived owners must append all their scratch/body-hook declarations
    // before making any forward-wide residency decision.
    std::vector<BufferDecl> causal_baseline_buffer_declarations(
            const LayoutContext& ctx) {
        // KV_FIRST mode is selected (dynamic_config) for every bidirectional
        // no-mask forward, regardless of chunk count. Key the KV_FIRST buffer
        // layout on that SAME condition — NOT on kv_insert_chunk_size>0 — so q_kv
        // is declared whenever emit_kv_first_body runs, including the single-chunk
        // case where the framework leaves kv_insert_chunk_size==0 (KV-insert chunk
        // == compute chunk); otherwise emit would reference an undeclared q_kv.
        // derive kv_first_layout from the LayoutContext
        // parameter (`ctx`), NOT from the runtime InferenceContext (this->ctx()).
        // declare_buffers MUST be a pure function of the hashed LayoutContext —
        // run_all_layers threads is_causal into layout_ctx and the hash folds in
        // the same derived bit (compute_params_hash_impl), so a handle that
        // switches causal↔bidirectional re-allocates instead of reusing the wrong
        // layout. q_kv is KvInsert-scope and aliases with the Compute buffers, so
        // both the auto-chunk probe estimate and the final allocation stay ≈ the
        // non-KV_FIRST footprint (multi-chunk HALO vit_cache path unchanged).
        const bool kv_first_layout = !ctx.is_causal && !ctx.use_attn_mask;
        int64_t comp_cs = ctx.chunk_size;
        int64_t kv_cs = ctx.effective_kv_cs();
        int64_t wide_cs = std::max(comp_cs, kv_cs);
        // Batch decode: B sequences' rows are packed into the GEMM M dim, so
        // every ROW-PARALLEL slot (residual/q/k/v/mlp/down/lm_head_out) holds
        // B * chunk rows. sdpa_tmp deliberately does NOT scale: SDPA is emitted
        // once per sequence at seq_q = chunk, so its scratch stays per-sequence.
        // batch_size == 1 (everything but Qwen3 batch decode) → all sizes
        // byte-identical to before.
        int64_t bs = ctx.batch_size > 0 ? ctx.batch_size : 1;
        int64_t h  = hidden_size();
        int64_t nq = num_q_heads();
        int64_t nkv = num_kv_heads();
        int64_t hd = head_dim();
        int64_t is_ = intermediate_size();
        int64_t nl  = num_layers();

        // attn_tp() == min(NUM_CORES, nkv). For nkv >= NUM_CORES (Qwen3/Llama)
        // this is NUM_CORES (no-op); for nkv < NUM_CORES (Wall-OSS nkv=2) the
        // whole attention phase runs tensor-parallel across attn_tp() cores.
        int tp = attn_tp();
        int64_t local_q = nq / tp;
        int64_t local_kv = nkv * hd / tp;

        auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };
        // DMA-safe per-core bias slot (round element count up to a 256 multiple,
        // mirrors the vision encoder's bias buffers).
        auto dma_safe = [&A](int64_t elems) -> int64_t {
            return A(((elems + 255) / 256) * 256 * DWIDTH);
        };
        int64_t q_bias_sz  = dma_safe(local_q * hd);   // (nq/tp)*hd per core
        int64_t kv_bias_sz = dma_safe(local_kv);       // nkv*hd/tp per core

        int64_t res = A(bs * wide_cs * h * DWIDTH);
        int64_t work_res = res;
        const int64_t attn_rows = comp_cs;
        int64_t q   = A(bs * attn_rows * local_q * hd * DWIDTH);
        int64_t output = A(bs * comp_cs * local_q * hd * DWIDTH);
        // KV_FIRST Q-split (mirrors Gemma q_kv/q_comp): Phase-1 Q lives in its own
        // KvInsert-scope slot sized at the KV-insert chunk, aliasing with the
        // Compute-scope buffers in the disjoint Phase-2 window. Without this split
        // q/output were LayerWide → lost the attention↔MLP phase-aliasing and the
        // seq=1566 bidirectional LM OOM'd at cs=464 (board-confirmed: KV_FIRST
        // footprint 7.98 MB vs non-KV_FIRST 6.35 MB). Non-KV_FIRST omits q_kv and
        // comp_scope==LayerWide → the SEQUENTIAL layout stays byte-identical.
        int64_t q_kv = A(bs * kv_cs * local_q * hd * DWIDTH);
        int64_t kv  = A(bs * (kv_first_layout ? kv_cs : comp_cs)
                        * local_kv * DWIDTH);
        int64_t mlp = A(bs * comp_cs * (is_ / NUM_CORES) * DWIDTH);
        int64_t down_sz = A(bs * comp_cs * h * DWIDTH);
        int64_t nw  = A(h * DWIDTH);
        int64_t hnw = A(hd * DWIDTH);

        // P4: sdpa_tmp must be sized for the mask type that will ACTUALLY run.
        //   - use_attn_mask → MASK_2D (4): Wall-OSS action-denoise rectangular mask.
        //   - KV_FIRST bidirectional → MASK_NONE (0): is_causal=false runs mask_type=0
        //     (see the no-mask SDPA dispatch in build_layer_subgraph: `sdpa_causal ? 1 : 0`).
        //   - causal PREFILL (comp_cs>1) → LTM (1): the dispatch picks mask 1 for
        //     every chunk of len>1 (build_layer_subgraph: sdpa_causal=is_causal&&seq_len>1).
        //   - causal DECODE (comp_cs<=1) → MASK_NONE (0): a single-token chunk has
        //     seq_len==1 so the SAME dispatch picks mask 0, NOT LTM. (comp_cs here is
        //     ctx.chunk_size = alloc_ctx.chunk_size = chunks[0].len = min(chunk_size,
        //     seq_len) (fused_model_base.cpp run_all_layers); for seq_len==1 decode that
        //     is 1 even though compute_chunks' auto-scan minimum is 16 — so this branch
        //     is LIVE for text_decode, NOT dead.)
        // LTM imposes the extra tk%tm==0 constraint
        // (rpu_helpers.h:139/166/257) that MASK_NONE does NOT → choose_tile_k picks
        // tile_k_LTM ≤ tile_k_NONE → sizing a path with mask=1 can UNDER-allocate vs
        // a real mask=0 launch (SDPA writes past sdpa_tmp, corrupting adjacent SPM).
        // Mask scratch invariant: the original
        // `kv_first_layout ? 0 : 1` sized the causal DECODE path (comp_cs==1) at LTM
        // while the seq_len==1 dispatch runs MASK_NONE — an under-allocation latent
        // landmine for HALO text_decode (component C). Mirror the dispatch predicate
        // EXACTLY: size mask=0 whenever the launch will (KV_FIRST, OR causal with the
        // only chunk len==1); keep mask=1 for causal prefill (comp_cs>1) so the
        // Qwen3/Llama/Gemma/pi05 prefill layout stays byte-identical (LTM dominates
        // any len-1 tail chunk at comp_cs>1). Mirrors GR00T bidirectional self-attn
        // tmp sizing (rpu_gr00t_dit_model / rpu_gr00t_vl_encoder_model).
        const int sdpa_tmp_mask =
            ctx.use_attn_mask ? 4 : ((kv_first_layout || comp_cs <= 1) ? 0 : 1);
        int64_t tmp = A(sdpa_compute_tmp_v16_size(
            make_sdpa_config(sdpa_tmp_mask), comp_cs) * 32);
        const bool spm_kv_by_mha =
            ctx.attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        if (spm_kv_by_mha) {
            // The raw-SPM kernel consumes V transposed as
            // [batch,nKVHeadPerCore,headDim,seq]. Reuse sdpa_tmp for that
            // one-shot value, but size it for the complete tensor.
            tmp = std::max(tmp, A(bs * comp_cs * local_kv * DWIDTH));
        }

        // P4: explicit 2D mask SPM slot. Rows = chunk (cs), cols = max KV
        // (= mask->size(-1) = P+H, base fused_model_base.cpp:791-793), padded to
        // v16. Only declared on the explicit-mask path so the no-mask SPM layout
        // (P1/P2/Qwen3/Llama/Gemma/pi05) stays byte-identical. Mirrors Gemma
        // rpu_gemma_model.cpp:477-479,513.
        int64_t mask_sz = ctx.use_attn_mask
            ? A(comp_cs * CeilDiv(ctx.max_kv_seq_len, (int64_t)16) * 32)
            : 0;

        std::vector<BufferDecl> decls;
        const BufferScope attn_kv_scope =
            kv_first_layout ? BufferScope::KvInsert : BufferScope::LayerWide;
        const BufferScope comp_scope =
            kv_first_layout ? BufferScope::Compute : BufferScope::LayerWide;

        // Structural (alive all phases)
        decls.push_back({"residual1",  res,      1, 8, StorageClass::Temp, 0, nullptr});
        decls.push_back({"input_norm", work_res, 1, 8, StorageClass::Temp, 0, nullptr});
        decls.push_back({"residual2",  0,    0, 0, StorageClass::Temp, 0, "input_norm"});

        // Attention. KV_FIRST: q (Phase-2 reload) and output (SDPA out) are
        // Compute-scope so they alias with the MLP buffers; Phase-1 Q uses the
        // KvInsert-scope q_kv slot (declared only in KV_FIRST). Non-KV_FIRST:
        // comp_scope==LayerWide and q_kv is omitted → byte-identical layout.
        // NOTE: q_kv is keyed on kv_first_layout (= !ctx.is_causal && !ctx.use_attn_mask;
        // Use the attention layout mode, not kv_insert_chunk_size>0 — see the kv_first_layout
        // derivation at the top of this method), so it is declared for EVERY bidirectional
        // no-mask forward, INCLUDING the single-chunk case where kv_insert_chunk_size==0
        // (all HALO vit_cache seqs).
        // Packed half 1 Q is consumed first; its dead slice then carries half
        // 0's SDPA output beside half 1's output for the two-half O-proj.
        const int q_phase_end = 5;
        // The admitted packed decode's consecutive Q/K/V declarations have
        // identical lifetimes. Stable first-fit places their 1024/256/256 B
        // slices contiguously; the emitter also checks the resulting addresses.
        // Multi-token prefill retains the existing lifetimes and layout.
        const bool packed_4b_decode = qwen3vl_4b_packed_decode_qkv(
            ctx.chunk_size, ctx.is_causal, ctx.use_attn_mask, ctx.batch_size);
        const int kv_phase_end = packed_4b_decode ? q_phase_end
            : 4;
        const int output_phase_end = 5;
        const int oproj_phase_start = 5;
        // Raw V -> V^T and by-MHA happen while K/V are still live. Phase 4
        // prevents sdpa_tmp from aliasing its V input; the DDR path retains its
        // established phase-5 layout byte-for-byte.
        const int sdpa_tmp_phase =
            spm_kv_by_mha ? 4 : 5;
        decls.push_back({"q",          q,    2, q_phase_end,
                         StorageClass::Temp, 0, nullptr, comp_scope});
        if (kv_first_layout) {
            decls.push_back({"q_kv",   q_kv, 2, 4, StorageClass::Temp, 0, nullptr, attn_kv_scope});
        }
        decls.push_back({"k",          kv,   2, kv_phase_end,
                         StorageClass::Temp, 0, nullptr, attn_kv_scope});
        decls.push_back({"v",          kv,   2, kv_phase_end,
                         StorageClass::Temp, 0, nullptr, attn_kv_scope});
        decls.push_back({"output",     output, 3, output_phase_end,
                         StorageClass::Temp, 0, nullptr, comp_scope});
        decls.push_back({"oproj",      work_res, oproj_phase_start, 5,
                         StorageClass::Temp, 0, nullptr, comp_scope});
        decls.push_back({"sdpa_tmp",   tmp, sdpa_tmp_phase, sdpa_tmp_phase,
                         StorageClass::Temp, 0, nullptr, comp_scope});

        // P4: explicit 2D mask slot (phase 3..5 — conservative window covering the
        // SDPA consumer so it cannot alias with sdpa_tmp). Conditional
        // so no-mask forwards keep the original layout.
        if (ctx.use_attn_mask) {
            decls.push_back({"sdpa_mask", mask_sz, 3, 5, StorageClass::Temp, 0, nullptr, comp_scope});
        }

        // MLP (phase 7-8, aliased with attention). The sealed multiview Text
        // composite aliases full `up` [7,7] with `down` [8,8].
        const bool exact_multiview_text_layout =
            qwen3vl_multiview_text_dry_prepared_ ||
            qwen3vl_multiview_text_composite_prepared_;
        const int down_phase_start = exact_multiview_text_layout ? 8 : 7;
        if (decode_gate_up_fusion_) {
            decls.push_back({"gate_up", 2 * mlp, 7, 8,
                             StorageClass::Temp, 0, nullptr, comp_scope});
            if (prefill_packed_gate_up_strided_) {
                decls.push_back({"gate_compact", mlp, 7, 8,
                                 StorageClass::Temp, 0, nullptr, comp_scope});
            }
        } else {
            decls.push_back({"gate",   mlp,      7, 8,
                             StorageClass::Temp, 0, nullptr, comp_scope});
            {
                decls.push_back({"up",     mlp,      7, 7,
                                 StorageClass::Temp, 0, nullptr, comp_scope});
            }
        }
        decls.push_back({"down",       down_sz, down_phase_start, 8,
                         StorageClass::Temp, 0, nullptr, comp_scope});

        // ★ Phase 2.5: decode-only fused lm_head output buffer.
        //   The fixed delivery also reuses this single-row slot for last-prefill logits.
        //   Used at last layer when fuse_lm_head_=true && seq_len=1.
        //   Phase (8, 9): first_step=8 forces a different SPM offset from
        //   residual1 (1-8); without that, the aliasing algorithm would
        //   reuse residual1's slot and the lm_head GEMM would overwrite its
        //   own input. See SPM-aliasing notes.
        //   Batch decode: the fused GEMM runs M = bs, so this holds bs rows of
        //   the col-partitioned vocab slice.
        {
            decls.push_back({"lm_head_out",
                             A(bs * (QWEN3_FUSED_LM_HEAD_MAX_VOCAB / lm_head_tp()) * DWIDTH),
                             8, 9, StorageClass::Temp, 0, nullptr});
        }

        // ★ R-Phase 2 (Qwen3-VL): DeepStack scratch buffer.
        //   Only allocated when the user opted into DeepStack injection via
        //   set_weights(deepstack_lang_layers=...). Lifetime is a one-tick
        //   window after the MLP residual (phase 8) and before final_norm
        //   (phase 9) — the framework may alias this slot with anything whose
        //   lifetime range doesn't overlap [8, 8].
        //
        //   Per-instance SPM cost (chunk_size=128):
        //     Qwen3-VL-2B (h=2048):  128 × 2048 × 2 = 512 KB / core
        //     Qwen3-VL-4B (h=2560):  128 × 2560 × 2 = 640 KB / core
        //     Qwen3-VL-8B (h=4096):  128 × 4096 × 2 = 1024 KB / core
        if (!deepstack_lang_layers_.empty() &&
            qwen3vl_pooler_z1_retained_deepstack_count_ == 0 &&
            !qwen3vl_multiview_text_dry_prepared_ &&
            !qwen3vl_multiview_text_composite_prepared_) {
            decls.push_back({"deepstack_scratch",
                             res,  // chunk_size × hidden × DWIDTH (same as residual1)
                             8, 8, StorageClass::Temp, 0, nullptr});
        }

        // ─────────────────────────────────────────────────────────────────────
        // Norm-weight preloads — D-502 PersistentPerLayer + .preload_callback.
        // Callbacks emit only DMA launches into the caller-owned capture.
        // Captured-by-this layer_weights_[L] tensors / final_norm_w_ are
        // stable across forwards because set_weights writes them once and
        // invalidates the graph cache via model_state_gen.
        // ─────────────────────────────────────────────────────────────────────
        // B1: only the replay-safe AdaRMS path re-DMAs these every step, so only it
        // needs the DDR-matching ascending layout. Gated on adarms_mutable_ ⇒ every
        // other decoder (and plain build-per-step AdaRMS) keeps the original
        // descending addresses, byte-identical.
        const bool adarms_fused =
            adarms_mutable_ && adarms_fused_bcast_enabled_;

        if (norm_slab_active()) {
            BufferDecl slab;
            slab.name      = "norm_slab";
            slab.size      = norm_slab_size_bytes();
            slab.storage   = StorageClass::Persistent;
            slab.per_layer = 0;
            slab.scope     = BufferScope::LayerWide;
            slab.preload_callback =
                [this](FusedModelBase&, int /*layer*/, uint32_t core0_addr) {
                    this->ctx().consume_physical_route(FmbRouteFamily::MUTABLE_DMA,
                        QWEN3VL_DELIVERY_NORM_SLAB_SITE, 1, 0,
                        bind_cold_topology_arguments({norm_slab_w_.numel()}));
                    rpu_launch_ddr_broadcast_spm_dma(
                        norm_slab_w_, 0, norm_slab_w_.numel(), core0_addr);
                };
            decls.push_back(slab);
        } else {
            BufferDecl input_norm;
            input_norm.name      = "norm_w";
            input_norm.size      = nw;
            input_norm.storage   = StorageClass::PersistentPerLayer;
            input_norm.per_layer = nl;
            input_norm.scope     = BufferScope::LayerWide;
            input_norm.reverse_layer_alloc = adarms_fused;
            input_norm.preload_callback =
                [this](FusedModelBase&, int L, uint32_t core0_addr) {
                    const at::Tensor& weight =
                        layer_weights_[L].input_norm_w;
                    if (qwen3vl_pooler_z1_dispatch_ ||
                        qwen3vl_multiview_text_composite_dispatch_) {
                        if (this->ctx().has_complete_physical_manifest()) {
                            this->ctx().consume_physical_route(
                                FmbRouteFamily::MUTABLE_DMA,
                                CAUSAL_DECODER_INPUT_NORM_POOLER_PRELOAD_SITE,
                                static_cast<int64_t>(
                                    CausalDecoderMutableDmaRoute::
                                        DDR_BROADCAST_TO_SPM),
                                /*resolved_flags=*/0,
                                bind_cold_topology_arguments({L, hidden_size(),
                                 (qwen3vl_pooler_z1_dispatch_ ||
                                  qwen3vl_multiview_text_composite_dispatch_)
                                     ? 1 : 0}),
                                L);
                        }
                        rpu_launch_ddr_broadcast_spm_dma(
                            weight, 0, hidden_size(), core0_addr, num_cores());
                    } else {
                        if (this->ctx().has_complete_physical_manifest()) {
                            this->ctx().consume_physical_route(
                                FmbRouteFamily::MUTABLE_DMA,
                                CAUSAL_DECODER_INPUT_NORM_DIRECT_PRELOAD_SITE,
                                static_cast<int64_t>(
                                    CausalDecoderMutableDmaRoute::
                                        DDR_BROADCAST_TO_SPM),
                                /*resolved_flags=*/0,
                                bind_cold_topology_arguments({L, hidden_size(),
                                 (qwen3vl_pooler_z1_dispatch_ ||
                                  qwen3vl_multiview_text_composite_dispatch_)
                                     ? 1 : 0}),
                                L);
                        }
                        rpu_launch_ddr_broadcast_spm_dma(
                            weight.data_ptr<c10::Half>(), hidden_size(),
                            core0_addr, num_cores());
                    }
                };
            decls.push_back(input_norm);

            BufferDecl post_norm;
            post_norm.name      = "post_norm_w";
            post_norm.size      = nw;
            post_norm.storage   = StorageClass::PersistentPerLayer;
            post_norm.per_layer = nl;
            post_norm.scope     = BufferScope::LayerWide;
            post_norm.reverse_layer_alloc = adarms_fused;
            post_norm.preload_callback =
                [this](FusedModelBase&, int L, uint32_t core0_addr) {
                    const at::Tensor& weight =
                        layer_weights_[L].post_norm_w;
                    if (qwen3vl_pooler_z1_dispatch_ ||
                        qwen3vl_multiview_text_composite_dispatch_) {
                        if (this->ctx().has_complete_physical_manifest()) {
                            this->ctx().consume_physical_route(
                                FmbRouteFamily::MUTABLE_DMA,
                                CAUSAL_DECODER_POST_NORM_POOLER_PRELOAD_SITE,
                                static_cast<int64_t>(
                                    CausalDecoderMutableDmaRoute::
                                        DDR_BROADCAST_TO_SPM),
                                /*resolved_flags=*/0,
                                bind_cold_topology_arguments({L, hidden_size(),
                                 (qwen3vl_pooler_z1_dispatch_ ||
                                  qwen3vl_multiview_text_composite_dispatch_)
                                     ? 1 : 0}),
                                L);
                        }
                        rpu_launch_ddr_broadcast_spm_dma(
                            weight, 0, hidden_size(), core0_addr, num_cores());
                    } else {
                        if (this->ctx().has_complete_physical_manifest()) {
                            this->ctx().consume_physical_route(
                                FmbRouteFamily::MUTABLE_DMA,
                                CAUSAL_DECODER_POST_NORM_DIRECT_PRELOAD_SITE,
                                static_cast<int64_t>(
                                    CausalDecoderMutableDmaRoute::
                                        DDR_BROADCAST_TO_SPM),
                                /*resolved_flags=*/0,
                                bind_cold_topology_arguments({L, hidden_size(),
                                 (qwen3vl_pooler_z1_dispatch_ ||
                                  qwen3vl_multiview_text_composite_dispatch_)
                                     ? 1 : 0}),
                                L);
                        }
                        rpu_launch_ddr_broadcast_spm_dma(
                            weight.data_ptr<c10::Half>(), hidden_size(),
                            core0_addr, num_cores());
                    }
                };
            decls.push_back(post_norm);
        }

        // AdaRMS FiLM shift vectors (per-layer [hidden]); only when adarms_.
        // Broadcast-added after each RMSNorm in build_layer_subgraph.
        if (adarms_) {
            BufferDecl in_shift;
            in_shift.name      = "input_shift";
            in_shift.size      = nw;
            in_shift.storage   = StorageClass::PersistentPerLayer;
            in_shift.per_layer = nl;
            in_shift.scope     = BufferScope::LayerWide;
            in_shift.reverse_layer_alloc = adarms_fused;
            in_shift.preload_callback =
                [this](FusedModelBase&, int L, uint32_t core0_addr) {
                    if (this->ctx().has_complete_physical_manifest()) {
                        this->ctx().consume_physical_route(
                            FmbRouteFamily::MUTABLE_DMA,
                            CAUSAL_DECODER_ADARMS_INPUT_SHIFT_PRELOAD_SITE,
                            static_cast<int64_t>(
                                CausalDecoderMutableDmaRoute::
                                    DDR_BROADCAST_TO_SPM),
                            /*resolved_flags=*/0,
                            bind_cold_topology_arguments({L, hidden_size(), adarms_ ? 1 : 0,
                             adarms_mutable_ ? 1 : 0,
                             adarms_fused_bcast_enabled_ ? 1 : 0}),
                            L);
                    }
                    rpu_launch_ddr_broadcast_spm_dma(
                        input_shift_[L].data_ptr<c10::Half>(),
                        hidden_size(), core0_addr, num_cores());
                };
            decls.push_back(in_shift);

            BufferDecl po_shift;
            po_shift.name      = "post_shift";
            po_shift.size      = nw;
            po_shift.storage   = StorageClass::PersistentPerLayer;
            po_shift.per_layer = nl;
            po_shift.scope     = BufferScope::LayerWide;
            po_shift.reverse_layer_alloc = adarms_fused;
            po_shift.preload_callback =
                [this](FusedModelBase&, int L, uint32_t core0_addr) {
                    if (this->ctx().has_complete_physical_manifest()) {
                        this->ctx().consume_physical_route(
                            FmbRouteFamily::MUTABLE_DMA,
                            CAUSAL_DECODER_ADARMS_POST_SHIFT_PRELOAD_SITE,
                            static_cast<int64_t>(
                                CausalDecoderMutableDmaRoute::
                                    DDR_BROADCAST_TO_SPM),
                            /*resolved_flags=*/0,
                            bind_cold_topology_arguments({L, hidden_size(), adarms_ ? 1 : 0,
                             adarms_mutable_ ? 1 : 0,
                             adarms_fused_bcast_enabled_ ? 1 : 0}),
                            L);
                    }
                    rpu_launch_ddr_broadcast_spm_dma(
                        post_shift_[L].data_ptr<c10::Half>(),
                        hidden_size(), core0_addr, num_cores());
                };
            decls.push_back(po_shift);
        }

        // q_norm_w / k_norm_w only present when has_qk_norm_ (Qwen3 yes,
        // Llama no — set in set_weights based on input list presence).
        if (has_qk_norm_ && !norm_slab_active()) {
            {
                BufferDecl q_norm;
                q_norm.name      = "q_norm_w";
                q_norm.size      = hnw;
                q_norm.storage   = StorageClass::PersistentPerLayer;
                q_norm.per_layer = nl;
                q_norm.scope     = BufferScope::LayerWide;
                q_norm.preload_callback =
                    [this](FusedModelBase&, int L, uint32_t core0_addr) {
                        const at::Tensor& weight =
                            layer_weights_[L].q_norm_w;
                        if (qwen3vl_pooler_z1_dispatch_ ||
                            qwen3vl_multiview_text_composite_dispatch_) {
                            if (this->ctx().has_complete_physical_manifest()) {
                                this->ctx().consume_physical_route(
                                    FmbRouteFamily::MUTABLE_DMA,
                                    CAUSAL_DECODER_Q_NORM_POOLER_PRELOAD_SITE,
                                    static_cast<int64_t>(
                                        CausalDecoderMutableDmaRoute::
                                            DDR_BROADCAST_TO_SPM),
                                    /*resolved_flags=*/0,
                                    bind_cold_topology_arguments({L, head_dim(), has_qk_norm_ ? 1 : 0,
                                     (qwen3vl_pooler_z1_dispatch_ ||
                                      qwen3vl_multiview_text_composite_dispatch_)
                                         ? 1 : 0}),
                                    L);
                            }
                            rpu_launch_ddr_broadcast_spm_dma(
                                weight, 0, head_dim(), core0_addr, num_cores());
                        } else {
                            if (this->ctx().has_complete_physical_manifest()) {
                                this->ctx().consume_physical_route(
                                    FmbRouteFamily::MUTABLE_DMA,
                                    CAUSAL_DECODER_Q_NORM_DIRECT_PRELOAD_SITE,
                                    static_cast<int64_t>(
                                        CausalDecoderMutableDmaRoute::
                                            DDR_BROADCAST_TO_SPM),
                                    /*resolved_flags=*/0,
                                    bind_cold_topology_arguments({L, head_dim(), has_qk_norm_ ? 1 : 0,
                                     (qwen3vl_pooler_z1_dispatch_ ||
                                      qwen3vl_multiview_text_composite_dispatch_)
                                         ? 1 : 0}),
                                    L);
                            }
                            rpu_launch_ddr_broadcast_spm_dma(
                                weight.data_ptr<c10::Half>(), head_dim(),
                                core0_addr, num_cores());
                        }
                    };
                decls.push_back(q_norm);
            }
            {
                BufferDecl k_norm;
                k_norm.name      = "k_norm_w";
                k_norm.size      = hnw;
                k_norm.storage   = StorageClass::PersistentPerLayer;
                k_norm.per_layer = nl;
                k_norm.scope     = BufferScope::LayerWide;
                k_norm.preload_callback =
                    [this](FusedModelBase&, int L, uint32_t core0_addr) {
                        const at::Tensor& weight =
                            layer_weights_[L].k_norm_w;
                        if (qwen3vl_pooler_z1_dispatch_ ||
                            qwen3vl_multiview_text_composite_dispatch_) {
                            if (this->ctx().has_complete_physical_manifest()) {
                                this->ctx().consume_physical_route(
                                    FmbRouteFamily::MUTABLE_DMA,
                                    CAUSAL_DECODER_K_NORM_POOLER_PRELOAD_SITE,
                                    static_cast<int64_t>(
                                        CausalDecoderMutableDmaRoute::
                                            DDR_BROADCAST_TO_SPM),
                                    /*resolved_flags=*/0,
                                    bind_cold_topology_arguments({L, head_dim(), has_qk_norm_ ? 1 : 0,
                                     (qwen3vl_pooler_z1_dispatch_ ||
                                      qwen3vl_multiview_text_composite_dispatch_)
                                         ? 1 : 0}),
                                    L);
                            }
                            rpu_launch_ddr_broadcast_spm_dma(
                                weight, 0, head_dim(), core0_addr, num_cores());
                        } else {
                            if (this->ctx().has_complete_physical_manifest()) {
                                this->ctx().consume_physical_route(
                                    FmbRouteFamily::MUTABLE_DMA,
                                    CAUSAL_DECODER_K_NORM_DIRECT_PRELOAD_SITE,
                                    static_cast<int64_t>(
                                        CausalDecoderMutableDmaRoute::
                                            DDR_BROADCAST_TO_SPM),
                                    /*resolved_flags=*/0,
                                    bind_cold_topology_arguments({L, head_dim(), has_qk_norm_ ? 1 : 0,
                                     (qwen3vl_pooler_z1_dispatch_ ||
                                      qwen3vl_multiview_text_composite_dispatch_)
                                         ? 1 : 0}),
                                    L);
                            }
                            rpu_launch_ddr_broadcast_spm_dma(
                                weight.data_ptr<c10::Half>(), head_dim(),
                                core0_addr, num_cores());
                        }
                    };
                decls.push_back(k_norm);
            }
        }

        // q_bias / k_bias / v_bias only present when has_qkv_bias_ (Qwen2.5 /
        // Wall-OSS). Each is a persistent per-layer SPM slot scattered per
        // col-partition (core c gets the contiguous output-channel slice that
        // the col-swizzled weight produces on core c — bias is NOT swizzled,
        // matching the verified vision-encoder layout). GQA: Q slot is
        // (nq/tp)*hd per core, K/V slots are nkv*hd/tp per core (different
        // sizes — vision assumes nkv==nq and cannot be copied verbatim).
        if (has_qkv_bias_) {
            auto add_bias = [&](const char* name, int64_t sz,
                                at::Tensor LayerWeights::* member,
                                int64_t role) {
                BufferDecl b;
                b.name      = name;
                b.size      = sz;
                b.storage   = StorageClass::PersistentPerLayer;
                b.per_layer = nl;
                b.scope     = BufferScope::LayerWide;
                b.preload_callback =
                    [this, member, role](FusedModelBase&, int L,
                                         uint32_t core0_addr) {
                        int tp_ = attn_tp();
                        const at::Tensor& bw = layer_weights_[L].*member;
                        int64_t per_core = bw.numel() / tp_;
                        if (this->ctx().has_complete_physical_manifest()) {
                            this->ctx().consume_physical_route(
                                FmbRouteFamily::MUTABLE_DMA,
                                CAUSAL_DECODER_QKV_BIAS_PRELOAD_SITE,
                                static_cast<int64_t>(
                                    CausalDecoderMutableDmaRoute::
                                        DDR_SCATTER_TO_SPM),
                                /*resolved_flags=*/0,
                                bind_cold_topology_arguments({L, role, per_core, per_core * DWIDTH, tp_,
                                 has_qkv_bias_ ? 1 : 0}),
                                3 * L + role);
                        }
                        rpu_launch_ddr_scatter_spm_dma(
                            bw.data_ptr<c10::Half>(),
                            per_core, per_core * DWIDTH,
                            core0_addr, tp_);
                    };
                decls.push_back(b);
            };
            add_bias("q_bias", q_bias_sz,  &LayerWeights::q_bias_w, 0);
            add_bias("k_bias", kv_bias_sz, &LayerWeights::k_bias_w, 1);
            add_bias("v_bias", kv_bias_sz, &LayerWeights::v_bias_w, 2);
        }

        // Tiled NVFP4 consumes FP32 tensor scales from SPM in both accumulation modes:
        // params 24/25 carry this projection-family array address and param 27
        // carries layer_id. These model-wide arrays are persistent so graph
        // replay cannot observe out-of-band SPM contents.
        if (nvfp4_) {
            auto add_nvfp4_ts =
                [&](const char* name, at::Tensor CausalDecoderModel::* member,
                    int64_t role) {
                    BufferDecl ts;
                    ts.name       = name;
                    const int64_t scale_elems = (this->*member).numel();
                    ts.size       = A(scale_elems * static_cast<int64_t>(sizeof(float)));
                    ts.storage    = StorageClass::Persistent;
                    ts.per_layer  = 0;
                    ts.scope      = BufferScope::LayerWide;
                    ts.preload_callback =
                        [this, member, role](FusedModelBase&, int,
                                             uint32_t core0_addr) {
                            const int64_t scale_elems =
                                (this->*member).numel();
                            if (this->ctx().has_complete_physical_manifest()) {
                                this->ctx().consume_physical_route(
                                    FmbRouteFamily::MUTABLE_DMA,
                                    CAUSAL_DECODER_NVFP4_SCALE_PRELOAD_SITE,
                                    static_cast<int64_t>(
                                        CausalDecoderMutableDmaRoute::
                                            DDR_BROADCAST_TO_SPM),
                                    /*resolved_flags=*/0,
                                    bind_cold_topology_arguments({role, scale_elems, 2 * scale_elems,
                                     nvfp4_ ? 1 : 0}),
                                    role);
                            }
                            auto* src = reinterpret_cast<c10::Half*>(
                                (this->*member).data_ptr<float>());
                            rpu_launch_ddr_broadcast_spm_dma(
                                src, 2 * scale_elems, core0_addr, num_cores());
                        };
                    decls.push_back(ts);
                };
            add_nvfp4_ts("nvfp4_q_ts", &CausalDecoderModel::q_nvfp4_tensor_scales_, 0);
            add_nvfp4_ts("nvfp4_k_ts", &CausalDecoderModel::k_nvfp4_tensor_scales_, 1);
            add_nvfp4_ts("nvfp4_v_ts", &CausalDecoderModel::v_nvfp4_tensor_scales_, 2);
            add_nvfp4_ts("nvfp4_o_ts", &CausalDecoderModel::o_nvfp4_tensor_scales_, 3);
            add_nvfp4_ts("nvfp4_gate_ts", &CausalDecoderModel::gate_nvfp4_tensor_scales_, 4);
            add_nvfp4_ts("nvfp4_up_ts", &CausalDecoderModel::up_nvfp4_tensor_scales_, 5);
            add_nvfp4_ts("nvfp4_down_ts", &CausalDecoderModel::down_nvfp4_tensor_scales_, 6);
        }

        // ★ final_norm fuse: single Persistent slot. The same final_norm_w_
        //   tensor is read at the last layer's RMSNorm. Persistent (not
        //   PersistentPerLayer) because it's one weight, not 28.
        if (!norm_slab_active()) {
            BufferDecl final_norm;
            final_norm.name     = "final_norm_w";
            final_norm.size     = nw;
            final_norm.storage  = StorageClass::Persistent;
            final_norm.per_layer = 0;
            final_norm.scope    = BufferScope::LayerWide;
            final_norm.preload_callback =
                [this](FusedModelBase&, int /*layer*/, uint32_t core0_addr) {
                    if (qwen3vl_pooler_z1_dispatch_ ||
                        qwen3vl_multiview_text_composite_dispatch_) {
                        if (this->ctx().has_complete_physical_manifest()) {
                            this->ctx().consume_physical_route(
                                FmbRouteFamily::MUTABLE_DMA,
                                CAUSAL_DECODER_FINAL_NORM_POOLER_PRELOAD_SITE,
                                static_cast<int64_t>(
                                    CausalDecoderMutableDmaRoute::
                                        DDR_BROADCAST_TO_SPM),
                                /*resolved_flags=*/0,
                                bind_cold_topology_arguments({hidden_size(),
                                 (qwen3vl_pooler_z1_dispatch_ ||
                                  qwen3vl_multiview_text_composite_dispatch_)
                                     ? 1 : 0}));
                        }
                        rpu_launch_ddr_broadcast_spm_dma(
                            final_norm_w_, 0, hidden_size(), core0_addr, num_cores());
                    } else {
                        if (this->ctx().has_complete_physical_manifest()) {
                            this->ctx().consume_physical_route(
                                FmbRouteFamily::MUTABLE_DMA,
                                CAUSAL_DECODER_FINAL_NORM_DIRECT_PRELOAD_SITE,
                                static_cast<int64_t>(
                                    CausalDecoderMutableDmaRoute::
                                        DDR_BROADCAST_TO_SPM),
                                /*resolved_flags=*/0,
                                bind_cold_topology_arguments({hidden_size(),
                                 (qwen3vl_pooler_z1_dispatch_ ||
                                  qwen3vl_multiview_text_composite_dispatch_)
                                     ? 1 : 0}));
                        }
                        rpu_launch_ddr_broadcast_spm_dma(
                            final_norm_w_.data_ptr<c10::Half>(), hidden_size(),
                            core0_addr, num_cores());
                    }
                };
            decls.push_back(final_norm);
        }

        return decls;
    }

    // DeepStack can move Temp offsets without changing the generic params
    // hash. The other uncovered gates here
    // (adarms_, has_qk_norm_, has_qkv_bias_, nvfp4_ and the nvfp4 scale numels)
    // all guard Persistent/PersistentPerLayer decls, which the persistent-shape
    // check in ensure_allocated already turns into a hard abort — folding those
    // in would convert a loud error into a silent re-layout, which is the wrong
    // direction. Only emptiness matters; no size depends on the layer indices.
    // Subclasses that append their own decls chain onto this value.
    int64_t subclass_layout_hash() const override {
        int64_t h = detail::layout_mix(
            0, deepstack_lang_layers_.empty() ? 0 : 1);
        if (rhinovla_high_precision_) {
            h = detail::layout_mix(h, RHINOVLA_TEXT_HIGH_RMSNORM_SITE);
            h = detail::layout_mix(h, rhinovla_high_precision_fusions_);
        }
        if (explicit_mask_residency_requested_) {
            h = detail::layout_mix(h, CAUSAL_DECODER_MASK_SPM_SCHEDULE_SITE);
        }
        h = detail::layout_mix(h, decode_gate_up_fusion_ ? 1 : 0);
        h = detail::layout_mix(h, prefill_packed_gate_up_strided_ ? 1 : 0);
        if (qwen3vl_4b_packed_qkv_weights_) {
            h = detail::layout_mix(h, CAUSAL_DECODER_4B_W8_PACKED_QKV_SITE);
        }
        if (qwen3vl_4b_decode_qk_fused_) {
            h = detail::layout_mix(h, QWEN3VL_4B_DECODE_QK_NORM_MROPE_SITE);
        }
        if (qwen3vl_4b_decode_qk_kv_fused_) {
            h = detail::layout_mix(h, QWEN3VL_4B_DECODE_QK_NORM_MROPE_KV_INSERT_SITE);
        }
        if (qwen3vl_4b_decode_o_ring_norm_fused_) {
            h = detail::layout_mix(h, QWEN3VL_4B_DECODE_O_RING_NORM_SITE);
        }
        if (configured_vocab_size_ != 0 || num_cores() != 8) {
            h = detail::layout_mix(h, 2);  // cold template schema version
            for (const auto value : {num_cores(), attn_tp(), mlp_tp(), lm_head_tp(), NUM_CORES}) {
                h = detail::layout_mix(h, value);
            }
        }
        return h;
    }

    bool subclass_spm_kv_by_mha_eligible(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t position) const override {
        if (num_cores() != 8 || !qwen3_spm_kv_by_mha_enabled_ ||
            typeid(*this) != typeid(CausalDecoderModel) ||
            layout.attention_policy !=
                AttentionExecutionPolicy::SPM_KV_BY_MHA ||
            position != 0 || !layout.is_causal || layout.use_attn_mask ||
            layout.batch_size != 1 ||
            plan.chunk_mode != ChunkMode::SEQUENTIAL ||
            plan.input.chunks.size() != 1 ||
            plan.qkv.chunks.size() != 1 ||
            plan.compute.chunks.size() != 1 || plan.spans.size() != 1) {
            return false;
        }

        const ChunkInfo& chunk = plan.compute.chunks.front();
        const int64_t seq = chunk.len;
        if ((seq != 16 && seq != 32 && seq != 64 && seq != 128) ||
            chunk.offset != 0 || chunk.kv_seq_len != seq ||
            plan.qkv.chunks.front().len != seq ||
            plan.input.chunks.front().len != seq ||
            plan.spans.front().offset != 0 || plan.spans.front().len != seq ||
            layout.chunk_size != seq || layout.max_kv_seq_len != seq) {
            return false;
        }

        const bool exact_dense_profile =
            num_q_heads() == 16 && num_kv_heads() == 8 &&
            head_dim() == 128 && num_layers() == 28 &&
            ((hidden_size() == 1024 && intermediate_size() == 3072) ||
             (hidden_size() == 2048 && intermediate_size() == 6144));
        if (!exact_dense_profile || layer_weights_.size() != 28 ||
            !use_silu_ || !has_qk_norm_ || has_qkv_bias_ ||
            nvfp4_ || has_mrope_ || partial_mrope_active_ ||
            !deepstack_lang_layers_.empty() || adarms_ || adarms_mutable_ ||
            adarms_unroll_ || adarms_schedule_select_ ||
            rope_position_base_ >= 0 || batch_decode_active_ ||
            wall_z1_dispatch_ || qwen3vl_pooler_z1_dispatch_ ||
            qwen3vl_multiview_text_composite_dispatch_ ||
            planning_mode_ != CausalDecoderPlanningMode::ORDINARY ||
            sdpa_kernel_ != SdpaKernelType::FLASH_ATTN_SPM) {
            return false;
        }

        return std::all_of(
            layer_weights_.begin(), layer_weights_.end(),
            [](const LayerWeights& layer) {
                return layer.q_w.scalar_type() == at::kHalf &&
                    layer.k_w.scalar_type() == at::kHalf &&
                    layer.v_w.scalar_type() == at::kHalf &&
                    layer.o_w.scalar_type() == at::kHalf &&
                    layer.gate_w.scalar_type() == at::kHalf &&
                    layer.up_w.scalar_type() == at::kHalf &&
                    layer.down_w.scalar_type() == at::kHalf;
            });
    }

    // Publish every physical route consumed by preload callbacks inherited
    // from CausalDecoderModel::declare_buffers().  Derived owners that build
    // their own COMPLETE descriptor must compose this fragment as well; the
    // callbacks still execute inside their Graph capture.
    void append_causal_decoder_preload_manifest_routes(
        FmbPhysicalExecutionManifest& manifest) const {
        TORCH_CHECK(
            manifest.state == FmbPhysicalManifestState::COMPLETE,
            "RPU_PLANNER_REJECT:CAPABILITY: causal decoder persistent "
            "preloads require a COMPLETE physical manifest");

        const bool tensor_preload =
            qwen3vl_pooler_z1_dispatch_ ||
            qwen3vl_multiview_text_composite_dispatch_ ||
            planning_mode_ ==
                CausalDecoderPlanningMode::QWEN3VL_POOLER_Z1 ||
            planning_mode_ ==
                CausalDecoderPlanningMode::QWEN3VL_MULTIVIEW_TEXT;
        const int64_t broadcast_selector = static_cast<int64_t>(
            CausalDecoderMutableDmaRoute::DDR_BROADCAST_TO_SPM);
        const int64_t scatter_selector = static_cast<int64_t>(
            CausalDecoderMutableDmaRoute::DDR_SCATTER_TO_SPM);
        const auto append = [&](int64_t site_id, int64_t selector,
                                std::vector<int64_t> arguments = {},
                                int64_t invocation = 0) {
            arguments = bind_cold_topology_arguments(std::move(arguments));
            manifest.routes.push_back({
                site_id, FmbRouteFamily::MUTABLE_DMA, selector,
                /*flags=*/0, std::move(arguments), invocation});
        };

        if (norm_slab_active()) {
            append(
                QWEN3VL_DELIVERY_NORM_SLAB_SITE, broadcast_selector,
                {norm_slab_w_.numel()});
        } else {
            for (int64_t layer = 0; layer < num_layers(); ++layer) {
                append(
                    tensor_preload
                        ? CAUSAL_DECODER_INPUT_NORM_POOLER_PRELOAD_SITE
                        : CAUSAL_DECODER_INPUT_NORM_DIRECT_PRELOAD_SITE,
                    broadcast_selector,
                    {layer, hidden_size(), tensor_preload ? 1 : 0}, layer);
                append(
                    tensor_preload
                        ? CAUSAL_DECODER_POST_NORM_POOLER_PRELOAD_SITE
                        : CAUSAL_DECODER_POST_NORM_DIRECT_PRELOAD_SITE,
                    broadcast_selector,
                    {layer, hidden_size(), tensor_preload ? 1 : 0}, layer);
                if (adarms_) {
                    append(
                        CAUSAL_DECODER_ADARMS_INPUT_SHIFT_PRELOAD_SITE,
                        broadcast_selector,
                        {layer, hidden_size(), adarms_ ? 1 : 0,
                         adarms_mutable_ ? 1 : 0,
                         adarms_fused_bcast_enabled_ ? 1 : 0},
                        layer);
                    append(
                        CAUSAL_DECODER_ADARMS_POST_SHIFT_PRELOAD_SITE,
                        broadcast_selector,
                        {layer, hidden_size(), adarms_ ? 1 : 0,
                         adarms_mutable_ ? 1 : 0,
                         adarms_fused_bcast_enabled_ ? 1 : 0},
                        layer);
                }
                if (has_qk_norm_) {
                    append(
                        tensor_preload
                            ? CAUSAL_DECODER_Q_NORM_POOLER_PRELOAD_SITE
                            : CAUSAL_DECODER_Q_NORM_DIRECT_PRELOAD_SITE,
                        broadcast_selector,
                        {layer, head_dim(), has_qk_norm_ ? 1 : 0,
                         tensor_preload ? 1 : 0},
                        layer);
                    append(
                        tensor_preload
                            ? CAUSAL_DECODER_K_NORM_POOLER_PRELOAD_SITE
                            : CAUSAL_DECODER_K_NORM_DIRECT_PRELOAD_SITE,
                        broadcast_selector,
                        {layer, head_dim(), has_qk_norm_ ? 1 : 0,
                         tensor_preload ? 1 : 0},
                        layer);
                }
                if (has_qkv_bias_) {
                    const int64_t q_per_core =
                        num_q_heads() * head_dim() / attn_tp();
                    const int64_t kv_per_core =
                        num_kv_heads() * head_dim() / attn_tp();
                    for (int64_t role = 0; role < 3; ++role) {
                        const int64_t per_core =
                            role == 0 ? q_per_core : kv_per_core;
                        append(
                            CAUSAL_DECODER_QKV_BIAS_PRELOAD_SITE,
                            scatter_selector,
                            {layer, role, per_core, per_core * DWIDTH,
                             attn_tp(), has_qkv_bias_ ? 1 : 0},
                            3 * layer + role);
                    }
                }
            }
        }

        if (nvfp4_) {
            const auto append_nvfp4_scale = [&](int64_t role,
                                                const at::Tensor& scale) {
                append(
                    CAUSAL_DECODER_NVFP4_SCALE_PRELOAD_SITE,
                    broadcast_selector,
                    {role, scale.numel(), 2 * scale.numel(),
                     nvfp4_ ? 1 : 0},
                    role);
            };
            append_nvfp4_scale(0, q_nvfp4_tensor_scales_);
            append_nvfp4_scale(1, k_nvfp4_tensor_scales_);
            append_nvfp4_scale(2, v_nvfp4_tensor_scales_);
            append_nvfp4_scale(3, o_nvfp4_tensor_scales_);
            append_nvfp4_scale(4, gate_nvfp4_tensor_scales_);
            append_nvfp4_scale(5, up_nvfp4_tensor_scales_);
            append_nvfp4_scale(6, down_nvfp4_tensor_scales_);
        }

        if (!norm_slab_active()) {
            append(
                tensor_preload
                    ? CAUSAL_DECODER_FINAL_NORM_POOLER_PRELOAD_SITE
                    : CAUSAL_DECODER_FINAL_NORM_DIRECT_PRELOAD_SITE,
                broadcast_selector,
                {hidden_size(), tensor_preload ? 1 : 0});
        }
    }

    bool qwen3vl_4b_w8_resident_prefill(
        int64_t physical_len, int64_t position, bool is_causal,
        bool use_attn_mask, int64_t batch_size, int64_t num_chunks) const {
        return qwen3vl_4b_w8_profile_ && physical_len > 1 &&
            position == 0 && is_causal && !use_attn_mask && batch_size == 1 &&
            num_chunks > 1 && !linear_acc32_ && !adarms_ &&
            planning_mode_ == CausalDecoderPlanningMode::ORDINARY &&
            !qwen3vl_pooler_z1_dispatch_ &&
            !qwen3vl_multiview_text_composite_dispatch_;
    }

    bool qwen3vl_ordinary_resident_prefill(
        int64_t physical_len, int64_t position, bool is_causal,
        bool use_attn_mask, int64_t batch_size, int64_t num_chunks) const {
        return qwen3vl_ordinary_resident_prefill_profile_ && physical_len > 1 &&
            position == 0 && is_causal && !use_attn_mask && batch_size == 1 &&
            num_chunks > 1 && !adarms_ && planning_mode_ == CausalDecoderPlanningMode::ORDINARY &&
            !decode_gate_up_fusion_ && !pre_layers_residual1_ready_ && !wall_z1_dispatch_ &&
            !qwen3vl_pooler_z1_dispatch_ &&
            !qwen3vl_multiview_text_composite_dispatch_;
    }

    std::vector<int64_t> qwen3vl_ordinary_prefill_schedule_arguments(
        int64_t physical_len, int64_t chunk_size, int64_t num_chunks) const {
        return {physical_len, chunk_size, num_chunks, num_layers(),
                hidden_size(), intermediate_size(), num_q_heads(),
                num_kv_heads(), head_dim(), num_cores(),
                static_cast<int64_t>(layer_weights_.front().q_w.scalar_type()),
                linear_acc32_ ? 1 : 0};
    }


    bool qwen3vl_decode_gemv(int64_t physical_len) const {
        return (qwen3vl_4b_w8_profile_ || qwen3vl_fp16_decode_profile_) &&
            physical_len == 1 &&
            !linear_acc32_ && !adarms_ &&
            planning_mode_ == CausalDecoderPlanningMode::ORDINARY;
    }

    bool qwen3vl_decode_mlp(
        int64_t physical_len, int64_t rows, bool is_causal,
        bool use_attn_mask, int64_t batch_size) const {
        return qwen3vl_4b_w8_profile_ && qwen3vl_decode_gemv(physical_len) && rows == 1 &&
            is_causal && !use_attn_mask && batch_size == 1 &&
            !qwen3vl_pooler_z1_dispatch_ && !qwen3vl_multiview_text_composite_dispatch_;
    }

    bool qwen3vl_4b_decode_qk_norm_mrope(
        int64_t physical_len, int64_t rows, bool is_causal,
        bool use_attn_mask, int64_t batch_size, bool partial_mrope) const {
        return (qwen3vl_4b_decode_qk_fused_ || qwen3vl_4b_decode_qk_kv_fused_) &&
            physical_len == 1 && rows == 1 && is_causal &&
            !use_attn_mask && batch_size == 1 &&
            !adarms_ &&
            planning_mode_ == CausalDecoderPlanningMode::ORDINARY &&
            !qwen3vl_pooler_z1_dispatch_ && !qwen3vl_multiview_text_composite_dispatch_ &&
            has_qk_norm_ && has_mrope_ && !partial_mrope &&
            num_q_heads() == 32 && num_kv_heads() == 8 && head_dim() == 128 &&
            attn_tp() == 8 && eps_ == 1e-6;
    }

    bool qwen3vl_4b_decode_o_ring_norm(
        int64_t physical_len, int64_t rows, bool is_causal,
        bool use_attn_mask, int64_t batch_size, bool partial_mrope) const {
        return qwen3vl_4b_decode_o_ring_norm_fused_ &&
            qwen3vl_decode_mlp(physical_len, rows, is_causal,
                                use_attn_mask, batch_size) &&
            !partial_mrope && !wall_z1_dispatch_ &&
            hidden_size() == 2560 && num_q_heads() == 32 &&
            num_kv_heads() == 8 && head_dim() == 128 &&
            num_cores() == 8 && attn_tp() == 8 && eps_ == 1e-6;
    }

    static at::Tensor retain_qwen3vl_4b_o_scale(const at::Tensor& scale) {
        TORCH_CHECK(scale.scalar_type() == at::kHalf && scale.is_contiguous() &&
                        scale.dim() == 1 && scale.numel() == 2560,
                    "4B fused O/ring/norm requires FP16 per-channel scales [2560]");
        // Original N320 GEMV reads 512 scales for its last 320-channel block.
        // Keep logical [2560], with initialized backing for the 192-half tail.
        auto backing = at::empty({2560 + 256}, scale.options());
        auto* dst = backing.data_ptr<c10::Half>();
        std::memcpy(dst, scale.data_ptr(), scale.nbytes());
        std::fill_n(dst + 2560, 256, c10::Half(0.0f));
        rpu_ddr_flush_force_sized(dst, backing.nbytes());
        return backing.narrow(0, 0, 2560);
    }

    static at::Tensor retain_qwen3vl_4b_gateup_scale(const at::Tensor& scale) {
        TORCH_CHECK(scale.scalar_type() == at::kHalf && scale.is_contiguous() &&
                        scale.dim() == 1 && scale.numel() == 9728,
                    "4B fused Gate/Up requires FP16 per-channel scales [9728]");
        // The inherited GEMV scale loader reads 256 channels for its N96 tail.
        // Own initialized backing for that overfetch while keeping logical [N].
        // RPU tensors use HostDDR, as in the cold QKV/GateUp packing above.
        auto backing = at::empty({9728 + 256}, scale.options());
        auto* dst = backing.data_ptr<c10::Half>();
        std::memcpy(dst, scale.data_ptr(), scale.nbytes());
        std::fill_n(dst + 9728, 256, c10::Half(0.0f));
        rpu_ddr_flush_force_sized(dst, backing.nbytes());
        return backing.narrow(0, 0, 9728);
    }

    static bool qwen3vl_4b_qkv_storage_contiguous(
        const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
        if (!q.defined() || !k.defined() || !v.defined() ||
            q.scalar_type() != at::kChar || k.scalar_type() != at::kChar ||
            v.scalar_type() != at::kChar || !q.is_contiguous() ||
            !k.is_contiguous() || !v.is_contiguous() ||
            q.sizes() != at::IntArrayRef({4096, 2560}) ||
            k.sizes() != at::IntArrayRef({1024, 2560}) ||
            v.sizes() != at::IntArrayRef({1024, 2560})) return false;
        const auto* storage = q.storage().unsafeGetStorageImpl();
        return storage == k.storage().unsafeGetStorageImpl() &&
            storage == v.storage().unsafeGetStorageImpl() &&
            q.storage_offset() % 256 == 0 &&
            k.storage_offset() == q.storage_offset() + q.numel() &&
            v.storage_offset() == k.storage_offset() + k.numel() &&
            static_cast<uint64_t>(v.storage_offset() + v.numel()) <=
                q.storage().nbytes();
    }

    static void copy_qwen3vl_4b_qkv_scales(
        const at::Tensor& packed, const at::Tensor& q,
        const at::Tensor& k, const at::Tensor& v) {
        auto* dst = packed.data_ptr<c10::Half>();
        const auto* qs = q.data_ptr<c10::Half>();
        const auto* ks = k.data_ptr<c10::Half>();
        const auto* vs = v.data_ptr<c10::Half>();
        for (int core = 0; core < NUM_CORES; ++core) {
            std::memcpy(dst + core * 768, qs + core * 512, 1024);
            std::memcpy(dst + core * 768 + 512, ks + core * 128, 256);
            std::memcpy(dst + core * 768 + 640, vs + core * 128, 256);
        }
    }

    bool qwen3vl_4b_packed_decode_qkv(
        int64_t physical_len, bool is_causal, bool use_attn_mask,
        int64_t batch_size) const {
        return qwen3vl_4b_packed_qkv_weights_ &&
            qwen3vl_decode_gemv(physical_len) && is_causal &&
            !use_attn_mask && batch_size == 1;
    }

    CausalDecoderLinearRoute decoder_projection_route(int64_t physical_len) const {
        return qwen3vl_decode_gemv(physical_len)
            ? CausalDecoderLinearRoute::GEMV : CausalDecoderLinearRoute::AUTO_TILE;
    }

    bool qwen3vl_4b_prefill_swiglu(
        int64_t physical_len, int64_t rows, int64_t position,
        bool is_causal, bool use_attn_mask, int64_t batch_size) const {
        return qwen3vl_4b_w8_profile_ && physical_len > 1 &&
            rows >= 16 && rows <= 256 && rows % 16 == 0 &&
            position == 0 && is_causal && !use_attn_mask && batch_size == 1 &&
            !linear_acc32_ && !adarms_ &&
            planning_mode_ == CausalDecoderPlanningMode::ORDINARY &&
            !qwen3vl_pooler_z1_dispatch_ && !qwen3vl_multiview_text_composite_dispatch_;
    }

    FmbPhysicalExecutionManifest physical_manifest_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t physical_len, int64_t logical_len,
        int64_t position) const override {
        FmbPhysicalExecutionManifest manifest;
        manifest.state = FmbPhysicalManifestState::COMPLETE;
        manifest.logical_length = logical_len;
        manifest.physical_length = physical_len;
        manifest.execution_padding_rows = physical_len - logical_len;
        // The retained decode descriptor is canonical at position zero and is
        // reused for every live position.  Its KV horizon therefore covers the
        // installed RoPE/cache envelope, while prefill records the exact end.
        manifest.kv_logical_length = physical_len == 1 && !qwen3vl_2b_w8_profile_
            ? cos_.size(0) : position + logical_len;
        manifest.kv_insert_physical_rows = physical_len;
        manifest.graph_lifecycle = planning_graph_lifecycle_;
        // All ordinary Linear formats share the immutable accumulation choice.
        manifest.linear_accumulation = (linear_acc32_
                   ? FmbLinearAccumulationPolicy::ACC32
                   : FmbLinearAccumulationPolicy::ACC16);

        auto append = [&](FmbRouteFamily family, int64_t site_id,
                          int64_t selector, int64_t flags = 0,
                          std::vector<int64_t> arguments = {},
                          int64_t invocation = 0) {
            if (num_cores() != 8 &&
                (family == FmbRouteFamily::LINEAR ||
                 family == FmbRouteFamily::ALL_REDUCE ||
                 family == FmbRouteFamily::MUTABLE_DMA)) {
                arguments = bind_cold_topology_arguments(std::move(arguments));
            }
            manifest.routes.push_back({site_id, family, selector, flags,
                                       std::move(arguments), invocation});
        };
        auto append_attention = [&](int64_t site_id,
                                    AttentionExecutionPolicy policy,
                                    int64_t invocation) {
            append(FmbRouteFamily::ATTENTION, site_id,
                   static_cast<int64_t>(policy), 0, {physical_len}, invocation);
        };
        auto append_linear = [&](int64_t site_id, CausalDecoderLinearRoute route) {
            append(FmbRouteFamily::LINEAR, site_id,
                   static_cast<int64_t>(route));
        };
        auto append_ring = [&](int64_t site_id, int64_t rows,
                               int64_t invocation) {
            append(FmbRouteFamily::ALL_REDUCE, site_id,
                   causal_ring_route(rows, hidden_size()), 0, {}, invocation);
        };
        auto append_kv = [&](int64_t site_id, int64_t invocation,
                             int64_t insert_position, int64_t rows,
                             int64_t spm_rows, bool dynamic_position) {
            const int64_t padded_rows = Align(rows, static_cast<int64_t>(16));
            const int64_t physical_rows = !dynamic_position &&
                    spm_rows >= padded_rows && insert_position % 16 == 0
                ? padded_rows : rows;
            uint32_t capabilities = KV_INSERT_CAP_V2;
            if (!dynamic_position) capabilities |= KV_INSERT_CAP_V16;
            if (physical_rows != rows) capabilities |= KV_INSERT_CAP_PAD16;
            const KvInsertSegmentPlan kv_plan = dynamic_position
                ? rpu_resolve_kvinsert_segment_plan(
                      /*position=*/0, /*logical_rows=*/1,
                      /*physical_rows=*/1, attn_tp(), num_kv_heads(),
                      head_dim(), capabilities, KvInsertRoute::V2)
                : resolve_kvinsert_plan_auto(
                      site_id, manifest.graph_lifecycle,
                      insert_position, rows, physical_rows, attn_tp(),
                      num_kv_heads(), head_dim(), capabilities);
            const KvInsertRouteArguments arguments =
                rpu_kvinsert_route_arguments(
                    kv_plan, attn_tp(), num_kv_heads(), head_dim());
            append(FmbRouteFamily::KV_INSERT, site_id,
                   static_cast<int64_t>(kv_plan.route()),
                   CAUSAL_DECODER_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED |
                       (dynamic_position
                            ? KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION : 0),
                   {arguments.begin(), arguments.end()}, invocation);
        };

        const bool partial_mrope = planning_rope_mode_ == 1;
        const bool pooler_z1_mode = qwen3vl_pooler_z1_dispatch_ ||
            planning_mode_ ==
                CausalDecoderPlanningMode::QWEN3VL_POOLER_Z1;
        const bool multiview_mode =
            qwen3vl_multiview_text_composite_dispatch_ ||
            planning_mode_ ==
                CausalDecoderPlanningMode::QWEN3VL_MULTIVIEW_TEXT;
        const bool tensor_rope = pooler_z1_mode || multiview_mode;
        const int64_t rope_site = !has_mrope_
            ? CAUSAL_DECODER_ROPE_1D_SITE
            : partial_mrope
                ? (tensor_rope ? CAUSAL_DECODER_PARTIAL_MROPE_TENSOR_SITE
                               : CAUSAL_DECODER_PARTIAL_MROPE_POINTER_SITE)
                : (tensor_rope ? CAUSAL_DECODER_MROPE_TENSOR_SITE
                               : CAUSAL_DECODER_MROPE_POINTER_SITE);
        const CausalDecoderRopeRoute rope_route = !has_mrope_
            ? CausalDecoderRopeRoute::ROPE_1D
            : partial_mrope ? CausalDecoderRopeRoute::PARTIAL_MROPE
                            : CausalDecoderRopeRoute::MROPE;

        const int64_t rows = physical_len;

        const int64_t topology_selector = static_cast<int64_t>(
            CausalDecoderGraphScheduleRoute::TOPOLOGY);
        append(FmbRouteFamily::GRAPH_SCHEDULE,
               CAUSAL_DECODER_POOLER_TOPOLOGY_SITE, topology_selector, 0,
               {pooler_z1_mode ? 1 : 0, multiview_mode ? 1 : 0});
        append(FmbRouteFamily::GRAPH_SCHEDULE,
               CAUSAL_DECODER_ADARMS_TOPOLOGY_SITE, topology_selector, 0,
               {adarms_ ? 1 : 0, adarms_mutable_ ? 1 : 0,
                adarms_unroll_ ? 1 : 0, adarms_schedule_select_ ? 1 : 0,
                adarms_fused_bcast_enabled_ ? 1 : 0});
        append(FmbRouteFamily::GRAPH_SCHEDULE,
               CAUSAL_DECODER_QK_NORM_TOPOLOGY_SITE, topology_selector, 0,
               {has_qk_norm_ ? 1 : 0});
        append(FmbRouteFamily::GRAPH_SCHEDULE,
               CAUSAL_DECODER_QKV_BIAS_TOPOLOGY_SITE, topology_selector, 0,
               {has_qkv_bias_ ? 1 : 0});
        append(FmbRouteFamily::GRAPH_SCHEDULE,
               CAUSAL_DECODER_NVFP4_TOPOLOGY_SITE, topology_selector, 0,
               {nvfp4_ ? 1 : 0});
        append(FmbRouteFamily::GRAPH_SCHEDULE,
               CAUSAL_DECODER_DEEPSTACK_TOPOLOGY_SITE, topology_selector, 0,
               causal_deepstack_topology_arguments(deepstack_lang_layers_));
        append(FmbRouteFamily::GRAPH_SCHEDULE,
               CAUSAL_DECODER_LM_HEAD_TOPOLOGY_SITE, topology_selector, 0,
               {fuse_lm_head_ ? 1 : 0, vocab_size_});

        if (qwen3vl_4b_w8_resident_prefill(
                physical_len, position, layout.is_causal, layout.use_attn_mask,
                layout.batch_size, plan.compute.chunks.size())) {
            TORCH_CHECK(plan.chunk_mode == ChunkMode::SEQUENTIAL,
                        "Qwen3-VL 4B W8 resident prefill requires a causal stage plan");
            append(FmbRouteFamily::GRAPH_SCHEDULE,
                   CAUSAL_DECODER_4B_W8_PREFILL_SCHEDULE_SITE, 1, 0,
                   {physical_len, plan.compute.plan.chunk_size,
                    static_cast<int64_t>(plan.compute.chunks.size()), num_layers()});
        }
        if (qwen3vl_ordinary_resident_prefill(
                physical_len, position, layout.is_causal, layout.use_attn_mask,
                layout.batch_size, plan.compute.chunks.size())) {
            TORCH_CHECK(plan.chunk_mode == ChunkMode::SEQUENTIAL,
                        "Qwen3-VL ordinary resident prefill requires a causal stage plan");
            append(FmbRouteFamily::GRAPH_SCHEDULE,
                   CAUSAL_DECODER_VL_PREFILL_SCHEDULE_SITE, 1, 0,
                   qwen3vl_ordinary_prefill_schedule_arguments(
                       physical_len, plan.compute.plan.chunk_size,
                       static_cast<int64_t>(plan.compute.chunks.size())));
        }

        if (qwen3vl_2b_w8_profile_) {
            TORCH_CHECK(layout.is_causal && !layout.use_attn_mask &&
                layout.batch_size == 1 && (physical_len == 576 || physical_len == 1),
                "fixed delivery physical domain only admits P576/decode");
            append(FmbRouteFamily::GRAPH_SCHEDULE, QWEN3VL_DELIVERY_TOPOLOGY_SITE,
                   1, lm_head_exact_candidates_, delivery_topology_arguments());
            for (const auto& chunk : plan.compute.chunks) {
                TORCH_CHECK(chunk.len == (physical_len == 1 ? 1 : 192),
                    "fixed delivery planner requires C192x3 or one decode row");
                append(FmbRouteFamily::LINEAR, QWEN3VL_DELIVERY_QKV_SITE,
                       chunk.len == 192 ? 5 : 1, 0,
                       {chunk.len, 4096, 2048, 1, 8}, chunk.idx);
                append(FmbRouteFamily::ROPE, QWEN3VL_DELIVERY_ROPE_SITE, 4, 0,
                       {chunk.len, 2, 1, 128, 8}, chunk.idx);
                append(FmbRouteFamily::GRAPH_SCHEDULE,
                       chunk.len == 1 ? QWEN3VL_DELIVERY_KV_DECODE_PAIR_SITE
                                      : QWEN3VL_DELIVERY_KV_PAIR_SITE,
                       chunk.len == 1 ? 2 : 1, 0,
                       delivery_kv_arguments(chunk.len, position + chunk.offset), chunk.idx);
                append(FmbRouteFamily::LINEAR, QWEN3VL_DELIVERY_MLP_LINEAR_SITE,
                       1, 0, {chunk.len, 12288, 2048, 1, 8}, chunk.idx);
                append(FmbRouteFamily::LINEAR, QWEN3VL_DELIVERY_MLP_DOWN_SITE,
                       1, 0, {chunk.len, 2048, 6144, 0, 8}, chunk.idx);
                append(FmbRouteFamily::ACTIVATION,
                       chunk.len == 1 ? QWEN3VL_DELIVERY_MLP_DECODE_ACTIVATION_SITE
                                      : QWEN3VL_DELIVERY_MLP_ACTIVATION_SITE,
                       chunk.len == 1 ? 1 : 2, 0, {chunk.len, 768, 8}, chunk.idx);
                append(FmbRouteFamily::ALL_REDUCE, QWEN3VL_DELIVERY_MLP_REDUCE_SITE,
                       causal_ring_route(chunk.len, 2048), 0, {}, chunk.idx);
                for (int64_t role = 0; role < 3; ++role) {
                    const int64_t norm_rows = chunk.len * (role == 1 ? 2 : 1);
                    append(FmbRouteFamily::NORMALIZATION, QWEN3VL_DELIVERY_RMS_SITE,
                           static_cast<int64_t>(norm_rows % 64 == 0
                               ? RpuRmsNormSpmRoute::QWEN3VL_V64 : RpuRmsNormSpmRoute::BASE),
                           0, {norm_rows, role == 0 ? 2048 : 128}, 3 * chunk.idx + role);
                }
            }
            if (fuse_lm_head_ && lm_head_exact_candidates_ > 0) {
                append(FmbRouteFamily::MUTABLE_DMA, QWEN3VL_DELIVERY_LM_HIDDEN_DMA_SITE,
                       4, 0, {hidden_size(), vocab_size_});
            }
        }

        const bool qwen3vl4b_fused_o =
            plan.chunk_mode == ChunkMode::SEQUENTIAL &&
            plan.compute.chunks.size() == 1 &&
            layout.attention_policy != AttentionExecutionPolicy::SPM_KV_BY_MHA &&
            qwen3vl_4b_decode_o_ring_norm(physical_len,
                plan.compute.chunks.front().len, layout.is_causal,
                layout.use_attn_mask, layout.batch_size, partial_mrope);
        const bool qwen3vl4b_fused_qk_kv =
            qwen3vl_4b_decode_qk_kv_fused_ &&
            layout.attention_policy != AttentionExecutionPolicy::SPM_KV_BY_MHA &&
            plan.chunk_mode == ChunkMode::SEQUENTIAL &&
            plan.compute.chunks.size() == 1 &&
            qwen3vl_4b_decode_qk_norm_mrope(physical_len,
                plan.compute.chunks.front().len, layout.is_causal,
                layout.use_attn_mask, layout.batch_size, partial_mrope);
        const bool qwen3vl4b_fused_qk =
            (qwen3vl_4b_decode_qk_fused_ || qwen3vl4b_fused_qk_kv) &&
            plan.chunk_mode == ChunkMode::SEQUENTIAL &&
            plan.compute.chunks.size() == 1 &&
            qwen3vl_4b_decode_qk_norm_mrope(physical_len,
                plan.compute.chunks.front().len, layout.is_causal,
                layout.use_attn_mask, layout.batch_size, partial_mrope);
        if (qwen3vl4b_fused_qk_kv) {
            append(FmbRouteFamily::NORMALIZATION,
                   QWEN3VL_4B_DECODE_QK_NORM_MROPE_KV_INSERT_SITE, 1, 0,
                   {1, 4, 1, 128, 8, 24, 20, 20},
                   plan.compute.chunks.front().idx);
        } else if (qwen3vl4b_fused_qk) {
            append(FmbRouteFamily::NORMALIZATION,
                   QWEN3VL_4B_DECODE_QK_NORM_MROPE_SITE, 1, 0,
                   {1, 4, 1, 128, 8, 24, 20, 20},
                   plan.compute.chunks.front().idx);
        }

        if (ordinary_rmsnorm_route_enabled() && layout.is_causal &&
            layout.batch_size == 1 &&
            plan.chunk_mode == ChunkMode::SEQUENTIAL) {
            for (const ChunkInfo& chunk : plan.compute.chunks) {
                const auto append_ordinary_rmsnorm =
                    [&](int64_t role) {
                        const int64_t norm_rows = chunk.len *
                            (role == 1 ? num_q_heads() / attn_tp() :
                             role == 2 ? num_kv_heads() / attn_tp() : 1);
                        const int64_t cols =
                            role == 0 ? hidden_size() : head_dim();
                        if (rhinovla_high_precision_) {
                            append(FmbRouteFamily::NORMALIZATION,
                                   RHINOVLA_TEXT_HIGH_RMSNORM_SITE,
                                   static_cast<int64_t>(rpu_resolve_high_precision_rmsnorm_spm_route(norm_rows, cols)),
                                   0, {norm_rows, cols, 1}, 3 * chunk.idx + role);
                            return;
                        }
                        append(
                            FmbRouteFamily::NORMALIZATION,
                            CAUSAL_DECODER_ORDINARY_RMSNORM_SITE,
                            static_cast<int64_t>(
                                ordinary_rmsnorm_spm_contract_.resolve(
                                    norm_rows, cols)),
                            0,
                            ordinary_rmsnorm_spm_contract_.manifest_arguments(
                                norm_rows, cols),
                            3 * chunk.idx + role);
                    };
                if (rhinovla_high_precision_) {
                    append(FmbRouteFamily::GRAPH_SCHEDULE, RHINOVLA_TEXT_HIGH_ACTIVATION_SITE,
                           1, rhinovla_high_precision_fusions_ ? 1 : 0,
                           {chunk.len, intermediate_size() / mlp_tp(), 1}, chunk.idx);
                }
                append_ordinary_rmsnorm(/*role=*/0);
                if (has_qk_norm_ && !qwen3vl4b_fused_qk) {
                    append_ordinary_rmsnorm(/*role=*/1);
                    append_ordinary_rmsnorm(/*role=*/2);
                }
            }
        }

        append_causal_decoder_preload_manifest_routes(manifest);
        if (explicit_mask_residency_requested_ && layout.use_attn_mask) {
            TORCH_CHECK(!layout.is_causal && plan.compute.chunks.size() == 1,
                        "resident explicit mask requires one noncausal chunk");
            // Complete declarations are recomputed for this native candidate.
            // Never publish a dry-probe decision into mutable model state.
            auto declarations = const_cast<CausalDecoderModel*>(this)
                ->explicit_mask_baseline_buffer_declarations(layout);
            const auto mask_plan = detail::plan_forward_spm_residency(
                declarations, {"sdpa_mask"});
            append(FmbRouteFamily::GRAPH_SCHEDULE,
                   CAUSAL_DECODER_MASK_SPM_SCHEDULE_SITE,
                   static_cast<int64_t>(mask_plan.retained
                       ? CausalMaskSpmSchedule::FIRST_BODY_FIRST_LAYER
                       : CausalMaskSpmSchedule::EVERY_LAYER), 0,
                   {physical_len, position + physical_len, attn_tp(), num_cores(),
                    explicit_mask_body_iterations(), num_layers(),
                    mask_plan.original_peak_bytes, mask_plan.retained_peak_bytes});
        }

        if (!qwen3vl_2b_w8_profile_ && !qwen3vl4b_fused_qk) {
            append(FmbRouteFamily::ROPE, rope_site,
                   static_cast<int64_t>(rope_route), 0,
                   {has_mrope_ ? 1 : 0,
                    (has_mrope_ && partial_mrope) ? 1 : 0,
                    tensor_rope ? 1 : 0});
        }
        if (adarms_mutable_ && adarms_fused_bcast_enabled_ &&
            adarms_schedule_select_) {
            append(FmbRouteFamily::MUTABLE_DMA,
                   CAUSAL_DECODER_ADARMS_MUTABLE_SITE,
                   static_cast<int64_t>(
                       CausalDecoderMutableDmaRoute::DDR_BROADCAST_TO_SPM));
        }

        if (plan.chunk_mode == ChunkMode::KV_FIRST) {
            append_linear(CAUSAL_DECODER_KV_FIRST_Q_SITE,
                          CausalDecoderLinearRoute::AUTO_TILE);
            append_linear(CAUSAL_DECODER_KV_FIRST_K_SITE,
                          CausalDecoderLinearRoute::AUTO_TILE);
            append_linear(CAUSAL_DECODER_KV_FIRST_V_SITE,
                          CausalDecoderLinearRoute::AUTO_TILE);
            for (const ChunkInfo& chunk : plan.qkv.chunks) {
                append_kv(CAUSAL_DECODER_KV_FIRST_KV_SITE, chunk.idx,
                          position + chunk.offset, chunk.len, 0, false);
            }
            for (const ChunkInfo& chunk : plan.compute.chunks) {
                append_attention(CAUSAL_DECODER_KV_FIRST_ATTN_SITE,
                                 AttentionExecutionPolicy::DDR_KV, chunk.idx);
                append_ring(CAUSAL_DECODER_KV_FIRST_REDUCE_SITE,
                            chunk.len, chunk.idx);
            }
            append(FmbRouteFamily::ALL_REDUCE,
                   CAUSAL_DECODER_KV_FIRST_PREPARE_SITE,
                   static_cast<int64_t>(
                       CausalDecoderAllReduceRoute::PREPARE_RING_INPUT));
            append_linear(CAUSAL_DECODER_KV_FIRST_O_SITE,
                          CausalDecoderLinearRoute::AUTO_TILE);
            if (!deepstack_lang_layers_.empty() &&
                !multiview_mode && !pooler_z1_mode) {
                append(FmbRouteFamily::MUTABLE_DMA,
                       CAUSAL_DECODER_KV_FIRST_DEEPSTACK_SITE,
                       static_cast<int64_t>(
                           CausalDecoderMutableDmaRoute::DDR_BROADCAST_TO_SPM));
            }
            if (fuse_lm_head_ && (physical_len == 1 || fuse_prefill_last_lm_head_)) {
                append_linear(CAUSAL_DECODER_KV_FIRST_LM_HEAD_SITE,
                              CausalDecoderLinearRoute::AUTO_TILE);
                append(FmbRouteFamily::MUTABLE_DMA,
                       CAUSAL_DECODER_KV_FIRST_LM_HEAD_DMA_SITE,
                       static_cast<int64_t>(
                           CausalDecoderMutableDmaRoute::SPM_SCATTER_TO_DDR));
            }
        } else {
            if (qwen3vl_4b_packed_decode_qkv(
                    physical_len, layout.is_causal, layout.use_attn_mask,
                    layout.batch_size)) {
                append(FmbRouteFamily::LINEAR, CAUSAL_DECODER_4B_W8_PACKED_QKV_SITE,
                       static_cast<int64_t>(CausalDecoderLinearRoute::GEMV), 0,
                       {1, 6144, 2560, 8, 512, 128, 128});
            } else {
                append_linear(CAUSAL_DECODER_Q_SITE,
                              decoder_projection_route(physical_len));
                append_linear(CAUSAL_DECODER_K_SITE,
                              decoder_projection_route(physical_len));
                append_linear(CAUSAL_DECODER_V_SITE,
                              decoder_projection_route(physical_len));
            }
            LayoutContext spm_layout = layout;
            spm_layout.attention_policy =
                AttentionExecutionPolicy::SPM_KV_BY_MHA;
            const bool raw_spm = subclass_spm_kv_by_mha_eligible(
                plan, spm_layout, position);
            for (const ChunkInfo& chunk : plan.compute.chunks) {
                if (!qwen3vl_2b_w8_profile_ && !qwen3vl4b_fused_qk_kv) {
                    append_kv(CAUSAL_DECODER_KV_INSERT_SITE, chunk.idx,
                              physical_len == 1 ? 0 : position + chunk.offset,
                              chunk.len, kv_insert_spm_rows_, physical_len == 1);
                }
                if (raw_spm) {
                    append_attention(CAUSAL_DECODER_RAW_SPM_ATTN_SITE,
                                     AttentionExecutionPolicy::SPM_KV_BY_MHA,
                                     chunk.idx);
                } else {
                    append_attention(CAUSAL_DECODER_MASKED_ATTENTION_SITE,
                                     AttentionExecutionPolicy::DDR_KV,
                                     chunk.idx);
                }
                if (!qwen3vl4b_fused_o) {
                    {
                        append_ring(CAUSAL_DECODER_REDUCE_SITE,
                                    chunk.len, chunk.idx);
                    }
                }
            }
            if (qwen3vl4b_fused_o) {
                append(FmbRouteFamily::LINEAR, QWEN3VL_4B_DECODE_O_RING_NORM_SITE,
                       1, 0, {1, 2560, 4096, 512, 8, 320, 1, 63488},
                       plan.compute.chunks.front().idx);
            } else {
                append(FmbRouteFamily::ALL_REDUCE,
                       CAUSAL_DECODER_PREPARE_SITE,
                       static_cast<int64_t>(
                           CausalDecoderAllReduceRoute::PREPARE_RING_INPUT));
                if (ordinary_row_weight_reuse_plan_enabled(
                        plan, layout.is_causal, layout.batch_size)) {
                    for (const auto& chunk : plan.compute.chunks) {
                        const bool reuse = rpu_pl_tiling::supports_w8a16_row_weight_reuse(
                            chunk.len, hidden_size(), num_q_heads() * head_dim(),
                            0, attn_tp());
                        append(FmbRouteFamily::LINEAR, CAUSAL_DECODER_O_SITE,
                               static_cast<int64_t>(reuse
                                   ? CausalDecoderLinearRoute::ROW_WEIGHT_REUSE
                                   : CausalDecoderLinearRoute::AUTO_TILE),
                               0, ordinary_o_route_arguments(chunk.len, reuse), chunk.idx);
                    }
                } else {
                    append_linear(CAUSAL_DECODER_O_SITE,
                                  decoder_projection_route(physical_len));
                }
            }
            if (!deepstack_lang_layers_.empty() &&
                !multiview_mode && !pooler_z1_mode) {
                append(FmbRouteFamily::MUTABLE_DMA,
                       CAUSAL_DECODER_DEEPSTACK_SITE,
                       static_cast<int64_t>(
                           CausalDecoderMutableDmaRoute::DDR_BROADCAST_TO_SPM));
            }
            if (fuse_lm_head_ && (physical_len == 1 || fuse_prefill_last_lm_head_)) {
                append_linear(CAUSAL_DECODER_LM_HEAD_SITE,
                              CausalDecoderLinearRoute::AUTO_TILE);
                append(FmbRouteFamily::MUTABLE_DMA,
                       CAUSAL_DECODER_LM_HEAD_DMA_SITE,
                       static_cast<int64_t>(
                           CausalDecoderMutableDmaRoute::SPM_SCATTER_TO_DDR));
            }
        }

        bool needs_shared_mlp = false;
        for (const ChunkInfo& chunk : plan.compute.chunks) {
            if (plan.chunk_mode == ChunkMode::SEQUENTIAL &&
                qwen3vl_4b_prefill_swiglu(physical_len, chunk.len, position,
                    layout.is_causal, layout.use_attn_mask, layout.batch_size)) {
                append(FmbRouteFamily::LINEAR, QWEN3VL_4B_PREFILL_GATEUP_SITE,
                       1, 0, {chunk.len, 9728, 2560, 8, 320, 112}, chunk.idx);
                append(FmbRouteFamily::LINEAR, QWEN3VL_4B_PREFILL_DOWN_SITE,
                       static_cast<int64_t>(CausalDecoderLinearRoute::AUTO_TILE),
                       0, {chunk.len, 2560, 9728, 0, 8}, chunk.idx);
                append_ring(QWEN3VL_4B_PREFILL_MLP_REDUCE_SITE, chunk.len, chunk.idx);
            } else if (plan.chunk_mode == ChunkMode::SEQUENTIAL &&
                qwen3vl_decode_mlp(physical_len, chunk.len,
                    layout.is_causal, layout.use_attn_mask, layout.batch_size)) {
                if (qwen3vl_4b_decode_gateup_fused_) {
                    append(FmbRouteFamily::LINEAR, QWEN3VL_4B_DECODE_MLP_GATEUP_FUSED_SITE,
                           1, 0, {1, 9728, 2560, 8, 160, 96}, chunk.idx);
                } else {
                    append(FmbRouteFamily::LINEAR, QWEN3VL_4B_DECODE_MLP_GATE_SITE,
                           static_cast<int64_t>(CausalDecoderLinearRoute::GEMV),
                           0, {1, intermediate_size(), hidden_size(), 1, 8}, chunk.idx);
                    append(FmbRouteFamily::LINEAR, QWEN3VL_4B_DECODE_MLP_UP_SITE,
                           static_cast<int64_t>(CausalDecoderLinearRoute::GEMV),
                           0, {1, intermediate_size(), hidden_size(), 1, 8}, chunk.idx);
                    append(FmbRouteFamily::ACTIVATION, QWEN3VL_4B_DECODE_MLP_ACTIVATION_SITE,
                           1, 0, {1, intermediate_size() / 8, 8}, chunk.idx);
                }
                append(FmbRouteFamily::LINEAR, QWEN3VL_4B_DECODE_MLP_DOWN_SITE,
                       static_cast<int64_t>(CausalDecoderLinearRoute::GEMV),
                       0, {1, hidden_size(), intermediate_size(), 0, 8}, chunk.idx);
                append_ring(QWEN3VL_4B_DECODE_MLP_REDUCE_SITE, 1, chunk.idx);
            } else {
                needs_shared_mlp = true;
            }
        }
        uint32_t shared_routes = 0;
        if (multiview_mode) {
            shared_routes |= FMB_SHARED_LAYER_INPUT_ROW_RUN_DMA;
        } else if (!pooler_z1_mode &&
                   planning_mode_ != CausalDecoderPlanningMode::WALL_Z1 &&
                   !wall_z1_dispatch_ && !pre_layers_residual1_ready_) {
            shared_routes |= FMB_SHARED_LAYER_INPUT_DMA;
        }
        if (needs_shared_mlp && !decode_gate_up_fusion_) {
            shared_routes |= (qwen3vl_decode_gemv(physical_len)
                ? FMB_SHARED_MLP_GEMV : FMB_SHARED_MLP_AUTO_TILE) |
                FMB_SHARED_MLP_RING_REDUCE;
            if (mlp_tp() != num_cores()) shared_routes |= FMB_SHARED_MLP_PREPARE_INPUT;

        }
        append_fmb_shared_runtime_routes(
            manifest, plan, hidden_size(), shared_routes,
            layout.batch_size, num_cores(), mlp_tp());

        if (ordinary_silu_mul_route_enabled() && layout.is_causal &&
            layout.batch_size == 1 && plan.chunk_mode == ChunkMode::SEQUENTIAL) {
            for (const auto& chunk : plan.compute.chunks) {
                append(FmbRouteFamily::ACTIVATION, FMB_SHARED_MLP_SILU_MUL_SITE,
                       1, 0, {chunk.len, intermediate_size() / mlp_tp(), mlp_tp()},
                       chunk.idx);
            }
        }

        std::sort(manifest.routes.begin(), manifest.routes.end(),
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
        const FmbPhysicalExecutionManifest& manifest) const override {
        // Route receipts below independently prove the active physical branch;
        // the lifecycle is supplied by the same typed planner call that minted
        // the descriptor and is limited to the four FMB lifecycle values there.
        return {true, manifest.graph_lifecycle};
    }

    // ═══════════════════════════════════════════════════════════════════════
    // static/dynamic config
    // ═══════════════════════════════════════════════════════════════════════

    SpmFmbTraversalCapability spm_fmb_traversal_capability() const override {
        return qwen3vl_multiview_text_composite_prepared_
            ? SpmFmbTraversalCapability::CanonicalTraversal
            : SpmFmbTraversalCapability::Unsealed;
    }

    SpmFmbPreloadCapability spm_fmb_preload_capability() const override {
        return qwen3vl_multiview_text_composite_prepared_
            ? SpmFmbPreloadCapability::PersistentOutsideResolvedWindow
            : SpmFmbPreloadCapability::Unsealed;
    }

    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        // Python GraphSignature/GraphCache owns admission; C++ run_all_layers
        // only emits the op stream. Qwen3 lm_head 融合仍在 decode
        // (seq_len=1) 时由 build_layer_subgraph 末尾追加 lm_head GEMM kernel。
        cfg.num_layers = num_layers();
        // Force single group: cross_layer_batch_size = num_layers().
        //
        // Rationale (matches Gemma/AdaRMS/SigLIP/QwenPi05 — see
        // rpu_siglip_model.cpp:358 for the authoritative note): the global
        // runtime knob `g_cross_layer_batch_size` (default 12, set via
        // `torch.rpu.set_cross_layer_batch_size`) leaks across subclasses
        // and is not what Phase 1 validated. Every v3 subclass pins a
        // single group. If per-Qwen3 multi-group is needed later, wire an
        // explicit per-subclass knob — do not re-enable the global one.
        //
        // Removed dead read of get_cross_layer_batch_size() — previous code
        // read the knob, assigned cfg.cross_layer_batch_size, and
        // immediately overwrote it with num_layers() (codex-review
        // merge-readiness finding). AdaRMS has the same dead-write pattern
        // at rpu_adarms_model.cpp:272-274 — follow-up in a separate commit.
        cfg.cross_layer_batch_size = num_layers();
        cfg.kv_first_fn = static_cast<void(FusedModelBase::*)(int, const ChunkInfo&)>(
                              &CausalDecoderModel::emit_kv_first_body);
        cfg.kv_first_chunk_plan_fn =
            static_cast<ChunkPlan(FusedModelBase::*)(const ChunkPlan&)>(
                &CausalDecoderModel::plan_kv_first_chunks);
        // RhinoVLA fast-replay opt-in (set via causal_decoder_set_fast_replay).
        // Checked Qwen3-VL 4B W8 decode owns its body cursor and patch set;
        // prefill fast-replay on the same handle must not publish the generic
        // FMB body-skip policy for decode.
        const bool checked_decode_replay = qwen3vl_decode_replay_scope_active_;
        const bool owner_safe_prefill_fast_replay =
            fast_replay_prefill_candidate_ && fast_replay_owner_bound_;
        cfg.fast_replay_skip_layer_loop =
            !qwen3vl_multiview_text_composite_prepared_ &&
            !checked_decode_replay && fast_replay_skip_layer_loop_ &&
            (qwen3vl_2b_w8_profile_ || owner_safe_prefill_fast_replay);
        // preload-skip is only safe alongside skip_layer_loop (the cursor is set
        // absolutely past the preload region); gate them together so a lone
        // preload-skip can't desync the REPLAY cursor.
        cfg.fast_replay_skip_preload =
            !qwen3vl_multiview_text_composite_prepared_ &&
            !checked_decode_replay && preload_replay_skip_ &&
            cfg.fast_replay_skip_layer_loop;
        if (qwen3vl_decode_replay_scope_active_) {
            cfg.checked_layer_body_replay_fn =
                static_cast<bool(FusedModelBase::*)()>(
                    &CausalDecoderModel::try_checked_layer_body_replay);
        }
        cfg.batch_decode_active = batch_decode_active_;
        return cfg;
    }

    ModelDynamicConfig planning_dynamic_config(
        const ChunkPlan& plan) override {
        const bool saved = planning_domain_query_;
        auto restore = c10::make_scope_exit(
            [&] { planning_domain_query_ = saved; });
        planning_domain_query_ = true;
        return dynamic_config(plan);
    }

    ModelDynamicConfig dynamic_config(const ChunkPlan& plan) override {
        ModelDynamicConfig cfg;
        cfg.attention_policy = qwen3_spm_kv_by_mha_enabled_
            ? AttentionExecutionPolicy::AUTO
            : AttentionExecutionPolicy::DDR_KV;
        // Per-forward runtime hook (NOT cached): selects the execution chunk mode
        // from the SAME (is_causal, attention_mask) that run_all_layers threads
        // into layout_ctx. this is safe because the resulting
        // KV_FIRST layout is now part of the allocation identity — declare_buffers
        // derives kv_first_layout from the hashed LayoutContext and the hash folds
        // in the derived bit + kv_insert_chunk_size (compute_params_hash_impl).
        if (!ctx().is_causal && !ctx().attention_mask.has_value()) {
            cfg.chunk_mode = ChunkMode::KV_FIRST;
        } else {
            cfg.chunk_mode = ChunkMode::SEQUENTIAL;
        }
        cfg.inter_layer_io = InterLayerIO::AUTO;     // chunks==1 → SPM_RESIDENT, 否则 → DDR_PINGPONG

        // Every earlier chunk has populated each layer's causal KV before the
        // next chunk visits that layer. Keep its hidden in residual1 across
        // all layers; first/last-layer DMA and DeepStack retain chunk offsets.
        if (qwen3vl_4b_w8_resident_prefill(
                ctx().seq_len, ctx().position, ctx().is_causal,
                ctx().attention_mask.has_value(), ctx().batch_size,
                plan.num_chunks)) {
            cfg.inter_layer_io = InterLayerIO::SPM_RESIDENT;
            cfg.chunk_outer_within_group = true;
        }
        if (qwen3vl_ordinary_resident_prefill(
                ctx().seq_len, ctx().position, ctx().is_causal,
                ctx().attention_mask.has_value(), ctx().batch_size,
                plan.num_chunks)) {
            // residual1 already spans all compute phases. Removing the
            // inter-layer DDR copies adds no buffers and does not replan chunks.
            cfg.inter_layer_io = InterLayerIO::SPM_RESIDENT;
            cfg.chunk_outer_within_group = true;
        }

        // Controlled Qwen3-VL P576/C192 resident schedule; every
        // shape/topology guard is repeated here so
        // plain Qwen, larger Qwen3-VL profiles, and arbitrary prompts keep the
        // existing layer-major DDR schedule.
        if (qwen3vl_2b_w8_profile_ &&
            ctx().is_causal && !ctx().attention_mask.has_value() &&
            ctx().position == 0 && ctx().seq_len == 576 &&
            hidden_size() == 2048 && intermediate_size() == 6144 &&
            num_layers() == 28 && num_q_heads() == 16 &&
            num_kv_heads() == 8 && head_dim() == 128 &&
            qwen3vl_2b_w8_profile_) {
            TORCH_CHECK(
                plan.chunk_size == 192 && plan.num_chunks == 3,
                "Qwen3-VL 2B P576 resident schedule requires C192x3; got "
                "chunk_size=", plan.chunk_size,
                ", num_chunks=", plan.num_chunks);
            cfg.chunk_mode = ChunkMode::SEQUENTIAL;
            cfg.inter_layer_io = InterLayerIO::SPM_RESIDENT;
            cfg.chunk_outer_within_group = true;
        }

        // Wall-OSS permutation Z1 is an exact-profile canary whose text input
        // already lives in the composite physical arena.  Keep each C96 chunk
        // in residual1 while it traverses all 36 layers; the ordinary Qwen/Wall
        // paths retain AUTO's multi-chunk DDR ping-pong schedule.
        if (wall_z1_dispatch_) {
            TORCH_CHECK(ctx().is_causal && !ctx().attention_mask.has_value() &&
                            ctx().position == 0 &&
                            ctx().seq_len == WALL_Z1_CANARY_EXECUTION_LEN,
                        "Wall-OSS text Z1 resident schedule requires exact "
                        "causal prefill at position 0 with seq_len=192");
            TORCH_CHECK(plan.chunk_size == WALL_Z1_CANARY_CHUNK_SIZE &&
                            plan.num_chunks == 2 &&
                            wall_z1_prepared_chunk_size_ ==
                                WALL_Z1_CANARY_CHUNK_SIZE,
                        "Wall-OSS text Z1 resident schedule requires exactly "
                        "two C96 chunks; got chunk_size=", plan.chunk_size,
                        ", num_chunks=", plan.num_chunks,
                        ", prepared_chunk_size=",
                        wall_z1_prepared_chunk_size_);
            cfg.chunk_mode = ChunkMode::SEQUENTIAL;
            cfg.inter_layer_io = InterLayerIO::SPM_RESIDENT;
            cfg.chunk_outer_within_group = true;
        }

        if (qwen3vl_pooler_z1_dispatch_) {
            TORCH_CHECK(ctx().is_causal && !ctx().attention_mask.has_value() &&
                            ctx().position == 0 &&
                            ctx().seq_len ==
                                QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN,
                        "Qwen3-VL pooler Z1 resident schedule requires exact "
                        "causal prefill at position 0 with seq_len=128");
            TORCH_CHECK(
                plan.chunk_size == QWEN3VL_POOLER_Z1_CANARY_CHUNK_SIZE &&
                    plan.num_chunks == 1 &&
                    qwen3vl_pooler_z1_prepared_chunk_size_ ==
                        QWEN3VL_POOLER_Z1_CANARY_CHUNK_SIZE,
                "Qwen3-VL pooler Z1 resident schedule requires one C128 "
                "chunk; got chunk_size=", plan.chunk_size,
                ", num_chunks=", plan.num_chunks,
                ", prepared_chunk_size=",
                qwen3vl_pooler_z1_prepared_chunk_size_);
            cfg.chunk_mode = ChunkMode::SEQUENTIAL;
            cfg.inter_layer_io = InterLayerIO::SPM_RESIDENT;
            cfg.chunk_outer_within_group = true;
        }

        if (qwen3vl_multiview_text_composite_dispatch_) {
            TORCH_CHECK(
                ctx().is_causal && !ctx().attention_mask.has_value() &&
                    ctx().position == 0 &&
                    ctx().seq_len ==
                        QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN,
                "Qwen3-VL multiview Text resident schedule requires exact "
                "causal prefill at position 0 with seq_len=225");
            TORCH_CHECK(
                plan.chunk_size ==
                        QWEN3VL_MULTIVIEW_TEXT_ALLOCATION_CHUNK_SIZE &&
                    plan.num_chunks == 1,
                "Qwen3-VL multiview Text resident schedule requires native "
                "C240 resolving to one allocation C225 chunk; got chunk_size=",
                plan.chunk_size, ", num_chunks=", plan.num_chunks);
            cfg.chunk_mode = ChunkMode::SEQUENTIAL;
            cfg.inter_layer_io = InterLayerIO::SPM_RESIDENT;
            cfg.chunk_outer_within_group = false;
        }

        // ── P4 (Wall-OSS): prepare the explicit 2D mask once per forward ──
        // Mirrors Gemma's build_chunk_masks() hook (rpu_gemma_model.cpp:406): the
        // base has set ctx_ (attention_mask/position/seq_len) and computed `plan`
        // before this call. The action-denoise rectangular [H, P+H] mask only
        // works as one upload per layer when there is a single chunk, so assert
        // it (action horizon H <= chunk_size; A3 chunk-validator is mask-aware so
        // cs>=H is reachable). sdpa_prepare_mask copies the (CPU) mask to a stable
        // RPU DDR slot and returns mask_type=MASK_2D(4).
        if (ctx().attention_mask.has_value()) {
            TORCH_CHECK(plan.num_chunks == 1,
                        "CausalDecoderModel: explicit 2D mask path requires a "
                        "single chunk (got num_chunks=", plan.num_chunks,
                        "). Action horizon must be <= resolved chunk_size.");
            if (planning_domain_query_) return cfg;
            const int64_t seq_q = ctx().seq_len;
            const int64_t seq_k = ctx().position + ctx().seq_len;   // P + H
            // Skip the re-prepare (CPU→RPU upload + stable-slot copy) when the
            // caller passes the SAME (host-memoized) mask at the same dims — the
            // Wall-OSS action mask is _action_mask_cached (stable ptr across frames
            // for a fixed prefix), and this instance is the slot's only same-shape
            // user between forwards. Mirrors the vision #23 last_window_mask_ptr_
            // cache. A different ptr / seq → re-prepare (correct for a length change).
            const at::Tensor& input_mask = *ctx().attention_mask;
            const auto& version_counter =
                input_mask.unsafeGetTensorImpl()->version_counter();
            const bool version_enabled = version_counter.enabled();
            const int64_t input_version = version_enabled
                ? static_cast<int64_t>(version_counter.current_version()) : -1;
            const bool cache_hit =
                version_enabled && last_attn_mask_ref_.defined() &&
                last_attn_mask_ref_.is_same(input_mask) &&
                input_version == last_attn_mask_version_ &&
                seq_q == last_attn_seq_q_ && seq_k == last_attn_seq_k_;
            if (!cache_hit) {
                // sdpa_prepare_mask() publishes into a shape-keyed stable slot.
                // Invalidate the old source stamp before that slot can be
                // overwritten: if preparation or the mutation check throws, a
                // retry must upload again instead of falsely reusing the old
                // source identity with the newly modified slot contents.
                last_attn_mask_ref_ = at::Tensor{};
                last_attn_mask_version_ = -1;
                last_attn_seq_q_ = -1;
                last_attn_seq_k_ = -1;
                PreparedMask next_prepared_attn_mask = sdpa_prepare_mask(
                    ctx().attention_mask, /*is_causal=*/false, seq_q, seq_k,
                    sdpa_stable_mask_cache());
                TORCH_CHECK(
                    !version_enabled ||
                        version_counter.current_version() == input_version,
                    "CausalDecoderModel: attention mask mutated while its "
                    "stable Graph input was being prepared");
                // Keep the source alive so allocator reuse cannot turn an
                // unrelated tensor into a false pointer hit.  Versioned
                // identity also refreshes an in-place mutated mask.  Tensors
                // without version tracking conservatively miss every time.
                prepared_attn_mask_ = std::move(next_prepared_attn_mask);
                last_attn_mask_ref_ = input_mask;
                last_attn_mask_version_ = input_version;
                last_attn_seq_q_ = seq_q;
                last_attn_seq_k_ = seq_k;
            }
        }
        return cfg;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // subclass_chunk_size_valid — gate auto chunk_size search on SDPA tile
    // constraints.
    //
    // Bug history: without this hook the framework's auto-pick (linear scan
    // in fused_model_base.cpp:compute_chunks_impl) only checked SPM budget
    // and silently landed on SDPA-illegal chunk sizes — repro at seq=468
    // (hi=480) and seq=512: the picked cs failed sdpa_is_valid_chunk_size's
    // tile_m_v16 * tile_n_v16 * tile_k_v16 ≤ 1024 / VLM ≤ 400 / mask-tiling
    // constraints. seq=384 happened to pick a valid cs and worked, masking
    // the bug. test_qwen3_prefill.py historically worked around this with a
    // manual set_chunk_size(256). The fix moves the check into the auto path.
    //
    // sdpa_is_valid_chunk_size lives in rpu_helpers.h:118 and is the same
    // function the SDPA kernel-launch path uses to validate its config — so
    // by definition any cs we approve here will not be rejected downstream.
    // ═══════════════════════════════════════════════════════════════════════
    bool subclass_chunk_size_valid(int64_t cs, int64_t seq_len, int64_t position) const override {
        // The explicit-mask implementation prepares one rectangular mask for
        // the whole query. Multi-chunk candidates are not part of its domain.
        if (ctx().attention_mask.has_value() && cs < seq_len) return false;
        // Causal prefill requires any trailing physical chunk to contain at least one
        // v16. Apply the same guard to AUTO and EXACT plans; another aligned chunk
        // size may preserve a valid off-grid logical request. A single physical chunk
        // has no tail launch. KV_FIRST full-KV/MASK_NONE uses a separate contract and
        // does not inherit this causal-only restriction.
        const int64_t tail = seq_len % cs;
        if (ctx().is_causal && seq_len > cs && tail > 0 && tail < 16) return false;
        // MR-A: certified envelope, per candidate. This is the half that closes
        // the EXPLICIT-OVERRIDE route — compute_chunks_impl:499 validates an
        // override through THIS hook but never reads chunk_size_cap, so a guard
        // living only in the cap hook stops the auto scan and nothing else
        // (causal_decoder_set_chunk_size_override -> compute_chunks_impl -> here).
        // Full route ledger: docs/roadmap/chunk_certified_envelope.md §4.
        if (seq_len > 1 && !chunk_within_envelope(cs)) return false;
        // Same reasoning for the pre-existing per-handle cap: it too was read
        // ONLY by the auto scan, so an override sailed straight past it.
        // Qwen3_5Model::subclass_chunk_size_valid (rpu_qwen3_5_model.cpp:943)
        // already had this check; CausalDecoderModel did not. equal_two_prefill_
        // is excluded because its own branch below is strictly tighter.
        if (seq_len > 1 && configured_chunk_size_cap_ > 0 && !equal_two_prefill_
            && cs > configured_chunk_size_cap_) {
            return false;
        }
        if (equal_two_prefill_ && configured_chunk_size_cap_ > 0
            && position == 0 && seq_len > configured_chunk_size_cap_
            && cs != seq_len / 2) {
            return false;
        }
        // P4: validate against the mask type that will run. The LTM acc checks
        // (sQryAcc%16 / %sQry, rpu_helpers.h:336-340) gate chunk selection and
        // would reject an arbitrary-position action chunk before MASK_2D even
        // runs; with mask=4 those checks are skipped so cs>=H is reachable.
        // 无显式 mask 时一律按 LTM(1) 校验 chunk-size(保守):LTM tile 约束(sQryAcc%16 等)比
        // MASK_NONE 更严,把 auto-scan 逼到更小的 chunk —— 小 chunk = 小 SDPA 临时区 = 不撞 SPM 顶
        // LTM-valid chunk 必然 NONE-runnable,故 launch 仍按 is_causal=false 跑 MASK_NONE(:1285)结果正确。
        // KV_FIRST bidirectional (is_causal=false, no mask): the kernel runs
        // MASK_NONE (attn_mask_type=0), which does NOT enforce the
        // position-dependent LTM sQryAcc%16 / %sQry checks (rpu_helpers.h:336,
        // gated on attn_mask_type==1). At a non-16-aligned prefix position those
        // checks reject EVERY cs (prefill frame≥1, e.g. position=1566 →
        // sQryAcc=1566, 1566%16=14≠0) although the MASK_NONE kernel runs fine.
        // Validate the position-INDEPENDENT LTM tile/VLM/sync + C3 constraints at
        // position=0 (sQryAcc=0 → the real-position acc check is skipped): yields
        // the same proven position-0 valid set, so no SPM over-pick, and unblocks
        // multi-frame prefill (position = f·frame_len). The SPM budget check in
        // compute_chunks_impl independently caps cs against actual free space.
        if (!ctx().is_causal && !ctx().attention_mask.has_value()) {
            return sdpa_is_valid_chunk_size(
                make_sdpa_config(1), cs, seq_len, /*position=*/0);
        }
        int mask = ctx().attention_mask.has_value() ? 4 : 1;
        return sdpa_is_valid_chunk_size(make_sdpa_config(mask), cs, seq_len, position);
    }

    int64_t subclass_chunk_size_cap(int64_t seq_len, int64_t position) const override {
        // MR-A: hard half of the certified envelope. Called ONCE per resolve, and
        // — this is what makes it a complete gate — called UNCONDITIONALLY by both
        // planner entry points before compute_chunks_impl branches on the override
        // (fused_model_base.cpp:615 dry, :1024 forward). So the deny lands on the
        // override route too, and it lands BEFORE any SPM allocation (step 7).
        const int64_t envelope_chunk =
            enforce_chunk_envelope(seq_len, position, "CausalDecoderModel");
        // Tighter of the two ceilings wins; 0 means "no ceiling from this source".
        auto tighten = [](int64_t a, int64_t b) {
            if (a <= 0) return b;
            if (b <= 0) return a;
            return std::min(a, b);
        };
        if (equal_two_prefill_ && configured_chunk_size_cap_ > 0
            && position == 0 && seq_len > configured_chunk_size_cap_) {
            TORCH_CHECK(
                seq_len % 32 == 0,
                "RPU_PLANNER_REJECT:CAPABILITY: equal-two prefill requires "
                "a 32-aligned execution length, got ",
                seq_len);
            TORCH_CHECK(
                seq_len <= 2 * configured_chunk_size_cap_,
                "RPU_PLANNER_REJECT:CAPABILITY: equal-two prefill exceeds "
                "the validated two-chunk range: seq_len=",
                seq_len, ", chunk_size_cap=", configured_chunk_size_cap_,
                ", max_seq_len=", 2 * configured_chunk_size_cap_);
            // seq_len/2 is already <= configured_chunk_size_cap_ by the check
            // above, but it must also respect the certified envelope.
            TORCH_CHECK(
                envelope_chunk <= 0 || seq_len / 2 <= envelope_chunk,
                "RPU_PLANNER_REJECT:CAPABILITY: equal-two prefill would "
                "need chunk=", seq_len / 2,
                ", above the certified envelope chunk=", envelope_chunk,
                " (seq_len=", seq_len, "). See "
                "docs/roadmap/chunk_certified_envelope.md.");
            return seq_len / 2;
        }
        return tighten(configured_chunk_size_cap_, envelope_chunk);
    }

    ChunkPlan plan_kv_first_chunks(const ChunkPlan& compute_plan) {
        // Reuse the compute chunk plan for KV insertion: compute_chunks_impl checks
        // the real KV_FIRST declare_buffers footprint with kv_cs == comp_cs.
        // An independent KV chunk size would need the same complete aliased-layout
        // accounting before allocation; a compute-only temporary estimate is insufficient.
        // Phase 2 attends the full cache, so this reuse changes staging granularity only.
        // kv_insert_chunk_size remains part of the allocation key.
        return compute_plan;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // HALO WS-2 /  per-tensor QK RoPE dispatch — **single
    // source of truth**. The 3-way (1D-RoPE / M-RoPE / partial-M-RoPE) decision
    // used to be hand-copied in build_layer_subgraph (sequential) AND
    // emit_kv_first_body (KV_FIRST); the partial_mrope branch was added to the
    // former only, so M-RoPE+no-mask+KV_FIRST silently used the wrong frequencies.
    // Both paths now route here so a new RoPE mode can never be wired into one
    // copy and forgotten in the other. Pointers/masks are derived from members
    // (matching the per-forward keepalive contract): cos_il/position_ids are only
    // dereferenced on their active leaf. rotary_dim == dim at all current sites.
    // ─────────────────────────────────────────────────────────────────────────
    void consume_causal_route(
        FmbRouteFamily family, int64_t site_id, int64_t selector,
        int64_t invocation = 0) const {
        TORCH_CHECK(ctx().has_complete_physical_manifest(),
                    "CausalDecoderModel production route requires a COMPLETE "
                    "physical descriptor");
        const auto topology = cold_topology_arguments();
        ctx().consume_physical_route(
            family, site_id, selector, /*flags=*/0,
            num_cores() == 8 ? at::IntArrayRef{} : at::IntArrayRef(topology),
            invocation);
    }

    void consume_qwen3vl_4b_w8_prefill_schedule(
        int layer_idx, const ChunkInfo& chunk) {
        const auto& plan = ctx().stage_plan;
        if (!qwen3vl_4b_w8_resident_prefill(
                ctx().seq_len, ctx().position, ctx().is_causal,
                ctx().attention_mask.has_value(), ctx().batch_size,
                plan.compute.chunks.size())) return;
        TORCH_CHECK(
            plan.chunk_mode == ChunkMode::SEQUENTIAL &&
                ctx().input_in_spm == (layer_idx != 0) &&
                ctx().output_to_spm == (layer_idx + 1 != num_layers()),
            "Qwen3-VL 4B W8 prefill requires one full-layer SPM-resident group");
        if (layer_idx == 0 && chunk.idx == 0) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                CAUSAL_DECODER_4B_W8_PREFILL_SCHEDULE_SITE,
                /*chunk-major SPM-resident=*/1, /*flags=*/0,
                {ctx().seq_len, plan.compute.plan.chunk_size,
                 static_cast<int64_t>(plan.compute.chunks.size()), num_layers()});
        }
    }

    void consume_qwen3vl_ordinary_prefill_schedule(
        int layer_idx, const ChunkInfo& chunk) {
        const auto& plan = ctx().stage_plan;
        if (!qwen3vl_ordinary_resident_prefill(
                ctx().seq_len, ctx().position, ctx().is_causal,
                ctx().attention_mask.has_value(), ctx().batch_size,
                plan.compute.chunks.size())) return;
        TORCH_CHECK(
            plan.chunk_mode == ChunkMode::SEQUENTIAL &&
                ctx().input_in_spm == (layer_idx != 0) &&
                ctx().output_to_spm == (layer_idx + 1 != num_layers()),
            "Qwen3-VL ordinary prefill requires one full-layer SPM-resident group");
        if (layer_idx == 0 && chunk.idx == 0) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                CAUSAL_DECODER_VL_PREFILL_SCHEDULE_SITE,
                /*chunk-major SPM-resident=*/1, /*flags=*/0,
                qwen3vl_ordinary_prefill_schedule_arguments(
                    ctx().seq_len, plan.compute.plan.chunk_size,
                    static_cast<int64_t>(plan.compute.chunks.size())));
        }
    }

    // A measured schedule candidate for ordinary row-parallel O projections.
    // Capability absence preserves the original AUTO_TILE descriptor. Once
    // selected, the per-chunk route and low-level launcher both fail closed.
    bool ordinary_row_weight_reuse_plan_enabled(
        const FmbThreeStageChunkPlan& plan, bool is_causal, int64_t batch_size) const {
        if (typeid(*this) != typeid(CausalDecoderModel) ||
            num_cores() != 8 || attn_tp() != 8 || !is_causal || batch_size != 1 ||
            plan.chunk_mode != ChunkMode::SEQUENTIAL || linear_acc32_ ||
            has_mrope_ || has_qkv_bias_ || nvfp4_ || decode_gate_up_fusion_ || adarms_ ||
            !deepstack_lang_layers_.empty() || pre_layers_residual1_ready_ ||
            planning_mode_ != CausalDecoderPlanningMode::ORDINARY ||
            wall_z1_dispatch_ || qwen3vl_pooler_z1_dispatch_ ||
            qwen3vl_multiview_text_composite_dispatch_ || layer_weights_.empty()) {
            return false;
        }
        if (!std::any_of(plan.compute.chunks.begin(), plan.compute.chunks.end(),
                [&](const ChunkInfo& chunk) {
                    return rpu_pl_tiling::supports_w8a16_row_weight_reuse(
                        chunk.len, hidden_size(), num_q_heads() * head_dim(),
                        0, attn_tp());
                })) return false;
        if (!std::all_of(layer_weights_.begin(), layer_weights_.end(),
                [](const LayerWeights& layer) {
                    return layer.o_w.defined() && layer.o_w.scalar_type() == at::kChar &&
                        layer.o_ws.defined() && layer.o_ws.scalar_type() == at::kHalf;
                })) return false;
        return KernelCache::instance().has_loaded(
            KernelId::W8A16_ROW_WEIGHT_REUSE_M352N80K256);
    }

    std::vector<int64_t> ordinary_o_route_arguments(int64_t rows, bool reuse) const {
        std::vector<int64_t> arguments{
            rows, hidden_size(), num_q_heads() * head_dim(), 0, attn_tp()};
        if (reuse) {
            // Full-local-K capacity and the preserved FP16 K128 retirement.
            arguments.insert(arguments.end(), {352, 80, 256, 128});
        }
        return arguments;
    }

    bool consume_ordinary_o_route(
        int64_t site_id, int64_t rows, int64_t chunk_index) const {
        TORCH_CHECK(site_id == CAUSAL_DECODER_O_SITE,
                    "ordinary O schedule received a foreign route site");
        if (!ordinary_row_weight_reuse_plan_enabled(
                ctx().stage_plan, ctx().is_causal, ctx().batch_size)) {
            consume_causal_route(FmbRouteFamily::LINEAR, site_id,
                static_cast<int64_t>(decoder_projection_route(seq_len_)));
            return false;
        }
        TORCH_CHECK(ctx().has_complete_physical_manifest(),
                    "row weight reuse requires a COMPLETE physical descriptor");
        const bool reuse = rpu_pl_tiling::supports_w8a16_row_weight_reuse(
            rows, hidden_size(), num_q_heads() * head_dim(), 0, attn_tp());
        ctx().consume_physical_route(FmbRouteFamily::LINEAR, site_id,
            static_cast<int64_t>(reuse ? CausalDecoderLinearRoute::ROW_WEIGHT_REUSE
                                      : CausalDecoderLinearRoute::AUTO_TILE),
            0, ordinary_o_route_arguments(rows, reuse), chunk_index);
        return reuse;
    }

    // Admit the ordinary FP16 SwiGLU dataflow only. Derived VLA owners and
    // packed/mixed/composite paths keep their independently selected schedules.
    bool ordinary_silu_mul_route_enabled() const {
        if (typeid(*this) != typeid(CausalDecoderModel) ||
            num_cores() != 8 || mlp_tp() != 8 || !use_silu_ || linear_acc32_ ||
            intermediate_size() <= 0 || intermediate_size() % 128 != 0 ||
            has_mrope_ || has_qkv_bias_ || nvfp4_ || decode_gate_up_fusion_ || adarms_ ||
            !deepstack_lang_layers_.empty() || pre_layers_residual1_ready_ ||
            planning_mode_ != CausalDecoderPlanningMode::ORDINARY ||
            wall_z1_dispatch_ || qwen3vl_pooler_z1_dispatch_ ||
            qwen3vl_multiview_text_composite_dispatch_ || layer_weights_.empty()) {
            return false;
        }
        return std::all_of(layer_weights_.begin(), layer_weights_.end(),
            [](const LayerWeights& layer) {
                return layer.gate_w.scalar_type() == at::kHalf &&
                    layer.up_w.scalar_type() == at::kHalf &&
                    layer.down_w.scalar_type() == at::kHalf;
            });
    }

    bool use_planned_silu_mul() const {
        return ordinary_silu_mul_route_enabled() &&
            ctx().has_complete_physical_manifest() && ctx().is_causal &&
            ctx().batch_size == 1 &&
            ctx().stage_plan.chunk_mode == ChunkMode::SEQUENTIAL;
    }

    // This is an execution-ABI gate, not a model allowlist.  Exact delivery,
    // mixed-dtype, and AdaRMS owners keep their separately manifested routes.
    void consume_rhinovla_high_activation(int64_t rows, int64_t invocation) const {
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            RHINOVLA_TEXT_HIGH_ACTIVATION_SITE, 1, rhinovla_high_precision_fusions_ ? 1 : 0,
            {rows, intermediate_size() / mlp_tp(), 1}, invocation);
    }

    bool ordinary_rmsnorm_route_enabled() const {
        return (rhinovla_high_precision_ || ordinary_rmsnorm_spm_contract_.has_vector_payload()) &&
            !adarms_ && !qwen3vl_2b_w8_profile_;
    }

    RpuRmsNormSpmRoute consume_ordinary_rmsnorm_route(
        int64_t rows, int64_t cols, int64_t chunk_index, int64_t role) const {
        if (!ordinary_rmsnorm_route_enabled() || !ctx().is_causal ||
            ctx().batch_size != 1 ||
            ctx().stage_plan.chunk_mode != ChunkMode::SEQUENTIAL) {
            return RpuRmsNormSpmRoute::BASE;
        }
        if (rhinovla_high_precision_) {
            const auto route = rpu_resolve_high_precision_rmsnorm_spm_route(rows, cols);
            ctx().consume_physical_route(FmbRouteFamily::NORMALIZATION,
                RHINOVLA_TEXT_HIGH_RMSNORM_SITE, static_cast<int64_t>(route), 0,
                {rows, cols, 1}, 3 * chunk_index + role);
            return route;
        }
        // The generic contract selects a vector payload only for eligible
        // operator shapes. Decode and unaligned tails remain BASE.
        const auto route =
            ordinary_rmsnorm_spm_contract_.resolve(rows, cols);
        TORCH_CHECK(
            ctx().has_complete_physical_manifest(),
            "ordinary RMSNorm route requires a COMPLETE physical descriptor");
        ctx().consume_physical_route(
            FmbRouteFamily::NORMALIZATION,
            CAUSAL_DECODER_ORDINARY_RMSNORM_SITE,
            static_cast<int64_t>(route), /*resolved_flags=*/0,
            ordinary_rmsnorm_spm_contract_.manifest_arguments(rows, cols),
            3 * chunk_index + role);
        return route;
    }

    RpuRmsNormSpmRoute consume_decoder_rmsnorm_route(
        int64_t rows, int64_t cols, int64_t invocation) const {
        if (qwen3vl_2b_w8_profile_) {
            return consume_delivery_rmsnorm_route(rows, cols, invocation, 0);
        }
        if (ordinary_rmsnorm_route_enabled() && ctx().is_causal &&
            ctx().batch_size == 1 &&
            ctx().stage_plan.chunk_mode == ChunkMode::SEQUENTIAL) {
            return consume_ordinary_rmsnorm_route(
                rows, cols, invocation, /*role=*/0);
        }
        return RpuRmsNormSpmRoute::BASE;
    }

    KvInsertSegmentPlan consume_causal_kv_plan(
        int64_t site_id, int64_t invocation, int64_t live_position,
        int64_t logical_rows) const {
        TORCH_CHECK(ctx().has_complete_physical_manifest(),
                    "CausalDecoderModel KV route requires a COMPLETE physical "
                    "descriptor");
        const FmbRouteManifestEntry& route = ctx().find_physical_route(
            FmbRouteFamily::KV_INSERT, site_id, invocation);
        const KvInsertSegmentPlan template_plan =
            restore_kvinsert_plan(
                site_id, route.arguments, attn_tp(), num_kv_heads(), head_dim());
        const bool dynamic_position =
            (route.flags & KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION) != 0;
        TORCH_CHECK(!dynamic_position ||
                        (logical_rows == 1 &&
                         template_plan.segment(0).position == 0),
                    "CausalDecoderModel dynamic KV route is only valid for a "
                    "canonical single-token decode template");
        const KvInsertSegmentPlan live_plan = dynamic_position
            ? rpu_rebase_kvinsert_segment_plan_position(
                  template_plan, live_position, attn_tp(), num_kv_heads(),
                  head_dim())
            : template_plan;
        TORCH_CHECK(
            live_plan.logical_rows() == logical_rows &&
                live_plan.segment(0).position == live_position &&
                (kv_insert_spm_rows_ == 0 ||
                 live_plan.physical_rows() <= kv_insert_spm_rows_),
            "CausalDecoderModel KV descriptor geometry drift at site ",
            site_id, ", invocation ", invocation);
        ctx().consume_physical_route(
            FmbRouteFamily::KV_INSERT, site_id,
            static_cast<int64_t>(template_plan.route()), route.flags,
            route.arguments, invocation);
        return live_plan;
    }

    void emit_qk_rope(const char* in_slot, const char* out_slot,
                      int64_t n_heads, int64_t dim, int64_t seq_len,
        int64_t cos_sin_start, int tp, uint32_t spm_offset_bytes = 0) {
        if (has_mrope_ && partial_mrope_active_) {
            if (qwen3vl_pooler_z1_dispatch_ ||
                qwen3vl_multiview_text_composite_dispatch_) {
                ctx().consume_physical_route(
                    FmbRouteFamily::ROPE,
                    CAUSAL_DECODER_PARTIAL_MROPE_TENSOR_SITE,
                    static_cast<int64_t>(
                        CausalDecoderRopeRoute::PARTIAL_MROPE),
                    /*resolved_flags=*/0,
                    {has_mrope_ ? 1 : 0,
                     (has_mrope_ && partial_mrope_active_) ? 1 : 0,
                     (qwen3vl_pooler_z1_dispatch_ ||
                      qwen3vl_multiview_text_composite_dispatch_)
                         ? 1 : 0});
                rpu_launch_partial_mrope_spm_kernel(
                    addr(0, in_slot) + spm_offset_bytes,
                    addr(0, out_slot) + spm_offset_bytes,
                    cos_il_keepalive_, sin_il_keepalive_, cos_sin_start,
                    seq_len, n_heads, dim, /*rotary_dim=*/dim, tp);
            } else {
                ctx().consume_physical_route(
                    FmbRouteFamily::ROPE,
                    CAUSAL_DECODER_PARTIAL_MROPE_POINTER_SITE,
                    static_cast<int64_t>(
                        CausalDecoderRopeRoute::PARTIAL_MROPE),
                    /*resolved_flags=*/0,
                    {has_mrope_ ? 1 : 0,
                     (has_mrope_ && partial_mrope_active_) ? 1 : 0,
                     (qwen3vl_pooler_z1_dispatch_ ||
                      qwen3vl_multiview_text_composite_dispatch_)
                         ? 1 : 0});
                rpu_launch_partial_mrope_spm_kernel(
                    addr(0, in_slot) + spm_offset_bytes,
                    addr(0, out_slot) + spm_offset_bytes,
                    cos_il_keepalive_.data_ptr<c10::Half>(),
                    sin_il_keepalive_.data_ptr<c10::Half>(),
                    cos_sin_start, seq_len, n_heads, dim,
                    /*rotary_dim=*/dim, tp);
            }
        } else if (has_mrope_ &&
                   (qwen3vl_pooler_z1_dispatch_ ||
                    qwen3vl_multiview_text_composite_dispatch_)) {
            ctx().consume_physical_route(
                FmbRouteFamily::ROPE, CAUSAL_DECODER_MROPE_TENSOR_SITE,
                static_cast<int64_t>(CausalDecoderRopeRoute::MROPE),
                /*resolved_flags=*/0,
                {has_mrope_ ? 1 : 0,
                 (has_mrope_ && partial_mrope_active_) ? 1 : 0,
                 (qwen3vl_pooler_z1_dispatch_ ||
                  qwen3vl_multiview_text_composite_dispatch_)
                     ? 1 : 0});
            rpu_launch_mrope_spm_kernel(
                addr(0, in_slot) + spm_offset_bytes,
                addr(0, out_slot) + spm_offset_bytes,
                cos_, sin_, position_ids_keepalive_, cos_sin_start,
                mrope_strobe_masks_, seq_len, n_heads, dim, tp);
        } else if (has_mrope_) {
            ctx().consume_physical_route(
                FmbRouteFamily::ROPE, CAUSAL_DECODER_MROPE_POINTER_SITE,
                static_cast<int64_t>(CausalDecoderRopeRoute::MROPE),
                /*resolved_flags=*/0,
                {has_mrope_ ? 1 : 0,
                 (has_mrope_ && partial_mrope_active_) ? 1 : 0,
                 (qwen3vl_pooler_z1_dispatch_ ||
                  qwen3vl_multiview_text_composite_dispatch_)
                     ? 1 : 0});
            rpu_launch_mrope_spm_kernel(
                addr(0, in_slot) + spm_offset_bytes,
                addr(0, out_slot) + spm_offset_bytes,
                cos_.data_ptr<c10::Half>(), sin_.data_ptr<c10::Half>(),
                position_ids_keepalive_.data_ptr<int32_t>(), cos_sin_start,
                mrope_strobe_masks_, seq_len, n_heads, dim, tp);
        } else {
            ctx().consume_physical_route(
                FmbRouteFamily::ROPE, CAUSAL_DECODER_ROPE_1D_SITE,
                static_cast<int64_t>(CausalDecoderRopeRoute::ROPE_1D),
                /*resolved_flags=*/0,
                {has_mrope_ ? 1 : 0,
                 (has_mrope_ && partial_mrope_active_) ? 1 : 0,
                 (qwen3vl_pooler_z1_dispatch_ ||
                  qwen3vl_multiview_text_composite_dispatch_)
                     ? 1 : 0});
            rpu_launch_rope_spm_kernel(
                addr(0, in_slot) + spm_offset_bytes,
                addr(0, out_slot) + spm_offset_bytes,
                cos_.data_ptr<c10::Half>(), sin_.data_ptr<c10::Half>(),
                seq_len, n_heads, dim, cos_sin_start, tp);
        }
    }

    void emit_qk_rope_pair(
        const char* q_in_slot, const char* q_out_slot,
        const char* k_in_slot, const char* k_out_slot,
        int64_t q_heads, int64_t k_heads, int64_t dim,
        int64_t seq_len, int64_t cos_sin_start, int tp, int64_t chunk_index) {
        const bool fused =
            has_mrope_ && partial_mrope_active_ && q_heads == 2 &&
            k_heads == 1 && dim == 128 && tp == NUM_CORES &&
            ((seq_len == 192 && fused_prefill_qk_mrope_) ||
             (seq_len == 1 && fused_decode_qk_mrope_));
        if (fused) {
            ctx().consume_physical_route(FmbRouteFamily::ROPE,
                QWEN3VL_DELIVERY_ROPE_SITE, 4, 0,
                {seq_len, q_heads, k_heads, dim, tp}, chunk_index);
            rpu_launch_partial_mrope_qk_spm_kernel(
                addr(0, q_in_slot), addr(0, q_out_slot),
                addr(0, k_in_slot), addr(0, k_out_slot),
                cos_il_keepalive_, sin_il_keepalive_,
                cos_sin_start, seq_len, q_heads, k_heads,
                dim, /*rotary_dim=*/dim, tp);
            return;
        }
        emit_qk_rope(
            q_in_slot, q_out_slot, q_heads, dim,
            seq_len, cos_sin_start, tp);
        emit_qk_rope(
            k_in_slot, k_out_slot, k_heads, dim,
            seq_len, cos_sin_start, tp);
    }

    void emit_qwen3vl_4b_decode_qk_norm_mrope(
        int layer_idx, const ChunkInfo& chunk, int64_t position) {
        TORCH_CHECK(qwen3vl_4b_decode_qk_fused_ &&
                        ctx().stage_plan.chunk_mode == ChunkMode::SEQUENTIAL &&
                        ctx().stage_plan.compute.chunks.size() == 1 &&
                        qwen3vl_4b_decode_qk_norm_mrope(seq_len_, chunk.len,
                            ctx().is_causal, ctx().attention_mask.has_value(),
                            ctx().batch_size, partial_mrope_active_),
                    "4B fused QK leaf requires its cold exact decode capability");
        ctx().consume_physical_route(FmbRouteFamily::NORMALIZATION,
            QWEN3VL_4B_DECODE_QK_NORM_MROPE_SITE, 1, 0,
            {1, 4, 1, 128, 8, 24, 20, 20}, chunk.idx);
        rpu_launch_qwen3vl_4b_decode_qk_norm_mrope_spm_kernel(
            addr(0, "q"), addr(0, "k"), norm_q_addr(layer_idx, 0),
            norm_k_addr(layer_idx, 0), cos_, sin_, position_ids_keepalive_,
            position, eps_);
    }

    void emit_qwen3vl_4b_decode_qk_norm_mrope_kv_insert(
        int layer_idx, const ChunkInfo& chunk, int64_t position) {
        TORCH_CHECK(qwen3vl_4b_decode_qk_kv_fused_ &&
                        ctx().attention_policy == AttentionExecutionPolicy::DDR_KV &&
                        cache_batch_slot_ == 0 && chunk.offset == 0 &&
                        ctx().stage_plan.chunk_mode == ChunkMode::SEQUENTIAL &&
                        ctx().stage_plan.compute.chunks.size() == 1 &&
                        qwen3vl_4b_decode_qk_norm_mrope(seq_len_, chunk.len,
                            ctx().is_causal, ctx().attention_mask.has_value(),
                            ctx().batch_size, partial_mrope_active_),
                    "4B fused QK/KV leaf requires its cold exact DDR decode capability");
        ctx().consume_physical_route(FmbRouteFamily::NORMALIZATION,
            QWEN3VL_4B_DECODE_QK_NORM_MROPE_KV_INSERT_SITE, 1, 0,
            {1, 4, 1, 128, 8, 24, 20, 20}, chunk.idx);
        rpu_launch_qwen3vl_4b_decode_qk_norm_mrope_kv_insert_spm_kernel(
            addr(0, "q"), addr(0, "k"), addr(0, "v"),
            norm_q_addr(layer_idx, 0), norm_k_addr(layer_idx, 0),
            cos_, sin_, position_ids_keepalive_,
            (*ctx().k_caches)[layer_idx], (*ctx().v_caches)[layer_idx],
            position, eps_);
    }

    void emit_qwen3vl_4b_decode_o_ring_norm(
        const LayerWeights& lw, int layer_idx, const ChunkInfo& chunk,
        uint32_t input_residual) {
        TORCH_CHECK(ctx().stage_plan.chunk_mode == ChunkMode::SEQUENTIAL &&
                        ctx().stage_plan.compute.chunks.size() == 1 &&
                        ctx().attention_policy == AttentionExecutionPolicy::DDR_KV &&
                        chunk.offset == 0 && input_residual == addr(0, "residual1") &&
                        qwen3vl_4b_decode_o_ring_norm(seq_len_, chunk.len,
                            ctx().is_causal, ctx().attention_mask.has_value(),
                            ctx().batch_size, partial_mrope_active_),
                    "4B fused O/ring/norm requires its cold exact ordinary decode capability");
        ctx().consume_physical_route(FmbRouteFamily::LINEAR,
            QWEN3VL_4B_DECODE_O_RING_NORM_SITE, 1, 0,
            {1, 2560, 4096, 512, 8, 320, 1, 63488}, chunk.idx);
        rpu_launch_qwen3vl_4b_decode_o_ring_norm_spm_kernel(
            addr(0, "output"), lw.o_w, lw.o_ws, addr(0, "oproj"),
            input_residual, addr(0, "residual2"), norm_post_addr(layer_idx, 0), eps_);
    }

    void emit_qwen3vl_4b_prefill_mlp(const LayerWeights& lw, const ChunkInfo& chunk) {
        const int64_t rows = chunk.len;
        ctx().consume_physical_route(FmbRouteFamily::LINEAR,
            QWEN3VL_4B_PREFILL_GATEUP_SITE, 1, 0,
            {rows, 9728, 2560, 8, 320, 112}, chunk.idx);
        rpu_launch_qwen3vl_4b_prefill_gate_up_swiglu_w8a16_spm_kernel(
            addr(0, "residual1"), lw.gate_w, lw.up_w, addr(0, "gate"),
            rows, lw.gate_ws, lw.up_ws);
        ctx().consume_physical_route(FmbRouteFamily::LINEAR,
            QWEN3VL_4B_PREFILL_DOWN_SITE,
            static_cast<int64_t>(CausalDecoderLinearRoute::AUTO_TILE),
            0, {rows, 2560, 9728, 0, 8}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "gate"), lw.down_w, addr(0, "down"),
            rows, 2560, 9728, /*partition=*/0, 8, /*bias=*/0, lw.down_ws);
        ctx().consume_physical_route(FmbRouteFamily::ALL_REDUCE,
            QWEN3VL_4B_PREFILL_MLP_REDUCE_SITE, causal_ring_route(rows, 2560),
            0, {}, chunk.idx);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "down"), addr(0, "residual2"), addr(0, "residual1"),
            rows, 2560, 8, 8);
    }

    void emit_qwen3vl_decode_mlp(const LayerWeights& lw, const ChunkInfo& chunk) {
        TORCH_CHECK(ctx().stage_plan.chunk_mode == ChunkMode::SEQUENTIAL &&
                qwen3vl_decode_mlp(seq_len_, chunk.len, ctx().is_causal,
                    ctx().attention_mask.has_value(), ctx().batch_size),
                    "Qwen3-VL W8 decode MLP escaped its ordinary TP8 M1 contract");
        if (qwen3vl_4b_decode_gateup_fused_) {
            ctx().consume_physical_route(FmbRouteFamily::LINEAR,
                QWEN3VL_4B_DECODE_MLP_GATEUP_FUSED_SITE, 1, 0,
                {1, 9728, 2560, 8, 160, 96}, chunk.idx);
            rpu_launch_qwen3vl_4b_decode_gate_up_swiglu_w8a16_spm_kernel(
                addr(0, "residual1"), lw.gate_w, lw.up_w, addr(0, "gate"),
                lw.gate_ws, lw.up_ws);
        } else {
            ctx().consume_physical_route(FmbRouteFamily::LINEAR,
                QWEN3VL_4B_DECODE_MLP_GATE_SITE,
                static_cast<int64_t>(CausalDecoderLinearRoute::GEMV),
                0, {1, intermediate_size(), hidden_size(), 1, 8}, chunk.idx);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "residual1"), lw.gate_w, addr(0, "gate"),
                1, intermediate_size(), hidden_size(), /*partition=*/1, 8, /*bias=*/0, lw.gate_ws,
                /*nvfp4_tensor_scale_spm_addr=*/0, /*nvfp4_layer_id=*/0,
                /*force_acc32=*/false, /*prefer_gemv=*/true);
            ctx().consume_physical_route(FmbRouteFamily::LINEAR,
                QWEN3VL_4B_DECODE_MLP_UP_SITE,
                static_cast<int64_t>(CausalDecoderLinearRoute::GEMV),
                0, {1, intermediate_size(), hidden_size(), 1, 8}, chunk.idx);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "residual1"), lw.up_w, addr(0, "up"),
                1, intermediate_size(), hidden_size(), /*partition=*/1, 8, /*bias=*/0, lw.up_ws,
                /*nvfp4_tensor_scale_spm_addr=*/0, /*nvfp4_layer_id=*/0,
                /*force_acc32=*/false, /*prefer_gemv=*/true);
            ctx().consume_physical_route(FmbRouteFamily::ACTIVATION,
                QWEN3VL_4B_DECODE_MLP_ACTIVATION_SITE, 1, 0,
                {1, intermediate_size() / 8, 8}, chunk.idx);
            rpu_launch_silu_mul_spm_kernel(
                addr(0, "gate"), addr(0, "up"), addr(0, "gate"), intermediate_size() / 8, 8);
        }
        ctx().consume_physical_route(FmbRouteFamily::LINEAR,
            QWEN3VL_4B_DECODE_MLP_DOWN_SITE,
            static_cast<int64_t>(CausalDecoderLinearRoute::GEMV),
            0, {1, hidden_size(), intermediate_size(), 0, 8}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "gate"), lw.down_w, addr(0, "down"),
            1, hidden_size(), intermediate_size(), /*partition=*/0, 8, /*bias=*/0, lw.down_ws,
            /*nvfp4_tensor_scale_spm_addr=*/0, /*nvfp4_layer_id=*/0,
            /*force_acc32=*/false, /*prefer_gemv=*/true);
        ctx().consume_physical_route(FmbRouteFamily::ALL_REDUCE,
            QWEN3VL_4B_DECODE_MLP_REDUCE_SITE, causal_ring_route(1, hidden_size()),
            0, {}, chunk.idx);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "down"), addr(0, "residual2"), addr(0, "residual1"),
            1, hidden_size(), 8, 8);
    }

    void emit_qwen3vl_2b_mlp_pipeline(
        const LayerWeights& lw, int64_t seq_len, int64_t h,
        int64_t nq, int64_t nkv, int64_t hd, int tp, int64_t chunk_index) {
        TORCH_CHECK(
            qwen3vl_2b_w8_profile_ && decode_gate_up_fusion_ && lw.gate_up_w.defined() &&
                (seq_len == 1 || (seq_len == 192 && prefill_packed_gate_up_strided_)) &&
                use_silu_ && num_layers() == 28 && h == 2048 &&
                intermediate_size() == 6144 && nq == 16 && nkv == 8 &&
                hd == 128 && tp == NUM_CORES,
            "packed gate/up path escaped the Qwen3-VL 2B topology guard");
        const int64_t intermediate = intermediate_size();
        const int64_t elems = seq_len * (intermediate / NUM_CORES);
        uint32_t activated_addr;
        ctx().consume_physical_route(FmbRouteFamily::LINEAR,
            QWEN3VL_DELIVERY_MLP_LINEAR_SITE, 1, 0,
            {seq_len, 12288, 2048, 1, 8}, chunk_index);
        const uint32_t packed_addr = addr(0, "gate_up");
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "residual1"), lw.gate_up_w, packed_addr,
            seq_len, 2 * intermediate, h, /*partition=*/1, NUM_CORES,
            /*bias_spm_addr=*/0, lw.gate_up_ws,
            /*nvfp4_tensor_scale_spm_addr=*/0,
            /*nvfp4_layer_id=*/0, /*force_acc32=*/false,
            /*prefer_gemv=*/false, /*qkv_planar_direct=*/false);
        if (seq_len == 192) {
            ctx().consume_physical_route(FmbRouteFamily::ACTIVATION,
                QWEN3VL_DELIVERY_MLP_ACTIVATION_SITE, 2, 0,
                {seq_len, 768, 8}, chunk_index);
            activated_addr = addr(0, "gate_compact");
            rpu_launch_silu_mul_exact_strided_spm_kernel(
                packed_addr, activated_addr, seq_len,
                intermediate / NUM_CORES, NUM_CORES);
        } else {
            ctx().consume_physical_route(FmbRouteFamily::ACTIVATION,
                QWEN3VL_DELIVERY_MLP_DECODE_ACTIVATION_SITE, 1, 0,
                {seq_len, 768, 8}, chunk_index);
            activated_addr = packed_addr;
            rpu_launch_silu_mul_spm_kernel(
                packed_addr, packed_addr + static_cast<uint32_t>(elems * DWIDTH),
                activated_addr, elems, NUM_CORES);
        }
        ctx().consume_physical_route(FmbRouteFamily::LINEAR,
            QWEN3VL_DELIVERY_MLP_DOWN_SITE, 1, 0,
            {seq_len, 2048, 6144, 0, 8}, chunk_index);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            activated_addr, lw.down_w, addr(0, "down"),
            seq_len, h, intermediate, /*partition=*/0, NUM_CORES,
            /*bias_spm_addr=*/0, lw.down_ws,
            /*nvfp4_tensor_scale_spm_addr=*/0,
            /*nvfp4_layer_id=*/0, /*force_acc32=*/false,
            /*prefer_gemv=*/false, /*qkv_planar_direct=*/false);
        ctx().consume_physical_route(FmbRouteFamily::ALL_REDUCE,
            QWEN3VL_DELIVERY_MLP_REDUCE_SITE, causal_ring_route(seq_len, h),
            0, {}, chunk_index);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "down"), addr(0, "residual2"), addr(0, "residual1"),
            seq_len, h, NUM_CORES, NUM_CORES);
    }

    // Replay-safe AdaRMS (set_adarms_step_mutable): before each RMSNorm, re-load this
    // layer's scale (norm_w slot) and shift (input_shift/post_shift slot) into SPM from the
    // stable keepalive. Recorded into the graph body ⇒ re-executes on every REPLAY, picking
    // up the per-step contents the adapter wrote. No-op unless adarms_mutable_ (so every
    // other decoder — and the plain build-per-step AdaRMS — is unaffected).
    // B1: one broadcast DMA fills all `N` layers of a per-layer [hidden] SPM slot.
    // Requires the slot's layers to be ASCENDING and gap-free — guaranteed by
    // BufferDecl::reverse_layer_alloc, VERIFIED here rather than assumed: if the
    // layout is ever the other way round (flag mismatch, decl edit, allocator
    // change) a silent write would scramble scale/shift across layers and only show
    // up as a wrong action. TORCH_CHECK turns that into a loud build-time failure.
    // BUILD-time only (not per REPLAY), so the N address lookups are free.
    uint32_t validate_adarms_stack_layout(int64_t N, int64_t h, const char* slot) {
        TORCH_CHECK(N >= 2, "validate_adarms_stack_layout: needs >=2 layers, got ", N);
        const uint32_t a0 = layer_addr(0, 0, slot);
        // SpmAllocator::ALIGN == 256 (rpu_spm_allocator.h:36); spelled out rather
        // than pulled from a helper so the stride here cannot drift from the decl's
        // `nw = A(h*DWIDTH)` without the check below firing.
        const uint32_t stride = (uint32_t)(((h * DWIDTH + 255) / 256) * 256);
        for (int64_t L = 1; L < N; ++L) {
            TORCH_CHECK(layer_addr((int)L, 0, slot) == a0 + (uint32_t)L * stride,
                "validate_adarms_stack_layout: SPM slot '", slot, "' is not ascending "
                "contiguous at layer ", L, " (got ", layer_addr((int)L, 0, slot),
                ", want ", a0 + (uint32_t)L * stride, ", stride ", stride,
                "). AdaRMS fused broadcast requires reverse_layer_alloc on this decl.");
        }
        return a0;
    }

    void emit_adarms_stack_bcast(c10::Half* ddr, int64_t N, int64_t h, const char* slot) {
        rpu_launch_ddr_broadcast_spm_dma(
            ddr, N * h, validate_adarms_stack_layout(N, h, slot), num_cores());
    }

    void emit_adarms_stack_bcast_mutable(
        const uint64_t* live_base, int64_t N, int64_t h, const char* slot)
    {
        consume_causal_route(
            FmbRouteFamily::MUTABLE_DMA,
            CAUSAL_DECODER_ADARMS_MUTABLE_SITE,
            static_cast<int64_t>(
                CausalDecoderMutableDmaRoute::DDR_BROADCAST_TO_SPM));
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            live_base, /*src_offset_bytes=*/0, N * h,
            validate_adarms_stack_layout(N, h, slot), num_cores());
    }

    void emit_adarms_mut_refresh_input(int64_t L) {
        if (!adarms_mutable_) return;  // guarded ⇒ input_shift_[L] is resized (safe)
        const int64_t h = hidden_size();
        if (adarms_fused_bcast_enabled_) {
            // Whole 36-layer stack in one DMA per role, emitted once at layer 0.
            // Emitting at L==0 (instead of hoisting out of the layer loop) keeps the
            // build structure untouched: the DMA still lands in the graph body ahead
            // of every layer's norm, so all layers see their data before use.
            if (L != 0) return;
            const int64_t N = num_layers();
            if (adarms_schedule_select_) {
                emit_adarms_stack_bcast_mutable(
                    &adarms_is_live_base_, N, h, "norm_w");
                emit_adarms_stack_bcast_mutable(
                    &adarms_ish_live_base_, N, h, "input_shift");
                return;
            }
            emit_adarms_stack_bcast(adarms_unroll_
                ? adarms_is_.data_ptr<c10::Half>() + ctx().body_iter * N * h
                : input_scale_ka_.data_ptr<c10::Half>(), N, h, "norm_w");
            emit_adarms_stack_bcast(adarms_unroll_
                ? adarms_ish_.data_ptr<c10::Half>() + ctx().body_iter * N * h
                : input_shift_ka_.data_ptr<c10::Half>(), N, h, "input_shift");
            return;
        }
        // In-graph unroll: read this body_iter's (scale, shift) slice from the
        // [num_steps, num_layers, hidden] fold buffers. Single-step (adarms_unroll_
        // false) keeps the mutable-keepalive src ⇒ byte-identical.
        c10::Half* scale_ptr = adarms_unroll_
            ? adarms_is_.data_ptr<c10::Half>() + (ctx().body_iter * num_layers() + L) * h
            : layer_weights_[L].input_norm_w.data_ptr<c10::Half>();
        c10::Half* shift_ptr = adarms_unroll_
            ? adarms_ish_.data_ptr<c10::Half>() + (ctx().body_iter * num_layers() + L) * h
            : input_shift_[L].data_ptr<c10::Half>();
        rpu_launch_ddr_broadcast_spm_dma(scale_ptr, h, layer_addr((int)L, 0, "norm_w"), num_cores());
        rpu_launch_ddr_broadcast_spm_dma(shift_ptr, h, layer_addr((int)L, 0, "input_shift"), num_cores());
    }
    void emit_adarms_mut_refresh_post(int64_t L) {
        if (!adarms_mutable_) return;
        const int64_t h = hidden_size();
        if (adarms_fused_bcast_enabled_) {
            if (L != 0) return;
            const int64_t N = num_layers();
            if (adarms_schedule_select_) {
                emit_adarms_stack_bcast_mutable(
                    &adarms_ps_live_base_, N, h, "post_norm_w");
                emit_adarms_stack_bcast_mutable(
                    &adarms_psh_live_base_, N, h, "post_shift");
                return;
            }
            emit_adarms_stack_bcast(adarms_unroll_
                ? adarms_ps_.data_ptr<c10::Half>() + ctx().body_iter * N * h
                : post_scale_ka_.data_ptr<c10::Half>(), N, h, "post_norm_w");
            emit_adarms_stack_bcast(adarms_unroll_
                ? adarms_psh_.data_ptr<c10::Half>() + ctx().body_iter * N * h
                : post_shift_ka_.data_ptr<c10::Half>(), N, h, "post_shift");
            return;
        }
        c10::Half* scale_ptr = adarms_unroll_
            ? adarms_ps_.data_ptr<c10::Half>() + (ctx().body_iter * num_layers() + L) * h
            : layer_weights_[L].post_norm_w.data_ptr<c10::Half>();
        c10::Half* shift_ptr = adarms_unroll_
            ? adarms_psh_.data_ptr<c10::Half>() + (ctx().body_iter * num_layers() + L) * h
            : post_shift_[L].data_ptr<c10::Half>();
        rpu_launch_ddr_broadcast_spm_dma(scale_ptr, h, layer_addr((int)L, 0, "post_norm_w"), num_cores());
        rpu_launch_ddr_broadcast_spm_dma(shift_ptr, h, layer_addr((int)L, 0, "post_shift"), num_cores());
    }

    void emit_kv_first_body(int layer_idx, const ChunkInfo& chunk) {
        const auto& lw = layer_weights_[layer_idx];
        int64_t seq_len = chunk.len;
        const int64_t rope_base =
            (!has_mrope_ && rope_position_base_ >= 0) ? rope_position_base_ : ctx().position;
        int64_t cos_sin_start = rope_base + chunk.offset;
        int64_t h = hidden_size();
        int64_t nq = num_q_heads();
        int64_t nkv = num_kv_heads();
        int64_t hd = head_dim();
        int tp = attn_tp();
        const Z1LayerInputSelection z1_input =
            select_z1_layer_input(layer_idx, chunk);
        const uint32_t input_residual = z1_input.addr;

        if (qwen3vl_multiview_text_composite_dispatch_ && layer_idx == 0) {
            emit_qwen3vl_multiview_text_complement_ingress(
                layer_idx, chunk, input_residual);
        } else if (!ctx().input_in_spm && !z1_input.external &&
                   !(pre_layers_residual1_ready_ && layer_idx == 0)) {
            emit_layer_input_dma(layer_idx, chunk);
        }

        emit_adarms_mut_refresh_input(layer_idx);  // replay-safe AdaRMS: refresh scale+shift SPM
        rpu_launch_rmsnorm_spm_kernel(
            input_residual, addr(0, "input_norm"),
            norm_input_addr(layer_idx, 0),   // adarms_: scale rides here
            seq_len, h, eps_, RpuRmsNormSpmRoute::BASE, num_cores());
        if (adarms_) {  // AdaRMS FiLM: input_norm += input_shift (LingBot-VLA expert)
            rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                layer_addr(layer_idx, 0, "input_shift"),
                addr(0, "input_norm"), addr(0, "input_norm"),
                seq_len, h, c10::Half(1.0f), ValuOpType::ADD, /*is_bopa=*/false);
        }

        consume_causal_route(
            FmbRouteFamily::LINEAR, CAUSAL_DECODER_KV_FIRST_Q_SITE,
            static_cast<int64_t>(CausalDecoderLinearRoute::AUTO_TILE));
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.q_w, addr(0, "q_kv"),
            seq_len, nq * hd, h, /*partition=*/1, /*num_cores=*/tp,
            qkv_fused_bias_addr(layer_idx, "q_bias"), lw.q_ws,
            nvfp4_scale_addr("nvfp4_q_ts"), (uint16_t)layer_idx,
            /*force_acc32=*/linear_acc32_);
        emit_nvfp4_qkv_bias(
            layer_idx, "q_bias", "q_kv", seq_len, nq * hd / tp);
        consume_causal_route(
            FmbRouteFamily::LINEAR, CAUSAL_DECODER_KV_FIRST_K_SITE,
            static_cast<int64_t>(CausalDecoderLinearRoute::AUTO_TILE));
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.k_w, addr(0, "k"),
            seq_len, nkv * hd, h, /*partition=*/1, /*num_cores=*/tp,
            qkv_fused_bias_addr(layer_idx, "k_bias"), lw.k_ws,
            nvfp4_scale_addr("nvfp4_k_ts"), (uint16_t)layer_idx,
            /*force_acc32=*/linear_acc32_);
        emit_nvfp4_qkv_bias(
            layer_idx, "k_bias", "k", seq_len, nkv * hd / tp);
        consume_causal_route(
            FmbRouteFamily::LINEAR, CAUSAL_DECODER_KV_FIRST_V_SITE,
            static_cast<int64_t>(CausalDecoderLinearRoute::AUTO_TILE));
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.v_w, addr(0, "v"),
            seq_len, nkv * hd, h, /*partition=*/1, /*num_cores=*/tp,
            qkv_fused_bias_addr(layer_idx, "v_bias"), lw.v_ws,
            nvfp4_scale_addr("nvfp4_v_ts"), (uint16_t)layer_idx,
            /*force_acc32=*/linear_acc32_);
        emit_nvfp4_qkv_bias(
            layer_idx, "v_bias", "v", seq_len, nkv * hd / tp);

        // Q-split: Phase-1 Q lives in the KvInsert-scope "q_kv" slot. The QK-norm
        // intermediate reuses "input_norm" as scratch (free after the q/k/v
        // linears above consumed the LN output, and before the k-norm below
        // overwrites it in sequence) — this avoids a Compute-scope "output"
        // write inside the KvInsert subgraph. local_q_heads*hd <= h, so the
        // LayerWide input_norm slot is always large enough.
        int64_t local_q_heads = nq / tp;
        int64_t local_kv_dim  = nkv * hd / tp;
        bool has_qk_head_norm = has_qk_norm_;
        // RoPE 经 emit_qk_rope 单一真相源:自动含 partial_mrope 叶,
        // 与 build_layer_subgraph 同枝 → KV_FIRST 不再漏抄 RoPE 分支。Q-split:Phase-1 Q
        // 在 KvInsert-scope "q_kv";QK-norm 中间值复用 "input_norm" 作 scratch。
        if (has_qk_head_norm) {
            int64_t local_kv_heads = nkv / tp;
            // Q: q_kv → input_norm (norm) → q_kv (RoPE)
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "q_kv"), addr(0, "input_norm"),
                norm_q_addr(layer_idx, 0),
                seq_len * local_q_heads, hd, eps_, RpuRmsNormSpmRoute::BASE, attn_tp());
            emit_qk_rope("input_norm", "q_kv", local_q_heads, hd, seq_len, cos_sin_start, tp);
            // K: k → input_norm (norm) → k (RoPE)
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "k"), addr(0, "input_norm"),
                norm_k_addr(layer_idx, 0),
                seq_len * local_kv_heads, hd, eps_, RpuRmsNormSpmRoute::BASE, attn_tp());
            emit_qk_rope("input_norm", "k", local_kv_heads, hd, seq_len, cos_sin_start, tp);
        } else {
            emit_qk_rope("q_kv", "q_kv", local_q_heads, hd, seq_len, cos_sin_start, tp);
            emit_qk_rope("k", "k", 1, local_kv_dim, seq_len, cos_sin_start, tp);
        }

        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        const KvInsertSegmentPlan kv_plan = consume_causal_kv_plan(
            CAUSAL_DECODER_KV_FIRST_KV_SITE, chunk.idx,
            ctx().position + chunk.offset, seq_len);
        rpu_launch_insert_kvcache_spm_unified_with_plan(
            k_cache, v_cache, addr_offset("k").value,
            addr_offset("v").value,
            nkv, hd, tp,
            /*k_cache_batch_offset_elems=*/0,
            /*v_cache_batch_offset_elems=*/0,
            /*spm_rows=*/0, kv_plan);

        TORCH_CHECK(
            q_ddr_slots_.count({seq_len_, (num_q_heads() / attn_tp()) * head_dim()}) == 1,
            "CausalDecoderModel: Q staging slot not allocated in KV_FIRST mode");
        const at::Tensor& q_slot = q_ddr_slot();
        int64_t q_local_elems = seq_len * local_q_heads * hd;
        c10::Half* q_ddr_base = q_slot.data_ptr<c10::Half>();
        int64_t q_row_stride = local_q_heads * hd;
        int64_t q_elem_offset = chunk.offset * q_row_stride;
        int64_t q_core_stride_bytes = q_slot.size(1) * q_row_stride * DWIDTH;
        rpu_launch_spm_scatter_ddr_dma(
            addr(0, "q_kv"), q_ddr_base + q_elem_offset,
            q_local_elems, q_core_stride_bytes,
            /*num_cores=*/tp);
    }

    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {
        consume_qwen3vl_4b_w8_prefill_schedule(layer_idx, chunk);
        consume_qwen3vl_ordinary_prefill_schedule(layer_idx, chunk);
        if (layer_idx == 0 && chunk.idx == 0) {
            const int64_t topology_selector = static_cast<int64_t>(
                CausalDecoderGraphScheduleRoute::TOPOLOGY);
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                CAUSAL_DECODER_POOLER_TOPOLOGY_SITE, topology_selector,
                /*resolved_flags=*/0,
                {qwen3vl_pooler_z1_dispatch_ ? 1 : 0,
                 qwen3vl_multiview_text_composite_dispatch_ ? 1 : 0});
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                CAUSAL_DECODER_ADARMS_TOPOLOGY_SITE, topology_selector,
                /*resolved_flags=*/0,
                {adarms_ ? 1 : 0, adarms_mutable_ ? 1 : 0,
                 adarms_unroll_ ? 1 : 0,
                 adarms_schedule_select_ ? 1 : 0,
                 adarms_fused_bcast_enabled_ ? 1 : 0});
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                CAUSAL_DECODER_QK_NORM_TOPOLOGY_SITE, topology_selector,
                /*resolved_flags=*/0, {has_qk_norm_ ? 1 : 0});
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                CAUSAL_DECODER_QKV_BIAS_TOPOLOGY_SITE, topology_selector,
                /*resolved_flags=*/0, {has_qkv_bias_ ? 1 : 0});
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                CAUSAL_DECODER_NVFP4_TOPOLOGY_SITE, topology_selector,
                /*resolved_flags=*/0, {nvfp4_ ? 1 : 0});
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                CAUSAL_DECODER_DEEPSTACK_TOPOLOGY_SITE, topology_selector,
                /*resolved_flags=*/0,
                causal_deepstack_topology_arguments(deepstack_lang_layers_));
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                CAUSAL_DECODER_LM_HEAD_TOPOLOGY_SITE, topology_selector,
                /*resolved_flags=*/0,
                {fuse_lm_head_ ? 1 : 0, vocab_size_});
            if (qwen3vl_2b_w8_profile_) {
                ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
                    QWEN3VL_DELIVERY_TOPOLOGY_SITE, 1, lm_head_exact_candidates_,
                    delivery_topology_arguments());
            }
        }

        const auto& lw = layer_weights_[layer_idx];
        int64_t seq_len = chunk.len;
        // HALO WS-2: RoPE table index. Legacy (rope_position_base_ < 0) and the
        // M-RoPE keepalive both index from ctx().position; the 1D override
        // decouples the RoPE position base from the cache-insert offset (KV-insert
        // below stays on ctx().position). chunk.offset advances within the new
        // block, which is correct because HALO text positions are contiguous.
        const int64_t rope_base =
            (!has_mrope_ && rope_position_base_ >= 0) ? rope_position_base_ : ctx().position;
        int64_t cos_sin_start = rope_base + chunk.offset;
        int64_t L = num_layers();
        bool is_last_layer = (layer_idx == L - 1);
        // A one-row tail of a multi-token prefill is still prefill.  Decode-only
        // work must follow the whole request length, not this chunk's length.
        const bool decode_forward = ctx().seq_len == 1;
        int64_t h = hidden_size();
        int64_t nq = num_q_heads();
        int64_t nkv = num_kv_heads();
        int64_t hd = head_dim();
        // Attention tensor-parallel factor: min(NUM_CORES, nkv). NUM_CORES for
        // nkv>=8 (Qwen3/Llama no-op), 2 for Wall-OSS nkv=2. MLP stays NUM_CORES.
        int tp = attn_tp();
        const Z1LayerInputSelection z1_input =
            select_z1_layer_input(layer_idx, chunk);
        const uint32_t input_residual = z1_input.addr;

        const bool emit_prefill_last_lm_head =
            fuse_prefill_last_lm_head_ && ctx().is_causal &&
            !decode_forward && chunk.offset + chunk.len == ctx().seq_len;
        const bool emit_fused_lm_head = is_last_layer && fuse_lm_head_ &&
            (decode_forward || emit_prefill_last_lm_head);
        if (emit_fused_lm_head) {
            TORCH_CHECK(lm_head_logits_out_.defined()
                            && lm_head_logits_dst_base_ != 0,
                        "CausalDecoderModel: fused lm_head output was "
                        "not allocated for the whole forward");
        }

        // Batch decode packs independent sequence rows into GEMM's M dimension, so
        // row-parallel operations share launches and weights. KV insertion, SDPA and
        // fused-lm_head output DMA remain per-sequence because each owns a cache slot.
        //
        // Optional DDR flushing scales with total cache bytes. Boundary pointer
        // deduplication removes duplicate addresses, but does not reduce a cache's extent.
        //
        // Batched decode is admitted only at seq_len == 1 with a shared position.
        // Presenting [bs,heads,hd] as one row with bs*heads keeps RoPE on that position
        // for every sequence without changing the underlying memory layout.
        const int64_t bs = ctx().batch_size;
        const int64_t rows = bs * seq_len;

        if (bs > 1) {
            TORCH_CHECK(seq_len == 1,
                        "CausalDecoderModel: batch decode requires chunk.len == 1, got ",
                        seq_len);
            TORCH_CHECK(ctx().is_causal && !ctx().attention_mask.has_value(),
                        "CausalDecoderModel: batch decode requires the causal "
                        "no-explicit-mask path");
        }

        if (!ctx().is_causal && !ctx().attention_mask.has_value()) {
            // KV_FIRST Phase-2 needs local_q_heads here; the sequential path
            // declares its own copy later at the Phase-3 QK-norm site (this
            // branch always returns, so the two declarations never collide).
            int64_t local_q_heads = nq / tp;
            if (qwen3vl_multiview_text_composite_dispatch_ &&
                layer_idx == 0) {
                emit_qwen3vl_multiview_text_complement_ingress(
                    layer_idx, chunk, input_residual);
            } else if (!ctx().input_in_spm && !z1_input.external &&
                       !(pre_layers_residual1_ready_ && layer_idx == 0)) {
                emit_layer_input_dma(layer_idx, chunk);
            }

            int64_t q_local_elems = seq_len * local_q_heads * hd;
            const at::Tensor& q_slot = q_ddr_slot();
            c10::Half* q_ddr_base = q_slot.data_ptr<c10::Half>();
            int64_t q_row_stride = local_q_heads * hd;
            int64_t q_elem_offset = chunk.offset * q_row_stride;
            int64_t q_core_stride = q_slot.size(1) * q_row_stride * DWIDTH;
            rpu_launch_ddr_scatter_spm_dma(
                q_ddr_base + q_elem_offset,
                /*elements_per_core=*/q_local_elems,
                /*core_stride_bytes=*/q_core_stride,
                addr(0, "q"),
                /*num_cores=*/tp);

            int64_t total_kv_seq_len = ctx().position + seq_len_;
            auto& k_cache = (*ctx().k_caches)[layer_idx];
            auto& v_cache = (*ctx().v_caches)[layer_idx];
            ctx().consume_physical_route(
                FmbRouteFamily::ATTENTION,
                CAUSAL_DECODER_KV_FIRST_ATTN_SITE,
                static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                /*resolved_flags=*/0, {seq_len_}, chunk.idx);
            rpu_launch_sdpa_spm_dispatch(
                sdpa_kernel_,
                k_cache, v_cache, 0 /*MASK_NONE*/, c10::nullopt,
                addr_offset("q").value,
                addr_offset("output").value,
                addr_offset("sdpa_tmp").value, 0,
                seq_len, nq, nkv, hd, total_kv_seq_len, tp, NUM_CORES);

            consume_causal_route(
                FmbRouteFamily::ALL_REDUCE,
                CAUSAL_DECODER_KV_FIRST_PREPARE_SITE,
                static_cast<int64_t>(
                    CausalDecoderAllReduceRoute::PREPARE_RING_INPUT));
            rpu_prepare_ring_all_reduce_input(
                addr(0, "oproj"), seq_len, h, tp, num_cores());
            consume_causal_route(
                FmbRouteFamily::LINEAR, CAUSAL_DECODER_KV_FIRST_O_SITE,
                static_cast<int64_t>(CausalDecoderLinearRoute::AUTO_TILE));
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "output"), lw.o_w, addr(0, "oproj"),
                seq_len, h, nq * hd, /*partition=*/0, /*num_cores=*/tp,
                /*bias_spm_addr=*/0, lw.o_ws,
                nvfp4_scale_addr("nvfp4_o_ts"), (uint16_t)layer_idx,
                /*force_acc32=*/linear_acc32_);
            consume_causal_route(
                FmbRouteFamily::ALL_REDUCE,
                CAUSAL_DECODER_KV_FIRST_REDUCE_SITE,
                causal_ring_route(seq_len, h), chunk.idx);
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), input_residual,
                addr(0, "residual2"), seq_len, h, tp, num_cores());
            emit_adarms_mut_refresh_post(layer_idx);  // replay-safe AdaRMS: refresh scale+shift SPM
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "residual2"), addr(0, "residual1"),
                norm_post_addr(layer_idx, 0),   // adarms_: scale rides here
                seq_len, h, eps_, RpuRmsNormSpmRoute::BASE, num_cores());
            if (adarms_) {  // AdaRMS FiLM: residual1 (normed) += post_shift
                rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                    layer_addr(layer_idx, 0, "post_shift"),
                    addr(0, "residual1"), addr(0, "residual1"),
                    seq_len, h, c10::Half(1.0f), ValuOpType::ADD, /*is_bopa=*/false);
            }
            if (decode_gate_up_fusion_) {
                emit_qwen3vl_2b_mlp_pipeline(
                    lw, seq_len, h, nq, nkv, hd, tp, chunk.idx);
            } else {
                emit_mlp_pipeline(
                    lw.gate_w, lw.up_w, lw.down_w, seq_len,
                    use_silu_ ? ActivationKind::SILU : ActivationKind::NONE,
                    lw.gate_ws, lw.up_ws, lw.down_ws,
                    nvfp4_scale_addr("nvfp4_gate_ts"),
                    nvfp4_scale_addr("nvfp4_up_ts"),
                    nvfp4_scale_addr("nvfp4_down_ts"),
                    (uint16_t)layer_idx,
                    /*down_out_bf16=*/false,
                    /*residual_is_bf16=*/false,
                    /*acc32=*/linear_acc32_);
            }

            if (qwen3vl_multiview_text_composite_dispatch_) {
                emit_qwen3vl_multiview_text_retained_deepstack_add(
                    layer_idx, chunk, addr(0, "residual1"));
            } else if (qwen3vl_pooler_z1_dispatch_ &&
                qwen3vl_pooler_z1_retained_deepstack_count_ > 0) {
                emit_qwen3vl_pooler_z1_retained_deepstack_add(
                    layer_idx, chunk, addr(0, "residual1"));
            } else if (!deepstack_lang_layers_.empty()) {
                auto it = std::find(deepstack_lang_layers_.begin(),
                                    deepstack_lang_layers_.end(),
                                    static_cast<int64_t>(layer_idx));
                if (it != deepstack_lang_layers_.end()) {
                    const int idx_in_list = static_cast<int>(
                        std::distance(deepstack_lang_layers_.begin(), it));
                    consume_causal_route(
                        FmbRouteFamily::MUTABLE_DMA,
                        CAUSAL_DECODER_KV_FIRST_DEEPSTACK_SITE,
                        static_cast<int64_t>(
                            CausalDecoderMutableDmaRoute::
                                DDR_BROADCAST_TO_SPM));
                const uint32_t deepstack_spm = qwen3vl_2b_w8_profile_
                        ? addr(0, "down")
                        : addr(0, "deepstack_scratch");
                    rpu_launch_ddr_broadcast_spm_dma_mutable(
                        &deepstack_dense_src_base_[idx_in_list],
                        /*src_offset_bytes=*/chunk.offset * h * DWIDTH,
                        /*num_elements=*/chunk.len * h,
                        deepstack_spm,
                        num_cores());
                    rpu_launch_eltwise_binary_spm_kernel(
                        addr(0, "residual1"),
                        deepstack_spm,
                        addr(0, "residual1"),
                        chunk.len * h,
                        ValuOpType::ADD,
                        c10::Half(1.0f),
                        num_cores());
                }
            }

            if (is_last_layer) {
                rpu_launch_rmsnorm_spm_kernel(
                    addr(0, "residual1"), addr(0, "residual1"),
                    norm_final_addr(0),
                    seq_len, h, eps_, RpuRmsNormSpmRoute::BASE, num_cores());
            }

            if (is_last_layer && fuse_lm_head_ && decode_forward) {
                // The shared fused-lm-head path uses the generated auto-tile
                // family for FP16 and W8A16, including decode at M=1.
                consume_causal_route(
                    FmbRouteFamily::LINEAR,
                    CAUSAL_DECODER_KV_FIRST_LM_HEAD_SITE,
                    static_cast<int64_t>(
                        CausalDecoderLinearRoute::AUTO_TILE));
                rpu_launch_linear_spm_to_spm_acc16_kernel(
                    addr(0, "residual1"), lm_head_w_, addr(0, "lm_head_out"),
                    /*M=*/1, /*N=*/vocab_size_, /*K=*/h,
                    /*partition=*/1, /*num_cores=*/lm_head_tp(),
                    /*bias_spm_addr=*/0,
                    lm_head_w_scale_,
                    /*nvfp4_tensor_scale_spm_addr=*/0,
                    /*nvfp4_layer_id=*/0,
                    /*force_acc32=*/linear_acc32_);
                const int64_t local_n = vocab_size_ / lm_head_tp();
                consume_causal_route(
                    FmbRouteFamily::MUTABLE_DMA,
                    CAUSAL_DECODER_KV_FIRST_LM_HEAD_DMA_SITE,
                    static_cast<int64_t>(
                        CausalDecoderMutableDmaRoute::SPM_SCATTER_TO_DDR));
                rpu_launch_spm_scatter_ddr_dma_mutable(
                    addr(0, "lm_head_out"),
                    &lm_head_logits_dst_base_,
                    /*dst_offset_bytes=*/0,
                    /*elements_per_core=*/local_n,
                    /*core_stride_bytes=*/local_n * DWIDTH,
                    lm_head_tp());
            } else if (!ctx().output_to_spm) {
                emit_layer_output_dma(layer_idx, chunk);
            }
            return;
        }

        // ─────────────────────────────────────────────────────────────────────
        // Input DMA (if input is not already in SPM) — framework-internal detail:
        //   layer 0 reads ctx().hidden_states (request tensor);
        //   layer > 0 reads the DDR chain buffer (framework owns ping-pong);
        //   ctx().input_in_spm == true skips DMA (intra-group SPM-resident).
        // ─────────────────────────────────────────────────────────────────────
        if (qwen3vl_multiview_text_composite_dispatch_ && layer_idx == 0) {
            emit_qwen3vl_multiview_text_complement_ingress(
                layer_idx, chunk, input_residual);
        } else if (!ctx().input_in_spm && !z1_input.external &&
                   !(pre_layers_residual1_ready_ && layer_idx == 0)) {
            emit_layer_input_dma(layer_idx, chunk);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Phase 1: Input RMSNorm
        //
        // Static Qwen uses the packed norm slab; dynamic/bias variants retain
        // the per-layer slot. Both are preloaded once per BUILD.
        // ─────────────────────────────────────────────────────────────────────
        {
            emit_adarms_mut_refresh_input(layer_idx);  // replay-safe AdaRMS: refresh scale+shift SPM
            {
                rpu_launch_rmsnorm_spm_kernel(
                    input_residual, addr(0, "input_norm"),
                    norm_input_addr(layer_idx, 0),   // adarms_: scale rides here
                    rows, h, eps_,
                    consume_decoder_rmsnorm_route(rows, h, chunk.idx), num_cores());
            }
            if (adarms_) {  // AdaRMS FiLM: input_norm += input_shift (LingBot-VLA expert)
                rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                    layer_addr(layer_idx, 0, "input_shift"),
                    addr(0, "input_norm"), addr(0, "input_norm"),
                    rows, h, c10::Half(1.0f), ValuOpType::ADD, /*is_bopa=*/false);
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Phase 2: QKV Linear
        // ─────────────────────────────────────────────────────────────────────
        // QKV col-partition over attn_tp() cores. Optional bias (Qwen2.5/Wall-OSS)
        // is fused for FP16/W8/W4. NVFP4 uses the same persistent SPM bias slots
        // but adds them separately to avoid the fused-bias replay hazard.
        // Batch decode: M = rows, so the q/k/v weights stream from DDR once for
        // all `bs` sequences instead of once per sequence.
        const bool direct_prefill_qkv =
            bs == 1 && seq_len == 192 &&
            prefill_packed_qkv_planar_direct_ && lw.qkv_w.defined();
        const bool fused_decode_qkv =
            bs == 1 && seq_len == 1 && decode_qkv_fusion_ &&
            lw.qkv_w.defined();
        if (qwen3vl_4b_packed_decode_qkv(
                seq_len_, ctx().is_causal, ctx().attention_mask.has_value(), bs)) {
            TORCH_CHECK(rows == 1 && lw.qkv_w.defined() && lw.qkv_ws.defined(),
                        "4B packed QKV escaped its cold decode contract");
            ctx().consume_physical_route(FmbRouteFamily::LINEAR,
                CAUSAL_DECODER_4B_W8_PACKED_QKV_SITE,
                static_cast<int64_t>(CausalDecoderLinearRoute::GEMV), 0,
                {1, 6144, 2560, 8, 512, 128, 128});
            const uint32_t q_addr = addr(0, "q");
            TORCH_CHECK(addr(0, "k") == q_addr + 1024 &&
                            addr(0, "v") == q_addr + 1280,
                        "4B packed QKV requires contiguous 1024/256/256 B SPM slices");
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "input_norm"), lw.qkv_w, q_addr,
                1, 6144, 2560, /*partition=*/1, /*num_cores=*/8,
                /*bias_spm_addr=*/0, lw.qkv_ws,
                /*nvfp4_tensor_scale_spm_addr=*/0, /*nvfp4_layer_id=*/0,
                /*force_acc32=*/false, /*prefer_gemv=*/true);
        } else if (direct_prefill_qkv || fused_decode_qkv) {
            ctx().consume_physical_route(FmbRouteFamily::LINEAR,
                QWEN3VL_DELIVERY_QKV_SITE, direct_prefill_qkv ? 5 : 1, 0,
                {rows, 4096, 2048, 1, 8}, chunk.idx);
            const uint32_t q_addr = addr(0, "q");
            const uint32_t k_addr = addr(0, "k");
            const uint32_t v_addr = addr(0, "v");
            const uint32_t q_local_bytes = static_cast<uint32_t>(
                rows * (nq * hd / tp) * DWIDTH);
            const uint32_t kv_local_bytes = static_cast<uint32_t>(
                rows * (nkv * hd / tp) * DWIDTH);
            TORCH_CHECK(
                k_addr == q_addr + q_local_bytes &&
                    v_addr == k_addr + kv_local_bytes,
                "Qwen3-VL fused QKV requires contiguous Q/K/V SPM buffers, "
                "got q=", q_addr, " k=", k_addr, " v=", v_addr);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "input_norm"), lw.qkv_w, q_addr,
                rows, (nq + 2 * nkv) * hd, h,
                /*partition=*/1, /*num_cores=*/tp,
                /*bias_spm_addr=*/0, lw.qkv_ws,
                /*nvfp4_tensor_scale_spm_addr=*/0,
                /*nvfp4_layer_id=*/0, /*force_acc32=*/false,
                /*prefer_gemv=*/false, /*qkv_planar_direct=*/direct_prefill_qkv);
        } else {
            consume_causal_route(
                FmbRouteFamily::LINEAR, CAUSAL_DECODER_Q_SITE,
                static_cast<int64_t>(decoder_projection_route(seq_len_)));
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "input_norm"), lw.q_w, addr(0, "q"),
                rows, nq * hd, h, /*partition=*/1, /*num_cores=*/tp,
                qkv_fused_bias_addr(layer_idx, "q_bias"), lw.q_ws,
                nvfp4_scale_addr("nvfp4_q_ts"), (uint16_t)layer_idx,
                /*force_acc32=*/linear_acc32_,
                /*prefer_gemv=*/qwen3vl_decode_gemv(seq_len_), /*qkv_planar_direct=*/false);
            emit_nvfp4_qkv_bias(
                layer_idx, "q_bias", "q", rows, nq * hd / tp);
            consume_causal_route(
                FmbRouteFamily::LINEAR, CAUSAL_DECODER_K_SITE,
                static_cast<int64_t>(decoder_projection_route(seq_len_)));
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "input_norm"), lw.k_w, addr(0, "k"),
                rows, nkv * hd, h, /*partition=*/1, /*num_cores=*/tp,
                qkv_fused_bias_addr(layer_idx, "k_bias"), lw.k_ws,
                nvfp4_scale_addr("nvfp4_k_ts"), (uint16_t)layer_idx,
                /*force_acc32=*/linear_acc32_,
                /*prefer_gemv=*/qwen3vl_decode_gemv(seq_len_), /*qkv_planar_direct=*/false);
            emit_nvfp4_qkv_bias(
                layer_idx, "k_bias", "k", rows, nkv * hd / tp);
            consume_causal_route(
                FmbRouteFamily::LINEAR, CAUSAL_DECODER_V_SITE,
                static_cast<int64_t>(decoder_projection_route(seq_len_)));
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "input_norm"), lw.v_w, addr(0, "v"),
                rows, nkv * hd, h, /*partition=*/1, /*num_cores=*/tp,
                qkv_fused_bias_addr(layer_idx, "v_bias"), lw.v_ws,
                nvfp4_scale_addr("nvfp4_v_ts"), (uint16_t)layer_idx,
                /*force_acc32=*/linear_acc32_,
                /*prefer_gemv=*/qwen3vl_decode_gemv(seq_len_), /*qkv_planar_direct=*/false);
            emit_nvfp4_qkv_bias(
                layer_idx, "v_bias", "v", rows, nkv * hd / tp);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Phase 3: QK RMSNorm + RoPE
        // ─────────────────────────────────────────────────────────────────────
        int64_t local_q_heads = nq / tp;
        int64_t local_kv_dim  = nkv * hd / tp;
        // codex HIGH-1+HIGH-2 (Phase 06.1 / D-6-17): branch on the explicit
        // has_qk_norm_ field set by set_weights, NOT the (nkv >= NUM_CORES)
        // heuristic. The old heuristic incorrectly fired for Llama-3.2-1B
        // (nkv=8 == NUM_CORES) and forced QK-RMSNorm on Llama tensors that
        // were never gathered (Llama's LlamaAttention has no q_norm/k_norm).
        bool has_qk_head_norm = has_qk_norm_;

        // RoPE 表指针 / strobe masks / partial-mrope keepalive 现由 emit_qk_rope 内部从成员
        // 按 has_mrope_/partial_mrope_active_ 派生(共享来源),此处不再展开。

        // Phase 06.1 / R4-01 Path B / REN-05: op-boundary cos/sin parity dump.
        // Captures the EXACT cos/sin slice the rope kernel is about to consume,
        // plus rope_meta tuple proving the kernel was invoked with the expected
        // (ctx().position, chunk.offset, cos_sin_start, seq_len, layer_idx, chunk_idx).
        // Gated behind get_debug_export(); zero default-path codegen impact.
        if (get_debug_export()) {
            // partial-M-RoPE 的 rope kernel 实际消费 cos_il/sin_il
            // keepalive 表(交错格式,dim=hd/2),非 cos_/sin_。debug dump 须跟随活跃叶
            // (emit_qk_rope 同款 has_mrope_×partial_mrope_active_ 派生),否则会误报正在喂给
            // kernel 的 RoPE 表。HALO(has_mrope_=false)走 else → cos_/sin_,与修前一致。
            const bool partial_dump = has_mrope_ && partial_mrope_active_;
            const at::Tensor& cos_src = partial_dump ? cos_il_keepalive_ : cos_;
            const at::Tensor& sin_src = partial_dump ? sin_il_keepalive_ : sin_;
            int64_t cos_dim = cos_src.size(-1);  // head_dim or head_dim/2 (kernel format)
            auto cos_slice = cos_src.narrow(0, cos_sin_start, seq_len).contiguous().clone();
            auto sin_slice = sin_src.narrow(0, cos_sin_start, seq_len).contiguous().clone();
            int64_t chunk_idx = chunk.idx;
            std::string cos_key = "L" + std::to_string(layer_idx) + "_cos_at_kernel_input_chunk" + std::to_string(chunk_idx);
            std::string sin_key = "L" + std::to_string(layer_idx) + "_sin_at_kernel_input_chunk" + std::to_string(chunk_idx);
            std::string meta_key = "L" + std::to_string(layer_idx) + "_rope_meta_chunk" + std::to_string(chunk_idx);
            g_debug_tensors[cos_key] = cos_slice;
            g_debug_tensors[sin_key] = sin_slice;
            auto meta = at::tensor({(double)ctx().position, (double)chunk.offset,
                                    (double)cos_sin_start, (double)seq_len,
                                    (double)layer_idx, (double)chunk_idx,
                                    (double)cos_dim, (double)(partial_dump ? 1 : 0)},
                                   at::TensorOptions().dtype(at::kDouble));
            g_debug_tensors[meta_key] = meta;
        }

        // RoPE 经 emit_qk_rope 单一真相源:has_mrope_ ×
        // partial_mrope_active_ → 1D-RoPE / M-RoPE / partial-M-RoPE 三变体,与
        // emit_kv_first_body 同枝(共享 cos/sin 表 + cos_sin_start = ctx().position +
        // chunk.offset 语义)。新增 RoPE 模式只改 helper 一处,两路自动同步。
        const bool qwen3vl4b_fused_qk_kv = qwen3vl_4b_decode_qk_kv_fused_ &&
            ctx().attention_policy == AttentionExecutionPolicy::DDR_KV &&
            ctx().stage_plan.chunk_mode == ChunkMode::SEQUENTIAL &&
            ctx().stage_plan.compute.chunks.size() == 1 &&
            qwen3vl_4b_decode_qk_norm_mrope(seq_len_, chunk.len, ctx().is_causal,
                ctx().attention_mask.has_value(), bs, partial_mrope_active_);
        if (qwen3vl4b_fused_qk_kv) {
            emit_qwen3vl_4b_decode_qk_norm_mrope_kv_insert(layer_idx, chunk, cos_sin_start);
        } else if (qwen3vl_4b_decode_qk_fused_ &&
            ctx().stage_plan.chunk_mode == ChunkMode::SEQUENTIAL &&
            ctx().stage_plan.compute.chunks.size() == 1 &&
            qwen3vl_4b_decode_qk_norm_mrope(seq_len_, chunk.len, ctx().is_causal,
                ctx().attention_mask.has_value(), bs, partial_mrope_active_)) {
            emit_qwen3vl_4b_decode_qk_norm_mrope(layer_idx, chunk, cos_sin_start);
        } else if (has_qk_head_norm) {
            // Q/K norm weights are BUILD-preloaded into either the packed slab
            // or the fallback per-layer slots (see declare_buffers).
            int64_t local_kv_heads = nkv / tp;

            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "q"), addr(0, "output"),
                norm_q_addr(layer_idx, 0),
                rows * local_q_heads, hd, eps_,
                qwen3vl_2b_w8_profile_ ? consume_delivery_rmsnorm_route(rows * local_q_heads, hd, chunk.idx, 1)
                    : consume_ordinary_rmsnorm_route(rows * local_q_heads, hd, chunk.idx, 1),
                attn_tp());
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "k"), addr(0, "input_norm"),
                norm_k_addr(layer_idx, 0),
                rows * local_kv_heads, hd, eps_,
                qwen3vl_2b_w8_profile_ ? consume_delivery_rmsnorm_route(rows * local_kv_heads, hd, chunk.idx, 2)
                    : consume_ordinary_rmsnorm_route(rows * local_kv_heads, hd, chunk.idx, 2),
                attn_tp());
            if (bs == 1) {
                emit_qk_rope_pair(
                    "output", "q", "input_norm", "k",
                    local_q_heads, local_kv_heads, hd,
                    seq_len, cos_sin_start, tp, chunk.idx);
            } else {
                emit_qk_rope(
                    "output", "q", bs * local_q_heads, hd,
                    seq_len, cos_sin_start, tp);
                emit_qk_rope(
                    "input_norm", "k", bs * local_kv_heads, hd,
                    seq_len, cos_sin_start, tp);
            }
        } else {
            // No QK head norm, in-place RoPE
            {
                emit_qk_rope("q", "q", bs * local_q_heads, hd, seq_len,
                             cos_sin_start, tp);
                emit_qk_rope("k", "k", bs, local_kv_dim, seq_len,
                             cos_sin_start, tp);
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Phase 4: KV cache insert + SDPA + O_proj
        //
        // Pitfall 3 structural fix: SDPA and KV-insert need SPM OFFSETS
        // (not absolute addresses). Use addr_offset("name").value — the typed
        // SpmOffset can't be confused with the absolute addr() return value.
        // ─────────────────────────────────────────────────────────────────────
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];

        // Batch decode strides. SPM buffer sizes in declare_buffers are byte
        // counts, so addr_offset() values are byte offsets: sequence b's rows
        // start one per-sequence block in. The caches are [batch, ...]
        // contiguous, so slot b starts numel/batch elements in.
        const int64_t k_spm_stride = seq_len * local_kv_dim * DWIDTH;
        const int64_t q_spm_stride = seq_len * local_q_heads * hd * DWIDTH;
        // Slot base: batched decode owns slots 0..bs-1 (cache_batch_slot_ == 0);
        // batched prefill runs bs == 1 against slot cache_batch_slot_.
        const int64_t slot0 = cache_batch_slot_;
        const bool multi_slot = (bs > 1 || slot0 > 0);
        const int64_t k_cache_batch_stride =
            multi_slot ? k_cache.numel() / k_cache.size(0) : 0;
        const int64_t v_cache_batch_stride =
            multi_slot ? v_cache.numel() / v_cache.size(0) : 0;
        if (multi_slot) {
            TORCH_CHECK(k_cache.size(0) >= slot0 + bs && v_cache.size(0) >= slot0 + bs,
                        "CausalDecoderModel: batch decode needs KV caches with "
                        "batch dim >= ", slot0 + bs, ", got k=", k_cache.size(0),
                        " v=", v_cache.size(0));
            // SDPA takes ONE cache offset for both K and V. K and V are the same
            // size in RPUCache (sKeyVx == sValVx; the last two swizzle dims are
            // transposed but equal in product), so one stride is correct — assert
            // it rather than assume it.
            TORCH_CHECK(k_cache_batch_stride == v_cache_batch_stride,
                        "CausalDecoderModel: batch decode assumes equal K/V batch "
                        "strides, got k=", k_cache_batch_stride,
                        " v=", v_cache_batch_stride);
        }

        // Each sequence owns cache slot b, so KV-insert is per-sequence. All
        // `bs` sequences sit at the SAME position (equal-length batch), so one
        // frozen typed topology is reused while only cache/SPM bases move.
        if (qwen3vl_2b_w8_profile_) {
            TORCH_CHECK(bs == 1 && slot0 == 0 && nkv == 8 && hd == 128 && tp == 8,
                        "fixed delivery paired KV geometry drift");
            const int64_t insert_position = ctx().position + chunk.offset;
            const bool decode = seq_len == 1;
            (void)rpu_resolve_kvinsert_segment_plan(insert_position, seq_len, seq_len,
                tp, nkv, hd, KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16,
                decode ? KvInsertRoute::V2 : KvInsertRoute::ALIGNED_V16);
            if (decode) {
                ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
                    QWEN3VL_DELIVERY_KV_DECODE_PAIR_SITE, 2, 0,
                    delivery_kv_arguments(seq_len, insert_position), chunk.idx);
                rpu_launch_insert_kv_cache_spm_unified_decode(k_cache, v_cache,
                    insert_position, addr_offset("k").value, addr_offset("v").value,
                    seq_len, nkv, hd, tp);
            } else {
                ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
                    QWEN3VL_DELIVERY_KV_PAIR_SITE, 1, 0,
                    delivery_kv_arguments(seq_len, insert_position), chunk.idx);
                rpu_launch_insert_kv_cache_spm_unified_v16(k_cache, v_cache,
                    insert_position, addr_offset("k").value, addr_offset("v").value,
                    seq_len, nkv, hd, tp);
            }
        } else if (!qwen3vl4b_fused_qk_kv) {
        const KvInsertSegmentPlan kv_plan = consume_causal_kv_plan(
            CAUSAL_DECODER_KV_INSERT_SITE, chunk.idx,
            ctx().position + chunk.offset, seq_len);
        for (int64_t b = 0; b < bs; ++b) {
            rpu_launch_insert_kvcache_spm_unified_with_plan(
                k_cache, v_cache,
                addr_offset("k").value + (uint32_t)(b * k_spm_stride),
                addr_offset("v").value + (uint32_t)(b * k_spm_stride),
                nkv, hd, tp,
                (slot0 + b) * k_cache_batch_stride,
                (slot0 + b) * v_cache_batch_stride,
                kv_insert_spm_rows_, kv_plan);
        }

        }

        int64_t kv_seq_len = chunk.kv_seq_len;

        // P4: explicit 2D mask path vs original LTM/NONE. When base reports an
        // explicit mask (is_causal=false + mask), use the MASK_2D type prepared
        // in dynamic_config and upload the mask to SPM right before SDPA. The
        // The generic sdpa_mask Temp slot is phase-aliased, so it is uploaded
        // every layer. An opt-in subclass may extend the slot across the body
        // and upload it once at body 0/layer 0. Single chunk is asserted in
        // dynamic_config, so seq_q=seq_len and seq_k=kv_seq_len (= P+H) are
        // constant across the (only) chunk.
        int mask_type;
        uint32_t mask_off = 0;
        if (ctx().attention_mask.has_value()) {
            mask_type = prepared_attn_mask_.mask_type;          // MASK_2D (4)
            mask_off  = addr_offset("sdpa_mask").value;
            emit_explicit_mask_spm(layer_idx, chunk, seq_len, kv_seq_len, tp, mask_off);
        } else {
            bool sdpa_causal = ctx().is_causal && (seq_len > 1);
            mask_type = sdpa_causal ? 1 : 0;                    // MASK_LTM / MASK_NONE
        }

        // SDPA runs on attn_tp() cores; virtual_num_cores=NUM_CORES governs the
        // KV-cache 7-D swizzle layout (matches QwenPI05 attn_tp path). For nkv>=8
        // (tp==NUM_CORES) this equals the previous (8, -1) default behavior.
        if (ctx().attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA) {
            TORCH_CHECK(
                bs == 1 && ctx().position == 0 && chunk.offset == 0 &&
                    seq_len == kv_seq_len && mask_type == 1 &&
                    tp == NUM_CORES,
                "CausalDecoderModel raw-SPM by-MHA escaped its exact short "
                "Qwen3 prefill contract");
            ctx().consume_physical_route(
                FmbRouteFamily::ATTENTION,
                CAUSAL_DECODER_RAW_SPM_ATTN_SITE,
                static_cast<int64_t>(
                    AttentionExecutionPolicy::SPM_KV_BY_MHA),
                /*resolved_flags=*/0, {seq_len_}, chunk.idx);
            rpu_launch_v_transpose_spm(
                addr(0, "v"), addr(0, "sdpa_tmp"),
                /*batch=*/1, kv_seq_len, nkv, hd, tp);
            rpu_launch_sdpa_by_mha_spm(
                addr(0, "q"), addr(0, "k"), addr(0, "sdpa_tmp"),
                addr(0, "output"), /*mask_spm=*/0, mask_type,
                1.0 / std::sqrt(static_cast<double>(hd)),
                /*batch=*/1, seq_len, kv_seq_len, nq, nkv, hd, tp);
        } else {
            // Batch decode: one launch per sequence, each reading only its own
            // cache slot. The launches safely reuse the one sdpa_tmp scratch.
            ctx().consume_physical_route(
                FmbRouteFamily::ATTENTION,
                CAUSAL_DECODER_MASKED_ATTENTION_SITE,
                static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                /*resolved_flags=*/0, {seq_len_}, chunk.idx);
            for (int64_t b = 0; b < bs; ++b) {
                rpu_launch_sdpa_spm_dispatch(
                    sdpa_kernel_,
                    k_cache, v_cache, mask_type, c10::nullopt,
                    addr_offset("q").value + (uint32_t)(b * q_spm_stride),
                    addr_offset("output").value + (uint32_t)(b * q_spm_stride),
                    addr_offset("sdpa_tmp").value, mask_off,
                    seq_len, nq, nkv, hd, kv_seq_len, tp, NUM_CORES,
                    (slot0 + b) * k_cache_batch_stride);
            }
        }

        // Layer 0 establishes a full residual2 with the existing chain. Later
        // layers consume/update the compact carry and normalize directly to
        // residual1 for the MLP.
        if (ctx().stage_plan.chunk_mode == ChunkMode::SEQUENTIAL &&
            ctx().stage_plan.compute.chunks.size() == 1 &&
            ctx().attention_policy == AttentionExecutionPolicy::DDR_KV &&
            qwen3vl_4b_decode_o_ring_norm(seq_len_, chunk.len, ctx().is_causal,
                ctx().attention_mask.has_value(), bs, partial_mrope_active_)) {
            emit_qwen3vl_4b_decode_o_ring_norm(lw, layer_idx, chunk, input_residual);
        } else {
            // O-proj row-partition over attn_tp() cores; reduce back to NUM_CORES.
            consume_causal_route(
                FmbRouteFamily::ALL_REDUCE, CAUSAL_DECODER_PREPARE_SITE,
                static_cast<int64_t>(
                    CausalDecoderAllReduceRoute::PREPARE_RING_INPUT));
            rpu_prepare_ring_all_reduce_input(
                addr(0, "oproj"), rows, h, tp, num_cores());
            const bool reuse_o_weight = consume_ordinary_o_route(
                CAUSAL_DECODER_O_SITE, rows, chunk.idx);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "output"), lw.o_w, addr(0, "oproj"),
                rows, h, nq * hd, /*partition=*/0, /*num_cores=*/tp,
                /*bias_spm_addr=*/0, lw.o_ws,
                nvfp4_scale_addr("nvfp4_o_ts"), (uint16_t)layer_idx,
                /*force_acc32=*/linear_acc32_,
                /*prefer_gemv=*/qwen3vl_decode_gemv(seq_len_), /*qkv_planar_direct=*/false,
                /*prefer_row_weight_reuse=*/reuse_o_weight);
            {
                consume_causal_route(
                    FmbRouteFamily::ALL_REDUCE,
                    CAUSAL_DECODER_REDUCE_SITE,
                    causal_ring_route(rows, h), chunk.idx);
                rpu_launch_all_reduce_sum_residual_kernel(
                    addr(0, "oproj"), input_residual,
                    addr(0, "residual2"), rows, h, tp, num_cores());
            }
            emit_adarms_mut_refresh_post(layer_idx);
            {
                rpu_launch_rmsnorm_spm_kernel(
                    addr(0, "residual2"), addr(0, "residual1"),
                    norm_post_addr(layer_idx, 0), rows, h, eps_,
                    consume_decoder_rmsnorm_route(rows, h, chunk.idx), num_cores());
            }
            if (adarms_) {
                rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                    layer_addr(layer_idx, 0, "post_shift"),
                    addr(0, "residual1"), addr(0, "residual1"),
                    rows, h, c10::Half(1.0f), ValuOpType::ADD,
                    /*is_bopa=*/false);
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Phase 7-8: MLP (gate, up, silu?, mul, down) + reduce + residual → residual1
        // ─────────────────────────────────────────────────────────────────────
        if (ctx().stage_plan.chunk_mode == ChunkMode::SEQUENTIAL &&
            qwen3vl_4b_prefill_swiglu(seq_len_, chunk.len, ctx().position,
                ctx().is_causal, ctx().attention_mask.has_value(), bs)) {
            emit_qwen3vl_4b_prefill_mlp(lw, chunk);
        } else if (ctx().stage_plan.chunk_mode == ChunkMode::SEQUENTIAL &&
            qwen3vl_decode_mlp(seq_len_, chunk.len, ctx().is_causal,
                ctx().attention_mask.has_value(), bs)) {
            emit_qwen3vl_decode_mlp(lw, chunk);
        } else if (decode_gate_up_fusion_) {

            emit_qwen3vl_2b_mlp_pipeline(
                lw, rows, h, nq, nkv, hd, tp, chunk.idx);
        } else {
            if (rhinovla_high_precision_) consume_rhinovla_high_activation(rows, chunk.idx);
            emit_mlp_pipeline(
                lw.gate_w, lw.up_w, lw.down_w, rows,
                use_silu_ ? ActivationKind::SILU : ActivationKind::NONE,
                lw.gate_ws, lw.up_ws, lw.down_ws,
                nvfp4_scale_addr("nvfp4_gate_ts"),
                nvfp4_scale_addr("nvfp4_up_ts"),
                nvfp4_scale_addr("nvfp4_down_ts"),
                (uint16_t)layer_idx,
                /*down_out_bf16=*/false,
                /*residual_is_bf16=*/false,
                /*acc32=*/linear_acc32_,
                /*fuse_silu_mul=*/use_planned_silu_mul(),
                /*skip_down=*/false,
                /*bind_silu_mul_route=*/use_planned_silu_mul(),
                /*residual_spm_addr=*/0,
                /*pi05_xor3=*/false, /*pi05_nvfp4_v2=*/false,
                /*prefer_gemv=*/qwen3vl_decode_gemv(seq_len_),
                rhinovla_high_precision_ ? RpuUnaryPrecision::HIGH : RpuUnaryPrecision::BASE,
                rhinovla_high_precision_fusions_);
        }

        // ─────────────────────────────────────────────────────────────────────
        // ★ Phase 8.5 (R-Phase 2): DeepStack visual_embed inject
        //
        // When layer_idx ∈ deepstack_lang_layers_, add a dense visual_embed
        // tensor to residual1 (which currently holds the post-MLP hidden state).
        // The dense tensor is supplied per-forward by the Python adapter and
        // patched into the cached graph via mutable DMA (cursor-patch only the
        // src base; the [chunk.offset → spm_dst] mapping is BUILD-baked).
        //
        // For text-only / decode forwards the caller passes a zero buffer (view
        // into a persistent zero keepalive [MAX_KEEPALIVE_SEQ, hidden]) so the
        // add reduces to a no-op while keeping a single GraphCache slot
        // (findings F22B + F27 — buffer MUST be sized to MAX_KEEPALIVE_SEQ
        // because src_offset_bytes is baked at BUILD; chunk N > 0 reads past
        // the per-forward seq_len region).
        //
        // Injection must precede final_norm fuse: the mergers' outputs are
        // added at HF Qwen3VLTextModel BEFORE the model.norm call —
        // re-ordering corrupts the post-norm hidden state.
        // ─────────────────────────────────────────────────────────────────────
        if (qwen3vl_multiview_text_composite_dispatch_) {
            emit_qwen3vl_multiview_text_retained_deepstack_add(
                layer_idx, chunk, addr(0, "residual1"));
        } else if (qwen3vl_pooler_z1_dispatch_ &&
            qwen3vl_pooler_z1_retained_deepstack_count_ > 0) {
            emit_qwen3vl_pooler_z1_retained_deepstack_add(
                layer_idx, chunk, addr(0, "residual1"));
        } else if (!deepstack_lang_layers_.empty()) {
            auto it = std::find(deepstack_lang_layers_.begin(),
                                deepstack_lang_layers_.end(),
                                static_cast<int64_t>(layer_idx));
            if (it != deepstack_lang_layers_.end()) {
                const int idx_in_list = static_cast<int>(
                    std::distance(deepstack_lang_layers_.begin(), it));
                consume_causal_route(
                    FmbRouteFamily::MUTABLE_DMA,
                    CAUSAL_DECODER_DEEPSTACK_SITE,
                    static_cast<int64_t>(
                        CausalDecoderMutableDmaRoute::DDR_BROADCAST_TO_SPM));
            const uint32_t deepstack_spm = qwen3vl_2b_w8_profile_
                    ? addr(0, "down")
                    : addr(0, "deepstack_scratch");
                rpu_launch_ddr_broadcast_spm_dma_mutable(
                    &deepstack_dense_src_base_[idx_in_list],
                    /*src_offset_bytes=*/
                        chunk_row_offset_(chunk) * h * DWIDTH,
                    /*num_elements=*/rows * h,
                    deepstack_spm,
                    num_cores());
                rpu_launch_eltwise_binary_spm_kernel(
                    addr(0, "residual1"),
                    deepstack_spm,
                    addr(0, "residual1"),
                    rows * h,
                    ValuOpType::ADD,
                    c10::Half(1.0f),
                    num_cores());
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // ★ final_norm fuse (only at the very last layer)
        //   HF Qwen3Model.forward calls self.norm(hidden_states) before returning.
        //   We fuse it into this kernel to avoid a DDR round-trip.
        //   In-place: residual1 → final_norm_w RMSNorm → residual1
        //
        //   final_norm 始终在此处 fuse (prefill 和 decode 都做).
        //   decode (seq_len=1) + fuse_lm_head_=true 时, 后面的 Output DMA 段会
        //   追加 lm_head GEMM 并跳过 spm2ddr, 实现全程 SPM 的 fused decode.
        // ─────────────────────────────────────────────────────────────────────
        if (is_last_layer) {
            // final_norm_w is BUILD-preloaded into the packed slab or the
            // fallback single Persistent slot (see declare_buffers).
            {
                rpu_launch_rmsnorm_spm_kernel(
                    addr(0, "residual1"), addr(0, "residual1"),
                    norm_final_addr(0),
                    rows, h, eps_,
                    consume_decoder_rmsnorm_route(rows, h, chunk.idx), num_cores());
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Output DMA / fused lm_head
        // ─────────────────────────────────────────────────────────────────────
        if (emit_fused_lm_head) {
            // ★ Phase 2.5 fused lm_head (v2 legacy fused-decoder
            //   file documenting this pattern was deleted in v5-08; the v3
            //   path below is the canonical implementation).
            //
            //   此时 residual1 已经是 normed hidden (final_norm fuse 输出).
            //   追加一个 col-partition GEMM 到当前 main graph batch:
            //     input  = residual1 (normed, M=1, K=hidden_size, 在 SPM)
            //     weight = lm_head_w_ (DDR, col-partition swizzled)
            //     output = lm_head_out (SPM, vocab_size/8 per core)
            //
            //   随后用 mutable scatter DMA 将每个 core 的 vocab shard 写到
            //   lm_head_logits_out_ 的连续 DDR 区间。这个 DMA 节点跟 lm_head
            //   GEMM 一起进入 GraphCache BUILD/REPLAY,避免 graph scope 内
            //   host memcpy 读未执行的 SPM。
            //
            //   lm-head 无条件走 generated auto-tile；W8A16 默认 ACC16，
            //   fp16 仅由 per-handle 策略切到 ACC32。
            //
            //   SPM 峰值: lm_head_out = MAX_VOCAB/8 * 2B ≈ 50KB/core
            //   (Phase 9, 与 attention Phase 2-5 / MLP Phase 7-8 不重叠)
            //   Batch decode: M = rows, so the ~150k-row lm_head weight streams
            //   from DDR ONCE for all `bs` sequences — the single biggest weight
            //   read of a decode step.
            consume_causal_route(
                FmbRouteFamily::LINEAR, CAUSAL_DECODER_LM_HEAD_SITE,
                static_cast<int64_t>(CausalDecoderLinearRoute::AUTO_TILE));
            const int64_t lm_head_rows =
                emit_prefill_last_lm_head ? 1 : rows;
            const uint32_t lm_head_input_addr = addr(0, "residual1") +
                static_cast<uint32_t>(
                    (emit_prefill_last_lm_head ? rows - 1 : 0) * h * DWIDTH);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                lm_head_input_addr,        // final normed row(s), in SPM
                lm_head_w_,                // DDR weight [vocab, hidden]
                addr(0, "lm_head_out"),    // SPM output per core, [rows][local_n]
                /*M=*/lm_head_rows,
                /*N=*/vocab_size_,
                /*K=*/h,
                /*partition=*/1,           // col partition
                /*num_cores=*/lm_head_tp(),
                /*bias_spm_addr=*/0,
                lm_head_w_scale_,
                /*nvfp4_tensor_scale_spm_addr=*/0,
                /*nvfp4_layer_id=*/0,
                /*force_acc32=*/linear_acc32_);
            const int64_t local_n = vocab_size_ / lm_head_tp();
            // The logits tensor is [bs, 1, vocab]; core c owns vocab slice
            // [c*local_n, (c+1)*local_n). SPM holds [rows][local_n] per core, so
            // one scatter per row moves that row's 8 shards into its vocab span.
            // bs == 1 collapses to exactly the previous single call.
            const int64_t output_width = vocab_size_ +
                (lm_head_exact_candidates_ > 0 ? h : 0);
            for (int64_t b = 0; b < bs; ++b) {
                consume_causal_route(
                    FmbRouteFamily::MUTABLE_DMA,
                    CAUSAL_DECODER_LM_HEAD_DMA_SITE,
                    static_cast<int64_t>(
                        CausalDecoderMutableDmaRoute::SPM_SCATTER_TO_DDR));
                rpu_launch_spm_scatter_ddr_dma_mutable(
                    addr(0, "lm_head_out") + (uint32_t)(b * local_n * DWIDTH),
                    &lm_head_logits_dst_base_,
                    /*dst_offset_bytes=*/b * output_width * DWIDTH,
                    /*elements_per_core=*/local_n,
                    /*core_stride_bytes=*/local_n * DWIDTH,
                    lm_head_tp());
                if (lm_head_exact_candidates_ > 0) {
                    ctx().consume_physical_route(FmbRouteFamily::MUTABLE_DMA,
                        QWEN3VL_DELIVERY_LM_HIDDEN_DMA_SITE, 4, 0,
                        {h, vocab_size_});
                    rpu_launch_spm_copy_ddr_dma_mutable(
                        lm_head_input_addr +
                            static_cast<uint32_t>(b * h * DWIDTH),
                        &lm_head_logits_dst_base_,
                        /*dst_offset_bytes=*/
                            (b * output_width + vocab_size_) * DWIDTH,
                        /*num_elements=*/h);
                }
            }
        } else if (!ctx().output_to_spm) {
            // 正常路径: spm2ddr 写回 output_tensor_ (或 chain buffer)
            emit_layer_output_dma(layer_idx, chunk);
        }
    }

protected:  // LingBot2 sparse-MoE subclass reuses the decoder's layer state.
    // Chunk size remains per-handle. Exact composite paths scope any prepared
    // override to one forward and restore the handle's cold policy.

    struct Z1LayerInputSelection {
        uint32_t addr = 0;
        bool external = false;
    };

    Z1LayerInputSelection select_z1_layer_input(
        int layer_idx,
        const ChunkInfo& chunk) const;
    uint32_t wall_z1_layer_input_residual_addr(
        int layer_idx,
        const ChunkInfo& chunk) const;
    uint32_t qwen3vl_pooler_z1_layer_input_residual_addr(
        int layer_idx,
        const ChunkInfo& chunk) const;
    void validate_wall_oss_z1_contract() const;
    void validate_wall_oss_z1_mrope_inputs(
        const at::Tensor& position_ids,
        const at::Tensor& rope_cos_il,
        const at::Tensor& rope_sin_il) const;
    void validate_qwen3vl_pooler_z1_contract() const;
    void validate_qwen3vl_pooler_z1_model_profile() const;
    void validate_qwen3vl_pooler_z1_model_profile(
        int64_t retained_deepstack_count,
        int64_t real_len) const;
    void validate_qwen3vl_pooler_z1_position_ids(
        const at::Tensor& position_ids) const;
    void validate_qwen3vl_pooler_z1_rope_il(
        const at::Tensor& rope_cos_il,
        const at::Tensor& rope_sin_il) const;
    bool qwen3vl_pooler_z1_retained_deepstack_bindings_complete() const;
    bool qwen3vl_pooler_z1_retained_deepstack_bindings_empty() const;
    void emit_qwen3vl_pooler_z1_retained_deepstack_add(
        int layer_idx,
        const ChunkInfo& chunk,
        uint32_t residual_addr);
    void validate_qwen3vl_multiview_text_model_profile() const;
    void validate_qwen3vl_multiview_text_position_ids(
        const at::Tensor& position_ids) const;
    void validate_qwen3vl_multiview_text_rope_il(
        const at::Tensor& rope_cos_il,
        const at::Tensor& rope_sin_il) const;
    void validate_qwen3vl_multiview_text_contract() const;
    void emit_qwen3vl_multiview_text_complement_ingress(
        int layer_idx,
        const ChunkInfo& chunk,
        uint32_t residual_addr);
    void emit_qwen3vl_multiview_text_retained_deepstack_add(
        int layer_idx,
        const ChunkInfo& chunk,
        uint32_t residual_addr);

    std::vector<int64_t> delivery_topology_arguments() const {
        return {28, 2048, 6144, 16, 8, 128, 576, 192, 608,
                decode_qkv_fusion_ ? 1 : 0, decode_gate_up_fusion_ ? 1 : 0,
                prefill_packed_qkv_planar_direct_ ? 1 : 0,
                prefill_packed_gate_up_strided_ ? 1 : 0,
                fused_prefill_kv_insert_ ? 1 : 0, fused_decode_kv_insert_ ? 1 : 0,
                fused_prefill_qk_mrope_ ? 1 : 0, fused_decode_qk_mrope_ ? 1 : 0,
                fuse_prefill_last_lm_head_ ? 1 : 0,
                fast_replay_skip_layer_loop_ ? 1 : 0, preload_replay_skip_ ? 1 : 0};
    }

    static std::vector<int64_t> delivery_kv_arguments(int64_t rows, int64_t position) {
        // The fused scalar-position ABI owns four bounded decode graphs. Its
        // descriptor binds the actual cache position, capacity and K/V geometry.
        return {rows, position, 608, 1, 8, 128, 8, rows == 1 ? 1 : 0};
    }

    RpuRmsNormSpmRoute consume_delivery_rmsnorm_route(
        int64_t rows, int64_t cols, int64_t chunk_index, int64_t role) const {
        TORCH_CHECK(qwen3vl_2b_w8_profile_, "delivery RMS route requires its exact profile");
        const auto route = rows % 64 == 0
            ? RpuRmsNormSpmRoute::QWEN3VL_V64 : RpuRmsNormSpmRoute::BASE;
        ctx().consume_physical_route(FmbRouteFamily::NORMALIZATION,
            QWEN3VL_DELIVERY_RMS_SITE, static_cast<int64_t>(route), 0,
            {rows, cols}, 3 * chunk_index + role);
        return route;
    }

    bool norm_slab_model_eligible() const {
        return qwen3vl_2b_w8_profile_ && has_qk_norm_ && !has_qkv_bias_;
    }

    bool norm_slab_active() const {
        return norm_slab_model_eligible() && !adarms_;
    }

    int64_t norm_hidden_bytes() const {
        return Align(hidden_size() * DWIDTH, 256);
    }

    int64_t norm_head_bytes() const {
        return Align(head_dim() * DWIDTH, 256);
    }

    int64_t norm_reverse_layer_offset(
        int layer, int64_t region_base, int64_t field_bytes) const {
        return region_base + (num_layers() - 1 - layer) * field_bytes;
    }

    int64_t norm_final_offset_bytes() const { return 0; }

    int64_t norm_k_region_bytes() const {
        return num_layers() * norm_head_bytes();
    }

    int64_t norm_q_region_bytes() const {
        return norm_k_region_bytes();
    }

    int64_t norm_k_offset_bytes(int layer) const {
        return norm_reverse_layer_offset(
            layer, norm_hidden_bytes(), norm_head_bytes());
    }

    int64_t norm_q_offset_bytes(int layer) const {
        return norm_reverse_layer_offset(
            layer, norm_hidden_bytes() + norm_k_region_bytes(),
            norm_head_bytes());
    }

    int64_t norm_post_region_base_bytes() const {
        return norm_hidden_bytes() + norm_k_region_bytes() +
            norm_q_region_bytes();
    }

    int64_t norm_post_offset_bytes(int layer) const {
        return norm_reverse_layer_offset(
            layer, norm_post_region_base_bytes(), norm_hidden_bytes());
    }

    int64_t norm_input_region_base_bytes() const {
        return norm_post_region_base_bytes() +
            num_layers() * norm_hidden_bytes();
    }

    int64_t norm_input_offset_bytes(int layer) const {
        return norm_reverse_layer_offset(
            layer, norm_input_region_base_bytes(), norm_hidden_bytes());
    }

    int64_t norm_slab_size_bytes() const {
        return norm_input_region_base_bytes() +
            num_layers() * norm_hidden_bytes();
    }

    uint32_t norm_field_addr(int core, int64_t field_offset) const {
        return addr(core, "norm_slab") +
            static_cast<uint32_t>(field_offset);
    }

    uint32_t norm_input_addr(int layer, int core) const {
        return norm_slab_active()
            ? norm_field_addr(core, norm_input_offset_bytes(layer))
            : layer_addr(layer, core, "norm_w");
    }

    uint32_t norm_post_addr(int layer, int core) const {
        return norm_slab_active()
            ? norm_field_addr(core, norm_post_offset_bytes(layer))
            : layer_addr(layer, core, "post_norm_w");
    }

    uint32_t norm_q_addr(int layer, int core) const {
        return norm_slab_active()
            ? norm_field_addr(core, norm_q_offset_bytes(layer))
            : layer_addr(layer, core, "q_norm_w");
    }

    uint32_t norm_k_addr(int layer, int core) const {
        return norm_slab_active()
            ? norm_field_addr(core, norm_k_offset_bytes(layer))
            : layer_addr(layer, core, "k_norm_w");
    }

    uint32_t norm_final_addr(int core) const {
        return norm_slab_active()
            ? norm_field_addr(core, norm_final_offset_bytes())
            : addr(core, "final_norm_w");
    }

    int64_t causal_ring_route(int64_t rows, int64_t cols) const {
        return fmb_ring_all_reduce_route_selector(rows, cols, num_cores());
    }

    std::array<int64_t, 6> cold_topology_arguments() const {
        return {2, num_cores(), attn_tp(), mlp_tp(), lm_head_tp(), NUM_CORES};
    }

    std::vector<int64_t> bind_cold_topology_arguments(
            std::vector<int64_t> arguments) const {
        if (num_cores() != 8) {
            const auto topology = cold_topology_arguments();
            arguments.insert(arguments.end(), topology.begin(), topology.end());
        }
        return arguments;
    }

    int lm_head_tp() const {
        return configured_lm_head_tp_;
    }

    uint32_t nvfp4_scale_addr(const char* name) const {
        return nvfp4_ ? addr(0, name) : 0;
    }

    uint32_t qkv_fused_bias_addr(int layer_idx, const char* name) const {
        return has_qkv_bias_ && !nvfp4_
            ? layer_addr(layer_idx, 0, name)
            : 0u;
    }

    void emit_nvfp4_qkv_bias(
        int layer_idx, const char* bias_name, const char* output_name,
        int64_t seq_len, int64_t local_n)
    {
        if (!nvfp4_ || !has_qkv_bias_) return;
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            layer_addr(layer_idx, 0, bias_name),
            addr(0, output_name),
            addr(0, output_name),
            seq_len, local_n, c10::Half(1.0f),
            ValuOpType::ADD, /*is_bopa=*/false);
    }

    void emit_explicit_mask_spm(int layer_idx, const ChunkInfo& chunk,
                               int64_t seq_len, int64_t kv_seq_len, int tp,
                               uint32_t mask_off) {
        // Existing Wall ownership is unchanged. New zero-growth adopters use
        // this forward's native-validated COMPLETE route as their authority.
        bool retain_mask = keep_explicit_mask_in_spm_;
        if (explicit_mask_residency_requested_) {
            retain_mask = false;  // legacy callers remain per-layer
            if (ctx().has_complete_physical_manifest()) {
                const auto& schedule = ctx().find_physical_route(
                    FmbRouteFamily::GRAPH_SCHEDULE,
                    CAUSAL_DECODER_MASK_SPM_SCHEDULE_SITE, chunk.idx);
                retain_mask = schedule.selector == static_cast<int64_t>(
                    CausalMaskSpmSchedule::FIRST_BODY_FIRST_LAYER);
                TORCH_CHECK(
                    (retain_mask || schedule.selector == static_cast<int64_t>(
                        CausalMaskSpmSchedule::EVERY_LAYER)) &&
                        schedule.flags == 0 && schedule.arguments.size() == 8,
                    "Causal explicit mask SPM schedule is malformed");
                TORCH_CHECK(
                    schedule.arguments[6] >= 0 && schedule.arguments[7] >= 0 &&
                        (!retain_mask || schedule.arguments[7] <= schedule.arguments[6]),
                    "Causal retained mask must not grow temporary SPM");
                if (ctx().body_iter == 0 && layer_idx == 0 && chunk.idx == 0) {
                    ctx().consume_physical_route(
                        FmbRouteFamily::GRAPH_SCHEDULE,
                        CAUSAL_DECODER_MASK_SPM_SCHEDULE_SITE,
                        schedule.selector, 0,
                        {seq_len, kv_seq_len, tp, num_cores(),
                         explicit_mask_body_iterations(), num_layers(),
                         schedule.arguments[6], schedule.arguments[7]}, chunk.idx);
                }
            }
        }
        if (!retain_mask ||
            (ctx().body_iter == 0 && layer_idx == 0 && chunk.idx == 0)) {
            sdpa_dma_mask_to_spm(prepared_attn_mask_, mask_off,
                                 seq_len, kv_seq_len, tp);
        }
    }

    SdpaConfig make_sdpa_config(int mask = 1) const {
        // attn_tp() == NUM_CORES for nkv>=8 (no-op); attn_tp() for Wall-OSS nkv=2.
        // Drives sdpa_tmp sizing + chunk-validity consistently with the SDPA call.
        // Qwen3-0.6B has a measured product=16 failure (C912..C960) even
        // though that product satisfies the generator's multiple-of-eight
        // structural predicate.  Keep the shared helper permissive for
        // profiles with evidence, while carrying this profile's numerical
        // capability ceiling into the planner validity check.  Geometry is
        // used instead of a model-name string so one-layer diagnostic fixtures
        // retain the same safety contract.  This is a compatibility guard for
        // the existing Qwen3 path; the unified planner's profile capability
        // manifest is the eventual source of this datum (and must replace this
        // geometry predicate before any new model is admitted).
        const bool qwen3_06b_shape =
            hidden_size() == 1024 && intermediate_size() == 3072 &&
            num_q_heads() == 16 && num_kv_heads() == 8 && head_dim() == 128 &&
            has_qk_norm_ && !has_qkv_bias_;
        return {sdpa_kernel_, head_dim(), num_q_heads(), num_kv_heads(), attn_tp(),
                mask, qwen3_06b_shape ? 8 : 0};
    }

    struct Qwen3VlDecodeReplayState;
    std::shared_ptr<Qwen3VlDecodeReplayState> qwen3vl_decode_replay_state_;
    bool qwen3vl_decode_replay_scope_active_ = false;
    bool begin_qwen3vl_decode_replay(
        const at::Tensor& hidden, at::TensorList k, at::TensorList v,
        const std::optional<at::Tensor>& mask, int64_t position, bool is_causal,
        at::IntArrayRef descriptor);
    void finish_qwen3vl_decode_replay();
    void cancel_qwen3vl_decode_replay() noexcept;
    bool try_checked_layer_body_replay();

    // ── Model state (stored in C++, set via set_weights) ──
    std::vector<int64_t> causal_kvinsert_cost_weight_identity() const {
        if (layer_weights_.empty()) return {};
        std::vector<int64_t> identity{1};
        if (rhinovla_high_precision_) {
            identity.insert(identity.end(), {RHINOVLA_TEXT_HIGH_RMSNORM_SITE,
                                            rhinovla_high_precision_fusions_ ? 1 : 0});
        }
        if (num_cores() != 8) {
            identity.insert(identity.end(),
                            {0x434f5245, num_cores(), attn_tp(), mlp_tp(),
                             lm_head_tp(), 8, configured_vocab_size_});
        }
        append_kvinsert_cost_scalar_identity(identity, eps_);
        identity.insert(identity.end(), {
            static_cast<int64_t>(has_qk_norm_),
            static_cast<int64_t>(has_qkv_bias_),
            static_cast<int64_t>(fast_replay_skip_layer_loop_),
            static_cast<int64_t>(preload_replay_skip_),
            static_cast<int64_t>(use_silu_),
            static_cast<int64_t>(nvfp4_),
            static_cast<int64_t>(qwen3_spm_kv_by_mha_enabled_),
            static_cast<int64_t>(adarms_),
            static_cast<int64_t>(adarms_mutable_),
            static_cast<int64_t>(adarms_fused_bcast_enabled_),
            static_cast<int64_t>(adarms_unroll_),
            static_cast<int64_t>(adarms_schedule_select_),
            static_cast<int64_t>(fuse_lm_head_),
            static_cast<int64_t>(equal_two_prefill_),
            static_cast<int64_t>(linear_acc32_),
            static_cast<int64_t>(has_mrope_),
            static_cast<int64_t>(configured_chunk_size_cap_),
            static_cast<int64_t>(
                ordinary_rmsnorm_spm_contract_.capability),
            static_cast<int64_t>(vocab_size_),
            static_cast<int64_t>(sdpa_kernel_)});
        if (qwen3vl_4b_decode_qk_fused_) {
            identity.push_back(QWEN3VL_4B_DECODE_QK_NORM_MROPE_SITE);
        }
        if (qwen3vl_4b_decode_qk_kv_fused_) {
            identity.push_back(QWEN3VL_4B_DECODE_QK_NORM_MROPE_KV_INSERT_SITE);
        }
        if (qwen3vl_4b_decode_o_ring_norm_fused_) {
            identity.push_back(QWEN3VL_4B_DECODE_O_RING_NORM_SITE);
        }
        identity.push_back(static_cast<int64_t>(mrope_section_.size()));
        identity.insert(identity.end(), mrope_section_.begin(), mrope_section_.end());
        identity.push_back(static_cast<int64_t>(deepstack_lang_layers_.size()));
        identity.insert(identity.end(), deepstack_lang_layers_.begin(), deepstack_lang_layers_.end());
        identity.push_back(static_cast<int64_t>(layer_weights_.size()));
        for (const auto& weights : layer_weights_) {
            for (const auto* tensor : {
                    &weights.q_w, &weights.k_w, &weights.v_w, &weights.o_w,
                    &weights.q_norm_w, &weights.k_norm_w, &weights.input_norm_w, &weights.post_norm_w,
                    &weights.gate_w, &weights.up_w, &weights.down_w, &weights.q_bias_w,
                    &weights.k_bias_w, &weights.v_bias_w, &weights.q_ws, &weights.k_ws,
                    &weights.v_ws, &weights.o_ws, &weights.gate_ws, &weights.up_ws,
                    &weights.down_ws}) {
                append_kvinsert_cost_tensor_identity(identity, *tensor);
            }
        }
        for (const auto* tensor : {
                &final_norm_w_, &q_nvfp4_tensor_scales_, &k_nvfp4_tensor_scales_, &v_nvfp4_tensor_scales_,
                &o_nvfp4_tensor_scales_, &gate_nvfp4_tensor_scales_, &up_nvfp4_tensor_scales_, &down_nvfp4_tensor_scales_,
                &lm_head_w_, &lm_head_w_scale_}) {
            append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
        return identity;
    }

    std::vector<int64_t> kvinsert_cost_weight_identity() const override {
        // Derived owners must append their own installed weights/policies.
        return typeid(*this) == typeid(CausalDecoderModel)
            ? causal_kvinsert_cost_weight_identity() : std::vector<int64_t>{};
    }

    std::vector<LayerWeights> layer_weights_;
    at::Tensor cos_, sin_;
    at::Tensor final_norm_w_;
    at::Tensor norm_slab_w_;
    double eps_ = 1e-6;
    bool use_silu_ = true;
    bool nvfp4_ = false;
    bool planning_domain_query_ = false;
    bool qwen3_spm_kv_by_mha_enabled_ = true;
    at::Tensor q_nvfp4_tensor_scales_, k_nvfp4_tensor_scales_;
    at::Tensor v_nvfp4_tensor_scales_, o_nvfp4_tensor_scales_;
    at::Tensor gate_nvfp4_tensor_scales_, up_nvfp4_tensor_scales_;
    at::Tensor down_nvfp4_tensor_scales_;

    // ── AdaRMS FiLM (LingBot-VLA action expert) ──
    // When adarms_ (set via set_adarms_step), the per-layer input/post RMSNorm
    // becomes FiLM: rmsnorm(x)·scale + shift. The SCALE rides on the existing
    // input_norm_w / post_norm_w slots (caller passes folded scale=(1+γ)·weight
    // as those); this adds only the per-layer SHIFT + a broadcast shift-add after
    // each norm. Default off ⇒ plain RMS, byte-identical for all existing decoders.
    bool adarms_ = false;
    std::vector<at::Tensor> input_shift_, post_shift_;  // per-layer [hidden] fp16 RPU
    // Replay-safe AdaRMS (set_adarms_step_mutable): stable-addr [num_layers, hidden]
    // keepalives. Contents refreshed per Euler step; the per-layer scale/shift SPM slots
    // are re-DMA'd from these on every REPLAY via emit_adarms_mut_refresh (build-body op),
    // so the expert graph is built once and replayed for steps 1..N. Default off ⇒ the
    // plain build-per-step path above is unchanged (byte-identical for every other decoder).
    bool adarms_mutable_ = false;
    // Cold per native decoder handle: this flag also chooses the Persistent
    // per-layer allocation direction, so the emit path must never re-read a
    // potentially changed process environment.
    bool adarms_fused_bcast_enabled_ = false;
    at::Tensor input_scale_ka_, post_scale_ka_, input_shift_ka_, post_shift_ka_;
    // In-graph denoise-unroll AdaRMS (LingBot-VLA lingbot_denoise op). When
    // adarms_unroll_, the per-body refresh DMAs read this body_iter's (scale, shift)
    // slice from stable [num_steps, num_layers, hidden] buffers at offset
    // (body_iter*num_layers + L)*hidden — so ONE graph unrolls all num_steps Euler
    // steps with per-step FiLM. Requires adarms_mutable_ (declare_buffers + FiLM add).
    // Default off ⇒ emit_adarms_mut_refresh_* keeps the mutable-keepalive path
    // byte-identical. Set via set_adarms_unroll (contents are call-invariant).
    bool adarms_unroll_ = false;
    at::Tensor adarms_is_, adarms_ish_, adarms_ps_, adarms_psh_;  // [num_steps, num_layers, hidden] fp16 RPU
    // Host-bounce-free schedule selection (LingBot2 controlled path). The four schedule
    // tensors above own immutable [num_steps,num_layers,hidden] storage. Captured mutable
    // DMAs retain pointers to the live-base MEMBERS below; select_adarms_schedule_step only
    // advances their values. DDR->SPM refresh remains unchanged.
    bool adarms_schedule_select_ = false;
    int64_t adarms_schedule_step_ = -1;
    uint64_t adarms_is_schedule_base_ = 0, adarms_ish_schedule_base_ = 0;
    uint64_t adarms_ps_schedule_base_ = 0, adarms_psh_schedule_base_ = 0;
    uint64_t adarms_is_live_base_ = 0, adarms_ish_live_base_ = 0;
    uint64_t adarms_ps_live_base_ = 0, adarms_psh_live_base_ = 0;

    // ── Phase 2.5: fused lm_head 状态 (per-instance, set via set_lm_head) ──
    // Cold model geometry is independent of the optional fused-head lifetime.
    int64_t configured_vocab_size_ = 0;  // 0 preserves legacy eight-core callers
    int configured_lm_head_tp_ = 8;
    bool cold_routes_configured_ = false;
    at::Tensor lm_head_w_;        // FP16/W8 [V,H] or packed W4 [V,H/2], RPU
    at::Tensor lm_head_w_scale_;  // W8 [V] or W4 pgrp controller-striped scale
    bool fuse_lm_head_ = false;
    bool fuse_prefill_last_lm_head_ = false;
    int64_t lm_head_exact_candidates_ = 0;
    // Explicit sequential-greedy service contract. Disabled by default so the
    // public logits path retains P4 fresh-output semantics. When enabled for the
    // exact top-2 Qwen3-VL service, callers must consume each result before the
    // next forward and must not retain/accumulate logits tensors.
    bool reuse_lm_head_output_ = false;
    int64_t vocab_size_ = 0;      // lm_head_w_.size(0)
    at::Tensor lm_head_logits_out_;   // graph-written logits (+hidden for rerank)
    uint64_t lm_head_logits_dst_base_ = 0;  // mutable DMA destination base
    // Adapter-owned per-handle auto-planner ceiling. The legacy process-global
    // override still wins when nonzero and remains an exact chunk request.
    int64_t configured_chunk_size_cap_ = 0;
    // Wall TP8 prefill policy: one chunk while seq<=cap, otherwise exactly two
    // equal chunks. Build-time immutable; Python pads the execution length to 32.
    bool equal_two_prefill_ = false;
    bool linear_acc32_ = false;
    // Bound by the first exact reduced-profile weight install, never an env knob.
    bool reduced_qwen3vl_profile_ = false;
    RpuRmsNormSpmContract ordinary_rmsnorm_spm_contract_;
    bool rhinovla_high_precision_ = false;
    bool rhinovla_precision_bound_ = false;
    bool rhinovla_high_precision_fusions_ = false;
    bool rhinovla_precision_fusions_bound_ = false;
    // Scoped only while the native A6 domain is enumerated.  These are inputs
    // that do not exist in LayoutContext but select a real physical route.
    int64_t planning_rope_mode_ = 0;
    FmbGraphLifecycle planning_graph_lifecycle_ =
        FmbGraphLifecycle::RETAINED_CACHE;
    CausalDecoderPlanningMode planning_mode_ =
        CausalDecoderPlanningMode::ORDINARY;
    // Snapshotted in set_weights and admitted only by the exact Qwen3-VL 2B
    // W8 topology. Keeping these per handle avoids changing any shared Qwen,
    // Llama, Wall-OSS, or Qwen3.5 execution path.
    bool qwen3vl_2b_w8_profile_ = false;
    bool qwen3vl_ordinary_resident_prefill_profile_ = false;
    bool qwen3vl_4b_w8_profile_ = false;
    bool qwen3vl_fp16_decode_profile_ = false;
    bool qwen3vl_4b_decode_gateup_fused_ = false;
    bool qwen3vl_4b_decode_qk_fused_ = false;
    bool qwen3vl_4b_decode_qk_kv_fused_ = false;
    bool qwen3vl_4b_decode_o_ring_norm_fused_ = false;
    bool qwen3vl_4b_packed_qkv_weights_ = false;
    at::Tensor qwen3vl_4b_qkv_scale_bank_;
    bool decode_qkv_fusion_ = false;
    bool decode_gate_up_fusion_ = false;
    bool prefill_packed_gate_up_strided_ = false;
    bool prefill_packed_qkv_deinterleave_ = false;
    bool prefill_packed_qkv_planar_direct_ = false;
    bool fused_prefill_kv_insert_ = false;
    bool fused_decode_kv_insert_ = false;
    bool fused_prefill_qk_mrope_ = false;
    bool fused_decode_qk_mrope_ = false;

    // ── HALO WS-2: 1D-RoPE position-base override (cache-insert ↔ RoPE decouple) ──
    //
    // HALO MoT compresses image tokens' RoPE positions: a cond-regime text
    // prefill inserts new K/V at cache offset `ctx().position` (e.g. 4698, the
    // image-prefix length) but those text tokens carry RoPE positions that
    // continue from the *compressed* image position (e.g. 3), NOT the cache
    // offset. The generic 1D path couples both to `ctx().position` (cos_sin_start
    // = ctx().position + chunk.offset), which is correct only when the two
    // coincide (img_uncond: insert 0 == rope 0). Set per-forward from the
    // cos_sin_offset arg; -1 = legacy "use ctx().position". Honored ONLY on the
    // 1D-RoPE path (!has_mrope_): the M-RoPE keepalive is filled at row
    // `ctx().position` (forward():545), so its cos_sin_start must stay coupled.
    int64_t rope_position_base_ = -1;

    // ── R-Phase 1 (Qwen3-VL): M-RoPE state ──
    bool has_mrope_ = false;
    std::vector<int32_t> mrope_section_;                            // [T, H, W] half-dim split
    std::array<std::array<uint16_t, 16>, 3> mrope_strobe_masks_{};  // T/H/W × WARP_SIZE strobe
    at::Tensor position_ids_keepalive_;                             // [MAX_KEEPALIVE_SEQ, 3] int32 RPU DDR

    // ── partial_mrope path (wall-oss perf): host-baked interleaved cos/sin ──
    // When the caller supplies per-forward interleaved cos/sin (rope_cos_il), the
    // 4 M-RoPE launchers swap from llama_mrope_interleave (in-kernel position_ids
    // gather + strobe masks) to partial_mrope (streams these tables). Stable-addr
    // keepalives [MAX_KEEPALIVE_SEQ, head_dim/2] fp16; contents refreshed per
    // forward in the prologue (FAST_REPLAY-safe). partial_mrope_active_ is set
    // per-forward from rope_cos_il.defined() and read by build_layer_subgraph.
    bool partial_mrope_active_ = false;
    at::Tensor cos_il_keepalive_;                                   // [MAX_KEEPALIVE_SEQ, hd/2] fp16 RPU DDR
    at::Tensor sin_il_keepalive_;                                   // [MAX_KEEPALIVE_SEQ, hd/2] fp16 RPU DDR
    using KeepaliveSegmentKey = std::pair<int64_t, int64_t>;
    struct KeepaliveSourceStamp {
        at::Tensor tensor;
        int64_t version;
    };
    using KeepaliveSourceCache =
        std::map<KeepaliveSegmentKey, KeepaliveSourceStamp>;
    static constexpr size_t KEEPALIVE_SOURCE_CACHE_MAX = 128;

    static bool keepalive_segment_hit(
        const KeepaliveSourceCache& cache,
        const KeepaliveSegmentKey& key,
        const at::Tensor& source) {
        const auto& version_counter = source.unsafeGetTensorImpl()->version_counter();
        if (!version_counter.enabled()) return false;
        const auto it = cache.find(key);
        return it != cache.end() && it->second.tensor.is_same(source) &&
            it->second.version == version_counter.current_version();
    }

    static void record_keepalive_segment(
        KeepaliveSourceCache& cache,
        const KeepaliveSegmentKey& key,
        const at::Tensor& source) {
        const int64_t begin = key.first;
        const int64_t end = begin + key.second;
        for (auto it = cache.begin(); it != cache.end();) {
            const int64_t other_begin = it->first.first;
            const int64_t other_end = other_begin + it->first.second;
            if (begin < other_end && other_begin < end) {
                it = cache.erase(it);
            } else {
                ++it;
            }
        }
        if (cache.size() >= KEEPALIVE_SOURCE_CACHE_MAX) {
            cache.clear();
        }
        const auto& version_counter = source.unsafeGetTensorImpl()->version_counter();
        if (version_counter.enabled()) {
            cache.emplace(key, KeepaliveSourceStamp{
                source, static_cast<int64_t>(version_counter.current_version())});
        }
    }

    KeepaliveSourceCache ka_pos_sources_;
    KeepaliveSourceCache ka_cos_sources_;
    KeepaliveSourceCache ka_sin_sources_;
    // Direct composite executors prime keepalives out of band and use this
    // single stamp as their prepared-state contract. The ordinary forward path
    // above uses the bounded segment caches instead.
    int64_t ka_last_position_ = -1;
    int64_t ka_last_seq_ = -1;
    const void* ka_last_pos_src_ = nullptr;
    int64_t ka_last_pos_src_version_ = -1;
    const void* ka_last_cos_src_ = nullptr;
    int64_t ka_last_cos_src_version_ = -1;
    const void* ka_last_sin_src_ = nullptr;
    int64_t ka_last_sin_src_version_ = -1;

    // ── R-Phase 2 (Qwen3-VL): DeepStack state ──
    //
    // deepstack_lang_layers_   — text-decoder layers (in order) that receive
    //                            a visual_embed eltwise add (Qwen3-VL 2B/4B:
    //                            [0, 1, 2]).
    // deepstack_dense_src_base_ — live DDR addresses cursor-patched by mutable
    //                             DMA at each REPLAY. Updated by forward() from
    //                             the caller-supplied dense visual_embeds[i].
    // deepstack_dense_refs_    — keeps the caller's dense visual_embeds[i]
    //                            tensors alive across the synchronous forward
    //                            (mirrors AdaRMS cond_ref_ and keeps DMA inputs
    //                            alive for the synchronous captured forward).
    std::vector<int64_t> deepstack_lang_layers_;
    std::array<uint64_t, 3> deepstack_dense_src_base_{};
    std::vector<at::Tensor> deepstack_dense_refs_;
    at::Tensor deepstack_shared_flush_source_;
    int64_t deepstack_shared_flush_version_ = -1;

    // ── P4 (Wall-OSS): explicit 2D mask state ──
    //
    // Set once per forward in dynamic_config(plan) when base reports an explicit
    // mask (ctx().attention_mask.has_value(), i.e. is_causal=false + mask given).
    // Cached across the per-layer build_layer_subgraph calls of a single forward;
    // ignored on no-mask forwards (ctx().attention_mask empty → LTM/NONE path).
    // mask_type=MASK_2D(4) for the action-denoise rectangular [H, P+H] mask.
    PreparedMask prepared_attn_mask_;
    // sdpa_prepare_mask cache (mirrors vision #23): the explicit 2D mask is
    // host-memoized → stable identity across frames for a fixed prefix, but
    // sdpa_prepare_mask re-uploads it [seq_q,seq_k] CPU→RPU + stable-slot copy
    // every forward. Skip only when the same live tensor, its mutation version,
    // and dimensions are unchanged. The strong ref prevents allocator ABA.
    at::Tensor last_attn_mask_ref_;
    int64_t last_attn_mask_version_ = -1;
    int64_t last_attn_seq_q_ = -1;
    int64_t last_attn_seq_k_ = -1;

    // Wall Action opts into these only after its declare_buffers() reserves the
    // padded K/V rows, gives the explicit mask a full-body lifetime, and has
    // its pre-layer hook populate residual1 on every core.
    int64_t kv_insert_spm_rows_ = 0;
    bool keep_explicit_mask_in_spm_ = false;
    bool explicit_mask_residency_requested_ = false;
    bool explicit_mask_residency_configured_ = false;
    bool pre_layers_residual1_ready_ = false;

    // ── KV_FIRST bidirectional no-mask state ──
    // Q staging keyed by {seq_len, q_ddr_local_dim} (= (num_q_heads()/attn_tp())*
    // head_dim()) — stable per-shape slots, NEVER reallocated, so an already-captured
    // graph's FIXED scatter DMA address stays valid across REPLAY even after another
    // shape's forward (cold-panel re-review: the earlier single grow-only q_ddr_buf_
    // reallocated on a larger seq_len [size(1) < seq_len_in], invalidating a smaller
    // shape's baked address — the SAME replay bug as the SigLIP #6 fix; this MR's
    // qwen3 KV_FIRST [T1] shared the pattern but the original ai-review only flagged
    // SigLIP). Mirrors SigLIP q_ddr_slots_ / temp_ddr_slots_. Bounded by the handle's
    // distinct shape set (a handful in practice); a generous cap (alloc site) catches
    // pathological runaway loudly — per-shape slots can't be freed without breaking
    // the replay-address stability they exist to provide.
    std::map<std::pair<int64_t, int64_t>, at::Tensor> q_ddr_slots_;
    at::Tensor& q_ddr_slot() {
        return q_ddr_slots_.at({seq_len_, (num_q_heads() / attn_tp()) * head_dim()});
    }
    int64_t seq_len_ = 0;

    // Wall-OSS merger zero-copy canary: fixed to two C96 causal prefill chunks.
    // `active_` starts at successful physical adoption (before port binding),
    // so ordinary forward and destroy fail closed through partial begin errors.
    static constexpr int64_t WALL_Z1_CANARY_HIDDEN = 2048;
    static constexpr int64_t WALL_Z1_CANARY_EXECUTION_LEN = 192;
    static constexpr int64_t WALL_Z1_CANARY_REAL_LEN = 180;
    static constexpr int64_t WALL_Z1_CANARY_CHUNK_SIZE = 96;
    int64_t wall_z1_prepared_execution_len_ = 0;
    int64_t wall_z1_prepared_real_len_ = 0;
    int64_t wall_z1_prepared_chunk_size_ = 0;
    uint32_t wall_z1_destination_addr_ = 0;
    uint64_t wall_z1_epoch_ = 0;
    uint64_t wall_z1_plan_hash_ = 0;
    uint64_t wall_z1_position_ids_addr_ = 0;
    uint64_t wall_z1_rope_cos_il_addr_ = 0;
    uint64_t wall_z1_rope_sin_il_addr_ = 0;
    bool wall_z1_active_ = false;
    bool wall_z1_bound_ = false;
    bool wall_z1_dispatch_ = false;
    bool wall_z1_inputs_primed_ = false;
    std::vector<int64_t> wall_z1_stage_descriptor_;

    // Qwen3-VL pooler Z1: fixed N=256-patch canary.  The vision pooler
    // contributes 64 rows to a P128/S72 Text storage port.  Retained DeepStack
    // consumes either one [64,2048] port at layer 0 or three ports at layers
    // {0,1,2} without allocating the ordinary 524,288-byte dense-DDR
    // deepstack_scratch.  All modes keep the real C128/H2048 component
    // temporary arena at 1,966,080 bytes/core.
    static constexpr int64_t QWEN3VL_POOLER_Z1_CANARY_LAYERS = 28;
    static constexpr int64_t QWEN3VL_POOLER_Z1_GROOT_LAYERS = 16;
    static constexpr int64_t QWEN3VL_POOLER_Z1_CANARY_HIDDEN = 2048;
    static constexpr int64_t QWEN3VL_POOLER_Z1_CANARY_INTERMEDIATE = 6144;
    static constexpr int64_t QWEN3VL_POOLER_Z1_CANARY_Q_HEADS = 16;
    static constexpr int64_t QWEN3VL_POOLER_Z1_CANARY_KV_HEADS = 8;
    static constexpr int64_t QWEN3VL_POOLER_Z1_CANARY_HEAD_DIM = 128;
    static constexpr int64_t QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN = 128;
    static constexpr int64_t QWEN3VL_POOLER_Z1_CANARY_REAL_LEN = 72;
    static constexpr int64_t QWEN3VL_POOLER_Z1_GROOT_REAL_LEN = 82;
    static constexpr int64_t QWEN3VL_POOLER_Z1_CANARY_CHUNK_SIZE = 128;
    static constexpr int64_t QWEN3VL_POOLER_Z1_DS1_LAYER = 0;
    static constexpr int64_t QWEN3VL_POOLER_Z1_DS1_ROW_BEGIN = 4;
    static constexpr int64_t QWEN3VL_POOLER_Z1_DS1_ROWS = 64;
    static constexpr std::array<int64_t, 3>
        QWEN3VL_POOLER_Z1_DEEPSTACK_LAYERS = {0, 1, 2};
    static constexpr size_t QWEN3VL_POOLER_Z1_CANARY_TEMPORARY_BYTES =
        1966080;
    int64_t qwen3vl_pooler_z1_prepared_execution_len_ = 0;
    int64_t qwen3vl_pooler_z1_prepared_real_len_ = 0;
    int64_t qwen3vl_pooler_z1_prepared_chunk_size_ = 0;
    uint32_t qwen3vl_pooler_z1_destination_addr_ = 0;
    std::array<uint32_t, 3> qwen3vl_pooler_z1_deepstack_addrs_{};
    int64_t qwen3vl_pooler_z1_retained_deepstack_count_ = 0;
    int64_t qwen3vl_pooler_z1_deepstack_bound_count_ = 0;
    uint64_t qwen3vl_pooler_z1_epoch_ = 0;
    uint64_t qwen3vl_pooler_z1_plan_hash_ = 0;
    uint64_t qwen3vl_pooler_z1_position_ids_addr_ = 0;
    bool qwen3vl_pooler_z1_partial_mrope_ = false;
    bool qwen3vl_pooler_z1_active_ = false;
    bool qwen3vl_pooler_z1_bound_ = false;
    bool qwen3vl_pooler_z1_dispatch_ = false;
    bool qwen3vl_pooler_z1_inputs_primed_ = false;
    std::vector<int64_t> qwen3vl_pooler_z1_stage_descriptor_;
    std::vector<int64_t> qwen3vl_pooler_z1_selected_stage_descriptor_;
    at::Tensor qwen3vl_pooler_z1_output_owner_;

    bool qwen3vl_multiview_text_dry_prepared_ = false;
    int64_t qwen3vl_multiview_text_dry_execution_len_ = 0;
    int64_t qwen3vl_multiview_text_dry_resolved_chunk_size_ = 0;
    int64_t qwen3vl_multiview_text_dry_allocation_chunk_size_ = 0;

    static constexpr int64_t QWEN3VL_MULTIVIEW_TEXT_LAYERS = 36;
    static constexpr int64_t QWEN3VL_MULTIVIEW_TEXT_HIDDEN = 2560;
    static constexpr int64_t QWEN3VL_MULTIVIEW_TEXT_INTERMEDIATE = 9728;
    static constexpr int64_t QWEN3VL_MULTIVIEW_TEXT_Q_HEADS = 32;
    static constexpr int64_t QWEN3VL_MULTIVIEW_TEXT_KV_HEADS = 8;
    static constexpr int64_t QWEN3VL_MULTIVIEW_TEXT_HEAD_DIM = 128;
    static constexpr int64_t QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN = 225;
    static constexpr int64_t QWEN3VL_MULTIVIEW_TEXT_RESOLVED_CHUNK_SIZE = 240;
    static constexpr int64_t QWEN3VL_MULTIVIEW_TEXT_ALLOCATION_CHUNK_SIZE = 225;
    static constexpr int64_t QWEN3VL_MULTIVIEW_TEXT_IMAGE_ROWS = 64;
    static constexpr size_t QWEN3VL_MULTIVIEW_TEXT_TEMPORARY_BYTES =
        4163072;
    static constexpr uint64_t QWEN3VL_MULTIVIEW_TEXT_PROFILE_VERSION =
        UINT64_C(0x51564d5654533236);
    static constexpr std::array<int64_t, 3>
        QWEN3VL_MULTIVIEW_TEXT_IMAGE_ROW_BEGINS = {1, 67, 133};
    static constexpr std::array<std::array<int64_t, 2>, 4>
        QWEN3VL_MULTIVIEW_TEXT_COMPLEMENT_RUNS = {
            std::array<int64_t, 2>{0, 1},
            std::array<int64_t, 2>{65, 2},
            std::array<int64_t, 2>{131, 2},
            std::array<int64_t, 2>{197, 28},
        };
    bool qwen3vl_multiview_text_composite_prepared_ = false;
    bool qwen3vl_multiview_text_composite_dispatch_ = false;
    bool qwen3vl_multiview_text_composite_inputs_primed_ = false;
    int64_t qwen3vl_multiview_text_composite_execution_len_ = 0;
    int64_t qwen3vl_multiview_text_composite_resolved_chunk_size_ = 0;
    int64_t qwen3vl_multiview_text_composite_allocation_chunk_size_ = 0;
    uint64_t qwen3vl_multiview_text_composite_position_ids_addr_ = 0;
    uint64_t qwen3vl_multiview_text_composite_rope_cos_il_addr_ = 0;
    uint64_t qwen3vl_multiview_text_composite_rope_sin_il_addr_ = 0;
    std::array<uint64_t, 4>
        qwen3vl_multiview_text_complement_src_bases_{};
    std::vector<int64_t> qwen3vl_multiview_text_stage_descriptor_;
    at::Tensor qwen3vl_multiview_text_output_owner_;

    // ── Kernel selection (matches existing Qwen3 v2) ──
    SdpaKernelType sdpa_kernel_ = SdpaKernelType::FLASH_ATTN_SPM;
};

}  // namespace v3

namespace v3::wall_oss_z1_internal {

SpmPipelineComponentLayout prepare_text(int64_t handle,
                                        int64_t execution_len,
                                        int64_t real_len);
SpmDense2DSpec text_destination_spec(int64_t handle,
                                     int64_t execution_len);
void prime_text_inputs(int64_t handle,
                       const at::Tensor& position_ids,
                       const at::Tensor& rope_cos_il,
                       const at::Tensor& rope_sin_il,
                       int64_t execution_len);
void rollback_text_inputs(int64_t handle);
void unprepare_text(int64_t handle);
void adopt_text(int64_t handle,
                const SpmPipelineLease& lease,
                const SpmTensorView& scratch);
void bind_text_destination(int64_t handle,
                           const SpmPipelineLease& lease,
                           const SpmPortView& destination);
void validate_text(int64_t handle,
                   const SpmPipelineLease& lease);
void clear_text(int64_t handle, uint64_t epoch, uint64_t plan_hash);
at::Tensor forward_text_z1(int64_t handle,
                           const at::Tensor& hidden_shape_carrier,
                           at::TensorList k_caches,
                           at::TensorList v_caches,
                           const at::Tensor& position_ids,
                           const at::Tensor& rope_cos_il,
                           const at::Tensor& rope_sin_il,
                           uint64_t epoch,
                           uint64_t plan_hash);
void check_text_destroy_allowed(int64_t handle);

}  // namespace v3::wall_oss_z1_internal

namespace v3::qwen3vl_pooler_z1_internal {

SpmPipelineComponentLayout prepare_text(int64_t handle,
                                        int64_t execution_len,
                                        int64_t real_len);
SpmPipelineComponentLayout prepare_text_deepstack1(
    int64_t handle,
    int64_t execution_len,
    int64_t real_len);
SpmPipelineComponentLayout prepare_text_deepstack3(
    int64_t handle,
    int64_t execution_len,
    int64_t real_len);
SpmPipelineCausalPrefillDryLayout prepare_multiview_text_dry(
    int64_t handle,
    int64_t execution_len);
void cancel_multiview_text_dry(int64_t handle);
SpmPipelineComponentLayout prepare_multiview_text_composite(
    int64_t handle,
    int64_t execution_len, at::IntArrayRef selected_descriptor);
SpmFmbResolvedExecutionProfile resolve_multiview_text_profile(
    int64_t handle);
FusedModelBase& multiview_text_owner(int64_t handle);
std::vector<int64_t> multiview_text_stage_descriptor(int64_t handle);
SpmFmbResolvedPhaseManifest seal_multiview_text_manifest(
    int64_t handle,
    const SpmFmbResolvedExecutionProfile& profile,
    SpmScratchId arena,
    uint32_t arena_base);
SpmFmbConsumerRowSliceEndpoint seal_multiview_text_consumer(
    int64_t handle,
    const SpmFmbResolvedExecutionProfile& profile,
    const SpmFmbResolvedPhaseManifest& manifest,
    SpmPortId destination_port,
    int64_t row_begin,
    int layer_idx);
void prime_multiview_text_inputs(
    int64_t handle,
    const at::Tensor& position_ids,
    int64_t execution_len,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il);
void rollback_multiview_text_inputs(int64_t handle);
void unprepare_multiview_text_composite(int64_t handle);
at::Tensor forward_multiview_text_composite(
    int64_t handle,
    const at::Tensor& hidden_states,
    at::TensorList k_caches,
    at::TensorList v_caches,
    const at::Tensor& position_ids,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il);
void stage_multiview_text_outer_fast_component(
    int64_t handle,
    ::GraphKernelRegisterCensusGuard& guard,
    at::TensorList k_caches,
    at::TensorList v_caches);
void bind_multiview_text_outer_fast_input(
    int64_t handle,
    ::GraphKernelRegisterCensusGuard& guard,
    const at::Tensor& hidden_states);
at::Tensor multiview_text_outer_fast_output(int64_t handle);
void prime_text_inputs(int64_t handle,
                       const at::Tensor& position_ids,
                       int64_t execution_len,
                       const std::optional<at::Tensor>& rope_cos_il =
                           std::nullopt,
                       const std::optional<at::Tensor>& rope_sin_il =
                           std::nullopt);
void rollback_text_inputs(int64_t handle);
void unprepare_text(int64_t handle);
SpmDense2DSpec text_destination_spec(int64_t handle,
                                     int64_t execution_len);
SpmDense2DSpec text_deepstack_spec(int64_t handle, int64_t ordinal);
SpmDense2DSpec text_deepstack1_spec(int64_t handle);
void adopt_text(int64_t handle,
                const SpmPipelineLease& lease,
                const SpmTensorView& scratch);
void bind_text_destination(int64_t handle,
                           const SpmPipelineLease& lease,
                           const SpmPortView& destination);
void bind_text_deepstack(int64_t handle,
                         const SpmPipelineLease& lease,
                         const SpmPortView& deepstack,
                         int64_t ordinal);
void bind_text_deepstack1(int64_t handle,
                          const SpmPipelineLease& lease,
                          const SpmPortView& deepstack1);
void validate_text(int64_t handle,
                   const SpmPipelineLease& lease);
void clear_text(int64_t handle, uint64_t epoch, uint64_t plan_hash);
void stage_text_outer_fast_component(
    int64_t handle,
    ::GraphKernelRegisterCensusGuard& guard,
    at::TensorList k_caches,
    at::TensorList v_caches);
at::Tensor text_outer_fast_output(int64_t handle);
at::Tensor forward_text_z1(int64_t handle,
                           const at::Tensor& hidden_shape_carrier,
                           at::TensorList k_caches,
                           at::TensorList v_caches,
                           const at::Tensor& position_ids,
                           uint64_t epoch,
                           uint64_t plan_hash);
void check_text_destroy_allowed(int64_t handle);

}  // namespace v3::qwen3vl_pooler_z1_internal
