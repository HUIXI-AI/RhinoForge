#include "pi05_kernel_policy.h"
// src/fused/rpu_pi05_denoise_step_model.cpp
//
// Pi0.5 num_steps fused denoise step model (FusedModelBase v3 subclass).
//
// Phase 1 skeleton: compiles, asserts on shapes, kernels are NOT emitted.
// Phase 2 fills emit_pre_layers_body / build_layer_subgraph / emit_post_layers_body.

#include "rpu_pi05_denoise_step_model.h"
#include "rpu_pi05_producer_chain.h"
#include "rpu_pi05_q_rope.h"
#include "model_handle_registry.h"
#include "rpu_kernel_decls.h"
#include "rpu_kernel_cache.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_runtime_state.h"
#include "rpu_spm_allocator.h"

#include <algorithm>
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

using namespace at;
using namespace ::rhino_lkn;

#define DWIDTH 2

namespace v3 {

// PI05_DENOISE_FIXED_KERNEL_BASIS: remaining non-manifest launchers implement
// fixed denoise math or BufferDecl transport with no route selector. Any future
// candidate must first be represented by a typed physical-manifest route.

namespace {
int64_t pi05_gateup_bundle_layers() {
    const char* value = std::getenv("RPU_PI05_DENOISE_GATEUP_RESIDENT_LAYERS");
    if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) return 0;
    if (std::strcmp(value, "1") == 0) return 1;
    TORCH_CHECK(std::strcmp(value, "5") == 0,
                "RPU_PI05_DENOISE_GATEUP_RESIDENT_LAYERS must be 0, 1 or 5");
    return 5;
}
}  // namespace

Pi05DenoiseStepModel::Pi05DenoiseStepModel(bool linear_acc32)
    : linear_acc32_(linear_acc32), pi05_ring_xor3_opt_in_(!linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_RING_XOR3")),
      pi05_k_rope_insert_opt_in_(!linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_K_ROPE_INSERT")),
      pi05_gateup_resident_opt_in_(!linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_DENOISE_GATEUP_RESIDENT")),
      pi05_adarms_body_stream_opt_in_(!linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_DENOISE_ADARMS_BODY_STREAM")),
      pi05_gateup_resident_layers_(linear_acc32_ ? 0 : pi05_gateup_bundle_layers()),
      pi05_elide_unused_dma_opt_in_(!linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_DENOISE_ELIDE_UNUSED_DMA")),
      pi05_spm_action_staging_opt_in_(!linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_DENOISE_SPM_ACTION_STAGING")),
      pi05_q_rope_epilogue_opt_in_(!linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_DENOISE_Q_ROPE_EPILOGUE")),
      pi05_xor3_gated_attn_opt_in_(
          !linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_DENOISE_XOR3_GATED_ATTN")),
      pi05_kv1_stripe_gather_opt_in_(
          !linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_DENOISE_KV1_GATHER")),
      pi05_kv1_direct_cache_opt_in_(
          !linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_DENOISE_KV1_DIRECT_CACHE")),
      pi05_kv1_pair_owner_opt_in_(
          !linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_DENOISE_KV1_PAIR_OWNER")),
      pi05_geglu_m50_opt_in_(
          !linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_DENOISE_GEGLU_M50")),
      pi05_xor3_gated_mlp_opt_in_(
          !linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_DENOISE_XOR3_GATED_MLP")),
      pi05_w4_fastpath_opt_in_(!linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_DENOISE_W4_FASTPATH")),
      pi05_w4_geglu_m50_opt_in_(!linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_DENOISE_W4_GEGLU_M50")),
      pi05_w4_geglu_n64_opt_in_(!linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_DENOISE_W4_GEGLU_N64")),
      // The historical NVFP4_GEGLU_M50 flag cannot reopen mixed ACC32/ACC16.
      // Only the separate ACC16 producer may join the uniform ACC16 stack.
      pi05_nvfp4_geglu_m50_opt_in_(
          !linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_DENOISE_NVFP4_GEGLU_ACC16_M50")) {
    TORCH_CHECK(!pi05_nvfp4_geglu_m50_opt_in_ ||
        (pi05_w4_fastpath_opt_in_ && pi05_xor3_gated_mlp_opt_in_ &&
         !pi05_w4_geglu_m50_opt_in_ && !pi05_w4_geglu_n64_opt_in_ && !pi05_geglu_m50_opt_in_),
        "Pi NVFP4 GeGLU requires its W4 continuation stack and no INT4/W8 GeGLU producer");
    TORCH_CHECK(!pi05_w4_geglu_n64_opt_in_ || pi05_w4_geglu_m50_opt_in_,
                "Pi W4 GeGLU N64 requires the complete W4_GEGLU_M50 stack");
    TORCH_CHECK(!pi05_w4_geglu_m50_opt_in_ ||
                    (pi05_w4_fastpath_opt_in_ && pi05_xor3_gated_mlp_opt_in_ &&
                     !pi05_geglu_m50_opt_in_),
                "Pi W4 GeGLU requires W4_FASTPATH=1, XOR3_GATED_MLP=1 "
                "and the W8 GEGLU_M50 flag disabled");
    TORCH_CHECK(!pi05_w4_fastpath_opt_in_ ||
        (pi05_ring_xor3_opt_in_ && pi05_k_rope_insert_opt_in_ &&
         pi05_kv1_pair_owner_opt_in_ && pi05_kv1_direct_cache_opt_in_ &&
         pi05_xor3_gated_attn_opt_in_ && !pi05_geglu_m50_opt_in_ &&
         !pi05_kv1_stripe_gather_opt_in_ &&
         !pi05_gateup_resident_opt_in_ && !pi05_adarms_body_stream_opt_in_ &&
         pi05_gateup_resident_layers_ == 0 && !pi05_elide_unused_dma_opt_in_ &&
         !pi05_spm_action_staging_opt_in_ && !pi05_q_rope_epilogue_opt_in_),
        "Pi W4 fastpath requires its exact KV1 pair/direct and XOR3-attention stack");
    TORCH_CHECK(!pi05_gateup_resident_layers_ ||
                    (pi05_adarms_body_stream_opt_in_ && !pi05_gateup_resident_opt_in_),
                "Pi0.5 Gate/Up bundle requires ADARMS_BODY_STREAM=1 and legacy GATEUP_RESIDENT=0");
    TORCH_CHECK(
        !(pi05_kv1_stripe_gather_opt_in_ && pi05_kv1_direct_cache_opt_in_),
        "RPU_PI05_DENOISE_KV1_GATHER and "
        "RPU_PI05_DENOISE_KV1_DIRECT_CACHE are mutually exclusive");
    TORCH_CHECK(
        !pi05_kv1_pair_owner_opt_in_ || pi05_kv1_direct_cache_opt_in_,
        "RPU_PI05_DENOISE_KV1_PAIR_OWNER requires "
        "RPU_PI05_DENOISE_KV1_DIRECT_CACHE=1");
    TORCH_CHECK(
        !pi05_geglu_m50_opt_in_ ||
            (pi05_kv1_pair_owner_opt_in_ && pi05_xor3_gated_attn_opt_in_),
        "RPU_PI05_DENOISE_GEGLU_M50 requires KV1_PAIR_OWNER=1 "
        "and XOR3_GATED_ATTN=1");
    TORCH_CHECK(
        !pi05_xor3_gated_mlp_opt_in_ || pi05_geglu_m50_opt_in_ ||
            pi05_w4_fastpath_opt_in_,
        "RPU_PI05_DENOISE_XOR3_GATED_MLP requires DENOISE_GEGLU_M50=1 "
        "or the exact DENOISE_W4_FASTPATH=1 profile");
}
Pi05DenoiseStepModel::~Pi05DenoiseStepModel() = default;

namespace {

// These two camera profiles share the same M50 Action math and storage ABI.
// Keep the real prefix in every descriptor; never pad P544 into P800.
bool pi05_action_prefix_admitted(int64_t prefix) {
    return prefix == 544 || prefix == 576 || prefix == 608 || prefix == 640 || prefix == 800 || prefix == 832 || prefix == 864 || prefix == 896;
}

// Bounded schedule extension. Per-body kernels are unchanged; the
// prepared table, COMPLETE manifest, graph signature and body count bind the
// selected schedule. This does not admit other denoise counts or certify quality.
bool pi05_action_steps_admitted(int64_t steps) {
    return steps == 4 || steps == 10;
}

template <typename Layers>
bool pi05_is_fp16_action(const Layers& layers) {
    return !layers.empty() && layers.front().q_w.defined() &&
        layers.front().q_w.scalar_type() == at::kHalf;
}

template <typename Layers>
bool pi05_fp16_projection_owners_admitted(const Layers& layers) {
    if (layers.size() != 18) return false;
    const auto weight = [](const at::Tensor& value, int64_t n, int64_t k) {
        return value.defined() && value.device().type() == at::kPrivateUse1 &&
            value.scalar_type() == at::kHalf && value.is_contiguous() &&
            value.storage_offset() == 0 &&
            value.sizes() == at::IntArrayRef({n, k}) &&
            value.nbytes() == n * k * sizeof(c10::Half);
    };
    return std::all_of(layers.begin(), layers.end(), [&](const auto& layer) {
        return weight(layer.q_w, 2048, 1024) && weight(layer.k_w, 2048, 1024) &&
            weight(layer.v_w, 2048, 1024) && weight(layer.o_w, 1024, 2048) &&
            weight(layer.gate_proj_w, 4096, 1024) &&
            weight(layer.up_proj_w, 4096, 1024) &&
            weight(layer.down_proj_w, 1024, 4096) &&
            !layer.q_ws.defined() && !layer.k_ws.defined() &&
            !layer.v_ws.defined() && !layer.o_ws.defined() &&
            !layer.gate_ws.defined() && !layer.up_ws.defined() &&
            !layer.down_ws.defined();
    });
}

constexpr int64_t PI05_DENOISE_CORE_PROFILE_SITE = 645159008074800571LL;
constexpr int64_t PI05_DENOISE_COND_PREPARE_SITE = 4991200776089332343LL;

constexpr int64_t PI05_DENOISE_NVFP4_GEGLU_POLICY_SITE = 0x50494e34474c504fLL;
constexpr int64_t PI05_DENOISE_NVFP4_GEGLU_LINEAR_SITE = 0x50494e34474c4c49LL;
constexpr int64_t PI05_DENOISE_NVFP4_GEGLU_M320N64K128 = 13;
constexpr int64_t PI05_DENOISE_NVFP4_GEGLU_ACC16_M320N64K128 = 16;
constexpr int64_t PI05_DENOISE_W4_FASTPATH_POLICY_SITE = 0x5049573446415354LL;
constexpr int64_t PI05_DENOISE_W4_GEGLU_M50_POLICY_SITE = 0x50495734474c504fLL;
constexpr int64_t PI05_DENOISE_W4_GEGLU_M50_LINEAR_SITE = 0x50495734474c4c49LL;
enum class Pi05DenoiseW4GeGluLinearRouteSelector : int64_t {
    PGRP_M512N48K128 = 11,
    PGRP_M464N64K128 = 12,
};
constexpr int64_t PI05_DENOISE_W4_GEGLU_M50_M512N48K128 =
    static_cast<int64_t>(Pi05DenoiseW4GeGluLinearRouteSelector::PGRP_M512N48K128);
constexpr int64_t PI05_DENOISE_W4_GEGLU_M50_M464N64K128 =
    static_cast<int64_t>(Pi05DenoiseW4GeGluLinearRouteSelector::PGRP_M464N64K128);
constexpr int64_t PI05_DENOISE_Q_LINEAR_SITE = 799604933306379572LL;
constexpr int64_t PI05_DENOISE_K_LINEAR_SITE = 1383321966524453364LL;
constexpr int64_t PI05_DENOISE_V_LINEAR_SITE = 5742068087814012512LL;
constexpr int64_t PI05_DENOISE_KV1_POLICY_SITE = 0x5049354b5631504fLL;
constexpr int64_t PI05_DENOISE_KV1_DIRECT_POLICY_SITE = 0x5049354b56314443LL;
constexpr int64_t PI05_DENOISE_KV1_K_GATHER_SITE = 0x5049354b56314b47LL;
constexpr int64_t PI05_DENOISE_KV1_V_GATHER_SITE = 0x5049354b56315647LL;
constexpr int64_t PI05_DENOISE_KV1_PAIR_OWNER_SITE = 0x5049354b56315052LL;
constexpr int64_t PI05_DENOISE_KV1_FIXED_M592N32 = 7;
constexpr int64_t PI05_DENOISE_KV1_DIRECT_M592N32 = 8;
constexpr int64_t PI05_DENOISE_KV1_PAIR_OWNER_M50N32X2K1024 = 9;
constexpr int64_t PI05_DENOISE_KV1_PAIR_OWNER_FP16_M50N32X2K1024 = 14;
constexpr int64_t PI05_DENOISE_GEGLU_M50_POLICY_SITE = 0x50493547454c504fLL;
constexpr int64_t PI05_DENOISE_GEGLU_M50_LINEAR_SITE = 0x50493547454c4c49LL;
constexpr int64_t PI05_DENOISE_GEGLU_M50_M512N48K128 = 10;
constexpr int64_t PI05_DENOISE_GEGLU_FP16_M50_M608N32K128 = 15;
int64_t pi05_kv1_pair_owner_selector(bool fp16) {
    return fp16 ? PI05_DENOISE_KV1_PAIR_OWNER_FP16_M50N32X2K1024
                : PI05_DENOISE_KV1_PAIR_OWNER_M50N32X2K1024;
}
int64_t pi05_geglu_m50_selector(bool fp16) {
    return fp16 ? PI05_DENOISE_GEGLU_FP16_M50_M608N32K128
                : PI05_DENOISE_GEGLU_M50_M512N48K128;
}
constexpr int64_t PI05_DENOISE_XOR3_GATED_MLP_POLICY_SITE = 0x5049354d4c50504fLL;
constexpr int64_t PI05_DENOISE_XOR3_GATED_MLP_REDUCE_SITE = 0x5049354d4c504152LL;
constexpr int64_t PI05_DENOISE_XOR3_GATED_MLP_M50N1024 = 7;
std::vector<int64_t> pi05_kv1_policy_arguments(bool enabled, int64_t prefix) {
    // ABI, cold flag, steps, P, M, logical D, local D, K, TP.
    return {1, enabled ? 1 : 0, 10, prefix, 50, 256, 32, 1024, 8};
}
std::vector<int64_t> pi05_kv1_linear_arguments(int64_t prefix) {
    // ABI, steps, P, M, logical D, local D, K, TP, tile M/N/K, grid.
    return {1, 10, prefix, 50, 256, 32, 1024, 8, 592, 32, 128, 1, 1, 1};
}
std::vector<int64_t> pi05_kv1_gather_arguments(int64_t prefix) {
    // ABI, steps, P, rows, local D, logical D, dwidth, cores, schedule.
    return {1, 10, prefix, 50, 32, 256, 2, 8,
            static_cast<int64_t>(RpuAllGatherSchedule::LITTLE_CHUNK)};
}
std::vector<int64_t> pi05_kv1_direct_policy_arguments(
    bool enabled, int64_t logical_position, int64_t prefix) {
    // ABI, cold flag, exact denoise/cache geometry, pair-map, fixed producer,
    // direct consumer, payload ABI, explicit register count and grid.
    return {
        1, enabled ? 1 : 0, 10, prefix, 50, 64, prefix + 50, 256, 8, 32, 16,
        1024, 1024, 18, 8, 2048, 0, 8,
        static_cast<int64_t>(KernelId::PI05_KV1_STRIPE_W8A16_M592N32),
        static_cast<int64_t>(
            KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M50D256P64),
        592, 32, 128, 1, 68, 4, 1, 1, 0, logical_position};
}
std::vector<int64_t> pi05_kv1_pair_owner_policy_arguments(
    bool enabled, int64_t logical_position, int64_t prefix, bool fp16,
    int64_t steps = 10) {
    // ABI2, cold flag, exact denoise/cache geometry, pair-map, full owner
    // shapes, pair producer/direct consumer IDs, register ABI and grids,
    // batch offsets, logical position and zero compact-companion bytes.
    return {
        fp16 ? 3 : 2, enabled ? 1 : 0, steps, prefix, 50, 64, prefix + 50, 256, 8, 32, 16,
        1024, 1024, 18, 8, 2048, 0, 8,
        static_cast<int64_t>(
            fp16 ? KernelId::PI05_DENOISE_KV1_PAIR_OWNER_FP16_M50N32X2K1024
                 : KernelId::PI05_DENOISE_KV1_PAIR_OWNER_W8A16_M50N32X2K1024),
        static_cast<int64_t>(
            KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M50D256P64),
        2048, 1024, fp16 ? 0 : 2048, 68, 2, 1, 1, 0, 68, 4, 1, 1, 0,
        logical_position, 0};
}
std::vector<int64_t> pi05_kv1_direct_linear_arguments(int64_t prefix) {
    // ABI2 differentiates the pair-map owner from stripe-gather's logical-D
    // owner although both use the same exact generated M592/N32 program.
    return {2, 10, prefix, 50, 256, 32, 16, 1024, 8,
            0, 8, 592, 32, 128, 1, 1, 1};
}
std::vector<int64_t> pi05_kv1_pair_owner_linear_arguments(
    int64_t prefix, bool fp16, int64_t steps = 10) {
    // ABI, steps, P, M, logical D, local D, D16, K, TP, pair db{c,c+8},
    // full weight/scale owner shapes, register count and grid.
    return {fp16 ? 2 : 1, steps, prefix, 50, 256, 32, 16, 1024, 8,
            0, 8, 2048, 1024, fp16 ? 0 : 2048, 68, 2, 1, 1, 0};
}
constexpr int64_t PI05_DENOISE_Q_ROPE_SITE = 1426665372740531219LL;
constexpr int64_t PI05_DENOISE_K_ROPE_SITE = 1167105004526897534LL;
constexpr int64_t PI05_DENOISE_KV_INSERT_SITE = 1046939961912038728LL;
// Owner-local variant bit; does not overlap shared dynamic-position or
// ROPE-table-residency bits. K-ROPE and KV_INSERT must agree on this flag.
constexpr int64_t PI05_DENOISE_K_ROPE_INSERT_FLAG = 1;
constexpr int64_t PI05_DENOISE_KV1_GATHERED_SOURCE_FLAG = 2;
constexpr int64_t PI05_DENOISE_KV1_DIRECT_CACHE_FLAG = 4;
std::vector<int64_t> pi05_k_rope_insert_arguments(
    int64_t logical_position, int64_t k_capacity, int64_t v_capacity, int64_t prefix) {
    // logical RoPE position, version, steps, physical P, M, padded K/V rows,
    // valid KV, QH, KVH, D, TP, H, layers, mask, K/V capacity, grid, batch offsets.
    return {logical_position, 1, 10, prefix, 50, 64, prefix + 50, 8, 8, 256, 8,
            1024, 18, 4, k_capacity, v_capacity, 2, 1, 1, 0, 0};
}
constexpr int64_t PI05_DENOISE_ATTENTION_SITE = 5847544569257884056LL;
constexpr int64_t PI05_DENOISE_MASK_SCHEDULE_SITE = 3223896414807375044LL;
constexpr int64_t PI05_DENOISE_PREPARE_ALL_REDUCE_SITE =
    4334879675031300667LL;
constexpr int64_t PI05_DENOISE_O_LINEAR_SITE = 907382291428438484LL;
constexpr int64_t PI05_DENOISE_ATTENTION_ALL_REDUCE_SITE =
    1690541798343266616LL;
constexpr int64_t PI05_DENOISE_PRE_X_DMA_SITE = 1463285755170168969LL;
constexpr int64_t PI05_DENOISE_PRE_COND_DMA_SITE = 787521761862726649LL;
constexpr int64_t PI05_DENOISE_PRE_LINEAR_SITE = 7775333525413465564LL;
constexpr int64_t PI05_DENOISE_POST_LINEAR_SITE = 8195424785009525971LL;
constexpr int64_t PI05_DENOISE_POST_LOOP_DMA_SITE = 7012523434720195011LL;
constexpr int64_t PI05_DENOISE_POST_STEP_DMA_SITE = 5240453042848998434LL;
constexpr int64_t PI05_DENOISE_GEMV_LINEAR_SITE = 4681095848313937529LL;
constexpr int64_t PI05_DENOISE_GEMV_ALL_REDUCE_SITE = 7943696681017534225LL;
constexpr int64_t PI05_DENOISE_ADARMS_TABLE_DMA_SITE = 6143677821685490261LL;
constexpr int64_t PI05_DENOISE_ATTN_NORM_SITE = 0x504930354e4f5201LL;
constexpr int64_t PI05_DENOISE_MLP_NORM_SITE = 0x504930354e4f5202LL;
constexpr int64_t PI05_DENOISE_FINAL_NORM_SITE = 0x504930354e4f5203LL;
constexpr int64_t PI05_DENOISE_GATEUP_RESIDENT_SITE = 0x5049303557384c30LL;
constexpr int64_t PI05_DENOISE_GATEUP_PRELOAD_SITE = 0x504930355738444dLL;
constexpr int64_t PI05_DENOISE_RESIDENT_W8_BYTES = 524288;
constexpr int64_t PI05_DENOISE_BUNDLE_LINEAR_SITE = 0x5049303542554c30LL;
constexpr int64_t PI05_DENOISE_BUNDLE_PRELOAD_SITE = 0x504930354255444dLL;
constexpr int64_t PI05_DENOISE_ADARMS_BODY_BYTES = 227328;
std::vector<int64_t> pi05_adarms_body_arguments(int64_t body) {
    TORCH_CHECK(body >= 0 && body < 10, "Pi0.5 AdaRMS body DMA index out of range");
    // version, steps, layers, H, rows/body, row width, TP, bytes/body, DDR offset.
    return {1, 10, 18, 1024, 37, 3072, 8, 227328, body * 227328};
}
std::vector<int64_t> pi05_gateup_bundle_arguments(int64_t layers) {
    TORCH_CHECK(layers == 1 || layers == 5, "Pi0.5 Gate/Up bundle layer count drift");
    // version, first layer, count, projections, steps, P, M, N, K, TP,
    // matrix bytes, core stride, tile M/N/K, body-table bytes, core-major codec.
    return {1, 0, layers, 2, 10, 800, 50, 4096, 1024, 8, 524288,
            layers * 2 * 524288, 512, 48, 128, 227328, 1};
}
std::vector<int64_t> pi05_gateup_bundle_linear_arguments(
    int64_t layers, int64_t layer, int64_t projection) {
    TORCH_CHECK(layer >= 0 && layer < layers && projection >= 0 && projection < 2,
                "Pi0.5 Gate/Up bundle projection index out of range");
    auto args = pi05_gateup_bundle_arguments(layers);
    args.insert(args.end(), {layer, projection, (2 * layer + projection) * 524288});
    return args;
}
// Owner-local physical launcher, deliberately distinct from shared AUTO_TILE.
constexpr int64_t PI05_DENOISE_SPM_WEIGHT_M512N48K128 = 5;
std::vector<int64_t> pi05_gateup_resident_arguments() {
    // version, layer, steps, P, M, N, K, TP, local bytes, tile M/N/K.
    return {1, 0, 10, 800, 50, 4096, 1024, 8, 524288, 512, 48, 128};
}

constexpr int64_t PI05_DENOISE_UNUSED_OUTPUT_SITE = 0x50493035444d4101LL;
constexpr int64_t PI05_DENOISE_ELIDE_UNUSED_DMA = 1;
constexpr int64_t PI05_DENOISE_SPM_ACTION_STAGING = 2;
constexpr int64_t PI05_DENOISE_ACTION_SPM_PRODUCER_SITE = 0x50493035444d4102LL;
constexpr int64_t PI05_DENOISE_ACTION_SPM_INPUT_SITE = 0x50493035444d4103LL;
std::vector<int64_t> pi05_graph_dma_arguments(int64_t prefix) {
    // version, steps, final body, P, M, KV, H, I, layers, QH, KVH, D, TP,
    // action width and actual cache capacity. Logical RoPE remains independent.
    return {1, 10, 9, prefix, 50, prefix + 50, 1024, 4096, 18, 8, 8, 256, 8, 32, 2048};
}

bool pi05_graph_dma_shape_admitted(
    const FmbThreeStageChunkPlan& plan, int64_t physical_len,
    int64_t logical_len, int64_t position, int64_t batch_size,
    bool use_attn_mask, bool is_causal, AttentionExecutionPolicy attention) {
    if (physical_len != 50 || logical_len != 50 || !pi05_action_prefix_admitted(position) ||
        batch_size != 1 || !use_attn_mask || is_causal ||
        attention != AttentionExecutionPolicy::DDR_KV ||
        plan.chunk_mode != ChunkMode::SEQUENTIAL || plan.spans.size() != 1 ||
        plan.spans.front().offset != 0 || plan.spans.front().len != 50) return false;
    const auto one = [position](const auto& chunks) {
        return chunks.size() == 1 && chunks.front().idx == 0 &&
            chunks.front().offset == 0 && chunks.front().len == 50 &&
            chunks.front().kv_seq_len == position + 50;
    };
    return one(plan.input.chunks) && one(plan.qkv.chunks) && one(plan.compute.chunks);
}

bool pi05_graph_dma_write_final_body(int64_t body, int64_t steps) {
    TORCH_CHECK(steps == 10 && body >= 0 && body < 10,
                "Pi0.5 DMA elision requires the original ten body iterations");
    return body == 9;
}

void pi05_validate_action_spm_copy(
    uint64_t source, uint64_t destination, uint64_t spm_base, uint64_t usable_bytes) {
    constexpr uint64_t bytes = 50 * 1024 * sizeof(c10::Half);
    TORCH_CHECK((source % 256) == 0 && (destination % 256) == 0 &&
                    source >= spm_base && destination >= spm_base &&
                    bytes <= usable_bytes &&
                    source - spm_base <= usable_bytes - bytes &&
                    destination - spm_base <= usable_bytes - bytes &&
                    (source >= destination ? source - destination >= bytes
                                           : destination - source >= bytes),
                "Pi0.5 action SPM copy requires distinct aligned full-size "
                "planner-owned source/destination slots");
}

void pi05_require_stable_graph_dma_capture() {
    TORCH_CHECK(RpuKernelGraph::has_active(),
                "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 optimized DMA "
                "descriptor requires a signed replayable Graph; legacy "
                "PASSTHROUGH must use its original descriptor-free path");
    auto& graph = RpuKernelGraph::active();
    TORCH_CHECK((graph.state() == RpuKernelGraph::State::RECORDING ||
                 graph.state() == RpuKernelGraph::State::REPLAYING) &&
                    graph.replayable(),
                "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 optimized DMA "
                "requires stable BUILD/REPLAY");
    const auto stamp = graph.op_stream_stamp();
    TORCH_CHECK(stamp.signature_identity != 0 && stamp.build_generation != 0,
                "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 optimized DMA "
                "requires a signed Graph generation");
}

constexpr int64_t PI05_DENOISE_ZERO_GATED_RESIDUAL_SITE = 6975132819647256073LL;

enum class Pi05DenoiseAllReduceRoute : int64_t {
    PREPARE_RING_INPUT = 3,
};

enum class Pi05DenoiseRopeRoute : int64_t {
    ROPE_1D = 1,
    FULL_ROPE_TILED = 2,
};

enum class Pi05DenoiseNormRoute : int64_t {
    RMS_THEN_SHIFT = 1,
    FUSED_H1024 = 2,
};

enum class Pi05DenoiseResidualRoute : int64_t {
    ZERO_WITH_SEPARATE_GATE = 2,
    ZERO_WITH_FUSED_GATE_H1024 = 3,
    ZERO_WITH_XOR3_GATED_ATTN_H1024 = 4,
};

enum class Pi05DenoiseMutableDmaRoute : int64_t {
    DDR_BROADCAST_TO_SPM = 1,
    DDR_SCATTER_TO_SPM = 2,
    SPM_COPY_TO_DDR = 3,
    DDR_BROADCAST_TABLE_ONCE = 4,
    SPM_COPY_FINAL_BODY_TO_DDR = 6,
    DDR_BROADCAST_TABLE_BODY = 7,
};

enum class Pi05DenoiseMaskScheduleRoute : int64_t {
    FIRST_BODY_FIRST_LAYER = 1,
};

int64_t pi05_denoise_ring_route(int64_t rows, int64_t cols, int cores) {
    return fmb_ring_all_reduce_route_selector(rows, cols, cores);
}

void check_pi05_scale_lists(
    const char* ctx,
    int64_t n_layers,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w,
    at::TensorList o_w, at::TensorList gate_w, at::TensorList up_w,
    at::TensorList down_w,
    at::TensorList q_ws, at::TensorList k_ws, at::TensorList v_ws,
    at::TensorList o_ws, at::TensorList gate_ws, at::TensorList up_ws,
    at::TensorList down_ws, bool nvfp4)
{
    const bool has_scale = !q_ws.empty();
    auto check_scale_list = [&](const at::TensorList& list, const char* name) {
        if (has_scale) {
            TORCH_CHECK(static_cast<int64_t>(list.size()) == n_layers,
                        ctx, ": ", name, ".size()=", list.size(),
                        " != num_layers=", n_layers);
        } else {
            TORCH_CHECK(list.empty(),
                        ctx, ": ", name,
                        " must be empty unless all quantized scale lists are provided");
        }
    };
    check_scale_list(k_ws,    "k_w_scale");
    check_scale_list(v_ws,    "v_w_scale");
    check_scale_list(o_ws,    "o_w_scale");
    check_scale_list(gate_ws, "gate_scale");
    check_scale_list(up_ws,   "up_scale");
    check_scale_list(down_ws, "down_scale");
    if (has_scale) {
        TORCH_CHECK(static_cast<int64_t>(q_ws.size()) == n_layers,
                    ctx, ": q_w_scale.size()=", q_ws.size(),
                    " != num_layers=", n_layers);
    }

    auto check_weight_dtype = [&](const at::Tensor& w,
                                  const at::Tensor& scale,
                                  const char* name,
                                  int64_t i) {
        TORCH_CHECK(w.device().type() == at::kPrivateUse1 && w.is_contiguous(),
                    ctx, ": ", name, "[", i, "] must be contiguous RPU tensor");
        TORCH_CHECK(w.scalar_type() == at::kHalf || w.scalar_type() == at::kChar
                    || w.scalar_type() == at::kByte,
                    ctx, ": ", name, "[", i, "] must be fp16 / int8 / packed-uint8, got ",
                    w.scalar_type());
        if (has_scale) {
            TORCH_CHECK(w.scalar_type() == at::kChar || w.scalar_type() == at::kByte,
                        ctx, ": quantized mode requires int8(W8A16) or uint8(packed-INT4) ",
                        name, "[", i, "], got ", w.scalar_type());
            TORCH_CHECK(scale.defined() && (scale.scalar_type() == at::kHalf || (nvfp4 && scale.scalar_type() == at::kByte))
                        && scale.device().type() == at::kPrivateUse1
                        && scale.is_contiguous(),
                        ctx, ": ", name, "_scale[", i,
                        "] must be a contiguous fp16 RPU tensor");
            if (w.scalar_type() == at::kChar) {
                TORCH_CHECK(scale.scalar_type() == at::kHalf && scale.dim() == 1 && scale.numel() == w.size(0),
                            ctx, ": W8A16 ", name, "_scale[", i,
                            "] must be per-channel [N=", w.size(0), "]");
            } else if (nvfp4) {
                TORCH_CHECK(scale.scalar_type() == at::kByte && scale.dim() == 2 &&
                                scale.size(0) == w.size(1) / 8 && scale.size(1) == w.size(0),
                            "Pi05 NVFP4 v2 requires FP8 striped [K/16,N] scales");
            } else {
                TORCH_CHECK(scale.dim() == 2 &&
                                (scale.size(0) == 32 || scale.size(0) == 64 ||
                                 scale.size(0) == 128),
                            ctx, ": W4A16 ", name, "_scale[", i,
                            "] must be a controller-striped pgrp 2D payload");
            }
        } else {
            TORCH_CHECK(w.scalar_type() != at::kChar && w.scalar_type() != at::kByte,
                        ctx, ": quantized ", name, "[", i,
                        "] requires scale lists");
        }
    };

    for (int64_t i = 0; i < n_layers; ++i) {
        if (nvfp4) {
            TORCH_CHECK(has_scale && q_w[i].scalar_type() == at::kByte &&
                o_w[i].scalar_type() == at::kByte && gate_w[i].scalar_type() == at::kByte &&
                up_w[i].scalar_type() == at::kByte && down_w[i].scalar_type() == at::kByte &&
                k_w[i].scalar_type() == at::kChar && v_w[i].scalar_type() == at::kChar,
                "Pi05 NVFP4 v2 scope must be Q/O/Gate/Up/Down FP4, K/V W8");
        }
        check_weight_dtype(q_w[i],    has_scale ? q_ws[i]    : at::Tensor(), "q_w", i);
        check_weight_dtype(k_w[i],    has_scale ? k_ws[i]    : at::Tensor(), "k_w", i);
        check_weight_dtype(v_w[i],    has_scale ? v_ws[i]    : at::Tensor(), "v_w", i);
        check_weight_dtype(o_w[i],    has_scale ? o_ws[i]    : at::Tensor(), "o_w", i);
        check_weight_dtype(gate_w[i], has_scale ? gate_ws[i] : at::Tensor(), "gate_w", i);
        check_weight_dtype(up_w[i],   has_scale ? up_ws[i]   : at::Tensor(), "up_w", i);
        check_weight_dtype(down_w[i], has_scale ? down_ws[i] : at::Tensor(), "down_w", i);
    }
}

}  // namespace

// ============================================================================
// set_weights — D-503: invalidate_model_state() must be LAST non-empty stmt.
// ============================================================================
namespace {
constexpr int64_t PI05_DENOISE_Q_EPILOGUE_POLICY_SITE = 0x5049355152504f4cLL;
constexpr int64_t PI05_DENOISE_Q_EPILOGUE_LINEAR_SITE = 0x50493551524c494eLL;
constexpr int64_t PI05_DENOISE_Q_PAIR16_ROPE = 6;

bool pi05_q_rope_tables_admitted(const at::Tensor& cos,const at::Tensor& sin) {
    const auto tensor=[](const at::Tensor& t) {
        return t.defined() && t.device().type()==at::kPrivateUse1 &&
            t.scalar_type()==at::kHalf && t.layout()==c10::Layout::Strided &&
            t.is_contiguous() && t.dim()==2 && t.size(1)==128;
    };
    return tensor(cos) && tensor(sin) && cos.sizes()==sin.sizes() &&
        cos.size(0)>=50 && cos.size(0)<=UINT32_MAX/256;
}

std::vector<int64_t> pi05_q_rope_policy_arguments(
    bool enabled, int64_t prefix, int64_t steps = 10) {
    // Policy ABI, cold flag, steps, P, M, Nlocal, K, D, TP.
    return {1,enabled?1:0,steps,prefix,50,256,1024,256,8};
}
} // namespace

std::vector<int64_t> Pi05DenoiseStepModel::pi05_q_rope_arguments(int64_t logical_position) const {

    return {logical_position,1,10,planned_prefix_len_,50,256,1024,256,8,1,0,592,32,128,16,8,
            cos_.defined() && cos_.dim()==2 ? cos_.size(0) : 0,256};
}

bool Pi05DenoiseStepModel::use_pi05_q_rope_epilogue(int64_t rows,int64_t logical_position) const {
    if (!pi05_q_rope_epilogue_opt_in_ || planned_num_steps_ != 10 ||
        !pi05_q_rope_tables_owned_ ||
        !use_pi05_graph_dma_profile(rows) || layer_weights_.size()!=18 ||
        !pi05_q_rope_tables_admitted(cos_,sin_) || logical_position<0 ||
        logical_position>cos_.size(0) || 50>cos_.size(0)-logical_position) return false;
    return std::all_of(layer_weights_.begin(),layer_weights_.end(),[](const LayerWeights& w) {
        return w.q_w.defined() && w.q_w.device().type()==at::kPrivateUse1 &&
            w.q_w.scalar_type()==at::kChar && w.q_w.is_contiguous() &&
            w.q_w.sizes()==at::IntArrayRef({2048,1024}) &&
            w.q_ws.defined() && w.q_ws.device().type()==at::kPrivateUse1 &&
            w.q_ws.scalar_type()==at::kHalf && w.q_ws.is_contiguous() &&
            w.q_ws.sizes()==at::IntArrayRef({2048});
    });
}

void Pi05DenoiseStepModel::validate_pi05_q_rope_manifest(
    const FmbPhysicalExecutionManifest& manifest) const {
    const int64_t logical_position=rope_position_>=0 ? rope_position_ : planned_prefix_len_;
    if (pi05_q_rope_epilogue_opt_in_)
        TORCH_CHECK(use_pi05_q_rope_epilogue(manifest.physical_length,logical_position) &&
            manifest.state==FmbPhysicalManifestState::COMPLETE &&
            manifest.logical_length==50 && manifest.execution_padding_rows==0 &&
            manifest.kv_logical_length == planned_prefix_len_ + 50 &&
            manifest.graph_lifecycle==FmbGraphLifecycle::COMPOSITE_CHILD &&
            manifest.linear_accumulation==linear_accumulation_policy(),
            "Pi Q-RoPE requires exact owned W8 ten-step COMPLETE P800/M50 profile");
    size_t policies=0,fusions=0,paired_k_rope=0;
    for (const auto& route:manifest.routes) {
        if (route.site_id==PI05_DENOISE_Q_EPILOGUE_POLICY_SITE) {
            ++policies;
            TORCH_CHECK(route.family==FmbRouteFamily::GRAPH_SCHEDULE && route.selector==1 &&
                route.flags==0 && route.invocation==0 &&
                route.arguments==pi05_q_rope_policy_arguments(pi05_q_rope_epilogue_opt_in_, planned_prefix_len_, planned_num_steps_),
                "Pi Q-RoPE cold policy mismatch");
        } else if (route.site_id==PI05_DENOISE_Q_EPILOGUE_LINEAR_SITE) {
            ++fusions;
            TORCH_CHECK(pi05_q_rope_epilogue_opt_in_ && route.family==FmbRouteFamily::LINEAR &&
                route.selector==PI05_DENOISE_Q_PAIR16_ROPE && route.flags==0 && route.invocation==0 &&
                route.arguments==pi05_q_rope_arguments(logical_position),
                "Pi Q-RoPE fused physical route or logical position mismatch");
        } else if (pi05_q_rope_epilogue_opt_in_) {
            TORCH_CHECK(route.site_id!=PI05_DENOISE_Q_LINEAR_SITE &&
                route.site_id!=PI05_DENOISE_Q_ROPE_SITE,
                "Pi Q-RoPE forbids stale standalone Q Linear/RoPE routes");
            if (route.site_id==PI05_DENOISE_K_ROPE_SITE) {
                ++paired_k_rope;
                TORCH_CHECK(route.family==FmbRouteFamily::ROPE && route.invocation==0 &&
                    !route.arguments.empty() && route.arguments.front()==logical_position,
                    "Pi Q/K routes disagree on frozen logical RoPE position");
            }
        }
    }
    TORCH_CHECK(policies==1 && fusions==(pi05_q_rope_epilogue_opt_in_?1:0),
                "Pi Q-RoPE policy/fusion route missing or duplicated");
    TORCH_CHECK(!pi05_q_rope_epilogue_opt_in_ || paired_k_rope==1,
                "Pi Q-RoPE is missing its original K-RoPE invocation");
}

void Pi05DenoiseStepModel::validate_pi05_runtime_policies(
    at::IntArrayRef descriptor) const {
    // All policies inspect the same immutable input within this call. Decode
    // once, but validate afresh on every forward (including READY replay): no
    // descriptor, installed-weight or cold-profile checks are cached away.
    const std::optional<FmbPrefillStageCandidate> decoded = descriptor.empty()
        ? std::nullopt
        : std::make_optional(decode_fmb_prefill_stage_candidate(descriptor));
    const auto* candidate = decoded ? &*decoded : nullptr;
    validate_pi05_q_rope_policy(candidate);
    validate_pi05_w4_fastpath_policy(candidate);
    validate_pi05_xor3_gated_attn_policy(candidate);
    validate_pi05_kv1_stripe_gather_policy(candidate);
    validate_pi05_kv1_direct_cache_policy(candidate);
    validate_pi05_w4_geglu_m50_policy(candidate);
    validate_pi05_nvfp4_geglu_m50_policy(candidate);
    validate_pi05_geglu_m50_policy(candidate);
    validate_pi05_xor3_gated_mlp_policy(candidate);
}

void Pi05DenoiseStepModel::validate_pi05_q_rope_policy(const FmbPrefillStageCandidate* candidate) const {
    if (candidate == nullptr) {
        TORCH_CHECK(!pi05_q_rope_epilogue_opt_in_,
                    "Pi Q-RoPE requires a complete signed loop descriptor");
        return; // unchanged descriptor-free cold-OFF legacy route
    }
    validate_pi05_q_rope_manifest(candidate->physical_manifest);
    if (pi05_q_rope_epilogue_opt_in_) {
        TORCH_CHECK(pi05_graph_dma_shape_admitted(candidate->stage_plan,
            candidate->physical_manifest.physical_length,candidate->physical_manifest.logical_length,
            planned_prefix_len_,1,true,false,AttentionExecutionPolicy::DDR_KV),
            "Pi Q-RoPE descriptor chunk schedule mismatch");
        pi05_require_stable_graph_dma_capture();
    }
}

bool Pi05DenoiseStepModel::use_pi05_kv1_stripe_gather(int64_t rows) const {
    if (!pi05_kv1_stripe_gather_opt_in_ || !planned_route_profile_valid_ ||
        !planned_loop_mode_ || planned_num_steps_ != 10 ||
        !pi05_action_prefix_admitted(planned_prefix_len_) || rows != 50 || chunk_size_ != 50 ||
        hidden_size() != 1024 || intermediate_size() != 4096 ||
        num_layers() != 18 || num_q_heads() != 8 || num_kv_heads() != 8 ||
        head_dim() != 256 || attn_tp() != 8 || !pi05_k_rope_insert_opt_in_ ||
        !pi05_k_rope_insert_w8a16_ || planned_kv_physical_rows_ != 64 ||
        planned_k_rope_cache_capacity_ != 2048 ||
        layer_weights_.size() != 18) {
        return false;
    }
    return std::all_of(
        layer_weights_.begin(), layer_weights_.end(),
        [](const LayerWeights& weights) {
            const auto weight_ok = [](const at::Tensor& tensor) {
                return tensor.defined() &&
                    tensor.device().type() == at::kPrivateUse1 &&
                    tensor.scalar_type() == at::kChar && tensor.is_contiguous() &&
                    tensor.sizes() == at::IntArrayRef({256, 1024}) &&
                    tensor.nbytes() == 256 * 1024;
            };
            const auto scale_ok = [](const at::Tensor& tensor) {
                return tensor.defined() &&
                    tensor.device().type() == at::kPrivateUse1 &&
                    tensor.scalar_type() == at::kHalf && tensor.is_contiguous() &&
                    tensor.sizes() == at::IntArrayRef({256}) &&
                    tensor.nbytes() == 256 * sizeof(c10::Half);
            };
            return weight_ok(weights.k_kv1_w) && weight_ok(weights.v_kv1_w) &&
                scale_ok(weights.k_kv1_ws) && scale_ok(weights.v_kv1_ws);
        });
}

void Pi05DenoiseStepModel::validate_pi05_kv1_stripe_gather_manifest(
    const FmbPhysicalExecutionManifest& manifest) const {
    const bool enabled = use_pi05_kv1_stripe_gather(manifest.physical_length);
    if (pi05_kv1_stripe_gather_opt_in_) {
        TORCH_CHECK(
            enabled && manifest.state == FmbPhysicalManifestState::COMPLETE &&
                manifest.logical_length == 50 &&
                manifest.execution_padding_rows == 0 &&
                manifest.kv_logical_length == planned_prefix_len_ + 50 &&
                manifest.graph_lifecycle == FmbGraphLifecycle::COMPOSITE_CHILD &&
                manifest.linear_accumulation == linear_accumulation_policy(),
            "Pi KV1 stripe-gather requires exact W8A16 ten-step P800/M50 owner");
    }

    size_t policies = 0, compact_k = 0, compact_v = 0;
    size_t gather_k = 0, gather_v = 0, gathered_sources = 0;
    for (const auto& route : manifest.routes) {
        if (route.site_id == PI05_DENOISE_KV1_POLICY_SITE) {
            ++policies;
            TORCH_CHECK(
                route.family == FmbRouteFamily::GRAPH_SCHEDULE &&
                    route.selector == 1 && route.flags == 0 &&
                    route.invocation == 0 &&
                    route.arguments == pi05_kv1_policy_arguments(
                        pi05_kv1_stripe_gather_opt_in_, planned_prefix_len_),
                "Pi KV1 cold policy route drifted");
        } else if (route.site_id == PI05_DENOISE_K_LINEAR_SITE &&
                   route.selector == PI05_DENOISE_KV1_FIXED_M592N32) {
            ++compact_k;
            TORCH_CHECK(enabled && route.family == FmbRouteFamily::LINEAR &&
                            route.flags == 0 && route.invocation == 0 &&
                            route.arguments == pi05_kv1_linear_arguments(planned_prefix_len_),
                        "Pi KV1 compact K route drifted");
        } else if (route.site_id == PI05_DENOISE_V_LINEAR_SITE &&
                   route.selector == PI05_DENOISE_KV1_FIXED_M592N32) {
            ++compact_v;
            TORCH_CHECK(enabled && route.family == FmbRouteFamily::LINEAR &&
                            route.flags == 0 && route.invocation == 0 &&
                            route.arguments == pi05_kv1_linear_arguments(planned_prefix_len_),
                        "Pi KV1 compact V route drifted");
        } else if (route.site_id == PI05_DENOISE_KV1_K_GATHER_SITE) {
            ++gather_k;
            TORCH_CHECK(enabled && route.family == FmbRouteFamily::COLLECTIVE &&
                            route.selector == static_cast<int64_t>(
                                RpuAllGatherSchedule::LITTLE_CHUNK) &&
                            route.flags == 0 && route.invocation == 0 &&
                            route.arguments == pi05_kv1_gather_arguments(planned_prefix_len_),
                        "Pi KV1 K gather route drifted");
        } else if (route.site_id == PI05_DENOISE_KV1_V_GATHER_SITE) {
            ++gather_v;
            TORCH_CHECK(enabled && route.family == FmbRouteFamily::COLLECTIVE &&
                            route.selector == static_cast<int64_t>(
                                RpuAllGatherSchedule::LITTLE_CHUNK) &&
                            route.flags == 0 && route.invocation == 0 &&
                            route.arguments == pi05_kv1_gather_arguments(planned_prefix_len_),
                        "Pi KV1 V gather route drifted");
        } else if (route.site_id == PI05_DENOISE_K_ROPE_SITE ||
                   route.site_id == PI05_DENOISE_KV_INSERT_SITE) {
            if (route.flags & PI05_DENOISE_KV1_GATHERED_SOURCE_FLAG) {
                ++gathered_sources;
                TORCH_CHECK(enabled,
                            "Pi KV1 gathered-source flag reached a cold OFF owner");
            }
        }
    }
    TORCH_CHECK(
        policies == (pi05_kv1_stripe_gather_opt_in_ ? 1 : 0) &&
            compact_k == (enabled ? 1 : 0) &&
            compact_v == (enabled ? 1 : 0) && gather_k == (enabled ? 1 : 0) &&
            gather_v == (enabled ? 1 : 0) &&
            gathered_sources == (enabled ? 2 : 0),
        "Pi KV1 physical descriptor is incomplete or duplicated");
}

void Pi05DenoiseStepModel::validate_pi05_kv1_stripe_gather_policy(
    const FmbPrefillStageCandidate* candidate) const {
    if (candidate == nullptr) {
        TORCH_CHECK(!pi05_kv1_stripe_gather_opt_in_,
                    "Pi KV1 stripe-gather requires a complete signed loop descriptor");
        return;
    }
    validate_pi05_kv1_stripe_gather_manifest(candidate->physical_manifest);
    if (pi05_kv1_stripe_gather_opt_in_) {
        TORCH_CHECK(
            pi05_graph_dma_shape_admitted(
                candidate->stage_plan, candidate->physical_manifest.physical_length,
                candidate->physical_manifest.logical_length, planned_prefix_len_,
                1, true, false, AttentionExecutionPolicy::DDR_KV),
            "Pi KV1 stripe-gather descriptor chunk schedule drifted");
        pi05_require_stable_graph_dma_capture();
    }
}

bool Pi05DenoiseStepModel::use_pi05_kv1_direct_cache(int64_t rows) const {
    const int64_t logical_position =
        rope_position_ >= 0 ? rope_position_ : planned_prefix_len_;
    if (!pi05_kv1_direct_cache_opt_in_ ||
        (pi05_kv1_pair_owner_opt_in_
             ? !pi05_kv1_pair_owner_full_owners_
             : !pi05_kv1_direct_pair_owners_) ||
        !planned_route_profile_valid_ || !planned_loop_mode_ ||
        !pi05_action_steps_admitted(planned_num_steps_) ||
        (planned_num_steps_ != 10 &&
         (!pi05_kv1_pair_owner_opt_in_ || !use_spm_adarms_table())) ||
        !pi05_action_prefix_admitted(planned_prefix_len_) ||
        rows != 50 || chunk_size_ != 50 || hidden_size() != 1024 ||
        intermediate_size() != 4096 || max_action_dim_ != 32 ||
        num_layers() != 18 || num_q_heads() != 8 || num_kv_heads() != 8 ||
        head_dim() != 256 || attn_tp() != 8 ||
        !pi05_projection_fastpath_admitted() || planned_kv_physical_rows_ != 64 ||
        planned_k_rope_cache_capacity_ != 2048 ||
        layer_weights_.size() != 18 ||
        !pi05_q_rope_tables_admitted(cos_, sin_) ||
        cos_.storage_offset() != 0 || sin_.storage_offset() != 0 ||
        logical_position < 0 || logical_position > cos_.size(0) ||
        50 > cos_.size(0) - logical_position) {
        return false;
    }
    const bool fp16 = pi05_fp16_projection_owners_admitted(layer_weights_);
    if (fp16) {
        // The compact companion producer remains W8-only. FP16 uses its
        // independent full-owner program with no fabricated quantization scales.
        return pi05_kv1_pair_owner_opt_in_ &&
            KernelCache::instance().has_loaded(
                KernelId::PI05_DENOISE_KV1_PAIR_OWNER_FP16_M50N32X2K1024);
    }
    return std::all_of(
        layer_weights_.begin(), layer_weights_.end(),
        [this](const LayerWeights& weights) {
            const int64_t owner_n =
                pi05_kv1_pair_owner_opt_in_ ? 2048 : 256;
            const auto weight_ok = [owner_n](const at::Tensor& tensor) {
                return tensor.defined() &&
                    tensor.device().type() == at::kPrivateUse1 &&
                    tensor.scalar_type() == at::kChar && tensor.is_contiguous() &&
                    tensor.storage_offset() == 0 &&
                    tensor.sizes() == at::IntArrayRef({owner_n, 1024}) &&
                    tensor.nbytes() == owner_n * 1024;
            };
            const auto scale_ok = [owner_n](const at::Tensor& tensor) {
                return tensor.defined() &&
                    tensor.device().type() == at::kPrivateUse1 &&
                    tensor.scalar_type() == at::kHalf && tensor.is_contiguous() &&
                    tensor.storage_offset() == 0 &&
                    tensor.sizes() == at::IntArrayRef({owner_n}) &&
                    tensor.nbytes() == owner_n * sizeof(c10::Half);
            };
            if (pi05_kv1_pair_owner_opt_in_) {
                return weight_ok(weights.k_w) && weight_ok(weights.v_w) &&
                    scale_ok(weights.k_ws) && scale_ok(weights.v_ws);
            }
            return weight_ok(weights.k_kv1_w) &&
                weight_ok(weights.v_kv1_w) &&
                scale_ok(weights.k_kv1_ws) &&
                scale_ok(weights.v_kv1_ws);
        });
}

bool Pi05DenoiseStepModel::use_pi05_kv1_pair_owner(int64_t rows) const {
    return pi05_kv1_pair_owner_opt_in_ &&
        use_pi05_kv1_direct_cache(rows);
}

int64_t Pi05DenoiseStepModel::pi05_w4_geglu_m50_selector() const {
    return pi05_w4_geglu_n64_opt_in_ ? PI05_DENOISE_W4_GEGLU_M50_M464N64K128
                                   : PI05_DENOISE_W4_GEGLU_M50_M512N48K128;
}

std::vector<int64_t> Pi05DenoiseStepModel::pi05_w4_geglu_m50_arguments() const {
    const int64_t gs = !layer_weights_.empty() && layer_weights_[0].q_ws.defined() &&
            layer_weights_[0].q_ws.dim() == 2 ? layer_weights_[0].q_ws.size(0) : 0;
    const int64_t local_groups = gs > 0 ? 1024 / gs : 0;
    const int64_t scale_elements = ((local_groups + 3) / 4) * 8 * 8 * 4 * 64;
    // Independent W4 ABI: packed nibble weights, FP16 activation/scales,
    // original ACC16 Gate/GELU/Up/MUL retirement, and the complete owner profile.
    // Bind tile/grid/register count, four shift8 DDR pairs, GS registers,
    // both physical packed owner layouts, scale block stride/alignment,
    // and absolute SPM roles (residual1 input / gate output), with no extra slab.
    std::vector<int64_t> arguments{
        1, pi05_w4_geglu_m50_opt_in_ ? 1 : 0,
        static_cast<int64_t>(KernelId::PI05_DENOISE_GATE_UP_GEGLU_WINT4A16_PGRP_M512N48K128),
        4, 16, 16, 1, 1, 1, 1,
        512, 48, 128, 68, 11, 1, 1,
        2, 3, 20, 21, 32, 33, 34, 35, 8,
        24, gs, 25, local_groups, 26, 27, 32768,
        4096, 512, 2097152, 4096, 512, 2097152,
        gs, gs > 0 ? scale_elements / gs : 0, scale_elements, scale_elements * 2,
        256, 4096, 256,
        1, 102400, 2, 51200, 0, 0, 2048, 1024, 1, 1, 0,
        static_cast<int64_t>(FmbLinearAccumulationPolicy::ACC16),
        static_cast<int64_t>(FmbGraphLifecycle::COMPOSITE_CHILD)};
    const auto profile = pi05_w4_fastpath_arguments();
    arguments.insert(arguments.end(), profile.begin(), profile.end());
    if (pi05_w4_geglu_n64_opt_in_) {
        // ABI2 is the separately admitted GS128 M464/N64 producer. Keep the
        // complete ABI1 payload byte-for-byte when the N64 cold flag is off.
        arguments[0] = 2;
        arguments[2] = static_cast<int64_t>(
            KernelId::PI05_DENOISE_GATE_UP_GEGLU_WINT4A16_PGRP_M464N64K128);
        arguments[10] = 464;  // tile M
        arguments[11] = 64;   // tile N
        arguments[14] = 8;    // grid X
        arguments.insert(arguments.end(), {
            pi05_w4_geglu_n64_opt_in_ ? 1 : 0, 128});
    }
    return arguments;
}

bool Pi05DenoiseStepModel::use_pi05_w4_geglu_m50(int64_t rows) const {
    // Reuse the complete five-projection physical W4 proof and W8 K/V owners.
    // This is a separate producer; uint8 owners never enter W8 GeGLU admission.
    return planned_num_steps_ == 10 &&
        pi05_w4_geglu_m50_opt_in_ && pi05_w4_fastpath_opt_in_ &&
        !nvfp4_.enabled() && !pi05_geglu_m50_opt_in_ && use_pi05_xor3_gated_mlp(rows) &&
        (!pi05_w4_geglu_n64_opt_in_ || layer_weights_[0].q_ws.size(0) == 128);
}

void Pi05DenoiseStepModel::validate_pi05_w4_geglu_m50_manifest(
    const FmbPhysicalExecutionManifest& manifest) const {
    const bool enabled = use_pi05_w4_geglu_m50(manifest.physical_length);
    if (pi05_w4_geglu_m50_opt_in_) {
        TORCH_CHECK(
            enabled && manifest.state == FmbPhysicalManifestState::COMPLETE &&
                manifest.logical_length == 50 && manifest.physical_length == 50 &&
                manifest.execution_padding_rows == 0 &&
                manifest.kv_logical_length == planned_prefix_len_ + 50 &&
                manifest.kv_insert_physical_rows == 64 &&
                manifest.graph_lifecycle == FmbGraphLifecycle::COMPOSITE_CHILD &&
                manifest.linear_accumulation == linear_accumulation_policy(),
            "Pi W4 M50 GeGLU requires exact COMPLETE ten-step W4 pgrp owners");
    }
    if (pi05_w4_geglu_m50_opt_in_) {
        validate_pi05_w4_fastpath_manifest(manifest);
        validate_pi05_xor3_gated_mlp_manifest(manifest);
    }
    size_t policies = 0;
    size_t linears = 0;
    for (const auto& route : manifest.routes) {
        TORCH_CHECK(!pi05_w4_geglu_m50_opt_in_ ||
                        (route.site_id != PI05_DENOISE_GEGLU_M50_POLICY_SITE &&
                         route.site_id != PI05_DENOISE_GEGLU_M50_LINEAR_SITE &&
                         (route.family != FmbRouteFamily::LINEAR ||
                          route.selector != PI05_DENOISE_GEGLU_M50_M512N48K128)),
                    "Pi W4 GeGLU cannot consume the W8 producer or policy");
        TORCH_CHECK(
            route.family != FmbRouteFamily::LINEAR ||
                (route.selector != PI05_DENOISE_W4_GEGLU_M50_M512N48K128 &&
                 route.selector != PI05_DENOISE_W4_GEGLU_M50_M464N64K128) ||
                route.site_id == PI05_DENOISE_W4_GEGLU_M50_LINEAR_SITE,
            "Pi W4 M50 GeGLU selector escaped its single producer site");
        if (route.site_id == PI05_DENOISE_W4_GEGLU_M50_POLICY_SITE) {
            ++policies;
            TORCH_CHECK(
                pi05_w4_geglu_m50_opt_in_ &&
                    route.family == FmbRouteFamily::GRAPH_SCHEDULE &&
                    route.selector == 1 && route.flags == 0 &&
                    route.invocation == 0 &&
                    route.arguments == pi05_w4_geglu_m50_arguments(),
                "Pi W4 M50 GeGLU cold policy route drifted");
        } else if (route.site_id == PI05_DENOISE_W4_GEGLU_M50_LINEAR_SITE) {
            ++linears;
            TORCH_CHECK(
                enabled && route.family == FmbRouteFamily::LINEAR &&
                    route.selector == pi05_w4_geglu_m50_selector() &&
                    route.flags == 0 && route.invocation == 0 &&
                    route.arguments == pi05_w4_geglu_m50_arguments(),
                "Pi W4 M50 GeGLU producer route drifted or escaped its cold owner");
        }
    }
    TORCH_CHECK(
        policies == (pi05_w4_geglu_m50_opt_in_ ? 1 : 0) &&
            linears == (enabled ? 1 : 0),
        "Pi W4 M50 GeGLU descriptor is incomplete or duplicated");
}

void Pi05DenoiseStepModel::validate_pi05_w4_geglu_m50_policy(
    const FmbPrefillStageCandidate* candidate) const {
    if (candidate == nullptr) {
        TORCH_CHECK(!pi05_w4_geglu_m50_opt_in_,
                    "Pi W4 M50 GeGLU requires a complete signed loop descriptor");
        return;
    }
    validate_pi05_w4_geglu_m50_manifest(candidate->physical_manifest);
    if (pi05_w4_geglu_m50_opt_in_) {
        TORCH_CHECK(
            pi05_graph_dma_shape_admitted(
                candidate->stage_plan,
                candidate->physical_manifest.physical_length,
                candidate->physical_manifest.logical_length,
                planned_prefix_len_, 1, true, false,
                AttentionExecutionPolicy::DDR_KV),
            "Pi W4 M50 GeGLU descriptor chunk schedule drifted");
        pi05_require_stable_graph_dma_capture();
    }
}

std::vector<int64_t> Pi05DenoiseStepModel::pi05_nvfp4_geglu_m50_arguments() const {
    // ABI2 binds the separate ACC16 producer and its FP16 K128 retirement.
    // Register/storage geometry is unchanged from the historical ACC32 ABI1.
    std::vector<int64_t> args{2, pi05_nvfp4_geglu_m50_opt_in_ ? 1 : 0,
        static_cast<int64_t>(KernelId::PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M320N64K128),
        50, 4096, 1024, 512, 8, 320, 64, 128, 68, 8, 1, 1,
        2, 3, 20, 21, 32, 33, 34, 35, 24, 25, 36, 37, 27,
        16, 8, 32, 24, 18, 2, 16, 16, 16, 16,
        102400, 51200, 0, 8192, 2048, 1024};
    const auto profile = pi05_w4_fastpath_arguments();
    args.insert(args.end(), profile.begin(), profile.end());
    return args;
}

bool Pi05DenoiseStepModel::use_pi05_nvfp4_geglu_m50(int64_t rows) const {
    return !linear_acc32_ && pi05_nvfp4_geglu_m50_opt_in_ && nvfp4_.enabled() &&
        num_cores() == 8 && attn_tp() == 8 && mlp_tp() == 8 && rows == 50 &&
        planned_num_steps_ == 10 &&
        (planned_prefix_len_ == 544 || planned_prefix_len_ == 800) &&
        pi05_nvfp4_weight_profile_admitted() && use_pi05_xor3_gated_mlp(rows) &&
        KernelCache::instance().has_loaded(
            KernelId::PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M320N64K128);
}

void Pi05DenoiseStepModel::validate_pi05_nvfp4_geglu_m50_manifest(
    const FmbPhysicalExecutionManifest& manifest) const {
    const bool enabled = use_pi05_nvfp4_geglu_m50(manifest.physical_length);
    if (pi05_nvfp4_geglu_m50_opt_in_) {
        TORCH_CHECK(enabled && manifest.state == FmbPhysicalManifestState::COMPLETE &&
            manifest.logical_length == 50 && manifest.physical_length == 50 &&
            manifest.execution_padding_rows == 0 && manifest.kv_logical_length == planned_prefix_len_ + 50 &&
            manifest.kv_insert_physical_rows == 64 &&
            manifest.graph_lifecycle == FmbGraphLifecycle::COMPOSITE_CHILD &&
            manifest.linear_accumulation == linear_accumulation_policy(),
            "Pi NVFP4 GeGLU requires exact COMPLETE striped-v2 Action owners");
        validate_pi05_w4_fastpath_manifest(manifest);
        validate_pi05_xor3_gated_mlp_manifest(manifest);
    }
    size_t policies = 0, linears = 0;
    for (const auto& route : manifest.routes) {
        TORCH_CHECK(!pi05_nvfp4_geglu_m50_opt_in_ ||
            (route.site_id != PI05_DENOISE_W4_GEGLU_M50_POLICY_SITE &&
             route.site_id != PI05_DENOISE_W4_GEGLU_M50_LINEAR_SITE &&
             route.site_id != PI05_DENOISE_GEGLU_M50_POLICY_SITE &&
             route.site_id != PI05_DENOISE_GEGLU_M50_LINEAR_SITE),
            "Pi NVFP4 GeGLU cannot consume INT4/W8 producers");
        TORCH_CHECK(route.family != FmbRouteFamily::LINEAR ||
            (route.selector != PI05_DENOISE_NVFP4_GEGLU_M320N64K128 &&
             route.selector != PI05_DENOISE_NVFP4_GEGLU_ACC16_M320N64K128) ||
            route.site_id == PI05_DENOISE_NVFP4_GEGLU_LINEAR_SITE,
            "Pi NVFP4 GeGLU selector escaped its producer site");
        if (route.site_id == PI05_DENOISE_NVFP4_GEGLU_POLICY_SITE) {
            ++policies;
            TORCH_CHECK(pi05_nvfp4_geglu_m50_opt_in_ &&
                route.family == FmbRouteFamily::GRAPH_SCHEDULE && route.selector == 1 &&
                route.flags == 0 && route.invocation == 0 &&
                route.arguments == pi05_nvfp4_geglu_m50_arguments(),
                "Pi NVFP4 GeGLU policy drifted or escaped its cold owner");
        } else if (route.site_id == PI05_DENOISE_NVFP4_GEGLU_LINEAR_SITE) {
            ++linears;
            TORCH_CHECK(enabled && route.family == FmbRouteFamily::LINEAR &&
                route.selector == PI05_DENOISE_NVFP4_GEGLU_ACC16_M320N64K128 &&
                route.flags == 0 && route.invocation == 0 &&
                route.arguments == pi05_nvfp4_geglu_m50_arguments(),
                "Pi NVFP4 GeGLU producer drifted or escaped its cold owner");
        }
    }
    TORCH_CHECK(policies == (pi05_nvfp4_geglu_m50_opt_in_ ? 1 : 0) &&
                linears == (enabled ? 1 : 0), "Pi NVFP4 GeGLU descriptor incomplete/duplicated");
}

void Pi05DenoiseStepModel::validate_pi05_nvfp4_geglu_m50_policy(
    const FmbPrefillStageCandidate* candidate) const {
    if (candidate == nullptr) {
        TORCH_CHECK(!pi05_nvfp4_geglu_m50_opt_in_, "Pi NVFP4 GeGLU requires a complete descriptor");
        return;
    }
    validate_pi05_nvfp4_geglu_m50_manifest(candidate->physical_manifest);
    if (pi05_nvfp4_geglu_m50_opt_in_) {
        TORCH_CHECK(pi05_graph_dma_shape_admitted(candidate->stage_plan, 50, 50,
            planned_prefix_len_, 1, true, false, AttentionExecutionPolicy::DDR_KV),
            "Pi NVFP4 GeGLU chunk schedule drifted");
        pi05_require_stable_graph_dma_capture();
    }
}

std::vector<int64_t> Pi05DenoiseStepModel::pi05_geglu_m50_arguments() const {
    // ABI, real cold flag, exact denoise profile, kernel/tile/register/grid
    // ABI, four DDR register pairs, byte strides, partition/dtype/bias and
    // input/output/extra SPM bytes. Shared by descriptor and cost identity.
    if (pi05_is_fp16_action(layer_weights_)) {
        // ABI2: two FP16 weight owners, no scale operands, canonical FP16
        // M608/N32 auto-tile family and its original ACC16 retirement.
        return {2, pi05_geglu_m50_opt_in_ ? 1 : 0,
            planned_num_steps_, planned_prefix_len_, 50, 64, planned_prefix_len_ + 50,
            1024, 4096, 512, 18, 8, 8, 256, 8, 2048,
            static_cast<int64_t>(KernelId::PI05_DENOISE_GATE_UP_GEGLU_FP16_M608N32K128),
            608, 32, 128, 68, 16, 1, 1, 0, 2, 3, 24, 25,
            2048, 1024, 1, 0, 0, 102400, 51200, 0, 16, 0};
    }
    return {
        1, pi05_geglu_m50_opt_in_ ? 1 : 0,
        planned_num_steps_, planned_prefix_len_, 50, 64, planned_prefix_len_ + 50, 1024, 4096, 512, 18, 8, 8, 256, 8, 2048,
        static_cast<int64_t>(
            KernelId::PI05_DENOISE_GATE_UP_GEGLU_W8A16_M512N48K128),
        512, 48, 128, 68, 11, 1, 1, 0,
        2, 3, 20, 21, 24, 25, 26, 27,
        2048, 1024, 1, 1, 0, 102400, 51200, 0};
}

bool Pi05DenoiseStepModel::use_pi05_geglu_m50(int64_t rows) const {
    if (!pi05_geglu_m50_opt_in_ || !use_pi05_kv1_pair_owner(rows) ||
        !use_pi05_xor3_gated_attn(rows)) {
        return false;
    }
    if (pi05_is_fp16_action(layer_weights_)) {
        return pi05_fp16_projection_owners_admitted(layer_weights_) &&
            KernelCache::instance().has_loaded(
                KernelId::PI05_DENOISE_GATE_UP_GEGLU_FP16_M608N32K128);
    }
    return std::all_of(
        layer_weights_.begin(), layer_weights_.end(),
        [](const LayerWeights& weights) {
            const auto weight_ok = [](const at::Tensor& tensor) {
                return tensor.defined() &&
                    tensor.device().type() == at::kPrivateUse1 &&
                    tensor.scalar_type() == at::kChar && tensor.is_contiguous() &&
                    tensor.storage_offset() == 0 &&
                    tensor.sizes() == at::IntArrayRef({4096, 1024}) &&
                    tensor.nbytes() == 4096 * 1024;
            };
            const auto scale_ok = [](const at::Tensor& tensor) {
                return tensor.defined() &&
                    tensor.device().type() == at::kPrivateUse1 &&
                    tensor.scalar_type() == at::kHalf && tensor.is_contiguous() &&
                    tensor.storage_offset() == 0 &&
                    tensor.sizes() == at::IntArrayRef({4096}) &&
                    tensor.nbytes() == 4096 * sizeof(c10::Half);
            };
            return weight_ok(weights.gate_proj_w) &&
                weight_ok(weights.up_proj_w) &&
                scale_ok(weights.gate_ws) && scale_ok(weights.up_ws);
        });
}

void Pi05DenoiseStepModel::validate_pi05_geglu_m50_manifest(
    const FmbPhysicalExecutionManifest& manifest) const {
    const bool enabled = use_pi05_geglu_m50(manifest.physical_length);
    if (pi05_geglu_m50_opt_in_) {
        TORCH_CHECK(
            enabled && manifest.state == FmbPhysicalManifestState::COMPLETE &&
                manifest.logical_length == 50 && manifest.physical_length == 50 &&
                manifest.execution_padding_rows == 0 &&
                manifest.kv_logical_length == planned_prefix_len_ + 50 &&
                manifest.kv_insert_physical_rows == 64 &&
                manifest.graph_lifecycle == FmbGraphLifecycle::COMPOSITE_CHILD &&
                manifest.linear_accumulation == linear_accumulation_policy(),
            "Pi M50 GeGLU requires exact COMPLETE four/ten-step FP16/W8A16 owners");
    }
    size_t policies = 0;
    size_t linears = 0;
    for (const auto& route : manifest.routes) {
        TORCH_CHECK(
            route.family != FmbRouteFamily::LINEAR ||
                (route.selector != PI05_DENOISE_GEGLU_M50_M512N48K128 &&
                 route.selector != PI05_DENOISE_GEGLU_FP16_M50_M608N32K128) ||
                route.site_id == PI05_DENOISE_GEGLU_M50_LINEAR_SITE,
            "Pi M50 GeGLU selector escaped its single producer site");
        if (route.site_id == PI05_DENOISE_GEGLU_M50_POLICY_SITE) {
            ++policies;
            TORCH_CHECK(
                pi05_geglu_m50_opt_in_ &&
                    route.family == FmbRouteFamily::GRAPH_SCHEDULE &&
                    route.selector == 1 && route.flags == 0 &&
                    route.invocation == 0 &&
                    route.arguments == pi05_geglu_m50_arguments(),
                "Pi M50 GeGLU cold policy route drifted");
        } else if (route.site_id == PI05_DENOISE_GEGLU_M50_LINEAR_SITE) {
            ++linears;
            TORCH_CHECK(
                enabled && route.family == FmbRouteFamily::LINEAR &&
                    route.selector == pi05_geglu_m50_selector(pi05_is_fp16_action(layer_weights_)) &&
                    route.flags == 0 && route.invocation == 0 &&
                    route.arguments == pi05_geglu_m50_arguments(),
                "Pi M50 GeGLU producer route drifted or escaped its cold owner");
        }
    }
    TORCH_CHECK(
        policies == (pi05_geglu_m50_opt_in_ ? 1 : 0) &&
            linears == (enabled ? 1 : 0),
        "Pi M50 GeGLU descriptor is incomplete or duplicated");
}

void Pi05DenoiseStepModel::validate_pi05_geglu_m50_policy(
    const FmbPrefillStageCandidate* candidate) const {
    if (candidate == nullptr) {
        TORCH_CHECK(!pi05_geglu_m50_opt_in_,
                    "Pi M50 GeGLU requires a complete signed loop descriptor");
        return;
    }
    validate_pi05_geglu_m50_manifest(candidate->physical_manifest);
    if (pi05_geglu_m50_opt_in_) {
        TORCH_CHECK(
            pi05_graph_dma_shape_admitted(
                candidate->stage_plan,
                candidate->physical_manifest.physical_length,
                candidate->physical_manifest.logical_length,
                planned_prefix_len_, 1, true, false,
                AttentionExecutionPolicy::DDR_KV),
            "Pi M50 GeGLU descriptor chunk schedule drifted");
        pi05_require_stable_graph_dma_capture();
    }
}

std::vector<int64_t> Pi05DenoiseStepModel::pi05_xor3_gated_mlp_arguments() const {
    // ABI, actual cold flag, exact profile, kernel/launcher ABI, grid/dtype;
    // semantic SPM roles (down/zero/skip/gate/output) and byte extents;
    // alignment, extra SPM/DDR bytes, rounds/shards; SCU/SCM locations;
    // separate FP16 multiply/add retirement and the prepared-zero contract.
    std::vector<int64_t> arguments{
        1, pi05_xor3_gated_mlp_opt_in_ ? 1 : 0,
        planned_num_steps_, planned_prefix_len_, 50, 64, planned_prefix_len_ + 50, 1024, 4096, 18, 8, 8, 256, 2048,
        static_cast<int64_t>(KernelId::PI05_XOR3_GATED_RESIDUAL_M50N1024),
        1, 8, 1, 1, 2,
        1, 102400, 2, 102400, 3, 102400, 4, 2048, 5, 102400,
        256, 0, 0, 1, 6400, 63488,
        4096, 4, 8, 10, 4112, 4128, 4144, 1, 1, 1};
    if (pi05_w4_fastpath_opt_in_) {
        // Version 2 is the same FP16 continuation after ordinary W4 GeLU
        // and auto-tile Down; it cannot consume the W8 GeGLU policy.
        arguments[0] = 2;
        const auto w4 = pi05_w4_fastpath_arguments();
        arguments.insert(arguments.end(), w4.begin(), w4.end());
    }
    return arguments;
}

bool Pi05DenoiseStepModel::use_pi05_xor3_gated_mlp(int64_t rows) const {
    if (pi05_w4_fastpath_opt_in_) {
        return pi05_xor3_gated_mlp_opt_in_ && !pi05_geglu_m50_opt_in_ &&
            pi05_w4_fastpath_profile_ &&
            (nvfp4_.enabled() ? pi05_nvfp4_weight_profile_admitted() : pi05_w4_weight_profile_admitted()) &&
            use_pi05_kv1_pair_owner(rows) && use_pi05_xor3_gated_attn(rows);
    }
    if (!pi05_xor3_gated_mlp_opt_in_ || !use_pi05_geglu_m50(rows)) {
        return false;
    }
    if (pi05_is_fp16_action(layer_weights_)) {
        return pi05_fp16_projection_owners_admitted(layer_weights_);
    }
    // GeGLU binds the exact ten-step/M50/H1024/I4096/18-layer/TP8
    // KV1 pair-owner/XOR3-attention stack. Also pin every Down W8A16 owner.
    return std::all_of(
        layer_weights_.begin(), layer_weights_.end(),
        [](const LayerWeights& weights) {
            const auto& weight = weights.down_proj_w;
            const auto& scale = weights.down_ws;
            return weight.defined() && scale.defined() &&
                weight.device().type() == at::kPrivateUse1 &&
                scale.device().type() == at::kPrivateUse1 &&
                weight.scalar_type() == at::kChar &&
                scale.scalar_type() == at::kHalf &&
                weight.is_contiguous() && scale.is_contiguous() &&
                weight.storage_offset() == 0 && scale.storage_offset() == 0 &&
                weight.sizes() == at::IntArrayRef({1024, 4096}) &&
                scale.sizes() == at::IntArrayRef({1024}) &&
                weight.nbytes() == 1024 * 4096 &&
                scale.nbytes() == 1024 * sizeof(c10::Half);
        });
}

void Pi05DenoiseStepModel::validate_pi05_xor3_gated_mlp_manifest(
    const FmbPhysicalExecutionManifest& manifest) const {
    const bool enabled = use_pi05_xor3_gated_mlp(manifest.physical_length);
    if (pi05_xor3_gated_mlp_opt_in_) {
        TORCH_CHECK(
            enabled && manifest.state == FmbPhysicalManifestState::COMPLETE &&
                manifest.logical_length == 50 && manifest.physical_length == 50 &&
                manifest.execution_padding_rows == 0 &&
                manifest.kv_logical_length == planned_prefix_len_ + 50 &&
                manifest.kv_insert_physical_rows == 64 &&
                manifest.graph_lifecycle == FmbGraphLifecycle::COMPOSITE_CHILD &&
                manifest.linear_accumulation == linear_accumulation_policy(),
            "Pi MLP XOR3+gated requires exact COMPLETE W8-GeGLU or W4-fastpath owners");
    }
    size_t policies = 0, reductions = 0;
    for (const auto& route : manifest.routes) {
        TORCH_CHECK(
            route.family != FmbRouteFamily::ALL_REDUCE ||
                route.selector != PI05_DENOISE_XOR3_GATED_MLP_M50N1024 ||
                route.site_id == PI05_DENOISE_XOR3_GATED_MLP_REDUCE_SITE,
            "Pi MLP XOR3+gated selector escaped its continuation site");
        TORCH_CHECK(
            !pi05_xor3_gated_mlp_opt_in_ ||
                route.site_id != FMB_SHARED_MLP_RING_REDUCE_SITE,
            "Pi MLP XOR3+gated cannot consume the original shared MLP ring route");
        if (route.site_id == PI05_DENOISE_XOR3_GATED_MLP_POLICY_SITE) {
            ++policies;
            TORCH_CHECK(
                pi05_xor3_gated_mlp_opt_in_ &&
                    route.family == FmbRouteFamily::GRAPH_SCHEDULE &&
                    route.selector == 1 && route.flags == 0 &&
                    route.invocation == 0 &&
                    route.arguments == pi05_xor3_gated_mlp_arguments(),
                "Pi MLP XOR3+gated cold policy route drifted");
        } else if (route.site_id == PI05_DENOISE_XOR3_GATED_MLP_REDUCE_SITE) {
            ++reductions;
            TORCH_CHECK(
                enabled && route.family == FmbRouteFamily::ALL_REDUCE &&
                    route.selector == PI05_DENOISE_XOR3_GATED_MLP_M50N1024 &&
                    route.flags == 0 && route.invocation == 0 &&
                    route.arguments == pi05_xor3_gated_mlp_arguments(),
                "Pi MLP XOR3+gated continuation drifted or escaped its cold owner");
        }
    }
    TORCH_CHECK(
        policies == (pi05_xor3_gated_mlp_opt_in_ ? 1 : 0) &&
            reductions == (enabled ? 1 : 0),
        "Pi MLP XOR3+gated descriptor is incomplete or duplicated");
}

void Pi05DenoiseStepModel::validate_pi05_xor3_gated_mlp_policy(
    const FmbPrefillStageCandidate* candidate) const {
    if (candidate == nullptr) {
        TORCH_CHECK(!pi05_xor3_gated_mlp_opt_in_,
                    "Pi MLP XOR3+gated requires a complete signed loop descriptor");
        return;
    }
    validate_pi05_xor3_gated_mlp_manifest(candidate->physical_manifest);
    if (pi05_xor3_gated_mlp_opt_in_) {
        TORCH_CHECK(
            pi05_graph_dma_shape_admitted(
                candidate->stage_plan,
                candidate->physical_manifest.physical_length,
                candidate->physical_manifest.logical_length,
                planned_prefix_len_, 1, true, false,
                AttentionExecutionPolicy::DDR_KV),
            "Pi MLP XOR3+gated descriptor chunk schedule drifted");
        pi05_require_stable_graph_dma_capture();
    }
}

void Pi05DenoiseStepModel::validate_pi05_kv1_direct_cache_manifest(
    const FmbPhysicalExecutionManifest& manifest) const {
    const int64_t logical_position =
        rope_position_ >= 0 ? rope_position_ : planned_prefix_len_;
    const bool enabled = use_pi05_kv1_direct_cache(manifest.physical_length);
    const bool pair_owner =
        use_pi05_kv1_pair_owner(manifest.physical_length);
    if (pi05_kv1_direct_cache_opt_in_) {
        TORCH_CHECK(
            enabled && manifest.state == FmbPhysicalManifestState::COMPLETE &&
                manifest.logical_length == 50 && manifest.physical_length == 50 &&
                manifest.execution_padding_rows == 0 &&
                manifest.kv_logical_length == planned_prefix_len_ + 50 &&
                manifest.kv_insert_physical_rows == 64 &&
                manifest.graph_lifecycle == FmbGraphLifecycle::COMPOSITE_CHILD &&
                manifest.linear_accumulation == linear_accumulation_policy(),
            "Pi KV1 direct-cache requires exact W8A16 ten-step "
            "P800/M50/PAD64/capacity2048 owners");
    }

    size_t policies = 0;
    size_t pair_owners = 0;
    size_t pair_k = 0;
    size_t pair_v = 0;
    size_t direct_sources = 0;
    size_t forbidden_gathers = 0;
    for (const auto& route : manifest.routes) {
        TORCH_CHECK(
            route.family != FmbRouteFamily::LINEAR ||
                (route.selector != PI05_DENOISE_KV1_PAIR_OWNER_M50N32X2K1024 &&
                 route.selector != PI05_DENOISE_KV1_PAIR_OWNER_FP16_M50N32X2K1024) ||
                route.site_id == PI05_DENOISE_KV1_PAIR_OWNER_SITE,
            "Pi KV1 pair-owner selector escaped its single producer site");
        TORCH_CHECK(
            !(route.flags & PI05_DENOISE_KV1_DIRECT_CACHE_FLAG) ||
                route.site_id == PI05_DENOISE_K_ROPE_SITE ||
                route.site_id == PI05_DENOISE_KV_INSERT_SITE,
            "Pi KV1 direct-cache flag escaped its paired RoPE/KV sites");
        if (route.site_id == PI05_DENOISE_KV1_DIRECT_POLICY_SITE) {
            ++policies;
            const auto expected_policy = pi05_kv1_pair_owner_opt_in_
                ? pi05_kv1_pair_owner_policy_arguments(
                      pi05_kv1_pair_owner_opt_in_, logical_position,
                      planned_prefix_len_, pi05_is_fp16_action(layer_weights_), planned_num_steps_)
                : pi05_kv1_direct_policy_arguments(
                      pi05_kv1_direct_cache_opt_in_, logical_position, planned_prefix_len_);
            TORCH_CHECK(
                route.family == FmbRouteFamily::GRAPH_SCHEDULE &&
                    route.selector == 1 && route.flags == 0 &&
                    route.invocation == 0 &&
                    route.arguments == expected_policy,
                "Pi KV1 direct-cache cold policy route drifted");
        } else if (route.site_id == PI05_DENOISE_KV1_PAIR_OWNER_SITE) {
            ++pair_owners;
            TORCH_CHECK(
                pair_owner && route.family == FmbRouteFamily::LINEAR &&
                    route.selector ==
                        pi05_kv1_pair_owner_selector(pi05_is_fp16_action(layer_weights_)) &&
                    route.flags == 0 && route.invocation == 0 &&
                    route.arguments ==
                        pi05_kv1_pair_owner_linear_arguments(planned_prefix_len_, pi05_is_fp16_action(layer_weights_), planned_num_steps_),
                "Pi KV1 pair-owner producer route drifted or escaped its "
                "cold owner");
        } else if (route.site_id == PI05_DENOISE_K_LINEAR_SITE &&
                   route.selector == PI05_DENOISE_KV1_DIRECT_M592N32) {
            ++pair_k;
            TORCH_CHECK(
                enabled && !pair_owner &&
                    route.family == FmbRouteFamily::LINEAR &&
                    route.flags == 0 && route.invocation == 0 &&
                    route.arguments == pi05_kv1_direct_linear_arguments(planned_prefix_len_),
                "Pi KV1 direct-cache pair K route drifted");
        } else if (route.site_id == PI05_DENOISE_V_LINEAR_SITE &&
                   route.selector == PI05_DENOISE_KV1_DIRECT_M592N32) {
            ++pair_v;
            TORCH_CHECK(
                enabled && !pair_owner &&
                    route.family == FmbRouteFamily::LINEAR &&
                    route.flags == 0 && route.invocation == 0 &&
                    route.arguments == pi05_kv1_direct_linear_arguments(planned_prefix_len_),
                "Pi KV1 direct-cache pair V route drifted");
        } else if (route.site_id == PI05_DENOISE_KV1_K_GATHER_SITE ||
                   route.site_id == PI05_DENOISE_KV1_V_GATHER_SITE) {
            if (pi05_kv1_direct_cache_opt_in_) ++forbidden_gathers;
        } else if (route.site_id == PI05_DENOISE_K_ROPE_SITE ||
                   route.site_id == PI05_DENOISE_KV_INSERT_SITE) {
            if (route.flags & PI05_DENOISE_KV1_DIRECT_CACHE_FLAG) {
                ++direct_sources;
                TORCH_CHECK(
                    enabled && route.flags == PI05_DENOISE_KV1_DIRECT_CACHE_FLAG,
                    "Pi KV1 direct-cache route flags are not exclusive");
                if (route.site_id == PI05_DENOISE_K_ROPE_SITE) {
                    TORCH_CHECK(
                        route.family == FmbRouteFamily::ROPE &&
                            route.selector == static_cast<int64_t>(
                                Pi05DenoiseRopeRoute::FULL_ROPE_TILED) &&
                            route.invocation == 0 &&
                            route.arguments ==
                                std::vector<int64_t>{logical_position},
                        "Pi KV1 direct-cache K-RoPE semantic route drifted");
                } else {
                    const auto frozen = restore_kvinsert_plan_from_manifest(
                        manifest, PI05_DENOISE_KV_INSERT_SITE,
                        route.invocation,
                        /*tp=*/8, /*nkv=*/8, /*hd=*/256);
                    const auto canonical = rpu_kvinsert_route_arguments(
                        frozen, /*tp=*/8, /*nkv=*/8, /*hd=*/256);
                    TORCH_CHECK(
                        route.family == FmbRouteFamily::KV_INSERT &&
                            route.selector == static_cast<int64_t>(
                                KvInsertRoute::PAD16_V16) &&
                            route.invocation == 0 &&
                            frozen.route() == KvInsertRoute::PAD16_V16 &&
                            frozen.logical_rows() == 50 &&
                            frozen.physical_rows() == 64 &&
                            frozen.segment_count() == 1 &&
                            frozen.segment(0).kernel == KvInsertKernel::V16 &&
                            frozen.segment(0).position == planned_prefix_len_ &&
                            frozen.segment(0).token_offset == 0 &&
                            frozen.segment(0).rows == 64 &&
                            route.arguments == std::vector<int64_t>(
                                canonical.begin(), canonical.end()),
                        "Pi KV1 direct-cache KV semantic route drifted");
                }
            }
        }
    }
    TORCH_CHECK(
        policies == (pi05_kv1_direct_cache_opt_in_ ? 1 : 0) &&
            pair_owners == (pair_owner ? 1 : 0) &&
            pair_k == (enabled && !pair_owner ? 1 : 0) &&
            pair_v == (enabled && !pair_owner ? 1 : 0) &&
            direct_sources == (enabled ? 2 : 0) && forbidden_gathers == 0,
        "Pi KV1 direct-cache descriptor is incomplete, duplicated, or "
        "contains a forbidden gather");
}

void Pi05DenoiseStepModel::validate_pi05_kv1_direct_cache_policy(
    const FmbPrefillStageCandidate* candidate) const {
    if (candidate == nullptr) {
        TORCH_CHECK(
            !pi05_kv1_direct_cache_opt_in_,
            "Pi KV1 direct-cache requires a complete signed loop descriptor");
        return;
    }
    validate_pi05_kv1_direct_cache_manifest(candidate->physical_manifest);
    if (pi05_kv1_direct_cache_opt_in_) {
        TORCH_CHECK(
            pi05_graph_dma_shape_admitted(
                candidate->stage_plan,
                candidate->physical_manifest.physical_length,
                candidate->physical_manifest.logical_length,
                planned_prefix_len_, 1, true, false,
                AttentionExecutionPolicy::DDR_KV),
            "Pi KV1 direct-cache descriptor chunk schedule drifted");
        pi05_require_stable_graph_dma_capture();
    }
}

void Pi05DenoiseStepModel::emit_pi05_q_rope_epilogue(
    const LayerWeights& weights,const ChunkInfo& chunk) {
    const int64_t logical_position=(rope_position_>=0 ? rope_position_ : ctx().position)+chunk.offset;
    TORCH_CHECK(pi05_q_rope_epilogue_opt_in_ && loop_mode_ && num_steps_==10 &&
        ctx().has_complete_physical_manifest() && chunk.idx==0 && chunk.offset==0 &&
        use_pi05_q_rope_epilogue(chunk.len,logical_position) && ctx().position == planned_prefix_len_ &&
        ctx().batch_size==1 && !ctx().is_causal &&
        ctx().attention_policy==AttentionExecutionPolicy::DDR_KV,
        "Pi Q-RoPE runtime owner/profile drift");
    pi05_require_stable_graph_dma_capture();
    ctx().consume_physical_route(FmbRouteFamily::LINEAR,PI05_DENOISE_Q_EPILOGUE_LINEAR_SITE,
        PI05_DENOISE_Q_PAIR16_ROPE,0,pi05_q_rope_arguments(logical_position),chunk.idx);
    rpu_launch_pi05_q_rope_spm_kernel(addr(0,"input_norm"),weights.q_w,addr(0,"q"),
        weights.q_ws,cos_,sin_,logical_position);
}

void Pi05DenoiseStepModel::set_rope_position(int64_t position) {
    rope_position_ = position;
}

void Pi05DenoiseStepModel::set_adarms_table(const at::Tensor& table) {
    TORCH_CHECK(!RpuKernelGraph::has_active() && num_layers() > 0
                    && get_last_resolved_chunk_size() == 0,
                "Pi0.5 AdaRMS table must be set after weights and before forward");
    TORCH_CHECK(table.defined() && table.dim() == 3 && table.size(0) > 0
                    && table.size(0) * 1100 < 32000
                    && table.size(1) == 2 * num_layers() + 1
                    && table.size(2) == 3 * hidden_size()
                    && table.scalar_type() == at::kHalf
                    && table.device().type() == at::kPrivateUse1
                    && table.is_contiguous(),
                "Pi0.5 AdaRMS table must be contiguous RPU fp16 [steps,2L+1,3H]");
    adarms_table_ = table.clone();
    rpu_ddr_flush_force(adarms_table_.data_ptr<c10::Half>());
    adarms_table_src_base_ = ::rhino_lkn::RpuGetDevAddr(adarms_table_.data_ptr());
    invalidate_model_state();
}

void Pi05DenoiseStepModel::set_configured_chunk_size(int64_t chunk_size) {
    TORCH_CHECK(chunk_size == 0
                    || (chunk_size >= 16 && chunk_size % 16 == 0),
                "Pi0.5 action chunk size must be 0 (auto) or a positive "
                "multiple of 16, got ", chunk_size);
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Pi0.5 action chunk size must be set before the first forward");
    const int64_t single_chunk_min = ((chunk_size_ + 15) / 16) * 16;
    TORCH_CHECK(chunk_size == 0 || chunk_size >= single_chunk_min,
                "Pi0.5 action expert requires one chunk for the full action "
                "horizon (", chunk_size_, " rows); configured chunk size must "
                "be 0 (auto) or at least ", single_chunk_min, ", got ",
                chunk_size);
    configured_chunk_size_ = chunk_size;
    invalidate_model_state();
}

std::vector<int64_t> Pi05DenoiseStepModel::resolve_action_stage_domain(
    int64_t execution_len, int64_t logical_len, int64_t position,
    int64_t kv_len, int64_t cache_capacity,
    int64_t requested_chunk_size, bool prefer_pad16,
    bool loop_mode, int64_t num_steps) {
    detail::validate_fmb_planning_shape(
        execution_len, position, "Pi0.5 denoise action planner");
    TORCH_CHECK(
        logical_len > 0 && logical_len == execution_len,
        "RPU_PLANNER_REJECT:CAPABILITY: Pi0.5 denoise action rows cannot "
        "use execution padding");
    TORCH_CHECK(
        kv_len == position + execution_len,
        "RPU_PLANNER_REJECT:CAPABILITY: Pi0.5 denoise mask width must equal "
        "prefix plus action rows");
    TORCH_CHECK(
        cache_capacity >= kv_len,
        "RPU_PLANNER_REJECT:CAPABILITY: Pi0.5 denoise cache capacity does "
        "not cover the logical action rows");
    TORCH_CHECK(
        num_steps > 0 && (!loop_mode || num_steps * 1100 < 32000),
        "RPU_PLANNER_REJECT:CAPABILITY: invalid Pi0.5 denoise body count ",
        num_steps);
    TORCH_CHECK(
        loop_mode || num_steps == 1,
        "RPU_PLANNER_REJECT:CAPABILITY: stepwise Pi0.5 denoise descriptor "
        "must describe one body");
    TORCH_CHECK(!loop_mode || !adarms_table_.defined()
                    || adarms_table_.size(0) == num_steps,
                "Pi0.5 AdaRMS table does not match requested denoise steps");
    const int64_t required_chunk = ((execution_len + 15) / 16) * 16;
    TORCH_CHECK(
        requested_chunk_size == 0 || requested_chunk_size == required_chunk,
        "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 denoise exact chunk must "
        "equal ceil16(action rows): requested=", requested_chunk_size,
        " required=", required_chunk);

    // Oracle requests are temporary. Each cached caller restores its own
    // COMPLETE descriptor below; neither a later query nor a failed query may
    // change the layout used by a previously prepared execution.
    std::vector<int64_t> result;
    auto query_scope = capture_kvinsert_cost_layout_scope();
    query_scope([&] {
        planned_route_profile_valid_ = true;
        planned_loop_mode_ = loop_mode;
        planned_num_steps_ = num_steps;
        planned_prefix_len_ = position;
        planned_k_rope_cache_capacity_ = cache_capacity;
        const int64_t padded_rows = ((execution_len + 15) / 16) * 16;
        planned_kv_physical_rows_ =
            prefer_pad16 && position % 16 == 0 &&
                position + padded_rows <= cache_capacity
            ? padded_rows
            : execution_len;
    TORCH_CHECK(!pi05_w4_fastpath_opt_in_ ||
                    (pi05_w4_fastpath_profile_ && use_pi05_graph_dma_profile(execution_len) &&
                     use_pi05_kv1_pair_owner(execution_len)),
                "Pi W4 fastpath requires its exact prepared M50 COMPLETE profile");
    TORCH_CHECK(!pi05_adarms_body_stream_opt_in_ || use_pi05_adarms_body_stream(),
                "Pi0.5 body-table route requires the original ten-step P800/M50 table profile");
    TORCH_CHECK(!pi05_gateup_resident_layers_ || use_pi05_gateup_bundle(execution_len),
                "Pi0.5 Gate/Up bundle requires installed exact W8A16 TP8 M50 weights");
    TORCH_CHECK(!pi05_q_rope_epilogue_opt_in_ ||
                    use_pi05_q_rope_epilogue(execution_len,rope_position_>=0?rope_position_:position),
                "Pi Q-RoPE requires exact owned W8 ten-step P800/M50 profile");
    TORCH_CHECK(!pi05_xor3_gated_attn_opt_in_ ||
                    use_pi05_xor3_gated_attn(execution_len),
                "Pi attention XOR3+gated continuation requires the exact "
                "ten-step W8A16 P800/M50 owner profile");
    TORCH_CHECK(!pi05_kv1_stripe_gather_opt_in_ ||
                    use_pi05_kv1_stripe_gather(execution_len),
                "Pi KV1 stripe-gather requires all compact W8A16 owners and "
                "the exact ten-step P800/M50 profile");
    TORCH_CHECK(!pi05_kv1_direct_cache_opt_in_ ||
                    use_pi05_kv1_direct_cache(execution_len),
                "Pi KV1 direct-cache requires pair-mapped W8A16 owners, "
                "retained admitted RoPE table owners and the exact ten-step "
                "P800/M50 profile");
    TORCH_CHECK(!pi05_nvfp4_geglu_m50_opt_in_ || use_pi05_nvfp4_geglu_m50(execution_len),
                "Pi NVFP4 GeGLU requires its exact prepared striped-v2 stack");
    TORCH_CHECK(!pi05_w4_geglu_m50_opt_in_ ||
                    use_pi05_w4_geglu_m50(execution_len),
                "Pi W4 GeGLU requires its complete ten-step P800/M50 pgrp stack");
    TORCH_CHECK(!pi05_geglu_m50_opt_in_ ||
                    use_pi05_geglu_m50(execution_len),
                "Pi M50 GeGLU requires the accepted KV1 pair-owner/"
                "XOR3 W8A16 stack and complete Gate/Up owners");
    TORCH_CHECK(!pi05_xor3_gated_mlp_opt_in_ ||
                    use_pi05_xor3_gated_mlp(execution_len),
                "Pi MLP XOR3+gated requires the exact GeGLU stack and Down owners");
        auto mask_shape = at::empty(
            {1, 1, execution_len, kv_len},
            at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
        result = encode_fmb_prefill_stage_domain(
            resolve_prefill_stage_domain_for_shape(
                execution_len, position,
                std::optional<at::Tensor>(mask_shape),
                /*is_causal=*/false, requested_chunk_size,
                /*logical_len=*/logical_len));
    });
    return result;
}

void Pi05DenoiseStepModel::restore_planned_route_profile(
    at::IntArrayRef descriptor, bool loop_mode, int64_t num_steps,
    int64_t prefix_len, int64_t cache_capacity) {
    TORCH_CHECK(
        num_steps > 0 && (loop_mode ? num_steps <= (32000 - 1) / 1100 : num_steps == 1),
        "RPU_PLANNER_REJECT:EXACT_MISMATCH: invalid Pi0.5 denoise body count");
    TORCH_CHECK(
        prefix_len >= 0 && cache_capacity >= prefix_len + chunk_size_,
        "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 denoise cache capacity "
        "does not cover the actual action rows");
    int64_t kv_physical_rows = chunk_size_;
    if (descriptor.empty()) {
        // Preserve the original immediate diagnostic path without borrowing
        // the most recent oracle's padding or branch. It carries no Graph proof.
        TORCH_CHECK(!RpuKernelGraph::has_active(),
                    "Pi0.5 denoise Graph requires a COMPLETE descriptor");
    } else {
        const auto prepared_planned = prepare_stage_candidate(descriptor);
        const auto& planned = prepared_planned->candidate();
        const auto& manifest = planned.physical_manifest;
        TORCH_CHECK(
            manifest.state == FmbPhysicalManifestState::COMPLETE &&
                manifest.graph_lifecycle == FmbGraphLifecycle::COMPOSITE_CHILD &&
                manifest.linear_accumulation == linear_accumulation_policy() &&
                manifest.logical_length == chunk_size_ &&
                manifest.physical_length == chunk_size_ &&
                manifest.execution_padding_rows == 0 &&
                manifest.kv_logical_length == prefix_len + chunk_size_,
            "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 denoise descriptor geometry drift");
        kv_physical_rows = manifest.kv_insert_physical_rows;
        const int64_t padded_rows = ((chunk_size_ + 15) / 16) * 16;
        TORCH_CHECK(
            (kv_physical_rows == chunk_size_ ||
             (kv_physical_rows == padded_rows && prefix_len % 16 == 0)) &&
                prefix_len + kv_physical_rows <= cache_capacity,
            "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 denoise descriptor "
            "has illegal pad16/cache capacity");
        bool found_body = false;
        bool found_output = false;
        bool found_mask_schedule = false;
        const int64_t output_site = loop_mode ? PI05_DENOISE_POST_LOOP_DMA_SITE
                                              : PI05_DENOISE_POST_STEP_DMA_SITE;
        for (const auto& route : manifest.routes) {
            if (route.family == FmbRouteFamily::GRAPH_SCHEDULE &&
                route.site_id == PI05_DENOISE_MASK_SCHEDULE_SITE) {
                TORCH_CHECK(
                    !found_mask_schedule && route.invocation == 0 &&
                        route.flags == 0 &&
                        route.selector == static_cast<int64_t>(
                            Pi05DenoiseMaskScheduleRoute::FIRST_BODY_FIRST_LAYER) &&
                        route.arguments == std::vector<int64_t>({
                            chunk_size_, prefix_len + chunk_size_, attn_tp(),
                            num_cores(), loop_mode ? 1 : 0, num_steps, num_layers()}),
                    "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 denoise "
                    "mask lifetime/schedule route drift");
                found_mask_schedule = true;
            }
            if (route.family == FmbRouteFamily::LINEAR &&
                route.site_id == PI05_DENOISE_PRE_LINEAR_SITE) {
                TORCH_CHECK(
                    !found_body && route.invocation == 0 && route.flags == 0 &&
                        route.selector == static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE) &&
                        route.arguments == std::vector<int64_t>({loop_mode ? 1 : 0, num_steps}),
                    "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 denoise "
                    "descriptor mode/body count differs from this forward");
                found_body = true;
            }
            if (route.family == FmbRouteFamily::MUTABLE_DMA &&
                route.site_id == output_site) {
                TORCH_CHECK(
                    !found_output && route.invocation == 0 && route.flags == 0 &&
                        route.selector == static_cast<int64_t>(Pi05DenoiseMutableDmaRoute::SPM_COPY_TO_DDR) &&
                        route.arguments.empty(),
                    "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 denoise output DMA route drift");
                found_output = true;
            }
        }
        TORCH_CHECK(found_body && found_output,
                    "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 denoise "
                    "descriptor omits the actual body/output route");
        TORCH_CHECK(found_mask_schedule,
                    "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 denoise "
                    "descriptor omits the mask lifetime/schedule route");
    }
    // Policies depend on the actual forward's restored profile. Oracle queries
    // are scoped and deliberately leave no profile behind. Roll back on any
    // policy failure so a rejected descriptor cannot poison a cached request.
    auto fields = std::tie(planned_route_profile_valid_, planned_loop_mode_,
        planned_num_steps_, planned_kv_physical_rows_, planned_prefix_len_,
        planned_k_rope_cache_capacity_);
    const auto previous = std::make_tuple(planned_route_profile_valid_, planned_loop_mode_,
        planned_num_steps_, planned_kv_physical_rows_, planned_prefix_len_,
        planned_k_rope_cache_capacity_);
    bool accepted = false;
    auto rollback = c10::make_scope_exit([&] { if (!accepted) fields = previous; });
    planned_route_profile_valid_ = true;
    planned_loop_mode_ = loop_mode;
    planned_num_steps_ = num_steps;
    planned_kv_physical_rows_ = kv_physical_rows;
    planned_prefix_len_ = prefix_len;
    planned_k_rope_cache_capacity_ = cache_capacity;
    validate_pi05_runtime_policies(descriptor);
    accepted = true;
}

void Pi05DenoiseStepModel::validate_planned_route_profile(
    bool loop_mode, int64_t num_steps) const {
    TORCH_CHECK(
        planned_route_profile_valid_ &&
            planned_loop_mode_ == loop_mode &&
            planned_num_steps_ == num_steps,
        "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 denoise forward branch "
        "does not match the native descriptor request");
    TORCH_CHECK(!loop_mode || !adarms_table_.defined()
                    || adarms_table_.size(0) == num_steps,
                "Pi0.5 AdaRMS table does not match forward denoise steps");
}

void Pi05DenoiseStepModel::set_weights(
    at::TensorList q_w, at::TensorList k_w,
    at::TensorList v_w, at::TensorList o_w,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    at::TensorList attn_dense_w_list, at::TensorList attn_dense_b_list,
    at::TensorList mlp_dense_w_list,  at::TensorList mlp_dense_b_list,
    const at::Tensor& final_norm_dense_w, const at::Tensor& final_norm_dense_b,
    const at::Tensor& action_in_proj_w,   const at::Tensor& action_in_proj_b,
    const at::Tensor& action_out_proj_w,  const at::Tensor& action_out_proj_b,
    const at::Tensor& cos, const at::Tensor& sin,
    int64_t hidden_size, int64_t max_action_dim, int64_t chunk_size,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t num_layers, double eps,
    at::TensorList q_w_scale, at::TensorList k_w_scale,
    at::TensorList v_w_scale, at::TensorList o_w_scale,
    at::TensorList gate_scale, at::TensorList up_scale,
    at::TensorList down_scale,
    at::TensorList attn_dense_w_scale, at::TensorList mlp_dense_w_scale,
    const at::Tensor& final_norm_dense_w_scale, at::TensorList nvfp4_tensor_scales)
{
    TORCH_CHECK(num_layers > 0 && hidden_size > 0 && head_dim > 0
                && num_q_heads > 0 && num_kv_heads > 0
                && max_action_dim > 0 && chunk_size > 0,
                "Pi05DenoiseStepModel::set_weights: dim params must be positive");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0,
                "Pi05DenoiseStepModel::set_weights: num_q_heads must be divisible by num_kv_heads");
    auto check_list = [&](const at::TensorList& list, const char* name) {
        TORCH_CHECK(static_cast<int64_t>(list.size()) == num_layers,
                    "Pi05DenoiseStepModel::set_weights: ", name,
                    ".size()=", list.size(), " != num_layers=", num_layers);
    };
    check_list(q_w, "q_w");
    check_list(k_w, "k_w");
    check_list(v_w, "v_w");
    check_list(o_w, "o_w");
    check_list(gate_w, "gate_w");
    check_list(up_w, "up_w");
    check_list(down_w, "down_w");
    check_list(attn_dense_w_list, "attn_dense_w_list");
    check_list(attn_dense_b_list, "attn_dense_b_list");
    check_list(mlp_dense_w_list, "mlp_dense_w_list");
    check_list(mlp_dense_b_list, "mlp_dense_b_list");
    TORCH_CHECK(!pi05_nvfp4_geglu_m50_opt_in_ || !nvfp4_tensor_scales.empty(),
                "Pi ACC16 NVFP4 GeGLU requires NVFP4 owners; legacy INT4 is unchanged");
    check_pi05_scale_lists(
        "Pi05DenoiseStepModel::set_weights", num_layers,
        q_w, k_w, v_w, o_w, gate_w, up_w, down_w,
        q_w_scale, k_w_scale, v_w_scale, o_w_scale,
        gate_scale, up_scale, down_scale, !nvfp4_tensor_scales.empty());
    Pi05Nvfp4Tables nvfp4_tables;
    nvfp4_tables.install(nvfp4_tensor_scales, num_layers, num_cores());
    const bool has_scale = !q_w_scale.empty();
    // AdaRMS dense W8A16 is independent of the decoder-projection W8A16 above
    // (the dense can be quantized while projections are fp16, or vice versa).
    const bool has_attn_dense_scale = !attn_dense_w_scale.empty();
    const bool has_mlp_dense_scale = !mlp_dense_w_scale.empty();
    const bool has_final_dense_scale =
        final_norm_dense_w_scale.defined()
        && final_norm_dense_w_scale.numel() != 0;
    TORCH_CHECK(has_attn_dense_scale == has_mlp_dense_scale
                && has_attn_dense_scale == has_final_dense_scale,
                "Pi05DenoiseStepModel::set_weights: dense scales must be all "
                "provided or all empty");
    const bool has_dense_scale = has_attn_dense_scale;
    if (has_dense_scale) {
        TORCH_CHECK(static_cast<int64_t>(attn_dense_w_scale.size()) == num_layers
                    && static_cast<int64_t>(mlp_dense_w_scale.size()) == num_layers,
                    "Pi05DenoiseStepModel::set_weights: dense scale lists must "
                    "have num_layers entries when provided");
    }

    auto check_dense = [&](const at::Tensor& weight,
                           const at::Tensor& bias,
                           const at::Tensor& scale,
                           const char* name) {
        TORCH_CHECK(weight.defined() && weight.dim() == 2
                    && weight.size(0) == 3 * hidden_size
                    && weight.size(1) == hidden_size
                    && weight.device().type() == at::kPrivateUse1
                    && weight.is_contiguous(),
                    "Pi05DenoiseStepModel::set_weights: ", name,
                    " weight must be contiguous RPU [3 * hidden_size, "
                    "hidden_size]");
        const auto expected_weight_dtype =
            has_dense_scale ? at::kChar : at::kHalf;
        TORCH_CHECK(weight.scalar_type() == expected_weight_dtype,
                    "Pi05DenoiseStepModel::set_weights: ", name,
                    " weight dtype must be ", expected_weight_dtype,
                    ", got ", weight.scalar_type());
        TORCH_CHECK(bias.defined() && bias.dim() == 1
                    && bias.numel() == 3 * hidden_size
                    && bias.scalar_type() == at::kHalf
                    && bias.device().type() == at::kPrivateUse1
                    && bias.is_contiguous(),
                    "Pi05DenoiseStepModel::set_weights: ", name,
                    " bias must be contiguous fp16 RPU [3 * hidden_size]");
        if (has_dense_scale) {
            TORCH_CHECK(scale.defined() && scale.dim() == 1
                        && scale.numel() == 3 * hidden_size
                        && scale.scalar_type() == at::kHalf
                        && scale.device().type() == at::kPrivateUse1
                        && scale.is_contiguous(),
                        "Pi05DenoiseStepModel::set_weights: ", name,
                        " scale must be contiguous fp16 RPU [3 * hidden_size]");
        }
    };
    for (int64_t i = 0; i < num_layers; ++i) {
        check_dense(
            attn_dense_w_list[i], attn_dense_b_list[i],
            has_dense_scale ? attn_dense_w_scale[i] : at::Tensor(),
            "attn_dense");
        check_dense(
            mlp_dense_w_list[i], mlp_dense_b_list[i],
            has_dense_scale ? mlp_dense_w_scale[i] : at::Tensor(),
            "mlp_dense");
    }
    check_dense(
        final_norm_dense_w, final_norm_dense_b,
        has_dense_scale ? final_norm_dense_w_scale : at::Tensor(),
        "final_norm_dense");

    // intermediate_size from down_w[0].size(1) (col after row-partition swizzle
    // gives intermediate as the second dim of [hidden, intermediate]).
    // packed-INT4 (uint8) down weight is [hidden, intermediate/2] (nibble-packed
    // K), so size(1) is half the logical intermediate — double it back.
    const int64_t intermediate_size =
        (down_w[0].scalar_type() == at::kByte) ? down_w[0].size(1) * 2
                                               : down_w[0].size(1);
    if (num_cores() != 8) {
        const auto dtype = q_w[0].scalar_type();
        TORCH_CHECK(dtype == at::kHalf || dtype == at::kChar,
                    "reduced Pi0.5 supports FP16 or W8A16 projections");
        TORCH_CHECK(max_action_dim == 32 && chunk_size == 50,
                    "reduced Pi0.5 requires action horizon 50 and width 32");
        for (const auto* tensor : {&action_in_proj_w, &action_in_proj_b,
                                   &action_out_proj_w, &action_out_proj_b})
            TORCH_CHECK(tensor->defined() && tensor->scalar_type() == at::kHalf &&
                            tensor->device().type() == at::kPrivateUse1 &&
                            tensor->is_contiguous(),
                        "Pi0.5 action projections must remain contiguous FP16 RPU");
        TORCH_CHECK(action_in_proj_w.sizes() == at::IntArrayRef({1024, 32}) &&
                        action_out_proj_w.sizes() == at::IntArrayRef({32, 1024}) &&
                        action_in_proj_b.sizes() == at::IntArrayRef({1024}) &&
                        action_out_proj_b.sizes() == at::IntArrayRef({32}),
                    "Pi0.5 reduced single-core action projection shape mismatch");
        const int64_t expected_physical = validate_pi05_reduced_geometry(
            num_cores(), num_q_heads, num_kv_heads, head_dim,
            hidden_size, 4096, num_layers, true, dtype == at::kChar);
        TORCH_CHECK(intermediate_size == expected_physical,
                    "Pi0.5 denoise physical MLP width/precision mismatch");
        for (int64_t i = 0; i < num_layers; ++i) {
            std::array<std::array<int64_t, 2>, 7> shapes;
            size_t j = 0;
            for (const auto* tensor : {&q_w[i], &k_w[i], &v_w[i],
                    &o_w[i], &gate_w[i], &up_w[i], &down_w[i]}) {
                TORCH_CHECK(tensor->scalar_type() == dtype && tensor->dim() == 2,
                            "Pi0.5 reduced projection precision/rank mismatch");
                shapes[j++] = {tensor->size(0), tensor->size(1)};
            }
            validate_pi05_reduced_projection_shapes(
                shapes, hidden_size, expected_physical);
        }
    }
    logical_intermediate_size_ = num_cores() == 8 ? intermediate_size : 4096;
    reduced_w8a16_ = q_w[0].scalar_type() == at::kChar;
    set_model_params(num_q_heads, num_kv_heads, head_dim,
                     hidden_size, intermediate_size);
    set_num_layers(num_layers);
    eps_              = eps;
    max_action_dim_   = max_action_dim;
    chunk_size_       = chunk_size;
    local_q_heads_    = num_q_heads / attn_tp();
    local_kv_dim_     = num_kv_heads * head_dim / attn_tp();

    nvfp4_ = std::move(nvfp4_tables);
    layer_weights_.clear();
    layer_weights_.reserve(num_layers);
    pi05_k_rope_insert_w8a16_ = has_scale;
    for (int64_t i = 0; i < num_layers; ++i) {
        layer_weights_.push_back({
            q_w[i], k_w[i], v_w[i], o_w[i],
            gate_w[i], up_w[i], down_w[i],
            has_scale ? rpu_retain_linear_quant_scale(q_w[i], q_w_scale[i]) : at::Tensor(),
            has_scale ? rpu_retain_linear_quant_scale(k_w[i], k_w_scale[i]) : at::Tensor(),
            has_scale ? rpu_retain_linear_quant_scale(v_w[i], v_w_scale[i]) : at::Tensor(),
            has_scale ? rpu_retain_linear_quant_scale(o_w[i], o_w_scale[i]) : at::Tensor(),
            has_scale ? rpu_retain_linear_quant_scale(gate_w[i], gate_scale[i]) : at::Tensor(),
            has_scale ? rpu_retain_linear_quant_scale(up_w[i], up_scale[i]) : at::Tensor(),
            has_scale ? rpu_retain_linear_quant_scale(down_w[i], down_scale[i]) : at::Tensor(),
            attn_dense_w_list[i], attn_dense_b_list[i],
            mlp_dense_w_list[i],  mlp_dense_b_list[i],
            has_dense_scale ? attn_dense_w_scale[i] : at::Tensor(),
            has_dense_scale ? mlp_dense_w_scale[i]  : at::Tensor(),
        });
    }
    // Resolve all decoder projection precision once, including mixed W4/W8.
    for (const auto& lw : layer_weights_) {
        for (const auto* weight : {&lw.q_w, &lw.k_w, &lw.v_w, &lw.o_w,
                                  &lw.gate_proj_w, &lw.up_proj_w,
                                  &lw.down_proj_w}) {
            pi05_k_rope_insert_w8a16_ &= weight->scalar_type() == at::kChar;
        }
    }
    pi05_w4_fastpath_profile_ = pi05_w4_fastpath_opt_in_ &&
        (nvfp4_.enabled() ? pi05_nvfp4_weight_profile_admitted() : pi05_w4_weight_profile_admitted());
    TORCH_CHECK(!pi05_w4_fastpath_opt_in_ || pi05_w4_fastpath_profile_,
                "Pi W4 fastpath requires exact Q/O/Gate/Up/Down packed W4 and K/V W8 owners");
    pi05_kv1_direct_pair_owners_ = false;
    pi05_kv1_pair_owner_full_owners_ = false;
    if (pi05_kv1_stripe_gather_opt_in_ || pi05_kv1_direct_cache_opt_in_) {
        TORCH_CHECK(
            (!pi05_kv1_stripe_gather_opt_in_ || pi05_k_rope_insert_opt_in_) &&
                pi05_projection_fastpath_admitted() &&
                hidden_size == 1024 && intermediate_size == 4096 &&
                num_q_heads == 8 && num_kv_heads == 8 && head_dim == 256 &&
                num_layers == 18 && chunk_size == 50 &&
                (!pi05_kv1_direct_cache_opt_in_ || max_action_dim == 32) &&
                attn_tp() == 8,
            "Pi KV1 optimized routes require exact installed Action owners");
        if (pi05_kv1_pair_owner_opt_in_) {
            for (const auto& lw : layer_weights_) {
                rpu_validate_pi05_kv1_pair_owner_w8a16(
                    lw.k_w, lw.k_ws, lw.v_w, lw.v_ws);
            }
            size_t pair_companion_bytes = 0;
            for (const auto& lw : layer_weights_) {
                for (const auto* companion : {
                        &lw.k_kv1_w, &lw.v_kv1_w,
                        &lw.k_kv1_ws, &lw.v_kv1_ws}) {
                    TORCH_CHECK(
                        !companion->defined(),
                        "Pi KV1 pair-owner must not install compact companions");
                    pair_companion_bytes +=
                        companion->defined() ? companion->nbytes() : 0;
                }
            }
            TORCH_CHECK(
                pair_companion_bytes == 0,
                "Pi KV1 pair-owner compact companion footprint must be 0B");
            pi05_kv1_pair_owner_full_owners_ = true;
        } else {
            for (auto& lw : layer_weights_) {
                auto compact_k = pi05_kv1_direct_cache_opt_in_
                    ? rpu_pack_pi05_kv1_direct_pair_w8a16(lw.k_w, lw.k_ws)
                    : rpu_pack_pi05_kv1_stripe_w8a16(lw.k_w, lw.k_ws);
                auto compact_v = pi05_kv1_direct_cache_opt_in_
                    ? rpu_pack_pi05_kv1_direct_pair_w8a16(lw.v_w, lw.v_ws)
                    : rpu_pack_pi05_kv1_stripe_w8a16(lw.v_w, lw.v_ws);
                lw.k_kv1_w = std::move(compact_k.first);
                lw.k_kv1_ws = std::move(compact_k.second);
                lw.v_kv1_w = std::move(compact_v.first);
                lw.v_kv1_ws = std::move(compact_v.second);
            }
            pi05_kv1_direct_pair_owners_ = pi05_kv1_direct_cache_opt_in_;
            if (pi05_kv1_direct_cache_opt_in_) {
                size_t direct_owner_bytes = 0;
                for (const auto& lw : layer_weights_) {
                    direct_owner_bytes += lw.k_kv1_w.nbytes() +
                        lw.v_kv1_w.nbytes() + lw.k_kv1_ws.nbytes() +
                        lw.v_kv1_ws.nbytes();
                }
                TORCH_CHECK(
                    direct_owner_bytes == 9455616,
                    "Pi KV1 direct-cache companion footprint drifted from "
                    "9,455,616 bytes: got ", direct_owner_bytes);
            }
        }
    }
    final_norm_dense_w_  = final_norm_dense_w;
    final_norm_dense_b_  = final_norm_dense_b;
    final_norm_dense_ws_ = final_norm_dense_w_scale;
    action_in_proj_w_   = action_in_proj_w;
    action_in_proj_b_   = action_in_proj_b;
    action_out_proj_w_  = action_out_proj_w;
    action_out_proj_b_  = action_out_proj_b;
    TORCH_CHECK(!(pi05_q_rope_epilogue_opt_in_ ||
                  pi05_kv1_direct_cache_opt_in_) ||
                    pi05_q_rope_tables_admitted(cos,sin),
                "Pi optimized RoPE/cache routes require FP16 [rows,128] "
                "tables with safe 32-bit row stride");
    // Q's fused epilogue keeps its accepted immutable table copies. Direct-cache
    // retains the installed table owners without another allocation; its
    // launcher also records both owners as graph keepalives.
    cos_ = pi05_q_rope_epilogue_opt_in_ ? cos.clone() : cos;
    sin_ = pi05_q_rope_epilogue_opt_in_ ? sin.clone() : sin;
    pi05_q_rope_tables_owned_ = pi05_q_rope_epilogue_opt_in_;
    if (pi05_q_rope_tables_owned_) {
        rpu_ddr_flush_force(cos_.data_ptr<c10::Half>());
        rpu_ddr_flush_force(sin_.data_ptr<c10::Half>());
    }

    // Spec §3.3 / §C5: stable RPU staging tensor allocated ONCE — data_ptr
    // valid across step + sample_actions. Passed as hidden_states to satisfy
    // FMB shape contract; layer 0 reads it via FMB's hidden_in_src_base_.
    auto opts = at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1);
    action_emb_stage_ = at::empty({1, chunk_size, hidden_size}, opts);
    adarms_table_ = at::Tensor();
    adarms_table_src_base_ = 0;

    resident_gate_w8_ = at::Tensor();
    resident_up_w8_ = at::Tensor();
    resident_gate_src_base_ = resident_up_src_base_ = 0;
    if (pi05_gateup_resident_opt_in_ && pi05_k_rope_insert_w8a16_
            && hidden_size == 1024 && intermediate_size == 4096
            && num_q_heads == 8 && num_kv_heads == 8 && head_dim == 256
            && num_layers == 18 && chunk_size == 50 && attn_tp() == 8) {
        resident_gate_w8_ = rpu_pack_pi05_gateup_resident_w8(layer_weights_[0].gate_proj_w);
        resident_up_w8_ = rpu_pack_pi05_gateup_resident_w8(layer_weights_[0].up_proj_w);
        resident_gate_src_base_ = RpuGetDevAddr(resident_gate_w8_.data_ptr());
        resident_up_src_base_ = RpuGetDevAddr(resident_up_w8_.data_ptr());
    }

    resident_gateup_bundle_ = at::Tensor();
    resident_gateup_bundle_src_base_ = 0;
    if (pi05_gateup_resident_layers_) {
        TORCH_CHECK(pi05_k_rope_insert_w8a16_ && hidden_size == 1024 &&
                        intermediate_size == 4096 && num_q_heads == 8 &&
                        num_kv_heads == 8 && head_dim == 256 && num_layers == 18 &&
                        chunk_size == 50 && attn_tp() == 8,
                    "Pi0.5 Gate/Up bundle supports only the exact W8A16 action profile");
        std::vector<at::Tensor> cpu_shards;
        for (int64_t layer = 0; layer < pi05_gateup_resident_layers_; ++layer) {
            for (const auto* weight : {&layer_weights_[layer].gate_proj_w,
                                       &layer_weights_[layer].up_proj_w}) {
                // Reuse the checked installed-swizzle decoder; concatenate raw
                // signed bytes by core. This is a cold copy, never requantization.
                cpu_shards.push_back(rpu_pack_pi05_gateup_resident_w8(*weight).cpu());
            }
        }
        resident_gateup_bundle_ = at::cat(cpu_shards, 1).to(gate_w[0].device());
        rpu_ddr_flush_force(resident_gateup_bundle_.data_ptr<int8_t>());
        resident_gateup_bundle_src_base_ = RpuGetDevAddr(resident_gateup_bundle_.data_ptr());
    }

    invalidate_model_state();  // D-503 last non-empty stmt
}

// ============================================================================
// step_forward — Phase 1 stub: validate shapes, do NOT run kernels yet.
// Phase 2 fills run_all_layers call + mask cache.
// ============================================================================
void Pi05DenoiseStepModel::step_forward(
    const at::Tensor& x_t_rpu,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    const at::Tensor& cond_step,
    const at::Tensor& attention_mask_4d,
    at::Tensor& v_t_buf,
    int64_t prefix_len,
    at::IntArrayRef planned_stage_descriptor)
{
    TORCH_CHECK(!pi05_w4_geglu_m50_opt_in_,
                "Pi W4 GeGLU supports only the ten-step loop API");
    TORCH_CHECK(!pi05_xor3_gated_mlp_opt_in_,
                "Pi MLP XOR3+gated supports only the ten-step loop API");
    TORCH_CHECK(!pi05_geglu_m50_opt_in_,
                "Pi M50 GeGLU supports only the ten-step loop API");
    TORCH_CHECK(!pi05_kv1_stripe_gather_opt_in_,
                "Pi KV1 stripe-gather supports only the ten-step loop API");
    TORCH_CHECK(!pi05_kv1_direct_cache_opt_in_,
                "Pi KV1 direct-cache supports only the ten-step loop API");
    TORCH_CHECK(!pi05_q_rope_epilogue_opt_in_,
                "Pi Q-RoPE does not support the single-step API");
    TORCH_CHECK(num_layers() > 0,
                "Pi05DenoiseStepModel::step_forward called before set_weights");
    TORCH_CHECK(x_t_rpu.device().type() == at::kPrivateUse1
                && x_t_rpu.scalar_type() == at::kHalf
                && x_t_rpu.dim() == 3
                && x_t_rpu.size(0) == 1
                && x_t_rpu.size(1) == chunk_size_
                && x_t_rpu.size(2) == max_action_dim_
                && x_t_rpu.is_contiguous(),
                "Pi05DenoiseStepModel::step_forward: x_t_rpu shape "
                "[1,", chunk_size_, ",", max_action_dim_,
                "] fp16 contig RPU required");
    TORCH_CHECK(v_t_buf.sizes() == x_t_rpu.sizes()
                && v_t_buf.scalar_type() == at::kHalf
                && v_t_buf.device().type() == at::kPrivateUse1
                && v_t_buf.is_contiguous(),
                "Pi05DenoiseStepModel::step_forward: v_t_buf shape/dtype mismatch");
    TORCH_CHECK(cond_step.device().type() == at::kPrivateUse1
                && cond_step.scalar_type() == at::kHalf
                && cond_step.dim() == 1
                && cond_step.size(0) == hidden_size()
                && cond_step.is_contiguous(),
                "Pi05DenoiseStepModel::step_forward: cond_step must be 1D [",
                hidden_size(), "] fp16 contig RPU");
    TORCH_CHECK(prefix_len >= 0,
                "Pi05DenoiseStepModel::step_forward: prefix_len must be >= 0");

    // Spec §2.3 runtime assert — kv_cache must have headroom for
    // prefix + suffix. (Layout-aware: max_seq_len stored on the cache
    // is the dim that K's [...sKeyVx, ..., sKeyChunk...] sums to.)
    if (!k_caches.empty()) {
        const at::Tensor& kc = k_caches[0];
        // Re-derive max_seq_len from the cache shape: sKeyVx * sKeyChunk
        // (sKeyVx = dim 1, sKeyChunk = dim 5 in the 7-D K layout — see
        // python/rpu_backend/api/cache.py:97-105).
        const int64_t cache_max_seq = kc.size(1) * kc.size(5);
        TORCH_CHECK(prefix_len + chunk_size_ <= cache_max_seq,
                    "Pi05DenoiseStepModel::step_forward: prefix_len(",
                    prefix_len, ") + chunk_size(", chunk_size_,
                    ") exceeds k_cache max_seq_len=", cache_max_seq);
    }

    restore_planned_route_profile(
        planned_stage_descriptor, /*loop_mode=*/false, /*num_steps=*/1,
        prefix_len, k_caches.empty() ? prefix_len + chunk_size_
                                    : k_caches[0].size(1) * k_caches[0].size(5));

    // Mode-switch guard: a prior denoise_loop_forward call may have left
    // loop_mode_==true (which would set body_iterations=num_steps_ and corrupt
    // this single-step forward). Reset to the single-step graph before building.
    if (loop_mode_) {
        loop_mode_ = false;
        num_steps_ = 1;
        dt_pinned_ = false;  // loop graph baked dt_; single-step must rebuild cleanly
        invalidate_model_state(/*planning_domain_changed=*/false);
    }
    pi05_graph_dma_policy_ = 0;
    validate_planned_route_profile(/*loop_mode=*/false, /*num_steps=*/1);
    TORCH_CHECK(prepare_pi05_graph_dma_policy(planned_stage_descriptor) == 0,
                "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 single-step forward "
                "cannot consume an optimized ten-step DMA descriptor");
    TORCH_CHECK(
        !RpuKernelGraph::has_active() || !planned_stage_descriptor.empty(),
        "Pi0.5 denoise production Graph forward requires one complete "
        "planner stage descriptor");

    // Per-step bookkeeping.
    const int64_t action_chunk = configured_chunk_size_ > 0
        ? configured_chunk_size_ : ((chunk_size_ + 15) / 16) * 16;
    set_chunk_size_override(action_chunk);
    // Keepalives across the deferred graph — pitfalls.md C-1. All three bases
    // below are dereferenced when the graph EXECUTES (RpuKernelGraph::end(),
    // i.e. when the caller's `with cache.capture(sig):` exits), not when this
    // function returns; a caller that drops its local frees the DDR block and
    // the driver may re-issue its device VA before then. Same shape as
    // denoise_loop_forward's x0_ref_/cond_all_ref_/x_out_ref_ below. v_t_buf is
    // the worst case: it is a WRITE destination, so a re-issued VA means the
    // graph writes over whatever now owns that block, not merely reads garbage.
    x_t_ref_  = x_t_rpu;
    cond_ref_ = cond_step;
    v_t_ref_  = v_t_buf;

    // Spec §4.1 — update mutable bases + flush caller's DDR-dirty lines
    // (FMB applies its own Boundary guard inside run_all_layers; calling
    // rpu_ddr_flush_force here is safe and required for graph BUILD).
    rpu_ddr_flush_force(x_t_rpu.data_ptr<c10::Half>());
    rpu_ddr_flush_force(cond_step.data_ptr<c10::Half>());
    rpu_ddr_flush_force(v_t_buf.data_ptr<c10::Half>());

    x_t_src_base_  = ::rhino_lkn::RpuGetDevAddr(x_t_rpu.data_ptr());
    cond_src_base_ = ::rhino_lkn::RpuGetDevAddr(cond_step.data_ptr());
    v_t_dst_base_  = ::rhino_lkn::RpuGetDevAddr(v_t_buf.data_ptr());

    // SDPA mask cache — mirror AdaRMSModel pattern (rpu_adarms_model.cpp:255-289).
    // Factored into a shared helper (also called by denoise_loop_forward).
    prepare_sdpa_mask_cached(attention_mask_4d, prefix_len);

    // Drive run_all_layers with action_emb_stage_ as the FMB-contract
    // hidden_states tensor (satisfies hidden_size_=1024 shape check). The
    // actual numerical input flowed via emit_pre_layers_body ([A1..A4])
    // → action_emb_stage_ DDR; layer 0's emit_layer_input_dma reads it
    // through FMB's hidden_in_src_base_ mutable path automatically.
    (void) run_all_layers(action_emb_stage_, k_caches, v_caches,
                          std::optional<at::Tensor>(attention_mask_4d),
                          /*position=*/prefix_len, /*is_causal=*/false,
                          /*planned_chunk_size=*/0,
                          planned_stage_descriptor);

    // Output v_t was written to v_t_buf via emit_post_layers_body ([Z3] DMA).
}

// ============================================================================
// prepare_sdpa_mask_cached — shared SDPA mask prepare + cache.
// Called before FMB/Graph replay by both forward forms. The first-body SPM
// upload reads this stable DDR slot on EVERY invocation, so changed request
// masks must be published even when the host layer body is skipped.
// ============================================================================
void Pi05DenoiseStepModel::prepare_sdpa_mask_cached(
    const at::Tensor& attention_mask_4d, int64_t prefix_len)
{
    const int64_t prep_seq_q = chunk_size_;
    const int64_t prep_seq_k = prefix_len + chunk_size_;
    const bool    prep_is_causal = false;
    const bool    has_mask = attention_mask_4d.defined();
    const bool version_enabled = has_mask &&
        attention_mask_4d.unsafeGetTensorImpl()->version_counter().enabled();
    const int64_t input_version = version_enabled
        ? static_cast<int64_t>(attention_mask_4d.unsafeGetTensorImpl()
                                   ->version_counter().current_version())
        : -1;
    const bool cache_hit = version_enabled
                        && prepared_mask_input_ref_.defined()
                        && prepared_mask_input_ref_.is_same(attention_mask_4d)
                        && prepared_mask_input_version_ == input_version
                        && (prepared_mask_seq_q_ == prep_seq_q)
                        && (prepared_mask_seq_k_ == prep_seq_k)
                        && (prepared_mask_is_causal_ == prep_is_causal);
    if (!cache_hit) {
        // A failed preparation may already have rewritten the stable slot.
        // Retire the old stamp first; retries must never accept stale content.
        prepared_mask_input_ref_ = at::Tensor{};
        prepared_mask_input_version_ = -1;
        PreparedMask next_prepared_mask = sdpa_prepare_mask(
            has_mask ? std::optional<at::Tensor>(attention_mask_4d)
                     : std::nullopt,
            prep_is_causal, prep_seq_q, prep_seq_k,
            sdpa_stable_mask_cache());
        TORCH_CHECK(
            !version_enabled ||
                attention_mask_4d.unsafeGetTensorImpl()->version_counter()
                    .current_version() == input_version,
            "Pi0.5 denoise attention mask mutated while its stable Graph "
            "input was being prepared");
        prepared_mask_ = std::move(next_prepared_mask);
        prepared_mask_input_ref_  = has_mask ? attention_mask_4d : at::Tensor();
        prepared_mask_input_version_ = input_version;
        prepared_mask_seq_q_      = prep_seq_q;
        prepared_mask_seq_k_      = prep_seq_k;
        prepared_mask_is_causal_  = prep_is_causal;
    }
}

// ============================================================================
// denoise_loop_forward — Task 1 scaffolding: N-step in-graph unroll.
//
// Gated by loop_mode_: static_config() sets body_iterations=num_steps_, so
// run_all_layers loops the body (and the pre/post hooks) num_steps times in a
// single graph. Task 1 only wires the plumbing — the per-iteration cond load
// (Task 2) and on-device Euler (Task 3) are NOT here yet, so the post-hook
// still emits the single-step v_t output. No Python calls this op yet, so the
// single-step pi05_denoise_step_forward path stays byte-identical (loop_mode_
// is false unless this op is invoked).
// ============================================================================
void Pi05DenoiseStepModel::denoise_loop_forward(
    at::Tensor x0_rpu, std::vector<at::Tensor> k_caches,
    std::vector<at::Tensor> v_caches, at::Tensor cond_all,
    at::Tensor attention_mask_4d, at::Tensor x_out,
    double dt, int64_t prefix_len, int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor) {
    TORCH_CHECK(num_layers() > 0,
                "Pi05DenoiseStepModel::denoise_loop_forward called before set_weights");
    const int64_t cs  = chunk_size_;
    const int64_t mad = max_action_dim_;
    TORCH_CHECK(x0_rpu.dim() == 3 && x0_rpu.size(0) == 1 && x0_rpu.size(1) == cs
        && x0_rpu.size(2) == mad && x0_rpu.scalar_type() == at::kHalf
        && x0_rpu.is_contiguous() && x0_rpu.device().type() == at::kPrivateUse1,
        "denoise_loop_forward: x0_rpu must be [1,", cs, ",", mad, "] fp16 contig rpu");
    TORCH_CHECK(cond_all.dim() == 2 && cond_all.size(0) == num_steps
        && cond_all.size(1) == hidden_size() && cond_all.scalar_type() == at::kHalf
        && cond_all.is_contiguous() && cond_all.device().type() == at::kPrivateUse1,
        "denoise_loop_forward: cond_all must be [", num_steps, ",", hidden_size(),
        "] fp16 contig rpu");
    TORCH_CHECK(x_out.sizes() == x0_rpu.sizes() && x_out.scalar_type() == at::kHalf
        && x_out.is_contiguous() && x_out.device().type() == at::kPrivateUse1,
        "denoise_loop_forward: x_out must match x0_rpu shape, fp16 contig rpu");
    TORCH_CHECK(num_steps >= 1, "num_steps must be >= 1");
    // Node-count guard: a batch overflow is SILENT (build_batch return ignored).
    // Cap is pending.size() <= 32768; budget ~1100 nodes/body with margin.
    TORCH_CHECK(num_steps * 1100 < 32000,
        "denoise_loop_forward: num_steps(", num_steps, ") * 1100 exceeds 32000 node cap");
    TORCH_CHECK(prefix_len >= 0, "denoise_loop_forward: prefix_len must be >= 0");
    if (!k_caches.empty()) {
        const at::Tensor& k0 = k_caches[0];
        const int64_t cache_max_seq = k0.size(1) * k0.size(5);
        TORCH_CHECK(prefix_len + cs <= cache_max_seq,
            "denoise_loop_forward: prefix_len(", prefix_len, ") + chunk(", cs,
            ") exceeds k_cache max_seq_len=", cache_max_seq);
    }

    // Reject cold/descriptor/position drift before mode mutation or launch.
    TORCH_CHECK(!pi05_w4_geglu_m50_opt_in_ ||
        (num_steps == 10 && pi05_action_prefix_admitted(prefix_len) &&
         attention_mask_4d.sizes() == at::IntArrayRef({1, 1, 50, prefix_len + 50})),
        "Pi W4 GeGLU loop arguments changed the exact ten-step profile");
    TORCH_CHECK(!pi05_xor3_gated_mlp_opt_in_ ||
        (pi05_action_steps_admitted(num_steps) && pi05_action_prefix_admitted(prefix_len) &&
         attention_mask_4d.sizes() == at::IntArrayRef({1, 1, 50, prefix_len + 50})),
        "Pi MLP XOR3+gated loop arguments changed the exact four/ten-step profile");
    TORCH_CHECK(!pi05_geglu_m50_opt_in_ ||
        (pi05_action_steps_admitted(num_steps) && pi05_action_prefix_admitted(prefix_len) &&
         attention_mask_4d.sizes() == at::IntArrayRef({1, 1, 50, prefix_len + 50})),
        "Pi M50 GeGLU loop arguments changed the exact four/ten-step profile");
    TORCH_CHECK(!pi05_q_rope_epilogue_opt_in_ ||
        (num_steps==10 && pi05_action_prefix_admitted(prefix_len) &&
         attention_mask_4d.sizes()==at::IntArrayRef({1,1,50,prefix_len + 50})),
        "Pi Q-RoPE loop arguments changed the exact ten-step profile");
    // dt baked into the in-graph Euler at BUILD (Task 3); identical across the N
    // unrolled iterations. Pin it now so the contract holds once Task 3 reads it.
    TORCH_CHECK(dt < 0.0, "denoise_loop_forward: dt must be < 0 (Pi05 flow-matching dt = -1/num_steps)");
    const c10::Half dt_half = c10::Half(static_cast<float>(dt));

    restore_planned_route_profile(
        planned_stage_descriptor, /*loop_mode=*/true, num_steps,
        prefix_len, k_caches.empty() ? prefix_len + cs
                                    : k_caches[0].size(1) * k_caches[0].size(5));

    // (Re)build when entering loop mode, or when loop cardinality / horizon changes.
    if (!loop_mode_ || num_steps_ != num_steps || chunk_size_ != cs) {
        loop_mode_ = true;
        num_steps_ = num_steps;
        dt_pinned_ = false;
        invalidate_model_state(/*planning_domain_changed=*/false);
    }
    pi05_graph_dma_policy_ = 0;
    validate_planned_route_profile(/*loop_mode=*/true, num_steps);
    validate_pi05_storage_policy(planned_stage_descriptor);
    pi05_graph_dma_policy_ = prepare_pi05_graph_dma_policy(planned_stage_descriptor);
    TORCH_CHECK(
        !RpuKernelGraph::has_active() || !planned_stage_descriptor.empty(),
        "Pi0.5 denoise loop production Graph forward requires one complete "
        "planner stage descriptor");
    if (!dt_pinned_) { dt_ = dt_half; dt_pinned_ = true; }
    else TORCH_CHECK(dt_ == dt_half, "dt changed across replays (baked into the graph)");

    const int64_t action_chunk = configured_chunk_size_ > 0
        ? configured_chunk_size_ : ((chunk_size_ + 15) / 16) * 16;
    set_chunk_size_override(action_chunk);

    // SDPA mask cache (shared helper; same block step_forward uses).
    prepare_sdpa_mask_cached(attention_mask_4d, prefix_len);

    // Keepalives across the synchronous forward + flush caller's DDR-dirty lines.
    x0_ref_ = x0_rpu; cond_all_ref_ = cond_all; x_out_ref_ = x_out;
    rpu_ddr_flush_force(x0_rpu.data_ptr<c10::Half>());
    rpu_ddr_flush_force(cond_all.data_ptr<c10::Half>());
    rpu_ddr_flush_force(x_out.data_ptr<c10::Half>());
    x0_src_base_       = ::rhino_lkn::RpuGetDevAddr(x0_rpu.data_ptr());
    cond_all_src_base_ = ::rhino_lkn::RpuGetDevAddr(cond_all.data_ptr());
    x_out_dst_base_    = ::rhino_lkn::RpuGetDevAddr(x_out.data_ptr());
    // The loop-mode pre/post hooks read x0_src_base_ / cond_all_src_base_ (with
    // per-body offsets) and x_out_dst_base_ directly — no single-step base bridge.

    // ONE forward — run_all_layers loops body_iterations(=num_steps_) internally;
    // the pre/post hooks fire per iteration with ctx().body_iter advancing.
    (void) run_all_layers(action_emb_stage_, k_caches, v_caches,
                          std::optional<at::Tensor>(attention_mask_4d),
                          /*position=*/prefix_len, /*is_causal=*/false,
                          /*planned_chunk_size=*/0,
                          planned_stage_descriptor);
}

// ============================================================================
// FMB virtuals — Phase 1 skeleton (Phase 2 fills declare_buffers + build_layer_subgraph)
// ============================================================================
ModelStaticConfig Pi05DenoiseStepModel::static_config() {
    ModelStaticConfig cfg;
    cfg.num_layers             = num_layers();
    cfg.cross_layer_batch_size = num_layers();
    // EXT-Pi05 hooks (Phase 2 fills the bodies).
    cfg.pre_layers_fn  = reinterpret_cast<void (FusedModelBase::*)()>(
        &Pi05DenoiseStepModel::emit_pre_layers_body);
    cfg.post_layers_fn = reinterpret_cast<void (FusedModelBase::*)()>(
        &Pi05DenoiseStepModel::emit_post_layers_body);
    cfg.body_iterations = loop_mode_ ? num_steps_ : 1;  // EXT-unroll (denoise loop)
    return cfg;
}

ModelDynamicConfig Pi05DenoiseStepModel::dynamic_config(const ChunkPlan& /*plan*/) {
    ModelDynamicConfig cfg;
    cfg.chunk_mode     = ChunkMode::SEQUENTIAL;
    cfg.inter_layer_io = InterLayerIO::AUTO;
    cfg.attention_policy = AttentionExecutionPolicy::DDR_KV;
    return cfg;
}

bool Pi05DenoiseStepModel::subclass_chunk_size_valid(
    int64_t chunk_size, int64_t seq_len, int64_t /*position*/) const {
    // The action suffix is bidirectional. Splitting it would hide future
    // action rows from an earlier chunk.
    return chunk_size >= seq_len;
}

FmbPhysicalExecutionManifest
Pi05DenoiseStepModel::physical_manifest_for_candidate(
    const FmbThreeStageChunkPlan& plan, const LayoutContext& layout,
    int64_t physical_len, int64_t logical_len, int64_t position) const {
    TORCH_CHECK(
        planned_route_profile_valid_ && layout.use_attn_mask &&
            !layout.is_causal && plan.compute.chunks.size() == 1,
        "Pi0.5 denoise COMPLETE descriptor requires one masked action chunk");

    FmbPhysicalExecutionManifest manifest;
    manifest.state = FmbPhysicalManifestState::COMPLETE;
    manifest.logical_length = logical_len;
    manifest.physical_length = physical_len;
    manifest.execution_padding_rows = physical_len - logical_len;
    manifest.kv_logical_length = position + logical_len;
    manifest.kv_insert_physical_rows = planned_kv_physical_rows_;
    manifest.graph_lifecycle = FmbGraphLifecycle::COMPOSITE_CHILD;
    // The cold policy applies to decoder, condition and action projections.
    // NVFP4 uses the same selected accumulation with its striped-v2 ABI.
    manifest.linear_accumulation = linear_accumulation_policy();

    auto append = [&](FmbRouteFamily family, int64_t site_id,
                      int64_t selector,
                      std::vector<int64_t> arguments = {},
                      int64_t invocation = 0) {
        manifest.routes.push_back({
            site_id, family, selector, /*flags=*/0,
            std::move(arguments), invocation});
    };
    auto append_linear = [&](int64_t site_id, int64_t invocation = 0) {
        append(FmbRouteFamily::LINEAR, site_id,
               static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
               {}, invocation);
    };
    auto append_dma = [&](int64_t site_id,
                          Pi05DenoiseMutableDmaRoute route) {
        append(FmbRouteFamily::MUTABLE_DMA, site_id,
               static_cast<int64_t>(route));
    };

    if (num_cores() != 8) {
        append(FmbRouteFamily::GRAPH_SCHEDULE, PI05_DENOISE_CORE_PROFILE_SITE,
               1, core_profile_arguments());
    }
    if (condition_tp() != num_cores()) {
        append(FmbRouteFamily::ALL_REDUCE, PI05_DENOISE_COND_PREPARE_SITE,
               3, {condition_tp(), num_cores(), 1});
    }
    if (pi05_w4_fastpath_opt_in_) {
        append(FmbRouteFamily::GRAPH_SCHEDULE, PI05_DENOISE_W4_FASTPATH_POLICY_SITE,
               1, pi05_w4_fastpath_arguments());
    }
    if (pi05_nvfp4_geglu_m50_opt_in_) {
        TORCH_CHECK(use_pi05_nvfp4_geglu_m50(physical_len) &&
            pi05_graph_dma_shape_admitted(plan, physical_len, logical_len, position,
                layout.batch_size, layout.use_attn_mask, layout.is_causal, layout.attention_policy),
            "Pi NVFP4 GeGLU native candidate changed the exact cold profile");
        append(FmbRouteFamily::GRAPH_SCHEDULE, PI05_DENOISE_NVFP4_GEGLU_POLICY_SITE,
               1, pi05_nvfp4_geglu_m50_arguments());
        append(FmbRouteFamily::LINEAR, PI05_DENOISE_NVFP4_GEGLU_LINEAR_SITE,
               PI05_DENOISE_NVFP4_GEGLU_ACC16_M320N64K128, pi05_nvfp4_geglu_m50_arguments());
    }
    if (pi05_w4_geglu_m50_opt_in_) {
        TORCH_CHECK(
            use_pi05_w4_geglu_m50(physical_len) &&
                pi05_graph_dma_shape_admitted(
                    plan, physical_len, logical_len, position,
                    layout.batch_size, layout.use_attn_mask,
                    layout.is_causal, layout.attention_policy),
            "Pi W4 M50 GeGLU native candidate changed the exact cold profile");
        append(FmbRouteFamily::GRAPH_SCHEDULE,
               PI05_DENOISE_W4_GEGLU_M50_POLICY_SITE, 1,
               pi05_w4_geglu_m50_arguments());
        append(FmbRouteFamily::LINEAR, PI05_DENOISE_W4_GEGLU_M50_LINEAR_SITE,
               pi05_w4_geglu_m50_selector(),
               pi05_w4_geglu_m50_arguments());
    }
    if (pi05_geglu_m50_opt_in_) {
        TORCH_CHECK(
            use_pi05_geglu_m50(physical_len) &&
                pi05_graph_dma_shape_admitted(
                    plan, physical_len, logical_len, position,
                    layout.batch_size, layout.use_attn_mask,
                    layout.is_causal, layout.attention_policy),
            "Pi M50 GeGLU native candidate changed the exact cold profile");
        append(FmbRouteFamily::GRAPH_SCHEDULE,
               PI05_DENOISE_GEGLU_M50_POLICY_SITE, 1,
               pi05_geglu_m50_arguments());
        append(FmbRouteFamily::LINEAR, PI05_DENOISE_GEGLU_M50_LINEAR_SITE,
               pi05_geglu_m50_selector(pi05_is_fp16_action(layer_weights_)),
               pi05_geglu_m50_arguments());
    }
    if (pi05_xor3_gated_mlp_opt_in_) {
        TORCH_CHECK(
            use_pi05_xor3_gated_mlp(physical_len) &&
                pi05_graph_dma_shape_admitted(
                    plan, physical_len, logical_len, position,
                    layout.batch_size, layout.use_attn_mask,
                    layout.is_causal, layout.attention_policy),
            "Pi MLP XOR3+gated native candidate changed the exact cold profile");
        append(FmbRouteFamily::GRAPH_SCHEDULE,
               PI05_DENOISE_XOR3_GATED_MLP_POLICY_SITE, 1,
               pi05_xor3_gated_mlp_arguments());
        append(FmbRouteFamily::ALL_REDUCE,
               PI05_DENOISE_XOR3_GATED_MLP_REDUCE_SITE,
               PI05_DENOISE_XOR3_GATED_MLP_M50N1024,
               pi05_xor3_gated_mlp_arguments());
    }
    append(FmbRouteFamily::GRAPH_SCHEDULE,PI05_DENOISE_Q_EPILOGUE_POLICY_SITE,
           1,pi05_q_rope_policy_arguments(pi05_q_rope_epilogue_opt_in_, planned_prefix_len_, planned_num_steps_));
    if (pi05_kv1_stripe_gather_opt_in_) {
        append(FmbRouteFamily::GRAPH_SCHEDULE, PI05_DENOISE_KV1_POLICY_SITE,
               1, pi05_kv1_policy_arguments(/*enabled=*/true, planned_prefix_len_));
    }
    if (pi05_kv1_direct_cache_opt_in_) {
        append(
            FmbRouteFamily::GRAPH_SCHEDULE,
            PI05_DENOISE_KV1_DIRECT_POLICY_SITE, 1,
            pi05_kv1_pair_owner_opt_in_
                ? pi05_kv1_pair_owner_policy_arguments(
                      /*enabled=*/pi05_kv1_pair_owner_opt_in_,
                      rope_position_ >= 0 ? rope_position_ : position,
                      planned_prefix_len_, pi05_is_fp16_action(layer_weights_), planned_num_steps_)
                : pi05_kv1_direct_policy_arguments(
                      /*enabled=*/true,
                      rope_position_ >= 0 ? rope_position_ : position, planned_prefix_len_));
    }
    TORCH_CHECK(!pi05_q_rope_epilogue_opt_in_ ||
        (use_pi05_q_rope_epilogue(physical_len,rope_position_>=0?rope_position_:position) &&
         pi05_graph_dma_shape_admitted(plan,physical_len,logical_len,position,
             layout.batch_size,layout.use_attn_mask,layout.is_causal,layout.attention_policy)),
        "Pi Q-RoPE native candidate does not match the exact cold owner");
    append_dma(PI05_DENOISE_PRE_X_DMA_SITE,
               Pi05DenoiseMutableDmaRoute::DDR_BROADCAST_TO_SPM);
    const bool table_route = planned_loop_mode_ && adarms_table_.defined();
    if (use_pi05_gateup_resident(physical_len) && logical_len == 50) {
        for (int64_t projection = 0; projection < 2; ++projection) {
            append(FmbRouteFamily::LINEAR, PI05_DENOISE_GATEUP_RESIDENT_SITE,
                   PI05_DENOISE_SPM_WEIGHT_M512N48K128,
                   pi05_gateup_resident_arguments(), projection);
            append(FmbRouteFamily::MUTABLE_DMA, PI05_DENOISE_GATEUP_PRELOAD_SITE,
                   /*raw W8 scatter once per graph=*/5,
                   pi05_gateup_resident_arguments(), projection);
        }
    }
    if (use_pi05_gateup_bundle(physical_len) && logical_len == 50) {
        append(FmbRouteFamily::MUTABLE_DMA, PI05_DENOISE_BUNDLE_PRELOAD_SITE,
               5, pi05_gateup_bundle_arguments(pi05_gateup_resident_layers_));
        for (int64_t layer = 0; layer < pi05_gateup_resident_layers_; ++layer) {
            for (int64_t projection = 0; projection < 2; ++projection) {
                append(FmbRouteFamily::LINEAR, PI05_DENOISE_BUNDLE_LINEAR_SITE,
                       PI05_DENOISE_SPM_WEIGHT_M512N48K128,
                       pi05_gateup_bundle_linear_arguments(
                           pi05_gateup_resident_layers_, layer, projection),
                       2 * layer + projection);
            }
        }
    }
    if (!table_route) {
        append_dma(PI05_DENOISE_PRE_COND_DMA_SITE,
                   Pi05DenoiseMutableDmaRoute::DDR_SCATTER_TO_SPM);
    }
    append(
        FmbRouteFamily::LINEAR, PI05_DENOISE_PRE_LINEAR_SITE,
        static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
        {planned_loop_mode_ ? 1 : 0, planned_num_steps_});
    append_linear(PI05_DENOISE_POST_LINEAR_SITE);
    const int64_t norm_route = static_cast<int64_t>(use_spm_adarms_table()
        ? Pi05DenoiseNormRoute::FUSED_H1024
        : Pi05DenoiseNormRoute::RMS_THEN_SHIFT);
    append(FmbRouteFamily::NORMALIZATION, PI05_DENOISE_FINAL_NORM_SITE,
           norm_route, {physical_len, hidden_size()});
    const bool graph_dma_profile = use_pi05_graph_dma_profile(physical_len) &&
        pi05_graph_dma_shape_admitted(plan, physical_len, logical_len, position,
            layout.batch_size, layout.use_attn_mask, layout.is_causal,
            layout.attention_policy);
    const bool elide_unused = graph_dma_profile && pi05_elide_unused_dma_opt_in_;
    if (elide_unused) {
        append(FmbRouteFamily::GRAPH_SCHEDULE, PI05_DENOISE_UNUSED_OUTPUT_SITE,
               /*last-layer SPM consumer, no DDR materialization=*/1,
               pi05_graph_dma_arguments(planned_prefix_len_));
        append(FmbRouteFamily::MUTABLE_DMA, PI05_DENOISE_POST_LOOP_DMA_SITE,
               static_cast<int64_t>(Pi05DenoiseMutableDmaRoute::SPM_COPY_FINAL_BODY_TO_DDR),
               pi05_graph_dma_arguments(planned_prefix_len_));
    } else {
        append_dma(
            planned_loop_mode_ ? PI05_DENOISE_POST_LOOP_DMA_SITE
                               : PI05_DENOISE_POST_STEP_DMA_SITE,
            Pi05DenoiseMutableDmaRoute::SPM_COPY_TO_DDR);
    }
    const bool spm_staging = graph_dma_profile && pi05_spm_action_staging_opt_in_;
    if (spm_staging) {
        append(FmbRouteFamily::GRAPH_SCHEDULE, PI05_DENOISE_ACTION_SPM_PRODUCER_SITE,
               /*core0 action GEMM output stays in SPM=*/1, pi05_graph_dma_arguments(planned_prefix_len_));
        append(FmbRouteFamily::GRAPH_SCHEDULE, PI05_DENOISE_ACTION_SPM_INPUT_SITE,
               /*core0 source to all8 residual1 slots, stride0=*/1, pi05_graph_dma_arguments(planned_prefix_len_));
    }
    if (use_pi05_adarms_body_stream()) {
        for (int64_t body = 0; body < 10; ++body) {
            append(FmbRouteFamily::MUTABLE_DMA, PI05_DENOISE_ADARMS_TABLE_DMA_SITE,
                   static_cast<int64_t>(Pi05DenoiseMutableDmaRoute::DDR_BROADCAST_TABLE_BODY),
                   pi05_adarms_body_arguments(body), body);
        }
    } else if (table_route) {
        append(FmbRouteFamily::MUTABLE_DMA, PI05_DENOISE_ADARMS_TABLE_DMA_SITE,
               static_cast<int64_t>(use_spm_adarms_table()
                   ? Pi05DenoiseMutableDmaRoute::DDR_BROADCAST_TABLE_ONCE
                   : Pi05DenoiseMutableDmaRoute::DDR_BROADCAST_TO_SPM),
               {adarms_table_.size(0), num_layers()});
    } else {
        append_linear(PI05_DENOISE_GEMV_LINEAR_SITE);
        append(
            FmbRouteFamily::ALL_REDUCE,
            PI05_DENOISE_GEMV_ALL_REDUCE_SITE,
            pi05_denoise_ring_route(/*rows=*/1, /*cols=*/3 * hidden_size(), num_cores()));
    }

    constexpr uint32_t kv_capabilities =
        KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 |
        KV_INSERT_CAP_PAD16 | KV_INSERT_CAP_HYBRID2 |
        KV_INSERT_CAP_HYBRID3;
    append(FmbRouteFamily::GRAPH_SCHEDULE,
           PI05_DENOISE_ZERO_GATED_RESIDUAL_SITE,
           pi05_denoise_residual_route(physical_len), {hidden_size()});
    const int64_t rope_base = rope_position_ >= 0
        ? rope_position_ : position;
    for (const ChunkInfo& chunk : plan.compute.chunks) {
        const int64_t invocation = chunk.idx;
        for (const int64_t site : {PI05_DENOISE_ATTN_NORM_SITE,
                                   PI05_DENOISE_MLP_NORM_SITE}) {
            append(FmbRouteFamily::NORMALIZATION, site, norm_route,
                   {chunk.len, hidden_size()}, invocation);
        }
        if (pi05_q_rope_epilogue_opt_in_)
            append(FmbRouteFamily::LINEAR,PI05_DENOISE_Q_EPILOGUE_LINEAR_SITE,
                   PI05_DENOISE_Q_PAIR16_ROPE,pi05_q_rope_arguments(rope_base+chunk.offset),invocation);
        else
            append_linear(PI05_DENOISE_Q_LINEAR_SITE, invocation);
        const bool kv1_stripe = use_pi05_kv1_stripe_gather(chunk.len);
        const bool kv1_direct = use_pi05_kv1_direct_cache(chunk.len);
        const bool kv1_pair_owner = use_pi05_kv1_pair_owner(chunk.len);
        if (kv1_pair_owner) {
            append(FmbRouteFamily::LINEAR,
                   PI05_DENOISE_KV1_PAIR_OWNER_SITE,
                   pi05_kv1_pair_owner_selector(pi05_is_fp16_action(layer_weights_)),
                   pi05_kv1_pair_owner_linear_arguments(planned_prefix_len_, pi05_is_fp16_action(layer_weights_), planned_num_steps_), invocation);
        } else if (kv1_direct) {
            append(FmbRouteFamily::LINEAR, PI05_DENOISE_K_LINEAR_SITE,
                   PI05_DENOISE_KV1_DIRECT_M592N32,
                   pi05_kv1_direct_linear_arguments(planned_prefix_len_), invocation);
            append(FmbRouteFamily::LINEAR, PI05_DENOISE_V_LINEAR_SITE,
                   PI05_DENOISE_KV1_DIRECT_M592N32,
                   pi05_kv1_direct_linear_arguments(planned_prefix_len_), invocation);
        } else if (kv1_stripe) {
            append(FmbRouteFamily::LINEAR, PI05_DENOISE_K_LINEAR_SITE,
                   PI05_DENOISE_KV1_FIXED_M592N32,
                   pi05_kv1_linear_arguments(planned_prefix_len_), invocation);
            append(FmbRouteFamily::COLLECTIVE,
                   PI05_DENOISE_KV1_K_GATHER_SITE,
                   static_cast<int64_t>(RpuAllGatherSchedule::LITTLE_CHUNK),
                   pi05_kv1_gather_arguments(planned_prefix_len_), invocation);
            append(FmbRouteFamily::LINEAR, PI05_DENOISE_V_LINEAR_SITE,
                   PI05_DENOISE_KV1_FIXED_M592N32,
                   pi05_kv1_linear_arguments(planned_prefix_len_), invocation);
            append(FmbRouteFamily::COLLECTIVE,
                   PI05_DENOISE_KV1_V_GATHER_SITE,
                   static_cast<int64_t>(RpuAllGatherSchedule::LITTLE_CHUNK),
                   pi05_kv1_gather_arguments(planned_prefix_len_), invocation);
        } else {
            append_linear(PI05_DENOISE_K_LINEAR_SITE, invocation);
            append_linear(PI05_DENOISE_V_LINEAR_SITE, invocation);
        }
        if (!pi05_q_rope_epilogue_opt_in_) append(
            FmbRouteFamily::ROPE, PI05_DENOISE_Q_ROPE_SITE,
            static_cast<int64_t>(head_dim() == 256
                ? Pi05DenoiseRopeRoute::FULL_ROPE_TILED : Pi05DenoiseRopeRoute::ROPE_1D),
            {rope_base + chunk.offset}, invocation);
        const KvInsertSegmentPlan kv_plan =
            resolve_kvinsert_plan_auto(
                PI05_DENOISE_KV_INSERT_SITE, manifest.graph_lifecycle,
                position + chunk.offset, chunk.len,
                planned_kv_physical_rows_,
                attn_tp(), num_kv_heads(), head_dim(), kv_capabilities);
        const bool k_fused = physical_len == 50 && logical_len == 50 &&
            chunk.offset == 0 && layout.attention_policy ==
                AttentionExecutionPolicy::DDR_KV &&
            !kv1_direct &&
            use_pi05_k_rope_insert(chunk.len, position,
                rope_base + chunk.offset, layout.batch_size, kv_plan);
        const int64_t kv_flags = kv1_direct
            ? PI05_DENOISE_KV1_DIRECT_CACHE_FLAG
            : (k_fused ? PI05_DENOISE_K_ROPE_INSERT_FLAG : 0) |
                (kv1_stripe ? PI05_DENOISE_KV1_GATHERED_SOURCE_FLAG : 0);
        manifest.routes.push_back({
            PI05_DENOISE_K_ROPE_SITE, FmbRouteFamily::ROPE,
            static_cast<int64_t>(head_dim() == 256
                ? Pi05DenoiseRopeRoute::FULL_ROPE_TILED : Pi05DenoiseRopeRoute::ROPE_1D),
            kv_flags,
            k_fused ? pi05_k_rope_insert_arguments(rope_base + chunk.offset,
                          planned_k_rope_cache_capacity_, planned_k_rope_cache_capacity_, planned_prefix_len_)
                    : std::vector<int64_t>{rope_base + chunk.offset}, invocation});
        const KvInsertRouteArguments kv_arguments =
            rpu_kvinsert_route_arguments(
                kv_plan, attn_tp(), num_kv_heads(), head_dim());
        manifest.routes.push_back({
            PI05_DENOISE_KV_INSERT_SITE, FmbRouteFamily::KV_INSERT,
            static_cast<int64_t>(kv_plan.route()),
            kv_flags,
            {kv_arguments.begin(), kv_arguments.end()}, invocation});
        append(
            FmbRouteFamily::ATTENTION, PI05_DENOISE_ATTENTION_SITE,
            static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
            {}, invocation);
        // This route is part of the physical descriptor: a retained Graph
        // using the old per-layer schedule cannot masquerade as this graph.
        append(
            FmbRouteFamily::GRAPH_SCHEDULE, PI05_DENOISE_MASK_SCHEDULE_SITE,
            static_cast<int64_t>(
                Pi05DenoiseMaskScheduleRoute::FIRST_BODY_FIRST_LAYER),
            {chunk.len, position + physical_len, attn_tp(), num_cores(),
             planned_loop_mode_ ? 1 : 0, planned_num_steps_, num_layers()},
            invocation);
        append(
            FmbRouteFamily::ALL_REDUCE,
            PI05_DENOISE_PREPARE_ALL_REDUCE_SITE,
            static_cast<int64_t>(
                Pi05DenoiseAllReduceRoute::PREPARE_RING_INPUT),
            {}, invocation);
        append_linear(PI05_DENOISE_O_LINEAR_SITE, invocation);
        append(
            FmbRouteFamily::ALL_REDUCE,
            PI05_DENOISE_ATTENTION_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(chunk.len, hidden_size(), use_pi05_ring_xor3(chunk.len), num_cores()),
            {}, invocation);
    }
    append_fmb_shared_runtime_routes(
        manifest, plan, hidden_size(),
        (spm_staging ? 0 : FMB_SHARED_LAYER_INPUT_DMA) |
            FMB_SHARED_MLP_AUTO_TILE |
            (pi05_xor3_gated_mlp_opt_in_ ? 0 : FMB_SHARED_MLP_RING_REDUCE),
        1, use_pi05_ring_xor3(50), num_cores(), mlp_tp());
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
    if (nvfp4_.enabled()) {
        for (auto& route : manifest.routes) {
            if (route.family == FmbRouteFamily::LINEAR &&
                (route.site_id == PI05_DENOISE_Q_LINEAR_SITE || route.site_id == PI05_DENOISE_O_LINEAR_SITE ||
                 route.site_id == FMB_SHARED_MLP_AUTO_TILE_SITE))
                route.selector = static_cast<int64_t>((linear_acc32_ ? FmbLinearRouteSelector::PI05_NVFP4_V2_ACC32
                                                   : FmbLinearRouteSelector::PI05_NVFP4_V2_ACC16));
        }
    }
    return manifest;
}

FmbPhysicalManifestForwardCapability
Pi05DenoiseStepModel::physical_manifest_forward_capability(
    const FmbPhysicalExecutionManifest& manifest) const {
    validate_pi05_w4_fastpath_manifest(manifest);
    validate_pi05_q_rope_manifest(manifest);
    validate_pi05_xor3_gated_attn_manifest(manifest);
    validate_pi05_kv1_stripe_gather_manifest(manifest);
    validate_pi05_kv1_direct_cache_manifest(manifest);
    validate_pi05_w4_geglu_m50_manifest(manifest);
    validate_pi05_nvfp4_geglu_m50_manifest(manifest);
    validate_pi05_geglu_m50_manifest(manifest);
    validate_pi05_xor3_gated_mlp_manifest(manifest);
    return {true, FmbGraphLifecycle::COMPOSITE_CHILD};
}

bool Pi05DenoiseStepModel::use_spm_adarms_table() const {
    // The prepared schedule is independent of camera count and projection
    // precision. Allocate the actual four/ten-body table and bind its count
    // before planning; no ten-step table can be reused for four-step replay.
    return planned_route_profile_valid_ && planned_loop_mode_
        && pi05_action_steps_admitted(planned_num_steps_)
        && pi05_action_prefix_admitted(planned_prefix_len_)
        && chunk_size_ == 50 && hidden_size() == 1024 && num_layers() == 18
        && num_cores() == 8 && attn_tp() == 8 && adarms_table_.defined()
        && adarms_table_.size(0) == planned_num_steps_;
}

bool Pi05DenoiseStepModel::use_pi05_adarms_body_stream() const {
    // Keep use_spm_adarms_table true: norm/shift and gated-residual fusion
    // consume the same bytes whether the slot holds ten bodies or one body.
    return planned_num_steps_ == 10 && pi05_adarms_body_stream_opt_in_ &&
        planned_prefix_len_ == 800 && use_spm_adarms_table();
}

bool Pi05DenoiseStepModel::use_pi05_gateup_bundle(int64_t rows) const {
    return (pi05_gateup_resident_layers_ == 1 || pi05_gateup_resident_layers_ == 5)
        && !pi05_gateup_resident_opt_in_ && use_pi05_adarms_body_stream()
        && pi05_k_rope_insert_w8a16_ && rows == 50
        && intermediate_size() == 4096 && num_q_heads() == 8
        && num_kv_heads() == 8 && head_dim() == 256
        && resident_gateup_bundle_.defined();
}

void Pi05DenoiseStepModel::validate_pi05_storage_policy(at::IntArrayRef descriptor) const {
    const bool stream = use_pi05_adarms_body_stream();
    const bool bundle = use_pi05_gateup_bundle(chunk_size_);
    TORCH_CHECK(stream == pi05_adarms_body_stream_opt_in_ &&
                    bundle == (pi05_gateup_resident_layers_ != 0),
                "Pi0.5 storage route no longer matches its cold profile");
    if (descriptor.empty()) {
        TORCH_CHECK(!stream && !bundle, "Pi0.5 storage route requires a complete descriptor");
        return;
    }
    const auto candidate = decode_fmb_prefill_stage_candidate(descriptor);
    const auto& manifest = candidate.physical_manifest;
    if (stream || bundle) {
        TORCH_CHECK(manifest.state == FmbPhysicalManifestState::COMPLETE &&
                        manifest.graph_lifecycle == FmbGraphLifecycle::COMPOSITE_CHILD &&
                        manifest.linear_accumulation == linear_accumulation_policy() &&
                        manifest.execution_padding_rows == 0 && manifest.kv_logical_length == planned_prefix_len_ + 50 &&
                        pi05_graph_dma_shape_admitted(candidate.stage_plan,
                            manifest.physical_length, manifest.logical_length,
                            planned_prefix_len_, 1, true, false, AttentionExecutionPolicy::DDR_KV),
                    "Pi0.5 storage route requires exact P800/M50 COMPLETE loop geometry");
        pi05_require_stable_graph_dma_capture();
    }
    int64_t table_routes = 0, preload_routes = 0, linear_routes = 0;
    for (const auto& route : manifest.routes) {
        if (route.site_id == PI05_DENOISE_ADARMS_TABLE_DMA_SITE) {
            if (stream) {
                TORCH_CHECK(route.family == FmbRouteFamily::MUTABLE_DMA &&
                                route.selector == static_cast<int64_t>(Pi05DenoiseMutableDmaRoute::DDR_BROADCAST_TABLE_BODY) &&
                                route.flags == 0 && route.arguments == pi05_adarms_body_arguments(route.invocation),
                            "Pi0.5 body-table descriptor offset/selector drift");
                ++table_routes;
            } else {
                TORCH_CHECK(route.selector != static_cast<int64_t>(Pi05DenoiseMutableDmaRoute::DDR_BROADCAST_TABLE_BODY),
                            "Pi0.5 body-table descriptor supplied to a cold OFF owner");
            }
        } else if (route.site_id == PI05_DENOISE_BUNDLE_PRELOAD_SITE) {
            TORCH_CHECK(bundle && route.family == FmbRouteFamily::MUTABLE_DMA &&
                            route.selector == 5 && route.flags == 0 && route.invocation == 0 &&
                            route.arguments == pi05_gateup_bundle_arguments(pi05_gateup_resident_layers_),
                        "Pi0.5 bundle preload descriptor does not match its cold owner");
            ++preload_routes;
        } else if (route.site_id == PI05_DENOISE_BUNDLE_LINEAR_SITE) {
            TORCH_CHECK(bundle && route.family == FmbRouteFamily::LINEAR &&
                            route.selector == PI05_DENOISE_SPM_WEIGHT_M512N48K128 && route.flags == 0 &&
                            route.arguments == pi05_gateup_bundle_linear_arguments(
                                pi05_gateup_resident_layers_, route.invocation / 2, route.invocation % 2),
                        "Pi0.5 bundle linear descriptor does not match its layer/projection");
            ++linear_routes;
        }
    }
    TORCH_CHECK(table_routes == (stream ? 10 : 0) &&
                    preload_routes == (bundle ? 1 : 0) &&
                    linear_routes == (bundle ? pi05_gateup_resident_layers_ * 2 : 0),
                "Pi0.5 storage descriptor is incomplete");
}


bool Pi05DenoiseStepModel::pi05_w4_weight_profile_admitted() const {
    if (layer_weights_.size() != 18 || hidden_size() != 1024 ||
        intermediate_size() != 4096 || max_action_dim_ != 32 || chunk_size_ != 50 ||
        num_q_heads() != 8 || num_kv_heads() != 8 || head_dim() != 256 || attn_tp() != 8) return false;
    const auto& first_scale = layer_weights_.front().q_ws;
    if (!first_scale.defined() || first_scale.dim() != 2) return false;
    const int64_t gs = first_scale.size(0);
    if (gs != 32 && gs != 64 && gs != 128) return false;
    const auto tensor_ok = [](const at::Tensor& t, at::ScalarType dtype) {
        return t.defined() && t.device().type() == at::kPrivateUse1 &&
            t.scalar_type() == dtype && t.is_contiguous() && t.storage_offset() == 0;
    };
    const auto w4 = [&](const at::Tensor& w, const at::Tensor& s,
                        int64_t n, int64_t k, int partition) {
        const int64_t local_g = k / gs / (partition == 0 ? 8 : 1);
        const int64_t local_n = n / (partition == 1 ? 8 : 1);
        const int64_t scale_numel = ((local_g + 3) / 4) * ((local_n + 63) / 64) * 8 * 4 * 64;
        return tensor_ok(w, at::kByte) && tensor_ok(s, at::kHalf) &&
            w.sizes() == at::IntArrayRef({n, k / 2}) && w.nbytes() == n * k / 2 &&
            s.sizes() == at::IntArrayRef({gs, scale_numel / gs}) && s.nbytes() == scale_numel * 2;
    };
    const auto kv8 = [&](const at::Tensor& w, const at::Tensor& s) {
        return tensor_ok(w, at::kChar) && tensor_ok(s, at::kHalf) &&
            w.sizes() == at::IntArrayRef({2048, 1024}) && s.sizes() == at::IntArrayRef({2048});
    };
    return std::all_of(layer_weights_.begin(), layer_weights_.end(), [&](const LayerWeights& w) {
        return w4(w.q_w, w.q_ws, 2048, 1024, 1) && w4(w.o_w, w.o_ws, 1024, 2048, 0) &&
            w4(w.gate_proj_w, w.gate_ws, 4096, 1024, 1) && w4(w.up_proj_w, w.up_ws, 4096, 1024, 1) &&
            w4(w.down_proj_w, w.down_ws, 1024, 4096, 0) && kv8(w.k_w, w.k_ws) && kv8(w.v_w, w.v_ws);
    });
}

bool Pi05DenoiseStepModel::pi05_nvfp4_weight_profile_admitted() const {
    if (!nvfp4_.enabled() || nvfp4_.values.size() != 5 ||
        layer_weights_.size() != 18 || hidden_size() != 1024 ||
        intermediate_size() != 4096 || max_action_dim_ != 32 || chunk_size_ != 50 ||
        num_q_heads() != 8 || num_kv_heads() != 8 || head_dim() != 256 || attn_tp() != 8)
        return false;
    const auto tensor_ok = [](const at::Tensor& t, at::ScalarType dtype) {
        return t.defined() && t.device().type() == at::kPrivateUse1 &&
            t.scalar_type() == dtype && t.is_contiguous() && t.storage_offset() == 0;
    };
    const auto fp4 = [&](const at::Tensor& w, const at::Tensor& scale, int64_t n, int64_t k) {
        return tensor_ok(w, at::kByte) && tensor_ok(scale, at::kByte) &&
            w.sizes() == at::IntArrayRef({n, k / 2}) && w.nbytes() == n * k / 2 &&
            scale.sizes() == at::IntArrayRef({k / 16, n}) && scale.nbytes() == n * k / 16;
    };
    const auto kv8 = [&](const at::Tensor& w, const at::Tensor& scale) {
        return tensor_ok(w, at::kChar) && tensor_ok(scale, at::kHalf) &&
            w.sizes() == at::IntArrayRef({2048, 1024}) && scale.sizes() == at::IntArrayRef({2048});
    };
    return std::all_of(nvfp4_.values.begin(), nvfp4_.values.end(), [&](const at::Tensor& t) {
        return tensor_ok(t, at::kFloat) && t.sizes() == at::IntArrayRef({24});
    }) && std::all_of(layer_weights_.begin(), layer_weights_.end(), [&](const LayerWeights& w) {
        return fp4(w.q_w, w.q_ws, 2048, 1024) && fp4(w.o_w, w.o_ws, 1024, 2048) &&
            fp4(w.gate_proj_w, w.gate_ws, 4096, 1024) && fp4(w.up_proj_w, w.up_ws, 4096, 1024) &&
            fp4(w.down_proj_w, w.down_ws, 1024, 4096) && kv8(w.k_w, w.k_ws) && kv8(w.v_w, w.v_ws);
    });
}

bool Pi05DenoiseStepModel::pi05_projection_fastpath_admitted() const {
    return pi05_k_rope_insert_w8a16_ ||
        pi05_fp16_projection_owners_admitted(layer_weights_) ||
        (pi05_w4_fastpath_opt_in_ && pi05_w4_fastpath_profile_);
}

std::vector<int64_t> Pi05DenoiseStepModel::pi05_w4_fastpath_arguments() const {
    if (nvfp4_.enabled()) {
        // ABI2 binds striped-v2 E2M1/FP8 K16 scales, five FP32[24] tables,
        // the selected accumulation and the unchanged W8 K/V owner layout.
        return {2, pi05_w4_fastpath_opt_in_ ? 1 : 0, pi05_w4_fastpath_profile_ ? 1 : 0,
                planned_num_steps_, planned_prefix_len_, 50, 64, planned_prefix_len_ + 50, 2048, 1024, 4096, 18, 8, 8, 256, 32,
                4, 8, 8, 4, 4, 4, 4, 16, 8, 32, 2, 5, 24,
                static_cast<int64_t>(linear_accumulation_policy()),
                static_cast<int64_t>((linear_acc32_ ? FmbLinearRouteSelector::PI05_NVFP4_V2_ACC32
                                                   : FmbLinearRouteSelector::PI05_NVFP4_V2_ACC16))};
    }
    const int64_t gs = layer_weights_.empty() || !layer_weights_.front().q_ws.defined()
        ? 0 : layer_weights_.front().q_ws.size(0);
    // ABI, immutable opt-in, proven storage, exact profile and projection precisions.
    // The existing per-tensor cost identity also binds every layer's physical layout.
    return {1, pi05_w4_fastpath_opt_in_ ? 1 : 0, pi05_w4_fastpath_profile_ ? 1 : 0,
            planned_num_steps_, planned_prefix_len_, 50, 64, planned_prefix_len_ + 50, 2048, 1024, 4096, 18, 8, 8, 256, 32,
            4, 8, 8, 4, 4, 4, 4, gs, 8, 8, 1};
}

void Pi05DenoiseStepModel::validate_pi05_w4_fastpath_manifest(const FmbPhysicalExecutionManifest& m) const {
    if (pi05_w4_fastpath_opt_in_) {
        TORCH_CHECK(pi05_w4_fastpath_profile_ && use_pi05_graph_dma_profile(m.physical_length) &&
            use_pi05_kv1_pair_owner(m.physical_length) && use_pi05_xor3_gated_attn(m.physical_length) &&
            m.state == FmbPhysicalManifestState::COMPLETE && m.logical_length == 50 &&
            m.physical_length == 50 && m.execution_padding_rows == 0 &&
            m.kv_logical_length == planned_prefix_len_ + 50 && m.kv_insert_physical_rows == 64 &&
            m.graph_lifecycle == FmbGraphLifecycle::COMPOSITE_CHILD &&
            m.linear_accumulation == linear_accumulation_policy(),
            "Pi W4 fastpath requires an exact COMPLETE mixed-precision profile");
    }
    size_t count = 0;
    for (const auto& r : m.routes) {
        if (r.site_id != PI05_DENOISE_W4_FASTPATH_POLICY_SITE) continue;
        ++count;
        TORCH_CHECK(pi05_w4_fastpath_opt_in_ && r.family == FmbRouteFamily::GRAPH_SCHEDULE &&
            r.selector == 1 && r.flags == 0 && r.invocation == 0 && r.arguments == pi05_w4_fastpath_arguments(),
            "Pi W4 fastpath policy drifted or escaped its cold owner");
    }
    TORCH_CHECK(count == (pi05_w4_fastpath_opt_in_ ? 1 : 0), "Pi W4 fastpath policy missing/duplicated");
}

void Pi05DenoiseStepModel::validate_pi05_w4_fastpath_policy(const FmbPrefillStageCandidate* candidate) const {
    if (candidate == nullptr) {
        TORCH_CHECK(!pi05_w4_fastpath_opt_in_, "Pi W4 fastpath requires a complete signed descriptor");
        return;
    }
    validate_pi05_w4_fastpath_manifest(candidate->physical_manifest);
    if (pi05_w4_fastpath_opt_in_) {
        TORCH_CHECK(pi05_graph_dma_shape_admitted(candidate->stage_plan, 50, 50,
            planned_prefix_len_, 1, true, false, AttentionExecutionPolicy::DDR_KV),
            "Pi W4 fastpath chunk schedule drifted");
        pi05_require_stable_graph_dma_capture();
    }
}

bool Pi05DenoiseStepModel::use_pi05_graph_dma_profile(int64_t rows) const {
    return use_spm_adarms_table() &&
        (planned_num_steps_ == 10 ||
         (!pi05_elide_unused_dma_opt_in_ && !pi05_spm_action_staging_opt_in_ &&
          (!pi05_w4_fastpath_opt_in_ || nvfp4_.enabled()))) &&
        pi05_projection_fastpath_admitted() && rows == 50 &&
        intermediate_size() == 4096 && num_q_heads() == 8 &&
        num_kv_heads() == 8 && head_dim() == 256 && max_action_dim_ == 32 &&
        planned_k_rope_cache_capacity_ == 2048;
}

int64_t Pi05DenoiseStepModel::prepare_pi05_graph_dma_policy(
    at::IntArrayRef descriptor) const {
    if (descriptor.empty()) return 0; // original synchronous legacy path
    const auto candidate = decode_fmb_prefill_stage_candidate(descriptor);
    const auto& manifest = candidate.physical_manifest;
    const auto find = [&](FmbRouteFamily family, int64_t site) {
        const FmbRouteManifestEntry* found = nullptr;
        for (const auto& route : manifest.routes) {
            if (route.site_id == site) {
                TORCH_CHECK(route.family == family && found == nullptr &&
                                route.invocation == 0,
                            "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 DMA policy "
                            "has wrong family, duplicate site or wrong invocation");
                found = &route;
            }
        }
        return found;
    };
    const auto* unused = find(FmbRouteFamily::GRAPH_SCHEDULE, PI05_DENOISE_UNUSED_OUTPUT_SITE);
    const auto* output = find(FmbRouteFamily::MUTABLE_DMA, PI05_DENOISE_POST_LOOP_DMA_SITE);
    const bool exact = manifest.state == FmbPhysicalManifestState::COMPLETE &&
        manifest.graph_lifecycle == FmbGraphLifecycle::COMPOSITE_CHILD &&
        manifest.linear_accumulation == linear_accumulation_policy() &&
        manifest.execution_padding_rows == 0 && manifest.kv_logical_length == planned_prefix_len_ + 50 &&
        use_pi05_graph_dma_profile(manifest.physical_length) &&
        pi05_graph_dma_shape_admitted(candidate.stage_plan, manifest.physical_length,
            manifest.logical_length, planned_prefix_len_, 1, true, false,
            AttentionExecutionPolicy::DDR_KV);
    const bool elide_unused = exact && pi05_elide_unused_dma_opt_in_;
    const bool spm_staging = exact && pi05_spm_action_staging_opt_in_;
    const auto* producer = find(FmbRouteFamily::GRAPH_SCHEDULE, PI05_DENOISE_ACTION_SPM_PRODUCER_SITE);
    const auto* input = find(FmbRouteFamily::GRAPH_SCHEDULE, PI05_DENOISE_ACTION_SPM_INPUT_SITE);
    const auto* ddr_input = find(FmbRouteFamily::MUTABLE_DMA, FMB_SHARED_LAYER_INPUT_DMA_SITE);
    TORCH_CHECK((producer != nullptr) == spm_staging && (input != nullptr) == spm_staging,
                "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 action SPM producer/consumer "
                "does not match cold/profile admission");
    if (spm_staging) {
        TORCH_CHECK(producer->selector == 1 && producer->flags == 0 &&
                        producer->arguments == pi05_graph_dma_arguments(planned_prefix_len_) &&
                        input->selector == 1 && input->flags == 0 &&
                        input->arguments == pi05_graph_dma_arguments(planned_prefix_len_) && !ddr_input,
                    "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 SPM action routes "
                    "must be canonical and replace the shared DDR input route");
    }
    TORCH_CHECK((unused != nullptr) == elide_unused,
                "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 unused-output route "
                "does not match cold/profile admission");
    if (elide_unused) {
        TORCH_CHECK(unused->selector == 1 && unused->flags == 0 &&
                        unused->arguments == pi05_graph_dma_arguments(planned_prefix_len_) && output &&
                        output->selector == static_cast<int64_t>(
                            Pi05DenoiseMutableDmaRoute::SPM_COPY_FINAL_BODY_TO_DDR) &&
                        output->flags == 0 && output->arguments == pi05_graph_dma_arguments(planned_prefix_len_),
                    "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 paired output "
                    "routes are not canonical");
    } else {
        TORCH_CHECK(!output || output->selector != static_cast<int64_t>(
                        Pi05DenoiseMutableDmaRoute::SPM_COPY_FINAL_BODY_TO_DDR),
                    "RPU_PLANNER_REJECT:EXACT_MISMATCH: unpaired Pi0.5 final-only DMA");
    }
    const int64_t policy = (elide_unused ? PI05_DENOISE_ELIDE_UNUSED_DMA : 0) |
                          (spm_staging ? PI05_DENOISE_SPM_ACTION_STAGING : 0);
    if (policy != 0) pi05_require_stable_graph_dma_capture();
    return policy;
}

void Pi05DenoiseStepModel::validate_pi05_graph_dma_runtime() const {
    if (pi05_graph_dma_policy_ == 0) return;
    pi05_require_stable_graph_dma_capture();
    TORCH_CHECK(loop_mode_ && num_steps_ == 10 &&
                    ctx().has_complete_physical_manifest() &&
                    ctx().physical_manifest().graph_lifecycle == FmbGraphLifecycle::COMPOSITE_CHILD &&
                    use_pi05_graph_dma_profile(ctx().seq_len) &&
                    ctx().attention_mask.has_value() && ctx().attention_mask->dim() == 4 &&
                    ctx().attention_mask->size(0) == 1 && ctx().attention_mask->size(1) == 1 &&
                    ctx().attention_mask->size(2) == 50 && ctx().attention_mask->size(3) == planned_prefix_len_ + 50 &&
                    pi05_graph_dma_shape_admitted(ctx().stage_plan, ctx().seq_len,
                        ctx().physical_manifest().logical_length, ctx().position,
                        ctx().batch_size, true, ctx().is_causal, ctx().attention_policy),
                "RPU_PLANNER_REJECT:EXACT_MISMATCH: Pi0.5 DMA runtime changed "
                "the admitted loop/shape/lifecycle");
    TORCH_CHECK(ctx().k_caches && ctx().v_caches &&
                    ctx().k_caches->size() == 18 && ctx().v_caches->size() == 18,
                "Pi0.5 optimized DMA requires all 18 cache owners");
    for (const auto* caches : {ctx().k_caches, ctx().v_caches}) {
        for (const auto& cache : *caches) {
            TORCH_CHECK(cache.dim() == 7 && cache.size(0) == 1 &&
                            cache.size(1) == 128 && cache.size(2) == 1 &&
                            cache.size(3) == 16 && cache.size(4) == 8 &&
                            cache.size(5) == 16 && cache.size(6) == 16,
                        "Pi0.5 optimized DMA requires actual capacity2048 D256 cache owners");
        }
    }
}

bool Pi05DenoiseStepModel::use_pi05_gateup_resident(int64_t rows) const {
    return planned_num_steps_ == 10 && pi05_gateup_resident_opt_in_ &&
        planned_prefix_len_ == 800 && use_spm_adarms_table()
        && pi05_k_rope_insert_w8a16_ && rows == 50
        && intermediate_size() == 4096 && num_q_heads() == 8
        && num_kv_heads() == 8 && head_dim() == 256
        && resident_gate_w8_.defined() && resident_up_w8_.defined();
}

bool Pi05DenoiseStepModel::use_pi05_k_rope_insert(
    int64_t rows, int64_t position, int64_t logical_position,
    int64_t batch_size, const KvInsertSegmentPlan& plan) const {
    if (!pi05_k_rope_insert_opt_in_ || planned_num_steps_ != 10 ||
        !pi05_projection_fastpath_admitted() ||
        !use_spm_adarms_table() || rows != 50 || position != planned_prefix_len_ ||
        batch_size != 1 || planned_k_rope_cache_capacity_ < position + 64 ||
        num_q_heads() != 8 || num_kv_heads() != 8 || head_dim() != 256 ||
        plan.route() != KvInsertRoute::PAD16_V16 ||
        plan.logical_rows() != 50 || plan.physical_rows() != 64 ||
        plan.segment_count() != 1 || plan.segment(0).kernel != KvInsertKernel::V16 ||
        plan.segment(0).position != position || plan.segment(0).token_offset != 0 ||
        plan.segment(0).rows != 64) {
        return false;
    }
    const auto table_ok = [](const at::Tensor& table) {
        return table.defined() && table.device().type() == at::kPrivateUse1 &&
            table.scalar_type() == at::kHalf && table.is_contiguous() &&
            table.dim() == 2 && table.size(1) == 128;
    };
    return table_ok(cos_) && table_ok(sin_) && cos_.sizes() == sin_.sizes() &&
        logical_position >= 0 && logical_position <= UINT32_MAX &&
        logical_position <= cos_.size(0) && 50 <= cos_.size(0) - logical_position;
}

void Pi05DenoiseStepModel::rebind_kvinsert_exact_candidate(
    FmbPrefillStageCandidate& candidate, const LayoutContext& layout,
    int64_t site_id, int64_t invocation, const KvInsertSegmentPlan& plan) const {
    if (site_id != PI05_DENOISE_KV_INSERT_SITE) return;
    auto& manifest = candidate.physical_manifest;
    const auto find_site = [&](FmbRouteFamily family, int64_t site) {
        return std::find_if(manifest.routes.begin(), manifest.routes.end(),
            [&](const FmbRouteManifestEntry& route) {
                return route.family == family && route.site_id == site &&
                    route.invocation == invocation;
            });
    };
    const auto rope = find_site(FmbRouteFamily::ROPE, PI05_DENOISE_K_ROPE_SITE);
    const auto kv = find_site(FmbRouteFamily::KV_INSERT, PI05_DENOISE_KV_INSERT_SITE);
    const auto& chunks = candidate.stage_plan.compute.chunks;
    const auto chunk = std::find_if(chunks.begin(), chunks.end(),
        [&](const ChunkInfo& value) { return value.idx == invocation; });
    TORCH_CHECK(rope != manifest.routes.end() && kv != manifest.routes.end() &&
                    chunk != chunks.end() && !rope->arguments.empty(),
                "Pi0.5 exact KV mint lost its paired K-RoPE site or compute chunk");
    // Logical RoPE is frozen in the admitted source descriptor. It is not the
    // physical KV position and must not come from a later mutable rope_position_.
    const int64_t logical_position = rope->arguments.front();
    const bool source_fused =
        (rope->flags & PI05_DENOISE_K_ROPE_INSERT_FLAG) != 0;
    const bool source_gathered =
        (rope->flags & PI05_DENOISE_KV1_GATHERED_SOURCE_FLAG) != 0;
    const bool source_direct =
        (rope->flags & PI05_DENOISE_KV1_DIRECT_CACHE_FLAG) != 0;
    TORCH_CHECK((rope->flags & ~(PI05_DENOISE_K_ROPE_INSERT_FLAG |
                                  PI05_DENOISE_KV1_GATHERED_SOURCE_FLAG |
                                  PI05_DENOISE_KV1_DIRECT_CACHE_FLAG)) == 0 &&
                    kv->flags == rope->flags &&
                    source_gathered == use_pi05_kv1_stripe_gather(chunk->len) &&
                    source_direct == use_pi05_kv1_direct_cache(chunk->len) &&
                    !(source_direct && (source_fused || source_gathered)) &&
                    rope->selector == static_cast<int64_t>(head_dim() == 256
                        ? Pi05DenoiseRopeRoute::FULL_ROPE_TILED : Pi05DenoiseRopeRoute::ROPE_1D) &&
                    rope->arguments == (source_fused
                        ? pi05_k_rope_insert_arguments(logical_position,
                              planned_k_rope_cache_capacity_, planned_k_rope_cache_capacity_, planned_prefix_len_)
                        : std::vector<int64_t>{logical_position}),
                "Pi0.5 exact KV mint source K-RoPE metadata is not canonical");
    const bool k_fused = manifest.physical_length == 50 &&
        manifest.logical_length == 50 && chunks.size() == 1 && chunk->offset == 0 &&
        layout.use_attn_mask && !layout.is_causal &&
        layout.attention_policy == AttentionExecutionPolicy::DDR_KV &&
        !source_direct &&
        use_pi05_k_rope_insert(chunk->len, plan.segment(0).position,
                              logical_position, layout.batch_size, plan);
    const bool kv1_stripe = use_pi05_kv1_stripe_gather(chunk->len);
    const bool kv1_direct = use_pi05_kv1_direct_cache(chunk->len);
    rope->flags = kv1_direct
        ? PI05_DENOISE_KV1_DIRECT_CACHE_FLAG
        : (k_fused ? PI05_DENOISE_K_ROPE_INSERT_FLAG : 0) |
            (kv1_stripe ? PI05_DENOISE_KV1_GATHERED_SOURCE_FLAG : 0);
    rope->arguments = k_fused
        ? pi05_k_rope_insert_arguments(logical_position,
              planned_k_rope_cache_capacity_, planned_k_rope_cache_capacity_, planned_prefix_len_)
        : std::vector<int64_t>{logical_position};
    kv->flags = rope->flags;
    // KV selector and all22 codec words are already authoritative from mint.
    // Leave them intact; the unmodified exact forward consumer checks both sites.
}

bool Pi05DenoiseStepModel::use_pi05_ring_xor3(int64_t rows) const {
    return pi05_ring_xor3_opt_in_ && rows == 50 && use_spm_adarms_table() &&
        pi05_projection_fastpath_admitted();
}

bool Pi05DenoiseStepModel::use_pi05_xor3_gated_attn(int64_t rows) const {
    return pi05_xor3_gated_attn_opt_in_ && use_pi05_ring_xor3(rows) &&
        use_pi05_graph_dma_profile(rows) && pi05_k_rope_insert_opt_in_ &&
        !pi05_gateup_resident_opt_in_ && !pi05_adarms_body_stream_opt_in_ &&
        pi05_gateup_resident_layers_ == 0 &&
        !pi05_elide_unused_dma_opt_in_ && !pi05_spm_action_staging_opt_in_ &&
        !pi05_q_rope_epilogue_opt_in_;
}

int64_t Pi05DenoiseStepModel::pi05_denoise_residual_route(int64_t rows) const {
    if (use_pi05_xor3_gated_attn(rows)) {
        return static_cast<int64_t>(
            Pi05DenoiseResidualRoute::ZERO_WITH_XOR3_GATED_ATTN_H1024);
    }
    return static_cast<int64_t>(use_spm_adarms_table()
        ? Pi05DenoiseResidualRoute::ZERO_WITH_FUSED_GATE_H1024
        : Pi05DenoiseResidualRoute::ZERO_WITH_SEPARATE_GATE);
}

void Pi05DenoiseStepModel::validate_pi05_xor3_gated_attn_manifest(
    const FmbPhysicalExecutionManifest& manifest) const {
    if (pi05_xor3_gated_attn_opt_in_) {
        TORCH_CHECK(use_pi05_xor3_gated_attn(manifest.physical_length) &&
                        manifest.state == FmbPhysicalManifestState::COMPLETE &&
                        manifest.physical_length == 50 &&
                        manifest.logical_length == 50 &&
                        manifest.execution_padding_rows == 0 &&
                        manifest.kv_logical_length == planned_prefix_len_ + 50 &&
                        manifest.graph_lifecycle == FmbGraphLifecycle::COMPOSITE_CHILD &&
                        manifest.linear_accumulation == linear_accumulation_policy(),
                    "Pi attention XOR3+gated continuation requires exact "
                    "W8A16 P800/M50 COMPLETE loop geometry");
    }
    const int64_t expected_residual = pi05_denoise_residual_route(
        manifest.physical_length);
    const int64_t expected_reduce = fmb_ring_all_reduce_route_selector(
        manifest.physical_length, hidden_size(),
        use_pi05_ring_xor3(manifest.physical_length), num_cores());
    size_t residual_routes = 0;
    size_t attention_reduce_routes = 0;
    for (const auto& route : manifest.routes) {
        if (route.site_id == PI05_DENOISE_ZERO_GATED_RESIDUAL_SITE) {
            ++residual_routes;
            TORCH_CHECK(route.family == FmbRouteFamily::GRAPH_SCHEDULE &&
                            route.selector == expected_residual &&
                            route.flags == 0 && route.invocation == 0 &&
                            route.arguments == std::vector<int64_t>{hidden_size()},
                        "Pi attention residual schedule route is not canonical");
        } else if (route.site_id == PI05_DENOISE_ATTENTION_ALL_REDUCE_SITE) {
            ++attention_reduce_routes;
            TORCH_CHECK(route.family == FmbRouteFamily::ALL_REDUCE &&
                            route.selector == expected_reduce &&
                            route.flags == 0 && route.invocation == 0 &&
                            route.arguments.empty(),
                        "Pi attention all-reduce route is not canonical");
        }
    }
    TORCH_CHECK(residual_routes == 1 && attention_reduce_routes == 1,
                "Pi attention residual/all-reduce routes are missing or duplicated");
}

void Pi05DenoiseStepModel::validate_pi05_xor3_gated_attn_policy(
    const FmbPrefillStageCandidate* candidate) const {
    if (candidate == nullptr) {
        TORCH_CHECK(!pi05_xor3_gated_attn_opt_in_,
                    "Pi attention XOR3+gated continuation requires a complete "
                    "signed loop descriptor");
        return;
    }
    validate_pi05_xor3_gated_attn_manifest(candidate->physical_manifest);
    if (pi05_xor3_gated_attn_opt_in_) {
        TORCH_CHECK(pi05_graph_dma_shape_admitted(
                        candidate->stage_plan,
                        candidate->physical_manifest.physical_length,
                        candidate->physical_manifest.logical_length,
                        planned_prefix_len_, 1, true, false,
                        AttentionExecutionPolicy::DDR_KV),
                    "Pi attention XOR3+gated descriptor chunk schedule mismatch");
        pi05_require_stable_graph_dma_capture();
    }
}

void Pi05DenoiseStepModel::emit_gated_residual(
    uint32_t input_spm, uint32_t residual_spm, uint32_t gate_spm, int64_t rows) {
    // The pre-layer hook consumes this whole-body schedule's typed route.
    if (use_spm_adarms_table()) {
        rpu_launch_pi05_gated_residual_spm_kernel(
            input_spm, residual_spm, gate_spm, input_spm, rows, num_cores());
    } else {
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            gate_spm, input_spm, input_spm, rows, hidden_size(),
            c10::Half(1.0), ValuOpType::MUL, /*is_bopa=*/false, num_cores());
        rpu_launch_eltwise_binary_spm_kernel(
            input_spm, residual_spm, input_spm, rows * hidden_size(),
            ValuOpType::ADD, c10::Half(1.0), num_cores());
    }
}

void Pi05DenoiseStepModel::emit_adarms_norm_shift(
    uint32_t input_spm, uint32_t output_spm, uint32_t scale_spm,
    uint32_t shift_spm, int64_t rows, int64_t site_id, int64_t invocation) {
    // Share the verified resident-table profile's cold predicate. The kernel
    // keeps the old FP16 intermediate rounding and uses no additional SPM.
    const bool fused = use_spm_adarms_table();
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::NORMALIZATION, site_id,
            static_cast<int64_t>(fused ? Pi05DenoiseNormRoute::FUSED_H1024
                                     : Pi05DenoiseNormRoute::RMS_THEN_SHIFT),
            /*resolved_flags=*/0, {rows, hidden_size()}, invocation);
    }
    if (fused) {
        rpu_launch_pi05_adarms_norm_shift_spm_kernel(
            input_spm, output_spm, scale_spm, shift_spm, rows, eps_, num_cores());
    } else {
        rpu_launch_rmsnorm_spm_kernel(
            input_spm, output_spm, scale_spm, rows, hidden_size(), eps_, RpuRmsNormSpmRoute::BASE, num_cores());
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            shift_spm, output_spm, output_spm, rows, hidden_size(),
            c10::Half(1.0), ValuOpType::ADD, /*is_bopa=*/false, num_cores());
    }
}

std::vector<BufferDecl> Pi05DenoiseStepModel::declare_buffers(const LayoutContext& ctx) {
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
    int64_t q     = A(cs * local_q * hd * DWIDTH);
    // KV PAD16 may insert ceil16(cs) rows while compute still executes cs.
    // Reserve the same padded K/V extent in both planner probes and live layout.
    int64_t kv    = A(CeilDiv(cs, (int64_t)16) * 16 * local_kv * DWIDTH);
    int64_t out   = A(cs * local_q * hd * DWIDTH);
    int64_t oproj = A(cs * h * DWIDTH);
    int64_t mlp   = A(cs * (is_ / mlp_tp()) * DWIDTH);

    SdpaConfig sdpa_cfg{SdpaKernelType::FLASH_ATTN_SPM, hd, nq, nkv, attn_tp(), ctx.use_attn_mask ? 4 : 1};
    int64_t tmp = A(sdpa_compute_tmp_v16_size(sdpa_cfg, cs) * 32);
    int64_t mask_sz = ctx.use_attn_mask
        ? A(cs * CeilDiv(ctx.max_kv_seq_len, (int64_t)16) * 32)
        : 0;

    int64_t cond_sz     = A(h * DWIDTH);
    int64_t gemv_out_sz = A(3 * h * DWIDTH);

    // Pi05-specific sizes (spec §3.2)
    int64_t x_t_spm_sz       = A(cs * max_action_dim_ * DWIDTH);     // 3.2 KB
    int64_t action_emb_sz    = A(cs * h * DWIDTH);                   // 100 KB
    int64_t v_t_core0_sz     = A(cs * max_action_dim_ * DWIDTH);     // 3.2 KB

    // Persistent bias buffers (spec §3.1)
    int64_t in_bias_sz   = A(h * DWIDTH);                            // 2 KB
    int64_t out_bias_sz  = A(max_action_dim_ * DWIDTH);              // 64 B

    constexpr BufferScope ALL = BufferScope::LayerWide;
    using SC = StorageClass;

    // D-502 preload callbacks for persistent bias buffers.
    auto in_bias_preload = [this](FusedModelBase& /*self*/, int /*L*/,
                                  uint32_t core0_addr) {
        rpu_launch_ddr_broadcast_spm_dma(
            action_in_proj_b_.data_ptr<c10::Half>(),
            hidden_size(), core0_addr, num_cores());
    };
    auto out_bias_preload = [this](FusedModelBase& /*self*/, int /*L*/,
                                   uint32_t core0_addr) {
        rpu_launch_ddr_broadcast_spm_dma(
            action_out_proj_b_.data_ptr<c10::Half>(),
            max_action_dim_, core0_addr, num_cores());
    };

    std::vector<BufferDecl> decls{
        // Structural (mirror AdaRMSModel — re-uses build_layer_subgraph verbatim)
        {"residual1",    res,     0, 0, SC::Temp, 0, nullptr, ALL},
        {"input_norm",   res,     0, 0, SC::Temp, 0, nullptr, ALL},
        {"residual2",    res,     0, 0, SC::Temp, 0, nullptr, ALL},
        {"q",            q,       0, 0, SC::Temp, 0, nullptr, ALL},
        {"k",            kv,      0, 0, SC::Temp, 0, nullptr, ALL},
        {"v",            kv,      0, 0, SC::Temp, 0, nullptr, ALL},
        {"output",       out,     0, 0, SC::Temp, 0, nullptr, ALL},
        {"oproj",        oproj,   0, 0, SC::Temp, 0, nullptr, ALL},
        {"sdpa_tmp",     tmp,     0, 0, SC::Temp, 0, nullptr, ALL},
        // All Pi05 temps (including BOTH body hooks) are LayerWide [0,0].
        // The allocator therefore gives mask a disjoint slot for the full
        // forward, including transitions between denoise bodies. Keep this
        // all-body lifetime if phases are refined later: first-body upload
        // relies on pre/post hooks never aliasing this slot. No extra SPM.
        {"sdpa_mask",    mask_sz, 0, 0, SC::Temp, 0, nullptr, ALL},
        {"gate",         mlp,     0, 0, SC::Temp, 0, nullptr, ALL},
        {"up",           mlp,     0, 0, SC::Temp, 0, nullptr, ALL},
        {"down",         res,     0, 0, SC::Temp, 0, nullptr, ALL},
        // AdaRMS extras
        {"cond",         cond_sz,     0, 0, SC::Temp, 0, nullptr, ALL},
        {"attn_gemv",    gemv_out_sz, 0, 0, SC::Temp, 0, nullptr, ALL},
        {"mlp_gemv",     gemv_out_sz, 0, 0, SC::Temp, 0, nullptr, ALL},
        {"bias_temp",    gemv_out_sz, 0, 0, SC::Temp, 0, nullptr, ALL},
        {"gemv_partial", gemv_out_sz, 0, 0, SC::Temp, 0, nullptr, ALL},
        // Pi05-specific temp (spec §3.2; action_emb_spm DROPPED — see plan §C5)
        {"x_t_spm",              x_t_spm_sz,    0, 0, SC::Temp, 0, nullptr, ALL},
        {"action_emb_core0_spm", action_emb_sz, 0, 0, SC::Temp, 0, nullptr, ALL},
        {"final_gemv",           gemv_out_sz,   0, 0, SC::Temp, 0, nullptr, ALL},
        {"final_out_spm",        res,           0, 0, SC::Temp, 0, nullptr, ALL},
        {"v_t_core0_spm",        v_t_core0_sz,  0, 0, SC::Temp, 0, nullptr, ALL},
        // Pi05-specific persistent (spec §3.1)
        {"action_in_proj_b_spm",  in_bias_sz,  0, 0, SC::Persistent, 0, nullptr, ALL,
            in_bias_preload},
        {"action_out_proj_b_spm", out_bias_sz, 0, 0, SC::Persistent, 0, nullptr, ALL,
            out_bias_preload},
    };
    if (use_spm_adarms_table()) {
        // No alias: each body consumes all37 rows before the next body DMA.
        const int64_t table_bytes = use_pi05_adarms_body_stream()
            ? PI05_DENOISE_ADARMS_BODY_BYTES : adarms_table_.numel() * DWIDTH;
        decls.push_back({"adarms_table", A(table_bytes),
                        0, 0, SC::Temp, 0, nullptr, ALL});
    }
    if (use_pi05_gateup_resident(chunk_size_)) {
        // LayerWide, non-aliased, all-phase: keep raw bytes through all ten
        // bodies. Both ceil16(50)=64 planner probes and exact 50-row layouts
        // must budget these slots; admission is tied to the action profile.
        decls.push_back({"resident_gate_w8", PI05_DENOISE_RESIDENT_W8_BYTES,
                        0, 0, SC::Temp, 0, nullptr, ALL});
        decls.push_back({"resident_up_w8", PI05_DENOISE_RESIDENT_W8_BYTES,
                        0, 0, SC::Temp, 0, nullptr, ALL});
    }
    if (use_pi05_gateup_bundle(chunk_size_)) {
        // Both exact M50 layout and ceil16 M64 probes carry this complete
        // non-aliased slot, shared by every body but never by another owner.
        decls.push_back({"resident_gateup_bundle",
                        pi05_gateup_resident_layers_ * 2 * PI05_DENOISE_RESIDENT_W8_BYTES,
                        0, 0, SC::Temp, 0, nullptr, ALL});
    }
    nvfp4_.declare(decls, num_cores());
    return decls;
}

void Pi05DenoiseStepModel::build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) {
    if (num_cores() != 8 && layer_idx == 0 && chunk.idx == 0 &&
        ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            PI05_DENOISE_CORE_PROFILE_SITE, 1, 0, core_profile_arguments());
    }
    const auto& lw = layer_weights_[layer_idx];
    int64_t seq_len = chunk.len;
    int64_t cos_sin_start = ctx().position + chunk.offset;
    int64_t h = hidden_size();
    int64_t nq = num_q_heads();
    int64_t nkv = num_kv_heads();
    int64_t hd = head_dim();
    int tp = attn_tp();
    bool is_chunk_zero = (chunk.idx == 0);

    // Phase 0 cond DMA removed — owned by emit_pre_layers_body ([A2]) to
    // avoid double emission (plan §C2).

    // --------------------------------------------------------------------
    // Phase A: per-layer GEMV at chunk.idx == 0
    // Computes [scale | shift | gate] from cond * dense_w + bias, with
    // +1.0 baked into the scale slice. Result persists in SPM for this
    // layer's remaining chunks.
    // --------------------------------------------------------------------
    uint32_t attn_params = addr(0, "attn_gemv");
    uint32_t mlp_params = addr(0, "mlp_gemv");
    if (is_chunk_zero) {
        attn_params = adarms_gemv_to_spm(addr(0, "cond"),
                           lw.attn_dense_w, lw.attn_dense_b,
                           addr(0, "attn_gemv"),
                           addr(0, "bias_temp"),
                           addr(0, "gemv_partial"),
                           lw.attn_dense_ws, 2 * layer_idx);
        mlp_params = adarms_gemv_to_spm(addr(0, "cond"),
                           lw.mlp_dense_w,  lw.mlp_dense_b,
                           addr(0, "mlp_gemv"),
                           addr(0, "bias_temp"),
                           addr(0, "gemv_partial"),
                           lw.mlp_dense_ws, 2 * layer_idx + 1);
    }

    // GEMV slice offsets inside attn_gemv / mlp_gemv (computed from core 0
    // base address; broadcast DMA means every core shares the same offsets).
    uint32_t attn_scale = attn_params;
    uint32_t attn_shift = attn_scale + (uint32_t)(h * DWIDTH);
    uint32_t attn_gate  = attn_scale + (uint32_t)(h * DWIDTH * 2);
    uint32_t mlp_scale  = mlp_params;
    uint32_t mlp_shift  = mlp_scale + (uint32_t)(h * DWIDTH);
    uint32_t mlp_gate   = mlp_scale + (uint32_t)(h * DWIDTH * 2);

    // --------------------------------------------------------------------
    // Phase 1: layer input DMA -> "residual1"
    // Skip if previous layer left residual1 in SPM (SPM_RESIDENT mode).
    // --------------------------------------------------------------------
    if (!ctx().input_in_spm) {
        if (pi05_graph_dma_policy_ & PI05_DENOISE_SPM_ACTION_STAGING) {
            TORCH_CHECK(layer_idx == 0,
                        "Pi0.5 SPM action staging reached a non-initial group boundary");
            emit_pi05_action_spm_input(chunk);
        } else {
            emit_layer_input_dma(layer_idx, chunk);  // writes to "residual1"
        }
    }

    // --------------------------------------------------------------------
    // Phase 2: AdaRMSNorm (attn side): residual1 -> input_norm
    // (1+scale) was applied during GEMV, so RMSNorm uses attn_scale directly.
    // The exact ten-step table profile fuses the shift after FP16 rounding.
    // --------------------------------------------------------------------
    emit_adarms_norm_shift(
        addr(0, "residual1"), addr(0, "input_norm"),
        attn_scale, attn_shift, seq_len, PI05_DENOISE_ATTN_NORM_SITE, chunk.idx);

    // --------------------------------------------------------------------
    // Phase 3: QKV Linear (attn_tp cores, col partition)
    // --------------------------------------------------------------------
    if (pi05_q_rope_epilogue_opt_in_) {
        emit_pi05_q_rope_epilogue(lw,chunk);
    } else {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, PI05_DENOISE_Q_LINEAR_SITE,
                static_cast<int64_t>(nvfp4_.enabled() ? (linear_acc32_ ? FmbLinearRouteSelector::PI05_NVFP4_V2_ACC32
                                                   : FmbLinearRouteSelector::PI05_NVFP4_V2_ACC16) : FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0, {}, chunk.idx);
        }
        if (nvfp4_.enabled()) {
            rpu_launch_pi05_nvfp4_v2_kernel(addr(0, "input_norm"), lw.q_w, addr(0, "q"),
                seq_len, nq * hd, h, 1, tp, lw.q_ws,
                addr(0, Pi05Nvfp4Tables::names[0]), layer_idx, linear_acc32_);
        } else {
            rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.q_w, addr(0, "q"),
            seq_len, nq * hd, h,
            /*partition=*/1, tp, /*bias_spm_addr=*/0, lw.q_ws, 0, 0,
            /*force_acc32=*/linear_acc32_);
        }
    }
    const bool kv1_stripe = use_pi05_kv1_stripe_gather(seq_len);
    const bool kv1_direct = use_pi05_kv1_direct_cache(seq_len);
    const bool kv1_pair_owner = use_pi05_kv1_pair_owner(seq_len);
    if (kv1_pair_owner) {
        TORCH_CHECK(
            ctx().has_complete_physical_manifest() && loop_mode_ &&
                pi05_action_steps_admitted(num_steps_) && num_steps_ == planned_num_steps_ &&
                chunk.idx == 0 && chunk.offset == 0 &&
                ctx().position == planned_prefix_len_ && ctx().body_iter >= 0 &&
                ctx().body_iter < num_steps_,
            "Pi KV1 pair-owner reached a non-exact graph body");
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, PI05_DENOISE_KV1_PAIR_OWNER_SITE,
            pi05_kv1_pair_owner_selector(pi05_is_fp16_action(layer_weights_)),
            /*resolved_flags=*/0,
            pi05_kv1_pair_owner_linear_arguments(planned_prefix_len_, pi05_is_fp16_action(layer_weights_), planned_num_steps_), chunk.idx);
        rpu_launch_pi05_kv1_pair_owner_w8a16_m50n32x2k1024_kernel(
            addr(0, "input_norm"),
            lw.k_w, lw.k_ws, addr(0, "k"),
            lw.v_w, lw.v_ws, addr(0, "v"));
    } else if (kv1_direct) {
        TORCH_CHECK(
            ctx().has_complete_physical_manifest() && loop_mode_ &&
                pi05_action_steps_admitted(num_steps_) && num_steps_ == planned_num_steps_ &&
                chunk.idx == 0 && chunk.offset == 0 &&
                ctx().position == planned_prefix_len_ && ctx().body_iter >= 0 &&
                ctx().body_iter < num_steps_,
            "Pi KV1 direct-cache reached a non-exact graph body");
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, PI05_DENOISE_K_LINEAR_SITE,
            PI05_DENOISE_KV1_DIRECT_M592N32, /*resolved_flags=*/0,
            pi05_kv1_direct_linear_arguments(planned_prefix_len_), chunk.idx);
        rpu_launch_pi05_kv1_stripe_w8a16_m592n32_kernel(
            addr(0, "input_norm"), lw.k_kv1_w, addr(0, "k"), lw.k_kv1_ws);
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, PI05_DENOISE_V_LINEAR_SITE,
            PI05_DENOISE_KV1_DIRECT_M592N32, /*resolved_flags=*/0,
            pi05_kv1_direct_linear_arguments(planned_prefix_len_), chunk.idx);
        rpu_launch_pi05_kv1_stripe_w8a16_m592n32_kernel(
            addr(0, "input_norm"), lw.v_kv1_w, addr(0, "v"), lw.v_kv1_ws);
    } else if (kv1_stripe) {
        TORCH_CHECK(ctx().has_complete_physical_manifest() && loop_mode_ &&
                        num_steps_ == 10 && chunk.idx == 0 && chunk.offset == 0,
                    "Pi KV1 stripe-gather reached a non-exact graph body");
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, PI05_DENOISE_K_LINEAR_SITE,
            PI05_DENOISE_KV1_FIXED_M592N32, /*resolved_flags=*/0,
            pi05_kv1_linear_arguments(planned_prefix_len_), chunk.idx);
        rpu_launch_pi05_kv1_stripe_w8a16_m592n32_kernel(
            addr(0, "input_norm"), lw.k_kv1_w, addr(0, "k"), lw.k_kv1_ws);
        ctx().consume_physical_route(
            FmbRouteFamily::COLLECTIVE, PI05_DENOISE_KV1_K_GATHER_SITE,
            static_cast<int64_t>(RpuAllGatherSchedule::LITTLE_CHUNK),
            /*resolved_flags=*/0, pi05_kv1_gather_arguments(planned_prefix_len_), chunk.idx);
        RpuKernelGraph::active().stage_kernel_no_ddr(
            KernelId::ALL_GATHER_MULTI_CORE_LITTLE_CHUNK,
            GraphKernelNoDdrProof::AllGatherSpm);
        rpu_launch_all_gather_spm_kernel(
            addr(0, "k"), addr(0, "v"), seq_len, /*chunk_elems=*/32,
            sizeof(c10::Half), tp, RpuAllGatherSchedule::LITTLE_CHUNK);
    } else {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, PI05_DENOISE_K_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0, {}, chunk.idx);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.k_w, addr(0, "k"),
            seq_len, nkv * hd, h,
            /*partition=*/1, tp, /*bias_spm_addr=*/0, lw.k_ws, 0, 0,
            /*force_acc32=*/linear_acc32_);
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, PI05_DENOISE_V_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0, {}, chunk.idx);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.v_w, addr(0, "v"),
            seq_len, nkv * hd, h,
            /*partition=*/1, tp, /*bias_spm_addr=*/0, lw.v_ws, 0, 0,
            /*force_acc32=*/linear_acc32_);
    }

    // --------------------------------------------------------------------
    // Phase 4: RoPE (in-place, no QK RMSNorm for AdaRMS Gemma)
    // --------------------------------------------------------------------
    int64_t local_kv_heads = nkv / tp;
    c10::Half* cos_ptr = cos_.data_ptr<c10::Half>();
    c10::Half* sin_ptr = sin_.data_ptr<c10::Half>();
    // RoPE reads the LOGICAL position, the KV insert below the physical row
    // (see rope_position_). Equal only when the prefix has no pad rows.
    const int64_t rope_start = (rope_position_ >= 0)
        ? rope_position_ + chunk.offset : cos_sin_start;
    if (!pi05_q_rope_epilogue_opt_in_) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ROPE, PI05_DENOISE_Q_ROPE_SITE,
                static_cast<int64_t>(head_dim() == 256
                    ? Pi05DenoiseRopeRoute::FULL_ROPE_TILED : Pi05DenoiseRopeRoute::ROPE_1D),
                /*resolved_flags=*/0, {rope_start}, chunk.idx);
        }
        if (hd == 256) {
            rpu_launch_partial_mrope_spm_kernel(
                addr(0, "q"), addr(0, "q"), cos_, sin_,
                rope_start, seq_len, local_q_heads_, hd, hd, tp);
        } else {
            rpu_launch_rope_spm_kernel(
                addr(0, "q"), addr(0, "q"),
                cos_ptr, sin_ptr,
                seq_len, local_q_heads_, hd, rope_start, tp);
        }
    }
    auto& k_cache = (*ctx().k_caches)[layer_idx];
    auto& v_cache = (*ctx().v_caches)[layer_idx];
    KvInsertSegmentPlan kv_plan = [&] {
        if (!ctx().has_complete_physical_manifest()) {
            return rpu_resolve_kvinsert_segment_plan_auto(
                cos_sin_start, seq_len, seq_len, tp, nkv, hd,
                KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 |
                    KV_INSERT_CAP_HYBRID2 | KV_INSERT_CAP_HYBRID3);
        }
        const auto& route = ctx().find_physical_route(
            FmbRouteFamily::KV_INSERT, PI05_DENOISE_KV_INSERT_SITE,
            chunk.idx);
        KvInsertSegmentPlan frozen =
            restore_kvinsert_plan(
                PI05_DENOISE_KV_INSERT_SITE, route.arguments, tp, nkv, hd);
        TORCH_CHECK(
            frozen.logical_rows() == seq_len &&
                (frozen.physical_rows() == seq_len ||
                 frozen.physical_rows() == ((seq_len + 15) / 16) * 16) &&
                frozen.segment(0).position == cos_sin_start,
            "Pi0.5 denoise KV descriptor geometry drift");
        return frozen;
    }();
    const auto cache_abi_ok = [](const at::Tensor& cache) {
        return cache.defined() && cache.device().type() == at::kPrivateUse1 &&
            cache.scalar_type() == at::kHalf && cache.is_contiguous() &&
            cache.dim() == 7 && cache.size(0) == 1 && cache.size(1) >= 54 &&
            cache.size(2) == 1 && cache.size(3) == 16 && cache.size(4) == 8 &&
            cache.size(5) == 16 && cache.size(6) == 16;
    };
    const bool k_fused = ctx().has_complete_physical_manifest() &&
        cache_abi_ok(k_cache) && cache_abi_ok(v_cache) &&
        k_cache.sizes() == v_cache.sizes() &&
        chunk.offset == 0 && ctx().seq_len == 50 &&
        ctx().position == planned_prefix_len_ && ctx().seq_len == 50 && !ctx().is_causal &&
        prepared_mask_.mask_type == 4 &&
        ctx().attention_policy == AttentionExecutionPolicy::DDR_KV &&
        !kv1_direct &&
        use_pi05_k_rope_insert(seq_len, cos_sin_start, rope_start,
                              ctx().batch_size, kv_plan);
    if (ctx().has_complete_physical_manifest()) {
        const int64_t k_capacity = k_cache.dim() == 7 ? k_cache.size(1)*16 : 0;
        const int64_t v_capacity = v_cache.dim() == 7 ? v_cache.size(1)*16 : 0;
        const int64_t kv_route_flags = kv1_direct
            ? PI05_DENOISE_KV1_DIRECT_CACHE_FLAG
            : (k_fused ? PI05_DENOISE_K_ROPE_INSERT_FLAG : 0) |
                (kv1_stripe ? PI05_DENOISE_KV1_GATHERED_SOURCE_FLAG : 0);
        // Both sites are consumed before either implementation is dispatched.
        // The KV codec remains exactly the original22 words, without extensions.
        ctx().consume_physical_route(
            FmbRouteFamily::ROPE, PI05_DENOISE_K_ROPE_SITE,
            static_cast<int64_t>(head_dim() == 256
                ? Pi05DenoiseRopeRoute::FULL_ROPE_TILED : Pi05DenoiseRopeRoute::ROPE_1D),
            kv_route_flags,
            k_fused ? pi05_k_rope_insert_arguments(rope_start, k_capacity, v_capacity, planned_prefix_len_)
                    : std::vector<int64_t>{rope_start}, chunk.idx);
        const auto kv_arguments = rpu_kvinsert_route_arguments(kv_plan, tp, nkv, hd);
        ctx().consume_physical_route(
            FmbRouteFamily::KV_INSERT, PI05_DENOISE_KV_INSERT_SITE,
            static_cast<int64_t>(kv_plan.route()),
            kv_route_flags,
            kv_arguments, chunk.idx);
    }
    if (!k_fused && !kv1_direct) {
        const uint32_t k_spm = addr(0, kv1_stripe ? "v" : "k");
        if (hd == 256) {
            rpu_launch_partial_mrope_spm_kernel(
                k_spm, k_spm, cos_, sin_,
                rope_start, seq_len, local_kv_heads, hd, hd, tp);
        } else {
            rpu_launch_rope_spm_kernel(
                k_spm, k_spm,
                cos_ptr, sin_ptr,
                seq_len, local_kv_heads, hd, rope_start, tp);
        }
    }

    // --------------------------------------------------------------------
    // Phase 5: KV cache insert at absolute position.
    //
    // Pitfall 3 structural fix: KV-insert takes SPM OFFSETS, not absolute
    // addresses. addr_offset("name").value is compile-time distinct from
    // addr(core, "name").
    // --------------------------------------------------------------------
    if (!kv1_direct && kv_plan.physical_rows() > seq_len) {
        const int64_t local_kv = nkv * hd / tp;
        const int64_t pad_elems =
            (kv_plan.physical_rows() - seq_len) * local_kv;
        const uint32_t offset = static_cast<uint32_t>(
            seq_len * local_kv * DWIDTH);
        if (!kv1_stripe) {
            rpu_launch_memset_spm_multicore(
                addr(0, "k") + offset, pad_elems, tp);
        }
        rpu_launch_memset_spm_multicore(
            addr(0, "v") + offset, pad_elems, tp);
    }
    if (kv1_direct) {
        // The combined consumer performs K rotate-half RoPE, K/V physical-head
        // replication, V transpose and canonical +0 padding internally.
    } else if (k_fused) {
        // The existing K-pad is now before the fused RoPE, so all64 rows are
        // initialized when the two RoPE warps hand off to the original V16 copy.
        rpu_launch_pi05_k_rope_insert_spm_kernel(
            addr_offset(kv1_stripe ? "v" : "k").value,
            cos_, sin_, k_cache, rope_start,
            kv_plan.physical_rows(), kv_plan);
    } else {
        rpu_launch_insert_kcache_spm_unified_with_plan(
            k_cache, addr_offset(kv1_stripe ? "v" : "k").value, nkv, hd, tp,
            /*cache_batch_offset_elems=*/0, kv_plan.physical_rows(), kv_plan);
    }
    if (kv1_stripe) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, PI05_DENOISE_V_LINEAR_SITE,
            PI05_DENOISE_KV1_FIXED_M592N32, /*resolved_flags=*/0,
            pi05_kv1_linear_arguments(planned_prefix_len_), chunk.idx);
        rpu_launch_pi05_kv1_stripe_w8a16_m592n32_kernel(
            addr(0, "input_norm"), lw.v_kv1_w, addr(0, "k"), lw.v_kv1_ws);
        ctx().consume_physical_route(
            FmbRouteFamily::COLLECTIVE, PI05_DENOISE_KV1_V_GATHER_SITE,
            static_cast<int64_t>(RpuAllGatherSchedule::LITTLE_CHUNK),
            /*resolved_flags=*/0, pi05_kv1_gather_arguments(planned_prefix_len_), chunk.idx);
        RpuKernelGraph::active().stage_kernel_no_ddr(
            KernelId::ALL_GATHER_MULTI_CORE_LITTLE_CHUNK,
            GraphKernelNoDdrProof::AllGatherSpm);
        rpu_launch_all_gather_spm_kernel(
            addr(0, "k"), addr(0, "v"), seq_len, /*chunk_elems=*/32,
            sizeof(c10::Half), tp, RpuAllGatherSchedule::LITTLE_CHUNK);
        if (kv_plan.physical_rows() > seq_len) {
            const int64_t pad_elems =
                (kv_plan.physical_rows() - seq_len) * (nkv * hd / tp);
            const uint32_t offset = static_cast<uint32_t>(
                seq_len * (nkv * hd / tp) * DWIDTH);
            rpu_launch_memset_spm_multicore(addr(0, "v") + offset, pad_elems, tp);
        }
    }
    if (kv1_direct) {
        rpu_launch_pi05_kv1_direct_cache_m50d256p64_kernel(
            addr_offset("k").value, addr_offset("v").value,
            k_cache, v_cache, cos_, sin_, rope_start, kv_plan);
    } else {
        rpu_launch_insert_vcache_spm_unified_with_plan(
            v_cache, addr_offset("v").value, nkv, hd, tp,
            /*cache_batch_offset_elems=*/0, kv_plan.physical_rows(), kv_plan);
    }

    // --------------------------------------------------------------------
    // Phase 6: SDPA (causal or 2D mask).
    //
    // Pitfall 3 structural fix: SDPA args (q / output / sdpa_tmp /
    // sdpa_mask) take SPM offsets via addr_offset(...).value.
    // --------------------------------------------------------------------
    int64_t kv_seq_len = ctx().position + ctx().seq_len;
    // One masked action chunk, invariant across layers and denoise bodies.
    // declare_buffers owns a disjoint all-body slot, so pre/post hooks cannot
    // overwrite it. The recorded first upload still executes every REPLAY;
    // prepare_sdpa_mask_cached publishes changed request content beforehand.
    if (ctx().body_iter == 0 && layer_idx == 0 && chunk.idx == 0) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE, PI05_DENOISE_MASK_SCHEDULE_SITE,
                static_cast<int64_t>(
                    Pi05DenoiseMaskScheduleRoute::FIRST_BODY_FIRST_LAYER),
                /*resolved_flags=*/0,
                {seq_len, kv_seq_len, tp, num_cores(), loop_mode_ ? 1 : 0,
                 loop_mode_ ? num_steps_ : 1, num_layers()}, chunk.idx);
        }
        sdpa_dma_mask_to_spm(prepared_mask_,
                             addr_offset("sdpa_mask").value,
                             seq_len, kv_seq_len, tp);
    }
    int mask_type = prepared_mask_.mask_type;
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_attention_route(
            PI05_DENOISE_ATTENTION_SITE,
            AttentionExecutionPolicy::DDR_KV, chunk.idx);
    }
    rpu_launch_sdpa_spm_unified_kernel_v2(
        k_cache, v_cache, mask_type, c10::nullopt,
        addr_offset("q").value,
        addr_offset("output").value,
        addr_offset("sdpa_tmp").value,
        addr_offset("sdpa_mask").value,
        seq_len, nq, nkv, hd,
        kv_seq_len, tp, /*physical_kv_cores=*/8);

    // --------------------------------------------------------------------
    // Phase 7: O_proj (row partition, attn_tp cores)
    // --------------------------------------------------------------------
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE,
            PI05_DENOISE_PREPARE_ALL_REDUCE_SITE,
            static_cast<int64_t>(
                Pi05DenoiseAllReduceRoute::PREPARE_RING_INPUT),
            /*resolved_flags=*/0, {}, chunk.idx);
    }
    rpu_prepare_ring_all_reduce_input(
        addr(0, "oproj"), seq_len, h, tp, num_cores());
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, PI05_DENOISE_O_LINEAR_SITE,
            static_cast<int64_t>(nvfp4_.enabled() ? (linear_acc32_ ? FmbLinearRouteSelector::PI05_NVFP4_V2_ACC32
                                                   : FmbLinearRouteSelector::PI05_NVFP4_V2_ACC16) : FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
    }
    if (nvfp4_.enabled()) {
            rpu_launch_pi05_nvfp4_v2_kernel(addr(0, "output"), lw.o_w, addr(0, "oproj"),
                seq_len, h, nq * hd, 0, tp, lw.o_ws,
                addr(0, Pi05Nvfp4Tables::names[1]), layer_idx, linear_acc32_);
        } else {
            rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "output"), lw.o_w, addr(0, "oproj"),
        seq_len, h, nq * hd,
        /*partition=*/0, tp, /*bias_spm_addr=*/0, lw.o_ws, 0, 0,
            /*force_acc32=*/linear_acc32_);
        }

    // --------------------------------------------------------------------
    // Phase 8: all_reduce_sum_residual (attn)
    // residual2 = reduce_sum(oproj, attn_tp cores) + zero
    // Output goes to all 8.
    // --------------------------------------------------------------------
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE,
            PI05_DENOISE_ATTENTION_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(seq_len, h, use_pi05_ring_xor3(seq_len), num_cores()),
            /*resolved_flags=*/0, {}, chunk.idx);
    }
    const bool xor3_gated_attn = use_pi05_xor3_gated_attn(seq_len);
    if (xor3_gated_attn) {
        pi05_require_stable_graph_dma_capture();
        rpu_launch_pi05_xor3_gated_residual_spm_kernel(
            addr(0, "oproj"), addr(0, "final_out_spm"),
            addr(0, "residual1"), attn_gate, addr(0, "residual2"));
    } else {
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "final_out_spm"), addr(0, "residual2"),
            seq_len, h, tp, num_cores(), use_pi05_ring_xor3(seq_len));
    }

    // --------------------------------------------------------------------
    // Phase 9: GATED attn residual
    // residual2 = attn_gate * residual2 + residual1.
    // The ring used zero residual, so no add-then-sub round trip is needed.
    // --------------------------------------------------------------------
    if (!xor3_gated_attn) {
        emit_gated_residual(addr(0, "residual2"), addr(0, "residual1"),
                            attn_gate, seq_len);
    }

    // --------------------------------------------------------------------
    // Phase 10: Post-attention AdaRMSNorm (MLP side): residual2 -> residual1
    // (1+scale_mlp) baked into mlp_scale; then add shift_mlp.
    // --------------------------------------------------------------------
    emit_adarms_norm_shift(
        addr(0, "residual2"), addr(0, "residual1"),
        mlp_scale, mlp_shift, seq_len, PI05_DENOISE_MLP_NORM_SITE, chunk.idx);

    // --------------------------------------------------------------------
    // Phase 11: MLP pipeline
    // --------------------------------------------------------------------
    const bool xor3_gated_mlp = use_pi05_xor3_gated_mlp(seq_len);
    TORCH_CHECK(!pi05_xor3_gated_mlp_opt_in_ || xor3_gated_mlp,
                "Pi MLP XOR3+gated runtime owner/profile drifted");
    if (pi05_geglu_m50_opt_in_) {
        emit_pi05_geglu_m50_mlp(lw, chunk, mlp_gate);
    } else if (pi05_w4_fastpath_opt_in_ && xor3_gated_mlp) {
        emit_pi05_w4_xor3_mlp(lw, chunk, mlp_gate, layer_idx);
    } else if ((layer_idx == 0 && use_pi05_gateup_resident(seq_len)) ||
            (layer_idx < pi05_gateup_resident_layers_ && use_pi05_gateup_bundle(seq_len))) {
        TORCH_CHECK(chunk.idx == 0 && chunk.offset == 0,
                    "Pi0.5 resident Gate/Up requires the complete action chunk");
        emit_resident_gateup_mlp(lw, seq_len, layer_idx);
    } else {
      emit_mlp_pipeline(lw.gate_proj_w, lw.up_proj_w, lw.down_proj_w,
                      seq_len, ActivationKind::GELU,
                      lw.gate_ws, lw.up_ws, lw.down_ws,
                      nvfp4_.enabled() ? addr(0, "nvfp4_gate_ts") : 0,
                      nvfp4_.enabled() ? addr(0, "nvfp4_up_ts") : 0,
                      nvfp4_.enabled() ? addr(0, "nvfp4_down_ts") : 0, layer_idx,
                      /*down/residual bf16=*/false, false, /*acc32=*/linear_acc32_, false, false,
                      /*bind_silu_mul_route=*/false,
                      /*ring residual=*/addr(0, "final_out_spm"), use_pi05_ring_xor3(seq_len), nvfp4_.enabled());
    }

    // --------------------------------------------------------------------
    // Phase 13: GATED MLP residual
    // The MLP ring also used zero: residual1 = reduce(down).
    // Apply the gate, then add the original residual2 once.
    // --------------------------------------------------------------------
    if (!xor3_gated_mlp) {
        emit_gated_residual(addr(0, "residual1"), addr(0, "residual2"),
                            mlp_gate, seq_len);
    }

    // --------------------------------------------------------------------
    // Phase 14: Output DMA (unless output stays in SPM for next layer)
    // --------------------------------------------------------------------
    if (!ctx().output_to_spm) {
        if (pi05_graph_dma_policy_ & PI05_DENOISE_ELIDE_UNUSED_DMA) {
            TORCH_CHECK(layer_idx == 17 && chunk.idx == 0 && chunk.len == 50,
                        "Pi0.5 unused-output DMA elision reached a non-final group boundary");
            // The immediate post-body norm reads residual1 directly. This
            // Graph caller discards FMB's returned DDR output tensor.
        } else {
            emit_layer_output_dma(layer_idx, chunk);  // from "residual1"
        }
    }
}

void Pi05DenoiseStepModel::emit_pi05_action_spm_input(const ChunkInfo& chunk) {
    TORCH_CHECK(chunk.idx == 0 && chunk.offset == 0 && chunk.len == 50,
                "Pi0.5 SPM action input requires the original whole action chunk");
    TORCH_CHECK(SPM_ALLOC.is_initialized(), "Pi0.5 SPM action allocator is not initialized");
    const uint32_t source = addr(0, "action_emb_core0_spm");
    const uint32_t destination = addr(0, "residual1");
    pi05_validate_action_spm_copy(source, destination, SPM_ALLOC.addr(0, 0),
                                 SpmAllocator::SPM_USABLE);
    ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
        PI05_DENOISE_ACTION_SPM_INPUT_SITE, 1, 0, pi05_graph_dma_arguments(planned_prefix_len_));
    // Source is always core0. Existing scatter stride0 copies these exact bits
    // to every core, including core0, and preserves all14 cross-stream barriers.
    rpu_launch_spm_scatter_spm_dma(source, 50 * 1024, 0, destination, num_cores());
}

void Pi05DenoiseStepModel::emit_pi05_geglu_m50_mlp(
    const LayerWeights& weights, const ChunkInfo& chunk, uint32_t mlp_gate) {
    TORCH_CHECK(
        use_pi05_geglu_m50(chunk.len) && loop_mode_ &&
            pi05_action_steps_admitted(num_steps_) && num_steps_ == planned_num_steps_ &&
            ctx().has_complete_physical_manifest() &&
            chunk.idx == 0 && chunk.offset == 0 && ctx().position == planned_prefix_len_ &&
            ctx().batch_size == 1 && !ctx().is_causal &&
            ctx().attention_policy == AttentionExecutionPolicy::DDR_KV,
        "Pi M50 GeGLU runtime owner/profile drifted");
    pi05_require_stable_graph_dma_capture();
    ctx().consume_physical_route(
        FmbRouteFamily::LINEAR, PI05_DENOISE_GEGLU_M50_LINEAR_SITE,
        pi05_geglu_m50_selector(pi05_is_fp16_action(layer_weights_)), 0,
        pi05_geglu_m50_arguments(), chunk.idx);
    rpu_launch_pi05_denoise_gate_up_geglu_w8a16_m50_kernel(
        addr(0, "residual1"), weights.gate_proj_w, weights.up_proj_w,
        addr(0, "gate"), weights.gate_ws, weights.up_ws);
    emit_pi05_down_and_mlp_residual(weights, chunk, mlp_gate);
}

void Pi05DenoiseStepModel::emit_pi05_w4_xor3_mlp(
    const LayerWeights& weights, const ChunkInfo& chunk, uint32_t mlp_gate, int layer_idx) {
    TORCH_CHECK(
        pi05_w4_fastpath_opt_in_ && !pi05_geglu_m50_opt_in_ &&
            use_pi05_xor3_gated_mlp(chunk.len) && loop_mode_ &&
            pi05_action_steps_admitted(num_steps_) && num_steps_ == planned_num_steps_ &&
            ctx().has_complete_physical_manifest() &&
            chunk.idx == 0 && chunk.offset == 0 && ctx().position == planned_prefix_len_ &&
            ctx().batch_size == 1 && !ctx().is_causal &&
            ctx().attention_policy == AttentionExecutionPolicy::DDR_KV,
        "Pi W4 MLP XOR3 runtime owner/profile drifted");
    pi05_require_stable_graph_dma_capture();
    if (pi05_nvfp4_geglu_m50_opt_in_) {
        TORCH_CHECK(use_pi05_nvfp4_geglu_m50(chunk.len), "Pi NVFP4 GeGLU runtime owner/profile drifted");
        ctx().consume_physical_route(FmbRouteFamily::LINEAR, PI05_DENOISE_NVFP4_GEGLU_LINEAR_SITE,
            PI05_DENOISE_NVFP4_GEGLU_ACC16_M320N64K128, 0, pi05_nvfp4_geglu_m50_arguments(), chunk.idx);
        rpu_launch_pi05_denoise_gate_up_geglu_nvfp4_acc16_m50_kernel(
            addr(0, "residual1"), weights.gate_proj_w, weights.up_proj_w,
            addr(0, "gate"), weights.gate_ws, weights.up_ws,
            addr(0, "nvfp4_gate_ts"), addr(0, "nvfp4_up_ts"), layer_idx);
    } else if (pi05_w4_geglu_m50_opt_in_) {
        TORCH_CHECK(use_pi05_w4_geglu_m50(chunk.len),
                    "Pi W4 GeGLU runtime owner/profile drifted");
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, PI05_DENOISE_W4_GEGLU_M50_LINEAR_SITE,
            pi05_w4_geglu_m50_selector(), 0,
            pi05_w4_geglu_m50_arguments(), chunk.idx);
        if (pi05_w4_geglu_n64_opt_in_) {
            rpu_launch_pi05_denoise_gate_up_geglu_wint4a16_pgrp_m50n64_kernel(
                addr(0, "residual1"), weights.gate_proj_w, weights.up_proj_w,
                addr(0, "gate"), weights.gate_ws, weights.up_ws);
        } else {
            rpu_launch_pi05_denoise_gate_up_geglu_wint4a16_pgrp_m50_kernel(
                addr(0, "residual1"), weights.gate_proj_w, weights.up_proj_w,
                addr(0, "gate"), weights.gate_ws, weights.up_ws);
        }
    } else {
        // Keep the original generated W4 Gate/GELU/Up/MUL and FP16 retirement.
        // Shared FMB returns after MUL; only Down and its continuation move here.
        emit_mlp_pipeline(weights.gate_proj_w, weights.up_proj_w, weights.down_proj_w,
            chunk.len, ActivationKind::GELU,
            weights.gate_ws, weights.up_ws, weights.down_ws,
            nvfp4_.enabled() ? addr(0, "nvfp4_gate_ts") : 0,
            nvfp4_.enabled() ? addr(0, "nvfp4_up_ts") : 0,
            nvfp4_.enabled() ? addr(0, "nvfp4_down_ts") : 0, layer_idx,
            /*down/residual bf16=*/false, false, /*acc32=*/linear_acc32_, false, true,
            /*bind_silu_mul_route=*/false,
            /*ring residual=*/addr(0, "final_out_spm"), use_pi05_ring_xor3(chunk.len), nvfp4_.enabled());
    }
    emit_pi05_down_and_mlp_residual(weights, chunk, mlp_gate, layer_idx);
}

void Pi05DenoiseStepModel::emit_pi05_down_and_mlp_residual(
    const LayerWeights& weights, const ChunkInfo& chunk, uint32_t mlp_gate, int layer_idx) {
    const int64_t rows = chunk.len;
    TORCH_CHECK(!pi05_xor3_gated_mlp_opt_in_ || use_pi05_xor3_gated_mlp(rows),
                "Pi MLP XOR3+gated runtime owner/profile drifted");
    // Down preserves its installed precision and FP16 output retirement.
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, FMB_SHARED_MLP_AUTO_TILE_SITE,
            static_cast<int64_t>(nvfp4_.enabled() ? (linear_acc32_ ? FmbLinearRouteSelector::PI05_NVFP4_V2_ACC32
                                                   : FmbLinearRouteSelector::PI05_NVFP4_V2_ACC16)
                                                  : FmbLinearRouteSelector::AUTO_TILE), 0, {});
    }
    if (nvfp4_.enabled()) {
        rpu_launch_pi05_nvfp4_v2_kernel(addr(0, "gate"), weights.down_proj_w,
            addr(0, "down"), rows, 1024, 4096, 0, 8, weights.down_ws,
            addr(0, "nvfp4_down_ts"), layer_idx, linear_acc32_);
    } else {
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "gate"), weights.down_proj_w, addr(0, "down"),
        rows, 1024, 4096, /*partition=*/0, 8, 0, weights.down_ws, 0, 0,
            /*force_acc32=*/linear_acc32_);
    }
    if (pi05_xor3_gated_mlp_opt_in_) {
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE, PI05_DENOISE_XOR3_GATED_MLP_REDUCE_SITE,
            PI05_DENOISE_XOR3_GATED_MLP_M50N1024, 0,
            pi05_xor3_gated_mlp_arguments(), chunk.idx);
        // The launcher checks all five absolute SPM ranges are disjoint.
        // reduce(down) -> FP16 gate multiply -> FP16 residual2 addition.
        rpu_launch_pi05_xor3_gated_residual_spm_kernel(
            addr(0, "down"), addr(0, "final_out_spm"),
            addr(0, "residual2"), mlp_gate, addr(0, "residual1"));
        return;
    }
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE, FMB_SHARED_MLP_RING_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(rows, 1024, use_pi05_ring_xor3(rows)),
            0, {}, ctx().physical_route_invocation);
    }
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "down"), addr(0, "final_out_spm"), addr(0, "residual1"),
        rows, 1024, 8, 8, use_pi05_ring_xor3(rows));
}


void Pi05DenoiseStepModel::emit_resident_gateup_mlp(
    const LayerWeights& weights, int64_t rows, int64_t layer_idx) {
    // Same independent Gate -> GELU -> Up -> MUL -> Down -> ring sequence as
    // emit_mlp_pipeline(GELU, ACC16). Only the first two raw-weight reads move.
    for (int64_t projection = 0; projection < 2; ++projection) {
        const bool bundle = use_pi05_gateup_bundle(rows);
        const int64_t invocation = bundle ? 2 * layer_idx + projection : projection;
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR,
                bundle ? PI05_DENOISE_BUNDLE_LINEAR_SITE : PI05_DENOISE_GATEUP_RESIDENT_SITE,
                PI05_DENOISE_SPM_WEIGHT_M512N48K128, 0,
                bundle ? pi05_gateup_bundle_linear_arguments(
                             pi05_gateup_resident_layers_, layer_idx, projection)
                       : pi05_gateup_resident_arguments(), invocation);
        }
        rpu_launch_pi05_gateup_resident_w8_kernel(
            addr(0, "residual1"),
            bundle ? addr(0, "resident_gateup_bundle") +
                         static_cast<uint32_t>(invocation * PI05_DENOISE_RESIDENT_W8_BYTES)
                   : addr(0, projection == 0 ? "resident_gate_w8" : "resident_up_w8"),
            addr(0, projection == 0 ? "gate" : "up"),
            projection == 0 ? weights.gate_ws : weights.up_ws);
        if (projection == 0) {
            rpu_launch_eltwise_unary_spm_kernel(
                addr(0, "gate"), addr(0, "gate"), rows * (4096 / 8),
                ValuOpType::ADD, GeluMode::TANH);
        }
    }
    rpu_launch_eltwise_binary_spm_kernel(
        addr(0, "gate"), addr(0, "up"), addr(0, "gate"),
        rows * (4096 / 8), ValuOpType::MUL, c10::Half(1.0));
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, FMB_SHARED_MLP_AUTO_TILE_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE), 0, {});
    }
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "gate"), weights.down_proj_w, addr(0, "down"),
        rows, 1024, 4096, /*partition=*/0, 8, 0, weights.down_ws, 0, 0,
            /*force_acc32=*/linear_acc32_);
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE, FMB_SHARED_MLP_RING_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(rows, 1024, use_pi05_ring_xor3(rows)),
            0, {}, ctx().physical_route_invocation);
    }
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "down"), addr(0, "final_out_spm"), addr(0, "residual1"),
        rows, 1024, 8, 8, use_pi05_ring_xor3(rows));
}

void Pi05DenoiseStepModel::emit_pi05_gateup_bundle_preload() {
    TORCH_CHECK(use_pi05_gateup_bundle(chunk_size_) && loop_mode_ &&
                    num_steps_ == 10 && ctx().body_iter == 0 &&
                    ctx().has_complete_physical_manifest(),
                "Pi0.5 Gate/Up bundle preload requires its complete loop descriptor");
    ctx().consume_physical_route(
        FmbRouteFamily::MUTABLE_DMA, PI05_DENOISE_BUNDLE_PRELOAD_SITE, 5, 0,
        pi05_gateup_bundle_arguments(pi05_gateup_resident_layers_));
    const int64_t bytes = pi05_gateup_resident_layers_ * 2 * PI05_DENOISE_RESIDENT_W8_BYTES;
    const uint32_t destination = addr(0, "resident_gateup_bundle");
    const uint64_t spm_base = SPM_ALLOC.addr(0, 0);
    TORCH_CHECK(resident_gateup_bundle_.scalar_type() == at::kChar &&
                    resident_gateup_bundle_.device().type() == at::kPrivateUse1 &&
                    resident_gateup_bundle_.is_contiguous() &&
                    resident_gateup_bundle_.dim() == 2 &&
                    resident_gateup_bundle_.size(0) == 8 &&
                    resident_gateup_bundle_.size(1) == bytes &&
                    resident_gateup_bundle_.nbytes() == static_cast<size_t>(8 * bytes) &&
                    resident_gateup_bundle_src_base_ == RpuGetDevAddr(resident_gateup_bundle_.data_ptr()) &&
                    destination >= spm_base && destination % 256 == 0 &&
                    static_cast<uint64_t>(bytes) <= SpmAllocator::SPM_USABLE &&
                    destination - spm_base <= SpmAllocator::SPM_USABLE - bytes,
                "Pi0.5 Gate/Up bundle owner/base/SPM extent drift");
    TORCH_CHECK(graph_dma::active(), "Pi0.5 Gate/Up bundle requires Graph BUILD or REPLAY");
    rpu_ddr_flush(resident_gateup_bundle_.data_ptr<int8_t>());
    RpuKernelGraph::active().keep_alive(resident_gateup_bundle_);
    graph_dma::stage_census_mutable_burst(
        resident_gateup_bundle_src_base_, resident_gateup_bundle_, GraphOuterFastDmaSide::Source,
        0, resident_gateup_bundle_.nbytes(), 8);
    rpu_launch_ddr_scatter_spm_dma_mutable(
        &resident_gateup_bundle_src_base_, 0, bytes / DWIDTH, bytes, destination, 8);
}

void Pi05DenoiseStepModel::emit_pre_layers_body() {
    if (ctx().body_iter==0 && ctx().has_complete_physical_manifest()) {
        if (pi05_w4_fastpath_opt_in_) {
            ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
                PI05_DENOISE_W4_FASTPATH_POLICY_SITE, 1, 0, pi05_w4_fastpath_arguments());
        }
        if (pi05_xor3_gated_mlp_opt_in_) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                PI05_DENOISE_XOR3_GATED_MLP_POLICY_SITE, 1, 0,
                pi05_xor3_gated_mlp_arguments());
        }
        if (pi05_nvfp4_geglu_m50_opt_in_) {
            ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
                PI05_DENOISE_NVFP4_GEGLU_POLICY_SITE, 1, 0, pi05_nvfp4_geglu_m50_arguments());
        }
        if (pi05_w4_geglu_m50_opt_in_) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                PI05_DENOISE_W4_GEGLU_M50_POLICY_SITE, 1, 0,
                pi05_w4_geglu_m50_arguments());
        }
        if (pi05_geglu_m50_opt_in_) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                PI05_DENOISE_GEGLU_M50_POLICY_SITE, 1, 0,
                pi05_geglu_m50_arguments());
        }
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            PI05_DENOISE_Q_EPILOGUE_POLICY_SITE,1,0,
            pi05_q_rope_policy_arguments(pi05_q_rope_epilogue_opt_in_, planned_prefix_len_, planned_num_steps_));
        if (pi05_kv1_stripe_gather_opt_in_) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                PI05_DENOISE_KV1_POLICY_SITE, 1, 0,
                pi05_kv1_policy_arguments(/*enabled=*/true, planned_prefix_len_));
        }
        if (pi05_kv1_direct_cache_opt_in_) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                PI05_DENOISE_KV1_DIRECT_POLICY_SITE, 1, 0,
                pi05_kv1_pair_owner_opt_in_
                    ? pi05_kv1_pair_owner_policy_arguments(
                          /*enabled=*/pi05_kv1_pair_owner_opt_in_,
                          rope_position_ >= 0
                              ? rope_position_ : ctx().position,
                          planned_prefix_len_, pi05_is_fp16_action(layer_weights_), planned_num_steps_)
                    : pi05_kv1_direct_policy_arguments(
                          /*enabled=*/true,
                          rope_position_ >= 0
                              ? rope_position_ : ctx().position, planned_prefix_len_));
        }
    }
    validate_pi05_graph_dma_runtime();
    if (pi05_graph_dma_policy_ & PI05_DENOISE_ELIDE_UNUSED_DMA) {
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            PI05_DENOISE_UNUSED_OUTPUT_SITE, 1, 0, pi05_graph_dma_arguments(planned_prefix_len_));
    }
    // Spec §2.4 — pre-layers emission. Uses *_mutable DMA variants from
    // the first runnable graph (no FIXED-first scaffolding — codex round 1
    // risk). Mutable bases are written in step_forward before run_all_layers.

    const int64_t h    = hidden_size();
    const int64_t cs   = chunk_size_;
    const int64_t mad  = max_action_dim_;

    if (use_pi05_gateup_bundle(cs) && ctx().body_iter == 0) {
        emit_pi05_gateup_bundle_preload();
    }
    if (use_pi05_gateup_resident(cs) && ctx().body_iter == 0) {
        TORCH_CHECK(loop_mode_ && num_steps_ == 10,
                    "Pi0.5 resident Gate/Up loop profile drift");
        for (int64_t projection = 0; projection < 2; ++projection) {
            if (ctx().has_complete_physical_manifest()) {
                ctx().consume_physical_route(
                    FmbRouteFamily::MUTABLE_DMA, PI05_DENOISE_GATEUP_PRELOAD_SITE,
                    5, 0, pi05_gateup_resident_arguments(), projection);
            }
            rpu_launch_pi05_gateup_resident_w8_preload(
                projection == 0 ? resident_gate_src_base_ : resident_up_src_base_,
                projection == 0 ? resident_gate_w8_ : resident_up_w8_,
                addr(0, projection == 0 ? "resident_gate_w8" : "resident_up_w8"));
        }
    }

    // final_out_spm is declared ALL-phase and first written by the final
    // norm after all layers. Re-zero it each body: the preceding Euler step
    // leaves the previous final norm here. This requires no extra SPM.
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE,
            PI05_DENOISE_ZERO_GATED_RESIDUAL_SITE,
            pi05_denoise_residual_route(cs),
            /*resolved_flags=*/0, {h});
    }
    rpu_launch_memset_spm_multicore(
        addr(0, "final_out_spm"), cs * h, num_cores());

    if (use_pi05_adarms_body_stream()) {
        TORCH_CHECK(loop_mode_ && num_steps_ == 10 && ctx().has_complete_physical_manifest(),
                    "Pi0.5 body-table DMA requires its complete loop descriptor");
        const int64_t body = ctx().body_iter;
        ctx().consume_physical_route(
            FmbRouteFamily::MUTABLE_DMA, PI05_DENOISE_ADARMS_TABLE_DMA_SITE,
            static_cast<int64_t>(Pi05DenoiseMutableDmaRoute::DDR_BROADCAST_TABLE_BODY),
            0, pi05_adarms_body_arguments(body), body);
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            adarms_table_src_base_, adarms_table_, body * PI05_DENOISE_ADARMS_BODY_BYTES,
            PI05_DENOISE_ADARMS_BODY_BYTES / DWIDTH, addr(0, "adarms_table"), num_cores());
    } else if (use_spm_adarms_table() && ctx().body_iter == 0) {
        TORCH_CHECK(loop_mode_ && num_steps_ == adarms_table_.size(0),
                    "Pi0.5 resident AdaRMS table/loop profile drift");
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA, PI05_DENOISE_ADARMS_TABLE_DMA_SITE,
                static_cast<int64_t>(Pi05DenoiseMutableDmaRoute::DDR_BROADCAST_TABLE_ONCE),
                /*resolved_flags=*/0, {adarms_table_.size(0), num_layers()});
        }
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &adarms_table_src_base_, 0, adarms_table_.numel(),
            addr(0, "adarms_table"), num_cores());
    }

    // [A1] x_t DDR → x_t_spm broadcast (MUTABLE — REPLAY rewrites the base
    // via Queue_t::update_dma_kernel). In loop mode, load the initial noise x0
    // ONLY on body_iter==0; iters 1..N-1 read the in-SPM Euler result the
    // post-hook leaves in x_t_spm (Task 3). Single-step always loads.
    if (!loop_mode_ || ctx().body_iter == 0) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                PI05_DENOISE_PRE_X_DMA_SITE,
                static_cast<int64_t>(
                    Pi05DenoiseMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                /*resolved_flags=*/0);
        }
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            loop_mode_ ? &x0_src_base_ : &x_t_src_base_,
            /*src_offset_bytes=*/0,
            /*num_elements=*/cs * mad,
            /*spm_addr_unified=*/addr(0, "x_t_spm"),
            /*num_cores=*/num_cores());
    }

    // [A2] cond_step DDR → "cond" SPM scatter (MUTABLE — REPLAY rewrites
    // cond_src_base_). Mirrors AdaRMSModel Phase 0 — now exclusively owned
    // by pre_layers_fn (the layer body's Phase 0 is removed; plan §C2).
    if (!loop_mode_ || !adarms_table_.defined()) {
        const int64_t local_k           = h / condition_tp();
        const int64_t core_stride_bytes = local_k * DWIDTH;
        // In loop mode, scatter cond_all[body_iter] (row stride h*DWIDTH bytes);
        // single-step uses cond_src_base_ at offset 0.
        const int64_t cond_off_b = loop_mode_ ? ctx().body_iter * h * DWIDTH : 0;
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                PI05_DENOISE_PRE_COND_DMA_SITE,
                static_cast<int64_t>(
                    Pi05DenoiseMutableDmaRoute::DDR_SCATTER_TO_SPM),
                /*resolved_flags=*/0);
        }
        rpu_launch_ddr_scatter_spm_dma_mutable(
            loop_mode_ ? &cond_all_src_base_ : &cond_src_base_,
            /*src_offset_bytes=*/cond_off_b,
            /*elements_per_core=*/local_k,
            /*core_stride_bytes=*/core_stride_bytes,
            /*spm_addr_unified=*/addr(0, "cond"),
            /*num_cores=*/condition_tp());
    }

    // [A3] action_in_proj single-core SPM linear with bias.
    // M=cs=50, N=H_ada=1024, K=max_action_dim=32 (K%16==0 OK; num_cores=1
    // skips the K%(16*8) requirement). Bias preloaded into
    // action_in_proj_b_spm via D-502 callback (Task 2.1).
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, PI05_DENOISE_PRE_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {loop_mode_ ? 1 : 0, loop_mode_ ? num_steps_ : 1});
    }
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        /*input_spm_addr=*/addr(0, "x_t_spm"),
        /*weight=*/action_in_proj_w_,
        /*output_spm_addr=*/addr(0, "action_emb_core0_spm"),
        /*M=*/cs, /*N=*/h, /*K=*/mad,
        /*partition=*/1, /*num_cores=*/1,
        /*bias_spm_addr=*/addr(0, "action_in_proj_b_spm"), at::Tensor(), 0, 0,
            /*force_acc32=*/linear_acc32_);

    // [A4] core-0 SPM → action_emb_stage_ DDR copy (100 KB; FIXED —
    // action_emb_stage_ data_ptr stable across step + sample_actions per
    // spec §3.3). The layer body's Phase 1 (emit_layer_input_dma) then
    // reads from action_emb_stage_ via FMB's hidden_in_src_base_ mutable
    // mechanism. [A5] from spec §2.4 is DROPPED — see plan §C5.
    if (pi05_graph_dma_policy_ & PI05_DENOISE_SPM_ACTION_STAGING) {
        // This producer site pairs with the layer0 SPM-copy consumer. The
        // staging tensor remains FMB's shape/lifetime owner, with no DDR read.
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            PI05_DENOISE_ACTION_SPM_PRODUCER_SITE, 1, 0, pi05_graph_dma_arguments(planned_prefix_len_));
    } else {
        rpu_launch_spm_copy_ddr_dma(
            /*spm_addr_unified=*/addr(0, "action_emb_core0_spm"),
            /*ddr_ptr=*/action_emb_stage_.data_ptr<c10::Half>(),
            /*num_elements=*/cs * h);
    }
}

void Pi05DenoiseStepModel::emit_post_layers_body() {
    // Spec §2.4 — post-layers. Pre-condition: build_layer_subgraph for the
    // last layer left "residual1" with the final layer output in broadcast SPM.

    const int64_t h   = hidden_size();
    const int64_t cs  = chunk_size_;
    const int64_t mad = max_action_dim_;

    // [Z1] Final PiGemmaRMSNorm — mirrors AdaRMS Phase A + Phase 2 (plain;
    // no gated residual on final norm). GEMV(cond * final_dense_w + bias)
    // → final_gemv. RMSNorm + shift → final_out_spm, preserving FP16 rounding.
    const uint32_t final_scale = adarms_gemv_to_spm(addr(0, "cond"),
                       final_norm_dense_w_, final_norm_dense_b_,
                       addr(0, "final_gemv"),
                       addr(0, "bias_temp"),
                       addr(0, "gemv_partial"),
                       final_norm_dense_ws_, 2 * num_layers());
    const uint32_t final_shift = final_scale + (uint32_t)(h * DWIDTH);
    // Gate slice (offset 2*h*DWIDTH) is unused for final norm.
    emit_adarms_norm_shift(
        addr(0, "residual1"), addr(0, "final_out_spm"),
        final_scale, final_shift, cs, PI05_DENOISE_FINAL_NORM_SITE);

    // [Z2] action_out_proj uses the action owner's cold accumulation policy.
    // Explicit ACC32 retains the previous FP32 accumulation at this boundary.
    // M=cs, N=max_action_dim=32, K=H_ada=1024 (K=1024, K%16==0).
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, PI05_DENOISE_POST_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0);
    }
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        /*input_spm_addr=*/addr(0, "final_out_spm"),
        /*weight=*/action_out_proj_w_,
        /*output_spm_addr=*/addr(0, "v_t_core0_spm"),
        /*M=*/cs, /*N=*/mad, /*K=*/h,
        /*partition=*/1, /*num_cores=*/1,
        /*bias_spm_addr=*/addr(0, "action_out_proj_b_spm"),
        /*scale=*/{}, /*nvfp4_tensor_scale_spm_addr=*/0,
        /*nvfp4_layer_id=*/0, /*force_acc32=*/linear_acc32_);

    if (loop_mode_) {
        // [Z3-loop] On-device Euler: x_t_spm = x_t_spm + dt_·v_t_core0_spm
        // (in-place, core 0). x and v are both [cs, mad] (equal widths). dt_ is
        // baked at BUILD. x_t_spm is private (Pi05's all-phase-0 buffer layout
        // gives every buffer its own slot — see rpu_spm_allocator.cpp:86-97), so
        // this result survives the layer loop and persists to the next body's
        // [A1]-skipped / [A3] read.
        rpu_launch_eltwise_binary_spm_kernel(
            /*a=*/addr(0, "x_t_spm"),
            /*b=*/addr(0, "v_t_core0_spm"),
            /*output=*/addr(0, "x_t_spm"),
            /*num_elements=*/cs * mad,
            ValuOpType::ADD, /*alpha=*/dt_, /*num_cores=*/1);
        // Every Euler update remains in order in x_t_spm. Only the final
        // action has a DDR consumer outside this admitted composite Graph.
        const bool final_only = pi05_graph_dma_policy_ & PI05_DENOISE_ELIDE_UNUSED_DMA;
        if (!final_only || pi05_graph_dma_write_final_body(ctx().body_iter, num_steps_)) {
            if (ctx().has_complete_physical_manifest()) {
                ctx().consume_physical_route(
                    FmbRouteFamily::MUTABLE_DMA, PI05_DENOISE_POST_LOOP_DMA_SITE,
                    static_cast<int64_t>(final_only
                        ? Pi05DenoiseMutableDmaRoute::SPM_COPY_FINAL_BODY_TO_DDR
                        : Pi05DenoiseMutableDmaRoute::SPM_COPY_TO_DDR), 0,
                    final_only ? pi05_graph_dma_arguments(planned_prefix_len_) : std::vector<int64_t>{});
            }
            rpu_launch_spm_copy_ddr_dma_mutable(
                addr(0, "x_t_spm"), &x_out_dst_base_, 0, cs * mad);
        }
    } else {
        // [Z3] core-0 v_t_core0_spm → v_t_buf DDR (MUTABLE — REPLAY rewrites
        // v_t_dst_base_). Sig per src/core/rpu_kernel_decls.h:1216-1220:
        //   (uint32_t spm_addr_unified, const uint64_t* live_dst_base,
        //    int64_t dst_offset_bytes, int64_t num_elements)
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                PI05_DENOISE_POST_STEP_DMA_SITE,
                static_cast<int64_t>(
                    Pi05DenoiseMutableDmaRoute::SPM_COPY_TO_DDR),
                /*resolved_flags=*/0);
        }
        rpu_launch_spm_copy_ddr_dma_mutable(
            /*spm_addr_unified=*/addr(0, "v_t_core0_spm"),
            /*live_dst_base=*/&v_t_dst_base_,
            /*dst_offset_bytes=*/0,
            /*num_elements=*/cs * mad);
    }
}

uint32_t Pi05DenoiseStepModel::adarms_gemv_to_spm(uint32_t cond_spm_addr,
                                              const at::Tensor& dense_w,
                                              const at::Tensor& dense_b,
                                              uint32_t gemv_out_spm_addr,
                                              uint32_t bias_temp_spm_addr,
                                              uint32_t partial_spm_addr,
                                              const at::Tensor& dense_scale,
                                              int64_t table_row)
{
    int64_t h = hidden_size();
    int64_t out_features = 3 * h;

    if (loop_mode_ && adarms_table_.defined()) {
        TORCH_CHECK(table_row >= 0 && table_row < 2 * num_layers() + 1
                        && ctx().body_iter >= 0 && ctx().body_iter < adarms_table_.size(0),
                    "Pi0.5 AdaRMS table row/body out of range");
        const int64_t offset =
            (ctx().body_iter * (2 * num_layers() + 1) + table_row) * out_features * DWIDTH;
        if (use_spm_adarms_table()) {
            const int64_t local_offset = use_pi05_adarms_body_stream()
                ? table_row * out_features * DWIDTH : offset;
            return addr(0, "adarms_table") + static_cast<uint32_t>(local_offset);
        }
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA, PI05_DENOISE_ADARMS_TABLE_DMA_SITE,
                static_cast<int64_t>(Pi05DenoiseMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                /*resolved_flags=*/0, {adarms_table_.size(0), num_layers()});
        }
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &adarms_table_src_base_, offset, out_features, gemv_out_spm_addr,
            num_cores());
        return gemv_out_spm_addr;
    }

    // Step 1: DMA bias -> bias_temp (replicated across all 8 cores, acts
    // as the residual input to the fused all_reduce+bias kernel).
    rpu_launch_ddr_broadcast_spm_dma(
        dense_b.data_ptr<c10::Half>(), out_features, bias_temp_spm_addr, num_cores());
    if (condition_tp() != num_cores()) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(FmbRouteFamily::ALL_REDUCE,
                PI05_DENOISE_COND_PREPARE_SITE, 3, 0,
                {condition_tp(), num_cores(), 1});
        }
        rpu_prepare_ring_all_reduce_input(partial_spm_addr, 1, out_features,
                                          condition_tp(), num_cores());
    }
    // Step 2: Row-partition GEMV. partial_spm_addr holds the per-core
    // partial [3H] along the local_k = H/NUM_CORES split.
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, PI05_DENOISE_GEMV_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0);
    }
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        cond_spm_addr, dense_w, partial_spm_addr,
        /*M=*/1, /*N=*/out_features, /*K=*/h,
        /*partition=*/0, /*num_cores=*/condition_tp(),
        /*bias_spm_addr=*/0,
        /*scale=*/dense_scale, 0, 0,
            /*force_acc32=*/linear_acc32_);  // W8A16 when defined; fp16 when empty
    // Step 3: Fused all_reduce + bias. reduce(partial) + bias -> gemv_out
    // broadcast to all output cores. Input and output MUST be distinct
    // SPM regions.
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE,
            PI05_DENOISE_GEMV_ALL_REDUCE_SITE,
            pi05_denoise_ring_route(/*rows=*/1, out_features, num_cores()),
            /*resolved_flags=*/0);
    }
    rpu_launch_all_reduce_sum_residual_kernel(
        partial_spm_addr, bias_temp_spm_addr, gemv_out_spm_addr,
        /*M=*/1, /*N=*/out_features,
        /*input_num_cores=*/condition_tp(), /*output_num_cores=*/num_cores());
    // Step 4: (1+scale) -- add +1.0 only to the first hidden_size elements
    rpu_launch_eltwise_binary_scalar_spm_kernel(
        gemv_out_spm_addr, c10::Half(1.0),
        gemv_out_spm_addr, h, ValuOpType::ADD, num_cores());
    return gemv_out_spm_addr;
}

}  // namespace v3

// ============================================================================
// Registry + C API
// ============================================================================
using Pi05DenoiseStepRegistry = ModelHandleRegistry<v3::Pi05DenoiseStepModel>;

std::vector<int64_t> rpu_pi05_denoise_step_planner_cache_identity(int64_t handle) {
    return Pi05DenoiseStepRegistry::get(handle, "rpu_pi05_denoise_step_planner_cache_identity")
        ->planner_cache_identity();
}

void rpu_pi05_denoise_step_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk) {
    Pi05DenoiseStepRegistry::get(handle, "rpu_pi05_denoise_step_set_chunk_envelope")
        ->set_chunk_envelope(max_kv_len, chunk);
}

void rpu_pi05_denoise_step_bind_kvinsert_costs(
        int64_t handle, at::IntArrayRef identity,
        const std::string& catalog_sha256, at::IntArrayRef certificate_rows) {
    Pi05DenoiseStepRegistry::get(handle, "rpu_pi05_denoise_step_bind_kvinsert_costs")
        ->bind_kvinsert_costs(identity, catalog_sha256, certificate_rows);
}

std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_pi05_denoise_step_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor, int64_t site_id,
    int64_t invocation, int64_t route) {
    return Pi05DenoiseStepRegistry::get(handle, "rpu_pi05_denoise_step_kvinsert_exact_candidate")
        ->mint_kvinsert_exact_candidate(descriptor, site_id, invocation, route);
}

KvInsertCostDomainQuery rpu_pi05_denoise_step_kvinsert_cost_domain(
        int64_t handle, at::IntArrayRef descriptor) {
    return Pi05DenoiseStepRegistry::get(handle, "rpu_pi05_denoise_step_kvinsert_cost_domain")
        ->kvinsert_cost_domain("pi05_denoise_step", descriptor);
}

std::string rpu_pi05_denoise_step_kvinsert_cost_catalog_sha256(int64_t handle) {
    return Pi05DenoiseStepRegistry::get(
        handle, "rpu_pi05_denoise_step_kvinsert_cost_catalog_sha256")
        ->kvinsert_cost_catalog_sha256();
}

void rpu_pi05_denoise_step_set_execution_core_count(int64_t handle, int64_t cores) {
    Pi05DenoiseStepRegistry::get(handle, "rpu_pi05_denoise_step_set_execution_core_count")
        ->set_execution_cores(cores);
}

std::vector<int64_t> rpu_pi05_denoise_step_get_execution_topology(int64_t handle) {
    return Pi05DenoiseStepRegistry::get(handle, "rpu_pi05_denoise_step_get_execution_topology")
        ->execution_topology();
}

int64_t rpu_pi05_denoise_step_create(bool linear_acc32) { return Pi05DenoiseStepRegistry::create(linear_acc32); }
void    rpu_pi05_denoise_step_destroy(int64_t h) {
    Pi05DenoiseStepRegistry::destroy(h, "rpu_pi05_denoise_step_destroy");
}

void rpu_pi05_denoise_step_set_rope_position(int64_t handle,
                                            int64_t position) {
    Pi05DenoiseStepRegistry::get(
        handle, "rpu_pi05_denoise_step_set_rope_position")
        ->set_rope_position(position);
}

void rpu_pi05_denoise_step_set_chunk_size(int64_t handle,
                                         int64_t chunk_size) {
    Pi05DenoiseStepRegistry::get(
        handle, "rpu_pi05_denoise_step_set_chunk_size")
        ->set_configured_chunk_size(chunk_size);
}

void rpu_pi05_denoise_step_set_adarms_table(int64_t handle,
                                           const at::Tensor& table) {
    Pi05DenoiseStepRegistry::get(handle, "rpu_pi05_denoise_step_set_adarms_table")
        ->set_adarms_table(table);
}

int64_t rpu_pi05_denoise_step_get_resolved_chunk_size(int64_t handle) {
    return Pi05DenoiseStepRegistry::get(
        handle, "rpu_pi05_denoise_step_get_resolved_chunk_size")
        ->get_last_resolved_chunk_size();
}

std::vector<int64_t> rpu_pi05_denoise_step_resolve_action_stage_domain(
    int64_t handle, int64_t execution_len, int64_t logical_len,
    int64_t position, int64_t kv_len, int64_t cache_capacity,
    int64_t requested_chunk_size, bool prefer_pad16,
    bool loop_mode, int64_t num_steps) {
    return Pi05DenoiseStepRegistry::get(
               handle,
               "rpu_pi05_denoise_step_resolve_action_stage_domain")
        ->resolve_action_stage_domain(
            execution_len, logical_len, position, kv_len, cache_capacity,
            requested_chunk_size, prefer_pad16, loop_mode, num_steps);
}

void rpu_pi05_denoise_step_set_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w,
    at::TensorList v_w, at::TensorList o_w,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    at::TensorList attn_dense_w, at::TensorList attn_dense_b,
    at::TensorList mlp_dense_w,  at::TensorList mlp_dense_b,
    const at::Tensor& final_norm_dense_w, const at::Tensor& final_norm_dense_b,
    const at::Tensor& action_in_proj_w,   const at::Tensor& action_in_proj_b,
    const at::Tensor& action_out_proj_w,  const at::Tensor& action_out_proj_b,
    const at::Tensor& cos, const at::Tensor& sin,
    int64_t hidden_size, int64_t max_action_dim, int64_t chunk_size,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t num_layers, double eps,
    at::TensorList q_w_scale, at::TensorList k_w_scale,
    at::TensorList v_w_scale, at::TensorList o_w_scale,
    at::TensorList gate_scale, at::TensorList up_scale,
    at::TensorList down_scale,
    at::TensorList attn_dense_w_scale, at::TensorList mlp_dense_w_scale,
    const at::Tensor& final_norm_dense_w_scale, const std::optional<std::vector<at::Tensor>>& nvfp4_tensor_scales)
{
    Pi05DenoiseStepRegistry::get(handle, "rpu_pi05_denoise_step_set_weights")
        ->set_weights(q_w, k_w, v_w, o_w, gate_w, up_w, down_w,
                      attn_dense_w, attn_dense_b,
                      mlp_dense_w,  mlp_dense_b,
                      final_norm_dense_w, final_norm_dense_b,
                      action_in_proj_w,   action_in_proj_b,
                      action_out_proj_w,  action_out_proj_b,
                      cos, sin, hidden_size, max_action_dim, chunk_size,
                      num_q_heads, num_kv_heads, head_dim, num_layers, eps,
                      q_w_scale, k_w_scale, v_w_scale, o_w_scale,
                      gate_scale, up_scale, down_scale,
                      attn_dense_w_scale, mlp_dense_w_scale,
                      final_norm_dense_w_scale, nvfp4_tensor_scales.value_or(std::vector<at::Tensor>{}));
}

void rpu_pi05_denoise_step_forward(
    int64_t handle,
    const at::Tensor& x_t_rpu,
    std::vector<at::Tensor> k_caches,
    std::vector<at::Tensor> v_caches,
    const at::Tensor& cond_step,
    const at::Tensor& attention_mask_4d,
    at::Tensor v_t_buf,
    int64_t prefix_len,
    at::IntArrayRef planned_stage_descriptor)
{
    Pi05DenoiseStepRegistry::get(handle, "rpu_pi05_denoise_step_forward")
        ->step_forward(x_t_rpu, k_caches, v_caches, cond_step,
                       attention_mask_4d, v_t_buf, prefix_len,
                       planned_stage_descriptor);
}

void rpu_pi05_denoise_loop_forward(
    int64_t handle,
    at::Tensor x0_rpu,
    std::vector<at::Tensor> k_caches,
    std::vector<at::Tensor> v_caches,
    at::Tensor cond_all,
    at::Tensor attention_mask_4d,
    at::Tensor x_out,
    double dt,
    int64_t prefix_len,
    int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor)
{
    Pi05DenoiseStepRegistry::get(handle, "rpu_pi05_denoise_loop_forward")
        ->denoise_loop_forward(x0_rpu, k_caches, v_caches, cond_all,
                               attention_mask_4d, x_out, dt, prefix_len,
                               num_steps, planned_stage_descriptor);
}
