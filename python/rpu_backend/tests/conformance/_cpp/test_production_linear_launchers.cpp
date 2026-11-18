// Calls the linked production launchers. No copied register fill or SDK bypass.
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include "rpu_pi05_prefill_pair.h"
#include "rpu_pi05_down_pair.h"
#include "rpu_pi05_producer_chain.h"
#include "rpu_pi05_vision_fc.h"
#include "rpu_pi05_q_rope.h"
#include "rpu_kvinsert_segment_plan.h"
#include "rpu_hyvla_gateup_w8.h"
#include "rpu_hyvla_vlm_gateup_w8.h"
#include "fused/hyvla_kv_grouped.h"
#include "fused/hyvla_w8_o_resident.h"
#include "fused/hyvla_o_ring_postnorm.h"
#include "fused/hyvla_w8_o_ring_postnorm.h"
#include <torch/library.h>
#include <cstring>
#include <sstream>
#include <tuple>

namespace {
using Result = std::tuple<std::vector<at::Tensor>, std::string, std::string, std::string>;

std::string production_kernel_lookup_rejections() {
    RpuExecutionCoordinator::require_graph_quiescent("production kernel lookup rejection conformance");
    RpuExecutionCoordinator::check_current_thread_execution_allowed("production kernel lookup rejection conformance");
    auto& cache = KernelCache::instance();
    TORCH_CHECK(cache.is_initialized(), "production kernel cache must be initialized");
    std::ostringstream result;
    result << "{\"rejected\":[";
    size_t index = 0;
    for (const char* name : {"conformance_unknown_linear_payload_20261021",
                            "lingbot2_routed_col_fp16_m608n32",
                            "wall_prefill_qkv_fp16_tile_owner_m320x2n64k128",
                            "shared_silu_bias_triplet_w16a16_acc16_m320n80k128",
                            "shared_silu_bias_triplet_w8a16_acc16_m336n64k128"}) {
        TORCH_CHECK(cache.get_kernel(name) == nullptr, "unknown/retired production kernel was loadable: ", name);
        TORCH_CHECK(cache.get_program(name) == nullptr, "unknown/retired production program was cached: ", name);
        TORCH_CHECK(cache.graph_snapshot_program_contracts(name) == 0,
                    "unknown/retired production program retained Graph authority: ", name);
        if (index++) result << ',';
        result << '"' << name << '"';
    }
    const char* current = KERNEL_ID_NAMES[static_cast<size_t>(KernelId::QWEN3VL_PREFILL_QKV_PLANAR_DIRECT)];
    TORCH_CHECK(cache.get_kernel(current) != nullptr && cache.get_program(current) != nullptr,
                "current production kernel lookup failed after negative probes");
    TORCH_CHECK(cache.graph_snapshot_program_contracts(current) == 3,
                "current production kernel lacks release-bound Graph authority");
    result << "],\"current_positive\":\"" << current << "\",\"current_contracts\":3,"
           << "\"compute_submitted\":false}";
    return result.str();
}

// Images are raw per-core SPM bytes, prepared by the Python numeric fixture.
// Every image retains a 256-byte guard on both sides, including scratch.
Result production_linear_launchers(
    const std::string& profile, at::TensorList owners,
    at::TensorList images_a, at::TensorList images_b,
    at::IntArrayRef outputs, int64_t rows, int64_t cores, int64_t mode) {
    RpuExecutionCoordinator::require_graph_quiescent("production launcher conformance");
    RpuExecutionCoordinator::check_current_thread_execution_allowed("production launcher conformance");
    TORCH_CHECK(!RpuKernelGraph::has_active(), "launcher conformance needs an idle Graph context");
    TORCH_CHECK(cores == 1 || cores == 8, "launcher conformance supports TP1/8");
    TORCH_CHECK(images_a.size() == images_b.size() && !images_a.empty(), "SPM A/B image arity differs");
    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    SpmAllocator::ScopedTemporary temporary;
    std::vector<uint32_t> offsets, addresses;
    std::vector<size_t> bytes;
    for (size_t i = 0; i < images_a.size(); ++i) {
        const auto& a = images_a[i]; const auto& b = images_b[i];
        TORCH_CHECK(a.device().is_cpu() && b.device().is_cpu() &&
            a.scalar_type() == at::kByte && b.scalar_type() == at::kByte &&
            a.is_contiguous() && b.is_contiguous() && a.dim() == 2 &&
            a.size(0) == cores && a.size(1) > 0 && a.sizes() == b.sizes(),
            "SPM images must be equal CPU uint8 [cores,bytes] shapes");
        const size_t size = a.size(1);
        const auto offset = SPM_ALLOC.alloc_temporary(((size + 255) & ~size_t(255)) + 512);
        offsets.push_back(offset + 256); addresses.push_back(SPM_ALLOC.addr(0, offset + 256));
        bytes.push_back(size);
    }
    for (auto output : outputs) TORCH_CHECK(output >= 0 && size_t(output) < offsets.size(), "output image index");
    auto s = [&](size_t i) { TORCH_CHECK(i < addresses.size(), "missing SPM image ", i); return addresses[i]; };
    std::vector<at::Tensor> installed_owners(owners.begin(), owners.end());
    if (profile == "pi_nvfp4_linear_db_tscale") {
        TORCH_CHECK(installed_owners.size() == 2, "Linear tscale requires weight/scale owners");
        installed_owners[1] = rpu_retain_nvfp4_tile_scale_candidate(installed_owners[1]);
    } else if (profile == "pi_nvfp4_geglu_db_tscale") {
        TORCH_CHECK(installed_owners.size() == 4, "GeGLU tscale requires two weight/scale owners");
        for (int index : {2, 3}) installed_owners[index] = rpu_retain_nvfp4_tile_scale_candidate(installed_owners[index]);
    } else if (profile == "pi_nvfp4_geglu_reference") {
        // The default launcher reads tile-contiguous companions whenever the REF
        // has that payload; pack them as weight installation does (owners 4/5).
        TORCH_CHECK(installed_owners.size() == 4, "GeGLU reference requires two weight/scale owners");
        for (int index : {2, 3})
            installed_owners.push_back(rpu_pack_pi05_nvfp4_tile_scale(installed_owners[index], 4096, 1024, 1));
    }
    auto w = [&](size_t i) -> at::Tensor {
        TORCH_CHECK(i < installed_owners.size(), "missing DDR owner ", i);
        return installed_owners[i].numel() ? installed_owners[i] : at::Tensor{};
    };
    std::vector<std::pair<size_t, size_t>> scale_preloads; // image index, DDR owner index
    std::vector<at::Tensor> scale_preload_cpu;
    if (profile == "pi_nvfp4_linear_db_sspm") {
        TORCH_CHECK(installed_owners.size() == 2 && addresses.size() == 4,
                    "Linear SSPM requires weight/per-core-scale and four SPM images");
        scale_preloads = {{3, 1}};
    } else if (profile == "pi_q_kv1_pair_sspm" || profile == "pi_q_kv1_pair_cache_sspm") {
        TORCH_CHECK(installed_owners.size() == (profile == "pi_q_kv1_pair_cache_sspm" ? 10 : 6) && addresses.size() == 6,
                    "Q+KV1 SSPM requires six DDR owners and six SPM images");
        scale_preloads = {{4, 1}};
    } else if (profile == "pi_nvfp4_geglu_db_sspm") {
        TORCH_CHECK(installed_owners.size() == 4 && addresses.size() == 6,
                    "GeGLU SSPM requires two weight/scale owners and six SPM images");
        scale_preloads = {{4, 2}, {5, 3}};
    }
    for (const auto& [image, owner] : scale_preloads) {
        TORCH_CHECK(w(owner).scalar_type() == at::kByte && w(owner).dim() == 4 &&
                        w(owner).size(0) == cores && w(owner).is_contiguous() &&
                        w(owner).nbytes() == bytes[image] * cores,
                    "SSPM image must own a complete per-core scale slab");
        scale_preload_cpu.push_back(w(owner).cpu().contiguous());
    }
    at::Tensor resident, resident_cpu;
    if (profile == "pi_resident_preload") resident = rpu_pack_pi05_gateup_resident_w8(w(0));
    if (profile == "g05_partial_preload") resident = rpu_pack_g05_down_partial_w8a16(w(0));
    uint64_t resident_address = 0;
    if (resident.defined()) {
        resident_cpu = resident.cpu().contiguous();
        resident_address = rhino_lkn::RpuGetDevAddr(resident.data_ptr());
        TORCH_CHECK(resident_cpu.size(0) == cores && resident_cpu.size(1) == int64_t(bytes[1]),
                    "resident preload image must match the real packed owner");
    }
    const auto emit = [&] {
        if (profile == "pi_resident_preload") {
            rpu_launch_pi05_gateup_resident_w8_preload(resident_address, resident, s(1));
            rpu_launch_pi05_gateup_resident_w8_kernel(s(0), s(1), s(2), w(1));
        } else if (profile == "g05_partial_preload") {
            rpu_launch_g05_down_partial_w8a16_preload(resident, s(1));
            rpu_launch_g05_down_partial_w8a16_acc16_kernel(s(0), w(0), s(1), s(2), w(1));
        } else if (profile == "pi_q_rope") {
            rpu_launch_pi05_q_rope_spm_kernel(s(0), w(0), s(1), w(1), w(2), w(3), mode);
        } else if (profile == "lingbot_packed_gateup") {
            if (w(0).scalar_type() == at::kHalf)
                rpu_launch_lingbot2_shared_gateup_resident_silu_fp16_spm_kernel(s(0), w(0), s(1));
            else rpu_launch_lingbot2_shared_gateup_resident_silu_w8_spm_kernel(s(0), w(0), w(1), s(1));
        } else if (profile == "wall_action_qkv_owner") {
            WallSortingConfig sorting; sorting.fp16_transfer_mask = 4;
            const auto cosine = w(6), sine = w(7);
            rpu_launch_wall_action_qkv_w8a16_multibase_spm_kernel(s(0), w(0), w(1), w(2),
                s(1), s(2), s(3), s(4), s(5), s(6), w(3), w(4), w(5),
                &cosine, &sine, mode, 0, 0, 0, true, sorting, rows);
        } else if (profile == "wall_vision_triplet") {
            WallSortingConfig sorting;
            if (w(0).scalar_type() == at::kHalf) sorting.fp16_transfer_mask = 1;
            rpu_launch_wall_vision_gate_up_swiglu_w8a16_spm_kernel(s(0), w(0), w(1), s(3),
                s(6), s(7), w(2), w(3), s(1), s(2), s(4), s(5), sorting, rows);
        } else if (profile == "wall_vision_shared_bias") {
            WallSortingConfig sorting;
            sorting.vision_row_reuse = true;
            sorting.silu_mul = true;
            rpu_launch_wall_vision_gate_up_swiglu_w8a16_spm_kernel(s(0), w(0), w(1), s(1),
                s(2), s(3), w(2), w(3), 0, 0, 0, 0, sorting, rows);
        } else if (profile == "wall_bf16_output") {
            rpu_launch_linear_spm_to_spm_acc32_out_bf16_kernel(s(0), w(0), s(1),
                rows, 1280, 3584, 0, 8, s(2));
        } else if (profile == "wall_action_ring") {
            WallSortingConfig sorting; sorting.fp16_transfer_mask = 4; sorting.silu_mul = rows == 32;
            rpu_launch_wall_action_linear_ring_postnorm_spm_kernel(s(0), w(0), s(1), s(2),
                mode ? s(2) : s(3), mode ? s(3) : s(4), mode ? s(4) : s(5),
                w(1), mode != 0, 1e-6, 0, sorting, rows);
        } else if (profile == "g05_gate_kv_owner") {
            rpu_launch_g05_gate_kv_owner_only_w8a16_acc16_kernel(s(0), w(0), w(1), w(2),
                w(3), w(4), w(5), s(1), s(2), s(3));
        } else if (profile == "pi_compact_kv") {
            rpu_launch_pi05_kv1_stripe_w8a16_m592n32_kernel(s(0), w(0), s(1), w(1));
        } else if (profile == "pi_paired_geglu") {
            rpu_launch_pi05_prefill_gate_up_geglu_weight_outer_w8a16_c400x2_kernel(s(0), s(1),
                w(0), w(1), s(2), s(3), w(2), w(3), rows);
        } else if (profile == "pi_pair") {
            rpu_launch_pi05_prefill_pair_spm_kernel(mode ? Pi05PrefillPairedProjection::QkvColumn : Pi05PrefillPairedProjection::AttentionOutputRow,
                s(0), s(1), w(0), s(2), s(3), w(1), rows);
        } else if (profile == "pi_down_pair") {
            rpu_launch_pi05_down_pair_spm_kernel(s(0), s(1), w(0), s(2), s(3), w(1), rows);
        } else if (profile == "pi_vision_fc") {
            rpu_launch_pi05_vision_fc_weight_outer_spm_kernel(mode ? Pi05VisionFcProjection::Fc1Column : Pi05VisionFcProjection::Fc2Row,
                s(0), w(0), s(1), s(2), w(1));
        } else if (profile == "pi_a8_online") {
            rpu_launch_pi05_prefill_gate_up_geglu_online_w8a8_c272_kernel(s(0), s(1), s(2), w(0), w(1), w(2), w(3));
        } else if (profile == "pi_w4_geglu") {
            rpu_launch_pi05_denoise_gate_up_geglu_w4a16_pgrp_m50_kernel(s(0), w(0), w(1), s(1), w(2), w(3), rows);
        } else if (profile == "pi_a8_split") {
            rpu_launch_pi05_prefill_gate_up_geglu_w8a8_split_kernel(s(0), s(1), s(2), w(0), w(1), s(3), s(4), w(2), w(3), rows);
        } else if (profile == "qwen_qkv_planar") {
            rpu_launch_linear_spm_to_spm_acc16_kernel(s(0), w(0), s(1),
                rows, 4096, 2048, 1, 8, 0, w(1), 0, 0, false, false, true);
        } else if (profile == "pi_prefill_kv_owner") {
            rpu_launch_pi05_prefill_kv1_pair_owner_w8a16_m400n32x2k2048_kernel(
                s(0), w(0), w(2), s(1), w(1), w(3), s(2), rows);
        } else if (profile == "gr00t_qkv") {
            rpu_launch_gr00t_qkv_multibase_w8_kernel(s(0), w(0), w(1), w(2), s(1), s(2), s(3), s(4), s(5), s(6), w(3), w(4), w(5));
        } else if (profile == "lingbot_qkv") {
            rpu_launch_lingbot2_qkv_multibase_t51_spm_kernel(s(0), w(0), w(1), w(2), s(1), s(2), s(3), s(4), s(5), s(6), w(3), w(4), w(5));
        } else if (profile == "lingbot_prefill_qkv") {
            rpu_launch_lingbot2_prefill_qkv_m224_spm_kernel(s(0), w(0), w(1), w(2),
                s(1), s(2), s(3), w(3), w(4), w(5));
        } else if (profile == "g05_kv_owner") {
            if (w(0).scalar_type() == at::kHalf)
                rpu_launch_g05_kv_owner_linear_fp16_acc32_kernel(s(0), w(0), w(1), s(1), s(2));
            else rpu_launch_g05_kv_owner_linear_w8a16_acc32_kernel(s(0), w(0), w(1), w(2), w(3), s(1), s(2), mode != 0);
        } else if (profile == "pi_kv_owner") {
            rpu_launch_pi05_kv1_pair_owner_w8a16_m50n32x2k1024_kernel(s(0), w(0), w(2), s(1), w(1), w(3), s(2));
        } else if (profile == "pi_kv_owner_db") {
            TORCH_CHECK(rows == 50 && cores == 8, "Pi KV DB probe requires M50/TP8");
            rpu_launch_pi05_kv_owner_db_candidate_kernel(s(0), w(0), w(2), s(1), w(1), w(3), s(2));
        } else if (profile == "pi_nvfp4_geglu_reference") {
            TORCH_CHECK(rows == 50 && cores == 8, "Pi GeGLU reference requires M50/TP8");
            rpu_launch_pi05_denoise_gate_up_geglu_nvfp4_acc16_m50_kernel(s(0), w(0), w(1), s(1), w(2), w(3), s(2), s(3), 0,
                rows, w(4), w(5));
        } else if (profile == "pi_nvfp4_geglu_db" || profile == "pi_nvfp4_geglu_db_tscale") {
            TORCH_CHECK(rows == 50 && cores == 8, "Pi GeGLU DB probe requires M50/TP8");
            rpu_launch_pi05_nvfp4_geglu_db_candidate_kernel(s(0), w(0), w(1), s(1), w(2), w(3), s(2), s(3), 0,
                profile == "pi_nvfp4_geglu_db_tscale");
        } else if (profile == "pi_nvfp4_geglu_db_sspm") {
            TORCH_CHECK(rows == 50 && cores == 8, "Pi GeGLU SSPM probe requires M50/TP8");
            rpu_launch_pi05_nvfp4_geglu_sspm_candidate_kernel(
                s(0), w(0), w(1), s(1), w(2), w(3), s(4), s(5), s(2), s(3), 0);
        } else if (profile == "pi_nvfp4_linear_db_sspm") {
            TORCH_CHECK(mode >= 0 && mode <= 2, "Pi Linear SSPM probe modes are Q/O/Down");
            const int64_t n = mode == 0 ? 2048 : 1024;
            const int64_t k = mode == 0 ? 1024 : mode == 1 ? 2048 : 4096;
            rpu_launch_pi05_nvfp4_sspm_candidate_kernel(
                s(0), w(0), s(1), rows, n, k, mode == 0 ? 1 : 0, cores, w(1), s(3), s(2), 0);
        } else if (profile == "pi_q_kv1_pair_sspm" || profile == "pi_q_kv1_pair_cache_sspm") {
            TORCH_CHECK(cores == 8 && mode >= 0 && mode < 18,
                        "Q+KV1 SSPM conformance requires TP8/layer0..17");
            rpu_launch_pi05_q_kv1_sspm_candidate_kernel(
                s(0), w(0), w(1), s(4), s(5), s(1),
                w(2), w(3), s(2), w(4), w(5), s(3), rows, static_cast<uint16_t>(mode));
            if (profile == "pi_q_kv1_pair_cache_sspm") {
                TORCH_CHECK(rows == 50, "Production Pi KV writer requires M50/PAD64");
                const auto plan = rpu_resolve_kvinsert_segment_plan(
                    544, 50, 64, 8, 8, 256,
                    KV_INSERT_CAP_V16 | KV_INSERT_CAP_PAD16, KvInsertRoute::PAD16_V16);
                // Consume the producer's live SPM directly in this same Graph.
                // The original writer alone converts these typed offsets.
                rpu_launch_pi05_kv1_direct_cache_m50d256p64_kernel(
                    offsets[2], offsets[3], w(6), w(7), w(8), w(9), 531, plan);
            }
        } else if (profile == "pi_nvfp4_linear_reference" || profile == "pi_nvfp4_linear_db" || profile == "pi_nvfp4_linear_db_tscale") {
            TORCH_CHECK(mode >= 0 && mode <= 2, "Pi Linear DB probe modes are Q/O/Down");
            const int64_t n = mode == 0 ? 2048 : 1024;
            const int64_t k = mode == 0 ? 1024 : mode == 1 ? 2048 : 4096;
            if (profile == "pi_nvfp4_linear_reference")
                rpu_launch_pi05_nvfp4_v2_kernel(s(0), w(0), s(1), rows, n, k, mode == 0 ? 1 : 0, cores, w(1), s(2), 0, false);
            else
                rpu_launch_pi05_nvfp4_db_candidate_kernel(s(0), w(0), s(1), rows, n, k,
                    mode == 0 ? 1 : 0, cores, w(1), s(2), 0,
                    profile == "pi_nvfp4_linear_db_tscale");
        } else if (profile == "g05_qgate_w8") {
            rpu_launch_g05_q_gate_linear_w8a16_acc32_kernel(s(0), w(0), w(1), w(2), w(3), s(1), s(2), mode != 0);
        } else if (profile == "g05_gateup_fp16") {
            rpu_launch_g05_gate_up_linear_fp16_acc32_kernel(s(0), w(0), w(1), s(1), s(2));
        } else if (profile == "g05_vision_fc2") {
            rpu_launch_g05_vision_fc2_weight_outer_linear_w8a16_acc16_kernel(s(0), w(0), s(1), s(2), w(1));
        } else if (profile == "hyvla_gateup_fp16") {
            rpu_launch_hyvla_expert_gateup_swiglu_fp16_spm_kernel(s(0), w(0), w(1), s(1), rows);
        } else if (profile == "hyvla_gateup_w8") {
            launch_hyvla_gateup_w8(s(0), s(1), mode ? s(2) : 0,
                mode ? s(3) : 0, mode ? s(4) : 0, rows,
                w(0), w(1), w(2), w(3), mode != 0, false, mode == 2);
        } else if (profile == "hyvla_resident_fp16") {
            rpu_launch_hyvla_gateup_resident_spm_kernel(s(0), s(1), s(2), s(3), s(4), rows);
        } else if (profile == "qwen_gateup") {
            rpu_launch_qwen3vl_4b_decode_gate_up_swiglu_w8a16_spm_kernel(s(0), w(0), w(1), s(1), w(2), w(3));
        } else if (profile == "qwen_o_norm") {
            rpu_launch_qwen3vl_4b_decode_o_ring_norm_spm_kernel(s(0), w(0), w(1), s(1), s(2), s(3), s(4), 1e-6);
        } else if (profile == "lingbot_router") {
            rpu_launch_lingbot2_router_sigmoid_pad_spm_kernel(s(0), w(0), s(1), s(2));
        } else if (profile == "lingbot_fc1") {
            rpu_launch_lingbot2_vision_fc1_gelu_spm_kernel(s(0), w(0), s(1), s(2), w(1));
        } else if (profile == "lingbot_grouped_expert_y4") {
            const auto scale = w(1).defined() ? c10::optional<at::Tensor>(w(1)) : c10::nullopt;
            rpu_launch_grouped_parallel_linear_spm_kernel(s(0), w(0), s(1), scale, s(2), 0,
                128, 768, 32, true, 8, 204, mode, false, true);
        } else if (profile == "lingbot_grouped") {
            const auto scale = w(1).defined() ? c10::optional<at::Tensor>(w(1)) : c10::nullopt;
            rpu_launch_lingbot2_grouped_gateup_routed_spm_kernel(s(0), s(1), w(0), s(2), scale, s(3), mode);
        } else if (profile == "lingbot_grouped_swiglu") {
            const auto scale = w(1).defined() ? c10::optional<at::Tensor>(w(1)) : c10::nullopt;
            rpu_launch_lingbot2_grouped_gateup_routed_swiglu_route_spm_kernel(s(0), s(1), s(2), w(0), s(3), scale, s(4), mode);
        } else if (profile == "lingbot_ordered") {
            const auto scale = w(1).defined() ? c10::optional<at::Tensor>(w(1)) : c10::nullopt;
            rpu_launch_lingbot2_grouped_down_ordered_combine_spm_kernel(s(0), w(0), s(1), scale, s(2), s(3), s(4), s(5), mode);
        } else if (profile == "lingbot_down_shuffle") {
            rpu_launch_lingbot2_grouped_down_w8_scale_shuffle_spm_kernel(s(0), w(0), s(1), w(1), s(2));
        } else if (profile == "lingbot_routed") {
            rpu_launch_lingbot2_routed_col_spm_kernel(s(0), w(0), s(1), s(2), w(1));
        } else if (profile == "wall_action_gateup") {
            WallSortingConfig sorting; sorting.fp16_transfer_mask = 4; sorting.silu_mul = rows == 32;
            rpu_launch_wall_action_gate_up_w8a16_multibase_spm_kernel(s(0), w(0), w(1), s(1), s(2), w(2), w(3),
                mode ? s(3) : 0, mode ? s(4) : 0, sorting, rows);
        } else if (profile == "wall_prefill_gateup") {
            WallSortingConfig sorting; sorting.fp16_transfer_mask = 2;
            rpu_launch_wall_prefill_gate_up_swiglu_w8a16_spm_kernel(s(0), w(0), w(1), s(1), s(2), rows, w(2), w(3), sorting);
        } else if (profile == "gr00t_ff1_resident") {
            rpu_launch_gr00t_ff1_resident_w8_kernel(s(0), s(2), s(1), s(3), w(0));
        } else if (profile == "g05_gateup_resident_fp16") {
            rpu_launch_g05_gateup_resident_fp16_kernel(s(0), s(2), s(1));
        } else if (profile == "g05_gateup_resident_w8") {
            rpu_launch_g05_gateup_resident_w8a16_kernel(s(0), s(2), s(1), w(0), true);
        } else if (profile == "g05_o_resident_w8") {
            rpu_launch_g05_o_resident_w8a16_acc16_kernel(s(0), s(2), s(1), w(0), false);
        } else if (profile == "g05_qgate_fp16") {
            rpu_launch_g05_q_gate_linear_fp16_acc32_kernel(s(0), w(0), w(1), s(1), s(2));
        } else if (profile == "g05_gateup_fp16_acc16") {
            rpu_launch_g05_gate_up_linear_fp16_acc16_kernel(s(0), w(0), w(1), s(1), s(2));
        } else if (profile == "g05_gateup_w8_acc32") {
            rpu_launch_g05_gate_up_linear_w8a16_acc32_kernel(s(0), w(0), w(1), w(2), w(3), s(1), s(2), true);
        } else if (profile == "g05_text_gateup_fp16_acc16") {
            rpu_launch_g05_text_gate_up_linear_fp16_acc16_kernel(s(0), w(0), w(1), s(1), s(2));
        } else if (profile == "g05_text_gateup_fp16_acc32") {
            rpu_launch_g05_text_gate_up_linear_fp16_acc32_kernel(s(0), w(0), w(1), s(1), s(2));
        } else if (profile == "g05_text_gateup_w8_acc32") {
            rpu_launch_g05_text_gate_up_linear_w8a16_acc32_kernel(s(0), w(0), w(1), s(1), s(2), w(2), w(3), true);
        } else if (profile == "g05_vision_qkv_fp16") {
            rpu_launch_g05_vision_qkv_linear_fp16_acc32_kernel(s(0), w(0), w(1), w(2),
                s(1), s(2), s(3), s(4), s(5), s(6));
        } else if (profile == "g05_vision_qkv_w8") {
            rpu_launch_g05_vision_qkv_linear_w8a16_acc32_kernel(s(0), w(0), w(1), w(2),
                s(1), s(2), s(3), s(4), s(5), s(6), w(3), w(4), w(5), true);
        } else if (profile == "lingbot_vision_qkv_w8") {
            rpu_launch_lingbot2_vision_qkv_joint_spm_kernel(s(0), w(0), w(1), w(2),
                s(1), s(2), s(3), s(4), s(5), s(6), w(3), w(4), w(5));
        } else if (profile == "wall_qkv_pair_fp16") {
            WallSortingConfig sorting; sorting.fp16_transfer_mask |= 2u;
            sorting.silu_mul = rows == 224 || rows == 256 || rows == 336 || rows == 384;
            rpu_launch_wall_qkv_w8a16_multibase_spm_kernel(s(0), w(0), w(1), w(2),
                s(1), s(2), s(3), s(4), s(5), s(6), w(3), w(4), w(5), rows, mode != 0, sorting);
        } else if (profile == "wall_vision_row_followup_w8") {
            WallSortingConfig sorting; sorting.vision_row_reuse = true; sorting.silu_mul = true;
            rpu_launch_wall_vision_row_weight_outer_w8a16_spm_kernel(s(0), w(0), s(1),
                rows, w(0).size(0), w(0).size(1), s(2), w(1), sorting);
        } else if (profile == "causal_o_row_followup_w8") {
            rpu_launch_linear_spm_to_spm_acc16_kernel(s(0), w(0), s(1),
                rows, w(0).size(0), w(0).size(1), /*partition=*/0, cores,
                /*bias_spm_addr=*/0, w(1), /*nvfp4_tensor_scale_spm_addr=*/0,
                /*nvfp4_layer_id=*/0, /*force_acc32=*/false, /*prefer_gemv=*/false,
                /*qkv_planar_direct=*/false, /*prefer_row_weight_reuse=*/true);
        } else if (profile == "wall_qkv_w8") {
            WallSortingConfig sorting;
            sorting.silu_mul = rows == 224 || rows == 256 || rows == 336 || rows == 384;
            rpu_launch_wall_qkv_w8a16_multibase_spm_kernel(s(0), w(0), w(1), w(2),
                s(1), s(2), s(3), s(4), s(5), s(6), w(3), w(4), w(5), rows, false, sorting);
        } else if (profile == "wall_vision_qkv_fp16_followup") {
            WallSortingConfig sorting; sorting.fp16_transfer_mask |= 1u;
            rpu_launch_wall_vision_qkv_w8a16_multibase_spm_kernel(s(0), w(0), w(1), w(2),
                s(1), s(2), s(3), s(4), s(5), s(6), w(3), w(4), w(5), sorting);
        } else if (profile == "wall_vision_qkv_w8") {
            WallSortingConfig sorting;
            rpu_launch_wall_vision_qkv_w8a16_multibase_spm_kernel(s(0), w(0), w(1), w(2),
                s(1), s(2), s(3), s(4), s(5), s(6), w(3), w(4), w(5), sorting);
        } else if (profile == "generic_spm_fp16_acc32") {
            rpu_launch_linear_spm_to_spm_kernel(s(0), w(0), s(1), rows, 384, 256, 1, 8, s(2));
        } else if (profile == "wall_vision_row_w8") {
            WallSortingConfig sorting; sorting.vision_row_reuse = true;
            rpu_launch_wall_vision_row_weight_outer_w8a16_spm_kernel(s(0), w(0), s(1), rows, 1280, 1280, s(2), w(1), sorting);
        } else if (profile == "qwen_prefill_swiglu_w8") {
            rpu_launch_qwen3vl_4b_prefill_gate_up_swiglu_w8a16_spm_kernel(s(0), w(0), w(1), s(1), rows, w(2), w(3));
        } else if (profile == "wall_base_prefill_swiglu_fp16") {
            rpu_launch_wall_base_prefill_gate_up_shared_fp16_spm_kernel(s(0), w(0), w(1), s(1), rows);
        } else if (profile == "wall_base_shared_raw") {
            rpu_launch_wall_base_prefill_gate_up_shared_input_spm_kernel(s(0), w(0), w(1), s(1), s(2), rows, w(2), w(3));
        } else if (profile == "wall_base_shared_silu") {
            rpu_launch_wall_base_prefill_gate_up_shared_silu_spm_kernel(s(0), w(0), w(1), s(1), s(2), rows, w(2), w(3));
        } else if (profile == "wall_base_shared_no_up") {
            rpu_launch_wall_base_prefill_gate_up_shared_silu_no_up_spm_kernel(s(0), w(0), w(1), s(1), s(2), rows, w(2), w(3));
        } else if (profile == "lingbot_prefill_swiglu_w8") {
            rpu_launch_lingbot2_prefill_gate_up_shared_a_w8_m224_spm_kernel(s(0), w(0), w(1), s(1), w(2), w(3));
        } else if (profile == "pi_prefill_geglu_w8") {
            rpu_launch_pi05_gate_up_geglu_w8a16_spm_kernel(s(0), w(0), w(1), s(1), w(2), w(3));
        } else if (profile == "pi_denoise_geglu_w8") {
            rpu_launch_pi05_denoise_gate_up_geglu_w8a16_m50_kernel(s(0), w(0), w(1), s(1), w(2), w(3), rows);
        } else if (profile == "pi_w4_geglu_n64") {
            rpu_launch_pi05_denoise_gate_up_geglu_w4a16_pgrp_m50n64_kernel(s(0), w(0), w(1), s(1), w(2), w(3), rows);
        } else if (profile == "wall_down_fp16_c320x2") {
            TORCH_CHECK(w(0).scalar_type() == at::kHalf, "Wall FP16 Down fixture requires Half weights");
            WallSortingConfig sorting; sorting.fp16_transfer_mask = 2;
            rpu_launch_wall_prefill_down_w8a16_c320x2_spm_kernel(s(0), w(0), s(1), s(2), w(1), sorting);
        } else if (profile == "wall_down_c320x2") {
            WallSortingConfig sorting;
            rpu_launch_wall_prefill_down_w8a16_c320x2_spm_kernel(s(0), w(0), s(1), s(2), w(1), sorting);
        } else if (profile == "wall_oproj_c320x2") {
            WallSortingConfig sorting;
            rpu_launch_wall_prefill_oproj_w8a16_c320x2_spm_kernel(s(0), w(0), s(1), s(2), w(1), sorting);
        } else if (profile == "hyvla_header_vlm_gateup") {
            launch_hyvla_vlm_gateup_w8(s(0), s(1), rows, w(0), w(1), w(2), w(3));
        } else if (profile == "hyvla_header_kv") {
            launch_hyvla_kv_grouped_spm(s(0), {w(0), w(1)}, {w(2), w(3)}, {s(1), s(2)}, rows, cores);
        } else if (profile == "hyvla_header_o_resident") {
            launch_hyvla_w8_o_resident_spm(s(0), w(0), s(1), s(2), rows);
        } else if (profile == "hyvla_header_ring_fp16") {
            launch_hyvla_o_ring_postnorm_spm(s(0), w(0), s(1), s(2), s(3), s(4), rows, 1024, 64, 1e-5, cores);
        } else if (profile == "hyvla_header_ring_w8") {
            launch_hyvla_w8_o_ring_postnorm_spm(s(0), w(0), w(1), s(1), s(2), s(3), s(4), rows, 1024, 64, 1e-5, cores);
        } else if (profile == "generic_grouped_uint4") {
            rpu_launch_grouped_parallel_linear_uint4_spm_kernel(s(0), w(0), s(1), w(1), w(2), s(2),
                48, 256, 5, true, cores, 32, rows, mode != 0);
        } else if (profile == "generic_grouped_device_m") {
            rpu_launch_grouped_parallel_linear_spm_kernel(s(0), w(0), s(1), c10::nullopt, s(2), s(3),
                256, 512, 3, mode == 0, cores, rows, 2, false);
        } else if (profile == "generic_grouped_w8_device_m") {
            rpu_launch_grouped_parallel_linear_spm_kernel(s(0), w(0), s(1), w(1), s(2), s(3),
                1024, 256, 3, mode == 0, cores, rows, 1, true);
        } else if (profile.rfind("generic_grouped_followup_", 0) == 0) {
            int64_t n = 256, k = 512, weight_mode = 2;
            bool acc32 = true;
            if (profile == "generic_grouped_followup_fp16_acc32_n80") {
            } else if (profile == "generic_grouped_followup_fp16_acc32_n32") {
                n = 1024; k = 256;
            } else if (profile == "generic_grouped_followup_w8_acc16_n48") {
                weight_mode = 1; acc32 = false;
            } else if (profile == "generic_grouped_followup_w8_acc32_n48") {
                weight_mode = 1;
            } else if (profile == "generic_grouped_followup_fp8_acc32_n112") {
                n = 1024; k = 256; weight_mode = 5;
            } else if (profile == "generic_grouped_followup_fp8_acc16_n48") {
                weight_mode = 5; acc32 = false;
            } else if (profile == "generic_grouped_followup_fp8_acc32_n48") {
                weight_mode = 5;
            } else TORCH_CHECK(false, "unknown grouped followup profile: ", profile);
            const c10::optional<at::Tensor> scale = weight_mode == 2
                ? c10::nullopt : c10::optional<at::Tensor>(w(1));
            rpu_launch_grouped_parallel_linear_spm_kernel(s(0), w(0), s(1), scale, s(2), s(3),
                n, k, 3, mode == 0, cores, rows, weight_mode, acc32);
        } else TORCH_CHECK(false, "unknown production launcher profile: ", profile);
    };
    if (profile == "pi_prefill_kv_owner")
        rpu_validate_pi05_prefill_kv1_pair_owner_w8a16(w(0), w(2), w(1), w(3));
    const bool probe_halfwords = profile == "pi_q_rope" || profile == "wall_action_ring";
    if (probe_halfwords) {
        // Exercise the actual SDK overload contract. These rejected writes
        // never enqueue; the real launcher below resets its own register image.
        auto* probe = GET_KERNEL(KernelId::QWEN3VL_PREFILL_QKV_PLANAR_DIRECT);
        TORCH_CHECK(probe, "register-width probe payload absent");
        for (uint32_t slot : {33u, 129u, 255u}) {
            TORCH_CHECK(probe->set_regs(slot, int32_t{0}) != rhino_lkn::kKernelRegOk,
                        "SDK unexpectedly accepted an odd-index 32-bit register write");
            TORCH_CHECK(probe->set_regs(slot, uint16_t{0}) == rhino_lkn::kKernelRegOk,
                        "SDK rejected a valid 16-bit register write");
        }
        probe->reset_regs();
    }
    RpuKernelGraph graph;
    GraphSignature signature; signature.op_id_str = "production_launcher_" + profile;
    signature.shapes = {rows, cores, mode};
    std::vector<at::Tensor> result;
    std::ostringstream metadata;
    metadata << "{\"sdk_halfword_probe\":" << (probe_halfwords ? "true" : "false") << ",\"invocations\":[";
    uint64_t lifetime = 0, generation = 0;
    std::vector<at::Tensor> initial_caches;
    if (profile == "pi_q_kv1_pair_cache_sspm") {
        TORCH_CHECK(rows == 50, "Production Pi KV writer requires M50/PAD64");
        for (size_t i : {6u, 7u}) initial_caches.push_back(w(i).cpu().clone());
    }
    try {
        for (int iteration = 0; iteration < 4; ++iteration) {
            // Reset the complete cache outside capture so every replay must
            // perform its writes anew. No producer output crosses the host.
            for (size_t i = 0; i < initial_caches.size(); ++i)
                installed_owners[6 + i].copy_(initial_caches[i]);
            const auto images = iteration % 2 ? images_b : images_a;
            for (size_t i = 0; i < offsets.size(); ++i) {
                for (int core = 0; core < cores; ++core) {
                    auto* ptr = static_cast<uint8_t*>(SPM_ALLOC.cpu_ptr(core, offsets[i]));
                    std::memset(ptr - 256, 0xa5, 256);
                    std::memcpy(ptr, images[i].data_ptr<uint8_t>() + core * bytes[i], bytes[i]);
                    std::memset(ptr + bytes[i], 0xa5, 256);
                }
            }
            graph.begin(signature);
            TORCH_CHECK(graph.state() == (iteration == 0 ? RpuKernelGraph::State::RECORDING : RpuKernelGraph::State::REPLAYING),
                "launcher graph recaptured instead of replaying");
            // Keep the actual aligned view; its Tensor retains the larger
            // backing Storage even when installation produced a nonzero offset.
            for (const auto& owner : installed_owners) if (owner.numel()) graph.keep_alive(owner);
            if (resident.defined()) graph.keep_alive(resident);
            emit(); graph.end();
            TORCH_CHECK(graph.state() == RpuKernelGraph::State::BUILT, "launcher graph was not retained");
            const auto stats = graph.debug_stats();
            TORCH_CHECK(stats.kernel_count > 0, "empty launcher graph");
            if (iteration == 0) { lifetime = stats.graph_lifetime_id; generation = stats.build_generation; }
            TORCH_CHECK(stats.graph_lifetime_id == lifetime && stats.build_generation == generation,
                "launcher replay changed Graph identity");
            if (iteration) metadata << ',';
            metadata << "{\"build_generation\":" << generation << ",\"execution_ordinal\":" << stats.execution_ordinal
                << ",\"kernels\":" << stats.kernel_count << ",\"snapshot_contracts\":" << stats.batch_snapshot_effective_contracts << '}';
            for (size_t i = 0; i < offsets.size(); ++i) for (int core = 0; core < cores; ++core) {
                const auto* ptr = static_cast<const uint8_t*>(SPM_ALLOC.cpu_ptr(core, offsets[i]));
                for (size_t byte = 0; byte < 256; ++byte)
                    TORCH_CHECK(ptr[-256 + int(byte)] == 0xa5 && ptr[bytes[i] + byte] == 0xa5,
                        "launcher SPM guard changed: image ", i, " core ", core, " iteration ", iteration);
            }
            // Input, bias, route tables and resident weights must remain
            // unchanged. The online A8 stage and documented collective scratch
            // are the only additional writable regions outside returned outputs.
            for (size_t i = 0; i < offsets.size(); ++i) {
                const auto preload = std::find_if(scale_preloads.begin(), scale_preloads.end(),
                    [&](const auto& entry) { return entry.first == i; });
                if (preload != scale_preloads.end()) {
                    const auto& expected = scale_preload_cpu[preload - scale_preloads.begin()];
                    for (int core = 0; core < cores; ++core)
                        TORCH_CHECK(std::memcmp(SPM_ALLOC.cpu_ptr(core, offsets[i]),
                            expected.data_ptr<uint8_t>() + core * bytes[i], bytes[i]) == 0,
                            "SSPM scale DMA/copy differs or compute modified a scale slab");
                    continue;
                }
                if (resident.defined() && i == 1) {
                    for (int core = 0; core < cores; ++core)
                        TORCH_CHECK(std::memcmp(SPM_ALLOC.cpu_ptr(core, offsets[i]),
                            static_cast<const uint8_t*>(resident_cpu.data_ptr()) + core * bytes[i], bytes[i]) == 0,
                            "production preload differs from its retained compact INT8 owner");
                    continue;
                }
                bool writable = std::find(outputs.begin(), outputs.end(), int64_t(i)) != outputs.end();
                writable = writable || (profile == "pi_a8_online" && i < 2)
                    || (profile == "pi_a8_split" && i == 2)
                    || ((profile == "qwen_o_norm" || profile == "wall_action_ring" || profile == "hyvla_header_ring_fp16" || profile == "hyvla_header_ring_w8") && i == 1)
                    || (profile == "wall_prefill_gateup" && i == 2);
                if (writable) continue;
                for (int core = 0; core < cores; ++core)
                    TORCH_CHECK(std::memcmp(SPM_ALLOC.cpu_ptr(core, offsets[i]),
                        images[i].data_ptr<uint8_t>() + core * bytes[i], bytes[i]) == 0,
                        "launcher immutable SPM image changed: ", i, " core ", core);
            }
            for (const auto i : outputs) {
                auto out = at::empty({cores, int64_t(bytes[i])}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
                for (int core = 0; core < cores; ++core)
                    std::memcpy(out.data_ptr<uint8_t>() + core * bytes[i], SPM_ALLOC.cpu_ptr(core, offsets[i]), bytes[i]);
                result.push_back(std::move(out));
            }
            for (size_t i = 0; i < initial_caches.size(); ++i) {
                auto cache = installed_owners[6 + i].cpu().contiguous();
                auto bytes_out = at::empty({static_cast<int64_t>(cache.nbytes())},
                    at::TensorOptions().dtype(at::kByte).device(at::kCPU));
                std::memcpy(bytes_out.data_ptr<uint8_t>(), cache.data_ptr(), cache.nbytes());
                result.push_back(std::move(bytes_out));
            }
        }
        metadata << "],\"spm_addresses\":[";
        for (size_t i = 0; i < addresses.size(); ++i) { if (i) metadata << ','; metadata << addresses[i]; }
        metadata << "],\"scale_dma_preloads\":[";
        for (size_t i = 0; i < scale_preloads.size(); ++i) {
            const auto [image, owner] = scale_preloads[i];
            if (i) metadata << ',';
            metadata << "{\"image\":" << image << ",\"owner\":" << owner
                     << ",\"spm_address\":" << addresses[image]
                     << ",\"per_core_bytes\":" << bytes[image] << ",\"cores\":" << cores << '}';
        }
        metadata << "],\"ddr_addresses\":[";
        for (size_t i = 0; i < owners.size(); ++i) { if (i) metadata << ','; metadata << (owners[i].numel() ? rhino_lkn::RpuGetDevAddr(owners[i].data_ptr()) : 0); }
        metadata << "]}";
        const auto nodes = graph.dump_nodes_jsonl(), args = graph.dump_kernel_args_jsonl();
        graph.invalidate();
        return {std::move(result), nodes, args, metadata.str()};
    } catch (...) { graph.abort(); throw; }
}

TORCH_LIBRARY_FRAGMENT(rpu, m) {
    m.def("test_production_kernel_lookup_rejections() -> str", TORCH_FN(production_kernel_lookup_rejections));
    m.def("test_production_linear_launchers(str profile, Tensor[] owners, Tensor[] images_a, Tensor[] images_b, int[] outputs, int rows, int cores, int mode=0) -> (Tensor[], str, str, str)",
        TORCH_FN(production_linear_launchers));
}
}  // namespace
