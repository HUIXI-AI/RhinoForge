// Qwen3 all-layers FusedModelBase implementation.
// SDPA and KV-insert use typed SPM offsets; weight updates invalidate model state.
//
// Execution scope:
//   - fuse_lm_head = false by default (lm_head computed by outer HF Qwen3ForCausalLM.lm_head).
//   - final_norm fused into last layer's build_layer_subgraph (SigLIP pattern);
//     no post_fn is configured.
//   - is_causal = true only (Qwen3 fused SDPA is causal-only).
//   - batch_size == 1 except explicitly admitted plain-Qwen3 causal decode;
//     batched prefill is emitted one equal-length sequence/cache slot at a time.
//
// Framework contract: FusedModelBase.

#include "rpu_qwen3_model.h"  // v3::CausalDecoderModel + constants
#include "model_handle_registry.h"
#include "rpu_spm_pipeline.h"

#include <c10/util/ScopeExit.h>

#include <limits>

using namespace at;
using namespace ::rhino_lkn;

namespace v3 {

SpmPipelineComponentLayout CausalDecoderModel::prepare_wall_oss_z1_layout(
    int64_t execution_len,
    int64_t real_len) {
    TORCH_CHECK(!qwen3vl_multiview_text_dry_prepared_ &&
                    !qwen3vl_multiview_text_composite_prepared_ &&
                    !qwen3vl_multiview_text_force_block_twostage_chunk_v2_,
                "Wall-OSS text Z1 cannot replace a prepared multiview dry "
                "layout or live composite; cancel or unprepare it first");
    TORCH_CHECK(!wall_z1_active_,
                "Wall-OSS text Z1: cannot prepare while a lease is active");
    TORCH_CHECK(execution_len == WALL_Z1_CANARY_EXECUTION_LEN &&
                    real_len == WALL_Z1_CANARY_REAL_LEN,
                "Wall-OSS text Z1 canary accepts only execution/real=(192,180), got (",
                execution_len, ",", real_len, ")");
    TORCH_CHECK(hidden_size() == WALL_Z1_CANARY_HIDDEN,
                "Wall-OSS text Z1 canary accepts only hidden size 2048, got ",
                hidden_size());
    TORCH_CHECK(num_layers() > 0 && !layer_weights_.empty() &&
                    final_norm_w_.defined(),
                "Wall-OSS text Z1: decoder weights must be set before prepare");
    TORCH_CHECK(has_mrope_ && mrope_section_.size() == 3 &&
                    mrope_section_[0] == 16 &&
                    mrope_section_[1] == 24 &&
                    mrope_section_[2] == 24 && head_dim() == 128,
                "Wall-OSS text Z1 canary requires exact M-RoPE profile "
                "section=[16,24,24], head_dim=128");
    TORCH_CHECK(deepstack_lang_layers_.empty(),
                "Wall-OSS text Z1 canary does not accept DeepStack");
    TORCH_CHECK(position_ids_keepalive_.defined() &&
                    cos_il_keepalive_.defined() && sin_il_keepalive_.defined() &&
                    position_ids_keepalive_.size(0) >= execution_len &&
                    position_ids_keepalive_.size(1) == 3 &&
                    cos_il_keepalive_.size(0) >= execution_len &&
                    cos_il_keepalive_.size(1) == 64 &&
                    sin_il_keepalive_.sizes() == cos_il_keepalive_.sizes(),
                "Wall-OSS text Z1: M-RoPE keepalives are not initialized for "
                "[192,3] positions and [192,64] cos/sin");
    set_chunk_size_override(0);
    const int64_t resolved_chunk_size = resolve_chunk_size_for_shape(
        execution_len, /*position=*/0, std::nullopt, /*is_causal=*/true);
    TORCH_CHECK(resolved_chunk_size == WALL_Z1_CANARY_CHUNK_SIZE,
                "Wall-OSS text Z1 resident canary requires native chunk size 96, "
                "got ", resolved_chunk_size,
                " for execution_len=", execution_len);

    LayoutContext layout;
    layout.chunk_size = resolved_chunk_size;
    layout.max_kv_seq_len = WALL_Z1_CANARY_EXECUTION_LEN;
    layout.num_layers = num_layers();
    layout.use_attn_mask = false;
    layout.is_causal = true;
    auto prepared = prepare_spm_pipeline_component(layout);
    wall_z1_prepared_execution_len_ = execution_len;
    wall_z1_prepared_real_len_ = real_len;
    wall_z1_prepared_chunk_size_ = resolved_chunk_size;
    wall_z1_position_ids_addr_ = 0;
    wall_z1_rope_cos_il_addr_ = 0;
    wall_z1_rope_sin_il_addr_ = 0;
    wall_z1_inputs_primed_ = false;
    ka_last_position_ = -1;
    ka_last_seq_ = -1;
    ka_last_pos_src_ = nullptr;
    ka_last_pos_src_version_ = -1;
    ka_last_cos_src_ = nullptr;
    ka_last_cos_src_version_ = -1;
    ka_last_sin_src_ = nullptr;
    ka_last_sin_src_version_ = -1;
    return prepared;
}

SpmDense2DSpec CausalDecoderModel::wall_oss_z1_destination_spec(
    int64_t execution_len) const {
    TORCH_CHECK(execution_len == WALL_Z1_CANARY_EXECUTION_LEN,
                "Wall-OSS text Z1 destination requires 192 rows, got ",
                execution_len);
    TORCH_CHECK(hidden_size() == WALL_Z1_CANARY_HIDDEN,
                "Wall-OSS text Z1 destination requires hidden size 2048, got ",
                hidden_size());
    SpmDense2DSpec spec;
    spec.rows = execution_len;
    spec.cols = hidden_size();
    spec.validate();
    return spec;
}

void CausalDecoderModel::validate_wall_oss_z1_mrope_inputs(
    const at::Tensor& position_ids,
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il) const {
    TORCH_CHECK(position_ids.defined() && position_ids.dim() == 2 &&
                    position_ids.size(0) == WALL_Z1_CANARY_EXECUTION_LEN &&
                    position_ids.size(1) == 3 &&
                    position_ids.device().type() == at::kPrivateUse1 &&
                    position_ids.scalar_type() == at::kInt &&
                    position_ids.is_contiguous(),
                "Wall-OSS text Z1 position_ids must be stable contiguous RPU "
                "int32 [192,3]");
    TORCH_CHECK(rope_cos_il.defined() && rope_sin_il.defined() &&
                    rope_cos_il.dim() == 2 &&
                    rope_cos_il.size(0) == WALL_Z1_CANARY_EXECUTION_LEN &&
                    rope_cos_il.size(1) == 64 &&
                    rope_sin_il.sizes() == rope_cos_il.sizes() &&
                    rope_cos_il.device().type() == at::kPrivateUse1 &&
                    rope_sin_il.device().type() == at::kPrivateUse1 &&
                    rope_cos_il.scalar_type() == at::kHalf &&
                    rope_sin_il.scalar_type() == at::kHalf &&
                    rope_cos_il.is_contiguous() && rope_sin_il.is_contiguous(),
                "Wall-OSS text Z1 rope_cos_il/rope_sin_il must be stable "
                "contiguous RPU FP16 [192,64]");
}

void CausalDecoderModel::prime_wall_oss_z1_inputs(
    const at::Tensor& position_ids,
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il,
    int64_t execution_len) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "Wall-OSS text Z1 input prime must run outside Graph capture");
    validate_wall_oss_z1_contract();
    TORCH_CHECK(!wall_z1_active_,
                "Wall-OSS text Z1 input prime must precede lease adoption");
    TORCH_CHECK(execution_len == WALL_Z1_CANARY_EXECUTION_LEN,
                "Wall-OSS text Z1 input prime requires execution_len=192, got ",
                execution_len);
    validate_wall_oss_z1_mrope_inputs(
        position_ids, rope_cos_il, rope_sin_il);
    const uint64_t position_ids_addr =
        RpuGetDevAddr(position_ids.data_ptr<int32_t>());
    const uint64_t rope_cos_il_addr =
        RpuGetDevAddr(rope_cos_il.data_ptr<c10::Half>());
    const uint64_t rope_sin_il_addr =
        RpuGetDevAddr(rope_sin_il.data_ptr<c10::Half>());
    if (wall_z1_inputs_primed_) {
        TORCH_CHECK(wall_z1_position_ids_addr_ == position_ids_addr &&
                        wall_z1_rope_cos_il_addr_ == rope_cos_il_addr &&
                        wall_z1_rope_sin_il_addr_ == rope_sin_il_addr,
                    "Wall-OSS text Z1 input re-prime address drifted");
    }

    auto position_dst =
        position_ids_keepalive_.narrow(0, /*start=*/0, execution_len);
    auto cos_dst = cos_il_keepalive_.narrow(0, /*start=*/0, execution_len);
    auto sin_dst = sin_il_keepalive_.narrow(0, /*start=*/0, execution_len);
    position_dst.copy_(position_ids);
    cos_dst.copy_(rope_cos_il);
    sin_dst.copy_(rope_sin_il);
    rpu_ddr_flush_force_sized(
        position_dst.data_ptr<int32_t>(),
        static_cast<size_t>(execution_len) * 3 * sizeof(int32_t));
    const size_t rope_bytes =
        static_cast<size_t>(execution_len) * 64 * sizeof(c10::Half);
    rpu_ddr_flush_force_sized(cos_dst.data_ptr<c10::Half>(), rope_bytes);
    rpu_ddr_flush_force_sized(sin_dst.data_ptr<c10::Half>(), rope_bytes);

    ka_last_position_ = 0;
    ka_last_seq_ = execution_len;
    ka_last_pos_src_ = position_ids.data_ptr();
    ka_last_pos_src_version_ =
        position_ids.unsafeGetTensorImpl()->version_counter().enabled()
        ? static_cast<int64_t>(position_ids.unsafeGetTensorImpl()
                                   ->version_counter().current_version())
        : -1;
    ka_last_cos_src_ = rope_cos_il.data_ptr();
    ka_last_cos_src_version_ =
        rope_cos_il.unsafeGetTensorImpl()->version_counter().enabled()
        ? static_cast<int64_t>(rope_cos_il.unsafeGetTensorImpl()
                                   ->version_counter().current_version())
        : -1;
    ka_last_sin_src_ = rope_sin_il.data_ptr();
    ka_last_sin_src_version_ =
        rope_sin_il.unsafeGetTensorImpl()->version_counter().enabled()
        ? static_cast<int64_t>(rope_sin_il.unsafeGetTensorImpl()
                                   ->version_counter().current_version())
        : -1;
    wall_z1_position_ids_addr_ = position_ids_addr;
    wall_z1_rope_cos_il_addr_ = rope_cos_il_addr;
    wall_z1_rope_sin_il_addr_ = rope_sin_il_addr;
    TORCH_CHECK(wall_z1_position_ids_addr_ != 0 &&
                    wall_z1_rope_cos_il_addr_ != 0 &&
                    wall_z1_rope_sin_il_addr_ != 0,
                "Wall-OSS text Z1 input prime resolved a zero RPU address");
    wall_z1_inputs_primed_ = true;
}

void CausalDecoderModel::rollback_wall_oss_z1_inputs() {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "Wall-OSS text Z1 input rollback must run outside Graph capture");
    TORCH_CHECK(!wall_z1_active_ && !wall_z1_bound_ && !wall_z1_dispatch_,
                "Wall-OSS text Z1 input rollback requires a primed-only component");
    wall_z1_position_ids_addr_ = 0;
    wall_z1_rope_cos_il_addr_ = 0;
    wall_z1_rope_sin_il_addr_ = 0;
    wall_z1_inputs_primed_ = false;
    ka_last_position_ = -1;
    ka_last_seq_ = -1;
    ka_last_pos_src_ = nullptr;
    ka_last_pos_src_version_ = -1;
    ka_last_cos_src_ = nullptr;
    ka_last_cos_src_version_ = -1;
    ka_last_sin_src_ = nullptr;
    ka_last_sin_src_version_ = -1;
}

void CausalDecoderModel::unprepare_wall_oss_z1_layout() {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "Wall-OSS text Z1 unprepare must run outside Graph capture");
    TORCH_CHECK(!wall_z1_active_ && !wall_z1_bound_ && !wall_z1_dispatch_,
                "Wall-OSS text Z1 unprepare requires no active component lease");
    rollback_wall_oss_z1_inputs();
    wall_z1_prepared_execution_len_ = 0;
    wall_z1_prepared_real_len_ = 0;
    wall_z1_prepared_chunk_size_ = 0;
}

void CausalDecoderModel::validate_wall_oss_z1_contract() const {
    TORCH_CHECK(wall_z1_prepared_execution_len_ ==
                        WALL_Z1_CANARY_EXECUTION_LEN &&
                    wall_z1_prepared_real_len_ == WALL_Z1_CANARY_REAL_LEN,
                "Wall-OSS text Z1: fixed (192,180) layout was not prepared");
    TORCH_CHECK(hidden_size() == WALL_Z1_CANARY_HIDDEN,
                "Wall-OSS text Z1: hidden size drifted after prepare");
    TORCH_CHECK(wall_z1_prepared_chunk_size_ >= 16 &&
                    wall_z1_prepared_chunk_size_ <=
                        wall_z1_prepared_execution_len_ &&
                    wall_z1_prepared_chunk_size_ % 16 == 0,
                "Wall-OSS text Z1: prepared chunk size is invalid: ",
                wall_z1_prepared_chunk_size_);
    TORCH_CHECK(has_mrope_ && mrope_section_.size() == 3 &&
                    mrope_section_[0] == 16 &&
                    mrope_section_[1] == 24 &&
                    mrope_section_[2] == 24 && head_dim() == 128 &&
                    deepstack_lang_layers_.empty(),
                "Wall-OSS text Z1: exact M-RoPE/DeepStack contract drifted "
                "after prepare");
}

void CausalDecoderModel::adopt_wall_oss_z1_layout(
    const SpmPipelineLease& lease,
    const SpmTensorView& scratch) {
    validate_wall_oss_z1_contract();
    TORCH_CHECK(!wall_z1_active_,
                "Wall-OSS text Z1: a lease is already active");
    TORCH_CHECK(wall_z1_inputs_primed_,
                "Wall-OSS text Z1: inputs must be primed before lease adoption");
    adopt_spm_pipeline_component(lease, scratch);
    wall_z1_epoch_ = lease.epoch();
    wall_z1_plan_hash_ = lease.plan_hash();
    wall_z1_active_ = true;
}

void CausalDecoderModel::bind_wall_oss_z1_destination(
    const SpmPipelineLease& lease,
    const SpmPortView& destination) {
    validate_wall_oss_z1_contract();
    TORCH_CHECK(wall_z1_active_ && !wall_z1_bound_,
                "Wall-OSS text Z1: destination binding requires one active, unbound lease");
    TORCH_CHECK(wall_z1_epoch_ == lease.epoch() &&
                    wall_z1_plan_hash_ == lease.plan_hash(),
                "Wall-OSS text Z1: stale destination binding lease");
    const auto expected =
        wall_oss_z1_destination_spec(wall_z1_prepared_execution_len_);
    TORCH_CHECK(destination.spec() == expected,
                "Wall-OSS text Z1: destination port spec mismatch");
    wall_z1_destination_addr_ =
        destination.resolve_physical_addr(/*core=*/0, lease);
    wall_z1_bound_ = true;
}

void CausalDecoderModel::validate_wall_oss_z1_layout(
    const SpmPipelineLease& lease) const {
    validate_wall_oss_z1_contract();
    TORCH_CHECK(wall_z1_active_ && wall_z1_bound_,
                "Wall-OSS text Z1: no complete active binding");
    TORCH_CHECK(wall_z1_epoch_ == lease.epoch() &&
                    wall_z1_plan_hash_ == lease.plan_hash(),
                "Wall-OSS text Z1: stale lease binding");
    TORCH_CHECK(wall_z1_destination_addr_ != 0,
                "Wall-OSS text Z1: destination address is unset");
    validate_spm_pipeline_component(lease);
}

void CausalDecoderModel::clear_wall_oss_z1_layout(
    uint64_t epoch,
    uint64_t plan_hash) {
    TORCH_CHECK(wall_z1_active_,
                "Wall-OSS text Z1: no active lease to clear");
    TORCH_CHECK(wall_z1_epoch_ == epoch &&
                    wall_z1_plan_hash_ == plan_hash,
                "Wall-OSS text Z1: stale clear token");
    TORCH_CHECK(!wall_z1_dispatch_,
                "Wall-OSS text Z1: cannot clear during component dispatch");
    release_spm_pipeline_component(epoch, plan_hash);
    wall_z1_destination_addr_ = 0;
    wall_z1_epoch_ = 0;
    wall_z1_plan_hash_ = 0;
    wall_z1_position_ids_addr_ = 0;
    wall_z1_rope_cos_il_addr_ = 0;
    wall_z1_rope_sin_il_addr_ = 0;
    wall_z1_inputs_primed_ = false;
    ka_last_position_ = -1;
    ka_last_seq_ = -1;
    ka_last_pos_src_ = nullptr;
    ka_last_pos_src_version_ = -1;
    ka_last_cos_src_ = nullptr;
    ka_last_cos_src_version_ = -1;
    ka_last_sin_src_ = nullptr;
    ka_last_sin_src_version_ = -1;
    wall_z1_active_ = false;
    wall_z1_bound_ = false;
}

at::Tensor CausalDecoderModel::forward_wall_oss_z1(
    const at::Tensor& hidden_shape_carrier,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    const at::Tensor& position_ids,
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il,
    uint64_t epoch,
    uint64_t plan_hash) {
    TORCH_CHECK(RpuKernelGraph::has_active(),
                "Wall-OSS text Z1 forward requires one active outer Graph");
    validate_wall_oss_z1_contract();
    TORCH_CHECK(wall_z1_active_ && wall_z1_bound_ &&
                    wall_z1_epoch_ == epoch &&
                    wall_z1_plan_hash_ == plan_hash,
                "Wall-OSS text Z1 forward received a stale epoch/plan hash");
    TORCH_CHECK(wall_z1_inputs_primed_,
                "Wall-OSS text Z1 forward requires primed M-RoPE inputs");
    TORCH_CHECK(!wall_z1_dispatch_,
                "Wall-OSS text Z1 forward is not reentrant");
    TORCH_CHECK(hidden_shape_carrier.defined() &&
                    hidden_shape_carrier.dim() == 3 &&
                    hidden_shape_carrier.size(0) == 1 &&
                    hidden_shape_carrier.size(1) ==
                        WALL_Z1_CANARY_EXECUTION_LEN &&
                    hidden_shape_carrier.size(2) == WALL_Z1_CANARY_HIDDEN &&
                    hidden_shape_carrier.device().type() == at::kPrivateUse1 &&
                    hidden_shape_carrier.scalar_type() == at::kHalf &&
                    hidden_shape_carrier.is_contiguous(),
                "Wall-OSS text Z1 shape carrier must be contiguous RPU FP16 "
                "[1,192,2048]");
    validate_wall_oss_z1_mrope_inputs(
        position_ids, rope_cos_il, rope_sin_il);

    const uint64_t position_ids_addr =
        RpuGetDevAddr(position_ids.data_ptr<int32_t>());
    const uint64_t rope_cos_il_addr =
        RpuGetDevAddr(rope_cos_il.data_ptr<c10::Half>());
    const uint64_t rope_sin_il_addr =
        RpuGetDevAddr(rope_sin_il.data_ptr<c10::Half>());
    TORCH_CHECK(wall_z1_position_ids_addr_ == position_ids_addr &&
                    wall_z1_rope_cos_il_addr_ == rope_cos_il_addr &&
                    wall_z1_rope_sin_il_addr_ == rope_sin_il_addr,
                "Wall-OSS text Z1 position/M-RoPE input address drifted "
                "after Graph-external prime");
    TORCH_CHECK(ka_last_position_ == 0 &&
                    ka_last_seq_ == WALL_Z1_CANARY_EXECUTION_LEN &&
                    ka_last_pos_src_ == position_ids.data_ptr() &&
                    ka_last_cos_src_ == rope_cos_il.data_ptr() &&
                    ka_last_sin_src_ == rope_sin_il.data_ptr(),
                "Wall-OSS text Z1 primed keepalive cache drifted before forward");

    wall_z1_dispatch_ = true;
    try {
        at::Tensor result = forward(
            hidden_shape_carrier, k_caches, v_caches,
            /*attention_mask=*/std::nullopt,
            /*position=*/0,
            /*is_causal=*/true,
            /*position_ids=*/position_ids,
            /*deepstack_dense_visual_embeds=*/std::nullopt,
            /*rope_cos_il=*/rope_cos_il,
            /*rope_sin_il=*/rope_sin_il,
            /*cos_sin_offset=*/-1);
        TORCH_CHECK(get_last_resolved_chunk_size() ==
                        wall_z1_prepared_chunk_size_,
                    "Wall-OSS text Z1 chunk drifted after prepare: expected ",
                    wall_z1_prepared_chunk_size_, ", got ",
                    get_last_resolved_chunk_size());
        wall_z1_dispatch_ = false;
        return result;
    } catch (...) {
        wall_z1_dispatch_ = false;
        throw;
    }
}

uint32_t CausalDecoderModel::wall_z1_layer_input_residual_addr(
    int layer_idx,
    const ChunkInfo& chunk) const {
    if (!wall_z1_bound_ || layer_idx != 0) {
        return addr(0, "residual1");
    }
    TORCH_CHECK(wall_z1_active_ && wall_z1_dispatch_,
                "Wall-OSS text Z1: external residual read outside Z1 dispatch");
    TORCH_CHECK(wall_z1_destination_addr_ != 0,
                "Wall-OSS text Z1: destination address is unset");
    TORCH_CHECK(chunk.offset >= 0 && chunk.len > 0 &&
                    chunk.offset <= wall_z1_prepared_execution_len_ &&
                    chunk.len <=
                        wall_z1_prepared_execution_len_ - chunk.offset,
                "Wall-OSS text Z1: layer-0 chunk escapes destination port");
    const uint64_t byte_offset =
        static_cast<uint64_t>(chunk.offset) *
        static_cast<uint64_t>(hidden_size()) * sizeof(c10::Half);
    TORCH_CHECK(byte_offset <= std::numeric_limits<uint32_t>::max() -
                                   wall_z1_destination_addr_,
                "Wall-OSS text Z1: destination address overflows uint32");
    return wall_z1_destination_addr_ + static_cast<uint32_t>(byte_offset);
}

void CausalDecoderModel::check_wall_oss_z1_destroy_allowed() const {
    TORCH_CHECK(!wall_z1_active_ && !wall_z1_bound_ &&
                    !wall_z1_inputs_primed_ &&
                    wall_z1_prepared_execution_len_ == 0 &&
                    wall_z1_prepared_real_len_ == 0 &&
                    wall_z1_prepared_chunk_size_ == 0,
                "cannot destroy the causal decoder handle while its "
                "Wall-OSS Z1 layout is prepared, primed, or active; "
                "clear the outer GraphCache and unprepare Z1 first");
}

void CausalDecoderModel::validate_qwen3vl_pooler_z1_model_profile() const {
    validate_qwen3vl_pooler_z1_model_profile(
        qwen3vl_pooler_z1_retained_deepstack_count_,
        qwen3vl_pooler_z1_prepared_real_len_);
}

void CausalDecoderModel::validate_qwen3vl_pooler_z1_model_profile(
    int64_t retained_deepstack_count,
    int64_t real_len) const {
    TORCH_CHECK(
        retained_deepstack_count == 0 || retained_deepstack_count == 1 ||
            retained_deepstack_count == 3,
        "Qwen3-VL pooler Z1 retained DeepStack count must be exactly 0, 1, "
        "or 3, got ", retained_deepstack_count);
    const bool qwen3vl_profile =
        num_layers() == QWEN3VL_POOLER_Z1_CANARY_LAYERS &&
        real_len == QWEN3VL_POOLER_Z1_CANARY_REAL_LEN;
    const bool groot_profile =
        num_layers() == QWEN3VL_POOLER_Z1_GROOT_LAYERS &&
        real_len == QWEN3VL_POOLER_Z1_GROOT_REAL_LEN &&
        retained_deepstack_count == 3;
    TORCH_CHECK((qwen3vl_profile || groot_profile) &&
                    hidden_size() == QWEN3VL_POOLER_Z1_CANARY_HIDDEN &&
                    intermediate_size() ==
                        QWEN3VL_POOLER_Z1_CANARY_INTERMEDIATE &&
                    num_q_heads() == QWEN3VL_POOLER_Z1_CANARY_Q_HEADS &&
                    num_kv_heads() == QWEN3VL_POOLER_Z1_CANARY_KV_HEADS &&
                    head_dim() == QWEN3VL_POOLER_Z1_CANARY_HEAD_DIM,
                "Qwen3-VL pooler Z1 requires either the exact Qwen3-VL 2B "
                "profile (layers=28, real=72) or GR00T production profile "
                "(layers=16, real=82, retained_count=3), with hidden=2048, "
                "intermediate=6144, q_heads=16, kv_heads=8, head_dim=128");
    TORCH_CHECK(has_mrope_ && mrope_section_.size() == 3 &&
                    mrope_section_[0] == 24 &&
                    mrope_section_[1] == 20 &&
                    mrope_section_[2] == 20,
                "Qwen3-VL pooler Z1 requires default M-RoPE section "
                "[24,20,20]");
    bool deepstack_profile_ok = retained_deepstack_count == 0
        ? deepstack_lang_layers_.empty()
        : static_cast<int64_t>(deepstack_lang_layers_.size()) ==
              retained_deepstack_count;
    for (int64_t ordinal = 0;
         deepstack_profile_ok && ordinal < retained_deepstack_count;
         ++ordinal) {
        deepstack_profile_ok =
            deepstack_lang_layers_[static_cast<size_t>(ordinal)] ==
            QWEN3VL_POOLER_Z1_DEEPSTACK_LAYERS[
                static_cast<size_t>(ordinal)];
    }
    TORCH_CHECK(
        deepstack_profile_ok,
        "Qwen3-VL pooler Z1 retained DeepStack layers must be exactly {}, "
        "{0}, or {0,1,2} for counts 0, 1, or 3");
    TORCH_CHECK(!nvfp4_ && !has_qkv_bias_ && has_qk_norm_ && !adarms_,
                "Qwen3-VL pooler Z1 requires plain FP16 Qwen3 weights with "
                "QK norm and without NVFP4, QKV bias, or AdaRMS");
    TORCH_CHECK(static_cast<int64_t>(layer_weights_.size()) == num_layers() &&
                    final_norm_w_.defined() && cos_.defined() && sin_.defined(),
                "Qwen3-VL pooler Z1 decoder weights are incomplete");

    auto require_fp16_owner = [](const at::Tensor& tensor,
                                 const char* name,
                                 int64_t layer,
                                 int64_t min_numel) {
        TORCH_CHECK(
            tensor.defined() && tensor.scalar_type() == at::kHalf &&
                tensor.device().type() == at::kPrivateUse1 &&
                tensor.layout() == c10::Layout::Strided &&
                tensor.is_contiguous() && tensor.numel() >= min_numel,
            "Qwen3-VL pooler Z1 requires contiguous FP16 RPU ", name,
            " with at least ", min_numel, " elements at layer ", layer);
    };
    auto require_fp16_vector = [&require_fp16_owner](
                                     const at::Tensor& tensor,
                                     const char* name,
                                     int64_t layer,
                                     int64_t numel) {
        require_fp16_owner(tensor, name, layer, numel);
        TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == numel,
                    "Qwen3-VL pooler Z1 requires ", name, " to be a ",
                    numel, "-element vector at layer ", layer);
    };
    auto require_fp16_matrix_min = [&require_fp16_owner](
                                        const at::Tensor& tensor,
                                        const char* name,
                                        int64_t layer,
                                        int64_t min_numel) {
        require_fp16_owner(tensor, name, layer, min_numel);
        // Swizzling is in-place and the launcher consumes N*K elements, but
        // the physical matrix metadata is not part of this Z1 contract.
        TORCH_CHECK(tensor.dim() == 2,
                    "Qwen3-VL pooler Z1 requires 2D ", name,
                    " at layer ", layer);
    };
    const int64_t h = hidden_size();
    const int64_t q_width = num_q_heads() * head_dim();
    const int64_t kv_width = num_kv_heads() * head_dim();
    const int64_t inter = intermediate_size();
    for (int64_t layer = 0; layer < num_layers(); ++layer) {
        const auto& weights = layer_weights_[static_cast<size_t>(layer)];
        require_fp16_matrix_min(weights.q_w, "q_w", layer, q_width * h);
        require_fp16_matrix_min(weights.k_w, "k_w", layer, kv_width * h);
        require_fp16_matrix_min(weights.v_w, "v_w", layer, kv_width * h);
        require_fp16_matrix_min(weights.o_w, "o_w", layer, h * q_width);
        require_fp16_vector(
            weights.q_norm_w, "q_norm_w", layer, head_dim());
        require_fp16_vector(
            weights.k_norm_w, "k_norm_w", layer, head_dim());
        require_fp16_vector(
            weights.input_norm_w, "input_norm_w", layer, h);
        require_fp16_vector(
            weights.post_norm_w, "post_norm_w", layer, h);
        require_fp16_matrix_min(
            weights.gate_w, "gate_w", layer, inter * h);
        require_fp16_matrix_min(
            weights.up_w, "up_w", layer, inter * h);
        require_fp16_matrix_min(
            weights.down_w, "down_w", layer, h * inter);
        TORCH_CHECK(
            !weights.q_ws.defined() && !weights.k_ws.defined() &&
                !weights.v_ws.defined() && !weights.o_ws.defined() &&
                !weights.gate_ws.defined() && !weights.up_ws.defined() &&
                !weights.down_ws.defined(),
            "Qwen3-VL pooler Z1 rejects quantized linear weights at layer ",
            layer);
    }
    require_fp16_vector(final_norm_w_, "final_norm_w", /*layer=*/-1, h);
    const int64_t rope_cols = head_dim() / 2;
    require_fp16_owner(
        cos_, "cos", /*layer=*/-1,
        QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN * rope_cols);
    require_fp16_owner(
        sin_, "sin", /*layer=*/-1,
        QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN * rope_cols);
    TORCH_CHECK(cos_.dim() == 2 && sin_.sizes() == cos_.sizes() &&
                    cos_.size(0) >=
                        QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN &&
                    cos_.size(1) == rope_cols,
                "Qwen3-VL pooler Z1 cos/sin owners must be matching "
                "[>=128,head_dim/2] tables");
    TORCH_CHECK(position_ids_keepalive_.defined() &&
                    position_ids_keepalive_.dim() == 2 &&
                    position_ids_keepalive_.size(0) >=
                        QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN &&
                    position_ids_keepalive_.size(1) == 3 &&
                    position_ids_keepalive_.scalar_type() == at::kInt &&
                    position_ids_keepalive_.device().type() ==
                        at::kPrivateUse1 &&
                    position_ids_keepalive_.layout() ==
                        c10::Layout::Strided &&
                    position_ids_keepalive_.is_contiguous(),
                "Qwen3-VL pooler Z1 position owner must be contiguous int32 "
                "RPU [>=128,3]");
}

bool CausalDecoderModel::
qwen3vl_pooler_z1_retained_deepstack_bindings_empty() const {
    if (qwen3vl_pooler_z1_deepstack_bound_count_ != 0) {
        return false;
    }
    for (uint32_t addr : qwen3vl_pooler_z1_deepstack_addrs_) {
        if (addr != 0) return false;
    }
    return true;
}

bool CausalDecoderModel::
qwen3vl_pooler_z1_retained_deepstack_bindings_complete() const {
    const int64_t count = qwen3vl_pooler_z1_retained_deepstack_count_;
    if ((count != 0 && count != 1 && count != 3) ||
        qwen3vl_pooler_z1_deepstack_bound_count_ != count) {
        return false;
    }
    for (size_t ordinal = 0;
         ordinal < qwen3vl_pooler_z1_deepstack_addrs_.size(); ++ordinal) {
        const bool requested = ordinal < static_cast<size_t>(count);
        const uint32_t addr = qwen3vl_pooler_z1_deepstack_addrs_[ordinal];
        if (requested != (addr != 0)) return false;
        if (!requested) continue;
        for (size_t prior = 0; prior < ordinal; ++prior) {
            if (addr == qwen3vl_pooler_z1_deepstack_addrs_[prior]) {
                return false;
            }
        }
    }
    return true;
}

SpmPipelineComponentLayout
CausalDecoderModel::prepare_qwen3vl_pooler_z1_layout(
    int64_t execution_len,
    int64_t real_len,
    int64_t retained_deepstack_count) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                    "Qwen3-VL pooler Z1 prepare must run outside Graph capture");
    TORCH_CHECK(!qwen3vl_multiview_text_dry_prepared_ &&
                    !qwen3vl_multiview_text_composite_prepared_ &&
                    !qwen3vl_multiview_text_force_block_twostage_chunk_v2_,
                "Qwen3-VL pooler Z1 cannot replace a prepared multiview dry "
                "layout or live composite; cancel or unprepare it first");
    TORCH_CHECK(qwen3vl_pooler_z1_prepared_execution_len_ == 0 &&
                    !qwen3vl_pooler_z1_active_ &&
                    !qwen3vl_pooler_z1_bound_ &&
                    qwen3vl_pooler_z1_retained_deepstack_count_ == 0 &&
                    qwen3vl_pooler_z1_retained_deepstack_bindings_empty() &&
                    !qwen3vl_pooler_z1_dispatch_,
                "Qwen3-VL pooler Z1 layout is already prepared or active");
    TORCH_CHECK(wall_z1_prepared_execution_len_ == 0 && !wall_z1_active_,
                "Qwen3-VL pooler Z1 cannot share a decoder handle with a "
                "prepared Wall-OSS Z1 component");
    TORCH_CHECK(
        execution_len == QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN &&
            (real_len == QWEN3VL_POOLER_Z1_CANARY_REAL_LEN ||
             real_len == QWEN3VL_POOLER_Z1_GROOT_REAL_LEN),
        "Qwen3-VL pooler Z1 accepts only execution/real=(128,72) or "
        "GR00T (128,82), got (", execution_len, ",", real_len, ")");
    TORCH_CHECK(
        retained_deepstack_count == 0 || retained_deepstack_count == 1 ||
            retained_deepstack_count == 3,
        "Qwen3-VL pooler Z1 prepare retained DeepStack count must be exactly "
        "0, 1, or 3, got ", retained_deepstack_count);
    // The count is cold for the prepared component: profile validation and
    // declare_buffers() must observe the same value.  A failure from here is
    // intentionally fail-closed until unprepare resets the mode.
    qwen3vl_pooler_z1_retained_deepstack_count_ = retained_deepstack_count;
    validate_qwen3vl_pooler_z1_model_profile(
        retained_deepstack_count, real_len);

    set_chunk_size_override(QWEN3VL_POOLER_Z1_CANARY_CHUNK_SIZE);
    LayoutContext layout;
    layout.chunk_size = QWEN3VL_POOLER_Z1_CANARY_CHUNK_SIZE;
    layout.max_kv_seq_len = QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN;
    layout.num_layers = num_layers();
    layout.use_attn_mask = false;
    layout.is_causal = true;
    // declare_buffers() consults the cold retained count so DS1 and DS3 omit
    // ordinary dense-DDR deepstack_scratch from the exact component manifest.
    auto prepared = prepare_spm_pipeline_component(layout);
    TORCH_CHECK(
        prepared.temporary_bytes ==
            QWEN3VL_POOLER_Z1_CANARY_TEMPORARY_BYTES,
        "Qwen3-VL pooler Z1 C128/H2048 temporary layout drifted: expected ",
        QWEN3VL_POOLER_Z1_CANARY_TEMPORARY_BYTES, ", got ",
        prepared.temporary_bytes);

    qwen3vl_pooler_z1_prepared_execution_len_ = execution_len;
    qwen3vl_pooler_z1_prepared_real_len_ = real_len;
    qwen3vl_pooler_z1_prepared_chunk_size_ =
        QWEN3VL_POOLER_Z1_CANARY_CHUNK_SIZE;
    qwen3vl_pooler_z1_destination_addr_ = 0;
    qwen3vl_pooler_z1_deepstack_addrs_.fill(0);
    qwen3vl_pooler_z1_deepstack_bound_count_ = 0;
    qwen3vl_pooler_z1_position_ids_addr_ = 0;
    qwen3vl_pooler_z1_partial_mrope_ = false;
    qwen3vl_pooler_z1_inputs_primed_ = false;
    ka_last_position_ = -1;
    ka_last_seq_ = -1;
    ka_last_pos_src_ = nullptr;
    ka_last_pos_src_version_ = -1;
    ka_last_cos_src_ = nullptr;
    ka_last_cos_src_version_ = -1;
    ka_last_sin_src_ = nullptr;
    ka_last_sin_src_version_ = -1;
    return prepared;
}

SpmDense2DSpec CausalDecoderModel::qwen3vl_pooler_z1_destination_spec(
    int64_t execution_len) const {
    TORCH_CHECK(execution_len ==
                    QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN,
                "Qwen3-VL pooler Z1 destination requires 128 rows, got ",
                execution_len);
    TORCH_CHECK(hidden_size() == QWEN3VL_POOLER_Z1_CANARY_HIDDEN,
                "Qwen3-VL pooler Z1 destination requires hidden size 2048, "
                "got ", hidden_size());
    SpmDense2DSpec spec;
    spec.rows = QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN;
    spec.cols = QWEN3VL_POOLER_Z1_CANARY_HIDDEN;
    spec.validate();
    return spec;
}

SpmDense2DSpec CausalDecoderModel::qwen3vl_pooler_z1_deepstack_spec(
    int64_t ordinal) const {
    TORCH_CHECK(
        ordinal >= 0 &&
            ordinal < qwen3vl_pooler_z1_retained_deepstack_count_,
        "Qwen3-VL pooler Z1 retained DeepStack ordinal ", ordinal,
        " is outside prepared prefix [0, ",
        qwen3vl_pooler_z1_retained_deepstack_count_, ")");
    TORCH_CHECK(hidden_size() == QWEN3VL_POOLER_Z1_CANARY_HIDDEN,
                "Qwen3-VL pooler Z1 retained DeepStack port requires hidden "
                "size 2048, got ", hidden_size());
    SpmDense2DSpec spec;
    spec.rows = QWEN3VL_POOLER_Z1_DS1_ROWS;
    spec.cols = QWEN3VL_POOLER_Z1_CANARY_HIDDEN;
    spec.validate();
    return spec;
}

SpmDense2DSpec CausalDecoderModel::qwen3vl_pooler_z1_deepstack1_spec() const {
    return qwen3vl_pooler_z1_deepstack_spec(/*ordinal=*/0);
}

void CausalDecoderModel::validate_qwen3vl_pooler_z1_position_ids(
    const at::Tensor& position_ids) const {
    TORCH_CHECK(position_ids.defined() && position_ids.dim() == 2 &&
                    position_ids.size(0) ==
                        QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN &&
                    position_ids.size(1) == 3 &&
                    position_ids.device().type() == at::kPrivateUse1 &&
                    position_ids.scalar_type() == at::kInt &&
                    position_ids.is_contiguous(),
                "Qwen3-VL pooler Z1 position_ids must be contiguous RPU "
                "int32 [128,3]");
}

void CausalDecoderModel::validate_qwen3vl_pooler_z1_rope_il(
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il) const {
    TORCH_CHECK(
        qwen3vl_pooler_z1_prepared_real_len_ ==
                QWEN3VL_POOLER_Z1_CANARY_REAL_LEN &&
            qwen3vl_pooler_z1_retained_deepstack_count_ == 3,
        "Qwen3-VL pooler partial M-RoPE is scoped to the generic S72 "
        "retained-DS3 profile; GR00T and the legacy Z1/DS1 profiles remain "
        "on llama_mrope_interleave");
    TORCH_CHECK(
        rope_cos_il.defined() && rope_sin_il.defined() &&
            rope_cos_il.dim() == 2 &&
            rope_cos_il.size(0) ==
                QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN &&
            rope_cos_il.size(1) == head_dim() / 2 &&
            rope_sin_il.sizes() == rope_cos_il.sizes() &&
            rope_cos_il.device().type() == at::kPrivateUse1 &&
            rope_sin_il.device().type() == at::kPrivateUse1 &&
            rope_cos_il.scalar_type() == at::kHalf &&
            rope_sin_il.scalar_type() == at::kHalf &&
            rope_cos_il.is_contiguous() && rope_sin_il.is_contiguous(),
        "Qwen3-VL pooler partial M-RoPE tables must be matching "
        "contiguous RPU FP16 [128,head_dim/2]");
}

void CausalDecoderModel::validate_qwen3vl_pooler_z1_contract() const {
    TORCH_CHECK(qwen3vl_pooler_z1_prepared_execution_len_ ==
                        QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN &&
                    (qwen3vl_pooler_z1_prepared_real_len_ ==
                         QWEN3VL_POOLER_Z1_CANARY_REAL_LEN ||
                     qwen3vl_pooler_z1_prepared_real_len_ ==
                         QWEN3VL_POOLER_Z1_GROOT_REAL_LEN) &&
                    qwen3vl_pooler_z1_prepared_chunk_size_ ==
                        QWEN3VL_POOLER_Z1_CANARY_CHUNK_SIZE,
                "Qwen3-VL pooler Z1 fixed P128/C128 Qwen S72 or GR00T S82 "
                "layout was not prepared");
    validate_qwen3vl_pooler_z1_model_profile();
}

void CausalDecoderModel::prime_qwen3vl_pooler_z1_inputs(
    const at::Tensor& position_ids,
    int64_t execution_len,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "Qwen3-VL pooler Z1 input prime must run outside Graph capture");
    validate_qwen3vl_pooler_z1_contract();
    TORCH_CHECK(!qwen3vl_pooler_z1_active_ &&
                    !qwen3vl_pooler_z1_bound_ &&
                    qwen3vl_pooler_z1_retained_deepstack_bindings_empty() &&
                    !qwen3vl_pooler_z1_dispatch_,
                "Qwen3-VL pooler Z1 input prime must precede lease adoption");
    TORCH_CHECK(execution_len ==
                    QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN,
                "Qwen3-VL pooler Z1 input prime requires execution_len=128, "
                "got ", execution_len);
    validate_qwen3vl_pooler_z1_position_ids(position_ids);
    TORCH_CHECK(
        rope_cos_il.has_value() == rope_sin_il.has_value(),
        "Qwen3-VL pooler Z1 input prime requires both or neither "
        "rope_cos_il/rope_sin_il");
    const bool partial_mrope = rope_cos_il.has_value();
    if (partial_mrope) {
        validate_qwen3vl_pooler_z1_rope_il(*rope_cos_il, *rope_sin_il);
    }
    const uint64_t position_ids_addr =
        RpuGetDevAddr(position_ids.data_ptr<int32_t>());
    TORCH_CHECK(position_ids_addr != 0,
                "Qwen3-VL pooler Z1 input prime resolved a zero RPU address");
    if (qwen3vl_pooler_z1_inputs_primed_) {
        TORCH_CHECK(qwen3vl_pooler_z1_position_ids_addr_ == position_ids_addr,
                    "Qwen3-VL pooler Z1 input re-prime address drifted");
    }

    auto position_dst = position_ids_keepalive_.narrow(
        0, /*start=*/0, QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN);
    position_dst.copy_(position_ids);
    rpu_ddr_flush_force_sized(
        position_dst.data_ptr<int32_t>(),
        static_cast<size_t>(QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN) *
            3 * sizeof(int32_t));

    if (partial_mrope) {
        auto cos_dst = cos_il_keepalive_.narrow(
            0, /*start=*/0, QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN);
        auto sin_dst = sin_il_keepalive_.narrow(
            0, /*start=*/0, QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN);
        cos_dst.copy_(*rope_cos_il);
        sin_dst.copy_(*rope_sin_il);
        const size_t table_bytes =
            static_cast<size_t>(QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN) *
            static_cast<size_t>(head_dim() / 2) * sizeof(c10::Half);
        rpu_ddr_flush_force_sized(
            cos_dst.data_ptr<c10::Half>(), table_bytes);
        rpu_ddr_flush_force_sized(
            sin_dst.data_ptr<c10::Half>(), table_bytes);
        // The Graph consumes model-owned keepalives, not the caller's table
        // addresses.  This makes BUILD/REPLAY independent of temporary Python
        // table storage once the synchronous prime returns.
        ka_last_cos_src_ = cos_dst.data_ptr();
        ka_last_cos_src_version_ =
            cos_dst.unsafeGetTensorImpl()->version_counter().enabled()
            ? static_cast<int64_t>(cos_dst.unsafeGetTensorImpl()
                                       ->version_counter().current_version())
            : -1;
        ka_last_sin_src_ = sin_dst.data_ptr();
        ka_last_sin_src_version_ =
            sin_dst.unsafeGetTensorImpl()->version_counter().enabled()
            ? static_cast<int64_t>(sin_dst.unsafeGetTensorImpl()
                                       ->version_counter().current_version())
            : -1;
    } else {
        ka_last_cos_src_ = nullptr;
        ka_last_cos_src_version_ = -1;
        ka_last_sin_src_ = nullptr;
        ka_last_sin_src_version_ = -1;
    }

    ka_last_position_ = 0;
    ka_last_seq_ = QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN;
    ka_last_pos_src_ = position_ids.data_ptr();
    ka_last_pos_src_version_ =
        position_ids.unsafeGetTensorImpl()->version_counter().enabled()
        ? static_cast<int64_t>(position_ids.unsafeGetTensorImpl()
                                   ->version_counter().current_version())
        : -1;
    partial_mrope_active_ = partial_mrope;
    qwen3vl_pooler_z1_position_ids_addr_ = position_ids_addr;
    qwen3vl_pooler_z1_partial_mrope_ = partial_mrope;
    qwen3vl_pooler_z1_inputs_primed_ = true;
}

void CausalDecoderModel::rollback_qwen3vl_pooler_z1_inputs() {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "Qwen3-VL pooler Z1 input rollback must run outside Graph capture");
    TORCH_CHECK(!qwen3vl_pooler_z1_active_ &&
                    !qwen3vl_pooler_z1_bound_ &&
                    qwen3vl_pooler_z1_retained_deepstack_bindings_empty() &&
                    !qwen3vl_pooler_z1_dispatch_,
                "Qwen3-VL pooler Z1 input rollback requires a primed-only "
                "component");
    qwen3vl_pooler_z1_position_ids_addr_ = 0;
    qwen3vl_pooler_z1_partial_mrope_ = false;
    qwen3vl_pooler_z1_inputs_primed_ = false;
    partial_mrope_active_ = false;
    ka_last_position_ = -1;
    ka_last_seq_ = -1;
    ka_last_pos_src_ = nullptr;
    ka_last_pos_src_version_ = -1;
    ka_last_cos_src_ = nullptr;
    ka_last_cos_src_version_ = -1;
    ka_last_sin_src_ = nullptr;
    ka_last_sin_src_version_ = -1;
}

void CausalDecoderModel::unprepare_qwen3vl_pooler_z1_layout() {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "Qwen3-VL pooler Z1 unprepare must run outside Graph capture");
    TORCH_CHECK(!qwen3vl_pooler_z1_active_ &&
                    !qwen3vl_pooler_z1_bound_ &&
                    qwen3vl_pooler_z1_retained_deepstack_bindings_empty() &&
                    !qwen3vl_pooler_z1_dispatch_,
                "Qwen3-VL pooler Z1 unprepare requires no active component "
                "lease");
    rollback_qwen3vl_pooler_z1_inputs();
    qwen3vl_pooler_z1_prepared_execution_len_ = 0;
    qwen3vl_pooler_z1_prepared_real_len_ = 0;
    qwen3vl_pooler_z1_prepared_chunk_size_ = 0;
    qwen3vl_pooler_z1_deepstack_addrs_.fill(0);
    qwen3vl_pooler_z1_retained_deepstack_count_ = 0;
    qwen3vl_pooler_z1_deepstack_bound_count_ = 0;
    qwen3vl_pooler_z1_output_owner_ = at::Tensor{};
}

void CausalDecoderModel::adopt_qwen3vl_pooler_z1_layout(
    const SpmPipelineLease& lease,
    const SpmTensorView& scratch) {
    validate_qwen3vl_pooler_z1_contract();
    TORCH_CHECK(!qwen3vl_pooler_z1_active_,
                "Qwen3-VL pooler Z1 lease is already active");
    TORCH_CHECK(qwen3vl_pooler_z1_inputs_primed_,
                "Qwen3-VL pooler Z1 inputs must be primed before lease "
                "adoption");
    adopt_spm_pipeline_component(lease, scratch);
    qwen3vl_pooler_z1_epoch_ = lease.epoch();
    qwen3vl_pooler_z1_plan_hash_ = lease.plan_hash();
    qwen3vl_pooler_z1_active_ = true;
}

void CausalDecoderModel::bind_qwen3vl_pooler_z1_destination(
    const SpmPipelineLease& lease,
    const SpmPortView& destination) {
    validate_qwen3vl_pooler_z1_contract();
    TORCH_CHECK(qwen3vl_pooler_z1_active_ &&
                    !qwen3vl_pooler_z1_bound_,
                "Qwen3-VL pooler Z1 destination binding requires one active, "
                "unbound lease");
    TORCH_CHECK(qwen3vl_pooler_z1_epoch_ == lease.epoch() &&
                    qwen3vl_pooler_z1_plan_hash_ == lease.plan_hash(),
                "Qwen3-VL pooler Z1 stale destination binding lease");
    const auto expected = qwen3vl_pooler_z1_destination_spec(
        qwen3vl_pooler_z1_prepared_execution_len_);
    TORCH_CHECK(destination.spec() == expected,
                "Qwen3-VL pooler Z1 destination port spec mismatch");
    qwen3vl_pooler_z1_destination_addr_ =
        destination.resolve_physical_addr(/*core=*/0, lease);
    TORCH_CHECK(qwen3vl_pooler_z1_destination_addr_ != 0,
                "Qwen3-VL pooler Z1 destination resolved a zero address");
    qwen3vl_pooler_z1_bound_ = true;
}

void CausalDecoderModel::bind_qwen3vl_pooler_z1_deepstack(
    const SpmPipelineLease& lease,
    const SpmPortView& deepstack,
    int64_t ordinal) {
    validate_qwen3vl_pooler_z1_contract();
    TORCH_CHECK(
        qwen3vl_pooler_z1_retained_deepstack_count_ > 0,
        "Qwen3-VL pooler Z1 retained DeepStack binding requires a retained "
        "prepare mode");
    TORCH_CHECK(qwen3vl_pooler_z1_active_ &&
                    ordinal == qwen3vl_pooler_z1_deepstack_bound_count_ &&
                    ordinal >= 0 &&
                    ordinal < qwen3vl_pooler_z1_retained_deepstack_count_,
                "Qwen3-VL pooler Z1 retained DeepStack ports must bind as an "
                "exact prefix; expected ordinal ",
                qwen3vl_pooler_z1_deepstack_bound_count_, ", got ", ordinal);
    TORCH_CHECK(qwen3vl_pooler_z1_epoch_ == lease.epoch() &&
                    qwen3vl_pooler_z1_plan_hash_ == lease.plan_hash(),
                "Qwen3-VL pooler Z1 stale retained DeepStack binding lease");
    const auto expected = qwen3vl_pooler_z1_deepstack_spec(ordinal);
    TORCH_CHECK(deepstack.spec() == expected,
                "Qwen3-VL pooler Z1 retained DeepStack port spec mismatch at "
                "ordinal ", ordinal);
    const uint32_t addr =
        deepstack.resolve_physical_addr(/*core=*/0, lease);
    TORCH_CHECK(addr != 0,
                "Qwen3-VL pooler Z1 retained DeepStack port resolved a zero "
                "address at ordinal ", ordinal);
    for (int64_t prior = 0; prior < ordinal; ++prior) {
        TORCH_CHECK(
            qwen3vl_pooler_z1_deepstack_addrs_[static_cast<size_t>(prior)] !=
                addr,
            "Qwen3-VL pooler Z1 retained DeepStack ports alias at ordinals ",
            prior, " and ", ordinal);
    }
    qwen3vl_pooler_z1_deepstack_addrs_[static_cast<size_t>(ordinal)] = addr;
    ++qwen3vl_pooler_z1_deepstack_bound_count_;
}

void CausalDecoderModel::bind_qwen3vl_pooler_z1_deepstack1(
    const SpmPipelineLease& lease,
    const SpmPortView& deepstack1) {
    bind_qwen3vl_pooler_z1_deepstack(
        lease, deepstack1, /*ordinal=*/0);
}

void CausalDecoderModel::validate_qwen3vl_pooler_z1_layout(
    const SpmPipelineLease& lease) const {
    validate_qwen3vl_pooler_z1_contract();
    TORCH_CHECK(qwen3vl_pooler_z1_active_ && qwen3vl_pooler_z1_bound_,
                "Qwen3-VL pooler Z1 has no complete active binding");
    TORCH_CHECK(qwen3vl_pooler_z1_epoch_ == lease.epoch() &&
                    qwen3vl_pooler_z1_plan_hash_ == lease.plan_hash(),
                "Qwen3-VL pooler Z1 stale lease binding");
    TORCH_CHECK(qwen3vl_pooler_z1_destination_addr_ != 0,
                "Qwen3-VL pooler Z1 destination address is unset");
    TORCH_CHECK(qwen3vl_pooler_z1_retained_deepstack_bindings_complete(),
                "Qwen3-VL pooler Z1 retained DeepStack binding prefix is "
                "incomplete, aliased, or leaked into pooler-only mode");
    validate_spm_pipeline_component(lease);
}

void CausalDecoderModel::clear_qwen3vl_pooler_z1_layout(
    uint64_t epoch,
    uint64_t plan_hash) {
    TORCH_CHECK(qwen3vl_pooler_z1_active_,
                "Qwen3-VL pooler Z1 has no active lease to clear");
    TORCH_CHECK(qwen3vl_pooler_z1_epoch_ == epoch &&
                    qwen3vl_pooler_z1_plan_hash_ == plan_hash,
                "Qwen3-VL pooler Z1 stale clear token");
    TORCH_CHECK(!qwen3vl_pooler_z1_dispatch_,
                "Qwen3-VL pooler Z1 cannot clear during component dispatch");
    release_spm_pipeline_component(epoch, plan_hash);
    qwen3vl_pooler_z1_destination_addr_ = 0;
    qwen3vl_pooler_z1_deepstack_addrs_.fill(0);
    qwen3vl_pooler_z1_epoch_ = 0;
    qwen3vl_pooler_z1_plan_hash_ = 0;
    qwen3vl_pooler_z1_position_ids_addr_ = 0;
    qwen3vl_pooler_z1_partial_mrope_ = false;
    qwen3vl_pooler_z1_inputs_primed_ = false;
    partial_mrope_active_ = false;
    ka_last_position_ = -1;
    ka_last_seq_ = -1;
    ka_last_pos_src_ = nullptr;
    ka_last_pos_src_version_ = -1;
    ka_last_cos_src_ = nullptr;
    ka_last_cos_src_version_ = -1;
    ka_last_sin_src_ = nullptr;
    ka_last_sin_src_version_ = -1;
    qwen3vl_pooler_z1_active_ = false;
    qwen3vl_pooler_z1_bound_ = false;
    qwen3vl_pooler_z1_deepstack_bound_count_ = 0;
}

at::Tensor CausalDecoderModel::forward_qwen3vl_pooler_z1(
    const at::Tensor& hidden_shape_carrier,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    const at::Tensor& position_ids,
    uint64_t epoch,
    uint64_t plan_hash) {
    TORCH_CHECK(RpuKernelGraph::has_active(),
                "Qwen3-VL pooler Z1 forward requires one active outer Graph");
    validate_qwen3vl_pooler_z1_contract();
    TORCH_CHECK(qwen3vl_pooler_z1_active_ &&
                    qwen3vl_pooler_z1_bound_ &&
                    qwen3vl_pooler_z1_epoch_ == epoch &&
                    qwen3vl_pooler_z1_plan_hash_ == plan_hash,
                "Qwen3-VL pooler Z1 forward received a stale epoch/plan hash");
    TORCH_CHECK(qwen3vl_pooler_z1_retained_deepstack_bindings_complete(),
                "Qwen3-VL pooler Z1 forward requires its exact retained "
                "DeepStack binding prefix");
    TORCH_CHECK(qwen3vl_pooler_z1_inputs_primed_,
                "Qwen3-VL pooler Z1 forward requires primed position_ids");
    TORCH_CHECK(!qwen3vl_pooler_z1_dispatch_,
                "Qwen3-VL pooler Z1 forward is not reentrant");
    TORCH_CHECK(hidden_shape_carrier.defined() &&
                    hidden_shape_carrier.dim() == 3 &&
                    hidden_shape_carrier.size(0) == 1 &&
                    hidden_shape_carrier.size(1) ==
                        QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN &&
                    hidden_shape_carrier.size(2) ==
                        QWEN3VL_POOLER_Z1_CANARY_HIDDEN &&
                    hidden_shape_carrier.device().type() == at::kPrivateUse1 &&
                    hidden_shape_carrier.scalar_type() == at::kHalf &&
                    hidden_shape_carrier.is_contiguous(),
                "Qwen3-VL pooler Z1 shape carrier must be contiguous RPU "
                "FP16 [1,128,2048]");
    validate_qwen3vl_pooler_z1_position_ids(position_ids);
    const uint64_t position_ids_addr =
        RpuGetDevAddr(position_ids.data_ptr<int32_t>());
    TORCH_CHECK(qwen3vl_pooler_z1_position_ids_addr_ == position_ids_addr,
                "Qwen3-VL pooler Z1 position_ids address drifted after "
                "Graph-external prime");
    TORCH_CHECK(ka_last_position_ == 0 &&
                    ka_last_seq_ ==
                        QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN &&
                    ka_last_pos_src_ == position_ids.data_ptr() &&
                    (qwen3vl_pooler_z1_partial_mrope_ ==
                        partial_mrope_active_) &&
                    (qwen3vl_pooler_z1_partial_mrope_
                         ? (ka_last_cos_src_ ==
                                cos_il_keepalive_.data_ptr<c10::Half>() &&
                            ka_last_sin_src_ ==
                                sin_il_keepalive_.data_ptr<c10::Half>())
                         : (ka_last_cos_src_ == nullptr &&
                            ka_last_sin_src_ == nullptr)),
                "Qwen3-VL pooler Z1 primed keepalive cache drifted before "
                "forward");

    std::optional<at::Tensor> rope_cos_il = std::nullopt;
    std::optional<at::Tensor> rope_sin_il = std::nullopt;
    if (qwen3vl_pooler_z1_partial_mrope_) {
        rope_cos_il = cos_il_keepalive_.narrow(
            0, /*start=*/0, QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN);
        rope_sin_il = sin_il_keepalive_.narrow(
            0, /*start=*/0, QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN);
    }

    qwen3vl_pooler_z1_dispatch_ = true;
    try {
        at::Tensor result = forward(
            hidden_shape_carrier, k_caches, v_caches,
            /*attention_mask=*/std::nullopt,
            /*position=*/0,
            /*is_causal=*/true,
            /*position_ids=*/position_ids,
            /*deepstack_dense_visual_embeds=*/std::nullopt,
            /*rope_cos_il=*/rope_cos_il,
            /*rope_sin_il=*/rope_sin_il,
            /*cos_sin_offset=*/-1);
        TORCH_CHECK(get_last_resolved_chunk_size() ==
                        QWEN3VL_POOLER_Z1_CANARY_CHUNK_SIZE,
                    "Qwen3-VL pooler Z1 chunk drifted after prepare: expected ",
                    QWEN3VL_POOLER_Z1_CANARY_CHUNK_SIZE, ", got ",
                    get_last_resolved_chunk_size());
        const at::Tensor& registry_output = output_tensor();
        TORCH_CHECK(result.defined() && registry_output.defined() &&
                        result.is_same(registry_output),
                    "Qwen3-VL pooler Z1 forward did not return the current "
                    "registry-stable output");
        if (!qwen3vl_pooler_z1_output_owner_.defined()) {
            qwen3vl_pooler_z1_output_owner_ = result;
        }
        TORCH_CHECK(
            qwen3vl_pooler_z1_output_owner_.is_same(registry_output) &&
                qwen3vl_pooler_z1_output_owner_.storage()
                        .unsafeGetStorageImpl() ==
                    registry_output.storage().unsafeGetStorageImpl() &&
                qwen3vl_pooler_z1_output_owner_.data_ptr() ==
                    registry_output.data_ptr(),
            "Qwen3-VL pooler Z1 registry output identity drifted");
        qwen3vl_pooler_z1_dispatch_ = false;
        return result;
    } catch (...) {
        qwen3vl_pooler_z1_dispatch_ = false;
        throw;
    }
}

void CausalDecoderModel::stage_qwen3vl_pooler_z1_outer_fast_component(
    GraphKernelRegisterCensusGuard& guard,
    at::TensorList k_caches,
    at::TensorList v_caches) const {
    validate_qwen3vl_pooler_z1_contract();
    TORCH_CHECK(qwen3vl_pooler_z1_active_ &&
                    qwen3vl_pooler_z1_bound_ &&
                    !qwen3vl_pooler_z1_dispatch_ &&
                    qwen3vl_pooler_z1_epoch_ != 0 &&
                    qwen3vl_pooler_z1_plan_hash_ != 0 &&
                    qwen3vl_pooler_z1_destination_addr_ != 0 &&
                    qwen3vl_pooler_z1_retained_deepstack_bindings_complete(),
                "Qwen3-VL pooler Z1 outer-fast component is not active and "
                "bound");
    TORCH_CHECK(qwen3vl_pooler_z1_inputs_primed_ &&
                    qwen3vl_pooler_z1_position_ids_addr_ != 0 &&
                    ka_last_position_ == 0 &&
                    ka_last_seq_ ==
                        QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN &&
                    ka_last_pos_src_ != nullptr &&
                    (qwen3vl_pooler_z1_partial_mrope_ ==
                        partial_mrope_active_) &&
                    (qwen3vl_pooler_z1_partial_mrope_
                         ? (ka_last_cos_src_ ==
                                cos_il_keepalive_.data_ptr<c10::Half>() &&
                            ka_last_sin_src_ ==
                                sin_il_keepalive_.data_ptr<c10::Half>())
                         : (ka_last_cos_src_ == nullptr &&
                            ka_last_sin_src_ == nullptr)),
                "Qwen3-VL pooler Z1 outer-fast position prime is stale");
    stage_spm_outer_fast_component(guard, k_caches, v_caches);
}

at::Tensor CausalDecoderModel::qwen3vl_pooler_z1_outer_fast_output() const {
    TORCH_CHECK(RpuKernelGraph::has_active(),
                "Qwen3-VL pooler Z1 outer-fast output requires one active "
                "outer Graph");
    validate_qwen3vl_pooler_z1_contract();
    TORCH_CHECK(qwen3vl_pooler_z1_active_ &&
                    qwen3vl_pooler_z1_bound_ &&
                    !qwen3vl_pooler_z1_dispatch_ &&
                    qwen3vl_pooler_z1_inputs_primed_ &&
                    qwen3vl_pooler_z1_retained_deepstack_bindings_complete(),
                "Qwen3-VL pooler Z1 outer-fast output requires one active, "
                "bound component");
    const at::Tensor& current = output_tensor();
    TORCH_CHECK(
        current.defined() && current.dim() == 3 && current.size(0) == 1 &&
            current.size(1) == QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN &&
            current.size(2) == QWEN3VL_POOLER_Z1_CANARY_HIDDEN &&
            current.scalar_type() == at::kHalf &&
            current.device().type() == at::kPrivateUse1 &&
            current.layout() == c10::Layout::Strided &&
            current.is_contiguous(),
        "Qwen3-VL pooler Z1 outer-fast registry output shape/device drifted");
    TORCH_CHECK(
        qwen3vl_pooler_z1_output_owner_.defined() &&
            qwen3vl_pooler_z1_output_owner_.is_same(current) &&
            qwen3vl_pooler_z1_output_owner_.storage()
                    .unsafeGetStorageImpl() ==
                current.storage().unsafeGetStorageImpl() &&
            qwen3vl_pooler_z1_output_owner_.data_ptr() == current.data_ptr(),
        "Qwen3-VL pooler Z1 outer-fast registry output identity drifted");
    return current;
}

void CausalDecoderModel::emit_qwen3vl_pooler_z1_retained_deepstack_add(
    int layer_idx,
    const ChunkInfo& chunk,
    uint32_t residual_addr) {
    int64_t ordinal = -1;
    for (int64_t candidate = 0;
         candidate < qwen3vl_pooler_z1_retained_deepstack_count_;
         ++candidate) {
        if (layer_idx == QWEN3VL_POOLER_Z1_DEEPSTACK_LAYERS[
                             static_cast<size_t>(candidate)]) {
            ordinal = candidate;
            break;
        }
    }
    if (ordinal < 0) return;
    TORCH_CHECK(qwen3vl_pooler_z1_retained_deepstack_bindings_complete() &&
                    qwen3vl_pooler_z1_active_ &&
                    qwen3vl_pooler_z1_dispatch_ &&
                    qwen3vl_pooler_z1_deepstack_addrs_[
                        static_cast<size_t>(ordinal)] != 0,
                "Qwen3-VL pooler Z1 retained DeepStack add requires one "
                "complete active port prefix");
    TORCH_CHECK(chunk.idx == 0 && chunk.offset == 0 &&
                    chunk.len == QWEN3VL_POOLER_Z1_CANARY_CHUNK_SIZE,
                "Qwen3-VL pooler Z1 retained DeepStack add requires the exact "
                "C128 chunk");
    TORCH_CHECK(residual_addr != 0,
                "Qwen3-VL pooler Z1 retained DeepStack residual address is "
                "zero");
    const uint64_t row_offset =
        static_cast<uint64_t>(QWEN3VL_POOLER_Z1_DS1_ROW_BEGIN) *
        static_cast<uint64_t>(QWEN3VL_POOLER_Z1_CANARY_HIDDEN) *
        sizeof(c10::Half);
    TORCH_CHECK(row_offset <= std::numeric_limits<uint32_t>::max() -
                                  residual_addr,
                "Qwen3-VL pooler Z1 retained DeepStack residual row address "
                "overflows uint32");
    const uint32_t visual_residual_addr =
        residual_addr + static_cast<uint32_t>(row_offset);
    rpu_launch_eltwise_binary_spm_kernel(
        visual_residual_addr,
        qwen3vl_pooler_z1_deepstack_addrs_[static_cast<size_t>(ordinal)],
        visual_residual_addr,
        QWEN3VL_POOLER_Z1_DS1_ROWS *
            QWEN3VL_POOLER_Z1_CANARY_HIDDEN,
        ValuOpType::ADD,
        c10::Half(1.0f),
        NUM_CORES);
}

uint32_t CausalDecoderModel::qwen3vl_pooler_z1_layer_input_residual_addr(
    int layer_idx,
    const ChunkInfo& chunk) const {
    if (!qwen3vl_pooler_z1_bound_ || layer_idx != 0) {
        return addr(0, "residual1");
    }
    TORCH_CHECK(qwen3vl_pooler_z1_active_ &&
                    qwen3vl_pooler_z1_dispatch_,
                "Qwen3-VL pooler Z1 external residual read outside Z1 "
                "dispatch");
    TORCH_CHECK(qwen3vl_pooler_z1_destination_addr_ != 0,
                "Qwen3-VL pooler Z1 destination address is unset");
    TORCH_CHECK(chunk.offset >= 0 && chunk.len > 0 &&
                    chunk.offset <=
                        qwen3vl_pooler_z1_prepared_execution_len_ &&
                    chunk.len <=
                        qwen3vl_pooler_z1_prepared_execution_len_ -
                            chunk.offset,
                "Qwen3-VL pooler Z1 layer-0 chunk escapes destination port");
    const uint64_t byte_offset =
        static_cast<uint64_t>(chunk.offset) *
        static_cast<uint64_t>(QWEN3VL_POOLER_Z1_CANARY_HIDDEN) *
        sizeof(c10::Half);
    TORCH_CHECK(
        byte_offset <= std::numeric_limits<uint32_t>::max() -
                           qwen3vl_pooler_z1_destination_addr_,
        "Qwen3-VL pooler Z1 destination address overflows uint32");
    return qwen3vl_pooler_z1_destination_addr_ +
        static_cast<uint32_t>(byte_offset);
}

void CausalDecoderModel::validate_qwen3vl_multiview_text_model_profile() const {
    TORCH_CHECK(
        num_layers() == QWEN3VL_MULTIVIEW_TEXT_LAYERS &&
            hidden_size() == QWEN3VL_MULTIVIEW_TEXT_HIDDEN &&
            intermediate_size() == QWEN3VL_MULTIVIEW_TEXT_INTERMEDIATE &&
            num_q_heads() == QWEN3VL_MULTIVIEW_TEXT_Q_HEADS &&
            num_kv_heads() == QWEN3VL_MULTIVIEW_TEXT_KV_HEADS &&
            head_dim() == QWEN3VL_MULTIVIEW_TEXT_HEAD_DIM,
        "Qwen3-VL multiview Text composite admits only the exact "
        "LingBot2 S225/L36/H2560/I9728/Q32/KV8/D128 profile");
    TORCH_CHECK(
        get_chunk_size_override() == 0 &&
            configured_chunk_size_cap_ == 0 && !equal_two_prefill_,
        "Qwen3-VL multiview Text composite requires the native auto-chunk "
        "policy with per-handle override/cap and "
        "equal-two disabled");
    TORCH_CHECK(
        has_mrope_ && mrope_section_.size() == 3 &&
            mrope_section_[0] == 24 && mrope_section_[1] == 20 &&
            mrope_section_[2] == 20 &&
            deepstack_lang_layers_ == std::vector<int64_t>({0, 1, 2}),
        "Qwen3-VL multiview Text composite requires M-RoPE [24,20,20] "
        "and DeepStack consumers [0,1,2]");
    TORCH_CHECK(
        !nvfp4_ && !has_qkv_bias_ && has_qk_norm_ && !adarms_ &&
            static_cast<int64_t>(layer_weights_.size()) == num_layers() &&
            final_norm_w_.defined() && cos_.defined() && sin_.defined() &&
            position_ids_keepalive_.defined(),
        "Qwen3-VL multiview Text composite requires complete supported "
        "Qwen3 weights, M-RoPE tables, and position keepalive");
}

void CausalDecoderModel::validate_qwen3vl_multiview_text_position_ids(
    const at::Tensor& position_ids) const {
    TORCH_CHECK(
        position_ids.defined() && position_ids.dim() == 2 &&
            position_ids.size(0) == QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN &&
            position_ids.size(1) == 3 &&
            position_ids.device().type() == at::kPrivateUse1 &&
            position_ids.scalar_type() == at::kInt &&
            position_ids.is_contiguous(),
        "Qwen3-VL multiview Text position_ids must be contiguous RPU "
        "int32 [225,3]");
}

void CausalDecoderModel::validate_qwen3vl_multiview_text_rope_il(
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il) const {
    TORCH_CHECK(
        rope_cos_il.defined() && rope_sin_il.defined() &&
            rope_cos_il.dim() == 2 &&
            rope_cos_il.size(0) == QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN &&
            rope_cos_il.size(1) == QWEN3VL_MULTIVIEW_TEXT_HEAD_DIM / 2 &&
            rope_sin_il.sizes() == rope_cos_il.sizes() &&
            rope_cos_il.device().type() == at::kPrivateUse1 &&
            rope_sin_il.device().type() == at::kPrivateUse1 &&
            rope_cos_il.scalar_type() == at::kHalf &&
            rope_sin_il.scalar_type() == at::kHalf &&
            rope_cos_il.is_contiguous() && rope_sin_il.is_contiguous(),
        "Qwen3-VL multiview Text rope_cos_il/rope_sin_il must be matching "
        "contiguous RPU FP16 [225,64]");
}

void CausalDecoderModel::validate_qwen3vl_multiview_text_contract() const {
    TORCH_CHECK(
        qwen3vl_multiview_text_composite_prepared_ &&
            qwen3vl_multiview_text_composite_execution_len_ ==
                QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN &&
            qwen3vl_multiview_text_composite_resolved_chunk_size_ ==
                QWEN3VL_MULTIVIEW_TEXT_RESOLVED_CHUNK_SIZE &&
            qwen3vl_multiview_text_composite_allocation_chunk_size_ ==
                QWEN3VL_MULTIVIEW_TEXT_ALLOCATION_CHUNK_SIZE,
        "Qwen3-VL multiview Text fixed S225/C240->C225 layout was not "
        "prepared");
    validate_qwen3vl_multiview_text_model_profile();
}

SpmPipelineComponentLayout
CausalDecoderModel::prepare_qwen3vl_multiview_text_composite_layout(
    int64_t execution_len) {
    return prepare_qwen3vl_multiview_text_composite_layout(
        execution_len, /*force_block_twostage_chunk_v2=*/false);
}

SpmPipelineComponentLayout
CausalDecoderModel::prepare_qwen3vl_multiview_text_composite_layout(
    int64_t execution_len,
    bool force_block_twostage_chunk_v2) {
    TORCH_CHECK(
        !RpuKernelGraph::has_active(),
        "Qwen3-VL multiview Text composite prepare must run outside Graph "
        "capture");
    TORCH_CHECK(
        execution_len == QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN &&
            !qwen3vl_multiview_text_dry_prepared_ &&
            !qwen3vl_multiview_text_composite_prepared_ &&
            !qwen3vl_multiview_text_composite_dispatch_ &&
            !qwen3vl_multiview_text_composite_inputs_primed_ &&
            !qwen3vl_multiview_text_force_block_twostage_chunk_v2_ &&
            qwen3vl_multiview_text_composite_execution_len_ == 0 &&
            qwen3vl_multiview_text_composite_resolved_chunk_size_ == 0 &&
            qwen3vl_multiview_text_composite_allocation_chunk_size_ == 0 &&
            qwen3vl_multiview_text_composite_position_ids_addr_ == 0 &&
            qwen3vl_multiview_text_composite_rope_cos_il_addr_ == 0 &&
            qwen3vl_multiview_text_composite_rope_sin_il_addr_ == 0 &&
            std::all_of(
                qwen3vl_multiview_text_complement_src_bases_.begin(),
                qwen3vl_multiview_text_complement_src_bases_.end(),
                [](uint64_t slot) { return slot == 0; }) &&
            !qwen3vl_multiview_text_output_owner_.defined(),
        "Qwen3-VL multiview Text composite prepare requires one fresh exact "
        "S225 lifecycle");
    TORCH_CHECK(
        !wall_z1_active_ && !wall_z1_bound_ && !wall_z1_dispatch_ &&
            wall_z1_prepared_execution_len_ == 0 &&
            !qwen3vl_pooler_z1_active_ && !qwen3vl_pooler_z1_bound_ &&
            !qwen3vl_pooler_z1_dispatch_ &&
            qwen3vl_pooler_z1_prepared_execution_len_ == 0,
        "Qwen3-VL multiview Text composite cannot share a handle with a "
        "prepared physical Z1 profile");
    validate_qwen3vl_multiview_text_model_profile();

    qwen3vl_multiview_text_force_block_twostage_chunk_v2_ =
        force_block_twostage_chunk_v2;
    qwen3vl_multiview_text_composite_prepared_ = true;
    auto rollback = c10::make_scope_exit([&] {
        qwen3vl_multiview_text_composite_prepared_ = false;
        qwen3vl_multiview_text_force_block_twostage_chunk_v2_ = false;
    });
    const SpmPipelineCausalPrefillShape shape =
        resolve_spm_pipeline_causal_prefill_shape_for_cpu_contract(
            execution_len);
    TORCH_CHECK(
        shape.resolved_chunk_size ==
                QWEN3VL_MULTIVIEW_TEXT_RESOLVED_CHUNK_SIZE &&
            shape.chunks.size() == 1 && shape.chunks.front().idx == 0 &&
            shape.chunks.front().offset == 0 &&
            shape.chunks.front().len == execution_len &&
            shape.chunks.front().kv_seq_len == execution_len &&
            shape.kv_insert_chunks.size() == 1 &&
            shape.kv_insert_chunks.front().idx == 0 &&
            shape.kv_insert_chunks.front().offset == 0 &&
            shape.kv_insert_chunks.front().len == execution_len &&
            shape.kv_insert_chunks.front().kv_seq_len == execution_len &&
            shape.allocation_layout.chunk_size ==
                QWEN3VL_MULTIVIEW_TEXT_ALLOCATION_CHUNK_SIZE &&
            shape.chunk_mode == ChunkMode::SEQUENTIAL &&
            shape.inter_layer_io == InterLayerIO::SPM_RESIDENT &&
            !shape.chunk_outer_within_group,
        "Qwen3-VL multiview Text composite expected native C240 -> one "
        "allocation C225 causal-resident plan");
    const SpmPipelineComponentLayout prepared =
        prepare_spm_pipeline_component(shape.allocation_layout);
    TORCH_CHECK(
        prepared.temporary_bytes ==
            QWEN3VL_MULTIVIEW_TEXT_TEMPORARY_BYTES,
        "Qwen3-VL multiview Text composite scratch drifted: expected ",
        QWEN3VL_MULTIVIEW_TEXT_TEMPORARY_BYTES, " bytes/core, got ",
        prepared.temporary_bytes);

    qwen3vl_multiview_text_composite_execution_len_ = execution_len;
    qwen3vl_multiview_text_composite_resolved_chunk_size_ =
        shape.resolved_chunk_size;
    qwen3vl_multiview_text_composite_allocation_chunk_size_ =
        shape.allocation_layout.chunk_size;
    rollback.release();
    return prepared;
}

SpmFmbResolvedExecutionProfile
CausalDecoderModel::resolve_qwen3vl_multiview_text_composite_profile() {
    validate_qwen3vl_multiview_text_contract();
    TORCH_CHECK(
        !qwen3vl_multiview_text_composite_dispatch_,
        "Qwen3-VL multiview Text profile requires an idle component");
    SpmFmbResolvedProfileRequest request;
    request.version =
        qwen3vl_multiview_text_force_block_twostage_chunk_v2_
        ? QWEN3VL_MULTIVIEW_TEXT_TWOSTAGE_CHUNK_V2_PROFILE_VERSION
        : QWEN3VL_MULTIVIEW_TEXT_PROFILE_VERSION;
    request.chunks = {{0, 0, QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN,
                       QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN}};
    return resolve_spm_pipeline_execution_profile_for_build_trace(request);
}

SpmFmbResolvedPhaseManifest
CausalDecoderModel::seal_qwen3vl_multiview_text_composite_manifest(
    const SpmFmbResolvedExecutionProfile& profile,
    SpmScratchId arena,
    uint32_t arena_base) const {
    validate_qwen3vl_multiview_text_contract();
    return seal_spm_pipeline_live_resolved_manifest(
        profile, arena, arena_base);
}

SpmFmbConsumerRowSliceEndpoint
CausalDecoderModel::seal_qwen3vl_multiview_text_consumer_endpoint(
    const SpmFmbResolvedExecutionProfile& profile,
    const SpmFmbResolvedPhaseManifest& manifest,
    SpmPortId destination_port,
    int64_t row_begin,
    int layer_idx) const {
    validate_qwen3vl_multiview_text_contract();
    TORCH_CHECK(
        layer_idx >= 0 && layer_idx <= 2 &&
            std::find(
                QWEN3VL_MULTIVIEW_TEXT_IMAGE_ROW_BEGINS.begin(),
                QWEN3VL_MULTIVIEW_TEXT_IMAGE_ROW_BEGINS.end(),
                row_begin) !=
                QWEN3VL_MULTIVIEW_TEXT_IMAGE_ROW_BEGINS.end(),
        "Qwen3-VL multiview Text consumer endpoint requires layer 0/1/2 "
        "and row begin 1/67/133");
    SpmDense2DSpec storage;
    storage.dtype = SpmPortDType::Fp16;
    storage.rows = QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN;
    storage.cols = QWEN3VL_MULTIVIEW_TEXT_HIDDEN;
    storage.distribution = SpmPortDistribution::Replicated;
    storage.validate();
    const SpmFmbConsumerOccurrenceSelector occurrence{
        SpmFmbOccurrenceKind::LayerBody,
        /*body_id=*/0,
        /*group_id=*/0,
        layer_idx,
        /*chunk_index=*/0};
    return seal_spm_pipeline_consumer_row_slice_endpoint(
        profile, manifest, "residual1", /*allocation_layer=*/-1,
        SpmChunkKey(destination_port, /*ordinal=*/0), storage, row_begin,
        QWEN3VL_MULTIVIEW_TEXT_IMAGE_ROWS, occurrence);
}

void CausalDecoderModel::prime_qwen3vl_multiview_text_inputs(
    const at::Tensor& position_ids,
    int64_t execution_len,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il) {
    TORCH_CHECK(
        !RpuKernelGraph::has_active(),
        "Qwen3-VL multiview Text input prime must run outside Graph capture");
    validate_qwen3vl_multiview_text_contract();
    TORCH_CHECK(
        execution_len == QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN &&
            !qwen3vl_multiview_text_composite_dispatch_,
        "Qwen3-VL multiview Text input prime requires an idle component");
    validate_qwen3vl_multiview_text_position_ids(position_ids);
    TORCH_CHECK(
        rope_cos_il.has_value() == rope_sin_il.has_value(),
        "Qwen3-VL multiview Text input prime requires both or neither "
        "rope_cos_il/rope_sin_il");
    const bool partial_mrope = rope_cos_il.has_value();
    uint64_t rope_cos_il_addr = 0;
    uint64_t rope_sin_il_addr = 0;
    if (partial_mrope) {
        validate_qwen3vl_multiview_text_rope_il(
            *rope_cos_il, *rope_sin_il);
        rope_cos_il_addr =
            RpuGetDevAddr(rope_cos_il->data_ptr<c10::Half>());
        rope_sin_il_addr =
            RpuGetDevAddr(rope_sin_il->data_ptr<c10::Half>());
        TORCH_CHECK(
            rope_cos_il_addr != 0 && rope_sin_il_addr != 0,
            "Qwen3-VL multiview Text partial M-RoPE source resolved a zero "
            "RPU address");
    }
    const uint64_t position_ids_addr =
        RpuGetDevAddr(position_ids.data_ptr<int32_t>());
    TORCH_CHECK(position_ids_addr != 0,
                "Qwen3-VL multiview Text position resolved a zero RPU address");
    if (qwen3vl_multiview_text_composite_inputs_primed_) {
        TORCH_CHECK(
            qwen3vl_multiview_text_composite_position_ids_addr_ ==
                    position_ids_addr &&
                qwen3vl_multiview_text_composite_rope_cos_il_addr_ ==
                    rope_cos_il_addr &&
                qwen3vl_multiview_text_composite_rope_sin_il_addr_ ==
                    rope_sin_il_addr,
            "Qwen3-VL multiview Text M-RoPE re-prime address drifted");
    }
    auto position_dst = position_ids_keepalive_.narrow(
        0, /*start=*/0, QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN);
    position_dst.copy_(position_ids);
    rpu_ddr_flush_force_sized(
        position_dst.data_ptr<int32_t>(),
        static_cast<size_t>(QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN) * 3 *
            sizeof(int32_t));
    if (partial_mrope) {
        const int64_t half = QWEN3VL_MULTIVIEW_TEXT_HEAD_DIM / 2;
        auto cos_dst = cos_il_keepalive_.narrow(
            0, /*start=*/0, QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN);
        auto sin_dst = sin_il_keepalive_.narrow(
            0, /*start=*/0, QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN);
        cos_dst.copy_(*rope_cos_il);
        sin_dst.copy_(*rope_sin_il);
        const size_t il_bytes =
            static_cast<size_t>(QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN) *
            static_cast<size_t>(half) * sizeof(c10::Half);
        rpu_ddr_flush_force_sized(
            cos_dst.data_ptr<c10::Half>(), il_bytes);
        rpu_ddr_flush_force_sized(
            sin_dst.data_ptr<c10::Half>(), il_bytes);
    }
    ka_last_position_ = 0;
    ka_last_seq_ = QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN;
    ka_last_pos_src_ = position_ids.data_ptr();
    ka_last_pos_src_version_ =
        position_ids.unsafeGetTensorImpl()->version_counter().enabled()
        ? static_cast<int64_t>(position_ids.unsafeGetTensorImpl()
                                   ->version_counter().current_version())
        : -1;
    ka_last_cos_src_ =
        partial_mrope ? rope_cos_il->data_ptr() : nullptr;
    ka_last_cos_src_version_ =
        partial_mrope &&
                rope_cos_il->unsafeGetTensorImpl()->version_counter().enabled()
            ? static_cast<int64_t>(rope_cos_il->unsafeGetTensorImpl()
                                       ->version_counter().current_version())
            : -1;
    ka_last_sin_src_ =
        partial_mrope ? rope_sin_il->data_ptr() : nullptr;
    ka_last_sin_src_version_ =
        partial_mrope &&
                rope_sin_il->unsafeGetTensorImpl()->version_counter().enabled()
            ? static_cast<int64_t>(rope_sin_il->unsafeGetTensorImpl()
                                       ->version_counter().current_version())
            : -1;
    partial_mrope_active_ = partial_mrope;
    qwen3vl_multiview_text_composite_position_ids_addr_ = position_ids_addr;
    qwen3vl_multiview_text_composite_rope_cos_il_addr_ = rope_cos_il_addr;
    qwen3vl_multiview_text_composite_rope_sin_il_addr_ = rope_sin_il_addr;
    qwen3vl_multiview_text_composite_inputs_primed_ = true;
}

void CausalDecoderModel::rollback_qwen3vl_multiview_text_inputs() {
    TORCH_CHECK(
        !RpuKernelGraph::has_active() &&
            !qwen3vl_multiview_text_composite_dispatch_,
        "Qwen3-VL multiview Text input rollback requires one idle "
        "component outside Graph capture");
    qwen3vl_multiview_text_composite_position_ids_addr_ = 0;
    qwen3vl_multiview_text_composite_rope_cos_il_addr_ = 0;
    qwen3vl_multiview_text_composite_rope_sin_il_addr_ = 0;
    qwen3vl_multiview_text_composite_inputs_primed_ = false;
    partial_mrope_active_ = false;
    ka_last_position_ = -1;
    ka_last_seq_ = -1;
    ka_last_pos_src_ = nullptr;
    ka_last_pos_src_version_ = -1;
    ka_last_cos_src_ = nullptr;
    ka_last_cos_src_version_ = -1;
    ka_last_sin_src_ = nullptr;
    ka_last_sin_src_version_ = -1;
}

void CausalDecoderModel::unprepare_qwen3vl_multiview_text_composite_layout() {
    TORCH_CHECK(
        !RpuKernelGraph::has_active() &&
            qwen3vl_multiview_text_composite_prepared_ &&
            !qwen3vl_multiview_text_composite_dispatch_,
        "Qwen3-VL multiview Text unprepare requires one idle "
        "component outside Graph capture");
    rollback_qwen3vl_multiview_text_inputs();
    // These addresses remain part of a retained Graph's mutable-DMA topology.
    // The Z2 owner invalidates and retires that Graph before unprepare; a
    // position-only rollback must not silently make a parked Graph unusable.
    qwen3vl_multiview_text_complement_src_bases_.fill(0);
    qwen3vl_multiview_text_output_owner_ = at::Tensor();
    qwen3vl_multiview_text_composite_prepared_ = false;
    qwen3vl_multiview_text_force_block_twostage_chunk_v2_ = false;
    qwen3vl_multiview_text_composite_execution_len_ = 0;
    qwen3vl_multiview_text_composite_resolved_chunk_size_ = 0;
    qwen3vl_multiview_text_composite_allocation_chunk_size_ = 0;
}

at::Tensor CausalDecoderModel::forward_qwen3vl_multiview_text_composite(
    const at::Tensor& hidden_states,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    const at::Tensor& position_ids,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il) {
    validate_qwen3vl_multiview_text_contract();
    require_spm_pipeline_composite_consumer_physical_authority();
    TORCH_CHECK(
        RpuKernelGraph::has_active() &&
            qwen3vl_multiview_text_composite_inputs_primed_ &&
            !qwen3vl_multiview_text_composite_dispatch_,
        "Qwen3-VL multiview Text forward requires one primed, non-reentrant "
        "occurrence in the active outer Graph");
    TORCH_CHECK(
        hidden_states.defined() && hidden_states.dim() == 3 &&
            hidden_states.size(0) == 1 &&
            hidden_states.size(1) == QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN &&
            hidden_states.size(2) == QWEN3VL_MULTIVIEW_TEXT_HIDDEN &&
            hidden_states.device().type() == at::kPrivateUse1 &&
            hidden_states.scalar_type() == at::kHalf &&
            hidden_states.is_contiguous(),
        "Qwen3-VL multiview Text input must be contiguous RPU FP16 "
        "[1,225,2560]");
    validate_qwen3vl_multiview_text_position_ids(position_ids);
    TORCH_CHECK(
        rope_cos_il.has_value() == rope_sin_il.has_value(),
        "Qwen3-VL multiview Text forward requires both or neither "
        "rope_cos_il/rope_sin_il");
    const bool partial_mrope = rope_cos_il.has_value();
    uint64_t rope_cos_il_addr = 0;
    uint64_t rope_sin_il_addr = 0;
    if (partial_mrope) {
        validate_qwen3vl_multiview_text_rope_il(
            *rope_cos_il, *rope_sin_il);
        rope_cos_il_addr =
            RpuGetDevAddr(rope_cos_il->data_ptr<c10::Half>());
        rope_sin_il_addr =
            RpuGetDevAddr(rope_sin_il->data_ptr<c10::Half>());
    }
    TORCH_CHECK(
        qwen3vl_multiview_text_composite_position_ids_addr_ ==
                RpuGetDevAddr(position_ids.data_ptr<int32_t>()) &&
            qwen3vl_multiview_text_composite_rope_cos_il_addr_ ==
                rope_cos_il_addr &&
            qwen3vl_multiview_text_composite_rope_sin_il_addr_ ==
                rope_sin_il_addr &&
            ka_last_position_ == 0 &&
            ka_last_seq_ == QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN &&
            ka_last_pos_src_ == position_ids.data_ptr() &&
            ka_last_cos_src_ ==
                (partial_mrope ? rope_cos_il->data_ptr() : nullptr) &&
            ka_last_sin_src_ ==
                (partial_mrope ? rope_sin_il->data_ptr() : nullptr) &&
            partial_mrope_active_ == partial_mrope,
        "Qwen3-VL multiview Text primed M-RoPE keepalive drifted");

    qwen3vl_multiview_text_composite_dispatch_ = true;
    auto clear_dispatch = c10::make_scope_exit(
        [&] { qwen3vl_multiview_text_composite_dispatch_ = false; });
    at::Tensor result = forward(
        hidden_states, k_caches, v_caches,
        /*attention_mask=*/std::nullopt,
        /*position=*/0,
        /*is_causal=*/true,
        /*position_ids=*/position_ids,
        /*deepstack_dense_visual_embeds=*/std::nullopt,
        /*rope_cos_il=*/rope_cos_il,
        /*rope_sin_il=*/rope_sin_il,
        /*cos_sin_offset=*/-1);
    TORCH_CHECK(
        get_last_resolved_chunk_size() ==
            QWEN3VL_MULTIVIEW_TEXT_RESOLVED_CHUNK_SIZE,
        "Qwen3-VL multiview Text resolved chunk drifted after prepare");
    const at::Tensor& registry_output = output_tensor();
    TORCH_CHECK(result.defined() && registry_output.defined() &&
                    result.is_same(registry_output),
                "Qwen3-VL multiview Text forward did not return the current "
                "registry-stable output");
    if (!qwen3vl_multiview_text_output_owner_.defined()) {
        qwen3vl_multiview_text_output_owner_ = result;
    }
    TORCH_CHECK(
        qwen3vl_multiview_text_output_owner_.is_same(registry_output) &&
            qwen3vl_multiview_text_output_owner_.storage()
                    .unsafeGetStorageImpl() ==
                registry_output.storage().unsafeGetStorageImpl() &&
            qwen3vl_multiview_text_output_owner_.data_ptr() ==
                registry_output.data_ptr(),
        "Qwen3-VL multiview Text registry output identity drifted");
    return result;
}

void CausalDecoderModel::stage_qwen3vl_multiview_text_outer_fast_component(
    GraphKernelRegisterCensusGuard& guard,
    at::TensorList k_caches,
    at::TensorList v_caches) const {
    validate_qwen3vl_multiview_text_contract();
    TORCH_CHECK(
        RpuKernelGraph::has_active() &&
            (RpuKernelGraph::active().state() ==
                 RpuKernelGraph::State::RECORDING ||
             RpuKernelGraph::active().state() ==
                 RpuKernelGraph::State::REPLAYING) &&
            qwen3vl_multiview_text_composite_inputs_primed_ &&
            !qwen3vl_multiview_text_composite_dispatch_ &&
            qwen3vl_multiview_text_composite_position_ids_addr_ != 0 &&
            qwen3vl_multiview_text_composite_rope_cos_il_addr_ != 0 &&
            qwen3vl_multiview_text_composite_rope_sin_il_addr_ != 0 &&
            ka_last_position_ == 0 &&
            ka_last_seq_ == QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN &&
            ka_last_pos_src_ != nullptr && ka_last_cos_src_ != nullptr &&
            ka_last_sin_src_ != nullptr && partial_mrope_active_,
        "Qwen3-VL multiview Text outer-fast component requires one primed "
        "idle occurrence in the active retained Graph");
    stage_spm_outer_fast_component(guard, k_caches, v_caches);
}

void CausalDecoderModel::bind_qwen3vl_multiview_text_outer_fast_input(
    GraphKernelRegisterCensusGuard& guard,
    const at::Tensor& hidden_states) {
    validate_qwen3vl_multiview_text_contract();
    TORCH_CHECK(
        RpuKernelGraph::has_active() &&
            RpuKernelGraph::active().state() ==
                RpuKernelGraph::State::REPLAYING &&
            qwen3vl_multiview_text_composite_inputs_primed_ &&
            !qwen3vl_multiview_text_composite_dispatch_ &&
            qwen3vl_multiview_text_composite_position_ids_addr_ != 0 &&
            qwen3vl_multiview_text_composite_rope_cos_il_addr_ != 0 &&
            qwen3vl_multiview_text_composite_rope_sin_il_addr_ != 0 &&
            ka_last_position_ == 0 &&
            ka_last_seq_ == QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN &&
            ka_last_pos_src_ != nullptr && ka_last_cos_src_ != nullptr &&
            ka_last_sin_src_ != nullptr && partial_mrope_active_,
        "Qwen3-VL multiview Text outer-fast input requires one primed idle "
        "REPLAY occurrence");
    TORCH_CHECK(
        hidden_states.defined() && hidden_states.dim() == 3 &&
            hidden_states.size(0) == 1 &&
            hidden_states.size(1) == QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN &&
            hidden_states.size(2) == QWEN3VL_MULTIVIEW_TEXT_HIDDEN &&
            hidden_states.device().type() == at::kPrivateUse1 &&
            hidden_states.scalar_type() == at::kHalf &&
            hidden_states.layout() == c10::Layout::Strided &&
            hidden_states.is_contiguous(),
        "Qwen3-VL multiview Text outer-fast input must be contiguous RPU "
        "FP16 [1,225,2560]");
    TORCH_CHECK(
        std::all_of(
            qwen3vl_multiview_text_complement_src_bases_.begin(),
            qwen3vl_multiview_text_complement_src_bases_.end(),
            [](uint64_t slot) { return slot != 0; }),
        "Qwen3-VL multiview Text outer-fast mutable input slots were not "
        "published by retained BUILD");

    const size_t row_bytes =
        static_cast<size_t>(QWEN3VL_MULTIVIEW_TEXT_HIDDEN) *
        sizeof(c10::Half);
    for (size_t ordinal = 0;
         ordinal < QWEN3VL_MULTIVIEW_TEXT_COMPLEMENT_RUNS.size();
         ++ordinal) {
        const auto& run = QWEN3VL_MULTIVIEW_TEXT_COMPLEMENT_RUNS[ordinal];
        guard.bind_fast_mutable_dma(
            qwen3vl_multiview_text_complement_src_bases_[ordinal],
            hidden_states, GraphOuterFastDmaSide::Source,
            static_cast<size_t>(run[0]) * row_bytes,
            static_cast<size_t>(run[1]) * row_bytes);
    }
}

at::Tensor CausalDecoderModel::qwen3vl_multiview_text_outer_fast_output() const {
    TORCH_CHECK(
        RpuKernelGraph::has_active() &&
            RpuKernelGraph::active().state() ==
                RpuKernelGraph::State::REPLAYING,
        "Qwen3-VL multiview Text outer-fast output requires one active "
        "REPLAY Graph");
    validate_qwen3vl_multiview_text_contract();
    TORCH_CHECK(
            qwen3vl_multiview_text_composite_inputs_primed_ &&
            !qwen3vl_multiview_text_composite_dispatch_ &&
            qwen3vl_multiview_text_composite_position_ids_addr_ != 0 &&
            qwen3vl_multiview_text_composite_rope_cos_il_addr_ != 0 &&
            qwen3vl_multiview_text_composite_rope_sin_il_addr_ != 0 &&
            ka_last_position_ == 0 &&
            ka_last_seq_ == QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN &&
            ka_last_pos_src_ != nullptr && ka_last_cos_src_ != nullptr &&
            ka_last_sin_src_ != nullptr && partial_mrope_active_,
        "Qwen3-VL multiview Text outer-fast output requires one primed idle "
        "component");
    const at::Tensor& current = output_tensor();
    TORCH_CHECK(
        current.defined() && current.dim() == 3 && current.size(0) == 1 &&
            current.size(1) == QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN &&
            current.size(2) == QWEN3VL_MULTIVIEW_TEXT_HIDDEN &&
            current.scalar_type() == at::kHalf &&
            current.device().type() == at::kPrivateUse1 &&
            current.layout() == c10::Layout::Strided &&
            current.is_contiguous(),
        "Qwen3-VL multiview Text outer-fast registry output shape/device "
        "drifted");
    TORCH_CHECK(
        qwen3vl_multiview_text_output_owner_.defined() &&
            qwen3vl_multiview_text_output_owner_.is_same(current) &&
            qwen3vl_multiview_text_output_owner_.storage()
                    .unsafeGetStorageImpl() ==
                current.storage().unsafeGetStorageImpl() &&
            qwen3vl_multiview_text_output_owner_.data_ptr() ==
                current.data_ptr(),
        "Qwen3-VL multiview Text outer-fast registry output identity "
        "drifted");
    return current;
}

void CausalDecoderModel::emit_qwen3vl_multiview_text_complement_ingress(
    int layer_idx,
    const ChunkInfo& chunk,
    uint32_t residual_addr) {
    TORCH_CHECK(
        qwen3vl_multiview_text_composite_dispatch_ && layer_idx == 0 &&
            chunk.idx == 0 && chunk.offset == 0 &&
            chunk.len == QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN &&
            chunk.kv_seq_len == QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN &&
            residual_addr == addr(0, "residual1"),
        "Qwen3-VL multiview Text complement ingress requires the exact "
        "layer-0 S225 residual callback");
    for (size_t ordinal = 0;
         ordinal < QWEN3VL_MULTIVIEW_TEXT_COMPLEMENT_RUNS.size();
         ++ordinal) {
        const auto& run = QWEN3VL_MULTIVIEW_TEXT_COMPLEMENT_RUNS[ordinal];
        emit_layer_input_row_run_dma(
            qwen3vl_multiview_text_complement_src_bases_[ordinal], layer_idx,
            chunk, run[0], run[1], "residual1");
    }
}

void CausalDecoderModel::emit_qwen3vl_multiview_text_retained_deepstack_add(
    int layer_idx,
    const ChunkInfo& chunk,
    uint32_t residual_addr) {
    if (layer_idx < 0 || layer_idx > 2) return;
    TORCH_CHECK(
        qwen3vl_multiview_text_composite_dispatch_ &&
            chunk.idx == 0 && chunk.offset == 0 &&
            chunk.len == QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN &&
            residual_addr == addr(0, "residual1"),
        "Qwen3-VL multiview Text retained DeepStack add requires the exact "
        "bound S225 callback");
    for (size_t image = 0; image < 3; ++image) {
        emit_spm_pipeline_composite_run_add(
            layer_idx, chunk,
            QWEN3VL_MULTIVIEW_TEXT_IMAGE_ROW_BEGINS[image],
            QWEN3VL_MULTIVIEW_TEXT_IMAGE_ROWS, "residual1");
    }
}

SpmPipelineCausalPrefillDryLayout
CausalDecoderModel::prepare_qwen3vl_multiview_text_dry_layout(
    int64_t execution_len) {
    TORCH_CHECK(
        !RpuKernelGraph::has_active(),
        "Qwen3-VL multiview Text dry prepare must run outside Graph capture");
    TORCH_CHECK(
        !qwen3vl_multiview_text_dry_prepared_ &&
            qwen3vl_multiview_text_dry_execution_len_ == 0 &&
            qwen3vl_multiview_text_dry_resolved_chunk_size_ == 0 &&
            qwen3vl_multiview_text_dry_allocation_chunk_size_ == 0 &&
            !qwen3vl_multiview_text_composite_prepared_ &&
            !qwen3vl_multiview_text_composite_dispatch_ &&
            !qwen3vl_multiview_text_composite_inputs_primed_ &&
            !qwen3vl_multiview_text_force_block_twostage_chunk_v2_,
        "Qwen3-VL multiview Text dry prepare requires a fresh dry lifecycle");
    TORCH_CHECK(
        !wall_z1_active_ && !wall_z1_bound_ && !wall_z1_dispatch_ &&
            !wall_z1_inputs_primed_ &&
            wall_z1_prepared_execution_len_ == 0 &&
            wall_z1_prepared_real_len_ == 0 &&
            wall_z1_prepared_chunk_size_ == 0 &&
            wall_z1_destination_addr_ == 0 && wall_z1_epoch_ == 0 &&
            wall_z1_plan_hash_ == 0 && wall_z1_position_ids_addr_ == 0 &&
            wall_z1_rope_cos_il_addr_ == 0 &&
            wall_z1_rope_sin_il_addr_ == 0 &&
            !qwen3vl_pooler_z1_active_ &&
            !qwen3vl_pooler_z1_bound_ &&
            !qwen3vl_pooler_z1_dispatch_ &&
            !qwen3vl_pooler_z1_inputs_primed_ &&
            qwen3vl_pooler_z1_prepared_execution_len_ == 0 &&
            qwen3vl_pooler_z1_prepared_real_len_ == 0 &&
            qwen3vl_pooler_z1_prepared_chunk_size_ == 0 &&
            qwen3vl_pooler_z1_destination_addr_ == 0 &&
            qwen3vl_pooler_z1_retained_deepstack_count_ == 0 &&
            qwen3vl_pooler_z1_deepstack_bound_count_ == 0 &&
            qwen3vl_pooler_z1_retained_deepstack_bindings_empty() &&
            qwen3vl_pooler_z1_epoch_ == 0 &&
            qwen3vl_pooler_z1_plan_hash_ == 0 &&
            qwen3vl_pooler_z1_position_ids_addr_ == 0 &&
            !qwen3vl_pooler_z1_output_owner_.defined(),
        "Qwen3-VL multiview Text dry prepare cannot share a handle with a "
        "prepared physical Z1 profile");
    TORCH_CHECK(
        execution_len == 225 && num_layers() == 36 &&
            hidden_size() == 2560 && intermediate_size() == 9728 &&
            num_q_heads() == 32 && num_kv_heads() == 8 &&
            head_dim() == 128,
        "Qwen3-VL multiview Text dry prepare accepts only the exact "
        "LingBot2 S225/L36/H2560/I9728/Q32/KV8/D128 profile");
    TORCH_CHECK(
        get_chunk_size_override() == 0 &&
            configured_chunk_size_cap_ == 0 && !equal_two_prefill_,
        "Qwen3-VL multiview Text dry prepare requires the native auto-chunk "
        "policy with per-handle override/cap and "
        "equal-two disabled");
    TORCH_CHECK(
        has_mrope_ && mrope_section_.size() == 3 &&
            mrope_section_[0] == 24 && mrope_section_[1] == 20 &&
            mrope_section_[2] == 20 &&
            deepstack_lang_layers_.size() == 3 &&
            deepstack_lang_layers_[0] == 0 &&
            deepstack_lang_layers_[1] == 1 &&
            deepstack_lang_layers_[2] == 2,
        "Qwen3-VL multiview Text dry prepare requires M-RoPE [24,20,20] "
        "and DeepStack consumers [0,1,2]");
    TORCH_CHECK(
        !has_qkv_bias_ && has_qk_norm_ && !adarms_ &&
            static_cast<int64_t>(layer_weights_.size()) == num_layers() &&
            final_norm_w_.defined() && cos_.defined() && sin_.defined(),
        "Qwen3-VL multiview Text dry prepare requires complete supported "
        "Qwen3 weights");

    qwen3vl_multiview_text_dry_prepared_ = true;
    auto rollback = c10::make_scope_exit([&] {
        qwen3vl_multiview_text_dry_prepared_ = false;
    });
    SpmPipelineCausalPrefillShape shape =
        resolve_spm_pipeline_causal_prefill_shape_for_cpu_contract(
            execution_len);
    TORCH_CHECK(
        shape.resolved_chunk_size == 240 && shape.chunks.size() == 1 &&
            shape.chunks.front().idx == 0 &&
            shape.chunks.front().offset == 0 &&
            shape.chunks.front().len == execution_len &&
            shape.chunks.front().kv_seq_len == execution_len &&
            shape.allocation_layout.chunk_size == execution_len &&
            shape.chunk_mode == ChunkMode::SEQUENTIAL &&
            shape.inter_layer_io == InterLayerIO::SPM_RESIDENT,
        "Qwen3-VL multiview Text dry prepare expected native C240 -> one "
        "allocation C225 causal-resident plan");
    SpmPipelineComponentLayout component =
        prepare_spm_pipeline_component_for_cpu_contract(
            shape.allocation_layout);

    qwen3vl_multiview_text_dry_execution_len_ = execution_len;
    qwen3vl_multiview_text_dry_resolved_chunk_size_ =
        shape.resolved_chunk_size;
    qwen3vl_multiview_text_dry_allocation_chunk_size_ =
        shape.allocation_layout.chunk_size;
    rollback.release();
    return {std::move(shape), component};
}

void CausalDecoderModel::cancel_qwen3vl_multiview_text_dry_layout() {
    TORCH_CHECK(
        !RpuKernelGraph::has_active(),
        "Qwen3-VL multiview Text dry cancel must run outside Graph capture");
    TORCH_CHECK(
        qwen3vl_multiview_text_dry_prepared_ &&
            qwen3vl_multiview_text_dry_execution_len_ == 225 &&
            qwen3vl_multiview_text_dry_resolved_chunk_size_ == 240 &&
            qwen3vl_multiview_text_dry_allocation_chunk_size_ == 225,
        "Qwen3-VL multiview Text dry cancel requires one exact prepared dry "
        "profile");
    cancel_spm_pipeline_component_for_cpu_contract();
    qwen3vl_multiview_text_dry_prepared_ = false;
    qwen3vl_multiview_text_dry_execution_len_ = 0;
    qwen3vl_multiview_text_dry_resolved_chunk_size_ = 0;
    qwen3vl_multiview_text_dry_allocation_chunk_size_ = 0;
}

CausalDecoderModel::Z1LayerInputSelection
CausalDecoderModel::select_z1_layer_input(
    int layer_idx,
    const ChunkInfo& chunk) const {
    TORCH_CHECK(!(wall_z1_bound_ && qwen3vl_pooler_z1_bound_),
                "CausalDecoderModel cannot bind Wall-OSS and Qwen3-VL Z1 "
                "destinations simultaneously");
    if (qwen3vl_multiview_text_composite_dispatch_ && layer_idx == 0) {
        return {addr(0, "residual1"), true};
    }
    if (qwen3vl_pooler_z1_bound_ && layer_idx == 0) {
        return {qwen3vl_pooler_z1_layer_input_residual_addr(layer_idx, chunk),
                true};
    }
    if (wall_z1_bound_ && layer_idx == 0) {
        return {wall_z1_layer_input_residual_addr(layer_idx, chunk), true};
    }
    return {addr(0, "residual1"), false};
}

void CausalDecoderModel::check_qwen3vl_pooler_z1_destroy_allowed() const {
    TORCH_CHECK(!qwen3vl_pooler_z1_active_ &&
                    !qwen3vl_pooler_z1_bound_ &&
                    qwen3vl_pooler_z1_retained_deepstack_count_ == 0 &&
                    qwen3vl_pooler_z1_retained_deepstack_bindings_empty() &&
                    !qwen3vl_pooler_z1_dispatch_ &&
                    !qwen3vl_pooler_z1_inputs_primed_ &&
                    qwen3vl_pooler_z1_prepared_execution_len_ == 0 &&
                    qwen3vl_pooler_z1_prepared_real_len_ == 0 &&
                    qwen3vl_pooler_z1_prepared_chunk_size_ == 0 &&
                    !qwen3vl_multiview_text_dry_prepared_ &&
                    qwen3vl_multiview_text_dry_execution_len_ == 0 &&
                    qwen3vl_multiview_text_dry_resolved_chunk_size_ == 0 &&
                    qwen3vl_multiview_text_dry_allocation_chunk_size_ == 0 &&
                    !qwen3vl_multiview_text_composite_prepared_ &&
                    !qwen3vl_multiview_text_composite_dispatch_ &&
                    !qwen3vl_multiview_text_composite_inputs_primed_ &&
                    !qwen3vl_multiview_text_force_block_twostage_chunk_v2_ &&
                    qwen3vl_multiview_text_composite_execution_len_ == 0 &&
                    qwen3vl_multiview_text_composite_resolved_chunk_size_ == 0 &&
                    qwen3vl_multiview_text_composite_allocation_chunk_size_ == 0 &&
                    qwen3vl_multiview_text_composite_position_ids_addr_ == 0 &&
                    qwen3vl_multiview_text_composite_rope_cos_il_addr_ == 0 &&
                    qwen3vl_multiview_text_composite_rope_sin_il_addr_ == 0 &&
                    std::all_of(
                        qwen3vl_multiview_text_complement_src_bases_.begin(),
                        qwen3vl_multiview_text_complement_src_bases_.end(),
                        [](uint64_t slot) { return slot == 0; }) &&
                    !qwen3vl_multiview_text_output_owner_.defined(),
                "cannot destroy the causal decoder handle while its Qwen3-VL "
                "pooler Z1 or multiview dry layout is prepared, primed, or "
                "active; clear the outer GraphCache and unprepare/cancel first");
}

}  // namespace v3
// =============================================================================
// Instance registry — ModelHandleRegistry<v3::CausalDecoderModel>
//
// Handle 单调递增且不复用;GraphCache 由 Python adapter 按模型实例持有。
// Thread-safety / 并发 destroy 限制见 model_handle_registry.h 注释。
// =============================================================================

using CausalDecoderRegistry = ModelHandleRegistry<v3::CausalDecoderModel>;

namespace v3::wall_oss_z1_internal {

SpmPipelineComponentLayout prepare_text(
    int64_t handle,
    int64_t execution_len,
    int64_t real_len) {
    return CausalDecoderRegistry::get(handle, "wall_oss_z1_prepare_text")
        ->prepare_wall_oss_z1_layout(execution_len, real_len);
}

SpmDense2DSpec text_destination_spec(
    int64_t handle,
    int64_t execution_len) {
    return CausalDecoderRegistry::get(
               handle, "wall_oss_z1_text_destination_spec")
        ->wall_oss_z1_destination_spec(execution_len);
}

void prime_text_inputs(
    int64_t handle,
    const at::Tensor& position_ids,
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il,
    int64_t execution_len) {
    CausalDecoderRegistry::get(handle, "wall_oss_z1_prime_text_inputs")
        ->prime_wall_oss_z1_inputs(
            position_ids, rope_cos_il, rope_sin_il, execution_len);
}

void rollback_text_inputs(int64_t handle) {
    CausalDecoderRegistry::get(handle, "wall_oss_z1_rollback_text_inputs")
        ->rollback_wall_oss_z1_inputs();
}

void unprepare_text(int64_t handle) {
    CausalDecoderRegistry::get(handle, "wall_oss_z1_unprepare_text")
        ->unprepare_wall_oss_z1_layout();
}

void adopt_text(
    int64_t handle,
    const SpmPipelineLease& lease,
    const SpmTensorView& scratch) {
    CausalDecoderRegistry::get(handle, "wall_oss_z1_adopt_text")
        ->adopt_wall_oss_z1_layout(lease, scratch);
}

void bind_text_destination(
    int64_t handle,
    const SpmPipelineLease& lease,
    const SpmPortView& destination) {
    CausalDecoderRegistry::get(
        handle, "wall_oss_z1_bind_text_destination")
        ->bind_wall_oss_z1_destination(lease, destination);
}

void validate_text(int64_t handle, const SpmPipelineLease& lease) {
    CausalDecoderRegistry::get(handle, "wall_oss_z1_validate_text")
        ->validate_wall_oss_z1_layout(lease);
}

void clear_text(int64_t handle, uint64_t epoch, uint64_t plan_hash) {
    CausalDecoderRegistry::get(handle, "wall_oss_z1_clear_text")
        ->clear_wall_oss_z1_layout(epoch, plan_hash);
}

at::Tensor forward_text_z1(
    int64_t handle,
    const at::Tensor& hidden_shape_carrier,
    at::TensorList k_caches,
    at::TensorList v_caches,
    const at::Tensor& position_ids,
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il,
    uint64_t epoch,
    uint64_t plan_hash) {
    auto* model =
        CausalDecoderRegistry::get(handle, "wall_oss_z1_forward_text");
    std::vector<at::Tensor> kc(k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> vc(v_caches.begin(), v_caches.end());
    return model->forward_wall_oss_z1(
        hidden_shape_carrier, kc, vc,
        position_ids, rope_cos_il, rope_sin_il,
        epoch, plan_hash);
}

void check_text_destroy_allowed(int64_t handle) {
    CausalDecoderRegistry::get(
        handle, "wall_oss_z1_check_text_destroy_allowed")
        ->check_wall_oss_z1_destroy_allowed();
}

}  // namespace v3::wall_oss_z1_internal

namespace v3::qwen3vl_pooler_z1_internal {

SpmPipelineComponentLayout prepare_text(
    int64_t handle,
    int64_t execution_len,
    int64_t real_len) {
    return CausalDecoderRegistry::get(
               handle, "qwen3vl_pooler_z1_prepare_text")
        ->prepare_qwen3vl_pooler_z1_layout(
            execution_len, real_len, /*retained_deepstack_count=*/0);
}

SpmPipelineComponentLayout prepare_text_deepstack1(
    int64_t handle,
    int64_t execution_len,
    int64_t real_len) {
    return CausalDecoderRegistry::get(
               handle, "qwen3vl_pooler_z1_prepare_text_deepstack1")
        ->prepare_qwen3vl_pooler_z1_layout(
            execution_len, real_len, /*retained_deepstack_count=*/1);
}

SpmPipelineComponentLayout prepare_text_deepstack3(
    int64_t handle,
    int64_t execution_len,
    int64_t real_len) {
    return CausalDecoderRegistry::get(
               handle, "qwen3vl_pooler_z1_prepare_text_deepstack3")
        ->prepare_qwen3vl_pooler_z1_layout(
            execution_len, real_len, /*retained_deepstack_count=*/3);
}

void prime_text_inputs(
    int64_t handle,
    const at::Tensor& position_ids,
    int64_t execution_len,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il) {
    CausalDecoderRegistry::get(
        handle, "qwen3vl_pooler_z1_prime_text_inputs")
        ->prime_qwen3vl_pooler_z1_inputs(
            position_ids, execution_len, rope_cos_il, rope_sin_il);
}

void rollback_text_inputs(int64_t handle) {
    CausalDecoderRegistry::get(
        handle, "qwen3vl_pooler_z1_rollback_text_inputs")
        ->rollback_qwen3vl_pooler_z1_inputs();
}

void unprepare_text(int64_t handle) {
    CausalDecoderRegistry::get(
        handle, "qwen3vl_pooler_z1_unprepare_text")
        ->unprepare_qwen3vl_pooler_z1_layout();
}

SpmPipelineCausalPrefillDryLayout prepare_multiview_text_dry(
    int64_t handle,
    int64_t execution_len) {
    return CausalDecoderRegistry::get(
               handle, "qwen3vl_multiview_prepare_text_dry")
        ->prepare_qwen3vl_multiview_text_dry_layout(execution_len);
}

void cancel_multiview_text_dry(int64_t handle) {
    CausalDecoderRegistry::get(
        handle, "qwen3vl_multiview_cancel_text_dry")
        ->cancel_qwen3vl_multiview_text_dry_layout();
}

SpmPipelineComponentLayout prepare_multiview_text_composite(
    int64_t handle,
    int64_t execution_len,
    bool force_block_twostage_chunk_v2) {
    return CausalDecoderRegistry::get(
               handle, "qwen3vl_multiview_prepare_text_composite")
        ->prepare_qwen3vl_multiview_text_composite_layout(
            execution_len, force_block_twostage_chunk_v2);
}

SpmPipelineComponentLayout prepare_multiview_text_composite(
    int64_t handle,
    int64_t execution_len) {
    return prepare_multiview_text_composite(
        handle, execution_len,
        /*force_block_twostage_chunk_v2=*/false);
}

SpmFmbResolvedExecutionProfile resolve_multiview_text_profile(
    int64_t handle) {
    return CausalDecoderRegistry::get(
               handle, "qwen3vl_multiview_resolve_text_profile")
        ->resolve_qwen3vl_multiview_text_composite_profile();
}

FusedModelBase& multiview_text_owner(int64_t handle) {
    return *CausalDecoderRegistry::get(
        handle, "qwen3vl_multiview_text_owner");
}

SpmFmbResolvedPhaseManifest seal_multiview_text_manifest(
    int64_t handle,
    const SpmFmbResolvedExecutionProfile& profile,
    SpmScratchId arena,
    uint32_t arena_base) {
    return CausalDecoderRegistry::get(
               handle, "qwen3vl_multiview_seal_text_manifest")
        ->seal_qwen3vl_multiview_text_composite_manifest(
            profile, arena, arena_base);
}

SpmFmbConsumerRowSliceEndpoint seal_multiview_text_consumer(
    int64_t handle,
    const SpmFmbResolvedExecutionProfile& profile,
    const SpmFmbResolvedPhaseManifest& manifest,
    SpmPortId destination_port,
    int64_t row_begin,
    int layer_idx) {
    return CausalDecoderRegistry::get(
               handle, "qwen3vl_multiview_seal_text_consumer")
        ->seal_qwen3vl_multiview_text_consumer_endpoint(
            profile, manifest, destination_port, row_begin, layer_idx);
}

void prime_multiview_text_inputs(
    int64_t handle,
    const at::Tensor& position_ids,
    int64_t execution_len,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il) {
    CausalDecoderRegistry::get(
        handle, "qwen3vl_multiview_prime_text_inputs")
        ->prime_qwen3vl_multiview_text_inputs(
            position_ids, execution_len, rope_cos_il, rope_sin_il);
}

void rollback_multiview_text_inputs(int64_t handle) {
    CausalDecoderRegistry::get(
        handle, "qwen3vl_multiview_rollback_text_inputs")
        ->rollback_qwen3vl_multiview_text_inputs();
}

void unprepare_multiview_text_composite(int64_t handle) {
    CausalDecoderRegistry::get(
        handle, "qwen3vl_multiview_unprepare_text_composite")
        ->unprepare_qwen3vl_multiview_text_composite_layout();
}

at::Tensor forward_multiview_text_composite(
    int64_t handle,
    const at::Tensor& hidden_states,
    at::TensorList k_caches,
    at::TensorList v_caches,
    const at::Tensor& position_ids,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il) {
    std::vector<at::Tensor> k_cache_vec(
        k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> v_cache_vec(
        v_caches.begin(), v_caches.end());
    return CausalDecoderRegistry::get(
               handle, "qwen3vl_multiview_forward_text_composite")
        ->forward_qwen3vl_multiview_text_composite(
            hidden_states, k_cache_vec, v_cache_vec, position_ids,
            rope_cos_il, rope_sin_il);
}

void stage_multiview_text_outer_fast_component(
    int64_t handle,
    GraphKernelRegisterCensusGuard& guard,
    at::TensorList k_caches,
    at::TensorList v_caches) {
    CausalDecoderRegistry::get(
        handle, "qwen3vl_multiview_stage_text_outer_fast_component")
        ->stage_qwen3vl_multiview_text_outer_fast_component(
            guard, k_caches, v_caches);
}

void bind_multiview_text_outer_fast_input(
    int64_t handle,
    GraphKernelRegisterCensusGuard& guard,
    const at::Tensor& hidden_states) {
    CausalDecoderRegistry::get(
        handle, "qwen3vl_multiview_bind_text_outer_fast_input")
        ->bind_qwen3vl_multiview_text_outer_fast_input(
            guard, hidden_states);
}

at::Tensor multiview_text_outer_fast_output(int64_t handle) {
    return CausalDecoderRegistry::get(
               handle, "qwen3vl_multiview_text_outer_fast_output")
        ->qwen3vl_multiview_text_outer_fast_output();
}

SpmDense2DSpec text_destination_spec(
    int64_t handle,
    int64_t execution_len) {
    return CausalDecoderRegistry::get(
               handle, "qwen3vl_pooler_z1_text_destination_spec")
        ->qwen3vl_pooler_z1_destination_spec(execution_len);
}

SpmDense2DSpec text_deepstack_spec(int64_t handle, int64_t ordinal) {
    return CausalDecoderRegistry::get(
               handle, "qwen3vl_pooler_z1_text_deepstack_spec")
        ->qwen3vl_pooler_z1_deepstack_spec(ordinal);
}

SpmDense2DSpec text_deepstack1_spec(int64_t handle) {
    return text_deepstack_spec(handle, /*ordinal=*/0);
}

void bind_text_deepstack(
    int64_t handle,
    const SpmPipelineLease& lease,
    const SpmPortView& deepstack,
    int64_t ordinal) {
    CausalDecoderRegistry::get(
        handle, "qwen3vl_pooler_z1_bind_text_deepstack")
        ->bind_qwen3vl_pooler_z1_deepstack(lease, deepstack, ordinal);
}

void bind_text_deepstack1(
    int64_t handle,
    const SpmPipelineLease& lease,
    const SpmPortView& deepstack1) {
    bind_text_deepstack(
        handle, lease, deepstack1, /*ordinal=*/0);
}

void adopt_text(
    int64_t handle,
    const SpmPipelineLease& lease,
    const SpmTensorView& scratch) {
    CausalDecoderRegistry::get(handle, "qwen3vl_pooler_z1_adopt_text")
        ->adopt_qwen3vl_pooler_z1_layout(lease, scratch);
}

void bind_text_destination(
    int64_t handle,
    const SpmPipelineLease& lease,
    const SpmPortView& destination) {
    CausalDecoderRegistry::get(
        handle, "qwen3vl_pooler_z1_bind_text_destination")
        ->bind_qwen3vl_pooler_z1_destination(lease, destination);
}

void validate_text(int64_t handle, const SpmPipelineLease& lease) {
    CausalDecoderRegistry::get(handle, "qwen3vl_pooler_z1_validate_text")
        ->validate_qwen3vl_pooler_z1_layout(lease);
}

void clear_text(int64_t handle, uint64_t epoch, uint64_t plan_hash) {
    CausalDecoderRegistry::get(handle, "qwen3vl_pooler_z1_clear_text")
        ->clear_qwen3vl_pooler_z1_layout(epoch, plan_hash);
}

void stage_text_outer_fast_component(
    int64_t handle,
    GraphKernelRegisterCensusGuard& guard,
    at::TensorList k_caches,
    at::TensorList v_caches) {
    CausalDecoderRegistry::get(
        handle, "qwen3vl_pooler_z1_stage_text_outer_fast_component")
        ->stage_qwen3vl_pooler_z1_outer_fast_component(
            guard, k_caches, v_caches);
}

at::Tensor text_outer_fast_output(int64_t handle) {
    return CausalDecoderRegistry::get(
               handle, "qwen3vl_pooler_z1_text_outer_fast_output")
        ->qwen3vl_pooler_z1_outer_fast_output();
}

at::Tensor forward_text_z1(
    int64_t handle,
    const at::Tensor& hidden_shape_carrier,
    at::TensorList k_caches,
    at::TensorList v_caches,
    const at::Tensor& position_ids,
    uint64_t epoch,
    uint64_t plan_hash) {
    auto* model = CausalDecoderRegistry::get(
        handle, "qwen3vl_pooler_z1_forward_text");
    std::vector<at::Tensor> kc(k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> vc(v_caches.begin(), v_caches.end());
    return model->forward_qwen3vl_pooler_z1(
        hidden_shape_carrier, kc, vc, position_ids, epoch, plan_hash);
}

void check_text_destroy_allowed(int64_t handle) {
    CausalDecoderRegistry::get(
        handle, "qwen3vl_pooler_z1_check_text_destroy_allowed")
        ->check_qwen3vl_pooler_z1_destroy_allowed();
}

}  // namespace v3::qwen3vl_pooler_z1_internal

// =============================================================================
// Public C API for TORCH_LIBRARY_IMPL wrappers (file-scope, not namespaced)
// =============================================================================

int64_t rpu_causal_decoder_create() {
    return CausalDecoderRegistry::create();
}

void rpu_causal_decoder_destroy(int64_t handle) {
    v3::wall_oss_z1_internal::check_text_destroy_allowed(handle);
    v3::qwen3vl_pooler_z1_internal::check_text_destroy_allowed(handle);
    CausalDecoderRegistry::destroy(handle, "rpu_causal_decoder_destroy");
}

void rpu_causal_decoder_set_weights(
    int64_t handle,
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
    at::IntArrayRef mrope_section,
    at::IntArrayRef deepstack_lang_layers,
    const std::optional<std::vector<at::Tensor>>& q_bias_list,
    const std::optional<std::vector<at::Tensor>>& k_bias_list,
    const std::optional<std::vector<at::Tensor>>& v_bias_list)
{
    // Optional Tensor[] schema args arrive as std::optional<vector>; unwrap to
    // TensorList (empty when None) for the model's set_weights.
    at::TensorList q_bias = q_bias_list ? at::TensorList(*q_bias_list) : at::TensorList{};
    at::TensorList k_bias = k_bias_list ? at::TensorList(*k_bias_list) : at::TensorList{};
    at::TensorList v_bias = v_bias_list ? at::TensorList(*v_bias_list) : at::TensorList{};
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        q_norm_list, k_norm_list,
        input_norm_list, post_norm_list,
        gate_list, up_list, down_list,
        cos, sin, final_norm_w,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size,
        eps, use_silu,
        mrope_section,
        deepstack_lang_layers,
        q_bias, k_bias, v_bias,
        {}, {}, {}, {}, {}, {}, {});
}

void rpu_causal_decoder_set_weights_w8a16(
    int64_t handle,
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
    at::IntArrayRef mrope_section,
    at::IntArrayRef deepstack_lang_layers,
    at::TensorList q_w_scale_list,
    at::TensorList k_w_scale_list,
    at::TensorList v_w_scale_list,
    at::TensorList o_w_scale_list,
    at::TensorList gate_scale_list,
    at::TensorList up_scale_list,
    at::TensorList down_scale_list,
    const std::optional<std::vector<at::Tensor>>& q_bias_list,
    const std::optional<std::vector<at::Tensor>>& k_bias_list,
    const std::optional<std::vector<at::Tensor>>& v_bias_list)
{
    at::TensorList q_bias = q_bias_list ? at::TensorList(*q_bias_list) : at::TensorList{};
    at::TensorList k_bias = k_bias_list ? at::TensorList(*k_bias_list) : at::TensorList{};
    at::TensorList v_bias = v_bias_list ? at::TensorList(*v_bias_list) : at::TensorList{};
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        q_norm_list, k_norm_list,
        input_norm_list, post_norm_list,
        gate_list, up_list, down_list,
        cos, sin, final_norm_w,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size,
        eps, use_silu,
        mrope_section,
        deepstack_lang_layers,
        q_bias, k_bias, v_bias,
        q_w_scale_list, k_w_scale_list, v_w_scale_list, o_w_scale_list,
        gate_scale_list, up_scale_list, down_scale_list);
}

void rpu_causal_decoder_set_weights_nvfp4(
    int64_t handle,
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
    at::IntArrayRef mrope_section,
    at::IntArrayRef deepstack_lang_layers,
    at::TensorList q_w_scale_list, at::TensorList k_w_scale_list,
    at::TensorList v_w_scale_list, at::TensorList o_w_scale_list,
    at::TensorList gate_scale_list, at::TensorList up_scale_list,
    at::TensorList down_scale_list,
    const at::Tensor& q_tensor_scales, const at::Tensor& k_tensor_scales,
    const at::Tensor& v_tensor_scales, const at::Tensor& o_tensor_scales,
    const at::Tensor& gate_tensor_scales, const at::Tensor& up_tensor_scales,
    const at::Tensor& down_tensor_scales,
    const std::optional<std::vector<at::Tensor>>& q_bias_list,
    const std::optional<std::vector<at::Tensor>>& k_bias_list,
    const std::optional<std::vector<at::Tensor>>& v_bias_list)
{
    at::TensorList q_bias = q_bias_list ? at::TensorList(*q_bias_list) : at::TensorList{};
    at::TensorList k_bias = k_bias_list ? at::TensorList(*k_bias_list) : at::TensorList{};
    at::TensorList v_bias = v_bias_list ? at::TensorList(*v_bias_list) : at::TensorList{};
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        q_norm_list, k_norm_list,
        input_norm_list, post_norm_list,
        gate_list, up_list, down_list,
        cos, sin, final_norm_w,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size,
        eps, use_silu,
        mrope_section, deepstack_lang_layers,
        q_bias, k_bias, v_bias,
        q_w_scale_list, k_w_scale_list, v_w_scale_list, o_w_scale_list,
        gate_scale_list, up_scale_list, down_scale_list,
        q_tensor_scales, k_tensor_scales, v_tensor_scales, o_tensor_scales,
        gate_tensor_scales, up_tensor_scales, down_tensor_scales);
}

// Phase 2.5: per-instance fused lm_head 入口
void rpu_causal_decoder_set_lm_head(
    int64_t handle,
    const at::Tensor& lm_head_w,
    const std::optional<at::Tensor>& lm_head_scale) {
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder")
        ->set_lm_head(lm_head_w, lm_head_scale);
}

void rpu_causal_decoder_clear_lm_head(int64_t handle) {
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder")->clear_lm_head();
}

// RhinoVLA text-prefill fast-replay opt-ins (per-model).
void rpu_causal_decoder_set_fast_replay(int64_t handle, bool enabled) {
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder")
        ->set_fast_replay_skip_layer_loop(enabled);
}

void rpu_causal_decoder_set_preload_replay_skip(int64_t handle, bool enabled) {
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder")
        ->set_preload_replay_skip(enabled);
}

void rpu_causal_decoder_set_chunk_size_cap(int64_t handle, int64_t chunk_size_cap) {
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder_set_chunk_size_cap")
        ->set_configured_chunk_size_cap(chunk_size_cap);
}

// Per-handle replacement for the deleted process-global set_chunk_size.
//
// Unlike the cap, the override may change on a live handle. The resolved chunk
// is part of the prefill Graph signature, so a change creates a distinct cache
// entry. It must not invalidate other signatures.
void rpu_causal_decoder_set_chunk_size_override(int64_t handle, int64_t chunk_size) {
    TORCH_CHECK(chunk_size == 0
                    || (chunk_size >= 16 && chunk_size % 16 == 0),
                "causal_decoder_set_chunk_size_override: chunk_size must be 0 "
                "(auto) or a positive multiple of 16, got ", chunk_size);
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder_set_chunk_size_override")
        ->set_chunk_size_override(chunk_size);
}

int64_t rpu_causal_decoder_get_chunk_size_override(int64_t handle) {
    return CausalDecoderRegistry::get(
               handle, "rpu_causal_decoder_get_chunk_size_override")
        ->get_chunk_size_override();
}

// Declare this handle's certified chunk envelope. Without it the planner
// refuses to prefill.
void rpu_causal_decoder_set_chunk_envelope(int64_t handle, int64_t max_kv_len,
                                           int64_t chunk) {
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder_set_chunk_envelope")
        ->set_chunk_envelope(max_kv_len, chunk);
}

void rpu_causal_decoder_set_equal_two_prefill(int64_t handle, bool enabled) {
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder_set_equal_two_prefill")
        ->set_equal_two_prefill(enabled);
}

// AdaRMS FiLM step-update (LingBot-VLA action expert). Per Euler denoise step,
// the host passes the folded per-layer scale (= (1+γ(cond))·rms_weight, rides on
// the input/post norm-weight slots) + shift (= β(cond)). See rpu_qwen3_model.h
// set_adarms_step. Enables the gated shift-add in build_layer_subgraph.
void rpu_causal_decoder_set_adarms_step(
    int64_t handle,
    at::TensorList input_scales, at::TensorList input_shifts,
    at::TensorList post_scales,  at::TensorList post_shifts) {
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder_set_adarms_step")
        ->set_adarms_step(input_scales, input_shifts, post_scales, post_shifts);
}

// Replay-safe variant: each arg is one [num_layers, hidden] fp16 RPU tensor (not a list).
// First call builds the mutable graph; subsequent calls refresh keepalives in place (no
// rebuild) so the expert graph replays across Euler steps. See rpu_qwen3_model.h.
void rpu_causal_decoder_set_adarms_step_mutable(
    int64_t handle,
    const at::Tensor& input_scale, const at::Tensor& input_shift,
    const at::Tensor& post_scale,  const at::Tensor& post_shift) {
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder_set_adarms_step_mutable")
        ->set_adarms_step_mutable(input_scale, input_shift, post_scale, post_shift);
}

at::Tensor rpu_causal_decoder_forward(
    int64_t handle,
    const at::Tensor& hidden_states,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position,
    bool is_causal,
    const std::optional<at::Tensor>& position_ids,
    const std::optional<std::vector<at::Tensor>>& deepstack_dense_visual_embeds,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il,
    int64_t cos_sin_offset,
    int64_t batch_slot,
    bool allow_batch_decode)
{
    // TensorList → std::vector<at::Tensor> (shallow copy of refcounted tensors)
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());

    return CausalDecoderRegistry::get(handle, "rpu_causal_decoder")->forward(
        hidden_states, k_caches, v_caches, attention_mask, position, is_causal,
        position_ids, deepstack_dense_visual_embeds, rope_cos_il, rope_sin_il,
        cos_sin_offset, batch_slot, allow_batch_decode);
}

// Per-handle resolved chunk_size.
// Returns 0 if no forward has run yet for this handle (sentinel default).
int64_t rpu_causal_decoder_get_resolved_chunk_size(int64_t handle) {
    return CausalDecoderRegistry::get(handle, "rpu_causal_decoder_get_resolved_chunk_size")
        ->get_last_resolved_chunk_size();
}

int64_t rpu_causal_decoder_resolve_prefill_chunk_size(
    int64_t handle, int64_t execution_len, int64_t position) {
    return CausalDecoderRegistry::get(
        handle, "rpu_causal_decoder_resolve_prefill_chunk_size")
        ->resolve_prefill_chunk_size(execution_len, position);
}
