from __future__ import annotations
from pathlib import Path
import re
ROOT = Path(__file__).resolve().parents[4]
VISION_CPP = ROOT / 'src' / 'fused' / 'rpu_qwen25vl_vision_model.cpp'
DECODER_CPP = ROOT / 'src' / 'fused' / 'rpu_qwen3_model.h'
FUSED_BASE_CPP = ROOT / 'src' / 'core' / 'fused_model_base.cpp'
LINEAR_CPP = ROOT / 'src' / 'ops' / 'rpu_linear.cpp'
RMS_CPP = ROOT / 'src' / 'ops' / 'rpu_rmsnorm.cpp'
MEMCPY_CPP = ROOT / 'src' / 'ops' / 'rpu_memcpy.cpp'
ALL_GATHER_CPP = ROOT / 'src' / 'ops' / 'rpu_all_gather.cpp'
REDUCE_CPP = ROOT / 'src' / 'core' / 'rpu_runtime_extras.cpp'
CACHE_H = ROOT / 'src' / 'core' / 'rpu_kernel_cache.h'
CACHE_INC = ROOT / 'src' / 'core' / 'rpu_kernel_cache.inc'
KERNEL_DECLS_H = ROOT / 'src' / 'core' / 'rpu_kernel_decls.h'
VISION_PY = ROOT / 'python' / 'rpu_backend' / 'adapters' / 'wall_oss' / 'vision.py'
LLM_PY = ROOT / 'python' / 'rpu_backend' / 'adapters' / 'wall_oss' / 'llm.py'

def test_block17_transition_and_bf16_residual_state_are_explicit():
    source = VISION_CPP.read_text(encoding='utf-8')
    vision_py = VISION_PY.read_text(encoding='utf-8')
    resolver_start = source.index('static VisionDispatch resolve_vision_dispatch(')
    resolver = source[resolver_start:source.index('class Qwen25VLVisionModel')]
    assert resolver.count('VisionSchedule::MIMC') == 2
    assert resolver.count('VisionSchedule::MISC') == 1
    qkv_start = source.index('void emit_exact_mimc_qkv_body(')
    compute_start = source.index('void build_exact_mimc_compute_body(')
    normal_start = source.index('void build_layer_subgraph(', compute_start)
    qkv = source[qkv_start:compute_start]
    compute = source[compute_start:normal_start]
    packed = qkv + compute
    normal = source[normal_start:]
    assert 'QWEN25VL_VISION_MIXED_BOUNDARY_LAYER = 17' in source
    assert 'layer_idx > QWEN25VL_VISION_MIXED_BOUNDARY_LAYER' in source
    assert 'layer_idx == QWEN25VL_VISION_MIXED_BOUNDARY_LAYER' in source
    for body in (packed, normal):
        assert 'rpu_launch_linear_spm_to_spm_acc32_out_bf16_kernel(' in body
    assert packed.count('rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel(') == 1
    assert normal.count('rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel(') >= 2
    assert 'rpu_launch_all_reduce_bf16_partial_fp16_residual_spm_kernel(' in normal
    assert normal.count('rpu_launch_all_reduce_fp16_partial_bf16_residual_spm_kernel(') >= 2
    assert 'rpu_launch_wall_vision_all_reduce_bf16_partial_fp16_residual_m576_inplace_kernel(' in packed
    assert 'rpu_launch_wall_vision_all_reduce_fp16_partial_bf16_residual_m576_inplace_kernel(' in packed
    assert 'rpu_launch_wall_vision_all_reduce_sum_residual_m576_inplace_kernel(' in packed
    assert 'rpu_launch_all_reduce_fp16_partial_bf16_residual_local_spm_kernel(' in packed
    assert 'rpu_launch_all_reduce_sum_residual_local_spm_kernel(' in packed
    assert 'rpu_launch_all_reduce_fp16_partial_bf16_residual_ddr_kernel(' not in packed
    for required in ('enum class VisionSchedule', 'enum class VisionImplementation', 'resolve_vision_dispatch', 'VisionSchedule::MIMC, VisionImplementation::ExactByMha', 'VisionImplementation::LegacyLayerGroup', '!is_legacy_layer_group()', 'fused_merger_enabled_ && merger_out_hidden_ > 0', '!is_exact_mimc() || hybrid_w8a16', 'struct ExactMimcPlan', 'kQkvRows = 1728', 'kAttentionRows = 576', 'kMlpRows = 576', 'num_layers() == 32 && hidden_size() == 1280', 'num_q_heads() == 16 && head_dim() == 80', 'const bool hybrid_w8a16 = intermediate_size() == 3584', 'i == QWEN25VL_VISION_MIXED_BOUNDARY_LAYER', '? at::kHalf : at::kChar', '!get_debug_export()'):
        assert required in source
    for required in ('const int64_t packed_seq = ExactMimcPlan::kQkvRows', 'const at::Tensor& scale', 'lw.q_ws', 'lw.o_ws', 'lw.gate_ws', 'lw.down_ws', 'addr(0, "o_result")', 'addr(0, "compact_residual")', 'rpu_launch_spm_local_shard_copy_kernel(', 'rpu_launch_spm_local_copy_dma(', 'ExactMimcPlan::kResidualShardElements', 'ExactMimcPlan::kResidualShardBytes', 'rpu_launch_all_reduce_sum_residual_local_spm_kernel(', 'rpu_launch_all_reduce_fp16_partial_bf16_residual_local_spm_kernel(', 'rpu_launch_wall_vision_all_reduce_bf16_partial_fp16_residual_m576_inplace_kernel(', 'rpu_launch_wall_vision_all_reduce_fp16_partial_bf16_residual_m576_inplace_kernel(', 'rpu_launch_wall_vision_all_reduce_sum_residual_m576_inplace_kernel(', 'rpu_launch_wall_down_postnorm_m576_fp16_partial_fp16_residual_kernel(', 'rpu_launch_wall_down_postnorm_m576_bf16_partial_fp16_residual_kernel(', 'rpu_launch_wall_down_postnorm_m576_fp16_partial_bf16_residual_kernel(', 'mlp_rows, h, is_, /*row=*/0, NUM_CORES'):
        assert required in packed
    assert qkv.count('rpu_launch_v_transpose_spm(') == 2
    assert qkv.count('rpu_launch_sdpa_by_mha_spm(') == 2
    assert qkv.count('rpu_launch_wall_window_sdpa_by_mha_spm(') == 1
    assert 'AttentionExecutionPolicy::SPM_KV_BY_MHA' in source
    assert 'subclass_spm_kv_by_mha_eligible(' in source
    sdpa = (ROOT / 'src' / 'ops' / 'rpu_sdpa_vctxlen.cpp').read_text(encoding='utf-8')
    assert 'q_off != output_off' in sdpa
    assert 'requires distinct Q/output buffers' in sdpa
    assert 'rpu_launch_insert_kcache_spm_unified(' not in packed
    assert 'rpu_launch_insert_vcache_spm_unified(' not in packed
    assert 'rpu_launch_sdpa_spm_unified_kernel_v2(' not in packed
    assert 'emit_layer_chain_scratch_store(' not in packed
    assert 'stable_layer_input_ddr_ptr(' not in packed
    assert 'layer_chain_scratch_ddr_ptr(' not in packed
    assert compute.count('emit_layer_output_dma(') == 1
    assert 'attn_scratch, mask_addr, mask_type, attn_scale' in packed
    assert 'const uint32_t attn_scratch = addr(0, "input_norm")' in packed
    assert '+ packed_h_bytes - packed_q_bytes' in packed
    assert 'attention_output, lw.o_w, addr(0, "oproj")' in packed
    assert 'rpu_launch_spm_scatter_spm_dma(' not in packed
    assert 'merger_active() ? 15 : 9' in source
    assert '{"compact_residual"' in source
    assert 'w8_direct_sdpa_output ? 3 : 4' in source
    assert 'w8_direct_sdpa_output ? 4 : 3' in source
    assert '"gate",         compact_local_inout' in source
    assert '? ExactMimcPlan::kImages * mlp_inter' in source
    assert 'if (!fused_w8_gate_up)' in source
    assert '{"up", mlp_inter, 7, 7,' in source
    assert 'fused_w8_gate_up ? 1 : 3' in source
    assert 'fused_w8_gate_up ? 9 : 3' in source
    assert 'packed_qkv,   1, 9,' in source
    assert 'const int64_t mlp_begin = chunk.offset;' in compute
    assert 'if (chunk.idx == 0) {' in compute
    assert 'exact_mimc_w8_qkv_enabled()\n                    ? addr(0, "v") : addr(0, "q")' in compute
    assert 'attention_output, lw.o_w, addr(0, "oproj")' in compute
    assert qkv.count('rpu_launch_spm_local_copy_dma(') == 1
    assert 'rpu_launch_spm_local_shard_copy_dma(' not in qkv
    assert 'addr(0, "v"), mask_addr, mask_type, attn_scale' in qkv
    assert 'exact_mimc_native_window_sdpa_enabled()' in qkv
    assert 'if (native_window_sdpa && !is_full)' in qkv
    assert 'if (!is_full && !native_window_sdpa)' in qkv
    assert 'fullatt_block_indexes_ ==' in source
    assert 'std::vector<int64_t>({7, 15, 23, 31})' in source
    selector_start = source.index('bool exact_mimc_native_window_sdpa_enabled() const')
    selector = source[selector_start:source.index('bool merger_active()', selector_start)]
    assert 'exact_mimc_fused_w8_gate_up_enabled()' in selector
    assert 'window_mask_present_' in selector
    assert 'std::vector<int64_t>({7, 15, 23, 31})' in selector
    assert 'cu_window_ == std::vector<int64_t>({' in selector
    for boundary_group in ('0, 64, 128, 192, 256, 320, 384', '448, 512, 528, 544, 560, 576'):
        assert boundary_group in selector
    assert source.count('exact_mimc_w8_qkv_enabled()') >= 7
    assert source.count('exact_mimc_fused_w8_gate_up_enabled()') >= 6
    assert 'lw.gate_ws.defined() && lw.up_ws.defined()' in source
    params_start = source.index('void emit_exact_mimc_layer_params(')
    params = source[params_start:qkv_start]
    assert 'if (layer_idx == 0) {' in params
    assert 'bn.norm1_w.data_ptr<c10::Half>(), h, addr(0, "norm1_w")' in params
    phase1 = qkv[qkv.index('// Phase 1:'):qkv.index('// Phase 2:')]
    assert phase1.count('rpu_launch_spm_local_shard_copy_kernel(') == 1
    assert phase1.count('rpu_launch_rmsnorm_spm_kernel(') == 1
    assert 'rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel(' not in phase1
    assert 'const bool has_next_layer = layer_idx + 1 < num_layers();' in compute
    assert 'if (has_next_layer && chunk.idx == 0)' in compute
    assert 'next_bn.norm1_w.data_ptr<c10::Half>()' in compute
    assert 'addr(0, "norm1_w"), eps_' in compute
    assert compute.index('if (has_next_layer) {') < compute.index('rpu_launch_wall_vision_all_reduce_bf16_partial_fp16_residual_m576_inplace_kernel(')
    assert compute.index('rpu_launch_all_reduce_sum_residual_local_spm_kernel(') < compute.index('rpu_launch_wall_vision_gate_up_swiglu_w8a16_spm_kernel(')
    assert 'cfg.kv_first_fn' in source
    assert '&Qwen25VLVisionModel::emit_exact_mimc_qkv_body' in source
    assert 'cfg.kv_first_chunk_plan_fn' in source
    assert '&Qwen25VLVisionModel::plan_exact_mimc_qkv_chunks' in source
    assert 'return {ExactMimcPlan::kQkvRows, 1};' in source
    assert 'cfg.chunk_mode = ChunkMode::KV_FIRST;' in source
    exact_decl_start = source.index('if (is_exact_mimc()) {', source.index('declare_buffers('))
    exact_decls = source[exact_decl_start:source.index('return decls;', exact_decl_start)]
    assert 'if (!fused_w8_gate_up)' in exact_decls
    assert 'fused_w8_gate_up ? 1 : 3' in exact_decls
    assert 'fused_w8_gate_up ? 9 : 3' in exact_decls
    assert 'if (window_mask_present_ && !native_window_sdpa)' in exact_decls
    assert '{"sdpa_out"' not in exact_decls
    assert '"down_partial"' not in exact_decls
    assert '"down_partial"' not in compute
    assert '{"merger_input"' not in source
    assert 'layer_idx + 1 == num_layers() && !merger_active()' in packed
    assert 'merger_input_addr = addr(0, "input_norm")' in source
    assert 'is_exact_mimc() ? merger_input_addr : 0' in source
    assert 'rpu_launch_spm_copy_ddr_dma_mutable(\n                merger_out_addr' in source
    dynamic_start = source.index('ModelDynamicConfig dynamic_config(')
    dynamic = source[dynamic_start:source.index('declare_buffers(', dynamic_start)]
    exact_dynamic = dynamic[dynamic.index('if (is_exact_mimc())'):dynamic.index('if (is_legacy_layer_group())')]
    assert 'InterLayerIO::SPM_RESIDENT' in exact_dynamic
    assert 'InterLayerIO::DDR_PINGPONG' not in exact_dynamic
    shard_elements = 576 * 1280 // 8
    compact_bytes = 3 * shard_elements * 2
    assert shard_elements == 92160
    assert compact_bytes == 552960
    mask_bytes = 576 * 576 * 2
    unused_up_bytes = 576 * 448 * 2
    assert mask_bytes == 663552
    assert unused_up_bytes == 516096
    assert mask_bytes - unused_up_bytes == 147456
    assert 7630848 + 591360 == 8222208
    assert 8303616 - 8222208 == 81408
    assert 6967296 + 591360 == 7558656
    assert 8303616 - 7558656 == 744960
    assert 6635520 + 591360 == 7226880
    assert 8303616 - 7226880 == 1076736
    full_layers = {7, 15, 23, 31}
    assert len(full_layers) == 4
    assert 32 - len(full_layers) == 28
    native_window_mask_uploads = 0
    assert native_window_mask_uploads == 0
    for required in ('kCompactResidualBytesPerCore =', "kValidatedTempPeakBytes = 7'630'848", "kNativeWindowTempPeakBytes = 6'967'296", "kNativeWindowLocalInoutTempPeakBytes = 6'635'520", 'is_ == 3456 && local_inter == 432', 'is_ == 3584 && local_inter == 448', "kFullVlaSuperPersistentBytes = 591'360", 'detail::estimate_temporary_total(decls)', 'temp_peak <= ExactMimcPlan::kValidatedTempPeakBytes', 'temp_peak <= ExactMimcPlan::kNativeWindowLocalInoutTempPeakBytes'):
        assert required in source
    boundary_parse_start = source.index('const bool has_window_boundaries')
    boundary_parse = source[boundary_parse_start:source.index('if (mixed_precision_)', boundary_parse_start)]
    assert '!per_window_sdpa_enabled_ || has_window_boundaries' in boundary_parse
    assert '!has_window_boundaries || per_window_sdpa_enabled_ ||' in boundary_parse
    assert 'is_exact_mimc()' in boundary_parse
    assert 'num_patches_in / image_batch_count_' in boundary_parse
    assert 'boundaries[cu.numel() - 1] == boundary_seq' in boundary_parse
    assert 'cu_window_.assign(boundaries, boundaries + cu.numel())' in boundary_parse
    assert 'return window_index_multi, reverses, mask, cu0' in vision_py
    group_start = vision_py.index('def _forward_group(')
    group = vision_py[group_start:vision_py.index('def build_wall_oss_vision(', group_start)]
    assert 'self._select_native_window_boundaries(' in group
    assert 'graph_dyn_dims = _vision_graph_dyn_dims(' in group
    assert 'graph_dyn_dims.extend(a6_plan.graph_key_words())' in group
    assert 'dyn_dims=graph_dyn_dims' in group
    assert 'native_window_boundaries,' in group
    preload_start = source.index('void emit_preload_weights()')
    preload_end = source.index('void emit_exact_mimc_layer_params(', preload_start)
    preload = source[preload_start:preload_end]
    assert 'exact_mimc_fused_w8_gate_up_enabled()' in preload
    assert preload.count('sdpa_dma_mask_to_spm(') == 4
    assert '!exact_mimc_native_window_sdpa_enabled()' in preload
    assert 'exact_mimc_native_window_sdpa_enabled() ? 1 : 0' in source
    assert 'ExactMimcPlan::kAttentionRows' in preload
    assert 'if (!exact_mimc_fused_w8_gate_up_enabled())' in qkv
    reduce = REDUCE_CPP.read_text(encoding='utf-8')
    assert 'residual_base = static_cast<uint32_t>(residual_dev_addr >> 8)' in reduce
    assert 'residual_local_shard ? 2 : (residual_ddr != nullptr)' in reduce
    declarations = KERNEL_DECLS_H.read_text(encoding='utf-8')
    for launcher in ('rpu_launch_wall_vision_all_reduce_sum_residual_m576_inplace_kernel', 'rpu_launch_wall_vision_all_reduce_bf16_partial_fp16_residual_m576_inplace_kernel', 'rpu_launch_wall_vision_all_reduce_fp16_partial_bf16_residual_m576_inplace_kernel', 'rpu_launch_wall_down_postnorm_m576_fp16_partial_fp16_residual_kernel', 'rpu_launch_wall_down_postnorm_m576_bf16_partial_fp16_residual_kernel', 'rpu_launch_wall_down_postnorm_m576_fp16_partial_bf16_residual_kernel'):
        assert f'void {launcher}(' in reduce
        assert f'void {launcher}(' in declarations
    assert 'exact_m576_postnorm_id' in reduce
    assert 'input_spm == output_spm' in reduce
    assert 'distinct full residual, compact raw, and gamma' in reduce
    memcpy = MEMCPY_CPP.read_text(encoding='utf-8')
    shard_start = memcpy.index('void rpu_launch_spm_local_shard_copy_dma(')
    shard_end = memcpy.index('void rpu_launch_spm_local_copy_dma(', shard_start)
    shard_copy = memcpy[shard_start:shard_end]
    assert 'src_off + static_cast<uint64_t>(c) * core_stride_bytes' in shard_copy
    assert 'SPM_ALLOC.addr(c, static_cast<uint32_t>(dst_off))' in shard_copy
    assert 'src0' not in shard_copy
    all_gather = ALL_GATHER_CPP.read_text(encoding='utf-8')
    local_copy_start = all_gather.index('void rpu_launch_spm_local_shard_copy_kernel(')
    local_copy_end = all_gather.index('// =============================================================================\n// rpu_all_gather_test', local_copy_start)
    local_copy = all_gather[local_copy_start:local_copy_end]
    for required in ('KernelId::ALL_GATHER_MULTI_CORE', 'GraphKernelNoDdrProof::AllGatherSpm', 'kernel->set_regs(0, static_cast<uint16_t>(1))', 'kernel->set_regs(1, static_cast<uint16_t>(1))', 'src_off + static_cast<uint64_t>(j) * core_stride_bytes', 'S + 56 + j * 16 + j', 'SPM_ALLOC.addr(j, static_cast<uint32_t>(dst_off))', 'queue->set_broadcast_mode(true)', 'overlapping same-core copy is unsupported'):
        assert required in local_copy
    assert 'KernelId::ALL_GATHER_MULTI_CORE_LITTLE_CHUNK' not in local_copy
    assert 'graph_dma::' not in local_copy

def test_python_signature_and_shape_gate_include_mixed_mode():
    source = VISION_PY.read_text(encoding="utf-8")
    # Read once; the reduced-core refusal list may name it again.
    assert source.count('rpu_env_bool(\n        "RPU_WALL_OSS_VISION_MIXED_PRECISION"') == 1
    assert "qwen25vl_vision_set_execution_routes" in source
    assert source.count("int(self._mixed_precision)") == 2
    assert "any(n != 576 for n in counts)" in source
    assert "max_seq = min(max_seq, 576)" in source
    assert "mixed_w8a16 and L == 17" in source
    assert 'down_key = f"{p}.mlp.down_proj.weight"' in source
    assert 'cpu_st.get_tensor(down_key)' in source
    assert 'check_padding=False' in source

def test_decoder_layer2_transition_and_bf16_residual_state_are_explicit():
    decoder = DECODER_CPP.read_text(encoding='utf-8')
    llm = LLM_PY.read_text(encoding='utf-8')
    base = FUSED_BASE_CPP.read_text(encoding='utf-8')
    quant_check = decoder.split('auto check_quant_pair =', 1)[1].split('auto check_nvfp4_tensor_scales', 1)[0]
    assert 'weights[i].scalar_type() == at::kHalf' in quant_check
    assert '!has_nvfp4' in quant_check
    assert 'weights[i].scalar_type() == at::kChar' in quant_check
    assert 'weights[i].scalar_type() == at::kByte' in quant_check
    assert 'std::getenv("RPU_WALL_OSS_PREFILL_MIXED_PRECISION")' not in decoder
    assert 'std::getenv("RPU_WALL_OSS_PREFILL_STAGED_MLP")' not in decoder
    assert 'wall_prefill_mixed_requested_' in decoder
    assert 'wall_prefill_staged_requested_' in decoder
    assert 'causal_decoder_set_wall_prefill_routes' in llm
    assert 'N == 36 && hidden_size == 2048 && intermediate_size == 11008' in decoder
    assert 'num_kv_heads == 8' in decoder
    assert 'const bool hybrid_w8a16 = has_scale && !has_nvfp4' in decoder
    assert 'const bool hybrid_w4a16 = has_scale && !has_nvfp4' in decoder
    assert 'lw.q_w.scalar_type() != at::kByte' in decoder
    assert 'i == 2 ? at::kHalf : at::kChar' in decoder
    assert 'seq_len_ == 672 && seq_len == 336' in decoder
    assert 'sdpa_call_valid(mixed_cfg, 336, 0)' in decoder
    assert 'sdpa_call_valid(mixed_cfg, 112, 336)' in decoder
    assert 'sdpa_call_valid(mixed_cfg, 224, 448)' in decoder
    assert 'chunk.offset == 336' in decoder
    assert 'constexpr int64_t kFirstRows = 112' in decoder
    assert 'constexpr int64_t kSecondRows = 224' in decoder
    assert 'layer_idx > 2' in decoder
    assert 'layer_idx == 2' in decoder
    assert decoder.count('rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel(') == 4
    assert 'rpu_launch_all_reduce_fp16_partial_bf16_residual_spm_kernel(' in decoder
    assert 'rpu_launch_wall_ring_postnorm_fp16_partial_bf16_residual_spm_kernel(' in decoder
    assert 'rpu_launch_wall_ring_postnorm_fp16_partial_bf16_residual_local_spm_kernel(' in decoder
    assert 'build_wall_p640_staged_layer_subgraph' in decoder
    assert 'rpu_launch_wall_prefill_gate_up_swiglu_w8a16_spm_kernel(' in decoder
    assert 'addr(0, "gate"), addr(0, "down"), kPackedRows' in decoder
    assert 'rpu_launch_spm_local_shard_copy_dma(' in decoder
    assert 'rpu_launch_all_reduce_sum_residual_local_spm_kernel(' in decoder
    staged_start = decoder.index('void build_wall_p640_staged_layer_subgraph(')
    staged = decoder[staged_start:decoder.index('void build_layer_subgraph(', staged_start)]
    decoder_flat = ' '.join(decoder.split())
    staged_flat = ' '.join(staged.split())
    assert 'decls.push_back({"input_norm", work_res, 1, 8,' in decoder_flat
    assert 'decls.push_back({"residual_local", A(work_res / NUM_CORES), 1, 8,' in decoder_flat
    assert 'A(2 * work_res / NUM_CORES)' not in decoder_flat
    assert 'addr(0, "residual_local")' in staged
    assert '!transition_to_bf16 && half == 0' in staged
    assert staged.count('rpu_launch_spm_local_shard_copy_dma(') == 1
    assert 'rpu_launch_spm_local_copy_dma(' not in staged
    assert 'if (half == 0)' in staged
    assert 'rpu_launch_all_reduce_sum_residual_kernel(' in staged
    raw_select = 'const uint32_t raw_residual_local_spm = half == 0 ? addr(0, "residual_local") : addr(0, "input_norm");'
    assert staged_flat.count(raw_select) == 2
    assert 'const bool normalized_compact_carry = layer_idx > 3;' in staged
    assert 'if (!normalized_compact_carry)' in staged
    assert 'rpu_launch_all_reduce_fp16_partial_bf16_residual_local_spm_kernel(' not in staged
    assert 'const uint32_t partial_spm = down_weight_outer && half == 1 ? hidden_half : addr(0, "down");' in staged_flat
    assert 'rpu_launch_wall_ring_postnorm_fp16_partial_bf16_residual_local_spm_kernel( partial_spm, raw_residual_local_spm, hidden_half, next_norm_w, eps_);' in staged_flat
    assert staged.count('rpu_launch_wall_ring_postnorm_pair_fp16_partial_bf16_residual_local_spm_kernel(') == 2
    assert 'layer_addr(layer_idx + 1, 0, "norm_w")' in staged
    assert 'addr(0, "final_norm_w")' in staged
    assert 'rpu_launch_wall_ring_postnorm_fp16_partial_bf16_residual_spm_kernel( addr(0, "oproj"), hidden_half, raw_residual_local_spm, layer_addr(layer_idx, 0, "post_norm_w"), eps_);' in staged_flat
    assert 'rpu_launch_wall_ring_postnorm_fp16_partial_bf16_residual_local_spm_kernel( addr(0, "oproj"), raw_residual_local_spm, hidden_half, layer_addr(layer_idx, 0, "post_norm_w"), eps_);' in staged_flat
    assert 'rpu_launch_all_reduce_fp16_partial_bf16_residual_spm_kernel(' not in staged
    assert 'rpu_launch_all_reduce_bf16_partial_fp16_residual_spm_kernel( addr(0, "down"), addr(0, "input_norm"), hidden_half,' in staged_flat
    transition_start = staged.index('if (transition_to_bf16) {')
    transition_end = staged.index('if (normalized_compact_carry) {', transition_start)
    transition = staged[transition_start:transition_end]
    assert 'rpu_launch_all_reduce_bf16_partial_fp16_residual_spm_kernel(' in transition
    assert 'rpu_launch_wall_ring_postnorm' not in transition
    assert staged.index('rpu_launch_wall_qkv_w8a16_multibase_spm_kernel(') < staged.index('rpu_launch_wall_ring_postnorm_fp16_partial_bf16_residual_spm_kernel(')
    half1_call = staged.index('c10::nullopt, q_half1_off, output_half1_off,')
    half0_call = staged.index('c10::nullopt, addr_offset("q").value, q_half1_off,')
    oproj_call = staged.index('rpu_launch_wall_prefill_oproj_w8a16_c320x2_spm_kernel(')
    packed_gate = staged.index('rpu_launch_wall_prefill_gate_up_swiglu_w8a16_spm_kernel(', oproj_call)
    down_call = staged.index('rpu_launch_wall_prefill_down_w8a16_c320x2_spm_kernel(', packed_gate)
    assert half1_call < half0_call < oproj_call < packed_gate < down_call
    assert 'addr(0, "input_norm")' not in staged[packed_gate:down_call]
    resident_start = decoder.index('// Exact Wall RTC prefill stays SPM-resident')
    resident = decoder[resident_start:decoder.index('if (qwen3vl_pooler_z1_dispatch_)', resident_start)]
    assert 'decoder_mixed_precision_ && ctx().is_causal' in resident
    assert 'ctx().position == 0' in resident
    assert set(re.findall('ctx\\(\\)\\.seq_len == (\\d+)', resident)) == {'640', '672'}
    assert set(re.findall('ctx\\(\\)\\.seq_len == (\\d+)\\s*&&\\s*plan\\.chunk_size == (\\d+)', resident)) == {('640', '320'), ('672', '336')}
    assert resident.count('plan.num_chunks == 2') == 2
    assert 'cfg.chunk_mode = ChunkMode::SEQUENTIAL' in resident
    assert 'cfg.inter_layer_io = InterLayerIO::SPM_RESIDENT' in resident
    assert '!(wall_prefill_staged_mlp_ && ctx().seq_len == 640)' in resident
    assert 'if (down_out_bf16)' in base
    assert 'if (residual_is_bf16)' in base
    assert 'rpu_launch_linear_spm_to_spm_acc32_out_bf16_kernel(' in base
    assert 'rpu_launch_all_reduce_bf16_partial_fp16_residual_spm_kernel(' in base
    assert 'rpu_launch_all_reduce_fp16_partial_bf16_residual_spm_kernel(' in base

def test_python_prefill_admission_and_signature_include_mixed_mode():
    source = LLM_PY.read_text(encoding='utf-8')
    assert '"RPU_WALL_OSS_PREFILL_MIXED_PRECISION"' in source
    assert source.count('_prefill_mixed_precision_enabled()') == 4
    assert 'int(mixed_precision)' in source
    assert 'layer_indices == list(range(36))' in source
    assert 'H == 2048 and NQ == 16 and NKV == 2 and HD == 128' in source
    assert 'op_id != "wall_oss_llm_embeds"' in source
    assert 'seq == 672 and execution_len == 672' in source
    assert 'planned_chunk_size == 336' in source
    assert '_PREFILL_CHUNK_POLICY_VERSION = 4' in source
    assert 'self._prefill_mixed_precision and real_len == 672' in source
    assert 'max_seq_len >= 640' in source
    assert source.count('mixed_quantized and L == 2') == 4
    assert 'if down_int4 and not (mixed_quantized and L == 2):' in source
    assert 'validate_rtc_w4_checkpoint_pair(ckpt_dir, fp16_ckpt_dir)' in source
    assert 'down_key = f"{p}.moe.experts.{expert}.down_proj.weight"' in source
    assert 'fp16_st.get_tensor(down_key)' in source
