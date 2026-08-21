// Dispatcher schemas and implementations included by rpu_backend.cpp.
// This fragment must remain in that translation unit: it is source organization,
// not an independently compiled registration component.

TORCH_LIBRARY(rpu, m) {
  m.def("conv2d_mc_nhwc(Tensor input_nhwc, Tensor weight_nhwc, Tensor? bias, int[] stride, int[] padding, int[] dilation, int core_num) -> Tensor");
  m.def("rms_norm(Tensor input, int[] normalized_shape, Tensor weight, float eps=1e-5) -> Tensor");
  // RoPE: apply_rotary_pos_emb(query, key, cos, sin, position_ids) -> (query_rot, key_rot)
  m.def("apply_rotary_pos_emb(Tensor query, Tensor key, Tensor cos, Tensor sin, Tensor position_ids) -> (Tensor, Tensor)");
  // Wall-OSS on-device embed/cast ops.
  m.def("gather_embedding(Tensor vocab, Tensor ids) -> Tensor",
        TORCH_FN(rpu_gather_embedding));
  m.def("assemble_inputs_embeds(Tensor vocab, Tensor ids, Tensor merged, int img_start) -> Tensor",
        TORCH_FN(rpu_assemble_inputs_embeds));
  m.def("assemble_inputs_embeds_rev(Tensor vocab, Tensor ids, Tensor merged_window, Tensor rev_idx, Tensor runs) -> Tensor",
        TORCH_FN(rpu_assemble_inputs_embeds_rev));
  m.def("cast_uint8_fp16(Tensor input) -> Tensor", TORCH_FN(rpu_cast_uint8_fp16));
  m.def("cast_uint8_fp16_into(Tensor input, Tensor(a!) out) -> ()", TORCH_FN(rpu_cast_uint8_fp16_into));
  m.def("linear_into(Tensor input, Tensor weight, Tensor? bias, Tensor(a!) out) -> ()",
        TORCH_FN(rpu_linear_into));
  // SDPA unified layout: [batch, seq_len, heads, head_dim] instead of standard [batch, heads, seq_len, head_dim]
  m.def("scaled_dot_product_attention_unified(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, float? scale=None, bool enable_gqa=False) -> Tensor");
  // Fused QKV Attention (SPM版本，减少DDR搬运)
  // Note: Now takes separate q_norm_weight and k_norm_weight for Qwen3 compatibility
  // Note: o_weight added to fuse o_proj linear into the attention
  // Fused decoder families use the all-layers causal_decoder_forward path.
  // CausalDecoder All-Layers-Once (Qwen3 + Llama + future CausalLM share)
  // create/destroy don't need device dispatch (no tensor args), register directly
  m.def("causal_decoder_create", TORCH_FN(rpu_causal_decoder_create));
  m.def("causal_decoder_destroy", TORCH_FN(rpu_causal_decoder_destroy));
  // set_weights/forward take tensors → declare schema here, impl in PrivateUse1/AutogradPrivateUse1
  //
  // Trailing default kwargs are reserved for optional M-RoPE and DeepStack.
  // Append-only — existing
  // callers unaffected. Inner impl currently rejects non-empty values with
  // a clear "not yet implemented" error until those options are supported.
  m.def("causal_decoder_set_weights(int handle, Tensor[] q_w_list, Tensor[] k_w_list, Tensor[] v_w_list, Tensor[] o_w_list, Tensor[] q_norm_list, Tensor[] k_norm_list, Tensor[] input_norm_list, Tensor[] post_norm_list, Tensor[] gate_list, Tensor[] up_list, Tensor[] down_list, Tensor cos, Tensor sin, Tensor final_norm_w, int num_q_heads, int num_kv_heads, int head_dim, int hidden_size, int intermediate_size, float eps, bool use_silu, int[] mrope_section=[], int[] deepstack_lang_layers=[], Tensor[]? q_bias_list=None, Tensor[]? k_bias_list=None, Tensor[]? v_bias_list=None) -> ()");
  m.def("causal_decoder_set_weights_w8a16(int handle, Tensor[] q_w_list, Tensor[] k_w_list, Tensor[] v_w_list, Tensor[] o_w_list, Tensor[] q_norm_list, Tensor[] k_norm_list, Tensor[] input_norm_list, Tensor[] post_norm_list, Tensor[] gate_list, Tensor[] up_list, Tensor[] down_list, Tensor cos, Tensor sin, Tensor final_norm_w, int num_q_heads, int num_kv_heads, int head_dim, int hidden_size, int intermediate_size, float eps, bool use_silu, int[] mrope_section, int[] deepstack_lang_layers, Tensor[] q_w_scale_list, Tensor[] k_w_scale_list, Tensor[] v_w_scale_list, Tensor[] o_w_scale_list, Tensor[] gate_scale_list, Tensor[] up_scale_list, Tensor[] down_scale_list, Tensor[]? q_bias_list=None, Tensor[]? k_bias_list=None, Tensor[]? v_bias_list=None) -> ()");
  m.def("causal_decoder_set_weights_nvfp4(int handle, Tensor[] q_w_list, Tensor[] k_w_list, Tensor[] v_w_list, Tensor[] o_w_list, Tensor[] q_norm_list, Tensor[] k_norm_list, Tensor[] input_norm_list, Tensor[] post_norm_list, Tensor[] gate_list, Tensor[] up_list, Tensor[] down_list, Tensor cos, Tensor sin, Tensor final_norm_w, int num_q_heads, int num_kv_heads, int head_dim, int hidden_size, int intermediate_size, float eps, bool use_silu, int[] mrope_section, int[] deepstack_lang_layers, Tensor[] q_w_scale_list, Tensor[] k_w_scale_list, Tensor[] v_w_scale_list, Tensor[] o_w_scale_list, Tensor[] gate_scale_list, Tensor[] up_scale_list, Tensor[] down_scale_list, Tensor q_tensor_scales, Tensor k_tensor_scales, Tensor v_tensor_scales, Tensor o_tensor_scales, Tensor gate_tensor_scales, Tensor up_tensor_scales, Tensor down_tensor_scales, Tensor[]? q_bias_list=None, Tensor[]? k_bias_list=None, Tensor[]? v_bias_list=None) -> ()");
  m.def("causal_decoder_forward(int handle, Tensor hidden_states, Tensor[] k_caches, Tensor[] v_caches, Tensor? attention_mask, int position, bool is_causal, Tensor? position_ids=None, Tensor[]? deepstack_dense_visual_embeds=None, Tensor? rope_cos_il=None, Tensor? rope_sin_il=None, int cos_sin_offset=-1, int batch_slot=0, bool allow_batch_decode=False) -> Tensor");
  // LingBot-VLA AdaRMS expert: per-step FiLM scale/shift update (Tensor[] args → PrivateUse1 dispatch).
  m.def("causal_decoder_set_adarms_step(int handle, Tensor[] input_scales, Tensor[] input_shifts, Tensor[] post_scales, Tensor[] post_shifts) -> ()");
  m.def("causal_decoder_set_adarms_step_mutable(int handle, Tensor input_scale, Tensor input_shift, Tensor post_scale, Tensor post_shift) -> ()");
  // Phase 2.5: per-instance fused lm_head 入口. set takes tensor → 走 PrivateUse1 dispatch.
  // clear 不带 tensor, 直接注册即可。
  m.def("causal_decoder_set_lm_head(int handle, Tensor lm_head_w, Tensor? lm_head_scale=None) -> ()");
  m.def("causal_decoder_clear_lm_head", TORCH_FN(rpu_causal_decoder_clear_lm_head));
  m.def("causal_decoder_set_fast_replay(int handle, bool enabled) -> ()",
        TORCH_FN(rpu_causal_decoder_set_fast_replay));
  m.def("causal_decoder_set_preload_replay_skip(int handle, bool enabled) -> ()",
        TORCH_FN(rpu_causal_decoder_set_preload_replay_skip));
  m.def("causal_decoder_set_chunk_size_cap(int handle, int chunk_size_cap) -> ()",
        TORCH_FN(rpu_causal_decoder_set_chunk_size_cap));
  // Per-handle chunk authority replaces the deleted process global
  // (runtime.control.set_chunk_size / g_chunk_size_override).
  m.def("causal_decoder_set_chunk_size_override(int handle, int chunk_size) -> ()",
        TORCH_FN(rpu_causal_decoder_set_chunk_size_override));
  m.def("causal_decoder_get_chunk_size_override(int handle) -> int",
        TORCH_FN(rpu_causal_decoder_get_chunk_size_override));
  // Deny-by-default certified chunk envelope. Every CausalDecoderModel
  // handle must declare one before its first PREFILL or the planner refuses.
  m.def("causal_decoder_set_chunk_envelope(int handle, int max_kv_len, int chunk) -> ()",
        TORCH_FN(rpu_causal_decoder_set_chunk_envelope));
  m.def("causal_decoder_set_equal_two_prefill(int handle, bool enabled) -> ()",
        TORCH_FN(rpu_causal_decoder_set_equal_two_prefill));
  m.def("lm_head_logits_top1(Tensor logits) -> Tensor");
  // Per-handle resolved chunk_size
  // getter (read-only, int -> int). DEFAULT block ONLY (no PrivateUse1 / AutogradPrivateUse1
  // — same pattern as causal_decoder_create / causal_decoder_destroy: int-only ops without
  // tensor args don't need device dispatch).
  m.def("causal_decoder_get_resolved_chunk_size", TORCH_FN(rpu_causal_decoder_get_resolved_chunk_size));
  m.def("causal_decoder_resolve_prefill_chunk_size",
        TORCH_FN(rpu_causal_decoder_resolve_prefill_chunk_size));
  // Greedy-decode argmax on the host without the D2H copy / fp32 temp that
  // `logits.cpu().float().argmax(-1)` pays. An explicit op rather than an
  // aten::argmax override, so callers opt in and generic argmax is untouched.
  m.def("argmax_lastdim_host(Tensor self) -> Tensor");
  // Qwen3.5 hybrid: heterogeneous full/GDN dispatch + dual cache.
  m.def("qwen3_5_create", TORCH_FN(rpu_qwen3_5_create));
  m.def("qwen3_5_destroy", TORCH_FN(rpu_qwen3_5_destroy));
  m.def("qwen3_5_spm_z2_prepare(int vision_handle, int text_handle, int num_patches, int execution_len, int image_row_begin, int real_len) -> int", TORCH_FN(rpu_qwen3_5_spm_z2_prepare));
  m.def("qwen3_5_spm_z2_begin(int vision_handle, int text_handle, int num_patches, int execution_len, int image_row_begin, int real_len, int plan_hash) -> int", TORCH_FN(rpu_qwen3_5_spm_z2_begin));
  m.def("qwen3_5_spm_z2_end(int epoch, int plan_hash) -> ()", TORCH_FN(rpu_qwen3_5_spm_z2_end));
  m.def("qwen3_5_spm_z2_export_image_slice(int epoch, int plan_hash) -> Tensor", TORCH_FN(rpu_qwen3_5_spm_z2_export_image_slice));
  m.def("qwen3_5_spm_z2_pipeline_forward(int vision_handle, int text_handle, Tensor vision_input, Tensor[] vision_k_caches, Tensor[] vision_v_caches, Tensor step0_pos, Tensor text_hidden, Tensor[] text_k_caches, Tensor[] text_v_caches, Tensor[] gdn_states, Tensor[] conv_states, int epoch, int plan_hash) -> Tensor");
  m.def("wall_oss_spm_z1_prepare(int vision_handle, int text_handle, int num_patches, int execution_len, int image_row_begin, int real_len) -> int", TORCH_FN(rpu_wall_oss_spm_z1_prepare));
  m.def("wall_oss_spm_z1_unprepare(int vision_handle, int text_handle, int plan_hash) -> ()", TORCH_FN(rpu_wall_oss_spm_z1_unprepare));
  m.def("wall_oss_spm_z1_begin(int vision_handle, int text_handle, int num_patches, int execution_len, int image_row_begin, int real_len, Tensor window_mask, Tensor position_ids, Tensor rope_cos_il, Tensor rope_sin_il, int plan_hash) -> int", TORCH_FN(rpu_wall_oss_spm_z1_begin));
  m.def("wall_oss_spm_z1_end(int epoch, int plan_hash) -> ()", TORCH_FN(rpu_wall_oss_spm_z1_end));
  m.def("wall_oss_spm_z1_export_destination(int epoch, int plan_hash) -> Tensor", TORCH_FN(rpu_wall_oss_spm_z1_export_destination));
  m.def("qwen3vl_multiview_spm_dry_probe(int vision_handle, int text_handle, int num_patches, int execution_len) -> int[]", TORCH_FN(rpu_qwen3vl_multiview_spm_dry_probe));
  m.def("qwen3vl_multiview_spm_z2_canary_prepare(int vision_handle, int text_handle, Tensor position_ids) -> int", TORCH_FN(rpu_qwen3vl_multiview_spm_z2_canary_prepare));
  m.def("qwen3vl_multiview_spm_z2_canary_forward(int vision_handle, int text_handle, Tensor[] vision_inputs, Tensor[] vision_k_caches, Tensor[] vision_v_caches, Tensor text_hidden, Tensor[] text_k_caches, Tensor[] text_v_caches, Tensor position_ids, int plan_hash) -> Tensor");
  m.def("qwen3vl_multiview_spm_z2_canary_stats(int vision_handle, int text_handle, int plan_hash) -> int[]", TORCH_FN(rpu_qwen3vl_multiview_spm_z2_canary_stats));
  m.def("qwen3vl_multiview_spm_z2_canary_close(int vision_handle, int text_handle, int plan_hash) -> ()", TORCH_FN(rpu_qwen3vl_multiview_spm_z2_canary_close));
  m.def("lingbot2_multiview_spm_z2_prepare(int vision_handle, int text_handle, int expert_handle, Tensor position_ids, Tensor rope_cos_il, Tensor rope_sin_il) -> int", TORCH_FN(rpu_lingbot2_multiview_spm_z2_prepare));
  m.def("lingbot2_multiview_spm_z2_refresh_text_inputs(int vision_handle, int text_handle, int expert_handle, Tensor position_ids, Tensor rope_cos_il, Tensor rope_sin_il, int plan_hash) -> ()", TORCH_FN(rpu_lingbot2_multiview_spm_z2_refresh_text_inputs));
  m.def("lingbot2_multiview_spm_z2_forward(int vision_handle, int text_handle, int expert_handle, Tensor[] vision_inputs, Tensor[] vision_k_caches, Tensor[] vision_v_caches, Tensor text_hidden, Tensor[] text_k_caches, Tensor[] text_v_caches, Tensor position_ids, Tensor rope_cos_il, Tensor rope_sin_il, int plan_hash) -> ()");
  m.def("lingbot2_multiview_spm_z2_stats(int vision_handle, int text_handle, int expert_handle, int plan_hash) -> int[]", TORCH_FN(rpu_lingbot2_multiview_spm_z2_stats));
  m.def("lingbot2_multiview_spm_z2_close(int vision_handle, int text_handle, int expert_handle, int plan_hash) -> ()", TORCH_FN(rpu_lingbot2_multiview_spm_z2_close));
  m.def("qwen3vl_pooler_spm_z1_prepare(int vision_handle, int text_handle, int num_patches, int execution_len, int image_row_begin, int real_len, int retained_deepstack_count=0) -> int", TORCH_FN(rpu_qwen3vl_pooler_spm_z1_prepare));
  m.def("qwen3vl_pooler_spm_z1_unprepare(int vision_handle, int text_handle, int plan_hash) -> ()", TORCH_FN(rpu_qwen3vl_pooler_spm_z1_unprepare));
  m.def("qwen3vl_pooler_spm_z1_plan_stats(int vision_handle, int text_handle, int plan_hash) -> int[]", TORCH_FN(rpu_qwen3vl_pooler_spm_z1_plan_stats));
  m.def("qwen3vl_pooler_spm_z1_begin(int vision_handle, int text_handle, int num_patches, int execution_len, int image_row_begin, int real_len, Tensor position_ids, int plan_hash, Tensor? rope_cos_il=None, Tensor? rope_sin_il=None) -> int", TORCH_FN(rpu_qwen3vl_pooler_spm_z1_begin));
  m.def("qwen3vl_pooler_spm_z1_end(int epoch, int plan_hash) -> ()", TORCH_FN(rpu_qwen3vl_pooler_spm_z1_end));
  m.def("qwen3vl_pooler_spm_z1_abort(int epoch, int plan_hash) -> ()", TORCH_FN(rpu_qwen3vl_pooler_spm_z1_abort));
  m.def("qwen3vl_pooler_spm_z1_cancel(int epoch, int plan_hash) -> ()", TORCH_FN(rpu_qwen3vl_pooler_spm_z1_cancel));
  m.def("qwen3vl_pooler_spm_z1_pipeline_forward(int vision_handle, int text_handle, Tensor vision_input, Tensor[] vision_k_caches, Tensor[] vision_v_caches, Tensor text_hidden, Tensor[] text_k_caches, Tensor[] text_v_caches, Tensor position_ids, int epoch, int plan_hash) -> Tensor");
  m.def("wall_oss_spm_z1_pipeline_forward(int vision_handle, int text_handle, Tensor vision_input, Tensor[] vision_k_caches, Tensor[] vision_v_caches, Tensor window_mask, Tensor text_hidden, Tensor[] text_k_caches, Tensor[] text_v_caches, Tensor position_ids, Tensor rope_cos_il, Tensor rope_sin_il, int epoch, int plan_hash) -> Tensor");
  m.def("qwen3_5_set_weights(int handle, Tensor[] q_w_list, Tensor[] k_w_list, Tensor[] v_w_list, Tensor[] o_w_list, Tensor[] q_norm_list, Tensor[] k_norm_list, Tensor[] attn_gate_list, Tensor[] input_norm_list, Tensor[] post_norm_list, Tensor[] gate_list, Tensor[] up_list, Tensor[] down_list, Tensor cos, Tensor sin, Tensor final_norm_w, int[] layer_is_full, int num_q_heads, int num_kv_heads, int head_dim, int hidden_size, int intermediate_size, float eps, bool use_silu, int[] mrope_section, Tensor[] gdn_in_z_list, Tensor[] gdn_out_list, Tensor[] gdn_A_log_list, Tensor[] gdn_dt_bias_list, Tensor[] gdn_norm_list, int gdn_num_v_heads, int gdn_key_head_dim, int gdn_value_head_dim, int gdn_conv_dim, int gdn_conv_kernel, Tensor[] gdn_q_list, Tensor[] gdn_k_list, Tensor[] gdn_v_list, Tensor[] gdn_cq_list, Tensor[] gdn_ck_list, Tensor[] gdn_cv_list, Tensor[] gdn_b_bg_list, Tensor[] gdn_a_bg_list) -> ()");
  m.def("qwen3_5_forward(int handle, Tensor hidden_states, Tensor[] k_caches, Tensor[] v_caches, Tensor[] gdn_states, Tensor[] conv_states, Tensor? attention_mask, int position, bool is_causal) -> Tensor");
  m.def("qwen3_5_set_prefill_rope(int handle, Tensor cos, Tensor sin) -> ()");
  // int-only (no tensor arg) ⇒ DEFAULT block only, no device dispatch (mirror
  // qwen3_5_get_resolved_chunk_size below; schema inferred from the function signature).
  m.def("qwen3_5_set_valid_prefill_len", TORCH_FN(rpu_qwen3_5_set_valid_prefill_len));
  m.def("qwen3_5_set_mrope_position_delta", TORCH_FN(rpu_qwen3_5_set_mrope_position_delta));
  m.def("qwen3_5_set_chunk_size_cap", TORCH_FN(rpu_qwen3_5_set_chunk_size_cap));
  m.def("qwen3_5_set_prefill_chunk_size",
        TORCH_FN(rpu_qwen3_5_set_prefill_chunk_size));
  m.def("qwen3_5_set_chunk_envelope", TORCH_FN(rpu_qwen3_5_set_chunk_envelope));
  m.def("qwen3_5_set_linear_acc32", TORCH_FN(rpu_qwen3_5_set_linear_acc32));
  m.def("qwen3_5_set_fast_replay", TORCH_FN(rpu_qwen3_5_set_fast_replay));
  m.def("qwen3_5_enable_action_mode", TORCH_FN(rpu_qwen3_5_enable_action_mode));
  m.def("qwen3_5_set_action_io_weights(int handle, Tensor input_w, Tensor input_b, Tensor output_w, Tensor output_b, int action_dim, int action_dim_pad, int action_len) -> ()");
  m.def("qwen3_5_action_forward(int handle, Tensor hidden_states, Tensor adaptive_mod, Tensor[] k_caches, Tensor[] v_caches, int[] prefix_lens) -> Tensor");
  m.def("qwen3_5_action_step_forward(int handle, Tensor action, Tensor action_keep_mask, Tensor(a!) action_out, float delta_t, Tensor adaptive_mod, Tensor[] k_caches, Tensor[] v_caches, int[] prefix_lens, int num_steps) -> Tensor");
  m.def("qwen3_5_action_rtc_step_forward(int handle, Tensor action, Tensor action_keep_mask, Tensor(a!) action_out, Tensor action_prefix, int action_prefix_len, float delta_t, Tensor adaptive_mod, Tensor[] k_caches, Tensor[] v_caches, int[] prefix_lens, int num_steps) -> Tensor");
  m.def("qwen3_5_action_rtc_trace_forward(int handle, Tensor action, Tensor action_keep_mask, Tensor(a!) action_out, Tensor action_prefix, int action_prefix_len, float delta_t, Tensor adaptive_mod, Tensor[] k_caches, Tensor[] v_caches, int[] prefix_lens, Tensor(b!) trace_pre_action, Tensor(c!) trace_action_embed, Tensor(d!) trace_final_hidden, Tensor(e!) trace_velocity, Tensor(f!) trace_post_action) -> Tensor");
  m.def("qwen3_5_resolve_prefill_chunk_size",
        TORCH_FN(rpu_qwen3_5_resolve_prefill_chunk_size));
  // Resolved chunk_size getter (int -> int, DEFAULT block only — no device dispatch, same as
  // causal_decoder_get_resolved_chunk_size). Read from Python after a prefill forward.
  m.def("qwen3_5_get_resolved_chunk_size", TORCH_FN(rpu_qwen3_5_get_resolved_chunk_size));
  // Gemma All-Layers-Once
  m.def("gemma_create", TORCH_FN(rpu_gemma_create));
  m.def("gemma_destroy", TORCH_FN(rpu_gemma_destroy));
  m.def("gemma_set_weights(int handle, Tensor[] q_w_list, Tensor[] k_w_list, Tensor[] v_w_list, Tensor[] o_w_list, Tensor[] input_norm_list, Tensor[] post_norm_list, Tensor[] gate_list, Tensor[] up_list, Tensor[] down_list, Tensor cos, Tensor sin, Tensor final_norm_w, int num_q_heads, int num_kv_heads, int head_dim, int hidden_size, int intermediate_size, float eps, Tensor[] q_w_scale_list, Tensor[] k_w_scale_list, Tensor[] v_w_scale_list, Tensor[] o_w_scale_list, Tensor[] gate_scale_list, Tensor[] up_scale_list, Tensor[] down_scale_list) -> ()");
  m.def("gemma_forward(int handle, Tensor hidden_states, Tensor[] k_caches, Tensor[] v_caches, Tensor? attention_mask, int position, bool is_causal, int chunk_size=0) -> Tensor");
  m.def("gemma_set_prefill_rope(int handle, Tensor cos, Tensor sin) -> ()");
  m.def("gemma_resolve_prefill_chunk_size", TORCH_FN(rpu_gemma_resolve_prefill_chunk_size));
  m.def("gemma_get_resolved_chunk_size", TORCH_FN(rpu_gemma_get_resolved_chunk_size));
  // No Tensor argument -> BackendSelect cannot route it; define WITH the impl
  // here (same shape as the *_create/*_destroy handle ops above).
  m.def("pi05_denoise_step_set_rope_position", TORCH_FN(rpu_pi05_denoise_step_set_rope_position));
  m.def("pi05_denoise_step_set_chunk_size", TORCH_FN(rpu_pi05_denoise_step_set_chunk_size));
  m.def("pi05_denoise_step_get_resolved_chunk_size",
        TORCH_FN(rpu_pi05_denoise_step_get_resolved_chunk_size));
  m.def("adarms_set_rope_position", TORCH_FN(rpu_adarms_set_rope_position));
  // Gemma4 All-Layers-Once (E4B text decoder, mixed sliding/global geometry)
  m.def("gemma4_create", TORCH_FN(rpu_gemma4_create));
  m.def("gemma4_destroy", TORCH_FN(rpu_gemma4_destroy));
  m.def("gemma4_set_weights(int handle, Tensor[] q_w_list, Tensor[] k_w_list, Tensor[] v_w_list, Tensor[] o_w_list, Tensor[] q_norm_list, Tensor[] k_norm_list, Tensor[] input_norm_list, Tensor[] post_attn_norm_list, Tensor[] pre_ff_norm_list, Tensor[] post_ff_norm_list, Tensor[] gate_list, Tensor[] up_list, Tensor[] down_list, Tensor cos_sliding, Tensor sin_sliding, Tensor cos_global, Tensor sin_global, Tensor final_norm_w, Tensor[] ple_gate_list, Tensor[] ple_proj_list, Tensor[] ple_post_norm_list, Tensor layer_scalar, int[] layer_type_ids, int[] head_dim_per_layer, int[] num_kv_heads_per_layer, int[] kv_source_layer, int[] is_kv_shared, int num_q_heads, int num_kv_heads_max, int head_dim_max, int hidden_size, int intermediate_size, int ple_dim, float eps, float qk_scale, int sliding_window) -> ()");
  m.def("gemma4_forward(int handle, Tensor hidden_states, Tensor[] k_caches, Tensor[] v_caches, Tensor? attention_mask, int position, bool is_causal, int chunk_size=0, Tensor? side_input=None) -> Tensor");
  // AdaRMSModel (Pi0.5 Action Expert all-layers-once)
  m.def("adarms_create", TORCH_FN(rpu_adarms_create));
  m.def("adarms_destroy", TORCH_FN(rpu_adarms_destroy));
  m.def("adarms_set_weights(int handle, Tensor[] q_w_list, Tensor[] k_w_list, Tensor[] v_w_list, Tensor[] o_w_list, Tensor[] gate_list, Tensor[] up_list, Tensor[] down_list, Tensor[] attn_dense_w_list, Tensor[] attn_dense_b_list, Tensor[] mlp_dense_w_list, Tensor[] mlp_dense_b_list, Tensor cos, Tensor sin, int num_q_heads, int num_kv_heads, int head_dim, int hidden_size, int intermediate_size, float eps, Tensor[] q_w_scale_list, Tensor[] k_w_scale_list, Tensor[] v_w_scale_list, Tensor[] o_w_scale_list, Tensor[] gate_scale_list, Tensor[] up_scale_list, Tensor[] down_scale_list) -> ()");
  m.def("adarms_forward(int handle, Tensor hidden_states, Tensor cond, Tensor[] k_caches, Tensor[] v_caches, Tensor? attention_mask, int position, bool is_causal) -> Tensor");
  // GR00T-N1.7 flow-matching DiT action head (src/fused/rpu_gr00t_dit_model.cpp)
  m.def("gr00t_dit_create", TORCH_FN(rpu_gr00t_dit_create));
  m.def("gr00t_dit_destroy", TORCH_FN(rpu_gr00t_dit_destroy));
  m.def("gr00t_dit_set_weights(int handle, Tensor[] to_q_list, Tensor[] to_k_list, Tensor[] to_v_list, Tensor[] to_out_list, Tensor[] ff1_list, Tensor[] ff2_list, Tensor[] adaln_w_list, Tensor[] adaln_b_list, Tensor[] to_q_b_list, Tensor[] to_k_b_list, Tensor[] to_v_b_list, Tensor[] to_out_b_list, Tensor[] ff1_b_list, Tensor[] ff2_b_list, int[] block_is_cross, int num_heads, int head_dim, int hidden_size, int ff_inter, int cross_dim, float eps) -> ()");
  m.def("gr00t_dit_set_weights_w8a16(int handle, Tensor[] to_q_list, Tensor[] to_k_list, Tensor[] to_v_list, Tensor[] to_out_list, Tensor[] ff1_list, Tensor[] ff2_list, Tensor[] adaln_w_list, Tensor[] adaln_b_list, Tensor[] to_q_b_list, Tensor[] to_k_b_list, Tensor[] to_v_b_list, Tensor[] to_out_b_list, Tensor[] ff1_b_list, Tensor[] ff2_b_list, int[] block_is_cross, int num_heads, int head_dim, int hidden_size, int ff_inter, int cross_dim, float eps, Tensor[] to_q_scale_list, Tensor[] to_k_scale_list, Tensor[] to_v_scale_list, Tensor[] to_out_scale_list, Tensor[] ff1_scale_list, Tensor[] ff2_scale_list, Tensor[] adaln_scale_list) -> ()");
  m.def("gr00t_dit_forward(int handle, Tensor sa_embs, Tensor cond, Tensor[] k_caches, Tensor[] v_caches, Tensor? attention_mask, Tensor? vl_embeds=None, Tensor? cross_mask=None, Tensor? cross_mask_img=None) -> Tensor");
  // InternVLA-N1 standalone flow-matching NextDiT core.
  m.def("internvla_nextdit_create", TORCH_FN(rpu_internvla_nextdit_create));
  m.def("internvla_nextdit_destroy", TORCH_FN(rpu_internvla_nextdit_destroy));
  m.def("internvla_nextdit_set_16b_sdpa", TORCH_FN(rpu_internvla_nextdit_set_16b_sdpa));
  m.def("internvla_nextdit_set_fused_qkq_weights(int handle, Tensor[] weights) -> ()");
  m.def("internvla_nextdit_set_step_prepare_weights(int handle, Tensor weight, Tensor bias, Tensor fold_norm_weight) -> ()");
  m.def("internvla_nextdit_set_weights(int handle, Tensor[] self_q_w, Tensor[] self_k_w, Tensor[] self_v_w, Tensor[] cross_q_w, Tensor[] cross_k_w, Tensor[] cross_v_w, Tensor[] out_w, Tensor[] ff1_w, Tensor[] ff2_w, Tensor[] ff3_w, Tensor[] norm1_w, Tensor[] norm2_w, Tensor[] ffn_norm1_w, Tensor[] ffn_norm2_w, Tensor[] self_qn_w, Tensor[] self_qn_b, Tensor[] self_kn_w, Tensor[] self_kn_b, Tensor[] cross_qn_w, Tensor[] cross_qn_b, Tensor[] cross_kn_w, Tensor[] cross_kn_b, Tensor[] context_norm_w, Tensor[] cross_gate_heads, Tensor final_out_w, Tensor final_out_b, Tensor final_ln_w, Tensor final_ln_b, Tensor action_enc_w, Tensor action_enc_b, Tensor action_position, Tensor action_dec_w, Tensor action_dec_b, int num_heads, int head_dim, int hidden_size, int ff_intermediate, float eps, float final_norm_eps) -> ()");
  m.def("internvla_nextdit_forward(int handle, Tensor x, Tensor modulation, Tensor final_scale, Tensor raw_actions, bool use_action_encoder, bool populate_cross_cache, bool denoise_unroll, bool prepare_modulation_on_rpu, float[] euler_steps, Tensor cross_context, Tensor[] self_k_caches, Tensor[] self_v_caches, Tensor[] cross_k_caches, Tensor[] cross_v_caches, Tensor layer_outputs, bool collect_layer_outputs=False) -> Tensor");
  // InternVLA-N1 NavDP controlled-evaluation subsystem.
  m.def("navdp_create", TORCH_FN(rpu_navdp_create));
  m.def("navdp_destroy", TORCH_FN(rpu_navdp_destroy));
  m.def("navdp_set_former_mode", TORCH_FN(rpu_navdp_set_former_mode));
  m.def("navdp_set_weights(int handle, Tensor[] sq_w, Tensor[] sk_w, Tensor[] sv_w, Tensor[] so_w, Tensor[] sq_b, Tensor[] sk_b, Tensor[] sv_b, Tensor[] so_b, Tensor[] cq_w, Tensor[] ck_w, Tensor[] cv_w, Tensor[] co_w, Tensor[] cq_b, Tensor[] ck_b, Tensor[] cv_b, Tensor[] co_b, Tensor[] ff1_w, Tensor[] ff1_b, Tensor[] ff2_w, Tensor[] ff2_b, Tensor[] n1_w, Tensor[] n1_b, Tensor[] n2_w, Tensor[] n2_b, Tensor[] n3_w, Tensor[] n3_b, int num_heads, int head_dim, int hidden_size, int ff_inter, int memory_len, int predict_size, int action_dim, float eps) -> ()");
  m.def("navdp_forward(int handle, Tensor tgt, Tensor memory, Tensor[] k_caches, Tensor[] v_caches, Tensor? tgt_mask=None) -> Tensor");
  m.def("navdp_set_denoise_glue(int handle, Tensor ie_w, Tensor ie_b, Tensor ah_w, Tensor ah_b, Tensor fln_w, Tensor fln_b, Tensor out_pos) -> ()");
  m.def("navdp_denoise_loop_forward(int handle, Tensor x0, Tensor mem_all, Tensor sigz, Tensor coeff, Tensor[] k_caches, Tensor[] v_caches, Tensor out) -> Tensor");
  // GR00T-N1.7 Component A — vlln + 4-layer vl_self_attention (src/fused/rpu_gr00t_vl_encoder_model.cpp)
  m.def("gr00t_vl_encoder_create", TORCH_FN(rpu_gr00t_vl_encoder_create));
  m.def("gr00t_vl_encoder_destroy", TORCH_FN(rpu_gr00t_vl_encoder_destroy));
  m.def("gr00t_vl_encoder_set_weights(int handle, Tensor[] norm1_w_list, Tensor[] norm1_b_list, Tensor[] to_q_list, Tensor[] to_q_b_list, Tensor[] to_k_list, Tensor[] to_k_b_list, Tensor[] to_v_list, Tensor[] to_v_b_list, Tensor[] to_out_list, Tensor[] to_out_b_list, Tensor[] norm3_w_list, Tensor[] norm3_b_list, Tensor[] ff0_list, Tensor[] ff0_b_list, Tensor[] ff2_list, Tensor[] ff2_b_list, Tensor vlln_w, Tensor vlln_b, Tensor inv_w_rms, int num_heads, int head_dim, int hidden_size, int ff_inter, float eps) -> ()");
  m.def("gr00t_vl_encoder_set_weights_w8a16(int handle, Tensor[] norm1_w_list, Tensor[] norm1_b_list, Tensor[] to_q_list, Tensor[] to_q_b_list, Tensor[] to_k_list, Tensor[] to_k_b_list, Tensor[] to_v_list, Tensor[] to_v_b_list, Tensor[] to_out_list, Tensor[] to_out_b_list, Tensor[] norm3_w_list, Tensor[] norm3_b_list, Tensor[] ff0_list, Tensor[] ff0_b_list, Tensor[] ff2_list, Tensor[] ff2_b_list, Tensor vlln_w, Tensor vlln_b, Tensor inv_w_rms, int num_heads, int head_dim, int hidden_size, int ff_inter, float eps, Tensor[] to_q_scale_list, Tensor[] to_k_scale_list, Tensor[] to_v_scale_list, Tensor[] to_out_scale_list, Tensor[] ff0_scale_list, Tensor[] ff2_scale_list) -> ()");
  m.def("gr00t_vl_encoder_forward(int handle, Tensor raw, Tensor[] k_caches, Tensor[] v_caches) -> Tensor");
  m.def("gr00t_vl_encoder_forward_z2(int handle, Tensor raw, Tensor[] k_caches, Tensor[] v_caches, int epoch, int plan_hash) -> ()");
  // GR00T-N1.7 Component B — gr00t_denoise (extends the gr00t_dit handle; per-step glue + Euler)
  m.def("gr00t_denoise_set_weights(int handle, Tensor se1_w, Tensor se1_b, Tensor se2_w, Tensor se2_b, Tensor ae1_w, Tensor ae1_b, Tensor ae2a_w, Tensor ae2t_w, Tensor ae2_b, Tensor ae3_w, Tensor ae3_b, Tensor dec1_w, Tensor dec1_b, Tensor dec2_w, Tensor dec2_b, Tensor po1_w, Tensor po1_b, Tensor po2_w, Tensor po2_b, Tensor cond_table, Tensor tau_table, Tensor pos_table, int action_dim, int action_dim_pad, int num_steps, float dt, Tensor? se2_scale=None, Tensor? ae2a_scale=None, Tensor? ae2t_scale=None, Tensor? ae3_scale=None, Tensor? dec1_scale=None, Tensor? dec2_scale=None, Tensor? po1_scale=None, Tensor? po2_scale=None) -> ()");
  m.def("gr00t_denoise_step_forward(int handle, Tensor actions, Tensor state, int step_idx, Tensor[] k_caches, Tensor[] v_caches, Tensor vl_embeds, Tensor tmask, Tensor imask) -> Tensor");
  m.def("gr00t_denoise_unroll_forward(int handle, Tensor noise, Tensor state, Tensor[] k_caches, Tensor[] v_caches, Tensor vl_embeds, Tensor tmask, Tensor imask) -> Tensor");
  m.def("gr00t_denoise_unroll_forward_z2(int handle, Tensor noise, Tensor state, Tensor[] k_caches, Tensor[] v_caches, Tensor tmask, Tensor imask, int epoch, int plan_hash) -> Tensor");
  m.def("gr00t_spm_z2_prepare(int vl_handle, int denoise_handle, int seq_len) -> int", TORCH_FN(rpu_gr00t_spm_z2_prepare));
  m.def("gr00t_spm_z2_unprepare(int vl_handle, int denoise_handle, int plan_hash) -> ()", TORCH_FN(rpu_gr00t_spm_z2_unprepare));
  m.def("gr00t_spm_z2_begin(int vl_handle, int denoise_handle, int seq_len, int plan_hash) -> int", TORCH_FN(rpu_gr00t_spm_z2_begin));
  m.def("gr00t_spm_z2_end(int epoch, int plan_hash) -> ()", TORCH_FN(rpu_gr00t_spm_z2_end));
  m.def("gr00t_spm_z2_prepare_masks(int denoise_handle, Tensor tmask, Tensor imask, int plan_hash) -> ()");
  m.def("gr00t_spm_z2_pipeline_forward(int vl_handle, int denoise_handle, Tensor raw, Tensor[] vl_k_caches, Tensor[] vl_v_caches, Tensor noise, Tensor state, Tensor[] denoise_k_caches, Tensor[] denoise_v_caches, int epoch, int plan_hash) -> Tensor");
  // Pi0.5 denoise-loop driver (plain driver op).
  // set_weights takes action_in_proj + action_out_proj weights;
  //   forward re-computes action_in_proj(x_t) + action_out_proj(suffix_tail) per step.
  // num_steps is a runtime forward argument, not baked at set_weights.
  // No per-token positional-offsets Tensor is present in the schema;
  //   scalar `prefix_len` is threaded through to the existing AdaRMS C API
  //   (constant across denoise steps).
  m.def("pi05_create", TORCH_FN(rpu_pi05_create));
  m.def("pi05_destroy", TORCH_FN(rpu_pi05_destroy));
  m.def("pi05_set_weights(int handle, int adarms_handle, Tensor action_in_proj_w, Tensor action_in_proj_b, Tensor action_out_proj_w, Tensor action_out_proj_b, int chunk_size, int max_action_dim, int width) -> ()");
  m.def("pi05_forward(int handle, Tensor initial_noise, Tensor adarms_cond_list, Tensor? attention_mask, Tensor[] k_caches_list, Tensor[] v_caches_list, int prefix_len, int num_steps, float dt) -> Tensor");
  // Pi0.5 num_steps fused denoise step. create/destroy use direct TORCH_FN
  // registration because schema-only m.def for zero-Tensor ops does not dispatch.
  m.def("pi05_denoise_step_create",  TORCH_FN(rpu_pi05_denoise_step_create));
  m.def("pi05_denoise_step_destroy", TORCH_FN(rpu_pi05_denoise_step_destroy));
  m.def("pi05_denoise_step_set_weights("
        "int handle, "
        "Tensor[] q_w, Tensor[] k_w, Tensor[] v_w, Tensor[] o_w, "
        "Tensor[] gate_w, Tensor[] up_w, Tensor[] down_w, "
        "Tensor[] attn_dense_w, Tensor[] attn_dense_b, "
        "Tensor[] mlp_dense_w, Tensor[] mlp_dense_b, "
        "Tensor final_norm_dense_w, Tensor final_norm_dense_b, "
        "Tensor action_in_proj_w, Tensor action_in_proj_b, "
        "Tensor action_out_proj_w, Tensor action_out_proj_b, "
        "Tensor cos, Tensor sin, "
        "int hidden_size, int max_action_dim, int chunk_size, "
        "int num_q_heads, int num_kv_heads, int head_dim, int num_layers, "
        "float eps, "
        "Tensor[] q_w_scale, Tensor[] k_w_scale, Tensor[] v_w_scale, Tensor[] o_w_scale, "
        "Tensor[] gate_scale, Tensor[] up_scale, Tensor[] down_scale, "
        "Tensor[] attn_dense_w_scale, Tensor[] mlp_dense_w_scale, "
        "Tensor final_norm_dense_w_scale) -> ()");
  m.def("pi05_denoise_step_forward("
        "int handle, Tensor x_t_rpu, Tensor[] k_caches, Tensor[] v_caches, "
        "Tensor cond_step, Tensor attention_mask_4d, Tensor v_t_buf, "
        "int prefix_len) -> ()");
  m.def("pi05_denoise_loop_forward("
        "int handle, Tensor x0_rpu, "
        "Tensor[] k_caches, Tensor[] v_caches, Tensor cond_all, "
        "Tensor attention_mask_4d, Tensor x_out, float dt, int prefix_len, "
        "int num_steps) -> ()");
  // WallOssActionStepModel (Wall-OSS-0.5 fused action-denoise step; inherits
  // CausalDecoderModel). create/destroy are zero-Tensor -> TORCH_FN m.def only.
  m.def("wall_oss_action_step_create",  TORCH_FN(rpu_wall_oss_action_step_create));
  m.def("wall_oss_action_step_destroy", TORCH_FN(rpu_wall_oss_action_step_destroy));
  m.def("wall_oss_action_step_set_weights("
        "int handle, "
        "Tensor[] q_w, Tensor[] k_w, Tensor[] v_w, Tensor[] o_w, "
        "Tensor[] input_norm, Tensor[] post_norm, "
        "Tensor[] gate_w, Tensor[] up_w, Tensor[] down_w, "
        "Tensor cos, Tensor sin, Tensor final_norm_w, "
        "int num_q_heads, int num_kv_heads, int head_dim, "
        "int hidden_size, int intermediate_size, float eps, bool use_silu, "
        "int[] mrope_section, Tensor[] q_bias, Tensor[] k_bias, Tensor[] v_bias, "
        "Tensor[] q_w_scale, Tensor[] k_w_scale, Tensor[] v_w_scale, "
        "Tensor[] o_w_scale, Tensor[] gate_scale, Tensor[] up_scale, Tensor[] down_scale, "
        "Tensor w_comb, Tensor w3, Tensor proj_back, Tensor w2_t, Tensor w2_a, "
        "int action_dim, int k_comb_pad, int chunk_size, "
        "Tensor? q_tensor_scales=None, Tensor? k_tensor_scales=None, "
        "Tensor? v_tensor_scales=None, Tensor? o_tensor_scales=None, "
        "Tensor? gate_tensor_scales=None, Tensor? up_tensor_scales=None, "
        "Tensor? down_tensor_scales=None) -> ()");
  m.def("wall_oss_action_step_forward("
        "int handle, Tensor x_t_rpu, "
        "Tensor[] k_caches, Tensor[] v_caches, Tensor b_step, Tensor? te_step, "
        "Tensor? attention_mask, Tensor position_ids, Tensor v_t_buf, "
        "Tensor x_out_buf, float dt, int prefix_len, "
        "Tensor? rope_cos_il=None, Tensor? rope_sin_il=None) -> Tensor");
  m.def("wall_oss_action_denoise_loop_forward("
        "int handle, Tensor x0_rpu, "
        "Tensor[] k_caches, Tensor[] v_caches, Tensor b_all, Tensor? te_all, "
        "Tensor? attention_mask, Tensor position_ids, Tensor x_traj, "
        "Tensor v_traj, float dt, int prefix_len, int num_steps, "
        "Tensor? rope_cos_il=None, Tensor? rope_sin_il=None, "
        "Tensor? rtc_prefix=None, int rtc_delay=0, "
        "Tensor? velocity_mask=None, Tensor? velocity_override=None) -> ()");
  // LingBot-VLA-V2 sparse-MoE action expert. The ten Euler steps are
  // orchestrated by Python and reuse one GraphCache entry.
  m.def("lingbot_v2_moe_create", TORCH_FN(rpu_lingbot_v2_moe_create));
  m.def("lingbot_v2_moe_destroy", TORCH_FN(rpu_lingbot_v2_moe_destroy));
  m.def("lingbot_v2_moe_set_weights("
        "int handle, "
        "Tensor[] q_w, Tensor[] k_w, Tensor[] v_w, Tensor[] o_w, "
        "Tensor[] input_norm, Tensor[] post_norm, "
        "Tensor[] shared_gate_w, Tensor[] shared_up_w, Tensor[] shared_down_w, "
        "Tensor cos, Tensor sin, Tensor final_norm_w, "
        "int num_q_heads, int num_kv_heads, int head_dim, "
        "int hidden_size, int shared_inter_pad, float eps, "
        "Tensor[] q_bias, Tensor[] k_bias, Tensor[] v_bias, "
        "Tensor[] router_gate_w, Tensor[] router_bias, "
        "Tensor[] expert_gate_w, Tensor[] expert_up_w, Tensor[] expert_down_w, "
        "int num_experts, int top_k, int routed_inter, "
        "float routed_scaling, int chunk_size) -> ()");
  m.def("lingbot_v2_moe_set_packed_weights("
        "int handle, Tensor[] gate_packed, Tensor[] up_packed, "
        "Tensor[] down_packed) -> ()");
  m.def("lingbot_v2_moe_set_packed_scales("
        "int handle, Tensor[] gate_scale, Tensor[] up_scale, "
        "Tensor[] down_scale) -> ()");
  m.def("lingbot_v2_moe_set_base_scales("
        "int handle, Tensor[] q_ws, Tensor[] k_ws, Tensor[] v_ws, Tensor[] o_ws, "
        "Tensor[] gate_ws, Tensor[] up_ws, Tensor[] down_ws) -> ()");
  m.def("lingbot_v2_moe_set_adarms_step_mutable("
        "int handle, Tensor input_scale, Tensor input_shift, "
        "Tensor post_scale, Tensor post_shift) -> ()");
  m.def("lingbot_v2_moe_bind_adarms_schedule("
        "int handle, Tensor input_scale, Tensor input_shift, "
        "Tensor post_scale, Tensor post_shift) -> ()");
  m.def("lingbot_v2_moe_set_denoise_unroll("
        "int handle, Tensor state_w, Tensor state_b, "
        "Tensor wc, Tensor mo, Tensor mo_bias, Tensor op, Tensor op_bias, "
        "Tensor time_all, Tensor adarms_is, Tensor adarms_ish, "
        "Tensor adarms_ps, Tensor adarms_psh, "
        "int state_dim, int action_dim, int state_dim_pad, "
        "int action_dim_pad, int num_steps) -> ()");
  m.def("lingbot_v2_moe_denoise_unroll_forward("
        "int handle, Tensor x0_rpu, Tensor state_rpu, "
        "Tensor[] k_caches, Tensor[] v_caches, Tensor? attention_mask, "
        "Tensor x_final, float dt, int prefix_len, int cos_sin_offset, "
        "int num_steps) -> ()");
  m.def("lingbot_v2_moe_forward("
        "int handle, Tensor hidden_states, "
        "Tensor[] k_caches, Tensor[] v_caches, "
        "Tensor? attention_mask, int position, "
        "int cos_sin_offset=-1, int adarms_schedule_step=-1) -> Tensor");
  // HaloActionExpertModel (HALO action expert MoT step; inherits
  // CausalDecoderModel). create/destroy are zero-Tensor -> TORCH_FN m.def only.
  m.def("halo_action_expert_create",  TORCH_FN(rpu_halo_action_expert_create));
  m.def("halo_action_expert_destroy", TORCH_FN(rpu_halo_action_expert_destroy));
  m.def("halo_action_expert_set_weights("
        "int handle, "
        "Tensor[] q_w, Tensor[] k_w, Tensor[] v_w, Tensor[] o_w, "
        "Tensor[] q_norm, Tensor[] k_norm, "
        "Tensor[] input_norm, Tensor[] post_norm, "
        "Tensor[] gate_w, Tensor[] up_w, Tensor[] down_w, "
        "Tensor cos, Tensor sin, Tensor final_norm_w, "
        "int num_q_heads, int num_kv_heads, int head_dim, "
        "int hidden_size, int intermediate_size, float eps, bool use_silu, "
        "Tensor[] q_bias, Tensor[] k_bias, Tensor[] v_bias) -> ()");
  m.def("halo_action_expert_set_moe_weights("
        "int handle, "
        "Tensor[] q_w, Tensor[] k_w, Tensor[] v_w, Tensor[] o_w, "
        "Tensor[] q_norm, Tensor[] k_norm, "
        "Tensor[] input_norm, Tensor[] post_norm, "
        "Tensor[] gate_w, Tensor[] up_w, Tensor[] down_w, "
        "Tensor[] q_bias, Tensor[] k_bias, Tensor[] v_bias, "
        "Tensor final_norm_moe_w, Tensor text_row_mask, int chunk_size) -> ()");
  m.def("halo_action_expert_step_forward("
        "int handle, Tensor x_emb, Tensor[] k_caches, Tensor[] v_caches, "
        "Tensor cos, Tensor sin, Tensor attn_mask_4d, int prefix_len) -> Tensor");

  // Hy-VLA VLM prefill（MoT 双权重，自 HALO 派生）
  m.def("hyvla_vlm_create",  TORCH_FN(rpu_hyvla_vlm_create));
  m.def("hyvla_vlm_destroy", TORCH_FN(rpu_hyvla_vlm_destroy));
  m.def("hyvla_vlm_set_weights("
        "int handle, "
        "Tensor[] q_w, Tensor[] k_w, Tensor[] v_w, Tensor[] o_w, "
        "Tensor[] q_norm, Tensor[] k_norm, "
        "Tensor[] input_norm, Tensor[] post_norm, "
        "Tensor[] gate_w, Tensor[] up_w, Tensor[] down_w, "
        "Tensor cos, Tensor sin, Tensor final_norm_w, "
        "int num_q_heads, int num_kv_heads, int head_dim, "
        "int hidden_size, int intermediate_size, float eps, bool use_silu, "
        "Tensor[] q_bias, Tensor[] k_bias, Tensor[] v_bias) -> ()");
  m.def("hyvla_vlm_set_weights_w8a16("
        "int handle, "
        "Tensor[] q_w, Tensor[] k_w, Tensor[] v_w, Tensor[] o_w, "
        "Tensor[] q_norm, Tensor[] k_norm, "
        "Tensor[] input_norm, Tensor[] post_norm, "
        "Tensor[] gate_w, Tensor[] up_w, Tensor[] down_w, "
        "Tensor cos, Tensor sin, Tensor final_norm_w, "
        "int num_q_heads, int num_kv_heads, int head_dim, "
        "int hidden_size, int intermediate_size, float eps, bool use_silu, "
        "Tensor[] q_bias, Tensor[] k_bias, Tensor[] v_bias, "
        "Tensor[] q_ws, Tensor[] k_ws, Tensor[] v_ws, Tensor[] o_ws, "
        "Tensor[] gate_ws, Tensor[] up_ws, Tensor[] down_ws) -> ()");
  m.def("hyvla_vlm_set_moe_weights("
        "int handle, "
        "Tensor[] q_w, Tensor[] k_w, Tensor[] v_w, Tensor[] o_w, "
        "Tensor[] q_norm, Tensor[] k_norm, "
        "Tensor[] input_norm, Tensor[] post_norm, "
        "Tensor[] gate_w, Tensor[] up_w, Tensor[] down_w, "
        "Tensor[] q_bias, Tensor[] k_bias, Tensor[] v_bias, "
        "Tensor final_norm_moe_w, Tensor text_row_mask, int chunk_size) -> ()");
  m.def("hyvla_vlm_set_moe_weights_w8a16("
        "int handle, "
        "Tensor[] q_w, Tensor[] k_w, Tensor[] v_w, Tensor[] o_w, "
        "Tensor[] q_norm, Tensor[] k_norm, "
        "Tensor[] input_norm, Tensor[] post_norm, "
        "Tensor[] gate_w, Tensor[] up_w, Tensor[] down_w, "
        "Tensor[] q_bias, Tensor[] k_bias, Tensor[] v_bias, "
        "Tensor final_norm_moe_w, Tensor text_row_mask, int chunk_size, "
        "Tensor[] q_ws, Tensor[] k_ws, Tensor[] v_ws, Tensor[] o_ws, "
        "Tensor[] gate_ws, Tensor[] up_ws, Tensor[] down_ws) -> ()");
  m.def("hyvla_vlm_step_forward("
        "int handle, Tensor x_emb, Tensor[] k_caches, Tensor[] v_caches, "
        "Tensor cos, Tensor sin, Tensor attn_mask_4d, int prefix_len) -> Tensor");
  // HyVlaExpertModel (action expert denoise, 单塔 rope→QK-norm)
  m.def("hyvla_expert_create",  TORCH_FN(rpu_hyvla_expert_create));
  m.def("hyvla_expert_destroy", TORCH_FN(rpu_hyvla_expert_destroy));
  m.def("hyvla_expert_set_weights("
        "int handle, "
        "Tensor[] q_w, Tensor[] k_w, Tensor[] v_w, Tensor[] o_w, "
        "Tensor[] q_norm, Tensor[] k_norm, "
        "Tensor[] input_norm, Tensor[] post_norm, "
        "Tensor[] gate_w, Tensor[] up_w, Tensor[] down_w, "
        "Tensor cos, Tensor sin, Tensor final_norm_w, "
        "int num_q_heads, int num_kv_heads, int head_dim, "
        "int hidden_size, int intermediate_size, float eps, "
        "int chunk_size) -> ()");
  m.def("hyvla_expert_set_weights_w8a16("
        "int handle, "
        "Tensor[] q_w, Tensor[] k_w, Tensor[] v_w, Tensor[] o_w, "
        "Tensor[] q_norm, Tensor[] k_norm, "
        "Tensor[] input_norm, Tensor[] post_norm, "
        "Tensor[] gate_w, Tensor[] up_w, Tensor[] down_w, "
        "Tensor cos, Tensor sin, Tensor final_norm_w, "
        "int num_q_heads, int num_kv_heads, int head_dim, "
        "int hidden_size, int intermediate_size, float eps, "
        "int chunk_size, "
        "Tensor[] q_ws, Tensor[] k_ws, Tensor[] v_ws, Tensor[] o_ws, "
        "Tensor[] gate_ws, Tensor[] up_ws, Tensor[] down_ws) -> ()");
  m.def("hyvla_expert_step_forward("
        "int handle, Tensor x_emb, Tensor[] k_caches, Tensor[] v_caches, "
        "Tensor cos, Tensor sin, Tensor attn_mask_4d, int prefix_len) -> Tensor");
  m.def("hyvla_expert_set_action_weights("
        "int handle, Tensor wc, Tensor mo, Tensor mo_bias, "
        "Tensor op, Tensor op_bias, Tensor time_all, "
        "int action_dim, int num_steps) -> ()");
  m.def("hyvla_expert_unroll_forward("
        "int handle, Tensor x0_rpu, Tensor[] k_caches, Tensor[] v_caches, "
        "Tensor state_emb, Tensor cos, Tensor sin, Tensor attn_mask_4d, "
        "Tensor x_traj, float dt, int prefix_len, int num_steps) -> ()");
  // HaloImageFlowModel (mode="gen" KV_FIRST 多块非因果 单流 gen 孪生)
  m.def("halo_image_flow_create",  TORCH_FN(rpu_halo_image_flow_create));
  m.def("halo_image_flow_destroy", TORCH_FN(rpu_halo_image_flow_destroy));
  // Per-handle chunk, replacing step_forward's per-forward read of
  // the deleted process global.
  m.def("halo_image_flow_set_chunk_size_override(int handle, int chunk_size) -> ()",
        TORCH_FN(rpu_halo_image_flow_set_chunk_size_override));
  m.def("halo_image_flow_set_weights("
        "int handle, "
        "Tensor[] q_w, Tensor[] k_w, Tensor[] v_w, Tensor[] o_w, "
        "Tensor[] q_norm, Tensor[] k_norm, "
        "Tensor[] input_norm, Tensor[] post_norm, "
        "Tensor[] gate_w, Tensor[] up_w, Tensor[] down_w, "
        "Tensor cos, Tensor sin, Tensor final_norm_w, "
        "int num_q_heads, int num_kv_heads, int head_dim, "
        "int hidden_size, int intermediate_size, float eps, bool use_silu, "
        "Tensor[] q_bias, Tensor[] k_bias, Tensor[] v_bias) -> ()");
  m.def("halo_image_flow_set_gen_weights("
        "int handle, "
        "Tensor[] q_w, Tensor[] k_w, Tensor[] v_w, Tensor[] o_w, "
        "Tensor[] q_norm, Tensor[] k_norm, "
        "Tensor[] input_norm, Tensor[] post_norm, "
        "Tensor[] gate_w, Tensor[] up_w, Tensor[] down_w, "
        "Tensor[] q_bias, Tensor[] k_bias, Tensor[] v_bias, "
        "Tensor final_norm_gen_w, Tensor row_mask, int query_len) -> ()");
  m.def("halo_image_flow_step_forward("
        "int handle, Tensor x_emb, Tensor[] k_caches, Tensor[] v_caches, "
        "Tensor cos, Tensor sin, int prefix_len) -> Tensor");
  // QwenPI05Model (QwenPI05 fused action expert, per-forward cos/sin)
  m.def("qwenpi05_create", TORCH_FN(rpu_qwenpi05_create));
  m.def("qwenpi05_destroy", TORCH_FN(rpu_qwenpi05_destroy));
  m.def("qwenpi05_set_weights(int handle, "
        "Tensor[] q_w_list, Tensor[] k_w_list, "
        "Tensor[] v_w_list, Tensor[] o_w_list, "
        "Tensor[] gate_list, Tensor[] up_list, Tensor[] down_list, "
        "Tensor[] attn_dense_w_list, Tensor[] attn_dense_b_list, "
        "Tensor[] mlp_dense_w_list, Tensor[] mlp_dense_b_list, "
        "Tensor[] q_norm_list, Tensor[] k_norm_list, "
        "int num_q_heads, int num_kv_heads, int head_dim, "
        "int hidden_size, int intermediate_size, float eps) -> ()");
  m.def("qwenpi05_forward(int handle, "
        "Tensor hidden_states, Tensor cond, "
        "Tensor[] k_caches, Tensor[] v_caches, "
        "Tensor cos, Tensor sin, "
        "Tensor? attention_mask, "
        "int position, bool is_causal) -> Tensor");
  m.def("qwenpi05_debug_flush_dumps", TORCH_FN(rpu_qwenpi05_debug_flush_dumps));
  // RhinoVLAModel — independent fork of QwenPI05 (same expert schema, separate op)
  m.def("rhino_vla_create", TORCH_FN(rpu_rhino_vla_create));
  m.def("rhino_vla_destroy", TORCH_FN(rpu_rhino_vla_destroy));
  m.def("rhino_vla_set_fast_replay(int handle, bool enabled) -> ()",
        TORCH_FN(rpu_rhino_vla_set_fast_replay));
  m.def("rhino_vla_set_denoise_loop_weights("
        "int handle, "
        "Tensor action_in_w, Tensor action_in_b, "
        "Tensor action_time_in_w, Tensor action_time_in_b, "
        "Tensor time_in_action_w, Tensor time_in_time_w, Tensor time_in_b, "
        "Tensor time_out_w, Tensor time_out_b, "
        "Tensor state_w, Tensor state_b, "
        "Tensor state_mask_w, Tensor state_mask_b, "
        "Tensor action_mask_w, Tensor action_mask_b, "
        "Tensor final_norm_w, Tensor final_norm_b, "
        "Tensor action_out_w, Tensor action_out_b, "
        "Tensor cos, Tensor sin, "
        "int action_dim, int action_dim_pad, "
        "int state_dim, int state_dim_pad, "
        "int action_horizon, int suffix_len) -> ()");
  m.def("rhino_vla_set_denoise_loop_adarms_tables("
        "int handle, Tensor pair_table, Tensor final_table) -> ()");
  m.def("rhino_vla_set_denoise_loop_time_proj_table("
        "int handle, Tensor table) -> ()");
  m.def("rhino_vla_set_weights(int handle, "
        "Tensor[] q_w_list, Tensor[] k_w_list, "
        "Tensor[] v_w_list, Tensor[] o_w_list, "
        "Tensor[] gate_list, Tensor[] up_list, Tensor[] down_list, "
        "Tensor[] attn_dense_w_list, Tensor[] attn_dense_b_list, "
        "Tensor[] mlp_dense_w_list, Tensor[] mlp_dense_b_list, "
        "Tensor[] pair_dense_w_list, Tensor[] pair_dense_b_list, "
        "Tensor[] q_norm_list, Tensor[] k_norm_list, "
        "int num_q_heads, int num_kv_heads, int head_dim, "
        "int hidden_size, int intermediate_size, float eps) -> ()");
  m.def("rhino_vla_forward(int handle, "
        "Tensor hidden_states, Tensor cond, "
        "Tensor[] k_caches, Tensor[] v_caches, "
        "Tensor cos, Tensor sin, "
        "Tensor? attention_mask, "
        "int position, bool is_causal) -> Tensor");
  m.def("rhino_vla_set_rope_position", TORCH_FN(rpu_rhino_vla_set_rope_position));
  m.def("rhino_vla_denoise_loop_forward(int handle, "
        "Tensor x0_rpu, Tensor[] k_caches, Tensor[] v_caches, "
        "Tensor cond_all, Tensor attention_mask, "
        "Tensor state_rpu, Tensor state_mask_rpu, Tensor action_mask_rpu, "
        "Tensor x_out, float dt, int prefix_len, int num_steps) -> Tensor");
  m.def("rhino_vla_debug_flush_dumps", TORCH_FN(rpu_rhino_vla_debug_flush_dumps));
  // SigLIPModel direct-handle all-layers-once API.
  m.def("siglip_create", TORCH_FN(rpu_siglip_create));
  m.def("siglip_destroy", TORCH_FN(rpu_siglip_destroy));
  m.def("siglip_set_weights(int handle, "
        "Tensor[] q_w_list, Tensor[] k_w_list, "
        "Tensor[] v_w_list, Tensor[] o_w_list, "
        "Tensor[] fc1_w_list, Tensor[] fc2_w_list, "
        "Tensor[] ln1_w_list, Tensor[] ln1_b_list, "
        "Tensor[] ln2_w_list, Tensor[] ln2_b_list, "
        "Tensor[] q_b_list, Tensor[] k_b_list, "
        "Tensor[] v_b_list, Tensor[] o_b_list, "
        "Tensor[] fc1_b_list, Tensor[] fc2_b_list, "
        "Tensor post_ln_w, Tensor post_ln_b, "
        "Tensor proj_w, Tensor proj_b, "
        "int num_heads, int head_dim, int hidden_size, "
        "int intermediate_size, int projection_dim, float eps) -> ()");
  m.def("siglip_set_weights_w8a16(int handle, "
        "Tensor[] q_w_list, Tensor[] k_w_list, "
        "Tensor[] v_w_list, Tensor[] o_w_list, "
        "Tensor[] fc1_w_list, Tensor[] fc2_w_list, "
        "Tensor[] ln1_w_list, Tensor[] ln1_b_list, "
        "Tensor[] ln2_w_list, Tensor[] ln2_b_list, "
        "Tensor[] q_b_list, Tensor[] k_b_list, "
        "Tensor[] v_b_list, Tensor[] o_b_list, "
        "Tensor[] fc1_b_list, Tensor[] fc2_b_list, "
        "Tensor post_ln_w, Tensor post_ln_b, "
        "Tensor proj_w, Tensor proj_b, "
        "int num_heads, int head_dim, int hidden_size, "
        "int intermediate_size, int projection_dim, float eps, "
        "Tensor[] q_ws_list, Tensor[] k_ws_list, "
        "Tensor[] v_ws_list, Tensor[] o_ws_list, "
        "Tensor[] fc1_ws_list, Tensor[] fc2_ws_list) -> ()");
  m.def("siglip_model_set_patch_emb(int handle, Tensor weight, Tensor pos_emb, int kernel_size, int stride) -> ()");
  m.def("siglip_set_chunk_size", TORCH_FN(rpu_siglip_set_chunk_size));
  m.def("siglip_get_resolved_chunk_size", TORCH_FN(rpu_siglip_get_resolved_chunk_size));
  m.def("siglip_patch_embed(int handle, Tensor input) -> Tensor");
  m.def("siglip_patch_embed_multi(int handle, Tensor[] images) -> Tensor");
  m.def("siglip_forward(int handle, Tensor input, Tensor[] k_caches, Tensor[] v_caches) -> Tensor");
  m.def("siglip_forward_packed(int handle, Tensor hidden, int image_batch_count, Tensor[] k_caches, Tensor[] v_caches) -> Tensor");
  m.def("siglip_forward_multi(int handle, Tensor[] images, Tensor[] k_caches, Tensor[] v_caches) -> Tensor");

  // HYViT2-400M AnyRes ViT — 自 SigLIP 派生；set_weights 无 post_ln_w/b（24 参）。
  m.def("hyvit2_create", TORCH_FN(rpu_hyvit2_create));
  m.def("hyvit2_destroy", TORCH_FN(rpu_hyvit2_destroy));
  m.def("hyvit2_set_weights(int handle, "
        "Tensor[] q_w_list, Tensor[] k_w_list, "
        "Tensor[] v_w_list, Tensor[] o_w_list, "
        "Tensor[] fc1_w_list, Tensor[] fc2_w_list, "
        "Tensor[] ln1_w_list, Tensor[] ln1_b_list, "
        "Tensor[] ln2_w_list, Tensor[] ln2_b_list, "
        "Tensor[] q_b_list, Tensor[] k_b_list, "
        "Tensor[] v_b_list, Tensor[] o_b_list, "
        "Tensor[] fc1_b_list, Tensor[] fc2_b_list, "
        "Tensor proj_w, Tensor proj_b, "
        "int num_heads, int head_dim, int hidden_size, "
        "int intermediate_size, int projection_dim, float eps) -> ()");
  m.def("hyvit2_set_weights_w8a16(int handle, "
        "Tensor[] q_w_list, Tensor[] k_w_list, "
        "Tensor[] v_w_list, Tensor[] o_w_list, "
        "Tensor[] fc1_w_list, Tensor[] fc2_w_list, "
        "Tensor[] ln1_w_list, Tensor[] ln1_b_list, "
        "Tensor[] ln2_w_list, Tensor[] ln2_b_list, "
        "Tensor[] q_b_list, Tensor[] k_b_list, "
        "Tensor[] v_b_list, Tensor[] o_b_list, "
        "Tensor[] fc1_b_list, Tensor[] fc2_b_list, "
        "Tensor proj_w, Tensor proj_b, "
        "int num_heads, int head_dim, int hidden_size, "
        "int intermediate_size, int projection_dim, float eps, "
        "Tensor[] q_ws_list, Tensor[] k_ws_list, "
        "Tensor[] v_ws_list, Tensor[] o_ws_list, "
        "Tensor[] fc1_ws_list, Tensor[] fc2_ws_list) -> ()");
  m.def("hyvit2_model_set_patch_emb(int handle, Tensor weight, Tensor pos_emb, int kernel_size, int stride) -> ()");
  m.def("hyvit2_patch_embed(int handle, Tensor input) -> Tensor");
  m.def("hyvit2_patch_embed_multi(int handle, Tensor[] images) -> Tensor");
  m.def("hyvit2_forward(int handle, Tensor input, Tensor[] k_caches, Tensor[] v_caches) -> Tensor");
  m.def("hyvit2_forward_packed(int handle, Tensor hidden, int image_batch_count, Tensor[] k_caches, Tensor[] v_caches) -> Tensor");
  m.def("hyvit2_forward_multi(int handle, Tensor[] images, Tensor[] k_caches, Tensor[] v_caches) -> Tensor");
  m.def("hyvla_merger_pool(Tensor nxc) -> Tensor");
  m.def("hyvla_merger_combine(Tensor nxc, Tensor sc) -> Tensor");
  m.def("hyvla_merger_fused(Tensor nxc, Tensor wa, Tensor b0, Tensor wb, Tensor w2, "
        "Tensor b2, Tensor wp2, Tensor bp2, Tensor? wp1=None, Tensor? bp1=None) "
        "-> Tensor");

  // Qwen3-VL Vision Encoder. See src/fused/rpu_qwen3vl_vision_model.cpp.
  m.def("qwen3vl_vision_create", TORCH_FN(rpu_qwen3vl_vision_create));
  m.def("qwen3vl_vision_destroy", TORCH_FN(rpu_qwen3vl_vision_destroy));
  m.def("qwen3vl_vision_set_weights(int handle, "
        "Tensor[] q_w_list, Tensor[] k_w_list, "
        "Tensor[] v_w_list, Tensor[] o_w_list, "
        "Tensor[] fc1_w_list, Tensor[] fc2_w_list, "
        "Tensor[] ln1_w_list, Tensor[] ln1_b_list, "
        "Tensor[] ln2_w_list, Tensor[] ln2_b_list, "
        "Tensor[] q_b_list, Tensor[] k_b_list, "
        "Tensor[] v_b_list, Tensor[] o_b_list, "
        "Tensor[] fc1_b_list, Tensor[] fc2_b_list, "
        "int num_heads, int head_dim, "
        "int hidden_size, int intermediate_size, "
        "float eps, int[] deepstack_visual_indexes=[]) -> ()");
  m.def("qwen3vl_vision_set_weights_w8a16(int handle, "
        "Tensor[] q_w_list, Tensor[] k_w_list, "
        "Tensor[] v_w_list, Tensor[] o_w_list, "
        "Tensor[] fc1_w_list, Tensor[] fc2_w_list, "
        "Tensor[] ln1_w_list, Tensor[] ln1_b_list, "
        "Tensor[] ln2_w_list, Tensor[] ln2_b_list, "
        "Tensor[] q_b_list, Tensor[] k_b_list, "
        "Tensor[] v_b_list, Tensor[] o_b_list, "
        "Tensor[] fc1_b_list, Tensor[] fc2_b_list, "
        "int num_heads, int head_dim, "
        "int hidden_size, int intermediate_size, "
        "float eps, int[] deepstack_visual_indexes, "
        "Tensor[] q_scale_list, Tensor[] k_scale_list, "
        "Tensor[] v_scale_list, Tensor[] o_scale_list, "
        "Tensor[] fc1_scale_list, Tensor[] fc2_scale_list) -> ()");
  m.def("qwen3vl_vision_set_rope(int handle, Tensor freq_cos, Tensor freq_sin) -> ()");
  m.def("qwen3vl_vision_set_fast_replay(int handle, bool enabled) -> ()",
        TORCH_FN(rpu_qwen3vl_vision_set_fast_replay));
  m.def("qwen3vl_vision_set_preload_replay_skip(int handle, bool enabled) -> ()",
        TORCH_FN(rpu_qwen3vl_vision_set_preload_replay_skip));
  m.def("qwen3vl_vision_set_chunk_size(int handle, int chunk_size) -> ()",
        TORCH_FN(rpu_qwen3vl_vision_set_chunk_size));
  m.def("qwen3vl_vision_get_resolved_chunk_size(int handle) -> int",
        TORCH_FN(rpu_qwen3vl_vision_get_resolved_chunk_size));
  m.def("qwen3vl_vision_set_bake_merger(int handle, bool enabled) -> ()",
        TORCH_FN(rpu_qwen3vl_vision_set_bake_merger));
  // Tensor targets, not integer addresses. The model retains
  // these, so the scatter cannot outlive its destination; the old int form held
  // no reference and depended on the caller keeping its tensors alive by
  // convention. Tensors are also the only form that can be checked for device /
  // dtype / contiguity, all of which the scatter DMA assumes.
  m.def("qwen3vl_vision_set_prefix_scatter(int handle, Tensor inputs_embeds, "
        "Tensor[] dense, int[] run_starts, int[] run_lengths) -> ()",
        TORCH_FN(rpu_qwen3vl_vision_set_prefix_scatter));
  m.def("qwen3vl_vision_set_merger_weights(int handle, Tensor ln_q_w, Tensor ln_q_b, "
        "Tensor m0_w, Tensor m0_b, Tensor m2_w, Tensor m2_b, int out_hidden, float ln_eps) -> ()");
  m.def("qwen3vl_vision_set_deepstack_merger_weights(int handle, Tensor[] ln_q_w, Tensor[] ln_q_b, "
        "Tensor[] m0_w, Tensor[] m0_b, Tensor[] m2_w, Tensor[] m2_b) -> ()");
  m.def("qwen3vl_vision_pop_deepstack_merged(int handle) -> Tensor[]",
        TORCH_FN(rpu_qwen3vl_vision_pop_deepstack_merged));
  m.def("qwen3vl_vision_pop_pooler_merged(int handle) -> Tensor",
        TORCH_FN(rpu_qwen3vl_vision_pop_pooler_merged));
  m.def("qwen3vl_vision_pop_merged(int handle) -> Tensor[]",
        TORCH_FN(rpu_qwen3vl_vision_pop_merged));
  // These two have only an int handle as input — there is no input Tensor for
  // PyTorch's backend dispatch to extract a device key from. Register the
  // implementation directly via TORCH_FN at m.def time so it runs for every
  // backend (no backend-specific kernel needed — both ops just return tensors
  // stored as members on the C++ model instance).
  m.def("qwen3vl_vision_position_idx_keepalive(int handle) -> Tensor",
        TORCH_FN(rpu_qwen3vl_vision_position_idx_keepalive));
  m.def("qwen3vl_vision_forward(int handle, Tensor input, "
        "Tensor[] k_caches, Tensor[] v_caches, int num_patches, "
        "int image_batch_count=1) -> Tensor");
  m.def("qwen3vl_vision_pop_deepstack_snapshots(int handle) -> Tensor[]",
        TORCH_FN(rpu_qwen3vl_vision_pop_deepstack_snapshots));

  // Qwen3.5 Vision Encoder (Option 2 — qwen3vl vision − DeepStack).
  // See src/fused/rpu_qwen3_5_vision_model.cpp. No pop_deepstack op; set_weights
  // has no deepstack_visual_indexes arg.
  m.def("qwen3_5_vision_create", TORCH_FN(rpu_qwen3_5_vision_create));
  m.def("qwen3_5_vision_destroy", TORCH_FN(rpu_qwen3_5_vision_destroy));
  // Resolved chunk_size getter (int -> int, DEFAULT block only — no device dispatch,
  // same as qwen3_5_get_resolved_chunk_size). Read from Python after a forward.
  m.def("qwen3_5_vision_get_resolved_chunk_size", TORCH_FN(rpu_qwen3_5_vision_get_resolved_chunk_size));
  // Per-handle vision chunk cap. Two live handles can now hold
  // DIFFERENT caps; before this the first forward in the process pinned one
  // cap for every handle created afterwards.
  m.def("qwen3_5_vision_set_chunk_size_cap", TORCH_FN(rpu_qwen3_5_vision_set_chunk_size_cap));
  // Per-layer debug snapshots (int -> Tensor, DEFAULT block). Read after a get_debug_export() forward.
  m.def("qwen3_5_vision_get_dbg_hidden", TORCH_FN(rpu_qwen3_5_vision_get_dbg_hidden));
  m.def("qwen3_5_vision_get_dbg_q", TORCH_FN(rpu_qwen3_5_vision_get_dbg_q));
  m.def("qwen3_5_vision_set_weights(int handle, "
        "Tensor[] q_w_list, Tensor[] k_w_list, "
        "Tensor[] v_w_list, Tensor[] o_w_list, "
        "Tensor[] fc1_w_list, Tensor[] fc2_w_list, "
        "Tensor[] ln1_w_list, Tensor[] ln1_b_list, "
        "Tensor[] ln2_w_list, Tensor[] ln2_b_list, "
        "Tensor[] q_b_list, Tensor[] k_b_list, "
        "Tensor[] v_b_list, Tensor[] o_b_list, "
        "Tensor[] fc1_b_list, Tensor[] fc2_b_list, "
        "int num_heads, int head_dim, "
        "int hidden_size, int intermediate_size, "
        "float eps) -> ()");
  m.def("qwen3_5_vision_set_rope(int handle, Tensor freq_cos, Tensor freq_sin) -> ()");
  // Int-handle-only ops (no input Tensor for backend dispatch) — register the
  // implementation directly via TORCH_FN at m.def time, like the qwen3vl ones.
  m.def("qwen3_5_vision_position_idx_keepalive(int handle) -> Tensor",
        TORCH_FN(rpu_qwen3_5_vision_position_idx_keepalive));
  // STEP 0 opt-in (patch_embed + uploaded HF-exact positions in-graph).
  // set_patch_embed flips forward() to raw folded patches; forward receives the
  // per-grid position tensor through a mutable-source DMA.
  m.def("qwen3_5_vision_set_patch_embed(int handle, Tensor pe_w, Tensor pe_b) -> ()");
  // Patch merger opt-in (LN → reshape → fc1/GELU/fc2 → text hidden, in-graph as a
  // post_layers_fn). See set_merger in rpu_qwen3_5_vision_model.h.
  m.def("qwen3_5_vision_set_merger(int handle, Tensor fc1_w, Tensor fc1_b, "
        "Tensor fc2_w, Tensor fc2_b, Tensor norm_w, Tensor norm_b, "
        "int out_hidden_size) -> ()");
  m.def("qwen3_5_vision_set_temporal(int handle, Tensor temporal_pe, "
        "int num_frames, int patches_per_frame) -> ()");
  // Cold exact-profile policy. No Tensor argument, so register directly in
  // the default block rather than relying on device dispatch.
  m.def("qwen3_5_vision_set_allreduce_force_non_v2",
        TORCH_FN(rpu_qwen3_5_vision_set_allreduce_force_non_v2));
  m.def("qwen3_5_vision_get_allreduce_force_non_v2",
        TORCH_FN(rpu_qwen3_5_vision_get_allreduce_force_non_v2));
  m.def("qwen3_5_vision_get_merger_out(int handle) -> Tensor",
        TORCH_FN(rpu_qwen3_5_vision_get_merger_out));
  m.def("qwen3_5_vision_forward(int handle, Tensor input, "
        "Tensor[] k_caches, Tensor[] v_caches, int num_patches, "
        "Tensor? step0_pos=None, "
        "Tensor(a!)? fusion_target=None, int[] fusion_row_starts=[], "
        "int temporal_num_frames=1, int camera_batch_count=1) -> Tensor");

  // Qwen2.5-VL Vision Encoder. See src/fused/rpu_qwen25vl_vision_model.cpp.
  m.def("qwen25vl_vision_create", TORCH_FN(rpu_qwen25vl_vision_create));
  m.def("qwen25vl_vision_destroy", TORCH_FN(rpu_qwen25vl_vision_destroy));
  m.def("qwen25vl_vision_set_per_window_sdpa",
        TORCH_FN(rpu_qwen25vl_vision_set_per_window_sdpa));
  m.def("qwen25vl_vision_set_weights(int handle, "
        "Tensor[] q_w_list, Tensor[] k_w_list, "
        "Tensor[] v_w_list, Tensor[] o_w_list, "
        "Tensor[] gate_w_list, Tensor[] up_w_list, Tensor[] down_w_list, "
        "Tensor[] norm1_w_list, Tensor[] norm2_w_list, "
        "Tensor[] q_b_list, Tensor[] k_b_list, "
        "Tensor[] v_b_list, Tensor[] o_b_list, "
        "Tensor[] gate_b_list, Tensor[] up_b_list, Tensor[] down_b_list, "
        "int num_heads, int head_dim, "
        "int hidden_size, int intermediate_size, "
        "float eps, int[] fullatt_block_indexes=[]) -> ()");
  m.def("qwen25vl_vision_set_weights_w8a16(int handle, "
        "Tensor[] q_w_list, Tensor[] k_w_list, "
        "Tensor[] v_w_list, Tensor[] o_w_list, "
        "Tensor[] gate_w_list, Tensor[] up_w_list, Tensor[] down_w_list, "
        "Tensor[] norm1_w_list, Tensor[] norm2_w_list, "
        "Tensor[] q_b_list, Tensor[] k_b_list, "
        "Tensor[] v_b_list, Tensor[] o_b_list, "
        "Tensor[] gate_b_list, Tensor[] up_b_list, Tensor[] down_b_list, "
        "int num_heads, int head_dim, "
        "int hidden_size, int intermediate_size, "
        "float eps, int[] fullatt_block_indexes, "
        "Tensor[] q_w_scale_list, Tensor[] k_w_scale_list, "
        "Tensor[] v_w_scale_list, Tensor[] o_w_scale_list, "
        "Tensor[] gate_scale_list, Tensor[] up_scale_list, "
        "Tensor[] down_scale_list) -> ()");
  m.def("qwen25vl_vision_set_merger_weights(int handle, Tensor ln_q_w, "
        "Tensor m0_w, Tensor m0_b, Tensor m2_w, Tensor m2_b, "
        "int out_hidden) -> ()");
  m.def("qwen25vl_vision_set_rope(int handle, Tensor freq_cos, Tensor freq_sin) -> ()");
  // No Tensor argument means the dispatcher cannot infer PrivateUse1.  Bind
  // the implementation in the schema block, like the Qwen3-VL/DINO control
  // ops above, so both auto (0) and exact per-handle requests are callable.
  m.def("qwen25vl_vision_set_chunk_size(int handle, int chunk_size) -> ()",
        TORCH_FN(rpu_qwen25vl_vision_set_chunk_size));
  m.def("qwen25vl_vision_position_idx_keepalive(int handle) -> Tensor",
        TORCH_FN(rpu_qwen25vl_vision_position_idx_keepalive));
  m.def("qwen25vl_vision_forward(int handle, Tensor input, "
        "Tensor[] k_caches, Tensor[] v_caches, int num_patches, "
        "Tensor? window_mask=None, int image_batch_count=1, "
        "Tensor? cu_window_seqlens=None) -> Tensor");

  // DINOv3 ViT Vision Encoder. See src/fused/rpu_dinov3_vision_model.cpp.
  m.def("dinov3_vision_create", TORCH_FN(rpu_dinov3_vision_create));
  m.def("dinov3_vision_destroy", TORCH_FN(rpu_dinov3_vision_destroy));
  m.def("dinov3_vision_set_weights(int handle, "
        "Tensor[] q_w_list, Tensor[] k_w_list, "
        "Tensor[] v_w_list, Tensor[] o_w_list, "
        "Tensor[] up_w_list, Tensor[] down_w_list, "
        "Tensor[] ln1_w_list, Tensor[] ln1_b_list, "
        "Tensor[] ln2_w_list, Tensor[] ln2_b_list, "
        "Tensor[] q_b_list, Tensor[] v_b_list, Tensor[] o_b_list, "
        "Tensor[] up_b_list, Tensor[] down_b_list, "
        "Tensor[] gamma_attn_list, Tensor[] gamma_mlp_list, "
        "int num_heads, int head_dim, "
        "int hidden_size, int intermediate_size, "
        "float eps) -> ()");
  m.def("dinov3_vision_set_rope(int handle, Tensor freq_cos, Tensor freq_sin) -> ()");
  // No input Tensor for backend dispatch — register implementation directly.
  m.def("dinov3_vision_position_idx_keepalive(int handle) -> Tensor",
        TORCH_FN(rpu_dinov3_vision_position_idx_keepalive));
  m.def("dinov3_vision_set_chunk_size",
        TORCH_FN(rpu_dinov3_vision_set_chunk_size));
  m.def("dinov3_vision_resolve_chunk_size",
        TORCH_FN(rpu_dinov3_vision_resolve_chunk_size));
  m.def("dinov3_vision_get_resolved_chunk_size",
        TORCH_FN(rpu_dinov3_vision_get_resolved_chunk_size));
  m.def("dinov3_vision_forward(int handle, Tensor input, "
        "Tensor[] k_caches, Tensor[] v_caches, int num_tokens) -> Tensor");
  m.def("dinov3_vision_set_acc32(int handle, bool enabled) -> ()",
        TORCH_FN(rpu_dinov3_vision_set_acc32));
  m.def("dinov3_vision_set_16b_sdpa(int handle, bool enabled) -> ()",
        TORCH_FN(rpu_dinov3_vision_set_16b_sdpa));
  m.def("dinov3_vision_set_identity_rope(int handle, bool enabled) -> ()",
        TORCH_FN(rpu_dinov3_vision_set_identity_rope));
  m.def("dinov3_vision_set_final_norm(int handle, Tensor weight, Tensor bias) -> ()");
  m.def("dinov3_vision_set_output_mlp(int handle, Tensor first_weight, "
        "Tensor first_bias, Tensor second_weight, Tensor second_bias) -> ()");

  // Process-global public reset: invalidate every live top-level Graph while
  // preserving cache key slots so the next admitted capture records afresh.
  m.def("reset_graph_cache", TORCH_FN(invalidate_registered_rpu_kernel_graphs));
  // Release conv2d multi-core SPM cache
  m.def("release_conv2d_spm_cache", &rpu_release_conv2d_spm_cache);
  // SpmAllocator
  m.def("spm_alloc_init", &process_guarded_spm_alloc_init);
  // graph-aware reset_temporary：PASSTHROUGH/BUILT 走原 SPM_ALLOC
  // .reset_temporary();RECORDING 期 mark_non_replayable("spm_alloc_reset_temporary
  // during RECORDING") + 仍执行 host reset；REPLAYING fail-loud。
  m.def("spm_alloc_reset_temporary", &graph_aware_spm_alloc_reset_temporary);
  m.def("spm_alloc_reset_all", &process_guarded_spm_alloc_reset_all);
  m.def("spm_alloc_dump", [](const std::string& label) { SPM_ALLOC.dump(label.c_str()); });

  // Debug-only LingBot2 router and intermediate read-back helpers.
  m.def("lingbot_v2_moe_debug_rw_stage", &rpu_lingbot_v2_moe_debug_rw_stage);
  m.def("lingbot_v2_moe_debug_wtm_stage", &rpu_lingbot_v2_moe_debug_wtm_stage);
  m.def("lingbot_v2_moe_debug_l0_out", &rpu_lingbot_v2_moe_debug_l0_out);
  m.def("lingbot_v2_moe_debug_innorm", &rpu_lingbot_v2_moe_debug_innorm);
  m.def("lingbot_v2_moe_debug_gcap", &rpu_lingbot_v2_moe_debug_gcap);
  m.def("lingbot_v2_moe_debug_acap", &rpu_lingbot_v2_moe_debug_acap);
  m.def("lingbot_v2_moe_debug_scap", &rpu_lingbot_v2_moe_debug_scap);
  m.def("lingbot_v2_moe_debug_ccap", &rpu_lingbot_v2_moe_debug_ccap);
  m.def("lingbot_v2_moe_debug_router_h", &rpu_lingbot_v2_moe_debug_router_h);
  m.def("lingbot_v2_moe_debug_layer_in", &rpu_lingbot_v2_moe_debug_layer_in);
  m.def("lingbot_v2_moe_debug_attn_resid", &rpu_lingbot_v2_moe_debug_attn_resid);
  m.def("lingbot_v2_moe_debug_packed", &rpu_lingbot_v2_moe_debug_packed);

}

// Implementation for RPU device
TORCH_LIBRARY_IMPL(rpu, PrivateUse1, m) {
  m.impl("conv2d_mc_nhwc", TORCH_FN(rpu_conv2d_mc_nhwc));
  m.impl("rms_norm", TORCH_FN(rpu_rmsnorm));
  m.impl("apply_rotary_pos_emb", TORCH_FN(rpu_apply_rotary_pos_emb));
  m.impl("scaled_dot_product_attention_unified", TORCH_FN(rpu_scaled_dot_product_attention_unified));
  m.impl("causal_decoder_set_weights", TORCH_FN(rpu_causal_decoder_set_weights));
  m.impl("causal_decoder_set_weights_w8a16", TORCH_FN(rpu_causal_decoder_set_weights_w8a16));
  m.impl("causal_decoder_set_weights_nvfp4", TORCH_FN(rpu_causal_decoder_set_weights_nvfp4));
  m.impl("causal_decoder_set_adarms_step", TORCH_FN(rpu_causal_decoder_set_adarms_step));
  m.impl("causal_decoder_set_adarms_step_mutable", TORCH_FN(rpu_causal_decoder_set_adarms_step_mutable));
  m.impl("causal_decoder_forward", TORCH_FN(rpu_causal_decoder_forward));
  m.impl("qwen3_5_set_weights", TORCH_FN(rpu_qwen3_5_set_weights));
  m.impl("qwen3_5_forward", TORCH_FN(rpu_qwen3_5_forward));
  m.impl("qwen3_5_action_forward", TORCH_FN(rpu_qwen3_5_action_forward));
  m.impl("qwen3_5_set_action_io_weights", TORCH_FN(rpu_qwen3_5_set_action_io_weights));
  m.impl("qwen3_5_action_step_forward", TORCH_FN(rpu_qwen3_5_action_step_forward));
  m.impl("qwen3_5_action_rtc_step_forward", TORCH_FN(rpu_qwen3_5_action_rtc_step_forward));
  m.impl("qwen3_5_action_rtc_trace_forward", TORCH_FN(rpu_qwen3_5_action_rtc_trace_forward));
  m.impl("qwen3_5_set_prefill_rope", TORCH_FN(rpu_qwen3_5_set_prefill_rope));
  m.impl("qwen3_5_spm_z2_pipeline_forward", TORCH_FN(rpu_qwen3_5_spm_z2_pipeline_forward));
  m.impl("wall_oss_spm_z1_pipeline_forward", TORCH_FN(rpu_wall_oss_spm_z1_pipeline_forward));
  m.impl("qwen3vl_multiview_spm_z2_canary_forward", TORCH_FN(rpu_qwen3vl_multiview_spm_z2_canary_forward));
  m.impl("lingbot2_multiview_spm_z2_forward", TORCH_FN(rpu_lingbot2_multiview_spm_z2_forward));
  m.impl("qwen3vl_pooler_spm_z1_pipeline_forward", TORCH_FN(rpu_qwen3vl_pooler_spm_z1_pipeline_forward));
  m.impl("causal_decoder_set_lm_head", TORCH_FN(rpu_causal_decoder_set_lm_head));
  m.impl("lm_head_logits_top1", TORCH_FN(rpu_lm_head_logits_top1));
  m.impl("argmax_lastdim_host", TORCH_FN(rpu_argmax_lastdim_host));
  m.impl("gemma_set_weights", TORCH_FN(rpu_gemma_set_weights));
  m.impl("gemma_forward", TORCH_FN(rpu_gemma_forward));
  m.impl("gemma_set_prefill_rope", TORCH_FN(rpu_gemma_set_prefill_rope));
  m.impl("gemma4_set_weights", TORCH_FN(rpu_gemma4_set_weights));
  m.impl("gemma4_forward", TORCH_FN(rpu_gemma4_forward));
  m.impl("adarms_set_weights", TORCH_FN(rpu_adarms_set_weights));
  m.impl("adarms_forward", TORCH_FN(rpu_adarms_forward));
  m.impl("gr00t_dit_set_weights", TORCH_FN(rpu_gr00t_dit_set_weights));
  m.impl("gr00t_dit_set_weights_w8a16", TORCH_FN(rpu_gr00t_dit_set_weights_w8a16));
  m.impl("gr00t_dit_forward", TORCH_FN(rpu_gr00t_dit_forward));
  m.impl("internvla_nextdit_set_weights", TORCH_FN(rpu_internvla_nextdit_set_weights));
  m.impl("internvla_nextdit_set_fused_qkq_weights",
         TORCH_FN(rpu_internvla_nextdit_set_fused_qkq_weights));
  m.impl("internvla_nextdit_set_step_prepare_weights",
         TORCH_FN(rpu_internvla_nextdit_set_step_prepare_weights));
  m.impl("internvla_nextdit_forward", TORCH_FN(rpu_internvla_nextdit_forward));
  m.impl("gr00t_vl_encoder_set_weights", TORCH_FN(rpu_gr00t_vl_encoder_set_weights));
  m.impl("gr00t_vl_encoder_set_weights_w8a16", TORCH_FN(rpu_gr00t_vl_encoder_set_weights_w8a16));
  m.impl("gr00t_vl_encoder_forward", TORCH_FN(rpu_gr00t_vl_encoder_forward));
  m.impl("gr00t_vl_encoder_forward_z2", TORCH_FN(rpu_gr00t_vl_encoder_forward_z2));
  m.impl("gr00t_denoise_set_weights", TORCH_FN(rpu_gr00t_denoise_set_weights));
  m.impl("gr00t_denoise_step_forward", TORCH_FN(rpu_gr00t_denoise_step_forward));
  m.impl("gr00t_denoise_unroll_forward", TORCH_FN(rpu_gr00t_denoise_unroll_forward));
  m.impl("gr00t_denoise_unroll_forward_z2", TORCH_FN(rpu_gr00t_denoise_unroll_forward_z2));
  m.impl("gr00t_spm_z2_prepare_masks", TORCH_FN(rpu_gr00t_spm_z2_prepare_masks));
  m.impl("gr00t_spm_z2_pipeline_forward", TORCH_FN(rpu_gr00t_spm_z2_pipeline_forward));
  m.impl("pi05_set_weights", TORCH_FN(rpu_pi05_set_weights));
  m.impl("pi05_forward", TORCH_FN(rpu_pi05_forward));
  m.impl("pi05_denoise_step_set_weights", TORCH_FN(rpu_pi05_denoise_step_set_weights));
  m.impl("pi05_denoise_step_forward",     TORCH_FN(rpu_pi05_denoise_step_forward));
  m.impl("pi05_denoise_loop_forward",     TORCH_FN(rpu_pi05_denoise_loop_forward));
  m.impl("wall_oss_action_step_set_weights", TORCH_FN(rpu_wall_oss_action_step_set_weights));
  m.impl("wall_oss_action_step_forward",     TORCH_FN(rpu_wall_oss_action_step_forward));
  m.impl("wall_oss_action_denoise_loop_forward", TORCH_FN(rpu_wall_oss_action_denoise_loop_forward));
  m.impl("lingbot_v2_moe_set_weights", TORCH_FN(rpu_lingbot_v2_moe_set_weights));
  m.impl("lingbot_v2_moe_set_packed_weights",
         TORCH_FN(rpu_lingbot_v2_moe_set_packed_weights));
  m.impl("lingbot_v2_moe_set_packed_scales",
         TORCH_FN(rpu_lingbot_v2_moe_set_packed_scales));
  m.impl("lingbot_v2_moe_set_base_scales",
         TORCH_FN(rpu_lingbot_v2_moe_set_base_scales));
  m.impl("lingbot_v2_moe_set_adarms_step_mutable",
         TORCH_FN(rpu_lingbot_v2_moe_set_adarms_step_mutable));
  m.impl("lingbot_v2_moe_bind_adarms_schedule",
         TORCH_FN(rpu_lingbot_v2_moe_bind_adarms_schedule));
  m.impl("lingbot_v2_moe_set_denoise_unroll",
         TORCH_FN(rpu_lingbot_v2_moe_set_denoise_unroll));
  m.impl("lingbot_v2_moe_denoise_unroll_forward",
         TORCH_FN(rpu_lingbot_v2_moe_denoise_unroll_forward));
  m.impl("lingbot_v2_moe_forward", TORCH_FN(rpu_lingbot_v2_moe_forward));
  // HALO
  m.impl("halo_action_expert_set_weights",     TORCH_FN(rpu_halo_action_expert_set_weights));
  m.impl("halo_action_expert_set_moe_weights", TORCH_FN(rpu_halo_action_expert_set_moe_weights));
  m.impl("halo_action_expert_step_forward",    TORCH_FN(rpu_halo_action_expert_step_forward));
  m.impl("hyvla_vlm_set_weights",     TORCH_FN(rpu_hyvla_vlm_set_weights));
  m.impl("hyvla_vlm_set_weights_w8a16",     TORCH_FN(rpu_hyvla_vlm_set_weights_w8a16));
  m.impl("hyvla_vlm_set_moe_weights", TORCH_FN(rpu_hyvla_vlm_set_moe_weights));
  m.impl("hyvla_vlm_set_moe_weights_w8a16", TORCH_FN(rpu_hyvla_vlm_set_moe_weights_w8a16));
  m.impl("hyvla_vlm_step_forward",    TORCH_FN(rpu_hyvla_vlm_step_forward));
  m.impl("hyvla_expert_set_weights",  TORCH_FN(rpu_hyvla_expert_set_weights));
  m.impl("hyvla_expert_set_weights_w8a16",  TORCH_FN(rpu_hyvla_expert_set_weights_w8a16));
  m.impl("hyvla_expert_step_forward", TORCH_FN(rpu_hyvla_expert_step_forward));
  m.impl("hyvla_expert_set_action_weights", TORCH_FN(rpu_hyvla_expert_set_action_weights));
  m.impl("hyvla_expert_unroll_forward",     TORCH_FN(rpu_hyvla_expert_unroll_forward));
  m.impl("halo_image_flow_set_weights", TORCH_FN(rpu_halo_image_flow_set_weights));
  m.impl("halo_image_flow_set_gen_weights", TORCH_FN(rpu_halo_image_flow_set_gen_weights));
  m.impl("halo_image_flow_step_forward",TORCH_FN(rpu_halo_image_flow_step_forward));
  m.impl("qwenpi05_set_weights", TORCH_FN(rpu_qwenpi05_set_weights));
  m.impl("qwenpi05_forward", TORCH_FN(rpu_qwenpi05_forward));
  m.impl("qwenpi05_debug_flush_dumps", TORCH_FN(rpu_qwenpi05_debug_flush_dumps));
  m.impl("rhino_vla_set_weights", TORCH_FN(rpu_rhino_vla_set_weights));
  m.impl("rhino_vla_forward", TORCH_FN(rpu_rhino_vla_forward));
  m.impl("rhino_vla_set_denoise_loop_weights",
         TORCH_FN(rpu_rhino_vla_set_denoise_loop_weights));
  m.impl("rhino_vla_set_denoise_loop_adarms_tables",
         TORCH_FN(rpu_rhino_vla_set_denoise_loop_adarms_tables));
  m.impl("rhino_vla_set_denoise_loop_time_proj_table",
         TORCH_FN(rpu_rhino_vla_set_denoise_loop_time_proj_table));
  m.impl("rhino_vla_denoise_loop_forward",
         TORCH_FN(rpu_rhino_vla_denoise_loop_forward));
  m.impl("rhino_vla_debug_flush_dumps", TORCH_FN(rpu_rhino_vla_debug_flush_dumps));
  m.impl("navdp_set_weights", TORCH_FN(rpu_navdp_set_weights));
  m.impl("navdp_forward", TORCH_FN(rpu_navdp_forward));
  m.impl("navdp_set_denoise_glue", TORCH_FN(rpu_navdp_set_denoise_glue));
  m.impl("navdp_denoise_loop_forward", TORCH_FN(rpu_navdp_denoise_loop_forward));
  // SigLIP uses the all-layers model path below.
  m.impl("siglip_set_weights", TORCH_FN(rpu_siglip_set_weights));
  m.impl("siglip_set_weights_w8a16", TORCH_FN(rpu_siglip_set_weights_w8a16));
  m.impl("siglip_model_set_patch_emb", TORCH_FN(rpu_siglip_model_set_patch_emb));
  m.impl("siglip_patch_embed", TORCH_FN(rpu_siglip_patch_embed));
  m.impl("siglip_patch_embed_multi", TORCH_FN(rpu_siglip_patch_embed_multi));
  m.impl("siglip_forward", TORCH_FN(rpu_siglip_forward));
  m.impl("siglip_forward_packed", TORCH_FN(rpu_siglip_forward_packed));
  m.impl("siglip_forward_multi", TORCH_FN(rpu_siglip_forward_multi));

  // HYViT2-400M AnyRes ViT
  m.impl("hyvit2_set_weights", TORCH_FN(rpu_hyvit2_set_weights));
  m.impl("hyvit2_set_weights_w8a16", TORCH_FN(rpu_hyvit2_set_weights_w8a16));
  m.impl("hyvit2_model_set_patch_emb", TORCH_FN(rpu_hyvit2_model_set_patch_emb));
  m.impl("hyvit2_patch_embed", TORCH_FN(rpu_hyvit2_patch_embed));
  m.impl("hyvit2_patch_embed_multi", TORCH_FN(rpu_hyvit2_patch_embed_multi));
  m.impl("hyvit2_forward", TORCH_FN(rpu_hyvit2_forward));
  m.impl("hyvit2_forward_packed", TORCH_FN(rpu_hyvit2_forward_packed));
  m.impl("hyvit2_forward_multi", TORCH_FN(rpu_hyvit2_forward_multi));
  m.impl("hyvla_merger_pool", TORCH_FN(rpu_hyvla_merger_pool));
  m.impl("hyvla_merger_combine", TORCH_FN(rpu_hyvla_merger_combine));
  m.impl("hyvla_merger_fused", TORCH_FN(rpu_hyvla_merger_fused));

  // Qwen3-VL Vision Encoder
  m.impl("qwen3vl_vision_set_weights", TORCH_FN(rpu_qwen3vl_vision_set_weights));
  m.impl("qwen3vl_vision_set_weights_w8a16", TORCH_FN(rpu_qwen3vl_vision_set_weights_w8a16));
  m.impl("qwen3vl_vision_set_rope", TORCH_FN(rpu_qwen3vl_vision_set_rope));
  m.impl("qwen3vl_vision_set_merger_weights", TORCH_FN(rpu_qwen3vl_vision_set_merger_weights));
  m.impl("qwen3vl_vision_set_deepstack_merger_weights", TORCH_FN(rpu_qwen3vl_vision_set_deepstack_merger_weights));
  m.impl("qwen3vl_vision_forward", TORCH_FN(rpu_qwen3vl_vision_forward));

  // Qwen3.5 Vision Encoder (Option 2 — qwen3vl vision − DeepStack)
  m.impl("qwen3_5_vision_set_weights", TORCH_FN(rpu_qwen3_5_vision_set_weights));
  m.impl("qwen3_5_vision_set_rope", TORCH_FN(rpu_qwen3_5_vision_set_rope));
  m.impl("qwen3_5_vision_set_patch_embed", TORCH_FN(rpu_qwen3_5_vision_set_patch_embed));
  m.impl("qwen3_5_vision_set_merger", TORCH_FN(rpu_qwen3_5_vision_set_merger));
  m.impl("qwen3_5_vision_set_temporal", TORCH_FN(rpu_qwen3_5_vision_set_temporal));
  m.impl("qwen3_5_vision_forward", TORCH_FN(rpu_qwen3_5_vision_forward));

  // Qwen2.5-VL Vision Encoder
  m.impl("qwen25vl_vision_set_weights", TORCH_FN(rpu_qwen25vl_vision_set_weights));
  m.impl("qwen25vl_vision_set_weights_w8a16", TORCH_FN(rpu_qwen25vl_vision_set_weights_w8a16));
  m.impl("qwen25vl_vision_set_merger_weights", TORCH_FN(rpu_qwen25vl_vision_set_merger_weights));
  m.impl("qwen25vl_vision_set_rope", TORCH_FN(rpu_qwen25vl_vision_set_rope));
  m.impl("qwen25vl_vision_forward", TORCH_FN(rpu_qwen25vl_vision_forward));

  // DINOv3 ViT Vision Encoder
  m.impl("dinov3_vision_set_weights", TORCH_FN(rpu_dinov3_vision_set_weights));
  m.impl("dinov3_vision_set_rope", TORCH_FN(rpu_dinov3_vision_set_rope));
  m.impl("dinov3_vision_forward", TORCH_FN(rpu_dinov3_vision_forward));
  m.impl("dinov3_vision_set_final_norm",
         TORCH_FN(rpu_dinov3_vision_set_final_norm));
  m.impl("dinov3_vision_set_output_mlp",
         TORCH_FN(rpu_dinov3_vision_set_output_mlp));
}

TORCH_LIBRARY_IMPL(rpu, CPU, m) {
  m.impl("argmax_lastdim_host", TORCH_FN(rpu_argmax_lastdim_host));
}

// AutogradPrivateUse1 注册成 per-op fallthrough,而不是直接调真 impl。
// 原因:Autograd 在 dispatch key 优先级里高于 Python,如果 AutogradPrivateUse1
// 注册成 [kernel] 直跳真实现,FakeTensor 在 rpu device 上的 keyset (含
// AutogradPrivateUse1) 会让 dispatcher 选这条路 → 撞 PrivateUse1 真实现 →
// 撞 fake storage。Fallthrough 让 Autograd 跳过去,dispatcher 继续往下找,
// 命中 Python(FakeTensor 路径)或 PrivateUse1(真实部署路径)。
//
// PT 不允许 per-namespace m.fallback(),只能 per-op 注册 fallthrough。
//
// Inference 路径:tensor 不带 requires_grad,AutogradPrivateUse1 不在 keyset,
//                直接 PrivateUse1 win,行为不变。
// 训练路径:tensor.requires_grad=True 时 AutogradPrivateUse1 在 keyset,
//          fallthrough 让 dispatcher 走到 PrivateUse1,但不会自动追踪 grad
//          —— 我们的 op 本来就不支持 autograd(inference-only),这是正确语义。
//
// 配套 Python 端:fake_dispatch.py 给每个 op 注册 @torch.library.impl(..., "Python")
// shim,Dynamo / FakeTensorMode 路径上 dispatcher 命中 Python 调 shim 返回
// torch.empty_like(...)。
TORCH_LIBRARY_IMPL(rpu, AutogradPrivateUse1, m) {
  m.impl("conv2d_mc_nhwc", torch::CppFunction::makeFallthrough());
  m.impl("rms_norm", torch::CppFunction::makeFallthrough());
  m.impl("apply_rotary_pos_emb", torch::CppFunction::makeFallthrough());
  m.impl("scaled_dot_product_attention_unified",
         torch::CppFunction::makeFallthrough());
  m.impl("causal_decoder_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("causal_decoder_set_weights_w8a16", torch::CppFunction::makeFallthrough());
  m.impl("causal_decoder_set_weights_nvfp4", torch::CppFunction::makeFallthrough());
  m.impl("causal_decoder_set_adarms_step", torch::CppFunction::makeFallthrough());
  m.impl("causal_decoder_set_adarms_step_mutable", torch::CppFunction::makeFallthrough());
  m.impl("causal_decoder_forward", torch::CppFunction::makeFallthrough());
  m.impl("qwen3_5_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("qwen3_5_forward", torch::CppFunction::makeFallthrough());
  m.impl("qwen3_5_action_forward", torch::CppFunction::makeFallthrough());
  m.impl("qwen3_5_set_action_io_weights", torch::CppFunction::makeFallthrough());
  m.impl("qwen3_5_action_step_forward", torch::CppFunction::makeFallthrough());
  m.impl("qwen3_5_action_rtc_step_forward", torch::CppFunction::makeFallthrough());
  m.impl("qwen3_5_action_rtc_trace_forward", torch::CppFunction::makeFallthrough());
  m.impl("qwen3_5_set_prefill_rope", torch::CppFunction::makeFallthrough());
  m.impl("qwen3_5_spm_z2_pipeline_forward", torch::CppFunction::makeFallthrough());
  m.impl("wall_oss_spm_z1_pipeline_forward", torch::CppFunction::makeFallthrough());
  m.impl("qwen3vl_multiview_spm_z2_canary_forward", torch::CppFunction::makeFallthrough());
  m.impl("lingbot2_multiview_spm_z2_forward", torch::CppFunction::makeFallthrough());
  m.impl("qwen3vl_pooler_spm_z1_pipeline_forward", torch::CppFunction::makeFallthrough());
  m.impl("causal_decoder_set_lm_head", torch::CppFunction::makeFallthrough());
  m.impl("lm_head_logits_top1", torch::CppFunction::makeFallthrough());
  m.impl("argmax_lastdim_host", torch::CppFunction::makeFallthrough());
  m.impl("gemma_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("gemma_forward", torch::CppFunction::makeFallthrough());
  m.impl("gemma_set_prefill_rope", torch::CppFunction::makeFallthrough());
  m.impl("gemma4_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("gemma4_forward", torch::CppFunction::makeFallthrough());
  m.impl("adarms_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("adarms_forward", torch::CppFunction::makeFallthrough());
  m.impl("gr00t_dit_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("gr00t_dit_set_weights_w8a16", torch::CppFunction::makeFallthrough());
  m.impl("gr00t_dit_forward", torch::CppFunction::makeFallthrough());
  m.impl("internvla_nextdit_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("internvla_nextdit_set_fused_qkq_weights",
         torch::CppFunction::makeFallthrough());
  m.impl("internvla_nextdit_set_step_prepare_weights",
         torch::CppFunction::makeFallthrough());
  m.impl("internvla_nextdit_forward", torch::CppFunction::makeFallthrough());
  m.impl("gr00t_vl_encoder_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("gr00t_vl_encoder_set_weights_w8a16", torch::CppFunction::makeFallthrough());
  m.impl("gr00t_vl_encoder_forward", torch::CppFunction::makeFallthrough());
  m.impl("gr00t_vl_encoder_forward_z2", torch::CppFunction::makeFallthrough());
  m.impl("gr00t_denoise_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("gr00t_denoise_step_forward", torch::CppFunction::makeFallthrough());
  m.impl("gr00t_denoise_unroll_forward", torch::CppFunction::makeFallthrough());
  m.impl("gr00t_denoise_unroll_forward_z2", torch::CppFunction::makeFallthrough());
  m.impl("gr00t_spm_z2_prepare_masks", torch::CppFunction::makeFallthrough());
  m.impl("gr00t_spm_z2_pipeline_forward", torch::CppFunction::makeFallthrough());
  m.impl("pi05_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("pi05_forward", torch::CppFunction::makeFallthrough());
  m.impl("pi05_denoise_step_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("pi05_denoise_step_forward",     torch::CppFunction::makeFallthrough());
  m.impl("pi05_denoise_loop_forward",     torch::CppFunction::makeFallthrough());
  m.impl("wall_oss_action_step_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("wall_oss_action_step_forward",     torch::CppFunction::makeFallthrough());
  m.impl("wall_oss_action_denoise_loop_forward", torch::CppFunction::makeFallthrough());
  m.impl("lingbot_v2_moe_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("lingbot_v2_moe_set_packed_weights", torch::CppFunction::makeFallthrough());
  m.impl("lingbot_v2_moe_set_packed_scales", torch::CppFunction::makeFallthrough());
  m.impl("lingbot_v2_moe_set_base_scales", torch::CppFunction::makeFallthrough());
  m.impl("lingbot_v2_moe_set_adarms_step_mutable",
         torch::CppFunction::makeFallthrough());
  m.impl("lingbot_v2_moe_bind_adarms_schedule",
         torch::CppFunction::makeFallthrough());
  m.impl("lingbot_v2_moe_set_denoise_unroll",
         torch::CppFunction::makeFallthrough());
  m.impl("lingbot_v2_moe_denoise_unroll_forward",
         torch::CppFunction::makeFallthrough());
  m.impl("lingbot_v2_moe_forward", torch::CppFunction::makeFallthrough());
  // HALO
  m.impl("halo_action_expert_set_weights",     torch::CppFunction::makeFallthrough());
  m.impl("halo_action_expert_set_moe_weights", torch::CppFunction::makeFallthrough());
  m.impl("halo_action_expert_step_forward",    torch::CppFunction::makeFallthrough());
  m.impl("hyvit2_patch_embed",        torch::CppFunction::makeFallthrough());
  m.impl("hyvit2_patch_embed_multi",  torch::CppFunction::makeFallthrough());
  m.impl("hyvit2_forward",            torch::CppFunction::makeFallthrough());
  m.impl("hyvit2_forward_packed",     torch::CppFunction::makeFallthrough());
  m.impl("hyvit2_forward_multi",      torch::CppFunction::makeFallthrough());
  m.impl("hyvla_merger_pool",         torch::CppFunction::makeFallthrough());
  m.impl("hyvla_merger_combine",      torch::CppFunction::makeFallthrough());
  m.impl("hyvla_merger_fused",        torch::CppFunction::makeFallthrough());
  m.impl("hyvla_vlm_set_weights",     torch::CppFunction::makeFallthrough());
  m.impl("hyvla_vlm_set_weights_w8a16",     torch::CppFunction::makeFallthrough());
  m.impl("hyvla_vlm_set_moe_weights", torch::CppFunction::makeFallthrough());
  m.impl("hyvla_vlm_set_moe_weights_w8a16", torch::CppFunction::makeFallthrough());
  m.impl("hyvla_vlm_step_forward",    torch::CppFunction::makeFallthrough());
  m.impl("hyvla_expert_set_weights",  torch::CppFunction::makeFallthrough());
  m.impl("hyvla_expert_set_weights_w8a16",  torch::CppFunction::makeFallthrough());
  m.impl("hyvla_expert_step_forward", torch::CppFunction::makeFallthrough());
  m.impl("hyvla_expert_set_action_weights", torch::CppFunction::makeFallthrough());
  m.impl("hyvla_expert_unroll_forward",     torch::CppFunction::makeFallthrough());
  m.impl("halo_image_flow_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("halo_image_flow_set_gen_weights", torch::CppFunction::makeFallthrough());
  m.impl("halo_image_flow_step_forward",torch::CppFunction::makeFallthrough());
  m.impl("qwenpi05_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("qwenpi05_forward", torch::CppFunction::makeFallthrough());
  m.impl("qwenpi05_debug_flush_dumps", torch::CppFunction::makeFallthrough());
  m.impl("rhino_vla_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("rhino_vla_forward", torch::CppFunction::makeFallthrough());
  m.impl("rhino_vla_set_denoise_loop_weights", torch::CppFunction::makeFallthrough());
  m.impl("rhino_vla_set_denoise_loop_adarms_tables", torch::CppFunction::makeFallthrough());
  m.impl("rhino_vla_set_denoise_loop_time_proj_table", torch::CppFunction::makeFallthrough());
  m.impl("rhino_vla_denoise_loop_forward", torch::CppFunction::makeFallthrough());
  m.impl("rhino_vla_debug_flush_dumps", torch::CppFunction::makeFallthrough());
  m.impl("navdp_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("navdp_forward", torch::CppFunction::makeFallthrough());
  m.impl("navdp_set_denoise_glue", torch::CppFunction::makeFallthrough());
  m.impl("navdp_denoise_loop_forward", torch::CppFunction::makeFallthrough());
  m.impl("siglip_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("siglip_set_weights_w8a16", torch::CppFunction::makeFallthrough());
  m.impl("siglip_model_set_patch_emb", torch::CppFunction::makeFallthrough());
  m.impl("siglip_patch_embed", torch::CppFunction::makeFallthrough());
  m.impl("siglip_patch_embed_multi", torch::CppFunction::makeFallthrough());
  m.impl("siglip_forward", torch::CppFunction::makeFallthrough());
  m.impl("siglip_forward_packed", torch::CppFunction::makeFallthrough());
  m.impl("siglip_forward_multi", torch::CppFunction::makeFallthrough());
  m.impl("qwen3vl_vision_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("qwen3vl_vision_set_weights_w8a16", torch::CppFunction::makeFallthrough());
  m.impl("qwen3vl_vision_set_rope", torch::CppFunction::makeFallthrough());
  m.impl("qwen3vl_vision_set_merger_weights", torch::CppFunction::makeFallthrough());
  m.impl("qwen3vl_vision_set_deepstack_merger_weights", torch::CppFunction::makeFallthrough());
  m.impl("qwen3vl_vision_forward", torch::CppFunction::makeFallthrough());
  m.impl("qwen3_5_vision_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("qwen3_5_vision_set_rope", torch::CppFunction::makeFallthrough());
  m.impl("qwen3_5_vision_set_patch_embed", torch::CppFunction::makeFallthrough());
  m.impl("qwen3_5_vision_set_merger", torch::CppFunction::makeFallthrough());
  m.impl("qwen3_5_vision_set_temporal", torch::CppFunction::makeFallthrough());
  m.impl("qwen3_5_vision_forward", torch::CppFunction::makeFallthrough());
  m.impl("qwen25vl_vision_set_weights", torch::CppFunction::makeFallthrough());
  m.impl("qwen25vl_vision_set_weights_w8a16", torch::CppFunction::makeFallthrough());
  m.impl("qwen25vl_vision_set_merger_weights", torch::CppFunction::makeFallthrough());
  m.impl("qwen25vl_vision_set_rope", torch::CppFunction::makeFallthrough());
  m.impl("qwen25vl_vision_forward", torch::CppFunction::makeFallthrough());
}


TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  // Register empty / creation ops
  m.impl("empty_strided", TORCH_FN(rpu_empty_strided));
  m.impl("empty.memory_format", TORCH_FN(rpu_empty));
  m.impl("view", at::native::view);
  m.impl("view.Tensor", at::native::view);

  // view operations
  m.impl("expand", TORCH_FN(rpu_expand));
  m.impl("unsqueeze", TORCH_FN(rpu_unsqueeze));

  // creation ops
  m.impl("arange", TORCH_FN(rpu_arange));
  m.impl("arange.start", TORCH_FN(rpu_arange_start));
  m.impl("arange.start_step", TORCH_FN(rpu_arange_start_step));

  // copy / helper
  m.impl("copy_", TORCH_FN(rpu_copy_));
  m.impl("contiguous", TORCH_FN(rpu_contiguous));
  m.impl("_to_copy", TORCH_FN(rpu_to_copy));
  m.impl("_copy_from_and_resize",
         TORCH_FN(
             rpu__copy_from_and_resize)); // internal op used in printing paths
  // add
  m.impl("add.Tensor", TORCH_FN(rpu_add));
  m.impl("add_.Tensor", TORCH_FN(rpu_add_));
  // add_.Scalar - 零拷贝优化
  m.impl("add_.Scalar",
         [](at::Tensor &self, const c10::Scalar &other,
           const c10::Scalar &alpha) -> at::Tensor & {
           auto cpu_self = rpu_to_cpu_zerocopy(self);
           cpu_self.add_(other, alpha);
           rpu_ddr_flush_force(self.data_ptr());
           // 无需 copy_ 回来，共享内存
           return self;
         });

  // sub
  m.impl("sub.Tensor", TORCH_FN(rpu_sub));
  m.impl("sub_.Tensor", TORCH_FN(rpu_sub_));
  m.impl("sub_.Scalar", TORCH_FN(rpu_sub_scalar_));

  // rsub — aten::rsub.Scalar has a CompositeExplicitAutograd decomposition
  // that rewrites `rsub(self, scalar)` into `sub(scalar_0d_cpu_tensor, self)`,
  // which then routes through our rpu_sub with a CPU scalar as operand `a` and
  // fails `binary_op: a not on RPU`. CompositeExplicitAutograd runs before the
  // `m.fallback(custom_cpu_fallback)` wrapper, so we must register rsub
  // explicitly for PrivateUse1.
  m.impl("rsub.Scalar", [](const at::Tensor &self, const c10::Scalar &other,
                           const c10::Scalar &alpha) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto cpu_result = at::rsub(cpu_self, other, alpha);
    return cpu_result.to(self.device());
  });
  m.impl("rsub.Tensor", [](const at::Tensor &self, const at::Tensor &other,
                           const c10::Scalar &alpha) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto cpu_other = (other.device().type() == c10::DeviceType::PrivateUse1)
                     ? rpu_to_cpu_zerocopy(other) : other;
    auto cpu_result = at::rsub(cpu_self, cpu_other, alpha);
    return cpu_result.to(self.device());
  });

  // Eager/non-fused GELU deliberately uses the graph-aware PrivateUse1 CPU
  // fallback.  Register it explicitly because aten::gelu has a Composite
  // default backend kernel that would otherwise decompose before reaching the
  // backend fallback. Fused models select formula-specific SPM TANH/ERF kernels.
  m.impl("gelu",
         torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("tanh", TORCH_FN(rpu_tanh));
  m.impl("isfinite", TORCH_FN(isfinite_fallback_to_cpu));
  m.impl("normal_", TORCH_FN(rpu_normal_));
  m.impl("matmul", TORCH_FN(rpu_matmul));
  m.impl("bmm", rpu_bmm);
  m.impl("softmax.int", rpu_softmax);
  m.impl("slice.Tensor", torch::CppFunction::makeFallthrough());
  m.impl("as_strided", TORCH_FN(at::native::as_strided_tensorimpl));
  // m.impl("_softmax", rpu_softmax);
  // ne - 零拷贝优化
  m.impl("ne.Tensor", [](const Tensor &self, const Tensor &other) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto cpu_other = rpu_to_cpu_zerocopy(other);
    auto result = at::empty(self.sizes(), self.options().dtype(at::kBool));
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::ne_out(cpu_result, cpu_self, cpu_other);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  });
  m.impl("ne.Scalar", [](const at::Tensor &self, const c10::Scalar &other) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto result = at::empty(self.sizes(), self.options().dtype(at::kBool));
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::ne_out(cpu_result, cpu_self, other);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  });
  // div
  m.impl("div.Tensor", TORCH_FN(rpu_div));
  m.impl("div_.Tensor", TORCH_FN(rpu_div_));
  m.impl("div.Scalar", TORCH_FN(rpu_div_scalar));
  // bitwise_and - 零拷贝优化
  m.impl("bitwise_and.Tensor", [](const at::Tensor &a, const at::Tensor &b) {
    auto cpu_a = rpu_to_cpu_zerocopy(a);
    auto cpu_b = rpu_to_cpu_zerocopy(b);
    auto result = at::empty_like(a);
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::bitwise_and_out(cpu_result, cpu_a, cpu_b);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  });
  // masked_select - 零拷贝输入优化（输出形状不定，需复制）
  m.impl("masked_select", [](const at::Tensor &self, const at::Tensor &mask) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto cpu_mask = rpu_to_cpu_zerocopy(mask);
    auto result = at::masked_select(cpu_self, cpu_mask);

    if (result.numel() == 0) {
      return at::empty({0}, self.options().device(self.device()));
    }
    return result.to(self.device());
  });
  // abs - 零拷贝优化
  m.impl("abs", [](const Tensor &self) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto result = at::empty_like(self);
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::abs_out(cpu_result, cpu_self);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  });

  // min - 零拷贝输入优化（输出是标量）
  m.impl("min", [](const Tensor &self) {
    auto cpu = rpu_to_cpu_zerocopy(self);
    auto result = at::min(cpu);
    return result.to(self.device());
  });

  // max - 零拷贝输入优化（输出是标量）
  m.impl("max", [](const Tensor &self) {
    auto cpu = rpu_to_cpu_zerocopy(self);
    auto result = at::max(cpu);
    return result.to(self.device());
  });
  m.impl("transpose.int", rpu_transpose);
  // PyTorch layout, per the ATen schema — NOT the unified op. Sharing one
  // function between the two forced a head_dim==128 layout sniff that read
  // ordinary PyTorch-layout callers as unified and returned silent garbage.
  m.impl("scaled_dot_product_attention",
         TORCH_FN(rpu_scaled_dot_product_attention));
  m.impl("layer_norm", rpu_layernorm);
  m.impl("dropout", rpu_dropout);
  m.impl("broadcast_tensors", rpu_broadcast);
  m.impl("linear", TORCH_FN(rpu_linear));
  m.impl("reshape", TORCH_FN(reshape_fallback));
  m.impl("permute", torch::CppFunction::makeFallthrough());

  // ============ RPU Native & CPU Fallback implementations ============

  // mul - RPU native implementation
  m.impl("mul.Tensor", TORCH_FN(rpu_mul));
  m.impl("mul.Scalar", TORCH_FN(rpu_mul_scalar));
  m.impl("mul_.Tensor", TORCH_FN(rpu_mul_));
  m.impl("mul_.Scalar", TORCH_FN(rpu_mul_scalar_));

  // cat - 原生 memcpy 实现
  m.impl("cat", TORCH_FN(rpu_cat));

  // neg - RPU native implementation (neg(x) = -1 * x)
  m.impl("neg", TORCH_FN(rpu_neg));

  // embedding - 零拷贝输入
  m.impl("embedding", [](const at::Tensor &weight, const at::Tensor &indices,
                         int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
    MANUAL_FALLBACK_LOG("embedding");
    auto cpu_weight = rpu_to_cpu_zerocopy(weight, /*flush=*/false);
    auto cpu_indices = rpu_to_cpu_zerocopy(indices);
    auto cpu_out = at::embedding(cpu_weight, cpu_indices, padding_idx, scale_grad_by_freq, sparse);
    // 输出形状由 indices 决定，需要分配新内存并复制
    return cpu_out.to(weight.device());
  });

  // index_select - 零拷贝输入优化
  m.impl("index_select", [](const at::Tensor &self, int64_t dim, const at::Tensor &index) {
    MANUAL_FALLBACK_LOG("index_select");
    auto cpu_self = rpu_to_cpu_zerocopy(self, /*flush=*/false);
    auto cpu_index = rpu_to_cpu_zerocopy(index);
    auto cpu_out = at::index_select(cpu_self, dim, cpu_index);
    return cpu_out.to(self.device());
  });

  // mean - 零拷贝输入
  m.impl("mean", [](const at::Tensor &self, c10::optional<at::ScalarType> dtype) {
    MANUAL_FALLBACK_LOG("mean");
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto cpu_out = at::mean(cpu_self, dtype);
    // 输出是标量，直接转换设备即可（很小，拷贝开销可忽略）
    return cpu_out.to(self.device());
  });

  m.impl("mean.dim", TORCH_FN(rpu_mean_dim));

  // argmax/argmin - RPU native
  // m.impl("argmax", TORCH_FN(rpu_argmax));
  m.impl("argmin", TORCH_FN(rpu_argmin));

  // pow - RPU native
  m.impl("pow.Tensor_Scalar", TORCH_FN(rpu_pow_tensor_scalar));
  m.impl("pow.Tensor_Tensor", TORCH_FN(rpu_pow_tensor_tensor));
  m.impl("pow.Scalar", TORCH_FN(rpu_pow_scalar_tensor));

  // rsqrt - RPU native
  m.impl("rsqrt", TORCH_FN(rpu_rsqrt));

  // silu - RPU native
  m.impl("silu", TORCH_FN(rpu_silu));

  // sum - 零拷贝输入
  m.impl("sum", [](const at::Tensor &self, c10::optional<at::ScalarType> dtype) {
    MANUAL_FALLBACK_LOG("sum");
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto cpu_out = at::sum(cpu_self, dtype);
    return cpu_out.to(self.device());
  });

  m.impl("sum.dim_IntList", [](const at::Tensor &self, at::OptionalIntArrayRef dim,
                               bool keepdim, c10::optional<at::ScalarType> dtype) {
    MANUAL_FALLBACK_LOG("sum.dim_IntList");
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto cpu_out = at::sum(cpu_self, dim, keepdim, dtype);
    return cpu_out.to(self.device());
  });

  // div_ in-place - 已在前面使用 rpu_div_ 注册

  // fill_ - 零拷贝实现
  m.impl("fill_.Scalar", [](at::Tensor &self, const c10::Scalar &value) -> at::Tensor& {
    MANUAL_FALLBACK_LOG("fill_.Scalar");
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    cpu_self.fill_(value);
    auto storage = self.storage();
    rpu_ddr_flush_force_sized(
        storage.data_ptr().get(), storage.nbytes());
    return self;
  });

  m.impl("fill_.Tensor", [](at::Tensor &self, const at::Tensor &value) -> at::Tensor& {
    MANUAL_FALLBACK_LOG("fill_.Tensor");
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto cpu_value = rpu_to_cpu_zerocopy(value);
    cpu_self.fill_(cpu_value);
    auto storage = self.storage();
    rpu_ddr_flush_force_sized(
        storage.data_ptr().get(), storage.nbytes());
    return self;
  });

  // cos/sin - RPU native
  m.impl("cos", TORCH_FN(rpu_cos));
  m.impl("sin", TORCH_FN(rpu_sin));

  // all - 零拷贝输入
  m.impl("all", [](const at::Tensor &self) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto cpu_out = at::all(cpu_self);
    return cpu_out.to(self.device());
  });

  m.impl("all.dim", [](const at::Tensor &self, int64_t dim, bool keepdim) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto cpu_out = at::all(cpu_self, dim, keepdim);
    return cpu_out.to(self.device());
  });

  // sub - 已在前面使用 rpu_sub 注册

  // sqrt/exp/log - RPU native
  m.impl("sqrt", TORCH_FN(rpu_sqrt));
  m.impl("exp", TORCH_FN(rpu_exp));
  m.impl("log", TORCH_FN(rpu_log));

  // clamp - 零拷贝优化
  m.impl("clamp", [](const at::Tensor &self,
                     const c10::optional<c10::Scalar> &min,
                     const c10::optional<c10::Scalar> &max) {
    MANUAL_FALLBACK_LOG("clamp");
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto result = at::empty_like(self);
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::clamp_out(cpu_result, cpu_self, min, max);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  });

  // where.self - use non-out variant to handle broadcast correctly
  // e.g. where(cond=[B,1,N,N], self=scalar, other=scalar) -> [B,1,N,N]
  m.impl("where.self", [](const at::Tensor &condition, const at::Tensor &self, const at::Tensor &other) {
    MANUAL_FALLBACK_LOG("where.self");
    auto cpu_cond = rpu_to_cpu_zerocopy(condition);
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto cpu_other = rpu_to_cpu_zerocopy(other);
    auto cpu_result = at::where(cpu_cond, cpu_self, cpu_other);
    return cpu_result.to(condition.device());
  });

  // squeeze
  m.impl("squeeze", [](const at::Tensor &self) {
    // squeeze is a view operation, try zero-copy first
    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    std::vector<int64_t> new_sizes, new_strides;
    for (size_t i = 0; i < sizes.size(); ++i) {
      if (sizes[i] != 1) {
        new_sizes.push_back(sizes[i]);
        new_strides.push_back(strides[i]);
      }
    }
    if (new_sizes.empty()) {
      new_sizes.push_back(1);
      new_strides.push_back(1);
    }
    auto storage = self.storage();
    auto new_impl = c10::make_intrusive<at::TensorImpl>(
        c10::Storage(std::move(storage)),
        c10::DispatchKeySet{c10::DispatchKey::PrivateUse1}, self.dtype());
    new_impl->set_storage_offset(self.storage_offset());
    new_impl->set_sizes_and_strides(new_sizes, new_strides);
    return at::Tensor(std::move(new_impl));
  });

  m.impl("squeeze.dim", [](const at::Tensor &self, int64_t dim) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim || self.size(dim) != 1) {
      // No squeeze needed, return view of same tensor
      return self;
    }
    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    sizes.erase(sizes.begin() + dim);
    strides.erase(strides.begin() + dim);

    auto storage = self.storage();
    auto new_impl = c10::make_intrusive<at::TensorImpl>(
        c10::Storage(std::move(storage)),
        c10::DispatchKeySet{c10::DispatchKey::PrivateUse1}, self.dtype());
    new_impl->set_storage_offset(self.storage_offset());
    new_impl->set_sizes_and_strides(sizes, strides);
    return at::Tensor(std::move(new_impl));
  });

  // zeros_like / ones_like
  m.impl("zeros_like", [](const at::Tensor &self,
                          c10::optional<at::ScalarType> dtype,
                          c10::optional<at::Layout> layout,
                          c10::optional<at::Device> device,
                          c10::optional<bool> pin_memory,
                          c10::optional<at::MemoryFormat> memory_format) {
    auto target_device = device.value_or(self.device());
    auto target_dtype = dtype.value_or(self.scalar_type());
    auto result = at::empty(self.sizes(),
                            at::TensorOptions().dtype(target_dtype).device(target_device));
    result.zero_();
    return result;
  });

  m.impl("ones_like", [](const at::Tensor &self,
                         c10::optional<at::ScalarType> dtype,
                         c10::optional<at::Layout> layout,
                         c10::optional<at::Device> device,
                         c10::optional<bool> pin_memory,
                         c10::optional<at::MemoryFormat> memory_format) {
    auto target_device = device.value_or(self.device());
    auto target_dtype = dtype.value_or(self.scalar_type());
    auto result = at::empty(self.sizes(),
                            at::TensorOptions().dtype(target_dtype).device(target_device));
    result.fill_(1);
    return result;
  });

  // zero_ - 零拷贝优化
  m.impl("zero_", [](at::Tensor &self) -> at::Tensor& {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    cpu_self.zero_();
    auto storage = self.storage();
    rpu_ddr_flush_force_sized(
        storage.data_ptr().get(), storage.nbytes());
    return self;
  });

  // gt / lt / ge / le / eq comparisons - 零拷贝优化
  m.impl("gt.Tensor", [](const at::Tensor &self, const at::Tensor &other) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto cpu_other = rpu_to_cpu_zerocopy(other);
    auto result = at::empty(self.sizes(), self.options().dtype(at::kBool));
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::gt_out(cpu_result, cpu_self, cpu_other);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  });
  m.impl("gt.Scalar", [](const at::Tensor &self, const c10::Scalar &other) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto result = at::empty(self.sizes(), self.options().dtype(at::kBool));
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::gt_out(cpu_result, cpu_self, other);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  });
  m.impl("lt.Tensor", [](const at::Tensor &self, const at::Tensor &other) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto cpu_other = rpu_to_cpu_zerocopy(other);
    auto out_shape = at::infer_size(self.sizes(), other.sizes());
    auto result = at::empty(out_shape, self.options().dtype(at::kBool));
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::lt_out(cpu_result, cpu_self, cpu_other);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  });
  m.impl("lt.Scalar", [](const at::Tensor &self, const c10::Scalar &other) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto result = at::empty(self.sizes(), self.options().dtype(at::kBool));
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::lt_out(cpu_result, cpu_self, other);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  });
  // Note: .Tensor comparison ops use infer_size to compute broadcast output shape.
  // e.g. [B,1,N] <= [B,N,1] -> output [B,N,N]. Without this, at::*_out tries to
  // resize a non-resizable RPU storage and crashes.
  m.impl("ge.Tensor", [](const at::Tensor &self, const at::Tensor &other) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto cpu_other = rpu_to_cpu_zerocopy(other);
    auto out_shape = at::infer_size(self.sizes(), other.sizes());
    auto result = at::empty(out_shape, self.options().dtype(at::kBool));
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::ge_out(cpu_result, cpu_self, cpu_other);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  });
  m.impl("ge.Scalar", [](const at::Tensor &self, const c10::Scalar &other) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto result = at::empty(self.sizes(), self.options().dtype(at::kBool));
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::ge_out(cpu_result, cpu_self, other);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  });
  m.impl("le.Tensor", [](const at::Tensor &self, const at::Tensor &other) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto cpu_other = rpu_to_cpu_zerocopy(other);
    auto out_shape = at::infer_size(self.sizes(), other.sizes());
    auto result = at::empty(out_shape, self.options().dtype(at::kBool));
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::le_out(cpu_result, cpu_self, cpu_other);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  });
  m.impl("le.Scalar", [](const at::Tensor &self, const c10::Scalar &other) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto result = at::empty(self.sizes(), self.options().dtype(at::kBool));
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::le_out(cpu_result, cpu_self, other);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  });
  m.impl("eq.Tensor", [](const at::Tensor &self, const at::Tensor &other) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto cpu_other = rpu_to_cpu_zerocopy(other);
    auto out_shape = at::infer_size(self.sizes(), other.sizes());
    auto result = at::empty(out_shape, self.options().dtype(at::kBool));
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::eq_out(cpu_result, cpu_self, cpu_other);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  });
  m.impl("eq.Scalar", [](const at::Tensor &self, const c10::Scalar &other) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto result = at::empty(self.sizes(), self.options().dtype(at::kBool));
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::eq_out(cpu_result, cpu_self, other);
    rpu_ddr_flush_force(result.data_ptr());
    return result;
  });

  // convolution_overrideable - RPU native Conv2d with CPU fallback
  // Uses RPU SPM-based conv kernel when possible (fp16, cin/g%16==0, fits in SPM).
  // Falls back to CPU (fp32 compute) for unsupported cases.
  m.impl("convolution_overrideable", TORCH_FN(rpu_conv2d));
}

// 同 rpu:: namespace block:per-op fallthrough,让 FakeTensor / Dynamo trace
// 路径上 Autograd 不压住 Python key (FakeTensor 子类标准 __torch_dispatch__
// 内置处理 aten op)。真实部署:tensor.requires_grad=False,Autograd 不在
// keyset,直接 PrivateUse1 (block 2648) 走 rpu_linear / rpu_softmax 等
// 真 kernel,行为不变。训练路径:fallthrough → PrivateUse1 真 kernel,
// 不追踪 grad — 跟改前 TORCH_FN(rpu_linear) 一样不追 grad (rpu_linear 内部
// 没构造 backward node)。
//
// 注意:不能给 PrivateUse1 (block 2648) 加 fallthrough,那条是真实部署路径。
// 只对 AutogradPrivateUse1 做 fallthrough,让 FakeTensor 路径越过 Autograd
// 落到 Python (Python key 优先级 > PrivateUse1) 由 FakeTensorMode 接住。
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  m.impl("isfinite", torch::CppFunction::makeFallthrough());
  m.impl("softmax.int", torch::CppFunction::makeFallthrough());
  m.impl("scaled_dot_product_attention",
         torch::CppFunction::makeFallthrough());
  m.impl("layer_norm", torch::CppFunction::makeFallthrough());
  m.impl("dropout", torch::CppFunction::makeFallthrough());
  m.impl("broadcast_tensors", torch::CppFunction::makeFallthrough());
  m.impl("linear", torch::CppFunction::makeFallthrough());
  m.impl("contiguous", torch::CppFunction::makeFallthrough());
  m.impl("reshape", torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
}
