"""Run actual Vision QKV cold/manifest/callback code on host doubles, not hardware."""
import importlib
import json

from test_g05_vision_qkv_spm_native import ROOT, _compile_qkv, _qkv_native_cpp


def test_qkv_triple_cold_owners_complete_manifest_and_phase2_fallback(tmp_path):
    # Reuse the real transport/layout/manifest fixture; only replace the scenario.
    # This keeps the old-REF transport test and the combined projection test on
    # the same extracted production callback, rather than copying its branches.
    cpp = _qkv_native_cpp().split('int main(){', 1)[0] + r'''
int main(){
 Model m;LayoutContext layout;auto&cache=KernelCache::instance();cache.qkv=true;
 for(int layer=0;layer<24;++layer){auto&w=m.layer_weights_[layer];w.q_w.owner=1000+3*layer;w.k_w.owner=w.q_w.owner+1;w.v_w.owner=w.q_w.owner+2;auto&n=m.layer_bias_norm_[layer];for(auto*t:{&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.fc1_w,&w.fc2_w,&n.ln1_w,&n.ln1_b,&n.ln2_w,&n.ln2_b,&n.q_b,&n.k_b,&n.v_b,&n.o_b,&n.fc1_b,&n.fc2_b})t->track_owner=true;}
 // Immutable proof is installed before patch/RoPE/merger geometry is known.
 m.has_step0_=m.has_rope_=m.has_merger_=false;m.prepare_g05_owner_ln();
 assert(m.g05_qkv_linear_available_&&m.g05_qkv_spm_available_&&!m.is_g05_qkv_spm_profile());
 m.has_step0_=m.has_rope_=m.has_merger_=true;
 const auto hash=m.subclass_layout_hash();const auto identity=m.kvinsert_cost_weight_identity();
 m.plan(layout);const auto&r=m.context.find_physical_route(FmbRouteFamily::LINEAR,G05_VISION_QKV_LINEAR_SITE,0);
 assert(G05_VISION_QKV_LINEAR_SITE==3357339527802085618LL);
 assert(r.selector==1&&r.flags==0&&r.arguments==std::vector<int64_t>({2,256,128,1024,8,32,368,64,128,2,6,2048,256,3,256,1,8,0,1,2,32,40,20,34,42,4,36,44,6,38,46}));
 auto has=[&](const Model&x,int64_t site){return std::any_of(x.context.routes.begin(),x.context.routes.end(),[&](auto&r){return r.site==site;});};
 for(auto s:{QWEN35_VISION_Q_SITE,QWEN35_VISION_K_SITE,QWEN35_VISION_V_SITE})assert(!has(m,s));
 assert(has(m,QWEN35_VISION_Q_ROPE_SITE)&&has(m,QWEN35_VISION_K_ROPE_SITE));
 // Consume the combined descriptor before input DMA or LN; raw outputs still
 // flow to both original RoPE calls. Compute consumes only its transport phase.
 m.context.input_in_spm=false;
 const auto metadata=at::Tensor::metadata_reads;const int queries=cache.queries;
 for(int layer=0;layer<24;++layer){emitted.clear();m.emit_kv_first_body(layer,{});m.compute(layer,{});assert(emitted==std::vector<std::string>({"input","ln","qkv","rope","rope","input"}));}
 assert(at::Tensor::metadata_reads==metadata&&cache.queries==queries);
 for(size_t argument=0;argument<r.arguments.size();++argument){auto x=m;for(auto&route:x.context.routes)if(route.site==G05_VISION_QKV_LINEAR_SITE)++route.arguments[argument];rejects_before_emission([&]{x.emit_kv_first_body(0,{});});}
 for(int field=0;field<6;++field){auto x=m;for(auto&route:x.context.routes)if(route.site==G05_VISION_QKV_LINEAR_SITE){if(field==0)++route.selector;if(field==1)++route.flags;if(field==2)++route.invocation;if(field==3)route.family=FmbRouteFamily::ROPE;if(field==4)route.arguments.pop_back();if(field==5)++route.site;}rejects_before_emission([&]{x.emit_kv_first_body(0,{});});}
 {auto x=m;x.context.routes.erase(std::remove_if(x.context.routes.begin(),x.context.routes.end(),[](auto&r){return r.site==G05_VISION_QKV_LINEAR_SITE;}),x.context.routes.end());rejects_before_emission([&]{x.emit_kv_first_body(0,{});});}
 {auto x=m;x.context.complete=false;rejects_before_emission([&]{x.emit_kv_first_body(0,{});});}
 {auto x=m;x.alias=false;rejects_before_emission([&]{x.emit_kv_first_body(0,{});});}
 {auto x=m;x.context.stage_plan.compute.chunks[0].offset=1;rejects_before_emission([&]{x.emit_kv_first_body(0,{});});}
 for(int field=0;field<4;++field){ChunkInfo chunk;if(field==0)chunk.idx=1;if(field==1)chunk.offset=1;if(field==2)chunk.len=128;if(field==3)chunk.kv_seq_len=512;rejects_before_emission([&]{m.emit_kv_first_body(0,chunk);});}
 // Old REF keeps all three projection owners and the independent raw transport.
 cache.qkv=false;m.prepare_g05_owner_ln();m.plan(layout);
 assert(!m.g05_qkv_linear_available_&&m.g05_qkv_spm_available_&&!has(m,G05_VISION_QKV_LINEAR_SITE));
 assert(m.subclass_layout_hash()!=hash&&m.kvinsert_cost_weight_identity()!=identity);
 for(auto s:{QWEN35_VISION_Q_SITE,QWEN35_VISION_K_SITE,QWEN35_VISION_V_SITE})assert(has(m,s));
 emitted.clear();m.emit_kv_first_body(0,{});m.compute(0,{});assert(emitted==std::vector<std::string>({"input","ln","linear","linear","linear","rope","rope","input"}));
 cache.qkv=true;m.prepare_g05_owner_ln();assert(m.subclass_layout_hash()==hash&&m.kvinsert_cost_weight_identity()==identity);
 // All installed Q/K/V weight and bias owners participate, including layer 23.
 for(int layer=0;layer<24;++layer)for(int tensor=0;tensor<6;++tensor){auto x=m;auto&w=x.layer_weights_[layer];auto&n=x.layer_bias_norm_[layer];at::Tensor*ts[]={&w.q_w,&w.k_w,&w.v_w,&n.q_b,&n.k_b,&n.v_b};ts[tensor]->dtype=2;x.prepare_g05_owner_ln();assert(!x.g05_qkv_linear_available_&&!x.g05_qkv_spm_available_);}
 auto cold_reject=[&](auto change){auto x=m;change(x);x.prepare_g05_owner_ln();assert(!x.g05_qkv_linear_available_&&!x.g05_qkv_spm_available_);};
 cold_reject([](auto&x){x.layer_weights_.back().v_w.present=false;});cold_reject([](auto&x){x.layer_bias_norm_.back().q_b.location=0;});cold_reject([](auto&x){x.layer_weights_.back().k_w.contiguous=false;});cold_reject([](auto&x){x.layer_weights_.pop_back();});cold_reject([](auto&x){x.layer_bias_norm_.pop_back();});cold_reject([](auto&x){x.linear_acc32_=false;});cold_reject([](auto&x){x.has_temporal_=true;});cold_reject([](auto&x){x.cores=4;});
 {auto x=m;x.layer_weights_.back().v_w.owner+=1;assert(x.kvinsert_cost_weight_identity()!=identity);}
 {auto x=m;x.layer_bias_norm_.back().v_b.owner+=1;assert(x.kvinsert_cost_weight_identity()!=identity);}
 // A payload does not widen admission: different geometry and DDR planning
 // retain original projections. Raw -> DDR -> raw is safe on the same owner.
 auto ordinary=[&](auto change){auto x=m;change(x);x.plan(layout);assert(!has(x,G05_VISION_QKV_LINEAR_SITE));for(auto s:{QWEN35_VISION_Q_SITE,QWEN35_VISION_K_SITE,QWEN35_VISION_V_SITE})assert(has(x,s));emitted.clear();x.emit_kv_first_body(0,{});assert(std::count(emitted.begin(),emitted.end(),"linear")==3&&std::count(emitted.begin(),emitted.end(),"qkv")==0);};
 ordinary([](auto&x){x.current_output_patches_=128;});ordinary([](auto&x){x.current_camera_batch_count_=3;});ordinary([](auto&x){x.current_compact_step0_=true;});ordinary([](auto&x){x.z2_bound_=true;});ordinary([](auto&x){x.debug=true;});
 for(auto policy:{AttentionExecutionPolicy::SPM_KV_BY_MHA,AttentionExecutionPolicy::DDR_KV,AttentionExecutionPolicy::SPM_KV_BY_MHA}){
  m.context.attention_policy=layout.attention_policy=policy;m.plan(layout);emitted.clear();m.emit_kv_first_body(23,{});m.compute(23,{});
  if(policy==AttentionExecutionPolicy::SPM_KV_BY_MHA)assert(emitted==std::vector<std::string>({"input","ln","qkv","rope","rope","input"}));
  else assert(emitted==std::vector<std::string>({"input","ln","linear","linear","linear","rope","rope","kv-ddr","q-store","input","q-load"}));
 }
}
'''
    cpp = cpp.replace('assert(q.owner!=k.owner&&q.owner!=v.owner&&k.owner!=v.owner);',
                      'assert(q.owner==1000+3*((qb-0x180000)/4096)&&k.owner==q.owner+1&&v.owner==q.owner+2);')
    _compile_qkv(tmp_path, cpp)
