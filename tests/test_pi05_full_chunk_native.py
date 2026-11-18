"""Regress real Pi full-prefix capacity admission and canonical COMPLETE routes.

The shared fixture runs production declarations, allocator and SDPA helpers on
host state. No RPU callbacks execute. This covers the two integration failures:
unbound C16 capacity probes must be admissible, and full-prefix manifests must
use the common canonical ordering before validation and runtime consumption.
"""
from pathlib import Path

from test_pi05_t128_two_pair_layout import _declaration_fixture_code
from test_siglip_attention_dry_live_contract import _function, _run_native

ROOT = Path(__file__).resolve().parents[1]


def _full_chunk_fixture_code():
    code = _declaration_fixture_code()
    helpers = (ROOT / "src/core/rpu_helpers.h").read_text()
    sdpa = "#define SDPA_MAX_VLM_SIZE 400\n#define RPU_SDPA_TILE_K_CONSERVATIVE_MODE 1\n"
    for marker in (
        "inline int64_t sdpa_tile_m_v16_initial(", "struct SdpaTiling",
        "enum class SdpaKernelType", "struct SdpaConfig",
        "inline uint32_t GetMhaAttnUnivVlmNeed(",
        "inline std::optional<int64_t> choose_tile_k_aggressive(",
        "inline std::optional<int64_t> choose_tile_k_conservative(",
        "inline std::optional<int64_t> choose_tile_k(",
        "inline bool sdpa_try_compute_tiling(", "inline SdpaTiling sdpa_compute_tiling(",
        "inline int64_t sdpa_compute_tmp_v16_size(", "static inline bool sdpa_call_valid(",
        "inline bool sdpa_by_mha_spm_is_valid(",
    ):
        sdpa += _function(helpers, marker) + (";" if marker.startswith(("struct ", "enum ")) else "") + "\n"
    stub = "struct SdpaConfig {};\nint64_t sdpa_compute_tmp_v16_size(SdpaConfig,int64_t){return 0;}"
    assert stub in code
    code = code.replace(stub, sdpa)
    code = code.replace("SdpaConfig make_sdpa_config(int)const{return {};}",
                        "SdpaConfig make_sdpa_config(int mask)const{return "
                        "{SdpaKernelType::FLASH_ATTN_SPM,256,8,8,8,mask};}")

    inc = (ROOT / "src/fused/rpu_gemma_pi05_full_chunk.inc").read_text()
    model = (ROOT / "src/fused/rpu_gemma_model.cpp").read_text()
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    norm = decls[decls.index("enum RpuRmsNormCapability"):
                 decls.index("RpuRmsNormSpmContract rpu_snapshot_rmsnorm_spm_contract(")]
    constants = "\n".join(line for line in model.splitlines() if line.startswith((
        "constexpr int64_t PI05_PREFILL_KV1_DIRECT_ROUTE_FLAG",
        "constexpr int64_t GEMMA_ATTN_RAW_KERNEL_INCOMPATIBLE_DDR_REQUIRED")))
    constants += "\n" + _function(model, "enum GemmaAttentionRouteFlags") + ";"
    methods = "\n".join(_function(inc, marker) + (";" if marker.startswith("enum ") else "")
                        for marker in (
        "enum class Pi05FullSite", "int64_t pi05_full_b(", "int64_t pi05_full_d(",
        "int64_t pi05_full_gate_scratch(", "int64_t pi05_full_root_bytes(",
        "int64_t pi05_full_mask_bytes(", "int64_t pi05_full_tmp_bytes(",
        "std::vector<BufferDecl> pi05_full_chunk_decls(", "KernelId pi05_full_norm_kernel(",
        "bool pi05_full_chunk_domain(", "std::vector<int64_t> pi05_full_chunk_arguments(",
        "FmbRouteManifestEntry pi05_full_route(", "std::vector<FmbRouteManifestEntry> pi05_full_routes(",
        "FmbPhysicalExecutionManifest pi05_full_chunk_manifest(",
        "void pi05_full_chunk_validate_manifest(",
    ))
    methods += "\n" + next(line for line in inc.splitlines()
                             if line.startswith("static constexpr int64_t PI05_FULL_SITE_BASE"))
    methods += "\n" + "\n".join(_function(model, "    " + marker) for marker in (
        "KernelId pi05_prefill_kv1_pair_kernel(", "KernelId pi05_prefill_gateup_pair_kernel(",
        "FmbLinearAccumulationPolicy pi05_prefill_linear_accumulation(",
    ))
    methods += r"""
    bool pi05_full_chunk_owner()const{return pi05_full_chunk_;}
    bool linear_acc32_=false;
    RpuRmsNormSpmContract rmsnorm_spm_contract_{RPU_RMSNORM_CAP_ALL};
    FusedModelBase::ChunkEnvelope envelope{2048,800};
    const FusedModelBase::ChunkEnvelope& chunk_envelope()const{return envelope;}
"""
    stub = "std::vector<BufferDecl> pi05_full_chunk_decls(const LayoutContext&, std::vector<BufferDecl>) { assert(false); return {}; }"
    assert stub in code
    code = code.replace(stub, methods)
    code = code.replace("struct Owner {", norm + "\n" + constants +
                        "\nbool pi05_kernel_opt_in(const char*){return true;}\nstruct Owner {")
    return '#include "ops/rpu_linear_w8a8_contract.h"\n#include "ops/rpu_kvinsert_segment_plan.h"\n' + code


def test_full_prefix_capacity_and_manifest_regressions(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    code = _full_chunk_fixture_code() + r'''

using Range=std::pair<uint64_t,uint64_t>;
void disjoint(std::initializer_list<Range> ranges) {
 for(auto i=ranges.begin();i!=ranges.end();++i) {
  assert(i->first%32==0 && i->second>0);
  for(auto j=ranges.begin();j!=i;++j)
   assert(i->first+i->second<=j->first || j->first+j->second<=i->first);
 }
}
void verify_layouts(){
 for(int64_t c:{272,400}) {
  Owner o;o.pi05_pair_rows_=c;o.pi05_full_chunk_=true;o.pi05_prefill_a8_=true;o.pi05_prefill_a8_panel512_=c==272;
  const int64_t p=2*c,b=c*4096,d=c*512,mask=p*p*2;
  LayoutContext ctx;ctx.chunk_size=ctx.kv_insert_chunk_size=p;ctx.max_kv_seq_len=p;ctx.num_layers=18;ctx.use_attn_mask=true;ctx.is_causal=false;ctx.attention_policy=AttentionExecutionPolicy::DDR_KV;
  int probe_count=0;
  for(int64_t candidate=16;candidate<=p;candidate+=16) {
   LayoutContext probe=ctx;probe.chunk_size=candidate;probe.kv_insert_chunk_size=0;
   for(auto policy:{AttentionExecutionPolicy::DDR_KV,AttentionExecutionPolicy::SPM_KV_BY_MHA}) {
    probe.attention_policy=policy;
    const auto declarations=o.declare_buffers(probe);
    const auto capacity=plan_cpu_temporary_layout(declarations);
    assert(capacity.required_extent>0);++probe_count;
    const bool full=std::any_of(declarations.begin(),declarations.end(),[](const auto& d){return std::string(d.name)=="pi05_full_r1";});
    assert(full==(candidate==p && policy==AttentionExecutionPolicy::DDR_KV));
   }
  }
  for(int variant=0;variant<3;++variant) {
   LayoutContext rejected=ctx;rejected.physical_manifest_fingerprint=123;
   if(variant==0) rejected.chunk_size=16;
   if(variant==1) rejected.attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;
   if(variant==2) rejected.max_kv_seq_len=p+16;
   bool blocked=false;
   try{o.declare_buffers(rejected);}catch(const c10::Error& e){blocked=std::string(e.what()).find("expected_P=")!=std::string::npos;}
   assert(blocked);
  }
  ctx.physical_manifest_fingerprint=123;
  const auto ds=o.declare_buffers(ctx);const auto plan=plan_cpu_temporary_layout(ds);
  int roots=0;int64_t persistent=0;
  for(const auto& x:ds) {
   if(x.storage==StorageClass::Temp) {assert(!x.alias_of && x.scope==BufferScope::LayerWide);++roots;}
   else persistent+=Align(x.size,256)*(x.storage==StorageClass::PersistentPerLayer?x.per_layer:1);
  }
  assert(roots==4 && persistent==151552);
  const uint64_t base=256,r0=base+plan.offsets.at("residual1"),r1=base+plan.offsets.at("pi05_full_r1"),rc0=base+plan.offsets.at("pi05_full_rc0"),rc1=base+plan.offsets.at("pi05_full_rc1");
  const uint64_t root=o.pi05_full_root_bytes(),packed=o.pi05_a8_root_bytes(),scratch=o.pi05_full_gate_scratch(),tmp=o.pi05_full_tmp_bytes();
  disjoint({{r0,root},{r1,root},{rc0,2*d},{rc1,2*d}});
  assert(plan.required_extent==2*root+4*d);
  const uint64_t total=plan.required_extent+656640,budget=SpmAllocator::SPM_PLANNING_BUDGET/256*256;
  assert(total<=budget);
  assert(mask+2*d+tmp<=root && packed+scratch<=root && b+2*d+2*c*64<=root && b+p*2<=root);
  disjoint({{r0,2*b},{r1,b},{rc1,p*2}}); // full Norm -> full Q8/SX
  auto qs=rpu_pl_w8a8::physical_shape(p,2048,2048,rpu_pl_w8a8::Partition::Col,8,false);
  auto qd=rpu_pl_w8a8::make_descriptor(qs,{uint32_t(r1),uint32_t(r1+b),0,uint32_t(rc1)},base,SpmAllocator::SPM_USABLE);
  assert(qd.ordinal==19);
  for(int h=0;h<2;++h) {
   disjoint({{r0+h*b,b},{r1+b+2*d,c*64},{r1+b+2*d+c*64,c*64},{r1+b,2*d}});
   disjoint({{r1+h*b,b},{rc0+h*d,d},{rc1+h*d,d},{r0+h*c*2080,c*2080},{r0+p*2080+h*c*2,c*2+(c==272?32:0)}});
  }
  disjoint({{r1+b,2*d},{r0,mask},{r0+mask,2*d},{r0+mask+2*d,tmp}});
  disjoint({{r0+mask,d},{r0+mask+d,d},{r1,b},{r1+b,b}});
  disjoint({{r0,p*2080},{r0+p*2080,p*2+30},{r0+packed,scratch},{r1,b},{r1+b,b}});
  disjoint({{r1,2*b},{r0,b},{r0+b,p*2}});
  auto down=rpu_pl_w8a8::physical_shape(p,2048,16384,rpu_pl_w8a8::Partition::Row,8,false);
  auto dd=rpu_pl_w8a8::make_descriptor(down,{uint32_t(r0),uint32_t(r1),0,uint32_t(r0+b)},base,SpmAllocator::SPM_USABLE);
  assert(dd.ordinal==27);
  for(int h=0;h<2;++h) disjoint({{r1+h*b,b},{rc1+h*d,d},{r0+h*b,b}});
  bool refused=false;
  try {rpu_pl_w8a8::make_descriptor(qs,{uint32_t(r1),uint32_t(r1+b-32),0,uint32_t(rc1)},base,SpmAllocator::SPM_USABLE);} catch(const std::invalid_argument&) {refused=true;}
  assert(refused);
  std::cout<<"P"<<p<<" roots="<<roots<<" temp="<<plan.required_extent<<" fixed=656640 total="<<total<<" remaining="<<budget-total
    <<" R0="<<r0-base<<" R1="<<r1-base<<" RC0="<<rc0-base<<" RC1="<<rc1-base<<" rootbytes="<<root<<" mask="<<mask<<" attention_out="<<mask<<" tmp_offset="<<mask+2*d<<" tmp_bytes="<<tmp<<" packed="<<packed<<" gate_scratch="<<scratch<<" Q_tile="<<qd.ordinal<<" Down_tile="<<dd.ordinal<<" intervals=PASS negative_overlap=PASS\n";
 }
}

template<class F> void rejects(F fn){bool failed=false;try{fn();}catch(const c10::Error&){failed=true;}assert(failed);}
void verify_manifests(){
 for(int64_t c:{272,400}) {
  Owner o;o.pi05_pair_rows_=c;o.pi05_full_chunk_=true;o.pi05_prefill_a8_=true;o.pi05_prefill_a8_panel512_=c==272;
  const int64_t p=2*c;
  LayoutContext lc;lc.chunk_size=lc.kv_insert_chunk_size=p;lc.max_kv_seq_len=p;lc.num_layers=18;lc.use_attn_mask=true;lc.is_causal=false;lc.attention_policy=AttentionExecutionPolicy::DDR_KV;
  const auto plan=compose_fmb_default_three_stage_chunk_plan(lc,p,0,ChunkMode::KV_FIRST);
  const auto m=o.pi05_full_chunk_manifest(plan,lc,p,p,0);
  rejects([&]{o.pi05_full_chunk_manifest(plan,lc,p,p,16);});
  auto wrong_plan=plan;wrong_plan.compute.plan.chunk_size=c;
  rejects([&]{o.pi05_full_chunk_manifest(wrong_plan,lc,p,p,0);});
  validate_fmb_physical_manifest(m,p,0);o.pi05_full_chunk_validate_manifest(m);
  FmbPrefillStageCandidate candidate{p,p,p,plan,m};
  const auto wire=encode_fmb_prefill_stage_candidate(candidate);
  const auto decoded=decode_fmb_prefill_stage_candidate(wire);
  assert(fmb_physical_manifest_fingerprint(decoded.physical_manifest)==fmb_physical_manifest_fingerprint(m));
  FmbPhysicalManifestConsumer consumer(m);
  rejects([&]{consumer.require_all_consumed();});
  const auto first=o.pi05_full_routes().front();
  rejects([&]{consumer.consume_route(first.family,first.site_id,first.selector+1,
      first.flags,first.arguments,first.invocation);});
  for(const auto& r:o.pi05_full_routes())
   consumer.consume_route(r.family,r.site_id,r.selector,r.flags,r.arguments,r.invocation);
  consumer.require_all_consumed();
  assert(consumer.attention_layout_policy()==AttentionExecutionPolicy::DDR_KV);
  assert(fmb_rope_table_residency(m)==FmbRopeTableResidency::UNSPECIFIED);
  auto bad=m;std::swap(bad.routes.front(),bad.routes.back());rejects([&]{validate_fmb_physical_manifest(bad,p,0);});
  bad=m;bad.routes.insert(bad.routes.begin(),bad.routes.front());rejects([&]{validate_fmb_physical_manifest(bad,p,0);});
  bad=m;for(auto& r:bad.routes)if(r.family==FmbRouteFamily::ATTENTION)r.selector=static_cast<int64_t>(AttentionExecutionPolicy::AUTO);
  rejects([&]{validate_fmb_physical_manifest(bad,p,0);});
  bad=m;for(auto& r:bad.routes)if(r.family==FmbRouteFamily::ROPE)r.flags=FMB_ROUTE_FLAG_ROPE_TABLE_DDR|FMB_ROUTE_FLAG_ROPE_TABLE_SPM;
  rejects([&]{validate_fmb_physical_manifest(bad,p,0);});
  bad=m;bad.routes.front().arguments.push_back(-1);rejects([&]{validate_fmb_physical_manifest(bad,p,0);});
  std::cout<<"P"<<p<<" actual_routes="<<m.routes.size()<<" native_validator=PASS encode_decode=PASS all_consumers=PASS bad_order_duplicate_attention_rope_negative=PASS\n";
 }
}

int main(){verify_layouts();verify_manifests();}
'''
    output = _run_native(tmp_path, code, torch, extra_sources=(
        ROOT / "src/core/fmb_three_stage_chunk_plan.cpp",
        ROOT / "src/ops/rpu_kvinsert_segment_plan.cpp",
    ))
    for prefix in (544, 800):
        assert f"P{prefix} roots=4" in output
        assert f"P{prefix} actual_routes=" in output
    assert "native_validator=PASS" in output
