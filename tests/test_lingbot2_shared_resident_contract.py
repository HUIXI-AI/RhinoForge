"""Execute LingBot shared resident COMPLETE plan/consume against generated tiles."""
from pathlib import Path
import re

from test_siglip_attention_dry_live_contract import _function, _run_native

ROOT = Path(__file__).resolve().parents[1]


def test_generated_shared_resident_geometry_and_stale_descriptor_rejection(tmp_path):
    import torch

    model = (ROOT / 'src/fused/rpu_lingbot_v2_moe_model.cpp').read_text()
    generated = (ROOT / 'src/ops/rpu_specialized_linear_contracts.h').read_text()
    launcher = (ROOT / 'src/ops/rpu_linear.cpp').read_text()
    # The generated body is exact. Only SDK enum numeric values are irrelevant
    # to this host execution of the real plan/consume branches.
    ids = sorted(set(re.findall(r'KernelId::(\w+)', generated)))
    generated = generated[generated.index('namespace rpu_specialized_linear {'):]
    helper = _function(model, 'std::vector<int64_t> lingbot_shared_resident_arguments(')
    enum = _function(model, 'enum class LingbotV2LinearRoute') + ';'
    sites = '\n'.join(re.search(rf'constexpr int64_t {name}\s*=\s*[^;]+;', model).group()
                      for name in ('L2_SPARSE_SHARED_GATE_UP_RESIDENT_SILU_W8',
                                   'L2_SPARSE_SHARED_GATE_UP_RESIDENT_SILU_FP16'))
    emit_section = model[model.index('    auto emit_shared = [&]() {'):]
    plan_section = model[model.index('        if (sparse_shared_gate_up_resident_silu_w8_) {',
                                    model.index('LingbotV2MoeExpertModel::physical_manifest_for_candidate(')):]
    flags = ('sparse_shared_gate_up_resident_silu_w8_', 'sparse_shared_gate_up_resident_silu_fp16_')
    emit = ' else '.join(_function(emit_section, f'if ({flag})') for flag in flags)
    plan = ' else '.join(_function(plan_section, f'if ({flag})') for flag in flags)
    for suffix in ('w8', 'fp16'):
        actual = _function(launcher, f'void rpu_launch_lingbot2_shared_gateup_resident_silu_{suffix}_spm_kernel(')
        assert f'KernelId::LINGBOT2_SHARED_GATEUP_RESIDENT_SILU_{suffix.upper()}_T51' in actual
        assert 'queue->enqueu_kernel(*kernel, rpu_specialized_linear::grid(id)' in actual
    program = r'''
#include "core/fused_model_base.h"
#include <cassert>
#include <iostream>
using namespace v3;
constexpr int NUM_CORES=8;
enum class KernelId : uint16_t {__KERNEL_IDS__};
__GENERATED__
__ENUM__
__SITES__
__HELPER__
int launched=0;
void rpu_launch_lingbot2_shared_gateup_resident_silu_w8_spm_kernel(uint32_t x,int w,int s,uint32_t y){
 assert(x==0x1000&&w==11&&s==22&&y==0x2000);++launched;
}
void rpu_launch_lingbot2_shared_gateup_resident_silu_fp16_spm_kernel(uint32_t x,int w,uint32_t y){
 assert(x==0x1000&&w==11&&y==0x2000);++launched;
}
struct Context {
 FmbPhysicalManifestConsumer consumer;
 void consume_physical_route(FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags,
                             const std::vector<int64_t>&arguments){
  consumer.consume_route(f,site,selector,flags,arguments,0);
 }
};
struct Model {
 bool sparse_shared_gate_up_resident_silu_w8_=false,sparse_shared_gate_up_resident_silu_fp16_=false;
 std::vector<int> shared_gate_up_{11},shared_gate_up_scales_{22};
 Context context;Context&ctx(){return context;}
 int64_t intermediate_size()const{return 768;}int64_t hidden_size()const{return 768;}
 uint32_t addr(int,const char*name){return std::string(name)=="residual1"?0x1000:0x2000;}
 FmbPhysicalExecutionManifest plan()const {
  FmbPhysicalExecutionManifest manifest;manifest.state=FmbPhysicalManifestState::COMPLETE;
  manifest.logical_length=manifest.physical_length=51;
  const int64_t logical_len=51;
  auto append=[&](FmbRouteFamily f,int64_t s,int64_t r,int64_t flags,std::vector<int64_t>a){
   manifest.routes.push_back({s,f,r,flags,std::move(a),0});
  };
  __PLAN__
  return manifest;
 }
 void emit(){const int64_t seq_len=51,shared=768,h=768,layer_idx=0;__EMIT__}
};
int main(){
 int rejected=0;
 for(bool w8:{false,true}){
  Model m;m.sparse_shared_gate_up_resident_silu_w8_=w8;m.sparse_shared_gate_up_resident_silu_fp16_=!w8;
  auto plan=m.plan();assert(plan.routes.size()==1);
  const auto id=w8?KernelId::LINGBOT2_SHARED_GATEUP_RESIDENT_SILU_W8_T51:
                   KernelId::LINGBOT2_SHARED_GATEUP_RESIDENT_SILU_FP16_T51;
  const auto c=rpu_specialized_linear::contract(id);const auto grid=rpu_specialized_linear::grid(id);
  const auto args=plan.routes[0].arguments;
  assert(args==std::vector<int64_t>({51,192,768,1,8,w8?1:2,96,c.m,c.n,c.k,
                                   (768+c.k-1)/c.k,grid[0],grid[1],grid[2],0x92,0x3e00}));
  assert(c.m==(w8?64:528)&&c.n==(w8?32:48)&&c.k==128&&
         grid==std::vector<uint16_t>({static_cast<uint16_t>(w8?3:2),1,1}));
  m.context.consumer.reset(plan);launched=0;m.emit();assert(launched==1);
  m.context.consumer.require_all_consumed();
  auto reject=[&](const FmbPhysicalExecutionManifest&bad){
   m.context.consumer.reset(bad);launched=0;bool failed=false;
   try{m.emit();}catch(const c10::Error&error){
    assert(std::string(error.what()).find("EXACT_MISMATCH")!=std::string::npos);failed=true;
   }
   assert(failed&&launched==0);++rejected;
  };
  auto old=plan;old.routes[0].arguments={51,192,768,1,8,w8?1:2,96,w8?32:48,128,6,w8?3:2,0x92,0x3e00};
  reject(old);
  if(w8){
   // The preceding current-ABI W8 schedule had K768 and one retirement.
   // Reject its complete 16-field identity; current K128/six-pass is valid.
   auto prior=plan;prior.routes[0].arguments[9]=768;prior.routes[0].arguments[10]=1;
   reject(prior);
  }else{
   // The preceding current-ABI FP16 schedule was M64/N32/K768/grid3.
   // Its COMPLETE owner must not admit the restored N48/K128 schedule.
   auto prior=plan;prior.routes[0].arguments[7]=64;prior.routes[0].arguments[8]=32;
   prior.routes[0].arguments[9]=768;prior.routes[0].arguments[10]=1;
   prior.routes[0].arguments[11]=3;reject(prior);
  }
  for(size_t i=0;i<args.size();++i){auto changed=plan;++changed.routes[0].arguments[i];reject(changed);}
 }
 assert(rejected==36);
 std::cout<<"PASS: 2 actual plan/consume branches; 2 legacy, 2 prior schedules and 32 field-drift descriptors rejected before launch\n";
}
'''
    for key, value in dict(KERNEL_IDS=','.join(ids), GENERATED=generated, ENUM=enum,
                           SITES=sites, HELPER=helper, PLAN=plan, EMIT=emit).items():
        program = program.replace('__' + key + '__', value)
    output = _run_native(tmp_path, program, torch,
                         extra_sources=[ROOT / 'src/core/fmb_three_stage_chunk_plan.cpp'])
    assert '2 legacy, 2 prior schedules and 32 field-drift' in output
    print(output, end='')
