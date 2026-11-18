"""Native N1200 attention admission and unchanged workspace, without hardware."""
from pathlib import Path
import shutil
import subprocess

import pytest
from test_fmb_standalone_layout_rebuild import _definition

ROOT = Path(__file__).resolve().parents[1]


def test_n1200_tm160_native_admission_and_workspace(tmp_path):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    helpers = (ROOT/'src/core/rpu_helpers.h').read_text()
    header = (ROOT/'src/ops/rpu_qwen3vl_vision_sdpa.h').read_text()
    vision = (ROOT/'src/fused/rpu_qwen3vl_vision_model.cpp').read_text()
    definitions = '\n'.join(_definition(helpers, marker) for marker in (
        'struct SdpaTiling {', 'inline uint32_t GetMhaAttnUnivVlmNeed(',
        'inline std::optional<int64_t> choose_tile_k_conservative('))
    definitions = definitions.replace('};\ninline', '};\ninline')
    # _definition ends at the closing brace; the aggregate needs its semicolon.
    definitions = definitions.replace('tile_m, tile_n, tile_k;\n}', 'tile_m, tile_n, tile_k;\n};')
    methods = '\n'.join(_definition(vision, marker) for marker in (
        'bool ordinary_n1200_tm160_profile() const',
        'bool n1200_tm160_eligible(', 'void set_n1200_tm160(bool enabled)'))
    program = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(c,...) do{if(!(c))throw std::runtime_error("rejected");}while(0)
#define SDPA_MAX_VLM_SIZE 400
template<class T> T CeilDiv(T x,T y){return (x+y-1)/y;}
''' + definitions + '\n' + header.replace('#pragma once','').replace('#include "rpu_helpers.h"','') + r'''
enum class ChunkMode{KV_FIRST,SEQUENTIAL};
struct ChunkInfo{int idx;int64_t offset,len;};
struct Stage{struct{int64_t chunk_size=608;}plan;std::vector<ChunkInfo>chunks{{0,0,608},{1,608,592}};};
struct FmbThreeStageChunkPlan{ChunkMode chunk_mode=ChunkMode::KV_FIRST;Stage compute;};
struct Owner{
 bool ordinary=true,delivery_compatibility_=false,prefix_scatter_active_=false;
 bool native_composite_prepare_active_=false,z1_adopted_=false,z1_bound_=false;
 bool multiview_dry_prepared_=false,multiview_composite_prepared_=false,n1200_tm160_enabled_=false;
 int64_t cores=8,image_batch_count_=1,z1_prepared_num_patches_=0,resolved=0,invalidations=0;
 std::vector<int64_t>deepstack_visual_indexes_{5,11,17};
 int64_t num_cores()const{return cores;}
 bool ordinary_w8_resident_profile()const{return ordinary;}
 int64_t get_last_resolved_chunk_size()const{return resolved;}
 void invalidate_model_state(){++invalidations;}
''' + methods + r'''
};
template<class F>void rejects(F f){bool caught=false;try{f();}catch(const std::runtime_error&){caught=true;}assert(caught);}
int main(){
 for(int64_t q:{608,592}){
  auto t=rpu_qwen3vl_n1200_tm160_tiling(q,1200,327680);assert(t&&t->tile_m==160&&t->tile_n==64&&t->tile_k==256);
  assert(GetMhaAttnUnivVlmNeed(10,4,16,4,4,4,0)==312);
  assert(rpu_qwen3vl_n1200_tm160_tiling(q,1200,262144));
  assert(!rpu_qwen3vl_n1200_tm160_tiling(q,1200,262143));
 }
 for(int64_t q:{0,1,576,600,624,1200})assert(!rpu_qwen3vl_n1200_tm160_tiling(q,1200,327680));
 assert(!rpu_qwen3vl_n1200_tm160_tiling(608,1216,327680));
 Owner o;FmbThreeStageChunkPlan p;
 assert(!o.n1200_tm160_eligible(p,1200,1200,0));o.set_n1200_tm160(true);
 assert(o.invalidations==1&&o.n1200_tm160_eligible(p,1200,1200,0));
 for(int fault=0;fault<10;++fault){auto bad=p;int64_t n=1200,l=1200,pos=0;
  switch(fault){case 0:n=1216;break;case 1:l=1199;break;case 2:pos=16;break;
   case 3:bad.chunk_mode=ChunkMode::SEQUENTIAL;break;case 4:bad.compute.plan.chunk_size=592;break;
   case 5:bad.compute.chunks.pop_back();break;case 6:bad.compute.chunks[0].offset=16;break;
   case 7:bad.compute.chunks[1].len=608;break;case 8:bad.compute.chunks[1].offset=592;break;
   case 9:bad.compute.chunks[1].idx=0;break;}
  assert(!o.n1200_tm160_eligible(bad,n,l,pos));
 }
 for(int fault=0;fault<11;++fault){Owner b;
  switch(fault){case 0:b.ordinary=false;break;case 1:b.image_batch_count_=2;break;
   case 2:b.delivery_compatibility_=true;break;case 3:b.prefix_scatter_active_=true;break;
   case 4:b.native_composite_prepare_active_=true;break;case 5:b.z1_prepared_num_patches_=1200;break;
   case 6:b.z1_adopted_=true;break;case 7:b.z1_bound_=true;break;
   case 8:b.multiview_dry_prepared_=true;break;case 9:b.multiview_composite_prepared_=true;break;
   case 10:b.deepstack_visual_indexes_={5,11,23};break;}
  rejects([&]{b.set_n1200_tm160(true);});assert(!b.n1200_tm160_enabled_&&b.invalidations==0);
 }
 o.resolved=608;rejects([&]{o.set_n1200_tm160(false);});assert(o.n1200_tm160_enabled_&&o.invalidations==1);
}
'''
    cpp, binary = tmp_path/'probe.cpp', tmp_path/'probe'
    cpp.write_text(program)
    result=subprocess.run([compiler,'-std=c++17',str(cpp),'-o',str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    result=subprocess.run([str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_n1200_tm160_is_owned_by_complete_descriptor():
    source=(ROOT/'src/fused/rpu_qwen3vl_vision_model.cpp').read_text()
    manifest=_definition(source,'FmbPhysicalExecutionManifest make_physical_manifest_for_candidate(')
    assert '!layout.use_attn_mask' in manifest
    assert 'QWEN3VL_N1200_TM160_ATTN_SITE' in manifest
    dispatch=_definition(source,'void build_layer_subgraph(')
    assert 'route.site_id == QWEN3VL_N1200_TM160_ATTN_SITE' in dispatch
    assert 'n1200_tm160_arguments(chunk), chunk.idx' in dispatch
    assert dispatch.index('ctx().consume_physical_route(\n                    FmbRouteFamily::ATTENTION, QWEN3VL_N1200_TM160_ATTN_SITE') < dispatch.index('rpu_launch_sdpa_qwen3vl_n1200_tm160(')
    identity=_definition(source,'std::vector<int64_t> kvinsert_cost_weight_identity() const')
    assert 'n1200_tm160_enabled_' in identity
