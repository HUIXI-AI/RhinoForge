"""HyViT2 four-lane packed ViT: real declarations through the real allocator plan.

The board rejected the 3x196 packed ViT at four lanes (4/6 cores, and W4 at
eight) with required=7,217,664 against available=7,043,584 bytes. The live
set fits; first-fit order did not: oproj (phase 4) and the MASK_ONCE mask were
placed after the KvInsert k/v, leaving a hole fc1 and sdpa_tmp could not use.
This compiles the native declare_buffers and plan_temporary_aliased unchanged
and checks the peak the planner admits. It is not a board run.
"""
from pathlib import Path
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition

ROOT = Path(__file__).resolve().parents[1]
BOARD_AVAILABLE = 7_043_584  # frame-2 ViT budget the board reported at four lanes


def fmb_layout_prelude():
    """Host C++ prelude, left open inside ``namespace v3``: the real SDPA tiling,
    allocator plan and FMB declaration types a model's declare_buffers needs."""
    fmb = (ROOT / "src/core/fused_model_base.h").read_text()
    helpers = (ROOT / "src/core/rpu_helpers.h").read_text()
    allocator = (ROOT / "src/core/rpu_spm_allocator.cpp").read_text()
    code = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "core/rpu_spm_allocator.h"
#define TORCH_CHECK(x, ...) do {if(!(x))throw std::runtime_error(#x);}while(0)
#define DWIDTH 2
#define SDPA_MAX_VLM_SIZE 400
template<class A,class B> int64_t CeilDiv(A a,B b){return (a+b-1)/b;}
template<class A,class B> int64_t Align(A a,B b){return CeilDiv(a,b)*b;}
'''
    for marker in ("struct SdpaTiling", "enum class SdpaKernelType", "struct SdpaConfig"):
        code += _definition(helpers, marker) + ";\n"
    for marker in (
        "inline int64_t sdpa_tile_m_v16_initial(", "inline uint32_t GetMhaAttnUnivVlmNeed(",
        "inline std::optional<int64_t> choose_tile_k_aggressive(",
        "inline std::optional<int64_t> choose_tile_k_conservative(",
        "inline std::optional<int64_t> choose_tile_k(", "inline bool sdpa_try_compute_tiling(",
        "inline SdpaTiling sdpa_compute_tiling(",
    ):
        code += _definition(helpers, marker) + "\n"
    code += _definition(allocator, "SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(")
    code += "\nnamespace v3 {class FusedModelBase;\n"
    for marker in ("enum class StorageClass", "enum class BufferScope", "struct BufferDecl",
                   "enum class AttentionExecutionPolicy", "enum class FmbRopeTableResidency",
                   "enum class FmbForwardOperandResidency", "struct LayoutContext"):
        code += _definition(fmb, marker) + ";\n"
    return code


def _program():
    vit = (ROOT / "src/fused/rpu_hyvit2_vision_model.cpp").read_text()
    code = fmb_layout_prelude()
    code += _definition(vit, "struct HYViT2ColdConfig") + ";\n"
    declare = _definition(vit, "std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override")
    code += r'''
struct Probe {
 int64_t cores=4,image_batch_count_=3,seq_len_=588;bool cold_per_image_attention_=false;
 HYViT2ColdConfig cold_config_;bool q_inplace_=false;int64_t kv_pad_rows_=0;
 int64_t hidden_size()const{return 1152;}int64_t num_q_heads()const{return 16;}
 int64_t head_dim()const{return 80;}int64_t intermediate_size()const{return 4352;}
 int64_t num_cores()const{return cores;}int64_t num_layers()const{return 27;}
'''
    code += declare.replace(" override {", " {", 1) + "\n};\n}\n"
    code += r'''
using namespace v3;
static const BufferDecl& slot(const std::vector<BufferDecl>& d,const char* n){
 for(const auto& b:d)if(std::string(b.name)==n)return b;throw std::runtime_error(n);}
static size_t index(const std::vector<BufferDecl>& d,const char* n){
 for(size_t i=0;i<d.size();++i)if(std::string(d[i].name)==n)return i;throw std::runtime_error(n);}
int main(){
 Probe m;m.cold_config_.mask_once=m.cold_config_.q_inplace=m.cold_config_.kvpad16=true;
 LayoutContext lc;lc.chunk_size=592;
 for(int64_t cores:{4,8}){
  m.cores=cores;const auto d=m.declare_buffers(lc);
  std::vector<SpmAllocator::AllocRequest> requests;std::vector<size_t> source;
  for(size_t i=0;i<d.size();++i){
   if(d[i].storage!=StorageClass::Temp||d[i].alias_of)continue;
   requests.push_back({d[i].size,d[i].phase_start,d[i].phase_end,
                       static_cast<int>(d[i].scope)});source.push_back(i);
  }
  const auto plan=SpmAllocator::plan_temporary_aliased(requests);
  if(cores==8){
   // Eight lanes keep their measured declaration order and phases.
   assert(slot(d,"oproj").phase_start==4);
   assert(index(d,"sdpa_mask")>index(d,"sdpa_tmp"));
   continue;
  }
  // Every LayerWide slot sits below every KvInsert/Compute slot, so k/v's
  // space is reused by sdpa_out/sdpa_tmp/fc1 instead of leaving a hole.
  size_t wide_end=0,narrow_begin=SIZE_MAX;
  for(size_t r=0;r<requests.size();++r){
   const auto& b=d[source[r]];const size_t end=plan.offsets[r]+Align(b.size,int64_t{256});
   if(b.scope==BufferScope::LayerWide)wide_end=std::max(wide_end,end);
   else narrow_begin=std::min<size_t>(narrow_begin,plan.offsets[r]);
  }
  assert(wide_end==narrow_begin);
  assert(slot(d,"sdpa_mask").scope==BufferScope::LayerWide&&slot(d,"oproj").phase_start==1);
  std::cout<<"four-lane peak "<<plan.peak_bytes<<"\n";
  assert(plan.peak_bytes==6459904);
 }
}
'''
    return code


def test_four_lane_packed_vit_places_layer_wide_first_and_fits_the_board_budget(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    cpp, binary = tmp_path / "hyvit2_layout.cpp", tmp_path / "hyvit2_layout"
    cpp.write_text(_program())
    result = subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(ROOT / "src"),
                             str(cpp), "-o", str(binary)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    peak = int(result.stdout.split()[-1])
    assert peak < BOARD_AVAILABLE
