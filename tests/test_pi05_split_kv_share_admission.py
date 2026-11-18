"""Pi0.5 split-KV K/V share admission, compiled from the production sources.

Layers [0, N) of the split-KV attention keep their K/V blocks in an SPM share
across the denoise bodies. The layout and the launch both ask
split_kv_share_layers(); they must get the same N and per-layer bytes, the
declared share must hold N of them, and the layout with the share must leave
the headroom that the other Pi0.5 owners' SPM takes. The NVFP4 scale slab, when
installed, is planned first and the share takes what it leaves; the resident
Q weights then take what both leave, the same N for the layout and the launch.
A ten-step loop streams its AdaRMS table one body at a time next to the slab,
and without it exactly when the ten-body table keeps the share from holding
every layer (three cameras at H50).
The share launch splits the plan's padded rows over more warps (split_kv_share_tiles)
with the plan's tile_k and rows_pad, so every layout above holds. No device is
involved; the kernel's arithmetic is gated by its strict CModel receipt in
rhino-ops.
"""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src/fused/rpu_pi05_denoise_step_model.cpp"


def _program():
    helpers = (ROOT / "src/core/rpu_helpers.h").read_text()
    allocator = (ROOT / "src/core/rpu_spm_allocator.cpp").read_text()
    allocator_header = (ROOT / "src/core/rpu_spm_allocator.h").read_text()
    base_header = (ROOT / "src/core/fused_model_base.h").read_text()
    decls_header = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    sdpa = (ROOT / "src/ops/rpu_sdpa.cpp").read_text()
    model = MODEL.read_text()
    nvfp4 = (ROOT / "src/fused/pi05_nvfp4.h").read_text()
    program = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("native rejected"); } while(0)
#define SDPA_MAX_VLM_SIZE 400
#define RPU_SDPA_TILE_K_CONSERVATIVE_MODE 1
#define DWIDTH 2
#define V16_NUM 16
#define MAX_CORES 8
template<class T> T CeilDiv(T a,T b) {return (a+b-1)/b;}
template<class T,class U> T Align(T a,U b) {return CeilDiv(a,static_cast<T>(b))*b;}
"""
    for marker in ("struct SdpaTiling", "enum class SdpaKernelType", "struct SdpaConfig"):
        program += _definition(helpers, marker) + ";\n"
    for marker in (
        "inline int64_t sdpa_tile_m_v16_initial(",
        "inline uint32_t GetMhaAttnUnivVlmNeed(",
        "inline std::optional<int64_t> choose_tile_k_aggressive(",
        "inline std::optional<int64_t> choose_tile_k_conservative(",
        "inline std::optional<int64_t> choose_tile_k(",
        "inline bool sdpa_try_compute_tiling(",
        "inline SdpaTiling sdpa_compute_tiling(",
        "inline int64_t sdpa_compute_tmp_v16_size(",
    ):
        program += _definition(helpers, marker) + "\n"
    budget = re.search(r"static constexpr size_t SPM_TOTAL[\s\S]*?SPM_PLANNING_BUDGET =\s*[^;]+;",
                       allocator_header)[0]
    program += "class SpmAllocator {public:\n  static constexpr size_t ALIGN=256;\n"
    program += budget + "\n  static size_t align_up(size_t n) {return (n+ALIGN-1)&~(ALIGN-1);}\n"
    for marker in ("struct AllocRequest", "struct AliasedPlan"):
        program += _definition(allocator_header, marker) + ";\n"
    program += "static AliasedPlan plan_temporary_aliased(const std::vector<AllocRequest>&);};\n"
    program += _definition(allocator, "SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(")
    program += "\nclass FusedModelBase {};\n"
    for marker in ("enum class StorageClass", "enum class BufferScope", "struct BufferDecl"):
        program += _definition(base_header, marker) + ";\n"
    program += "namespace detail {\n"
    program += _definition(
        (ROOT / "src/core/fused_model_base.cpp").read_text(),
        "int64_t detail::estimate_temporary_total(").replace("detail::", "", 1)
    program += "\n}\n"
    program += r"""
struct LayoutContext {int64_t chunk_size=0,max_kv_seq_len=0;bool use_attn_mask=false;};
enum class KernelId {PI05_SPLIT_KV_PACK_Q,PI05_FLASH_ATTN_SPLIT_KV_DP,PI05_SPLIT_KV_MERGE,
                     PI05_FLASH_ATTN_SPLIT_KV_SPM_DP,PI05_Q_KV1_PAIR_DB_WSPM};
struct KernelCache {
  std::array<bool,5> loaded{true,true,true,true,true};
  static KernelCache& instance(){static KernelCache k;return k;}
  bool has_loaded(KernelId id)const{return loaded.at(int(id));}
};
"""
    program += _definition(decls_header, "struct Pi05SplitKvPlan") + ";\n"
    program += "\n".join(re.findall(r"^constexpr (?:int64_t|float) kSplitKv\w+ = [^\n]+", sdpa, re.M))
    program += "\n"
    for marker in (
        "int64_t split_kv_vlm_need(",
        "bool split_kv_fits(",
        "std::pair<int64_t, int64_t> split_kv_share_tiles(",
        "int64_t Pi05SplitKvPlan::qp_bytes(",
        "int64_t Pi05SplitKvPlan::part_bytes(",
        "int64_t Pi05SplitKvPlan::tmp_bytes(",
        "int64_t Pi05SplitKvPlan::mask_bytes(",
        "int64_t Pi05SplitKvPlan::share_bytes(",
        "std::optional<Pi05SplitKvPlan> pi05_split_kv_attention_plan(",
        "bool pi05_split_kv_spm_share(",
        "int64_t split_kv_start(",
        "uint16_t pi05_split_kv_rotated_skip_mask(",
    ):
        program += _definition(sdpa, marker) + "\n"
    program += re.search(r"^constexpr int64_t PI05_DENOISE_SPLIT_KV_SHARE_HEADROOM = [^\n]+", model, re.M)[0]
    program += "\n" + re.search(r"^constexpr int64_t PI05_DENOISE_NVFP4_SPM_SCALE_Q[^;]+;", model, re.M)[0]
    program += "\n" + re.search(r"^constexpr int64_t PI05_DENOISE_Q_SPM_WEIGHT_LAYER = [^\n]+", model, re.M)[0]
    program += r"""
namespace c10 {using Half=uint16_t;}
struct FakeTensor {bool present=false;bool defined()const{return present;}
  template<class T>T* data_ptr()const{return nullptr;} int64_t numel()const{return 10*37*3072;}};
void rpu_launch_ddr_broadcast_spm_dma(const uint16_t*,int64_t,uint32_t,int) {}
struct Nvfp4Tables {
  std::vector<FakeTensor> values;
  const char* names[5]={"nvfp4_q_ts","nvfp4_o_ts","nvfp4_gate_ts","nvfp4_up_ts","nvfp4_down_ts"};
  __NVFP4_DECLARE__
};
constexpr int64_t PI05_DENOISE_ADARMS_BODY_BYTES=37*3072*2;
constexpr int64_t PI05_DENOISE_RESIDENT_W8_BYTES=512*1024;
constexpr int64_t PI05_DENOISE_ROPE_ROW_BYTES=128*2;
class Pi05DenoiseStepModel {public:
  bool w8=false; int64_t max_action_dim_=32, kv_source_heads_=1;
  FakeTensor action_in_proj_b_,action_out_proj_b_,adarms_table_,nvfp4_spm_scale_,q_spm_weight_;
  bool q_fused=true;
  bool pi05_q_in_kv1_pair_owner()const{return q_fused;}
  bool geglu_route=true;
  Nvfp4Tables nvfp4_;
  int64_t chunk_size_=50,pi05_gateup_resident_layers_=0;
  bool planned_route_profile_valid_=true,planned_loop_mode_=true;
  int64_t planned_num_steps_=10;
  uint64_t gen=0;
  uint64_t installed_model_state_generation()const{return gen;}
  mutable std::array<int64_t,5> adarms_table_share_key_{-1,-1,-1,-1,-1};
  mutable bool adarms_table_limits_share_=false;
  bool adarms_table_limits_split_kv_share() const;
  std::set<std::pair<int64_t,int64_t>> split_kv_layouts_;
  bool use_spm_adarms_table()const{return true;}
  // The real decision; force_resident gives the share the resident table would leave.
  bool pi05_adarms_body_stream_opt_in_=false,force_resident=false;
  int64_t planned_prefix_len_=544;
  bool real_body_stream()const;
  bool use_pi05_adarms_body_stream()const{return !force_resident&&real_body_stream();}
  bool use_pi05_gateup_resident(int64_t)const{return false;}
  bool use_pi05_gateup_bundle(int64_t)const{return false;}
  bool use_pi05_spm_rope_rows_layout()const{return true;}
  bool use_pi05_nvfp4_geglu_m50(int64_t rows)const{return geglu_route&&rows==chunk_size_;}
  int64_t hidden_size()const{return 1024;}
  int64_t num_q_heads()const{return 8;}
  int64_t num_kv_heads()const{return 8;}
  int64_t head_dim()const{return 256;}
  int attn_tp()const{return 8;}
  int num_cores()const{return 8;}
  int mlp_tp()const{return 8;}
  int num_layers()const{return 18;}
  int64_t intermediate_size()const{return 4096;}
  std::optional<Pi05SplitKvPlan> split_kv_plan(int64_t,int64_t)const;
  bool split_kv_share_admitted() const;
  std::vector<BufferDecl> capacity_buffer_declarations(int64_t,int64_t,bool=false) const;
  static int64_t spm_room_above(const std::vector<BufferDecl>&);
  bool nvfp4_spm_scale_admitted() const;
  int64_t nvfp4_spm_scale_layers(int64_t,int64_t) const;
  std::pair<int64_t,int64_t> split_kv_share_layers(int64_t,int64_t,bool=false) const;
  bool q_spm_weight_admitted() const;
  int64_t q_spm_weight_layers(int64_t,int64_t) const;
  int64_t q_spm_weight_layers(int64_t,int64_t,int64_t,std::pair<int64_t,int64_t>) const;
  std::vector<BufferDecl> declare_buffers(const LayoutContext&);
  std::vector<BufferDecl> base_buffer_declarations(const LayoutContext&,bool=false) const;
};
"""
    program += _definition(model, "bool Pi05DenoiseStepModel::use_pi05_adarms_body_stream(").replace(
        "use_pi05_adarms_body_stream(", "real_body_stream(", 1) + "\n"
    for marker in (
        "bool Pi05DenoiseStepModel::adarms_table_limits_split_kv_share(",
        "std::optional<Pi05SplitKvPlan> Pi05DenoiseStepModel::split_kv_plan(",
        "bool Pi05DenoiseStepModel::split_kv_share_admitted(",
        "std::vector<BufferDecl> Pi05DenoiseStepModel::capacity_buffer_declarations(",
        "int64_t Pi05DenoiseStepModel::spm_room_above(",
        "bool Pi05DenoiseStepModel::nvfp4_spm_scale_admitted(",
        "int64_t Pi05DenoiseStepModel::nvfp4_spm_scale_layers(",
        "std::pair<int64_t, int64_t> Pi05DenoiseStepModel::split_kv_share_layers(",
        "bool Pi05DenoiseStepModel::q_spm_weight_admitted(",
        "int64_t Pi05DenoiseStepModel::q_spm_weight_layers(int64_t max_kv_seq_len, int64_t chunk)",
        "int64_t Pi05DenoiseStepModel::q_spm_weight_layers(int64_t max_kv_seq_len, int64_t chunk,",
        "std::vector<BufferDecl> Pi05DenoiseStepModel::declare_buffers(",
        "std::vector<BufferDecl> Pi05DenoiseStepModel::base_buffer_declarations(",
    ):
        program += _definition(model, marker) + "\n"
    program += r"""
const BufferDecl* find(const std::vector<BufferDecl>& decls,const char* name){
  for(auto& d:decls) if(std::strcmp(d.name,name)==0) return &d;
  return nullptr;
}
int64_t fixed(const std::vector<BufferDecl>& decls){
  int64_t total=0;
  for(auto& d:decls){
    if(d.alias_of||d.storage==StorageClass::Temp) continue;
    total+=Align(d.size,(int64_t)256)*(d.storage==StorageClass::Persistent?1:d.per_layer);
  }
  return total;
}
int main(){
  auto& cache=KernelCache::instance();
  Pi05DenoiseStepModel model;
  for(bool w8:{false,true}) for(bool scale:{false,true}) for(int64_t rows:{10,50}) for(int64_t prefix:{544,800,928,960}) {
    if(scale&&!w8) continue;
    // A new installation (set_weights) advances the generation.
    ++model.gen;
    model.w8=w8;model.nvfp4_.values.resize(w8?5:0);model.chunk_size_=rows;
    // set_weights packs the resident Q source whenever it installs the slab.
    model.nvfp4_spm_scale_.present=model.q_spm_weight_.present=scale;
    model.planned_prefix_len_=prefix;
    const int64_t chunk=CeilDiv(rows,(int64_t)16)*16,kv=prefix+rows;
    const auto decls=model.declare_buffers({chunk,kv,true});
    const auto [layers,bytes]=model.split_kv_share_layers(kv,chunk);
    const BufferDecl* share=find(decls,"skv_share");
    assert(find(decls,"skv_q")&&layers>0&&share);
    assert(share->size==layers*bytes&&share->storage==StorageClass::Temp&&!share->alias_of&&
           share->scope==BufferScope::LayerWide&&share->phase_start==0&&share->phase_end==0);
    // Installed, every layer's scales fit in every profile; the launch asks the same.
    const int64_t scale_layers=model.nvfp4_spm_scale_layers(kv,chunk);
    const BufferDecl* slab=find(decls,"nvfp4_spm_scale");
    assert(scale_layers==(scale?model.num_layers():0)&&!!slab==scale);
    if(scale) assert(slab->size==scale_layers*PI05_DENOISE_NVFP4_SPM_SCALE_LAYER&&
                     slab->storage==StorageClass::Temp&&!slab->alias_of&&
                     slab->scope==BufferScope::LayerWide&&slab->phase_start==0&&slab->phase_end==0);
    for(int64_t seq=1;seq<=chunk;++seq) assert(model.nvfp4_spm_scale_layers(kv,seq)==scale_layers);
    // Every row count the launch may run gets the same N and bytes, and its plan fits a layer's share.
    for(int64_t seq=1;seq<=chunk;++seq){
      assert(model.split_kv_share_layers(kv,seq)==std::make_pair(layers,bytes));
      const auto p=model.split_kv_plan(seq,kv);
      assert(p->share_bytes()<=bytes);
      // The share launch only regroups the plan's padded rows into more, shorter row tiles: the
      // same tile_k, and rows_pad (the mask, packed rows and partials) covered exactly.
      const auto [m,gx]=split_kv_share_tiles(*p);
      assert(1<=m&&m<=p->tile_m_v16&&gx>=p->grid_x&&gx*16*m==p->rows_pad&&
             split_kv_fits(m,p->tile_k_v16)&&(gx<=8||m==p->tile_m_v16));
    }
    // The layout with the share leaves the headroom for the other owners' SPM.
    assert(fixed(decls)+detail::estimate_temporary_total(decls)<=
           (int64_t)SpmAllocator::SPM_PLANNING_BUDGET-PI05_DENOISE_SPLIT_KV_SHARE_HEADROOM);
    // One more layer would not fit, unless every layer already has its share.
    if(layers<model.num_layers()){
      auto more=decls;
      for(auto& d:more) if(std::strcmp(d.name,"skv_share")==0) d.size+=bytes;
      assert(fixed(more)+detail::estimate_temporary_total(more)>
             (int64_t)SpmAllocator::SPM_PLANNING_BUDGET-PI05_DENOISE_SPLIT_KV_SHARE_HEADROOM);
    }
    // The resident Q weights take what the slab and the share leave: the same N for every row
    // count, at most the slab's layers, and no more layers up to the slab's would fit.
    const int64_t q_layers=model.q_spm_weight_layers(kv,chunk);
    const BufferDecl* qrun=find(decls,"q_spm_weight");
    assert(!!qrun==(q_layers>0)&&q_layers<=scale_layers);
    if(qrun) assert(qrun->size==q_layers*PI05_DENOISE_Q_SPM_WEIGHT_LAYER&&
                    qrun->storage==StorageClass::Temp&&!qrun->alias_of&&
                    qrun->scope==BufferScope::LayerWide&&qrun->phase_start==0&&qrun->phase_end==0);
    for(int64_t seq=1;seq<=chunk;++seq) assert(model.q_spm_weight_layers(kv,seq)==q_layers);
    for(int64_t n=q_layers+1;scale&&n<=scale_layers;++n){
      auto more=decls;
      if(qrun) for(auto& d:more){if(std::strcmp(d.name,"q_spm_weight")==0) d.size=n*PI05_DENOISE_Q_SPM_WEIGHT_LAYER;}
      else more.push_back({"q_spm_weight",n*PI05_DENOISE_Q_SPM_WEIGHT_LAYER,0,0,StorageClass::Temp,0,nullptr,
                           BufferScope::LayerWide});
      assert(fixed(more)+detail::estimate_temporary_total(more)>
             (int64_t)SpmAllocator::SPM_PLANNING_BUDGET-PI05_DENOISE_SPLIT_KV_SHARE_HEADROOM);
    }
    // The slot-rotated writer stores slot h of each block b it writes unless
    // bit (h - b) mod 8 of the mask is set; reader core c reads block b of its
    // member's range [S, E) from slot (c + b - S) mod 8. Exactly the read slots
    // are stored, and the four-core groups leave slots out.
    const auto plan=model.split_kv_plan(rows,kv);
    const int64_t first=prefix/16,end=(prefix+chunk)/16;
    const uint16_t mask=pi05_split_kv_rotated_skip_mask(*plan,first,end);
    for(int64_t b=first;b<end;++b) for(int h=0;h<8;++h){
      bool read=false;
      for(int c=0;c<8;++c){
        const int64_t k=c%plan->group;
        const int64_t start=split_kv_start(*plan,k);
        read|=start<=b&&b<split_kv_start(*plan,k+1)&&(c+b-start)%8==h;
      }
      assert(read==!((mask>>((h-b)&7))&1));
    }
    assert(plan->group!=4||mask!=0);
    // A streaming loop keeps one body of its AdaRMS table, and the share gets the other nine
    // bodies' rows: always next to the scale slab, otherwise exactly when the ten-body table
    // keeps the share from holding every layer.
    const bool stream=model.use_pi05_adarms_body_stream();
    assert(find(decls,"adarms_table")->size==(stream?37*3072*2:10*37*3072*2));
    model.force_resident=true;
    const int64_t resident=model.split_kv_share_layers(kv,chunk).first;
    assert(model.nvfp4_spm_scale_layers(kv,chunk)==scale_layers);
    model.force_resident=false;
    assert(stream==(scale||resident<model.num_layers()));
    assert(stream?layers>resident:layers==resident);
    std::cout<<w8<<","<<scale<<","<<rows<<","<<prefix<<": "<<layers<<"x"<<bytes<<" scale "<<scale_layers
             <<" resident "<<resident<<" stream "<<stream<<" q "<<q_layers<<"\n";
  }
  // The measured shapes: H10 P544 one warp -> three, H50 P544 three -> five.
  for(auto [t,kv,m0,g0,m1,g1]:{std::array<int64_t,6>{10,554,3,1,1,3},std::array<int64_t,6>{50,594,5,3,3,5}}){
    const auto p=model.split_kv_plan(t,kv);
    assert(p->tile_m_v16==m0&&p->grid_x==g0&&split_kv_share_tiles(*p)==std::make_pair(m1,g1));
  }
  // Resident Q also needs the fused pair owner, its payload and the packed source.
  model.w8=true;model.nvfp4_.values.resize(5);model.chunk_size_=10;model.planned_prefix_len_=544;
  model.nvfp4_spm_scale_.present=model.q_spm_weight_.present=true;
  assert(model.q_spm_weight_layers(554,16)>0);
  auto no_q=[&]{
    assert(model.q_spm_weight_layers(554,16)==0&&!find(model.declare_buffers({16,554,true}),"q_spm_weight"));
  };
  cache.loaded[4]=false;no_q();cache.loaded[4]=true;
  model.q_fused=false;no_q();model.q_fused=true;
  model.q_spm_weight_.present=false;no_q();model.q_spm_weight_.present=true;
  model.nvfp4_spm_scale_.present=false;no_q();
  // Not admitted: no share is declared and the launch reads DDR.
  auto none=[&](const char* why){
    model.chunk_size_=10;
    const auto decls=model.declare_buffers({16,554,true});
    assert(find(decls,"skv_q")&&!find(decls,"skv_share"));
    assert(model.split_kv_share_layers(554,10)==std::make_pair(int64_t(0),int64_t(0)));
    std::cout<<"none: "<<why<<"\n";
  };
  model.w8=false;model.nvfp4_.values.clear();model.nvfp4_spm_scale_.present=false;
  cache.loaded[3]=false;none("payload absent");cache.loaded[3]=true;
  // The scale slab shares the loop gates and also needs the fused GeGLU route.
  model.nvfp4_spm_scale_.present=true;
  auto no_slab=[&]{
    assert(model.nvfp4_spm_scale_layers(554,10)==0&&!find(model.declare_buffers({16,554,true}),"nvfp4_spm_scale"));
  };
  model.chunk_size_=10;
  model.geglu_route=false;no_slab();model.geglu_route=true;
  model.planned_loop_mode_=false;no_slab();model.planned_loop_mode_=true;
  model.planned_num_steps_=1;no_slab();model.planned_num_steps_=10;
  model.planned_route_profile_valid_=false;no_slab();model.planned_route_profile_valid_=true;
  model.nvfp4_spm_scale_.present=false;
  model.planned_loop_mode_=false;none("per-step graphs");model.planned_loop_mode_=true;
  model.planned_num_steps_=1;none("one step");model.planned_num_steps_=10;
  model.planned_route_profile_valid_=false;none("no planned profile");
}
"""
    return program.replace("__NVFP4_DECLARE__", _definition(
        nvfp4, "void declare(std::vector<BufferDecl>& decls, int cores) const"))


def test_split_kv_share_layers_agree_between_layout_and_launch_and_fit(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    cpp = tmp_path / "pi05_split_kv_share.cpp"
    binary = tmp_path / "pi05_split_kv_share"
    cpp.write_text(_program())
    build = subprocess.run([compiler, "-std=c++17", "-O0", str(cpp), "-o", str(binary)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr[-4000:]
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 28 and sum(line.startswith("none: ") for line in lines) == 4
