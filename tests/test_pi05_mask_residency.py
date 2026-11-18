"""Pi05 mask lifetime and typed schedule contracts; no device initialization.

The compiled test uses the production declarations, SDPA sizing, and allocator.
It does not emulate model math or infer hardware performance from transfer counts.
"""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/fused/rpu_pi05_denoise_step_model.cpp"


def test_mask_lifetime_covers_every_declared_body_hook_scratch():
    source = SOURCE.read_text()
    declare = _definition(source, "std::vector<BufferDecl> Pi05DenoiseStepModel::base_buffer_declarations(")
    # A future phase/scope refactor must explicitly prove cross-body lifetime:
    # OutsideLayerLoop scratch would otherwise alias the all-body mask.
    rows = re.findall(
        r'\{"([^"]+)",\s*([^,]+),\s*(\d+),\s*(\d+),\s*SC::(\w+),\s*0,\s*nullptr,\s*(\w+)',
        declare,
    )
    assert {row[0] for row in rows} >= {
        "sdpa_mask", "x_t_spm", "action_emb_core0_spm", "cond",
        "final_gemv", "final_out_spm", "v_t_core0_spm",
    }
    assert 'constexpr BufferScope ALL = BufferScope::LayerWide;' in declare
    temporary = [row for row in rows if row[4] == "Temp"]
    assert all(row[2:4] == ("0", "0") and row[5] == "ALL" for row in temporary)
    assert len(temporary) == declare.count("SC::Temp,")
    # The NVFP4 scale slab, the split-KV K/V share and the resident Q weights
    # are the temps added outside the base list; all must keep the same
    # all-body lifetime, or a hook clobbers them between bodies.
    wrapper = _definition(source, "std::vector<BufferDecl> Pi05DenoiseStepModel::declare_buffers(")
    added = re.findall(r'decls\.push_back\(\{"([^"]+)",\s*([^,]+),\s*(\d+),\s*(\d+),\s*'
                       r'StorageClass::(\w+),\s*0,\s*nullptr,\s*BufferScope::(\w+)', wrapper)
    assert added == [
        ("nvfp4_spm_scale", "scale_layers * PI05_DENOISE_NVFP4_SPM_SCALE_LAYER", "0", "0", "Temp", "LayerWide"),
        ("skv_share", "layers * bytes", "0", "0", "Temp", "LayerWide"),
        ("q_spm_weight", "layers * PI05_DENOISE_Q_SPM_WEIGHT_LAYER", "0", "0", "Temp", "LayerWide"),
    ]
    assert wrapper.count("push_back(") == 3


def test_mask_schedule_is_admitted_and_consumed_before_attention():
    source = SOURCE.read_text()
    restore = _definition(source, "void Pi05DenoiseStepModel::restore_planned_route_profile(")
    manifest = _definition(source, "Pi05DenoiseStepModel::physical_manifest_for_candidate(")
    body = _definition(source, "void Pi05DenoiseStepModel::build_layer_subgraph(")
    first = _definition(body, "if (ctx().body_iter == 0 && layer_idx == 0 && chunk.idx == 0)")
    assert "PI05_DENOISE_MASK_SCHEDULE_SITE" in restore
    assert "found_mask_schedule" in restore
    assert "mask lifetime/schedule route drift" in restore
    assert "PI05_DENOISE_MASK_SCHEDULE_SITE" in manifest
    assert "planned_loop_mode_ ? 1 : 0, planned_num_steps_, num_layers()" in manifest
    assert "FmbRouteFamily::GRAPH_SCHEDULE" in first
    assert first.index("consume_physical_route(") < first.index("sdpa_dma_mask_to_spm(")
    assert "addr_offset(\"sdpa_mask\").value" in first
    assert body.count("sdpa_dma_mask_to_spm(") == first.count("sdpa_dma_mask_to_spm(") == 1
    # Merging the older optimized pre-body hook must not add a second owner.
    pre_body = _definition(source, "void Pi05DenoiseStepModel::emit_pre_layers_body(")
    assert "sdpa_dma_mask_to_spm(" not in pre_body
    assert body.index(first) < body.index("rpu_launch_sdpa_spm_unified_kernel_v2(")


def test_mask_mutation_invalidates_before_stable_slot_publication():
    source = SOURCE.read_text()
    prepare = _definition(source, "void Pi05DenoiseStepModel::prepare_sdpa_mask_cached(")
    assert "version_enabled\n" in prepare
    assert "prepared_mask_input_ref_.is_same(attention_mask_4d)" in prepare
    assert "prepared_mask_input_version_ == input_version" in prepare
    miss = _definition(prepare, "if (!cache_hit)")
    assert miss.index("prepared_mask_input_ref_ = at::Tensor{};") < miss.index("sdpa_prepare_mask(")
    assert miss.index("prepared_mask_input_version_ = -1;") < miss.index("sdpa_prepare_mask(")
    assert miss.index("current_version()") < miss.index("prepared_mask_ = std::move(")
    for entry in ("step_forward(", "denoise_loop_forward("):
        forward = _definition(source, "void Pi05DenoiseStepModel::" + entry)
        assert forward.index("prepare_sdpa_mask_cached(") < forward.index("run_all_layers(")


def _allocator_program():
    helpers = (ROOT / "src/core/rpu_helpers.h").read_text()
    allocator = (ROOT / "src/core/rpu_spm_allocator.cpp").read_text()
    allocator_header = (ROOT / "src/core/rpu_spm_allocator.h").read_text()
    source = SOURCE.read_text()
    program = r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("native rejected"); } while(0)
#define SDPA_MAX_VLM_SIZE 400
#define RPU_SDPA_TILE_K_CONSERVATIVE_MODE 1
#define DWIDTH 2
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
    program += r"""
class SpmAllocator {public:
  static constexpr size_t ALIGN=256;
  static size_t align_up(size_t n) {return (n+ALIGN-1)&~(ALIGN-1);}
"""
    for marker in ("struct AllocRequest", "struct AliasedPlan"):
        program += _definition(allocator_header, marker) + ";\n"
    program += "static AliasedPlan plan_temporary_aliased(const std::vector<AllocRequest>&);};\n"
    program += _definition(allocator, "SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(")
    program += r"""
enum class StorageClass {Temp,Persistent};
enum class BufferScope {LayerWide};
class FusedModelBase {};
struct BufferDecl {
  std::string name; int64_t bytes; int first_step,last_step; StorageClass storage;
  int num_layers; void* initializer; BufferScope scope;
  std::function<void(FusedModelBase&,int,uint32_t)> preload;
};
struct LayoutContext {int64_t chunk_size,max_kv_seq_len;bool use_attn_mask;};
namespace c10 {using Half=uint16_t;}
struct FakeTensor {template<class T>T* data_ptr()const{return nullptr;} int64_t numel()const{return 10*37*3072;}};
void rpu_launch_ddr_broadcast_spm_dma(const uint16_t*,int64_t,uint32_t,int) {}
struct Nvfp4Tables {
  std::vector<FakeTensor> values;
  const char* names[5]={"nvfp4_q_ts","nvfp4_o_ts","nvfp4_gate_ts","nvfp4_up_ts","nvfp4_down_ts"};
  __NVFP4_DECLARE__
};
constexpr int64_t PI05_DENOISE_ADARMS_BODY_BYTES=37*3072*2;
constexpr int64_t PI05_DENOISE_RESIDENT_W8_BYTES=512*1024;
constexpr int64_t PI05_DENOISE_ROPE_ROW_BYTES=128*2;
struct Pi05SplitKvPlan {
  int64_t qp_bytes()const{return 0;} int64_t part_bytes()const{return 0;}
  int64_t tmp_bytes()const{return 0;} int64_t mask_bytes()const{return 0;}
};
class Pi05DenoiseStepModel {public:
  int cores=8; bool w8=false; int64_t max_action_dim_=32;
  FakeTensor action_in_proj_b_,action_out_proj_b_,adarms_table_;
  Nvfp4Tables nvfp4_;
  int64_t chunk_size_=50,pi05_gateup_resident_layers_=0;
  bool use_spm_adarms_table()const{return cores==8;}
  bool use_pi05_adarms_body_stream()const{return false;}
  bool use_pi05_gateup_resident(int64_t)const{return false;}
  bool use_pi05_gateup_bundle(int64_t)const{return false;}
  bool use_pi05_spm_rope_rows_layout()const{return true;}
  int64_t hidden_size()const{return 1024;}
  int64_t num_q_heads()const{return 8;}
  int64_t num_kv_heads()const{return cores==8?8:4;}
  int64_t head_dim()const{return 256;}
  int attn_tp()const{return cores==8?8:4;}
  int num_cores()const{return cores;}
  int mlp_tp()const{return cores;}
  int64_t intermediate_size()const{return cores==6?(w8?4224:4128):4096;}
  std::optional<Pi05SplitKvPlan> split_kv_plan(int64_t,int64_t)const{return std::nullopt;}
  std::vector<BufferDecl> base_buffer_declarations(const LayoutContext&, bool resident_table = false) const;
};
"""
    program += _definition(source, "std::vector<BufferDecl> Pi05DenoiseStepModel::base_buffer_declarations(")
    program += r"""
int main() {
  Pi05DenoiseStepModel model;
  for (int cores:{4,6,8}) for(bool w8:{false,true}) for(int64_t kv:{288,832,1024}) {
    model.cores=cores;model.w8=w8;
    model.nvfp4_.values.resize(cores==8 && w8?5:0);
    auto decls=model.base_buffer_declarations({50,kv,true});
    std::vector<SpmAllocator::AllocRequest> requests;
    size_t mask=0,persistent=0;
    for(auto& d:decls) {
      if(d.storage==StorageClass::Persistent){persistent+=d.bytes;continue;}
      if(d.name=="sdpa_mask")mask=requests.size();
      requests.push_back({d.bytes,d.first_step,d.last_step,static_cast<int>(d.scope)});
    }
    auto plan=SpmAllocator::plan_temporary_aliased(requests);
    // This is this model's contribution, not all simultaneously live owners.
    assert(plan.peak_bytes+persistent<=8303616);
    size_t begin=plan.offsets[mask],end=begin+requests[mask].bytes;
    std::vector<unsigned char> spm(plan.peak_bytes,0);
    std::fill(spm.begin()+begin,spm.begin()+end,0x5a);
    // Scribble every other declared temp as both hooks/all layers/all steps do.
    // Any alias introduced by a phase/scope change corrupts the saved mask.
    for(int body=0;body<10;++body) for(size_t i=0;i<requests.size();++i) {
      if(i==mask)continue;
      size_t lo=plan.offsets[i],hi=lo+requests[i].bytes;
      assert(hi<=begin || end<=lo);
      std::fill(spm.begin()+lo,spm.begin()+hi,static_cast<unsigned char>(body+1));
    }
    assert(std::all_of(spm.begin()+begin,spm.begin()+end,[](auto x){return x==0x5a;}));
    // Negative control: an outside-hook scope must fail this lifetime proof.
    auto bad=requests;bad[mask].scope=3;
    auto bad_plan=SpmAllocator::plan_temporary_aliased(bad);
    bool alias=false;
    for(size_t i=0;i<bad.size();++i) if(i!=mask) {
      size_t a=bad_plan.offsets[mask],b=bad_plan.offsets[i];
      alias|=a<b+bad[i].bytes && b<a+bad[mask].bytes;
    }
    assert(alias);
    std::cout<<cores<<","<<w8<<","<<kv<<": "<<plan.peak_bytes+persistent<<"\n";
  }
}
"""
    nvfp4 = (ROOT / "src/fused/pi05_nvfp4.h").read_text()
    return program.replace("__NVFP4_DECLARE__", _definition(
        nvfp4, "void declare(std::vector<BufferDecl>& decls, int cores) const"))


def test_native_allocator_preserves_mask_through_body_hook_writes(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    cpp = tmp_path / "pi05_mask_lifetime.cpp"
    binary = tmp_path / "pi05_mask_lifetime"
    cpp.write_text(_allocator_program())
    subprocess.run([compiler, "-std=c++17", "-O0", str(cpp), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert len(result.stdout.splitlines()) == 18
