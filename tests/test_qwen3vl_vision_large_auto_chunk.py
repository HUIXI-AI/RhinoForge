"""Exercise the native cold large-image cap without a device or model load."""

from pathlib import Path
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition


ROOT = Path(__file__).resolve().parents[1]


def test_large_auto_cap_is_cold_bounded_and_preserves_exact_requests(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = (ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp").read_text()
    constants = "\n".join(
        line for line in source.splitlines()
        if line.startswith("constexpr int64_t QWEN3VL_VISION_")
        and any(name in line for name in (
            "DEFAULT_CHUNK_SIZE", "LARGE_IMAGE_THRESHOLD", "LARGE_IMAGE_CHUNK_CAP"))
    )
    methods = "\n".join(
        _definition(source, marker).replace(" override", "")
        for marker in (
            "void set_large_image_auto_chunk(bool enabled)",
            "bool large_image_auto_chunk_profile() const",
            "bool ordinary_fp16_chunked_merger_profile() const",
            "bool fp16_chunked_merger_active() const",
            "void set_fp16_compact_encoder(bool enabled)",
            "bool fp16_compact_encoder_active() const",
            "bool compact_encoder_active() const",
            "int64_t subclass_chunk_size_cap(",
            "void set_configured_chunk_size(int64_t chunk_size)",
        )
    )
    program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
namespace at {
constexpr int kHalf=1, kChar=2, kBFloat16=3;
struct Tensor {int dtype=kHalf;int scalar_type()const{return dtype;}};
}
struct Weights {at::Tensor q_w,k_w,v_w,o_w,fc1_w,fc2_w;};
''' + constants + r'''
struct Vision {
 int64_t cores=8,layers=24,hidden=1024,intermediate=4096,heads=16,dim=64;
 int64_t orig_head_dim_=64,image_batch_count_=1,z1_prepared_num_patches_=0;
 bool linear_acc32_=true,delivery_compatibility_=false,merger=false,prefix_scatter_active_=false;
 bool pipeline=false,native_composite_prepare_active_=false,z1_adopted_=false,z1_bound_=false;
 bool multiview_dry_prepared_=false,multiview_composite_prepared_=false;
 bool large_image_auto_chunk_=false;
 bool fp16_chunked_merger_enabled_=false,fused_merger_enabled_=false;
 bool fp16_compact_encoder_enabled_=false,tap_merger_enabled_=false;
 int64_t current_num_patches_=0,merge_hidden_=4096,merger_out_hidden_=2560;
 at::Tensor merger_m0_w_,merger_m2_w_;
 std::vector<at::Tensor> ds_merger_m0_w_{3},ds_merger_m2_w_{3};
 int64_t configured_chunk_size_=0,last_resolved=0,invalidations=0;
 std::vector<int64_t> deepstack_visual_indexes_{5,11,17};
 std::vector<Weights> layer_weights_=std::vector<Weights>(24);
 int64_t max_keepalive_seq()const{return 32768;}
 int64_t num_cores()const{return cores;}int64_t num_layers()const{return layers;}
 int64_t hidden_size()const{return hidden;}int64_t intermediate_size()const{return intermediate;}
 int64_t num_q_heads()const{return heads;}int64_t head_dim()const{return dim;}
 bool merger_active()const{return merger;}bool pipeline_dispatch_active()const{return pipeline;}
 // W8 owner/scale admission is executed by the shared raw-SPM harness. This
 // legacy-cap facade intentionally exposes only the FP16 owner.
 bool w8_compact_encoder_active()const{return false;}
 int64_t get_last_resolved_chunk_size()const{return last_resolved;}
 void invalidate_model_state(){++invalidations;}
''' + methods + r'''
};
template<class F>void rejected(F f){bool raised=false;try{f();}catch(const std::runtime_error&){raised=true;}assert(raised);}
int main(){
 Vision v;
 rejected([&]{v.subclass_chunk_size_cap(32769,0);});
 rejected([&]{v.set_configured_chunk_size(65536);});
 // The shared Vision/VLA installer has not granted the root-only opt-in.
 assert(v.subclass_chunk_size_cap(1200,0)==144);
 v.set_large_image_auto_chunk(true);
 assert(v.large_image_auto_chunk_&&v.invalidations==1&&v.configured_chunk_size_==0);
 for(int n:{16,256,512,784,1024})assert(v.subclass_chunk_size_cap(n,0)==144);
 for(int n:{1040,1200,1536,4096,8192,16384,32768})assert(v.subclass_chunk_size_cap(n,0)==608);
 // It remains a cap: exact configuration is separate and wins both ways.
 v.set_configured_chunk_size(144);
 assert(v.subclass_chunk_size_cap(1200,0)==144&&v.large_image_auto_chunk_);
 v.set_configured_chunk_size(320);
 assert(v.subclass_chunk_size_cap(1200,0)==320);
 v.set_configured_chunk_size(0);
 assert(v.subclass_chunk_size_cap(1200,0)==608);
 v.set_large_image_auto_chunk(false);
 assert(v.subclass_chunk_size_cap(1200,0)==144);
 v.last_resolved=144;auto count=v.invalidations;
 rejected([&]{v.set_large_image_auto_chunk(true);});
 assert(!v.large_image_auto_chunk_&&v.invalidations==count);
 Vision live;live.set_large_image_auto_chunk(true);live.last_resolved=608;count=live.invalidations;
 rejected([&]{live.set_large_image_auto_chunk(false);});
 assert(live.large_image_auto_chunk_&&live.invalidations==count);
 // Nearby geometry, reduced cores, quantized native weights and private
 // consumers may not inherit the ordinary FP16 cap, even through this setter.
 for(int fault=0;fault<22;++fault){
  Vision x;
  switch(fault){
   case 0:x.cores=4;break;case 1:x.layers=32;break;
   case 2:x.hidden=1152;break;case 3:x.intermediate=4352;break;
   case 4:x.heads=12;break;case 5:x.dim=80;break;case 6:x.orig_head_dim_=72;break;
   case 7:x.linear_acc32_=false;break;case 8:x.delivery_compatibility_=true;break;
   case 9:x.merger=true;break;case 10:x.prefix_scatter_active_=true;break;
   case 11:x.pipeline=true;break;case 12:x.native_composite_prepare_active_=true;break;
   case 13:x.z1_prepared_num_patches_=256;break;case 14:x.z1_adopted_=true;break;
   case 15:x.z1_bound_=true;break;case 16:x.multiview_dry_prepared_=true;break;
   case 17:x.multiview_composite_prepared_=true;break;
   case 18:x.deepstack_visual_indexes_={8,16,24};break;
   case 19:x.layer_weights_.pop_back();break;
   case 20:x.layer_weights_.back().fc2_w.dtype=at::kChar;break;
   case 21:x.image_batch_count_=2;break;
  }
  rejected([&]{x.set_large_image_auto_chunk(true);});
  assert(!x.large_image_auto_chunk_&&x.invalidations==0);
 }
 // Source BF16 AWQ becomes Half during installation; only installed dtype
 // matters. A real BF16 native matrix cannot silently enter the FP16 route.
 Vision bf16;bf16.layer_weights_[0].q_w.dtype=at::kBFloat16;
 rejected([&]{bf16.set_large_image_auto_chunk(true);});
 bf16.layer_weights_[0].q_w.dtype=at::kHalf;bf16.set_large_image_auto_chunk(true);
 assert(bf16.subclass_chunk_size_cap(1200,0)==608);
 // Later private-composite preparation cannot reuse an enabled ordinary cap.
 Vision private_owner;private_owner.set_large_image_auto_chunk(true);
 private_owner.multiview_composite_prepared_=true;
 assert(private_owner.subclass_chunk_size_cap(1200,0)==144);
 // The existing packed/fused and W8 routing rules are unchanged.
 Vision packed;packed.image_batch_count_=2;
 assert(packed.subclass_chunk_size_cap(1200,0)==1200);
 Vision fused;fused.merger=true;fused.linear_acc32_=false;
 assert(fused.subclass_chunk_size_cap(1200,0)==1200);
 // Compact encoder is a separate cold capability. Enabling the merger or
 // large-image cap alone must not silently select the resident C1200 route.
 for(int64_t out:{2048,2560}){
  Vision compact;compact.current_num_patches_=1200;
  compact.merger_out_hidden_=out;compact.fused_merger_enabled_=true;
  compact.fp16_chunked_merger_enabled_=true;
  assert(!compact.fp16_compact_encoder_active());
  assert(compact.subclass_chunk_size_cap(1200,0)==608);
  compact.set_fp16_compact_encoder(true);
  assert(compact.fp16_compact_encoder_enabled_&&compact.invalidations==1);
  assert(compact.fp16_compact_encoder_active());
  assert(compact.subclass_chunk_size_cap(1200,0)==1200);
  for(int exact:{144,256,608}){
   compact.set_configured_chunk_size(exact);
   assert(!compact.fp16_compact_encoder_active());
   assert(compact.subclass_chunk_size_cap(1200,0)==exact);
  }
  compact.set_configured_chunk_size(0);
  for(int n:{256,784,1024,1184,1216,1536}){
   compact.current_num_patches_=n;
   assert(!compact.fp16_compact_encoder_active());
  }
  compact.current_num_patches_=1200;compact.last_resolved=1200;
  auto previous=compact.invalidations;
  rejected([&]{compact.set_fp16_compact_encoder(false);});
  assert(compact.fp16_compact_encoder_enabled_&&compact.invalidations==previous);
 }
 for(int fault=0;fault<8;++fault){
  Vision bad;bad.fused_merger_enabled_=true;bad.fp16_chunked_merger_enabled_=true;
  switch(fault){
   case 0:bad.fp16_chunked_merger_enabled_=false;break;
   case 1:bad.linear_acc32_=false;break;
   case 2:bad.merger_out_hidden_=4096;break;
   case 3:bad.ds_merger_m0_w_[2].dtype=at::kChar;break;
   case 4:bad.layer_weights_.back().fc2_w.dtype=at::kChar;break;
   case 5:bad.cores=4;break;
   case 6:bad.multiview_composite_prepared_=true;break;
   case 7:bad.last_resolved=608;break;
  }
  rejected([&]{bad.set_fp16_compact_encoder(true);});
  assert(!bad.fp16_compact_encoder_enabled_&&bad.invalidations==0);
 }
}
'''
    cpp, binary = tmp_path / "large_cap.cpp", tmp_path / "large_cap"
    cpp.write_text(program)
    built = subprocess.run(
        [compiler, "-std=c++17", "-O0", str(cpp), "-o", str(binary)],
        capture_output=True, text=True,
    )
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(binary)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stdout + ran.stderr
