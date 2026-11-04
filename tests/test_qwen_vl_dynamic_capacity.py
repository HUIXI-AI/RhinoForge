"""Host checks for storage-sized position owners and actual execution limits."""
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

from test_qwen3vl_prefill_gateup_launcher import definition


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("capacity", [8193, 16384, 32768])
def test_deepstack_zero_owner_uses_installed_capacity(monkeypatch, capacity):
    from rpu_backend.adapters.qwen3_vl import text

    zeros = torch.zeros

    def host_zeros(shape, **kwargs):
        assert kwargs.pop("device") == "rpu"
        return zeros(shape, **kwargs)

    monkeypatch.setattr(text.torch, "zeros", host_zeros)
    model = SimpleNamespace(
        _rpu_mrope_capacity=capacity,
        config=SimpleNamespace(hidden_size=8, max_position_embeddings=4096),
    )
    owner = text.get_or_create_zero_keepalive(model)
    assert owner.shape == (capacity, 8)
    assert torch.count_nonzero(owner) == 0
    views = text.make_zero_visual_embeds(model, capacity)
    assert len(views) == 3
    assert all(view.data_ptr() == owner.data_ptr() for view in views)
    assert text.get_or_create_zero_keepalive(model) is owner
    assert text.make_zero_visual_embeds(model, capacity)[0] is views[0]
    with pytest.raises(ValueError, match="exceeds"):
        text.make_zero_visual_embeds(model, capacity + 1)
    with pytest.raises(ValueError, match="exceeds installed"):
        text.get_or_create_zero_keepalive(model, max_seq_len=capacity + 1)
    assert model._rpu_deepstack_zero_keepalive is owner


def test_native_vision_capacity_binding_and_decoder_register_domain(tmp_path):
    """Compile real setters/allocation blocks; only Tensor and base are doubled."""
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required")
    q3 = (ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp").read_text()
    q25 = (ROOT / "src/fused/rpu_qwen25vl_vision_model.cpp").read_text()
    dino = (ROOT / "src/fused/rpu_dinov3_vision_model.cpp").read_text()
    decoder = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    q3_alloc = definition(q3[q3.index("// Capacity is cold-bound"):],
                          "if (!position_idx_keepalive_.defined()")
    q25_alloc = definition(q25[q25.index("void set_rope_tables("):],
                           "if (!position_idx_keepalive_.defined()")
    sdpa_start = decoder.index("        TORCH_CHECK(seq_len_in <= int64_t{UINT16_MAX}")
    sdpa_guard = decoder[sdpa_start:decoder.index(";", sdpa_start) + 1]
    program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while(false)
#define TORCH_INTERNAL_ASSERT(condition) TORCH_CHECK(condition)
int64_t Align(int64_t value,int64_t alignment){return ((value+alignment-1)/alignment)*alignment;}
namespace at {
constexpr int kShort=1,kPrivateUse1=2;
struct TensorOptions {TensorOptions& dtype(int){return *this;}TensorOptions& device(int){return *this;}};
struct Tensor {
 int64_t rows=0;std::shared_ptr<std::vector<int16_t>> owner;
 bool defined()const{return bool(owner);}int64_t size(int)const{return rows;}
 void zero_(){std::fill(owner->begin(),owner->end(),0);}
};
Tensor zeros(std::vector<int64_t> shape,TensorOptions){return {shape[0],std::make_shared<std::vector<int16_t>>(shape[0]*shape[1])};}
Tensor empty(std::vector<int64_t> shape,TensorOptions opts){return zeros(shape,opts);}
}
struct Base {
 int64_t capacity_=0,chunk_=0;bool planned=false;
 void set_chunk_envelope(int64_t capacity,int64_t chunk){
  TORCH_CHECK(capacity>0 && !planned && (chunk==0 || (chunk>=16 && chunk%16==0)));
  capacity_=capacity;chunk_=chunk;
 }
};
struct Qwen3:Base {
 at::Tensor position_idx_keepalive_;
 __Q3_SETTER__
 void install_positions(){const int64_t capacity=capacity_;__Q3_ALLOC__}
};
struct Qwen25:Base {
 int64_t max_keepalive_seq_=0;at::Tensor position_idx_keepalive_;
 bool generic=true;int64_t image_batch_count_=1;bool is_generic()const{return generic;}
 __Q25_SETTER__
 __Q25_VALID__
 void install_positions(){__Q25_ALLOC__}
};
constexpr int64_t DINOV3_VISION_NUM_SPECIAL_TOKENS=5;
struct Dino:Base {at::Tensor position_idx_keepalive_;__DINO_SETTER__};
template<class F>void rejects(F f){bool caught=false;try{f();}catch(const std::runtime_error&){caught=true;}assert(caught);}
void check_sdpa(int64_t seq_len_in,int64_t position){__SDPA__}
template<class Model>void check_vision(){
 for(int64_t capacity:{8193,16384,32768}){
  Model model;rejects([&]{model.set_execution_envelope(0,256);});
  model.set_execution_envelope(capacity,256);model.install_positions();
  assert(model.capacity_==capacity && model.position_idx_keepalive_.size(0)==capacity);
  auto owner=model.position_idx_keepalive_.owner;
  model.set_execution_envelope(capacity,128);model.install_positions();
  assert(model.position_idx_keepalive_.owner==owner && model.chunk_==128);
  rejects([&]{model.set_execution_envelope(capacity+16,128);});
  assert(model.capacity_==capacity && model.position_idx_keepalive_.owner==owner);
  model.planned=true;rejects([&]{model.set_execution_envelope(capacity,128);});
 }
}
int main(){
 check_vision<Qwen3>();check_vision<Qwen25>();
 Qwen25 q25;assert(q25.subclass_chunk_size_valid(65520,65520,0));
 assert(!q25.subclass_chunk_size_valid(65536,65521,0));
 assert(!q25.subclass_chunk_size_valid(65536,65536,0));
 q25.generic=false;q25.image_batch_count_=2;
 assert(q25.subclass_chunk_size_valid(65520,131040,0));
 assert(!q25.subclass_chunk_size_valid(65536,131072,0));
 for(int64_t capacity:{8197,16389,32773}){
  Dino model;for(int64_t bad:{0,5})rejects([&]{model.set_cache_envelope(bad,256);});
  model.set_cache_envelope(capacity,256);auto owner=model.position_idx_keepalive_.owner;
  assert(model.position_idx_keepalive_.size(0)==capacity-5);
  model.set_cache_envelope(capacity,128);assert(model.position_idx_keepalive_.owner==owner);
  rejects([&]{model.set_cache_envelope(capacity+16,128);});
  assert(model.capacity_==capacity && model.position_idx_keepalive_.owner==owner);
  model.planned=true;rejects([&]{model.set_cache_envelope(capacity,128);});
 }
 constexpr int64_t max_rows=int64_t{UINT16_MAX}*16;
 check_sdpa(32768,0);check_sdpa(1,max_rows-1);check_sdpa(max_rows,0);
 rejects([&]{check_sdpa(1,max_rows);});rejects([&]{check_sdpa(max_rows+1,0);});
 rejects([&]{check_sdpa(1,std::numeric_limits<int64_t>::max());});
}
'''
    for key, value in {
        "Q3_SETTER": definition(q3, "void set_execution_envelope("),
        "Q25_SETTER": definition(q25, "void set_execution_envelope("),
        "Q25_VALID": definition(q25, "bool subclass_chunk_size_valid(").replace(" override", ""),
        "DINO_SETTER": definition(dino, "void set_cache_envelope("),
        "Q3_ALLOC": q3_alloc, "Q25_ALLOC": q25_alloc, "SDPA": sdpa_guard,
    }.items():
        program = program.replace(f"__{key}__", value)
    cpp, exe = tmp_path / "dynamic_capacity.cpp", tmp_path / "dynamic_capacity"
    cpp.write_text(program)
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    str(cpp), "-o", str(exe)], check=True,
                   capture_output=True, text=True)
    subprocess.run([str(exe)], check=True, capture_output=True, text=True)
