"""DINO storage may grow independently of its RoPE coordinate/register widths."""
from pathlib import Path
import subprocess

import pytest
import torch

from rpu_backend.adapters import dinov3
from rpu_backend.api.errors import RPUBackendError
from test_qwen3vl_prefill_gateup_launcher import definition


@pytest.mark.parametrize("grid", [(1, 32768), (32768, 1)])
def test_position_axis_accepts_last_nonnegative_int16_coordinate(grid):
    positions = dinov3.build_position_idx(*grid)
    assert positions.dtype == torch.int16
    assert positions.max().item() == 32767
    dinov3._validate_dinov3_input_bounds(
        *grid, 32773, max_hw=32768, max_seq_len=65536)


@pytest.mark.parametrize("grid", [(1, 32769), (32769, 1)])
def test_position_axis_rejects_wraparound_before_position_allocation(monkeypatch, grid):
    monkeypatch.setattr(dinov3.torch, "arange",
                        lambda *a, **kw: pytest.fail("invalid axis reached allocation"))
    with pytest.raises(ValueError, match="int16"):
        dinov3.build_position_idx(*grid)
    with pytest.raises(RPUBackendError, match="int16"):
        dinov3._validate_dinov3_input_bounds(
            *grid, 32774, max_hw=32769, max_seq_len=65536)
    with pytest.raises(ValueError, match="int16"):
        dinov3.build_dinov3_rope_tables(head_dim=64, max_hw=32769, device="cpu")


def test_patch_count_preserves_rope_register_limit_without_limiting_cache():
    dinov3._validate_dinov3_input_bounds(
        255, 257, 65540, max_hw=257, max_seq_len=131072)
    with pytest.raises(RPUBackendError, match="uint16 token"):
        dinov3._validate_dinov3_input_bounds(
            256, 256, 65541, max_hw=256, max_seq_len=131072)


def test_native_rope_binding_and_execution_register_bounds(tmp_path):
    source = (Path(__file__).resolve().parents[1] /
              "src/fused/rpu_dinov3_vision_model.cpp").read_text()
    program = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <utility>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error("rejected"); } while(false)
namespace at {
constexpr int kHalf=1,kPrivateUse1=2;
struct Device {int type()const{return kPrivateUse1;}};
struct Tensor {
 int64_t rows;bool defined()const{return true;}int dim()const{return 2;}
 std::pair<int64_t,int64_t> sizes()const{return {rows,16};}
 int64_t size(int)const{return rows;}int scalar_type()const{return kHalf;}
 Device device()const{return {};}bool is_contiguous()const{return true;}
};
}
constexpr int64_t DINOV3_VISION_NUM_SPECIAL_TOKENS=5;
constexpr int64_t DINOV3_SDPA_MAX_KV_ROWS=int64_t{UINT16_MAX}*16;
__EXECUTION_VALID__
struct Model {
 at::Tensor freq_cos_{0},freq_sin_{0};int64_t max_hw_=0;bool has_rope_=false;
 int invalidations=0;void invalidate_model_state(){++invalidations;}
 __SET_ROPE__
};
int main(){
 Model model;
 model.set_rope_tables({32768},{32768});
 assert(model.max_hw_==32768 && model.has_rope_ && model.invalidations==1);
 for(int64_t bad:{0,32769}){
  bool rejected=false;try{model.set_rope_tables({bad},{bad});}
  catch(const std::runtime_error&){rejected=true;}
  assert(rejected && model.max_hw_==32768 && model.invalidations==1);
 }
 assert(dinov3_execution_length_valid(65540,false));
 assert(!dinov3_execution_length_valid(65541,false));
 assert(dinov3_execution_length_valid(131072,true));
 assert(dinov3_execution_length_valid(DINOV3_SDPA_MAX_KV_ROWS,true));
 assert(!dinov3_execution_length_valid(DINOV3_SDPA_MAX_KV_ROWS+1,true));
 assert(!dinov3_execution_length_valid(5,false));
}
'''
    program = program.replace("__EXECUTION_VALID__", definition(
        source, "bool dinov3_execution_length_valid("))
    program = program.replace("__SET_ROPE__", definition(
        source, "void set_rope_tables("))
    cpp, exe = tmp_path / "dino.cpp", tmp_path / "dino"
    cpp.write_text(program)
    subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    str(cpp), "-o", str(exe)], check=True, capture_output=True, text=True)
    subprocess.run([str(exe)], check=True)
