"""Exercise the typed retained callers, including independent owner failures."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_exact_current_abi_and_owner_admission(tmp_path):
    code = '#include "' + str(ROOT / 'src/ops/rpu_pi_wall_fp16_retained_contracts.h') + '"\n' + r'''
#include <cassert>
#include <string>
using namespace rpu_wall_fp16_retained;
struct Kernel {
  std::array<uint16_t, 68> gp{};
  void set_regs(int p, uint16_t v) { gp.at(p) = v; }
  uint32_t u32(int p) const { return gp[p] | uint32_t(gp[p+1]) << 16; }
};
template<class T> void reject(const T& a) {
  Kernel k; k.gp.fill(0xa55a);
  try { write(k,a); assert(false); } catch(const std::invalid_argument&) {}
  for(auto v:k.gp) assert(v==0xa55a); // Reject before any register mutation.
  try { select(a); assert(false); } catch(const std::invalid_argument&) {}
}
int main() {
  PairedDownArgs p{{0x40000500,0,0x40200000,0,640,2048,1376,0,8,2752,4096,0,0,2},
                   0x40100000,0x40400000,320};
  Kernel k; write(k,p);
  assert(k.gp[8]==0 && k.gp[9]==640 && k.u32(10)==2048 && k.gp[12]==1376);
  assert(k.gp[13]==86 && k.gp[14]==0 && k.gp[15]==8);
  assert(k.u32(48)==p.second_input && k.u32(50)==p.second_output && k.gp[52]==320);
  assert(k.u32(22)==0 && k.u32(24)==0 && k.gp[26]==2 && k.gp[27]==2);
  auto s=select(p);
  assert(s.grid_x==32 && s.grid_y==1 && s.tile.m_tile==320 && s.tile.k_tile==128);
  assert(std::string(rpu_pi_wall::candidate_names[s.ordinal])==
      "paired_linear_retained_w16a16_acc16_m320x2n64k128_nobias");
  auto contiguous=p; contiguous.second_input=p.linear.input_spm_addr+320*2752;
  write(k,contiguous); assert(k.u32(48)==contiguous.second_input);
  auto lower=p; lower.linear.input_spm_addr=0x40500100; lower.second_input=0x40000500;
  lower.linear.output_spm_addr=0x40100000; lower.second_output=0x40300000;
  write(k,lower); assert(k.u32(0)>k.u32(48));
  auto bad=p;
  bad.second_input=0; reject(bad);
  bad=p; bad.second_input=p.linear.input_spm_addr+16; reject(bad);
  bad=p; bad.second_input=UINT64_C(0x100000000); reject(bad);
  bad=p; bad.second_input=0xfffffff0; reject(bad);
  bad=p; bad.second_output=p.linear.output_spm_addr+16; reject(bad);
  bad=p; bad.second_output=p.linear.input_spm_addr; reject(bad);
  bad=p; bad.second_output+=1; reject(bad);
  bad=p; bad.rows_per_input=319; reject(bad);
  bad=p; bad.acc32=true; reject(bad);
  bad=p; bad.zero_point=true; reject(bad);
  bad=p; bad.linear.local_m=639; reject(bad);
  bad=p; bad.linear.local_n=2032; reject(bad);
  bad=p; bad.linear.local_k=1360; reject(bad);
  bad=p; bad.linear.partition=1; reject(bad);
  bad=p; bad.linear.num_cores=4; reject(bad);
  bad=p; bad.linear.dtype_mode=1; reject(bad);
  bad=p; bad.linear.x_dtype=3; reject(bad);
  bad=p; bad.linear.has_bias=1; reject(bad);
  bad=p; bad.linear.bias_spm_addr=0x40700000; reject(bad);
  bad=p; bad.linear.scale_dev_addr=0x82000000; reject(bad);
  bad=p; bad.linear.input_m_byte_step+=32; reject(bad);
  bad=p; bad.linear.output_m_byte_step+=32; reject(bad);
  SharedSiluArgs q{{0x40000500,0,0x40300000,0,640,1376,2048,0,8,4096,2752,0,1,2}};
  write(k,q); s=select(q);
  assert(k.gp[8]==1 && k.gp[9]==640 && k.u32(10)==1376 && k.gp[12]==2048);
  assert(s.grid_x==15 && s.grid_y==1 && s.tile.m_tile==224 && s.tile.n_tile==96);
  assert(std::string(rpu_pi_wall::candidate_names[s.ordinal])==
      "shared_silu_retained_w16a16_acc16_m224x3n96k128_nobias");
  auto wrong=q; wrong.fused_silu=false; reject(wrong);
  wrong=q; wrong.acc32=true; reject(wrong);
  wrong=q; wrong.zero_point=true; reject(wrong);
  wrong=q; wrong.linear.local_m=320; reject(wrong);
  wrong=q; wrong.linear.local_n=1360; reject(wrong);
  wrong=q; wrong.linear.local_k=2032; reject(wrong);
  wrong=q; wrong.linear.partition=0; reject(wrong);
  wrong=q; wrong.linear.num_cores=4; reject(wrong);
  wrong=q; wrong.linear.dtype_mode=1; reject(wrong);
  wrong=q; wrong.linear.x_dtype=3; reject(wrong);
  wrong=q; wrong.linear.has_bias=1; reject(wrong);
  wrong=q; wrong.linear.bias_spm_addr=0x40700000; reject(wrong);
  wrong=q; wrong.linear.scale_dev_addr=0x82000000; reject(wrong);
  wrong=q; wrong.linear.input_m_byte_step+=32; reject(wrong);
  wrong=q; wrong.linear.output_m_byte_step+=32; reject(wrong);
  wrong=q; wrong.linear.output_spm_addr=wrong.linear.input_spm_addr+16; reject(wrong);
  // Existing generic ABI retains bias support and its own selection.
  auto generic=p.linear; generic.has_bias=1; generic.bias_spm_addr=0x40700000;
  rpu_pl_tiling::set_parallel_linear_regs(k,generic); assert(k.gp[14]==1);
  s=rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear,
       rpu_pl_tiling::WeightType::Fp16,false,320,2048,1376);
  assert(std::string(rpu_pi_wall::candidate_names[s.ordinal])==
       "paired_linear_w16a16_acc16_m768n64k64");
}
'''
    source = tmp_path / 'retained.cpp'
    source.write_text(code)
    exe = tmp_path / 'retained'
    subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    str(source), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
