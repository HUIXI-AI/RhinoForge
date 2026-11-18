"""Execute the sole new-ABI selector/writer, including precision/group boundaries.

The full measured and interpolation parity matrix lives in test_linear_tiling_cache.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    compiler = shutil.which("g++")
    assert compiler, "Unified Linear host conformance needs g++"
    directory = tmp_path_factory.mktemp("unified-linear")
    source = directory / "probe.cpp"
    source.write_text(r'''
#include "src/ops/rpu_linear_tiling.h"
#include <array>
#include <cassert>
#include <iostream>
struct Kernel {
  std::array<uint16_t,128> regs{};
  void set_regs(int r,uint16_t value) { regs.at(r)=value; }
  void set_regs(int,uint32_t)=delete; // Reject SDK's adjacent-register overload.
};
int main() {
  using namespace rpu_pl_tiling;
  // Every call uses the new packet. Geometry, precision and addresses must
  // never regress to the retired packet, including >4 GiB device owners.
  const ParallelLinearRegisterArgs a{0x12345678,UINT64_C(0x2345678900),
      0x34567890,0x45678900,41,768,1536,1,8,3072,1536,
      UINT64_C(0x3456789a00),1,1};
  Kernel packet;
  set_parallel_linear_regs(packet,a);
  assert(packet.regs[8]==1 && packet.regs[9]==41);
  assert(packet.regs[10]==768 && packet.regs[11]==0);
  assert(packet.regs[12]==1536 && packet.regs[13]==96);
  assert(packet.regs[14]==1 && packet.regs[15]==8);
  assert(packet.regs[16]==3072 && packet.regs[18]==1536);
  assert(packet.regs[22]==0 && packet.regs[23]==0);
  assert(packet.regs[26]==2 && packet.regs[27]==1);
  assert(packet.regs[2]==0x6789 && packet.regs[3]==0x2345);
  assert(packet.regs[20]==0x789a && packet.regs[21]==0x3456);
  assert(packet.regs[0]==0x5678 && packet.regs[1]==0x1234);
  for(uint16_t mode:{1,2,3,5}) {
    auto precision=a;precision.dtype_mode=mode;
    set_parallel_linear_regs(packet,precision);
    assert(packet.regs[27]==mode);
  }
  int m,n,k,mode,acc,grouped,q;
  while(std::cin>>m>>n>>k>>mode>>acc>>grouped>>q) {
    try {
      WeightType type;
      switch(mode) {
        case 1:type=WeightType::Int8;break;
        case 2:type=WeightType::Fp16;break;
        case 3:type=WeightType::Bf16;break;
        case 5:type=WeightType::Fp8E4M3;break;
        case 8:type=WeightType::Uint4;break;
        default:throw std::invalid_argument("unsupported precision");
      }
      auto t=select_tile(m,n,k,type,acc,
                         grouped?Family::Grouped:Family::Parallel,q);
      const auto name=grouped?grouped_kernel_name(type,acc,t):parallel_kernel_name(type,acc,t);
      std::cout<<t.m_tile<<' '<<t.n_tile<<' '<<t.k_tile<<' '<<name<<'\n';
    } catch(const std::invalid_argument&) { std::cout<<"invalid\n"; }
  }
}
''')
    exe = directory / "probe"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT),
                    str(source), "-o", str(exe)], check=True, capture_output=True, text=True)
    return exe


def invoke(probe, cases):
    result = subprocess.run([str(probe)], input="".join(
        " ".join(map(str, row)) + "\n" for row in cases),
        capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


def test_packet_modes_and_invalid_geometry(probe):
    cases = [(0,32,128,2,0,0,0), (65536,32,128,2,0,0,0),
             (1,17,128,2,0,0,0), (1,32,16,1,0,0,0),
             (1,32,128,8,0,0,16), (1,32,64,8,0,0,128),
             (1,32,128,2,0,0,32), (1,32,128,0,0,0,0)]
    assert invoke(probe,cases) == ["invalid"]*len(cases)


def test_precision_and_group_compatibility_tiles(probe):
    # BF16/FP8 and Q64/Q128 are unmeasured by the FP16/I8/Q32 table.
    # Their explicit compatibility choice must come from the current selector.
    cases = [(41,768,1536,mode,acc,g,0)
             for mode in (3,5) for acc in (0,1) for g in (0,1)]
    cases += [(17,96,512,8,acc,g,q)
              for acc in (0,1) for g in (0,1) for q in (64,128)]
    for case,row in zip(cases,invoke(probe,cases)):
        assert row.split()[1:3] == ["64" if case[5] else "128", "128"]


def test_grouped_uint4_tiles_preserve_quantization_groups(probe):
    shapes = [(64,2048), (2048,64), (96,384), (128,1024)]
    cases = [(1,n,k,8,acc,1,q) for n,k in shapes
             for acc in (0,1) for q in (32,64,128) if k%q==0]
    rows = invoke(probe,cases)
    for case,row in zip(cases,rows):
        m,n,k,name = row.split()
        assert int(m)>0 and int(n)%16==0 and int(k)%case[6]==0
        assert name.startswith("grouped_parallel_linear_")
