"""Compile the production W8A8 descriptor and exercise its ABI boundaries.

No device or mocked ATen is needed: these checks cover the real header used by
both immediate and retained-graph launches, not a Python reimplementation.
"""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def contract_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    assert compiler, "W8A8 launch contract checks require g++"
    folder = tmp_path_factory.mktemp("w8a8_contract")
    source = folder / "contract.cpp"
    source.write_text(r'''
#include "rpu_linear_w8a8_contract.h"
#include <cassert>
#include <iostream>
#include <set>
#include <string>
using namespace rpu_pl_w8a8;
using P = Partition;
constexpr uint32_t base = 0x10000000;
constexpr uint64_t usable = 8 * 1024 * 1024 - 1024;
uint64_t align32(uint64_t n) { return (n + 31) / 32 * 32; }
Addresses addresses(const Shape& s) {
  const auto z = spans(s);
  Addresses a{};
  a.x = base;
  a.sx = base + align32(z.x);
  a.y = a.sx + align32(z.sx);
  a.bias = s.bias ? a.y + align32(z.y) : 0;
  a.w = 0xabcdef0000ULL;
  a.sw = 0x1234560000ULL;
  return a;
}
uint32_t read32(const Descriptor& d, int slot) {
  return uint32_t(d.regs[slot]) | uint32_t(d.regs[slot + 1]) << 16;
}
template<class F> void rejects(F f) {
  bool rejected = false;
  try { f(); } catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}
void catalog() {
  std::set<std::string> names;
  std::set<std::size_t> ordinals;
  int resident = 0;
  for (const auto& c : installed_candidates()) {
    const auto k = c.compiled_local_k ? c.compiled_local_k : 32;
    const auto t = validate_tile(c.tile, k, true);
    assert(t.vlm_rows > 0 && t.vlm_rows <= 400);
    assert(installed_name(t, k) == std::string(c.name));
    names.emplace(c.name); ordinals.insert(c.ordinal);
    if (c.compiled_local_k) {
      ++resident;
      // Resident symbol identity includes compiled local K even when the
      // alternative K would fit and still satisfy SINGLE_K.
      rejects([&]{ installed_name(t, k == 32 ? 64 : k - 32); });
    } else {
      assert(installed_name(t, 65504) == std::string(c.name));
    }
  }
  assert(names.size() == 29 && ordinals.size() == 29 && resident == 10);
  assert(*ordinals.begin() == 0 && *ordinals.rbegin() == 28);
  Shape qkv{800, 256, 2048, 8, P::Col, false};
  auto t = select_tile(qkv);
  assert(t.m == 144 && t.n == 48 && t.k == 512 && t.resident);
  assert(t.vlm_rows == 393 && t.keep_sw && t.sx_fused && t.hoist_valu);
  assert(std::string(t.source) == "pi05_compact:lang_qkv_m800");
  // Each complete measured-key field matters; unknown shapes never inherit
  // a nearby model's measurement or use the A16 tile-capacity formula.
  for (auto changed : {Shape{799,256,2048,8,P::Col,false},
                       Shape{800,272,2048,8,P::Col,false},
                       Shape{800,256,2016,8,P::Col,false},
                       Shape{800,256,2048,7,P::Col,false},
                       Shape{800,256,2048,8,P::Row,false},
                       Shape{800,256,2048,8,P::Col,true}}) {
    t = select_tile(changed);
    assert(t.m == 128 && t.n == 64 && t.k == 128 && !t.resident);
    assert(t.vlm_rows == 112);
    assert(std::string(t.source) == "unmeasured_general_fallback");
  }
  auto unsupported = select_tile(qkv);
  unsupported.m = 128;  // Fits VLM but there is no matching fixed symbol.
  validate_tile(unsupported, qkv.local_k, qkv.bias);
  rejects([&]{ installed_name(unsupported, qkv.local_k); });
  t = select_tile({768,1152,160,8,P::Row,true});
  assert(t.m == 464 && t.n == 64 && t.k == 160 && t.n_major);
  assert(t.single_k && t.hoist_valu && t.vlm_rows == 397);
  t = select_tile({768,544,1152,8,P::Col,true});
  assert(t.keep_bias && t.keep_sw && t.vlm_rows == 392);
  rejects([&]{ validate_tile(t, 1152, false); });
  t.n_major = true;
  rejects([&]{ validate_tile(t, 1152, true); });
}
void dispatch() {
  // Every supported core domain and both partitions use the same signed A8
  // ABI. N and K are physical storage dimensions, never implicitly padded.
  for (int cores = 1; cores <= 8; ++cores) {
    for (auto part : {P::Row, P::Col}) {
      const auto s = physical_shape(17, part == P::Col ? 80 * cores : 80,
                                  part == P::Row ? 96 * cores : 96,
                                  part, cores, true);
      const auto a = addresses(s);
      const auto d = make_descriptor(s, a, base, usable);
      assert(read32(d,0) == a.x && read32(d,4) == a.y);
      assert(read32(d,6) == a.bias && read32(d,30) == a.sx);
      assert(read32(d,2) == a.w / 256 && read32(d,20) == a.sw / 256);
      assert(d.regs[8] == (part == P::Col));
      assert(d.regs[9] == 17 && read32(d,10) == 80);
      assert(d.regs[12] == 96 && d.regs[13] == 6);
      assert(d.regs[14] == 1 && d.regs[15] == cores);
      assert(read32(d,16) == 96);  // INT8 input pitch: no Half multiplier.
      assert(read32(d,18) == 160 && d.regs[26] == 2 && d.regs[27] == 1);
      assert((d.grid == std::array<uint16_t,3>{2,1,1}));
      for (int slot : {22,23,24,25,28,29}) assert(d.regs[slot] == 0);
      const auto z = spans(s);
      assert(z.sx == 64 && z.x == 17 * 96 && z.y == 17 * 80 * 2);
      assert(z.weight == uint64_t(80 * 96 * cores));
      assert(z.scale == uint64_t(160 * (part == P::Col ? cores : 1)));
      auto pending = a; pending.w = pending.sw = 0;
      const auto graph = make_descriptor(s, pending, base, usable);
      assert(read32(graph,2) == 0 && read32(graph,20) == 0);
    }
  }
  Shape measured{800,256,2048,8,P::Col,false};
  auto d = make_descriptor(measured,addresses(measured),base,usable);
  assert((d.grid == std::array<uint16_t,3>{6,1,1}));
  measured.m = 400;
  d = make_descriptor(measured,addresses(measured),base,usable);
  assert(d.tile.m == 416 && !d.tile.resident && d.grid[1] == 1);
  measured.m = 801;
  d = make_descriptor(measured,addresses(measured),base,usable);
  assert((d.grid == std::array<uint16_t,3>{4,7,1}));
  // Explicit override is allowed only for installed kernels and preserves all
  // safety checks. The caller must benchmark it before claiming optimality.
  auto explicit_tile = installed_candidates()[7].tile;
  d = make_descriptor(measured,addresses(measured),base,usable,&explicit_tile);
  assert(d.ordinal == 7 && d.grid[1] == 3);
  // The full-prefix Q route shares tile19 at both camera geometries. Keep
  // actual M in the descriptor and validate the entire output, including
  // the rows beyond the old half-prefix allocation.
  for (int64_t m : {544,800}) {
    const auto s = physical_shape(m,2048,2048,P::Col,8,false);
    const auto a = addresses(s);
    const auto full = make_descriptor(s,a,base,usable);
    const auto bytes = spans(s);
    assert(full.ordinal == 19 && full.tile.resident && full.tile.vlm_rows == 393);
    assert((full.grid == std::array<uint16_t,3>{6,1,1}));
    assert(full.regs[8] == 1 && full.regs[9] == m && full.regs[12] == 2048);
    assert(read32(full,10) == 256 && read32(full,16) == 2048);
    assert(read32(full,18) == 512 && read32(full,30) == a.sx);
    assert(bytes.x == uint64_t(m*2048) && bytes.y == uint64_t(m*512));
    assert(bytes.sx == uint64_t(m*2));
    rejects([&]{ make_descriptor(s,a,base,uint64_t(a.y-base)+bytes.y/2); });
  }
  // Production paired Pi Down reuses the original TP8 row-packed W/SW.
  // The two compact quantizers feed one contiguous full-M resident call;
  // M544 must not fall back to the generic tile while M800 stays unchanged.
  for (int64_t m : {544,800}) {
    const auto s = physical_shape(m,2048,16384,P::Row,8,false);
    const auto a = addresses(s);
    const auto full = make_descriptor(s,a,base,usable);
    const auto bytes = spans(s);
    assert(full.ordinal == 27 && full.tile.resident && full.tile.vlm_rows == 397);
    assert((full.grid == std::array<uint16_t,3>{26,1,1}));
    assert(full.regs[8] == 0 && full.regs[9] == m && full.regs[12] == 2048);
    assert(read32(full,10) == 2048 && read32(full,16) == 2048);
    assert(read32(full,18) == 4096 && full.regs[26] == 2 && full.regs[27] == 1);
    assert(read32(full,30) == a.sx && bytes.sx == uint64_t(m*2));
    assert(bytes.x == uint64_t(m*2048) && bytes.y == uint64_t(m*4096));
    assert(bytes.weight == 2048ULL*16384 && bytes.scale == 2048*2);
    // A half-sized Y window is insufficient even though each old A16
    // output was a separate half. Reject it before any launch is staged.
    rejects([&]{ make_descriptor(s,a,base,uint64_t(a.y-base)+bytes.y/2); });
  }
}
void local_candidates() {
  struct Expected {
    Shape shape;
    std::size_t ordinal;
    std::array<uint16_t,3> grid;
  };
  const std::array<Expected,25> trials{{
      {{272,256,2048,8,P::Col,false},10,{6,1,1}},
      {{272,2048,2048,8,P::Row,false},27,{26,1,1}},
      {{544,256,2048,8,P::Col,false},19,{6,1,1}},
      {{544,2048,2048,8,P::Row,false},27,{26,1,1}},
      {{512,160,1152,8,P::Col,true},16,{3,1,1}},
      {{512,1152,160,8,P::Row,true},24,{18,1,1}},
      {{512,544,1152,8,P::Col,true},21,{9,1,1}},
      {{512,1152,544,8,P::Row,true},20,{15,1,1}},
      {{128,3200,5120,8,P::Col,false},3,{50,1,1}},
      {{128,5120,3200,8,P::Row,false},1,{107,1,1}},
      {{192,1024,5120,8,P::Col,false},3,{16,1,1}},
      {{192,128,5120,8,P::Col,false},3,{2,1,1}},
      {{192,5120,1024,8,P::Row,false},3,{80,1,1}},
      {{192,3200,5120,8,P::Col,false},3,{50,1,1}},
      {{192,5120,3200,8,P::Row,false},3,{80,1,1}},
      {{256,1024,5120,8,P::Col,false},4,{16,1,1}},
      {{256,128,5120,8,P::Col,false},4,{2,1,1}},
      {{256,5120,1024,8,P::Row,false},15,{80,1,1}},
      {{256,3200,5120,8,P::Col,false},4,{50,1,1}},
      {{256,5120,3200,8,P::Row,false},4,{80,1,1}},
      {{512,1024,5120,8,P::Col,false},15,{16,1,1}},
      {{512,128,5120,8,P::Col,false},15,{2,1,1}},
      {{512,5120,1024,8,P::Row,false},15,{80,1,1}},
      {{512,3200,5120,8,P::Col,false},15,{50,1,1}},
      {{512,5120,3200,8,P::Row,false},15,{80,1,1}},
  }};
  for (const auto& trial : trials) {
    const auto& s = trial.shape;
    const auto d = make_descriptor(s,addresses(s),base,usable);
    assert(d.ordinal == trial.ordinal && d.grid == trial.grid);
    // M is the actual runtime row count, not the larger tile or source M.
    assert(d.regs[9] == s.m && d.regs[12] == s.local_k);
    assert(std::string(d.tile.source).find("local_candidate:") == 0);
    assert(d.tile.vlm_rows <= 400);
    if (s.m == 256) {
      assert((s.m + d.tile.m - 1) / d.tile.m == 1 && d.grid[1] == 1);
      assert(d.tile.n == 64 && !d.tile.resident);
      assert(d.tile.m == (trial.ordinal == 15 ? 512 : 272));
      assert(d.tile.k == (trial.ordinal == 15 ? 128 : 384));
    }
    if (s.m == 512 && !s.bias) {
      assert(d.tile.m == 512 && d.tile.n == 64 && d.tile.k == 128);
      assert(d.grid[1] == 1 && !d.tile.resident);
    }
    if (d.tile.resident)
      assert(installed_candidates()[d.ordinal].compiled_local_k == s.local_k);
    // Local trials must not silently widen into neighboring shapes/profiles.
    auto nearby = s;
    nearby.m += 1;
    assert(std::string(select_tile(nearby).source) == "unmeasured_general_fallback");
    nearby = s; nearby.cores = 7;
    assert(std::string(select_tile(nearby).source) == "unmeasured_general_fallback");
    nearby = s; nearby.bias = !s.bias;
    assert(std::string(select_tile(nearby).source) == "unmeasured_general_fallback");
    nearby = s; nearby.local_n += 16;
    assert(std::string(select_tile(nearby).source) == "unmeasured_general_fallback");
    nearby = s; nearby.local_k += 32;
    assert(std::string(select_tile(nearby).source) == "unmeasured_general_fallback");
    nearby = s; nearby.partition = s.partition == P::Col ? P::Row : P::Col;
    assert(std::string(select_tile(nearby).source) == "unmeasured_general_fallback");
  }
  // All upstream measured choices remain exactly as published.
  for (const auto& entry : detail::exact) {
    const auto selected = select_tile(entry.shape);
    assert(installed_candidate(selected,entry.shape.local_k).ordinal == entry.ordinal);
    assert(std::string(selected.source) == entry.source);
  }
}
void bounds() {
  for (auto s : {Shape{0,16,32,1,P::Col,false},
                 Shape{65536,16,32,1,P::Col,false},
                 Shape{1,15,32,1,P::Col,false},
                 Shape{1,(1<<20)+16,32,1,P::Col,false},
                 Shape{1,16,65536,1,P::Col,false},
                 Shape{1,16,33,1,P::Col,false},
                 Shape{1,16,32,0,P::Col,false},
                 Shape{1,16,32,9,P::Col,false},
                 Shape{1,16,32,1,static_cast<P>(2),false}})
    rejects([&]{ select_tile(s); });
  rejects([]{ physical_shape(3,65,96,P::Col,2,false); });
  rejects([]{ physical_shape(3,80,95,P::Row,2,false); });
  rejects([]{ physical_shape(3,80,96,P::Col,3,false); });
  validate_shape({65535,1<<20,65504,8,P::Row,false});
  Shape s{17,80,96,4,P::Col,true};
  const auto a = addresses(s);
  auto fail = [&](Addresses changed, uint32_t b=base, uint64_t bytes=usable) {
    rejects([&]{ make_descriptor(s,changed,b,bytes); });
  };
  auto changed = a; changed.sx = changed.x; fail(changed);
  changed = a; changed.y = changed.sx; fail(changed);
  changed = a; changed.bias = changed.y; fail(changed);
  changed = a; changed.x = base - 32; fail(changed);
  changed = a; changed.sx++; fail(changed);
  changed = a; changed.y = base + usable - 32; fail(changed);
  changed = a; changed.w++; fail(changed);
  changed = a; changed.sw = (1ULL<<40) - 256; fail(changed);
  changed = a; changed.w = (1ULL<<40) - 256; fail(changed);
  changed = a; changed.w = 1ULL<<40; fail(changed);
  fail(a,base,0); fail(a,base,8*1024*1024+1);
  fail(a,0xffffffe0,64);
  auto t = select_tile(s); t.m = 528;
  rejects([&]{ validate_tile(t,96,true); });
  t = select_tile(s); t.keep_sw = true;
  rejects([&]{ validate_tile(t,96,true); });
  t = select_tile(s); t.m = -16;
  rejects([&]{ validate_tile(t,96,true); });
  s.bias = false;
  rejects([&]{ make_descriptor(s,a,base,usable); });
  s = {65535,65536,65504,8,P::Col,false};
  rejects([&]{ make_descriptor(s,{},base,usable); });
}
int main(int argc,char** argv) {
  assert(argc == 2);
  const std::string mode=argv[1];
  if(mode=="catalog") catalog();
  else if(mode=="dispatch") dispatch();
  else if(mode=="local") local_candidates();
  else if(mode=="bounds") bounds();
  else if(mode=="select") {
    int64_t m,n,k; int cores,part,bias;
    while (std::cin>>m>>n>>k>>cores>>part>>bias) {
      const Shape s{m,n,k,cores,static_cast<P>(part),bool(bias)};
      const auto t=select_tile(s);
      std::cout<<installed_name(t,k)<<' '<<t.vlm_rows<<' '<<t.source<<'\n';
    }
  } else return 2;
}
''')
    binary = folder / "contract"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         "-I", str(ROOT / "src/ops"), str(source), "-o", str(binary)],
        check=True, capture_output=True, text=True,
    )
    return binary


@pytest.mark.parametrize("mode", ["catalog", "dispatch", "local", "bounds"])
def test_production_w8a8_contract(contract_binary, mode):
    subprocess.run([str(contract_binary), mode], check=True, capture_output=True, text=True)
