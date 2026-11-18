"""Host selector checks, including differential tests against rhino-ops source.

Set RHINO_OPS_SOURCE to the matching main checkout for the source differential
checks; a sibling ops-main checkout is detected during coordinated development.
Set RHINO_OPS_EXPANSION_SOURCE to the matching expansion checkout for its
backend metadata exporter when the two branches use separate checkouts.
"""

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def upstream():
    path = Path(os.environ.get("RHINO_OPS_SOURCE", ROOT.parent / "ops-main"))
    source = path / "generator/parallel_linear_tiling.py"
    if not source.is_file():
        pytest.skip("Set RHINO_OPS_SOURCE for source-backed selector differential checks")
    spec = importlib.util.spec_from_file_location("parallel_linear_tiling", source)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield path, module
    if previous is None:
        del sys.modules[spec.name]
    else:
        sys.modules[spec.name] = previous


@pytest.fixture(scope="module")
def selector_probe(tmp_path_factory):
    compiler = shutil.which("g++")
    assert compiler, "selector checks require a C++ compiler"
    directory = tmp_path_factory.mktemp("linear-tiling")
    source = directory / "probe.cpp"
    source.write_text(r'''
#include "src/ops/rpu_linear_tiling.h"
#include <atomic>
#include <cassert>
#include <iostream>
#include <set>
#include <thread>
using namespace rpu_pl_tiling;
void same(TilePick a,TilePick b) {
  assert(a.m_tile==b.m_tile && a.n_tile==b.n_tile && a.k_tile==b.k_tile);
}
void memo() {
  detail::TileSelectionCache cache;
  TilePick t;
  assert(!cache.lookup(0,1,32,32,t));
  for(int i=0;i<1024;++i) {
    const uint8_t family=i%24;
    cache.store(family,i,UINT64_C(1)<<32 | unsigned(i),i+1,{i,i+1,i+2});
    assert(cache.lookup(family,i,UINT64_C(1)<<32 | unsigned(i),i+1,t));
    same(t,{i,i+1,i+2});
    assert(!cache.lookup(family,i,i,i+1,t));
    assert(!cache.lookup(family+1,i,UINT64_C(1)<<32 | unsigned(i),i+1,t));
  }
  unsigned retained=0;
  for(int i=0;i<1024;++i)
    if(cache.lookup(i%24,i,UINT64_C(1)<<32 | unsigned(i),i+1,t)) {
      ++retained; same(t,{i,i+1,i+2});
    }
  assert(retained>0 && retained<=cache.capacity);
  std::atomic<int> ready{0};
  std::array<const void*,4> locations{};
  std::vector<std::thread> threads;
  for(int id=0;id<4;++id) threads.emplace_back([&,id] {
    locations[id]=&detail::selection_cache();
    ++ready; while(ready.load()!=4) std::this_thread::yield();
    for(int i=0;i<160;++i) for(int dtype=0;dtype<4;++dtype) {
      const auto type=static_cast<WeightType>(dtype);
      const int m=17+(i+id)%97,n=32+32*(i%33),k=128+32*(i%37);
      for(bool acc32:{false,true}) for(auto family:{Family::Parallel,Family::Grouped}) {
        const auto expected=detail::measured_select(m,n,k,type,acc32,family);
        same(select_tile(m,n,k,type,acc32,family),expected);
        same(select_tile(m,n,k,type,acc32,family),expected);
      }
    }
  });
  for(auto& t:threads)t.join();
  assert(std::set<const void*>(locations.begin(),locations.end()).size()==4);
}
void catalog() {
  std::set<std::string> seen;
  for(std::size_t i=0;i<kParallelCandidateCount;++i) {
    const auto& c=detail::parallel_candidates[i];
    assert(parallel_kernel_ordinal(c.weight,c.acc32,c.tile)==int(i));
    assert(parallel_kernel_name(c.weight,c.acc32,c.tile)==std::string(c.name));
    same(make_tile(c.weight,c.acc32,c.tile.n_tile,c.tile.k_tile),c.tile);
    assert(seen.emplace(c.name).second);
    std::cout<<"P "<<c.name<<"\n";
  }
  seen.clear();
  for(std::size_t i=0;i<kGroupedCandidateCount;++i) {
    const auto& c=detail::grouped_candidates[i];
    assert(grouped_kernel_ordinal(c.weight,c.acc32,c.tile)==int(i));
    same(make_tile(c.weight,c.acc32,c.tile.n_tile,c.tile.k_tile,Family::Grouped),c.tile);
    assert(seen.emplace(c.name).second);
    std::cout<<"G "<<c.name<<"\n";
  }
}
int main(int argc,char** argv) {
  if(argc==2) {
    const std::string mode=argv[1];
    if(mode=="memo") memo();
    else if(mode=="catalog") catalog();
    else return 2;
    return 0;
  }
  char mode;int dtype,acc,family,m,k,group,xd,nt,kt;int64_t n;
  while(std::cin>>mode>>dtype>>acc>>family>>m>>n>>k>>group>>xd>>nt>>kt) {
    try {
      TilePick t;
      if(mode=='C') t=make_tile(static_cast<WeightType>(dtype),acc,nt,kt,static_cast<Family>(family),m);
      else if(mode=='V')t=select_tile_nvfp4_acc32(m,int(n),k);
      else t=select_tile(m,n,k,static_cast<WeightType>(dtype),acc,static_cast<Family>(family),group,xd,nt,kt);
      std::cout<<t.m_tile<<' '<<t.n_tile<<' '<<t.k_tile<<' '<<t.metadata_rows<<' '
               <<t.zp_offset<<' '<<t.vlm_rows<<' '<<t.bias_offset<<' '<<t.msizes_offset<<'\n';
    } catch(const std::invalid_argument&) {std::cout<<"ERROR\n";}
  }
}
''')
    executable = directory / "probe"
    built = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-pthread", "-I", str(ROOT),
         str(source), "-o", str(executable)],
        capture_output=True, text=True, timeout=60,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    return executable


def probe(executable, queries):
    result = subprocess.run([str(executable)], input="\n".join(queries) + "\n",
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return [None if line == "ERROR" else tuple(map(int, line.split()))
            for line in result.stdout.splitlines()]


def values(tile):
    return (tile.m, tile.n, tile.k, tile.metadata_rows, tile.zp_offset,
            tile.vlm_rows, tile.bias_offset, tile.msizes_offset)


def _normalize_tiling_comment(content):
    # The pinned exporter includes a build-host label in this one comment.
    # Preserve every other byte, including its source SHA and all table data.
    lines = content.splitlines(keepends=True)
    if len(lines) > 2 and hashlib.sha256(lines[2]).hexdigest() == (
        "04f0441189c54bd460abe1e31e5b3727711397d26a4e5232b7f800f36769ebcc"
    ):
        lines[2] = b"// Included inside rpu_pl_tiling::detail; FP16 activation, G32 W4.\n"
    return b"".join(lines)


def test_generated_metadata_matches_upstream(upstream, tmp_path):
    path, _ = upstream
    expansion = Path(os.environ.get("RHINO_OPS_EXPANSION_SOURCE", path))
    exporter = expansion / "rpu_backend/ddr_expansion_ops/generator/export_backend_linear_tiling.py"
    assert exporter.is_file(), "Set RHINO_OPS_EXPANSION_SOURCE to the matching expansion checkout"
    (tmp_path / "src/ops").mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, "-B", str(exporter), "--rhino-ops", str(path),
         "--backend-root", str(tmp_path)], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("rpu_linear_tiling_data.h", "rpu_linear_candidates.inc"):
        expected = (tmp_path / "src/ops" / name).read_bytes()
        actual = (ROOT / "src/ops" / name).read_bytes()
        if name == "rpu_linear_tiling_data.h":
            expected = _normalize_tiling_comment(expected)
            actual = _normalize_tiling_comment(actual)
        assert actual == expected, f"Stale generated selector: {name}"


def test_tiling_comment_normalization_preserves_other_bytes():
    # The source-backed exporter test above covers the pinned legacy line.
    portable = (b"// Generated\n// source SHA256: abc\n"
                b"// Included inside rpu_pl_tiling::detail; FP16 activation, G32 W4.\n"
                b"inline constexpr int table[] = {1, 2};\n")
    assert _normalize_tiling_comment(portable) == portable
    for changed in (
        portable.replace(b"abc", b"def"),
        portable.replace(b"{1, 2}", b"{1, 3}"),
        portable.replace(b"// Generated", b"// Other"),
        portable.replace(b"G32 W4.", b"G32 W4, other-host."),
        portable + portable.splitlines(keepends=True)[2],
    ):
        assert _normalize_tiling_comment(changed) != portable


def test_all_exact_and_unmeasured_queries_match_python(selector_probe, upstream):
    _, reference = upstream
    queries, expected = [], []
    rng = random.Random(42)
    for di, dtype in enumerate(("fp16", "int8", "int4", "uint4")):
        for acc in (16, 32):
            entries = reference._load_table_data()["tables"][f"{dtype}_a{acc}"]
            shapes = [tuple(e["shape"]) for e in entries]
            # Non-table shapes, different M at an exact N/K, small and far
            # extrapolation, tails, plus a selection larger than memo capacity.
            shapes += [(17 + rng.randrange(1200), 32 * rng.randrange(1,700),
                        32 * rng.randrange(1,700)) for _ in range(80)]
            shapes += [(entry["shape"][0] + 1, *entry["shape"][1:])
                       for entry in entries[::31]]
            shapes += [(1,32,32),(65535,32,32),(17,524288,32768),(1,32,65504)]
            for family, fname in enumerate(("parallel", "grouped")):
                for m, n, k in shapes:
                    queries.append(f"S {di} {int(acc==32)} {family} {m} {n} {k} -1 2 0 0")
                    try:
                        expected.append(values(reference.select_tile(m,n,k,dtype,acc,family=fname)))
                    except ValueError:
                        expected.append(None)
    actual = probe(selector_probe, queries)
    assert len(actual) == len(expected)
    differences = [(q,a,e) for q,a,e in zip(queries,actual,expected) if a != e]
    assert not differences, differences[:20]


def test_capacity_explicit_and_precision_paths(selector_probe, upstream):
    _, reference = upstream
    queries, expected = [], []
    for di, dtype in enumerate(("fp16", "int8", "int4", "uint4")):
        for acc in (16,32):
            for family, fname in enumerate(("parallel","grouped")):
                for n in range(32,129,16):
                    for k in (32,64,96,128,192,256,384,512,768):
                        queries.append(f"C {di} {int(acc==32)} {family} 0 32 32 -1 2 {n} {k}")
                        try:
                            expected.append(values(reference.make_tile(dtype,acc,n,k,family=fname)))
                        except ValueError:
                            expected.append(None)
    assert probe(selector_probe, queries) == expected
    # Default BF16/FP8 and G64/G128 deliberately do not use measured G32/FP16.
    rows = [
        ("S 4 0 0 17 512 4096 0 3 0 0", reference.make_tile("bf16",16,128,128)),
        ("S 5 1 0 17 512 4096 0 2 0 0", reference.make_tile("int8",32,128,128)),
        ("S 2 0 0 17 512 4096 64 2 0 0", reference.make_tile("int4",16,128,128)),
        ("S 3 1 1 17 512 4096 128 2 0 0", reference.make_tile("uint4",32,64,128,family="grouped")),
        ("S 2 1 0 17 512 4096 64 2 64 256", reference.make_tile("int4",32,64,256)),
    ]
    assert probe(selector_probe,[q for q,_ in rows]) == [values(t) for _,t in rows]
    rejected = [
        "S 0 0 0 0 512 4096 -1 2 0 0",  # M0
        "S 2 0 0 1 48 4096 32 2 0 0",  # logical N48 must first pad to N64
        "S 0 0 0 1 32 65536 -1 2 0 0",  # K ABI overflow
        "S 3 0 0 1 64 128 128 2 64 96", # incomplete quant group
        "S 0 0 0 1 32 128 -1 2 0 256",  # K requires N
        "S 0 0 0 1 16777216 128 -1 2 32 128", # grid overflow
        "S 0 0 0 1 32 128 32 2 0 0",    # W16 has no quant group
        "S 2 0 0 1 32 128 32 2 48 128", # W4 N tile must align32
    ]
    assert probe(selector_probe, rejected) == [None] * len(rejected)


def test_catalog_and_bounded_thread_local_cache(selector_probe):
    for mode in ("memo", "catalog"):
        result = subprocess.run([str(selector_probe),mode], capture_output=True,
                                text=True,timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr


def test_nvfp4_retains_its_independent_selector(selector_probe):
    fixture = json.loads((ROOT / "tests/fixtures/parallel_linear_nvfp4_preserved.json").read_text())
    shapes = [row["shape"] for row in fixture["queries"]]
    expected = [tuple(row["m_n_k"]) for row in fixture["queries"]]
    actual = probe(selector_probe,[f"V 0 0 0 {m} {n} {k} -1 2 0 0" for m,n,k in shapes])
    assert [t[:3] for t in actual] == expected
