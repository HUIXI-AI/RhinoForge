"""Bounded device-M adoption: real host selection and COMPLETE identities.

Set RHINO_OPS_EXPANSION_SOURCE to the matching expansion checkout. Main selector
checks separately use RHINO_OPS_SOURCE; the two source identities are independent.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_generated_device_contract_matches_ops_source(monkeypatch):
    configured = os.environ.get("RHINO_OPS_EXPANSION_SOURCE")
    if not configured:
        pytest.skip("RHINO_OPS_EXPANSION_SOURCE is not set")
    generator = Path(configured) / "rpu_backend/ddr_expansion_ops/generator"
    paths = tuple(generator / f"grouped_parallel_linear_{suffix}.py"
                  for suffix in ('device_m', 'w8_device_m', 'extra_device_m'))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        message = "RHINO_OPS_EXPANSION_SOURCE needs device-M generators: " + ", ".join(missing)
        if configured:
            pytest.fail(message)
        pytest.skip(message)
    monkeypatch.syspath_prepend(str(generator))
    for path in paths:
        suffix = path.stem.removeprefix("grouped_parallel_linear_")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        assert (ROOT / f"src/ops/rpu_grouped_{suffix}_contract.h").read_text() == module.emit_cpp()


def test_real_auto_selection_and_boundaries(tmp_path):
    native = (ROOT / "src/ops/rpu_grouped_parallel_linear.cpp").read_text()
    start = native.index("std::array<int64_t, 3> rpu_select_grouped_parallel_linear_uint4_tile(")
    public_query = native[start:native.index("\nvoid rpu_launch_grouped_parallel_linear_uint4_spm_kernel(", start)]
    cpp = tmp_path / "dispatch.cpp"
    cpp.write_text(r'''
#include "rpu_grouped_linear_tiling.h"
#include <cassert>
#include <stdexcept>
using namespace rpu_grouped_tiling;
__PUBLIC_QUERY__
template<class F> void rejected(F fn) {
    bool failed=false; try { fn(); } catch(const std::invalid_argument&) { failed=true; }
    assert(failed);
}
int main() {
    for(int total : {1,16,95,96,97,128,224,256,1024,65535}) {
        const auto d=select_dispatch(total,256,512,2,false);
        assert(d.tile.n_tile==80 && d.tile.k_tile==512 && d.tile.m_tile==96);
        assert(d.kind==(total>96 ? DispatchKind::Fp16Acc16DeviceM : DispatchKind::Static));
        // Force the same geometry explicitly: that identifies the static
        // program, even if the geometry equals today's automatic candidate.
        assert(select_dispatch(total,256,512,2,false,2,80,512).kind==DispatchKind::Static);
        assert(select_dispatch(total,256,512,2,false,2,80,0).kind==DispatchKind::Static);
        rejected([&]{select_dispatch(total,256,512,2,false,2,0,512);});
        assert(select_dispatch(total,256,512,2,true).kind==
            (total>80 ? DispatchKind::Fp16Acc32N80DeviceM : DispatchKind::Static));
        assert(select_dispatch(total,256,512,1,false).kind==
            (total>48 ? DispatchKind::W8Acc16N48DeviceM : DispatchKind::Static));
        assert(select_dispatch(total,256,512,5,false).kind==
            (total>48 ? DispatchKind::W8Acc16N48DeviceM : DispatchKind::Static));
        assert(select_dispatch(total,256,512,2,false,3).kind==DispatchKind::Static);
    }
    for(int total : {1,47,48,49,96,256,65535}) {
        const auto d=select_dispatch(total,1024,256,1,true);
        assert(d.tile.n_tile==112 && d.tile.k_tile==384 && d.tile.m_tile==48);
        assert(d.kind==(total>48 ? DispatchKind::W8Acc32DeviceM : DispatchKind::Static));
        assert(select_dispatch(total,1024,256,1,true,2,112,384).kind==DispatchKind::Static);
        assert(select_dispatch(total,1024,256,1,true,2,112,0).kind==DispatchKind::Static);
        assert(select_dispatch(total,1024,256,1,false).kind==DispatchKind::Static);
        assert(select_dispatch(total,1024,256,2,true).kind==
            (total>80 ? DispatchKind::Fp16Acc32N32DeviceM : DispatchKind::Static));
        // FP8 N112 candidate regressed against the real N64/K128 default.
        assert(select_dispatch(total,1024,256,5,true).kind==DispatchKind::Static);
        assert(select_dispatch(total,1024,256,1,true,3).kind==DispatchKind::Static);
        const auto identity=dispatch_identity(total,1024,256,1,true);
        assert(identity.back()==(total>48 ? 2 : 0));
        const auto contextual=select_dispatch(total,1024,256,1,true,2,0,0,32);
        assert((dispatch_identity(total,1024,256,1,true,32)==std::array<int64_t,5>{
            total,contextual.tile.m_tile,contextual.tile.n_tile,
            contextual.tile.k_tile,static_cast<int64_t>(contextual.kind)}));
    }
    // Paired FP8 Gate geometries select a static byte kernel with WDType5.
    // The expert count is required; ordinary callers with no context stay default.
    for(int total : {8,512,1024}) for(int n : {64,96}) {
        const int expected_n=n==96?48:64, expected_m=n==96?256:224;
        const auto d=select_dispatch(total,n,2048,5,false,2,0,0,256);
        assert(d.tile.n_tile==expected_n && d.tile.k_tile==256 && d.tile.m_tile==expected_m);
        assert(d.kind==DispatchKind::Static);
        assert((dispatch_identity(total,n,2048,5,false,256)==
                std::array<int64_t,5>{total,expected_m,expected_n,256,0}));
        rpu_pl_tiling::grouped_kernel_ordinal(
            rpu_pl_tiling::WeightType::Fp8E4M3,false,d.tile);
        const auto unknown=select_dispatch(total,n,2048,5,false);
        assert(unknown.tile.n_tile==64 && unknown.tile.k_tile==128 &&
               unknown.tile.m_tile==448 && unknown.kind==DispatchKind::Static);
    }
    struct Fp8Boundary {int m,e,n,k,mode,x,nt,kt;bool acc32;};
    for(const auto& c : {Fp8Boundary{7,256,64,2048,5,2,0,0,false},
                        Fp8Boundary{9,256,64,2048,5,2,0,0,false},
                        Fp8Boundary{511,256,64,2048,5,2,0,0,false},
                        Fp8Boundary{513,256,64,2048,5,2,0,0,false},
                        Fp8Boundary{1023,256,64,2048,5,2,0,0,false},
                        Fp8Boundary{1025,256,64,2048,5,2,0,0,false},
                        Fp8Boundary{8,255,64,2048,5,2,0,0,false},
                        Fp8Boundary{8,257,64,2048,5,2,0,0,false},
                        Fp8Boundary{8,256,64,2048,5,3,0,0,false},
                        Fp8Boundary{8,256,64,2048,5,2,0,0,true},
                        Fp8Boundary{8,256,64,2048,1,2,0,0,false},
                        Fp8Boundary{8,256,64,2048,2,2,0,0,false},
                        Fp8Boundary{8,256,80,2048,5,2,0,0,false},
                        Fp8Boundary{8,256,128,2048,5,2,0,0,false},
                        Fp8Boundary{8,256,64,2016,5,2,0,0,false},
                        Fp8Boundary{8,256,64,2048,5,2,64,128,false},
                        Fp8Boundary{8,256,64,2048,5,2,64,0,false},
                        Fp8Boundary{8,256,64,2048,5,2,32,256,false},
                        Fp8Boundary{8,256,2048,64,5,2,0,0,false},
                        Fp8Boundary{512,256,2048,64,5,2,0,0,false},
                        Fp8Boundary{1024,256,2048,64,5,2,0,0,false},
                        Fp8Boundary{512,256,256,512,5,2,0,0,false},
                        Fp8Boundary{7,256,96,2048,5,2,0,0,false},
                        Fp8Boundary{9,256,96,2048,5,2,0,0,false},
                        Fp8Boundary{511,256,96,2048,5,2,0,0,false},
                        Fp8Boundary{513,256,96,2048,5,2,0,0,false},
                        Fp8Boundary{1023,256,96,2048,5,2,0,0,false},
                        Fp8Boundary{1025,256,96,2048,5,2,0,0,false},
                        Fp8Boundary{8,255,96,2048,5,2,0,0,false},
                        Fp8Boundary{8,257,96,2048,5,2,0,0,false},
                        Fp8Boundary{8,256,96,2048,5,3,0,0,false},
                        Fp8Boundary{8,256,96,2048,5,2,0,0,true},
                        Fp8Boundary{8,256,96,2048,1,2,0,0,false},
                        Fp8Boundary{8,256,96,2048,2,2,0,0,false},
                        Fp8Boundary{8,256,112,2048,5,2,0,0,false},
                        Fp8Boundary{8,256,96,2016,5,2,0,0,false},
                        Fp8Boundary{8,256,96,2080,5,2,0,0,false},
                        Fp8Boundary{8,256,96,2048,5,2,64,128,false},
                        Fp8Boundary{8,256,96,2048,5,2,48,0,false},
                        Fp8Boundary{8,256,96,2048,5,2,48,256,false},
                        Fp8Boundary{8,256,2048,96,5,2,0,0,false},
                        Fp8Boundary{512,256,2048,96,5,2,0,0,false},
                        Fp8Boundary{1024,256,2048,96,5,2,0,0,false}}) {
        const auto unchanged=select_dispatch(c.m,c.n,c.k,c.mode,c.acc32,c.x,c.nt,c.kt);
        const auto contextual=select_dispatch(c.m,c.n,c.k,c.mode,c.acc32,c.x,c.nt,c.kt,c.e);
        assert(contextual.tile.n_tile==unchanged.tile.n_tile &&
               contextual.tile.m_tile==unchanged.tile.m_tile &&
               contextual.tile.k_tile==unchanged.tile.k_tile &&
               contextual.kind==unchanged.kind);
    }
    rejected([]{select_dispatch(8,64,2048,5,false,2,0,256,256);});
    rejected([]{select_dispatch(8,96,2048,5,false,2,0,256,256);});
    assert(!rpu_grouped_w8_device_m::eligible(256,1,true,80,384));
    assert(!rpu_grouped_w8_device_m::eligible(256,1,true,112,256));
    assert(!rpu_grouped_device_m::eligible(128,2,false,64,512));
    assert(!rpu_grouped_device_m::eligible(128,2,false,80,256));
    const auto packed=dispatch_identity(128,256,512,2,false);
    assert((packed==std::array<int64_t,5>{128,96,80,512,1}));
    // A descriptor based on source tokens, not top-k assignments, must differ.
    assert(packed!=dispatch_identity(16,256,512,2,false));
    assert(packed!=dispatch_identity(128,256,512,2,true));
    rejected([]{select_dispatch(0,128,512,2,false);});
    rejected([]{select_dispatch(-1,128,512,2,false);});
    rejected([]{select_dispatch(65536,128,512,2,false);});
    rejected([]{select_dispatch(128,128,512,6,false);});
    assert(rpu_grouped_device_m::kMaxGroups==32768);
    select_uint4(1,32768,32,128);
    select_uint4(65535,1,32,128);
    rejected([]{select_uint4(65536,1,32,128);});
    rejected([]{select_uint4(1,32769,32,128);});
    // Logical N16 is padded to the same installed tile as physical N32.
    const auto logical=select_uint4(1,1,16,128);
    const auto physical=select_uint4(1,1,32,128);
    assert(logical.n==physical.n && logical.m==physical.m && logical.k==physical.k);
    // Wide shape values must be rejected before narrowing to the kernel ABI.
    rejected([]{select_uint4(1,1,(INT64_C(1)<<32)+32,128);});
    rejected([]{select_uint4(1,1,32,(INT64_C(1)<<32)+128);});
    rejected([]{select_uint4(1,1,INT64_MAX,128);});
    rejected([]{select_uint4(1,1,1,128);});
    // Exercise the real public query, including its maximum admitted packed
    // N32 and G32 K boundaries; it must not truncate before the helper sees K.
    const auto query=rpu_select_grouped_parallel_linear_uint4_tile(1,1,16,128,32,false);
    assert((query==std::array<int64_t,3>{logical.n,logical.m,logical.k}));
    // Expected installed tiles come from paired TP4 T/U/W and TP8 D/E measurements.
    // TP4 Down M512/M1024 retain their defaults despite concentrated wins.
    struct Measured {int m,n,k,group,nt,mt,kt;};
    for(const auto& c : {Measured{8,128,2048,128,32,288,256},
                        Measured{512,128,2048,128,32,288,256},
                        Measured{1024,128,2048,128,32,288,256},
                        Measured{8,2048,128,64,128,288,128},
                        Measured{512,2048,128,64,64,448,128},
                        Measured{1024,2048,128,64,64,448,128},
                        Measured{8,64,2048,128,32,64,768},
                        Measured{512,64,2048,128,32,64,768},
                        Measured{1024,64,2048,128,32,64,768},
                        Measured{8,2048,64,64,128,128,256},
                        Measured{512,2048,64,64,128,128,256},
                        Measured{1024,2048,64,64,128,288,128}}) {
        const auto selected=rpu_select_grouped_parallel_linear_uint4_tile(
            c.m,256,c.n,c.k,c.group,false);
        assert((selected==std::array<int64_t,3>{c.nt,c.mt,c.kt}));
        const auto installed=rpu_pl_tiling::make_tile(
            rpu_pl_tiling::WeightType::Uint4,false,c.nt,c.kt,
            rpu_pl_tiling::Family::Grouped);
        assert(installed.m_tile==c.mt);
        rpu_pl_tiling::grouped_kernel_ordinal(
            rpu_pl_tiling::WeightType::Uint4,false,installed);
    }
    // Neighboring keys must retain the unmodified main selector. Logical N112
    // pads to128 but must not alias the measured logical N128 shard.
    struct Unmeasured {int m,experts,n,k,group;bool acc32;};
    for(const auto& c : {Unmeasured{7,256,128,2048,128,false},
                        Unmeasured{9,256,128,2048,128,false},
                        Unmeasured{511,256,128,2048,128,false},
                        Unmeasured{513,256,128,2048,128,false},
                        Unmeasured{1023,256,128,2048,128,false},
                        Unmeasured{1025,256,128,2048,128,false},
                        Unmeasured{8,255,128,2048,128,false},
                        Unmeasured{8,257,128,2048,128,false},
                        Unmeasured{8,256,112,2048,128,false},
                        Unmeasured{8,256,144,2048,128,false},
                        Unmeasured{8,256,128,1920,128,false},
                        Unmeasured{8,256,128,2048,64,false},
                        Unmeasured{8,256,128,2048,32,false},
                        Unmeasured{8,256,128,2048,128,true},
                        Unmeasured{512,256,2048,128,64,false},
                        Unmeasured{9,256,2048,128,64,false},
                        Unmeasured{8,255,2048,128,64,false},
                        Unmeasured{8,256,2032,128,64,false},
                        Unmeasured{8,256,2048,256,64,false},
                        Unmeasured{8,256,2048,128,32,false},
                        Unmeasured{8,256,2048,128,128,false},
                        Unmeasured{8,256,2048,128,64,true},
                        Unmeasured{7,256,64,2048,128,false},
                        Unmeasured{9,256,64,2048,128,false},
                        Unmeasured{511,256,64,2048,128,false},
                        Unmeasured{513,256,64,2048,128,false},
                        Unmeasured{1023,256,64,2048,128,false},
                        Unmeasured{1025,256,64,2048,128,false},
                        Unmeasured{8,255,64,2048,128,false},
                        Unmeasured{8,257,64,2048,128,false},
                        Unmeasured{8,256,48,2048,128,false},
                        Unmeasured{8,256,80,2048,128,false},
                        Unmeasured{8,256,64,1920,128,false},
                        Unmeasured{8,256,64,2048,64,false},
                        Unmeasured{8,256,64,2048,32,false},
                        Unmeasured{8,256,64,2048,128,true},
                        Unmeasured{9,256,2048,64,64,false},
                        Unmeasured{513,256,2048,64,64,false},
                        Unmeasured{1025,256,2048,64,64,false},
                        Unmeasured{8,255,2048,64,64,false},
                        Unmeasured{8,256,2032,64,64,false},
                        Unmeasured{8,256,2048,192,64,false},
                        Unmeasured{8,256,2048,64,32,false},
                        Unmeasured{8,256,2048,64,64,true}}) {
        const auto baseline=rpu_pl_tiling::select_tile(
            1,(c.n+31)/32*32,c.k,rpu_pl_tiling::WeightType::Uint4,c.acc32,
            rpu_pl_tiling::Family::Grouped,c.group);
        const auto selected=rpu_select_grouped_parallel_linear_uint4_tile(
            c.m,c.experts,c.n,c.k,c.group,c.acc32);
        assert((selected==std::array<int64_t,3>{baseline.n_tile,baseline.m_tile,baseline.k_tile}));
    }
    rpu_select_grouped_parallel_linear_uint4_tile(65535,32768,1048544,65504,32,false);
    rejected([]{rpu_select_grouped_parallel_linear_uint4_tile(1,1,1048560,128,32,false);});
    rejected([]{rpu_select_grouped_parallel_linear_uint4_tile(1,1,32,65536,32,false);});
    rejected([]{rpu_select_grouped_parallel_linear_uint4_tile(1,1,(INT64_C(1)<<32)+32,128,32,false);});
    rejected([]{rpu_select_grouped_parallel_linear_uint4_tile(1,1,32,(INT64_C(1)<<32)+128,32,false);});
    rejected([]{rpu_select_grouped_parallel_linear_uint4_tile(1,1,32,128,(INT64_C(1)<<32)+32,false);});
}
'''.replace("__PUBLIC_QUERY__", public_query))
    exe = tmp_path / "dispatch"
    subprocess.run(["c++", "-std=c++17", "-O2", "-I", str(ROOT / "src/core"),
                    "-I", str(ROOT / "src/ops"), str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_extra_device_policies_preserve_static_boundaries_and_forcing(tmp_path):
    cpp = tmp_path / "extra_dispatch.cpp"
    cpp.write_text(r'''
#include "rpu_grouped_linear_tiling.h"
#include <cassert>
using namespace rpu_grouped_tiling;
struct Case { int n,k,mode,acc32,threshold,mt,nt,kt,kind; };
int main() {
    const Case cases[] = {
        {256,512,2,1,80,80,80,512,3},
        {1024,256,2,1,80,80,32,768,4},
        {256,512,1,0,48,48,48,768,5},
        {256,512,1,1,96,96,48,512,6},
        {1024,256,5,1,48,48,112,384,0},
        {256,512,5,0,48,48,48,768,5},
        {256,512,5,1,96,96,48,512,6},
    };
    for (const auto& c : cases) {
        for (int total : {1,c.threshold-1,c.threshold,c.threshold+1,256,65535}) {
            const auto d=select_dispatch(total,c.n,c.k,c.mode,c.acc32);
            const bool fp8_static=c.mode==5 && (c.kind==0 || total<=c.threshold);
            const auto expected=fp8_static ? rpu_pl_tiling::make_tile(
                rpu_pl_tiling::WeightType::Fp8E4M3,c.acc32,64,128,
                rpu_pl_tiling::Family::Grouped) : rpu_pl_tiling::TilePick{c.nt,c.mt,c.kt};
            assert(d.tile.m_tile==expected.m_tile && d.tile.n_tile==expected.n_tile &&
                   d.tile.k_tile==expected.k_tile);
            assert(static_cast<int>(d.kind)==(total>c.threshold ? c.kind : 0));
            const auto id=dispatch_identity(total,c.n,c.k,c.mode,c.acc32);
            assert((id==std::array<int64_t,5>{total,expected.m_tile,expected.n_tile,expected.k_tile,
                                             total>c.threshold ? c.kind : 0}));
            assert(select_dispatch(total,c.n,c.k,c.mode,c.acc32,2,c.nt,c.kt).kind==DispatchKind::Static);
            assert(select_dispatch(total,c.n,c.k,c.mode,c.acc32,2,c.nt,0).kind==DispatchKind::Static);
            assert(rpu_grouped_extra_device_m::select_kind(
                total,c.mode,c.acc32,c.nt,c.kt,3,false)==0);
            assert(rpu_grouped_extra_device_m::select_kind(
                total,c.mode,c.acc32,c.nt,c.kt,2,true)==0);
        }
    }
    // Appending policy identities must not renumber previously built recipes.
    static_assert(static_cast<int>(DispatchKind::Static)==0);
    static_assert(static_cast<int>(DispatchKind::Fp16Acc16DeviceM)==1);
    static_assert(static_cast<int>(DispatchKind::W8Acc32DeviceM)==2);
}
''')
    exe = tmp_path / "extra_dispatch"
    subprocess.run(["c++", "-std=c++17", "-O2", "-I", str(ROOT / "src/core"),
                    "-I", str(ROOT / "src/ops"), str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_python_packing_rejects_expert_index_wrap_before_transform():
    import torch
    from rpu_backend.quant.fp8 import (
        MAX_GROUPED_EXPERTS,_validate_grouped_experts,pack_grouped_weight,pack_grouped_scale)
    from rpu_backend.quant.grouped_uint4 import pack_grouped_uint4,pack_grouped_uint4_metadata
    import pytest
    assert MAX_GROUPED_EXPERTS==32768
    _validate_grouped_experts(32768)
    assert ((32768-1)//16*16)*2+31==65535
    with pytest.raises(ValueError,match='32768'):_validate_grouped_experts(32769)
    # Metadata owners are inexpensive enough to exercise the admitted upper
    # boundary; rejected weights use tiny dimensions and never enter swizzle.
    assert pack_grouped_scale(torch.zeros(32768,1,dtype=torch.float16)).shape==(32768,128)
    for invoke in (
        lambda:pack_grouped_weight(torch.zeros(32769,1,1,dtype=torch.float16),partition='col',num_cores=1),
        lambda:pack_grouped_scale(torch.zeros(32769,1,dtype=torch.float16)),
        lambda:pack_grouped_uint4(torch.zeros(32769,1,1,dtype=torch.uint8),partition='col',num_cores=1),
        lambda:pack_grouped_uint4_metadata(torch.zeros(32769,1,1,dtype=torch.float16),partition='col',num_cores=1),
    ):
        with pytest.raises(ValueError,match='32768'):invoke()


def test_actual_callers_use_assignment_rows_and_keep_gp9_reserved():
    qwen = (ROOT / "src/fused/rpu_qwen3_5_moe_model.cpp").read_text()
    lingbot = (ROOT / "src/fused/rpu_lingbot_v2_moe_model.cpp").read_text()
    native = (ROOT / "src/ops/rpu_grouped_parallel_linear.cpp").read_text()
    assert "num_cores, m_total, weight_mode, linear_acc32_" in qwen
    from test_fmb_standalone_layout_rebuild import _definition
    # All three producers use one selector-derived descriptor. The executable
    # MLP fixture also mutates every dispatch/ABI field before the live launch.
    helper = _definition(qwen, "std::vector<int64_t> qwen35_grouped_route_arguments(")
    launch = _definition(qwen, "void Qwen3_5MoeModel::launch_grouped_linear(")
    cold = _definition(qwen, "void Qwen3_5MoeModel::append_mlp_manifest_routes(")
    assert "m_total, n, k, weight_mode, acc32, num_experts" in helper
    assert "m_total, n, k, num_experts, is_col_parallel, num_cores" in launch
    assert "t * moe_.top_k, local_i, hidden_size(), moe_.num_experts" in cold
    assert "t * moe_.top_k, hidden_size(), local_i, moe_.num_experts" in cold
    assert launch.count("qwen35_grouped_route_arguments(") == 1
    assert cold.count("qwen35_grouped_route_arguments(") == 2
    assert "total_m, local_n, local_k, static_cast<int>(mode), linear_acc32,\n        2, 0, 0, group" in native
    assert "r = seq_len * k" in lingbot
    assert "num_cores(), r, sparse_weight_mode_, false" in lingbot
    assert "logical_len * top_k_, col ? 2 * local_i : hidden_size()" in lingbot
    assert "r, 2 * local_i, h, sparse_weight_mode_, false" in lingbot
    assert "r, h, local_i, sparse_weight_mode_, false" in lingbot
    writer = native.split("rpu_pl_tiling::set_parallel_linear_regs(*kernel, {", 1)[1].split("});", 1)[0]
    assert "bias_spm_addr,\n        0, static_cast<uint32_t>(local_n)" in writer
    assert "total_m" not in writer  # New kernel reads per-expert m_sizes, never GP9.
    assert '"grouped_parallel_linear_m96n80k512_device_m", RHINO_EXPANSION_OP_LIB_PATH' in (
        ROOT / "src/core/rpu_kernel_cache.inc").read_text()


def test_lingbot_actual_argument_blocks_match_cold_manifest(tmp_path):
    source=(ROOT/'src/fused/rpu_lingbot_v2_moe_model.cpp').read_text()
    def block(start, stop):
        begin=source.index(start)
        return source[begin:source.index(stop,begin)]
    gate=block('std::vector<int64_t> arguments{2 * local_i, h, e, 1, num_cores(), sparse_weight_mode_};',
               'ctx().consume_physical_route')
    down=block('std::vector<int64_t> arguments{h, local_i, e, 0, num_cores(), sparse_weight_mode_};',
               'ctx().consume_physical_route')
    planner=block('std::vector<int64_t> arguments{\n                col ? 2 * local_i : hidden_size()',
                  'append(FmbRouteFamily::LINEAR, down_scale_shuffle ?')
    cpp=tmp_path/'lingbot.cpp'
    cpp.write_text(r'''
#include "rpu_grouped_linear_tiling.h"
#include <cassert>
#include <vector>
int64_t h=512,local_i=128,e=32,num_experts_=32,top_k_=4,tp_rows_=64;
constexpr int NUM_CORES=8;
constexpr int num_cores(){return NUM_CORES;}
int sparse_weight_mode_=2;
bool sparse_gate_up_expert_y4_=false;
int64_t hidden_size(){return h;}
int64_t Align(int64_t n,int64_t a){return (n+a-1)/a*a;}
std::vector<int64_t> emit(bool col,int64_t seq_len){
    const int64_t r=seq_len*top_k_;
    if(col){
''' + gate + r'''
        return arguments;
    }else{
''' + down + r'''
        return arguments;
    }
}
std::vector<int64_t> plan(bool col,int64_t logical_len){
    const bool routed=false,fused_activation=false,down_scale_shuffle=false;
''' + planner + r'''
    return arguments;
}
int main(){
    for(int mode:{1,2})for(int rows:{1,24,25,51})for(bool col:{false,true}){
        sparse_weight_mode_=mode;
        const auto expected=plan(col,rows),actual=emit(col,rows);
        assert(expected==actual && actual.size()==11 && actual[6]==rows*4);
        auto corrupted=expected;corrupted[6]=rows;
        assert(corrupted!=actual);
        corrupted=expected;corrupted.back()^=1;assert(corrupted!=actual);
        if(col)assert(actual.back()==(mode==2 ? (rows*4>96 ? 1 : 0) : (rows*4>48 ? 5 : 0)));
    }
    sparse_gate_up_expert_y4_=true;
    assert(plan(true,51)==emit(true,51) && emit(true,51).size()==6);
}
''')
    exe=tmp_path/'lingbot'
    subprocess.run(['c++','-std=c++17','-I',str(ROOT/'src/core'),'-I',str(ROOT/'src/ops'),
                    str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
