"""Exercise native grouped admission with the production selector and GP writer."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("routed", [False, True])
def test_native_lingbot_expert_y4_input_and_public_entry(monkeypatch, tmp_path, routed):
    """The ordinary expert path consumes 204 packed rows, not routed token IDs."""
    import ast
    import importlib.util
    import sys
    import torch

    root = Path(__file__).resolve().parents[1]
    packing = root / "python/rpu_backend/quant/fp8.py"
    spec = importlib.util.spec_from_file_location("rpu_backend.quant.fp8", packing)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "rpu_backend.quant.fp8", module)
    harness = root / "python/rpu_backend/tests/conformance/production_linear_launchers.py"
    nodes = [n for n in ast.parse(harness.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in
             {"raw", "activation", "grouped_fixture"}]
    namespace = {"torch": torch}
    exec(compile(ast.Module(nodes, []), str(harness), "exec"), namespace)
    original_to = torch.Tensor.to

    def cpu_placement(tensor, *args, **kwargs):
        if args and args[0] == "rpu":
            assert len(args) == 1 and not kwargs
            return tensor
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_placement)
    case = "lingbot_grouped_w8" if routed else "lingbot_grouped_expert_y4_w8"
    profile, rows, cores, mode, owners, images, outputs, expected, linear = \
        namespace["grouped_fixture"](case)
    assert (rows, cores, mode, linear) == (51, 8, 1, True)
    assert profile == ("lingbot_grouped" if routed else "lingbot_grouped_expert_y4")
    assert outputs == ([3] if routed else [2])
    # Decode the actual grouped packed weights, independently of the fixture's
    # logical construction, and use the count prefix to select each row block.
    logical = owners[0].view(32, 8, 24, 8, 16, 32) \
        .permute(0, 3, 1, 4, 2, 5).reshape(32, 8, 128, 768)
    scales = owners[1].reshape(32, 8, 128)
    active = [0, 7, 16, 31]
    for arm, image in enumerate(images):
        x = image[0].view(torch.float16).reshape(8, 51 if routed else 204, 768)
        counts = image[2 if routed else 1].view(torch.int16).reshape(8, 128)
        assert counts.sum(1).tolist() == [204] * 8
        assert counts[0].nonzero().flatten().tolist() == active
        tokens = image[1].view(torch.int16).reshape(8, 256) if routed else None
        assert len(image) == (4 if routed else 3)
        for a, expert in enumerate(active):
            weights = logical[expert]
            assert torch.equal(weights.ne(0).sum(-1), torch.ones(8, 128, dtype=torch.int64))
            indices = weights.ne(0).to(torch.int64).argmax(-1)
            values = weights.gather(-1, indices[..., None]).squeeze(-1).float()
            selected = torch.stack([x[c, tokens[c, a*51:(a+1)*51].long()]
                                    for c in range(8)]) if routed else x[:, a*51:(a+1)*51]
            reference = (selected.gather(2, indices[:, None, :].expand(-1, 51, -1)).float()
                         * values[:, None, :] * scales[expert, :, None, :].float()).half()
            assert torch.equal(reference, expected[arm][0][:, a*51:(a+1)*51])
        assert not image[outputs[0]].any()
    assert not torch.equal(expected[0][0], expected[1][0])
    if routed:
        return
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    cpp = (root / "python/rpu_backend/tests/conformance/_cpp/test_production_linear_launchers.cpp").read_text()
    body = cpp.split('} else if (profile == "lingbot_grouped_expert_y4") {', 1)[1] \
        .split('} else if (profile == "lingbot_grouped") {', 1)[0]
    # Execute the real harness branch and inspect the actual public call's
    # argument order; the production function is covered by the native case.
    program = r'''
#include <cassert>
#include <cstdint>
#include <optional>
namespace at { struct Tensor { int tag; bool defined() const { return true; } }; }
namespace c10 { template<class T> using optional=std::optional<T>; constexpr auto nullopt=std::nullopt; }
bool called=false;
void rpu_launch_grouped_parallel_linear_spm_kernel(uint32_t x, at::Tensor w,
    uint32_t counts, c10::optional<at::Tensor> scale, uint32_t y, uint32_t bias,
    int64_t n, int64_t k, int64_t groups, bool col, int cores, int64_t total,
    int64_t mode, bool acc32, bool expert) {
  assert(x==100 && w.tag==0 && counts==101 && scale->tag==1 && y==102 && bias==0);
  assert(n==128 && k==768 && groups==32 && col && cores==8 && total==204);
  assert(mode==1 && !acc32 && expert); called=true;
}
int main() {
  auto s=[](int i){return uint32_t(100+i);}; auto w=[](int i){return at::Tensor{i};};
  int mode=1;
''' + body + '\nassert(called);\n}\n'
    source = tmp_path / "expert_y4_entry.cpp"
    source.write_text(program)
    binary = tmp_path / "expert_y4_entry"
    subprocess.run([compiler, "-std=c++17", str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_routed_col_uses_precision_grid_and_current_registers(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/ops/rpu_grouped_parallel_linear.cpp").read_text()
    source = source[source.index("void rpu_launch_lingbot2_routed_col_spm_kernel("):]
    body = source[source.index("    kernel->reset_regs();"):source.index("\n}")]
    program = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>
struct Kernel {
  std::array<uint16_t,128> regs{};
  void reset_regs(){regs.fill(0);}
  void set_regs(int i,uint16_t value){regs.at(i)=value;}
};
void set_reg_u32(Kernel* k,int i,uint32_t value){k->set_regs(i,value);k->set_regs(i+1,value>>16);}
struct Queue {
  bool broadcast=false; std::vector<uint16_t> grid;
  void set_broadcast_mode(bool value){broadcast=value;}
  void enqueu_kernel(Kernel&,std::vector<uint16_t> value,std::vector<int> cores){
    assert(cores==std::vector<int>({0,1,2,3,4,5,6,7}));grid=value;
  }
} queue_storage;
Queue* GET_QUEUE(int n){assert(n==8);return &queue_storage;}
struct Writer {void write(Kernel& k){k.set_regs(2,0x1234);}};
void run(bool w8) {
  constexpr int rows=51,experts=32,local_k=768,output_n=2048,num_cores=8;
  uint32_t input_spm_addr=0x100000,output_spm_addr=0x200000,counts_spm_addr=0x300000;
  Kernel storage;auto* kernel=&storage;Writer register_writer;
''' + body + r'''
  assert(storage.regs[8]==1 && storage.regs[9]==51);
  assert(storage.regs[10]==2048 && storage.regs[12]==768 && storage.regs[13]==48);
  assert(storage.regs[26]==2 && storage.regs[27]==(w8?1:2));
  assert(storage.regs[48]==0 && storage.regs[49]==0x30);
  assert(storage.regs[64]==(w8?2:1) && storage.regs[65]==1 && storage.regs[66]==32);
  assert(queue_storage.broadcast && queue_storage.grid==std::vector<uint16_t>({uint16_t(w8?2:1),1,32}));
}
int main(){run(false);run(true);}
'''
    unit = tmp_path / "routed_col_grid.cpp"
    unit.write_text(program)
    binary = tmp_path / "routed_col_grid"
    subprocess.run([compiler, "-std=c++17", str(unit), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_current_grouped_admits_u32_rows_and_rejects_overflow(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/ops/rpu_grouped_parallel_linear.cpp").read_text()
    start = source.index("    TORCH_CHECK(num_cores >= 1",
                         source.index("void launch_grouped_parallel_linear_spm("))
    end = source.index("    TORCH_CHECK(packed_weight.defined()", start)
    geometry = source[start:end]
    start = source.index("    const uint64_t weight_group_bytes =", end)
    end = source.index("    TORCH_CHECK(static_cast<uint64_t>(packed_weight.nbytes())", start)
    extent = source[start:end]
    def assignment_rows_guard(variable):
        start = source.index(f"    TORCH_CHECK({variable} > 0 && {variable} <= UINT16_MAX,")
        end = source.index(";", start) + 1
        return source[start:end]
    program = r"""
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error(#__VA_ARGS__); } while (0)
#include "rpu_linear_tiling.h"
namespace c10 { struct Half { uint16_t bits; }; }
void geometry(int64_t local_n, int64_t local_k, int64_t group, int num_cores) {
""" + geometry + r"""
}
uint64_t extent(int64_t local_n, int64_t local_k, int64_t group,
                int num_cores, uint64_t weight_dwidth) {
""" + extent + r"""
    return expected_weight_bytes;
}
struct Kernel {
    std::array<uint16_t,64> regs{};
    void set_regs(int i, uint16_t value) { regs.at(i) = value; }
    uint32_t pair(int i) const { return uint32_t(regs[i]) | (uint32_t(regs[i+1]) << 16); }
};
void uint4_assignment_rows(int64_t m_total) {
""" + assignment_rows_guard("m_total") + r"""
}
void eager_assignment_rows(int64_t m) {
""" + assignment_rows_guard("m") + r"""
}
template<class F> void rejects(F function) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
int main() {
    uint4_assignment_rows(65535);
    eager_assignment_rows(65535);
    for (int64_t m : {int64_t{-1},int64_t{0},int64_t{65536},INT64_MAX}) {
        rejects([&] { uint4_assignment_rows(m); });
        rejects([&] { eager_assignment_rows(m); });
    }
    for (const auto nk : {std::pair<int,int>{32768,32}, {16,32768}, {65536,32}}) {
        geometry(nk.first, nk.second, 1, 1);
        const auto tile = rpu_pl_tiling::select_tile(1,nk.first,nk.second,
            rpu_pl_tiling::WeightType::Fp16,false,rpu_pl_tiling::Family::Grouped);
        assert(rpu_pl_tiling::grouped_kernel_ordinal(
            rpu_pl_tiling::WeightType::Fp16,false,tile) >= 0);
        assert(extent(nk.first,nk.second,1,1,2) ==
               uint64_t(nk.first)*nk.second*2);
        Kernel kernel;
        rpu_pl_tiling::set_parallel_linear_regs(kernel,{
            0x100000,0,0x200000,0,0,uint32_t(nk.first),uint32_t(nk.second),
            0,1,uint32_t(nk.second)*2,uint32_t(nk.first)*2,0,1,2});
        assert(kernel.pair(10) == uint32_t(nk.first));
        assert(kernel.pair(16) == uint32_t(nk.second)*2);
        assert(kernel.pair(18) == uint32_t(nk.first)*2);
    }
    rejects([] { geometry(16,65536,1,1); });
    rejects([] { geometry(16,31,1,1); });
    rejects([] { geometry(17,32,1,1); });
    rejects([] { geometry(INT64_C(2147483648),32,1,1); });
    rejects([] { geometry(16,32,65536,1); });
    geometry(16,32,32768,1);
    rejects([] { geometry(16,32,32769,1); });
    rejects([] { geometry(16,32,1,9); });
    // N and K individually fit; their expert stride cannot fit GP34/35.
    geometry(INT64_C(2147483632),65520,32768,8);
    rejects([] { extent(INT64_C(2147483632),65520,32768,8,2); });
    // A nonaligned expert must fail before its 256-byte unit conversion.
    rejects([] { extent(1,1,1,1,2); });
}
"""
    unit = tmp_path / "grouped_stride.cpp"
    unit.write_text(program)
    binary = tmp_path / "grouped_stride"
    subprocess.run([compiler, "-std=c++17", "-I", str(root / "src/ops"),
                    str(unit), "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
