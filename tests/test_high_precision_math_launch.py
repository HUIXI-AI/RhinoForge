"""Execute the shared HIGH/Newton launchers against a board-free queue ABI."""

from pathlib import Path
import shutil
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    return (ROOT / path).read_text()


def _block(source, marker):
    start = source.index(marker)
    # Ignore braces inside comments and literals while preserving source offsets.
    clean = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/',
                   lambda match: " " * len(match[0]), source, flags=re.S)
    brace = clean.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (clean[end] == "{") - (clean[end] == "}")
        end += 1
    return source[start:end]



def _run(tmp_path, name, program):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    source, binary = tmp_path / f"{name}.cpp", tmp_path / name
    source.write_text(program)
    built = subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(ROOT / "src/core"), "-I", str(ROOT / "src/ops"),
                            str(source), "-o", str(binary)],
                           capture_output=True, text=True, timeout=30)
    assert built.returncode == 0, built.stderr
    subprocess.run([str(binary)], check=True, timeout=30)


COMMON = r'''
#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>
#include "rpu_linear_tiling.h"
#include "rpu_pi_wall_tiling.h"
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
namespace c10 { struct Half { uint16_t x; Half(double v = 0): x(uint16_t(v * 1024)) {} }; }
uint16_t half_to_u16(c10::Half h) { return h.x; }
size_t CeilDiv(size_t a, size_t b) { return (a + b - 1) / b; }
template<class F> void rejects(F fn) {
  bool caught = false; try { fn(); } catch (const std::runtime_error&) { caught = true; }
  assert(caught);
}
'''


def test_unary_high_launch_abi_formula_and_cold_ddr_staging(tmp_path):
    declarations = _read("src/core/rpu_kernel_decls.h")
    source = _read("src/ops/rpu_eltwise_unary.cpp")
    unary_decls = declarations[declarations.index("enum class GeluMode"):
                               declarations.index("// Sum over rows")]
    kernel_ids = _block(_read("src/core/rpu_kernel_cache.h"), "enum class KernelId") + ";\n"
    harness = r'''
enum class ValuOpType { ADD, HARDSIGMOID, SOFTPLUS, SILU, TANH, CLIP };
constexpr int FP16_RAB_DATA_TYPE_CFG = 5;
struct Info { int op = 1, custom_op = 2; };
Info get_valu_op_info(ValuOpType) { return {}; }
namespace at {
constexpr int kPrivateUse1 = 1, kHalf = 2;
struct Device {
  int value = kPrivateUse1;
  int type() const { return value; }
  bool operator==(Device b) const { return value == b.value; }
};
struct Tensor {
  c10::Half* ptr;
  std::vector<int64_t> shape;
  Device dev;
  int dtype = kHalf;
  bool contiguous = true;
  const std::vector<int64_t>& sizes() const { return shape; }
  int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
  int scalar_type() const { return dtype; }
  Device device() const { return dev; }
  bool is_contiguous() const { return contiguous; }
  template<class T> T* data_ptr() const { return reinterpret_cast<T*>(ptr); }
};
}
namespace graph_dma { bool captured = false; bool active() { return captured; } }
struct SpmAllocator {
  size_t mark = 96, allocated = 0;
  bool initialized = false;
  static SpmAllocator& instance() { static SpmAllocator s; return s; }
  bool is_initialized() const { return initialized; }
  void init() { initialized = true; }
  uint32_t alloc_temporary(size_t n) { allocated = n; mark += n; return 0; }
  uint32_t addr(int c, uint32_t off) { assert(c == 0 && off == 0); return 0x40001000; }
  struct ScopedTemporary {
    size_t saved = instance().mark;
    ~ScopedTemporary() { instance().mark = saved; }
  };
};
#define SPM_ALLOC SpmAllocator::instance()
std::vector<c10::Half> slab;
bool staged = false;
int dma_in = 0, dma_out = 0, device_address_queries = 0;
struct ScopedEagerAtomicBatch {
  ScopedEagerAtomicBatch(bool, int) {}
  size_t recorded_node_count() const noexcept { return 0; }
  static size_t default_flush_node_threshold() noexcept { return 45000; }
  void flush_and_reopen() {}
  void submit() {}
};
void rpu_launch_ddr_broadcast_spm_dma_immediate(c10::Half* x, int64_t n, uint32_t addr, int cores) {
  assert(addr == 0x40001000 && cores == 1 && n * 2 <= int64_t(SPM_ALLOC.allocated));
  assert(n % 8 == 0 && !staged);
  slab.assign(x, x + n); staged = true; ++dma_in;
}
void rpu_launch_spm_copy_ddr_dma_immediate(uint32_t addr, c10::Half* out, int64_t n) {
  assert(addr == 0x40001000 && staged && n == int64_t(slab.size()));
  std::copy(slab.begin(), slab.end(), out); staged = false; ++dma_out;
}
void rpu_launch_ddr_broadcast_spm_dma(c10::Half* x, int64_t n, uint32_t addr, int cores) {
  rpu_launch_ddr_broadcast_spm_dma_immediate(x, n, addr, cores);
}
void rpu_launch_spm_copy_ddr_dma(uint32_t addr, c10::Half* out, int64_t n) {
  rpu_launch_spm_copy_ddr_dma_immediate(addr, out, n);
}
void rpu_ddr_flush(void*) {}
uint64_t RpuGetDevAddr(void*) { ++device_address_queries; return 0x1200000000ull; }
struct Kernel_t {
  std::array<uint16_t, 64> regs{};
  void reset_regs() { regs.fill(0); }
  void set_regs(int r, uint16_t value) { regs.at(r) = value; }
} kernel;
KernelId selected, missing = KernelId::_COUNT;
int lookups = 0;
Kernel_t* GET_KERNEL(KernelId id) { ++lookups; selected = id; return id == missing ? nullptr : &kernel; }
struct Queue {
  bool broadcast = false, fail = false;
  int cores = 0, launches = 0;
  std::array<uint16_t, 3> grid{};
  void set_broadcast_mode(bool b) { broadcast = b; }
  void enqueu_kernel(Kernel_t&, std::array<uint16_t,3> g, std::vector<uint8_t> ids) {
    if (fail) { staged = false; throw std::runtime_error("queue rejected"); }
    assert(ids.size() == size_t(cores) && (cores == 1 || broadcast));
    for (int i = 0; i < cores; ++i) assert(ids[i] == i);
    grid = g; ++launches;
    if (staged) {
      assert(cores == 1 && broadcast && kernel.regs[61] == 1);
      assert(g[0] == 1 && kernel.regs[1] == slab.size());
      // A queue sentinel verifies DMA lifetime/data movement; this is not a
      // numerical implementation or a replacement for hardware validation.
      for (auto& x : slab) ++x.x;
    }
  }
} queue;
Queue* GET_QUEUE(int cores) { queue.cores = cores; return &queue; }
'''
    checks = r'''
int main() {
  static_assert(int(RpuUnaryPrecision::BASE) == 0 && int(RpuUnaryPrecision::HIGH) == 1);
  for (int family = 0; family < 4; ++family) {
    const auto op = family == 0 ? ValuOpType::SILU : family == 1 ? ValuOpType::TANH : ValuOpType::ADD;
    const auto mode = family == 2 ? GeluMode::ERF : family == 3 ? GeluMode::TANH : GeluMode::NONE;
    const auto id = family == 0 ? KernelId::UNARY_SILU_HIGH_PRECISION : family == 1
        ? KernelId::UNARY_TANH_HIGH_PRECISION : family == 2
        ? KernelId::UNARY_GELU_ERF_HIGH_PRECISION : KernelId::UNARY_GELU_TANH_HIGH_PRECISION;
    const int64_t block = family == 0 ? 15872 : family == 1 ? 25600 : family == 2 ? 15360 : 20480;
    for (int cores : {1, 4, 6, 8}) for (int64_t n : {int64_t(1), block-1, block, block+1, 2*block+13}) {
      rpu_launch_eltwise_unary_spm_kernel(0x40001000, 0x40001000, n, op, mode, cores, RpuUnaryPrecision::HIGH);
      assert(selected == id && kernel.regs[0] == block);
      assert(kernel.regs[1] == n - (CeilDiv(n, block)-1)*block);
      assert(kernel.regs[8] + (uint32_t(kernel.regs[9]) << 16) == block*2);
      assert(kernel.regs[4] + (uint32_t(kernel.regs[5]) << 16) == 0x40001000);
      assert(kernel.regs[6] == kernel.regs[4] && kernel.regs[7] == kernel.regs[5]);
      assert(kernel.regs[61] == 1 && queue.grid[0] == CeilDiv(n, block));
    }
    missing = id;
    const int before = queue.launches;
    rejects([&] { rpu_launch_eltwise_unary_spm_kernel(1, 1, 8, op, mode, 8, RpuUnaryPrecision::HIGH); });
    assert(queue.launches == before); missing = KernelId::_COUNT;
  }
  rpu_launch_eltwise_unary_spm_kernel(1, 1, 8, ValuOpType::TANH);
  assert(selected == KernelId::UNARY_SPM);
  rpu_launch_eltwise_unary_spm_kernel(1, 1, 8, ValuOpType::ADD, true);
  assert(selected == KernelId::UNARY_GELU_TANH_SPM);
  rpu_launch_eltwise_unary_spm_kernel(1, 1, 8, ValuOpType::SILU, false, 8, RpuUnaryPrecision::HIGH);
  assert(selected == KernelId::UNARY_SILU_HIGH_PRECISION);
  rpu_launch_eltwise_unary_spm_kernel(1, 1, 8, ValuOpType::ADD, true, 8, RpuUnaryPrecision::HIGH);
  assert(selected == KernelId::UNARY_GELU_TANH_HIGH_PRECISION);
  for (auto mode : {GeluMode::ERF_ULTRA, GeluMode(77)}) {
    rejects([&] { rpu_launch_eltwise_unary_spm_kernel(1,1,8,ValuOpType::ADD,mode,8,RpuUnaryPrecision::HIGH); });
  }
  rejects([] { rpu_launch_eltwise_unary_spm_kernel(1,1,8,ValuOpType::ADD,GeluMode::NONE,8,RpuUnaryPrecision::HIGH); });
  rejects([] { rpu_launch_eltwise_unary_spm_kernel(1,1,8,ValuOpType::SILU,GeluMode::ERF,8,RpuUnaryPrecision::HIGH); });
  rejects([] { rpu_launch_eltwise_unary_spm_kernel(1,1,8,ValuOpType::TANH,GeluMode::TANH,8,RpuUnaryPrecision::HIGH); });
  rejects([] { rpu_launch_eltwise_unary_spm_kernel(1,1,8,ValuOpType::SILU,GeluMode::NONE,8,RpuUnaryPrecision(77)); });
  rejects([] { rpu_launch_eltwise_unary_spm_kernel(1,1,0,ValuOpType::SILU); });
  rejects([] { rpu_launch_eltwise_unary_spm_kernel(1,1,8,ValuOpType::SILU,GeluMode::NONE,0); });
  rejects([] { rpu_launch_eltwise_unary_spm_kernel(1,1,int64_t(25600)*65536,ValuOpType::TANH); });

  rpu_launch_eltwise_unary_spm_kernel(1,1,8,ValuOpType::CLIP,GeluMode::NONE,8,RpuUnaryPrecision::BASE,1.0,2.0);
  assert(selected == KernelId::UNARY_SPM);
  assert(kernel.regs[59] == c10::Half(2.0).x && kernel.regs[60] == c10::Half(1.0).x);
  const int before_clip_rejections = queue.launches;
  rejects([] { rpu_launch_eltwise_unary_spm_kernel(1,1,8,ValuOpType::CLIP,GeluMode::NONE,8,RpuUnaryPrecision::BASE,2.0,1.0); });
  rejects([] { rpu_launch_eltwise_unary_spm_kernel(1,1,8,ValuOpType::CLIP,GeluMode::NONE,8,RpuUnaryPrecision::BASE,-70000.0,1.0); });
  rejects([] { rpu_launch_eltwise_unary_spm_kernel(1,1,8,ValuOpType::CLIP,GeluMode::NONE,8,RpuUnaryPrecision::BASE,0.0,INFINITY); });
  rejects([] { rpu_launch_eltwise_unary_spm_kernel(1,1,8,ValuOpType::CLIP,GeluMode::NONE,8,RpuUnaryPrecision::BASE,NAN,1.0); });
  rejects([] { rpu_launch_eltwise_unary_spm_kernel(1,1,8,ValuOpType::SILU,GeluMode::NONE,8,RpuUnaryPrecision::BASE,1.0,2.0); });
  assert(queue.launches == before_clip_rejections);

  for (auto op : {ValuOpType::SILU, ValuOpType::TANH}) {
    const int64_t block = op == ValuOpType::SILU ? 15872 : 25600;
    const int64_t count = block*3 + 8;
    std::vector<c10::Half> pixels(count), result(count);
    at::Tensor input{pixels.data(), {count}}, output{result.data(), {count}};
    assert(reinterpret_cast<uintptr_t>(pixels.data()) > UINT32_MAX);
    const size_t mark = SPM_ALLOC.mark;
    for (uint16_t sentinel : {uint16_t(23), uint16_t(79), uint16_t(23)}) {
      for (auto& x : pixels) x.x = sentinel;
      dma_in = dma_out = 0;
      rpu_launch_eltwise_unary_kernel(input, output, op, RpuUnaryPrecision::HIGH);
      assert(dma_in == 4 && dma_out == 4 && !staged && SPM_ALLOC.mark == mark);
      assert(SPM_ALLOC.allocated == size_t(block*2));
      for (const auto x : result) assert(x.x == sentinel+1);
    }
    rpu_launch_eltwise_unary_kernel(input, input, op, RpuUnaryPrecision::HIGH);
    for (const auto x : pixels) assert(x.x == 24);
    graph_dma::captured = true;
    const int before = lookups;
    rejects([&] { rpu_launch_eltwise_unary_kernel(input, output, op, RpuUnaryPrecision::HIGH); });
    assert(lookups == before && SPM_ALLOC.mark == mark);
    graph_dma::captured = false;
    queue.fail = true;
    rejects([&] { rpu_launch_eltwise_unary_kernel(input, output, op, RpuUnaryPrecision::HIGH); });
    assert(SPM_ALLOC.mark == mark); queue.fail = false;
    output.contiguous = false;
    rejects([&] { rpu_launch_eltwise_unary_kernel(input, output, op, RpuUnaryPrecision::HIGH); });
    output.contiguous = true;
    output.dev.value = 0;
    rejects([&] { rpu_launch_eltwise_unary_kernel(input, output, op, RpuUnaryPrecision::HIGH); });
    output.dev.value = at::kPrivateUse1;
    output.ptr = input.ptr+8;
    rejects([&] { rpu_launch_eltwise_unary_kernel(input, output, op, RpuUnaryPrecision::HIGH); });
    output.ptr = result.data();
    input.shape = output.shape = {7};
    rejects([&] { rpu_launch_eltwise_unary_kernel(input, output, op, RpuUnaryPrecision::HIGH); });
    input.shape = output.shape = {8};
    rejects([&] { rpu_launch_eltwise_unary_kernel(input, output, ValuOpType::ADD, RpuUnaryPrecision::HIGH); });
    missing = op == ValuOpType::SILU ? KernelId::UNARY_SILU_HIGH_PRECISION : KernelId::UNARY_TANH_HIGH_PRECISION;
    const int dma_before = dma_in;
    rejects([&] { rpu_launch_eltwise_unary_kernel(input, output, op, RpuUnaryPrecision::HIGH); });
    assert(dma_in == dma_before); missing = KernelId::_COUNT;
  }
  assert(device_address_queries == 0);
  std::vector<c10::Half> values(8);
  at::Tensor t{values.data(), {8}};
  rpu_launch_eltwise_unary_kernel(t, t, ValuOpType::TANH);
  assert(selected == KernelId::UNARY_DDR && device_address_queries == 2 && kernel.regs[61] == 3);
}
'''
    precision = _block(_read("src/core/rpu_math_precision.h"), "enum class RpuUnaryPrecision") + ";\n"
    _run(tmp_path, "unary_high", COMMON + kernel_ids + harness + precision + unary_decls
         + _block(source, "void rpu_launch_eltwise_unary_spm_kernel(")
         + _block(source, "void rpu_launch_eltwise_unary_kernel(") + checks)


def test_newton_routes_and_atomic_cold_payload_admission(tmp_path):
    header = _read("src/core/rpu_kernel_cache.h")
    declarations = _read("src/core/rpu_kernel_decls.h")
    source = _read("src/ops/rpu_rmsnorm.cpp")
    rms_decls = declarations[declarations.index("inline constexpr int64_t RPU_RMSNORM_SCM_U16_MAX"):
                             declarations.index("enum class RpuMixedRmsNormSpmRoute")]
    harness = r'''
constexpr int UNIFIED_NUM_CORES = 8;
struct Kernel_t {
  std::array<uint16_t,64> regs{};
  void reset_regs() { regs.fill(0); }
  void set_regs(int r, uint16_t value) { regs.at(r) = value; }
} kernel;
KernelId selected, missing = KernelId::_COUNT;
Kernel_t* GET_KERNEL(KernelId id) { selected = id; return id == missing ? nullptr : &kernel; }
struct Queue {
  int cores = 0, launches = 0;
  bool broadcast = false;
  std::array<uint16_t,3> grid{};
  void set_broadcast_mode(bool b) { broadcast = b; }
  void enqueu_kernel(Kernel_t&, std::array<uint16_t,3> g, std::vector<uint8_t> ids) {
    assert(broadcast && ids.size() == size_t(cores));
    for (int i = 0; i < cores; ++i) assert(ids[i] == i);
    ++launches; grid = g;
  }
} queue;
Queue* GET_QUEUE(int cores) { queue.cores = cores; return &queue; }
struct KernelCache {
  KernelPayloadAvailability available;
  static KernelCache& instance() { static KernelCache c; return c; }
  bool has_loaded(KernelId id) const { return available.has_loaded(id); }
};
'''
    checks = r'''
int main() {
  static_assert(int(RpuRmsNormSpmRoute::BASE) == 1 && int(RpuRmsNormSpmRoute::V32_GRID8) == 6);
  static_assert(int(RpuRmsNormSpmRoute::NEWTON) == 7 && int(RpuRmsNormSpmRoute::NEWTON_V32_GRID8) == 9);
  for (int cores : {1,4,6,8}) for (int64_t rows : {31,62,240,256,480,65535,65536,524256,524272}) {
    const auto route = rpu_resolve_high_precision_rmsnorm_spm_route(rows,128);
    const auto expected = rows%32 == 0 ? RpuRmsNormSpmRoute::NEWTON_V32_GRID8
        : rows%16 == 0 ? RpuRmsNormSpmRoute::NEWTON_V16_GRID8 : RpuRmsNormSpmRoute::NEWTON;
    assert(route == expected);
    rpu_launch_rmsnorm_spm_kernel(0x40001000,0x40002000,0x40003000,rows,128,1e-6,route,cores);
    const auto id = route == RpuRmsNormSpmRoute::NEWTON ? KernelId::RMS_NORM_NEWTON_SPM
        : route == RpuRmsNormSpmRoute::NEWTON_V16_GRID8 ? KernelId::RMS_NORM_SPM_NEWTON_V16 : KernelId::RMS_NORM_SPM_NEWTON_V32;
    assert(selected == id);
    assert(queue.grid[0] == (route == RpuRmsNormSpmRoute::NEWTON ? rows : 8));
    assert(kernel.regs[3] == (route == RpuRmsNormSpmRoute::NEWTON ? 1 : rows/8));
    assert(kernel.regs[0] == 128 && kernel.regs[2] == 256);
    assert(kernel.regs[4] + (uint32_t(kernel.regs[5])<<16) == 0x40001000);
    assert(kernel.regs[6] + (uint32_t(kernel.regs[7])<<16) == 0x40002000);
    assert(kernel.regs[8] + (uint32_t(kernel.regs[9])<<16) == 0x40003000);
    missing = id; const int before = queue.launches;
    rejects([&] { rpu_launch_rmsnorm_spm_kernel(1,1,2,rows,128,1e-6,route,cores); });
    assert(queue.launches == before); missing = KernelId::_COUNT;
  }
  assert(rpu_resolve_high_precision_rmsnorm_spm_route(32,8192) == RpuRmsNormSpmRoute::NEWTON_V32_GRID8);
  assert(rpu_resolve_high_precision_rmsnorm_spm_route(32,8208) == RpuRmsNormSpmRoute::NEWTON);
  for (auto pair : {std::array<int64_t,2>{0,128}, {32,0}, {32,127}, {32,32768}, {524288,128}, {65537,128}, {65536,8208}}) {
    rejects([&] { (void)rpu_resolve_high_precision_rmsnorm_spm_route(pair[0],pair[1]); });
  }
  rejects([] { rpu_launch_rmsnorm_spm_kernel(1,1,2,31,128,1e-6,RpuRmsNormSpmRoute::NEWTON_V16_GRID8,8); });
  rejects([] { rpu_launch_rmsnorm_spm_kernel(1,1,2,240,128,1e-6,RpuRmsNormSpmRoute::NEWTON_V32_GRID8,8); });
  rejects([] { rpu_launch_rmsnorm_spm_kernel(1,1,2,32,8208,1e-6,RpuRmsNormSpmRoute::NEWTON_V32_GRID8,8); });
  rejects([] { rpu_launch_rmsnorm_spm_kernel(1,1,2,65536,128,1e-6,RpuRmsNormSpmRoute::NEWTON,8); });
  rejects([] { rpu_launch_rmsnorm_spm_kernel(1,1,2,31,128,1e-6,RpuRmsNormSpmRoute::NEWTON,9); });
  rpu_launch_rmsnorm_spm_kernel(1,1,2,31,128,1e-6,RpuRmsNormSpmRoute::BASE,8);
  assert(selected == KernelId::RMS_NORM_BF16_SPM && kernel.regs[3] == 0 && queue.grid[0] == 31);
  rpu_launch_rmsnorm_spm_kernel(1,1,2,240,128,1e-6,RpuRmsNormSpmRoute::V16_GRID8,8);
  assert(selected == KernelId::RMS_NORM_SPM_V16 && kernel.regs[3] == 30 && queue.grid[0] == 8);

  auto& availability = KernelCache::instance().available;
  rejects([] { rpu_require_high_precision_math_kernels(); });
  KernelPayloadAvailability::ExtendedPayloads extended;
  const std::array<KernelId,4> unary = {KernelId::UNARY_SILU_HIGH_PRECISION,
      KernelId::UNARY_TANH_HIGH_PRECISION, KernelId::UNARY_GELU_ERF_HIGH_PRECISION,
      KernelId::UNARY_GELU_TANH_HIGH_PRECISION};
  const auto publish = [&](bool base, bool v16, bool v32) {
    availability.publish(true,true,false,false,false,false,extended,false,false,false,false,base,v16,v32);
  };
  availability.publish(true,true);
  assert(!availability.has_loaded(KernelId::RMS_NORM_NEWTON_SPM));
  rejects([] { rpu_require_high_precision_math_kernels(); });
  for (auto id : unary) extended.set(size_t(id)-KernelPayloadAvailability::kExtendedBegin);
  for (int bit = 0; bit < 3; ++bit) {
    publish(bit!=0,bit!=1,bit!=2);
    rejects([] { rpu_require_high_precision_math_kernels(); });
    assert(availability.has_loaded(KernelId::RMS_NORM_NEWTON_SPM) == (bit!=0));
    assert(availability.has_loaded(KernelId::RMS_NORM_SPM_NEWTON_V16) == (bit!=1));
    assert(availability.has_loaded(KernelId::RMS_NORM_SPM_NEWTON_V32) == (bit!=2));
  }
  for (auto id : unary) {
    extended.reset(size_t(id)-KernelPayloadAvailability::kExtendedBegin);
    publish(true,true,true);
    rejects([] { rpu_require_high_precision_math_kernels(); });
    extended.set(size_t(id)-KernelPayloadAvailability::kExtendedBegin);
  }
  publish(true,true,true); rpu_require_high_precision_math_kernels();
  availability.withdraw(); rejects([] { rpu_require_high_precision_math_kernels(); });
  publish(true,true,true); rpu_require_high_precision_math_kernels();
}
'''
    _run(tmp_path, "newton", COMMON
         + _block(header, "enum class KernelId") + ";\n"
         + _block(header, "class KernelPayloadAvailability final") + ";\n"
         + harness + rms_decls
         + _block(declarations, "inline RpuRmsNormSpmRoute rpu_resolve_high_precision_rmsnorm_spm_route(")
         + _block(source, "void rpu_launch_rmsnorm_spm_kernel(")
         + _block(_read("src/ops/rpu_eltwise_unary.cpp"), "void rpu_require_high_precision_math_kernels()")
         + checks)


def test_high_payload_names_are_optional_and_main_ref_owned():
    cache = _read("src/core/rpu_kernel_cache.inc")
    names = _read("src/core/rpu_kernel_cache.h")
    optional = _block(cache, "bool is_optional_kernel(")
    for symbol in ("unary_silu_high_precision", "unary_tanh_high_precision",
                   "unary_gelu_erf_high_precision", "unary_gelu_tanh_high_precision"):
        assert f'{{"{symbol}", RHINO_OP_LIB_PATH}}' in cache
        assert f'"{symbol}"' in optional
        assert f'"{symbol}"' in names
