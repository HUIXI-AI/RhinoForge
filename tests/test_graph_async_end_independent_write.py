"""Host contract for host writes that skip the asynchronous Graph end wait.

`src/graph/graph_async_end.cpp` is compiled whole against doubles of the SDK
queue/flush, the DMA submission lease and the logical DDR registry; the copy
preflight is extracted from `src/core/rpu_tensor_ops.inc`.  Tensors are real
ATen tensors (PrivateUse1 storage over host buffers).  No board, SDK or Graph
execution is involved.
"""
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]

_PRELUDE = r'''
#include <ATen/ATen.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "CHECK failed line %d: %s\n", __LINE__, #cond); \
    std::exit(1); } } while (0)

struct Hooks : at::PrivateUse1HooksInterface {
    at::Device getDeviceFromPtr(void*) const override {
        return at::Device(c10::DeviceType::PrivateUse1, 0);
    }
};

at::Tensor rpu_view(void* ptr, int64_t elems) {
    return at::from_blob(ptr, {elems}, [](void*) {},
                         at::TensorOptions().dtype(at::kHalf).device(
                             c10::Device(c10::DeviceType::PrivateUse1, 0)));
}
'''

_ASYNC_DOUBLES = r'''
namespace rhino_lkn {
class Buffer_t {};
class Queue_t {
public:
    int finishes = 0;
    void finish() { ++finishes; }
};
Buffer_t g_owner;
void RpuDdrFlush(void*) {}
}  // namespace rhino_lkn
class RpuKernelGraph {};
class RpuDmaSubmissionLease {
public:
    RpuDmaSubmissionLease() = default;
    RpuDmaSubmissionLease(RpuDmaSubmissionLease&&) noexcept = default;
    RpuDmaSubmissionLease& operator=(RpuDmaSubmissionLease&&) noexcept = default;
    explicit operator bool() const { return true; }
};
struct RpuDmaEndpoint {
    rhino_lkn::Buffer_t* owner = nullptr;
    size_t offset = 0;
    uint64_t allocation_id = 0;
};
struct FakeAllocation { const char* base; size_t bytes; uint64_t id; };
std::vector<FakeAllocation> g_allocations;
RpuDmaEndpoint rpu_resolve_cpu_dma_endpoint(const void* ptr, size_t bytes,
                                            const char* where) {
    const char* p = static_cast<const char*>(ptr);
    for (const auto& a : g_allocations) {
        if (p >= a.base && p + bytes <= a.base + a.bytes) {
            return {&rhino_lkn::g_owner, static_cast<size_t>(p - a.base), a.id};
        }
    }
    TORCH_CHECK(false, where, ": CPU DDR DMA range is outside every live "
                "logical allocation");
}
void rpu_graph_async_end_drain_deferred_frees() noexcept {}
void rpu_graph_async_end_quarantine_deferred_frees() noexcept {}
'''

_ASYNC_MAIN = r'''
static at::Tensor g_host_tensor;
static bool independent(const void* dst, size_t bytes,
                        const c10::Storage& src = g_host_tensor.storage()) {
    return rpu_graph_async_end_independent_write(dst, bytes, src);
}

// CPU views over memory the CPU allocator did not hand out: built the way
// zerocopy_to_cpu builds its views (no allocator), or by another allocator.
struct OtherAllocator final : c10::Allocator {
    c10::DataPtr allocate(size_t) override { return {}; }
    void copy_data(void*, const void*, std::size_t) const override {}
};
static OtherAllocator g_other_allocator;
static c10::Storage foreign_cpu_storage(void* ptr, size_t bytes,
                                        c10::Allocator* allocator = nullptr) {
    return c10::Storage(c10::Storage::use_byte_size_t(), bytes,
                        c10::DataPtr(ptr, at::Device(at::kCPU)), allocator, false);
}

int main() {
    at::RegisterPrivateUse1HooksInterface(new Hooks());
    alignas(512) static c10::Half mask[256], rope[256], other[256];
    alignas(64) static char unregistered[4096];
    g_host_tensor = at::zeros({32}, at::kHalf);
    void* host = g_host_tensor.data_ptr();
    g_allocations = {{reinterpret_cast<char*>(mask), sizeof(mask), 11},
                     {reinterpret_cast<char*>(rope), sizeof(rope), 12},
                     {reinterpret_cast<char*>(other), sizeof(other), 13}};
    const std::vector<std::string> prefill_only = {"rpu_async_owner_prefill"};
    rhino_lkn::Queue_t queue;
    RpuKernelGraph prefill, vision;
    std::vector<void*> flush;
    std::vector<at::Tensor> refs;

    // Marks need a live RPU DDR storage.
    bool threw = false;
    try { rpu_graph_async_end_mark_independent(at::zeros({4}), prefill_only); }
    catch (const c10::Error&) { threw = true; }
    CHECK(threw);
    threw = false;
    try {
        rpu_graph_async_end_mark_independent(
            rpu_view(unregistered, 16), prefill_only);
    } catch (const c10::Error&) { threw = true; }
    CHECK(threw);
    rpu_graph_async_end_mark_independent(rpu_view(mask, 256), prefill_only);
    rpu_graph_async_end_mark_independent(rpu_view(mask, 256), prefill_only);
    rpu_graph_async_end_mark_independent(rpu_view(rope, 256), prefill_only);

    // Nothing pending, or an unsealed record: the caller waits as usual.
    CHECK(!independent(mask, 64));
    CHECK(rpu_graph_async_end_begin(&queue, &prefill, RpuDmaSubmissionLease{}));
    CHECK(!independent(mask, 64));
    CHECK(rpu_graph_async_end_seal(&prefill, "rpu_async_owner_prefill", flush, refs));

    CHECK(independent(mask, sizeof(mask)));
    CHECK(independent(rope + 16, 64));
    CHECK(!independent(other, 64));                 // unmarked storage
    CHECK(!independent(mask, sizeof(mask) + 2));    // leaves the storage
    CHECK(!independent(host, 8));                   // not RPU DDR at all
    CHECK(!independent(mask, 64, foreign_cpu_storage(rope, sizeof(rope))));  // zero-copy view
    CHECK(!independent(mask, 64, foreign_cpu_storage(rope, sizeof(rope),
                                                     &g_other_allocator)));
    CHECK(!independent(mask, 64, c10::Storage()));  // no storage
    CHECK(!independent(mask, 0));
    CHECK(rpu_graph_async_end_pending() && queue.finishes == 0);
    rpu_graph_async_end_wait("test");
    CHECK(!rpu_graph_async_end_pending() && queue.finishes == 1);
    CHECK(!independent(mask, 64));

    // A mark names Graphs: another pending Graph still makes the write wait.
    CHECK(rpu_graph_async_end_begin(&queue, &vision, RpuDmaSubmissionLease{}));
    CHECK(rpu_graph_async_end_seal(&vision, "rpu_async_owner_vision", flush, refs));
    CHECK(!independent(mask, 64));
    rpu_graph_async_end_wait("test");

    const std::string stats = rpu_graph_async_end_stats();
    CHECK(stats.find("rpu_async_owner_prefill|independent_write count=2 ") !=
          std::string::npos);
    CHECK(stats.find("rpu_async_owner_vision|independent_write") == std::string::npos);

    // Stats resets keep the marks: they are owner state, not diagnostics.
    rpu_graph_async_end_reset_stats();
    CHECK(rpu_graph_async_end_begin(&queue, &prefill, RpuDmaSubmissionLease{}));
    CHECK(rpu_graph_async_end_seal(&prefill, "rpu_async_owner_prefill", flush, refs));
    CHECK(independent(mask, 64));
    rpu_graph_async_end_wait("test");
    g_host_tensor.reset();
    std::puts("ok");
    return 0;
}
'''

_COPY_TEMPLATE = r'''
bool g_pending = true;
bool g_independent = false;
std::vector<std::string> g_waits;
struct IndependentCall { const void* dst; size_t bytes; const void* src; };
std::vector<IndependentCall> g_calls;
bool rpu_graph_async_end_pending() noexcept { return g_pending; }
void rpu_graph_async_end_wait(const char* reason) noexcept { g_waits.push_back(reason); }
bool rpu_graph_async_end_independent_write(const void* dst, size_t bytes,
                                           const c10::Storage& src) noexcept {
    g_calls.push_back({dst, bytes, src.data()});
    return g_independent;
}
using at::Tensor;

at::Tensor& preflight(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    (void)non_blocking;
@PREFLIGHT@
    return self;
}

int main() {
    at::RegisterPrivateUse1HooksInterface(new Hooks());
    alignas(512) static c10::Half slot[64];
    auto dst = rpu_view(slot, 64);
    auto host = at::zeros({64}, at::kHalf);

    // Marked: one lookup over the destination range and the host source.
    g_independent = true;
    preflight(dst, host, false);
    CHECK(g_waits.empty() && g_calls.size() == 1);
    CHECK(g_calls[0].dst == slot && g_calls[0].bytes == sizeof(slot));
    CHECK(g_calls[0].src == host.storage().data_ptr().get());

    // Unmarked: the ordinary wait.
    g_independent = false;
    preflight(dst, host, false);
    CHECK(g_calls.size() == 2 && g_waits == std::vector<std::string>{"copy_to_rpu"});

    // Reads from RPU, RPU->RPU and converting writes always wait unasked.
    g_independent = true;
    g_waits.clear();
    g_calls.clear();
    auto host_out = at::zeros({64}, at::kHalf);
    preflight(host_out, dst, false);
    alignas(512) static c10::Half peer[64];
    auto peer_view = rpu_view(peer, 64);
    preflight(dst, peer_view, false);
    auto host_float = at::zeros({64}, at::kFloat);
    preflight(dst, host_float, false);
    CHECK(g_calls.empty());
    CHECK((g_waits == std::vector<std::string>{"copy_from_rpu", "copy_from_rpu",
                                                "copy_to_rpu"}));

    // Nothing pending: no lookup at all.
    g_pending = false;
    g_waits.clear();
    preflight(dst, host, false);
    CHECK(g_calls.empty() && g_waits.empty());
    std::puts("ok");
    return 0;
}
'''


def _function(source, marker):
    start = source.index(marker)
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
                    lambda match: " " * len(match[0]), source)
    brace = masked.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    return source[start:end]


def _run(tmp_path, name, source):
    import torch

    compiler = shutil.which("c++")
    assert compiler, "the async-end independence contract requires c++"
    cpp, binary = tmp_path / f"{name}.cpp", tmp_path / name
    cpp.write_text(source)
    torch_root = Path(torch.__file__).resolve().parent
    command = [compiler, "-std=c++17", "-O0", "-D_GLIBCXX_USE_CXX11_ABI=" +
               str(int(torch.compiled_with_cxx11_abi())), "-I", str(ROOT / "src"),
               "-I", str(torch_root / "include"),
               "-I", str(torch_root / "include/torch/csrc/api/include"), str(cpp),
               "-L", str(torch_root / "lib"), "-Wl,-rpath," + str(torch_root / "lib"),
               "-ltorch_cpu", "-lc10", "-o", str(binary)]
    built = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert built.returncode == 0, built.stdout + built.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "ok"


def test_marked_allocation_skips_only_its_own_pending_graph(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    module = (ROOT / "src/graph/graph_async_end.cpp").read_text()
    for include in ('#include "core/rpu_dma_endpoint.h"\n', '#include "rpu_ops.h"\n'):
        assert module.count(include) == 1
        module = module.replace(include, "")
    _run(tmp_path, "async_end", _PRELUDE + _ASYNC_DOUBLES + module + _ASYNC_MAIN)


def test_copy_preflight_asks_only_for_same_dtype_host_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    copy_source = (ROOT / "src/core/rpu_tensor_ops.inc").read_text()
    copy = _function(copy_source, "at::Tensor &rpu_copy_impl(")
    preflight = copy[copy.index("{") + 1:copy.index("  const bool same_dtype")]
    assert "rpu_graph_async_end_independent_write(" in preflight
    _run(tmp_path, "copy_preflight",
         _PRELUDE + _COPY_TEMPLATE.replace("@PREFLIGHT@", preflight))
