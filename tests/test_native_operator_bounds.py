"""Exercise real wrapper admission before narrowing or device-side effects."""

from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _function(path, signature):
    source = (ROOT / path).read_text()
    start = source.index(signature)
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_native_operator_admission(tmp_path):
    compiler = shutil.which("g++")
    assert compiler, "native wrapper checks require g++"
    functions = []
    for path, signature, boundary in (
        ("rpu_rope.cpp", "void rpu_launch_rope_spm_kernel(\n    uint32_t", "  // Flush DDR cos/sin"),
        ("rpu_rmsnorm.cpp", "void rpu_launch_rmsnorm_spm_kernel(", "  size_t dwidth = "),
        ("rpu_sdpa.cpp", "static void launch_sdpa_spm_unified_impl(", "  size_t dwidth = "),
    ):
        body = _function("src/ops/" + path, signature)
        # Stop where actual cache staging/DDR access begins, retaining the
        # original wrapper signature and all preceding arithmetic/checks.
        functions.append(body[:body.index(boundary)] + "  ++staged;\n}")
    functions.append(_function("src/ops/rpu_sdpa.cpp", "void rpu_launch_sdpa_spm_unified_kernel_v2("))
    for signature in ("void validate_parallel_linear_geometry(",
                      "void validate_parallel_linear_grid("):
        functions.append(_function("src/ops/rpu_linear.cpp", signature))
    ddr = _function("src/ops/rpu_linear.cpp", "void rpu_launch_linear_ddr_kernel(")
    functions.append(ddr[:ddr.index("  TORCH_CHECK(input.scalar_type()")] + "  ++staged;\n}")
    harness = r'''
#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
constexpr int ROPE_NUM_CORES=8, MAX_CORES=8, NUM_CORES=8, UNIFIED_NUM_CORES=8, RPU_RMSNORM_MAX_COLS=32752;
namespace c10 { using Half=uint16_t; template<class T> using optional=std::optional<T>; }
namespace at {
constexpr int kHalf=0, kByte=1;
struct Tensor {
    std::vector<int64_t> shape;
    int scalar_type() const { return kHalf; }
    size_t dim() const { return shape.size(); }
    int64_t size(size_t i) const { return shape.at(i); }
};
}
using at::Tensor;
struct SdpaTiling {};
void check_linear_w4a16_operands(const Tensor&, const Tensor&, const Tensor&, c10::optional<Tensor>) {
    throw std::runtime_error("FP16 admission fixture must not enter W4");
}
enum class RpuRmsNormSpmRoute { BASE, V16, V32, QWEN3VL_V64 };
enum class KernelId { SDPA_FLASH_ATTN_SPM_16B, SDPA_FLASH_ATTN_SPM };
int64_t CeilDiv(int64_t n, int64_t d) { assert(d>0); return (n+d-1)/d; }
int staged=0;
__FUNCTIONS__
template<class F> void rejects(F&& f) {
    const int before=staged;
    bool failed=false; try { f(); } catch (const std::runtime_error&) { failed=true; }
    assert(failed && staged==before);
}
int main() {
    auto rope = [](int64_t rows, int64_t dim, int64_t pos, int cores=8) {
        rpu_launch_rope_spm_kernel(0,0,nullptr,nullptr,rows,2,dim,pos,cores);
    };
    rope(1,128,4096); rope(16,128,0);
    rejects([&] { rope(1,128,0,0); }); rejects([&] { rope(65536,128,0); });
    rejects([&] { rope(1,127,0); }); rejects([&] { rope(2,128,UINT32_MAX); });
    rejects([&] { rope(1,128,-1); });
    auto rms = [](int64_t cols, int cores=8) {
        rpu_launch_rmsnorm_spm_kernel(0,0,0,16,cols,1e-6,RpuRmsNormSpmRoute::BASE,cores);
    };
    rms(128); rejects([&] { rms(127); });
    rms(128,4); rejects([&] { rms(128,0); }); rejects([&] { rms(128,9); });
    at::Tensor cache;
    auto sdpa = [&](int64_t q, int64_t kv, int cores=8, int vcores=-1, int64_t length=4096) {
        rpu_launch_sdpa_spm_unified_kernel_v2(cache,cache,1,{},
            0,0,0,0,16,q,kv,128,length,cores,vcores,0,false,false);
    };
    sdpa(16,8); sdpa(16,2); // Existing sharded and replicated KV cases.
    rejects([&] { sdpa(16,8,0); }); rejects([&] { sdpa(16,0); });
    rejects([&] { sdpa(24,16); }); rejects([&] { sdpa(16,8,8,9); });
    rejects([&] { sdpa(65535,1,1,8); });
    rejects([&] { sdpa(16,8,8,-1,INT64_MAX); });
    validate_parallel_linear_geometry(1,151936,4096,1,8);
    validate_parallel_linear_geometry(16,1024,2048,0,8);
    rejects([] { validate_parallel_linear_geometry(16,1024,2048,0,0); });
    rejects([] { validate_parallel_linear_geometry(16,1024,2048,2,8); });
    rejects([] { validate_parallel_linear_geometry(16,1040,2048,1,8); });
    rejects([] { validate_parallel_linear_geometry(INT64_MAX,1024,2048,1,8); });
    validate_parallel_linear_grid(65535,65535,1,1);
    rejects([] { validate_parallel_linear_grid(65536,128,1,16); });
    rejects([] { validate_parallel_linear_grid(128,128,0,16); });
    auto linear = [](at::Tensor x, at::Tensor w, at::Tensor y) {
        rpu_launch_linear_ddr_kernel(x,w,y,{},false,1,nullptr,0,nullptr,8,{},{});
    };
    linear({{16,1024}},{{2048,1024}},{{16,2048}});
    rejects([&] { linear({{16,1024}},{{2048,1024}},{{16,1024}}); });
    rejects([&] { linear({{16,2,512}},{{2048,1024}},{{16,2048}}); });
}
'''.replace("__FUNCTIONS__", "\n".join(functions))
    source = tmp_path / "operator_bounds.cpp"
    source.write_text(harness)
    binary = source.with_suffix("")
    result = subprocess.run([compiler, "-std=c++17", "-O0", str(source), "-o", str(binary)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
