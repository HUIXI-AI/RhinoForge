"""Exercise the native scale-retention body with synthetic device addresses."""
from pathlib import Path
import shutil
import subprocess


def test_packed_scales_keep_payload_and_satisfy_controller_and_execution_stripes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/ops/rpu_linear.cpp").read_text()
    start = source.index("at::Tensor rpu_retain_linear_quant_scale(")
    end = source.index("\n}\n", start) + 2
    helper = source[start:end]
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>
#define NUM_CORES 8
#define TORCH_CHECK(c,...) do {if (!(c)) throw std::runtime_error("rejected");} while(0)
#define TORCH_INTERNAL_ASSERT(c,...) assert(c)
namespace c10 { using Half = uint16_t; enum class DeviceType { PrivateUse1 }; }
namespace at {
enum ScalarType { kByte, kHalf, kChar };
struct Storage { uint64_t address; std::vector<c10::Half> data; };
struct Device { c10::DeviceType type() const {return c10::DeviceType::PrivateUse1;} };
struct Tensor {
    std::shared_ptr<Storage> owner;
    std::vector<int64_t> shape;
    int64_t offset = 0;
    ScalarType dtype = kHalf;
    bool defined() const { return bool(owner); }
    ScalarType scalar_type() const {return dtype;}
    int dim() const {return shape.size();}
    Device device() const {return {};}
    bool is_contiguous() const {return true;}
    int64_t numel() const {return std::accumulate(shape.begin(),shape.end(),int64_t{1},std::multiplies<int64_t>());}
    template<class T> T* data_ptr() const {return reinterpret_cast<T*>(owner->address + offset * 2);}
    int options() const {return 0;}
    auto sizes() const {return shape;}
    Tensor narrow(int, int64_t first, int64_t count) const {
        auto result = *this; result.offset += first; result.shape = {count}; return result;
    }
    Tensor view(const std::vector<int64_t>& dims) const {auto result = *this; result.shape = dims; return result;}
    void copy_(const Tensor& other) {
        std::copy_n(other.owner->data.begin()+other.offset,numel(),owner->data.begin()+offset);
    }
};
Tensor empty(std::initializer_list<int64_t> shape, int) {
    const int64_t size = *shape.begin();
    return {std::make_shared<Storage>(Storage{0x10000,std::vector<c10::Half>(size)}),shape};
}
}
uint64_t RpuGetDevAddr(c10::Half* value) {return reinterpret_cast<uintptr_t>(value);}
__HELPER__
int main() {
    for (int cores : {4,6,8}) for (uint64_t address : {0x1000,0x2000,0x3000,0x3100}) {
        at::Tensor scale{std::make_shared<at::Storage>(at::Storage{address,std::vector<c10::Half>(1024)}),{32,32}};
        std::iota(scale.owner->data.begin(),scale.owner->data.end(),uint16_t{1});
        auto weight = scale; weight.dtype = at::kByte;
        const auto result = rpu_retain_linear_quant_scale(weight,scale,cores);
        const auto actual_address = RpuGetDevAddr(result.data_ptr<c10::Half>());
        assert(actual_address % 4096 == 0 && actual_address % (cores * 512) == 0);
        assert(result.sizes() == scale.sizes());
        assert(std::equal(scale.owner->data.begin(),scale.owner->data.end(),result.owner->data.begin()+result.offset));
        if (address % 4096 == 0 && address % (cores * 512) == 0) assert(result.owner == scale.owner);
        auto w8 = weight; w8.dtype = at::kChar;
        assert(rpu_retain_linear_quant_scale(w8,scale,cores).owner == scale.owner);
    }
}
'''.replace("__HELPER__", helper)
    compiler = shutil.which("g++")
    assert compiler
    path = tmp_path / "scale_alignment.cpp"
    path.write_text(cpp)
    binary = path.with_suffix("")
    built = subprocess.run([compiler, "-std=c++17", str(path), "-o", str(binary)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    subprocess.run([str(binary)], check=True)
