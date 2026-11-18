"""Exercise the real eager LayerNorm SPM transfer helpers without a device."""
from pathlib import Path
import subprocess


def test_layernorm_spm_copy_preserves_logical_extent_and_aligned_access(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/ops/rpu_layernorm.cpp").read_text()
    start = source.index("namespace {\n\n// CPU mappings of SPM")
    end = source.index("}  // namespace", start) + len("}  // namespace")
    helpers = source[start:end]
    launcher = source[source.index("void rpu_launch_layernorm_kernel("):]
    launcher = launcher[:launcher.index("// Direct Address SPM LayerNorm Kernel")]
    assert launcher.count("layernorm_copy_ddr_to_spm(") == 3
    assert launcher.count("layernorm_copy_spm_to_ddr(") == 1
    assert "size_t input_n_byte_step = lc * dwidth;" in launcher
    assert "sizeof(c10::Half)" in launcher
    code = r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
using std::size_t;
static size_t transfer_bytes = 0;
static void checked_copy(void* dst, const void* src, size_t bytes) {
  assert(reinterpret_cast<uintptr_t>(dst) % 32 == 0);
  assert(reinterpret_cast<uintptr_t>(src) % 32 == 0);
  assert(bytes % 32 == 0);
  transfer_bytes = bytes;
  std::memcpy(dst, src, bytes);
}
void ddr_to_spm(void* dst, const void* src, size_t bytes) {
  checked_copy(dst, src, bytes);
}
void spm_to_ddr(void* dst, const void* src, size_t bytes) {
  checked_copy(dst, src, bytes);
}
''' + helpers + r'''
static uint8_t* align32(std::vector<uint8_t>& storage) {
  return storage.data() + ((-reinterpret_cast<uintptr_t>(storage.data())) & 31);
}
int main() {
  // Affine vectors and N=17 packed rows on both sides of the 512 boundary.
  // Offset views also force staging when their logical size is aligned.
  for (size_t channels : {121U, 256U, 512U, 513U, 768U, 1551U}) {
    for (size_t rows : {1U, 17U}) {
      for (size_t offset : {0U, 2U}) {
        const size_t bytes = rows * channels * 2;
        const size_t padded = (bytes + 31) & ~size_t(31);
        std::vector<uint8_t> input_storage(bytes + 160, 0xa5);
        std::vector<uint8_t> output_storage(bytes + 160, 0xa5);
        std::vector<uint8_t> spm_storage(padded + 160, 0xc7);
        auto* input = align32(input_storage) + 32 + offset;
        auto* output = align32(output_storage) + 32 + offset;
        auto* spm = align32(spm_storage) + 32;
        for (size_t i=0; i<bytes; ++i) input[i] = (i * 19 + 7) % 251;
        const auto original_input = input_storage;
        layernorm_copy_ddr_to_spm(spm, input, bytes);
        assert(transfer_bytes == padded);
        assert(std::memcmp(spm, input, bytes) == 0);
        assert(std::all_of(spm + bytes, spm + padded, [](auto b){return b==0;}));
        assert(std::all_of(spm - 32, spm, [](auto b){return b==0xc7;}));
        assert(std::all_of(spm + padded, spm + padded + 32, [](auto b){return b==0xc7;}));
        assert(input_storage == original_input);
        // The output copy must never touch the user tensor's rounded tail.
        std::fill(spm + bytes, spm + padded, 0xee);
        layernorm_copy_spm_to_ddr(output, spm, bytes);
        assert(transfer_bytes == padded);
        assert(std::memcmp(output, input, bytes) == 0);
        assert(std::all_of(output - 32, output, [](auto b){return b==0xa5;}));
        assert(std::all_of(output + bytes, output + bytes + 32, [](auto b){return b==0xa5;}));
      }
    }
  }
}
'''
    cpp = tmp_path / "layernorm_copy.cpp"
    exe = tmp_path / "layernorm_copy"
    cpp.write_text(code)
    subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
