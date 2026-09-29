"""Execute native T160 address/declaration code without an RPU or model weights."""
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _block(source, start):
    begin = source.index("{", start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def test_native_t160_residuals_fit_four_roots_and_preserve_live_attention(tmp_path):
    source = (ROOT / "src/fused/rpu_gemma_model.cpp").read_text()
    names = (
        "pi05_pair_rows", "pi05_prefix_rows", "pi05_pair_slab_bytes",
        "pi05_pair_owner_bytes", "pi05_pair_owner_elements",
        "pi05_pair_single_compact", "pi05_pair_slab_compact",
        "pi05_pair_spill_elements", "pi05_pair_compact_root_bytes",
        "pi05_pair_scratch_mask", "pi05_pair_mask_bytes",
        "pi05_pair_input_compact_address", "pi05_a8_bytes", "pi05_a8_root_bytes",
    )
    methods = []
    for name in names:
        import re
        match = re.search(r"    (?:int64_t|bool|uint32_t) " + name + r"\([^\n]*\) const \{", source)
        assert match, name
        methods.append(source[match.start():source.index("{", match.start())] + _block(source, match.start()))
    paired = _block(source, source.index("        if (pi05_down_pair_layout(ctx)) {"))
    code = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error("bounds"); } while (0)
int64_t Align(int64_t n, int64_t a) { return (n+a-1)/a*a; }
enum class StorageClass { Temp, Persistent };
struct BufferDecl {
 const char* name; int64_t size; int start, end; StorageClass storage;
 int per_layer; const char* alias; int scope;
};
struct Model {
 int64_t pi05_pair_rows_ = 464;
 bool pi05_prefill_o_pair_ = true, pi05_prefill_owner_norm_ = true, pi05_prefill_a8_ = false;
 uint32_t addr(int, const char* name) const {
  assert(std::string(name) == "pi05_pair_s3"); return 0x40700000;
 }
''' + "\n".join(methods) + r'''
 std::vector<BufferDecl> declare() {
  std::vector<BufferDecl> decls; constexpr int ALL=0;
''' + paired + r'''
 }
};
int main() {
 Model m;
 const auto d=m.pi05_pair_owner_bytes(), slab=m.pi05_pair_slab_bytes();
 assert(slab == 8*d && m.pi05_pair_compact_root_bytes() == 0);
 assert(m.pi05_pair_spill_elements()*2 == 2*d);
 const auto s3=m.addr(0,"pi05_pair_s3");
 const auto c0=m.pi05_pair_input_compact_address(0)-s3;
 const auto c1=m.pi05_pair_input_compact_address(1)-s3;
 // SDPA writes the first two head-sized stripes; both inputs must survive.
 assert(c0 >= 2*d && c1 >= 2*d && c0+d <= slab && c1+d <= slab);
 assert(c0+d <= c1 || c1+d <= c0);
 // Norm0 writes raw0 in [0,D), Norm1 writes raw1 in [D,2D).
 assert(2*d <= std::min(c0,c1));
 for (bool a8 : {false,true}) {
  m.pi05_prefill_a8_=a8;
  auto decls=m.declare(); int roots=0; int64_t temporary=0;
  for (const auto& b:decls) if (!b.alias) { ++roots; temporary+=Align(b.size,256); }
  assert(roots==4);
  // Measured available temporary space after the retained Vision buffers,
  // native fixed allocations and the existing 64 KiB reserve.
  assert(temporary <= 7667712);
  assert(temporary == (a8 ? 7633920 : 7602176));
 }
}
'''
    path = tmp_path / "layout.cpp"
    path.write_text(code)
    binary = tmp_path / "layout"
    subprocess.run(["c++", "-std=c++17", str(path), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
