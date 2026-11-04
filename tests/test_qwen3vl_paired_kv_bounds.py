"""Exercise complete paired-KV admission with the actual RPUCache layout."""
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("capacity", [608, 16384, 32768])
def test_paired_kv_admits_real_cache_and_rejects_invalid_owners_before_staging(tmp_path, capacity):
    from rpu_backend.api.cache import RPUCache

    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    cache = RPUCache(num_layers=1, batch_size=1, max_seq_len=capacity,
                     num_kv_heads=8, head_dim=128, device="cpu", attn_tp=8)
    key, value = cache.k_caches[0], cache.v_caches[0]
    source = (ROOT / "src/ops/rpu_llama_kvcache.cpp").read_text()
    start = source.index("    TORCH_CHECK(", source.index("void launch_owned_paired_kv_insert("))
    end = source.index("    const KernelId kernel_id =", start)
    guard = source[start:end]
    from test_qwen3vl_prefill_gateup_launcher import definition
    header = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    start = header.index('"Call torch.ops.rpu.causal_decoder_set_weights(...) first.");')
    delivery_guard = definition(header[start:], "if (qwen3vl_2b_w8_profile_)")
    # Keep every admission check before typed DDR register staging. Only device
    # allocation is doubled; dimensions and byte extents come from real RPUCache.
    program = r"""
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
namespace c10 { enum class DeviceType { CPU, PrivateUse1 }; }
namespace at {
enum ScalarType { kHalf, kFloat };
struct Device {
    c10::DeviceType kind = c10::DeviceType::PrivateUse1;
    int index = 0;
    c10::DeviceType type() const { return kind; }
    bool operator==(Device other) const { return kind == other.kind && index == other.index; }
};
struct Tensor {
    uintptr_t address;
    std::vector<int64_t> shape;
    int storage_id = 1;
    bool owned = true, contiguous = true;
    Device where;
    ScalarType dtype = kHalf;
    void* data_ptr() const { return reinterpret_cast<void*>(address); }
    size_t nbytes() const { size_t bytes = 2; for (auto d : shape) bytes *= d; return bytes; }
    bool defined() const { return owned; }
    Device device() const { return where; }
    ScalarType scalar_type() const { return dtype; }
    int64_t dim() const { return shape.size(); }
    bool is_contiguous() const { return contiguous; }
    int64_t size(size_t dim) const { return shape.at(dim); }
    struct ShapeView {const std::vector<int64_t>& shape;const std::vector<int64_t>& vec()const{return shape;}};
    ShapeView sizes()const{return {shape};}
};
}
bool delivery_admits(at::Tensor key,at::Tensor value){
    at::Tensor hidden_states{0,{1,576,2048}};
    std::vector<at::Tensor> k_caches(28,key),v_caches(28,value);
    bool qwen3vl_2b_w8_profile_=true,is_causal=true;int64_t position=0;
    std::optional<int> attention_mask;
    try{__DELIVERY_GUARD__}catch(const std::runtime_error&){return false;}return true;
}
int staged = 0;
void admit(at::Tensor k_cache, at::Tensor v_cache, int64_t position,
           int64_t seq_len, bool decode, int64_t num_kv_heads, int64_t head_dim, int num_cores) {
    constexpr bool lingbot_m224 = false;
    __GUARD__
    ++staged;
}
void check(at::Tensor key, at::Tensor value, bool expected, int64_t position=576,
           int64_t rows=1, bool decode=true, int64_t heads=8, int64_t hd=128, int cores=8) {
    const auto before = staged;
    bool accepted = true;
    try { admit(key, value, position, rows, decode, heads, hd, cores); }
    catch (const std::runtime_error&) { accepted = false; }
    assert(accepted == expected);
    assert(staged == before + int(expected));
}
int main(int argc, char** argv) {
    assert(argc == 17);
    std::vector<int64_t> k_shape, v_shape;
    for (int i=1; i<=7; ++i) k_shape.push_back(std::stoll(argv[i]));
    for (int i=8; i<=14; ++i) v_shape.push_back(std::stoll(argv[i]));
    at::Tensor key{0x10000000, k_shape}, value{0x20000000, v_shape};
    assert(key.nbytes() == std::stoull(argv[15]) && value.nbytes() == std::stoull(argv[16]));
    assert(delivery_admits(key,value));
    auto unequal=value;unequal.shape[1]++;assert(!delivery_admits(key,unequal));
    const int64_t capacity=key.shape[1]*16;
    // The actual P576/C192 caller and all four actual decode positions.
    for (auto pos : {0,192,384}) check(key,value,true,pos,192,false);
    for (auto pos : {576,577,578,579}) check(key,value,true,pos);
    check(key,value,true,capacity-1);  // last owned token
    check(key,value,true,capacity-192,192,false);  // exactly fills the owner
    check(key,value,false,capacity);
    check(key,value,false,capacity-176,192,false);
    check(key,value,false,-1);
    check(key,value,false,1,192,false);  // prefill requires 16 alignment
    check(key,value,false,0,191,false);
    check(key,value,false,576,2,true);
    check(key,value,false,576,1,true,4);
    check(key,value,false,576,1,true,8,64);
    check(key,value,false,576,1,true,8,128,4);
    check(key,value,false,std::numeric_limits<int64_t>::max());
    check(key,value,false,std::numeric_limits<uint32_t>::max());
    // Old incorrect key permutation has identical bytes but violates real ABI.
    auto bad=key; std::swap(bad.shape[4],bad.shape[5]); check(bad,value,false);
    for (int dim : {0,2,3,4,5,6}) {
        bad=key; ++bad.shape[dim]; check(bad,value,false);
        bad=value; ++bad.shape[dim]; check(key,bad,false);
    }
    bad=value; --bad.shape[1]; check(key,bad,false);
    bad=key; bad.shape[1]=0; check(bad,value,false);
    bad=key; bad.shape.pop_back(); check(bad,value,false);
    bad=key; bad.owned=false; check(bad,value,false);
    bad=key; bad.contiguous=false; check(bad,value,false);
    bad=value; bad.contiguous=false; check(key,bad,false);
    bad=key; bad.dtype=at::kFloat; check(bad,value,false);
    bad=value; bad.where.kind=c10::DeviceType::CPU; check(key,bad,false);
    bad=value; bad.where.index=1; check(key,bad,false);
    // All overlap cases now pass the complete shape/geometry admission first.
    const auto extent=key.nbytes();
    auto ranges=[&](uintptr_t a,uintptr_t b,int storage,bool expected) {
        auto k=key,v=value;k.address=a;v.address=b;v.storage_id=storage;check(k,v,expected);
    };
    constexpr uintptr_t base=0x10000000;
    ranges(base,base+256,1,false);
    ranges(base+256,base,1,false);
    ranges(base,base,2,false);
    ranges(base,base+256,2,false);
    ranges(base,base+extent,1,true);
    ranges(base+extent,base,1,true);
    ranges(base,base+extent+256,1,true);
    ranges(base,base+extent-1,2,false);
    const auto high=std::numeric_limits<uintptr_t>::max();
    ranges(high-extent+1,high-extent/2+1,2,false);
    ranges(high-extent/2+1,high-extent+1,2,false);
    ranges(high-2*extent+1,high-extent+1,1,true);
}
""".replace("__GUARD__", guard).replace("__DELIVERY_GUARD__", delivery_guard)
    path = tmp_path / "paired_kv_admission.cpp"
    path.write_text(program)
    binary = tmp_path / "paired_kv_admission"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", str(path), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(binary), *map(str, key.shape), *map(str, value.shape),
                    str(key.nbytes), str(value.nbytes)], check=True)


@pytest.mark.parametrize("entry", [
    "launch_owned_paired_kv_insert",
    "rpu_launch_lingbot2_insert_kv_t51_spm_kernel",
])
def test_paired_kv_stages_typed_owners_before_graph_lookup(tmp_path, entry):
    """Execute the production availability/staging/lookup sequence on CPU."""
    from test_fmb_standalone_layout_rebuild import _definition

    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = (ROOT / "src/ops/rpu_llama_kvcache.cpp").read_text()
    body = _definition(source, f"void {entry}(")
    # Include the whole resource admission and typed owner handoff. A raw
    # graph lookup moved before its receipt must still enter this slice.
    first = min(body.index("    TORCH_CHECK(KernelCache::instance().get("),
                body.index("    Kernel_t* kernel = GET_KERNEL("))
    handoff = body[first:body.index("    rpu_ddr_flush(", first)]
    program = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(ok, ...) do {if(!(ok))throw std::runtime_error(#ok);}while(false)
enum class KernelId {Paired};
struct Kernel_t {} kernel_instance;
struct KernelCache {
    bool loaded=true;
    static KernelCache& instance(){static KernelCache cache;return cache;}
    Kernel_t* get(KernelId)const{return loaded?&kernel_instance:nullptr;}
};
enum class GraphDdrRegisterRole {KeyCache,ValueCache};
enum class GraphDdrRegisterAccess {Write};
enum class GraphDdrRegisterEncoding {DevAddrShift8LoHi};
struct GraphDdrRegisterAbi {
    GraphDdrRegisterRole role; GraphDdrRegisterAccess access;
    GraphDdrRegisterEncoding encoding; int lo,hi;
};
struct GraphDdrRegisterOperandSpec {GraphDdrRegisterAbi abi;int owner;};
struct RpuKernelGraph {
    bool pending=false,deny=false;int stages=0,lookups=0,keepalives=0;
    static RpuKernelGraph& active(){static RpuKernelGraph graph;return graph;}
    int stage_kernel_ddr_registers(KernelId,const std::vector<GraphDdrRegisterOperandSpec>& operands){
        TORCH_CHECK(!pending&&!deny);
        assert(operands.size()==2);
        for(int i=0;i<2;++i){const auto& op=operands[i];
            assert(op.owner==i+1 && op.abi.lo==8+4*i && op.abi.hi==9+4*i);
            assert(op.abi.role==(i?GraphDdrRegisterRole::ValueCache:GraphDdrRegisterRole::KeyCache));
            assert(op.abi.access==GraphDdrRegisterAccess::Write);
            assert(op.abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
        }
        pending=true;++stages;return stages;
    }
    void keep_alive(int owner){assert(owner==1||owner==2);++keepalives;}
};
Kernel_t* get_graph_kernel(KernelId){auto& graph=RpuKernelGraph::active();
    TORCH_CHECK(graph.pending);graph.pending=false;++graph.lookups;return &kernel_instance;}
#define GET_KERNEL(id) get_graph_kernel(id)
bool queue_available=true;int queue_instance;
int* GET_QUEUE(int cores){assert(cores==8);return queue_available?&queue_instance:nullptr;}
void handoff(bool lingbot_m224){
    const KernelId kernel_id=KernelId::Paired,kKernelId=KernelId::Paired;
    const int num_cores=8,k_cache=1,v_cache=2;
    __HANDOFF__
    (void)queue;(void)kernel;
}
template<class F>void rejected(F fn){auto& graph=RpuKernelGraph::active();
    const int stages=graph.stages,lookups=graph.lookups,keepalives=graph.keepalives;
    bool caught=false;try{fn();}catch(const std::runtime_error&){caught=true;}
    assert(caught&&!graph.pending&&graph.stages==stages&&graph.lookups==lookups&&graph.keepalives==keepalives);
}
int main(){
    auto& graph=RpuKernelGraph::active();
    // The graph double enforces the real census failure, independently of the
    // production ordering: obtaining a register kernel without staging fails.
    rejected([&]{get_graph_kernel(KernelId::Paired);});
    for(bool lingbot:{false,true}){
        handoff(lingbot);assert(!graph.pending&&graph.stages==graph.lookups);
        KernelCache::instance().loaded=false;rejected([&]{handoff(lingbot);});
        KernelCache::instance().loaded=true;
        queue_available=false;rejected([&]{handoff(lingbot);});queue_available=true;
        graph.deny=true;rejected([&]{handoff(lingbot);});graph.deny=false;
    }
    assert(graph.stages==2&&graph.lookups==2);
}
'''.replace("__HANDOFF__", handoff)
    path = tmp_path / "paired_kv_handoff.cpp"
    path.write_text(program)
    binary = path.with_suffix("")
    built = subprocess.run([compiler, "-std=c++17", str(path), "-o", str(binary)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
