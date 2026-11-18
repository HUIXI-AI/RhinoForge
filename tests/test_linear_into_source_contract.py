from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_conv_slab_uses_compact_dma_and_new_abi_row_pitch(tmp_path):
    """Execute the merged geometry, DMA call and register writer on the host."""
    import subprocess
    from test_fmb_standalone_layout_rebuild import _definition

    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    body = _definition(source, "void rpu_launch_linear_ddr_kernel(")
    geometry = body[body.index("  const size_t a_dwidth ="):body.index("  c10::Half *xp =")]
    start = body.index("      rpu_launch_ddr_broadcast_spm_dma(xp + row0 * in_pitch,")
    upload = body[start:body.index(";", start) + 1]
    start = body.index("      rpu_launch_linear_spm_to_spm_kernel(")
    launch = body[start:body.index(";", start) + 1]
    shared = _definition(source, "void rpu_launch_linear_spm_to_spm_kernel(")
    shared_geometry = shared[shared.index("  size_t dwidth ="):shared.index("  auto tile =")]
    start = shared.index("  rpu_pl_tiling::set_parallel_linear_regs(")
    registers = shared[start:shared.index("  register_writer.write(*kernel);", start)]
    program = r'''
#include <cassert>
#include <cstdint>
#include <map>
#include <vector>
#include "TILING_HEADER"
namespace c10 { using Half = uint16_t; }
struct Tensor {
 size_t rows,columns;
 size_t size(int axis)const{return axis?columns:rows;}
 size_t element_size()const{return 2;}
};
struct LinearConv1dSlab {int64_t channels,taps,dilation;};
struct Kernel {
 std::map<int,uint16_t> regs;
 void set_regs(int slot,uint16_t value){regs[slot]=value;}
 uint32_t u32(int slot)const{return uint32_t(regs.at(slot))|(uint32_t(regs.at(slot+1))<<16);}
};
const uint16_t* uploaded=nullptr;size_t uploaded_elements=0;
void rpu_launch_ddr_broadcast_spm_dma(const uint16_t* src,size_t count,uint32_t addr,int cores){
 assert(addr==4096&&cores==8);uploaded=src;uploaded_elements=count;
}
Kernel actual;
void rpu_launch_linear_spm_to_spm_kernel(uint32_t input_spm_addr,const Tensor& weight,
 uint32_t output_spm_addr,int64_t M,int64_t N,int64_t K,int partition,int num_cores,
 uint32_t bias_spm_addr,uint32_t input_m_byte_step=0){
 SHARED_GEOMETRY
 actual=Kernel{};auto* kernel=&actual;
 REGISTERS
}
void run(bool slab,size_t channels,size_t rows,size_t row0,size_t cur_m){
 const int partition=1,num_cores=8;
 const int64_t logical_m=rows,logical_k=slab?7*channels:channels,halo=slab?6:0;
 Tensor input{rows+size_t(halo),channels},output{rows,1024};
 const LinearConv1dSlab description{int64_t(channels),7,1};
 const LinearConv1dSlab* conv_slab=slab?&description:nullptr;
 GEOMETRY
 assert(m==rows&&k==size_t(logical_k)&&n==1024&&in_pitch==channels);
 assert(row0+cur_m<=rows);
 std::vector<uint16_t> storage(input.rows*input.columns);
 auto* xp=storage.data();const uint32_t x_addr=4096,y_addr=8192,b_addr=0;
 UPLOAD
 assert(uploaded==storage.data()+row0*channels);
 assert(uploaded_elements==(cur_m+(slab?6:0))*channels);
 assert(uploaded+uploaded_elements<=storage.data()+storage.size());
 Tensor weight{n,k};
 LAUNCH
 assert(actual.u32(16)==channels*2&&actual.u32(18)==256);
 assert(actual.regs.at(8)==1&&actual.regs.at(9)==cur_m);
 assert(actual.regs.at(12)==logical_k&&actual.regs.at(13)==logical_k/16);
 assert(actual.regs.at(26)==2&&actual.regs.at(27)==2);
 // The same public helper still supplies the full local-K pitch to ordinary
 // callers that omit the optional conv pitch.
 rpu_launch_linear_spm_to_spm_kernel(x_addr,weight,y_addr,cur_m,n,k,
                                    partition,num_cores,b_addr);
 assert(actual.u32(16)==logical_k*2);
}
int main(){
 for(bool slab:{false,true})for(size_t channels:{16u,64u,768u}){
  run(slab,channels,1,0,1);
  run(slab,channels,17,0,5);
  run(slab,channels,17,16,1);
 }
}
'''
    program = program.replace("TILING_HEADER", str(ROOT / "src/ops/rpu_linear_tiling.h"))
    program = program.replace("SHARED_GEOMETRY", shared_geometry)
    program = program.replace("GEOMETRY", geometry).replace("UPLOAD", upload)
    program = program.replace("LAUNCH", launch).replace("REGISTERS", registers)
    cpp, binary = tmp_path / "conv_slab.cpp", tmp_path / "conv_slab"
    cpp.write_text(program)
    subprocess.run(["g++", "-std=c++17", "-O0", str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_public_linear_into_validates_every_raw_ddr_operand():
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text(encoding="utf-8")
    body = source[
        source.index("void rpu_linear_into("):
        source.index("// O_proj Linear: SPM input -> DDR output")
    ]

    for operand in ("input", "weight", "output"):
        assert f"{operand}.scalar_type() == at::kHalf" in body
        assert f"{operand}.is_contiguous()" in body
        assert f"{operand}.device().type() == at::kPrivateUse1" in body

    assert "bias.scalar_type() == at::kHalf" in body
    assert "bias.dim() == 1 && bias.numel() == out_features" in body
    assert "bias.is_contiguous()" in body
    assert "bias.device().type() == at::kPrivateUse1" in body

    launch = body.index("rpu_launch_linear_ddr_kernel(")
    for check in (
        "weight.scalar_type() == at::kHalf",
        "weight.is_contiguous()",
        "bias.dim() == 1 && bias.numel() == out_features",
        "bias.is_contiguous()",
    ):
        assert body.index(check) < launch


def test_ddr_linear_publications_cover_dirty_inputs_before_dma(tmp_path):
    """Compile real publication blocks and DMA wrappers against separate caches.

    Only allocation, queue submission and the GEMM consumer are stand-ins. Graph
    REPLAY executes the BUILD DMA nodes, while real wrappers re-emit/validate the
    same nodes and publish current CPU writes before deferred execution.
    """
    import shutil
    import subprocess

    from test_fmb_standalone_layout_rebuild import _definition

    compiler = shutil.which("g++")
    assert compiler is not None, "publication regression requires a C++ compiler"
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    launcher = _definition(source, "void rpu_launch_linear_ddr_kernel(")
    publication = launcher[launcher.index("  uint32_t b_addr = 0;"):
        launcher.index("  for (size_t row0 = 0;")]
    loop = launcher.index("  for (size_t row0 = 0;")
    upload = launcher[launcher.index("    if (partition == 0) {", loop):
                      launcher.index("    if (w8a16 || w4a16 || !use_acc32)", loop)]
    dma_source = (ROOT / "src/ops/rpu_memcpy.cpp").read_text()
    wrappers = "\n".join(_definition(dma_source, prefix) for prefix in (
        "void rpu_launch_ddr_broadcast_spm_dma_immediate(",
        "void rpu_launch_ddr_scatter_spm_dma_immediate(",
        "static inline void emit_scatter_pre_barriers(",
        "static inline void emit_scatter_post_barriers(",
        "void rpu_launch_ddr_broadcast_spm_dma(c10::Half*",
        "void rpu_launch_ddr_scatter_spm_dma(c10::Half*",
        "void rpu_launch_ddr_scatter_spm_dma_mutable(const uint64_t*",
    ))
    profile = (ROOT / "src/core/rpu_profile.h").read_text()
    macros = "\n".join(
        profile[profile.index(f"#define {name}("):].split("\n\n", 1)[0]
        for name in ("rpu_ddr_flush", "rpu_ddr_flush_force")
    )
    harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
namespace c10 { using Half = float; }
bool g_rpu_ddr_flush_enabled = true, g_rpu_ddr_flush_force_enabled = true;
struct Memory { std::vector<float> cpu, device; explicit Memory(size_t n):cpu(n),device(n){} };
std::vector<std::shared_ptr<Memory>> memories;
std::vector<void*> flushed;
std::vector<size_t> flushed_bytes;
bool publication_only=false;
std::map<uint64_t,std::vector<float>> spm;
std::pair<Memory*,size_t> lookup(uint64_t address) {
    for (auto& memory : memories) {
        auto begin = reinterpret_cast<uint64_t>(memory->cpu.data());
        if (address >= begin && address < begin + memory->cpu.size()*sizeof(float))
            return {memory.get(), (address-begin)/sizeof(float)};
    }
    throw std::runtime_error("unowned DDR address");
}
namespace rhino_lkn {
void RpuDdrFlush(void* pointer) {
    flushed.push_back(pointer);
    flushed_bytes.push_back(0);
    if(publication_only) return;
    auto [memory,offset] = lookup(reinterpret_cast<uint64_t>(pointer));
    std::copy(memory->cpu.begin()+offset,memory->cpu.end(),memory->device.begin()+offset);
}
void RpuDdrFlush(void* pointer,size_t bytes) {
    flushed.push_back(pointer);
    flushed_bytes.push_back(bytes);
    if(publication_only) return;
    auto [memory,offset] = lookup(reinterpret_cast<uint64_t>(pointer));
    assert(bytes%sizeof(float)==0 && offset+bytes/sizeof(float)<=memory->cpu.size());
    std::copy_n(memory->cpu.begin()+offset,bytes/sizeof(float),memory->device.begin()+offset);
}
}
uint64_t RpuGetDevAddr(void* pointer) {
    if(publication_only) throw std::runtime_error("publication boundary");
    return reinterpret_cast<uint64_t>(pointer);
}
namespace at {
struct Tensor {
    std::shared_ptr<Memory> memory;
    Tensor()=default;
    explicit Tensor(size_t n):memory(std::make_shared<Memory>(n)){memories.push_back(memory);}
    template<class T> T* data_ptr() const {return reinterpret_cast<T*>(memory->cpu.data());}
    int options() const {return 0;}
};
Tensor empty(std::initializer_list<int64_t> shape,int) {
    return Tensor(std::accumulate(shape.begin(),shape.end(),size_t(1),std::multiplies<size_t>()));
}
}
struct Node {
    uint64_t fixed_src=0,dst=0;
    const uint64_t* live_src=nullptr;
    int64_t offset=0;
    size_t bytes=0;
    std::function<void()> consumer;
};
void execute(const Node& node) {
    if (node.consumer) { node.consumer(); return; }
    auto [memory,offset] = lookup(node.live_src ? *node.live_src+node.offset : node.fixed_src);
    assert(offset+node.bytes/sizeof(float) <= memory->device.size());
    spm[node.dst] = std::vector<float>(memory->device.begin()+offset,
                                     memory->device.begin()+offset+node.bytes/sizeof(float));
}
namespace graph_dma {
enum class Mode { Eager, Build, Replay };
Mode mode=Mode::Eager;
size_t cursor=0;
std::vector<Node> nodes;
bool active(){return mode!=Mode::Eager;}
void emit(Node node) {
    if (mode==Mode::Eager) {execute(node);return;}
    if (mode==Mode::Build) nodes.push_back(std::move(node));
    else {
        const auto& built=nodes.at(cursor++);
        assert(built.fixed_src==node.fixed_src && built.live_src==node.live_src &&
               built.offset==node.offset && built.dst==node.dst && built.bytes==node.bytes &&
               bool(built.consumer)==bool(node.consumer));
    }
}
void emit_fixed(uint64_t src,uint64_t dst,size_t bytes,int){emit({src,dst,nullptr,0,bytes,{}});}
void emit_mutable_src(const uint64_t* src,int64_t offset,uint64_t dst,size_t bytes,int){
    emit({0,dst,src,offset,bytes,{}});
}
void emit_barrier(int,int){}
void finish() {
    if (!active()) return;
    assert(mode!=Mode::Replay || cursor==nodes.size());
    for (const auto& node:nodes) execute(node);
}
}
struct RpuKernelGraph {
    static void check_direct_immediate_dma_allowed(const char*) {assert(!graph_dma::active());}
};
struct Queue_t {std::vector<Node> nodes;void add_barrier(int,int){} };
struct QueueCache {
    Queue_t queue;
    static QueueCache& instance(){static QueueCache cache;return cache;}
    Queue_t* get(uint8_t){return &queue;}
};
void rpu_add_dma_checked(Queue_t& q,uint64_t src,uint64_t dst,size_t bytes,int,const char*){
    q.nodes.push_back({src,dst,nullptr,0,bytes,{}});
}
void submit_immediate_batch(Queue_t* q,const char*){
    for(const auto& node:q->nodes) execute(node);
    q->nodes.clear();
}
uint64_t spm_unified_to_per_core_abs(int core,uint32_t address){return uint64_t(core)*1048576+address;}
struct Allocator {uint32_t addr(int,uint32_t n){return n;} uint32_t alloc_temporary(size_t){return 256;} } SPM_ALLOC;
float expected_input=0,expected_bias=0,expected_weight=0;
int consumed=0;
void consume(uint32_t x_addr,uint32_t b_addr,size_t rows,size_t row0,size_t local_k,
             size_t local_n,int cores,int partition,bool has_bias,const at::Tensor& weight){
    graph_dma::emit({0,0,nullptr,0,0,[=]{
        ++consumed;
        assert(weight.memory->device[0]==expected_weight);
        for(int core=0;core<cores;++core) {
            const auto& x=spm.at(spm_unified_to_per_core_abs(core,x_addr));
            assert(x.size()==rows*local_k);
            for(size_t i=0;i<x.size();++i) {
                float expected=expected_input;
                if(expected) expected+=float((row0+i/local_k)*10+i%local_k%4);
                assert(x[i]==expected);
            }
            if(has_bias && partition!=0) {
                const auto& b=spm.at(spm_unified_to_per_core_abs(core,b_addr));
                assert(b.size()==local_n);
                for(float value:b) assert(value==expected_bias);
            }
        }
    }});
}
'''
    transport = r'''
void transport(const at::Tensor& input,const at::Tensor& weight,const at::Tensor& bias,
               bool has_bias,int partition,const uint64_t* live_bias_base,
               size_t m,size_t m_chunk) {
    const size_t k=32,n=32,num_cores=2,a_dwidth=sizeof(c10::Half);
    const size_t local_k=partition==0?k/num_cores:k;
    // Plain-GEMM geometry: a row spans the whole local K and no halo rows are
    // staged. Only the conv-slab route narrows the pitch to one channel row and
    // uploads the extra rows the chunk's last windows reach into.
    const size_t in_pitch=local_k;
    const int64_t halo=0;
    const size_t local_n=partition==0?n:n/num_cores;
    const bool kernel_applies_bias=has_bias && partition!=0;
    auto xp=input.data_ptr<c10::Half>();
    auto wp=weight.data_ptr<c10::Half>();
    auto bp=has_bias?bias.data_ptr<c10::Half>():nullptr;
    const uint32_t x_addr=4096;
PUBLICATION
    for(size_t row0=0;row0<m;row0+=m_chunk) {
        const size_t cur_m=std::min(m-row0,m_chunk);
UPLOAD
        consume(x_addr,b_addr,cur_m,row0,local_k,local_n,num_cores,partition,has_bias,weight);
    }
}
'''.replace("PUBLICATION", publication).replace("UPLOAD", upload)
    main = r'''
int main(){
    int cases=0;
    for(bool normal:{false,true}) for(bool force:{false,true})
    for(bool captured:{false,true}) for(int bias_mode:{0,1,2})
    for(int partition:{0,1}) for(size_t m:{size_t(1),size_t(5)}) {
        if(captured && partition==0) continue; // production rejects row Graphs
        if(partition==0 && bias_mode==2) continue;
        memories.clear();spm.clear();graph_dma::nodes.clear();
        at::Tensor input(m*32),weight(32*32),bias_a(32),bias_b(32);
        uint64_t live_bias=0;
        const size_t chunk=m==1?1:2,chunks=(m+chunk-1)/chunk;
        g_rpu_ddr_flush_enabled=normal;g_rpu_ddr_flush_force_enabled=force;
        for(int leg:{0,1,2}) {
            const float value=leg==1?20:10,bias_value=leg==1?7:3;
            auto& bias=leg==1 && bias_mode==2?bias_b:bias_a;
            // Raw CPU writes: no tensor copy/version/publication occurs here.
            for(size_t row=0;row<m;++row) for(size_t col=0;col<32;++col)
                input.memory->cpu[row*32+col]=value+float(row*10+col%4);
            std::fill(weight.memory->cpu.begin(),weight.memory->cpu.end(),value+2);
            std::fill(bias.memory->cpu.begin(),bias.memory->cpu.end(),bias_value);
            live_bias=RpuGetDevAddr(bias.data_ptr<float>());
            flushed.clear();consumed=0;graph_dma::cursor=0;
            graph_dma::mode=captured?(leg==0?graph_dma::Mode::Build:graph_dma::Mode::Replay):graph_dma::Mode::Eager;
            expected_input=(partition==0?(normal||force):normal)?value:0;
            expected_weight=normal?value+2:0;
            expected_bias=normal?bias_value:0;
            transport(input,weight,bias,bias_mode!=0,partition,bias_mode==2?&live_bias:nullptr,m,chunk);
            if(captured) assert(consumed==0);
            graph_dma::finish();assert(consumed==int(chunks));
            auto count=[&](void* ptr){return std::count(flushed.begin(),flushed.end(),ptr);};
            size_t input_flushes=0;
            for(void* ptr:flushed) if(lookup(reinterpret_cast<uint64_t>(ptr)).first==input.memory.get()) ++input_flushes;
            assert(input_flushes==(normal?(partition==0?1:chunks):0));
            assert(count(weight.data_ptr<float>())==int(normal));
            assert(count(bias.data_ptr<float>())==int(normal && bias_mode!=0));
            assert(g_rpu_ddr_flush_enabled==normal && g_rpu_ddr_flush_force_enabled==force);
            ++cases;
        }
    }
    int range_cases=0;
    for(bool normal:{false,true}) for(bool force:{false,true})
    for(int cores:{1,8}) for(int64_t stride:std::vector<int64_t>{-16,0,16,32,64})
    for(bool broadcast:{false,true}) {
        memories.clear();spm.clear();graph_dma::mode=graph_dma::Mode::Eager;
        at::Tensor input(256);
        auto* pointer=input.data_ptr<float>()+128;
        g_rpu_ddr_flush_enabled=normal;g_rpu_ddr_flush_force_enabled=force;
        for(float value:{17.0f,29.0f,17.0f}) {
            std::fill(input.memory->cpu.begin(),input.memory->cpu.end(),value);
            flushed.clear();flushed_bytes.clear();
            if(broadcast) rpu_launch_ddr_broadcast_spm_dma_immediate(pointer,8,4096,cores);
            else rpu_launch_ddr_scatter_spm_dma_immediate(pointer,8,stride,4096,cores);
            assert(flushed.size()==size_t(normal));
            const bool sized=broadcast||stride>=0;
            const size_t bytes=broadcast?32:sized?size_t((cores-1)*stride+32):0;
            if(normal) assert(flushed_bytes[0]==(sized?bytes:0));
            for(size_t i=0;i<input.memory->device.size();++i) {
                const bool published=normal && i>=128 && (!sized||i<128+bytes/sizeof(float));
                assert(input.memory->device[i]==(published?value:0));
            }
            for(int core=0;core<cores;++core) {
                const auto& copied=spm.at(spm_unified_to_per_core_abs(core,4096));
                assert(copied.size()==8);
                const int64_t first=128+(broadcast?0:core*stride/int64_t(sizeof(float)));
                for(size_t i=0;i<copied.size();++i)
                    assert(copied[i]==input.memory->device.at(first+i));
            }
            assert(g_rpu_ddr_flush_enabled==normal && g_rpu_ddr_flush_force_enabled==force);
            ++range_cases;
        }
    }
    // Arithmetic-only malformed ranges stop immediately after publication;
    // no queue or fake memory access can turn overflow into a plausible DMA.
    int fallback_cases=0;
    publication_only=true;
    for(bool normal:{false,true}) for(bool force:{false,true})
    for(int kind:{0,1,2}) {
        g_rpu_ddr_flush_enabled=normal;g_rpu_ddr_flush_force_enabled=force;
        flushed.clear();flushed_bytes.clear();
        auto* pointer=reinterpret_cast<float*>(kind==2?uintptr_t(4096):
            std::numeric_limits<uintptr_t>::max()-15);
        bool stopped=false;
        try {
            if(kind==0) rpu_launch_ddr_broadcast_spm_dma_immediate(pointer,8,4096,8);
            else rpu_launch_ddr_scatter_spm_dma_immediate(pointer,8,
                kind==2?std::numeric_limits<int64_t>::max()-15:32,4096,8);
        } catch(const std::runtime_error& error) {
            stopped=std::string(error.what())=="publication boundary";
        }
        assert(stopped && flushed.size()==size_t(normal));
        if(normal) assert(flushed_bytes[0]==0);
        ++fallback_cases;
    }
    std::cout<<cases<<" publication cases, "<<range_cases
             <<" source-range cases, "<<fallback_cases<<" overflow cases passed\n";
}
'''
    cpp = tmp_path / "linear_publication.cpp"
    cpp.write_text(harness + macros + "\n" + wrappers + transport + main)
    binary = tmp_path / "linear_publication"
    built = subprocess.run([compiler, "-std=c++17", "-O2", str(cpp), "-o", str(binary)],
                           text=True, capture_output=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(binary)], text=True, capture_output=True)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == "192 publication cases, 240 source-range cases, 12 overflow cases passed"
