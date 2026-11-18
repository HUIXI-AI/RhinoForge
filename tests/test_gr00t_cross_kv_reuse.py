"""Exercise the native cross-KV schedule and stale-descriptor rejection on host."""
from pathlib import Path
import re
import shutil
import subprocess

from test_siglip_attention_dry_live_contract import _function


ROOT = Path(__file__).resolve().parents[1]


def test_cross_kv_reuse_refreshes_each_graph_and_rejects_schedule_drift(tmp_path):
    source = (ROOT / "src/fused/rpu_gr00t_dit_model.cpp").read_text()
    methods = "\n".join(_function(source, marker) for marker in (
        "    bool cross_kv_reuse_profile(",
        "    int64_t cross_kv_schedule(",
        "    int64_t cross_kv_bank_stride(",
        "    int64_t cross_kv_bank_bytes(",
        "    uint32_t cross_kv_bank_offset(",
        "    void append_cross_kv_buffers(",
        "    std::vector<int64_t> cross_kv_schedule_arguments(",
        "    bool cross_kv_projection_needed(",
        "    bool cross_v_transpose_profile(",
        "    std::vector<int64_t> cross_v_transpose_arguments(",
        "    bool cross_v_transpose_reused(",
    ))
    constants = re.search(
        r"constexpr int64_t GR00T_DIT_CROSS_KV_SCHEDULE_SITE.*?\n};",
        source, re.S)[0]
    env_reader = _function(source, "bool gr00t_cross_kv_reuse_opt_in()")
    env_reader += "\n" + _function(source, "bool gr00t_cross_v_transpose_reuse_opt_in()")
    constants += "\n" + re.search(r"constexpr int64_t GR00T_DIT_CROSS_V_TRANSPOSE_SCHEDULE_SITE = .*?;", source)[0]
    constants += "\n" + "\n".join(re.findall(
        r"constexpr int64_t GR00T_DIT_CROSS_(?:RAW|DDR)_ATTENTION_SITE = .*?;", source))
    cross = _function(source, "    void build_cross_attn_block(")
    selection = cross[cross.index("        int64_t attention_site ="):
                      cross.index("        const bool raw_bank =")]
    operands = cross[cross.index("        const bool raw_bank ="):
                     cross.index("        auto& k_cache =")]
    insert = _function(source, "    KvInsertSegmentPlan consume_kv_insert_plan(")
    tail = insert[insert.index("        const int64_t local_qhd ="):
                  insert.rindex("        return plan;")]
    # Compile the actual policy/consumer, not a Python translation of its logic.
    cpp = r'''
#include <cassert>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#define NUM_CORES 8
#define DWIDTH 2
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
int64_t Align(int64_t n, int64_t a) { return (n+a-1)/a*a; }
int64_t gr00t_pad16(int64_t n) { return Align(n,16); }
struct SpmAllocator { static constexpr int64_t SPM_USABLE=8387584; };
enum class AttentionExecutionPolicy { DDR_KV, SPM_KV_BY_MHA };
enum class FmbRouteFamily { GRAPH_SCHEDULE, ATTENTION };
enum class StorageClass { Temp };
enum class BufferScope { LayerWide };
struct BufferDecl { const char* name; int64_t size; int phase_start,phase_end;
    StorageClass storage; int per_layer; void* preload; BufferScope scope; };
struct LayoutContext { int64_t chunk_size=41,batch_size=1;
    AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::DDR_KV; };
struct PadPlan { int64_t physical; int64_t physical_rows() const { return physical; } };
std::vector<std::pair<uint32_t,int64_t>> memsets;
void rpu_launch_memset_spm_multicore(uint32_t address,int64_t elements,int=8) { memsets.push_back({address,elements}); }
__CONSTANTS__
__ENV_READER__
struct ChunkInfo { int64_t len=41, idx=0; };
struct Context {
    bool complete=true;
    int64_t batch_size=1, body_iter=0, selector=0;
    std::vector<int64_t> arguments;
    mutable int consumed=0;
    struct Route { FmbRouteFamily family; int64_t site_id, invocation;
                   AttentionExecutionPolicy policy; };
    std::vector<Route> routes;
    const Context& physical_manifest() const { return *this; }
    AttentionExecutionPolicy physical_attention_policy_for_site(int64_t site, int64_t invocation) const {
        for (const auto& route : routes)
            if (route.family==FmbRouteFamily::ATTENTION && route.site_id==site && route.invocation==invocation)
                return route.policy;
        throw std::runtime_error("missing attention site");
    }
    bool has_complete_physical_manifest() const { return complete; }
    void consume_physical_route(FmbRouteFamily, int64_t site, int64_t resolved,
            int64_t flags, const std::vector<int64_t>& args, int64_t invocation) const {
        TORCH_CHECK(site==GR00T_DIT_CROSS_KV_SCHEDULE_SITE && flags==0 && invocation==0);
        TORCH_CHECK(selector==resolved && arguments==args);
        ++consumed;
    }
};
struct Model {
    const bool cross_kv_reuse_opt_in_=gr00t_cross_kv_reuse_opt_in();
    const bool cross_v_transpose_reuse_opt_in_=gr00t_cross_v_transpose_reuse_opt_in();
    bool denoise_ready_=true, denoise_mode_=true, loop_mode_=true, z2_bound_=false;
    int64_t num_steps_=4, z2_prepared_seq_len_=0, cross_dim_=2048, vl_seq_=82;
    int64_t layers=32, h=1536, ff=6144, q=32, kv=32, hd=48;
    struct LayerWeights { bool is_cross; };
    std::vector<LayerWeights> layer_weights_=std::vector<LayerWeights>(32);
    Model() { for (int i=0;i<32;++i) layer_weights_[i].is_cross=i%2==0; }
    Context context;
    const Context& ctx() const { return context; }
    int64_t num_layers() const { return layers; }
    int64_t hidden_size() const { return h; }
    int64_t intermediate_size() const { return ff; }
    int64_t num_q_heads() const { return q; }
    int64_t num_kv_heads() const { return kv; }
    int64_t head_dim() const { return hd; }
    int64_t attn_tp() const { return NUM_CORES; }
    int num_cores() const { return NUM_CORES; }
    uint32_t addr(int, const char* name) const {
        const std::string slot(name);
        if (slot=="cross_k_bank") return 0x10200000;
        if (slot=="cross_v_bank") return 0x10400000;
        return slot=="k" ? 0x10010000 : 0x10020000;
    }
    __METHODS__
    AttentionExecutionPolicy cross_policy(const ChunkInfo& chunk) const {
        __SELECTION__
        return attention_policy;
    }
    std::pair<uint32_t,uint32_t> cross_operands(const ChunkInfo& chunk, int layer_idx,
            AttentionExecutionPolicy attention_policy) const {
        const int64_t seq_len=chunk.len;
        __OPERANDS__
        return {k_spm,v_spm};
    }
    void clear_tail(int64_t true_seq, const PadPlan& plan, uint32_t k_spm=0, uint32_t v_spm=0) {
        __TAIL__
    }
    void record(const ChunkInfo& c, bool raw=false) {
        context.selector=cross_kv_schedule(c.len,context.batch_size,raw);
        context.arguments=cross_kv_schedule_arguments(c.len,context.batch_size);
    }
};
int main() {
    const auto ddr=AttentionExecutionPolicy::DDR_KV, raw=AttentionExecutionPolicy::SPM_KV_BY_MHA;
    ChunkInfo chunk;
    unsetenv("RPU_GR00T_CROSS_V_TRANSPOSE_REUSE");
    unsetenv("RPU_GR00T_CROSS_KV_REUSE");
    Model off;
    setenv("RPU_GR00T_CROSS_KV_REUSE","1",1);
    Model model;
    off.context.body_iter=3;
    assert(off.cross_kv_projection_needed(chunk,ddr) && off.context.consumed==0);
    setenv("RPU_GR00T_CROSS_KV_REUSE","0",1);
    model.record(chunk);
    assert(model.context.selector==2); // cold owner does not reread the environment
    model.context.routes={{FmbRouteFamily::ATTENTION,777,0,raw},
        {FmbRouteFamily::ATTENTION,GR00T_DIT_CROSS_DDR_ATTENTION_SITE,0,ddr}};
    assert(model.cross_policy(chunk)==ddr); // an unrelated raw self-attention site cannot enable cross raw
    model.context.routes.back()={FmbRouteFamily::ATTENTION,GR00T_DIT_CROSS_RAW_ATTENTION_SITE,0,raw};
    assert(model.cross_policy(chunk)==raw);
    for (int request=0;request<3;++request) {
        for (int body=0;body<4;++body) {
            model.context.body_iter=body;
            assert(model.cross_kv_projection_needed(chunk,ddr)==(body==0));
        }
    }
    model.record(chunk,true);
    assert(model.context.selector==3);
    for (int request=0;request<3;++request) {
        for (int body=0;body<4;++body) {
            model.context.body_iter=body;
            assert(model.cross_kv_projection_needed(chunk,raw)==(body==0));
        }
    }
    LayoutContext layout;
    std::vector<BufferDecl> buffers;
    model.append_cross_kv_buffers(buffers,layout);
    assert(buffers.empty()); // pure DDR never pays for the raw banks
    layout.attention_policy=raw;
    off.append_cross_kv_buffers(buffers,layout);
    assert(buffers.empty());
    for (auto profile : {std::pair{4,82},std::pair{4,148},std::pair{4,256},std::pair{4,193},std::pair{4,225},std::pair{4,289},std::pair{4,238},std::pair{4,270},std::pair{4,334},std::pair{10,334},std::pair{10,1},std::pair{10,97},std::pair{10,256},std::pair{10,169},std::pair{10,214}}) {
        model.num_steps_=profile.first;const int64_t rows=profile.second;model.vl_seq_=rows;
        for (auto policy : {ddr,raw}) {
            model.record(chunk,policy==raw);
            for (int request=0;request<3;++request)for (int body=0;body<model.num_steps_;++body) {
                model.context.body_iter=body;
                assert(model.cross_kv_projection_needed(chunk,policy)==(body==0));
            }
        }
        buffers.clear();
        model.append_cross_kv_buffers(buffers,layout);
        const int64_t stride=Align(rows,16)*192*2, bytes=16*stride;
        if (rows==169) assert(Align(rows,16)==176 && stride==67584 && bytes==1081344 && model.context.body_iter==9);
        if (rows==214) assert(stride==86016 && bytes==1376256 && model.context.body_iter==9);
        assert(buffers.size()==2 && buffers[0].size==bytes && buffers[1].size==bytes);
        assert(buffers[0].phase_start==0 && buffers[0].phase_end==0 &&
               buffers[0].scope==BufferScope::LayerWide);
        for (int layer=0;layer<32;layer+=2) {
            const uint32_t offset=model.cross_kv_bank_offset(layer);
            assert(offset==layer/2*stride && offset%256==0 && offset+stride<=bytes);
            const auto operands=model.cross_operands(chunk,layer,raw);
            assert(operands.first==model.addr(0,"cross_k_bank")+offset &&
                   operands.second==model.addr(0,"cross_v_bank")+offset);
            memsets.clear();
            model.clear_tail(rows,{Align(rows,16)},operands.first,operands.second);
            assert(memsets[0].first==operands.first+rows*192*2 &&
                   memsets[1].first==operands.second+rows*192*2 &&
                   memsets[0].second==(Align(rows,16)-rows)*192);
            assert(memsets[0].first+memsets[0].second*2==operands.first+stride);
        }
    }
    // A supported ten-step length still needs its own bank-stride descriptor.
    for (auto lengths : {std::pair{169,214},std::pair{214,169}}) {
        model.num_steps_=10;model.vl_seq_=lengths.first;model.record(chunk,true);
        model.vl_seq_=lengths.second;const int consumed=model.context.consumed;
        try { model.cross_kv_projection_needed(chunk,raw); assert(false); }
        catch (const std::runtime_error&) {}
        assert(model.context.consumed==consumed);
    }
    model.num_steps_=4;model.vl_seq_=82;
    memsets.clear();
    model.clear_tail(41,{48}); // self attention retains the original scratch defaults
    assert(memsets[0].first==model.addr(0,"k")+41*192*2 && memsets[1].first==model.addr(0,"v")+41*192*2);
    for (int layer : {-1,1,32}) {
        try { model.cross_kv_bank_offset(layer); assert(false); } catch (const std::runtime_error&) {}
    }
    // A DDR-reuse descriptor cannot be used by raw attention.
    model.record(chunk);
    try { model.cross_kv_projection_needed(chunk,raw); assert(false); }
    catch (const std::runtime_error&) {}
    for (int mutation=0;mutation<11;++mutation) {
        Model changed=model;
        switch(mutation) {
          case 0: changed.num_steps_=3;break;
          case 1: changed.vl_seq_=148;break;
          case 2: changed.layers=31;break;
          case 3: changed.h=1024;break;
          case 4: changed.loop_mode_=false;break;
          case 5: changed.context.batch_size=2;break;
          case 6: changed.layer_weights_[0].is_cross=false;break;
          case 7: changed.context.selector=1;break;
          case 8: changed.num_steps_=10;break;
          case 9: changed.vl_seq_=214;break;
          case 10: changed.num_steps_=10;changed.vl_seq_=214;break;
        }
        try { changed.cross_kv_projection_needed(chunk,ddr); assert(false); }
        catch (const std::runtime_error&) {}
    }
    for (int fallback=0;fallback<10;++fallback) {
        Model changed=model;
        switch(fallback) {
          case 0: changed.loop_mode_=false;break;
          case 1: changed.denoise_mode_=false;break;
          case 2: changed.z2_bound_=true;break;
          case 3: changed.z2_prepared_seq_len_=82;break;
          case 4: changed.num_steps_=1;break;
          case 5: changed.vl_seq_=0;break;
          case 6: changed.context.batch_size=2;break;
          case 7: changed.num_steps_=10;changed.vl_seq_=0;break;
          case 8: changed.num_steps_=10;changed.vl_seq_=-1;break;
          case 9: changed.num_steps_=5;break;
        }
        changed.record(chunk);
        assert(changed.context.selector==1 && changed.cross_kv_projection_needed(chunk,ddr));
    }
    model.context.complete=false;
    assert(model.cross_kv_projection_needed(chunk,ddr));
    setenv("RPU_GR00T_CROSS_KV_REUSE","invalid",1);
    try { Model invalid; assert(false); } catch (const std::runtime_error&) {}
}
'''.replace("__CONSTANTS__", constants).replace("__ENV_READER__", env_reader).replace("__METHODS__", methods).replace("__SELECTION__", selection).replace("__OPERANDS__", operands).replace("__TAIL__", tail)
    compiler = shutil.which("c++")
    assert compiler, "native GR00T schedule test requires a C++ compiler"
    path, executable = tmp_path / "cross_kv.cpp", tmp_path / "cross_kv"
    path.write_text(cpp)
    result = subprocess.run([compiler, "-std=c++17", str(path), "-o", str(executable)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=10)
