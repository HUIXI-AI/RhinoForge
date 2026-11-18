"""Run the production driver with CPU tensors and deferred launch endpoints.

The device-kind check uses CPU and private state is exposed for fault injection.
The driver and shared manifest consumer remain production code; launch endpoints
are deferred CPU doubles and do not certify device kernels.
"""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_native_driver_deferred_workspace_and_retirement(tmp_path):
    torch = pytest.importorskip("torch")
    compiler = shutil.which("g++")
    assert compiler
    source = (ROOT / "src/fused/rpu_pi05_model.cpp").read_text()
    start = source.index("class Pi05Model {")
    end = source.index("\n\n// =====", start)
    body = source[start:end].replace("at::kPrivateUse1", "at::kCPU").replace("private:", "public:")
    cpp = tmp_path / "pi_native_driver.cpp"
    child_source = (ROOT / "src/fused/rpu_adarms_model.cpp").read_text()
    mask_start = child_source.index("    void prepare_attention_mask(")
    mask_end = child_source.index("\n    at::Tensor forward(", mask_start)
    mask_method = child_source[mask_start:mask_end]
    cpp.write_text(PREAMBLE + MASK_HARNESS_PREFIX + mask_method + MASK_HARNESS_SUFFIX + body + MAIN)
    torch_root = Path(torch.__file__).resolve().parent
    binary = tmp_path / "pi_native_driver"
    command = [compiler, "-std=c++17", "-O0", "-ffunction-sections", "-fdata-sections",
               f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
               "-I", str(ROOT / "src"),
               "-I", str(torch_root / "include"),
               "-I", str(torch_root / "include/torch/csrc/api/include"),
               str(cpp), str(ROOT / "src/core/fmb_three_stage_chunk_plan.cpp"),
               "-Wl,--gc-sections", "-L", str(torch_root / "lib"), "-ltorch_cpu", "-lc10",
               f"-Wl,-rpath,{torch_root / 'lib'}", "-o", str(binary)]
    built = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert built.returncode == 0, built.stdout + built.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.strip() == "PASS: actual Pi driver workspace, deferred CPU endpoints"


PREAMBLE = r'''
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include "core/fused_model_base.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <tuple>
#include <vector>
namespace rhino_lkn {
uint64_t RpuGetDevAddr(void* p) { return reinterpret_cast<uint64_t>(p); }
}
#define rpu_ddr_flush_force_sized(p, n) ((void)0)
enum class CopyKind { DDR_TO_DDR };
struct MemcpyNodeData {
    void* dst; const void* src; size_t bytes; CopyKind kind;
    uint64_t dst_dev_addr, src_dev_addr;
};
class RpuKernelGraph;
struct GraphOpStreamStamp {
    const RpuKernelGraph* graph;
    uint64_t build_generation;
    size_t position;
};
class RpuKernelGraph {
public:
    enum class State { RECORDING, REPLAYING, PASSTHROUGH, BUILT };
    static RpuKernelGraph& active() { static RpuKernelGraph graph; return graph; }
    State state() const { return status; }
    GraphOpStreamStamp op_stream_stamp() const { return {this, generation, cursor}; }
    std::weak_ptr<const uint8_t> ddr_buffer_lifetime() { return lifetime; }
    void keep_alive(const at::Tensor& t) { owners.push_back(t); }
    void begin(bool replay=false) {
        if (!replay) { ++generation; lifetime=std::make_shared<const uint8_t>(0); calls.clear(); }
        status=replay ? State::REPLAYING : State::RECORDING; cursor=0; owners.clear();
    }
    void record(const std::string& kind, std::function<void()> work) {
        if (status==State::PASSTHROUGH) { work(); return; }
        if (status==State::RECORDING) calls.push_back({kind, std::move(work)});
        else { TORCH_CHECK(cursor<calls.size() && calls[cursor].first==kind, "node drift"); calls[cursor].second=std::move(work); }
        ++cursor;
    }
    void record_branch(uint64_t key, int, const char*) {
        record("branch:"+std::to_string(key), [] {});
    }
    void capture_data(MemcpyNodeData data) {
        TORCH_CHECK(data.bytes>0 && data.bytes%32==0 && data.bytes <= (size_t{8}<<20), "DMA boundary");
        record("copy", [data] { std::memcpy(data.dst, data.src, data.bytes); });
    }
    void end() {
        TORCH_CHECK(cursor==calls.size(), "incomplete replay");
        for (const auto& call : calls) call.second();
        status=State::BUILT; owners.clear();
    }
    void clear() { lifetime.reset(); calls.clear(); owners.clear(); cursor=0; status=State::PASSTHROUGH; }
    size_t pending() const { return cursor; }
    size_t graph_size() const { return calls.size(); }
private:
    State status=State::PASSTHROUGH;
    uint64_t generation=0;
    size_t cursor=0;
    std::shared_ptr<const uint8_t> lifetime;
    std::vector<std::pair<std::string,std::function<void()>>> calls;
    std::vector<at::Tensor> owners;
};
namespace graph_dma {
bool active() { auto s=RpuKernelGraph::active().state(); return s==RpuKernelGraph::State::RECORDING || s==RpuKernelGraph::State::REPLAYING; }
}
enum class ValuOpType { ADD };
static at::Tensor borrowed(void* data, std::vector<int64_t> shape) {
    return at::from_blob(data, shape, at::TensorOptions().dtype(at::kHalf));
}
void rpu_launch_linear_ddr_kernel(const at::Tensor& input, const at::Tensor& weight,
        at::Tensor& output, const at::Tensor& bias, bool has_bias, int partition,
        const uint64_t* live_dst_base, int64_t offset, const uint64_t* live_bias,
        int cores, const at::Tensor&, std::optional<bool> acc32) {
    TORCH_CHECK(offset==0 && live_bias==nullptr && cores==8 && acc32.value(), "fixed physical projection policy");
    TORCH_CHECK(partition==1 && live_dst_base, "real Linear guard remains required");
    auto* in=input.data_ptr(); auto* w=weight.data_ptr(); auto* b=bias.defined()?bias.data_ptr():nullptr;
    auto in_shape=input.sizes().vec(), w_shape=weight.sizes().vec(), out_shape=output.sizes().vec();
    const auto output_ptr=output.data_ptr();
    RpuKernelGraph::active().record("linear", [=] {
        TORCH_CHECK(*live_dst_base==rhino_lkn::RpuGetDevAddr(output_ptr), "live destination owner drift");
        auto x=borrowed(in,in_shape).to(at::kFloat), weight=borrowed(w,w_shape).to(at::kFloat);
        auto value=at::linear(x,weight,has_bias?borrowed(b,{w_shape[0]}).to(at::kFloat):at::Tensor{}).to(at::kHalf);
        borrowed(output_ptr,out_shape).copy_(value);
    });
}
at::Tensor rpu_adarms_forward(int64_t, const at::Tensor& input, const at::Tensor& cond,
        std::vector<at::Tensor>& keys, std::vector<at::Tensor>& values,
        const std::optional<at::Tensor>&, int64_t, bool, at::IntArrayRef) {
    auto output=at::empty_like(input);
    auto* in=input.data_ptr(); auto* c=cond.data_ptr(); auto* out=output.data_ptr();
    auto shape=input.sizes().vec(); auto width=cond.numel();
    std::vector<std::pair<void*,std::vector<int64_t>>> cache_ptrs;
    for (const auto& k:keys) cache_ptrs.push_back({k.data_ptr(),k.sizes().vec()});
    for (const auto& v:values) cache_ptrs.push_back({v.data_ptr(),v.sizes().vec()});
    RpuKernelGraph::active().record("child",[=] {
        auto value=(borrowed(in,shape).to(at::kFloat)*0.25+borrowed(c,{width}).to(at::kFloat)*0.125).to(at::kHalf);
        borrowed(out,shape).copy_(value);
        for (auto& cache:cache_ptrs) borrowed(cache.first,cache.second).fill_(42);
    });
    return output;
}
void rpu_launch_eltwise_binary_sameshape_kernel(const at::Tensor& first,
        const at::Tensor& second, c10::Half dt, at::Tensor& output, ValuOpType) {
    auto* a=first.data_ptr(); auto* b=second.data_ptr(); auto* out=output.data_ptr();
    auto shape=first.sizes().vec();
    RpuKernelGraph::active().record("euler",[=] {
        borrowed(out,shape).copy_((borrowed(a,shape).to(at::kFloat)+float(dt)*borrowed(b,shape).to(at::kFloat)).to(at::kHalf));
    });
}
'''


MASK_HARNESS_PREFIX = r'''
struct PreparedMaskForTest {};
static int mask_preparations = 0;
PreparedMaskForTest sdpa_prepare_mask(const std::optional<at::Tensor>&,
        bool, int64_t, int64_t, int) {
    TORCH_CHECK(RpuKernelGraph::active().pending() == 0, "late mask host read");
    ++mask_preparations;
    return {};
}
class MaskOwner {
public:
    int num_layers() const { return 1; }
    int sdpa_stable_mask_cache() { return 0; }
    PreparedMaskForTest prepared_mask_;
    at::Tensor prepared_mask_input_ref_;
    int64_t prepared_mask_seq_q_ = 0, prepared_mask_seq_k_ = 0;
    bool prepared_mask_is_causal_ = false;
'''

MASK_HARNESS_SUFFIX = r'''
};
static MaskOwner mask_owner;
void rpu_adarms_prepare_graph_mask(int64_t, const std::optional<at::Tensor>& mask,
        int64_t rows, int64_t position, bool causal) {
    mask_owner.prepare_attention_mask(mask, rows, position, causal, true);
}
'''


MAIN = r'''
template<class F> void rejects(F&& call, const std::string& message) {
    bool rejected=false;
    try { call(); } catch(const c10::Error& e) { rejected=std::string(e.what()).find(message)!=std::string::npos; }
    TORCH_CHECK(rejected, "missing expected rejection: ", message);
}
int main() {
    at::set_num_threads(1);
    auto options=at::TensorOptions().dtype(at::kHalf);
    auto projection=at::eye(128,options), bias=at::zeros({128},options);
    auto noise=at::arange(16*128,options).view({1,16,128})/2048;
    auto original=noise.clone(), cond=at::ones({1,1,128},options);
    std::vector<at::Tensor> keys{at::ones({1,32,1,128},options)}, values{at::ones_like(keys[0])*2};
    auto old_key=keys[0].clone(), old_value=values[0].clone();
    auto mask=at::zeros({1,1,16,16},options);
    std::vector<int64_t> descriptor{1};
    Pi05Model model;
    model.set_weights(1,projection,bias,projection,bias,16,128,128);
    auto& graph=RpuKernelGraph::active();
    // Actual child preparation method: a cold empty Graph is safe, pending
    // work and READY cache misses reject before touching the prepared slot.
    graph.begin();
    rpu_adarms_prepare_graph_mask(1,mask,16,0,false);
    TORCH_CHECK(mask_preparations==1 && graph.pending()==0,"cold mask preparation recorded work");
    graph.record("upstream", [] {});
    auto changed_mask=mask.clone();
    rejects([&]{rpu_adarms_prepare_graph_mask(1,changed_mask,16,0,false);},"prepare a new attention mask before recorded work");
    TORCH_CHECK(mask_preparations==1 && graph.pending()==1,"late mask miss changed graph/cache");
    graph.end();
    graph.begin(true);
    rpu_adarms_prepare_graph_mask(1,mask,16,0,false);
    TORCH_CHECK(mask_preparations==1 && graph.pending()==0,"READY mask hit performed work");
    rejects([&]{rpu_adarms_prepare_graph_mask(1,changed_mask,16,0,false);},"REPLAY requires");
    rejects([&]{rpu_adarms_prepare_graph_mask(1,mask,16,1,false);},"REPLAY requires");
    TORCH_CHECK(mask_preparations==1 && graph.pending()==0,"READY mask miss changed graph/cache");
    graph.clear();
    mask_owner=MaskOwner{};
    std::vector<at::Tensor> held, expected;
    for (int i=0;i<5;++i) {
        auto request=(original*(i%2?0.5:1.0)).clone();
        auto condition=(cond*(i%2?0.25:1.0)).clone();
        graph.begin(i!=0);
        auto result=model.forward(request,condition,mask,keys,values,0,1,-1,descriptor);
        rejects([&]{model.set_weights(1,projection,bias,projection,bias,16,128,128);},"retire captured");
        rejects([&]{model.require_retired_graphs("destroy");},"retire captured");
        graph.end();
        TORCH_CHECK(mask_preparations==2,"driver repeated cold mask preparation");
        auto reference=(request.to(at::kFloat)-(request.to(at::kFloat)*0.25+condition.to(at::kFloat)*0.125).to(at::kHalf).to(at::kFloat)).to(at::kHalf);
        TORCH_CHECK(at::equal(result,reference), "deferred one-step output");
        held.push_back(result); expected.push_back(reference);
        for(size_t j=0;j<held.size();++j) TORCH_CHECK(at::equal(held[j],expected[j]),"retained result overwritten");
        TORCH_CHECK(at::equal(keys[0],old_key) && at::equal(values[0],old_value) && at::equal(noise,original),"caller input mutated");
    }
    TORCH_CHECK(at::equal(held[0],held[2]) && !at::equal(held[0],held[1]), "A/B/A");
    graph.clear();
    // Exercise the actual shared consumer with the actual driver's manifest.
    using namespace v3;
    const auto manifest=model.projection_manifest(16,0);
    TORCH_CHECK(manifest.routes.size()==2 && manifest.linear_accumulation==FmbLinearAccumulationPolicy::ACC32,
                "complete fixed projection manifest");
    FmbPhysicalManifestConsumer complete(manifest);
    for(const auto& route:manifest.routes) complete.consume_route(route.family,route.site_id,route.selector,route.flags,route.arguments);
    complete.require_all_consumed();
    auto missing=manifest;missing.routes.pop_back();
    FmbPhysicalManifestConsumer absent(missing);
    const auto& removed=manifest.routes.back();
    rejects([&]{absent.consume_route(removed.family,removed.site_id,removed.selector,removed.flags,removed.arguments);},"no route");
    FmbPhysicalManifestConsumer wrong(manifest);
    auto arguments=manifest.routes.front().arguments;arguments[1]+=128;
    const auto& first_route=manifest.routes.front();
    rejects([&]{wrong.consume_route(first_route.family,first_route.site_id,first_route.selector,first_route.flags,arguments);},"actually dispatched");
    FmbPhysicalManifestConsumer incomplete(manifest);
    incomplete.consume_route(first_route.family,first_route.site_id,first_route.selector,first_route.flags,first_route.arguments);
    rejects([&]{incomplete.require_all_consumed();},"not consumed");
    auto changed=manifest;changed.routes.front().arguments[0]+=16;
    TORCH_CHECK(fmb_physical_manifest_fingerprint(changed)!=fmb_physical_manifest_fingerprint(manifest),"manifest identity ignores geometry");
    model.set_weights(1,projection,bias,projection,bias,16,128,128);
    graph.begin();
    auto two_cond=at::cat({cond,cond},0);
    rejects([&]{model.forward(noise,two_cond,mask,keys,values,0,2,-0.5,descriptor);},"exactly one denoise step");
    TORCH_CHECK(graph.pending()==0,"multi-step rejection scheduled work");
    model.require_retired_graphs("no allocation after early rejection");
    graph.clear();
    graph.begin();
    auto first=model.forward(noise,cond,mask,keys,values,0,1,-1,descriptor);
    const auto before=graph.pending();
    rejects([&]{model.forward(noise,cond,mask,keys,values,0,1,-1,descriptor);},"one driver invocation");
    TORCH_CHECK(graph.pending()==before,"second invocation scheduled work");
    graph.end(); graph.clear();
    graph.begin();
    auto malformed=cond.view({1,128});
    rejects([&]{model.forward(noise,malformed,mask,keys,values,0,1,-1,descriptor);},"noise or condition shape");
    TORCH_CHECK(graph.pending()==0,"invalid input scheduled work");
    graph.clear();
    graph.begin(true);
    rejects([&]{model.forward(noise,cond,mask,keys,values,0,1,-1,descriptor);},"no matching workspace");
    graph.clear();
    model.require_retired_graphs("destroy");
    // Retiring a graph also permits an ordinary eager multi-step caller.
    auto eager=model.forward(noise,two_cond,mask,keys,values,0,2,-0.5,descriptor);
    auto eager_expected=noise.clone();
    for(int step=0;step<2;++step) eager_expected=(eager_expected.to(at::kFloat)-0.5*(eager_expected.to(at::kFloat)*0.25+cond.to(at::kFloat)*0.125).to(at::kHalf).to(at::kFloat)).to(at::kHalf);
    TORCH_CHECK(at::equal(eager,eager_expected),"eager multi-step changed");
    std::cout << "PASS: actual Pi driver workspace, deferred CPU endpoints\n";
}
'''
