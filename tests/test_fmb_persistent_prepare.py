"""CPU/source contracts for the cold-only shared Persistent allocation barrier."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def _source(path):
    return (ROOT / path).read_text()


def _section(text, start, end):
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


def test_both_native_entries_reach_shared_cold_prepare_without_resolving_a_plan():
    for family, path in (("siglip", "src/fused/rpu_siglip_model.cpp"),
                         ("causal_decoder", "src/fused/rpu_qwen3_model.cpp"),
                         ("qwen3_5", "src/fused/rpu_qwen3_5_model.cpp"),
                         ("qwen3_5_vision", "src/fused/rpu_qwen3_5_vision_model.cpp")):
        name = f"rpu_{family}_prepare_persistent_spm"
        source = _source(path)
        entry = _section(source, f"void {name}(", "\n}\n")
        assert "->prepare_persistent_spm(" in entry
        assert "resolve_" not in entry and "forward(" not in entry
        assert name in _source("src/core/rpu_kernel_decls.h")
        assert f"TORCH_FN({name})" in _source("src/core/rpu_dispatch_registrations.inc")
    source = _source("src/core/fused_model_base.cpp")
    entry = _section(source, "void FusedModelBase::prepare_persistent_spm(", "\n}\n")
    assert "!RpuKernelGraph::has_active()" in entry
    assert 'RpuExecutionCleanupGuard exclusive_planning("prepare_persistent_spm");' in entry
    assert entry.index("RpuExecutionCleanupGuard") < entry.index("declare_buffers(")
    assert "pimpl_->last_resolved_chunk_size_ == 0" in entry
    assert "!pimpl_->valid_" in entry
    assert "allocate_persistent_impl(" in entry
    for forbidden in ("run_all_layers(", "ensure_allocated_impl(", "set_chunk_size_override(",
                      "run_preload_callbacks_(", "cached_preload_gen_ =", "valid_ = true",
                      "last_resolved_chunk_size_ = ", "stage_plan_fingerprint ="):
        assert forbidden not in entry
    assert entry.index("subclass_chunk_size_cap(") < entry.index("allocate_persistent_impl(")
    ensure = _section(source, "static void ensure_allocated_impl(", "\n}\n")
    assert "!RpuKernelGraph::has_active()" not in ensure
    assert "if (persistent_needs_alloc) {\n            allocate_persistent_impl(" in ensure
    assert "if (persistent_needs_alloc) {\n            s.persistent_layout_version_++;" in ensure
    assert "if (persistent_invalidated) s.preload_callbacks_dirty_ = true;" in ensure
    # Preparation cannot pre-consume either real BUILD-time preload path.
    assert "run_preload_callbacks_(pimpl_->last_decls_, *this);" in source
    assert "std::invoke(static_cfg.preload_fn, *this);\n        pimpl_->weights_dirty_ = false;" in source


def test_peer_persistent_floor_is_fixed_before_cold_domain_collection():
    # Pure arithmetic from the two installed HALO profiles, NOT a native chunk
    # measurement. A valid cold layout can otherwise fail after its peer starts.
    usable = 8 * 1024 * 1024 - 1024
    planning = usable * 99 // 100
    align = lambda n: (n + 255) // 256 * 256
    dma = lambda n: align((n + 255) // 256 * 256 * 2)
    vision = 26 * (6 * dma(1152) + 3 * dma(2 * 80) + dma(4352 // 8)) + 2 * dma(1152)
    language = 28 * (2 * align(1536 * 2) + 2 * align(128 * 2)
                     + dma(6 * 128) + 2 * dma(128)) + align(1536 * 2)
    assert (vision, language) == (484352, 261120)
    budget = lambda own, peer: min(planning - own, usable - 65536 - own - peer)
    illustrative_temp = align(7700000)
    assert budget(vision, language) < illustrative_temp <= budget(vision, 0)
    assert budget(vision, language) == budget(language, vision) == 7576576
    builder = _section(_source("python/rpu_backend/adapters/halo/vit_cache.py"),
                       "def build_halo_vit_cache(", "\n    return runner")
    first = builder.index("torch.ops.rpu.siglip_prepare_persistent_spm(")
    second = builder.index("torch.ops.rpu.causal_decoder_prepare_persistent_spm(")
    assert builder.index("lm_cache = RPUCache(") < first < second < builder.index("execution_controller.attach(")


def test_qwen35_language_norm_preloads_follow_each_real_graph_dispatch(tmp_path):
    """Compile existing declaration callbacks, config and FMB dispatch slices.

    Allocator/DMA/Graph edges are CPU doubles, not numerical or hardware credit.
    This guards the reviewed prefill/decode equivalence against a clean-SPM
    shortcut; the original long-2B gate still owes both hardware lifecycles.
    """
    import re

    source = _source("src/fused/rpu_qwen3_5_model.cpp")
    base = _source("src/core/fused_model_base.cpp")
    header = _source("src/core/fused_model_base.h")
    config = _section(source, "ModelStaticConfig Qwen3_5Model::static_config()", "\n}\n") + "\n}\n"
    # The checked decode body hook is bound only inside the checked decode
    # scope; every forward exercised below runs outside it.
    assert "if (checked_decode_replay_scope_active_) {\n        cfg.checked_layer_body_replay_fn" in config
    assert config.count("checked_layer_body_replay_fn") == 1
    defaults = "\n".join(re.findall(
        r"bool (?:fast_replay_skip_\w+|clean_recording_skip_preload_callbacks) = false;", header))
    declarations = _section(source, "    // Per-layer input_layernorm weight", "    if (has_qk_norm_) {")
    start = source.index('        d.name = "final_norm_w";', source.index(declarations))
    # Isolate the complete final-norm declaration, excluding later Action scratch.
    final_block = source[source.rfind("    {\n", 0, start):
                         source.index("\n    }\n", start) + len("\n    }\n")]
    preload_start = base.index("void FusedModelBase::run_preload_callbacks_(")
    early_gate = _section(base[preload_start:], "    const bool inside_graph = graph_dma::active();", "    // graph-naive:")
    # This ordinary Qwen3.5 harness exercises callback dispatch/bookkeeping.
    # Composite receipt capture is covered by test_composite_preload_route_receipt;
    # do not copy its trailing half without its leader context into this slice.
    dispatch = _section(base[preload_start:], "    for (const auto& d : decls)",
                        "    if (record_composite_preload_routes)")
    dispatch += _section(base[preload_start:], "    // EXT-4: clear BOTH", "\n}\n")
    outer = _section(base, "    const bool skip_preload_callbacks_replay =", "\n    // 9. group count")
    harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
namespace c10 {using Half=uint16_t;}
struct FusedModelBase {};
struct ModelStaticConfig {int num_layers=0,cross_layer_batch_size=0,body_iterations=1;
    void (FusedModelBase::*pre_layers_fn)()=nullptr, (FusedModelBase::*post_layers_fn)()=nullptr;
    bool (FusedModelBase::*checked_layer_body_replay_fn)()=nullptr;
DEFAULTS
};
enum class StorageClass {Persistent,PersistentPerLayer};
struct BufferDecl {const char* name; int64_t size; StorageClass storage; int per_layer=0;
    std::function<void(FusedModelBase&,int,uint32_t)> preload_callback;};
struct Tensor {c10::Half value=7; template<class T> T* data_ptr(){return &value;}};
std::unordered_map<uint32_t,c10::Half> spm;
std::vector<int> dma_cores;
void rpu_launch_ddr_broadcast_spm_dma(const c10::Half* src,int64_t,uint32_t dst,int cores){
    assert(cores==4 || cores==6 || cores==8);
    spm[dst]=*src; dma_cores.push_back(cores);
}
struct RpuKernelGraph {enum class State {RECORDING,REPLAYING};
    static inline bool enabled=true; static inline State current=State::RECORDING;
    static bool has_active(){return enabled;} static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
    State state(){return current;}};
namespace graph_dma {bool active(){return RpuKernelGraph::enabled;}}
struct {int persistent_generation(){return 7;} uint32_t addr(int,uint32_t x){return x;}} SPM_ALLOC;
struct Qwen3_5Model: FusedModelBase {
    bool fast_replay_active_=true,action_step_active_=false,action_loop_active_=false;
    bool checked_decode_replay_scope_active_=false;
    bool try_checked_layer_body_replay(){return false;}
    int action_num_steps_=1, active_cores=8;
    struct {Tensor input_norm_w,post_norm_w;} layer_weights_[2]; Tensor final_norm_w_;
    int num_layers(){return 2;} int hidden_size(){return 16;}
    int num_cores(){return active_cores;}
    void emit_action_input_projection(){} void emit_action_output_projection(){}
    ModelStaticConfig static_config();
    struct State {struct {bool complete=true; bool has_complete_physical_manifest(){return complete;}} ctx_;
        int cached_preload_gen_=7,pipeline_lease_epoch_=0; bool preload_callbacks_dirty_=false;
        bool resident_preload_priming_=false;
        std::vector<std::string> resident_preload_names_;
        std::unordered_map<std::string,uint32_t> persistent_offsets_{{"final_norm_w",50}};
        std::vector<std::unordered_map<std::string,uint32_t>> persistent_per_layer_offsets_{
            {{"norm_w",10},{"post_norm_w",20}},{{"norm_w",30},{"post_norm_w",40}}};
        std::vector<BufferDecl> last_decls_;
    } state; State* pimpl_=&state;
    std::vector<BufferDecl> declare_buffers(){int nl=num_layers(),nw=256;std::vector<BufferDecl> decls;
DECLARATIONS
        return decls;
    }
    bool resident_preloads_enabled()const{return false;}
    void validate_resident_preloads_(){assert(!resident_preloads_enabled());}
    bool consume_spm_pipeline_composite_preload_follower(){return false;}
    void run_preload_callbacks_(const std::vector<BufferDecl>& decls,FusedModelBase& self){
EARLY_GATE
        auto invoke_preload_callback=[&](const BufferDecl& d,int layer,uint32_t addr){d.preload_callback(self,layer,addr);};
DISPATCH
    }
    void forward_prelude(){const auto static_cfg=static_config(); bool deep_full_skip=false;
        // Outside the checked decode scope this config binds no checked hook.
        assert(static_cfg.checked_layer_body_replay_fn==nullptr);
        const bool checked_body_replay=false;
        assert(!static_cfg.fast_replay_skip_full_body);
OUTER
    }
};
CONFIG
int main(){
    {Qwen3_5Model scoped; scoped.checked_decode_replay_scope_active_=true;
     assert(scoped.static_config().checked_layer_body_replay_fn!=nullptr);}
    for(int cores:{4,6,8}) {
    Qwen3_5Model m; m.active_cores=cores; m.state.last_decls_=m.declare_buffers();
    RpuKernelGraph::enabled=true;
    // Bounded one-shot RECORDING, then a new decode BUILD, then stable REPLAY.
    // Clean generations must not suppress any graph's actual norm preloads.
    for(auto mode:{RpuKernelGraph::State::RECORDING,RpuKernelGraph::State::RECORDING,RpuKernelGraph::State::REPLAYING}){
        RpuKernelGraph::current=mode; spm.clear(); dma_cores.clear(); m.forward_prelude();
        for(uint32_t address:{10,20,30,40,50}) assert(spm.at(address)==7);
        assert(dma_cores==std::vector<int>(5,cores));
        assert(!m.state.preload_callbacks_dirty_ && m.state.cached_preload_gen_==7);
    }
    RpuKernelGraph::enabled=false; spm.clear();dma_cores.clear();m.forward_prelude();assert(spm.at(50)==7);
    assert(dma_cores==std::vector<int>(5,cores));
    m.state.ctx_.complete=false;spm.clear();dma_cores.clear();m.forward_prelude();assert(spm.empty());
    assert(dma_cores.empty());
}}
'''
    for marker, text in (("DEFAULTS", defaults), ("DECLARATIONS", declarations + final_block),
                         ("EARLY_GATE", early_gate), ("DISPATCH", dispatch),
                         ("OUTER", outer), ("CONFIG", config)):
        harness = harness.replace(marker, text)
    cpp, binary = tmp_path / "q35_norm_preload.cpp", tmp_path / "q35_norm_preload"
    cpp.write_text(harness)
    compiled = subprocess.run(["g++", "-std=c++17", str(cpp), "-o", str(binary)], capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr


def test_nextdit_prepare_preloads_belong_to_each_new_graph(tmp_path):
    """Compile actual config, preload callbacks and FMB dispatch/COMPLETE checks.

    Allocator and DMA edges are doubles. This proves preload admission and
    route consumption, not full graph execution or numerical equivalence.
    """
    import re

    source = _source("src/fused/rpu_internvla_nextdit_model.cpp")
    base = _source("src/core/fused_model_base.cpp")
    header = _source("src/core/fused_model_base.h")
    shared = _source("src/core/fmb_three_stage_chunk_plan.cpp")
    config = _section(source, "    ModelStaticConfig static_config() override {",
                      "\n    ModelDynamicConfig dynamic_config(").replace(" override", "")
    declarations = _section(source, "            BufferDecl step_bias;",
                            "\n        }\n\n        // Cores 0..5")
    routes = _section(source, "            append_dma(\n                kNextDitPrepareBiasPreloadDmaSite,",
                      "\n        }\n        append_dma(")
    complete = (_section(shared, "void FmbPhysicalManifestConsumer::require_all_consumed() const",
                         "\n}\n") + "\n}\n").replace("FmbPhysicalManifestConsumer::", "")
    defaults = "\n".join(re.findall(
        r"bool (?:fast_replay_skip_\w+|clean_recording_skip_preload_callbacks) = false;", header))
    constants = "\n".join(re.findall(
        r"constexpr int64_t kNextDitPrepare(?:Bias|Fold)PreloadDmaSite = \d+LL;", source))
    preload = base[base.index("void FusedModelBase::run_preload_callbacks_("):]
    early = _section(preload, "    const bool inside_graph = graph_dma::active();", "    // graph-naive:")
    dispatch = _section(preload, "    for (const auto& d : decls)",
                        "    if (record_composite_preload_routes)")
    dispatch += _section(preload, "    // EXT-4: clear BOTH", "\n}\n")
    outer = _section(base, "    const bool skip_preload_callbacks_replay =", "\n    // 9. group count")
    harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("route rejected"); } while(0)
constexpr int NEXTDIT_DEVICE_CORES=8, NEXTDIT_DWIDTH=2, ALL=1;
CONSTANTS
namespace c10 {using Half=uint16_t;}
struct FusedModelBase {};
struct ModelStaticConfig {int num_layers=0,cross_layer_batch_size=0,body_iterations=1; DEFAULTS };
enum class StorageClass {Persistent,PersistentPerLayer};
enum class FmbRouteFamily {MUTABLE_DMA=6};
enum class NextDitDmaRoute {DDR_TO_SPM=1};
struct FmbRouteManifestEntry {int64_t site_id;FmbRouteFamily family;int64_t selector,flags;
    std::vector<int64_t> arguments;int64_t invocation=0;};
struct Manifest {std::vector<FmbRouteManifestEntry> routes;};
struct Context {
    Manifest manifest_;std::vector<uint8_t> consumed_;
    bool complete()const{return true;} bool has_complete_physical_manifest()const{return true;}
    const Manifest& manifest()const{return manifest_;}
    void bind(Manifest m){manifest_=std::move(m);consumed_.assign(manifest_.routes.size(),0);}
    void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,
                                std::vector<int64_t> arguments,int64_t invocation=0){
        auto it=std::find_if(manifest_.routes.begin(),manifest_.routes.end(),[&](const auto& row){
            return row.site_id==site && row.family==family && row.invocation==invocation;});
        TORCH_CHECK(it!=manifest_.routes.end());
        TORCH_CHECK(it->selector==selector && it->flags==flags && it->arguments==arguments);
        consumed_[it-manifest_.routes.begin()]=1;
    }
COMPLETE
};
struct BufferDecl {const char* name;int64_t size=0;StorageClass storage;int scope=0,per_layer=0;
    std::function<void(FusedModelBase&,int,uint32_t)> preload_callback;};
struct Tensor {c10::Half value=7;template<class T>T* data_ptr(){return &value;}};
std::vector<uint32_t> dma_addresses;
int ordinary_preloads=0;
void rpu_launch_ddr_scatter_spm_dma(const c10::Half*,int64_t n,int64_t stride,uint32_t dst,int cores){
    assert(n==2352 && stride==4704 && cores==8);dma_addresses.push_back(dst);}
void rpu_launch_ddr_broadcast_spm_dma(const c10::Half*,int64_t n,uint32_t dst,int cores){
    assert(n==18432 && cores==1);dma_addresses.push_back(dst);}
struct RpuKernelGraph {enum class State {RECORDING,REPLAYING};
    static inline State current=State::RECORDING;
    static bool has_active(){return true;} static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
    State state(){return current;}};
namespace graph_dma {bool active(){return true;}}
struct {int persistent_generation(){return 7;}uint32_t addr(int,uint32_t offset){return offset;}} SPM_ALLOC;
struct NextDiT: FusedModelBase {
    bool denoise_unroll_=true,use_action_encoder_=true,packed_action_batch_=true,prepare_modulation_on_rpu_=false;
    int denoise_steps_=10,action_batch_size_=4;
    Tensor step_prepare_b_,step_fold_norm_w_;
    int64_t num_layers(){return 12;}int64_t hidden_size(){return 384;}int64_t A(int64_t n){return n;}
    struct State {Context ctx_;int cached_preload_gen_=7,pipeline_lease_epoch_=0;
        bool preload_callbacks_dirty_=true,resident_preload_priming_=false;
        std::vector<std::string> resident_preload_names_;
        std::unordered_map<std::string,uint32_t> persistent_offsets_{{"step_bias",10},{"step_fold_norm",20},{"ordinary",30}};
        std::vector<std::unordered_map<std::string,uint32_t>> persistent_per_layer_offsets_;
        std::vector<BufferDecl> last_decls_;
    } state;State* pimpl_=&state;
    Context& ctx(){return state.ctx_;}
CONFIG
    Manifest preload_manifest(){Manifest manifest;
        const auto append_dma=[&](int64_t site,NextDitDmaRoute selector,std::vector<int64_t> args){
            manifest.routes.push_back({site,FmbRouteFamily::MUTABLE_DMA,int64_t(selector),0,std::move(args),0});};
        if(prepare_modulation_on_rpu_){
            const int64_t step_local=(num_layers()*4+1)*hidden_size()/NEXTDIT_DEVICE_CORES;
ROUTES
        }return manifest;
    }
    std::vector<BufferDecl> declare_preloads(){std::vector<BufferDecl> out;
        if(prepare_modulation_on_rpu_){const int64_t h=hidden_size();
            const int64_t step_local=(num_layers()*4+1)*h/NEXTDIT_DEVICE_CORES;
DECLARATIONS
        }
        BufferDecl ordinary;ordinary.name="ordinary";ordinary.storage=StorageClass::Persistent;
        ordinary.preload_callback=[](FusedModelBase&,int,uint32_t){++ordinary_preloads;};
        out.push_back(ordinary);return out;
    }
    bool resident_preloads_enabled()const{return false;}
    void validate_resident_preloads_(){assert(!resident_preloads_enabled());}
    bool consume_spm_pipeline_composite_preload_follower(){return false;}
    void run_preload_callbacks_(const std::vector<BufferDecl>& decls,FusedModelBase& self){
EARLY
        auto invoke_preload_callback=[&](const BufferDecl& d,int layer,uint32_t address){d.preload_callback(self,layer,address);};
DISPATCH
    }
    void prelude(){const auto static_cfg=static_config();
        const bool deep_full_skip=false,checked_body_replay=false;
OUTER
    }
};
int main(){
    for(bool prepare:{false,true}){
        NextDiT owner;owner.prepare_modulation_on_rpu_=prepare;
        owner.state.last_decls_=owner.declare_preloads();
        const auto cfg=owner.static_config();
        assert(cfg.fast_replay_skip_preload && cfg.fast_replay_skip_layer_loop);
        // Two distinct BUILD invocations: cold/dirty, then same-generation clean.
        for(int build=0;build<2;++build){
            RpuKernelGraph::current=RpuKernelGraph::State::RECORDING;
            owner.ctx().bind(owner.preload_manifest());dma_addresses.clear();ordinary_preloads=0;
            owner.prelude();owner.ctx().require_all_consumed();
            assert(dma_addresses==(prepare?std::vector<uint32_t>{10,20}:std::vector<uint32_t>{}));
            assert(ordinary_preloads==int(prepare || build==0));
            assert(owner.state.cached_preload_gen_==7 && !owner.state.preload_callbacks_dirty_);
        }
        // REPLAY keeps the existing fast-skip path. Exact Graph receipt inheritance
        // is a separate shared contract; this proof does not fabricate its receipt.
        RpuKernelGraph::current=RpuKernelGraph::State::REPLAYING;
        dma_addresses.clear();ordinary_preloads=0;owner.prelude();
        assert(dma_addresses.empty() && ordinary_preloads==0);
    }
    // Neither declared preload may be silently omitted from a new COMPLETE BUILD.
    for(const char* missing:{"step_bias","step_fold_norm"}){
        NextDiT owner;owner.prepare_modulation_on_rpu_=true;
        owner.state.last_decls_=owner.declare_preloads();owner.state.preload_callbacks_dirty_=false;
        for(auto& d:owner.state.last_decls_)if(std::string(d.name)==missing)d.preload_callback={};
        owner.ctx().bind(owner.preload_manifest());RpuKernelGraph::current=RpuKernelGraph::State::RECORDING;
        dma_addresses.clear();owner.prelude();assert(dma_addresses.size()==1);
        bool rejected=false;try{owner.ctx().require_all_consumed();}catch(const std::runtime_error&){rejected=true;}
        assert(rejected);
    }
}
'''
    for marker, value in (("CONSTANTS", constants), ("DEFAULTS", defaults), ("COMPLETE", complete),
                          ("CONFIG", config), ("ROUTES", routes), ("DECLARATIONS", declarations),
                          ("EARLY", early), ("DISPATCH", dispatch), ("OUTER", outer)):
        if marker == "DEFAULTS":
            harness = harness.replace(" DEFAULTS ", " " + value + " ")
        else:
            harness = harness.replace("\n" + marker + "\n", "\n" + value + "\n")
    cpp, binary = tmp_path / "nextdit_preload.cpp", tmp_path / "nextdit_preload"
    cpp.write_text(harness)
    compiled = subprocess.run(["g++", "-std=c++17", str(cpp), "-o", str(binary)],
                              capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr


def _run_extracted_production_prepare_and_wrappers_on_cpu(tmp_path, *, gr00t_only=False):
    """Compile the unchanged function bodies; only allocator/owner edges are fake.

    This tests cold control logic, not SDK allocation, native model execution,
    or hardware BUILD/REPLAY. The original gate retains those hardware duties.
    """
    source = _source("src/core/fused_model_base.cpp")
    bodies = [
        _section(source, "static int64_t compute_persistent_hash_impl(", "\n}\n") + "\n}\n",
        _section(source, "static void allocate_persistent_impl(", "\n}\n") + "\n}\n",
        _section(source, "void FusedModelBase::prepare_persistent_spm(", "\n}\n") + "\n}\n",
    ]
    families = (("qwen3vl_vision", "src/fused/rpu_qwen3vl_vision_model.cpp"),
                ("gr00t_vl_encoder", "src/fused/rpu_gr00t_vl_encoder_model.cpp"),
                ("gr00t_dit", "src/fused/rpu_gr00t_dit_model.cpp")) if gr00t_only else (
                ("siglip", "src/fused/rpu_siglip_model.cpp"),
                ("causal_decoder", "src/fused/rpu_qwen3_model.cpp"),
                ("qwen3_5", "src/fused/rpu_qwen3_5_model.cpp"),
                ("qwen3_5_vision", "src/fused/rpu_qwen3_5_vision_model.cpp"))
    for family, path in families:
        name = f"rpu_{family}_prepare_persistent_spm"
        bodies.append(_section(_source(path), f"void {name}(", "\n}\n") + "\n}\n")
        assert name in _source("src/core/rpu_kernel_decls.h")
        assert f"TORCH_FN({name})" in _source("src/core/rpu_dispatch_registrations.inc")
    harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#define TORCH_CHECK(test, ...) do { if (!(test)) throw std::runtime_error("rejected"); } while (0)
enum class StorageClass {Temp, TempPerLayer, Persistent, PersistentPerLayer};
class FusedModelBase;
struct BufferDecl {
    const char* name; int64_t size; StorageClass storage;
    int per_layer = 0; const char* alias_of = nullptr; bool reverse_layer_alloc = false;
    std::function<void(FusedModelBase&,int,uint32_t)> preload_callback{};
};
struct LayoutContext {
    int64_t chunk_size=0, num_layers=0, max_kv_seq_len=0;
    bool use_attn_mask=false, is_causal=true;
};
struct RpuKernelGraph { static inline bool active=false; static bool has_active(){return active;} };
struct RpuExecutionCleanupGuard {
    static inline bool foreign_busy=false, held=false;
    explicit RpuExecutionCleanupGuard(const char*) {
        TORCH_CHECK(!foreign_busy && !held); held=true;
    }
    ~RpuExecutionCleanupGuard(){held=false;}
};
struct FakeAllocator {
    uint32_t floor=65536; int generation=0, init_count=0, allocations=0;
    bool initialized=false;
    void reset_all(){assert(RpuExecutionCleanupGuard::held || RpuKernelGraph::active); ++generation;}
    bool is_initialized(){return initialized;}
    void init(){initialized=true; ++init_count;}
    uint32_t alloc_super_persistent(int64_t size){
        assert(RpuExecutionCleanupGuard::held || RpuKernelGraph::active);
        uint32_t aligned=(size+255)/256*256;
        TORCH_CHECK(aligned<=floor);
        floor-=aligned; ++allocations; return floor;
    }
} SPM_ALLOC;
namespace detail {
void validate_fmb_planning_shape(int64_t n,int64_t p,const char*) {
    TORCH_CHECK(n>0 && p>=0 && n<=std::numeric_limits<int64_t>::max()-p);
}
}
class FusedModelBase {
public:
    struct Impl {
        bool persistent_allocated_=false, valid_=false, allocation_declaration_valid_=false;
        bool weights_dirty_=false, preload_callbacks_dirty_=false;
        std::vector<std::string> resident_preload_names_;
        std::vector<BufferDecl> resident_preload_decls_;
        int64_t persistent_usage_=0, persistent_layout_hash_=0, last_resolved_chunk_size_=0;
        int64_t num_layers_=2, hidden_size_=16, num_q_heads_=1, num_kv_heads_=1, head_dim_=16, intermediate_size_=16;
        uint64_t persistent_layout_version_=0, cached_preload_gen_=0, alloc_gen_=0;
        struct {const void* hidden_states=nullptr;} ctx_;
        std::unordered_map<std::string,uint32_t> persistent_offsets_;
        std::vector<std::unordered_map<std::string,uint32_t>> persistent_per_layer_offsets_;
    } state;
    struct Envelope {int64_t max_kv_len=64; bool declared()const{return max_kv_len>0;}} envelope;
    Impl* pimpl_=&state;
    std::vector<BufferDecl> decls{
        {"global", 1, StorageClass::Persistent},
        {"normal", 257, StorageClass::PersistentPerLayer, 2},
        {"reverse", 256, StorageClass::PersistentPerLayer, 2, nullptr, true},
        {"alias", 1000000, StorageClass::Persistent, 0, "global"},
        {"scratch", 1000000, StorageClass::Temp},
        {"scratch-layer", 1000000, StorageClass::TempPerLayer, 2}};
    LayoutContext last_layout;
    int64_t last_envelope_length=0, last_envelope_position=-1;
    std::vector<BufferDecl> declare_buffers(const LayoutContext& c) {
        assert(c.chunk_size==16 && c.num_layers==2 && c.max_kv_seq_len>0);
        last_layout=c;
        return decls;
    }
    const Envelope& chunk_envelope(){return envelope;}
    int64_t subclass_chunk_size_cap(int64_t n,int64_t p) {
        TORCH_CHECK(envelope.declared() && n<=envelope.max_kv_len && p<=envelope.max_kv_len-n);
        last_envelope_length=n; last_envelope_position=p;
        return 0;
    }
    bool resident_preloads_enabled()const{return !state.resident_preload_names_.empty();}
    void prepare_persistent_spm(int64_t,int64_t,bool,int64_t=0);
};
struct SigLIPRegistry {
    static inline FusedModelBase* owner=nullptr;
    static FusedModelBase* get(int64_t h,const char*){TORCH_CHECK(h==41);return owner;}
};
struct CausalDecoderRegistry {
    static inline FusedModelBase* owner=nullptr;
    static FusedModelBase* get(int64_t h,const char*){TORCH_CHECK(h==42);return owner;}
};
using Qwen3_5Registry = CausalDecoderRegistry;
using Qwen3_5VisionRegistry = SigLIPRegistry;
using Qwen3VLVisionRegistry = SigLIPRegistry;
using Gr00tVlEncoderRegistry = CausalDecoderRegistry;
struct Gr00tDiTRegistry {
    static inline FusedModelBase* owner=nullptr;
    static FusedModelBase* get(int64_t h,const char*){TORCH_CHECK(h==43);return owner;}
};
template<class F> void rejects(F call) {
    const int count=SPM_ALLOC.allocations, generation=SPM_ALLOC.generation;
    const uint32_t floor=SPM_ALLOC.floor;
    bool rejected=false; try {call();} catch(const std::runtime_error&){rejected=true;}
    assert(rejected && count==SPM_ALLOC.allocations && generation==SPM_ALLOC.generation && floor==SPM_ALLOC.floor);
    assert(!RpuExecutionCleanupGuard::held);
}
'''
    main = r'''
int main() {
    FusedModelBase vision, language;
    SigLIPRegistry::owner=&vision; CausalDecoderRegistry::owner=&language;
    auto prepare=[&]{rpu_siglip_prepare_persistent_spm(41,32,1);};
    RpuExecutionCleanupGuard::foreign_busy=true; rejects(prepare); RpuExecutionCleanupGuard::foreign_busy=false;
    RpuKernelGraph::active=true; rejects(prepare); RpuKernelGraph::active=false;
    vision.state.valid_=true; rejects(prepare); vision.state.valid_=false;
    vision.state.allocation_declaration_valid_=true; rejects(prepare); vision.state.allocation_declaration_valid_=false;
    vision.state.last_resolved_chunk_size_=16; rejects(prepare); vision.state.last_resolved_chunk_size_=0;
    vision.state.ctx_.hidden_states=&vision; rejects(prepare); vision.state.ctx_.hidden_states=nullptr;
    vision.state.num_layers_=0; rejects(prepare); vision.state.num_layers_=2;
    rejects([&]{rpu_siglip_prepare_persistent_spm(41,31,2);});
    rejects([&]{rpu_siglip_prepare_persistent_spm(41,32,0);});
    rejects([&]{rpu_causal_decoder_prepare_persistent_spm(42,64,1,false,0);});
    rejects([&]{rpu_causal_decoder_prepare_persistent_spm(42,16,-1,false,0);});
    rejects([&]{rpu_causal_decoder_prepare_persistent_spm(42,16,std::numeric_limits<int64_t>::max(),false,0);});
    rejects([&]{rpu_causal_decoder_prepare_persistent_spm(42,16,0,true,16);});
    rejects([&]{rpu_causal_decoder_prepare_persistent_spm(42,16,0,false,65);});
    prepare();
    assert(!RpuExecutionCleanupGuard::held);
    auto& s=vision.state;
    assert(s.persistent_offsets_.at("global")==65280);
    assert(s.persistent_per_layer_offsets_[0].at("normal")==64768);
    assert(s.persistent_per_layer_offsets_[1].at("normal")==64256);
    assert(s.persistent_per_layer_offsets_[1].at("reverse")==64000);
    assert(s.persistent_per_layer_offsets_[0].at("reverse")==63744);
    assert(s.persistent_offsets_.size()==1 && s.persistent_usage_==1027);
    assert(SPM_ALLOC.floor==63744 && SPM_ALLOC.allocations==5 && SPM_ALLOC.generation==1 && SPM_ALLOC.init_count==1);
    assert(!s.valid_ && !s.allocation_declaration_valid_ && s.last_resolved_chunk_size_==0);
    assert(s.persistent_layout_version_==1 && s.weights_dirty_ && s.preload_callbacks_dirty_);
    assert(s.cached_preload_gen_==0 && s.alloc_gen_==0 && s.ctx_.hidden_states==nullptr);
    // Same Persistent shape despite a different request size is a true no-op,
    // including dirty flags: do not unconditionally dirty repeated preparation.
    s.weights_dirty_=false; s.preload_callbacks_dirty_=false;
    rpu_siglip_prepare_persistent_spm(41,48,1);
    assert(SPM_ALLOC.allocations==5 && SPM_ALLOC.generation==1 && s.persistent_layout_version_==1);
    assert(!s.weights_dirty_ && !s.preload_callbacks_dirty_);
    vision.decls[1].reverse_layer_alloc=true; rejects(prepare); vision.decls[1].reverse_layer_alloc=false;
    vision.decls[0].size=257; rejects(prepare); vision.decls[0].size=1;
    rpu_causal_decoder_prepare_persistent_spm(42,32,0,false,0);
    assert(SPM_ALLOC.floor==61952 && SPM_ALLOC.allocations==10 && SPM_ALLOC.generation==2);
    // New Q35 entries preserve the original cold-only and idempotence rules.
    rpu_qwen3_5_prepare_persistent_spm(42,32);
    rpu_qwen3_5_vision_prepare_persistent_spm(41,32);
    assert(SPM_ALLOC.floor==61952 && SPM_ALLOC.allocations==10);
    rejects([&]{rpu_qwen3_5_prepare_persistent_spm(0,32);});
    rejects([&]{rpu_qwen3_5_vision_prepare_persistent_spm(41,0);});
    language.state.last_resolved_chunk_size_=16;
    rejects([&]{rpu_qwen3_5_prepare_persistent_spm(42,32);});
    // The legacy helper is legal during BUILD; only the public cold entry is
    // outside-Graph. It still ignores aliases/Temp and preserves layer order.
    FusedModelBase legacy;
    RpuKernelGraph::active=true;
    allocate_persistent_impl(legacy.state, legacy.decls, compute_persistent_hash_impl(legacy.decls));
    assert(legacy.state.persistent_allocated_ && SPM_ALLOC.allocations==15);
}
'''
    if gr00t_only:
        main = r'''
int main() {
    FusedModelBase vision, vl, dit;
    Qwen3VLVisionRegistry::owner=&vision;
    Gr00tVlEncoderRegistry::owner=&vl;
    Gr00tDiTRegistry::owner=&dit;
    struct Entry {
        FusedModelBase* owner; int64_t handle, execution_len;
        void (*prepare)(int64_t,int64_t);
    };
    const Entry entries[] = {
        {&vision,41,256,rpu_qwen3vl_vision_prepare_persistent_spm},
        {&vl,42,256,rpu_gr00t_vl_encoder_prepare_persistent_spm},
        {&dit,43,41,rpu_gr00t_dit_prepare_persistent_spm}};
    for(const auto& entry:entries) {
        auto& owner=*entry.owner;
        owner.envelope.max_kv_len=entry.execution_len;
        auto prepare=[&]{entry.prepare(entry.handle,entry.execution_len);};
        rejects([&]{entry.prepare(0,entry.execution_len);});
        rejects([&]{entry.prepare(entry.handle,0);});
        rejects([&]{entry.prepare(entry.handle,entry.execution_len+1);});
        RpuKernelGraph::active=true; rejects(prepare); RpuKernelGraph::active=false;
        RpuExecutionCleanupGuard::foreign_busy=true; rejects(prepare);
        RpuExecutionCleanupGuard::foreign_busy=false;
        owner.state.valid_=true; rejects(prepare); owner.state.valid_=false;
        owner.state.last_resolved_chunk_size_=16; rejects(prepare);
        owner.state.last_resolved_chunk_size_=0;
        owner.state.num_layers_=0; rejects(prepare); owner.state.num_layers_=2;
        const int allocations=SPM_ALLOC.allocations;
        prepare();
        assert(!RpuExecutionCleanupGuard::held);
        assert(owner.last_envelope_length==entry.execution_len);
        assert(owner.last_envelope_position==0);
        assert(owner.last_layout.chunk_size==16);
        assert(owner.last_layout.max_kv_seq_len==entry.execution_len);
        assert(!owner.last_layout.is_causal && !owner.last_layout.use_attn_mask);
        assert(owner.state.persistent_allocated_ && owner.state.persistent_layout_version_==1);
        assert(SPM_ALLOC.allocations==allocations+5);
        assert(owner.state.weights_dirty_ && owner.state.preload_callbacks_dirty_);
        assert(!owner.state.valid_ && !owner.state.allocation_declaration_valid_);
        assert(owner.state.last_resolved_chunk_size_==0);
        assert(owner.state.cached_preload_gen_==0 && owner.state.alloc_gen_==0);
        const uint32_t floor=SPM_ALLOC.floor;
        owner.state.weights_dirty_=false; owner.state.preload_callbacks_dirty_=false;
        // The envelope request is not a chunk request. C16 Persistent
        // declarations remain idempotent for another legal execution length.
        entry.prepare(entry.handle,16);
        assert(owner.last_envelope_length==16 && owner.last_layout.chunk_size==16);
        assert(SPM_ALLOC.allocations==allocations+5 && SPM_ALLOC.floor==floor);
        assert(owner.state.persistent_layout_version_==1);
        assert(!owner.state.weights_dirty_ && !owner.state.preload_callbacks_dirty_);
        owner.state.allocation_declaration_valid_=true; rejects(prepare);
    }
}
'''
    cpp, binary = tmp_path / "persistent.cpp", tmp_path / "persistent"
    cpp.write_text(harness + "\n".join(bodies) + main)
    compiled = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                               str(cpp), "-o", str(binary)], capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert run.returncode == 0, run.stderr


def test_extracted_production_prepare_and_wrappers_on_cpu(tmp_path):
    _run_extracted_production_prepare_and_wrappers_on_cpu(tmp_path)


def test_gr00t_prepare_wrappers_delegate_cold_envelope_and_reject_invalid_owners(tmp_path):
    _run_extracted_production_prepare_and_wrappers_on_cpu(tmp_path, gr00t_only=True)


def test_gr00t_installed_persistent_declarations_survive_plain_step_unroll(tmp_path):
    """Execute real declaration callbacks/hash, not a Python native double."""
    from test_fmb_standalone_layout_rebuild import _definition

    model = _source("src/fused/rpu_gr00t_dit_model.cpp")
    base = _source("src/core/fused_model_base.cpp")
    # All Persistent declarations/callbacks live in the complete baseline
    # builder. The temporary-only lifetime transform is exercised separately
    # by test_gr00t_mask_residency; do not capture unrelated adjacent methods.
    declaration = _definition(model, "std::vector<BufferDecl> baseline_buffer_declarations(")
    declaration = declaration.replace("baseline_buffer_declarations", "declare_buffers", 1).replace(
        "const LayoutContext& ctx) const", "const LayoutContext& ctx) override", 1)
    manifest_start = model.index("        if (denoise_ready_) {", model.index("FmbPhysicalExecutionManifest manifest;"))
    manifest = model[manifest_start:model.index("        if (denoise_mode_) {", manifest_start)]
    persistent_hash = _section(base, "static int64_t compute_persistent_hash_impl(", "\n}\n") + "\n}\n"
    header = _source("src/core/fused_model_base.h")
    residency_enum = _section(header, "enum class FmbForwardOperandResidency", "\n};") + "\n};\n"
    resident_bytes = _section(model, "constexpr int64_t GR00T_DIT_FF1_RESIDENT_BYTES", ";") + ";\n"
    harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <string>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(test, ...) do { if (!(test)) throw std::runtime_error("rejected"); } while (0)
FORWARD_RESIDENCY_ENUM
RESIDENT_BYTES
namespace at {struct Tensor {};}
constexpr int64_t NUM_CORES=8, DWIDTH=2;
int64_t Align(int64_t n,int64_t a){return (n+a-1)/a*a;}
int64_t CeilDiv(int64_t n,int64_t a){return (n+a-1)/a;}
int64_t gr00t_pad16(int64_t n){return Align(n,16);}
int64_t sdpa_compute_tmp_v16_size(int,int64_t n){return n*16;}
enum class StorageClass {Temp, TempPerLayer, Persistent, PersistentPerLayer};
enum class BufferScope {LayerWide};
enum class AttentionExecutionPolicy {DDR_KV, SPM_KV_BY_MHA};
enum class FmbRouteFamily {MUTABLE_DMA};
enum class Gr00tDiTMutableDmaRoute {DDR_BROADCAST_TO_SPM=2};
constexpr int64_t GR00T_DIT_DENOISE_TABLE_PRELOAD_SITE=1729441700529011334LL;
constexpr int64_t GR00T_DIT_DENOISE_BIAS_PRELOAD_SITE=941780559855581702LL;
struct Route {int64_t site; FmbRouteFamily family; int64_t selector, flags; std::vector<int64_t> args; int64_t invocation;};
struct Context {
    std::vector<Route> routes;
    void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,
                               std::vector<int64_t> args,int64_t invocation){
        routes.push_back({site,family,selector,flags,args,invocation});
    }
};
struct LayoutContext {
    int64_t chunk_size=16, batch_size=1;
    AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::DDR_KV;
    FmbForwardOperandResidency forward_operand_residency=FmbForwardOperandResidency::UNSPECIFIED;
};
struct BufferDecl;
struct FusedModelBase {virtual std::vector<BufferDecl> declare_buffers(const LayoutContext&)=0;};
struct BufferDecl {
    const char* name=nullptr; int64_t size=0; int phase_start=0,phase_end=0;
    StorageClass storage=StorageClass::Temp; int per_layer=0; const char* alias_of=nullptr;
    BufferScope scope=BufferScope::LayerWide; bool reverse_layer_alloc=false;
    std::function<void(FusedModelBase&,int,uint32_t)> preload_callback;
};
struct DmaCall {
    const at::Tensor* source; std::vector<int64_t> args;
    bool operator==(const DmaCall& other)const{return source==other.source && args==other.args;}
};
std::vector<DmaCall> dma_calls;
void rpu_launch_ddr_broadcast_spm_dma(const at::Tensor& source,int64_t offset,int64_t count,uint32_t addr,int64_t cores=8){
    dma_calls.push_back({&source,{0,offset,count,addr,cores}});
}
void rpu_launch_ddr_scatter_spm_dma(const at::Tensor& source,int64_t offset,int64_t count,int64_t stride,uint32_t addr,int64_t cores){
    dma_calls.push_back({&source,{1,offset,count,stride,addr,cores}});
}
void rpu_launch_memset_spm_multicore(uint32_t addr,int64_t count){
    dma_calls.push_back({nullptr,{2,addr,count}});
}
struct Gr00tDiTModel : FusedModelBase {
    struct LayerWeights {at::Tensor to_q_b,to_k_b,to_v_b,to_out_b,ff1_b,ff2_b;};
    std::vector<LayerWeights> layer_weights_=std::vector<LayerWeights>(32);
    at::Tensor ln_ones_,ln_zeros_,tau_table_,pos_table_,se1_b_,se2_b_,ae1_b_,ae2_b_,ae3_b_,dec1_b_,dec2_b_,po1_b_,po2_b_;
    bool denoise_ready_=true, denoise_mode_=true, loop_mode_=true;
    int64_t vl_seq_=82,cross_dim_=2048,action_horizon_=40,action_dim_pad_=144,num_steps_=4,enc_mid_=1024,dec_mid_=1024,model_out_dim_=1024;
    int64_t hidden_size(){return 1536;} int64_t num_q_heads(){return 32;}
    int64_t head_dim(){return 48;} int64_t intermediate_size(){return 6144;}
    int64_t num_layers(){return 32;} int make_sdpa_config(int n){return n;}
    Context context; Context& ctx(){return context;}
    // These opt-ins append Temp slots only. Persistent shape/preload identity
    // is exercised with the cold default residency; retain the actual FF1
    // declaration condition rather than stripping it from the source slice.
    void append_cross_kv_buffers(std::vector<BufferDecl>&,const LayoutContext&){}
    bool adaln_table_profile(int64_t,int64_t){return false;}
    int64_t adaln_table_body_bytes(){return 0;}
    bool ff1_resident_profile(int64_t,int64_t){return false;}
'''
    main = r'''
};
int main() {
    int64_t four_step_hash=0;
    for(int64_t steps:{4,10}) {
        Gr00tDiTModel model; LayoutContext layout;
        model.num_steps_=steps; model.denoise_mode_=false; model.loop_mode_=false;
        // Shared cold preparation seeds declarations at C16 before any
        // execution mode is active. It must match later actual/capacity rows.
        auto initial=model.declare_buffers(layout);
        const auto hash=compute_persistent_hash_impl(initial);
        if(steps==4) four_step_hash=hash; else assert(hash!=four_step_hash);
        auto preload=[&](const std::vector<BufferDecl>& decls) {
            std::vector<std::string> names;
            model.context.routes.clear(); dma_calls.clear();
            for(const auto& d:decls) {
                if(d.storage!=StorageClass::Persistent && d.storage!=StorageClass::PersistentPerLayer) continue;
                names.push_back(d.name);
                assert(d.preload_callback);
                const int layers=d.storage==StorageClass::PersistentPerLayer ? d.per_layer : 1;
                for(int layer=0;layer<layers;++layer) d.preload_callback(model,layer,1024);
            }
            const auto expected=model.preload_routes();
            assert(model.context.routes.size()==expected.size() && !expected.empty());
            for(size_t i=0;i<expected.size();++i) {
                const auto& a=model.context.routes[i]; const auto& b=expected[i];
                assert(a.site==b.site && a.family==b.family && a.selector==b.selector && a.flags==b.flags && a.args==b.args && a.invocation==b.invocation);
                assert(a.args[0]==1 && a.args[1]==steps);
            }
            return names;
        };
        const auto names=preload(initial);
        const auto cold_dma=dma_calls;
        assert(!cold_dma.empty());
        for(int64_t rows:{16,41,48}) for(int mode:{0,1,2,0,2}) {
            layout.chunk_size=rows;
            model.denoise_mode_=mode!=0; model.loop_mode_=mode==2;
            model.vl_seq_=mode==1 ? 148 : 82;
            auto decls=model.declare_buffers(layout);
            assert(compute_persistent_hash_impl(decls)==hash);
            bool action_scratch=false;
            for(const auto& d:decls) action_scratch |= std::string(d.name)=="x_t_spm";
            assert(action_scratch==(mode!=0));
            assert(preload(decls)==names && dma_calls==cold_dma);
        }
        model.denoise_ready_=false; model.denoise_mode_=false;
        auto plain_only=model.declare_buffers(layout);
        assert(compute_persistent_hash_impl(plain_only)!=hash && model.preload_routes().empty());
        for(const auto& d:plain_only) assert(std::string(d.name)!="tau_table");
    }
}
'''
    harness = harness.replace("FORWARD_RESIDENCY_ENUM", residency_enum).replace("RESIDENT_BYTES", resident_bytes)
    cpp = tmp_path / "gr00t_persistent.cpp"
    binary = tmp_path / "gr00t_persistent"
    cpp.write_text(harness + declaration + "\nstd::vector<Route> preload_routes(){struct {std::vector<Route> routes;} manifest;\n" + manifest + "return manifest.routes;}\n" + "};\n" + persistent_hash + main.replace("};\nint main()", "int main()", 1))
    compiled = subprocess.run(["g++", "-std=c++17", str(cpp), "-o", str(binary)], capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert run.returncode == 0, run.stderr
