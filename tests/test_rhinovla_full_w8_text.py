"""Full-profile text quantization preserves the seven real owner lists."""
from types import SimpleNamespace as NS
from pathlib import Path
import re
import shutil
import subprocess

import pytest
import torch

from rpu_backend.adapters.rhinovla import pipeline, precision


def text_model():
    model = torch.nn.Module()
    model.config = NS(hidden_size=256, intermediate_size=512, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=64, num_hidden_layers=2)
    model.layers = torch.nn.ModuleList()
    model.embed_tokens = torch.nn.Embedding(16, 256)
    model.norm = torch.nn.LayerNorm(256)
    for _ in range(2):
        layer = torch.nn.Module()
        layer.self_attn, layer.mlp = torch.nn.Module(), torch.nn.Module()
        shapes = ((256, 256), (128, 256), (128, 256), (256, 256),
                  (512, 256), (512, 256), (256, 512))
        for path, (n, k) in zip(precision.TEXT_PROJECTIONS, shapes, strict=True):
            parent, name = path.split(".")
            setattr(getattr(layer, parent), name, torch.nn.Linear(k, n, bias=False))
        model.layers.append(layer)
    return model.half()


def test_quantizes_all_text_linears_and_retains_floating_embedding_norm():
    model = text_model()
    embedding = model.embed_tokens.weight.clone()
    gamma = model.norm.weight.clone()
    original = model.layers[0].self_attn.k_proj.weight.detach().clone().float()
    precision.quantize_text_prefix(model)
    lists = precision.text_scale_lists(model)
    assert len(lists) == 7 and all(len(group) == 2 for group in lists)
    for index, path in enumerate(precision.TEXT_PROJECTIONS):
        for depth, layer in enumerate(model.layers):
            linear = layer.get_submodule(path)
            assert linear.weight.dtype == torch.int8
            assert lists[index][depth] is linear.weight_scale
            assert lists[index][depth].dtype == torch.float16
    linear = model.layers[0].self_attn.k_proj
    restored = linear.weight.float() * linear.weight_scale.float()[:, None]
    assert ((original - restored).abs() <= linear.weight_scale.float()[:, None] * .501).all()
    assert torch.equal(model.embed_tokens.weight, embedding)
    assert torch.equal(model.norm.weight, gamma)
    with pytest.raises(ValueError, match="fresh"):
        precision.quantize_text_prefix(model)


@pytest.mark.parametrize("fault", ("nonfinite", "bias", "shape", "extra"))
def test_bad_last_projection_rejects_without_partial_quantization(fault):
    model = text_model()
    last = model.layers[-1].mlp.down_proj
    if fault == "nonfinite":
        with torch.no_grad(): last.weight[-1, -1] = float("nan")
    elif fault == "bias":
        last.bias = torch.nn.Parameter(torch.zeros(last.out_features).half())
    elif fault == "shape":
        last.weight = torch.nn.Parameter(last.weight[:-1].detach())
    else:
        model.extra = torch.nn.Linear(256, 256).half()
    with pytest.raises(ValueError):
        precision.quantize_text_prefix(model)
    assert all(weight.dtype == torch.float16 for weight in model.parameters())
    assert not any(hasattr(module, "weight_scale") for module in model.modules())


def test_scale_gather_rejects_missing_owner():
    model = text_model()
    precision.quantize_text_prefix(model)
    del model.layers[-1].mlp.down_proj.weight_scale
    with pytest.raises(ValueError, match="complete"):
        precision.text_scale_lists(model)


@pytest.mark.parametrize("high_precision,high_precision_fusions,vector_k_norm,vector_q_norm,partial_rope,aligned_kv,fault", [
    (high, fused, False, False, False, False, fault)
    for high, fused in ((False, False), (True, False), (True, True), (False, True))
    for fault in (None, "adarms", "merger", "streaming", "text", "vision", "precision")
] + [(True, False, True, False, False, True, None),
     (False, False, True, False, False, True, None),
     (True, False, True, False, False, False, None),
     (True, False, False, True, False, False, None),
     (False, False, False, True, False, False, None),
     (True, True, True, True, False, True, None),
     (True, False, False, False, True, False, None),
     (False, False, False, False, True, False, None),
     (True, True, True, True, True, True, None)])
def test_full_profile_cold_admission_and_snapshot(
        monkeypatch, fault, high_precision, high_precision_fusions, vector_k_norm, vector_q_norm, partial_rope, aligned_kv):
    from rpu_backend.runtime import hw_attrs
    from rpu_backend.adapters.rhinovla.fused import _RhinoVLANativeColdConfig

    for name in _RhinoVLANativeColdConfig.__dataclass_fields__:
        monkeypatch.delenv("RPU_RHINOVLA_" + name.upper(), raising=False)
    monkeypatch.setenv("RPU_RHINOVLA_FULL_W8A16", "0" if fault == "precision" else "1")
    monkeypatch.setenv("RPU_RHINOVLA_HIGH_PRECISION", "1" if high_precision else "0")
    monkeypatch.setenv("RPU_RHINOVLA_HIGH_PRECISION_FUSIONS", "1" if high_precision_fusions else "0")
    monkeypatch.setenv("RPU_RHINOVLA_VECTOR_K_NORM", "1" if vector_k_norm else "0")
    monkeypatch.setenv("RPU_RHINOVLA_VECTOR_Q_NORM", "1" if vector_q_norm else "0")
    monkeypatch.setenv("RPU_RHINOVLA_PARTIAL_ROPE", "1" if partial_rope else "0")
    monkeypatch.setenv("RPU_RHINOVLA_ALIGNED_KV", "1" if aligned_kv else "0")
    monkeypatch.setenv("RPU_RHINOVLA_ADARMS_RESIDENT", "1" if high_precision else "0")
    monkeypatch.setenv("RPU_RHINOVLA_PRECOMPUTE_ADARMS", "0" if fault == "adarms" else "1")
    monkeypatch.setattr(hw_attrs, "validate_preinstall", lambda _: None)
    tc = NS(num_hidden_layers=28, hidden_size=2048, intermediate_size=6144,
            num_attention_heads=16, num_key_value_heads=8, head_dim=128)
    vc = NS(depth=24, hidden_size=1024, intermediate_size=4096, num_heads=16)
    if fault == "text": tc.num_hidden_layers = 27
    if fault == "vision": vc.depth = 23
    model = NS(qwen=NS(model=NS(model=NS(language_model=object(), visual=object()),
                               config=NS(text_config=tc, vision_config=vc))))
    cfg = NS(depth=18, mlp_dim=3072, action_horizon=30, use_state_token=True,
             width=1024, num_attention_heads=16, num_key_value_heads=8, head_dim=128)
    execution = {"action": {"denoise_unroll": True}, "vision": {
        "rpu_patch_embed": True, "rpu_mergers": True,
        "fused_merger": fault != "merger", "rpu_mergers_streaming": fault == "streaming",
    }}
    def preflight():
        return pipeline._preflight_rhinovla_cold_install(
            qin={}, prefix_len=230, steps=10, rpu_execution=execution, model=model,
            action_bundle=(object(), NS(), object(), cfg), flow_direction="official_descending")
    if partial_rope and not high_precision:
        with pytest.raises(ValueError, match="PARTIAL_ROPE requires HIGH_PRECISION"):
            preflight()
    elif vector_k_norm and not (high_precision and aligned_kv):
        with pytest.raises(ValueError, match="VECTOR_K_NORM requires HIGH_PRECISION and ALIGNED_KV"):
            preflight()
    elif vector_q_norm and not high_precision:
        with pytest.raises(ValueError, match="VECTOR_Q_NORM requires HIGH_PRECISION"):
            preflight()
    elif high_precision_fusions and not high_precision:
        with pytest.raises(ValueError, match="HIGH_PRECISION_FUSIONS requires HIGH_PRECISION"):
            preflight()
    elif fault == "precision" and not high_precision:
        assert not preflight()["high_precision"]
    elif fault:
        with pytest.raises(ValueError, match="FULL_W8A16|HIGH_PRECISION|PRECOMPUTE_ADARMS"):
            preflight()
    else:
        cold = preflight()
        monkeypatch.setenv("RPU_RHINOVLA_FULL_W8A16", "0")
        monkeypatch.setenv("RPU_RHINOVLA_HIGH_PRECISION", "0" if high_precision else "1")
        monkeypatch.setenv("RPU_RHINOVLA_HIGH_PRECISION_FUSIONS", "0" if high_precision_fusions else "1")
        monkeypatch.setenv("RPU_RHINOVLA_VECTOR_K_NORM", "0" if vector_k_norm else "1")
        monkeypatch.setenv("RPU_RHINOVLA_VECTOR_Q_NORM", "0" if vector_q_norm else "1")
        monkeypatch.setenv("RPU_RHINOVLA_PARTIAL_ROPE", "0" if partial_rope else "1")
        monkeypatch.setenv("RPU_RHINOVLA_ALIGNED_KV", "0" if aligned_kv else "1")
        assert cold["full_w8a16"] and cold["expert_w8a16"]
        assert cold["high_precision"] == high_precision
        assert cold["high_precision_fusions"] == high_precision_fusions
        assert cold["vector_k_norm"] == vector_k_norm
        assert cold["vector_q_norm"] == vector_q_norm
        assert cold["partial_rope"] == partial_rope


def test_high_precision_text_fusion_real_caller_and_shared_dispatch(tmp_path):
    """Compile real cold admission and the full shared MLP; stub only device calls."""
    from test_fmb_standalone_layout_rebuild import _definition

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    root = Path(__file__).resolve().parents[1]
    header = (root / "src/core/fused_model_base.h").read_text()
    base = (root / "src/core/fused_model_base.cpp").read_text()
    decoder = (root / "src/fused/rpu_qwen3_model.h").read_text()
    declaration = header[header.index("    void emit_mlp_pipeline("):].split(");", 1)[0] + ");"
    implementation = _definition(base, "void FusedModelBase::emit_mlp_pipeline(")
    setter = _definition(decoder, "    void set_rhinovla_high_precision_fusions(")
    consume = _definition(decoder, "    void consume_rhinovla_high_activation(")
    start = decoder.index("            if (rhinovla_high_precision_) consume_rhinovla_high_activation(rows, chunk.idx);")
    caller = decoder[start:decoder.index("rhinovla_high_precision_fusions_);", start) +
                     len("rhinovla_high_precision_fusions_);")]
    producer = re.search(r"append\(FmbRouteFamily::GRAPH_SCHEDULE, RHINOVLA_TEXT_HIGH_ACTIVATION_SITE,[\s\S]*?\);", decoder).group(0)
    constants = "\n".join(re.search(
        rf"inline constexpr int64_t {name}\s*=\s*\d+LL;", header + decoder).group(0)
        for name in sorted(set(re.findall(r"\bFMB_SHARED_MLP_\w+", implementation)) |
                           {"RHINOVLA_TEXT_HIGH_ACTIVATION_SITE"}))
    program = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
#define RECORD_FUNCTION(...)
namespace at {
enum ScalarType {kChar, kHalf};
struct Tensor {
    int id=0; ScalarType dtype=kHalf;
    bool defined() const {return id!=0;}
    ScalarType scalar_type() const {return dtype;}
};
}
namespace c10 {struct Half {explicit Half(float){}};}
enum class ActivationKind {SILU,GELU,NONE};
enum class RpuUnaryPrecision {BASE,HIGH};
enum class ValuOpType {SILU,ADD,MUL};
enum class GeluMode {NONE,TANH};
enum class FmbRouteFamily {LINEAR,ACTIVATION,ALL_REDUCE,GRAPH_SCHEDULE};
enum class FmbLinearRouteSelector {PI05_NVFP4_V2_ACC32,PI05_NVFP4_V2_ACC16,GEMV,AUTO_TILE,ACC32_OUT_BF16};
enum class FmbSharedAllReduceRouteSelector {PREPARE_RING_INPUT,BF16_PARTIAL_FP16_RESIDUAL,FP16_PARTIAL_BF16_RESIDUAL};
struct WallSortingConfig {unsigned fp16_transfer_mask=0;bool silu_mul=false;};
''' + constants + r'''
std::vector<std::string> launched;
struct Context {
    int64_t physical_route_invocation=0, expected_flags=0;
    void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,
                                std::vector<int64_t> args,int64_t invocation) const {
        TORCH_CHECK(family==FmbRouteFamily::GRAPH_SCHEDULE &&
            site==RHINOVLA_TEXT_HIGH_ACTIVATION_SITE && selector==1 && flags==expected_flags &&
            (args==std::vector<int64_t>{240,768,1}) && invocation==3);
    }
};
void consume_shared_runtime_route(Context&,FmbRouteFamily,int64_t,int64_t,int64_t=0,
                                  std::vector<int64_t> = {},int=8,int=8) {}
int64_t fmb_ring_all_reduce_route_selector(int64_t,int64_t,bool,int){return 1;}
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t,const at::Tensor& weight,uint32_t,
    int64_t rows,int64_t n,int64_t k,int partition,int cores,int,const at::Tensor& scale,
    uint32_t,uint16_t,bool acc32,bool gemv) {
    assert(rows==240 && cores==8 && !acc32 && !gemv);
    assert(weight.dtype==at::kChar && scale.dtype==at::kHalf && scale.id==weight.id+10);
    assert((weight.id==3 && n==2048 && k==6144 && partition==0) ||
           (weight.id!=3 && n==6144 && k==2048 && partition==1));
    launched.push_back(weight.id==1?"gate":weight.id==2?"up":"down");
}
void rpu_launch_eltwise_unary_spm_kernel(uint32_t x,uint32_t out,int64_t elems,ValuOpType op,
    GeluMode,int cores,RpuUnaryPrecision precision=RpuUnaryPrecision::BASE) {
    assert(x==1 && out==x && elems==240*768 && op==ValuOpType::SILU && cores==8);
    launched.push_back(precision==RpuUnaryPrecision::HIGH?"high_silu":"base_silu");
}
void rpu_launch_rhinovla_silu_high_mul_spm_kernel(uint32_t x,uint32_t y,uint32_t out,int64_t elems,int cores) {
    assert(x==1 && y==2 && out==x && elems==240*768 && cores==8);
    launched.push_back("high_silu_mul");
}
void rpu_launch_eltwise_binary_spm_kernel(uint32_t x,uint32_t y,uint32_t out,int64_t elems,
    ValuOpType op,c10::Half,int cores) {
    assert(x==1 && y==2 && out==x && elems==240*768 && op==ValuOpType::MUL && cores==8);
    launched.push_back("mul");
}
template<class... A> void rpu_launch_all_reduce_sum_residual_kernel(A...){launched.push_back("reduce");}
// Any alternate math path is a test failure, including a scale-dropping Linear.
template<class... A> void unexpected(A...){throw std::logic_error("unexpected launcher");}
#define rpu_launch_pi05_nvfp4_v2_kernel unexpected
#define rpu_launch_wall_action_gate_up_w8a16_multibase_spm_kernel unexpected
#define rpu_launch_silu_mul_spm_kernel unexpected
#define rpu_prepare_ring_all_reduce_input unexpected
#define rpu_launch_linear_spm_to_spm_acc32_out_bf16_kernel unexpected
#define rpu_launch_all_reduce_bf16_partial_fp16_residual_spm_kernel unexpected
#define rpu_launch_all_reduce_fp16_partial_bf16_residual_spm_kernel unexpected
bool active=false,quiescent=true,thread_allowed=true,kernels=true;
struct RpuKernelGraph {static bool has_active(){return active;}};
struct RpuExecutionCoordinator {
    static void require_graph_quiescent(const char*) {TORCH_CHECK(quiescent);}
    static void check_current_thread_execution_allowed(const char*) {TORCH_CHECK(thread_allowed);}
};
void rpu_require_high_precision_fusion_kernels(){TORCH_CHECK(kernels);}
bool wall_action_fused_rows_supported(int64_t, const WallSortingConfig&) { return false; }
class FusedModelBase {
public:
    struct Impl {Context ctx_;int64_t hidden_size_=2048,intermediate_size_=6144;} impl;
    Impl* pimpl_=&impl;
    bool rhinovla_precision_bound_=false,rhinovla_high_precision_=false;
    bool rhinovla_precision_fusions_bound_=false,rhinovla_high_precision_fusions_=false;
    bool use_silu_=true,linear_acc32_=false;
    WallSortingConfig wall_sorting_;
    bool wall_small_prefill_fused=false,wall_fp16_long_prefill_fused=false;
    bool wall_base_nvfp4_down_ring_norm=false,is_last_layer=false;
    int generation=0,last_chunk=0;
    int get_last_resolved_chunk_size() const {return last_chunk;}
    void invalidate_model_state(){++generation;}
    int num_cores() const {return 8;}
    int mlp_tp() const {return 8;}
    int64_t intermediate_size() const {return impl.intermediate_size_;}
    const Context& ctx() const {return impl.ctx_;}
    uint32_t addr(int,const char* name){return std::string(name)=="gate"?1:std::string(name)=="up"?2:3;}
    uint32_t nvfp4_scale_addr(const char*) const {return 0;}
    bool use_planned_silu_mul() const {return false;}
    bool qwen3vl_decode_gemv(int64_t) const {return false;}
    bool wall_action_fused_rows_supported(int64_t,const WallSortingConfig&) const {return false;}
''' + declaration + setter + consume + r'''
    void bind_expected_policy() {
        struct {int64_t len=240,idx=3;} chunk;
        auto append=[&](FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,
                        std::vector<int64_t> args,int64_t invocation) {
            impl.ctx_.expected_flags=flags;
            impl.ctx_.consume_physical_route(family,site,selector,flags,args,invocation);
        };
''' + producer + r'''
    }
    void run_text(bool bad_legacy=false,bool bad_wall=false,bool bad_bf16=false) {
        struct {at::Tensor gate_w{1,at::kChar},up_w{2,at::kChar},down_w{3,at::kChar};
                at::Tensor gate_ws{11,at::kHalf},up_ws{12,at::kHalf},down_ws{13,at::kHalf};} lw;
        struct {int64_t idx=3;} chunk;
        const int64_t rows=240,layer_idx=0,seq_len_=240;
        const bool transition_to_bf16=bad_bf16,residual_is_bf16=false;
        const bool wall_p640_staged0_w8a16=bad_legacy,wall_action_w8a16_silu_mul=bad_wall;
        const bool rtc_prefill_mixed=false,wall_action_fuse_silu_mul_=false;
        const bool wall_action_linear_ring_postnorm=false,wall_action_resident_layer=false;
        const bool wall_action_fp16_transfer=false;
        const bool wall_small_prefill_fused=false,wall_fp16_long_prefill_fused=false;
        const bool wall_base_nvfp4_down_ring_norm=false,is_last_layer=false;
        const uint32_t wall_action_weight_base=0,WALL_ACTION_GATE_OFFSET=0,WALL_ACTION_UP_OFFSET=0;
''' + caller + r'''
    }
};
''' + implementation + r'''
template<class F> void reject(F fn) {
    launched.clear();bool rejected=false;
    try {fn();} catch(const std::runtime_error&) {rejected=true;}
    assert(rejected && launched.empty());
}
int main() {
    FusedModelBase base;
    reject([&]{base.set_rhinovla_high_precision_fusions(true);});
    assert(base.generation==0 && !base.rhinovla_precision_fusions_bound_);
    base.run_text();
    assert((launched==std::vector<std::string>{"gate","base_silu","up","mul","down","reduce"}));
    launched.clear();
    // An existing caller that omits both precision arguments retains BASE.
    base.emit_mlp_pipeline({1,at::kChar},{2,at::kChar},{3,at::kChar},240,ActivationKind::SILU,
                           {11,at::kHalf},{12,at::kHalf},{13,at::kHalf});
    assert((launched==std::vector<std::string>{"gate","base_silu","up","mul","down","reduce"}));
    for (int fault=0;fault<6;++fault) {
        FusedModelBase model;model.rhinovla_high_precision_=model.rhinovla_precision_bound_=true;
        if(fault==0) active=true;
        if(fault==1) quiescent=false;
        if(fault==2) thread_allowed=false;
        if(fault==3) kernels=false;
        if(fault==4) model.last_chunk=1;
        if(fault==5) model.rhinovla_precision_bound_=false;
        reject([&]{model.set_rhinovla_high_precision_fusions(true);});
        assert(!model.rhinovla_high_precision_fusions_ && !model.rhinovla_precision_fusions_bound_ && model.generation==0);
        active=false;quiescent=thread_allowed=kernels=true;
    }
    for(bool fused : {false,true}) {
        FusedModelBase model;model.rhinovla_high_precision_=model.rhinovla_precision_bound_=true;
        model.set_rhinovla_high_precision_fusions(fused);
        assert(model.generation==1 && model.rhinovla_precision_fusions_bound_);
        model.bind_expected_policy();
        assert(model.impl.ctx_.expected_flags==(fused?1:0));
        launched.clear();model.run_text();
        assert(launched==(fused?std::vector<std::string>{"gate","up","high_silu_mul","down","reduce"}:
                               std::vector<std::string>{"gate","high_silu","up","mul","down","reduce"}));
        reject([&]{model.set_rhinovla_high_precision_fusions(!fused);});
        assert(model.generation==1 && model.rhinovla_high_precision_fusions_==fused);
        model.impl.ctx_.expected_flags^=1;
        reject([&]{model.run_text();});
        model.impl.ctx_.expected_flags^=1;
        if(fused) {
            reject([&]{model.run_text(true);});
            reject([&]{model.run_text(false,true);});
            reject([&]{model.run_text(false,false,true);});
            model.rhinovla_high_precision_=false;
            reject([&]{model.run_text();});
        }
    }
}
'''
    source, executable = tmp_path / "text_high_fusion.cpp", tmp_path / "text_high_fusion"
    source.write_text(program)
    built = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(executable)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stdout + built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stdout + ran.stderr
