"""Cold admission, canonical FP16 packing and native SPM launcher contracts."""
import dataclasses

import pytest
import torch

from rpu_backend.adapters.hy_vla import runtime, weights
from test_hyvla_lazy_weights import _clear_hyvla_cold_environment
from test_hyvla_paired_kv import COMMON, ROOT, block, compile_run

KEY = "RPU_HY_VLA_EXPERT_GATEUP_RESIDENT"
LAYERS_KEY = "RPU_HY_VLA_EXPERT_GATEUP_RESIDENT_LAYERS"


def test_twelve_layer_option_is_strict_w8_only_and_frozen(monkeypatch):
    _clear_hyvla_cold_environment(monkeypatch)
    env = {KEY: "1", LAYERS_KEY: "12", "RPU_HY_VLA_W8A16": "all",
           "RPU_HY_VLA_ATTN_TP8": "1"}
    plan = runtime.prepare_hy_vla_cold_plan(env)
    env[LAYERS_KEY] = "6"
    assert plan.weights.gateup_resident_layers == 12
    assert runtime.prepare_hy_vla_cold_plan(env).weights.gateup_resident_layers == 6
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.weights.gateup_resident_layers = 6
    for value in ("0", "7", "11", "13", "24", "all", "6.0", "012", "-12"):
        with pytest.raises(ValueError, match=LAYERS_KEY):
            runtime.prepare_hy_vla_cold_plan(env | {LAYERS_KEY: value})
    for override in ({KEY: "0"}, {"RPU_HY_VLA_W8A16": "0"},
                     {"RPU_HY_VLA_W4A16": "expert"}):
        with pytest.raises(ValueError, match=LAYERS_KEY):
            runtime.prepare_hy_vla_cold_plan(env | {LAYERS_KEY: "12"} | override)


@pytest.mark.parametrize("count", [6, 12])
def test_w8_builder_retains_all_requested_installed_layer_bytes(monkeypatch, count):
    calls = []
    class Source:
        def __getitem__(self, key):
            calls.append(key)
            li = int(key.split(".")[1])
            tag = 2 * li + (1 if "gate_proj" in key else 2)
            return torch.tensor([tag if ".mlp_v." in key and "down_proj" not in key else 0])
    # Stand in for installed INT8 DDR tensors, including all FP16 carrier bit patterns.
    base = torch.arange(2048 * 1024, dtype=torch.int32).reshape(2048, 1024)
    def quant(raw, partition, cores, bits):
        assert bits == 8
        q = (base + int(raw.item())).to(torch.int8) if raw.item() else torch.zeros(1, dtype=torch.int8)
        return q, torch.ones(1, dtype=torch.float16)
    monkeypatch.setattr(weights, "_quant", quant)
    monkeypatch.setattr(weights, "_dup_kv", lambda raw, *a, **kw: raw)
    monkeypatch.setattr(weights, "_rpu16", lambda raw: raw.half())
    cold = weights._HyVlaWeightColdPlan.from_environment({KEY: "1", LAYERS_KEY: str(count),
        "RPU_HY_VLA_W8A16": "all", "RPU_HY_VLA_ATTN_TP8": "1"})
    result = weights._build_expert(Source(), Source(), weights.HyVlaConfig(), cold)
    bundle, = result["gateup_resident"]
    assert bundle.dtype == torch.float16 and bundle.shape == (8, count * 262144)
    pairs = bundle.view(torch.int8).reshape(8, count, 2, 262144)
    for li in range(count):
        for j, key in enumerate(("gate_w", "up_w")):
            installed = result[key][li].reshape(16, 32, 8, 512)
            for core in range(8):
                assert torch.equal(pairs[core, li, j], installed[:, :, core].reshape(-1))
    for li in range(32):
        for kind in ("gate", "up"):
            assert calls.count(f"layers.{li}.mlp_v.{kind}_proj.weight") == 1


def test_cold_residency_is_opt_in_strict_and_frozen(monkeypatch):
    _clear_hyvla_cold_environment(monkeypatch)
    assert not runtime.prepare_hy_vla_cold_plan().weights.gateup_resident
    environment = {KEY: "1", "RPU_HY_VLA_ATTN_TP8": "1"}
    plan = runtime.prepare_hy_vla_cold_plan(environment)
    environment[KEY] = "0"
    assert plan.weights.gateup_resident
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.weights.gateup_resident = False
    for value in ("all", "expert", "2", "yes"):
        with pytest.raises(ValueError, match="strict boolean"):
            runtime.prepare_hy_vla_cold_plan({KEY: value})


@pytest.mark.parametrize("key,value", [
    ("RPU_HY_VLA_W4A16", "expert"),
    ("RPU_HY_VLA_ATTN_TP8", "0"), ("RPU_HY_VLA_ACTION_MLP_MC", "0"),
    ("RPU_HY_VLA_DENOISE_UNROLL", "0"), ("RPU_HY_VLA_SILU_MUL", "vlm"),
])
def test_incompatible_modes_reject_before_checkpoint_or_owner(monkeypatch, tmp_path, key, value):
    _clear_hyvla_cold_environment(monkeypatch)
    monkeypatch.setenv(KEY, "1")
    monkeypatch.setenv("RPU_HY_VLA_ATTN_TP8", "1")
    monkeypatch.setenv(key, value)
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    def unexpected(*args, **kwargs):
        pytest.fail("incompatible residency reached checkpoint or owner")
    monkeypatch.setattr(runtime, "preflight_hy_vla_checkpoint", unexpected)
    monkeypatch.setattr("rpu_backend.api.causal_lm._claim_live_instance", unexpected)
    with pytest.raises(ValueError, match=KEY):
        runtime.build_hy_vla(checkpoint)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_compact_tiles_match_existing_ddr_swizzle_and_rounding(dtype):
    # Non-contiguous, core/tile-dependent source with non-FP16 values. Compare
    # against the pre-existing layout transformer, not a second packing recipe.
    raw = ((torch.arange(2048 * 1024).reshape(1024, 2048) % 8191) / 97.3).to(dtype).T
    compact = weights._pack_expert_gateup_weight(raw)
    ddr = weights.tp_col_swizzle_mc_weight(raw.half(), 8).reshape(16, 64, 8, 16, 16)
    assert compact.dtype == torch.float16 and compact.is_contiguous()
    assert compact.shape == (8, 262144)
    for core in range(8):
        assert torch.equal(compact[core].reshape(16, 64, 16, 16).view(torch.int16),
                           ddr[:, :, core].contiguous().view(torch.int16))
    with pytest.raises(ValueError, match="CPU"):
        weights._pack_expert_gateup_weight(raw[:1024])
    with pytest.raises(ValueError, match="floating-point"):
        weights._pack_expert_gateup_weight(raw.long())


def test_builder_reads_gateup_once_and_retains_six_distinct_layer_pairs(monkeypatch):
    calls = []
    class Source:
        def __getitem__(self, key):
            calls.append(key)
            layer = int(key.split(".")[1])
            if ".mlp_v.gate_proj" in key or ".mlp_v.up_proj" in key:
                value = layer + (0.25 if "gate_proj" in key else 0.5)
                return torch.full((2048, 1024), value)
            return torch.ones(1)
    monkeypatch.setattr(weights, "_quant", lambda *args: (torch.zeros(1), None))
    monkeypatch.setattr(weights, "_dup_kv", lambda raw, *args, **kwargs: raw)
    monkeypatch.setattr(weights, "_rpu16", lambda raw: raw.half())
    cold = weights._HyVlaWeightColdPlan.from_environment({KEY: "1", "RPU_HY_VLA_ATTN_TP8": "1"})
    result = weights._build_expert(Source(), Source(), weights.HyVlaConfig(), cold)
    bundle, = result["gateup_resident"]
    assert bundle.shape == (8, 3145728)
    pairs = bundle.reshape(8, 6, 2, 262144)
    for layer in range(6):
        assert torch.all(pairs[:, layer, 0] == layer + 0.25)
        assert torch.all(pairs[:, layer, 1] == layer + 0.5)
    for layer in range(32):
        for kind in ("gate", "up"):
            assert calls.count(f"layers.{layer}.mlp_v.{kind}_proj.weight") == 1
    assert len(result["gate_w"]) == len(result["up_w"]) == 32
    with pytest.raises(ValueError, match="exact"):
        weights._build_expert(Source(), Source(), dataclasses.replace(weights.HyVlaConfig(), layers=31), cold)


@pytest.mark.parametrize("fused", [False, True])
def test_actual_launcher_spm_boundaries_and_typed_registers(tmp_path, fused):
    launch = block((ROOT / "src/ops/rpu_hyvla_gateup_resident.h").read_text(),
                   "inline void launch_hyvla_gateup_resident_spm(")
    source = COMMON + r'''
#include "ops/rpu_linear_tiling.h"
enum class KernelId : uint16_t {WALL_ACTION_GATE_UP_FP16_SPM_WEIGHT_M592N32K128};
struct Kernel {std::array<uint16_t,132> regs{};int counter_writes=0;
 void reset_regs(){regs.fill(0);}
 uint32_t set_regs(uint32_t i,uint16_t v){regs.at(i)=v;return 0;}
 uint32_t set_regs(uint32_t i,uint32_t v){assert(i==128 && v==0);++counter_writes;regs.at(i)=v;return 0;}
 uint32_t set_regs(uint32_t,int32_t)=delete;
} kernel;
int lookups=0,queued=0;bool available=true;
Kernel* get_kernel(){++lookups;return available?&kernel:nullptr;}
#define GET_KERNEL(id) get_kernel()
struct RpuKernelGraph {static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 Kernel* get_kernel_reset(const std::string& name){assert(name=="hyvla_resident_swiglu_fp16_v2");return get_kernel();}};
struct Queue {bool broadcast=false;void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel&,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
  assert(broadcast && grid==std::vector<uint16_t>({6,1,1}));
  assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));++queued;}
} queue;
#define GET_QUEUE(n) (&::queue)
struct SpmAllocator {static constexpr uint32_t SPM_USABLE=8387584;bool ready=true;
 uint32_t base=0x20000000;bool is_initialized()const{return ready;}
 uint32_t addr(int c,uint32_t off)const{return base+uint32_t(c)*0x1000000u+off;}
} spm;
#define SPM_ALLOC spm
LAUNCH
int main(){
 const std::array<uint32_t,5> good{0x20000000,0x20020000,0x20030000,0x20100000,0x20180000};
 auto run=[&](std::array<uint32_t,5> a,bool expected,int rows=51){
  const int old=queued,before=lookups;bool accepted=true;
  try{launch_hyvla_gateup_resident_spm(a[0],a[1],a[2],a[3],a[4],rows,FUSED);}
  catch(const std::exception&){accepted=false;}
  assert(accepted==expected && queued==old+int(expected));
  assert(kernel.counter_writes == 0);
  if(available)assert(lookups==before+int(expected));
 };
 run(good,true);
 auto pair=[&](int lo){return uint32_t(kernel.regs[lo])|(uint32_t(kernel.regs[lo+1])<<16);};
 assert(pair(0)==good[0] && pair(2)==good[3] && pair(4)==good[1]);
 assert(pair(32)==good[4] && pair(36)==good[2]);
 assert(kernel.regs[8]==1 && kernel.regs[9]==51 && pair(10)==256 && kernel.regs[12]==1024 && kernel.regs[13]==64);
 assert(kernel.regs[14]==0 && kernel.regs[15]==8 && pair(16)==2048 && pair(18)==512);
 assert(pair(20)==0 && pair(22)==0 && pair(24)==0 && kernel.regs[26]==2 && kernel.regs[27]==2);
 for(int rows:{0,20,50,64})run(good,false,rows);
 for(int i=0;i<5;++i){auto a=good;++a[i];run(a,false);a=good;a[i]=0x1fffff00;run(a,false);
  a=good;a[i]=0x207fff00;run(a,false);}
 for(int i=0;i<5;++i)for(int j=0;j<i;++j){auto a=good;a[i]=a[j];run(a,false);}
 auto edge=good;edge[4]=spm.base+SpmAllocator::SPM_USABLE-524288;run(edge,true);
 edge[4]+=256;run(edge,false);
 spm.ready=false;run(good,false);spm.ready=true;
 spm.base=0xf8ff0000;auto overflow=good;for(auto& a:overflow)a+=0xd8ff0000;run(overflow,false);
 spm.base=0x20000000;available=false;run(good,false);
}
'''
    source = source.replace("FUSED", "true" if fused else "false")
    if fused: source = source.replace("({6,1,1})", "({3,1,1})")
    compile_run(tmp_path, "resident_launch", source.replace("LAUNCH", launch))


def test_actual_owner_cold_install_and_profile(tmp_path):
    decoder = (ROOT / "src/fused/rpu_hyvla_expert_model.cpp").read_text()
    methods = "\n".join(block(decoder, marker).replace("HyVlaExpertModel::", "") for marker in (
        "bool HyVlaExpertModel::optimized_prefix_supported(",
        "bool HyVlaExpertModel::gateup_resident_profile(",
        "std::vector<int64_t> HyVlaExpertModel::gateup_resident_arguments(",
        "bool HyVlaExpertModel::down_nextnorm_enabled(",
        "std::vector<int64_t> HyVlaExpertModel::down_nextnorm_arguments(",
        "bool HyVlaExpertModel::down_nvfp4_n128_enabled(",
        "std::vector<int64_t> HyVlaExpertModel::down_nvfp4_n128_arguments(",
        "bool HyVlaExpertModel::down_nvfp4_nextnorm_enabled(",
        "std::vector<int64_t> HyVlaExpertModel::down_nvfp4_nextnorm_arguments(",
        "bool HyVlaExpertModel::qk_rope_enabled(",
        "std::vector<int64_t> HyVlaExpertModel::qk_rope_arguments(",
        "bool HyVlaExpertModel::sdpa_k128_enabled(",
        "std::vector<int64_t> HyVlaExpertModel::sdpa_k128_arguments(",
        "bool HyVlaExpertModel::sdpa_m16_enabled(",
        "std::vector<int64_t> HyVlaExpertModel::sdpa_m16_arguments(",
        "bool HyVlaExpertModel::kv_grouped_enabled(",
        "std::vector<int64_t> HyVlaExpertModel::kv_grouped_arguments(",
        "int HyVlaExpertModel::nvfp4_qkv_mode(",
        "bool HyVlaExpertModel::nvfp4_q_n32_enabled(",
        "std::vector<int64_t> HyVlaExpertModel::nvfp4_q_n32_arguments(",
        "std::vector<int64_t> HyVlaExpertModel::nvfp4_qkv_arguments(",
        "bool HyVlaExpertModel::qk_rope_norm_enabled(",
        "std::vector<int64_t> HyVlaExpertModel::qk_rope_norm_arguments(",
        "bool HyVlaExpertModel::qk_rope_norm_kv_enabled(",
        "std::vector<int64_t> HyVlaExpertModel::qk_rope_norm_kv_arguments(",
        "bool HyVlaExpertModel::paired_kv_enabled(",
        "std::vector<int64_t> HyVlaExpertModel::paired_kv_arguments(",
        "bool HyVlaExpertModel::o_resident_enabled(",
        "std::vector<int64_t> HyVlaExpertModel::o_resident_arguments(",
        "bool HyVlaExpertModel::o_ring_postnorm_enabled(",
        "std::vector<int64_t> HyVlaExpertModel::o_ring_postnorm_arguments(",
        "bool HyVlaExpertModel::ring_postnorm_enabled(",
        "std::vector<int64_t> HyVlaExpertModel::ring_postnorm_arguments(",
        "bool HyVlaExpertModel::xor3_norm_enabled(",
        "bool HyVlaExpertModel::resident_fp8_scales_enabled(",
        "std::vector<int64_t> HyVlaExpertModel::resident_fp8_scales_arguments(",
        "bool HyVlaExpertModel::resident_swiglu_enabled(",
        "std::vector<int64_t> HyVlaExpertModel::resident_swiglu_arguments(",
        "bool HyVlaExpertModel::gateup_swiglu_enabled(",
        "std::vector<int64_t> HyVlaExpertModel::gateup_swiglu_arguments(",
        "void HyVlaExpertModel::set_gateup_resident_weights("))
    methods = methods.replace("std::vector<int64_t> paired_kv_arguments(", "static std::vector<int64_t> paired_kv_arguments(")
    header = (ROOT / "src/fused/rpu_hyvla_expert_model.h").read_text()
    byte_methods = "\n".join(block(header, marker) for marker in (
        "int64_t gateup_weight_bytes(", "int64_t gateup_layer_bytes(",
        "int64_t gateup_resident_layers(", "int64_t gateup_bundle_bytes(", "int64_t resident_storage_bytes("))
    identity_method = block(header, "std::vector<int64_t> kvinsert_cost_weight_identity(").replace(" override", "")
    source = COMMON + r'''
namespace at {constexpr auto kPrivateUse1=c10::DeviceType::PrivateUse1;}
int64_t Align(int64_t n,int64_t a){return ((n+a-1)/a)*a;}
enum class RpuRmsNormSpmRoute {BASE,V16_GRID8,HYVLA_V16_BITCOPY_GRID8};
struct RpuRmsNormSpmContract {bool v16_bitcopy=true;bool vector=true;int capability=1;bool q_vector=true,k_vector=true;
 RpuRmsNormSpmRoute resolve(int r,int c)const{assert((r==64 && c==1024)||((r==112||r==64)&&c==128));
  return vector && (c!=128 || (r==112?q_vector:k_vector))?RpuRmsNormSpmRoute::HYVLA_V16_BITCOPY_GRID8:RpuRmsNormSpmRoute::BASE;}};
struct KernelCache {bool available=true;
 static KernelCache& instance(){static KernelCache c;return c;}
 void* get_program(const char*){return available?this:nullptr;}
};
void append_kvinsert_cost_scalar_identity(std::vector<int64_t>&,double){}
void append_kvinsert_cost_tensor_identity(std::vector<int64_t>&,const at::Tensor&){}
struct Model {
 static constexpr int64_t kGateUpWeightBytes=524288;
 static constexpr int64_t kGateUpResidentLayers=6,kGateUpLayerBytes=1048576,kGateUpBundleBytes=6291456;
 struct LayerW {at::Tensor q_w,k_w,v_w,o_w,gate_w,up_w,down_w;
  at::Tensor q_ws,k_ws,v_ws,o_ws,gate_ws,up_ws,down_ws;};
 std::vector<LayerW> layers_{32};at::Tensor gateup_resident_bundle_;
 bool cold_w8_gateup_n48_available_=false;bool cold_w8_optimizations_available_=false;
 bool cold_w8_resident_swiglu_available_=false,cold_w8_o_ring_postnorm_available_=false,cold_xor3_norm_available_=false;
 bool cold_nvfp4_resident_swiglu_available_=false;
 bool cold_nvfp4_resident_scales_available_=false,cold_nvfp4_resident_scales_swiglu_available_=false;
 int64_t resident_fp8_scale_layers_=0;
 bool cold_w8_o_resident_available_=false;int64_t o_resident_layers_=0;
 bool nvfp4_=false,cold_nvfp4_gateup_available_=false,cold_nvfp4_o_ring_postnorm_available_=false,cold_nvfp4_o_n128_available_=false;
 bool cold_nvfp4_gateup_input_reuse_available_=false;
 bool cold_nvfp4_down_n128_available_=false;
 bool cold_nvfp4_down_nextnorm_available_=false;
 bool cold_nvfp4_down_nextnorm_n128_available_=false;
 bool cold_fast_replay_=false,cold_fast_replay_preload_=false,cold_mask_once_=false;
 bool cold_kvpad16_=false,cold_paired_kv_available_=false;
 at::Tensor wc_,mo_,mo_bias_,op_,op_bias_;
 std::vector<int64_t> causal_kvinsert_cost_weight_identity()const{return {7};}
 bool cold_gateup_swiglu_available_=true,cold_resident_swiglu_available_=false;
 bool cold_ring_postnorm_available_=false,cold_o_ring_postnorm_available_=false,cold_rmsnorm_pad16_=true;
 bool cold_down_nextnorm_available_=false,cold_qk_rope_available_=false,cold_partial_rope_=true;
 bool cold_qk_rope_norm_available_=false,cold_qk_rope_norm_kv_available_=false;
 bool cold_fp16_kv_grouped_available_=false,cold_w8_kv_grouped_available_=false;
 bool cold_nvfp4_kv_available_=false,cold_nvfp4_qkv_available_=false;
 bool cold_nvfp4_q_n32_available_=false,cold_nvfp4_qkv_overlap_available_=false;
 bool cold_nvfp4_o_overlap_available_=false,cold_nvfp4_down_overlap_available_=false;
 bool cold_nvfp4_gateup_overlap_available_=false,cold_nvfp4_resident_overlap_available_=false;
 bool cold_sdpa_m16_available_=false,cold_sdpa_k128_available_=false;
 double eps_expert_=1e-5;RpuRmsNormSpmContract cold_rmsnorm_spm_contract_{};

 bool unroll_mode_=true,cold_silu_mul_=true;int64_t num_steps_=10,expert_chunk_size_=51,action_mlp_cores_=8;
 int l=32,h=1024,i=2048,q=16,kv=8,d=128,tp=8,chunk=0,invalidated=0;
 int num_layers()const{return l;}int hidden_size()const{return h;}int intermediate_size()const{return i;}
 int num_q_heads()const{return q;}int num_kv_heads()const{return kv;}int head_dim()const{return d;}
 int attn_tp()const{return tp;}int get_last_resolved_chunk_size()const{return chunk;}
 void invalidate_model_state(){++invalidated;}
 Model(){gateup_resident_bundle_.present=false;for(auto& w:layers_)
  for(auto* s:{&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.gate_ws,&w.up_ws,&w.down_ws})s->present=false;}
 BYTE_METHODS
 IDENTITY_METHOD
 METHODS
};
int main(){
 Model base;at::Tensor bundle;bundle.shape={8,3145728};
 auto install=[&](Model& m,at::Tensor b,bool expected,int64_t o_layers=0){bool accepted=true;const int old=m.invalidated;
  try{m.set_gateup_resident_weights(b,o_layers);}catch(const std::exception&){accepted=false;}
  assert(accepted==expected && m.invalidated==old+int(expected));};
 assert(!base.gateup_swiglu_enabled(51,240));
 assert(base.gateup_resident_arguments()==std::vector<int64_t>({4,6,32,51,1024,2048,8,10,1048576,6291456,112,112,384,6,2,32,20,34,4,36}));
 for(int* p:{&base.l,&base.h,&base.i,&base.q,&base.kv,&base.d,&base.tp}){
  ++*p;install(base,bundle,false);--*p;}
 for(int64_t* p:{&base.num_steps_,&base.expert_chunk_size_,&base.action_mlp_cores_}){
  ++*p;install(base,bundle,false);--*p;}
 base.unroll_mode_=false;install(base,bundle,false);base.unroll_mode_=true;
 base.cold_silu_mul_=false;install(base,bundle,false);base.cold_silu_mul_=true;
 base.layers_.back().up_w.dtype=at::kChar;install(base,bundle,false);base.layers_.back().up_w.dtype=at::kHalf;
 base.layers_.back().up_ws.present=true;install(base,bundle,false);base.layers_.back().up_ws.present=false;
 base.chunk=51;install(base,bundle,false);base.chunk=0;
 for(int i=0;i<2;++i){auto b=bundle;--b.shape[i];install(base,b,false);}
 {auto b=bundle;b.contiguous=false;install(base,b,false);}
 {auto b=bundle;b.dtype=at::kFloat;install(base,b,false);}
 {auto b=bundle;b.where.kind=c10::DeviceType::CPU;install(base,b,false);}
 {auto b=bundle;b.where.index=1;install(base,b,false);}
 KernelCache::instance().available=false;install(base,bundle,false);KernelCache::instance().available=true;
 install(base,bundle,true);assert(base.gateup_resident_bundle_.address==bundle.address);
 assert(base.gateup_swiglu_enabled(51,240));
 assert(!base.o_ring_postnorm_enabled(51,240));
 base.cold_o_ring_postnorm_available_=true;assert(!base.o_ring_postnorm_enabled(51,240));
 assert(!base.ring_postnorm_enabled(51,240));
 base.cold_ring_postnorm_available_=true;assert(base.ring_postnorm_enabled(51,240));
 assert(base.o_ring_postnorm_enabled(51,240));
 assert(!base.qk_rope_enabled(51,240));base.cold_qk_rope_available_=true;
 assert(base.qk_rope_enabled(51,240));
 assert(!base.down_nextnorm_enabled(51,240));base.cold_down_nextnorm_available_=true;
 assert(base.down_nextnorm_enabled(51,240));
 assert(!base.down_nextnorm_enabled(50,240) && !base.down_nextnorm_enabled(51,112));
 base.cold_qk_rope_available_=false;assert(!base.down_nextnorm_enabled(51,240));base.cold_qk_rope_available_=true;
 assert(!base.qk_rope_enabled(50,240) && !base.qk_rope_enabled(51,112));
 base.cold_partial_rope_=false;assert(!base.qk_rope_enabled(51,240));base.cold_partial_rope_=true;
 assert(!base.o_ring_postnorm_enabled(50,240) && !base.o_ring_postnorm_enabled(51,112));
 assert(!base.ring_postnorm_enabled(50,240) && !base.ring_postnorm_enabled(51,112));
 base.cold_rmsnorm_pad16_=false;assert(!base.ring_postnorm_enabled(51,240));base.cold_rmsnorm_pad16_=true;
 base.eps_expert_=1e-6;assert(!base.ring_postnorm_enabled(51,240));base.eps_expert_=1e-5;
 base.cold_rmsnorm_spm_contract_.vector=false;assert(!base.ring_postnorm_enabled(51,240));base.cold_rmsnorm_spm_contract_.vector=true;
 base.cold_gateup_swiglu_available_=false;assert(!base.ring_postnorm_enabled(51,240));base.cold_gateup_swiglu_available_=true;
 assert(!base.resident_swiglu_enabled(51,240));
 base.cold_resident_swiglu_available_=true;assert(base.resident_swiglu_enabled(51,240));
 assert(!base.resident_swiglu_enabled(50,240) && !base.resident_swiglu_enabled(51,112));
 base.cold_gateup_swiglu_available_=false;assert(!base.resident_swiglu_enabled(51,240));
 base.cold_gateup_swiglu_available_=true;
 assert(base.gateup_swiglu_arguments(240)==std::vector<int64_t>({4,32,6,26,51,240,1024,2048,256,8,10,96,112,384,3,2,32,20,34,26,27,28,29}));
 assert(!base.gateup_swiglu_enabled(50,240) && !base.gateup_swiglu_enabled(51,112));
 base.cold_gateup_swiglu_available_=false;assert(!base.gateup_swiglu_enabled(51,240));
 base.cold_gateup_swiglu_available_=true;
 base.layers_.back().down_w.dtype=at::kChar;assert(!base.gateup_swiglu_enabled(51,240));
 base.layers_.back().down_w.dtype=at::kHalf;
 install(base,bundle,false);
 Model quant;at::Tensor qb;qb.shape={8,1572864};
 for(auto& w:quant.layers_){
  for(auto* t:{&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.gate_w,&w.up_w,&w.down_w})t->dtype=at::kChar;
  for(auto* t:{&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.gate_ws,&w.up_ws,&w.down_ws}){t->present=true;t->shape={2048};}
 }
 install(quant,qb,false);quant.cold_w8_optimizations_available_=true;
 {auto bad=qb;bad.shape[1]=7*262144;install(quant,bad,false);}
 assert(quant.gateup_resident_arguments()[0]==5);
 Model mixed=quant;
 install(mixed,bundle,false,12);mixed.cold_w8_o_resident_available_=true;
 install(mixed,qb,false,12);install(mixed,bundle,false,1);install(base,bundle,false,12);
 install(mixed,bundle,true,12);
 assert(mixed.gateup_resident_layers()==6 && mixed.gateup_bundle_bytes()==3145728 && mixed.resident_storage_bytes()==6291456);
 assert(mixed.o_resident_enabled(51,240) && !mixed.o_resident_enabled(50,240) && !mixed.o_resident_enabled(51,112));
 mixed.cold_ring_postnorm_available_=true;mixed.cold_w8_o_ring_postnorm_available_=true;
 assert(!mixed.o_ring_postnorm_enabled(51,240));
 assert(mixed.o_resident_arguments(240)==std::vector<int64_t>({2,32,51,240,1024,2048,8,10,0,12,3145728,3145728,288,128,128,8,20,21}));
 install(mixed,bundle,false,12);
 // nvfp4_ models the base setter having validated packed tensors and seven tables.
 Model nv;nv.nvfp4_=true;at::Tensor nv_bundle;nv_bundle.shape={8,1572864};
 for(auto& w:nv.layers_){
  auto set=[&](at::Tensor& t,at::Tensor& scale,int n,int k){
   t.dtype=at::kByte;t.shape={n,k/2};scale.present=true;scale.dtype=at::kByte;scale.shape={k/16,n};};
  set(w.q_w,w.q_ws,2048,1024);set(w.k_w,w.k_ws,1024,1024);set(w.v_w,w.v_ws,1024,1024);
  set(w.o_w,w.o_ws,1024,2048);set(w.gate_w,w.gate_ws,2048,1024);set(w.up_w,w.up_ws,2048,1024);set(w.down_w,w.down_ws,1024,2048);
 }
 install(nv,nv_bundle,false);nv.cold_nvfp4_gateup_available_=true;
 for(int layers:{0,6,11,13,23,25,32}){auto bad=nv_bundle;bad.shape[1]=layers*131072;install(nv,bad,false);}
 install(nv,nv_bundle,false,12);
 nv.nvfp4_=false;install(nv,nv_bundle,false);nv.nvfp4_=true;
 nv.layers_.back().down_ws.dtype=at::kHalf;install(nv,nv_bundle,false);nv.layers_.back().down_ws.dtype=at::kByte;
 for(int scale_count:{4,8})for(bool fused:{false,true}){
  Model scales=nv;auto carrier=nv_bundle;carrier.shape[1]=3145728+scale_count*16384;
  scales.cold_nvfp4_resident_swiglu_available_=fused;
  install(scales,carrier,false);
  auto& correct=fused?scales.cold_nvfp4_resident_scales_swiglu_available_:scales.cold_nvfp4_resident_scales_available_;
  auto& wrong=fused?scales.cold_nvfp4_resident_scales_available_:scales.cold_nvfp4_resident_scales_swiglu_available_;
  wrong=true;install(scales,carrier,false);wrong=false;correct=true;
  for(int delta:{-128,128,32768}){auto bad=carrier;bad.shape[1]+=delta;install(scales,bad,false);}
  install(scales,carrier,false,12);assert(scales.resident_fp8_scale_layers_==0);
  scales.cold_nvfp4_resident_overlap_available_=true;install(scales,carrier,false);
  scales.cold_nvfp4_resident_overlap_available_=false;
  const auto before=scales.kvinsert_cost_weight_identity();
  install(scales,carrier,true);
  assert(scales.resident_fp8_scale_layers_==scale_count && scales.gateup_resident_layers()==24);
  assert(scales.gateup_bundle_bytes()==6291456 && scales.resident_storage_bytes()==6291456+scale_count*32768);
  assert(scales.resident_fp8_scales_enabled(51,240));
  assert(!scales.resident_fp8_scales_enabled(50,240) && !scales.resident_fp8_scales_enabled(51,112));
  assert(scales.gateup_swiglu_arguments(240)[2]==24 && scales.gateup_swiglu_arguments(240)[3]==8);
  const std::vector<int64_t> recipe{1,32,0,scale_count,51,240,1024,2048,256,8,10,
    24,6291456,scale_count*32768,6291456+scale_count*32768,32768,16384,320,64,128,8,20,34,512,24,36,27,int(fused)};
  assert(scales.resident_fp8_scales_arguments(240)==recipe);
  assert(scales.resident_fp8_scales_arguments(224)[5]==224);
  for(int position:{128,144,160,176,192,208,224,240}){
   assert(scales.resident_fp8_scales_enabled(51,position));
   auto expected=recipe;expected[5]=position;
   assert(scales.resident_fp8_scales_arguments(position)==expected);
  }
  assert(before!=scales.kvinsert_cost_weight_identity());
  const auto identity=scales.kvinsert_cost_weight_identity();
  for(bool* flag:{&scales.cold_nvfp4_resident_scales_available_,&scales.cold_nvfp4_resident_scales_swiglu_available_}){
   *flag=!*flag;assert(scales.kvinsert_cost_weight_identity()!=identity);*flag=!*flag;}
  correct=false;assert(!scales.resident_fp8_scales_enabled(51,240));correct=true;
  scales.resident_fp8_scale_layers_=7;assert(!scales.resident_fp8_scales_enabled(51,240));
  assert(scales.kvinsert_cost_weight_identity()!=identity);scales.resident_fp8_scale_layers_=scale_count;
  install(scales,carrier,false);
 }
 // Each optional overlap keeps its original producer and adds one schedule receipt.
 for(int which=0;which<4;++which){
  Model overlap=nv;auto carrier=nv_bundle;carrier.shape[1]=24*131072;
  overlap.cold_nvfp4_o_n128_available_=true;
  overlap.cold_nvfp4_down_nextnorm_available_=true;
  overlap.cold_nvfp4_resident_swiglu_available_=true;
  bool* flags[]{&overlap.cold_nvfp4_o_overlap_available_,&overlap.cold_nvfp4_down_overlap_available_,
    &overlap.cold_nvfp4_gateup_overlap_available_,&overlap.cold_nvfp4_resident_overlap_available_};
  *flags[which]=true;
  if(which==0)overlap.cold_nvfp4_o_n128_available_=false;
  if(which==1)overlap.cold_nvfp4_down_nextnorm_available_=false;
  if(which==2)overlap.cold_nvfp4_gateup_input_reuse_available_=true;
  if(which==3)overlap.cold_nvfp4_resident_swiglu_available_=false;
  install(overlap,carrier,false);
  overlap.cold_nvfp4_o_n128_available_=true;overlap.cold_nvfp4_down_nextnorm_available_=true;
  overlap.cold_nvfp4_gateup_input_reuse_available_=false;overlap.cold_nvfp4_resident_swiglu_available_=true;
  install(overlap,carrier,true);
  const auto arguments=[&](){switch(which){
   case 0:return overlap.o_ring_postnorm_arguments(240);
   case 1:return overlap.down_nvfp4_nextnorm_arguments(240);
   case 2:return overlap.gateup_swiglu_arguments(240);
   default:return overlap.resident_swiglu_arguments(240);}};
  const auto identity=overlap.kvinsert_cost_weight_identity();
  const auto recipe=arguments();*flags[which]=false;
  assert(overlap.kvinsert_cost_weight_identity()!=identity);
  auto expected=arguments();expected[0]=std::array<int,4>{5,4,6,4}[which];
  expected.insert(expected.end(),{106,105,0,1,1});assert(recipe==expected);
 }
 for(int64_t count:{12,24}){
  Model candidate=nv;auto carrier=nv_bundle;carrier.shape[1]=count*131072;
  install(candidate,carrier,true);
  assert(candidate.gateup_resident_layers()==count && candidate.gateup_bundle_bytes()==count*262144);
  assert(candidate.resident_storage_bytes()==count*262144);
  assert(candidate.gateup_resident_arguments()==std::vector<int64_t>({3,count,32,51,1024,2048,8,10,262144,count*262144,320,64,128,8,24,27,32}));
  assert(candidate.gateup_swiglu_arguments(240)==std::vector<int64_t>({4,32,count,32-count,51,240,1024,2048,256,8,10,480,32,128,8,24,36,27,128,130}));
  {
   // Change only position-bearing fields; every P240 recipe remains the
   // existing golden, including optional producer and tail schedules.
   auto prefix=candidate;
   prefix.cold_ring_postnorm_available_=prefix.cold_qk_rope_available_=true;
   prefix.cold_down_nextnorm_available_=prefix.cold_nvfp4_down_nextnorm_available_=true;
   prefix.cold_qk_rope_norm_available_=prefix.cold_qk_rope_norm_kv_available_=true;
   prefix.cold_paired_kv_available_=prefix.cold_kvpad16_=true;
   prefix.cold_nvfp4_qkv_available_=prefix.cold_nvfp4_qkv_overlap_available_=true;
   prefix.cold_nvfp4_resident_swiglu_available_=prefix.cold_nvfp4_o_n128_available_=true;
   prefix.cold_sdpa_m16_available_=true;
   const auto check=[](const std::vector<int64_t>& actual,std::vector<int64_t> expected,
                       int position,std::initializer_list<size_t> fields){
    for(auto field:fields)expected[field]=position;assert(actual==expected);
   };
   const auto baseline_plan=exact_plan();
   for(int position:{128,144,160,176,192,208,224,240}){
    const auto plan=exact_plan(position);
    assert(prefix.gateup_swiglu_enabled(51,position) && prefix.resident_swiglu_enabled(51,position));
    assert(prefix.qk_rope_norm_kv_enabled(51,position,plan) && prefix.paired_kv_enabled(plan));
    assert(prefix.o_ring_postnorm_enabled(51,position) && prefix.down_nvfp4_nextnorm_enabled(51,position));
    assert(prefix.nvfp4_qkv_mode(51,position)==3 && prefix.sdpa_m16_enabled(51,position)==(count==24));
    check(prefix.gateup_swiglu_arguments(position),prefix.gateup_swiglu_arguments(240),position,{5});
    check(prefix.resident_swiglu_arguments(position),prefix.resident_swiglu_arguments(240),position,{5});
    check(prefix.ring_postnorm_arguments(position),prefix.ring_postnorm_arguments(240),position,{4});
    check(prefix.o_ring_postnorm_arguments(position),prefix.o_ring_postnorm_arguments(240),position,{4});
    check(prefix.down_nextnorm_arguments(position),prefix.down_nextnorm_arguments(240),position,{4});
    check(prefix.down_nvfp4_nextnorm_arguments(position),prefix.down_nvfp4_nextnorm_arguments(240),position,{4});
    check(prefix.down_nvfp4_n128_arguments(position),prefix.down_nvfp4_n128_arguments(240),position,{3});
    check(prefix.o_resident_arguments(position),prefix.o_resident_arguments(240),position,{3});
    check(prefix.qk_rope_arguments(position),prefix.qk_rope_arguments(240),position,{3});
    check(prefix.qk_rope_norm_arguments(position),prefix.qk_rope_norm_arguments(240),position,{3});
    check(prefix.nvfp4_q_n32_arguments(position),prefix.nvfp4_q_n32_arguments(240),position,{3});
    for(int mode:{2,3})check(prefix.nvfp4_qkv_arguments(mode,position),prefix.nvfp4_qkv_arguments(mode,240),position,{3});
    check(prefix.qk_rope_norm_kv_arguments(plan,position),prefix.qk_rope_norm_kv_arguments(baseline_plan,240),position,{3,36});
    auto attention=prefix.sdpa_m16_arguments(240);attention[3]=position;attention[4]=position+51;
    attention[20]=Align(position+51,16);assert(prefix.sdpa_m16_arguments(position)==attention);
    for(int wrong:{128,144,160,176,192,208,224,240})if(wrong!=position){
     auto wrong_plan=exact_plan(wrong);assert(!prefix.qk_rope_norm_kv_enabled(51,position,wrong_plan));
     bool rejected=false;try{prefix.qk_rope_norm_kv_arguments(wrong_plan,position);}catch(const std::exception&){rejected=true;}
     assert(rejected);
    }
   }
   for(int invalid:{-1,0,112,127,129,143,145,159,161,175,177,191,193,207,209,223,225,239,241,256}){
    assert(!prefix.optimized_prefix_supported(invalid) && !prefix.gateup_swiglu_enabled(51,invalid));
    assert(!prefix.qk_rope_norm_enabled(51,invalid) && !prefix.o_ring_postnorm_enabled(51,invalid));
    assert(!prefix.down_nvfp4_nextnorm_enabled(51,invalid) && !prefix.nvfp4_qkv_mode(51,invalid));
   }
  }
  // Resident NV4 fusion has its own payload and recipe; the remaining DDR
  // layer interval and resident preload bytes must remain unchanged.
  {
   const auto old_ddr=candidate.gateup_swiglu_arguments(240),old_gu=candidate.gateup_resident_arguments();
   const auto identity=candidate.kvinsert_cost_weight_identity();
   assert(!candidate.resident_swiglu_enabled(51,240));
   candidate.cold_nvfp4_resident_swiglu_available_=true;
   assert(candidate.resident_swiglu_enabled(51,240));
   const auto fused_identity=candidate.kvinsert_cost_weight_identity();
   size_t changed=0;for(size_t i=0;i<identity.size();++i)changed+=identity[i]!=fused_identity[i];assert(changed==1);
   assert(candidate.resident_swiglu_arguments(240)==std::vector<int64_t>({3,32,0,count,51,240,1024,2048,256,8,10,
    320,64,128,8,128,1,1,13056,32512,13056,48,24,36,27,32,2,32,20,34,38}));
   assert(candidate.gateup_swiglu_arguments(240)==old_ddr && candidate.gateup_resident_arguments()==old_gu);
   assert(!candidate.resident_swiglu_enabled(50,240) && !candidate.resident_swiglu_enabled(51,112));
   for(bool* available:{&candidate.cold_nvfp4_gateup_available_,&candidate.gateup_resident_bundle_.present,
                        &candidate.cold_silu_mul_,&candidate.unroll_mode_}){
    *available=false;assert(!candidate.resident_swiglu_enabled(51,240));*available=true;
   }
   for(int bad:{6,13,23,25,32}){candidate.gateup_resident_bundle_.shape[1]=bad*131072;assert(!candidate.resident_swiglu_enabled(51,240));}
   candidate.gateup_resident_bundle_.shape[1]=count*131072;
   candidate.layers_.back().up_ws.dtype=at::kHalf;assert(!candidate.resident_swiglu_enabled(51,240));candidate.layers_.back().up_ws.dtype=at::kByte;
   candidate.cold_nvfp4_resident_swiglu_available_=false;
   assert(!candidate.resident_swiglu_enabled(51,240) && candidate.kvinsert_cost_weight_identity()==identity);
  }
  if(count==24){
   const auto resident_recipe=candidate.gateup_resident_arguments();
   const auto original_ddr_recipe=candidate.gateup_swiglu_arguments(240);
   candidate.cold_nvfp4_gateup_input_reuse_available_=true;
   assert(candidate.gateup_resident_profile() && candidate.gateup_swiglu_enabled(51,240));
   assert(candidate.gateup_resident_arguments()==resident_recipe);
   assert(candidate.gateup_swiglu_arguments(240)==std::vector<int64_t>({5,32,24,8,51,240,1024,2048,256,8,10,480,32,128,8,24,36,27,128,130,28,256,64,296,356,2}));
   // The optional candidate cannot replace the required DDR/resident payloads.
   candidate.cold_nvfp4_gateup_available_=false;
   assert(!candidate.gateup_swiglu_enabled(51,240) && !candidate.gateup_resident_profile());
   candidate.cold_nvfp4_gateup_available_=true;
   candidate.cold_nvfp4_gateup_input_reuse_available_=false;
   assert(candidate.gateup_swiglu_arguments(240)==original_ddr_recipe);
  }
  candidate.cold_ring_postnorm_available_=true;candidate.cold_qk_rope_available_=true;candidate.cold_down_nextnorm_available_=true;
  assert(candidate.ring_postnorm_enabled(51,240) && candidate.qk_rope_enabled(51,240) && candidate.down_nextnorm_enabled(51,240));
  assert(!candidate.sdpa_m16_enabled(51,240));
  {const auto before=candidate.kvinsert_cost_weight_identity();candidate.cold_sdpa_m16_available_=true;
   const auto after=candidate.kvinsert_cost_weight_identity();size_t changed=0;
   for(size_t i=0;i<before.size();++i)changed+=before[i]!=after[i];assert(changed==1);
   assert(candidate.sdpa_m16_enabled(51,240)==(candidate.gateup_resident_layers()==24));
   assert(!candidate.sdpa_m16_enabled(50,240) && !candidate.sdpa_m16_enabled(51,112));
   assert(candidate.sdpa_m16_arguments(240)==std::vector<int64_t>({1,32,51,240,291,16,8,128,8,2,4,10,24,16,128,256,4,2,8,262144,304,1,8,16,1,128}));
   candidate.cold_sdpa_m16_available_=false;assert(candidate.kvinsert_cost_weight_identity()==before);
  }
  assert(!candidate.sdpa_k128_enabled(51,240));
  {const auto before=candidate.kvinsert_cost_weight_identity();candidate.cold_sdpa_k128_available_=true;
   const auto after=candidate.kvinsert_cost_weight_identity();size_t changed=0;
   for(size_t i=0;i<before.size();++i)changed+=before[i]!=after[i];assert(changed==1);
   assert(!candidate.sdpa_k128_enabled(50,240) && !candidate.sdpa_k128_enabled(51,112));
   const std::vector<int64_t> recipe{1,32,51,240,291,16,8,128,8,2,4,10,24,64,128,128,1,2,8,32768,304,4,8,8,1,128};
   assert(candidate.sdpa_k128_arguments(240)==recipe);
   for(int position:{128,144,160,176,192,208,224,240}){
    assert(candidate.sdpa_k128_enabled(51,position)==(count==24));
    auto expected=recipe;expected[3]=position;expected[4]=position+51;expected[20]=Align(position+51,16);
    assert(candidate.sdpa_k128_arguments(position)==expected);}
   candidate.cold_sdpa_m16_available_=true;bool rejected=false;
   try{candidate.sdpa_k128_enabled(51,240);}catch(const std::exception&){rejected=true;}assert(rejected);
   candidate.cold_sdpa_m16_available_=false;candidate.cold_sdpa_k128_available_=false;
   assert(candidate.kvinsert_cost_weight_identity()==before);
  }
  assert(!candidate.nvfp4_qkv_mode(51,240));
  auto qkv_identity=candidate.kvinsert_cost_weight_identity();
  candidate.cold_nvfp4_kv_available_=true;
  assert(candidate.nvfp4_qkv_mode(51,240)==2);
  assert(candidate.kvinsert_cost_weight_identity()!=qkv_identity);
  assert(candidate.nvfp4_qkv_mode(50,240)==0 && candidate.nvfp4_qkv_mode(51,112)==0);
  candidate.cold_nvfp4_qkv_available_=true;assert(candidate.nvfp4_qkv_mode(51,240)==3);
  const auto canonical_qkv=candidate.nvfp4_qkv_arguments(3,240);
  for(bool* flag:{&candidate.cold_nvfp4_q_n32_available_,&candidate.cold_nvfp4_qkv_overlap_available_}){
   const auto before=candidate.kvinsert_cost_weight_identity();*flag=true;
   const auto after=candidate.kvinsert_cost_weight_identity();size_t changed=0;
   for(size_t i=0;i<before.size();++i)changed+=before[i]!=after[i];assert(changed==1);
   if(flag==&candidate.cold_nvfp4_q_n32_available_){
    assert(candidate.nvfp4_q_n32_enabled(51,240) && candidate.nvfp4_qkv_mode(51,240)==2);
    assert(candidate.nvfp4_q_n32_arguments(240)==std::vector<int64_t>({1,32,51,240,1024,2048,256,8,10,480,32,128,8,1,2,3,20,21,24,27,32,16,32}));
    assert(candidate.nvfp4_qkv_arguments(3,240)==canonical_qkv);
   }else{
    assert(candidate.nvfp4_qkv_mode(51,240)==3 && !candidate.nvfp4_q_n32_enabled(51,240));
    auto expected=canonical_qkv;expected[0]=2;expected.insert(expected.end(),{106,105,0,1,1});
    assert(candidate.nvfp4_qkv_arguments(3,240)==expected);
    assert(candidate.nvfp4_qkv_arguments(2,240)[0]==1);
   }
   assert(candidate.nvfp4_qkv_mode(50,240)==0 && candidate.nvfp4_qkv_mode(51,112)==0);
   *flag=false;assert(candidate.kvinsert_cost_weight_identity()==before);
  }
  candidate.cold_nvfp4_q_n32_available_=true;candidate.cold_nvfp4_qkv_overlap_available_=true;
  bool conflict=false;try{candidate.nvfp4_qkv_mode(51,240);}catch(const std::exception&){conflict=true;}assert(conflict);
  candidate.cold_nvfp4_qkv_overlap_available_=false;candidate.cold_nvfp4_kv_available_=false;
  conflict=false;try{candidate.nvfp4_qkv_mode(51,240);}catch(const std::exception&){conflict=true;}assert(conflict);
  candidate.cold_nvfp4_q_n32_available_=false;candidate.cold_nvfp4_qkv_overlap_available_=true;
  candidate.cold_nvfp4_qkv_available_=false;
  conflict=false;try{candidate.nvfp4_qkv_mode(51,240);}catch(const std::exception&){conflict=true;}assert(conflict);
  candidate.cold_nvfp4_qkv_overlap_available_=false;candidate.cold_nvfp4_qkv_available_=true;
  candidate.cold_nvfp4_kv_available_=false;assert(candidate.nvfp4_qkv_mode(51,240)==3);
  candidate.cold_nvfp4_qkv_available_=false;assert(!candidate.nvfp4_qkv_mode(51,240));
  assert(!candidate.qk_rope_norm_enabled(51,240));
  const auto before_qknorm=candidate.kvinsert_cost_weight_identity();
  candidate.cold_qk_rope_norm_available_=true;
  assert(candidate.qk_rope_norm_enabled(51,240));
  auto qknorm_recipe=candidate.qk_rope_arguments(240);
  qknorm_recipe.insert(qknorm_recipe.end(),{112,64,128,14,8,8,32,44,128,130,2,5,1,100000,1});
  assert(qknorm_recipe.size()==33 && candidate.qk_rope_norm_arguments(240)==qknorm_recipe);
  const auto after_qknorm=candidate.kvinsert_cost_weight_identity();
  assert(before_qknorm.size()==after_qknorm.size());
  size_t qknorm_changes=0;for(size_t i=0;i<before_qknorm.size();++i)qknorm_changes+=before_qknorm[i]!=after_qknorm[i];
  assert(qknorm_changes==1);
  const auto valid_kv_plan=exact_plan();
  assert(!candidate.qk_rope_norm_kv_enabled(51,240,valid_kv_plan));
  candidate.cold_qk_rope_norm_kv_available_=true;
  assert(!candidate.qk_rope_norm_kv_enabled(51,240,valid_kv_plan));
  candidate.cold_paired_kv_available_=true;candidate.cold_kvpad16_=true;
  assert(candidate.qk_rope_norm_kv_enabled(51,240,valid_kv_plan));
  auto qknorm_kv_recipe=qknorm_recipe;
  qknorm_kv_recipe.insert(qknorm_kv_recipe.end(),{1,51,64,240,8,8,128,2,4,132,3,134,13,
   0,8,1,9,2,14,3,10,6,16,7,17,8,20,9,21,10,22,11,23,12,24,13,25,21,26});
  assert(qknorm_kv_recipe.size()==72 && candidate.qk_rope_norm_kv_arguments(valid_kv_plan,240)==qknorm_kv_recipe);
  for(auto bad:{exact_plan(224),exact_plan(240,51,KvInsertRoute::V2,51),exact_plan(240,64,KvInsertRoute::ALIGNED_V16,64)})
   assert(!candidate.qk_rope_norm_kv_enabled(51,240,bad));
  for(bool* enabled:{&candidate.cold_qk_rope_norm_available_,&candidate.cold_paired_kv_available_,
                    &candidate.cold_kvpad16_,&candidate.cold_rmsnorm_spm_contract_.q_vector,
                    &candidate.cold_rmsnorm_spm_contract_.k_vector}){
   *enabled=false;assert(!candidate.qk_rope_norm_kv_enabled(51,240,valid_kv_plan));*enabled=true;
  }
  const auto identity_with_paired=candidate.kvinsert_cost_weight_identity();
  candidate.cold_qk_rope_norm_kv_available_=false;
  const auto identity_without_paired=candidate.kvinsert_cost_weight_identity();
  qknorm_changes=0;for(size_t i=0;i<identity_with_paired.size();++i)qknorm_changes+=identity_with_paired[i]!=identity_without_paired[i];
  assert(qknorm_changes==1);
  candidate.cold_paired_kv_available_=false;candidate.cold_kvpad16_=false;
  assert(!candidate.qk_rope_norm_enabled(50,240) && !candidate.qk_rope_norm_enabled(51,112));
  for(bool* enabled:{&candidate.cold_rmsnorm_spm_contract_.vector,&candidate.cold_rmsnorm_spm_contract_.q_vector,
                    &candidate.cold_rmsnorm_spm_contract_.k_vector,&candidate.cold_qk_rope_available_,&candidate.cold_partial_rope_}){
   *enabled=false;assert(!candidate.qk_rope_norm_enabled(51,240));*enabled=true;
  }
  candidate.cold_qk_rope_norm_available_=false;
  assert(!candidate.qk_rope_norm_enabled(51,240) && candidate.kvinsert_cost_weight_identity()==before_qknorm);
  if(count==24){
   assert(!candidate.down_nvfp4_n128_enabled(51,240));
   const auto r16=candidate.down_nextnorm_arguments(240);
   candidate.cold_nvfp4_down_n128_available_=true;
   assert(candidate.down_nvfp4_n128_enabled(51,240));
   assert(candidate.down_nvfp4_n128_arguments(240)==std::vector<int64_t>({1,32,51,240,1024,2048,256,8,10,176,128,128,1,8,0,24,27,32,2,3,20,21}));
   assert(candidate.down_nextnorm_enabled(51,240) && candidate.down_nextnorm_arguments(240)==r16);
   assert(!candidate.down_nvfp4_n128_enabled(50,240) && !candidate.down_nvfp4_n128_enabled(51,112));
   candidate.l=31;assert(!candidate.down_nvfp4_n128_enabled(51,240));candidate.l=32;
   candidate.nvfp4_=false;assert(!candidate.down_nvfp4_n128_enabled(51,240));candidate.nvfp4_=true;
   candidate.layers_.back().down_ws.dtype=at::kHalf;
   assert(!candidate.down_nvfp4_n128_enabled(51,240));candidate.layers_.back().down_ws.dtype=at::kByte;
   candidate.cold_nvfp4_down_n128_available_=false;assert(!candidate.down_nvfp4_n128_enabled(51,240));
   const auto old_identity=candidate.kvinsert_cost_weight_identity();
   assert(!candidate.down_nvfp4_nextnorm_enabled(51,240));
   candidate.cold_nvfp4_down_nextnorm_available_=true;
   const auto new_identity=candidate.kvinsert_cost_weight_identity();
   assert(new_identity.size()==old_identity.size() && new_identity!=old_identity);
   size_t changed=0;for(size_t i=0;i<old_identity.size();++i)changed+=old_identity[i]!=new_identity[i];
   assert(changed==1);
   auto fused=r16;fused[0]=2;
   fused.insert(fused.end(),{320,64,128,2,8,256,2048,2,3,20,21,24,27,32,32,50,128});
   assert(candidate.down_nvfp4_nextnorm_enabled(51,240) && candidate.down_nvfp4_nextnorm_arguments(240)==fused);
   candidate.cold_nvfp4_down_n128_available_=true;
   assert(candidate.down_nvfp4_nextnorm_enabled(51,240) && !candidate.down_nvfp4_n128_enabled(51,240));
   const auto n64_identity=candidate.kvinsert_cost_weight_identity();
   candidate.cold_nvfp4_down_nextnorm_n128_available_=true;
   const auto n128_identity=candidate.kvinsert_cost_weight_identity();
   assert(n128_identity.size()==n64_identity.size());changed=0;
   for(size_t i=0;i<n64_identity.size();++i)changed+=n64_identity[i]!=n128_identity[i];
   assert(changed==1);
   auto fused_n128=fused;fused_n128[0]=3;fused_n128[23]=176;fused_n128[24]=128;fused_n128[26]=1;
   assert(candidate.down_nvfp4_nextnorm_arguments(240)==fused_n128);
   assert(candidate.down_nvfp4_nextnorm_enabled(51,240) && !candidate.down_nvfp4_n128_enabled(51,240));
   candidate.cold_nvfp4_down_nextnorm_available_=false;
   assert(!candidate.down_nvfp4_nextnorm_enabled(51,240) && candidate.down_nvfp4_n128_enabled(51,240));
   candidate.cold_nvfp4_down_nextnorm_available_=true;
   candidate.cold_nvfp4_down_nextnorm_n128_available_=false;
   assert(candidate.down_nvfp4_nextnorm_arguments(240)==fused && candidate.kvinsert_cost_weight_identity()==n64_identity);
   assert(!candidate.down_nvfp4_nextnorm_enabled(50,240) && !candidate.down_nvfp4_nextnorm_enabled(51,112));
   candidate.cold_down_nextnorm_available_=false;
   assert(!candidate.down_nvfp4_nextnorm_enabled(51,240) && candidate.down_nvfp4_n128_enabled(51,240));
   candidate.cold_down_nextnorm_available_=true;
   candidate.cold_nvfp4_down_nextnorm_available_=false;
   assert(candidate.down_nvfp4_n128_enabled(51,240));candidate.cold_nvfp4_down_n128_available_=false;
   assert(candidate.kvinsert_cost_weight_identity()==old_identity);
  }
  assert(!candidate.o_ring_postnorm_enabled(51,240));candidate.cold_nvfp4_o_ring_postnorm_available_=true;assert(candidate.o_ring_postnorm_enabled(51,240));
  auto nv_o=candidate.ring_postnorm_arguments(240);nv_o[0]=3;nv_o.insert(nv_o.end(),{320,64,128,2,16,256,2048,2,3,20,21,24,27,32});
  assert(candidate.o_ring_postnorm_arguments(240)==nv_o);
  if(count==24){
   candidate.cold_nvfp4_o_n128_available_=true;
   auto n128=candidate.ring_postnorm_arguments(240);n128[0]=4;
   n128.insert(n128.end(),{176,128,128,1,8,256,2048,2,3,20,21,24,27,32});
   assert(candidate.o_ring_postnorm_enabled(51,240) && candidate.o_ring_postnorm_arguments(240)==n128);
   candidate.cold_nvfp4_o_ring_postnorm_available_=false;
   assert(candidate.o_ring_postnorm_enabled(51,240) && candidate.o_ring_postnorm_arguments(240)==n128);
   assert(!candidate.o_ring_postnorm_enabled(50,240) && !candidate.o_ring_postnorm_enabled(51,112));
   candidate.cold_nvfp4_o_n128_available_=false;
   assert(!candidate.o_ring_postnorm_enabled(51,240));
   candidate.cold_nvfp4_o_ring_postnorm_available_=true;
   assert(candidate.o_ring_postnorm_arguments(240)==nv_o);
  }
  candidate.cold_w8_resident_swiglu_available_=true;assert(!candidate.resident_swiglu_enabled(51,240));
  candidate.cold_xor3_norm_available_=true;assert(!candidate.xor3_norm_enabled());
  install(candidate,carrier,false);
 }
 Model quant12=quant;install(quant12,bundle,true);
 assert(quant12.gateup_resident_layers()==12 && quant12.gateup_bundle_bytes()==6291456);
 assert(quant12.gateup_resident_arguments()==std::vector<int64_t>({5,12,32,51,1024,2048,8,10,524288,6291456,288,128,128,4,2,32,20,34,4,36}));
 assert(quant12.gateup_swiglu_arguments(240)==std::vector<int64_t>({5,32,12,20,51,240,1024,2048,256,8,10,192,128,128,2,2,32,20,34,26,27,28,29}));
 install(quant12,qb,false);
 Model fp16;auto oversized=bundle;oversized.shape[1]*=2;install(fp16,oversized,false);install(quant,oversized,false);
 quant.layers_.back().down_ws.present=false;install(quant,qb,false);quant.layers_.back().down_ws.present=true;
 install(quant,qb,true);assert(quant.gateup_swiglu_enabled(51,240));
 quant.cold_resident_swiglu_available_=true;quant.cold_o_ring_postnorm_available_=true;
 quant.cold_ring_postnorm_available_=true;quant.cold_qk_rope_available_=true;quant.cold_down_nextnorm_available_=true;
 base.cold_nvfp4_kv_available_=true;quant.cold_nvfp4_kv_available_=true;
 base.cold_nvfp4_qkv_available_=true;quant.cold_nvfp4_qkv_available_=true;
 base.cold_nvfp4_q_n32_available_=true;quant.cold_nvfp4_q_n32_available_=true;
 base.cold_nvfp4_qkv_overlap_available_=true;quant.cold_nvfp4_qkv_overlap_available_=true;
 assert(!base.nvfp4_qkv_mode(51,240) && !quant.nvfp4_qkv_mode(51,240));
 assert(!base.nvfp4_q_n32_enabled(51,240) && !quant.nvfp4_q_n32_enabled(51,240));
 base.cold_sdpa_m16_available_=true;quant.cold_sdpa_m16_available_=true;
 assert(!base.sdpa_m16_enabled(51,240) && !quant.sdpa_m16_enabled(51,240));
 base.cold_qk_rope_norm_available_=true;quant.cold_qk_rope_norm_available_=true;
 assert(base.qk_rope_norm_enabled(51,240) && quant.qk_rope_norm_enabled(51,240));
 base.cold_qk_rope_norm_kv_available_=true;quant.cold_qk_rope_norm_kv_available_=true;
 assert(!base.qk_rope_norm_kv_enabled(51,240,exact_plan()) && !quant.qk_rope_norm_kv_enabled(51,240,exact_plan()));
 quant.cold_nvfp4_o_n128_available_=true;
 assert(!quant.resident_swiglu_enabled(51,240) && !quant.o_ring_postnorm_enabled(51,240));
 assert(quant.down_nextnorm_enabled(51,240));
 quant.cold_w8_o_ring_postnorm_available_=true;assert(quant.o_ring_postnorm_enabled(51,240));
 assert(quant.o_ring_postnorm_arguments(240)[0]==7 && quant.o_ring_postnorm_arguments(240).back()==50);
 quant.cold_xor3_norm_available_=true;
 assert(quant.xor3_norm_enabled() && !quant.o_ring_postnorm_enabled(51,240));
 assert(quant.ring_postnorm_arguments(240)[0]==2 && quant.ring_postnorm_arguments(240).back()==3);
 base.cold_xor3_norm_available_=true;assert(!base.xor3_norm_enabled());
 quant.cold_xor3_norm_available_=false;
 quant.cold_w8_resident_swiglu_available_=true;assert(quant.resident_swiglu_enabled(51,240));
 assert(quant.resident_swiglu_arguments(240)[0]==5);
 const auto fp16_resident=base.resident_swiglu_arguments(240),w8_resident=quant.resident_swiglu_arguments(240);
 const bool fp16_resident_enabled=base.resident_swiglu_enabled(51,240),w8_resident_enabled=quant.resident_swiglu_enabled(51,240);
 base.cold_nvfp4_resident_swiglu_available_=true;quant.cold_nvfp4_resident_swiglu_available_=true;
 assert(base.resident_swiglu_enabled(51,240)==fp16_resident_enabled && quant.resident_swiglu_enabled(51,240)==w8_resident_enabled);
 assert(base.resident_swiglu_arguments(240)==fp16_resident && quant.resident_swiglu_arguments(240)==w8_resident);
 quant12.cold_w8_resident_swiglu_available_=true;assert(quant12.resident_swiglu_arguments(240)[3]==12);
 assert(quant.gateup_swiglu_arguments(240)[0]==5);
 quant.cold_w8_gateup_n48_available_=true;
 assert(quant.gateup_swiglu_arguments(240)==std::vector<int64_t>({5,32,6,26,51,240,1024,2048,256,8,10,192,128,128,2,2,32,20,34,26,27,28,29}));
 base.cold_w8_gateup_n48_available_=true;
 assert(base.gateup_swiglu_arguments(240)[0]==4);
 auto fp16_gu=base.gateup_swiglu_arguments(240),w8_gu=quant.gateup_swiglu_arguments(240);
 base.cold_nvfp4_gateup_input_reuse_available_=true;quant.cold_nvfp4_gateup_input_reuse_available_=true;
 assert(base.gateup_swiglu_arguments(240)==fp16_gu && quant.gateup_swiglu_arguments(240)==w8_gu);
 auto fp16_o=base.o_ring_postnorm_arguments(240);base.cold_nvfp4_o_n128_available_=true;
 assert(base.o_ring_postnorm_arguments(240)==fp16_o);
 base.cold_o_ring_postnorm_available_=false;assert(!base.o_ring_postnorm_enabled(51,240));
 base.cold_nvfp4_down_n128_available_=true;quant.cold_nvfp4_down_n128_available_=true;
 assert(!base.down_nvfp4_n128_enabled(51,240) && !quant.down_nvfp4_n128_enabled(51,240));
 auto fp16_r16=base.down_nextnorm_arguments(240),w8_r16=quant.down_nextnorm_arguments(240);
 base.cold_nvfp4_down_nextnorm_available_=true;quant.cold_nvfp4_down_nextnorm_available_=true;
 base.cold_nvfp4_down_nextnorm_n128_available_=true;quant.cold_nvfp4_down_nextnorm_n128_available_=true;
 assert(!base.down_nvfp4_nextnorm_enabled(51,240) && !quant.down_nvfp4_nextnorm_enabled(51,240));
 assert(base.down_nextnorm_arguments(240)==fp16_r16 && quant.down_nextnorm_arguments(240)==w8_r16);
 base.cold_paired_kv_available_=quant.cold_paired_kv_available_=true;
 base.cold_kvpad16_=quant.cold_kvpad16_=true;
 for(int position:{128,144,160,176,192,208,224,240}){
  assert(base.optimized_prefix_supported(position) && quant.optimized_prefix_supported(position));
  assert(base.gateup_swiglu_enabled(51,position) && quant.gateup_swiglu_enabled(51,position));
  assert(base.paired_kv_enabled(exact_plan(position)) && quant.paired_kv_enabled(exact_plan(position)));
  for(auto* model:{&base,&quant}){
   const auto plan=exact_plan(position);
   assert(model->qk_rope_norm_kv_enabled(51,position,plan));
   auto fused=model->qk_rope_norm_kv_arguments(exact_plan(),240);
   fused[3]=fused[36]=position;assert(model->qk_rope_norm_kv_arguments(plan,position)==fused);
   auto fallback=*model;fallback.cold_qk_rope_norm_kv_available_=false;
   assert(!fallback.qk_rope_norm_kv_enabled(51,position,plan) && fallback.qk_rope_norm_enabled(51,position));
   fallback.cold_qk_rope_norm_available_=false;
   assert(!fallback.qk_rope_norm_enabled(51,position) && fallback.qk_rope_enabled(51,position));
   for(bool* cap:{&model->cold_rmsnorm_spm_contract_.q_vector,&model->cold_rmsnorm_spm_contract_.k_vector}){
    *cap=false;assert(!model->qk_rope_norm_kv_enabled(51,position,plan) && model->qk_rope_enabled(51,position));*cap=true;
   }
   fallback=*model;fallback.gateup_resident_bundle_.present=false;
   assert(!fallback.qk_rope_norm_enabled(51,position) && !fallback.qk_rope_norm_kv_enabled(51,position,plan));
   fallback=*model;fallback.layers_.back().q_w.dtype=at::kByte;
   assert(!fallback.qk_rope_norm_kv_enabled(51,position,plan));
   assert(!model->qk_rope_norm_kv_enabled(51,position,exact_plan(position==240?224:240)));
   auto gu=model->gateup_swiglu_arguments(240);gu[5]=position;assert(model->gateup_swiglu_arguments(position)==gu);
   auto rope=model->qk_rope_arguments(240);rope[3]=position;assert(model->qk_rope_arguments(position)==rope);
   auto ring=model->ring_postnorm_arguments(240);ring[4]=position;assert(model->ring_postnorm_arguments(position)==ring);
   assert(model->paired_kv_arguments(exact_plan(position))==std::vector<int64_t>({1,51,64,position,8,8,128}));
  }
 }
 for(int invalid:{-1,0,112,127,129,143,145,159,161,175,177,191,193,207,209,223,225,239,241,256}){
  assert(!base.optimized_prefix_supported(invalid) && !quant.optimized_prefix_supported(invalid));
  assert(!base.gateup_swiglu_enabled(51,invalid) && !quant.gateup_swiglu_enabled(51,invalid));
  assert(!base.qk_rope_norm_enabled(51,invalid) && !quant.qk_rope_norm_enabled(51,invalid));
 }
}
'''
    compile_run(tmp_path, "resident_owner", source.replace("BYTE_METHODS", byte_methods).replace("IDENTITY_METHOD", identity_method).replace("METHODS", methods))


def test_actual_manifest_rejects_partial_stale_or_forged_residency(tmp_path):
    method = block((ROOT / "src/fused/rpu_hyvla_expert_model.cpp").read_text(),
                   "FmbPhysicalManifestForwardCapability\nHyVlaExpertModel::physical_manifest_forward_capability(")
    decoder = (ROOT / "src/fused/rpu_hyvla_expert_model.cpp").read_text()
    method += "\n" + block(decoder, "bool HyVlaExpertModel::optimized_prefix_supported(")
    method += "\n" + block(decoder, "bool HyVlaExpertModel::gateup_swiglu_enabled(")
    method += "\n" + block(decoder, "std::vector<int64_t> HyVlaExpertModel::gateup_swiglu_arguments(")
    method += "\n" + block(decoder, "bool HyVlaExpertModel::down_nextnorm_enabled(")
    method += "\n" + block(decoder, "std::vector<int64_t> HyVlaExpertModel::down_nextnorm_arguments(")
    method += "\n" + block(decoder, "bool HyVlaExpertModel::down_nvfp4_n128_enabled(")
    method += "\n" + block(decoder, "std::vector<int64_t> HyVlaExpertModel::down_nvfp4_n128_arguments(")
    method += "\n" + block(decoder, "bool HyVlaExpertModel::down_nvfp4_nextnorm_enabled(")
    method += "\n" + block(decoder, "std::vector<int64_t> HyVlaExpertModel::down_nvfp4_nextnorm_arguments(")
    method += "\n" + block(decoder, "bool HyVlaExpertModel::qk_rope_enabled(")
    method += "\n" + block(decoder, "std::vector<int64_t> HyVlaExpertModel::qk_rope_arguments(")
    method += "\n" + block(decoder, "bool HyVlaExpertModel::sdpa_k128_enabled(")
    method += "\n" + block(decoder, "std::vector<int64_t> HyVlaExpertModel::sdpa_k128_arguments(")
    method += "\n" + block(decoder, "bool HyVlaExpertModel::sdpa_m16_enabled(")
    method += "\n" + block(decoder, "std::vector<int64_t> HyVlaExpertModel::sdpa_m16_arguments(")
    method += "\n" + block(decoder, "bool HyVlaExpertModel::kv_grouped_enabled(")
    method += "\n" + block(decoder, "std::vector<int64_t> HyVlaExpertModel::kv_grouped_arguments(")
    method += "\n" + block(decoder, "int HyVlaExpertModel::nvfp4_qkv_mode(")
    method += "\n" + block(decoder, "bool HyVlaExpertModel::nvfp4_q_n32_enabled(")
    method += "\n" + block(decoder, "std::vector<int64_t> HyVlaExpertModel::nvfp4_q_n32_arguments(")
    method += "\n" + block(decoder, "std::vector<int64_t> HyVlaExpertModel::nvfp4_qkv_arguments(")
    method += "\n" + block(decoder, "bool HyVlaExpertModel::qk_rope_norm_enabled(")
    method += "\n" + block(decoder, "std::vector<int64_t> HyVlaExpertModel::qk_rope_norm_arguments(")
    method += "\n" + block(decoder, "bool HyVlaExpertModel::qk_rope_norm_kv_enabled(")
    method += "\n" + block(decoder, "std::vector<int64_t> HyVlaExpertModel::qk_rope_norm_kv_arguments(")
    method += "\n" + block(decoder, "bool HyVlaExpertModel::o_resident_enabled(")
    method += "\n" + block(decoder, "std::vector<int64_t> HyVlaExpertModel::o_resident_arguments(")
    method += "\n" + block(decoder, "bool HyVlaExpertModel::o_ring_postnorm_enabled(")
    method += "\n" + block(decoder, "std::vector<int64_t> HyVlaExpertModel::o_ring_postnorm_arguments(")
    method += "\n" + block(decoder, "bool HyVlaExpertModel::ring_postnorm_enabled(")
    method += "\n" + block(decoder, "std::vector<int64_t> HyVlaExpertModel::ring_postnorm_arguments(")
    method += "\n" + block(decoder, "bool HyVlaExpertModel::xor3_norm_enabled(")
    causal = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    method += "\n" + block(causal, "std::vector<int64_t> bind_cold_topology_arguments(")
    method += "\n" + block(decoder, "bool HyVlaExpertModel::resident_fp8_scales_enabled(")
    method += "\n" + block(decoder, "std::vector<int64_t> HyVlaExpertModel::resident_fp8_scales_arguments(")
    method += "\n" + block(decoder, "bool HyVlaExpertModel::resident_swiglu_enabled(")
    method += "\n" + block(decoder, "std::vector<int64_t> HyVlaExpertModel::resident_swiglu_arguments(")
    source = COMMON + r'''
int64_t Align(int64_t n,int64_t a){return ((n+a-1)/a)*a;}
enum class RpuRmsNormSpmRoute {BASE,V16_GRID8,HYVLA_V16_BITCOPY_GRID8};
struct RpuRmsNormSpmContract {bool v16_bitcopy=true;bool vector=true,q_vector=true,k_vector=true;
 RpuRmsNormSpmRoute resolve(int r,int c)const{assert((r==64 && c==1024)||((r==112||r==64)&&c==128));
  return vector && (c!=128 || (r==112?q_vector:k_vector))?RpuRmsNormSpmRoute::HYVLA_V16_BITCOPY_GRID8:RpuRmsNormSpmRoute::BASE;}};
enum class FmbRouteFamily {GRAPH_SCHEDULE,MUTABLE_DMA,KV_INSERT,LINEAR,ACTIVATION,ATTENTION};
enum class AttentionExecutionPolicy {DDR_KV=1,SPM_KV_BY_MHA=2};
enum class FmbLinearRouteSelector {AUTO_TILE=1};
enum class FmbGraphLifecycle {COMPOSITE_CHILD};
enum class HyVlaExpertDmaRoute {DDR_SCATTER_TO_SPM_FIXED=7};
enum class CausalDecoderMutableDmaRoute {DDR_BROADCAST_TO_SPM=1};
enum class HyVlaExpertKvPaddingRoute {PAD16_ROWS=2};
constexpr int64_t HYVLA_EXPERT_KV_PADDING_SITE=3353793944280539666LL;
constexpr int64_t HYVLA_EXPERT_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED=2;
constexpr int64_t CAUSAL_DECODER_NVFP4_SCALE_PRELOAD_SITE=5605101652280608153LL;
constexpr int64_t HYVLA_EXPERT_KV_SITE=1,HYVLA_EXPERT_PAIRED_KV_SITE=2;
constexpr int64_t HYVLA_EXPERT_GATEUP_SWIGLU_SITE=623508127183406970LL;
constexpr int64_t HYVLA_EXPERT_RESIDENT_SWIGLU_SITE=623508127183406990LL;
constexpr int64_t HYVLA_EXPERT_NVFP4_RESIDENT_SWIGLU_SITE=623508127183407110LL;
constexpr int64_t HYVLA_EXPERT_NVFP4_RESIDENT_SCALES_SITE=623508127183407140LL;
constexpr int64_t HYVLA_EXPERT_RING_POSTNORM_SITE=623508127183407010LL;
constexpr int64_t HYVLA_EXPERT_DOWN_NEXTNORM_SITE=623508127183407040LL;
constexpr int64_t HYVLA_EXPERT_DOWN_NVFP4_N128_SITE=623508127183407050LL;
constexpr int64_t HYVLA_EXPERT_DOWN_NVFP4_NEXTNORM_SITE=623508127183407070LL;
constexpr int64_t HYVLA_EXPERT_DOWN_LINEAR_SITE=2699481217144264799LL;
constexpr int64_t HYVLA_EXPERT_MLP_ALL_REDUCE_SITE=41,HYVLA_EXPERT_FINAL_NORM_SITE=42;
constexpr int64_t HYVLA_EXPERT_QK_ROPE_SITE=623508127183407030LL;
constexpr int64_t HYVLA_EXPERT_QK_ROPE_NORM_SITE=623508127183407080LL;
constexpr int64_t HYVLA_EXPERT_QK_ROPE_NORM_KV_SITE=623508127183407100LL;
constexpr int64_t HYVLA_EXPERT_SDPA_K128_SITE=623508127183407150LL;
constexpr int64_t HYVLA_EXPERT_SDPA_M16_SITE=623508127183407130LL, HYVLA_EXPERT_ATTN_SITE=623508127183406900LL;
constexpr int64_t HYVLA_EXPERT_KV_GROUPED_SITE=623508127183407180LL;
constexpr int64_t HYVLA_EXPERT_NVFP4_Q_N32_SITE=623508127183407120LL;
constexpr int64_t HYVLA_EXPERT_NVFP4_QKV_SITE=623508127183407090LL, HYVLA_EXPERT_QKV_LINEAR_SITE=1236943476379916654LL;
constexpr int64_t HYVLA_EXPERT_Q_NORM_SITE=35,HYVLA_EXPERT_K_NORM_SITE=36;
constexpr int64_t HYVLA_EXPERT_Q_PARTIAL_ROPE_SITE=31,HYVLA_EXPERT_K_PARTIAL_ROPE_SITE=32,HYVLA_EXPERT_Q_ROPE_SITE=33,HYVLA_EXPERT_K_ROPE_SITE=34;
constexpr int64_t HYVLA_EXPERT_O_RING_POSTNORM_SITE=623508127183407020LL;
constexpr int64_t HYVLA_EXPERT_O_RESIDENT_SITE=623508127183407060LL;
constexpr int64_t HYVLA_EXPERT_O_LINEAR_SITE=23;
constexpr int64_t HYVLA_EXPERT_ATTN_ALL_REDUCE_SITE=21,HYVLA_EXPERT_POST_NORM_SITE=22;
constexpr int64_t HYVLA_EXPERT_MLP_SILU_MUL_SITE=11,HYVLA_EXPERT_MLP_SILU_SITE=12,HYVLA_EXPERT_MLP_MUL_SITE=13;
constexpr int64_t HYVLA_EXPERT_GATE_LINEAR_SITE=4,HYVLA_EXPERT_UP_LINEAR_SITE=5;
constexpr int64_t HYVLA_EXPERT_GATEUP_RESIDENT_SITE=623508127183406950LL;
constexpr int64_t HYVLA_EXPERT_GATEUP_PRELOAD_SITE=623508127183406951LL;
struct FmbRouteManifestEntry {FmbRouteFamily family;int64_t site_id,selector,flags,invocation;
 std::vector<int64_t> arguments;};
struct FmbPhysicalExecutionManifest {int logical_length=51,physical_length=51,kv_logical_length=291;
 std::vector<FmbRouteManifestEntry> routes;};
struct FmbPhysicalManifestForwardCapability {bool supported=false;FmbGraphLifecycle lifecycle;};
auto restore_kvinsert_plan_from_manifest(const FmbPhysicalExecutionManifest& m,int64_t site,int64_t invocation,int cores,int heads,int dim){
 for(const auto& r:m.routes)if(r.site_id==site && r.invocation==invocation){
  if(r.arguments.empty())return exact_plan(); // Unrelated historical fixture receipts.
  return rpu_kvinsert_segment_plan_from_route_arguments(r.arguments,cores,heads,dim);
 }throw std::runtime_error("missing KV route");
}
struct Model {
 static constexpr int64_t kGateUpResidentLayers=6,kGateUpBundleBytes=6291456;
 at::Tensor gateup_resident_bundle_;bool profile=true,cold_gateup_swiglu_available_=false;
 bool cold_w8_resident_swiglu_available_=false,cold_w8_o_ring_postnorm_available_=false,cold_xor3_norm_available_=false;
 bool cold_nvfp4_resident_swiglu_available_=false;
 bool cold_nvfp4_resident_scales_available_=false,cold_nvfp4_resident_scales_swiglu_available_=false;
 int64_t resident_fp8_scale_layers_=0;
 bool cold_w8_o_resident_available_=false;int64_t o_resident_layers_=0;
 bool nvfp4_=false,cold_nvfp4_gateup_available_=false,cold_nvfp4_o_ring_postnorm_available_=false,cold_nvfp4_o_n128_available_=false;
 bool cold_nvfp4_gateup_input_reuse_available_=false;
 bool cold_nvfp4_down_n128_available_=false;
 bool cold_nvfp4_down_nextnorm_available_=false;
 bool cold_nvfp4_down_nextnorm_n128_available_=false;
 bool cold_resident_swiglu_available_=false;
 bool cold_ring_postnorm_available_=false,cold_o_ring_postnorm_available_=false,cold_rmsnorm_pad16_=true;
 bool cold_down_nextnorm_available_=false,cold_qk_rope_available_=false,cold_partial_rope_=true;
 bool cold_qk_rope_norm_available_=false,cold_qk_rope_norm_kv_available_=false;
 bool cold_fp16_kv_grouped_available_=false,cold_w8_kv_grouped_available_=false;
 bool cold_nvfp4_kv_available_=false,cold_nvfp4_qkv_available_=false;
 bool cold_nvfp4_q_n32_available_=false,cold_nvfp4_qkv_overlap_available_=false;
 bool cold_nvfp4_o_overlap_available_=false,cold_nvfp4_down_overlap_available_=false;
 bool cold_nvfp4_gateup_overlap_available_=false,cold_nvfp4_resident_overlap_available_=false;
 bool cold_sdpa_m16_available_=false,cold_sdpa_k128_available_=false;
 double eps_expert_=1e-5;RpuRmsNormSpmContract cold_rmsnorm_spm_contract_{};

 int64_t weight_bytes=524288,resident_layers=6;
 int64_t gateup_weight_bytes()const{return weight_bytes;}
 int64_t gateup_resident_layers()const{return resident_layers;}
 int64_t gateup_bundle_bytes()const{return resident_layers*2*weight_bytes;}
 int64_t resident_storage_bytes()const{return gateup_bundle_bytes() + o_resident_layers_*262144 + resident_fp8_scale_layers_*32768;}
 bool cold_w8_gateup_n48_available_=false;
 bool gateup_resident_profile()const{return profile;}
 int attn_tp()const{return 8;}int num_kv_heads()const{return 8;}int head_dim()const{return 128;}
 int num_cores()const{return 8;}std::array<int64_t,6> cold_topology_arguments()const{return {2,8,8,8,8,8};}
 bool paired=false;
 bool paired_kv_enabled(const KvInsertSegmentPlan& p)const{return paired && p.segment_count()==1 &&
  optimized_prefix_supported(p.segment(0).position) &&
  rpu_kvinsert_route_arguments(p,8,8,128)==rpu_kvinsert_route_arguments(exact_plan(p.segment(0).position),8,8,128);}
 static std::vector<int64_t> paired_kv_arguments(const KvInsertSegmentPlan& p){return {1,p.logical_rows(),p.physical_rows(),p.segment(0).position,8,8,128};}
 std::vector<int64_t> gateup_resident_arguments()const{
  std::vector<int64_t> out{nvfp4_?3:(weight_bytes==262144?2:1),resident_layers,32,51,1024,2048,8,10,2*weight_bytes,gateup_bundle_bytes()};
  if(nvfp4_)out.insert(out.end(),{320,64,128,8,24,27,32});return out;}
 METHOD
};
int main(){
 Model m;FmbPhysicalExecutionManifest manifest;
 manifest.routes={{FmbRouteFamily::KV_INSERT,HYVLA_EXPERT_KV_SITE,0,0,0,{}},
  {FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_GATEUP_RESIDENT_SITE,1,0,0,m.gateup_resident_arguments()},
  {FmbRouteFamily::MUTABLE_DMA,HYVLA_EXPERT_GATEUP_PRELOAD_SITE,7,0,0,{6,3145728,6291456,8,0}}};
 auto admit=[&](const FmbPhysicalExecutionManifest& v){return m.physical_manifest_forward_capability(v).supported;};
 assert(admit(manifest));

 // Grouped FP16/W8 KV has one exact receipt and still requires the original Q.
 for(bool w8:{false,true}){
  Model grouped=m;grouped.weight_bytes=w8?262144:524288;
  grouped.cold_gateup_swiglu_available_=true;
  grouped.cold_fp16_kv_grouped_available_=true;grouped.cold_w8_kv_grouped_available_=true;
  auto valid=manifest;valid.routes[1].arguments=grouped.gateup_resident_arguments();
  valid.routes[2].arguments={6,grouped.gateup_bundle_bytes()/2,grouped.gateup_bundle_bytes(),8,0};
  valid.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_GATEUP_SWIGLU_SITE,1,0,0,grouped.gateup_swiglu_arguments(240)});
  valid.routes.push_back({FmbRouteFamily::LINEAR,HYVLA_EXPERT_QKV_LINEAR_SITE,1,0,0,{}});
  const auto recipe=grouped.kv_grouped_arguments(240);
  assert(recipe.size()==30 && recipe[1]==(w8?8:16) && recipe[11]==(w8?352:208) && recipe[14]==(w8?6:4));
  auto accepts=[&](const FmbPhysicalExecutionManifest& v){return grouped.physical_manifest_forward_capability(v).supported;};
  assert(!accepts(valid));
  valid.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_KV_GROUPED_SITE,1,0,0,recipe});
  assert(accepts(valid));
  for(size_t i=0;i<recipe.size();++i){auto bad=valid;++bad.routes.back().arguments[i];assert(!accepts(bad));}
  for(int f=0;f<6;++f){auto bad=valid;auto& r=bad.routes.back();
   if(f==0)r.family=FmbRouteFamily::LINEAR;if(f==1)++r.selector;if(f==2)++r.flags;if(f==3)++r.invocation;
   if(f==4)bad.routes.push_back(r);if(f==5)++bad.physical_length;assert(!accepts(bad));}
  {auto bad=valid;bad.routes.erase(bad.routes.end()-2);assert(!accepts(bad));}
  {auto bad=valid;bad.routes.push_back(valid.routes[valid.routes.size()-2]);assert(!accepts(bad));}
  {auto bad=valid;bad.routes[bad.routes.size()-2].arguments={51};assert(!accepts(bad));}
  for(int p:{128,144,160,176,192,208,224,240}){assert(grouped.kv_grouped_enabled(51,p));
   auto args=recipe;args[4]=p;assert(grouped.kv_grouped_arguments(p)==args);}
  for(int p:{112,127,129,239,241,256})assert(!grouped.kv_grouped_enabled(51,p));
  assert(!grouped.kv_grouped_enabled(50,240));
  grouped.nvfp4_=true;assert(!grouped.kv_grouped_enabled(51,240));grouped.nvfp4_=false;
  bool& marker=w8?grouped.cold_w8_kv_grouped_available_:grouped.cold_fp16_kv_grouped_available_;
  marker=false;assert(!accepts(valid) && !grouped.kv_grouped_enabled(51,240));
  valid.routes.pop_back();assert(accepts(valid));marker=true;
  grouped.gateup_resident_bundle_.present=false;assert(!grouped.kv_grouped_enabled(51,240));
 }
 // NVFP4 COMPLETE requires seven distinct, full FP32[32] table roles and its own recipes.
 FmbPhysicalExecutionManifest nv12_manifest;
 for(int64_t count:{12,24}){Model nv;nv.nvfp4_=true;nv.weight_bytes=131072;nv.resident_layers=count;
  nv.cold_nvfp4_gateup_available_=true;nv.cold_nvfp4_o_ring_postnorm_available_=true;
  nv.cold_ring_postnorm_available_=true;nv.cold_qk_rope_available_=true;nv.cold_down_nextnorm_available_=true;
  FmbPhysicalExecutionManifest nm;
  nm.routes={{FmbRouteFamily::KV_INSERT,HYVLA_EXPERT_KV_SITE,0,0,0,{}},
   {FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_GATEUP_RESIDENT_SITE,1,0,0,nv.gateup_resident_arguments()},
   {FmbRouteFamily::MUTABLE_DMA,HYVLA_EXPERT_GATEUP_PRELOAD_SITE,7,0,0,{count,count*131072,count*262144,8,0}},
   {FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_GATEUP_SWIGLU_SITE,1,0,0,nv.gateup_swiglu_arguments(240)},
   {FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_O_RING_POSTNORM_SITE,1,0,0,nv.o_ring_postnorm_arguments(240)},
   {FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_QK_ROPE_SITE,1,0,0,nv.qk_rope_arguments(240)},
   {FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_DOWN_NEXTNORM_SITE,1,0,0,nv.down_nextnorm_arguments(240)}};
  auto accepts=[&](const FmbPhysicalExecutionManifest& v){try{return nv.physical_manifest_forward_capability(v).supported;}catch(const std::exception&){return false;}};
  assert(!accepts(nm));
  for(int role=0;role<7;++role)nm.routes.push_back({FmbRouteFamily::MUTABLE_DMA,CAUSAL_DECODER_NVFP4_SCALE_PRELOAD_SITE,1,0,role,{role,32,64,1}});
  assert(accepts(nm));
  for(int which=0;which<4;++which){
   Model overlap=nv;auto baseline=nm;size_t site=0;
   if(which==0){overlap.cold_nvfp4_o_n128_available_=true;site=4;
    baseline.routes[site].arguments=overlap.o_ring_postnorm_arguments(240);}
   if(which==1){overlap.cold_nvfp4_down_nextnorm_available_=true;site=6;
    baseline.routes[site]={FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_DOWN_NVFP4_NEXTNORM_SITE,1,0,0,overlap.down_nvfp4_nextnorm_arguments(240)};}
   if(which==2)site=3;
   if(which==3){overlap.cold_nvfp4_resident_swiglu_available_=true;site=baseline.routes.size();
    baseline.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_NVFP4_RESIDENT_SWIGLU_SITE,1,0,0,overlap.resident_swiglu_arguments(240)});}
   auto admits=[&](const FmbPhysicalExecutionManifest& v){try{return overlap.physical_manifest_forward_capability(v).supported;}catch(const std::exception&){return false;}};
   assert(admits(baseline));
   bool* flags[]{&overlap.cold_nvfp4_o_overlap_available_,&overlap.cold_nvfp4_down_overlap_available_,
    &overlap.cold_nvfp4_gateup_overlap_available_,&overlap.cold_nvfp4_resident_overlap_available_};
   *flags[which]=true;assert(!admits(baseline));
   auto combined=baseline;auto& recipe=combined.routes[site].arguments;
   recipe[0]=std::array<int,4>{5,4,6,4}[which];recipe.insert(recipe.end(),{106,105,0,1,1});
   assert(admits(combined));
   for(size_t i=0;i<recipe.size();++i){auto bad=combined;++bad.routes[site].arguments[i];assert(!admits(bad));}
   for(int f=0;f<4;++f){auto bad=combined;auto& r=bad.routes[site];
    if(f==0)r.family=FmbRouteFamily::LINEAR;if(f==1)++r.selector;if(f==2)++r.flags;if(f==3)++r.invocation;assert(!admits(bad));}
   {auto bad=combined;bad.routes.push_back(bad.routes[site]);assert(!admits(bad));}
   {auto bad=combined;bad.routes.erase(bad.routes.begin()+site);assert(!admits(bad));}
   {auto bad=baseline;bad.routes[site].arguments[0]=recipe[0];assert(!admits(bad));}
   *flags[which]=false;assert(!admits(combined) && admits(baseline));
  }
  if(count==24)for(int scale_count:{4,8})for(bool fused:{false,true}){
   auto combined=nm;
   nv.resident_fp8_scale_layers_=scale_count;nv.cold_nvfp4_resident_swiglu_available_=fused;
   nv.cold_nvfp4_resident_scales_available_=!fused;nv.cold_nvfp4_resident_scales_swiglu_available_=fused;
   combined.routes[2].arguments={24,3145728+scale_count*16384,6291456+scale_count*32768,8,0};
   if(fused)combined.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_NVFP4_RESIDENT_SWIGLU_SITE,1,0,0,nv.resident_swiglu_arguments(240)});
   assert(!accepts(combined));
   const auto recipe=nv.resident_fp8_scales_arguments(240);
   combined.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_NVFP4_RESIDENT_SCALES_SITE,1,0,0,recipe});
   assert(accepts(combined) && !accepts(nm));
   for(size_t i=0;i<recipe.size();++i){auto bad=combined;++bad.routes.back().arguments[i];assert(!accepts(bad));}
   for(int f=0;f<4;++f){auto bad=combined;auto& r=bad.routes.back();if(f==0)r.family=FmbRouteFamily::LINEAR;
    if(f==1)++r.selector;if(f==2)++r.flags;if(f==3)++r.invocation;assert(!accepts(bad));}
   {auto bad=combined;bad.routes.push_back(bad.routes.back());assert(!accepts(bad));}
   {auto bad=combined;bad.routes[2].arguments={24,3145728,6291456,8,0};assert(!accepts(bad));}
   {auto bad=combined;const int other=12-scale_count;
    bad.routes.back().arguments[3]=other;bad.routes.back().arguments[13]=other*32768;
    bad.routes.back().arguments[14]=6291456+other*32768;
    bad.routes[2].arguments={24,3145728+other*16384,6291456+other*32768,8,0};assert(!accepts(bad));}
   {auto bad=combined;--bad.kv_logical_length;assert(!accepts(bad));}
   bool& cap=fused?nv.cold_nvfp4_resident_scales_swiglu_available_:nv.cold_nvfp4_resident_scales_available_;
   cap=false;assert(!accepts(combined));cap=true;
   nv.resident_fp8_scale_layers_=0;assert(!accepts(combined));
   nv.cold_nvfp4_resident_swiglu_available_=false;
   nv.cold_nvfp4_resident_scales_available_=false;nv.cold_nvfp4_resident_scales_swiglu_available_=false;
   assert(accepts(nm));
  }
  {
   auto old=nm;old.routes.push_back({FmbRouteFamily::ACTIVATION,HYVLA_EXPERT_MLP_SILU_MUL_SITE,3,0,0,{1,13056,8}});
   assert(accepts(old));
   auto combined=nm;
   const std::vector<int64_t> recipe{3,32,0,count,51,240,1024,2048,256,8,10,
    320,64,128,8,128,1,1,13056,32512,13056,48,24,36,27,32,2,32,20,34,38};
   assert(recipe.size()==31 && nv.resident_swiglu_arguments(240)==recipe);
   combined.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_NVFP4_RESIDENT_SWIGLU_SITE,1,0,0,recipe});
   assert(!accepts(combined));nv.cold_nvfp4_resident_swiglu_available_=true;
   assert(accepts(combined) && !accepts(old) && !accepts(nm));
   for(size_t i=0;i<recipe.size();++i){auto bad=combined;++bad.routes.back().arguments[i];assert(!accepts(bad));}
   for(int field=0;field<4;++field){auto bad=combined;auto& route=bad.routes.back();
    if(field==0)route.family=FmbRouteFamily::ACTIVATION;if(field==1)++route.selector;
    if(field==2)++route.flags;if(field==3)++route.invocation;assert(!accepts(bad));}
   for(int delta:{-1,1}){auto bad=combined;if(delta<0)bad.routes.back().arguments.pop_back();
    else bad.routes.back().arguments.push_back(0);assert(!accepts(bad));}
   {auto bad=combined;bad.routes.push_back(bad.routes.back());assert(!accepts(bad));}
   {auto bad=combined;bad.routes.back().site_id=HYVLA_EXPERT_RESIDENT_SWIGLU_SITE;assert(!accepts(bad));}
   {auto bad=combined;bad.routes.back().arguments[3]=count==12?24:12;assert(!accepts(bad));}
   for(auto site:{HYVLA_EXPERT_RESIDENT_SWIGLU_SITE,HYVLA_EXPERT_MLP_SILU_MUL_SITE,
                  HYVLA_EXPERT_MLP_SILU_SITE,HYVLA_EXPERT_MLP_MUL_SITE,
                  HYVLA_EXPERT_GATE_LINEAR_SITE,HYVLA_EXPERT_UP_LINEAR_SITE}){
    auto bad=combined;FmbRouteManifestEntry extra{FmbRouteFamily::GRAPH_SCHEDULE,site,1,0,0,{}};
    bad.routes.push_back(extra);assert(!accepts(bad));
    bad=combined;bad.routes.insert(bad.routes.begin(),extra);assert(!accepts(bad));
   }
   // Preload, resident producer ownership and remaining DDR layers are still required.
   for(size_t index:{1,2,3}){auto bad=combined;bad.routes.erase(bad.routes.begin()+index);assert(!accepts(bad));}
   for(int which=0;which<3;++which){auto bad=combined;
    if(which==0)++bad.logical_length;if(which==1)++bad.physical_length;if(which==2)++bad.kv_logical_length;
    assert(!accepts(bad));}
   for(bool* enabled:{&nv.cold_nvfp4_gateup_available_,&nv.gateup_resident_bundle_.present,&nv.profile,&nv.nvfp4_}){
    *enabled=false;assert(!accepts(combined));*enabled=true;
   }
   for(int bad:{6,13,23,25,32}){nv.resident_layers=bad;assert(!accepts(combined));}nv.resident_layers=count;
   nv.cold_nvfp4_resident_swiglu_available_=false;assert(!accepts(combined) && accepts(old) && accepts(nm));
  }
  for(int mode:{2,3}){
   auto grouped=nm;
   auto recipe=nv.nvfp4_qkv_arguments(mode,240);assert(recipe.size()==30);
   grouped.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_NVFP4_QKV_SITE,1,0,0,recipe});
   size_t index=grouped.routes.size()-1;
   const FmbRouteManifestEntry qlinear{FmbRouteFamily::LINEAR,HYVLA_EXPERT_QKV_LINEAR_SITE,1,0,0,{}};
   if(mode==2)grouped.routes.push_back(qlinear);
   assert(!accepts(grouped));
   nv.cold_nvfp4_kv_available_=true;nv.cold_nvfp4_qkv_available_=mode==3;
   assert(nv.nvfp4_qkv_mode(51,240)==mode && accepts(grouped) && !accepts(nm));
   for(size_t i=0;i<recipe.size();++i){auto bad=grouped;++bad.routes[index].arguments[i];assert(!accepts(bad));}
   for(int field=0;field<4;++field){auto bad=grouped;auto& route=bad.routes[index];
    if(field==0)route.family=FmbRouteFamily::LINEAR;if(field==1)++route.selector;
    if(field==2)++route.flags;if(field==3)++route.invocation;assert(!accepts(bad));}
   for(int delta:{-1,1}){auto bad=grouped;if(delta<0)bad.routes[index].arguments.pop_back();
    else bad.routes[index].arguments.push_back(0);assert(!accepts(bad));}
   {auto bad=grouped;bad.routes.push_back(bad.routes[index]);assert(!accepts(bad));}
   {auto bad=grouped;bad.routes.erase(bad.routes.begin()+index);assert(!accepts(bad));}
   {auto bad=grouped;++bad.physical_length;assert(!accepts(bad));}
   {auto bad=grouped;bad.routes.push_back(qlinear);assert(!accepts(bad));}
   if(mode==2){auto bad=grouped;bad.routes.pop_back();assert(!accepts(bad));
    for(int field=0;field<5;++field){bad=grouped;auto& route=bad.routes.back();
     if(field==0)route.family=FmbRouteFamily::GRAPH_SCHEDULE;if(field==1)++route.selector;
     if(field==2)++route.flags;if(field==3)++route.invocation;
     if(field==4)route.arguments.push_back(1);assert(!accepts(bad));}
    nv.cold_nvfp4_qkv_available_=true;assert(!accepts(grouped));
   }else{nv.cold_nvfp4_qkv_available_=false;assert(!accepts(grouped));}
   nv.cold_nvfp4_kv_available_=false;nv.cold_nvfp4_qkv_available_=false;
   assert(!accepts(grouped) && accepts(nm));
  }
  for(bool k128:{false,true}){
   bool& selected=k128?nv.cold_sdpa_k128_available_:nv.cold_sdpa_m16_available_;
   bool& other=k128?nv.cold_sdpa_m16_available_:nv.cold_sdpa_k128_available_;
   auto candidate=nm;
   candidate.routes.push_back({FmbRouteFamily::ATTENTION,HYVLA_EXPERT_ATTN_SITE,1,0,0,{}});
   candidate.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,(k128?HYVLA_EXPERT_SDPA_K128_SITE:HYVLA_EXPERT_SDPA_M16_SITE),1,0,0,k128?nv.sdpa_k128_arguments(240):nv.sdpa_m16_arguments(240)});
   assert(!accepts(candidate));selected=true;
   assert(accepts(candidate)==(count==24));
   other=true;assert(!accepts(candidate));other=false;
   if(count==24){
    assert(!accepts(nm));
    const size_t first=candidate.routes.size()-2;
    for(size_t r=first;r<candidate.routes.size();++r){
     for(size_t i=0;i<candidate.routes[r].arguments.size();++i){auto bad=candidate;++bad.routes[r].arguments[i];assert(!accepts(bad));}
     for(int field=0;field<4;++field){auto bad=candidate;auto& route=bad.routes[r];
      if(field==0)route.family=FmbRouteFamily::LINEAR;if(field==1)++route.selector;
      if(field==2)++route.flags;if(field==3)++route.invocation;assert(!accepts(bad));}
     {auto bad=candidate;bad.routes[r].arguments.push_back(0);assert(!accepts(bad));}
     {auto bad=candidate;bad.routes.push_back(bad.routes[r]);assert(!accepts(bad));}
     {auto bad=candidate;bad.routes.erase(bad.routes.begin()+r);assert(!accepts(bad));}
    }
    {auto bad=candidate;bad.routes.back().arguments.pop_back();assert(!accepts(bad));}
    for(int dim=0;dim<3;++dim){auto bad=candidate;if(dim==0)++bad.logical_length;if(dim==1)++bad.physical_length;if(dim==2)++bad.kv_logical_length;assert(!accepts(bad));}
    for(bool* required:{&nv.nvfp4_,&nv.profile,&nv.cold_nvfp4_gateup_available_}){*required=false;assert(!accepts(candidate));*required=true;}
    nv.resident_layers=12;assert(!accepts(candidate));nv.resident_layers=count;
   }
   selected=false;assert(!accepts(candidate) && accepts(nm));
  }
  // Candidate libraries bind distinct real producer recipes. Old mode2/mode3
  // manifests cannot silently stand in for Q N32 or for overlapped QKV.
  for(bool overlap:{false,true}){
   nv.cold_nvfp4_kv_available_=true;nv.cold_nvfp4_qkv_available_=true;
   const auto old_qkv=nv.nvfp4_qkv_arguments(overlap?3:2,240);
   nv.cold_nvfp4_q_n32_available_=!overlap;nv.cold_nvfp4_qkv_overlap_available_=overlap;
   const int mode=overlap?3:2;assert(nv.nvfp4_qkv_mode(51,240)==mode);
   auto candidate=nm;
   auto grouped_recipe=nv.nvfp4_qkv_arguments(mode,240);
   if(overlap){auto expected=old_qkv;expected[0]=2;expected.insert(expected.end(),{106,105,0,1,1});assert(grouped_recipe==expected);}
   else assert(grouped_recipe==old_qkv);
   candidate.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_NVFP4_QKV_SITE,1,0,0,grouped_recipe});
   const size_t first=candidate.routes.size()-1;
   if(!overlap)candidate.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_NVFP4_Q_N32_SITE,1,0,0,nv.nvfp4_q_n32_arguments(240)});
   assert(accepts(candidate) && !accepts(nm));
   for(size_t r=first;r<candidate.routes.size();++r){
    for(size_t i=0;i<candidate.routes[r].arguments.size();++i){auto bad=candidate;++bad.routes[r].arguments[i];assert(!accepts(bad));}
    for(int field=0;field<4;++field){auto bad=candidate;auto& route=bad.routes[r];
     if(field==0)route.family=FmbRouteFamily::LINEAR;if(field==1)++route.selector;
     if(field==2)++route.flags;if(field==3)++route.invocation;assert(!accepts(bad));}
    for(int delta:{-1,1}){auto bad=candidate;if(delta<0)bad.routes[r].arguments.pop_back();else bad.routes[r].arguments.push_back(0);assert(!accepts(bad));}
    {auto bad=candidate;bad.routes.push_back(bad.routes[r]);assert(!accepts(bad));}
    {auto bad=candidate;bad.routes.erase(bad.routes.begin()+r);assert(!accepts(bad));}
   }
   const FmbRouteManifestEntry qlinear{FmbRouteFamily::LINEAR,HYVLA_EXPERT_QKV_LINEAR_SITE,1,0,0,{}};
   {auto bad=candidate;bad.routes.push_back(qlinear);assert(!accepts(bad));}
   {auto old=candidate;if(overlap)old.routes[first].arguments=old_qkv;else old.routes.back()=qlinear;assert(!accepts(old));}
   for(int dim=0;dim<3;++dim){auto bad=candidate;if(dim==0)++bad.logical_length;if(dim==1)++bad.physical_length;if(dim==2)++bad.kv_logical_length;assert(!accepts(bad));}
   for(bool* required:{&nv.nvfp4_,&nv.profile,&nv.cold_nvfp4_gateup_available_}){*required=false;assert(!accepts(candidate));*required=true;}
   bool* flag=overlap?&nv.cold_nvfp4_qkv_overlap_available_:&nv.cold_nvfp4_q_n32_available_;
   *flag=false;assert(!accepts(candidate));*flag=true;
   bool* dependency=overlap?&nv.cold_nvfp4_qkv_available_:&nv.cold_nvfp4_kv_available_;
   *dependency=false;assert(!accepts(candidate));*dependency=true;
   nv.cold_nvfp4_q_n32_available_=true;nv.cold_nvfp4_qkv_overlap_available_=true;
   assert(!accepts(candidate));
   nv.cold_nvfp4_q_n32_available_=false;nv.cold_nvfp4_qkv_overlap_available_=false;
   nv.cold_nvfp4_kv_available_=false;nv.cold_nvfp4_qkv_available_=false;
   assert(!accepts(candidate) && accepts(nm));
  }
  // The fused route owns all padded Q/K normalization and cannot coexist with
  // the old R15 or either separate head norm receipt, in either route order.
  {
   auto combined=nm;
   auto recipe=nv.qk_rope_arguments(240);
   recipe.insert(recipe.end(),{112,64,128,14,8,8,32,44,128,130,2,5,1,100000,1});
   assert(recipe.size()==33 && nv.qk_rope_norm_arguments(240)==recipe);
   combined.routes[5]={FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_QK_ROPE_NORM_SITE,1,0,0,recipe};
   assert(!accepts(combined));
   nv.cold_qk_rope_norm_available_=true;
   assert(!accepts(nm) && accepts(combined));
   for(size_t i=0;i<recipe.size();++i){auto bad=combined;++bad.routes[5].arguments[i];assert(!accepts(bad));}
   for(int i=0;i<4;++i){auto bad=combined;auto& route=bad.routes[5];
    if(i==0)route.family=FmbRouteFamily::LINEAR;if(i==1)++route.selector;
    if(i==2)++route.flags;if(i==3)++route.invocation;assert(!accepts(bad));}
   for(int extra:{-1,1}){auto bad=combined;
    if(extra<0)bad.routes[5].arguments.pop_back();else bad.routes[5].arguments.push_back(0);
    assert(!accepts(bad));}
   {auto bad=combined;bad.routes.erase(bad.routes.begin()+5);assert(!accepts(bad));}
   {auto bad=combined;bad.routes.push_back(bad.routes[5]);assert(!accepts(bad));}
   {auto bad=combined;++bad.physical_length;assert(!accepts(bad));}
   for(auto site:{HYVLA_EXPERT_QK_ROPE_SITE,HYVLA_EXPERT_Q_NORM_SITE,HYVLA_EXPERT_K_NORM_SITE,
                  HYVLA_EXPERT_Q_PARTIAL_ROPE_SITE,HYVLA_EXPERT_K_PARTIAL_ROPE_SITE,
                  HYVLA_EXPERT_Q_ROPE_SITE,HYVLA_EXPERT_K_ROPE_SITE}){
    FmbRouteManifestEntry old{FmbRouteFamily::GRAPH_SCHEDULE,site,1,0,0,{}};
    auto bad=combined;bad.routes.push_back(old);assert(!accepts(bad));
    bad=combined;bad.routes.insert(bad.routes.begin(),old);assert(!accepts(bad));
   }
   for(bool* cap:{&nv.cold_rmsnorm_spm_contract_.q_vector,&nv.cold_rmsnorm_spm_contract_.k_vector}){
    *cap=false;assert(!accepts(combined) && accepts(nm));*cap=true;
   }
   nv.cold_qk_rope_norm_available_=false;assert(!accepts(combined) && accepts(nm));
  }
  {
   // R31 consumes the same typed KV plan and padding, replacing exactly R29
   // plus the paired writer, independently of the projection precision.
   const auto check_qk_kv=[](Model nv,FmbPhysicalExecutionManifest nm){
   const auto accepts=[&](const FmbPhysicalExecutionManifest& v){
    try{return nv.physical_manifest_forward_capability(v).supported;}catch(const std::exception&){return false;}
   };
   nv.cold_qk_rope_norm_available_=true;nv.paired=true;
   auto legacy=nm;
   const auto plan=exact_plan();const auto wire=rpu_kvinsert_route_arguments(plan,8,8,128);
   legacy.routes[0]={FmbRouteFamily::KV_INSERT,HYVLA_EXPERT_KV_SITE,3,2,0,{wire.begin(),wire.end()}};
   legacy.routes[5]={FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_QK_ROPE_NORM_SITE,1,0,0,nv.qk_rope_norm_arguments(240)};
   legacy.routes.push_back({FmbRouteFamily::KV_INSERT,HYVLA_EXPERT_KV_PADDING_SITE,2,0,0,{1,51,64,64,8,8,128}});
   legacy.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_PAIRED_KV_SITE,1,0,0,nv.paired_kv_arguments(plan)});
   assert(accepts(legacy));
   auto combined=legacy;combined.routes.pop_back();
   auto recipe=nv.qk_rope_norm_arguments(240);
   recipe.insert(recipe.end(),{1,51,64,240,8,8,128,2,4,132,3,134,13,
    0,8,1,9,2,14,3,10,6,16,7,17,8,20,9,21,10,22,11,23,12,24,13,25,21,26});
   assert(recipe.size()==72 && nv.qk_rope_norm_kv_arguments(plan,240)==recipe);
   combined.routes[5]={FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_QK_ROPE_NORM_KV_SITE,1,0,0,recipe};
   assert(!accepts(combined));nv.cold_qk_rope_norm_kv_available_=true;
   assert(accepts(combined) && !accepts(legacy));
   // The original writer packet used invalid SDK registers88/92. Reject its
   // old49-field receipt even when all shape, plan and cold owners match.
   {auto stale=combined;auto& a=stale.routes[5].arguments;a.resize(40);
    a.insert(a.end(),{80,4,132,3,134,88,89,92,93});assert(!accepts(stale));}
   {auto reversed=combined;std::reverse(reversed.routes.begin(),reversed.routes.end());assert(accepts(reversed));}
   for(size_t r:{size_t(0),size_t(5),combined.routes.size()-1}){
    auto bad=combined;bad.routes.erase(bad.routes.begin()+r);assert(!accepts(bad));
    bad=combined;bad.routes.push_back(bad.routes[r]);assert(!accepts(bad));
    for(size_t i=0;i<combined.routes[r].arguments.size();++i){bad=combined;++bad.routes[r].arguments[i];assert(!accepts(bad));}
    bad=combined;bad.routes[r].arguments.pop_back();assert(!accepts(bad));
    bad=combined;bad.routes[r].arguments.push_back(0);assert(!accepts(bad));
    for(int i=0;i<4;++i){bad=combined;auto& route=bad.routes[r];
     if(i==0)route.family=FmbRouteFamily::LINEAR;if(i==1)++route.selector;
     if(i==2)++route.flags;if(i==3)++route.invocation;assert(!accepts(bad));}
   }
   for(auto site:{HYVLA_EXPERT_QK_ROPE_NORM_SITE,HYVLA_EXPERT_PAIRED_KV_SITE,
                  HYVLA_EXPERT_QK_ROPE_SITE,HYVLA_EXPERT_Q_NORM_SITE,HYVLA_EXPERT_K_NORM_SITE,
                  HYVLA_EXPERT_Q_PARTIAL_ROPE_SITE,HYVLA_EXPERT_K_PARTIAL_ROPE_SITE,
                  HYVLA_EXPERT_Q_ROPE_SITE,HYVLA_EXPERT_K_ROPE_SITE}){
    FmbRouteManifestEntry old{FmbRouteFamily::GRAPH_SCHEDULE,site,1,0,0,{}};
    auto bad=combined;bad.routes.push_back(old);assert(!accepts(bad));
    bad=combined;bad.routes.insert(bad.routes.begin(),old);assert(!accepts(bad));
   }
   for(int i=0;i<3;++i){auto bad=combined;
    if(i==0)++bad.logical_length;if(i==1)++bad.physical_length;if(i==2)++bad.kv_logical_length;
    assert(!accepts(bad));}
   for(auto wrong:{exact_plan(224),exact_plan(240,51,KvInsertRoute::V2,51),exact_plan(240,64,KvInsertRoute::ALIGNED_V16,64)}){
    auto bad=combined;const auto wrong_wire=rpu_kvinsert_route_arguments(wrong,8,8,128);
    bad.routes[0].arguments={wrong_wire.begin(),wrong_wire.end()};bad.routes[0].selector=int64_t(wrong.route());
    assert(!accepts(bad));
   }
   for(bool* enabled:{&nv.cold_qk_rope_norm_available_,&nv.paired,&nv.profile,
                     &nv.cold_rmsnorm_spm_contract_.q_vector,&nv.cold_rmsnorm_spm_contract_.k_vector}){
    *enabled=false;assert(!accepts(combined));*enabled=true;
   }
   nv.cold_qk_rope_norm_kv_available_=false;assert(!accepts(combined) && accepts(legacy));
   nv.cold_qk_rope_norm_available_=false;nv.paired=false;assert(accepts(nm));
   };
   check_qk_kv(nv,nm);
   for(bool quantized:{false,true}){
    auto model=nv;model.nvfp4_=false;model.weight_bytes=quantized?262144:524288;
    model.resident_layers=quantized?12:6;model.cold_gateup_swiglu_available_=true;
    model.cold_w8_o_ring_postnorm_available_=quantized;
    auto baseline=nm;
    baseline.routes.erase(std::remove_if(baseline.routes.begin(),baseline.routes.end(),[](const auto& r){
     return r.site_id==CAUSAL_DECODER_NVFP4_SCALE_PRELOAD_SITE;
    }),baseline.routes.end());
    baseline.routes[1].arguments=model.gateup_resident_arguments();
    baseline.routes[2].arguments={model.resident_layers,model.resident_storage_bytes()/2,model.resident_storage_bytes(),8,0};
    baseline.routes[3].arguments=model.gateup_swiglu_arguments(240);
    baseline.routes[4]={FmbRouteFamily::GRAPH_SCHEDULE,
     quantized?HYVLA_EXPERT_O_RING_POSTNORM_SITE:HYVLA_EXPERT_RING_POSTNORM_SITE,
     1,0,0,quantized?model.o_ring_postnorm_arguments(240):model.ring_postnorm_arguments(240)};
    check_qk_kv(model,baseline);
   }
  }
  if(count==24){
   // Final NV4 paths share one real prefix and real serialized KV plan.
   // Exercise retained QKV overlap/R31/R32/Down/O together, with optional
   // attention and resident FP8 scales independently present or absent.
   auto prefix=nv;prefix.paired=true;
   prefix.cold_qk_rope_norm_available_=prefix.cold_qk_rope_norm_kv_available_=true;
   prefix.cold_nvfp4_qkv_available_=prefix.cold_nvfp4_qkv_overlap_available_=true;
   prefix.cold_nvfp4_resident_swiglu_available_=prefix.cold_nvfp4_o_n128_available_=true;
   prefix.cold_nvfp4_down_nextnorm_available_=true;
   const auto accepts_prefix=[&](const FmbPhysicalExecutionManifest& x){
    try{return prefix.physical_manifest_forward_capability(x).supported;}catch(const std::exception&){return false;}
   };
   const auto make=[&](int position){
    auto out=nm;out.kv_logical_length=position+51;
    const auto p=exact_plan(position);const auto wire=rpu_kvinsert_route_arguments(p,8,8,128);
    out.routes[0]={FmbRouteFamily::KV_INSERT,HYVLA_EXPERT_KV_SITE,3,2,0,{wire.begin(),wire.end()}};
    out.routes[2].arguments={24,prefix.resident_storage_bytes()/2,prefix.resident_storage_bytes(),8,0};
    out.routes[3].arguments=prefix.gateup_swiglu_arguments(position);
    out.routes[4].arguments=prefix.o_ring_postnorm_arguments(position);
    out.routes[5]={FmbRouteFamily::GRAPH_SCHEDULE,
     prefix.cold_qk_rope_norm_kv_available_?HYVLA_EXPERT_QK_ROPE_NORM_KV_SITE:HYVLA_EXPERT_QK_ROPE_NORM_SITE,
     1,0,0,prefix.cold_qk_rope_norm_kv_available_?prefix.qk_rope_norm_kv_arguments(p,position):prefix.qk_rope_norm_arguments(position)};
    out.routes[6]={FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_DOWN_NVFP4_NEXTNORM_SITE,1,0,0,prefix.down_nvfp4_nextnorm_arguments(position)};
    out.routes.push_back({FmbRouteFamily::KV_INSERT,HYVLA_EXPERT_KV_PADDING_SITE,2,0,0,{1,51,64,64,8,8,128}});
    if(!prefix.cold_qk_rope_norm_kv_available_)
     out.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_PAIRED_KV_SITE,1,0,0,prefix.paired_kv_arguments(p)});
    out.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_NVFP4_QKV_SITE,1,0,0,prefix.nvfp4_qkv_arguments(3,position)});
    out.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_NVFP4_RESIDENT_SWIGLU_SITE,1,0,0,prefix.resident_swiglu_arguments(position)});
    out.routes.push_back({FmbRouteFamily::ATTENTION,HYVLA_EXPERT_ATTN_SITE,1,0,0,{}});
    if(prefix.cold_sdpa_m16_available_)
     out.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_SDPA_M16_SITE,1,0,0,prefix.sdpa_m16_arguments(position)});
    if(prefix.resident_fp8_scale_layers_)
     out.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_NVFP4_RESIDENT_SCALES_SITE,1,0,0,prefix.resident_fp8_scales_arguments(position)});
    return out;
   };
   for(bool attention:{false,true})for(int scale_count:{0,4,8}){
    prefix.cold_sdpa_m16_available_=attention;
    prefix.resident_fp8_scale_layers_=scale_count;prefix.cold_nvfp4_resident_scales_swiglu_available_=scale_count!=0;
    for(int position:{128,144,160,176,192,208,224,240}){
     const auto good=make(position);assert(accepts_prefix(good));
     for(int other:{128,144,160,176,192,208,224,240})if(other!=position){
      auto wrong=make(other);assert(accepts_prefix(wrong));
      // The complete receipt from another legal prefix is stale here.
      for(size_t r=0;r<good.routes.size();++r)if(good.routes[r].arguments!=wrong.routes[r].arguments){
       auto bad=good;bad.routes[r].arguments=wrong.routes[r].arguments;assert(!accepts_prefix(bad));
      }
      auto bad=good;bad.kv_logical_length=other+51;assert(!accepts_prefix(bad));
      for(size_t field:{size_t(3),size_t(36)}){bad=good;bad.routes[5].arguments[field]=other;assert(!accepts_prefix(bad));}
     }
     for(int field=0;field<3;++field){auto bad=good;
      if(field==0)--bad.logical_length;if(field==1)++bad.physical_length;if(field==2)++bad.kv_logical_length;
      assert(!accepts_prefix(bad));}
     for(auto wrong:{exact_plan(position,51,KvInsertRoute::V2,51),exact_plan(position,64,KvInsertRoute::ALIGNED_V16,64)}){
      auto bad=good;const auto wire=rpu_kvinsert_route_arguments(wrong,8,8,128);
      bad.routes[0].arguments={wire.begin(),wire.end()};bad.routes[0].selector=int64_t(wrong.route());assert(!accepts_prefix(bad));
     }
     prefix.cold_qk_rope_norm_kv_available_=false;
     auto fallback=make(position);assert(accepts_prefix(fallback) && !accepts_prefix(good));
     prefix.cold_qk_rope_norm_kv_available_=true;assert(!accepts_prefix(fallback));
     prefix.cold_nvfp4_qkv_overlap_available_=false;fallback=make(position);
     assert(accepts_prefix(fallback) && !accepts_prefix(good));prefix.cold_nvfp4_qkv_overlap_available_=true;
    }
    for(int outside:{112,256})assert(!accepts_prefix(make(outside)));
   }
  }
  if(count==12)nv12_manifest=nm;
  else {
   assert(!accepts(nv12_manifest));
   // Each receipt must bind GU24 independently of its identical ABI version.
   for(size_t route:{1,2,3}){
    auto stale=nm;stale.routes[route].arguments=nv12_manifest.routes[route].arguments;
    assert(!accepts(stale));
   }
   assert(nm.routes[1].arguments[0]==3 && nm.routes[1].arguments[1]==24 && nm.routes[1].arguments[9]==6291456);
   assert(nm.routes[2].arguments==std::vector<int64_t>({24,3145728,6291456,8,0}));
   assert(nm.routes[3].arguments[0]==4 && nm.routes[3].arguments[2]==24 && nm.routes[3].arguments[3]==8);
   auto abi3=nm;
   nv.cold_nvfp4_o_n128_available_=true;
   assert(!accepts(abi3));
   auto n128=nv.ring_postnorm_arguments(240);n128[0]=4;
   n128.insert(n128.end(),{176,128,128,1,8,256,2048,2,3,20,21,24,27,32});
   assert(nv.o_ring_postnorm_arguments(240)==n128);
   nm.routes[4].arguments=n128;assert(accepts(nm));
   // A changed version alone cannot relabel the old two-wave producer.
   auto stale=abi3;stale.routes[4].arguments[0]=4;assert(!accepts(stale));
   nv.cold_nvfp4_o_ring_postnorm_available_=false;assert(accepts(nm));
   nv.cold_nvfp4_o_n128_available_=false;assert(!accepts(nm));
   nv.cold_nvfp4_o_ring_postnorm_available_=true;assert(accepts(abi3) && !accepts(nm));
   nv.cold_nvfp4_o_n128_available_=true;assert(accepts(nm));
   auto gu_abi4=nm;nv.cold_nvfp4_gateup_input_reuse_available_=true;
   assert(!accepts(gu_abi4));
   const std::vector<int64_t> input_reuse{5,32,24,8,51,240,1024,2048,256,8,10,480,32,128,8,24,36,27,128,130,28,256,64,296,356,2};
   assert(nv.gateup_swiglu_arguments(240)==input_reuse);
   nm.routes[3].arguments=input_reuse;assert(accepts(nm));
   // New ABI identifies the complete preload/VMM recipe, not just a version.
   stale=gu_abi4;stale.routes[3].arguments[0]=5;assert(!accepts(stale));
   nv.cold_nvfp4_gateup_input_reuse_available_=false;
   assert(accepts(gu_abi4) && !accepts(nm));
   nv.cold_nvfp4_gateup_input_reuse_available_=true;assert(accepts(nm));
   nv.cold_nvfp4_gateup_available_=false;assert(!accepts(nm));
   nv.cold_nvfp4_gateup_available_=true;assert(accepts(nm));
   const FmbRouteManifestEntry old_down{FmbRouteFamily::LINEAR,HYVLA_EXPERT_DOWN_LINEAR_SITE,1,0,0,{51,1024,2048,0,8}};
   auto auto_down=nm;auto_down.routes.push_back(old_down);assert(accepts(auto_down));
   nv.cold_nvfp4_down_n128_available_=true;
   assert(!accepts(nm) && !accepts(auto_down));
   const std::vector<int64_t> down_n128{1,32,51,240,1024,2048,256,8,10,176,128,128,1,8,0,24,27,32,2,3,20,21};
   assert(nv.down_nvfp4_n128_arguments(240)==down_n128);
   nm.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_DOWN_NVFP4_N128_SITE,1,0,0,down_n128});
   assert(accepts(nm));
   auto mixed=nm;mixed.routes.push_back(old_down);assert(!accepts(mixed));
   mixed=nm;mixed.routes.insert(mixed.routes.begin(),old_down);assert(!accepts(mixed));
   nv.cold_nvfp4_down_n128_available_=false;assert(!accepts(nm) && accepts(auto_down));
   nv.cold_nvfp4_down_n128_available_=true;assert(accepts(nm));
   {
    auto combined=nm;
    const auto r16=nm.routes[6],r26=nm.routes.back();
    nv.cold_nvfp4_down_nextnorm_available_=true;assert(!accepts(nm));
    auto recipe=nv.down_nextnorm_arguments(240);recipe[0]=2;
    recipe.insert(recipe.end(),{320,64,128,2,8,256,2048,2,3,20,21,24,27,32,32,50,128});
    assert(nv.down_nvfp4_nextnorm_arguments(240)==recipe);
    combined.routes.erase(combined.routes.begin()+6);
    combined.routes.back()={FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_DOWN_NVFP4_NEXTNORM_SITE,1,0,0,recipe};
    assert(accepts(combined));
    nv.cold_nvfp4_down_n128_available_=false;assert(accepts(combined));
    nv.cold_nvfp4_down_n128_available_=true;assert(accepts(combined));
    const auto abi2=combined;
    nv.cold_nvfp4_down_nextnorm_n128_available_=true;assert(!accepts(abi2));
    recipe[0]=3;recipe[23]=176;recipe[24]=128;recipe[26]=1;
    assert(recipe.size()==40 && nv.down_nvfp4_nextnorm_arguments(240)==recipe);
    combined.routes.back().arguments=recipe;assert(accepts(combined));
    auto stale=abi2;stale.routes.back().arguments[0]=3;assert(!accepts(stale));
    nv.cold_nvfp4_down_nextnorm_n128_available_=false;assert(accepts(abi2) && !accepts(combined));
    nv.cold_nvfp4_down_nextnorm_n128_available_=true;assert(accepts(combined));
    for(const auto& old:{r16,r26,old_down}){
     auto bad=combined;bad.routes.push_back(old);assert(!accepts(bad));
     bad=combined;bad.routes.insert(bad.routes.begin(),old);assert(!accepts(bad));
    }
    auto bad=combined;bad.routes.pop_back();assert(!accepts(bad));
    bad=combined;bad.routes.push_back(bad.routes.back());assert(!accepts(bad));
    for(size_t i=0;i<recipe.size();++i){bad=combined;++bad.routes.back().arguments[i];assert(!accepts(bad));}
    for(int i=0;i<4;++i){bad=combined;auto& route=bad.routes.back();
     if(i==0)route.family=FmbRouteFamily::LINEAR;if(i==1)++route.selector;
     if(i==2)++route.flags;if(i==3)++route.invocation;assert(!accepts(bad));}
    bad=combined;++bad.physical_length;assert(!accepts(bad));
    nv.cold_down_nextnorm_available_=false;assert(!accepts(combined));nv.cold_down_nextnorm_available_=true;
    nv.cold_nvfp4_down_nextnorm_available_=false;assert(!accepts(combined) && accepts(nm));
    nv.cold_nvfp4_down_nextnorm_n128_available_=false;
   }
  }
  for(size_t j=1;j<nm.routes.size();++j){
   auto bad=nm;bad.routes.erase(bad.routes.begin()+j);assert(!accepts(bad));
   bad=nm;bad.routes.push_back(bad.routes[j]);assert(!accepts(bad));
   for(size_t a=0;a<nm.routes[j].arguments.size();++a){bad=nm;++bad.routes[j].arguments[a];assert(!accepts(bad));}
   bad=nm;++bad.routes[j].selector;assert(!accepts(bad));
   bad=nm;++bad.routes[j].flags;assert(!accepts(bad));
   bad=nm;bad.routes[j].family=FmbRouteFamily::KV_INSERT;assert(!accepts(bad));
   bad=nm;++bad.routes[j].invocation;assert(!accepts(bad));
  }
  for(int role:{-1,7}){auto bad=nm;bad.routes.back().invocation=role;assert(!accepts(bad));}
  {auto bad=nm;bad.routes[1].arguments[0]=2;assert(!accepts(bad));}
  {auto bad=nm;bad.routes[2].arguments[0]=6;assert(!accepts(bad));}
  nv.nvfp4_=false;assert(!accepts(nm));
 }
 // Equal-capacity GU6+O12 requires its own count/offset/storage receipt.
 {auto old=manifest;m.weight_bytes=262144;m.resident_layers=6;m.o_resident_layers_=12;
  m.cold_w8_o_resident_available_=true;m.cold_gateup_swiglu_available_=true;
  manifest.routes[1].arguments=m.gateup_resident_arguments();
  manifest.routes[2].arguments={6,3145728,6291456,8,0};
  manifest.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_GATEUP_SWIGLU_SITE,1,0,0,m.gateup_swiglu_arguments(240)});
  assert(!admit(manifest));
  manifest.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_O_RESIDENT_SITE,1,0,0,m.o_resident_arguments(240)});
  assert(admit(manifest));
  for(size_t i=0;i<manifest.routes.back().arguments.size();++i){auto bad=manifest;++bad.routes.back().arguments[i];assert(!admit(bad));}
  for(int i=0;i<4;++i){auto bad=manifest;auto& route=bad.routes.back();
   if(i==0)route.family=FmbRouteFamily::KV_INSERT;if(i==1)++route.selector;
   if(i==2)++route.flags;if(i==3)++route.invocation;assert(!admit(bad));}
  {auto bad=manifest;bad.routes.push_back(bad.routes.back());assert(!admit(bad));}
  {auto bad=manifest;bad.routes[2].arguments[0]=12;assert(!admit(bad));}
  m.cold_w8_o_resident_available_=false;assert(!admit(manifest));
  m.o_resident_layers_=0;m.cold_gateup_swiglu_available_=false;m.weight_bytes=524288;manifest=old;
 }
 // Same total bytes as FP16-six must not admit a stale count/precision recipe.
 {auto old=manifest;m.weight_bytes=262144;m.resident_layers=12;
  assert(!admit(old));manifest.routes[1].arguments=m.gateup_resident_arguments();
  manifest.routes[2].arguments={12,3145728,6291456,8,0};assert(admit(manifest));
  auto stale=manifest;stale.routes[2].arguments[0]=6;assert(!admit(stale));
  m.cold_gateup_swiglu_available_=true;assert(!admit(manifest));
  manifest.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_GATEUP_SWIGLU_SITE,1,0,0,m.gateup_swiglu_arguments(240)});
  assert(admit(manifest));stale=manifest;stale.routes.back().arguments[2]=6;stale.routes.back().arguments[3]=26;
  assert(!admit(stale));m.cold_gateup_swiglu_available_=false;m.weight_bytes=524288;m.resident_layers=6;manifest=old;
 }
 for(size_t r:{1,2}){
  auto v=manifest;v.routes.erase(v.routes.begin()+r);assert(!admit(v));
  v=manifest;v.routes.push_back(v.routes[r]);assert(!admit(v));
  v=manifest;v.routes[r].family=FmbRouteFamily::KV_INSERT;assert(!admit(v));
  v=manifest;++v.routes[r].selector;assert(!admit(v));
  v=manifest;++v.routes[r].flags;assert(!admit(v));
  v=manifest;++v.routes[r].invocation;assert(!admit(v));
  for(size_t a=0;a<manifest.routes[r].arguments.size();++a){
   v=manifest;++v.routes[r].arguments[a];assert(!admit(v));}
 }
 auto v=manifest;++v.logical_length;assert(!admit(v));
 v=manifest;++v.kv_logical_length;assert(!admit(v));
 m.profile=false;assert(!admit(manifest));m.profile=true;
 m.gateup_resident_bundle_.present=false;assert(!admit(manifest));
 v=manifest;v.routes.resize(1);assert(admit(v));
 m.gateup_resident_bundle_.present=true;assert(!admit(v));
 m.cold_gateup_swiglu_available_=true;assert(!admit(manifest));
 manifest.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_GATEUP_SWIGLU_SITE,1,0,0,m.gateup_swiglu_arguments(240)});
 assert(admit(manifest));
 for(size_t a=0;a<manifest.routes.back().arguments.size();++a){
  v=manifest;++v.routes.back().arguments[a];assert(!admit(v));}
 v=manifest;v.routes.push_back(v.routes.back());assert(!admit(v));
 v=manifest;v.routes.back().family=FmbRouteFamily::KV_INSERT;assert(!admit(v));
 v=manifest;++v.routes.back().flags;assert(!admit(v));
 v=manifest;++v.routes.back().selector;assert(!admit(v));
 v=manifest;++v.routes.back().invocation;assert(!admit(v));
 for(auto site:{HYVLA_EXPERT_GATE_LINEAR_SITE,HYVLA_EXPERT_UP_LINEAR_SITE}){
  v=manifest;v.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,site,1,0,0,{}});assert(!admit(v));}
 for(int* dim:{&manifest.logical_length,&manifest.physical_length,&manifest.kv_logical_length}){
  ++*dim;assert(!admit(manifest));--*dim;}
 m.cold_gateup_swiglu_available_=false;assert(!admit(manifest));
 m.cold_gateup_swiglu_available_=true;m.cold_resident_swiglu_available_=true;
 assert(!admit(manifest));
 manifest.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_RESIDENT_SWIGLU_SITE,1,0,0,m.resident_swiglu_arguments(240)});
 assert(admit(manifest));
 {auto bad=manifest;bad.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_NVFP4_RESIDENT_SWIGLU_SITE,1,0,0,{}});
  assert(!admit(bad));m.cold_nvfp4_resident_swiglu_available_=true;assert(!admit(bad) && admit(manifest));
  m.cold_nvfp4_resident_swiglu_available_=false;}
 for(size_t a=0;a<manifest.routes.back().arguments.size();++a){v=manifest;++v.routes.back().arguments[a];assert(!admit(v));}
 v=manifest;v.routes.push_back(v.routes.back());assert(!admit(v));
 v=manifest;v.routes.back().family=FmbRouteFamily::KV_INSERT;assert(!admit(v));
 v=manifest;++v.routes.back().flags;assert(!admit(v));
 v=manifest;++v.routes.back().selector;assert(!admit(v));
 v=manifest;++v.routes.back().invocation;assert(!admit(v));
 for(auto site:{HYVLA_EXPERT_MLP_SILU_MUL_SITE,HYVLA_EXPERT_MLP_SILU_SITE,HYVLA_EXPERT_MLP_MUL_SITE}){
  v=manifest;v.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,site,1,0,0,{}});assert(!admit(v));}
 m.cold_resident_swiglu_available_=false;assert(!admit(manifest));manifest.routes.pop_back();assert(admit(manifest));
 m.cold_ring_postnorm_available_=true;assert(!admit(manifest));
 manifest.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_RING_POSTNORM_SITE,1,0,0,m.ring_postnorm_arguments(240)});
 assert(admit(manifest));
 for(size_t a=0;a<manifest.routes.back().arguments.size();++a){v=manifest;++v.routes.back().arguments[a];assert(!admit(v));}
 v=manifest;v.routes.push_back(v.routes.back());assert(!admit(v));
 v=manifest;v.routes.back().family=FmbRouteFamily::KV_INSERT;assert(!admit(v));
 v=manifest;++v.routes.back().flags;assert(!admit(v));
 v=manifest;++v.routes.back().selector;assert(!admit(v));
 v=manifest;++v.routes.back().invocation;assert(!admit(v));
 v=manifest;++v.physical_length;assert(!admit(v));
 for(auto site:{HYVLA_EXPERT_ATTN_ALL_REDUCE_SITE,HYVLA_EXPERT_POST_NORM_SITE}){
  v=manifest;v.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,site,1,0,0,{}});assert(!admit(v));}
 m.cold_ring_postnorm_available_=false;assert(!admit(manifest));m.cold_ring_postnorm_available_=true;
 m.cold_o_ring_postnorm_available_=true;assert(!admit(manifest));manifest.routes.pop_back();assert(!admit(manifest));
 manifest.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_O_RING_POSTNORM_SITE,1,0,0,m.o_ring_postnorm_arguments(240)});
 assert(admit(manifest));
 for(size_t a=0;a<manifest.routes.back().arguments.size();++a){v=manifest;++v.routes.back().arguments[a];assert(!admit(v));}
 v=manifest;v.routes.push_back(v.routes.back());assert(!admit(v));
 v=manifest;v.routes.back().family=FmbRouteFamily::KV_INSERT;assert(!admit(v));
 v=manifest;++v.routes.back().flags;assert(!admit(v));
 v=manifest;++v.routes.back().selector;assert(!admit(v));
 v=manifest;++v.routes.back().invocation;assert(!admit(v));
 v=manifest;++v.physical_length;assert(!admit(v));
 for(auto site:{HYVLA_EXPERT_O_LINEAR_SITE,HYVLA_EXPERT_RING_POSTNORM_SITE,HYVLA_EXPERT_ATTN_ALL_REDUCE_SITE,HYVLA_EXPERT_POST_NORM_SITE}){
  v=manifest;v.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,site,1,0,0,{}});assert(!admit(v));}
 m.cold_o_ring_postnorm_available_=false;assert(!admit(manifest));
 manifest.routes.pop_back();
 manifest.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_RING_POSTNORM_SITE,1,0,0,m.ring_postnorm_arguments(240)});
 assert(admit(manifest));m.cold_qk_rope_available_=true;assert(!admit(manifest));
 manifest.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_QK_ROPE_SITE,1,0,0,m.qk_rope_arguments(240)});
 assert(admit(manifest));
 for(size_t i=0;i<manifest.routes.back().arguments.size();++i){auto bad=manifest;++bad.routes.back().arguments[i];assert(!admit(bad));}
 for(int i=0;i<4;++i){auto bad=manifest;auto& route=bad.routes.back();
  if(i==0)route.family=FmbRouteFamily::KV_INSERT;if(i==1)++route.selector;
  if(i==2)++route.flags;if(i==3)++route.invocation;assert(!admit(bad));}
 {auto bad=manifest;bad.routes.push_back(bad.routes.back());assert(!admit(bad));}
 for(auto site:{HYVLA_EXPERT_Q_PARTIAL_ROPE_SITE,HYVLA_EXPERT_K_PARTIAL_ROPE_SITE,HYVLA_EXPERT_Q_ROPE_SITE,HYVLA_EXPERT_K_ROPE_SITE}){
  auto bad=manifest;bad.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,site,1,0,0,{}});assert(!admit(bad));}
 m.cold_partial_rope_=false;assert(!admit(manifest));m.cold_partial_rope_=true;
 m.cold_qk_rope_available_=false;assert(!admit(manifest));m.cold_qk_rope_available_=true;
 assert(admit(manifest));m.cold_down_nextnorm_available_=true;assert(!admit(manifest));
 manifest.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,HYVLA_EXPERT_DOWN_NEXTNORM_SITE,1,0,0,m.down_nextnorm_arguments(240)});
 assert(admit(manifest));
 for(size_t i=0;i<manifest.routes.back().arguments.size();++i){auto bad=manifest;++bad.routes.back().arguments[i];assert(!admit(bad));}
 for(int i=0;i<4;++i){auto bad=manifest;auto& route=bad.routes.back();
  if(i==0)route.family=FmbRouteFamily::KV_INSERT;if(i==1)++route.selector;
  if(i==2)++route.flags;if(i==3)++route.invocation;assert(!admit(bad));}
 {auto bad=manifest;bad.routes.push_back(bad.routes.back());assert(!admit(bad));}
 for(auto site:{HYVLA_EXPERT_MLP_ALL_REDUCE_SITE,HYVLA_EXPERT_FINAL_NORM_SITE}){
  auto bad=manifest;bad.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,site,1,0,0,{}});assert(!admit(bad));}
 m.cold_down_nextnorm_available_=false;assert(!admit(manifest));
}
'''
    compile_run(tmp_path, "resident_manifest", source.replace("METHOD", method.replace("HyVlaExpertModel::", "")))


def test_expert_cold_marker_uses_lazy_lookup_and_cannot_change_after_binding(tmp_path):
    method = block((ROOT / 'src/fused/rpu_hyvla_expert_model.cpp').read_text(),
                   'void HyVlaExpertModel::set_runtime_config(').replace('HyVlaExpertModel::', '')
    source = COMMON + r'''
constexpr uint32_t RPU_RMSNORM_CAP_BASE=1,RPU_RMSNORM_CAP_V16=4;
struct RpuRmsNormSpmContract {uint32_t capability;bool v16_bitcopy=false;};
RpuRmsNormSpmContract rpu_snapshot_rmsnorm_spm_contract(uint32_t c){return {c};}
struct KernelCache {
 bool marker=true,payload=true,loaded=false,nv4_n128=false,nv4_input_reuse=false,nv4_down=false,nv4_down_norm=false,nv4_down_norm_n128=false,qk_norm=false,qk_norm_kv=false,nv4_resident=false;int calls=0;bool q_n32=false,qkv_overlap=false,sdpa_m16=false,sdpa_k128=false;
 bool o_overlap=false,down_overlap=false,gateup_overlap=false,resident_overlap=false;
 static KernelCache& instance(){static KernelCache c;return c;}
 void* get_kernel(const std::string& name){
  if(name=="hyvla_expert_o_ring_postnorm_nvfp4_s51_n128_overlap")return o_overlap?this:nullptr;
  if(name=="hyvla_expert_down_nextnorm_nvfp4_s51_overlap")return down_overlap?this:nullptr;
  if(name=="hyvla_nvfp4_gateup_swiglu_s51_overlap")return gateup_overlap?this:nullptr;
  if(name=="hyvla_nvfp4_resident_swiglu_s51_overlap")return resident_overlap?this:nullptr;
  if(name=="hyvla_expert_kv_fp16_s51_v2")return marker?this:nullptr;
  if(name=="hyvla_expert_kv_w8_s51_v2")return payload?this:nullptr;
  if(name=="hyvla_expert_kv_nvfp4_s51")return payload?this:nullptr;
  if(name=="hyvla_nvfp4_resident_swiglu_s51")return nv4_resident?this:nullptr;
  if(name=="hyvla_nvfp4_resident_scales_s51")return marker?this:nullptr;
  if(name=="hyvla_nvfp4_resident_scales_swiglu_s51")return payload?this:nullptr;
  if(name=="hyvla_expert_qkv_nvfp4_s51")return marker?this:nullptr;
  if(name=="hyvla_expert_q_nvfp4_s51_n32")return q_n32?this:nullptr;
  if(name=="hyvla_expert_qkv_nvfp4_s51_overlap")return qkv_overlap?this:nullptr;
  if(name=="hyvla_expert_sdpa_k128_s51")return sdpa_k128?this:nullptr;
  if(name=="hyvla_expert_sdpa_m16_s51")return sdpa_m16?this:nullptr;
  if(name=="hyvla_expert_qk_rope_norm_s51")return qk_norm?this:nullptr;
  if(name=="hyvla_expert_qk_rope_norm_kv_s51")return qk_norm_kv?this:nullptr;
  if(name=="hyvla_expert_o_ring_postnorm_nvfp4_s51_n128")return nv4_n128?this:nullptr;
  if(name=="hyvla_nvfp4_gateup_swiglu_input_reuse_s51")return nv4_input_reuse?this:nullptr;
  if(name=="hyvla_expert_down_nvfp4_s51_n128")return nv4_down?this:nullptr;
  if(name=="hyvla_expert_down_nextnorm_nvfp4_s51")return nv4_down_norm?this:nullptr;
  if(name=="hyvla_expert_down_nextnorm_nvfp4_s51_n128")return nv4_down_norm_n128?this:nullptr;
  if(name=="hyvla_nvfp4_gateup_swiglu_s51" || name=="hyvla_nvfp4_gateup_resident_s51" || name=="hyvla_expert_o_ring_postnorm_nvfp4_s51" || name=="hyvla_expert_o_resident_w8_v2" || name=="hyvla_expert_o_ring_postnorm_w8_s51_v2" || name=="hyvla_expert_xor3_norm_s51" || name=="hyvla_resident_swiglu_w8_v2" || name=="hyvla_gateup_resident_pair_w8_v2" || name=="hyvla_expert_gateup_swiglu_w8_v2")return nullptr;
  if(name=="hyvla_resident_swiglu_fp16_v2" || name=="hyvla_expert_ring_postnorm_s51" || name=="hyvla_expert_o_ring_postnorm_fp16_s51_v2" || name=="hyvla_expert_qk_rope_s51" || name=="hyvla_expert_down_nextnorm_s51")return payload?this:nullptr;
  assert(name=="hyvla_expert_gateup_swiglu_fp16_v2");++calls;loaded=marker&&payload;return loaded?this:nullptr;}
 void* get_program(const std::string& name){
  if(name=="hyvla_expert_gateup_swiglu_fp16_v2")return loaded?this:nullptr;
  assert(name=="wall_prefill_gate_up_swiglu_fp16_m320n112k128");return payload?this:nullptr;}
};
struct Model {
 bool cold_w8_gateup_n48_available_=false;bool cold_w8_optimizations_available_=false;
 bool cold_config_bound_=false,production_config_bound_=false,cold_gateup_swiglu_available_=false;
 bool cold_w8_resident_swiglu_available_=false,cold_w8_o_ring_postnorm_available_=false,cold_xor3_norm_available_=false;
 bool cold_nvfp4_resident_swiglu_available_=false;
 bool cold_nvfp4_resident_scales_available_=false,cold_nvfp4_resident_scales_swiglu_available_=false;
 int64_t resident_fp8_scale_layers_=0;
 bool cold_w8_o_resident_available_=false;int64_t o_resident_layers_=0;
 bool nvfp4_=false,cold_nvfp4_gateup_available_=false,cold_nvfp4_o_ring_postnorm_available_=false,cold_nvfp4_o_n128_available_=false;
 bool cold_nvfp4_gateup_input_reuse_available_=false;
 bool cold_nvfp4_down_n128_available_=false;
 bool cold_nvfp4_down_nextnorm_available_=false;
 bool cold_nvfp4_down_nextnorm_n128_available_=false;
 bool cold_resident_swiglu_available_=false,cold_ring_postnorm_available_=false,cold_o_ring_postnorm_available_=false,cold_down_nextnorm_available_=false,cold_qk_rope_available_=false;
 bool cold_qk_rope_norm_available_=false,cold_qk_rope_norm_kv_available_=false;
 bool cold_fp16_kv_grouped_available_=false,cold_w8_kv_grouped_available_=false;
 bool cold_nvfp4_kv_available_=false,cold_nvfp4_qkv_available_=false;
 bool cold_nvfp4_q_n32_available_=false,cold_nvfp4_qkv_overlap_available_=false;
 bool cold_nvfp4_o_overlap_available_=false,cold_nvfp4_down_overlap_available_=false;
 bool cold_nvfp4_gateup_overlap_available_=false,cold_nvfp4_resident_overlap_available_=false;
 bool cold_sdpa_m16_available_=false,cold_sdpa_k128_available_=false;
 bool cold_fast_replay_=false,cold_fast_replay_preload_=false,cold_mask_once_=false;
 bool cold_silu_mul_=false,cold_kvpad16_=false,cold_partial_rope_=false,cold_rmsnorm_pad16_=false;
 bool cold_paired_kv_available_=false;RpuRmsNormSpmContract cold_rmsnorm_spm_contract_{};
 int invalidations=0;void invalidate_model_state(){++invalidations;}
 METHOD
};
int main(){
 for(bool marker:{false,true})for(bool payload:{false,true})for(bool n128:{false,true}){
  auto& c=KernelCache::instance();c={marker,payload,false,n128,!n128,marker!=n128,payload!=n128,marker!=payload,n128!=marker,n128!=payload,(n128!=marker)!=payload,0};c.q_n32=marker!=n128;c.qkv_overlap=payload!=n128;c.sdpa_m16=marker!=payload;c.sdpa_k128=payload!=n128;c.o_overlap=marker;c.down_overlap=payload;c.gateup_overlap=n128;c.resident_overlap=marker!=payload;Model m;
  assert(!c.get_program("hyvla_expert_gateup_swiglu_fp16_v2"));
  m.set_runtime_config(true,true,true,true,false,true,false);
  assert(m.cold_gateup_swiglu_available_==(marker&&payload) && c.calls==1);
  assert(m.cold_nvfp4_o_n128_available_==n128);
  assert(m.cold_nvfp4_gateup_input_reuse_available_==false);
  assert(m.cold_nvfp4_down_n128_available_==false);
  assert(m.cold_nvfp4_down_nextnorm_available_==(payload!=n128));
  assert(m.cold_nvfp4_down_nextnorm_n128_available_==false);
  assert(m.cold_qk_rope_norm_available_==(n128!=marker));
  assert(m.cold_qk_rope_norm_kv_available_==(n128!=payload));
  assert(m.cold_nvfp4_resident_swiglu_available_==((n128!=marker)!=payload));
  assert(m.cold_fp16_kv_grouped_available_==marker && m.cold_w8_kv_grouped_available_==payload);
  assert(m.cold_nvfp4_kv_available_==payload && m.cold_nvfp4_qkv_available_==marker);
  assert(m.cold_nvfp4_resident_scales_available_==marker && m.cold_nvfp4_resident_scales_swiglu_available_==payload);
  assert(m.cold_nvfp4_q_n32_available_==false);
  assert(m.cold_nvfp4_qkv_overlap_available_==(payload!=n128));
  assert(m.cold_sdpa_m16_available_==false);
  assert(m.cold_sdpa_k128_available_==false);
  assert(m.cold_nvfp4_o_overlap_available_==false && m.cold_nvfp4_down_overlap_available_==false);
  assert(m.cold_nvfp4_gateup_overlap_available_==false && m.cold_nvfp4_resident_overlap_available_==false);
  assert(!m.cold_nvfp4_gateup_available_);
  assert(m.cold_config_bound_ && m.invalidations==1 && m.cold_resident_swiglu_available_==payload);
  c.marker=!marker;c.payload=!payload;c.nv4_n128=!n128;c.nv4_input_reuse=n128;c.nv4_down=marker==n128;c.nv4_down_norm=payload==n128;c.nv4_down_norm_n128=marker==payload;c.qk_norm=n128==marker;c.qk_norm_kv=n128==payload;c.nv4_resident=(n128!=marker)==payload;c.q_n32=marker==n128;c.qkv_overlap=payload==n128;c.sdpa_m16=marker==payload;c.sdpa_k128=payload==n128;c.o_overlap=!marker;c.down_overlap=!payload;c.gateup_overlap=!n128;c.resident_overlap=marker==payload;bool rejected=false;
  try{m.set_runtime_config(true,true,true,true,false,true,false);}catch(const std::exception&){rejected=true;}
  assert(rejected && c.calls==1 && m.cold_gateup_swiglu_available_==(marker&&payload));
  assert(m.cold_nvfp4_o_n128_available_==n128);
  assert(m.cold_nvfp4_gateup_input_reuse_available_==false);
  assert(m.cold_nvfp4_down_n128_available_==false);
  assert(m.cold_nvfp4_down_nextnorm_available_==(payload!=n128));
  assert(m.cold_nvfp4_down_nextnorm_n128_available_==false);
  assert(m.cold_qk_rope_norm_available_==(n128!=marker));
  assert(m.cold_qk_rope_norm_kv_available_==(n128!=payload));
  assert(m.cold_nvfp4_resident_swiglu_available_==((n128!=marker)!=payload));
  assert(m.cold_fp16_kv_grouped_available_==marker && m.cold_w8_kv_grouped_available_==payload);
  assert(m.cold_nvfp4_kv_available_==payload && m.cold_nvfp4_qkv_available_==marker);
  assert(m.cold_nvfp4_resident_scales_available_==marker && m.cold_nvfp4_resident_scales_swiglu_available_==payload);
  assert(m.cold_nvfp4_q_n32_available_==false);
  assert(m.cold_nvfp4_qkv_overlap_available_==(payload!=n128));
  assert(m.cold_sdpa_m16_available_==false);
  assert(m.cold_sdpa_k128_available_==false);
  assert(m.cold_nvfp4_o_overlap_available_==false && m.cold_nvfp4_down_overlap_available_==false);
  assert(m.cold_nvfp4_gateup_overlap_available_==false && m.cold_nvfp4_resident_overlap_available_==false);
  assert(m.cold_resident_swiglu_available_==payload && m.cold_ring_postnorm_available_==payload && m.cold_o_ring_postnorm_available_==payload && m.cold_qk_rope_available_==payload && m.cold_down_nextnorm_available_==payload);
 }
}
'''
    compile_run(tmp_path, 'expert_gateup_cold', source.replace('METHOD', method))


def test_w8_installed_packing_preserves_quantized_tiles_and_opaque_dma_bits():
    plan = runtime.prepare_hy_vla_cold_plan({KEY: "1", "RPU_HY_VLA_ATTN_TP8": "1",
                                            "RPU_HY_VLA_W8A16": "all"})
    assert plan.weights.expert_bits == 8 and plan.weights.gateup_resident
    # All byte values, including pairs that are NaN/Inf if viewed as FP16.
    installed = torch.arange(2048*1024, dtype=torch.int32).to(torch.int8).reshape(2048,1024)
    compact = weights._pack_expert_gateup_w8_installed(installed)
    carrier = compact.view(torch.float16).half().clone()
    assert torch.equal(carrier.view(torch.int8), compact)
    ddr = installed.reshape(16,32,8,512)
    for core in range(8):
        assert torch.equal(compact[core], ddr[:,:,core].reshape(-1))
    with pytest.raises(ValueError, match="INT8"):
        weights._pack_expert_gateup_w8_installed(installed.half())


@pytest.mark.parametrize("suffix", ["", "_v"])
def test_w8_mot_loader_keeps_branch_scales_without_resident_expert_packing(monkeypatch, suffix):
    class Source:
        def __getitem__(self, key): return torch.ones(1)
    calls = []
    def quant(raw, p, nc, bits):
        calls.append(bits)
        return torch.zeros(1, dtype=torch.int8), torch.ones(1, dtype=torch.float16)
    monkeypatch.setattr(weights, "_quant", quant)
    monkeypatch.setattr(weights, "_dup_kv", lambda raw, *args, **kwargs: raw)
    monkeypatch.setattr(weights, "_rpu16", lambda raw: raw.half())
    cold = weights._HyVlaWeightColdPlan.from_environment({"RPU_HY_VLA_W8A16": "all"})
    result = weights._build_mot_branch(Source(), weights.HyVlaConfig(), suffix, cold)
    assert calls == [8] * (32 * 7)
    assert all(len(v) == 32 for v in result.values())
    assert all(t.dtype == torch.float16 for key,v in result.items() if key.endswith("_ws") for t in v)


def test_mixed_residency_option_is_cold_strict_and_w8_only(monkeypatch):
    _clear_hyvla_cold_environment(monkeypatch)
    o_key = "RPU_HY_VLA_EXPERT_O_RESIDENT_LAYERS"
    env = {KEY: "1", LAYERS_KEY: "6", o_key: "12", "RPU_HY_VLA_W8A16": "all", "RPU_HY_VLA_ATTN_TP8": "1"}
    plan = runtime.prepare_hy_vla_cold_plan(env)
    env[o_key] = "0"
    assert plan.weights.o_resident_layers == 12 and plan.weights.gateup_resident_layers == 6
    assert runtime.prepare_hy_vla_cold_plan(env).weights.o_resident_layers == 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.weights.o_resident_layers = 0
    for value in ("1", "6", "11", "13", "012", "12.0", "true"):
        with pytest.raises(ValueError, match=o_key):
            runtime.prepare_hy_vla_cold_plan(env | {o_key: value})
    for override in ({KEY: "0"}, {LAYERS_KEY: "12"}, {"RPU_HY_VLA_W8A16": "0"}, {"RPU_HY_VLA_W4A16": "expert"}):
        with pytest.raises(ValueError, match=o_key):
            runtime.prepare_hy_vla_cold_plan(env | {o_key: "12"} | override)


def test_mixed_resident_builder_packs_exact_six_gateup_and_twelve_o(monkeypatch):
    from rpu_backend.adapters.hy_vla.resident_o import pack_expert_o_w8_installed
    class Source:
        def __getitem__(self, key):
            li = int(key.split(".")[1])
            tag = 3 * li + (1 if "gate_proj" in key else 2 if "up_proj" in key else 3)
            return torch.tensor([tag if "gate_proj" in key or "up_proj" in key or "o_proj_v" in key else 0])
    base = torch.arange(2048*1024, dtype=torch.int32)
    def quant(raw, partition, cores, bits):
        assert bits == 8
        q = (base + int(raw.item())).to(torch.int8).reshape((1024,2048) if partition == 0 else (2048,1024)) if raw.item() else torch.zeros(1,dtype=torch.int8)
        return q, torch.ones(1,dtype=torch.float16)
    monkeypatch.setattr(weights, "_quant", quant)
    monkeypatch.setattr(weights, "_dup_kv", lambda raw,*a,**kw:raw)
    monkeypatch.setattr(weights, "_rpu16", lambda raw:raw.half())
    cold=weights._HyVlaWeightColdPlan.from_environment({KEY:"1",LAYERS_KEY:"6","RPU_HY_VLA_EXPERT_O_RESIDENT_LAYERS":"12","RPU_HY_VLA_W8A16":"all","RPU_HY_VLA_ATTN_TP8":"1"})
    result=weights._build_expert(Source(),Source(),weights.HyVlaConfig(),cold)
    bundle,=result["gateup_resident"]
    assert bundle.shape==(8,3145728) and bundle.dtype==torch.float16
    raw=bundle.view(torch.int8)
    gu=raw[:,:3145728].reshape(8,6,2,262144)
    out=raw[:,3145728:].reshape(8,12,262144)
    for li in range(6):
        for j,key in enumerate(("gate_w","up_w")):
            assert torch.equal(gu[:,li,j],weights._pack_expert_gateup_w8_installed(result[key][li]))
    for li in range(12):
        assert torch.equal(out[:,li],pack_expert_o_w8_installed(result["o_w"][li]))


def test_actual_m16_attention_baseline_declares_complete_scratch(tmp_path):
    decoder = (ROOT / 'src/fused/rpu_hyvla_expert_model.cpp').read_text()
    methods = '\n'.join(block(decoder, marker).replace('HyVlaExpertModel::', '') for marker in (
        'bool HyVlaExpertModel::sdpa_k128_enabled(',
        'bool HyVlaExpertModel::sdpa_m16_enabled(',
        'std::vector<BufferDecl> HyVlaExpertModel::baseline_buffer_declarations(',
    )).replace('const_cast<HyVlaExpertModel*>', 'const_cast<Model*>')
    grow = block(decoder, 'static bool hyvla_grow_slot(')
    source = COMMON + r'''
#include <cstring>
int64_t Align(int64_t n,int64_t a){return ((n+a-1)/a)*a;}
namespace v3 {
enum class StorageClass {Temp};enum class BufferScope {LayerWide};
struct BufferDecl {const char* name=nullptr;int64_t size=0;int phase_start=0,phase_end=0;
 StorageClass storage=StorageClass::Temp;int per_layer=0;const char* alias_of=nullptr;
 BufferScope scope=BufferScope::LayerWide;};
}
using namespace v3;
enum class AttentionExecutionPolicy {DDR_KV,SPM_KV_BY_MHA};
struct LayoutContext {int64_t chunk_size=51,max_kv_seq_len=291,batch_size=1;
 bool use_attn_mask=true;AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::DDR_KV;};
constexpr int64_t HYVLA_S51_M16_SDPA_TMP_BYTES=262144,HYVLA_S51_K128_SDPA_TMP_BYTES=32768;
GROW
void hyvla_widen_sdpa_mask_slot(std::vector<BufferDecl>&){assert(false);}
void hyvla_grow_kv_slots(std::vector<BufferDecl>&,int64_t,int64_t){assert(false);}
struct SpmAllocator {static constexpr int ALIGN=256;};
struct Model {
 bool nvfp4_=true,cold_sdpa_m16_available_=false,cold_sdpa_k128_available_=false,profile=true;
 bool cold_mask_once_=false,cold_kvpad16_=false,cold_rmsnorm_pad16_=false,unroll_mode_=true;
 int64_t action_dim_pad_=32,action_mlp_cores_=8,resident_layers=24,o_resident_layers_=0;
 at::Tensor gateup_resident_bundle_;bool missing_tmp=false;int64_t original_tmp=65536,resident_fp8_scale_layers_=0;
 int hidden_size()const{return 1024;}int head_dim()const{return 128;}
 int num_kv_heads()const{return 8;}int num_q_heads()const{return 16;}int attn_tp()const{return 8;}
 bool gateup_resident_profile()const{return profile;}
 bool gateup_swiglu_enabled(int64_t r,int64_t p)const{return profile && gateup_resident_bundle_.defined() && r==51 && (p>=128 && p<=240 && p%16==0);}
 int64_t gateup_resident_layers()const{return resident_layers;}
 int64_t resident_storage_bytes()const{return resident_layers*262144+resident_fp8_scale_layers_*32768;}
 std::vector<BufferDecl> causal_baseline_buffer_declarations(const LayoutContext&){
  std::vector<BufferDecl> d{{"q",28672,2,5},{"output",28672,2,5},{"sdpa_mask",31232,2,5},{"oproj",104448,5,5}};
  if(!missing_tmp)d.push_back({"sdpa_tmp",original_tmp,5,5});return d;}
 METHODS
};
int main(){
 Model m;LayoutContext c;
 const auto size=[](const std::vector<BufferDecl>& d,const char* name){for(const auto& b:d)if(std::strcmp(b.name,name)==0){if(!b.alias_of)return b.size;for(const auto& owner:d)if(std::strcmp(owner.name,b.alias_of)==0)return owner.size;}return int64_t(-1);};
 assert(size(m.baseline_buffer_declarations(c),"sdpa_tmp")==65536);
 m.cold_sdpa_m16_available_=true;
 auto d=m.baseline_buffer_declarations(c);
 assert(size(d,"sdpa_tmp")==262144 && size(d,"gateup_resident_weights")==6291456);
 for(int scale_count:{4,8}){m.resident_fp8_scale_layers_=scale_count;auto ds=m.baseline_buffer_declarations(c);
 assert(size(ds,"gateup_resident_weights")==6291456+scale_count*32768 && size(ds,"sdpa_tmp")==262144 && m.resident_layers==24);
 for(const auto& b:ds)if(std::strcmp(b.name,"gateup_resident_weights")==0)assert(b.phase_start==0 && b.phase_end==9 && !b.alias_of);}
 m.resident_fp8_scale_layers_=0;
 for(int position:{128,144,160,176,192,208,224,240}){auto short_layout=c;short_layout.max_kv_seq_len=position+51;
  assert(size(m.baseline_buffer_declarations(short_layout),"sdpa_tmp")==262144);}
 assert(size(d,"q")==28672 && size(d,"output")==28672 && size(d,"sdpa_mask")==31232);
 for(const auto& b:d)if(std::strcmp(b.name,"sdpa_tmp")==0)assert(b.phase_start==5 && b.phase_end==5 && !b.alias_of);
 m.original_tmp=524288;assert(size(m.baseline_buffer_declarations(c),"sdpa_tmp")==524288);m.original_tmp=65536;
 for(int field=0;field<3;++field){auto bad=c;if(field==0)bad.use_attn_mask=false;
  if(field==1)bad.batch_size=2;if(field==2)bad.attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;
  bool rejected=false;try{m.baseline_buffer_declarations(bad);}catch(const std::exception&){rejected=true;}assert(rejected);}
 m.missing_tmp=true;bool rejected=false;try{m.baseline_buffer_declarations(c);}catch(const std::exception&){rejected=true;}assert(rejected);m.missing_tmp=false;
 for(int field=0;field<4;++field){auto other=c;if(field==0)other.chunk_size=50;if(field==1)other.max_kv_seq_len=163;
  if(field==2)m.nvfp4_=false;if(field==3)m.resident_layers=12;
  assert(size(m.baseline_buffer_declarations(other),"sdpa_tmp")==65536);m.nvfp4_=true;m.resident_layers=24;}
 m.cold_sdpa_m16_available_=false;m.cold_sdpa_k128_available_=true;
 for(int position:{128,144,160,176,192,208,224,240})for(int scale_count:{0,4,8}){
  auto short_layout=c;short_layout.max_kv_seq_len=position+51;m.resident_fp8_scale_layers_=scale_count;
  const auto dk=m.baseline_buffer_declarations(short_layout);
  assert(size(dk,"sdpa_tmp")== (scale_count==8?104448:32768) && size(dk,"gateup_resident_weights")==6291456+scale_count*32768);
  for(const auto& b:dk)if(std::strcmp(b.name,"sdpa_tmp")==0)assert(b.phase_start==5 && b.phase_end==5 && (scale_count==8?(b.alias_of&&std::strcmp(b.alias_of,"oproj")==0):!b.alias_of));
 }
 m.resident_fp8_scale_layers_=0;
 for(int field=0;field<3;++field){auto bad=c;if(field==0)bad.use_attn_mask=false;
  if(field==1)bad.batch_size=2;if(field==2)bad.attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;
  bool rejected=false;try{m.baseline_buffer_declarations(bad);}catch(const std::exception&){rejected=true;}assert(rejected);}
 for(int field=0;field<4;++field){auto other=c;if(field==0)other.chunk_size=50;if(field==1)other.max_kv_seq_len=163;
  if(field==2)m.nvfp4_=false;if(field==3)m.resident_layers=12;
  assert(size(m.baseline_buffer_declarations(other),"sdpa_tmp")==65536);m.nvfp4_=true;m.resident_layers=24;}
 m.original_tmp=32768;assert(size(m.baseline_buffer_declarations(c),"sdpa_tmp")==32768);
 m.original_tmp=32767;rejected=false;try{m.baseline_buffer_declarations(c);}catch(const std::exception&){rejected=true;}assert(rejected);m.original_tmp=65536;
 m.missing_tmp=true;rejected=false;try{m.baseline_buffer_declarations(c);}catch(const std::exception&){rejected=true;}assert(rejected);m.missing_tmp=false;
 m.cold_sdpa_m16_available_=true;rejected=false;try{m.baseline_buffer_declarations(c);}catch(const std::exception&){rejected=true;}assert(rejected);
}
'''
    compile_run(tmp_path, 'm16_baseline', source.replace('GROW', grow).replace('METHODS', methods))
