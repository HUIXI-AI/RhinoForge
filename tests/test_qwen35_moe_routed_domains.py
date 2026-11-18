"""Exercise mixed routed/shared ownership through the real pack and cache paths."""
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import torch

from test_qwen35_moe_rtn_weights import _checkpoint, _reader, _inverse_grouped
from test_grouped_uint4_metadata import inverse_metadata


@pytest.mark.parametrize("profile", ["rtn", "gptq"])
@pytest.mark.parametrize("budget", [4, 6, 8])
def test_real_rtn_and_gptq_install_pack_keeps_routed_logical_width(tmp_path, profile, budget):
    from rpu_backend.adapters.qwen3_5_moe.weights import prepare_moe_layer_weights

    expected = _checkpoint(tmp_path, profile=profile)
    source = _reader(tmp_path, profile).layer_source(0)
    experts = SimpleNamespace(_qwen3_5_moe_int4_source=source,
                              rpu_weight_mode=6, rpu_quant_profile=source.profile_id)
    def linear(n, k):
        return torch.nn.Linear(k, n, bias=False, dtype=torch.float16)
    mlp = SimpleNamespace(experts=experts, gate=linear(256, 128),
        shared_expert=SimpleNamespace(gate_proj=linear(512, 128),
            up_proj=linear(512, 128), down_proj=linear(128, 512)),
        shared_expert_gate=linear(1, 128))
    routed = 4 if budget == 6 else budget
    packed = prepare_moe_layer_weights(mlp, num_cores=budget,
                                      routed_num_cores=routed, device="cpu")
    assert packed.routed_weight_mode == 6
    for role, name, partition, n, k in (
        ("gate_proj", "routed_gate", "col", 512, 128),
        ("up_proj", "routed_up", "col", 512, 128),
        ("down_proj", "routed_down", "row", 128, 512),
    ):
        value = getattr(packed, name)
        assert value.numel() == 2 * n * k // 2
        actual = _inverse_grouped(value, n=n, k=k, partition=partition, cores=routed)
        assert torch.equal(actual, torch.stack([x[0] for x in expected[role]]))
        scales = getattr(packed, name + "_scale")
        reference = torch.stack([x[1] for x in expected[role]])
        if partition == "row":
            reference = reference.repeat_interleave(2, dim=1)
        actual_scale = inverse_metadata(scales, global_q=reference.shape[1],
                                        global_n=n, partition=partition, cores=routed)
        assert torch.equal(actual_scale.view(torch.int16), reference.view(torch.int16))
    width = 576 if budget == 6 else 512
    assert packed.shared_gate.shape == (width, 128)
    assert packed.shared_up.shape == (width, 128)
    assert packed.shared_down.shape == (128, width)
    with pytest.raises(ValueError, match="already consumed"):
        source.prepare(num_cores=routed)


def test_explicit_routed_layout_mismatch_fails_before_consuming_source(monkeypatch):
    from rpu_backend.adapters.qwen3_5_moe import weights

    monkeypatch.setattr(weights, "_prepare_routed", lambda *a: pytest.fail("source consumed"))
    mlp = SimpleNamespace(experts=SimpleNamespace(rpu_weight_mode=6))
    for bad in (6, 8, True, 4.0):
        with pytest.raises(ValueError, match="cold profile"):
            weights.prepare_moe_layer_weights(mlp, num_cores=6, routed_num_cores=bad)


@pytest.mark.parametrize("profile", ["rtn", "gptq"])
def test_real_quant_profile_selects_routed4_and_rejects_hot_reconfigure(monkeypatch, profile):
    from rpu_backend.adapters import qwen3_5_moe as adapter
    from rpu_backend.adapters.qwen3_5_moe.cores import core_topology
    from test_qwen35_moe_python_integration import _exact_config
    from test_qwen35_moe_rtn_weights import _gptq_quant_config

    config = _exact_config()
    config.pop("quant_config")
    config["quantization_config"] = _gptq_quant_config()
    if profile == "rtn":
        config["quantization_config"] = {"bits": 4, "group_size": 128,
            "sym": True, "desc_act": False, "quant_method": "gptq"}
        config["rtn_metadata"] = {"algorithm": "RTN", "manifest": "derivation_manifest.json",
                                  "calibration": False}
    execution = {"model": {"num_cores": 6}, "prefill": {"linear_acc32": False}}
    from transformers import Qwen3_5MoeConfig
    config = Qwen3_5MoeConfig.from_dict(config)
    topology = core_topology(execution, config=config)
    assert (topology.routed_tp, topology.mlp_tp) == (4, 6)
    monkeypatch.setattr(adapter, "_qwen3_5_moe_cold_config_only", lambda _: False)
    instance = SimpleNamespace(model=SimpleNamespace(config=config),
        preflight_execution=adapter.Qwen3_5MoeAdapter.preflight_execution)
    with pytest.raises(ValueError, match="immutable after (RPU|device) installation"):
        adapter._validate_qwen3_5_moe_execution_reconfigure(instance, execution)


@pytest.mark.parametrize("mode", [5, 6])
@pytest.mark.parametrize("budget", [4, 6, 8])
def test_component_identity_and_cache_are_bound_to_installed_quant_mode(monkeypatch, mode, budget):
    from rpu_backend.adapters.qwen3_5_moe import cores, quant_scope, text
    from rpu_backend.adapters.qwen3_5_moe import Qwen3_5MoeAdapter
    from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache
    from test_qwen35_moe_core_profiles import config

    cfg = config()
    monkeypatch.setattr(quant_scope, "validate_exact_profile", lambda *a, **k: {"profile": mode})
    monkeypatch.setattr(quant_scope, "runtime_capabilities",
        lambda profile: SimpleNamespace(allowed_cores=(4, 6, 8), routed_weight_mode=profile))
    execution = {"model": {"num_cores": budget}}
    topology = cores.core_topology(execution, config=cfg)
    narrow = 8 if budget == 8 else 4
    routed = 4 if budget == 6 and mode == 6 else budget
    expected = (2, budget, narrow, narrow, narrow, budget, routed, budget, budget, 8)
    assert topology.identity() == cores.full_topology_tuple(topology) == expected
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_get_execution_topology",
        lambda _: (budget, narrow, budget, budget, 8), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_get_execution_topology_v2",
        lambda _: expected, raising=False)
    state = SimpleNamespace(handle=17, execution_topology=topology,
                            max_seq_len=64, chunk_envelope=(8192, 512))
    model = SimpleNamespace(config=cfg, _rpu_execution=execution, _rpu_qwen3_5_moe=state)
    cache = Qwen3_5MoeCache.from_model(model, 64, device="cpu")
    assert cache.moe_execution_topology == expected
    assert cache.attn_tp == narrow and cache.physical_kv_cores == 8
    monkeypatch.setattr(text, "_qwen35_moe_retirement_ready", lambda value: value is state)
    assert text._preflight_qwen3_5_moe_text(model, cache, 1) == (1, 0, None)
    cache.reset()
    assert cache.moe_execution_topology == expected
    wrong = list(expected)
    wrong[6] = 6 if routed == 4 else 4
    cache.moe_execution_topology = tuple(wrong)
    with pytest.raises(ValueError, match="cache core layout"):
        text._preflight_qwen3_5_moe_text(model, cache, 1)
    # Generic decoder geometry is unchanged by quantization. The complete MoE
    # identity must reject a stale routed owner even when that geometry matches.
    if budget == 6:
        monkeypatch.setattr(quant_scope, "validate_exact_profile",
            lambda *a, **k: {"profile": 5 if mode == 6 else 6})
        monkeypatch.setattr(torch, "zeros", lambda *a, **k: pytest.fail("stale cache allocated"))
        with pytest.raises(ValueError, match="installed core layout"):
            Qwen3_5MoeCache.from_model(model, 64, device="cpu")


def test_real_native_merge_never_reads_unproduced_routed_cores(tmp_path):
    from test_qwen35_moe_core_native import _definition, _run_cpp, NATIVE

    method = _definition(NATIVE.read_text(), "void Qwen3_5MoeModel::emit_moe_merge_and_residual(")
    code = r'''
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
namespace v3 {
enum class FmbRouteFamily { ALL_REDUCE };
constexpr int QWEN35_MOE_ALL_REDUCE_SITE=1;
int owner=6, routed=4, scatter_count=0, reduce_count=0;
std::array<double,8> partial, routed_output;
double result=0;
void rpu_launch_scatternd_add_u16_fp16_spm_kernel(uint32_t,uint32_t,uint32_t,
        int64_t t,int64_t r,int64_t h,int nc) {
    assert(t>0 && r==t*8 && h==2048 && nc==routed);
    scatter_count=nc;
    for(int core=0;core<nc;++core) {
        assert(std::isfinite(routed_output[core]));
        partial[core]+=routed_output[core];
    }
}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t,uint32_t,uint32_t,
        int64_t,int64_t,int producers,int consumers) {
    assert(producers==owner && consumers==owner);
    reduce_count=producers; result=10;
    for(int core=0;core<producers;++core)result+=partial[core];
}
struct Qwen3_5MoeModel {
    struct {int64_t top_k=8;} moe_;
    int hidden_size()const{return 2048;}
    int routing_tp()const{return owner;}
    int routed_expert_tp()const{return routed;}
    int num_cores()const{return owner;}
    uint32_t addr(int,const char*)const{return 0;}
    int text_ring_route(int64_t,int64_t)const{return 2;}
    void consume_manifest_route(FmbRouteFamily,int,int,int64_t){}
    void emit_moe_merge_and_residual(int64_t,int64_t,uint32_t,uint32_t);
};
__METHOD__
}
int main(){
    using namespace v3;
    for(int cores:{4,6,8})for(bool uint4:{false,true})for(int rows:{1,64,128}) {
        owner=cores; routed=uint4 && owner==6?4:owner;
        double expected=10;
        partial.fill(std::numeric_limits<double>::quiet_NaN());
        routed_output.fill(std::numeric_limits<double>::quiet_NaN());
        for(int c=0;c<owner;++c){partial[c]=c+1;expected+=c+1;}
        for(int c=0;c<routed;++c){routed_output[c]=100+c;expected+=100+c;}
        Qwen3_5MoeModel model;model.emit_moe_merge_and_residual(0,rows,0,0);
        assert(result==expected && scatter_count==routed && reduce_count==owner);
        for(int c=routed;c<owner;++c)assert(partial[c]==c+1);
    }
}
'''.replace("__METHOD__", method)
    _run_cpp(tmp_path, code)


def test_real_native_layout_and_cost_identities_bind_each_component(tmp_path):
    from test_qwen35_moe_core_native import _definition, _run_cpp, NATIVE, ROOT

    source = NATIVE.read_text()
    header = (ROOT / "src/fused/rpu_qwen3_5_moe_model.h").read_text()
    base = (ROOT / "src/core/fused_model_base.h").read_text()
    helper_names = ("int gdn_tp", "int router_tp", "int routing_tp", "int routed_expert_tp",
                    "int shared_expert_tp", "int lm_head_tp", "int64_t physical_routed_intermediate",
                    "int64_t physical_shared_intermediate")
    helpers = "\n".join(_definition(header, "    " + name + "() const") for name in helper_names)
    cost = _definition(source, "std::vector<int64_t> Qwen3_5MoeModel::kvinsert_cost_weight_identity() const")
    layout_version = re.search(
        r"constexpr int64_t kResidentProjectionLayoutVersion = \d+;", source).group()
    lw_names = sorted(set(re.findall(r"&lw\.(\w+)", cost)))
    mw_names = sorted(set(re.findall(r"(?:&mw\.|identity, mw\.)(\w+)", cost)))
    methods = "\n".join(_definition(source, marker) for marker in (
        "bool Qwen3_5MoeModel::has_gdn_glast_local_capability() const",
        "std::vector<int64_t> Qwen3_5MoeModel::gdn_glast_local_identity() const",
        "int64_t Qwen3_5MoeModel::minimum_eager_spm_bytes() const",
        "std::vector<int64_t> Qwen3_5MoeModel::cold_topology_arguments() const",
        "std::vector<int64_t> Qwen3_5MoeModel::exact_profile_arguments() const",
        "int64_t Qwen3_5MoeModel::subclass_layout_hash() const",
        "std::vector<int64_t> Qwen3_5MoeModel::kvinsert_cost_weight_identity() const"))
    local_schedule = "\n".join(re.findall(
        r"constexpr int64_t QWEN35_MOE_(?:SHARED_ALPHA_LOCAL_SITE|RESIDENT_SHARED_ALPHA_LOCAL_SITE|GDN_GLAST_LOCAL_SITE)\s*=.*?;", source, re.S))
    local_schedule += "\n" + _definition(source, "std::vector<int64_t> shared_alpha_local_identity(int nc)")
    code = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#include "core/rpu_grouped_linear_tiling.h"
#include "core/rpu_spm_allocator.h"
namespace c10 {struct Half {uint16_t storage;};}
#define Align(x,a) (((x)+(a)-1)/(a)*(a))
#define NUM_CORES 8
namespace v3 {
constexpr int kRoutedUint4Mode=6;
__LOCAL_SCHEDULE__
__LAYOUT_VERSION__
namespace detail { __MIX__ }
struct Tensor {bool defined()const{return true;}int64_t size(int)const{return 128;}};
void append_kvinsert_cost_scalar_identity(std::vector<int64_t>& v,double){v.push_back(0);}
void append_kvinsert_cost_tensor_identity(std::vector<int64_t>& v,const Tensor&){v.push_back(1);}
struct Qwen3_5MoeModel {
    int owner=6;
    struct Moe {int64_t num_experts=256,top_k=8,routed_intermediate=512,shared_intermediate=512,tensor_parallel=6;} moe_;
    struct Layer {Tensor __LW__;};
    struct Expert {Tensor __MW__;int64_t routed_weight_mode=5,routed_group_size=128,routed_down_group_size=64;};
    std::vector<Layer> layer_weights_{Layer{}};
    std::vector<Expert> moe_layer_weights_{Expert{}};
    std::vector<uint8_t> layer_is_full_{1};
    Tensor cos_,sin_,final_norm_w_,expert_ids_,token_ids_,router_bridge_;
    Tensor resident_route_ordinals_,resident_route_tokens_;
    int64_t gdn_nvh_=32,gdn_dk_=128,gdn_dv_=128,gdn_conv_dim_=8192,gdn_kc_=4;
    bool linear_acc32_=false,retained_prefill_graph_=false,has_qk_norm_=true;
    bool resident_prefill_=false,resident_uint4_n32k32_installed_=false;
    bool has_gdn_=true;
    bool has_gdn_glast_local_capability()const;
    std::vector<int64_t> gdn_glast_local_identity()const;
    double eps_=1e-6;
    int num_cores()const{return owner;}
    int attn_tp()const{return owner==6?4:owner;}
    int mlp_tp()const{return owner;}
    int num_layers()const{return 40;}
    int hidden_size()const{return 2048;}
    int num_q_heads()const{return 16;}
    int num_kv_heads()const{return owner==8?8:4;}
    int head_dim()const{return 256;}
__HELPERS__
    int64_t minimum_eager_spm_bytes()const;
    std::vector<int64_t> cold_topology_arguments()const;
    std::vector<int64_t> exact_profile_arguments()const;
    int64_t subclass_layout_hash()const;
    std::vector<int64_t> kvinsert_cost_weight_identity()const;
};
__METHODS__
}
int main(){
    using namespace v3;
    for(int cores:{4,6,8}){
        Qwen3_5MoeModel m;m.owner=cores;m.moe_.tensor_parallel=cores;
        const auto fp8_cold=m.cold_topology_arguments();
        const auto fp8_profile=m.exact_profile_arguments();
        const auto fp8_hash=m.subclass_layout_hash();
        const auto fp8_cost=m.kvinsert_cost_weight_identity();
        m.moe_layer_weights_[0].routed_weight_mode=6;
        const auto int4_cold=m.cold_topology_arguments();
        const auto int4_profile=m.exact_profile_arguments();
        const auto int4_hash=m.subclass_layout_hash();
        const auto int4_cost=m.kvinsert_cost_weight_identity();
        std::vector<int64_t> schedule_prefix{1};
        const auto alpha=shared_alpha_local_identity(m.shared_expert_tp());
        schedule_prefix.insert(schedule_prefix.end(),alpha.begin(),alpha.end());
        if(m.has_gdn_glast_local_capability()){
            const auto glast=m.gdn_glast_local_identity();
            schedule_prefix.insert(schedule_prefix.end(),glast.begin(),glast.end());
        }
        assert(int4_cost.size()>=schedule_prefix.size());
        assert(std::equal(schedule_prefix.begin(),schedule_prefix.end(),int4_cost.begin()));
        assert(fp8_hash!=int4_hash && fp8_cost!=int4_cost);
        assert(m.minimum_eager_spm_bytes()==(cores==6?87808:0));
        if(cores==6){
            assert(fp8_cold.size()==11 && int4_cold.size()==11);
            assert(fp8_cold.back()==87808 && int4_cold.back()==87808);
            assert(fp8_cold[6]==6 && int4_cold[6]==4);
            assert(int4_cold[5]==6 && int4_cold[7]==6);
            assert(fp8_profile[9]==576 && int4_profile[9]==512);
            assert(fp8_profile[11]==576 && int4_profile[11]==576);
            assert(std::equal(int4_cold.begin(),int4_cold.end(),int4_cost.begin()+schedule_prefix.size()));
        }else{
            assert(fp8_cold.size()==7 && int4_cold.size()==7);
            assert(fp8_cold==int4_cold && fp8_profile==int4_profile);
        }
        // Replanning without changing installed owners reproduces identity;
        // changing mode is only emulated here to expose accidental collisions.
        assert(m.subclass_layout_hash()==int4_hash);
        assert(m.kvinsert_cost_weight_identity()==int4_cost);
        m.moe_layer_weights_[0].routed_weight_mode=5;
        assert(m.subclass_layout_hash()==fp8_hash);
        assert(m.kvinsert_cost_weight_identity()==fp8_cost);
        if (cores==8) {
            m.resident_prefill_=true;
            const auto resident_hash=m.subclass_layout_hash();
            const auto resident_cost=m.kvinsert_cost_weight_identity();
            assert(resident_hash!=fp8_hash && resident_cost!=fp8_cost);
            m.resident_uint4_n32k32_installed_=true;
            assert(m.subclass_layout_hash()!=resident_hash);
            assert(m.kvinsert_cost_weight_identity()!=resident_cost);
            m.resident_prefill_=m.resident_uint4_n32k32_installed_=false;
            assert(m.subclass_layout_hash()==fp8_hash);
            assert(m.kvinsert_cost_weight_identity()==fp8_cost);
        }
    }
}
'''
    for marker, value in {"__LOCAL_SCHEDULE__": local_schedule, "__LAYOUT_VERSION__": layout_version, "__MIX__": _definition(base, "inline int64_t layout_mix("),
                          "__LW__": ",".join(lw_names), "__MW__": ",".join(mw_names),
                          "__HELPERS__": helpers, "__METHODS__": methods}.items():
        code = code.replace(marker, value)
    _run_cpp(tmp_path, code)


def test_native_install_metadata_matches_routed4_not_shared6(tmp_path):
    from test_qwen35_moe_core_native import _definition, _run_cpp, NATIVE

    install = _definition(NATIVE.read_text(), "void Qwen3_5MoeModel::set_weights(")
    admission = install[install.index("    const bool routed_uint4"):
                        install.index("    const auto check_list")]
    start = install.index("                const auto metadata =")
    metadata = install[start:install.index("            } else {", start)]
    code = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "core/rpu_grouped_linear_tiling.h"
#define TORCH_CHECK(ok, ...) do {if(!(ok))throw std::runtime_error(#ok);}while(false)
constexpr int kRoutedUint4Mode=6,kRoutedInt8Mode=1,kRoutedUint4GroupSize=128,kRoutedUint4DownGroupSize=64;
struct Tensor {std::vector<int64_t> shape;int64_t bytes;};
struct Model {
    int owner=6;
    bool linear_acc32_=false;
    int num_cores()const{return owner;}
    void check(int mode,bool stale) {
        const int64_t N=40,routed_weight_mode=mode;
        const int64_t num_q_heads=16,num_kv_heads=owner==8?8:4,head_dim=256,hidden_size=2048;
        const int64_t num_experts=256,top_k=8,routed_intermediate=512,shared_intermediate=512;
        const int64_t tensor_parallel=owner,gdn_num_v_heads=32,gdn_key_head_dim=128;
        const int64_t gdn_value_head_dim=128,gdn_conv_dim=8192,gdn_conv_kernel=4;
        const std::array<int64_t,3> mrope_section{11,11,10};
        std::vector<int64_t> layer_is_full(40);
        for(int l=0;l<40;++l)layer_is_full[l]=l%4==3;
__ADMISSION__
        assert(routed_nc==(owner==6 && mode==6?4:owner));
        assert(physical_routed_intermediate==(owner==6 && mode!=6?576:512));
        assert(physical_shared_intermediate==(owner==6?576:512));
        if(mode!=6)return;
        for(size_t j=0;j<3;++j){
            const int64_t pack_cores=stale?6:(owner==6?4:owner);
            const int64_t pack_width=stale?768:512;
            const auto packed=rpu_grouped_tiling::uint4_metadata_layout(
                j==2?2048:pack_width/pack_cores,j==2?pack_width/pack_cores:2048,
                j==2?64:128,pack_cores);
            Tensor tensor{{256,packed.q_blocks,packed.n_blocks,pack_cores,4,64},
                          256*packed.expert_bytes};
            const int64_t i=0;
            const std::array<std::pair<const Tensor*,const char*>,3> routed_scales{{
                {&tensor,"gate"},{&tensor,"up"},{&tensor,"down"}}};
            auto check_shape=[](const Tensor& t,std::vector<int64_t> shape,const char*,int64_t){
                TORCH_CHECK(t.shape==shape);
            };
            auto check_bytes=[](const Tensor& t,const char*,int64_t,int64_t bytes){
                TORCH_CHECK(t.bytes==bytes);
            };
__METADATA__
        }
    }
};
int main(){
    for(int cores:{4,6,8})for(int mode:{1,5,6}){Model model;model.owner=cores;model.check(mode,false);}
    Model model;bool rejected=false;
    try{model.check(6,true);}catch(const std::runtime_error&){rejected=true;}
    assert(rejected);
}
'''.replace("__ADMISSION__", admission).replace("__METADATA__", metadata)
    _run_cpp(tmp_path, code)
