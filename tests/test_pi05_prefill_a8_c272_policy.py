"""Execute the real Pi C272 owner/domain/manifest code without device allocation."""
from pathlib import Path
from pi_wall_host_catalog import CATALOG_INCLUDE, absolute_candidate_includes

from test_siglip_attention_dry_live_contract import _function, _run_native


ROOT = Path(__file__).resolve().parents[1]


def test_c272_a8_exact_domain_manifest_and_payload_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    source = (ROOT / "src/fused/rpu_gemma_model.cpp").read_text()
    cache = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    names = ("bool pi05_prefill_a8_short(",
             "std::vector<int64_t> pi05_prefill_a8_short_arguments(",
             "bool pi05_prefill_a8_short_owner(",
             "bool pi05_prefill_a8_short_domain(",
             "void pi05_prefill_a8_short_validate_manifest(",
             "FmbLinearAccumulationPolicy pi05_prefill_linear_accumulation(")
    methods = "\n".join(_function(source, "    " + name) for name in names)
    # The only tensor substitution is the device boundary. Geometry, tensor
    # dtype/shape/contiguity, every cold flag and payload admission stay real.
    methods = methods.replace("at::kPrivateUse1", "at::kCPU")
    constants = "\n".join(line for line in source.splitlines()
                          if line.startswith("constexpr int64_t PI05_GATEUP_A8_C272_"))
    dry = _function(source, "    FmbPhysicalExecutionManifest physical_manifest_for_candidate(")
    begin = dry.index("        TORCH_CHECK(!pi05_prefill_a8_short() ||")
    guard = dry[begin:dry.index(");", begin) + 2]
    append = _function(dry, "        if (pi05_prefill_a8_short()) {")
    availability = _function(cache, "class KernelPayloadAvailability final")
    kernel_ids = absolute_candidate_includes(_function(cache, "enum class KernelId"))
    code = CATALOG_INCLUDE + r'''
#include "core/fused_model_base.h"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <array>
#include <atomic>
#include <cassert>
#include <iostream>
using namespace v3;
''' + kernel_ids + ";\n" + availability + r''';
KernelId pi_wall_kernel_id(std::size_t ordinal) { return static_cast<KernelId>(static_cast<unsigned>(KernelId::PI_WALL_BEGIN)+ordinal); }
struct KernelCache {
    KernelPayloadAvailability availability;
    static KernelCache& instance() { static KernelCache cache; return cache; }
    bool has_loaded(KernelId id) const { return availability.has_loaded(id); }
};
''' + constants + r'''
namespace v3 {
struct Owner {
    bool linear_acc32_=false;
    bool pi05_prefill_a8_=true, pi05_prefill_gateup_pair_=false;
    bool pi05_prefill_o_pair_=false;
    bool pi05_prefill_qkv_pair_=false, pi05_prefill_down_pair_=false;
    bool pi05_prefill_owner_norm_=false, pi05_prefill_kv1_pair_direct_=false;
    bool pi05_prefill_kv_only_=false, pi05_ring_xor3_opt_in_=false;
    int cores=8, attention_tp=8, mlp_cores=8, layers=18, dma_policy=0;
    int64_t hidden=2048, intermediate=16384, q_heads=8, kv_heads=8, hd=256;
    FusedModelBase::ChunkEnvelope envelope{2048,272};
    struct Weights { at::Tensor gate_proj_w,up_proj_w,gate_ws,up_ws; };
    std::vector<Weights> layer_weights_;
    int num_cores() const { return cores; }
    int attn_tp() const { return attention_tp; }
    int mlp_tp() const { return mlp_cores; }
    int num_layers() const { return layers; }
    int64_t hidden_size() const { return hidden; }
    int64_t intermediate_size() const { return intermediate; }
    int64_t num_q_heads() const { return q_heads; }
    int64_t num_kv_heads() const { return kv_heads; }
    int64_t head_dim() const { return hd; }
    int pi05_prefill_dma_policy() const { return dma_policy; }
    const FusedModelBase::ChunkEnvelope& chunk_envelope() const { return envelope; }
    Owner() {
        const auto weight=at::empty({16384,2048},at::TensorOptions().dtype(at::kChar));
        const auto scale=at::ones({16384},at::TensorOptions().dtype(at::kHalf));
        layer_weights_.assign(18,{weight,weight,scale,scale});
    }
''' + methods + r'''
    FmbPhysicalExecutionManifest dry(const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout, int64_t physical_len=544,
        int64_t logical_len=544, int64_t position=0) const {
''' + guard + r'''
        FmbPhysicalExecutionManifest manifest;
        manifest.state=FmbPhysicalManifestState::COMPLETE;
        manifest.logical_length=logical_len; manifest.physical_length=physical_len;
        manifest.execution_padding_rows=physical_len-logical_len;
        manifest.kv_logical_length=position+logical_len;
        manifest.kv_insert_physical_rows=physical_len;
        manifest.graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;
        manifest.linear_accumulation=pi05_prefill_linear_accumulation();
        auto append=[&](FmbRouteFamily family,int64_t site,int64_t selector,
                       int64_t flags=0,std::vector<int64_t> args={},int64_t invocation=0) {
            manifest.routes.push_back({site,family,selector,flags,std::move(args),invocation});
        };
''' + append + r'''
        return manifest;
    }
};
} // namespace v3: retain the production method's namespace lookup.
template<class F> void rejects(F run) {
    bool failed=false;
    try { run(); } catch (const c10::Error&) { failed=true; }
    assert(failed);
}
int main() {
    at::set_num_threads(1);
    auto& payload=KernelCache::instance().availability;
    KernelPayloadAvailability::ExtendedPayloads current;
    current.set(static_cast<size_t>(pi_wall_kernel_id(rpu_pi_wall::select(
        rpu_pi_wall::Operation::A8OnlineGelu,rpu_pl_tiling::WeightType::Int8,true,272,2048,2048).ordinal))
        - KernelPayloadAvailability::kExtendedBegin);
    payload.publish(true,true,false,false,false,false,current);
    Owner owner;
    FmbThreeStageChunkPlan plan;
    plan.chunk_mode=ChunkMode::KV_FIRST;
    plan.compute.chunks={{0,0,272,272},{1,272,272,544}};
    plan.qkv.chunks=plan.compute.chunks;
    LayoutContext layout;
    layout.chunk_size=layout.kv_insert_chunk_size=272;
    layout.max_kv_seq_len=544;layout.batch_size=1;
    layout.use_attn_mask=true;layout.is_causal=false;
    layout.attention_policy=AttentionExecutionPolicy::DDR_KV;
    assert(owner.pi05_prefill_a8_short_owner());
    assert(owner.pi05_prefill_a8_short_domain(plan,layout,544,544,0));
    const auto valid=owner.dry(plan,layout);
    assert(valid.routes.size()==3);
    assert(valid.routes[0].arguments[16]==17 && valid.routes[0].arguments[17]==18);
    owner.pi05_prefill_a8_short_validate_manifest(valid);
    auto canonical=valid;
    // Production finalizes route ordering in the shared-route appender before
    // constructing a consumer. Retain that real canonicalization here too.
    append_fmb_shared_runtime_routes(canonical,plan,2048,0);
    validate_fmb_physical_manifest(canonical,544,0);
    FmbPhysicalManifestConsumer consumer;
    consumer.reset(canonical);
    consumer.consume_route(FmbRouteFamily::GRAPH_SCHEDULE,PI05_GATEUP_A8_C272_POLICY_SITE,
                           1,1,owner.pi05_prefill_a8_short_arguments(),0);
    for(int chunk=0;chunk!=2;++chunk)
        consumer.consume_route(FmbRouteFamily::LINEAR,PI05_GATEUP_A8_C272_LINEAR_SITE,
                               1,0,owner.pi05_prefill_a8_short_arguments(),chunk);

    for(int mutation=0;mutation!=21;++mutation) {
        auto bad_plan=plan;auto bad_layout=layout;
        int64_t physical=544,logical=544,position=0;
        switch(mutation) {
        case 0:physical=800;logical=800;break;
        case 1:physical=560;break;
        case 2:logical=543;break;
        case 3:position=16;break;
        case 4:bad_layout.chunk_size=256;break;
        case 5:bad_layout.kv_insert_chunk_size=544;break;
        case 6:bad_layout.batch_size=2;break;
        case 7:bad_layout.use_attn_mask=false;break;
        case 8:bad_layout.is_causal=true;break;
        case 9:bad_layout.max_kv_seq_len=800;break;
        case 10:bad_layout.attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;break;
        case 11:bad_plan.chunk_mode=ChunkMode::SEQUENTIAL;break;
        case 12:bad_plan.compute.chunks.pop_back();break;
        case 13:bad_plan.qkv.chunks.pop_back();break;
        case 14:bad_plan.compute.chunks[1].idx=0;break;
        case 15:bad_plan.qkv.chunks[1].offset=256;break;
        case 16:bad_plan.compute.chunks[1].len=256;break;
        case 17:bad_plan.qkv.chunks[1].kv_seq_len=800;break;
        case 18:bad_plan.compute.chunks.push_back(bad_plan.compute.chunks[1]);break;
        case 19:owner.envelope.max_kv_len=0;break;
        case 20:owner.envelope.max_kv_len=4096;break;
        }
        assert(!owner.pi05_prefill_a8_short_domain(bad_plan,bad_layout,physical,logical,position));
        rejects([&]{owner.dry(bad_plan,bad_layout,physical,logical,position);});
        owner.envelope.max_kv_len=2048;
    }
    for(int mutation=0;mutation!=23;++mutation) {
        auto bad=valid;
        switch(mutation) {
        case 0:bad.state=FmbPhysicalManifestState::UNSPECIFIED;break;
        case 1:bad.logical_length=800;break;
        case 2:bad.physical_length=560;break;
        case 3:bad.execution_padding_rows=16;break;
        case 4:bad.kv_logical_length=800;break;
        case 5:bad.kv_insert_physical_rows=560;break;
        case 6:bad.graph_lifecycle=FmbGraphLifecycle::UNSPECIFIED;break;
        case 7:bad.linear_accumulation=FmbLinearAccumulationPolicy::ACC16;break;
        case 8:bad.routes.erase(bad.routes.begin());break;
        case 9:bad.routes.pop_back();break;
        case 10:bad.routes.push_back(bad.routes[0]);break;
        case 11:bad.routes.push_back(bad.routes[1]);break;
        case 12:bad.routes[0].family=FmbRouteFamily::LINEAR;break;
        case 13:bad.routes[1].family=FmbRouteFamily::GRAPH_SCHEDULE;break;
        case 14:bad.routes[0].selector=2;break;
        case 15:bad.routes[0].flags=0;break;
        case 16:bad.routes[1].flags=1;break;
        case 17:bad.routes[0].invocation=1;break;
        case 18:bad.routes[1].invocation=-1;break;
        case 19:bad.routes[2].invocation=2;break;
        case 20:bad.routes[2].arguments[16]=18;break; // Never expand A8 to tail layer.
        case 21:bad.routes[1].arguments[13]=544;break; // sx reservation is ABI-bound.
        case 22:bad.routes[2].invocation=0;break;
        }
        rejects([&]{owner.pi05_prefill_a8_short_validate_manifest(bad);});
    }
    owner.pi05_prefill_a8_=false;
    rejects([&]{owner.pi05_prefill_a8_short_validate_manifest(valid);});
    auto off=owner.dry(plan,layout);
    assert(off.routes.empty() && off.linear_accumulation==FmbLinearAccumulationPolicy::ACC16);
    owner.pi05_prefill_a8_short_validate_manifest(off);
    owner.pi05_prefill_a8_=true;owner.pi05_prefill_gateup_pair_=true;
    // The existing P800 pair remains a distinct owner; no C272 route or payload.
    auto pair=owner.dry(plan,layout,800,800);
    assert(pair.routes.empty() && pair.linear_accumulation==FmbLinearAccumulationPolicy::MIXED_BY_SITE);
    rejects([&]{owner.pi05_prefill_a8_short_validate_manifest(valid);});
    owner.pi05_prefill_gateup_pair_=false;
    for(auto flag : {&owner.pi05_prefill_o_pair_,
        &owner.pi05_prefill_qkv_pair_,&owner.pi05_prefill_down_pair_,&owner.pi05_prefill_owner_norm_,
        &owner.pi05_prefill_kv1_pair_direct_,&owner.pi05_prefill_kv_only_,&owner.pi05_ring_xor3_opt_in_}) {
        *flag=true;assert(!owner.pi05_prefill_a8_short_owner());
        rejects([&]{owner.pi05_prefill_a8_short_validate_manifest(valid);});*flag=false;
    }
    for(auto value : {&owner.cores,&owner.attention_tp,&owner.mlp_cores,&owner.layers,&owner.dma_policy}) {
        const int saved=*value;*value=6;
        assert(!owner.pi05_prefill_a8_short_owner());*value=saved;
    }
    auto& tail=owner.layer_weights_.back();
    const auto saved_weight=tail.gate_proj_w,saved_scale=tail.up_ws;
    tail.gate_proj_w=at::empty({16384,2048},at::TensorOptions().dtype(at::kHalf));
    assert(!owner.pi05_prefill_a8_short_owner());tail.gate_proj_w=saved_weight;
    tail.up_ws=at::ones({16384},at::TensorOptions().dtype(at::kFloat));
    assert(!owner.pi05_prefill_a8_short_owner());tail.up_ws=saved_scale;
    payload.publish(true,true,false,false,false,false);
    assert(!owner.pi05_prefill_a8_short_owner());
    rejects([&]{owner.dry(plan,layout);});
    rejects([&]{owner.pi05_prefill_a8_short_validate_manifest(valid);});
    owner.pi05_prefill_gateup_pair_=true;
    owner.pi05_prefill_a8_short_validate_manifest(pair); // P800 does not require C272 payload.
    owner.pi05_prefill_gateup_pair_=false;
    payload.withdraw();
    rejects([&]{owner.pi05_prefill_a8_short_owner();});
    payload.publish(true,true,false,false,false,false,current);
    owner.pi05_prefill_a8_short_validate_manifest(valid);
    std::cout << "Pi C272 A8 real owner/domain/manifest/payload: PASS\n";
}
'''
    result = _run_native(tmp_path, code, torch,
                         extra_sources=(ROOT / "src/core/fmb_three_stage_chunk_plan.cpp",))
    assert "Pi C272 A8 real owner/domain/manifest/payload: PASS" in result
