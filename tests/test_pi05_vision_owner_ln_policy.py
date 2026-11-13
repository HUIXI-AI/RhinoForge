"""Run the real owner-LN descriptor helpers with CPU tensors and no device load."""
from pathlib import Path

from test_siglip_attention_dry_live_contract import _function, _run_native


ROOT = Path(__file__).resolve().parents[1]


def test_owner_ln_exact_profile_dry_live_and_missing_payloads(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    source = (ROOT / "src/fused/rpu_siglip_model.cpp").read_text()
    names = (
        "FmbLinearAccumulationPolicy linear_accumulation_policy(",
        "void require_pi05_vision_owner_ln_payloads(",
        "std::vector<int64_t> pi05_vision_owner_ln_arguments(",
        "bool pi05_vision_supported_owner(",
        "void require_pi05_vision_owner_ln_profile(",
        "void validate_pi05_vision_owner_ln_manifest(",
        "void consume_pi05_vision_owner_ln_route(",
        "bool use_pi05_ring_xor3(",
    )
    methods = "\n".join(_function(source, "    " + name) for name in names)
    constants = "\n".join(line for line in source.splitlines()
                          if line.startswith("constexpr int64_t SIGLIP_") and any(
                              key in line for key in ("ALL_REDUCE_SITE", "OWNER_LN_", "RAW_SPM_ATTN_SITE")))
    dry = _function(source, "    FmbPhysicalExecutionManifest physical_manifest_for_candidate(")
    appends = []
    for site in ("SIGLIP_ATTN_ALL_REDUCE_SITE", "SIGLIP_MLP_ALL_REDUCE_SITE"):
        begin = dry.index("            append(\n                FmbRouteFamily::ALL_REDUCE,\n                " + site)
        end = dry.index("chunk.idx);", begin) + len("chunk.idx);")
        appends.append(dry[begin:end])
    code = r'''
#include "core/fused_model_base.h"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <array>
#include <cassert>
#include <iostream>
using namespace v3;
enum class KernelId : uint16_t { PI05_VISION_OWNER_REDUCE_LAYERNORM_M768N1152,
    PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M768N1152,
    PI05_VISION_OWNER_REDUCE_LAYERNORM_M512N1152,
    PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M512N1152 };
struct KernelCache {
    bool owner=true, compact=true, owner512=true, compact512=true;
    static KernelCache& instance() { static KernelCache cache; return cache; }
    bool has_loaded(KernelId id) const {
        switch (id) {
        case KernelId::PI05_VISION_OWNER_REDUCE_LAYERNORM_M768N1152: return owner;
        case KernelId::PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M768N1152: return compact;
        case KernelId::PI05_VISION_OWNER_REDUCE_LAYERNORM_M512N1152: return owner512;
        default: return compact512;
        }
    }
};
''' + constants + r'''
struct HostContext {
    FmbPhysicalManifestConsumer consumer;
    bool has_complete_physical_manifest() const { return consumer.complete(); }
    const FmbRouteManifestEntry& consume_physical_route(
        FmbRouteFamily family, int64_t site, int64_t selector, int64_t flags,
        at::IntArrayRef args, int64_t invocation) const {
        return consumer.consume_route(family,site,selector,flags,args,invocation);
    }
};
struct Owner {
    bool linear_acc32_=false;
    bool pi05_vision_owner_ln_opt_in_=true, pi05_ring_xor3_opt_in_=true;
    int cores=8, layers=27;
    int64_t image_batch_count_=3, orig_head_dim_=72, projection_dim_=2048, width=1152;
    double eps_=1e-6;
    struct LayerWeights { at::Tensor q_w,k_w,v_w,o_w,fc1_w,fc2_w,q_ws,k_ws,v_ws,o_ws,fc1_ws,fc2_ws; };
    std::vector<LayerWeights> layer_weights_;
    std::vector<int> layer_bias_norm_=std::vector<int>(27);
    HostContext state;
    HostContext& ctx() { return state; }
    int num_cores() const { return cores; }
    int num_layers() const { return layers; }
    int64_t hidden_size() const { return width; }
    int64_t intermediate_size() const { return 4352; }
    int64_t num_q_heads() const { return 16; }
    int64_t num_kv_heads() const { return 16; }
    int64_t head_dim() const { return 80; }
    Owner() {
        auto w=at::ones({1},at::TensorOptions().dtype(at::kChar));
        auto s=at::ones({1},at::TensorOptions().dtype(at::kHalf));
        layer_weights_.assign(27,{w,w,w,w,w,w,s,s,s,s,s,s});
    }
''' + methods + r'''
    FmbPhysicalExecutionManifest dry() const {
        FmbPhysicalExecutionManifest manifest;
        manifest.state=FmbPhysicalManifestState::COMPLETE;
        const int64_t rows=image_batch_count_*256;
        manifest.logical_length=manifest.physical_length=rows;
        manifest.kv_logical_length=manifest.kv_insert_physical_rows=rows;
        manifest.graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;
        manifest.linear_accumulation=FmbLinearAccumulationPolicy::ACC16;
        manifest.routes.push_back({SIGLIP_RAW_SPM_ATTN_SITE,FmbRouteFamily::ATTENTION,
            static_cast<int64_t>(AttentionExecutionPolicy::SPM_KV_BY_MHA),0,
            {image_batch_count_,256,rows,rows,16,80},0});
        auto append=[&](FmbRouteFamily family,int64_t site,int64_t selector,
                       std::vector<int64_t> args,int64_t invocation) {
            manifest.routes.push_back({site,family,selector,0,std::move(args),invocation});
        };
        const ChunkInfo chunk{0,0,rows,rows};
''' + "\n".join(appends) + r'''
        return manifest;
    }
};
template<class F> void rejects(F run) {
    bool failed=false;
    try { run(); } catch (const c10::Error&) { failed=true; }
    assert(failed);
}
int main() {
    at::set_num_threads(1);
    Owner owner;
    auto valid=owner.dry();
    owner.require_pi05_vision_owner_ln_profile(768,3);
    owner.validate_pi05_vision_owner_ln_manifest(valid);
    owner.state.consumer.reset(valid);
    owner.consume_pi05_vision_owner_ln_route(false,{0,0,768,768});
    owner.consume_pi05_vision_owner_ln_route(true,{0,0,768,768});
    for (bool compact : {false,true}) {
        auto bad=valid; bad.routes[compact?2:1].arguments[8]+=32;
        owner.state.consumer.reset(bad);
        rejects([&]{owner.consume_pi05_vision_owner_ln_route(compact,{0,0,768,768});});
    }
    for (int mutation=0;mutation<20;++mutation) {
        auto bad=valid;
        switch(mutation) {
        case 0: bad.state=FmbPhysicalManifestState::UNSPECIFIED;break;
        case 1: bad.logical_length=767;break;
        case 2: bad.physical_length=784;break;
        case 3: bad.execution_padding_rows=16;break;
        case 4: bad.kv_logical_length=784;break;
        case 5: bad.kv_insert_physical_rows=784;break;
        case 6: bad.graph_lifecycle=FmbGraphLifecycle::UNSPECIFIED;break;
        case 7: bad.linear_accumulation=FmbLinearAccumulationPolicy::MIXED_BY_SITE;break;
        case 8: bad.routes.erase(bad.routes.begin()+1);break;
        case 9: bad.routes.pop_back();break;
        case 10: bad.routes.push_back(bad.routes[1]);break;
        case 11: bad.routes[2].selector=6;break;
        case 12: bad.routes[1].family=FmbRouteFamily::GRAPH_SCHEDULE;break;
        case 13: bad.routes[1].flags=1;break;
        case 14: bad.routes[2].invocation=1;break;
        case 15: bad.routes[0].selector=static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV);break;
        case 16: bad.routes[0].arguments[0]=2;break;
        case 17: bad.routes[2].site_id=123;break;
        case 18: bad.routes[1].arguments[10]=100000;break;
        case 19: bad.routes[2].arguments=bad.routes[1].arguments;break;
        }
        rejects([&]{owner.validate_pi05_vision_owner_ln_manifest(bad);});
    }
    for (int rows : {256,512,752,784})
        rejects([&]{owner.require_pi05_vision_owner_ln_profile(rows,3);});
    rejects([&]{owner.require_pi05_vision_owner_ln_profile(768,2);});
    for (int cores : {4,6}) {
        owner.cores=cores;rejects([&]{owner.validate_pi05_vision_owner_ln_manifest(valid);});
    }
    owner.cores=8;owner.eps_=1e-5;
    rejects([&]{owner.validate_pi05_vision_owner_ln_manifest(valid);});owner.eps_=1e-6;
    owner.layer_weights_[26].fc2_w=owner.layer_weights_[26].fc2_w.to(at::kHalf);
    rejects([&]{owner.validate_pi05_vision_owner_ln_manifest(valid);});
    owner.layer_weights_[26].fc2_w=owner.layer_weights_[26].fc2_w.to(at::kChar);
    for (bool missing_owner : {false,true}) {
        auto& cache=KernelCache::instance();
        cache.owner=!missing_owner;cache.compact=missing_owner;
        rejects([&]{owner.validate_pi05_vision_owner_ln_manifest(valid);});
    }
    auto& cache=KernelCache::instance();cache.owner=cache.compact=true;
    // Both real precision owners and both exact physical camera shapes consume
    // the same explicit ABI. Changed dtype with leftover scales is invalid.
    for (bool fp16 : {false,true}) {
        auto w=at::ones({1},at::TensorOptions().dtype(fp16?at::kHalf:at::kChar));
        auto scale=fp16?at::Tensor():at::ones({1},at::TensorOptions().dtype(at::kHalf));
        owner.layer_weights_.assign(27,{w,w,w,w,w,w,scale,scale,scale,scale,scale,scale});
        for (int64_t images : {2,3}) {
            owner.image_batch_count_=images;
            const int64_t rows=images*256;
            auto descriptor=owner.dry();
            owner.require_pi05_vision_owner_ln_profile(rows,images);
            owner.validate_pi05_vision_owner_ln_manifest(descriptor);
            owner.state.consumer.reset(descriptor);
            for (bool compact : {false,true})
                owner.consume_pi05_vision_owner_ln_route(compact,{0,0,rows,rows});
            auto wrong_camera=descriptor;wrong_camera.routes[0].arguments[0]=5-images;
            rejects([&]{owner.validate_pi05_vision_owner_ln_manifest(wrong_camera);});
            auto wrong_abi=descriptor;wrong_abi.routes[1].arguments[5]=images==2?96:64;
            rejects([&]{owner.validate_pi05_vision_owner_ln_manifest(wrong_abi);});
        }
        if (fp16) {
            owner.layer_weights_[26].fc2_ws=at::ones({1},at::TensorOptions().dtype(at::kHalf));
            rejects([&]{owner.require_pi05_vision_owner_ln_profile(768,3);});
            owner.layer_weights_[26].fc2_ws=at::Tensor();
        }
    }
    owner.image_batch_count_=2;
    auto two=owner.dry();
    cache.owner512=false;
    rejects([&]{owner.validate_pi05_vision_owner_ln_manifest(two);});
    owner.require_pi05_vision_owner_ln_profile(768,3); // Existing M768 REF remains sufficient.
    cache.owner512=true;cache.compact512=false;
    rejects([&]{owner.validate_pi05_vision_owner_ln_manifest(two);});
    cache.owner=cache.compact=cache.owner512=cache.compact512=false;
    rejects([&]{owner.require_pi05_vision_owner_ln_payloads(0);});
    owner.image_batch_count_=3;
    owner.pi05_vision_owner_ln_opt_in_=false;
    rejects([&]{owner.validate_pi05_vision_owner_ln_manifest(valid);});
    const auto off=owner.dry();
    assert(off.routes[1].selector==6 && off.routes[2].selector==6);
    assert(off.routes[1].arguments.empty() && off.routes[2].arguments.empty());
    owner.validate_pi05_vision_owner_ln_manifest(off); // No optional payload dependency.
    owner.require_pi05_vision_owner_ln_profile(256,1);
    std::cout<<"Vision owner-LN native profile, payload, dry/live ABI: PASS\n";
}
'''
    result = _run_native(tmp_path, code, torch,
                         extra_sources=(ROOT / "src/core/fmb_three_stage_chunk_plan.cpp",))
    assert "Vision owner-LN native profile, payload, dry/live ABI: PASS" in result
