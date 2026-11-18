"""Execute the actual paired Prefill cold geometry/dtype/payload guards."""
from pathlib import Path
from pi_wall_host_catalog import CATALOG_INCLUDE
import re

from test_siglip_attention_dry_live_contract import _function, _run_native

ROOT = Path(__file__).resolve().parents[1]


def test_exact_pair_profiles_keep_fp16_unquantized_and_require_payloads(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    source = (ROOT / "src/fused/rpu_gemma_model.cpp").read_text()
    signatures = (
        "void set_prefill_pair_rows(", "int64_t pi05_pair_rows(",
        "int64_t pi05_prefix_rows(", "int64_t pi05_pair_slab_bytes(",
        "int64_t pi05_pair_owner_bytes(", "int64_t pi05_pair_owner_elements(",
        "int64_t pi05_pair_mask_bytes(", "int64_t pi05_pair_tile_rows(",
        "bool pi05_pair_single_compact(", "bool pi05_pair_slab_compact(",
        "int64_t pi05_pair_spill_elements(", "bool pi05_pair_scratch_mask(", "int64_t pi05_pair_mask_selector(",
        "bool pi05_pair_mask_upload(", "bool pi05_pair_fp16(",
        "bool pi05_pair_weight(", "KernelId pi05_prefill_kv1_pair_kernel(",
        "KernelId pi05_prefill_gateup_pair_kernel(",
        "int64_t pi05_a8_bytes(", "int64_t pi05_a8_root_bytes(",
        "std::vector<int64_t> pi05_prefill_gateup_pair_arguments(",
        "std::vector<int64_t> pi05_prefill_kv1_pair_direct_policy_arguments(",
        "bool pi05_prefill_extended_payloads_loaded(",
        "bool pi05_prefill_dma_geometry(", "bool pi05_prefill_dma_layout(",
        "bool pi05_prefill_mask_domain_eligible(",
        "std::vector<int64_t> pi05_owner_norm_arguments(",
        "std::vector<int64_t> pi05_down_pair_arguments(",
    )
    methods = "\n".join(_function(source, "    " + signature) for signature in signatures)
    methods = methods.replace("at::kPrivateUse1", "at::kCPU")
    ids = sorted(set(re.findall(r"KernelId::(\w+)", methods)))
    code = CATALOG_INCLUDE + r'''
#include "core/fused_model_base.h"
#include "core/rpu_profile.h"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cassert>
#include <iostream>
#include <set>
using namespace v3;
enum class KernelId : uint16_t {
''' + ",\n".join(ids) + r'''};
KernelId pi_wall_kernel_id(std::size_t ordinal) { return static_cast<KernelId>(1000+ordinal); }
struct KernelCache {
    std::set<KernelId> missing;
    static KernelCache& instance() { static KernelCache c; return c; }
    bool has_loaded(KernelId id) const { return !missing.count(id); }
};
namespace v3 {
struct Owner {
    int64_t pi05_pair_rows_=400;
    bool pi05_pair_rows_bound_=false;
    bool pi05_prefill_carry_ab_=true,pi05_prefill_mask_resident_=true,pi05_prefill_kv_only_=true;
    bool pi05_prefill_a8_=false,pi05_prefill_o_pair_=true,pi05_prefill_gateup_pair_=true;
    bool pi05_prefill_down_pair_=true,pi05_prefill_owner_norm_=true,pi05_prefill_kv1_pair_direct_=true;
    bool pi05_prefill_qkv_pair_=false;
    int cores=8,layers=18;
    int num_cores() const { return cores; }
    int num_layers() const { return layers; }
    int attn_tp() const { return 8; }
    int64_t hidden_size() const { return 2048; }
    int64_t intermediate_size() const { return 16384; }
    int64_t num_q_heads() const { return 8; }
    int64_t num_kv_heads() const { return 8; }
    int64_t head_dim() const { return 256; }
    struct LayerWeights {
        at::Tensor q_w,k_w,v_w,o_w,gate_proj_w,up_proj_w,down_proj_w;
        at::Tensor q_ws,k_ws,v_ws,o_ws,gate_ws,up_ws,down_ws;
    };
    std::vector<LayerWeights> layer_weights_;
    void install(bool fp16) {
        const auto options=at::TensorOptions().dtype(fp16?at::kHalf:at::kChar);
        const auto q=at::empty({2048,2048},options),g=at::empty({16384,2048},options);
        const auto d=at::empty({2048,16384},options);
        const auto qs=fp16?at::Tensor():at::ones({2048},at::TensorOptions().dtype(at::kHalf));
        const auto gs=fp16?at::Tensor():at::ones({16384},at::TensorOptions().dtype(at::kHalf));
        layer_weights_.assign(18,{q,q,q,q,g,g,d,qs,qs,qs,qs,gs,gs,qs});
    }
''' + methods + r'''
};
} // namespace v3: preserve the production detail namespace lookup.
template<class F> void rejects(F f) { bool bad=false;try{f();}catch(const c10::Error&){bad=true;}assert(bad); }
int main() {
    at::set_num_threads(1);
    // Link and execute the real shared manifest selector. Every admitted
    // payload must agree with the launcher's exact ring family; nearby
    // unsupported rows retain their ordinary shared route.
    const auto xor3=static_cast<int64_t>(FmbSharedAllReduceRouteSelector::RING_PI05_XOR3);
    for(int64_t rows:{272,288,304,320,336,400,416,432,448,464,480}) {
        assert(fmb_ring_all_reduce_route_selector(rows,2048,true,8)==xor3);
        assert(fmb_ring_all_reduce_route_selector(rows,2048,true,4)!=xor3);
    }
    assert(fmb_ring_all_reduce_route_selector(352,2048,true,8)!=xor3);
    assert(fmb_ring_all_reduce_route_selector(352,2048,true,8)==
           fmb_ring_all_reduce_route_selector(352,2048,false,8));
    for(bool fp16:{false,true}) for(int64_t rows:{272,288,304,320,400,416,432,448}) {
        Owner owner;
        rejects([&]{owner.set_prefill_pair_rows(144);});
        rejects([&]{owner.set_prefill_pair_rows(352);});
        owner.set_prefill_pair_rows(rows);owner.set_prefill_pair_rows(rows);
        rejects([&]{owner.set_prefill_pair_rows(rows==272?400:272);});
        owner.install(fp16);
        rejects([&]{owner.set_prefill_pair_rows(rows);});
        assert(owner.pi05_prefill_dma_geometry());
        for(int cores:{4,6}) {
            owner.cores=cores;assert(!owner.pi05_prefill_dma_geometry());
        }
        owner.cores=8;
        assert(owner.pi05_pair_fp16()==fp16);
        assert(owner.pi05_prefix_rows()==rows*2);
        assert(owner.pi05_pair_slab_bytes()==rows*4096);
        assert(owner.pi05_pair_owner_bytes()==rows*512);
        assert(owner.pi05_pair_mask_bytes()==rows*rows*4);
        assert(owner.pi05_pair_owner_bytes()%256==0);
        assert(owner.pi05_pair_scratch_mask()==(rows==416 || rows==432 || rows==448));
        assert(owner.pi05_pair_tile_rows()==(rows==448?96:(rows==432?112:(rows==416?128:160))));
        const auto down=owner.pi05_down_pair_arguments();
        const auto selectedDown=rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear,
            fp16?rpu_pl_tiling::WeightType::Fp16:rpu_pl_tiling::WeightType::Int8,false,rows,2048,2048);
        auto currentDown=[&](std::vector<int64_t> expected) {
            expected[0]+=20;expected[5]=selectedDown.tile.n_tile;expected[6]=selectedDown.grid_x;
            expected[15]=selectedDown.tile.m_tile;
            expected.insert(expected.end(),{selectedDown.tile.m_tile,selectedDown.tile.k_tile,
                selectedDown.grid_y,48,50,52,26,27});
            return expected;
        };
        if(rows==448) {
            assert(down==currentDown({8,448,2048,2048,8,80,26,4,0,1,1,2,3,0,1,96,3,3,1,917504,1}));
            for(bool* required:{&owner.pi05_prefill_carry_ab_, &owner.pi05_prefill_mask_resident_, &owner.pi05_prefill_kv_only_}) {
                *required=false;assert(!owner.pi05_prefill_dma_geometry());*required=true;
            }
        } else if(rows==416 || rows==432) {
            assert(down==currentDown({7,rows,2048,2048,8,80,26,4,0,1,2,2,3,0,1,rows==432?112:128,3,3}));
            assert(owner.pi05_pair_mask_bytes()<=owner.pi05_pair_slab_bytes());
        } else {
            assert(down==currentDown({4,rows,2048,2048,8,80,26,4,1,1,2,2,3,0,1,160}));
        }
        // The last layer is KV-only in these profiles. A resident half1
        // uploads once, whereas C416 must reload both overwritten S1 masks.
        for(int half:{0,1}) {
            int uploads=0;
            for(int layer=0;layer!=17;++layer) uploads+=owner.pi05_pair_mask_upload(layer,half);
            assert(uploads==((rows==416 || rows==432 || rows==448 || half==0)?17:1));
            assert(owner.pi05_pair_mask_selector(half)==((rows==416 || rows==432 || rows==448)?3:half+1));
        }
        rejects([&]{owner.pi05_pair_mask_selector(2);});
        const auto args=owner.pi05_owner_norm_arguments();
        assert(args[1]==rows && args[4]==rows/8 && args[5]==rows*512 && args[6]==rows*4096);
        const auto gate=owner.pi05_prefill_gateup_pair_arguments();
        const auto selectedGate=rpu_pi_wall::select(rpu_pi_wall::Operation::PairedGelu,
            fp16?rpu_pl_tiling::WeightType::Fp16:rpu_pl_tiling::WeightType::Int8,false,rows,2048,2048);
        assert(gate==std::vector<int64_t>({fp16?6:5,rows,2048,16384,2048,8,
            selectedGate.tile.m_tile,selectedGate.tile.n_tile,selectedGate.tile.k_tile,selectedGate.grid_x,
            4,1,0,2,3,0,4,48,50,1,1,1,selectedGate.grid_y,52,2,32,20,34,26,27}));
        if(rows==288) {
            assert(owner.pi05_prefill_kv1_pair_kernel()==(fp16
                ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::KvOwner, rpu_pl_tiling::WeightType::Fp16, false, 288, 32, 2048).ordinal)
                : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::KvOwner, rpu_pl_tiling::WeightType::Int8, false, 288, 32, 2048).ordinal)));
            assert(owner.pi05_prefill_gateup_pair_kernel()==(fp16
                ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedGelu, rpu_pl_tiling::WeightType::Fp16, false, 288, 2048, 2048).ordinal)
                : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedGelu, rpu_pl_tiling::WeightType::Int8, false, 288, 2048, 2048).ordinal)));
            const auto direct=owner.pi05_prefill_kv1_pair_direct_policy_arguments();
            assert(direct[1]==576 && direct[2]==576 && direct[6]==288);
            assert(direct[27]==static_cast<int64_t>(owner.pi05_prefill_kv1_pair_kernel()));
            assert(direct[28]==static_cast<int64_t>(KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M288D256P288));
            auto& cache=KernelCache::instance();
            // Missing any required C288 payload must reject the cold profile;
            // the similarly named C272/C416 payloads cannot substitute for it.
            for(const auto id:{
                    KernelId::PI05_RING_XOR3_M288N2048,
                    KernelId::PI05_OWNER_NORM_FULL_RESIDUAL_M288N2048,
                    KernelId::PI05_OWNER_NORM_COMPACT_RESIDUAL_M288N2048,
                    KernelId::PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M288N2048,
                    KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M288D256P288,
                    fp16 ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Fp16, false, 288, 2048, 256).ordinal)
                         : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Int8, false, 288, 2048, 256).ordinal),
                    fp16 ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Fp16, false, 288, 2048, 2048).ordinal)
                         : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Int8, false, 288, 2048, 2048).ordinal),
                    owner.pi05_prefill_kv1_pair_kernel(),owner.pi05_prefill_gateup_pair_kernel()}) {
                cache.missing.insert(id);assert(!owner.pi05_prefill_dma_geometry());cache.missing.clear();
            }
            cache.missing.insert(KernelId::PI05_OWNER_NORM_COMPACT_A8_M288N2048);
            assert(owner.pi05_prefill_dma_geometry());cache.missing.clear();
        }
        if(rows==304) {
            assert(owner.pi05_prefill_kv1_pair_kernel()==(fp16
                ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::KvOwner, rpu_pl_tiling::WeightType::Fp16, false, 304, 32, 2048).ordinal)
                : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::KvOwner, rpu_pl_tiling::WeightType::Int8, false, 304, 32, 2048).ordinal)));
            assert(owner.pi05_prefill_gateup_pair_kernel()==(fp16
                ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedGelu, rpu_pl_tiling::WeightType::Fp16, false, 304, 2048, 2048).ordinal)
                : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedGelu, rpu_pl_tiling::WeightType::Int8, false, 304, 2048, 2048).ordinal)));
            const auto direct=owner.pi05_prefill_kv1_pair_direct_policy_arguments();
            assert(direct[1]==608 && direct[2]==608 && direct[6]==304);
            assert(direct[27]==static_cast<int64_t>(owner.pi05_prefill_kv1_pair_kernel()));
            assert(direct[28]==static_cast<int64_t>(KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M304D256P304));
            auto& cache=KernelCache::instance();
            // Missing any required C304 payload must reject the cold profile;
            // the similarly named C272/C416 payloads cannot substitute for it.
            for(const auto id:{
                    KernelId::PI05_RING_XOR3_M304N2048,
                    KernelId::PI05_OWNER_NORM_FULL_RESIDUAL_M304N2048,
                    KernelId::PI05_OWNER_NORM_COMPACT_RESIDUAL_M304N2048,
                    KernelId::PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M304N2048,
                    KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M304D256P304,
                    fp16 ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Fp16, false, 304, 2048, 256).ordinal)
                         : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Int8, false, 304, 2048, 256).ordinal),
                    fp16 ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Fp16, false, 304, 2048, 2048).ordinal)
                         : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Int8, false, 304, 2048, 2048).ordinal),
                    owner.pi05_prefill_kv1_pair_kernel(),owner.pi05_prefill_gateup_pair_kernel()}) {
                cache.missing.insert(id);assert(!owner.pi05_prefill_dma_geometry());cache.missing.clear();
            }
            cache.missing.insert(KernelId::PI05_OWNER_NORM_COMPACT_A8_M304N2048);
            assert(owner.pi05_prefill_dma_geometry());cache.missing.clear();
        }
        if(rows==320) {
            assert(owner.pi05_prefill_kv1_pair_kernel()==(fp16
                ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::KvOwner, rpu_pl_tiling::WeightType::Fp16, false, 320, 32, 2048).ordinal)
                : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::KvOwner, rpu_pl_tiling::WeightType::Int8, false, 320, 32, 2048).ordinal)));
            assert(owner.pi05_prefill_gateup_pair_kernel()==(fp16
                ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedGelu, rpu_pl_tiling::WeightType::Fp16, false, 320, 2048, 2048).ordinal)
                : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedGelu, rpu_pl_tiling::WeightType::Int8, false, 320, 2048, 2048).ordinal)));
            const auto direct=owner.pi05_prefill_kv1_pair_direct_policy_arguments();
            assert(direct[1]==640 && direct[2]==640 && direct[6]==320);
            assert(direct[27]==static_cast<int64_t>(owner.pi05_prefill_kv1_pair_kernel()));
            assert(direct[28]==static_cast<int64_t>(KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M320D256P320));
            auto& cache=KernelCache::instance();
            // Missing any required C320 payload must reject the cold profile;
            // the similarly named C272/C416 payloads cannot substitute for it.
            for(const auto id:{
                    KernelId::PI05_RING_XOR3_M320N2048,
                    KernelId::PI05_OWNER_NORM_FULL_RESIDUAL_M320N2048,
                    KernelId::PI05_OWNER_NORM_COMPACT_RESIDUAL_M320N2048,
                    KernelId::PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M320N2048,
                    KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M320D256P320,
                    fp16 ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Fp16, false, 320, 2048, 256).ordinal)
                         : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Int8, false, 320, 2048, 256).ordinal),
                    fp16 ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Fp16, false, 320, 2048, 2048).ordinal)
                         : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Int8, false, 320, 2048, 2048).ordinal),
                    owner.pi05_prefill_kv1_pair_kernel(),owner.pi05_prefill_gateup_pair_kernel()}) {
                cache.missing.insert(id);assert(!owner.pi05_prefill_dma_geometry());cache.missing.clear();
            }
            cache.missing.insert(KernelId::PI05_OWNER_NORM_COMPACT_A8_M320N2048);
            assert(owner.pi05_prefill_dma_geometry());cache.missing.clear();
        }
        if(rows==432) {
            assert(owner.pi05_prefill_kv1_pair_kernel()==(fp16
                ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::KvOwner, rpu_pl_tiling::WeightType::Fp16, false, 432, 32, 2048).ordinal)
                : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::KvOwner, rpu_pl_tiling::WeightType::Int8, false, 432, 32, 2048).ordinal)));
            assert(owner.pi05_prefill_gateup_pair_kernel()==(fp16
                ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedGelu, rpu_pl_tiling::WeightType::Fp16, false, 432, 2048, 2048).ordinal)
                : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedGelu, rpu_pl_tiling::WeightType::Int8, false, 432, 2048, 2048).ordinal)));
            const auto direct=owner.pi05_prefill_kv1_pair_direct_policy_arguments();
            assert(direct[1]==864 && direct[2]==864 && direct[6]==432);
            assert(direct[27]==static_cast<int64_t>(owner.pi05_prefill_kv1_pair_kernel()));
            assert(direct[28]==static_cast<int64_t>(KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M432D256P432));
            auto& cache=KernelCache::instance();
            // Missing any required C432 payload must reject the cold profile;
            // the similarly named C272/C288/C416 payloads cannot substitute for it.
            for(const auto id:{
                    KernelId::PI05_RING_XOR3_M432N2048,
                    KernelId::PI05_OWNER_NORM_FULL_RESIDUAL_M432N2048,
                    KernelId::PI05_OWNER_NORM_COMPACT_RESIDUAL_M432N2048,
                    KernelId::PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M432N2048,
                    KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M432D256P432,
                    fp16 ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Fp16, false, 432, 2048, 256).ordinal)
                         : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Int8, false, 432, 2048, 256).ordinal),
                    fp16 ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Fp16, false, 432, 2048, 2048).ordinal)
                         : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Int8, false, 432, 2048, 2048).ordinal),
                    owner.pi05_prefill_kv1_pair_kernel(),owner.pi05_prefill_gateup_pair_kernel()}) {
                cache.missing.insert(id);assert(!owner.pi05_prefill_dma_geometry());cache.missing.clear();
            }
            cache.missing.insert(KernelId::PI05_OWNER_NORM_COMPACT_A8_M432N2048);
            assert(owner.pi05_prefill_dma_geometry());cache.missing.clear();
        }
        if(rows==448) {
            assert(owner.pi05_prefill_kv1_pair_kernel()==(fp16
                ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::KvOwner, rpu_pl_tiling::WeightType::Fp16, false, 448, 32, 2048).ordinal)
                : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::KvOwner, rpu_pl_tiling::WeightType::Int8, false, 448, 32, 2048).ordinal)));
            assert(owner.pi05_prefill_gateup_pair_kernel()==(fp16
                ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedGelu, rpu_pl_tiling::WeightType::Fp16, false, 448, 2048, 2048).ordinal)
                : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedGelu, rpu_pl_tiling::WeightType::Int8, false, 448, 2048, 2048).ordinal)));
            const auto direct=owner.pi05_prefill_kv1_pair_direct_policy_arguments();
            assert(direct[1]==896 && direct[2]==896 && direct[6]==448);
            assert(direct[27]==static_cast<int64_t>(owner.pi05_prefill_kv1_pair_kernel()));
            assert(direct[28]==static_cast<int64_t>(KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M448D256P448));
            auto& cache=KernelCache::instance();
            // Missing any required C448 payload must reject the cold profile;
            // the similarly named C272/C288/C416 payloads cannot substitute for it.
            for(const auto id:{
                    KernelId::PI05_RING_XOR3_M448N2048,
                    KernelId::PI05_OWNER_NORM_FULL_RESIDUAL_M448N2048,
                    KernelId::PI05_OWNER_NORM_COMPACT_RESIDUAL_M448N2048,
                    KernelId::PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M448N2048,
                    KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M448D256P448,
                    fp16 ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Fp16, false, 448, 2048, 256).ordinal)
                         : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Int8, false, 448, 2048, 256).ordinal),
                    fp16 ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Fp16, false, 448, 2048, 2048).ordinal)
                         : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinear, rpu_pl_tiling::WeightType::Int8, false, 448, 2048, 2048).ordinal),
                    owner.pi05_prefill_kv1_pair_kernel(),owner.pi05_prefill_gateup_pair_kernel()}) {
                cache.missing.insert(id);assert(!owner.pi05_prefill_dma_geometry());cache.missing.clear();
            }
            cache.missing.insert(KernelId::PI05_OWNER_NORM_COMPACT_A8_M448N2048);
            assert(owner.pi05_prefill_dma_geometry());cache.missing.clear();
        }
        FmbThreeStageChunkPlan plan;plan.chunk_mode=ChunkMode::KV_FIRST;
        plan.compute.chunks={{0,0,rows,rows},{1,rows,rows,2*rows}};plan.qkv.chunks=plan.compute.chunks;
        LayoutContext layout;layout.chunk_size=layout.kv_insert_chunk_size=rows;
        layout.max_kv_seq_len=2*rows;layout.batch_size=1;layout.use_attn_mask=true;
        layout.is_causal=false;layout.attention_policy=AttentionExecutionPolicy::DDR_KV;
        assert(owner.pi05_prefill_mask_domain_eligible(plan,layout,2*rows,0));
        auto wrong=layout;wrong.max_kv_seq_len=288;
        assert(!owner.pi05_prefill_mask_domain_eligible(plan,wrong,2*rows,0));
        assert(!owner.pi05_prefill_mask_domain_eligible(plan,layout,2*rows,16));
        wrong=layout;wrong.chunk_size=rows==272?400:272;
        assert(!owner.pi05_prefill_dma_layout(wrong));
        auto& weight=owner.layer_weights_.back();
        const auto saved=weight.up_ws;
        weight.up_ws=fp16?at::ones({16384},at::TensorOptions().dtype(at::kHalf)):at::Tensor();
        assert(!owner.pi05_prefill_dma_geometry());weight.up_ws=saved;
        const auto old_weight=weight.down_proj_w;
        weight.down_proj_w=at::empty({2048,16384},at::TensorOptions().dtype(fp16?at::kChar:at::kHalf));
        assert(!owner.pi05_prefill_dma_geometry());weight.down_proj_w=old_weight;
        if(rows!=400 || fp16) {
            auto& cache=KernelCache::instance();cache.missing.insert(owner.pi05_prefill_kv1_pair_kernel());
            assert(!owner.pi05_prefill_dma_geometry());cache.missing.clear();
            owner.pi05_prefill_gateup_pair_=false;
            assert(!owner.pi05_prefill_dma_geometry());owner.pi05_prefill_gateup_pair_=true;
        }
        if(rows==288 || rows==304 || rows==320 || rows==416 || rows==432 || rows==448) {
            owner.pi05_prefill_o_pair_=false;
            assert(!owner.pi05_pair_scratch_mask());
            assert(!owner.pi05_prefill_dma_geometry());owner.pi05_prefill_o_pair_=true;
            if(!fp16) {
                owner.pi05_prefill_a8_=true;
                assert(owner.pi05_prefill_gateup_pair_kernel()==
                    (rows==432 ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::A8PairedGelu, rpu_pl_tiling::WeightType::Int8, true, 432, 2048, 2048).ordinal)
                               : rows==448 ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::A8PairedGelu, rpu_pl_tiling::WeightType::Int8, true, 448, 2048, 2048).ordinal)
                               : rows==288 ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::A8PairedGelu, rpu_pl_tiling::WeightType::Int8, true, 288, 2048, 2048).ordinal)
                               : rows==304 ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::A8PairedGelu, rpu_pl_tiling::WeightType::Int8, true, 304, 2048, 2048).ordinal)
                               : rows==320 ? pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::A8PairedGelu, rpu_pl_tiling::WeightType::Int8, true, 320, 2048, 2048).ordinal)
                               : pi_wall_kernel_id(rpu_pi_wall::select(rpu_pi_wall::Operation::A8PairedGelu, rpu_pl_tiling::WeightType::Int8, true, 416, 2048, 2048).ordinal)));
                assert(owner.pi05_prefill_dma_geometry());
                auto& cache=KernelCache::instance();
                cache.missing.insert(rows==432 ? KernelId::PI05_OWNER_NORM_COMPACT_A8_M432N2048
                                               : rows==448 ? KernelId::PI05_OWNER_NORM_COMPACT_A8_M448N2048
                                               : rows==288 ? KernelId::PI05_OWNER_NORM_COMPACT_A8_M288N2048
                                               : rows==304 ? KernelId::PI05_OWNER_NORM_COMPACT_A8_M304N2048
                               : rows==320 ? KernelId::PI05_OWNER_NORM_COMPACT_A8_M320N2048
                                               : KernelId::PI05_OWNER_NORM_COMPACT_A8_M416N2048);
                assert(!owner.pi05_prefill_dma_geometry());cache.missing.clear();
                cache.missing.insert(owner.pi05_prefill_gateup_pair_kernel());
                assert(!owner.pi05_prefill_dma_geometry());cache.missing.clear();
                owner.pi05_prefill_a8_=false;
            }
        }
        if(fp16) {
            owner.pi05_prefill_a8_=true;
            assert(!owner.pi05_prefill_dma_geometry());owner.pi05_prefill_a8_=false;
        } else {
            owner.pi05_prefill_a8_=true;
            const auto a8=owner.pi05_prefill_gateup_pair_arguments();
            assert(a8.size()==32 && a8[0]==7 && a8[1]==rows && a8[6]==48);
            assert(a8[19]==2080 && a8[20]==(rows<400?256:384));
            assert(a8[21]==owner.pi05_a8_root_bytes() && a8[22]==rows*2*2080);
            if(rows==304) {
                assert(a8[21]==1265920 && a8[22]==1264640);
                assert(a8[21]-a8[22]-608*2>=32);
                assert(!owner.pi05_pair_scratch_mask());
                assert(owner.pi05_pair_mask_selector(1)==2);
            }
            if(rows==320) {
                assert(a8[21]==1332736 && a8[22]==1331200);
                assert(a8[21]-a8[22]-640*2>=32);
                assert(!owner.pi05_pair_scratch_mask());
                assert(owner.pi05_pair_mask_selector(1)==2);
            }
            if(rows==288) {
                assert(a8[21]==1199360 && a8[22]==1198080);
                // Both complete M48 tiles and short tiles can read30B beyond
                // the last scale; the declared root also fits its32B producer.
                assert(a8[21]-a8[22]-576*2>=32);
                assert(!owner.pi05_pair_scratch_mask());
                assert(owner.pi05_pair_mask_selector(1)==2);
            }
            owner.pi05_prefill_a8_=false;
        }
        assert(owner.pi05_prefill_dma_geometry());
    }
    for(bool fp16:{false,true}) for(int64_t rows:{336,464}) {
        Owner owner;owner.set_prefill_pair_rows(rows);owner.install(fp16);
        for(bool a8:{false,true}) {
            if(fp16 && a8) continue;
            owner.pi05_prefill_a8_=a8;
            assert(owner.pi05_prefill_dma_geometry());
            assert(owner.pi05_prefix_rows()==2*rows);
            assert(owner.pi05_pair_tile_rows()==(rows==464?80:160));
            auto& cache=KernelCache::instance();
            cache.missing.insert(owner.pi05_prefill_gateup_pair_kernel());
            assert(!owner.pi05_prefill_dma_geometry());cache.missing.clear();
            const auto args=owner.pi05_prefill_gateup_pair_arguments();
            if(a8) assert(args[20]==(rows==464?448:320));
        }
    }
    // C480 (three-camera T192, P960) is the C464 layout at 480 rows: FP16
    // and W8A16 only, each of its five per-row payloads required; no A8.
    for(bool fp16:{false,true}) {
        Owner owner;owner.set_prefill_pair_rows(480);owner.install(fp16);
        assert(owner.pi05_prefix_rows()==960);
        assert(owner.pi05_pair_single_compact() && owner.pi05_pair_slab_compact());
        assert(owner.pi05_pair_scratch_mask());
        assert(owner.pi05_prefill_dma_geometry());
        auto& cache=KernelCache::instance();
        for(KernelId id:{KernelId::PI05_RING_XOR3_M480N2048,
                KernelId::PI05_OWNER_NORM_FULL_RESIDUAL_M480N2048,
                KernelId::PI05_OWNER_NORM_COMPACT_RESIDUAL_M480N2048,
                KernelId::PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M480N2048,
                KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M480D256P480,
                owner.pi05_prefill_kv1_pair_kernel(),owner.pi05_prefill_gateup_pair_kernel()}) {
            cache.missing.insert(id);assert(!owner.pi05_prefill_dma_geometry());cache.missing.clear();
        }
        if(!fp16) {
            owner.pi05_prefill_a8_=true;assert(!owner.pi05_prefill_dma_geometry());
            owner.pi05_prefill_a8_=false;
        }
        assert(owner.pi05_prefill_dma_geometry());
    }
    std::cout<<"Exact C272/C288/C304/C320/C400/C416/C432/C448/C480 FP16/W8 paired profile and mask guards: PASS\n";
}
'''
    result = _run_native(tmp_path, code, torch,
                         extra_sources=(ROOT / "src/core/fmb_three_stage_chunk_plan.cpp",))
    assert "Exact C272/C288/C304/C320/C400/C416/C432/C448/C480 FP16/W8 paired profile and mask guards: PASS" in result


def test_prefill_pair_cold_validator_uses_real_fp16_and_w8_inverse_swizzles(tmp_path, monkeypatch):
    monkeypatch.setenv('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
    import torch
    source=(ROOT/'src/ops/rpu_linear.cpp').read_text()
    swizzle=_function(source,'at::Tensor tp_col_swizzle_mc_weight(')
    validator=_function(source,'void rpu_validate_pi05_prefill_kv1_pair_owner_w8a16(').replace(
        'at::kPrivateUse1','at::kCPU')
    code=r'''
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cassert>
#include <cstring>
#include <map>
struct RpuKernelGraph {
 static bool active;
 static bool has_active(){return active;}
};
bool RpuKernelGraph::active=false;
uint64_t RpuGetDevAddr(const void* p){
 static std::map<const void*,uint64_t> pointers;
 auto it=pointers.find(p);if(it!=pointers.end())return it->second;
 uint64_t address=0x82000000ULL+pointers.size()*0x1000000ULL;
 pointers.emplace(p,address);return address;
}
'''+swizzle+'\n'+validator+r'''
template<class F>void rejects(F f){bool caught=false;try{f();}catch(const c10::Error&){caught=true;}assert(caught);}
int main(){
 at::set_num_threads(1);
 for(bool half:{false,true}){
  auto dtype=half?at::kHalf:at::kChar;
  auto logical=at::randint(-10,11,{256,2048},at::kInt).to(dtype);
  auto k=tp_col_swizzle_mc_weight(logical.repeat({8,1}),8,half?2:1);
  auto v=tp_col_swizzle_mc_weight((logical+1).repeat({8,1}),8,half?2:1);
  at::Tensor ks,vs;
  if(!half){ks=at::ones({2048},at::kHalf);vs=at::ones({2048},at::kHalf);}
  rpu_validate_pi05_prefill_kv1_pair_owner_w8a16(k,ks,v,vs);
  auto broken=logical.repeat({8,1});broken.index({700,341}).fill_(33);
  broken=tp_col_swizzle_mc_weight(broken,8,half?2:1);
  rejects([&]{rpu_validate_pi05_prefill_kv1_pair_owner_w8a16(broken,ks,v,vs);});
  rejects([&]{rpu_validate_pi05_prefill_kv1_pair_owner_w8a16(k,ks,k,ks);});
  RpuKernelGraph::active=true;
  rejects([&]{rpu_validate_pi05_prefill_kv1_pair_owner_w8a16(k,ks,v,vs);});
  RpuKernelGraph::active=false;
  if(half){
   rejects([&]{rpu_validate_pi05_prefill_kv1_pair_owner_w8a16(k,at::ones({2048},at::kHalf),v,vs);});
   rejects([&]{rpu_validate_pi05_prefill_kv1_pair_owner_w8a16(k,ks,v.to(at::kChar),vs);});
  }else{
   auto badscale=ks.clone();badscale.index({800}).fill_(2);
   rejects([&]{rpu_validate_pi05_prefill_kv1_pair_owner_w8a16(k,badscale,v,vs);});
   rejects([&]{rpu_validate_pi05_prefill_kv1_pair_owner_w8a16(k,at::Tensor(),v,vs);});
  }
 }
}
'''
    _run_native(tmp_path,code,torch)
