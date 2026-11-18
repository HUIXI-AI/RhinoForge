"""Gemma KV_FIRST planning and live dispatch agree on the full KV window."""
from pathlib import Path
import re
from test_siglip_attention_dry_live_contract import _function, _run_native

ROOT = Path(__file__).resolve().parents[1]


def test_gemma_kv_first_routes_use_complete_physical_window(tmp_path, monkeypatch):
    # Compile the actual owner predicate, manifest attention append/reasons and
    # live exact consumption with the real FMB planner and strict consumer.
    # CPU-only shape state supplies geometry; no backend or model is loaded.
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    S = ROOT
    function = _function
    src = (S / 'src/fused/rpu_gemma_model.cpp').read_text()
    rh = (S / 'src/core/rpu_helpers.h').read_text()
    predicate = function(src, '    bool subclass_spm_kv_by_mha_eligible(').replace(' const override', ' const')
    accumulation = function(src, '    FmbLinearAccumulationPolicy pi05_prefill_linear_accumulation(')
    manifest = function(src, '    FmbPhysicalExecutionManifest physical_manifest_for_candidate(')
    # Keep the real traversal guard and generic attention state; Pi-specific
    # cold owner guards are exercised by the optimized-profile tests.
    # The full-chunk owner now has an earlier, separate cold guard. Select the
    # actual generic traversal check rather than the first TORCH_CHECK token.
    guard_start = manifest.index('        TORCH_CHECK(\n            plan.chunk_mode == ChunkMode::KV_FIRST,')
    guard = manifest[guard_start:manifest.index(';', guard_start) + 1] + '\n'
    prefix = guard + manifest[manifest.index('        FmbPhysicalExecutionManifest manifest;'):manifest.index('        if (num_cores() != 8)')]
    start = manifest.index('            append(\n                FmbRouteFamily::ATTENTION,')
    append = manifest[start:manifest.index(';', start) + 1]
    # Mutate only the historical KV-window error for a native negative control.
    assert append.count('position + physical_len') == 1
    new_append = append
    append = append.replace('position + physical_len', 'chunk.kv_seq_len')
    live = function(src, '    void build_layer_subgraph(')
    start = live.index('        int64_t total_kv_seq_len =')
    total = live[start:live.index('\n', start)]
    start = live.index('        const bool raw_spm =')
    context = live[start:live.index('        if (raw_spm) {', start)]
    start = live.index('            ctx().consume_physical_route(', live.index('        if (raw_spm) {'))
    raw_consume = live[start:live.index(';', start)+1]
    start = live.index('            ctx().consume_physical_route(', live.index('        } else {', start))
    ddr_consume = live[start:live.index(';', start)+1]
    constants = '\n'.join(line for line in src.splitlines() if line.startswith('constexpr int64_t GEMMA_') and 'ATTENTION_SITE' in line)
    flags = function(src, 'enum GemmaAttentionRouteFlags') + ';'
    capability = function(rh, 'inline uint32_t GetMhaAttnUnivVlmNeed(') + '\n' + function(rh, 'inline bool sdpa_by_mha_spm_is_valid(')
    vlm_limit, = re.findall(r'^#define SDPA_MAX_VLM_SIZE .+$', rh, re.M)
    cpp = r'''
    #include "core/fused_model_base.h"
    #include "core/rpu_profile.h"
    #include <cassert>
    #include <fstream>
    #include <iostream>
    #include <limits>
    using namespace v3;
    ''' + vlm_limit + '\n' + capability + '\n' + constants + '\n' + flags + r'''
    struct HostContext {
        int64_t position=0;
        AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::DDR_KV;
        FmbPhysicalManifestConsumer consumer;
        const FmbRouteManifestEntry& find_physical_route(FmbRouteFamily f,int64_t s,int64_t i) const {
            return consumer.find_route(f,s,i);
        }
        const FmbRouteManifestEntry& consume_physical_route(FmbRouteFamily f,int64_t s,int64_t v,int64_t flags,at::IntArrayRef a,int64_t i) const {
            return consumer.consume_route(f,s,v,flags,a,i);
        }
    };
    struct Mask { int64_t mask_type=4; };
    struct Owner {
        int cores=4; int64_t seq_len_=0;
        bool pi05_prefill_a8_=false,linear_acc32_=false;
        std::vector<Mask> chunk_masks_;
        HostContext state;
        const HostContext& ctx() const { return state; }
        int64_t num_q_heads() const { return 8; }
        int64_t num_kv_heads() const { return cores==8?8:4; }
        int64_t head_dim() const { return 256; }
        int attn_tp() const { return cores==8?8:4; }
    ''' + predicate + accumulation
    for name, ap in [('old_dry', append), ('new_dry', new_append)]:
        cpp += '''
        FmbPhysicalExecutionManifest ''' + name + '''(const FmbThreeStageChunkPlan& plan,const LayoutContext& layout,int64_t physical_len,int64_t logical_len,int64_t position) const {
    ''' + prefix + '''
            for (const ChunkInfo& chunk : plan.compute.chunks) {
    ''' + ap + '''
            }
            validate_fmb_physical_manifest(manifest,physical_len,position);
            return manifest;
        }
    '''
    cpp += r'''
        void dispatch(const ChunkInfo& chunk) const {
            const int64_t seq_len=chunk.len,nq=num_q_heads(),nkv=num_kv_heads(),hd=head_dim(),tp=attn_tp();
    ''' + total + '\n' + context + '\n        if (raw_spm) {\n' + raw_consume + '\n        } else {\n' + ddr_consume + r'''
            }
        }
    };
    FmbThreeStageChunkPlan plan(int64_t length,int64_t chunk_size,int64_t position) {
        std::vector<ChunkInfo> chunks;
        for (int64_t off=0,i=0;off<length;off+=chunk_size,++i) {
            auto len=std::min(chunk_size,length-off);
            chunks.push_back({static_cast<int>(i),off,len,position+off+len});
        }
        return compose_fmb_three_stage_chunk_plan(chunks,chunks,chunks,{{0,length,0}},
            ChunkMode::KV_FIRST,{FmbSpanBoundaryPolicy::ALLOW_CROSS,
            FmbSpanBoundaryPolicy::ALLOW_CROSS,FmbSpanBoundaryPolicy::ALLOW_CROSS});
    }
    bool rejects(const Owner& owner,const ChunkInfo& chunk) {
        try { owner.dispatch(chunk); }
        catch (const c10::Error& e) {
            return std::string(e.what_without_backtrace()).find("EXACT_MISMATCH")!=std::string::npos;
        }
        return false;
    }
    int main() {
        std::ifstream maps("/proc/self/maps");
        const std::string loaded((std::istreambuf_iterator<char>(maps)),{});
        assert(loaded.find("rpu_backend.so")==std::string::npos);
        assert(loaded.find("librhino")==std::string::npos);
        for(bool acc32:{false,true}) {
        int cases=0,old_rejected_chunks=0,new_consumed_chunks=0,raw_cases=0;
        for(int cores:{4,6,8}) for(int64_t length:{16,64,200,768,968})
        for(int64_t chunk_size:{64,256,1024}) for(int64_t position:{0,32,768})
        for(int64_t stale:{0,16,1024}) for(int64_t padding:{0,7}) {
            Owner owner; owner.cores=cores; owner.seq_len_=stale; owner.state.position=position;
            owner.linear_acc32_=acc32;
            auto stage=plan(length,chunk_size,position);
            LayoutContext layout;
            layout.chunk_size=stage.compute.chunks[0].len;
            layout.kv_insert_chunk_size=layout.chunk_size;
            layout.max_kv_seq_len=position+length;
            layout.use_attn_mask=true;layout.is_causal=false;
            auto old=owner.old_dry(stage,layout,length,length-padding,position);
            auto fixed=owner.new_dry(stage,layout,length,length-padding,position);
            const auto accumulation=acc32 ? FmbLinearAccumulationPolicy::ACC32
                                          : FmbLinearAccumulationPolicy::ACC16;
            assert(old.linear_accumulation==accumulation);
            assert(fixed.linear_accumulation==accumulation);
            owner.chunk_masks_.resize(stage.compute.chunks.size());
            owner.seq_len_=length; // Actual forward publication, after cold oracle.
            for(size_t i=0;i<stage.compute.chunks.size();++i) {
                auto& chunk=stage.compute.chunks[i];
                assert(old.routes[i].flags==fixed.routes[i].flags);
                assert(old.routes[i].selector==fixed.routes[i].selector);
                assert(old.routes[i].arguments[2]==position+chunk.offset+chunk.len);
                assert(fixed.routes[i].arguments[2]==position+length);
                owner.state.consumer.reset(old);
                bool old_reject=rejects(owner,chunk);
                assert(old_reject==(i+1<stage.compute.chunks.size()));
                old_rejected_chunks+=old_reject;
                owner.state.consumer.reset(fixed);
                owner.dispatch(chunk);++new_consumed_chunks;
                auto bad=fixed;
                ++bad.routes[i].arguments[2];
                owner.state.consumer.reset(bad);assert(rejects(owner,chunk));
                bad=fixed;bad.routes[i].selector=static_cast<int64_t>(AttentionExecutionPolicy::SPM_KV_BY_MHA);
                owner.state.consumer.reset(bad);assert(rejects(owner,chunk));
            }
            owner.state.consumer.reset(fixed);
            for(const auto& chunk:stage.compute.chunks) owner.dispatch(chunk);
            owner.state.consumer.require_all_consumed();
            // A legal raw single chunk consumes the same full physical window.
            layout.attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;
            if(owner.subclass_spm_kv_by_mha_eligible(stage,layout,position)) {
                auto raw=owner.new_dry(stage,layout,length,length-padding,position);
                assert(raw.routes.size()==1 && raw.routes[0].flags==0);
                assert(raw.routes[0].arguments[2]==position+length);
                owner.state.attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;
                owner.state.consumer.reset(raw);
                owner.dispatch(stage.compute.chunks.front());++raw_cases;
            }
            // Gemma supports KV_FIRST exclusively. Its real manifest guard still
            // refuses the other planner traversal; it is not a fallback profile.
            stage.chunk_mode=ChunkMode::SEQUENTIAL;
            bool reject_mode=false;
            try { (void)owner.new_dry(stage,layout,length,length-padding,position); }
            catch(const c10::Error& e) {reject_mode=std::string(e.what_without_backtrace()).find("KV_FIRST traversal")!=std::string::npos;}
            assert(reject_mode);
            ++cases;
        }
        assert(raw_cases>0);
        std::cout<<"acc="<<(acc32?32:16)<<" raw_cases="<<raw_cases<<" cases="<<cases<<" old_nonfinal_exact_mismatch="<<old_rejected_chunks
            <<" fixed_consumed_chunks="<<new_consumed_chunks<<"\n";
        }
    }
    '''
    output = _run_native(tmp_path, cpp, torch, extra_sources=(
        S / "src/core/fmb_three_stage_chunk_plan.cpp",))
    assert "acc=16 raw_cases=" in output and "acc=32 raw_cases=" in output
    assert output.count("cases=810") == 2
    assert output.count("old_nonfinal_exact_mismatch=1836") == 2
    assert output.count("fixed_consumed_chunks=2646") == 2
