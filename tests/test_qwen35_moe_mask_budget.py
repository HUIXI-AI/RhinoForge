"""CPU regression: real MoE declarations, FMB capacity predicate and typed route.

Tensor/DMA and allocator watermarks are facades; layout arithmetic, first-fit
placement, cold selection, route binding and per-forward schedule are native.
"""
from pathlib import Path
import json
import runpy
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def source_probe():
    helpers = runpy.run_path(str(ROOT / 'tests/test_qwen35_gdn_mask_residency.py'))
    definition = helpers['definition']
    code = helpers['source_probe']().split('\nint main() {', 1)[0]
    base = (ROOT / 'src/core/fused_model_base.h').read_text()
    base_cpp = (ROOT / 'src/core/fused_model_base.cpp').read_text()
    native = (ROOT / 'src/fused/rpu_qwen3_5_moe_model.cpp').read_text()
    wire = '\n'.join(definition(base, marker) + ';' for marker in (
        'enum class FmbRouteFamily', 'enum class FmbPhysicalManifestState',
        'enum class FmbGraphLifecycle', 'enum class FmbLinearAccumulationPolicy',
        'enum class FmbPrefillRankingPolicy', 'struct FmbPrefillRanking',
        'struct FmbRouteManifestEntry', 'struct FmbPhysicalExecutionManifest'))
    code = code.replace('enum class FmbRouteFamily {MUTABLE_DMA,GRAPH_SCHEDULE};', wire)
    shim = definition(code, 'class Qwen3_5MoeModel :') + ';'
    new_shim = shim.replace('    std::vector<BufferDecl> declare_buffers(const LayoutContext&, int64_t sdpa_workspace_bytes = -1);', '''
    std::vector<BufferDecl> declare_buffers(const LayoutContext&);
    bool resident_prefill_ = false;
    Tensor resident_route_ordinals_{512,32768,true},resident_route_tokens_{512,32768,true};
    int64_t resident_router_capacity() const;
    bool resident_route_capacity_valid(int64_t) const;
    using ResidentMixerPlan = qwen35_moe_planning::Plan;
    mutable bool resident_planning_active_ = false;
    mutable int64_t resident_planning_rows_ = 0, resident_planning_logical_rows_ = 0;
    mutable std::map<int64_t, std::optional<ResidentMixerPlan>> resident_plans_;
    std::optional<ResidentMixerPlan> resident_execution_plan_;
    uint64_t resident_execution_fingerprint_ = 0;
    mutable std::optional<ResidentMixerPlan> resident_detached_plan_;
    mutable uint64_t resident_detached_fingerprint_ = 0;
    const ResidentMixerPlan* resident_plan_for_layout(const LayoutContext&) const;
    const ResidentMixerPlan& resident_plan_for_outer(const ChunkInfo&) const;
    bool plan_resident_candidate(int64_t,int64_t,int64_t) const;
    std::vector<ChunkInfo> planned_resident_mixer_chunks(const ChunkInfo&) const;
    std::vector<int64_t> resident_schedule_arguments(const ChunkInfo&) const;
    void restore_resident_plan(const FmbPrefillStageCandidate&);
    ResidentMixerPlan decode_resident_plan(const FmbPrefillStageCandidate&) const;
    FmbPrefillRanking prefill_candidate_ranking(const FmbPrefillStageCandidate&,const LayoutContext&) const;
    std::vector<BufferDecl> declare_legacy_buffers(const LayoutContext&,int64_t=-1) const;
    std::vector<BufferDecl> declare_active_buffers(const LayoutContext&) const;
    std::vector<BufferDecl> declare_resident_buffers(const LayoutContext&) const;
    std::vector<BufferDecl> declare_resident_plan_buffers(const LayoutContext&,int64_t,int64_t) const;
    FmbForwardOperandResidency gdn_mask_residency_for_candidate(const LayoutContext&) const;
    FmbForwardOperandResidency physical_forward_operand_residency(const FmbPhysicalExecutionManifest&) const;
    bool declared_spm_layout_fits(const std::vector<BufferDecl>&) const;
''')
    code = code.replace(shim, new_shim)
    code = '#include "fused/qwen35_moe_mixer_plan.h"\n#include <chrono>\n' + code
    code = code.replace('class FusedModelBase {};', '''
struct FmbPrefillStageCandidate {
    int64_t compute_chunk_size = 0;
    FmbPhysicalExecutionManifest physical_manifest;
    FmbThreeStageChunkPlan stage_plan;
    FmbPrefillRanking ranking;
};
class FusedModelBase {};
''')
    old = definition(code, 'std::vector<BufferDecl> Qwen3_5MoeModel::declare_buffers(')
    code = code.replace(old, definition(native, 'std::vector<BufferDecl> Qwen3_5MoeModel::declare_legacy_buffers('))
    code += '''
#undef SPM_ALLOC
struct Watermarks {
    int64_t free=0, temporary=0;
    int64_t free_space() const {return free;}
    int64_t temporary_used() const {return temporary;}
} watermarks;
#define SPM_ALLOC watermarks
namespace v3 {
'''
    for marker in ('struct FixedSpmOverhead', 'static FixedSpmOverhead estimate_fixed_overhead(',
                   'static int64_t available_temporary_spm_after_reset(', 'static bool final_chunk_layout_fits('):
        code += definition(base_cpp, marker) + (';\n' if marker.startswith('struct') else '\n')
    code += '''
bool own_installed=false; int budget_queries=0;
bool Qwen3_5MoeModel::declared_spm_layout_fits(const std::vector<BufferDecl>& rows) const {
    ++budget_queries; return final_chunk_layout_fits(rows,own_installed);
}
'''
    import re
    code += "\n".join(re.findall(r"constexpr int64_t kResident\w+ = \d+;", native)) + "\n"
    code += "\n".join(re.findall(
        r"constexpr int64_t QWEN35_MOE_RESIDENT_SCHEDULE_SITE = \d+LL;", native)) + "\n"
    helpers_source = (ROOT / 'src/core/rpu_helpers.h').read_text()
    shuffle_source = (ROOT / 'src/ops/rpu_moe_index_shuffle.cpp').read_text()
    code += definition(shuffle_source, 'struct FullKvsortParams') + ';\n'
    code += definition(shuffle_source, 'FullKvsortParams make_full_kvsort_params(') + '\n'
    code += definition(shuffle_source, 'std::pair<int64_t, int64_t> rpu_moe_kvsort_fp16_spm_sizes(') + '\n'
    code = code.replace('\nnamespace v3 { class FusedModelBase;',
        '\n' + definition(helpers_source, 'static inline bool sdpa_call_valid(') +
        '\nnamespace v3 { class FusedModelBase;', 1)
    code += definition(native, 'int64_t resident_mixer_sdpa_tmp_bytes(') + '\n'
    for marker in ('std::vector<BufferDecl> Qwen3_5MoeModel::declare_active_buffers(',
                   'int64_t Qwen3_5MoeModel::resident_router_capacity(',
                   'bool Qwen3_5MoeModel::resident_route_capacity_valid(',
                   'std::vector<BufferDecl> Qwen3_5MoeModel::declare_resident_buffers(',
                   'std::vector<BufferDecl> Qwen3_5MoeModel::declare_resident_plan_buffers(',
                   'const Qwen3_5MoeModel::ResidentMixerPlan*\nQwen3_5MoeModel::resident_plan_for_layout(',
                   'bool Qwen3_5MoeModel::plan_resident_candidate(',
                   'const Qwen3_5MoeModel::ResidentMixerPlan&\nQwen3_5MoeModel::resident_plan_for_outer(',
                   'std::vector<ChunkInfo> Qwen3_5MoeModel::planned_resident_mixer_chunks(',
                   'std::vector<int64_t> Qwen3_5MoeModel::resident_schedule_arguments(',
                   'void Qwen3_5MoeModel::restore_resident_plan(',
                   'Qwen3_5MoeModel::ResidentMixerPlan Qwen3_5MoeModel::decode_resident_plan(',
                   'FmbPrefillRanking Qwen3_5MoeModel::prefill_candidate_ranking(',
                   'std::vector<BufferDecl> Qwen3_5MoeModel::declare_buffers(',
                   'FmbForwardOperandResidency Qwen3_5MoeModel::gdn_mask_residency_for_candidate(',
                   'std::vector<int64_t> Qwen3_5MoeModel::gdn_mask_schedule_arguments(',
                   'FmbForwardOperandResidency Qwen3_5MoeModel::physical_forward_operand_residency('):
        code += definition(native, marker) + '\n'
    code += r'''
void set_watermarks(int64_t held,int64_t persistent,bool warm) {
    own_installed=warm;watermarks.temporary=4096;
    watermarks.free=SpmAllocator::SPM_USABLE-held-(warm?persistent:0)-4096;
}
template<class F> void rejects(F fn) {
    bool rejected=false;try {fn();}catch(const std::runtime_error&){rejected=true;}assert(rejected);
}
}
int main() {
    using namespace v3;
    using Mode=FmbForwardOperandResidency;
    Qwen3_5MoeModel model;
    model.hidden=2048;model.nq=16;model.layers=40;model.gdn_nvh_=32;model.gdn_conv_dim_=8192;
    model.layer_is_full_.resize(40);
    for(int i=0;i<40;++i)model.layer_is_full_[i]=(i%4==3);
    LayoutContext lc;lc.chunk_size=128;lc.num_layers=40;lc.max_kv_seq_len=8192;
    for(int cores:{4,6,8}) {
        model.cores=cores;model.nkv=model.attn_tp();model.moe_.tensor_parallel=cores;
        auto legacy=model.declare_buffers(lc);
        const auto fixed=estimate_fixed_overhead(legacy);
        assert(fixed.persistent==643072 && fixed.temp_per_layer==0);
        const auto trial=detail::make_forward_spm_residency_candidate(legacy,{"gdn_c_tril","gdn_c_strict"});
        assert(trial.valid && trial.retained_peak_bytes-trial.original_peak_bytes==16384);
        for(bool warm:{false,true}) {
            set_watermarks(0,fixed.persistent,warm);
            assert(model.gdn_mask_residency_for_candidate(lc)==Mode::FORWARD);
            const int64_t threshold=SpmAllocator::SPM_USABLE-fixed.persistent-65536-trial.retained_peak_bytes;
            set_watermarks(threshold,fixed.persistent,warm);
            assert(model.gdn_mask_residency_for_candidate(lc)==Mode::FORWARD);
            set_watermarks(threshold+256,fixed.persistent,warm);
            assert(final_chunk_layout_fits(legacy,warm));
            assert(!final_chunk_layout_fits(trial.declarations,warm));
            assert(model.gdn_mask_residency_for_candidate(lc)==Mode::PER_LAYER);
            // Descriptor-bound route never changes under pressure; final FMB
            // admission rejects the now-infeasible retained layout instead.
            lc.forward_operand_residency=Mode::FORWARD;
            budget_queries=0;
            assert(model.gdn_mask_residency_for_candidate(lc)==Mode::FORWARD);
            auto retained=model.declare_buffers(lc);
            assert(budget_queries==0 && !final_chunk_layout_fits(retained,warm));
            lc.forward_operand_residency=Mode::PER_LAYER;
            same_declarations(model.declare_buffers(lc),legacy);
            lc.forward_operand_residency=Mode::UNSPECIFIED;
            same_declarations(model.declare_buffers(lc),legacy);
        }
        std::cout<<"TP"<<cores<<" old="<<trial.original_peak_bytes<<" retained="<<trial.retained_peak_bytes<<" fixed="<<fixed.persistent<<"\n";
    }
    model.cores=8;model.nkv=8;model.moe_.tensor_parallel=8;
    set_watermarks(1363200,643072,false);
    lc.chunk_size=64;lc.planning_chunk_capacity=0;
    assert(model.gdn_mask_residency_for_candidate(lc)==Mode::FORWARD);
    lc.planning_chunk_capacity=128;
    assert(model.gdn_mask_residency_for_candidate(lc)==Mode::PER_LAYER);
    lc.planning_chunk_capacity=0;lc.chunk_size=16;
    lc.forward_operand_residency=Mode::FORWARD;
    rejects([&]{model.declare_buffers(lc);});
    lc.forward_operand_residency=Mode::UNSPECIFIED;lc.chunk_size=192;
    set_watermarks(0,643072,false);
    for(int cores:{4,6,8}) {
        model.cores=cores;model.nkv=model.attn_tp();model.moe_.tensor_parallel=cores;
        for(int rows:{64,128,192,256,512}) {
            lc.chunk_size=rows;
            const auto declarations=model.declare_buffers(lc);
            const int64_t mlp_rows=std::min(rows,128);
            assert(slot(declarations,"moe_shuffled_input").size==mlp_rows*8*2048*2);
            assert(slot(declarations,"moe_router_probs").size==mlp_rows*256*2);
            assert(slot(declarations,"moe_partial").size==mlp_rows*2048*2);
            for(const char* stream:{"residual1","input_norm"}) {
                const auto& d=slot(declarations,stream);
                assert(d.size==rows*2048*2 && d.phase_start==1 && d.phase_end==kMoePhaseReduce);
            }
            std::vector<SpmAllocator::AllocRequest> requests;
            std::vector<const BufferDecl*> roots;
            for(const auto& d:declarations) if(d.storage==StorageClass::Temp && !d.alias_of) {
                requests.push_back({d.size,d.phase_start,d.phase_end,static_cast<int>(d.scope)});
                roots.push_back(&d);
            }
            const auto placement=SpmAllocator::plan_temporary_aliased(requests);
            for(size_t i=0;i<roots.size();++i) {
                const std::string name=roots[i]->name;
                if(name!="residual1" && name!="input_norm")continue;
                for(size_t j=0;j<roots.size();++j) {
                    if(std::string(roots[j]->name).find("moe_")!=0)continue;
                    assert(placement.offsets[i]+roots[i]->size<=placement.offsets[j] ||
                           placement.offsets[j]+roots[j]->size<=placement.offsets[i]);
                }
            }
            const bool fits=final_chunk_layout_fits(declarations,false);
            assert(fits==(rows<=192 || (rows==256 && cores!=4)));
            std::cout<<"MLP128 TP"<<cores<<" C"<<rows<<" peak="
                     <<detail::estimate_temporary_total(declarations)<<" fits="<<fits<<"\n";
        }
    }
    model.cores=8;model.nkv=8;model.moe_.tensor_parallel=8;

    auto& c=model.context;c.stage_plan.compute.plan={128,2};
    c.stage_plan.compute.chunks={{0,0,128,128},{1,128,128,256}};
    c.expected=model.gdn_mask_schedule_arguments(c.stage_plan);
    assert(c.expected==std::vector<int64_t>({64,64,2,0,30,128,2,8,8}));
    FmbPhysicalExecutionManifest manifest;manifest.state=FmbPhysicalManifestState::COMPLETE;
    manifest.physical_length=256;
    manifest.routes.push_back({QWEN35_MOE_GDN_MASK_RESIDENCY_SITE,FmbRouteFamily::GRAPH_SCHEDULE,2,0,c.expected});
    assert(model.physical_forward_operand_residency(manifest)==Mode::FORWARD);
    manifest.routes[0].selector=1;
    assert(model.physical_forward_operand_residency(manifest)==Mode::PER_LAYER);
    for(int field=0;field<9;++field) {
        auto wrong=manifest;++wrong.routes[0].arguments[field];
        rejects([&]{model.physical_forward_operand_residency(wrong);});
    }
    for(int selector:{0,3}) {
        auto wrong=manifest;wrong.routes[0].selector=selector;
        rejects([&]{model.physical_forward_operand_residency(wrong);});
    }
    auto wrong=manifest;wrong.routes.clear();rejects([&]{model.physical_forward_operand_residency(wrong);});
    wrong=manifest;wrong.routes[0].flags=1;rejects([&]{model.physical_forward_operand_residency(wrong);});
    wrong=manifest;wrong.routes[0].invocation=1;rejects([&]{model.physical_forward_operand_residency(wrong);});
    for(int selector:{1,2}) for(int replay=0;replay<3;++replay) {
        c.route.selector=selector;dma_calls=0;
        for(int layer=0;layer<40;++layer)if(!model.layer_is_full_[layer])
            for(const auto& chunk:c.stage_plan.compute.chunks)model.emit_masks(layer,chunk,false);
        assert(dma_calls==(selector==2?2:120));
    }
    c.complete=false;dma_calls=0;
    for(int layer=0;layer<40;++layer)if(!model.layer_is_full_[layer])
        for(const auto& chunk:c.stage_plan.compute.chunks)model.emit_masks(layer,chunk,false);
    assert(dma_calls==120);
    dma_calls=0;model.emit_masks(0,{0,0,1,1},true);assert(dma_calls==0);
    manifest.physical_length=1;manifest.routes.clear();
    assert(model.physical_forward_operand_residency(manifest)==Mode::UNSPECIFIED);
}
'''
    return code


def test_real_moe_budget_and_bound_schedule(tmp_path):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    cpp, exe = tmp_path/'moe.cpp', tmp_path/'moe'
    cpp.write_text(source_probe())
    result = subprocess.run([compiler,'-std=c++17','-O0','-I',str(ROOT/'src'),str(cpp),'-o',str(exe)],
                            text=True,capture_output=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe)],text=True,capture_output=True)
    print(result.stdout)
    assert result.returncode == 0, result.stderr


def resident_planning_probe():
    """Exercise real admission/SDPA code; tensor owners and watermarks are facades."""
    definition = runpy.run_path(str(ROOT / 'tests/test_qwen35_gdn_mask_residency.py'))['definition']
    native = (ROOT / 'src/fused/rpu_qwen3_5_moe_model.cpp').read_text()
    helpers = (ROOT / 'src/core/rpu_helpers.h').read_text()
    base = (ROOT / 'src/core/fused_model_base.cpp').read_text()
    code = source_probe().split('\nint main() {', 1)[0]
    code = code.replace('    bool resident_prefill_ = false;', '''
    bool resident_prefill_ = false;
    bool chunk_within_envelope(int64_t cs) const {return cs <= 8192;}
    bool subclass_chunk_size_valid(int64_t, int64_t, int64_t) const;
''')
    code = code.replace(
        'return {SdpaKernelType::FLASH_ATTN_SPM_VCTXLEN,256,nq,nkv,attn_tp(),mask};',
        'return {SdpaKernelType::FLASH_ATTN_SPM,256,nq,nkv,attn_tp(),mask};')
    for marker in ('inline bool sdpa_is_valid_chunk_size(',):
        code += definition(helpers, marker) + '\n'
    code += '\nnamespace v3 {\n'
    code += definition(base, 'static std::vector<ChunkInfo> make_fmb_chunks(') + '\n'
    for marker in ('bool Qwen3_5MoeModel::subclass_chunk_size_valid(',):
        code += definition(native, marker) + '\n'
    code += r'''
}
void begin_request(v3::Qwen3_5MoeModel& model,int64_t physical,int64_t logical) {
    model.resident_planning_active_=true;
    model.resident_planning_rows_=physical;
    model.resident_planning_logical_rows_=logical;
    model.resident_plans_.clear();
}
v3::FmbPrefillStageCandidate save_candidate(
        const v3::Qwen3_5MoeModel& model,int64_t capacity,int64_t physical,int64_t logical) {
    using namespace v3;
    FmbPrefillStageCandidate candidate;
    candidate.compute_chunk_size=capacity;
    candidate.stage_plan.compute.chunks=make_fmb_chunks(physical,0,capacity);
    candidate.physical_manifest.physical_length=physical;
    candidate.physical_manifest.logical_length=logical;
    for(const auto& outer:candidate.stage_plan.compute.chunks) {
        FmbRouteManifestEntry route;
        route.family=FmbRouteFamily::GRAPH_SCHEDULE;
        route.site_id=QWEN35_MOE_RESIDENT_SCHEDULE_SITE;
        route.selector=1;route.arguments=model.resident_schedule_arguments(outer);
        route.invocation=outer.idx;
        candidate.physical_manifest.routes.push_back(route);
    }
    LayoutContext layout;layout.chunk_size=capacity;
    candidate.ranking=model.prefill_candidate_ranking(candidate,layout);
    return candidate;
}
int main(int argc,char** argv) {
    using namespace v3;
    Qwen3_5MoeModel model;
    model.hidden=2048;model.nq=16;model.nkv=8;model.cores=8;model.layers=40;
    model.gdn_nvh_=32;model.gdn_conv_dim_=8192;model.moe_.tensor_parallel=8;
    model.layer_is_full_.resize(40);for(int i=0;i<40;++i)model.layer_is_full_[i]=(i%4==3);
    model.resident_prefill_=true;
    model.expert_ids_.present=true;
    set_watermarks(0,643072,false);
    const auto cfg=model.make_sdpa_config();
    if(argc==2 && std::string(argv[1])=="benchmark") {
        for(int64_t logical:{641,753,1539,4096,7169,8192}) {
            const int64_t physical=Align(logical,64);
            begin_request(model,physical,logical);
            budget_queries=0;
            const auto start=std::chrono::steady_clock::now();
            int64_t admitted=0;
            std::vector<std::pair<int64_t,int64_t>> ranking_rows;
            for(int64_t c=64;c<=physical;c+=64) {
                if(!model.subclass_chunk_size_valid(c,physical,0))continue;
                ++admitted;
                const auto candidate=save_candidate(model,c,physical,logical);
                assert(candidate.ranking.policy==FmbPrefillRankingPolicy::OUTER_THEN_MIXER_LARGER_WINDOW_V2);
                ranking_rows.emplace_back(c,candidate.ranking.mixer_calls);
            }
            const double elapsed=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-start).count();
            std::cout<<"{\"P\":"<<logical<<",\"E\":"<<physical
                     <<",\"wall_ms\":"<<elapsed<<",\"layout_queries\":"
                     <<budget_queries<<",\"admitted_outer_candidates\":"
                     <<admitted<<",\"route_stage\":\"full_kvsort\",\"ranking_candidates\":[";
            bool first=true;for(const auto& [c,calls]:ranking_rows) {
                if(!first)std::cout<<",";first=false;
                std::cout<<"["<<c<<","<<calls<<"]";
            }
            std::cout<<"]}\n";
        }
        return 0;
    }
    // The old fixed C384 schedule had an illegal Q320@384 call. The model
    // now keeps C704 with its real full-SPM-tested 320+320+64 schedule.
    begin_request(model,704,641);
    assert(model.subclass_chunk_size_valid(704,704,0));
    const auto cliff=*model.resident_plans_.at(704);
    assert(cliff.mixer_capacity==320 && cliff.workspace_bytes==163840);
    assert(cliff.total_calls==3 && cliff.windows[0].uniform_rows==320);
    assert(!sdpa_call_valid(cfg,320,384));
    LayoutContext lc;lc.num_layers=40;lc.max_kv_seq_len=8192;lc.chunk_size=704;
    const auto cliff_declarations=model.declare_buffers(lc);
    assert(final_chunk_layout_fits(cliff_declarations,false));
    assert(slot(cliff_declarations,"sdpa_tmp").size==cliff.workspace_bytes);
    // A second candidate cannot replace the first candidate's selected layout.
    assert(model.subclass_chunk_size_valid(640,704,0));
    same_declarations(model.declare_buffers(lc),cliff_declarations);
    const int cached_queries=budget_queries;
    assert(model.subclass_chunk_size_valid(704,704,0));
    assert(budget_queries==cached_queries);
    // Restore the exact saved schedule after the cold map has gone away.
    const auto candidate=save_candidate(model,704,704,641);
    model.resident_planning_active_=false;model.resident_plans_.clear();
    model.restore_resident_plan(candidate);
    model.resident_execution_fingerprint_=101;
    lc.physical_manifest_fingerprint=101;
    same_declarations(model.declare_buffers(lc),cliff_declarations);
    auto corrupt=candidate;++corrupt.physical_manifest.routes[0].arguments[10];
    rejects([&]{model.restore_resident_plan(corrupt);});
    corrupt=candidate;++corrupt.ranking.mixer_calls;
    rejects([&]{model.restore_resident_plan(corrupt);});
    corrupt=candidate;corrupt.ranking={};
    rejects([&]{model.restore_resident_plan(corrupt);});
    auto v1_compatible=candidate;
    v1_compatible.ranking.policy=FmbPrefillRankingPolicy::OUTER_THEN_MIXER_V1;
    model.restore_resident_plan(v1_compatible);
    assert(model.resident_execution_plan_->total_calls==candidate.ranking.mixer_calls);
    lc.physical_manifest_fingerprint=102;
    rejects([&]{model.declare_buffers(lc);});
    lc.physical_manifest_fingerprint=0;

    begin_request(model,768,753);
    assert(model.subclass_chunk_size_valid(768,768,0));
    const auto p753=*model.resident_plans_.at(768);
    assert(p753.mixer_capacity==384 && p753.workspace_bytes==196608);
    assert(p753.total_calls==2);
    begin_request(model,768,768);
    assert(model.subclass_chunk_size_valid(768,768,0));
    const auto descriptor_a=save_candidate(model,768,768,768);
    // Actual Q256 tail workspace is larger than actual Q384 workspace.
    assert(!sdpa_is_valid_chunk_size(cfg,384,4096,0));
    begin_request(model,4096,4096);
    assert(model.subclass_chunk_size_valid(768,4096,0));
    assert(model.resident_plans_.at(768)->workspace_bytes==262144);
    int calls=0;for(const auto& outer:make_fmb_chunks(4096,0,768)) {
        for(const auto& sub:model.planned_resident_mixer_chunks(outer)) {
            const int64_t expected_rows=calls<10 ? 384 : 256;
            assert(sub.len==expected_rows && sub.offset==calls*384 &&
                   sub.kv_seq_len==sub.offset+expected_rows);
            ++calls;
        }
    }
    assert(calls==11);
    const auto descriptor_b=save_candidate(model,768,4096,4096);
    // Same outer C, different request lengths/workspace, detached A->B->A
    // restoration must use each descriptor, never the latest planning cache.
    model.resident_planning_active_=false;model.resident_plans_.clear();
    budget_queries=0;
    for(const auto* descriptor:{&descriptor_a,&descriptor_b,&descriptor_a}) {
        model.restore_resident_plan(*descriptor);
        model.resident_execution_fingerprint_=
            descriptor->physical_manifest.physical_length;
        lc.chunk_size=768;
        lc.physical_manifest_fingerprint=model.resident_execution_fingerprint_;
        const auto declarations=model.declare_buffers(lc);
        const int64_t expected=descriptor==&descriptor_b?262144:196608;
        assert(slot(declarations,"sdpa_tmp").size==expected);
        assert(model.resident_execution_plan_->workspace_bytes==expected);
    }
    assert(budget_queries==0);
    // Detached descriptor layout identity is independent of the installed
    // execution descriptor. The actual base owner-scope lifecycle is outside
    // this source-extracted fixture.
    model.resident_detached_plan_=model.decode_resident_plan(descriptor_b);
    model.resident_detached_fingerprint_=303;
    lc.physical_manifest_fingerprint=303;
    assert(slot(model.declare_buffers(lc),"sdpa_tmp").size==262144);
    lc.physical_manifest_fingerprint=model.resident_execution_fingerprint_;
    assert(slot(model.declare_buffers(lc),"sdpa_tmp").size==196608);
    model.resident_detached_plan_.reset();
    lc.physical_manifest_fingerprint=303;
    rejects([&]{model.declare_buffers(lc);});
    assert(budget_queries==0);
    lc.physical_manifest_fingerprint=0;
    begin_request(model,7232,7169);
    assert(model.subclass_chunk_size_valid(768,7232,0));
    const auto p7169=*model.resident_plans_.at(768);
    assert(p7169.total_calls==20 && p7169.workspace_bytes==262144);
    assert(p7169.windows.back().offset==6912 &&
           p7169.windows.back().uniform_rows==256);
    // An entirely padded mixer call cannot enter the current GDN prefill
    // emitter; a wider actual call retaining positive valid rows is legal.
    begin_request(model,128,33);
    assert(!model.subclass_chunk_size_valid(64,128,0));
    assert(model.subclass_chunk_size_valid(128,128,0));
    assert(model.resident_plans_.at(128)->total_calls==1);
    // Compare any model-side resource shortcut against the full helper plus
    // complete allocator at feasible and infeasible boundaries. This does
    // not assume that fit is monotone in S or D.
    begin_request(model,4096,4096);
    for(int64_t capacity:{704,768,832,1024}) {
        LayoutContext trial;trial.chunk_size=capacity;trial.max_kv_seq_len=4096;
        trial.num_layers=40;trial.is_causal=true;
        const auto reference=qwen35_moe_planning::select_mixer_plan(
            {capacity,4096,4096,0,64},
            [&](const qwen35_moe_planning::Call& call)->std::optional<int64_t> {
                if(call.valid_rows<=0 || !sdpa_call_valid(cfg,call.rows,call.absolute_start))
                    return std::nullopt;
                return Align(sdpa_compute_tmp_v16_size(cfg,call.rows)*32,256);
            },
            [&](int64_t s,int64_t d) {
                const auto declarations=model.declare_resident_plan_buffers(trial,s,d);
                return qwen35_moe_planning::LayoutResult{
                    final_chunk_layout_fits(declarations,false),
                    static_cast<uint64_t>(detail::estimate_temporary_total(declarations))};
            });
        const bool admitted=model.subclass_chunk_size_valid(capacity,4096,0);
        assert(admitted==reference.plan.has_value());
        if(admitted) {
            const auto& actual=*model.resident_plans_.at(capacity);
            assert(actual.mixer_capacity==reference.plan->mixer_capacity);
            assert(actual.workspace_bytes==reference.plan->workspace_bytes);
            assert(actual.total_calls==reference.plan->total_calls);
        }
    }
    // All actual calls, declaration capacities and full layout use one plan.
    for(int64_t physical=64;physical<=8192;physical+=64) {
        begin_request(model,physical,physical-1);
        const int64_t capacity=std::min(int64_t{768},physical);
        assert(model.subclass_chunk_size_valid(capacity,physical,0));
        const auto& plan=*model.resident_plans_.at(capacity);
        lc.chunk_size=capacity;const auto ds=model.declare_buffers(lc);
        assert(final_chunk_layout_fits(ds,false));
        assert(slot(ds,"sdpa_tmp").size==plan.workspace_bytes);
        int64_t actual_tmp=0;
        for(const auto& outer:make_fmb_chunks(physical,0,capacity)) {
            for(const auto& sub:model.planned_resident_mixer_chunks(outer)) {
                assert(sdpa_call_valid(cfg,sub.len,sub.offset));
                assert(sub.len<=plan.mixer_capacity);
                actual_tmp=std::max(actual_tmp,
                    Align(sdpa_compute_tmp_v16_size(cfg,sub.len)*32,256));
            }
        }
        assert(actual_tmp==plan.workspace_bytes);
    }
    begin_request(model,4096,4096);
    for(int capacity:{0,16,63,65,832})assert(!model.subclass_chunk_size_valid(capacity,4096,0));
    for(int p:{63,4095}) {
        begin_request(model,p,p);
        assert(!model.subclass_chunk_size_valid(768,p,0));
    }
    begin_request(model,4096,4096);
    for(int64_t pos:{int64_t{1},int64_t{-1},std::numeric_limits<int64_t>::max()})
        rejects([&]{model.subclass_chunk_size_valid(768,4096,pos);});
    // No fitting layout is cached as a rejection; later declarations fail
    // instead of silently reverting to fixed mixer384.
    set_watermarks(SpmAllocator::SPM_USABLE,643072,false);
    begin_request(model,768,768);
    assert(!model.subclass_chunk_size_valid(768,768,0));
    lc.chunk_size=768;rejects([&]{model.declare_buffers(lc);});
    model.resident_planning_active_=false;
    for(int pos:{0,4097,8191}) {
        model.resident_prefill_=false;const bool legacy=model.subclass_chunk_size_valid(16,1,pos);
        model.resident_prefill_=true;assert(legacy && model.subclass_chunk_size_valid(16,1,pos)==legacy);
    }
    lc.chunk_size=16;
    same_declarations(model.declare_active_buffers(lc),model.declare_legacy_buffers(lc));
    std::cout<<"real declarations+allocator: C704=320+320+64; P753 tmp192KiB; "
             <<"P7169 tail256+64; 128 extents; descriptor restore; C16 decode unchanged\n";
}
'''
    return code


def test_resident_admission_matches_nested_fmb_mixer_calls(tmp_path):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    cpp, exe = tmp_path / 'admission.cpp', tmp_path / 'admission'
    cpp.write_text(resident_planning_probe())
    result = subprocess.run([compiler, '-std=c++17', '-O2', '-I', str(ROOT / 'src'),
                             str(cpp), '-o', str(exe)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe)], text=True, capture_output=True)
    print(result.stdout)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe), 'benchmark'], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(row) for row in result.stdout.splitlines()]
    assert [row['P'] for row in rows] == [641, 753, 1539, 4096, 7169, 8192]
    from test_execution_planner import _MODULE as planner, _ranked_domain
    expected = {641: 704, 753: 768, 1539: 768, 4096: 768, 7169: 768, 8192: 768}
    for row in rows:
        selected_plan = planner.plan_prefill(row['P'], row['E'], 0, alignment=64,
            resolve_stage_domain=lambda e: _ranked_domain(
                e, row['ranking_candidates'], logical_len=row['P'], policy=2))
        assert selected_plan.selected.stage_tuple.compute_chunk == expected[row['P']]
    print(result.stdout)


def test_moe_smoke_uses_registered_native_api():
    import ast
    import re
    script = ROOT/'python/rpu_backend/tests/ops/test_qwen35_moe_mask_replay.py'
    registration = (ROOT/'src/core/rpu_dispatch_registrations.inc').read_text()
    native = (ROOT/'src/fused/rpu_qwen3_5_moe_model.cpp').read_text()
    for node in ast.walk(ast.parse(script.read_text())):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        name = node.func.attr
        if not name.startswith('qwen3_5_moe_'):
            continue
        registered = re.search(r'm\.def\("' + re.escape(name) + r'([("])', registration)
        assert registered, name
        if registered[1] == '(':
            schema = re.search(r'm\.def\("' + re.escape(name) + r'\((.*?)\) ->', registration)[1]
            parameters = [part.rsplit(' ', 1)[-1].split('=')[0] for part in schema.split(', ')]
        else:
            # TORCH_FN infers its schema from this real C++ definition.
            signature = re.search(r'\brpu_' + re.escape(name) + r'\(([^)]*)\)', native)[1]
            parameters = [part.strip().rsplit(' ', 1)[-1] for part in signature.split(',') if part.strip()]
        assert len(node.args) + len(node.keywords) == len(parameters), (name, parameters)
        assert all(k.arg in parameters for k in node.keywords), name


def test_resident_native_layout_and_lifetimes(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    code = source_probe().split("\nint main() {", 1)[0] + r"""
int main() {
    using namespace v3;
    Qwen3_5MoeModel model;
    model.hidden=2048;model.nq=16;model.layers=40;model.gdn_nvh_=32;model.gdn_conv_dim_=8192;
    model.cores=8;model.nkv=8;model.moe_.tensor_parallel=8;model.resident_prefill_=true;
    model.expert_ids_.present=true;
    model.layer_is_full_.resize(40);
    for(int i=0;i<40;++i)model.layer_is_full_[i]=(i%4==3);
    LayoutContext lc;lc.num_layers=40;lc.max_kv_seq_len=8192;
    for(int w:{64,128,256,320,384,512,704,768}) {
        lc.chunk_size=w;
        auto ds=model.declare_buffers(lc);
        auto fixed=estimate_fixed_overhead(ds);
        set_watermarks(0,fixed.persistent,false);
        assert(final_chunk_layout_fits(ds,false));
        std::vector<SpmAllocator::AllocRequest> requests;
        std::vector<std::string> names;
        for(const auto& d:ds) if(d.storage==StorageClass::Temp&&!d.alias_of){
            requests.push_back({d.size,d.phase_start,d.phase_end,static_cast<int>(d.scope)});
            names.push_back(d.name);
        }
        auto plan=SpmAllocator::plan_temporary_aliased(requests);
        auto index=[&](const char* n) {auto it=std::find(names.begin(),names.end(),n);assert(it!=names.end());return it-names.begin();};
        auto disjoint=[&](const char* a,const char* b) {
            auto i=index(a),j=index(b);
            assert(plan.offsets[i]+requests[i].bytes<=plan.offsets[j] ||
                   plan.offsets[j]+requests[j].bytes<=plan.offsets[i]);
        };
        disjoint("residual1","moe_resident_input");
        disjoint("residual1","input_norm");
        disjoint("residual1","moe_ring_output");
        disjoint("moe_resident_input","moe_ring_output");
        disjoint("moe_resident_input","moe_routed_gate");
        disjoint("moe_resident_input","moe_shared_gate");
        disjoint("moe_resident_route_keys","moe_resident_sorted_experts");
        disjoint("moe_resident_ids","moe_resident_indices_i32");
        assert(requests[index("residual1")].bytes==w*2048*2);
        assert(requests[index("moe_resident_input")].bytes==w*2048*2);
        assert(requests[index("moe_resident_batch_counts")].bytes==CeilDiv(w*8,2048)*256*2);
        const int64_t total=fixed.persistent+fixed.temp_per_layer+plan.peak_bytes+65536;
        std::cout<<"W"<<w<<" temp="<<plan.peak_bytes<<" fixed="<<fixed.persistent
                 <<" total="<<total<<" headroom="<<(SpmAllocator::SPM_USABLE-total)<<std::endl;
    }
    lc.chunk_size=16;
    same_declarations(model.declare_buffers(lc),model.declare_legacy_buffers(lc));
    lc.chunk_size=832;
    assert(!final_chunk_layout_fits(model.declare_buffers(lc),false));
}
"""
    cpp, exe = tmp_path/"resident.cpp", tmp_path/"resident"
    cpp.write_text(code)
    result = subprocess.run([compiler,"-std=c++17","-O0","-I",str(ROOT/"src"),str(cpp),"-o",str(exe)],text=True,capture_output=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe)],text=True,capture_output=True)
    print(result.stdout)
    assert result.returncode == 0, result.stderr
