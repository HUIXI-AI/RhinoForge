"""Real FMB resource admission reserves the post-forward eager head row."""
from pathlib import Path
import subprocess

from test_qwen35_moe_mask_budget import source_probe as moe_source_probe
from test_qwen35_gdn_mask_residency import definition

ROOT = Path(__file__).resolve().parents[1]


def source_probe():
    core=(ROOT/'src/core/fused_model_base.cpp').read_text()
    header=(ROOT/'src/core/fused_model_base.h').read_text()
    moe=(ROOT/'src/fused/rpu_qwen3_5_moe_model.cpp').read_text()
    linear=(ROOT/'src/ops/rpu_linear.cpp').read_text()
    code=moe_source_probe().split('\nint main() {',1)[0]
    for marker in ('static int64_t available_temporary_spm_after_reset(', 'static bool final_chunk_layout_fits('):
     code=code.replace(definition(code,marker), definition(core,marker))
    code=code.replace('class FusedModelBase {};','class FusedModelBase {public: struct Impl {bool persistent_allocated_=false;};};')
    code=code.replace('int64_t temporary_used() const {return temporary;}', 'int64_t temporary_used() const {return temporary;}\n    int64_t total_used() const {return SpmAllocator::SPM_USABLE-free;}')
    # Both reduced models have the original attention/head4 domain in the facade.
    start=code.index('class Qwen3_5MoeModel :')
    code=code[:start]+code[start:].replace('public:\n','public:\n    int64_t minimum_eager_spm_bytes() const;\n    int lm_head_tp() const {return num_cores()==6 ? 4 : num_cores();}\n',1)
    # Use actual current routed/shared domains, including UINT4 root6/routed4.
    moe_header=(ROOT/'src/fused/rpu_qwen3_5_moe_model.h').read_text()
    moe_shim=definition(code,'class Qwen3_5MoeModel :')
    updated_shim=moe_shim
    for name in ('int routed_expert_tp() const', 'int shared_expert_tp() const',
                 'int64_t physical_routed_intermediate() const', 'int64_t physical_shared_intermediate() const'):
        updated_shim=updated_shim.replace(definition(updated_shim,name),definition(moe_header,name))
    code=code.replace(moe_shim,updated_shim)
    code=code.replace('return final_chunk_layout_fits(rows,own_installed);','return final_chunk_layout_fits(rows,own_installed,minimum_eager_spm_bytes());')
    code+='\nnamespace v3 {\nnamespace detail {\n'+definition(header,'struct FmbPreparedSpmCost')+';\n}\n'
    for marker in ('int64_t detail::FmbPreparedSpmCost::available(', 'bool detail::FmbPreparedSpmCost::fits(', 'static void validate_final_chunk_layout_fits(', 'struct ChunkSizeProbe', 'static std::vector<ChunkSizeProbe> enumerate_chunk_size_domain_impl('):
     code+=definition(core,marker)+ (';\n' if marker.startswith('struct') else '\n')
    code+=definition(moe,'int64_t Qwen3_5MoeModel::minimum_eager_spm_bytes() const')+'\n}\n'
    fragment=linear[linear.index('  // Per core, per row: x +'):linear.index('  size_t m_chunk = col_scatter ?')]
    code+='''\nbool eager_head_accepts(int64_t bytes) {
        watermarks.free=bytes; watermarks.temporary=0;
        const size_t a_dwidth=2,m=1,k=2048,n=248320,local_n=n/4,local_k=k;
        // The eager head is a plain GEMM: its A row spans the whole local K and
        // it stages no halo. Only the conv-slab route narrows the pitch.
        const size_t in_pitch=local_k;
        const int64_t halo=0;
        const bool kernel_applies_bias=false,w8a16=false,w4a16=false,in_graph=false;
        // An ordinary DDR head: no conv slab and no live destination base.
        const void* conv_slab=nullptr;
        const uint64_t* live_dst_base=nullptr;
        const int partition=1;
        try {
    '''+fragment+'''\n    return true;
        } catch(const std::runtime_error&) {return false;}
    }\n'''
    code+=r'''
    int main() {
        using namespace v3;
        using Mode=FmbForwardOperandResidency;
        Qwen3_5MoeModel model;
        model.hidden=2048;model.nq=16;model.layers=40;model.gdn_nvh_=32;model.gdn_conv_dim_=8192;
        model.layer_is_full_.resize(40);
        for(int i=0;i<40;++i)model.layer_is_full_[i]=(i%4==3);
        model.cores=6;model.nkv=4;model.moe_.tensor_parallel=6;
        const int64_t reserve=model.minimum_eager_spm_bytes();
        assert(reserve==129280);
        assert(!eager_head_accepts(109568));
        assert(!eager_head_accepts(129024));
        assert(eager_head_accepts(reserve));
        assert(eager_head_accepts(reserve+256));
        for(int cores:{4,8}) {model.cores=cores;assert(model.minimum_eager_spm_bytes()==0);}

        // The old failing geometry can fit the fused layout while leaving too
        // little for eager. Check the actual generic helpers at the exact boundary.
        std::vector<BufferDecl> artificial={{"body",1048576,0,1,StorageClass::Temp}};
        watermarks.temporary=0;watermarks.free=1048576+109568;
        assert(final_chunk_layout_fits(artificial,false));
        assert(!final_chunk_layout_fits(artificial,false,reserve));
        watermarks.free=1048576+reserve;
        assert(final_chunk_layout_fits(artificial,false,reserve));
        validate_final_chunk_layout_fits(artificial,false,"head",reserve);
        watermarks.free-=256;
        rejects([&]{validate_final_chunk_layout_fits(artificial,false,"head",reserve);});

        // Cached costs remain pure; each call uses the live free-space and reserve.
        detail::FmbPreparedSpmCost saved{1024,2048,1048576};
        assert(saved.fits(1048576+1024+2048+reserve,false,SpmAllocator::SPM_PLANNING_BUDGET,reserve));
        assert(!saved.fits(1048576+1024+2048+reserve-256,false,SpmAllocator::SPM_PLANNING_BUDGET,reserve));
        assert(saved.fits(1048576+2048+reserve,true,SpmAllocator::SPM_PLANNING_BUDGET,reserve));
        assert(!saved.fits(1048576+2048+reserve-256,true,SpmAllocator::SPM_PLANNING_BUDGET,reserve));
        rejects([&]{saved.available(8000000,false,SpmAllocator::SPM_PLANNING_BUDGET,-256);});
        rejects([&]{saved.available(8000000,false,SpmAllocator::SPM_PLANNING_BUDGET,129025);});
        rejects([&]{saved.available(8000000,false,SpmAllocator::SPM_PLANNING_BUDGET,SpmAllocator::SPM_TOTAL);});

        int observations=0,admitted=0,rejected=0,default_zero=0;
        LayoutContext lc;lc.num_layers=40;lc.max_kv_seq_len=8192;
        for(int mode:{5,6}) for(int cores:{4,6,8}) {
            model.cores=cores;model.nkv=model.attn_tp();model.moe_.tensor_parallel=cores;
            for(auto& layer:model.moe_layer_weights_)layer.routed_weight_mode=mode;
            if(cores==6 && mode==6) {assert(model.routed_expert_tp()==4);assert(model.physical_routed_intermediate()==512);}
            if(cores==6 && mode==5) {assert(model.routed_expert_tp()==6);assert(model.physical_routed_intermediate()==576);}
            const int64_t need=model.minimum_eager_spm_bytes();
            for(int rows=64;rows<=640;rows+=64) for(Mode residency:{Mode::PER_LAYER,Mode::FORWARD}) {
                lc.chunk_size=rows;lc.forward_operand_residency=residency;
                const auto ds=model.declare_buffers(lc);
                const auto fixed=estimate_fixed_overhead(ds);
                const int64_t peak=detail::estimate_temporary_total(ds);
                const detail::FmbPreparedSpmCost cost{fixed.persistent,fixed.temp_per_layer,peak};
                for(bool warm:{false,true}) for(int64_t held:{int64_t{0},int64_t{32768},int64_t{131072}}) {
                    set_watermarks(held,fixed.persistent,warm);
                    const int64_t reset=watermarks.free+watermarks.temporary;
                    const int64_t old_expected=std::min(std::max<int64_t>(SpmAllocator::SPM_PLANNING_BUDGET-fixed.total(),0),
                        std::max<int64_t>(reset-65536-fixed.temp_per_layer-(warm?0:fixed.persistent),0));
                    assert(available_temporary_spm_after_reset(ds,warm,0)==old_expected);
                    assert(cost.available(reset,warm,SpmAllocator::SPM_PLANNING_BUDGET,0)==old_expected);
                    ++default_zero;
                    assert(cost.available(reset,warm,SpmAllocator::SPM_PLANNING_BUDGET,need)==available_temporary_spm_after_reset(ds,warm,need));
                    bool fits=final_chunk_layout_fits(ds,warm,need);
                    assert(fits==cost.fits(reset,warm,SpmAllocator::SPM_PLANNING_BUDGET,need));
                    ++observations;
                    if(fits) {
                        ++admitted;
                        const int64_t free=reset-fixed.temp_per_layer-(warm?0:fixed.persistent)-peak;
                        assert(free>=std::max<int64_t>(65536,need));
                        if(cores==6)assert(eager_head_accepts(free));
                    } else ++rejected;
                }
            }
            model.cores=cores;lc.forward_operand_residency=Mode::PER_LAYER;
            set_watermarks(0,643072,false);
            FusedModelBase::Impl state;
            auto probes=enumerate_chunk_size_domain_impl(state,
                [&](const LayoutContext& c){return model.declare_buffers(c);},
                [](int64_t rows){return rows%64==0;},640,4096,lc,true,need);
            bool any=false;
            for(const auto& p:probes)if(p.valid) {
                LayoutContext c=lc;c.chunk_size=p.chunk_size;
                const auto ds=model.declare_buffers(c);
                assert(p.fits==final_chunk_layout_fits(ds,false,need));
                any|=p.fits;
            }
            assert(any);
        }
        std::cout<<"head-strict-boundary PASS, real-layout cases="<<observations
                 <<" admitted="<<admitted<<" rejected="<<rejected<<" default0="<<default_zero
                 <<", cached-cost/live/direct-domain contracts PASS\n";
    }
    '''
    return code


def test_eager_head_reserve_uses_actual_layout_and_live_resource_boundaries(tmp_path):
    source, executable = tmp_path / "head.cpp", tmp_path / "head"
    source.write_text(source_probe())
    built = subprocess.run(["c++", "-std=c++17", "-O0", "-I", str(ROOT / "src"),
                            str(source), "-o", str(executable)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "head-strict-boundary PASS" in result.stdout
    assert "cached-cost/live/direct-domain contracts PASS" in result.stdout
