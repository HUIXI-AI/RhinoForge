"""Native regression for the exact 2B six-core C640 workspace schedule."""
from pathlib import Path
import shutil
import subprocess

import pytest
from test_qwen35_gdn_mask_residency import definition, source_probe

ROOT = Path(__file__).resolve().parents[1]


def workspace_source_probe():
    code=source_probe()
    base=(ROOT/'src/core/fused_model_base.cpp').read_text()
    fixed=definition(base,'struct FixedSpmOverhead')+';\n'+definition(base,'static FixedSpmOverhead estimate_fixed_overhead(')
    code=code.replace('const BufferDecl& slot(',fixed+'\nconst BufferDecl& slot(',1)
    # Probe calls the exact installed D resolver body, with addresses coming from the actual declarations.
    s=(ROOT/'src/fused/rpu_qwen3_5_model.cpp').read_text();start=s.index('    const bool input_workspace = !decode && use_text_gdn_input_workspace();',s.index('void Qwen3_5Model::build_gdn('));end=s.index('    const bool initial_prefill_chunk',start)
    fragment=s[start:end]
    shim='''    uint32_t emitted_address(const char* name,int64_t L,bool decode=false) {
            ChunkInfo chunk{0,0,L,L};
            auto addr=[&](int,const char* n) -> uint32_t {
                LayoutContext lc;lc.chunk_size=L;lc.num_layers=layers;
                auto rows=declare_buffers(lc);
                const auto it=std::find_if(rows.begin(),rows.end(),[&](const auto& d){return std::string(d.name)==n;});
                assert(it!=rows.end());
                return it->alias_of && std::string(it->alias_of)=="input_norm" || std::string(n)=="input_norm" ? 4096 : 8192;
            };
            EMIT_FRAGMENT
            return D(name);
        }
    '''.replace('EMIT_FRAGMENT',fragment)
    code=code.replace('class Qwen3_5Model : public FusedModelBase {\npublic:','class Qwen3_5Model : public FusedModelBase {\npublic:\n'+shim,1)
    body=r'''
        Qwen3_5Model model;model.cores=6;model.override_chunk=640;model.hidden=2048;
        model.inter=6144;model.nq=8;model.nkv=4;model.layers=24;
        model.layer_is_full_.resize(24);for(int i=0;i<24;++i)model.layer_is_full_[i]=(i%4==3);
        assert(model.use_text_gdn_input_workspace());
        for(int64_t rows:{64,128,256,512,576,640}) {
            LayoutContext lc;lc.chunk_size=rows;lc.max_kv_seq_len=4096;lc.num_layers=24;
            const auto decls=model.declare_buffers(lc);
            const auto args=model.gdn_input_workspace_arguments(rows);
            model.context.expected=args;
            std::vector<const BufferDecl*> views;
            for(const auto& d:decls) if(model.gdn_input_workspace_slot(d.name)>=0) {
                views.push_back(&d);assert(d.alias_of && std::string(d.alias_of)=="input_norm");
                const auto offset=model.gdn_input_workspace_offset(d.name,rows);
                assert(offset>=0 && offset+d.size<=slot(decls,"input_norm").size);
                assert(d.phase_start>1 && d.phase_end<17);
                assert(model.emitted_address(d.name,rows)==4096+offset);
            }
            assert(views.size()==8);
            for(const auto* a:views) for(const auto* b:views) if(a!=b) {
                const auto ao=model.gdn_input_workspace_offset(a->name,rows),bo=model.gdn_input_workspace_offset(b->name,rows);
                const bool address_overlap=ao<bo+b->size && bo<ao+a->size;
                const bool time_overlap=a->phase_start<=b->phase_end && b->phase_start<=a->phase_end;
                assert(!address_overlap || !time_overlap);
            }
            assert(!slot(decls,"gdn_c_coreout").alias_of);
            assert(!slot(decls,"gdn_c_z").alias_of);
            const auto temp=detail::estimate_temporary_total(decls);
            const auto fixed=estimate_fixed_overhead(decls).total();
            assert(temp+fixed<=SpmAllocator::SPM_PLANNING_BUDGET);
            std::cout<<"candidate rows="<<rows<<" temporary="<<temp<<" fixed="<<fixed<<" total="<<temp+fixed<<" budget="<<SpmAllocator::SPM_PLANNING_BUDGET<<"\n";
        }
        for(int64_t bad:{0,1,63,65,641,704}) {
            bool rejected=false;try {model.gdn_input_workspace_arguments(bad);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
        }
        model.context.complete=false;bool rejected=false;
        try {model.emitted_address("gdn_c_query",640);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
        model.context.complete=true;
        for(int core:{4,8}){model.cores=core;assert(!model.use_text_gdn_input_workspace());}
        for(int core:{4,8}) {
            model.cores=core;LayoutContext lc;lc.chunk_size=640;lc.num_layers=24;
            model.override_chunk=640;auto exact=model.declare_buffers(lc);
            model.override_chunk=0;same_declarations(exact,model.declare_buffers(lc));
        }
        model.cores=6;model.override_chunk=640;
        LayoutContext too_large;too_large.chunk_size=704;too_large.num_layers=24;
        assert(!slot(model.declare_buffers(too_large),"gdn_c_query").alias_of);
        model.cores=6;for(int64_t cap:{0,512,576,704}){model.override_chunk=cap;assert(!model.use_text_gdn_input_workspace());}
        model.override_chunk=640;model.hidden=1024;assert(!model.use_text_gdn_input_workspace());model.hidden=2048;
        model.action_mode_=true;assert(!model.use_text_gdn_input_workspace());model.action_mode_=false;
        model.z2_bound_=true;assert(!model.use_text_gdn_input_workspace());model.z2_bound_=false;
        model.gdn_nvh_=32;assert(!model.use_text_gdn_input_workspace());model.gdn_nvh_=16;
        // Decode retains its original separate buffers even for an exact-C640 owner.
        LayoutContext decode;decode.chunk_size=1;auto ds=model.declare_buffers(decode);
        assert(std::none_of(ds.begin(),ds.end(),[](const auto& d){return std::string(d.name)=="gdn_c_query";}));
    '''
    code=code.replace('int main() {','int residency_regression_main() {',1)
    code=code.replace('    Qwen3_5Model dense;schedule_test(dense);','    Qwen3_5Model dense;schedule_test(dense);return 0;',1)
    code+='\nint main(){residency_regression_main();using namespace v3;\n'+body+'\n}\n'
    return code


def test_gdn_input_workspace_lifetimes_addresses_and_budget(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    source, executable = tmp_path / "workspace.cpp", tmp_path / "workspace"
    source.write_text(workspace_source_probe())
    result = subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(ROOT / "src"),
                             str(source), "-o", str(executable)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(executable)], text=True, capture_output=True)
    print(result.stdout)
    assert result.returncode == 0, result.stderr
