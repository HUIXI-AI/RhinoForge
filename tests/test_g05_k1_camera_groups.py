"""Host execution of actual K1 camera-group routes; no hardware claims."""
from pathlib import Path
from pi_wall_host_catalog import with_actual_linear_contracts
import ast
import re
import pytest

from test_g05_vision_owner_model import _block
from test_siglip_attention_dry_live_contract import _run_native

ROOT = Path(__file__).resolve().parents[1]


def _source_method(source, signature):
    name=re.search(r'(\w+)\(',signature)[1]
    return signature+_block(source,'Qwen3_5VisionModel::'+name+'(')


def test_camera_groups_real_complete_mode_camera_isolation_and_spm(tmp_path, monkeypatch):
    import test_g05_k1_b3_model as legacy
    monkeypatch.setenv('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
    import torch
    captured=[]
    with monkeypatch.context() as scoped:
        scoped.setattr(legacy,'_run_native',lambda path,cpp,*a,**k: captured.append(cpp))
        legacy.test_k1_b3_plan_and_camera_isolation(tmp_path,scoped,False)
    cpp=captured[0].split('    int main(){')[0]
    source=(ROOT/'src/fused/rpu_qwen3_5_vision_model.cpp').read_text()
    signatures=[
        'bool g05_k1_camera_group_available(int64_t cameras) const',
        'bool is_g05_k1_camera_group_profile() const',
        'bool g05_k1_camera_group_schedule(const FmbThreeStageChunkPlan& plan) const',
        'std::vector<int64_t> g05_k1_camera_group_arguments() const',
        'void append_g05_k1_camera_group_routes(FmbPhysicalExecutionManifest& manifest) const',
        'void consume_g05_k1_camera_group_schedule(const ChunkInfo& chunk,bool compute)',
        'void emit_g05_k1_camera_group_layer(int layer,const ChunkInfo& chunk)',
    ]
    for signature in signatures:
        name=re.search(r'(\w+)\(',signature)[1]
        stub=re.search(r' [^\n]*\b'+name+r'\([^\n]*\}\n',cpp)
        if stub:cpp=cpp[:stub.start()]+cpp[stub.end():]
    definitions='\n'.join(_source_method(source,sig) for sig in signatures)
    cpp=cpp.replace('    };\n    template<class F>',definitions+'\n    };\n    template<class F>')
    constants='\n'.join(re.findall(r'constexpr int64_t G05_VISION_K1_CAMERA_GROUP_\w+_SITE = \d+LL;',source))
    cpp=cpp.replace('    using namespace v3;', '#include "ops/rpu_linear_tiling.h"\n    using namespace v3;\n'+constants)
    cpp=cpp.replace('std::vector<std::string> emitted;int camera=-1;int layer_value=0;',
        'std::vector<std::string> emitted;int camera=-1;int layer_value=0;bool expected_acc32=false;int expected_rows=512;')
    cpp=cpp.replace('&&v==V&&!acc32','&&v==V&&acc32==expected_acc32')
    old='void rpu_launch_linear_spm_to_spm_acc16_kernel'+_block(cpp,'void rpu_launch_linear_spm_to_spm_acc16_kernel')
    begin=cpp.index('void rpu_launch_linear_spm_to_spm_acc16_kernel')
    end=begin+len(cpp[begin:cpp.index('{',begin)])+len(_block(cpp,'void rpu_launch_linear_spm_to_spm_acc16_kernel'))
    cpp=cpp[:begin]+r'''void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t in,const at::Tensor&,uint32_t out,int64_t m,int64_t n,int64_t k,int part,int cores,uint32_t,const at::Tensor&,int,int,bool acc32){
       assert(cores==8&&acc32==expected_acc32);
       if(k==4096){assert(m==expected_rows&&n==1024&&part==0&&in==F&&out==P);emitted.push_back("fc2-generated");return;}
       assert(m==256&&k==1024);
       if(part==0){assert(n==1024&&in==O&&out==P+camera*524288);emitted.push_back("o");}
       else{assert(part==1&&n==4096&&in==P+camera*524288&&out==F+camera*262144);emitted.push_back("fc1");}}
    '''+cpp[end:]
    cpp=cpp.replace('assert(rows==768);return 768;', 'assert(rows==current_output_patches_);return rows;')
    cpp=cpp.replace('assert(c.len==768&&c.offset==0);emitted.push_back("input768");', 'assert(c.len==expected_rows&&c.offset==0);emitted.push_back("input");')
    cpp=cpp.replace('assert(c.len==768&&c.offset==0);emitted.push_back("output768");', 'assert(c.len==expected_rows&&c.offset==0);emitted.push_back("output");')
    cpp=cpp.replace('bool has_step0_=true,has_merger_=true;', 'std::array<bool,2>g05_k1_generated_group_available_{{true,true}};\n bool has_step0_=true,has_merger_=true;')
    cpp+=r'''
int main(){
  for(bool high:{false,true})for(int bits:{16,32})for(int cams:{2,3}){
    if(bits==16&&cams==3)continue;
    expected_high=high;Model m;m.high_precision_=high;m.linear_acc32_=expected_acc32=bits==32;m.current_camera_batch_count_=cams;
    m.current_output_patches_=m.context.seq_len=expected_rows=cams*256;
    assert(m.g05_k1_camera_group_available(cams)&&m.is_g05_k1_camera_group_profile());
    assert(!m.g05_k1_camera_group_available(1)&&!m.g05_k1_camera_group_available(4));
    auto execution=m.resolve_execution_plan(expected_rows);
    assert(execution.input_chunk_size==256&&execution.compute_chunk_size==expected_rows&&execution.qkv_chunk_size==expected_rows&&execution.stages.spans.size()==size_t(cams));
    m.context.stage_plan=execution.stages;
    FmbPrefillStageCandidate c;c.input_chunk_size=256;c.qkv_chunk_size=c.compute_chunk_size=expected_rows;c.stage_plan=m.context.stage_plan;
    auto& manifest=c.physical_manifest;manifest.state=FmbPhysicalManifestState::COMPLETE;
    manifest.logical_length=manifest.physical_length=manifest.kv_logical_length=manifest.kv_insert_physical_rows=expected_rows;
    manifest.graph_lifecycle=FmbGraphLifecycle::RETAINED_CACHE;
    manifest.linear_accumulation=expected_acc32?FmbLinearAccumulationPolicy::ACC32:FmbLinearAccumulationPolicy::ACC16;
    m.append_g05_k1_camera_group_routes(manifest);
    std::sort(manifest.routes.begin(),manifest.routes.end(),[](const auto&a,const auto&b){return std::tie(a.family,a.site_id,a.invocation)<std::tie(b.family,b.site_id,b.invocation);});
    auto wire=encode_fmb_prefill_stage_candidate(c);auto restored=decode_fmb_prefill_stage_candidate(wire);
    assert(restored.physical_manifest.linear_accumulation==manifest.linear_accumulation);
    assert(fmb_physical_manifest_fingerprint(restored.physical_manifest)==fmb_physical_manifest_fingerprint(manifest));
    const ChunkInfo full{0,0,expected_rows,expected_rows};
    for(int iteration=0;iteration<3;++iteration){
      emitted.clear();m.context.consumer.reset(restored.physical_manifest);
      m.consume_g05_k1_camera_group_schedule(full,false);m.emit_g05_k1_camera_group_layer(0,full);m.context.consumer.require_all_consumed();
      std::vector<std::string>expected{"input"};
      for(int cam=0;cam<cams;++cam){expected.push_back("ln"+std::to_string(cam));expected.insert(expected.end(),{"qkv","rope","rope","transpose","attention","prepare","o","ownerln","fc1","gelu"});}
      expected.push_back("fc2-generated");for(int cam=0;cam<cams;++cam)expected.push_back("compact"+std::to_string(cam));expected.push_back("output");assert(emitted==expected);
    }
    for(size_t r=0;r<manifest.routes.size();++r)for(size_t a=0;a<manifest.routes[r].arguments.size();++a){auto bad=manifest;++bad.routes[r].arguments[a];m.context.consumer.reset(bad);reject([&]{m.emit_g05_k1_camera_group_layer(0,full);});}
    for(int fault=0;fault<11;++fault){auto x=m;x.context.consumer.reset(manifest);if(fault==0)x.context.complete=false;if(fault==1)x.context.position=1;if(fault==2)x.context.seq_len=256;if(fault==3)x.context.attention_policy=AttentionExecutionPolicy::DDR_KV;if(fault==4)x.context.stage_plan.spans[1].group_id=0;if(fault==5)x.context.stage_plan.input.chunks[1].offset=0;if(fault==6)x.current_temporal_num_frames_=6;if(fault==7)x.g05_k1_generated_group_available_[cams-2]=false;if(fault==8)x.current_output_patches_-=256;if(fault==9)x.linear_acc32_=!expected_acc32;if(fault==10)x.g05_w8a16_=false;reject([&]{x.emit_g05_k1_camera_group_layer(0,full);});}
    Model other=m;other.linear_acc32_=!m.linear_acc32_;assert(other.g05_k1_camera_group_arguments()!=m.g05_k1_camera_group_arguments());
  }
  Model legacy;assert(legacy.g05_k1_b3_available()&&!legacy.is_g05_k1_camera_group_profile());
  std::cout<<"B2 ACC16 / B2+B3 ACC32 real FMB codec+consumer, isolated per-camera M256, real-mode generated FC2: PASS\n";
}
'''
    _run_native(tmp_path,cpp,torch,extra_sources=(ROOT/'src/core/fmb_three_stage_chunk_plan.cpp',))


def test_camera_group_native_actual_joint_spm_declarations(tmp_path):
    import subprocess
    source=(ROOT/'src/fused/rpu_qwen3_5_vision_model.cpp').read_text()
    body=_block(source,'std::vector<BufferDecl> Qwen3_5VisionModel::declare_buffers(')
    # Execute actual requested-profile buffer branch and the exact allocator;
    # STEP0 and merger declarations below are taken from their actual source.
    begin=body.index('    std::vector<BufferDecl> decls;')
    end=body.index('    } else if (current_temporal_num_frames_ > 1)',begin)
    group=body[begin:end]+'\n    }\n'
    group=group.replace('is_g05_k1_b3_profile() || is_g05_k1_camera_group_profile()','true')
    initializers=[]
    for name in ('step0 = {','merger = {','std::vector<BufferDecl> persistent = {'):
        pattern=body.index(name,body.index('const int64_t s0_io') if name.startswith('step0') else body.index('std::vector<BufferDecl> merger;'))
        opening=body.index('{',pattern);depth=1;i=opening+1
        while depth:depth+=(body[i]=='{')-(body[i]=='}');i+=1
        initializers.append(body[opening:i])
    planner=(ROOT/'src/core/rpu_spm_allocator.cpp').read_text()
    native_planner='SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(const std::vector<AllocRequest>& requests)'+_block(planner,'SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(')
    cpp=r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
enum class StorageClass{Temp,Persistent,PersistentPerLayer};enum class BufferScope{LayerWide,OutsideLayerLoop=3};
struct BufferDecl{const char*name;int64_t size;int phase_start,phase_end;StorageClass storage;int per_layer;const char*alias_of;BufferScope scope=BufferScope::LayerWide;};
struct SpmAllocator{struct AllocRequest{int64_t bytes;int first_step,last_step,scope;};struct AliasedPlan{std::vector<uint32_t>offsets;size_t peak_bytes=0;};static size_t align_up(size_t n){return (n+255)&~size_t(255);}static AliasedPlan plan_temporary_aliased(const std::vector<AllocRequest>&);};
__PLANNER__
int main(){for(int cameras:{2,3}){
  int current_camera_batch_count_=cameras;constexpr BufferScope ALL=BufferScope::LayerWide,OUTSIDE=BufferScope::OutsideLayerLoop;
  __GROUP__
  const int64_t s0_io=786432,cs0=256,s0_pos_width=1024,h=1024;
  const int64_t minput=1048576,mfc2=524288,mfc1=131072,local_mh=512,oh=2048;
  const int64_t norm_w_sz=2048,q_bias_sz=512,fc1_bias_sz=1024,full_bias_sz=2048,rope_tbl_sz=4096;const int nl=24;
  auto A=[](int64_t bytes){return (bytes+255)/256*256;};auto dma_safe=[&](int64_t elems){return A((elems+255)/256*256*2);};constexpr int DWIDTH=2;
  auto num_cores=[](){return 8;};
  std::vector<BufferDecl>step0=__STEP0__,merger=__MERGER__,persistent_decls=__PERSISTENT__;
  decls.insert(decls.end(),step0.begin(),step0.end());decls.insert(decls.end(),merger.begin(),merger.end());decls.insert(decls.end(),persistent_decls.begin(),persistent_decls.end());
  std::vector<SpmAllocator::AllocRequest>requests;int64_t persistent=0;
  for(const auto&d:decls)if(!d.alias_of){if(d.storage==StorageClass::Temp)requests.push_back({d.size,d.phase_start,d.phase_end,int(d.scope)});else persistent+=A(d.size)*(d.storage==StorageClass::PersistentPerLayer?d.per_layer:1);}
  assert(persistent==373760);const auto plan=SpmAllocator::plan_temporary_aliased(requests);const int64_t joint=plan.peak_bytes+persistent;
  assert(plan.peak_bytes==size_t(cameras==2?4063232:5898240));assert(joint==(cameras==2?4436992:6272000));assert(joint+65536<8303708);
  assert(plan.offsets[0]!=plan.offsets[1]&&plan.offsets[1]!=plan.offsets[2]);
  std::cout<<"C="<<cameras<<" joint_spm="<<joint<<" budget_slack="<<8303708-joint<<"\n";
}}
'''.replace('__GROUP__',group).replace('__STEP0__',initializers[0]).replace('__MERGER__',initializers[1]).replace('__PERSISTENT__',initializers[2]).replace('__PLANNER__',native_planner)
    path=tmp_path/'layout.cpp';path.write_text(cpp);exe=tmp_path/'layout'
    built=subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror',str(path),'-o',str(exe)],capture_output=True,text=True)
    assert built.returncode==0,built.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr


def test_camera_group_adapter_cold_guards_and_actual_grid(tmp_path,monkeypatch):
    import torch
    from types import SimpleNamespace
    path=ROOT/'python/rpu_backend/adapters/g05/runtime.py';tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='configure_g05_vision_camera_batch')
    ns=dict(torch=torch,_G05_RELEASE_CAMERA_NAMES=('head_rgb','left_wrist_rgb','right_wrist_rgb'),_G05_TWO_CAMERA_NAMES=('head_rgb','left_wrist_rgb'),_g05_camera_profile_matches=lambda p,m: bool(p.match))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    configure=ns[node.name]
    for cameras,bits in ((2,16),(2,32),(3,32),(3,16)):
        calls=[];session=SimpleNamespace(cold=True)
        def require_cold():
            if not session.cold:raise ValueError('live')
        session.require_cold=require_cold
        vision=SimpleNamespace(_rpu_vision_k1_camera_group_available=(True,True),_rpu_vision_exact_chunk_size=None,_rpu_vision_chunk_size_cap=0)
        def forward(*a,**kw):
            calls.append(kw['_rpu_camera_batch_count']);vision._rpu_vision_last_a6_plan={'selected':{'physical_metadata':{'camera_batch_count':cameras}}};return 42
        vision._rpu_g05_vision_forward_impl=forward
        p=SimpleNamespace(model=SimpleNamespace(vision_tower=vision),_execution_session=session,_rpu_g05_fm_only_customer_release=True,_rpu_g05_camera_names=ns['_G05_RELEASE_CAMERA_NAMES'][:cameras],match=True)
        configure(p,cameras);wrapper=vision._rpu_g05_vision_forward_impl
        grid=torch.tensor([[1,16,16]]*cameras)
        assert wrapper(None,grid,_rpu_temporal_num_frames=1)==42 and calls==[cameras]
        with pytest.raises(ValueError):wrapper(None,grid[:1],_rpu_temporal_num_frames=1)
        with pytest.raises(ValueError):wrapper(None,grid,_rpu_temporal_num_frames=6)
        session.cold=False
        with pytest.raises(ValueError):configure(p,1)
        session.cold=True;configure(p,1);assert vision._rpu_g05_vision_forward_impl is forward
        vision._rpu_vision_k1_camera_group_available=(False,False)
        with pytest.raises(ValueError):configure(p,cameras)
        assert vision._rpu_g05_vision_forward_impl is forward
        vision._rpu_vision_k1_camera_group_available=(True,True);vision._rpu_vision_exact_chunk_size=256
        with pytest.raises(ValueError):configure(p,cameras)
        vision._rpu_vision_exact_chunk_size=None;p.match=False
        if cameras==2:
            with pytest.raises(ValueError):configure(p,cameras)
        with pytest.raises(ValueError):configure(p,3 if cameras==2 else 2)
    with pytest.raises(ValueError):configure(p,True)


def test_camera_group_cold_actual_generated_capabilities_and_no_hot_lookup(tmp_path):
    vision_tree=ast.parse((ROOT/'python/rpu_backend/adapters/qwen3_5/vision.py').read_text())
    ready=next(node.value for node in vision_tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_QWEN3_5_VISION_READY_ATTRS' for t in node.targets))
    assert '_rpu_vision_k1_camera_group_available' in ast.literal_eval(ready)
    from test_g05_vision_qkv_spm_native import _qkv_native_cpp,_compile_qkv
    cpp=_qkv_native_cpp().split('int main(){')[0]
    source=(ROOT/'src/fused/rpu_qwen3_5_vision_model.cpp').read_text()
    for name in ('g05_k1_camera_group_available','is_g05_k1_camera_group_profile','g05_k1_camera_group_arguments'):
        stub=re.search(r' [^\n]*\b'+name+r'\([^\n]*\}\n',cpp)
        assert stub,name
        cpp=cpp[:stub.start()]+cpp[stub.end():]
    methods='\n'.join(_source_method(source,sig) for sig in (
        'bool g05_k1_camera_group_available(int64_t cameras) const',
        'bool is_g05_k1_camera_group_profile() const',
        'std::vector<int64_t> g05_k1_camera_group_arguments() const'))
    cpp=cpp.replace(' std::vector<int64_t>kvinsert_cost_weight_identity()const',methods+'\n std::vector<int64_t>kvinsert_cost_weight_identity()const')
    # The actual cold source links without either dispatcher-private helper.
    cpp_without_comments = re.sub(r'//[^\n]*|/\*.*?\*/', '', cpp, flags=re.S)
    assert 'autotile_linear_kernel_id' not in cpp_without_comments
    assert 'autotile_linear_acc32_kernel_id' not in cpp_without_comments
    cpp=cpp.replace('bool qkv=false,w8=false,acc16=false,owner=false;', 'bool qkv=false,w8=false,acc16=false,owner=false,generated16=true,generated32=true;')
    cpp=cpp.replace('void* get(KernelId id){return has_loaded(id)?this:nullptr;}', 'void* get(KernelId id){++queries;for(bool acc32:{false,true})for(int cameras:{2,3}){auto tile=acc32?rpu_pl_tiling::select_tile_acc32(cameras*256,1024,512,false):rpu_pl_tiling::select_tile_acc16(cameras*256,1024,512,false);if(id==parallel_linear_kernel_id(rpu_pl_tiling::WeightType::Int8,acc32,tile))return (acc32?generated32:generated16)?this:nullptr;}throw std::runtime_error("generic only");}')
    # Production atomic has_loaded has no registered snapshot for generic IDs.
    cpp=cpp.replace('bool has_loaded(KernelId id){++queries;', 'bool has_loaded(KernelId id){++queries;if(static_cast<size_t>(id)>=static_cast<size_t>(KernelId::PL_AUTOTILE_BEGIN))throw std::runtime_error("unregistered generic availability");',1)
    cpp=cpp.replace(' bool g05_k1_b3_available_=false;', ' bool g05_k1_b3_available(int64_t=0)const{return g05_k1_b3_available_;}\n bool g05_k1_b3_available_=false;')
    cpp+=r'''
int main(){
 for(bool acc32:{false,true}){
  Model m;m.linear_acc32_=acc32;
  auto&cache=KernelCache::instance();cache.owner=cache.w8=cache.acc16=cache.qkv=cache.generated16=cache.generated32=true;
  for(auto&w:m.layer_weights_){at::Tensor*weights[]={&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.fc1_w,&w.fc2_w};at::Tensor*scales[]={&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.fc1_ws,&w.fc2_ws};
   for(int i=0;i<6;++i){weights[i]->dtype=at::kChar;weights[i]->rows=i==4?4096:1024;weights[i]->cols=i==5?4096:1024;scales[i]->present=true;scales[i]->dtype=at::kHalf;scales[i]->rank=1;scales[i]->rows=weights[i]->rows;}}
  m.prepare_g05_owner_ln();assert(m.g05_k1_generated_group_available_[0]);assert(m.g05_k1_generated_group_available_[1]==acc32);
  m.current_camera_batch_count_=2;m.current_output_patches_=512;assert(m.g05_k1_camera_group_available(2)&&m.is_g05_k1_camera_group_profile());
  const int queries=cache.queries;for(int i=0;i<100;++i){assert(m.g05_k1_camera_group_available(2));assert(m.is_g05_k1_camera_group_profile());assert(m.g05_k1_camera_group_arguments()[11]==(acc32?32:16));}assert(cache.queries==queries);
  if(acc32)cache.generated32=false;else cache.generated16=false;
  m.prepare_g05_owner_ln();assert(!m.g05_k1_generated_group_available_[0]&&!m.g05_k1_camera_group_available(2));
  cache.generated16=cache.generated32=true;m.prepare_g05_owner_ln();assert(m.g05_k1_camera_group_available(2));
  m.layer_weights_.back().fc2_ws.contiguous=false;m.prepare_g05_owner_ln();assert(!m.g05_k1_generated_group_available_[0]&&!m.g05_k1_generated_group_available_[1]);
 }
}
'''
    _compile_qkv(tmp_path,cpp)


def test_camera_group_cold_linkage_and_real_private_dispatch_mapping(tmp_path, monkeypatch):
    """Link the actual cold methods without invented dispatcher-global symbols."""
    import ctypes
    import subprocess
    import test_g05_vision_qkv_spm_native as qkv

    vision=(ROOT/'src/fused/rpu_qwen3_5_vision_model.cpp').read_text()
    assert 'autotile_linear_kernel_id' not in vision
    assert 'autotile_linear_acc32_kernel_id' not in vision
    captured=[]
    with monkeypatch.context() as scope:
        scope.setattr(qkv,'_compile_qkv',lambda path,cpp:captured.append(cpp))
        test_camera_group_cold_actual_generated_capabilities_and_no_hot_lookup(tmp_path)
    assert len(captured)==1
    cold_cpp=tmp_path/'cold_link.cpp'
    cold_so=tmp_path/'cold_link.so'
    cold_cpp.write_text(captured[0])
    result=subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-fPIC','-shared','-Wl,--no-undefined','-I',str(ROOT/'src/core'),str(cold_cpp),'-o',str(cold_so)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    undefined=subprocess.run(['nm','-D','--undefined-only',str(cold_so)],capture_output=True,text=True,check=True).stdout
    assert 'autotile_linear' not in undefined
    assert ctypes.CDLL(str(cold_so)).main()==0

    # Preserve the real source's anonymous namespace and compare its two ID
    # choices with the canonical IDs now used directly by Vision cold admission.
    linear=(ROOT/'src/ops/rpu_linear.cpp').read_text()
    from test_qwen3vl_prefill_gateup_launcher import definition
    private_mapping = "namespace {\n" + "\n".join(definition(linear, signature) for signature in (
        "KernelId autotile_linear_kernel_id(bool is_fp16, const rpu_pl_tiling::TilePick& tile)",
        "KernelId autotile_linear_acc32_kernel_id(bool is_fp16, const rpu_pl_tiling::TilePick& tile) {")) + "\n}\n"
    cache_header=(ROOT/'src/core/rpu_kernel_cache.h').read_text()
    begin=cache_header.index('enum class KernelId : uint16_t {')
    enum=cache_header[begin:cache_header.index('};',begin)+2]
    mapping_cpp=tmp_path/'mapping.cpp'
    mapping_so=tmp_path/'mapping.so'
    mapping_cpp.write_text(with_actual_linear_contracts('#include <cstdint>\n#include <stdexcept>\n#define TORCH_CHECK(ok,...) do{if(!(ok))throw std::runtime_error("rejected");}while(false)\n'+enum+'\n'+private_mapping+'''\nextern "C" int selected_acc16(){return static_cast<int>(autotile_linear_kernel_id(false,{48,512,128}));}
extern "C" int canonical_acc16(){return static_cast<int>(KernelId::PL_U_INT8_ACC16_M512N48K128);}
extern "C" int selected_acc32(){return static_cast<int>(autotile_linear_acc32_kernel_id(false,{80,288,128}));}
extern "C" int canonical_acc32(){return static_cast<int>(KernelId::PL_U_INT8_ACC32_M288N80K128);}
'''))
    result=subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-fPIC','-shared','-Wl,--no-undefined',str(mapping_cpp),'-o',str(mapping_so)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    undefined=subprocess.run(['nm','-D','--undefined-only',str(mapping_so)],capture_output=True,text=True,check=True).stdout
    assert 'autotile_linear' not in undefined
    actual=ctypes.CDLL(str(mapping_so))
    assert actual.selected_acc16()==actual.canonical_acc16()
    assert actual.selected_acc32()==actual.canonical_acc32()
    assert actual.canonical_acc16()!=actual.canonical_acc32()
