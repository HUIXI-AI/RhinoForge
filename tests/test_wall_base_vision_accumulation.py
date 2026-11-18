"""Wall base Vision cold accumulation and actual emission, without device access."""
import ast
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from test_wall_base_paired_kv import block, compile_run

ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / 'src/fused/rpu_qwen25vl_vision_model.cpp'
PYTHON = ROOT / 'python/rpu_backend/adapters/wall_oss/vision.py'


def test_actual_accumulation_setter_and_base_geometry_fail_closed(tmp_path):
    source = CPP.read_text()
    methods = '\n'.join(block(source, marker) for marker in (
        '    void set_linear_acc32(', '    void validate_acc32_base_input()',
    ))
    cpp = r'''
#include <cassert>
#include <stdexcept>
#define TORCH_CHECK(ok,...) do{if(!(ok))throw std::runtime_error(#ok);}while(false)
struct Model {
 bool has_rope_=false,z1_adopted_=false,z1_bound_=false,linear_acc32_=false;
 bool mixed_precision_enabled_=false,layer_group_enabled_=false,exact_mimc_enabled_=false;
 bool mixed_precision_=false,per_window_sdpa_enabled_=false,generic=true,pair=false;
 int layers=0,hidden=1280,heads=16,dim=80,inter=3584,invalidations=0;
 int valid_patches_per_image_=0,current_num_patches_=512,image_batch_count_=2;
 int num_layers()const{return layers;}int hidden_size()const{return hidden;}
 int num_q_heads()const{return heads;}int num_kv_heads()const{return heads;}
 int head_dim()const{return dim;}int intermediate_size()const{return inter;}
 bool is_generic()const{return generic;}bool padded_pair_enabled()const{return pair;}
 void invalidate_model_state(){++invalidations;}
 METHODS
};
template<class F>void reject(F f){try{f();assert(false);}catch(const std::runtime_error&) {}}
int main(){
 for(int bad=0;bad<7;++bad){Model m;
  if(bad==0)m.layers=32;if(bad==1)m.has_rope_=true;if(bad==2)m.z1_bound_=true;
  if(bad==3)m.mixed_precision_enabled_=true;if(bad==4)m.layer_group_enabled_=true;
  if(bad==5)m.exact_mimc_enabled_=true;if(bad==6)m.per_window_sdpa_enabled_=true;
  reject([&]{m.set_linear_acc32(true);});assert(!m.linear_acc32_&&m.invalidations==0);
 }
 Model m;m.set_linear_acc32(true);assert(m.linear_acc32_&&m.invalidations==1);m.layers=32;
 for(int images:{1,2,3}){m.image_batch_count_=images;m.current_num_patches_=256*images;m.validate_acc32_base_input();}
 m.image_batch_count_=1;m.current_num_patches_=324;m.validate_acc32_base_input();
 m.image_batch_count_=2;m.current_num_patches_=672;m.valid_patches_per_image_=324;m.pair=true;m.validate_acc32_base_input();
 m.pair=false;reject([&]{m.validate_acc32_base_input();});
 m.valid_patches_per_image_=0;m.current_num_patches_=648;reject([&]{m.validate_acc32_base_input();});
 m.current_num_patches_=512;m.mixed_precision_=true;reject([&]{m.validate_acc32_base_input();});
 m.linear_acc32_=false;m.validate_acc32_base_input(); // Original RTC/large-shape admission untouched.
}
'''.replace('METHODS', methods)
    compile_run(tmp_path, 'wall_vision_accumulation_cold', cpp)


def test_actual_32_layer_projection_handoff_and_all_nine_linear_flags(tmp_path, monkeypatch):
    """Extend the existing actual-emitter fixture, checking the real final bool."""
    import test_wall_base_vision_owner_ring_norm as owner
    source = CPP.read_text()
    layer = block(source, '    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override')
    remaining = []
    for weight in ('lw.q_w', 'lw.k_w', 'lw.v_w', 'merger_m0_w_', 'merger_m2_w_'):
        search = layer if weight.startswith('lw.') else source[source.index('        // 2. m0 GEMM'):]
        start = search.index('rpu_launch_linear_spm_to_spm_acc16_kernel(')
        while weight not in search[start:search.index(';', start)]:
            start = search.index('rpu_launch_linear_spm_to_spm_acc16_kernel(', start+1)
        remaining.append(search[start:search.index(';', start)+1])
    probe = '''
 void probe_other_linears(){
  const int64_t layer_idx=0,seq_len=672,h=1280,nq=16,hd=80,mrows=168,mh=5120,oh=2048;
  const auto& lw=layer_weights_.front();at::Tensor merger_m0_w_,merger_m2_w_;
  const uint32_t normalized_addr=INPUT,partial_addr=PARTIAL;
  EXTRA
 }
'''.replace('EXTRA', '\n'.join(remaining))
    def build(path, name, cpp):
        cpp = cpp.replace('int o_calls=0,', 'bool expected_force=false,extra=false;int extra_calls=0;\nint o_calls=0,')
        old = '''template<class... A>void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t in,
 const at::Tensor&,uint32_t out,A...){linear(in,out);}'''
        assert old in cpp
        cpp = cpp.replace(old, '''void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t in,
 const at::Tensor&,uint32_t out,int64_t,int64_t,int64_t,int,int,uint32_t,
 const at::Tensor&,uint32_t scale,uint16_t layer,bool force){
 assert(scale==0&&layer==0&&force==expected_force);
 if(extra){++extra_calls;return;}linear(in,out);
}''')
        cpp = cpp.replace('const at::Tensor&,uint32_t out,A...){linear(in,out);}',
                          'const at::Tensor&,uint32_t out,A...){assert(!expected_force);linear(in,out);}')
        cpp = cpp.replace('assert(content(in)==FULL_NORM&&in==PARTIAL&&out==GATE);',
                          'assert(!expected_force);assert(content(in)==FULL_NORM&&in==PARTIAL&&out==GATE);')
        cpp = cpp.replace(' void emit_owner_chain(int64_t layer_idx){', probe+'\n void emit_owner_chain(int64_t layer_idx){')
        cpp = cpp.replace('for(bool fp16:{false,true})for(int variant:{0,1,2}){',
                          'for(bool acc:{false,true})for(bool fp16:{false,true})for(int variant:{0,1,2}){')
        cpp = cpp.replace('Model m=exact(672,fp16);',
                          'Model m=exact(672,fp16);m.linear_acc32_=acc;expected_force=acc;')
        cpp = cpp.replace('assert(linear_calls==(fp16?128:64)&&mlp_calls==32);',
                          'assert(linear_calls==((fp16||acc)?128:64)&&mlp_calls==32);\n'
                          'extra=true;extra_calls=0;m.probe_other_linears();assert(extra_calls==5);extra=false;')
        compile_run(path, name+'_accumulation', cpp)
    monkeypatch.setattr(owner, 'compile_run', build)
    owner.test_actual_fp16_w8_owner_routes_and_full_layer_handoffs(tmp_path)


def test_actual_projection_manifest_and_accumulation_replay_identity(tmp_path):
    import re
    from test_wall_base_vision_ragged_routes import _source
    source = CPP.read_text()
    manifest = block(source, '    FmbPhysicalExecutionManifest physical_manifest_for_candidate(')
    mode_start = manifest.index('        manifest.linear_accumulation =')
    mode = manifest[mode_start:manifest.index(';', mode_start)+1]
    start = manifest.index('                const int64_t row_route = small_image_acc16_row_projection_enabled')
    producer = manifest[start:manifest.index('                if (small_image_qk_rope_enabled', start)]
    sites = '\n'.join(re.findall(r'constexpr int64_t QWEN25_[A-Z0-9_]+\s*=\s*\d+LL;', source))
    for name in ('QWEN25_SMALL_RAGGED_SDPA_SITE', 'QWEN25_WALL_SORTING_SITE'):
        sites = re.sub(r'constexpr int64_t '+name+r'\s*=\s*\d+LL;', '', sites)
    enums = ('namespace v3 {enum class FmbLinearRouteSelector {AUTO_TILE=1};}\n'
             +block(source, 'enum class Qwen25VisionLinearRoute')+';\n'
             +block(source, 'constexpr int64_t qwen25_vision_linear_route()'))
    cpp = _source()
    cpp = cpp.replace('struct Model {', sites+'\n'+enums+'\nstruct Model {',1)
    methods = block(source, '    std::vector<int64_t> vision_row_arguments(')+'\n'+block(source, '    std::vector<int64_t> small_image_gate_up_arguments(')+r'''
 auto projection_manifest(const ChunkInfo& chunk){
  FmbPhysicalExecutionManifest manifest;
  MODE
  auto append=[&](FmbRouteFamily family,int64_t site_id,int64_t selector,int64_t invocation=0,
                 std::vector<int64_t> arguments={},int64_t flags=0){
   manifest.routes.push_back({site_id,family,selector,flags,std::move(arguments),invocation});};
  auto append_linear=[&](int64_t site,int64_t invocation){append(FmbRouteFamily::LINEAR,site,qwen25_vision_linear_route(),invocation);};
  const bool has_boundary_layer=false;
  PRODUCER
  return manifest;
 }
'''.replace('MODE',mode).replace('PRODUCER',producer)
    cpp=cpp.replace(' void bind(){',methods+'\n void bind(){',1)
    cpp+=r'''
int main(){
 publish();
 for(bool fp16:{false,true})for(bool acc:{false,true})for(int rows:{324,512,672,768}){
  auto m=exact(rows,fp16);m.linear_acc32_=acc;
  m.small_ragged_sdpa_available_=false;
  if(rows==512||rows==768){m.image_batch_count_=rows/256;m.schedule_=VisionSchedule::MISC;
   m.cu_window_={0,64,128,192,256};}
  auto p=m.projection_manifest({0,0,rows,rows});
  assert(p.linear_accumulation==(acc?FmbLinearAccumulationPolicy::ACC32:FmbLinearAccumulationPolicy::ACC16));
  assert(m.physical_manifest_forward_capability(p).supported);
  auto wrong=p;wrong.linear_accumulation=acc?FmbLinearAccumulationPolicy::ACC16:FmbLinearAccumulationPolicy::ACC32;
  assert(!m.physical_manifest_forward_capability(wrong).supported);
  int fused=0,automatic=0,row=0;
  for(auto& r:p.routes){
   fused+=r.selector==int64_t(Qwen25VisionLinearRoute::WALL_SINGLE_BASE_GATE_UP_SWIGLU);
   row+=r.selector==int64_t(Qwen25VisionLinearRoute::WALL_ROW_FULL_LOCAL_K);
   automatic+=r.selector==qwen25_vision_linear_route();
  }
  bool gate_fusion=!fp16&&!acc&&(rows==324||rows==672);
  assert(fused==int(gate_fusion));assert(row==(acc?0:2));
  assert(automatic==(gate_fusion?3:5)+(acc?2:0));
  assert(p.routes.size()==size_t(gate_fusion?6:7));
  auto old=m;old.linear_acc32_=false;auto old_args=old.sorting_schedule_arguments();
  auto args=m.sorting_schedule_arguments();
  if(acc){assert(args.back()==32);args.pop_back();}assert(args==old_args);
  if(rows==512||rows==768)assert(m.small_image_native_window_enabled());
  if(rows==672)assert(m.padded_pair_enabled());
 }
}
'''
    compile_run(tmp_path, 'wall_vision_accumulation_manifest', cpp)


def _function(name, namespace, *, nested=False):
    tree = ast.parse(PYTHON.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(PYTHON), 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('control,mode,rejected', [
    ('ordinary', False, False), ('ordinary', True, False),
    ('ordinary', 1, True), ('mixed', True, True),
    ('exact', True, True), ('layer_group', True, True),
    ('per_window', True, True), ('mixed', False, False),
])
def test_actual_builder_checks_cold_mode_before_checkpoint_access(monkeypatch, control, mode, rejected):
    from typing import Sequence
    execution = ModuleType('rpu_backend.api._execution')
    execution.normalize_rpu_execution = runpy.run_path(
        str(ROOT/'python/rpu_backend/api/_execution.py'))['normalize_rpu_execution']
    execution.ExecutionSession = type('ExecutionSession', (), {})
    execution._require_execution_process_safe = lambda: None
    monkeypatch.setitem(sys.modules, execution.__name__, execution)
    tree=ast.parse(PYTHON.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_wall_oss_vision')
    stop=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='ckpt_dir' for t in n.targets))
    node.body=node.body[:stop]+[ast.Return(value=ast.Name(id='linear_acc32',ctx=ast.Load()))]
    ast.fix_missing_locations(node)
    env=dict(Sequence=Sequence, WallOssVision=type('WallOssVision',(),{}),
             _claim_experimental_schedule=lambda: ((0,True) if control=='exact' else
                                                    (32,False) if control=='layer_group' else (0,False)),
             rpu_env_bool=lambda name,**kw: (control=='mixed' if name.endswith('MIXED_PRECISION')
                                             else name.endswith('FUSED_MERGER')),
             sorting_config_from_env=lambda:SimpleNamespace(fp16_mask=7,vision_row=True,silu_mul=True))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(PYTHON),'exec'),env)
    kwargs=dict(window=True,execution_config={'vision':{'linear_acc32':mode}},
                per_window_sdpa=control=='per_window')
    if rejected:
        with pytest.raises((ValueError,TypeError)):
            env['build_wall_oss_vision'](**kwargs)
    else:
        assert env['build_wall_oss_vision'](**kwargs) is mode


def test_actual_python_pending_handle_binds_before_weights_and_cleans_failure(monkeypatch):
    events = []
    execution = ModuleType('rpu_backend.api._execution')
    execution._require_execution_process_safe = lambda: None
    monkeypatch.setitem(sys.modules, execution.__name__, execution)
    retirement = ModuleType('rpu_backend.adapters.wall_oss._retirement')
    retirement._cleanup_build_failure = lambda error, keepalive, cleanup: cleanup()
    monkeypatch.setitem(sys.modules, retirement.__name__, retirement)
    ops = SimpleNamespace(qwen25vl_vision_create=lambda: 17)
    for name in ('set_execution_routes', 'set_linear_acc32', 'set_chunk_envelope',
                 'set_chunk_size', 'set_rope', 'set_merger_weights'):
        setattr(ops, 'qwen25vl_vision_'+name, lambda *a, _n=name: events.append((_n, a)))
    ops.qwen25vl_vision_position_idx_keepalive = lambda handle: 'keepalive'
    env = {'torch': SimpleNamespace(ops=SimpleNamespace(rpu=ops)), '_RMSNORM_CAP_BASE': 1,
           '_destroy_qwen25vl_vision_handle': lambda handle: events.append(('destroy', handle))}
    configure = _function('_configure_qwen25vl_vision_handle', env)
    args = dict(set_weights=lambda *a: events.append(('weights', a)), weight_args=(),
                freq_cos='cos',freq_sin='sin',merger_args=(),max_seq_len=4096)
    for mode in (False, True):
        events.clear(); assert configure(**args, linear_acc32=mode) == (17, 'keepalive')
        names = [n for n, _ in events]
        assert ('set_linear_acc32' in names) == mode
        if mode:
            assert names.index('set_execution_routes') < names.index('set_linear_acc32') < names.index('weights')
            assert events[names.index('set_linear_acc32')][1] == (17, True)
    def reject(*args): raise RuntimeError('fixed RTC mode')
    ops.qwen25vl_vision_set_linear_acc32 = reject
    events.clear()
    with pytest.raises(RuntimeError, match='fixed RTC'):
        configure(**args, linear_acc32=True)
    assert events[-1] == ('destroy', 17) and not any(n=='weights' for n,_ in events)


@pytest.mark.parametrize('mode', (False, True))
def test_actual_device_patch_and_eager_merger_use_typed_acc32_only(mode):
    calls=[]
    class Value:
        def to(self,*a,**k): return self
        def reshape(self,*a): return self
        def unsqueeze(self,*a): return self
        def contiguous(self): return self
    value=Value()
    ops=SimpleNamespace(linear_with_accumulation=lambda *a: (calls.append(('typed',a)) or value),
                        rms_norm=lambda *a:value)
    env=dict(torch=SimpleNamespace(Tensor=Value,ops=SimpleNamespace(rpu=ops)),
             F=SimpleNamespace(linear=lambda *a:(calls.append(('legacy',a)) or value),gelu=lambda x:x),
             _VISION_CORES=8,HID=1280,ln_q_w_rpu='norm',EPS=1e-6,merge_hidden=5120,
             m0_w_rpu='w0',m0_b_rpu='b0',m2_w_rpu='w2',m2_b_rpu='b2',linear_acc32=mode)
    patch=_function('_consume_patch_embed_input',env)
    patch(SimpleNamespace(_linear_acc32=mode),value,'patch',None,324,4)
    merger=_function('merger',env)
    assert merger(value,device_out=True) is value
    assert len(calls)==3 and {kind for kind,_ in calls}==({'typed'} if mode else {'legacy'})
    if mode:
        assert all(args[3:]==(1,8,True) for _,args in calls)
        assert [args[1] for _,args in calls]==['patch','w0','w2']
