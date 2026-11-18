"""Execute the partial Down cold owner, COMPLETE routes and actual caller."""
import importlib.util
from pathlib import Path
import re
import subprocess

from test_g05_q_gate_native import _compile_run
from test_g05_qkv_gate_native import _qkv_gate_model_cpp
from test_g05_vision_owner_model import _block

ROOT = Path(__file__).resolve().parents[1]


def test_partial_down_owner_reset_compiles_with_real_aten(tmp_path):
    """A Tensor double cannot expose ATen's ambiguous ``owner = {}`` overloads."""
    source = (ROOT / 'src/fused/rpu_qwen3_5_model.cpp').read_text()
    statements = []
    for marker in ('void Qwen3_5Model::prepare_action_down_partial(',
                   'void Qwen3_5Model::set_weights('):
        body = _block(source, marker)
        assignments = re.findall(r'^\s*action_down_partial_\s*=\s*[^;]+;',
                                 body, re.MULTILINE)
        assert assignments, f'missing owner reset in {marker}'
        statements.append(assignments[0])
    torch_spec = importlib.util.find_spec('torch')
    assert torch_spec is not None and torch_spec.origin
    include = Path(torch_spec.origin).parent / 'include'
    assert (include / 'ATen/core/Tensor.h').is_file()
    cpp = tmp_path / 'g05_down_real_aten_reset.cpp'
    cpp.write_text('#include <ATen/core/Tensor.h>\n' + '\n'.join(
        f'void reset_{index}(at::Tensor& action_down_partial_) {{{statement}\n}}'
        for index, statement in enumerate(statements)))
    compiled = subprocess.run(
        ['c++', '-std=c++17', '-fsyntax-only', '-I', str(include), str(cpp)],
        capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr


def test_partial_down_cold_complete_fallback_and_lifetime(tmp_path):
    source = (ROOT / 'src/fused/rpu_qwen3_5_model.cpp').read_text()
    cpp = _qkv_gate_model_cpp().split('int main(){', 1)[0]
    cpp = cpp.replace('idx=2,offset=0', 'idx=0,offset=0')
    cpp = cpp.replace('bool present=true,contiguous=true,replicated=true;',
                      'uint64_t address=0x100000;int64_t storage_offset_=0;bool present=true,contiguous=true,replicated=true;')
    cpp = cpp.replace('int dim()const{return rank;}',
                      'int64_t storage_offset()const{return storage_offset_;}void*data_ptr()const{return reinterpret_cast<void*>(address);}size_t nbytes()const{return rows*(rank==2?cols:1)*(dtype==at::kHalf?2:1);}int dim()const{return rank;}')
    cpp = cpp.replace('DeviceType type()const{return value;}',
                      'DeviceType type()const{return value;}bool operator!=(const Device&b)const{return value!=b.value;}')
    cpp = cpp.replace('struct Weights ', 'uint64_t RpuGetDevAddr(void*p){return reinterpret_cast<uint64_t>(p);}\nstruct Weights ', 1)
    cpp = cpp.replace('enum class KernelId{', 'enum class KernelId{G05_DOWN_PARTIAL672_W8A16_ACC16_M32N1024K512,')
    cpp = cpp.replace('bool q=true,kv=true,fanout=true,joint=true;', 'bool down=true;bool q=true,kv=true,fanout=true,joint=true;')
    cpp = cpp.replace('++queries;if(!acc16_available', '++queries;if(id==KernelId::G05_DOWN_PARTIAL672_W8A16_ACC16_M32N1024K512)return down;if(!acc16_available')
    cpp = cpp.replace('LINEAR,ROPE,KV_INSERT,OTHER', 'LINEAR,ROPE,KV_INSERT,OTHER,MUTABLE_DMA')
    native = r'''
int packs=0,preloads=0,generation=0;
at::Tensor rpu_pack_g05_down_partial_w8a16(const at::Tensor&w){assert(w.owner==200&&w.rows==1024&&w.cols==4096);++packs;auto t=w;t.owner=500000+generation;t.rows=8;t.cols=344064;return t;}
void rpu_launch_g05_down_partial_w8a16_preload(const at::Tensor&w,uint32_t out){assert(w.rows==8&&w.cols==344064&&w.owner==500000+generation&&out==0x40700000);++preloads;}
void rpu_launch_g05_down_partial_w8a16_acc16_kernel(uint32_t in,const at::Tensor&w,uint32_t resident,uint32_t out,const at::Tensor&s){assert(in==0x40010000&&resident==0x40700000&&out==0x40020000&&w.owner==200&&s.owner==10200);launches.push_back("partial-down");}
enum class StorageClass{Temp};enum class BufferScope{LayerWide};
struct BufferDecl{const char*name;int64_t size;int first,last;StorageClass storage;int per_layer;const char*alias;BufferScope scope;};
'''
    dma_enum = 'enum class Qwen35TextMutableDmaRoute : int64_t ' + _block(source, 'enum class Qwen35TextMutableDmaRoute') + ';\n'
    cpp = cpp.replace('struct Model{', native + dma_enum + '\nstruct Model{', 1)
    for stub in (
        ' bool use_action_down_partial(int64_t,int64_t)const{return false;}\n',
        ' std::vector<int64_t>action_down_partial_arguments(bool)const{return {};}\n',
        ' void prepare_action_down_partial(){}\n',
        ' template<class...T>void launch_action_down_partial(T...){assert(false);}\n',
        ' bool use_action_o_resident(int64_t)const{return false;}\n',
    ):
        assert stub in cpp
        cpp = cpp.replace(stub, '')
    methods = [
        ('void prepare_action_down_partial()', 'void Qwen3_5Model::prepare_action_down_partial('),
        ('bool use_action_down_partial(int64_t rows,int64_t layer)const', 'bool Qwen3_5Model::use_action_down_partial('),
        ('std::vector<int64_t>action_down_partial_arguments(bool preload)const', 'std::vector<int64_t> Qwen3_5Model::action_down_partial_arguments('),
        ('void preload_action_down_partial()', 'void Qwen3_5Model::preload_action_down_partial('),
        ('void launch_action_down_partial(int layer,const ChunkInfo&chunk)', 'void Qwen3_5Model::launch_action_down_partial('),
        ('bool use_action_o_resident(int64_t rows)const', 'bool Qwen3_5Model::use_action_o_resident('),
    ]
    manifest = _block(source, 'FmbPhysicalExecutionManifest Qwen3_5Model::physical_manifest_for_candidate(')
    declarations = _block(source, 'std::vector<BufferDecl> Qwen3_5Model::declare_legacy_buffers(')
    mlp = _block(source, 'void Qwen3_5Model::emit_mlp_and_output(')
    caller = mlp[mlp.index('    if (use_action_nvfp4_down_resident(seq_len,layer_idx))'):mlp.index('    if (action_mlp_tail_handoff(layer_idx,chunk,true))')]
    additions = '\n'.join(decl + _block(source, marker) for decl, marker in methods) + r'''
 int64_t body_iter=0;
 static int64_t qwen35_text_dma_route(Qwen35TextMutableDmaRoute r){return static_cast<int64_t>(r);}
 void down_plan(){ChunkInfo chunk;int64_t physical_len=32;struct{std::vector<Route>routes;}manifest;if(use_action_down_partial(chunk.len,0))__LINEAR__ if(use_action_down_partial(physical_len,0))__DMA__ routes=manifest.routes;}
 std::vector<BufferDecl>down_buffers(){std::vector<BufferDecl>decls;if(use_action_down_partial(action_len_,0))__DECLARATION__ return decls;}
 void down_stage(int layer_idx,ChunkInfo chunk={}){auto&lw=layer_weights_[layer_idx];int64_t seq_len=chunk.len,h=hidden_size();auto down_scale=lw.down_ws;uint32_t product_buf=addr(0,"gate");__CALLER__}
 void fixture(){linear_acc32_=false;action_o_resident_available_=true;action_o_resident_.resize(6);for(int layer=0;layer<24;++layer){auto&w=layer_weights_[layer];auto&t=w.down_w;t.dtype=at::kChar;t.rows=1024;t.cols=4096;t.owner=200+layer;t.tag="ddr-down";t.address=0x1000000+layer*0x500000;auto&s=w.down_ws;s.rank=1;s.rows=1024;s.owner=10200+layer;s.address=0x10000000+layer*4096;}}
'''
    replacements = {
        '__LINEAR__': _block(manifest, 'if (use_action_down_partial(chunk.len,0))'),
        '__DMA__': _block(manifest, 'if (use_action_down_partial(physical_len,0))'),
        '__DECLARATION__': _block(declarations, 'if (use_action_down_partial(action_len_,0))'),
        '__CALLER__': caller,
    }
    for marker, value in replacements.items():
        additions = additions.replace(marker, value)
    cpp = cpp.replace(' void loop(){', additions + '\n void loop(){')
    cpp = cpp.replace('uint32_t addr(int,const char*n)const' + _block(cpp, 'uint32_t addr(int,const char*n)const'),
                      'uint32_t addr(int,const char*n)const{std::string s(n);return s=="action_down_partial"?0x40700000:s=="gate"?0x40010000:0x40020000;}')
    cpp += r'''
int main(){
 auto&cache=KernelCache::instance();Model m;m.fixture();m.install();assert(packs==1&&m.action_down_partial_available_&&m.action_o_resident_.size()==6);
 assert(!m.use_action_down_partial(32,0)&&m.down_buffers().empty());m.loop();m.down_plan();assert(m.routes.size()==2&&m.use_action_down_partial(32,0));
 auto buffers=m.down_buffers();assert(buffers.size()==1&&buffers[0].size==344064&&buffers[0].first==0&&buffers[0].last==31&&buffers[0].scope==BufferScope::LayerWide);
 assert(m.routes[0].args==std::vector<int64_t>({2,0,24,10,32,1024,4096,512,8,16,672,14,344064,48,48,768,22,0,32,2,20,15,8,8,0,1}));
 auto identity=m.kvinsert_cost_weight_identity();auto hash=m.subclass_layout_hash();int q=cache.queries;
 cache.down=false;for(int step=0;step<10;++step){m.body_iter=step;m.preload_action_down_partial();for(int layer=0;layer<24;++layer){launches.clear();m.down_stage(layer);assert(launches==std::vector<std::string>({layer==0?"partial-down":"ddr-down"}));}}
 assert(cache.queries==q&&preloads==1&&packs==1);cache.down=true;
 m.body_iter=0;m.preload_action_down_partial();assert(preloads==2);
 for(int which=0;which<2;++which)for(size_t a=0;a<m.routes[which].args.size();++a){auto x=m;++x.routes[which].args[a];auto before=preloads;if(which)reject([&]{x.preload_action_down_partial();});else reject([&]{x.down_stage(0);});assert(preloads==before);}
 for(int which=0;which<2;++which)for(int fault=0;fault<8;++fault){auto x=m;auto&r=x.routes[which];if(fault==0)x.complete=false;if(fault==1)++r.selector;if(fault==2)++r.flags;if(fault==3)++r.invocation;if(fault==4)++r.site;if(fault==5)r.family=FmbRouteFamily::OTHER;if(fault==6)r.args.pop_back();if(fault==7)x.routes.clear();auto before=preloads;if(which)reject([&]{x.preload_action_down_partial();});else reject([&]{x.down_stage(0);});assert(preloads==before);}
 for(auto bad:std::vector<ChunkInfo>{{31,0,0},{32,1,0},{32,0,1}})reject([&]{m.launch_action_down_partial(0,bad);});
 for(int layer:{-1,1,23,24})reject([&]{m.launch_action_down_partial(layer,{});});
 for(int body:{-1,10}){auto x=m;x.body_iter=body;reject([&]{x.preload_action_down_partial();});}
 for(int fault=0;fault<7;++fault){auto x=m;if(fault==0)x.action_num_steps_=1;if(fault==1)x.action_step_active_=false;if(fault==2)x.action_loop_active_=false;if(fault==3)x.action_len_=16;if(fault==4)x.action_mode_=false;if(fault==5)x.action_o_resident_.back().present=false;if(fault==6)x.linear_acc32_=true;assert(!x.use_action_down_partial(32,0)&&x.down_buffers().empty());x.down_plan();assert(x.routes.empty());}
 {auto x=m;++x.action_down_partial_.owner;assert(x.kvinsert_cost_weight_identity()!=identity);}
 {auto x=m;++x.layer_weights_[0].down_ws.owner;assert(x.kvinsert_cost_weight_identity()!=identity);}
 // The optional payload can disappear only at cold installation; all 24 O owners remain intact.
 cache.down=false;auto old=m;old.install();old.down_plan();assert(!old.action_down_partial_available_&&old.action_o_resident_.size()==6&&old.use_action_o_resident(32)&&old.routes.empty());assert(old.subclass_layout_hash()!=hash&&old.kvinsert_cost_weight_identity()!=identity);launches.clear();old.down_stage(0);assert(launches==std::vector<std::string>({"ddr-down"}));cache.down=true;
 // Malformed layer0 owners cannot become a partial cache or launch a transfer.
 for(int fault=0;fault<16;++fault){auto x=m;auto&w=x.layer_weights_[0].down_w;auto&s=x.layer_weights_[0].down_ws;if(fault==0)w.dtype=at::kHalf;if(fault==1)w.cols=2048;if(fault==2)w.storage_offset_=256;if(fault==3)w.contiguous=false;if(fault==4)w.address=0;if(fault==5)++w.address;if(fault==6)w.address=(UINT64_C(1)<<40)-256;if(fault==7)s.address=w.address;if(fault==8)s.rows=512;if(fault==9)s.storage_offset_=128;if(fault==10)s.location=c10::DeviceType::CPU;if(fault==11)s.dtype=at::kChar;if(fault==12)s.present=false;if(fault==13)x.linear_acc32_=true;if(fault==14)x.action_o_resident_available_=false;if(fault==15)w.present=false;int before=packs;x.prepare_action_down_partial();assert(!x.action_down_partial_available_&&packs==before&&x.action_o_resident_.size()==6);}
 ++generation;m.install();assert(m.action_down_partial_.owner==500001&&m.kvinsert_cost_weight_identity()!=identity);
 m.reset();assert(!m.action_down_partial_available_&&m.action_down_partial_.owner==0&&!m.use_action_down_partial(32,0));m.down_plan();assert(m.routes.empty());
}
'''
    _compile_run(tmp_path, 'partial_down_model', cpp)
