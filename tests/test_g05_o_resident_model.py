"""Execute actual resident-O cold/planner/caller bodies with host owner doubles."""
from pathlib import Path
import re

from test_g05_q_gate_native import _compile_run
from test_g05_qkv_gate_native import _qkv_gate_model_cpp
from test_g05_vision_owner_model import _block

ROOT = Path(__file__).resolve().parents[1]


def test_o_resident_cold_owners_complete_fallback_and_lifetime(tmp_path):
    source = (ROOT / 'src/fused/rpu_qwen3_5_model.cpp').read_text()
    cpp = _qkv_gate_model_cpp().split('int main(){', 1)[0]
    cpp = cpp.replace('idx=2,offset=0', 'idx=0,offset=0')
    cpp = cpp.replace('bool present=true,contiguous=true,replicated=true;',
                      'uint64_t address=0x100000;int64_t storage_offset_=0;bool present=true,contiguous=true,replicated=true;')
    cpp = cpp.replace('int dim()const{return rank;}',
                      'int64_t storage_offset()const{return storage_offset_;}void*data_ptr()const{return reinterpret_cast<void*>(address);}int64_t numel()const{return present?rows*(rank==2?cols:1):0;}size_t nbytes()const{return rows*(rank==2?cols:1)*(dtype==at::kHalf?2:1);}int dim()const{return rank;}')
    cpp = cpp.replace('DeviceType type()const{return value;}',
                      'DeviceType type()const{return value;}bool operator!=(const Device&b)const{return value!=b.value;}')
    cpp = cpp.replace('struct Weights ', 'uint64_t RpuGetDevAddr(void*p){return reinterpret_cast<uint64_t>(p);}\nstruct Weights ', 1)
    ids = ['G05_O_RESIDENT_W8A16_ACC16_M32N1024K256', 'G05_GATE_UP_W8A16_ACC32_M32N512K1024',
           'G05_GATE_UP_W8A16_ACC16_M32N512K1024', 'G05_GATE_UP_FP16_ACC32_M32N512K1024',
           'G05_GATE_UP_FP16_ACC16_M32N512K1024', 'G05_GATE_UP_PAIRED_NVFP4_ACC32_M32N512K1024_N128',
           'G05_SILU_ROUND_MUL_M32N512', 'G05_GATEUP_RESIDENT_W8A16_ACC16_M352N96K128',
           'G05_GATEUP_RESIDENT_W8A16_ACC32_M240N96K128', 'G05_GATEUP_RESIDENT_FP16_ACC32_M304N80K128']
    cpp = cpp.replace('enum class KernelId{', 'enum class KernelId{' + ','.join(ids) + ',')
    cpp = cpp.replace('bool q=true,kv=true,fanout=true,joint=true;',
                      'bool o=true,pair=true,silu=true;bool q=true,kv=true,fanout=true,joint=true;')
    cpp = cpp.replace('++queries;if(!acc16_available',
                      '++queries;if(id==KernelId::G05_O_RESIDENT_W8A16_ACC16_M32N1024K256)return o;'
                      'if(id==KernelId::G05_SILU_ROUND_MUL_M32N512)return silu;'
                      'if(id==KernelId::G05_GATE_UP_W8A16_ACC16_M32N512K1024)return pair;'
                      'if(!acc16_available')
    cpp = cpp.replace('LINEAR,ROPE,KV_INSERT,OTHER', 'LINEAR,ROPE,KV_INSERT,OTHER,MUTABLE_DMA,ALL_REDUCE,ACTIVATION')
    native = r'''
int opacks=0,gpacks=0,owner_generation=0;
std::vector<std::vector<int64_t>> packed_layers;
std::vector<std::pair<int64_t,uint32_t>> dma_calls;
std::vector<int64_t> o_scales;bool expected_n128=true;
at::Tensor rpu_pack_g05_o_resident_w8a16(const std::vector<at::Tensor>&ws){
 assert(ws.size()==4);std::vector<int64_t>ids;for(int i=0;i<4;++i){assert(ws[i].dtype==at::kChar&&ws[i].rows==1024&&ws[i].cols==2048&&ws[i].owner==ws[0].owner+i);ids.push_back(ws[i].owner);}
 packed_layers.push_back(ids);++opacks;auto t=ws[0];t.owner=500000+1000*owner_generation+ws[0].owner-200;t.rows=8;t.cols=1048576;return t;
}
at::Tensor rpu_pack_g05_gateup_resident_w8a16(const at::Tensor&g,const at::Tensor&u){assert(g.dtype==at::kChar&&u.dtype==at::kChar);++gpacks;return g;}
at::Tensor rpu_pack_g05_gateup_resident_fp16(const at::Tensor&g,const at::Tensor&){++gpacks;return g;}
void rpu_launch_g05_gateup_resident_w8a16_preload(const at::Tensor&t,uint32_t dst){assert(t.dtype==at::kChar&&t.rows==8&&t.cols==1048576);dma_calls.push_back({t.owner,dst});}
void rpu_launch_g05_o_resident_w8a16_acc16_kernel(uint32_t in,uint32_t w,uint32_t out,const at::Tensor&s,bool n128){assert(n128==expected_n128);assert(in==0x40010000&&out==0x40020000);auto layer=(w-0x40100000)/262144;assert(layer<24&&w==0x40100000+layer*262144&&s.owner==10200+layer&&s.rows==1024&&s.rank==1);o_scales.push_back(s.owner);launches.push_back("resident-o");}
void rpu_prepare_ring_all_reduce_input(uint32_t out,int64_t rows,int64_t h,int64_t tp,int64_t cores){assert(out==0x40020000&&rows==32&&h==1024&&tp==8&&cores==8);launches.push_back("prepare-ring");}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t out,uint32_t residual,uint32_t dst,int64_t rows,int64_t h,int64_t tp,int64_t cores){assert(out==0x40020000&&residual==0x40030000&&dst==0x40040000&&rows==32&&h==1024&&tp==8&&cores==8);launches.push_back("ring");}
enum class StorageClass{Temp};enum class BufferScope{LayerWide};
struct BufferDecl{const char*name;int64_t size;int first,last;StorageClass storage;int per_layer;const char*alias;BufferScope scope;};
'''
    enum = 'enum class Qwen35TextMutableDmaRoute : int64_t ' + _block(source, 'enum class Qwen35TextMutableDmaRoute') + ';\n'
    enum += 'enum class Qwen35TextAllReduceRoute : int64_t ' + _block(source, 'enum class Qwen35TextAllReduceRoute') + ';\n'
    constants = '\n'.join(re.findall(r'constexpr int64_t QWEN35_TEXT_FULL_(?:PREPARE_)?ALL_REDUCE_SITE\s*=.*?;', source, re.S))
    cpp = cpp.replace('struct Model{', native + enum + constants + '\nstruct Model{', 1)
    # Replace only the helper's intentional out-of-scope doubles with real bodies.
    cpp = cpp.replace(' bool use_action_o_resident(int64_t)const{return false;}\n', '')
    cpp = cpp.replace(' std::vector<int64_t>action_o_resident_arguments(int64_t,bool)const{return {};}\n', '')
    cpp = cpp.replace('bool use_action_gateup_resident(int64_t)const{return false;}', '')
    cpp = cpp.replace('bool use_action_gateup_pair(int64_t)const{return false;}', '')
    cpp = cpp.replace('bool use_action_silu_round_mul(int64_t)const{return false;}', '')
    old = 'void prepare_action_gateup_resident(){action_mod_body_w8_=layer_weights_[0].q_w.dtype==at::kChar;}void prepare_action_gateup_pair(){}void prepare_action_silu_round_mul(){}void prepare_action_o_resident(){}'
    assert old in cpp
    methods = [
        ('void prepare_action_o_resident()', 'void Qwen3_5Model::prepare_action_o_resident('),
        ('bool use_action_o_resident(int64_t rows)const', 'bool Qwen3_5Model::use_action_o_resident('),
        ('std::vector<int64_t>action_o_resident_arguments(int64_t layer,bool preload)const', 'std::vector<int64_t> Qwen3_5Model::action_o_resident_arguments('),
        ('void preload_action_o_resident()', 'void Qwen3_5Model::preload_action_o_resident('),
        ('void launch_action_o_resident(int layer,const ChunkInfo&chunk)', 'void Qwen3_5Model::launch_action_o_resident('),
    ]
    for stem in ['gateup_resident', 'gateup_pair', 'silu_round_mul']:
        methods += [(f'void prepare_action_{stem}()', f'void Qwen3_5Model::prepare_action_{stem}('),
                    (f'bool use_action_{stem}(int64_t rows)const', f'bool Qwen3_5Model::use_action_{stem}(')]
    methods += [(f'std::vector<int64_t>action_{stem}_arguments()const', f'std::vector<int64_t> Qwen3_5Model::action_{stem}_arguments(')
                for stem in ['gateup_pair', 'silu_round_mul']]
    cpp = cpp.replace(old, '\n'.join(d + _block(source, marker) for d, marker in methods))
    cpp = cpp.replace('uint32_t addr(int,const char*n)const' + _block(cpp, 'uint32_t addr(int,const char*n)const'),
                      'uint32_t addr(int,const char*n)const{std::string s(n);return s=="action_o_resident"?0x40100000:s=="output"?0x40010000:s=="oproj"?0x40020000:s=="residual1"?0x40030000:0x40040000;}')
    manifest = _block(source, 'FmbPhysicalExecutionManifest Qwen3_5Model::physical_manifest_for_candidate(')
    decls = _block(source, 'std::vector<BufferDecl> Qwen3_5Model::declare_legacy_buffers(')
    linear = _block(manifest, 'if (use_action_o_resident(chunk.len))')
    dma = _block(manifest, 'if (use_action_o_resident(physical_len))')
    allocation = _block(decls, 'if (use_action_o_resident(action_len_))')
    attention = _block(source, 'void Qwen3_5Model::build_full_attention(')
    attention = attention[attention.index('    // O-proj row-partition'):attention.rindex('}')]
    # Action tail and Text owner fusion are separately exercised; this
    # fixture enters the original Action RTC/ordinary ring fallback.
    first=attention.index('    if (action_norm_emitted')
    fallback=attention.index('if (use_action_ring_rtc_residual(seq_len))',first)
    attention=attention[:first]+'    '+attention[fallback:]

    additions = r'''
 int64_t body_iter=0;
 static int64_t qwen35_text_dma_route(Qwen35TextMutableDmaRoute r){return static_cast<int64_t>(r);}
 int64_t text_ring_route(int64_t,int64_t)const{return 2;}
 uint32_t layer_input_residual_addr(int,const ChunkInfo&)const{return 0x40030000;}
 void consume_manifest_route(FmbRouteFamily f,int64_t site,int64_t,int64_t){assert(f==FmbRouteFamily::ALL_REDUCE&&(site==QWEN35_TEXT_FULL_PREPARE_ALL_REDUCE_SITE||site==QWEN35_TEXT_FULL_ALL_REDUCE_SITE));}
 void apply_adaptive_residual_gate(uint32_t,uint32_t,const ChunkInfo&){launches.push_back("adaptive");}
 void o_stage(int layer_idx,ChunkInfo chunk={}){auto&lw=layer_weights_[layer_idx];int64_t seq_len=chunk.len,h=hidden_size(),nq=num_q_heads(),hd=head_dim(),tp=attn_tp();__O_CALLER__}
 void o_plan(){ChunkInfo chunk;int64_t physical_len=32;struct{std::vector<Route>routes;}manifest;if(use_action_o_resident(chunk.len))__LINEAR__ if(use_action_o_resident(physical_len))__DMA__ routes=manifest.routes;}
 std::vector<BufferDecl>o_buffers(){std::vector<BufferDecl>decls;if(use_action_o_resident(action_len_))__ALLOCATION__ return decls;}
 void fixture(){linear_acc32_=false;for(int layer=0;layer<24;++layer){auto&w=layer_weights_[layer];for(auto*t:{&w.q_w,&w.k_w,&w.v_w,&w.attn_gate_w,&w.o_w,&w.gate_w,&w.up_w,&w.down_w})t->dtype=at::kChar;w.o_w.rows=1024;w.o_w.cols=2048;w.o_w.owner=200+layer;w.o_w.tag="ddr-o";w.o_ws.rows=1024;w.o_ws.rank=1;w.o_ws.owner=10200+layer;w.gate_w.rows=w.up_w.rows=4096;w.gate_w.cols=w.up_w.cols=1024;w.gate_ws.rows=w.up_ws.rows=4096;w.gate_ws.rank=w.up_ws.rank=1;}}
'''.replace('__O_CALLER__', attention).replace('__LINEAR__', linear).replace('__DMA__', dma).replace('__ALLOCATION__', allocation)
    cpp = cpp.replace(' void loop(){', additions + '\n void loop(){')
    cpp += r'''
int main(){
 auto&cache=KernelCache::instance();Model m;m.fixture();m.install();assert(opacks==6&&gpacks==0&&m.action_o_resident_available_&&m.action_mod_body_w8_&&!m.action_gateup_resident_available_&&m.action_gateup_resident_.empty()&&m.action_gateup_resident_layers()==0);
 assert(packed_layers.size()==6);for(int b=0;b<6;++b)assert(packed_layers[b]==std::vector<int64_t>({200+4*b,201+4*b,202+4*b,203+4*b}));
 assert(!m.use_action_o_resident(32)&&m.o_buffers().empty());m.loop();m.o_plan();assert(m.routes.size()==30&&m.use_action_o_resident(32));
 auto buffers=m.o_buffers();assert(buffers.size()==1&&buffers[0].size==6291456&&buffers[0].first==0&&buffers[0].last==31&&buffers[0].scope==BufferScope::LayerWide);
 assert(m.action_gateup_pair_arguments()[15]==0&&m.action_silu_round_mul_arguments().back()==0);
 assert(m.routes[23].args==std::vector<int64_t>({2,23,24,4,6,10,32,1024,2048,256,8,16,262144,1048576,6291456,64,80,512,13,0,6029312,20,15,8,8,0,1}));
 auto identity=m.kvinsert_cost_weight_identity();auto hash=m.subclass_layout_hash();int q=cache.queries;
 cache.o=false;for(int body=0;body<10;++body){m.body_iter=body;m.preload_action_o_resident();for(int layer=0;layer<24;++layer){launches.clear();m.o_stage(layer);assert(launches==std::vector<std::string>({"prepare-ring","resident-o","ring","adaptive"}));}}
 assert(cache.queries==q&&dma_calls.size()==6&&o_scales.size()==240&&opacks==6&&gpacks==0);cache.o=true;
 for(int b=0;b<6;++b)assert(dma_calls[b]==std::make_pair(int64_t(500000+4*b),uint32_t(0x40100000+b*1048576)));
 m.body_iter=0;m.preload_action_o_resident();assert(dma_calls.size()==12);
 // All preload descriptors are checked before any transfer, including the last bundle.
 for(size_t a=0;a<m.routes.back().args.size();++a){auto x=m;++x.routes.back().args[a];auto n=dma_calls.size();reject([&]{x.preload_action_o_resident();});assert(dma_calls.size()==n);}
 for(size_t a=0;a<m.routes[23].args.size();++a){auto x=m;++x.routes[23].args[a];reject([&]{x.launch_action_o_resident(23,{});});}
 for(int fault=0;fault<7;++fault){auto x=m;auto&r=x.routes[23];if(fault==0)x.complete=false;if(fault==1)++r.selector;if(fault==2)++r.flags;if(fault==3)++r.invocation;if(fault==4)++r.site;if(fault==5)r.family=FmbRouteFamily::OTHER;if(fault==6)x.routes.erase(x.routes.begin()+23);reject([&]{x.launch_action_o_resident(23,{});});}
 for(auto bad:std::vector<ChunkInfo>{{31,0,0},{32,1,0},{32,0,1}})reject([&]{m.launch_action_o_resident(0,bad);});
 reject([&]{m.launch_action_o_resident(-1,{});});reject([&]{m.launch_action_o_resident(24,{});});reject([&]{m.action_o_resident_arguments(1,true);});
 {auto x=m;x.body_iter=10;reject([&]{x.preload_action_o_resident();});}
 for(int fault=0;fault<5;++fault){auto x=m;if(fault==0)x.action_num_steps_=1;if(fault==1)x.action_loop_active_=false;if(fault==2)x.action_step_active_=false;if(fault==3)x.action_len_=16;if(fault==4)x.action_mode_=false;assert(!x.use_action_o_resident(32)&&x.o_buffers().empty());}
 {auto x=m;x.action_o_resident_.back().present=false;assert(!x.use_action_o_resident(32));}
 {auto x=m;++x.action_o_resident_.back().owner;assert(x.kvinsert_cost_weight_identity()!=identity);}
 {auto x=m;++x.layer_weights_[23].o_ws.owner;assert(x.kvinsert_cost_weight_identity()!=identity);}
 // New optional symbol absent keeps all six previous GateUp owners and the original O/ring.
 cache.o=false;auto old=m;old.install();old.o_plan();assert(!old.action_o_resident_available_&&old.action_o_resident_.empty()&&old.use_action_gateup_resident(32)&&old.action_gateup_resident_.size()==6&&gpacks==6&&old.routes.empty());assert(old.action_gateup_pair_arguments()[15]==6&&old.action_silu_round_mul_arguments().back()==6);launches.clear();old.o_stage(23);assert(launches==std::vector<std::string>({"prepare-ring","ddr-o","ring","adaptive"}));assert(old.subclass_layout_hash()!=hash&&old.kvinsert_cost_weight_identity()!=identity);cache.o=true;
 // A missing replacement pair/activation must not evict the old resident MLP.
 for(int missing=0;missing<2;++missing){auto x=m;cache.pair=missing!=0;cache.silu=missing!=1;int before=opacks;x.install();assert(!x.action_o_resident_available_&&x.use_action_gateup_resident(32)&&opacks==before);}cache.pair=cache.silu=true;
 // Validate every O owner before the first cold pack; aligned narrow views fall back.
 for(int layer=0;layer<24;++layer)for(int fault=0;fault<9;++fault){auto x=m;auto&w=x.layer_weights_[layer];if(fault==0)w.o_ws.rows=512;if(fault==1)w.o_ws.storage_offset_=128;if(fault==2)w.o_w.storage_offset_=256;if(fault==3)w.o_ws.address=0;if(fault==4)w.o_w.address+=1;if(fault==5)w.o_ws.address=(UINT64_C(1)<<40)-256;if(fault==6)w.o_w.contiguous=false;if(fault==7)w.o_ws.location=c10::DeviceType::CPU;if(fault==8)w.o_w.cols=1024;int before=opacks;x.install();assert(!x.action_o_resident_available_&&x.action_o_resident_.empty()&&opacks==before&&x.use_action_gateup_resident(32));}
 {auto x=m;x.linear_acc32_=true;int before=opacks;x.install();assert(!x.action_o_resident_available_&&x.use_action_gateup_resident(32)&&opacks==before);}
 {auto x=m;x.layer_weights_[23].down_w.dtype=at::kHalf;int before=opacks;x.install();assert(!x.action_o_resident_available_&&!x.action_mod_body_w8_&&opacks==before);}
 // Unified resident-O capability is frozen cold and consumes tile/grid in every COMPLETE route.
 auto fast=m;fast.install();fast.o_plan();assert(fast.action_o_resident_n128_);
 for(const auto&r:fast.routes){assert(r.args[15]==64&&r.args[16]==80&&r.args[17]==512&&r.args[18]==13);}
 q=cache.queries;
 fast.body_iter=0;fast.preload_action_o_resident();for(int l=0;l<24;++l)fast.o_stage(l);assert(cache.queries==q);
 for(size_t a=0;a<fast.routes[23].args.size();++a){auto bad=fast;++bad.routes[23].args[a];auto count=o_scales.size();reject([&]{bad.launch_action_o_resident(23,{});});assert(o_scales.size()==count);}
 fast.reset();assert(!fast.action_o_resident_n128_&&!fast.action_o_resident_available_);
 ++owner_generation;m.install();m.o_plan();assert(m.action_o_resident_.back().owner==501020&&m.kvinsert_cost_weight_identity()!=identity);
 m.reset();assert(!m.action_o_resident_available_&&m.action_o_resident_.empty()&&!m.action_mod_body_w8_&&!m.use_action_o_resident(32));
}
'''
    _compile_run(tmp_path, 'o_resident_model', cpp)
