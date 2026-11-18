"""Execute actual G05 Q/gate cold, route, caller and launcher bodies on CPU doubles."""
from pathlib import Path
from pi_wall_host_catalog import with_actual_linear_contracts
import subprocess
import re

from test_siglip_cold_profile import _body
from test_g05_vision_owner_model import _block

ROOT = Path(__file__).resolve().parents[1]


def _with_actual_g05_site_helpers(cpp):
    """Keep compiled model doubles on the production precision/site mapping."""
    source = (ROOT / 'src/fused/rpu_qwen3_5_model.cpp').read_text()
    helpers = ('text_gateup_pair_site', 'action_gateup_pair_site',
               'action_nvfp4_swiglu_site', 'action_nvfp4_o_resident_site',
               'action_nvfp4_down_resident_site', 'high_precision_site')
    definitions = []
    constants = []
    if ('has_text_gdn_glast_local_capability(' in cpp and not re.search(
            r'bool\s+has_text_gdn_glast_local_capability\(', cpp)):
        # These focused doubles do not install the independent text GDN local
        # schedule. Its real capability/layout tests exercise that path.
        definitions.append('bool has_text_gdn_glast_local_capability() const { return false; }')
        definitions.append('std::vector<int64_t> text_gdn_glast_local_identity() const '
                           '{ TORCH_CHECK(false, "GDN local schedule is not installed in this fixture"); return {}; }')
    if ('use_text_gdn_input_workspace(' in cpp and not re.search(
            r'bool\s+use_text_gdn_input_workspace\(', cpp)):
        # These fixtures bind the existing G05 profiles. The separate Qwen
        # six-core GDN workspace has its own native admission/owner tests.
        definitions.append('bool use_text_gdn_input_workspace() const { return false; }')
        definitions.append('std::vector<int64_t> gdn_input_workspace_arguments(int64_t) const '
                           '{ TORCH_CHECK(false, "GDN workspace is not bound in this fixture"); return {}; }')
        header = (ROOT / 'src/fused/rpu_qwen3_5_model.h').read_text()
        definitions.append(re.search(
            r'static constexpr int64_t kTextGdnInputWorkspaceSite\s*=.*?;', header)[0])
    for high_site in ('QWEN35_ACTION_SILU_HIGH_MUL_SITE',
                      'QWEN35_ACTION_NEWTON_NORM_SHIFT_SITE'):
        if (high_site in cpp and not re.search(
                r'constexpr\s+int64_t\s+' + high_site + r'\s*=', cpp)):
            constants.append(high_site)
    # Historical doubles exercise BASE. Keep that default explicit while
    # dedicated HIGH tests supply the actual policy, helpers and descriptors.
    if re.search(r'\bhigh_precision_\b', cpp) and not re.search(r'\bbool\s+high_precision_\b', cpp):
        definitions.append('bool high_precision_ = false;')
    if ('high_precision_arguments(' in cpp and not re.search(
            r'std::vector\s*<\s*int64_t\s*>\s+high_precision_arguments\(', cpp)):
        definitions.append('std::vector<int64_t> high_precision_arguments() const '
                           '{ TORCH_CHECK(!high_precision_); return {}; }')
    if ('consume_high_precision_math(' in cpp and not re.search(
            r'void\s+consume_high_precision_math\(', cpp)):
        definitions.append('void consume_high_precision_math() '
                           '{ TORCH_CHECK(false, "BASE double cannot execute HIGH"); }')
    if ('action_silu_high_mul_available_' in cpp and not re.search(
            r'\bbool\s+action_silu_high_mul_available_\b', cpp)):
        definitions.append('bool action_silu_high_mul_available_ = false;')
    if ('use_action_silu_high_mul(' in cpp and not re.search(
            r'bool\s+use_action_silu_high_mul\(', cpp)):
        definitions.append('bool use_action_silu_high_mul(int64_t) const '
                           '{ TORCH_CHECK(!high_precision_); return false; }')
    if ('action_silu_high_mul_arguments(' in cpp and not re.search(
            r'std::vector\s*<\s*int64_t\s*>\s+action_silu_high_mul_arguments\(', cpp)):
        definitions.append('std::vector<int64_t> action_silu_high_mul_arguments() const '
                           '{ TORCH_CHECK(!high_precision_); return {}; }')
    if ('prepare_action_silu_high_mul(' in cpp and not re.search(
            r'void\s+prepare_action_silu_high_mul\(', cpp)):
        definitions.append('void prepare_action_silu_high_mul() '
                           '{ TORCH_CHECK(!high_precision_); }')
    if ('launch_action_silu_high_mul(' in cpp and not re.search(
            r'void\s+launch_action_silu_high_mul\(', cpp)):
        definitions.append('void launch_action_silu_high_mul(int,const ChunkInfo&) '
                           '{ TORCH_CHECK(false, "BASE double cannot execute HIGH fusion"); }')
    if (('action_newton_norm_shift_available_' in cpp or
         'use_action_newton_norm_shift(' in cpp) and not re.search(
            r'\bbool\s+action_newton_norm_shift_available_\b', cpp)):
        definitions.append('bool action_newton_norm_shift_available_ = false;')
    if ('use_action_newton_norm_shift(' in cpp and not re.search(
            r'bool\s+use_action_newton_norm_shift\(', cpp)):
        definitions.append('bool use_action_newton_norm_shift(int64_t) const '
                           '{ TORCH_CHECK(!high_precision_ || '
                           '!action_newton_norm_shift_available_); return false; }')
    if ('action_newton_norm_shift_arguments(' in cpp and not re.search(
            r'std::vector\s*<\s*int64_t\s*>\s+action_newton_norm_shift_arguments\(', cpp)):
        definitions.append('std::vector<int64_t> action_newton_norm_shift_arguments() const '
                           '{ TORCH_CHECK(!high_precision_); return {}; }')
    if ('prepare_action_newton_norm_shift(' in cpp and not re.search(
            r'void\s+prepare_action_newton_norm_shift\(', cpp)):
        definitions.append('void prepare_action_newton_norm_shift() '
                           '{ TORCH_CHECK(!high_precision_); }')
    if ('apply_action_newton_norm_shift(' in cpp and not re.search(
            r'void\s+apply_action_newton_norm_shift\(', cpp)):
        definitions.append('void apply_action_newton_norm_shift(uint32_t,uint32_t,const ChunkInfo&) '
                           '{ TORCH_CHECK(false, "BASE double cannot execute HIGH normalization"); }')
    declarations = []
    if 'unary_precision(' in cpp and not re.search(r'RpuUnaryPrecision\s+unary_precision\(', cpp):
        definitions.append('RpuUnaryPrecision unary_precision() const ' + _block(
            source, 'RpuUnaryPrecision Qwen3_5Model::unary_precision('))
        if not re.search(r'enum\s+class\s+RpuUnaryPrecision\b', cpp):
            declarations.append('enum class RpuUnaryPrecision : uint8_t ' + _block((ROOT / 'src/core/rpu_math_precision.h').read_text(),
                                       'enum class RpuUnaryPrecision') + ';')
        # Existing launcher doubles now accept the real optional precision
        # argument and reject a HIGH launch in a BASE-only fixture.
        cpp = re.sub(
            r'(void\s+rpu_launch_eltwise_unary_spm_kernel\([^{};]*?)(\)\s*\{)',
            lambda match: (match[1] + ', RpuUnaryPrecision precision = RpuUnaryPrecision::BASE'
                           + match[2] + '\n assert(precision == RpuUnaryPrecision::BASE);')
            if match[1].count(',') == 5 else match[0], cpp)
    if 'rmsnorm_route(' in cpp and not re.search(r'RpuRmsNormSpmRoute\s+rmsnorm_route\(', cpp):
        decls = (ROOT / 'src/core/rpu_kernel_decls.h').read_text()
        norm_enum = 'enum class RpuRmsNormSpmRoute : int64_t ' + _block(decls, 'enum class RpuRmsNormSpmRoute') + ';'
        if re.search(r'enum\s+class\s+RpuRmsNormSpmRoute\b', cpp):
            cpp = re.sub(r'enum\s+class\s+RpuRmsNormSpmRoute[^{}]*\{[^{}]*\};',
                         '', cpp)
        declarations.append(norm_enum)
        declarations.append('inline constexpr int64_t RPU_RMSNORM_SCM_U16_MAX = 65535;\n'
                            'inline constexpr int64_t RPU_RMSNORM_MAX_COLS = '
                            'RPU_RMSNORM_SCM_U16_MAX / 2;\n'
                            'RpuRmsNormSpmRoute rpu_resolve_high_precision_rmsnorm_spm_route('
                            'int64_t rows, int64_t cols) ' + _block(
                                decls, 'inline RpuRmsNormSpmRoute rpu_resolve_high_precision_rmsnorm_spm_route('))
        definitions.append('RpuRmsNormSpmRoute rmsnorm_route(int64_t rows, int64_t cols) const '
                           + _block(source, 'RpuRmsNormSpmRoute Qwen3_5Model::rmsnorm_route('))
        cpp = re.sub(
            r'(void\s+rpu_launch_rmsnorm_spm_kernel\([^{};]*?)(\)\s*\{)',
            lambda match: (match[1] + ', RpuRmsNormSpmRoute route = RpuRmsNormSpmRoute::BASE, int cores = 8'
                           + match[2] + '\n assert(route == RpuRmsNormSpmRoute::BASE && cores == 8);')
            if match[1].count(',') == 5 else match[0], cpp)
    for helper in helpers:
        if helper + '(' not in cpp or re.search(r'int64_t\s+' + helper + r'\(\)\s*const\s*{', cpp):
            continue
        body = _block(source, 'int64_t Qwen3_5Model::' + helper + '(')
        definitions.append('int64_t ' + helper + '() const ' + body)
        for constant in re.findall(r'QWEN35_[A-Z0-9_]+', body):
            if constant in constants or re.search(r'constexpr\s+int64_t\s+' + constant + r'\s*=', cpp):
                continue
            constants.append(constant)
    if not definitions and not constants and not declarations:
        return cpp
    declarations.extend(re.search(r'constexpr int64_t ' + constant + r'\s*=.*?;', source, re.S)[0]
                        for constant in constants)
    match = re.search(r'struct Model\s*{', cpp)
    assert match, 'site helpers require the actual Model double'
    cpp = cpp[:match.end()] + '\n'.join(definitions) + '\n' + cpp[match.end():]
    # Precision types must precede launcher prototypes as well as Model.
    global_point = re.search(r'void\s+rpu_launch_(?:eltwise_unary|rmsnorm)_spm_kernel\(', cpp)
    at = global_point.start() if global_point else match.start()
    return cpp[:at] + '\n'.join(declarations) + '\n' + cpp[at:]


def _compile_run(tmp_path, name, cpp):
    cpp = with_actual_linear_contracts(_with_actual_g05_site_helpers(cpp))
    path, exe = tmp_path / (name + '.cpp'), tmp_path / name
    path.write_text(cpp)
    compiled = subprocess.run(['c++', '-std=c++17', '-O0', '-Wall', '-Wextra', '-Werror',
                               '-I', str(ROOT / 'src/ops'), '-I', str(ROOT / 'src/core'), str(path), '-o', str(exe)],
                              capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_q_gate_cold_owners_manifest_and_phase2_fallback(tmp_path):
    source = (ROOT / 'src/fused/rpu_qwen3_5_model.cpp').read_text()
    methods = [
        ('void prepare_action_q_gate()', 'void Qwen3_5Model::prepare_action_q_gate('),
        ('bool use_action_q_gate(int64_t rows) const', 'bool Qwen3_5Model::use_action_q_gate('),
        ('bool use_action_mod_body() const', 'bool Qwen3_5Model::use_action_mod_body('),
        ('std::vector<int64_t> action_q_gate_arguments() const', 'std::vector<int64_t> Qwen3_5Model::action_q_gate_arguments('),
        ('void launch_action_q_gate(int layer,const ChunkInfo& chunk)', 'void Qwen3_5Model::launch_action_q_gate('),
        ('bool use_action_kv_owner(int64_t rows) const', 'bool Qwen3_5Model::use_action_kv_owner('),
    ]
    manifest = _block(source, 'if (use_action_q_gate(chunk.len))')
    begin = source.index('    const bool owner_cache =')
    end = source.index('    // Phase 3: QK RMSNorm', begin)
    caller = source[begin:end]
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(x,...) do {if(!(x))throw std::runtime_error("rejected");}while(0)
int tensor_proofs=0;
namespace c10{enum class DeviceType{PrivateUse1,CPU};}
namespace at{constexpr int kHalf=1,kChar=3;
struct Tensor{
 bool present=true,contiguous=true;int dtype=kHalf,rank=2;int64_t rows=2048,cols=1024;uint64_t owner=0;std::string tag;
 c10::DeviceType location=c10::DeviceType::PrivateUse1;
 bool defined()const{++tensor_proofs;return present;}int scalar_type()const{return dtype;}bool is_contiguous()const{return contiguous;}
 struct Device{c10::DeviceType value;c10::DeviceType type()const{return value;}};Device device()const{return {location};}
 int dim()const{return rank;}int64_t size(int i)const{return i?cols:rows;}
};}
enum class KernelId{G05_Q_GATE_FP16_ACC32_M32N256K1024,G05_Q_GATE_W8A16_ACC32_M32N256K1024,G05_Q_GATE_W8A16_ACC16_M32N256K1024};
struct KernelCache{bool available=true;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId)const{return available;}};
struct Weights{at::Tensor q_w,attn_gate_w,k_w,v_w,q_ws,k_ws,v_ws,attn_gate_ws;};
enum class FmbRouteFamily{LINEAR};constexpr int64_t QWEN35_ACTION_Q_GATE_SITE=2511688968361667267LL,QWEN35_ACTION_Q_GATE_W8_SITE=5855704986459519851LL;
struct ChunkInfo{int64_t len=32,idx=2;};
struct Route{int64_t site;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t>args;int64_t invocation;};
std::vector<std::string> launches;std::vector<std::pair<uint64_t,uint64_t>>weights_seen;
void rpu_launch_g05_q_gate_linear_fp16_acc32_kernel(uint32_t in,const at::Tensor&q,const at::Tensor&gate,uint32_t qo,uint32_t go){assert(in==0x10000&&qo==0x20000&&go==0x24000&&q.owner!=gate.owner);launches.push_back("q+gate");weights_seen.push_back({q.owner,gate.owner});}
std::vector<bool>linear_modes;
void rpu_launch_g05_q_gate_linear_w8a16_acc32_kernel(uint32_t,const at::Tensor&a,const at::Tensor&b,const at::Tensor&as,const at::Tensor&bs,uint32_t,uint32_t,bool linear_acc32=true){linear_modes.push_back(linear_acc32);assert(a.dtype==at::kChar&&b.dtype==at::kChar&&as.rank==1&&bs.rank==1&&as.owner!=bs.owner);launches.push_back("q+gate-w8");weights_seen.push_back({a.owner,b.owner});}
struct Model{
 bool use_action_nvfp4(int64_t)const{return false;}
 bool use_action_gate_kv_owner(int64_t,int64_t)const{return false;}void launch_action_gate_kv_owner(int,const ChunkInfo&){assert(false);}

 bool action_mod_body_fp16_=false,action_mod_body_w8_=false;
 bool action_gateup_resident_w8_=false,action_kv_owner_w8_=false,action_q_gate_w8_=false,action_gateup_pair_w8_=false,text_gateup_pair_w8_=false;
 bool action_mode_=true,linear_acc32_=true,has_gdn_=false,action_q_gate_available_=false,action_kv_owner_available_=false;
 bool action_step_active_=false,action_loop_active_=false,complete=true;
 int64_t action_len_=32,action_num_steps_=1,layers=24,h=1024,intermediate=4096,cores=8,heads=8,kv=8,hd=256,tp=8;
 std::vector<Weights>layer_weights_{24};std::vector<uint8_t>layer_is_full_=std::vector<uint8_t>(24,1);std::vector<Route>routes;
 Model(){for(int i=0;i<24;++i){auto&w=layer_weights_[i];w.q_w.owner=2*i+1;w.attn_gate_w.owner=2*i+2;w.q_w.tag="q";w.attn_gate_w.tag="gate";w.k_w.tag="k";w.v_w.tag="v";}}
 int64_t num_layers()const{return layers;}int64_t hidden_size()const{return h;}int64_t intermediate_size()const{return intermediate;}
 int64_t num_cores()const{return cores;}int64_t attn_tp()const{return tp;}int64_t mlp_tp()const{return tp;}int64_t num_q_heads()const{return heads;}int64_t num_kv_heads()const{return kv;}int64_t head_dim()const{return hd;}
 Model&ctx(){return *this;}bool has_complete_physical_manifest()const{return complete;}
 uint32_t addr(int,const char*n)const{std::string name(n);return name=="input_norm"?0x10000:name=="q"?0x20000:name=="attn_gate"?0x24000:0x30000;}
 void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t>args,int64_t inv){auto i=std::find_if(routes.begin(),routes.end(),[&](auto&r){return r.site==site&&r.invocation==inv;});TORCH_CHECK(i!=routes.end()&&i->family==family&&i->selector==selector&&i->flags==flags&&i->args==args);}
 void launch_linear(int64_t,uint32_t,const at::Tensor&w,uint32_t,int64_t,int64_t,int64_t,int,int64_t,int,const at::Tensor&,int=-1,int=-1){launches.push_back(w.tag);}
 int64_t position=245;bool use_action_owner_cache(int64_t,int64_t)const{return false;}void launch_action_qkv_gate_owner_only(int,const ChunkInfo&){assert(false);}
 bool use_action_qkv_gate(int64_t)const{return false;}void launch_action_qkv_gate(int,const ChunkInfo&){assert(false);}
 void launch_action_kv_owner(int,const ChunkInfo&){launches.push_back("kv");}
 __METHODS__
 void plan(ChunkInfo chunk={}){struct{std::vector<Route>routes;}manifest;if(use_action_q_gate(chunk.len))__MANIFEST__ routes=manifest.routes;}
 void phase2(int layer_idx,ChunkInfo chunk={}){const auto&lw=layer_weights_[layer_idx];int64_t seq_len=chunk.len,nq=num_q_heads(),nkv=num_kv_heads(),hd=head_dim(),h=hidden_size(),tp=attn_tp();__CALLER__}
};
template<class F>void reject(F f){auto before=launches;try{f();assert(false);}catch(const std::runtime_error&){}assert(launches==before);}
int main(){
 Model m;m.prepare_action_q_gate();const int cold_proofs=tensor_proofs;assert(m.action_q_gate_available_&&cold_proofs>=48);
 m.plan();assert(m.routes.empty()&&!m.use_action_q_gate(32));m.phase2(0);assert(launches==std::vector<std::string>({"q","k","v","gate"}));
 m.action_step_active_=m.action_loop_active_=true;m.action_num_steps_=10;m.plan();
 assert(m.routes.size()==1&&m.routes[0].args==std::vector<int64_t>({2,32,256,1024,8,32,80,80,512,4,8,2048,512,2,32,20,34,2,4,36,1}));
 for(int layer=0;layer<24;++layer){launches.clear();m.phase2(layer);assert(launches==std::vector<std::string>({"q+gate","k","v"}));assert(weights_seen.back()==std::make_pair(uint64_t(2*layer+1),uint64_t(2*layer+2)));}
 assert(tensor_proofs==cold_proofs);m.action_kv_owner_available_=true;launches.clear();m.phase2(0);assert(launches==std::vector<std::string>({"q+gate","kv"}));
 m.complete=false;reject([&]{m.phase2(0);});m.complete=true;m.routes[0].args[12]=1024;reject([&]{m.phase2(0);});m.plan();
 reject([&]{m.launch_action_q_gate(24,{});});reject([&]{m.launch_action_q_gate(-1,{});});reject([&]{m.launch_action_q_gate(0,{31,2});});reject([&]{m.launch_action_q_gate(0,{32,3});});
 KernelCache::instance().available=false;m.prepare_action_q_gate();assert(!m.action_q_gate_available_);m.plan();assert(m.routes.empty());launches.clear();m.phase2(0);assert(launches==std::vector<std::string>({"q","kv","gate"}));
 KernelCache::instance().available=true;m.prepare_action_q_gate();assert(m.action_q_gate_available_);
 for(int which:{0,1})for(int fault=0;fault<7;++fault){auto x=m;auto&w=which?x.layer_weights_.back().attn_gate_w:x.layer_weights_.back().q_w;
  if(fault==0){w.present=false;}if(fault==1){w.dtype=2;}if(fault==2){w.location=c10::DeviceType::CPU;}if(fault==3){w.contiguous=false;}if(fault==4){w.rank=3;}if(fault==5){w.rows=1024;}if(fault==6){w.cols=2048;}
  x.prepare_action_q_gate();assert(!x.action_q_gate_available_);x.plan();assert(x.routes.empty());
 }
 for(int fault=0;fault<8;++fault){auto x=m;if(fault==0)x.linear_acc32_=false;if(fault==1)x.has_gdn_=true;if(fault==2)x.layer_is_full_.back()=0;if(fault==3)x.heads=16;if(fault==4)x.kv=2;if(fault==5)x.intermediate=2048;if(fault==6)x.tp=4;if(fault==7)x.layer_weights_.pop_back();x.prepare_action_q_gate();assert(!x.action_q_gate_available_);}
 for(int fault=0;fault<4;++fault){auto x=m;if(fault==0)x.action_num_steps_=1;if(fault==1)x.action_loop_active_=false;if(fault==2)x.action_step_active_=false;if(fault==3)x.action_len_=16;assert(!x.use_action_q_gate(32));x.plan();assert(x.routes.empty());}
 // W8 uses the same cold/COMPLETE flow with separately bound scale owners.
 for(auto&w:m.layer_weights_){w.q_w.dtype=w.attn_gate_w.dtype=at::kChar;w.q_ws.rank=w.attn_gate_ws.rank=1;w.q_ws.owner=w.q_w.owner+100;w.attn_gate_ws.owner=w.attn_gate_w.owner+200;}
 m.prepare_action_q_gate();assert(m.action_q_gate_available_&&m.action_q_gate_w8_);m.plan();
 assert(m.routes[0].site==QWEN35_ACTION_Q_GATE_W8_SITE&&m.routes[0].args==std::vector<int64_t>({2,32,256,1024,8,32,96,48,512,6,12,2048,512,2,32,20,34,1,4,36,1}));
 const int w8_cold_proofs=tensor_proofs;
 for(int layer=0;layer<24;++layer){launches.clear();m.phase2(layer);assert(launches==std::vector<std::string>({"q+gate-w8","kv"}));}
 assert(tensor_proofs==w8_cold_proofs);m.routes[0].args[16]=20;reject([&]{m.phase2(0);});m.plan();
 {auto x=m;x.linear_acc32_=false;x.action_mod_body_w8_=true;x.prepare_action_q_gate();x.plan();assert(x.use_action_q_gate(32)&&x.routes[0].args[5]==16&&x.routes[0].args[6]==48);auto valid=x.routes;x.routes=m.routes;reject([&]{x.phase2(0);});x.routes=valid;linear_modes.clear();launches.clear();x.phase2(23);assert(linear_modes==std::vector<bool>({false})&&launches==std::vector<std::string>({"q+gate-w8","kv"}));}
 m.layer_weights_.back().attn_gate_ws.present=false;m.prepare_action_q_gate();assert(!m.action_q_gate_available_&&!m.action_q_gate_w8_);m.layer_weights_.back().attn_gate_ws.present=true;
 KernelCache::instance().available=false;m.prepare_action_q_gate();m.plan();assert(!m.action_q_gate_available_&&m.routes.empty());launches.clear();m.phase2(0);assert(launches==std::vector<std::string>({"q","kv","gate"}));

}
'''
    for marker, value in {'__METHODS__': '\n'.join(decl + _body(source, signature) for decl, signature in methods),
                          '__MANIFEST__': manifest, '__CALLER__': caller}.items():
        cpp = cpp.replace(marker, value)
    _compile_run(tmp_path, 'cold', cpp)


def test_q_gate_actual_launcher_independent_ddr_owners_registers_and_overlap_rejection(tmp_path):
    source = (ROOT / 'src/ops/rpu_linear.cpp').read_text()
    cpp = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>
#include "rpu_linear_tiling.h"
#define TORCH_CHECK(x,...) do {if(!(x))throw std::runtime_error("rejected");}while(0)
namespace c10{using Half=uint16_t;enum class DeviceType{PrivateUse1,CPU};}
namespace at{constexpr int kHalf=1,kChar=3;
struct Tensor{uint64_t address;bool present=true,contiguous=true;int dtype=kHalf,rank=2;int64_t rows=2048,cols=1024;c10::DeviceType location=c10::DeviceType::PrivateUse1;
 bool defined()const{return present;}int scalar_type()const{return dtype;}bool is_contiguous()const{return contiguous;}int dim()const{return rank;}int64_t size(int i)const{return i?cols:rows;}
 struct Device{c10::DeviceType value;c10::DeviceType type()const{return value;}};Device device()const{return {location};}template<class T>T*data_ptr()const{return reinterpret_cast<T*>(address);}
};}
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8303616;bool initialized=true;bool is_initialized()const{return initialized;}uint32_t addr(int core,int off)const{assert(core==0);return 0x20000000+off;}}allocator;
#define SPM_ALLOC allocator
enum class KernelId{G05_Q_GATE_FP16_ACC32_M32N256K1024};constexpr auto ID=KernelId::G05_Q_GATE_FP16_ACC32_M32N256K1024;
struct KernelCache{bool available=true;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId)const{return available;}};
enum class GraphDdrRegisterRole{ModelWeight};enum class GraphDdrRegisterAccess{Read};enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{GraphDdrRegisterAbi abi;at::Tensor owner;};
struct Kernel{std::map<int,uint16_t>regs;void reset_regs(){regs.clear();}void set_regs(int i,uint16_t v){regs[i]=v;}void set_regs(int,uint32_t)=delete;};
struct Submission{std::map<int,uint16_t>regs;std::vector<GraphDdrRegisterOperandSpec>operands;};struct Writer{void write(Kernel&);};
struct RpuKernelGraph{bool staged=false,pending=false,written=false;int stages=0,lookups=0;Kernel kernel;std::vector<GraphDdrRegisterOperandSpec>operands;std::vector<Submission>submissions;
 static RpuKernelGraph&active(){static RpuKernelGraph graph;return graph;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec>specs){assert(id==ID&&!staged&&!pending&&specs.size()==2);staged=true;written=false;operands=std::move(specs);++stages;return {};}
 Kernel*get(KernelId id){assert(id==ID&&staged&&!pending&&KernelCache::instance().available);pending=true;++lookups;return &kernel;}
};
void Writer::write(Kernel&kernel){auto&g=RpuKernelGraph::active();assert(g.staged&&g.pending&&!g.written);for(auto&operand:g.operands){uint64_t v=operand.owner.address>>8;kernel.set_regs(operand.abi.lo,static_cast<uint16_t>(v));kernel.set_regs(operand.abi.hi,static_cast<uint16_t>(v>>16));}g.written=true;}
#define GET_KERNEL(id) RpuKernelGraph::active().get(id)
std::vector<uint64_t>flushed;void rpu_ddr_flush(c10::Half*p){flushed.push_back(reinterpret_cast<uint64_t>(p));}
struct Queue{bool broadcast=false;void set_broadcast_mode(bool v){broadcast=v;}void enqueu_kernel(Kernel&k,std::vector<uint16_t>grid,std::vector<uint8_t>cores){auto&g=RpuKernelGraph::active();assert(broadcast&&g.staged&&g.pending&&g.written);assert(grid==std::vector<uint16_t>({8,1,1})&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));g.submissions.push_back({k.regs,g.operands});g.operands.clear();g.staged=g.pending=g.written=false;}}queue;
Queue*get_queue(int cores){assert(cores==8);return &queue;}
#define GET_QUEUE(cores) get_queue(cores)
void launch(uint32_t input,const at::Tensor&query_weight,const at::Tensor&gate_weight,uint32_t query_output,uint32_t gate_output)__BODY__
uint32_t pair(const Submission&s,int reg){return uint32_t(s.regs.at(reg))|(uint32_t(s.regs.at(reg+1))<<16);}
template<class F>void reject(F f){auto&g=RpuKernelGraph::active();auto before=g.submissions.size(),flushes=flushed.size();int stages=g.stages,lookups=g.lookups;try{f();assert(false);}catch(const std::runtime_error&){}assert(g.submissions.size()==before&&flushed.size()==flushes&&g.stages==stages&&g.lookups==lookups&&!g.staged&&!g.pending);}
int main(){
 auto&g=RpuKernelGraph::active();const uint32_t base=allocator.addr(0,0);
 for(uint32_t pass=0;pass<2;++pass){const uint32_t in=base+0x20000+pass*0x40000,q=in+65536,gate=q+16384;at::Tensor qw{0x1234567800ULL+pass*0x400000},gw{0x2345678900ULL+pass*0x800000};launch(in,qw,gw,q,gate);const auto&s=g.submissions.back();
  assert(pair(s,0)==in&&pair(s,4)==q&&pair(s,36)==gate);assert(pair(s,2)==qw.address>>8&&pair(s,32)==gw.address>>8);
  assert(pair(s,6)==0&&s.regs.at(8)==1&&s.regs.at(9)==32&&pair(s,10)==256&&s.regs.at(12)==1024&&s.regs.at(13)==64&&s.regs.at(14)==0&&s.regs.at(15)==8);
  assert(pair(s,16)==2048&&pair(s,18)==512&&pair(s,20)==0&&pair(s,22)==0&&pair(s,24)==0&&s.regs.at(26)==2&&s.regs.at(27)==2&&s.regs.at(28)==0&&s.regs.at(29)==0);
  assert(s.operands.size()==2&&s.operands[0].owner.address==qw.address&&s.operands[1].owner.address==gw.address);
  for(int i=0;i<2;++i){auto&abi=s.operands[i].abi;assert(abi.role==GraphDdrRegisterRole::ModelWeight&&abi.access==GraphDdrRegisterAccess::Read&&abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi&&abi.lo==(i?32:2)&&abi.hi==(i?33:3));}
  assert(flushed[pass*2]==qw.address&&flushed[pass*2+1]==gw.address);
 }
 const at::Tensor qw{0x1234567800ULL},gw{0x2345678900ULL};const uint32_t in=base+0x20000,q=in+65536,gate=q+16384;
 for(auto regions:{std::array<uint32_t,3>{in,in,gate},{in,q,q},{in,q,in+256},{in,q-256,gate},{in,q,gate-256},{in+1,q,gate},{base-256,q,gate},{in,q,base+SpmAllocator::SPM_USABLE-8192}})reject([&]{launch(regions[0],qw,gw,regions[1],regions[2]);});
 allocator.initialized=false;reject([&]{launch(in,qw,gw,q,gate);});allocator.initialized=true;KernelCache::instance().available=false;reject([&]{launch(in,qw,gw,q,gate);});KernelCache::instance().available=true;
 for(int which:{0,1})for(int fault=0;fault<7;++fault){auto first=qw,second=gw;auto&w=which?second:first;if(fault==0){w.present=false;}if(fault==1){w.dtype=2;}if(fault==2){w.contiguous=false;}if(fault==3){w.location=c10::DeviceType::CPU;}if(fault==4){w.rank=1;}if(fault==5){w.rows=1024;}if(fault==6){w.cols=512;}reject([&]{launch(in,first,second,q,gate);});}
}
'''
    cpp = cpp.replace('__BODY__', _body(source, 'void rpu_launch_g05_q_gate_linear_fp16_acc32_kernel('))
    _compile_run(tmp_path, 'launcher', cpp)


