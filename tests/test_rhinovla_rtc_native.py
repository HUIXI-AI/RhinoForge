"""Run the production RTC admission and row/DMA routing bodies without a board."""
from pathlib import Path
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition


ROOT = Path(__file__).resolve().parents[1]


def test_native_rtc_admission_row_groups_and_mutable_prefix(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/fused/rpu_rhino_vla_model.cpp").read_text()
    extraction_source = source.replace("at::TensorList cold_weights = {}", "at::TensorList cold_weights = at::TensorList()")
    extraction_source = extraction_source.replace("at::TensorList cold_scales = {}", "at::TensorList cold_scales = at::TensorList()")
    extraction_source = extraction_source.replace("at::TensorList cold_biases = {}", "at::TensorList cold_biases = at::TensorList()")
    methods = "\n".join(_definition(extraction_source, prefix) for prefix in (
        "void set_denoise_loop_rtc_tables(",
        "void prepare_rtc(",
        "std::vector<int64_t> planner_cache_identity() const",
        "template <typename Emit>\n    void for_each_rtc_group(",
        "void emit_conditioned_norm_shift(",
        "void emit_rtc_prefix_clamp(",
    ))
    helper = _definition(source, "void check_rhino_full_w8_projection(")
    program = r'''
#include <cassert>
#include <cmath>
#include <numeric>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
constexpr int DWIDTH=2;
namespace c10 { using Half = float; }
namespace at {
enum { kHalf=1, kPrivateUse1=2, kChar=3 };
using IntArrayRef = std::vector<int64_t>;
struct Device { int kind=kPrivateUse1; int type() const { return kind; }
    bool operator==(Device d) const { return kind==d.kind; } };
struct Tensor {
    IntArrayRef shape; int dtype=kHalf; Device dev; bool valid=true, contiguous=true;
    std::vector<float> data;
    Tensor(IntArrayRef sizes={}, int type=kHalf): shape(sizes),dtype(type),data(6144,1.0f) {}
    bool defined() const { return valid; }
    Device device() const { return dev; }
    int scalar_type() const { return dtype; }
    bool is_contiguous() const { return contiguous; }
    const IntArrayRef& sizes() const { return shape; }
    int64_t dim() const { return shape.size(); }
    int64_t size(int i) const { return shape.at(i); }
    int64_t numel() const { return std::accumulate(shape.begin(),shape.end(),int64_t(1),std::multiplies<int64_t>()); }
    int64_t nbytes() const { return numel()*2; }
    template<class T=void> T* data_ptr() const { return reinterpret_cast<T*>(const_cast<float*>(data.data())); }
};
using TensorList=std::vector<Tensor>;
}
void rpu_ddr_flush_force_sized(void*,int64_t) {}
HELPER
bool active_graph=false, blocked_thread=false;
struct RpuExecutionCoordinator {
    static void require_graph_quiescent(const char*) { TORCH_CHECK(!active_graph); }
    static void check_current_thread_execution_allowed(const char*) { TORCH_CHECK(!blocked_thread); }
};
namespace rhino_lkn { uint64_t RpuGetDevAddr(void* p) { return reinterpret_cast<uint64_t>(p); } }
struct FusedModelBase { std::vector<int64_t> planner_cache_identity() const { return {1, 42}; } };
enum class FmbRouteFamily { MUTABLE_DMA };
enum class RhinoMutableDmaRoute { DDR_BROADCAST_TO_SPM };
constexpr int64_t RHINO_RTC_PREFIX_DMA_SITE=91;
std::vector<int64_t> consumed;
struct Context {
    void consume_physical_route(FmbRouteFamily, int64_t site, int64_t, int,
                                std::vector<int64_t> args) {
        assert(site==RHINO_RTC_PREFIX_DMA_SITE); consumed=args;
    }
};
uint64_t* dma_base=nullptr;
int64_t dma_count=-1;
void rpu_launch_ddr_broadcast_spm_dma_mutable(uint64_t* base, int64_t offset,
        int64_t count, uint32_t target, int cores) {
    assert(offset==0 && target==4096 && cores==1);
    dma_base=base; dma_count=count;
}
struct Model : FusedModelBase {
    struct Config { bool precompute_adarms=true, skip_adarms_gemv=false,
        adarms_resident=false, expert_fusions=false, packed_qkv=false,
        precompute_time_proj=false, fold_action_time_in=false; } cold_config_;
    bool rtc_mode_=false, denoise_loop_weights_ready_=true, loop_mode_=true,
        dt_pinned_=true, direct_action_input_=true, expert_w8a16_=false,
        expert_cond_w8a16_=false, full_action_w8a16_=false, high_precision_=false,
        denoise_adarms_tables_ready_=false, fast_replay_skip_layer_loop_=true;
    int64_t rtc_delay_=0, num_steps_=10, action_horizon_=30, suffix_len_=31,
        action_dim_=96, action_dim_pad_=96, state_dim_=96, state_dim_pad_=96,
        rope_position_=75, last_chunk=0;
    float dt_=-0.1f;
    at::Tensor adarms_pair_table_, adarms_final_table_;
    at::TensorList full_cold_weights_, full_cold_scales_, full_cold_biases_;
    bool gates_tanh_precomputed_=false;
    uint64_t adarms_pair_table_src_base_=0, adarms_final_table_src_base_=0,
        action_prefix_src_base_=0;
    int generation=0;
    Context context;
    std::vector<std::vector<int64_t>> norms;
    int64_t num_layers() const { return 18; }
    int64_t hidden_size() const { return 1024; }
    int64_t intermediate_size() const { return 3072; }
    int64_t num_q_heads() const { return 16; }
    int64_t num_kv_heads() const { return 8; }
    int64_t head_dim() const { return 128; }
    int64_t attn_tp() const { return 8; }
    int64_t get_last_resolved_chunk_size() const { return last_chunk; }
    void invalidate_model_state() { ++generation; }
    Context& ctx() { return context; }
    uint32_t addr(int core, const char*) { assert(core==0); return 4096; }
    void emit_norm_shift(uint32_t input, uint32_t output, uint32_t scale,
                         uint32_t shift, int64_t rows) {
        norms.push_back({input, output, scale, shift, rows});
    }
METHODS
};
template<class F> void reject(F fn) {
    bool rejected=false; try { fn(); } catch (const std::runtime_error&) { rejected=true; }
    assert(rejected);
}
int main() {
    at::Tensor pair({7,10,18,3,6144}), final({7,10,3,3072});
    for (int failure=0; failure<18; ++failure) {
        Model m; auto p=pair; auto f=final;
        switch(failure) {
            case 0: m.last_chunk=31; break;
            case 1: m.rtc_mode_=true; break;
            case 2: m.action_horizon_=50; break;
            case 3: m.num_steps_=9; break;
            case 4: m.dt_=0.1f; break;
            case 5: m.cold_config_.skip_adarms_gemv=true; break;
            case 6: m.high_precision_=true; break;
            case 7: m.expert_cond_w8a16_=true; break;
            case 8: m.cold_config_.precompute_adarms=false; break;
            case 9: p.shape={7,10,3,18,6144}; break;
            case 10: f.shape={10,3,3072}; break;
            case 11: p.contiguous=false; break;
            case 12: active_graph=true; break;
            case 13: blocked_thread=true; break;
            case 14: m.direct_action_input_=false; break;
            case 15: m.cold_config_.precompute_time_proj=true; break;
            case 16: m.cold_config_.fold_action_time_in=true; break;
            case 17: m.dt_=-0.2f; break;
        }
        reject([&]{m.set_denoise_loop_rtc_tables(p,f);});
        assert(m.generation==0 && !m.denoise_adarms_tables_ready_);
        active_graph=blocked_thread=false;
    }
    // All existing precision scopes accept RTC without changing their owner
    // precision, including optional resident/packed/fusion settings.
    for(int profile=0;profile<5;++profile) {
        Model p;
        p.expert_w8a16_=profile>=1;
        p.expert_cond_w8a16_=profile>=2;
        p.full_action_w8a16_=profile>=3;
        p.high_precision_=profile==4;
        p.cold_config_.adarms_resident=true;
        p.cold_config_.expert_fusions=true;
        p.cold_config_.packed_qkv=profile==0;
        at::TensorList weights,scales,biases;
        const int count=profile>=3?19:profile==2?18:0;
        for(int i=0;i<count;++i) {
            const int n=i==18?3072:6144;
            weights.emplace_back(at::IntArrayRef{n,1024},at::kChar);
            scales.emplace_back(at::IntArrayRef{n}); biases.emplace_back(at::IntArrayRef{n});
        }
        auto bind=[&] { p.set_denoise_loop_rtc_tables(pair,final,weights,scales,biases,profile==4,profile==4); };
        if(count) {
            auto original=weights;
            weights.pop_back(); reject(bind); weights=original;
            weights[0].dtype=at::kHalf; reject(bind); weights=original;
            scales[0].data[0]=0; reject(bind); scales[0].data[0]=1;
            biases[0].shape={1}; reject(bind); biases[0].shape={6144};
            assert(p.generation==0 && p.full_cold_weights_.empty() && !p.rtc_mode_);
        }
        bind(); assert(p.full_cold_weights_.size()==count && p.generation==1);
        assert(p.gates_tanh_precomputed_==(profile==4) && p.high_precision_==(profile==4));
        for(int delay=0;delay<=6;++delay) { p.prepare_rtc(delay); assert(p.rtc_delay_==delay); }
    }
    Model m;
    reject([&]{m.prepare_rtc(0);});
    m.set_denoise_loop_rtc_tables(pair,final);
    assert(m.generation==1 && m.rtc_mode_ && m.denoise_adarms_tables_ready_);
    reject([&]{m.set_denoise_loop_rtc_tables(pair,final);});
    auto identity0=m.planner_cache_identity();
    for(int delay=0;delay<=6;++delay) {
        m.prepare_rtc(delay);
        assert(m.generation==1); // A semantic selection must preserve captured owners.
        auto identity=m.planner_cache_identity(); assert(identity.back()==delay);
        m.norms.clear();
        m.emit_conditioned_norm_shift(100,200,300,400,31,12288);
        assert(m.norms.size()==(delay==0 ? 2 : 3));
        assert(m.norms.front()==std::vector<int64_t>({100,200,300,400,1}));
        if(delay) assert(m.norms[1]==std::vector<int64_t>({2148,2248,12588,12688,delay}));
        assert(m.norms.back()==std::vector<int64_t>({100+(1+delay)*2048,
            200+(1+delay)*2048,24876,24976,30-delay}));
        dma_base=nullptr; consumed.clear();
        m.action_prefix_src_base_=1000+delay;
        m.emit_rtc_prefix_clamp();
        if(delay) {
            assert(dma_base==&m.action_prefix_src_base_ && dma_count==delay*96);
            assert(consumed==std::vector<int64_t>({delay*96}));
            // Captured DMA stores a pointer to the current source address.
            m.action_prefix_src_base_=9000+delay; assert(*dma_base==9000+delay);
        } else assert(dma_base==nullptr && consumed.empty());
    }
    m.prepare_rtc(0); assert(m.planner_cache_identity()==identity0 && m.generation==1);
    reject([&]{m.prepare_rtc(-1);}); reject([&]{m.prepare_rtc(7);});
    assert(m.rtc_delay_==0 && m.generation==1);
    reject([&]{m.emit_conditioned_norm_shift(100,200,300,400,30,12288);});
    Model ordinary;
    ordinary.emit_conditioned_norm_shift(100,200,300,400,31,12288);
    assert(ordinary.norms==std::vector<std::vector<int64_t>>({{100,200,300,400,31}}));
}
'''.replace("METHODS", methods).replace("HELPER", helper)
    path = tmp_path / "rtc.cpp"
    path.write_text(program)
    binary = tmp_path / "rtc"
    build = subprocess.run([compiler, "-std=c++17", "-O0", str(path), "-o", str(binary)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_rtc_resident_actual_spm_layout_fits_without_group_alias(tmp_path):
    from test_rhinovla_mask_residency import _declaration_program, _run_cpp

    fmb = (ROOT / "src/core/fused_model_base.cpp").read_text()
    header = (ROOT / "src/core/fused_model_base.h").read_text()
    budget = "namespace detail {" + _definition(header, "struct FmbPreparedSpmCost") + ";}\n"
    budget += _definition(fmb, "int64_t detail::FmbPreparedSpmCost::available(") + "\n"
    budget += _definition(fmb, "bool detail::FmbPreparedSpmCost::fits(") + "\n"
    program = _declaration_program() + budget + r'''
int main() {
 RhinoVLAModel m;
 m.rtc_mode_=true;
 m.denoise_loop_weights_ready_=m.direct_action_input_=true;
 m.action_dim_pad_=m.state_dim_pad_=96;
 for(bool resident:{false,true})for(bool aligned:{false,true})
 for(bool packed:{false,true})for(bool vector_q:{false,true})for(int64_t rows:{31,32}) {
  m.cold_config_.adarms_resident=resident;
  m.cold_config_.aligned_kv=aligned;
  m.cold_config_.packed_qkv=packed;
  m.cold_config_.gated_no_sub=true;
  m.vector_q_norm_=vector_q;
  LayoutContext l;l.chunk_size=rows;l.max_kv_seq_len=272;
  l.use_attn_mask=true;l.is_causal=false;
  auto ds=m.declare_buffers(l);
  std::vector<SpmAllocator::AllocRequest> req;
  int64_t persistent=0;size_t pair=0,final=0;
  for(const auto& d:ds) {
   if(d.storage==StorageClass::Persistent||d.storage==StorageClass::PersistentPerLayer)
    persistent+=Align(d.size,int64_t{256})*(d.storage==StorageClass::PersistentPerLayer?d.per_layer:1);
   if(d.storage!=StorageClass::Temp||d.alias_of)continue;
   if(std::string(d.name)=="adarms_pair_resident") {
    pair=req.size(); assert(d.size==3*18*6*1024*2);
   }
   if(std::string(d.name)=="adarms_final_resident") {
    final=req.size(); assert(d.size==3*3*1024*2);
   }
   req.push_back({d.size,d.phase_start,d.phase_end,static_cast<int>(d.scope)});
  }
  auto plan=SpmAllocator::plan_temporary_aliased(req);
  // Run the real planner budget logic, including previously installed
  // Vision/Text owners and the extra 64 KiB physical reserve. Cover C32 dry
  // candidates as well as the actual M31 suffix at physical prefix P240.
  constexpr int64_t usable=8*1024*1024-1024,budget=usable*99/100;
  detail::FmbPreparedSpmCost cost{persistent,0,int64_t(plan.peak_bytes)};
  for(int64_t prior:{0,512*1024,1024*1024,2*1024*1024}) {
   assert(cost.fits(usable-prior,false,budget));
   assert(cost.fits(usable-prior-persistent,true,budget));
  }
  if(resident) {
   auto obsolete=req;
   obsolete[pair].bytes*=10;obsolete[final].bytes*=10;
   auto whole_loop=SpmAllocator::plan_temporary_aliased(obsolete);
   detail::FmbPreparedSpmCost old{persistent,0,int64_t(whole_loop.peak_bytes)};
   assert(old.fits(usable,false,budget));
   assert(!old.fits(usable-512*1024,false,budget));
  }
  if(resident)for(size_t group:{pair,final})for(size_t j=0;j<req.size();++j)if(j!=group) {
   assert(plan.offsets[group]+req[group].bytes<=plan.offsets[j] ||
          plan.offsets[j]+req[j].bytes<=plan.offsets[group]);
  }
  std::cout<<resident<<","<<aligned<<","<<packed<<","<<vector_q<<","<<persistent+plan.peak_bytes<<"\n";
 }
 return 0;
}
}
int main(){return v3::main();}
'''
    output = _run_cpp(tmp_path, program)
    assert len(output.strip().splitlines()) == 32


def test_rtc_resident_dma_and_fused_gate_groups_execute_selected_delay(tmp_path):
    from test_rhinovla_mask_residency import _run_cpp

    source = (ROOT / "src/fused/rpu_rhino_vla_model.cpp").read_text()
    methods = "\n".join(_definition(source, prefix) for prefix in (
        "void emit_resident_adarms_tables()",
        "template <typename Emit>\n    void for_each_rtc_group(",
        "void emit_rtc_gated_residual(",
    ))
    program = r'''
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
constexpr int64_t DWIDTH=2, NUM_CORES=8;
namespace c10 {using Half=float;}
enum class FmbRouteFamily {GRAPH_SCHEDULE,MUTABLE_DMA};
enum class RhinoMutableDmaRoute {DDR_BROADCAST_TO_SPM};
enum class ValuOpType {TANH,MUL,ADD};
constexpr int64_t RHINO_ADARMS_RESIDENT_SITE=1,RHINO_RESIDENT_PAIR_DMA_SITE=2,RHINO_RESIDENT_FINAL_DMA_SITE=3;
struct Route {int64_t site,flags;std::vector<int64_t> args;};
struct Context {
 int64_t body_iter=0;
 std::vector<Route> routes;
 void consume_physical_route(FmbRouteFamily,int64_t site,int64_t,int flags,std::vector<int64_t> args) {
  routes.push_back({site,flags,args});
 }
};
struct Dma {uint64_t* source;int64_t offset,elements;uint32_t destination;};
std::vector<Dma> dmas;
void rpu_launch_ddr_broadcast_spm_dma_mutable(uint64_t* src,int64_t offset,int64_t elements,uint32_t dest,int cores) {
 assert(cores==8);dmas.push_back({src,offset,elements,dest});
}
constexpr uint32_t update_base=0,residual_base=31*1024*2,gate_base=62*1024*2;
std::vector<float> spm(100*1024,0);
std::vector<int64_t> tanh_addresses,fused_rows,mul_rows;
int adds=0;
void rpu_launch_eltwise_unary_spm_kernel(uint32_t a,uint32_t b,int64_t elements,ValuOpType op,bool,int cores,int precision) {
 assert(op==ValuOpType::TANH && a==b && elements==1024 && cores==8);
 assert(precision==1);tanh_addresses.push_back(a);
 for(int64_t i=0;i<elements;++i)spm[a/2+i]=std::tanh(spm[a/2+i]);
}
void rpu_launch_pi05_gated_residual_spm_kernel(uint32_t a,uint32_t b,uint32_t gate,uint32_t out,int64_t rows,int cores) {
 assert(out==a && cores==8);fused_rows.push_back(rows);
 for(int64_t row=0;row<rows;++row)for(int64_t h=0;h<1024;++h)
  spm[out/2+row*1024+h]=spm[a/2+row*1024+h]*spm[gate/2+h]+spm[b/2+row*1024+h];
}
void rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(uint32_t gate,uint32_t a,uint32_t out,int64_t rows,int64_t h,float alpha,ValuOpType op,bool,int=8) {
 assert(h==1024 && alpha==1 && op==ValuOpType::MUL);mul_rows.push_back(rows);
 for(int64_t row=0;row<rows;++row)for(int64_t col=0;col<h;++col)
  spm[out/2+row*h+col]=spm[a/2+row*h+col]*spm[gate/2+col];
}
void rpu_launch_eltwise_binary_spm_kernel(uint32_t a,uint32_t b,uint32_t out,int64_t elements,ValuOpType op,float alpha,int cores) {
 assert(op==ValuOpType::ADD && alpha==1 && cores==8 && elements==31*1024);++adds;
 for(int64_t i=0;i<elements;++i)spm[out/2+i]=spm[a/2+i]+spm[b/2+i];
}
struct Model {
 bool rtc_mode_=true,gates_tanh_precomputed_=false;
 struct {bool expert_fusions=false;}cold_config_;
 int64_t rtc_delay_=0,num_steps_=10,action_horizon_=30,suffix_len_=31;
 uint64_t adarms_pair_table_src_base_=100,adarms_final_table_src_base_=200;
 Context context;
 Context& ctx(){return context;}
 int64_t hidden_size()const{return 1024;}int64_t num_layers()const{return 18;}int num_cores()const{return 8;}
 int unary_precision()const{return 1;}
 uint32_t addr(int core,const char* name) {
  assert(core==0);return std::strcmp(name,"adarms_pair_resident")==0?4096:8000000;
 }
METHODS
};
int main() {
 Model m;
 for(bool rtc:{false,true})for(int delay=0;delay<=6;++delay)for(int step=0;step<10;++step) {
  m.rtc_mode_=rtc;m.rtc_delay_=delay;m.context.body_iter=step;
  m.context.routes.clear();dmas.clear();
  m.emit_resident_adarms_tables();
  const int64_t groups=rtc?3:1,resident_steps=rtc?1:10;
  const int64_t pair=groups*resident_steps*18*6*1024,final=groups*resident_steps*3*1024;
  const int64_t table_step=rtc?delay*10+step:0;
  assert(dmas.size()==2 && m.context.routes.size()==3);
  assert(dmas[0].source==&m.adarms_pair_table_src_base_ && dmas[0].elements==pair);
  assert(dmas[1].source==&m.adarms_final_table_src_base_ && dmas[1].elements==final);
  assert(dmas[0].offset==table_step*pair*2 && dmas[1].offset==table_step*final*2);
  assert(m.context.routes[0].args==std::vector<int64_t>({resident_steps,18,1024,(pair+final)*2}));
  assert(m.context.routes[0].flags==(rtc?1:0));
  assert(m.context.routes[1].args==std::vector<int64_t>({pair}));
  assert(m.context.routes[2].args==std::vector<int64_t>({final}));
  m.adarms_pair_table_src_base_+=10;assert(*dmas[0].source==m.adarms_pair_table_src_base_);
 }
 m.rtc_mode_=true;
 for(int delay=0;delay<=6;++delay)for(bool fused:{false,true})for(bool precomputed:{false,true}) {
  m.rtc_delay_=delay;m.cold_config_.expert_fusions=fused;m.gates_tanh_precomputed_=precomputed;
  tanh_addresses.clear();fused_rows.clear();mul_rows.clear();adds=0;
  for(int group=0;group<3;++group)for(int h=0;h<1024;++h)
   spm[gate_base/2+group*6*1024+h]=precomputed?std::tanh(float(group+1)/4):float(group+1)/4;
  for(int row=0;row<31;++row)for(int h=0;h<1024;++h) {
   spm[update_base/2+row*1024+h]=float(row+1)/32;
   spm[residual_base/2+row*1024+h]=-float(row+1)/64;
  }
  m.emit_rtc_gated_residual(update_base,residual_base,gate_base);
  const std::vector<int64_t> rows=delay?std::vector<int64_t>{1,delay,30-delay}:std::vector<int64_t>{1,30};
  assert((fused?fused_rows:mul_rows)==rows);
  assert((fused?mul_rows:fused_rows).empty() && adds==(fused?0:1));
  assert(tanh_addresses.size()==(precomputed?0:rows.size()));
  for(int row=0;row<31;++row)for(int h=0;h<1024;++h) {
   const int group=row==0?0:row<=delay?1:2;
   const float expected=float(row+1)/32*std::tanh(float(group+1)/4)-float(row+1)/64;
   assert(std::abs(spm[row*1024+h]-expected)<1e-6);
  }
 }
}
'''.replace("METHODS", methods)
    _run_cpp(tmp_path, program)


def test_high_gate_precompute_admits_scalar_and_rtc_rows_before_launch(tmp_path):
    from test_rhinovla_mask_residency import _run_cpp

    source = (ROOT / "src/fused/rpu_rhino_vla_model.cpp").read_text()
    wrapper = _definition(source, "at::Tensor rpu_rhino_vla_tanh_high_precision(")
    program = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do {if(!(condition))throw std::runtime_error("rejected");}while(0)
namespace at {
enum {kPrivateUse1=1,kHalf=2};
struct Device {int kind=kPrivateUse1;int type()const{return kind;}};
struct Tensor {
 std::vector<int64_t> shape;bool valid=true,contiguous=true;int dtype=kHalf;Device dev;
 bool defined()const{return valid;}Device device()const{return dev;}
 int scalar_type()const{return dtype;}bool is_contiguous()const{return contiguous;}
 int64_t dim()const{return shape.size();}int64_t size(int i)const{return shape.at(i);}
};
int allocations=0;
Tensor empty_like(const Tensor& t){++allocations;return t;}
}
bool active_graph=false,blocked_thread=false;
struct RpuExecutionCoordinator {
 static void require_graph_quiescent(const char*){TORCH_CHECK(!active_graph);}
 static void check_current_thread_execution_allowed(const char*){TORCH_CHECK(!blocked_thread);}
};
enum class ValuOpType {TANH};
enum class RpuUnaryPrecision {HIGH};
int payload_checks=0,launches=0;
void rpu_require_high_precision_math_kernels(){++payload_checks;}
void rpu_launch_eltwise_unary_kernel(const at::Tensor& input,at::Tensor& output,ValuOpType op,RpuUnaryPrecision precision) {
 assert(input.shape==output.shape && op==ValuOpType::TANH && precision==RpuUnaryPrecision::HIGH);++launches;
}
WRAPPER
template<class F>void reject(F fn){bool failed=false;try{fn();}catch(const std::runtime_error&){failed=true;}assert(failed);}
int main(){
 for(int64_t rows:{10,210}) {
  at::Tensor input{{rows,2048}};
  auto result=rpu_rhino_vla_tanh_high_precision(input);assert(result.shape==input.shape);
 }
 assert(launches==2 && at::allocations==2 && payload_checks==2);
 for(int kind=0;kind<9;++kind) {
  at::Tensor input{{210,2048}};
  switch(kind) {
   case 0:input.shape={209,2048};break;
   case 1:input.shape={211,2048};break;
   case 2:input.shape={210,1024};break;
   case 3:input.shape={7,10,3,2048};break;
   case 4:input.valid=false;break;
   case 5:input.contiguous=false;break;
   case 6:input.dev.kind=9;break;
   case 7:input.dtype=9;break;
   case 8:input.shape={0,2048};break;
  }
  reject([&]{rpu_rhino_vla_tanh_high_precision(input);});
  assert(launches==2 && at::allocations==2 && payload_checks==2);
 }
 for(bool blocked:{false,true}) {
  active_graph=!blocked;blocked_thread=blocked;
  reject([&]{rpu_rhino_vla_tanh_high_precision(at::Tensor{{210,2048}});});
  assert(launches==2 && at::allocations==2 && payload_checks==2);
 }
}
'''.replace("WRAPPER", wrapper)
    _run_cpp(tmp_path, program)
