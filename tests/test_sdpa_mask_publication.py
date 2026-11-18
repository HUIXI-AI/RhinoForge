"""Run production mask preparation with observable CPU/device copy boundaries."""
from pathlib import Path
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition


def test_mask_goes_directly_to_its_stable_slot_with_padding_and_ordinals(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (Path(__file__).parents[1] / "src/ops/rpu_sdpa.cpp").read_text()
    # The CPU reduce/pad helper and the publication that calls it.
    method = (_definition(source, "static at::Tensor sdpa_mask_padded_cpu(") + "\n"
              + _definition(source, "PreparedMask sdpa_prepare_mask("))
    program = r'''
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
int publications=0, downloads=0;
namespace c10 { template<class T> using optional=std::optional<T>; enum class DeviceType { CPU=0, device=1 }; }
namespace at {
constexpr int kCPU=0,kPrivateUse1=1,kHalf=2,kFloat=3;
struct Device { int value; c10::DeviceType type() const {return static_cast<c10::DeviceType>(value);} };
struct Options { int dev=0,dtype=kHalf; Options device(int d) const {return {d,dtype};} };
struct Tensor {
    std::vector<int64_t> shape;
    std::shared_ptr<std::vector<float>> data;
    Options opts;
    Tensor()=default;
    Tensor(std::vector<int64_t> s,Options o={}):shape(s),opts(o) {
        int64_t count=1; for(auto n:s)count*=n; data=std::make_shared<std::vector<float>>(count,0.f);
    }
    bool defined() const {return bool(data);}
    int dim() const {return shape.size();}
    int64_t size(int i) const {return shape.at(i);}
    int scalar_type() const {return opts.dtype;}
    Device device() const {return {opts.dev};}
    Options options() const {return opts;}
    Tensor select(int axis,int index) const {
        assert(axis==0 && index==0 && shape[0]==1); auto t=*this; t.shape.erase(t.shape.begin());return t;
    }
    Tensor contiguous() const {return *this;}
    Tensor to(int kind) const {
        auto t=*this;
        if(kind==kPrivateUse1)throw std::runtime_error("temporary device upload");
        if(kind==kCPU) {if(opts.dev==kPrivateUse1)++downloads; t.opts.dev=kCPU;}
        else t.opts.dtype=kind;
        return t;
    }
    void copy_(const Tensor& src) {
        assert(opts.dev==kPrivateUse1 && src.opts.dev==kCPU && shape==src.shape);
        *data=*src.data; ++publications;
    }
};
Tensor constant_pad_nd(const Tensor& t,std::initializer_list<int64_t> pad,float value) {
    auto n=*(pad.begin()+1);Tensor out({t.size(0),t.size(1)+n},t.opts);
    for(int r=0;r<t.size(0);++r)for(int c=0;c<out.size(1);++c)
        (*out.data)[r*out.size(1)+c]=c<t.size(1)?(*t.data)[r*t.size(1)+c]:value;
    return out;
}
}
int64_t Align(int64_t n,int64_t a){return (n+a-1)/a*a;}
struct PreparedMask {at::Tensor ddr_tensor;int mask_type=0;};
struct SdpaStableMaskCache {
    std::map<std::tuple<int64_t,int64_t,int64_t>,at::Tensor> slots;
    at::Tensor stable_slot(int64_t q,int64_t k,at::Options opts,int64_t ordinal) {
        assert(opts.dev==at::kPrivateUse1 && opts.dtype==at::kHalf);
        auto key=std::make_tuple(q,k,ordinal);auto it=slots.find(key);
        if(it==slots.end())it=slots.emplace(key,at::Tensor({q,k},opts)).first;
        return it->second;
    }
};
METHOD
int main() {
    SdpaStableMaskCache cache;
    at::Tensor input({1,1,2,17},{at::kCPU,at::kFloat});
    for(int i=0;i<34;++i)(*input.data)[i]=float(i);
    auto first=sdpa_prepare_mask(input,false,2,17,cache,0);
    assert(publications==1 && first.mask_type==4 && first.ddr_tensor.shape==std::vector<int64_t>({2,32}));
    for(int r=0;r<2;++r)for(int c=0;c<32;++c) {
        float v=(*first.ddr_tensor.data)[r*32+c];
        if(c<17)assert(v==float(r*17+c));else assert(std::isinf(v) && v<0);
    }
    auto other=sdpa_prepare_mask(input,false,2,17,cache,1);
    assert(other.ddr_tensor.data!=first.ddr_tensor.data);
    (*input.data)[0]=99;
    auto replay=sdpa_prepare_mask(input,false,2,17,cache,0);
    assert(replay.ddr_tensor.data==first.ddr_tensor.data && (*first.ddr_tensor.data)[0]==99);
    assert((*other.ddr_tensor.data)[0]==0 && publications==3);
    at::Tensor resident({2,16},{at::kPrivateUse1,at::kHalf});
    sdpa_prepare_mask(resident,false,2,16,cache,0);
    assert(downloads==1 && publications==4);
    assert(sdpa_prepare_mask(input,true,2,17,cache,0).mask_type==1);
    assert(sdpa_prepare_mask({},false,2,17,cache,0).mask_type==0);
    bool rejected=false;try{sdpa_prepare_mask(input,false,3,17,cache,0);}catch(const std::runtime_error&){rejected=true;}
    assert(rejected && publications==4);
}
'''.replace("METHOD", method)
    path, binary = tmp_path / "mask.cpp", tmp_path / "mask"
    path.write_text(program)
    subprocess.run([compiler, "-std=c++17", str(path), "-o", str(binary)], check=True,
                   capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_gr00t_cpu_masks_use_actual_frontdoor_stable_slots_and_force_policy(tmp_path):
    """Real ATen/native receipt/copy bytes; device, SDK and execution are doubles."""
    import re
    import torch
    from test_siglip_attention_dry_live_contract import _function, _run_native
    from test_gemma_private_mask_publication import _mask_publication_native_prefix

    root = Path(__file__).parents[1]
    source = (root / "src/fused/rpu_gr00t_dit_model.cpp").read_text()
    prefix, prepared, cache_decl, cache_impl, builder = _mask_publication_native_prefix()
    prefix = prefix.replace(" bool is_inference()const", " bool requires_grad()const{return value.requires_grad();}\n bool is_neg()const{return value.is_neg();}bool is_conj()const{return value.is_conj();}\n bool is_inference()const")
    def method(marker):
        # Function arguments include default {} Tensor/descriptor values.
        start = source.index(marker)
        cursor = source.index("(", start) + 1
        depth = 1
        while depth:
            depth += (source[cursor] == "(") - (source[cursor] == ")")
            cursor += 1
        return source[start:cursor] + _function(source[cursor:], "{")

    logic = source[source.index("    struct Gr00tMaskSpan {"):source.index("    std::vector<int64_t> kvinsert_cost_weight_identity()")]
    ordinary = _function(source, "    PreparedMask prep_mask_into(")
    unroll = method("    at::Tensor unroll_forward(")
    unroll_call = next(line for line in unroll.splitlines() if "prepare_unroll_masks(" in line)
    bridge = _function(source, "at::Tensor rpu_gr00t_denoise_unroll_forward(")
    clears = []
    for name in ("set_weights", "set_denoise_weights"):
        body = method(f"    void {name}(")
        clears.append(next(line for line in body.splitlines() if "unroll_mask_publications_ = {};" in line))
    # Execute the actual preparation statements from every other writer, all
    # of which must revoke the ordinary-unroll receipts before any copy.
    routes = []
    for name in ("forward", "step_forward", "prepare_z2_masks", "unroll_forward_z2"):
        body = method(("    void " if name == "prepare_z2_masks" else "    at::Tensor ") + name + "(")
        calls = re.findall(r"prep_mask_into\([\s\S]*?\);", body)
        assert len(calls) == 2, name
        calls = [call.replace("cross_mask_img", "imask").replace("cross_mask", "tmask") for call in calls]
        routes.append(f"case {len(routes)}: " + " ".join(calls) + " break;")
    code = r"""
using HostTensorList=c10::ArrayRef<HostTensor>;
struct Model{
 int64_t vl_seq_=225;uint64_t generation=1;SdpaStableMaskCache cache;
 PreparedMask prepared_cross_mask_,prepared_cross_mask_img_;
 HostTensor text_mask_slot_,image_mask_slot_;
 uint64_t installed_model_state_generation()const{return generation;}
 SdpaStableMaskCache& sdpa_stable_mask_cache(){return cache;}
 void reinstall(bool denoise,bool fail){if(denoise){__CLEAR_DENOISE__}else{__CLEAR_WEIGHTS__}
  TORCH_CHECK(!fail,"failed weight validation");++generation;}
 __ORDINARY__
 __LOGIC__
 void other_writer(int route,const HostTensor& tmask,const HostTensor& imask){
  int64_t seq_q=tmask.size(0);switch(route){__ROUTES__ default:assert(false);}
 }
 HostTensor unroll_forward(const HostTensor& noise,const HostTensor& state,
  std::vector<HostTensor>& keys,std::vector<HostTensor>& values,
  const HostTensor& vl,const HostTensor& tmask,const HostTensor& imask,at::IntArrayRef descriptor){
  assert(noise.device().type()==at::kPrivateUse1&&state.device().type()==at::kPrivateUse1&&vl.device().type()==at::kPrivateUse1);
  assert(descriptor==at::IntArrayRef({19,23}));int64_t seq_q=tmask.size(0);
  __UNROLL_CALL__
  // Numerical consumer double reads the separately published DDR, not the CPU
  // backing. It returns a fresh CPU observation; no native math/Graph claim.
  auto output=make_tensor({1,40,144},at::kCPU);
  auto value=reinterpret_cast<const c10::Half*>(text_mask_slot_.store->ddr.data())[0];
  output.value.fill_(float(value));return output;
 }
};
Model* live_model=nullptr;
struct Gr00tDiTRegistry{static Model* get(int64_t h,const char*){assert(h==7);return live_model;}};
__BRIDGE__
int64_t version(const HostTensor& t){return t.is_inference()?-1:t.value._version();}
void raw_fill(const HostTensor& t,float v){for(int64_t i=0;i<t.numel();++i)t.data_ptr<c10::Half>()[i]=v;}
void raw_bits(const HostTensor& t,uint16_t v){for(int64_t i=0;i<t.numel();++i)static_cast<uint16_t*>(t.data_ptr())[i]=v;}
bool empty(const Model& m){return !m.unroll_mask_publications_[0].published&&!m.unroll_mask_publications_[1].published;}
void verify(const Model& m,const HostTensor& text,const HostTensor& image,bool published){
 for(auto pair:{std::make_pair(text,m.text_mask_slot_),std::make_pair(image,m.image_mask_slot_)}){
  auto source=pair.first.value;
  if(source.dim()==3)source=source.select(0,0);
  source=source.to(at::kHalf).contiguous();
  auto expected=at::constant_pad_nd(source,{0,pair.second.size(1)-source.size(1)},-std::numeric_limits<float>::infinity());
  assert(std::memcmp(pair.second.data_ptr(),expected.data_ptr(),expected.nbytes())==0);
  if(published)assert(std::memcmp(pair.second.store->ddr.data(),expected.data_ptr(),expected.nbytes())==0);
 }
}
HostTensor feed(Model& m,const HostTensor& a,const HostTensor& b){
 live_model=&m;auto noise=make_tensor({1,40,144},at::kPrivateUse1),state=make_tensor({1,1,144},at::kPrivateUse1),vl=make_tensor({1,m.vl_seq_,2048},at::kPrivateUse1);
 return rpu_gr00t_denoise_unroll_forward(7,noise,state,{}, {},vl,a,b,{19,23});
}
void expect(Model& m,const HostTensor& a,const HostTensor& b,size_t f,size_t c){
 static int checkpoint=0;++checkpoint;auto before_f=flushes.size(),before_c=copies;feed(m,a,b);
 if(flushes.size()-before_f!=f||copies-before_c!=c)std::cerr<<"checkpoint "<<checkpoint<<" expected "<<f<<","<<c<<" actual "<<flushes.size()-before_f<<","<<copies-before_c<<"\n";
 assert(flushes.size()-before_f==f&&copies-before_c==c);
}
int main(){at::set_num_threads(1);
 // Actual measured 41x225 masks, offset sources, all force/normal and CPU/device
 // origin combinations, ordinary/no_grad/inference. No version-as-content key.
 for(int mode:{0,1,2})for(bool normal:{false,true})for(bool force:{false,true})
 for(bool ar:{false,true})for(bool br:{false,true}){
  c10::InferenceMode inference(mode==2);at::AutoGradMode grad(mode==0);
  g_rpu_ddr_flush_enabled=normal;g_rpu_ddr_flush_force_enabled=force;Model m;
  auto a=make_tensor({43,225},ar?at::kPrivateUse1:at::kCPU).narrow(0,1,41);
  auto b=make_tensor({43,225},br?at::kPrivateUse1:at::kCPU).narrow(0,1,41);
  auto av=version(a),bv=version(b);int previous=0;auto retained=at::empty({0},at::kHalf);
  void* ap=nullptr;void* bp=nullptr;int64_t dst_version=0;
  for(int v:{1,1,3,3,1,1}){
   raw_fill(a,v);raw_fill(b,-v);auto f=flushes.size(),c=copies;
   bool hit=force&&!ar&&!br&&v==previous;auto out=feed(m,a,b);
   assert(flushes.size()-f==(force&&!hit?2+int(ar)+int(br):0));
   assert(copies-c==(hit?0:2+int(ar)+int(br)));verify(m,a,b,force);
   if(!ap){ap=m.text_mask_slot_.data_ptr();bp=m.image_mask_slot_.data_ptr();dst_version=version(m.text_mask_slot_);retained=out.value.clone();}
   else{assert(ap==m.text_mask_slot_.data_ptr()&&bp==m.image_mask_slot_.data_ptr());
    if(mode!=2){dst_version+=!hit;assert(version(m.text_mask_slot_)==dst_version);}}
   assert(ap!=bp&&version(a)==av&&version(b)==bv);
   assert(retained.eq(force?1:777).all().item<bool>());
   assert(empty(m)==!(force&&!ar&&!br));previous=v;
  }
 }
 // Boundaries on actual strong owners and mapped/device byte stores.
 for(bool inference:{false,true}){
  c10::InferenceMode context(inference);g_rpu_ddr_flush_enabled=false;g_rpu_ddr_flush_force_enabled=true;
  Model m;auto a=make_tensor({41,225},at::kCPU),b=make_tensor({41,225},at::kCPU);raw_fill(a,2);raw_fill(b,-2);
  expect(m,a,b,2,2);expect(m,a,b,0,0);auto owner=m.text_mask_slot_;auto ov=version(owner);
  // Only changed ordinal publishes; tail bits and signed zero/NaN payloads count.
  for(uint16_t bits:{uint16_t(0x8000),uint16_t(0),uint16_t(0x7e11),uint16_t(0x7e12)}){
   raw_bits(a,bits);expect(m,a,b,1,1);expect(m,a,b,0,0);verify(m,a,b,true);}
  m.text_mask_slot_.data_ptr<c10::Half>()[239]=0;expect(m,a,b,1,1);verify(m,a,b,true);
  auto replacement=make_tensor({41,225},at::kCPU);std::memcpy(replacement.data_ptr(),a.data_ptr(),a.nbytes());expect(m,replacement,b,0,0);
  // Any other writer revokes BOTH receipts before its first preparation.
  for(int route=0;route<4;++route){auto f=flushes.size();m.other_writer(route,a,b);assert(flushes.size()-f==2&&empty(m));expect(m,a,b,2,2);expect(m,a,b,0,0);}
  for(bool denoise:{false,true})for(bool fail:{false,true}){bool caught=false;try{m.reinstall(denoise,fail);}catch(const c10::Error&){caught=true;}
   assert(caught==fail&&empty(m));expect(m,a,b,2,2);}
  // Generation mismatch retires the old pair conservatively; the following
  // successful refresh establishes authority for this generation.
  ++m.generation;expect(m,a,b,2,2);expect(m,a,b,2,2);expect(m,a,b,0,0);
  g_rpu_ddr_flush_force_enabled=false;expect(m,a,b,0,2);assert(empty(m));
  raw_fill(a,13);expect(m,a,b,0,2);g_rpu_ddr_flush_force_enabled=true;expect(m,a,b,2,2);expect(m,a,b,0,0);
  // Failed first or second SDK publication clears the complete pair. CPU bytes
  // may already match, so retry must still publish both destinations.
  for(int fail:{0,1}){raw_fill(a,20+fail);raw_fill(b,-20-fail);flushes_until_failure=fail;bool caught=false;
   try{feed(m,a,b);}catch(const c10::Error&){caught=true;}assert(caught&&empty(m));expect(m,a,b,2,2);expect(m,a,b,0,0);}
  auto retained=m.text_mask_slot_;m.unroll_mask_publications_[0].mask.ddr_tensor.value=owner.value.narrow(1,0,224);
  expect(m,a,b,2,2);verify(m,a,b,true);
  m.vl_seq_=226;auto c=make_tensor({41,226},at::kCPU);raw_fill(c,9);expect(m,c,c,2,2);expect(m,c,c,0,0);
  assert(m.text_mask_slot_.is_same(retained)); // padded shape unchanged, logical K still matters
  auto d=make_tensor({42,226},at::kCPU);raw_fill(d,8);expect(m,d,d,2,2);
  assert(!m.text_mask_slot_.is_same(retained));
  bool caught=false;auto wrong=d.narrow(1,0,225);try{feed(m,wrong,d);}catch(const c10::Error&){caught=true;}assert(caught&&empty(m));
 }
 // Public CPU fallbacks never grant receipts. Pair eligibility must be checked
 // before the other, otherwise a mixed pair could hide a changed destination.
 {Model m;g_rpu_ddr_flush_force_enabled=true;auto a=make_tensor({41,225},at::kCPU);raw_fill(a,2);
  std::vector<HostTensor> alternatives;
  alternatives.push_back({at::full({41,450},4,at::kHalf).slice(1,0,450,2),{},c10::Device(at::kCPU)});
  alternatives.push_back({at::full({41,225},4,at::kFloat),{},c10::Device(at::kCPU)});
  // Lazy negative view has positive raw bytes but negative logical pad values.
  alternatives.push_back({at::_neg_view(at::full({41,225},2,at::kHalf)),{},c10::Device(at::kCPU)});
  alternatives.push_back({at::full({1,41,225},4,at::kHalf),{},c10::Device(at::kCPU)});
  expect(m,a,a,2,2);
  for(const auto& b:alternatives){expect(m,a,a,0,0);expect(m,a,b,2,2);assert(empty(m));verify(m,a,b,true);expect(m,a,a,2,2);}
  auto grad=make_tensor({41,225},at::kCPU);grad.value.set_requires_grad(true);
  expect(m,a,grad,2,2);assert(empty(m));expect(m,a,a,2,2);
  // Source full backing includes either private slot even if active bytes do
  // not. Actual from_blob views model unannounced CPU aliases of mapped DDR.
  for(int target:{0,1}){
   auto slot=target?m.image_mask_slot_:m.text_mask_slot_;
   auto ap=reinterpret_cast<uintptr_t>(a.data_ptr()),sp=reinterpret_cast<uintptr_t>(slot.data_ptr());
   auto begin=std::min(ap,sp),end=std::max(ap+a.nbytes(),sp+slot.nbytes());
   auto wide=at::from_blob(reinterpret_cast<void*>(begin),{int64_t((end-begin)/2)},at::kHalf);
   auto alias=HostTensor{wide.narrow(0,int64_t((ap-begin)/2),a.numel()).reshape(a.value.sizes()),{},c10::Device(at::kCPU)};
   expect(m,a,alias,2,2);assert(empty(m));expect(m,a,a,2,2);
  }
  auto bad=a;bad.override_span=true;bad.storage_base=std::numeric_limits<uintptr_t>::max()-1;bad.storage_bytes=64;
  expect(m,bad,a,2,2);assert(empty(m));expect(m,a,a,2,2);
  // Active raw alias of the aligned private slot must copy+publish, never hit.
  m.vl_seq_=224;auto aligned=make_tensor({41,224},at::kCPU);raw_fill(aligned,6);expect(m,aligned,aligned,2,2);
  auto slot=m.text_mask_slot_;auto alias=HostTensor{at::from_blob(slot.data_ptr(),{41,224},at::kHalf),{},c10::Device(at::kCPU)};
  raw_fill(alias,21);expect(m,alias,aligned,2,2);assert(empty(m));verify(m,alias,aligned,true);
 }
 // Returning outside InferenceMode must preserve the original copy_ rejection.
 {auto model=std::make_unique<Model>();auto a=make_tensor({41,225},at::kCPU);raw_fill(a,3);
  {c10::InferenceMode context(true);expect(*model,a,a,2,2);expect(*model,a,a,0,0);}
  bool caught=false;try{feed(*model,a,a);}catch(const c10::Error&){caught=true;}assert(caught&&empty(*model));
  {c10::InferenceMode context(true);expect(*model,a,a,2,2);expect(*model,a,a,0,0);}
 }
 std::cout<<"GR ordinary-unroll private mask publication authority passed\n";
}
"""
    code = (code.replace("__LOGIC__", logic).replace("__ORDINARY__", ordinary)
            .replace("__UNROLL_CALL__", unroll_call).replace("__BRIDGE__", bridge)
            .replace("__CLEAR_WEIGHTS__", clears[0]).replace("__CLEAR_DENOISE__", clears[1])
            .replace("__ROUTES__", "\n".join(routes)))
    native = prepared + cache_decl + cache_impl + builder + code
    native = native.replace("at::TensorOptions", "HostOptions").replace("at::Tensor", "HostTensor")
    _run_native(tmp_path, prefix + native, torch)
