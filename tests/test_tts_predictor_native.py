"""Real ATen sampling and production loop with board-free Graph/device doubles."""
from pathlib import Path
import subprocess

import torch

ROOT = Path(__file__).resolve().parents[1]


def test_native_predictor_sampling_lifetime_and_failures(tmp_path, monkeypatch):
    monkeypatch.delenv("RPU_TTS_SAMPLING_EXPERIMENT", raising=False)
    source = (ROOT / "src/ops/rpu_tts_predictor.cpp").read_text()
    source = source[source.index("at::Tensor rpu_tts_predictor_generate("):]
    sampler = (ROOT / "src/ops/tts_sampling_experiment.h").read_text()
    checked_result = sampler[sampler.index("static at::Tensor tts_sampling_checked_result("):
                     sampler.index("// Experimental host wrapper")]
    harness = r'''
#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/record_function.h>
#include <cassert>
#include <cmath>
#include <limits>
#include <cstdlib>
struct TtsSamplingExperiment {
 explicit TtsSamplingExperiment(const std::string&) { TORCH_CHECK(false, "unexpected experimental path"); }
 void reset() {}
 at::Tensor run(const at::Tensor&,const at::Tensor&,int) { return {}; }
 at::Tensor finish() { return {}; }
};
int fail=0, forwards=0, touches=0;
bool ties=false;
struct GraphSignature{std::string op_id_str;std::vector<int64_t> dyn_dims{1};
 bool operator==(const GraphSignature&o)const{return op_id_str==o.op_id_str&&dyn_dims==o.dyn_dims;}};
struct RpuKernelGraph{
 enum class State{BUILT,REPLAYING}; State s=State::BUILT; GraphSignature sig;
 static inline bool active=false;bool replayable()const{return true;}bool has_built_signature()const{return true;}
 const GraphSignature& built_signature()const{return sig;}State state()const{return s;}
 static bool has_active(){return active;}
 void begin(const GraphSignature&){assert(!active);active=true;s=State::REPLAYING;}
 void end(){TORCH_CHECK(fail!=2,"end failed");active=false;s=State::BUILT;}
 void abort(){active=false;s=State::BUILT;}
};
struct RpuGraphCache{
 std::shared_ptr<RpuKernelGraph> p=std::make_shared<RpuKernelGraph>(),d=std::make_shared<RpuKernelGraph>();
 RpuGraphCache(){p->sig.op_id_str="qwen3_tts_prefill_predictor";d->sig.op_id_str="qwen3_tts_predictor";}
 std::shared_ptr<RpuKernelGraph> lookup_shared(const GraphSignature&s){return s==p->sig?p:s==d->sig?d:nullptr;}
 void touch_entry(const GraphSignature&,bool replay){assert(replay);++touches;}
};
at::Tensor current_logits;
at::Tensor scores_for(int index){
 auto s=at::linspace(-2,2,2048).to(at::kHalf);
 if(ties)s.narrow(0,1988,60).fill_(2);
 return s.roll({index},{0}).view({1,1,2048});
}
at::Tensor rpu_causal_decoder_forward(int64_t,const at::Tensor&x,at::TensorList,at::TensorList,
 const std::optional<at::Tensor>&,int64_t pos,bool,const std::optional<at::Tensor>&,
 const std::optional<std::vector<at::Tensor>>&,const std::optional<at::Tensor>&,
 const std::optional<at::Tensor>&,int64_t,int64_t,bool,int64_t,at::IntArrayRef){
 TORCH_CHECK(fail!=1,"forward failed");int index=forwards++%15;assert(pos==(index==0?0:index+1));
 if(index==0){assert(x.size(1)==16);assert(x.narrow(1,2,14).eq(0).all().item<bool>());}
 else {assert(x.size(1)==1);assert(x.eq(index-1).all().item<bool>());}
 current_logits=scores_for(index);if(fail==3)current_logits.fill_(NAN);
 return fail==4?x.narrow(2,0,1):x;
}
at::Tensor rpu_causal_decoder_tts_predictor_logits(int64_t){return current_logits;}
'''
    cases = r'''
int main(){
 auto logits_template=at::zeros({1,1,2048},at::kHalf);
 auto reusable=at::zeros({1,1,1024},at::kHalf);
 assert(tts_sampling_embedding_output(logits_template,reusable).is_same(reusable));
 auto fresh=tts_sampling_embedding_output(logits_template,{});fresh.fill_(1);auto held=fresh.clone();
 auto next=tts_sampling_embedding_output(logits_template,{});
 assert(fresh.data_ptr()!=next.data_ptr());
 next.fill_(3);assert(at::equal(fresh,held));
 for(int bad=0;bad<4;++bad){
  bool rejected=false;
  auto invalid=bad==0?reusable.to(at::kFloat):bad==1?reusable.narrow(2,0,1023):
      bad==2?at::empty({1,1,2048},at::kHalf).slice(2,0,2048,2):reusable.squeeze(0);
  try{tts_sampling_embedding_output(logits_template,invalid);}catch(const c10::Error&){rejected=true;}
  assert(rejected);
 }
 auto valid=at::ones({15,2048},at::kHalf);
 auto ids=at::zeros({15,8},at::kInt);
 ids.select(1,0).copy_(at::arange(15,at::kInt));
 auto result=tts_sampling_checked_result(valid,ids);
 assert(result.scalar_type()==at::kLong&&result.sizes().equals({1,15}));
 auto packed=at::ones({15,2064},at::kHalf);
 auto* packed_bytes=reinterpret_cast<uint8_t*>(packed.data_ptr<c10::Half>());
 for(int i=0;i<15;++i)*reinterpret_cast<int32_t*>(packed_bytes+i*4128+4096)=i;
 assert(at::equal(tts_sampling_checked_packed_result(packed),result));
 packed[14][0]=0;
 bool packed_rejected=false;try{tts_sampling_checked_packed_result(packed);}catch(const c10::Error&){packed_rejected=true;}
 assert(packed_rejected);packed[14][0]=1;
 ids.fill_(2047);assert(result[0][14].item<int>()==14);
 assert(tts_sampling_checked_result(valid,ids).eq(2047).all().item<bool>());
 for(float value:{0.f,-1.f,2.f,float(NAN),float(INFINITY)}){
  valid[14][2047]=value;
  bool rejected=false;try{tts_sampling_checked_result(valid,ids);}catch(const c10::Error&){rejected=true;}
  assert(rejected);valid.fill_(1);
 }
 for(int value:{-1,2048}){
  ids[14][0]=value;
  bool rejected=false;try{tts_sampling_checked_result(valid,ids);}catch(const c10::Error&){rejected=true;}
  assert(rejected);ids.zero_();
 }
 for(int fault=0;fault<3;++fault){
  bool rejected=false;try{
   tts_sampling_checked_result(fault==0?valid.narrow(0,0,14):fault==1?valid.to(at::kFloat):valid,
                               fault==2?ids.to(at::kLong):ids);
  }catch(const c10::Error&){rejected=true;}
  assert(rejected);
 }
 // Synthetic near-tie: uncentered FP16 scaling flips the FP32 winner.
 // Reference for the separately CModel-verified device centering operator.
 auto tts_sampling_centered_logits=[](const at::Tensor& scores){
  return (scores-scores.max()).to(at::kHalf).contiguous();
 };
 auto logits=at::zeros({1,1,2048});
 logits[0][0][0]=20;logits[0][0][1]=24;logits.narrow(2,2,48).fill_(16);
 auto saved_logits=logits.clone();
 auto noise=at::full_like(logits,100);noise[0][0][0]=std::exp(-4.45);noise[0][0][1]=1;
 auto choose=[&](at::Tensor scores,bool half){
  auto threshold=std::get<0>(scores.topk(50,-1)).select(-1,49).unsqueeze(-1);
  auto prob=scores.masked_fill(scores<threshold,-INFINITY).softmax(-1);
  auto race=half?prob.to(at::kHalf).to(at::kFloat)/noise.to(at::kHalf).to(at::kFloat):prob/noise;
  return (half?race.to(at::kHalf):race).argmax().item<int>();
 };
 const float temperature=float(c10::Half(0.9));
 assert(choose(logits/0.9,false)==0);
 assert(choose((logits/temperature).to(at::kHalf).to(at::kFloat),true)==1);
 auto centered=tts_sampling_centered_logits(logits);
 assert(centered.scalar_type()==at::kHalf&&centered.is_contiguous());
 assert(centered.max().item<float>()==0&&at::equal(logits,saved_logits));
 assert(at::equal(centered,tts_sampling_centered_logits(logits+32)));
 assert(tts_sampling_centered_logits(at::full({1,1,2048},100)).eq(0).all().item<bool>());
 auto extremes=tts_sampling_centered_logits(at::linspace(-100,100,2048).view({1,1,2048}));
 assert(at::isfinite(extremes).all().item<bool>()&&extremes.min().item<float>()==-200);
 assert(choose((centered.to(at::kFloat)/temperature).to(at::kHalf).to(at::kFloat),true)==0);
 RpuGraphCache cache;std::vector<at::Tensor> weights,kv(5,at::zeros({1},at::kHalf));
 for(int i=0;i<14;++i)weights.push_back(at::full({2048,1024},float(i)));
 auto input=at::zeros({1,2,1024});auto progress=at::zeros({4},at::kLong);
 auto run=[&](){return rpu_tts_predictor_generate(1,input,weights,kv,kv,cache,cache.p->sig,cache.d->sig,{1},{2},16,progress);};
 for(int seed:{0,17,73})for(bool tie:{false,true}){
  ties=tie;forwards=touches=0;progress.zero_();at::manual_seed(seed);
  auto actual=run();auto saved=actual.clone();
  auto rng=at::detail::getDefaultCPUGenerator().get_state();
  at::manual_seed(seed);std::vector<at::Tensor> expected;
  for(int i=0;i<15;++i){
   auto scores=scores_for(i).select(1,0).to(at::kFloat)/0.9;
   auto top=std::get<0>(scores.topk(50,-1));
   scores.masked_fill_(scores<top.select(1,49).unsqueeze(1),-INFINITY);
   expected.push_back(at::multinomial(scores.softmax(-1),1,false));
  }
  assert(at::equal(actual,at::cat(expected,1)));
  assert(at::equal(rng,at::detail::getDefaultCPUGenerator().get_state()));
  assert(progress[0].item<int>()==15&&progress[1].item<int>()==16&&progress[2].item<int>()==15&&progress[3].item<int>()==15);
  assert(touches==15&&!RpuKernelGraph::has_active());progress.zero_();run();assert(at::equal(actual,saved));
 }
 for(int fault=1;fault<=7;++fault){
  fail=fault;forwards=touches=0;progress.zero_();auto rng=at::detail::getDefaultCPUGenerator().get_state();
  if(fault==5)input.fill_(INFINITY);
  if(fault==6)input.fill_(1e10);
  if(fault==7)weights[0]=at::zeros({2047,1024});
  bool rejected=false;try{run();}catch(const c10::Error&){rejected=true;}
  assert(rejected&&!RpuKernelGraph::has_active());assert(at::equal(rng,at::detail::getDefaultCPUGenerator().get_state()));
  if(fault==3){assert(progress[0].item<int>()==1&&progress[2].item<int>()==1&&progress[3].item<int>()==0);}
  if(fault>=5)assert(forwards==0);
  input.zero_();weights[0]=at::zeros({2048,1024});
 }
 fail=0;forwards=0;progress.zero_();run();
 setenv("RPU_TTS_SAMPLING_EXPERIMENT","unused",1);
 progress.zero_();forwards=0;
 for(auto& weight:weights)weight=at::zeros({1,1024},at::kHalf).expand({2048,1024});
 bool rejected=false;try{run();}catch(const c10::Error& e){
  rejected=std::string(e.what()).find("contiguous embeddings")!=std::string::npos;
 }
 assert(rejected&&forwards==0);
 unsetenv("RPU_TTS_SAMPLING_EXPERIMENT");
}
'''
    cpp, exe = tmp_path / "predictor.cpp", tmp_path / "predictor"
    cpp.write_text(harness + checked_result + source + cases)
    root = Path(torch.__file__).resolve().parent
    subprocess.run(["g++", "-std=c++17", "-O0",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I", str(root / "include"), str(cpp), "-L", str(root / "lib"),
        "-ltorch_cpu", "-lc10", f"-Wl,-rpath,{root / 'lib'}", "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
