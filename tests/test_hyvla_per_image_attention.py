"""Execute the actual per-image call schedule against strict launch metadata."""
from pathlib import Path
from test_hyvla_paired_kv import block, compile_run, COMMON

SOURCE = (Path(__file__).resolve().parents[1] / 'src/fused/rpu_hyvit2_vision_model.cpp').read_text()


def test_per_image_slices_and_tail_capacity(tmp_path):
    body = SOURCE[SOURCE.index('    void build_layer_subgraph('):]
    emit = block(body, '        if (cold_per_image_attention_) {')
    begin = SOURCE.index('        const int64_t raw_kv_rows = raw_spm')
    extent = SOURCE[begin:SOURCE.index(';', begin) + 1]
    source = COMMON + r'''
enum class AttentionExecutionPolicy{DDR_KV,SPM_KV_BY_MHA};
enum class FmbRouteFamily{ATTENTION};
constexpr int64_t HYVIT2_PER_IMAGE_RAW_SPM_ATTN_SITE=623508127183406940LL;
struct Chunk{int64_t idx=0,offset=0,len=588;};
int consumes=0,transposes=0,attentions=0;
struct Context{
 AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;
 void consume_physical_route(FmbRouteFamily,int64_t site,int64_t route,int64_t flags,
   std::vector<int64_t> args,int64_t invocation){
  assert(site==HYVIT2_PER_IMAGE_RAW_SPM_ATTN_SITE && route==1 && flags==0 && invocation==0);
  assert(args==std::vector<int64_t>({3,196,588,588,16,80,72,8,208}));++consumes;
 }
};
uint32_t addr(int,const char* name){
 const std::string n=name;
 if(n=="v")return 0x100000;if(n=="k")return 0x200000;
 if(n=="q_comp")return 0x300000;if(n=="sdpa_out")return 0x400000;
 if(n=="sdpa_tmp")return 0x500000;throw std::runtime_error("unknown slot");
}
void rpu_launch_v_transpose_spm(uint32_t v,uint32_t tmp,int b,int s,int heads,int d,int cores){
 assert(v==0x100000u+uint32_t(transposes)*196*160*2 && tmp==0x500000);
 assert(b==1 && s==196 && heads==16 && d==80 && cores==8);++transposes;
}
void rpu_launch_sdpa_by_mha_spm(uint32_t q,uint32_t k,uint32_t v,uint32_t o,
 uint32_t mask,int mask_type,double scale,int b,int sq,int sk,int qh,int kh,int d,int cores){
 const uint32_t offset=uint32_t(attentions)*196*160*2;
 assert(q==0x300000u+offset && k==0x200000u+offset && o==0x400000u+offset);
 assert(v==0x500000 && mask==0 && mask_type==0);
 assert(b==1 && sq==196 && sk==196 && qh==16 && kh==16 && d==80 && cores==8 && scale==0.125);
 assert(transposes==attentions+1);++attentions;
}
template<class T>T Align(T x,T y){return (x+y-1)/y*y;}
struct Model{
 bool cold_per_image_attention_=true,cold_kv_spm_only_=true;
 int64_t image_batch_count_=3,seq_len_=588,orig_head_dim_=72;
 int num_cores()const{return 8;}
 Context context;Context& ctx(){return context;}
 void emit(Chunk chunk={}){
  const int64_t seq_len=chunk.len,nq=16,hd=80,DWIDTH=2;const double attn_scale=0.125;
  EMIT
 }
 int64_t rows(int64_t kv_cs=592,bool raw_spm=true){ EXTENT return raw_kv_rows; }
};
int main(){
 Model m;m.emit();assert(consumes==1 && transposes==3 && attentions==3);
 assert(m.rows()==608 && (3-1)*196+208<=m.rows());
 m.cold_per_image_attention_=false;assert(m.rows()==592);m.cold_per_image_attention_=true;
 for(int which=0;which<4;++which){
  Model bad;Chunk c;
  if(which==0)bad.cold_kv_spm_only_=false;
  if(which==1)bad.image_batch_count_=2;
  if(which==2)bad.context.attention_policy=AttentionExecutionPolicy::DDR_KV;
  if(which==3)c.offset=196;
  bool rejected=false;try{bad.emit(c);}catch(const std::exception&){rejected=true;}
  assert(rejected && consumes==1 && attentions==3 && transposes==3);
 }
}
'''
    compile_run(tmp_path, 'per_image', source.replace('EMIT', emit).replace('EXTENT', extent))
