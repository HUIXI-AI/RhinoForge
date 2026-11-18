// Ordinary ParallelLinear v3: N32K32 W4, runtime precision, variable M/N/K.
// Generated data mirrors rhino-ops main generator/parallel_linear_tiling.py.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "rpu_grouped_device_m_contract.h"

namespace rpu_pl_tiling {

enum class WeightType : uint8_t { Fp16, Int8, Int4, Uint4, Bf16, Fp8E4M3 };
enum class Family : uint8_t { Parallel, Grouped, Lora };
// The final source is a new-ABI fallback for unmeasured precision/layout
// combinations. It is deliberately named as such: no pre-migration
// ParallelLinear launcher or register contract is accepted here.
enum class TileSource : uint8_t {
  Explicit, Exact, CapacityFiltered, Fallback, UnmeasuredFallback
};

// Keep the original first two members for call sites; K is now part of identity.
struct TilePick {
  int n_tile = 0, m_tile = 0, k_tile = 128;
  int metadata_rows = 0, zp_offset = 0, vlm_rows = 0;
  int bias_offset = 0, msizes_offset = 0;
  TileSource source = TileSource::Explicit;
};
struct CandidateInfo {
  WeightType weight;
  bool acc32;
  TilePick tile;
  const char* name;
};

inline constexpr int ceil_div_positive(int value, int divisor) {
  return value / divisor + (value % divisor != 0);
}
inline constexpr bool is_w4(WeightType type) {
  return type == WeightType::Int4 || type == WeightType::Uint4;
}
inline constexpr int variant_index(WeightType type) {
  switch (type) {
    case WeightType::Fp16: case WeightType::Bf16: return 0;
    case WeightType::Int8: case WeightType::Fp8E4M3: return 1;
    case WeightType::Int4: case WeightType::Uint4: return 2;
  }
  return -1;
}
inline constexpr int k_index(int k) {
  switch (k) {
    case 32:return 0; case 64:return 1; case 96:return 2; case 128:return 3;
    case 192:return 4; case 256:return 5; case 384:return 6;
    case 512:return 7; case 768:return 8; default:return -1;
  }
}

inline TilePick make_tile(WeightType type, bool acc32, int n, int k,
                          Family family = Family::Parallel, int m = 0) {
  const int variant = variant_index(type);
  const bool w4 = is_w4(type);
  if (variant < 0 || n < 32 || n > 128 || n % (w4 ? 32 : 16) ||
      k_index(k) < 0 || (!w4 && k == 96) ||
      (family != Family::Parallel && family != Family::Grouped && family != Family::Lora))
    throw std::invalid_argument("Unsupported ParallelLinear N/K tile or family");
  const int nv=n/16, kv=k/16, zp=w4 ? k/32 : 0;
  int metadata=w4 ? std::max(2*zp,nv) : 0;
  int fixed=variant==0 ? nv*kv : variant==1 ? nv*(k/32+kv+1)
                  : (n/32)*(k/32)+nv*kv+metadata+(!acc32);
  const int coefficient=kv+(acc32 ? 2 : 1)*nv;
  const int reserve=family==Family::Grouped ? nv+1 : 0;
  int ceiling=(400-fixed-reserve)/coefficient*16;
  if (family==Family::Lora) {
    if(acc32) throw std::invalid_argument("LoRA supports ACC16 only");
    metadata=w4 ? std::max(k/16,nv) : 0;
    fixed=variant==0 ? kv*nv : variant==1 ? (kv+k/32+1)*nv
                                      : kv*nv+(n/32)*(k/32)+metadata;
    ceiling=std::min({(400-fixed)/(nv+kv),(399-16)/10,
                     (399-2*nv)/(2*nv+2),400/8})*16;
  }
  if(ceiling<16 || m<0 || (m && (m%16 || m>ceiling)))
    throw std::invalid_argument("ParallelLinear tile exceeds 400 VLM rows");
  if(!m) m=ceiling;
  const int base=fixed+m/16*coefficient;
  int vlm=base+reserve;
  if(family==Family::Lora) {
    const int mv=m/16;
    const int rv=std::min((399-8*mv)/(mv+8),(399-2*mv*nv)/(mv+nv));
    const int batch=400/(8*mv);
    if(base>400 || rv<2 || batch<=0)
      throw std::invalid_argument("LoRA tile exceeds rank/workspace capacity");
    vlm=std::max({base,mv*(rv+8)+1+8*rv,mv*(2*nv+rv)+1+nv*rv,8*mv*batch});
  }
  return {n,m,k,metadata,zp,vlm,reserve ? base : 0,reserve ? base+nv : 0};
}

namespace detail {
struct RankedTile { int n,k; double cost; };
struct MeasuredShape {
  int m,n,k;
  uint32_t ranked_offset;
  uint16_t ranked_count;
  std::array<double,3> feature;
};
struct MeasuredTable {
  const MeasuredShape* shapes;
  std::size_t count;
  const RankedTile* ranked;
  std::array<double,3> lower,upper;
};

#include "rpu_linear_tiling_data.h"

inline const MeasuredTable& measured_table(WeightType type,bool acc32) {
  int index=type==WeightType::Fp16 ? 0 : type==WeightType::Int8 ? 1 :
            type==WeightType::Int4 ? 2 : type==WeightType::Uint4 ? 3 : -1;
  if(index<0) throw std::invalid_argument("Precision has no measured ParallelLinear table");
  return *measured_tables[index*2+acc32];
}

// Small per-thread memo; all identity fields are compared after hashing.
// No model/tensor/device pointers survive a launch or cross library versions.
class TileSelectionCache {
 public:
  static constexpr std::size_t capacity=128;
  bool lookup(uint8_t family,int m,int64_t n,int k,TilePick& result) const noexcept {
    const auto& e=entries_[slot(family,m,n,k)];
    if(!e.valid || e.family!=family || e.m!=m || e.n!=n || e.k!=k) return false;
    result=e.result; return true;
  }
  void store(uint8_t family,int m,int64_t n,int k,TilePick result) noexcept {
    entries_[slot(family,m,n,k)]={family,m,n,k,result,true};
  }
 private:
  struct Entry {
    uint8_t family=0; int m=0; int64_t n=0; int k=0; TilePick result{}; bool valid=false;
  };
  std::array<Entry,capacity> entries_{};
  static std::size_t slot(uint8_t family,int m,int64_t n,int k) noexcept {
    uint64_t h=static_cast<uint32_t>(m);
    for(uint64_t v : {static_cast<uint64_t>(n),static_cast<uint64_t>(k),uint64_t{family}})
      h=h*UINT64_C(0x9e3779b97f4a7c15)+v;
    h^=h>>33; h*=UINT64_C(0xff51afd7ed558ccd); h^=h>>33;
    return h&(capacity-1);
  }
};
inline TileSelectionCache& selection_cache() {
  static thread_local TileSelectionCache cache;
  return cache;
}

inline int candidate_ordinal(WeightType type,bool acc32,int n,int k,Family family) {
  const int vi=variant_index(type),ki=k_index(k);
  if(vi<0 || ki<0 || n<32 || n>128 || n%16) return -1;
  const int ni=(n-32)/16;
  switch(family) {
    case Family::Parallel:return parallel_ordinals[vi][acc32][ni][ki];
    case Family::Grouped:return grouped_ordinals[vi][acc32][ni][ki];
    case Family::Lora:return lora_ordinals[vi][acc32][ni][ki];
  }
  return -1;
}
inline const CandidateInfo* candidate_data(Family family) {
  switch(family) {
    case Family::Parallel:return parallel_candidates;
    case Family::Grouped:return grouped_candidates;
    case Family::Lora:return lora_candidates;
  }
  throw std::invalid_argument("Unknown ParallelLinear family");
}

inline TilePick measured_select(int m,int64_t n,int k,WeightType type,bool acc32,Family family) {
  const int query_m=family==Family::Grouped ? 1 : m;
  // LoRA copies the ordinary PL decision, then uses its own workspace capacity.
  const Family scoring_family=family==Family::Lora ? Family::Parallel : family;
  const auto& table=measured_table(type,acc32);
  auto legal=[&](const RankedTile& tile) {
    return candidate_ordinal(type,acc32,tile.n,tile.k,scoring_family)>=0;
  };
  const auto key=std::make_tuple(query_m,n,k);
  const auto* end=table.shapes+table.count;
  const auto* found=std::lower_bound(table.shapes,end,key,[](const MeasuredShape& e,const auto& q) {
    return std::make_tuple(e.m,int64_t{e.n},e.k)<q;
  });
  if(found!=end && std::make_tuple(found->m,int64_t{found->n},found->k)==key) {
    for(unsigned j=0;j<found->ranked_count;++j) {
      const auto& rank=table.ranked[found->ranked_offset+j];
      if(!legal(rank)) continue;
      auto result=make_tile(type,acc32,rank.n,rank.k,family);
      result.source=j ? TileSource::CapacityFiltered : TileSource::Exact;
      return result;
    }
  }
  const std::array<double,3> feature{std::log2(ceil_div_positive(query_m,16)*16.0),
                                   std::log2(static_cast<double>(n)),std::log2(double(k))};
  bool within=true;
  for(unsigned i=0;i<3;++i)
    within=within && feature[i]>=table.lower[i]-1 && feature[i]<=table.upper[i]+1;
  const MeasuredShape* reference=nullptr;
  double best_distance=std::numeric_limits<double>::infinity();
  const bool identical_nk=std::any_of(table.shapes,end,[&](const MeasuredShape& e) {
    return e.n==n && e.k==k;
  });
  if(within) for(std::size_t i=0;i<table.count;++i) {
    const auto& entry=table.shapes[i];
    bool valid=false;
    for(unsigned j=0;j<entry.ranked_count && !valid;++j)
      valid=legal(table.ranked[entry.ranked_offset+j]);
    if(!valid) continue;
    const bool same=entry.n==n && entry.k==k;
    if(identical_nk && !same) continue;
    const double dm=feature[0]-entry.feature[0];
    const double dn=4*(feature[1]-entry.feature[1]);
    const double dk=feature[2]-entry.feature[2];
    const double distance=dm*dm+dn*dn+dk*dk;
    // The generated table is shape-sorted, preserving Python's shape tie break.
    if(distance<best_distance) {
      reference=&entry; best_distance=distance;
    }
  }
  int nt=64,kt=128;
  if(reference) {
    auto best=std::make_tuple(std::numeric_limits<double>::infinity(),0,0);
    for(unsigned j=0;j<reference->ranked_count;++j) {
      const auto& rank=table.ranked[reference->ranked_offset+j];
      if(!legal(rank)) continue;
      const auto capacity=make_tile(type,acc32,rank.n,rank.k);
      const double cm=double(ceil_div_positive(query_m,capacity.m_tile))/
                      ceil_div_positive(reference->m,capacity.m_tile);
      const double cn=double((n+rank.n-1)/rank.n)/ceil_div_positive(reference->n,rank.n);
      const double ck=double(ceil_div_positive(k,rank.k))/ceil_div_positive(reference->k,rank.k);
      const auto score=std::make_tuple(rank.cost*std::pow(cm,0.5)*std::pow(cn*ck,0.25),rank.n,rank.k);
      if(score<best) best=score;
    }
    nt=std::get<1>(best); kt=std::get<2>(best);
  }
  auto result=make_tile(type,acc32,nt,kt,family);
  result.source=TileSource::Fallback;
  return result;
}
} // namespace detail

inline constexpr std::size_t kParallelCandidateCount=std::size(detail::parallel_candidates);
inline constexpr std::size_t kGroupedCandidateCount=std::size(detail::grouped_candidates);
inline constexpr std::size_t kLoraCandidateCount=std::size(detail::lora_candidates);
inline int parallel_kernel_ordinal(WeightType type,bool acc32,int nt,int kt) {
  const int id=detail::candidate_ordinal(type,acc32,nt,kt,Family::Parallel);
  if(id<0) throw std::invalid_argument("No installed ParallelLinear candidate");
  return id;
}
inline int grouped_kernel_ordinal(WeightType type,bool acc32,int nt,int kt) {
  const int id=detail::candidate_ordinal(type,acc32,nt,kt,Family::Grouped);
  if(id<0) throw std::invalid_argument("No installed GroupedParallelLinear candidate");
  return id;
}
inline int parallel_kernel_ordinal(WeightType type,bool acc32,const TilePick& tile) {
  const int id=parallel_kernel_ordinal(type,acc32,tile.n_tile,tile.k_tile);
  if(detail::parallel_candidates[id].tile.m_tile!=tile.m_tile)
    throw std::invalid_argument("M_TILE differs from installed ParallelLinear candidate");
  return id;
}
inline int grouped_kernel_ordinal(WeightType type,bool acc32,const TilePick& tile) {
  const int id=grouped_kernel_ordinal(type,acc32,tile.n_tile,tile.k_tile);
  if(detail::grouped_candidates[id].tile.m_tile!=tile.m_tile)
    throw std::invalid_argument("M_TILE differs from installed GroupedParallelLinear candidate");
  return id;
}
inline const char* parallel_kernel_name(WeightType type,bool acc32,const TilePick& tile) {
  return detail::parallel_candidates[parallel_kernel_ordinal(type,acc32,tile)].name;
}
inline const char* grouped_kernel_name(WeightType type,bool acc32,const TilePick& tile) {
  return detail::grouped_candidates[grouped_kernel_ordinal(type,acc32,tile)].name;
}

// Inputs are physical per-core dimensions. Explicit N alone selects K128.
// BF16 / FP8 and G64/G128 are supported through unmeasured explicit tiles.
inline TilePick select_tile(int m,int64_t n,int k,WeightType type,bool acc32,
                            Family family=Family::Parallel,int group_size=-1,
                            uint16_t x_dtype=2,int explicit_n=0,int explicit_k=0) {
  const bool w4=is_w4(type);
  if(group_size==-1) group_size=w4 ? 32 : 0;
  const int ka=w4 ? std::max(32,group_size) : variant_index(type)==0 ? 16 : 32;
  if(m<=0 || m>65535 || n<=0 || n>UINT32_MAX || k<=0 || k>65535 ||
     n%(w4 ? 32 : 16) || ka<=0 || k%ka ||
     (w4 ? (group_size!=32 && group_size!=64 && group_size!=128) : group_size!=0) ||
     (x_dtype!=2 && x_dtype!=3))
    throw std::invalid_argument("Pass aligned physical ParallelLinear dimensions and supported precision");
  TilePick result;
  if(explicit_n || explicit_k) {
    if(!explicit_n) throw std::invalid_argument("Explicit K tile requires N tile");
    result=make_tile(type,acc32,explicit_n,explicit_k ? explicit_k : 128,family);
  } else if(type==WeightType::Bf16 || type==WeightType::Fp8E4M3 || x_dtype!=2 ||
            (w4 && group_size!=32)) {
    result=make_tile(type,acc32,family==Family::Parallel ? 128 : 64,128,family);
    result.source=TileSource::UnmeasuredFallback;
  } else {
    const uint8_t cache_id=static_cast<uint8_t>(static_cast<int>(type)*8+
                         static_cast<int>(family)*2+acc32);
    auto& cache=detail::selection_cache();
    if(!cache.lookup(cache_id,m,n,k,result)) {
      result=detail::measured_select(m,n,k,type,acc32,family);
      cache.store(cache_id,m,n,k,result);
    }
  }
  if((w4 && result.k_tile%group_size) || (n+result.n_tile-1)/result.n_tile>65535 ||
     (family==Family::Grouped && (n+15)/16>65535))
    throw std::invalid_argument("Selected tile exceeds group or grid ABI");
  return result;
}
inline TilePick select_grouped_tile(int64_t n,int k,WeightType type,bool acc32,
                                    int group_size=-1,uint16_t x_dtype=2,int nt=0,int kt=0) {
  return select_tile(1,n,k,type,acc32,Family::Grouped,group_size,x_dtype,nt,kt);
}
inline TilePick select_tile_acc16(int m,int n,int k,bool fp16) {
  return select_tile(m,n,k,fp16 ? WeightType::Fp16 : WeightType::Int8,false);
}
inline TilePick select_tile_acc32(int m,int n,int k,bool fp16) {
  return select_tile(m,n,k,fp16 ? WeightType::Fp16 : WeightType::Int8,true);
}
inline TilePick select_tile_int4(int m,int n,int k) { return select_tile(m,n,k,WeightType::Int4,false); }
inline TilePick select_tile_int4_acc32(int m,int n,int k) { return select_tile(m,n,k,WeightType::Int4,true); }
inline TilePick select_tile_uint4(int m,int n,int k) { return select_tile(m,n,k,WeightType::Uint4,false); }
inline TilePick select_tile_uint4_acc32(int m,int n,int k) { return select_tile(m,n,k,WeightType::Uint4,true); }

inline int mtile_w16(int n) { return make_tile(WeightType::Fp16,false,n,128).m_tile; }
inline int mtile_w8a16(int n) { return make_tile(WeightType::Int8,false,n,128).m_tile; }
inline int mtile_fp16_acc32(int n) { return make_tile(WeightType::Fp16,true,n,128).m_tile; }
inline int mtile_w8a16_acc32(int n) { return make_tile(WeightType::Int8,true,n,128).m_tile; }
inline int mtile_int4(int n) { return make_tile(WeightType::Int4,false,n,128).m_tile; }
inline int mtile_int4_acc32(int n) { return make_tile(WeightType::Int4,true,n,128).m_tile; }
inline std::string autotile_kernel_name(bool fp16,int m,int n,int k) {
  return parallel_kernel_name(fp16 ? WeightType::Fp16 : WeightType::Int8,false,{n,m,k});
}
inline std::string autotile_kernel_name_acc32(bool fp16,int m,int n,int k) {
  return parallel_kernel_name(fp16 ? WeightType::Fp16 : WeightType::Int8,true,{n,m,k});
}
inline std::string autotile_kernel_name_int4(int m,int n,int k) {
  return parallel_kernel_name(WeightType::Int4,false,{n,m,k});
}
inline std::string autotile_kernel_name_int4_acc32(int m,int n,int k) {
  return parallel_kernel_name(WeightType::Int4,true,{n,m,k});
}
inline std::string autotile_kernel_name_uint4(int m,int n,int k) {
  return parallel_kernel_name(WeightType::Uint4,false,{n,m,k});
}
inline std::string autotile_kernel_name_uint4_acc32(int m,int n,int k) {
  return parallel_kernel_name(WeightType::Uint4,true,{n,m,k});
}
inline std::vector<std::string> autotile_kernel_names() {
  std::vector<std::string> names;
  names.reserve(kParallelCandidateCount);
  for(const auto& c:detail::parallel_candidates) names.emplace_back(c.name);
  return names;
}
inline std::vector<std::string> grouped_autotile_kernel_names() {
  std::vector<std::string> names;
  names.reserve(kGroupedCandidateCount);
  for(const auto& c:detail::grouped_candidates) names.emplace_back(c.name);
  return names;
}
inline constexpr bool supports_w8a16_row_weight_reuse(
    int64_t m,int64_t n,int64_t k,int part,int cores) noexcept {
  return m==512 && n==1024 && k==2048 && part==0 && cores==8;
}

#include "rpu_linear_nvfp4_tiling.h"

inline std::string autotile_kernel_name_nvfp4_acc16(int m,int n) {
  return "parallel_linear_wnvfp4a16_acc16_m"+std::to_string(m)+"n"+std::to_string(n)+"k128";
}
inline std::string autotile_kernel_name_nvfp4_acc32(int m,int n) {
  return "parallel_linear_wnvfp4a16_acc32_m"+std::to_string(m)+"n"+std::to_string(n)+"k128";
}

// dtype_mode is the hardware WDType (FP16=2, BF16=3, I8=1, FP8=5).
// Existing initializer order is retained; every writer emits the new ABI.
struct ParallelLinearRegisterArgs {
  uint32_t input_spm_addr;
  uint64_t weight_dev_addr;
  uint32_t output_spm_addr,bias_spm_addr,local_m,local_n,local_k;
  uint16_t has_bias,num_cores;
  uint32_t input_m_byte_step,output_m_byte_step;
  uint64_t scale_dev_addr;
  uint16_t partition,dtype_mode;
  uint16_t x_dtype=2;
};
enum class FusedRegister : int {
  SecondWeight=32,SecondScale=34,SecondOutput=36,SecondBias=38,
  ThirdWeight=40,ThirdScale=42,ThirdOutput=44,ThirdBias=46,Extra=48
};
template <typename Kernel>
inline void set_u32_reg_pair(Kernel& kernel,int slot,uint64_t value) {
  if(value>UINT32_MAX) throw std::invalid_argument("ParallelLinear uint32 register overflow");
  kernel.set_regs(slot,static_cast<uint16_t>(value));
  kernel.set_regs(slot+1,static_cast<uint16_t>(value>>16));
}
inline void validate_linear_registers(const ParallelLinearRegisterArgs& a) {
  if(a.local_m>65535 || a.local_k==0 || a.local_k>65535 || a.local_k%16 ||
     a.local_n==0 || a.num_cores<1 || a.num_cores>8 || a.partition>1 ||
     a.has_bias>1 || (a.x_dtype!=2 && a.x_dtype!=3) ||
     (a.dtype_mode!=1 && a.dtype_mode!=2 && a.dtype_mode!=3 &&
      a.dtype_mode!=5 && a.dtype_mode!=8 && a.dtype_mode!=9) ||
     a.weight_dev_addr%256 || a.scale_dev_addr%256 ||
     a.weight_dev_addr>>40 || a.scale_dev_addr>>40)
    throw std::invalid_argument("Invalid ParallelLinear register arguments");
}
template <typename Kernel>
inline void set_linear_address_regs(Kernel& kernel,const ParallelLinearRegisterArgs& a) {
  set_u32_reg_pair(kernel,0,a.input_spm_addr);
  set_u32_reg_pair(kernel,2,a.weight_dev_addr>>8);
  set_u32_reg_pair(kernel,4,a.output_spm_addr);
  set_u32_reg_pair(kernel,6,a.bias_spm_addr);
  set_u32_reg_pair(kernel,10,a.local_n);
  kernel.set_regs(12,static_cast<uint16_t>(a.local_k));
  kernel.set_regs(13,static_cast<uint16_t>(a.local_k/16));
  kernel.set_regs(14,a.has_bias);
  kernel.set_regs(15,a.num_cores);
  set_u32_reg_pair(kernel,16,a.input_m_byte_step);
  set_u32_reg_pair(kernel,18,a.output_m_byte_step);
  set_u32_reg_pair(kernel,20,a.scale_dev_addr>>8);
}
template <typename Kernel>
inline void set_parallel_linear_regs(Kernel& kernel,const ParallelLinearRegisterArgs& a) {
  validate_linear_registers(a);
  set_linear_address_regs(kernel,a);
  kernel.set_regs(8,a.partition);
  kernel.set_regs(9,static_cast<uint16_t>(a.local_m));
  set_u32_reg_pair(kernel,22,0);
  set_u32_reg_pair(kernel,24,0);
  kernel.set_regs(26,a.x_dtype);
  kernel.set_regs(27,a.dtype_mode);
  kernel.set_regs(28,uint16_t{0});
  kernel.set_regs(29,uint16_t{0});
}
template <typename Kernel>
inline void set_parallel_linear_fused_regs(Kernel& kernel,const ParallelLinearRegisterArgs& a) {
  set_parallel_linear_regs(kernel,a);
}
struct ParallelLinearW4GroupRegisterArgs {
  ParallelLinearRegisterArgs linear;
  uint16_t group_size,local_groups;
  uint32_t scale_group_block_byte_step;
  uint64_t negative_zero_dev_addr=0;
  bool is_uint4=false;
};
template <typename Kernel>
inline void set_parallel_linear_w4_group_regs(Kernel& kernel,const ParallelLinearW4GroupRegisterArgs& a) {
  if((a.group_size!=32 && a.group_size!=64 && a.group_size!=128) ||
     a.linear.local_k%a.group_size || a.linear.local_k/a.group_size!=a.local_groups ||
     a.linear.local_n%16 || a.negative_zero_dev_addr%256 || a.negative_zero_dev_addr>>40)
    throw std::invalid_argument("Invalid N32K32 W4 group register arguments");
  auto linear=a.linear;
  linear.dtype_mode=a.is_uint4 ? 8 : 9;
  set_parallel_linear_regs(kernel,linear);
  set_u32_reg_pair(kernel,22,a.is_uint4 ? a.negative_zero_dev_addr>>8 : 0);
  set_u32_reg_pair(kernel,24,a.scale_group_block_byte_step);
  kernel.set_regs(28,a.group_size);
  kernel.set_regs(29,a.local_groups);
}
struct ParallelLinearInt4GroupRegisterArgs {
  ParallelLinearRegisterArgs linear;
  uint16_t group_size,local_groups;
  uint32_t scale_group_block_byte_step;
};
template <typename Kernel>
inline void set_parallel_linear_int4_group_regs(Kernel& kernel,const ParallelLinearInt4GroupRegisterArgs& a) {
  set_parallel_linear_w4_group_regs(kernel,{a.linear,a.group_size,a.local_groups,a.scale_group_block_byte_step});
}
struct GroupedParallelLinearRegisterArgs {
  ParallelLinearW4GroupRegisterArgs linear;
  uint16_t groups;
  uint32_t m_sizes_spm_addr;
  uint64_t weight_group_byte_step,scale_group_byte_step,zero_group_byte_step=0;
};
template <typename Kernel>
inline void set_grouped_parallel_linear_regs(Kernel& kernel,const GroupedParallelLinearRegisterArgs& a) {
  if(!a.groups || a.groups>rpu_grouped_device_m::kMaxGroups ||
     a.weight_group_byte_step%256 || a.scale_group_byte_step%256 ||
     a.zero_group_byte_step%256 || a.weight_group_byte_step>=(UINT64_C(1)<<32) ||
     a.scale_group_byte_step>=(UINT64_C(1)<<32) || a.zero_group_byte_step>=(UINT64_C(1)<<32))
    throw std::invalid_argument("Invalid GroupedParallelLinear group stride");
  if(a.linear.group_size) set_parallel_linear_w4_group_regs(kernel,a.linear);
  else set_parallel_linear_regs(kernel,a.linear.linear);
  kernel.set_regs(30,a.groups);
  kernel.set_regs(31,uint16_t{0});
  set_u32_reg_pair(kernel,32,a.m_sizes_spm_addr);
  set_u32_reg_pair(kernel,34,a.weight_group_byte_step>>8);
  set_u32_reg_pair(kernel,36,a.scale_group_byte_step>>8);
  set_u32_reg_pair(kernel,38,a.zero_group_byte_step>>8);
}

// NVFP4 is a separate, unchanged hardware ABI, not a compatibility branch of PL.
struct ParallelLinearNvfp4RegisterArgs {
  ParallelLinearRegisterArgs linear;
  uint32_t scale_kpair_byte_step,tensor_scale_spm_addr;
  uint16_t layer_id;
};
template <typename Kernel>
inline void set_parallel_linear_nvfp4_regs(Kernel& kernel,const ParallelLinearNvfp4RegisterArgs& a) {
  auto validated=a.linear; validated.dtype_mode=2;
  validate_linear_registers(validated);
  if(a.linear.local_k%64) throw std::invalid_argument("NVFP4 localK must align to 64");
  set_linear_address_regs(kernel,a.linear);
  kernel.set_regs(8,static_cast<uint16_t>(a.linear.local_m));
  kernel.set_regs(9,uint16_t{0});
  kernel.set_regs(13,static_cast<uint16_t>(a.linear.local_k/64));
  set_u32_reg_pair(kernel,22,a.scale_kpair_byte_step);
  set_u32_reg_pair(kernel,24,a.tensor_scale_spm_addr);
  kernel.set_regs(26,a.linear.partition);
  kernel.set_regs(27,a.layer_id);
}
} // namespace rpu_pl_tiling
