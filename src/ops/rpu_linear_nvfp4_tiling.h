// NVFP4 retains its independent K128 candidate family and measured policy.
// Included inside rpu_pl_tiling; this is not the ordinary W4 interface.
namespace nvfp4_detail {
struct PLEntry { int M, n, k, tile; };
constexpr int kFallbackNeighbors = 5;
constexpr double kFallbackDistanceEpsilon = 0.01;
constexpr std::array<int, 7> kTileGrid{32,48,64,80,96,112,128};
template <std::size_t N> constexpr std::size_t table_size(const PLEntry (&)[N]) { return N; }
static constexpr PLEntry PL_EXACT_NVFP4_ACC32[] = {
    {1,32,2048,112}, {1,32,4096,80}, {1,64,2048,32}, {1,64,2560,32}, {1,64,3584,32}, {1,128,1024,32},
    {1,128,2048,32}, {1,128,2304,32}, {1,128,2560,32}, {1,128,4096,32}, {1,128,5120,32}, {1,256,1024,64},
    {1,256,2048,64}, {1,256,2304,64}, {1,256,2560,64}, {1,320,2048,128}, {1,384,1024,128}, {1,384,2048,128},
    {1,384,2560,128}, {1,448,3584,128}, {1,512,1024,64}, {1,512,2048,64}, {1,512,2304,64}, {1,512,2560,64},
    {1,512,4096,64}, {1,576,3584,64}, {1,576,4096,64}, {1,640,2048,64}, {1,640,5120,64}, {1,768,2048,128},
    {1,768,2560,128}, {1,768,4096,128}, {1,896,5120,128}, {1,1024,256,96}, {1,1024,384,96}, {1,1024,2048,64},
    {1,1024,5120,64}, {1,1152,2304,128}, {1,1152,2560,128}, {1,1216,2560,128}, {1,1280,2560,128}, {1,1280,5120,128},
    {1,1376,2048,128}, {1,1536,4096,64}, {1,2048,256,64}, {1,2048,512,64}, {1,2048,768,64}, {1,2048,1024,64},
    {1,2048,2048,64}, {1,2048,4096,64}, {1,2176,5120,128}, {1,2304,256,64}, {1,2304,1152,128}, {1,2368,3584,128},
    {1,2560,256,64}, {1,2560,512,128}, {1,2560,1152,64}, {1,2560,1216,64}, {1,2560,1280,64}, {1,3200,5120,128},
    {1,3584,448,64}, {1,3584,2368,128}, {1,4096,512,64}, {1,4096,1536,64}, {1,4096,2048,128}, {1,5120,640,64},
    {1,5120,1024,128}, {1,5120,2176,128}, {1,5120,3200,128}, {1,16032,2048,128}, {1,18992,1024,128}, {1,18992,2048,128},
    {1,18992,2560,128}, {1,18992,4096,128}, {1,18992,5120,128}, {1,19008,2048,128}, {1,19008,3584,128}, {1,31040,2560,128},
    {1,32000,2304,128}, {1,32144,2048,128}, {31,32,1024,128}, {31,128,1024,32}, {31,256,1024,64}, {31,320,1024,112},
    {31,384,1024,128}, {31,512,1024,64}, {31,1024,256,64}, {31,1024,384,64}, {31,1024,512,96}, {32,32,1024,80},
    {32,128,1024,32}, {32,256,1024,64}, {32,320,1024,112}, {32,384,1024,128}, {32,512,1024,64}, {32,1024,256,128},
    {32,1024,384,64}, {32,1024,512,64}, {50,32,1024,80}, {50,128,1024,32}, {50,256,1024,64}, {50,320,1024,96},
    {50,384,1024,128}, {50,512,1024,48}, {50,1024,256,64}, {50,1024,384,64}, {50,1024,512,64}, {72,512,4096,64},
    {72,576,4608,128}, {72,640,5120,64}, {72,2048,512,64}, {72,2048,640,64}, {72,4096,576,64}, {180,512,4096,80},
    {180,576,4608,96}, {180,640,5120,96}, {180,2048,512,96}, {180,2048,640,96}, {180,4096,576,96}, {192,512,4096,80},
    {192,576,4608,96}, {192,640,5120,96}, {192,2048,512,112}, {192,2048,640,96}, {192,4096,576,96}, {256,32,2048,128},
    {256,32,4096,128}, {256,64,2048,32}, {256,64,2560,32}, {256,64,3584,32}, {256,128,1024,48}, {256,128,2048,48},
    {256,128,2304,48}, {256,128,2560,48}, {256,128,4096,48}, {256,128,5120,48}, {256,256,1024,64}, {256,256,2048,64},
    {256,256,2304,64}, {256,256,2560,64}, {256,320,2048,80}, {256,384,1024,80}, {256,384,2048,64}, {256,384,2560,64},
    {256,448,3584,80}, {256,512,1024,80}, {256,512,2048,80}, {256,512,2304,80}, {256,512,2560,80}, {256,512,4096,80},
    {256,576,3584,80}, {256,576,4096,80}, {256,640,2048,80}, {256,640,5120,80}, {256,768,2048,80}, {256,768,2560,80},
    {256,768,4096,80}, {256,896,5120,80}, {256,1024,256,80}, {256,1024,384,80}, {256,1024,2048,80}, {256,1024,5120,80},
    {256,1152,2304,80}, {256,1152,2560,80}, {256,1216,2560,80}, {256,1280,2560,80}, {256,1280,5120,80}, {256,1376,2048,80},
    {256,1536,4096,80}, {256,2048,256,80}, {256,2048,512,80}, {256,2048,768,80}, {256,2048,1024,80}, {256,2048,2048,80},
    {256,2048,4096,80}, {256,2176,5120,80}, {256,2304,256,80}, {256,2304,1152,80}, {256,2368,3584,80}, {256,2560,256,80},
    {256,2560,512,80}, {256,2560,1152,64}, {256,2560,1216,64}, {256,2560,1280,80}, {256,3200,5120,80}, {256,3584,448,80},
    {256,3584,2368,80}, {256,4096,512,80}, {256,4096,1536,80}, {256,4096,2048,80}, {256,5120,640,80}, {256,5120,1024,80},
    {256,5120,2176,64}, {256,5120,3200,64}, {288,128,1024,48}, {288,128,1536,48}, {288,128,4096,48}, {288,144,1152,48},
    {288,144,1536,64}, {288,160,1280,64}, {288,256,1024,64}, {288,384,1024,64}, {288,432,1152,64}, {288,432,1280,64},
    {288,480,1280,64}, {288,512,1024,64}, {288,512,4096,64}, {288,544,1152,64}, {288,576,4608,64}, {288,640,5120,64},
    {288,1024,128,64}, {288,1024,256,64}, {288,1024,512,64}, {288,1152,192,64}, {288,1152,576,64}, {288,1280,192,64},
    {288,1280,448,64}, {288,1536,4096,64}, {288,2048,512,64}, {288,2048,640,64}, {288,4096,512,64}, {288,4096,576,64},
    {288,5120,2176,64}, {512,128,1024,64}, {512,128,4096,64}, {512,256,1024,64}, {512,512,4096,80}, {512,1024,256,80},
    {512,1536,4096,80}, {512,4096,512,80}, {512,5120,2176,80}, {720,128,1024,64}, {720,128,1536,64}, {720,144,1152,48},
    {720,144,1536,80}, {720,160,1280,80}, {720,384,1024,80}, {720,432,1152,48}, {720,432,1280,96}, {720,480,1280,96},
    {720,512,1024,96}, {720,544,1152,48}, {720,1024,128,96}, {720,1024,512,96}, {720,1152,192,96}, {720,1152,576,48},
    {720,1280,192,96}, {720,1280,448,48}, {768,128,1024,64}, {768,128,1536,64}, {768,128,4096,64}, {768,144,1152,48},
    {768,144,1536,80}, {768,160,1280,80}, {768,256,1024,96}, {768,384,1024,80}, {768,432,1152,48}, {768,432,1280,80},
    {768,480,1280,80}, {768,512,1024,80}, {768,512,4096,80}, {768,544,1152,48}, {768,1024,128,80}, {768,1024,256,80},
    {768,1024,512,80}, {768,1152,192,80}, {768,1152,576,48}, {768,1280,192,80}, {768,1280,448,80}, {768,4096,512,80},
    {1152,128,1024,64}, {1152,128,1536,64}, {1152,144,1152,80}, {1152,144,1536,80}, {1152,160,1280,80}, {1152,384,1024,96},
    {1152,432,1152,64}, {1152,432,1280,96}, {1152,480,1280,96}, {1152,512,1024,96}, {1152,544,1152,64}, {1152,1024,128,96},
    {1152,1024,512,96}, {1152,1152,192,96}, {1152,1152,576,48}, {1152,1280,192,96}, {1152,1280,448,64},
};
static_assert(table_size(PL_EXACT_NVFP4_ACC32) == 287, "EXACT_TILE_TABLE_NVFP4_ACC32 count drifted");

inline int measured_neighbor_tile(const PLEntry* table, std::size_t count,
                                  int m, int local_n, int local_k,
                                  int (*m_tile)(int)) {
  if (m <= 0 || local_n <= 0 || local_k <= 0) {
    throw std::invalid_argument("ParallelLinear dimensions must be positive");
  }

  std::array<int, kTileGrid.size()> candidates{};
  std::size_t candidate_count = 0;
  for (int tile : kTileGrid) {
    if (m_tile(tile) >= m) candidates[candidate_count++] = tile;
  }
  if (candidate_count == 0) {
    candidates = kTileGrid;
    candidate_count = candidates.size();
  }

  auto candidate_index = [&](int tile) {
    for (std::size_t i = 0; i < candidate_count; ++i) {
      if (candidates[i] == tile) return static_cast<int>(i);
    }
    return -1;
  };

  std::vector<std::pair<double, int>> neighbours;
  neighbours.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const PLEntry& e = table[i];
    if (candidate_index(e.tile) < 0) continue;
    const double dm = std::log2(static_cast<double>(m) / e.M);
    const double dn = 4.0 * std::log2(static_cast<double>(local_n) / e.n);
    const double dk = 0.5 * std::log2(static_cast<double>(local_k) / e.k);
    neighbours.emplace_back(dm * dm + dn * dn + dk * dk, e.tile);
  }
  std::stable_sort(neighbours.begin(), neighbours.end(),
                   [](const auto& lhs, const auto& rhs) {
                     return lhs.first < rhs.first;
                   });

  auto physical_key = [&](std::size_t i) {
    const int tile = candidates[i];
    return std::make_tuple(ceil_div_positive(m, m_tile(tile)),
                           ceil_div_positive(local_n, tile) * tile, tile);
  };
  if (neighbours.empty()) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < candidate_count; ++i) {
      if (physical_key(i) < physical_key(best)) best = i;
    }
    return candidates[best];
  }

  std::array<double, kTileGrid.size()> votes{};
  const std::size_t vote_count =
      std::min<std::size_t>(kFallbackNeighbors, neighbours.size());
  for (std::size_t i = 0; i < vote_count; ++i) {
    const int index = candidate_index(neighbours[i].second);
    votes[static_cast<std::size_t>(index)] +=
        1.0 / (neighbours[i].first + kFallbackDistanceEpsilon);
  }

  auto ranked_key = [&](std::size_t i) {
    return std::tuple_cat(std::make_tuple(-votes[i]), physical_key(i));
  };
  std::size_t best = 0;
  for (std::size_t i = 1; i < candidate_count; ++i) {
    if (ranked_key(i) < ranked_key(best)) best = i;
  }
  return candidates[best];
}

inline int mtile_nvfp4_acc32(int n) {
  switch (n) {
    case 128: return 176;
    case 112: return 208;
    case 96: return 240;
    case 80: return 272;
    case 64: return 320;
    case 48: return 384;
    case 32: return 480;
    default: throw std::invalid_argument("unsupported NVFP4 n_tile");
  }
}


} // namespace nvfp4_detail
inline int mtile_nvfp4_acc32(int n) { return nvfp4_detail::mtile_nvfp4_acc32(n); }
inline TilePick select_tile_nvfp4_acc32(int m, int n, int k) {
  if (m <= 0 || n <= 0 || k <= 0 || m > 65535 || k > 65535)
    throw std::invalid_argument("NVFP4 dimensions exceed the kernel ABI");
  auto& cache = detail::selection_cache();
  TilePick result;
  if (cache.lookup(255,m,n,k,result)) return result;
  int nt=0;
  using namespace nvfp4_detail;
  const auto* first=std::begin(PL_EXACT_NVFP4_ACC32);
  const auto* end=std::end(PL_EXACT_NVFP4_ACC32);
  const auto key=std::make_tuple(m,n,k);
  const auto* found=std::lower_bound(first,end,key,[](const PLEntry& e,const auto& q) {
    return std::make_tuple(e.M,e.n,e.k)<q;
  });
  if(found!=end && std::make_tuple(found->M,found->n,found->k)==key) nt=found->tile;
  if(!nt) nt=measured_neighbor_tile(first,table_size(PL_EXACT_NVFP4_ACC32),m,n,k,nvfp4_detail::mtile_nvfp4_acc32);
  result={nt,nvfp4_detail::mtile_nvfp4_acc32(nt),128};
  cache.store(255,m,n,k,result);
  return result;
}
// ACC16 currently shares the ACC32 geometry and bootstrap heuristic only.
inline TilePick select_tile_nvfp4_acc16(int m,int n,int k) { return select_tile_nvfp4_acc32(m,n,k); }
