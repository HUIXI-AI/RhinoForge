"""Execute the production M672 leaf packet and its failure boundaries on host."""
from pathlib import Path

from test_wall_base_paired_kv import block, compile_run


ROOT = Path(__file__).resolve().parents[1]


def test_actual_vision_owner_ring_norm_packet_ranges_and_failures(tmp_path):
    runtime = (ROOT / 'src/core/rpu_runtime_extras.cpp').read_text()
    launch = '\n'.join(block(runtime, marker) for marker in (
        'static void launch_wall_base_owner_ring_norm(',
        'void rpu_launch_wall_base_vision_owner_ring_norm_m672_kernel(',
        'void rpu_launch_wall_base_vision_down_owner_ring_norm_m672_kernel(',
    ))
    cache = (ROOT / 'src/core/rpu_kernel_cache.h').read_text()
    kernel_ids = block(cache, 'enum class KernelId : uint16_t') + ';'
    kernel_names = block(cache, 'static constexpr const char* KERNEL_ID_NAMES[]') + ';'
    optional = block((ROOT / 'src/core/rpu_kernel_cache.inc').read_text(),
                     'static bool is_optional_kernel(')
    scm = block((ROOT / 'src/core/rpu_helpers.h').read_text(),
                'inline void rpu_set_legacy_scm_u16_checked(')
    source = r'''
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <c10/util/Half.h>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
OPTIONAL
KERNEL_IDS
KERNEL_NAMES
KernelId active_id = KernelId::WALL_BASE_VISION_OWNER_RING_NORM_M672;
size_t variant_index(KernelId id) {
    assert(id == KernelId::WALL_BASE_VISION_OWNER_RING_NORM_M672 ||
           id == KernelId::WALL_BASE_VISION_DOWN_OWNER_RING_NORM_M672);
    return id == KernelId::WALL_BASE_VISION_DOWN_OWNER_RING_NORM_M672;
}
enum class GraphKernelNoDdrProof {RingAllReduceSpmResidual};
struct Kernel {
    std::array<uint16_t, 4352> regs{};
    std::vector<uint32_t> counter_writes;
    int reject = -1;
    void reset_regs() { regs.fill(0xffff); counter_writes.clear(); }
    uint32_t set_regs(uint32_t reg, uint16_t value) {
        if (int(reg) == reject) return 5;
        regs.at(reg) = value;
        return 0;
    }
    uint32_t set_regs(uint32_t reg, uint32_t value) {
        if (int(reg) == reject) return 5;
        assert(reg % 2 == 0 && reg >= 128 && reg < 256);
        regs.at(reg) = uint16_t(value);
        regs.at(reg + 1) = uint16_t(value >> 16);
        counter_writes.push_back(reg);
        return 0;
    }
    uint32_t set_regs(uint32_t, int32_t) = delete;
} kernel;
namespace rhino_lkn { using Kernel_t = Kernel; constexpr int kKernelRegOk = 0; }
using rhino_lkn::kKernelRegOk;
constexpr int SCM_REG_OFFSET = 4096;
SCM
struct KernelCache {
    std::array<bool, 2> loaded{true, true}, kernel_available{true, true};
    static auto& instance() { static KernelCache c; return c; }
    bool has_loaded(KernelId id) const { return loaded[variant_index(id)]; }
};
int staged = 0, queued = 0;
struct RpuKernelGraph {
    bool census = false, reject = false;
    static auto& active() { static RpuKernelGraph g; return g; }
    bool kernel_register_census_active() const { return census; }
    void stage_kernel_no_ddr(KernelId id, GraphKernelNoDdrProof proof) {
        assert(id == active_id);
        assert(census && proof == GraphKernelNoDdrProof::RingAllReduceSpmResidual);
        if (reject) throw std::runtime_error("proof denied");
        ++staged;
    }
};
struct Queue {
    bool broadcast = false;
    void set_broadcast_mode(bool value) { broadcast = value; }
    void enqueu_kernel(Kernel& k, std::vector<uint16_t> grid, std::vector<uint8_t> cores) {
        assert(broadcast && grid == std::vector<uint16_t>({8, 1, 1}));
        assert(cores == std::vector<uint8_t>({0, 1, 2, 3, 4, 5, 6, 7}));
        assert(k.counter_writes.size() == 64);
        for (int i = 0; i < 64; ++i) {
            const uint32_t reg = 128 + 2 * i;
            assert(k.counter_writes[i] == reg && k.regs[reg] == 0 && k.regs[reg + 1] == 0);
            // Leave live epochs behind: the next packet must clear them.
            k.regs[reg] = 19; k.regs[reg + 1] = 3;
        }
        ++queued;
    }
} queue;
Kernel* get_kernel(KernelId id) {
    assert(id == active_id);
    return KernelCache::instance().kernel_available[variant_index(id)] ? &kernel : nullptr;
}
Queue* get_queue(int cores) { assert(cores == 8); return &queue; }
#define GET_KERNEL(id) get_kernel(id)
#define GET_QUEUE(n) get_queue(n)
struct SpmAllocator {
    static constexpr uint32_t SPM_USABLE = 8387584;
    bool ready = true;
    uint32_t base = 0x20000000;
    bool is_initialized() const { return ready; }
    uint32_t addr(int core, uint32_t offset) const { return base + core * 0x1000000u + offset; }
} spm;
#define SPM_ALLOC spm
LAUNCH
uint32_t u32(int reg) { return kernel.regs[reg] | (uint32_t(kernel.regs[reg + 1]) << 16); }
uint16_t sldc(int index) { return kernel.regs[(4096 + index) ^ 1]; }
uint32_t scm_pair(int field, int core) {
    return sldc(field + core) | (uint32_t(sldc(field + 8 + core)) << 16);
}
int main() {
    assert(is_optional_kernel("wall_base_vision_owner_ring_norm_m672_h1280"));
    assert(is_optional_kernel("wall_base_vision_down_owner_ring_norm_m672_h1280"));
    assert(!is_optional_kernel("llama_insert_kv_cache_multiwarp_v16"));
    assert(std::string(KERNEL_ID_NAMES[static_cast<size_t>(KernelId::WALL_BASE_VISION_OWNER_RING_NORM_M672)]) ==
           "wall_base_vision_owner_ring_norm_m672_h1280");
    assert(std::string(KERNEL_ID_NAMES[static_cast<size_t>(KernelId::WALL_BASE_VISION_DOWN_OWNER_RING_NORM_M672)]) ==
           "wall_base_vision_down_owner_ring_norm_m672_h1280");
    static_assert(KernelId::WALL_BASE_VISION_OWNER_RING_NORM_M672 !=
                  KernelId::WALL_BASE_VISION_DOWN_OWNER_RING_NORM_M672);
    constexpr uint32_t base = 0x20000000, full = 1720320, gamma_bytes = 2560;
    constexpr uint32_t partial = base + 0x1000, residual = partial + full + 256;
    constexpr uint32_t raw = residual + full + 256, gamma = raw + full + 256;
    using Slabs = std::array<uint32_t, 4>;
    const Slabs exact{partial, residual, raw, gamma};
    std::array<uint16_t, 4352> o_packet{};
    for (bool residual_norm : {false, true}) {
    active_id = residual_norm ? KernelId::WALL_BASE_VISION_DOWN_OWNER_RING_NORM_M672
                              : KernelId::WALL_BASE_VISION_OWNER_RING_NORM_M672;
    const auto invoke = residual_norm ? rpu_launch_wall_base_vision_down_owner_ring_norm_m672_kernel
                                      : rpu_launch_wall_base_vision_owner_ring_norm_m672_kernel;
    const uint32_t normalized = residual_norm ? residual : partial;
    const size_t selected = variant_index(active_id), other = 1 - selected;
    auto run = [&](Slabs slabs, bool expected, double eps = 1e-6) {
        const int old_queued = queued;
        bool accepted = true;
        try {
            invoke(slabs[0], slabs[1], slabs[2], slabs[3], eps);
        } catch (const std::exception&) { accepted = false; }
        assert(accepted == expected && queued == old_queued + int(expected));
    };
    for (bool census : {false, true}) {
        RpuKernelGraph::active().census = census;
        for (int replay = 0; replay < 2; ++replay) {
            const int old_staged = staged;
            queue.broadcast = false;
            run(exact, true);
            assert(staged == old_staged + int(census));
            assert(kernel.regs[0] == 8 && kernel.regs[1] == 2 && u32(2) == 63488);
            assert(u32(4) == residual - base && kernel.regs[6] == 0);
            assert(kernel.regs[8] == 1280 && kernel.regs[9] == c10::Half(1.0f / 1280).x);
            assert(kernel.regs[10] == 2560 && kernel.regs[11] == 12);
            assert(u32(12) == raw && u32(14) == normalized && u32(16) == gamma);
            assert(kernel.regs[18] == c10::Half(1e-6).x);
            assert(kernel.regs[64] == 8 && kernel.regs[65] == 1 && kernel.regs[66] == 1);
            for (int core = 0; core < 8; ++core) {
                assert(scm_pair(0, core) == core * 0x1000000u + partial - base);
                assert(scm_pair(16, core) == core * 0x1000000u + raw - base);
                assert(scm_pair(32, core) == core * 215040u);
                assert(scm_pair(48, core) == 107520);
                assert(scm_pair(64, core) == core * 0x1000000u + normalized - base);
            }
            if (!residual_norm) {
                o_packet = kernel.regs;
            } else {
                // Down changes only the normalizer output and gather peer
                // table. Every other byte of the validated O packet stays.
                auto expected = o_packet;
                expected[14] = uint16_t(residual);
                expected[15] = uint16_t(residual >> 16);
                for (int core = 0; core < 8; ++core) {
                    const uint32_t peer = core * 0x1000000u + residual - base;
                    expected[(4096 + 64 + core) ^ 1] = uint16_t(peer);
                    expected[(4096 + 72 + core) ^ 1] = uint16_t(peer >> 16);
                }
                assert(kernel.regs == expected);
            }
        }
    }
    RpuKernelGraph::active().census = false;
    for (int i = 0; i < 4; ++i) {
        for (uint32_t address : {uint32_t{0}, base - 256, exact[i] + 2,
                                 exact[i] + 32, base + 0x1000000u}) {
            auto bad = exact; bad[i] = address; run(bad, false);
        }
        auto edge = exact;
        edge[i] = base + SpmAllocator::SPM_USABLE - (i == 3 ? gamma_bytes : full);
        run(edge, true);
        edge[i] += 256; run(edge, false);
        for (int j = 0; j < i; ++j) {
            auto bad = exact; bad[i] = bad[j]; run(bad, false);
            bad[i] = bad[j] + 256; run(bad, false);
            bad[i] = bad[j] + full - 256; run(bad, false);
        }
    }
    spm.ready = false; run(exact, false); spm.ready = true;
    for (uint32_t high_base : {0xf8fff000u, 0xfe000000u}) {
        spm.base = high_base;
        Slabs moved{};
        for (int i = 0; i < 4; ++i) moved[i] = exact[i] - base + high_base;
        run(moved, false);
    }
    spm.base = base;
    for (double eps : {0.0, -1.0, 1e-5, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) run(exact, false, eps);
    KernelCache::instance().loaded[selected] = false; run(exact, false);
    KernelCache::instance().loaded[selected] = true;
    KernelCache::instance().loaded[other] = false; run(exact, true);
    KernelCache::instance().loaded[other] = true;
    KernelCache::instance().kernel_available[selected] = false; run(exact, false);
    KernelCache::instance().kernel_available[selected] = true;
    KernelCache::instance().kernel_available[other] = false; run(exact, true);
    KernelCache::instance().kernel_available[other] = true;
    for (int reg : {0, 3, 8, 13, 18, 64, 128, 132, 254, 4096, 4160, 4175}) {
        kernel.reject = reg; run(exact, false); kernel.reject = -1;
    }
    RpuKernelGraph::active().census = true;
    RpuKernelGraph::active().reject = true;
    run(exact, false);
    RpuKernelGraph::active().reject = false;
    }
}
'''
    for key, value in (('OPTIONAL', optional), ('SCM', scm), ('LAUNCH', launch),
                       ('KERNEL_IDS', kernel_ids), ('KERNEL_NAMES', kernel_names)):
        source = source.replace('\n' + key + '\n', '\n' + value + '\n')
    compile_run(tmp_path, 'wall_vision_owner_ring_norm', source)


def test_actual_fp16_w8_owner_routes_and_full_layer_handoffs(tmp_path):
    """Execute real route production/emission; leaf stubs check raw ownership."""
    import re
    from test_wall_base_vision_ragged_routes import _source

    vision = (ROOT / 'src/fused/rpu_qwen25vl_vision_model.cpp').read_text()
    decls = (ROOT / 'src/core/rpu_kernel_decls.h').read_text()
    source = '#include <tuple>\n' + _source()
    source = source.replace('auto& ctx(){return context;}', 'Context& ctx(){return context;}')
    # Reuse the existing actual-admission/COMPLETE fixture, replacing only its
    # deliberately disabled owner stubs with the production methods under test.
    source = source.replace(
        'enum class RpuRmsNormSpmRoute {BASE};',
        decls[decls.index('enum RpuRmsNormCapability'):
              decls.index('enum class RpuMixedRmsNormSpmRoute')]
        + block(decls, 'inline bool rpu_rmsnorm_capability_valid(')
        + block(decls, 'inline RpuRmsNormSpmRoute rpu_resolve_rmsnorm_spm_route('))
    source = source.replace(
        'std::vector<FmbRouteManifestEntry> routes;};',
        'std::vector<FmbRouteManifestEntry> routes; '
        'int64_t logical_length=648,physical_length=672,execution_padding_rows=24;};')
    source = source.replace('bool ready=true;', 'bool ready=true,input_in_spm=false,output_to_spm=true;')
    source = source.replace('const auto& manifest()const{return value;}',
                            'const auto& manifest()const{return value;} '
                            'const auto& physical_manifest()const{return value;}')
    for stub in (
        ' RpuRmsNormSpmRoute small_image_rmsnorm_route(int64_t)const{return RpuRmsNormSpmRoute::BASE;}',
        ' bool small_image_owner_ring_norm_enabled(int64_t)const{return false;}',
        ' bool small_image_down_owner_norm_enabled(int64_t)const{return false;}',
    ):
        assert stub in source
        source = source.replace(stub, '')
    methods = '\n'.join(block(vision, marker) for marker in (
        '    RpuRmsNormSpmRoute small_image_rmsnorm_route(',
        '    RpuRmsNormSpmRoute consume_small_image_rmsnorm_route(',
        '    bool small_image_owner_ring_norm_enabled(',
        '    std::vector<int64_t> small_image_owner_ring_norm_arguments(',
        '    void consume_small_image_owner_ring_norm(',
        '    bool small_image_down_owner_norm_enabled(',
        '    void validate_small_image_down_owner_plan(',
        '    void validate_small_image_down_owner_context(',
        '    std::vector<int64_t> small_image_down_owner_norm_arguments(',
        '    void consume_small_image_down_owner_norm(',
        '    std::vector<int64_t> small_image_gate_up_arguments(',
        '    std::vector<int64_t> vision_row_arguments(',
    ))
    cold = []
    for name in ('small_owner_ring_norm_available_', 'small_down_owner_norm_available_'):
        start = vision.index('        ' + name + ' =')
        cold.append(vision[start:vision.index(';', start) + 1])
    source = source.replace(' void bind(){', ' void bind(){' + '\n'.join(cold))
    source = source.replace(' KernelPayloadAvailability availability;', '''
 bool o_kernel=true,o_program=true,down_kernel=true,down_program=true;
 void* get_kernel(const std::string& s)const{
  return (s=="wall_base_vision_owner_ring_norm_m672_h1280"?o_kernel:down_kernel)?(void*)1:nullptr;}
 void* get_program(const std::string& s)const{
  return (s=="wall_base_vision_owner_ring_norm_m672_h1280"?o_program:down_program)?(void*)1:nullptr;}
 KernelPayloadAvailability availability;''')
    # Extract the actual selected normalization and collective producer branches.
    start = vision.index('                const auto small_norm = small_image_rmsnorm_route(chunk.len);')
    stop = vision.index('                const int64_t row_route', start)
    producer = vision[start:stop]
    producer += block(vision[stop:], '                if (has_pre_boundary_layer)')
    sort_start = vision.index('        std::sort(\n            manifest.routes.begin()')
    sort_routes = vision[sort_start:vision.index('        return manifest;', sort_start)]
    schedule = block(vision, '    int64_t small_image_schedule_flags() const')
    assert 'small_image_owner_ring_norm_enabled' in schedule
    layer = block(vision, '    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override')
    start = layer.index('        const bool down_owner_norm =')
    head = layer[start:layer.index('        // Phase 2:', start)]
    start = layer.index('        // Phase 4:')
    tail = layer[start:layer.index('        if (get_debug_export() && per_layer_debug_buf_.defined())', start)]
    sites = '\n'.join(re.findall(r'constexpr int64_t QWEN25_[A-Z0-9_]+\s*=\s*\d+LL;', vision))
    # _source already declares two stable identities.
    for name in ('QWEN25_SMALL_RAGGED_SDPA_SITE', 'QWEN25_WALL_SORTING_SITE'):
        sites = re.sub(r'constexpr int64_t '+name+r'\s*=\s*\d+LL;', '', sites)
    globals_ = sites + '\nnamespace v3 { enum class FmbLinearRouteSelector { AUTO_TILE=1 }; }\n' + '\n'.join(block(vision, marker)+';' for marker in (
        'enum class Qwen25VisionAllReduceRoute', 'enum class Qwen25VisionLinearRoute')) + r'''
constexpr int QWEN25VL_VISION_MIXED_BOUNDARY_LAYER=17;
int64_t qwen25_vision_linear_route(){return 0;}
int64_t qwen25_vision_ring_route(int64_t m,int64_t n){assert(m==672&&n==1280);return 7;}
enum class ValuOpType {SILU,MUL};
constexpr uint32_t RES=0x1000,INPUT=0x2000,PARTIAL=0x3000,SDPA=0x4000,GATE=0x5000,UP=0x6000;
enum Content {EMPTY,FULL_RAW,OWNER_RAW,FULL_NORM,PARTIALS,ACTIVATION};
std::array<Content,7> contents;
int o_calls=0,down_calls=0,ring_calls=0,norm_calls=0,linear_calls=0,mlp_calls=0;
Content& content(uint32_t a){assert(a%4096==0&&a/4096<contents.size());return contents[a/4096];}
void rpu_launch_rmsnorm_spm_kernel(uint32_t in,uint32_t out,uint32_t gamma,
 int64_t m,int64_t n,double eps,RpuRmsNormSpmRoute route){
 assert(m==672&&n==1280&&eps==1e-6&&gamma>=0x100000&&route==RpuRmsNormSpmRoute::V32_GRID8);
 assert(content(in)==FULL_RAW);content(out)=FULL_NORM;++norm_calls;
}
void rpu_launch_wall_base_vision_owner_ring_norm_m672_kernel(
 uint32_t partial,uint32_t residual,uint32_t raw,uint32_t gamma,double eps){
 assert(partial==PARTIAL&&residual==RES&&raw==INPUT&&eps==1e-6);
 assert(content(partial)==PARTIALS&&(content(residual)==FULL_RAW||content(residual)==OWNER_RAW));
 assert(gamma==0x100000+o_calls*256+128);++o_calls;
 content(raw)=OWNER_RAW;content(partial)=FULL_NORM;
}
void rpu_launch_wall_base_vision_down_owner_ring_norm_m672_kernel(
 uint32_t partial,uint32_t residual,uint32_t raw,uint32_t gamma,double eps){
 assert(partial==PARTIAL&&residual==INPUT&&raw==RES&&eps==1e-6);
 assert(content(partial)==PARTIALS&&content(residual)==OWNER_RAW);
 assert(gamma==0x100000+(down_calls+1)*256);++down_calls;
 content(raw)=OWNER_RAW;content(residual)=FULL_NORM;
}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t p,uint32_t r,uint32_t out,
 int64_t m,int64_t n,int producers,int consumers){
 assert(p==PARTIAL&&m==672&&n==1280&&producers==8&&consumers==8);
 assert(content(p)==PARTIALS&&(content(r)==FULL_RAW||content(r)==OWNER_RAW));
 assert((r==RES&&out==INPUT)||(r==INPUT&&out==RES));content(out)=FULL_RAW;++ring_calls;
}
void linear(uint32_t in,uint32_t out){
 if(in==SDPA){assert(out==PARTIAL);content(out)=PARTIALS;}
 else if(in==GATE){assert(out==PARTIAL&&content(in)==ACTIVATION);content(out)=PARTIALS;}
 else {assert(in==PARTIAL&&content(in)==FULL_NORM&&(out==GATE||out==UP));content(out)=ACTIVATION;}
 ++linear_calls;
}
template<class... A>void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t in,
 const at::Tensor&,uint32_t out,A...){linear(in,out);}
template<class... A>void rpu_launch_wall_vision_row_weight_outer_w8a16_spm_kernel(uint32_t in,
 const at::Tensor&,uint32_t out,A...){linear(in,out);}
template<class... A>void rpu_launch_wall_vision_gate_up_swiglu_w8a16_spm_kernel(uint32_t in,
 const at::Tensor&,const at::Tensor&,uint32_t out,A...){
 assert(content(in)==FULL_NORM&&in==PARTIAL&&out==GATE);content(out)=ACTIVATION;++mlp_calls;}
void rpu_launch_silu_mul_spm_kernel(uint32_t g,uint32_t u,uint32_t o,int64_t elems,int cores){
 assert(g==GATE&&u==UP&&o==GATE&&content(g)==ACTIVATION&&content(u)==ACTIVATION);
 assert(elems==672*448&&cores==8);++mlp_calls;
}
template<class... A>void rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel(A...){assert(false);}
template<class... A>void rpu_launch_all_reduce_fp16_partial_bf16_residual_spm_kernel(A...){assert(false);}
template<class... A>void rpu_launch_all_reduce_bf16_partial_fp16_residual_spm_kernel(A...){assert(false);}
template<class... A>void rpu_launch_linear_spm_to_spm_acc32_out_bf16_kernel(A...){assert(false);}
template<class... A>void rpu_launch_eltwise_unary_spm_kernel(A...){assert(false);}
template<class... A>void rpu_launch_eltwise_binary_spm_kernel(A...){assert(false);}
'''
    source = source.replace('struct Model {', globals_ + '\nstruct Model {', 1)
    source = source.replace(' Context context;', r'''
 bool small_owner_ring_norm_available_=false,small_down_owner_norm_available_=false;
 uint32_t small_rmsnorm_capability_=RPU_RMSNORM_CAP_BASE|RPU_RMSNORM_CAP_V32;
 double eps_=1e-6;at::Tensor merger_ln_q_w_;
''' + block(vision, '    struct LayerBiasNorm {') + r''';
 std::vector<LayerBiasNorm> layer_bias_norm_=std::vector<LayerBiasNorm>(32);
''' + methods + r'''
 void consume_manifest_route(FmbRouteFamily family,int64_t site,int64_t selector,
    int64_t invocation,std::vector<int64_t> args={}){
  if(family==FmbRouteFamily::ALL_REDUCE)
   ctx().consume_physical_route(family,site,selector,0,args,invocation);
  else assert(family==FmbRouteFamily::LINEAR); // unrelated projection route is unchanged
 }
 int consume_mixed_rmsnorm_route(int64_t,int64_t,int64_t,int64_t){assert(false);return 0;}
 uint32_t layer_addr(int layer,int core,const std::string& name){
  assert(layer>=0&&layer<32&&core==0);
  return 0x100000+layer*256+(name=="norm2_w"?128:0);
 }
 void emit_layer_input_dma(int layer,const ChunkInfo&){assert(layer==0&&content(RES)==FULL_RAW);}
 void emit_debug_phase(int,const char*){assert(!debug);}
 auto produce_owner_routes(const FmbThreeStageChunkPlan& plan){
  if(small_image_down_owner_norm_enabled(current_num_patches_))validate_small_image_down_owner_plan(plan);
  FmbPhysicalExecutionManifest manifest;
  auto append=[&](FmbRouteFamily f,int64_t site,int64_t selector,int64_t invocation=0,
                  std::vector<int64_t> args={},int64_t flags=0){
   manifest.routes.push_back({site,f,selector,flags,std::move(args),invocation});};
  auto append_ring=[&](int64_t site,int64_t rows,int64_t cols,int64_t invocation){
   append(FmbRouteFamily::ALL_REDUCE,site,qwen25_vision_ring_route(rows,cols),invocation);};
  const bool has_pre_boundary_layer=true;
  for(const auto& chunk:plan.compute.chunks){
PRODUCER
  }
SORT_OWNER
  return manifest;
 }
 void emit_owner_chain(int64_t layer_idx){
  const auto chunk=ctx().stage_plan.compute.chunks.at(0);
  const int64_t seq_len=chunk.len,h=hidden_size(),nq=num_q_heads(),hd=head_dim(),
                is_=intermediate_size(),local_inter=is_/8;
  const bool residual_is_bf16=false;const auto& lw=layer_weights_.at(layer_idx);
HEAD
TAIL
 }
 Context context;''')
    old_addr = block(source, ' uint32_t addr(int core,const std::string& name)')
    source = source.replace(old_addr, r''' uint32_t addr(int core,const std::string& name){assert(core==0);
  return name=="residual1"?RES:name=="input_norm"?INPUT:name=="oproj"?PARTIAL:
         name=="sdpa_out"?SDPA:name=="gate"?GATE:UP;}''')
    source = source.replace(' m.context.stage_plan=plan_for(rows);m.bind();return m;', r'''
 m.merger_ln_q_w_.present=true;
 for(auto& bn:m.layer_bias_norm_){bn.norm1_w.present=bn.norm2_w.present=true;}
 m.context.stage_plan=plan_for(rows);m.bind();return m;''')
    source = source.replace('\nPRODUCER\n', '\n' + producer + '\n')
    source = source.replace('\nHEAD\n', '\n' + head + '\n').replace('\nTAIL\n', '\n' + tail + '\n')
    source = source.replace('SORT_OWNER', sort_routes)
    source += r'''
int main(){
 publish();auto& cache=KernelCache::instance();
 for(bool fp16:{false,true})for(int variant:{0,1,2}){
  // Both new symbols, O only, and neither symbol (the original REF path).
  cache.o_kernel=cache.o_program=variant<2;
  cache.down_kernel=cache.down_program=variant==0;
  Model m=exact(672,fp16);
  assert(m.small_image_owner_ring_norm_enabled(672)==(variant<2));
  assert(m.small_image_down_owner_norm_enabled(672)==(variant==0));
  m.context.value=m.produce_owner_routes(m.context.stage_plan);
  auto count=[&](int64_t site){return std::count_if(m.context.value.routes.begin(),m.context.value.routes.end(),
   [&](const auto& r){return r.site_id==site;});};
  assert(count(QWEN25_SMALL_INPUT_RMSNORM_SITE)==1);
  assert(count(QWEN25_SMALL_POST_RMSNORM_SITE)==(variant==2));
  assert(count(QWEN25_SMALL_OWNER_RING_NORM_SITE)==(variant<2));
  assert(count(QWEN25_SMALL_DOWN_OWNER_NORM_SITE)==(variant==0?31:0));
  assert(count(QWEN25_ATTN_REDUCE_SITE)==(variant==2));
  assert(count(QWEN25_DOWN_REDUCE_SITE)==1); // final raw output is always published
  contents.fill(EMPTY);content(RES)=FULL_RAW;
  o_calls=down_calls=ring_calls=norm_calls=linear_calls=mlp_calls=0;
  for(int layer=0;layer<32;++layer){
   m.context.input_in_spm=layer>0;m.context.output_to_spm=layer<31;
   m.context.consumed_.assign(m.context.value.routes.size(),0);
   m.emit_owner_chain(layer);
   if(variant==0&&layer<31){assert(content(RES)==OWNER_RAW&&content(INPUT)==FULL_NORM);}
   else assert(content(RES)==FULL_RAW);
  }
  assert(o_calls==(variant<2?32:0)&&down_calls==(variant==0?31:0));
  assert(norm_calls==(variant==0?1:variant==1?32:64));
  assert(ring_calls==(variant==0?1:variant==1?32:64));
  assert(linear_calls==(fp16?128:64)&&mlp_calls==32);
  assert(content(RES)==FULL_RAW); // merger reads all 672 physical rows on every core
 }
 cache.o_kernel=cache.o_program=cache.down_kernel=cache.down_program=true;
 for(bool fp16:{false,true}){
  auto m=exact(672,fp16);auto original=m.produce_owner_routes(m.context.stage_plan);
  // Exercise the actual COMPLETE lookup, including route removal and stale
  // per-layer gamma/alias arguments; a receipt for another layer cannot match.
  for(int64_t site:{QWEN25_SMALL_OWNER_RING_NORM_SITE,QWEN25_SMALL_DOWN_OWNER_NORM_SITE}){
   for(int mutation=0;mutation<8;++mutation){
    auto bad=m;bad.context.value=original;
    auto it=std::find_if(bad.context.value.routes.begin(),bad.context.value.routes.end(),
      [&](const auto& r){return r.site_id==site;});assert(it!=bad.context.value.routes.end());
    switch(mutation){case 0:bad.context.ready=false;break;case 1:it->selector++;break;
     case 2:it->flags=1;break;case 3:it->invocation=-1;break;
     case 4:it->family=FmbRouteFamily::LINEAR;break;case 5:it->arguments.pop_back();break;
     case 6:bad.context.value.routes.erase(it);break;case 7:it->arguments[6]++;break;}
    bad.context.consumed_.resize(bad.context.value.routes.size());
    rejects([&]{if(site==QWEN25_SMALL_OWNER_RING_NORM_SITE)bad.consume_small_image_owner_ring_norm(672,0);
                else bad.consume_small_image_down_owner_norm(672,0);});
   }
   auto it=std::find_if(original.routes.begin(),original.routes.end(),[&](const auto& r){return r.site_id==site;});
   for(size_t arg=0;arg<it->arguments.size();++arg){
    auto bad=m;bad.context.value=original;
    auto r=std::find_if(bad.context.value.routes.begin(),bad.context.value.routes.end(),[&](const auto& r){return r.site_id==site;});
    ++r->arguments[arg];bad.context.consumed_.resize(bad.context.value.routes.size());
    rejects([&]{if(site==QWEN25_SMALL_OWNER_RING_NORM_SITE)bad.consume_small_image_owner_ring_norm(672,0);
                else bad.consume_small_image_down_owner_norm(672,0);});
   }
  }
  for(int marker=0;marker<4;++marker){
   cache.o_kernel=cache.o_program=cache.down_kernel=cache.down_program=true;
   if(marker==0)cache.o_kernel=false;if(marker==1)cache.o_program=false;
   if(marker==2)cache.down_kernel=false;if(marker==3)cache.down_program=false;
   auto bad=exact(672,fp16);
   assert(!bad.small_image_down_owner_norm_enabled(672));
   assert(bad.small_image_owner_ring_norm_enabled(672)==(marker>=2));
  }
  cache.o_kernel=cache.o_program=cache.down_kernel=cache.down_program=true;
 }
}
'''
    compile_run(tmp_path, 'wall_vision_owner_layer_handoff', source)
