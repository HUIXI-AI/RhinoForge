// Resident-input A/B: production W8A16 versus device quantize + W8A8.
// Row variants both include the same actual production zero-residual ring.
// No host upload, output read, host quantization, tensor or SPM allocation is timed.
// Immediate timing retains launcher overhead; retained Graph timing skips it.

#include "graph/execution_coordinator.h"
#include "graph/graph_infra.h"
#include "graph/graph_runtime.h"
#include "rhino_launch_buffer.h"
#include "rpu_kernel_decls.h"
#include "rpu_spm_allocator.h"
#include "ops/rpu_pi05_owner_norm.h"
#include "ops/rpu_linear_w8a8_contract.h"

#include <ATen/ATen.h>
#include <c10/util/Half.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

using namespace rhino_lkn;

namespace {
size_t aligned(size_t bytes) { return (bytes + 255) & ~size_t(255); }

// Test-only explicit selection. Production launchers and their selection remain
// unchanged; a trial cannot silently substitute a missing/incompatible payload.
rpu_pl_w8a8::Tile qwen_tile_probe(int64_t m, int64_t n, int64_t k,
                                int64_t partition, int64_t cores, bool bias,
                                int64_t projections, bool producer, int64_t ordinal) {
    const bool shape = (m == 128 || m == 192) &&
        ((partition == 1 && k == 5120 && (n == 8192 || n == 1024 || n == 25600)) ||
         (partition == 0 && n == 5120 && (k == 8192 || k == 25600)));
    TORCH_CHECK(shape && cores == 8 && !bias && projections == 1 && !producer,
                "W8A8 tile probe requires one exact Qwen32 C128/C192 TP8 projection without bias/producer");
    TORCH_CHECK(ordinal >= 0 && ordinal < static_cast<int64_t>(rpu_pl_w8a8::kCandidateCount),
                "W8A8 tile probe ordinal is outside the installed catalog");
    const auto local_k = partition == 0 ? k / cores : k;
    const auto candidate = rpu_pl_w8a8::installed_candidates()[ordinal];
    TORCH_CHECK(!candidate.compiled_local_k || candidate.compiled_local_k == local_k,
                "W8A8 tile probe resident payload has a different compiled local K");
    return rpu_pl_w8a8::validate_tile(candidate.tile, local_k, false);
}

constexpr KernelId kProbeLinearIds[] = {
#define RPU_W8A8_CANDIDATE(id, name, ordinal, m, n, k, lk, flags) KernelId::id,
#include "ops/rpu_linear_w8a8_candidates.inc"
#undef RPU_W8A8_CANDIDATE
};

// The caller validates owners and constructs this real ABI descriptor before
// timing. Use the same typed binding and graph-aware queue as production; never
// publish a process-global tile override or patch a production descriptor.
void launch_tile_probe(const rpu_pl_w8a8::Descriptor& descriptor,
                       const at::Tensor& weight, const at::Tensor& scale) {
    const auto id = kProbeLinearIds[descriptor.ordinal];
    auto& graph = RpuKernelGraph::active();
    auto writer = graph.stage_kernel_ddr_registers(id,
        std::vector<GraphDdrRegisterOperandSpec>{
            {{GraphDdrRegisterRole::Int8ModelWeight, GraphDdrRegisterAccess::Read,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, weight},
            {{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, scale}});
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel, "W8A8 tile probe payload unavailable: ", descriptor.name);
    graph.keep_alive(weight);
    graph.keep_alive(scale);
    rpu_ddr_flush(weight.data_ptr());
    rpu_ddr_flush(scale.data_ptr());
    kernel->reset_regs();
    for (uint32_t index = 0; index < descriptor.regs.size(); ++index)
        TORCH_CHECK(kernel->set_regs(index, descriptor.regs[index]) == kKernelRegOk,
                    "W8A8 tile probe register rejected: ", index);
    writer.write(*kernel);
    for (uint32_t index = 0; index < descriptor.grid.size(); ++index)
        TORCH_CHECK(kernel->set_regs(64 + index, descriptor.grid[index]) == kKernelRegOk,
                    "W8A8 tile probe grid rejected");
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel,
        {descriptor.grid[0], descriptor.grid[1], descriptor.grid[2]}, {0, 1, 2, 3, 4, 5, 6, 7});
}

at::Tensor aligned_owner(const at::Tensor& source) {
    const auto address = RpuGetDevAddr(source.data_ptr());
    TORCH_CHECK(address, "W8A8 bench: missing DDR address");
    if (address % 256 == 0) return source;
    const auto item = source.element_size();
    auto storage = at::empty({source.numel() + 256 / item}, source.options());
    const auto base = RpuGetDevAddr(storage.data_ptr());
    const auto offset = (256 - base % 256) % 256;
    auto result = storage.narrow(0, offset / item, source.numel()).view(source.sizes());
    result.copy_(source);
    TORCH_CHECK(RpuGetDevAddr(result.data_ptr()) % 256 == 0,
                "W8A8 bench: failed to align DDR owner");
    return result;
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> pipeline_bench_impl(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& scale,
    const c10::optional<at::Tensor>& bias, int64_t partition, int64_t cores,
    int64_t warmup, int64_t runs, bool a_acc32, bool retained_graph, bool breakdown,
    int64_t projections, const c10::optional<at::Tensor>& norm_gamma,
    const c10::optional<at::Tensor>& norm_beta, double norm_eps,
    at::Tensor* norm_reference_out, int producer = 0,
    const c10::optional<at::Tensor>& residual = c10::nullopt,
    int64_t tile_ordinal = -1) {
    RpuExecutionCoordinator::require_graph_quiescent("W8A8 pipeline benchmark");
    RpuExecutionCoordinator::check_current_thread_execution_allowed("W8A8 pipeline benchmark");
    TORCH_CHECK(RpuKernelGraph::active().state() == RpuKernelGraph::State::PASSTHROUGH,
                "W8A8 bench requires no outer Graph execution");
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 &&
                    weight.device() == x.device() && scale.device() == x.device(),
                "W8A8 bench: operands must share the RPU device");
    TORCH_CHECK(x.scalar_type() == at::kHalf && weight.scalar_type() == at::kChar &&
                    scale.scalar_type() == at::kHalf && x.is_contiguous() &&
                    weight.is_contiguous() && scale.is_contiguous(),
                "W8A8 bench: contiguous FP16 X/SW and INT8 W required");
    const bool owner_norm = producer == 3;
    const bool gelu = producer == 2;
    const bool layer_norm = norm_gamma.has_value() && !owner_norm;
    const bool has_producer = layer_norm || gelu || owner_norm;
    TORCH_CHECK(x.dim() == (owner_norm ? 3 : 2) && weight.dim() == 2 && scale.dim() == 1,
                "W8A8 bench: expected X[M,K], packed W[N,K], SW[N]");
    const int64_t m = x.size(owner_norm ? 1 : 0), k = x.size(owner_norm ? 2 : 1), n = weight.size(0);
    TORCH_CHECK(norm_gamma.has_value() == norm_beta.has_value() &&
                    has_producer == (norm_reference_out != nullptr),
                "W8A8 norm bench requires gamma, beta and a baseline norm output");
    if (layer_norm || owner_norm) {
        TORCH_CHECK(norm_gamma.has_value() && norm_beta.has_value(),
                    "W8A8 normalization producer requires gamma and beta");
        TORCH_CHECK(partition == 1 && projections == (owner_norm ? 1 : 3) && k == 1152 &&
                        m > 0 && m <= 768 && std::isfinite(norm_eps) && norm_eps > 0,
                    "W8A8 norm bench requires column TP, three projections, H1152, M1..768 and positive finite eps");
        for (const auto* affine : {&*norm_gamma, &*norm_beta}) {
            TORCH_CHECK(affine->device() == x.device() &&
                            affine->scalar_type() == at::kHalf && affine->dim() == 1 &&
                            affine->numel() == k && affine->is_contiguous(),
                        "W8A8 norm bench requires contiguous FP16 gamma/beta [1152] on RPU");
        }
    }
    if (gelu || owner_norm) {
        TORCH_CHECK(cores == 8 && (m == 512 || m == 768) && projections == 1 &&
                        !a_acc32 && residual.has_value(),
                    "W8A8 producer bench requires TP8/M512|768/ACC16, one projection and residual");
        TORCH_CHECK((gelu && partition == 0 && k == 4352 && n == 1152) ||
                        (owner_norm && partition == 1 && k == 1152 && n == 4352 &&
                         x.size(0) == cores && norm_eps == 1e-6),
                    "W8A8 producer bench requires exact Vision FC2 or owner-LN2/FC1 geometry");
        TORCH_CHECK(residual->device() == x.device() && residual->scalar_type() == at::kHalf &&
                        residual->is_contiguous() && residual->dim() == 2 &&
                        residual->size(0) == m && residual->size(1) == 1152,
                    "W8A8 producer bench residual must be contiguous FP16 [M,1152] on RPU");
    }
    TORCH_CHECK(m > 0 && m <= 65535 && n > 0 && n <= (1 << 20) &&
                    k > 0 && k <= 8 * 65504 && weight.size(1) == k && scale.numel() == n,
                "W8A8 bench: invalid physical dimensions");
    TORCH_CHECK((partition == 0 || partition == 1) && cores >= 1 && cores <= 8,
                "W8A8 bench: partition=0(row)/1(col), cores=1..8");
    TORCH_CHECK(projections == 1 || (projections == 3 && partition == 1),
                "W8A8 bench: projections must be 1, or 3 for column partition only");
    TORCH_CHECK(warmup >= 1 && warmup <= 10 && runs >= 1 && runs <= 30,
                "W8A8 bench: warmup=1..10, runs=1..30");
    TORCH_CHECK((partition == 0 ? k : n) % cores == 0,
                "W8A8 bench: TP dimensions must be divisible");
    const int64_t local_k = partition == 0 ? k / cores : k;
    const int64_t local_n = partition == 0 ? n : n / cores;
    TORCH_CHECK(tile_ordinal >= -1, "W8A8 tile probe uses -1 for production selection");
    const bool tile_probe = tile_ordinal >= 0;
    TORCH_CHECK(!tile_probe || retained_graph,
                "W8A8 explicit tile probe requires retained Graph timing");
    rpu_pl_w8a8::Tile explicit_tile;
    if (tile_probe) {
        explicit_tile = qwen_tile_probe(m, n, k, partition, cores, bias.has_value(),
                                       projections, has_producer, tile_ordinal);
        TORCH_CHECK(KernelCache::instance().has_loaded(kProbeLinearIds[tile_ordinal]),
                    "W8A8 tile probe requires its selected main payload");
    }
    TORCH_CHECK(local_k <= 65504 && local_k % 32 == 0 && local_n % 16 == 0,
                "W8A8 bench requires physical per-core K32/N16 padding");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device() == x.device() && bias->scalar_type() == at::kHalf &&
                        bias->is_contiguous() && bias->dim() == 1 && bias->numel() == n,
                    "W8A8 bench: bias must be contiguous FP16 [N] on RPU");
    }

    const size_t input_rows = has_producer ? ((m + 15) / 16) * 16 : m;
    const size_t x_bytes = aligned(input_rows * local_k * 2);
    const size_t q_bytes = aligned(m * local_k);
    const size_t sx_bytes = aligned(((m + 15) / 16) * 32);
    const size_t y_bytes = aligned(m * local_n * 2);
    const size_t bias_bytes = bias.has_value() ? aligned(local_n * 2) : 0;
    const size_t norm_bytes = has_producer ? x_bytes : 0;
    const size_t affine_bytes = (layer_norm || owner_norm) ? aligned(k * 2) : 0;
    const size_t guard_bytes = has_producer || tile_probe ? 256 : 0;
    const size_t y_guard_bytes = tile_probe ? 256 : 0;
    const size_t owner_residual_bytes = owner_norm ? aligned(m * k * 2) : 0;
    const size_t compact_raw_bytes = owner_norm ? aligned(m * k * 2 / cores) : 0;
    const size_t footprint = x_bytes + q_bytes + sx_bytes + projections * y_bytes + bias_bytes +
                             (partition == 0 ? 2 * y_bytes : 0) + norm_bytes +
                             2 * affine_bytes + (owner_norm ? 6 : 4) * guard_bytes + owner_residual_bytes + compact_raw_bytes + 2 * y_guard_bytes;
    std::array<at::Tensor, 3> weights, scales;
    rpu_ddr_flush_force(x.data_ptr());
    for (int64_t projection = 0; projection < projections; ++projection) {
        // Three real independent DDR owners; identical values simplify the CPU
        // reference without giving the projections a shared weight cache address.
        weights[projection] = aligned_owner(projections == 1 ? weight : weight.clone());
        scales[projection] = aligned_owner(projections == 1 ? scale : scale.clone());
        rpu_ddr_flush_force(weights[projection].data_ptr());
        rpu_ddr_flush_force(scales[projection].data_ptr());
        for (int64_t prior = 0; prior < projection; ++prior) {
            TORCH_CHECK(RpuGetDevAddr(weights[projection].data_ptr()) !=
                            RpuGetDevAddr(weights[prior].data_ptr()) &&
                            RpuGetDevAddr(scales[projection].data_ptr()) !=
                            RpuGetDevAddr(scales[prior].data_ptr()),
                        "W8A8 bench requires independent projection W/SW addresses");
        }
    }
    if (bias.has_value()) rpu_ddr_flush_force(bias->data_ptr());

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    const auto used = SPM_ALLOC.total_used();
    TORCH_CHECK(used <= SpmAllocator::SPM_PLANNING_BUDGET &&
                    footprint <= SpmAllocator::SPM_PLANNING_BUDGET - used,
                "W8A8 bench SPM footprint exceeds remaining planning budget: used=",
                used, " requested=", footprint);
    SpmAllocator::ScopedTemporary temporary;
    const auto x_off = SPM_ALLOC.alloc_temporary(x_bytes);
    const auto q_storage = SPM_ALLOC.alloc_temporary(q_bytes + 2 * guard_bytes);
    const auto q_off = q_storage + guard_bytes;
    const auto sx_storage = SPM_ALLOC.alloc_temporary(sx_bytes + 2 * guard_bytes);
    const auto sx_off = sx_storage + guard_bytes;
    const auto norm_off = has_producer ? SPM_ALLOC.alloc_temporary(norm_bytes) : x_off;
    const auto gamma_off = affine_bytes ? SPM_ALLOC.alloc_temporary(affine_bytes) : 0;
    const auto beta_off = affine_bytes ? SPM_ALLOC.alloc_temporary(affine_bytes) : 0;
    const auto owner_residual_off = owner_norm ? SPM_ALLOC.alloc_temporary(owner_residual_bytes) : 0;
    const auto compact_raw_storage = owner_norm ? SPM_ALLOC.alloc_temporary(compact_raw_bytes + 2 * guard_bytes) : 0;
    const auto compact_raw_off = owner_norm ? compact_raw_storage + guard_bytes : 0;
    // Each projection owns one disjoint, 256-byte-aligned output slot.
    const auto y_storage = SPM_ALLOC.alloc_temporary(projections * y_bytes + 2 * y_guard_bytes);
    const auto y_off = y_storage + y_guard_bytes;
    const auto bias_off = bias.has_value() ? SPM_ALLOC.alloc_temporary(bias_bytes) : 0;
    const auto zero_off = partition == 0 ? SPM_ALLOC.alloc_temporary(y_bytes) : 0;
    const auto final_off = partition == 0 ? SPM_ALLOC.alloc_temporary(y_bytes) : y_off;
    // SPM is device-mapped memory. AArch64 libc zero memset may use DC ZVA,
    // which raises SIGBUS on this mapping; initialize through the production
    // device fill instead. Both fills remain outside the measured A/B region.
    if (partition == 0)
        rpu_launch_fill_spm_kernel(SPM_ALLOC.addr(0, zero_off), y_bytes / 2,
                                  c10::Half(0.0f), cores);
    if (bias.has_value())
        rpu_launch_fill_spm_kernel(SPM_ALLOC.addr(0, bias_off), bias_bytes / 2,
                                  c10::Half(0.0f), cores);
    if (has_producer) {
        rpu_launch_fill_spm_kernel(SPM_ALLOC.addr(0, x_off), x_bytes / 2,
                                  c10::Half(0.0f), cores);
        rpu_launch_fill_spm_kernel(SPM_ALLOC.addr(0, norm_off), norm_bytes / 2,
                                  c10::Half(0.0f), cores);
        if (norm_gamma.has_value()) {
            rpu_ddr_flush_force(norm_gamma->data_ptr());
            rpu_ddr_flush_force(norm_beta->data_ptr());
        }
    }
    if (residual.has_value()) rpu_ddr_flush_force(residual->data_ptr());
    for (int core = 0; core < cores; ++core) {
        auto* dst = static_cast<c10::Half*>(SPM_ALLOC.cpu_ptr(core, x_off));
        const auto* src = x.data_ptr<c10::Half>();
        for (int64_t row = 0; row < m; ++row) {
            std::memcpy(dst + row * local_k,
                        src + (owner_norm ? core * m * k : 0) + row * k + (partition == 0 ? core * local_k : 0),
                        local_k * sizeof(c10::Half));
        }
        if (bias.has_value()) {
            auto* target = SPM_ALLOC.cpu_ptr(core, bias_off);
            if (partition == 1 || core == 0) {
                std::memcpy(target, bias->data_ptr<c10::Half>() +
                    (partition == 1 ? core * local_n : 0), local_n * sizeof(c10::Half));
            }
        }
        if (affine_bytes) {
            std::memcpy(SPM_ALLOC.cpu_ptr(core, gamma_off), norm_gamma->data_ptr(), k * 2);
            std::memcpy(SPM_ALLOC.cpu_ptr(core, beta_off), norm_beta->data_ptr(), k * 2);
        }
        if (owner_norm)
            std::memcpy(SPM_ALLOC.cpu_ptr(core, owner_residual_off), residual->data_ptr(), m * k * 2);
        if (gelu)
            std::memcpy(SPM_ALLOC.cpu_ptr(core, zero_off),
                        residual->data_ptr<c10::Half>() + core * (m / cores) * n,
                        (m / cores) * n * 2);
    }
    const auto x_addr = SPM_ALLOC.addr(0, x_off);
    const auto q_addr = SPM_ALLOC.addr(0, q_off);
    const auto sx_addr = SPM_ALLOC.addr(0, sx_off);
    const auto y_addr = SPM_ALLOC.addr(0, y_off);
    const auto bias_addr = bias.has_value() ? SPM_ALLOC.addr(0, bias_off) : 0;
    const auto final_addr = SPM_ALLOC.addr(0, final_off);
    const auto zero_addr = partition == 0 ? SPM_ALLOC.addr(0, zero_off) : 0;
    const auto norm_addr = SPM_ALLOC.addr(0, norm_off);
    const auto gamma_addr = affine_bytes ? SPM_ALLOC.addr(0, gamma_off) : 0;
    const auto beta_addr = affine_bytes ? SPM_ALLOC.addr(0, beta_off) : 0;
    const auto owner_residual_addr = owner_norm ? SPM_ALLOC.addr(0, owner_residual_off) : 0;
    const auto compact_raw_addr = owner_norm ? SPM_ALLOC.addr(0, compact_raw_off) : 0;
    rpu_pl_w8a8::Descriptor explicit_descriptor;
    if (tile_probe) {
        for (const auto* owner : {&weights[0], &scales[0]}) {
            const auto offset = owner->storage_offset();
            const auto address = RpuGetDevAddr(owner->data_ptr());
            constexpr uint64_t limit = UINT64_C(1) << 40;
            TORCH_CHECK(offset >= 0 && static_cast<bool>(owner->storage()) &&
                            uint64_t(offset) <= owner->storage().nbytes() / owner->element_size() &&
                            owner->nbytes() <= owner->storage().nbytes() - uint64_t(offset) * owner->element_size() &&
                            address && address % 256 == 0 && address < limit && owner->nbytes() <= limit - address,
                        "W8A8 tile probe DDR owner backing/address is invalid");
        }
        const auto wa = RpuGetDevAddr(weights[0].data_ptr()), sa = RpuGetDevAddr(scales[0].data_ptr());
        TORCH_CHECK(wa + weights[0].nbytes() <= sa || sa + scales[0].nbytes() <= wa,
                    "W8A8 tile probe W/SW owners overlap");
        const auto shape = rpu_pl_w8a8::physical_shape(m, n, k,
            static_cast<rpu_pl_w8a8::Partition>(partition), cores, false);
        explicit_descriptor = rpu_pl_w8a8::make_descriptor(shape,
            {q_addr, y_addr, 0, sx_addr}, SPM_ALLOC.addr(0, 0), SpmAllocator::SPM_USABLE,
            &explicit_tile);
        TORCH_CHECK(explicit_descriptor.ordinal == static_cast<size_t>(tile_ordinal),
                    "W8A8 tile probe resolved a different ordinal");
        std::cout << "W8A8_TILE_PROBE ordinal=" << tile_ordinal
                  << " kernel=" << explicit_descriptor.name << " grid=" << explicit_descriptor.grid[0]
                  << ',' << explicit_descriptor.grid[1] << " vlm_rows=" << explicit_tile.vlm_rows << std::endl;
    }
    const auto baseline_norm = [&]() {
        if (owner_norm)
            rpu_launch_pi05_vision_owner_layernorm_spm_kernel(x_addr, owner_residual_addr,
                compact_raw_addr, norm_addr, gamma_addr, beta_addr, norm_eps, m);
        else if (gelu)
            rpu_launch_eltwise_unary_spm_kernel(x_addr, norm_addr, m * local_k,
                ValuOpType::ADD, GeluMode::TANH, cores);
        else
            rpu_launch_layernorm_spm_kernel(x_addr, norm_addr, gamma_addr, beta_addr,
                                          m, k, norm_eps, false, 0, cores);
    };
    const auto fused_norm_quantize = [&]() {
        if (owner_norm)
            rpu_launch_pi05_vision_owner_layernorm_quantize_w8a8_spm_kernel(
                x_addr, owner_residual_addr, compact_raw_addr, q_addr, sx_addr,
                gamma_addr, beta_addr, norm_eps, m);
        else if (gelu)
            rpu_launch_gelu_tanh_quantize_w8a8_spm_kernel(x_addr, q_addr, sx_addr, m, cores);
        else
            rpu_launch_layer_norm_quantize_w8a8_spm_kernel(
                x_addr, q_addr, sx_addr, gamma_addr, beta_addr, m, norm_eps, cores);
    };
    const c10::Half guard_value(-7.0f);
    if (tile_probe) {
        for (const auto span : std::array<std::pair<size_t, size_t>, 3>{{
                {q_storage, q_bytes + 2 * guard_bytes},
                {sx_storage, sx_bytes + 2 * guard_bytes},
                {y_storage, y_bytes + 2 * y_guard_bytes}}})
            rpu_launch_fill_spm_kernel(SPM_ALLOC.addr(0, span.first), span.second / 2, guard_value, cores);
    }
    if (owner_norm)
        rpu_launch_fill_spm_kernel(SPM_ALLOC.addr(0, compact_raw_storage),
            (compact_raw_bytes + 2 * guard_bytes) / 2, guard_value, cores);
    std::vector<std::vector<uint8_t>> expected_raw(cores);
    const auto verify_bytes = [&](const std::vector<uint8_t>& expected, int core, size_t offset,
                                  const char* label) {
        const auto* actual = static_cast<const uint8_t*>(SPM_ALLOC.cpu_ptr(core, offset));
        for (size_t index = 0; index < expected.size(); ++index)
            TORCH_CHECK(actual[index] == expected[index],
                        "W8A8 producer bench: ", label, " mismatch at core=", core,
                        " byte=", index, " expected=", int(expected[index]),
                        " actual=", int(actual[index]));
    };
    const auto check_guards = [&]() {
        if (!guard_bytes) return;
        for (int core = 0; core < cores; ++core) {
            for (const auto offset : std::array<size_t, 4>{q_storage, q_off + q_bytes,
                                                          sx_storage, sx_off + sx_bytes}) {
                const auto* values = static_cast<const c10::Half*>(SPM_ALLOC.cpu_ptr(core, offset));
                for (size_t index = 0; index < guard_bytes / 2; ++index)
                    TORCH_CHECK(values[index].x == guard_value.x,
                                "W8A8 norm bench: Q/SX guard changed at core=", core,
                                " offset=", offset, " half=", index);
            }
            if (tile_probe) {
                for (const auto offset : std::array<size_t, 2>{y_storage, y_off + y_bytes}) {
                    const auto* values = static_cast<const c10::Half*>(SPM_ALLOC.cpu_ptr(core, offset));
                    for (size_t index = 0; index < y_guard_bytes / 2; ++index)
                        TORCH_CHECK(values[index].x == guard_value.x,
                                    "W8A8 tile probe Y guard changed at core=", core);
                }
            }
            if (owner_norm) {
                for (const auto offset : std::array<size_t, 2>{compact_raw_storage, compact_raw_off + compact_raw_bytes}) {
                    const auto* values = static_cast<const c10::Half*>(SPM_ALLOC.cpu_ptr(core, offset));
                    for (size_t index = 0; index < guard_bytes / 2; ++index)
                        TORCH_CHECK(values[index].x == guard_value.x,
                                    "W8A8 owner bench: compact raw guard changed at core=", core);
                }
                if (!expected_raw[core].empty())
                    verify_bytes(expected_raw[core], core, compact_raw_off, "compact raw");
            }
        }
    };
    if (has_producer) {
        const auto poison_quantized = [&]() {
            rpu_launch_fill_spm_kernel(SPM_ALLOC.addr(0, q_storage),
                (q_bytes + 2 * guard_bytes) / 2, guard_value, cores);
            rpu_launch_fill_spm_kernel(SPM_ALLOC.addr(0, sx_storage),
                (sx_bytes + 2 * guard_bytes) / 2, guard_value, cores);
        };
        // Establish the real FP16 LayerNorm reference and all-core Q/SX parity
        // before BUILD. CPU LayerNorm is not the fused kernel's semantic oracle.
        baseline_norm();
        *norm_reference_out = at::empty({m, k}, at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
        for (int core = 0; core < cores; ++core) {
            const auto* source = static_cast<const c10::Half*>(SPM_ALLOC.cpu_ptr(core, norm_off));
            auto* target = norm_reference_out->data_ptr<c10::Half>();
            if (partition == 0) {
                for (int64_t row = 0; row < m; ++row)
                    std::memcpy(target + row * k + core * local_k,
                                source + row * local_k, local_k * 2);
            } else if (core == 0) {
                std::memcpy(target, source, m * k * 2);
            } else {
                TORCH_CHECK(std::memcmp(target, source, m * k * 2) == 0,
                            "W8A8 producer bench: baseline normalized gather differs across cores");
            }
            if (owner_norm) {
                expected_raw[core].resize(compact_raw_bytes);
                std::memcpy(expected_raw[core].data(), SPM_ALLOC.cpu_ptr(core, compact_raw_off), compact_raw_bytes);
            }
        }
        TORCH_CHECK(at::isfinite(*norm_reference_out).all().item<bool>(),
                    "W8A8 producer bench: baseline FP16 producer contains nonfinite values");
        poison_quantized();
        rpu_launch_dynamic_quantize_w8a8_spm_kernel(norm_addr, q_addr, sx_addr, m, local_k, cores);
        check_guards();
        std::vector<std::vector<uint8_t>> expected_q(cores), expected_sx(cores);
        for (int core = 0; core < cores; ++core) {
            expected_q[core].resize(q_bytes);
            expected_sx[core].resize(sx_bytes);
            std::memcpy(expected_q[core].data(), SPM_ALLOC.cpu_ptr(core, q_off), q_bytes);
            std::memcpy(expected_sx[core].data(), SPM_ALLOC.cpu_ptr(core, sx_off), sx_bytes);
        }
        poison_quantized();
        if (owner_norm)
            rpu_launch_fill_spm_kernel(compact_raw_addr, compact_raw_bytes / 2, guard_value, cores);
        fused_norm_quantize();
        check_guards();
        for (int core = 0; core < cores; ++core) {
            verify_bytes(expected_q[core], core, q_off, "Q");
            verify_bytes(expected_sx[core], core, sx_off, "SX");
        }
        std::cout << "W8A8_PRODUCER_PARITY producer=" << producer << " cores=" << cores << " q_bytes=" << q_bytes
                  << " sx_bytes=" << sx_bytes << " exact=1 guards=1 compact_raw=" << owner_norm << std::endl;
    }
    std::array<uint32_t, 3> projection_y_addrs{};
    for (int64_t projection = 0; projection < projections; ++projection)
        projection_y_addrs[projection] = SPM_ALLOC.addr(0, y_off + projection * y_bytes);
    const std::vector<int64_t> output_shape = projections == 1
        ? std::vector<int64_t>{m, n} : std::vector<int64_t>{projections, m, n};
    auto output_a = at::empty(output_shape, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    auto output_b = at::empty_like(output_a);
    // 0: W8A16, 1: quantize + W8A8, 2: quantize only, 3: W8A8 only.
    // Every projection includes the same row ring; quantize alone has none.
    const int arms = breakdown ? 4 : 2;
    const int ring_kernels = partition == 0 ? 1 : 0;
    const std::array<size_t, 4> kernel_counts{
        static_cast<size_t>(projections + ring_kernels + (has_producer ? 1 : 0)),
        static_cast<size_t>(1 + projections + ring_kernels),
        1, static_cast<size_t>(projections + ring_kernels)};
    auto times = at::empty({runs, arms}, at::TensorOptions().dtype(at::kDouble).device(at::kCPU));
    auto* samples = times.data_ptr<double>();
    const auto launch = [&](int which) {
        if (has_producer && which == 0) baseline_norm();
        if (which == 1 || which == 2) {
            if (has_producer) fused_norm_quantize();
            else rpu_launch_dynamic_quantize_w8a8_spm_kernel(x_addr, q_addr, sx_addr, m, local_k, cores);
        }
        if (which != 2) {
            for (int64_t projection = 0; projection < projections; ++projection) {
                const auto projection_y = projection_y_addrs[projection];
                if (which == 0) {
                    rpu_launch_linear_spm_to_spm_acc16_kernel(
                        norm_addr, weights[projection], projection_y, m, n, k, partition,
                        cores, bias_addr, scales[projection], 0, 0, a_acc32);
                } else if (tile_probe) {
                    launch_tile_probe(explicit_descriptor, weights[projection], scales[projection]);
                } else {
                    rpu_launch_linear_w8a8_spm_kernel(
                        q_addr, sx_addr, weights[projection], scales[projection], projection_y,
                        m, n, k, partition, cores, bias_addr);
                }
            }
        }
        if (partition == 0 && which != 2) {
            if (gelu)
                rpu_launch_pi05_vision_compact_residual_spm_kernel(y_addr, zero_addr, final_addr, m);
            else
                rpu_launch_all_reduce_sum_residual_kernel(y_addr, zero_addr, final_addr, m, n, cores, cores);
        }
    };
    // This scoped diagnostic owns stable SPM and DDR buffers until all entries
    // retire. It does not enable a production FMB census or alter its rules.
    // Declaration order retires the cache before temporary SPM, including on error.
    GraphRuntimePolicy runtime_policy;
    runtime_policy.execution_core_count = static_cast<int>(cores);
    std::unique_ptr<RpuGraphCache> cache;
    std::array<GraphSignature, 4> signatures;
    std::array<RpuKernelGraph*, 4> graphs{};
    std::array<uint64_t, 4> generations{}, topologies{}, submits{}, replay_counts{};
    if (retained_graph) {
        cache = std::make_unique<RpuGraphCache>(arms, runtime_policy);
        for (int which = 0; which < arms; ++which) {
            auto& signature = signatures[which];
            signature.op_id = has_producer ? INT64_C(0x573841384C4E) + producer : INT64_C(0x573841384142);
            signature.op_id_str = has_producer ? "w8a8_producer_pipeline_bench" : "w8a8_resident_pipeline_bench";
            signature.shapes = {m, n, k};
            signature.dyn_dims = {which, partition, cores, bias.has_value(), a_acc32, projections, producer};
            if (tile_probe) signature.dyn_dims.push_back(tile_ordinal);
            signature.dtypes = {static_cast<uint8_t>(at::kHalf),
                                static_cast<uint8_t>(at::kChar)};
            graphs[which] = &cache->get_or_create(signature);
        }
    }
    const auto execute = [&](int which, bool build = false) {
        if (!retained_graph) {
            launch(which);
            return;
        }
        auto& graph = *graphs[which];
        try {
            graph.begin(signatures[which]);
            TORCH_CHECK(graph.state() == (build ? RpuKernelGraph::State::RECORDING
                                               : RpuKernelGraph::State::REPLAYING),
                        "W8A8 bench requires exactly one BUILD per side then REPLAY");
            if (build) launch(which);
            else graph.skip_op_stream_for_fast_replay();
            graph.end();  // Retained synchronous batch; all buffers remain live.
        } catch (...) {
            if (RpuKernelGraph::is_on_active_stack(graph)) graph.abort();
            throw;
        }
    };
    // Bookkeeping and lifecycle checks stay outside every measured interval.
    const auto check_graph = [&](int which, bool build = false, int64_t replays = 1) {
        if (!retained_graph) return;
        auto& graph = *graphs[which];
        if (build) {
            cache->touch_entry(signatures[which], false);
        } else {
            for (int64_t replay = 0; replay < replays; ++replay)
                cache->touch_entry(signatures[which], true);
            replay_counts[which] += replays;
        }
        const auto& stats = graph.debug_stats();
        if (build) {
            generations[which] = graph.build_generation();
            topologies[which] = graph.build_topology_hash();
        }
        TORCH_CHECK(graph.state() == RpuKernelGraph::State::BUILT && graph.replayable() &&
                        stats.kernel_count == kernel_counts[which] &&
                        stats.segment_count > 0 && topologies[which] != 0 &&
                        graph.build_generation() == generations[which] &&
                        graph.build_topology_hash() == topologies[which] &&
                        stats.hw_batch_submit_total > submits[which] &&
                        cache->size() == static_cast<size_t>(arms) && cache->cache_invariant_ok(),
                    "W8A8 bench lost stable retained Graph execution");
        submits[which] = stats.hw_batch_submit_total;
        bool found = false;
        for (const auto& entry : cache->snapshot()) {
            if (entry.signature == signatures[which]) {
                found = true;
                TORCH_CHECK(entry.replay_count == replay_counts[which] &&
                                entry.recapture_count == 0,
                            "W8A8 bench replay/recapture counters drifted");
            }
        }
        TORCH_CHECK(found, "W8A8 bench retained entry disappeared");
    };
    if (retained_graph) {
        // B and quant-only BUILD generate real Q/SX before PL-only BUILD.
        for (int which = 0; which < arms; ++which) {
            execute(which, true);
            check_graph(which, true);
        }
    } else if (breakdown) {
        // Make PL-only's input explicit even before the first warmup visit.
        execute(2);
    }
    // Each arm runs one contiguous W2 -> R5 block, giving every timed call the
    // same predecessor entry instead of mixing same-entry and switched-entry
    // launches in an odd-sample median.
    // In Graph mode the earlier BUILD is additional and untimed. Bookkeeping,
    // allocation and output reads stay outside each complete measurement block.
    for (int which = 0; which < arms; ++which) {
        for (int64_t visit = 0; visit < warmup; ++visit) {
            execute(which);
        }
        for (int64_t visit = 0; visit < runs; ++visit) {
            const auto begin = std::chrono::steady_clock::now();
            execute(which);  // Synchronous immediate or retained Graph path.
            const auto end = std::chrono::steady_clock::now();
            samples[arms * visit + which] =
                std::chrono::duration<double, std::nano>(end - begin).count();
        }
        check_graph(which, false, warmup + runs);
        check_guards();  // Includes compact raw parity after each complete replay block.
    }
    // One separate untimed diagnostic launch per projection. Sharing output storage
    // keeps C400 row within SPM and avoids CPU reads between measured A/B calls.
    // Breakdown additionally checks quant-only -> PL-only against complete B.
    const c10::Half nan(std::numeric_limits<float>::quiet_NaN());
    at::Tensor output_pl;
    if (breakdown) output_pl = at::empty_like(output_b);
    for (int slot = 0; slot < (breakdown ? 3 : 2); ++slot) {
        const int which = slot == 2 ? 3 : slot;
        if (which == 3) {
            execute(2);
            check_graph(2);
        }
        for (int64_t projection = 0; projection < projections; ++projection) {
            for (int core = 0; core < cores; ++core) {
                auto* partial = static_cast<c10::Half*>(
                    SPM_ALLOC.cpu_ptr(core, y_off + projection * y_bytes));
                std::fill(partial, partial + m * local_n, nan);
                auto* output = static_cast<c10::Half*>(
                    SPM_ALLOC.cpu_ptr(core, final_off + projection * y_bytes));
                std::fill(output, output + m * local_n, nan);
            }
        }
        execute(which);
        check_graph(which);
        auto* dst = (which == 0 ? output_a : which == 1 ? output_b : output_pl).data_ptr<float>();
        if (partition == 0) {
            const auto* src = static_cast<const c10::Half*>(SPM_ALLOC.cpu_ptr(0, final_off));
            if (gelu) {
                for (int core = 1; core < cores; ++core)
                    TORCH_CHECK(std::memcmp(src, SPM_ALLOC.cpu_ptr(core, final_off), m * n * 2) == 0,
                                "W8A8 GELU/FC2 bench: compact residual full output differs across cores");
            }
            for (int64_t index = 0; index < m * n; ++index) dst[index] = static_cast<float>(src[index]);
        } else {
            for (int64_t projection = 0; projection < projections; ++projection) {
                for (int core = 0; core < cores; ++core) {
                    const auto* src = static_cast<const c10::Half*>(
                        SPM_ALLOC.cpu_ptr(core, final_off + projection * y_bytes));
                    for (int64_t row = 0; row < m; ++row) {
                        for (int64_t column = 0; column < local_n; ++column) {
                            dst[projection * m * n + row * n + core * local_n + column] =
                                static_cast<float>(src[row * local_n + column]);
                        }
                    }
                }
            }
        }
        TORCH_CHECK(at::isfinite(which == 0 ? output_a : which == 1 ? output_b : output_pl).all().item<bool>(),
                    "W8A8 bench: nonfinite output");
    }
    if (breakdown) {
        TORCH_CHECK(at::equal(output_pl, output_b),
                    "W8A8 bench: standalone quantize -> PL differs from complete B");
    }
    if (retained_graph && (breakdown || projections == 3)) {
        const std::array<const char*, 4> names = owner_norm
            ? std::array<const char*, 4>{"owner_ln_w8a16", "owner_ln_quantize_w8a8",
                                        "owner_ln_quantize_only", "w8a8_only"}
            : gelu
            ? std::array<const char*, 4>{"gelu_w8a16_compact_residual", "gelu_quantize_w8a8_compact_residual",
                                        "gelu_quantize_only", "w8a8_compact_residual_only"}
            : layer_norm
            ? std::array<const char*, 4>{"layer_norm_w8a16", "layer_norm_quantize_w8a8",
                                        "layer_norm_quantize_only", "w8a8_only"}
            : std::array<const char*, 4>{"w8a16", "quantize_w8a8", "quantize_only", "w8a8_only"};
        for (int which = 0; which < arms; ++which) {
            const auto& stats = graphs[which]->debug_stats();
            std::cout << "W8A8_GRAPH_ENTRY arm=" << names[which]
                      << " projections=" << projections
                      << " kernels=" << stats.kernel_count
                      << " segments=" << stats.segment_count
                      << " hw_submits=" << stats.hw_batch_submit_total
                      << " replays=" << replay_counts[which] << std::endl;
        }
    }
    check_guards();
    if (cache) cache->clear();
    return {output_a, output_b, times};
}
}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> rpu_test_w8a8_pipeline_bench(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& scale,
    const c10::optional<at::Tensor>& bias, int64_t partition, int64_t cores,
    int64_t warmup, int64_t runs, bool a_acc32, bool retained_graph, bool breakdown,
    int64_t projections, int64_t tile_ordinal) {
    return pipeline_bench_impl(x, weight, scale, bias, partition, cores, warmup, runs,
                              a_acc32, retained_graph, breakdown, projections,
                              c10::nullopt, c10::nullopt, 0.0, nullptr, 0,
                              c10::nullopt, tile_ordinal);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> rpu_test_w8a8_norm_pipeline_bench(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& scale,
    const c10::optional<at::Tensor>& bias, const at::Tensor& gamma, const at::Tensor& beta,
    double eps, int64_t cores, int64_t warmup, int64_t runs,
    bool a_acc32, bool retained_graph, bool breakdown) {
    at::Tensor norm_reference;
    auto [a, b, times] = pipeline_bench_impl(
        x, weight, scale, bias, 1, cores, warmup, runs, a_acc32, retained_graph,
        breakdown, 3, gamma, beta, eps, &norm_reference);
    return {a, b, times, norm_reference};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> rpu_test_w8a8_producer_pipeline_bench(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& scale,
    const c10::optional<at::Tensor>& bias, const at::Tensor& residual,
    const c10::optional<at::Tensor>& gamma, const c10::optional<at::Tensor>& beta,
    const std::string& producer, double eps, int64_t warmup, int64_t runs,
    bool retained_graph, bool breakdown) {
    TORCH_CHECK(producer == "gelu_tanh" || producer == "owner_layer_norm",
                "W8A8 producer bench: expected gelu_tanh or owner_layer_norm");
    const bool owner = producer == "owner_layer_norm";
    TORCH_CHECK(gamma.has_value() == owner && beta.has_value() == owner,
                "W8A8 producer bench: affine gamma/beta are required only for owner LayerNorm");
    at::Tensor reference;
    auto [a, b, times] = pipeline_bench_impl(x, weight, scale, bias, owner ? 1 : 0,
        8, warmup, runs, false, retained_graph, breakdown, 1, gamma, beta, eps,
        &reference, owner ? 3 : 2, residual);
    return {a, b, times, reference};
}
