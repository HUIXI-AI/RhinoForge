// Internal Pi0.5 pre-final-norm component; the public policy uses its own executor.
// Eager execution supports N denoise steps. Graph execution supports one step
// and one driver invocation per capture, with one dedicated AdaRMS child.
// Graph copies and projections use persistent workspaces; returned tensors are fresh.
// Caller prefix KV stays read-only: scratch copies receive the child's suffix writes.
// Time conditions are supplied by the caller. Projections retain FP16 TP8/ACC32.
// This component omits the public policy's final AdaRMS norm and CPU-FP32 tail.
// The scalar prefix position matches logical RoPE only for an unpadded prefix.

#include "rpu_kernel_decls.h"            // rpu_adarms_forward forward declaration
#include "model_handle_registry.h"
#include "graph/graph_runtime.h"
#include "core/fused_model_base.h"

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

class Pi05Model {
public:
    Pi05Model() = default;
    ~Pi05Model() = default;

    void set_weights(
        int64_t adarms_handle,
        const at::Tensor& action_in_proj_w,
        const at::Tensor& action_in_proj_b,
        const at::Tensor& action_out_proj_w,
        const at::Tensor& action_out_proj_b,
        int64_t chunk_size,
        int64_t max_action_dim,
        int64_t width)
    {
        TORCH_CHECK(chunk_size > 0,
                    "Pi05Model::set_weights: chunk_size must be > 0, got ", chunk_size);
        TORCH_CHECK(max_action_dim > 0 && width > 0,
                    "Pi05Model::set_weights: max_action_dim + width must be > 0");
        TORCH_CHECK(adarms_handle >= 0,
                    "Pi05Model::set_weights: adarms_handle must be >= 0, got ", adarms_handle);
        require_retired_graphs("Pi05Model::set_weights");
        adarms_handle_      = adarms_handle;
        action_in_proj_w_   = action_in_proj_w;
        action_in_proj_b_   = action_in_proj_b;
        action_out_proj_w_  = action_out_proj_w;
        action_out_proj_b_  = action_out_proj_b;
        chunk_size_         = chunk_size;
        max_action_dim_     = max_action_dim;
        width_              = width;
    }

    // Internal entry; no current public select_action caller.
    //
    // Shapes (all device fp16 unless noted):
    //   initial_noise:      [B=1, chunk_size, max_action_dim]      — x_t seed (device fp16)
    //   adarms_cond_list:   [N_steps, B, width]                    — Python precomputed
    //   attention_mask:     [B, 1, S, S] fp16                      — 4-D causal+pad (optional)
    //   k_caches_list:      TensorList of prefix KV (one per Expert layer)   — read-only
    //   v_caches_list:      TensorList of prefix KV                          — read-only
    //   prefix_len:         int                                    — VLM prefix length
    //   num_steps:          int                                    — Finding 8 (runtime)
    //   dt:                 float                                  — -1.0/num_steps
    //
    // CR-R5 BLOCKER 2: NO per-token positional-offset Tensor arg. AdaRMS
    //   advances RoPE from `position=prefix_len` (scalar) across the suffix
    //   internally.
    at::Tensor forward(
        const at::Tensor& initial_noise,
        const at::Tensor& adarms_cond_list,
        const std::optional<at::Tensor>& attention_mask,
        at::TensorList k_caches_list,
        at::TensorList v_caches_list,
        int64_t prefix_len,
        int64_t num_steps,
        double dt,
        at::IntArrayRef planned_stage_descriptor)
    {
        TORCH_CHECK(num_steps > 0,
                    "Pi05Model::forward: num_steps must be > 0, got ", num_steps);
        TORCH_CHECK(prefix_len >= 0,
                    "Pi05Model::forward: prefix_len must be >= 0, got ", prefix_len);
        TORCH_CHECK(initial_noise.device().type() == at::kPrivateUse1,
                    "Pi05Model::forward: initial_noise must be on device");
        TORCH_CHECK(initial_noise.dim() == 3,
                    "Pi05Model::forward: initial_noise must be 3-D [B, chunk_size, max_action_dim]");
        // Finding 5: batch=1 invariant for Phase 3 (matches D-17 + v2.15).
        TORCH_CHECK(initial_noise.size(0) == 1,
                    "Pi05Model::forward: batch_size must be 1 (Phase 3 single-batch invariant); "
                    "got batch_size=", initial_noise.size(0));
        TORCH_CHECK(adarms_cond_list.size(0) == num_steps,
                    "Pi05Model::forward: adarms_cond_list length (", adarms_cond_list.size(0),
                    ") != num_steps (", num_steps, ")");
        TORCH_CHECK(adarms_handle_ >= 0,
                    "Pi05Model::forward: adarms_handle_ not set (call set_weights first)");
        TORCH_CHECK(k_caches_list.size() == v_caches_list.size(),
                    "Pi05Model::forward: k_caches_list.size (", k_caches_list.size(),
                    ") != v_caches_list.size (", v_caches_list.size(), ")");

        if (graph_dma::active()) {
            TORCH_CHECK(num_steps == 1,
                        "Pi05Model Graph supports exactly one denoise step; "
                        "use the public denoise-loop executor for multiple steps");
            return forward_graph(initial_noise, adarms_cond_list, attention_mask,
                                 k_caches_list, v_caches_list, prefix_len, dt,
                                 planned_stage_descriptor);
        }

        // x_t evolves across steps. Start as the input noise (RPU).
        // P4 mitigation — caller's tensor stays untouched.
        at::Tensor x_t = initial_noise.clone();

        using namespace at::indexing;

        for (int64_t step = 0; step < num_steps; ++step) {
            // Finding 1 (CR-R3): re-compute action_emb from EVOLVING x_t.
            // at::linear dispatches via registered aten::linear → rpu_linear
            // (swizzle-aware). A raw matrix-multiply via a::matmul-style path
            // is PROHIBITED.
            at::Tensor action_emb = at::linear(
                x_t,
                action_in_proj_w_,
                action_in_proj_b_.defined() ? action_in_proj_b_ : at::Tensor{});
            // action_emb: [B, chunk_size, width]  (lerobot:703 action_time_emb)

            // Finding 5: adarms cond is 1-D [width]; squeeze batch dim.
            at::Tensor cond_step = adarms_cond_list.select(0, step)
                                                    .squeeze(0)
                                                    .contiguous();

            // ***CR-R5 BLOCKER 3 (F9 real fix): EXPLICIT per-entry .clone() of prefix K/V.***
            // Matches lerobot modeling_pi05.py:889
            //   `past_key_values = copy.deepcopy(past_key_values)`
            // before each denoise_step call. AdaRMS writes suffix K/V slots at
            // position >= prefix_len; without .clone(), those writes corrupt
            // step N+1's prefix read (shared at::Tensor device storage).
            //
            // NO "probe" conditional — AdaRMS MUST write K/V entries (that's
            // how kv-cache-insert kernels work). We commit to .clone() always.
            std::vector<at::Tensor> k_caches_step;
            std::vector<at::Tensor> v_caches_step;
            k_caches_step.reserve(k_caches_list.size());
            v_caches_step.reserve(v_caches_list.size());
            for (const auto& k : k_caches_list) {
                k_caches_step.emplace_back(k.clone());
            }
            for (const auto& v : v_caches_list) {
                v_caches_step.emplace_back(v.clone());
            }

            // --- Expert forward (AdaRMS) via C API — Finding 2 + CR-R5 BLOCKER 2 ---
            // CR-R5 BLOCKER 2 (F10 real fix): scalar position=prefix_len.
            //
            // Lerobot's per-token positional-offsets tensor at modeling_pi05.py:883-884 is
            //   `prefix_offsets + cumsum(suffix_pad_masks) - 1`
            // For Pi0.5's `suffix_pad_masks = ones(B, chunk_size)`, this is
            //   [prefix_len, prefix_len+1, ..., prefix_len+chunk_size-1]
            // INVARIANT across the outer `for step` loop at line 829 — the
            // denoise-step index never enters the stored offsets. AdaRMS's C API
            // (rpu_kernel_decls.h:802-810) takes SCALAR `int64_t position` and
            // advances RoPE from that offset for the contiguous suffix. That's
            // what `position=prefix_len` provides only for an unpadded prefix.
            // Public Pi05 handles logical RoPE and physical KV rows separately.
            //
            // Iteration 3/4 tried to bake the denoise-step index into the scalar
            // position arg (treated the step counter as a positional offset).
            // That was a bug. It is DELETED here.
            at::Tensor suffix_out = rpu_adarms_forward(
                adarms_handle_,
                action_emb,
                cond_step,
                k_caches_step,
                v_caches_step,
                attention_mask,
                /*position=*/prefix_len,     // CR-R5 BLOCKER 2: scalar base offset; the step index does NOT enter this arg.
                /*is_causal=*/false,
                planned_stage_descriptor);

            // --- Tail: slice last chunk_size_, action_out_proj, integrator ---
            at::Tensor suffix_tail = suffix_out.index({Slice(), Slice(-chunk_size_, None), Slice()});

            // Finding 1 continued: action_out_proj consumes EVOLVED suffix_out.
            // at::linear (swizzle-aware).
            at::Tensor v_t = at::linear(
                suffix_tail,
                action_out_proj_w_,
                action_out_proj_b_.defined() ? action_out_proj_b_ : at::Tensor{});

            // Euler integrator: x_t = x_t + dt * v_t  (lerobot:858)
            x_t = at::add(x_t, v_t, /*alpha=*/dt);

            // k_caches_step / v_caches_step drop here; allocator recycles.
        }

        return x_t;   // device tensor (D-701 internal surface)
    }

    int64_t adarms_handle() const { return adarms_handle_; }

    void require_retired_graphs(const char* operation) {
        retire_workspaces();
        TORCH_CHECK(graph_workspaces_.empty(), operation,
                    ": retire captured driver graphs before replacing weights or destroying the owner");
    }

private:
    static constexpr int64_t PI05_INPUT_LINEAR_SITE = 3560820799875199063LL;
    static constexpr int64_t PI05_OUTPUT_LINEAR_SITE = 1109120764477258114LL;

    struct GraphWorkspace {
        std::weak_ptr<const uint8_t> lifetime;
        v3::FmbPhysicalExecutionManifest projection_manifest;
        at::Tensor noise, action, suffix, velocity;
        std::vector<at::Tensor> keys, values;
        uint64_t action_base = 0, velocity_base = 0;
    };
    using WorkspaceKey = std::tuple<const RpuKernelGraph*, uint64_t, size_t>;
    std::map<WorkspaceKey, GraphWorkspace> graph_workspaces_;

    void retire_workspaces() {
        for (auto it = graph_workspaces_.begin(); it != graph_workspaces_.end();) {
            if (it->second.lifetime.expired()) it = graph_workspaces_.erase(it);
            else ++it;
        }
    }

    v3::FmbPhysicalExecutionManifest projection_manifest(int64_t rows, int64_t prefix_len) const {
        using namespace v3;
        TORCH_CHECK(prefix_len <= std::numeric_limits<int64_t>::max() - rows,
                    "Pi05Model Graph: prefix plus row count overflows");
        FmbPhysicalExecutionManifest manifest;
        manifest.state = FmbPhysicalManifestState::COMPLETE;
        manifest.logical_length = manifest.physical_length = rows;
        manifest.kv_logical_length = prefix_len + rows;
        manifest.kv_insert_physical_rows = rows;
        manifest.graph_lifecycle = FmbGraphLifecycle::COMPOSITE_CHILD;
        manifest.linear_accumulation = FmbLinearAccumulationPolicy::ACC32;
        const auto add_projection = [&](int64_t site, const at::Tensor& weight) {
            manifest.routes.push_back({site, FmbRouteFamily::LINEAR,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE), 0,
                {rows, weight.size(0), weight.size(1), 8, 1}, 0});
        };
        add_projection(PI05_INPUT_LINEAR_SITE, action_in_proj_w_);
        add_projection(PI05_OUTPUT_LINEAR_SITE, action_out_proj_w_);
        std::sort(manifest.routes.begin(), manifest.routes.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.site_id < rhs.site_id; });
        validate_fmb_physical_manifest(manifest, rows, prefix_len);
        return manifest;
    }

    static void require_tensor(const at::Tensor& tensor, const char* role) {
        TORCH_CHECK(tensor.defined() &&
                        tensor.device().type() == at::kPrivateUse1 &&
                        tensor.scalar_type() == at::kHalf && tensor.is_contiguous(),
                    "Pi05Model Graph: ", role, " must be contiguous device FP16");
    }

    // Copy nodes execute between their producer and consumer kernels, including
    // on REPLAY. An immediate copy_ here would read an unexecuted child output.
    // DDR_TO_DDR resolves both managed endpoints through the existing DMA guard.
    static void capture_copy(at::Tensor& destination, const at::Tensor& source) {
        TORCH_CHECK(destination.sizes() == source.sizes() &&
                        destination.scalar_type() == source.scalar_type() &&
                        destination.is_contiguous() && source.is_contiguous(),
                    "Pi05Model Graph: copy geometry changed");
        constexpr size_t max_bytes = size_t{8} << 20;
        const size_t bytes = source.nbytes();
        TORCH_CHECK(bytes > 0 && bytes % 32 == 0,
                    "Pi05Model Graph: copy size must be a positive multiple of 32 bytes");
        auto& graph = RpuKernelGraph::active();
        graph.keep_alive(source);
        graph.keep_alive(destination);
        rpu_ddr_flush_force_sized(source.data_ptr(), bytes);
        rpu_ddr_flush_force_sized(destination.data_ptr(), bytes);
        for (size_t offset = 0; offset < bytes; offset += max_bytes) {
            const size_t count = std::min(max_bytes, bytes - offset);
            auto* dst = static_cast<uint8_t*>(destination.data_ptr()) + offset;
            const auto* src = static_cast<const uint8_t*>(source.data_ptr()) + offset;
            graph.capture_data(MemcpyNodeData{
                dst, src, count, CopyKind::DDR_TO_DDR,
                ::rhino_lkn::RpuGetDevAddr(dst), ::rhino_lkn::RpuGetDevAddr(const_cast<uint8_t*>(src))});
        }
    }

    at::Tensor forward_graph(
        const at::Tensor& initial_noise, const at::Tensor& conditions,
        const std::optional<at::Tensor>& mask,
        at::TensorList keys, at::TensorList values,
        int64_t prefix_len, double dt, at::IntArrayRef descriptor) {
        require_tensor(initial_noise, "initial_noise");
        require_tensor(conditions, "conditions");
        TORCH_CHECK(initial_noise.size(1) == chunk_size_ &&
                        initial_noise.size(2) == max_action_dim_ &&
                        conditions.sizes() == at::IntArrayRef({1, 1, width_}),
                    "Pi05Model Graph: noise or condition shape does not match installed weights");
        TORCH_CHECK(!descriptor.empty(),
                    "Pi05Model Graph requires the complete child stage descriptor");
        for (const auto& tensor : keys) require_tensor(tensor, "key cache");
        for (const auto& tensor : values) require_tensor(tensor, "value cache");
        for (const auto& weight : {action_in_proj_w_, action_out_proj_w_}) {
            require_tensor(weight, "projection weight");
            TORCH_CHECK(weight.dim() == 2 && weight.size(0) % 128 == 0 &&
                            weight.size(1) % 16 == 0,
                        "Pi05Model Graph requires TP8 col-partition projection weights");
        }
        TORCH_CHECK(action_in_proj_w_.sizes() == at::IntArrayRef({width_, max_action_dim_}) &&
                        action_out_proj_w_.sizes() == at::IntArrayRef({max_action_dim_, width_}),
                    "Pi05Model Graph: projection shape mismatch");
        for (const auto& item : {std::make_pair(action_in_proj_b_, width_),
                                 std::make_pair(action_out_proj_b_, max_action_dim_)}) {
            if (!item.first.defined()) continue;
            require_tensor(item.first, "projection bias");
            TORCH_CHECK(item.first.dim() == 1 && item.first.numel() == item.second,
                        "Pi05Model Graph: projection bias shape mismatch");
        }

        // Cold RPU mask readback must precede all parent Graph nodes. A cached
        // mask is reused without host copies on READY replay.
        rpu_adarms_prepare_graph_mask(adarms_handle_, mask, chunk_size_, prefix_len, false);
        using namespace v3;
        const auto requested_manifest = projection_manifest(initial_noise.size(1), prefix_len);
        auto& graph = RpuKernelGraph::active();
        const auto stamp = graph.op_stream_stamp();
        retire_workspaces();
        const WorkspaceKey key{stamp.graph, stamp.build_generation, stamp.position};
        for (const auto& existing : graph_workspaces_) {
            TORCH_CHECK(std::get<0>(existing.first) != stamp.graph ||
                            std::get<1>(existing.first) != stamp.build_generation ||
                            std::get<2>(existing.first) == stamp.position,
                        "Pi05Model Graph permits only one driver invocation per capture");
        }
        auto it = graph_workspaces_.find(key);
        if (it == graph_workspaces_.end()) {
            TORCH_CHECK(graph.state() == RpuKernelGraph::State::RECORDING,
                        "Pi05Model Graph: REPLAY has no matching workspace");
            TORCH_CHECK(graph_workspaces_.size() < 64,
                        "Pi05Model Graph: at most 64 live driver invocations may be retained");
            GraphWorkspace fresh;
            fresh.lifetime = graph.ddr_buffer_lifetime();
            fresh.projection_manifest = requested_manifest;
            fresh.noise = at::empty(initial_noise.sizes(), initial_noise.options());
            fresh.action = at::empty({1, chunk_size_, width_}, initial_noise.options());
            fresh.suffix = at::empty_like(fresh.action);
            fresh.velocity = at::empty_like(fresh.noise);
            for (const auto& tensor : keys) fresh.keys.push_back(at::empty_like(tensor));
            for (const auto& tensor : values) fresh.values.push_back(at::empty_like(tensor));
            fresh.action_base = ::rhino_lkn::RpuGetDevAddr(fresh.action.data_ptr());
            fresh.velocity_base = ::rhino_lkn::RpuGetDevAddr(fresh.velocity.data_ptr());
            it = graph_workspaces_.emplace(key, std::move(fresh)).first;
        }
        auto& work = it->second;
        TORCH_CHECK(fmb_physical_manifest_fingerprint(requested_manifest) ==
                        fmb_physical_manifest_fingerprint(work.projection_manifest),
                    "Pi05Model Graph: projection manifest changed during REPLAY");
        FmbPhysicalManifestConsumer routes(work.projection_manifest);
        graph.record_branch(routes.fingerprint(), {}, "native_pi05_projection_manifest");
        TORCH_CHECK(work.keys.size() == keys.size() && work.values.size() == values.size(),
                    "Pi05Model Graph: cache count changed during REPLAY");
        capture_copy(work.noise, initial_noise);
        for (size_t index = 0; index < keys.size(); ++index) {
            capture_copy(work.keys[index], keys[index]);
            capture_copy(work.values[index], values[index]);
        }
        // Both operands remain owned by this invocation for the retained graph's
        // lifetime. The required destination base lives in the same stable map node.
        auto input_matrix = work.noise.view({chunk_size_, max_action_dim_});
        auto action_matrix = work.action.view({chunk_size_, width_});
        routes.consume_route(FmbRouteFamily::LINEAR, PI05_INPUT_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE), 0,
            {input_matrix.size(0), action_in_proj_w_.size(0), input_matrix.size(1), 8, 1});
        rpu_launch_linear_ddr_kernel(input_matrix, action_in_proj_w_, action_matrix,
                                    action_in_proj_b_, action_in_proj_b_.defined(), 1,
                                    &work.action_base, 0, nullptr, 8, {}, true);
        graph.keep_alive(conditions);
        if (mask && mask->defined()) graph.keep_alive(*mask);
        auto suffix = rpu_adarms_forward(adarms_handle_, work.action, conditions.view({width_}),
                                        work.keys, work.values, mask, prefix_len, false,
                                        descriptor);
        capture_copy(work.suffix, suffix);
        auto suffix_matrix = work.suffix.view({chunk_size_, width_});
        auto velocity_matrix = work.velocity.view({chunk_size_, max_action_dim_});
        routes.consume_route(FmbRouteFamily::LINEAR, PI05_OUTPUT_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE), 0,
            {suffix_matrix.size(0), action_out_proj_w_.size(0), suffix_matrix.size(1), 8, 1});
        rpu_launch_linear_ddr_kernel(suffix_matrix, action_out_proj_w_, velocity_matrix,
                                    action_out_proj_b_, action_out_proj_b_.defined(), 1,
                                    &work.velocity_base, 0, nullptr, 8, {}, true);
        routes.require_all_consumed();
        auto result = at::empty_like(work.noise);
        graph.keep_alive(result);
        // PI05_EULER_FIXED_KERNEL_BASIS: FP16 ADD with the caller's Euler dt;
        // no model-selected kernel or layout alternative exists at this site.
        rpu_launch_eltwise_binary_sameshape_kernel(
            work.noise, work.velocity, c10::Half(static_cast<float>(dt)), result, ValuOpType::ADD);
        return result;
    }

    int64_t adarms_handle_ = -1;

    at::Tensor action_in_proj_w_,  action_in_proj_b_;
    at::Tensor action_out_proj_w_, action_out_proj_b_;

    int64_t chunk_size_ = 0;
    int64_t max_action_dim_ = 0;
    int64_t width_ = 0;
};


// =============================================================================
// Registry + public C API (mirror of rpu_adarms_* at rpu_adarms_model.cpp:714-767).
// =============================================================================

using Pi05Registry = ModelHandleRegistry<Pi05Model>;

int64_t rpu_pi05_create() {
    return Pi05Registry::create();
}

void rpu_pi05_destroy(int64_t handle) {
    Pi05Registry::get(handle, "rpu_pi05_destroy")->require_retired_graphs("rpu_pi05_destroy");
    Pi05Registry::destroy(handle, "rpu_pi05_destroy");
}

void rpu_pi05_set_weights(
    int64_t handle,
    int64_t adarms_handle,
    const at::Tensor& action_in_proj_w,
    const at::Tensor& action_in_proj_b,
    const at::Tensor& action_out_proj_w,
    const at::Tensor& action_out_proj_b,
    int64_t chunk_size,
    int64_t max_action_dim,
    int64_t width)
{
    Pi05Registry::get(handle, "rpu_pi05_set_weights")->set_weights(
        adarms_handle,
        action_in_proj_w,  action_in_proj_b,
        action_out_proj_w, action_out_proj_b,
        chunk_size, max_action_dim, width);
}

// CR-R5 BLOCKER 2: NO per-token positional-offsets Tensor argument.
at::Tensor rpu_pi05_forward(
    int64_t handle,
    const at::Tensor& initial_noise,
    const at::Tensor& adarms_cond_list,
    const std::optional<at::Tensor>& attention_mask,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    int64_t prefix_len,
    int64_t num_steps,
    double dt,
    at::IntArrayRef planned_stage_descriptor)
{
    return Pi05Registry::get(handle, "rpu_pi05_forward")->forward(
        initial_noise, adarms_cond_list, attention_mask,
        k_caches_list, v_caches_list, prefix_len, num_steps, dt,
        planned_stage_descriptor);
}
