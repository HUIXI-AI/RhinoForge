// src/fused/rpu_pi05_model.cpp
// Pi0.5 denoise-loop driver op (Phase 3 — not a FusedModelBase subclass per Q2 / D-05).
// Current scope: internal component API, NOT the public Pi05Policy executor.
// The Python policy uses pi05_denoise_{loop,step}_forward or its AdaRMS baseline.
// This driver omits that baseline's final AdaRMS norm and CPU-FP32 tail, and
// its scalar prefix_len matches logical RoPE only for an unpadded prefix.
// The historical design sketch below is not a public-policy parity claim.
//
// Architecture (iteration 5 / CR-R5 corrections to D-CR3-SHAPE1):
//   Python (Pi05Pytorch.sample_actions — patched by Plan 03-03) pre-computes:
//     adarms_cond_list: [N_steps, B, width]  (via time_mlp_in/out — safe: no x_t dep)
//     attention_mask:   [B, 1, S, S]         (4-D causal+pad; invariant across steps)
//     k_caches_list, v_caches_list: TensorList of PREFIX KV (from VLM prefill; read-only)
//   Then calls torch.ops.rpu.pi05_forward ONCE.
//   This C++ op runs the N-step denoise for-loop internally:
//     for (step = 0; step < num_steps; ++step) {
//         action_emb = at::linear(x_t, action_in_proj_w_, action_in_proj_b_);      // Finding 1
//         cond_step  = adarms_cond_list.select(0, step).squeeze(0).contiguous();   // Finding 5
//
//         // CR-R5 BLOCKER 3: per-step .clone() of each prefix K/V entry — matches
//         //   lerobot modeling_pi05.py:889 past_key_values = copy.deepcopy(past_key_values).
//         std::vector<at::Tensor> k_step, v_step;
//         for (const auto& k : k_caches_list) k_step.emplace_back(k.clone());
//         for (const auto& v : v_caches_list) v_step.emplace_back(v.clone());
//
//         // CR-R5 BLOCKER 2: scalar position = prefix_len (constant across steps) —
//         //   lerobot's per-token positional offsets for the Pi0.5 suffix are
//         //   invariant across the denoise-step loop. AdaRMS advances RoPE
//         //   internally for the suffix from that scalar base.
//         suffix_out = rpu_adarms_forward(adarms_handle_, action_emb, cond_step,
//                                          k_step, v_step, attention_mask,
//                                          /*position=*/prefix_len, /*is_causal=*/false);
//
//         suffix_tail = suffix_out.index({.., Slice(-chunk_size_, None), ..});
//         v_t        = at::linear(suffix_tail, action_out_proj_w_, action_out_proj_b_);
//         x_t        = at::add(x_t, v_t, /*alpha=*/dt);   // lerobot:858
//
//         // k_step / v_step drop at scope end; allocator recycles for next step.
//     }
//     return x_t;
//
// Why NOT a v3 framework subclass (Q2 closes):
//   Pi05Model is an orchestration driver over subsystem public APIs rather
//   than a transformer layer stack with one SPM layout. The Python adapter
//   owns graph capture around those top-level calls; nested lifecycle scopes
//   are prohibited.
//
// Why action_in_proj / action_out_proj stay in C++ (Finding 1):
//   action_in_proj(x_t) and action_out_proj(suffix_tail) consume the EVOLVING
//   x_t / suffix_out. Python-side precompute would embed stale initial noise.
//   BOTH use at::linear (swizzle-aware via registered rpu_linear); a raw
//   matrix-multiply via a::matmul-style path is PROHIBITED (would corrupt
//   swizzled weights).
//
// Why time_mlp_* stays in Python:
//   The time_mlp_in → SiLU → time_mlp_out → SiLU pipeline consumes
//   time_feat[step] (from a static schedule), NOT x_t. Safe to precompute in
//   Python as `adarms_cond_list`. Keeps the hot per-step C++ body minimal.
//
// Why the prefix K/V is .clone()-ed per step (CR-R5 BLOCKER 3):
//   AdaRMS's kv-cache-insert kernel WRITES to the K/V buffer entries at the
//   suffix offset (position..position+suffix_len). Shallow-copy of at::Tensor
//   handles (iteration 3/4 plan) aliases the same RPU DDR storage — step N's
//   writes corrupt step N+1's prefix read. Lerobot's
//   `past_key_values = copy.deepcopy(past_key_values)` (modeling_pi05.py:889)
//   is the contract: fresh scratch per step.
//
// Why scalar `position=prefix_len` (CR-R5 BLOCKER 2):
//   Lerobot's per-token positional offsets tensor at modeling_pi05.py:883-884
//   is `prefix_offsets + cumsum(suffix_pad_masks) - 1`. For Pi0.5's
//   `suffix_pad_masks = ones(B, chunk_size)`, this collapses to
//   `[prefix_len, prefix_len+1, ..., prefix_len+chunk_size-1]`, which is
//   INVARIANT across the outer `for step` loop at line 829. The denoise-step
//   index never enters the stored positional values. AdaRMS's C API
//   (rpu_kernel_decls.h:802-810) accepts a scalar `int64_t position` and
//   advances RoPE internally from that offset for the contiguous suffix.
//   `position=prefix_len` is exactly right. Iteration 3/4's attempt to
//   increment the scalar by the step index was a straight BUG and is DELETED.

#include "rpu_kernel_decls.h"            // rpu_adarms_forward forward declaration
#include "model_handle_registry.h"

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <cstdint>
#include <vector>

// NOTE (Finding 2): NO cross-TU C++ class forward declaration for the Expert model.
// We call the C API rpu_adarms_forward (declared in rpu_kernel_decls.h) directly.
// This removes the cross-translation-unit class-boundary call that would require
// the full Expert class definition to link.


// =============================================================================
// Pi05Model — plain driver class (NOT a v3::FusedModelBase subclass).
//   Holds: adarms_handle, action_in_proj + action_out_proj weights, denoise loop config.
//   Runs:  forward() = C++ for-loop invoking rpu_adarms_forward() per step.
// =============================================================================
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
    // Shapes (all RPU fp16 unless noted):
    //   initial_noise:      [B=1, chunk_size, max_action_dim]      — x_t seed (RPU fp16)
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
                    "Pi05Model::forward: initial_noise must be on RPU device");
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
            // step N+1's prefix read (shared at::Tensor RPU storage).
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

        return x_t;   // RPU tensor (D-701 internal surface)
    }

    int64_t adarms_handle() const { return adarms_handle_; }

private:
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
