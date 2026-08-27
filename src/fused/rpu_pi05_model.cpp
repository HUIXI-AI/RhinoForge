// src/fused/rpu_pi05_model.cpp
// Pi0.5 denoise-loop driver op; not a FusedModelBase subclass.
//
// Architecture:
//   Python (Pi05Pytorch.sample_actions) pre-computes:
//     adarms_cond_list: [N_steps, B, width]  (via time_mlp_in/out — safe: no x_t dep)
//     attention_mask:   [B, 1, S, S]         (4-D causal+pad; invariant across steps)
//     k_caches_list, v_caches_list: TensorList of PREFIX KV (from VLM prefill; read-only)
//   Then calls torch.ops.rpu.pi05_forward ONCE.
//   This C++ op runs the N-step denoise for-loop internally:
//     for (step = 0; step < num_steps; ++step) {
//         action_emb = at::linear(x_t, action_in_proj_w_, action_in_proj_b_);
//         cond_step  = adarms_cond_list.select(0, step).squeeze(0).contiguous();
//
//         // Clone each prefix K/V entry for every step, matching the reference
//         // model's copy.deepcopy contract for past_key_values.
//         std::vector<at::Tensor> k_step, v_step;
//         for (const auto& k : k_caches_list) k_step.emplace_back(k.clone());
//         for (const auto& v : v_caches_list) v_step.emplace_back(v.clone());
//
//         // Scalar position = prefix_len and remains constant across steps.
//         //   lerobot's per-token positional offsets for the Pi0.5 suffix are
//         //   invariant across the denoise-step loop. AdaRMS advances RoPE
//         //   internally for the suffix from that scalar base.
//         suffix_out = rpu_adarms_forward(adarms_handle_, action_emb, cond_step,
//                                          k_step, v_step, attention_mask,
//                                          /*position=*/prefix_len, /*is_causal=*/false);
//
//         suffix_tail = suffix_out.index({.., Slice(-chunk_size_, None), ..});
//         v_t        = at::linear(suffix_tail, action_out_proj_w_, action_out_proj_b_);
//         x_t        = at::add(x_t, v_t, /*alpha=*/dt);   // Euler update
//
//         // k_step / v_step drop at scope end; allocator recycles for next step.
//     }
//     return x_t;
//
// Why this is not a FusedModelBase subclass:
//   Pi05Model is an orchestration driver over subsystem public APIs rather
//   than a transformer layer stack with one SPM layout. The Python adapter
//   owns graph capture around those top-level calls; nested lifecycle scopes
//   are prohibited.
//
// Why action_in_proj / action_out_proj stay in C++:
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
// Why the prefix K/V is cloned per step:
//   AdaRMS's kv-cache-insert kernel WRITES to the K/V buffer entries at the
//   suffix offset (position..position+suffix_len). Shallow-copy of at::Tensor
//   handles aliases the same RPU DDR storage — step N's writes corrupt step
//   N+1's prefix read. The reference model's copy.deepcopy contract requires
//   fresh scratch per step.
//
// Why scalar `position=prefix_len`:
//   The reference per-token positional offsets are
//   `prefix_offsets + cumsum(suffix_pad_masks) - 1`. For Pi0.5's
//   `suffix_pad_masks = ones(B, chunk_size)`, this collapses to
//   `[prefix_len, prefix_len+1, ..., prefix_len+chunk_size-1]`, which is
//   INVARIANT across the outer `for step` loop. The denoise-step index never
//   enters the stored positional values. AdaRMS's C API accepts a scalar
//   `int64_t position` and
//   advances RoPE internally from that offset for the contiguous suffix.
//   `position=prefix_len` is therefore the required contract.

#include "rpu_kernel_decls.h"            // rpu_adarms_forward forward declaration
#include "model_handle_registry.h"

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <cstdint>
#include <vector>

// There is no cross-TU C++ class forward declaration for the Expert model.
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

    // Primary entry — Python calls ONCE per select_action.
    //
    // Shapes (all RPU fp16 unless noted):
    //   initial_noise:      [B=1, chunk_size, max_action_dim]      — x_t seed (RPU fp16)
    //   adarms_cond_list:   [N_steps, B, width]                    — Python precomputed
    //   attention_mask:     [B, 1, S, S] fp16                      — 4-D causal+pad (optional)
    //   k_caches_list:      TensorList of prefix KV (one per Expert layer)   — read-only
    //   v_caches_list:      TensorList of prefix KV                          — read-only
    //   prefix_len:         int                                    — VLM prefix length
    //   num_steps:          int                                    — runtime value
    //   dt:                 float                                  — -1.0/num_steps
    //
    // No per-token positional-offset Tensor is needed. AdaRMS
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
        double dt)
    {
        TORCH_CHECK(num_steps > 0,
                    "Pi05Model::forward: num_steps must be > 0, got ", num_steps);
        TORCH_CHECK(prefix_len >= 0,
                    "Pi05Model::forward: prefix_len must be >= 0, got ", prefix_len);
        TORCH_CHECK(initial_noise.device().type() == at::kPrivateUse1,
                    "Pi05Model::forward: initial_noise must be on RPU device");
        TORCH_CHECK(initial_noise.dim() == 3,
                    "Pi05Model::forward: initial_noise must be 3-D [B, chunk_size, max_action_dim]");
        // The denoise-loop driver supports batch size 1.
        TORCH_CHECK(initial_noise.size(0) == 1,
                    "Pi05Model::forward: batch_size must be 1; "
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
        // Keep the caller's tensor untouched.
        at::Tensor x_t = initial_noise.clone();

        using namespace at::indexing;

        for (int64_t step = 0; step < num_steps; ++step) {
            // Re-compute action_emb from the evolving x_t.
            // at::linear dispatches via registered aten::linear → rpu_linear
            // (swizzle-aware). A raw matrix-multiply via a::matmul-style path
            // is PROHIBITED.
            at::Tensor action_emb = at::linear(
                x_t,
                action_in_proj_w_,
                action_in_proj_b_.defined() ? action_in_proj_b_ : at::Tensor{});
            // action_emb: [B, chunk_size, width]

            // AdaRMS cond is 1-D [width]; squeeze the batch dimension.
            at::Tensor cond_step = adarms_cond_list.select(0, step)
                                                    .squeeze(0)
                                                    .contiguous();

            // Clone each prefix K/V entry explicitly.
            // Matches the reference copy.deepcopy contract:
            //   `past_key_values = copy.deepcopy(past_key_values)`
            // before each denoise_step call. AdaRMS writes suffix K/V slots at
            // position >= prefix_len; without .clone(), those writes corrupt
            // step N+1's prefix read (shared at::Tensor RPU storage).
            //
            // AdaRMS writes K/V entries, so cloning is unconditional.
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

            // --- Expert forward (AdaRMS) via C API ---
            // The scalar position is prefix_len.
            //
            // The reference per-token positional-offsets tensor is
            //   `prefix_offsets + cumsum(suffix_pad_masks) - 1`
            // For Pi0.5's `suffix_pad_masks = ones(B, chunk_size)`, this is
            //   [prefix_len, prefix_len+1, ..., prefix_len+chunk_size-1]
            // INVARIANT across the outer `for step` loop — the
            // denoise-step index never enters the stored offsets. AdaRMS's C API
            // takes SCALAR `int64_t position` and
            // advances RoPE from that offset for the contiguous suffix. That's
            // exactly what `position=prefix_len` provides.
            //
            at::Tensor suffix_out = rpu_adarms_forward(
                adarms_handle_,
                action_emb,
                cond_step,
                k_caches_step,
                v_caches_step,
                attention_mask,
                /*position=*/prefix_len,     // Scalar base offset; step does not enter this arg.
                /*is_causal=*/false);

            // --- Tail: slice last chunk_size_, action_out_proj, integrator ---
            at::Tensor suffix_tail = suffix_out.index({Slice(), Slice(-chunk_size_, None), Slice()});

            // action_out_proj consumes the evolved suffix_out.
            // at::linear (swizzle-aware).
            at::Tensor v_t = at::linear(
                suffix_tail,
                action_out_proj_w_,
                action_out_proj_b_.defined() ? action_out_proj_b_ : at::Tensor{});

            // Euler integrator: x_t = x_t + dt * v_t  (lerobot:858)
            x_t = at::add(x_t, v_t, /*alpha=*/dt);

            // k_caches_step / v_caches_step drop here; allocator recycles.
        }

        return x_t;   // RPU tensor returned by the driver.
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
// Registry + public C API, mirroring the rpu_adarms_* API.
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

// No per-token positional-offsets Tensor argument is required.
at::Tensor rpu_pi05_forward(
    int64_t handle,
    const at::Tensor& initial_noise,
    const at::Tensor& adarms_cond_list,
    const std::optional<at::Tensor>& attention_mask,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    int64_t prefix_len,
    int64_t num_steps,
    double dt)
{
    return Pi05Registry::get(handle, "rpu_pi05_forward")->forward(
        initial_noise, adarms_cond_list, attention_mask,
        k_caches_list, v_caches_list, prefix_len, num_steps, dt);
}
