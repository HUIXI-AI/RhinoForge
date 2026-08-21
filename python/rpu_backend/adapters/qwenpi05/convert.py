"""
QwenPI05 Action Model Converter for RPU (op-by-op)

Converts a QwenPI05 action expert (ActionExpert + ActionIO) to run
on RPU via individual op dispatch. No fused decoder layer — each
Linear/RMSNorm/SDPA/RoPE op dispatches independently to RPU kernels.

Architecture handled:
  - QwenPI05ActionExpert: 10 transformer layers
    - AdaRMSNorm (Qwen3VLTextRMSNorm + cond Linear for scale/shift/gate)
    - QwenPI05ActionAttention (GQA: 16 heads, 8 kv_heads, head_dim 128)
    - Qwen3VLTextMLP (SiLU-gated: gate_proj, up_proj, down_proj)
    - Gated residual connections
  - QwenPI05ActionIO: action/state/timestep projections

Usage:
    from rpu_backend.adapters.qwenpi05.convert import convert_qwenpi05_action_model_for_rpu

    action_expert, action_io = convert_qwenpi05_action_model_for_rpu(
        action_expert, action_io
    )
    # action_expert and action_io are now on RPU with converted weights
"""

import copy

import torch
import torch.nn as nn


def _patch_rmsnorm_for_rpu():
    """Patch Qwen3VLTextRMSNorm to use RPU kernel when on RPU device.

    The standard RMSNorm upcasts to float32, which is inefficient on RPU.
    The patched version uses torch.ops.rpu.rms_norm for RPU tensors and
    falls back to the original implementation for CPU/CUDA.

    Safe to call multiple times (idempotent via _rpu_patched flag).
    """
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextRMSNorm,
        )
    except ImportError:
        print("[QwenPI05-RPU] Warning: Qwen3VLTextRMSNorm not available, skipping patch")
        return

    if getattr(Qwen3VLTextRMSNorm, '_rpu_patched', False):
        return

    def rpu_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.device.type == "rpu":
            return torch.ops.rpu.rms_norm(
                hidden_states,
                [self.weight.shape[0]],
                self.weight,
                self.variance_epsilon,
            )
        else:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return (self.weight * hidden_states).to(input_dtype)

    Qwen3VLTextRMSNorm.forward = rpu_forward
    Qwen3VLTextRMSNorm._rpu_patched = True
    print("[QwenPI05-RPU] Patched Qwen3VLTextRMSNorm for RPU dispatch")


class _RPURotaryEmbWrapper(nn.Module):
    """Wrapper that computes M-RoPE on CPU, then transfers cos/sin to RPU.

    The Qwen3-VL M-RoPE forward uses float32 matmul and autocast context
    managers that don't work on RPU. This wrapper keeps the computation
    on CPU (it's cheap — just frequency table lookup) and moves the
    output to match the caller's device/dtype.

    The inner CPU rotary_emb is stored via object.__setattr__ to prevent
    nn.Module.to() / .half() from recursively moving it off CPU.
    """

    def __init__(self, cpu_rotary_emb: nn.Module):
        super().__init__()
        # Bypass nn.Module.__setattr__ so _cpu_emb is NOT registered as
        # a submodule. This prevents .half() / .to("rpu") from touching it.
        object.__setattr__(self, '_cpu_emb', cpu_rotary_emb)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        target_device = x.device
        target_dtype = x.dtype
        # Compute on CPU (float32 matmul + autocast are reliable there)
        dummy_x = torch.zeros(1, dtype=torch.float32, device="cpu")
        pos_cpu = position_ids.to("cpu")
        cos, sin = self._cpu_emb(dummy_x, pos_cpu)
        return cos.to(device=target_device, dtype=target_dtype), \
               sin.to(device=target_device, dtype=target_dtype)


def _clone_rotary_emb_for_rpu(rotary_emb: nn.Module) -> nn.Module:
    """Create an RPU-compatible rotary embedding wrapper.

    The action expert shares rotary_emb with the VLM (which stays on CPU).
    We clone the underlying module and wrap it so that M-RoPE computation
    stays on CPU (cheap, reliable) while outputs are transferred to RPU.
    """
    cloned = copy.deepcopy(rotary_emb)
    # Keep the clone on CPU — the wrapper handles device transfer
    return _RPURotaryEmbWrapper(cloned)


def convert_qwenpi05_action_model_for_rpu(
    action_expert,
    action_io,
    *,
    patch_rmsnorm: bool = True,
):
    """Convert QwenPI05 action model for RPU inference (op-by-op dispatch).

    Steps:
      1. Patch Qwen3VLTextRMSNorm for RPU kernel dispatch
      2. Clone rotary embedding to RPU (decouple from VLM's CPU copy)
      3. Convert to fp16
      4. Convert linear weights for RPU layout (row/col partition swizzle)
      5. Move modules to RPU device

    Args:
        action_expert: QwenPI05ActionExpert instance (will be modified in-place).
        action_io: QwenPI05ActionIO instance (will be modified in-place).
        patch_rmsnorm: Whether to patch Qwen3VLTextRMSNorm class. Set False
                       if already patched elsewhere.

    Returns:
        Tuple of (action_expert, action_io) on RPU device with converted weights.
    """
    from rpu_backend.runtime.weights import convert_linear_weights_inplace

    # Fail loudly on double-conversion. `convert_
    # linear_weights_inplace` mutates Linear.weight.data irreversibly; a
    # second call on an already-swizzled module double-swizzles the
    # weights. Stamp a per-module flag on the first successful conversion
    # and raise on subsequent invocations, including calls made through this
    # converter directly or through Pi0.5 integration code.
    if getattr(action_expert, "_rpu_swizzled", False) or \
       getattr(action_io, "_rpu_swizzled", False):
        raise RuntimeError(
            "convert_qwenpi05_action_model_for_rpu: action_expert and/or "
            "action_io already have _rpu_swizzled=True from a prior call. "
            "convert_linear_weights_inplace mutates weights irreversibly; "
            "a second conversion would double-swizzle the weights. Reload the "
            "action model fresh before retrying."
        )
    if getattr(action_expert, "_rpu_swizzle_started", False) or \
       getattr(action_io, "_rpu_swizzle_started", False):
        raise RuntimeError(
            "convert_qwenpi05_action_model_for_rpu: prior conversion "
            "attempt started mutation but did not complete. Weights are "
            "in an undefined state. Reload the action model fresh."
        )

    # Step 1: Patch RMSNorm class for RPU
    if patch_rmsnorm:
        _patch_rmsnorm_for_rpu()

    # Step 2: Clone rotary embedding to RPU
    if hasattr(action_expert, 'qwen_rotary_emb'):
        action_expert.qwen_rotary_emb = _clone_rotary_emb_for_rpu(
            action_expert.qwen_rotary_emb
        )
        print(f"[QwenPI05-RPU] Cloned rotary_emb to RPU")

    # Step 3: Convert to fp16
    action_expert = action_expert.half()
    action_io = action_io.half()

    # Step 4: Convert linear weights for RPU layout
    # This swizzles weight matrices for optimal RPU GEMV/GEMM partitioning.
    # The `cond` Linear in AdaRMSNorm is converted normally (unlike Pi0.5's
    # `dense` which is skipped for separate expand+swizzle in fused mode).
    # Stamp _rpu_swizzle_started before mutation so a failure
    # mid-way trips the guard at the top of this function on retry.
    action_expert._rpu_swizzle_started = True
    action_io._rpu_swizzle_started = True
    expert_info = convert_linear_weights_inplace(action_expert)
    io_info = convert_linear_weights_inplace(action_io)

    n_expert = sum(1 for v in expert_info.values() if v >= 0)
    n_io = sum(1 for v in io_info.values() if v >= 0)
    n_fallback = sum(1 for v in {**expert_info, **io_info}.values() if v < 0)
    print(f"[QwenPI05-RPU] Converted {n_expert} expert + {n_io} IO linear layers "
          f"({n_fallback} fallback)")

    # Step 5: Move to RPU device
    action_expert = action_expert.to("rpu")
    action_io = action_io.to("rpu")
    print("[QwenPI05-RPU] Moved action model to RPU device")

    # Stamp _rpu_swizzled only on full success (after weight
    # swizzle + device move). A retry finds this flag and raises cleanly.
    action_expert._rpu_swizzled = True
    action_io._rpu_swizzled = True

    return action_expert, action_io


def run_qwenpi05_action_on_rpu(
    action_expert,
    action_io,
    *,
    prefix_key_values,
    prefix_mask: torch.Tensor,
    state: torch.Tensor | None,
    x_t: torch.Tensor,
    time: torch.Tensor,
    prefix_layer_offset: int,
):
    """Run a single QwenPI05 action forward pass with model on RPU.

    Handles device transfer: inputs on CPU → RPU for expert → CPU for decode.

    Args:
        action_expert: QwenPI05ActionExpert on RPU.
        action_io: QwenPI05ActionIO on RPU.
        prefix_key_values: List of (K, V) tuples from Qwen3-VL prefix encoding.
                           Can be on any device (will be transferred to RPU).
        prefix_mask: Bool mask for prefix tokens, shape (B, P).
        state: Optional state vector, shape (B, state_dim) or (B, 1, state_dim).
        x_t: Noisy action tensor, shape (B, H, action_dim), float32.
        time: Diffusion timestep, shape (B,), float32 in [0, 1].
        prefix_layer_offset: Layer offset for prefix KV indexing.

    Returns:
        Predicted velocity tensor, shape (B, H, action_dim), float32 on CPU.
    """
    device = torch.device("rpu")
    dtype = torch.float16

    # Transfer inputs to RPU in fp16
    x_t_rpu = x_t.to(device=device, dtype=dtype)
    # Keep timestep on CPU: create_sinusoidal_pos_embedding uses float64
    # on non-CPU devices, which RPU doesn't support. The sinusoidal
    # computation is cheap; embed_suffix transfers the result to RPU via
    # `.to(action_embeds.device, action_embeds.dtype)`.
    time_cpu = time.to(device="cpu", dtype=torch.float32)
    state_rpu = state.to(device=device, dtype=dtype) if state is not None else None
    # Keep prefix_mask on CPU — make_suffix_attn_mask and
    # bool_mask_to_attention_bias compute on CPU internally.
    prefix_mask_cpu = prefix_mask.to(device="cpu")

    # Transfer prefix K/V to RPU
    prefix_kv_rpu = []
    for k, v in prefix_key_values:
        prefix_kv_rpu.append((
            k.to(device=device, dtype=dtype),
            v.to(device=device, dtype=dtype),
        ))

    # Embed suffix (action_io on RPU)
    suffix_embeds, suffix_mask, adarms_cond = action_io.embed_suffix(
        state_rpu, x_t_rpu, time_cpu
    )

    # Compute suffix position IDs on CPU (sum/cumsum trigger RPU fallbacks)
    prefix_offsets = prefix_mask.long().sum(dim=-1)[:, None]
    suffix_pos = prefix_offsets + torch.cumsum(suffix_mask.cpu().long(), dim=1) - 1
    position_ids = suffix_pos[None, ...].expand(3, -1, -1).to(device)

    # Run action expert (on RPU).
    # prefix_mask and suffix_mask stay on CPU — make_suffix_attn_mask
    # and bool_mask_to_attention_bias compute on CPU and transfer results.
    suffix_hidden = action_expert(
        suffix_embeds,
        prefix_key_values=prefix_kv_rpu,
        prefix_mask=prefix_mask_cpu,
        suffix_mask=suffix_mask.cpu(),
        position_ids=position_ids,
        adarms_cond=adarms_cond,
        prefix_layer_offset=prefix_layer_offset,
    )

    # Decode actions (move to CPU for float32 projection)
    suffix_hidden_cpu = suffix_hidden.to(device="cpu", dtype=torch.float32)
    action_horizon = action_io.config.action_horizon
    action_hidden = suffix_hidden_cpu[:, -action_horizon:]
    # action_out_proj is on RPU in fp16, but we need fp32 output.
    # Move the slice to RPU, project, then back to CPU.
    action_hidden_rpu = action_hidden.to(device=device, dtype=dtype)
    pred = action_io.action_out_proj(action_hidden_rpu)
    return pred.to(device="cpu", dtype=torch.float32)


def predict_action_rpu(
    action_expert,
    action_io,
    *,
    prefix_key_values,
    prefix_mask: torch.Tensor,
    state: torch.Tensor | None,
    action_horizon: int,
    action_dim: int,
    prefix_layer_offset: int,
    num_steps: int = 10,
) -> torch.Tensor:
    """Run full DDIM denoising loop on RPU.

    Args:
        action_expert: QwenPI05ActionExpert on RPU.
        action_io: QwenPI05ActionIO on RPU.
        prefix_key_values: Prefix KV cache (any device).
        prefix_mask: (B, P) bool mask.
        state: Optional state (B, state_dim).
        action_horizon: Number of action steps to predict.
        action_dim: Dimension of each action step.
        prefix_layer_offset: Layer offset for prefix KV.
        num_steps: DDIM denoising steps.

    Returns:
        Predicted actions, shape (B, action_horizon, action_dim), float32 CPU.
    """
    bsz = prefix_mask.shape[0]
    x_t = torch.randn(bsz, action_horizon, action_dim, dtype=torch.float32)
    dt = -1.0 / num_steps
    time = torch.ones(bsz, dtype=torch.float32)

    for _ in range(num_steps):
        pred_velocity = run_qwenpi05_action_on_rpu(
            action_expert,
            action_io,
            prefix_key_values=prefix_key_values,
            prefix_mask=prefix_mask,
            state=state,
            x_t=x_t,
            time=time,
            prefix_layer_offset=prefix_layer_offset,
        )
        x_t = x_t + dt * pred_velocity
        time = (time + dt).clamp(min=0.0)

    return x_t
