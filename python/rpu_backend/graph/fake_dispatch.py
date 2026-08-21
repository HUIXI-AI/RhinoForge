"""
rpu::* 自定义 op 的 Python dispatch fake shim。

背景:
  PT 2.10 的 dispatcher 在 FakeTensor 输入下走 op 时:
    - FakeTensor 的 keyset 含 Python(由 __torch_dispatch__ 协议自带)
    - 但我们的 rpu::* op 没注册 Python impl
    - dispatcher 跳过 Python → 落到 PrivateUse1 真实现
    - 真实现读 fake_tensor.data_ptr() → "data not allocated"

  C++ Meta dispatch 不解决这个问题 —— Meta key 只在 tensor.device='meta'
  时被选中,不影响 device='rpu' 的 FakeTensor。

  解法:给 rpu::* op 注册 Python dispatch impl。Python key 在 FakeTensor 的
  keyset 里,优先级高于 PrivateUse1。dispatcher 选 Python → 调本文件里的
  shim → 返回 fake output,不进真实 PrivateUse1 kernel。

不变量:
  - 真 RPU tensor 的 keyset **没有** Python key(只有 PrivateUse1 等 backend
    keys)。所以真实部署路径不受影响。
  - 一旦 shim 误中真 tensor(任何 wrapper 把 Python key 加进去),立刻 fail
    loud,不静默错算。

import 这个模块就触发注册;Python 包 __init__.py 在 .so 加载后 import 它。
"""
import torch
from torch._subclasses.fake_tensor import FakeTensor


def _has_fake(x):
    if isinstance(x, FakeTensor):
        return True
    if isinstance(x, (tuple, list)):
        return any(_has_fake(v) for v in x)
    if isinstance(x, dict):
        return any(_has_fake(v) for v in x.values())
    return False


def _require_fake(*args):
    """Python dispatch shim 只在 FakeTensor 路径上被调。一旦真
    tensor 进来就抛错,避免 silent 错误绕过真实 RPU 实现。"""
    if not any(_has_fake(a) for a in args):
        raise RuntimeError(
            "rpu Python dispatch fake shim hit non-Fake inputs; "
            "this means dispatcher routed real tensor to Python key, which "
            "should not happen. Check FakeTensor mode / tensor subclass setup.")


_SIGLIP_PATCH_SIZE = 14
_SIGLIP_PATCH_HIDDEN_SIZE = 1152
_SIGLIP_PROJECTOR_DIM = 2048


def _siglip_num_patches(image):
    if image.dim() != 4:
        raise RuntimeError(
            f"SigLIP fake dispatch expects 4D [B,C,H,W] image input, got "
            f"{tuple(image.shape)}")
    h = image.shape[-2]
    w = image.shape[-1]
    out_h = (h - _SIGLIP_PATCH_SIZE) // _SIGLIP_PATCH_SIZE + 1
    out_w = (w - _SIGLIP_PATCH_SIZE) // _SIGLIP_PATCH_SIZE + 1
    return out_h * out_w


def _siglip_patch_embed_shape(image, image_count):
    return (1, image_count * _siglip_num_patches(image),
            _SIGLIP_PATCH_HIDDEN_SIZE)


def _siglip_projector_shape_from_hidden(hidden):
    if hidden.dim() != 3:
        raise RuntimeError(
            f"SigLIP fake dispatch expects 3D [1,S,H] hidden input, got "
            f"{tuple(hidden.shape)}")
    return (hidden.shape[0], hidden.shape[1], _SIGLIP_PROJECTOR_DIM)


@torch.library.impl("rpu::rms_norm", "Python")
def _meta_rms_norm(input, normalized_shape, weight, eps=1e-5):
    _require_fake(input, weight)
    return torch.empty_like(input)


@torch.library.impl("rpu::apply_rotary_pos_emb", "Python")
def _meta_apply_rotary_pos_emb(query, key, cos, sin, position_ids):
    _require_fake(query, key)
    return torch.empty_like(query), torch.empty_like(key)


@torch.library.impl("rpu::scaled_dot_product_attention_unified", "Python")
def _meta_sdpa_unified(query, key, value, attn_mask=None, dropout_p=0.0,
                       is_causal=False, scale=None, enable_gqa=False):
    _require_fake(query, key, value)
    return torch.empty_like(query)


# Fused all-layers-once entry points。每个 op 包整层 forward,返回 Tensor
# 与 hidden_states 同 shape/dtype。k_caches/v_caches 在 C++ 内 in-place
# mutate,schema 未声明 (a!)/(b!) alias,但 op 作为 opaque 单元,Dynamo 不
# DCE 也不重排。
#
# 配套 C++ 改动:TORCH_LIBRARY_IMPL(rpu, AutogradPrivateUse1) 全部 op 改
# makeFallthrough()。否则 Autograd 优先级高于 Python,直接撞真 kernel
# to preserve the opaque fused operation.
@torch.library.impl("rpu::causal_decoder_forward", "Python")
def _meta_causal_decoder_forward(handle, hidden_states, k_caches, v_caches,
                                  attention_mask, position, is_causal,
                                  position_ids=None,
                                  deepstack_dense_visual_embeds=None,
                                  # Meta 须接收完整 schema，体内不消费这些参数。
                                  rope_cos_il=None, rope_sin_il=None,
                                  cos_sin_offset=-1, batch_slot=0,
                                  allow_batch_decode=False):
    _require_fake(hidden_states)
    return torch.empty_like(hidden_states)


@torch.library.impl("rpu::halo_action_expert_step_forward", "Python")
def _meta_halo_action_expert_step_forward(handle, x_emb, k_caches, v_caches,
                                          cos, sin, attn_mask_4d, prefix_len):
    # HALO action expert (mode=act) 单步: 返回末隐藏, 与 x_emb 同 shape/dtype
    # [1,cs,hidden]。HaloActionExpertModel 继承 CausalDecoderModel, 返回基类
    # output_tensor_ (= x_emb shape)。与上面 causal_decoder_forward 同模式。
    _require_fake(x_emb)
    return torch.empty_like(x_emb)


@torch.library.impl("rpu::halo_image_flow_step_forward", "Python")
def _meta_halo_image_flow_step_forward(handle, x_emb, k_caches, v_caches,
                                       cos, sin, prefix_len):
    # HALO image_flow (双流 MoT 去噪单步): 返回末隐藏 hidden [1,N,H], 与 x_emb
    # 同 shape/dtype (HaloImageFlowModel 返回基类 output_tensor_)。无 meta 时
    # FakeTensor/torch.compile 命中本 op 会落真 PrivateUse1 实现读 fake data_ptr
    # → 复现本文件开头描述的失败模式。与 action_expert 同模式。
    _require_fake(x_emb)
    return torch.empty_like(x_emb)


@torch.library.impl("rpu::lm_head_logits_top1", "Python")
def _meta_lm_head_logits_top1(logits):
    _require_fake(logits)
    return torch.empty(
        tuple(logits.shape[:-1]),
        dtype=torch.long,
        device=logits.device,
    )


@torch.library.impl("rpu::argmax_lastdim_host", "Python")
def _meta_argmax_lastdim_host(logits):
    _require_fake(logits)
    return torch.empty(
        tuple(logits.shape[:-1]),
        dtype=torch.long,
        device="cpu",
    )


@torch.library.impl("rpu::gemma_forward", "Python")
def _meta_gemma_forward(handle, hidden_states, k_caches, v_caches,
                        attention_mask, position, is_causal, chunk_size=0):
    _require_fake(hidden_states)
    return torch.empty_like(hidden_states)


@torch.library.impl("rpu::adarms_forward", "Python")
def _meta_adarms_forward(handle, hidden_states, cond, k_caches, v_caches,
                         attention_mask, position, is_causal):
    _require_fake(hidden_states)
    return torch.empty_like(hidden_states)


@torch.library.impl("rpu::internvla_nextdit_forward", "Python")
def _meta_internvla_nextdit_forward(
        handle, x, modulation, final_scale, raw_actions, use_action_encoder,
        populate_cross_cache, denoise_unroll, prepare_modulation_on_rpu,
        euler_steps,
        cross_context,
        self_k_caches, self_v_caches, cross_k_caches, cross_v_caches,
        layer_outputs, collect_layer_outputs):
    _require_fake(x)
    if use_action_encoder:
        return torch.empty(
            (raw_actions.shape[0], raw_actions.shape[1], 16),
            dtype=x.dtype,
            device=x.device,
        )
    return torch.empty_like(x)


@torch.library.impl("rpu::siglip_forward", "Python")
def _meta_siglip_forward(handle, input, k_caches, v_caches):
    _require_fake(input)
    if input.dim() == 4:
        shape = (1, input.shape[0] * _siglip_num_patches(input),
                 _SIGLIP_PROJECTOR_DIM)
    else:
        shape = _siglip_projector_shape_from_hidden(input)
    return torch.empty(shape, dtype=input.dtype, device=input.device)


@torch.library.impl("rpu::siglip_patch_embed", "Python")
def _meta_siglip_patch_embed(handle, input):
    _require_fake(input)
    return torch.empty(
        _siglip_patch_embed_shape(input, input.shape[0]),
        dtype=input.dtype,
        device=input.device,
    )


@torch.library.impl("rpu::siglip_patch_embed_multi", "Python")
def _meta_siglip_patch_embed_multi(handle, images):
    _require_fake(images)
    if not images:
        raise RuntimeError("SigLIP fake dispatch got empty image list")
    image = images[0]
    return torch.empty(
        _siglip_patch_embed_shape(image, len(images)),
        dtype=image.dtype,
        device=image.device,
    )


@torch.library.impl("rpu::siglip_forward_packed", "Python")
def _meta_siglip_forward_packed(handle, hidden, image_batch_count,
                                k_caches, v_caches):
    _require_fake(hidden)
    return torch.empty(
        _siglip_projector_shape_from_hidden(hidden),
        dtype=hidden.dtype,
        device=hidden.device,
    )


@torch.library.impl("rpu::siglip_forward_multi", "Python")
def _meta_siglip_forward_multi(handle, images, k_caches, v_caches):
    _require_fake(images)
    if not images:
        raise RuntimeError("SigLIP fake dispatch got empty image list")
    image = images[0]
    return torch.empty(
        (1, len(images) * _siglip_num_patches(image), _SIGLIP_PROJECTOR_DIM),
        dtype=image.dtype,
        device=image.device,
    )


@torch.library.impl("rpu::conv2d_mc_nhwc", "Python")
def _meta_conv2d_mc_nhwc(input, weight, *args, **kwargs):
    """SigLIP patch embedding。Dynamo 真要 trace 它的话需要精确算 (b,h_out,
    w_out,c_out) shape;现在用 empty_like(input) 偷懒 — 通常 SigLIP 整体
    被 rpu::siglip_forward 包,conv2d_mc_nhwc 不直接出现在 export 的 FX gm 里。
    将来真单独 trace,改成精确算输出 shape。"""
    _require_fake(input, weight)
    return torch.empty_like(input)


# Hy-Embodied owns nested GraphCache captures, so torch.compile is unsupported
# for that execution model. Partial shape shims are unsafe because the
# PROJ1_IN_MERGER path returns 1152-wide Vision rows and the merger/unroll ops
# cannot be represented by a partial fake graph.
_HYVLA_COMPILE_UNSUPPORTED = (
    "HyEmbodiedPolicy/Hy-VLA torch.compile is unsupported: adapter owns "
    "nested GraphCache captures"
)


def _reject_hyvla_fake(*args):
    _require_fake(*args)
    raise RuntimeError(_HYVLA_COMPILE_UNSUPPORTED)


@torch.library.impl("rpu::hyvit2_patch_embed", "Python")
def _fake_hyvit2_patch_embed(*args):
    return _reject_hyvla_fake(*args)


@torch.library.impl("rpu::hyvit2_patch_embed_multi", "Python")
def _fake_hyvit2_patch_embed_multi(*args):
    return _reject_hyvla_fake(*args)


@torch.library.impl("rpu::hyvit2_forward", "Python")
def _fake_hyvit2_forward(*args):
    return _reject_hyvla_fake(*args)


@torch.library.impl("rpu::hyvit2_forward_packed", "Python")
def _fake_hyvit2_forward_packed(*args):
    return _reject_hyvla_fake(*args)


@torch.library.impl("rpu::hyvit2_forward_multi", "Python")
def _fake_hyvit2_forward_multi(*args):
    return _reject_hyvla_fake(*args)


@torch.library.impl("rpu::hyvla_merger_pool", "Python")
def _fake_hyvla_merger_pool(*args):
    return _reject_hyvla_fake(*args)


@torch.library.impl("rpu::hyvla_merger_combine", "Python")
def _fake_hyvla_merger_combine(*args):
    return _reject_hyvla_fake(*args)


@torch.library.impl("rpu::hyvla_merger_fused", "Python")
def _fake_hyvla_merger_fused(*args):
    return _reject_hyvla_fake(*args)


@torch.library.impl("rpu::hyvla_vlm_step_forward", "Python")
def _fake_hyvla_vlm_step_forward(*args):
    return _reject_hyvla_fake(*args)


@torch.library.impl("rpu::hyvla_expert_step_forward", "Python")
def _fake_hyvla_expert_step_forward(*args):
    return _reject_hyvla_fake(*args)


@torch.library.impl("rpu::hyvla_expert_unroll_forward", "Python")
def _fake_hyvla_expert_unroll_forward(*args):
    return _reject_hyvla_fake(*args)
