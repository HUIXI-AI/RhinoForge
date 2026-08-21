"""HALO action expert (mode="act") adapter package.

HALO 是 BAGEL 系 Mixture-of-Transformers VLA：同一 28 层 Qwen2 拓扑下，每层带
text / act 两套权重（act 孪生键以 ``*_moe_act`` 命名），act step 是非因果
prefix-attention（query 长度 cs=18，KV prefix=4706，12/2 GQA，head_dim=128，
attn_tp=2）。C++ 侧由 ``HaloActionExpertModel`` 承载：
``torch.ops.rpu.halo_action_expert_{create,destroy,set_weights,set_moe_weights,
step_forward}``。本包是其 Python 装载/驱动面：

  * :mod:`weights` — safetensors (HALO ema bf16) → 预 swizzle fp16 RPU 权重 +
    RoPE 表 + text_row_mask；
  * :mod:`runtime` — :class:`HaloStepRunner`：handle 装载、prefix KV 灌入、
    boundary host 数学（in/out proj、time embedding、sinusoidal pos）、单步
    capture/replay。

boundary 投影在 host，decoder 主体进入 ``step_forward``。数值正确性需通过硬件
端到端验证。
"""
from __future__ import annotations

# ``build_cfg_branch`` 故意**不**在此公开导出 ——
# 生产路径必须经 ``HaloStepRunner.make_cfg_branch``(单调递增 epoch,杜绝手凑 epoch 撞 key →
# 陈旧 replay)。需直接构造(测试/高级用途)者显式 ``from ...runtime import build_cfg_branch``,
# 该路径自带「reach into submodule」信号。
from rpu_backend.adapters.halo.runtime import (
    HaloCfgBranch,
    HaloRollout,
    HaloStepRunner,
    build_halo_action_expert,
    euler_action_schedule,
)
from rpu_backend.adapters.halo.text_cache import (
    HaloTextCacheUpdate,
    build_causal_prefix_mask,
    build_halo_text_cache,
    build_seq_rope_tables,
)
from rpu_backend.adapters.halo.text_decode import (
    HaloTextDecodeStep,
    build_halo_text_decode,
)
from rpu_backend.adapters.halo.image_flow import (
    CFGBranch,
    HaloImageFlowStep,
    build_halo_image_flow,
)
from rpu_backend.adapters.halo.vit_cache import (
    HaloVitCacheUpdate,
    VitCacheConfig,
    build_halo_vit_cache,
)
from rpu_backend.adapters.halo.weights import HaloConfig


def build_native_prefix_banks(
    ckpt_path,
    *,
    img_uncond_text_pack,
    cond_text_pack=None,
    frames=None,
    cfg=HaloConfig(),
    text_cache=None,
    vit_cache=None,
):
    """编排 HALO action 部署路径的 ``cfg_img`` prefix-KV 银行(全 native,单 handle 轻路)。

    action 路径(``use_subtask=use_goal_image=False``)= A(text_cache)+ B(vit_cache)
    + 动作专家;三 CFG 分支各需一份 prefix-KV 银行(``build_cfg_branch`` 消费)。本函数
    只产单 handle 即可完成的 ``cfg_img`` 轻路:

      * ``cfg_img`` [8] —— text_cache img_uncond regime(空前缀,position [0..7])。
        该银行只需要 text-cache 路径。

    ``cfg_text`` [4698](vit_cache.prefill_frames)与 ``cond`` [4706](text_cache cond,
    prefix=cfg_text)需 vit_cache + text_cache 两 handle —— 单进程并存会 OOM(见 SPM 注记)。
    调用方需在隔离进程中预建这两类 bank，再交给
    ``HaloStepRunner.make_cfg_branch``。

    Args:
        ckpt_path: HALO ema safetensors。
        img_uncond_text_pack: dict 含 ``packed_text_ids`` [8] + ``packed_text_position_ids``
            [0..7]。
        cond_text_pack: 保留参数(仅 ``frames`` 路用,本函数已不走);传值不影响 cfg_img 轻路。
        frames: ``None`` = 只产 cfg_img(1a 轻路,不建 vit runner / 不载第二份 ckpt);
            **非 None 时本函数 fail-fast**(见 Raises)，由调用方提供预建 bank。
        text_cache / vit_cache: 预建 runner(复用以省 ckpt 重载);``None`` = 内部 build。

    Returns:
        ``{"cfg_img": bank}``,bank = ``Dict[layer → (k, v)]`` 各 ``[8, nkv, hd]`` CPU
        (``HaloStepRunner.make_cfg_branch`` 入口契约)。

    Raises:
        NotImplementedError: ``frames is not None`` —— 单进程三银行路在 HALO 统一 SpmAllocator
            下板上 OOM，在建任何 handle 之前 fail-fast；调用方需提供分阶段预建 bank。

    SPM 注记:HALO 统一 SpmAllocator SPM 紧 —— vit_cache + text_cache 两 handle 并存时,
    text_cache 持久 SPM 挤掉 vit prefill_frames 预算 → OOM(需 6405120、可用 6332416)。
    全 3 银行单进程路在 HALO 上不可用；需分进程/分阶段预建
    ``cfg_text`` 和 ``cond`` bank（各只活一个 handle）。frames=None 轻路
    (仅 cfg_img,单 handle)不受此限。
    """
    if frames is not None:
        # 单进程同时建 vit_cache + text_cache 两 handle 的全 3 银行路
        # 在 HALO 统一 SpmAllocator 下板上 OOM(text_cache 持久 SPM ~672KB 挤掉 vit
        # prefill_frames 预算,需 6405120、可用 6332416)→ 在建任何 handle 之前
        # fail-fast(真 fail-fast + board-free 可测),改用分阶段构建(各进程只活一个
        # handle,进程退出释放全部 SPM)。cfg_text/cond bank 需由调用方分阶段预建，
        # 本函数只支持 frames=None 轻路。
        raise NotImplementedError(
            "build_native_prefix_banks(frames=...) 单进程三银行路在 HALO 统一 SpmAllocator "
            "下 OOM(vit_cache + text_cache 两 handle 并存)。请在隔离进程中使用 "
            "build_halo_vit_cache/build_halo_text_cache 预建 bank，再交给 "
            "HaloStepRunner.make_cfg_branch。本函数仅支持 "
            "frames=None 轻路(仅 cfg_img,单 handle)。")
    owns_text_cache = text_cache is None
    tc = text_cache if text_cache is not None else build_halo_text_cache(
        ckpt_path, cfg=cfg
    )
    try:
        return {
            "cfg_img": tc.prefill(
                img_uncond_text_pack["packed_text_ids"],
                img_uncond_text_pack["packed_text_position_ids"],
                prefix_kv=None,
            )
        }
    finally:
        if owns_text_cache:
            tc.close()


__all__ = [
    "CFGBranch",
    "HaloCfgBranch",
    "HaloConfig",
    "HaloImageFlowStep",
    "HaloRollout",
    "HaloStepRunner",
    "HaloTextCacheUpdate",
    "HaloTextDecodeStep",
    "HaloVitCacheUpdate",
    "VitCacheConfig",
    "build_causal_prefix_mask",
    "build_halo_action_expert",
    "build_halo_image_flow",
    "build_halo_text_cache",
    "build_halo_text_decode",
    "build_halo_vit_cache",
    "build_native_prefix_banks",
    "build_seq_rope_tables",
    "euler_action_schedule",
]
