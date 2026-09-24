"""HALO action expert (mode="act") → rpu_backend adapter package.

HALO 是 BAGEL 系 Mixture-of-Transformers VLA：同一 28 层 Qwen2 拓扑下，每层带
text / act 两套权重（act 孪生键以 ``*_moe_act`` 命名），act step 是非因果
prefix-attention（query 长度 cs=18，KV prefix=4706，12/2 GQA，head_dim=128，
attn_tp=2）。C++ 侧由 ``HaloActionExpertModel``承载：
``torch.ops.rpu.halo_action_expert_{create,destroy,set_weights,set_moe_weights,
step_forward}``。本包是其 Python 装载/驱动面：

  * :mod:`weights` — safetensors (HALO ema bf16) → 预 swizzle fp16 RPU 权重 +
    RoPE 表 + text_row_mask；
  * :mod:`runtime` — :class:`HaloStepRunner`：handle 装载、prefix KV 灌入、
    boundary host 数学（in/out proj、time embedding、sinusoidal pos）、单步
    capture/replay。

M1 边界与先例 wall_oss/pi05 一致：boundary 投影在 host，decoder 主体进
``step_forward``。构造逻辑本身不构成硬件正确性或数值证据；
精确配置的验证应遵循公开模型验证策略，见
``docs/validation_policy.md``。
"""
from __future__ import annotations

# NOTE:``build_cfg_branch`` 故意**不**在此公开导出 ——
# 生产路径必须经 ``HaloStepRunner.make_cfg_branch``(单调递增 epoch,杜绝手凑 epoch 撞 key →
# 陈旧 replay)。需直接构造(测试/高级用途)者显式 ``from ...runtime import build_cfg_branch``,
# 该路径自带「reach into submodule」信号。
from rpu_backend.adapters.halo.runtime import (
    HALO_EXECUTION_COMPONENTS,
    HaloCfgBranch,
    HaloExecutionController,
    HaloRollout,
    HaloStepRunner,
    build_halo_action_expert,
    euler_action_schedule,
    resolve_halo_execution,
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
from rpu_backend.adapters.halo.weights import HaloConfig, validate_halo_core_layout


def build_native_prefix_banks(
    ckpt_path,
    *,
    img_uncond_text_pack,
    cond_text_pack=None,
    frames=None,
    cfg=HaloConfig(),
    text_cache=None,
    vit_cache=None,
    rpu_execution=None,
    execution_controller=None,
):
    """编排 HALO action 部署路径的 ``cfg_img`` prefix-KV 银行(全 native,单 handle 轻路)。

    action 路径(``use_subtask=use_goal_image=False``)= A(text_cache)+ B(vit_cache)
    + 动作专家;三 CFG 分支各需一份 prefix-KV 银行(``build_cfg_branch`` 消费)。本函数
    只产单 handle 即可完成的 ``cfg_img`` 轻路:

      * ``cfg_img`` [8] —— text_cache img_uncond regime(空前缀,position [0..7])。
        最便宜的全 native 银行(纯 A 路径),1a 门坐实。

    ``cfg_text`` [4698](vit_cache.prefill_frames)与 ``cond`` [4706](text_cache cond,
    prefix=cfg_text)需 vit_cache + text_cache 两 handle —— 单进程并存会 OOM(见 SPM 注记)
    → 改用分阶段脚本,不在本函数内。

    Args:
        ckpt_path: HALO ema safetensors。
        img_uncond_text_pack: dict 含 ``packed_text_ids`` [8] + ``packed_text_position_ids``
            [0..7](golden ``img_uncond/text_pack.pt``)。
        cond_text_pack: 保留参数(仅 ``frames`` 路用,本函数已不走);传值不影响 cfg_img 轻路。
        frames: ``None`` = 只产 cfg_img(1a 轻路,不建 vit runner / 不载第二份 ckpt);
            **非 None 时本函数 fail-fast**(见 Raises),改用分阶段脚本。
        text_cache / vit_cache: 预建 runner(复用以省 ckpt 重载);``None`` = 内部 build。

    Returns:
        ``{"cfg_img": bank}``,bank = ``Dict[layer → (k, v)]`` 各 ``[8, nkv, hd]`` CPU
        (``build_cfg_branch`` 入口契约)。

    Raises:
        NotImplementedError: ``frames is not None`` —— 单进程三银行路在 HALO 统一 SpmAllocator
            下板上 OOM,在建任何 handle 之前 fail-fast,改用分阶段脚本(见 SPM 注记)。

    SPM 注记:HALO 统一 SpmAllocator SPM 紧 —— vit_cache + text_cache 两 handle 并存时,
    text_cache 持久 SPM 挤掉 vit prefill_frames 预算 → OOM(板实测 6405120 需 vs 6332416 free)。
    全 3 银行单进程路在 HALO 上不可用;改用分进程/分阶段(各只活一个 handle),见
    halo `scripts/halo_build_native_banks_stage1.py`(--phase vc/tc)。frames=None 轻路
    (仅 cfg_img,单 handle)不受此限,1a 已板验。
    """
    validate_halo_core_layout(cfg)
    if frames is not None:
        # 单进程同时建 vit_cache + text_cache 两 handle 的全 3 银行路
        # 在 HALO 统一 SpmAllocator 下板上 OOM(text_cache 持久 SPM ~672KB 挤掉 vit
        # prefill_frames 预算,实测 6405120 需 vs 6332416 free)→ 在建任何 handle 之前
        # fail-fast(真 fail-fast + board-free 可测),改用分阶段构建(各进程只活一个
        # handle,进程退出释放全部 SPM)。cfg_text/cond 的 prefill_frames + concat 逻辑见
        # 分阶段脚本,本函数只支持 frames=None 轻路。
        raise NotImplementedError(
            "build_native_prefix_banks(frames=...) 单进程三银行路在 HALO 统一 SpmAllocator "
            "下 OOM(vit_cache + text_cache 两 handle 并存)。改用分阶段构建(各进程只活一个 "
            "handle):halo scripts/halo_build_native_banks_stage1.py --phase vc|tc。本函数仅支持 "
            "frames=None 轻路(仅 cfg_img,单 handle;1a 已板验)。")
    owns_text_cache = text_cache is None
    if text_cache is not None and rpu_execution is not None:
        raise ValueError(
            "build_native_prefix_banks: rpu_execution must be bound when "
            "constructing text_cache, not after a runner is supplied"
        )
    if (
        text_cache is not None
        and execution_controller is not None
        and getattr(text_cache, "_halo_execution_controller", None)
        is not execution_controller
    ):
        raise ValueError(
            "build_native_prefix_banks: supplied text_cache belongs to a "
            "different HALO execution controller"
        )
    tc = text_cache if text_cache is not None else build_halo_text_cache(
        ckpt_path,
        cfg=cfg,
        rpu_execution=rpu_execution,
        execution_controller=execution_controller,
    )
    try:
        result = {
            "cfg_img": tc.prefill(
                img_uncond_text_pack["packed_text_ids"],
                img_uncond_text_pack["packed_text_position_ids"],
                prefix_kv=None,
            )
        }
    except BaseException as error:
        if owns_text_cache:
            try:
                tc.close()
            except BaseException as cleanup_error:
                error.add_note(f"HALO prefix-bank cleanup also failed: {cleanup_error!r}")
        raise
    if owns_text_cache:
        tc.close()
    return result


__all__ = [
    "CFGBranch",
    "HaloCfgBranch",
    "HaloConfig",
    "HaloExecutionController",
    "HALO_EXECUTION_COMPONENTS",
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
    "resolve_halo_execution",
]
