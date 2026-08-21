"""Hy-Embodied-0.5-VLA 运行时：三座子系统的生命周期 + `image → action`。

    3×image ──ViT──> img_emb ──组装──> prefix_embs ──VLM prefill──> prefix KV
                                        ──expert denoise ×10──> action

C++ 侧由三个 fused 子系统承载（`torch.ops.rpu.hyvit2_* / hyvla_vlm_* /
hyvla_expert_*`）；本模块负责装载、图缓存与生命周期。

生命周期约束：

1. `weights.load_hy_vla_weights()` 在任何 forward 前一次性完成全部权重转换与上传。
2. handle 的创建和 `set_weights` 必须原子完成，不能留下未绑定权重的 handle。
3. 非持久子系统使用后必须 destroy，并用新的 `GraphCache` 重建下一次执行状态。
4. 持久子系统只在权重、KV cache、SPM 地址和 mutable DMA 约束保持稳定时复用；
   prompt 布局变化会使全部持久子系统失效并重建。
5. BUILD 的输出是本次 forward 的有效输出，不需要额外丢弃或预热。
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

import rpu_backend
from rpu_backend.api.cache import RPUCache
from rpu_backend.adapters.hy_vla.weights import (HyVlaConfig, HyVlaWeights,
                                                 build_rope_tables,
                                                 build_time_embed,
                                                 build_unroll_weights,
                                                 load_hy_vla_weights)


def _setdefault_env(
    key: str,
    value: str,
    snapshot: dict[str, str | None] | None,
) -> str:
    """Set one builder default while recording only the key we own."""
    if snapshot is not None and key not in snapshot:
        snapshot[key] = os.environ.get(key)
    return os.environ.setdefault(key, value)


class _DirectHyVlaOwner:
    """Weak-referenceable process owner for the low-level builder entry."""


# `RPU_HY_VLA_DENOISE_UNROLL` keeps the Euler sequence in the fused expert
# graph and SPM. Its device-fp16 GEMMs are not bit-equivalent to host fp32.
def _unroll_default() -> bool:
    return os.environ.get("RPU_HY_VLA_DENOISE_UNROLL", "0").strip() in (
        "1", "true", "True", "on")


# Packed ViT uses per-image attention segments; images cannot attend across
# segments and no padding mask is required.
def _vit_packed_default() -> bool:
    return os.environ.get("RPU_HY_VLA_VIT_PACKED", "0").strip() in (
        "1", "true", "True", "on")


def _fused_merger_default() -> bool:
    """merger 组轴是否走 device。**必须与 C++ 侧的 `RPU_HY_VLA_FUSED_MERGER` 同开同关**
    —— 它改的是 ViT 尾部的行布局（member-major），不是数值。"""
    return os.environ.get("RPU_HY_VLA_FUSED_MERGER", "0").strip() in (
        "1", "true", "True", "on")


def _merger_in_graph_default() -> bool:
    """merger 余部是否整条收进一个图内 op（`rpu.hyvla_merger_fused`）。默认 **OFF**。

    要求 `RPU_HY_VLA_FUSED_MERGER=1`（member-major 布局）—— 单独开无效，见 `_run_vit`。
    """
    return os.environ.get("RPU_HY_VLA_MERGER_IN_GRAPH", "0").strip() in (
        "1", "true", "True", "on")


def _prefix_template_default() -> bool:
    """prefix 骨架是否缓存复用。默认 **ON**，输出逐位不变（见 `_prefix_tmpl`）。"""
    return os.environ.get("RPU_HY_VLA_PREFIX_TEMPLATE", "1").strip() not in (
        "0", "false", "False", "off")


def _pe_in_graph_default() -> bool:
    """patch_embed 是否收进 ViT 的 capture。默认 **ON**。

    需与 C++ 的 `RPU_HY_VLA_PATCH_EMBED_MC` 同时启用。
    """
    return os.environ.get("RPU_HY_VLA_PATCH_EMBED_IN_GRAPH", "1").strip() not in (
        "0", "false", "False", "off")


def _proj1_in_merger_default() -> bool:
    """Whether fused merger owns `proj1`; enabled by default.

    Python and native code must use the same setting because it changes the
    ViT tail layout and merger input shape.
    """
    return os.environ.get("RPU_HY_VLA_PROJ1_IN_MERGER", "1").strip() not in (
        "0", "false", "False", "off")


def _persist_default() -> frozenset:
    """返回要跨帧存活的子系统名集合。`1`/`on` = 三座全留；也可给逗号分隔的子集
    （`vlm,expert`）用于隔离实验。"""
    v = os.environ.get("RPU_HY_VLA_PERSIST_HANDLES", "0").strip()
    if v in ("1", "true", "True", "on"):
        return frozenset(("vit", "vlm", "expert"))
    if v in ("0", "", "false", "False", "off"):
        return frozenset()
    return frozenset(x.strip() for x in v.split(",") if x.strip())

MASK_NEG = -50000.0   # fp16 安全的 additive mask 常量。bf16 常用的 -2.38e38
                      # 在 fp16 上会溢出；score 量级 ~1e2，
                      # 加 -50000 后仍在 fp16 范围内，softmax 结果为 0。


def _rpu16(t: torch.Tensor) -> torch.Tensor:
    return t.to(dtype=torch.float16, device="rpu").contiguous()


class _Sub:
    """一座子系统的生命周期：create+bind → 用 → destroy。

    create 与 bind 绑成一步（不留空 handle）；每次 open 配一个新的
    `GraphCache`；`close()` 必定 destroy 并 `reset_all()`。
    """

    def __init__(self, name: str, create, destroy, bind: Callable[[int], None],
                 sig: "rpu_backend.graph.GraphSignature", persist: bool = False,
                 variant=None):
        self._name, self._create, self._destroy, self._bind = name, create, destroy, bind
        self.sig = sig
        self.persist = persist        # True ⇒ 出作用域不 destroy，跨帧存活
        self.variant = variant        # bind 闭包的语义指纹；变了就必须重建
        self.handle: Optional[int] = None
        self.gc: Optional["rpu_backend.graph.GraphCache"] = None

    def open(self) -> None:
        assert self.handle is None, f"{self._name}: 已经 open"
        handle = int(self._create())
        try:
            self._bind(handle)          # ← 与 create 绑成一步：不留空 handle
            graph_cache = rpu_backend.graph.GraphCache()
        except BaseException as exc:
            cleanup_ok = True
            try:
                self._destroy(handle)
            except BaseException:
                cleanup_ok = False
            try:
                torch.ops.rpu.spm_alloc_reset_all()
            except BaseException:
                cleanup_ok = False
            self.handle = None
            self.gc = None
            if not cleanup_ok:
                raise RuntimeError(
                    f"{self._name}: open 失败且 native 清理未完成；"
                    "请重启进程后再创建 runner。") from exc
            raise
        self.handle = handle
        self.gc = graph_cache

    def close(self) -> None:
        if self.handle is None:
            return
        handle = self.handle
        first_error = None
        try:
            if self.gc is not None:
                self.gc.clear()
        except BaseException as exc:
            first_error = exc
        try:
            self._destroy(handle)
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        try:
            torch.ops.rpu.spm_alloc_reset_all()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        self.handle = None
        self.gc = None
        if first_error is not None:
            raise first_error

    def __enter__(self):
        if self.handle is None:       # 持久模式下只有第一帧真正 open
            self.open()
        return self

    def __exit__(self, *exc):
        if not self.persist:
            self.close()
        return False


@dataclasses.dataclass
class HyVlaRunner:
    """一次 `get_action` = 一条完整的 image → action 链路。

    `RPU_HY_VLA_PERSIST_HANDLES=1` 时三座跨帧存活，第一帧创建 handle、绑定权重并
    BUILD，之后走 REPLAY；`=0` 时每帧重建并重新 preload 权重。
    prompt 相关常量变化会通过 `_cached` 使持久状态失效并重建。
    权重本身只在 `build_hy_vla()` 时 swizzle 一次，常驻 RPU DDR。
    """
    w: HyVlaWeights
    prefix_len: int                 # S = ceil16(有效 prefix 行数)
    _embed: Callable = dataclasses.field(repr=False, default=None)
    _vit_cache: "RPUCache" = dataclasses.field(repr=False, default=None)
    _cache: "RPUCache" = dataclasses.field(repr=False, default=None)
    _out_proj_wt: torch.Tensor = dataclasses.field(repr=False, default=None)
    _out_proj_b: torch.Tensor = dataclasses.field(repr=False, default=None)
    _const_key: tuple = dataclasses.field(repr=False, default=None)
    _const_val: tuple = dataclasses.field(repr=False, default=None)
    _unroll: bool = dataclasses.field(default_factory=_unroll_default)
    _vit_packed: bool = dataclasses.field(default_factory=_vit_packed_default)
    _fused_merger: bool = dataclasses.field(default_factory=_fused_merger_default)
    _merger_in_graph: bool = dataclasses.field(
        default_factory=_merger_in_graph_default)
    _pe_in_graph: bool = dataclasses.field(default_factory=_pe_in_graph_default)
    _proj1_in_merger: bool = dataclasses.field(
        default_factory=_proj1_in_merger_default)
    _prefix_template: bool = dataclasses.field(
        default_factory=_prefix_template_default)
    _prefix_key: tuple = dataclasses.field(repr=False, default=None)
    _prefix_val: tuple = dataclasses.field(repr=False, default=None)
    _persist: frozenset = dataclasses.field(default_factory=_persist_default)
    _psubs: dict = dataclasses.field(repr=False, default=None)
    _uw: tuple = dataclasses.field(repr=False, default=None)
    _ux0: torch.Tensor = dataclasses.field(repr=False, default=None)
    _utraj: torch.Tensor = dataclasses.field(repr=False, default=None)
    _closed: bool = dataclasses.field(repr=False, default=False, init=False)

    def __post_init__(self):
        self._embed = build_time_embed(self.w.cfg, self.w.host)
        c = self.w.cfg
        # `action_out_proj` 是 N=32 的小 GEMM；预转置让逐步路径直接使用 addmm。
        # 大 GEMM 仍保持 `F.linear`，避免不必要的转置副本。
        self._out_proj_wt = self.w.host["action_out_proj.weight"].t().contiguous()
        self._out_proj_b = self.w.host["action_out_proj.bias"]
        self._const_key, self._const_val = {}, {}
        self._psubs = {}
        if self._unroll:
            # 图内 unroll 的常驻缓冲：x0 行 0 是 state 位（恒 0，encoder 对它的输出
            # 从不进 emb_stage_，是死路），行 1: 每帧填 noise；x_traj 收逐步轨迹，
            # 终值 = [-1]。两者地址跨帧稳定 ⇒ mutable DMA 每帧只改基址。
            self._uw = build_unroll_weights(c, self.w.host)
            self._ux0 = torch.zeros(1, c.suffix_len, c.action_dim,
                                    dtype=torch.float16, device="rpu")
            self._utraj = torch.empty(c.num_steps, c.suffix_len, c.action_dim,
                                      dtype=torch.float16, device="rpu")
        # 两条 KV cache **建一次、跨帧复用**。形状只依赖 cfg 与 prefix_len，二者
        # 在 runner 生命周期内不变。
        # 复用**不依赖零初始化**：`reset_to_position()` 的契约是 insert kernel
        # overwrite-before-read（见 `api/cache.py`），三段各自 reset —— ViT 与
        # prefill 到 0、denoise 每步到 S。
        # C++ 侧 `kv_seq_len` 使用逻辑长度（ViT `seq_len_`；VLM/expert
        # `position+seq_len`），物理 padding 行不参与读取。
        self._total = ((self.prefix_len + c.suffix_len + 15) // 16) * 16
        # packed 路径把 N 相机拼成一条 N*196 的序列 ⇒ ViT 的 KV cache 要够长。
        self._vit_cache = RPUCache(
            num_layers=c.vit_layers, batch_size=1,
            max_seq_len=c.vit_seq * (c.num_cameras if self._vit_packed else 1),
            num_kv_heads=c.vit_heads, head_dim=c.vit_head_dim_padded, attn_tp=8)
        # ⚠️ VLM 与 expert **共享这一个 cache**（expert 把 suffix KV 追加在
        # prefix 之后）⇒ TP8 的 KV 槽位复制必须**两座一起做**，不能只改一座。
        self._cache = RPUCache(
            num_layers=c.layers, batch_size=1, max_seq_len=self._total,
            num_kv_heads=c.kv_heads_rpu, head_dim=c.head_dim, attn_tp=c.attn_tp)

    # ── 子系统获取（持久 or 每帧新建）────────────────────────────────────
    def _sub(self, name, create, destroy, bind, sig, variant=None) -> "_Sub":
        """持久模式下三座子系统跨帧存活 —— handle 与 GraphCache 都不重建。

        复用成立的前提是图中绑定的状态跨帧保持稳定：
          · 层 0 的 hidden 输入走 `hidden_in_src_base_` 的 **mutable DMA**，每次
            REPLAY 重新解引用，因此调用方每帧新建的
            `hid` / `x_prefix` 地址漂移**不影响**；
          · 显式 2D mask 由 `dynamic_config` 拷进稳定 DDR 槽（同上注释）；
          · 权重、KV cache（`__post_init__` 建一次）、SPM 绝对地址全部稳定 ——
            SPM 是确定性 bump 分配器，每帧三段的分配序列完全相同 ⇒ 地址逐帧相同；
          · rope 表 / additive mask / x0 / x_traj 由步 6 与 unroll 钉成跨帧同一批张量。
        前提被打破的唯一入口是 prompt 变了（rope/mask 要重算），由 `_cached` 的
        miss 分支负责把三座全部关掉重建。

        SPM 前提：三座共存需要 super_persistent ≤ 8191 − VLM prefill temp(4912.5)。
        行掩码使用行向量后 super_persistent = 1521 KB，余量 1757.5 KB。
        """
        if self._closed:
            raise RuntimeError("HyVlaRunner 已 close，不能重建子系统 handle。")
        if name not in self._persist:
            return _Sub(name, create, destroy, bind, sig)
        s = self._psubs.get(name)
        if s is not None and s.variant != variant:
            # bind 的语义变了（例如 expert 的 unroll 开关被翻转）⇒ 缓存的 handle 绑的是
            # 旧语义，必须重建。否则会出现"handle 没 set_action_weights 却走 unroll"。
            s.persist = False
            s.close()
            del self._psubs[name]
            s = None
        if s is None:
            s = _Sub(name, create, destroy, bind, sig, persist=True, variant=variant)
            self._psubs[name] = s
        return s

    def _drop_psubs(self) -> None:
        """关掉全部持久子系统（prompt 布局变了，或 runner 析构）。"""
        subs = list(self._psubs.values())
        self._psubs.clear()
        first_error = None
        for s in subs:
            s.persist = False
            try:
                s.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        """Deterministically retire persistent graph/handle ownership."""
        if self._closed:
            return
        try:
            self._drop_psubs()
        finally:
            self._closed = True

    def __del__(self):
        # 持久 handle 必须显式释放：super-persistent 只在活实例数归零时整体回收，
        # 泄漏会让下一个 runner 在 S=240 撞 SPM。测试里 `del runner; gc.collect()`
        # 之后才建第二个 runner，依赖的就是这里。
        try:
            self.close()
        except Exception:
            pass

    # ── 跨帧常量缓存 ──────────────────────────────────────────────────────
    def _cached(self, tag: str, key: Sequence[torch.Tensor], build: Callable):
        """缓存**只依赖 prompt 布局**的每帧常量（rope 表 / additive mask / modality）。

        它们是 `prefill_pos` / `prefill_mask` / `modality_mask` / `denoise_mask` /
        `suffix_pos` 的纯函数，而这几个输入在同一条 prompt 上跨帧不变 —— 但**不能
        假定**不变（prompt 变了长度就变了），所以按内容比对：CPU 上几次 `torch.equal`
        是微秒级，省下的是 fp64 `outer`+cos/sin、240 次 Python 索引的 padding 对角
        补口、以及 4 次 H2D 上传（vmask 115 KB / emask 30 KB / 两组 rope 表）。

        复用同一批 RPU 张量对图是**中性偏好**的：每帧仍新建 handle + 新 GraphCache，
        BUILD 会重新 bake 指针；地址不变只是少一次分配。
        """
        prev = self._const_key.get(tag)
        if prev is not None and all(
                a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
                for a, b in zip(prev, key)):
            return self._const_val[tag]
        # miss ⇒ rope 表 / mask 要换一批张量，而它们的地址被已 BUILD 的图 bake 了
        # ⇒ 持久子系统必须全部作废重建（见 `_sub` 的前提清单）。
        self._drop_psubs()
        val = build()
        self._const_key[tag] = [t.clone() for t in key]
        self._const_val[tag] = val
        return val

    # ── prefix 组装（ViT → VLM 的接缝）────────────────────────────────────
    def assemble_prefix(self, img_embs: Sequence[torch.Tensor],
                        lang_tokens: torch.Tensor) -> torch.Tensor:
        """`[bos, hy_user] + 每图(vs + 7行×(7 patch + 1 split) + ve) + lang`。

        `embed_tokens` 与 `lm_head` 是 tied（ckpt 无独立 embed_tokens），所以直接
        用 `tok_emb` 查表；5 个特殊 token 的 id 由反查得到，见 `HyVlaConfig`。
        """
        c, tok = self.w.cfg, self.w.tok_emb
        g, D = c.grid, c.proj_dim
        if self._prefix_template:
            buf, slots, _ = self._prefix_tmpl(len(img_embs), lang_tokens)
            for sl, ie in zip(slots, img_embs):
                sl.copy_(ie.reshape(g, g, D))
            return buf
        split = tok[c.tok_vision_split]
        parts = [tok[c.tok_bos][None], tok[c.tok_user][None]]
        for ie in img_embs:
            parts.append(tok[c.tok_vision_start][None])
            grid = ie.reshape(g, g, D)
            grid = torch.cat([grid, split[None, None].expand(g, 1, D)], dim=1)
            parts.append(grid.reshape(g * (g + 1), D))
            parts.append(tok[c.tok_vision_end][None])
        parts.append(tok[lang_tokens.long()])
        return torch.cat(parts, 0)[None]

    def _prefix_tmpl(self, n_img: int, lang_tokens: torch.Tensor):
        """prefix 的**常量骨架** + 每图 49 行的写入视图，逐帧复用。

        只有 3×49 行的图像 embedding 逐帧变化；
        bos / user / vision_start / 每行末尾的 split / vision_end / lang 全是常量。

        ⇒ 骨架建一次（键：图数 + `lang_tokens` 内容），每帧只 `copy_` 3 次
        49×2048 的图像行。**逐位不变** —— 写进去的是同样的值、同样的位置。

        ⚠️ 返回的是**跨帧复用的同一块 host buffer**。唯一的消费者是
        `_run_vlm_prefill` 里同一帧内的 `_rpu16(prefix[:, :S])`（立刻拷上 RPU），
        没有跨帧存活的引用；调用方若要留着它必须自己 clone。
        """
        prev = self._prefix_key
        if not (prev is not None and prev[0] == n_img
                and torch.equal(prev[1], lang_tokens)):
            c, tok, g, D = self.w.cfg, self.w.tok_emb, self.w.cfg.grid, self.w.cfg.proj_dim
            rows = g + 1                       # 每行 7 个 patch + 1 个 split
            per = g * rows                     # 每图 56 行
            lang = tok[lang_tokens.long()]
            # The cached skeleton is fp16, matching its only consumer.
            buf = torch.empty((1, 2 + n_img * (per + 2) + lang.shape[0], D),
                              dtype=torch.float16)
            buf[0, 0] = tok[c.tok_bos]
            buf[0, 1] = tok[c.tok_user]
            slots = []
            for i in range(n_img):
                base = 2 + i * (per + 2)
                buf[0, base] = tok[c.tok_vision_start]
                blk = buf[0, base + 1: base + 1 + per].view(g, rows, D)
                blk[:, g] = tok[c.tok_vision_split]     # 每行末尾的 split
                slots.append(blk[:, :g])                # ← 逐帧只写这个视图
                buf[0, base + 1 + per] = tok[c.tok_vision_end]
            buf[0, 2 + n_img * (per + 2):] = lang
            self._prefix_key = (n_img, lang_tokens.clone())
            self._prefix_val = (buf, slots, per)
        return self._prefix_val

    # ── 三段 ──────────────────────────────────────────────────────────────
    def _run_vit(self, images: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        c, w = self.w.cfg, self.w
        cache = self._vit_cache

        def bind(h):
            # W8A16 与 fp16 走**同一个** C++ set_weights，只是多传 6 条 scale 列表
            # （`rpu_hyvit2_set_weights` 自己就是拿空列表调它）。scale 非空 ⇒
            # `check_hyvit2_w8a16_scale_lists` 强制权重必须是 int8。
            args = (h, w.vit["qw"], w.vit["kw"], w.vit["vw"], w.vit["ow"],
                    w.vit["f1w"], w.vit["f2w"], w.vit["n1w"], w.vit["n1b"],
                    w.vit["n2w"], w.vit["n2b"], w.vit["qb"], w.vit["kb"], w.vit["vb"],
                    w.vit["ob"], w.vit["f1b"], w.vit["f2b"],
                    w.vit_proj1_w, w.vit_proj1_b,
                    c.vit_heads, c.vit_head_dim_padded, c.vit_hidden,
                    c.vit_inter_padded, c.proj_dim, c.vit_eps)
            if w.vit["qws"]:
                torch.ops.rpu.hyvit2_set_weights_w8a16(
                    *args, w.vit["qws"], w.vit["kws"], w.vit["vws"],
                    w.vit["ows"], w.vit["f1ws"], w.vit["f2ws"])
            else:
                torch.ops.rpu.hyvit2_set_weights(*args)
            torch.ops.rpu.hyvit2_model_set_patch_emb(
                h, w.vit_patch_gemm_w, w.vit_pos_fused, 16, 16)

        n_img = len(images)
        vit_seq = c.vit_seq * (n_img if self._vit_packed else 1)
        sub = self._sub("vit", torch.ops.rpu.hyvit2_create,
                        torch.ops.rpu.hyvit2_destroy, bind,
                        rpu_backend.graph.GraphSignature(
                       op_id="hyvit2_vision_compute",
                       shapes=[vit_seq, c.vit_layers, c.vit_head_dim_padded],
                       dyn_dims=[c.vit_heads, 8, n_img if self._vit_packed else 1, 0],
                       dtypes=[torch.float16]))

        def fwd(img):
            cache.reset_to_position(0)
            # 非 packed 路径的 patch_embed 留在 capture 之外。
            hid = torch.ops.rpu.hyvit2_patch_embed(sub.handle, _rpu16(img))
            with sub.gc.capture(sub.sig):
                return torch.ops.rpu.hyvit2_forward(
                    sub.handle, hid,
                    [cache.k_caches[i] for i in range(c.vit_layers)],
                    [cache.v_caches[i] for i in range(c.vit_layers)])

        with sub:
            # BUILD output is valid. Camera outputs remain live through the
            # batched merger and must be read before temporary SPM is reset.
            if self._vit_packed:
                # Packed inputs use isolated minibatch SDPA segments.
                # Graph patch embedding requires the matching native switch;
                # `_rpu16` H2D remains outside capture.
                cache.reset_to_position(0)
                imgs_rpu = [_rpu16(im) for im in images]
                kc = [cache.k_caches[i] for i in range(c.vit_layers)]
                vc = [cache.v_caches[i] for i in range(c.vit_layers)]
                if self._pe_in_graph:
                    with sub.gc.capture(sub.sig):
                        hid = torch.ops.rpu.hyvit2_patch_embed_multi(
                            sub.handle, imgs_rpu)
                        o = torch.ops.rpu.hyvit2_forward_packed(
                            sub.handle, hid, n_img, kc, vc)
                else:
                    hid = torch.ops.rpu.hyvit2_patch_embed_multi(
                        sub.handle, imgs_rpu)
                    with sub.gc.capture(sub.sig):
                        o = torch.ops.rpu.hyvit2_forward_packed(
                            sub.handle, hid, n_img, kc, vc)
                stacked = o.reshape(n_img, -1, self._vit_out_dim())
            else:
                outs = [fwd(im) for im in images]
                stacked = torch.cat(
                    [o.reshape(1, -1, self._vit_out_dim()) for o in outs], 0)
            # `_prefix_template` 的唯一消费者是 fp16 prefix 骨架，因此无需先在
            # device 上扩宽为 fp32。
            if self._merger_in_graph:
                # 整条 merger 余部收进一个图内 op。**先回收 ViT 的 temporary**：
                # ViT capture 与 merger 的临时 SPM 不能共存。`o` 已落定在 DDR，
                # reset_temporary 不会修改它。
                # 读回仍然在 capture **之外** —— capture 内 `.cpu()` 拿到的是空值。
                torch.ops.rpu.spm_alloc_reset_temporary()
                with sub.gc.capture(self._merger_sig(n_img)):
                    emb_rpu = self._merger_rest(stacked)
                embs = emb_rpu.cpu()
            else:
                embs = self._merger_rest(stacked).cpu()
            if not self._prefix_template:
                embs = embs.float()   # 老 `assemble_prefix` 要与 fp32 的 tok 表 cat
            torch.ops.rpu.spm_alloc_reset_temporary()
        return [embs[i] for i in range(len(images))]

    def _vit_out_dim(self) -> int:
        """ViT 塔的输出维。`_proj1_in_merger` 打开时 proj1 不在塔尾 ⇒ 是 hidden。"""
        c = self.w.cfg
        return (c.vit_hidden if (self._proj1_in_merger and self._fused_merger)
                else c.proj_dim)

    def _merger_sig(self, n_img: int) -> "rpu_backend.graph.GraphSignature":
        """merger 图的签名。与 ViT 共用同一个 `GraphCache`（两个 entry），
        因为它俩的生命周期完全一致（都挂在 `_Sub("vit")` 上）。"""
        c = self.w.cfg
        return rpu_backend.graph.GraphSignature(
            op_id="hyvla_merger_fused",
            # ⚠️ 输入维必须进签名：`_proj1_in_merger` 翻转会改变本图的输入形状，
            #    否则切开关时会命中上一档的缓存图。
            shapes=[n_img * c.vit_seq // 4, self._vit_out_dim(), 4],
            dyn_dims=[8, n_img, 0, 0],
            dtypes=[torch.float16])

    def _merger_rest(self, proj1_out: torch.Tensor) -> torch.Tensor:
        """merger 余部：DwPooler(group-softmax) → GELU → proj2。

        batch 维 B = 相机数（生产走 B=3 合批，见 `_run_vit`）。

        The host path performs group-axis rearrangement and reductions in
        fp16 while GEMMs remain on RPU. The graph path uses the fused merger.
        """
        M, D = self.w.merger_rest, self.w.cfg.proj_dim
        B0 = proj1_out.shape[0] if proj1_out.dim() == 3 else 1
        if self._merger_in_graph:
            # The fused graph op owns the complete merger remainder.
            assert self._fused_merger, \
                "RPU_HY_VLA_MERGER_IN_GRAPH 需要 member-major 布局，必须同时开 " \
                "RPU_HY_VLA_FUSED_MERGER"
            # proj1 进图 ⇒ 输入是 [4,G,vit_hidden]，权重/偏置多传两条；
            # 关掉时传 None，走与落地前逐位相同的老链路。
            p1 = self._proj1_in_merger
            return torch.ops.rpu.hyvla_merger_fused(
                proj1_out.reshape(4, -1, self._vit_out_dim()),
                M["pooler.predictor.0.weight_a"], M["pooler.predictor.0.bias"],
                M["pooler.predictor.0.weight_b"],
                M["pooler.predictor.2.weight"], M["pooler.predictor.2.bias"],
                M["proj2.weight"], M["proj2.bias"],
                self.w.vit_proj1_w if p1 else None,
                self.w.vit_proj1_b if p1 else None).reshape(B0, -1, D)
        # ⚠️ 下面全是 host 回退路径。C++ 的 ViT 尾部只看 PROJ1_IN_MERGER +
        #    member-major，**不看 MERGER_IN_GRAPH** ⇒ 关掉图内 merger 却留着
        #    PROJ1_IN_MERGER，host 侧会拿到没 proj1 过的 [S,1152]，**静默算错**。
        assert not (self._proj1_in_merger and self._fused_merger), (
            "RPU_HY_VLA_PROJ1_IN_MERGER=1 只在图内 merger 下成立；"
            "关掉 RPU_HY_VLA_MERGER_IN_GRAPH 时请一并设 "
            "RPU_HY_VLA_PROJ1_IN_MERGER=0")
        lin = lambda t, k: F.linear(
            t.reshape(-1, t.shape[-1]).contiguous(), M[k + ".weight"], M[k + ".bias"]
        ).reshape(*t.shape[:-1], -1)
        if self._fused_merger:
            # C++ 尾部（`RPU_HY_VLA_FUSED_MERGER`）已经把行序换成 member-major
            # ⇒ 这里的 reshape 是纯 view，**没有 permute、没有 host 往返**。
            nxc = proj1_out.reshape(4, -1, D)
            # pooled = Σ_m nxc[m]，平铺成 [4,G,D]。**没除 4** —— 1/4 折进了 Wb
            # （fp16 里乘 0.25 是精确的 2 的幂缩放），少一次 kernel。
            pooled = torch.ops.rpu.hyvla_merger_pool(nxc)
            # predictor.0 按 K 拆成两半，避免中间 `cat`。
            # 两半输出同形直接相加，**顺带避开首轴隐式广播**（那会静默产 NaN）。
            sc = (F.linear(nxc.reshape(-1, D), M["pooler.predictor.0.weight_a"],
                           M["pooler.predictor.0.bias"])
                  + F.linear(pooled.reshape(-1, D), M["pooler.predictor.0.weight_b"]))
            sc = lin(F.gelu(sc), "pooler.predictor.2").reshape(4, -1, D)
            out = torch.ops.rpu.hyvla_merger_combine(nxc, sc)
            return lin(F.gelu(out), "proj2").reshape(B0, -1, D)
        nxc = proj1_out.cpu().reshape(B0, 7, 2, 7, 2, D).permute(
            0, 1, 3, 2, 4, 5).reshape(B0, 7, 7, 4, D).contiguous()
        pooled = nxc.mean(-2, keepdim=True).expand(-1, -1, -1, 4, -1)
        sc = lin(_rpu16(torch.cat([nxc, pooled], -1)), "pooler.predictor.0")
        sc = lin(F.gelu(sc), "pooler.predictor.2")
        out = _rpu16((nxc * F.softmax(sc.cpu(), dim=-2)).sum(-2))
        return lin(F.gelu(out), "proj2").reshape(B0, -1, D)

    # ── 对外接口 ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def get_action(self, images: Sequence[torch.Tensor],
                   lang_tokens: torch.Tensor,
                   prefill_mask: torch.Tensor, prefill_pos: torch.Tensor,
                   modality_mask: torch.Tensor,
                   denoise_mask: torch.Tensor, suffix_pos: torch.Tensor,
                   noise: torch.Tensor,
                   state: Optional[torch.Tensor] = None,
                   state_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """一次完整推理 → `action [1, 50, 32]`。

        Args:
            images:        3 × `[1,3,224,224]` fp32。
            lang_tokens:   `[64]` int，prompt 的 token id（含 padding）。
            prefill_mask:  `[240,240]` bool，模型的 prefill attention mask。
            prefill_pos:   `[240]`，prefix 的 position_ids。
            modality_mask: `[240]`，True = vision 行（走 `_v` 塔）。
            denoise_mask:  `[51,291]` bool，denoise 的 attention mask。
            suffix_pos:    `[51]`，suffix 的 position_ids（从 prefix **有效**长度起算）。
            noise:         `[1,50,32]` fp32，Euler 的初始 x_t。
            state / state_emb: 二选一。`state` 走 `state_proj`；`state_emb`
                直接提供 `[1,1,1024]` 的已投影 state token。
        """
        if self._closed:
            raise RuntimeError("HyVlaRunner 已 close，不能再次推理。")
        w = self.w
        assert (state is None) != (state_emb is None), "state / state_emb 二选一"

        # ── 1) ViT ────────────────────────────────────────────────────────
        img_embs = self._run_vit(images)

        # ── 2) 组装 prefix（host）──────────────────────────────────────────
        prefix = self.assemble_prefix(img_embs, lang_tokens)

        # ── 3) VLM prefill ────────────────────────────────────────────────
        cache, total = self._cache, self._total
        self._run_vlm_prefill(cache, prefix, prefill_pos, prefill_mask,
                              modality_mask)

        # ── 4) expert denoise ×10 ─────────────────────────────────────────
        st = (state_emb if state_emb is not None else
              F.linear(state, w.host["state_proj.weight"],
                       w.host["state_proj.bias"])[:, None, :])
        return self._run_denoise(cache, total, prefill_pos, suffix_pos,
                                 denoise_mask, noise, st)

    # ── 分段实现 ──────────────────────────────────────────────────────────

    def _run_vlm_prefill(self, cache: RPUCache, prefix: torch.Tensor,
                         prefill_pos: torch.Tensor, prefill_mask: torch.Tensor,
                         modality_mask: torch.Tensor) -> None:
        """MoT 双权重 prefill —— 把 prefix KV 写进 `cache` 的 `[0,S)`。"""
        c, w, S = self.w.cfg, self.w, self.prefix_len

        def build_consts():
            vcos, vsin = build_rope_tables(prefill_pos[:S], c)
            vm = prefill_mask[:S, :S].clone()
            valid = prefill_mask.any(-1)[:S]
            for i in range(S):
                if not valid[i]:
                    vm[i, i] = True  # padding 行开对角口：CPU softmax 对全 -inf 行
                                     # 退化成均匀分布，RPU kernel 未必；因果性保证
                                     # 这些行的值流不回有效行。
            return (vcos, vsin,
                    _rpu16(torch.where(vm, 0.0, MASK_NEG).reshape(1, 1, S, S)),
                    _rpu16(1.0 - modality_mask[:S].float()))

        vcos, vsin, vmask, vmod = self._cached(
            "prefill", (prefill_pos, prefill_mask, modality_mask), build_consts)
        x_prefix = _rpu16(prefix[:, :S])

        # W8A16 时改调 `*_w8a16` 变体（多传 7 条 scale），C++ 侧落到同一个
        # set_weights_hyvla / set_moe_weights。**两条孪生各自独立**：
        # text 与 `_v` 是两组不同权重，各带各的 scale 列表。
        _P = ("q_w", "k_w", "v_w", "o_w", "q_norm", "k_norm",
              "input_norm", "post_norm", "gate_w", "up_w", "down_w")
        _S = ("q_ws", "k_ws", "v_ws", "o_ws", "gate_ws", "up_ws", "down_ws")

        def bind_vlm(h):
            targs = (h, *[w.vlm_text[k] for k in _P],
                     vcos, vsin, w.vlm_final_norm,
                     c.num_q_heads, c.kv_heads_rpu, c.head_dim, c.vlm_hidden,
                     c.vlm_inter, c.eps, True, [], [], [])
            margs = (h, *[w.vlm_vision[k] for k in _P],
                     [], [], [], w.vlm_final_norm, vmod, S)
            # 两条孪生独立选择位宽：
            # `set_weights_hyvla` 与 `set_moe_weights` 是两个独立入口，各自按
            # "有没有给 scale 列表"（`moe_q8`）独立校验 dtype，全程按张量
            # `scalar_type()` 派发 kernel，没有任何模型级的 w8a16 标志。
            if w.vlm_text["q_ws"]:
                torch.ops.rpu.hyvla_vlm_set_weights_w8a16(
                    *targs, *[w.vlm_text[k] for k in _S])
            else:
                torch.ops.rpu.hyvla_vlm_set_weights(*targs)
            if w.vlm_vision["q_ws"]:
                torch.ops.rpu.hyvla_vlm_set_moe_weights_w8a16(
                    *margs, *[w.vlm_vision[k] for k in _S])
            else:
                torch.ops.rpu.hyvla_vlm_set_moe_weights(*margs)

        vlm = self._sub("vlm", torch.ops.rpu.hyvla_vlm_create,
                        torch.ops.rpu.hyvla_vlm_destroy, bind_vlm,
                        rpu_backend.graph.GraphSignature(
                       op_id="hyvla_vlm_prefill", shapes=[S, c.vlm_hidden],
                       dyn_dims=[c.layers, 0, c.attn_tp, 0], dtypes=[torch.float16]))

        def prefill():
            cache.reset_to_position(0)
            with vlm.gc.capture(vlm.sig):
                torch.ops.rpu.hyvla_vlm_step_forward(
                    vlm.handle, x_prefix, cache.k_caches, cache.v_caches,
                    vcos, vsin, vmask, 0)
            torch.ops.rpu.spm_alloc_reset_temporary()

        with vlm:
            prefill()          # 唯一一次 forward（= BUILD），cache 里留下的就是 prefix KV

    def _run_denoise(self, cache: RPUCache, total: int,
                     prefill_pos: torch.Tensor, suffix_pos: torch.Tensor,
                     denoise_mask: torch.Tensor, noise: torch.Tensor,
                     st: torch.Tensor) -> torch.Tensor:
        """10 步 Euler，在 `cache` 的 `[S,S+51)` 上追加 suffix KV → `action`。"""
        c, w, S, L = self.w.cfg, self.w, self.prefix_len, self.w.cfg.suffix_len

        def build_consts():
            pos_tbl = torch.zeros(total, dtype=torch.float64)
            pos_tbl[:S] = prefill_pos[:S].double()
            pos_tbl[S:S + L] = suffix_pos.double()
            ecos, esin = build_rope_tables(pos_tbl, c)
            # denoise mask 的 prefix 块必须跟随 S（例如 S=192 时为 192+51）。
            dm = torch.cat([denoise_mask[:, :S], denoise_mask[:, -L:]], dim=1)
            return (ecos, esin,
                    _rpu16(torch.where(dm, 0.0, MASK_NEG).reshape(1, 1, L, S + L)))

        ecos, esin, emask = self._cached(
            "denoise", (prefill_pos, suffix_pos, denoise_mask), build_consts)

        def bind_exp(h):
            eargs = (h, *[w.expert[k] for k in ("q_w", "k_w", "v_w", "o_w",
                                                "q_norm", "k_norm", "input_norm",
                                                "post_norm", "gate_w", "up_w",
                                                "down_w")],
                     ecos, esin, w.expert_final_norm,
                     c.num_q_heads, c.kv_heads_rpu, c.head_dim, c.expert_hidden,
                     c.expert_inter, c.eps, L)
            if w.expert["q_ws"]:
                # ⚠️ unroll 的 encoder/out_proj（`set_action_weights` 的六个
                #    单核张量）不量化；它们是 M=51 的小 GEMM。
                torch.ops.rpu.hyvla_expert_set_weights_w8a16(
                    *eargs, *[w.expert[k] for k in ("q_ws", "k_ws", "v_ws",
                                                    "o_ws", "gate_ws", "up_ws",
                                                    "down_ws")])
            else:
                torch.ops.rpu.hyvla_expert_set_weights(*eargs)
            if self._unroll:
                torch.ops.rpu.hyvla_expert_set_action_weights(
                    h, *self._uw, c.action_dim, c.num_steps)

        exp = self._sub("expert", torch.ops.rpu.hyvla_expert_create,
                        torch.ops.rpu.hyvla_expert_destroy, bind_exp,
                        rpu_backend.graph.GraphSignature(
                       op_id="hyvla_expert_denoise_unroll" if self._unroll
                       else "hyvla_expert_denoise",
                       shapes=[L, c.expert_hidden],
                       dyn_dims=[c.layers, S, c.attn_tp,
                                 c.num_steps if self._unroll else 0],
                       dtypes=[torch.float16]),
                        variant=self._unroll)

        dt = -1.0 / c.num_steps

        if self._unroll:
            # 一次 forward 跑完 10 步。cache 只在进图前 reset 一次 —— 图内每个 body
            # 都以 position=S 插 KV，覆写同一段 51 行，与逐步 reset 等价。
            self._ux0[:, 1:, :].copy_(_rpu16(noise))
            with exp:
                cache.reset_to_position(S)
                with exp.gc.capture(exp.sig):
                    torch.ops.rpu.hyvla_expert_unroll_forward(
                        exp.handle, self._ux0, cache.k_caches, cache.v_caches,
                        _rpu16(st.reshape(-1)), ecos, esin, emask,
                        self._utraj, dt, S, c.num_steps)
                out = self._utraj[-1:].cpu().float()   # 读回早于 reset_temporary
                torch.ops.rpu.spm_alloc_reset_temporary()
            return out[:, 1:, :]

        def one_step(xt, k):
            emb = _rpu16(torch.cat([st, self._embed(xt, k)], dim=1))
            # 每步 reset 到 prefix 尾，保持 prefix KV 不变：
            # prefix KV：suffix 的 51 行每步覆写同一段，用完即弃。
            cache.reset_to_position(S)
            with exp.gc.capture(exp.sig):
                o = torch.ops.rpu.hyvla_expert_step_forward(
                    exp.handle, emb, cache.k_caches, cache.v_caches,
                    ecos, esin, emask, S)
            so = o.float().cpu()              # capture 退出后立刻读回
            torch.ops.rpu.spm_alloc_reset_temporary()
            up = torch.addmm(self._out_proj_b, so[0, -c.n_action:],
                             self._out_proj_wt)          # 见 `__post_init__`
            return xt + dt * up.reshape(1, -1, c.action_dim)

        # Keep the host Euler loop single-threaded and restore the caller setting.
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
          with exp:
            # k=0 是 BUILD，其输出就是第一个有效 Euler step。
            xt = noise.clone()
            for k in range(c.num_steps):
                xt = one_step(xt, k)
            return xt
        finally:
            torch.set_num_threads(prev_threads)


def build_hy_vla(
    ckpt_path: "str | Path",
    pos_embedding: torch.Tensor | None = None,
    prefix_len: int = 240,
    cfg: HyVlaConfig = HyVlaConfig(),
    *,
    _env_snapshot: dict[str, str | None] | None = None,
    _owner: object | None = None,
) -> HyVlaRunner:
    """装载 Hy-VLA：权重一次性全部上 RPU（规则 1），返回可反复调用的 runner。

    Args:
        pos_embedding: ViT 位置编码。**默认 `None` = 从 ckpt 自己的 `pos_embed`
            重采样**（`weights.sample_vit_pos_embedding`），无需外挂位置编码文件。
            传入张量则原样使用。
        prefix_len: `S = ceil16(有效 prefix 行数)`。合法区间 `{16,32,…,240}`。
            上限 240 同时是需求上限（`tokenizer_max_length=64` ⇒ 最坏
            `ceil16(176+64)=240`）和能力上限（S=256 差 375 KB 撞 SPM，
            S=272 撞 SDPA 的 `grid×gqa ≤ 8`）。默认取 240，覆盖受支持配置的最坏
            情形；短 prompt 想省时间才需要
            显式传 `ceil16(有效行数)`。

        ⚠️ 这是低层入口，但不是生命周期逃生口。一旦开始原生权重物化，
        该 Python 进程同样不能再构建另一个 RPU runner/model；需要重启进程。
    """
    if prefix_len % 16 or not 16 <= prefix_len <= 240:
        raise ValueError(f"prefix_len 必须是 16 的倍数且在 [16,240]，得到 {prefix_len}")
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"Hy-VLA 权重不存在: {ckpt_path}")
    from rpu_backend.api.causal_lm import (
        _claim_live_instance,
        _poison_live_instance,
    )

    direct_owner = _owner is None
    lifecycle_owner = _DirectHyVlaOwner() if direct_owner else _owner
    _claim_live_instance(lifecycle_owner)
    _poison_live_instance(
        lifecycle_owner,
        "Hy-VLA native materialization freezes process-global runtime switches",
    )
    # Hy-VLA 的模型级默认值。所有设置都使用 setdefault，因此显式 env/toml
    # 仍然优先，其他模型不会被这里覆盖。
    # Two-stage all-reduce 改变 fp16 求和顺序，不保证逐位等价。
    _setdefault_env("RPU_ALLREDUCE_TWOSTAGE", "1", _env_snapshot)
    # Keep the Euler sequence in the fused expert graph and SPM. Some GEMMs
    # use device fp16 rather than host fp32; set 0 for the stepwise path.
    _setdefault_env("RPU_HY_VLA_DENOISE_UNROLL", "1", _env_snapshot)
    # 三座子系统跨帧存活所必需的框架开关：多实例共存时，任一实例 Path-1 的
    # `reset_all()` 会推进 `persistent_generation_`；保留其他实例的 persistent
    # generation 才能继续 REPLAY。单实例模型下行为不变。
    _setdefault_env("RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN", "1", _env_snapshot)
    # 三座子系统跨帧复用 BUILD 和权重 preload。稳定地址前提见
    # `HyVlaRunner._sub`；prompt 布局变化会使缓存失效并重建。
    _setdefault_env("RPU_HY_VLA_PERSIST_HANDLES", "1", _env_snapshot)
    # Pack three camera streams with an explicit block-diagonal attention mask.
    _setdefault_env("RPU_HY_VLA_VIT_PACKED", "1", _env_snapshot)
    # KV-insert uses v16 for the aligned prefix and v2 for a disjoint tail.
    # The switch applies only when the launch shape satisfies v16 constraints.
    _setdefault_env("RPU_KVINSERT_HYBRID_V16", "1", _env_snapshot)
    # Permit v16 at non-eight-core TP only when KV heads divide the core count;
    # non-aligned expert sequences continue through the hybrid path.
    _setdefault_env("RPU_KVINSERT_V16_ANY_TP", "1", _env_snapshot)
    # Device merger uses member-major rows and device-fp16 group reductions.
    # Native and Python settings must agree because this changes the row layout.
    _setdefault_env("RPU_HY_VLA_FUSED_MERGER", "1", _env_snapshot)
    # 将 pool/combine、4 次 `F.linear` 和 2 次 `F.gelu` 收进单个图内 merger op。
    # GEMM 使用 acc32 以保持数值稳定。
    # ⚠️ 依赖 `RPU_HY_VLA_FUSED_MERGER=1` 的 member-major 布局（Python 侧有断言）。
    _setdefault_env("RPU_HY_VLA_MERGER_IN_GRAPH", "1", _env_snapshot)
    # patch_embed 多核执行与图内 capture 必须配套开启。权重 swizzle 的核数由
    # `weights._patch_embed_cores` 使用同一环境变量选择。
    _setdefault_env("RPU_HY_VLA_PATCH_EMBED_MC", "1", _env_snapshot)
    _setdefault_env("RPU_HY_VLA_PATCH_EMBED_IN_GRAPH", "1", _env_snapshot)
    # Partial-TP two-stage all-reduce zeros inactive core slots first. This
    # profile guarantees those slots are dead; the shared default remains off.
    _setdefault_env("RPU_ALLREDUCE_PARTIAL_TWOSTAGE", "1", _env_snapshot)
    # Profile-scoped exact tile controls. These shared selectors remain globally
    # off and are fixed before the Hy-VLA handles are installed.
    _setdefault_env("RPU_LINEAR_TILE_GY1", "1", _env_snapshot)
    _setdefault_env("RPU_LINEAR_TILE_M64", "32", _env_snapshot)
    _setdefault_env("RPU_LINEAR_TILE_M384W", "80", _env_snapshot)
    # 三座各自跳过 REPLAY 期重走 op stream（`fast_replay_skip_layer_loop`）。图已经
    # BUILT 后无需在 REPLAY 期重走 op stream；真正执行由 `graph.end()` 提交。
    # 契约：body 不得有 kernel 会读的 per-call
    # host 副作用。本模型的三条可变输入都走合法出口：层 0 的 hidden 走
    # `hidden_in_src_base_` mutable DMA、显式 2D mask 由 dynamic_config 拷进稳定
    # DDR 槽、unroll 的 x0/x_traj 走 op 序言里的 `*_mutable(&member)` 基址。
    _setdefault_env("RPU_HY_VLA_FAST_REPLAY", "1", _env_snapshot)
    # 权重未变时跳过 REPLAY 期的 preload host 重走；已记录的 DMA 节点仍会重放到
    # 同一 persistent SPM。非持久或 weights_dirty 的 handle 自动回退完整路径。
    _setdefault_env("RPU_HY_VLA_FAST_REPLAY_PRELOAD", "1", _env_snapshot)
    # 把 4 个 KV head 各复制一份成 8 个槽位，使 `attn_tp` 从
    # `min(8,4)=4` 自动变成 8，**q/k/v/o 四个投影从 4 核变 8 核**。
    # 槽位映射正好对得上：q 按 `_col(·,8)` 落成核 c ↔ q head 2c,2c+1，原始 gqa
    # 分组是 head h ↔ kv head h//4 ⇒ 核 c 要 `floor(2c/4)` = `floor(c/2)` = 槽 c。
    # ⚠️ **VLM 与 expert 共享同一个 KV cache，必须一起开**（`_cache` 那里有注解）。
    # 代价：k/v 权重与 KV cache 的 DDR 各翻倍（cache 19→38 MB）、SDPA 读 KV 翻倍。
    # ⚠️ 8 核与 4 核的归约树不同，不保证逐位等价。启用后
    # `RPU_ALLREDUCE_PARTIAL_TWOSTAGE` 不再为这些投影触发。
    _setdefault_env("RPU_HY_VLA_ATTN_TP8", "1", _env_snapshot)
    # Keep the invariant 2D mask in layer-wide SPM for the full layer body;
    # its buffer must not overlap phase-local Q/K/V storage.
    _setdefault_env("RPU_HY_VLA_MASK_ONCE", "1", _env_snapshot)
    # Single-query-chunk ViT aliases Q across its two phases. The alias is
    # valid only while the layer-wide lifetime covers both consumers.
    _setdefault_env("RPU_HY_VLA_Q_INPLACE", "1", _env_snapshot)
    # Round KV storage capacity to 16 rows. Logical `kv_seq_len` excludes the
    # padding region, so padded cache rows are never visible to SDPA.
    _setdefault_env("RPU_HY_VLA_KVPAD16", "1", _env_snapshot)
    # Fused `silu_mul` defaults to expert. Accepted values are `expert`, `vlm`,
    # `1` for both, or empty to disable.
    _setdefault_env("RPU_HY_VLA_SILU_MUL", "expert", _env_snapshot)
    # `llama_rms_norm` 的多行变体按 M 自动选择最宽的可用 V。
    # ⚠️ 只在 `M % V == 0` 时生效 ⇒ 本模型只有 **VLM 座**吃到（M=240 / 480）；
    #    expert 的 51/102 与 ViT 的 588 自动回落基版（ViT 走的还是 layer_norm）。
    # `auto` 不放宽 `M % V == 0` 的前提。
    _setdefault_env("RPU_RMSNORM_VWARP", "auto", _env_snapshot)
    # profiler 未启用时跳过空闲的 `record_function`；启用 profiler 时完整保留区间。
    # ⚠️ 只在 `torch.autograd._profiler_enabled()` 为 False 时跳过
    # ⇒ **开着 profiler 时行为逐字不变**，trace 里的区间一个不少。
    # 共享文件（`_bootstrap.py`）⇒ 全局默认 OFF，只在这里 setdefault。
    _setdefault_env("RPU_SKIP_IDLE_RECORD_FUNCTION", "1", _env_snapshot)
    # 使用融合 all-reduce，减少 reduce-scatter 与 all-gather 之间的单独提交。
    # ⚠️ 融合版使用 fp32 累加，而两阶段路径是分段求和，因此不保证逐位等价。
    _setdefault_env("RPU_ALLREDUCE_FUSED", "1", _env_snapshot)
    # VLM 与 expert 使用 `partial_mrope`。当 `rotary_dim == head_dim` 时，它与
    # `llama_rope` 使用相同逐 token 表；`cos_sin_start` 是表内行偏移。
    # 取值：`expert` / `vlm` / `1`（两座全开）/ 空（关）。C++ 侧默认 OFF，只在这里 setdefault。
    _setdefault_env("RPU_HY_VLA_PARTIAL_ROPE", "expert,vlm", _env_snapshot)
    # Skip parameter sync only for replays with no layer-loop register writes;
    # mutable DMA inputs remain updated separately. The shared default is off.
    _setdefault_env("RPU_FASTREPLAY_SKIP_SYNC", "1", _env_snapshot)
    # 把超过 v2 上限的 all-reduce 按行拆块，使其继续使用 two-stage v2。
    # `kV2Cap = 400×1024`，而**本模型三座里正好两座超标**：
    #     expert 51×1024 =  52,224  <= cap
    #     VLM   240×2048 = 491,520  >  cap
    #     ViT   588×1152 = 677,376  >  cap
    # ⚠️ 分块改变求和结合顺序，不保证逐位等价。不要同时启用
    # `RPU_ALLREDUCE_CHUNK_BALANCED`。
    _setdefault_env("RPU_ALLREDUCE_CHUNK_V2", "1", _env_snapshot)
    # Hy-VLA 将 v2 的 M*N 上限设为 700000，使三座受支持形状
    # 52,224 / 491,520 / 677,376 均可单块执行。CHUNK_V2 保留为更大形状的兜底。
    # ⚠️ 不拆块会改变求和顺序；该上限只覆盖 M*N <= 677,376 的当前模型形状。
    # ⚠️ 共享 launcher，全局默认仍是 409600，只在这里 setdefault ⇒ 其他模型逐位不变。
    _setdefault_env("RPU_ALLREDUCE_V2_CAP", "700000", _env_snapshot)
    # W8A16 expert M=51 使用 n32；该选择只改变 tile/grid，不改变算子语义。
    # 不将 w16 的 `prefer_gy1_tile` 设置套用到 W8A16。
    _setdefault_env("RPU_LINEAR_TILE_W8_M64", "32", _env_snapshot)
    _setdefault_env("RPU_LINEAR_TILE_W8_M64N", "32", _env_snapshot)
    # ⚠️ **`RPU_HY_VLA_W8A16` 故意不在这里 setdefault。**
    # W8A16 会改变数值，必须由部署方显式 opt-in。
    # 取值：`vit` / `vlm` / `expert`（可逗号组合）或 `1`/`all`。
    # DDR caching allocator（`RPU_HY_VLA_CACHING_ALLOC=0` 关掉）。
    # 设为 0 可禁用缓存分配器，便于在部署环境中回退。
    if os.environ.get("RPU_HY_VLA_CACHING_ALLOC", "1").strip() not in (
            "0", "false", "False", "off"):
        torch.rpu.set_caching_allocator(True)
    runner = HyVlaRunner(w=load_hy_vla_weights(ckpt_path, pos_embedding, cfg),
                         prefix_len=prefix_len)
    if direct_owner:
        # Keep the anonymous low-level claim alive with its runner. Never add
        # this back-reference for the facade: weakref.finalize(policy, ..., runner)
        # strongly owns runner, so runner -> policy would make policy immortal.
        runner._lifecycle_owner = lifecycle_owner
    return runner
