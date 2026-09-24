"""HALO text_decode_step（mode="und" subtask 文本自回归解码单步）→ RPU（WS-2 组件 C）。

旁路口径（``use_subtask=True``，**非部署 action 路径**）下 ``Bagel.generate_text``
(bagel.py:1355) 每步：``forward_inference(mode="und", is_causal=True)`` 产 post-norm
hidden（``packed_query_sequence``，bagel.py:1403）→ ``lm_head``（:1404）→ logits →
argmax（``do_sample=False``，:1410）。assess verdict = **Adapter-only**（同 sibling
A，``text_cache.py``）：复用通用 ``causal_decoder_*`` op + HALO und/base 权重，
**零新 C++**。C 唯一比 A 多的（vocab logits）走 **host** ``F.linear``：HALO 实际
``language_model.lm_head.weight`` = [151933,1536]，151933 % NUM_CORES(8) = 5 ≠ 0 →
片上 fused lm_head 的 ``vocab % 8 == 0`` 硬约束（``rpu_qwen3_model.h:635/681``）拒绝
它（pad 151933→152064 + argmax 前 mask 131 行垃圾 = gratuitous）；host 投影镜像动作
专家 M1 边界，零对齐风险。

机制：
  * **seq=1 decode**：query = 1 token，attend 长前缀 P。传 ``is_causal=True`` +
    ``attention_mask=None``：入口守卫（``rpu_qwen3_model.h:502`` 拒 ``is_causal=false``
    且无 mask）放行；``seq_len==1`` 时内层 ``sdpa_causal = is_causal && (seq_len > 1)``
    = false → **MASK_NONE**（full-attend ``kv_seq_len = P+1``，tail 因果，**不触发 LTM
    前缀对齐拒绝**），故 C **无需** A 的显式 2D causal-prefix mask（``:1287``）。
  * **RoPE 解耦**（与 A cond regime 同源根因，已证）：HALO MoT 压缩图像 token 的 RoPE
    （``bagel.py:387-390``：kv_len +num_img+2，rope 只 +1，同 ``curr_position_id``），
    decode token 的语义 position ``P_sem``（rope 续尾，实测≈11）与 KV 落位 ``P_idx``
    （物理 kv_len）**解耦**。复用 A 的 ``cos_sin_offset``：
    seq=1 时单 chunk ``chunk.offset=0`` ⇒ ``cos_sin_start == cos_sin_offset``，设
    ``cos_sin_offset=P_sem`` / ``position=P_idx``（``rpu_qwen3_model.h:1078``）。
  * **lm_head host**：forward 返回 post-final-norm hidden ``[1,1,H]``，**capture 退出
    后**读（graph end() 写入，作用域内提前读会得全 0；同 ``runtime.py:259``）→ host
    ``F.linear(hidden.float(), lm_head_w.float())`` → logits → argmax。

验收（先验证再定 gate，2026-06-16 拍板）：cond 长前缀 regime argmax-token-match 硬门
+ hidden cos/MSE 诊断（cond 可能继承 A 的 L16 v3.x kernel 残差，先实测再定 hidden 是否
入硬门）；clean 短前缀 regime（如构造）hidden cos≥0.999 证 port 正确。proof 边界：数值
正确性由 @hardware .81 测试对 golden gate，本模块自身不构成硬件证据。

权重复用：``weights.py:load_halo_branches`` 的 text（und/base）分支 + 新增 host
``lm_head_w``。前缀注入复用 ``runtime._swizzle_prefix_7d``（7-D cache 布局，与 kernel
DMA 契约一致）。RoPE 表复用 A 的 ``build_seq_rope_tables``（顺序 1D RoPE，theta=1e6）。
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Tuple

import torch

import rpu_backend
from rpu_backend.adapters.halo.runtime import _halo_execution_serialized as execution_serialized
from rpu_backend.api.cache import RPUCache
from rpu_backend.adapters.halo.runtime import (
    HALO_TEXT_DECODE_COMPONENT,
    HaloExecutionController,
    _HaloResources,
    _cleanup_halo_build,
    _close_halo_runner,
    _S_CHUNK,
    _coerce_halo_execution_controller,
    _halo_component_state,
    _record_halo_plan,
)
from rpu_backend.adapters.halo.text_cache import _inject_text_prefix, build_seq_rope_tables
from rpu_backend.adapters.halo.weights import (
    validate_halo_core_layout,
    HaloBranchWeights,
    HaloConfig,
    HaloHostWeights,
    load_halo_branches,
)
from rpu_backend.runtime.execution_planner import (
    GRAPH_RETAINED_CACHE,
    _decode_native_stage_domain,
)
from rpu_backend.runtime.decoder import plan_native_component_execution


def _destroy_causal_decoder(handle: int) -> None:
    """释放通用 CausalDecoderModel handle（weakref.finalize 在 GC 时调）。"""
    torch.ops.rpu.causal_decoder_destroy(handle)


@dataclasses.dataclass
class HaloTextDecodeStep:
    """HALO und 文本自回归解码-单步 runner（经 :func:`build_halo_text_decode` 构造）。

    复用通用 ``causal_decoder_*`` op + HALO und/base 权重 + host lm_head。
    ``_cos``/``_sin`` 是顺序 1D-RoPE 表（与 handle 等寿命，set_weights 已绑同一份）。
    所有权：RPU 权重/表/cache 在 runner 存活期有效；``_handle`` 由 finalize 释放。
    """

    cfg: HaloConfig
    text: HaloBranchWeights
    host: HaloHostWeights
    cache: RPUCache
    max_seq_len: int
    _handle: int
    _graph_cache: "rpu_backend.graph.GraphCache"
    _cos: torch.Tensor
    _sin: torch.Tensor
    _decode_descriptor: tuple[int, ...]
    _handle_finalizer: object | None = dataclasses.field(
        init=False, default=None, repr=False
    )
    _closed: bool = dataclasses.field(init=False, default=False, repr=False)
    # Prefix contents update in place; only changed DMA banks advance the key.
    _prefix_epoch: int = dataclasses.field(init=False, default=0)
    _prefix_cache_identity: tuple | None = dataclasses.field(init=False, default=None, repr=False)

    def close(self) -> None:
        """Clear graph ownership, then retire the decoder handle once."""
        _close_halo_runner(self, graphs=(self._graph_cache,),
                           handles={"_handle": (self._handle, _destroy_causal_decoder)})

    def _embed_token(self, token_id: torch.Tensor) -> torch.Tensor:
        """单 decode token → ``[1, 1, H]`` fp16 RPU。

        跟 HALO runtime 路径（``generate_text`` 走 ``embed_tokens`` + 同 A 的
        autocast embed）：``F.embedding`` → ``.float()`` → ``.to(bf16).to(fp16)``，
        匹配 golden 的 bf16-autocast embed。
        """
        ids = token_id.to(torch.long).reshape(1).cpu()
        emb = torch.nn.functional.embedding(ids, self.host.embed_tokens).float()  # [1,H]
        emb = emb.to(torch.bfloat16).to(torch.float16)
        return emb.unsqueeze(0).to(device="rpu").contiguous()                     # [1,1,H]

    def _inject_prefix(
        self, prefix_k: List[torch.Tensor], prefix_v: List[torch.Tensor], prefix_len: int
    ) -> None:
        """灌入 28 层 decode 前缀 K/V（CUDA NaiveCache ``[P,2,128]`` bf16/fp32）。

        前缀 = 图像 prefill + 文本 prompt prefill（cond regime 实测 P=4706）。
        复用 text_cache 的稳址 7-D 写入，再 ``reset_to_position(P)``（新 token K/V
        落 ``[P, P+1)``）；换址才 invalidate / 更新 signature，内容变化不换 key。
        """
        _inject_text_prefix(self, prefix_k, prefix_v, prefix_len)

    def _execution_plan(self, position: int):
        # The native M=1 getter emits a position-zero decode template; forward
        # binds the actual KV/RoPE positions. Its retained lifecycle is not the
        # controller's logical COMPOSITE_CHILD label.
        if type(position) is not int or position < 0:
            raise ValueError("HALO text decode position must be a non-negative integer")

        def domain(_length):
            # A cost query/calibration requires this handle's live admission,
            # not the descriptor saved when the runner was constructed.
            descriptor = tuple(torch.ops.rpu.causal_decoder_resolve_decode_stage_descriptor(self._handle))
            native, = _decode_native_stage_domain(
                (1, 1, len(descriptor), *descriptor),
                execution_len=1, logical_len=1, position=0,
                graph_mode=GRAPH_RETAINED_CACHE,
            )
            if (dict(native.physical_metadata).get("manifest_state") != 1
                    or native.input_chunk != native.qkv_chunk
                    or native.qkv_chunk != native.compute_chunk):
                raise ValueError("HALO text decode requires one COMPLETE fixed native stage")
            return (native,)

        stage_config, generation = _halo_component_state(
            self, HALO_TEXT_DECODE_COMPONENT, "decode")
        _, plan = plan_native_component_execution(
            1,
            component_id=HALO_TEXT_DECODE_COMPONENT, stage="decode",
            generation=generation, execution={"decode": stage_config},
            resolve_stage_domain=domain,
            graph_mode=GRAPH_RETAINED_CACHE,
            physical_metadata=((f"component:{HALO_TEXT_DECODE_COMPONENT}", 1),
                ("fixed_abi_singleton", 1), ("position", position), ("stage:decode", 1)),
            queue_owner_id=int(self._handle),
            execution_owner=self, execution_component=HALO_TEXT_DECODE_COMPONENT,
            execution_native=("causal_decoder", int(self._handle)),
            plan_signature=(),
            graph_cache=self._graph_cache,
        )
        self._decode_descriptor = plan.selected.stage_tuple.physical_descriptor
        return plan

    @torch.no_grad()
    @execution_serialized
    def decode_step(
        self,
        start_token_id: torch.Tensor,
        query_position_id: torch.Tensor,
        prefix_kv: Tuple[List[torch.Tensor], List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """一次 und 解码单步。返回 ``(hidden [H], logits [vocab], argmax_token)`` CPU fp32。

        I/O 契约（golden ``step0/`` 同名字段，golden 由 ``generate_text`` 首步捕获）：
          * ``start_token_id`` 标量/``[1]`` int —— ``packed_start_tokens``（首步 = BOS）。
          * ``query_position_id`` 标量/``[1]`` int —— ``packed_query_position_ids``，
            驱动 1D RoPE 的**语义** position ``P_sem``（cond regime 与 KV 落位偏移
            解耦：实测 ≈11 vs 落位 4706）。
          * ``prefix_kv`` = (28 层 k, 28 层 v) 各 ``[P,2,128]`` —— decode 进入时的前缀
            KV（图像 + 文本 prompt prefill，cond regime golden ``step0/prefix_kv.pt``）。

        走 ``is_causal=True`` + ``attention_mask=None``（seq=1 → MASK_NONE，过入口守卫，
        无需显式 mask）；KV 插入于 ``position=P``，RoPE 由 ``cos_sin_offset=P_sem`` 驱动
        （C++ WS-2 解耦，见 ``rpu_qwen3_model.h:rope_position_base_``）。
        """
        cfg = self.cfg
        prefix_k, prefix_v = prefix_kv
        prefix_len = int(prefix_k[0].shape[0])
        if prefix_len + 1 > self.max_seq_len:
            raise ValueError(
                f"prefix_len+1 {prefix_len + 1} > max_seq_len {self.max_seq_len}")

        pos = query_position_id.to(torch.long).reshape(-1).cpu()
        if pos.numel() != 1:
            raise ValueError(f"query_position_id 须单元素，得 shape {tuple(pos.shape)}")
        rope_base = int(pos[0].item())                       # P_sem（RoPE 表起点）
        if rope_base < 0 or rope_base + 1 > self.max_seq_len:
            raise ValueError(
                f"rope_base {rope_base} 越界 [0,{self.max_seq_len})（RoPE 表行数不足）")

        # Validate the actual native singleton before prefix DMA or embedding.
        plan = self._execution_plan(prefix_len)
        descriptor = plan.selected.stage_tuple.physical_descriptor
        self._inject_prefix(prefix_k, prefix_v, prefix_len)
        ie = self._embed_token(start_token_id)               # [1,1,H] fp16 RPU
        # sig 含 prefix_len（KV 落位 P_idx）+ seq=1 + rope_base（RoPE 起点 P_sem，
        # BUILD 期烘焙进 cos_sin_start kernel 常量）+ prefix_epoch（DMA 指针代数）。
        # Codex Q5：decode 每步 P_idx/P_sem 递增，必须进 key 防陈旧 replay 用错
        # KV 落位 / 错 SDPA 长度（kv_seq_len = position+off+len 建图期烘焙）。
        sig = rpu_backend.graph.GraphSignature(
            op_id="halo_text_decode_step",
            shapes=[1, cfg.hidden_size],
            dyn_dims=[cfg.num_layers, prefix_len, 1, rope_base,
                      self._prefix_epoch, *plan.graph_key_words()],
            dtypes=[torch.float16],
        )
        with self._graph_cache.capture(sig):
            hidden_out = torch.ops.rpu.causal_decoder_forward(
                self._handle, ie, self.cache.k_caches, self.cache.v_caches,
                None,                # attention_mask=None（无显式 mask）
                prefix_len,          # position：新 K/V 插入于 [P, P+1)
                True,                # is_causal=True：过入口守卫（rpu_qwen3_model.h:502
                                     # 拒 is_causal=false+无 mask）；seq=1 时内层
                                     # sdpa_causal=is_causal&&(seq_len>1)=False → MASK_NONE
                                     # （full-attend P+1，tail 因果），不触发 LTM 前缀对齐检查
                None,                # position_ids（M-RoPE 专用，1D 路径忽略）
                [],                  # deepstack_dense_visual_embeds（无）
                cos_sin_offset=rope_base,  # 1D RoPE 表起点（与落位解耦）。
                planned_stage_descriptor=descriptor,
                # Pass cos_sin_offset by name after the optional GR00T RoPE tensors.
            )
        self.cache.update_position(1)
        # capture 退出**后**读 hidden（graph end() 已写；作用域内提前读得全 0）。
        hidden = hidden_out[0, 0].float().cpu()              # [H] post-final-norm
        logits = torch.nn.functional.linear(
            hidden, self.host.lm_head_w.float())             # [vocab] host fp32 投影
        token = int(torch.argmax(logits).item())
        _record_halo_plan(
            self, HALO_TEXT_DECODE_COMPONENT, "decode", plan,
            authority="NATIVE_FMB_DESCRIPTOR", kv_route="DDR_REQUIRED",
        )
        return hidden, logits, token


def build_halo_text_decode(
    ckpt_path: "str | Path",
    *,
    cfg: HaloConfig = HaloConfig(),
    max_seq_len: int = 4736,
    rpu_execution=None,
    execution_controller: HaloExecutionController | None = None,
) -> HaloTextDecodeStep:
    """装载 HALO und/base 权重 + host lm_head → 通用 causal_decoder handle → set_weights。

    Args:
        ckpt_path: HALO ema.safetensors（bf16）。
        max_seq_len: KV cache 容量 + 顺序 RoPE 表行数，须 16 倍数且 ≥ 部署最长
            decode 落位（cond 前缀 4706 + decode tokens → 默认 4736）。
    """
    validate_halo_core_layout(cfg)
    execution_controller = _coerce_halo_execution_controller(
        rpu_execution,
        execution_controller,
        entry_point="build_halo_text_decode",
        components=(HALO_TEXT_DECODE_COMPONENT,),
    )
    if max_seq_len % _S_CHUNK != 0:
        raise ValueError(f"max_seq_len {max_seq_len} must be a multiple of {_S_CHUNK}")
    text, _act, host = load_halo_branches(Path(ckpt_path), cfg, load_lm_head=True)
    # CLASS-D 契约:lm_head_w 现为 Optional,text_decode 是唯一消费者
    # (host F.linear vocab 投影,见 decode_step)。在 adapter 边界显式报错 → 误用/缺 key 早爆,
    # 而非 decode_step 里 `lm_head_w.float()` 的晚发 AttributeError。
    if host.lm_head_w is None:
        raise ValueError(
            "build_halo_text_decode 需 lm_head,但 load_halo_branches(load_lm_head=True) 未返回 "
            "lm_head_w(ckpt 缺 language_model.lm_head.weight?)。")
    cos, sin = build_seq_rope_tables(cfg, max_seq_len)
    handle = int(torch.ops.rpu.causal_decoder_create())
    resources = _HaloResources(execution_controller, handles={"_handle": (handle, _destroy_causal_decoder)})
    try:
        # 通用 set_weights arg 序（rpu_qwen3_model.cpp:54-89），与 A 同一份 und/base
        # 权重（q_norm/k_norm Qwen3 式 qk-norm + q/k/v_bias Qwen2 式 qkv bias，
        # mrope_section=[] 1D RoPE、deepstack_lang_layers=[]）。**不** set_lm_head
        # （走 host F.linear；fused 路径 vocab%8 拒绝 151933）。
        torch.ops.rpu.causal_decoder_set_weights(
            handle,
            text.q_w, text.k_w, text.v_w, text.o_w,
            text.q_norm, text.k_norm,
            text.input_norm, text.post_norm,
            text.gate_w, text.up_w, text.down_w,
            cos, sin, text.final_norm_w,
            cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim,
            cfg.hidden_size, cfg.intermediate_size,
            cfg.rms_norm_eps, True,       # use_silu (SwiGLU)
            [], [],                       # mrope_section, deepstack_lang_layers
            text.q_bias, text.k_bias, text.v_bias)
        # Use the same max_seq_len bound as the KV cache and RoPE table.
        # Register it before SPM planning to prevent oversized chunk selection.
        # A zero chunk size preserves automatic planning.
        torch.ops.rpu.causal_decoder_set_chunk_envelope(handle, int(max_seq_len), 0)
        decode_descriptor = tuple(
            torch.ops.rpu.causal_decoder_resolve_decode_stage_descriptor(handle)
        )
        if not decode_descriptor:
            raise RuntimeError("HALO text decode planner returned no descriptor")
        cache = RPUCache(
            num_layers=cfg.num_layers, batch_size=1, max_seq_len=max_seq_len,
            num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim, attn_tp=cfg.attn_tp)
        graph = rpu_backend.graph.GraphCache()
        resources.graphs.append(graph)
        runner = HaloTextDecodeStep(
            cfg=cfg, text=text, host=host, cache=cache, max_seq_len=max_seq_len,
            _handle=handle, _graph_cache=graph,
            _cos=cos, _sin=sin, _decode_descriptor=decode_descriptor)
        resources.bind(runner)
        execution_controller.attach(
            runner, {HALO_TEXT_DECODE_COMPONENT: (runner._graph_cache,)}
        )
    except BaseException as build_error:
        # 构造任一步失败 → 孤儿 handle 立即回收（同 build_halo_text_cache）。
        _cleanup_halo_build(resources, build_error)
        raise
    return runner
