"""Controlled InternVLA-N1 System-2 Qwen2.5-VL-7B RPU evaluator.

System-2 produces `vlm_tokens = hidden_states[-1][:, -4:, :]` = [1,4,3584] (the final-layer
hidden at the 4 latent-query positions) consumed by NavDP System-1. This adapter runs the
28-layer text decoder on RPU through the generic `causal_decoder_*` op family — the SAME fused
subsystem Wall-OSS reuses (`src/fused/rpu_qwen3_model.{cpp,h}`); NO new C++.

Delta vs `adapters/wall_oss/llm.py`:
  * Stock non-MoT keys: `model.layers.L.self_attn.{q,k,v,o}_proj.weight` (q/k/v carry `.bias`,
    o has none) and separate `mlp.{gate,up,down}_proj.weight` — no fused qkv/gate_up split.
  * 7B dims: 28L / H3584 / NQ28 / NKV4 / HD128 / INTER18944 / mrope[16,24,24] / theta1e6 / eps1e-6.
  * `attn_tp`: NKV=4. Default 4 (28%4==0 clean GQA, 7 q/kv). The Wall-OSS ATTN_TP8 kv-replication
    does NOT apply here — 28 q-heads is not a multiple of 8 (would break num_q%num_kv and the
    8-core q partition). So this port runs the attention phase 4-core.
  * lm_head is NOT fused (vocab 151668 % 8 != 0) and not needed for vlm_tokens — embed stays CPU.

The head/tail glue (embed lookup, image-embed scatter, latent-query append, 3D M-RoPE
position_ids, last-4 slice) is done by the caller / `forward_embeds`; this class owns only the
28-layer decoder prefill. Reuses wall_oss's swizzle / RoPE / cache / graph-capture machinery.
Full-hidden numerics are record-only in the exact full-chain contract; this remains a
controlled evaluator rather than a generic supported-model path.
"""
from __future__ import annotations

import json
import weakref
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

import torch

import rpu_backend
from rpu_backend.runtime.rope_partial import build_interleaved_mrope_cos_sin
from rpu_backend.runtime.control import rpu_env_bool
from rpu_backend.api.cache import RPUCache
from rpu_backend.runtime.weights import tp_col_swizzle_mc_weight, tp_row_swizzle_mc_weight
from rpu_backend.adapters.wall_oss.llm import (
    _SafeTensorStore, _build_rope_tables, _to_rpu_half, _to_rpu_int8,
    _partial_mrope_enabled, _destroy_causal_decoder_handle,
)
from ._policy import require_controlled_evaluation, verify_asset_manifest


_PROFILE = {
    "hidden_size": 3584,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "intermediate_size": 18944,
    "num_hidden_layers": 28,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
}
_MROPE_SECTION = [16, 24, 24]
_MAX_SEQ_LEN = 4096
_CERTIFIED_MAX_KV_LEN = 1936
_CERTIFIED_CHUNK_SIZE = 176
_CUSTOMER_LOGICAL_PREFILL = 1904
_CUSTOMER_EXECUTION_PREFILL = 1936


def _expected_checkpoint_profile() -> dict[str, tuple[tuple[int, ...], str]]:
    hidden, kv, inter = 3584, 512, 18944
    expected = {"model.norm.weight": ((hidden,), "F16")}
    projections = {
        "self_attn.q_proj": (hidden, hidden),
        "self_attn.k_proj": (kv, hidden),
        "self_attn.v_proj": (kv, hidden),
        "self_attn.o_proj": (hidden, hidden),
        "mlp.gate_proj": (inter, hidden),
        "mlp.up_proj": (inter, hidden),
        "mlp.down_proj": (hidden, inter),
    }
    for layer in range(28):
        prefix = f"model.layers.{layer}."
        for name, shape in projections.items():
            expected[prefix + name + ".weight"] = (shape, "I8")
            expected[prefix + name + ".weight_scale"] = ((shape[0],), "F16")
        expected.update({
            prefix + "self_attn.q_proj.bias": ((hidden,), "F16"),
            prefix + "self_attn.k_proj.bias": ((kv,), "F16"),
            prefix + "self_attn.v_proj.bias": ((kv,), "F16"),
            prefix + "input_layernorm.weight": ((hidden,), "F16"),
            prefix + "post_attention_layernorm.weight": ((hidden,), "F16"),
        })
    return expected


def _validate_checkpoint_profile(store: _SafeTensorStore) -> None:
    errors = []
    for key, (shape, dtype) in _expected_checkpoint_profile().items():
        shard = store.weight_map.get(key)
        if shard is None:
            errors.append(f"missing {key}")
        else:
            header, _ = store._header(shard)
            metadata = header.get(key)
            actual_shape = None if metadata is None else tuple(metadata.get("shape", ()))
            actual_dtype = None if metadata is None else metadata.get("dtype")
            if actual_shape != shape or actual_dtype != dtype:
                errors.append(
                    f"{key}: shape/dtype={actual_shape}/{actual_dtype}, "
                    f"expected {shape}/{dtype}"
                )
        if len(errors) >= 8:
            break
    if errors:
        raise ValueError(
            "InternVLA-N1 System-2 checkpoint does not match the fixed 28-layer "
            f"W8A16 profile: {'; '.join(errors)}"
        )


class Qwen25VLBackbone:
    """Stock Qwen2.5-VL-7B text decoder on RPU (System-2 prefill).

    Construct via :func:`build_qwen25vl_backbone`. Call :meth:`forward_embeds` with
    pre-assembled `inputs_embeds` [1, seq, 3584] (vision scattered into image-token
    positions + latent queries appended) and real 3D M-RoPE `position_ids` [seq, 3];
    returns the final-normed `last_hidden` [1, seq, 3584] on RPU. Slice `[:, -4:, :]`
    for the vlm_tokens.
    """

    def __init__(self, *, handle, cache, graph_cache, hidden_size, num_layers,
                 head_dim, rope_theta, mrope_section, partial_mrope, host_cache):
        self._handle = handle
        self._handle_finalizer = weakref.finalize(
            self, _destroy_causal_decoder_handle, handle
        )
        self._head_dim = head_dim
        self._rope_theta = rope_theta
        self._mrope_section = mrope_section
        self._partial_mrope = partial_mrope
        self.cache = cache
        self._graph_cache = graph_cache
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._prefill_rope_cache = {}
        self._prefill_position_cache = {}
        self._prefill_input_cache = {}
        self._host_cache = host_cache

    def destroy(self) -> None:
        """Release graph/native resources. Safe to call more than once."""
        if getattr(self, "_handle", None) is None:
            return
        graph_cache = getattr(self, "_graph_cache", None)
        if graph_cache is not None:
            graph_cache.clear()
        finalizer = getattr(self, "_handle_finalizer", None)
        if finalizer is not None and getattr(finalizer, "alive", False):
            torch.ops.rpu.causal_decoder_destroy(self._handle)
            finalizer.detach()
        elif finalizer is None:
            torch.ops.rpu.causal_decoder_destroy(self._handle)
        self._handle = None

    def _prefill_inputs_embeds(self, inputs_embeds: torch.Tensor, seq: int):
        if inputs_embeds.device.type != "cpu" or not self._host_cache:
            return _to_rpu_half(inputs_embeds)
        source = inputs_embeds.to(dtype=torch.float16).contiguous()
        cached = self._prefill_input_cache.get(seq)
        if cached is None or cached.shape != source.shape:
            cached = source.to(device="rpu").contiguous()
            self._prefill_input_cache[seq] = cached
        else:
            cached.copy_(source)
        return cached

    def _prefill_position_ids(self, position_ids: torch.Tensor, seq: int):
        c = self._prefill_position_cache.get(seq) if self._host_cache else None
        if c is not None and torch.equal(c[0], position_ids):
            return c[1]
        pos = position_ids.to(dtype=torch.int32, device="rpu").contiguous()
        if self._host_cache:
            self._prefill_position_cache[seq] = (position_ids.clone(), pos)
        return pos

    def _prefill_cos_sin(self, position_ids: torch.Tensor, seq: int):
        if not self._partial_mrope:
            return None, None
        c = self._prefill_rope_cache.get(seq) if self._host_cache else None
        if c is not None and torch.equal(c[0], position_ids):
            return c[1], c[2]
        cos_cpu, sin_cpu = build_interleaved_mrope_cos_sin(
            position_ids.to(torch.int64), head_dim=self._head_dim,
            rope_theta=self._rope_theta, mrope_section=self._mrope_section)
        cos_il, sin_il = cos_cpu.to("rpu").contiguous(), sin_cpu.to("rpu").contiguous()
        if self._host_cache:
            self._prefill_rope_cache[seq] = (position_ids.clone(), cos_il, sin_il)
        return cos_il, sin_il

    @torch.no_grad()
    def forward_embeds(self, inputs_embeds: torch.Tensor,
                       position_ids: torch.Tensor) -> torch.Tensor:
        """Prefill from pre-assembled inputs_embeds [1, seq, H] + 3D M-RoPE position_ids
        [seq, 3] → final-normed last_hidden [1, seq, H]."""
        assert inputs_embeds.dim() == 3 and inputs_embeds.size(0) == 1, (
            f"forward_embeds: inputs_embeds must be [1, seq, H], got {tuple(inputs_embeds.shape)}")
        logical_seq = int(inputs_embeds.size(1))
        assert position_ids.shape == (logical_seq, 3), (
            "position_ids must be [seq, 3], got "
            f"{tuple(position_ids.shape)}")
        customer_prefill = logical_seq == _CUSTOMER_LOGICAL_PREFILL
        execution_seq = (
            _CUSTOMER_EXECUTION_PREFILL if customer_prefill else logical_seq
        )
        requested_chunk = _CERTIFIED_CHUNK_SIZE if customer_prefill else 0
        torch.ops.rpu.causal_decoder_set_chunk_size_override(
            self._handle, requested_chunk)
        planned_chunk = int(
            torch.ops.rpu.causal_decoder_resolve_prefill_chunk_size(
                self._handle, execution_seq, 0))
        if customer_prefill and planned_chunk != _CERTIFIED_CHUNK_SIZE:
            raise RuntimeError(
                "InternVLA System-2 validated prefill plan drift: "
                f"expected E{_CUSTOMER_EXECUTION_PREFILL}/"
                f"C{_CERTIFIED_CHUNK_SIZE}, got E{execution_seq}/"
                f"C{planned_chunk}")

        if execution_seq != logical_seq:
            pad_rows = execution_seq - logical_seq
            inputs_embeds = torch.nn.functional.pad(
                inputs_embeds, (0, 0, 0, pad_rows))
            position_ids = torch.cat(
                [position_ids, position_ids.new_zeros((pad_rows, 3))], dim=0)

        ie = self._prefill_inputs_embeds(inputs_embeds, execution_seq)
        pos = self._prefill_position_ids(position_ids, execution_seq)
        assert pos.shape == (execution_seq, 3), (
            "position_ids must match the physical execution length, got "
            f"{tuple(pos.shape)}")
        cos_il, sin_il = self._prefill_cos_sin(position_ids, execution_seq)
        self.cache.reset_to_position(0)
        sig = rpu_backend.graph.GraphSignature(
            op_id="internvla_s2_backbone",
            shapes=[logical_seq, execution_seq, self.hidden_size],
            dyn_dims=[self.num_layers, planned_chunk],
            dtypes=[torch.float16])
        with self._graph_cache.capture(sig):
            raw = torch.ops.rpu.causal_decoder_forward(
                self._handle, ie, self.cache.k_caches, self.cache.v_caches,
                None, self.cache.position, True, pos, [], cos_il, sin_il)
        resolved_chunk = int(
            torch.ops.rpu.causal_decoder_get_resolved_chunk_size(self._handle))
        if resolved_chunk != planned_chunk:
            raise RuntimeError(
                "InternVLA System-2 prefill dry/forward chunk mismatch: "
                f"planned={planned_chunk}, resolved={resolved_chunk}, "
                f"logical_len={logical_seq}, execution_len={execution_seq}")
        vars(self)["_rpu_last_execution_plan"] = {
            "logical_len": logical_seq,
            "execution_len": execution_seq,
            "chunk_size": resolved_chunk,
            "padding_rows": execution_seq - logical_seq,
        }
        self.cache.update_position(logical_seq)
        return raw[:, :logical_seq]


def build_qwen25vl_backbone(
    ckpt_dir: str,
    *,
    layer_indices: Sequence[int] | None = None,
    max_seq_len: int = _MAX_SEQ_LEN,
    attn_tp: int = 4,
    w8a16: bool = True,
    asset_manifest: Mapping[str | Path, str] | None = None,
) -> Qwen25VLBackbone:
    """Load the stock Qwen2.5-VL-7B text decoder and install it on RPU.

    Args:
        ckpt_dir: dir with config.json + model.safetensors(.index.json) (stock keys).
        layer_indices: must be ``None``; partial profiles fail closed.
        attn_tp: attention tensor-parallel factor (= attention cores). 4 for NKV=4.
        w8a16: load int8 weight + fp16 per-out-channel scale (fits the board's disk;
            needs a pre-quantized checkpoint with `.weight_scale` tensors).
    """
    require_controlled_evaluation(asset_manifest)
    partial_mrope = _partial_mrope_enabled()
    host_cache = rpu_env_bool("RPU_INTERNVLA_HOST_CACHE", default=True)
    root = Path(ckpt_dir).expanduser().resolve()
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    single_path = root / "model.safetensors"
    verify_asset_manifest((config_path,), asset_manifest)
    with config_path.open(encoding="utf-8") as stream:
        cfg = json.load(stream)
    actual_head_dim = cfg.get("head_dim")
    if actual_head_dim is None:
        hidden = cfg.get("hidden_size")
        heads = cfg.get("num_attention_heads")
        if isinstance(hidden, int) and isinstance(heads, int) and heads > 0:
            actual_head_dim = hidden // heads
    actual_profile = dict(cfg)
    actual_profile["head_dim"] = actual_head_dim
    errors = [
        f"{key}={actual_profile.get(key)!r} (expected {expected!r})"
        for key, expected in _PROFILE.items()
        if actual_profile.get(key) != expected
    ]
    actual_mrope = cfg.get("rope_scaling", {}).get("mrope_section")
    if actual_mrope != _MROPE_SECTION:
        errors.append(f"mrope_section={actual_mrope!r} (expected {_MROPE_SECTION!r})")
    if layer_indices is not None:
        errors.append("layer_indices must be None (partial checkpoints are not accepted)")
    if max_seq_len != _MAX_SEQ_LEN:
        errors.append(
            f"max_seq_len={max_seq_len!r} (expected {_MAX_SEQ_LEN})"
        )
    if attn_tp != 4:
        errors.append(f"attn_tp={attn_tp!r} (expected 4)")
    if w8a16 is not True:
        errors.append("w8a16 must be True")
    if errors:
        raise ValueError(
            "InternVLA-N1 System-2 does not match the controlled 7B W8A16 "
            f"profile: {'; '.join(errors)}"
        )
    H = cfg["hidden_size"]
    NQ = cfg["num_attention_heads"]
    NKV = cfg["num_key_value_heads"]
    HD = cfg.get("head_dim") or (H // NQ)
    INTER = cfg["intermediate_size"]
    EPS = cfg["rms_norm_eps"]
    THETA = cfg["rope_theta"]
    MROPE = list(cfg["rope_scaling"]["mrope_section"])
    assert NQ % attn_tp == 0 and NKV % (attn_tp // NKV if attn_tp > NKV else 1) == 0
    assert NQ % NKV == 0, f"GQA needs NQ({NQ}) % NKV({NKV}) == 0"

    layer_indices = list(range(cfg["num_hidden_layers"]))
    num_layers = len(layer_indices)

    checkpoint_descriptor = index_path if index_path.is_file() else single_path
    verify_asset_manifest((checkpoint_descriptor,), asset_manifest)
    st = _SafeTensorStore(str(root), mmap=False)  # stream w/o mmap residency (10 GB peak budget)
    shard_paths = tuple((root / shard).resolve() for shard in sorted(set(st.weight_map.values())))
    if any(root not in path.parents for path in shard_paths):
        raise ValueError("InternVLA-N1 checkpoint index contains a shard outside ckpt_dir")
    verify_asset_manifest((config_path, *shard_paths), asset_manifest)
    _validate_checkpoint_profile(st)
    def gf(key): return st.get_tensor(key)

    q_w, k_w, v_w, o_w = [], [], [], []
    q_b, k_b, v_b = [], [], []
    gate, up, down = [], [], []
    q_s, k_s, v_s, o_s, gate_s, up_s, down_s = [], [], [], [], [], [], []
    in_norm, post_norm = [], []

    def col(w):
        return (_to_rpu_int8(tp_col_swizzle_mc_weight(w, attn_tp, dwidth=1)) if w8a16
                else _to_rpu_half(tp_col_swizzle_mc_weight(w.half(), attn_tp)))
    def row(w, tp):
        return (_to_rpu_int8(tp_row_swizzle_mc_weight(w, tp, dwidth=1)) if w8a16
                else _to_rpu_half(tp_row_swizzle_mc_weight(w.half(), tp)))
    def col8(w):
        return (_to_rpu_int8(tp_col_swizzle_mc_weight(w, 8, dwidth=1)) if w8a16
                else _to_rpu_half(tp_col_swizzle_mc_weight(w.half(), 8)))

    for L in layer_indices:
        p = f"model.layers.{L}"
        q_w.append(col(gf(f"{p}.self_attn.q_proj.weight")))
        k_w.append(col(gf(f"{p}.self_attn.k_proj.weight")))
        v_w.append(col(gf(f"{p}.self_attn.v_proj.weight")))
        o_w.append(row(gf(f"{p}.self_attn.o_proj.weight"), attn_tp))
        q_b.append(_to_rpu_half(gf(f"{p}.self_attn.q_proj.bias")))
        k_b.append(_to_rpu_half(gf(f"{p}.self_attn.k_proj.bias")))
        v_b.append(_to_rpu_half(gf(f"{p}.self_attn.v_proj.bias")))
        gate.append(col8(gf(f"{p}.mlp.gate_proj.weight")))
        up.append(col8(gf(f"{p}.mlp.up_proj.weight")))
        down.append(row(gf(f"{p}.mlp.down_proj.weight"), 8))
        in_norm.append(_to_rpu_half(gf(f"{p}.input_layernorm.weight")))
        post_norm.append(_to_rpu_half(gf(f"{p}.post_attention_layernorm.weight")))
        if w8a16:
            q_s.append(_to_rpu_half(gf(f"{p}.self_attn.q_proj.weight_scale")))
            k_s.append(_to_rpu_half(gf(f"{p}.self_attn.k_proj.weight_scale")))
            v_s.append(_to_rpu_half(gf(f"{p}.self_attn.v_proj.weight_scale")))
            o_s.append(_to_rpu_half(gf(f"{p}.self_attn.o_proj.weight_scale")))
            gate_s.append(_to_rpu_half(gf(f"{p}.mlp.gate_proj.weight_scale")))
            up_s.append(_to_rpu_half(gf(f"{p}.mlp.up_proj.weight_scale")))
            down_s.append(_to_rpu_half(gf(f"{p}.mlp.down_proj.weight_scale")))

    final_norm_w = _to_rpu_half(gf("model.norm.weight"))
    cos, sin = _build_rope_tables(HD, THETA, max_seq_len)

    set_weights = (torch.ops.rpu.causal_decoder_set_weights_w8a16 if w8a16
                   else torch.ops.rpu.causal_decoder_set_weights)
    args = [q_w, k_w, v_w, o_w, [], [], in_norm, post_norm,
            gate, up, down, cos, sin, final_norm_w,
            NQ, NKV, HD, H, INTER, EPS, True, MROPE, []]
    if w8a16:
        args.extend([q_s, k_s, v_s, o_s, gate_s, up_s, down_s, q_b, k_b, v_b])
    else:
        args.extend([q_b, k_b, v_b])
    handle = torch.ops.rpu.causal_decoder_create()
    configured = False
    try:
        set_weights(handle, *args)
        torch.ops.rpu.causal_decoder_set_chunk_envelope(
            handle, _CERTIFIED_MAX_KV_LEN, _CERTIFIED_CHUNK_SIZE)
        torch.ops.rpu.causal_decoder_set_chunk_size_override(handle, 0)
        cache = RPUCache(
            num_layers=num_layers, batch_size=1, max_seq_len=max_seq_len,
            num_kv_heads=NKV, head_dim=HD, attn_tp=attn_tp)
        result = Qwen25VLBackbone(
            handle=handle, cache=cache,
            graph_cache=rpu_backend.graph.GraphCache(), hidden_size=H,
            num_layers=num_layers, head_dim=HD, rope_theta=THETA,
            mrope_section=MROPE, partial_mrope=partial_mrope,
            host_cache=host_cache)
        configured = True
        return result
    finally:
        if not configured:
            _destroy_causal_decoder_handle(handle)
