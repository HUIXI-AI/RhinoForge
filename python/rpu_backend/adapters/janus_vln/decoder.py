"""Janus FP16 weights for the existing biased Qwen2.5 fused decoder.

Controlled component builder; not registered as end-to-end model support.
"""
import torch

import rpu_backend
from rpu_backend.adapters.wall_oss.llm import (
    WallOssLLM, _build_rope_tables,
    _configure_causal_decoder_handle,
)
from rpu_backend.api.cache import RPUCache
from rpu_backend.runtime.weights import tp_col_swizzle_mc_weight, tp_row_swizzle_mc_weight
from .memory import checkpoint_bytes, memory_plan
from .checkpoint import CheckpointStore


def build_decoder(checkpoint, *, max_seq_len=4096):
    from rpu_backend.api._execution import _require_execution_process_safe
    _require_execution_process_safe()
    weights, largest = checkpoint_bytes(checkpoint)
    memory_plan(weights, largest, text_capacity=max_seq_len)
    store = CheckpointStore(checkpoint)
    projections = {
        "self_attn.q_proj": (3584, 3584), "self_attn.k_proj": (512, 3584),
        "self_attn.v_proj": (512, 3584), "self_attn.o_proj": (3584, 3584),
        "mlp.gate_proj": (18944, 3584), "mlp.up_proj": (18944, 3584),
        "mlp.down_proj": (3584, 18944),
    }
    expected = {"model.norm.weight": (3584,)}
    for layer in range(28):
        prefix = f"model.layers.{layer}."
        expected.update({prefix + name + ".weight": shape for name, shape in projections.items()})
        expected.update({prefix + name: shape for name, shape in {
            "self_attn.q_proj.bias": (3584,), "self_attn.k_proj.bias": (512,),
            "self_attn.v_proj.bias": (512,), "input_layernorm.weight": (3584,),
            "post_attention_layernorm.weight": (3584,),
        }.items()})
    for name, shape in expected.items():
        shard = store.weight_map.get(name)
        if shard is None or tuple(store._header(shard)[0][name]["shape"]) != shape:
            raise ValueError(f"Janus decoder tensor mismatch: {name}")

    def read(name):
        return store.get_tensor(name).half()

    packed = {name: [] for name in projections}
    q_bias, k_bias, v_bias, norm, post_norm = [], [], [], [], []
    for layer in range(28):
        prefix = f"model.layers.{layer}."
        print(f"Janus decoder loading layer {layer + 1}/28", flush=True)
        for name in projections:
            tp = 4 if name.startswith("self_attn") else 8
            swizzle = tp_row_swizzle_mc_weight if name.endswith(("o_proj", "down_proj")) else tp_col_swizzle_mc_weight
            packed[name].append(swizzle(read(prefix + name + ".weight"), tp).to("rpu"))
        for target, name in ((q_bias, "self_attn.q_proj.bias"),
                             (k_bias, "self_attn.k_proj.bias"),
                             (v_bias, "self_attn.v_proj.bias"),
                             (norm, "input_layernorm.weight"),
                             (post_norm, "post_attention_layernorm.weight")):
            target.append(read(prefix + name).to("rpu"))
    cos, sin = _build_rope_tables(128, 1e6, max_seq_len)
    args = [*(packed[name] for name in list(projections)[:4]), [], [], norm, post_norm,
            *(packed[name] for name in list(projections)[4:]), cos, sin,
            read("model.norm.weight").to("rpu"), 28, 4, 128, 3584, 18944, 1e-6,
            True, [16, 24, 24], [], q_bias, k_bias, v_bias]
    cache = RPUCache(num_layers=28, batch_size=1, max_seq_len=max_seq_len,
                     num_kv_heads=4, head_dim=128, attn_tp=4)
    graphs = rpu_backend.graph.GraphCache(max_entries=1)
    handle = _configure_causal_decoder_handle(
        set_weights=torch.ops.rpu.causal_decoder_set_weights, weight_args=args,
        max_seq_len=max_seq_len, chunk_size_cap=64, equal_two_prefill=False)
    model = None
    try:
        model = WallOssLLM.__new__(WallOssLLM)
        model._handle, model._graph_cache, model._closed = handle, graphs, False
        model._gc_retirement_enabled = False
        WallOssLLM.__init__(model, handle=handle, cache=cache, graph_cache=graphs,
            embed_w=None, hidden_size=3584, num_layers=28, head_dim=128,
            rope_theta=1e6, mrope_section=[16, 24, 24], prefill_chunk_size_cap=64,
            prefill_pad16=True, prefill_equal_two=False,
            attn_geometry=dict(num_q_heads=28, num_kv_heads=4, num_cores=4,
                               head_dim=128, attn_mask_type=1),
            partial_mrope=True, execution_config={"prefill": {"padding_budget": 64}})
        return model
    except BaseException as error:
        from rpu_backend.adapters.wall_oss._retirement import _cleanup_build_failure
        if model is not None:
            model._gc_retirement_enabled = False
        _cleanup_build_failure(error, (handle, args, cache, graphs),
                               lambda: torch.ops.rpu.causal_decoder_destroy(handle))
        raise
