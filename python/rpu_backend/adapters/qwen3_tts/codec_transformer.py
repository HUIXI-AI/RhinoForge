"""Bounded FP16 codec frontend; the remaining codec retains CPU FP32."""
import types

import torch


def _validate_core(core):
    config = core.config
    geometry = tuple(getattr(config, key) for key in (
        "num_hidden_layers", "hidden_size", "intermediate_size",
        "num_attention_heads", "num_key_value_heads", "head_dim"))
    if (geometry != (8, 512, 1024, 16, 16, 64) or len(core.layers) != 8
            or config.sliding_window != 72 or config.hidden_act != "silu"
            or config.attention_bias or core.norm.variance_epsilon != 1e-5
            or core.input_proj.weight.shape != (512, 1024)
            or core.output_proj.weight.shape != (1024, 512)):
        raise ValueError("unsupported TTS codec frontend geometry")
    for layer in core.layers:
        if (not isinstance(layer.self_attn.q_norm, torch.nn.Identity)
                or not isinstance(layer.self_attn.k_norm, torch.nn.Identity)
                or layer.attention_type != "sliding_attention"
                or layer.input_layernorm.variance_epsilon != 1e-5
                or layer.post_attention_layernorm.variance_epsilon != 1e-5
                or layer.self_attn.o_proj.bias is not None
                or layer.mlp.down_proj.bias is not None
                or layer.self_attn_layer_scale.scale.shape != (512,)
                or layer.mlp_layer_scale.scale.shape != (512,)):
            raise ValueError("unsupported TTS codec frontend layer semantics")
    if any(value.dtype != torch.float32 or value.device.type != "cpu"
           for value in core.parameters()):
        raise ValueError("TTS codec frontend installation requires clean CPU FP32 weights")


def _folded_weight(module, scale=None):
    weight = module.weight.detach()
    if scale is not None:
        weight = weight * scale.detach()[:, None]
    return weight.half()


def install_codec_transformer(owner, core):
    """Attach one parent-owned decoder; never mutate the FP32 source weights."""
    from rpu_backend.api.cache import RPUCache
    from rpu_backend.graph import GraphCache
    from rpu_backend.runtime._native_retirement import _InstalledNativeResource
    from rpu_backend.runtime.weights import transform_linear_weight
    from rpu_backend.runtime.topology import DecoderTopology

    _validate_core(core)
    set_codec_mode = getattr(torch.ops.rpu, "causal_decoder_set_tts_codec_mode", None)
    if set_codec_mode is None:
        return False
    cores = 8 if getattr(owner, "_num_cores", 8) == 8 else 4
    topology = DecoderTopology(cores, cores, cores, cores)
    graph = GraphCache(runtime_policy=getattr(owner, "_graph_runtime_policy", None))
    handle = torch.ops.rpu.causal_decoder_create(num_cores=cores, vocab_size=0)
    part = dict(handle=handle, graph=graph, calls=0, tokens=0, planner=core,
                cpu_forward=core.forward, topology=topology, io_cores=cores)
    owner._components["codec_transformer"] = part
    owner._native_codec_transformer_handle = handle
    part["resource"] = _InstalledNativeResource(
        owner, handle, torch.ops.rpu.causal_decoder_destroy, graphs=(graph,),
        keepalive=part, label="TTS codec frontend", handle_name="_native_codec_transformer_handle")
    part["resource"].take_ownership(owner)
    groups = []
    for path, partition, scale_path in (
            ("self_attn.q_proj", 1, None), ("self_attn.k_proj", 1, None),
            ("self_attn.v_proj", 1, None), ("self_attn.o_proj", 0, "self_attn_layer_scale")):
        groups.append([transform_linear_weight(_folded_weight(layer.get_submodule(path),
            layer.get_submodule(scale_path).scale if scale_path else None), partition, cores).to("rpu")
            for layer in core.layers])
    groups.extend([[], []])
    for path in ("input_layernorm", "post_attention_layernorm"):
        groups.append([layer.get_submodule(path).weight.detach().half().to("rpu") for layer in core.layers])
    for path, partition, scale_path in (
            ("mlp.gate_proj", 1, None), ("mlp.up_proj", 1, None),
            ("mlp.down_proj", 0, "mlp_layer_scale")):
        groups.append([transform_linear_weight(_folded_weight(layer.get_submodule(path),
            layer.get_submodule(scale_path).scale if scale_path else None), partition, cores).to("rpu")
            for layer in core.layers])
    cos, sin = core.rotary_emb(torch.zeros(1, 1, 512, dtype=torch.float16), torch.arange(64).view(1, -1))
    cos, sin = (value[0, :, :32].contiguous().to("rpu") for value in (cos, sin))
    norm = core.norm.weight.detach().half().to("rpu")
    part["storage"] = (groups, cos, sin, norm)
    torch.ops.rpu.causal_decoder_set_weights(handle, *groups, cos, sin, norm,
                                            16, 16, 64, 512, 1024, 1e-5, True)
    set_codec_mode(handle)
    actual = tuple(torch.ops.rpu.causal_decoder_topology(handle))
    if actual != topology.identity()[1:]:
        raise RuntimeError(f"TTS codec native topology differs from its weight layout: {actual}")
    part["native_topology"] = actual
    torch.ops.rpu.causal_decoder_set_linear_acc32(handle, True)
    # This codec's own bounded profile; it is not a Qwen3/Llama registry alias.
    torch.ops.rpu.causal_decoder_set_chunk_envelope(handle, 64, 64)
    core._rpu_decoder_handle = handle
    core._rpu_decoder_graph_cache = graph
    core._rpu_decoder_topology = topology
    part["cache"] = RPUCache(8, 1, 64, 16, 64, attn_tp=cores)
    part["io_weights"] = [transform_linear_weight(module.weight.detach().half(), 1, cores).to("rpu")
                          for module in (core.input_proj, core.output_proj)]
    part["io_bias"] = [module.bias.detach().half().to("rpu") for module in (core.input_proj, core.output_proj)]
    owner._restore.append((core, core.forward))

    def forward(_self, inputs_embeds=None, **kwargs):
        if any(value is not None and value is not False for value in kwargs.values()):
            raise ValueError("TTS codec frontend requires an uncached unmasked full sequence")
        return _forward(part, inputs_embeds)

    core.forward = types.MethodType(forward, core)
    return True


def _forward(part, hidden):
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from rpu_backend.graph import GraphSignature
    from rpu_backend.runtime.decoder import _text_prefill_execution_plan

    if (hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != 1024
            or not 1 <= hidden.shape[1] <= 255 or hidden.dtype != torch.float32
            or hidden.device.type != "cpu" or not torch.isfinite(hidden).all()):
        raise ValueError("TTS codec frontend requires finite CPU FP32 [1, 1–255, 1024]")
    count = hidden.shape[1]
    counts = part.setdefault("request_input_lengths", {})
    counts[str(count)] = counts.get(str(count), 0) + 1
    if count > 63:
        # The original FP32 path preserves the codec's 72-frame sliding attention.
        part["last_device"] = "cpu"
        part["last_execution_plan"] = {"device": "cpu", "logical_len": count,
            "torch_num_threads": torch.get_num_threads(),
            "inference_mode": torch.is_inference_mode_enabled(),
            "reason": "sequence exceeds 63 device codec frames"}
        return part["cpu_forward"](inputs_embeds=hidden)
    part["last_device"] = "rpu"
    cache, graph, handle = part["cache"], part["graph"], part["handle"]
    cache.reset_to_position(0)
    if count == 1:
        execution_len = 1
        selected = None
        descriptor = list(torch.ops.rpu.causal_decoder_resolve_decode_stage_descriptor(handle))
        key = descriptor
    else:
        execution_len, _, plan = _text_prefill_execution_plan(part["planner"], handle, cache, count)
        descriptor = plan.selected.stage_tuple.physical_descriptor
        selected = plan.selected
        key = plan.graph_key_words()
    signature = GraphSignature(op_id="qwen3_tts_codec_frontend", shapes=[1, count, execution_len, 512],
                               dyn_dims=[handle, *key], dtypes=[torch.float16])
    if part.get("signature") != signature:
        if "signature" in part:
            graph.evict(part["signature"])
    values = hidden.half()
    if execution_len > count:
        values = torch.nn.functional.pad(values, (0, 0, 0, execution_len - count))
    projected = torch.ops.rpu.linear_with_accumulation(values.contiguous().to("rpu").view(execution_len, 1024),
        part["io_weights"][0], part["io_bias"][0], 1, part["io_cores"], True).view(1, execution_len, 512)
    with graph.capture(signature):
        raw = torch.ops.rpu.causal_decoder_forward(handle, projected, cache.k_caches, cache.v_caches,
                                                  None, 0, True, planned_stage_descriptor=descriptor)
    part["last_execution_plan"] = {"device": "rpu", "logical_len": count,
        "execution_len": execution_len, "padding_rows": execution_len - count,
        "physical_descriptor": descriptor, "selected": selected,
        "torch_num_threads": torch.get_num_threads(), "linear_acc32": True,
        "inference_mode": torch.is_inference_mode_enabled(),
        "io_cores": part["io_cores"]}
    part["signature"] = signature
    result = torch.ops.rpu.linear_with_accumulation(raw[:, :count].reshape(count, 512),
        part["io_weights"][1], part["io_bias"][1], 1, part["io_cores"], True).cpu().float().reshape(1, count, 1024)
    if not torch.isfinite(result).all():
        raise FloatingPointError("non-finite TTS codec frontend output")
    part["calls"] += 1
    part["tokens"] += count
    return BaseModelOutputWithPast(last_hidden_state=result)
