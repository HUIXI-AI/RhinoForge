"""Finite Pi0.5 core profiles; weights are prepared once before installation."""
from numbers import Integral


def core_topology(execution=None):
    from rpu_backend.runtime.topology import DecoderTopology, execution_core_count

    cores = execution_core_count(execution)
    attention = 8 if cores == 8 else 4
    return DecoderTopology(cores, attention, cores, attention)


def model_topology(model):
    return getattr(model, "_rpu_decoder_topology", None) or core_topology()


def physical_intermediate_size(logical, cores, dtype):
    import torch

    if cores != 6:
        return logical
    if logical not in (16384, 4096) or dtype not in (torch.float16, torch.int8):
        raise ValueError("Pi0.5 MLP6 requires exact prefix/expert FP16 or W8A16 weights")
    alignment = 6 * (32 if dtype == torch.int8 else 16)
    return (logical + alignment - 1) // alignment * alignment


def vision_batch_enabled(execution=None):
    from rpu_backend.runtime import rpu_env_bool

    # Reduced profiles always use serial256. The historical batching override
    # only selects an eight-core profile; it cannot change a reduced topology.
    return (core_topology(execution).num_cores == 8
            and rpu_env_bool("RPU_PI05_SIGLIP_BATCH", default=True))


def validate_reduced_policy(policy, topology):
    """Validate the entire raw model before the first irreversible conversion."""
    if topology.num_cores == 8:
        return
    import torch

    pawe = policy.model.paligemma_with_expert
    decoders = (pawe.paligemma.model.language_model, pawe.gemma_expert.model)
    quant = getattr(policy, "_pi05_quant_config", {}) or {}
    if quant and quant.get("method") not in (None, "w8a16"):
        raise ValueError("Pi0.5 reduced cores support FP16 and W8A16 only")
    dtype = decoders[0].layers[0].self_attn.q_proj.weight.dtype
    if dtype not in (torch.float16, torch.int8):
        raise ValueError("Pi0.5 reduced cores require original FP16 or W8A16 weights")
    for model, hidden, intermediate in zip(decoders, (2048, 1024), (16384, 4096)):
        fields = ("num_hidden_layers", "hidden_size", "intermediate_size",
                  "num_attention_heads", "num_key_value_heads", "head_dim")
        geometry = tuple(getattr(model.config, name, None) for name in fields)
        if (any(isinstance(x, bool) or not isinstance(x, Integral) for x in geometry)
                or geometry != (18, hidden, intermediate, 8, 1, 256)
                or len(model.layers) != 18):
            raise ValueError("Pi0.5 reduced cores require exact Gemma2B/Gemma300M geometry")
        for layer in model.layers:
            for owner, names in ((layer.self_attn, (
                    ("q_proj", (2048, hidden)), ("k_proj", (256, hidden)),
                    ("v_proj", (256, hidden)), ("o_proj", (hidden, 2048)))),
                    (layer.mlp, (("gate_proj", (intermediate, hidden)),
                                 ("up_proj", (intermediate, hidden)),
                                 ("down_proj", (hidden, intermediate))))):
                for name, shape in names:
                    projection = getattr(owner, name)
                    weight = projection.weight
                    if (not isinstance(projection, torch.nn.Linear)
                            or weight.device.type != "cpu" or weight.dtype != dtype
                            or tuple(weight.shape) != shape or not weight.is_contiguous()
                            or (projection.out_features, projection.in_features) != shape
                            or projection.bias is not None
                            or hasattr(projection, "_rpu_linear_partition")):
                        raise ValueError(f"Pi0.5 reduced {name} requires original cold logical weights")
        from .w8a16 import detect_and_validate_gemma_decoder_w8a16
        detect_and_validate_gemma_decoder_w8a16(model, label="reduced Pi0.5")
    tower = pawe.paligemma.model.vision_tower
    vision = getattr(tower, "vision_model", tower)
    config = vision.config
    if (tuple(getattr(config, name, None) for name in (
            "num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads"))
            != (27, 1152, 4304, 16) or len(vision.encoder.layers) != 27):
        raise ValueError("Pi0.5 reduced cores require the exact SigLIP27 tower")
    if (policy.model.config.max_action_dim != 32
            or policy.model.config.chunk_size != 50
            or policy.model.action_in_proj.out_features != 1024):
        raise ValueError("Pi0.5 reduced cores require the full 50x32 action profile")


def pad_decoder_mlp_inplace(model, topology):
    """Zero-pad cold MLP weights and W8 output scales, retaining logical config."""
    import torch

    logical = model.config.intermediate_size
    dtype = model.layers[0].mlp.gate_proj.weight.dtype
    physical = physical_intermediate_size(logical, topology.mlp_tp, dtype)
    if physical == logical:
        return
    # Preflight all leaves before mutating any of them.
    for layer in model.layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            leaf = getattr(layer.mlp, name)
            axis = 1 if name == "down_proj" else 0
            if (leaf.weight.device.type != "cpu" or leaf.weight.dtype != dtype
                    or leaf.weight.shape[axis] != logical
                    or hasattr(leaf, "_rpu_linear_partition")):
                raise ValueError("Pi0.5 MLP padding requires unswizzled cold logical weights")
            if dtype == torch.int8 and name != "down_proj":
                scale = getattr(leaf, "weight_scale", None)
                if scale is None or tuple(scale.shape) != (logical,):
                    raise ValueError("Pi0.5 padded W8 MLP requires logical output-channel scales")
    for layer in model.layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            leaf = getattr(layer.mlp, name)
            axis = 1 if name == "down_proj" else 0
            shape = list(leaf.weight.shape)
            shape[axis] = physical
            padded = leaf.weight.new_zeros(shape)
            padded.narrow(axis, 0, logical).copy_(leaf.weight.detach())
            leaf.weight = torch.nn.Parameter(padded, requires_grad=False)
            if axis == 0:
                leaf.out_features = physical
                if dtype == torch.int8:
                    scale = leaf.weight_scale.new_ones(physical)
                    scale[:logical].copy_(leaf.weight_scale)
                    leaf._buffers["weight_scale"] = scale
            else:
                leaf.in_features = physical


def graph_cache_options(cores):
    if cores == 8:
        return {}
    from rpu_backend.graph import GraphRuntimePolicy

    return {"runtime_policy": GraphRuntimePolicy.from_environment(execution_core_count=cores)}


def validate_native_topology(kind, handle, topology, logical, dtype):
    import torch

    if topology.num_cores == 8:
        return
    condition = topology.attn_tp if kind in ("adarms", "pi05_denoise_step") else 1
    expected = (1, topology.num_cores, topology.attn_tp, topology.mlp_tp, condition,
                8, logical, physical_intermediate_size(logical, topology.mlp_tp, dtype))
    actual = tuple(getattr(torch.ops.rpu, kind + "_get_execution_topology")(handle))
    if actual != expected:
        raise RuntimeError(f"Pi0.5 {kind} native topology mismatch: {actual} != {expected}")
