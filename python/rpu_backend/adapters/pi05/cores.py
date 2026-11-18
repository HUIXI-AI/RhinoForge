"""Finite Pi0.5 core profiles; weights are prepared once before installation."""
from numbers import Integral

# The action horizon is the checkpoint's `chunk_size`; the finite admitted set is
# the same one `pi05_action_rows_admitted` holds in
# src/fused/rpu_pi05_denoise_step_model.cpp and `_ACTION_HORIZONS` in
# examples/runners/pi05/optimized.py.
ACTION_HORIZONS = (10, 50)


def core_topology(execution=None):
    from rpu_backend.runtime.topology import DecoderTopology, execution_core_count

    cores = execution_core_count(execution)
    attention = 8 if cores == 8 else 4
    return DecoderTopology(cores, attention, cores, attention)


def model_topology(model):
    return getattr(model, "_rpu_decoder_topology", None) or core_topology()


# Per-lane K granule of the row-partitioned Down: 16 FP16, 32 INT8 or 64 E2M1.
_K_GRANULE = {"float16": 16, "int8": 32, "uint8": 64}
# Action NVFP4: these expert projections hold E2M1 codes; K/V and the VLM stay W8.
NVFP4_ACTION_PROJECTIONS = frozenset(("q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))


def physical_intermediate_size(logical, cores, dtype):
    """Physical MLP width; uint8 means Action NVFP4 E2M1 codes."""
    import torch

    if cores != 6:
        return logical
    if (logical not in (16384, 4096) or dtype not in (torch.float16, torch.int8, torch.uint8)
            or (dtype == torch.uint8 and logical != 4096)):
        raise ValueError("Pi0.5 MLP6 requires exact prefix/expert FP16 or W8A16, "
                         "or Action NVFP4 weights")
    alignment = 6 * _K_GRANULE[str(dtype).removeprefix("torch.")]
    return (logical + alignment - 1) // alignment * alignment


def vision_batch_enabled(execution=None):
    from rpu_backend.runtime import rpu_env_bool

    cores = core_topology(execution).num_cores
    if not rpu_env_bool("RPU_PI05_SIGLIP_BATCH", default=True):
        return False
    if cores == 8:
        return True
    # Four-lane SigLIP (under 4- or 6-core Pi0.5 budgets) admits packed N=2/3
    # cameras when the raw-SPM KV + Q-resident profile is cold-enabled.
    return (
        cores in (4, 6)
        and not rpu_env_bool("RPU_PI05_SIGLIP_W4A16", default=False)
        and rpu_env_bool("RPU_PI05_VISION_KV_SPM_ONLY", default=False)
        and rpu_env_bool("RPU_PI05_VISION_Q_RESIDENT", default=False)
    )


def validate_reduced_policy(policy, topology):
    """Validate the entire raw model before the first irreversible conversion."""
    if topology.num_cores == 8:
        return
    import torch

    pawe = policy.model.paligemma_with_expert
    decoders = (pawe.paligemma.model.language_model, pawe.gemma_expert.model)
    quant = getattr(policy, "_pi05_quant_config", {}) or {}
    nvfp4 = bool(quant) and quant.get("method") == "nvfp4a16"
    if quant and quant.get("method") not in (None, "w8a16", "nvfp4a16"):
        raise ValueError("Pi0.5 reduced cores support FP16, W8A16 and Action NVFP4 only")
    if nvfp4:
        from .w4pack import int4_child_names
        if (quant.get("nvfp4_abi") != "striped_v2" or quant.get("nvfp4_block_size") != 16
                or int4_child_names(quant, component="vlm")
                or int4_child_names(quant, component="expert") != NVFP4_ACTION_PROJECTIONS):
            raise ValueError("Pi0.5 reduced NVFP4 requires Action-only striped_v2 block16 "
                             "Q/O/Gate/Up/Down with W8 K/V and VLM")
    dtype = decoders[0].layers[0].self_attn.q_proj.weight.dtype
    if dtype not in (torch.float16, torch.int8) or (nvfp4 and dtype != torch.int8):
        raise ValueError("Pi0.5 reduced cores require original FP16 or W8A16 weights")
    for expert, model, hidden, intermediate in zip(
            (False, True), decoders, (2048, 1024), (16384, 4096)):
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
                    fp4 = nvfp4 and expert and name in NVFP4_ACTION_PROJECTIONS
                    if fp4 and (getattr(projection, "_pi05_nvfp4_abi", None) != "striped_v2"
                                or getattr(projection, "_pi05_nvfp4_packed", False)):
                        raise ValueError(f"Pi0.5 reduced {name} requires unpacked striped_v2 codes")
                    if (not isinstance(projection, torch.nn.Linear)
                            or weight.device.type != "cpu"
                            or weight.dtype != (torch.uint8 if fp4 else dtype)
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
            or policy.model.config.chunk_size not in ACTION_HORIZONS
            or policy.model.action_in_proj.out_features != 1024):
        raise ValueError(
            "Pi0.5 reduced cores require action horizon "
            f"{' or '.join(str(h) for h in sorted(ACTION_HORIZONS))} and width 32")


def pad_decoder_mlp_inplace(model, topology):
    """Zero-pad cold MLP weights and their scales, retaining logical config.

    W8 output-channel scales pad with ones. NVFP4 FP8 block scales [K/16,N]
    pad with zero, which is what the quantizer emits for an all-zero block;
    the FP32 tensor scale is per projection and does not change.
    """
    import torch

    logical = model.config.intermediate_size
    dtype = model.layers[0].mlp.gate_proj.weight.dtype
    physical = physical_intermediate_size(logical, topology.mlp_tp, dtype)
    if physical == logical:
        return
    nvfp4 = dtype == torch.uint8
    # Preflight all leaves before mutating any of them.
    for layer in model.layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            leaf = getattr(layer.mlp, name)
            axis = 1 if name == "down_proj" else 0
            if (leaf.weight.device.type != "cpu" or leaf.weight.dtype != dtype
                    or leaf.weight.shape[axis] != logical
                    or hasattr(leaf, "_rpu_linear_partition")):
                raise ValueError("Pi0.5 MLP padding requires unswizzled cold logical weights")
            scale = getattr(leaf, "weight_scale", None)
            if nvfp4:
                n, k = leaf.weight.shape
                if (getattr(leaf, "_pi05_nvfp4_abi", None) != "striped_v2"
                        or getattr(leaf, "_pi05_nvfp4_packed", False)
                        or scale is None or scale.dtype != torch.uint8
                        or tuple(scale.shape) != (k // 16, n)):
                    raise ValueError("Pi0.5 padded NVFP4 MLP requires unpacked logical "
                                     "codes and [K/16,N] FP8 block scales")
            elif dtype == torch.int8 and name != "down_proj":
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
            if nvfp4:
                # Gate/Up add scale columns (N); Down adds scale rows (K/16).
                scale = leaf.weight_scale
                size = list(scale.shape)
                size[1 - axis] = physical // 16 if axis else physical
                block = scale.new_zeros(size)
                block.narrow(1 - axis, 0, scale.shape[1 - axis]).copy_(scale)
                leaf._buffers["weight_scale"] = block
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
