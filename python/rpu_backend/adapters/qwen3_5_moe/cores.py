"""Fixed cold topologies for the exact mixed-E4M3 35B-A3B model."""
from rpu_backend.api.errors import UnsupportedModelError
from rpu_backend.runtime.topology import DecoderTopology


def core_topology(execution=None):
    cores = (execution or {}).get("model", {}).get("num_cores", 8)
    if type(cores) is not int or cores not in (4, 6, 8):
        raise UnsupportedModelError(
            "Qwen3.5-35B-A3B requires fixed model.num_cores=4, 6 or 8"
        )
    attention = 8 if cores == 8 else 4
    return DecoderTopology(cores, attention, cores, attention)


def full_topology_tuple(topology):
    """Versioned text owners; Vision attests its own composite child separately."""
    return (2, topology.num_cores, topology.attn_tp, topology.attn_tp,
            topology.attn_tp, topology.mlp_tp, topology.mlp_tp, topology.mlp_tp,
            topology.lm_head_tp, topology.physical_kv_cores)


def validate_native_topology(handle, topology):
    import torch
    from rpu_backend.adapters.qwen3_5.cores import topology_tuple

    actual = tuple(torch.ops.rpu.qwen3_5_moe_get_execution_topology(handle))
    expected = topology_tuple(topology)
    if actual != expected:
        raise RuntimeError(
            f"Qwen3.5-MoE native/Python topology mismatch: {actual}, expected {expected}"
        )
    actual_full = tuple(torch.ops.rpu.qwen3_5_moe_get_execution_topology_v2(handle))
    expected_full = full_topology_tuple(topology)
    if actual_full != expected_full:
        raise RuntimeError(
            f"Qwen3.5-MoE native/Python component topology mismatch: {actual_full}, "
            f"expected {expected_full}"
        )
