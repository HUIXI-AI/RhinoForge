"""Cold component domains for the admitted 35B-A3B checkpoint profiles."""
from dataclasses import dataclass

from rpu_backend.api.errors import UnsupportedModelError
from rpu_backend.runtime.topology import DecoderTopology


@dataclass(frozen=True)
class MoeTopology(DecoderTopology):
    routed_weight_mode: int = 5

    def __post_init__(self):
        super().__post_init__()
        if type(self.routed_weight_mode) is not int or self.routed_weight_mode not in (5, 6):
            raise ValueError("MoE routed weight mode must be E4M3 or UINT4")

    @property
    def routed_tp(self):
        return 4 if self.num_cores == 6 and self.routed_weight_mode == 6 else self.num_cores

    def identity(self):
        return full_topology_tuple(self)


def core_topology(execution=None, *, config=None):
    cores = (execution or {}).get("model", {}).get("num_cores", 8)
    if type(cores) is not int or cores not in (4, 6, 8):
        raise UnsupportedModelError(
            "Qwen3.5-35B-A3B requires fixed model.num_cores=4, 6 or 8"
        )
    mode = 5
    if config is not None:
        from .quant_scope import runtime_capabilities, validate_exact_profile

        capabilities = runtime_capabilities(validate_exact_profile(config)["profile"])
        if cores not in capabilities.allowed_cores:
            raise UnsupportedModelError("Qwen3.5-MoE profile does not admit the selected core budget")
        mode = capabilities.routed_weight_mode
    attention = 8 if cores == 8 else 4
    return MoeTopology(cores, attention, cores, attention, routed_weight_mode=mode)


def full_topology_tuple(topology):
    """Versioned text owners; Vision attests its own composite child separately."""
    return (2, topology.num_cores, topology.attn_tp, topology.attn_tp,
            topology.attn_tp, topology.num_cores, topology.routed_tp, topology.mlp_tp,
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
