"""Cold memory accounting for the fixed JanusVLN Extra FP16 profile.

This is a controlled port, not a public supported-model admission.
"""
import json
import math
import struct
from pathlib import Path


def checkpoint_bytes(root):
    """Read safetensors headers only; do not load or transform weights."""
    root = Path(root).resolve()
    cfg = json.loads((root / "config.json").read_text())
    expected = {
        "architectures": ["Qwen2_5_VLForConditionalGenerationForJanusVLN"],
        "hidden_size": 3584, "num_hidden_layers": 28,
        "num_attention_heads": 28, "num_key_value_heads": 4,
        "intermediate_size": 18944, "vocab_size": 152064,
        "reference_frame": "first", "use_vggt_feature": True,
        "use_sliding_window": False, "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6, "hidden_act": "silu", "attention_dropout": 0.0,
    }
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("checkpoint does not match JanusVLN Extra")
    if cfg.get("rope_scaling", {}).get("mrope_section") != [16, 24, 24]:
        raise ValueError("Janus requires M-RoPE [16,24,24]")
    vision = {"depth": 32, "hidden_size": 1280, "num_heads": 16,
              "intermediate_size": 3420, "patch_size": 14, "temporal_patch_size": 2,
              "spatial_merge_size": 2, "window_size": 112,
              "fullatt_block_indexes": [7, 15, 23, 31]}
    if any(cfg.get("vision_config", {}).get(k) != v for k, v in vision.items()):
        raise ValueError("checkpoint does not match the Janus vision profile")
    if cfg.get("lam", 0.2) != 0.2:
        raise ValueError("Janus feature fusion requires lambda=0.2")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    totals = {}
    largest = 0
    seen = set()
    manifest = root / "stream_manifest.json"
    streamed = json.loads(manifest.read_text()) if manifest.exists() else None
    for shard in sorted(set(index["weight_map"].values())):
        path = (root / shard).resolve()
        if root not in path.parents:
            raise ValueError("checkpoint shard escapes its directory")
        if streamed:
            entry = streamed["shards"][shard]
            length, header, file_size = entry["header_length"], entry["header"], entry["file_size"]
        else:
            with path.open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                if length > 64 * 1024**2:
                    raise ValueError("oversized safetensors header")
                header = json.loads(stream.read(length))
            file_size = path.stat().st_size
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key in seen or index["weight_map"].get(key) != shard:
                raise ValueError("checkpoint index/header mismatch")
            seen.add(key)
            if meta["dtype"] not in ("BF16", "F16"):
                raise ValueError(f"unexpected weight dtype: {key}")
            size = math.prod(meta["shape"]) * 2
            begin, end = meta["data_offsets"]
            if (begin < 0 or end - begin != size
                    or 8 + length + end > file_size):
                raise ValueError(f"invalid tensor size: {key}")
            group = key.split(".")[0]
            totals[group] = totals.get(group, 0) + size
            largest = max(largest, size)
    if seen != set(index["weight_map"]):
        raise ValueError("checkpoint tensors are missing")
    return totals, largest


def memory_plan(weight_bytes, largest_tensor_bytes, *, text_capacity=4096,
                checkpoint_on_tmpfs=True):
    if type(text_capacity) is not int or not 1 <= text_capacity <= 4096:
        raise ValueError("controlled text capacity must be in [1,4096]")
    # Raw, pre-normalization K/V: [24,2,1,16,frames,1615,64].
    frame = 2 * 16 * 1615 * 64 * 2
    weights = sum(weight_bytes.values())
    allocations = {
        "weights_fp16": weights,
        "checkpoint_tmpfs": weights if checkpoint_on_tmpfs else 0,
        "spatial_kv_56_frames": 24 * 56 * frame,
        # During append/crop, old + 57-frame new + cropped tensors may coexist.
        "spatial_append_crop_scratch": (57 + 56) * frame,
        "text_kv": 28 * 2 * text_capacity * 4 * 128 * 2,
        "streaming_weight_conversion_scratch": 2 * largest_tensor_bytes,
        "normalized_active_layer_keys": 57 * 1615 * 1024 * 4,
        "cpu_online_attention_scratch": 32 * 1024**2,
        "frontend_activation_reserve": 2 * 1024**3,
        "bounded_graph_and_device_workspace_reserve": 512 * 1024**2,
        "rpu_weight_padding_reserve": 64 * 1024**2,
        "runtime_os_reserve": 4 * 1024**3,
    }
    return {
        "allocations_bytes": allocations,
        "conservative_host_peak_bytes": sum(allocations.values()),
        "estimated_device_allocation_bytes": (
            weights
            + allocations["text_kv"] + allocations["bounded_graph_and_device_workspace_reserve"]
            + allocations["rpu_weight_padding_reserve"]),
        "spatial_tokens_at_step_57": 57 * 1615,
        "query_chunk": 128, "key_chunk": 1024,
        "attention_location": "RPU FP16 tiled QK/PV/projections; CPU FP32 online softmax",
        "spm_scope": "bounded shared-arena Linear tiles; no full-frame norm or history in SPM",
        "status": "planning estimate; native decoder/frontend residency needs measurement",
    }


def admit_memory(weight_bytes, largest_tensor_bytes, meminfo):
    # Files already exist at admission; tmpfs occupancy is reflected in MemAvailable.
    plan = memory_plan(weight_bytes, largest_tensor_bytes, checkpoint_on_tmpfs=False)
    required = {"MemAvailable": plan["conservative_host_peak_bytes"],
                "CmaFree": plan["estimated_device_allocation_bytes"]}
    for key, size in required.items():
        if meminfo.get(key, 0) < size:
            raise MemoryError(f"Janus needs {size} bytes of {key}, available={meminfo.get(key, 0)}")
    plan["admission_meminfo_bytes"] = {key: meminfo[key] for key in required}
    plan["admission_limit"] = "CmaFree is a necessary capacity check, not a contiguous-allocation guarantee"
    return plan
