"""Bounded, indexed access to a source-bound routed-only INT4 checkpoint."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from safetensors import safe_open
import torch

from rpu_backend.quant.grouped_uint4 import (
    pack_grouped_uint4, pack_grouped_uint4_metadata, runtime_group_scales,
)


_EXPERT = re.compile(
    r"^model\.language_model\.layers\.(0|[1-9]\d*)\.mlp\.experts\.(0|[1-9]\d*)\."
    r"(gate_proj|up_proj|down_proj)\.(qweight|scales|qzeros|g_idx)$"
)
_ROLES = ("gate_proj", "up_proj", "down_proj")
_SUFFIXES = ("qweight", "scales", "qzeros", "g_idx")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"INT4 metadata repeats key {key!r}")
        result[key] = value
    return result


def _json(path: Path):
    return json.loads(path.read_text(), object_pairs_hook=_unique_object)


@dataclass(frozen=True)
class _Shard:
    path: Path
    size: int
    mtime_ns: int

    def check(self):
        stat = self.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (self.size, self.mtime_ns):
            raise ValueError(f"INT4 checkpoint changed after admission: {self.path}")


class IndexedInt4Checkpoint:
    """Common indexed inventory and lazy access, with source-specific hooks.

    Cold admission checks metadata, sizes and inventory, not full payload hashes.
    Each source owns its decoder and provenance contract; callers cannot select
    a zero-point convention independently of the checkpoint type.
    """

    profile_id: str
    identity_descriptor = None

    def check_shard_record(self, filename, stat):
        raise NotImplementedError

    def check_shard_metadata(self, filename, metadata):
        raise NotImplementedError

    def decode_matrix(self, **tensors):
        raise NotImplementedError

    def __init__(self, directory, *, num_layers, num_experts, hidden_size,
                 intermediate_size):
        self.directory = Path(directory).resolve()
        if ".partial" in self.directory.name:
            raise ValueError("INT4 loading requires the published checkpoint, not staging")
        self.num_layers, self.num_experts = num_layers, num_experts
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        if any(type(v) is not int or v <= 0 for v in
               (num_layers, num_experts, hidden_size, intermediate_size)):
            raise ValueError("INT4 geometry must contain positive integers")
        index_bytes = (self.directory / "model.safetensors.index.json").read_bytes()
        self.index_sha256 = hashlib.sha256(index_bytes).hexdigest()
        index = json.loads(index_bytes, object_pairs_hook=_unique_object)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("INT4 checkpoint requires a nonempty weight_map")
        self.shards = {}
        self.shard_descriptors = []
        self.tensor_shards = {}
        self.regular_keys = {}
        self.matrix_keys = {layer: {role: {} for role in _ROLES}
                            for layer in range(num_layers)}
        total_bytes = 0
        seen = set()
        if any(not isinstance(name, str) or Path(name).name != name
               or not name.endswith(".safetensors") for name in weight_map.values()):
            raise ValueError("INT4 index shard names must be local safetensors basenames")
        for filename in sorted(set(weight_map.values())):
            path = self.directory / filename
            if path.resolve().parent != self.directory:
                raise ValueError("INT4 shard resolves outside its checkpoint directory")
            stat = path.stat()
            self.check_shard_record(filename, stat)
            shard = _Shard(path, stat.st_size, stat.st_mtime_ns)
            with path.open("rb") as handle:
                size_bytes = handle.read(8)
                if len(size_bytes) != 8:
                    raise ValueError(f"INT4 shard header is truncated: {filename}")
                header_len = struct.unpack("<Q", size_bytes)[0]
                if header_len > min(stat.st_size - 8, 256 * 1024**2):
                    raise ValueError(f"INT4 shard header length is invalid: {filename}")
                header_bytes = handle.read(header_len)
                header = json.loads(header_bytes, object_pairs_hook=_unique_object)
            shard.check()
            self.shards[filename] = shard
            self.shard_descriptors.append({
                "filename": filename, "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
            })
            expected = {key for key, value in weight_map.items() if value == filename}
            actual = set(header) - {"__metadata__"}
            if actual != expected or seen & actual:
                raise ValueError(f"INT4 shard keys disagree with its index: {filename}")
            seen.update(actual)
            metadata = header.get("__metadata__", {})
            self.check_shard_metadata(filename, metadata)
            intervals = []
            for key in actual:
                row = header[key]
                itemsize = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4}.get(row["dtype"])
                shape, offsets = row["shape"], row["data_offsets"]
                if (itemsize is None or not isinstance(shape, list)
                        or any(type(v) is not int or v < 0 for v in shape)
                        or not isinstance(offsets, list) or len(offsets) != 2
                        or any(type(v) is not int or v < 0 for v in offsets)
                        or offsets[1] - offsets[0] != math.prod(shape) * itemsize):
                    raise ValueError(f"INT4 tensor has invalid shape/dtype/payload: {key}")
                intervals.append(row["data_offsets"])
                total_bytes += row["data_offsets"][1] - row["data_offsets"][0]
                self.tensor_shards[key] = filename
                match = _EXPERT.fullmatch(key)
                if match is None:
                    main_expert = (key.startswith("model.language_model.layers.")
                                   and ".mlp.experts." in key)
                    if (main_expert or row["dtype"] not in ("BF16", "F16", "F32")
                            or key.endswith((".qweight", ".qzeros", ".scales", ".g_idx", "_scale"))):
                        raise ValueError(f"INT4 tensor lies outside the routed-only scope: {key}")
                    self.regular_keys.setdefault(filename, []).append(key)
                    continue
                layer, expert = int(match[1]), int(match[2])
                role, suffix = match[3], match[4]
                if not (0 <= layer < num_layers and 0 <= expert < num_experts):
                    raise ValueError(f"INT4 expert/layer index out of range: {key}")
                n, k = self.shape(role)
                dtype, shape = {
                    "qweight": ("I32", [k // 8, n]),
                    "scales": ("F16", [k // 128, n]),
                    "qzeros": ("I32", [k // 128, n // 8]),
                    "g_idx": ("I32", [k]),
                }[suffix]
                if row["dtype"] != dtype or row["shape"] != shape:
                    raise ValueError(f"INT4 {key} expected {dtype} {shape}, got {row}")
                self.matrix_keys[layer][role].setdefault(expert, {})[suffix] = key
            end = 0
            for lo, hi in sorted(intervals):
                if lo != end or hi < lo:
                    raise ValueError(f"INT4 shard has invalid tensor offsets: {filename}")
                end = hi
            if end + header_len + 8 != stat.st_size:
                raise ValueError(f"INT4 shard payload size mismatch: {filename}")
        if total_bytes != index.get("metadata", {}).get("total_size"):
            raise ValueError("INT4 index total_size disagrees with its shards")
        for layer, roles in self.matrix_keys.items():
            for role, experts in roles.items():
                if set(experts) != set(range(num_experts)) or any(
                        set(keys) != set(_SUFFIXES) for keys in experts.values()):
                    raise ValueError(f"INT4 incomplete expert quartet: layer={layer}, role={role}")

    def shape(self, role):
        if role == "down_proj":
            return self.hidden_size, self.intermediate_size
        return self.intermediate_size, self.hidden_size

    @contextmanager
    def open_shards(self, filenames):
        with ExitStack() as stack:
            opened = {}
            for filename in sorted(set(filenames)):
                shard = self.shards[filename]
                shard.check()
                opened[filename] = stack.enter_context(
                    safe_open(str(shard.path), framework="pt", device="cpu"))
            yield opened
            # Catch mutations during decoding as well as between layers,
            # before the caller can upload the completed layer's payload.
            for filename in opened:
                self.shards[filename].check()

    def layer_source(self, layer):
        if type(layer) is not int or not 0 <= layer < self.num_layers:
            raise ValueError("INT4 layer source index out of range")
        return RoutedInt4LayerSource(self, layer)


class RoutedInt4LayerSource:
    """One layer's lazy matrix references; consumed after successful upload."""

    def __init__(self, checkpoint: IndexedInt4Checkpoint, layer: int):
        self.checkpoint = checkpoint
        self.layer = layer
        self.identity = checkpoint.identity
        self.profile_id = checkpoint.profile_id
        self.consumed = False

    def prepare(self, *, num_cores=8):
        # Preserve the tuple interface; both callers use published N32K32.
        projections = list(self.iter_prepared_projections(num_cores=num_cores))
        return (*(item[1] for item in projections),
                *(item[2] for item in projections), 6)

    def iter_prepared_projections(self, *, num_cores=8):
        """Yield one verified N32K32 projection plus metadata at a time.

        No release occurs here. The installing caller releases the source only
        after all owners and the rest of the layer upload successfully.
        Shard identity is checked on context exit before each yield.
        """
        if type(num_cores) is not int or num_cores not in (4, 6, 8):
            raise ValueError("Qwen3.5 routed INT4 requires TP4, TP6 or TP8")
        if self.consumed or self.checkpoint is None:
            raise ValueError("Qwen3.5 INT4 expert source was already consumed")
        checkpoint = self.checkpoint
        if (checkpoint.profile_id != self.profile_id or checkpoint.identity != self.identity):
            raise ValueError("INT4 expert source identity/profile changed after admission")
        for role in _ROLES:
            n, k = checkpoint.shape(role)
            partition = "row" if role == "down_proj" else "col"
            physical_n, physical_k = n, k
            if num_cores == 6:
                if (k if partition == "row" else n) != 512:
                    raise ValueError("Qwen3.5 reduced INT4 requires logical expert width 512")
                if partition == "row":
                    physical_k = 768
                else:
                    physical_n = 768
            keys = checkpoint.matrix_keys[self.layer][role]
            filenames = {checkpoint.tensor_shards[key]
                         for quartet in keys.values() for key in quartet.values()}
            from rpu_backend.quant.int4_pgrp_pack import w4_storage_shape
            storage_n, storage_k = w4_storage_shape(
                physical_n, physical_k, int(partition == "col"), num_cores)
            packed = torch.empty(checkpoint.num_experts, storage_n * storage_k, dtype=torch.uint8)
            physical_scales = None
            with checkpoint.open_shards(filenames) as opened:
                for expert in range(checkpoint.num_experts):
                    tensors = {suffix: opened[checkpoint.tensor_shards[key]].get_tensor(key)
                               for suffix, key in keys[expert].items()}
                    codes, logical_scales = checkpoint.decode_matrix(**tensors, n=n, k=k)
                    if (physical_n, physical_k) != (n, k):
                        padded = codes.new_full((physical_n, physical_k), 8)
                        padded[:n, :k].copy_(codes)
                        padded_scales = logical_scales.new_ones((physical_k // 128, physical_n))
                        padded_scales[:k // 128, :n].copy_(logical_scales)
                        codes, logical_scales = padded, padded_scales
                    physical, _group = runtime_group_scales(
                        logical_scales, k=physical_k, partition=partition, num_cores=num_cores)
                    # The native Down contract uses G64 for every core count.
                    # Repeating G128 scale bits changes no quantized values.
                    if partition == "row" and _group == 128:
                        physical = physical.repeat_interleave(2, dim=0).contiguous()
                    if physical_scales is None:
                        physical_scales = torch.empty(
                            checkpoint.num_experts, *physical.shape, dtype=torch.float16)
                    packed[expert].copy_(pack_grouped_uint4(
                        codes.unsqueeze(0), partition=partition, num_cores=num_cores)[0])
                    physical_scales[expert].copy_(physical)
                    del tensors, codes, logical_scales, physical
            # Both call paths reuse this metadata. Yield only after the shard
            # identity guard has checked the complete projection.
            metadata = pack_grouped_uint4_metadata(
                physical_scales, partition=partition, num_cores=num_cores)
            del physical_scales
            yield role, packed, metadata
            del packed, metadata

    def release(self):
        self.consumed = True
        self.checkpoint = None
