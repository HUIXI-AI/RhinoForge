"""Tensor-at-a-time loading through a loopback SSH tunnel, or local files."""
import hashlib
import json
from urllib.parse import quote
from urllib.request import urlopen

import torch
from rpu_backend.adapters.wall_oss.llm import _SafeTensorStore


class CheckpointStore(_SafeTensorStore):
    def __init__(self, root, mmap=False):
        super().__init__(root, mmap=False)
        manifest = self.root / "stream_manifest.json"
        self.stream = json.loads(manifest.read_text()) if manifest.exists() else None
        if self.stream and self.stream["base_url"] != "http://127.0.0.1:18765":
            raise ValueError("streaming checkpoint requires the fixed loopback SSH tunnel")
        if self.stream:
            if self.stream.get("checkpoint_revision") != "8631eee38a4c688256473ec6a59917fb5eec58ff":
                raise ValueError("stream manifest has the wrong checkpoint revision")
            for filename, field in (("config.json", "config_sha256"),
                                    ("model.safetensors.index.json", "index_sha256")):
                if hashlib.sha256((self.root / filename).read_bytes()).hexdigest() != self.stream.get(field):
                    raise ValueError(f"stream manifest identity mismatch: {filename}")

    def _header(self, shard):
        if self.stream:
            entry = self.stream["shards"][shard]
            return entry["header"], 8 + entry["header_length"]
        return super()._header(shard)

    def get_tensor(self, name):
        if not self.stream:
            return super().get_tensor(name)
        shard = self.weight_map[name]
        meta = self._header(shard)[0][name]
        begin, end = meta["data_offsets"]
        buffer = bytearray(end - begin)
        with urlopen(self.stream["base_url"] + "/tensor/" + quote(name, safe=""), timeout=180) as response:
            if int(response.headers["Content-Length"]) != len(buffer):
                raise ValueError(f"tensor length mismatch: {name}")
            view = memoryview(buffer)
            offset = 0
            while offset < len(buffer):
                count = response.readinto(view[offset:])
                if not count:
                    raise EOFError(f"truncated tensor: {name}")
                offset += count
        if hashlib.sha256(buffer).hexdigest() != self.stream["tensor_sha256"][name]:
            raise ValueError(f"tensor checksum mismatch: {name}")
        dtype = {"BF16": torch.bfloat16, "F16": torch.float16}[meta["dtype"]]
        return torch.frombuffer(buffer, dtype=dtype).reshape(meta["shape"])
