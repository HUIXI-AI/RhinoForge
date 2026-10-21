"""Dependency-free reader for the frozen Whisper encoder ONNX weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rpu_backend.api.errors import UnsupportedModelError


def _varint(data: memoryview, offset: int, end: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if offset >= end:
            raise UnsupportedModelError("Whisper encoder ONNX is truncated")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
    raise UnsupportedModelError("Whisper encoder ONNX has an invalid varint")


def _fields(data: memoryview):
    offset = 0
    end = len(data)
    while offset < end:
        key, offset = _varint(data, offset, end)
        number, wire = key >> 3, key & 7
        if number == 0:
            raise UnsupportedModelError("Whisper encoder ONNX has field zero")
        if wire == 0:
            value, offset = _varint(data, offset, end)
            yield number, wire, value
        elif wire == 1:
            if offset + 8 > end:
                raise UnsupportedModelError("Whisper encoder ONNX is truncated")
            yield number, wire, data[offset:offset + 8]
            offset += 8
        elif wire == 2:
            size, offset = _varint(data, offset, end)
            if size > end - offset:
                raise UnsupportedModelError("Whisper encoder ONNX is truncated")
            yield number, wire, data[offset:offset + size]
            offset += size
        elif wire == 5:
            if offset + 4 > end:
                raise UnsupportedModelError("Whisper encoder ONNX is truncated")
            yield number, wire, data[offset:offset + 4]
            offset += 4
        else:
            raise UnsupportedModelError(
                f"Whisper encoder ONNX uses unsupported protobuf wire type {wire}"
            )


def _tensor(payload: memoryview):
    dimensions = []
    data_type = None
    name = None
    raw = None
    for number, wire, value in _fields(payload):
        if number == 1:
            if wire == 0:
                dimensions.append(value)
            elif wire == 2:
                offset = 0
                while offset < len(value):
                    dimension, offset = _varint(value, offset, len(value))
                    dimensions.append(dimension)
        elif number == 2 and wire == 0:
            data_type = value
        elif number == 8 and wire == 2:
            try:
                name = bytes(value).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise UnsupportedModelError(
                    "Whisper encoder ONNX has an invalid tensor name"
                ) from exc
        elif number == 9 and wire == 2:
            raw = value
    return name, tuple(dimensions), data_type, raw


def load_float_initializers(
    path: str | Path, expected: dict[str, tuple[int, ...]]
) -> dict[str, np.ndarray]:
    """Read only named FLOAT raw-data tensors from one validated ONNX file."""
    storage = memoryview(Path(path).read_bytes())
    graphs = [value for number, wire, value in _fields(storage)
              if number == 7 and wire == 2]
    if len(graphs) != 1:
        raise UnsupportedModelError("Whisper encoder ONNX must contain one graph")
    loaded = {}
    for number, wire, value in _fields(graphs[0]):
        if number != 5 or wire != 2:
            continue
        name, dimensions, data_type, raw = _tensor(value)
        if name not in expected:
            continue
        shape = expected[name]
        if dimensions != shape or data_type != 1 or raw is None:
            raise UnsupportedModelError(
                f"Whisper encoder ONNX initializer mismatch: {name}"
            )
        elements = int(np.prod(shape, dtype=np.int64))
        if len(raw) != elements * np.dtype("<f4").itemsize:
            raise UnsupportedModelError(
                f"Whisper encoder ONNX initializer size mismatch: {name}"
            )
        loaded[name] = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
    missing = expected.keys() - loaded.keys()
    if missing:
        raise UnsupportedModelError(
            "Whisper encoder ONNX initializers are missing: "
            + ", ".join(sorted(missing))
        )
    return loaded


def load_encoder_weights(path: str | Path) -> dict[str, object]:
    expected = {
        "audioEncoder.positional_embedding": (1500, 512),
        "audioEncoder.ln_post.weight": (512,),
        "audioEncoder.ln_post.bias": (512,),
        "onnx::Conv_1218": (512, 80, 3),
        "onnx::Conv_1219": (512,),
        "onnx::Conv_1220": (512, 512, 3),
        "onnx::Conv_1221": (512,),
    }
    for layer in range(6):
        base = 1225 + 23 * layer
        expected.update({
            f"audioEncoder.blocks.{layer}.attn_ln.weight": (512,),
            f"audioEncoder.blocks.{layer}.attn_ln.bias": (512,),
            f"audioEncoder.blocks.{layer}.mlp_ln.weight": (512,),
            f"audioEncoder.blocks.{layer}.mlp_ln.bias": (512,),
            f"onnx::Add_{base}": (512,),
            f"onnx::MatMul_{base + 1}": (512, 512),
            f"onnx::MatMul_{base + 3}": (512, 512),
            f"onnx::Add_{base + 5}": (512,),
            f"onnx::MatMul_{base + 6}": (512, 512),
            f"onnx::Add_{base + 14}": (512,),
            f"onnx::MatMul_{base + 15}": (512, 512),
            f"onnx::Add_{base + 17}": (2048,),
            f"onnx::MatMul_{base + 18}": (512, 2048),
            f"onnx::Add_{base + 20}": (512,),
            f"onnx::MatMul_{base + 21}": (2048, 512),
        })
    tensors = load_float_initializers(path, expected)
    layers = []
    for layer in range(6):
        base = 1225 + 23 * layer
        layers.append({
            "ln1_w": tensors[f"audioEncoder.blocks.{layer}.attn_ln.weight"],
            "ln1_b": tensors[f"audioEncoder.blocks.{layer}.attn_ln.bias"],
            "ln2_w": tensors[f"audioEncoder.blocks.{layer}.mlp_ln.weight"],
            "ln2_b": tensors[f"audioEncoder.blocks.{layer}.mlp_ln.bias"],
            "q_b": tensors[f"onnx::Add_{base}"],
            "q_w": tensors[f"onnx::MatMul_{base + 1}"].T.copy(),
            "k_w": tensors[f"onnx::MatMul_{base + 3}"].T.copy(),
            "v_b": tensors[f"onnx::Add_{base + 5}"],
            "v_w": tensors[f"onnx::MatMul_{base + 6}"].T.copy(),
            "o_b": tensors[f"onnx::Add_{base + 14}"],
            "o_w": tensors[f"onnx::MatMul_{base + 15}"].T.copy(),
            "up_b": tensors[f"onnx::Add_{base + 17}"],
            "up_w": tensors[f"onnx::MatMul_{base + 18}"].T.copy(),
            "down_b": tensors[f"onnx::Add_{base + 20}"],
            "down_w": tensors[f"onnx::MatMul_{base + 21}"].T.copy(),
        })
    return {
        "conv1_w": tensors["onnx::Conv_1218"],
        "conv1_b": tensors["onnx::Conv_1219"],
        "conv2_w": tensors["onnx::Conv_1220"],
        "conv2_b": tensors["onnx::Conv_1221"],
        "position": tensors["audioEncoder.positional_embedding"],
        "final_w": tensors["audioEncoder.ln_post.weight"],
        "final_b": tensors["audioEncoder.ln_post.bias"],
        "layers": layers,
    }
