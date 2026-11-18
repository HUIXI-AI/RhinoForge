"""openpi pi05 host transforms for raw robot observations (CPU, no backend import).

An openpi pi05 policy does not consume the LeRobot preprocessor output. Its host
side resizes each camera frame with ``resize_with_pad``, quantile-normalizes the
physical state and writes it into the prompt as 256-bin integers ("discrete
state"), then right-pads the PaliGemma tokens. This module reproduces those
transforms without JAX, producing the flat tensor batch the Pi0.5 runner consumes,
plus the inverse action transform for the normalized chunk it returns.

Frames enter at the camera's own size (Kuavo: 848x480 real, 640x480 simulation,
the ``env.image_size`` of kuavo_deploy); the 224x224 model input is produced here
by the policy's own resize, as in openpi ``preprocess_observation``.

Sources (LeTools-Learning openpi fork eb8687c, kuavo_model/external_models/openpi):
``shared/image_tools.py:resize_with_pad``, ``transforms.py:Normalize/Unnormalize``
(quantile mode for pi05) and ``DeltaActions``, ``models/tokenizer.py:
PaligemmaTokenizer.tokenize`` and ``models/model.py:preprocess_observation``.
``resize_with_pad`` matches ``jax.image.resize`` (linear, antialias) in float32.
A few pixels can differ by one uint8 level from JAX's fused reduction order.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import functools
import json
import math
import os
from pathlib import Path
import platform
import threading
import warnings

# Kuavo cam_h / cam_l / cam_r in this order; pi05_base names its three slots so.
CAMERA_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
RAW_KEYS = ("images", "frame_size", "instruction", "state", "norm_stats", "tokenizer")
STATE_BINS = 256
# Camera frame sizes [width, height] of the Kuavo deployment: real robot, simulation.
FRAME_SIZES = ((848, 480), (640, 480))
MODEL_IMAGE_SIZE = 224


def validate_input_config(inputs: dict) -> bool:
    """Check the raw-observation keys of [input]; True when the raw mode is selected.

    Exactly one of ``batch`` (a prepared tensor dictionary) and ``images`` (raw
    frames transformed here) is allowed. Files are read only at run time.
    """
    raw = [key for key in RAW_KEYS + ("delta_action_mask",) if key in inputs]
    if "batch" in inputs:
        if raw:
            raise ValueError(f"[input].batch already owns its images, tokens and state; remove {raw}")
        return False
    missing = [key for key in RAW_KEYS if key not in inputs]
    if missing:
        raise ValueError("[input] requires batch, or images/frame_size/instruction/state/norm_stats/tokenizer "
                         f"for raw openpi inputs (missing {missing})")
    images = inputs["images"]
    if (not isinstance(images, list) or len(images) != len(CAMERA_KEYS)
            or any(not isinstance(path, str) or not path.strip() for path in images)):
        raise ValueError("[input].images requires three frame paths in base/left wrist/right wrist order")
    if inputs.get("cameras", len(CAMERA_KEYS)) != len(CAMERA_KEYS):
        raise ValueError("raw openpi inputs use exactly three cameras")
    frame_size = inputs["frame_size"]
    if (not isinstance(frame_size, list) or any(type(value) is not int for value in frame_size)
            or tuple(frame_size) not in FRAME_SIZES):
        raise ValueError("[input].frame_size must be the camera frame size [width, height]: "
                         + " or ".join(f"[{w}, {h}]" for w, h in FRAME_SIZES))
    if not isinstance(inputs["instruction"], str) or not inputs["instruction"].strip():
        raise ValueError("[input].instruction must be a nonempty string")
    state = inputs["state"]
    if (not isinstance(state, list) or not 1 <= len(state) <= 32
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in state)):
        raise ValueError("[input].state requires 1..32 finite physical state values")
    for key in ("norm_stats", "tokenizer"):
        if not isinstance(inputs[key], str) or not inputs[key].strip():
            raise ValueError(f"[input].{key} must be a nonempty path")
    mask = inputs.get("delta_action_mask")
    if mask is not None and (not isinstance(mask, list) or not mask or len(mask) > len(state)
                             or any(type(flag) is not bool for flag in mask)):
        raise ValueError("[input].delta_action_mask must be a nonempty boolean list no longer than [input].state")
    return True


@functools.lru_cache(maxsize=16)
def _resize_weights(in_size: int, out_size: int):
    """jax.image._compute_weight_mat: triangle kernel, antialias, no translation (read-only, cached)."""
    import numpy as np

    inv_scale = np.float32(in_size / out_size)
    kernel_scale = max(inv_scale, np.float32(1.0))
    sample = (np.arange(out_size, dtype=np.float32) + np.float32(0.5)) * inv_scale - np.float32(0.5)
    distance = np.abs(sample[None, :] - np.arange(in_size, dtype=np.float32)[:, None]) / kernel_scale
    weights = np.maximum(np.float32(0.0), np.float32(1.0) - distance).astype(np.float32)
    total = weights.sum(axis=0, keepdims=True, dtype=np.float32)
    eps = np.float32(1000.0 * np.finfo(np.float32).eps)
    weights = np.where(np.abs(total) > eps, weights / np.where(total != 0, total, np.float32(1.0)),
                       np.float32(0.0))
    inside = (sample >= -0.5) & (sample <= in_size - 0.5)
    weights = np.where(inside[None, :], weights, np.float32(0.0)).astype(np.float32)
    weights.flags.writeable = False
    return weights


def resize_geometry(width: int, height: int, size: int = MODEL_IMAGE_SIZE) -> dict:
    """Where resize_with_pad puts a width x height frame inside the size x size model image."""
    ratio = max(width / size, height / size)
    new_w, new_h = int(width / ratio), int(height / ratio)
    return {"frame_wh": [width, height], "model_wh": [size, size], "resized_wh": [new_w, new_h],
            "pad_top": (size - new_h) // 2, "pad_left": (size - new_w) // 2}


_WORKER_TLS = threading.local()
_POOL_LOCK = threading.Lock()
_CAMERA_POOL: concurrent.futures.ThreadPoolExecutor | None = None
_CAMERA_POOL_PID: int | None = None


def _init_camera_worker() -> None:
    import torch

    # Prevent nested 3x8 OpenMP thread storms while preserving the main thread's torch thread count.
    torch.set_num_threads(1)


def _camera_pool() -> concurrent.futures.ThreadPoolExecutor:
    global _CAMERA_POOL, _CAMERA_POOL_PID
    pid = os.getpid()
    if _CAMERA_POOL is not None and _CAMERA_POOL_PID == pid:
        return _CAMERA_POOL
    with _POOL_LOCK:
        if _CAMERA_POOL is None or _CAMERA_POOL_PID != pid:
            _CAMERA_POOL = concurrent.futures.ThreadPoolExecutor(
                max_workers=len(CAMERA_KEYS),
                thread_name_prefix="openpi-cam",
                initializer=_init_camera_worker,
            )
            _CAMERA_POOL_PID = pid
        return _CAMERA_POOL


_IS_AARCH64 = platform.machine().lower() in ("aarch64", "arm64")


@functools.lru_cache(maxsize=16)
def _resize_sparse_plan(in_size: int, out_size: int):
    """Cached sparse transpose of _resize_weights (>98.4% zeros on Kuavo 848x480 / 640x480).

    Matches host OpenBLAS ``np.matmul`` IEEE-754 rounding bit-for-bit on each ISA:
    - aarch64: OpenBLAS SGEMM uses unfused fmul+fadd -> matched 100% by CSR SpMM;
    - x86_64: OpenBLAS SGEMM uses AVX2 FMA (vfmadd) -> matched 100% by embedding_bag gather-FMA.
    """
    import numpy as np
    import torch

    wt = np.array(_resize_weights(in_size, out_size).T, copy=True, order="C")
    if _IS_AARCH64:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Sparse CSR.*", category=UserWarning)
            return ("csr", torch.from_numpy(wt).to_sparse_csr())
    nz_r, nz_c = np.nonzero(wt)
    offsets = np.zeros(out_size, dtype=np.int64)
    counts = np.bincount(nz_r, minlength=out_size)
    if out_size > 1:
        offsets[1:] = np.cumsum(counts[:-1])
    return (
        "ebag",
        torch.from_numpy(nz_c.astype(np.int64)),
        torch.from_numpy(offsets),
        torch.from_numpy(wt[nz_r, nz_c].astype(np.float32)),
    )


def _apply_sparse_plan(plan, mat):
    import torch
    import torch.nn.functional as F

    if plan[0] == "csr":
        return torch.matmul(plan[1], mat)
    _, indices, offsets, weights = plan
    return F.embedding_bag(indices, mat, offsets, per_sample_weights=weights, mode="sum")


def _uint8_tensor_2d(image, cur_h: int, cur_w: int):
    import numpy as np
    import torch

    contig = image if image.flags.c_contiguous else np.ascontiguousarray(image)
    if contig.flags.writeable:
        return torch.from_numpy(contig).view(cur_h, cur_w * 3)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable.*", category=UserWarning)
        return torch.from_numpy(contig).view(cur_h, cur_w * 3)


def _resize_core_uint8_planar(image, cur_h: int, cur_w: int, new_h: int, new_w: int, *, use_tls: bool):
    """Compute [new_w, 3, new_h] uint8 resized pixels via cached sparse row-gather."""
    import torch

    plan_h = _resize_sparse_plan(cur_h, new_h)
    plan_w = _resize_sparse_plan(cur_w, new_w)
    u8_2d = _uint8_tensor_2d(image, cur_h, cur_w)
    if use_tls:
        key = (cur_h, cur_w, new_h)
        ws = getattr(_WORKER_TLS, "workspace", None)
        if ws is None or ws[0] != key:
            ws = (
                key,
                torch.empty((cur_h, cur_w * 3), dtype=torch.float32),
                torch.empty((cur_w, 3 * new_h), dtype=torch.float32),
            )
            _WORKER_TLS.workspace = ws
        _, f32, yw = ws
        f32.copy_(u8_2d)
        y = _apply_sparse_plan(plan_h, f32)
        yw.view(cur_w, 3, new_h).copy_(y.view(new_h, cur_w, 3).permute(1, 2, 0))
    else:
        f32 = u8_2d.to(torch.float32)
        y = _apply_sparse_plan(plan_h, f32)
        yw = y.view(new_h, cur_w, 3).permute(1, 2, 0).contiguous().view(cur_w, 3 * new_h)
    z = _apply_sparse_plan(plan_w, yw)
    return torch.clamp_(torch.round_(z), 0.0, 255.0).to(torch.uint8).view(new_w, 3, new_h)


def _resize_frame_to_model_tensor(frame, height: int = 224, width: int = 224):
    """One uint8 [H,W,3] frame -> contiguous [1,3,height,width] float32 in [0,1] (worker entry)."""
    import torch

    cur_h, cur_w = frame.shape[:2]
    if (cur_h, cur_w) == (height, width):
        u8_2d = _uint8_tensor_2d(frame, cur_h, cur_w).view(cur_h, cur_w, 3)
        return u8_2d.to(torch.float32).div_(255.0).permute(2, 0, 1).unsqueeze(0).contiguous()
    ratio = max(cur_w / width, cur_h / height)
    new_h, new_w = int(cur_h / ratio), int(cur_w / ratio)
    u8_planar = _resize_core_uint8_planar(frame, cur_h, cur_w, new_h, new_w, use_tls=True)
    # Cast & divide by 255.0 while contiguous BEFORE permuting into the padded [1, 3, H, W] destination.
    unit_planar = u8_planar.to(torch.float32).div_(255.0)
    pad_h = (height - new_h) // 2
    pad_w = (width - new_w) // 2
    out = torch.zeros((1, 3, height, width), dtype=torch.float32)
    out[0, :, pad_h:pad_h + new_h, pad_w:pad_w + new_w] = unit_planar.permute(1, 2, 0)
    return out


def resize_with_pad(image, height: int = 224, width: int = 224):
    """openpi resize_with_pad for one uint8 [H,W,3] frame: keep aspect, linear+antialias, black pad."""
    import numpy as np
    import torch

    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("resize_with_pad expects a uint8 [H,W,3] frame")
    cur_h, cur_w = image.shape[:2]
    if (cur_h, cur_w) == (height, width):
        return image.copy()
    ratio = max(cur_w / width, cur_h / height)
    new_h, new_w = int(cur_h / ratio), int(cur_w / ratio)
    u8_planar = _resize_core_uint8_planar(image, cur_h, cur_w, new_h, new_w, use_tls=False)
    pad_h = (height - new_h) // 2
    pad_w = (width - new_w) // 2
    out = torch.zeros((height, width, 3), dtype=torch.uint8)
    out[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = u8_planar.permute(2, 0, 1)
    return out.numpy()


def load_frame(path):
    """One RGB frame as uint8 [H,W,3]: an image file (PNG/JPEG) or a .npy array."""
    import numpy as np

    path = Path(path).expanduser()
    if path.suffix == ".npy":
        frame = np.load(path, allow_pickle=False)
    else:
        from PIL import Image

        with Image.open(path) as image:
            frame = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"{path} must hold one uint8 RGB frame [H,W,3]")
    return frame


def check_frames(frames, frame_size) -> None:
    """Refuse a frame that is not uint8 [height, width, 3] at the configured camera size."""
    import numpy as np

    width, height = frame_size
    if len(frames) != len(CAMERA_KEYS):
        raise ValueError(f"openpi pi05 inputs need {len(CAMERA_KEYS)} frames in base/left wrist/right wrist order")
    for key, frame in zip(CAMERA_KEYS, frames):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.shape != (height, width, 3):
            shape = list(frame.shape) if hasattr(frame, "shape") else type(frame).__name__
            raise ValueError(f"{key} frame is {shape}; this configuration takes uint8 [{height}, {width}, 3] "
                             f"camera frames ({width}x{height})")


def load_norm_stats(path) -> dict:
    """openpi ``norm_stats.json``: {"norm_stats": {"state": {...}, "actions": {...}}}."""
    document = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    stats = document.get("norm_stats", document) if isinstance(document, dict) else None
    if not isinstance(stats, dict) or "state" not in stats:
        raise ValueError(f"{path} is not an openpi norm_stats.json with state statistics")
    for name, table in stats.items():
        if not isinstance(table, dict) or not {"q01", "q99"} <= set(table):
            raise ValueError(f"norm_stats.{name} needs q01/q99 (pi05 uses quantile normalization)")
    return stats


def normalize_quantile(values, stats: dict):
    """openpi Normalize(use_quantiles=True); statistics may be longer than the value."""
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    q01 = np.asarray(stats["q01"], dtype=np.float64)[: values.shape[-1]]
    q99 = np.asarray(stats["q99"], dtype=np.float64)[: values.shape[-1]]
    if q01.shape[-1] != values.shape[-1]:
        raise ValueError(f"norm stats cover {q01.shape[-1]} dims, value has {values.shape[-1]}")
    return (values - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def unnormalize_quantile(values, stats: dict):
    """openpi Unnormalize(use_quantiles=True); dims beyond the statistics pass through."""
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    q01 = np.asarray(stats["q01"], dtype=np.float64)
    q99 = np.asarray(stats["q99"], dtype=np.float64)
    dim = min(q01.shape[-1], values.shape[-1])
    result = values.copy()
    result[..., :dim] = (values[..., :dim] + 1.0) / 2.0 * (q99[:dim] - q01[:dim] + 1e-6) + q01[:dim]
    return result


def discretize_state(normalized_state):
    """openpi's 256-bin state integers; a value below -1 gives -1, at or above 1 gives 255."""
    import numpy as np

    return np.digitize(normalized_state, bins=np.linspace(-1, 1, STATE_BINS + 1)[:-1]) - 1


def format_prompt(instruction: str, bins) -> str:
    """PaligemmaTokenizer.tokenize prompt text for already discretized state bins."""
    cleaned = instruction.strip().replace("_", " ").replace("\n", " ")
    return f"Task: {cleaned}, State: {' '.join(map(str, bins))};\nAction: "


def pi05_prompt(instruction: str, normalized_state) -> str:
    """PaligemmaTokenizer.tokenize prompt text with the discretized state."""
    return format_prompt(instruction, discretize_state(normalized_state))


def reachable_token_counts(processor, instruction: str, state_dim: int, text_tokens: int | None = None):
    """Every BOS+prompt token count this instruction reaches over all discretized states.

    Each state bin (-1..255) is written as its own space-separated number, so a
    count is the all-zero prompt's count plus one independent cost per dimension;
    the costs are measured per dimension and the sum is checked on sampled
    states. A count above ``text_tokens`` is refused: that state could not run.
    """
    import numpy as np

    if type(state_dim) is not int or not 1 <= state_dim <= 32:
        raise ValueError("state_dim must be in 1..32")
    values = range(-1, STATE_BINS)

    def count(bins):
        return len(processor.encode(format_prompt(instruction, bins), add_bos=True))

    zero = [0] * state_dim
    base = count(zero)
    costs = []
    for dim in range(state_dim):
        row = {}
        for value in values:
            bins = list(zero)
            bins[dim] = value
            row[value] = count(bins) - base
        costs.append(row)
    rng = np.random.default_rng(0)
    for _ in range(256):
        bins = [int(value) for value in rng.integers(-1, STATE_BINS, size=state_dim)]
        expected = base + sum(costs[dim][value] for dim, value in enumerate(bins))
        if count(bins) != expected:
            raise RuntimeError("the tokenizer's state cost is not per-dimension additive; "
                               "list the prepared token counts explicitly")
    reachable = {base}
    for row in costs:
        deltas = set(row.values())
        reachable = {total + delta for total in reachable for delta in deltas}
    counts = sorted(reachable)
    if text_tokens is not None and counts[-1] > text_tokens:
        raise ValueError(f"the instruction reaches {counts[-1]} tokens with its state; "
                         f"text_tokens={text_tokens} is too small (shorten it or select a wider storage width)")
    return counts


def load_tokenizer(path):
    """openpi's PaliGemma SentencePiece model (``paligemma_tokenizer.model``)."""
    try:
        import sentencepiece
    except ImportError as error:
        raise ImportError("raw openpi inputs need the sentencepiece package for the PaliGemma tokenizer "
                          "(pip install sentencepiece)") from error
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"PaliGemma tokenizer model not found: {path}")
    return sentencepiece.SentencePieceProcessor(model_file=str(path))


def tokenize(processor, prompt: str, text_tokens: int):
    """BOS + prompt pieces, right-padded with id 0 / mask False to the storage width.

    openpi truncates an overlong prompt to its 200-token width; this entry refuses
    instead, because a cut state or ``Action:`` suffix changes the model input.
    """
    tokens = list(processor.encode(prompt, add_bos=True))
    if len(tokens) > text_tokens:
        raise ValueError(f"the prompt has {len(tokens)} tokens with its state; text_tokens={text_tokens} "
                         "is too small (shorten the instruction or select a wider storage width)")
    padding = text_tokens - len(tokens)
    return tokens + [0] * padding, [True] * len(tokens) + [False] * padding


def batch_from_frames(frames, state, instruction: str, stats: dict, processor, *,
                      text_tokens: int, frame_size=None) -> tuple[dict, dict]:
    """Three uint8 [H,W,3] frames (base/left/right) + physical state -> (runner batch, record).

    ``frame_size`` ([width, height]) refuses frames of any other size.
    """
    import numpy as np
    import torch

    if frame_size is not None:
        check_frames(frames, frame_size)
    elif len(frames) != len(CAMERA_KEYS):
        raise ValueError(f"openpi pi05 inputs need {len(CAMERA_KEYS)} frames in base/left wrist/right wrist order")
    for key, frame in zip(CAMERA_KEYS, frames):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"{key} frame must be a uint8 [H,W,3] array")
    # Pre-warm sparse resize plan cache on main thread, then dispatch 3 cameras in parallel with tokenization.
    for frame in frames:
        fh, fw = frame.shape[:2]
        if (fh, fw) != (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE):
            ratio = max(fw / MODEL_IMAGE_SIZE, fh / MODEL_IMAGE_SIZE)
            _resize_sparse_plan(fh, int(fh / ratio))
            _resize_sparse_plan(fw, int(fw / ratio))
    pool = _camera_pool()
    futures = [pool.submit(_resize_frame_to_model_tensor, frame, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE) for frame in frames]
    try:
        state = np.asarray(state, dtype=np.float64)
        normalized = normalize_quantile(state, stats["state"])
        prompt = pi05_prompt(instruction, normalized)
        tokens, mask = tokenize(processor, prompt, text_tokens)
        images = [future.result() for future in futures]
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    batch = {}
    for key, image in zip(CAMERA_KEYS, images):
        batch[f"observation.images.{key}"] = image
        batch[f"observation.images.{key}_padding_mask"] = torch.ones(1, dtype=torch.bool)
    # The pi05 action path reads the state only through the prompt; keep the value for records.
    batch["observation.state"] = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
    batch["observation.language.tokens"] = torch.tensor([tokens], dtype=torch.int64)
    batch["observation.language.attention_mask"] = torch.tensor([mask], dtype=torch.bool)
    height, width = frames[0].shape[:2]
    record = {"source": "raw openpi observation", "resize": "openpi resize_with_pad 224",
              "resize_geometry": resize_geometry(width, height),
              "state_dim": int(state.shape[0]), "prompt": prompt, "valid_text_tokens": int(sum(mask)),
              "text_tokens": text_tokens}
    return batch, record


def load_raw_inputs(inputs: dict, *, processor=None) -> dict:
    """Read the files of a raw [input] once: frames (checked against frame_size), stats, tokenizer."""
    frames, records = [], {}
    for key, path in zip(CAMERA_KEYS, inputs["images"]):
        frame = load_frame(path)
        records[key] = {"path": str(path), "shape": list(frame.shape)}
        frames.append(frame)
    if "frame_size" in inputs:
        check_frames(frames, inputs["frame_size"])
    return {"frames": frames, "frame_records": records, "stats": load_norm_stats(inputs["norm_stats"]),
            "processor": processor or load_tokenizer(inputs["tokenizer"])}


def batch_from_loaded(loaded: dict, inputs: dict, *, text_tokens: int, instruction: str | None = None):
    """The per-request host transforms of already loaded raw inputs -> (runner batch, record)."""
    return batch_from_frames(
        loaded["frames"], inputs["state"], inputs["instruction"] if instruction is None else instruction,
        loaded["stats"], loaded["processor"], text_tokens=text_tokens, frame_size=inputs.get("frame_size"))


def build_batch(inputs: dict, *, text_tokens: int, processor=None) -> tuple[dict, dict]:
    """Raw [input] -> (runner batch, record). Images are [1,3,224,224] float in [0,1]."""
    loaded = load_raw_inputs(inputs, processor=processor)
    batch, record = batch_from_loaded(loaded, inputs, text_tokens=text_tokens)
    record.update(frames=loaded["frame_records"], norm_stats=str(inputs["norm_stats"]),
                  tokenizer=str(inputs["tokenizer"]))
    return batch, record


def physical_actions(actions, inputs: dict, stats: dict | None = None):
    """Normalized [1,H,D] chunk -> robot actions: Unnormalize, then AbsoluteActions on the mask.

    None when the norm stats carry no action statistics. The mask selects the
    joints trained as deltas from the current state (openpi DeltaActions).
    ``stats`` are the already loaded norm stats (default: read [input].norm_stats).
    """
    import numpy as np
    import torch

    stats = load_norm_stats(inputs["norm_stats"]) if stats is None else stats
    if "actions" not in stats:
        return None
    values = absolute_actions(actions.detach().cpu().double().numpy(), stats["actions"],
                              inputs["state"], inputs.get("delta_action_mask"))
    return torch.from_numpy(values.astype(np.float32))


def absolute_actions(actions, action_stats: dict, state, delta_action_mask=None):
    """openpi Unnormalize (quantile), then AbsoluteActions on the masked dims (float64 numpy)."""
    import numpy as np

    values = unnormalize_quantile(actions, action_stats)
    if delta_action_mask:
        state = np.asarray(state, dtype=np.float64)
        dims = len(delta_action_mask)
        if dims > values.shape[-1] or dims > state.shape[0]:
            raise ValueError("delta_action_mask is longer than the action chunk or the physical state")
        # openpi AbsoluteActions: actions[..., :dims] += where(mask, state[:dims], 0).
        values[..., :dims] += np.where(np.asarray(delta_action_mask, dtype=bool), state[:dims], 0.0)
    return values


def checkpoint_view(source, destination, *, action_dim: int) -> dict:
    """A checkpoint directory whose config declares a robot's action/state width.

    Weight files are symlinked; only config.json is rewritten. The model still
    computes 32 action dims; PI05Policy.predict_action_chunk returns the first
    ``action_dim``, as for a checkpoint fine-tuned on that robot.
    """
    source, destination = Path(source).expanduser().resolve(), Path(destination).expanduser()
    if type(action_dim) is not int or not 1 <= action_dim <= 32:
        raise ValueError("action_dim must be in 1..32")
    if destination.exists():
        raise FileExistsError(f"{destination} already exists")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if action_dim > int(config.get("max_action_dim", 32)):
        raise ValueError("action_dim exceeds the checkpoint max_action_dim")
    for table, key in (("output_features", "action"), ("input_features", "observation.state")):
        feature = (config.get(table) or {}).get(key)
        if not isinstance(feature, dict) or len(feature.get("shape", ())) != 1:
            raise ValueError(f"{source}/config.json has no one-dimensional {table}.{key}")
        feature["shape"] = [action_dim]
    weights = [path for path in source.glob("*.safetensors") if path.name != "model_remapped.safetensors"]
    if not weights:
        raise ValueError(f"{source} has no safetensors weights")
    # The loader's key-remap cache depends only on the weights, so a current one
    # is shared; a stale or absent one is left for the loader to write in the view.
    remapped = source / "model_remapped.safetensors"
    share_remap = remapped.is_file() and all(
        path.stat().st_mtime_ns <= remapped.stat().st_mtime_ns for path in weights)
    destination.mkdir(parents=True)
    linked = []
    for path in sorted(source.iterdir()):
        if (path.name == "config.json" or not path.is_file()
                or (path.name == "model_remapped.safetensors" and not share_remap)):
            continue
        (destination / path.name).symlink_to(path)
        linked.append(path.name)
    (destination / "config.json").write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
    return {"source": str(source), "destination": str(destination), "action_dim": action_dim,
            "linked": linked}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    batch = commands.add_parser("batch", help="write the runner batch of a raw-input TOML")
    batch.add_argument("--config", type=Path, required=True)
    batch.add_argument("--out", type=Path, required=True)
    view = commands.add_parser("checkpoint-view", help="symlink a checkpoint with a robot action width")
    view.add_argument("--src", type=Path, required=True)
    view.add_argument("--dst", type=Path, required=True)
    view.add_argument("--action-dim", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "checkpoint-view":
        print(json.dumps(checkpoint_view(args.src, args.dst, action_dim=args.action_dim)))
        return 0
    import tomllib

    import torch
    from examples.common.config import normalize_config

    config = normalize_config(tomllib.loads(args.config.read_text(encoding="utf-8")))
    inputs = config["input"]
    if not validate_input_config(inputs):
        raise SystemExit("this TOML uses [input].batch; there are no raw inputs to transform")
    result, record = build_batch(inputs, text_tokens=inputs.get("text_tokens", 32))
    torch.save(result, args.out)
    print(json.dumps({key: record[key] for key in ("valid_text_tokens", "text_tokens", "state_dim")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
