"""Pi0.5 model loading, safetensors handling, FP16 casting, and hooks.

Owns the `_load_and_fp16_cast` + `_cache_is_stale` + `_install_fp16_hooks` + the
`load_and_construct_pi05_policy` helper used by
``Pi05Policy.from_pretrained``.
"""
from __future__ import annotations
import glob
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import torch
import torch.nn as nn

from rpu_backend.runtime.log import _LOG
from rpu_backend.api.errors import (
    RPUBackendError,
    RPUUnsupportedDtypeError,
    UnsupportedModelError,
)


# Serialize the temporary PI05Policy.__init__ patch.
_LOAD_LOCK = threading.Lock()


PI05_BASE_PUBLIC_PROFILE = "pi0.5-base"
PI05_LIBERO_PUBLIC_PROFILE = "pi0.5-libero-v044"
_PI05_BASE_MANIFEST_SHA256 = (
    "4133ecc2e9d9ec04afdff7f251857027b20c414952653dbd5022687e40b9ec23"
)
_PI05_BASE_FILES = {
    "config.json": "367869712a2847c27e95c431ecb03f17bec4eee01a63995e2fb2d91940752b53",
    "model.safetensors": "0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f",
    "policy_postprocessor.json": "142c8b622aeac5c14631e057279df058d3005f616d909cfe30a04524de0b0891",
    "policy_preprocessor.json": "be2cd1acc33229d42be0640cb55b5d5a4bb3e8d8e27ddfcdcf3f99f1fe3beaef",
}
_PI05_BASE_REMAPPED_SHA256 = (
    "b55bac5fc2a4e52dee8170fd14a9ab0c26dc5dfa4c2b2ae3fd7b2852dda74884"
)
_PI05_BASE_CAMERA_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)
_PI05_BASE_MAX_ATTENTION_TOKENS = 60
_PI05_BASE_VOCAB_SIZE = 257_152

_PI05_LIBERO_MANIFEST_SHA256 = (
    "bf035fe76a92cb4cfa3c14803e1b265f3f2ffecbeae9b214253b0a26deed0f39"
)
_PI05_LIBERO_FILES = {
    ".gitattributes": "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    "README.md": "1be8947f4a576eef20d423ade5f651db2b0c78b4d3c154660b9a93feaaf2d13e",
    "config.json": "2f6d4b96b032593e1de65b391ee7252a70227bff398ef1b94658fea17818142f",
    "model.safetensors": "877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae",
    "model_remapped.safetensors": "63b51a0897def2f9a7d7dc7d8e6b0ee3708514aadb9d5785d7c987af69129598",
    "policy_postprocessor.json": "37d719a6584600988e6c343306d2eaee0575109f235ad500313577f73e47e8a1",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors": (
        "a002c0df7f79c5b169c5a899ad151d4ea1bed246c7d82bd93ed1556558d517a9"
    ),
    "policy_preprocessor.json": "3d669ffc18d0364536735d8203076dcc69d3a6523aa5956ccaada3f9e1e6b748",
    "policy_preprocessor_step_2_normalizer_processor.safetensors": (
        "a002c0df7f79c5b169c5a899ad151d4ea1bed246c7d82bd93ed1556558d517a9"
    ),
    "train_config.json": "199518b79faeaef0877e9a1eb5f2df6884902476a7d4491c7f865142c3fe90fd",
}
_PI05_LIBERO_CAMERA_KEYS = (
    "observation.images.image",
    "observation.images.image2",
)


def _pi05_public_profile(profile: str) -> dict[str, Any]:
    if profile == PI05_BASE_PUBLIC_PROFILE:
        return {
            "repository": "lerobot/pi05_base",
            "revision": "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba",
            "manifest_sha256": _PI05_BASE_MANIFEST_SHA256,
            "files": _PI05_BASE_FILES,
            "remapped_sha256": _PI05_BASE_REMAPPED_SHA256,
            "camera_keys": _PI05_BASE_CAMERA_KEYS,
            "state_dim": 32,
        }
    if profile == PI05_LIBERO_PUBLIC_PROFILE:
        return {
            "repository": "lerobot/pi05_libero_finetuned_v044",
            "revision": "8e174154ef5f6c60a8da12ae99c303d8963138c1",
            "manifest_sha256": _PI05_LIBERO_MANIFEST_SHA256,
            "files": _PI05_LIBERO_FILES,
            "remapped_sha256": _PI05_LIBERO_FILES["model_remapped.safetensors"],
            "camera_keys": _PI05_LIBERO_CAMERA_KEYS,
            "state_dim": 8,
        }
    raise UnsupportedModelError(
        f"unknown Pi0.5 public profile {profile!r}; expected one of "
        f"{[PI05_BASE_PUBLIC_PROFILE, PI05_LIBERO_PUBLIC_PROFILE]!r}."
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pi05_public_request(
    batch: dict,
    *,
    profile: str | None,
    entry_point: str,
    num_steps: int | None = None,
) -> dict:
    """Validate a controlled Pi0.5 request without device or model mutation."""
    if profile is None:
        return batch
    identity = _pi05_public_profile(profile)
    if not isinstance(batch, dict):
        raise TypeError(
            f"{entry_point}: admission_batch must be a dict, got "
            f"{type(batch).__name__}"
        )
    camera_keys = identity["camera_keys"]
    request_keys = frozenset({
        *camera_keys,
        "observation.state",
        "observation.language.tokens",
        "observation.language.attention_mask",
    })
    if set(batch) != request_keys:
        missing = sorted(request_keys - set(batch))
        extra = sorted(set(batch) - request_keys)
        raise ValueError(
            f"{entry_point}: {profile} request keys drift; "
            f"missing={missing}, extra={extra}"
        )

    expected = {
        **{name: ((1, 3, 256, 256), torch.float32)
           for name in camera_keys},
        "observation.state": ((1, identity["state_dim"]), torch.float32),
        "observation.language.tokens": ((1, 200), torch.int64),
        "observation.language.attention_mask": ((1, 200), torch.bool),
    }
    for name, (shape, dtype) in expected.items():
        value = batch[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{entry_point}: {name} must be a torch.Tensor")
        if value.device.type != "cpu":
            raise ValueError(f"{entry_point}: {name} must remain on CPU for admission")
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(
                f"{entry_point}: {name} must be {shape}/{dtype}, got "
                f"{tuple(value.shape)}/{value.dtype}"
            )
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{entry_point}: {name} must contain only finite values")

    for name in camera_keys:
        image = batch[name]
        if bool((image < 0).any()) or bool((image > 1).any()):
            raise ValueError(f"{entry_point}: {name} must stay in the [0, 1] range")
    state = batch["observation.state"]
    if bool((state < -1).any()) or bool((state > 1).any()):
        raise ValueError(
            f"{entry_point}: normalized observation.state must stay in [-1, 1]"
        )
    tokens = batch["observation.language.tokens"]
    if bool((tokens < 0).any()) or bool((tokens >= _PI05_BASE_VOCAB_SIZE).any()):
        raise ValueError(
            f"{entry_point}: language tokens must be in [0, "
            f"{_PI05_BASE_VOCAB_SIZE})"
        )

    mask = batch["observation.language.attention_mask"]
    valid_tokens = int(mask.sum().item())
    if valid_tokens < 1 or valid_tokens > _PI05_BASE_MAX_ATTENTION_TOKENS:
        raise ValueError(
            f"{entry_point}: {profile} supports 1.."
            f"{_PI05_BASE_MAX_ATTENTION_TOKENS} attention tokens, got "
            f"{valid_tokens}; rejecting before model/Graph mutation."
        )
    expected_mask = torch.arange(mask.shape[1], device="cpu").unsqueeze(0) < valid_tokens
    if not torch.equal(mask, expected_mask):
        raise ValueError(
            f"{entry_point}: attention_mask must be one contiguous prefix "
            "followed by padding."
        )
    if num_steps is not None and num_steps != 10:
        raise ValueError(
            f"{entry_point}: {profile} requires num_steps=10, "
            f"got {num_steps}."
        )
    return batch


def _preflight_pi05_public_profile(
    model_path: str | os.PathLike[str],
    *,
    profile: str | None,
    admission_batch: dict | None,
) -> dict[str, Any] | None:
    """Bind exact public bytes, config, and request before weight loading."""
    if profile is None:
        if admission_batch is not None:
            raise ValueError("Pi05Policy admission_batch requires an explicit profile")
        return None
    identity = _pi05_public_profile(profile)
    if admission_batch is None:
        raise ValueError(
            f"Pi05Policy.from_pretrained: profile={profile!r} requires "
            "admission_batch so the request envelope is checked before weights load."
        )
    _validate_pi05_public_request(
        admission_batch,
        profile=profile,
        entry_point="Pi05Policy.from_pretrained",
        num_steps=10,
    )

    root = Path(model_path)
    for filename, expected_sha256 in identity["files"].items():
        path = root / filename
        if not path.is_file():
            raise UnsupportedModelError(
                f"Pi05Policy.from_pretrained: exact profile asset drift: {filename}"
            )
        if _sha256_file(path) != expected_sha256:
            raise UnsupportedModelError(
                f"Pi05Policy.from_pretrained: exact profile asset digest drift: {filename}"
            )
    source_names = sorted(
        path.name for path in root.glob("model*.safetensors")
        if path.name != "model_remapped.safetensors"
    )
    if source_names != ["model.safetensors"] or (root / "rpu_quant_config.json").exists():
        raise UnsupportedModelError(
            "Pi05Policy.from_pretrained: exact FP16 profile rejects additional "
            "source safetensors or quantization metadata."
        )
    remapped = root / "model_remapped.safetensors"
    if remapped.exists() and (
        not remapped.is_file()
        or _sha256_file(remapped) != identity["remapped_sha256"]
    ):
        raise UnsupportedModelError(
            "Pi05Policy.from_pretrained: generated public remap digest drift."
        )
    return {
        "profile": profile,
        "repository": identity["repository"],
        "revision": identity["revision"],
        "manifest_sha256": identity["manifest_sha256"],
        "maximum_attention_tokens": _PI05_BASE_MAX_ATTENTION_TOKENS,
    }


_PI05_W8A16_TARGETS = {
    "expert": ".paligemma_with_expert.gemma_expert.model.layers.",
    "vlm": ".paligemma_with_expert.paligemma.model.language_model.layers.",
}
_PI05_PROJ_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _is_pi05_w8a16_proj_weight(name: str, anchor: str) -> bool:
    return anchor in name and name.endswith(_PI05_PROJ_SUFFIXES)


def _load_pi05_rpu_quant_config(model_path: str) -> dict:
    path = os.path.join(model_path, "rpu_quant_config.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RPUBackendError(
            f"_load_and_fp16_cast: failed to read {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(cfg, dict):
        raise RPUBackendError(
            f"_load_and_fp16_cast: {path} must contain a JSON object"
        )

    method = cfg.get("method")
    if method == "w4a16_fake_int8":
        if cfg.get("storage") != "int8" or int(cfg.get("value_bits", 0)) != 4:
            raise RPUBackendError(
                "_load_and_fp16_cast: method=w4a16_fake_int8 requires "
                "storage=int8 and value_bits=4"
            )
        _LOG.info(
            "Pi0.5 fake-W4 checkpoint detected: "
            "method=w4a16_fake_int8, storage=int8, value_bits=4, kernel=w8a16"
        )
    elif method == "w4a16":
        if cfg.get("storage") != "int8" or int(cfg.get("value_bits", 0)) != 4:
            raise RPUBackendError(
                "_load_and_fp16_cast: method=w4a16 requires storage=int8 "
                "(int4 values stored in int8 on disk) and value_bits=4"
            )
        group_size = int(cfg.get("group_size", 0))
        group_wise = (
            group_size != 0
            or cfg.get("mode") == "group_wise_symmetric"
            or cfg.get("scale") == "per_group_GxN"
        )
        if group_wise and not (
            group_size in (32, 64, 128)
            and cfg.get("mode") == "group_wise_symmetric"
            and cfg.get("scale") == "per_group_GxN"
        ):
            raise RPUBackendError(
                "_load_and_fp16_cast: group-wise method=w4a16 requires "
                "group_size in {32,64,128}, mode=group_wise_symmetric, "
                "and scale=per_group_GxN"
            )
        mixed_int8 = {
            (suffix[: -len(".weight")] if suffix.endswith(".weight") else suffix)
            .rsplit(".", 1)[-1]
            for suffix in cfg.get("mixed_int8_suffixes", ())
            if isinstance(suffix, str)
        }
        if group_wise and not {"k_proj", "v_proj"}.issubset(mixed_int8):
            raise RPUBackendError(
                "_load_and_fp16_cast: group-wise method=w4a16 requires "
                "mixed_int8_suffixes to include k_proj and v_proj"
            )
        _LOG.info(
            "Pi0.5 mixed W4A16-KV8 checkpoint detected: method=w4a16, "
            "storage=int8, value_bits=4, kernel=wINT4a16_pgrp, group_size=%s; "
            "activations and KV cache remain FP16",
            group_size or "legacy-per-channel",
        )
    return cfg


def _pop_pi05_w8a16_tensors(
    policy, state_dict: dict[str, torch.Tensor], quant_config: dict | None = None
):
    quant_config = quant_config or {}
    group_size = int(quant_config.get("group_size", 0))
    mixed_int8 = {
        (suffix[: -len(".weight")] if suffix.endswith(".weight") else suffix)
        .rsplit(".", 1)[-1]
        for suffix in quant_config.get("mixed_int8_suffixes", ())
    }
    modules = dict(policy.named_modules())
    consumed = {}
    scale_names = [name for name in state_dict if name.endswith(".weight_scale")]
    for target, anchor in _PI05_W8A16_TARGETS.items():
        expected = sorted(
            f"{name}.weight" for name, module in modules.items()
            if isinstance(module, nn.Linear) and _is_pi05_w8a16_proj_weight(
                f"{name}.weight", anchor
            )
        )
        if not expected:
            continue

        int8_names = [
            name for name in expected
            if name in state_dict and state_dict[name].dtype == torch.int8
        ]
        target_scale_names = [
            name for name in scale_names
            if anchor in name and any(
                name.endswith(suffix.replace(".weight", ".weight_scale"))
                for suffix in _PI05_PROJ_SUFFIXES
            )
        ]
        if not int8_names:
            if target_scale_names:
                raise RPUBackendError(
                    "_load_and_fp16_cast: found Pi05 "
                    f"{target} weight_scale tensors without int8 {target} weights"
                )
            continue
        if len(int8_names) != len(expected):
            raise RPUBackendError(
                "_load_and_fp16_cast: partial Pi05 W8A16 "
                f"{target} checkpoint int8={len(int8_names)}/{len(expected)}"
            )

        for weight_name in expected:
            scale_name = weight_name[: -len(".weight")] + ".weight_scale"
            if scale_name not in state_dict:
                raise RPUBackendError(
                    f"_load_and_fp16_cast: missing {scale_name} for int8 {weight_name}"
                )
            weight = state_dict.pop(weight_name)
            scale = state_dict.pop(scale_name)
            group_w4 = (
                quant_config.get("method") == "w4a16"
                and group_size > 0
                and weight_name[: -len(".weight")].rsplit(".", 1)[-1]
                not in mixed_int8
            )
            if group_w4:
                if weight.size(1) % group_size:
                    raise RPUBackendError(
                        f"_load_and_fp16_cast: {weight_name}.shape[1]={weight.size(1)} "
                        f"is not divisible by group_size={group_size}"
                    )
                expected_scale_shape = (weight.size(1) // group_size, weight.size(0))
                scale_ok = (
                    scale.dtype == torch.float16
                    and tuple(scale.shape) == expected_scale_shape
                )
                scale_desc = f"2D fp16 [K/group_size,N]={expected_scale_shape}"
            else:
                scale_ok = (
                    scale.dtype == torch.float16
                    and scale.dim() == 1
                    and scale.numel() == weight.size(0)
                )
                scale_desc = f"1D fp16 [N]=[{weight.size(0)}]"
            if not scale_ok:
                raise RPUBackendError(
                    f"_load_and_fp16_cast: {scale_name} must be {scale_desc}, got "
                    f"dtype={scale.dtype}, shape={tuple(scale.shape)}"
                )
            module_name = weight_name[: -len(".weight")]
            consumed[module_name] = (weight.contiguous(), scale.contiguous())
    return consumed


def _install_pi05_w8a16_tensors(policy, consumed) -> None:
    if not consumed:
        return
    modules = dict(policy.named_modules())
    for module_name, (weight, scale) in consumed.items():
        module = modules.get(module_name)
        if module is None or not isinstance(module, nn.Linear):
            raise RPUBackendError(
                f"_load_and_fp16_cast: cannot install W8A16 tensor for {module_name}"
            )
        module.weight = nn.Parameter(weight, requires_grad=False)
        if "weight_scale" in module._buffers:
            module._buffers["weight_scale"] = scale
        else:
            module.register_buffer("weight_scale", scale)


def _pi05_consumed_keys(consumed) -> set[str]:
    keys: set[str] = set()
    for module_name in consumed:
        keys.add(f"{module_name}.weight")
        keys.add(f"{module_name}.weight_scale")
    return keys


# =============================================================================
# Stale-cache check.
# =============================================================================
def _cache_is_stale(source_paths: list, cache_path: str) -> bool:
    """True if the cached `cache_path` is stale vs any of `source_paths`.

    The check:
      - Uses `os.stat(path).st_mtime_ns` and `st_ctime_ns` for nanosecond
        precision.
      - Caller must EXCLUDE the cache file itself from source_paths (avoid
        self-matching).

    An obsolete size-based heuristic (doubled-size check) was removed because
    it only fired on extreme growth, missing normal-range size changes (e.g.
    a single added parameter in a shard, a weight-renaming that shrinks the
    file, etc.). A source replacement that preserves or rolls back mtime still
    changes ctime, so either timestamp being newer invalidates the cache.
    Self-exclusion of the cache file itself prevents the cache from matching
    itself as a source.

    """
    if not os.path.exists(cache_path):
        return True
    cache_stat = os.stat(cache_path)
    cache_mtime_ns = cache_stat.st_mtime_ns
    cache_ctime_ns = cache_stat.st_ctime_ns
    for src in source_paths:
        if not os.path.exists(src):
            continue
        source_stat = os.stat(src)
        # ``ctime_ns`` catches content replacement with a preserved mtime.
        # Callers exclude `model_remapped.safetensors` itself from source_paths
        # (enforced at call-site in _load_and_fp16_cast).
        if (
            source_stat.st_mtime_ns > cache_mtime_ns
            or source_stat.st_ctime_ns > cache_ctime_ns
        ):
            return True
    return False


# =============================================================================
# Load, cast, and validate a Pi0.5 policy.
# =============================================================================
def _load_and_fp16_cast(
    model_path: str,
    dtype: "torch.dtype",
    *,
    expected_remap_sha256: str | None = None,
    **lerobot_kwargs: Any,
) -> Any:
    """Load a LeRobot PI05Policy and cast it to FP16.

    ``_LOAD_LOCK`` protects the constructor patch. Cache freshness uses
    nanosecond timestamps and self-exclusion. Integrity checks reject
    empty sources, meta tensors, and missing or unexpected critical keys.
    """
    if dtype is not torch.float16:
        raise RPUUnsupportedDtypeError(
            f"_load_and_fp16_cast: dtype must be torch.float16, got {dtype}."
        )

    # Probe for LeRobot first so the actionable ImportError fires
    # before any "Loading..." line is printed, otherwise the UX implies load
    # before any loading message.
    try:
        import lerobot  # noqa: F401 — probe only
    except ImportError as e:
        raise ImportError(
            "Pi0.5 requires lerobot. Install with: pip install lerobot\n"
            "See docs/api_reference.md#policy-apis for Pi0.5 setup."
        ) from e

    _LOG.info("Loading Pi0.5 model from: %s (dtype=fp16)", model_path)
    t0 = time.time()

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy as LeRobotPI05
    from lerobot.configs.policies import PreTrainedConfig
    from safetensors.torch import load_file

    # Pre-load configuration normalization.
    config = PreTrainedConfig.from_pretrained(model_path, **lerobot_kwargs)
    config.device = "cpu"
    config.dtype = "float32"

    # Require at least one safetensors shard before loading.
    all_safetensors = glob.glob(os.path.join(model_path, "*.safetensors"))
    if not all_safetensors:
        raise RPUBackendError(
            f"_load_and_fp16_cast: no *.safetensors files found in {model_path}")
    quant_config = _load_pi05_rpu_quant_config(model_path)

    remapped_path = os.path.join(model_path, "model_remapped.safetensors")

    # Exclude the generated remap from its own source list.
    source_safetensors = [s for s in all_safetensors
                           if os.path.basename(s) != "model_remapped.safetensors"]

    if _cache_is_stale(source_safetensors, remapped_path):
        # Weight remapping is owned by the weights module.
        from rpu_backend.adapters.pi05.weights import _remap_and_save
        _remap_and_save(model_path, remapped_path)
    if (
        expected_remap_sha256 is not None
        and _sha256_file(Path(remapped_path)) != expected_remap_sha256
    ):
        raise UnsupportedModelError(
            "_load_and_fp16_cast: generated public remap digest drift."
        )

    # The PI05Policy.__init__ monkey-patch is process-global, so guard it
    # with `_LOAD_LOCK`.
    _orig_init = LeRobotPI05.__init__

    def _patched_init(self, cfg, **kw):
        saved_device = cfg.device
        cfg.device = None
        cfg.compile_model = False
        _orig_init(self, cfg, **kw)
        cfg.device = saved_device

    with _LOAD_LOCK:
        LeRobotPI05.__init__ = _patched_init
        try:
            with torch.device("meta"):
                policy = LeRobotPI05(config)
        finally:
            LeRobotPI05.__init__ = _orig_init

    # Load weights via safetensors.
    state_dict = load_file(remapped_path)
    w8a16_consumed = _pop_pi05_w8a16_tensors(policy, state_dict, quant_config)
    _consumed_method = quant_config.get("method")
    if _consumed_method in ("w4a16_fake_int8", "w4a16") and not w8a16_consumed:
        raise RPUBackendError(
            f"_load_and_fp16_cast: method={_consumed_method} declared but no "
            "Pi05 int8 projection tensors were consumed"
        )
    consumed_keys = _pi05_consumed_keys(w8a16_consumed)
    state_dict_keys = set(state_dict.keys())

    # Materialize meta buffers not present in state_dict.
    for name, buf in list(policy.named_buffers()):
        if buf.device.type != 'meta':
            continue
        if name in state_dict_keys:
            continue
        parts = name.split('.')
        mod = policy
        for p in parts[:-1]:
            mod = getattr(mod, p)
        buf_name = parts[-1]
        if buf_name == 'inv_freq':
            dim = buf.shape[0]
            base = 10000.0
            for cfg_attr in [mod, getattr(mod, 'config', None)]:
                if cfg_attr and hasattr(cfg_attr, 'rope_theta'):
                    base = cfg_attr.rope_theta
                    break
            new_buf = 1.0 / (base ** (torch.arange(0, dim * 2, 2, dtype=torch.float32) / (dim * 2)))
            mod.register_buffer(buf_name, new_buf, persistent=False)
        elif buf_name == 'position_ids':
            new_buf = torch.arange(buf.shape[-1]).unsqueeze(0)
            if len(buf.shape) > 2:
                new_buf = new_buf.expand(buf.shape)
            mod.register_buffer(buf_name, new_buf, persistent=False)
        elif buf_name == 'embed_scale' and hasattr(mod, 'scalar_embed_scale'):
            new_buf = torch.tensor(mod.scalar_embed_scale, dtype=buf.dtype)
            mod.register_buffer(buf_name, new_buf, persistent=False)
        else:
            buf_dtype = buf.dtype if buf.dtype != torch.float32 else torch.float32
            mod.register_buffer(buf_name, torch.zeros(buf.shape, dtype=buf_dtype))

    load_result = policy.load_state_dict(state_dict, assign=True, strict=False)

    # Validate critical checkpoint keys after loading.
    missing = list(getattr(load_result, 'missing_keys', []))
    unexpected = list(getattr(load_result, 'unexpected_keys', []))
    expected_param_names = {n for n, _ in policy.named_parameters()}
    critical_missing = [
        k for k in missing
        if k in expected_param_names and k not in consumed_keys
    ]
    if critical_missing:
        raise RPUBackendError(
            f"_load_and_fp16_cast: missing {len(critical_missing)} expected "
            f"parameter keys in state_dict (first 5): {critical_missing[:5]}")
    critical_unexpected = []
    for k in unexpected:
        if k not in state_dict:
            continue
        t = state_dict[k]
        if t.dim() == 2 and min(t.shape) >= 32:
            critical_unexpected.append(k)
    if critical_unexpected:
        raise RPUBackendError(
            f"_load_and_fp16_cast: unexpected critical Linear-sized tensors in "
            f"state_dict not consumed by model (first 5): {critical_unexpected[:5]}")

    # Apply the post-load FP16 cast and hooks.
    policy = policy.half()
    _install_pi05_w8a16_tensors(policy, w8a16_consumed)
    _install_fp16_hooks(policy)

    meta_params = [n for n, p in policy.named_parameters() if p.device.type == 'meta']
    meta_buffers = [n for n, b in policy.named_buffers() if b.device.type == 'meta']
    if meta_params or meta_buffers:
        raise RPUBackendError(
            f"_load_and_fp16_cast: found meta tensors post-load - "
            f"params={meta_params[:3]} buffers={meta_buffers[:3]}")

    if hasattr(config, 'compile_model'):
        config.compile_model = False

    policy = policy.eval()
    _method = quant_config.get("method")
    if w8a16_consumed and _method in ("w4a16_fake_int8", "w4a16"):
        _k = "kernel=w8a16" if _method == "w4a16_fake_int8" else "runtime-packed kernel=wINT4a16"
        suffix = f" + W4-values-in-int8 {_method} {len(w8a16_consumed)} linears ({_k})"
    elif w8a16_consumed:
        suffix = f" + W8A16 {len(w8a16_consumed)} linears"
    else:
        suffix = ""
    policy._pi05_quant_config = dict(quant_config)
    _LOG.info("Pi0.5 model loaded in %.1fs (fp16%s + hooks installed)",
              time.time() - t0, suffix)
    return policy


# =============================================================================
# FP16 hooks. The post-hook belongs on every module and casts FP32 outputs to
# FP16; the pre-hook remains limited to Linear and Conv2d modules.
# =============================================================================
def _install_fp16_hooks(policy) -> None:
    """Install dtype hooks while preserving the policy's FP16 boundary."""
    def _input_dtype_hook(module, args):
        w = module.weight
        if not w.is_floating_point():
            raise RPUBackendError(
                "Pi0.5 W8A16 Linear CPU forward is unsupported; "
                "call Pi05Policy.to('rpu') before inference."
            )
        new_args = []
        for a in args:
            if isinstance(a, torch.Tensor) and a.is_floating_point() and a.dtype != w.dtype:
                new_args.append(a.to(w.dtype))
            else:
                new_args.append(a)
        return tuple(new_args)

    def _to_fp16(x):
        if isinstance(x, torch.Tensor) and x.is_floating_point() and x.dtype == torch.float32:
            return x.half()
        return x

    def _output_dtype_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            return _to_fp16(output)
        elif isinstance(output, tuple):
            return tuple(_to_fp16(o) for o in output)
        return output

    for m in policy.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            m.register_forward_pre_hook(_input_dtype_hook)
        # The post-hook is unconditional and applies to every module.
        m.register_forward_hook(_output_dtype_hook)


# =============================================================================
# Public-policy construction helper.
# =============================================================================
def load_and_construct_pi05_policy(
    pretrained_name_or_path: str,
    *,
    dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = False,
    rpu_execution=None,
    profile: str | None = None,
    admission_batch: dict | None = None,
    **lerobot_kwargs: Any,
) -> "Any":
    """Load and construct the policy used by ``Pi05Policy.from_pretrained``.

    The FP16 guard runs before any LeRobot load. Custom remote model code is
    not supported.
    """
    if dtype is not torch.float16:
        raise RPUUnsupportedDtypeError(
            f"Pi05Policy requires dtype=torch.float16; got {dtype}.")
    if not isinstance(trust_remote_code, bool):
        raise TypeError(
            "Pi05Policy.from_pretrained: trust_remote_code must be bool, "
            f"got {trust_remote_code!r}"
        )
    if "trust_remote_code" in lerobot_kwargs:
        raise TypeError(
            "Pi05Policy.from_pretrained() got duplicate keyword argument "
            "'trust_remote_code' (pass it ONCE as the explicit kwarg).")
    if trust_remote_code is not False:
        raise ValueError(
            "Pi05Policy.from_pretrained requires trust_remote_code=False; "
            "custom model code is not supported."
        )
    admitted_profile = _preflight_pi05_public_profile(
        pretrained_name_or_path,
        profile=profile,
        admission_batch=admission_batch,
    )
    lerobot_kwargs["trust_remote_code"] = trust_remote_code
    lerobot_policy = _load_and_fp16_cast(
        pretrained_name_or_path,
        dtype,
        expected_remap_sha256=(
            _pi05_public_profile(admitted_profile["profile"])["remapped_sha256"]
            if admitted_profile is not None else None
        ),
        **lerobot_kwargs,
    )
    # Late import keeps the loader and policy modules circular-free.
    from rpu_backend.api.policy import Pi05Policy
    if rpu_execution is None:
        policy = Pi05Policy.from_lerobot_policy(lerobot_policy)
    else:
        policy = Pi05Policy.from_lerobot_policy(
            lerobot_policy, rpu_execution=rpu_execution
        )
    if admitted_profile is not None:
        policy._rpu_public_profile = admitted_profile["profile"]
        policy._rpu_public_profile_identity = dict(admitted_profile)
    return policy
