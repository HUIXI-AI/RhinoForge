"""Cold public LIBERO execution profiles; no developer asset paths or global defaults."""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping
from numbers import Integral
import os
import threading

_PROFILE_LOCK = threading.RLock()
_PRECISION_MAP = {
    "fp16": "fp16", "w8a16": "w8a16",
    "w8_action_nvfp4": "w4a16", "w8_prefill_a8_action_nvfp4": "w4a16",
}

_PRECISIONS = ("fp16", "w8a16", "w4a16", "int4_g128", "legacy_int4")

_ON = """
RING_XOR3 K_ROPE_INSERT KVINSERT_PAD16 DENOISE_UNROLL PREFILL_MASK_FP16 PREFILL_KV_ONLY
DENOISE_XOR3_GATED_MLP VISION_MERGE_SEGMENT VISION_Q_RESIDENT PREFILL_CARRY_AB
PREFILL_MASK_RESIDENT PREFILL_DOWN_PAIR PREFILL_OWNER_NORM PREFILL_O_PAIR
DENOISE_XOR3_GATED_ATTN VISION_KV_SPM_ONLY DENOISE_KV1_DIRECT_CACHE
DENOISE_KV1_PAIR_OWNER PREFILL_KV1_PAIR_DIRECT PREFILL_GATEUP_PAIR FUSED_DENOISE SIGLIP_BATCH
""".split()

_GENERIC_ON = frozenset("""
KVINSERT_PAD16 DENOISE_UNROLL PREFILL_MASK_FP16 FUSED_DENOISE SIGLIP_BATCH
""".split())

_OFF = """
GEGLU_PREFETCH SDPA_SINGLE_GROUP VISION_FC_WEIGHT_OUTER PREFILL_CARRY_A
PREFILL_ROPE_CACHE DENOISE_GATEUP_RESIDENT DENOISE_ELIDE_UNUSED_DMA
DENOISE_SPM_ACTION_STAGING DENOISE_Q_ROPE_EPILOGUE PREFILL_O_CHAIN
PREFILL_QKV_PAIR DENOISE_ADARMS_BODY_STREAM DENOISE_GATEUP_RESIDENT_LAYERS
DENOISE_KV1_GATHER DENOISE_NVFP4_GEGLU_ACC16_M50
""".split()

def profile_environment(
    precision: str, *, prefill_w8a8: bool | None = None,
    vision_owner_ln: bool | None = None, cameras: int = 3, text_tokens: int = 32,
) -> dict[str, str]:
    """Return cold settings; explicit opt-ins default off in the CLI.

    None retains compatibility with existing one-argument experiment callers
    that bind these two flags separately. It never enables either flag.
    """
    if precision not in _PRECISIONS:
        raise ValueError("precision must be fp16, w8a16, w4a16 (NVFP4), int4_g128, or legacy_int4")
    if type(cameras) is not int or cameras not in (2, 3):
        raise ValueError("cameras must be 2 or 3")
    _validate_text_profile(text_tokens, cameras=cameras, precision=precision)
    for name, value in (("prefill_w8a8", prefill_w8a8),
                        ("vision_owner_ln", vision_owner_ln)):
        if value is not None and type(value) is not bool:
            raise TypeError(f"{name} must be bool or None")
    if prefill_w8a8 and precision != "w4a16":
        raise ValueError("Prefill W8A8 is admitted only with Action NVFP4 (--precision w4a16)")
    result = {"RPU_PI05_" + name: str(int(cameras == 3 or precision in ("fp16", "w8a16", "w4a16") or name in _GENERIC_ON))
              for name in _ON}
    result.update({"RPU_PI05_" + name: "0" for name in _OFF})
    # Pin the two on-load quantizers so a nominal FP16 run cannot inherit W8.
    for name in ("SIGLIP_W8A16", "ADARMS_DENSE_W8A16"):
        result["RPU_PI05_" + name] = str(int(precision != "fp16"))
    result["RPU_PI05_SIGLIP_W8A16_SCOPE"] = "all"
    result["RPU_PI05_DENOISE_GEGLU_M50"] = str(int(precision in ("fp16", "w8a16")))
    for name in ("DENOISE_W4_FASTPATH", "DENOISE_W4_GEGLU_M50", "DENOISE_W4_GEGLU_N64"):
        result["RPU_PI05_" + name] = str(int(precision in ("int4_g128", "legacy_int4") and cameras == 3))
    result["RPU_PI05_DENOISE_NVFP4_GEGLU_M50"] = "0"
    if precision == "w4a16":
        # Share W8 K/V and FP16 continuations, with an explicit NVFP4
        # storage policy and its separate NVFP4 GeGLU producer. The action
        # owner's cold linear_acc32 setting selects accumulation precision.
        result["RPU_PI05_DENOISE_W4_FASTPATH"] = "1"
    for name, value in (("PREFILL_GATEUP_W8A8", prefill_w8a8),
                        ("VISION_OWNER_LN", vision_owner_ln)):
        if value is not None or precision == "fp16" or cameras == 2:
            result["RPU_PI05_" + name] = str(int(bool(value)))
    return result

def _validate_text_profile(text_tokens: int, *, cameras: int, precision: str | None = None) -> None:
    if type(text_tokens) is not int or text_tokens not in (32, 64, 96, 128):
        raise ValueError("text_tokens must be 32, 64, 96, or 128")
    if text_tokens != 32:
        if cameras not in (2, 3):
            raise ValueError(f"T{text_tokens} requires exactly two or three cameras")
        if precision is not None and precision not in ("fp16", "w8a16", "w4a16"):
            raise ValueError(f"T{text_tokens} requires FP16, W8A16, or Action NVFP4 precision")

def validate_checkpoint(config: dict, precision: str) -> None:
    if precision == "fp16":
        if config:
            raise ValueError("requested fp16 requires an unquantized checkpoint without rpu_quant_config")
        return
    method = {"w8a16": "w8a16", "w4a16": "nvfp4a16",
              "int4_g128": "w4a16", "legacy_int4": "w4a16"}[precision]
    if config.get("method") != method:
        raise ValueError(f"requested {precision} requires method={method}, checkpoint method={config.get('method')!r}")
    if precision != "w8a16":
        keep = {name.removesuffix(".weight").rsplit(".", 1)[-1]
                for name in config.get("mixed_int8_suffixes", [])}
        if config.get("int4_components") != ["expert"] or keep != {"k_proj", "v_proj"}:
            raise ValueError("optimized W4 requires Action-only Q/O/Gate/Up/Down; K/V and VLM W8")
        if precision == "w4a16":
            if config.get("nvfp4_abi") != "striped_v2" or config.get("nvfp4_block_size") != 16:
                raise ValueError("w4a16 defaults to NVFP4 striped_v2 block16, not legacy INT4")
        elif (config.get("int4_group_size") != 128 or
              (config.get("mode") == "group_symmetric") != (precision == "int4_g128")):
            raise ValueError("INT4 group metadata does not match requested precision")

def validate_camera_batch(batch: dict, image_features, *, cameras: int, text_tokens: int = 32) -> None:
    """Require the selected physical token width and configured image slots."""
    import torch

    if type(cameras) is not int or cameras not in (2, 3):
        raise ValueError("cameras must be 2 or 3")
    _validate_text_profile(text_tokens, cameras=cameras)
    configured = set(image_features or ())
    present = {key for key in batch if key.startswith("observation.images.")
               and not key.endswith("_padding_mask")}
    label = "three" if cameras == 3 else "two"
    if len(configured) != cameras or present != configured:
        raise ValueError(f"this optimized entry requires all {label} configured cameras and no additional image slots")
    if cameras == 2 and any(key.rsplit(".", 1)[-1].startswith("empty_camera_") for key in configured):
        raise ValueError("two-camera input requires two real VISUAL features, without empty_camera slots")
    orphan_masks = {key.removesuffix("_padding_mask") for key in batch
                    if key.startswith("observation.images.") and key.endswith("_padding_mask")} - configured
    if orphan_masks:
        raise ValueError("image padding masks must refer to configured cameras")
    for key in configured:
        image = batch[key]
        if not torch.is_tensor(image) or image.ndim != 4 or tuple(image.shape[:2]) != (1, 3):
            raise ValueError(f"{key} must be a batch-one [1,3,H,W] image tensor")
        mask = batch.get(key + "_padding_mask")
        if mask is not None and (not torch.is_tensor(mask) or mask.dtype != torch.bool
                                 or tuple(mask.shape) != (1,) or not bool(mask.all())):
            raise ValueError(f"all {label} cameras must have present (true) padding masks")
    tokens = batch.get("observation.language.tokens")
    if not torch.is_tensor(tokens) or tuple(tokens.shape) != (1, text_tokens):
        raise ValueError(f"this optimized entry requires exactly {text_tokens} language tokens")
    if text_tokens != 32:
        mask = batch.get("observation.language.attention_mask")
        if (not torch.is_tensor(mask) or mask.dtype != torch.bool or tuple(mask.shape) != (1, text_tokens)
                or not bool(mask.any()) or bool((~mask[:, :-1] & mask[:, 1:]).any())):
            raise ValueError(f"T{text_tokens} requires a nonempty boolean [1,{text_tokens}] language mask with right padding")


def normalize_profile(profile):
    if profile is None:
        return None
    if not isinstance(profile, Mapping) or set(profile) - {"precision", "num_cameras", "text_tokens"}:
        raise ValueError("optimized_profile accepts precision, num_cameras, and text_tokens")
    precision = profile.get("precision")
    cameras = profile.get("num_cameras")
    tokens = profile.get("text_tokens", 32)
    if precision not in _PRECISION_MAP:
        raise ValueError(f"optimized_profile.precision must be one of {tuple(_PRECISION_MAP)}")
    if type(cameras) is not int or cameras not in (2, 3):
        raise ValueError("optimized_profile.num_cameras must be 2 or 3")
    _validate_text_profile(tokens, cameras=cameras)
    return dict(precision=precision, num_cameras=cameras, text_tokens=tokens)


def execution_for_profile(profile, execution):
    if profile is None:
        return execution
    import copy
    result = copy.deepcopy(dict(execution or {}))
    # This helper also validates examples without importing the backend.
    cores = result.get("model", {}).get("num_cores", 8)
    if isinstance(cores, bool) or not isinstance(cores, Integral) or cores != 8:
        raise ValueError("optimized Pi0.5 paired profiles require model.num_cores=8")
    expected = (256 * profile["num_cameras"] + profile["text_tokens"]) // 2
    # Asset checks also call this after the policy freezes its configuration.
    # A frozen mapping deliberately survives deepcopy; detach the branch that
    # receives the profile's derived chunk size without mutating the owner.
    prefill = dict(result.get("prefill", {}))
    result["prefill"] = prefill
    if prefill.get("chunk_size", "auto") not in ("auto", expected):
        raise ValueError(f"optimized Pi0.5 requires prefill.chunk_size={expected}")
    if prefill.get("padding_rows", "auto") not in ("auto", 0) or prefill.get("padding_budget", 0) != 0:
        raise ValueError("optimized Pi0.5 paired prefill does not admit extra padding")
    prefill["chunk_size"] = expected
    return result


@contextmanager
def profile_environment_scope(profile, execution=None):
    if profile is None:
        yield
        return
    desired = profile_environment(
        _PRECISION_MAP[profile["precision"]],
        cameras=profile["num_cameras"], text_tokens=profile["text_tokens"],
        prefill_w8a8=profile["precision"] == "w8_prefill_a8_action_nvfp4",
        vision_owner_ln=True,
    )
    from rpu_backend.api._execution import resolve_pi05_execution_components
    _, _, _, action = resolve_pi05_execution_components(
        execution_for_profile(profile, execution), entry_point="optimized Pi0.5 environment")
    desired["RPU_PI05_DENOISE_NVFP4_GEGLU_ACC16_M50"] = str(int(
        "nvfp4" in profile["precision"] and profile["text_tokens"] == 32
        and not action.get("action", {}).get("linear_acc32", False)))
    # These internal cold flags bridge the existing native constructor ABI.
    # The public profile owns them only for the operation and restores the
    # process environment even when loading/installation/inference raises.
    with _PROFILE_LOCK:
        previous = {key: os.environ.get(key) for key in desired}
        conflicts = [key for key, value in desired.items()
                     if previous[key] is not None and previous[key] != value]
        if conflicts:
            raise ValueError("environment conflicts with optimized Pi0.5 profile: " + ", ".join(conflicts))
        try:
            os.environ.update(desired)
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def bind_profile(policy, profile):
    if profile is None:
        return
    adapter, raw = policy._adapter, policy._lerobot_policy
    if getattr(adapter, "_rpu_is_ready", False) or getattr(raw, "_rpu_swizzle_started", False):
        raise ValueError("optimized_profile is cold-only; load a fresh policy")
    config = raw.config
    if config is not raw.model.config:
        raise ValueError("optimized Pi0.5 requires one shared policy/model config")
    validate_checkpoint(getattr(raw, "_pi05_quant_config", {}) or {}, _PRECISION_MAP[profile["precision"]])
    if (config.chunk_size != 50 or config.max_action_dim != 32 or config.num_inference_steps != 10):
        raise ValueError("optimized Pi0.5 requires horizon 50, max_action_dim 32, and 10 denoise steps")
    features = dict(config.image_features)
    if profile["num_cameras"] == 2:
        selected = {key: feature for key, feature in features.items()
                    if not key.rsplit(".", 1)[-1].startswith("empty_camera_")}
        if len(selected) != 2:
            raise ValueError("two-camera profile requires exactly two configured real VISUAL fields")
        config.input_features = {key: feature for key, feature in config.input_features.items()
                                 if key not in features or key in selected}
        config.empty_cameras = 0
    if len(config.image_features) != profile["num_cameras"]:
        raise ValueError("configured camera count does not match optimized_profile")
    adapter._rpu_prefill_text_tokens = profile["text_tokens"]
    # The adapter and execution session already own the same constructor-time
    # PAD16 snapshot. All public buckets are aligned; preserve that snapshot.
    raw.model._pi05_prefill_mask_fp16 = True
    policy._optimized_profile = profile


def validate_profile_batch(policy, batch, num_steps=None):
    profile = getattr(policy, "_optimized_profile", None)
    if profile is None:
        return
    if num_steps not in (None, 10):
        raise ValueError("optimized Pi0.5 profile requires 10 denoise steps")
    validate_camera_batch(batch, policy._lerobot_policy.config.image_features,
                         cameras=profile["num_cameras"], text_tokens=profile["text_tokens"])
    import torch
    tokens = batch["observation.language.tokens"]
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("Pi0.5 language tokens must be int32 or int64")
    # Longer buckets already checked this request's mask in validate_camera_batch.
    if profile["text_tokens"] == 32:
        mask = batch.get("observation.language.attention_mask")
        if (not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.shape != tokens.shape
                or not bool(mask.any()) or bool((~mask[:, :-1] & mask[:, 1:]).any())):
            raise ValueError("Pi0.5 requires a nonempty boolean language mask with right padding")


def required_kernel_names(profile, execution=None):
    from rpu_backend.api._execution import resolve_pi05_execution_components
    _, _, _, action = resolve_pi05_execution_components(
        execution_for_profile(profile, execution), entry_point="optimized Pi0.5 assets")
    rows = (256 * profile["num_cameras"] + profile["text_tokens"]) // 2
    vision = 256 * profile["num_cameras"]
    precision = "fp16" if profile["precision"] == "fp16" else "w8a16"
    tile = 96 if rows == 448 else 112 if rows == 432 else 128 if rows == 416 else 160
    o_tile = tile if rows in (416, 432, 448) else 128
    names = [
        f"pi05_all_reduce_residual_xor3_m{vision}n1152",
        f"pi05_all_reduce_residual_xor3_m{rows}n2048",
        "pi05_all_reduce_residual_xor3_m50n1024",
        "pi05_k_rope_insert_m50d256p64",
        f"pi05_xor3_compact_residual_raw_full_m{rows}n2048",
        f"pi05_prefill_o_weight_outer_{precision}_c{rows}x2_m{o_tile}n80k128",
        f"pi05_down_weight_outer_{precision}_c{rows}x2_m{tile}n80k128",
        f"pi05_prefill_kv1_pair_owner_{precision}_m{rows}n32x2k2048",
        f"pi05_vision_owner_reduce_layernorm_m{vision}n1152",
        f"pi05_vision_xor3_compact_residual_raw_full_m{vision}n1152_epoch",
        f"pi05_owner_norm_full_residual_m{rows}n2048",
        f"pi05_owner_norm_compact_residual_m{rows}n2048",
        f"pi05_prefill_kv1_direct_cache_m{rows}d256p{rows}",
        "pi05_denoise_kv1_direct_cache_m50d256p64",
        f"pi05_denoise_kv1_pair_owner_{precision}_m50n32x2k1024",
        "pi05_adarms_norm_shift_h1024", "pi05_xor3_gated_residual_m50n1024",
    ]
    if profile["precision"] == "w8_prefill_a8_action_nvfp4":
        suffix = "" if rows == 400 else f"_c{rows}"
        names += [f"pi05_owner_norm_compact_a8_m{rows}n2048",
                  f"pi05_prefill_gate_up_geglu_w8a8_split{suffix}_m48n32k2048"]
    else:
        names.append(f"pi05_prefill_gate_up_geglu_weight_outer_{precision}_c{rows}x2_m{tile}n80k128")
    if "nvfp4" in profile["precision"]:
        accumulation = "acc32" if action.get("action", {}).get("linear_acc32", False) else "acc16"
        names += [f"parallel_linear_wnvfp4a16_{accumulation}_m320n64k128",
                  f"parallel_linear_wnvfp4a16_{accumulation}_m384n48k128"]
        if accumulation == "acc16" and profile["text_tokens"] == 32:
            names.append("pi05_denoise_gate_up_geglu_nvfp4_acc16_m320n64k128")
    else:
        names.append("pi05_denoise_gate_up_geglu_" + precision +
                     ("_m608n32k128" if precision == "fp16" else "_m512n48k128"))
    return names


def require_profile_assets(profile, execution=None):
    if profile is None:
        return
    import torch
    try:
        check = torch.ops.rpu.require_kernel_names
    except AttributeError as error:
        raise RuntimeError("optimized Pi0.5 requires the matching RhinoForge native runtime and public operator asset") from error
    check(required_kernel_names(profile, execution))
