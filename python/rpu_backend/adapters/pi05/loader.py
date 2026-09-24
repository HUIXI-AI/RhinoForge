"""Pi0.5 model loader: file I/O + safetensors + fp16 cast + hooks (PI05-01 reshape Plan 03-01).

Owns the `_load_and_fp16_cast` + `_cache_is_stale` + `_install_fp16_hooks` + the
`load_and_construct_pi05_policy` helper (D-3-03 delegation from Pi05Policy.from_pretrained).
Relocated VERBATIM from _adapter.py:149-389 (loader body) + 191-321 (main load) +
342-389 (hooks). `load_and_construct_pi05_policy` constructs the public facade.

v5-02 / B3: relocated from transformers/pi05/loader.py to adapters/pi05/loader.py.
Internal cross-references updated to absolute api/* + adapters/* paths.
"""
from __future__ import annotations
import glob
import json
import os
import threading
import time
import warnings
from typing import Any

import torch
import torch.nn as nn

from rpu_backend.runtime.log import _LOG
from rpu_backend.api.errors import RPUBackendError, RPUUnsupportedDtypeError


# Finding 13: serialize the PI05Policy.__init__ monkey-patch in
# _load_and_fp16_cast.
_LOAD_LOCK = threading.Lock()


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
    if method == "nvfp4a16":
        if (cfg.get("storage") != "uint8_codes" or cfg.get("nvfp4_abi") != "striped_v2"
                or cfg.get("nvfp4_block_size") != 16
                or cfg.get("int4_components") != ["expert"]):
            raise RPUBackendError("Pi05 NVFP4 requires Action-only uint8_codes, block16, striped_v2 ABI")
    if cfg.get("mode") == "group_symmetric" and (
            method != "w4a16" or cfg.get("int4_group_size") != 128):
        raise RPUBackendError("Pi05 grouped INT4 requires method=w4a16 and int4_group_size=128")
    return cfg


def _pop_pi05_w8a16_tensors(policy, state_dict: dict[str, torch.Tensor], quant_config: dict | None = None):
    quant_config = quant_config or {}
    from .w4pack import int4_child_names
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

        w4_children = int4_child_names(quant_config, component=target)
        int8_names = [
            name for name in expected
            if name in state_dict and state_dict[name].dtype in (torch.int8, torch.uint8)
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
            weight = state_dict[weight_name]
            scale = state_dict[scale_name]
            module_name = weight_name.removesuffix(".weight")
            module = modules[module_name]
            if tuple(weight.shape) != (module.out_features, module.in_features):
                raise RPUBackendError(f"{weight_name}: logical quantized weight shape mismatch")
            is_w4 = module_name.rsplit(".", 1)[-1] in w4_children
            nvfp4 = is_w4 and quant_config.get("method") == "nvfp4a16"
            grouped = is_w4 and quant_config.get("mode") in ("group_symmetric", "group_wise_symmetric")
            group_size = quant_config.get("group_size", quant_config.get("int4_group_size", 128))
            n, k = weight.shape
            expected = (k // 16, n) if nvfp4 else ((k // group_size, n) if grouped else (n,))
            expected_dtype = torch.uint8 if nvfp4 else torch.float16
            if scale.dtype != expected_dtype or tuple(scale.shape) != expected:
                raise RPUBackendError(f"{scale_name} must be {expected_dtype} {expected}; got {scale.dtype} {tuple(scale.shape)}")
            if weight.dtype != (torch.uint8 if nvfp4 else torch.int8):
                raise RPUBackendError(f"{weight_name}: dtype does not match quantization metadata")
            if is_w4 and (weight.max() > (15 if nvfp4 else 7) or (not nvfp4 and weight.min() < -8)):
                raise RPUBackendError(f"{weight_name}: codes outside declared 4-bit range")
            if not nvfp4 and (not torch.isfinite(scale).all() or (scale <= 0).any()):
                raise RPUBackendError(f"{scale_name}: scales must be finite and positive")
            values = (weight.contiguous(), scale.contiguous())
            if nvfp4:
                ts_name = module_name + ".tensor_scale"
                ts = state_dict.get(ts_name)
                if (ts is None or ts.dtype != torch.float32 or ts.numel() != 1
                        or not torch.isfinite(ts).all() or (ts <= 0).any()
                        or (scale > 126).any()):
                    raise RPUBackendError(f"{ts_name}: NVFP4 requires finite positive FP32 tensor scale and finite FP8 block scales")
                values += (state_dict.pop(ts_name).contiguous(),)
            state_dict.pop(weight_name)
            state_dict.pop(scale_name)
            consumed[module_name] = values
    return consumed


def _install_pi05_w8a16_tensors(policy, consumed) -> None:
    if not consumed:
        return
    modules = dict(policy.named_modules())
    for module_name, values in consumed.items():
        weight, scale = values[:2]
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
        if len(values) == 3:
            module.register_buffer("tensor_scale", values[2])
            module._pi05_nvfp4_abi = "striped_v2"
        elif weight.dtype == torch.int8 and scale.dim() == 2:
            module._pi05_int4_group_size = weight.size(1) // scale.size(0)


def _pi05_consumed_keys(consumed) -> set[str]:
    keys: set[str] = set()
    for module_name in consumed:
        keys.add(f"{module_name}.weight")
        keys.add(f"{module_name}.weight_scale")
        if len(consumed[module_name]) == 3:
            keys.add(f"{module_name}.tensor_scale")
    return keys


# =============================================================================
# CR-R5 BLOCKER 10 (F16 real fix) — strengthened stale-cache check.
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
# BLOCKER 5 + CR-R3 Finding 13 + 16 + CR-R5 BLOCKER 10 — _load_and_fp16_cast full port.
# =============================================================================
def _load_and_fp16_cast(model_path: str, dtype: "torch.dtype", **lerobot_kwargs: Any) -> Any:
    """Load a lerobot PI05Policy + cast to fp16. Full port of pi05_converter.py:75-125.

    Finding 13: _LOAD_LOCK wraps the __init__ monkey-patch.
    CR-R5 BLOCKER 10: stale-cache check uses st_mtime_ns + size + self-exclusion.
    Finding 16: integrity (empty glob, meta tensors, missing/unexpected critical keys).
    """
    if dtype is not torch.float16:
        raise RPUUnsupportedDtypeError(
            f"_load_and_fp16_cast: dtype must be torch.float16, got {dtype}."
        )

    # WR-02: probe for lerobot FIRST — the actionable ImportError should fire
    # before any "Loading..." line is printed, otherwise the UX implies load
    # started. (D-4-09 still applies to the wrap text; only the ordering moved.)
    try:
        import lerobot  # noqa: F401 — probe only
    except ImportError as e:
        raise ImportError(
            "Pi0.5 requires lerobot. Install with: pip install lerobot\n"
            "See docs/api_reference.md#Migration for Pi0.5 setup."
        ) from e

    _LOG.info("Loading Pi0.5 model from: %s (dtype=fp16)", model_path)
    t0 = time.time()

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy as LeRobotPI05
    from lerobot.configs.policies import PreTrainedConfig
    from safetensors.torch import load_file

    # Q4 pre-load mutation.
    config = PreTrainedConfig.from_pretrained(model_path, **lerobot_kwargs)
    config.device = "cpu"
    config.dtype = "float32"

    # Finding 16: empty safetensors glob rejected.
    all_safetensors = glob.glob(os.path.join(model_path, "*.safetensors"))
    if not all_safetensors:
        raise RPUBackendError(
            f"_load_and_fp16_cast: no *.safetensors files found in {model_path}")
    quant_config = _load_pi05_rpu_quant_config(model_path)

    remapped_path = os.path.join(model_path, "model_remapped.safetensors")

    # CR-R5 BLOCKER 10: EXCLUDE remapped from source list to avoid self-matching.
    source_safetensors = [s for s in all_safetensors
                           if os.path.basename(s) != "model_remapped.safetensors"]

    if _cache_is_stale(source_safetensors, remapped_path):
        # D-B3 (Phase 5 Plan 05-01): _remap_and_save lives in weights.py.
        # v5-02 / B3: relocated from transformers.pi05.weights.
        from rpu_backend.adapters.pi05.weights import _remap_and_save
        _remap_and_save(model_path, remapped_path)

    # Finding 13: guard the PI05Policy.__init__ monkey-patch with _LOAD_LOCK.
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
    if _consumed_method in ("w4a16_fake_int8", "w4a16", "nvfp4a16") and not w8a16_consumed:
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
        else:
            buf_dtype = buf.dtype if buf.dtype != torch.float32 else torch.float32
            mod.register_buffer(buf_name, torch.zeros(buf.shape, dtype=buf_dtype))

    load_result = policy.load_state_dict(state_dict, assign=True, strict=False)

    # Finding 16: critical-key integrity checks.
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

    # Post-load fp16 cast + hooks (CR-R5 BLOCKER 6 below).
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
# CR-R5 BLOCKER 6 + CR-R7 BLOCKER D — verbatim port of pi05_converter.py:226-257.
#
# The body below IS the legacy body, line-by-line. Iter-5's skeleton had two
# bugs that CR-R7 BLOCKER D corrects:
#   (1) Post-hook was registered ONLY on Linear/Conv2d (inside the isinstance).
#       Legacy registers on EVERY module (outside isinstance).
#   (2) Post-hook cast direction was "back to input dtype" in iter-5.
#       Legacy casts fp32 -> fp16 (unidirectional).
# =============================================================================
def _install_fp16_hooks(policy) -> None:
    """VERBATIM port of pi05_converter.py:226-257 _install_fp16_hooks.

    CR-R7 BLOCKER D (HIGH) CORRECTION: iter-5 skeleton had TWO bugs:
      1. Registered the post-hook ONLY on Linear/Conv2d modules (inside the
         `isinstance` block). The LEGACY registers the post-hook on EVERY module.
      2. Iter-5 post-hook cast fp16 output back to the ORIGINAL input dtype.
         The LEGACY casts fp32 output -> fp16 (unidirectional).

    Inner functions preserve legacy names verbatim for grep auditability.
    """
    # -- BEGIN VERBATIM PORT FROM pi05_converter.py:226-257 --
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
        # CR-R7 BLOCKER D: UNCONDITIONAL — post-hook registers on EVERY module,
        # regardless of type. Same nesting as pi05_converter.py:256-257.
        m.register_forward_hook(_output_dtype_hook)
    # -- END VERBATIM PORT --


# =============================================================================
# D-3-03: load_and_construct_pi05_policy — Pi05Policy.from_pretrained delegates here.
# =============================================================================
def load_and_construct_pi05_policy(
    pretrained_name_or_path: str,
    *,
    dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = False,
    rpu_execution=None,
    **lerobot_kwargs: Any,
) -> "Any":
    """D-01 canonical loader for Pi0.5. Delegated from Pi05Policy.from_pretrained per D-3-03.

    D-19 fp16 guard fires here BEFORE any lerobot load. trust_remote_code is an
    EXPLICIT kwarg with default False (D-CR3-SEC1 security).
    """
    if dtype is not torch.float16:
        raise RPUUnsupportedDtypeError(
            f"Pi05Policy requires dtype=torch.float16; got {dtype}. "
            "RPU kernels are fp16-primary (CLAUDE.md §Must-Never Rules).")
    if not isinstance(trust_remote_code, bool):
        raise TypeError(
            "Pi05Policy.from_pretrained: trust_remote_code must be bool, "
            f"got {trust_remote_code!r}"
        )
    if "trust_remote_code" in lerobot_kwargs:
        raise TypeError(
            "Pi05Policy.from_pretrained() got duplicate keyword argument "
            "'trust_remote_code' (pass it ONCE as the explicit kwarg).")
    if trust_remote_code:
        raise ValueError("RhinoForge requires trust_remote_code=False")
    lerobot_kwargs["trust_remote_code"] = trust_remote_code
    lerobot_policy = _load_and_fp16_cast(pretrained_name_or_path, dtype, **lerobot_kwargs)
    # v5-02 D-12: Pi05Policy lives in api/policy.py (was transformers/pi05/policy.py
    # in v4 with a thin shim). Late import keeps loader -> policy circular-free.
    from rpu_backend.api.policy import Pi05Policy
    if rpu_execution is None:
        return Pi05Policy.from_lerobot_policy(lerobot_policy)
    return Pi05Policy.from_lerobot_policy(
        lerobot_policy, rpu_execution=rpu_execution
    )
