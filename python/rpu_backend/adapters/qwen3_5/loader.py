"""Qwen3.5 checkpoint loader: read the quant declaration, build the full
conditional-generation skeleton on meta, and install int8 text weights + fp16
scales without letting HF cast them. Modern formats retain fp16 Vision; the
exact pre-v1 27B bundle loads only its text model.

Mirrors the Pi0.5 loader shape (``adapters/pi05/loader.py``): the quantization
type is a property of the checkpoint, not of the process. ``quant_config`` in
``config.json`` selects the branch; a declared method that is never consumed is
an error, and an unknown method is rejected rather than silently degraded to
FP16.

Kept separate from ``quant/load.py`` on purpose — that loader's projection-suffix
list is Qwen3-specific and does not cover the Qwen3.5 GDN ``linear_attn.in_proj_*``
projections. Scope here comes from ``quant_scope``, driven by ``layer_types``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
)

from rpu_backend.api.errors import (
    RPUBackendError, RPUUnsupportedDtypeError, UnsupportedModelError,
)
from rpu_backend.adapters.qwen3_5.quant_scope import (
    FP16, W4A16_PGRP, LEGACY_W8A16_PROFILE, is_full_from_layer_types,
    normalize_quant_config, resolve_quant_specs, scale_key,
    validate_legacy_text_profile,
)
from rpu_backend.runtime.log import _LOG


_CONFIG_KWARGS = frozenset({
    "cache_dir", "revision", "subfolder", "local_files_only",
    "trust_remote_code", "token", "force_download", "proxies",
})


def parse_quant_config(config):
    """Return the validated quant declaration, or ``None`` for a FP16 checkpoint."""
    quant = getattr(config, "quant_config", None)
    if quant is None:
        return None
    if not isinstance(quant, dict):
        raise RPUBackendError(
            f"Qwen3.5 quant_config must be a JSON object, got {type(quant).__name__}")

    try:
        is_full = is_full_from_layer_types(_layer_types(config))
        quant = normalize_quant_config(is_full, quant)
        resolve_quant_specs(is_full, quant)
        validate_legacy_text_profile(config)
    except ValueError as exc:
        raise UnsupportedModelError(str(exc)) from exc

    storage = quant.get("storage", "int8")
    scale_dtype = quant.get("scale_dtype", "float16")
    if storage != "int8" or scale_dtype != "float16":
        raise UnsupportedModelError(
            "Qwen3.5 quantized checkpoints require storage='int8' and "
            f"scale_dtype='float16', got {storage!r} and {scale_dtype!r}")
    if quant.get("format_version", 1) == 1:
        method = quant["method"]
        expected_bits = 4 if method == W4A16_PGRP else 8
        value_bits = int(quant.get("value_bits", expected_bits))
        if value_bits != expected_bits:
            raise UnsupportedModelError(
                f"method={method} requires value_bits={expected_bits}, got "
                f"{value_bits}")
    return dict(quant)


def _layer_types(config):
    text_config = getattr(config, "text_config", config)
    layer_types = list(getattr(text_config, "layer_types", None) or [])
    n_layers = int(text_config.num_hidden_layers)
    if len(layer_types) != n_layers:
        raise UnsupportedModelError(
            f"Qwen3.5 layer_types length {len(layer_types)} != num_hidden_layers "
            f"{n_layers}.")
    return layer_types


def _shard_paths(ckpt_dir: Path) -> list[Path]:
    single = ckpt_dir / "model.safetensors"
    if single.is_file():
        return [single]
    index_path = ckpt_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RPUBackendError(
            f"Qwen3.5 quantized checkpoint has neither model.safetensors nor "
            f"model.safetensors.index.json: {ckpt_dir}")
    with index_path.open() as f:
        weight_map = json.load(f)["weight_map"]
    shards = [ckpt_dir / name for name in sorted(set(weight_map.values()))]
    missing = [str(s) for s in shards if not s.is_file()]
    if missing:
        raise RPUBackendError(
            f"Qwen3.5 quantized checkpoint shard(s) missing: {missing}")
    return shards


def _rotary_buffer(module, name):
    """Rebuild a non-persistent rotary buffer that meta construction left empty.

    Two recipes exist in modeling_qwen3_5: the text rotary is config-driven
    (partial rotary / rope_parameters), while the vision rotary only keeps
    ``dim``/``theta``. The Qwen3 helper in ``quant/load.py`` covers the first
    shape only, which is why this is not shared.
    """
    if name not in ("inv_freq", "original_inv_freq"):
        return None

    config = getattr(module, "config", None)
    if config is not None and hasattr(module, "compute_default_rope_parameters"):
        rope_type = getattr(module, "rope_type", None) or "default"
        if rope_type == "default":
            rope_init_fn = module.compute_default_rope_parameters
        else:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
        inv_freq, attention_scaling = rope_init_fn(config, device=torch.device("cpu"))
        module.attention_scaling = attention_scaling
        inv_freq = inv_freq.detach().to(device="cpu").contiguous()
        return inv_freq.clone() if name == "original_inv_freq" else inv_freq

    dim = getattr(module, "dim", None)
    theta = getattr(module, "theta", None)
    if dim is not None and theta is not None and name == "inv_freq":
        return 1.0 / (
            theta ** (torch.arange(0, int(dim), 2, dtype=torch.float) / int(dim))
        )
    return None


def _materialize_meta_buffers(model) -> None:
    unresolved: list[str] = []
    for module_name, module in model.named_modules():
        for buffer_name, buffer in list(module._buffers.items()):
            if buffer is None or buffer.device.type != "meta":
                continue
            replacement = _rotary_buffer(module, buffer_name)
            if replacement is None:
                unresolved.append(
                    f"{module_name}.{buffer_name}" if module_name else buffer_name)
                continue
            module.register_buffer(buffer_name, replacement, persistent=False)
    if unresolved:
        raise RPUBackendError(
            f"Qwen3.5 loader left {len(unresolved)} buffer(s) unmaterialized: "
            f"{sorted(unresolved)[:8]}{' ...' if len(unresolved) > 8 else ''}")


def _pop_projection_tensors(state_dict, expected, modules):
    """Validate scope weights and remove quantized tensors for manual install.

    Popping before ``load_state_dict`` is mandatory: ``assign=True`` rebuilds the
    Parameter and PyTorch refuses to make an integer tensor require gradients,
    while the default copy path would silently cast the int8 codes to fp16 and
    drop the scales entirely.

    Returns ``(consumed, seen)`` — ``consumed`` keyed by MODEL module name,
    ``seen`` the checkpoint keys it covered.
    """
    consumed, seen = {}, set()
    for weight_key in sorted(k for k in state_dict if k in expected):
        spec = expected[weight_key]
        weight = state_dict[weight_key]
        scale_name = scale_key(weight_key)
        if spec.method == FP16:
            if weight.dtype != torch.float16:
                raise RPUBackendError(
                    f"Qwen3.5 FP16 projection {weight_key} must be fp16, got "
                    f"{weight.dtype}")
            if scale_name in state_dict:
                raise RPUBackendError(
                    f"Qwen3.5 FP16 projection {weight_key} must not carry "
                    f"{scale_name}")
            seen.add(weight_key)
            continue

        weight = state_dict.pop(weight_key)
        if scale_name not in state_dict:
            raise RPUBackendError(
                f"Qwen3.5 {spec.method} checkpoint has {weight_key} but no "
                f"{scale_name}")
        scale = state_dict.pop(scale_name)
        if weight.dtype != torch.int8:
            raise RPUBackendError(
                f"Qwen3.5 {spec.method} projection {weight_key} must be int8, got "
                f"{weight.dtype}")
        group_size = spec.group_size or 0
        want_dim = 2 if group_size else 1
        if scale.dtype != torch.float16 or scale.dim() != want_dim:
            raise RPUBackendError(
                f"Qwen3.5 {spec.method} {scale_name} must be {want_dim}D fp16, got dtype="
                f"{scale.dtype}, shape={tuple(scale.shape)}")
        if (not bool(torch.isfinite(scale).all())
                or not bool((scale > 0).all())):
            raise RPUBackendError(
                f"Qwen3.5 {spec.method} {scale_name} must contain only finite, "
                "positive values")
        if group_size:
            # group-wise: [G, N] with G = K / group_size.
            groups = weight.size(1) // group_size
            if (weight.size(1) % group_size or tuple(scale.shape)
                    != (groups, weight.size(0))):
                raise RPUBackendError(
                    f"Qwen3.5 W4A16 {scale_name} shape {tuple(scale.shape)} != "
                    f"[K/group_size={groups}, N={weight.size(0)}] for "
                    f"{weight_key} {tuple(weight.shape)}")
            code_min = int(weight.min().item())
            code_max = int(weight.max().item())
            if code_min < -8 or code_max > 7:
                raise RPUBackendError(
                    f"Qwen3.5 W4A16 {weight_key} codes must be in [-8, 7], got "
                    f"[{code_min}, {code_max}]")
        elif scale.numel() != weight.size(0):
            raise RPUBackendError(
                f"Qwen3.5 W8A16 {scale_name}.numel()={scale.numel()} != "
                f"{weight_key}.shape[0]={weight.size(0)}")
        # The quantized loader keeps the full ConditionalGeneration hierarchy,
        # so checkpoint and module paths are identical, including
        # `model.language_model`.
        module_name = weight_key[: -len(".weight")]
        module = modules.get(module_name)
        if not isinstance(module, nn.Linear):
            raise RPUBackendError(
                f"Qwen3.5 {spec.method} checkpoint has no Linear for {weight_key} "
                f"(resolved to {module_name})")
        if tuple(module.weight.shape) != tuple(weight.shape):
            raise RPUBackendError(
                f"Qwen3.5 {spec.method} {weight_key} has shape {tuple(weight.shape)}, "
                f"model expects {tuple(module.weight.shape)}")
        consumed[module_name] = (weight.contiguous(), scale.contiguous())
        seen.add(weight_key)

    stray = sorted(k for k in state_dict if k.endswith(".weight_scale"))
    if stray:
        raise RPUBackendError(
            f"Qwen3.5 checkpoint carries {len(stray)} weight_scale tensor(s) "
            f"outside the quantization scope: {stray[:5]}")
    return consumed, seen


def _install_quant_tensors(consumed, modules) -> None:
    for module_name, (weight, scale) in consumed.items():
        module = modules[module_name]
        module.weight = nn.Parameter(weight, requires_grad=False)
        if "weight_scale" in module._buffers:
            module._buffers["weight_scale"] = scale
        else:
            module.register_buffer("weight_scale", scale)


def _assert_expected_unexpected(unexpected, model) -> None:
    """Allow only the keys HF itself declares ignorable (e.g. the MTP head)."""
    patterns = [re.compile(p) for p in
                (getattr(model, "_keys_to_ignore_on_load_unexpected", None) or [])]
    unknown = [k for k in unexpected if not any(p.search(k) for p in patterns)]
    if unknown:
        raise RPUBackendError(
            f"Qwen3.5 checkpoint has {len(unknown)} tensor(s) the model does not "
            f"accept: {sorted(unknown)[:8]}{' ...' if len(unknown) > 8 else ''}")


def _legacy_tensor_locations(ckpt_dir: Path):
    """Read shard headers, rejecting duplicate keys before reading any weights."""
    index_path = ckpt_dir / "model.safetensors.index.json"
    weight_map = None
    if index_path.is_file():
        if (ckpt_dir / "model.safetensors").is_file():
            raise RPUBackendError("Qwen3.8 legacy checkpoint has ambiguous single/sharded layouts")
        with index_path.open() as stream:
            weight_map = json.load(stream)["weight_map"]
    locations, raw_locations = {}, {}
    for shard in _shard_paths(ckpt_dir):
        with safe_open(str(shard), framework="pt", device="cpu") as tensors:
            for raw_key in tensors.keys():
                if raw_key in raw_locations:
                    raise RPUBackendError(f"Qwen3.8 legacy checkpoint has duplicate tensor {raw_key!r}")
                raw_locations[raw_key] = shard.name
                # These are the only namespaces intentionally absent from the
                # text-only skeleton; do not materialize their tensors at all.
                if raw_key.startswith(("model.visual.", "mtp.")):
                    continue
                prefix = "model.language_model."
                key = "model." + raw_key[len(prefix):] if raw_key.startswith(prefix) else raw_key
                if key in locations:
                    raise RPUBackendError(f"Qwen3.8 legacy checkpoint has duplicate text tensor {key!r}")
                locations[key] = (shard, raw_key)
    if weight_map is not None and raw_locations != weight_map:
        raise RPUBackendError("Qwen3.8 legacy checkpoint shard contents disagree with the weight_map")
    return locations


def _load_legacy_qwen38_text_model(ckpt_dir: Path, config, quant: dict):
    """Stream the exact legacy text bundle without instantiating Vision or MTP."""
    from transformers import AutoModelForCausalLM

    text_config = getattr(config, "text_config", config)
    specs = resolve_quant_specs(is_full_from_layer_types(_layer_types(config)), quant)
    expected = {f"model.{relpath}.weight": spec for relpath, spec in specs.items()}
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(text_config)
    # assign=True inherits requires_grad from the skeleton. Inference swizzles
    # must not build autograd graphs that retain copies of the 27B weights.
    model.requires_grad_(False)
    modules = dict(model.named_modules())
    own_state = model.state_dict()
    locations = _legacy_tensor_locations(ckpt_dir)
    scale_keys = {scale_key(key) for key, spec in expected.items() if spec.method != FP16}
    unknown = sorted(set(locations) - set(own_state) - scale_keys)
    if unknown:
        raise RPUBackendError(
            "Qwen3.8 legacy checkpoint has out-of-scope text tensor(s): "
            f"{unknown[:8]}")
    missing = sorted((set(own_state) | set(expected) | scale_keys) - set(locations))
    if missing:
        raise RPUBackendError(
            f"Qwen3.8 legacy checkpoint is missing text tensor(s): {missing[:8]}")

    consumed = {}
    shards = sorted({path for path, _ in locations.values()})
    for shard in shards:
        state_dict = {}
        with safe_open(str(shard), framework="pt", device="cpu") as tensors:
            for key, (path, raw_key) in locations.items():
                if path != shard or key in scale_keys:
                    continue
                tensor = tensors.get_tensor(raw_key)
                if (key not in expected and own_state[key].is_floating_point()
                        and not tensor.is_floating_point()):
                    raise RPUBackendError(
                        f"Qwen3.8 legacy ordinary parameter {key} must be floating point")
                state_dict[key] = tensor
                if key in expected and expected[key].method != FP16:
                    scale_name = scale_key(key)
                    scale_path, raw_scale = locations[scale_name]
                    if scale_path == shard:
                        state_dict[scale_name] = tensors.get_tensor(raw_scale)
                    else:
                        # A valid safetensors index may place weight and scale
                        # in different shards; resolve the scale by its key.
                        with safe_open(str(scale_path), framework="pt", device="cpu") as scales:
                            state_dict[scale_name] = scales.get_tensor(raw_scale)
        shard_consumed, _ = _pop_projection_tensors(state_dict, expected, modules)
        consumed.update(shard_consumed)
        result = model.load_state_dict(state_dict, assign=True, strict=False)
        if result.unexpected_keys:
            raise RPUBackendError(
                f"Qwen3.8 legacy text loader produced unexpected keys: {result.unexpected_keys[:8]}")
        del state_dict, shard_consumed

    model.tie_weights()
    _materialize_meta_buffers(model)
    model.half()
    _install_quant_tensors(consumed, modules)
    meta = [name for name, tensor in (*model.named_parameters(), *model.named_buffers())
            if tensor.device.type == "meta"]
    if meta:
        raise RPUBackendError(f"Qwen3.8 legacy text loader left meta tensors: {meta[:8]}")
    # The returned text config owns the same executable scope as the loader;
    # the adapter installs it through the current planner/runtime path.
    model.config.quant_config = dict(quant)
    model._qwen3_5_quant_config = dict(quant)
    model._qwen3_5_legacy_text_only = True
    _LOG.info("Qwen3.8-27B legacy text checkpoint loaded (%d quantized projections)", len(consumed))
    return model.eval()


def load_qwen3_5_model(
    pretrained_name_or_path,
    *,
    config=None,
    dtype: torch.dtype = torch.float16,
    **hf_kwargs,
):
    """Load a Qwen3.5 checkpoint on CPU, honouring its ``quant_config``.

    FP16 checkpoints take the ordinary full ConditionalGeneration HF path.
    Quantized checkpoints build that same full hierarchy on meta, then replace
    only the declared text projections with W8/W4 tensors. Vision is present and
    fp16 for modern formats. The exact legacy 27B policy uses a text-only
    CausalLM skeleton and skips Vision/MTP. Quantized checkpoints must be a
    local directory produced by ``rpu_backend.quant.convert_qwen3_5``. Returns an eval-mode CPU
    model ready for ``Qwen3_5Adapter(...).to_rpu()``.

    The FP16 branch is byte-for-byte the call every existing caller already
    makes. That matters: this checkpoint stores the 24 GDN ``linear_attn.norm``
    tensors as fp32 while everything else is bf16, so `text_config.dtype`
    pinning the load to bf16 is observable in the final fp16 weights. Any other
    load route silently shifts them by one bf16 ulp and moves the parity hash.
    """
    if dtype is not torch.float16:
        raise RPUUnsupportedDtypeError(
            f"Qwen3.5 requires dtype=torch.float16, got {dtype}. RPU kernels are "
            "fp16-primary.")

    if config is None:
        cfg_kwargs = {k: v for k, v in hf_kwargs.items() if k in _CONFIG_KWARGS}
        config = AutoConfig.from_pretrained(pretrained_name_or_path, **cfg_kwargs)
    quant = parse_quant_config(config)

    if quant is None:
        # Preserve the exact public behaviour from before the adapter-specific
        # loader hook: FP16 owns the full Qwen3_5ForConditionalGeneration model,
        # including its lazy Vision tower. Using AutoModelForCausalLM here silently
        # drops `visual` and breaks controlled Vision evaluation.
        load_kwargs = dict(hf_kwargs)
        load_kwargs.setdefault("device_map", "cpu")
        load_kwargs.setdefault("low_cpu_mem_usage", True)
        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_name_or_path, torch_dtype=dtype, **load_kwargs)
        # text_config.dtype=bfloat16 overrides from_pretrained(dtype=fp16); the
        # RPU install path requires fp16.
        return model.half().eval()

    ckpt_dir = Path(pretrained_name_or_path).expanduser()
    if not ckpt_dir.is_dir():
        raise RPUBackendError(
            f"Qwen3.5 quantized checkpoints must be a local directory, got "
            f"{pretrained_name_or_path!r}. Download it first.")

    if quant.get("profile") == LEGACY_W8A16_PROFILE:
        return _load_legacy_qwen38_text_model(ckpt_dir, config, quant)

    is_full = is_full_from_layer_types(_layer_types(config))
    specs = resolve_quant_specs(is_full, quant)
    expected = {
        f"model.language_model.{relpath}.weight": spec
        for relpath, spec in specs.items()
    }

    # Build the full multimodal hierarchy on meta so Vision exists for every
    # precision without first materializing a duplicate fp16 text stack.
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)
    modules = dict(model.named_modules())

    consumed: dict = {}
    seen: set = set()
    unexpected: list[str] = []
    for shard in _shard_paths(ckpt_dir):
        state_dict = load_file(str(shard))
        shard_consumed, shard_seen = _pop_projection_tensors(
            state_dict, expected, modules)
        consumed.update(shard_consumed)
        seen |= shard_seen
        result = model.load_state_dict(state_dict, assign=True, strict=False)
        unexpected.extend(result.unexpected_keys)
        del state_dict

    not_consumed = sorted(set(expected) - seen)
    if not_consumed:
        raise RPUBackendError(
            f"Qwen3.5 checkpoint declares quant_config but "
            f"{len(not_consumed)} scope projection(s) were not present/validated: "
            f"{not_consumed[:8]}{' ...' if len(not_consumed) > 8 else ''}")
    _assert_expected_unexpected(unexpected, model)

    model.tie_weights()
    _materialize_meta_buffers(model)
    model.half()
    _install_quant_tensors(consumed, modules)

    meta_params = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    meta_buffers = [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if meta_params or meta_buffers:
        raise RPUBackendError(
            f"Qwen3.5 loader finished with meta tensors — params={meta_params[:3]} "
            f"buffers={meta_buffers[:3]}")

    # No `_rpu_` prefix on purpose: runtime/hw_attrs.py validates that namespace
    # against an allowlist. Mirrors Pi0.5's `policy._pi05_quant_config`.
    model._qwen3_5_quant_config = dict(quant)
    _LOG.info("Qwen3.5 %s checkpoint loaded (%d quantized projections)",
              quant.get("profile", quant.get("method")), len(consumed))
    return model.eval()
