"""PaliGemma2 internal adapter orchestrator."""
from __future__ import annotations

import copy
import gc
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

import torch

from rpu_backend.api.causal_lm import _claim_live_instance
from rpu_backend.api.errors import RPUBackendError, RPUUnsupportedDtypeError
from rpu_backend.adapters.paligemma.fallback import FallbackRecorder
from rpu_backend.adapters.paligemma.gemma2 import (
    _restore_instance_forward,
    patch_gemma2_model_for_rpu_all_layers_once,
)
from rpu_backend.runtime.weights import NUM_CORES as _RPU_NUM_CORES
from rpu_backend.adapters.siglip import patch_siglip_model_for_rpu_all_layers_once
from rpu_backend.runtime.hw_attrs import (
    install_hw_attr_validator,
    validate_postinstall,
    validate_preinstall,
)


_MODEL_INPUT_KEYS = frozenset({
    "pixel_values",
    "input_ids",
    "attention_mask",
    "token_type_ids",
})

_SWIZZLE_LOCK = threading.Lock()


def _paligemma_components(hf_model: Any) -> tuple[Any, Any, Any, Any, Any]:
    """Resolve PaliGemma components from HF nested and local fake layouts."""
    base_model = getattr(hf_model, "model", None)
    if (
        base_model is not None
        and hasattr(base_model, "vision_tower")
        and hasattr(base_model, "multi_modal_projector")
        and hasattr(base_model, "language_model")
    ):
        return (
            base_model,
            base_model.vision_tower,
            base_model.multi_modal_projector,
            base_model.language_model,
            hf_model.lm_head,
        )

    language_model = hf_model.language_model
    decoder = getattr(language_model, "model", language_model)
    lm_head = getattr(language_model, "lm_head", getattr(hf_model, "lm_head", None))
    return (
        hf_model,
        hf_model.vision_tower,
        hf_model.multi_modal_projector,
        decoder,
        lm_head,
    )


def _vision_inner(vision: Any) -> Any:
    return vision.vision_model if hasattr(vision, "vision_model") else vision


def _projector_in_features(projector: Any) -> int | None:
    leaf = getattr(projector, "linear", projector)
    value = getattr(leaf, "in_features", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _decoder_hidden_size(decoder: Any) -> int | None:
    config = getattr(decoder, "config", None)
    value = getattr(config, "hidden_size", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _image_features_from_vision_output(
    *,
    vision: Any,
    image_outputs: Any,
    projector: Any,
    decoder: Any,
) -> torch.Tensor:
    """Return image features for merge, accounting for RPU SigLIP fused projector."""
    hidden_states = image_outputs.last_hidden_state
    if len(hidden_states.shape) == 0:
        raise RPUBackendError("PaliGemma2 vision output must have a feature axis")

    vision_model = _vision_inner(vision)
    if getattr(vision_model, "_rpu_vision_handle", None) is None:
        return projector(hidden_states)

    last_dim = int(hidden_states.shape[-1])
    decoder_hidden = _decoder_hidden_size(decoder)
    if decoder_hidden is not None and last_dim == decoder_hidden:
        return hidden_states

    projector_input = _projector_in_features(projector)
    if projector_input is not None and last_dim == projector_input:
        return projector(hidden_states)

    raise RPUBackendError(
        "PaliGemma2 RPU SigLIP feature width mismatch: "
        f"vision_last_dim={last_dim}, "
        f"decoder_hidden_size={decoder_hidden}, "
        f"projector_in_features={projector_input}. "
        "RPU fused SigLIP should return either raw vision features or "
        "already-projected image features."
    )


def _bound_rpu_component(
    owner: Any,
    *,
    handle_attr: str,
    finalizer_attr: str,
    forward_name: str,
    graph_cache_attr: str | None = None,
) -> tuple[Any, Any] | None:
    """Return a component's live ownership pair when its runtime is callable."""
    handle = getattr(owner, handle_attr, None)
    finalizer = getattr(owner, finalizer_attr, None)
    if handle is None or finalizer is None:
        return None
    if not getattr(finalizer, "alive", False):
        return None
    if getattr(owner, "_rpu_cache", None) is None:
        return None
    if graph_cache_attr is not None and getattr(owner, graph_cache_attr, None) is None:
        return None
    if getattr(owner, "_rpu_lazy_init_checked", False) is not True:
        return None
    if getattr(owner, "_rpu_required_attrs", None) is None:
        return None
    installed_forward = vars(owner).get("forward")
    forward_function = getattr(installed_forward, "__func__", None)
    if not (
        getattr(installed_forward, "__self__", None) is owner
        and getattr(forward_function, "__name__", None) == forward_name
    ):
        return None
    return handle, finalizer


def _paligemma_component_runtime_state(hf_model: Any) -> dict[str, Any] | None:
    """Return complete SigLIP/Gemma2 ownership without trusting the top marker."""
    try:
        _, vision, _, decoder, _ = _paligemma_components(hf_model)
        vision_inner = _vision_inner(vision)
    except (AttributeError, TypeError):
        return None

    siglip_state = _bound_rpu_component(
        vision_inner,
        handle_attr="_rpu_vision_handle",
        finalizer_attr="_rpu_vision_handle_finalizer",
        graph_cache_attr="_rpu_siglip_graph_cache",
        forward_name="rpu_siglip_forward",
    )
    gemma2_state = _bound_rpu_component(
        decoder,
        handle_attr="_rpu_gemma2_handle",
        finalizer_attr="_rpu_gemma2_handle_finalizer",
        forward_name="rpu_gemma2_model_forward",
    )
    if siglip_state is None or gemma2_state is None:
        return None
    if getattr(decoder, "_rpu_gemma2_install_started", False) is not True:
        return None
    if getattr(decoder, "_rpu_gemma2_install_ready", False) is not True:
        return None
    for attr in (
        "_rpu_effective_num_kv_heads",
        "_rpu_attn_tp",
        "_rpu_max_seq_len",
    ):
        value = getattr(decoder, attr, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return None
    return {
        "siglip_handle": siglip_state[0],
        "gemma2_handle": gemma2_state[0],
    }


def _paligemma_runtime_state(hf_model: Any) -> dict[str, Any] | None:
    """Return complete outer runtime state; a marker alone is never ready."""
    if getattr(hf_model, "_rpu_swizzled", False) is not True:
        return None
    if getattr(hf_model, "_rpu_swizzle_started", False) is not True:
        return None
    return _paligemma_component_runtime_state(hf_model)


def _reset_adapter_runtime_state(adapter: Any) -> None:
    adapter._rpu_is_ready = False
    adapter._siglip_handle = None
    adapter._gemma2_handle = None


def _sync_adapter_runtime_state(adapter: Any, state: dict[str, Any]) -> None:
    adapter._siglip_handle = state["siglip_handle"]
    adapter._gemma2_handle = state["gemma2_handle"]
    adapter._rpu_is_ready = True


class PaliGemmaAdapter:
    """Internal orchestrator for PaliGemmaPolicy.to('rpu')."""

    def __init__(self, hf_model: Any) -> None:
        self._hf_model = hf_model
        self._rpu_is_ready = False
        self._siglip_handle: int | None = None
        self._gemma2_handle: int | None = None
        self.fallback_recorder = FallbackRecorder()
        state = _paligemma_runtime_state(hf_model)
        if state is not None:
            _sync_adapter_runtime_state(self, state)

    def _check_fp16(self) -> None:
        bad_dtypes: list[str] = []
        for name, param in _named(self._hf_model, "named_parameters"):
            if param.is_floating_point() and param.dtype != torch.float16:
                bad_dtypes.append(f"param {name}: {param.dtype}")
                if len(bad_dtypes) >= 5:
                    break
        if len(bad_dtypes) < 5:
            for name, buf in _named(self._hf_model, "named_buffers"):
                if buf.is_floating_point() and buf.dtype != torch.float16:
                    bad_dtypes.append(f"buffer {name}: {buf.dtype}")
                    if len(bad_dtypes) >= 5:
                        break
        if bad_dtypes:
            raise RPUUnsupportedDtypeError(
                "PaliGemmaAdapter.to_rpu(): floating params/buffers must be "
                "torch.float16 before swizzle. Offenders (first 5): "
                f"{bad_dtypes}. Reload with dtype=torch.float16 or call .half() "
                "on the PaliGemma model."
            )

    def to_rpu(self) -> Any:
        state = _paligemma_runtime_state(self._hf_model)
        if state is not None:
            _sync_adapter_runtime_state(self, state)
            return self._hf_model

        if self._rpu_is_ready or getattr(self._hf_model, "_rpu_swizzled", False):
            _reset_adapter_runtime_state(self)
            raise RPUBackendError(
                "PaliGemmaAdapter.to_rpu(): model is marked ready but its "
                "SigLIP/Gemma2 RPU runtime is incomplete; reload via "
                "PaliGemmaPolicy.from_pretrained(...)."
            )

        if getattr(self._hf_model, "_rpu_swizzle_started", False):
            raise RPUBackendError(
                "PaliGemmaAdapter.to_rpu(): prior swizzle attempt failed mid-way; "
                "submodel weights are in undefined state. Reload via "
                "PaliGemmaPolicy.from_pretrained(...)."
            )

        if not _SWIZZLE_LOCK.acquire(blocking=False):
            raise RPUBackendError(
                "PaliGemmaAdapter.to_rpu(): another PaliGemma swizzle in progress."
            )
        vision_inner = None
        decoder = None
        vision_had_instance_forward = False
        vision_original_instance_forward = None
        try:
            state = _paligemma_runtime_state(self._hf_model)
            if state is not None:
                _sync_adapter_runtime_state(self, state)
                return self._hf_model
            if self._rpu_is_ready or getattr(
                self._hf_model, "_rpu_swizzled", False
            ):
                _reset_adapter_runtime_state(self)
                raise RPUBackendError(
                    "PaliGemmaAdapter.to_rpu(): model is marked ready but its "
                    "SigLIP/Gemma2 RPU runtime is incomplete; reload via "
                    "PaliGemmaPolicy.from_pretrained(...)."
                )
            if getattr(self._hf_model, "_rpu_swizzle_started", False):
                raise RPUBackendError(
                    "PaliGemmaAdapter.to_rpu(): prior swizzle attempt failed "
                    "mid-way; reload via PaliGemmaPolicy.from_pretrained(...)."
                )

            validate_preinstall(self._hf_model)
            self._check_fp16()
            _, vision, projector, decoder, _ = _paligemma_components(
                self._hf_model
            )
            vision_inner = _vision_inner(vision)
            vision_had_instance_forward = "forward" in vars(vision_inner)
            vision_original_instance_forward = vars(vision_inner).get("forward")

            _claim_live_instance(self._hf_model)
            try:
                self._hf_model._rpu_swizzle_started = True

                _cfg_nkv = int(decoder.config.num_key_value_heads)
                decoder._rpu_attn_tp = min(_RPU_NUM_CORES, _cfg_nkv)
                decoder._rpu_max_seq_len = 4096

                patch_siglip_model_for_rpu_all_layers_once(
                    vision_inner,
                    projector,
                )
                patch_gemma2_model_for_rpu_all_layers_once(decoder)

                if hasattr(vision_inner, "encoder"):
                    vision_inner.encoder.to("rpu")
                if hasattr(vision_inner, "post_layernorm"):
                    vision_inner.post_layernorm.to("rpu")
                if hasattr(projector, "to"):
                    projector.to("rpu")
                decoder.to("rpu")
                if getattr(decoder, "embed_tokens", None) is not None:
                    decoder.embed_tokens.to("cpu")

                validate_postinstall(self._hf_model)
                install_hw_attr_validator(self._hf_model)
                component_state = _paligemma_component_runtime_state(
                    self._hf_model
                )
                if component_state is None:
                    raise RPUBackendError(
                        "PaliGemmaAdapter.to_rpu(): submodel installers returned "
                        "without complete live runtime ownership."
                    )
                _sync_adapter_runtime_state(self, component_state)
                self._hf_model._rpu_swizzled = True
                return self._hf_model
            except BaseException:
                _reset_adapter_runtime_state(self)
                try:
                    _cleanup_siglip_handle(
                        vision_inner,
                        had_instance_forward=vision_had_instance_forward,
                        original_forward=vision_original_instance_forward,
                    )
                except Exception:
                    pass
                try:
                    _cleanup_gemma2_handle(decoder)
                except Exception:
                    pass
                raise
        finally:
            _SWIZZLE_LOCK.release()

    def build_smoke_sample(self) -> dict[str, torch.Tensor]:
        from rpu_backend.adapters.paligemma.profile import (
            IMAGE_TOKEN_INDEX,
            NUM_IMAGE_TOKENS,
        )

        input_ids = torch.tensor(
            [[IMAGE_TOKEN_INDEX] * NUM_IMAGE_TOKENS + [2]],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[0, -1] = 1
        pixel_values = torch.zeros(1, 3, 224, 224, dtype=torch.float16)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def build_offset_placeholder_sample(self) -> dict[str, torch.Tensor]:
        from rpu_backend.adapters.paligemma.profile import (
            IMAGE_TOKEN_INDEX,
            NUM_IMAGE_TOKENS,
        )

        prefix = [2]
        suffix = [2]
        input_ids = torch.tensor(
            [prefix + [IMAGE_TOKEN_INDEX] * NUM_IMAGE_TOKENS + suffix],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.tensor(
            [[1] * len(prefix) + [0] * NUM_IMAGE_TOKENS + [1] * len(suffix)],
            dtype=torch.long,
        )
        pixel_values = torch.zeros(1, 3, 224, 224, dtype=torch.float16)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def run_numeric_probe(
        self,
        sample: dict[str, torch.Tensor],
        *,
        max_new_tokens: int = 4,
    ) -> dict[str, Any]:
        if not self._rpu_is_ready:
            raise RPUBackendError(
                "PaliGemmaAdapter.run_numeric_probe requires .to('rpu') first"
            )
        if max_new_tokens < 1:
            raise ValueError("PaliGemma2 numeric probe requires max_new_tokens >= 1")

        cpu_model = self._load_cpu_probe_model()
        with torch.inference_mode():
            cpu_tensors, cpu_tokens = self._collect_probe_tensors_and_tokens(
                cpu_model,
                sample,
                target_device="cpu",
                max_new_tokens=max_new_tokens,
            )
            # cpu_tensors are already .detach().to("cpu") (see
            # _collect_probe_tensors_and_tokens l.640-670); cpu_model is no
            # longer needed once we have cpu_tokens for forcing the RPU
            # decode path. Release before the RPU forward so the host doesn't
            # carry both CPU fp16 and RPU fp16 weight sets simultaneously.
            del cpu_model
            gc.collect()
            rpu_tensors, rpu_tokens = self._collect_probe_tensors_and_tokens(
                self._hf_model,
                sample,
                target_device="rpu",
                max_new_tokens=max_new_tokens,
                forced_decode_tokens=cpu_tokens,
            )

        result: dict[str, Any] = {}
        for key in (
            "scaled_image_features",
            "merged_inputs_embeds",
            "decode_next_token_embeddings",
            "prefill_last_hidden_state",
            "decode_final_token_hidden",
        ):
            result[key] = self._compare_tensor(cpu_tensors[key], rpu_tensors[key])
        result["logits"] = self._compare_logits(
            cpu_tensors["logits"],
            rpu_tensors["logits"],
        )
        result["metadata"] = {
            "forced_decode_tokens": cpu_tokens.reshape(-1).tolist(),
            "rpu_decode_input_tokens": rpu_tokens.reshape(-1).tolist(),
            "cpu_top1_token": _topk_tokens(cpu_tensors["logits"], 1),
            "rpu_top1_token": _topk_tokens(rpu_tensors["logits"], 1),
            "cpu_top10_tokens": _topk_tokens(cpu_tensors["logits"], 10),
            "rpu_top10_tokens": _topk_tokens(rpu_tensors["logits"], 10),
            "max_new_tokens": int(max_new_tokens),
            "fallback_events": [
                event.to_dict() for event in self.fallback_recorder.events
            ],
        }
        return result

    def run_token_probe(
        self,
        sample: dict[str, torch.Tensor],
        *,
        min_new_tokens: int,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        if not self._rpu_is_ready:
            raise RPUBackendError(
                "PaliGemmaAdapter.run_token_probe requires .to('rpu') first"
            )
        if min_new_tokens < 0 or max_new_tokens < 1:
            raise ValueError(
                "PaliGemma2 token probe requires min_new_tokens >= 0 and "
                "max_new_tokens >= 1"
            )
        if min_new_tokens > max_new_tokens:
            raise ValueError(
                "PaliGemma2 token probe requires min_new_tokens <= max_new_tokens"
            )

        cpu_model = self._load_cpu_probe_model()
        cpu_sample = _clone_sample_to_cpu(sample)
        with torch.inference_mode():
            cpu_output = cpu_model.generate(
                **cpu_sample,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
            # Extract everything we need from cpu_model BEFORE releasing it
            # — _eos_token_ids returns a set[int] (pure Python ints, no
            # tensor refs), cpu_output is already a CPU tensor. After this
            # del the host carries only the RPU fp16 weights during the
            # RPU generate below.
            eos_ids = _eos_token_ids(cpu_model)
            del cpu_model
            gc.collect()
            rpu_output = self.generate(
                **sample,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
            rpu_output = _truncate_generated_at_eos(
                rpu_output,
                input_len=int(cpu_sample["input_ids"].shape[1]),
                min_new_tokens=min_new_tokens,
                eos_token_ids=eos_ids,
            )
        cpu_tokens = cpu_output.detach().to("cpu", dtype=torch.long)
        rpu_tokens = rpu_output.detach().to("cpu", dtype=torch.long)
        return {
            "all_tokens_match": bool(torch.equal(cpu_tokens, rpu_tokens)),
            "cpu_tokens": cpu_tokens.reshape(-1).tolist(),
            "rpu_tokens": rpu_tokens.reshape(-1).tolist(),
        }

    def run_mask_probe(self, sample: dict[str, torch.Tensor]) -> dict[str, Any]:
        from rpu_backend.adapters.paligemma.generation import (
            build_prefill_additive_attention_mask,
        )
        from rpu_backend.adapters.paligemma.merge import (
            merge_image_features,
            reverse_token_types,
            validate_generate_inputs,
        )
        from rpu_backend.adapters.paligemma.profile import (
            IMAGE_TOKEN_INDEX,
            MAX_SEQ_LEN,
            NUM_IMAGE_TOKENS,
            TEXT_HIDDEN_SIZE,
        )

        cpu_sample = _clone_sample_to_cpu(sample)
        input_ids = cpu_sample["input_ids"]
        attention_mask = cpu_sample["attention_mask"]
        token_type_ids = cpu_sample["token_type_ids"]
        pixel_values = cpu_sample["pixel_values"]

        missing_token_type_ids_raises = False
        try:
            validate_generate_inputs(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=None,
                max_new_tokens=1,
                max_seq_len=MAX_SEQ_LEN,
            )
        except RPUBackendError:
            missing_token_type_ids_raises = True

        token_type_reverse_ok = bool(torch.equal(
            reverse_token_types(token_type_ids),
            1 - token_type_ids,
        ))

        local_mask = build_prefill_additive_attention_mask(
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            dtype=torch.float16,
        )
        hf_masks = _build_hf_prefill_masks(
            config=getattr(self._hf_model, "config", None),
            sample=cpu_sample,
        )
        local_allowed = _mask_allowed(local_mask)
        full_attention_matches_hf = bool(torch.equal(
            local_allowed,
            _mask_allowed(hf_masks["full_attention"]),
        ))
        sliding_attention_matches_hf = bool(torch.equal(
            local_allowed,
            _mask_allowed(hf_masks["sliding_attention"]),
        ))

        placeholder_count_mismatch_raises = False
        bad_input_ids = input_ids.clone()
        image_positions = (bad_input_ids == IMAGE_TOKEN_INDEX).nonzero()
        if image_positions.numel() > 0:
            row, col = image_positions[0].tolist()
            bad_input_ids[row, col] = 2
        try:
            merge_image_features(
                input_ids=bad_input_ids,
                inputs_embeds=torch.zeros(
                    int(input_ids.shape[0]),
                    int(input_ids.shape[1]),
                    TEXT_HIDDEN_SIZE,
                    dtype=torch.float16,
                ),
                image_features=torch.zeros(
                    int(input_ids.shape[0]),
                    NUM_IMAGE_TOKENS,
                    TEXT_HIDDEN_SIZE,
                    dtype=torch.float16,
                ),
            )
        except RPUBackendError:
            placeholder_count_mismatch_raises = True

        return {
            "missing_token_type_ids_raises": missing_token_type_ids_raises,
            "token_type_reverse": (
                "1-token_type_ids" if token_type_reverse_ok else "mismatch"
            ),
            "full_attention_matches_hf": full_attention_matches_hf,
            "sliding_attention_matches_hf": sliding_attention_matches_hf,
            "placeholder_count_mismatch_raises": placeholder_count_mismatch_raises,
        }

    def run_runtime_probe(self, sample: dict[str, torch.Tensor]) -> dict[str, Any]:
        del sample
        self.to_rpu()
        _, vision, projector, decoder, _ = _paligemma_components(self._hf_model)
        vision_inner = (
            vision.vision_model if hasattr(vision, "vision_model") else vision
        )

        siglip_handle = getattr(vision_inner, "_rpu_vision_handle", None)
        gemma2_handle = getattr(decoder, "_rpu_gemma2_handle", None)
        projector_leaf = projector.linear if hasattr(projector, "linear") else projector
        siglip_sentinel = bool(
            getattr(vision_inner, "_rpu_siglip_weights_converted", False)
        )
        projector_sentinel = bool(
            getattr(projector, "_rpu_weights_converted", False)
            or getattr(projector_leaf, "_rpu_weights_converted", False)
        )
        gemma2_sentinel = bool(
            gemma2_handle is not None
            and hasattr(decoder, "_rpu_attn_tp")
            and hasattr(decoder, "_rpu_max_seq_len")
        )
        model_sentinel = bool(getattr(self._hf_model, "_rpu_swizzled", False))
        self.to_rpu()
        second_siglip_handle = getattr(vision_inner, "_rpu_vision_handle", None)
        second_gemma2_handle = getattr(decoder, "_rpu_gemma2_handle", None)

        spm = _run_spm_contract_check()
        return {
            "siglip_reached_rpu": siglip_handle is not None,
            "projector_reached_rpu": projector_sentinel,
            "gemma2_reached_rpu": gemma2_handle is not None,
            "siglip_sentinel_set": siglip_sentinel,
            "projector_sentinel_set": projector_sentinel,
            "gemma2_sentinel_set": gemma2_sentinel,
            "model_swizzled_sentinel_set": model_sentinel,
            "second_to_rpu_no_double_swizzle": (
                siglip_handle == second_siglip_handle
                and gemma2_handle == second_gemma2_handle
            ),
            "spm_boundaries_reset": spm["returncode"] == 0,
            "spm_contract_stdout": spm["stdout"],
            "spm_contract_stderr": spm["stderr"],
        }

    def _compare_tensor(
        self,
        cpu_tensor: torch.Tensor,
        rpu_tensor: torch.Tensor,
    ) -> dict[str, Any]:
        cpu = cpu_tensor.detach().to("cpu", dtype=torch.float32)
        rpu = rpu_tensor.detach().to("cpu", dtype=torch.float32)
        diff = cpu - rpu
        return {
            "shape": list(cpu.shape),
            "mse": float(torch.mean(diff * diff).item()),
            "cosine": _cosine(cpu, rpu),
            "max_abs": float(torch.max(torch.abs(diff)).item()),
        }

    def _compare_logits(
        self,
        cpu_logits: torch.Tensor,
        rpu_logits: torch.Tensor,
    ) -> dict[str, Any]:
        metric = self._compare_tensor(cpu_logits, rpu_logits)
        metric.update({
            "top5_overlap": _topk_overlap(cpu_logits, rpu_logits, 5),
            "top10_overlap": _topk_overlap(cpu_logits, rpu_logits, 10),
            "cpu_top1_token": _topk_tokens(cpu_logits, 1),
            "rpu_top1_token": _topk_tokens(rpu_logits, 1),
            "cpu_top10_tokens": _topk_tokens(cpu_logits, 10),
            "rpu_top10_tokens": _topk_tokens(rpu_logits, 10),
        })
        return metric

    def _collect_probe_tensors(
        self,
        model: Any,
        sample: dict[str, torch.Tensor],
        *,
        target_device: str,
        max_new_tokens: int = 4,
        forced_decode_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        tensors, _ = self._collect_probe_tensors_and_tokens(
            model,
            sample,
            target_device=target_device,
            max_new_tokens=max_new_tokens,
            forced_decode_tokens=forced_decode_tokens,
        )
        return tensors

    def _collect_probe_tensors_and_tokens(
        self,
        model: Any,
        sample: dict[str, torch.Tensor],
        *,
        target_device: str,
        max_new_tokens: int,
        forced_decode_tokens: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        import rpu_backend.adapters.paligemma.generation as generation

        from rpu_backend.adapters.paligemma.merge import (
            merge_image_features,
            scale_image_features,
            validate_generate_inputs,
        )
        from rpu_backend.adapters.paligemma.profile import MAX_SEQ_LEN

        cpu_sample = _clone_sample_to_cpu(sample)
        pixel_values = cpu_sample["pixel_values"]
        input_ids = cpu_sample["input_ids"]
        attention_mask = cpu_sample["attention_mask"]
        token_type_ids = cpu_sample["token_type_ids"]
        validate_generate_inputs(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            max_new_tokens=max_new_tokens,
            max_seq_len=MAX_SEQ_LEN,
        )
        if max_new_tokens < 1:
            raise ValueError("PaliGemma2 numeric probe requires max_new_tokens >= 1")

        _, vision, projector, decoder, lm_head = _paligemma_components(model)
        text_embeds = decoder.embed_tokens(input_ids).to(dtype=torch.float16)
        pixel_values_for_model = generation._move_tensor(
            pixel_values,
            device=target_device,
            dtype=torch.float16,
        )
        image_outputs = vision(pixel_values_for_model, return_dict=True)
        image_features = _image_features_from_vision_output(
            vision=vision,
            image_outputs=image_outputs,
            projector=projector,
            decoder=decoder,
        )
        scaled_image_features = scale_image_features(image_features)
        scaled_image_features_cpu = scaled_image_features.detach().to("cpu")
        merged_cpu = merge_image_features(
            input_ids=input_ids,
            inputs_embeds=text_embeds,
            image_features=scaled_image_features_cpu,
        ).detach().to("cpu")

        input_len = int(input_ids.shape[1])
        prefill_mask = generation.build_prefill_additive_attention_mask(
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            dtype=torch.float16,
        )
        cache_position = torch.arange(input_len, dtype=torch.long)
        position_ids = (cache_position + 1).unsqueeze(0)
        prefill_outputs = decoder(
            inputs_embeds=generation._move_tensor(
                merged_cpu,
                device=target_device,
                dtype=torch.float16,
            ),
            attention_mask=prefill_mask,
            position_ids=position_ids,
            use_cache=True,
            cache_position=cache_position,
        )
        prefill_last_hidden_state = (
            prefill_outputs.last_hidden_state[:, -1:, :].detach().to("cpu")
        )
        logits = lm_head(prefill_last_hidden_state).detach().to("cpu")

        forced_tokens = None
        if forced_decode_tokens is not None:
            forced_tokens = forced_decode_tokens.detach().to(
                dtype=torch.long,
                device="cpu",
            )
            if forced_tokens.ndim == 1:
                forced_tokens = forced_tokens.view(1, -1)
            if int(forced_tokens.shape[1]) < max_new_tokens:
                raise RPUBackendError(
                    "PaliGemma2 numeric probe forced_decode_tokens length "
                    f"{int(forced_tokens.shape[1])} is less than "
                    f"max_new_tokens={max_new_tokens}"
                )

        if forced_tokens is None:
            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True,
            ).to(dtype=torch.long, device="cpu")
        else:
            next_token = forced_tokens[:, 0:1]
        generated_tokens = [next_token.detach().to("cpu")]
        decode_next_token_embeddings = None
        decode_final_token_hidden = prefill_last_hidden_state
        final_logits = logits
        past_key_values = prefill_outputs.past_key_values
        decode_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    attention_mask.shape[0],
                    1,
                    dtype=attention_mask.dtype,
                    device="cpu",
                ),
            ],
            dim=1,
        )

        for decode_idx in range(max_new_tokens - 1):
            current_embeddings = decoder.embed_tokens(next_token).to(
                dtype=torch.float16
            )
            if decode_next_token_embeddings is None:
                decode_next_token_embeddings = current_embeddings.detach().to("cpu")
            current_position = input_len + decode_idx
            decode_mask = generation.build_decode_additive_attention_mask(
                attention_mask=decode_attention_mask,
                key_value_len=current_position + 1,
                dtype=torch.float16,
            )
            decode_cache_position = torch.tensor(
                [current_position],
                dtype=torch.long,
            )
            decode_outputs = decoder(
                inputs_embeds=generation._move_tensor(
                    current_embeddings,
                    device=target_device,
                    dtype=torch.float16,
                ),
                attention_mask=decode_mask,
                position_ids=(decode_cache_position + 1).unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=decode_cache_position,
            )
            past_key_values = decode_outputs.past_key_values
            decode_final_token_hidden = (
                decode_outputs.last_hidden_state[:, -1:, :].detach().to("cpu")
            )
            final_logits = lm_head(decode_final_token_hidden).detach().to("cpu")
            if forced_tokens is None:
                next_token = torch.argmax(
                    final_logits[:, -1, :],
                    dim=-1,
                    keepdim=True,
                ).to(dtype=torch.long, device="cpu")
            else:
                next_token = forced_tokens[:, decode_idx + 1:decode_idx + 2]
            generated_tokens.append(next_token.detach().to("cpu"))
            decode_attention_mask = torch.cat(
                [
                    decode_attention_mask,
                    torch.ones(
                        decode_attention_mask.shape[0],
                        1,
                        dtype=decode_attention_mask.dtype,
                        device="cpu",
                    ),
                ],
                dim=1,
            )

        if decode_next_token_embeddings is None:
            decode_next_token_embeddings = decoder.embed_tokens(next_token).to(
                dtype=torch.float16
            ).detach().to("cpu")

        return (
            {
                "scaled_image_features": scaled_image_features_cpu,
                "merged_inputs_embeds": merged_cpu,
                "decode_next_token_embeddings": decode_next_token_embeddings,
                "prefill_last_hidden_state": prefill_last_hidden_state,
                "decode_final_token_hidden": decode_final_token_hidden,
                "logits": final_logits,
            },
            torch.cat(generated_tokens, dim=1),
        )

    def _load_cpu_probe_model(self) -> Any:
        from transformers import PaliGemmaForConditionalGeneration

        source = getattr(self._hf_model, "_paligemma_source_name_or_path", None)
        if not source:
            raise RPUBackendError(
                "PaliGemmaAdapter probe requires "
                "_paligemma_source_name_or_path on the HF model"
            )
        source_path = Path(str(source))
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            str(source),
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            local_files_only=source_path.exists(),
        )
        if hasattr(model, "half"):
            model = model.half()
        model.eval()
        return model

    def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        import rpu_backend.adapters.paligemma.generation as generation

        from rpu_backend.adapters.paligemma.merge import (
            merge_image_features,
            scale_image_features,
            validate_generate_inputs,
        )
        from rpu_backend.adapters.paligemma.profile import MAX_SEQ_LEN

        if args:
            raise ValueError(
                "PaliGemmaPolicy.generate accepts keyword inputs only"
            )

        generation_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in _MODEL_INPUT_KEYS
        }
        opts = generation.validate_generate_kwargs(generation_kwargs)

        if not self._rpu_is_ready:
            raise RPUBackendError(
                "PaliGemmaPolicy.generate requires .to('rpu') first"
            )

        pixel_values = _require_model_input(kwargs, "pixel_values")
        input_ids = _require_model_input(kwargs, "input_ids")
        attention_mask = _require_model_input(kwargs, "attention_mask")
        token_type_ids = _require_model_input(kwargs, "token_type_ids")
        validate_generate_inputs(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            max_new_tokens=opts.max_new_tokens,
            max_seq_len=MAX_SEQ_LEN,
        )

        base_model, vision, projector, decoder, lm_head = _paligemma_components(
            self._hf_model
        )
        text_config = getattr(
            getattr(self._hf_model, "config", None),
            "text_config",
            None,
        )
        if text_config is None:
            text_config = getattr(
                getattr(base_model, "config", None),
                "text_config",
                None,
            )

        text_embeds = decoder.embed_tokens(input_ids.to("cpu")).to(
            dtype=torch.float16
        )
        pixel_values_rpu = generation._move_tensor(
            pixel_values,
            device="rpu",
            dtype=torch.float16,
        )
        image_outputs = vision(pixel_values_rpu, return_dict=True)
        image_features = _image_features_from_vision_output(
            vision=vision,
            image_outputs=image_outputs,
            projector=projector,
            decoder=decoder,
        )
        image_features = scale_image_features(image_features)

        self.fallback_recorder.record(
            component="paligemma_outer",
            op="placeholder_mask_creation",
            stage="prefill",
            reason="first-version Python placeholder mask",
            shape={"input_ids": list(input_ids.shape)},
            dtype=str(input_ids.dtype),
            device_from=input_ids.device.type,
            device_to="cpu",
            bytes_moved=0,
        )
        image_features_cpu = image_features.detach().to("cpu")
        self.fallback_recorder.record(
            component="paligemma_outer",
            op="masked_scatter_merge",
            stage="prefill",
            reason="first-version Python image/text merge",
            shape={
                "inputs_embeds": list(text_embeds.shape),
                "image_features": list(image_features.shape),
            },
            dtype=str(text_embeds.dtype),
            device_from=getattr(image_features, "device", torch.device("cpu")).type,
            device_to="cpu",
            bytes_moved=int(image_features.numel() * image_features.element_size()),
        )
        merged = merge_image_features(
            input_ids=input_ids,
            inputs_embeds=text_embeds,
            image_features=image_features_cpu,
        )
        merged = generation._move_tensor(
            merged,
            device="rpu",
            dtype=torch.float16,
        )

        input_len = int(input_ids.shape[1])
        prefill_mask = generation.build_prefill_additive_attention_mask(
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            dtype=torch.float16,
        )
        self.fallback_recorder.record(
            component="paligemma_outer",
            op="attention_mask_mapping",
            stage="prefill",
            reason="first-version additive attention mask",
            shape={"attention_mask": list(prefill_mask.shape)},
            dtype=str(prefill_mask.dtype),
            device_from="cpu",
            device_to="cpu",
            bytes_moved=int(prefill_mask.numel() * prefill_mask.element_size()),
        )
        cache_position = torch.arange(input_len, dtype=torch.long)
        self.fallback_recorder.record(
            component="shape_index",
            op="position_or_cache_index",
            stage="prefill",
            reason="first-version cache_position construction",
            shape={"cache_position": list(cache_position.shape)},
            dtype=str(cache_position.dtype),
            device_from="cpu",
            device_to="cpu",
            bytes_moved=int(cache_position.numel() * cache_position.element_size()),
        )
        outputs = decoder(
            inputs_embeds=merged,
            attention_mask=prefill_mask,
            use_cache=True,
            cache_position=cache_position,
        )
        logits = generation.compute_logits_cpu_fallback(
            lm_head=lm_head,
            hidden_states=outputs.last_hidden_state[:, -1:, :],
            recorder=self.fallback_recorder,
            stage="prefill",
            final_logit_softcap=getattr(
                text_config,
                "final_logit_softcapping",
                None,
            ),
            # HF PaliGemma top-level lm_head parity: record configured
            # softcap metadata, but do not apply Gemma2ForCausalLM softcap.
            apply_final_logit_softcap=False,
        )

        generated = [input_ids]
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(next_token)
        past_key_values = outputs.past_key_values
        decode_attention_mask = torch.cat(
            [
                attention_mask.to("cpu"),
                torch.ones(
                    attention_mask.shape[0],
                    1,
                    dtype=attention_mask.dtype,
                    device="cpu",
                ),
            ],
            dim=1,
        )

        for _ in range(opts.max_new_tokens - 1):
            embeds = decoder.embed_tokens(next_token.to("cpu")).to(dtype=torch.float16)
            self.fallback_recorder.record(
                component="gemma2",
                op="cpu_embed_tokens",
                stage="decode",
                reason="first-version decode embedding fallback",
                shape={
                    "input_ids": list(next_token.shape),
                    "embeds": list(embeds.shape),
                },
                dtype=str(embeds.dtype),
                device_from="cpu",
                device_to="rpu",
                bytes_moved=int(embeds.numel() * embeds.element_size()),
            )
            current_position = input_len + len(generated) - 2
            decode_mask = generation.build_decode_additive_attention_mask(
                attention_mask=decode_attention_mask,
                key_value_len=current_position + 1,
                dtype=torch.float16,
            )
            self.fallback_recorder.record(
                component="paligemma_outer",
                op="attention_mask_mapping",
                stage="decode",
                reason="first-version decode additive attention mask",
                shape={"attention_mask": list(decode_mask.shape)},
                dtype=str(decode_mask.dtype),
                device_from="cpu",
                device_to="cpu",
                bytes_moved=int(decode_mask.numel() * decode_mask.element_size()),
            )
            cache_position = torch.tensor([current_position], dtype=torch.long)
            self.fallback_recorder.record(
                component="shape_index",
                op="position_or_cache_index",
                stage="decode",
                reason="first-version decode cache_position construction",
                shape={"cache_position": list(cache_position.shape)},
                dtype=str(cache_position.dtype),
                device_from="cpu",
                device_to="cpu",
                bytes_moved=int(cache_position.numel() * cache_position.element_size()),
            )
            outputs = decoder(
                inputs_embeds=generation._move_tensor(
                    embeds,
                    device="rpu",
                    dtype=torch.float16,
                ),
                attention_mask=decode_mask,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
            )
            past_key_values = outputs.past_key_values
            logits = generation.compute_logits_cpu_fallback(
                lm_head=lm_head,
                hidden_states=outputs.last_hidden_state[:, -1:, :],
                recorder=self.fallback_recorder,
                stage="decode",
                final_logit_softcap=getattr(
                    text_config,
                    "final_logit_softcapping",
                    None,
                ),
                # HF PaliGemma top-level lm_head parity: record configured
                # softcap metadata, but do not apply Gemma2ForCausalLM softcap.
                apply_final_logit_softcap=False,
            )
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token)
            decode_attention_mask = torch.cat(
                [
                    decode_attention_mask,
                    torch.ones(
                        decode_attention_mask.shape[0],
                        1,
                        dtype=decode_attention_mask.dtype,
                        device="cpu",
                    ),
                ],
                dim=1,
            )

        return torch.cat(generated, dim=1)

    def forward_rpu(self, *args: Any, **kwargs: Any) -> Any:
        raise RPUBackendError(
            "PaliGemmaPolicy._forward_rpu requires the PaliGemma2 RPU adapter."
        )


def _cleanup_siglip_handle(
    vision_inner: Any,
    *,
    had_instance_forward: bool | None = None,
    original_forward: Any = None,
) -> None:
    if vision_inner is None:
        return
    graph_cache = getattr(vision_inner, "_rpu_siglip_graph_cache", None)
    if graph_cache is not None and hasattr(graph_cache, "clear"):
        try:
            graph_cache.clear()
        except Exception:
            pass
    finalizer = getattr(vision_inner, "_rpu_vision_handle_finalizer", None)
    handle = getattr(vision_inner, "_rpu_vision_handle", None)
    if finalizer is not None and getattr(finalizer, "alive", False):
        try:
            finalizer()
        except Exception:
            pass
    elif finalizer is None and handle is not None:
        try:
            torch.ops.rpu.siglip_destroy(handle)
        except Exception:
            pass
    if had_instance_forward is not None:
        _restore_instance_forward(
            vision_inner,
            had_instance_forward=had_instance_forward,
            original_forward=original_forward,
        )
    for attr in (
        "_rpu_cache",
        "_rpu_siglip_graph_cache",
        "_rpu_lazy_init_checked",
        "_rpu_required_attrs",
        "_rpu_vision_handle",
        "_rpu_vision_handle_finalizer",
    ):
        vars(vision_inner).pop(attr, None)


def _cleanup_gemma2_handle(decoder: Any) -> None:
    if decoder is None:
        return
    session = getattr(decoder, "_execution_session", None)
    if session is not None:
        session.shutdown(lambda: _cleanup_gemma2_handle_unlocked(decoder, strict=True))
    else:
        _cleanup_gemma2_handle_unlocked(decoder)


def _cleanup_gemma2_handle_unlocked(decoder: Any, *, strict=False) -> None:
    graph_cache = getattr(decoder, "_rpu_gemma2_graph_cache", None)
    if graph_cache is not None:
        graph_cache.begin_warmup()
        graph_cache.clear()
    finalizer = getattr(decoder, "_rpu_gemma2_handle_finalizer", None)
    handle = getattr(decoder, "_rpu_gemma2_handle", None)
    if strict:
        if handle is not None and (finalizer is None or getattr(finalizer, "alive", False)):
            torch.ops.rpu.gemma2_destroy(handle)
            if finalizer is not None:
                finalizer.detach()
    elif finalizer is not None and getattr(finalizer, "alive", False):
        try:
            finalizer()
        except Exception:
            pass
    elif finalizer is None and handle is not None:
        try:
            torch.ops.rpu.gemma2_destroy(handle)
        except Exception:
            pass

    if hasattr(decoder, "_rpu_gemma2_had_instance_forward"):
        _restore_instance_forward(
            decoder,
            had_instance_forward=bool(
                decoder._rpu_gemma2_had_instance_forward
            ),
            original_forward=getattr(
                decoder,
                "_rpu_gemma2_original_instance_forward",
                None,
            ),
        )

    for attr in (
        "_rpu_effective_num_kv_heads",
        "_rpu_gemma2_handle",
        "_rpu_gemma2_handle_finalizer",
        "_rpu_cache",
        "_rpu_lazy_init_checked",
        "_rpu_required_attrs",
        "_rpu_gemma2_had_instance_forward",
        "_rpu_gemma2_original_instance_forward",
        "_rpu_gemma2_install_ready",
        "_execution_session",
        "_planner_cost_session",
        "_rpu_gemma2_graph_cache",
        "_fmb_execution_generation",
        "_rpu_execution",
        "_rpu_last_execution_plan",
        "reconfigure_rpu_execution",
        "close_rpu_execution",
        "_rpu_attn_tp",
        "_rpu_max_seq_len",
    ):
        vars(decoder).pop(attr, None)


def _named(model: Any, method_name: str) -> Any:
    method = getattr(model, method_name, None)
    if method is None:
        return ()
    return method()


def _require_model_input(kwargs: dict[str, Any], key: str) -> Any:
    if key not in kwargs:
        raise RPUBackendError(f"PaliGemma2 generate requires {key}")
    return kwargs[key]


def _clone_sample_to_cpu(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().to("cpu")
        for key, value in sample.items()
        if key in _MODEL_INPUT_KEYS
    }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_f = left.detach().to("cpu", dtype=torch.float32).reshape(-1)
    right_f = right.detach().to("cpu", dtype=torch.float32).reshape(-1)
    if left_f.numel() == 0 or right_f.numel() == 0:
        return 1.0
    left_norm = float(left_f.norm().item())
    right_norm = float(right_f.norm().item())
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(torch.nn.functional.cosine_similarity(left_f, right_f, dim=0).item())


def _topk_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    values = logits.detach().to("cpu", dtype=torch.float32)
    values = values[:, -1, :] if values.ndim == 3 else values.reshape(1, -1)
    return torch.topk(values, k=min(k, int(values.shape[-1])), dim=-1).indices


def _topk_tokens(logits: torch.Tensor, k: int) -> list[list[int]]:
    return _topk_logits(logits, k).to(dtype=torch.long).tolist()


def _topk_overlap(
    cpu_logits: torch.Tensor,
    rpu_logits: torch.Tensor,
    k: int,
) -> float:
    cpu_topk = _topk_logits(cpu_logits, k)
    rpu_topk = _topk_logits(rpu_logits, k)
    overlaps = []
    for cpu_row, rpu_row in zip(cpu_topk, rpu_topk, strict=True):
        overlaps.append(
            len(set(cpu_row.tolist()) & set(rpu_row.tolist())) / float(k)
        )
    return float(sum(overlaps) / len(overlaps)) if overlaps else 1.0


def _eos_token_ids(model: Any) -> set[int]:
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = getattr(getattr(model, "config", None), "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {int(eos)}
    return {int(item) for item in eos if item is not None}


def _truncate_generated_at_eos(
    tokens: torch.Tensor,
    *,
    input_len: int,
    min_new_tokens: int,
    eos_token_ids: set[int],
) -> torch.Tensor:
    if not eos_token_ids or tokens.ndim != 2:
        return tokens
    stop_start = int(input_len) + max(int(min_new_tokens), 0)
    for idx in range(stop_start, int(tokens.shape[1])):
        if int(tokens[0, idx].item()) in eos_token_ids:
            return tokens[:, :idx + 1]
    return tokens


def _mask_allowed(mask: torch.Tensor) -> torch.Tensor:
    mask_cpu = mask.detach().to("cpu")
    if mask_cpu.dtype == torch.bool:
        return mask_cpu
    return torch.isfinite(mask_cpu) & (mask_cpu == 0)


def _build_hf_prefill_masks(
    *,
    config: Any,
    sample: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    from transformers.models.paligemma.modeling_paligemma import (
        create_causal_mask_mapping,
    )

    if config is None:
        raise RPUBackendError("PaliGemma2 mask probe requires model.config")
    cfg = copy.deepcopy(config)
    text_config = cfg.get_text_config()
    text_config._attn_implementation = "eager"
    if hasattr(cfg, "text_config"):
        cfg.text_config._attn_implementation = "eager"

    input_ids = sample["input_ids"]
    seq_len = int(input_ids.shape[1])
    hidden_size = int(getattr(text_config, "hidden_size", 2304))
    inputs_embeds = torch.zeros(
        int(input_ids.shape[0]),
        seq_len,
        hidden_size,
        dtype=torch.float16,
    )
    cache_position = torch.arange(seq_len, dtype=torch.long)
    mapping = create_causal_mask_mapping(
        cfg,
        inputs_embeds,
        sample["attention_mask"],
        cache_position,
        None,
        (cache_position + 1).unsqueeze(0),
        sample["token_type_ids"],
        pixel_values=sample["pixel_values"],
        is_first_iteration=True,
        use_cache=True,
    )
    if isinstance(mapping, dict):
        full = mapping.get("full_attention")
        sliding = mapping.get("sliding_attention")
        if full is None and mapping:
            full = next(iter(mapping.values()))
        if sliding is None:
            sliding = full
    else:
        full = mapping
        sliding = mapping
    if full is None or sliding is None:
        raise RPUBackendError("PaliGemma2 HF mask probe returned no masks")
    return {"full_attention": full, "sliding_attention": sliding}


def _run_spm_contract_check() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[4]
    completed = subprocess.run(
        [sys.executable, "scripts/check_component_spm_contract.py"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
