"""Wall-OSS-0.5 full VLA orchestration (P4 E2E) → RPU.

Stitches the P1/P2/P3 subsystems into one image+text → action-chunk pipeline, mirroring
`tests/model/wall_oss/ref_wall_oss_e2e.py` one-for-one so the per-subsystem cos-sim checks
line up:

  processor (real FLOW prompt + image)
    → vision tower (P3 `WallOssVision`, window attention) → merged [N/4, 2048]
    → assemble inputs_embeds (scatter merged into <|image_pad|> positions, CPU)
    → real 3D mRoPE position_ids (transformers `get_rope_index`, vision grid + text)
    → expert-0 prefill (P1 `WallOssLLM.forward_embeds`) → shared prefix K/V
    → expert-1 10-step Euler denoise (P2 `WallOssAction.denoise_with_mask`, explicit 2D
      mask so the arbitrary-length prefix works — needs the P4a C++ change)
    → de-normalize (per-dataset normalizer) → 26-dim action chunk.

State (proprioception) rides in the text prompt as a discretized integer string
(`use_state_string_representation=true`): the FLOW prompt below replicates wall-x
`scripts/infer_robochallenge.py::get_text_flow` (system + camera setup + observation +
instruction + `Proprioception: <discretized>` + assistant marker) so the RPU produces
robot-usable actions, not just numbers aligned to a minimal-prompt golden. The trailing
`<|action|>×chunk` placeholder of the real prompt is dropped: the prefix-KV decomposition
denoises the action chunk separately (the placeholders are causal-after the assistant
marker, so they leave the prefix K/V untouched — numerically equivalent). Single image,
batch=1.
"""
from __future__ import annotations

import json
import os
import time
import types
from collections.abc import Mapping
from numbers import Integral

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoConfig
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as M

from rpu_backend.runtime import rpu_env_bool
from rpu_backend.api._execution import (
    bind_execution_session,
    execution_serialized,
    native_execution_reconfigure,
    normalize_rpu_execution,
    resolve_component_rpu_execution,
)
from rpu_backend.runtime.execution_planner import GRAPH_COMPOSITE_CHILD
from rpu_backend.adapters.wall_oss.vision import build_wall_oss_vision
from rpu_backend.adapters.wall_oss.action import (
    _dof_mask_is_row_constant,
    build_wall_oss_action,
)
from rpu_backend.adapters.wall_oss.llm import build_wall_oss_llm
from rpu_backend.adapters.wall_oss.checkpoint_validation import (
    validate_w8_nvfp4_checkpoint_pair,
)

IMAGE_PAD_ID = 151655

WALL_OSS_TEXT_COMPONENT = "text_backbone"
WALL_OSS_VISION_COMPONENT = "vision_encoder"
WALL_OSS_ACTION_COMPONENT = "action_expert"
WALL_OSS_EXECUTION_COMPONENTS = {
    WALL_OSS_TEXT_COMPONENT: {
        "prefill": ("chunk_size", "padding_rows", "padding_budget"),
    },
    WALL_OSS_VISION_COMPONENT: {"vision": ("chunk_size",)},
    WALL_OSS_ACTION_COMPONENT: {"action": ("chunk_size",)},
}


def resolve_wall_oss_execution_components(value, *, entry_point: str):
    """Resolve one canonical Wall config into its three physical children."""
    root = normalize_rpu_execution(
        value,
        entry_point=entry_point,
        supported_components=WALL_OSS_EXECUTION_COMPONENTS,
    )
    text = resolve_component_rpu_execution(
        root,
        WALL_OSS_TEXT_COMPONENT,
        entry_point=entry_point,
        supported_components=WALL_OSS_EXECUTION_COMPONENTS,
        profile_auto={"prefill": {"chunk_size": "auto"}},
    )
    vision = resolve_component_rpu_execution(
        root,
        WALL_OSS_VISION_COMPONENT,
        entry_point=entry_point,
        supported_components=WALL_OSS_EXECUTION_COMPONENTS,
        profile_auto={"vision": {"chunk_size": "auto"}},
    )
    action = resolve_component_rpu_execution(
        root,
        WALL_OSS_ACTION_COMPONENT,
        entry_point=entry_point,
        supported_components=WALL_OSS_EXECUTION_COMPONENTS,
        profile_auto={"action": {"chunk_size": "auto"}},
    )
    return root, text, vision, action


def _action_builder_execution(text, action):
    """Give the action builder its action child plus its owned text child."""
    return normalize_rpu_execution(
        {
            "prefill": dict(text.get("prefill", {})),
            "action": dict(action.get("action", {})),
        },
        entry_point="build_wall_oss_action",
        supported={
            "prefill": ("chunk_size", "padding_rows", "padding_budget"),
            "action": ("chunk_size",),
        },
    )


# wall-x `scripts/infer_robochallenge.py::_CAM_NAME_MAPPING` (raw view key → prose name
# used in the FLOW prompt's Camera Setup / Observation lines).
_CAM_NAME_MAPPING = {
    "face_view": "front view",
    "left_wrist_view": "left wrist view",
    "right_wrist_view": "right wrist view",
    "move1_view": "move view",
    "move2_view": "move view",
    "wall_view": "wall view",
    "top_view": "top view",
    "side_view": "side view",
    "global_view": "global view",
}


def _validate_normalizer_dataset(
    normalizer: Mapping,
    normalizer_propri: Mapping,
    dataset_key: str,
) -> None:
    """Require an exact action/proprioception normalizer pair for a dataset."""
    for label, values in (
        ("normalizer_action.pth", normalizer),
        ("normalizer_propri.pth", normalizer_propri),
    ):
        if not isinstance(values, Mapping):
            raise ValueError(
                f"Wall-OSS {label} must contain a mapping, got "
                f"{type(values).__name__}"
            )
        required = (f"min.{dataset_key}", f"delta.{dataset_key}")
        missing = [key for key in required if key not in values]
        if missing:
            available = sorted(
                key[len("min."):]
                for key in values
                if isinstance(key, str) and key.startswith("min.")
            )
            raise ValueError(
                f"Wall-OSS dataset_key={dataset_key!r} is missing {missing} in "
                f"{label}; available datasets: {available}"
            )


def _positive_runtime_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(
            f"Wall-OSS {name} must be a positive integer, got {value!r}"
        )
    return int(value)


def _discretize_proprioception(norm_active: np.ndarray, state_bins: int = 256) -> str:
    """wall-x state-string discretization (`get_text_flow`, state_str=True): map the
    already-normalized active proprioception dims ([-1,1]) into `state_bins` bins and
    space-join the integer bucket indices. `norm_active`: 1-D numpy array of active dims.
    (base wall-oss-0.5 trained with 256; the LIBERO finetune uses state_bins=512.)"""
    disc = np.digitize(norm_active, bins=np.linspace(-1, 1, state_bins + 1)[:-1]) - 1
    return " ".join(map(str, disc.tolist()))


def _build_flow_prompt(instruction: str, camera_names, dataset_name: str,
                       delta_action: bool, propri_str: str) -> str:
    """Replicate wall-x `get_text_flow` (state_str=True) MINUS the trailing
    `<|action|>×chunk` (prefix-KV denoise handles the action separately). The stock
    prompt follows the public Wall-X implementation."""
    short_prompt = rpu_env_bool("RPU_WALL_OSS_SHORT_PROMPT")
    multisuite_prologue = rpu_env_bool("RPU_WALL_OSS_MULTISUITE_PROLOGUE")
    generic_prologue = rpu_env_bool("RPU_WALL_OSS_GENERIC_PROLOGUE")
    if short_prompt:
        # Bare-instruction perf-characterization prompt: chat scaffold + one vision
        # marker per camera + the instruction only (drops system prologue, observation
        # labels, "Predict the next action…", proprioception). MUST stay byte-identical
        # to ref_wall_oss_e2e.py's --short branch.
        markers = "".join(
            "<|vision_start|><|image_pad|><|vision_end|>" for _ in camera_names)
        return (f"<|im_start|>user\n{markers} {instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n")
    action_space = "Rel EEF" if delta_action else "Abs EEF"
    cam = ", ".join(_CAM_NAME_MAPPING[c] for c in camera_names)
    if multisuite_prologue and generic_prologue:
        raise ValueError(
            "RPU_WALL_OSS_MULTISUITE_PROLOGUE and "
            "RPU_WALL_OSS_GENERIC_PROLOGUE are mutually exclusive")
    if multisuite_prologue:
        # EXACT byte-replication of the wall-x run_scripts/infer_libero_multi_suite.py prologue
        # (x2r/text_utils.get_prologue_with_embodied_information; dataset libero_all →
        # Embodiment "Single Franka", Frequency 10HZ, Abs EEF). Aligns the RPU eval to the repo's
        # four-suite harness, not scripts/infer_libero.py's shorter prologue.
        prologue = (
            "<|im_start|>system\nYou are an embodied vision-language-action (VLA) model "
            "controlling the robot with language instructions.\n Embodiment: Single Franka\n"
            " Camera Setup:")
        for _c in camera_names:
            prologue += f" {_CAM_NAME_MAPPING[_c]},"
        prologue += f"\n Frequency: 10HZ\n Action Space: {action_space}\n<|im_end|>\n"
    elif generic_prologue:
        # Official wall-x LIBERO-eval prologue (`x2robot_utils.text_templates.
        # get_prologue_with_embodied_information`, used by `scripts/infer_libero.py`): a generic
        # system line + Camera Setup + Action Space, WITHOUT the Embodiment/Frequency lines of the
        # infer_robochallenge `get_text_flow` prologue below.
        prologue = (
            "<|im_start|>system\nYou are an embodied vision-language-action model "
            f"controlling a robot with language instructions.\nCamera Setup: {cam}\n"
            f"Action Space: {action_space}"
            + os.environ.get("RPU_WALL_OSS_PROLOGUE_FILLER", "")  # DIAG: length-vs-content bisection
            + "\n<|im_end|>\n")
    else:
        emb = (
            dataset_name.split("_")[-1]
        )
        prologue = (
            f"<|im_start|>system\nYou are an embodied vision-language-action (VLA) model "
            f"controlling the robot with language instructions.\n Embodiment: {emb}\n "
            f"Camera Setup: {cam},\n Frequency: 32HZ\n Action Space: {action_space}\n<|im_end|>\n"
        )
    user_request = "<|im_start|>user\nObservation:"
    for c in camera_names:
        user_request += f" {_CAM_NAME_MAPPING[c]}: <|vision_start|><|image_pad|><|vision_end|>"
    user_request += "\nInstruction:"
    text_prompt = f"\nPredict the next action in robot action.\nProprioception: {propri_str}\n"
    user_message = f"{user_request} {instruction}{text_prompt}<|im_end|>\n"
    return prologue + user_message + "<|im_start|>assistant\n"


def _build_rope_helper(config, *, use_checkpoint_tokens_per_second: bool = False):
    """Bind HF's index helpers to the temporal convention used to train Wall-OSS."""
    # The checkpoint declares 2, but wall_x training used the classic image-only
    # convention with no tokens-per-second scaling of the temporal coordinate.
    if not use_checkpoint_tokens_per_second:
        config.vision_config.tokens_per_second = 1
    rope = types.SimpleNamespace(config=config)
    rope.get_vision_position_ids = types.MethodType(
        M.Qwen2_5_VLModel.get_vision_position_ids, rope)
    rope.get_rope_index = types.MethodType(M.Qwen2_5_VLModel.get_rope_index, rope)
    return rope


def _fused_denoise_enabled() -> bool:
    os.environ.setdefault("RPU_WALL_OSS_FUSED_DENOISE", "1")
    return rpu_env_bool("RPU_WALL_OSS_FUSED_DENOISE", default=True)


def _close_wall_component(component) -> None:
    """Retire a component only after its composite has cleared every graph."""
    retire_native = getattr(component, "_retire_native", None)
    if retire_native is not None:
        retire_native()
        return
    close = getattr(component, "close", None)
    if close is not None:
        close()


class WallOssVLA:
    """Full Wall-OSS-0.5 VLA on RPU. Construct via :func:`build_wall_oss_vla`."""

    def __init__(self, *, vision, action, processor, rope_helper, normalizer,
                 normalizer_propri, dataset_key, camera_names, delta_action,
                 state_bins: int = 256,
                 normalizer_clamp: bool = True, execution_config=None,
                 component_generations=None):
        self._gc_retirement_enabled = False
        resolved_execution = resolve_wall_oss_execution_components(
            execution_config,
            entry_point="WallOssVLA",
        )
        self.vision = vision               # WallOssVision (own cache)
        self.action = action               # WallOssAction (expert-1)
        self.llm = action.llm              # WallOssLLM (expert-0; shares cache with action)
        self._processor = processor
        # Call image_processor and tokenizer directly, reproducing the
        # AutoProcessor wrapper's <|image_pad|> expansion.
        # _fast_proc_ok: None=untested, True=validated bit-identical vs the wrapper on the
        # first call, False=fell back. RPU_WALL_OSS_FAST_PROCESSOR=0 forces the wrapper.
        self._fast_processor = rpu_env_bool(
            "RPU_WALL_OSS_FAST_PROCESSOR", default=True)
        self._fast_proc_ok = None
        # Fast image preproc: replicate Qwen2VLImageProcessor (rescale+normalize+patchify)
        # in direct Torch for single or multiple images that need no resize.
        # Compare against the HF processor on the first call.
        # RPU_WALL_OSS_FAST_IMGPROC=0 forces the HF processor.
        self._fast_imgproc = rpu_env_bool(
            "RPU_WALL_OSS_FAST_IMGPROC", default=True)
        self._fast_imgproc_ok = None
        # Normalize-fold patch_embed: when fast-imgproc is active, bake the image normalize
        # (rescale/mean/std) into the patch_embed weight (vision.enable_normalize_fold) and
        # emit raw uint8 pixel_values. The device patch_embed GEMM applies the
        # normalization and projection using a homogeneous bias column.
        # RPU_WALL_OSS_PE_NORM_FOLD=0 disables this transformation.
        self._pe_norm_fold = False
        if self._fast_imgproc and rpu_env_bool(
                "RPU_WALL_OSS_PE_NORM_FOLD", default=True):
            ip = self._processor.image_processor
            self._pe_norm_fold = self.vision.enable_normalize_fold(
                ip.rescale_factor, ip.image_mean, ip.image_std)
        self._rope = rope_helper           # config-bound get_rope_index helper
        self._normalizer = normalizer            # action normalizer (de-norm output)
        self._normalizer_propri = normalizer_propri   # proprioception normalizer (state string)
        self.dataset_key = dataset_key
        self.camera_names = list(camera_names)   # FLOW prompt Camera Setup / Observation
        self.delta_action = delta_action         # FLOW prompt Action Space (Rel/Abs EEF)
        self.state_bins = state_bins             # proprioception discretization bins (256 base / 512 libero ft)
        self.action_dim = action.action_dim
        self._normalizer_clamp = bool(normalizer_clamp)
        # Memoized get_rope_index output, keyed on the execution length, vision grid,
        # and exact image-pad positions. mRoPE does not depend on ordinary
        # text/proprioception token VALUES, but it does depend on where each image run
        # appears in the prompt. Recurring prompt structures reuse one entry.
        # RPU_WALL_OSS_HOST_CACHE=0 disables this cache.
        self._rope_cache = {}
        self._host_cache = rpu_env_bool("RPU_WALL_OSS_HOST_CACHE", default=True)
        # On-device embed assembly (graph-coverage campaign): gather token embeddings
        # via torch.ops.rpu.gather_embedding (llama_gather_embedding) and scatter the
        # vision tokens with an on-device slice copy, so inputs_embeds is built on the
        # RPU instead of via the CPU F.embedding(155765x2048 table) + masked_scatter +
        # a [1,seq,2048] fp32->fp16 upload round-trip. Bit-identical (gather of fp16
        # rows == fp16-cast of the gathered fp32 rows). RPU_WALL_OSS_ONDEVICE_EMBED=0
        # forces the legacy CPU path. _embed_w_rpu is the fp16 vocab table lazily
        # uploaded to the device on first use.
        self._ondevice_embed = rpu_env_bool(
            "RPU_WALL_OSS_ONDEVICE_EMBED", default=True)
        # Fused-assemble has the vision forward return RAW window-order `merged` + stash
        # `_pending_rev` for the on-device assembler to reverse. The legacy CPU embed path
        # (ondevice_embed=0) does NOT consume `_pending_rev` — it masked_scatters `merged`
        # directly → wrong vision-token order / RPU-vs-CPU device mismatch. So fused-assemble
        # is only valid alongside the on-device embed path; disable it on the CPU path.
        if not self._ondevice_embed and getattr(self.vision, "_fused_assemble", False):
            self.vision._fused_assemble = False
        self._embed_w_rpu = None
        # Optional deployment contract installed by prepare_graphs(): the accepted
        # REAL prefix range maps to a finite set of padded prefill/denoise execution
        # profiles. Once all three caches are READY, predict() validates the request
        # before the first graph capture so an online request can never BUILD a graph.
        self._prepared_graph_profile = None
        self._closed = False
        self._execution_generation = 0
        self._component_generations = (
            {
                component: 0 for component in WALL_OSS_EXECUTION_COMPONENTS
            }
            if component_generations is None
            else {
                component: int(component_generations[component])
                for component in WALL_OSS_EXECUTION_COMPONENTS
            }
        )
        self._execution_reconfigure_journal = None
        self._rpu_last_execution_plan = {}
        self._publish_execution_views(
            resolved_execution,
            generation=0,
            component_generations=self._component_generations,
        )
        self._enable_execution_reconfigure()
        self._execution_session = bind_execution_session(
            self,
            self._rpu_execution,
            entry_point="WallOssVLA",
            supported_components=WALL_OSS_EXECUTION_COMPONENTS,
            validate=self.validate_execution_reconfigure,
            apply=self.apply_execution_reconfigure,
            rollback=self.rollback_execution_reconfigure,
            graph_mode=GRAPH_COMPOSITE_CHILD,
        )

        for leaf, component in (
            (self.llm, WALL_OSS_TEXT_COMPONENT),
            (self.vision, WALL_OSS_VISION_COMPONENT),
            (self.action, WALL_OSS_ACTION_COMPONENT),
        ):
            self._execution_session._bind_planner_owner(leaf, component)
        for child in (self.vision, self.action, self.llm):
            child._retirement_owner = self

    def _runtime_complete(self) -> bool:
        """Return whether this VLA still owns every callable RPU component."""
        if getattr(self, "_closed", None) is not False:
            return False
        try:
            vision = self.vision
            action = self.action
            llm = self.llm
        except AttributeError:
            return False
        if getattr(action, "llm", None) is not llm:
            return False
        action_dim = getattr(self, "action_dim", None)
        if (
            isinstance(action_dim, bool)
            or not isinstance(action_dim, Integral)
            or action_dim < 1
        ):
            return False

        for component in (vision, action, llm):
            if getattr(component, "_closed", None) is not False:
                return False
            if getattr(component, "_handle", None) is None:
                return False
            if (getattr(component, "_retirement_owner", None) is not self
                    or getattr(component, "_retirement_failed", None) is not None):
                return False
            graph_cache = getattr(component, "_graph_cache", None)
            if graph_cache is None or not callable(
                getattr(graph_cache, "capture", None)
            ):
                return False

        if getattr(vision, "cache", None) is None:
            return False
        if getattr(vision, "_keepalive", None) is None:
            return False
        if getattr(llm, "cache", None) is None:
            return False
        for attr in (
            "_processor",
            "_rope",
            "_normalizer",
            "_normalizer_propri",
        ):
            if getattr(self, attr, None) is None:
                return False
        return callable(getattr(self, "predict", None)) and callable(
            getattr(self, "prepare_graphs", None)
        )

    def close(self) -> None:
        """Retire all children through the shared stop-the-world session."""
        from rpu_backend.adapters.wall_oss._retirement import _close
        _close(self, (self.vision, self.action, self.llm))

    def __del__(self):
        from rpu_backend.adapters.wall_oss._retirement import _gc_close
        _gc_close(self)

    def _enable_execution_reconfigure(self) -> None:
        required = {
            "causal_decoder_enable_execution_reconfigure",
            "qwen25vl_vision_enable_execution_reconfigure",
            (
                "wall_oss_action_enable_execution_reconfigure"
                if self.action._fused
                else "causal_decoder_enable_execution_reconfigure"
            ),
        }
        missing = sorted(name for name in required if not hasattr(torch.ops.rpu, name))
        if missing:
            raise RuntimeError(
                "Wall-OSS binary lacks native execution-reconfigure op(s): "
                + ", ".join(missing)
            )
        torch.ops.rpu.causal_decoder_enable_execution_reconfigure(
            int(self.llm._handle)
        )
        torch.ops.rpu.qwen25vl_vision_enable_execution_reconfigure(
            int(self.vision._handle)
        )
        if self.action._fused:
            torch.ops.rpu.wall_oss_action_enable_execution_reconfigure(
                int(self.action._handle)
            )
        else:
            torch.ops.rpu.causal_decoder_enable_execution_reconfigure(
                int(self.action._handle)
            )

    @staticmethod
    def _native_chunk(config, stage: str) -> int:
        chunk = config.get(stage, {}).get("chunk_size", "auto")
        return 0 if chunk == "auto" else int(chunk)

    def _resolve_execution(self, value, *, entry_point: str):
        return resolve_wall_oss_execution_components(
            value, entry_point=entry_point
        )

    def _publish_execution_views(
        self, resolved, *, generation: int, component_generations=None,
    ) -> None:
        root, text, vision, action = resolved
        if component_generations is not None:
            self._component_generations = dict(component_generations)
        self._rpu_execution = root
        self._text_execution = text
        self._vision_execution = vision
        self._action_execution = action
        self._execution_generation = int(generation)
        self._execution_components = {
            WALL_OSS_TEXT_COMPONENT: text,
            WALL_OSS_VISION_COMPONENT: vision,
            WALL_OSS_ACTION_COMPONENT: action,
        }
        children = (
            (self.llm, WALL_OSS_TEXT_COMPONENT, text),
            (self.vision, WALL_OSS_VISION_COMPONENT, vision),
            (self.action, WALL_OSS_ACTION_COMPONENT, action),
        )
        for child, component, config in children:
            state = vars(child)
            state["_rpu_execution"] = config
            state["_fmb_execution_component_id"] = component
            state["_fmb_execution_component_config"] = config
            state["_fmb_execution_generation"] = int(
                self._component_generations[component]
            )
        self.vision._execution_chunk_size = vision.get(
            "vision", {}
        ).get("chunk_size", "auto")
        self.action._execution_chunk_size = action.get(
            "action", {}
        ).get("chunk_size", "auto")

    @staticmethod
    def _reset_graph_owner(cache, *, label: str) -> None:
        cache.begin_warmup()
        cache.clear()
        if not cache.cache_invariant_ok():
            raise RuntimeError(
                f"Wall-OSS {label} GraphCache invariant failed during reconfigure"
            )

    def _reset_component_graphs(self, changed) -> None:
        seen = set()

        def reset(cache, label):
            if cache is not None and id(cache) not in seen:
                seen.add(id(cache))
                self._reset_graph_owner(cache, label=label)

        if WALL_OSS_VISION_COMPONENT in changed:
            reset(self.vision._graph_cache, "vision")
        if WALL_OSS_TEXT_COMPONENT in changed:
            reset(self.llm._graph_cache, "prefill")
            # The action graph identity includes the resulting prefix geometry.
            reset(self.action._graph_cache, "action")
        elif WALL_OSS_ACTION_COMPONENT in changed:
            reset(self.action._graph_cache, "action")
        self._prepared_graph_profile = None
        for component in changed:
            self._rpu_last_execution_plan.pop(component, None)

    def validate_execution_reconfigure(self, execution_config) -> None:
        if self._closed:
            raise RuntimeError("Wall-OSS runtime is closed")
        _root, text, _vision, _action = self._resolve_execution(
            execution_config,
            entry_point="WallOssVLA.reconfigure",
        )
        text_chunk = self._native_chunk(text, "prefill")
        if text_chunk > int(self.llm._prefill_chunk_size_cap):
            raise ValueError(
                "Wall-OSS prefill chunk_size exceeds the instance envelope: "
                f"requested={text_chunk}, "
                f"ceiling={self.llm._prefill_chunk_size_cap}"
            )
        required = {
            "execution_reconfigure_begin",
            "execution_reconfigure_commit",
            "execution_reconfigure_abort",
            "execution_reconfigure_abort_attempt",
            "causal_decoder_stage_chunk_size_override",
            "qwen25vl_vision_stage_chunk_size",
            (
                "wall_oss_action_stage_chunk_size"
                if self.action._fused
                else "causal_decoder_stage_chunk_size_override"
            ),
        }
        missing = sorted(name for name in required if not hasattr(torch.ops.rpu, name))
        if missing:
            raise RuntimeError(
                "Wall-OSS binary lacks native execution-reconfigure op(s): "
                + ", ".join(missing)
            )

    def _apply_execution_state(
        self,
        resolved,
        *,
        generation: int,
        component_generations=None,
        force_components=(),
    ) -> None:
        root, text, vision, action = resolved
        current = (
            self._text_execution,
            self._vision_execution,
            self._action_execution,
        )
        requested = (text, vision, action)
        components = (
            WALL_OSS_TEXT_COMPONENT,
            WALL_OSS_VISION_COMPONENT,
            WALL_OSS_ACTION_COMPONENT,
        )
        changed = {
            component
            for component, before, after in zip(components, current, requested)
            if before != after
        }
        changed.update(force_components)
        if component_generations is None:
            component_generations = dict(self._component_generations)
            for component in changed:
                component_generations[component] += 1

        if changed:
            with native_execution_reconfigure(
                torch.ops.rpu, journal=self._execution_reconfigure_journal,
            ) as token:
                if WALL_OSS_TEXT_COMPONENT in changed:
                    torch.ops.rpu.causal_decoder_stage_chunk_size_override(
                        int(self.llm._handle), int(token),
                        self._native_chunk(text, "prefill"),
                    )
                if WALL_OSS_VISION_COMPONENT in changed:
                    torch.ops.rpu.qwen25vl_vision_stage_chunk_size(
                        int(self.vision._handle), int(token),
                        self._native_chunk(vision, "vision"),
                    )
                if WALL_OSS_ACTION_COMPONENT in changed:
                    action_chunk = self._native_chunk(action, "action")
                    if self.action._fused:
                        torch.ops.rpu.wall_oss_action_stage_chunk_size(
                            int(self.action._handle), int(token), action_chunk,
                        )
                    else:
                        torch.ops.rpu.causal_decoder_stage_chunk_size_override(
                            int(self.action._handle), int(token), action_chunk,
                        )

        if self._execution_reconfigure_journal is not None:
            self._execution_reconfigure_journal["mutation_started"] = True
        self._reset_component_graphs(changed)
        self._rpu_last_execution_plan = {}
        self._publish_execution_views(
            (root, text, vision, action),
            generation=generation,
            component_generations=component_generations,
        )

    def apply_execution_reconfigure(
        self, old_config, new_config, generation: int, *, force_rebuild=False,
    ) -> None:
        self._execution_reconfigure_journal = {"mutation_started": False}
        old_resolved = self._resolve_execution(
            old_config, entry_point="WallOssVLA.reconfigure rollback"
        )
        new_resolved = self._resolve_execution(
            new_config, entry_point="WallOssVLA.reconfigure"
        )
        changed = {
            component
            for component, old, new in zip(
                WALL_OSS_EXECUTION_COMPONENTS,
                old_resolved[1:],
                new_resolved[1:],
            )
            if old != new
        }
        if force_rebuild:
            changed.update(WALL_OSS_EXECUTION_COMPONENTS)
        self._execution_reconfigure_journal = {
            "mutation_started": False,
            "resolved": old_resolved,
            "generation": self._execution_generation,
            "component_generations": dict(self._component_generations),
            "changed": changed,
        }
        self._apply_execution_state(
            new_resolved, generation=generation, force_components=changed,
        )
        # Retain undo state through the shared session's config publication.

    def rollback_execution_reconfigure(
        self, old_config, _new_config, generation: int,
    ) -> None:
        journal = self._execution_reconfigure_journal
        if journal is not None and not journal.get("mutation_started", False):
            self._execution_reconfigure_journal = None
            return
        if journal is None:
            resolved = self._resolve_execution(
                old_config, entry_point="WallOssVLA.reconfigure rollback"
            )
            changed = {
                component
                for component, current, old in zip(
                    WALL_OSS_EXECUTION_COMPONENTS,
                    (
                        self._text_execution,
                        self._vision_execution,
                        self._action_execution,
                    ),
                    resolved[1:],
                )
                if current != old
            }
            component_generations = dict(self._component_generations)
        else:
            resolved = journal["resolved"]
            generation = journal["generation"]
            changed = journal["changed"]
            component_generations = journal["component_generations"]
        self._apply_execution_state(
            resolved,
            generation=generation,
            component_generations=component_generations,
            force_components=changed,
        )
        self._execution_reconfigure_journal = None

    def _collect_component_plan_receipts(self) -> None:
        receipts = {}
        for component, child in (
            (WALL_OSS_VISION_COMPONENT, self.vision),
            (WALL_OSS_TEXT_COMPONENT, self.llm),
            (WALL_OSS_ACTION_COMPONENT, self.action),
        ):
            receipt = getattr(child, "_rpu_last_execution_plan", None)
            if not isinstance(receipt, Mapping):
                raise RuntimeError(
                    f"Wall-OSS {component} forward produced no execution receipt"
                )
            expected = int(self._component_generations[component])
            if (
                receipt.get("component") != component
                or int(receipt.get("generation", -1)) != expected
                or receipt.get("dry_forward_agreement") is not True
            ):
                raise RuntimeError(
                    f"Wall-OSS {component} execution receipt disagrees with "
                    f"generation {expected}"
                )
            receipts[component] = dict(receipt)
        self._rpu_last_execution_plan = receipts

    @staticmethod
    def _grid_key(grid: torch.Tensor) -> tuple[int, ...]:
        return tuple(int(x) for x in grid.reshape(-1).tolist())

    @staticmethod
    def _prefix_range_bounds(prefix_length_range) -> tuple[int, int]:
        if (not isinstance(prefix_length_range, (list, tuple))
                or len(prefix_length_range) != 2):
            raise ValueError(
                "prefix_length_range must be a two-item (min, max) sequence")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in prefix_length_range
        ):
            raise ValueError(
                "prefix_length_range entries must be integers, got "
                f"{prefix_length_range!r}"
            )
        lo, hi = (int(prefix_length_range[0]), int(prefix_length_range[1]))
        if lo <= 0 or hi < lo:
            raise ValueError(
                f"invalid prefix_length_range=({lo}, {hi}); require 0 < min <= max")
        return lo, hi

    def _prefill_execution_plan(
        self, real_len: int, *, horizon: int
    ) -> tuple[int, int]:
        """Use the public decoder planner with reserved action rows."""
        real_len = int(real_len)
        return self.llm.prefill_execution_plan(
            real_len, reserve_rows=int(horizon)
        )

    def _prefix_execution_profiles(
        self, prefix_length_range, *, horizon: int
    ) -> tuple[dict, ...]:
        """Map an inclusive REAL-prefix range to its finite execution profiles."""
        lo, hi = self._prefix_range_bounds(prefix_length_range)
        grouped = {}
        for real_len in range(lo, hi + 1):
            execution_len, chunk_size = self._prefill_execution_plan(
                real_len, horizon=int(horizon)
            )
            key = (int(execution_len), int(chunk_size))
            if key not in grouped:
                grouped[key] = [real_len, real_len]
            else:
                grouped[key][1] = real_len
        return tuple(
            {
                "execution_len": execution_len,
                "chunk_size": chunk_size,
                "real_min": bounds[0],
                "real_max": bounds[1],
            }
            for (execution_len, chunk_size), bounds in sorted(grouped.items())
        )

    def _validate_prepared_request(
        self,
        *,
        real_len: int,
        grid: torch.Tensor,
        horizon: int,
        num_steps: int,
        row_constant: bool,
    ) -> None:
        profile = self._prepared_graph_profile
        if profile is None:
            return
        current_denoise_mode = {
            "fused": bool(self.action._fused),
            "fp32_tail": bool(self.action._action_fp32_tail),
            "unroll": bool(self.action._denoise_unroll),
        }
        if current_denoise_mode != profile["denoise_mode"]:
            raise RuntimeError(
                "Wall-OSS denoise execution mode changed after graph preparation: "
                f"current={current_denoise_mode}, "
                f"prepared={profile['denoise_mode']}. No online graph BUILD was "
                "attempted; restore the prepared environment or restart.")
        if os.environ.get("WALL_OSS_PROFILE_PREFIX"):
            raise RuntimeError(
                "WALL_OSS_PROFILE_PREFIX is a diagnostic truncation and cannot be "
                "used with a frozen Wall-OSS graph profile")
        lo, hi = profile["prefix_length_range"]
        if not lo <= int(real_len) <= hi:
            raise RuntimeError(
                "Wall-OSS prefix length is outside the prepared graph range: "
                f"real_len={real_len}, prepared=[{lo}, {hi}]. No online graph "
                "BUILD was attempted; restart and prepare a wider range.")
        execution_len, chunk_size = self._prefill_execution_plan(
            int(real_len), horizon=int(horizon)
        )
        if (int(execution_len), int(chunk_size)) not in profile["execution_keys"]:
            raise RuntimeError(
                "Wall-OSS request maps to an unprepared execution profile: "
                f"real_len={real_len}, execution=({execution_len}, {chunk_size})")
        if self._grid_key(grid) != profile["grid_key"]:
            raise RuntimeError(
                "Wall-OSS image geometry differs from the prepared Vision graph: "
                f"grid={self._grid_key(grid)}, prepared={profile['grid_key']}")
        if int(horizon) != profile["horizon"]:
            raise RuntimeError(
                f"Wall-OSS horizon={horizon} differs from prepared "
                f"horizon={profile['horizon']}")
        if int(num_steps) != profile["num_steps"]:
            raise RuntimeError(
                f"Wall-OSS num_steps={num_steps} differs from prepared "
                f"num_steps={profile['num_steps']}")
        if bool(row_constant) != profile["row_constant"]:
            raise RuntimeError(
                "Wall-OSS dof_mask execution mode differs from the prepared graph "
                f"(row_constant={row_constant}, prepared={profile['row_constant']})")

    @execution_serialized
    @torch.no_grad()
    def prepare_graphs(
        self,
        *,
        images,
        instruction: str,
        proprioception: torch.Tensor,
        state_mask: torch.Tensor,
        noise: torch.Tensor,
        dof_mask: torch.Tensor,
        prefix_length_range,
        num_steps: int = 10,
    ) -> dict:
        """Prebuild a finite Wall-OSS execution envelope, then disable online BUILD.

        ``prefix_length_range`` is inclusive and refers to the real processor output
        length after image-token expansion, before execution padding. The supplied
        observation first materializes every subsystem's persistent SPM. Those
        throwaway graphs are cleared, then the same observation warms the stable
        processor/Vision/glue path. Remaining prefill/denoise profiles are built
        with shape-equivalent synthetic tensors; each profile is replayed A→B→A
        across its real-length bounds before READY is entered.
        """
        if self._prepared_graph_profile is not None:
            raise RuntimeError(
                "Wall-OSS graphs are already prepared and frozen for this policy")
        images = list(images) if isinstance(images, (list, tuple)) else [images]
        if len(images) != len(self.camera_names):
            raise ValueError(
                f"got {len(images)} images for {len(self.camera_names)} cameras "
                f"{self.camera_names}")
        if os.environ.get("WALL_OSS_PROFILE_PREFIX"):
            raise RuntimeError(
                "unset WALL_OSS_PROFILE_PREFIX before preparing production graphs")

        H = int(noise.size(1))
        self.action.validate_execution_horizon(H)
        lo, hi = self._prefix_range_bounds(prefix_length_range)
        steps = _positive_runtime_int("num_steps", num_steps)
        row_constant = _dof_mask_is_row_constant(dof_mask)

        # CPU-only admission check first: a bad example must fail before any graph
        # gets built. This also validates the exact image-processor geometry.
        propri_str = self._proprioception_string(proprioception, state_mask)
        text = _build_flow_prompt(
            instruction, self.camera_names, self.dataset_key,
            self.delta_action, propri_str)
        example_ids, _, example_grid = self._process_inputs(text, images)
        example_real_len = int(example_ids.size(1))
        if not lo <= example_real_len <= hi:
            raise ValueError(
                "prepare_graphs example is outside prefix_length_range: "
                f"example_real_len={example_real_len}, range=[{lo}, {hi}]")

        caches = (
            ("vision", self.vision._graph_cache),
            ("prefill", self.llm._graph_cache),
            ("denoise", self.action._graph_cache),
        )
        for _, cache in caches:
            cache.begin_warmup()
            cache.clear()
        try:
            # Native chunk planning depends on process-wide persistent SPM. Run
            # one complete throwaway request so Vision, Prefill, and Action have
            # all materialized before READY keys are derived. Rebuild every graph
            # below under that final footprint; a graph recorded earlier could
            # otherwise carry a plan from a temporarily roomier allocator state.
            self.predict(
                images, instruction, proprioception, state_mask, noise, dof_mask,
                num_steps=steps, debug=False)
            for _, cache in caches:
                cache.clear()
                cache.begin_warmup()
            torch.ops.rpu.spm_alloc_reset_temporary()

            profiles = self._prefix_execution_profiles(
                prefix_length_range, horizon=H
            )
            prefill_profiles = len(profiles)
            denoise_profiles = len(profiles)
            prefill_capacity = self.llm._graph_cache.max_entries()
            denoise_capacity = self.action._graph_cache.max_entries()
            if (prefill_profiles > prefill_capacity
                    or denoise_profiles > denoise_capacity):
                raise ValueError(
                    "prefix_length_range creates more execution profiles than "
                    "the Prefill/Denoise GraphCache capacity: "
                    f"prefill={prefill_profiles}/{prefill_capacity}, "
                    f"denoise={denoise_profiles}/{denoise_capacity}, "
                    f"range=[{lo}, {hi}]. Narrow the range or use coarser padding.")
            execution_keys = frozenset(
                (p["execution_len"], p["chunk_size"]) for p in profiles)
            profile = {
                "prefix_length_range": (lo, hi),
                "execution_keys": execution_keys,
                "grid_key": self._grid_key(example_grid),
                "horizon": H,
                "num_steps": steps,
                "row_constant": row_constant,


                "denoise_mode": {
                    "fused": bool(self.action._fused),
                    "fp32_tail": bool(self.action._action_fp32_tail),
                    "unroll": bool(self.action._denoise_unroll),
                },
            }
            self._prepared_graph_profile = profile

            # Warm the exact production seam, including image preprocessing,
            # Vision, embed assembly, mRoPE, denoise and result readback.
            warm_out = self.predict(
                images, instruction, proprioception, state_mask, noise, dof_mask,
                num_steps=steps, debug=False)
            vision_graph_count = self.vision._graph_cache.size()
            if vision_graph_count < 1:
                raise RuntimeError(
                    "Wall-OSS Vision prewarm produced no graph entries")

            for item in profiles:
                P = item["execution_len"]
                reps = [item["real_min"], item["real_max"], item["real_min"]]
                if item["real_min"] == item["real_max"]:
                    reps = [item["real_min"], item["real_min"]]
                for rep_index, real_len in enumerate(reps):
                    # The previous Denoise graph's temporary arena is dead. Reset
                    # at this subsystem/profile boundary so a new Prefill BUILD
                    # cannot stack its SPM plan on top of that high-water mark.
                    torch.ops.rpu.spm_alloc_reset_temporary()
                    inputs_embeds = torch.zeros(
                        (1, P, self.llm.hidden_size), dtype=torch.float16)
                    pos_prefix = torch.arange(
                        P, dtype=torch.int32)[:, None].expand(P, 3).contiguous()
                    if real_len < P:
                        # Match runtime padding: rows beyond the REAL prefix
                        # repeat its last position. Bucket min/max therefore
                        # exercise different same-P Prefill position/RoPE
                        # contents, not only different Denoise masks.
                        pos_prefix[real_len:] = pos_prefix[real_len - 1]
                    self.llm.forward_embeds(
                        inputs_embeds, pos_prefix, logical_len=real_len,
                        reserve_rows=H,
                    )
                    pos_action = torch.arange(
                        real_len, real_len + H, dtype=torch.int32
                    )[:, None].expand(H, 3).contiguous()
                    self.action.denoise_with_mask(
                        P, noise, dof_mask, pos_action, num_steps=steps,
                        real_prefix_len=real_len, debug=False)

            expected = {
                # Mixed fixed camera geometries may legitimately split into
                # multiple Vision subgroups/signatures; the full grid is still
                # locked by admission and synthetic profile probes never touch it.
                "vision": vision_graph_count,
                "prefill": len(profiles),
                "denoise": denoise_profiles,
            }
            actual = {name: cache.size() for name, cache in caches}
            if actual != expected:
                raise RuntimeError(
                    f"Wall-OSS graph prewarm produced {actual}, expected {expected}")
            for _, cache in caches:
                cache.freeze()
            # One final real observation under strict READY proves that all three
            # lookups replay (rather than relying only on snapshot validation) and
            # restores real position/RoPE host-cache contents after the synthetic
            # A→B→A probes. The first timed request is therefore steady-state too.
            ready_out = self.predict(
                images, instruction, proprioception, state_mask, noise, dof_mask,
                num_steps=steps, debug=False)
            ready_counts = {name: cache.size() for name, cache in caches}
            if ready_counts != actual:
                raise RuntimeError(
                    "Wall-OSS READY replay changed graph counts: "
                    f"before={actual}, after={ready_counts}")
            if not torch.equal(warm_out["x_norm"], ready_out["x_norm"]):
                max_abs = float(
                    (warm_out["x_norm"].float()
                     - ready_out["x_norm"].float()).abs().max())
                raise RuntimeError(
                    "Wall-OSS READY replay differs from the real WARMING build "
                    f"after synthetic A→B→A probes (x_norm max_abs={max_abs:.7g})")
        except Exception:
            # No partially READY deployment: make every cache build-capable again
            # so a caller may fix the range/input and retry in the same process.
            for _, cache in caches:
                cache.begin_warmup()
            self._prepared_graph_profile = None
            raise

        return {
            "prefix_length_range": (lo, hi),
            "example_real_prefix_len": example_real_len,
            "execution_profiles": tuple(dict(item) for item in profiles),
            "graph_counts": actual,
            "grid": profile["grid_key"],
            "horizon": H,
            "num_steps": steps,
            "row_constant": row_constant,


            "denoise_mode": dict(profile["denoise_mode"]),
            "phase": "READY",
            "ready_replay_validated": True,
        }

    def _fast_image_preproc(self, images):
        """Direct-torch Qwen2VLImageProcessor for the wall_oss deployment case: N equal-size
        images already a multiple of patch*merge (multi-camera, fixed shape, no resize). All
        N processed as ONE batched CPU tensor — rescale+normalize+patchify in a single pass,
        no per-image loop, no cat — so the HF processor's multi-image reshape/stack/cat (the
        dominant 3img-HEAD `aten::cat`) is skipped. Returns (pixel_values [N*n_patch, 1176]
        fp32, grid [N,3] int) or None when the assumption breaks (mixed size / needs resize /
        unexpected layout → np.stack or the resize check bails → caller uses the HF processor).
        The N images are stacked in order, so the output matches the HF concatenated layout
        (validated bit-exact on first call by _image_preproc); N=1 is the same path."""
        if not images:
            return None
        ip = self._processor.image_processor
        try:
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
            ps, ms, tp = ip.patch_size, ip.merge_size, ip.temporal_patch_size
            arrs = np.stack([np.asarray(im) for im in images])          # [N,H,W,3] (raises if mixed-shape)
            if arrs.ndim != 4 or arrs.shape[3] != 3:
                return None
            N, H, W = arrs.shape[0], int(arrs.shape[1]), int(arrs.shape[2])
            size = getattr(ip, "size", {}) or {}
            min_px = size.get("shortest_edge", getattr(ip, "min_pixels", 56 * 56))
            max_px = size.get("longest_edge", getattr(ip, "max_pixels", 28 * 28 * 1280))
            rH, rW = smart_resize(H, W, factor=ps * ms, min_pixels=min_px, max_pixels=max_px)
            if (rH, rW) != (H, W):
                return None                                # resize needed → HF bicubic parity risk
            gh, gw = rH // ps, rW // ps
            # Normalize-fold path: patchify RAW uint8 (no fp32 cast, no normalize passes) —
            # the normalize is baked into the patch_embed weight (vision.enable_normalize_fold).
            # Otherwise rescale+normalize in fp32 here (the HF-equivalent output).
            if self._pe_norm_fold:
                b = torch.from_numpy(arrs).permute(0, 3, 1, 2)             # [N,3,H,W] u8 view
            else:
                mean = torch.tensor(ip.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
                std = torch.tensor(ip.image_std, dtype=torch.float32).view(1, 3, 1, 1)
                b = torch.from_numpy(arrs).permute(0, 3, 1, 2).to(torch.float32)  # [N,3,H,W] (fresh, owned)
                b.mul_(ip.rescale_factor).sub_(mean).div_(std)          # rescale+normalize IN-PLACE (no intermediates)
            # Temporal frames are IDENTICAL for a still image, so broadcast the temporal
            # axis with expand (stride-0, NO copy) instead of repeat (which materializes a
            # 2x buffer). b is contiguous post-normalize → the view+expand stays a view and
            # final reshape().contiguous() is the only materialization. Equal
            # temporal frames make this equivalent to repeat.
            p = b.view(N, 1, 3, gh // ms, ms, ps, gw // ms, ms, ps)     # split H/W, temporal=1 (view)
            p = p.expand(N, tp, 3, gh // ms, ms, ps, gw // ms, ms, ps)  # temporal 1->tp, stride-0 (no copy)
            p = p.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)                    # matches HF transpose (N replaces t=1)
            pv = p.reshape(N * gh * gw, 3 * tp * ps * ps).contiguous()  # single materialization; per-image blocks adjacent
            grid = torch.tensor([[1, gh, gw]] * N, dtype=torch.long)
            return pv, grid
        except Exception:
            return None

    def _reconstruct_norm(self, pv_u8: torch.Tensor) -> torch.Tensor:
        """uint8 [seq, K] (normalize-fold path) → the normalized fp32 [seq, K] the HF
        Qwen2VLImageProcessor would produce, in the [ch, t, ps, ps] column layout.
        Validation-only (compares the fast-path layout/constants vs the real processor)."""
        ip = self._processor.image_processor
        ps, tp = ip.patch_size, ip.temporal_patch_size
        K = pv_u8.size(1)
        mean = torch.tensor(ip.image_mean, dtype=torch.float32)
        std = torch.tensor(ip.image_std, dtype=torch.float32)
        scale = (ip.rescale_factor / std).view(3, 1, 1, 1).expand(3, tp, ps, ps).reshape(K)
        shift = (mean / std).view(3, 1, 1, 1).expand(3, tp, ps, ps).reshape(K)
        return pv_u8.float() * scale - shift

    def _image_preproc(self, images):
        """(pixel_values fp32, grid) via the fast torch path when valid, else the HF
        image_processor. Self-validates the fast path bit-exact on the first call."""
        ip = self._processor.image_processor
        if self._fast_imgproc and self._fast_imgproc_ok is not False:
            fast = self._fast_image_preproc(images)
            if fast is not None:
                pv, grid = fast
                if self._fast_imgproc_ok is None:          # one-time self-validation
                    ref = ip(images=images, return_tensors="pt")
                    ref_pv = ref["pixel_values"].float()
                    # uint8 fold path → reconstruct the normalized fp32 the HF processor
                    # would produce, to compare layouts/constants on the same footing.
                    cmp_pv = self._reconstruct_norm(pv) if self._pe_norm_fold else pv
                    ok = (cmp_pv.shape == ref_pv.shape
                          and torch.allclose(cmp_pv, ref_pv, atol=1e-3, rtol=1e-3)
                          and bool(torch.equal(grid, ref["image_grid_thw"])))
                    self._fast_imgproc_ok = bool(ok)
                    if not ok:
                        import warnings
                        warnings.warn("wall_oss fast-imgproc mismatch vs Qwen2VLImageProcessor "
                                      "(transformers drift?); falling back for this session.")
                        return ref_pv, ref["image_grid_thw"]
                return pv, grid
        img = ip(images=images, return_tensors="pt")
        return img["pixel_values"].float(), img["image_grid_thw"]

    def _assemble_inputs_embeds_ondevice(self, input_ids: torch.Tensor,
                                         merged: torch.Tensor,
                                         padded_len: int) -> torch.Tensor:
        """Build inputs_embeds [1, padded_len, H] on the RPU device (no CPU embed-table
        gather, no masked_scatter, no [1,seq,H] fp32 upload round-trip, no torch.zeros
        pad). All coherency flushes are CPU<->RPU-boundary only: ids/merged uploads.

        input_ids [1, real_len] CPU int; merged [N_img, H] (vision output); padded_len
        >= real_len (v16 KV-insert alignment). Gather token embeddings on-device — the
        ids are pre-padded so the gather emits [padded_len, H] directly (pad rows gather
        vocab[0], a valid fp16 row that the action 2D mask discards) — then overwrite the
        contiguous <|image_pad|> run(s) with the vision tokens via a device slice copy.
        Multi-image = N contiguous runs filled from merged in image order."""
        if self._embed_w_rpu is None:
            # One-time: fp16 vocab table on the device (gather reads it via glarge,
            # so >4GB DDR is fine). _embed_w is [vocab, H] fp32 CPU.
            self._embed_w_rpu = self.llm._embed_w.to(torch.float16).contiguous().to("rpu")
        ids_flat = input_ids.reshape(-1)
        real_len = ids_flat.size(0)
        if padded_len > real_len:                       # tiny CPU pad → padded gather output
            ids_flat = F.pad(ids_flat, (0, padded_len - real_len))
        ids_rpu = ids_flat.to(torch.int32).to("rpu").contiguous()
        merged_rpu = merged.to(torch.float16).to("rpu").contiguous()
        pos = (input_ids.reshape(-1) == IMAGE_PAD_ID).nonzero(as_tuple=False).reshape(-1).tolist()
        n_img = len(pos)
        # Fused path: the vision tower returned RAW window-order merged + stashed the
        # window-reverse (RPU_WALL_OSS_FUSED_ASSEMBLE). One op gathers token embeddings
        # AND per-run reverse-gathers merged into the image positions — folding the
        # separate vision reverse-gather + the per-run scatter copies. `merged_rpu` here
        # is window-order; rev_idx maps spatial→window row.
        rev = getattr(self.vision, "_pending_rev", None)
        if rev is not None and n_img > 0:
            rev_idx, mu_list = rev
            runs = []                                   # (token_start, len) per image run
            i = 0
            while i < n_img:
                j = i
                while j + 1 < n_img and pos[j + 1] == pos[j] + 1:
                    j += 1
                runs.append((pos[i], j - i + 1)); i = j + 1
            run_lengths = [run[1] for run in runs]
            if run_lengths != list(mu_list):
                raise RuntimeError(
                    f"fused assemble: run lens {run_lengths} != vision "
                    f"merged-per-image {list(mu_list)}"
                )
            runs_t = torch.tensor(runs, dtype=torch.long)               # CPU [n_runs, 2]
            out = torch.ops.rpu.assemble_inputs_embeds_rev(
                self._embed_w_rpu, ids_rpu, merged_rpu, rev_idx, runs_t)  # [padded_len, H] fp16 RPU
            return out.unsqueeze(0)
        contiguous = n_img > 0 and (pos[-1] - pos[0] + 1 == n_img)
        if contiguous:
            # Single contiguous image block (single image): one fully on-device op —
            # gather + scatter + out all in SPM, no host memcpy, no flushes.
            out = torch.ops.rpu.assemble_inputs_embeds(
                self._embed_w_rpu, ids_rpu, merged_rpu, pos[0])         # [padded_len, H] fp16 RPU
            return out.unsqueeze(0)
        # Multi-image fallback: gather then fill each contiguous run with a device slice copy.
        embeds = torch.ops.rpu.gather_embedding(self._embed_w_rpu, ids_rpu)  # [padded_len, H] fp16 RPU
        off = 0
        i = 0
        while i < n_img:
            j = i
            while j + 1 < n_img and pos[j + 1] == pos[j] + 1:
                j += 1
            run_len = j - i + 1
            embeds[pos[i]:pos[i] + run_len].copy_(merged_rpu[off:off + run_len])
            off += run_len
            i = j + 1
        if off != merged_rpu.size(0):
            raise RuntimeError(
                f"placed {off} vision tokens != merged rows {merged_rpu.size(0)}"
            )
        return embeds.unsqueeze(0)                                          # [1, padded_len, H] fp16 RPU

    def _proprioception_string(self, proprioception: torch.Tensor,
                               state_mask: torch.Tensor) -> str:
        """raw proprioception [.,.,D] → normalizer_propri normalize ([-1,1]) → select active
        dims (state_mask) → discretized integer string (wall-x get_text_flow, state_str)."""
        pmin = self._normalizer_propri[f"min.{self.dataset_key}"]
        pdelta = self._normalizer_propri[f"delta.{self.dataset_key}"]
        raw = proprioception.float().reshape(-1)              # [D]
        norm = 2.0 * (raw - pmin) / pdelta - 1.0
        if self._normalizer_clamp:
            norm = torch.clamp(norm, -1.0, 1.0)
        active = norm[state_mask.reshape(-1).bool()]          # [D_active]
        return _discretize_proprioception(active.cpu().numpy(), self.state_bins)


    def _process_inputs(self, text: str, images):
        """text + images → (input_ids [1,seq], pixel_values [N_patch,1176] fp32, grid [N,3]).

        Equivalent to `self._processor(text=[text], images=images)` but skips the
        AutoProcessor.__call__ wrapper overhead by calling image_processor + tokenizer
        directly and replicating its <|image_pad|> expansion (transformers
        `processing_qwen2_5_vl.py::__call__`: each image_token → grid.prod()//merge_size²
        copies). Numerically identical (same sub-processors). On the FIRST call we also run
        the real wrapper and assert input_ids parity, so a transformers version that changes
        the expansion auto-falls-back for the session instead of producing wrong tokens."""
        proc = self._processor
        if not self._fast_processor or self._fast_proc_ok is False:
            inputs = proc(text=[text], images=images, return_tensors="pt")
            return inputs["input_ids"], inputs["pixel_values"].float(), inputs["image_grid_thw"]
        # HF uses this literal as its temporary expansion sentinel. Preserve its
        # behavior for the pathological case where user text already contains it.
        if "<|placeholder|>" in text:
            inputs = proc(text=[text], images=images, return_tensors="pt")
            return inputs["input_ids"], inputs["pixel_values"].float(), inputs["image_grid_thw"]
        ip = proc.image_processor
        _pi = rpu_env_bool("RPU_WALL_OSS_INSTRUMENT")
        _pt0 = time.perf_counter()
        pixel_values, grid = self._image_preproc(images)         # fast torch preproc (self-validated)
        _imgproc_ms = (time.perf_counter() - _pt0) * 1000; _pt0 = time.perf_counter()
        merge_len = ip.merge_size ** 2
        input_ids = proc.tokenizer([text], return_tensors="pt")["input_ids"]
        image_pos = (input_ids[0] == proc.image_token_id).nonzero(
            as_tuple=False).reshape(-1)
        if image_pos.numel() != grid.size(0):
            import warnings
            self._fast_proc_ok = False
            warnings.warn("wall_oss fast-processor image marker count mismatch vs image grid; "
                          "falling back to the wrapper for this session.")
            ref = proc(text=[text], images=images, return_tensors="pt")
            return ref["input_ids"], ref["pixel_values"].float(), ref["image_grid_thw"]
        repeats = torch.ones(input_ids.size(1), dtype=torch.long)
        repeats[image_pos] = grid.prod(dim=1).div(
            merge_len, rounding_mode="floor").to(dtype=torch.long)
        input_ids = input_ids.repeat_interleave(repeats, dim=1)
        if self._fast_proc_ok is None:                           # one-time self-validation
            ref = proc(text=[text], images=images, return_tensors="pt")
            self._fast_proc_ok = (input_ids.shape == ref["input_ids"].shape
                                  and bool(torch.equal(input_ids, ref["input_ids"])))
            if not self._fast_proc_ok:
                import warnings
                warnings.warn("wall_oss fast-processor input_ids mismatch vs AutoProcessor "
                              "(transformers drift?); falling back to the wrapper for this session.")
                return ref["input_ids"], ref["pixel_values"].float(), ref["image_grid_thw"]
        if _pi:
            print(f"[INSTR proc] imgproc={_imgproc_ms:.2f}  tokenize+expand={(time.perf_counter()-_pt0)*1000:.2f}")
        return input_ids, pixel_values, grid

    def _rope_index_cached(self, input_ids: torch.Tensor, grid: torch.Tensor,
                           real_len: int, padded_len: int):
        """Memoized mRoPE position ids → (pos_prefix [3,1,real_len], pos_prefix_2d
        [padded_len,3]). get_rope_index reads only the special-token (image/vision)
        positions + grid + length to assign positions — never ordinary
        text/proprioception token VALUES. Keyed on the exact image-pad layout so live
        prompt-recipe changes with the same length/grid cannot collide. padded_len is
        also semantic because it determines the returned 2D tensor's shape.
        RPU_WALL_OSS_HOST_CACHE=0 disables."""
        image_pad_mask = input_ids == IMAGE_PAD_ID
        image_pad_positions = tuple(
            image_pad_mask.reshape(-1).nonzero(as_tuple=False).reshape(-1).tolist()
        )
        key = (
            int(real_len),
            int(padded_len),
            tuple(grid.flatten().tolist()),
            image_pad_positions,
        )
        if self._host_cache and key in self._rope_cache:
            return self._rope_cache[key]
        mm_token_type_ids = image_pad_mask.int()
        attn = torch.ones_like(input_ids)
        pos_prefix, _ = self._rope.get_rope_index(
            input_ids, mm_token_type_ids, image_grid_thw=grid, attention_mask=attn)  # [3,1,real_len]
        pos_prefix_2d = pos_prefix.squeeze(1).transpose(0, 1).contiguous()           # [real_len, 3]
        if padded_len != real_len:
            npad = padded_len - real_len
            pos_prefix_2d = torch.cat(
                [pos_prefix_2d, pos_prefix_2d[-1:].expand(npad, 3)], dim=0).contiguous()
        if self._host_cache:
            self._rope_cache[key] = (pos_prefix, pos_prefix_2d)
        return pos_prefix, pos_prefix_2d

    @execution_serialized
    @torch.no_grad()
    def predict(self, image, instruction: str, proprioception: torch.Tensor,
                state_mask: torch.Tensor, noise: torch.Tensor,
                dof_mask: torch.Tensor, num_steps: int = 10, debug: bool = False) -> dict:
        """image: PIL.Image or list of PIL.Image (one per camera in camera_names);
        instruction: task string; proprioception/state_mask: raw robot state + active-dim
        mask ([D] or [1,1,D] CPU fp32); noise/dof_mask: [1,H,action_dim].

        debug: when False (production / perf), omit the merged/input-embedding diagnostics,
        the prefix_last device→CPU readback, and full per-step action trajectories. The
        final-normed last_hidden is NOT on the VLA critical path (the action denoiser reads
        per-layer K/V), so reading it back every frame is pure waste + a forced host↔device
        sync. The final action/x_norm are always returned. Set True for diagnostic tests.
        """
        # 1. processor — replicate the wall-x FLOW prompt (state as discretized string);
        #    one <|image_pad|> per camera is expanded to grid//merge^2 image tokens.
        images = list(image) if isinstance(image, (list, tuple)) else [image]
        if len(images) != len(self.camera_names):
            raise ValueError(
                f"got {len(images)} images for {len(self.camera_names)} cameras "
                f"{self.camera_names}"
            )
        steps = _positive_runtime_int("num_steps", num_steps)
        # Host-region instrumentation (RPU_WALL_OSS_INSTRUMENT=1): perf_counter walls
        # per region, profiler-free (vs torch profile which inflates per-op). Zero cost off.
        _instr = rpu_env_bool("RPU_WALL_OSS_INSTRUMENT")
        _pt = {}; _t0 = [time.perf_counter()]
        def _mk(name):
            if _instr:
                now = time.perf_counter(); _pt[name] = _pt.get(name, 0.0) + (now - _t0[0]) * 1000; _t0[0] = now
        propri_str = self._proprioception_string(proprioception, state_mask)
        text = _build_flow_prompt(instruction, self.camera_names, self.dataset_key,
                                  self.delta_action, propri_str)
        _mk("1_propri+prompt")
        input_ids, pixel_values, grid = self._process_inputs(text, images)
        n_img = int((input_ids == IMAGE_PAD_ID).sum())
        _mk("2_process_inputs(imgproc+tok)")

        # Profiling-only (WALL_OSS_PROFILE_PREFIX): truncate the prefix to a target
        # length for a clean 16-aligned v16-KV-insert profile (e.g. 320 = 3·64 vision +
        # 128 text). Image tokens precede the trailing text, so only trailing text is
        # dropped. The resulting action is meaningless — do NOT use with correctness.
        # Applied to input_ids up front so the embed assembly sees the final length.
        _ppfx = os.environ.get("WALL_OSS_PROFILE_PREFIX")
        if _ppfx:
            _tgt = int(_ppfx)
            if int((input_ids[:, :_tgt] == IMAGE_PAD_ID).sum()) != n_img:
                raise RuntimeError(
                    f"WALL_OSS_PROFILE_PREFIX={_tgt} would drop image tokens"
                )
            input_ids = input_ids[:, :_tgt]

        real_len = int(input_ids.size(1))
        H = int(noise.size(1))
        self.action.validate_execution_horizon(H)
        row_constant = _dof_mask_is_row_constant(dof_mask)
        self._validate_prepared_request(
            real_len=real_len, grid=grid, horizon=H,
            num_steps=steps, row_constant=row_constant)

        # Prefill and action share one cache. Reject an impossible logical
        # prefix before Vision; the final padded plan is resolved after Vision
        # has established its persistent SPM footprint.
        if real_len + H > int(self.llm.cache.max_seq_len):
            raise ValueError(
                "Wall-OSS logical prefix plus action horizon exceeds the "
                f"shared KV cache: real_len={real_len}, horizon={H}, "
                f"max_seq_len={self.llm.cache.max_seq_len}"
            )

        # 2. Vision starts a new subsystem lifetime. Reset before its eager
        # patch-embed path; Vision resets once more after patch_embed/gather at
        # the exact fused-graph boundary. Prepared-mode admission deliberately
        # runs first so an invalid request cannot mutate allocator state.
        torch.ops.rpu.spm_alloc_reset_temporary()
        merged = self.vision.forward(pixel_values, grid)
        _mk("3_vision.forward(wrap+hw)")
        if merged.size(0) != n_img:
            raise RuntimeError(
                f"vision tokens {merged.size(0)} != <|image_pad|> count {n_img}"
            )

        # Choose the execution length before embedding assembly. Prefixes that fit the
        # single-chunk cap retain 16-token KV-insert alignment. The shared
        # planner owns larger prefixes. Padding K/V is masked out in
        # the action 2D mask, and pos_action continues from the REAL prefix.
        # Free the vision graph's temporary SPM before the embed-assembly op. The vision
        # output (`merged`) is already read out, so the encoder/merger Temps are dead — but
        # a cached GraphCache graph leaves t_end at its peak (it doesn't reset, to keep baked
        # offsets for replay). The on-device gather_embedding (an immediate op) would then
        # bump on top of that peak: at 3-image batched vision (768 patches) the peak is high
        # enough that the [padded_len, H] gather output no longer fits → SpmAllocator OOM.
        # Resetting here (the documented vision→prefill subsystem boundary) makes the gather
        # allocate from t_end=0. Immediate (outside any capture) → plain reset; persistent
        # (rope tables / biases above p_start) is untouched; the next frame's vision replay
        # re-uses its baked [0,peak] region as before.
        torch.ops.rpu.spm_alloc_reset_temporary()
        padded_len, _ = self._prefill_execution_plan(
            real_len, horizon=H
        )
        _mk("4_glue:reset+pad")

        # 3. assemble inputs_embeds [1, padded_len, H] — scatter vision into <|image_pad|>.
        #    On-device (default): padded gather + slice-copy on the RPU (no CPU embed-table
        #    gather / masked_scatter / fp32 upload / torch.zeros pad). Legacy CPU path under =0.
        if self._ondevice_embed:
            inputs_embeds = self._assemble_inputs_embeds_ondevice(input_ids, merged, padded_len)
        else:
            embed = F.embedding(input_ids.cpu(), self.llm._embed_w).float()    # [1, real_len, 2048]
            vmask = (input_ids == IMAGE_PAD_ID).unsqueeze(-1).expand_as(embed)
            # merged is on RPU (the vision merger runs on-device); embed is CPU here, so move
            # device too — `.to(dtype)` alone left a CPU/RPU mismatch in masked_scatter.
            inputs_embeds = embed.masked_scatter(vmask, merged.to(device=embed.device, dtype=embed.dtype))
            if padded_len != real_len:
                inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, padded_len - real_len))

        _mk("5_glue:assemble_embeds")
        # 4-5. real 3D mRoPE position_ids (vision grid t/h/w, text increments) → padded
        #      [padded_len, 3]. Memoized on length/grid/image-pad layout (see
        #      _rope_index_cached); pad rows masked → positions arbitrary.
        pos_prefix, pos_prefix_2d = self._rope_index_cached(input_ids, grid, real_len, padded_len)
        _mk("6_glue:rope_index")

        # 6. expert-0 prefill → shared cache prefix K/V at [0, padded_len). cache.position = padded_len.
        prefix_last = self.llm.forward_embeds(
            inputs_embeds, pos_prefix_2d, logical_len=real_len,
            reserve_rows=H,
        )
        _mk("7_prefill.forward_embeds(wrap+hw)")
        P = self.llm.cache.position
        if P != padded_len:
            raise RuntimeError(
                f"prefill cache position {P} != execution length {padded_len}"
            )

        # 7. action positions continue after the REAL prefix (text-like → 3 axes equal). Computed
        # from the un-padded pos_prefix so the action RoPE is unaffected by the padding slots.
        next_pos = int(pos_prefix.max()) + 1
        pos_action = torch.arange(next_pos, next_pos + H, dtype=torch.int32)[:, None] \
            .expand(H, 3).contiguous()
        _mk("8_glue:pos_action")

        # 8. expert-1 masked denoise. P (padded) = insert slot + mask width + cache.position;
        # real_len = the prefix the action actually attends to (mask -inf on [real_len, P)).
        out = self.action.denoise_with_mask(
            P, noise, dof_mask, pos_action, num_steps=steps,
            real_prefix_len=real_len, debug=debug)
        _mk("9_denoise.with_mask(wrap+hw)")

        # 8. de-normalize (per-dataset): action = (x+1)/2 * delta + min.
        mn = self._normalizer[f"min.{self.dataset_key}"]
        dl = self._normalizer[f"delta.{self.dataset_key}"]
        action = (out["x_norm"] + 1) / 2 * dl + mn
        _mk("10_postproc(denorm)")
        if _instr:
            tot = sum(_pt.values())
            print("[INSTR predict] " + "  ".join(f"{k}={v:.2f}" for k, v in _pt.items())
                  + f"  | host_sum={tot:.2f}ms")

        self._collect_component_plan_receipts()

        # merged is RAW WINDOW-ORDER on the fused-assemble path (its reverse-gather is
        # folded into assemble_inputs_embeds_rev). Un-permute it for the p4b diagnostic
        # so the comparison lands in the model-semantic RASTER order the golden stores.
        # Window order is an internal permutation gated on RPU_WALL_OSS_FUSED_ASSEMBLE,
        # so asserting against it would tie a golden comparison to a runtime switch.
        # `rev_idx` maps spatial -> window row, so merged[rev_idx] is spatial order.
        # Debug-only; the production path is untouched.
        merged_diag = None
        if debug:
            merged_diag = merged.float().cpu()
            _rev = getattr(self.vision, "_pending_rev", None)
            if _rev is not None:
                merged_diag = merged_diag[_rev[0].cpu().long()]

        return {
            "action": action, "x_norm": out["x_norm"], "xs": out["xs"], "vs": out["vs"],
            # inputs_embeds/merged are diagnostics (p4b); skip in production so the
            # returned dict doesn't pin the device tensors each frame. merged may be a
            # device tensor (vision device-merged) → read back to CPU for the diagnostic.
            "merged": merged_diag,
            "inputs_embeds": inputs_embeds if debug else None, "pos_prefix": pos_prefix,
            "prefix_last": prefix_last.float().cpu() if debug else None, "input_ids": input_ids,
            "prefix_len": P, "real_prefix_len": real_len,
            "dt": out["dt"], "dataset_key": self.dataset_key,


            "text": text, "propri_str": propri_str,
        }


def build_wall_oss_vla(
    ckpt_dir: str | None = None,
    *,
    dataset_key: str = "berkeley_autolab_ur5",
    camera_names=("face_view",),
    delta_action: bool = False,
    state_bins: int = 256,
    max_seq_len: int = 2048,
    w8a16: bool = False,
    w4a16: bool = False,
    nvfp4a16: bool = False,
    w8_nvfp4a16: bool = False,
    nvfp4_wint4_scope: bool = False,
    fp16_ckpt_dir: str | None = None,
    rpu_execution=None,
) -> WallOssVLA:
    """Build the full VLA: vision tower + expert-0 prefill + expert-1 denoiser (sharing
    one RPUCache) + Qwen2.5-VL processor + action/proprioception normalizers.

    Args:
        ckpt_dir: Wall-OSS-0.5 checkpoint directory.
        dataset_key: exact per-robot normalizer key (e.g.
            "berkeley_autolab_ur5"). It must exist in both action and
            proprioception sidecars and also drives the FLOW prompt Embodiment tag.
        camera_names: ordered raw view keys (FLOW prompt Camera Setup / Observation); one
            <|image_pad|> emitted per camera. Default single front view.
        delta_action: FLOW prompt Action Space ("Rel EEF" if True else "Abs EEF").
        state_bins: proprioception state-string discretization bins (base wall-oss-0.5 =
            256; the LIBERO finetune trained with 512).
        max_seq_len: RoPE table + shared KV cache capacity (prefix + horizon).
        w8a16: load expert-0/expert-1 decoder weights and vision transformer
            block weights from the W8A16 checkpoint. CPU patch_embed/merger and
            action preprocessing/proj_back stay sourced from ``fp16_ckpt_dir``.
        w8_nvfp4a16: internal validation-only action hybrid: load Vision and
            expert-0 from the W8A16 ``ckpt_dir``, then quantize only
            expert-1/action to NVFP4 from ``fp16_ckpt_dir``. It improves the raw
            all-decoder result but still fails the rc7 strict per-token prefix
            gate, so it must not be exposed as a release/public precision mode.
        nvfp4_wint4_scope: load W8 Vision and expert-0 ``down_proj`` from the
            W8A16 ``ckpt_dir``; quantize the remaining expert-0 projections and
            all expert-1 projections to NVFP4 from ``fp16_ckpt_dir``. This is the
            packaged NVFP4 composition and exactly matches the WINT4A16 op scope.
        fp16_ckpt_dir: source checkpoint for processor/normalizers and CPU glue
            for quantized modes.
    """
    from rpu_backend.api._execution import _require_execution_process_safe
    _require_execution_process_safe()
    execution_config, text_execution, vision_execution, action_execution = (
        resolve_wall_oss_execution_components(
            rpu_execution,
            entry_point="build_wall_oss_vla",
        )
    )
    action_builder_execution = _action_builder_execution(
        text_execution, action_execution
    )
    if int(state_bins) < 2:   # mirror WallOssPolicy.from_checkpoint — fail here, not deep in np.digitize
        raise ValueError(f"state_bins must be >= 2, got {state_bins}")
    selected_precision = [
        name
        for name, enabled in (
            ("w8a16", w8a16),
            ("w4a16", w4a16),
            ("nvfp4a16", nvfp4a16),
            ("w8_nvfp4a16", w8_nvfp4a16),
            ("nvfp4_wint4_scope", nvfp4_wint4_scope),
        )
        if enabled
    ]
    if len(selected_precision) > 1:
        raise ValueError(
            "Wall-OSS precision modes are mutually exclusive, got "
            f"{selected_precision}"
        )
    # Resolve and validate every checkpoint identity before mutating the
    # process-wide runtime profile. A caller that catches a bad-pair error must
    # not inherit half-applied selectors in the same process.
    from rpu_backend.model_registry import model_path
    if ckpt_dir is None:
        if w4a16:
            default_model = "wall-oss-0.5-w4a16"
        elif w8a16 or w8_nvfp4a16 or nvfp4_wint4_scope:
            default_model = "wall-oss-0.5-w8a16"
        else:
            default_model = "wall-oss-0.5"
        ckpt_dir = str(model_path(default_model))
    if fp16_ckpt_dir is None:
        fp16_ckpt_dir = str(model_path("wall-oss-0.5"))
    if w8_nvfp4a16 or nvfp4_wint4_scope:
        validate_w8_nvfp4_checkpoint_pair(ckpt_dir, fp16_ckpt_dir)
    aux_ckpt_dir = (
        fp16_ckpt_dir
        if (w8a16 or w4a16 or nvfp4a16 or w8_nvfp4a16
            or nvfp4_wint4_scope)
        else ckpt_dir
    )
    # Wall-OSS keeps attention on all eight cores. Every residual reduction is
    # scheduled by the shared ring launcher; there is no model-scoped route.
    os.environ.setdefault("RPU_WALL_OSS_ATTN_TP8", "1")
    # Warm graph replay skips the body op-stream walk after refreshing current
    # inputs outside the body loop. Wall-OSS opts into this shared capability.
    os.environ.setdefault("RPU_WALL_OSS_FAST_REPLAY", "1")
    # The common Graph executor may omit redundant kernel scans only when its
    # replay-state and mutable-input guards allow it.
    os.environ.setdefault("RPU_FASTREPLAY_SKIP_SYNC", "1")
    # Full-body replay can reuse prepared SPM and weight-preload state. Vision
    # retains setup where its post_fn still reads the layout.
    os.environ.setdefault("RPU_DEEP_FAST_REPLAY", "1")
    # Fuse vision window reversal with token gathering and image-slot assembly.
    # Mixed-size groups retain the separate assembly path.
    os.environ.setdefault("RPU_WALL_OSS_FUSED_ASSEMBLE", "1")
    # Broadcast ViT 2D-RoPE tables into SPM for per-token lookup.
    os.environ.setdefault("RPU_WALL_OSS_VISION_ROPE_SPM", "1")
    # Include the vision merger in the graph through the base-class post_fn;
    # body-only replay preserves the post_fn's layout setup.
    os.environ.setdefault("RPU_WALL_OSS_VISION_FUSED_MERGER", "1")
    #  • RPU_WALL_OSS_PARTIAL_MROPE — required Qwen2.5-VL path: host-bake contiguous
    #    T/H/W frequency chunks and stream them through partial_mrope. The legacy C++
    #    launcher uses Qwen3-style THW interleaving and is incorrect for image positions,
    #    so an explicit `0` fails fast. Covers prefill and fused denoise.
    os.environ.setdefault("RPU_WALL_OSS_PARTIAL_MROPE", "1")
    # Validate every CPU-owned checkpoint artifact before any RPU model build or
    # irreversible weight transform. A dataset typo must never select an arbitrary
    # first normalizer after several GB of weights have already been constructed.
    normalizer = torch.load(
        f"{aux_ckpt_dir}/normalizer_action.pth",
        map_location="cpu",
        weights_only=True,
    )
    normalizer_propri = torch.load(
        f"{aux_ckpt_dir}/normalizer_propri.pth",
        map_location="cpu",
        weights_only=True,
    )
    _validate_normalizer_dataset(normalizer, normalizer_propri, dataset_key)
    processor = AutoProcessor.from_pretrained(
        aux_ckpt_dir, **({}))
    config = AutoConfig.from_pretrained(aux_ckpt_dir)
    flow_time_scale = 1.0
    rope = _build_rope_helper(
        config, use_checkpoint_tokens_per_second=False)

    fused = _fused_denoise_enabled()
    llm = action = vision = None
    try:
        if nvfp4_wint4_scope:
            llm = build_wall_oss_llm(
                fp16_ckpt_dir, expert=0, max_seq_len=max_seq_len,
                nvfp4a16=True, w8_down_ckpt_dir=ckpt_dir,
                execution_config=text_execution)
            action = build_wall_oss_action(
                fp16_ckpt_dir, llm=llm, max_seq_len=max_seq_len,
                nvfp4a16=True, fp16_ckpt_dir=fp16_ckpt_dir, fused=fused,
                execution_config=action_builder_execution)
        elif w8_nvfp4a16:
            llm = build_wall_oss_llm(
                ckpt_dir, expert=0, max_seq_len=max_seq_len, w8a16=True,
                execution_config=text_execution)
            action = build_wall_oss_action(
                fp16_ckpt_dir, llm=llm, max_seq_len=max_seq_len,
                nvfp4a16=True, fp16_ckpt_dir=fp16_ckpt_dir, fused=fused,
                execution_config=action_builder_execution)
        else:
            action = build_wall_oss_action(
                ckpt_dir, max_seq_len=max_seq_len,
                w8a16=w8a16, w4a16=w4a16, nvfp4a16=nvfp4a16,
                fp16_ckpt_dir=fp16_ckpt_dir, fused=fused,


                flow_time_scale=flow_time_scale,
                execution_config=action_builder_execution)  # expert-0 + expert-1
            llm = action.llm
        if w8_nvfp4a16 or nvfp4_wint4_scope:
            vision = build_wall_oss_vision(
                ckpt_dir, window=True, w8a16=True,
                fp16_ckpt_dir=fp16_ckpt_dir,
                execution_config=vision_execution)
        elif w4a16 or nvfp4a16:
            # Vision tower stays INT8 (loaded from the w8a16 ckpt) — the w4a16
            # ckpt's vision is fp16 (converter quantizes decoder only), and an
            # identical int8 vision keeps the decoder quantization A/B isolated.
            vision_ckpt = str(model_path("wall-oss-0.5-w8a16"))
            vision = build_wall_oss_vision(
                vision_ckpt, window=True, w8a16=True,
                fp16_ckpt_dir=fp16_ckpt_dir,
                execution_config=vision_execution)
        else:
            vision = build_wall_oss_vision(
                ckpt_dir, window=True, w8a16=w8a16,
                fp16_ckpt_dir=fp16_ckpt_dir,
                execution_config=vision_execution)
        runtime = WallOssVLA(
            vision=vision, action=action, processor=processor,
            rope_helper=rope, normalizer=normalizer,
            normalizer_propri=normalizer_propri, dataset_key=dataset_key,
            camera_names=camera_names, delta_action=delta_action,
            state_bins=state_bins,

            normalizer_clamp=True,
            execution_config=execution_config,
        )
        runtime._gc_retirement_enabled = True
        return runtime
    except BaseException as error:
        from rpu_backend.adapters.wall_oss._retirement import _cleanup_build_failure, _clear_graphs
        components = tuple(component for component in (vision, action, llm)
                           if component is not None)
        def cleanup():
            _clear_graphs(components)
            for component in components:
                _close_wall_component(component)
            for component in components:
                component._closed = True
        _cleanup_build_failure(error, components, cleanup)
        raise
