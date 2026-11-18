"""Strict Pi checkpoint loading for unpatched CPU references.

This file may be loaded directly by CPU tools without importing the backend.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import inspect
import hashlib
from importlib.metadata import version
import json
import sys

import torch


def align_siglip_checkpoint_keys(policy, state_dict: dict[str, torch.Tensor], *, error_type=ValueError) -> None:
    """Adapt the known HF SigLIP container change without rewriting cached assets."""
    prefix = "model.paligemma_with_expert.paligemma.model.vision_tower."
    legacy = prefix + "vision_model."
    expected = policy.state_dict()
    moves = []
    for name, value in state_dict.items():
        if name in expected or not name.startswith(legacy):
            continue
        target = prefix + name.removeprefix(legacy)
        if target not in expected:
            raise error_type(f"unknown Pi05 SigLIP checkpoint key: {name}")
        if target in state_dict:
            raise error_type(f"duplicate Pi05 SigLIP checkpoint key: {target}")
        if value.shape != expected[target].shape:
            raise error_type(f"Pi05 SigLIP checkpoint shape mismatch: {name}")
        moves.append((name, target))
    for old, new in moves:
        state_dict[new] = state_dict.pop(old)


def materialize_gemma_rope(module) -> bool:
    from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding

    if not isinstance(module, GemmaRotaryEmbedding):
        return False
    with torch.device("cpu"):
        reference = type(module)(module.config)
    for name in ("inv_freq", "original_inv_freq"):
        value = getattr(reference, name)
        module.register_buffer(name, value.detach().clone(), persistent=False)
    return True



def restore_pi_embedding_semantics(policy):
    """LeRobot scales language rows in embed_prefix; retain the plain embedding."""
    from transformers.models.gemma import modeling_gemma
    scaled_type = getattr(modeling_gemma, "GemmaTextScaledWordEmbedding", None)
    language = policy.model.paligemma_with_expert.paligemma.model.language_model
    embedding = language.embed_tokens
    if scaled_type is None or not isinstance(embedding, scaled_type):
        return False
    plain = torch.nn.Embedding(
        embedding.num_embeddings, embedding.embedding_dim,
        padding_idx=embedding.padding_idx, max_norm=embedding.max_norm,
        norm_type=embedding.norm_type, scale_grad_by_freq=embedding.scale_grad_by_freq,
        sparse=embedding.sparse, device="meta", dtype=embedding.weight.dtype)
    plain.weight = embedding.weight
    language.embed_tokens = plain
    return True


def restore_pi_image_semantics(policy):
    """Pi image tokens are the PaliGemma projector output, whatever transformers does.

    LeRobot 0.5.1 ``embed_image`` multiplies ``get_image_features().pooler_output``
    by sqrt(text hidden) to undo the 1/sqrt(hidden) that transformers 5.3's
    PaliGemma applied. Transformers 5.19 no longer divides, so on CPU every image
    token came out sqrt(2048) ~= 45.25 times too large (the RPU SigLIP path uses
    the projector output and was unaffected). Only for unpatched CPU policies:
    this instance attribute would shadow the RPU adapter's class-level patch.
    """
    expert = policy.model.paligemma_with_expert
    for name in ("vision_tower", "multi_modal_projector"):
        if not isinstance(getattr(expert.paligemma.model, name, None), torch.nn.Module):
            raise ValueError(f"Pi CPU reference requires PaliGemma {name}")

    def embed_image(image):
        # LeRobot's dtype handling: the vision tower and projector run in FP32.
        inner = expert.paligemma.model
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        output = inner.vision_tower(image)
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        features = inner.multi_modal_projector(hidden)
        return features if features.dtype == out_dtype else features.to(out_dtype)

    expert.embed_image = embed_image
    return True


def _prepare_cpu_classes():
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
    from lerobot.policies import pi_gemma
    from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding
    for cls in (PI05Policy, PI05Pytorch, pi_gemma.PiGemmaRMSNorm, GemmaRotaryEmbedding):
        if any(name.startswith("_rpu_") and bool(value) for name, value in vars(cls).items()):
            raise RuntimeError("Pi CPU reference requires a fresh process before device class patches")
    # HF 5.19 derives this position from the cache; LeRobot 0.5.1 still passes it.
    mask = pi_gemma.create_causal_mask
    if "cache_position" not in inspect.signature(mask).parameters:
        if not getattr(mask, "_pi05_cpu_signature_bridge", False):
            def compatible_mask(*args, cache_position=None, **kwargs):
                return mask(*args, **kwargs)
            compatible_mask._pi05_cpu_signature_bridge = True
            pi_gemma.create_causal_mask = compatible_mask
    return PI05Policy


def _checkpoint_state(model_path, policy):
    from safetensors.torch import load_file
    root = Path(model_path)
    files = sorted(path for path in root.glob("model*.safetensors")
                   if path.name != "model_remapped.safetensors")
    if not files:
        remapped = root / "model_remapped.safetensors"
        if not remapped.is_file():
            raise FileNotFoundError(f"No Pi checkpoint tensors in {root}")
        files = [remapped]
    state = {}
    for path in files:
        shard = load_file(str(path), device="cpu")
        duplicates = state.keys() & shard.keys()
        if duplicates:
            raise ValueError(f"Duplicate Pi checkpoint keys: {sorted(duplicates)[:3]}")
        state.update(shard)
    fixed = policy._fix_pytorch_state_dict_keys(state, policy.config)
    result = {}
    for name, value in fixed.items():
        target = name if name.startswith("model.") else "model." + name
        if target in result:
            raise ValueError(f"Duplicate normalized Pi checkpoint key: {target}")
        result[target] = value
    return result, files


def _materialize_cpu_buffers(policy, state):
    for name, buffer in list(policy.named_buffers()):
        if buffer.device.type != "meta" or name in state:
            continue
        module_name, _, key = name.rpartition(".")
        module = policy.get_submodule(module_name)
        if key in ("inv_freq", "original_inv_freq") and materialize_gemma_rope(module):
            continue
        if key == "position_ids":
            module.register_buffer(key, torch.arange(buffer.shape[-1], dtype=buffer.dtype).expand(buffer.shape),
                                   persistent=key not in module._non_persistent_buffers_set)
            continue
        raise ValueError(f"Unknown derived Pi CPU buffer: {name}")


def load_cpu_policy(model_path, *, config=None):
    """Load the original CPU policy with complete weights and its original dtype."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401 — register config type
    import lerobot
    import transformers

    policy_type = _prepare_cpu_classes()
    config = deepcopy(config) if config is not None else PreTrainedConfig.from_pretrained(model_path)
    if config.dtype not in ("float32", "bfloat16"):
        raise ValueError("Pi CPU reference dtype must be float32 or bfloat16")
    config.device = None
    config.compile_model = False
    try:
        with torch.device("meta"):
            policy = policy_type(config)
    finally:
        config.device = "cpu"
    restore_pi_embedding_semantics(policy)
    restore_pi_image_semantics(policy)
    state, paths = _checkpoint_state(model_path, policy)
    align_siglip_checkpoint_keys(policy, state)
    target = policy.state_dict()
    # Reject before loading, including small buffers and non-Linear tensors.
    if state.keys() != target.keys():
        raise ValueError(f"Pi CPU checkpoint keys mismatch: missing={sorted(target.keys()-state.keys())[:5]}, "
                         f"unexpected={sorted(state.keys()-target.keys())[:5]}")
    for name, value in state.items():
        expected = target[name]
        if value.shape != expected.shape:
            raise ValueError(f"Pi CPU checkpoint shape mismatch: {name}")
        if expected.is_floating_point() and not value.is_floating_point():
            raise ValueError(f"Quantized Pi checkpoint requires a separate floating-point CPU reference: {name}")
        state[name] = value.to(dtype=expected.dtype)
    _materialize_cpu_buffers(policy, state)
    policy.load_state_dict(state, strict=True, assign=True)
    if any(value.device.type != "cpu" for value in list(policy.parameters()) + list(policy.buffers())):
        raise RuntimeError("Pi CPU checkpoint left unmaterialized or non-CPU tensors")
    policy.eval()
    policy._cpu_checkpoint_identity = {
        "transformers": transformers.__version__, "transformers_file": transformers.__file__,
        "lerobot": version("lerobot"), "lerobot_file": lerobot.__file__,
        "cpu_loader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "lerobot_modeling_sha256": hashlib.sha256(Path(inspect.getsourcefile(policy_type)).read_bytes()).hexdigest(),
        "dtype": config.dtype, "strict": True, "parameter_count": len(list(policy.named_parameters())),
        "image_tokens": "multi_modal_projector(vision_tower(image)), no sqrt(hidden) factor",
        "checkpoint_files": [{"path": str(path.resolve()), "size": path.stat().st_size,
                              "mtime_ns": path.stat().st_mtime_ns} for path in paths],
    }
    return policy


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401 — register config type
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    config = PreTrainedConfig.from_pretrained(args.model)
    config.dtype = args.dtype
    policy = load_cpu_policy(args.model, config=config)
    torch.manual_seed(payload["seed"])
    with torch.inference_mode():
        actions = policy.predict_action_chunk(payload["batch"], **payload["predict_kwargs"])
    if not torch.isfinite(actions).all():
        raise ValueError("Pi CPU reference returned nonfinite actions")
    torch.save(actions.detach().cpu(), args.output)
    identity = dict(policy._cpu_checkpoint_identity,
                    request_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                    seed=payload["seed"], num_steps=payload["predict_kwargs"]["num_steps"],
                    fixed_noise="noise" in payload["predict_kwargs"],
                    backend_imported="rpu_backend" in sys.modules)
    Path(args.output + ".identity.json").write_text(json.dumps(identity, indent=2) + "\n")


if __name__ == "__main__":
    main()
