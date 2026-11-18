"""Strict CPU FP32 loading for the reviewed Base checkpoints."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

from ._manifest import CHECKPOINT_PROFILES, SOURCE_REVISION
from .source import sha256, verify_source


def preflight(checkpoint: str | Path, source: str | Path) -> dict:
    """Validate source, dependency and all checkpoint identities before model import."""
    source = verify_source(source)
    if importlib.metadata.version("transformers") not in ("5.3.0", "5.19.0"):
        raise ValueError("Qwen3-TTS requires transformers==5.3.0 or ==5.19.0")
    checkpoint = Path(checkpoint).resolve()
    # These optional HF files override the pinned tokenizer metadata.
    for name in ("tokenizer.json", "added_tokens.json", "special_tokens_map.json", "chat_template.jinja", "chat_templates"):
        if (checkpoint / name).exists():
            raise ValueError(f"Qwen3-TTS unreviewed tokenizer override: {name}")
    config_hash = sha256(checkpoint / "config.json")
    profile_name = next((name for name, profile in CHECKPOINT_PROFILES.items()
                         if profile["hashes"]["config.json"] == config_hash), None)
    if profile_name is None:
        raise ValueError("Qwen3-TTS checkpoint identity mismatch: config.json")
    profile = CHECKPOINT_PROFILES[profile_name]
    hashes = profile["hashes"]
    # Check all metadata before either large weight file is read.
    names = sorted(hashes, key=lambda name: name.endswith(".safetensors"))
    for name in names:
        expected = hashes[name]
        if sha256(checkpoint / name) != expected:
            raise ValueError(f"Qwen3-TTS checkpoint identity mismatch: {name}")
    return {"checkpoint": str(checkpoint), "source": str(source),
            "profile": profile_name, "checkpoint_revision": profile["revision"],
            "source_revision": SOURCE_REVISION}


def _import_source(source: Path):
    name = "_rhinoforge_reviewed_qwen3_tts_" + hashlib.sha256(str(source).encode()).hexdigest()[:16]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, source / "__init__.py", submodule_search_locations=[str(source)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except BaseException:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]
        raise
    finally:
        sys.dont_write_bytecode = old
    return module


def load_cpu_model(checkpoint: str | Path, source: str | Path):
    """Return the reviewed official wrapper, using CPU FP32 for every component.

    ``source`` is the prepared qwen_tts package directory. No AutoConfig/AutoModel
    registry is modified, and HF's meta/from_pretrained weight loader is not used.
    """
    identity = preflight(checkpoint, source)
    checkpoint, source = Path(identity["checkpoint"]), Path(identity["source"])
    package = _import_source(source)
    models = importlib.import_module(package.__name__ + ".core.models.modeling_qwen3_tts")
    codec_module = importlib.import_module(package.__name__ + ".core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2")
    codec_config_module = importlib.import_module(package.__name__ + ".core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2")
    processor_module = importlib.import_module(package.__name__ + ".core.models.processing_qwen3_tts")
    import torch
    from safetensors.torch import load_file
    from transformers import EncodecFeatureExtractor, Qwen2Tokenizer

    config = models.Qwen3TTSConfig.from_dict(json.loads((checkpoint / "config.json").read_text()))
    for part in (config, config.talker_config, config.talker_config.code_predictor_config):
        part._attn_implementation = "eager"
    codec_path = checkpoint / "speech_tokenizer"
    codec_config = codec_config_module.Qwen3TTSTokenizerV2Config.from_dict(json.loads((codec_path / "config.json").read_text()))
    # Ordinary CPU construction preserves nonpersistent RoPE buffers. Copying the
    # complete state strictly also avoids HF 5.3 reinitializing loaded parameters.
    with torch.device("cpu"):
        model = models.Qwen3TTSForConditionalGeneration(config).float()
        model.load_state_dict(load_file(str(checkpoint / "model.safetensors")), strict=True)
        codec_model = codec_module.Qwen3TTSTokenizerV2Model(codec_config).float()
        codec_model.load_state_dict(load_file(str(codec_path / "model.safetensors")), strict=True)
    model.eval()
    codec_model.eval()
    codec = package.Qwen3TTSTokenizer()
    codec.model, codec.config, codec.device = codec_model, codec_config, torch.device("cpu")
    codec.feature_extractor = EncodecFeatureExtractor.from_pretrained(str(codec_path), local_files_only=True)
    model.load_speech_tokenizer(codec)
    defaults = json.loads((checkpoint / "generation_config.json").read_text())
    model.load_generate_config(defaults)
    tokenizer = Qwen2Tokenizer.from_pretrained(str(checkpoint), local_files_only=True, is_local=True)
    # Preserve the upstream wrapper's regex policy without the 5.3 duplicate-kwarg bug.
    tokenizer._patch_mistral_regex(tokenizer, str(checkpoint), fix_mistral_regex=True,
                                  local_files_only=True, is_local=True)
    processor = processor_module.Qwen3TTSProcessor(tokenizer=tokenizer)
    wrapper = package.Qwen3TTSModel(model=model, processor=processor, generate_defaults=defaults)
    wrapper._rpu_checkpoint_identity = identity
    return wrapper
