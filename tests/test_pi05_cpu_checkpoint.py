"""The CPU reference uses complete weights and unpatched upstream inference."""
from pathlib import Path
import runpy

import pytest
import torch

pytest.importorskip("lerobot")
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05 import modeling_pi05
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from transformers import PaliGemmaConfig, SiglipVisionConfig
from safetensors.torch import save_file

HELPER = Path(__file__).resolve().parents[1] / "python/rpu_backend/adapters/pi05/checkpoint.py"
CPU = runpy.run_path(str(HELPER))
PREFIX = "model.paligemma_with_expert.paligemma.model.vision_tower."


@pytest.fixture
def tiny_policy(monkeypatch):
    CPU["_prepare_cpu_classes"]()
    monkeypatch.setattr(modeling_pi05, "get_gemma_config", lambda variant:
                        modeling_pi05.GemmaConfig(32, 1, 64, 2, 2, 16))
    original_mapping = modeling_pi05.CONFIG_MAPPING
    class SmallMapping:
        def __getitem__(self, name):
            if name != "paligemma":
                return original_mapping[name]
            return lambda: PaliGemmaConfig(
                vision_config=SiglipVisionConfig(hidden_size=32, intermediate_size=64,
                    num_hidden_layers=1, num_attention_heads=2, image_size=8, patch_size=4),
                projection_dim=32)
    monkeypatch.setattr(modeling_pi05, "CONFIG_MAPPING", SmallMapping())
    original_paligemma = modeling_pi05.PaliGemmaForConditionalGenerationWithPiGemma
    def small_paligemma(*, config):
        config.vision_config.projection_dim = config.text_config.hidden_size
        return original_paligemma(config=config)
    monkeypatch.setattr(modeling_pi05, "PaliGemmaForConditionalGenerationWithPiGemma", small_paligemma)
    config = PI05Config(dtype="float32", device="cpu", chunk_size=2, n_action_steps=2,
        max_action_dim=4, max_state_dim=4, image_resolution=(8, 8), num_inference_steps=2,
        input_features={"observation.images.camera": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))})
    policy = modeling_pi05.PI05Policy(config).eval()
    CPU["restore_pi_embedding_semantics"](policy)
    return policy


def checkpoint(tmp_path, policy, fault=None):
    state = {name if not name.startswith(PREFIX) or name.startswith(PREFIX+"vision_model.")
             else PREFIX+"vision_model."+name.removeprefix(PREFIX): value.detach().clone()
             for name, value in policy.state_dict().items()}
    if fault == "missing":
        state.pop(next(iter(state)))
    elif fault == "extra":
        state["model.unexpected.scalar"] = torch.zeros(1)
    elif fault == "shape":
        key = next(iter(state)); state[key] = state[key].reshape(-1)[:1]
    save_file(state, str(tmp_path/"model.safetensors"))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_real_cpu_policy_strict_load_and_predict(tiny_policy, tmp_path, dtype):
    tiny_policy.config.dtype = dtype
    tiny_policy.model.paligemma_with_expert.to_bfloat16_for_selected_params(dtype)
    checkpoint(tmp_path, tiny_policy)
    loaded = CPU["load_cpu_policy"](tmp_path, config=tiny_policy.config)
    assert loaded._cpu_checkpoint_identity["strict"] is True
    assert all(torch.equal(value, loaded.state_dict()[name]) for name, value in tiny_policy.state_dict().items())
    batch = {"observation.images.camera": torch.rand(1, 3, 8, 8),
             "observation.language.tokens": torch.tensor([[1, 2, 3, 4]]),
             "observation.language.attention_mask": torch.ones(1, 4, dtype=torch.bool),
             "observation.state": torch.zeros(1, 4)}
    tokens = batch["observation.language.tokens"]
    embedding = loaded.model.paligemma_with_expert.paligemma.model.language_model.embed_tokens
    assert type(embedding) is torch.nn.Embedding
    assert torch.equal(loaded.model.paligemma_with_expert.embed_language_tokens(tokens),
                       torch.nn.functional.embedding(tokens, embedding.weight))
    # Image tokens are the projector output for every transformers version.
    paligemma = loaded.model.paligemma_with_expert.paligemma.model
    image = torch.rand(1, 3, 8, 8) * 2 - 1
    with torch.inference_mode():
        projected = paligemma.multi_modal_projector(paligemma.vision_tower(image).last_hidden_state)
        assert torch.equal(loaded.model.paligemma_with_expert.embed_image(image), projected)
    noise = torch.randn(1, 2, 4)
    with torch.inference_mode():
        reference = tiny_policy.predict_action_chunk(batch, noise=noise.clone(), num_steps=2)
        actual = loaded.predict_action_chunk(batch, noise=noise.clone(), num_steps=2)
    assert actual.shape == reference.shape == (1, 2, 4)
    assert actual.dtype == reference.dtype == torch.float32
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()


@pytest.mark.parametrize("fault", ["missing", "extra", "shape"])
def test_incomplete_checkpoint_never_returns_partial_policy(tiny_policy, tmp_path, fault):
    checkpoint(tmp_path, tiny_policy, fault)
    with pytest.raises((ValueError, RuntimeError), match="checkpoint"):
        CPU["load_cpu_policy"](tmp_path, config=tiny_policy.config)


def test_unknown_meta_buffer_is_not_filled_with_zeros():
    model = torch.nn.Module()
    model.register_buffer("unknown", torch.empty(2, device="meta"))
    with pytest.raises(ValueError, match="Unknown derived"):
        CPU["_materialize_cpu_buffers"](model, {})


def test_cpu_reference_rejects_device_patched_classes(monkeypatch):
    monkeypatch.setattr(modeling_pi05.PI05Pytorch, "_rpu_patched_sample_actions", True, raising=False)
    with pytest.raises(RuntimeError, match="fresh process"):
        CPU["_prepare_cpu_classes"]()
