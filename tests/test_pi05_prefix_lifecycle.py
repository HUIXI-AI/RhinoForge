"""Exercise production Pi05 prepare/prefix/cache code with tiny host tensors.

Only device transfer, SigLIP, the native assembly kernel, and Graph caches are
replaced. Native view semantics are checked separately by the explicit board
suite; these tests check the real Python lifecycle and output/cache ownership.
"""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import warnings

import pytest
import torch

from rpu_backend.adapters import pi05, siglip
from rpu_backend.adapters.pi05 import runtime


class _Node:
    pass


class _HostRpuTensor:
    def __init__(self, tensor):
        self.tensor = tensor

    @property
    def device(self):
        return SimpleNamespace(type="rpu")

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def cpu(self):
        return self.tensor

    def squeeze(self, dim):
        return type(self)(self.tensor.squeeze(dim))

    def unsqueeze(self, dim):
        return type(self)(self.tensor.unsqueeze(dim))

    def contiguous(self):
        return type(self)(self.tensor.contiguous())


class _Cache:
    def __init__(self):
        self.clear()
        self.frozen = False

    def begin_warmup(self):
        self.frozen = False

    def clear(self):
        self.entries = 0
        self.replays = 0

    def call(self):
        if self.entries:
            self.replays += 1
        else:
            assert not self.frozen
            self.entries = 1

    def size(self):
        return self.entries

    def snapshot(self):
        return [SimpleNamespace(replay_count=self.replays, recapture_count=0)]

    def cache_invariant_ok(self):
        return self.entries == 1

    def freeze(self):
        self.frozen = True


@pytest.fixture
def prefix_runtime(monkeypatch):
    # Load the real patch file without importing the unrelated external model
    # stack. Its sole import-time architecture use captures the training method.
    external = ModuleType("lerobot.policies.pi05.modeling_pi05")
    external.PI05Pytorch = type("PI05Pytorch", (), {"embed_prefix": lambda *_: None})
    monkeypatch.setitem(sys.modules, external.__name__, external)
    path = Path(pi05.__file__).with_name("patches.py")
    spec = importlib.util.spec_from_file_location("_pi05_prefix_under_test", path)
    patches = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patches)

    original_to = torch.Tensor.to

    def transfer(tensor, *args, **kwargs):
        device = args[0] if args else kwargs.get("device")
        if device == "rpu" or getattr(device, "type", None) == "rpu":
            return _HostRpuTensor(tensor.clone())
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", transfer)
    calls = []

    def assemble(language, ids, image, image_start):
        assert image_start == 0
        result = language.tensor[ids.tensor.long()].clone()
        result[:image.shape[0]].copy_(image.tensor)
        calls.append((language, ids, image.tensor.is_inference()))
        return _HostRpuTensor(result)

    monkeypatch.setattr(torch.ops.rpu, "assemble_inputs_embeds", assemble, raising=False)
    monkeypatch.setattr(runtime, "_pi05_graph_runtime_profile", lambda *_: {
        "siglip_graph": True, "gemma_graph": True, "fused_denoise": True,
        "denoise_graph": True, "adarms_graph": False,
    })
    yield patches, calls


@pytest.mark.parametrize("prepare_inference", [False, True])
def test_prepare_then_changed_input_prefix_in_opposite_mode(
    prefix_runtime, monkeypatch, prepare_inference,
):
    patches, calls = prefix_runtime
    caches = [_Cache() for _ in range(3)]
    model, vision, vlm, expert = (_Node() for _ in range(4))
    vision._rpu_siglip_graph_cache = caches[0]
    vlm._rpu_gemma_graph_cache = caches[1]
    expert._rpu_adarms_graph_cache = _Cache()
    model._rpu_fused_denoise_graph_cache = caches[2]
    model.config = SimpleNamespace(num_inference_steps=1)
    model.training = False
    model._rpu_fuse_validated = True
    model._rpu_siglip_batch_n = 3
    model._pi05_prefix_assembly_ok = None
    model._pi05_prefix_assembly_enabled = True
    model._pi05_language_prefix_cache = None
    vlm.embed_tokens = torch.nn.Embedding(8, 16, dtype=torch.float16)
    with torch.no_grad():
        vlm.embed_tokens.weight.copy_(torch.arange(128).reshape(8, 16) / 128)
    model.paligemma_with_expert = SimpleNamespace(
        paligemma=SimpleNamespace(model=SimpleNamespace(
            vision_tower=vision, language_model=vlm,
        )),
        gemma_expert=SimpleNamespace(model=expert),
        embed_language_tokens=vlm.embed_tokens,
    )
    # Match a fused Graph's persistent output: allocated during prepare and
    # updated in place by the producer on all subsequent requests.
    persistent = None

    def packed_vision(_vision, images):
        nonlocal persistent
        if persistent is None:
            persistent = _HostRpuTensor(torch.empty(1, 6, 16, dtype=torch.float16))
        with torch.inference_mode():
            persistent.tensor.fill_(float(images[0]))
        return persistent

    monkeypatch.setattr(siglip, "_siglip_forward_images", packed_vision)
    masks = torch.ones(1, 2, dtype=torch.bool)
    image_masks = [torch.ones(1, dtype=torch.bool) for _ in range(3)]
    tokens = torch.tensor([[1, 2]])
    prefixes = []

    def predict(batch, *, num_steps):
        assert num_steps == 1
        for cache in caches:
            cache.call()
        prefix, pad, attention = patches._patched_embed_prefix(
            model, [batch["image"]] * 3, image_masks, batch["tokens"], masks,
        )
        assert pad.shape == attention.shape == (1, 8)
        assert pad.all() and not attention.any()
        prefixes.append(prefix)
        return prefix.cpu().float().clone()

    adapter = object.__new__(pi05.Pi05Adapter)
    adapter._rpu_is_ready = True
    adapter._lerobot_policy = SimpleNamespace(model=model, predict_action_chunk=predict)
    try:
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            with torch.inference_mode(prepare_inference):
                result = adapter.prepare_graphs({"image": 1, "tokens": tokens})
            assert result["phase"] == "READY" and result["action_bit_equal"]
            assert len(calls) == 3 and model._pi05_prefix_assembly_ok is True
            cached_language = model._pi05_language_prefix_cache["lang_rpu"]
            retained = prefixes[-1].cpu().clone()
            with torch.inference_mode(not prepare_inference), torch.no_grad():
                # Same prompt, changed image: the language cache is reusable,
                # but the image rows must be fetched from the producer anew.
                changed = predict({"image": 3, "tokens": tokens}, num_steps=1)
                assert model._pi05_language_prefix_cache["lang_rpu"] is cached_language
                assert (changed[:, :6] == 3).all()
                assert torch.equal(changed[:, 6:], retained[:, 6:])
                assert torch.equal(prefixes[-2].cpu(), retained)
                # Mutate the caller's token tensor in place: cached token
                # identity alone is insufficient, so its detached copy matters.
                tokens[0, 0] = 4
                changed_tokens = predict({"image": 3, "tokens": tokens}, num_steps=1)
                assert model._pi05_language_prefix_cache["lang_rpu"] is not cached_language
                assert not torch.equal(changed_tokens[:, 6:], changed[:, 6:])
                tokens[0, 0] = 1
                restored = predict({"image": 1, "tokens": tokens}, num_steps=1)
                assert torch.equal(restored, retained)
            cached_language = model._pi05_language_prefix_cache["lang_rpu"]
            with torch.no_grad():
                vlm.embed_tokens.weight.add_(1)
                changed_weights = predict({"image": 1, "tokens": tokens}, num_steps=1)
            assert model._pi05_language_prefix_cache["lang_rpu"] is not cached_language
            assert torch.equal(changed_weights[:, :6], retained[:, :6])
            assert not torch.equal(changed_weights[:, 6:], retained[:, 6:])
            assert torch.equal(prefixes[-2].cpu(), retained)
            assert not observed
            assert model._pi05_prefix_assembly_ok is True
            assert all(cache.size() == 1 and cache.replays == 6 for cache in caches)
            assert all(inference == prepare_inference for _, _, inference in calls)
    finally:
        runtime._PI05_PREPARED_GRAPH_PROFILES.pop(model, None)
