"""Controlled Janus image-to-token wiring; explicit CPU/RPU hybrid execution.

The caller supplies the frozen upstream model class and config. No remote code
is downloaded or executed by this adapter. Public model admission stays closed
until the complete policy has passed R1 acceptance.
"""
import threading
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .attention import adapt_attention, RPUProjection
from .vision import RPUPatchConv, RPUVisionPatch, RPUVisionAttention, install_rpu_linears
from .memory import checkpoint_bytes, admit_memory


def load_frontend(model_class, config, checkpoint):
    from accelerate import init_empty_weights
    from .checkpoint import CheckpointStore
    checkpoint_bytes(checkpoint)
    disk_config = json.loads((Path(checkpoint) / "config.json").read_text())
    # Reconstruct the math configuration from the admitted checkpoint, not caller overrides.
    config = type(config).from_dict(disk_config)
    config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"
    if (model_class.__name__ != "Qwen2_5_VLForConditionalGenerationForJanusVLN"
            or config.image_token_id != 151655 or config.vision_config.spatial_merge_size != 2
            or config.vision_start_token_id != 151652 or config.video_token_id != 151656):
        raise ValueError("the supplied upstream model/config does not match Janus")
    if config._attn_implementation != "sdpa":
        raise ValueError("the CPU frontend requires upstream SDPA attention")
    with init_empty_weights(include_buffers=False):
        model = model_class(config)
    model.model.layers = nn.ModuleList()
    model.vggt.depth_head = None
    model.vggt.point_head = None
    store = CheckpointStore(checkpoint)
    for prefix, module in (
        ("visual", model.visual), ("vggt.aggregator", model.vggt.aggregator),
        ("merger", model.merger), ("model.embed_tokens", model.model.embed_tokens),
        ("lm_head", model.lm_head),
    ):
        print(f"Janus frontend loading {prefix}", flush=True)
        for name, parameter in list(module.named_parameters()):
            key = prefix + "." + name
            value = store.get_tensor(key)
            if value.shape != parameter.shape:
                raise ValueError(f"frontend tensor mismatch: {key}")
            parent_name, _, leaf = name.rpartition(".")
            parent = module.get_submodule(parent_name) if parent_name else module
            setattr(parent, leaf, nn.Parameter(value.half(), requires_grad=False))
        # Nonpersistent normalization/position buffers were initialized on CPU.
        for name, buffer in module.named_buffers():
            if buffer.device.type == "meta":
                raise ValueError(f"unmaterialized frontend buffer: {prefix}.{name}")
        module.half().eval()
        if prefix == "visual":
            module.patch_embed = RPUVisionPatch(module.patch_embed)
            for block in module.blocks:
                block.attn = RPUVisionAttention(block.attn)
            install_rpu_linears(module)
        elif prefix == "vggt.aggregator":
            module.patch_embed.patch_embed.proj = RPUPatchConv(module.patch_embed.patch_embed.proj)
            for blocks in (module.patch_embed.blocks, module.frame_blocks, module.global_blocks):
                for block in blocks:
                    block.attn = adapt_attention(block.attn)
            install_rpu_linears(module)
        elif prefix == "merger":
            install_rpu_linears(module)
        elif prefix == "lm_head":
            model.lm_head = RPUProjection(module)
        print(f"Janus frontend ready {prefix}", flush=True)
    model.eval()
    return model


class JanusPolicy:
    """Batch-one, 644x490 VGGT / 34x46 Qwen grid, up to nine semantic images."""

    def __init__(self, frontend, decoder):
        self.frontend = frontend
        self.decoder = decoder
        self.spatial_cache = [None] * 24
        self.failed = False
        self.closed = False
        self._lock = threading.RLock()
        self._graph_length = None

    def reset(self):
        with self._lock:
            if self.failed or self.closed:
                raise RuntimeError("failed/closed policy must be reconstructed")
            self.spatial_cache = [None] * 24
            self.decoder.cache.reset_to_position(0)

    def close(self):
        with self._lock:
            if not self.closed:
                self.decoder.close()
                self.spatial_cache = []
                self.frontend = None
                self.closed = True

    def generate(self, **request):
        with self._lock:
            return self._generate(**request)

    @torch.inference_mode()
    def _generate(self, *, input_ids, pixel_values, image_grid_thw, images_vggt,
                 max_new_tokens=32):
        if self.closed or self.failed:
            raise RuntimeError("failed/closed policy must be reconstructed")
        if (input_ids.device.type != "cpu" or input_ids.ndim != 2
                or input_ids.shape[0] != 1 or input_ids.dtype != torch.int64
                or not torch.all((input_ids >= 0) & (input_ids < 152064))):
            raise ValueError("input_ids must be valid CPU batch-one int64 IDs")
        if (type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 32
                or input_ids.shape[1] + max_new_tokens + 64 > self.decoder.cache.max_seq_len):
            raise ValueError("request exceeds controlled token/padding capacity")
        if (image_grid_thw.device.type != "cpu" or image_grid_thw.dtype != torch.int64
                or image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3
                or not 1 <= len(image_grid_thw) <= 9
                or not torch.all(image_grid_thw == image_grid_thw.new_tensor([1, 34, 46]))):
            raise ValueError("only one to nine Qwen grids [1,34,46] are admitted")
        if (len(images_vggt) != 1 or images_vggt[0].shape != (1, 3, 490, 644)
                or images_vggt[0].device.type != "cpu"):
            raise ValueError("one current VGGT frame [1,3,490,644] is required")
        if pixel_values.shape != (len(image_grid_thw) * 1564, 1176):
            raise ValueError("invalid Qwen patch tensor")
        if (pixel_values.device.type != "cpu" or not pixel_values.is_floating_point()
                or not images_vggt[0].is_floating_point()
                or not torch.isfinite(pixel_values).all() or not torch.isfinite(images_vggt[0]).all()):
            raise ValueError("image tensors must be finite CPU floating-point values")
        if int((input_ids == 151655).sum()) != len(image_grid_thw) * 391:
            raise ValueError("image tokens must match visual features")
        model = self.frontend
        try:
            tokens, patch_start, self.spatial_cache = model.vggt.aggregator(
                images_vggt[0].half().unsqueeze(0), past_key_values=self.spatial_cache,
                use_cache=True, past_frame_idx=0)
            features = tokens[-2][0, :, patch_start:].reshape(1, 35, 46, 2048)
            features = features[:, :34].reshape(1, 17, 2, 23, 2, 2048)
            spatial = model.merger(features.permute(0, 1, 3, 2, 4, 5).contiguous().half())
            visual = model.visual(pixel_values.half(), grid_thw=image_grid_thw)
            visual = visual + 0.2 * spatial.repeat(len(image_grid_thw), 1)
            embeds = model.model.embed_tokens(input_ids)
            embeds[input_ids == 151655] = visual
            positions, _ = model.get_rope_index(input_ids, image_grid_thw=image_grid_thw)
            positions = positions[:, 0].transpose(0, 1).contiguous()
            generated = []
            # shortcut: recompute text prefill per token; replace with decode after native certification.
            for _ in range(max_new_tokens):
                size = embeds.shape[1]
                if self._graph_length is not None and self._graph_length != size:
                    # Previous logits were copied to CPU; retire before changing the sole graph.
                    self.decoder._graph_cache.clear()
                    if not self.decoder._graph_cache.cache_invariant_ok():
                        raise RuntimeError("Janus graph retirement invariant failed")
                self._graph_length = size
                execution_size, _ = self.decoder.prefill_execution_plan(size)
                hidden = self.decoder.forward_embeds(
                    F.pad(embeds, (0, 0, 0, execution_size - size)),
                    F.pad(positions, (0, 0, 0, execution_size - size)), logical_len=size)
                # Preserve same-shape source owners until native upload has refreshed.
                for cache in (self.decoder._prefill_pos_cache, self.decoder._prefill_rope_cache):
                    for key in list(cache):
                        if key != execution_size:
                            del cache[key]
                logits = model.lm_head(hidden[:, size - 1:size].cpu().half())
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Janus generated non-finite logits")
                token = logits[:, -1].argmax(-1, keepdim=True)
                generated.append(token)
                if token.item() == 151645:
                    break
                embeds = torch.cat((embeds, model.model.embed_tokens(token)), dim=1)
                positions = torch.cat((positions, positions.new_full((1, 3), int(positions.max()) + 1)))
            return torch.cat((input_ids, *generated), dim=1)
        except BaseException:
            # Aggregator updates layer states in place; a partial step cannot be reused.
            self.failed = True
            raise


def build_policy(model_class, config, checkpoint):
    from .decoder import build_decoder
    from rpu_backend.api.causal_lm import _claim_live_instance, _release_live_instance
    weights, largest = checkpoint_bytes(checkpoint)
    meminfo = {line.split(":")[0]: int(line.split()[1]) * 1024
               for line in Path("/proc/meminfo").read_text().splitlines()
               if line.startswith(("MemAvailable:", "CmaFree:"))}
    plan = admit_memory(weights, largest, meminfo)
    pending = JanusPolicy.__new__(JanusPolicy)
    _claim_live_instance(pending)
    try:
        # Small projection temporaries otherwise exhaust the driver's mapping limit.
        torch.rpu.set_caching_allocator(True)
        frontend = load_frontend(model_class, config, checkpoint)
        decoder = build_decoder(checkpoint)
        JanusPolicy.__init__(pending, frontend, decoder)
        pending.memory_plan = plan
        decoder._live_owner = pending
        return pending
    except BaseException:
        _release_live_instance(pending)
        raise
