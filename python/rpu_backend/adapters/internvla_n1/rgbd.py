"""InternVLA-N1 RGBD encoder on the board's CPU (fp32) — self-contained, pure torch.

Two weight-shared DINOv2 ViT-S towers (rgb + depth) + a 2-layer perceiver (former_net) →
rgbd_embed[B,32,384] for the NavDP System-1 memory. Reconstructs InternNav's
DAT_RGBD_Patch_Backbone but constructs the DINOv2 backbone DIRECTLY (skipping the DepthAnythingV2
wrapper, which pulls cv2/torchvision) so it runs on the board with only torch — no torchvision /
timm / cv2 / xformers (MemEffAttention falls back to standard attention when xformers is absent).

The DINOv2 source (`dinov2.py` + `dinov2_layers/`, pure torch, bundled by InternNav) must be
importable — pass its parent dir via `dinov2_src` (added to sys.path). Weights load from the
extracted `navdp_head.safetensors` under `rgbd_encoder.*`. This is the on-board CPU path; the RPU
port (dinov3 reuse + fp16 hybrid) is a separate, deferred effort.
"""
from __future__ import annotations
import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _module_source(module) -> Path | None:
    raw = getattr(module, "__file__", None)
    if not raw:
        return None
    return Path(raw).resolve()


def _assert_dinov2_import_source(src: Path) -> None:
    """Reject a cached or resolved DINOv2 module outside ``src``."""
    package_name = src.name
    module_name = f"{package_name}.dinov2"
    module = sys.modules.get(module_name)
    if module is not None and _module_source(module) != src / "dinov2.py":
        raise ImportError(
            f"{module_name} resolved outside the requested DINOv2 source tree"
        )

    package = sys.modules.get(package_name)
    if package is not None:
        roots = {
            Path(raw).resolve()
            for raw in getattr(package, "__path__", ())
        }
        if roots != {src}:
            raise ImportError(
                f"{package_name} resolved outside the requested DINOv2 source tree"
            )

    for name, loaded in tuple(sys.modules.items()):
        if name == package_name:
            continue
        if not name.startswith(package_name + "."):
            continue
        loaded_source = _module_source(loaded)
        try:
            relative = loaded_source.relative_to(src)
        except (AttributeError, ValueError):
            raise ImportError(
                f"{name} resolved outside the requested DINOv2 source tree"
            ) from None
        if relative.suffix != ".py":
            raise ImportError(
                f"{name} did not resolve to a manifest-bound Python source file"
            )


def _load_dinov2(src: Path):
    src = src.expanduser().resolve(strict=True)
    loaded = sorted(
        name
        for name in sys.modules
        if name == src.name or name.startswith(src.name + ".")
    )
    if loaded:
        raise ImportError(
            "the requested DINOv2 package name is already loaded; use a fresh "
            f"process so manifest verification binds executed source: {loaded[:8]}"
        )
    parent = str(src.parent)
    sys.path[:] = [entry for entry in sys.path if entry != parent]
    sys.path.insert(0, parent)
    importlib.invalidate_caches()
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module = importlib.import_module(f"{src.name}.dinov2")
    finally:
        sys.dont_write_bytecode = previous
    _assert_dinov2_import_source(src)
    return module.DINOv2


class RGBDEncoder(nn.Module):
    """Mirror of DAT_RGBD_Patch_Backbone (version=1.0, finetune=False)."""

    def __init__(self, dinov2_src: str, image_size=224, embed_size=384,
                 memory_size=2, input_dtype=torch.bfloat16, version=1.0):
        super().__init__()
        if (image_size, embed_size, memory_size, version) != (224, 384, 2, 1.0):
            raise ValueError(
                "InternVLA-N1 RGBD profile must be image_size=224, "
                "embed_size=384, memory_size=2, version=1.0"
            )
        src = Path(dinov2_src)                       # .../rgbd_src (a package dir)
        DINOv2 = _load_dinov2(src)                   # bundled pure-torch DINOv2

        self.image_size, self.memory_size = image_size, memory_size
        self.input_dtype, self.version = input_dtype, version
        self.rgb_model = DINOv2("vits").eval()
        self.depth_model = DINOv2("vits")
        _assert_dinov2_import_source(src.resolve(strict=True))
        self.register_buffer(
            "preprocess_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=input_dtype),
            persistent=False,
        )
        self.register_buffer(
            "preprocess_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=input_dtype),
            persistent=False,
        )
        self.former_query = nn.Embedding(memory_size * 16, 384)
        self.former_pe = nn.Embedding((memory_size * 2) * 256 if version > 0.0
                                      else (memory_size + 1) * 256, 384)
        self.former_net = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(384, 8, batch_first=True), 2)
        self.project_layer = nn.Linear(384, embed_size)

    @torch.no_grad()
    def forward(self, images, depths):
        dt, dev, S = self.input_dtype, images.device, self.image_size
        B, T = images.shape[0], images.shape[1]
        mean = self.preprocess_mean.reshape(1, 3, 1, 1).to(dev)
        std = self.preprocess_std.reshape(1, 3, 1, 1).to(dev)
        ti = images.to(dt).permute(0, 1, 4, 2, 3).reshape(-1, 3, S, S)
        ti = (ti - mean) / std
        image_token = self.rgb_model.get_intermediate_layers(ti)[0].reshape(B, T * 256, -1)
        td = depths.to(dt).permute(0, 1, 4, 2, 3).reshape(-1, 1, S, S)
        td = torch.cat([td, td, td], dim=1)
        depth_token = self.depth_model.get_intermediate_layers(td)[0].reshape(B, T * 256, -1)
        n_pe = (self.memory_size * 2) * 256 if self.version > 0.0 else (self.memory_size + 1) * 256
        pe = self.former_pe(torch.arange(n_pe, device=dev).expand(image_token.shape[0], n_pe))
        former_token = torch.cat((image_token, depth_token), dim=1) + pe
        q = self.former_query(torch.arange(self.memory_size * 16, device=dev)
                              .expand(image_token.shape[0], self.memory_size * 16))
        return self.project_layer(self.former_net(q, former_token))


def build_rgbd_encoder(safetensors_path: str | Path, dinov2_src: str,
                       device: str = "cpu") -> RGBDEncoder:
    """Strictly load ``rgbd_encoder.*`` onto the board CPU in fp32."""
    if device != "cpu":
        raise ValueError("use build_rgbd_encoder_rpu for the controlled RPU path")
    from safetensors.torch import load_file
    sd = load_file(str(safetensors_path))
    enc = RGBDEncoder(dinov2_src, input_dtype=torch.float32).eval()
    local = {k[len("rgbd_encoder."):]: v.float() for k, v in sd.items()
             if k.startswith("rgbd_encoder.")}
    enc.load_state_dict(local, strict=True)
    return enc.float().cpu()
