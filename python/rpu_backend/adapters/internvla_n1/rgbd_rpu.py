"""Numeric-blocked RGBD DINOv2-S tower via the reused RPU DINOv3 engine.

The fixed 224x224 path prepares patch14/CLS/position tokens on CPU, then runs all 12 transformer
blocks and final LayerNorm on RPU. Unsupported image shapes keep the general CPU token-preparation
fallback. See
[[internvla-n1-rgbd-rpu-derisk]]: fp16 is safe (max q·kᵀ 2670), k_bias is softmax-irrelevant, and
RoPE is no-op'd with identity tables so dinov3's NUM_SPECIAL_TOKENS=5 constant doesn't matter.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

from rpu_backend.runtime import UnsupportedModelError
from rpu_backend.runtime.control import rpu_env_bool
from rpu_backend.api.cache import RPUCache
from rpu_backend.runtime.weights import (
    tp_col_swizzle_mc_weight,
    tp_row_swizzle_mc_weight,
)
import rpu_backend
from ._policy import (
    require_controlled_evaluation,
    verify_asset_hashes,
    verify_asset_manifest,
)

NUM_CORES = 8


def _destroy_rgbd_tower_handle(handle: int) -> None:
    try:
        torch.ops.rpu.dinov3_vision_destroy(handle)
    except Exception:
        pass


def _configure_rgbd_tower_handle(configure) -> int:
    """Create/configure one DINO handle, releasing it on failure."""
    handle = torch.ops.rpu.dinov3_vision_create()
    configured = False
    try:
        configure(handle)
        configured = True
        return handle
    finally:
        if not configured:
            _destroy_rgbd_tower_handle(handle)


def _destroy_former_handle(handle: int) -> None:
    try:
        torch.ops.rpu.navdp_destroy(handle)
    except Exception:
        pass


def _configure_former_handle(configure) -> int:
    """Create/configure one Former handle, releasing it on failure."""
    handle = torch.ops.rpu.navdp_create()
    configured = False
    try:
        configure(handle)
        configured = True
        return handle
    finally:
        if not configured:
            _destroy_former_handle(handle)


def _verify_dinov2_source_tree(
    dinov2_src: str | Path,
    asset_manifest: Mapping[str | Path, str] | None,
    *,
    controlled_rpu: bool = True,
) -> Path:
    """Hash every importable source file before the dynamic DINOv2 import."""
    if controlled_rpu:
        require_controlled_evaluation(asset_manifest)
    raw_root = Path(dinov2_src).expanduser()
    if raw_root.is_symlink():
        raise UnsupportedModelError(
            "InternVLA-N1 RGBD source root must not be a symlink"
        )
    try:
        root = raw_root.resolve(strict=True)
    except OSError as exc:
        raise UnsupportedModelError(
            f"InternVLA-N1 RGBD source root is unavailable: {raw_root}"
        ) from exc
    if not root.is_dir() or not root.name.isidentifier():
        raise UnsupportedModelError(
            "InternVLA-N1 RGBD source root must be an importable package directory"
        )

    entries = tuple(sorted(root.rglob("*")))
    symlinks = [str(path) for path in entries if path.is_symlink()]
    if symlinks:
        raise UnsupportedModelError(
            "InternVLA-N1 RGBD source tree must not contain symlinks: "
            f"{symlinks[:8]}"
        )
    import_artifacts = [
        str(path)
        for path in entries
        if path.is_file()
        and (path.suffix in {".pyc", ".pyo", ".so", ".pyd"})
    ]
    if import_artifacts:
        raise UnsupportedModelError(
            "InternVLA-N1 RGBD source tree contains unverified import "
            f"artifacts: {import_artifacts[:8]}"
        )
    sources = tuple(
        path for path in entries if path.is_file() and path.suffix == ".py"
    )
    required = (root / "dinov2.py", root / "dinov2_layers" / "__init__.py")
    missing = [str(path) for path in required if path not in sources]
    if missing:
        raise UnsupportedModelError(
            "InternVLA-N1 RGBD source tree is incomplete: "
            f"missing={missing}"
        )
    verifier = verify_asset_manifest if controlled_rpu else verify_asset_hashes
    verifier(sources, asset_manifest)
    return root


def _hr(t):
    return t.to(torch.float16).to("rpu").contiguous()


def _pad_heads(w, nh, hd, target, dim0=True):
    """Zero-pad head rows (dim0) or cols (dim1) from nh→target heads."""
    pad = (target - nh) * hd
    if pad == 0:
        return w
    if dim0:
        return torch.cat([w, torch.zeros(pad, w.shape[1], dtype=w.dtype)], 0).contiguous()
    return torch.cat([w, torch.zeros(w.shape[0], pad, dtype=w.dtype)], 1).contiguous()


class RGBDTowerRPU:
    """One DINOv2 ViT-S tower: CPU glue + RPU blocks. Call forward(x[B,3,224,224])→[B,256,384]."""

    def __init__(
        self,
        dinov2_tower: nn.Module,
        nh=6,
        hd=64,
        hidden=384,
        inter=1536,
        eps=1e-6,
        *,
        input_mean=None,
        input_std=None,
        asset_manifest=None,
        asset_paths=None,
    ):
        require_controlled_evaluation(asset_manifest)
        if not asset_paths:
            raise UnsupportedModelError(
                "direct RGBDTowerRPU construction requires asset_paths for SHA-256 verification"
            )
        verify_asset_manifest(asset_paths, asset_manifest)
        final_norm_on_rpu = rpu_env_bool(
            "RPU_INTERNVLA_DINO_FINAL_NORM_RPU", True
        )
        norm_fold = rpu_env_bool("RPU_INTERNVLA_DINO_NORM_FOLD")
        if (nh, hd, hidden, inter, eps) != (6, 64, 384, 1536, 1e-6):
            raise ValueError("InternVLA-N1 RGBD RPU tower profile mismatch")
        blocks = list(getattr(dinov2_tower, "blocks", ()))
        if len(blocks) != 12:
            raise ValueError(f"InternVLA-N1 RGBD RPU tower requires 12 blocks, got {len(blocks)}")
        expected = {
            "attn.qkv.weight": (1152, 384), "attn.qkv.bias": (1152,),
            "attn.proj.weight": (384, 384), "attn.proj.bias": (384,),
            "mlp.fc1.weight": (1536, 384), "mlp.fc1.bias": (1536,),
            "mlp.fc2.weight": (384, 1536), "mlp.fc2.bias": (384,),
            "norm1.weight": (384,), "norm1.bias": (384,),
            "norm2.weight": (384,), "norm2.bias": (384,),
            "ls1.gamma": (384,), "ls2.gamma": (384,),
        }
        for layer, block in enumerate(blocks):
            for name, shape in expected.items():
                value = block
                for part in name.split("."):
                    value = getattr(value, part, None)
                if (
                    value is None
                    or tuple(value.shape) != shape
                    or not value.is_floating_point()
                ):
                    actual = None if value is None else tuple(value.shape)
                    raise ValueError(
                        f"InternVLA-N1 RGBD block {layer} {name} shape "
                        f"{actual}, expected {shape}"
                    )
        self.tower = dinov2_tower            # CPU DINOv2 (patch_embed, cls_token, pos_embed, norm, blocks)
        self.nh, self.hd, self.hidden, self.inter, self.eps = nh, hd, hidden, inter, eps
        self.tp = NUM_CORES                  # padded heads = 8
        self._gc = rpu_backend.graph.GraphCache()
        self._position_seq_len = None
        self._pos_embed_224 = None
        if getattr(self.tower, "num_register_tokens", 0) == 0:
            parameter = next(self.tower.parameters())
            placeholder = torch.empty(
                1, 257, hidden, dtype=parameter.dtype, device=parameter.device)
            self._pos_embed_224 = self.tower.interpolate_pos_encoding(
                placeholder, 224, 224).detach()
        self._final_norm_on_rpu = final_norm_on_rpu
        self.input_normalization_folded = (
            norm_fold
            and input_mean is not None
            and input_std is not None
        )
        self._folded_patch_weight = None
        self._folded_patch_bias = None
        if self.input_normalization_folded:
            conv = self.tower.patch_embed.proj
            mean = torch.as_tensor(input_mean).detach().float().view(1, -1, 1, 1)
            std = torch.as_tensor(input_std).detach().float().view(1, -1, 1, 1)
            weight = conv.weight.detach().float()
            self._folded_patch_weight = weight.contiguous()
            bias = (
                conv.bias.detach().float()
                if conv.bias is not None
                else torch.zeros(weight.shape[0], dtype=weight.dtype)
            )
            self._folded_patch_bias = (
                bias - (weight * (mean / std)).sum(dim=(1, 2, 3))
            ).contiguous()
        self.output_mlp_on_rpu = False
        self._output_mlp_keepalive = None

        def configure(handle):
            self.handle = handle
            self._install()
            self._install_stem_and_tail()
            # Board-validated DINOv2-S profile: ACC32 linears + 16-bank fp16 SDPA.
            torch.ops.rpu.dinov3_vision_set_acc32(handle, True)
            torch.ops.rpu.dinov3_vision_set_16b_sdpa(handle, True)

        self.handle = _configure_rgbd_tower_handle(configure)
        try:
            self._handle_finalizer = weakref.finalize(
                self, _destroy_rgbd_tower_handle, self.handle
            )
        except BaseException:
            _destroy_rgbd_tower_handle(self.handle)
            self.handle = None
            raise

    def set_output_mlp(
        self,
        first_weight: torch.Tensor,
        first_bias: torch.Tensor,
        second_weight: torch.Tensor,
        second_bias: torch.Tensor,
    ) -> None:
        if not self._final_norm_on_rpu:
            raise RuntimeError("DINO output MLP requires final norm on RPU")
        keepalive = [
            _hr(tp_col_swizzle_mc_weight(first_weight.half(), NUM_CORES)),
            _hr(first_bias),
            _hr(tp_row_swizzle_mc_weight(second_weight.half(), NUM_CORES)),
            _hr(second_bias),
        ]
        torch.ops.rpu.dinov3_vision_set_output_mlp(
            self.handle, *keepalive
        )
        self._output_mlp_keepalive = keepalive
        self.output_mlp_on_rpu = True

    def _install(self):
        blocks = list(self.tower.blocks)
        H, hd, nh, tgt = self.hidden, self.hd, self.nh, self.tp
        qw, kw, vw, ow, upw, dnw = [], [], [], [], [], []
        ln1w, ln1b, ln2w, ln2b = [], [], [], []
        qb, vb, ob, upb, dnb, ga, gm = [], [], [], [], [], [], []
        for blk in blocks:
            a = blk.attn
            W = a.qkv.weight.data.float(); B = a.qkv.bias.data.float()
            q, k, v = W[:H], W[H:2 * H], W[2 * H:]
            bq, _, bv = B[:H], B[H:2 * H], B[2 * H:]
            q = _pad_heads(q, nh, hd, tgt); k = _pad_heads(k, nh, hd, tgt); v = _pad_heads(v, nh, hd, tgt)
            o = _pad_heads(a.proj.weight.data.float(), nh, hd, tgt, dim0=False)
            qw.append(_hr(tp_col_swizzle_mc_weight(q.half(), self.tp)))
            kw.append(_hr(tp_col_swizzle_mc_weight(k.half(), self.tp)))
            vw.append(_hr(tp_col_swizzle_mc_weight(v.half(), self.tp)))
            ow.append(_hr(tp_row_swizzle_mc_weight(o.half(), self.tp)))
            upw.append(_hr(tp_col_swizzle_mc_weight(blk.mlp.fc1.weight.data.half(), self.tp)))
            dnw.append(_hr(tp_row_swizzle_mc_weight(blk.mlp.fc2.weight.data.half(), self.tp)))
            ln1w.append(_hr(blk.norm1.weight.data)); ln1b.append(_hr(blk.norm1.bias.data))
            ln2w.append(_hr(blk.norm2.weight.data)); ln2b.append(_hr(blk.norm2.bias.data))
            pad = (tgt - nh) * hd
            qb.append(_hr(torch.cat([bq, torch.zeros(pad)])))
            vb.append(_hr(torch.cat([bv, torch.zeros(pad)])))
            ob.append(_hr(a.proj.bias.data))
            upb.append(_hr(blk.mlp.fc1.bias.data)); dnb.append(_hr(blk.mlp.fc2.bias.data))
            ga.append(_hr(blk.ls1.gamma.data)); gm.append(_hr(blk.ls2.gamma.data))
        torch.ops.rpu.dinov3_vision_set_weights(
            self.handle, qw, kw, vw, ow, upw, dnw, ln1w, ln1b, ln2w, ln2b,
            qb, vb, ob, upb, dnb, ga, gm, tgt, hd, H, self.inter, self.eps)
        # identity RoPE (cos=1, sin=0) → rope_2d is a no-op; pos-embed added on CPU instead
        mx = 16
        cos = torch.ones(mx, hd // 4, dtype=torch.float16).to("rpu").contiguous()
        sin = torch.zeros(mx, hd // 4, dtype=torch.float16).to("rpu").contiguous()
        torch.ops.rpu.dinov3_vision_set_rope(self.handle, cos, sin)
        torch.ops.rpu.dinov3_vision_set_identity_rope(self.handle, True)
        self.keepalive = torch.ops.rpu.dinov3_vision_position_idx_keepalive(self.handle)
        self.num_layers = len(blocks)
        # Bidirectional vision SDPA inserts at position 0 and reads once, so a
        # standard cache sized to the worst-case token count is sufficient.
        self.cache = RPUCache(
            num_layers=self.num_layers,
            batch_size=1,
            max_seq_len=512,
            num_kv_heads=tgt,
            head_dim=hd,
            attn_tp=min(NUM_CORES, tgt),
        )

    def _install_stem_and_tail(self):
        if self._final_norm_on_rpu:
            torch.ops.rpu.dinov3_vision_set_final_norm(
                self.handle, _hr(self.tower.norm.weight.detach()),
                _hr(self.tower.norm.bias.detach()))

    @torch.no_grad()
    def forward(self, x, *, host_prepare_done=None):
        if self._pos_embed_224 is not None and tuple(x.shape[-2:]) == (224, 224):
            if self.input_normalization_folded:
                conv = self.tower.patch_embed.proj
                patches = F.conv2d(
                    x,
                    self._folded_patch_weight,
                    self._folded_patch_bias,
                    stride=conv.stride,
                    padding=conv.padding,
                    dilation=conv.dilation,
                    groups=conv.groups,
                ).flatten(2).transpose(1, 2)
                patches = self.tower.patch_embed.norm(patches)
            else:
                patches = self.tower.patch_embed(x)
            cls = self.tower.cls_token.expand(patches.shape[0], -1, -1)
            seq = torch.cat((cls, patches), dim=1) + self._pos_embed_224
        else:
            seq = self.tower.prepare_tokens_with_masks(x)
        B = x.shape[0]
        N = seq.shape[1]
        if self._position_seq_len != N:
            grid = int((N - 1) ** 0.5)
            # Identity RoPE still reads this persistent keepalive.  Its contents only
            # depend on the token shape, so avoid rebuilding/copying it every frame.
            rc = torch.zeros(N - 1, 2, dtype=torch.int16)
            idx = torch.arange(N - 1)
            rc[:, 0] = (idx // grid).to(torch.int16)
            rc[:, 1] = (idx % grid).to(torch.int16)
            self.keepalive[:N - 1] = rc.to(self.keepalive.device)
            self._position_seq_len = N
        outputs = []
        for b in range(B):
            source = _hr(seq[b:b + 1])
            if b == 0 and host_prepare_done is not None:
                host_prepare_done()
            sig = rpu_backend.graph.GraphSignature(
                op_id="rgbd_vits_tower",
                shapes=[N, self.hidden],
                dyn_dims=[self.num_layers],
                dtypes=[torch.float16],
            )
            with self._gc.capture(sig):
                r = torch.ops.rpu.dinov3_vision_forward(
                    self.handle, source, self.cache.k_caches, self.cache.v_caches, N)
            # FP16 D2H is half the traffic; the following CPU FP16→FP32 cast is exact.
            outputs.append(r.cpu().float())
        out = outputs[0] if B == 1 else torch.cat(outputs, dim=0)
        if not self._final_norm_on_rpu:
            out = self.tower.norm(out)
        return out[:, 1:, :]

    def destroy(self):
        if getattr(self, "handle", None) is None:
            return
        graph_cache = getattr(self, "_gc", None)
        if graph_cache is not None:
            graph_cache.clear()
        finalizer = getattr(self, "_handle_finalizer", None)
        if finalizer is not None and getattr(finalizer, "alive", False):
            torch.ops.rpu.dinov3_vision_destroy(self.handle)
            finalizer.detach()
        elif finalizer is None:
            torch.ops.rpu.dinov3_vision_destroy(self.handle)
        self.handle = None

    def graph_stats(self):
        """Return a serializable retained-Graph lifecycle receipt."""
        entries = list(self._gc.snapshot())
        return {
            "size": int(self._gc.size()),
            "invariant": bool(self._gc.cache_invariant_ok()),
            "entries": [
                {
                    "signature": repr(entry.signature),
                    "kernel_count": int(entry.kernel_count),
                    "data_node_count": int(entry.data_node_count),
                    "replay_count": int(entry.replay_count),
                    "recapture_count": int(entry.recapture_count),
                }
                for entry in entries
            ],
        }


def _col(w): return _hr(tp_col_swizzle_mc_weight(w.to(torch.float16), NUM_CORES))
def _row(w): return _hr(tp_row_swizzle_mc_weight(w.to(torch.float16), NUM_CORES))


class FormerRPU(nn.Module):
    """RGBD former_net perceiver on RPU — reuses the NavDP fused decoder via former_mode
    (POST-LN, ReLU, non-causal self-attn, memory_len=1024). Drop-in for `nn.TransformerDecoder`:
    __call__(q[B,32,384], memory[B,1024,384]) -> hidden[B,32,384] (CPU float; project_layer stays
    host). Weights: rgbd_encoder.former_net.*."""
    H, NH, HD, FF, ML, NQ = 384, 8, 48, 2048, 1024, 32

    def __init__(self, safetensors_path, *, asset_manifest=None):
        super().__init__()
        require_controlled_evaluation(asset_manifest)
        checkpoint = Path(safetensors_path).expanduser().resolve()
        verify_asset_manifest((checkpoint,), asset_manifest)
        from safetensors.torch import load_file
        sd = load_file(str(checkpoint)); P = "rgbd_encoder.former_net.layers"
        actual_layers = {
            int(key[len(P) + 1:].split(".", 1)[0])
            for key in sd
            if key.startswith(P + ".")
            and key[len(P) + 1:].split(".", 1)[0].isdigit()
        }
        if actual_layers != {0, 1}:
            raise ValueError(
                "InternVLA-N1 Former checkpoint profile mismatch: layer indices "
                f"{sorted(actual_layers)}, expected [0, 1]"
            )
        expected = {}
        for layer in range(2):
            prefix = f"{P}.{layer}."
            expected.update({
                prefix + "self_attn.in_proj_weight": (3 * self.H, self.H),
                prefix + "self_attn.in_proj_bias": (3 * self.H,),
                prefix + "self_attn.out_proj.weight": (self.H, self.H),
                prefix + "self_attn.out_proj.bias": (self.H,),
                prefix + "multihead_attn.in_proj_weight": (3 * self.H, self.H),
                prefix + "multihead_attn.in_proj_bias": (3 * self.H,),
                prefix + "multihead_attn.out_proj.weight": (self.H, self.H),
                prefix + "multihead_attn.out_proj.bias": (self.H,),
                prefix + "linear1.weight": (self.FF, self.H),
                prefix + "linear1.bias": (self.FF,),
                prefix + "linear2.weight": (self.H, self.FF),
                prefix + "linear2.bias": (self.H,),
                **{
                    prefix + f"norm{n}.{slot}": (self.H,)
                    for n in (1, 2, 3) for slot in ("weight", "bias")
                },
            })
        errors = [
            f"{key}: {tuple(sd[key].shape) if key in sd else 'missing'}, expected {shape}"
            for key, shape in expected.items()
            if key not in sd
            or tuple(sd[key].shape) != shape
            or not sd[key].is_floating_point()
        ]
        if errors:
            raise ValueError(
                "InternVLA-N1 Former checkpoint profile mismatch: "
                + "; ".join(errors[:8])
            )
        def gg(k): return sd[k].float()
        H = self.H
        L = {k: [] for k in ("sq_w sk_w sv_w so_w sq_b sk_b sv_b so_b cq_w ck_w cv_w co_w "
             "cq_b ck_b cv_b co_b ff1_w ff1_b ff2_w ff2_b n1_w n1_b n2_w n2_b n3_w n3_b").split()}
        for i in range(2):
            p = f"{P}.{i}"
            siw, sib = gg(f"{p}.self_attn.in_proj_weight"), gg(f"{p}.self_attn.in_proj_bias")
            ciw, cib = gg(f"{p}.multihead_attn.in_proj_weight"), gg(f"{p}.multihead_attn.in_proj_bias")
            L["sq_w"].append(_col(siw[0:H])); L["sk_w"].append(_col(siw[H:2*H])); L["sv_w"].append(_col(siw[2*H:3*H]))
            L["sq_b"].append(_hr(sib[0:H])); L["sk_b"].append(_hr(sib[H:2*H])); L["sv_b"].append(_hr(sib[2*H:3*H]))
            L["so_w"].append(_row(gg(f"{p}.self_attn.out_proj.weight"))); L["so_b"].append(_hr(gg(f"{p}.self_attn.out_proj.bias")))
            L["cq_w"].append(_col(ciw[0:H])); L["ck_w"].append(_col(ciw[H:2*H])); L["cv_w"].append(_col(ciw[2*H:3*H]))
            L["cq_b"].append(_hr(cib[0:H])); L["ck_b"].append(_hr(cib[H:2*H])); L["cv_b"].append(_hr(cib[2*H:3*H]))
            L["co_w"].append(_row(gg(f"{p}.multihead_attn.out_proj.weight"))); L["co_b"].append(_hr(gg(f"{p}.multihead_attn.out_proj.bias")))
            L["ff1_w"].append(_col(gg(f"{p}.linear1.weight"))); L["ff1_b"].append(_hr(gg(f"{p}.linear1.bias")))
            L["ff2_w"].append(_row(gg(f"{p}.linear2.weight"))); L["ff2_b"].append(_hr(gg(f"{p}.linear2.bias")))
            for n, src in (("n1", "norm1"), ("n2", "norm2"), ("n3", "norm3")):
                L[f"{n}_w"].append(_hr(gg(f"{p}.{src}.weight"))); L[f"{n}_b"].append(_hr(gg(f"{p}.{src}.bias")))
        self.cache = RPUCache(num_layers=2, batch_size=1, max_seq_len=1152, num_kv_heads=self.NH, head_dim=self.HD)
        self.gc = rpu_backend.graph.GraphCache()

        def configure(handle):
            torch.ops.rpu.navdp_set_former_mode(handle, True)
            torch.ops.rpu.navdp_set_weights(handle,
                L["sq_w"],L["sk_w"],L["sv_w"],L["so_w"],L["sq_b"],L["sk_b"],L["sv_b"],L["so_b"],
                L["cq_w"],L["ck_w"],L["cv_w"],L["co_w"],L["cq_b"],L["ck_b"],L["cv_b"],L["co_b"],
                L["ff1_w"],L["ff1_b"],L["ff2_w"],L["ff2_b"],
                L["n1_w"],L["n1_b"],L["n2_w"],L["n2_b"],L["n3_w"],L["n3_b"],
                self.NH, self.HD, self.H, self.FF, self.ML, self.NQ, 3, 1e-5)

        self.handle = _configure_former_handle(configure)
        try:
            self._handle_finalizer = weakref.finalize(
                self, _destroy_former_handle, self.handle
            )
        except BaseException:
            _destroy_former_handle(self.handle)
            self.handle = None
            raise

    @torch.no_grad()
    def forward(self, q, memory):
        outs = []
        for b in range(q.shape[0]):
            sig = rpu_backend.graph.GraphSignature(
                op_id="rgbd_former",
                shapes=[self.NQ, self.H],
                dyn_dims=[2, self.ML],
                dtypes=[torch.float16],
            )
            with self.gc.capture(sig):
                hid = torch.ops.rpu.navdp_forward(self.handle, _hr(q[b:b+1]), _hr(memory[b:b+1]),
                                                  self.cache.k_caches, self.cache.v_caches, None)
            outs.append(hid.float().cpu()[0])
        return torch.stack(outs, 0)

    def destroy(self):
        if getattr(self, "handle", None) is None:
            return
        graph_cache = getattr(self, "gc", None)
        if graph_cache is not None:
            graph_cache.clear()
        finalizer = getattr(self, "_handle_finalizer", None)
        if finalizer is not None and getattr(finalizer, "alive", False):
            torch.ops.rpu.navdp_destroy(self.handle)
            finalizer.detach()
        elif finalizer is None:
            torch.ops.rpu.navdp_destroy(self.handle)
        self.handle = None


def build_rgbd_encoder_rpu(
    safetensors_path,
    dinov2_src,
    *,
    asset_manifest: Mapping[str | Path, str] | None = None,
):
    """Full RGBD encoder with the two ViT-S towers on RPU (former_net stays board CPU).
    Returns an RGBDEncoder whose towers' `get_intermediate_layers` is monkey-patched to the
    RPU dinov3 engine, with the same forward(images, depths) interface as the CPU build."""
    require_controlled_evaluation(asset_manifest)
    checkpoint = Path(safetensors_path).expanduser().resolve()
    verify_asset_manifest((checkpoint,), asset_manifest)
    source_root = _verify_dinov2_source_tree(dinov2_src, asset_manifest)
    from rpu_backend.adapters.internvla_n1.rgbd import build_rgbd_encoder
    enc = build_rgbd_encoder(
        safetensors_path, str(source_root), device="cpu"
    ).float()  # fp32 CPU glue
    components = []
    try:
        rgb_tw = RGBDTowerRPU(
            enc.rgb_model, asset_manifest=asset_manifest,
            asset_paths=(checkpoint,))
        components.append(rgb_tw)
        dep_tw = RGBDTowerRPU(
            enc.depth_model, asset_manifest=asset_manifest,
            asset_paths=(checkpoint,))
        components.append(dep_tw)
        former = FormerRPU(checkpoint, asset_manifest=asset_manifest)
        components.append(former)

        # RPU tower.forward(x[B,3,224,224]) -> [B,256,384]
        # == get_intermediate_layers(x)[0]
        enc.rgb_model.get_intermediate_layers = (
            lambda x, *a, **k: [rgb_tw.forward(x)]
        )
        enc.depth_model.get_intermediate_layers = (
            lambda x, *a, **k: [dep_tw.forward(x)]
        )
        enc.input_dtype = torch.float32
        enc._rpu_towers = (rgb_tw, dep_tw)
        enc.former_net = former
        enc._rpu_former = former
        return enc
    except BaseException:
        for component in reversed(components):
            try:
                component.destroy()
            except Exception:
                pass
        raise
