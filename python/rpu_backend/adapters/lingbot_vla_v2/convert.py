"""LingBot-VLA V2 (robbyant/lingbot-vla-v2-6b) — checkpoint loader / RPU converter.

Reads the OFFICIAL stacked-expert safetensors layout directly and produces the fp16 RPU
weight set consumed by the `lingbot_v2_moe_*` op family.

⚠️  IMPORT CONTRACT: this module MUST stay importable on a host with NO RPU.
    There is NO module-level `import rpu_backend` (importing the package loads the built .so,
    whose module init touches /dev/rpu). Every rpu_backend symbol is imported INSIDE the
    function that needs it, and only the RPU-building functions do so. The CPU-only surface
    (`ckpt_reader`, `load_moe_layer_weights`, `expert_slice`, `load_host_weights`, `SWIZZLE_PLAN`)
    is pure torch + safetensors and can be loaded with no RPU present. Keep it that way: do NOT
    add a module-level rpu_backend import or relative (`from .x import`) imports.

Stacked expert layout (bare nn.Parameter — NO `.weight` suffix; Qwen2FusedExperts):
    mlp.experts.gate_proj [E=32, 512, 768]   -> expert e's Linear(768->512).weight = stacked[e]
    mlp.experts.up_proj   [E=32, 512, 768]
    mlp.experts.down_proj [E=32, 768, 512]   -> expert e's Linear(512->768).weight = stacked[e]
    mlp.gate.weight       [32, 768]          router
    mlp.e_score_correction_bias [32]         SELECTION-only bias
    mlp.shared_expert.{gate,up}_proj.weight [704, 768]
    mlp.shared_expert.down_proj.weight      [768, 704]

Convert to fp16 before swizzling, never after. Every builder below follows
half -> (pad) -> swizzle -> .to('rpu'). Build constant tensors on CPU before moving them to RPU.
"""
from __future__ import annotations

import glob
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

# ---- checkpoint key prefixes ----
VLM_P = "model.qwenvl_with_expert.qwenvl.model."
EXP_P = "model.qwenvl_with_expert.qwen_expert.model."

# ---- action-expert geometry for the supported kuavo_v2_depth profile ----
EXP_LAYERS = 36
EXP_HID = 768
NQ, NKV, HD = 32, 8, 128        # SAME head geometry as the VLM => shared prefix-KV cache works
RMS_EPS = 1e-6

# ---- MoE (all 36 layers) ----
NUM_EXPERTS = 32
TOP_K = 4
ROUTED_INTER = 512
SHARED_INTER = 704
# 704 is not a multiple of 128, which 8-core col-swizzle requires (N % (16*8) == 0). Zero-pad to
# 768 is NUMERICALLY EXACT for SwiGLU: the padded gate/up rows produce silu(0)*0 = 0, and the
# padded down_proj columns multiply those zeros.
SHARED_INTER_PAD = 768
ROUTED_SCALING = 4.0
NORM_TOPK_PROB = True           # fixed by the model's MoE contract
ROUTER_ACTIVATION = "sigmoid"

# ---- action / flow-matching ----
N_ACTION = 50                   # chunk_size
ACTION_DIM = 55                 # max_action_dim
STATE_DIM = 55                  # max_state_dim
NUM_STEPS = 10

# ---- core / partition plan ----
ATTN_TP = 8                     # C++ attn_tp() == min(NUM_CORES, num_kv_heads) == min(8, 8) == 8
MLP_CORES = 8
ROUTER_CORES = 1                # the router GEMM is emitted on core 0 only


# =============================================================================
# Per-checkpoint geometry.
#
# The constants above are the default geometry. `resolve_geometry(...)` reads a
# checkpoint's YAML when provided and validates every shape- and math-affecting
# field against the checkpoint and native runtime. ``geom=None`` uses the defaults.
# =============================================================================
_GEOM_KEYS = (
    "EXP_LAYERS", "EXP_HID", "NQ", "NKV", "HD", "RMS_EPS",
    "NUM_EXPERTS", "TOP_K", "ROUTED_INTER", "SHARED_INTER", "SHARED_INTER_PAD",
    "ROUTED_SCALING", "NORM_TOPK_PROB", "ROUTER_ACTIVATION",
    "N_ACTION", "ACTION_DIM", "STATE_DIM", "NUM_STEPS",
    "ATTN_TP", "MLP_CORES", "ROUTER_CORES",
)


def _default_geom() -> dict:
    """The hard-coded kuavo_v2 defaults as a dict (single source of truth = the module consts)."""
    return {k: globals()[k] for k in _GEOM_KEYS}


# =============================================================================
# Explicit per-tensor classification.
#
# EVERY Linear-like tensor in the action expert is classified here as either
# swizzled-and-fused (-> RPU, with an exact swizzle + core count) or KEPT OUT
# (-> host fp32, never uploaded). A module left on the RPU un-swizzled corrupts
# SILENTLY, so "unclassified" is not an allowed state.
#
#   swizzle: "col" = output-partitioned (N split over cores) -> tp_col_swizzle_mc_weight
#            "row" = input-partitioned  (K split over cores) -> tp_row_swizzle_mc_weight
#            None  = uploaded fp16 but NOT swizzled (vectors / biases)
#            "HOST"= never uploaded; stays CPU fp32
# =============================================================================
SWIZZLE_PLAN: dict[str, dict] = {
    # ---------------- attention (per layer) ----------------
    "self_attn.q_proj.weight":  dict(shape=(NQ * HD, EXP_HID),  swizzle="col", cores=ATTN_TP, dest="q_w"),
    "self_attn.k_proj.weight":  dict(shape=(NKV * HD, EXP_HID), swizzle="col", cores=ATTN_TP, dest="k_w"),
    "self_attn.v_proj.weight":  dict(shape=(NKV * HD, EXP_HID), swizzle="col", cores=ATTN_TP, dest="v_w"),
    "self_attn.o_proj.weight":  dict(shape=(EXP_HID, NQ * HD),  swizzle="row", cores=ATTN_TP, dest="o_w"),
    "self_attn.q_proj.bias":    dict(shape=(NQ * HD,),  swizzle=None, cores=0, dest="q_bias"),
    "self_attn.k_proj.bias":    dict(shape=(NKV * HD,), swizzle=None, cores=0, dest="k_bias"),
    "self_attn.v_proj.bias":    dict(shape=(NKV * HD,), swizzle=None, cores=0, dest="v_bias"),
    # ---------------- AdaRMS norms (per layer) ----------------
    # `.weight` is uploaded as the norm scale PLACEHOLDER; the real per-Euler-step scale/shift
    # ride in through lingbot_v2_moe_set_adarms_step_mutable (the fold is computed on the host).
    "input_layernorm.weight":            dict(shape=(EXP_HID,), swizzle=None, cores=0, dest="input_norm"),
    "post_attention_layernorm.weight":   dict(shape=(EXP_HID,), swizzle=None, cores=0, dest="post_norm"),
    # gamma/beta are [768,768] LINEARS (not norm scales). They consume ada_cond on the HOST and
    # never touch the RPU — folding them per (step, layer, slot) is what set_adarms_step_mutable eats.
    "input_layernorm.gamma.weight":          dict(shape=(EXP_HID, EXP_HID), swizzle="HOST", cores=0, dest="host:norm_W"),
    "input_layernorm.gamma.bias":            dict(shape=(EXP_HID,),         swizzle="HOST", cores=0, dest="host:norm_W"),
    "input_layernorm.beta.weight":           dict(shape=(EXP_HID, EXP_HID), swizzle="HOST", cores=0, dest="host:norm_W"),
    "input_layernorm.beta.bias":             dict(shape=(EXP_HID,),         swizzle="HOST", cores=0, dest="host:norm_W"),
    "post_attention_layernorm.gamma.weight": dict(shape=(EXP_HID, EXP_HID), swizzle="HOST", cores=0, dest="host:norm_W"),
    "post_attention_layernorm.gamma.bias":   dict(shape=(EXP_HID,),         swizzle="HOST", cores=0, dest="host:norm_W"),
    "post_attention_layernorm.beta.weight":  dict(shape=(EXP_HID, EXP_HID), swizzle="HOST", cores=0, dest="host:norm_W"),
    "post_attention_layernorm.beta.bias":    dict(shape=(EXP_HID,),         swizzle="HOST", cores=0, dest="host:norm_W"),
    # ---------------- MoE router (per layer) ----------------
    # The native contract requires router_gate_w and router_bias as fp16 RPU
    # tensors. The gate matrix is single-core COL-swizzled because the router
    # linear uses partition=1 and num_cores=1. The router accumulates in fp32,
    # while its operands remain fp16; reduced operand precision can still affect
    # top-k selection.
    "mlp.gate.weight":              dict(shape=(NUM_EXPERTS, EXP_HID), swizzle="col", cores=ROUTER_CORES, dest="router_gate_w"),
    "mlp.e_score_correction_bias":  dict(shape=(NUM_EXPERTS,),         swizzle=None,  cores=0, dest="router_bias"),
    # ---------------- MoE routed experts (per layer, stacked [E, out, in]) ----------------
    "mlp.experts.gate_proj": dict(shape=(NUM_EXPERTS, ROUTED_INTER, EXP_HID), swizzle="col", cores=MLP_CORES,
                                  dest="expert_gate_w", stacked=True),
    "mlp.experts.up_proj":   dict(shape=(NUM_EXPERTS, ROUTED_INTER, EXP_HID), swizzle="col", cores=MLP_CORES,
                                  dest="expert_up_w", stacked=True),
    "mlp.experts.down_proj": dict(shape=(NUM_EXPERTS, EXP_HID, ROUTED_INTER), swizzle="row", cores=MLP_CORES,
                                  dest="expert_down_w", stacked=True),
    # ---------------- MoE shared expert (per layer; 704 -> 768 zero-pad) ----------------
    "mlp.shared_expert.gate_proj.weight": dict(shape=(SHARED_INTER, EXP_HID), swizzle="col", cores=MLP_CORES,
                                               dest="shared_gate_w", pad=(SHARED_INTER_PAD, EXP_HID)),
    "mlp.shared_expert.up_proj.weight":   dict(shape=(SHARED_INTER, EXP_HID), swizzle="col", cores=MLP_CORES,
                                               dest="shared_up_w", pad=(SHARED_INTER_PAD, EXP_HID)),
    "mlp.shared_expert.down_proj.weight": dict(shape=(EXP_HID, SHARED_INTER), swizzle="row", cores=MLP_CORES,
                                               dest="shared_down_w", pad=(EXP_HID, SHARED_INTER_PAD)),
    # ---------------- expert final norm (model-level, not per layer) ----------------
    # final_norm_adanorm=false selects plain RMSNorm; the checkpoint has no norm.gamma/beta.
    "norm.weight": dict(shape=(EXP_HID,), swizzle=None, cores=0, dest="final_norm_w", model_level=True),
    # ---------------- host-only heads (KEPT OUT of the RPU entirely) ----------------
    # The default Euler loop is the host loop, so the suffix
    # encoders + the action head run as CPU fp32 F.linear and are never swizzled/uploaded.
    "model.state_proj":         dict(shape=(EXP_HID, STATE_DIM),      swizzle="HOST", cores=0, dest="host:head_W"),
    "model.action_in_proj":     dict(shape=(EXP_HID, ACTION_DIM),     swizzle="HOST", cores=0, dest="host:head_W"),
    "model.action_out_proj":    dict(shape=(ACTION_DIM, EXP_HID),     swizzle="HOST", cores=0, dest="host:head_W"),
    "model.action_time_mlp_in": dict(shape=(EXP_HID, 2 * EXP_HID),    swizzle="HOST", cores=0, dest="host:head_W"),
    "model.action_time_mlp_out": dict(shape=(EXP_HID, EXP_HID),       swizzle="HOST", cores=0, dest="host:head_W"),
    # ---------------- prefix align queries (KEPT OUT: host fp32, VLM-hidden-size 2560) ----------
    # These build the 8 current_depth + 8 future_depth prefix query tokens. They are consumed on
    # the HOST (mean-pool + one Linear each) and written straight into the prefix embeds, so they
    # are never swizzled or uploaded. VLM_HID (2560) != EXP_HID (768) — these live on the VLM side.
    "model.depth_align_embs":               dict(shape=(256, 2560),  swizzle="HOST", cores=0, dest="host:align_W"),
    "model.future_depth_align_embs":        dict(shape=(256, 2560),  swizzle="HOST", cores=0, dest="host:align_W"),
    "model.current_video_align_embs":       dict(shape=(256, 2560),  swizzle="HOST", cores=0, dest="host:align_W"),
    "model.future_video_align_embs":        dict(shape=(256, 2560),  swizzle="HOST", cores=0, dest="host:align_W"),
    "model.current_shared_task_proj.weight": dict(shape=(2560, 5120), swizzle="HOST", cores=0, dest="host:align_W"),
    "model.current_shared_task_proj.bias":   dict(shape=(2560,),      swizzle="HOST", cores=0, dest="host:align_W"),
    "model.future_shared_task_proj.weight":  dict(shape=(2560, 5120), swizzle="HOST", cores=0, dest="host:align_W"),
    "model.future_shared_task_proj.bias":    dict(shape=(2560,),      swizzle="HOST", cores=0, dest="host:align_W"),
}

# Tensors present in the checkpoint that this adapter deliberately does NOT consume. Listed so
# that "unclassified" never silently means "forgotten".
#
# ONLY the align *HEADS* are unused: they are distillation-only (they consume the expert's output
# task tokens to regress depth / video features during training) and are NOT on the sample_actions
# path. The align *EMBS* and *shared_task_proj* ARE consumed — they build the 8 current_depth + 8
# future_depth query tokens that the OFFICIAL prefix carries (see prefix.py / ALIGN_KEYS).
UNUSED_CKPT_PREFIXES = (
    "model.depth_align_head.", "model.future_depth_align_head.",
    "model.current_video_align_head.", "model.future_video_align_head.",
)

# Align query embeddings + shared task projections (host fp32; consumed by prefix.compute_align_tokens).
# `[256, 2560]` backbone embs -> mean-pooled to `[8, 2560]`; the shared projs are Linear(5120->2560).
ALIGN_KEYS = (
    "model.depth_align_embs",
    "model.future_depth_align_embs",
    "model.current_video_align_embs",
    "model.future_video_align_embs",
    "model.current_shared_task_proj.weight", "model.current_shared_task_proj.bias",
    "model.future_shared_task_proj.weight", "model.future_shared_task_proj.bias",
)

HEAD_KEYS = (
    "model.state_proj.weight", "model.state_proj.bias",
    "model.action_in_proj.weight", "model.action_in_proj.bias",
    "model.action_out_proj.weight", "model.action_out_proj.bias",
    "model.action_time_mlp_in.weight", "model.action_time_mlp_in.bias",
    "model.action_time_mlp_out.weight", "model.action_time_mlp_out.bias",
)


# =============================================================================
# CPU-only surface (no rpu_backend import reachable from here)
# =============================================================================
def ckpt_reader(ckpt_path: str):
    """Lazy CPU-fp32 getter over sharded BF16/FP32 safetensors.

    Returns (gw, key_to_reader). ``gw(key)`` always returns a CPU FP32 tensor;
    ``.float()`` normalizes the source checkpoint dtype before conversion.
    """
    files = sorted(glob.glob(os.path.join(ckpt_path, "*.safetensors")))
    if not files:
        raise ValueError(f"lingbot2: no .safetensors under {ckpt_path}")
    readers = [safe_open(f, framework="pt") for f in files]
    key_to_reader = {}
    for r in readers:
        for kk in r.keys():
            key_to_reader[kk] = r

    def gw(key):
        if key not in key_to_reader:
            raise KeyError(f"lingbot2: checkpoint has no tensor {key!r}")
        return key_to_reader[key].get_tensor(key).float()

    return gw, key_to_reader


def expert_slice(stacked: torch.Tensor, e: int) -> torch.Tensor:
    """THE stacked-expert contract: expert e's Linear.weight is `stacked[e]`.

    `stacked` is [E, out, in] (gate/up: [32,512,768]; down: [32,768,512]). This is the single
    place that defines the mapping used by the RPU builder and CPU-only consumers.
    """
    if stacked.dim() != 3:
        raise ValueError(f"expert_slice: expected a stacked 3-D [E,out,in] tensor, got {tuple(stacked.shape)}")
    if not (0 <= e < stacked.shape[0]):
        raise IndexError(f"expert_slice: expert {e} out of range for E={stacked.shape[0]}")
    return stacked[e]


def load_moe_layer_weights(ckpt_path_or_gw, layer: int, geom=None) -> dict[str, torch.Tensor]:
    """Read ONE layer's MoE weights (CPU fp32), shape-verified against the authoritative YAML.

    Accepts either a checkpoint dir path or an existing `gw` getter. Pure CPU — this is the
    function remains usable without an RPU. `geom` (from resolve_geometry) overrides the module
    defaults per checkpoint; geom=None ⇒ the kuavo_v2 defaults.
    """
    _g = geom if geom is not None else _default_geom()
    NUM_EXPERTS = _g["NUM_EXPERTS"]; EXP_HID = _g["EXP_HID"]
    ROUTED_INTER = _g["ROUTED_INTER"]; SHARED_INTER = _g["SHARED_INTER"]
    gw = ckpt_path_or_gw if callable(ckpt_path_or_gw) else ckpt_reader(ckpt_path_or_gw)[0]
    p = f"{EXP_P}layers.{layer}.mlp."
    t = {
        "gate":   gw(p + "gate.weight"),
        "bias":   gw(p + "e_score_correction_bias"),
        "e_gate": gw(p + "experts.gate_proj"),
        "e_up":   gw(p + "experts.up_proj"),
        "e_down": gw(p + "experts.down_proj"),
        "s_gate": gw(p + "shared_expert.gate_proj.weight"),
        "s_up":   gw(p + "shared_expert.up_proj.weight"),
        "s_down": gw(p + "shared_expert.down_proj.weight"),
    }
    expect = {
        "gate":   (NUM_EXPERTS, EXP_HID),
        "bias":   (NUM_EXPERTS,),
        "e_gate": (NUM_EXPERTS, ROUTED_INTER, EXP_HID),
        "e_up":   (NUM_EXPERTS, ROUTED_INTER, EXP_HID),
        "e_down": (NUM_EXPERTS, EXP_HID, ROUTED_INTER),
        "s_gate": (SHARED_INTER, EXP_HID),
        "s_up":   (SHARED_INTER, EXP_HID),
        "s_down": (EXP_HID, SHARED_INTER),
    }
    for k, want in expect.items():
        got = tuple(t[k].shape)
        if got != want:
            raise ValueError(f"lingbot2 layer {layer} {k}: shape {got} != YAML-derived {want}")
    return t


def load_router_correction_biases(ckpt_path_or_gw, geom=None) -> list[torch.Tensor]:
    """Preflight every router correction bias before any RPU upload.

    Nonzero values are supported: the device adds them to the FP16 sigmoid
    scores for strict top-k selection and gathers final weights from the
    unbiased scores.  This gate rejects malformed, non-finite, or FP16-
    unrepresentable values before a partially built device model can exist.
    """
    _g = geom if geom is not None else _default_geom()
    layers = int(_g["EXP_LAYERS"])
    experts = int(_g["NUM_EXPERTS"])
    gw = ckpt_path_or_gw if callable(ckpt_path_or_gw) else ckpt_reader(ckpt_path_or_gw)[0]
    result = []
    for layer in range(layers):
        key = f"{EXP_P}layers.{layer}.mlp.e_score_correction_bias"
        bias = gw(key)
        if bias.device.type != "cpu":
            raise ValueError(
                f"lingbot2 router correction bias layer {layer} must be preflighted on CPU"
            )
        if tuple(bias.shape) != (experts,):
            raise ValueError(
                f"lingbot2 router correction bias layer {layer}: shape "
                f"{tuple(bias.shape)} != ({experts},)"
            )
        if not bool(torch.isfinite(bias).all()):
            raise ValueError(
                f"lingbot2 router correction bias layer {layer} must be finite"
            )
        bias_fp16 = bias.detach().to(dtype=torch.float16)
        if not bool(torch.isfinite(bias_fp16).all()):
            raise ValueError(
                f"lingbot2 router correction bias layer {layer} is not representable in FP16"
            )
        if bool(((bias != 0) & (bias_fp16 == 0)).any()):
            raise ValueError(
                f"lingbot2 router correction bias layer {layer} contains a nonzero "
                "value that underflows to zero in FP16"
            )
        result.append(bias)
    return result


def load_host_weights(gw, geom=None) -> tuple[dict, dict]:
    """CPU-fp32 host weights: (head_W, norm_W).

    head_W  — suffix encoders + action head (host Euler loop; never uploaded).
    norm_W  — per-layer AdaRMS `.weight` + gamma/beta [768,768] linears (host fold).
    `geom` (resolve_geometry) sets the layer count per-checkpoint; None ⇒ kuavo_v2 default.
    """
    EXP_LAYERS = (geom if geom is not None else _default_geom())["EXP_LAYERS"]
    head_W = {k: gw(k) for k in HEAD_KEYS}
    norm_W = {}
    for L in range(EXP_LAYERS):
        for slot in ("input_layernorm", "post_attention_layernorm"):
            pf = f"{EXP_P}layers.{L}.{slot}"
            for suf in (".weight", ".gamma.weight", ".gamma.bias", ".beta.weight", ".beta.bias"):
                norm_W[pf + suf] = gw(pf + suf)
    return head_W, norm_W


def load_align_weights(gw) -> dict:
    """CPU-fp32 prefix align-query weights (embs + shared task projections).

    Only the tensors actually on the sample_actions path — the align HEADS are distillation-only
    and are deliberately not read (see UNUSED_CKPT_PREFIXES).
    """
    return {k: gw(k) for k in ALIGN_KEYS}


def resolve_geometry(gw=None, *, training_config_path=None, ckpt_dir=None) -> dict:
    """Resolve the action-expert geometry for THIS checkpoint (pure CPU: yaml + safetensors).

    Order:  hard-coded kuavo_v2 defaults
            -> overridden by `training_config_path` (the checkpoint's lingbotvla_cli.yaml),
               or a `lingbotvla_cli*.yaml` auto-found in `ckpt_dir`
            -> validated against the checkpoint's real tensor shapes (`gw`) and the as-built
               .so's fixed assumptions.
    Fails LOUD on any mismatch instead of building a silently-wrong graph. With no YAML it
    returns the defaults unchanged for the kuavo_v2 build.
    """
    geom = _default_geom()
    src = "hard-coded kuavo_v2 defaults"
    yaml_path = training_config_path
    if yaml_path is None and ckpt_dir is not None:
        cands = sorted(glob.glob(os.path.join(ckpt_dir, "lingbotvla_cli*.y*ml")))
        yaml_path = cands[0] if cands else None
    if yaml_path is not None:
        import yaml as _yaml
        with open(yaml_path) as _f:
            cfg = _yaml.safe_load(_f) or {}
        tr = dict(cfg.get("train", {}) or {})

        def _i(v):
            return None if v is None else int(v)

        overrides = {
            "EXP_HID":       _i(tr.get("expert_hidden_size")),
            "NUM_EXPERTS":   _i(tr.get("token_num_experts")),
            "TOP_K":         _i(tr.get("token_top_k")),
            "ROUTED_INTER":  _i(tr.get("token_moe_intermediate_size")),   # NOT expert_intermediate_size (dense)
            "SHARED_INTER":  _i(tr.get("token_shared_intermediate_size")),
            "ROUTED_SCALING": (None if tr.get("routed_scaling_factor") is None else float(tr["routed_scaling_factor"])),
            "ROUTER_ACTIVATION": tr.get("router_activation"),
            "N_ACTION":      _i(tr.get("chunk_size")),
            "ACTION_DIM":    _i(tr.get("max_action_dim")),
            "STATE_DIM":     _i(tr.get("max_state_dim")),
        }
        moe_layers = tr.get("token_moe_layers")
        if moe_layers:
            overrides["EXP_LAYERS"] = len(moe_layers)
        for k, v in overrides.items():
            if v is not None:
                geom[k] = v
        # 8-core col-swizzle needs N % 128 == 0; pad the shared-expert inter up (silu(0)*0 = 0 exact).
        geom["SHARED_INTER_PAD"] = ((geom["SHARED_INTER"] + 127) // 128) * 128
        src = os.fspath(yaml_path)

    # --- assumptions BAKED into the as-built .so (a mismatch needs a C++ change, not a config) ---
    if str(geom["ROUTER_ACTIVATION"]) != "sigmoid":
        raise ValueError(
            f"lingbot2 geometry (src={src}): router_activation={geom['ROUTER_ACTIVATION']!r}, but the as-built "
            f"rpu_backend .so hard-codes a SIGMOID router. A softmax router needs a C++ change; refusing to "
            f"build a silently-wrong graph.")
    if geom["ROUTED_INTER"] % geom["MLP_CORES"] != 0:
        raise ValueError(f"lingbot2 geometry (src={src}): routed_inter {geom['ROUTED_INTER']} is not divisible "
                         f"by MLP_CORES {geom['MLP_CORES']} (grouped-expert core slicing requires it).")
    if geom["SHARED_INTER_PAD"] % 128 != 0:
        raise ValueError(f"lingbot2 geometry (src={src}): shared_inter_pad {geom['SHARED_INTER_PAD']} is not a "
                         f"multiple of 128 (8-core col-swizzle requires N % 128 == 0).")

    # --- cross-check against the checkpoint's REAL tensor shapes (fail loud) ---
    if gw is not None:
        _validate_geom_against_ckpt(gw, geom, src)
    return geom


def _validate_geom_against_ckpt(gw, geom: dict, src: str) -> None:
    """Assert the resolved geometry actually describes the checkpoint's weights and the runtime consts."""
    E, I, H, S = geom["NUM_EXPERTS"], geom["ROUTED_INTER"], geom["EXP_HID"], geom["SHARED_INTER"]
    p0 = f"{EXP_P}layers.0.mlp."
    want = {
        "gate.weight":                    (E, H),
        "experts.gate_proj":              (E, I, H),
        "experts.up_proj":                (E, I, H),
        "experts.down_proj":              (E, H, I),
        "shared_expert.gate_proj.weight": (S, H),
        "shared_expert.down_proj.weight": (H, S),
    }
    for name, exp in want.items():
        got = tuple(gw(p0 + name).shape)
        if got != exp:
            raise ValueError(
                f"lingbot2 geometry (src={src}): layer-0 mlp.{name} checkpoint shape {got} != config-derived "
                f"{exp}. The training config does not describe these weights — refusing to build.")
    n = 0
    while True:
        try:
            gw(f"{EXP_P}layers.{n}.mlp.gate.weight")
        except KeyError:
            break
        n += 1
    if n != geom["EXP_LAYERS"]:
        raise ValueError(f"lingbot2 geometry (src={src}): EXP_LAYERS={geom['EXP_LAYERS']} but the checkpoint has "
                         f"{n} MoE layers.")
    # The host denoise loop (runtime.py) still uses its own module constants for these three dims;
    # they must equal the resolved geometry or get_action would replay a graph shaped for a different
    # checkpoint. Fail loud here rather than silently mis-run (full parameterization is a follow-up).
    for k in ("EXP_LAYERS", "EXP_HID", "N_ACTION"):
        if globals()[k] != geom[k]:
            raise ValueError(
                f"lingbot2 geometry (src={src}): {k}={geom[k]} differs from the built-in {k}={globals()[k]}. "
                f"The host denoise loop is not yet parameterized for this dimension — convert.py AND "
                f"runtime.py constants must be updated before this checkpoint can run.")


def head_pad_width(dim: int, multiple: int = 16) -> int:
    """Round an action/state dim up to `multiple` (the native-width fallback for odd dims)."""
    return ((dim + multiple - 1) // multiple) * multiple


def verify_head_partition() -> tuple[int, int]:
    """Check the action head's padded width is a LEGAL RPU linear shape BEFORE committing to it.

    action_dim 55 fails the native-width check, so an on-device head would pad 55 -> 64
    (roundup16). This asserts `get_linear_partition(768, 64, fp16) != -1`. Returns (pad, partition).

    NOTE: importing `rpu_backend.runtime.weights` pulls the rpu_backend package (and its .so), so
    this is NOT callable on an RPU-less host. The default
    runtime keeps the action head on the host in fp32, so 55->64
    is currently an unexercised contract kept here for a future on-device head.
    """
    from rpu_backend.runtime.weights import get_linear_partition
    pad = head_pad_width(ACTION_DIM)                 # 55 -> 64
    part = get_linear_partition(EXP_HID, pad, 2)     # in=768, out=64, dwidth=2 (fp16)
    if part == -1:
        raise ValueError(
            f"lingbot2: action head pad {ACTION_DIM}->{pad} is not a legal RPU linear "
            f"(get_linear_partition(768, {pad}, fp16) == -1). Pick another pad width."
        )
    return pad, part


# =============================================================================
# RPU-touching surface (rpu_backend imported lazily, inside the functions)
# =============================================================================
def _h(t: torch.Tensor) -> torch.Tensor:
    """fp32/any -> fp16 RPU contiguous; callers swizzle only after ``.half()``."""
    return t.detach().to(torch.float16).to("rpu").contiguous()


def _h_256b(t: torch.Tensor) -> torch.Tensor:
    """fp32/any -> retained contiguous fp16 RPU view with a 256-byte data pointer."""
    source = t.detach().to(torch.float16).contiguous()
    element_size = source.element_size()
    alignment = 256
    storage = torch.empty(
        source.numel() + alignment // element_size,
        dtype=source.dtype,
        device="rpu",
    )
    offset_bytes = (-storage.data_ptr()) % alignment
    if offset_bytes % element_size:
        raise RuntimeError(
            f"cannot form a {alignment}-byte aligned fp16 RPU view"
        )
    result = storage.narrow(
        0, offset_bytes // element_size, source.numel()
    ).view(source.shape)
    result.copy_(source)
    if not result.is_contiguous() or result.data_ptr() % alignment:
        raise RuntimeError("failed to retain a contiguous 256-byte aligned RPU view")
    return result


def _q(t: torch.Tensor) -> torch.Tensor:
    """dtype-PRESERVING RPU upload, for W8A16 int8 packed weights.

    _h() force-casts to fp16, which silently turns an int8 weight back into fp16 and makes the
    W8A16 path a no-op (the C++ then rejects the scales with "packed weights are fp16, not int8").
    """
    return t.detach().to("rpu").contiguous()


def _build_rope_tables(head_dim: int, theta: float, max_seq: int):
    """1-D RoPE cos/sin tables, fp16 RPU, shape [max_seq, head_dim/2]... (kernel format [max_seq, half*?]).

    ``theta`` is the VLM's rope_theta (5e6 for Qwen3-VL-4B), not a separately
    owned action-expert value: the VLM rotary embedding rotates the expert q/k.
    Plain 1-D is exact for the suffix because all three M-RoPE axes receive the
    same suffix positions, reducing interleaved M-RoPE sections [24,20,20] to
    ordinary 1-D RoPE.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    pos = torch.arange(max_seq, dtype=torch.float64)
    freqs = pos[:, None] * inv_freq[None, :]
    # partial_mrope encodes the table DDR addresses in 256-byte units.  Retain
    # aligned views instead of relying on the RPU allocator's weaker base
    # alignment; the Tensor view keeps its padded storage owner alive.
    return _h_256b(freqs.cos().float()), _h_256b(freqs.sin().float())


def build_expert_moe(
    gw, *, max_seq: int, rope_theta: float, geom=None, on_handle_created=None
):
    """Build the LingBot2 sparse-MoE action expert -> (handle, keep).

    The order is load-bearing: .half() -> (F.pad) -> swizzle -> .to('rpu').

    `keep` is the mandatory keepalive tuple: the C++ side holds raw pointers into these tensors
    (e.g. router_bias_[L].data_ptr<c10::Half>()), so dropping the Python refs frees DDR under a
    live handle. Store it on the policy and never let it go.

    `geom` (resolve_geometry) supplies the per-checkpoint geometry that is threaded into the C++
    set_weights call below; geom=None ⇒ the kuavo_v2 module defaults, so the build is unchanged.

    `on_handle_created`, when provided by the transactional runtime builder, is called
    immediately after native create and before any fallible setter. This publishes partial
    ownership so an outer construction rollback can destroy the handle.
    """
    from rpu_backend.runtime.weights import tp_col_swizzle_mc_weight, tp_row_swizzle_mc_weight

    # Per-checkpoint geometry → shadow the module defaults as locals (function body unchanged).
    _g = geom if geom is not None else _default_geom()
    EXP_LAYERS = _g["EXP_LAYERS"]; EXP_HID = _g["EXP_HID"]
    NQ = _g["NQ"]; NKV = _g["NKV"]; HD = _g["HD"]; RMS_EPS = _g["RMS_EPS"]
    NUM_EXPERTS = _g["NUM_EXPERTS"]; TOP_K = _g["TOP_K"]; ROUTED_INTER = _g["ROUTED_INTER"]
    SHARED_INTER = _g["SHARED_INTER"]; SHARED_INTER_PAD = _g["SHARED_INTER_PAD"]
    ROUTED_SCALING = _g["ROUTED_SCALING"]; N_ACTION = _g["N_ACTION"]
    ATTN_TP = _g["ATTN_TP"]; MLP_CORES = _g["MLP_CORES"]; ROUTER_CORES = _g["ROUTER_CORES"]

    # Complete CPU preflight before the first weight upload.  In particular,
    # nonzero correction bias is legal and must survive as a selection-only
    # input; NaN/Inf, FP16 overflow, or nonzero-to-zero underflow must fail
    # before device mutation.
    router_bias_cpu = load_router_correction_biases(gw, geom=_g)

    # RPU_LINGBOT2_BASE_W8A16=1 — int8 the action expert's attention q/k/v/o and the SHARED
    # expert gate/up/down. The router remains fp16 to preserve selection precision.
    # Default OFF => fp16, unchanged.
    BASE_W8A16 = os.environ.get("RPU_LINGBOT2_BASE_W8A16") == "1"
    if BASE_W8A16:
        from rpu_backend.quant._common import quantize_linear_per_channel
    q_ws, k_ws, v_ws, o_ws, shg_ws, shu_ws, shd_ws = [], [], [], [], [], [], []

    def col(w, cores):
        return _h(tp_col_swizzle_mc_weight(w.half(), cores))

    def row(w, cores):
        return _h(tp_row_swizzle_mc_weight(w.half(), cores))

    def col_q(w, cores, sink):
        """int8 col-swizzle + record the per-output-channel scale (dwidth=1 for int8)."""
        wq, sc = quantize_linear_per_channel(w.half())
        sink.append(_h(sc.half()))
        return _q(tp_col_swizzle_mc_weight(wq, cores, dwidth=1))

    def row_q(w, cores, sink):
        wq, sc = quantize_linear_per_channel(w.half())
        sink.append(_h(sc.half()))
        return _q(tp_row_swizzle_mc_weight(wq, cores, dwidth=1))

    q_w, k_w, v_w, o_w = [], [], [], []
    q_b, k_b, v_b = [], [], []
    in_norm, post_norm = [], []
    sh_gate, sh_up, sh_down = [], [], []
    router_gate, router_bias = [], []
    exp_gate, exp_up, exp_down = [], [], []

    npad = SHARED_INTER_PAD - SHARED_INTER            # 704 -> 768

    for L in range(EXP_LAYERS):
        p = f"{EXP_P}layers.{L}."
        # ---- attention: q/k/v are output-partitioned (col), o is input-partitioned (row) ----
        if BASE_W8A16:
            q_w.append(col_q(gw(p + "self_attn.q_proj.weight"), ATTN_TP, q_ws))
            k_w.append(col_q(gw(p + "self_attn.k_proj.weight"), ATTN_TP, k_ws))
            v_w.append(col_q(gw(p + "self_attn.v_proj.weight"), ATTN_TP, v_ws))
            o_w.append(row_q(gw(p + "self_attn.o_proj.weight"), ATTN_TP, o_ws))
        else:
            q_w.append(col(gw(p + "self_attn.q_proj.weight"), ATTN_TP))
            k_w.append(col(gw(p + "self_attn.k_proj.weight"), ATTN_TP))
            v_w.append(col(gw(p + "self_attn.v_proj.weight"), ATTN_TP))
            o_w.append(row(gw(p + "self_attn.o_proj.weight"), ATTN_TP))
        q_b.append(_h(gw(p + "self_attn.q_proj.bias")))   # bias: uploaded, NOT swizzled
        k_b.append(_h(gw(p + "self_attn.k_proj.bias")))
        v_b.append(_h(gw(p + "self_attn.v_proj.bias")))
        # ---- AdaRMS placeholders (real scale/shift ride in per Euler step) ----
        in_norm.append(_h(gw(p + "input_layernorm.weight")))
        post_norm.append(_h(gw(p + "post_attention_layernorm.weight")))
        # ---- shared expert: 704 -> 768 zero-pad, THEN swizzle (silu(0)*0 = 0 => exact) ----
        sg = F.pad(gw(p + "mlp.shared_expert.gate_proj.weight").half(), (0, 0, 0, npad))
        su = F.pad(gw(p + "mlp.shared_expert.up_proj.weight").half(), (0, 0, 0, npad))
        sd = F.pad(gw(p + "mlp.shared_expert.down_proj.weight").half(), (0, npad))
        if BASE_W8A16:
            sh_gate.append(col_q(sg, MLP_CORES, shg_ws))
            sh_up.append(col_q(su, MLP_CORES, shu_ws))
            sh_down.append(row_q(sd, MLP_CORES, shd_ws))
        else:
            sh_gate.append(_h(tp_col_swizzle_mc_weight(sg, MLP_CORES)))
            sh_up.append(_h(tp_col_swizzle_mc_weight(su, MLP_CORES)))
            sh_down.append(_h(tp_row_swizzle_mc_weight(sd, MLP_CORES)))
        # ---- router: fp16, SINGLE-core col-swizzle (see SWIZZLE_PLAN deviation note) ----
        router_gate.append(col(gw(p + "mlp.gate.weight"), ROUTER_CORES))
        router_bias.append(_h(router_bias_cpu[L]))
        # ---- routed experts: slice the stacked [E, out, in] Parameters ----
        eg = gw(p + "mlp.experts.gate_proj")
        eu = gw(p + "mlp.experts.up_proj")
        ed = gw(p + "mlp.experts.down_proj")
        for e in range(NUM_EXPERTS):
            # bank index is layer*E + e — the order these are appended IS the contract.
            exp_gate.append(col(expert_slice(eg, e), MLP_CORES))
            exp_up.append(col(expert_slice(eu, e), MLP_CORES))
            exp_down.append(row(expert_slice(ed, e), MLP_CORES))

    cos, sin = _build_rope_tables(HD, rope_theta, max_seq)
    final_norm_w = _h(gw(EXP_P + "norm.weight"))       # plain RMS (final_norm_adanorm: false)

    handle = torch.ops.rpu.lingbot_v2_moe_create()
    if on_handle_created is not None:
        on_handle_created(handle)
    torch.ops.rpu.lingbot_v2_moe_set_weights(
        handle,
        q_w, k_w, v_w, o_w,
        in_norm, post_norm,
        sh_gate, sh_up, sh_down,
        cos, sin, final_norm_w,
        NQ, NKV, HD,
        EXP_HID, SHARED_INTER_PAD, RMS_EPS,
        q_b, k_b, v_b,
        router_gate, router_bias,
        exp_gate, exp_up, exp_down,
        NUM_EXPERTS, TOP_K, ROUTED_INTER,
        ROUTED_SCALING, N_ACTION + 1,          # chunk_size = 51 suffix rows (state + 50 actions)
    )
    keep = (q_w, k_w, v_w, o_w, q_b, k_b, v_b, in_norm, post_norm,
            sh_gate, sh_up, sh_down, router_gate, router_bias,
            exp_gate, exp_up, exp_down, cos, sin, final_norm_w)
    if BASE_W8A16:
        torch.ops.rpu.lingbot_v2_moe_set_base_scales(handle, q_ws, k_ws, v_ws, o_ws,
                                                     shg_ws, shu_ws, shd_ws)
        keep = keep + (q_ws, k_ws, v_ws, o_ws, shg_ws, shu_ws, shd_ws)
    # ── GROUPED-EXPERTS opt-in: build core-slice-interleaved packed expert weights
    #    and bind them. Packed row/col order = c*(E*Ic) + e*Ic + s  (core c holds
    #    slice-c of EVERY expert), so the per-core packed GEMM computes all experts'
    #    slice-c at once and the routing weight w[seq,E] is identical on every core.
    if os.environ.get("RPU_LINGBOT2_GROUPED_EXPERTS") == "1":
        from rpu_backend.runtime.weights import tp_col_swizzle_mc_weight, tp_row_swizzle_mc_weight
        NC = MLP_CORES
        Ic = ROUTED_INTER // NC
        gate_packed, up_packed, down_packed = [], [], []
        gate_scale, up_scale, down_scale = [], [], []
        # RPU_LINGBOT2_EXPERT_W8A16=1 — int8 the ROUTED expert gate/up/down only.
        # RPU_LINGBOT2_EXPERT_W4A16=1 — int4 (nibble-packed) the same three GEMMs. W4 wins over W8;
        #   if both are set W4 takes precedence. Both reuse the acc16 launcher's native int8/int4
        #   dispatch (weight dtype selects the kernel) — no new quantisation machinery.
        W4A16 = os.environ.get("RPU_LINGBOT2_EXPERT_W4A16") == "1"
        W8A16 = (not W4A16) and os.environ.get("RPU_LINGBOT2_EXPERT_W8A16") == "1"
        if W8A16:
            from rpu_backend.quant._common import quantize_linear_per_channel
        if W4A16:
            from rpu_backend.quant.int4_pgrp_pack import (
                quantize_int4_group_wise,
                swizzle_int4_pgrp_scale,
                swizzle_pack_int4_pgrp,
            )
        for L in range(EXP_LAYERS):
            p = f"{EXP_P}layers.{L}."
            eg = gw(p + "mlp.experts.gate_proj")
            eu = gw(p + "mlp.experts.up_proj")
            ed = gw(p + "mlp.experts.down_proj")
            g_stack = torch.stack([expert_slice(eg, e).half() for e in range(NUM_EXPERTS)])  # [E,I,h]
            u_stack = torch.stack([expert_slice(eu, e).half() for e in range(NUM_EXPERTS)])  # [E,I,h]
            d_stack = torch.stack([expert_slice(ed, e).half() for e in range(NUM_EXPERTS)])  # [E,h,I]
            # gate/up: [E,I,h] -view-> [E,NC,Ic,h] -perm(1,0,2,3)-> [NC,E,Ic,h] -reshape-> [E*I,h]
            g_pk = g_stack.view(NUM_EXPERTS, NC, Ic, EXP_HID).permute(1, 0, 2, 3).reshape(NUM_EXPERTS * ROUTED_INTER, EXP_HID)
            u_pk = u_stack.view(NUM_EXPERTS, NC, Ic, EXP_HID).permute(1, 0, 2, 3).reshape(NUM_EXPERTS * ROUTED_INTER, EXP_HID)
            # down: [E,h,I] -view-> [E,h,NC,Ic] -perm(1,2,0,3)-> [h,NC,E,Ic] -reshape-> [h,E*I]
            d_pk = d_stack.view(NUM_EXPERTS, EXP_HID, NC, Ic).permute(1, 2, 0, 3).reshape(EXP_HID, NUM_EXPERTS * ROUTED_INTER)
            if W4A16:
                # Expert-only W4A16 pgrp: group-size 32 makes the uint8 weight + 2-D
                # fp16 scale dispatch to the generated autotile kernels. The pgrp pack
                # retains the same core-slice-interleaved logical matrices, but applies
                # the kernel's required deinterleave before nibble packing.
                gqv, gs, _ = quantize_int4_group_wise(g_pk, group_size=32)
                uqv, us, _ = quantize_int4_group_wise(u_pk, group_size=32)
                dqv, ds, _ = quantize_int4_group_wise(d_pk, group_size=32)
                gp = swizzle_pack_int4_pgrp(gqv, partition=1, num_cores=NC).view(NUM_EXPERTS * ROUTED_INTER, EXP_HID // 2)
                up = swizzle_pack_int4_pgrp(uqv, partition=1, num_cores=NC).view(NUM_EXPERTS * ROUTED_INTER, EXP_HID // 2)
                dp = swizzle_pack_int4_pgrp(dqv, partition=0, num_cores=NC).view(EXP_HID, NUM_EXPERTS * ROUTED_INTER // 2)
                gate_packed.append(gp.to("rpu").contiguous())
                up_packed.append(up.to("rpu").contiguous())
                down_packed.append(dp.to("rpu").contiguous())
                gate_scale.append(_h(swizzle_int4_pgrp_scale(
                    gs.half(), 32, partition=1, num_cores=NC)))
                up_scale.append(_h(swizzle_int4_pgrp_scale(
                    us.half(), 32, partition=1, num_cores=NC)))
                down_scale.append(_h(swizzle_int4_pgrp_scale(
                    ds.half(), 32, partition=0, num_cores=NC)))
            elif W8A16:
                # Expert-only W8A16. Reuses the shipped per-output-channel quantiser
                # (rpu_backend.quant._common) and the acc16 GEMM's native int8-weight +
                # fp16-scale path — no new quantisation machinery. int8 swizzles at
                # dwidth=1.
                # Only the ROUTED expert GEMMs are touched: router, attention, AdaRMS,
                # shared expert, residual and the action head all stay fp16.
                gq, gs = quantize_linear_per_channel(g_pk)
                uq, us = quantize_linear_per_channel(u_pk)
                dq, ds = quantize_linear_per_channel(d_pk)
                gate_packed.append(_q(tp_col_swizzle_mc_weight(gq, NC, dwidth=1)))
                up_packed.append(_q(tp_col_swizzle_mc_weight(uq, NC, dwidth=1)))
                down_packed.append(_q(tp_row_swizzle_mc_weight(dq, NC, dwidth=1)))
                gate_scale.append(_h(gs.half()))
                up_scale.append(_h(us.half()))
                down_scale.append(_h(ds.half()))
            else:
                gate_packed.append(_h(tp_col_swizzle_mc_weight(g_pk, NC)))
                up_packed.append(_h(tp_col_swizzle_mc_weight(u_pk, NC)))
                down_packed.append(_h(tp_row_swizzle_mc_weight(d_pk, NC)))
        torch.ops.rpu.lingbot_v2_moe_set_packed_weights(handle, gate_packed, up_packed, down_packed)
        keep = keep + (gate_packed, up_packed, down_packed)
        if W8A16 or W4A16:
            torch.ops.rpu.lingbot_v2_moe_set_packed_scales(handle, gate_scale, up_scale, down_scale)
            keep = keep + (gate_scale, up_scale, down_scale)
    return handle, keep
